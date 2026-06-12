import pandas as pd
import numpy as np
import os
import joblib
import mlflow
from  mlflow import sklearn
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)

FEATURES = [
    "hour", "dayofweek", "month", "is_weekend", "is_peak", "season",
    "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_24h", "pm25_lag_168h",
    "pm25_roll_6h", "pm25_roll_24h", "pm25_roll_72h", "pm25_std_24h",
    "no2_avg", "o3_avg", "pm10_avg"
]
TARGET  = "pm25_avg"
SEQ_LEN = 24

# ─────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────

def load_splits():
    train = pd.read_csv("data/processed/train.csv", parse_dates=["datetime"])
    val   = pd.read_csv("data/processed/val.csv",   parse_dates=["datetime"])
    test  = pd.read_csv("data/processed/test.csv",  parse_dates=["datetime"])
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


def evaluate(y_true, y_pred, label=""):
    rmse         = np.sqrt(mean_squared_error(y_true, y_pred))
    mae          = mean_absolute_error(y_true, y_pred)
    spike_mask   = y_true > 250
    spike_recall = 0.0
    if spike_mask.sum() > 0:
        spike_recall = (y_pred[spike_mask] > 250).sum() / spike_mask.sum() * 100
    print(f"  {label:<15} RMSE: {rmse:.2f}  MAE: {mae:.2f}  Spike recall: {spike_recall:.1f}%")
    return {"rmse": rmse, "mae": mae, "spike_recall": spike_recall}

# ─────────────────────────────────────────
# MODEL 1: XGBOOST
# ─────────────────────────────────────────

def train_xgboost(train, val, test):
    print("\n── XGBoost ─────────────────────────────")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val,   y_val   = val[FEATURES],   val[TARGET]
    X_test,  y_test  = test[FEATURES],  test[TARGET]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        early_stopping_rounds=20,
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    val_preds  = model.predict(X_val)
    test_preds = model.predict(X_test)

    print("\nValidation:")
    evaluate(y_val.values,  val_preds,  "XGBoost-val")
    print("Test:")
    test_metrics = evaluate(y_test.values, test_preds, "XGBoost-test")

    importance = pd.Series(
        model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    print(f"\nTop 5 features:\n{importance.head()}")

    joblib.dump(model, "models/xgboost_model.pkl")
    print("Saved → models/xgboost_model.pkl")

    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_params({
            "n_estimators": 500, "learning_rate": 0.05,
            "max_depth": 4, "min_child_weight": 5
        })
        mlflow.log_metrics(test_metrics)
        sklearn.log_model(model, name="xgboost")

    return model, test_preds

# ─────────────────────────────────────────
# MODEL 2: PROPHET
# ─────────────────────────────────────────

def train_prophet(train, val, test) -> tuple:
    print("\n── Prophet ─────────────────────────────")

    prophet_train = pd.concat([train, val])[["datetime", TARGET]].rename(
        columns={"datetime": "ds", TARGET: "y"}
    )

    model = Prophet(
        daily_seasonality="auto",
        weekly_seasonality="auto",
        yearly_seasonality="auto",
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10,
    )
    model.fit(prophet_train)

    future     = model.make_future_dataframe(periods=len(test), freq="h")
    forecast   = model.predict(future)
    test_preds = np.maximum(
        forecast["yhat"].to_numpy(dtype=np.float64)[-len(test):], 0.0
    )

    print("Test:")
    test_metrics = evaluate(test[TARGET].values, test_preds, "Prophet-test")

    joblib.dump(model, "models/prophet_model.pkl")
    print("Saved → models/prophet_model.pkl")

    with mlflow.start_run(run_name="prophet"):
        mlflow.log_params({
            "changepoint_prior_scale": 0.1,
            "seasonality_prior_scale": 10
        })
        mlflow.log_metrics(test_metrics)

    return model, test_preds

# ─────────────────────────────────────────
# MODEL 3: LSTM
# ─────────────────────────────────────────

class AQI_LSTM(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden, layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()


def train_lstm(train, val, test):
    print("\n── LSTM ─────────────────────────────────")

    # Normalization stats from train only — never val/test
    X_mean = train[FEATURES].values.astype(np.float32).mean(axis=0)
    X_std  = train[FEATURES].values.astype(np.float32).std(axis=0) + 1e-8

    def make_sequences(df):
        X = (df[FEATURES].values.astype(np.float32) - X_mean) / X_std
        y = df[TARGET].values.astype(np.float32)
        seqs, tgts = [], []
        for i in range(len(X) - SEQ_LEN):
            seqs.append(X[i : i + SEQ_LEN])
            tgts.append(y[i + SEQ_LEN])
        return torch.tensor(np.array(seqs)), torch.tensor(np.array(tgts))

    X_tr, y_tr = make_sequences(train)
    X_va, y_va = make_sequences(val)
    X_te, y_te = make_sequences(test)

    train_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=False)
    val_dl   = DataLoader(TensorDataset(X_va, y_va), batch_size=256, shuffle=False)
    test_dl  = DataLoader(TensorDataset(X_te, y_te), batch_size=256, shuffle=False)

    model     = AQI_LSTM(input_size=len(FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_val_loss, patience_counter = float("inf"), 0
    EPOCHS = 30

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for Xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_dl:
                val_loss += criterion(model(Xb), yb).item()

        train_loss /= len(train_dl)
        val_loss   /= len(val_dl)
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:02d}/{EPOCHS} | "
                  f"Train: {train_loss:.1f} | Val: {val_loss:.1f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "models/lstm_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best and evaluate
    model.load_state_dict(torch.load("models/lstm_best.pt", weights_only=True))
    model.eval()

    preds = []
    with torch.no_grad():
        for Xb, _ in test_dl:
            preds.extend(model(Xb).numpy())

    test_preds   = np.array(preds)
    test_metrics = evaluate(y_te.numpy(), test_preds, "LSTM-test")

    np.save("models/lstm_X_mean.npy", X_mean)
    np.save("models/lstm_X_std.npy",  X_std)
    print("Saved → models/lstm_best.pt")

    with mlflow.start_run(run_name="lstm"):
        mlflow.log_params({
            "hidden": 64, "layers": 2,
            "seq_len": SEQ_LEN, "epochs": EPOCHS
        })
        mlflow.log_metrics(test_metrics)

    return model, test_preds, X_mean, X_std

def ensemble_predict(xgb_preds, prophet_preds, lstm_preds, hours, horizons=None):
    if horizons is None:
        horizons = list(range(1, len(hours) + 1))

    preds = []
    for i, hour in enumerate(hours):
        xgb_p    = xgb_preds[i]   if i < len(xgb_preds)     else xgb_preds[-1]
        pro_p    = prophet_preds[i] if i < len(prophet_preds) else prophet_preds[-1]
        lst_p    = lstm_preds[i]   if i < len(lstm_preds)     else lstm_preds[-1]
        horizon  = horizons[i]     if i < len(horizons)       else i + 1

        # XGBoost dominates — best RMSE and spike recall
        # Prophet gets slightly more weight at late night (seasonal signal)
        # LSTM kept at 10% — poor spikes but adds sequence smoothing
        if 7 <= hour <= 10 or 17 <= hour <= 21:
            w = [0.80, 0.10, 0.10]   # peak hours — lag-driven spikes
        elif 0 <= hour <= 5:
            w = [0.70, 0.20, 0.10]   # late night — slight seasonal boost
        else:
            w = [0.75, 0.15, 0.10]   # default

        preds.append(w[0]*xgb_p + w[1]*pro_p + w[2]*lst_p)

    return np.array(preds)

# ─────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────

def train_all():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("aqi_forecasting")

    train, val, test = load_splits()

    xgb_model,     xgb_preds               = train_xgboost(train, val, test)
    prophet_model, prophet_preds            = train_prophet(train, val, test)
    lstm_model,    lstm_preds, X_mean, X_std = train_lstm(train, val, test)

    # Align lengths — LSTM loses first SEQ_LEN rows
    min_len       = min(len(xgb_preds), len(prophet_preds), len(lstm_preds))
    xgb_preds     = xgb_preds[-min_len:]
    prophet_preds = prophet_preds[-min_len:]
    test_hours    = test["hour"].values[-min_len:]
    y_test        = test[TARGET].values[-min_len:]

    print("\n── Ensemble ────────────────────────────")
    ensemble_preds = ensemble_predict(xgb_preds, prophet_preds, lstm_preds, test_hours)
    evaluate(y_test, ensemble_preds, "Ensemble-test")

    print("\n── Final Model Comparison ──────────────")
    evaluate(y_test, xgb_preds,      "XGBoost")
    evaluate(y_test, prophet_preds,  "Prophet")
    evaluate(y_test, lstm_preds,     "LSTM")
    evaluate(y_test, ensemble_preds, "Ensemble")

    print("\nAll models trained and saved to models/")
    return xgb_model, prophet_model, lstm_model

if __name__ == "__main__":
    train_all()
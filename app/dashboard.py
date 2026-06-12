import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.ingest import update_delhi_aqi
from src.preprocess import preprocess, calculate_aqi, get_aqi_category, get_health_advice, AQI_CATEGORIES
from src.model import ensemble_predict, AQI_LSTM, FEATURES, load_splits, TARGET, SEQ_LEN
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="Delhi AQI Forecast",
    page_icon="",
    layout="wide"
)

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────

SEASON_MAP = {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2,9:2, 10:3,11:3}

# ─────────────────────────────────────────
# LOAD MODELS — cached so they load once
# ─────────────────────────────────────────

@st.cache_resource
def load_models():
    xgb     = joblib.load("models/xgboost_model.pkl")
    prophet = joblib.load("models/prophet_model.pkl")

    lstm = AQI_LSTM(input_size=len(FEATURES))
    lstm.load_state_dict(torch.load(
        "models/lstm_best.pt",
        map_location="cpu",
        weights_only=True
    ))
    lstm.eval()

    X_mean = np.load("models/lstm_X_mean.npy")
    X_std  = np.load("models/lstm_X_std.npy")

    return xgb, prophet, lstm, X_mean, X_std

@st.cache_data
def load_data(refresh_id: int = 0) -> pd.DataFrame:
    return pd.read_csv(
        "data/processed/delhi_aqi_processed.csv",
        parse_dates=["datetime"]
    )

def refresh_latest_data() -> bool:
    city_df = update_delhi_aqi()
    if city_df is None:
        return False
    preprocess()
    return True

def compute_regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    spike_mask = y_true > 250
    spike_recall = 0.0
    if spike_mask.sum() > 0:
        spike_recall = (y_pred[spike_mask] > 250).sum() / spike_mask.sum() * 100
    return {
        "rmse": rmse,
        "mae": mae,
        "spike_recall": spike_recall,
    }

@st.cache_data
def compute_model_metrics(_xgb_model, _prophet_model, _lstm_model, X_mean, X_std):
    train, val, test = load_splits()

    xgb_preds = _xgb_model.predict(test[FEATURES])
    xgb_metrics = compute_regression_metrics(test[TARGET].values, xgb_preds)

    prophet_input = test[["datetime"]].rename(columns={"datetime": "ds"})
    prophet_preds = np.maximum(
        _prophet_model.predict(prophet_input)["yhat"].to_numpy(dtype=np.float32),
        0.0
    )
    prophet_metrics = compute_regression_metrics(test[TARGET].values, prophet_preds)

    X = (test[FEATURES].values.astype(np.float32) - X_mean) / X_std
    y = test[TARGET].values.astype(np.float32)
    seqs, tgts = [], []
    for i in range(len(X) - SEQ_LEN):
        seqs.append(X[i : i + SEQ_LEN])
        tgts.append(y[i + SEQ_LEN])
    if len(seqs) == 0:
        raise ValueError("Not enough test data for LSTM sequence evaluation.")
    X_te = torch.tensor(np.array(seqs), dtype=torch.float32)
    with torch.no_grad():
        lstm_preds = _lstm_model(X_te).numpy()
    lstm_metrics = compute_regression_metrics(np.array(tgts), lstm_preds)

    min_len = min(len(xgb_preds), len(prophet_preds), len(lstm_preds))
    xgb_align = xgb_preds[-min_len:]
    prophet_align = prophet_preds[-min_len:]
    lstm_align = lstm_preds[-min_len:]
    y_align = test[TARGET].values[-min_len:]
    ensemble_preds = ensemble_predict(
        xgb_align, prophet_align, lstm_align,
        test["hour"].values[-min_len:]
    )
    ensemble_metrics = compute_regression_metrics(y_align, ensemble_preds)

    return pd.DataFrame([
        {
            "Model": "XGBoost",
            "RMSE": xgb_metrics["rmse"],
            "MAE": xgb_metrics["mae"],
            "Spike Recall": f"{xgb_metrics['spike_recall']:.1f}%",
        },
        {
            "Model": "Prophet",
            "RMSE": prophet_metrics["rmse"],
            "MAE": prophet_metrics["mae"],
            "Spike Recall": f"{prophet_metrics['spike_recall']:.1f}%",
        },
        {
            "Model": "LSTM",
            "RMSE": lstm_metrics["rmse"],
            "MAE": lstm_metrics["mae"],
            "Spike Recall": f"{lstm_metrics['spike_recall']:.1f}%",
        },
        {
            "Model": "Ensemble",
            "RMSE": ensemble_metrics["rmse"],
            "MAE": ensemble_metrics["mae"],
            "Spike Recall": f"{ensemble_metrics['spike_recall']:.1f}%",
        },
    ])

# ─────────────────────────────────────────
# ROLLING 24-HOUR FORECAST
# Each predicted hour feeds into the next
# as a lag feature — true recursive forecast
# ─────────────────────────────────────────
def predict_next_24h(df, xgb_model, prophet_model, lstm_model, X_mean, X_std):
    last_time    = df["datetime"].iloc[-1]
    last_row     = df.iloc[-1]
    history      = df["pm25_avg"].values.tolist()
    hourly_avg   = df.groupby(df["datetime"].dt.hour)["pm25_avg"].mean().to_dict()
    recent_trend = np.mean(np.diff(history[-4:])) if len(history) >= 4 else 0.0

    # Short-term fallback for first 3 hours (trend + diurnal pattern)
    def short_term_pm25(future_hour, horizon):
        baseline     = history[-1] + recent_trend * horizon
        current_mean = hourly_avg.get(last_time.hour, history[-1])
        future_mean  = hourly_avg.get(future_hour,    history[-1])
        multiplier   = future_mean / current_mean if current_mean > 0 else 1.0
        return max(0.0, baseline * multiplier)

    # Prophet — predict all 24 hours at once
    future_times   = [last_time + timedelta(hours=h) for h in range(1, 25)]
    prophet_preds  = np.maximum(
        prophet_model.predict(pd.DataFrame({"ds": future_times}))["yhat"].to_numpy(dtype=np.float32),
        0.0
    )

    # LSTM — rolling sequence window (last 24 feature rows)
    lstm_features = df[FEATURES].tail(24).values.astype(np.float32)
    if len(lstm_features) < 24:
        pad = np.tile(lstm_features[0], (24 - len(lstm_features), 1))
        lstm_features = np.vstack([pad, lstm_features])

    forecasts = []

    for h, future_time in enumerate(future_times, start=1):
        hour  = future_time.hour
        dow   = future_time.dayofweek
        month = future_time.month

        # Lag features — use predicted history for steps beyond real data
        pm25_lag_1h   = history[-1]
        pm25_lag_3h   = history[-3]   if len(history) >= 3   else history[0]
        pm25_lag_24h  = history[-24]  if len(history) >= 24  else history[0]
        pm25_lag_168h = history[-168] if len(history) >= 168 else history[0]

        roll_6h  = float(np.mean(history[-6:]))
        roll_24h = float(np.mean(history[-24:]))
        roll_72h = float(np.mean(history[-72:]))
        std_24h  = float(np.std(history[-24:]))

        row = {
            "hour":          hour,
            "dayofweek":     dow,
            "month":         month,
            "is_weekend":    int(dow >= 5),
            "is_peak":       int(hour in [7,8,9,17,18,19,20]),
            "season":        SEASON_MAP[month],
            "pm25_lag_1h":   pm25_lag_1h,
            "pm25_lag_3h":   pm25_lag_3h,
            "pm25_lag_24h":  pm25_lag_24h,
            "pm25_lag_168h": pm25_lag_168h,
            "pm25_roll_6h":  roll_6h,
            "pm25_roll_24h": roll_24h,
            "pm25_roll_72h": roll_72h,
            "pm25_std_24h":  std_24h,
            "no2_avg":       float(last_row["no2_avg"]),
            "o3_avg":        float(last_row["o3_avg"]),
            "pm10_avg":      float(last_row["pm10_avg"]),
        }

        # XGBoost prediction
        xgb_pred = max(0.0, float(xgb_model.predict(pd.DataFrame([row]))[0]))

        # LSTM prediction
        lstm_input = (lstm_features - X_mean) / X_std
        with torch.no_grad():
            lstm_raw  = lstm_model(torch.tensor(lstm_input[None], dtype=torch.float32))
        lstm_pred = max(0.0, float(lstm_raw.reshape(-1)[0]))

        # Prophet prediction
        prophet_pred = float(prophet_preds[h - 1])

        # Ensemble
        if h <= 3:
            ensemble_pred = short_term_pm25(hour, h)
        else:
            ensemble_pred = float(ensemble_predict(
                np.array([xgb_pred],     dtype=np.float32),
                np.array([prophet_pred], dtype=np.float32),
                np.array([lstm_pred],    dtype=np.float32),
                [hour],
                horizons=[h]
            )[0])
        ensemble_pred = max(0.0, ensemble_pred)

        # ── Append to history BEFORE building next feature row
        history.append(ensemble_pred)

        # ── Update LSTM feature window with latest lags
        next_row = np.array([
            hour, dow, month,
            int(dow >= 5),
            int(hour in [7,8,9,17,18,19,20]),
            SEASON_MAP[month],
            history[-1],                                          # lag_1h = just predicted
            history[-3]   if len(history) >= 3   else history[0],
            history[-24]  if len(history) >= 24  else history[0],
            history[-168] if len(history) >= 168 else history[0],
            float(np.mean(history[-6:])),
            float(np.mean(history[-24:])),
            float(np.mean(history[-72:])),
            float(np.std(history[-24:])),
            float(last_row["no2_avg"]),
            float(last_row["o3_avg"]),
            float(last_row["pm10_avg"]),
        ], dtype=np.float32)
        lstm_features = np.vstack([lstm_features[1:], next_row])

        aqi_val    = calculate_aqi(
            pm25=ensemble_pred,
            pm10=row["pm10_avg"],
            no2=row["no2_avg"],
            o3=row["o3_avg"]
        )
        cat, color = get_aqi_category(aqi_val)

        forecasts.append({
            "datetime": future_time,
            "pm25":     round(ensemble_pred, 1),
            "aqi":      aqi_val,
            "category": cat,
            "color":    color,
            "hour":     hour,
        })

    return pd.DataFrame(forecasts)

# ─────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────

def plot_forecast(forecast_df, anchor_time=None, anchor_aqi=None):
    fig = go.Figure()

    bands = [
        (0,   50,  "rgba(0,228,0,0.1)",    "Good"),
        (51,  100, "rgba(146,208,80,0.1)",  "Satisfactory"),
        (101, 200, "rgba(255,255,0,0.1)",   "Moderate"),
        (201, 300, "rgba(255,126,0,0.1)",   "Poor"),
        (301, 400, "rgba(255,0,0,0.1)",     "Very Poor"),
        (401, 500, "rgba(126,0,35,0.1)",    "Severe"),
    ]
    for lo, hi, color, name in bands:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color, line_width=0,
                      annotation_text=name, annotation_position="right",
                      annotation_font_size=10)

    if anchor_time is not None and anchor_aqi is not None:
        fig.add_trace(go.Scatter(
            x=[anchor_time, forecast_df["datetime"].iloc[0]],
            y=[anchor_aqi,  forecast_df["aqi"].iloc[0]],
            mode="lines+markers", name="Current",
            line=dict(color="#888780", width=1, dash="dash"),
            marker=dict(color="#888780", size=6)
        ))

    fig.add_trace(go.Scatter(
        x=forecast_df["datetime"],
        y=forecast_df["aqi"],
        mode="lines+markers",
        name="Forecast AQI",
        line=dict(color="#378ADD", width=2.5),
        marker=dict(color=forecast_df["color"], size=8,
                    line=dict(color="white", width=1)),
        hovertemplate="<b>%{x|%H:%M}</b><br>AQI: %{y}<extra></extra>"
    ))

    fig.update_layout(
        title="24-hour AQI forecast — Delhi",
        xaxis_title="Time", yaxis_title="AQI",
        yaxis=dict(range=[0, 520]),
        height=400, hovermode="x unified",
        showlegend=False, margin=dict(r=100)
    )
    return fig


def plot_historical(df, days=30):
    fig = px.line(
        df.tail(days * 24), x="datetime", y="aqi",
        title=f"Historical AQI — last {days} days",
        labels={"aqi": "AQI", "datetime": "Date"}
    )
    fig.add_hrect(y0=300, y1=520, fillcolor="rgba(255,0,0,0.1)",
                  line_width=0, annotation_text="Hazardous zone")
    fig.update_traces(line_color="#D85A30")
    fig.update_layout(height=300)
    return fig


def plot_pollution_breakdown(df):
    last_48 = df.tail(48)
    fig     = go.Figure()
    for col, color, name in [
        ("pm25_avg", "#E24B4A", "PM2.5"),
        ("pm10_avg", "#EF9F27", "PM10"),
        ("no2_avg",  "#7F77DD", "NO2"),
        ("o3_avg",   "#1D9E75", "O3"),
    ]:
        fig.add_trace(go.Scatter(
            x=last_48["datetime"], y=last_48[col],
            name=name, line=dict(color=color)
        ))
    fig.update_layout(
        title="Pollutant levels — last 48 hours",
        height=300, xaxis_title="Time", yaxis_title="µg/m³"
    )
    return fig

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    st.title("Delhi AQI Forecasting System")
    st.caption("Real-time air quality forecast using XGBoost + Prophet + LSTM ensemble")

    if "refresh_counter" not in st.session_state:
        st.session_state.refresh_counter = 0

    # ── Sidebar ──
    st.sidebar.markdown("### Live data refresh")
    if st.sidebar.button("Fetch latest API data"):
        with st.spinner("Fetching and reprocessing..."):
            success = refresh_latest_data()
            if success:
                st.session_state.refresh_counter += 1
                st.cache_data.clear()
                st.success("Data updated successfully.")
            else:
                st.warning("No new data fetched or update failed.")

    # Load
    with st.spinner("Loading models and data..."):
        xgb, prophet, lstm, X_mean, X_std = load_models()
        df = load_data(st.session_state.refresh_counter)
        model_metrics = compute_model_metrics(xgb, prophet, lstm, X_mean, X_std)

    # Data freshness — outside cached function
    last_time = df["datetime"].iloc[-1]
    hours_ago = (pd.Timestamp.now() - last_time).total_seconds() / 3600
    st.sidebar.markdown("### Data freshness")
    st.sidebar.info(
        f"Last reading: {last_time.strftime('%b %d, %H:%M')}\n\n"
        f"({hours_ago:.0f} hours ago)"
    )

    # ── Current conditions ──
    latest       = df.iloc[-1]
    current_aqi  = calculate_aqi(
        pm25=latest["pm25_avg"], pm10=latest["pm10_avg"],
        no2=latest["no2_avg"],   o3=latest["o3_avg"]
    )
    cat, color = get_aqi_category(current_aqi)
    advice     = get_health_advice(cat)

    st.markdown("---")
    st.subheader("Current Conditions")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AQI",   f"{current_aqi:.0f}")
    col2.metric("PM2.5", f"{latest['pm25_avg']:.1f} µg/m³")
    col3.metric("PM10",  f"{latest['pm10_avg']:.1f} µg/m³")
    col4.metric("NO2",   f"{latest['no2_avg']:.1f} µg/m³")

    st.markdown(
        f"<div style='background:{color};padding:12px 20px;border-radius:8px;"
        f"display:inline-block;"
        f"color:{'black' if cat in ['Good','Satisfactory','Moderate'] else 'white'};"
        f"font-weight:500;font-size:18px'>{cat}</div>",
        unsafe_allow_html=True
    )
    st.info(advice)
    st.markdown("---")

    # ── 24-hour forecast ──
    st.subheader("24-Hour Forecast")
    with st.spinner("Generating forecast..."):
        forecast_df = predict_next_24h(df, xgb, prophet, lstm, X_mean, X_std)

    st.plotly_chart(
        plot_forecast(forecast_df, anchor_time=last_time, anchor_aqi=current_aqi),
        use_container_width=True
    )

    with st.expander("View hourly forecast table"):
        display_df = forecast_df[["datetime","pm25","aqi","category"]].copy()
        display_df["datetime"] = display_df["datetime"].dt.strftime("%H:%M")
        display_df.columns     = ["Time", "PM2.5 (µg/m³)", "AQI", "Category"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Alert ──
    max_aqi      = forecast_df["aqi"].max()
    max_cat, _   = get_aqi_category(max_aqi)
    max_time = pd.Timestamp(str(forecast_df.loc[forecast_df["aqi"].idxmax(), "datetime"])).strftime("%H:%M")
    if max_aqi > 300:
        st.error(f"ALERT: AQI forecast to reach {max_aqi:.0f} ({max_cat}) at {max_time}. Avoid outdoor activities.")
    elif max_aqi > 200:
        st.warning(f"NOTE: AQI forecast to reach {max_aqi:.0f} ({max_cat}). Sensitive groups take precautions.")

    st.markdown("---")

    # ── Historical ──
    st.subheader("Historical Trends")
    col1, col2 = st.columns([3, 1])
    with col2:
        days = st.selectbox("Show last", [7, 14, 30, 60], index=2)
    with col1:
        st.plotly_chart(plot_historical(df, days), use_container_width=True)
    st.plotly_chart(plot_pollution_breakdown(df), use_container_width=True)

    st.markdown("---")

    # ── AQI distribution ──
    st.subheader("AQI Distribution — Full Dataset")
    col1, col2 = st.columns(2)

    with col1:
        cat_counts         = df["aqi_category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Hours"]
        st.plotly_chart(px.pie(
            cat_counts, names="Category", values="Hours",
            color="Category",
            color_discrete_map={k: v[2] for k, v in AQI_CATEGORIES.items()},
            title="AQI category breakdown"
        ), use_container_width=True)

    with col2:
        monthly_avg         = df.groupby(df["datetime"].dt.month)["aqi"].mean().reset_index()
        monthly_avg.columns = ["Month", "Avg AQI"]
        monthly_avg["Month"] = monthly_avg["Month"].map({
            1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
            7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"
        })
        st.plotly_chart(px.bar(
            monthly_avg, x="Month", y="Avg AQI",
            title="Average AQI by month",
            color="Avg AQI", color_continuous_scale="RdYlGn_r"
        ), use_container_width=True)

    # ── Model performance ──
    st.markdown("---")
    st.subheader("Model Performance")
    st.dataframe(model_metrics, use_container_width=True, hide_index=True)
    st.caption(
        "Performance is computed on the current test split using the loaded models. "
        "XGBoost should currently be the strongest predictor in this dataset."
    )

    st.markdown("---")
    st.caption(
        "Data: OpenAQ v3 API · CPCB Delhi stations · "
        "Trained 2016–2018 · Validated 2025 · Tested 2026"
    )

if __name__ == "__main__":
    main()
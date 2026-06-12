import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────
# POLLUTANT BREAKPOINTS — India NAQI standard
# Single source of truth — imported by dashboard.py
# ─────────────────────────────────────────
POLLUTANT_BREAKPOINTS = {
    "pm25": [
        (0,   30,   0,  50), (30,  60,  51, 100),
        (60,  90, 101, 200), (90, 120, 201, 300),
        (120, 250, 301, 400), (250, 350, 401, 450),
        (350, 500, 451, 500)
    ],
    "pm10": [
        (0,   50,   0,  50), (50,  100,  51, 100),
        (100, 250, 101, 200), (250, 350, 201, 300),
        (350, 430, 301, 400), (430, 500, 401, 500)
    ],
    "no2": [
        (0,  40,   0,  50), (40,  80,  51, 100),
        (80, 180, 101, 200), (180, 280, 201, 300),
        (280, 400, 301, 400), (400, 440, 401, 500)
    ],
    "o3": [
        (0,   50,   0,  50), (50,  100,  51, 100),
        (100, 168, 101, 200), (168, 208, 201, 300),
        (208, 748, 301, 500)
    ]
}

AQI_CATEGORIES = {
    "Good":         (0,   50,  "#00e400"),
    "Satisfactory": (51,  100, "#92d050"),
    "Moderate":     (101, 200, "#ffff00"),
    "Poor":         (201, 300, "#ff7e00"),
    "Very Poor":    (301, 400, "#ff0000"),
    "Severe":       (401, 500, "#7e0023"),
}

# ─────────────────────────────────────────
# AQI HELPERS — single source of truth
# ─────────────────────────────────────────

def _subindex(value, breakpoints):
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= value <= bp_hi:
            return ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (value - bp_lo) + aqi_lo
    return None


def calculate_aqi(pm25=None, pm10=None, no2=None, o3=None):
    """Calculate India NAQI from pollutant values. Returns max subindex."""
    values     = {"pm25": pm25, "pm10": pm10, "no2": no2, "o3": o3}
    subindices = []
    for pollutant, value in values.items():
        if value is None or pd.isna(value) or value < 0:
            continue
        idx = _subindex(value, POLLUTANT_BREAKPOINTS[pollutant])
        if idx is not None:
            subindices.append(idx)
    return round(max(subindices), 1) if subindices else np.nan


def aqi_category(aqi):
    if pd.isna(aqi): return "Unknown"
    if aqi <= 50:    return "Good"
    if aqi <= 100:   return "Satisfactory"
    if aqi <= 200:   return "Moderate"
    if aqi <= 300:   return "Poor"
    if aqi <= 400:   return "Very Poor"
    return "Severe"


def get_aqi_category(aqi):
    """Returns (category_name, hex_color) for a given AQI value."""
    for cat, (lo, hi, color) in AQI_CATEGORIES.items():
        if lo <= aqi <= hi:
            return cat, color
    return "Severe", "#7e0023"


def get_health_advice(category):
    advice = {
        "Good":         "Air quality is good. Enjoy outdoor activities!",
        "Satisfactory": "Acceptable air quality. Sensitive people should limit prolonged outdoor exertion.",
        "Moderate":     "Sensitive groups should reduce outdoor exertion. Others are fine.",
        "Poor":         "Everyone should limit prolonged outdoor exertion. Sensitive groups stay indoors.",
        "Very Poor":    "Everyone should avoid outdoor activities. Keep windows closed.",
        "Severe":       "Emergency conditions. Stay indoors. Avoid all outdoor activity.",
    }
    return advice.get(category, "")

# ─────────────────────────────────────────
# STEP 1: Load & parse datetime
# ─────────────────────────────────────────

def load_data(path="data/raw/delhi_aqi_city_avg.csv"):
    df = pd.read_csv(path)

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["datetime"] = df["datetime"].dt.tz_convert("Asia/Kolkata")
    df["datetime"] = df["datetime"].dt.tz_localize(None)

    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['datetime'].min()} → {df['datetime'].max()}")
    return df

# ─────────────────────────────────────────
# STEP 2: Resample to hourly
# ─────────────────────────────────────────

def resample_hourly(df):
    df = df.set_index("datetime").resample("1h").mean().reset_index()
    print(f"After hourly resample: {len(df)} rows")
    return df

# ─────────────────────────────────────────
# STEP 3: Handle missing values
# ─────────────────────────────────────────

def handle_missing(df):
    print("\nMissing before cleaning:")
    print(df.isnull().sum())

    # PM2.5 and NO2 — small gaps
    for col in ["pm25_avg", "no2_avg"]:
        df[col] = df[col].ffill(limit=3)
        df[col] = df[col].bfill(limit=3)

    # O3 and PM10 — heavily missing, fill with hourly median
    for col in ["o3_avg", "pm10_avg"]:
        hourly_median = df.groupby(df["datetime"].dt.hour)[col].transform("median")
        df[col]       = df[col].fillna(hourly_median)

    # Any remaining nulls → column median
    for col in ["pm25_avg", "no2_avg", "o3_avg", "pm10_avg"]:
        df[col] = df[col].fillna(df[col].median())

    print("\nMissing after cleaning:")
    print(df.isnull().sum())
    return df

# ─────────────────────────────────────────
# STEP 4: Remove outliers
# ─────────────────────────────────────────

def remove_outliers(df):
    bounds = {
        "pm25_avg": (0, 1000),
        "pm10_avg": (0, 1500),
        "no2_avg":  (0, 500),
        "o3_avg":   (0, 300),
    }
    for col, (low, high) in bounds.items():
        bad = ((df[col] < low) | (df[col] > high)).sum()
        if bad > 0:
            print(f"  Clipping {bad} outliers in {col}")
        df[col] = df[col].clip(lower=low, upper=high)
    return df

# ─────────────────────────────────────────
# STEP 5: Feature engineering
# ─────────────────────────────────────────

def engineer_features(df):
    # Time features
    df["hour"]       = df["datetime"].dt.hour
    df["dayofweek"]  = df["datetime"].dt.dayofweek
    df["month"]      = df["datetime"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_peak"]    = df["hour"].isin([7,8,9,17,18,19,20]).astype(int)

    # Indian seasons
    season_map = {
        12:0, 1:0, 2:0,    # winter
        3:1,  4:1, 5:1,    # spring
        6:2,  7:2, 8:2, 9:2,  # monsoon
        10:3, 11:3         # autumn
    }
    df["season"] = df["month"].map(season_map)

    # Lag features
    df["pm25_lag_1h"]   = df["pm25_avg"].shift(1)
    df["pm25_lag_3h"]   = df["pm25_avg"].shift(3)
    df["pm25_lag_24h"]  = df["pm25_avg"].shift(24)
    df["pm25_lag_168h"] = df["pm25_avg"].shift(168)

    # Rolling features
    df["pm25_roll_6h"]  = df["pm25_avg"].rolling(6,  min_periods=1).mean()
    df["pm25_roll_24h"] = df["pm25_avg"].rolling(24, min_periods=1).mean()
    df["pm25_roll_72h"] = df["pm25_avg"].rolling(72, min_periods=1).mean()
    df["pm25_std_24h"]  = df["pm25_avg"].rolling(24, min_periods=1).std()

    # AQI — multi-pollutant
    df["aqi"] = df.apply(
        lambda row: calculate_aqi(
            pm25=row["pm25_avg"], pm10=row["pm10_avg"],
            no2=row["no2_avg"],   o3=row["o3_avg"]
        ), axis=1
    )
    df["aqi_category"] = df["aqi"].apply(aqi_category)
    df["is_hazardous"] = (df["aqi"] > 300).astype(int)

    # Drop first 168 rows where lag features are NaN
    df = df.dropna(subset=["pm25_lag_168h"]).reset_index(drop=True)

    print(f"\nAfter feature engineering: {len(df)} rows, {len(df.columns)} columns")
    return df

# ─────────────────────────────────────────
# STEP 6: Train / val / test split
# TIME-BASED — never shuffle time series
# ─────────────────────────────────────────

def split_data(df):
    train = df[df["datetime"].dt.year.isin([2016, 2017, 2018])]
    val   = df[df["datetime"].dt.year == 2025]
    test  = df[df["datetime"].dt.year == 2026]

    print(f"\nData splits:")
    print(f"  Train : {len(train)} rows | {train['datetime'].min().date()} → {train['datetime'].max().date()}")
    print(f"  Val   : {len(val)}   rows | {val['datetime'].min().date()} → {val['datetime'].max().date()}")
    print(f"  Test  : {len(test)}  rows | {test['datetime'].min().date()} → {test['datetime'].max().date()}")

    if len(val) == 0 or len(test) == 0:
        print("  WARNING: val or test is empty — check your data years")

    return train, val, test

# ─────────────────────────────────────────
# STEP 7: Quality report
# ─────────────────────────────────────────

def quality_report(df):
    print("\n── Final Quality Report ────────────────")
    print(f"  Total rows     : {len(df)}")
    print(f"  Total features : {len(df.columns)}")
    print(f"  Date range     : {df['datetime'].min().date()} → {df['datetime'].max().date()}")
    print(f"  PM2.5 mean     : {df['pm25_avg'].mean():.1f} µg/m³")
    print(f"  PM2.5 max      : {df['pm25_avg'].max():.1f} µg/m³")
    print(f"  Hazardous hours: {df['is_hazardous'].sum()} ({df['is_hazardous'].mean()*100:.1f}%)")
    print(f"\n  AQI distribution:")
    print(df["aqi_category"].value_counts().to_string())
    print("────────────────────────────────────────\n")

# ─────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────

def preprocess(input_path="data/raw/delhi_aqi_city_avg.csv"):
    os.makedirs("data/processed", exist_ok=True)

    df = load_data(input_path)
    df = resample_hourly(df)
    df = handle_missing(df)
    df = remove_outliers(df)
    df = engineer_features(df)

    df.to_csv("data/processed/delhi_aqi_processed.csv", index=False)
    print("Saved → data/processed/delhi_aqi_processed.csv")

    train, val, test = split_data(df)
    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv",     index=False)
    test.to_csv("data/processed/test.csv",   index=False)
    print("Saved → train.csv, val.csv, test.csv")

    quality_report(df)
    return df, train, val, test

if __name__ == "__main__":
    df, train, val, test = preprocess()
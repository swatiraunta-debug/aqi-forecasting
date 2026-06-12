import requests
import pandas as pd
import time
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

API_KEY  = os.getenv("OPENAQ_API_KEY", "")
BASE_URL = "https://api.openaq.org/v3"
HEADERS  = {"X-API-Key": API_KEY}

# Confirmed working Delhi CPCB stations
GOOD_LOCATION_IDS = [13, 17, 50, 235, 5613]

# ─────────────────────────────────────────
# STEP 1: Find Delhi locations
# ─────────────────────────────────────────

def get_delhi_locations():
    url    = f"{BASE_URL}/locations"
    params = {
        "coordinates": "28.6139,77.2090",
        "radius":      25000,
        "iso":         "IN",
        "limit":       50,
        "page":        1
    }

    print("Fetching Delhi locations...")
    r = requests.get(url, headers=HEADERS, params=params)

    if r.status_code != 200:
        print(f"Error {r.status_code}: {r.text}")
        r.raise_for_status()

    locations = r.json()["results"]

    # Keep only confirmed working stations
    locations = [l for l in locations if l["id"] in GOOD_LOCATION_IDS]
    print(f"Found {len(locations)} confirmed Delhi stations\n")

    for l in locations:
        print(f"  ID: {l['id']:<8} Name: {str(l.get('name','unknown')):<35}")

    return locations

# ─────────────────────────────────────────
# STEP 2: Get sensors for a location
# ─────────────────────────────────────────

def get_sensors_for_location(location_id):
    url = f"{BASE_URL}/locations/{location_id}/sensors"
    r   = requests.get(url, headers=HEADERS)

    if r.status_code != 200:
        print(f"    Could not fetch sensors for location {location_id}: {r.status_code}")
        return []

    sensors = r.json()["results"]
    return [
        s for s in sensors
        if s["parameter"]["name"] in ["pm25", "pm10", "no2", "o3"]
    ]

# ─────────────────────────────────────────
# STEP 3: Fetch hourly data for a sensor
# ─────────────────────────────────────────

def fetch_hourly_measurements(sensor_id, date_from, date_to):
    url         = f"{BASE_URL}/sensors/{sensor_id}/hours"
    all_results = []
    page        = 1

    while True:
        params = {
            "date_from": date_from,
            "date_to":   date_to,
            "limit":     1000,
            "page":      page
        }

        r = requests.get(url, headers=HEADERS, params=params)

        if r.status_code == 429:
            print(f"    Rate limited — waiting 60s...")
            time.sleep(60)
            continue

        if r.status_code == 408:
            print(f"    Timeout on sensor {sensor_id}, skipping...")
            break

        if r.status_code != 200:
            print(f"    Error {r.status_code} on sensor {sensor_id}, skipping...")
            break

        results = r.json()["results"]

        if not results:
            break

        all_results.extend(results)

        if len(results) < 1000:
            break

        page += 1
        time.sleep(0.5)

    return all_results

# ─────────────────────────────────────────
# STEP 4: Parse raw results into DataFrame
# ─────────────────────────────────────────

def parse_measurements(raw_results, sensor_id, parameter_name, loc_name, loc_id):
    rows = [{
        "datetime":      r["period"]["datetimeFrom"]["utc"],
        "value":         r["value"],
        "parameter":     parameter_name,
        "sensor_id":     sensor_id,
        "location_name": loc_name,
        "location_id":   loc_id
    } for r in raw_results]

    df             = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df             = df.sort_values("datetime").reset_index(drop=True)
    return df

# ─────────────────────────────────────────
# STEP 5: Aggregate to city-level average
# ─────────────────────────────────────────

def aggregate_to_city_avg(raw_df):
    city_avg = (
        raw_df
        .groupby(["datetime", "parameter"])["value"]
        .mean()
        .reset_index()
    )

    city_pivot = city_avg.pivot(
        index="datetime",
        columns="parameter",
        values="value"
    ).reset_index()

    city_pivot.columns.name = None

    # Rename only columns that exist — avoids KeyError if a pollutant is missing
    rename_map = {"pm25": "pm25_avg", "pm10": "pm10_avg",
                  "no2":  "no2_avg",  "o3":   "o3_avg"}
    city_pivot = city_pivot.rename(columns={
        k: v for k, v in rename_map.items() if k in city_pivot.columns
    })

    return city_pivot.sort_values("datetime").reset_index(drop=True)

# ─────────────────────────────────────────
# STEP 6: Quality check
# ─────────────────────────────────────────

def quality_check(df):
    print("\n── Quality Check ───────────────────────")
    print(f"  Rows       : {len(df)}")
    print(f"  Date range : {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"  Columns    : {list(df.columns)}")
    print(f"  Nulls      :\n{df.isnull().sum()}")
    coverage = round(len(df) / 8760 * 100, 1)
    print(f"  Coverage   : {coverage}% of one year")
    if coverage < 50:
        print("  ⚠️  WARNING: Less than 50% coverage.")
    elif coverage < 80:
        print("  ⚠️  NOTE: Gaps exist — preprocessing will fill them.")
    else:
        print("  ✓  Good coverage for modeling.")
    print("────────────────────────────────────────\n")

# ─────────────────────────────────────────
# CORE FETCH LOGIC (shared by both functions)
# ─────────────────────────────────────────

def _fetch_from_locations(date_from, date_to):
    """Fetch data from all confirmed Delhi stations for a date range."""
    locations = get_delhi_locations()
    if not locations:
        print("No Delhi locations found. Check your API key.")
        return None

    all_dfs = []
    for loc in locations:
        loc_id   = loc["id"]
        loc_name = loc.get("name", "unknown")
        print(f"\nProcessing: {loc_name} (ID: {loc_id})")

        sensors = get_sensors_for_location(loc_id)
        if not sensors:
            print("  No relevant sensors, skipping.")
            continue

        print(f"  Sensors: {[(s['id'], s['parameter']['name']) for s in sensors]}")

        seen_params = set()
        for sensor in sensors:
            param_name = sensor["parameter"]["name"]
            if param_name in seen_params:
                continue
            seen_params.add(param_name)

            sensor_id = sensor["id"]
            print(f"  Fetching {param_name} (sensor {sensor_id})...")

            raw = fetch_hourly_measurements(sensor_id, date_from, date_to)
            if raw:
                df = parse_measurements(raw, sensor_id, param_name, loc_name, loc_id)
                all_dfs.append(df)
                print(f"  ✓ {len(df)} rows")
            else:
                print(f"  ✗ No data returned")

            time.sleep(1)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else None

# ─────────────────────────────────────────
# DOWNLOAD — full historical fetch
# ─────────────────────────────────────────

def download_delhi_aqi(date_from="2016-01-01", date_to="2026-12-31"):
    os.makedirs("data/raw", exist_ok=True)

    raw_df = _fetch_from_locations(date_from, date_to)
    if raw_df is None:
        print("\nNo data fetched. Possible reasons:")
        print("  1. API key invalid")
        print("  2. No sensor data for this date range")
        print("  3. All requests timed out")
        return None

    raw_df.to_csv("data/raw/delhi_aqi_raw.csv", index=False)
    print(f"\nRaw data saved → data/raw/delhi_aqi_raw.csv ({len(raw_df)} rows)")

    city_df = aggregate_to_city_avg(raw_df)
    city_df.to_csv("data/raw/delhi_aqi_city_avg.csv", index=False)
    print(f"City average saved → data/raw/delhi_aqi_city_avg.csv ({len(city_df)} rows)")

    quality_check(city_df)
    return city_df

# ─────────────────────────────────────────
# UPDATE — incremental fetch (used by dashboard)
# ─────────────────────────────────────────

def update_delhi_aqi():
    raw_path  = Path("data/raw/delhi_aqi_raw.csv")
    city_path = Path("data/raw/delhi_aqi_city_avg.csv")

    # If no existing data, do a full download
    if not city_path.exists():
        print("No existing data found — running full download.")
        return download_delhi_aqi()

    # Find last recorded timestamp
    existing  = pd.read_csv(city_path, parse_dates=["datetime"])
    last_time = existing["datetime"].max()

    if last_time.tzinfo is None:
        last_time = last_time.tz_localize("UTC")
    else:
        last_time = last_time.tz_convert("UTC")

    start_time = last_time + pd.Timedelta(hours=1)
    end_time   = pd.Timestamp.now(tz="UTC").floor("h")

    if start_time >= end_time:
        print("Data is already up to date.")
        return existing

    date_from = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_to   = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"Fetching new data: {date_from} → {date_to}")

    new_raw = _fetch_from_locations(date_from, date_to)
    if new_raw is None:
        print("No new data fetched — returning existing data.")
        return existing

    # Merge with existing raw data
    if raw_path.exists():
        old_raw = pd.read_csv(raw_path, parse_dates=["datetime"])
        raw_df  = pd.concat([old_raw, new_raw], ignore_index=True)
    else:
        raw_df = new_raw

    # Deduplicate
    raw_df = (
        raw_df
        .drop_duplicates(subset=["datetime", "sensor_id", "parameter"], keep="last")
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_path, index=False)
    print(f"Updated raw data → {raw_path} ({len(raw_df)} rows)")

    city_df = aggregate_to_city_avg(raw_df)
    city_df.to_csv(city_path, index=False)
    print(f"Updated city average → {city_path} ({len(city_df)} rows)")

    quality_check(city_df)
    return city_df

# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    df = download_delhi_aqi()
    if df is not None:
        print("\nSample:")
        print(df.head(5).to_string())
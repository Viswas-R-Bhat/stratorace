"""
ingest_fastf1.py — StratoRace Real Data Pipeline
Downloads real lap-by-lap telemetry from FastF1 for 2022–2024 seasons.
Produces:
  - data/lap_telemetry.parquet  (every lap for every driver in every race)
  - data/tyre_model.json        (regression fits per compound)
  - data/race_meta.json         (race metadata: total laps, avg pit loss)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import fastf1

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "fastf1_cache")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

SEASONS = [2022, 2023]

# Races to skip (cancelled, incomplete, or problematic data)
SKIP_RACES = {"Miami Grand Prix"}

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

fastf1.Cache.enable_cache(CACHE_DIR)


def get_race_schedule(year):
    """Get all race events for a given year (exclude testing, sprints)."""
    schedule = fastf1.get_event_schedule(year)
    # Only get conventional races (RoundNumber > 0)
    races = schedule[schedule["RoundNumber"] > 0]
    return races


def process_session(year, event_name, round_number):
    """
    Load a race session and extract lap-level data for every driver.
    Returns a DataFrame with one row per driver per lap.
    """
    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load(telemetry=False, weather=True, messages=False)
    except Exception as e:
        print(f"  ✗ Failed to load {year} {event_name}: {e}")
        return None

    laps = session.laps
    if laps is None or laps.empty:
        print(f"  ✗ No lap data for {year} {event_name}")
        return None

    weather = session.weather_data
    total_laps = int(laps["LapNumber"].max())

    rows = []
    drivers = laps["Driver"].unique()

    for driver in drivers:
        driver_laps = laps[laps["Driver"] == driver].sort_values("LapNumber")

        # Track stint/pit information
        stint_number = 0
        num_pits_so_far = 0
        prev_stint = None

        for _, lap in driver_laps.iterrows():
            lap_num = int(lap["LapNumber"])

            # Compound and stint tracking
            compound = str(lap.get("Compound", "UNKNOWN")).upper()
            if compound not in ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"):
                compound = "UNKNOWN"
            # Normalize INTERMEDIATE → INTER
            if compound == "INTERMEDIATE":
                compound = "INTER"

            tyre_life = lap.get("TyreLife", np.nan)
            if pd.isna(tyre_life):
                tyre_life = 0
            else:
                tyre_life = int(tyre_life)

            # Detect pit stops from Stint column
            current_stint = lap.get("Stint", None)
            if prev_stint is not None and current_stint != prev_stint:
                num_pits_so_far += 1
                stint_number += 1
            prev_stint = current_stint

            # Lap time in seconds
            lap_time_td = lap.get("LapTime", pd.NaT)
            if pd.isna(lap_time_td):
                lap_time_seconds = np.nan
            else:
                lap_time_seconds = lap_time_td.total_seconds()

            # Position
            position = lap.get("Position", np.nan)
            if pd.notna(position):
                position = int(position)

            # Is this a pit in-lap or pit out-lap?
            is_pit_in = bool(lap.get("PitInTime", pd.NaT) is not pd.NaT and pd.notna(lap.get("PitInTime", pd.NaT)))
            is_pit_out = bool(lap.get("PitOutTime", pd.NaT) is not pd.NaT and pd.notna(lap.get("PitOutTime", pd.NaT)))

            # Pit duration
            pit_duration = np.nan
            if is_pit_in:
                pit_time = lap.get("PitInTime", pd.NaT)
                pit_out = lap.get("PitOutTime", pd.NaT)
                # Some laps have both PitInTime and PitOutTime
                # Otherwise approximate from lap time

            # Weather at this lap
            track_temp = np.nan
            air_temp = np.nan
            rainfall = False
            if weather is not None and not weather.empty:
                # Get weather closest to this lap's time
                lap_start = lap.get("LapStartDate", pd.NaT)
                if pd.notna(lap_start):
                    try:
                        # Find closest weather reading
                        time_deltas = abs(weather["Time"] - (lap_start - session.session_start_time if hasattr(session, 'session_start_time') else pd.Timedelta(0)))
                        closest_idx = time_deltas.idxmin() if not time_deltas.empty else None
                        if closest_idx is not None:
                            w = weather.loc[closest_idx]
                            track_temp = float(w.get("TrackTemp", np.nan))
                            air_temp = float(w.get("AirTemp", np.nan))
                            rainfall_val = w.get("Rainfall", False)
                            rainfall = bool(rainfall_val) if pd.notna(rainfall_val) else False
                    except Exception:
                        pass

            # Speed trap
            speed_st = lap.get("SpeedST", np.nan)
            if pd.notna(speed_st):
                speed_st = float(speed_st)

            rows.append({
                "year": year,
                "gp": event_name,
                "driver": driver,
                "lap_number": lap_num,
                "total_laps": total_laps,
                "compound": compound,
                "tyre_life": tyre_life,
                "lap_time_seconds": lap_time_seconds,
                "position": position,
                "track_temp": track_temp,
                "air_temp": air_temp,
                "rainfall": rainfall,
                "speed_st": speed_st,
                "is_pit_in": is_pit_in,
                "is_pit_out": is_pit_out,
                "stint_number": stint_number,
                "num_pits_so_far": num_pits_so_far,
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # ── Compute lap_time_delta (delta to race median for that lap) ────────
    lap_medians = df.groupby("lap_number")["lap_time_seconds"].median()
    df["lap_time_delta"] = df.apply(
        lambda r: r["lap_time_seconds"] - lap_medians.get(r["lap_number"], r["lap_time_seconds"])
        if pd.notna(r["lap_time_seconds"]) else np.nan,
        axis=1
    )

    # ── Compute gap_ahead and gap_behind from position + lap times ────────
    # For each lap, sort drivers by position, compute cumulative time gaps
    df["gap_ahead"] = np.nan
    df["gap_behind"] = np.nan

    for lap_num in df["lap_number"].unique():
        lap_df = df[df["lap_number"] == lap_num].sort_values("position")
        positions = lap_df["position"].values
        times = lap_df["lap_time_seconds"].values
        indices = lap_df.index.values

        for i in range(len(indices)):
            if i > 0 and pd.notna(times[i]) and pd.notna(times[i - 1]):
                gap = abs(times[i] - times[i - 1])
                df.loc[indices[i], "gap_ahead"] = min(gap, 60.0)
            elif i == 0:
                df.loc[indices[i], "gap_ahead"] = 60.0  # leader has no car ahead

            if i < len(indices) - 1 and pd.notna(times[i]) and pd.notna(times[i + 1]):
                gap = abs(times[i + 1] - times[i])
                df.loc[indices[i], "gap_behind"] = min(gap, 60.0)
            elif i == len(indices) - 1:
                df.loc[indices[i], "gap_behind"] = 60.0  # last car has no car behind

    return df


def compute_tyre_model(df):
    """
    Fit linear regression per compound: lap_time_delta ~ tyre_life
    Returns list of dicts with real regression metrics.
    """
    compounds_data = []

    # Filter: only non-pit laps, known compounds, valid times
    valid = df[
        (~df["is_pit_in"]) & (~df["is_pit_out"]) &
        (df["compound"].isin(["SOFT", "MEDIUM", "HARD", "INTER"])) &
        (df["lap_time_delta"].notna()) &
        (df["tyre_life"] > 0) &
        (df["lap_time_seconds"].notna()) &
        # Remove extreme outliers (safety cars, red flags)
        (df["lap_time_delta"].abs() < 10.0)
    ].copy()

    for compound in ["SOFT", "MEDIUM", "HARD", "INTER"]:
        comp_df = valid[valid["compound"] == compound]
        if len(comp_df) < 20:
            print(f"  ! Skipping {compound}: only {len(comp_df)} data points")
            continue

        X = comp_df["tyre_life"].values.reshape(-1, 1)
        y = comp_df["lap_time_delta"].values

        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        # Estimate cliff lap: where predicted delta exceeds 1.5 seconds
        cliff_lap = None
        for age in range(1, 60):
            pred = reg.predict([[age]])[0]
            if pred > 1.5:
                cliff_lap = age
                break
        if cliff_lap is None:
            cliff_lap = 40  # default if no cliff detected

        compounds_data.append({
            "Compound": compound,
            "N": int(len(comp_df)),
            "Slope": float(reg.coef_[0]),
            "Intercept": float(reg.intercept_),
            "R2": float(r2),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "CliffLap": cliff_lap,
        })

        print(f"  {compound:8s}  N={len(comp_df):5d}  Slope={reg.coef_[0]:.4f}  "
              f"R²={r2:.3f}  MAE={mae:.3f}s  Cliff≈Lap {cliff_lap}")

    return compounds_data


def compute_race_meta(df):
    """Compute per-race metadata: total laps, average pit stop time loss."""
    meta = {}

    for (year, gp), race_df in df.groupby(["year", "gp"]):
        total_laps = int(race_df["total_laps"].iloc[0])

        # Estimate pit loss: median lap time of pit-in laps minus median normal lap
        pit_laps = race_df[race_df["is_pit_in"] & race_df["lap_time_seconds"].notna()]
        normal_laps = race_df[
            (~race_df["is_pit_in"]) & (~race_df["is_pit_out"]) &
            race_df["lap_time_seconds"].notna() &
            (race_df["lap_time_delta"].abs() < 5.0)
        ]

        if len(pit_laps) > 0 and len(normal_laps) > 0:
            pit_loss = float(pit_laps["lap_time_seconds"].median() - normal_laps["lap_time_seconds"].median())
        else:
            pit_loss = 22.0  # F1 average if can't compute

        key = f"{year}_{gp}"
        meta[key] = {
            "year": int(year),
            "gp": gp,
            "total_laps": total_laps,
            "pit_loss_seconds": round(max(pit_loss, 15.0), 1),  # pit loss is at least 15s
        }

    return meta


def main():
    print("=" * 70)
    print("StratoRace - FastF1 Real Data Ingestion")
    print("Seasons:", SEASONS)
    print("=" * 70)

    all_dfs = []

    for year in SEASONS:
        print(f"\n{'-' * 40}")
        print(f"Season {year}")
        print(f"{'-' * 40}")

        schedule = get_race_schedule(year)
        print(f"  Found {len(schedule)} events")

        for _, event in schedule.iterrows():
            event_name = event["EventName"]
            round_num = int(event["RoundNumber"])

            if event_name in SKIP_RACES:
                print(f"  - Skipping {event_name}")
                continue

            print(f"  > {event_name} (Round {round_num})...", end=" ", flush=True)
            df = process_session(year, event_name, round_num)

            if df is not None:
                all_dfs.append(df)
                n_drivers = df["driver"].nunique()
                n_laps = len(df)
                print(f"OK {n_drivers} drivers, {n_laps} laps")
            else:
                print("FAIL skipped")

    if not all_dfs:
        print("\nFAIL No data collected. Check network connection / FastF1 availability.")
        return

    # -- Combine all data ------------------------------------------------------
    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n{'=' * 70}")
    print(f"Total: {len(full_df)} lap records across {full_df['gp'].nunique()} races")
    print(f"Drivers: {sorted(full_df['driver'].unique())}")
    print(f"Compounds: {sorted(full_df['compound'].unique())}")

    # ── Save lap telemetry ────────────────────────────────────────────────────
    parquet_path = os.path.join(OUTPUT_DIR, "lap_telemetry.parquet")
    full_df.to_parquet(parquet_path, index=False)
    print(f"\nOK Saved {parquet_path} ({len(full_df)} rows, {os.path.getsize(parquet_path) / 1e6:.1f} MB)")

    # ── Compute tyre model ────────────────────────────────────────────────────
    print("\n-- Tyre Degradation Model --")
    tyre_data = compute_tyre_model(full_df)
    tyre_path = os.path.join(OUTPUT_DIR, "tyre_model.json")
    with open(tyre_path, "w") as f:
        json.dump(tyre_data, f, indent=2)
    print(f"OK Saved {tyre_path}")

    # ── Compute race metadata ─────────────────────────────────────────────────
    print("\n-- Race Metadata --")
    race_meta = compute_race_meta(full_df)
    meta_path = os.path.join(OUTPUT_DIR, "race_meta.json")
    with open(meta_path, "w") as f:
        json.dump(race_meta, f, indent=2)
    print(f"OK Saved {meta_path} ({len(race_meta)} races)")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"  Races:       {full_df['gp'].nunique()}")
    print(f"  Lap records: {len(full_df)}")
    print(f"  Pit stops:   {full_df['is_pit_in'].sum()}")
    print(f"  Compounds:   {dict(full_df['compound'].value_counts())}")
    print(f"  Seasons:     {sorted(full_df['year'].unique())}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

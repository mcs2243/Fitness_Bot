import argparse
import os
from pathlib import Path

import pandas as pd


def load_strong(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Date"]).dt.date
    rename = {
        "Date": "date_time",
        "Workout Name": "workout_name",
        "Duration": "duration",
        "Exercise Name": "exercise",
        "Set Order": "set_order",
        "Weight": "weight_lb",
        "Reps": "reps",
        "Distance": "distance",
        "Seconds": "seconds",
        "Notes": "exercise_notes",
        "Workout Notes": "workout_notes",
        "RPE": "rpe",
    }
    df = df.rename(columns=rename)
    return df


def load_whoop(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Cycle start time"]).dt.date
    rename = {
        "Cycle start time": "cycle_start",
        "Cycle end time": "cycle_end",
        "Cycle timezone": "cycle_tz",
        "Recovery score %": "recovery_score",
        "Resting heart rate (bpm)": "rhr",
        "Heart rate variability (ms)": "hrv",
        "Skin temp (celsius)": "skin_temp_c",
        "Blood oxygen %": "spo2",
        "Day Strain": "day_strain",
        "Energy burned (cal)": "calories",
        "Max HR (bpm)": "max_hr",
        "Average HR (bpm)": "avg_hr",
        "Sleep onset": "sleep_onset",
        "Wake onset": "wake_onset",
        "Sleep performance %": "sleep_performance",
        "Respiratory rate (rpm)": "resp_rate",
        "Asleep duration (min)": "asleep_min",
        "In bed duration (min)": "in_bed_min",
        "Light sleep duration (min)": "light_sleep_min",
        "Deep (SWS) duration (min)": "deep_sleep_min",
        "REM duration (min)": "rem_sleep_min",
        "Awake duration (min)": "awake_min",
        "Sleep need (min)": "sleep_need_min",
        "Sleep debt (min)": "sleep_debt_min",
        "Sleep efficiency %": "sleep_efficiency",
        "Sleep consistency %": "sleep_consistency",
    }
    df = df.rename(columns=rename)
    return df


def merge_strong_whoop(df_strong: pd.DataFrame, df_whoop: pd.DataFrame) -> pd.DataFrame:
    whoop_cols = [
        "date",
        "recovery_score",
        "rhr",
        "hrv",
        "sleep_performance",
        "asleep_min",
        "day_strain",
        "calories",
        "avg_hr",
        "max_hr",
    ]
    whoop_trimmed = df_whoop[[c for c in whoop_cols if c in df_whoop.columns]].copy()
    merged = df_strong.merge(whoop_trimmed, on="date", how="left")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Clean and merge Strong + Whoop CSV exports.")
    parser.add_argument(
        "--strong-path",
        default=r"C:\Users\mcs22\Downloads\strong (7).csv",
        help="Path to the Strong export CSV.",
    )
    parser.add_argument(
        "--whoop-path",
        default=r"C:\Users\mcs22\Downloads\physiological_cycles.csv",
        help="Path to the Whoop physiological cycles CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/Strong_Whoop_cleaned_small.csv",
        help="Output path for the merged CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick iterations.",
    )
    args = parser.parse_args()

    strong_path = Path(args.strong_path)
    whoop_path = Path(args.whoop_path)
    if not strong_path.exists():
        raise FileNotFoundError(f"Strong CSV not found: {strong_path}")
    if not whoop_path.exists():
        raise FileNotFoundError(f"Whoop CSV not found: {whoop_path}")

    print(f"Loading Strong CSV from: {strong_path}")
    df_strong = load_strong(strong_path)
    print(f"Loaded {len(df_strong)} strong rows")

    print(f"Loading Whoop CSV from: {whoop_path}")
    df_whoop = load_whoop(whoop_path)
    print(f"Loaded {len(df_whoop)} whoop rows")

    merged = merge_strong_whoop(df_strong, df_whoop)
    if args.limit:
        merged = merged.head(args.limit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Wrote merged CSV to: {output_path} (rows: {len(merged)})")


if __name__ == "__main__":
    main()

import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
import io

def reduce_strong_whoop(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Map Strong + Whoop merged CSV into the minimal fields we prompt with."""
    # Keep only a concise subset to avoid blowing context.
    wanted = {
        "Date": "date",
        "Workout Name": "workout_name",
        "Exercise Name": "exercise",
        "Set Order": "set_order",
        "Weight": "weight_lb",
        "Reps": "reps",
        "RPE": "rpe",
        "Notes per Exercise": "exercise_notes",
        "Overall Notes per Workout": "workout_notes",
        "Recovery score %": "recovery_score",
        "Resting heart rate (bpm)": "rhr",
        "Heart rate variability (ms)": "hrv",
        "Sleep performance %": "sleep_performance",
        "Asleep duration (min)": "asleep_min",
        "Day Strain": "day_strain",
        "Energy burned (cal)": "calories",
        "Average HR (bpm)": "avg_hr",
        "Max HR (bpm)": "max_hr",
    }
    present_cols = {k: v for k, v in wanted.items() if k in df.columns}
    trimmed = df[list(present_cols.keys())].rename(columns=present_cols)
    return trimmed.to_dict(orient="records")


def parse_and_filter_dates(df: pd.DataFrame, months: Optional[int]) -> pd.DataFrame:
    """Parse date columns and filter to recent months if requested."""
    date_col = None
    for candidate in ["date", "Date", "Cycle start time"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if not date_col:
        return df

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    if months:
        max_date = df[date_col].max()
        cutoff = max_date - pd.DateOffset(months=months)
        filtered = df[df[date_col] >= cutoff]
        if not filtered.empty:
            df = filtered
    return df


def default_sample_workout() -> List[Dict[str, Any]]:
    """Provide a small in-memory sample so the script runs without files."""
    return [
        {
            "date": "2024-11-01",
            "exercise": "Back Squat",
            "sets": 4,
            "reps": 6,
            "weight_lb": 275,
            "rpe": 8.0,
            "notes": "Last set grindy; knees fine; bracing solid.",
            "sleep_hours": 7.2,
            "recovery_score": 78,
        },
        {
            "date": "2024-11-02",
            "exercise": "Bench Press",
            "sets": 5,
            "reps": 5,
            "weight_lb": 225,
            "rpe": 7.5,
            "notes": "Stable bar path; slight shoulder tightness.",
            "sleep_hours": 6.5,
            "recovery_score": 70,
        },
        {
            "date": "2024-11-02",
            "exercise": "Deadlift",
            "sets": 3,
            "reps": 5,
            "weight_lb": 365,
            "rpe": 8.5,
            "notes": "Grip slipping by last set; hamstrings sore.",
            "sleep_hours": 6.5,
            "recovery_score": 70,
        },
    ]


def load_workout_data(content: bytes, filename: str, limit: Optional[int] = None, months: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load workout data from bytes (uploaded file)."""
    
    if filename.lower().endswith(".json"):
        data = json.loads(content.decode("utf-8"))
        if isinstance(data, dict):
            data = data.get("workouts") or data.get("data") or [data]
        if not isinstance(data, list):
            raise ValueError("Expected a list of workout records in JSON.")
        return data[:limit] if limit else data

    if filename.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
        df = parse_and_filter_dates(df, months)
        # If this looks like the merged Strong/Whoop export, reduce fields
        if {"Workout Name", "Exercise Name"}.issubset(set(df.columns)):
            df_records = reduce_strong_whoop(df)
        else:
            df_records = df.to_dict(orient="records")

        if limit:
            df_records = df_records[-limit:]  # take most recent after sorting
        return df_records

    # Fallback or error
    raise ValueError("Unsupported file format. Please upload .csv or .json")


def format_workout_records(records: List[Dict[str, Any]]) -> str:
    """Render workout records into a readable bullet list for the LLM."""
    lines = []
    for row in records:
        parts = [
            f"Date: {row.get('date')}",
            f"Workout: {row.get('workout_name') or row.get('workout')}",
            f"Exercise: {row.get('exercise')}",
            f"Set: {row.get('set_order')}",
            f"Weight (lb): {row.get('weight_lb')}",
            f"Reps: {row.get('reps')}",
            f"RPE: {row.get('rpe')}",
            f"Exercise notes: {row.get('exercise_notes') or row.get('notes')}",
            f"Workout notes: {row.get('workout_notes')}",
            f"Recovery: {row.get('recovery_score')}",
            f"Sleep perf %: {row.get('sleep_performance')}",
            f"HRV: {row.get('hrv')}",
            f"RHR: {row.get('rhr')}",
            f"Day strain: {row.get('day_strain')}",
            f"Avg HR: {row.get('avg_hr')}",
            f"Max HR: {row.get('max_hr')}",
        ]
        # Filter out empty entries to keep the prompt concise.
        parts = [p for p in parts if not p.endswith("None")]
        lines.append(" | ".join(parts))
    return "\n".join(lines)

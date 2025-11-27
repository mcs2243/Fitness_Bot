import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import Client

load_dotenv()


class Config:
    LLM_MODEL = "gpt-4o-mini"
    LLM_TEMPERATURE = 0.7
    MAX_TOKENS = 500
    TIMEOUT = 30
    MAX_RETRIES = 2
    PROJECT_NAME = "Fitness_Bot"
    DATA_DIR = Path("data")


def enable_langsmith_if_configured() -> None:
    """Enable LangSmith tracing when an API key is present."""
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGSMITH_PROJECT"] = Config.PROJECT_NAME
        # Instantiate client so failures surface early
        Client(
            api_url=os.environ["LANGSMITH_ENDPOINT"],
            api_key=os.environ["LANGSMITH_API_KEY"],
        )
        print("LangSmith tracing: Enabled")
    else:
        print("LangSmith tracing: Disabled (LANGSMITH_API_KEY not set)")


def init_llm() -> ChatOpenAI:
    """Create the ChatOpenAI client with the configured defaults."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set in environment variables")

    return ChatOpenAI(
        model=Config.LLM_MODEL,
        temperature=Config.LLM_TEMPERATURE,
        max_tokens=Config.MAX_TOKENS,
        timeout=Config.TIMEOUT,
        max_retries=Config.MAX_RETRIES,
    )


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


def load_workout_data(path: Optional[str], limit: Optional[int]) -> List[Dict[str, Any]]:
    """Load workout data from JSON/CSV; use default CSV if present, else sample."""
    default_csv = find_latest_csv(Config.DATA_DIR)
    file_path: Optional[Path] = Path(path) if path else default_csv

    if file_path and not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    if file_path and file_path.suffix.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("workouts") or data.get("data") or [data]
        if not isinstance(data, list):
            raise ValueError("Expected a list of workout records in JSON.")
        return data[:limit] if limit else data

    if file_path and file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
        if limit:
            df = df.head(limit)
        # If this looks like the merged Strong/Whoop export, reduce fields
        if {"Workout Name", "Exercise Name"}.issubset(set(df.columns)):
            return reduce_strong_whoop(df)
        return df.to_dict(orient="records")

    # Fallback: embedded sample
    data = default_sample_workout()
    return data[:limit] if limit else data


def find_latest_csv(data_dir: Path) -> Optional[Path]:
    """Return the most recently modified CSV in the data directory."""

    if not data_dir.exists():
        return None

    candidates = sorted(
        data_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


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


def analyze_workout(llm: ChatOpenAI, formatted_data: str, goal: str) -> str:
    """Send the formatted workout data to the LLM for insights."""
    system = SystemMessage(
        content=(
            "You are an evidence-based bodybuilding coach. "
            "Given recent training logs and recovery inputs, you will: "
            "1) Summarize performance and recovery; "
            "2) Call out risks or form concerns; "
            "3) Recommend specific adjustments for the next session."
        )
    )
    user = HumanMessage(
        content=(
            f"User goal: {goal}\n\n"
            f"Recent workout data:\n{formatted_data}\n\n"
            "Return a short summary and clear next-step adjustments."
        )
    )
    response = llm.invoke([system, user])
    return response.content


def main():
    parser = argparse.ArgumentParser(
        description="Analyze workout data and return insights."
    )
    parser.add_argument(
        "--data",
        type=str,
        help=(
            "Path to workout data (.json or .csv). "
            "If omitted, tries your local Strong/Whoop CSV; else falls back to a sample."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Limit rows used from the dataset to keep prompts small (default: 30).",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="Build strength and keep fatigue in check while improving technique.",
        help="Short description of the user's current goal.",
    )
    args = parser.parse_args()

    enable_langsmith_if_configured()
    llm = init_llm()

    records = load_workout_data(args.data, args.limit)
    formatted = format_workout_records(records)

    print("=== Input Data ===")
    print(formatted)
    print("\n=== Insight ===")
    try:
        insight = analyze_workout(llm, formatted, args.goal)
        print(insight)
    except Exception as exc:
        print(f"Error while analyzing workout data: {exc}")


if __name__ == "__main__":
    main()

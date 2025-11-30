import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()


def find_date_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ["date", "Date", "Cycle start time", "date_time"]:
        if candidate in df.columns:
            return candidate
    return None


def filter_recent(df: pd.DataFrame, months: Optional[int]) -> pd.DataFrame:
    date_col = find_date_column(df)
    if not date_col:
        return df
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    if months and months > 0:
        max_date = df[date_col].max()
        cutoff = max_date - pd.DateOffset(months=months)
        recent = df[df[date_col] >= cutoff]
        if not recent.empty:
            df = recent
    return df


def reduce_strong_whoop(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Trim to the key fields to keep embeddings concise."""
    wanted = {
        "date_time": "date_time",
        "date": "date",
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
    present = {k: v for k, v in wanted.items() if k in df.columns}
    return df[list(present.keys())].rename(columns=present).to_dict(orient="records")


def format_record(rec: Dict[str, Any]) -> str:
    parts = []
    for key in [
        "date",
        "date_time",
        "workout_name",
        "exercise",
        "set_order",
        "weight_lb",
        "reps",
        "rpe",
        "exercise_notes",
        "workout_notes",
        "recovery_score",
        "sleep_performance",
        "hrv",
        "rhr",
        "day_strain",
        "avg_hr",
        "max_hr",
    ]:
        if key in rec and pd.notna(rec[key]):
            parts.append(f"{key}: {rec[key]}")
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Ingest Strong/Whoop CSV into a local Chroma store.")
    parser.add_argument("--data", required=True, help="Path to cleaned Strong/Whoop CSV.")
    parser.add_argument("--persist-dir", default="chroma_store", help="Chroma persistence directory.")
    parser.add_argument("--collection", default="fitness_logs", help="Chroma collection name.")
    parser.add_argument("--months", type=int, default=6, help="Keep last N months (anchored to latest date). Use 0 to disable.")
    parser.add_argument("--limit", type=int, default=500, help="Limit most recent rows after filtering.")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    df = filter_recent(df, args.months)
    if args.limit and args.limit > 0:
        df = df.tail(args.limit)
    if df.empty:
        raise ValueError("No rows to embed after filtering. Adjust --months/--limit.")

    # Reduce fields and format to text
    if {"Workout Name", "Exercise Name"}.issubset(df.columns):
        records = reduce_strong_whoop(df)
    else:
        records = df.to_dict(orient="records")

    texts = [format_record(rec) for rec in records]
    metadatas = [{"row": i} for i in range(len(records))]

    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY from env
    vs = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=args.collection,
        metadatas=metadatas,
    )
    vs.persist()
    print(f"Ingested {len(texts)} records into Chroma at {persist_dir} (collection='{args.collection}').")


if __name__ == "__main__":
    main()

import argparse
import asyncio
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import cognee
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Configure embedding model
os.environ["EMBEDDING_PROVIDER"] = "openai"
os.environ["EMBEDDING_MODEL"] = "text-embedding-3-small"  # or "text-embedding-3-large" for better quality

# Configuration
DEFAULT_CHUNK_SIZE = 10  # Process 10 rows at a time
DEFAULT_DATA_DIR = Path("data")
CHECKPOINT_NAME = "cognee_checkpoint.json"


def ensure_data_dir(data_dir: Path) -> Path:
    """Create a local filesystem root for uploaded datasets."""

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def stage_data_file(source_path: Path, data_dir: Path) -> Path:
    """Copy a user-provided dataset into the managed data directory."""

    if not source_path.exists():
        raise FileNotFoundError(f"Data file not found: {source_path}")

    ensure_data_dir(data_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = data_dir / f"{source_path.stem}_{timestamp}{source_path.suffix}"
    shutil.copy2(source_path, destination)
    print(f"Staged dataset in {destination}")
    return destination


def find_latest_data_file(data_dir: Path) -> Optional[Path]:
    """Return the most recently modified CSV file in the data directory."""

    ensure_data_dir(data_dir)
    candidates = sorted(
        data_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


def list_data_files(data_dir: Path) -> None:
    """Print available staged datasets for quick inspection."""

    ensure_data_dir(data_dir)
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        print("No staged CSV files found. Use --data-file to upload one.")
        return

    print("Available staged CSV datasets:")
    for f in files:
        modified = datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")
        print(f"- {f.name} (updated {modified})")


def load_checkpoint(checkpoint_file: Path) -> dict:
    if checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_processed": -1, "total_processed": 0}


def checkpoint_covers_dataset(checkpoint: dict, total_rows: int) -> bool:
    """Return True if the checkpoint already includes every row in the dataset."""

    last_processed = checkpoint.get("last_processed", -1)
    return last_processed >= total_rows - 1


def save_checkpoint(checkpoint_file: Path, last_processed: int, total_rows: int) -> None:
    checkpoint = {
        "last_processed": last_processed,
        "total_processed": min(total_rows, last_processed + 1),
        "timestamp": datetime.now().isoformat(),
    }
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f)


async def process_chunk(chunk: pd.DataFrame) -> bool:
    """Process a chunk of data and add to Cognee."""

    chunk_text = chunk.to_string()
    try:
        await cognee.add(chunk_text)
        return True
    except Exception as exc:  # pragma: no cover - runtime tracing
        print(f"Error processing chunk: {exc}")
        return False


async def setup_cognee(
    data_path: Path,
    checkpoint_file: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    force: bool = False,
) -> None:
    """Initialize Cognee and process data in chunks.

    If a checkpoint already covers the full dataset, the run is skipped unless
    ``force`` is True. This keeps existing embeddings intact when you only want
    to query or inspect them.
    """

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    total_rows = len(df)
    print(f"Loaded {total_rows} rows from {data_path}")

    checkpoint = load_checkpoint(checkpoint_file)
    start_idx = checkpoint.get("last_processed", -1) + 1

    if checkpoint_covers_dataset(checkpoint, total_rows) and not force:
        print(
            "Checkpoint already covers the dataset. Skipping embedding run. "
            "Use --force to rebuild."
        )
        return

    if force:
        print("Force rebuild requested: pruning existing Cognee data...")
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        start_idx = 0
    elif start_idx == 0:
        print("Initializing new Cognee database...")
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    else:
        print(f"Resuming from checkpoint at row {start_idx}")

    for i in tqdm(range(start_idx, total_rows, chunk_size), desc="Processing data"):
        chunk = df.iloc[i : i + chunk_size]
        success = await process_chunk(chunk)
        if success:
            save_checkpoint(checkpoint_file, i + len(chunk) - 1, total_rows)
        else:
            print(f"Stopped at row {i}")
            break

    print("Finalizing embeddings...")
    await cognee.cognify()
    print("Setup complete!")


async def query_cognee(query: str):
    """Search Cognee's knowledge graph."""

    print(f"Searching for: {query}")
    results = await cognee.search(query_text=query)
    return [r for r in results]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cognee setup and querying")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild embeddings from scratch, ignoring any existing checkpoint.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Optional query text to run after ensuring the checkpoint is ready.",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        help="Path to a CSV workout export to stage into the managed data directory.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory where staged datasets and checkpoints are stored (default: ./data).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of rows to embed per batch (default: 10).",
    )
    parser.add_argument(
        "--list-data",
        action="store_true",
        help="List staged datasets and exit without embedding.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    async def main():
        args = parse_args()

        data_dir = ensure_data_dir(Path(args.data_dir))
        checkpoint_file = data_dir / CHECKPOINT_NAME

        if args.list_data:
            list_data_files(data_dir)
            return

        staged_path: Optional[Path] = None
        if args.data_file:
            staged_path = stage_data_file(Path(args.data_file), data_dir)

        dataset = staged_path or find_latest_data_file(data_dir)
        if not dataset:
            raise FileNotFoundError(
                "No dataset found. Use --data-file to stage a CSV before running embeddings."
            )

        print(f"Using dataset: {dataset}")
        print("Cognee LLM key loaded:", "OK" if os.getenv("OPENAI_API_KEY") else "Missing!")

        await setup_cognee(
            data_path=dataset,
            checkpoint_file=checkpoint_file,
            chunk_size=args.chunk_size,
            force=args.force,
        )

        if args.query:
            results = await query_cognee(args.query)
            print("\nResults:")
            for i, r in enumerate(results, 1):
                print(f"{i}. {r}")
        else:
            while True:
                query = input("\nEnter your query (or 'quit' to exit): ")
                if query.lower() in ["quit", "exit", "q"]:
                    break

                results = await query_cognee(query)
                print("\nResults:")
                for i, r in enumerate(results, 1):
                    print(f"{i}. {r}")

    asyncio.run(main())

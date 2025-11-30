import os
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import cognee
import pandas as pd
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Configure embedding model
os.environ["EMBEDDING_PROVIDER"] = "openai"
os.environ["EMBEDDING_MODEL"] = "text-embedding-3-small"  # or "text-embedding-3-large" for better quality
EMBED_RATE_PER_1K = 0.00002  # text-embedding-3-small cost per 1K tokens

# Configuration defaults (can be overridden via CLI)
CHUNK_SIZE = 10  # Process 10 rows at a time
DEFAULT_DATA_PATH = Path("data/Strong_Whoop_cleaned_small.csv")
CHECKPOINT_FILE = "cognee_checkpoint.json"

print("Cognee LLM key loaded:", "OK" if os.getenv("OPENAI_API_KEY") else "Missing!")


def find_date_column(df: pd.DataFrame) -> str | None:
    for candidate in ["date", "Date", "Cycle start time"]:
        if candidate in df.columns:
            return candidate
    return None


def filter_recent(df: pd.DataFrame, months: int | None) -> pd.DataFrame:
    """Parse dates, sort ascending, and keep only the recent window if provided."""
    date_col = find_date_column(df)
    if not date_col:
        return df
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    if months and months > 0:
        cutoff = pd.Timestamp.now() - pd.DateOffset(months=months)
        df = df[df[date_col] >= cutoff]
    return df


def estimate_tokens_and_cost(df: pd.DataFrame) -> tuple[int, float]:
    """Rough token estimate assuming ~4 chars per token."""
    text = df.to_csv(index=False)
    tokens = int(len(text) / 4)
    cost = (tokens / 1000) * EMBED_RATE_PER_1K
    return tokens, cost

# Initialize checkpoint
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"last_processed": -1, "total_processed": 0}


def checkpoint_covers_dataset(checkpoint: dict) -> bool:
    """Return True if the checkpoint already includes every row in the dataset."""

    last_processed = checkpoint.get("last_processed", -1)
    return last_processed >= total_rows - 1


def save_checkpoint(last_processed):
    checkpoint = {
        "last_processed": last_processed,
        "total_processed": last_processed + 1,
        "timestamp": datetime.now().isoformat(),
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)

async def process_chunk(chunk):
    """Process a chunk of data and add to Cognee."""
    # Convert chunk to a meaningful string representation
    chunk_text = chunk.to_string()
    try:
        # Add chunk to Cognee
        await cognee.add(chunk_text)
        return True
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return False

async def setup_cognee(df: pd.DataFrame, force: bool = False):
    """Initialize Cognee and process data in chunks.

    If a checkpoint already covers the full dataset, the run is skipped unless
    ``force`` is True. This keeps existing embeddings intact when you only want
    to query or inspect them.
    """

    checkpoint = load_checkpoint()
    start_idx = checkpoint.get("last_processed", -1) + 1

    total_rows = len(df)

    if checkpoint_covers_dataset(checkpoint) and not force:
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

    for i in tqdm(range(start_idx, total_rows, CHUNK_SIZE), desc="Processing data"):
        chunk = df.iloc[i : i + CHUNK_SIZE]
        success = await process_chunk(chunk)
        if success:
            save_checkpoint(i + len(chunk) - 1)
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
        "--data",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to the cleaned Strong/Whoop CSV (default: data/Strong_Whoop_cleaned_small.csv).",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="Keep only the most recent N months of data (default: 6). Use 0 to disable.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Limit rows after filtering (most recent). Default: 500. Use 0 to disable.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the cost confirmation prompt and proceed.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Optional query text to run after ensuring the checkpoint is ready.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    async def main():
        args = parse_args()

        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        df = filter_recent(df, args.months)
        if args.limit and args.limit > 0:
            df = df.tail(args.limit)

        if df.empty:
            raise ValueError("No rows to embed after filtering. Check your --data/--months/--limit.")

        tokens, cost = estimate_tokens_and_cost(df)
        print(f"Prepared rows: {len(df)}")
        print(f"Estimated tokens: {tokens:,}")
        print(f"Estimated cost (@${EMBED_RATE_PER_1K}/1K tokens): ${cost:.4f}")

        if not args.yes:
            proceed = input("Proceed with embedding? [y/N]: ").strip().lower()
            if proceed not in ("y", "yes"):
                print("Aborting before embedding.")
                return

        await setup_cognee(df=df, force=args.force)

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

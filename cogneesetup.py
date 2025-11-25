import os
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv
import cognee
import pandas as pd
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Configure embedding model
os.environ["EMBEDDING_PROVIDER"] = "openai"
os.environ["EMBEDDING_MODEL"] = "text-embedding-3-small"  # or "text-embedding-3-large" for better quality

# Configuration
CHUNK_SIZE = 10  # Process 10 rows at a time
DATA_PATH = r"C:\Users\mcs22\OneDrive\Desktop\Strong_Whoop_cleaned_small.csv"
CHECKPOINT_FILE = "cognee_checkpoint.json"

# Debugging
print("Cognee LLM key loaded:", "OK" if os.getenv("OPENAI_API_KEY") else "Missing!")

# Load data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
total_rows = len(df)
print(f"Loaded {total_rows} rows")

# Initialize checkpoint
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"last_processed": -1, "total_processed": 0}

def save_checkpoint(last_processed):
    checkpoint = {
        "last_processed": last_processed,
        "total_processed": last_processed + 1,
        "timestamp": datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
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

async def setup_cognee():
    """Initialize Cognee and process data in chunks."""
    # Initialize Cognee
    checkpoint = load_checkpoint()
    start_idx = checkpoint["last_processed"] + 1
    
    if start_idx == 0:
        print("Initializing new Cognee database...")
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    else:
        print(f"Resuming from checkpoint at row {start_idx}")
    
    # Process data in chunks
    for i in tqdm(range(start_idx, total_rows, CHUNK_SIZE), desc="Processing data"):
        chunk = df.iloc[i:i + CHUNK_SIZE]
        success = await process_chunk(chunk)
        if success:
            save_checkpoint(i + len(chunk) - 1)
        else:
            print(f"Stopped at row {i}")
            break
    
    # Finalize with cognify
    print("Finalizing embeddings...")
    await cognee.cognify()
    print("Setup complete!")

async def query_cognee(query: str):
    """Search Cognee's knowledge graph."""
    print(f"Searching for: {query}")
    results = await cognee.search(query_text=query)
    return [r for r in results]

if __name__ == "__main__":
    async def main():
        # Run setup
        await setup_cognee()
        
        # Example query
        while True:
            query = input("\nEnter your query (or 'quit' to exit): ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            results = await query_cognee(query)
            print("\nResults:")
            for i, r in enumerate(results, 1):
                print(f"{i}. {r}")
    
    asyncio.run(main())

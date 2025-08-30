import os
import asyncio
from dotenv import load_dotenv   # <--- load keys from .env
import cognee

# Load environment variables from .env file
load_dotenv()

LLM_API_KEY = os.getenv("OPENAI_API_KEY")

# Debugging: make sure Cognee can see it
print("Cognee LLM key loaded:", "OK" if os.getenv("OPENAI_API_KEY") else "Missing!")


async def setup_cognee():
    """Reset Cognee and add initial content."""
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await cognee.add("Frodo carried the One Ring to Mordor.")
    await cognee.cognify()

async def query_cognee(query: str):
    """Search Cognee’s knowledge graph."""
    results = await cognee.search(query_text=query)
    return [r for r in results]

# Run standalone (optional, for testing Cognee alone)
if __name__ == "__main__":
    async def main():
        await setup_cognee()
        results = await query_cognee("What did Frodo do?")
        for r in results:
            print(r)
    asyncio.run(main())

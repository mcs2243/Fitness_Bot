import os
import asyncio
import pandas as pd
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import cognee
from datetime import datetime, timedelta

# LangChain / LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain.prompts import ChatPromptTemplate
from langsmith import Client
from openai import OpenAI

# Load environment variables
load_dotenv()

# --- Configuration ---
class Config:
    LLM_MODEL = "gpt-4o-mini"
    LLM_TEMPERATURE = 0.7
    MAX_TOKENS = 500
    TIMEOUT = 30
    MAX_RETRIES = 2

# --- Initialize Services ---
def initialize_services():
    """Initialize all required services and clients."""
    # Verify environment variables
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set in environment variables")
    
    # Initialize LangSmith if API key is available
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGSMITH_PROJECT"] = "Fitness_Bot"
        client = Client(
            api_url="https://api.smith.langchain.com",
            api_key=os.getenv("LANGSMITH_API_KEY")
        )
        print("LangSmith tracing: Enabled")
    else:
        print("LangSmith tracing: Disabled (LANGSMITH_API_KEY not found)")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=Config.LLM_MODEL,
        temperature=Config.LLM_TEMPERATURE,
        max_tokens=Config.MAX_TOKENS,
        timeout=Config.TIMEOUT,
        max_retries=Config.MAX_RETRIES,
        "workout data": formatted_results
    })

    # Test the prompt
    response = oai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"{user_input}\n\nWorkout Data:\n{formatted_results}"}],
        max_tokens=500,
        temperature=0.7
    )

    print("\nAnalysis of your workout data:")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    asyncio.run(run())
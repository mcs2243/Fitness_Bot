import os
import pandas as pd
from typing import Literal
import os
from dotenv import load_dotenv

# Load keys from .env
load_dotenv()

# LangChain / LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# --- API Keys / Environment ---
# (in production, store these in a .env file or system environment vars)
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "Fitness_Bot"
# Verify environment variables
print("OpenAI key loaded:", "OK" if os.getenv("OPENAI_API_KEY") else "Missing!")
print("LangSmith tracing enabled:", os.getenv("LANGSMITH_TRACING"))

# --- Example: Initialize a model ---
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=50,
    timeout=20,
    max_retries=2,
)

messages = [
    SystemMessage(content="You're a helpful assistant. Please keep answers short."),
    HumanMessage(content="Are you working?"),
]

response = llm.invoke(messages)

print("LLM Response:", response.content)
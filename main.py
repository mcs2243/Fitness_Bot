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
from langchain.prompts import ChatPromptTemplate
from langsmith import Client
from openai import OpenAI
from langchain_core.messages import convert_to_openai_messages

# --- API Keys / Environment ---
# (in production, store these in a .env file or system environment vars)
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "Fitness_Bot"
# Verify environment variables
print("OpenAI key loaded:", "OK" if os.getenv("OPENAI_API_KEY") else "Missing!")
print("LangSmith tracing enabled:", os.getenv("LANGSMITH_TRACING"))

# Initialize LangSmith client
client = Client(
    api_url="https://api.smith.langchain.com",
    api_key=os.getenv("LANGSMITH_API_KEY")
)

# Initialize your LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=50,
    timeout=20,
    max_retries=2,
)

# Connect to LangSmith and OpenAI
client = Client()
oai_client = OpenAI()

# Pull the prompt to use
# You can also specify a specific commit by passing the commit hash "my-prompt:<commit-hash>"
prompt = client.pull_prompt("short-assistant")

# Since our prompt only has one variable we could also pass in the value directly
# The code below is equivalent to formatted_prompt = prompt.invoke("What is the color of the sky?")
formatted_prompt = prompt.invoke({"user inputs": "What is the color of the sky?"})

# Test the prompt
response = oai_client.chat.completions.create(
    model="gpt-4o",
    messages=convert_to_openai_messages(formatted_prompt.messages),
)
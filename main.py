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

# Fetch the prompt object from LangSmith
prompt_obj = client.get_prompt("short-assistant")  # returns a LangSmith Prompt

# Initialize your LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=50,
    timeout=20,
    max_retries=2,
)

text = prompt_obj.template.format(user_input="Are you working?")
response = llm.invoke([HumanMessage(content=text)])
print("LLM Response:", response.content)
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langsmith import Client, traceable

load_dotenv()

class Config:
    LLM_MODEL = "gpt-4o-mini"
    LLM_TEMPERATURE = 0.7
    MAX_TOKENS = 1000
    TIMEOUT = 30
    MAX_RETRIES = 2
    PROJECT_NAME = "Fitness_Bot"

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

@traceable(name="chroma_retrieval")
def retrieve_chroma(query: str, chroma_dir: str = "chroma_store", collection: str = "fitness_logs", top_k: int = 5) -> str:
    try:
        embeddings = OpenAIEmbeddings()
        vs = Chroma(
            persist_directory=chroma_dir,
            collection_name=collection,
            embedding_function=embeddings,
        )
        docs = vs.similarity_search(query, k=top_k)
        header = "Retrieved prior context (Chroma):"
        body = "\n".join([f"- [{i}] {d.page_content}" for i, d in enumerate(docs, 1)])
        return f"\n\n{header}\n{body}"
    except Exception as exc:
        print(f"Chroma retrieval failed: {exc}. Continuing without retrieval.")
        return ""

def analyze_workout(llm: ChatOpenAI, formatted_data: str, goal: str, retrieval_context: str = "") -> str:
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
            f"{retrieval_context}\n\n"
            "Return a short summary and clear next-step adjustments. Use Markdown formatting."
        )
    )
    response = llm.invoke([system, user])
    return response.content

def chat_with_coach(llm: ChatOpenAI, history: List[Dict[str, str]], context: str) -> str:
    """Chat with the coach using history and context."""
    messages = [
        SystemMessage(content="You are an expert fitness coach. Use the provided workout context to answer the user's questions.")
    ]
    # Add context as a system message or initial user message
    if context:
        messages.append(SystemMessage(content=f"Context from uploaded data:\n{context}"))
        
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            # In a real app we'd use AIMessage, but for simplicity here:
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=msg["content"]))
            
    response = llm.invoke(messages)
    return response.content

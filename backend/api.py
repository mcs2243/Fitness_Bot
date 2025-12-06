from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import os

from backend.data_loader import load_workout_data, format_workout_records, default_sample_workout
from backend.llm_service import init_llm, enable_langsmith_if_configured, analyze_workout, chat_with_coach, retrieve_chroma

app = FastAPI(title="Fitness Bot API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
enable_langsmith_if_configured()
try:
    llm = init_llm()
except Exception as e:
    print(f"Warning: LLM init failed (check API key): {e}")
    llm = None

# In-memory storage for the session (simple version)
# In a real app, use a database or Redis
class SessionState:
    formatted_data: str = ""
    raw_records: List[Dict] = []

session = SessionState()

class AnalysisRequest(BaseModel):
    goal: str = "Build strength and keep fatigue in check"
    months: int = 6
    use_chroma: bool = False

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        records = load_workout_data(content, file.filename)
        session.raw_records = records
        session.formatted_data = format_workout_records(records)
        return {
            "message": f"Successfully loaded {len(records)} records.",
            "record_count": len(records),
            "preview": records[:3]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/data")
def get_data():
    if not session.raw_records:
        # Return sample data if nothing uploaded
        sample = default_sample_workout()
        return {"records": sample, "is_sample": True}
    return {"records": session.raw_records, "is_sample": False}

@app.post("/analyze")
def analyze(req: AnalysisRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not initialized")
    
    data_to_analyze = session.formatted_data
    if not data_to_analyze:
        # Use sample data if empty
        sample = default_sample_workout()
        data_to_analyze = format_workout_records(sample)

    retrieval_context = ""
    if req.use_chroma:
        retrieval_context = retrieve_chroma(req.goal)

    insight = analyze_workout(llm, data_to_analyze, req.goal, retrieval_context)
    return {"insight": insight}

@app.post("/chat")
def chat(req: ChatRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not initialized")
    
    response = chat_with_coach(llm, req.history, session.formatted_data)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)

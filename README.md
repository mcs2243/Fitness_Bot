# Fitness Bot (Agentic Fitness Coach)

An agentic fitness bot built with LangGraph, LangChain, LangSmith, and Cognee. It ingests unstructured workout notes plus structured workout/sleep data to return actionable coaching insights for serious lifters.

## What it does today
- Consolidates workout notes, set/rep/weight logs, and recovery inputs (e.g., Whoop/Apple Watch) to produce GPT-4o-driven analysis.
- Uses Cognee as a vector/RAG backing store for recalling prior sessions and context.
- Runs through a LangGraph orchestration with tool-calling to keep stateful conversations about your training.
- Hooks into LangSmith tracing (project: `Fitness_Bot`) to observe, debug, and improve the agent flows.

## Vision (target state)
“Fitness consolidation” (like Monarch/Rocket Money but for bodybuilding). The goal is an all-in-one AI coach that:
- Unifies data from MyFitnessPal (meals), Whoop/Apple Health (sleep/recovery), daily weight logs, and gym notes.
- Surfaces research-backed, personalized insights (e.g., “Did yesterday’s sleep impact today’s lifts?”).
- Automates programming adjustments and accepts in-workout feedback to tweak sessions on the fly.
- Provides one-click integrations, mobile-first UX, and stretch goals like photo-based physique scans for feedback.

**Target users**: busy 20–40-year-olds who train seriously, track diligently, and want coaching-quality guidance without $200–$250/mo human coaches.

**Ultimate end-state**: a proactive, multimodal coach that continuously analyzes your training, recovery, nutrition, and conversations to adjust programs and diet in real time. It delivers weekly insight summaries with actionable takeaways, critiques progress photos, adapts plans based on form videos and perceived muscle activation, and surfaces general health tips to keep your lifestyle aligned with your goals.

---

## Web Application Features
- **Web Dashboard**: Visualizes workout volume, recovery, and sleep trends.
- **AI Coach Chat**: Interactive chat interface with RAG-based insights.
- **Data Ingestion**: Supports Strong App (CSV) and Whoop (CSV) data.

## Tech Stack
- **Frontend**: React, Vite, TailwindCSS, Recharts.
- **Backend**: Python, FastAPI, LangChain, ChromaDB.

## Prerequisites
- Python 3.10+
- Node.js 18+
- OpenAI API Key

## How to Run

### 1. Backend Setup
The backend serves the API and handles the LLM logic.

1.  Navigate to the project root:
    ```bash
    cd "Fitness Bot"
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Create a `.env` file in the root directory with your API keys:
    ```env
    OPENAI_API_KEY=sk-...
    LANGSMITH_API_KEY=... (optional)
    ```
5.  Start the backend server:
    ```bash
    python -m uvicorn backend.api:app --reload --port 8000
    ```
    The API will be available at `http://localhost:8000`.

### 2. Frontend Setup
The frontend is a React application.

1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```
4.  Open your browser to `http://localhost:5173`.

## Usage
1.  **Upload Data**: On the Dashboard, drag and drop your Strong App or Whoop CSV files.
2.  **View Insights**: Check the charts for volume and recovery trends.
3.  **Chat with Coach**: Switch to the "Coach Chat" tab to ask questions like "How is my squat form?" or "Should I train heavy today?".

## Repo Structure
- `backend/`: FastAPI application and logic.
    - `api.py`: API endpoints.
    - `llm_service.py`: LangChain/OpenAI integration.
    - `data_loader.py`: Data parsing logic.
- `frontend/`: React application.
    - `src/components/`: Dashboard and Chat components.
- `scripts/`: Original data cleaning and ingestion scripts.
- `main.py`: Legacy CLI entry point.

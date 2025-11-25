## Fitness Bot (Agentic Fitness Coach)

An agentic fitness bot built with LangGraph, LangChain, LangSmith, and Cognee RAG DB. It ingests unstructured workout notes plus structured workout, sleep, and nutrition data, then returns actionable coaching insights for serious lifters.

### What it does today
- Consolidates workout notes, set/rep/weight logs, and recovery inputs (e.g., Whoop/Apple Watch) to produce GPT‑4o‑driven analysis.
- Uses Cognee as a vector/RAG backing store for recalling prior sessions and context.
- Runs through a LangGraph orchestration with tool-calling to keep stateful conversations about your training.
- Hooks into LangSmith tracing (project: `Fitness_Bot`) to observe, debug, and improve the agent flows.

### Vision (target state)
This is positioned as “fitness consolidation” (like Monarch/Rocket Money but for bodybuilding). The goal is an all‑in‑one AI coach that:
- Unifies data from MyFitnessPal (meals), Whoop/Apple Health (sleep/recovery), daily weight logs, and gym notes.
- Surfaces research-backed, personalized insights (e.g., “Did yesterday’s sleep impact today’s lifts?”).
- Automates programming adjustments and accepts in-workout feedback to tweak sessions on the fly.
- Provides one-click integrations, mobile-first UX, and stretch goals like photo-based physique scans for feedback.

Target users: busy 20–40-year-olds who train seriously, track diligently, and want coaching-quality guidance without $200–$250/mo human coaches.

### Repo structure
- `main.py` — entry point for the agent graph, OpenAI model config, LangSmith setup, and Cognee integration.
- `cogneesetup.py` — Cognee DB initialization and utility helpers.
- `requirements.txt` — Python dependencies.
- `.env` — example env file (not committed) for API keys; set `OPENAI_API_KEY` (and `LANGSMITH_API_KEY` to enable tracing).

### LangSmith usage
- Tracing is enabled when `LANGSMITH_API_KEY` is set (project: `Fitness_Bot`; endpoint: `https://api.smith.langchain.com`).
- Use LangSmith to inspect runs, tool calls, latency, token usage, and failures while iterating on prompts and graph logic.
- Good for: validating the multi-step LangGraph workflow, comparing prompt tweaks, and spotting failures in RAG retrievals.

### Getting started
1) Create and activate a virtualenv (not tracked): `python -m venv .venv && .\.venv\Scripts\activate`
2) Install deps: `pip install -r requirements.txt`
3) Set environment:
   - `OPENAI_API_KEY=<your key>`
   - Optional tracing: `LANGSMITH_API_KEY=<your key>`
4) Run the bot (example): `python main.py`

### Example interaction (conceptual)
- Input: unstructured workout notes + set/rep data + sleep score from Whoop.
- Output: summarized performance, recovery-adjusted recommendations, and next-session tweaks.

### Roadmap
- Add real connectors for MyFitnessPal, Whoop/Apple Health, and gym log apps (one-click OAuth + background sync).
- Enrich insights with evidence-based content tied to the user’s data.
- Mobile-first UI and on-gym feedback loop to adjust programming mid-session.
- Photo-based physique scans for progress tracking.
- Subscription model: free sync + baseline trends; premium for adaptive plans and coaching-style feedback.

### Monetization hypothesis
- Free: data sync + trend dashboards.
- Premium ($20/mo target): personalized insights, adaptive plans, coaching-style feedback; possible $25 onboarding → 2-month trial.




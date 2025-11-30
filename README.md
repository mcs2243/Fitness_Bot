## Fitness Bot (Agentic Fitness Coach)

An agentic fitness bot built with LangGraph, LangChain, LangSmith, and Cognee. It ingests unstructured workout notes plus structured workout/sleep data to return actionable coaching insights for serious lifters.

### What it does today
- Consolidates workout notes, set/rep/weight logs, and recovery inputs (e.g., Whoop/Apple Watch) to produce GPT-4o-driven analysis.
- Uses Cognee as a vector/RAG backing store for recalling prior sessions and context.
- Runs through a LangGraph orchestration with tool-calling to keep stateful conversations about your training.
- Hooks into LangSmith tracing (project: `Fitness_Bot`) to observe, debug, and improve the agent flows.

### Vision (target state)
“Fitness consolidation” (like Monarch/Rocket Money but for bodybuilding). The goal is an all-in-one AI coach that:
- Unifies data from MyFitnessPal (meals), Whoop/Apple Health (sleep/recovery), daily weight logs, and gym notes.
- Surfaces research-backed, personalized insights (e.g., “Did yesterday’s sleep impact today’s lifts?”).
- Automates programming adjustments and accepts in-workout feedback to tweak sessions on the fly.
- Provides one-click integrations, mobile-first UX, and stretch goals like photo-based physique scans for feedback.

Target users: busy 20–40-year-olds who train seriously, track diligently, and want coaching-quality guidance without $200–$250/mo human coaches.

Ultimate end-state: a proactive, multimodal coach that continuously analyzes your training, recovery, nutrition, and conversations to adjust programs and diet in real time. It delivers weekly insight summaries with actionable takeaways, critiques progress photos, adapts plans based on form videos and perceived muscle activation, and surfaces general health tips to keep your lifestyle aligned with your goals.

### Repo structure
- `main.py` — entry point CLI for analysis, OpenAI config, optional LangSmith tracing, and data loader.
- `scripts/clean_strong_whoop.py` — CLI to merge Strong app exports and Whoop physiological cycles into a cleaned CSV for analysis.
- `cogneesetup.py` — Cognee DB initialization and utility helpers (for embedding/querying, not required for the basic CLI).
- `requirements.txt` — Python dependencies.
- `.env` — set `OPENAI_API_KEY`; optional `LANGSMITH_API_KEY` enables tracing.

### LangSmith usage
- Tracing is enabled when `LANGSMITH_API_KEY` is set (project: `Fitness_Bot`; endpoint: `https://api.smith.langchain.com`).
- Use LangSmith to inspect runs, tool calls, latency, token usage, and failures while iterating on prompts and graph logic.

### Getting started
1) Create/activate venv: `python -m venv .venv && .\.venv\Scripts\activate`
2) Install deps: `pip install -r requirements.txt`
3) Set env: `OPENAI_API_KEY=<your key>` (optional `LANGSMITH_API_KEY=<your key>`)
4) Run analysis (with sample data fallback): `python main.py`
   - Or point to your cleaned CSV: `python main.py --data data/Strong_Whoop_cleaned_small.csv --limit 50 --months 6 --goal "Build strength with good recovery"`

### Data preparation (Strong + Whoop)
- Raw data is **not** in the repo. Provide your own CSVs:
  - Strong export (workout logs) as CSV.
  - Whoop “physiological cycles” export as CSV.
- Clean/merge via the script (prompts if paths are omitted):
  - Non-interactive: `python scripts/clean_strong_whoop.py --strong-path <path_to_strong.csv> --whoop-path <path_to_whoop.csv> --output data/Strong_Whoop_cleaned_small.csv --limit 500`
  - Interactive (just run and follow prompts): `python scripts/clean_strong_whoop.py`
- The cleaned file is ignored by Git (`data/` is in `.gitignore`). Point `main.py` to your cleaned CSV with `--data` or place it at `data/Strong_Whoop_cleaned_small.csv`.

### Running the analyzer
- With the cleaned CSV: `python main.py --data data/Strong_Whoop_cleaned_small.csv --limit 50 --months 6 --goal "Your goal"`
- `--months` filters to recent data (default 6). `--limit` keeps prompts small; both can be adjusted.
- If `--data` is omitted and no default file exists, the script falls back to a small embedded sample.

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
- Premium ($20/mo target): personalized insights, adaptive plans, coaching-style feedback; possible $25 onboarding + 2-month trial.

# personalOPS

A multi-agent LangGraph system for managing personal ops — gym tracking and study planning.

## Architecture
- **Supervisor agent** — routes requests to the right sub-agent
- **Gym Coach sub-agent** — RAG over personal workout logs + fitness knowledge base
- **Study Planner sub-agent** — RAG over personal notes and syllabi
- **Persistent, checkpointed state** — conversations resume across sessions

## Setup
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env  # then fill in your API keys
```

## Status
🚧 In progress — see commit history for build order.

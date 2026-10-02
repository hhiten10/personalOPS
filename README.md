# personalOPS

A multi-agent assistant for tracking workouts and planning study sessions, built with **LangGraph** and **Gemini**.

Tell it what you did ("Did 5x5 squats at 100kg today") and it saves the entry to a local vector store. Ask about your history ("How has my squat progressed?") and it answers from your own logs. It can also check your Google Calendar when planning what to do next.

## Features

- **Natural-language logging.** Workouts and study sessions are pulled out of normal chat messages and saved with a date. You don't fill in any forms.
- **RAG over your own data.** Answers come from your workout logs and study notes, stored in a local Chroma database.
- **Multi-intent routing.** One message can do several things. "I did legs today, what should I do tomorrow?" logs the workout *and* gives a recommendation in the same turn.
- **Calendar-aware.** It can read your upcoming Google Calendar events when planning. Access is read-only, so it can never change your calendar.
- **Persistent memory.** Conversations are saved to SQLite and pick up where you left off after a restart.
- **Two interfaces.** There's a terminal chat loop, and a Streamlit web UI that supports multiple conversations.
- **Free, local embeddings.** Embeddings come from `all-MiniLM-L6-v2` running on your CPU, so there are no embedding API costs.

## Architecture

```
                    ┌──────────────┐
      user ───────▶ │  supervisor  │  Gemini + structured output decides
                    └──────┬───────┘  which agent(s) to run
                           │ parallel fan-out (LangGraph Send)
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌────────────┐  ┌───────────────┐  ┌────────────┐
   │ gym_coach  │  │ study_planner │  │ log_writer │
   │ (retrieve) │  │  (retrieve)   │  │  (save)    │
   └─────┬──────┘  └───────┬───────┘  └─────┬──────┘
         └─────────────────┼────────────────┘
                           ▼
                    ┌──────────────┐
                    │ synthesizer  │  writes the single reply,
                    └──────┬───────┘  and can call the Calendar tool
                           ▼
                          END
```

| Node | Role | LLM calls |
|---|---|---|
| `supervisor` | Routes each turn to `gym_coach`, `study_planner`, `log_gym`, `log_study`, any combination of them, or none | 1 (`gemini-2.5-flash`) |
| `gym_coach` / `study_planner` | Search the matching Chroma collection for relevant entries | 0 |
| `log_writer` | Pulls the `{content, date}` out of the message and saves it to the right collection | 1 (`gemini-2.5-flash-lite`) |
| `synthesizer` | Combines everything the other agents found and writes the reply. Can call `get_upcoming_events` up to 3 times | 1+ (`gemini-2.5-flash`) |

The specialist agents only collect context. The synthesizer is the only node that writes a reply to you, so answers stay consistent even when several agents run in the same turn.

## Project structure

```
personalOPS/
├── main.py              # CLI chat loop
├── streamlit_app.py     # Web UI with multiple conversations
├── agents/              # gym_coach, study_planner, log_writer, synthesizer
├── graph/               # state, supervisor, graph assembly
├── rag/                 # ingest.py (writes) and retriever.py (searches)
├── memory/              # SQLite checkpointer
├── tools/               # Google Calendar tool (read-only)
├── tests/               # pytest suite
└── data/                # Chroma DB, checkpoints, optional log files (git-ignored)
```

## Getting started

### 1. Install

```bash
git clone https://github.com/hhiten10/personalOPS.git
cd personalOPS
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 2. Add your API key

```bash
cp .env.example .env
```

Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) and put it in `.env`:

```
GOOGLE_API_KEY=your-gemini-api-key-here
```

### 3. (Optional) Set up Google Calendar

1. In [Google Cloud Console](https://console.cloud.google.com/), enable the **Google Calendar API**.
2. Create an **OAuth client ID** of type *Desktop app* and download the JSON file.
3. Save it as `tools/credentials.json`.

The first time the assistant checks your calendar, a browser window opens for you to sign in. After that, the token is saved to `tools/token.json` and refreshes automatically. Both files are git-ignored.

### 4. Run

```bash
python main.py                  # terminal chat
streamlit run streamlit_app.py  # web UI
```

## Example conversation

```
You: Did push day — bench 4x8 at 70kg, OHP 3x10 at 40kg
Agent: Logged your push session for today. ...

You: Studied DBMS normalization for 2 hours, finished 3NF and BCNF
Agent: Saved your study session. ...

You: How's my bench progressing, and what's on my calendar this week?
Agent: ...
```

## Bulk import (optional)

To load existing notes instead of logging through chat, put `.txt` files in `data/workout_logs/` or `data/study_notes/`. If a filename contains a date (e.g. `2026-09-14_legs.txt`), that date is used. Otherwise the file is dated today.

```bash
python -m rag.ingest            # import the files
python -m rag.ingest --clear    # wipe both collections
```

## Tests

```bash
pytest
```

## Tech stack

[LangGraph](https://github.com/langchain-ai/langgraph) · [LangChain](https://github.com/langchain-ai/langchain) · Google Gemini · [Chroma](https://www.trychroma.com/) · HuggingFace sentence-transformers · SQLite · Streamlit · Google Calendar API

## Privacy

Everything except the Gemini API calls runs on your machine. Your logs, notes, vector store and chat history stay in `data/`, which is git-ignored.

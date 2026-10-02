# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

personalOPS is a multi-agent LangGraph system for personal ops (gym tracking + study
planning), run as a CLI chat loop backed by a persistent SQLite checkpointer. It is a
single-user, local-first app — no server, no auth beyond the Google Calendar OAuth flow.

## Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate          # Windows (this repo is developed on Windows)
pip install -r requirements.txt
cp .env.example .env           # then fill in API keys — see "Env vars" below

# Run the chat loop
python main.py

# Tests (no pytest.ini — plain discovery of tests/test_*.py)
pytest
pytest tests/test_graph.py -k test_graph_builds   # single test

# RAG ingestion
python -m rag.ingest            # batch-loads data/workout_logs/ and data/study_notes/
python -m rag.ingest --clear    # wipes both Chroma collections (gym_logs, study_notes)
```

### `requirements.txt` is incomplete

The file lists `langchain-openai` but the code actually runs on Gemini via
`langchain_google_genai`, plus `langchain_huggingface` for local embeddings and the
Google Calendar client libs. If you're setting up a fresh environment, also install:
`langchain-google-genai`, `langchain-huggingface`, `google-auth-oauthlib`,
`google-api-python-client`. Update `requirements.txt` if you touch dependencies.

### Env vars

`main.py` requires `GOOGLE_API_KEY` or `GEMINI_API_KEY` — note `.env.example` only
documents `OPENAI_API_KEY`, which is stale/unused by the current code path.

## Architecture

### Graph shape (`graph/build_graph.py`)

```
supervisor --(Send fan-out)--> {gym_coach, study_planner, log_writer} --> synthesizer --> END
```

- **`supervisor`** (`graph/supervisor.py`) is the only routing decision point. It reads
  the conversation and, via structured output (Gemini + pydantic `Routing` model),
  produces a list of `next_agent` targets: `gym_coach`, `study_planner`, `log_gym`,
  `log_study`, or an empty list. Multiple targets can be picked in the same turn (e.g.
  "I did legs today, what should I do tomorrow?" → `["log_gym", "gym_coach"]`).
- **`route_from_supervisor`** turns that list into parallel `Send` branches. `log_gym`
  and `log_study` both route to the *same* `log_writer` node — the node itself
  disambiguates using `next_agent`.
- **`gym_coach`** / **`study_planner`** (`agents/*.py`) do retrieval only — zero LLM
  calls. Each does a Chroma similarity search (`rag/retriever.py`) and tags results
  into `agent_scratchpad`.
- **`log_writer`** (`agents/log_writer.py`) extracts structured `{content, date}` from
  the user's message via a small Gemini call, then writes it into the appropriate
  Chroma collection through `rag/ingest.py:add_entry()`, and drops a confirmation chunk
  into the scratchpad.
- **`synthesizer`** (`agents/synthesizer.py`) is the single place a user-facing reply is
  generated. It reads everything in `agent_scratchpad` this turn, groups it by
  contributing agent, and prompts Gemini with the combined context. It also has access
  to `get_upcoming_events` (a read-only Google Calendar tool) via `bind_tools`, with a
  manual tool-call loop capped at `MAX_TOOL_ROUNDS = 3`. All branches converge here
  before `END` — there is no per-agent answer generation anywhere else.

### State (`graph/state.py`)

`PersonalOpsState` has three fields:
- `messages` — standard `add_messages`-reduced chat history.
- `next_agent` — routing targets set by the supervisor each turn.
- `agent_scratchpad` — accumulates `RetrievedChunk` dicts (`{agent, source, content}`)
  via a custom reducer. The supervisor clears it at the start of every turn by
  returning the sentinel `"__CLEAR__"` instead of a list — this is the mechanism that
  keeps scratchpad content scoped to a single turn even though nodes run in parallel
  and can't see each other's writes mid-turn.

### RAG (`rag/`)

Two Chroma collections, `gym_logs` and `study_notes`, persisted at `./data/chroma`
using a local HuggingFace embedding model (`all-MiniLM-L6-v2`, CPU, no API cost — this
is deliberate, not a placeholder). `rag/ingest.py` and `rag/retriever.py` each
instantiate their own module-level `_embeddings` / `_chroma_client` — both point at the
same `PERSIST_DIR`/`EMBEDDING_MODEL`, so keep them in sync if either changes.
`add_entry()` (chat-driven, single entry) is the primary write path; `ingest_documents()`
(batch file import from `data/workout_logs/` and `data/study_notes/`) is a fallback CLI
path, not used by the running graph.

### Persistence (`memory/checkpointer.py`)

`SqliteSaver` at `./data/checkpoints.sqlite3`, used as a context manager in `main.py`.
`main.py` currently hardcodes `THREAD_ID = "default-session"`, so all CLI runs resume
the same conversation thread.

### Calendar tool (`tools/calendar_tool.py`)

Read-only Google Calendar integration (`calendar.readonly` scope only — cannot
create/edit/delete events by design). Handles OAuth: first run opens a browser flow via
`credentials.json`, then caches to `token.json` (both in `tools/`) for silent refresh on
subsequent runs. Exposed to the graph as a single LangChain `@tool`,
`get_upcoming_events`, consumed only by the synthesizer.

"""
Log-writer node: extracts structured data from the user's message
and persists it into the appropriate Chroma collection via add_entry().
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from graph.state import PersonalOpsState, RetrievedChunk
from rag.ingest import add_entry

# Maps routing target → (collection_name, doc_type)
_ROUTE_MAP = {
    "log_gym": ("gym_logs", "workout_log"),
    "log_study": ("study_notes", "study_note"),
}

EXTRACT_PROMPT = """Extract the following from the user's message:

1. content: The actual information to log. Keep all details (exercises,
   sets, reps, weights, topics studied, chapters, durations, etc.).
   Clean it up slightly but do NOT remove any facts.
2. date: The date this happened in YYYY-MM-DD format.
   - If the user says "today", use {today}.
   - If the user says "yesterday", use the day before {today}.
   - If a specific date is mentioned, use that.
   - If no date is mentioned at all, assume today: {today}.

Respond with ONLY the extracted fields, nothing else."""


class ExtractedEntry(BaseModel):
    content: str = Field(description="The information to log, cleaned up")
    date: str = Field(description="YYYY-MM-DD date of the entry")


def _make_extractor():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    return llm.with_structured_output(ExtractedEntry)


def log_writer_node(state: PersonalOpsState) -> dict:
    """Parse the user message, extract content + date, and persist."""
    from datetime import date as date_type
    today = date_type.today().isoformat()

    # Determine which collection to write to
    targets = state.get("next_agent", [])
    log_targets = [t for t in targets if t in _ROUTE_MAP]

    if not log_targets:
        return {"agent_scratchpad": []}

    # Extract structured data from the latest user message
    user_message = state["messages"][-1].content
    extractor = _make_extractor()
    prompt = EXTRACT_PROMPT.format(today=today)
    result = extractor.invoke([
        SystemMessage(content=prompt),
        state["messages"][-1],
    ])

    confirmations = []

    for target in log_targets:
        collection_name, doc_type = _ROUTE_MAP[target]

        add_entry(
            collection_name=collection_name,
            doc_type=doc_type,
            text=result.content,
            date=result.date,
            source="chat_entry",
        )

        confirmations.append(
            RetrievedChunk(
                agent=target,
                source=collection_name,
                content=f"LOGGED to {collection_name}: \"{result.content}\" (date: {result.date})",
            )
        )

    return {"agent_scratchpad": confirmations}
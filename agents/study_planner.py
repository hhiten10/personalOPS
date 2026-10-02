"""
Study Planner sub-agent.

Retrieves relevant notes/syllabus context and tags it into the shared
scratchpad. Answer generation happens once, centrally, in the
synthesizer node — this node's only job is retrieval, so it costs
zero LLM calls.
"""

from graph.state import PersonalOpsState
from rag.retriever import retrieve


def study_planner_node(state: PersonalOpsState) -> dict:
    last_user_msg = state["messages"][-1].content

    chunks = retrieve(last_user_msg, collection_name="study_notes", k=4)

    tagged_chunks = [
        {"agent": "study_planner", "source": "study_notes", "content": chunk}
        for chunk in chunks
    ]

    return {"agent_scratchpad": tagged_chunks}
"""
Gym Coach sub-agent.

Retrieves relevant workout-log context and tags it into the shared
scratchpad. Answer generation happens once, centrally, in the
synthesizer node — this node's only job is retrieval, so it costs
zero LLM calls.
"""

from graph.state import PersonalOpsState
from rag.retriever import retrieve


def gym_coach_node(state: PersonalOpsState) -> dict:
    last_user_msg = state["messages"][-1].content

    chunks = retrieve(last_user_msg, collection_name="gym_logs", k=4)

    tagged_chunks = [
        {"agent": "gym_coach", "source": "gym_logs", "content": chunk}
        for chunk in chunks
    ]

    return {"agent_scratchpad": tagged_chunks}
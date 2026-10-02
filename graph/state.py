from typing import Annotated, List, Literal, Union
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


# All possible routing targets
AgentName = Literal["gym_coach", "study_planner", "log_gym", "log_study"]


class RetrievedChunk(TypedDict):
    agent: str
    source: str
    content: str


def scratchpad_reducer(
    left: List[RetrievedChunk],
    right: Union[List[RetrievedChunk], Literal["__CLEAR__"]],
) -> List[RetrievedChunk]:
    if right == "__CLEAR__":
        return []
    return left + right


class PersonalOpsState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: List[AgentName]
    agent_scratchpad: Annotated[List[RetrievedChunk], scratchpad_reducer]
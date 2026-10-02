"""
Builds the personalOPS graph.
"""

from typing import List, Union

from langgraph.graph import StateGraph, END
from langgraph.types import Send

from graph.state import PersonalOpsState
from graph.supervisor import make_supervisor_node
from agents.synthesizer import synthesizer_node
from agents.gym_coach import gym_coach_node
from agents.study_planner import study_planner_node
from agents.log_writer import log_writer_node


def route_from_supervisor(state: PersonalOpsState) -> Union[str, List[Send]]:
    next_agent = state.get("next_agent") or []
    if not next_agent:
        return END

    sends = []
    for agent_name in next_agent:
        # Both log targets share the same node
        if agent_name in ("log_gym", "log_study"):
            sends.append(Send("log_writer", state))
        else:
            sends.append(Send(agent_name, state))
    return sends


def build_graph(checkpointer=None):
    builder = StateGraph(PersonalOpsState)

    builder.add_node("supervisor", make_supervisor_node())
    builder.add_node("gym_coach", gym_coach_node)
    builder.add_node("study_planner", study_planner_node)
    builder.add_node("log_writer", log_writer_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.set_entry_point("supervisor")

    builder.add_conditional_edges("supervisor", route_from_supervisor)

    builder.add_edge("gym_coach", "synthesizer")
    builder.add_edge("study_planner", "synthesizer")
    builder.add_edge("log_writer", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile(checkpointer=checkpointer)
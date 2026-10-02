"""
Supervisor node: looks at the latest user message and decides which
specialist(s) should handle it this turn.
"""

from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from graph.state import PersonalOpsState, AgentName

SUPERVISOR_PROMPT = """You are the supervisor of a personal-productivity
multi-agent system. Given the conversation so far, decide which
specialist(s) should act next.

Available workers:
- gym_coach: QUERY workout history, exercise stats, fitness Q&A.
- study_planner: QUERY study schedules, syllabus tracking, notes Q&A.
- log_gym: The user wants to SAVE/LOG a new workout or exercise entry.
- log_study: The user wants to SAVE/LOG new study notes or a study session.

ROUTING RULES:
- If the user is ASKING a question about past data → gym_coach / study_planner
- If the user is TELLING you about something they did and wants it saved → log_gym / log_study
- Pick BOTH query agents if the question spans both domains.
- You can combine a log and a query (e.g. "I did legs today, what should I do tomorrow?" → ["log_gym", "gym_coach"])
- Pick an empty list if the request needs no specialist at all."""


class Routing(BaseModel):
    next_agent: List[AgentName] = Field(
        description="Which worker(s) should act next. Empty list if neither is needed."
    )


def make_supervisor_node():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    router = llm.with_structured_output(Routing)

    def supervisor_node(state: PersonalOpsState) -> dict:
        messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
        result = router.invoke(messages)
        return {
            "next_agent": result.next_agent,
            "agent_scratchpad": "__CLEAR__",
        }

    return supervisor_node
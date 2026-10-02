"""
Synthesizer node: the single place a final, user-facing answer gets
generated. Reads whatever gym_coach and/or study_planner retrieved
into agent_scratchpad this turn, optionally consults the read-only
Google Calendar tool, and writes one coherent reply.
"""

from typing import Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from graph.state import PersonalOpsState, RetrievedChunk
from tools.calendar_tool import get_upcoming_events

# Tools available to the synthesizer's LLM. Keep this list explicit —
# adding a new tool later just means appending it here and to this dict.
AVAILABLE_TOOLS = [get_upcoming_events]
TOOLS_BY_NAME = {t.name: t for t in AVAILABLE_TOOLS}

MAX_TOOL_ROUNDS = 3  # safety cap; prevents any pathological tool-call loop


def _as_text(content) -> str:
    """
    Normalize an LLM response's .content into plain display text.

    Gemini 2.5 models sometimes return content as a list of blocks
    (e.g. [{'type': 'text', 'text': '...', 'extras': {'signature': ...}}])
    instead of a plain string -- 'extras' carries an internal thought
    signature that's only needed to feed the response back into the next
    `invoke()` call within the same tool-calling round-trip, not for the
    final answer we hand to the user.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def synthesizer_node(state: PersonalOpsState) -> dict:
    scratchpad = state.get("agent_scratchpad", [])

    if not scratchpad:
        return {"messages": [AIMessage(content="I didn't find anything relevant to answer that.")]}

    by_agent: Dict[str, List[RetrievedChunk]] = {}
    for chunk in scratchpad:
        by_agent.setdefault(chunk["agent"], []).append(chunk)

    context_blocks = []
    for agent_name, chunks in by_agent.items():
        lines = [f"[{agent_name} — {c['source']}]\n{c['content']}" for c in chunks]
        context_blocks.append("\n\n".join(lines))
    context_text = "\n\n---\n\n".join(context_blocks)

    last_user_message = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    prompt = f"""You are a personal assistant combining information from one or more \
specialist agents (gym coaching, study planning) to answer the user's question.

User's question: {last_user_message}

Retrieved context:
{context_text}

You also have access to a get_upcoming_events tool that reads the user's real \
Google Calendar (read-only). Call it when the question involves scheduling, free \
time, or planning around existing commitments — don't call it for questions that \
don't need calendar awareness.

Instructions:
- The context contains raw facts (e.g. workout logs, study topics, deadlines) — it \
will rarely contain a pre-written answer to the user's exact question. Your job is \
to reason over these facts and construct a helpful, specific answer, including \
plans, recommendations, or advice when asked.
- Only say the context doesn't cover something if it's missing the underlying facts \
needed to reason about the question (e.g. no workout data at all when asked about \
gym schedule) — not merely because it lacks an explicit answer or conclusion.
- If context came from multiple specialists, weave it into a single coherent \
response rather than listing each specialist's answer separately.
- Be direct and concrete. Prefer specific suggestions over generic advice.."""

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    llm_with_tools = llm.bind_tools(AVAILABLE_TOOLS)

    messages = [HumanMessage(content=prompt)]

    for _ in range(MAX_TOOL_ROUNDS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # No tool requested — this is the final answer.
            return {"messages": [AIMessage(content=_as_text(response.content))]}

        # Model asked for one or more tools. Run each, append results, loop again.
        for call in response.tool_calls:
            tool_fn = TOOLS_BY_NAME.get(call["name"])
            if tool_fn is None:
                result = f"Error: unknown tool '{call['name']}'"
            else:
                try:
                    result = tool_fn.invoke(call["args"])
                except Exception as e:
                    result = f"Error running {call['name']}: {e}"

            messages.append(
                ToolMessage(content=str(result), tool_call_id=call["id"])
            )

    # Safety fallback if we somehow exhaust MAX_TOOL_ROUNDS without a final answer
    return {
        "messages": [
            AIMessage(content="I ran into trouble pulling that information together — try rephrasing your question.")
        ]
    }
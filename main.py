"""
Entry point for personalOPS.

Runs the compiled LangGraph agent in a simple CLI loop, backed by a
persistent SQLite checkpointer so state survives across separate runs
of this script. Later this can be swapped for a Streamlit UI or API server.
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from graph.build_graph import build_graph
from memory.checkpointer import get_checkpointer
from utils import get_logger

load_dotenv()
logger = get_logger(__name__)

THREAD_ID = "default-session"  # used by checkpointer to persist state


def main():
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your .env file before running.")

    os.makedirs("./data", exist_ok=True)

    with get_checkpointer() as checkpointer:
        app = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": THREAD_ID}}

        print("personalOPS ready. Type 'exit' to quit.\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input:
                continue

            logger.info("User turn: %s", user_input)

            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )

            print("Agent:", result["messages"][-1].content, "\n")


if __name__ == "__main__":
    main()
"""
Streamlit UI for personalOPS.

Same compiled graph and SQLite checkpointer as main.py's CLI loop, just
with a browser chat interface and proper multi-thread support (the CLI
hardcodes a single "default-session" thread; here you can create and
switch between separate conversations).

Run with:
    streamlit run streamlit_app.py
"""

import os
import sqlite3
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from graph.build_graph import build_graph
from memory.checkpointer import DB_PATH, get_persistent_checkpointer

load_dotenv()

st.set_page_config(page_title="personalOPS", page_icon="🗂️")


@st.cache_resource
def get_app():
    """
    Build the graph once per Streamlit process, backed by a checkpointer
    connection that stays open for the life of the process (see
    memory.checkpointer.get_persistent_checkpointer for why that can't be
    managed from this file's own globals).
    """
    checkpointer = get_persistent_checkpointer()
    return build_graph(checkpointer=checkpointer)


def list_threads() -> list[str]:
    """Distinct thread_ids already present in the checkpoint DB."""
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        rows = conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
        return sorted(r[0] for r in rows)
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def render_history(app, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = app.get_state(config)
    messages = state.values.get("messages", []) if state.values else []

    for message in messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage) and message.content:
            with st.chat_message("assistant"):
                st.markdown(message.content)


def main():
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        st.error("Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your .env file before running.")
        st.stop()

    os.makedirs("./data", exist_ok=True)
    app = get_app()

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "default-session"

    with st.sidebar:
        st.header("Conversations")

        threads = list_threads()
        if st.session_state.thread_id not in threads:
            threads = [st.session_state.thread_id] + threads

        selected = st.selectbox(
            "Thread",
            options=threads,
            index=threads.index(st.session_state.thread_id),
        )
        if selected != st.session_state.thread_id:
            st.session_state.thread_id = selected
            st.rerun()

        if st.button("New conversation"):
            st.session_state.thread_id = f"session-{uuid.uuid4().hex[:8]}"
            st.rerun()

    st.title("personalOPS")

    render_history(app, st.session_state.thread_id)

    user_input = st.chat_input("Ask about your workouts, studies, or log something new...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = app.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                )
                reply = result["messages"][-1].content
            st.markdown(reply)


if __name__ == "__main__":
    main()

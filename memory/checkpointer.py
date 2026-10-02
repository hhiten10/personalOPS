"""
Sets up the checkpointer that gives the graph persistent, resumable state.

Starting with SQLite (simple, file-based, good for a resume demo).
Can swap for Postgres later if you want a "production" story.
"""

from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = "./data/checkpoints.sqlite3"


def get_checkpointer():
    """
    Returns a context-manager-compatible SqliteSaver pointed at a local
    .sqlite3 file. Use with `with get_checkpointer() as checkpointer:`
    in main.py so the connection closes cleanly on exit.
    """
    return SqliteSaver.from_conn_string(DB_PATH)


_checkpointer_cm = None
_checkpointer = None


def get_persistent_checkpointer():
    """
    Enters get_checkpointer() once per process and keeps both the CM and
    the checkpointer it yields alive for the process's lifetime, via this
    module's globals.

    This is for callers like streamlit_app.py that can't use a `with`
    block because there's no single point where the app "exits" -- but
    critically, it also can't rely on its OWN module-level globals for
    this, because Streamlit reruns the whole script by exec'ing it into
    a brand-new module object every time (see
    streamlit.runtime.scriptrunner.script_runner._new_module) -- so any
    `X = None` at that script's top level resets on every rerun no
    matter what. This module, by contrast, is reached via a normal
    `import`, which Python caches in sys.modules and does not re-exec,
    so the globals here genuinely persist for the life of the process.
    """
    global _checkpointer_cm, _checkpointer
    if _checkpointer is None:
        _checkpointer_cm = get_checkpointer()
        _checkpointer = _checkpointer_cm.__enter__()
    return _checkpointer

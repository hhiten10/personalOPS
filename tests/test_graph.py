"""
Basic smoke tests for the compiled graph.
Expand as nodes get implemented.
"""


def test_graph_builds():
    """Sanity check: build_graph() should compile without error."""
    from graph.build_graph import build_graph
    app = build_graph()
    assert app is not None

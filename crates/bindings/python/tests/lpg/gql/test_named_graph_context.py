"""Named-graph context mutations preserve their structured error channel."""

import pytest


def test_drop_graph_reports_rejected_schema_default_without_changing_context(db):
    db.execute("CREATE SCHEMA guarded")
    db.set_schema("guarded")
    assert db.create_graph("guarded/kept")
    db.set_graph("kept")

    with pytest.raises(RuntimeError, match="default partition"):
        db.drop_graph("guarded/__default__")

    assert db.current_graph() == "kept"
    assert db.current_schema() == "guarded"
    assert "guarded/kept" in db.list_graphs()
    assert db.drop_graph("guarded/kept") is True
    assert db.current_graph() is None
    assert db.drop_graph("guarded/kept") is False

"""Tests for GrafeoError: structured error codes on the Python side (L24).

`GrafeoError` is a subclass of `RuntimeError` so legacy `except RuntimeError:`
paths keep working. New code can catch `GrafeoError` and inspect
`e.error_code` (string, e.g. "GRAFEO-Q001") and `e.is_retryable` (bool).
"""

import grafeo
import pytest


def test_grafeo_error_is_runtime_error_subclass():
    assert issubclass(grafeo.GrafeoError, RuntimeError)


def test_parse_error_raises_grafeo_error_with_query_code():
    db = grafeo.GrafeoDB()
    with pytest.raises(grafeo.GrafeoError) as exc:
        db.execute("THIS IS NOT VALID GQL")
    err = exc.value
    # Legacy callers can still catch RuntimeError
    assert isinstance(err, RuntimeError)
    # New callers can inspect structured attributes
    assert err.error_code.startswith("GRAFEO-Q")
    assert err.is_retryable is False


def test_semantic_error_raises_grafeo_error():
    db = grafeo.GrafeoDB()
    with pytest.raises(grafeo.GrafeoError) as exc:
        db.execute("SESSION SET SCHEMA nonexistent_schema_xyz")
    err = exc.value
    # Error code present, stable prefix
    assert err.error_code.startswith("GRAFEO-")
    assert isinstance(err.is_retryable, bool)


def test_legacy_runtime_error_catch_still_works():
    db = grafeo.GrafeoDB()
    with pytest.raises(RuntimeError):
        db.execute("NOT VALID")


@pytest.mark.parametrize(
    ("entity", "error_code"),
    [("node", "GRAFEO-V002"), ("edge", "GRAFEO-V003")],
)
def test_property_setter_rejects_missing_entity_without_mutation(entity, error_code):
    db = grafeo.GrafeoDB()
    node = db.create_node(["N"], {"score": 1})
    other = db.create_node(["N"])
    edge = db.create_edge(node.id, other.id, "R", {"score": 2})

    with pytest.raises(grafeo.GrafeoError) as exc:
        getattr(db, f"set_{entity}_property")(99_999, "score", 42)

    assert exc.value.error_code == error_code
    assert exc.value.is_retryable is False
    assert db.node_count == 2
    assert db.edge_count == 1
    assert db.get_node(99_999) is None
    assert db.get_edge(99_999) is None
    assert db.get_node(node.id).properties() == {"score": 1}
    assert db.get_edge(edge.id).properties() == {"score": 2}

    db.set_node_property(node.id, "score", 3)
    db.set_edge_property(edge.id, "score", 4)
    assert db.get_node(node.id).properties() == {"score": 3}
    assert db.get_edge(edge.id).properties() == {"score": 4}
    db.close()


@pytest.mark.parametrize("entity", ["node", "edge"])
@pytest.mark.skipif(
    not hasattr(grafeo, "NetworkXAdapter"), reason="algos feature not enabled"
)
def test_networkx_import_propagates_rejected_property(entity):
    nx = pytest.importorskip("networkx")
    graph = nx.DiGraph()
    graph.add_node("source", score=1)
    graph.add_node("target", score=2)
    graph.add_edge("source", "target", weight=3)
    attrs = graph.nodes["source"] if entity == "node" else graph.edges["source", "target"]
    attrs["oversized"] = "x" * (17 * 1024 * 1024)

    with pytest.raises(grafeo.GrafeoError) as exc:
        grafeo.NetworkXAdapter.from_networkx(graph)
    assert exc.value.error_code == "GRAFEO-Q006"
    assert "exceeds maximum size" in str(exc.value)
    assert exc.value.is_retryable is False

    del attrs["oversized"]
    adapter, mapping = grafeo.NetworkXAdapter.from_networkx(graph)
    assert set(mapping.values()) == {"source", "target"}
    restored = adapter.to_networkx()
    assert restored.nodes["source"]["score"] == 1
    assert restored.nodes["target"]["score"] == 2
    assert restored.edges["source", "target"]["weight"] == 3

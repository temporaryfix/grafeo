"""GQL change data capture (CDC) integration tests."""

import os
import shutil
from pathlib import Path

import pytest

from tests.fixtures.cdc import collect_history

try:
    from grafeo import GrafeoDB

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False


@pytest.fixture
def db():
    if not GRAFEO_AVAILABLE:
        pytest.skip("grafeo not installed")
    return GrafeoDB(cdc=True)


class TestCDC:
    def test_creation_responses_match_stored_null_property_absence(self, db):
        node = db.create_node(["N"], {"absent": None, "present": 1})
        target = db.create_node(["N"])
        edge = db.create_edge(node.id, target.id, "R", {"absent": None, "present": 2})
        assert node.get("absent") is None
        assert edge.get("absent") is None
        assert db.get_node(node.id).get("absent") is None
        assert db.get_edge(edge.id).get("absent") is None
        assert node.get("present").as_int() == 1
        assert edge.get("present").as_int() == 2

    def test_node_history_after_create(self, db):
        node = db.create_node(["Person"], {"name": "Alix"})
        history = collect_history(db, node.id)
        assert len(history) >= 1
        assert all(event["lpg_graph"] == [] for event in history)
        assert all(event["triple_graph"] is None for event in history)
        assert history[0]["labels"] == ["Person"]

    def test_named_lpg_coordinate_is_a_component_array(self, db):
        db.execute("CREATE GRAPH scoped")
        db.execute("USE GRAPH scoped")
        node = db.create_node(["Person"], {"name": "Scoped"})
        history = collect_history(db, node.id)
        assert history
        assert all(event["lpg_graph"] == ["scoped"] for event in history)
        assert all(event["triple_graph"] is None for event in history)

    def test_node_history_after_update(self, db):
        node = db.create_node(["Person"], {"name": "Alix"})
        db.set_node_property(node.id, "age", 30)
        history = collect_history(db, node.id)
        assert len(history) >= 2

    def test_named_creation_readback_does_not_alias_root_ids(self, db):
        root = db.create_node(["Root"], {"name": "Root"})
        root_target = db.create_node(["RootTarget"])
        root_edge = db.create_edge(root.id, root_target.id, "ROOT_EDGE", {"weight": 1})
        db.execute("CREATE GRAPH scoped")
        db.execute("USE GRAPH scoped")

        named = db.create_node(["Named", "Named"], {"name": "Scoped"})
        named_target = db.create_node(["NamedTarget"])
        named_edge = db.create_edge(named.id, named_target.id, "NAMED_EDGE", {"weight": 2})
        empty_edge = db.create_edge(named.id, named_target.id, "NO_PROPERTIES")
        assert named.id == root.id
        assert named_target.id == root_target.id
        assert named_edge.id == root_edge.id
        assert named.labels == ["Named"]
        assert named.get("name").as_str() == "Scoped"
        assert named_target.labels == ["NamedTarget"]
        assert named_edge.edge_type == "NAMED_EDGE"
        assert named_edge.get("weight").as_int() == 2
        assert empty_edge.edge_type == "NO_PROPERTIES"
        assert {tuple(event["lpg_graph"]) for event in collect_history(db, named.id)} == {(), ("scoped",)}
        assert {tuple(event["lpg_graph"]) for event in collect_history(db, named_edge.id, edge=True)} == {(), ("scoped",)}

        db.execute("USE GRAPH default")
        assert db.get_node(root.id).get("name").as_str() == "Root"
        assert db.get_edge(root_edge.id).get("weight").as_int() == 1

    def test_edge_history_after_create(self, db):
        a = db.create_node(["N"])
        b = db.create_node(["N"])
        edge = db.create_edge(a.id, b.id, "R")
        history = collect_history(db, edge.id, edge=True)
        assert len(history) >= 1
        assert history[0]["edge_type"] == "R"
        assert history[0]["src_id"] == a.id
        assert history[0]["dst_id"] == b.id

    def test_bounded_changes_resume_exactly(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        cursor = None
        changes = []
        while True:
            page = db.changes_after(cursor, 1, 4096)
            assert isinstance(page["next"], bytes) and len(page["next"]) == 97
            assert len(page["events"]) <= 1
            if page["next"] == cursor:
                break
            cursor = page["next"]
            changes.extend(page["events"])
        assert all(isinstance(e["graph_incarnation"], int) for e in changes)
        assert len(changes) >= 2

    def test_empty_history(self, db):
        # Node ID that doesn't exist
        history = collect_history(db, 9999)
        assert len(history) == 0

    def test_transaction_with_cdc_enabled(self, db):
        """Per-transaction CDC override tracks changes when enabled."""
        with db.begin_transaction_with_cdc(True) as tx:
            tx.execute("INSERT (:Person {name: 'Gus'})")
            tx.commit()
        result = db.execute("MATCH (n:Person {name: 'Gus'}) RETURN n")
        rows = list(result)
        assert len(rows) == 1
        node_id = rows[0]["n"]["_id"]
        history = collect_history(db, node_id)
        assert len(history) >= 1

    def test_transaction_with_cdc_disabled(self, db):
        """Per-transaction CDC override suppresses tracking when disabled."""
        # Insert with CDC explicitly disabled for this transaction
        with db.begin_transaction_with_cdc(False) as tx:
            tx.execute("INSERT (:Person {name: 'Vincent'})")
            tx.commit()
        result = db.execute("MATCH (n:Person {name: 'Vincent'}) RETURN n")
        rows = list(result)
        assert len(rows) == 1
        node_id = rows[0]["n"]["_id"]
        history = collect_history(db, node_id)
        assert len(history) == 0


def test_cursor_errors_disabled_capture_and_owned_close():
    db = GrafeoDB(cdc=True)
    db.create_node(["Retained"])
    db.disable_cdc()
    page = db.changes_after(None, 1, 4096)
    assert len(page["events"]) == 1
    db.create_node(["Uncaptured"])
    assert db.changes_after(page["next"], 1, 4096)["next"] == page["next"]
    with pytest.raises(Exception) as malformed:
        db.changes_after(b"bad", 1, 4096)
    assert malformed.value.error_code == "GRAFEO-S004"
    with GrafeoDB(cdc=True) as other:
        foreign = other.changes_after(None, 1, 4096)["next"]
    with pytest.raises(Exception) as foreign_error:
        db.changes_after(foreign, 1, 4096)
    assert foreign_error.value.error_code == "GRAFEO-S005"
    with pytest.raises(Exception) as limit:
        db.changes_after(None, 1, 1)
    assert limit.value.error_code == "GRAFEO-S001"
    saved = repr(page)
    db.close()
    with pytest.raises(Exception):
        db.changes_after(page["next"], 1, 4096)
    assert repr(page) == saved


def test_cursor_resumes_exact_tail_after_two_reopens(tmp_path):
    path = str(tmp_path / "cdc.grafeo")
    with GrafeoDB(path, cdc=True) as db:
        ids = [db.create_node([label]).id for label in ("First", "Second", "Third")]
        expected = db.changes_after(None, 3, 4096)["events"]
        assert [event["entity_id"] for event in expected] == ids
        assert len(set(ids)) == 3
        first = db.changes_after(None, 1, 4096)
        assert first["events"] == [expected[0]]
        cursor = first["next"]
    for index in (1, 2):
        with GrafeoDB(path) as db:
            page = db.changes_after(cursor, 1, 4096)
            assert page["events"] == [expected[index]]
            cursor = page["next"]
            if index == 2:
                eof = db.changes_after(cursor, 1, 4096)
                assert eof["events"] == []
                assert eof["next"] == cursor
    assert first["events"] == [expected[0]]


def test_bounded_entity_history_epoch_filter_and_resume(tmp_path):
    path = str(tmp_path / "entity-history.grafeo")
    db = GrafeoDB(path, cdc=True)
    node = db.create_node(["History"])
    db.set_node_property(node.id, "n", 1)
    db.set_node_property(node.id, "n", 2)
    first = db.node_history_after(node.id, None, 1, 4096)
    tail = db.node_history_after(node.id, first["next"], 2, 4096)
    assert len(first["events"]) == 1 and len(tail["events"]) == 2
    since = tail["events"][0]["epoch"]
    assert collect_history(db, node.id, since_epoch=since) == tail["events"]
    with pytest.raises(Exception) as malformed:
        db.node_history_after(node.id, b"bad", 1, 4096)
    assert malformed.value.error_code == "GRAFEO-S004"
    assert collect_history(db, 2**53 + 1) == []
    db.close()
    for _ in range(2):
        db = GrafeoDB(path)
        assert db.node_history_after(node.id, first["next"], 2, 4096) == tail
        db.close()
    with pytest.raises(Exception):
        db.node_history_after(node.id, first["next"], 1, 4096)


def test_retained_cut_preserves_stale_cursor_errors(tmp_path):
    fixture = os.environ.get("GRAFEO_CDC_EVICTED_FIXTURE")
    if not fixture:
        pytest.skip("generate the native C retained-cut fixture first")
    cursor = Path(fixture).with_suffix(".cursor").read_bytes()
    assert len(cursor) == 97
    target = tmp_path / "retained.grafeo"
    shutil.copyfile(fixture, target)
    with GrafeoDB(str(target)) as db:
        for read in (
            lambda: db.changes_after(cursor, 1, 4096),
            lambda: db.node_history_after(0, cursor, 1, 4096),
            lambda: db.edge_history_after(0, cursor, 1, 4096),
        ):
            with pytest.raises(Exception) as stale:
                read()
            assert stale.value.error_code == "GRAFEO-S006"
        assert len(db.changes_after(None, 1, 4096)["events"]) == 1

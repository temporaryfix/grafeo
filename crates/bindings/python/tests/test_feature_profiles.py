"""Public callers for isolated Python feature profiles.

The qualification runner supplies GRAFEO_EXPECT_STORAGE/GRAFEO_EXPECT_GQL
and selects the LPG or RDF controls for the compiled model.
"""

import os

import pytest

from grafeo import GrafeoDB


def test_lpg_crud_preserves_recursive_properties():
    db = GrafeoDB()
    try:
        props = {"name": "雪", "value": 42, "nested": {"ok": True, "items": [1, "two"]}}
        first = db.create_node(["First"], props)
        second = db.create_node(["Second"])
        assert db.get_node(first.id).properties() == props
        edge = db.create_edge(first.id, second.id, "LINK", {"value": 7})
        db.set_node_property(first.id, "value", 43)
        db.set_edge_property(edge.id, "value", 8)
        assert db.get_node(first.id).properties()["value"] == 43
        assert db.get_edge(edge.id).properties()["value"] == 8
        assert db.add_node_label(first.id, "Added")
        assert sorted(db.get_node_labels(first.id)) == ["Added", "First"]
        assert db.remove_node_label(first.id, "Added")
        assert db.remove_node_property(first.id, "nested")
        assert db.remove_edge_property(edge.id, "value")
        assert (db.node_count, db.edge_count) == (2, 1)
        assert db.delete_edge(edge.id)
        assert db.delete_node(second.id)
        assert (db.node_count, db.edge_count) == (1, 0)
    finally:
        db.close()


def test_lpg_explicit_isolation_commit_rollback_and_savepoint():
    db = GrafeoDB()
    try:
        for level in ("snapshot", "read_committed", "serializable"):
            with db.begin_transaction(isolation_level=level) as tx:
                kept = tx.create_node(["Kept"], {"level": level})
                assert db.get_node(kept.id) is None
                tx.savepoint("after_kept")
                removed = tx.create_node(["Removed"])
                tx.rollback_to_savepoint("after_kept")
                tx.commit()
            assert db.get_node(kept.id).properties() == {"level": level}
            assert db.get_node(removed.id) is None
        with db.begin_transaction() as tx:
            removed = tx.create_node(["RolledBack"])
            tx.rollback()
        assert db.get_node(removed.id) is None
        assert db.node_count == 3
    finally:
        db.close()


def test_lpg_administration_fork_and_storage(tmp_path):
    db = GrafeoDB()
    try:
        node = db.create_node(["Saved"], {"value": 42})
        assert db.info()["node_count"] == 1
        assert db.detailed_stats()["node_count"] == 1
        assert db.schema()["mode"] == "lpg"
        assert db.validate() == []
        assert db.memory_usage()["total_bytes"] > 0
        assert db.current_epoch() > 0
        assert not db.is_persistent and db.path is None
        copy = db.to_memory()
        try:
            assert copy.get_node(node.id).properties() == {"value": 42}
            copy.set_node_property(node.id, "value", 43)
            assert db.get_node(node.id).properties() == {"value": 42}
        finally:
            copy.close()
        path = str(tmp_path / "saved.grafeo")
        if os.environ.get("GRAFEO_EXPECT_STORAGE", "1") == "0":
            assert not hasattr(db, "save")
            assert not hasattr(db, "backup_full")
            with pytest.raises(Exception, match="(?i)persist"):
                GrafeoDB(path)
        else:
            assert callable(db.backup_full) and callable(db.backup_incremental)
            db.save(path)
            reopened = GrafeoDB.open(path)
            try:
                assert reopened.is_persistent and reopened.path == path
                assert reopened.get_node(node.id).properties() == {"value": 42}
            finally:
                reopened.close()
            copy = GrafeoDB.open_in_memory(path)
            try:
                assert not copy.is_persistent
                assert copy.get_node(node.id).properties() == {"value": 42}
            finally:
                copy.close()
    finally:
        db.close()


def test_lpg_parser_availability_matches_profile():
    db = GrafeoDB()
    try:
        if os.environ.get("GRAFEO_EXPECT_GQL", "1") == "0":
            with pytest.raises(Exception):
                db.execute("RETURN 1 AS value")
        else:
            assert list(db.execute("RETURN 1 AS value")) == [{"value": 1}]
    finally:
        db.close()


def test_rdf_shared_administration_save_and_fork(tmp_path):
    db = GrafeoDB(graph_model="rdf")
    quad = ("<urn:s>", "<urn:p>", '"雪"', "urn:graph")
    try:
        count, epoch = db.insert_rdf_quad(*quad)
        assert count == 1 and isinstance(epoch, int) and epoch > 0
        assert db.contains_rdf_quad(*quad)
        assert db.info()["mode"].lower() == "rdf"
        assert db.current_epoch() == epoch
        assert not db.is_persistent and db.path is None
        copy = db.to_memory()
        try:
            assert copy.contains_rdf_quad(*quad)
        finally:
            copy.close()
        path = str(tmp_path / "rdf.grafeo")
        db.save(path)
        reopened = GrafeoDB.open(path)
        try:
            assert reopened.contains_rdf_quad(*quad)
            assert reopened.is_persistent
        finally:
            reopened.close()
    finally:
        db.close()


@pytest.mark.parametrize("level", ["snapshot", "read_committed", "serializable"])
def test_rdf_savepoint_and_explicit_isolation(level):
    db = GrafeoDB(graph_model="rdf")
    kept = ("<urn:kept>", "<urn:p>", '"kept"', "urn:graph")
    removed = ("<urn:removed>", "<urn:p>", '"removed"', "urn:graph")
    try:
        with db.begin_transaction(isolation_level=level) as tx:
            assert tx.insert_rdf_quad(*kept) == 1
            assert not db.contains_rdf_quad(*kept)
            tx.savepoint("after_kept")
            assert tx.insert_rdf_quad(*removed) == 1
            tx.rollback_to_savepoint("after_kept")
            tx.commit()
        assert db.contains_rdf_quad(*kept)
        assert not db.contains_rdf_quad(*removed)
    finally:
        db.close()


def test_compact_preserves_parallel_edges_history_and_overlay():
    db = GrafeoDB()
    try:
        first = db.create_node(["Node"], {"value": 42})
        second = db.create_node(["Node"], {"value": 99})
        left = db.create_edge(first.id, second.id, "LINK", {"weight": 7})
        right = db.create_edge(first.id, second.id, "LINK", {"weight": 8})
        epoch = db.current_epoch()
        db.compact()

        def assert_historical_image():
            image = db.scrub_at_epoch(epoch)
            nodes = [node_id for frame in image["nodes"] for node_id in frame["node_ids"]]
            assert sorted(nodes) == sorted([first.id, second.id])
            edges = [
                row
                for frame in image["edges"]
                for row in zip(
                    frame["edge_ids"], frame["src_ids"], frame["dst_ids"],
                    frame["columns"]["weight"], strict=True,
                )
            ]
            assert sorted(edges) == sorted([
                (left.id, first.id, second.id, 7),
                (right.id, first.id, second.id, 8),
            ])
            assert db.get_node_at_epoch(first.id, epoch).properties() == {"value": 42}

        assert_historical_image()
        db.set_node_property(first.id, "value", 43)
        assert db.delete_edge(left.id)
        for _ in range(2):
            assert db.get_node(first.id).properties() == {"value": 43}
            assert db.get_edge(left.id) is None
            assert db.get_edge(right.id).properties() == {"weight": 8}
            assert (db.node_count, db.edge_count) == (2, 1)
            assert_historical_image()
            db.compact()
        assert_historical_image()
    finally:
        db.close()

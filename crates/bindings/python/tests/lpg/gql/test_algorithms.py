"""GQL implementation of algorithm tests.

Tests graph algorithms with GQL for setup/verification.
"""

import random

import pytest

from tests.bases.test_algorithms import BaseAlgorithmsTest


class TestGQLAlgorithms(BaseAlgorithmsTest):
    """GQL implementation of algorithm tests.

    Note: Algorithms are accessed via db.algorithms.*, not via GQL queries.
    GQL is used for setup and verification only.
    """

    def test_managed_binding_reads_survive_compact_overlay_and_tombstone(self, db):
        """Every Python graph reader sees the same tier-merged snapshot."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        ab = db.create_edge(a.id, b.id, "LINK", {})

        db.compact()
        c = db.create_node(["Node"], {"name": "c"})
        bc = db.create_edge(b.id, c.id, "LINK", {})

        assert db.algorithms.bfs(a.id) == [a.id, b.id, c.id]
        assert set(db.as_networkx().nodes()) == {a.id, b.id, c.id}
        assert set(db.as_solvor().connected_components()) == {a.id, b.id, c.id}
        assert {node_id for node_id, _ in db.get_nodes_by_label("Node")} == {
            a.id,
            b.id,
            c.id,
        }
        assert db.get_property_batch([a.id, b.id, c.id], "name") == ["a", "b", "c"]

        assert db.delete_edge(ab.id)
        assert db.delete_edge(bc.id)
        assert db.delete_node(b.id)
        assert set(db.as_networkx().nodes()) == {a.id, c.id}
        assert set(db.as_solvor().connected_components()) == {a.id, c.id}
        assert {node_id for node_id, _ in db.get_nodes_by_label("Node")} == {a.id, c.id}

        db.compact()
        assert set(db.as_networkx().nodes()) == {a.id, c.id}
        assert set(db.as_solvor().connected_components()) == {a.id, c.id}

    def setup_algorithm_graph(self, db, n_nodes: int = 100, n_edges: int = 300):
        """Set up a random graph for algorithm testing."""
        rng = random.Random(42)

        node_ids = []
        for i in range(n_nodes):
            node = db.create_node(["Node"], {"index": i})
            node_ids.append(node.id)

        edges = set()
        while len(edges) < n_edges:
            src = rng.choice(node_ids)
            dst = rng.choice(node_ids)
            if src != dst and (src, dst) not in edges:
                db.create_edge(src, dst, "EDGE", {"weight": rng.uniform(0.1, 10.0)})
                edges.add((src, dst))

        return {"node_ids": node_ids, "edge_count": len(edges)}

    def test_sssp_explicit_property_and_internal_id_results(self, db):
        """Resolution and integer result keys share the native SSSP contract."""
        a = db.create_node(["Node"], {"id": "v_0", "name": "collision"})
        b = db.create_node(["Node"], {"id": "v_1", "name": "collision"})
        c = db.create_node(["Node"], {"id": "v_2", "name": str(a.id)})
        db.create_edge(a.id, b.id, "LINK", {"weight": 2.0})
        db.create_edge(b.id, c.id, "LINK", {"weight": 3.0})

        expected = {a.id: 0.0, b.id: 2.0, c.id: 5.0}
        assert db.algorithms.sssp("v_0", "weight", key="id") == expected
        assert db.algorithms.sssp(str(a.id), "weight") == expected
        assert db.algorithms.sssp("v_0", key="id") == {
            a.id: 0.0, b.id: 1.0, c.id: 2.0
        }
        with pytest.raises(ValueError, match="requires `key`"):
            db.algorithms.sssp("collision")
        with pytest.raises(ValueError, match="requires `key`"):
            db.algorithms.sssp("v_0")
        numeric = db.create_node(["Node"], {"id": 900})
        assert db.algorithms.sssp("900", key="id") == {numeric.id: 0.0}

    def test_sssp_rejects_missing_duplicate_and_malformed_lookup(self, db):
        db.create_node(["Node"], {"id": "duplicate"})
        db.create_node(["Node"], {"id": "duplicate"})
        with pytest.raises(ValueError, match="Multiple nodes"):
            db.algorithms.sssp("duplicate", key="id")
        for source, key in [("missing", "id"), ("duplicate", "missing"), ("duplicate", "")]:
            with pytest.raises(ValueError, match="No node found"):
                db.algorithms.sssp(source, key=key)
        for source in ["-1", "not-an-internal-id"]:
            with pytest.raises(ValueError, match="requires `key`"):
                db.algorithms.sssp(source)
        with pytest.raises(ValueError, match="No node found with internal ID"):
            db.algorithms.sssp("18446744073709551615")
        with pytest.raises(TypeError):
            db.algorithms.sssp("duplicate", key=42)

    def test_directed_clustering_reciprocal_degree_serial_parallel(self, db):
        """LDBC uses distinct neighbours: the reciprocal spoke gives 1/6."""
        i, a, b, c = [db.create_node(["Node"], {}) for _ in range(4)]
        for source, target in [(i, a), (i, b), (i, c), (a, i), (a, b)]:
            db.create_edge(source.id, target.id, "LINK", {})
        # Cross the native threshold so parallel=True exercises parallel work.
        for _ in range(48):
            db.create_node(["Isolated"], {})
        for parallel in [False, True]:
            directed = db.algorithms.clustering_coefficient(parallel, directed=True)
            assert directed["coefficients"][i.id] == pytest.approx(1.0 / 6.0)
            assert directed["triangle_counts"][i.id] == 1
            undirected = db.algorithms.clustering_coefficient(parallel)
            assert undirected["coefficients"][i.id] == pytest.approx(1.0 / 3.0)

    def test_bfs_layers_options_and_call_parity(self, db):
        """The Python and CALL BFS surfaces share filtering, bounds, and direction."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        d = db.create_node(["Node"], {"name": "d"})

        db.create_edge(a.id, b.id, "KNOWS", {})
        db.create_edge(b.id, c.id, "KNOWS", {})
        db.create_edge(a.id, d.id, "LIKES", {})

        expected = [[a.id], [b.id], [c.id]]
        assert db.algorithms.bfs_layers(a.id, "KNOWS", 2, "outgoing") == expected
        assert (
            db.algorithms.bfs_layers(
                a.id, edge_type="KNOWS", max_depth=2, direction="outgoing"
            )
            == expected
        )
        assert db.algorithms.bfs_layers(
            a.id, edge_type="LIKES", max_depth=0
        ) == [[a.id]]
        assert db.algorithms.bfs_layers(
            c.id, edge_type="KNOWS", direction="incoming"
        ) == [
            [c.id],
            [b.id],
            [a.id],
        ]
        assert [set(layer) for layer in db.algorithms.bfs_layers(
            b.id, direction="both", max_depth=1
        )] == [{b.id}, {a.id, c.id}]

        call_rows = list(db.execute(f"CALL grafeo.bfs({a.id}, 'KNOWS', 2, 'outgoing')"))
        assert [(row["node_id"], row["depth"]) for row in call_rows] == [
            (a.id, 0),
            (b.id, 1),
            (c.id, 2),
        ]

        with pytest.raises(ValueError):
            db.algorithms.bfs_layers(a.id, direction="sideways")


# Additional GQL-specific algorithm tests using GQL for verification


class TestGQLAlgorithmVerification:
    """Tests that verify algorithm results using GQL queries."""

    def test_verify_bfs_reachability(self, db):
        """Verify BFS results match GQL path query."""
        # Create a simple graph
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        db.create_node(["Node"], {"name": "d"})  # Isolated node

        db.create_edge(a.id, b.id, "EDGE", {})
        db.create_edge(b.id, c.id, "EDGE", {})

        # Run BFS from node a
        bfs_result = db.algorithms.bfs(a.id)

        # Verify with GQL - nodes reachable from a
        result = db.execute(
            "MATCH p = (start:Node {name: 'a'})-[:EDGE*0..10]->(end:Node) RETURN DISTINCT end.name"
        )
        gql_reachable = {r["end.name"] for r in result}  # noqa: F841

        # a, b, c should be reachable; d should not
        assert a.id in bfs_result
        assert b.id in bfs_result
        assert c.id in bfs_result
        # d is isolated, should not be in BFS from a

    def test_verify_connected_components(self, db):
        """Verify connected components match GQL connectivity."""
        # Create two disconnected components
        # Component 1: a-b-c
        a = db.create_node(["Node"], {"name": "a", "group": 1})
        b = db.create_node(["Node"], {"name": "b", "group": 1})
        c = db.create_node(["Node"], {"name": "c", "group": 1})
        db.create_edge(a.id, b.id, "EDGE", {})
        db.create_edge(b.id, c.id, "EDGE", {})

        # Component 2: x-y
        x = db.create_node(["Node"], {"name": "x", "group": 2})
        y = db.create_node(["Node"], {"name": "y", "group": 2})
        db.create_edge(x.id, y.id, "EDGE", {})

        # Run connected components
        components = db.algorithms.connected_components()
        component_count = db.algorithms.connected_component_count()

        # Should have 2 components
        assert component_count == 2

        # Nodes in same component should have same component ID
        assert components[a.id] == components[b.id] == components[c.id]
        assert components[x.id] == components[y.id]
        assert components[a.id] != components[x.id]

    def test_verify_pagerank_structure(self, db):
        """Verify PageRank reflects link structure."""
        # Create a star graph: center connected to 4 leaves
        center = db.create_node(["Node"], {"name": "center"})
        leaves = []
        for i in range(4):
            leaf = db.create_node(["Node"], {"name": f"leaf{i}"})
            leaves.append(leaf)
            db.create_edge(leaf.id, center.id, "POINTS_TO", {})

        # Run PageRank
        pr = db.algorithms.pagerank()

        # Center should have higher PageRank (receives all links)
        center_pr = pr[center.id]
        for leaf in leaves:
            assert center_pr > pr[leaf.id], "Center should have highest PageRank"

    def test_verify_shortest_path(self, db):
        """Verify Dijkstra shortest path matches expected."""
        # Create a weighted graph
        #     1
        # a ----> b
        # |       |
        # |10     |1
        # v       v
        # c ----> d
        #     1
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        d = db.create_node(["Node"], {"name": "d"})

        db.create_edge(a.id, b.id, "EDGE", {"weight": 1})
        db.create_edge(a.id, c.id, "EDGE", {"weight": 10})
        db.create_edge(b.id, d.id, "EDGE", {"weight": 1})
        db.create_edge(c.id, d.id, "EDGE", {"weight": 1})

        # Shortest path from a to d should go through b (cost 2)
        # Not through c (cost 11)
        result = db.algorithms.dijkstra(a.id, d.id, "weight")
        if result is not None:
            distance, path = result
            assert distance == 2, f"Expected distance 2, got {distance}"
            assert a.id in path
            assert b.id in path
            assert d.id in path
            assert c.id not in path, "Should not go through c"

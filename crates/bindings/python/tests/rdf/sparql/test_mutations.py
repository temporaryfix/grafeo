"""SPARQL implementation of mutation tests.

Tests SPARQL Update operations (INSERT DATA, DELETE DATA, etc.).
Note: SPARQL mutations operate on RDF triples, not LPG nodes/edges.
"""

import pytest

# Try to import grafeo
try:
    from grafeo import GrafeoDB

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GRAFEO_AVAILABLE, reason="Grafeo Python bindings not installed")


@pytest.fixture
def db():
    """Create an RDF database for SPARQL mutation tests."""
    return GrafeoDB(graph_model="rdf")


class TestSPARQLMutations:
    """SPARQL Update mutation tests.

    Note: SPARQL mutations operate on RDF triples.
    The base mutation tests are for LPG, so we implement
    SPARQL-specific mutation tests here.
    """

    def test_insert_data_single_triple(self, db):
        """Test INSERT DATA with a single triple."""
        query = """
            INSERT DATA {
                <http://example.org/alix> <http://example.org/name> "Alix" .
            }
        """
        db.execute_sparql(query)

        # Verify the triple was inserted
        result = list(
            db.execute_sparql("""
            SELECT ?name WHERE {
                <http://example.org/alix> <http://example.org/name> ?name .
            }
        """)
        )
        assert len(result) > 0

    def test_insert_data_multiple_triples(self, db):
        """Test INSERT DATA with multiple triples."""
        query = """
            INSERT DATA {
                <http://example.org/alix> <http://example.org/name> "Alix" .
                <http://example.org/alix> <http://example.org/age> 30 .
                <http://example.org/alix> a <http://example.org/Person> .
            }
        """
        db.execute_sparql(query)

        # Verify triples were inserted
        result = list(
            db.execute_sparql("""
            SELECT ?p ?o WHERE {
                <http://example.org/alix> ?p ?o .
            }
        """)
        )
        assert len(result) >= 3

    def test_delete_data_single_triple(self, db):
        """Test DELETE DATA with a single triple."""
        # First insert
        db.execute_sparql("""
            INSERT DATA {
                <http://example.org/gus> <http://example.org/name> "Gus" .
            }
        """)

        # Then delete
        db.execute_sparql("""
            DELETE DATA {
                <http://example.org/gus> <http://example.org/name> "Gus" .
            }
        """)

        # Verify deletion
        result = list(
            db.execute_sparql("""
            SELECT ?name WHERE {
                <http://example.org/gus> <http://example.org/name> ?name .
            }
        """)
        )
        assert len(result) == 0

    def test_delete_where(self, db):
        """Test DELETE WHERE pattern matching."""
        # Insert test data
        db.execute_sparql("""
            INSERT DATA {
                <http://example.org/temp1> <http://example.org/status> "temporary" .
                <http://example.org/temp2> <http://example.org/status> "temporary" .
                <http://example.org/keep> <http://example.org/status> "permanent" .
            }
        """)

        # Delete all temporary items
        db.execute_sparql("""
            DELETE WHERE {
                ?s <http://example.org/status> "temporary" .
            }
        """)

        # Verify only permanent item remains
        result = list(
            db.execute_sparql("""
            SELECT ?s WHERE {
                ?s <http://example.org/status> ?status .
            }
        """)
        )
        assert len(result) == 1

    def test_modify_delete_insert(self, db):
        """Test DELETE/INSERT WHERE (modify operation)."""
        # Insert initial data
        db.execute_sparql("""
            INSERT DATA {
                <http://example.org/item> <http://example.org/version> 1 .
            }
        """)

        # Modify: delete old version, insert new version
        db.execute_sparql("""
            DELETE { ?s <http://example.org/version> ?old }
            INSERT { ?s <http://example.org/version> 2 }
            WHERE { ?s <http://example.org/version> ?old }
        """)

        # Verify version was updated
        result = list(
            db.execute_sparql("""
            SELECT ?v WHERE {
                <http://example.org/item> <http://example.org/version> ?v .
            }
        """)
        )
        assert len(result) == 1
        # Version should be 2 now


class TestSPARQLGraphManagement:
    """Tests for SPARQL graph management operations."""

    def test_create_graph(self, db):
        """Test CREATE GRAPH."""
        db.execute_sparql("""
            CREATE GRAPH <http://example.org/newgraph>
        """)
        # Graph creation should succeed

    def test_drop_graph(self, db):
        """Test DROP GRAPH."""
        db.execute_sparql("""
            CREATE GRAPH <http://example.org/tempgraph>
        """)
        db.execute_sparql("""
            DROP GRAPH <http://example.org/tempgraph>
        """)
        # Graph should be dropped

    def test_clear_default(self, db):
        """Test CLEAR DEFAULT."""
        # Insert data
        db.execute_sparql("""
            INSERT DATA {
                <http://example.org/s> <http://example.org/p> "value" .
            }
        """)

        # Clear default graph
        db.execute_sparql("CLEAR DEFAULT")

        # Verify data is gone
        result = list(
            db.execute_sparql("""
            SELECT ?s WHERE { ?s ?p ?o }
        """)
        )
        assert len(result) == 0

    def test_copy_graph(self, db):
        """Test COPY: copies triples, source remains intact."""
        # Set up: create source graph with data
        db.execute_sparql("""
            INSERT DATA {
                GRAPH <http://example.org/src> {
                    <http://example.org/a> <http://example.org/p> "hello" .
                    <http://example.org/b> <http://example.org/q> "world" .
                }
            }
        """)

        # Copy src -> dst
        db.execute_sparql("COPY <http://example.org/src> TO <http://example.org/dst>")

        # Source still has its data
        src_result = list(
            db.execute_sparql("""
            SELECT ?s ?o WHERE {
                GRAPH <http://example.org/src> { ?s ?p ?o }
            }
        """)
        )
        assert len(src_result) == 2, f"Source should retain 2 triples, got {len(src_result)}"

        # Destination has the same data
        dst_result = list(
            db.execute_sparql("""
            SELECT ?s ?o WHERE {
                GRAPH <http://example.org/dst> { ?s ?p ?o }
            }
        """)
        )
        assert len(dst_result) == 2, f"Dest should have 2 triples, got {len(dst_result)}"

    def test_move_graph(self, db):
        """Test MOVE: triples move to destination, source is removed."""
        db.execute_sparql("""
            INSERT DATA {
                GRAPH <http://example.org/origin> {
                    <http://example.org/x> <http://example.org/v> "data" .
                }
            }
        """)

        db.execute_sparql("MOVE <http://example.org/origin> TO <http://example.org/target>")

        # Source should be empty/gone
        src_result = list(
            db.execute_sparql("""
            SELECT ?s WHERE {
                GRAPH <http://example.org/origin> { ?s ?p ?o }
            }
        """)
        )
        assert len(src_result) == 0, "Source should be empty after MOVE"

        # Destination should have the data
        dst_result = list(
            db.execute_sparql("""
            SELECT ?s ?o WHERE {
                GRAPH <http://example.org/target> { ?s ?p ?o }
            }
        """)
        )
        assert len(dst_result) == 1, f"Target should have 1 triple, got {len(dst_result)}"

    def test_add_graph(self, db):
        """Test ADD: merges source into destination without removing source."""
        # Create two graphs with distinct data
        db.execute_sparql("""
            INSERT DATA {
                GRAPH <http://example.org/g1> {
                    <http://example.org/a> <http://example.org/p> "from-g1" .
                }
                GRAPH <http://example.org/g2> {
                    <http://example.org/b> <http://example.org/q> "from-g2" .
                }
            }
        """)

        # Add g1 into g2 (union)
        db.execute_sparql("ADD <http://example.org/g1> TO <http://example.org/g2>")

        # g1 unchanged
        g1_result = list(
            db.execute_sparql("""
            SELECT ?s WHERE {
                GRAPH <http://example.org/g1> { ?s ?p ?o }
            }
        """)
        )
        assert len(g1_result) == 1, "g1 should still have 1 triple"

        # g2 has both its own triple + g1's triple
        g2_result = list(
            db.execute_sparql("""
            SELECT ?s ?o WHERE {
                GRAPH <http://example.org/g2> { ?s ?p ?o }
            }
        """)
        )
        assert len(g2_result) == 2, f"g2 should have 2 triples after ADD, got {len(g2_result)}"

    def test_named_graph_isolation(self, db):
        """Test that triples in named graphs are invisible from default graph queries."""
        db.execute_sparql("""
            INSERT DATA {
                GRAPH <http://example.org/private> {
                    <http://example.org/secret> <http://example.org/val> "hidden" .
                }
            }
        """)

        # Query default graph — should see nothing from the named graph
        default_result = list(
            db.execute_sparql("""
            SELECT ?s WHERE { ?s ?p ?o }
        """)
        )
        assert len(default_result) == 0, (
            f"Default graph should not contain named graph triples, got {len(default_result)}"
        )

        # Query the named graph directly — should find the triple
        named_result = list(
            db.execute_sparql("""
            SELECT ?s WHERE {
                GRAPH <http://example.org/private> { ?s ?p ?o }
            }
        """)
        )
        assert len(named_result) == 1

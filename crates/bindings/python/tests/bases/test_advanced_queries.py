"""Base class for advanced query tests.

Covers GQL features that need dedicated test coverage:
- UNWIND
- MERGE (upsert)
- OPTIONAL MATCH
- CASE expressions
- NULL handling
- String predicates
- IN operator
"""

from abc import ABC, abstractmethod


class BaseAdvancedQueriesTest(ABC):
    """Abstract base class for advanced query tests."""

    def execute_query(self, db, query):
        """Execute a query using the default (GQL) parser."""
        return db.execute(query)

    # =========================================================================
    # SETUP
    # =========================================================================

    @abstractmethod
    def setup_social_graph(self, db):
        """Set up: Alix(30,NYC), Gus(25,LA), Harm(35,London) with KNOWS edges."""
        raise NotImplementedError

    # =========================================================================
    # UNWIND
    # =========================================================================

    def test_query_result_column_is_list_not_dicts(self, db):
        self.setup_social_graph(db)
        result = self.execute_query(db, "MATCH (n:Person) RETURN n.age AS age")
        ages = result.column("age")
        assert isinstance(ages, list)
        assert sorted(ages) == [25, 30, 35]
        assert result.column(0) == ages
        assert len(result) == 3

    def test_unwind_literal_list(self, db):
        result = self.execute_query(db, "UNWIND [1, 2, 3] AS x RETURN x")
        rows = list(result)
        assert len(rows) == 3

    def test_unwind_with_match(self, db):
        self.setup_social_graph(db)
        result = self.execute_query(
            db, "MATCH (n:Person {name: 'Alix'}) UNWIND [10, 20] AS x RETURN n.name, x"
        )
        rows = list(result)
        assert len(rows) == 2

    def test_unwind_empty_list(self, db):
        result = self.execute_query(db, "UNWIND [] AS x RETURN x")
        rows = list(result)
        assert len(rows) == 0

    # =========================================================================
    # MERGE
    # =========================================================================

    def test_merge_create_new(self, db):
        result = self.execute_query(db, "MERGE (n:Animal {species: 'Cat'}) RETURN n.species")
        rows = list(result)
        assert len(rows) == 1

    def test_merge_match_existing(self, db):
        self.setup_social_graph(db)
        before = db.node_count
        result = self.execute_query(db, "MERGE (n:Person {name: 'Alix'}) RETURN n.name")
        rows = list(result)
        assert len(rows) == 1
        assert db.node_count == before

    def test_merge_on_create_set(self, db):
        result = self.execute_query(
            db,
            "MERGE (n:Person {name: 'NewGuy'}) "
            "ON CREATE SET n.created = true RETURN n.name, n.created",
        )
        rows = list(result)
        assert len(rows) == 1

    def test_merge_on_match_set(self, db):
        self.setup_social_graph(db)
        result = self.execute_query(
            db,
            "MERGE (n:Person {name: 'Alix'}) ON MATCH SET n.found = true RETURN n.name, n.found",
        )
        rows = list(result)
        assert len(rows) == 1

    # =========================================================================
    # OPTIONAL MATCH
    # =========================================================================

    def test_optional_match_with_results(self, db):
        self.setup_social_graph(db)
        result = self.execute_query(
            db,
            "MATCH (a:Person {name: 'Alix'}) "
            "OPTIONAL MATCH (a)-[:KNOWS]->(b:Person) RETURN a.name, b.name",
        )
        rows = list(result)
        assert len(rows) >= 1

    def test_optional_match_null(self, db):
        self.setup_social_graph(db)
        result = self.execute_query(
            db,
            "MATCH (a:Person {name: 'Alix'}) OPTIONAL MATCH (a)-[:MANAGES]->(c) RETURN a, c",
        )
        rows = list(result)
        assert len(rows) >= 1, "OPTIONAL MATCH should produce at least 1 row"

    # =========================================================================
    # CASE expression
    # =========================================================================

    def test_case_when_then_else(self, db):
        self.setup_social_graph(db)
        result = self.execute_query(
            db,
            "MATCH (n:Person) RETURN n.name, "
            "CASE WHEN n.age > 30 THEN 'senior' ELSE 'junior' END AS category",
        )
        rows = list(result)
        assert len(rows) == 3

    # =========================================================================
    # NULL handling
    # =========================================================================

    def test_null_property_access(self, db):
        self.setup_social_graph(db)
        result = self.execute_query(db, "MATCH (n:Person {name: 'Alix'}) RETURN n.nonexistent")
        rows = list(result)
        assert len(rows) == 1

    # =========================================================================
    # String predicates
    # =========================================================================

    def test_string_starts_with(self, db):
        self.setup_social_graph(db)
        result = self.execute_query(
            db, "MATCH (n:Person) WHERE n.name STARTS WITH 'A' RETURN n.name"
        )
        rows = list(result)
        assert len(rows) == 1

    # =========================================================================
    # IN operator
    # =========================================================================

    def test_in_operator(self, db):
        self.setup_social_graph(db)
        result = self.execute_query(
            db,
            "MATCH (n:Person) WHERE n.name IN ['Alix', 'Gus'] RETURN n.name",
        )
        rows = list(result)
        assert len(rows) == 2

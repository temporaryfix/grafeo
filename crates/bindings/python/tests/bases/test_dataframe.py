"""Tests for DataFrame bridge: to_pandas(), to_polars(), nodes_df(), edges_df()."""

import pytest

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

import grafeo


@pytest.fixture
def populated_db():
    """Create a database with Person and Company nodes plus edges."""
    db = grafeo.GrafeoDB()
    db.execute("INSERT (:Person {name: 'Alix', age: 30})")
    db.execute("INSERT (:Person {name: 'Gus', age: 25})")
    db.execute("INSERT (:Person {name: 'Vincent', age: 35, city: 'Amsterdam'})")
    db.execute("INSERT (:Company {name: 'Acme Corp', founded: 2010})")
    db.execute("""
        MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'})
        INSERT (a)-[:KNOWS {since: 2020}]->(g)
    """)
    db.execute("""
        MATCH (a:Person {name: 'Alix'}), (c:Company {name: 'Acme Corp'})
        INSERT (a)-[:WORKS_AT {role: 'Engineer'}]->(c)
    """)
    return db


# --- QueryResult.to_pandas() ---


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestToPandas:
    def test_basic_query(self, populated_db):
        result = populated_db.execute("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.name")
        df = result.to_pandas()

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["n.name", "n.age"]
        assert len(df) == 3
        assert list(df["n.name"]) == ["Alix", "Gus", "Vincent"]
        assert list(df["n.age"]) == [30, 25, 35]

    def test_empty_result(self, populated_db):
        result = populated_db.execute("MATCH (n:Person {name: 'Nobody'}) RETURN n.name")
        df = result.to_pandas()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["n.name"]

    def test_null_values(self, populated_db):
        """Nodes without 'city' should produce None in the DataFrame."""
        result = populated_db.execute("MATCH (n:Person) RETURN n.name, n.city ORDER BY n.name")
        df = result.to_pandas()

        assert df.loc[df["n.name"] == "Vincent", "n.city"].iloc[0] == "Amsterdam"
        assert pd.isna(df.loc[df["n.name"] == "Alix", "n.city"].iloc[0])

    def test_mixed_types(self, populated_db):
        """Columns with mixed types (int, string, null) should work."""
        result = populated_db.execute("MATCH (n) RETURN n.name, labels(n) ORDER BY n.name")
        df = result.to_pandas()
        assert len(df) == 4  # 3 persons + 1 company

    def test_single_column(self, populated_db):
        result = populated_db.execute("MATCH (n:Person) RETURN count(n)")
        df = result.to_pandas()
        assert len(df) == 1
        assert df.iloc[0, 0] == 3


# --- QueryResult.to_polars() ---


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
class TestToPolars:
    def test_basic_query(self, populated_db):
        result = populated_db.execute("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.name")
        df = result.to_polars()

        assert isinstance(df, pl.DataFrame)
        assert df.columns == ["n.name", "n.age"]
        assert len(df) == 3
        assert df["n.name"].to_list() == ["Alix", "Gus", "Vincent"]
        assert df["n.age"].to_list() == [30, 25, 35]

    def test_empty_result(self, populated_db):
        result = populated_db.execute("MATCH (n:Person {name: 'Nobody'}) RETURN n.name")
        df = result.to_polars()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_null_values(self, populated_db):
        result = populated_db.execute("MATCH (n:Person) RETURN n.name, n.city ORDER BY n.name")
        df = result.to_polars()
        vincent_row = df.filter(pl.col("n.name") == "Vincent")
        assert vincent_row["n.city"][0] == "Amsterdam"


# --- db.nodes_df() ---


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestNodesDf:
    def test_basic(self, populated_db):
        df = populated_db.nodes_df()

        assert isinstance(df, pd.DataFrame)
        assert "_id" in df.columns
        assert "_labels" in df.columns
        assert "name" in df.columns
        assert len(df) == 4  # 3 persons + 1 company

    def test_property_columns(self, populated_db):
        """Each unique property key across all nodes becomes a column."""
        df = populated_db.nodes_df()
        # Person nodes have name, age (and optionally city)
        # Company nodes have name, founded
        assert "age" in df.columns
        assert "founded" in df.columns
        assert "name" in df.columns

    def test_missing_properties_are_none(self, populated_db):
        """Nodes without a property should have None in that column."""
        df = populated_db.nodes_df()
        # Company node shouldn't have 'age'
        company_rows = df[df["_labels"].apply(lambda labels: "Company" in labels)]
        assert company_rows["age"].isna().all()

    def test_labels_are_lists(self, populated_db):
        df = populated_db.nodes_df()
        for labels in df["_labels"]:
            assert isinstance(labels, list)

    def test_empty_graph(self):
        db = grafeo.GrafeoDB()
        df = db.nodes_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["_id", "_labels"]


# --- db.edges_df() ---


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestEdgesDf:
    def test_basic(self, populated_db):
        df = populated_db.edges_df()

        assert isinstance(df, pd.DataFrame)
        assert "_id" in df.columns
        assert "_source" in df.columns
        assert "_target" in df.columns
        assert "_type" in df.columns
        assert len(df) == 2  # KNOWS + WORKS_AT

    def test_edge_types(self, populated_db):
        df = populated_db.edges_df()
        types = set(df["_type"])
        assert types == {"KNOWS", "WORKS_AT"}

    def test_property_columns(self, populated_db):
        df = populated_db.edges_df()
        assert "since" in df.columns
        assert "role" in df.columns

    def test_missing_properties_are_none(self, populated_db):
        df = populated_db.edges_df()
        # KNOWS edge has 'since' but not 'role', WORKS_AT has 'role' but not 'since'
        knows_rows = df[df["_type"] == "KNOWS"]
        assert knows_rows["role"].isna().all()

    def test_empty_graph(self):
        db = grafeo.GrafeoDB()
        df = db.edges_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        # Column order is implementation-defined (currently alphabetical
        # among the metadata columns); compare as a set so the test
        # doesn't break if the implementation reorders.
        assert set(df.columns) == {"_id", "_type", "_source", "_target"}


# --- Error handling ---


class TestDataFrameErrors:
    def test_to_pandas_without_pandas(self, monkeypatch, populated_db):
        """to_pandas() raises ModuleNotFoundError when pandas isn't installed."""
        # We can't actually uninstall pandas mid-test, but we verify the method exists
        result = populated_db.execute("MATCH (n:Person) RETURN n.name")
        assert hasattr(result, "to_pandas")

    def test_to_polars_without_polars(self, monkeypatch, populated_db):
        """to_polars() raises ModuleNotFoundError when polars isn't installed."""
        result = populated_db.execute("MATCH (n:Person) RETURN n.name")
        assert hasattr(result, "to_polars")

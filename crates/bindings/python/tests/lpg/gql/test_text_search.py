"""GQL text search integration tests."""

import pytest

try:
    from grafeo import GrafeoDB

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False


@pytest.fixture
def db():
    if not GRAFEO_AVAILABLE:
        pytest.skip("grafeo not installed")
    return GrafeoDB()


@pytest.fixture
def text_fixture(db):
    """Database with text-indexed articles."""
    db.create_node(["Article"], {"title": "Rust graph database engine"})
    db.create_node(["Article"], {"title": "Python machine learning"})
    db.create_node(["Article"], {"title": "Rust systems programming"})
    owner = db.create_index("title", kind="text", label="Article")
    return db, owner


@pytest.fixture
def text_db(text_fixture):
    return text_fixture[0]


class TestTextSearch:
    @pytest.mark.parametrize("minimum", [None, 0, 3, 2**63])
    def test_minimum_survives_rebuild_and_reopen(self, tmp_path, minimum):
        path = str(tmp_path / "tokenizer.grafeo")
        db = GrafeoDB(path)
        db.execute("CREATE (:Doc {body: 'x ox fox'})")
        owner = db.create_index("body", kind="text", label="Doc", min_token_length=minimum)
        with pytest.raises(RuntimeError):
            db.create_index("body", kind="text", label="Doc", min_token_length=minimum)

        def check(database):
            for token in ["x", "ox", "fox"]:
                expected = int(len(token) >= (2 if minimum is None else minimum))
                assert len(database.text_search("Doc", "body", token, k=10)) == expected

        check(db)
        db.rebuild_index(owner)
        check(db)
        db.close()
        db = GrafeoDB(path)
        check(db)
        db.rebuild_index(owner)
        check(db)
        assert db.drop_index(owner) is True
        assert db.drop_index(owner) is False
        with pytest.raises(RuntimeError):
            db.rebuild_index(owner)
        db.close()

    @pytest.mark.parametrize("minimum", [-1, 1.5, "3", True, 2**128])
    def test_invalid_minimum_does_not_create_owner(self, db, minimum):
        with pytest.raises((ValueError, TypeError, OverflowError)):
            db.create_index("body", kind="text", label="Doc", min_token_length=minimum)
        owner = db.create_index("body", kind="text", label="Doc", min_token_length=3)
        assert owner == 0, "invalid requests must not consume owner IDs"
        assert db.drop_index(owner) is True

    @pytest.mark.parametrize("kind", ["property", "btree", "vector"])
    def test_minimum_is_text_only(self, db, kind):
        with pytest.raises(ValueError, match="requires kind='text'"):
            db.create_index("body", kind=kind, min_token_length=0)
        assert db.create_index("unchanged") == 0

    def test_text_search_basic(self, text_db):
        results = text_db.text_search("Article", "title", "Rust", k=10)
        assert len(results) >= 2

    def test_text_search_no_matches(self, text_db):
        results = text_db.text_search("Article", "title", "nonexistentxyz", k=10)
        assert len(results) == 0

    def test_text_search_after_mutation(self, text_db):
        text_db.create_node(["Article"], {"title": "Rust web framework"})
        results = text_db.text_search("Article", "title", "Rust", k=10)
        assert len(results) >= 3

    def test_drop_and_rebuild_text_index(self, text_fixture):
        text_db, owner = text_fixture
        # Search works
        r1 = text_db.text_search("Article", "title", "Rust", k=10)
        assert len(r1) > 0

        # Rebuild retains the same owner; dropping it does not authorize recreation.
        text_db.rebuild_index(owner)
        assert text_db.drop_index(owner) is True
        with pytest.raises(RuntimeError, match=r"(?i)index|owner"):
            text_db.rebuild_index(owner)
        replacement = text_db.create_index("title", kind="text", label="Article")
        assert replacement > owner

        # Search works again
        r2 = text_db.text_search("Article", "title", "Rust", k=10)
        assert len(r2) > 0

    def test_text_search_no_index_error(self, db):
        db.create_node(["Article"], {"title": "test"})
        with pytest.raises(Exception, match=r".+"):
            db.text_search("Article", "title", "test", k=10)

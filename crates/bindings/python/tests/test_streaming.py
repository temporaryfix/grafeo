"""Tests for db.execute_lazy() cursor-based query streaming.

Focus:
- Iterator protocol yields the same rows as db.execute()
- Early break drops the stream cleanly (subsequent queries still work)
- Columns are exposed before iteration starts
- Empty results behave like normal empty iterators
- Non-streamable queries (mutations, ORDER BY, session cmds, EXPLAIN) raise
- GIL is released during chunk pulls: other Python threads make progress
"""

import threading
import time
import asyncio
import gc

import pytest
import grafeo

pytestmark = pytest.mark.gql


def test_async_iterator_keeps_native_result_alive(db):
    async def execute():
        return await db.execute_async("UNWIND [1,2,3] AS x RETURN x")

    result = asyncio.run(execute())
    first = iter(result)
    second = iter(result)
    del result
    gc.collect()
    assert next(first) == [1]
    assert list(second) == [[1], [2], [3]]
    assert list(first) == [[2], [3]]


def test_compound_parameters_survive_eager_and_async_result_conversion(db):
    payload = ["x" * 8192, [1, None], {"key": "value"}]
    assert list(db.execute("RETURN $payload AS payload", {"payload": payload})) == [
        {"payload": payload}
    ]

    async def execute():
        return await db.execute_async("RETURN $payload AS payload", {"payload": payload})

    result = asyncio.run(execute())
    assert list(result) == [[payload]]


@pytest.fixture
def people_db(db):
    """Fresh in-memory DB seeded with five Person nodes."""
    for name, age in (
        ("Alix", 32),
        ("Gus", 28),
        ("Vincent", 45),
        ("Jules", 40),
        ("Mia", 24),
    ):
        db.create_node(["Person"], {"name": name, "age": age})
    return db


def _rows_as_dict_set(result):
    """Convert an iterable of dicts into a frozenset of sorted tuples for set comparison."""
    return frozenset(tuple(sorted(row.items())) for row in result)


def test_streaming_matches_materialized(people_db):
    query = "MATCH (p:Person) RETURN p.name AS name, p.age AS age"
    materialized = list(people_db.execute(query))
    streamed = list(people_db.execute_lazy(query))

    assert len(materialized) == len(streamed) == 5
    assert _rows_as_dict_set(materialized) == _rows_as_dict_set(streamed)


def test_streaming_yields_dicts_with_column_keys(people_db):
    rows = list(people_db.execute_lazy("MATCH (p:Person) RETURN p.name"))
    assert len(rows) == 5
    assert all(isinstance(row, dict) for row in rows)
    assert all("p.name" in row for row in rows)


def test_stream_columns_exposed_before_iteration(people_db):
    stream = people_db.execute_lazy("MATCH (p:Person) RETURN p.name AS name, p.age AS age")
    assert stream.columns == ["name", "age"]


def test_streaming_filter(people_db):
    rows = list(people_db.execute_lazy("MATCH (p:Person) WHERE p.age > 30 RETURN p.name AS name"))
    names = {row["name"] for row in rows}
    assert names == {"Alix", "Vincent", "Jules"}


def test_streaming_empty_result(people_db):
    rows = list(people_db.execute_lazy("MATCH (p:Person) WHERE p.age > 999 RETURN p.name"))
    assert rows == []


def test_streaming_early_break(people_db):
    # Pull one row, break, then make sure subsequent queries still work.
    stream = people_db.execute_lazy("MATCH (p:Person) RETURN p.name")
    first = next(iter(stream))
    assert "p.name" in first
    del stream

    # Subsequent query on the same DB works.
    rows = list(people_db.execute("MATCH (p:Person) RETURN p.name"))
    assert len(rows) == 5


def test_streaming_pre_cancel_is_typed_and_does_not_leak_to_next_stream(db):
    control = grafeo.QueryControl()
    control.cancel()
    with pytest.raises(grafeo.GrafeoError) as exc_info:
        db.execute_lazy("UNWIND range(1, 4) AS x RETURN x", control=control)
    assert exc_info.value.error_code == "GRAFEO-Q007"
    assert list(db.execute_lazy("RETURN 1 AS value")) == [{"value": 1}]


def test_streaming_accepts_parameters_and_limits(people_db):
    stream = people_db.execute_lazy(
        "RETURN $value AS value", {"value": 9}, max_rows=1, max_bytes=1024
    )
    assert list(stream) == [{"value": 9}]


def test_streaming_rejects_mutation(people_db):
    with pytest.raises(Exception) as exc_info:
        people_db.execute_lazy("INSERT (:Person {name: 'Butch'})")
    msg = str(exc_info.value).lower()
    assert "mutat" in msg or "cannot be streamed" in msg or "execute() instead" in msg


def test_streaming_rejects_order_by(people_db):
    with pytest.raises(Exception) as exc_info:
        people_db.execute_lazy("MATCH (p:Person) RETURN p.name AS n ORDER BY n")
    msg = str(exc_info.value).lower()
    assert "push" in msg or "cannot be streamed" in msg


def test_streaming_rejects_session_command(db):
    with pytest.raises(Exception) as exc_info:
        db.execute_lazy("SESSION SET GRAPH analytics")
    assert "session" in str(exc_info.value).lower()


def test_streaming_rejects_explain(people_db):
    with pytest.raises(Exception) as exc_info:
        people_db.execute_lazy("EXPLAIN MATCH (p:Person) RETURN p.name")
    msg = str(exc_info.value).lower()
    assert "explain" in msg or "cannot be streamed" in msg


def test_streaming_repr(people_db):
    stream = people_db.execute_lazy("MATCH (p:Person) RETURN p.name")
    assert "ResultStream" in repr(stream)


def test_concurrent_iteration_does_not_deadlock(people_db):
    """Two threads iterating separate streams must both complete without
    deadlocking. This is the reliably-observable consequence of GIL release
    in __next__: if GIL release were missing, the Rust mutex taken by a
    running __next__ on one thread could stall the other thread indefinitely
    once pyo3 tried to reacquire Python state.

    A stricter "timing-based" GIL-release test (comparing parallel vs serial
    iteration time) is intentionally avoided: on an in-memory store each row
    pull is sub-microsecond, so the per-row detach window is shorter than
    OS thread-switch granularity, and the signal is lost in scheduling noise.
    """
    row_count = 2_000
    for i in range(row_count):
        people_db.create_node(["Widget"], {"index": i})

    results = {"a": 0, "b": 0}
    errors: list[Exception] = []

    def iterate(key: str) -> None:
        try:
            for _row in people_db.execute_lazy("MATCH (w:Widget) RETURN w.index"):
                results[key] += 1
        except Exception as exc:  # noqa: BLE001 - want any failure here
            errors.append(exc)

    threads = [
        threading.Thread(target=iterate, args=("a",), daemon=True),
        threading.Thread(target=iterate, args=("b",), daemon=True),
    ]
    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)
    elapsed = time.monotonic() - start

    assert not errors, f"concurrent iteration failed: {errors}"
    assert all(not t.is_alive() for t in threads), "threads did not finish (deadlock?)"
    assert results == {"a": row_count, "b": row_count}, f"incomplete iteration: {results}"
    # Sanity cap: two threads of 2k rows should finish in far under 30s;
    # blowing past that points at GIL starvation even without strict timing.
    assert elapsed < 20.0, f"concurrent iteration took {elapsed:.2f}s"


def test_close_releases_publication_while_stream_object_remains_alive(people_db):
    stream = people_db.execute_lazy("MATCH (p:Person) RETURN p.name AS name")
    assert "name" in next(stream)
    stream.close()
    stream.close()
    assert list(stream) == []
    # The writer has its own deadline so a retained read publication guard
    # produces a bounded failure instead of hanging the test process.
    people_db.execute(
        "INSERT (:AfterStreamClose {value: 1})",
        control=grafeo.QueryControl(timeout_ms=2000),
    )
    assert stream.columns == ["name"]
    assert list(people_db.execute("MATCH (n:AfterStreamClose) RETURN n.value AS value")) == [{"value": 1}]


def test_context_manager_closes_after_early_exit_and_preserves_body_exception(people_db):
    class BodyFailure(Exception):
        pass

    failure = BodyFailure("body sentinel")
    with pytest.raises(BodyFailure) as error:
        with people_db.execute_lazy("MATCH (p:Person) RETURN p.name AS name") as stream:
            assert "name" in next(stream)
            raise failure
    assert error.value is failure
    assert list(stream) == []
    stream.close()
    people_db.execute("INSERT (:AfterContextClose)", control=grafeo.QueryControl(timeout_ms=2000))


def test_buffered_stream_cancel_is_error_once_and_close_resolution_is_repeatable(db):
    control = grafeo.QueryControl()
    stream = db.execute_lazy("UNWIND range(1, 20) AS x RETURN x", control=control)
    assert control.consumed
    assert next(stream) == {"x": 1}
    control.cancel()
    with pytest.raises(grafeo.GrafeoError) as error:
        next(stream)
    assert error.value.error_code == "GRAFEO-Q007"
    assert list(stream) == []
    # The primary cancellation is emitted once; close exposes cleanup only.
    stream.close()
    stream.close()
    assert list(db.execute_lazy("RETURN 2 AS value")) == [{"value": 2}]


def test_python_row_conversion_failure_closes_stream_before_next_pull(db):
    # The resident row fits the native query grant; the per-row Python copy
    # cannot fit the explicitly smaller conversion cap.
    payload = ["x" * 8192, [1, None]]
    stream = db.execute_lazy("RETURN $payload AS payload", {"payload": payload}, max_bytes=1024)
    with pytest.raises(grafeo.GrafeoError) as error:
        next(stream)
    assert error.value.error_code == "GRAFEO-S001"
    assert "Python result conversion" in str(error.value)
    assert list(stream) == []
    stream.close()
    stream.close()
    db.execute("INSERT (:AfterCopyDenial)", control=grafeo.QueryControl(timeout_ms=2000))


def test_stream_row_cap_emits_prefix_then_errors_once_and_releases_query(db):
    stream = db.execute_lazy("UNWIND [1, 2] AS x RETURN x", max_rows=1)
    assert next(stream) == {"x": 1}
    with pytest.raises(grafeo.GrafeoError) as error:
        next(stream)
    assert error.value.error_code == "GRAFEO-S001"
    assert list(stream) == []
    stream.close()
    stream.close()
    db.execute("INSERT (:AfterStreamRowCap)", control=grafeo.QueryControl(timeout_ms=2000))


def test_stream_row_cap_allows_exact_limit_and_empty_zero_limit(db):
    with db.execute_lazy("UNWIND [1] AS x RETURN x", max_rows=1) as exact:
        assert list(exact) == [{"x": 1}]
    with db.execute_lazy("UNWIND [] AS x RETURN x", max_rows=0) as empty:
        assert list(empty) == []
    nonempty = db.execute_lazy("UNWIND [1] AS x RETURN x", max_rows=0)
    with pytest.raises(grafeo.GrafeoError) as error:
        next(nonempty)
    assert error.value.error_code == "GRAFEO-S001"
    assert list(nonempty) == []
    nonempty.close()

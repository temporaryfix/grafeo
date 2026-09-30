"""Public execution-control and result-boundary witnesses."""

import asyncio
import threading
import time

import pytest

import grafeo


pytestmark = pytest.mark.gql


def _assert_code(exc_info, expected):
    assert isinstance(exc_info.value, grafeo.GrafeoError)
    assert exc_info.value.error_code == expected


def test_cancelled_control_is_single_use_and_does_not_poison_next_query(db):
    control = grafeo.QueryControl()
    control.cancel()

    with pytest.raises(grafeo.GrafeoError) as exc_info:
        db.execute("UNWIND range(1, 4) AS x RETURN x", control=control)
    _assert_code(exc_info, "GRAFEO-Q007")
    assert control.consumed
    with pytest.raises(ValueError, match="consumed"):
        db.execute("RETURN 1 AS value", control=control)
    assert list(db.execute("RETURN 1 AS value")) == [{"value": 1}]


def test_zero_deadline_is_typed_timeout(db):
    with pytest.raises(grafeo.GrafeoError) as exc_info:
        db.execute("RETURN 1 AS value", control=grafeo.QueryControl(timeout_ms=0))
    _assert_code(exc_info, "GRAFEO-Q003")


def test_row_and_byte_caps_fail_before_mutation_commit(db):
    with pytest.raises(grafeo.GrafeoError) as rows_error:
        db.execute(
            "INSERT (n:Bounded {value: 'row'}) RETURN n",
            max_rows=0,
        )
    _assert_code(rows_error, "GRAFEO-S001")
    assert list(db.execute("MATCH (n:Bounded) RETURN count(n) AS count")) == [
        {"count": 0}
    ]

    with pytest.raises(grafeo.GrafeoError) as bytes_error:
        db.execute("RETURN 'a large bounded value' AS value", max_bytes=1)
    _assert_code(bytes_error, "GRAFEO-S001")


def test_parameters_and_controls_reach_eager_lazy_and_async_routes(db):
    params = {"value": 17}
    assert list(
        db.execute("RETURN $value AS value", params, max_rows=1, max_bytes=16384)
    ) == [{"value": 17}]
    assert list(
        db.execute_lazy("RETURN $value AS value", params, max_rows=1, max_bytes=16384)
    ) == [{"value": 17}]

    async def run():
        return await db.execute_async(
            "RETURN $value AS value",
            params,
            control=grafeo.QueryControl(),
            max_rows=1,
            max_bytes=16384,
        )

    assert list(asyncio.run(run())) == [[17]]


def test_async_order_by_preserves_hidden_keys_multiplicity_and_parameters(db):
    for rank, name in ((5, "b"), (3, "a"), (1, "a"), (4, "c"), (2, "b")):
        db.create_node(["AsyncOrdered"], {"rank": rank, "name": name})
    query = (
        "MATCH (n:AsyncOrdered) WHERE n.rank >= $floor "
        "RETURN n.name AS name ORDER BY n.rank, n.name"
    )

    async def run():
        return await db.execute_async(
            query, {"floor": 1}, max_rows=5, max_bytes=131072
        )

    result = asyncio.run(run())
    assert result.columns == ["name"]
    assert list(result) == [["a"], ["b"], ["a"], ["c"], ["b"]]
    assert list(db.execute(query, {"floor": 1})) == [
        {"name": name} for name in ("a", "b", "a", "c", "b")
    ]


def test_async_order_by_keeps_native_and_python_result_limits_and_cancel(db):
    db.create_node(["AsyncBounded"], {"rank": 1})
    payload = _nested_copy_payload()
    query = "MATCH (n:AsyncBounded) RETURN $payload AS value ORDER BY n.rank"

    async def run():
        for limits in ({"max_rows": 0}, {"max_bytes": 1}, {"max_bytes": 8192}):
            with pytest.raises(grafeo.GrafeoError) as error:
                await db.execute_async(query, {"payload": payload}, **limits)
            _assert_code(error, "GRAFEO-S001")
            if limits.get("max_bytes") == 8192:
                assert "Python result conversion" in str(error.value)
        control = grafeo.QueryControl()
        control.cancel()
        with pytest.raises(grafeo.GrafeoError) as error:
            await db.execute_async(query, {"payload": payload}, control=control)
        _assert_code(error, "GRAFEO-Q007")
        assert control.consumed
        result = await db.execute_async(
            query, {"payload": payload}, max_rows=1, max_bytes=131072
        )
        assert list(result) == [[payload]]

    asyncio.run(run())


def test_async_fallback_preserves_mutations_and_selected_graph(db):
    async def run():
        await db.execute_async("CREATE GRAPH async_order_context")
        await db.execute_async("USE GRAPH async_order_context")
        await db.execute_async("INSERT (:AsyncContext {rank: 2}), (:AsyncContext {rank: 1})")
        query = "MATCH (n:AsyncContext) RETURN n.rank AS rank ORDER BY n.rank"
        assert list(await db.execute_async(query)) == [[1], [2]]
        assert list(db.execute(query)) == [{"rank": 1}, {"rank": 2}]
        await db.execute_async("USE GRAPH DEFAULT")
        assert list(await db.execute_async(query)) == []

    asyncio.run(run())


def test_explicit_language_routes_preserve_control_errors(db):
    for method_name, query in (
        ("execute_cypher", "RETURN 1 AS value"),
        ("execute_sparql", "SELECT * WHERE { VALUES ?value { 1 } }"),
    ):
        method = getattr(db, method_name, None)
        if method is None:
            continue
        control = grafeo.QueryControl()
        control.cancel()
        with pytest.raises(grafeo.GrafeoError) as exc_info:
            method(query, control=control, max_rows=1, max_bytes=16384)
        _assert_code(exc_info, "GRAFEO-Q007")


def test_transaction_route_accepts_control_and_caps(db):
    with db.begin_transaction() as tx:
        assert list(
            tx.execute(
                "RETURN $value AS value",
                {"value": 23},
                control=grafeo.QueryControl(),
                max_rows=1,
                max_bytes=16384,
            )
        ) == [{"value": 23}]


_BUSY_QUERY = (
    "MATCH (a:CancelWork), (b:CancelWork), (c:CancelWork), (d:CancelWork) "
    "WHERE a.i + b.i + c.i + d.i < 0 RETURN count(*) AS count"
)


def _seed_cancel_work(db):
    for index in range(128):
        db.create_node(["CancelWork"], {"i": index})


def test_eager_inflight_cancel_releases_gil_and_keeps_control_single_use(db):
    _seed_cancel_work(db)
    control = grafeo.QueryControl(timeout_ms=10000)
    entered = threading.Event()
    outcomes = []

    def execute():
        entered.set()
        try:
            outcomes.append(db.execute(_BUSY_QUERY, control=control))
        except Exception as error:
            outcomes.append(error)

    worker = threading.Thread(target=execute, daemon=True)
    worker.start()
    try:
        assert entered.wait(2)
        deadline = time.monotonic() + 2
        while not control.consumed and time.monotonic() < deadline:
            threading.Event().wait(0.001)
        assert control.consumed
        # If native execute holds the GIL, this thread cannot reach cancellation
        # until the native deadline returns. The query deliberately examines a
        # large Cartesian product while retaining only one aggregate row.
        threading.Event().wait(0.02)
        assert worker.is_alive(), "execution completed before an in-flight cancel was possible"
    finally:
        control.cancel()
        worker.join(12)
    assert not worker.is_alive()
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], grafeo.GrafeoError)
    assert outcomes[0].error_code == "GRAFEO-Q007"
    with pytest.raises(ValueError, match="consumed"):
        db.execute("RETURN 1 AS value", control=control)
    assert list(db.execute("RETURN 1 AS value")) == [{"value": 1}]



def test_deadline_interrupts_the_rejected_cartesian_input_loop(db):
    _seed_cancel_work(db)
    started = time.monotonic()
    with pytest.raises(grafeo.GrafeoError) as error:
        db.execute(_BUSY_QUERY, control=grafeo.QueryControl(timeout_ms=50))
    _assert_code(error, "GRAFEO-Q003")
    assert time.monotonic() - started < 3, "deadline was deferred until Cartesian input exhaustion"

def test_async_controls_cancel_independently_during_native_execution(db):
    _seed_cancel_work(db)

    async def run():
        first_control = grafeo.QueryControl(timeout_ms=10000)
        second_control = grafeo.QueryControl(timeout_ms=10000)
        first = asyncio.ensure_future(db.execute_async(_BUSY_QUERY, control=first_control))
        second = asyncio.ensure_future(db.execute_async(_BUSY_QUERY, control=second_control))
        try:
            await asyncio.sleep(0.03)
            assert first_control.consumed and second_control.consumed
            assert not first.done() and not second.done()
            first_control.cancel()
            with pytest.raises(grafeo.GrafeoError) as error:
                await asyncio.wait_for(first, 12)
            _assert_code(error, "GRAFEO-Q007")
            assert not second.done(), "cancelling the first query affected an independent owner"
            second_control.cancel()
            with pytest.raises(grafeo.GrafeoError) as error:
                await asyncio.wait_for(second, 12)
            _assert_code(error, "GRAFEO-Q007")
        finally:
            first_control.cancel()
            second_control.cancel()
            await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 12)
        result = await db.execute_async("RETURN 17 AS value", control=grafeo.QueryControl())
        assert list(result) == [[17]]

    asyncio.run(run())


def test_python_copy_only_denial_rolls_back_mutation_and_preserves_prior_tx_work(db):
    # One integer easily fits the native row cap. The error must specifically
    # come from the larger Python copied-output envelope, before publication.
    query = "INSERT (n:CopyDenied {value: 7}) RETURN n.value AS value"
    with pytest.raises(grafeo.GrafeoError) as error:
        db.execute(query, max_bytes=512)
    _assert_code(error, "GRAFEO-S001")
    assert "Python result conversion" in str(error.value)
    assert list(db.execute("MATCH (n:CopyDenied) RETURN count(n) AS count")) == [{"count": 0}]
    with db.begin_transaction() as tx:
        tx.execute("INSERT (:PriorCopyWork {value: 1})")
        with pytest.raises(grafeo.GrafeoError) as error:
            tx.execute(query, max_bytes=512)
        _assert_code(error, "GRAFEO-S001")
        assert "Python result conversion" in str(error.value)
        assert list(tx.execute("MATCH (n:CopyDenied) RETURN count(n) AS count")) == [{"count": 0}]
        assert list(tx.execute("MATCH (n:PriorCopyWork) RETURN n.value AS value")) == [{"value": 1}]
    assert list(db.execute("MATCH (n:PriorCopyWork) RETURN n.value AS value")) == [{"value": 1}]


def test_historical_query_keeps_snapshot_parameters_and_control(db):
    db.execute("INSERT (:ControlledHistory {value: 1})")
    epoch = db.current_epoch()
    db.execute("MATCH (n:ControlledHistory) SET n.value = 2")
    query = "MATCH (n:ControlledHistory) RETURN n.value + $offset AS value"
    control = grafeo.QueryControl()
    assert list(db.execute_at_epoch(query, epoch, {"offset": 10}, control=control, max_bytes=16384)) == [{"value": 11}]
    assert control.consumed
    assert list(db.execute(query, {"offset": 10})) == [{"value": 12}]
    cancelled = grafeo.QueryControl()
    cancelled.cancel()
    with pytest.raises(grafeo.GrafeoError) as error:
        db.execute_at_epoch(query, epoch, {"offset": 10}, control=cancelled)
    _assert_code(error, "GRAFEO-Q007")


@pytest.mark.parametrize("timeout", [-1, "bad", 1 << 100])
def test_invalid_control_timeout_is_rejected(timeout):
    with pytest.raises((ValueError, TypeError, OverflowError)):
        grafeo.QueryControl(timeout_ms=timeout)


def _nested_copy_payload():
    # Just one KiB of UTF-8 text; nested containers and the Python Unicode copy
    # exceed the small copied-output envelope while the native result fits.
    return {"outer": [None, {"text": "😀" * 256}, []], "empty": {}}


def test_nested_unicode_copy_cap_and_eager_materializers(db):
    payload = _nested_copy_payload()
    query = "RETURN $payload AS value"
    with pytest.raises(grafeo.GrafeoError) as error:
        db.execute(query, {"payload": payload}, max_bytes=8192)
    _assert_code(error, "GRAFEO-S001")
    assert "Python result conversion" in str(error.value)

    result = db.execute(query, {"payload": payload}, max_bytes=131072)
    assert result.columns == ["value"]
    assert result.scalar() == payload
    assert result.column("value") == [payload]
    assert result.column(0) == [payload]
    assert result.to_list() == [{"value": payload}]
    assert result[0] == {"value": payload}
    assert list(result) == [{"value": payload}]
    # Each materializer creates an independent copy of nested containers.
    copied = result.scalar()
    copied["outer"][1]["text"] = "changed"
    assert result.scalar() == payload
    assert result.nodes() == []
    assert result.edges() == []


def test_nested_unicode_copy_cap_and_async_rows_getters(db):
    payload = _nested_copy_payload()

    async def run():
        with pytest.raises(grafeo.GrafeoError) as error:
            await db.execute_async(
                "RETURN $payload AS value", {"payload": payload}, max_bytes=8192
            )
        _assert_code(error, "GRAFEO-S001")
        assert "Python result conversion" in str(error.value)
        return await db.execute_async(
            "RETURN $payload AS value", {"payload": payload}, max_bytes=131072
        )

    result = asyncio.run(run())
    assert result.columns == ["value"]
    assert result.rows() == [[payload]]
    assert list(result) == [[payload]]
    copied = result.rows()
    copied[0][0]["outer"].append("changed")
    assert result.rows() == [[payload]]
    assert result.nodes() == []
    assert result.edges() == []


def test_entity_copies_keep_selected_cap_on_eager_and_async_results(db):
    payload = "雪" * 256
    source = db.create_node(["CopySource"], {"payload": payload})
    target = db.create_node(["CopyTarget"])
    edge = db.create_edge(source.id, target.id, "COPY_EDGE", {"payload": payload})
    query = "MATCH (n:CopySource)-[r:COPY_EDGE]->() RETURN n AS node, r AS edge"

    with pytest.raises(grafeo.GrafeoError) as error:
        db.execute(query, max_bytes=16384)
    _assert_code(error, "GRAFEO-S001")
    assert "Python result conversion" in str(error.value)

    result = db.execute(query, max_bytes=131072)
    assert result.columns == ["node", "edge"]
    assert [node.id for node in result.nodes()] == [source.id]
    assert [item.id for item in result.edges()] == [edge.id]
    assert result.nodes()[0].properties()["payload"] == payload
    assert result.edges()[0].properties()["payload"] == payload
    assert result.to_list()[0]["node"]["payload"] == payload
    assert result.column("edge")[0]["payload"] == payload

    async def run():
        with pytest.raises(grafeo.GrafeoError) as error:
            await db.execute_async(query, max_bytes=16384)
        _assert_code(error, "GRAFEO-S001")
        assert "Python result conversion" in str(error.value)
        return await db.execute_async(query, max_bytes=131072)

    result = asyncio.run(run())
    assert result.columns == ["node", "edge"]
    assert [node.id for node in result.nodes()] == [source.id]
    assert [item.id for item in result.edges()] == [edge.id]
    assert result.nodes()[0].properties()["payload"] == payload
    assert result.edges()[0].properties()["payload"] == payload
    assert result.rows()[0][0]["payload"] == payload
    assert result.rows()[0][1]["payload"] == payload


def test_arrow_exports_admit_buffers_and_preserve_exact_rows(db):
    import pyarrow as pa

    query = "UNWIND range(1, 3) AS value RETURN value"
    small = db.execute(query, max_bytes=16384)
    assert small.scalar() == 1
    assert small.column("value") == [1, 2, 3]
    for export in (small.to_arrow_ipc, small.to_arrow):
        with pytest.raises(grafeo.GrafeoError) as error:
            export()
        _assert_code(error, "GRAFEO-S001")
        assert "Python result conversion" in str(error.value)
    assert small.column("value") == [1, 2, 3]

    result = db.execute(query, max_bytes=131072)
    expected = [{"value": 1}, {"value": 2}, {"value": 3}]
    encoded = result.to_arrow_ipc()
    assert isinstance(encoded, bytes)
    assert pa.ipc.open_stream(encoded).read_all().to_pylist() == expected
    assert result.to_arrow().to_pylist() == expected


def test_text_exports_admit_escaping_and_preserve_unicode(db):
    params = {
        "subject": "urn:copy:subject",
        "predicate": "urn:copy:predicate",
        "object": '雪"\\\n' * 16,
    }
    query = "RETURN $subject AS s, $predicate AS p, $object AS o"
    expected_row = {"s": params["subject"], "p": params["predicate"], "o": params["object"]}
    small = db.execute(query, params, max_bytes=8192)
    assert small.to_list() == [expected_row]
    for export in (small.to_ntriples, small.to_turtle):
        with pytest.raises(grafeo.GrafeoError) as error:
            export()
        _assert_code(error, "GRAFEO-S001")
        assert "Python result conversion" in str(error.value)
    assert small.to_list() == [expected_row]

    result = db.execute(query, params, max_bytes=131072)
    escaped = '雪\\"\\\\\\n' * 16
    triple = '<urn:copy:subject> <urn:copy:predicate> "' + escaped + '" .\n'
    assert result.to_ntriples() == triple
    assert result.to_turtle() == triple + "\n"


def test_async_dataframe_exports_preserve_typed_rows(db):
    async def run():
        return await db.execute_async(
            "UNWIND range(1, 3) AS value RETURN value, $label AS label",
            {"label": "雪"},
            max_bytes=131072,
        )

    result = asyncio.run(run())
    expected = [
        {"value": 1, "label": "雪"},
        {"value": 2, "label": "雪"},
        {"value": 3, "label": "雪"},
    ]
    assert result.to_pandas().to_dict(orient="records") == expected
    assert result.to_polars().to_dicts() == expected

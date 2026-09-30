using Xunit;

namespace Grafeo.Tests;

/// <summary>Bounded execution ownership, cancellation, and stream lifecycle tests.</summary>
public sealed class ExecutionOwnershipTests
{
    private static readonly string BusyQuery =
        "MATCH (a:CancelWork), (b:CancelWork), (c:CancelWork), (d:CancelWork) " +
        "WHERE a.i + b.i + c.i + d.i < 0 RETURN a.i";

    private static void SeedBusyQuery(GrafeoDB db)
    {
        db.Execute("UNWIND range(1, 128) AS i INSERT (:CancelWork {i: i})");
    }

    [Fact]
    public void PreCancelledControlDoesNotMutateAndIsSingleUse()
    {
        using var db = GrafeoDB.Memory();
        using var control = new QueryControl();
        control.Cancel();

        var error = Assert.Throws<QueryCanceledException>(() => db.ExecuteWithOptions(
            "INSERT (:Cancelled {i: 1}) RETURN 1", new ExecutionOptions { Control = control }));
        Assert.Equal("GRAFEO-Q007", error.Code);
        Assert.Empty(db.Execute("MATCH (n:Cancelled) RETURN n").Rows);
        Assert.Throws<InvalidOperationException>(() => db.ExecuteWithOptions("RETURN 1",
            new ExecutionOptions { Control = control }));

        using var fresh = new QueryControl();
        Assert.Single(db.ExecuteWithOptions("RETURN 1", new ExecutionOptions { Control = fresh }).Rows);
    }

    [Fact]
    public void ZeroTimeoutIsAnImmediateTypedDeadline()
    {
        using var db = GrafeoDB.Memory();
        using var control = new QueryControl(TimeSpan.Zero);

        var error = Assert.Throws<QueryException>(() => db.ExecuteWithOptions(
            "RETURN 1", new ExecutionOptions { Control = control }));
        Assert.Equal("GRAFEO-Q003", error.Code);
    }

    [Fact]
    public async Task CancellationTokenCancelsAnActiveNativeQuery()
    {
        using var db = GrafeoDB.Memory();
        SeedBusyQuery(db);
        using var source = new CancellationTokenSource();
        var running = db.ExecuteWithOptionsAsync(BusyQuery, cancellationToken: source.Token);
        Assert.ThrowsAny<GrafeoException>(() => db.Dispose());
        await Task.Delay(20);
        source.Cancel();

        var error = await Assert.ThrowsAsync<QueryCanceledException>(() => running.WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.Equal("GRAFEO-Q007", error.Code);
    }

    [Fact]
    public async Task DisposingConsumedControlDoesNotBreakInvocation()
    {
        using var db = GrafeoDB.Memory();
        SeedBusyQuery(db);
        using var source = new CancellationTokenSource();
        var control = new QueryControl();
        var running = db.ExecuteWithOptionsAsync(BusyQuery,
            new ExecutionOptions { Control = control }, cancellationToken: source.Token);
        await Task.Delay(20);
        control.Dispose();
        source.Cancel();

        var error = await Assert.ThrowsAsync<QueryCanceledException>(() => running.WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.Equal("GRAFEO-Q007", error.Code);
    }

    [Fact]
    public void LimitsRollbackDeniedStatementAndPreservePriorTransactionWrite()
    {
        using var db = GrafeoDB.Memory();
        using var tx = db.BeginTransaction();
        tx.Execute("INSERT (:Kept {i: 1})");

        var error = Assert.Throws<StorageException>(() => tx.ExecuteWithOptions(
            "INSERT (:Denied {payload: 'large'}) RETURN 1",
            new ExecutionOptions { MaxBytes = 1 }));
        Assert.Equal(GrafeoStatus.ResourceLimit, error.Status);
        Assert.Equal("GRAFEO-S001", error.Code);
        tx.Commit();

        Assert.Single(db.Execute("MATCH (n:Kept) RETURN n").Rows);
        Assert.Empty(db.Execute("MATCH (n:Denied) RETURN n").Rows);
    }

    [Fact]
    public void StreamChunksAllRowsAndCloseIsIdempotent()
    {
        using var db = GrafeoDB.Memory();
        using var stream = db.ExecuteStreamWithOptions(
            "UNWIND range(1, 2500) AS i RETURN i");
        var seen = new HashSet<long>();
        var count = 0;
        object? first = null;
        object? last = null;
        for (QueryResult? chunk; (chunk = stream.NextChunk(257)) is not null;)
        {
            Assert.InRange(chunk.Rows.Count, 1, 257);
            foreach (var row in chunk.Rows) Assert.True(seen.Add(Assert.IsType<long>(row["i"])));
            first ??= chunk.Rows[0]["i"];
            last = chunk.Rows[^1]["i"];
            count += chunk.Rows.Count;
        }

        Assert.Equal(2500, count);
        Assert.Equal(1L, first);
        Assert.Equal(2500L, last);
        stream.Close();
        stream.Close();
    }

    [Fact]
    public async Task IndependentStreamsCancelInIsolation()
    {
        using var db = GrafeoDB.Memory();
        SeedBusyQuery(db);
        using var firstControl = new QueryControl();
        using var secondControl = new QueryControl();
        var first = db.ExecuteStreamWithOptions(BusyQuery,
            new ExecutionOptions { Control = firstControl });
        using var second = db.ExecuteStreamWithOptions(
            "UNWIND range(1, 2500) AS i RETURN i",
            new ExecutionOptions { Control = secondControl });

        firstControl.Cancel();
        var error = Assert.Throws<QueryCanceledException>(() => first.Next());
        Assert.Equal("GRAFEO-Q007", error.Code);
        Assert.Throws<QueryCanceledException>(() => first.Close());
        Assert.Throws<QueryCanceledException>(() => first.Close());
        var count = 0;
        await foreach (var _ in second.RowsAsync())
            count++;
        Assert.Equal(2500, count);
        try
        {
            first.Dispose();
        }
        catch (QueryCanceledException)
        {
            // Dispose preserves the sticky cancellation failure after Close.
        }
    }
    [Fact]
    public void NativeAdmittedDeepResultDecodesAfterMutationAndInStream()
    {
        using var db = GrafeoDB.Memory();
        var nested = new string('[', 80) + "1" + new string(']', 80);
        var result = db.ExecuteWithOptions("INSERT (:DeepCopy) RETURN " + nested + " AS payload");
        object? value = Assert.Single(result.Rows)["payload"];
        for (var depth = 0; depth < 80; depth++)
            value = Assert.Single(Assert.IsAssignableFrom<IReadOnlyList<object?>>(value));
        Assert.Equal(1L, value);
        Assert.Single(db.Execute("MATCH (n:DeepCopy) RETURN n").Rows);
        using var stream = db.ExecuteStream("RETURN " + nested + " AS payload");
        var element = Assert.IsType<System.Text.Json.JsonElement>(stream.Next()!["payload"]);
        for (var depth = 0; depth < 80; depth++) element = element[0];
        Assert.Equal(1, element.GetInt32());
    }

    [Fact]
    public void LiveTransactionAndStreamRejectParentDisposalAndReleaseForRetry()
    {
        using var db = GrafeoDB.Memory();
        using (var tx = db.BeginTransaction())
        {
            Assert.ThrowsAny<GrafeoException>(() => db.Dispose());
            tx.Execute("INSERT (:Kept)");
            tx.Commit();
        }
        using (var stream = db.ExecuteStream("UNWIND range(1, 2500) AS i RETURN i"))
        {
            Assert.ThrowsAny<GrafeoException>(() => db.Dispose());
            foreach (var row in stream.Rows()) break;
            Assert.Throws<ObjectDisposedException>(() => stream.Next());
        }
        Assert.Single(db.Execute("MATCH (n:Kept) RETURN n").Rows);
        db.Dispose();
    }

    [Fact]
    public void CollectionByteLimitIsStickyAndReleasesCursor()
    {
        using var db = GrafeoDB.Memory();
        var stream = db.ExecuteStreamWithOptions("UNWIND range(1, 2500) AS i RETURN i",
            new ExecutionOptions { MaxBytes = 65536 });
        var error = Assert.Throws<StorageException>(() => stream.ToList());
        Assert.Equal("GRAFEO-S001", error.Code);
        Assert.Same(error, Assert.Throws<StorageException>(() => stream.Next()));
        Assert.Same(error, Assert.Throws<StorageException>(() => stream.Close()));
        db.Dispose();
    }
    [Fact]
    public async Task TransactionCancellationReleasesQueuedLeaseAndKeepsEarlierWrite()
    {
        using var db = GrafeoDB.Memory();
        SeedBusyQuery(db);
        using var tx = db.BeginTransaction();
        tx.Execute("INSERT (:Earlier)");
        using var source = new CancellationTokenSource();
        var running = tx.ExecuteWithOptionsAsync(BusyQuery, cancellationToken: source.Token);
        Assert.Throws<GrafeoException>(() => tx.Dispose());
        Assert.Throws<GrafeoException>(() => tx.Commit());
        source.Cancel();
        await Assert.ThrowsAsync<QueryCanceledException>(() => running.WaitAsync(TimeSpan.FromSeconds(10)));
        tx.Commit();
        Assert.Single(db.Execute("MATCH (n:Earlier) RETURN n").Rows);
    }

    [Fact]
    public async Task AsyncEnumeratorBreakClosesCursor()
    {
        using var db = GrafeoDB.Memory();
        await using var stream = await db.ExecuteStreamWithOptionsAsync("UNWIND range(1, 2500) AS i RETURN i");
        await foreach (var row in stream.RowsAsync()) break;
        Assert.Throws<ObjectDisposedException>(() => stream.Next());
        db.Dispose();
    }
    [Fact]
    public void ShapeAndBudgetBoundariesNeverDenyCopiesAfterCommit()
    {
        string[] shapes = ["1", "{}", "[]", "{a: [1, 2, 3], b: {c: 'text'}}",
            "'" + new string('x', 1000) + "'", new string('[', 80) + "1" + new string(']', 80)];
        ulong[] limits = [256, 1024, 4096, 16384, 65536, 1048576];
        var admitted = 0;
        var denied = 0;
        foreach (var shape in shapes)
        foreach (var limit in limits)
        {
            using var db = GrafeoDB.Memory();
            try
            {
                Assert.Single(db.ExecuteWithOptions("INSERT (:BudgetProbe) RETURN " + shape + " AS value",
                    new ExecutionOptions { MaxBytes = limit }).Rows);
                Assert.Single(db.Execute("MATCH (n:BudgetProbe) RETURN n").Rows);
                admitted++;
            }
            catch (StorageException error)
            {
                Assert.Equal("GRAFEO-S001", error.Code);
                Assert.Empty(db.Execute("MATCH (n:BudgetProbe) RETURN n").Rows);
                denied++;
            }
        }
        Assert.True(admitted > 0);
        Assert.True(denied > 0);
    }
    [Fact]
    public void TemporalMarkerMapsAndExtremeTimestampsCannotFailAfterCommit()
    {
        using var db = GrafeoDB.Memory();
        Dictionary<string, object?>[] payloads =
        [
            new() { ["$date"] = 1L }, new() { ["$time"] = false },
            new() { ["$duration"] = new List<object?> { 1L } },
            new() { ["$zoned_datetime"] = 2L },
            new() { ["$timestamp_us"] = long.MaxValue },
            new() { ["$timestamp_us"] = long.MinValue },
            new() { ["$timestamp_us"] = "ordinary" },
            new() { ["$date"] = 1L, ["other"] = 2L },
        ];
        foreach (var payload in payloads)
        {
            var result = db.ExecuteWithOptions("INSERT (:TemporalProbe) RETURN $payload AS value",
                parameters: new() { ["payload"] = payload });
            var actual = Assert.IsAssignableFrom<IReadOnlyDictionary<string, object?>>(Assert.Single(result.Rows)["value"]);
            Assert.Equal(payload.Keys, actual.Keys);
            foreach (var (key, value) in payload) Assert.Equal(value, actual[key]);
        }
        Assert.Equal(payloads.Length, db.Execute("MATCH (n:TemporalProbe) RETURN n").Rows.Count);
        var exact = db.ExecuteWithOptions("RETURN $value AS value", parameters: new()
        { ["value"] = new Dictionary<string, object?> { ["$timestamp_us"] = 1234567L } });
        Assert.Equal(DateTime.UnixEpoch.AddTicks(12345670), Assert.Single(exact.Rows)["value"]);
        foreach (var microseconds in new[] { -62135596800000000L, 253402300799999999L })
        {
            var boundary = db.ExecuteWithOptions("RETURN $value AS value", parameters: new()
            { ["value"] = new Dictionary<string, object?> { ["$timestamp_us"] = microseconds } });
            Assert.Equal(DateTime.UnixEpoch.AddTicks(microseconds * 10), Assert.Single(boundary.Rows)["value"]);
        }
    }
}

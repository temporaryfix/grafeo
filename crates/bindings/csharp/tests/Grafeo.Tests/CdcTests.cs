using System.Text.Json;
using Xunit;

namespace Grafeo.Tests;

public sealed class CdcTests
{
    [RetainedCutFact]
    public void RetainedCutPreservesStaleCursorInEveryPageReader()
    {
        var fixture = Environment.GetEnvironmentVariable("GRAFEO_CDC_EVICTED_FIXTURE")!;
        var cursor = File.ReadAllBytes(Path.ChangeExtension(fixture, ".cursor"));
        Assert.Equal(97, cursor.Length);
        var directory = Path.Combine(Path.GetTempPath(), "grafeo-retained-csharp-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            var path = Path.Combine(directory, "retained.grafeo");
            File.Copy(fixture, path);
            using var db = GrafeoDB.Open(path);
            foreach (var read in new Func<ChangePage>[] {
                () => db.ChangesAfter(cursor, 1, 4096),
                () => db.NodeHistoryAfter(0, 0, cursor, 1, 4096),
                () => db.EdgeHistoryAfter(0, 0, cursor, 1, 4096),
            })
                Assert.Equal("GRAFEO-S006", Assert.ThrowsAny<GrafeoException>(() => read()).Code);
            Assert.Single(db.ChangesAfter(null, 1, 4096).Events);
        }
        finally { Directory.Delete(directory, true); }
    }

    [Fact]
    public void OwnedPagesCarryCompleteCreationAndEntityHistory()
    {
        var db = GrafeoDB.Memory();
        db.CdcEnabled = true;
        Assert.True(db.CdcEnabled);
        var a = db.CreateNode(["N"], new() { ["large"] = 9007199254740993L });
        var b = db.CreateNode(["N"]);
        var edge = db.CreateEdge(a, b, "LINK");
        var pages = new List<ChangePage>();
        byte[]? cursor = null;
        foreach (var id in new[] { a, b, edge })
        {
            var page = db.ChangesAfter(cursor, 1, 4096);
            Assert.Equal((ulong)id, Assert.Single(page.Events).EntityId);
            Assert.Equal(97, page.Next.Length);
            pages.Add(page);
            cursor = page.Next;
        }
        var eof = db.ChangesAfter(cursor, 1, 4096);
        Assert.Empty(eof.Events);
        Assert.Equal(cursor, eof.Next);
        var node = Assert.Single(db.NodeHistoryAfter((ulong)a, 0, null, 1, 4096).Events);
        Assert.Equal(node.Epoch, Assert.Single(db.NodeHistoryAfter((ulong)a, node.Epoch, null, 1, 4096).Events).Epoch);
        Assert.Empty(db.NodeHistoryAfter((ulong)a, node.Epoch + 1, null, 1, 4096).Events);
        Assert.Empty(db.NodeHistoryAfter((ulong)a, ulong.MaxValue, null, 1, 4096).Events);
        Assert.Equal("edge", Assert.Single(db.EdgeHistoryAfter((ulong)edge, 0, null, 1, 4096).Events).EntityType);
        Assert.Empty(db.NodeHistoryAfter(ulong.MaxValue - 1, 0, null, 1, 4096).Events);
        db.Dispose();
        Assert.Equal(new[] { "N" }, pages[0].Events[0].Labels);
        Assert.Equal(9007199254740993L, pages[0].Events[0].After.GetProperty("large").GetInt64());
        Assert.Equal("LINK", pages[2].Events[0].EdgeType);
        Assert.Equal((ulong)a, pages[2].Events[0].SourceId);
        Assert.Equal((ulong)b, pages[2].Events[0].TargetId);
        Assert.NotNull(pages[0].Events[0].GraphIncarnation);
        Assert.Empty(pages[0].Events[0].LpgGraph!);
        Assert.Throws<ObjectDisposedException>(() => db.ChangesAfter(null, 1, 4096));
    }

    [Fact]
    public void InvalidBoundsAndCursorsRetainNativeCodes()
    {
        using var db = GrafeoDB.Memory();
        db.CdcEnabled = true;
        db.CreateNode(["N"]);
        foreach (var cursor in new[] { Array.Empty<byte>(), new byte[96], new byte[97], new byte[98] })
            Assert.Equal("GRAFEO-S004", Assert.ThrowsAny<GrafeoException>(() => db.ChangesAfter(cursor, 1, 4096)).Code);
        foreach (var bounds in new[] { (0, 4096), (-1, 4096), (1, 0), (1, -1) })
            Assert.Equal("GRAFEO-V001", Assert.ThrowsAny<GrafeoException>(() => db.ChangesAfter(null, bounds.Item1, bounds.Item2)).Code);
        Assert.Equal("GRAFEO-S001", Assert.ThrowsAny<GrafeoException>(() => db.ChangesAfter(null, 1, 1)).Code);
        using var other = GrafeoDB.Memory();
        other.CdcEnabled = true;
        var foreign = other.ChangesAfter(null, 1, 4096).Next;
        Assert.Equal("GRAFEO-S005", Assert.ThrowsAny<GrafeoException>(() => db.ChangesAfter(foreign, 1, 4096)).Code);
        Assert.Single(db.ChangesAfter(null, 1, 4096).Events);
    }

    [Fact]
    public void PageOneResumesAcrossTwoDurableReopens()
    {
        var directory = Path.Combine(Path.GetTempPath(), "grafeo-cdc-csharp-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, "store");
        try
        {
            byte[] cursor;
            ulong[] ids;
            using (var db = GrafeoDB.Open(path))
            {
                db.CdcEnabled = true;
                ids = Enumerable.Range(0, 3).Select(_ => (ulong)db.CreateNode(["N"])).ToArray();
                var first = db.ChangesAfter(null, 1, 4096);
                Assert.Equal(ids[0], Assert.Single(first.Events).EntityId);
                cursor = first.Next;
            }
            for (var i = 1; i < 3; i++)
            {
                using var db = GrafeoDB.Open(path);
                var page = db.ChangesAfter(cursor, 1, 4096);
                Assert.Equal(ids[i], Assert.Single(page.Events).EntityId);
                cursor = page.Next;
                if (i == 2)
                {
                    var eof = db.ChangesAfter(cursor, 1, 4096);
                    Assert.Empty(eof.Events);
                    Assert.Equal(cursor, eof.Next);
                }
            }
        }
        finally { Directory.Delete(directory, true); }
    }

    [Fact]
    public void JsonCoordinatesRetainUnsignedMaximum()
    {
        const string json = """
            {"entity_id":"18446744073709551615","epoch":"18446744073709551614",
             "timestamp":"18446744073709551613","graph_incarnation":"18446744073709551612",
             "src_id":"18446744073709551611","dst_id":"18446744073709551610",
             "triple_graph":"<urn:g>","triple_subject":"<urn:s>","triple_predicate":"<urn:p>",
             "triple_object":"\"value\""}
            """;
        var ev = JsonSerializer.Deserialize<ChangeEvent>(json)!;
        Assert.Equal(ulong.MaxValue, ev.EntityId);
        Assert.Equal(ulong.MaxValue - 1, ev.Epoch);
        Assert.Equal(ulong.MaxValue - 2, ev.Timestamp);
        Assert.Equal(ulong.MaxValue - 3, ev.GraphIncarnation);
        Assert.Equal(ulong.MaxValue - 4, ev.SourceId);
        Assert.Equal(ulong.MaxValue - 5, ev.TargetId);
        Assert.Equal("<urn:g>", ev.TripleGraph);
        Assert.Equal("<urn:s>", ev.TripleSubject);
        Assert.Equal("<urn:p>", ev.TriplePredicate);
        Assert.Equal("\"value\"", ev.TripleObject);
    }
}

internal sealed class RetainedCutFactAttribute : FactAttribute
{
    public RetainedCutFactAttribute()
    {
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("GRAFEO_CDC_EVICTED_FIXTURE")))
            Skip = "Generate the native C retained-cut fixture first";
    }
}

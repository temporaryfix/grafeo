using Xunit;

namespace Grafeo.Tests;

/// <summary>Canonical owner lifecycle and error transport for vector indexes.</summary>
public sealed class VectorTests
{
    [Fact]
    public void VectorOwnerRebuildAndDropPreserveMissingSemantics()
    {
        using var db = GrafeoDB.Memory();

        // Populate via GQL, then use only canonical owner mutations.
        db.Execute("INSERT (:Doc {title: 'test', emb: [1.0, 2.0, 3.0]})");
        var owner = db.CreateIndex(new CreateIndexRequest(IndexKind.Vector, "emb")
        {
            Label = "Doc", Dimensions = 3,
        });
        db.RebuildIndex(owner);

        var dropped = db.DropIndex(owner);
        Assert.True(dropped);
        Assert.False(db.DropIndex(owner));
        Assert.ThrowsAny<GrafeoException>(() => db.RebuildIndex(owner));
    }

    [Fact]
    public void UnknownOwnerDropReturnsFalseAndRebuildThrows()
    {
        using var db = GrafeoDB.Memory();

        var dropped = db.DropIndex(123456);
        Assert.False(dropped);
        Assert.ThrowsAny<GrafeoException>(() => db.RebuildIndex(123456));
    }

    [Fact]
    public void GraphQualifiedRequestsRetainSeparateOwnersAndErrors()
    {
        using var db = GrafeoDB.Memory();
        db.Execute("CREATE GRAPH scoped");
        var root = db.CreateIndex(new CreateIndexRequest(IndexKind.Property, "name"));
        var scoped = db.CreateIndex(new CreateIndexRequest(IndexKind.Property, "name") { Graph = new[] { "scoped" } });
        Assert.NotEqual(root, scoped);
        Assert.True(db.DropIndex(scoped));
        db.RebuildIndex(root);
        Assert.ThrowsAny<GrafeoException>(() => db.CreateIndex(new CreateIndexRequest(IndexKind.Property, "unique") { Graph = new[] { "scoped\0other" } }));
        Assert.ThrowsAny<GrafeoException>(() => db.CreateIndex(new CreateIndexRequest(IndexKind.Property, "bad") { Dimensions = 0 }));
    }

    [Fact]
    public void TextTokenizerPresenceSurvivesRebuildAndReopen()
    {
        using var db = GrafeoDB.Memory();
        db.Execute("INSERT (:Doc {body: 'x ox fox lengthy', default_body: 'x ox fox lengthy', zero_body: 'x ox fox lengthy'})");
        var request = new CreateIndexRequest(IndexKind.Text, "body") { Label = "Doc", MinTokenLength = 7 };
        var owner = db.CreateIndex(request);
        foreach (var kind in new[] { IndexKind.Property, IndexKind.BTree, IndexKind.Vector })
        {
            Assert.ThrowsAny<GrafeoException>(() => db.CreateIndex(new CreateIndexRequest(kind, "bad")
            {
                MinTokenLength = 0,
                Label = kind == IndexKind.Vector ? "Doc" : null,
                Dimensions = kind == IndexKind.Vector ? (nuint?)3 : null,
            }));
        }
        Assert.ThrowsAny<GrafeoException>(() => db.CreateIndex(request));
        var defaultOwner = db.CreateIndex(new CreateIndexRequest(IndexKind.Text, "default_body") { Label = "Doc" });
        Assert.Equal(owner + 1, defaultOwner);
        var zeroOwner = db.CreateIndex(new CreateIndexRequest(IndexKind.Text, "zero_body") { Label = "Doc", MinTokenLength = 0 });
        static void Check(GrafeoDB candidate)
        {
            foreach (var (property, token, count) in new[]
            {
                ("body", "fox", 0), ("body", "lengthy", 1),
                ("default_body", "x", 0), ("default_body", "ox", 1), ("zero_body", "x", 1),
            })
            {
                Assert.Equal(count, candidate.Execute($"CALL grafeo.search.text('Doc', '{property}', '{token}', 10)").Rows.Count);
            }
        }
        Check(db);
        foreach (var id in new[] { owner, defaultOwner, zeroOwner }) db.RebuildIndex(id);
        Check(db);
        var directory = Path.Combine(Path.GetTempPath(), $"grafeo-csharp-text-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            var path = Path.Combine(directory, "text-options.grafeo");
            db.Save(path);
            using var reopened = GrafeoDB.Open(path);
            Check(reopened);
            foreach (var id in new[] { owner, defaultOwner, zeroOwner }) reopened.RebuildIndex(id);
            Check(reopened);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}

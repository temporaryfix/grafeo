# Grafeo C# Bindings

C# bindings for the [Grafeo](https://grafeo.dev) graph database.

## Quick Start

```csharp
using Grafeo;

// Create an in-memory database
await using var db = GrafeoDB.Memory();

// Execute a GQL query
db.Execute("INSERT (:Person {name: 'Alix', age: 30})");

// Query with parameters
var result = db.ExecuteWithParams(
    "MATCH (p:Person) WHERE p.name = $name RETURN p.name, p.age",
    new Dictionary<string, object?> { ["name"] = "Alix" });

foreach (var row in result.Rows)
    Console.WriteLine($"{row["p.name"]}: {row["p.age"]}");

// Async execution
var asyncResult = await db.ExecuteAsync("MATCH (p:Person) RETURN p");

// ACID transactions with auto-rollback
using var tx = db.BeginTransaction();
tx.Execute("INSERT (:Person {name: 'Gus'})");
tx.Execute("INSERT (:Person {name: 'Vincent'})-[:KNOWS]->(:Person {name: 'Jules'})");
tx.Commit(); // rolls back automatically if not reached
```

## Bounded execution and cancellation

```csharp
using var source = new CancellationTokenSource();
var cancellationToken = source.Token;
using var control = new QueryControl(TimeSpan.FromSeconds(5));
var result = await db.ExecuteWithOptionsAsync(
    "MATCH (p:Person) RETURN p.name",
    new ExecutionOptions { Control = control, MaxRows = 10_000, MaxBytes = 16 * 1024 * 1024 },
    cancellationToken: cancellationToken);

await using var stream = await db.ExecuteStreamWithOptionsAsync(
    "MATCH (p:Person) RETURN p.name", cancellationToken: cancellationToken);
await foreach (var row in stream.RowsAsync(cancellationToken))
    Console.WriteLine(row["p.name"]);
```

Controls are single-use; their timeout starts at construction. Cancellation
reaches native execution and throws `QueryCanceledException` (`GRAFEO-Q007`);
deadlines use `QueryException` (`GRAFEO-Q003`), limits `StorageException`
(`GRAFEO-S001`). A failed statement preserves earlier transaction writes.
Async calls reserve native handles before scheduling. Dispose reports busy while
an operation or transaction owns the database; finish it and retry disposal.

The default eager/collection row cap is 1,000,000 and copy envelope is 64 MiB;
explicit zero is a real limit. Streams apply byte limits per row/chunk and an
optional total row limit. `NextChunk` returns at most 1024 rows. `ToList` also
bounds total collected copies and returns no partial list on failure. Iteration
closes on early break. Explicit `Close`/`Dispose` retains terminal errors, so
cleanup can throw. These limits cover result copies. Engine execution memory and application-retained
results require separate budgeting. Temporal markers with unexpected value types
or timestamps outside .NET's date range remain exact wire maps; supported
timestamps retain microsecond precision.

## Building from Source

1. Build the Grafeo C library:
   ```bash
   cargo build --release -p grafeo-c
   ```

2. Copy the native library to the test directory:
   - Windows: `copy target\release\grafeo_c.dll crates\bindings\csharp\tests\Grafeo.Tests\`
   - macOS: `cp target/release/libgrafeo_c.dylib crates/bindings/csharp/tests/Grafeo.Tests/`
   - Linux: `cp target/release/libgrafeo_c.so crates/bindings/csharp/tests/Grafeo.Tests/`

3. Build and test:
   ```bash
   cd crates/bindings/csharp
   dotnet build
   dotnet test
   ```

## Index owners

```csharp
uint owner = db.CreateIndex(new CreateIndexRequest(IndexKind.Vector, "embedding")
{
    Label = "Doc", Dimensions = 3,
});
db.RebuildIndex(owner);
bool dropped = db.DropIndex(owner);
```

The same request supports Property, BTree and Text. Empty `Graph` selects root;
each supplied string is one literal component, including empty names, separators
and NULs. Nullable options distinguish absence from explicit empty/zero values.
Drop returns false only for a missing owner; missing rebuild and engine failures
throw `GrafeoException`. Feature availability and read/search APIs are unchanged.

## Requirements

- .NET 8.0 or later
- `grafeo_c` native library (built from the `grafeo-c` crate)

## License

Apache-2.0

## Bounded change pages

Enable `db.CdcEnabled`, then call `ChangesAfter(cursor, maxEvents, maxBytes)`.
Pass `null` initially; resume with `page.Next`. `NodeHistoryAfter` and
`EdgeHistoryAfter` also accept an unsigned entity ID and inclusive minimum epoch.
Native coordinates are `ulong`; event properties retain exact JSON number tokens
(use `GetInt64`/`GetUInt64`). Creation events include labels, edge type/endpoints,
and graph identity. `ChangePage` contains managed data, survives database disposal,
and requires no disposal itself.

```csharp
byte[]? cursor = null;
while (true)
{
    var page = db.ChangesAfter(cursor, 100, 64 * 1024);
    foreach (var change in page.Events) Console.WriteLine(change.EntityId);
    if (cursor is not null && cursor.SequenceEqual(page.Next)) break;
    cursor = page.Next;
}
```

Both bounds must be positive. The byte bound counts native event encodings,
excluding JSON/page envelopes. Cursors are opaque canonical 97-byte values;
non-null empty values are invalid. Empty filtered pages can advance, so EOF is
an unchanged cursor, not simply an empty event list. Native structured errors
retain their codes, including invalid/foreign cursor, evicted cursor and resource
limits. These APIs require a native library built with CDC support.

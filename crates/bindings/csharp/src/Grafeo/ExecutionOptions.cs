namespace Grafeo;

/// <summary>Single-execution ownership, language, and explicit result limits.</summary>
public sealed class ExecutionOptions
{
    /// <summary>Optional single-use control. Its timeout starts at construction.</summary>
    public QueryControl? Control { get; init; }

    /// <summary>Total eager rows (default 1,000,000), or optional stream row cap.
    /// Zero is a real zero limit.</summary>
    public ulong? MaxRows { get; init; }

    /// <summary>Combined native and managed copy envelope, default 64 MiB.
    /// Streaming applies this per row/chunk. Native precommit admission reserves
    /// up to one quarter, capped at managed contiguous-copy capacity; managed
    /// conversion receives the remainder.</summary>
    public ulong? MaxBytes { get; init; }

    /// <summary>Query language; null or empty selects GQL.</summary>
    public string? Language { get; init; }
}

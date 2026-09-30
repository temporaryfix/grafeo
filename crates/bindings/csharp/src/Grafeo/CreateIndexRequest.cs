namespace Grafeo;

/// <summary>Canonical index families; unavailable features produce an engine error.</summary>
public enum IndexKind : uint
{
    Property = 0,
    BTree = 1,
    Text = 2,
    Vector = 3,
}

/// <summary>
/// Graph-qualified index creation. Empty Graph selects root; each string is one
/// literal component, including empty strings, separators and embedded NULs.
/// Null options are absent; explicit zero/empty values retain their presence.
/// Label is required for Text/Vector and absent for Property/BTree.
/// </summary>
public sealed record CreateIndexRequest(IndexKind Kind, string Property)
{
    public IReadOnlyList<string> Graph { get; init; } = Array.Empty<string>();
    public string? Name { get; init; }
    public string? Label { get; init; }
    public nuint? Dimensions { get; init; }
    public string? Metric { get; init; }
    public nuint? M { get; init; }
    public nuint? EfConstruction { get; init; }
    /// <summary>Text-only; null selects the default (2), while zero is valid.</summary>
    public nuint? MinTokenLength { get; init; }
    public string? Quantization { get; init; }
}

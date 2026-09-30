using System.Text.Json;
using System.Text.Json.Serialization;

namespace Grafeo;

/// <summary>One owned committed event. Native coordinates preserve all 64 bits;
/// property JSON retains exact integer tokens for GetInt64/GetUInt64.</summary>
[JsonNumberHandling(JsonNumberHandling.AllowReadingFromString)]
public sealed class ChangeEvent
{
    [JsonPropertyName("entity_id")]
    public ulong EntityId { get; init; }

    [JsonPropertyName("entity_type")]
    public string EntityType { get; init; } = "";

    [JsonPropertyName("kind")]
    public string Kind { get; init; } = "";

    [JsonPropertyName("epoch")]
    public ulong Epoch { get; init; }

    [JsonPropertyName("timestamp")]
    public ulong Timestamp { get; init; }

    [JsonPropertyName("graph_incarnation")]
    public ulong? GraphIncarnation { get; init; }

    [JsonPropertyName("before")]
    public JsonElement Before { get; init; }

    [JsonPropertyName("after")]
    public JsonElement After { get; init; }

    [JsonPropertyName("labels")]
    public string[]? Labels { get; init; }

    [JsonPropertyName("edge_type")]
    public string? EdgeType { get; init; }

    [JsonPropertyName("src_id")]
    public ulong? SourceId { get; init; }

    [JsonPropertyName("dst_id")]
    public ulong? TargetId { get; init; }

    [JsonPropertyName("lpg_graph")]
    public string[]? LpgGraph { get; init; }

    [JsonPropertyName("triple_graph")]
    public string? TripleGraph { get; init; }

    [JsonPropertyName("triple_subject")]
    public string? TripleSubject { get; init; }

    [JsonPropertyName("triple_predicate")]
    public string? TriplePredicate { get; init; }

    [JsonPropertyName("triple_object")]
    public string? TripleObject { get; init; }

}

/// <summary>Managed events and canonical exclusive cursor. No native handle remains;
/// stopping early requires no cleanup. Empty filtered pages can advance the cursor.</summary>
public sealed class ChangePage
{
    public IReadOnlyList<ChangeEvent> Events { get; }
    public byte[] Next { get; }

    internal ChangePage(string json, byte[] next)
    {
        Events = JsonSerializer.Deserialize<ChangeEvent[]>(json)
            ?? throw new InvalidDataException("Native CDC events must be an array");
        Next = next;
    }
}

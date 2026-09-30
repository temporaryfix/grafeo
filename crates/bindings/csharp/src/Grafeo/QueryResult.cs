// Entity types returned by query execution and CRUD operations.

namespace Grafeo;

/// <summary>A graph node with labels and properties.</summary>
/// <param name="Id">Unique node identifier.</param>
/// <param name="Labels">Node labels (e.g. "Person", "City").</param>
/// <param name="Properties">Key-value property map.</param>
public sealed record Node(
    long Id,
    IReadOnlyList<string> Labels,
    IReadOnlyDictionary<string, object?> Properties);

/// <summary>A directed graph edge connecting two nodes.</summary>
/// <param name="Id">Unique edge identifier.</param>
/// <param name="Type">Edge type (e.g. "KNOWS", "LIVES_IN").</param>
/// <param name="SourceId">Source node ID.</param>
/// <param name="TargetId">Target node ID.</param>
/// <param name="Properties">Key-value property map.</param>
public sealed record Edge(
    long Id,
    string Type,
    long SourceId,
    long TargetId,
    IReadOnlyDictionary<string, object?> Properties);

/// <summary>Result of a query execution.</summary>
/// <param name="Columns">Column names from the RETURN clause.</param>
/// <param name="Rows">Row data as column-name to value dictionaries.</param>
/// <param name="Nodes">Extracted node entities from the result.</param>
/// <param name="Edges">Extracted edge entities from the result.</param>
/// <param name="ExecutionTimeMs">Query execution time in milliseconds.</param>
/// <param name="RowsScanned">Number of rows scanned by the engine.</param>
public sealed record QueryResult(
    IReadOnlyList<string> Columns,
    IReadOnlyList<IReadOnlyDictionary<string, object?>> Rows,
    IReadOnlyList<Node> Nodes,
    IReadOnlyList<Edge> Edges,
    double ExecutionTimeMs,
    long RowsScanned);

/// <summary>A single vector search result (node ID + distance).</summary>
/// <param name="NodeId">The matching node's ID.</param>
/// <param name="Distance">Distance from the query vector.</param>
public sealed record VectorResult(long NodeId, double Distance);

// One decoder owns and frees the complete native result, including on failure.
internal static class QueryResultDecoder
{
    internal static QueryResult Decode(nint resultPtr, ulong maxBytes)
    {
        try
        {
            var remaining = maxBytes;
            var json = Native.ManagedCopyBudget.ReadUtf8(Native.NativeMethods.grafeo_result_json(resultPtr), ref remaining);
            var executionTime = Native.NativeMethods.grafeo_result_execution_time_ms(resultPtr);
            var scanned = checked((long)Native.NativeMethods.grafeo_result_rows_scanned(resultPtr));
            var rows = ValueConverter.ParseRows(json);
            var columns = ValueConverter.ExtractColumns(rows);
            var (nodes, edges) = ValueConverter.ExtractEntities(rows);
            return new QueryResult(columns, rows, nodes, edges, executionTime, scanned);
        }
        finally { Native.NativeMethods.grafeo_free_result(resultPtr); }
    }
}

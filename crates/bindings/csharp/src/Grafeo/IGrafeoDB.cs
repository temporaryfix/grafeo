namespace Grafeo;

/// <summary>
/// Interface for the Grafeo database, enabling testability via mocking.
/// </summary>
public interface IGrafeoDB : IDisposable, IAsyncDisposable
{
    // Query execution
    QueryResult ExecuteWithOptions(string query, ExecutionOptions? options = null,
        Dictionary<string, object?>? parameters = null, CancellationToken cancellationToken = default);
    Task<QueryResult> ExecuteWithOptionsAsync(string query, ExecutionOptions? options = null,
        Dictionary<string, object?>? parameters = null, CancellationToken cancellationToken = default);
    QueryResult Execute(string query);
    Task<QueryResult> ExecuteAsync(string query, CancellationToken ct = default);
    QueryResult ExecuteWithParams(string query, Dictionary<string, object?> parameters);
    QueryResult ExecuteLanguage(string language, string query, Dictionary<string, object?>? parameters = null);

    ResultStream ExecuteStreamWithOptions(string query, ExecutionOptions? options = null,
        Dictionary<string, object?>? parameters = null, CancellationToken cancellationToken = default);
    Task<ResultStream> ExecuteStreamWithOptionsAsync(string query, ExecutionOptions? options = null,
        Dictionary<string, object?>? parameters = null, CancellationToken cancellationToken = default);

    // Owned bounded change pages
    bool CdcEnabled { get; set; }
    ChangePage ChangesAfter(byte[]? cursor, int maxEvents, int maxBytes);
    ChangePage NodeHistoryAfter(ulong id, ulong sinceEpoch, byte[]? cursor, int maxEvents, int maxBytes);
    ChangePage EdgeHistoryAfter(ulong id, ulong sinceEpoch, byte[]? cursor, int maxEvents, int maxBytes);

    // Transactions
    ITransaction BeginTransaction();
    ITransaction BeginTransaction(IsolationLevel isolationLevel);

    // Node CRUD
    long CreateNode(IEnumerable<string> labels, Dictionary<string, object?>? properties = null);
    Node? GetNode(long id);
    bool DeleteNode(long id);
    void SetNodeProperty(long id, string key, object? value);
    bool RemoveNodeProperty(long id, string key);
    bool AddNodeLabel(long id, string label);
    bool RemoveNodeLabel(long id, string label);

    // Edge CRUD
    long CreateEdge(long sourceId, long targetId, string edgeType, Dictionary<string, object?>? properties = null);
    Edge? GetEdge(long id);
    bool DeleteEdge(long id);
    void SetEdgeProperty(long id, string key, object? value);
    bool RemoveEdgeProperty(long id, string key);

    // Catalog-owned index mutations
    uint CreateIndex(CreateIndexRequest request);
    bool DropIndex(uint owner);
    void RebuildIndex(uint owner);

    // Admin
    long NodeCount { get; }
    long EdgeCount { get; }
    IReadOnlyDictionary<string, object?> Info();
    void Save(string path);
    void ClearPlanCache();

    // Schema
    void SetSchema(string name);
    void ResetSchema();
    string? CurrentSchema();
}

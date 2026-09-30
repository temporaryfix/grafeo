// Primary database handle for the Grafeo graph database.

using System.Runtime.InteropServices;

using Grafeo.Native;

namespace Grafeo;

/// <summary>
/// Transaction isolation levels supported by Grafeo.
/// Values match the <c>GrafeoIsolationLevel</c> C enum.
/// </summary>
public enum IsolationLevel
{
    /// <summary>Read committed: each statement sees the latest committed data.</summary>
    ReadCommitted = 0,

    /// <summary>Snapshot isolation: the transaction sees a consistent snapshot taken at begin.</summary>
    Snapshot = 1,

    /// <summary>Serializable: full serializability, the strongest isolation guarantee.</summary>
    Serializable = 2,
}

/// <summary>
/// Primary handle to a Grafeo graph database.
/// Thread-safe: the underlying engine uses <c>Arc&lt;RwLock&gt;</c>.
/// Implements <see cref="IDisposable"/> and <see cref="IAsyncDisposable"/>
/// for deterministic cleanup via <c>using</c>/<c>await using</c>.
/// </summary>
public sealed class GrafeoDB : IGrafeoDB, IDisposable, IAsyncDisposable
{
    private readonly DatabaseHandle _handle;
    private volatile bool _disposed;
    private readonly object _gate = new();
    private int _active;

    private GrafeoDB(DatabaseHandle handle) => _handle = handle;

    // =========================================================================
    // Lifecycle
    // =========================================================================

    /// <summary>Create a new in-memory database.</summary>
    public static GrafeoDB Memory()
    {
        var ptr = NativeMethods.grafeo_open_memory();
        if (ptr == nint.Zero)
            throw GrafeoException.FromLastError();

        var handle = new DatabaseHandle();
        Marshal.InitHandle(handle, ptr);
        return new GrafeoDB(handle);
    }

    /// <summary>Open or create a persistent database at <paramref name="path"/>.</summary>
    public static GrafeoDB Open(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        var ptr = NativeMethods.grafeo_open(path);
        if (ptr == nint.Zero)
            throw GrafeoException.FromLastError();

        var handle = new DatabaseHandle();
        Marshal.InitHandle(handle, ptr);
        return new GrafeoDB(handle);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!Monitor.TryEnter(_gate)) throw Busy();
        try
        {
            if (_disposed) return;
            if (_active != 0) throw Busy();
            GrafeoException.ThrowIfFailed(NativeMethods.grafeo_close(_handle.DangerousGetHandle()));
            _handle.Closed = true;
            _disposed = true;
            _handle.Dispose();
        }
        finally { Monitor.Exit(_gate); }
    }

    /// <inheritdoc/>
    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }

    // =========================================================================
    // Query Execution
    // =========================================================================

    /// <summary>Execute a query with cancellation, a single-use owner, and result limits.</summary>
    public QueryResult ExecuteWithOptions(string query, ExecutionOptions? options = null,
        Dictionary<string, object?>? parameters = null, CancellationToken cancellationToken = default)
    {
        using var prepared = Prepare(query, options, parameters, cancellationToken);
        return prepared.Execute(false);
    }

    /// <summary>Reserve the database and query owner before scheduling native execution.</summary>
    public Task<QueryResult> ExecuteWithOptionsAsync(string query, ExecutionOptions? options = null,
        Dictionary<string, object?>? parameters = null, CancellationToken cancellationToken = default)
    {
        var prepared = Prepare(query, options, parameters, cancellationToken);
        try
        {
            return Task.Run(() => { using (prepared) return prepared.Execute(false); }, CancellationToken.None);
        }
        catch { prepared.Dispose(); throw; }
    }

    private PreparedQuery Prepare(string query, ExecutionOptions? options,
        Dictionary<string, object?>? parameters, CancellationToken token, bool streaming = false)
    {
        var lease = Acquire();
        return PreparedQuery.Create(lease, lease.Pointer, query,
            () => parameters is null ? null : ValueConverter.EncodeParams(parameters), options, token, streaming);
    }

    /// <summary>Execute a GQL query synchronously.</summary>
    public QueryResult Execute(string query) => ExecuteWithOptions(query);

    /// <summary>Execute a GQL query on the thread pool with native cancellation.</summary>
    public Task<QueryResult> ExecuteAsync(string query, CancellationToken ct = default)
        => ExecuteWithOptionsAsync(query, cancellationToken: ct);

    /// <summary>Execute a GQL query with parameters.</summary>
    public QueryResult ExecuteWithParams(string query, Dictionary<string, object?> parameters)
        => ExecuteWithOptions(query, parameters: parameters);

    /// <summary>Execute a parameterized query on the thread pool.</summary>
    public Task<QueryResult> ExecuteWithParamsAsync(string query,
        Dictionary<string, object?> parameters, CancellationToken ct = default)
        => ExecuteWithOptionsAsync(query, parameters: parameters, cancellationToken: ct);

    /// <summary>Execute a query in the given language with optional typed parameters.</summary>
    public QueryResult ExecuteLanguage(string language, string query, Dictionary<string, object?>? parameters = null)
        => ExecuteWithOptions(query, new ExecutionOptions { Language = language }, parameters);

    /// <summary>Execute a query in the given language on the thread pool.</summary>
    public Task<QueryResult> ExecuteLanguageAsync(string language, string query,
        Dictionary<string, object?>? parameters = null, CancellationToken ct = default)
        => ExecuteWithOptionsAsync(query, new ExecutionOptions { Language = language }, parameters, ct);

    /// <summary>Open a bounded lazy cursor over a read-only query.</summary>
    public ResultStream ExecuteStream(string query) => ExecuteStreamWithOptions(query);

    /// <summary>Open a bounded lazy cursor with cancellation and execution limits.</summary>
    public ResultStream ExecuteStreamWithOptions(string query, ExecutionOptions? options = null,
        Dictionary<string, object?>? parameters = null, CancellationToken cancellationToken = default)
    {
        using var prepared = Prepare(query, options, parameters, cancellationToken, true);
        return prepared.OpenStream();
    }

    /// <summary>Reserve the query owner before scheduling a lazy cursor opener.</summary>
    public Task<ResultStream> ExecuteStreamWithOptionsAsync(string query, ExecutionOptions? options = null,
        Dictionary<string, object?>? parameters = null, CancellationToken cancellationToken = default)
    {
        var prepared = Prepare(query, options, parameters, cancellationToken, true);
        try
        {
            return Task.Run(() => { using (prepared) return prepared.OpenStream(); }, CancellationToken.None);
        }
        catch { prepared.Dispose(); throw; }
    }

    /// <summary>Execute a Cypher query.</summary>
    public QueryResult ExecuteCypher(string query) => ExecuteLanguage("cypher", query);

    /// <summary>Execute a Cypher query on the thread pool.</summary>
    public Task<QueryResult> ExecuteCypherAsync(string query, CancellationToken ct = default)
        => ExecuteLanguageAsync("cypher", query, ct: ct);

    /// <summary>Execute a Sparql query.</summary>
    public QueryResult ExecuteSparql(string query) => ExecuteLanguage("sparql", query);

    /// <summary>Execute a Sparql query on the thread pool.</summary>
    public Task<QueryResult> ExecuteSparqlAsync(string query, CancellationToken ct = default)
        => ExecuteLanguageAsync("sparql", query, ct: ct);

    /// <summary>Execute a Gremlin query.</summary>
    public QueryResult ExecuteGremlin(string query) => ExecuteLanguage("gremlin", query);

    /// <summary>Execute a Gremlin query on the thread pool.</summary>
    public Task<QueryResult> ExecuteGremlinAsync(string query, CancellationToken ct = default)
        => ExecuteLanguageAsync("gremlin", query, ct: ct);

    /// <summary>Execute a Graphql query.</summary>
    public QueryResult ExecuteGraphql(string query) => ExecuteLanguage("graphql", query);

    /// <summary>Execute a Graphql query on the thread pool.</summary>
    public Task<QueryResult> ExecuteGraphqlAsync(string query, CancellationToken ct = default)
        => ExecuteLanguageAsync("graphql", query, ct: ct);

    /// <summary>Execute a Sql query.</summary>
    public QueryResult ExecuteSql(string query) => ExecuteLanguage("sql", query);

    /// <summary>Execute a Sql query on the thread pool.</summary>
    public Task<QueryResult> ExecuteSqlAsync(string query, CancellationToken ct = default)
        => ExecuteLanguageAsync("sql", query, ct: ct);

    // =========================================================================
    // Transactions
    // =========================================================================

    /// <summary>Begin a new ACID transaction with the default isolation level.</summary>
    public Transaction BeginTransaction() => BeginTransactionCore(NativeMethods.grafeo_begin_transaction);

    /// <summary>Begin a transaction with a specific isolation level.</summary>
    /// <param name="isolationLevel">
    /// Accepted values (case-insensitive): "read_committed" / "ReadCommitted",
    /// "snapshot" / "Snapshot" / "snapshot_isolation" / "SnapshotIsolation",
    /// "serializable" / "Serializable".
    /// </param>
    public Transaction BeginTransaction(string isolationLevel)
    {
        var level = ParseIsolationLevel(isolationLevel);
        return BeginTransactionCore(pointer => NativeMethods.grafeo_begin_transaction_with_isolation(pointer, level));
    }

    /// <summary>Begin a transaction with a specific isolation level.</summary>
    public Transaction BeginTransaction(IsolationLevel isolationLevel)
        => BeginTransactionCore(pointer => NativeMethods.grafeo_begin_transaction_with_isolation(pointer, (int)isolationLevel));

    private Transaction BeginTransactionCore(Func<nint, nint> begin)
    {
        var lease = Acquire();
        nint pointer = nint.Zero;
        try
        {
            pointer = begin(lease.Pointer);
            if (pointer == nint.Zero) throw GrafeoException.FromLastError(GrafeoStatus.Transaction);
            return new Transaction(pointer, lease);
        }
        catch
        {
            if (pointer != nint.Zero) NativeMethods.grafeo_free_transaction(pointer);
            lease.Dispose();
            throw;
        }
    }

    // Explicit interface implementations for ITransaction return type
    ITransaction IGrafeoDB.BeginTransaction() => BeginTransaction();
    ITransaction IGrafeoDB.BeginTransaction(IsolationLevel level) => BeginTransaction(level);

    // =========================================================================
    // Node CRUD
    // =========================================================================

    /// <summary>Create a node with labels and optional properties. Returns the new node ID.</summary>
    public long CreateNode(IEnumerable<string> labels, Dictionary<string, object?>? properties = null)
    {
        using var lease = Acquire();
        var labelsJson = System.Text.Json.JsonSerializer.Serialize(labels);
        var propsJson = properties is not null ? ValueConverter.EncodeParams(properties) : null;
        var id = NativeMethods.grafeo_create_node(lease.Pointer, labelsJson, propsJson);
        if (id == ulong.MaxValue)
            throw GrafeoException.FromLastError();
        return (long)id;
    }

    /// <summary>Get a node by ID. Returns null if not found.</summary>
    public Node? GetNode(long id)
    {
        using var lease = Acquire();
        var status = NativeMethods.grafeo_get_node(lease.Pointer, (ulong)id, out var nodePtr);
        if (status != (int)GrafeoStatus.Ok)
            return null;
        try
        {
            return ReadNode(nodePtr);
        }
        finally
        {
            NativeMethods.grafeo_free_node(nodePtr);
        }
    }

    /// <summary>Delete a node by ID. Returns true if deleted.</summary>
    public bool DeleteNode(long id)
    {
        using var lease = Acquire();
        return NativeMethods.grafeo_delete_node(lease.Pointer, (ulong)id) == 1;
    }

    /// <summary>Set a property on a node.</summary>
    public void SetNodeProperty(long id, string key, object? value)
    {
        using var lease = Acquire();
        var valueJson = ValueConverter.EncodeValue(value);
        GrafeoException.ThrowIfFailed(
            NativeMethods.grafeo_set_node_property(lease.Pointer, (ulong)id, key, valueJson));
    }

    /// <summary>Remove a property from a node. Returns true if removed.</summary>
    public bool RemoveNodeProperty(long id, string key)
    {
        using var lease = Acquire();
        return NativeMethods.grafeo_remove_node_property(lease.Pointer, (ulong)id, key) == 1;
    }

    /// <summary>Add a label to a node. Returns true if added.</summary>
    public bool AddNodeLabel(long id, string label)
    {
        using var lease = Acquire();
        return NativeMethods.grafeo_add_node_label(lease.Pointer, (ulong)id, label) == 1;
    }

    /// <summary>Remove a label from a node. Returns true if removed.</summary>
    public bool RemoveNodeLabel(long id, string label)
    {
        using var lease = Acquire();
        return NativeMethods.grafeo_remove_node_label(lease.Pointer, (ulong)id, label) == 1;
    }

    // =========================================================================
    // Edge CRUD
    // =========================================================================

    /// <summary>Create an edge between two nodes. Returns the new edge ID.</summary>
    public long CreateEdge(
        long sourceId,
        long targetId,
        string edgeType,
        Dictionary<string, object?>? properties = null)
    {
        using var lease = Acquire();
        var propsJson = properties is not null ? ValueConverter.EncodeParams(properties) : null;
        var id = NativeMethods.grafeo_create_edge(
            lease.Pointer, (ulong)sourceId, (ulong)targetId, edgeType, propsJson);
        if (id == ulong.MaxValue)
            throw GrafeoException.FromLastError();
        return (long)id;
    }

    /// <summary>Get an edge by ID. Returns null if not found.</summary>
    public Edge? GetEdge(long id)
    {
        using var lease = Acquire();
        var status = NativeMethods.grafeo_get_edge(lease.Pointer, (ulong)id, out var edgePtr);
        if (status != (int)GrafeoStatus.Ok)
            return null;
        try
        {
            return ReadEdge(edgePtr);
        }
        finally
        {
            NativeMethods.grafeo_free_edge(edgePtr);
        }
    }

    /// <summary>Delete an edge by ID. Returns true if deleted.</summary>
    public bool DeleteEdge(long id)
    {
        using var lease = Acquire();
        return NativeMethods.grafeo_delete_edge(lease.Pointer, (ulong)id) == 1;
    }

    /// <summary>Set a property on an edge.</summary>
    public void SetEdgeProperty(long id, string key, object? value)
    {
        using var lease = Acquire();
        var valueJson = ValueConverter.EncodeValue(value);
        GrafeoException.ThrowIfFailed(
            NativeMethods.grafeo_set_edge_property(lease.Pointer, (ulong)id, key, valueJson));
    }

    /// <summary>Remove a property from an edge. Returns true if removed.</summary>
    public bool RemoveEdgeProperty(long id, string key)
    {
        using var lease = Acquire();
        return NativeMethods.grafeo_remove_edge_property(lease.Pointer, (ulong)id, key) == 1;
    }

    // =========================================================================
    // Admin
    // =========================================================================

    /// <summary>Number of nodes in the database.</summary>
    public long NodeCount
    {
        get
        {
            using var lease = Acquire();
            return (long)NativeMethods.grafeo_node_count(lease.Pointer);
        }
    }

    /// <summary>Number of edges in the database.</summary>
    public long EdgeCount
    {
        get
        {
            using var lease = Acquire();
            return (long)NativeMethods.grafeo_edge_count(lease.Pointer);
        }
    }

    /// <summary>Get database info as a dictionary (version, node count, edge count, etc.).</summary>
    public IReadOnlyDictionary<string, object?> Info()
    {
        using var lease = Acquire();
        var ptr = NativeMethods.grafeo_info(lease.Pointer);
        if (ptr == nint.Zero)
            throw GrafeoException.FromLastError();
        try
        {
            var json = Marshal.PtrToStringUTF8(ptr)!;
            return ValueConverter.ParseObject(json);
        }
        finally
        {
            NativeMethods.grafeo_free_string(ptr);
        }
    }

    /// <summary>Get the Grafeo library version string.</summary>
    public static string Version
    {
        get
        {
            var ptr = NativeMethods.grafeo_version();
            return Marshal.PtrToStringUTF8(ptr) ?? "unknown";
        }
    }

    /// <summary>Save the database to a file path.</summary>
    public void Save(string path)
    {
        using var lease = Acquire();
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_save(lease.Pointer, path));
    }

    /// <summary>Clear all cached query plans, forcing re-parsing on next execution.</summary>
    public void ClearPlanCache()
    {
        using var lease = Acquire();
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_clear_plan_cache(lease.Pointer));
    }

    // =========================================================================
    // Schema Context
    // =========================================================================

    /// <summary>Set the current schema for subsequent queries.</summary>
    public void SetSchema(string name)
    {
        using var lease = Acquire();
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_set_schema(lease.Pointer, name));
    }

    /// <summary>Clear the current schema context.</summary>
    public void ResetSchema()
    {
        using var lease = Acquire();
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_reset_schema(lease.Pointer));
    }

    /// <summary>Get the current schema name, or <c>null</c> if none is set.</summary>
    public string? CurrentSchema()
    {
        using var lease = Acquire();
        var ptr = NativeMethods.grafeo_current_schema(lease.Pointer);
        return ptr == nint.Zero ? null : Marshal.PtrToStringUTF8(ptr);
    }

    // =========================================================================
    // Backup / Restore
    // =========================================================================

    /// <summary>Create a full backup at the given directory path.</summary>
    public void BackupFull(string path)
    {
        using var lease = Acquire();
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_backup_full(lease.Pointer, path));
    }

    /// <summary>Create an incremental backup at the given directory path.</summary>
    public void BackupIncremental(string path)
    {
        using var lease = Acquire();
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_backup_incremental(lease.Pointer, path));
    }

    /// <summary>Restore database to a specific epoch from a backup directory.</summary>
    /// <param name="backupDir">Directory containing backup files.</param>
    /// <param name="epoch">Target epoch to restore to.</param>
    /// <param name="outputPath">Path for the restored database.</param>
    public static void RestoreToEpoch(string backupDir, ulong epoch, string outputPath)
    {
        GrafeoException.ThrowIfFailed(
            NativeMethods.grafeo_restore_to_epoch(backupDir, epoch, outputPath));
    }

    // =========================================================================
    // Maintenance
    // =========================================================================

    /// <summary>Fold retained committed LPG history into a columnar base with a writable overlay.</summary>
    /// <remarks>
    /// Call again to fold later writes. Requires no active transactions or live Sessions.
    /// This is not a durability checkpoint or a history-retention lease.
    /// </remarks>
    public void Compact()
    {
        using var lease = Acquire();
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_compact(lease.Pointer));
    }

    // =========================================================================
    // Projections
    // =========================================================================

    /// <summary>Create a named graph projection that includes the specified node labels and edge types.</summary>
    /// <param name="name">Unique name for the projection.</param>
    /// <param name="nodeLabels">Node labels to include (null or empty for all).</param>
    /// <param name="edgeTypes">Edge types to include (null or empty for all).</param>
    /// <returns><c>true</c> if the projection was created.</returns>
    public bool CreateProjection(string name, IEnumerable<string>? nodeLabels = null, IEnumerable<string>? edgeTypes = null)
    {
        using var lease = Acquire();
        var labels = nodeLabels?.ToArray() ?? [];
        var types = edgeTypes?.ToArray() ?? [];

        // Marshal all strings manually (name + string arrays) to avoid
        // source-gen issues with mixed auto-marshalled + pointer parameters.
        var namePtr = Marshal.StringToCoTaskMemUTF8(name);
        var labelPtrs = labels.Select(Marshal.StringToCoTaskMemUTF8).ToArray();
        var typePtrs = types.Select(Marshal.StringToCoTaskMemUTF8).ToArray();

        try
        {
            unsafe
            {
                fixed (nint* lp = labelPtrs.Length > 0 ? labelPtrs : null)
                fixed (nint* tp = typePtrs.Length > 0 ? typePtrs : null)
                {
                    return NativeMethods.grafeo_create_projection(
                        lease.Pointer, namePtr,
                        (nint)lp, (nuint)labels.Length,
                        (nint)tp, (nuint)types.Length);
                }
            }
        }
        finally
        {
            Marshal.FreeCoTaskMem(namePtr);
            foreach (var p in labelPtrs) Marshal.FreeCoTaskMem(p);
            foreach (var p in typePtrs) Marshal.FreeCoTaskMem(p);
        }
    }

    /// <summary>Drop a named graph projection.</summary>
    /// <returns><c>true</c> if the projection existed, <c>false</c> otherwise.</returns>
    public bool DropProjection(string name)
    {
        using var lease = Acquire();
        return NativeMethods.grafeo_drop_projection(lease.Pointer, name);
    }

    /// <summary>List all named graph projections as a JSON array.</summary>
    public string ListProjections()
    {
        using var lease = Acquire();
        var ptr = NativeMethods.grafeo_list_projections(lease.Pointer);
        if (ptr == nint.Zero) return "[]";
        try
        {
            return Marshal.PtrToStringUTF8(ptr) ?? "[]";
        }
        finally
        {
            NativeMethods.grafeo_free_string(ptr);
        }
    }

    // =========================================================================
    // Change Data Capture
    // =========================================================================

    /// <summary>Gets or sets whether Change Data Capture is enabled.</summary>
    public bool CdcEnabled
    {
        get
        {
            using var lease = Acquire();
            return NativeMethods.grafeo_is_cdc_enabled(lease.Pointer);
        }
        set
        {
            using var lease = Acquire();
            NativeMethods.grafeo_set_cdc_enabled(lease.Pointer, value);
        }
    }

    /// <summary>Read an owned bounded page. Null starts at the retained floor; other cursors
    /// must contain 97 bytes. Both limits are positive. Bytes count native event encodings,
    /// excluding JSON/page envelopes. An unchanged cursor means EOF.</summary>
    public ChangePage ChangesAfter(byte[]? cursor, int maxEvents, int maxBytes) =>
        ReadChangePage(cursor, maxEvents, maxBytes, NativeMethods.grafeo_changes_after);

    /// <summary>Read node history at or after the inclusive epoch, with ChangesAfter bounds/ownership.</summary>
    public ChangePage NodeHistoryAfter(ulong id, ulong sinceEpoch, byte[]? cursor, int maxEvents, int maxBytes) =>
        ReadChangePage(cursor, maxEvents, maxBytes, (db, ptr, len, rows, bytes) =>
            NativeMethods.grafeo_node_history_after(db, id, sinceEpoch, ptr, len, rows, bytes));

    /// <summary>Read edge history at or after the inclusive epoch, with ChangesAfter bounds/ownership.</summary>
    public ChangePage EdgeHistoryAfter(ulong id, ulong sinceEpoch, byte[]? cursor, int maxEvents, int maxBytes) =>
        ReadChangePage(cursor, maxEvents, maxBytes, (db, ptr, len, rows, bytes) =>
            NativeMethods.grafeo_edge_history_after(db, id, sinceEpoch, ptr, len, rows, bytes));

    private ChangePage ReadChangePage(byte[]? cursor, int maxEvents, int maxBytes,
        Func<nint, nint, nuint, nuint, nuint, nint> read)
    {
        using var lease = Acquire();
        // Keep null distinct from an invalid empty cursor. Copy only canonical
        // length input: native length validation runs before any pointer read.
        var input = cursor is null ? nint.Zero : Marshal.AllocHGlobal(97);
        try
        {
            if (cursor?.Length == 97) Marshal.Copy(cursor, 0, input, 97);
            var page = read(lease.Pointer, input, (nuint)(cursor?.Length ?? 0),
                (nuint)Math.Max(0, maxEvents), (nuint)Math.Max(0, maxBytes));
            if (page == nint.Zero) throw GrafeoException.FromLastError();
            try
            {
                var next = new byte[97];
                Marshal.Copy(NativeMethods.grafeo_change_page_cursor(page), next, 0, next.Length);
                var json = Marshal.PtrToStringUTF8(NativeMethods.grafeo_change_page_events_json(page))
                    ?? throw new InvalidDataException("Native CDC page has no event JSON");
                return new ChangePage(json, next);
            }
            finally { NativeMethods.grafeo_free_change_page(page); }
        }
        finally { if (input != nint.Zero) Marshal.FreeHGlobal(input); }
    }

    // =========================================================================
    // Vector Search
    // =========================================================================

    /// <summary>Create one graph-qualified catalog owner.</summary>
    public uint CreateIndex(CreateIndexRequest request)
    {
        using var lease = Acquire();
        ArgumentNullException.ThrowIfNull(request);
        var allocations = new List<nint>();
        try
        {
            var encoding = new System.Text.UTF8Encoding(false, true);
            NativeUtf8 Span(string value)
            {
                ArgumentNullException.ThrowIfNull(value);
                var bytes = encoding.GetBytes(value);
                if (bytes.Length == 0) return default;
                var pointer = Marshal.AllocHGlobal(bytes.Length);
                allocations.Add(pointer);
                Marshal.Copy(bytes, 0, pointer, bytes.Length);
                return new NativeUtf8 { Data = pointer, Length = (nuint)bytes.Length };
            }
            var native = new NativeIndexRequest
            {
                Kind = (uint)request.Kind,
                Property = Span(request.Property),
                GraphCount = (nuint)request.Graph.Count,
            };
            if (request.Graph.Count != 0)
            {
                var stride = Marshal.SizeOf<NativeUtf8>();
                native.Graph = Marshal.AllocHGlobal(checked(stride * request.Graph.Count));
                allocations.Add(native.Graph);
                for (var i = 0; i < request.Graph.Count; i++)
                    Marshal.StructureToPtr(Span(request.Graph[i]), native.Graph + checked(i * stride), false);
            }
            if (request.Name is { } name) { native.Options |= 1; native.Name = Span(name); }
            if (request.Label is { } label) { native.Options |= 2; native.Label = Span(label); }
            if (request.Dimensions is { } dimensions) { native.Options |= 4; native.Dimensions = dimensions; }
            if (request.Metric is { } metric) { native.Options |= 8; native.Metric = Span(metric); }
            if (request.M is { } m) { native.Options |= 16; native.M = m; }
            if (request.EfConstruction is { } ef) { native.Options |= 32; native.EfConstruction = ef; }
            if (request.MinTokenLength is { } minimum) { native.Options |= 128; native.MinTokenLength = minimum; }
            if (request.Quantization is { } quantization) { native.Options |= 64; native.Quantization = Span(quantization); }
            GrafeoException.ThrowIfFailed(NativeMethods.grafeo_create_index(lease.Pointer, in native, out var owner));
            return owner;
        }
        finally
        {
            foreach (var pointer in allocations) Marshal.FreeHGlobal(pointer);
        }
    }

    /// <summary>Drop exactly this owner; absent owners return false.</summary>
    public bool DropIndex(uint owner)
    {
        using var lease = Acquire();
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_drop_index(lease.Pointer, owner, out var dropped));
        return dropped != 0;
    }

    /// <summary>Rebuild this owner with its resolved configuration. Absence is an error.</summary>
    public void RebuildIndex(uint owner)
    {
        using var lease = Acquire();
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_rebuild_index(lease.Pointer, owner));
    }

    /// <summary>Perform a vector similarity search.</summary>
    /// <returns>List of (nodeId, distance) results ordered by distance.</returns>
    public IReadOnlyList<VectorResult> VectorSearch(
        string label, string property, float[] query, int k, uint ef = 0)
    {
        using var lease = Acquire();
        unsafe
        {
            fixed (float* queryPtr = query)
            {
                var status = NativeMethods.grafeo_vector_search(
                    lease.Pointer, label, property,
                    queryPtr, (nuint)query.Length, (nuint)k, ef,
                    out var idsPtr, out var distsPtr, out var count);

                GrafeoException.ThrowIfFailed(status);
                return ReadVectorResults(idsPtr, distsPtr, count);
            }
        }
    }

    /// <summary>Perform a Maximal Marginal Relevance (MMR) search.</summary>
    public IReadOnlyList<VectorResult> MmrSearch(
        string label, string property, float[] query,
        int k, int fetchK, float lambda, int ef = 0)
    {
        using var lease = Acquire();
        unsafe
        {
            fixed (float* queryPtr = query)
            {
                var status = NativeMethods.grafeo_mmr_search(
                    lease.Pointer, label, property,
                    queryPtr, (nuint)query.Length, (nuint)k,
                    fetchK, lambda, ef,
                    out var idsPtr, out var distsPtr, out var count);

                GrafeoException.ThrowIfFailed(status);
                return ReadVectorResults(idsPtr, distsPtr, count);
            }
        }
    }

    // =========================================================================
    // Internals
    // =========================================================================

    internal static GrafeoException Busy() => new("Native owner is busy", GrafeoStatus.Database);

    internal NativeLease Acquire()
    {
        if (!Monitor.TryEnter(_gate)) throw Busy();
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var lease = new NativeLease(this);
            var retained = false;
            try
            {
                _handle.DangerousAddRef(ref retained);
                _active++;
                return lease;
            }
            catch
            {
                if (retained) _handle.DangerousRelease();
                throw;
            }
        }
        finally { Monitor.Exit(_gate); }
    }

    internal sealed class NativeLease : IDisposable
    {
        private GrafeoDB? _owner;
        internal nint Pointer { get; }
        internal NativeLease(GrafeoDB owner)
        {
            _owner = owner;
            Pointer = owner._handle.DangerousGetHandle();
        }
        public void Dispose()
        {
            var owner = Interlocked.Exchange(ref _owner, null);
            if (owner is null) return;
            lock (owner._gate)
            {
                owner._handle.DangerousRelease();
                owner._active--;
            }
        }
    }

    /// <summary>Parse a string isolation level name to the integer value expected by the C API.</summary>
    private static int ParseIsolationLevel(string isolationLevel) =>
        isolationLevel.ToLowerInvariant() switch
        {
            "read_committed" or "readcommitted" => (int)IsolationLevel.ReadCommitted,
            "snapshot" or "snapshot_isolation" or "snapshotisolation" => (int)IsolationLevel.Snapshot,
            "serializable" => (int)IsolationLevel.Serializable,
            _ => throw new ArgumentException(
                $"Unknown isolation level: '{isolationLevel}'. " +
                "Use \"read_committed\", \"snapshot\", or \"serializable\".",
                nameof(isolationLevel)),
        };

    /// <summary>Read a node from a native GrafeoNode pointer.</summary>
    private static Node ReadNode(nint nodePtr)
    {
        var id = (long)NativeMethods.grafeo_node_id(nodePtr);
        var labelsJsonPtr = NativeMethods.grafeo_node_labels_json(nodePtr);
        var propsJsonPtr = NativeMethods.grafeo_node_properties_json(nodePtr);

        var labelsJson = Marshal.PtrToStringUTF8(labelsJsonPtr) ?? "[]";
        var propsJson = Marshal.PtrToStringUTF8(propsJsonPtr) ?? "{}";

        var labels = ValueConverter.ParseStringArray(labelsJson);
        var properties = ValueConverter.ParseObject(propsJson);
        return new Node(id, labels, properties);
    }

    /// <summary>Read an edge from a native GrafeoEdge pointer.</summary>
    private static Edge ReadEdge(nint edgePtr)
    {
        var id = (long)NativeMethods.grafeo_edge_id(edgePtr);
        var sourceId = (long)NativeMethods.grafeo_edge_source_id(edgePtr);
        var targetId = (long)NativeMethods.grafeo_edge_target_id(edgePtr);
        var typePtr = NativeMethods.grafeo_edge_type(edgePtr);
        var propsPtr = NativeMethods.grafeo_edge_properties_json(edgePtr);

        var edgeType = Marshal.PtrToStringUTF8(typePtr) ?? "";
        var propsJson = Marshal.PtrToStringUTF8(propsPtr) ?? "{}";
        var properties = ValueConverter.ParseObject(propsJson);

        return new Edge(id, edgeType, sourceId, targetId, properties);
    }

    /// <summary>Read vector search results from native pointers, then free them.</summary>
    private static IReadOnlyList<VectorResult> ReadVectorResults(
        nint idsPtr, nint distsPtr, nuint count)
    {
        try
        {
            if (count == 0) return Array.Empty<VectorResult>();
            var length = checked((int)count);
            var results = new VectorResult[length];
            unsafe
            {
                var ids = (ulong*)idsPtr;
                var dists = (float*)distsPtr;
                for (var i = 0; i < length; i++)
                    results[i] = new VectorResult((long)ids[i], dists[i]);
            }
            return results;
        }
        finally
        {
            NativeMethods.grafeo_free_vector_results(idsPtr, distsPtr, count);
        }
    }
}

// Owns the handle reservation before serialization can invoke user code, and before
// thread-pool scheduling. A stream takes over the invocation when its opener runs.
internal sealed class PreparedQuery : IDisposable
{
    private readonly IDisposable _lease;
    private readonly nint _pointer;
    private readonly string _query;
    private readonly string? _paramsJson;
    private NativeInvocation? _invocation;

    private PreparedQuery(IDisposable lease, nint pointer, string query, string? paramsJson, NativeInvocation invocation)
    {
        _lease = lease; _pointer = pointer; _query = query; _paramsJson = paramsJson; _invocation = invocation;
    }

    internal static PreparedQuery Create(IDisposable lease, nint pointer, string query,
        Func<string?> encodeParams, ExecutionOptions? options, CancellationToken token, bool streaming = false)
    {
        NativeInvocation? invocation = null;
        try
        {
            ValidateText(query, nameof(query));
            var paramsJson = encodeParams();
            if (paramsJson is not null)
            {
                ValidateText(paramsJson, nameof(paramsJson));
                using var document = System.Text.Json.JsonDocument.Parse(paramsJson);
                if (document.RootElement.ValueKind != System.Text.Json.JsonValueKind.Object)
                    throw new ArgumentException("Parameters must be a JSON object", nameof(paramsJson));
            }
            invocation = NativeInvocation.Create(options, token, streaming);
            return new PreparedQuery(lease, pointer, query, paramsJson, invocation);
        }
        catch { invocation?.Dispose(); lease.Dispose(); throw; }
    }

    private static void ValidateText(string text, string name)
    {
        ArgumentNullException.ThrowIfNull(text, name);
        if (text.Contains('\0')) throw new ArgumentException("Embedded NUL is not supported", name);
        _ = new System.Text.UTF8Encoding(false, true).GetByteCount(text);
    }

    internal unsafe QueryResult Execute(bool transaction)
    {
        var invocation = _invocation!;
        nint result;
        fixed (QueryOptions* options = &invocation.Options)
        {
            result = transaction
                ? NativeMethods.grafeo_transaction_execute_with_options(_pointer, _query, _paramsJson, (nint)options)
                : NativeMethods.grafeo_execute_with_options(_pointer, _query, _paramsJson, (nint)options);
            if (result == nint.Zero) throw invocation.CaptureError(GrafeoStatus.Query);
        }
        return QueryResultDecoder.Decode(result, invocation.CopyBytes);
    }

    internal ResultStream OpenStream()
    {
        var invocation = _invocation!;
        _invocation = null;
        return ResultStream.Open(_pointer, _query, _paramsJson, invocation);
    }

    public void Dispose()
    {
        try { Interlocked.Exchange(ref _invocation, null)?.Dispose(); }
        finally { _lease.Dispose(); }
    }
}

using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;
using System.Text.Json;
using Grafeo.Native;

namespace Grafeo;

/// <summary>
/// Pulls bounded rows or chunks from one native read cursor. Close releases
/// query ownership; terminal failures remain observable. Unsupported physical
/// operators return a native error. Use Next/NextChunk for larger results.
/// </summary>
public sealed class ResultStream : IDisposable, IAsyncDisposable
{
    private readonly object _sync = new();
    private readonly NativeInvocation _invocation;
    private readonly IReadOnlyList<string> _columns;
    private nint _handle;
    private int _active;
    private int _closing;
    private bool _closed;
    private bool _terminal;
    private Exception? _failure;

    private ResultStream(nint handle, IReadOnlyList<string> columns, NativeInvocation invocation)
    { _handle = handle; _columns = columns; _invocation = invocation; }

    // This factory consumes invocation, including all failure paths.
    internal static unsafe ResultStream Open(nint database, string query, string? parameters, NativeInvocation invocation)
    {
        nint stream = 0;
        try
        {
            fixed (QueryOptions* options = &invocation.Options)
                stream = NativeMethods.grafeo_stream_open_with_options(database, query, parameters, (nint)options);
            if (stream == 0) throw invocation.CaptureError(GrafeoStatus.Query);
            var columnsPtr = NativeMethods.grafeo_stream_columns_json(stream);
            if (columnsPtr == 0) throw invocation.CaptureError(GrafeoStatus.Query);
            IReadOnlyList<string> columns;
            try
            {
                var budget = invocation.CopyBytes;
                columns = Array.AsReadOnly(ValueConverter.ParseStringArray(ManagedCopyBudget.ReadUtf8(columnsPtr, ref budget)).ToArray());
            }
            finally { NativeMethods.grafeo_free_string(columnsPtr); }
            return new ResultStream(stream, columns, invocation);
        }
        catch (Exception primary)
        {
            Exception? cleanup = null;
            if (stream != 0)
            {
                var status = NativeMethods.grafeo_stream_close(stream);
                if (status != 0) cleanup = invocation.CaptureError((GrafeoStatus)status);
                NativeMethods.grafeo_stream_free(stream);
            }
            invocation.Dispose();
            if (cleanup != null) primary.Data["GrafeoCleanupError"] = cleanup;
            throw;
        }
    }

    /// <summary>Immutable column names in native order.</summary>
    public IReadOnlyList<string> Columns => _columns;

    private void Check()
    {
        if (_failure != null) ExceptionDispatchInfo.Capture(_failure).Throw();
        if (_closed || Volatile.Read(ref _closing) != 0) throw new ObjectDisposedException(nameof(ResultStream));
    }

    private void Finish(Exception? primary = null)
    {
        if (_terminal)
        {
            if (_failure == null && primary != null) _failure = primary;
            if (_failure != null) ExceptionDispatchInfo.Capture(_failure).Throw();
            return;
        }
        _terminal = true;
        Exception? cleanup = null;
        try
        {
            var status = NativeMethods.grafeo_stream_close(_handle);
            if (status != 0) cleanup = _invocation.CaptureError((GrafeoStatus)status);
        }
        finally
        {
            NativeMethods.grafeo_stream_free(_handle);
            _handle = 0;
            _invocation.Dispose();
        }
        _failure = primary ?? cleanup;
        if (primary != null && cleanup != null) primary.Data["GrafeoCleanupError"] = cleanup;
        if (_failure != null) ExceptionDispatchInfo.Capture(_failure).Throw();
    }

    /// <summary>Returns the next row, or null at clean exhaustion.</summary>
    public Dictionary<string, object?>? Next()
    {
        var budget = _invocation.CopyBytes;
        return NextBounded(ref budget);
    }

    private Dictionary<string, object?>? NextBounded(ref ulong budget)
    {
        Interlocked.Increment(ref _active);
        try
        {
            lock (_sync)
            {
                Check();
                if (_terminal) return null;
                var status = NativeMethods.grafeo_stream_next_row_json(_handle, out var rowPtr);
                if (status != 0) { Finish(_invocation.CaptureError((GrafeoStatus)status)); return null; }
                if (rowPtr == 0) { Finish(); return null; }
                try
                {
                    var json = ManagedCopyBudget.ReadUtf8(rowPtr, ref budget);
                    return JsonSerializer.Deserialize<Dictionary<string, object?>>(json, new JsonSerializerOptions { MaxDepth = 1024 })
                        ?? throw new GrafeoException("Native row is not an object", GrafeoStatus.Serialization);
                }
                catch (Exception error) { Finish(error); throw; }
                finally { NativeMethods.grafeo_free_string(rowPtr); }
            }
        }
        finally { Interlocked.Decrement(ref _active); GC.KeepAlive(this); }
    }

    /// <summary>Returns at most maxRows (native cap 1024) and the copy byte cap.</summary>
    public QueryResult? NextChunk(int maxRows)
    {
        if (maxRows <= 0) throw new ArgumentOutOfRangeException(nameof(maxRows));
        Interlocked.Increment(ref _active);
        try
        {
            lock (_sync)
            {
                Check();
                if (_terminal) return null;
                var status = NativeMethods.grafeo_stream_next_chunk(_handle, (nuint)maxRows, out var result);
                if (status != 0) { Finish(_invocation.CaptureError((GrafeoStatus)status)); return null; }
                if (result == 0) { Finish(); return null; }
                try { return QueryResultDecoder.Decode(result, _invocation.CopyBytes); }
                catch (Exception error) { Finish(error); throw; }
            }
        }
        finally { Interlocked.Decrement(ref _active); GC.KeepAlive(this); }
    }

    /// <summary>Runs a native pull on the thread pool; token cancellation reaches native execution.</summary>
    public Task<Dictionary<string, object?>?> NextAsync(CancellationToken cancellationToken = default) =>
        Task.Run(() =>
        {
            using var registration = cancellationToken.Register(_invocation.Cancel);
            return Next();
        }, CancellationToken.None);

    /// <summary>Iterates rows and closes on exhaustion, error or early iterator disposal.</summary>
    public IEnumerable<Dictionary<string, object?>> Rows()
    {
        try
        {
            while (Next() is { } row) yield return row;
        }
        finally { Close(); }
    }

    /// <summary>Asynchronously iterates rows with native token cancellation and early close.</summary>
    public async IAsyncEnumerable<Dictionary<string, object?>> RowsAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        try
        {
            while (await NextAsync(cancellationToken).ConfigureAwait(false) is { } row) yield return row;
        }
        finally { await DisposeAsync().ConfigureAwait(false); }
    }

    /// <summary>Collects remaining rows within total row/byte caps; failure returns no partial list.</summary>
    public List<Dictionary<string, object?>> ToList()
    {
        var rows = new List<Dictionary<string, object?>>();
        var remaining = _invocation.CopyBytes;
        while (NextBounded(ref remaining) is { } row)
        {
            if ((ulong)rows.Count >= _invocation.CollectRows) FailCollection("Collection exceeds row cap");
            if (rows.Count == rows.Capacity)
            {
                if (rows.Count == Array.MaxLength) FailCollection("Collection exceeds managed array capacity");
                var capacity = rows.Capacity == 0 ? 4
                    : (int)Math.Min((long)rows.Capacity * 2, Array.MaxLength);
                var allocation = checked((ulong)capacity * (ulong)IntPtr.Size + 32);
                if (allocation > remaining) FailCollection("Collection capacity exceeds byte cap");
                remaining -= allocation;
                // Charge the entire new allocation while the old array is live.
                rows.Capacity = capacity;
            }
            rows.Add(row);
        }
        return rows;
    }

    private void FailCollection(string message)
    {
        lock (_sync) Finish(ManagedCopyBudget.Limit(message));
    }

    /// <summary>Interrupts active pulls, joins them and closes once; failures remain observable.</summary>
    public void Close()
    {
        Interlocked.Exchange(ref _closing, 1);
        if (Volatile.Read(ref _active) != 0) _invocation.Cancel();
        lock (_sync)
        {
            if (_closed)
            {
                if (_failure != null) ExceptionDispatchInfo.Capture(_failure).Throw();
                return;
            }
            _closed = true;
            try { Finish(); }
            finally { GC.SuppressFinalize(this); }
        }
    }

    public void Dispose() => Close();
    public ValueTask DisposeAsync() => new(Task.Run(Close, CancellationToken.None));
    ~ResultStream() { try { Close(); } catch { /* Finalizer cleanup is best effort; explicit Close is fallible. */ } }
}

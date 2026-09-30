// ACID transaction owner: one reserved native operation at a time.
using System.Runtime.InteropServices;
using Grafeo.Native;

namespace Grafeo;

/// <summary>An ACID transaction, automatically rolled back when disposed unfinished.</summary>
public sealed class Transaction : ITransaction, IDisposable, IAsyncDisposable
{
    private readonly TransactionHandle _handle;
    private readonly object _gate = new();
    private bool _active;
    private bool _finished;
    private bool _disposed;

    internal Transaction(nint ptr, GrafeoDB.NativeLease parentLease)
    {
        _handle = new TransactionHandle { ParentLease = parentLease };
        Marshal.InitHandle(_handle, ptr);
    }

    /// <summary>Execute a query with cancellation and bounded eager output.</summary>
    public QueryResult ExecuteWithOptions(string query, ExecutionOptions? options = null,
        Dictionary<string, object?>? parameters = null, CancellationToken cancellationToken = default)
    {
        using var prepared = Prepare(query, () => parameters is null ? null : ValueConverter.EncodeParams(parameters),
            options, cancellationToken);
        return prepared.Execute(true);
    }

    /// <summary>Reserve transaction and query owners before scheduling native execution.</summary>
    public Task<QueryResult> ExecuteWithOptionsAsync(string query, ExecutionOptions? options = null,
        Dictionary<string, object?>? parameters = null, CancellationToken cancellationToken = default)
    {
        var prepared = Prepare(query, () => parameters is null ? null : ValueConverter.EncodeParams(parameters),
            options, cancellationToken);
        return Schedule(prepared);
    }

    private PreparedQuery Prepare(string query, Func<string?> encodeParams,
        ExecutionOptions? options, CancellationToken token)
    {
        var lease = Acquire();
        return PreparedQuery.Create(lease, lease.Pointer, query, encodeParams, options, token);
    }

    private static Task<QueryResult> Schedule(PreparedQuery prepared)
    {
        try
        {
            return Task.Run(() => { using (prepared) return prepared.Execute(true); }, CancellationToken.None);
        }
        catch { prepared.Dispose(); throw; }
    }

    /// <summary>Execute a GQL query within this transaction.</summary>
    public QueryResult Execute(string query) => ExecuteWithOptions(query);

    /// <summary>Execute a GQL query on the thread pool with native cancellation.</summary>
    public Task<QueryResult> ExecuteAsync(string query, CancellationToken ct = default)
        => ExecuteWithOptionsAsync(query, cancellationToken: ct);

    /// <summary>Execute a query with typed parameters.</summary>
    public QueryResult ExecuteWithParams(string query, Dictionary<string, object?> parameters)
        => ExecuteWithOptions(query, parameters: parameters);

    /// <summary>Execute a parameterized query on the thread pool.</summary>
    public Task<QueryResult> ExecuteWithParamsAsync(string query,
        Dictionary<string, object?> parameters, CancellationToken ct = default)
        => ExecuteWithOptionsAsync(query, parameters: parameters, cancellationToken: ct);

    /// <summary>Execute a query in the given language with optional JSON object parameters.</summary>
    public QueryResult ExecuteLanguage(string language, string query, string? paramsJson = null)
    {
        using var prepared = Prepare(query, () => paramsJson, new ExecutionOptions { Language = language }, default);
        return prepared.Execute(true);
    }

    /// <summary>Execute a query in the given language on the thread pool.</summary>
    public Task<QueryResult> ExecuteLanguageAsync(string language, string query,
        string? paramsJson = null, CancellationToken ct = default)
        => Schedule(Prepare(query, () => paramsJson, new ExecutionOptions { Language = language }, ct));

    /// <summary>Commit the transaction and release its native owner on success.</summary>
    public void Commit() => Complete(true);

    /// <summary>Roll back the transaction and release its native owner on success.</summary>
    public void Rollback()
    {
        if (!Monitor.TryEnter(_gate)) throw GrafeoDB.Busy();
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_active) throw GrafeoDB.Busy();
            if (_finished) return;
        }
        finally { Monitor.Exit(_gate); }
        Complete(false);
    }

    private void Complete(bool commit)
    {
        using var lease = Acquire();
        var status = commit ? NativeMethods.grafeo_commit(lease.Pointer) : NativeMethods.grafeo_rollback(lease.Pointer);
        if (status != (int)GrafeoStatus.Ok)
            throw GrafeoException.FromLastError(GrafeoStatus.Transaction);
        lock (_gate)
        {
            _finished = true;
            _handle.Committed = true; // Neither completed branch needs another rollback.
            _handle.Dispose(); // The operation lease releases the final reference.
        }
    }

    /// <summary>Roll back unfinished work. Reject disposal while a call is reserved.</summary>
    public void Dispose()
    {
        if (!Monitor.TryEnter(_gate)) throw GrafeoDB.Busy();
        try
        {
            if (_disposed) return;
            if (_active) throw GrafeoDB.Busy();
            _disposed = true;
            _finished = true;
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

    private OperationLease Acquire()
    {
        if (!Monitor.TryEnter(_gate)) throw GrafeoDB.Busy();
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_active) throw GrafeoDB.Busy();
            if (_finished) throw new TransactionException("Transaction is already committed or rolled back");
            var lease = new OperationLease(this);
            var retained = false;
            try
            {
                _handle.DangerousAddRef(ref retained);
                _active = true;
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

    private sealed class OperationLease : IDisposable
    {
        private Transaction? _owner;
        internal nint Pointer { get; }
        internal OperationLease(Transaction owner)
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
                owner._active = false;
            }
        }
    }
}

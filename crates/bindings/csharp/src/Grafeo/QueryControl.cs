using Grafeo.Native;

namespace Grafeo;

/// <summary>Single-use native execution owner with independent cancellation.
/// Cancel and Dispose may run concurrently with execution. Dispose releases
/// this wrapper without cancelling or freeing an execution already started.</summary>
public sealed class QueryControl : IDisposable
{
    private readonly object _gate = new();
    private nint _owner;
    private nint _cancellation;
    private bool _consumed;
    private bool _disposed;

    /// <summary>Starts the timeout now; null means no deadline, zero means an
    /// immediate deadline, and positive values round up to milliseconds.</summary>
    public QueryControl(TimeSpan? timeout = null)
    {
        long milliseconds = -1;
        if (timeout is { } duration)
        {
            if (duration < TimeSpan.Zero)
                throw new ArgumentOutOfRangeException(nameof(timeout), "Timeout must be nonnegative.");
            milliseconds = duration.Ticks / TimeSpan.TicksPerMillisecond;
            if (duration.Ticks % TimeSpan.TicksPerMillisecond != 0)
                milliseconds = checked(milliseconds + 1);
        }
        _owner = NativeMethods.grafeo_query_control_create(milliseconds);
        if (_owner == nint.Zero)
            throw GrafeoException.FromLastError();
        _cancellation = NativeMethods.grafeo_query_control_cancel_handle(_owner);
        if (_cancellation == nint.Zero)
        {
            var error = GrafeoException.FromLastError();
            NativeMethods.grafeo_query_control_free(_owner);
            _owner = nint.Zero;
            throw error;
        }
    }

    /// <summary>Requests cancellation without waiting for the executing call.</summary>
    public void Cancel()
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            GrafeoException.ThrowIfFailed(NativeMethods.grafeo_cancel(_cancellation));
            GC.KeepAlive(this);
        }
    }

    internal (nint Owner, nint Cancellation) Begin()
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_consumed)
                throw new InvalidOperationException("Query control has already been consumed.");
            var cancellation = NativeMethods.grafeo_cancel_handle_clone(_cancellation);
            if (cancellation == nint.Zero)
                throw GrafeoException.FromLastError();
            _consumed = true;
            var owner = _owner;
            _owner = nint.Zero;
            GC.KeepAlive(this);
            return (owner, cancellation);
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true;
            if (_owner != nint.Zero) NativeMethods.grafeo_query_control_free(_owner);
            if (_cancellation != nint.Zero) NativeMethods.grafeo_cancel_handle_free(_cancellation);
            _owner = nint.Zero;
            _cancellation = nint.Zero;
        }
        GC.SuppressFinalize(this);
    }

    ~QueryControl() => Dispose();
}

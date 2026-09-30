// SafeHandle subclasses for automatic native resource cleanup.

using System.Runtime.InteropServices;

namespace Grafeo.Native;

/// Blittable layout shared with GrafeoQueryOptions in grafeo.h.
[StructLayout(LayoutKind.Sequential)]
internal struct QueryOptions
{
    internal nint Control;
    internal nuint MaxRows;
    internal nuint MaxBytes;
    internal nint Language;
}

/// <summary>
/// Safe handle wrapping a <c>GrafeoDatabase*</c>.
/// Calls <c>grafeo_close</c> + <c>grafeo_free_database</c> on release.
/// </summary>
internal sealed class DatabaseHandle : SafeHandle
{
    public DatabaseHandle() : base(nint.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == nint.Zero;

    internal bool Closed;

    protected override bool ReleaseHandle()
    {
        if (!Closed) NativeMethods.grafeo_close(handle);
        NativeMethods.grafeo_free_database(handle);
        return true;
    }
}

/// <summary>
/// Safe handle wrapping a <c>GrafeoTransaction*</c>.
/// Auto-rolls back and frees on release.
/// </summary>
internal sealed class TransactionHandle : SafeHandle
{
    public TransactionHandle() : base(nint.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == nint.Zero;

    internal volatile bool Committed;
    internal IDisposable? ParentLease;

    protected override bool ReleaseHandle()
    {
        try
        {
            if (!Committed) NativeMethods.grafeo_rollback(handle);
            NativeMethods.grafeo_free_transaction(handle);
            return true;
        }
        finally
        {
            ParentLease?.Dispose();
            ParentLease = null;
        }
    }
}

/// <summary>
/// Safe handle wrapping a <c>GrafeoResult*</c>.
/// Frees the result on release.
/// </summary>
internal sealed class ResultHandle : SafeHandle
{
    public ResultHandle() : base(nint.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        NativeMethods.grafeo_free_result(handle);
        return true;
    }
}

/// Safe handle for the query control's cancellation/deadline owner.
internal sealed class QueryControlHandle : SafeHandle
{
    public QueryControlHandle() : base(nint.Zero, ownsHandle: true) { }
    public override bool IsInvalid => handle == nint.Zero;
    protected override bool ReleaseHandle()
    {
        NativeMethods.grafeo_query_control_free(handle);
        return true;
    }
}

/// Safe handle for a cloneable cancellation authority.
internal sealed class CancelHandle : SafeHandle
{
    public CancelHandle() : base(nint.Zero, ownsHandle: true) { }
    public override bool IsInvalid => handle == nint.Zero;
    protected override bool ReleaseHandle()
    {
        NativeMethods.grafeo_cancel_handle_free(handle);
        return true;
    }
}

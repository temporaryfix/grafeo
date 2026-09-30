using System.Runtime.InteropServices;
using System.Text;

namespace Grafeo.Native;

/// Retains native allocations through an invocation and its cancellation callback.
internal sealed class NativeInvocation : IDisposable
{
    private readonly object _gate = new();
    private readonly object _disposeGate = new();
    private readonly CancellationToken _token;
    private CancellationTokenRegistration _registration;
    private nint _cancellation;
    private bool _disposed;

    internal QueryOptions Options;
    internal ulong CopyBytes { get; }
    internal ulong CollectRows { get; }

    private NativeInvocation(nint owner, nint cancellation, nint language,
        ulong rows, ulong bytes, ulong collectRows, CancellationToken token)
    {
        // .NET JSON strings and backing arrays use signed 32-bit lengths.
        // Keep this capacity denial inside native precommit admission.
        var nativeBytes = Math.Min(bytes / 4, (ulong)int.MaxValue);
        Options = new QueryOptions
        {
            Control = owner, MaxRows = checked((nuint)rows),
            MaxBytes = checked((nuint)nativeBytes), Language = language,
        };
        CopyBytes = bytes - nativeBytes;
        CollectRows = collectRows;
        _cancellation = cancellation;
        _token = token;
    }

    internal static NativeInvocation Create(ExecutionOptions? options,
        CancellationToken token, bool streaming)
    {
        // Snapshot and validate every option before reserving the single use.
        var control = options?.Control;
        var explicitRows = options?.MaxRows;
        var rows = explicitRows ?? (streaming ? (ulong)nuint.MaxValue : 1_000_000UL);
        var bytes = options?.MaxBytes ?? 64UL * 1024 * 1024;
        var language = options?.Language;
        if (rows > (ulong)nuint.MaxValue || bytes > (ulong)nuint.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(options), "Limits exceed the native size_t range.");
        if (language?.Contains('\0') == true)
            throw new ArgumentException("Language contains NUL.", nameof(options));
        if (language is not null)
            _ = new UTF8Encoding(false, true).GetByteCount(language);

        var languagePtr = string.IsNullOrEmpty(language) ? nint.Zero : Marshal.StringToCoTaskMemUTF8(language);
        QueryControl? temporary = null;
        NativeInvocation? invocation = null;
        nint owner = nint.Zero, cancellation = nint.Zero;
        try
        {
            control ??= temporary = new QueryControl();
            (owner, cancellation) = control.Begin();
            invocation = new NativeInvocation(owner, cancellation, languagePtr,
                rows, bytes, explicitRows ?? 1_000_000UL, token);
            languagePtr = nint.Zero;
            owner = cancellation = nint.Zero;
            // Register synchronously invokes cancellation for an already-cancelled
            // token, before the caller can enter native execution.
            invocation._registration = token.Register(static state => ((NativeInvocation)state!).Cancel(), invocation);
            return invocation;
        }
        catch
        {
            invocation?.Dispose();
            if (owner != nint.Zero) NativeMethods.grafeo_query_control_free(owner);
            if (cancellation != nint.Zero) NativeMethods.grafeo_cancel_handle_free(cancellation);
            throw;
        }
        finally
        {
            Marshal.FreeCoTaskMem(languagePtr);
            temporary?.Dispose();
        }
    }

    internal void Cancel()
    {
        lock (_gate)
        {
            if (_cancellation != nint.Zero)
                NativeMethods.grafeo_cancel(_cancellation);
        }
    }

    internal Exception CaptureError(GrafeoStatus status = GrafeoStatus.Database) =>
        Translate(GrafeoException.FromLastError(status));

    internal Exception CaptureError(int status) => CaptureError((GrafeoStatus)status);

    internal Exception Translate(Exception error) => error is GrafeoException native &&
        (native.Code == "GRAFEO-Q007" || native.Status == GrafeoStatus.Cancelled)
        ? new QueryCanceledException(native.Message, native.Code, _token, native)
        : error;

    public void Dispose()
    {
        lock (_disposeGate)
        {
            if (_disposed) return;
            // Dispose waits for an executing callback. Never hold _gate while
            // joining, since that callback acquires it to cancel safely.
            _registration.Dispose();
            lock (_gate)
            {
                _disposed = true;
                NativeMethods.grafeo_cancel_handle_free(_cancellation);
                _cancellation = nint.Zero;
                NativeMethods.grafeo_query_control_free(Options.Control);
                Options.Control = nint.Zero;
                Marshal.FreeCoTaskMem(Options.Language);
                Options.Language = nint.Zero;
            }
        }
    }
}

/// Predecode reservation against a shared native/managed copy envelope.
internal static class ManagedCopyBudget
{
    internal static StorageException Limit(string message) =>
        new(message, GrafeoStatus.ResourceLimit) { Code = "GRAFEO-S001" };

    internal static unsafe string ReadUtf8(nint pointer, ref ulong remaining)
    {
        if (pointer == nint.Zero)
            throw new GrafeoException("Null native JSON pointer.", GrafeoStatus.NullPointer);
        var data = (byte*)pointer;
        ulong cost = 0;
        int length = 0;
        bool inString = false, escaped = false;
        while (data[length] != 0)
        {
            if (length == int.MaxValue)
                throw Limit("Native JSON exceeds the managed string range.");
            byte value = data[length++];
            Charge(6, ref cost, remaining); // UTF-8 intermediates and UTF-16 copies.
            if (inString)
            {
                if (escaped) escaped = false;
                else if (value == (byte)'\\') escaped = true;
                else if (value == (byte)'"') inString = false;
                continue;
            }
            switch (value)
            {
                case (byte)'"': inString = true; Charge(64, ref cost, remaining); break;
                case (byte)'{': Charge(576, ref cost, remaining); break;
                case (byte)':': Charge(320, ref cost, remaining); break;
                case (byte)'[': Charge(128, ref cost, remaining); break;
                case (byte)',': Charge(96, ref cost, remaining); break;
            }
        }
        remaining -= cost;
        return Encoding.UTF8.GetString(new ReadOnlySpan<byte>(data, length));
    }

    private static void Charge(ulong amount, ref ulong cost, ulong limit)
    {
        if (amount > limit - cost)
            throw Limit("Managed JSON copies exceed the byte envelope.");
        cost += amount;
    }
}

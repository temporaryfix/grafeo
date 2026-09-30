// Error handling: status codes, exception hierarchy, and error retrieval.

using System.Runtime.InteropServices;

using Grafeo.Native;

namespace Grafeo;

/// <summary>
/// Status codes returned by grafeo-c FFI functions.
/// Values match the <c>GrafeoStatus</c> repr(C) enum in <c>error.rs</c>.
/// </summary>
public enum GrafeoStatus
{
    Ok = 0,
    Database = 1,
    Query = 2,
    Transaction = 3,
    Storage = 4,
    Io = 5,
    Serialization = 6,
    Internal = 7,
    NullPointer = 8,
    InvalidUtf8 = 9,
    Cancelled = 10,
    Deadline = 11,
    ResourceLimit = 12,
}

/// <summary>Base exception for all Grafeo errors.</summary>
public class GrafeoException : Exception
{
    /// <summary>The native status code that produced this error.</summary>
    public GrafeoStatus Status { get; }

    /// <summary>Stable native error code, when the native layer supplied one.</summary>
    public string? Code { get; internal set; }

    public GrafeoException(string message, GrafeoStatus status)
        : base(message) => Status = status;

    public GrafeoException(string message, GrafeoStatus status, Exception innerException)
        : base(message, innerException) => Status = status;

    /// <summary>
    /// Retrieve the last error message from the native layer and return a
    /// typed exception.
    /// </summary>
    internal static GrafeoException FromLastError(
        GrafeoStatus fallbackStatus = GrafeoStatus.Database)
    {
        var errorPtr = NativeMethods.grafeo_last_error();
        var message = errorPtr != nint.Zero
            ? Marshal.PtrToStringUTF8(errorPtr) ?? "Unknown error"
            : "Unknown error";
        var codePtr = NativeMethods.grafeo_last_error_code();
        var code = codePtr != nint.Zero ? Marshal.PtrToStringUTF8(codePtr) : null;
        var status = code switch
        {
            "GRAFEO-Q007" => GrafeoStatus.Cancelled,
            "GRAFEO-Q003" => GrafeoStatus.Deadline,
            "GRAFEO-S001" => GrafeoStatus.ResourceLimit,
            _ => fallbackStatus,
        };
        var error = Classify(status, message);
        error.Code = code;
        return error;
    }

    /// <summary>
    /// Map a native status code to the appropriate exception subclass.
    /// Mirrors <c>grafeo_bindings_common::error::classify_error</c>.
    /// </summary>
    internal static GrafeoException Classify(int statusCode, string message) =>
        Classify((GrafeoStatus)statusCode, message);

    internal static GrafeoException Classify(GrafeoStatus status, string message) =>
        status switch
        {
            GrafeoStatus.Query or GrafeoStatus.Cancelled or GrafeoStatus.Deadline => new QueryException(message, status),
            GrafeoStatus.Transaction => new TransactionException(message),
            GrafeoStatus.Storage or GrafeoStatus.Io or GrafeoStatus.ResourceLimit => new StorageException(message, status),
            GrafeoStatus.Serialization => new SerializationException(message),
            _ => new GrafeoException(message, status),
        };

    /// <summary>
    /// Throw if <paramref name="status"/> is not <see cref="GrafeoStatus.Ok"/>.
    /// </summary>
    internal static void ThrowIfFailed(int status)
    {
        if (status != (int)GrafeoStatus.Ok)
            throw FromLastError((GrafeoStatus)status);
    }
}

/// <summary>Query parsing or execution error.</summary>
public sealed class QueryException(string message, GrafeoStatus status = GrafeoStatus.Query)
    : GrafeoException(message, status);

/// <summary>Transaction lifecycle error (commit, rollback, isolation).</summary>
public sealed class TransactionException(string message)
    : GrafeoException(message, GrafeoStatus.Transaction);

/// <summary>Storage or I/O error (WAL, persistence, disk).</summary>
public sealed class StorageException(string message, GrafeoStatus status = GrafeoStatus.Storage)
    : GrafeoException(message, status);

/// <summary>JSON or value serialization error.</summary>
public sealed class SerializationException(string message)
    : GrafeoException(message, GrafeoStatus.Serialization);

/// <summary>A cancelled native execution, retaining its structured error code.</summary>
public sealed class QueryCanceledException : OperationCanceledException
{
    public string? Code { get; }
    public GrafeoStatus Status { get; }

    public QueryCanceledException(string message, string? code,
        CancellationToken cancellationToken = default, Exception? innerException = null)
        : base(message, innerException, cancellationToken)
    {
        Code = code;
        Status = GrafeoStatus.Cancelled;
    }
}

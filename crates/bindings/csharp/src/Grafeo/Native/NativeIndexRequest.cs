using System.Runtime.InteropServices;

namespace Grafeo.Native;

[StructLayout(LayoutKind.Sequential)]
internal struct NativeUtf8
{
    internal nint Data;
    internal nuint Length;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeIndexRequest
{
    internal uint Kind;
    internal uint Options;
    internal nint Graph;
    internal nuint GraphCount;
    internal NativeUtf8 Name;
    internal NativeUtf8 Label;
    internal NativeUtf8 Property;
    internal NativeUtf8 Metric;
    internal NativeUtf8 Quantization;
    internal nuint Dimensions;
    internal nuint M;
    internal nuint EfConstruction;
    internal nuint MinTokenLength;
}

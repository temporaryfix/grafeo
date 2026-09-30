using System.Reflection;
using Xunit;

namespace Grafeo.Tests;

public class ErrorTests
{
    [Theory]
    [InlineData(10, typeof(QueryException))]
    [InlineData(11, typeof(QueryException))]
    [InlineData(12, typeof(StorageException))]
    public void NativeControlStatusRetainsCategoryAndExactCode(int code, Type expected)
    {
        var classify = typeof(GrafeoException).GetMethod("Classify", BindingFlags.NonPublic | BindingFlags.Static,
            null, new[] { typeof(int), typeof(string) }, null);
        Assert.NotNull(classify);
        var error = Assert.IsAssignableFrom<GrafeoException>(classify.Invoke(null, new object[] { code, "native failure" }));
        Assert.IsType(expected, error);
        Assert.Equal((GrafeoStatus)code, error.Status);
    }
}

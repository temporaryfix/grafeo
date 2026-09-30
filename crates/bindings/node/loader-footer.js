
// Adapt the native null-terminated cursor to the standard async-iterator
// protocol. `return()` is awaited so `for await` early breaks release native
// resources before the loop's scope continues.
if (module.exports.ResultStream?.prototype && !module.exports.ResultStream.prototype[Symbol.asyncIterator]) {
  module.exports.ResultStream.prototype[Symbol.asyncIterator] = function asyncIterator() {
    const stream = this
    return {
      async next() {
        const value = await stream.next()
        return value === null ? { done: true, value: undefined } : { done: false, value }
      },
      async return() {
        await stream.close()
        return { done: true, value: undefined }
      },
      [Symbol.asyncIterator]() { return this },
    }
  }
}

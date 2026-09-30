// Public execution-control witnesses.  This file intentionally uses only the
// Node built-in test runner so it can load the native addon directly.
import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
// Load the package once: dlopen followed by require would register a second
// set of NAPI constructors and invalidate instanceof checks on earlier objects.
const require = createRequire(import.meta.url)
if (process.env.GRAFEO_NODE_LIBRARY) {
  process.env.NAPI_RS_NATIVE_LIBRARY_PATH ??= process.env.GRAFEO_NODE_LIBRARY
}
const { GrafeoDB, QueryControl } = require('../index.js')

function seed(count = 128) {
  const db = GrafeoDB.create()
  for (let i = 0; i < count; i += 1) db.createNode(['N'], { n: i })
  return db
}

function hasCode(code) {
  return (error) => error?.code === `GRAFEO-${code}`
}

describe('native execution controls', () => {
  it('is single-use and cancellation is observable before execution', async () => {
    const db = GrafeoDB.create()
    try {
      const control = new QueryControl()
      control.cancel()
      assert.equal(control.consumed, false)
      await assert.rejects(
        db.execute('MATCH (n) RETURN n', null, { control }),
        hasCode('Q007'),
      )
      assert.equal(control.consumed, true)
      assert.throws(() => db.execute('MATCH (n) RETURN n', null, { control }), /consumed|single-use/i)
    } finally {
      db.close()
    }
  })

  it('distinguishes a zero deadline from cancellation and keeps the event loop live', async () => {
    const db = GrafeoDB.create()
    try {
      const control = new QueryControl(0)
      let ticked = false
      const tick = new Promise((resolve) => setImmediate(() => { ticked = true; resolve() }))
      const query = db.execute('MATCH (n) RETURN n', null, { control })
      await assert.rejects(query, hasCode('Q003'))
      await tick
      assert.equal(ticked, true)
    } finally {
      db.close()
    }
  })

  it('cancels an in-flight Cartesian query and isolates independent owners', async () => {
    const db = seed()
    try {
      const cancelled = new QueryControl()
      let settled = false
      const observed = db.execute(
        'MATCH (a:N), (b:N), (c:N), (d:N) WHERE a.n + b.n + c.n + d.n < 0 RETURN a.n',
        null, { control: cancelled },
      ).then(
        (result) => { settled = true; return result },
        (error) => { settled = true; return error },
      )
      await new Promise((resolve) => setTimeout(resolve, 20))
      try {
        assert.equal(settled, false, 'the native query must still be running')
        assert.throws(() => db.close(), hasCode('T004'))
      } finally {
        cancelled.cancel()
      }
      assert.equal((await observed)?.code, 'GRAFEO-Q007')

      const other = new QueryControl()
      const result = await db.execute('MATCH (n:N) RETURN n.n', { unused: 1 }, { control: other })
      assert.equal(result.toArray().length, 128)
      assert.equal(other.consumed, true)
    } finally {
      db.close()
    }
  })

  it('enforces eager row and byte caps before a mutation commits', async () => {
    const db = GrafeoDB.create()
    try {
      await assert.rejects(
        db.execute('UNWIND [1, 2, 3] AS n RETURN n', null, { maxRows: 2 }),
        hasCode('S001'),
      )
      assert.equal(db.nodeCount(), 0)
      await assert.rejects(
        db.execute("INSERT (:N {payload: 'bounded'}) RETURN 1", null, { maxBytes: 1 }),
        hasCode('S001'),
      )
      assert.equal(db.nodeCount(), 0)
    } finally {
      db.close()
    }
  })

  it('denies copied output before publication and rolls back only the failed transaction statement', async () => {
    const db = GrafeoDB.create()
    try {
      const tx = db.beginTransaction()
      await tx.execute('INSERT (:Kept {n: 1})')
      await assert.rejects(
        tx.execute('INSERT (:Denied {payload: $payload}) RETURN $payload AS payload',
          { payload: 'x'.repeat(4096) }, { maxBytes: 16384 }),
        hasCode('S001'),
      )
      tx.commit()
      assert.equal(db.nodeCount(), 1)
      assert.equal((await db.execute('MATCH (n:Denied) RETURN n')).length, 0)
    } finally {
      db.close()
    }
  })

  it('enforces stream row and copied byte limits without truncation', async () => {
    const db = GrafeoDB.create()
    try {
      const rows = await db.executeStream('UNWIND [1, 2] AS n RETURN n', null, { maxRows: 1 })
      assert.deepEqual(await rows.next(), { n: 1 })
      await assert.rejects(rows.next(), hasCode('S001'))
      assert.equal(await rows.next(), null)
      await rows.close()
      const bytes = await db.executeStream('RETURN $payload AS payload',
        { payload: 'x'.repeat(4096) }, { maxBytes: 16384 })
      await assert.rejects(bytes.next(), hasCode('S001'))
      assert.equal(await bytes.next(), null)
      await bytes.close()
    } finally {
      db.close()
    }
  })

  it('keeps selected caps and values across copied result materializers', async () => {
    const db = GrafeoDB.create()
    try {
      const value = ['雪😀', [1, null], { key: 'value' }]
      const result = await db.execute('RETURN $value AS value', { value }, { maxBytes: 131072 })
      assert.deepEqual(result.scalar(), value)
      assert.deepEqual(result.get(0), { value })
      assert.deepEqual(result.toArray(), [{ value }])
      assert.deepEqual(result.rows(), [[value]])
      assert.deepEqual(result.columns, ['value'])
      assert.match(result.toString(), /value/)
      const small = await db.execute('RETURN 1 AS n', null, { maxBytes: 8192 })
      assert.throws(() => small.toArrowIPC(), hasCode('S001'))
      db.close()
      assert.deepEqual(result.rows(), [[value]])
    } finally {
      db.close()
    }
  })

  it('retains a raised byte cap on returned entity properties', async () => {
    const db = GrafeoDB.create()
    try {
      const payload = 'x'.repeat(4 * 1024 * 1024)
      db.createNode(['Large'], { payload })
      const result = await db.execute('MATCH (n:Large) RETURN n', null, { maxBytes: 256 * 1024 * 1024 })
      const [node] = result.nodes()
      assert(node)
      assert.deepEqual(node.labels, ['Large'])
      assert.equal(node.get('payload'), payload)
      assert.equal(node.properties().payload, payload)
    } finally {
      db.close()
    }
  })

  it('rejects database close promptly while a stream holds its lifecycle pin', async () => {
    const db = seed()
    const stream = await db.executeStream('MATCH (n:N) RETURN n.n')
    try {
      assert.throws(() => db.close(), hasCode('T004'))
      assert(await stream.next())
    } finally {
      await stream.close()
      db.close()
    }
    db.close()
  })

  it('cancels one stream without affecting a concurrently retained owner', async () => {
    const db = GrafeoDB.create()
    const control = new QueryControl()
    const first = await db.executeStream('UNWIND [1, 2] AS n RETURN n', null, { control })
    const second = await db.executeStream('UNWIND [3, 4] AS n RETURN n')
    try {
      control.cancel()
      await assert.rejects(first.next(), hasCode('Q007'))
      assert.deepEqual(await second.next(), { n: 3 })
      assert.deepEqual(await second.next(), { n: 4 })
      assert.equal(await second.next(), null)
      assert.equal(await first.next(), null)
    } finally {
      await first.close().catch((error) => assert.equal(error?.code, 'GRAFEO-Q007'))
      await second.close()
      db.close()
    }
  })

  it('supports parameterized streams and idempotent close', async () => {
    const db = GrafeoDB.create()
    try {
      const stream = await db.executeStream(
        'UNWIND [$value] AS n RETURN n',
        { value: 'ok' },
        { maxRows: 1 },
      )
      assert.deepEqual(await stream.next(), { n: 'ok' })
      assert.equal(await stream.next(), null)
      await stream.close()
      await stream.close()
      assert.equal(await stream.next(), null)
    } finally {
      db.close()
    }
  })

  it('passes options through explicit language and transaction routes', async () => {
    const db = GrafeoDB.create()
    try {
      const result = await db.executeCypher('RETURN $value AS value', { value: 7 }, { maxRows: 1 })
      assert.deepEqual(result.toArray(), [{ value: 7 }])
      const tx = db.beginTransaction()
      const txResult = await tx.execute('RETURN $value AS value', { value: 8 }, { maxRows: 1 })
      assert.deepEqual(txResult.toArray(), [{ value: 8 }])
      tx.rollback()
    } finally {
      db.close()
    }
  })

  it('reserves a transaction while its query is pending and restores it after cancellation', async () => {
    const db = seed()
    try {
      const tx = db.beginTransaction()
      const control = new QueryControl()
      const pending = tx.execute('MATCH (a:N), (b:N) RETURN a.n, b.n', null, { control })
      assert.throws(() => tx.commit(), /busy|executing/i)
      assert.throws(() => tx.rollback(), /busy|executing/i)
      assert.throws(() => tx.execute('CREATE (:N)'), /busy|executing/i)
      setImmediate(() => control.cancel())
      await assert.rejects(pending, hasCode('Q007'))
      tx.rollback()
      const result = await db.execute('RETURN 1 AS ok')
      assert.deepEqual(result.toArray(), [{ ok: 1 }])
    } finally {
      db.close()
    }
  })

  it('rejects reentrant transaction operations from an options getter', async () => {
    const db = seed()
    try {
      const tx = db.beginTransaction()
      const control = new QueryControl()
      const options = {
        get control() {
          assert.throws(() => tx.commit(), /busy|executing/i)
          return control
        },
      }
      const pending = tx.execute('MATCH (a:N), (b:N) RETURN a.n, b.n', null, options)
      setImmediate(() => control.cancel())
      await assert.rejects(pending, hasCode('Q007'))
      tx.rollback()
    } finally {
      db.close()
    }
  })

  it('keeps a dropped transaction query from blocking the event loop', async () => {
    const db = seed()
    try {
      assert.equal(typeof global.gc, 'function', 'run this witness with --expose-gc')
      let tx = db.beginTransaction()
      const pending = tx.execute('MATCH (a:N), (b:N) RETURN a.n, b.n')
      const finalized = { value: false }
      const registry = new FinalizationRegistry(() => { finalized.value = true })
      registry.register(tx, null)
      tx = null
      global.gc()
      await new Promise((resolve) => setImmediate(resolve))
      await pending
      // V8 schedules finalizers independently of the query's Promise. Wait
      // for that actual event before asserting native transaction cleanup.
      for (let i = 0; !finalized.value && i < 100; i += 1) {
        global.gc()
        await new Promise((resolve) => setImmediate(resolve))
      }
      assert.equal(finalized.value, true, 'the transaction wrapper was collected')
      assert.deepEqual((await db.execute('RETURN 1 AS ok')).toArray(), [{ ok: 1 }])
    } finally {
      db.close()
    }
  })

  it('closes native resources when the index.js async iterator is broken early', async () => {
    const wrapped = require('../index.js')
    const db = wrapped.GrafeoDB.create()
    try {
      for (let i = 0; i < 4; i += 1) db.createNode(['N'], { n: i })
      const stream = await db.executeStream('MATCH (n:N) RETURN n.n')
      for await (const row of stream) {
        assert.equal(typeof row['n.n'], 'number')
        break
      }
      assert.equal(await stream.next(), null)
    } finally {
      db.close()
    }
  })

  it('close interrupts an active stream pull without a separate cancel', async () => {
    const db = GrafeoDB.create()
    try {
      for (let i = 0; i < 128; i += 1) db.createNode(['CancelWork'], { i })
      const control = new QueryControl()
      const stream = await db.executeStream(
        'MATCH (a:CancelWork), (b:CancelWork), (c:CancelWork), (d:CancelWork) WHERE a.i + b.i + c.i + d.i < 0 RETURN a.i',
        null,
        { control },
      )
      let settled = false
      const observed = stream.next().then(
        (row) => { settled = true; return row },
        (error) => { settled = true; return error },
      )
      await new Promise((resolve) => setTimeout(resolve, 20))
      assert.equal(settled, false, 'the rejecting Cartesian pull must still be running')
      let timer
      try {
        await Promise.race([
          stream.close(),
          new Promise((_, reject) => {
            timer = setTimeout(() => {
              // Failure cleanup must not leave a stuck worker holding the DB.
              control.cancel()
              reject(new Error('stream close timed out'))
            }, 2_000)
          }),
        ]).catch((error) => assert.equal(error?.code, 'GRAFEO-Q007'))
      } finally {
        clearTimeout(timer)
      }
      const outcome = await observed
      assert.equal(outcome?.code, 'GRAFEO-Q007')
      assert.equal(await stream.next(), null)
      db.createNode(['AfterClose'], { ok: true })
      assert.equal(db.nodeCount(), 129)
    } finally {
      db.close()
    }
  })

  it('rejects invalid limits without consuming a control', () => {
    const db = GrafeoDB.create()
    try {
      const paramsControl = new QueryControl()
      assert.throws(() => db.execute('RETURN 1', [], { control: paramsControl }))
      assert.equal(paramsControl.consumed, false)
      assert.throws(() => db.execute('RETURN 1', null, { control: {} }))
      for (const options of [{ maxRows: -1 }, { maxRows: 1.5 }, { maxBytes: -1 }, { maxBytes: Infinity }, { maxBytes: Number.MAX_SAFE_INTEGER + 1 }]) {
        const control = new QueryControl()
        assert.throws(
          () => db.execute('RETURN 1 AS value', null, { ...options, control }),
          /maxRows|maxBytes|positive|integer/i,
        )
        assert.equal(control.consumed, false)
      }
    } finally {
      db.close()
    }
  })
})

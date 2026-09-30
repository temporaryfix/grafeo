import { it } from 'node:test'
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { copyFileSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
const require = createRequire(import.meta.url)
if (process.env.GRAFEO_NODE_LIBRARY) process.env.NAPI_RS_NATIVE_LIBRARY_PATH ??= process.env.GRAFEO_NODE_LIBRARY
const { GrafeoDB } = require('../index.js')
const code = expected => error => error?.code === `GRAFEO-${expected}`

it('resumes bounded owned pages with exact coordinates and structured errors', async () => {
  const db = GrafeoDB.create()
  const other = GrafeoDB.create()
  try {
    db.enableCdc()
    db.createNode(['First'])
    db.createNode(['Second'])
    db.disableCdc()
    db.createNode(['Uncaptured'])
    const first = await db.changesAfter(null, 1, 4096)
    assert.equal(first.events.length, 1)
    assert.equal(first.next.length, 97)
    for (const field of ['entity_id', 'epoch', 'timestamp', 'graph_incarnation']) {
      assert.match(first.events[0][field], /^\d+$/)
    }
    // Native HLC is larger than Number.MAX_SAFE_INTEGER; decimal transport is exact.
    assert.ok(BigInt(first.events[0].timestamp) > BigInt(Number.MAX_SAFE_INTEGER))
    const savedCursor = Buffer.from(first.next)
    const pending = db.changesAfter(first.next, 1, 4096)
    first.next.fill(0)
    const second = await pending
    assert.equal(second.events.length, 1)
    assert.notEqual(second.events[0].entity_id, first.events[0].entity_id)
    assert.deepEqual((await db.changesAfter(second.next, 1, 4096)).events, [])
    assert.deepEqual((await db.changesAfter(second.next, 1, 4096)).next, second.next)
    await assert.rejects(db.changesAfter(Buffer.from('bad'), 1, 4096), code('S004'))
    const foreign = await other.changesAfter(null, 1, 4096)
    await assert.rejects(db.changesAfter(foreign.next, 1, 4096), code('S005'))
    await assert.rejects(db.changesAfter(null, 1, 1), code('S001'))
    for (const bound of [0, -1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
      await assert.rejects(db.changesAfter(null, bound, 4096))
    }
    const saved = JSON.stringify(second.events)
    db.close()
    await assert.rejects(db.changesAfter(savedCursor, 1, 4096))
    assert.equal(JSON.stringify(second.events), saved)
  } finally {
    db.close()
    other.close()
  }
})

it('resumes three exact page1 events across two native reopens', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'grafeo-cdc-pages-'))
  const path = join(dir, 'store.grafeo')
  let db = GrafeoDB.create(path)
  try {
    db.enableCdc()
    const ids = ['First', 'Second', 'Third'].map(label => String(db.createNode([label]).id))
    const expected = (await db.changesAfter(null, 3, 4096)).events
    assert.deepEqual(expected.map(event => event.entity_id), ids)
    assert.equal(new Set(ids).size, 3)
    const first = await db.changesAfter(null, 1, 4096)
    assert.deepEqual(first.events, [expected[0]])
    let cursor = first.next
    db.close()
    for (const index of [1, 2]) {
      db = GrafeoDB.create(path)
      const page = await db.changesAfter(cursor, 1, 4096)
      assert.deepEqual(page.events, [expected[index]])
      cursor = page.next
      if (index === 2) {
        const eof = await db.changesAfter(cursor, 1, 4096)
        assert.deepEqual(eof.events, [])
        assert.deepEqual(eof.next, cursor)
      }
      db.close()
    }
    assert.deepEqual(first.events, [expected[0]])
  } finally {
    db.close()
    rmSync(dir, { recursive: true, force: true })
  }
})

it('resumes indexed entity history with exact IDs and an inclusive epoch selector', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'grafeo-entity-pages-'))
  const path = join(dir, 'store.grafeo')
  let db = GrafeoDB.create(path)
  try {
    db.enableCdc()
    const node = db.createNode(['History'])
    db.setNodeProperty(node.id, 'n', 1)
    db.setNodeProperty(node.id, 'n', 2)
    const id = String(node.id)
    const first = await db.nodeHistoryAfter(id, null, 1, 4096)
    const tail = await db.nodeHistoryAfter(id, first.next, 2, 4096)
    assert.equal(first.events.length, 1)
    assert.equal(tail.events.length, 2)
    const hidden = await db.nodeHistoryAfter(id, null, 1, 1, tail.events[0].epoch)
    assert.deepEqual(hidden.events, [])
    assert.deepEqual(hidden.next, first.next)
    const selected = await db.nodeHistoryAfter(id, hidden.next, 2, 4096, tail.events[0].epoch)
    assert.deepEqual(selected, tail)
    assert.deepEqual((await db.nodeHistoryAfter('9007199254740993', null, 1, 4096)).events, [])
    for (const invalid of ['', '-1', '1.5', '18446744073709551616']) {
      await assert.rejects(db.nodeHistoryAfter(invalid, null, 1, 4096))
    }
    await assert.rejects(db.nodeHistoryAfter(id, Buffer.from('bad'), 1, 4096), code('S004'))
    db.close()
    for (let pass = 0; pass < 2; pass += 1) {
      db = GrafeoDB.create(path)
      assert.deepEqual(await db.nodeHistoryAfter(id, first.next, 2, 4096), tail)
      db.close()
    }
    await assert.rejects(db.nodeHistoryAfter(id, first.next, 1, 4096))
  } finally {
    db.close()
    rmSync(dir, { recursive: true, force: true })
  }
})


it('preserves node labels and complete edge creation payloads', async () => {
  const db = GrafeoDB.create()
  try {
    db.enableCdc()
    const a = db.createNode(['From'])
    const b = db.createNode(['To'])
    const edge = db.createEdge(a.id, b.id, 'LINK')
    const node = await db.nodeHistoryAfter(String(a.id), null, 1, 4096)
    assert.deepEqual(node.events[0].labels, ['From'])
    const page = await db.edgeHistoryAfter(String(edge.id), null, 1, 4096)
    db.close()
    assert.equal(page.events[0].edge_type, 'LINK')
    assert.equal(page.events[0].src_id, String(a.id))
    assert.equal(page.events[0].dst_id, String(b.id))
    assert.equal(page.events[0].labels, null)
  } finally {
    db.close()
  }
})


it('retained cut preserves stale cursor errors in every page reader', {
  skip: process.env.GRAFEO_CDC_EVICTED_FIXTURE ? false : 'generate the native C retained-cut fixture first'
}, async () => {
  const fixture = process.env.GRAFEO_CDC_EVICTED_FIXTURE
  const cursor = readFileSync(fixture.replace(/\.grafeo$/, '.cursor'))
  assert.equal(cursor.length, 97)
  const dir = mkdtempSync(join(tmpdir(), 'grafeo-cdc-retained-node-'))
  const path = join(dir, 'retained.grafeo')
  let db
  try {
    copyFileSync(fixture, path)
    db = GrafeoDB.create(path)
    await assert.rejects(db.changesAfter(cursor, 1, 4096), code('S006'))
    await assert.rejects(db.nodeHistoryAfter('0', cursor, 1, 4096), code('S006'))
    await assert.rejects(db.edgeHistoryAfter('0', cursor, 1, 4096), code('S006'))
    assert.equal((await db.changesAfter(null, 1, 4096)).events.length, 1)
  } finally {
    db?.close()
    rmSync(dir, { recursive: true, force: true })
  }
})

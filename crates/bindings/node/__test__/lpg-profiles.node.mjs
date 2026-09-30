import { createRequire } from 'node:module'
import { it } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

// Raw libraries remain available for explicit feature-profile controls.
const addon = { exports: {} }
if (process.env.GRAFEO_NODE_LIBRARY) {
  process.dlopen(addon, process.env.GRAFEO_NODE_LIBRARY)
} else {
  addon.exports = createRequire(import.meta.url)('../index.js')
}
const { GrafeoDB } = addon.exports

it('exports exact node/edge CRUD in every LPG-capable profile', () => {
  const db = GrafeoDB.create()
  try {
    assert.equal(typeof db.createNode, 'function')
    const props = { name: '雪', value: 42, nested: { ok: true, items: [1, 'two'] } }
    const first = db.createNode(['First'], props)
    const second = db.createNode(['Second'])
    assert.deepEqual(db.getNode(first.id).properties(), props)
    const edge = db.createEdge(first.id, second.id, 'LINK', { value: 7 })
    assert.equal(db.getEdge(edge.id).get('value'), 7)
    db.setNodeProperty(first.id, 'value', 43)
    db.setEdgeProperty(edge.id, 'value', 8)
    assert.equal(db.getNode(first.id).get('value'), 43)
    assert.equal(db.getEdge(edge.id).get('value'), 8)
    assert.equal(db.addNodeLabel(first.id, 'Added'), true)
    assert.deepEqual(db.getNodeLabels(first.id).sort(), ['Added', 'First'])
    assert.equal(db.removeNodeLabel(first.id, 'Added'), true)
    assert.equal(db.removeEdgeProperty(edge.id, 'value'), true)
    assert.equal(db.removeNodeProperty(first.id, 'nested'), true)
    assert.equal(db.nodeCount(), 2)
    assert.equal(db.edgeCount(), 1)
    assert.equal(db.deleteEdge(edge.id), true)
    assert.equal(db.deleteNode(second.id), true)
    assert.equal(db.edgeCount(), 0)
    assert.equal(db.nodeCount(), 1)
  } finally { db.close() }
})

it('exports transaction createNode with commit and rollback visibility', () => {
  const db = GrafeoDB.create()
  try {
    const tx = db.beginTransaction()
    try {
      assert.equal(typeof tx.createNode, 'function')
      const id = tx.createNode(['Committed'])
      assert.equal(db.getNode(id), null)
      tx.commit()
      assert.deepEqual(db.getNode(id).labels, ['Committed'])
    } finally { if (tx.isActive) tx.rollback() }
    const rollback = db.beginTransaction()
    try {
      const id = rollback.createNode(['RolledBack'])
      rollback.rollback()
      assert.equal(db.getNode(id), null)
      assert.equal(db.nodeCount(), 1)
    } finally { if (rollback.isActive) rollback.rollback() }
  } finally { db.close() }
})

it('exposes persistence only with its backend and retains exact saved data', () => {
  const db = GrafeoDB.create()
  const dir = mkdtempSync(join(tmpdir(), 'grafeo-node-profile-'))
  try {
    if (process.env.GRAFEO_EXPECT_STORAGE === '0') {
      assert.equal(typeof db.save, 'undefined')
      assert.equal(typeof db.backupFull, 'undefined')
      assert.throws(() => GrafeoDB.create(join(dir, 'unsupported.grafeo')), /persist/i)
      return
    }
    assert.equal(typeof db.save, 'function')
    assert.equal(typeof db.backupFull, 'function')
    const node = db.createNode(['Saved'], { value: 42 })
    const path = join(dir, 'saved.grafeo')
    db.save(path)
    const reopened = GrafeoDB.open(path)
    try { assert.equal(reopened.getNode(node.id).get('value'), 42) }
    finally { reopened.close() }
  } finally {
    db.close()
    rmSync(dir, { recursive: true, force: true })
  }
})

it('keeps native parser-free while query profiles retain GQL', async () => {
  const db = GrafeoDB.create()
  try {
    if (process.env.GRAFEO_EXPECT_GQL === '0') {
      await assert.rejects(db.execute('RETURN 1 AS value'))
      assert.equal(typeof db.executeSparql, 'undefined')
    } else {
      assert.equal((await db.execute('RETURN 1 AS value')).scalar(), 1)
    }
  } finally { db.close() }
})

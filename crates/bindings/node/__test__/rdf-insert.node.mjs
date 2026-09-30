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
const quad = name => [`http://example.org/${name}`, 'http://example.org/p', '"value"']

it('saves and reopens RDF through the shared storage capability', () => {
  const db = GrafeoDB.create(undefined, 'rdf')
  const dir = mkdtempSync(join(tmpdir(), 'grafeo-node-rdf-save-'))
  try {
    const first = quad('saved')
    const named = [...quad('named'), 'http://example.org/graph']
    db.insertRdfQuads([first, named])
    const path = join(dir, 'rdf.grafeo')
    db.save(path)
    const reopened = GrafeoDB.open(path)
    try {
      assert.equal(reopened.graphModel(), 'rdf')
      assert.equal(reopened.containsRdfQuad(...first), true)
      assert.equal(reopened.containsRdfQuad(...named), true)
      assert.equal(reopened.containsRdfQuad(...quad('absent')), false)
    } finally { reopened.close() }
  } finally {
    db.close()
    rmSync(dir, { recursive: true, force: true })
  }
})

it('returns exact bulk receipts and numeric single-insert counts', () => {
  const db = GrafeoDB.create(undefined, 'rdf')
  try {
    const first = quad('first')
    assert.equal(db.insertRdfQuad(...first), 1)
    assert.equal(db.insertRdfQuad(...first), 0)
    const second = quad('second')
    const receipt = db.insertRdfQuads([first, second, second])
    assert.equal(receipt.length, 2)
    assert.equal(receipt[0], 1)
    assert.equal(typeof receipt[1], 'string')
    assert.match(receipt[1], /^\d+$/)
    assert.ok(BigInt(receipt[1]) > 0n)
    assert.equal(db.containsRdfQuad(...second), true)
    const duplicate = db.insertRdfQuads([first, second])
    assert.equal(duplicate[0], 0)
    assert.ok(BigInt(duplicate[1]) >= BigInt(receipt[1]))
    assert.deepEqual(JSON.parse(JSON.stringify(receipt)), receipt)
  } finally { db.close() }
})

it('preserves transaction counts, visibility, commit and rollback', () => {
  const db = GrafeoDB.create(undefined, 'rdf')
  try {
    const first = quad('committed')
    const tx = db.beginTransaction()
    assert.equal(tx.insertRdfQuad(...first), 1)
    assert.equal(tx.insertRdfQuads([first, quad('also-committed')]), 1)
    assert.equal(tx.containsRdfQuad(...first), true)
    assert.equal(db.containsRdfQuad(...first), false)
    tx.commit()
    assert.equal(db.containsRdfQuad(...first), true)
    const rolled = quad('rolled-back')
    const rollback = db.beginTransaction()
    assert.equal(rollback.insertRdfQuads([rolled, rolled]), 1)
    rollback.rollback()
    assert.equal(db.containsRdfQuad(...rolled), false)
  } finally { db.close() }
})

it('validates the complete bulk input before database or transaction mutation', () => {
  const db = GrafeoDB.create(undefined, 'rdf')
  try {
    const first = quad('not-inserted')
    assert.throws(() => db.insertRdfQuads([first, ['incomplete']]), /RDF quad/)
    assert.equal(db.containsRdfQuad(...first), false)
    const tx = db.beginTransaction()
    assert.throws(() => tx.insertRdfQuads([first, ['incomplete']]), /RDF quad/)
    assert.equal(tx.containsRdfQuad(...first), false)
    assert.equal(tx.insertRdfQuad(...first), 1)
    tx.rollback()
    assert.equal(db.containsRdfQuad(...first), false)
  } finally { db.close() }
})

import { createRequire } from 'node:module'
import { describe, it } from 'node:test'
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

const textDisabled = process.env.GRAFEO_EXPECT_TEXT_DISABLED === '1'

describe('Text tokenizer request validation', () => {
  it('rejects malformed and incompatible values without consuming an owner', async () => {
    const db = GrafeoDB.create()
    try {
      for (const minTokenLength of [null, false, '3', 3n, [], [3], new Number(3),
        { valueOf() { return 3 } }, -1, 1.5, NaN, Infinity, -Infinity, Number.MAX_SAFE_INTEGER + 1]) {
        const pending = db.createIndex({ kind: 'text', label: 'Doc', property: 'text', minTokenLength })
        assert(pending instanceof Promise)
        await assert.rejects(pending, /minTokenLength|integer|number/i)
      }
      for (const kind of [undefined, 'property', 'btree', 'vector']) {
        await assert.rejects(db.createIndex({ kind, property: 'text', minTokenLength: 3 }),
          /minTokenLength.*text/i)
      }
      await assert.rejects(db.createIndex({ kind: 'text', label: 'Doc', property: 'text', min_token_length: 3 }),
        /unknown/i)
      const inherited = Object.assign(Object.create({ minTokenLength: 3 }), {
        kind: 'text', label: 'Doc', property: 'text',
      })
      await assert.rejects(db.createIndex(inherited), /own properties/i)
      await assert.rejects(db.createIndex({ kind: 'text', label: 'Doc', property: 'text', minTokenLength: 3, graph: ['\ud800'] }),
        /UTF-16/i)
      assert.equal(await db.createIndex({ property: 'unchanged' }), 0)
    } finally {
      db.close()
    }
  })
})

const supported = textDisabled ? describe.skip : describe
supported('Text tokenizer owner persistence', () => {
  it('preserves token-sensitive search and owner IDs across rebuild and close/open', async () => {
    const directory = mkdtempSync(join(tmpdir(), 'grafeo-tokenizer-'))
    const path = join(directory, 'tokenizer.grafeo')
    let db = GrafeoDB.create(path)
    try {
      await db.execute("INSERT (:Doc {strict: 'q ox cat', standard: 'q ox cat', zero: 'q ox cat', huge: 'q ox cat'})")
      let reads = 0
      const strict = await db.createIndex({
        kind: 'text', label: 'Doc', property: 'strict', graph: [], name: 'strict-\u{1f680}',
        get minTokenLength() { reads += 1; return reads === 1 ? 3 : -1 },
      })
      assert.equal(reads, 1)
      assert.equal(strict, 0)
      await assert.rejects(db.createIndex({ kind: 'text', label: 'Doc', property: 'strict', minTokenLength: 3 }),
        /index|owner/i)
      const standard = await db.createIndex({ kind: 'text', label: 'Doc', property: 'standard' })
      const zero = await db.createIndex({ kind: 'text', label: 'Doc', property: 'zero', minTokenLength: 0 })
      const huge = await db.createIndex({ kind: 'text', label: 'Doc', property: 'huge', minTokenLength: Number.MAX_SAFE_INTEGER })
      assert.deepEqual([standard, zero, huge], [1, 2, 3])
      const checkTokens = async () => {
        assert.deepEqual(await db.textSearch('Doc', 'strict', 'ox', 10), [])
        assert.deepEqual((await db.textSearch('Doc', 'strict', 'cat', 10)).map(([id]) => id), [0])
        assert.deepEqual(await db.textSearch('Doc', 'standard', 'q', 10), [])
        assert.deepEqual((await db.textSearch('Doc', 'standard', 'ox', 10)).map(([id]) => id), [0])
        assert.deepEqual((await db.textSearch('Doc', 'zero', 'q', 10)).map(([id]) => id), [0])
        assert.deepEqual(await db.textSearch('Doc', 'huge', 'cat', 10), [])
      }
      await checkTokens()
      for (const owner of [strict, standard, zero, huge]) await db.rebuildIndex(owner)
      await checkTokens()
      db.close()
      db = GrafeoDB.open(path)
      await checkTokens()
      for (const owner of [strict, standard, zero, huge]) await db.rebuildIndex(owner)
      await checkTokens()
      assert.equal(await db.dropIndex(strict), true)
      assert.equal(await db.dropIndex(strict), false)
      await assert.rejects(db.rebuildIndex(strict), /index|owner/i)
      assert((await db.createIndex({ kind: 'text', label: 'Doc', property: 'strict', minTokenLength: 3 })) > huge)
    } finally {
      db.close()
      rmSync(directory, { recursive: true, force: true })
    }
  })
})

const disabled = textDisabled ? describe : describe.skip
disabled('Text tokenizer feature-disabled errors', () => {
  it('rejects valid Text requests without consuming an owner', async () => {
    const db = GrafeoDB.create()
    try {
      for (const minTokenLength of [undefined, 0, 3]) {
        await assert.rejects(db.createIndex({ kind: 'text', label: 'Doc', property: 'text', minTokenLength }),
          /text.*(feature|support)|feature.*text/i)
      }
      assert.equal(await db.createIndex({ property: 'unchanged' }), 0)
    } finally {
      db.close()
    }
  })
})

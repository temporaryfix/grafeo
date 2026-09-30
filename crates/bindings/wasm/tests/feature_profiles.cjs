// Run against the exact generated Node package for each LPG-capable profile.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const packageDir = path.resolve(process.argv[2]);
const { Database, QueryControl } = require(path.join(packageDir, 'grafeo_wasm.js'));
const expectQuery = process.argv[3] === '1';
const expectCompact = process.argv[4] === '1';
const declarations = fs.readFileSync(path.join(packageDir, 'grafeo_wasm.d.ts'), 'utf8');
const db = new Database();
for (const method of ['createNode', 'nodeCount', 'edgeCount', 'beginTransaction',
    'commitTransaction', 'rollbackTransaction', 'isTransactionActive', 'importLpg',
    'exportSnapshot', 'exportSnapshotSigned', 'schema', 'info', 'memoryUsage']) {
    assert.equal(typeof db[method], 'function', method);
    assert(declarations.includes(method + '('), method + ' declaration');
}
console.log('PASS WASM profile: generated public methods and declarations');

assert.equal(db.nodeCount(), 0);
assert.equal(db.edgeCount(), 0);
assert(Number.isSafeInteger(db.createNode(['Seed'])));
const imported = db.importLpg({
    nodes: [{ labels: ['Item'], properties: { name: '雪', nested: [true, null, 7] } },
            { labels: ['Item'], properties: { name: 'λ' } }],
    edges: [{ source: 0, target: 1, type: 'LINK' },
            { source: 0, target: 1, type: 'LINK' }],
});
assert.equal(imported.nodes, 2);
assert.equal(imported.edges, 2);
assert.equal(db.nodeCount(), 3);
assert.equal(db.edgeCount(), 2);
console.log('PASS WASM profile: direct creation, structured import and parallel edges');

db.beginTransaction();
assert.equal(db.isTransactionActive(), true);
assert(Number.isSafeInteger(db.createNode(['Committed'])));
assert.equal(db.nodeCount(), 3);
assert(db.commitTransaction() > 0);
assert.equal(db.isTransactionActive(), false);
assert.equal(db.nodeCount(), 4);
db.beginTransaction();
db.createNode(['RolledBack']);
db.rollbackTransaction();
assert.equal(db.isTransactionActive(), false);
assert.equal(db.nodeCount(), 4);
console.log('PASS WASM profile: direct transaction commit and rollback visibility');

const key = new Uint8Array(32).fill(7);
const snapshot = db.exportSnapshot();
const signed = db.exportSnapshotSigned(key);
const restored = Database.importSnapshot(snapshot);
const verified = Database.importSnapshotSigned(signed, key);
for (const copy of [restored, verified]) {
    assert.equal(copy.nodeCount(), 4);
    assert.equal(copy.edgeCount(), 2);
    copy.close(); copy.free();
}
assert.throws(() => Database.importSnapshotSigned(signed, new Uint8Array(32).fill(9)));
const tampered = signed.slice();
tampered[5] ^= 1;
assert.throws(() => Database.importSnapshotSigned(tampered, key));
console.log('PASS WASM profile: unsigned and authenticated snapshots retain exact counts');

assert.equal(db.info().node_count, 4);
assert.equal(typeof db.schema(), 'object');
assert.equal(typeof db.memoryUsage(), 'object');
if (expectCompact) {
    db.compact();
    assert.equal(db.nodeCount(), 4);
    assert.equal(db.edgeCount(), 2);
    db.createNode(['Overlay']);
    db.compact();
    assert.equal(db.nodeCount(), 5);
    assert.equal(db.edgeCount(), 2);
}
console.log('PASS WASM profile: shared administration and optional compact overlays');

if (expectQuery) {
    assert.equal(db.execute('MATCH ()-[e:LINK]->() RETURN count(e) AS n')[0].n, 2);
} else {
    const before = db.nodeCount();
    assert.throws(() => db.execute('RETURN 1'), { code: 'GRAFEO-Q004' });
    const control = new QueryControl();
    assert.throws(() => db.executeWithOptions('RETURN $value', control, undefined,
        { value: 1 }), { code: 'GRAFEO-Q004' });
    control.free();
    // This convenience method explicitly selects GQL, unlike the default path.
    assert.throws(() => db.executeWithParams('RETURN $value', { value: 1 }),
        { code: 'GRAFEO-Q002' });
    assert.equal(db.nodeCount(), before);
    db.createNode(['AfterUnsupportedQuery']);
    assert.equal(db.nodeCount(), before + 1);
}
db.close();
assert.throws(() => db.nodeCount());
db.free();
console.log('PASS WASM profile: query availability and closed-owner rejection');

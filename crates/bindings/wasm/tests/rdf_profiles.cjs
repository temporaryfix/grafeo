// Run against exact generated RDF, bare rdf-model and full Node packages.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const packageDir = path.resolve(process.argv[2]);
const { Database, QueryControl } = require(path.join(packageDir, 'grafeo_wasm.js'));
const expectQuery = process.argv[3] === '1';
const expectStream = process.argv[4] === '1';
const declarations = fs.readFileSync(path.join(packageDir, 'grafeo_wasm.d.ts'), 'utf8');
const db = Database.withGraphModel('rdf');
assert.equal(db.graphModel(), 'rdf');
assert.equal(db.info().is_persistent, false);
for (const method of ['info', 'importRdf', 'insertRdfQuad', 'insertRdfQuads', 'containsRdfQuad']) {
    assert.equal(typeof db[method], 'function');
    assert(declarations.includes(method + '('), method + ' declaration');
}
console.log('PASS WASM RDF: shared metadata and generated declarations');

const integer = 'http://www.w3.org/2001/XMLSchema#integer';
const imported = { triples: [
    { subject: 'urn:number', predicate: 'urn:p', object: { value: '007', datatype: integer } },
    { subject: 'urn:language', predicate: 'urn:p', object: { value: '雪', language: 'ja' } },
] };
assert.equal(db.importRdf(imported).triples, 2);
assert.equal(db.importRdf(imported).triples, 0);
assert(db.containsRdfQuad('<urn:number>', '<urn:p>', `"007"^^<${integer}>`));
assert(!db.containsRdfQuad('<urn:number>', '<urn:p>', `"7"^^<${integer}>`));
assert(db.containsRdfQuad('<urn:language>', '<urn:p>', '"雪"@ja'));
console.log('PASS WASM RDF: structured import, exact lexical/language terms and deduplication');

const quad = ['<urn:q>', '<urn:p>', '"雪"', 'urn:g'];
assert.equal(db.insertRdfQuads([quad, quad]), 1);
assert(db.containsRdfQuad(...quad));
assert(!db.containsRdfQuad(...quad.slice(0, 3)));
assert.equal(db.insertRdfQuad(...quad), 0);
console.log('PASS WASM RDF: exact named graph membership and quad duplicate receipt');

db.beginTransaction();
assert.equal(db.isTransactionActive(), true);
assert.equal(db.insertRdfQuad('<urn:rolled>', '<urn:p>', '"rolled"'), 1);
assert(db.containsRdfQuad('<urn:rolled>', '<urn:p>', '"rolled"'));
db.rollbackTransaction();
assert(!db.containsRdfQuad('<urn:rolled>', '<urn:p>', '"rolled"'));
db.beginTransaction();
assert.equal(db.insertRdfQuads([['<urn:kept>', '<urn:p>', '"kept"']]), 1);
assert(db.commitTransaction() > 0);
assert.equal(db.isTransactionActive(), false);
assert(db.containsRdfQuad('<urn:kept>', '<urn:p>', '"kept"'));
console.log('PASS WASM RDF: transaction read-your-writes, rollback and commit');

assert.throws(() => db.insertRdfQuads([
    ['<urn:must-not-appear>', '<urn:p>', '"first"'],
    ['"invalid subject"', '<urn:p>', '"second"'],
]));
assert(!db.containsRdfQuad('<urn:must-not-appear>', '<urn:p>', '"first"'));
assert(db.containsRdfQuad(...quad));
console.log('PASS WASM RDF: malformed bulk input rejects before partial publication');

const control = new QueryControl();
if (!expectStream) {
    assert.throws(() => db.executeStreamWithOptions('RETURN 1', control), { code: 'GRAFEO-Q004' });
    assert.equal(control.consumed, false);
}
const query = 'SELECT ?s WHERE { GRAPH <urn:g> { ?s <urn:p> "雪" } }';
if (expectQuery) {
    const denied = new QueryControl();
    assert.throws(() => db.executeWithOptions(query, denied,
        { language: 'sparql', maxRows: 0, maxBytes: 65536 }), { code: 'GRAFEO-S001' });
    denied.free();
    const rows = db.executeWithOptions(query, control,
        { language: 'sparql', maxRows: 1, maxBytes: 65536 });
    assert.equal(rows.length, 1);
    assert(String(rows[0].s).includes('urn:q'));
    assert.equal(Object.getPrototypeOf(rows[0]), null);
    assert.equal(control.consumed, true);
} else {
    assert.throws(() => db.executeWithOptions(query, control,
        { language: 'sparql', maxRows: 1, maxBytes: 65536 }), { code: 'GRAFEO-Q002' });
}
control.free();
assert(db.containsRdfQuad('<urn:kept>', '<urn:p>', '"kept"'));
db.close();
assert.throws(() => db.info());
db.free();
console.log('PASS WASM RDF: parser/copy limits, reusable owner and closed-handle rejection');

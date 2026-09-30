// Compile against the packed and installed package, not the source declarations.
import { GrafeoDB, QueryControl, type CreateIndexRequest, type ExecutionOptions,
  type JsonValue } from '@grafeo-db/js'

async function consumer() {
  const db = GrafeoDB.create(undefined, 'both')
  const model: 'lpg' | 'rdf' | 'both' = db.graphModel()
  const options: ExecutionOptions = { control: new QueryControl(), maxRows: 10, maxBytes: 4096 }
  const request: CreateIndexRequest = { kind: 'text', label: 'Doc', property: 'text', minTokenLength: 3 }
  const owner: number = await db.createIndex(request)
  await db.execute('RETURN 1', undefined, options)
  await db.executeCypher('RETURN 1', null, options)
  await db.executeSql('SELECT 1', null, options)
  await db.executeGremlin('g.V()', null, options)
  await db.executeGraphql('{ Doc { text } }', null, options)
  await db.executeSparql('SELECT * WHERE { ?s ?p ?o }', null, options)
  await db.executeLanguage('gql', 'RETURN 1', null, options)
  const stream = await db.executeStream('RETURN 1', null, options)
  const next: Promise<JsonValue | null> = stream.next()
  for await (const row of stream) { const value: JsonValue = row; void value; break }
  const page = await db.changesAfter(null, 10, 4096)
  const epoch: string = page.events[0].epoch
  // @ts-expect-error exact epochs are decimal strings
  const impreciseEpoch: number = page.events[0].epoch
  void epoch; void impreciseEpoch
  const close: Promise<void> = stream.close()
  db.insertRdfQuads([['urn:s', 'urn:p', '"v"'], ['urn:s', 'urn:p', '"v"', 'urn:g']])
  const tx = db.beginTransaction()
  await tx.execute('RETURN 1', null, options)
  await tx.executeCypher('RETURN 1', null, options)
  await tx.executeSql('SELECT 1', null, options)
  await tx.executeGremlin('g.V()', null, options)
  await tx.executeGraphql('{ Doc { text } }', null, options)
  await tx.executeSparql('SELECT * WHERE { ?s ?p ?o }', null, options)
  tx.insertRdfQuads([['urn:s', 'urn:p', '"v"']])
  // @ts-expect-error unknown graph model
  GrafeoDB.create(undefined, 'unknown')
  // @ts-expect-error unknown index kind
  db.createIndex({ property: 'x', kind: 'unknown' })
  // @ts-expect-error row budget must be numeric
  db.execute('RETURN 1', null, { maxRows: '10' })
  // @ts-expect-error transaction byte budget must be numeric
  tx.execute('RETURN 1', null, { maxBytes: '10' })
  // @ts-expect-error RDF tuples require at least three terms
  db.insertRdfQuads([['urn:s', 'urn:p']])
  tx.rollback()
  db.close()
  return { model, owner, next, close }
}
void consumer

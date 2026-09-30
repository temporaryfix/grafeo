A high-performance, embeddable graph database with a Rust core and no required C dependencies, supporting both **Labeled Property Graph (LPG)** and **RDF** data models. Optional allocators (jemalloc/mimalloc) and TLS use C libraries for performance.

## Features

- **Dual data model support**: LPG and RDF with optimized storage for each
- **Multi-language queries**: GQL, Cypher, Gremlin, GraphQL, SPARQL and SQL/PGQ
- Embeddable with zero external dependencies
- Multi-language bindings: Python (PyO3), Node.js (napi-rs), Go (CGO), C (FFI), C# (.NET P/Invoke), Dart (dart:ffi), WebAssembly (wasm-bindgen)
- In-memory and persistent storage modes
- MVCC transactions with snapshot isolation

## Query Language & Data Model Support

| Query Language | LPG | RDF | Status |
|----------------|-----|-----|--------|
| GQL (ISO/IEC 39075) | ✅ | - | Default (`embedded` profile) |
| Cypher (openCypher 9.0) | ✅ | - | `cypher` feature or `languages` group |
| Gremlin (Apache TinkerPop) | ✅ | - | `gremlin` feature or `languages` group |
| GraphQL | ✅ | ✅ | `graphql` feature or `languages` group |
| SPARQL (W3C 1.1) | - | ✅ | `sparql` feature or `languages` group |
| SQL/PGQ (SQL:2023) | ✅ | - | `sql-pgq` feature or `languages` group |

Grafeo uses a modular translator architecture where query languages are parsed into ASTs, then translated to a unified logical plan that executes against the appropriate storage backend (LPG or RDF).

### Data Models

- **LPG (Labeled Property Graph)**: Nodes with labels and properties, edges with types and properties. Ideal for social networks, knowledge graphs and application data.
- **RDF (Resource Description Framework)**: Triple-based storage (subject-predicate-object) with SPO/POS/OSP indexes. Ideal for semantic web, linked data and ontology-based applications.

## Installation

```bash
cargo add grafeo
```

By default, the `embedded` profile is enabled: GQL, AI features (vector/text/hybrid search, CDC), graph algorithms and parallel execution. Customize with feature groups:

```bash
cargo add grafeo                                             # Default (embedded profile)
cargo add grafeo --features lpg,rdf                          # Dual model: LPG + SPARQL/triples
cargo add grafeo --no-default-features --features temporal-host   # temporal host slice (same engine)
cargo add grafeo --no-default-features --features native     # Parser-free LPG + RDF quads
cargo add grafeo --no-default-features --features languages  # All languages, no AI
cargo add grafeo --no-default-features --features gql,ai     # GQL + AI features
cargo add grafeo --no-default-features --features gql        # Minimal: GQL only
cargo add grafeo --features embed                            # Add ONNX embeddings (opt-in)
```

## Quick Start

```rust
use grafeo::GrafeoDB;

fn main() -> Result<(), grafeo::Error> {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    // Create nodes
    session.execute("INSERT (:Person {name: 'Alix', age: 30})")?;
    session.execute("INSERT (:Person {name: 'Gus', age: 25})")?;

    // Query
    let result = session.execute("MATCH (p:Person) RETURN p.name, p.age")?;
    for row in result.rows() {
        println!("{:?}", row);
    }

    Ok(())
}
```

## License

Apache-2.0

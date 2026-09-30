//! Linear `MATCH (a)-[]->(b)-[]->(c)-[]->(a)` must close on `a`, not rebind it.

#![cfg(all(feature = "gql", feature = "lpg"))]

use grafeo_engine::GrafeoDB;

#[test]
fn linear_triangle_count_matches_comma_join() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["Person"]);
    let b = db.create_node(&["Person"]);
    let c = db.create_node(&["Person"]);
    db.create_edge(a, b, "KNOWS");
    db.create_edge(b, c, "KNOWS");
    db.create_edge(c, a, "KNOWS");

    let session = db.session();
    let linear = session
        .execute(
            "MATCH (x:Person)-[:KNOWS]->(y:Person)-[:KNOWS]->(z:Person)-[:KNOWS]->(x) RETURN count(*) AS n",
        )
        .unwrap();
    let comma = session
        .execute(
            "MATCH (x:Person)-[:KNOWS]->(y)-[:KNOWS]->(z), (z)-[:KNOWS]->(x) RETURN count(*) AS n",
        )
        .unwrap();
    let ln = linear.rows()[0][0].as_int64().unwrap();
    let cn = comma.rows()[0][0].as_int64().unwrap();
    assert_eq!(ln, cn, "linear cycle must match comma-join close");
    assert_eq!(ln, 3, "one directed 3-cycle, three start vertices");
}

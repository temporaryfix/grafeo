//! Connected catalog authority and prepared logical publication regressions.

use grafeo_engine::catalog::{Catalog, NamedConstraintDefinition, NamedConstraintKind};

#[test]
fn catalog_foundation_failed_constraint_admission_has_no_dictionary_prefix()
-> Result<(), Box<dyn std::error::Error>> {
    let catalog = Catalog::new();
    catalog.create_named_constraint(NamedConstraintDefinition {
        name: "occupied_name".to_string(),
        label: "Original".to_string(),
        properties: vec!["original_property".to_string()],
        kind: NamedConstraintKind::Unique,
    })?;
    let labels_before = catalog.all_labels();
    let properties_before = catalog.all_property_keys();
    let owner_before = catalog.get_named_constraint("occupied_name");

    let result = catalog.create_named_constraint(NamedConstraintDefinition {
        name: "occupied_name".to_string(),
        label: "Unpublished".to_string(),
        properties: vec![
            "unpublished_first".to_string(),
            "unpublished_second".to_string(),
        ],
        kind: NamedConstraintKind::Unique,
    });

    assert!(result.is_err(), "duplicate owner must reject normally");
    assert_eq!(catalog.all_labels(), labels_before);
    assert_eq!(catalog.all_property_keys(), properties_before);
    assert_eq!(catalog.get_named_constraint("occupied_name"), owner_before);
    Ok(())
}

#[cfg(all(feature = "lpg", feature = "gql", feature = "wal"))]
#[test]
fn catalog_foundation_exact_save_reopens_schema_and_named_index()
-> Result<(), Box<dyn std::error::Error>> {
    use grafeo_engine::GrafeoDB;
    let root = tempfile::tempdir()?;
    let destination = root.path().join("saved");
    let source = GrafeoDB::new_in_memory();
    source.session().execute("CREATE SCHEMA tenant")?;
    source
        .session()
        .execute("INSERT (:Person {email:'saved@example.test'})")?;
    source
        .session()
        .execute("CREATE INDEX email_owner FOR (n:Person) ON (n.email)")?;
    source.save(&destination)?;
    assert!(destination.is_file());
    let reopened = GrafeoDB::open(&destination)?;
    assert_eq!(reopened.export_snapshot()?, source.export_snapshot()?);
    assert_eq!(reopened.world_cut()?, source.world_cut()?);
    let session = reopened.session();
    let schemas = session.execute("SHOW SCHEMAS")?;
    assert!(
        schemas
            .rows()
            .iter()
            .any(|row| row.first().and_then(|value| value.as_str()) == Some("tenant"))
    );
    let indexes = session.execute("SHOW INDEXES")?;
    assert_eq!(indexes.rows().len(), 1);
    let index = indexes.rows().first().ok_or("missing saved index")?;
    assert_eq!(
        index.first().and_then(|value| value.as_str()),
        Some("email_owner")
    );
    let rows = session.execute("MATCH (n:Person) RETURN n.email")?;
    assert_eq!(rows.rows().len(), 1);
    assert_eq!(
        rows.rows()
            .first()
            .and_then(|row| row.first())
            .and_then(|value| value.as_str()),
        Some("saved@example.test")
    );
    Ok(())
}

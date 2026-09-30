//! Graph elements exposed to JavaScript - nodes and edges with their properties.

use std::collections::HashMap;

use napi::bindgen_prelude::*;
use napi::sys;
use napi_derive::napi;

use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};

use crate::error::native_to_js_error;
use crate::types;

/// A node in the graph with labels and properties.
#[napi]
#[derive(Clone)]
pub struct JsNode {
    pub(crate) id: NodeId,
    pub(crate) labels: Vec<String>,
    pub(crate) properties: HashMap<PropertyKey, Value>,
    conversion_limit: usize,
}

#[napi]
impl JsNode {
    /// Get the node ID.
    #[napi(getter)]
    pub fn id(&self) -> f64 {
        self.id.0 as f64
    }

    /// Get the node labels.
    #[napi(getter)]
    pub fn labels(&self, env: Env) -> Result<Vec<String>> {
        types::bounded_columns(&self.labels, self.conversion_limit)
            .map_err(|error| native_to_js_error(&env, error))
    }

    /// Get a property value by key.
    #[napi]
    pub fn get(&self, env: Env, key: String) -> Result<Unknown<'_>> {
        match self.properties.get(key.as_str()) {
            Some(v) => property_to_js(&env, v, self.conversion_limit),
            // SAFETY: env is valid and ToNapiValue produces a valid undefined value
            None => Ok(unsafe {
                Unknown::from_raw_unchecked(
                    env.raw(),
                    <() as ToNapiValue>::to_napi_value(env.raw(), ())?,
                )
            }),
        }
    }

    /// Get all properties as a plain object.
    #[napi]
    pub fn properties(&self, env: Env) -> Result<Object<'_>> {
        properties_to_object(&env, &self.properties, self.conversion_limit)
    }

    /// Check if the node has a specific label.
    #[napi(js_name = "hasLabel")]
    pub fn has_label(&self, label: String) -> bool {
        self.labels.iter().any(|l| l == &label)
    }

    /// String representation.
    #[napi(js_name = "toString")]
    pub fn to_string_js(&self, env: Env) -> Result<String> {
        let mut budget = types::CopyBudget::new(self.conversion_limit);
        budget
            .columns(&self.labels)
            .map_err(|error| native_to_js_error(&env, error))?;
        budget
            .charge(256)
            .map_err(|error| native_to_js_error(&env, error))?;
        for label in &self.labels {
            budget
                .repeated(label.len(), 16)
                .map_err(|error| native_to_js_error(&env, error))?;
        }
        Ok(format!(
            "(:{} {{id: {}}})",
            self.labels.join(":"),
            self.id.0
        ))
    }
}

impl JsNode {
    pub(crate) fn set_conversion_limit(&mut self, max_bytes: usize) {
        self.conversion_limit = max_bytes;
    }

    pub fn new(id: NodeId, labels: Vec<String>, properties: HashMap<PropertyKey, Value>) -> Self {
        Self {
            id,
            labels,
            properties,
            conversion_limit: types::default_conversion_limit(),
        }
    }
}

/// An edge (relationship) between two nodes with a type and properties.
#[napi]
#[derive(Clone)]
pub struct JsEdge {
    pub(crate) id: EdgeId,
    pub(crate) edge_type: String,
    pub(crate) source_id: NodeId,
    pub(crate) target_id: NodeId,
    pub(crate) properties: HashMap<PropertyKey, Value>,
    conversion_limit: usize,
}

#[napi]
impl JsEdge {
    /// Get the edge ID.
    #[napi(getter)]
    pub fn id(&self) -> f64 {
        self.id.0 as f64
    }

    /// Get the edge type (relationship type).
    #[napi(getter, js_name = "edgeType")]
    pub fn edge_type(&self, env: Env) -> Result<String> {
        types::CopyBudget::new(self.conversion_limit)
            .string(&self.edge_type)
            .map_err(|error| native_to_js_error(&env, error))?;
        Ok(self.edge_type.clone())
    }

    /// Get the source node ID.
    #[napi(getter, js_name = "sourceId")]
    pub fn source_id(&self) -> f64 {
        self.source_id.0 as f64
    }

    /// Get the target node ID.
    #[napi(getter, js_name = "targetId")]
    pub fn target_id(&self) -> f64 {
        self.target_id.0 as f64
    }

    /// Get a property value by key.
    #[napi]
    pub fn get(&self, env: Env, key: String) -> Result<Unknown<'_>> {
        match self.properties.get(key.as_str()) {
            Some(v) => property_to_js(&env, v, self.conversion_limit),
            // SAFETY: env is valid and ToNapiValue produces a valid undefined value
            None => Ok(unsafe {
                Unknown::from_raw_unchecked(
                    env.raw(),
                    <() as ToNapiValue>::to_napi_value(env.raw(), ())?,
                )
            }),
        }
    }

    /// Get all properties as a plain object.
    #[napi]
    pub fn properties(&self, env: Env) -> Result<Object<'_>> {
        properties_to_object(&env, &self.properties, self.conversion_limit)
    }

    /// String representation.
    #[napi(js_name = "toString")]
    pub fn to_string_js(&self, env: Env) -> Result<String> {
        let mut budget = types::CopyBudget::new(self.conversion_limit);
        budget
            .string(&self.edge_type)
            .map_err(|error| native_to_js_error(&env, error))?;
        budget
            .charge(256)
            .map_err(|error| native_to_js_error(&env, error))?;
        Ok(format!("()-[:{}]->() (id={})", self.edge_type, self.id.0))
    }
}

impl JsEdge {
    pub(crate) fn set_conversion_limit(&mut self, max_bytes: usize) {
        self.conversion_limit = max_bytes;
    }

    pub fn new(
        id: EdgeId,
        edge_type: String,
        source_id: NodeId,
        target_id: NodeId,
        properties: HashMap<PropertyKey, Value>,
    ) -> Self {
        Self {
            id,
            edge_type,
            source_id,
            target_id,
            properties,
            conversion_limit: types::default_conversion_limit(),
        }
    }
}

/// Create a JS object from a Grafeo property map.
pub(crate) fn properties_to_object<'value>(
    env: &Env,
    properties: &'value HashMap<PropertyKey, Value>,
    max_bytes: usize,
) -> Result<Object<'value>> {
    let mut budget = types::CopyBudget::new(max_bytes);
    // Native hash-table capacity can exceed its live entry count. Include it
    // separately from the copied JavaScript properties and nested values.
    let capacity = properties
        .capacity()
        .checked_add(1)
        .ok_or_else(|| native_to_js_error(env, types::copy_limit_error()))?;
    budget
        .repeated(
            capacity,
            2 * (std::mem::size_of::<(PropertyKey, Value)>() + 1),
        )
        .map_err(|error| native_to_js_error(env, error))?;
    budget
        .dict(properties.len())
        .map_err(|error| native_to_js_error(env, error))?;
    for (key, value) in properties {
        budget
            .string(key.as_str())
            .map_err(|error| native_to_js_error(env, error))?;
        budget
            .value(value)
            .map_err(|error| native_to_js_error(env, error))?;
    }
    let env = env.raw();
    let mut raw_obj = std::ptr::null_mut();
    // SAFETY: env is valid; napi_create_object writes to our out-pointer
    types::check_napi(unsafe { sys::napi_create_object(env, &raw mut raw_obj) })?;
    let mut obj = Object::from_raw(env, raw_obj);
    for (k, v) in properties {
        let val_raw = types::value_to_napi_admitted(env, v)?;
        // SAFETY: env and val_raw are valid napi values produced by value_to_napi
        let val_unknown = unsafe { Unknown::from_raw_unchecked(env, val_raw) };
        obj.set_named_property(k.as_str(), val_unknown)?;
    }
    Ok(obj)
}

fn property_to_js<'value>(
    env: &Env,
    value: &'value Value,
    max_bytes: usize,
) -> Result<Unknown<'value>> {
    types::CopyBudget::new(max_bytes)
        .value(value)
        .map_err(|error| native_to_js_error(env, error))?;
    let raw = types::value_to_napi_admitted(env.raw(), value)?;
    // SAFETY: the converter returned a live value in this callback environment;
    // JavaScript receives its own value and never borrows the native property.
    Ok(unsafe { Unknown::from_raw_unchecked(env.raw(), raw) })
}

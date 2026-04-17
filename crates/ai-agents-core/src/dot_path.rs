//! Dot-path navigation utility for serde_json::Value.

use std::collections::HashMap;

use serde_json::Value;

use crate::{AgentError, Result};

/// Get a nested value from a Value using dot-notation (e.g. "a.b.c").
/// Returns None if any segment along the path is missing.
pub fn get_dot_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    if path.is_empty() {
        return None;
    }
    let parts: Vec<&str> = path.split('.').collect();

    let mut current = value;
    for part in &parts {
        match current {
            Value::Object(map) => {
                current = map.get(*part)?;
            }
            _ => return None,
        }
    }
    Some(current)
}

/// Get a nested value from a HashMap root using dot-notation.
/// The first path segment is the map key; remaining segments traverse nested objects.
pub fn get_dot_path_from_map(map: &HashMap<String, Value>, path: &str) -> Option<Value> {
    if path.is_empty() {
        return None;
    }
    let parts: Vec<&str> = path.split('.').collect();

    let mut current: Option<&Value> = map.get(parts[0]);

    for part in &parts[1..] {
        match current {
            Some(Value::Object(obj)) => {
                current = obj.get(*part);
            }
            _ => return None,
        }
    }

    current.cloned()
}

/// Set a nested value in a Value using dot-notation.
/// Creates intermediate objects along the path if they do not exist.
pub fn set_dot_path(mut root: Value, path: &str, new_value: Value) -> Result<Value> {
    if path.is_empty() {
        return Err(AgentError::Config("Empty dot-path".into()));
    }
    let parts: Vec<&str> = path.split('.').collect();

    let mut current = &mut root;
    for (i, part) in parts.iter().enumerate() {
        if i == parts.len() - 1 {
            match current {
                Value::Object(map) => {
                    map.insert((*part).to_string(), new_value);
                    return Ok(root);
                }
                _ => {
                    return Err(AgentError::Config(format!(
                        "Cannot set field '{}': parent is not an object",
                        path
                    )));
                }
            }
        }

        if !current.is_object() {
            return Err(AgentError::Config(format!(
                "Cannot traverse '{}': segment '{}' is not an object",
                path, part
            )));
        }

        let map = current.as_object_mut().unwrap();
        if !map.contains_key(*part) {
            map.insert((*part).to_string(), Value::Object(serde_json::Map::new()));
        }
        current = map.get_mut(*part).unwrap();
    }

    Err(AgentError::Config(format!(
        "Failed to set dot-path '{}'",
        path
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // get_dot_path tests

    #[test]
    fn test_get_dot_path_simple() {
        let val = json!({"a": {"b": {"c": 42}}});
        assert_eq!(get_dot_path(&val, "a.b.c"), Some(&json!(42)));
    }

    #[test]
    fn test_get_dot_path_top_level() {
        let val = json!({"name": "Alice"});
        assert_eq!(get_dot_path(&val, "name"), Some(&json!("Alice")));
    }

    #[test]
    fn test_get_dot_path_missing() {
        let val = json!({"a": {"b": 1}});
        assert_eq!(get_dot_path(&val, "a.c"), None);
        assert_eq!(get_dot_path(&val, "x.y.z"), None);
    }

    #[test]
    fn test_get_dot_path_non_object() {
        let val = json!({"a": 42});
        assert_eq!(get_dot_path(&val, "a.b"), None);
    }

    // set_dot_path tests

    #[test]
    fn test_set_dot_path_simple() {
        let val = json!({"a": {"b": 1}});
        let result = set_dot_path(val, "a.b", json!(99)).unwrap();
        assert_eq!(result, json!({"a": {"b": 99}}));
    }

    #[test]
    fn test_set_dot_path_top_level() {
        let val = json!({"name": "old"});
        let result = set_dot_path(val, "name", json!("new")).unwrap();
        assert_eq!(result, json!({"name": "new"}));
    }

    #[test]
    fn test_set_dot_path_creates_intermediate() {
        let val = json!({});
        let result = set_dot_path(val, "a.b.c", json!(true)).unwrap();
        assert_eq!(result, json!({"a": {"b": {"c": true}}}));
    }

    #[test]
    fn test_set_dot_path_preserves_siblings() {
        let val = json!({"a": {"b": 1, "c": 2}});
        let result = set_dot_path(val, "a.b", json!(99)).unwrap();
        assert_eq!(result, json!({"a": {"b": 99, "c": 2}}));
    }

    #[test]
    fn test_set_dot_path_array_value() {
        let val = json!({"traits": {"personality": ["shy"]}});
        let result = set_dot_path(val, "traits.personality", json!(["bold", "brave"])).unwrap();
        assert_eq!(
            result,
            json!({"traits": {"personality": ["bold", "brave"]}})
        );
    }

    #[test]
    fn test_roundtrip_get_set() {
        let val = json!({"identity": {"name": "Alice", "role": "Guard"}});
        let name = get_dot_path(&val, "identity.name").cloned().unwrap();
        assert_eq!(name, json!("Alice"));

        let updated = set_dot_path(val, "identity.name", json!("Bob")).unwrap();
        assert_eq!(get_dot_path(&updated, "identity.name"), Some(&json!("Bob")));
        assert_eq!(
            get_dot_path(&updated, "identity.role"),
            Some(&json!("Guard"))
        );
    }

    // get_dot_path_from_map tests

    #[test]
    fn test_get_from_map_single_segment() {
        let mut map = HashMap::new();
        map.insert("name".to_string(), json!("Alice"));
        assert_eq!(get_dot_path_from_map(&map, "name"), Some(json!("Alice")));
    }

    #[test]
    fn test_get_from_map_multi_segment() {
        let mut map = HashMap::new();
        map.insert("user".to_string(), json!({"profile": {"age": 30}}));
        assert_eq!(
            get_dot_path_from_map(&map, "user.profile.age"),
            Some(json!(30))
        );
    }

    #[test]
    fn test_get_from_map_missing_root() {
        let map: HashMap<String, Value> = HashMap::new();
        assert_eq!(get_dot_path_from_map(&map, "missing"), None);
    }

    #[test]
    fn test_get_from_map_missing_nested() {
        let mut map = HashMap::new();
        map.insert("user".to_string(), json!({"name": "Alice"}));
        assert_eq!(get_dot_path_from_map(&map, "user.email"), None);
    }

    #[test]
    fn test_get_from_map_non_object_intermediate() {
        let mut map = HashMap::new();
        map.insert("count".to_string(), json!(42));
        assert_eq!(get_dot_path_from_map(&map, "count.value"), None);
    }

    #[test]
    fn test_set_dot_path_empty_path_error() {
        let val = json!({});
        assert!(set_dot_path(val, "", json!(1)).is_err());
    }

    #[test]
    fn test_set_dot_path_non_object_parent_error() {
        let val = json!({"a": 42});
        assert!(set_dot_path(val, "a.b", json!(1)).is_err());
    }
}

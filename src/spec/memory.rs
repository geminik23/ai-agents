//! Memory configuration types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for memory backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    #[serde(rename = "type")]
    pub memory_type: String,

    #[serde(default = "default_max_messages")]
    pub max_messages: usize,

    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

fn default_max_messages() -> usize {
    100
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            memory_type: "in-memory".to_string(),
            max_messages: default_max_messages(),
            extra: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_config_default() {
        let config = MemoryConfig::default();
        assert_eq!(config.memory_type, "in-memory");
        assert_eq!(config.max_messages, 100);
    }

    #[test]
    fn test_memory_config_deserialize() {
        let yaml = r#"
type: in-memory
max_messages: 50
"#;
        let config: MemoryConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.memory_type, "in-memory");
        assert_eq!(config.max_messages, 50);
    }

    #[test]
    fn test_memory_config_with_defaults() {
        let yaml = r#"
type: sqlite
"#;
        let config: MemoryConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.memory_type, "sqlite");
        assert_eq!(config.max_messages, 100); // default
    }

    #[test]
    fn test_memory_config_extra_fields() {
        let yaml = r#"
type: sqlite
max_messages: 200
db_path: "/path/to/db.sqlite"
"#;
        let config: MemoryConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.extra.contains_key("db_path"));
    }
}

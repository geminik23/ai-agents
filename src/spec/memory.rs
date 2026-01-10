//! Memory configuration types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::memory::{CompactingMemoryConfig, MemoryTokenBudget};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    #[serde(rename = "type", default = "default_memory_type")]
    pub memory_type: String,

    #[serde(default = "default_max_messages")]
    pub max_messages: usize,

    #[serde(default)]
    pub max_recent_messages: Option<usize>,

    #[serde(default)]
    pub compress_threshold: Option<usize>,

    #[serde(default)]
    pub summarize_batch_size: Option<usize>,

    #[serde(default)]
    pub token_budget: Option<MemoryTokenBudget>,

    #[serde(default)]
    pub summarizer_llm: Option<String>,

    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

fn default_memory_type() -> String {
    "in-memory".to_string()
}

fn default_max_messages() -> usize {
    100
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            memory_type: default_memory_type(),
            max_messages: default_max_messages(),
            max_recent_messages: None,
            compress_threshold: None,
            summarize_batch_size: None,
            token_budget: None,
            summarizer_llm: None,
            extra: HashMap::new(),
        }
    }
}

impl MemoryConfig {
    pub fn is_compacting(&self) -> bool {
        self.memory_type == "compacting"
    }

    pub fn to_compacting_config(&self) -> CompactingMemoryConfig {
        CompactingMemoryConfig {
            max_recent_messages: self.max_recent_messages.unwrap_or(50),
            compress_threshold: self.compress_threshold.unwrap_or(30),
            summarize_batch_size: self.summarize_batch_size.unwrap_or(10),
            max_summary_length: 2000,
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
        assert!(!config.is_compacting());
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
        assert_eq!(config.max_messages, 100);
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

    #[test]
    fn test_compacting_memory_config() {
        let yaml = r#"
type: compacting
max_messages: 100
max_recent_messages: 20
compress_threshold: 30
summarize_batch_size: 10
summarizer_llm: router
"#;
        let config: MemoryConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.is_compacting());
        assert_eq!(config.max_recent_messages, Some(20));
        assert_eq!(config.compress_threshold, Some(30));
        assert_eq!(config.summarize_batch_size, Some(10));
        assert_eq!(config.summarizer_llm, Some("router".to_string()));

        let compacting_config = config.to_compacting_config();
        assert_eq!(compacting_config.max_recent_messages, 20);
        assert_eq!(compacting_config.compress_threshold, 30);
    }

    #[test]
    fn test_memory_config_with_token_budget() {
        let yaml = r#"
type: compacting
max_messages: 100
token_budget:
  total: 8192
  allocation:
    summary: 2048
    recent_messages: 4096
    facts: 1024
  overflow_strategy: summarize_more
  warn_at_percent: 75
"#;
        let config: MemoryConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.token_budget.is_some());
        let budget = config.token_budget.unwrap();
        assert_eq!(budget.total, 8192);
        assert_eq!(budget.allocation.summary, 2048);
        assert_eq!(budget.warn_at_percent, 75);
    }

    #[test]
    fn test_to_compacting_config_defaults() {
        let config = MemoryConfig {
            memory_type: "compacting".to_string(),
            ..Default::default()
        };
        let compacting = config.to_compacting_config();
        assert_eq!(compacting.max_recent_messages, 50);
        assert_eq!(compacting.compress_threshold, 30);
        assert_eq!(compacting.summarize_batch_size, 10);
    }
}

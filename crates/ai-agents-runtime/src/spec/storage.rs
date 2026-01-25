//! Storage configuration types

use serde::{Deserialize, Serialize};

/// Storage configuration using tagged enum for type safety and extensibility
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StorageConfig {
    #[serde(rename = "none")]
    None,

    #[serde(rename = "file")]
    File(FileStorageConfig),

    #[serde(rename = "sqlite")]
    Sqlite(SqliteStorageConfig),

    #[serde(rename = "redis")]
    Redis(RedisStorageConfig),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileStorageConfig {
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqliteStorageConfig {
    pub path: String,

    #[serde(default)]
    pub table: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisStorageConfig {
    pub url: String,

    #[serde(default)]
    pub prefix: Option<String>,

    #[serde(default)]
    pub ttl_seconds: Option<u64>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        StorageConfig::None
    }
}

impl StorageConfig {
    pub fn none() -> Self {
        StorageConfig::None
    }

    pub fn file(path: impl Into<String>) -> Self {
        StorageConfig::File(FileStorageConfig { path: path.into() })
    }

    pub fn sqlite(path: impl Into<String>) -> Self {
        StorageConfig::Sqlite(SqliteStorageConfig {
            path: path.into(),
            table: None,
        })
    }

    pub fn redis(url: impl Into<String>) -> Self {
        StorageConfig::Redis(RedisStorageConfig {
            url: url.into(),
            prefix: None,
            ttl_seconds: None,
        })
    }

    pub fn is_none(&self) -> bool {
        matches!(self, StorageConfig::None)
    }

    pub fn is_file(&self) -> bool {
        matches!(self, StorageConfig::File(_))
    }

    pub fn is_sqlite(&self) -> bool {
        matches!(self, StorageConfig::Sqlite(_))
    }

    pub fn is_redis(&self) -> bool {
        matches!(self, StorageConfig::Redis(_))
    }

    pub fn storage_type(&self) -> &'static str {
        match self {
            StorageConfig::None => "none",
            StorageConfig::File(_) => "file",
            StorageConfig::Sqlite(_) => "sqlite",
            StorageConfig::Redis(_) => "redis",
        }
    }

    pub fn get_path(&self) -> Option<&str> {
        match self {
            StorageConfig::File(c) => Some(&c.path),
            StorageConfig::Sqlite(c) => Some(&c.path),
            _ => None,
        }
    }

    pub fn get_url(&self) -> Option<&str> {
        match self {
            StorageConfig::Redis(c) => Some(&c.url),
            _ => None,
        }
    }

    pub fn get_prefix(&self) -> &str {
        match self {
            StorageConfig::Redis(c) => c.prefix.as_deref().unwrap_or("agent:"),
            _ => "agent:",
        }
    }

    pub fn get_ttl(&self) -> Option<u64> {
        match self {
            StorageConfig::Redis(c) => c.ttl_seconds,
            _ => None,
        }
    }

    pub fn get_table(&self) -> Option<&str> {
        match self {
            StorageConfig::Sqlite(c) => c.table.as_deref(),
            _ => None,
        }
    }

    pub fn as_file(&self) -> Option<&FileStorageConfig> {
        match self {
            StorageConfig::File(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_sqlite(&self) -> Option<&SqliteStorageConfig> {
        match self {
            StorageConfig::Sqlite(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_redis(&self) -> Option<&RedisStorageConfig> {
        match self {
            StorageConfig::Redis(c) => Some(c),
            _ => None,
        }
    }
}

impl RedisStorageConfig {
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    pub fn with_ttl(mut self, ttl_seconds: u64) -> Self {
        self.ttl_seconds = Some(ttl_seconds);
        self
    }
}

impl SqliteStorageConfig {
    pub fn with_table(mut self, table: impl Into<String>) -> Self {
        self.table = Some(table.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_config_default() {
        let config = StorageConfig::default();
        assert!(config.is_none());
        assert!(!config.is_file());
        assert!(!config.is_sqlite());
        assert!(!config.is_redis());
        assert_eq!(config.storage_type(), "none");
    }

    #[test]
    fn test_storage_config_none_yaml() {
        let yaml = "type: none\n";
        let config: StorageConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.is_none());
    }

    #[test]
    fn test_storage_config_file() {
        let yaml = r#"
type: file
path: "./data/sessions"
"#;
        let config: StorageConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.is_file());
        assert_eq!(config.get_path(), Some("./data/sessions"));
        assert_eq!(config.storage_type(), "file");
    }

    #[test]
    fn test_storage_config_file_builder() {
        let config = StorageConfig::file("./data/sessions");
        assert!(config.is_file());
        assert_eq!(config.get_path(), Some("./data/sessions"));
    }

    #[test]
    fn test_storage_config_sqlite() {
        let yaml = r#"
type: sqlite
path: "./data/sessions.db"
"#;
        let config: StorageConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.is_sqlite());
        assert_eq!(config.get_path(), Some("./data/sessions.db"));
        assert_eq!(config.storage_type(), "sqlite");
    }

    #[test]
    fn test_storage_config_sqlite_with_table() {
        let yaml = r#"
type: sqlite
path: "./data/sessions.db"
table: "custom_sessions"
"#;
        let config: StorageConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.is_sqlite());
        assert_eq!(config.get_table(), Some("custom_sessions"));
    }

    #[test]
    fn test_storage_config_sqlite_builder() {
        let config = StorageConfig::sqlite("./data/sessions.db");
        assert!(config.is_sqlite());
        assert_eq!(config.get_path(), Some("./data/sessions.db"));
    }

    #[test]
    fn test_storage_config_redis() {
        let yaml = r#"
type: redis
url: "redis://localhost:6379"
prefix: "myagent:"
ttl_seconds: 86400
"#;
        let config: StorageConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.is_redis());
        assert_eq!(config.get_url(), Some("redis://localhost:6379"));
        assert_eq!(config.get_prefix(), "myagent:");
        assert_eq!(config.get_ttl(), Some(86400));
        assert_eq!(config.storage_type(), "redis");
    }

    #[test]
    fn test_storage_config_redis_builder() {
        let config = StorageConfig::redis("redis://localhost:6379");
        assert!(config.is_redis());
        assert_eq!(config.get_url(), Some("redis://localhost:6379"));
        assert_eq!(config.get_prefix(), "agent:");
        assert_eq!(config.get_ttl(), None);
    }

    #[test]
    fn test_storage_config_default_prefix() {
        let config = StorageConfig::default();
        assert_eq!(config.get_prefix(), "agent:");

        let config = StorageConfig::file("./data");
        assert_eq!(config.get_prefix(), "agent:");

        let yaml = r#"
type: redis
url: "redis://localhost:6379"
"#;
        let config: StorageConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.get_prefix(), "agent:");
    }

    #[test]
    fn test_storage_config_accessors() {
        let file_config = StorageConfig::file("./data");
        assert!(file_config.as_file().is_some());
        assert!(file_config.as_sqlite().is_none());
        assert!(file_config.as_redis().is_none());

        let sqlite_config = StorageConfig::sqlite("./data.db");
        assert!(sqlite_config.as_file().is_none());
        assert!(sqlite_config.as_sqlite().is_some());
        assert!(sqlite_config.as_redis().is_none());

        let redis_config = StorageConfig::redis("redis://localhost");
        assert!(redis_config.as_file().is_none());
        assert!(redis_config.as_sqlite().is_none());
        assert!(redis_config.as_redis().is_some());
    }

    #[test]
    fn test_redis_config_builder_methods() {
        let config = RedisStorageConfig {
            url: "redis://localhost:6379".to_string(),
            prefix: None,
            ttl_seconds: None,
        }
        .with_prefix("test:")
        .with_ttl(3600);

        assert_eq!(config.prefix, Some("test:".to_string()));
        assert_eq!(config.ttl_seconds, Some(3600));
    }

    #[test]
    fn test_sqlite_config_builder_methods() {
        let config = SqliteStorageConfig {
            path: "./data.db".to_string(),
            table: None,
        }
        .with_table("custom_table");

        assert_eq!(config.table, Some("custom_table".to_string()));
    }

    #[test]
    fn test_storage_config_serialization() {
        let config = StorageConfig::redis("redis://localhost:6379");
        let yaml = serde_yaml::to_string(&config).unwrap();
        assert!(yaml.contains("type: redis"));
        assert!(yaml.contains("url: redis://localhost:6379"));
    }
}

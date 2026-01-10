//! Persistence layer for agent state storage

mod snapshot;
mod storage;

#[cfg(feature = "sqlite")]
mod sqlite;

#[cfg(feature = "redis-storage")]
mod redis;

pub use snapshot::{AgentSnapshot, MemorySnapshot};
pub use storage::{AgentStorage, FileStorage};

#[cfg(feature = "sqlite")]
pub use sqlite::{SessionInfo, SessionMetadata, SessionOrderBy, SessionQuery, SqliteStorage};

#[cfg(feature = "redis-storage")]
pub use redis::{RedisSessionMeta, RedisStorage};

use crate::error::{AgentError, Result};
use crate::spec::StorageConfig;
use std::sync::Arc;

/// Create storage from configuration
pub async fn create_storage(config: &StorageConfig) -> Result<Option<Arc<dyn AgentStorage>>> {
    match config {
        StorageConfig::None => Ok(None),

        StorageConfig::File(file_config) => Ok(Some(Arc::new(FileStorage::new(&file_config.path)))),

        #[cfg(feature = "sqlite")]
        StorageConfig::Sqlite(sqlite_config) => {
            let storage = SqliteStorage::new(&sqlite_config.path).await?;
            Ok(Some(Arc::new(storage)))
        }

        #[cfg(not(feature = "sqlite"))]
        StorageConfig::Sqlite(_) => Err(AgentError::Config(
            "SQLite storage requires 'sqlite' feature".into(),
        )),

        #[cfg(feature = "redis-storage")]
        StorageConfig::Redis(redis_config) => {
            let mut storage = RedisStorage::new(&redis_config.url)?;
            if let Some(ref prefix) = redis_config.prefix {
                storage = storage.with_prefix(prefix);
            }
            if let Some(ttl) = redis_config.ttl_seconds {
                storage = storage.with_ttl(ttl);
            }
            Ok(Some(Arc::new(storage)))
        }

        #[cfg(not(feature = "redis-storage"))]
        StorageConfig::Redis(_) => Err(AgentError::Config(
            "Redis storage requires 'redis-storage' feature".into(),
        )),
    }
}

#[cfg(not(feature = "sqlite"))]
pub use sqlite_types::*;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_create_storage_none() {
        let config = StorageConfig::default();
        let storage = create_storage(&config).await.unwrap();
        assert!(storage.is_none());
    }

    #[tokio::test]
    async fn test_create_storage_file() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap();
        let config = StorageConfig::file(path);
        let storage = create_storage(&config).await.unwrap();
        assert!(storage.is_some());
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn test_create_storage_sqlite() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.db");
        let config = StorageConfig::sqlite(path.to_str().unwrap());
        let storage = create_storage(&config).await.unwrap();
        assert!(storage.is_some());
    }

    #[cfg(not(feature = "sqlite"))]
    #[tokio::test]
    async fn test_create_storage_sqlite_without_feature() {
        let config = StorageConfig::sqlite("./test.db");
        let result = create_storage(&config).await;
        assert!(result.is_err());
    }
}

#[cfg(not(feature = "sqlite"))]
mod sqlite_types {
    use chrono::{DateTime, Utc};
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;

    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    pub struct SessionMetadata {
        #[serde(default)]
        pub tags: Vec<String>,
        #[serde(default)]
        pub user_id: Option<String>,
        #[serde(default)]
        pub custom: HashMap<String, serde_json::Value>,
        #[serde(default)]
        pub priority: Option<i32>,
        #[serde(default)]
        pub expires_at: Option<DateTime<Utc>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SessionInfo {
        pub session_id: String,
        pub agent_id: String,
        pub created_at: DateTime<Utc>,
        pub updated_at: DateTime<Utc>,
        pub message_count: usize,
        pub current_state: Option<String>,
        #[serde(default)]
        pub metadata: SessionMetadata,
    }

    #[derive(Debug, Clone, Default)]
    pub struct SessionQuery {
        pub agent_id: Option<String>,
        pub state: Option<String>,
        pub tag: Option<String>,
        pub user_id: Option<String>,
        pub created_after: Option<DateTime<Utc>>,
        pub created_before: Option<DateTime<Utc>>,
        pub updated_after: Option<DateTime<Utc>>,
        pub limit: Option<u32>,
        pub offset: Option<u32>,
        pub order_by: SessionOrderBy,
    }

    #[derive(Debug, Clone, Default)]
    pub enum SessionOrderBy {
        #[default]
        UpdatedAtDesc,
        UpdatedAtAsc,
        CreatedAtDesc,
        CreatedAtAsc,
        MessageCountDesc,
    }
}

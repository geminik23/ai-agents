//! Storage backends for AI Agents framework

mod snapshot;
mod storage;

#[cfg(feature = "sqlite")]
mod sqlite;

#[cfg(feature = "redis-storage")]
mod redis;

pub use ai_agents_core::{AgentError, AgentSnapshot, AgentStorage, MemorySnapshot, Result};
pub use snapshot::StateMachineSnapshot;
pub use storage::FileStorage;

#[cfg(feature = "sqlite")]
pub use sqlite::{SessionInfo, SessionMetadata, SessionOrderBy, SessionQuery, SqliteStorage};

#[cfg(feature = "redis-storage")]
pub use redis::{RedisSessionMeta, RedisStorage};

use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StorageConfig {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "file")]
    File { path: String },
    #[serde(rename = "sqlite")]
    Sqlite { path: String },
    #[serde(rename = "redis")]
    Redis {
        url: String,
        #[serde(default)]
        prefix: Option<String>,
        #[serde(default)]
        ttl_seconds: Option<u64>,
    },
}

impl Default for StorageConfig {
    fn default() -> Self {
        StorageConfig::None
    }
}

pub async fn create_storage(config: &StorageConfig) -> Result<Option<Arc<dyn AgentStorage>>> {
    match config {
        StorageConfig::None => Ok(None),
        StorageConfig::File { path } => Ok(Some(Arc::new(FileStorage::new(path)))),

        #[cfg(feature = "sqlite")]
        StorageConfig::Sqlite { path } => {
            let storage = SqliteStorage::new(path).await?;
            Ok(Some(Arc::new(storage)))
        }

        #[cfg(not(feature = "sqlite"))]
        StorageConfig::Sqlite { .. } => Err(AgentError::Config(
            "SQLite storage requires 'sqlite' feature".into(),
        )),

        #[cfg(feature = "redis-storage")]
        StorageConfig::Redis {
            url,
            prefix,
            ttl_seconds,
        } => {
            let mut storage = RedisStorage::new(url)?;
            if let Some(p) = prefix {
                storage = storage.with_prefix(p);
            }
            if let Some(ttl) = ttl_seconds {
                storage = storage.with_ttl(*ttl);
            }
            Ok(Some(Arc::new(storage)))
        }

        #[cfg(not(feature = "redis-storage"))]
        StorageConfig::Redis { .. } => Err(AgentError::Config(
            "Redis storage requires 'redis-storage' feature".into(),
        )),
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

#[cfg(not(feature = "sqlite"))]
pub use sqlite_types::*;

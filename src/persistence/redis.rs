//! Redis storage backend for agent persistence

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{AgentSnapshot, AgentStorage};
use crate::error::{AgentError, Result};

#[cfg(feature = "redis-storage")]
use redis::AsyncCommands;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisSessionMeta {
    pub agent_id: String,
    pub message_count: usize,
    pub current_state: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[cfg(feature = "redis-storage")]
pub struct RedisStorage {
    client: redis::Client,
    prefix: String,
    default_ttl: Option<u64>,
}

#[cfg(feature = "redis-storage")]
impl RedisStorage {
    pub fn new(url: &str) -> Result<Self> {
        let client =
            redis::Client::open(url).map_err(|e| AgentError::Persistence(e.to_string()))?;
        Ok(Self {
            client,
            prefix: "agent:".to_string(),
            default_ttl: None,
        })
    }

    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    pub fn with_ttl(mut self, ttl_seconds: u64) -> Self {
        self.default_ttl = Some(ttl_seconds);
        self
    }

    fn session_key(&self, session_id: &str) -> String {
        format!("{}session:{}", self.prefix, session_id)
    }

    fn meta_key(&self, session_id: &str) -> String {
        format!("{}meta:{}", self.prefix, session_id)
    }

    fn agent_index_key(&self, agent_id: &str) -> String {
        format!("{}agent_sessions:{}", self.prefix, agent_id)
    }

    fn sessions_set_key(&self) -> String {
        format!("{}all_sessions", self.prefix)
    }

    async fn get_connection(&self) -> Result<redis::aio::MultiplexedConnection> {
        self.client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| AgentError::Persistence(e.to_string()))
    }
}

#[cfg(feature = "redis-storage")]
#[async_trait]
impl AgentStorage for RedisStorage {
    async fn save(&self, session_id: &str, snapshot: &AgentSnapshot) -> Result<()> {
        let mut conn = self.get_connection().await?;

        let data =
            serde_json::to_string(snapshot).map_err(|e| AgentError::Persistence(e.to_string()))?;

        let now = Utc::now().to_rfc3339();
        let meta = RedisSessionMeta {
            agent_id: snapshot.agent_id.clone(),
            message_count: snapshot.memory.messages.len(),
            current_state: snapshot
                .state_machine
                .as_ref()
                .map(|sm| sm.current_state.clone()),
            created_at: now.clone(),
            updated_at: now,
        };

        let meta_json =
            serde_json::to_string(&meta).map_err(|e| AgentError::Persistence(e.to_string()))?;

        let session_key = self.session_key(session_id);
        let meta_key = self.meta_key(session_id);
        let agent_index = self.agent_index_key(&snapshot.agent_id);
        let sessions_set = self.sessions_set_key();

        if let Some(ttl) = self.default_ttl {
            let _: () = conn
                .set_ex(&session_key, &data, ttl)
                .await
                .map_err(|e| AgentError::Persistence(e.to_string()))?;
            let _: () = conn
                .set_ex(&meta_key, &meta_json, ttl)
                .await
                .map_err(|e| AgentError::Persistence(e.to_string()))?;
        } else {
            let _: () = conn
                .set(&session_key, &data)
                .await
                .map_err(|e| AgentError::Persistence(e.to_string()))?;
            let _: () = conn
                .set(&meta_key, &meta_json)
                .await
                .map_err(|e| AgentError::Persistence(e.to_string()))?;
        }

        let _: () = conn
            .sadd(&agent_index, session_id)
            .await
            .map_err(|e| AgentError::Persistence(e.to_string()))?;

        let _: () = conn
            .sadd(&sessions_set, session_id)
            .await
            .map_err(|e| AgentError::Persistence(e.to_string()))?;

        Ok(())
    }

    async fn load(&self, session_id: &str) -> Result<Option<AgentSnapshot>> {
        let mut conn = self.get_connection().await?;
        let session_key = self.session_key(session_id);

        let data: Option<String> = conn
            .get(&session_key)
            .await
            .map_err(|e| AgentError::Persistence(e.to_string()))?;

        match data {
            Some(json) => {
                let snapshot = serde_json::from_str(&json)
                    .map_err(|e| AgentError::Persistence(e.to_string()))?;
                Ok(Some(snapshot))
            }
            None => Ok(None),
        }
    }

    async fn delete(&self, session_id: &str) -> Result<()> {
        let mut conn = self.get_connection().await?;

        let meta_key = self.meta_key(session_id);
        let meta_json: Option<String> = conn
            .get(&meta_key)
            .await
            .map_err(|e| AgentError::Persistence(e.to_string()))?;

        if let Some(json) = meta_json {
            if let Ok(meta) = serde_json::from_str::<RedisSessionMeta>(&json) {
                let agent_index = self.agent_index_key(&meta.agent_id);
                let _: () = conn
                    .srem(&agent_index, session_id)
                    .await
                    .map_err(|e| AgentError::Persistence(e.to_string()))?;
            }
        }

        let session_key = self.session_key(session_id);
        let sessions_set = self.sessions_set_key();

        let _: () = conn
            .del(&session_key)
            .await
            .map_err(|e| AgentError::Persistence(e.to_string()))?;
        let _: () = conn
            .del(&meta_key)
            .await
            .map_err(|e| AgentError::Persistence(e.to_string()))?;
        let _: () = conn
            .srem(&sessions_set, session_id)
            .await
            .map_err(|e| AgentError::Persistence(e.to_string()))?;

        Ok(())
    }

    async fn list_sessions(&self) -> Result<Vec<String>> {
        let mut conn = self.get_connection().await?;
        let sessions_set = self.sessions_set_key();

        let sessions: Vec<String> = conn
            .smembers(&sessions_set)
            .await
            .map_err(|e| AgentError::Persistence(e.to_string()))?;

        Ok(sessions)
    }
}

#[cfg(feature = "redis-storage")]
impl RedisStorage {
    pub async fn list_sessions_by_agent(&self, agent_id: &str) -> Result<Vec<String>> {
        let mut conn = self.get_connection().await?;
        let agent_index = self.agent_index_key(agent_id);

        let sessions: Vec<String> = conn
            .smembers(&agent_index)
            .await
            .map_err(|e| AgentError::Persistence(e.to_string()))?;

        Ok(sessions)
    }

    pub async fn exists(&self, session_id: &str) -> Result<bool> {
        let mut conn = self.get_connection().await?;
        let session_key = self.session_key(session_id);

        let exists: bool = conn
            .exists(&session_key)
            .await
            .map_err(|e| AgentError::Persistence(e.to_string()))?;

        Ok(exists)
    }

    pub async fn set_ttl(&self, session_id: &str, ttl_seconds: u64) -> Result<()> {
        let mut conn = self.get_connection().await?;
        let session_key = self.session_key(session_id);
        let meta_key = self.meta_key(session_id);

        let _: () = conn
            .expire(&session_key, ttl_seconds as i64)
            .await
            .map_err(|e| AgentError::Persistence(e.to_string()))?;
        let _: () = conn
            .expire(&meta_key, ttl_seconds as i64)
            .await
            .map_err(|e| AgentError::Persistence(e.to_string()))?;

        Ok(())
    }

    pub async fn get_meta(&self, session_id: &str) -> Result<Option<RedisSessionMeta>> {
        let mut conn = self.get_connection().await?;
        let meta_key = self.meta_key(session_id);

        let data: Option<String> = conn
            .get(&meta_key)
            .await
            .map_err(|e| AgentError::Persistence(e.to_string()))?;

        match data {
            Some(json) => {
                let meta = serde_json::from_str(&json)
                    .map_err(|e| AgentError::Persistence(e.to_string()))?;
                Ok(Some(meta))
            }
            None => Ok(None),
        }
    }

    pub async fn expire_sessions(&self, before: DateTime<Utc>) -> Result<usize> {
        let sessions = self.list_sessions().await?;
        let mut deleted = 0;

        for session_id in sessions {
            if let Some(meta) = self.get_meta(&session_id).await? {
                if let Ok(updated_at) = DateTime::parse_from_rfc3339(&meta.updated_at) {
                    if updated_at.with_timezone(&Utc) < before {
                        self.delete(&session_id).await?;
                        deleted += 1;
                    }
                }
            }
        }

        Ok(deleted)
    }
}

#[cfg(not(feature = "redis-storage"))]
pub struct RedisStorage {
    _private: (),
}

#[cfg(not(feature = "redis-storage"))]
impl RedisStorage {
    pub fn new(_url: &str) -> Result<Self> {
        Err(AgentError::Persistence(
            "Redis storage requires 'redis-storage' feature".to_string(),
        ))
    }
}

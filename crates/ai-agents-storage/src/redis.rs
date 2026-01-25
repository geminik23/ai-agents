//! Redis storage backend for agent persistence

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{AgentError, AgentSnapshot, AgentStorage, Result};

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
fn map_redis_err(e: redis::RedisError) -> AgentError {
    AgentError::Persistence(e.to_string())
}

#[cfg(feature = "redis-storage")]
impl RedisStorage {
    pub fn new(url: &str) -> Result<Self> {
        let client = redis::Client::open(url).map_err(map_redis_err)?;
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
            .map_err(map_redis_err)
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
        let current_state = snapshot
            .state_machine
            .as_ref()
            .map(|sm| sm.current_state.clone());

        let meta = RedisSessionMeta {
            agent_id: snapshot.agent_id.clone(),
            message_count: snapshot.memory.messages.len(),
            current_state,
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
            redis::cmd("SETEX")
                .arg(&session_key)
                .arg(ttl)
                .arg(&data)
                .query_async::<()>(&mut conn)
                .await
                .map_err(map_redis_err)?;
            redis::cmd("SETEX")
                .arg(&meta_key)
                .arg(ttl)
                .arg(&meta_json)
                .query_async::<()>(&mut conn)
                .await
                .map_err(map_redis_err)?;
        } else {
            redis::cmd("SET")
                .arg(&session_key)
                .arg(&data)
                .query_async::<()>(&mut conn)
                .await
                .map_err(map_redis_err)?;
            redis::cmd("SET")
                .arg(&meta_key)
                .arg(&meta_json)
                .query_async::<()>(&mut conn)
                .await
                .map_err(map_redis_err)?;
        }

        redis::cmd("SADD")
            .arg(&agent_index)
            .arg(session_id)
            .query_async::<()>(&mut conn)
            .await
            .map_err(map_redis_err)?;

        redis::cmd("SADD")
            .arg(&sessions_set)
            .arg(session_id)
            .query_async::<()>(&mut conn)
            .await
            .map_err(map_redis_err)?;

        Ok(())
    }

    async fn load(&self, session_id: &str) -> Result<Option<AgentSnapshot>> {
        let mut conn = self.get_connection().await?;
        let session_key = self.session_key(session_id);

        let data: Option<String> = redis::cmd("GET")
            .arg(&session_key)
            .query_async(&mut conn)
            .await
            .map_err(map_redis_err)?;

        match data {
            Some(ref json_str) => {
                let snapshot = serde_json::from_str(json_str)
                    .map_err(|e| AgentError::Persistence(e.to_string()))?;
                Ok(Some(snapshot))
            }
            None => Ok(None),
        }
    }

    async fn delete(&self, session_id: &str) -> Result<()> {
        let mut conn = self.get_connection().await?;

        let meta_key = self.meta_key(session_id);
        let meta_json: Option<String> = redis::cmd("GET")
            .arg(&meta_key)
            .query_async(&mut conn)
            .await
            .map_err(map_redis_err)?;

        if let Some(ref json_str) = meta_json {
            if let Ok(meta) = serde_json::from_str::<RedisSessionMeta>(json_str) {
                let agent_index = self.agent_index_key(&meta.agent_id);
                redis::cmd("SREM")
                    .arg(&agent_index)
                    .arg(session_id)
                    .query_async::<()>(&mut conn)
                    .await
                    .map_err(map_redis_err)?;
            }
        }

        let session_key = self.session_key(session_id);
        let sessions_set = self.sessions_set_key();

        redis::cmd("DEL")
            .arg(&session_key)
            .query_async::<()>(&mut conn)
            .await
            .map_err(map_redis_err)?;
        redis::cmd("DEL")
            .arg(&meta_key)
            .query_async::<()>(&mut conn)
            .await
            .map_err(map_redis_err)?;
        redis::cmd("SREM")
            .arg(&sessions_set)
            .arg(session_id)
            .query_async::<()>(&mut conn)
            .await
            .map_err(map_redis_err)?;

        Ok(())
    }

    async fn list_sessions(&self) -> Result<Vec<String>> {
        let mut conn = self.get_connection().await?;
        let sessions_set = self.sessions_set_key();

        let sessions: Vec<String> = redis::cmd("SMEMBERS")
            .arg(&sessions_set)
            .query_async(&mut conn)
            .await
            .map_err(map_redis_err)?;

        Ok(sessions)
    }
}

#[cfg(feature = "redis-storage")]
impl RedisStorage {
    pub async fn list_sessions_by_agent(&self, agent_id: &str) -> Result<Vec<String>> {
        let mut conn = self.get_connection().await?;
        let agent_index = self.agent_index_key(agent_id);

        let sessions: Vec<String> = redis::cmd("SMEMBERS")
            .arg(&agent_index)
            .query_async(&mut conn)
            .await
            .map_err(map_redis_err)?;

        Ok(sessions)
    }

    pub async fn exists(&self, session_id: &str) -> Result<bool> {
        let mut conn = self.get_connection().await?;
        let session_key = self.session_key(session_id);

        let exists: bool = redis::cmd("EXISTS")
            .arg(&session_key)
            .query_async(&mut conn)
            .await
            .map_err(map_redis_err)?;

        Ok(exists)
    }

    pub async fn set_ttl(&self, session_id: &str, ttl_seconds: u64) -> Result<()> {
        let mut conn = self.get_connection().await?;
        let session_key = self.session_key(session_id);
        let meta_key = self.meta_key(session_id);

        redis::cmd("EXPIRE")
            .arg(&session_key)
            .arg(ttl_seconds as i64)
            .query_async::<()>(&mut conn)
            .await
            .map_err(map_redis_err)?;
        redis::cmd("EXPIRE")
            .arg(&meta_key)
            .arg(ttl_seconds as i64)
            .query_async::<()>(&mut conn)
            .await
            .map_err(map_redis_err)?;

        Ok(())
    }

    pub async fn get_meta(&self, session_id: &str) -> Result<Option<RedisSessionMeta>> {
        let mut conn = self.get_connection().await?;
        let meta_key = self.meta_key(session_id);

        let data: Option<String> = redis::cmd("GET")
            .arg(&meta_key)
            .query_async(&mut conn)
            .await
            .map_err(map_redis_err)?;

        match data {
            Some(ref json_str) => {
                let meta = serde_json::from_str(json_str)
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

//! Redis storage backend for agent persistence

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use crate::StorageCapability;
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
const PERMANENT_EXPIRY_SCORE: f64 = 253_402_300_799.0;

#[cfg(feature = "redis-storage")]
const MUTATE_SESSION_SCRIPT: &str = r#"
local operation = ARGV[1]
local session_id = ARGV[2]
local agent_index_prefix = ARGV[3]
local permanent_score = tonumber(ARGV[4])

if operation ~= 'save' and operation ~= 'delete' and operation ~= 'set_ttl' then
    return redis.error_reply('unsupported session mutation')
end
if operation == 'set_ttl' and redis.call('EXISTS', KEYS[1]) == 0 then
    return 0
end

local function decode_meta(json)
    local ok, meta = pcall(cjson.decode, json)
    if not ok or type(meta) ~= 'table' or type(meta.agent_id) ~= 'string' then
        return nil
    end
    if type(meta.message_count) ~= 'number' or meta.message_count < 0 or meta.message_count ~= math.floor(meta.message_count) then
        return nil
    end
    if type(meta.created_at) ~= 'string' or type(meta.updated_at) ~= 'string' then
        return nil
    end
    if meta.current_state ~= cjson.null and type(meta.current_state) ~= 'string' then
        return nil
    end
    return meta
end

local previous_meta_json = redis.call('GET', KEYS[2])
local previous_meta = nil
if previous_meta_json then
    previous_meta = decode_meta(previous_meta_json)
    if not previous_meta then
        return redis.error_reply('session metadata is malformed')
    end
end

local new_meta = nil
if operation == 'save' then
    new_meta = decode_meta(ARGV[6])
    if not new_meta then
        return redis.error_reply('new session metadata is malformed')
    end
    if previous_meta and type(previous_meta.created_at) == 'string' then
        new_meta.created_at = previous_meta.created_at
    end
elseif operation == 'set_ttl' and not previous_meta then
    return redis.error_reply('session metadata is missing')
end

local ttl = nil
local expiry_score = permanent_score
if operation == 'save' or operation == 'set_ttl' then
    if ARGV[5] ~= '' then
        ttl = tonumber(ARGV[5])
        if not ttl or ttl < 0 or ttl ~= math.floor(ttl) then
            return redis.error_reply('session TTL must be a non-negative integer')
        end
        if operation == 'save' and ttl == 0 then
            return redis.error_reply('session TTL must be greater than zero')
        end
        local time = redis.call('TIME')
        local now = tonumber(time[1]) + tonumber(time[2]) / 1000000
        if now + ttl >= permanent_score then
            return redis.error_reply('session TTL exceeds the supported Redis range')
        end
        expiry_score = now + ttl
    elseif operation == 'set_ttl' then
        return redis.error_reply('session TTL is required')
    end
end

local previous_agent_index = nil
if previous_meta then
    previous_agent_index = agent_index_prefix .. previous_meta.agent_id
end
local current_agent_index = nil
if operation == 'save' then
    current_agent_index = agent_index_prefix .. ARGV[7]
elseif previous_meta then
    current_agent_index = previous_agent_index
end

local function validate_index(key)
    if not key then
        return true
    end
    local kind = redis.call('TYPE', key).ok
    return kind == 'none' or kind == 'set' or kind == 'zset'
end

if not validate_index(KEYS[3]) or not validate_index(previous_agent_index) or not validate_index(current_agent_index) then
    return redis.error_reply('session index has unsupported Redis type')
end

local function migrate_index(key)
    if not key then
        return
    end
    if redis.call('TYPE', key).ok == 'set' then
        local members = redis.call('SMEMBERS', key)
        redis.call('DEL', key)
        for _, member in ipairs(members) do
            redis.call('ZADD', key, permanent_score, member)
        end
    end
end

migrate_index(KEYS[3])
migrate_index(previous_agent_index)
if current_agent_index ~= previous_agent_index then
    migrate_index(current_agent_index)
end

if operation == 'delete' then
    redis.call('DEL', KEYS[1], KEYS[2])
    redis.call('ZREM', KEYS[3], session_id)
    if previous_agent_index then
        redis.call('ZREM', previous_agent_index, session_id)
    end
    return 1
end

if operation == 'save' then
    redis.call('SET', KEYS[1], ARGV[8])
    redis.call('SET', KEYS[2], cjson.encode(new_meta))
    if ttl then
        redis.call('EXPIRE', KEYS[1], ttl)
        redis.call('EXPIRE', KEYS[2], ttl)
    end
    if previous_agent_index and previous_agent_index ~= current_agent_index then
        redis.call('ZREM', previous_agent_index, session_id)
    end
    redis.call('ZADD', current_agent_index, expiry_score, session_id)
    redis.call('ZADD', KEYS[3], expiry_score, session_id)
    return 1
end

if operation == 'set_ttl' then
    if ttl <= 0 then
        redis.call('DEL', KEYS[1], KEYS[2])
        redis.call('ZREM', KEYS[3], session_id)
        if current_agent_index then
            redis.call('ZREM', current_agent_index, session_id)
        end
        return 1
    end
    redis.call('EXPIRE', KEYS[1], ttl)
    redis.call('EXPIRE', KEYS[2], ttl)
    redis.call('ZADD', KEYS[3], expiry_score, session_id)
    if current_agent_index then
        redis.call('ZADD', current_agent_index, expiry_score, session_id)
    end
    return 1
end

return redis.error_reply('unreachable session mutation')
"#;

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

    fn sessions_index_key(&self) -> String {
        format!("{}all_sessions", self.prefix)
    }

    async fn ensure_zset_index(
        connection: &mut redis::aio::MultiplexedConnection,
        key: &str,
    ) -> Result<()> {
        //
        // Existing SET indexes are migrated in one server-side operation before ZSET commands run.
        //
        redis::Script::new(
            r#"
            local kind = redis.call('TYPE', KEYS[1]).ok
            if kind == 'set' then
                local members = redis.call('SMEMBERS', KEYS[1])
                redis.call('DEL', KEYS[1])
                for _, session_id in ipairs(members) do
                    redis.call('ZADD', KEYS[1], ARGV[1], session_id)
                end
            elseif kind ~= 'none' and kind ~= 'zset' then
                return redis.error_reply('session index has unsupported Redis type: ' .. kind)
            end
            return 1
            "#,
        )
        .key(key)
        .arg(PERMANENT_EXPIRY_SCORE)
        .invoke_async::<i32>(connection)
        .await
        .map(|_| ())
        .map_err(map_redis_err)
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

        let session_key = self.session_key(session_id);
        let meta_key = self.meta_key(session_id);
        let meta = RedisSessionMeta {
            agent_id: snapshot.agent_id.clone(),
            message_count: snapshot.memory.messages.len(),
            current_state,
            created_at: now.clone(),
            updated_at: now,
        };
        let meta_json =
            serde_json::to_string(&meta).map_err(|e| AgentError::Persistence(e.to_string()))?;
        let ttl = self
            .default_ttl
            .map(|ttl| ttl.to_string())
            .unwrap_or_default();

        //
        // Snapshot, metadata, ownership, TTL, and indexes change under one Redis execution boundary.
        //
        redis::Script::new(MUTATE_SESSION_SCRIPT)
            .key(&session_key)
            .key(&meta_key)
            .key(self.sessions_index_key())
            .arg("save")
            .arg(session_id)
            .arg(format!("{}agent_sessions:", self.prefix))
            .arg(PERMANENT_EXPIRY_SCORE)
            .arg(ttl)
            .arg(meta_json)
            .arg(&snapshot.agent_id)
            .arg(data)
            .invoke_async::<i32>(&mut conn)
            .await
            .map(|_| ())
            .map_err(map_redis_err)
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

        let session_key = self.session_key(session_id);
        let meta_key = self.meta_key(session_id);

        redis::Script::new(MUTATE_SESSION_SCRIPT)
            .key(&session_key)
            .key(&meta_key)
            .key(self.sessions_index_key())
            .arg("delete")
            .arg(session_id)
            .arg(format!("{}agent_sessions:", self.prefix))
            .arg(PERMANENT_EXPIRY_SCORE)
            .arg("")
            .arg("")
            .arg("")
            .arg("")
            .invoke_async::<i32>(&mut conn)
            .await
            .map(|_| ())
            .map_err(map_redis_err)
    }

    async fn list_sessions(&self) -> Result<Vec<String>> {
        let mut conn = self.get_connection().await?;
        let sessions_index = self.sessions_index_key();
        Self::ensure_zset_index(&mut conn, &sessions_index).await?;
        redis::Script::new(
            r#"
            local time = redis.call('TIME')
            local now = tonumber(time[1]) + tonumber(time[2]) / 1000000
            local sessions = redis.call('ZRANGE', KEYS[1], 0, -1, 'WITHSCORES')
            local valid = {}
            for index = 1, #sessions, 2 do
                local session_id = sessions[index]
                local score = tonumber(sessions[index + 1])
                local ttl = redis.call('PTTL', ARGV[1] .. session_id)
                if ttl >= 0 then
                    if score <= now then
                        redis.call('ZADD', KEYS[1], now + ttl / 1000, session_id)
                    end
                    table.insert(valid, session_id)
                elseif ttl == -1 then
                    if score <= now then
                        redis.call('ZADD', KEYS[1], ARGV[2], session_id)
                    end
                    table.insert(valid, session_id)
                else
                    redis.call('ZREM', KEYS[1], session_id)
                end
            end
            return valid
            "#,
        )
        .key(&sessions_index)
        .arg(format!("{}session:", self.prefix))
        .arg(PERMANENT_EXPIRY_SCORE)
        .invoke_async(&mut conn)
        .await
        .map_err(map_redis_err)
    }
}

#[cfg(feature = "redis-storage")]
impl RedisStorage {
    pub async fn list_sessions_by_agent(&self, agent_id: &str) -> Result<Vec<String>> {
        let mut conn = self.get_connection().await?;
        let agent_index = self.agent_index_key(agent_id);
        Self::ensure_zset_index(&mut conn, &agent_index).await?;

        redis::Script::new(
            r#"
            local time = redis.call('TIME')
            local now = tonumber(time[1]) + tonumber(time[2]) / 1000000
            local sessions = redis.call('ZRANGE', KEYS[1], 0, -1, 'WITHSCORES')
            local valid = {}
            for index = 1, #sessions, 2 do
                local session_id = sessions[index]
                local score = tonumber(sessions[index + 1])
                local ttl = redis.call('PTTL', ARGV[1] .. session_id)
                local meta_json = redis.call('GET', ARGV[2] .. session_id)
                local owned = false
                if ttl ~= -2 and meta_json then
                    local decoded, meta = pcall(cjson.decode, meta_json)
                    owned = decoded and type(meta) == 'table' and type(meta.agent_id) == 'string' and meta.agent_id == ARGV[3]
                end
                if owned then
                    if ttl >= 0 and score <= now then
                        redis.call('ZADD', KEYS[1], now + ttl / 1000, session_id)
                    elseif ttl == -1 and score <= now then
                        redis.call('ZADD', KEYS[1], ARGV[4], session_id)
                    end
                    table.insert(valid, session_id)
                else
                    redis.call('ZREM', KEYS[1], session_id)
                end
            end
            return valid
            "#,
        )
        .key(&agent_index)
        .arg(format!("{}session:", self.prefix))
        .arg(format!("{}meta:", self.prefix))
        .arg(agent_id)
        .arg(PERMANENT_EXPIRY_SCORE)
        .invoke_async(&mut conn)
        .await
        .map_err(map_redis_err)
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

        redis::Script::new(MUTATE_SESSION_SCRIPT)
            .key(&session_key)
            .key(&meta_key)
            .key(self.sessions_index_key())
            .arg("set_ttl")
            .arg(session_id)
            .arg(format!("{}agent_sessions:", self.prefix))
            .arg(PERMANENT_EXPIRY_SCORE)
            .arg(ttl_seconds)
            .arg("")
            .arg("")
            .arg("")
            .invoke_async::<i32>(&mut conn)
            .await
            .map(|_| ())
            .map_err(map_redis_err)
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
            if let Some(meta) = self.get_meta(&session_id).await?
                && let Ok(updated_at) = DateTime::parse_from_rfc3339(&meta.updated_at)
                && updated_at.with_timezone(&Utc) < before
            {
                self.delete(&session_id).await?;
                deleted += 1;
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

#[cfg(all(test, feature = "redis-storage"))]
mod tests {
    use super::*;

    #[test]
    fn reports_snapshot_capability_without_connecting() {
        let storage = RedisStorage::new("redis://127.0.0.1/").unwrap();

        assert!(storage.supports(StorageCapability::Snapshot));
        assert!(!storage.supports(StorageCapability::SessionMetadata));
    }

    #[test]
    fn constructs_namespaced_keys_without_connecting() {
        let storage = RedisStorage::new("redis://127.0.0.1/")
            .unwrap()
            .with_prefix("test:");

        assert_eq!(storage.session_key("session"), "test:session:session");
        assert_eq!(storage.meta_key("session"), "test:meta:session");
        assert_eq!(
            storage.agent_index_key("agent"),
            "test:agent_sessions:agent"
        );
        assert_eq!(storage.sessions_index_key(), "test:all_sessions");
    }

    #[tokio::test]
    #[ignore = "requires a Redis service"]
    async fn indexes_repair_ttl_delete_and_agent_reassignment() {
        let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1/".to_string());
        let prefix = format!("ai-agents-test:{}:", uuid::Uuid::new_v4());
        let storage = RedisStorage::new(&url).unwrap().with_prefix(&prefix);
        storage
            .save("session", &AgentSnapshot::new("agent-a".to_string()))
            .await
            .unwrap();
        let mut connection = storage.get_connection().await.unwrap();
        let global_type: String = redis::cmd("TYPE")
            .arg(storage.sessions_index_key())
            .query_async(&mut connection)
            .await
            .unwrap();
        let agent_type: String = redis::cmd("TYPE")
            .arg(storage.agent_index_key("agent-a"))
            .query_async(&mut connection)
            .await
            .unwrap();
        assert_eq!(global_type, "zset");
        assert_eq!(agent_type, "zset");
        assert_eq!(
            storage.list_sessions_by_agent("agent-a").await.unwrap(),
            vec!["session"]
        );

        storage
            .save("session", &AgentSnapshot::new("agent-b".to_string()))
            .await
            .unwrap();
        let old_owner_score: Option<f64> = redis::cmd("ZSCORE")
            .arg(storage.agent_index_key("agent-a"))
            .arg("session")
            .query_async(&mut connection)
            .await
            .unwrap();
        let new_owner_score: Option<f64> = redis::cmd("ZSCORE")
            .arg(storage.agent_index_key("agent-b"))
            .arg("session")
            .query_async(&mut connection)
            .await
            .unwrap();
        assert!(old_owner_score.is_none());
        assert!(new_owner_score.is_some());
        assert!(
            storage
                .list_sessions_by_agent("agent-a")
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            storage.list_sessions_by_agent("agent-b").await.unwrap(),
            vec!["session"]
        );

        let concurrent_storage =
            std::sync::Arc::new(RedisStorage::new(&url).unwrap().with_prefix(&prefix));
        let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(9));
        let mut saves = Vec::new();
        for index in 0..8 {
            let storage = concurrent_storage.clone();
            let barrier = barrier.clone();
            saves.push(tokio::spawn(async move {
                barrier.wait().await;
                let agent_id = format!("concurrent-agent-{index}");
                storage
                    .save("concurrent", &AgentSnapshot::new(agent_id))
                    .await
                    .unwrap();
            }));
        }
        barrier.wait().await;
        for save in saves {
            save.await.unwrap();
        }
        let owner = concurrent_storage
            .get_meta("concurrent")
            .await
            .unwrap()
            .unwrap()
            .agent_id;
        let mut raw_owner_count = 0;
        for index in 0..8 {
            let agent_id = format!("concurrent-agent-{index}");
            let score: Option<f64> = redis::cmd("ZSCORE")
                .arg(concurrent_storage.agent_index_key(&agent_id))
                .arg("concurrent")
                .query_async(&mut connection)
                .await
                .unwrap();
            if score.is_some() {
                raw_owner_count += 1;
                assert_eq!(agent_id, owner);
            }
        }
        assert_eq!(raw_owner_count, 1);

        for index in 0..8 {
            let agent_id = format!("concurrent-agent-{index}");
            let sessions = concurrent_storage
                .list_sessions_by_agent(&agent_id)
                .await
                .unwrap();
            if agent_id == owner {
                assert_eq!(sessions, vec!["concurrent"]);
            } else {
                assert!(sessions.is_empty());
            }
        }

        storage.delete("session").await.unwrap();
        concurrent_storage.delete("concurrent").await.unwrap();
        assert!(storage.list_sessions().await.unwrap().is_empty());
        assert!(
            storage
                .list_sessions_by_agent("agent-b")
                .await
                .unwrap()
                .is_empty()
        );

        storage
            .save("missing-meta", &AgentSnapshot::new("agent-e".to_string()))
            .await
            .unwrap();
        redis::cmd("DEL")
            .arg(storage.meta_key("missing-meta"))
            .query_async::<()>(&mut connection)
            .await
            .unwrap();
        assert!(storage.set_ttl("missing-meta", 60).await.is_err());
        let missing_meta_ttl: i64 = redis::cmd("TTL")
            .arg(storage.session_key("missing-meta"))
            .query_async(&mut connection)
            .await
            .unwrap();
        assert_eq!(missing_meta_ttl, -1);
        storage.delete("missing-meta").await.unwrap();
        assert!(
            storage
                .list_sessions_by_agent("agent-e")
                .await
                .unwrap()
                .is_empty()
        );

        storage
            .save("scalar-meta", &AgentSnapshot::new("agent-f".to_string()))
            .await
            .unwrap();
        redis::cmd("SET")
            .arg(storage.meta_key("scalar-meta"))
            .arg("1")
            .query_async::<()>(&mut connection)
            .await
            .unwrap();
        assert!(
            storage
                .list_sessions_by_agent("agent-f")
                .await
                .unwrap()
                .is_empty()
        );
        redis::cmd("DEL")
            .arg(storage.meta_key("scalar-meta"))
            .query_async::<()>(&mut connection)
            .await
            .unwrap();
        storage.delete("scalar-meta").await.unwrap();

        let excessive_ttl = RedisStorage::new(&url)
            .unwrap()
            .with_prefix(&prefix)
            .with_ttl(u64::MAX);
        assert!(
            excessive_ttl
                .save("excessive-ttl", &AgentSnapshot::new("agent-g".to_string()))
                .await
                .is_err()
        );
        assert!(!excessive_ttl.exists("excessive-ttl").await.unwrap());
        let excessive_score: Option<f64> = redis::cmd("ZSCORE")
            .arg(excessive_ttl.sessions_index_key())
            .arg("excessive-ttl")
            .query_async(&mut connection)
            .await
            .unwrap();
        assert!(excessive_score.is_none());

        let expiring = RedisStorage::new(&url)
            .unwrap()
            .with_prefix(&prefix)
            .with_ttl(1);
        expiring
            .save("expiring", &AgentSnapshot::new("agent-c".to_string()))
            .await
            .unwrap();
        redis::pipe()
            .atomic()
            .cmd("ZADD")
            .arg(expiring.sessions_index_key())
            .arg(0)
            .arg("expiring")
            .ignore()
            .cmd("ZADD")
            .arg(expiring.agent_index_key("agent-c"))
            .arg(0)
            .arg("expiring")
            .ignore()
            .query_async::<()>(&mut connection)
            .await
            .unwrap();
        assert!(
            expiring
                .list_sessions()
                .await
                .unwrap()
                .contains(&"expiring".to_string())
        );
        assert_eq!(
            expiring.list_sessions_by_agent("agent-c").await.unwrap(),
            vec!["expiring"]
        );
        let repaired_score: f64 = redis::cmd("ZSCORE")
            .arg(expiring.sessions_index_key())
            .arg("expiring")
            .query_async(&mut connection)
            .await
            .unwrap();
        assert!(repaired_score > Utc::now().timestamp() as f64);

        storage
            .save("retimed", &AgentSnapshot::new("agent-d".to_string()))
            .await
            .unwrap();
        storage.set_ttl("retimed", 1).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(1_200)).await;
        assert!(expiring.list_sessions().await.unwrap().is_empty());
        assert!(
            expiring
                .list_sessions_by_agent("agent-c")
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            storage
                .list_sessions_by_agent("agent-d")
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    #[ignore = "requires a Redis service"]
    async fn legacy_set_indexes_migrate_to_expiry_zsets() {
        let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1/".to_string());
        let prefix = format!("ai-agents-legacy-index-test:{}:", uuid::Uuid::new_v4());
        let storage = RedisStorage::new(&url).unwrap().with_prefix(&prefix);
        let session_id = "legacy";
        let snapshot = AgentSnapshot::new("agent".to_string());
        let now = Utc::now().to_rfc3339();
        let metadata = RedisSessionMeta {
            agent_id: "agent".to_string(),
            message_count: 0,
            current_state: None,
            created_at: now.clone(),
            updated_at: now,
        };
        let mut connection = storage.get_connection().await.unwrap();
        redis::pipe()
            .atomic()
            .cmd("SET")
            .arg(storage.session_key(session_id))
            .arg(serde_json::to_string(&snapshot).unwrap())
            .ignore()
            .cmd("SET")
            .arg(storage.meta_key(session_id))
            .arg(serde_json::to_string(&metadata).unwrap())
            .ignore()
            .cmd("SADD")
            .arg(storage.sessions_index_key())
            .arg(session_id)
            .ignore()
            .cmd("SADD")
            .arg(storage.agent_index_key("agent"))
            .arg(session_id)
            .ignore()
            .query_async::<()>(&mut connection)
            .await
            .unwrap();

        assert_eq!(storage.list_sessions().await.unwrap(), vec![session_id]);
        assert_eq!(
            storage.list_sessions_by_agent("agent").await.unwrap(),
            vec![session_id]
        );
        let global_type: String = redis::cmd("TYPE")
            .arg(storage.sessions_index_key())
            .query_async(&mut connection)
            .await
            .unwrap();
        let agent_type: String = redis::cmd("TYPE")
            .arg(storage.agent_index_key("agent"))
            .query_async(&mut connection)
            .await
            .unwrap();
        assert_eq!(global_type, "zset");
        assert_eq!(agent_type, "zset");

        storage.delete(session_id).await.unwrap();
    }
}

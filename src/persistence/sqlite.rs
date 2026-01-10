//! SQLite storage backend for agent persistence

use std::str::FromStr;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{AgentSnapshot, AgentStorage};
use crate::error::{AgentError, Result};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionMetadata {
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default)]
    pub custom: std::collections::HashMap<String, serde_json::Value>,
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

#[cfg(feature = "sqlite")]
pub struct SqliteStorage {
    pool: sqlx::SqlitePool,
}

#[cfg(feature = "sqlite")]
impl SqliteStorage {
    pub async fn new(path: &str) -> Result<Self> {
        let pool = Self::connect(path).await?;
        let storage = Self { pool };
        storage.run_migrations().await?;
        Ok(storage)
    }

    pub async fn in_memory() -> Result<Self> {
        Self::new(":memory:").await
    }

    async fn connect(path: &str) -> Result<sqlx::SqlitePool> {
        let options = sqlx::sqlite::SqliteConnectOptions::from_str(path)
            .map_err(|e| AgentError::Persistence(e.to_string()))?
            .create_if_missing(true)
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal);

        sqlx::SqlitePool::connect_with(options)
            .await
            .map_err(|e| AgentError::Persistence(e.to_string()))
    }

    async fn run_migrations(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0,
                current_state TEXT,
                data TEXT NOT NULL,
                metadata TEXT
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| AgentError::Persistence(e.to_string()))?;

        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_sessions_agent_id ON sessions(agent_id)
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| AgentError::Persistence(e.to_string()))?;

        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at)
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| AgentError::Persistence(e.to_string()))?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS session_tags (
                session_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (session_id, tag),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| AgentError::Persistence(e.to_string()))?;

        Ok(())
    }

    fn extract_session_info(
        session_id: String,
        agent_id: String,
        created_at: String,
        updated_at: String,
        message_count: i64,
        current_state: Option<String>,
        metadata_json: Option<String>,
    ) -> SessionInfo {
        let metadata = metadata_json
            .and_then(|m| serde_json::from_str(&m).ok())
            .unwrap_or_default();

        SessionInfo {
            session_id,
            agent_id,
            created_at: DateTime::parse_from_rfc3339(&created_at)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            updated_at: DateTime::parse_from_rfc3339(&updated_at)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            message_count: message_count as usize,
            current_state,
            metadata,
        }
    }

    pub async fn save_with_metadata(
        &self,
        session_id: &str,
        snapshot: &AgentSnapshot,
        metadata: &SessionMetadata,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let data =
            serde_json::to_string(snapshot).map_err(|e| AgentError::Persistence(e.to_string()))?;
        let metadata_json =
            serde_json::to_string(metadata).map_err(|e| AgentError::Persistence(e.to_string()))?;

        let message_count = snapshot.memory.messages.len() as i64;
        let current_state = snapshot
            .state_machine
            .as_ref()
            .map(|sm| sm.current_state.clone());

        sqlx::query(
            r#"
            INSERT INTO sessions (session_id, agent_id, created_at, updated_at, message_count, current_state, data, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                message_count = excluded.message_count,
                current_state = excluded.current_state,
                data = excluded.data,
                metadata = excluded.metadata
            "#,
        )
        .bind(session_id)
        .bind(&snapshot.agent_id)
        .bind(&now)
        .bind(&now)
        .bind(message_count)
        .bind(&current_state)
        .bind(&data)
        .bind(&metadata_json)
        .execute(&self.pool)
        .await
        .map_err(|e| AgentError::Persistence(e.to_string()))?;

        sqlx::query("DELETE FROM session_tags WHERE session_id = ?")
            .bind(session_id)
            .execute(&self.pool)
            .await
            .map_err(|e| AgentError::Persistence(e.to_string()))?;

        for tag in &metadata.tags {
            sqlx::query("INSERT INTO session_tags (session_id, tag) VALUES (?, ?)")
                .bind(session_id)
                .bind(tag)
                .execute(&self.pool)
                .await
                .map_err(|e| AgentError::Persistence(e.to_string()))?;
        }

        Ok(())
    }

    pub async fn get_metadata(&self, session_id: &str) -> Result<Option<SessionMetadata>> {
        let row: Option<(Option<String>,)> =
            sqlx::query_as("SELECT metadata FROM sessions WHERE session_id = ?")
                .bind(session_id)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| AgentError::Persistence(e.to_string()))?;

        match row {
            Some((Some(metadata_json),)) => {
                let metadata = serde_json::from_str(&metadata_json)
                    .map_err(|e| AgentError::Persistence(e.to_string()))?;
                Ok(Some(metadata))
            }
            Some((None,)) => Ok(Some(SessionMetadata::default())),
            None => Ok(None),
        }
    }

    pub async fn list_sessions_by_agent(&self, agent_id: &str) -> Result<Vec<SessionInfo>> {
        let rows: Vec<(
            String,
            String,
            String,
            String,
            i64,
            Option<String>,
            Option<String>,
        )> = sqlx::query_as(
            r#"
            SELECT session_id, agent_id, created_at, updated_at, message_count, current_state, metadata
            FROM sessions
            WHERE agent_id = ?
            ORDER BY updated_at DESC
            "#,
        )
        .bind(agent_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| AgentError::Persistence(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(
                |(
                    session_id,
                    agent_id,
                    created_at,
                    updated_at,
                    message_count,
                    current_state,
                    metadata,
                )| {
                    Self::extract_session_info(
                        session_id,
                        agent_id,
                        created_at,
                        updated_at,
                        message_count,
                        current_state,
                        metadata,
                    )
                },
            )
            .collect())
    }

    pub async fn search_sessions(&self, query: &SessionQuery) -> Result<Vec<SessionInfo>> {
        let mut sql = String::from(
            r#"
            SELECT DISTINCT s.session_id, s.agent_id, s.created_at, s.updated_at,
                   s.message_count, s.current_state, s.metadata
            FROM sessions s
            LEFT JOIN session_tags t ON s.session_id = t.session_id
            WHERE 1=1
            "#,
        );

        if query.agent_id.is_some() {
            sql.push_str(" AND s.agent_id = ?");
        }
        if query.state.is_some() {
            sql.push_str(" AND s.current_state = ?");
        }
        if query.tag.is_some() {
            sql.push_str(" AND t.tag = ?");
        }
        if query.user_id.is_some() {
            sql.push_str(" AND json_extract(s.metadata, '$.user_id') = ?");
        }
        if query.created_after.is_some() {
            sql.push_str(" AND s.created_at >= ?");
        }
        if query.created_before.is_some() {
            sql.push_str(" AND s.created_at <= ?");
        }
        if query.updated_after.is_some() {
            sql.push_str(" AND s.updated_at >= ?");
        }

        sql.push_str(match query.order_by {
            SessionOrderBy::UpdatedAtDesc => " ORDER BY s.updated_at DESC",
            SessionOrderBy::UpdatedAtAsc => " ORDER BY s.updated_at ASC",
            SessionOrderBy::CreatedAtDesc => " ORDER BY s.created_at DESC",
            SessionOrderBy::CreatedAtAsc => " ORDER BY s.created_at ASC",
            SessionOrderBy::MessageCountDesc => " ORDER BY s.message_count DESC",
        });

        if let Some(limit) = query.limit {
            sql.push_str(&format!(" LIMIT {}", limit));
        }
        if let Some(offset) = query.offset {
            sql.push_str(&format!(" OFFSET {}", offset));
        }

        let mut q = sqlx::query_as::<
            _,
            (
                String,
                String,
                String,
                String,
                i64,
                Option<String>,
                Option<String>,
            ),
        >(&sql);

        if let Some(ref agent_id) = query.agent_id {
            q = q.bind(agent_id);
        }
        if let Some(ref state) = query.state {
            q = q.bind(state);
        }
        if let Some(ref tag) = query.tag {
            q = q.bind(tag);
        }
        if let Some(ref user_id) = query.user_id {
            q = q.bind(user_id);
        }
        if let Some(ref created_after) = query.created_after {
            q = q.bind(created_after.to_rfc3339());
        }
        if let Some(ref created_before) = query.created_before {
            q = q.bind(created_before.to_rfc3339());
        }
        if let Some(ref updated_after) = query.updated_after {
            q = q.bind(updated_after.to_rfc3339());
        }

        let rows = q
            .fetch_all(&self.pool)
            .await
            .map_err(|e| AgentError::Persistence(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(
                |(
                    session_id,
                    agent_id,
                    created_at,
                    updated_at,
                    message_count,
                    current_state,
                    metadata,
                )| {
                    Self::extract_session_info(
                        session_id,
                        agent_id,
                        created_at,
                        updated_at,
                        message_count,
                        current_state,
                        metadata,
                    )
                },
            )
            .collect())
    }

    pub async fn expire_sessions(&self, before: DateTime<Utc>) -> Result<usize> {
        let result = sqlx::query("DELETE FROM sessions WHERE updated_at < ?")
            .bind(before.to_rfc3339())
            .execute(&self.pool)
            .await
            .map_err(|e| AgentError::Persistence(e.to_string()))?;

        Ok(result.rows_affected() as usize)
    }

    pub async fn exists(&self, session_id: &str) -> Result<bool> {
        let row: Option<(i64,)> =
            sqlx::query_as("SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1")
                .bind(session_id)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| AgentError::Persistence(e.to_string()))?;

        Ok(row.is_some())
    }

    pub async fn get_session_info(&self, session_id: &str) -> Result<Option<SessionInfo>> {
        let row: Option<(
            String,
            String,
            String,
            String,
            i64,
            Option<String>,
            Option<String>,
        )> = sqlx::query_as(
            r#"
            SELECT session_id, agent_id, created_at, updated_at, message_count, current_state, metadata
            FROM sessions WHERE session_id = ?
            "#,
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| AgentError::Persistence(e.to_string()))?;

        Ok(row.map(
            |(
                session_id,
                agent_id,
                created_at,
                updated_at,
                message_count,
                current_state,
                metadata,
            )| {
                Self::extract_session_info(
                    session_id,
                    agent_id,
                    created_at,
                    updated_at,
                    message_count,
                    current_state,
                    metadata,
                )
            },
        ))
    }
}

#[cfg(feature = "sqlite")]
#[async_trait]
impl AgentStorage for SqliteStorage {
    async fn save(&self, session_id: &str, snapshot: &AgentSnapshot) -> Result<()> {
        self.save_with_metadata(session_id, snapshot, &SessionMetadata::default())
            .await
    }

    async fn load(&self, session_id: &str) -> Result<Option<AgentSnapshot>> {
        let row: Option<(String,)> =
            sqlx::query_as("SELECT data FROM sessions WHERE session_id = ?")
                .bind(session_id)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| AgentError::Persistence(e.to_string()))?;

        match row {
            Some((data,)) => {
                let snapshot = serde_json::from_str(&data)
                    .map_err(|e| AgentError::Persistence(e.to_string()))?;
                Ok(Some(snapshot))
            }
            None => Ok(None),
        }
    }

    async fn delete(&self, session_id: &str) -> Result<()> {
        sqlx::query("DELETE FROM sessions WHERE session_id = ?")
            .bind(session_id)
            .execute(&self.pool)
            .await
            .map_err(|e| AgentError::Persistence(e.to_string()))?;
        Ok(())
    }

    async fn list_sessions(&self) -> Result<Vec<String>> {
        let rows: Vec<(String,)> =
            sqlx::query_as("SELECT session_id FROM sessions ORDER BY updated_at DESC")
                .fetch_all(&self.pool)
                .await
                .map_err(|e| AgentError::Persistence(e.to_string()))?;

        Ok(rows.into_iter().map(|(id,)| id).collect())
    }
}

#[cfg(all(test, feature = "sqlite"))]
mod tests {
    use super::*;
    use crate::llm::{ChatMessage, Role};
    use crate::persistence::MemorySnapshot;

    async fn create_test_snapshot() -> AgentSnapshot {
        let mut snapshot = AgentSnapshot::new("test-agent".into());
        snapshot.memory = MemorySnapshot::new(vec![
            ChatMessage {
                role: Role::User,
                content: "Hello".to_string(),
                name: None,
                timestamp: None,
            },
            ChatMessage {
                role: Role::Assistant,
                content: "Hi there!".to_string(),
                name: None,
                timestamp: None,
            },
        ]);
        snapshot
    }

    #[tokio::test]
    async fn test_sqlite_crud() {
        let storage = SqliteStorage::in_memory().await.unwrap();
        let snapshot = create_test_snapshot().await;

        storage.save("session-1", &snapshot).await.unwrap();

        let loaded = storage.load("session-1").await.unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().agent_id, "test-agent");

        storage.delete("session-1").await.unwrap();
        let loaded = storage.load("session-1").await.unwrap();
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn test_sqlite_list_sessions() {
        let storage = SqliteStorage::in_memory().await.unwrap();

        storage
            .save("session-1", &create_test_snapshot().await)
            .await
            .unwrap();
        storage
            .save("session-2", &create_test_snapshot().await)
            .await
            .unwrap();

        let sessions = storage.list_sessions().await.unwrap();
        assert_eq!(sessions.len(), 2);
    }

    #[tokio::test]
    async fn test_sqlite_with_metadata() {
        let storage = SqliteStorage::in_memory().await.unwrap();
        let snapshot = create_test_snapshot().await;

        let metadata = SessionMetadata {
            tags: vec!["vip".to_string(), "support".to_string()],
            user_id: Some("user-123".to_string()),
            ..Default::default()
        };

        storage
            .save_with_metadata("session-1", &snapshot, &metadata)
            .await
            .unwrap();

        let loaded_metadata = storage.get_metadata("session-1").await.unwrap();
        assert!(loaded_metadata.is_some());
        let loaded_metadata = loaded_metadata.unwrap();
        assert_eq!(loaded_metadata.tags.len(), 2);
        assert_eq!(loaded_metadata.user_id, Some("user-123".to_string()));
    }

    #[tokio::test]
    async fn test_sqlite_list_by_agent() {
        let storage = SqliteStorage::in_memory().await.unwrap();

        let mut snapshot1 = create_test_snapshot().await;
        snapshot1.agent_id = "agent-A".to_string();

        let mut snapshot2 = create_test_snapshot().await;
        snapshot2.agent_id = "agent-B".to_string();

        storage.save("session-1", &snapshot1).await.unwrap();
        storage.save("session-2", &snapshot2).await.unwrap();
        storage.save("session-3", &snapshot1).await.unwrap();

        let sessions = storage.list_sessions_by_agent("agent-A").await.unwrap();
        assert_eq!(sessions.len(), 2);
    }

    #[tokio::test]
    async fn test_sqlite_search() {
        let storage = SqliteStorage::in_memory().await.unwrap();
        let snapshot = create_test_snapshot().await;

        let metadata = SessionMetadata {
            tags: vec!["vip".to_string()],
            user_id: Some("user-123".to_string()),
            ..Default::default()
        };

        storage
            .save_with_metadata("session-1", &snapshot, &metadata)
            .await
            .unwrap();

        let query = SessionQuery {
            tag: Some("vip".to_string()),
            ..Default::default()
        };

        let results = storage.search_sessions(&query).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_sqlite_exists() {
        let storage = SqliteStorage::in_memory().await.unwrap();

        assert!(!storage.exists("session-1").await.unwrap());

        storage
            .save("session-1", &create_test_snapshot().await)
            .await
            .unwrap();

        assert!(storage.exists("session-1").await.unwrap());
    }

    #[tokio::test]
    async fn test_sqlite_expire() {
        let storage = SqliteStorage::in_memory().await.unwrap();

        storage
            .save("session-1", &create_test_snapshot().await)
            .await
            .unwrap();

        let future = Utc::now() + chrono::Duration::hours(1);
        let expired = storage.expire_sessions(future).await.unwrap();
        assert_eq!(expired, 1);

        let sessions = storage.list_sessions().await.unwrap();
        assert!(sessions.is_empty());
    }

    #[tokio::test]
    async fn test_sqlite_get_session_info() {
        let storage = SqliteStorage::in_memory().await.unwrap();
        let snapshot = create_test_snapshot().await;

        storage.save("session-1", &snapshot).await.unwrap();

        let info = storage.get_session_info("session-1").await.unwrap();
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.session_id, "session-1");
        assert_eq!(info.agent_id, "test-agent");
        assert_eq!(info.message_count, 2);

        let not_found = storage.get_session_info("nonexistent").await.unwrap();
        assert!(not_found.is_none());
    }
}

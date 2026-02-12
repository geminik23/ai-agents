//! Storage trait for agent persistence

use async_trait::async_trait;

use crate::error::Result;
use crate::types::StateMachineSnapshot;

/// Snapshot of agent state for persistence
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentSnapshot {
    pub version: String,
    pub agent_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    #[serde(default)]
    pub state_machine: Option<StateMachineSnapshot>,
    pub memory: super::memory::MemorySnapshot,
    #[serde(default)]
    pub context: std::collections::HashMap<String, serde_json::Value>,
}

impl AgentSnapshot {
    pub fn new(agent_id: String) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            agent_id,
            timestamp: chrono::Utc::now(),
            state_machine: None,
            memory: super::memory::MemorySnapshot::default(),
            context: std::collections::HashMap::new(),
        }
    }

    pub fn with_state_machine(mut self, snapshot: StateMachineSnapshot) -> Self {
        self.state_machine = Some(snapshot);
        self
    }

    pub fn with_memory(mut self, snapshot: super::memory::MemorySnapshot) -> Self {
        self.memory = snapshot;
        self
    }

    pub fn with_context(
        mut self,
        context: std::collections::HashMap<String, serde_json::Value>,
    ) -> Self {
        self.context = context;
        self
    }
}

/// Core storage trait for persisting agent state.
///
/// Built-in backends: `FileStorage`, `SqliteStorage`, and `RedisStorage`.
/// Implement this for custom persistence (e.g., PostgreSQL, DynamoDB).
#[async_trait]
pub trait AgentStorage: Send + Sync {
    /// Persist an agent snapshot for the given session ID.
    async fn save(&self, session_id: &str, snapshot: &AgentSnapshot) -> Result<()>;
    /// Load a snapshot. Returns `None` if the session does not exist.
    async fn load(&self, session_id: &str) -> Result<Option<AgentSnapshot>>;
    /// Remove a session's persisted data.
    async fn delete(&self, session_id: &str) -> Result<()>;
    /// List all stored session IDs.
    async fn list_sessions(&self) -> Result<Vec<String>>;
}

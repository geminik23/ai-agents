//! Storage trait for agent persistence

use async_trait::async_trait;

use crate::error::{AgentError, Result};
use crate::types::{
    FactFilter, KeyFact, SessionFilter, SessionMetadata, SessionSummary, StateMachineSnapshot,
};

/// Minimal record of a spawned agent for session persistence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpawnedAgentEntry {
    pub id: String,
    pub name: String,
    pub spec_yaml: String,
}

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spawned_agents: Option<Vec<SpawnedAgentEntry>>,
    /// Persona snapshot (serialized as Value to avoid core->persona dependency).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persona: Option<serde_json::Value>,
    /// Relationship snapshot (serialized as Value to avoid core->relationships dependency).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relationships: Option<serde_json::Value>,
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
            spawned_agents: None,
            persona: None,
            relationships: None,
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

    pub fn with_spawned_agents(mut self, agents: Vec<SpawnedAgentEntry>) -> Self {
        self.spawned_agents = Some(agents);
        self
    }
}

/// An optional storage feature that a backend supports.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageCapability {
    Snapshot,
    SessionMetadata,
    SessionFiltering,
    ExpiryCleanup,
    ActorFacts,
    ActorRelationships,
    ActorDataDeletion,
}

impl std::fmt::Display for StorageCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Snapshot => f.write_str("snapshot"),
            Self::SessionMetadata => f.write_str("session metadata"),
            Self::SessionFiltering => f.write_str("session filtering"),
            Self::ExpiryCleanup => f.write_str("expiry cleanup"),
            Self::ActorFacts => f.write_str("actor facts"),
            Self::ActorRelationships => f.write_str("actor relationships"),
            Self::ActorDataDeletion => f.write_str("atomic actor data deletion"),
        }
    }
}

/// Core storage trait for persisting agent state.
///
/// Built-in backends: `FileStorage`, `SqliteStorage`, and `RedisStorage`.
/// Implement this for custom persistence (e.g., PostgreSQL, DynamoDB).
#[async_trait]
pub trait AgentStorage: Send + Sync {
    /// Return whether this backend implements a storage feature.
    fn supports(&self, capability: StorageCapability) -> bool {
        capability == StorageCapability::Snapshot
    }

    /// Persist an agent snapshot for the given session ID.
    async fn save(&self, session_id: &str, snapshot: &AgentSnapshot) -> Result<()>;
    /// Load a snapshot. Returns `None` if the session does not exist.
    async fn load(&self, session_id: &str) -> Result<Option<AgentSnapshot>>;
    /// Remove a session's persisted data.
    async fn delete(&self, session_id: &str) -> Result<()>;
    /// List all stored session IDs.
    async fn list_sessions(&self) -> Result<Vec<String>>;

    // --- Session metadata ---

    /// Atomically persist a snapshot and its metadata when the backend supports session metadata.
    async fn save_snapshot_with_metadata(
        &self,
        session_id: &str,
        snapshot: &AgentSnapshot,
        metadata: &SessionMetadata,
    ) -> Result<()> {
        let _ = (session_id, snapshot, metadata);
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::SessionMetadata,
        ))
    }

    /// Save session metadata alongside the snapshot.
    async fn save_metadata(&self, _session_id: &str, _metadata: &SessionMetadata) -> Result<()> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::SessionMetadata,
        ))
    }

    /// Load session metadata.
    async fn load_metadata(&self, _session_id: &str) -> Result<Option<SessionMetadata>> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::SessionMetadata,
        ))
    }

    /// List sessions matching a filter.
    async fn list_sessions_filtered(&self, _filter: &SessionFilter) -> Result<Vec<SessionSummary>> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::SessionFiltering,
        ))
    }

    /// Delete sessions that have expired based on TTL.
    async fn cleanup_expired(&self) -> Result<usize> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::ExpiryCleanup,
        ))
    }

    // --- Actor facts ---

    /// Persist facts for an actor. Merges with existing facts.
    async fn save_facts(&self, _agent_id: &str, _actor_id: &str, _facts: &[KeyFact]) -> Result<()> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::ActorFacts,
        ))
    }

    /// Load all facts for a specific actor.
    async fn load_facts(&self, _agent_id: &str, _actor_id: &str) -> Result<Vec<KeyFact>> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::ActorFacts,
        ))
    }

    /// Load facts matching a filter (cross-actor queries).
    async fn query_facts(&self, _agent_id: &str, _filter: &FactFilter) -> Result<Vec<KeyFact>> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::ActorFacts,
        ))
    }

    /// Delete a single fact by ID.
    async fn delete_fact(&self, _agent_id: &str, _actor_id: &str, _fact_id: &str) -> Result<()> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::ActorFacts,
        ))
    }

    /// Atomically delete facts, relationships, and sessions owned by an actor.
    async fn delete_actor_data(&self, _agent_id: &str, _actor_id: &str) -> Result<()> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::ActorDataDeletion,
        ))
    }

    /// Persist relationship data for an actor. Value is owned by the relationships crate.
    async fn save_relationship(
        &self,
        _agent_id: &str,
        _actor_id: &str,
        _relationship: &serde_json::Value,
    ) -> Result<()> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::ActorRelationships,
        ))
    }

    /// Load relationship data for a specific actor.
    async fn load_relationship(
        &self,
        _agent_id: &str,
        _actor_id: &str,
    ) -> Result<Option<serde_json::Value>> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::ActorRelationships,
        ))
    }

    /// List actor IDs that have relationship data for an agent.
    async fn list_relationship_actors(&self, _agent_id: &str) -> Result<Vec<String>> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::ActorRelationships,
        ))
    }

    /// Delete relationship data for a specific actor.
    async fn delete_relationship(&self, _agent_id: &str, _actor_id: &str) -> Result<()> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::ActorRelationships,
        ))
    }
}

/// Storage backend for testing unsupported persistence behavior.
pub struct NoopStorage;

#[async_trait]
impl AgentStorage for NoopStorage {
    fn supports(&self, _capability: StorageCapability) -> bool {
        false
    }

    async fn save(&self, _session_id: &str, _snapshot: &AgentSnapshot) -> Result<()> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::Snapshot,
        ))
    }
    async fn load(&self, _session_id: &str) -> Result<Option<AgentSnapshot>> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::Snapshot,
        ))
    }
    async fn delete(&self, _session_id: &str) -> Result<()> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::Snapshot,
        ))
    }
    async fn list_sessions(&self) -> Result<Vec<String>> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::Snapshot,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SnapshotOnlyStorage;

    struct TargetedActorStorage {
        fact_deletes: std::sync::atomic::AtomicUsize,
        relationship_deletes: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl AgentStorage for SnapshotOnlyStorage {
        async fn save(&self, _session_id: &str, _snapshot: &AgentSnapshot) -> Result<()> {
            Ok(())
        }

        async fn load(&self, _session_id: &str) -> Result<Option<AgentSnapshot>> {
            Ok(None)
        }

        async fn delete(&self, _session_id: &str) -> Result<()> {
            Ok(())
        }

        async fn list_sessions(&self) -> Result<Vec<String>> {
            Ok(vec![])
        }
    }

    #[async_trait]
    impl AgentStorage for TargetedActorStorage {
        fn supports(&self, capability: StorageCapability) -> bool {
            matches!(
                capability,
                StorageCapability::Snapshot
                    | StorageCapability::ActorFacts
                    | StorageCapability::ActorRelationships
            )
        }

        async fn save(&self, _session_id: &str, _snapshot: &AgentSnapshot) -> Result<()> {
            Ok(())
        }

        async fn load(&self, _session_id: &str) -> Result<Option<AgentSnapshot>> {
            Ok(None)
        }

        async fn delete(&self, _session_id: &str) -> Result<()> {
            Ok(())
        }

        async fn list_sessions(&self) -> Result<Vec<String>> {
            Ok(Vec::new())
        }

        async fn delete_fact(
            &self,
            _agent_id: &str,
            _actor_id: &str,
            _fact_id: &str,
        ) -> Result<()> {
            self.fact_deletes
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }

        async fn delete_relationship(&self, _agent_id: &str, _actor_id: &str) -> Result<()> {
            self.relationship_deletes
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    fn assert_unsupported<T>(result: Result<T>, expected: StorageCapability) {
        assert!(matches!(
            result,
            Err(AgentError::UnsupportedStorageCapability(capability)) if capability == expected
        ));
    }

    #[test]
    fn snapshot_only_custom_backend_uses_default_capability() {
        let storage = SnapshotOnlyStorage;

        assert!(storage.supports(StorageCapability::Snapshot));
        assert!(!storage.supports(StorageCapability::SessionMetadata));
        futures::executor::block_on(storage.save("session", &AgentSnapshot::new("agent".into())))
            .unwrap();
    }

    #[test]
    fn custom_backend_extensions_return_typed_errors() {
        let storage = SnapshotOnlyStorage;
        let metadata = SessionMetadata::default();
        let session_filter = SessionFilter::default();
        let fact_filter = FactFilter::default();
        let relationship = serde_json::json!({});

        assert_unsupported(
            futures::executor::block_on(storage.save_snapshot_with_metadata(
                "session",
                &AgentSnapshot::new("agent".into()),
                &metadata,
            )),
            StorageCapability::SessionMetadata,
        );
        assert_unsupported(
            futures::executor::block_on(storage.save_metadata("session", &metadata)),
            StorageCapability::SessionMetadata,
        );
        assert_unsupported(
            futures::executor::block_on(storage.load_metadata("session")),
            StorageCapability::SessionMetadata,
        );
        assert_unsupported(
            futures::executor::block_on(storage.list_sessions_filtered(&session_filter)),
            StorageCapability::SessionFiltering,
        );
        assert_unsupported(
            futures::executor::block_on(storage.cleanup_expired()),
            StorageCapability::ExpiryCleanup,
        );
        assert_unsupported(
            futures::executor::block_on(storage.save_facts("agent", "actor", &[])),
            StorageCapability::ActorFacts,
        );
        assert_unsupported(
            futures::executor::block_on(storage.load_facts("agent", "actor")),
            StorageCapability::ActorFacts,
        );
        assert_unsupported(
            futures::executor::block_on(storage.query_facts("agent", &fact_filter)),
            StorageCapability::ActorFacts,
        );
        assert_unsupported(
            futures::executor::block_on(storage.delete_fact("agent", "actor", "fact")),
            StorageCapability::ActorFacts,
        );
        assert_unsupported(
            futures::executor::block_on(storage.delete_actor_data("agent", "actor")),
            StorageCapability::ActorDataDeletion,
        );
        assert_unsupported(
            futures::executor::block_on(storage.save_relationship("agent", "actor", &relationship)),
            StorageCapability::ActorRelationships,
        );
        assert_unsupported(
            futures::executor::block_on(storage.load_relationship("agent", "actor")),
            StorageCapability::ActorRelationships,
        );
        assert_unsupported(
            futures::executor::block_on(storage.list_relationship_actors("agent")),
            StorageCapability::ActorRelationships,
        );
        assert_unsupported(
            futures::executor::block_on(storage.delete_relationship("agent", "actor")),
            StorageCapability::ActorRelationships,
        );
    }

    #[test]
    fn unsupported_composite_actor_deletion_does_not_call_targeted_deletes() {
        let storage = TargetedActorStorage {
            fact_deletes: std::sync::atomic::AtomicUsize::new(0),
            relationship_deletes: std::sync::atomic::AtomicUsize::new(0),
        };

        assert_unsupported(
            futures::executor::block_on(storage.delete_actor_data("agent", "actor")),
            StorageCapability::ActorDataDeletion,
        );
        assert_eq!(
            storage
                .fact_deletes
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
        assert_eq!(
            storage
                .relationship_deletes
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
    }

    #[test]
    fn noop_storage_rejects_snapshot_operations() {
        let storage = NoopStorage;
        let snapshot = AgentSnapshot::new("agent".into());

        assert!(!storage.supports(StorageCapability::Snapshot));
        assert_unsupported(
            futures::executor::block_on(storage.save("session", &snapshot)),
            StorageCapability::Snapshot,
        );
        assert_unsupported(
            futures::executor::block_on(storage.load("session")),
            StorageCapability::Snapshot,
        );
        assert_unsupported(
            futures::executor::block_on(storage.delete("session")),
            StorageCapability::Snapshot,
        );
        assert_unsupported(
            futures::executor::block_on(storage.list_sessions()),
            StorageCapability::Snapshot,
        );
    }
}

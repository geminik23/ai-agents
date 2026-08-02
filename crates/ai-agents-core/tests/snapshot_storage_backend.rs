use ai_agents_core::{AgentSnapshot, AgentStorage, Result, StorageCapability};
use async_trait::async_trait;

struct SnapshotStorage;

#[async_trait]
impl AgentStorage for SnapshotStorage {
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
}

#[test]
fn external_snapshot_backend_uses_conservative_defaults() {
    let storage = SnapshotStorage;

    assert!(storage.supports(StorageCapability::Snapshot));
    assert!(!storage.supports(StorageCapability::SessionMetadata));
}

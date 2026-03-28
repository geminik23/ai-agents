//! Storage adapter that namespaces session keys with an agent-specific prefix.

use std::sync::Arc;

use async_trait::async_trait;

use ai_agents_core::{AgentSnapshot, AgentStorage, Result};

/// Wraps a shared `AgentStorage` backend and transparently prepends `{prefix}/` to every session key, isolating each agent's data.
pub struct NamespacedStorage {
    inner: Arc<dyn AgentStorage>,
    prefix: String,
}

impl NamespacedStorage {
    pub fn new(inner: Arc<dyn AgentStorage>, prefix: impl Into<String>) -> Self {
        Self {
            inner,
            prefix: prefix.into(),
        }
    }

    /// Build the namespaced key: `"{prefix}/{session_id}"`.
    fn namespaced_key(&self, session_id: &str) -> String {
        format!("{}/{}", self.prefix, session_id)
    }
}

#[async_trait]
impl AgentStorage for NamespacedStorage {
    async fn save(&self, session_id: &str, snapshot: &AgentSnapshot) -> Result<()> {
        self.inner
            .save(&self.namespaced_key(session_id), snapshot)
            .await
    }

    async fn load(&self, session_id: &str) -> Result<Option<AgentSnapshot>> {
        self.inner.load(&self.namespaced_key(session_id)).await
    }

    async fn delete(&self, session_id: &str) -> Result<()> {
        self.inner.delete(&self.namespaced_key(session_id)).await
    }

    async fn list_sessions(&self) -> Result<Vec<String>> {
        let all = self.inner.list_sessions().await?;
        let prefix_slash = format!("{}/", self.prefix);
        Ok(all
            .into_iter()
            .filter_map(|s| s.strip_prefix(&prefix_slash).map(|rest| rest.to_string()))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_agents_core::AgentSnapshot;
    use parking_lot::RwLock;
    use std::collections::HashMap;

    /// Minimal in-memory storage for testing.
    struct MemStorage {
        data: RwLock<HashMap<String, AgentSnapshot>>,
    }

    impl MemStorage {
        fn new() -> Self {
            Self {
                data: RwLock::new(HashMap::new()),
            }
        }
    }

    #[async_trait]
    impl AgentStorage for MemStorage {
        async fn save(&self, session_id: &str, snapshot: &AgentSnapshot) -> Result<()> {
            self.data
                .write()
                .insert(session_id.to_string(), snapshot.clone());
            Ok(())
        }

        async fn load(&self, session_id: &str) -> Result<Option<AgentSnapshot>> {
            Ok(self.data.read().get(session_id).cloned())
        }

        async fn delete(&self, session_id: &str) -> Result<()> {
            self.data.write().remove(session_id);
            Ok(())
        }

        async fn list_sessions(&self) -> Result<Vec<String>> {
            Ok(self.data.read().keys().cloned().collect())
        }
    }

    #[tokio::test]
    async fn test_namespaced_save_load() {
        let inner = Arc::new(MemStorage::new());
        let ns = NamespacedStorage::new(inner.clone(), "agent_1");

        let snapshot = AgentSnapshot::new("agent_1".to_string());
        ns.save("session_a", &snapshot).await.unwrap();

        // Underlying storage should have the prefixed key
        assert!(inner.load("agent_1/session_a").await.unwrap().is_some());

        // Namespaced load should work with the unprefixed key
        assert!(ns.load("session_a").await.unwrap().is_some());
        assert!(ns.load("session_b").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_namespaced_list_sessions() {
        let inner = Arc::new(MemStorage::new());

        let ns1 = NamespacedStorage::new(inner.clone(), "npc_a");
        let ns2 = NamespacedStorage::new(inner.clone(), "npc_b");

        ns1.save("s1", &AgentSnapshot::new("npc_a".into()))
            .await
            .unwrap();
        ns1.save("s2", &AgentSnapshot::new("npc_a".into()))
            .await
            .unwrap();
        ns2.save("s1", &AgentSnapshot::new("npc_b".into()))
            .await
            .unwrap();

        let mut sessions1 = ns1.list_sessions().await.unwrap();
        sessions1.sort();
        assert_eq!(sessions1, vec!["s1", "s2"]);

        let sessions2 = ns2.list_sessions().await.unwrap();
        assert_eq!(sessions2, vec!["s1"]);
    }

    #[tokio::test]
    async fn test_namespaced_delete() {
        let inner = Arc::new(MemStorage::new());
        let ns = NamespacedStorage::new(inner.clone(), "agent_x");

        ns.save("sess", &AgentSnapshot::new("agent_x".into()))
            .await
            .unwrap();
        assert!(ns.load("sess").await.unwrap().is_some());

        ns.delete("sess").await.unwrap();
        assert!(ns.load("sess").await.unwrap().is_none());
    }
}

//! Storage adapter that isolates agent data in a shared backend.

use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;

use ai_agents_core::traits::storage::StorageCapability;
use ai_agents_core::{
    AgentError, AgentSnapshot, AgentStorage, FactFilter, KeyFact, Result, SessionFilter,
    SessionMetadata, SessionSummary,
};

const FORWARDED_CAPABILITIES: [StorageCapability; 6] = [
    StorageCapability::Snapshot,
    StorageCapability::SessionMetadata,
    StorageCapability::SessionFiltering,
    StorageCapability::ActorFacts,
    StorageCapability::ActorRelationships,
    StorageCapability::ActorDataDeletion,
];

/// Wraps shared storage with reversible flat keys scoped to one agent namespace.
pub struct NamespacedStorage {
    inner: Arc<dyn AgentStorage>,
    namespace: String,
    encoded_namespace: String,
}

impl NamespacedStorage {
    pub fn new(inner: Arc<dyn AgentStorage>, namespace: impl Into<String>) -> Self {
        let namespace = namespace.into();
        Self {
            inner,
            encoded_namespace: encode_component(&namespace),
            namespace,
        }
    }

    fn require(&self, capability: StorageCapability) -> Result<()> {
        if self.supports(capability) {
            Ok(())
        } else {
            Err(AgentError::UnsupportedStorageCapability(capability))
        }
    }

    fn encode_key(&self, kind: &str, value: &str) -> String {
        // Hex encoding keeps every generated key flat ASCII without exposing path separators.
        format!(
            "a7ns1_{}_{}_{}",
            kind,
            self.encoded_namespace,
            encode_component(value)
        )
    }

    fn decode_key(&self, kind: &str, value: &str) -> Option<String> {
        let marker = format!("a7ns1_{}_{}_", kind, self.encoded_namespace);
        value.strip_prefix(&marker).and_then(decode_component)
    }

    fn session_key(&self, session_id: &str) -> String {
        self.encode_key("session", session_id)
    }

    fn legacy_session_key(&self, session_id: &str) -> String {
        format!("{}/{}", self.namespace, session_id)
    }

    fn decode_legacy_session_key(&self, value: &str) -> Option<String> {
        value
            .strip_prefix(&format!("{}/", self.namespace))
            .map(str::to_owned)
    }

    fn decode_session_key(&self, value: &str) -> Result<Option<String>> {
        let marker = format!("a7ns1_session_{}_", self.encoded_namespace);
        if let Some(encoded) = value.strip_prefix(&marker) {
            return decode_component(encoded).map(Some).ok_or_else(|| {
                AgentError::Persistence(
                    "Storage returned a malformed session key for this namespace".into(),
                )
            });
        }
        Ok(self.decode_legacy_session_key(value))
    }

    fn agent_key(&self, agent_id: &str) -> String {
        self.encode_key("agent", agent_id)
    }

    fn actor_key(&self, actor_id: &str) -> String {
        self.encode_key("actor", actor_id)
    }

    fn decode_agent(&self, agent_id: &str) -> Result<String> {
        self.decode_key("agent", agent_id).ok_or_else(|| {
            AgentError::Persistence("Storage returned an agent outside this namespace".into())
        })
    }

    fn decode_actor(&self, actor_id: &str) -> Result<String> {
        self.decode_key("actor", actor_id).ok_or_else(|| {
            AgentError::Persistence("Storage returned an actor outside this namespace".into())
        })
    }

    fn encode_snapshot(&self, snapshot: &AgentSnapshot) -> AgentSnapshot {
        let mut snapshot = snapshot.clone();
        snapshot.agent_id = self.agent_key(&snapshot.agent_id);
        snapshot
    }

    fn decode_snapshot(&self, mut snapshot: AgentSnapshot) -> Result<AgentSnapshot> {
        snapshot.agent_id = self.decode_agent(&snapshot.agent_id)?;
        Ok(snapshot)
    }

    fn encode_metadata(&self, metadata: &SessionMetadata) -> SessionMetadata {
        let mut metadata = metadata.clone();
        metadata.actor_id = metadata.actor_id.map(|actor| self.actor_key(&actor));
        metadata.actors = metadata
            .actors
            .into_iter()
            .map(|actor| self.actor_key(&actor))
            .collect();
        metadata
    }

    fn decode_metadata(&self, mut metadata: SessionMetadata) -> Result<SessionMetadata> {
        metadata.actor_id = metadata
            .actor_id
            .map(|actor| self.decode_actor(&actor))
            .transpose()?;
        metadata.actors = metadata
            .actors
            .into_iter()
            .map(|actor| self.decode_actor(&actor))
            .collect::<Result<Vec<_>>>()?;
        Ok(metadata)
    }

    fn encode_fact(&self, fact: &KeyFact) -> KeyFact {
        let mut fact = fact.clone();
        fact.actor_id = fact.actor_id.map(|actor| self.actor_key(&actor));
        fact
    }

    fn decode_fact(&self, mut fact: KeyFact) -> Option<KeyFact> {
        if let Some(actor_id) = fact.actor_id.take() {
            fact.actor_id = Some(self.decode_key("actor", &actor_id)?);
        }
        Some(fact)
    }

    fn backend_session_filter(&self, filter: &SessionFilter) -> SessionFilter {
        let mut filter = filter.clone();
        filter.actor_id = None;
        filter.agent_id = None;
        filter.limit = None;
        filter
    }

    fn decode_session_summary(
        &self,
        mut summary: SessionSummary,
    ) -> Result<Option<SessionSummary>> {
        let raw_session = summary.session_id.clone();
        let Some(session_id) = self.decode_session_key(&raw_session)? else {
            return Ok(None);
        };
        summary.session_id = session_id;

        let current_marker = format!("a7ns1_session_{}_", self.encoded_namespace);
        if raw_session.starts_with(&current_marker) {
            summary.agent_id = self.decode_agent(&summary.agent_id)?;
            if let Some(actor_id) = summary.actor_id.take() {
                summary.actor_id = Some(self.decode_actor(&actor_id)?);
            }
        }
        Ok(Some(summary))
    }

    fn summary_matches_identity_filter(summary: &SessionSummary, filter: &SessionFilter) -> bool {
        filter
            .agent_id
            .as_ref()
            .is_none_or(|agent| &summary.agent_id == agent)
            && filter
                .actor_id
                .as_ref()
                .is_none_or(|actor| summary.actor_id.as_ref() == Some(actor))
    }
}

fn encode_component(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(value.len() * 2);
    for byte in value.as_bytes() {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn decode_component(value: &str) -> Option<String> {
    if !value.len().is_multiple_of(2) {
        return None;
    }
    let mut decoded = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        decoded.push((decode_nibble(pair[0])? << 4) | decode_nibble(pair[1])?);
    }
    String::from_utf8(decoded).ok()
}

fn decode_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

#[async_trait]
impl AgentStorage for NamespacedStorage {
    fn supports(&self, capability: StorageCapability) -> bool {
        FORWARDED_CAPABILITIES.contains(&capability) && self.inner.supports(capability)
    }

    async fn save(&self, session_id: &str, snapshot: &AgentSnapshot) -> Result<()> {
        self.require(StorageCapability::Snapshot)?;
        self.inner
            .save(
                &self.session_key(session_id),
                &self.encode_snapshot(snapshot),
            )
            .await?;
        self.inner
            .delete(&self.legacy_session_key(session_id))
            .await
    }

    async fn load(&self, session_id: &str) -> Result<Option<AgentSnapshot>> {
        self.require(StorageCapability::Snapshot)?;
        if let Some(snapshot) = self.inner.load(&self.session_key(session_id)).await? {
            return self.decode_snapshot(snapshot).map(Some);
        }

        //
        // Legacy reads remain side-effect free. A later save writes the new key and removes the legacy key.
        //
        self.inner.load(&self.legacy_session_key(session_id)).await
    }

    async fn delete(&self, session_id: &str) -> Result<()> {
        self.require(StorageCapability::Snapshot)?;
        self.inner.delete(&self.session_key(session_id)).await?;
        self.inner
            .delete(&self.legacy_session_key(session_id))
            .await
    }

    async fn list_sessions(&self) -> Result<Vec<String>> {
        self.require(StorageCapability::Snapshot)?;
        let mut sessions = BTreeSet::new();
        for session in self.inner.list_sessions().await? {
            if let Some(session) = self.decode_session_key(&session)? {
                sessions.insert(session);
            }
        }
        Ok(sessions.into_iter().collect())
    }

    async fn save_snapshot_with_metadata(
        &self,
        session_id: &str,
        snapshot: &AgentSnapshot,
        metadata: &SessionMetadata,
    ) -> Result<()> {
        self.require(StorageCapability::SessionMetadata)?;
        self.inner
            .save_snapshot_with_metadata(
                &self.session_key(session_id),
                &self.encode_snapshot(snapshot),
                &self.encode_metadata(metadata),
            )
            .await?;
        self.inner
            .delete(&self.legacy_session_key(session_id))
            .await
    }

    async fn save_metadata(&self, session_id: &str, metadata: &SessionMetadata) -> Result<()> {
        self.require(StorageCapability::SessionMetadata)?;
        self.inner
            .save_metadata(
                &self.session_key(session_id),
                &self.encode_metadata(metadata),
            )
            .await
    }

    async fn load_metadata(&self, session_id: &str) -> Result<Option<SessionMetadata>> {
        self.require(StorageCapability::SessionMetadata)?;
        self.inner
            .load_metadata(&self.session_key(session_id))
            .await?
            .map(|metadata| self.decode_metadata(metadata))
            .transpose()
    }

    async fn list_sessions_filtered(&self, filter: &SessionFilter) -> Result<Vec<SessionSummary>> {
        self.require(StorageCapability::SessionFiltering)?;
        let mut summaries = Vec::new();
        for summary in self
            .inner
            .list_sessions_filtered(&self.backend_session_filter(filter))
            .await?
        {
            if let Some(summary) = self.decode_session_summary(summary)?
                && Self::summary_matches_identity_filter(&summary, filter)
            {
                summaries.push(summary);
            }
        }
        if let Some(limit) = filter.limit {
            summaries.truncate(limit);
        }
        Ok(summaries)
    }

    async fn cleanup_expired(&self) -> Result<usize> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::ExpiryCleanup,
        ))
    }

    async fn save_facts(&self, agent_id: &str, actor_id: &str, facts: &[KeyFact]) -> Result<()> {
        self.require(StorageCapability::ActorFacts)?;
        let facts = facts
            .iter()
            .map(|fact| self.encode_fact(fact))
            .collect::<Vec<_>>();
        self.inner
            .save_facts(&self.agent_key(agent_id), &self.actor_key(actor_id), &facts)
            .await
    }

    async fn load_facts(&self, agent_id: &str, actor_id: &str) -> Result<Vec<KeyFact>> {
        self.require(StorageCapability::ActorFacts)?;
        Ok(self
            .inner
            .load_facts(&self.agent_key(agent_id), &self.actor_key(actor_id))
            .await?
            .into_iter()
            .filter_map(|fact| self.decode_fact(fact))
            .collect())
    }

    async fn query_facts(&self, agent_id: &str, filter: &FactFilter) -> Result<Vec<KeyFact>> {
        self.require(StorageCapability::ActorFacts)?;
        let mut filter = filter.clone();
        filter.actor_id = filter.actor_id.map(|actor| self.actor_key(&actor));
        Ok(self
            .inner
            .query_facts(&self.agent_key(agent_id), &filter)
            .await?
            .into_iter()
            .filter_map(|fact| self.decode_fact(fact))
            .collect())
    }

    async fn delete_fact(&self, agent_id: &str, actor_id: &str, fact_id: &str) -> Result<()> {
        self.require(StorageCapability::ActorFacts)?;
        self.inner
            .delete_fact(
                &self.agent_key(agent_id),
                &self.actor_key(actor_id),
                fact_id,
            )
            .await
    }

    async fn delete_actor_data(&self, agent_id: &str, actor_id: &str) -> Result<()> {
        self.require(StorageCapability::ActorDataDeletion)?;
        self.inner
            .delete_actor_data(&self.agent_key(agent_id), &self.actor_key(actor_id))
            .await
    }

    async fn save_relationship(
        &self,
        agent_id: &str,
        actor_id: &str,
        relationship: &serde_json::Value,
    ) -> Result<()> {
        self.require(StorageCapability::ActorRelationships)?;
        self.inner
            .save_relationship(
                &self.agent_key(agent_id),
                &self.actor_key(actor_id),
                relationship,
            )
            .await
    }

    async fn load_relationship(
        &self,
        agent_id: &str,
        actor_id: &str,
    ) -> Result<Option<serde_json::Value>> {
        self.require(StorageCapability::ActorRelationships)?;
        self.inner
            .load_relationship(&self.agent_key(agent_id), &self.actor_key(actor_id))
            .await
    }

    async fn list_relationship_actors(&self, agent_id: &str) -> Result<Vec<String>> {
        self.require(StorageCapability::ActorRelationships)?;
        Ok(self
            .inner
            .list_relationship_actors(&self.agent_key(agent_id))
            .await?
            .into_iter()
            .filter_map(|actor| self.decode_key("actor", &actor))
            .collect())
    }

    async fn delete_relationship(&self, agent_id: &str, actor_id: &str) -> Result<()> {
        self.require(StorageCapability::ActorRelationships)?;
        self.inner
            .delete_relationship(&self.agent_key(agent_id), &self.actor_key(actor_id))
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use ai_agents_core::FactCategory;
    use chrono::Utc;
    use parking_lot::RwLock;

    use super::*;

    const ALL_CAPABILITIES: &[StorageCapability] = &[
        StorageCapability::Snapshot,
        StorageCapability::SessionMetadata,
        StorageCapability::SessionFiltering,
        StorageCapability::ExpiryCleanup,
        StorageCapability::ActorFacts,
        StorageCapability::ActorRelationships,
        StorageCapability::ActorDataDeletion,
    ];

    #[derive(Default)]
    struct MemStorage {
        data: RwLock<HashMap<String, AgentSnapshot>>,
        metadata: RwLock<HashMap<String, SessionMetadata>>,
        facts: RwLock<HashMap<(String, String), Vec<KeyFact>>>,
        relationships: RwLock<HashMap<(String, String), serde_json::Value>>,
        last_session_filter: RwLock<Option<SessionFilter>>,
        expiry_calls: AtomicUsize,
        full_capabilities: bool,
    }

    impl MemStorage {
        fn full() -> Self {
            Self {
                full_capabilities: true,
                ..Self::default()
            }
        }
    }

    #[async_trait]
    impl AgentStorage for MemStorage {
        fn supports(&self, capability: StorageCapability) -> bool {
            if self.full_capabilities {
                ALL_CAPABILITIES.contains(&capability)
            } else {
                capability == StorageCapability::Snapshot
            }
        }

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

        async fn save_snapshot_with_metadata(
            &self,
            session_id: &str,
            snapshot: &AgentSnapshot,
            metadata: &SessionMetadata,
        ) -> Result<()> {
            self.save(session_id, snapshot).await?;
            self.save_metadata(session_id, metadata).await
        }

        async fn save_metadata(&self, session_id: &str, metadata: &SessionMetadata) -> Result<()> {
            self.metadata
                .write()
                .insert(session_id.to_string(), metadata.clone());
            Ok(())
        }

        async fn load_metadata(&self, session_id: &str) -> Result<Option<SessionMetadata>> {
            Ok(self.metadata.read().get(session_id).cloned())
        }

        async fn list_sessions_filtered(
            &self,
            filter: &SessionFilter,
        ) -> Result<Vec<SessionSummary>> {
            *self.last_session_filter.write() = Some(filter.clone());
            let metadata = self.metadata.read();
            let mut summaries = self
                .data
                .read()
                .iter()
                .filter_map(|(session_id, snapshot)| {
                    let meta = metadata.get(session_id)?;
                    if filter
                        .actor_id
                        .as_ref()
                        .is_some_and(|actor| meta.actor_id.as_ref() != Some(actor))
                        || filter
                            .agent_id
                            .as_ref()
                            .is_some_and(|agent| &snapshot.agent_id != agent)
                    {
                        return None;
                    }
                    Some(SessionSummary {
                        session_id: session_id.clone(),
                        agent_id: snapshot.agent_id.clone(),
                        actor_id: meta.actor_id.clone(),
                        tags: meta.tags.clone(),
                        created_at: meta.created_at,
                        last_active: meta.last_active,
                        message_count: meta.message_count,
                    })
                })
                .collect::<Vec<_>>();
            summaries.sort_by_key(|summary| std::cmp::Reverse(summary.last_active));
            if let Some(limit) = filter.limit {
                summaries.truncate(limit);
            }
            Ok(summaries)
        }

        async fn cleanup_expired(&self) -> Result<usize> {
            self.expiry_calls.fetch_add(1, Ordering::SeqCst);
            Ok(0)
        }

        async fn save_facts(
            &self,
            agent_id: &str,
            actor_id: &str,
            facts: &[KeyFact],
        ) -> Result<()> {
            self.facts
                .write()
                .insert((agent_id.to_string(), actor_id.to_string()), facts.to_vec());
            Ok(())
        }

        async fn load_facts(&self, agent_id: &str, actor_id: &str) -> Result<Vec<KeyFact>> {
            Ok(self
                .facts
                .read()
                .get(&(agent_id.to_string(), actor_id.to_string()))
                .cloned()
                .unwrap_or_default())
        }

        async fn query_facts(&self, agent_id: &str, filter: &FactFilter) -> Result<Vec<KeyFact>> {
            Ok(self
                .facts
                .read()
                .iter()
                .filter(|((stored_agent, stored_actor), _)| {
                    stored_agent == agent_id
                        && filter
                            .actor_id
                            .as_ref()
                            .is_none_or(|actor| stored_actor == actor)
                })
                .flat_map(|(_, facts)| facts.clone())
                .collect())
        }

        async fn delete_fact(&self, agent_id: &str, actor_id: &str, fact_id: &str) -> Result<()> {
            if let Some(facts) = self
                .facts
                .write()
                .get_mut(&(agent_id.to_string(), actor_id.to_string()))
            {
                facts.retain(|fact| fact.id != fact_id);
            }
            Ok(())
        }

        async fn delete_actor_data(&self, agent_id: &str, actor_id: &str) -> Result<()> {
            self.facts
                .write()
                .remove(&(agent_id.to_string(), actor_id.to_string()));
            Ok(())
        }

        async fn save_relationship(
            &self,
            agent_id: &str,
            actor_id: &str,
            relationship: &serde_json::Value,
        ) -> Result<()> {
            self.relationships.write().insert(
                (agent_id.to_string(), actor_id.to_string()),
                relationship.clone(),
            );
            Ok(())
        }

        async fn load_relationship(
            &self,
            agent_id: &str,
            actor_id: &str,
        ) -> Result<Option<serde_json::Value>> {
            Ok(self
                .relationships
                .read()
                .get(&(agent_id.to_string(), actor_id.to_string()))
                .cloned())
        }

        async fn list_relationship_actors(&self, agent_id: &str) -> Result<Vec<String>> {
            Ok(self
                .relationships
                .read()
                .keys()
                .filter(|(stored_agent, _)| stored_agent == agent_id)
                .map(|(_, actor)| actor.clone())
                .collect())
        }

        async fn delete_relationship(&self, agent_id: &str, actor_id: &str) -> Result<()> {
            self.relationships
                .write()
                .remove(&(agent_id.to_string(), actor_id.to_string()));
            Ok(())
        }
    }

    fn fact(actor_id: &str) -> KeyFact {
        KeyFact {
            id: "fact-1".into(),
            actor_id: Some(actor_id.into()),
            category: FactCategory::UserContext,
            content: "context".into(),
            confidence: 1.0,
            salience: 1.0,
            extracted_at: Utc::now(),
            last_accessed: None,
            source_message_id: None,
            source_language: None,
        }
    }

    #[tokio::test]
    async fn parent_and_sibling_keys_are_isolated_and_arbitrary_ids_round_trip() {
        let inner = Arc::new(MemStorage::full());
        let first = NamespacedStorage::new(inner.clone(), "parent/child");
        let second = NamespacedStorage::new(inner.clone(), "parent");
        let first_session = "sibling/../session\0😀";
        let second_session = "child/sibling/../session\0😀";

        first
            .save(first_session, &AgentSnapshot::new("agent/α".into()))
            .await
            .unwrap();
        second
            .save(second_session, &AgentSnapshot::new("agent/β".into()))
            .await
            .unwrap();

        let keys = inner.data.read().keys().cloned().collect::<Vec<_>>();
        assert_eq!(keys.len(), 2);
        assert_ne!(keys[0], keys[1]);
        assert!(keys.iter().all(|key| key.is_ascii() && !key.contains('/')));
        assert_eq!(first.list_sessions().await.unwrap(), vec![first_session]);
        assert_eq!(second.list_sessions().await.unwrap(), vec![second_session]);
        assert!(first.load(second_session).await.unwrap().is_none());
        assert_eq!(
            first.load(first_session).await.unwrap().unwrap().agent_id,
            "agent/α"
        );
    }

    #[tokio::test]
    async fn capabilities_are_derived_without_expiry_forwarding() {
        let inner = Arc::new(MemStorage::full());
        let storage = NamespacedStorage::new(inner.clone(), "agent");

        for capability in FORWARDED_CAPABILITIES {
            assert!(storage.supports(capability));
        }
        assert!(!storage.supports(StorageCapability::ExpiryCleanup));
        assert!(matches!(
            storage.cleanup_expired().await,
            Err(AgentError::UnsupportedStorageCapability(
                StorageCapability::ExpiryCleanup
            ))
        ));
        assert_eq!(inner.expiry_calls.load(Ordering::SeqCst), 0);

        let snapshot_only = NamespacedStorage::new(Arc::new(MemStorage::default()), "agent");
        assert!(snapshot_only.supports(StorageCapability::Snapshot));
        assert!(!snapshot_only.supports(StorageCapability::SessionMetadata));
        assert!(matches!(
            snapshot_only
                .save_metadata("session", &SessionMetadata::default())
                .await,
            Err(AgentError::UnsupportedStorageCapability(
                StorageCapability::SessionMetadata
            ))
        ));
    }

    #[tokio::test]
    async fn session_agent_and_actor_keys_are_transformed_and_stripped() {
        let inner = Arc::new(MemStorage::full());
        let storage = NamespacedStorage::new(inner.clone(), "spawned/😀");
        let sibling = NamespacedStorage::new(inner.clone(), "sibling");
        let session_id = "session/零";
        let agent_id = "agent/零";
        let actor_id = "actor/零";
        let metadata = SessionMetadata {
            actor_id: Some(actor_id.into()),
            actors: vec![actor_id.into(), "other/😀".into()],
            ..SessionMetadata::default()
        };

        storage
            .save(session_id, &AgentSnapshot::new(agent_id.into()))
            .await
            .unwrap();
        storage.save_metadata(session_id, &metadata).await.unwrap();
        sibling
            .save(
                "session/sibling",
                &AgentSnapshot::new("agent/sibling".into()),
            )
            .await
            .unwrap();
        sibling
            .save_metadata("session/sibling", &SessionMetadata::default())
            .await
            .unwrap();

        let raw_session = storage.session_key(session_id);
        let raw_snapshot = inner.data.read().get(&raw_session).cloned().unwrap();
        let raw_metadata = inner.metadata.read().get(&raw_session).cloned().unwrap();
        assert!(raw_snapshot.agent_id.is_ascii());
        assert_ne!(raw_snapshot.agent_id, agent_id);
        assert!(
            raw_metadata
                .actor_id
                .as_ref()
                .is_some_and(|actor| actor.is_ascii() && actor != actor_id)
        );
        let loaded_metadata = storage.load_metadata(session_id).await.unwrap().unwrap();
        assert_eq!(loaded_metadata.actor_id, metadata.actor_id);
        assert_eq!(loaded_metadata.actors, metadata.actors);
        assert_eq!(loaded_metadata.tags, metadata.tags);

        storage
            .save_facts(agent_id, actor_id, &[fact(actor_id)])
            .await
            .unwrap();
        let loaded_facts = storage.load_facts(agent_id, actor_id).await.unwrap();
        assert_eq!(loaded_facts[0].actor_id.as_deref(), Some(actor_id));
        let queried_facts = storage
            .query_facts(
                agent_id,
                &FactFilter {
                    actor_id: Some(actor_id.into()),
                    ..FactFilter::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(queried_facts[0].actor_id.as_deref(), Some(actor_id));
        let raw_fact_key = (storage.agent_key(agent_id), storage.actor_key(actor_id));
        let raw_actor = storage.actor_key(actor_id);
        assert_eq!(
            inner.facts.read()[&raw_fact_key][0].actor_id.as_deref(),
            Some(raw_actor.as_str())
        );

        let relationship = serde_json::json!({"actor_id": actor_id, "score": 3});
        storage
            .save_relationship(agent_id, actor_id, &relationship)
            .await
            .unwrap();
        assert_eq!(
            storage.load_relationship(agent_id, actor_id).await.unwrap(),
            Some(relationship)
        );
        assert_eq!(
            storage.list_relationship_actors(agent_id).await.unwrap(),
            vec![actor_id]
        );

        let filter = SessionFilter {
            actor_id: Some(actor_id.into()),
            agent_id: Some(agent_id.into()),
            ..SessionFilter::default()
        };
        let unfiltered = storage
            .list_sessions_filtered(&SessionFilter::default())
            .await
            .unwrap();
        assert_eq!(unfiltered.len(), 1);
        assert_eq!(unfiltered[0].session_id, session_id);

        let summaries = storage.list_sessions_filtered(&filter).await.unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].session_id, session_id);
        assert_eq!(summaries[0].agent_id, agent_id);
        assert_eq!(summaries[0].actor_id.as_deref(), Some(actor_id));
        let raw_filter = inner.last_session_filter.read().clone().unwrap();
        assert_eq!(raw_filter.agent_id, None);
        assert_eq!(raw_filter.actor_id, None);
        assert_eq!(raw_filter.limit, None);
    }

    #[tokio::test]
    async fn filtered_limit_is_applied_after_namespace_isolation() {
        let inner = Arc::new(MemStorage::full());
        let storage = NamespacedStorage::new(inner.clone(), "owned");
        let sibling = NamespacedStorage::new(inner.clone(), "sibling");
        let older = Utc::now() - chrono::Duration::minutes(1);
        let newer = Utc::now();
        let owned_metadata = SessionMetadata {
            last_active: older,
            ..SessionMetadata::default()
        };
        let sibling_metadata = SessionMetadata {
            last_active: newer,
            ..SessionMetadata::default()
        };

        storage
            .save("owned-session", &AgentSnapshot::new("agent".into()))
            .await
            .unwrap();
        storage
            .save_metadata("owned-session", &owned_metadata)
            .await
            .unwrap();
        sibling
            .save("sibling-session", &AgentSnapshot::new("agent".into()))
            .await
            .unwrap();
        sibling
            .save_metadata("sibling-session", &sibling_metadata)
            .await
            .unwrap();

        let summaries = storage
            .list_sessions_filtered(&SessionFilter {
                limit: Some(1),
                ..SessionFilter::default()
            })
            .await
            .unwrap();

        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].session_id, "owned-session");
        assert_eq!(
            inner.last_session_filter.read().as_ref().unwrap().limit,
            None
        );
    }

    #[tokio::test]
    async fn legacy_session_keys_are_read_and_migrated_on_save() {
        let inner = Arc::new(MemStorage::full());
        let storage = NamespacedStorage::new(inner.clone(), "child");
        let legacy_key = "child/legacy";
        inner
            .save(legacy_key, &AgentSnapshot::new("legacy-agent".into()))
            .await
            .unwrap();

        assert_eq!(
            storage.load("legacy").await.unwrap().unwrap().agent_id,
            "legacy-agent"
        );
        assert_eq!(storage.list_sessions().await.unwrap(), vec!["legacy"]);
        assert!(inner.data.read().contains_key(legacy_key));

        storage
            .save("legacy", &AgentSnapshot::new("new-agent".into()))
            .await
            .unwrap();
        assert!(!inner.data.read().contains_key(legacy_key));
        assert_eq!(
            storage.load("legacy").await.unwrap().unwrap().agent_id,
            "new-agent"
        );
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_legacy_session_keys_are_read_migrated_preferred_and_deleted() {
        let directory = std::env::temp_dir().join(format!(
            "ai-agents-namespaced-sqlite-{}",
            uuid::Uuid::new_v4()
        ));
        let path = directory.join("sessions.sqlite");
        let path = path.to_string_lossy().into_owned();

        {
            let inner = Arc::new(ai_agents_storage::SqliteStorage::new(&path).await.unwrap());
            let storage = NamespacedStorage::new(inner.clone(), "child");
            let legacy_key = "child/legacy";
            inner
                .save(legacy_key, &AgentSnapshot::new("legacy-agent".into()))
                .await
                .unwrap();

            assert_eq!(
                storage.load("legacy").await.unwrap().unwrap().agent_id,
                "legacy-agent"
            );
            assert_eq!(storage.list_sessions().await.unwrap(), vec!["legacy"]);
            assert!(inner.load(legacy_key).await.unwrap().is_some());

            storage
                .save("legacy", &AgentSnapshot::new("current-agent".into()))
                .await
                .unwrap();
            assert!(inner.load(legacy_key).await.unwrap().is_none());
            inner
                .save(legacy_key, &AgentSnapshot::new("stale-agent".into()))
                .await
                .unwrap();
            assert_eq!(
                storage.load("legacy").await.unwrap().unwrap().agent_id,
                "current-agent"
            );
            assert_eq!(storage.list_sessions().await.unwrap(), vec!["legacy"]);

            storage.delete("legacy").await.unwrap();
            assert!(storage.load("legacy").await.unwrap().is_none());
            assert!(inner.load(legacy_key).await.unwrap().is_none());
        }

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(feature = "redis-storage")]
    #[tokio::test]
    #[ignore = "requires a Redis service"]
    async fn redis_legacy_session_keys_are_read_migrated_preferred_and_deleted() {
        let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1/".into());
        let prefix = format!("ai-agents-namespace-test:{}:", uuid::Uuid::new_v4());
        let inner = Arc::new(
            ai_agents_storage::RedisStorage::new(&url)
                .unwrap()
                .with_prefix(prefix),
        );
        let storage = NamespacedStorage::new(inner.clone(), "child");
        let legacy_key = "child/legacy";
        inner
            .save(legacy_key, &AgentSnapshot::new("legacy-agent".into()))
            .await
            .unwrap();

        assert_eq!(
            storage.load("legacy").await.unwrap().unwrap().agent_id,
            "legacy-agent"
        );
        assert_eq!(storage.list_sessions().await.unwrap(), vec!["legacy"]);

        storage
            .save("legacy", &AgentSnapshot::new("current-agent".into()))
            .await
            .unwrap();
        assert!(inner.load(legacy_key).await.unwrap().is_none());
        inner
            .save(legacy_key, &AgentSnapshot::new("stale-agent".into()))
            .await
            .unwrap();
        assert_eq!(
            storage.load("legacy").await.unwrap().unwrap().agent_id,
            "current-agent"
        );

        storage.delete("legacy").await.unwrap();
        assert!(storage.load("legacy").await.unwrap().is_none());
        assert!(inner.load(legacy_key).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn malformed_current_namespace_key_fails_closed() {
        let inner = Arc::new(MemStorage::full());
        let storage = NamespacedStorage::new(inner.clone(), "child");
        inner.data.write().insert(
            format!("a7ns1_session_{}_not-hex", storage.encoded_namespace),
            AgentSnapshot::new("agent".into()),
        );

        assert!(matches!(
            storage.list_sessions().await,
            Err(AgentError::Persistence(message)) if message.contains("malformed session key")
        ));
    }

    #[tokio::test]
    async fn file_storage_composition_supports_long_flat_namespace_keys() {
        let directory = std::env::temp_dir().join(format!(
            "ai-agents-namespaced-storage-{}",
            uuid::Uuid::new_v4()
        ));
        let inner = Arc::new(ai_agents_storage::FileStorage::new(&directory));
        let storage = NamespacedStorage::new(inner, "n".repeat(128));
        let session_id = "session-segment/".repeat(256);
        let snapshot = AgentSnapshot::new("agent".repeat(128));

        storage.save(&session_id, &snapshot).await.unwrap();
        assert_eq!(
            storage.list_sessions().await.unwrap(),
            vec![session_id.clone()]
        );
        assert_eq!(
            storage.load(&session_id).await.unwrap().unwrap().agent_id,
            snapshot.agent_id
        );
        storage.delete(&session_id).await.unwrap();
        assert!(storage.load(&session_id).await.unwrap().is_none());

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn relationship_payload_is_opaque() {
        let inner = Arc::new(MemStorage::full());
        let storage = NamespacedStorage::new(inner.clone(), "child");
        let payload = serde_json::json!({
            "kind": "custom",
            "nested": { "actor_id": "payload-owned-value" }
        });

        storage
            .save_relationship("agent", "actor", &payload)
            .await
            .unwrap();

        let raw_key = (storage.agent_key("agent"), storage.actor_key("actor"));
        assert_eq!(inner.relationships.read().get(&raw_key), Some(&payload));
        assert_eq!(
            storage.load_relationship("agent", "actor").await.unwrap(),
            Some(payload)
        );
    }

    #[test]
    fn component_codec_round_trips_empty_ascii_and_unicode_values() {
        for value in ["", "plain", "a/b:c_0", "零😀\0"] {
            let encoded = encode_component(value);
            assert!(encoded.is_ascii());
            assert_eq!(decode_component(&encoded).as_deref(), Some(value));
        }
        assert!(decode_component("0").is_none());
        assert!(decode_component("zz").is_none());
    }
}

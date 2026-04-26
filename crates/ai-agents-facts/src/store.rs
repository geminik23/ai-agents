//! FactStore manages facts for one agent across all actors.

use std::sync::Arc;

use tracing::debug;

use ai_agents_core::types::{FactCategory, FactFilter, KeyFact};
use ai_agents_core::{AgentStorage, Result};

use crate::config::FactsConfig;

/// Manages facts for one agent across all actors.
pub struct FactStore {
    storage: Arc<dyn AgentStorage>,
    agent_id: String,
    config: FactsConfig,
}

impl FactStore {
    pub fn new(storage: Arc<dyn AgentStorage>, agent_id: String, config: FactsConfig) -> Self {
        Self {
            storage,
            agent_id,
            config,
        }
    }

    /// Add new facts for an actor, enforcing max_facts by evicting lowest priority rows.
    /// Returns the authoritative post-write set of facts for the actor so callers can
    /// refresh their cache without re-reading from storage.
    pub async fn add_facts(&self, actor_id: &str, mut facts: Vec<KeyFact>) -> Result<Vec<KeyFact>> {
        if facts.is_empty() {
            return self.storage.load_facts(&self.agent_id, actor_id).await;
        }

        let existing = self.storage.load_facts(&self.agent_id, actor_id).await?;
        let total = existing.len() + facts.len();

        if total > self.config.max_facts {
            // Merge existing and new, sort by priority, keep top max_facts.
            let mut all: Vec<KeyFact> = existing.clone();
            all.append(&mut facts);
            all.sort_by(|a, b| {
                b.priority()
                    .partial_cmp(&a.priority())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            all.truncate(self.config.max_facts);

            // Durable eviction: delete any existing row that did not survive the trim.
            let kept_ids: std::collections::HashSet<&str> =
                all.iter().map(|f| f.id.as_str()).collect();
            let mut evicted = 0usize;
            for old in &existing {
                if !kept_ids.contains(old.id.as_str()) {
                    if let Err(e) = self
                        .storage
                        .delete_fact(&self.agent_id, actor_id, &old.id)
                        .await
                    {
                        debug!(
                            "fact store: failed to delete evicted fact {}: {}",
                            old.id, e
                        );
                    } else {
                        evicted += 1;
                    }
                }
            }

            debug!(
                "fact store: evicted {} facts for actor {} (max_facts={})",
                evicted, actor_id, self.config.max_facts
            );

            // Upsert the kept set (new and surviving existing rows).
            self.storage
                .save_facts(&self.agent_id, actor_id, &all)
                .await?;

            Ok(all)
        } else {
            self.storage
                .save_facts(&self.agent_id, actor_id, &facts)
                .await?;
            // Return the post-write authoritative set.
            self.storage.load_facts(&self.agent_id, actor_id).await
        }
    }

    /// Load all facts for an actor.
    pub async fn get_facts(&self, actor_id: &str) -> Result<Vec<KeyFact>> {
        self.storage.load_facts(&self.agent_id, actor_id).await
    }

    /// Load facts filtered by category.
    pub async fn get_facts_by_category(
        &self,
        actor_id: &str,
        category: &FactCategory,
    ) -> Result<Vec<KeyFact>> {
        let filter = FactFilter {
            actor_id: Some(actor_id.to_string()),
            category: Some(category.clone()),
            ..Default::default()
        };
        self.storage.query_facts(&self.agent_id, &filter).await
    }

    /// Query facts across actors.
    pub async fn query(&self, filter: &FactFilter) -> Result<Vec<KeyFact>> {
        self.storage.query_facts(&self.agent_id, filter).await
    }

    /// Delete a specific fact.
    pub async fn delete_fact(&self, actor_id: &str, fact_id: &str) -> Result<()> {
        self.storage
            .delete_fact(&self.agent_id, actor_id, fact_id)
            .await
    }

    /// Delete all data for an actor (facts + sessions). Privacy compliance.
    pub async fn delete_actor_data(&self, actor_id: &str) -> Result<()> {
        self.storage
            .delete_actor_data(&self.agent_id, actor_id)
            .await
    }

    /// Format facts as a string for chat-time system prompt injection.
    ///
    /// Output is a plain bulleted list of fact statements, ordered by priority,
    /// truncated to fit `max_tokens`. Category labels and confidence scores are
    /// intentionally omitted because they look like metadata noise to the chat
    /// LLM and have caused the model to hedge ("I don't know your identity")
    /// even when the facts clearly state the user's name.
    ///
    /// Use the extractor's separate prompt format when feeding facts back into
    /// the fact extractor for dedup guidance.
    pub fn format_for_context(&self, facts: &[KeyFact], max_tokens: usize) -> String {
        if facts.is_empty() {
            return String::new();
        }

        // Sort by priority (highest first).
        let mut sorted: Vec<&KeyFact> = facts.iter().collect();
        sorted.sort_by(|a, b| {
            b.priority()
                .partial_cmp(&a.priority())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut result = String::new();
        let mut estimated_tokens = 0;

        for fact in sorted {
            // Plain sentence form. Trim trailing punctuation so we add exactly one period.
            let content = fact.content.trim().trim_end_matches(['.', '!', '?']);
            let line = format!("- {}.\n", content);

            // Rough token estimation: ~4 chars per token.
            let line_tokens = line.len() / 4 + 1;
            if estimated_tokens + line_tokens > max_tokens {
                break;
            }

            result.push_str(&line);
            estimated_tokens += line_tokens;
        }

        result
    }

    /// Get the agent ID this store manages.
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Get the underlying config.
    pub fn config(&self) -> &FactsConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_agents_core::AgentSnapshot;
    use ai_agents_core::types::FactCategory;
    use async_trait::async_trait;
    use chrono::Utc;
    use std::sync::Mutex;

    //
    // In-memory storage stub for exercising FactStore persistence semantics.
    //
    struct MemStorage {
        facts: Mutex<Vec<KeyFact>>,
    }

    impl MemStorage {
        fn new() -> Self {
            Self {
                facts: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl ai_agents_core::AgentStorage for MemStorage {
        async fn save(
            &self,
            _session_id: &str,
            _snapshot: &AgentSnapshot,
        ) -> ai_agents_core::Result<()> {
            Ok(())
        }
        async fn load(&self, _session_id: &str) -> ai_agents_core::Result<Option<AgentSnapshot>> {
            Ok(None)
        }
        async fn delete(&self, _session_id: &str) -> ai_agents_core::Result<()> {
            Ok(())
        }
        async fn list_sessions(&self) -> ai_agents_core::Result<Vec<String>> {
            Ok(vec![])
        }

        async fn save_facts(
            &self,
            _agent_id: &str,
            actor_id: &str,
            facts: &[KeyFact],
        ) -> ai_agents_core::Result<()> {
            let mut guard = self.facts.lock().unwrap();
            for f in facts {
                if let Some(existing) = guard.iter_mut().find(|e| e.id == f.id) {
                    *existing = f.clone();
                } else {
                    let mut fact = f.clone();
                    fact.actor_id = Some(actor_id.to_string());
                    guard.push(fact);
                }
            }
            Ok(())
        }

        async fn load_facts(
            &self,
            _agent_id: &str,
            actor_id: &str,
        ) -> ai_agents_core::Result<Vec<KeyFact>> {
            let guard = self.facts.lock().unwrap();
            Ok(guard
                .iter()
                .filter(|f| f.actor_id.as_deref() == Some(actor_id))
                .cloned()
                .collect())
        }

        async fn delete_fact(
            &self,
            _agent_id: &str,
            actor_id: &str,
            fact_id: &str,
        ) -> ai_agents_core::Result<()> {
            let mut guard = self.facts.lock().unwrap();
            guard.retain(|f| !(f.id == fact_id && f.actor_id.as_deref() == Some(actor_id)));
            Ok(())
        }
    }

    fn make_fact(category: FactCategory, content: &str, confidence: f32) -> KeyFact {
        KeyFact {
            id: uuid::Uuid::new_v4().to_string(),
            actor_id: Some("actor_1".to_string()),
            category,
            content: content.to_string(),
            confidence,
            salience: 1.0,
            extracted_at: Utc::now(),
            last_accessed: None,
            source_message_id: None,
            source_language: None,
        }
    }

    #[test]
    fn test_format_for_context_basic() {
        let config = FactsConfig::default();
        let store = FactStore::new(
            Arc::new(ai_agents_core::traits::storage::NoopStorage),
            "test-agent".to_string(),
            config,
        );

        let facts = vec![
            make_fact(FactCategory::UserPreference, "Likes coffee", 0.95),
            make_fact(FactCategory::Decision, "Chose plan A", 0.88),
        ];

        let result = store.format_for_context(&facts, 1000);
        // Plain sentence form, no category bracket, no confidence noise.
        assert!(result.contains("- Likes coffee."));
        assert!(result.contains("- Chose plan A."));
        assert!(!result.contains("[preference]"));
        assert!(!result.contains("confidence"));
    }

    #[test]
    fn test_format_for_context_strips_redundant_punctuation() {
        // Content already ending in '.', '!', or '?' should not produce '..'.
        let config = FactsConfig::default();
        let store = FactStore::new(
            Arc::new(ai_agents_core::traits::storage::NoopStorage),
            "test-agent".to_string(),
            config,
        );

        let facts = vec![
            make_fact(FactCategory::UserContext, "User name is Jay.", 0.9),
            make_fact(FactCategory::UserContext, "User works as engineer", 0.9),
        ];

        let result = store.format_for_context(&facts, 1000);
        assert!(result.contains("- User name is Jay.\n"));
        assert!(result.contains("- User works as engineer.\n"));
        assert!(!result.contains(".."));
    }

    #[test]
    fn test_format_for_context_respects_budget() {
        let config = FactsConfig::default();
        let store = FactStore::new(
            Arc::new(ai_agents_core::traits::storage::NoopStorage),
            "test-agent".to_string(),
            config,
        );

        let facts = vec![
            make_fact(
                FactCategory::UserPreference,
                "Fact one content here that is somewhat long to use tokens",
                0.95,
            ),
            make_fact(
                FactCategory::Decision,
                "Fact two content here that is also somewhat long",
                0.88,
            ),
            make_fact(FactCategory::Agreement, "Fact three content here", 0.80),
        ];

        // Very small budget should limit output.
        let result = store.format_for_context(&facts, 5);
        let lines: Vec<&str> = result.trim().lines().collect();
        assert!(lines.len() <= 2);
    }

    #[test]
    fn test_format_for_context_empty() {
        let config = FactsConfig::default();
        let store = FactStore::new(
            Arc::new(ai_agents_core::traits::storage::NoopStorage),
            "test-agent".to_string(),
            config,
        );

        let result = store.format_for_context(&[], 1000);
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_add_facts_durable_eviction() {
        // Writing more than max_facts must delete the evicted rows from storage,
        // not just skip them in the save set. A follow-up load_facts proves the
        // authoritative count stays at max_facts across reloads.
        let mut config = FactsConfig::default();
        config.max_facts = 3;

        let storage = Arc::new(MemStorage::new());
        let store = FactStore::new(storage.clone(), "agent".to_string(), config);

        let new_facts = vec![
            make_fact(FactCategory::UserPreference, "likes A", 0.99),
            make_fact(FactCategory::UserPreference, "likes B", 0.95),
            make_fact(FactCategory::UserPreference, "likes C", 0.90),
            make_fact(FactCategory::UserPreference, "likes D", 0.60),
            make_fact(FactCategory::UserPreference, "likes E", 0.50),
        ];

        let authoritative = store.add_facts("actor_1", new_facts).await.unwrap();
        assert_eq!(authoritative.len(), 3);

        // Reload from storage proves eviction was durable, not just in-memory.
        let reloaded = store.get_facts("actor_1").await.unwrap();
        assert_eq!(reloaded.len(), 3);

        let contents: Vec<&str> = reloaded.iter().map(|f| f.content.as_str()).collect();
        assert!(contents.contains(&"likes A"));
        assert!(contents.contains(&"likes B"));
        assert!(contents.contains(&"likes C"));
    }

    #[test]
    fn test_format_for_context_sorted_by_priority() {
        let config = FactsConfig::default();
        let store = FactStore::new(
            Arc::new(ai_agents_core::traits::storage::NoopStorage),
            "test-agent".to_string(),
            config,
        );

        let facts = vec![
            make_fact(FactCategory::Decision, "Low priority", 0.5),
            make_fact(FactCategory::UserPreference, "High priority", 0.99),
        ];

        let result = store.format_for_context(&facts, 1000);
        let lines: Vec<&str> = result.trim().lines().collect();
        assert!(lines[0].contains("High priority"));
        assert!(lines[1].contains("Low priority"));
    }
}

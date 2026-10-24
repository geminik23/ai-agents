use super::*;

struct EvictionFailure(Arc<ai_agents_storage::SqliteStorage>);

#[async_trait]
impl AgentStorage for EvictionFailure {
    fn task_providers(&self) -> Option<Vec<Arc<dyn LLMProvider>>> {
        Some(vec![])
    }
    fn supports(&self, capability: ai_agents_core::StorageCapability) -> bool {
        matches!(
            capability,
            ai_agents_core::StorageCapability::Snapshot
                | ai_agents_core::StorageCapability::ActorFacts
        )
    }
    async fn save(&self, id: &str, snapshot: &ai_agents_core::AgentSnapshot) -> Result<()> {
        self.0.save(id, snapshot).await
    }
    async fn load(&self, id: &str) -> Result<Option<ai_agents_core::AgentSnapshot>> {
        self.0.load(id).await
    }
    async fn delete(&self, id: &str) -> Result<()> {
        self.0.delete(id).await
    }
    async fn list_sessions(&self) -> Result<Vec<String>> {
        self.0.list_sessions().await
    }
    async fn save_facts(
        &self,
        agent: &str,
        actor: &str,
        facts: &[ai_agents_core::KeyFact],
    ) -> Result<()> {
        self.0.save_facts(agent, actor, facts).await
    }
    async fn load_facts(&self, agent: &str, actor: &str) -> Result<Vec<ai_agents_core::KeyFact>> {
        self.0.load_facts(agent, actor).await
    }
    async fn delete_fact(&self, _: &str, _: &str, _: &str) -> Result<()> {
        Err(AgentError::Persistence("selected eviction failed".into()))
    }
}

// A rejected eviction cannot publish a successful trimmed cache or allow automatic replay of a potentially partial fact write.
#[tokio::test]
async fn ad_fact_eviction_failure_retains_cache_and_task_custody() {
    let inner = Arc::new(ai_agents_storage::SqliteStorage::in_memory().await.unwrap());
    let storage = Arc::new(EvictionFailure(inner.clone()));
    let facts = Provider::new(
        &[
            r#"[{"category":"user_context","content":"User works at an observatory","confidence":0.99,"actor_id":"actor"}]"#,
        ],
        false,
    );
    let agent = Arc::new(crate::AgentBuilder::from_yaml("name: Actor\nsystem_prompt: test\nmemory:\n  facts:\n    enabled: true\n    auto_extract: true\n    max_facts: 1\n    extractor_llm: facts\ntools: []\n").unwrap().llm(Provider::new(&["done"], false)).llm_alias("facts", facts).storage(storage).auto_configure_features().unwrap().build().unwrap());
    agent.set_actor_id("actor").unwrap();
    let old = ai_agents_core::KeyFact {
        id: "old".into(),
        actor_id: Some("actor".into()),
        category: ai_agents_core::FactCategory::UserPreference,
        content: "User prefers old setting".into(),
        confidence: 0.1,
        salience: 1.0,
        extracted_at: chrono::Utc::now(),
        last_accessed: None,
        source_message_id: None,
        source_language: None,
    };
    inner
        .save_facts(&agent.info.id, "actor", &[old])
        .await
        .unwrap();
    let (runner, store) = runner(agent.clone(), 2);
    let result = runner.run("remember a new preference", None).await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::RecoveryRequired);
    let durable = inner.load_facts(&agent.info.id, "actor").await.unwrap();
    assert_eq!(durable.len(), 1);
    assert_eq!(durable[0].id, "old");
    assert_eq!(agent.actor_facts_cache.read()["actor"][0].id, "old");
    assert!(agent.chat("foreign").await.is_err());
    let payload: crate::autonomy::TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&result.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert!(
        payload
            .reservations
            .iter()
            .any(|r| r.state == TaskEffectState::Uncertain)
    );
}

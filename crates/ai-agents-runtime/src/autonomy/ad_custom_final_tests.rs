use super::*;
use ai_agents_core::{FactCategory, KeyFact};
use ai_agents_facts::{CategoryDefinition, FactExtractor, FactStore, FactsConfig};

struct UncertifiedExtractor;
#[async_trait]
impl FactExtractor for UncertifiedExtractor {
    async fn extract(
        &self,
        _: &[ChatMessage],
        _: &[KeyFact],
        _: Option<&str>,
        _: &[CategoryDefinition],
    ) -> Result<Vec<KeyFact>> {
        panic!("unsupported extractor must not run")
    }
}
struct CustomExtractor(Option<Arc<dyn LLMProvider>>);
#[async_trait]
impl FactExtractor for CustomExtractor {
    fn task_providers(&self) -> Option<Vec<Arc<dyn LLMProvider>>> {
        Some(self.0.iter().cloned().collect())
    }
    // A custom host integration returns proposals and uses the actual managed handle rather than a provider label.
    async fn extract(
        &self,
        _: &[ChatMessage],
        _: &[KeyFact],
        actor: Option<&str>,
        _: &[CategoryDefinition],
    ) -> Result<Vec<KeyFact>> {
        if let Some(provider) = &self.0 {
            provider
                .complete(&[ChatMessage::user("proposal")], None)
                .await?;
        }
        Ok(vec![KeyFact {
            id: "proposal".into(),
            actor_id: actor.map(str::to_string),
            category: FactCategory::UserPreference,
            content: "Prefers concise ticket updates".into(),
            confidence: 0.9,
            salience: 1.0,
            extracted_at: chrono::Utc::now(),
            last_accessed: None,
            source_message_id: None,
            source_language: None,
        }])
    }
}

// Externally installed extractors retain ordinary compatibility while task use requires explicit actual capability and shared accounting.
#[tokio::test]
async fn ad_final_custom_extractor_default_raw_managed_and_provider_free_contracts() {
    for mode in ["default", "raw", "managed", "free"] {
        let storage = Arc::new(ai_agents_storage::SqliteStorage::in_memory().await.unwrap());
        let auxiliary = Provider::new(&["proposal"], false);
        let extractor: Arc<dyn FactExtractor> = match mode {
            "default" => Arc::new(UncertifiedExtractor),
            "raw" => Arc::new(CustomExtractor(Some(auxiliary.clone()))),
            "managed" => Arc::new(CustomExtractor(Some(ai_agents_llm::managed_provider(
                auxiliary.clone(),
            )))),
            _ => Arc::new(CustomExtractor(None)),
        };
        let facts = FactsConfig {
            enabled: true,
            ..Default::default()
        };
        let runtime = crate::AgentBuilder::new()
            .system_prompt("support")
            .llm(Provider::new(&["done"], false))
            .storage(storage.clone())
            .build()
            .unwrap();
        let fact_store = Arc::new(FactStore::new(
            storage.clone(),
            runtime.info.id.clone(),
            facts.clone(),
        ));
        let agent = Arc::new(runtime.with_facts(fact_store, Some(extractor), None, Some(facts)));
        agent.set_actor_id("actor").unwrap();
        if matches!(mode, "default" | "raw") {
            assert!(agent.preflight_standalone_autonomy().is_err());
            assert_eq!(auxiliary.calls.load(Ordering::SeqCst), 0);
        } else {
            let (runner, _) = runner(agent.clone(), 2);
            let result = runner.run("remember the preference", None).await.unwrap();
            assert_eq!(result.run.status, TaskRunStatus::Completed);
            assert_eq!(
                result.run.counters.llm_attempts,
                if mode == "managed" { 2 } else { 1 }
            );
            assert_eq!(
                auxiliary.calls.load(Ordering::SeqCst),
                usize::from(mode == "managed")
            );
            assert_eq!(
                storage.load_facts(&agent.info.id, "actor").await.unwrap()[0].id,
                "proposal"
            );
        }
    }
}

struct PendingFacts {
    inner: Arc<ai_agents_storage::SqliteStorage>,
    entered: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
}
#[async_trait]
impl AgentStorage for PendingFacts {
    fn task_providers(&self) -> Option<Vec<Arc<dyn LLMProvider>>> {
        Some(vec![])
    }
    fn supports(&self, capability: ai_agents_core::StorageCapability) -> bool {
        self.inner.supports(capability)
    }
    async fn save(&self, id: &str, value: &ai_agents_core::AgentSnapshot) -> Result<()> {
        self.inner.save(id, value).await
    }
    async fn load(&self, id: &str) -> Result<Option<ai_agents_core::AgentSnapshot>> {
        self.inner.load(id).await
    }
    async fn delete(&self, id: &str) -> Result<()> {
        self.inner.delete(id).await
    }
    async fn list_sessions(&self) -> Result<Vec<String>> {
        self.inner.list_sessions().await
    }
    async fn load_facts(&self, agent: &str, actor: &str) -> Result<Vec<KeyFact>> {
        self.inner.load_facts(agent, actor).await
    }
    // Hold the actual actor-persistence tail, not a substitute maintenance callback or model response.
    async fn save_facts(&self, agent: &str, actor: &str, facts: &[KeyFact]) -> Result<()> {
        self.entered.add_permits(1);
        self.release.acquire().await.unwrap().forget();
        self.inner.save_facts(agent, actor, facts).await
    }
}
#[derive(Default)]
struct FactHook(AtomicUsize);
#[async_trait]
impl ai_agents_hooks::AgentHooks for FactHook {
    fn task_providers(&self) -> Option<Vec<Arc<dyn LLMProvider>>> {
        Some(vec![])
    }
    async fn on_facts_extracted(&self, _: &str, _: &[KeyFact]) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

// Cache publication, hooks and task completion cannot overtake actual actor storage; cancellation retains uncertain custody.
#[tokio::test]
async fn ad_final_actor_storage_tail_blocks_publication_and_preserves_cancel_custody() {
    for cancel in [false, true] {
        let storage = Arc::new(PendingFacts {
            inner: Arc::new(ai_agents_storage::SqliteStorage::in_memory().await.unwrap()),
            entered: tokio::sync::Semaphore::new(0),
            release: tokio::sync::Semaphore::new(0),
        });
        let hooks = Arc::new(FactHook::default());
        let facts = FactsConfig {
            enabled: true,
            ..Default::default()
        };
        let runtime = crate::AgentBuilder::new()
            .system_prompt("support")
            .llm(Provider::new(&["done"], false))
            .hooks(hooks.clone())
            .storage(storage.clone())
            .build()
            .unwrap();
        let fact_store = Arc::new(FactStore::new(
            storage.clone(),
            runtime.info.id.clone(),
            facts.clone(),
        ));
        let agent = Arc::new(runtime.with_facts(
            fact_store,
            Some(Arc::new(CustomExtractor(None))),
            None,
            Some(facts),
        ));
        agent.set_actor_id("actor").unwrap();
        let (runner, store) = runner(agent.clone(), 1);
        let mut work = Box::pin(runner.run("remember", None));
        tokio::select! { result = &mut work => panic!("storage tail skipped: {:?}", result.err()), permit = storage.entered.acquire() => permit.unwrap().forget() }
        assert!(
            agent
                .actor_facts_cache
                .read()
                .get("actor")
                .is_none_or(Vec::is_empty)
        );
        assert_eq!(hooks.0.load(Ordering::SeqCst), 0);
        assert!(agent.chat("foreign").await.is_err());
        if cancel {
            let run = store.list().await.unwrap().remove(0);
            runner.request_cancel(&run.key.run_id).await.unwrap();
        } else {
            storage.release.add_permits(1);
        }
        let done = tokio::time::timeout(std::time::Duration::from_secs(20), work)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            done.run.status,
            if cancel {
                TaskRunStatus::RecoveryRequired
            } else {
                TaskRunStatus::Completed
            }
        );
        assert_eq!(hooks.0.load(Ordering::SeqCst), usize::from(!cancel));
        assert_eq!(
            storage
                .inner
                .load_facts(&agent.info.id, "actor")
                .await
                .unwrap()
                .len(),
            usize::from(!cancel)
        );
        if cancel {
            assert!(agent.chat("still foreign").await.is_err());
        }
    }
}

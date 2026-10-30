use super::*;

#[cfg(feature = "sqlite")]
#[path = "ad_fact_failure_tests.rs"]
mod fact_failures;

// Pure auxiliary admission denial is a truthful limit; no callback/store effect or fake uncertainty is introduced.
#[tokio::test]
async fn ad_required_maintenance_cap_is_not_success_or_uncertain_dispatch() {
    for mode in ["inline_serial", "inline_parallel", "background"] {
        let main = Provider::new(&["done"], false);
        let auxiliary = Provider::new(&[RELATIONSHIP], false);
        let agent = Arc::new(relationship_agent(mode, main.clone(), auxiliary.clone()));
        agent.set_actor_id("actor").unwrap();
        let (runner, store) = runner(agent.clone(), 1);
        let result = runner.run("resolve", None).await.unwrap();
        assert_eq!(
            result.run.status,
            TaskRunStatus::LimitReached,
            "{mode}: {:?}",
            result.run.stop_reason
        );
        assert_eq!(result.run.counters.llm_attempts, 1);
        assert_eq!(auxiliary.calls.load(Ordering::SeqCst), 0);
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
                .all(|r| r.state == TaskEffectState::Completed)
        );
        assert!(agent.autonomy_owner.read().is_none());
    }
}

// Main and actual auxiliary requests use the same pinned fixed-point accounting, including request-matched refunds of unused cost bounds.
#[tokio::test]
async fn ad_maintenance_matching_price_settlement_uses_one_ledger() {
    let main = Provider::priced(&["working", "done"]);
    let auxiliary = Provider::priced(&[RELATIONSHIP]);
    let agent = Arc::new(relationship_agent("background", main, auxiliary));
    agent.set_actor_id("actor").unwrap();
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(
            agent.info.id.clone(),
            Some("actor".into()),
            "ad-price".into(),
        )
        .unwrap(),
    );
    let config = AutonomyConfig {
        defaults: AutonomyProfile {
            enabled: Some(true),
            max_turns: Some(4),
            max_llm_calls: Some(3),
            max_cost_usd: Some(serde_json::from_value(serde_json::json!("0.000006")).unwrap()),
            completion: Some(CompletionGate::ResponseContains("done".into())),
            ..Default::default()
        },
        ..Default::default()
    };
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        AutonomyHostCeilings {
            max_cost_usd: Some(serde_json::from_value(serde_json::json!("0.000006")).unwrap()),
            ..Default::default()
        },
        store.clone(),
        "ad-price".into(),
    )
    .unwrap();
    let result = runner.run("resolve", None).await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(result.run.counters.llm_attempts, 3);
    assert_eq!(result.run.counters.charged_micro_usd, 3);
    let payload: crate::autonomy::TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&result.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert_eq!(
        payload
            .reservations
            .iter()
            .filter(|r| r.reserved_micro_usd == 2 && r.charged_micro_usd == 1)
            .count(),
        3
    );
}

struct HiddenHook(Arc<Provider>);
#[async_trait]
impl ai_agents_hooks::AgentHooks for HiddenHook {
    fn task_providers(&self) -> Option<Vec<Arc<dyn LLMProvider>>> {
        Some(vec![self.0.clone()])
    }
}

// An enumerated raw callback provider is not made conforming by its registration or matching public name.
#[test]
fn ad_custom_hook_raw_provider_is_rejected_before_dispatch() {
    let raw = Provider::new(&["hidden"], false);
    let agent = crate::AgentBuilder::new()
        .system_prompt("test")
        .llm(Provider::new(&["done"], false))
        .hooks(Arc::new(HiddenHook(raw.clone())))
        .build()
        .unwrap();
    assert!(agent.preflight_standalone_autonomy().is_err());
    assert_eq!(raw.calls.load(Ordering::SeqCst), 0);
}

struct LoadedTail(Tail);
#[async_trait]
impl ai_agents_hooks::AgentHooks for LoadedTail {
    fn task_providers(&self) -> Option<Vec<Arc<dyn LLMProvider>>> {
        Some(vec![])
    }
    async fn on_relationship_loaded(&self, _: &str, _: &ai_agents_relationships::Relationship) {
        *self.0.execution.lock() = crate::autonomy::current_execution()
            .as_ref()
            .map(Arc::downgrade);
        self.0.entered.add_permits(1);
        self.0.release.acquire().await.unwrap().forget();
    }
}

// Pre-turn load callbacks are owned before the first provider request and remain protected after cancellation.
#[tokio::test]
async fn ad_preturn_load_hook_cancel_cannot_escape_accounting_or_owner() {
    let main = Provider::new(&["done"], false);
    let hook = Arc::new(LoadedTail(Tail::default()));
    let agent = Arc::new(
        relationship_agent(
            "background",
            main.clone(),
            Provider::new(&[RELATIONSHIP], false),
        )
        .with_hooks(hook.clone()),
    );
    agent.set_actor_id("actor").unwrap();
    let (runner, _) = runner(agent.clone(), 2);
    let mut work = Box::pin(runner.run("resolve", None));
    tokio::select! { result = &mut work => panic!("load callback skipped: {:?}", result.err()), permit = hook.0.entered.acquire() => permit.unwrap().forget() }
    let execution = hook.0.execution.lock().as_ref().unwrap().upgrade().unwrap();
    runner.request_cancel(&execution.run_id).await.unwrap();
    let result = work.await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::RecoveryRequired);
    assert_eq!(main.calls.load(Ordering::SeqCst), 0);
    assert!(agent.load_actor_relationship().await.is_err());
    assert!(
        agent
            .update_relationship_dimension("trust", 0.1, None)
            .await
            .is_err()
    );
    assert!(agent.chat("foreign").await.is_err());
}

// Facts and relationships use separate real role bindings while Memory/SQLite task backends share identical cumulative behavior.
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn ad_facts_relationships_and_actor_snapshot_memory_sqlite_matrix() {
    for sqlite_tasks in [false, true] {
        for (facts_mode, relationships_mode) in [
            ("inline_serial", "inline_serial"),
            ("inline_parallel", "inline_parallel"),
            ("background", "background"),
            ("background", "inline_serial"),
        ] {
            let storage = Arc::new(ai_agents_storage::SqliteStorage::in_memory().await.unwrap());
            let main = Provider::new(&["working", "done"], false);
            let facts = Provider::new(
                &[
                    r#"[{"category":"user_preference","content":"User prefers concise replies","confidence":0.95,"actor_id":"actor"}]"#,
                ],
                false,
            );
            let relationships = Provider::new(&[RELATIONSHIP], false);
            let yaml = format!(
                "name: Actor\nsystem_prompt: support\nruntime:\n  optimization:\n    enabled: true\n    post_turn:\n      facts:\n        mode: {facts_mode}\n      relationships:\n        mode: {relationships_mode}\nmemory:\n  facts:\n    enabled: true\n    auto_extract: true\n    extractor_llm: facts\n  relationships:\n    enabled: true\n    auto_update:\n      enabled: true\n      llm: relationships\n    persistence:\n      enabled: true\ntools: []\n"
            );
            let agent = Arc::new(
                crate::AgentBuilder::from_yaml(&yaml)
                    .unwrap()
                    .llm(main)
                    .llm_alias("facts", facts.clone())
                    .llm_alias("relationships", relationships.clone())
                    .storage(storage.clone())
                    .auto_configure_features()
                    .unwrap()
                    .build()
                    .unwrap(),
            );
            agent.set_actor_id("actor").unwrap();
            let store = Arc::new(if sqlite_tasks {
                ScopedTaskRunStore::new(
                    storage.clone(),
                    agent.info.id.clone(),
                    Some("actor".into()),
                    "actor-v1".into(),
                )
                .unwrap()
            } else {
                ScopedTaskRunStore::in_memory(
                    agent.info.id.clone(),
                    Some("actor".into()),
                    "actor-v1".into(),
                )
                .unwrap()
            });
            let config = AutonomyConfig {
                defaults: AutonomyProfile {
                    enabled: Some(true),
                    max_turns: Some(4),
                    max_llm_calls: Some(4),
                    completion: Some(CompletionGate::ResponseContains("done".into())),
                    ..Default::default()
                },
                ..Default::default()
            };
            let runner = AutonomyRunner::try_new(
                agent.clone(),
                config,
                AutonomyHostCeilings::default(),
                store,
                "actor-v1".into(),
            )
            .unwrap();
            let result = Box::pin(runner.run("remember the actor preference and resolve", None))
                .await
                .unwrap();
            assert_eq!(
                result.run.status,
                TaskRunStatus::Completed,
                "{sqlite_tasks}/{facts_mode}/{relationships_mode}: {:?}",
                result.run.stop_reason
            );
            assert_eq!(result.run.counters.llm_attempts, 4);
            assert_eq!(facts.calls.load(Ordering::SeqCst), 1);
            assert_eq!(relationships.calls.load(Ordering::SeqCst), 1);
            let durable = storage.load_facts(&agent.info.id, "actor").await.unwrap();
            assert_eq!(durable.len(), 1);
            assert_eq!(agent.actor_facts_cache.read()["actor"].len(), 1);
            assert_eq!(
                agent
                    .relationship_manager
                    .as_ref()
                    .unwrap()
                    .get("actor")
                    .unwrap()
                    .interaction_count,
                1
            );
            let session = agent.task_restore_session().await.unwrap();
            assert_eq!(session["actor_facts"]["actor"].as_array().unwrap().len(), 1);
            assert_eq!(session["extraction_count"], serde_json::json!(0));
            assert!(agent.background_maintenance.is_empty());
        }
    }
}

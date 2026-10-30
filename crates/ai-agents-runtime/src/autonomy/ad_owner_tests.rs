use super::*;

// The ordinary host mutation boundary is runtime-qualified for both coordinating and child reservations.
#[tokio::test]
async fn ad_final_root_and_child_reject_foreign_session_and_actor_mutation() {
    for child in [false, true] {
        let (parent, execution) = crate::autonomy::child_tests::execution(2).await;
        let target = if child {
            let agent = Arc::new(
                crate::AgentBuilder::new()
                    .system_prompt("child")
                    .llm(Provider::new(&["done"], false))
                    .build()
                    .unwrap(),
            );
            crate::autonomy::scope_execution(execution.clone(), agent.chat("work"))
                .await
                .unwrap();
            agent
        } else {
            parent.clone()
        };
        assert!(target.set_session_metadata(Default::default()).is_err());
        assert!(target.save_session("foreign").await.is_err());
        assert!(target.delete_actor_data("actor").await.is_err());
        assert!(target.current_session_id.read().is_none());
        assert!(
            target.save_state().await.is_ok(),
            "inspection is not mutation"
        );
        crate::autonomy::child_tests::finish(&parent, &execution).await;
        target.set_session_metadata(Default::default()).unwrap();
        target.delete_actor_data("actor").await.unwrap();
    }
}

// A safe pause releases its durable writer token, not the live runtime; actor deletion cannot erase that paused run.
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn ad_final_paused_actor_deletion_and_session_identity_preserve_both_storage_layouts() {
    for split in [false, true] {
        let storage = Arc::new(ai_agents_storage::SqliteStorage::in_memory().await.unwrap());
        let task_storage = if split {
            Arc::new(ai_agents_storage::SqliteStorage::in_memory().await.unwrap())
        } else {
            storage.clone()
        };
        let provider = Provider::new(
            &[r#"{"tool":"ask_user","arguments":{"question":"Confirm the ticket?"}}"#],
            false,
        );
        let agent = Arc::new(
            crate::AgentBuilder::from_yaml(
                "name: Owner\nsystem_prompt: support\ntools: [ask_user]\n",
            )
            .unwrap()
            .llm(provider.clone())
            .storage(storage.clone())
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap(),
        );
        agent.set_actor_id("actor").unwrap();
        agent.save_session("original").await.unwrap();
        let store = Arc::new(
            ScopedTaskRunStore::new(
                task_storage,
                agent.info.id.clone(),
                Some("actor".into()),
                "ad-owner.v1".into(),
            )
            .unwrap(),
        );
        let mut config = AutonomyConfig {
            defaults: AutonomyProfile {
                enabled: Some(true),
                max_llm_calls: Some(2),
                completion: Some(CompletionGate::ResponseContains("done".into())),
                ..Default::default()
            },
            ..Default::default()
        };
        config.defaults.hitl = Some(crate::autonomy::AutonomyHitlConfig {
            on_user_question: Some(crate::autonomy::InteractionAction::PauseRun),
            ..Default::default()
        });
        let runner = AutonomyRunner::try_new(
            agent.clone(),
            config,
            Default::default(),
            store.clone(),
            "ad-owner.v1".into(),
        )
        .unwrap();
        let paused = runner.run("resolve", None).await.unwrap();
        assert_eq!(paused.run.status, TaskRunStatus::Paused);
        let before = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
        assert!(before.owner_token.is_none());
        assert!(agent.delete_actor_data("actor").await.is_err());
        assert!(agent.save_session("foreign").await.is_err());
        assert!(agent.set_session_metadata(Default::default()).is_err());
        assert_eq!(agent.current_session_id.read().as_deref(), Some("original"));
        assert_eq!(
            serde_json::to_value(store.load(&paused.run.key.run_id).await.unwrap().unwrap())
                .unwrap(),
            serde_json::to_value(before).unwrap()
        );
        agent.save_to(storage.as_ref(), "export").await.unwrap();
        assert!(storage.load("export").await.unwrap().is_some());
        assert_eq!(agent.current_session_id.read().as_deref(), Some("original"));
        let stopped = runner
            .stop_paused(&paused.run.key.run_id, paused.run.revision, "host cleanup")
            .await
            .unwrap();
        assert_eq!(stopped.run.status, TaskRunStatus::Incomplete);
        agent.save_session("after").await.unwrap();
        agent.set_session_metadata(Default::default()).unwrap();
        agent.delete_actor_data("actor").await.unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    }
}

// Selecting a winner cannot misreport a closed, actually uninvoked alternative as a consumed failed/discarded branch.
#[tokio::test]
async fn ad_final_unreviewed_branch_drain_keeps_uninvoked_identity_and_disposition() {
    use crate::optimization::{
        RuntimeBranch, RuntimeBranchResult, RuntimeCommitBehavior, RuntimeOptimizationKind,
        RuntimeTaskPriority, RuntimeTaskPurpose, ScheduledBranchSet,
    };
    let provider = Provider::new(&["not dispatched"], false);
    let mut observation = ai_agents_observability::ObservabilityConfig {
        enabled: true,
        ..Default::default()
    };
    observation.export.write_report = false;
    observation.export.write_raw_events = true;
    // This fixture records only in-memory branch finalization and never invokes turn export.
    observation.export.path = std::env::temp_dir()
        .join(format!("ad-final-{}", uuid::Uuid::new_v4()))
        .to_string_lossy()
        .into_owned();
    let agent = crate::AgentBuilder::new()
        .system_prompt("branch")
        .llm(Provider::new(&["unused"], false))
        .observability(ai_agents_observability::ObservabilityManager::new(
            observation,
        ))
        .build()
        .unwrap();
    let (parent, execution) = crate::autonomy::child_tests::execution(2).await;
    let branch = RuntimeBranch::new(
        RuntimeTaskPurpose::MainResponse,
        RuntimeOptimizationKind::SpeculativeSkillRouting,
        RuntimeTaskPriority::Normal,
        RuntimeCommitBehavior::FinalResponse,
    );
    let id = branch.branch_id();
    let mut branches = ScheduledBranchSet::new(1).unwrap();
    let managed = ai_agents_llm::managed_provider(provider.clone());
    assert!(branches.schedule(
        branch,
        Box::pin(async move {
            match managed.complete(&[ChatMessage::user("loser")], None).await {
                Ok(_) => panic!("closed branch dispatched"),
                Err(error) => RuntimeBranchResult::Failed(error.into()),
            }
        })
    ));
    crate::autonomy::scope_execution(
        execution.clone(),
        agent.settle_pending_branches(&mut branches),
    )
    .await
    .unwrap();
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    let events = agent.observability().unwrap().raw_events();
    let dispositions: Vec<_> = events
        .iter()
        .filter(|event| event.dimensions.contains_key("runtime.branch_id"))
        .collect();
    // Public identifiers may be privacy-hashed; the private owner binds the original ID and the event must still expose a correlatable identity.
    assert!(execution.owner.observed_branches.lock().contains(&id));
    assert_eq!(dispositions.len(), 1);
    assert!(
        dispositions
            .iter()
            .all(|event| event.dimensions["runtime.branch_status"] == "cancelled")
    );
    crate::autonomy::child_tests::finish(&parent, &execution).await;
}

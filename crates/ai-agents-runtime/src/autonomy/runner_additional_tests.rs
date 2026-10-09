use super::runner_tests::{RecordingProvider, config, fixture};
use super::*;
use ai_agents_core::AgentStorage;
use std::sync::{Arc, atomic::Ordering};

// An independently passing alternative must not consume a forbidden optional judge call at the exact cap.
#[tokio::test]
async fn optional_judge_does_not_override_independent_exact_cap_completion() {
    let provider = RecordingProvider::new(&["done"]);
    let (agent, _, store) = fixture(provider.clone(), 1);
    let mut config = config(1);
    config.defaults.completion = Some(CompletionGate::Any(vec![
        CompletionGate::ResponseContains("done".into()),
        CompletionGate::Judge(JudgeGate {
            llm: None,
            pass_threshold: Some(0.5),
            criteria: vec!["quality".into()],
        }),
    ]));
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        AutonomyHostCeilings::default(),
        store,
        "prepared-v1".into(),
    )
    .unwrap();
    assert_eq!(
        runner.run("objective", None).await.unwrap().run.status,
        TaskRunStatus::Completed
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

// A retained owner confirms the stored request separately from uncertain provider cleanup.
#[tokio::test]
async fn retained_cancellation_interrupts_provider_wait_without_claiming_rollback() {
    let provider = RecordingProvider::gated();
    let (agent, runner, store) = fixture(provider.clone(), 1);
    let runner = Arc::new(runner);
    let task = tokio::spawn({
        let runner = runner.clone();
        async move { runner.run("objective", None).await }
    });
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        provider.entered.as_ref().unwrap().acquire(),
    )
    .await
    .unwrap()
    .unwrap()
    .forget();
    let listing = store.list().await.unwrap();
    let receipt = runner.request_cancel(&listing[0].key.run_id).await.unwrap();
    assert!(receipt.cancel_requested);
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(result.run.status, TaskRunStatus::RecoveryRequired);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert!(agent.reset().await.is_err());
}

// Both storage backends check final admission inside their own atomic mutation boundary.
async fn expired_final_contract(storage: Arc<dyn AgentStorage>) {
    let initial = super::tests::checkpoint("expired-final");
    storage.create_task_run(&initial).await.unwrap();
    let running = storage
        .mutate_task_run(
            &initial.key,
            &TaskRunMutation::Claim {
                expected_revision: 0,
                owner_token: "owner".into(),
            },
        )
        .await
        .unwrap();
    let error = storage
        .mutate_task_run(
            &initial.key,
            &TaskRunMutation::FinalCheckpoint {
                expected_revision: running.revision,
                owner_token: "owner".into(),
                status: TaskRunStatus::Completed,
                payload: running.payload,
                release: true,
                expires_at: None,
                deadline: std::time::Instant::now()
                    .checked_sub(std::time::Duration::from_secs(1))
                    .unwrap(),
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        ai_agents_core::AgentError::TaskRunStorage(TaskRunStorageError::AdmissionExpired)
    ));
    let retained = storage.load_task_run(&initial.key).await.unwrap().unwrap();
    assert_eq!(retained.status, TaskRunStatus::Running);
    assert_eq!(retained.revision, running.revision);
    assert_eq!(retained.owner_token.as_deref(), Some("owner"));
}

// An expired write cannot publish terminal success or consume the owner revision.
#[tokio::test]
async fn memory_backend_rejects_expired_final_admission() {
    expired_final_contract(Arc::new(ai_agents_storage::InMemoryTaskStorage::default())).await;
}

// SQLite follows the same contract after acquiring its immediate write transaction.
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_backend_rejects_expired_final_admission() {
    expired_final_contract(Arc::new(
        ai_agents_storage::SqliteStorage::in_memory().await.unwrap(),
    ))
    .await;
}

// A numeric cost ceiling is not permission to bypass the absent billable-request capability.
#[tokio::test]
async fn standalone_hard_cost_cap_remains_fail_closed_before_dispatch() {
    let provider = RecordingProvider::new(&["done"]);
    let (agent, _, store) = fixture(provider.clone(), 1);
    let mut config = config(1);
    config.defaults.max_cost_usd = Some(UsdAmount::parse("1.00").unwrap());
    let host = AutonomyHostCeilings {
        max_cost_usd: Some(UsdAmount::parse("1.00").unwrap()),
        ..Default::default()
    };
    let runner =
        AutonomyRunner::try_new(agent, config, host, store.clone(), "prepared-v1".into()).unwrap();
    assert!(
        runner
            .run("objective", None)
            .await
            .unwrap_err()
            .to_string()
            .contains("verified provider accounting binding")
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert!(store.list().await.unwrap().is_empty());
}

// Unwrapped preconstructed auxiliary handles cannot silently escape a requested task attempt cap.
#[test]
fn standalone_rejects_unmanaged_memory_summarizer_before_execution() {
    let provider = RecordingProvider::new(&["done"]);
    let memory = ai_agents_memory::CompactingMemory::with_default_config(Arc::new(
        ai_agents_memory::LLMSummarizer::new(provider.clone()),
    ));
    let agent = Arc::new(
        crate::AgentBuilder::new()
            .system_prompt("test")
            .llm(provider)
            .memory(Arc::new(memory))
            .build()
            .unwrap(),
    );
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "prepared-v1".into()).unwrap(),
    );
    assert!(
        AutonomyRunner::try_new(
            agent,
            config(1),
            AutonomyHostCeilings::default(),
            store,
            "prepared-v1".into()
        )
        .is_err()
    );
}

// Required stage progression is ordinary lifecycle work, not an early finish that spends forced-continuation allowance.
#[tokio::test]
async fn lifecycle_progression_does_not_apply_premature_failure_policy() {
    let provider = RecordingProvider::new(&["done", "done"]);
    let (agent, _, store) = fixture(provider.clone(), 3);
    let mut config = config(3);
    config.defaults.lifecycle =
        Some(serde_yaml::from_str("- id: explore\n- id: review\n").unwrap());
    config.defaults.premature_finish = Some(PrematureFinishConfig {
        action: Some(PrematureFinishAction::Fail),
        max_continuations: Some(1),
        prompt: None,
    });
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        AutonomyHostCeilings::default(),
        store,
        "prepared-v1".into(),
    )
    .unwrap();
    let result = runner.run("objective", None).await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(result.run.counters.turns, 2);
    assert_eq!(result.run.counters.continuations, 0);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
}

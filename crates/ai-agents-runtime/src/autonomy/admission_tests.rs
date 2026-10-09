use super::*;

use ai_agents_core::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
    Tool, ToolExecutionContext,
};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct TestDirectory(std::path::PathBuf);
impl TestDirectory {
    fn new() -> Self {
        let path =
            std::env::temp_dir().join(format!("autonomy-footprint-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}
impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct PricedProvider {
    calls: AtomicUsize,
    quote: u64,
    charge: Option<u64>,
    mismatch: bool,
}

#[async_trait]
impl LLMProvider for PricedProvider {
    async fn complete(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(LLMResponse::new(
            if call == 0 { "working" } else { "done" },
            FinishReason::Stop,
        ))
    }
    async fn complete_stream(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<
        Box<dyn futures::Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
        LLMError,
    > {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(futures::stream::empty()))
    }
    fn provider_name(&self) -> &str {
        "priced-test"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
    fn priced_capability_identity(&self) -> Option<String> {
        Some("priced-test.v1".into())
    }
    fn request_cost_bound(
        &self,
        request: &ProviderRequest<'_>,
    ) -> std::result::Result<Option<ProviderCostBound>, LLMError> {
        assert!(!request.messages.is_empty());
        Ok(Some(ProviderCostBound {
            provider_identity: "priced-test.config.v1".into(),
            pricing_identity: "flat-fee.v1".into(),
            request_id: request.request_id.into(),
            max_micro_usd: self.quote,
        }))
    }
    fn settle_request_cost(
        &self,
        bound: &ProviderCostBound,
        _: &LLMResponse,
    ) -> std::result::Result<Option<ProviderCostSettlement>, LLMError> {
        Ok(self.charge.map(|charge| {
            let mut bound = bound.clone();
            if self.mismatch {
                bound.request_id = "different-request".into();
            }
            ProviderCostSettlement {
                bound,
                charged_micro_usd: charge,
            }
        }))
    }
}

// The bound and matching usage come from the actual implementation, not an observability estimate.
fn priced_fixture(
    charge: Option<u64>,
    mismatch: bool,
) -> (Arc<PricedProvider>, AutonomyRunner, Arc<ScopedTaskRunStore>) {
    priced_fixture_storage(
        charge,
        mismatch,
        Arc::new(ai_agents_storage::InMemoryTaskStorage::default()),
    )
}

// Both supported backends acknowledge the same priced schema and conditional successor rules.
fn priced_fixture_storage(
    charge: Option<u64>,
    mismatch: bool,
    storage: Arc<dyn ai_agents_core::AgentStorage>,
) -> (Arc<PricedProvider>, AutonomyRunner, Arc<ScopedTaskRunStore>) {
    let provider = Arc::new(PricedProvider {
        calls: AtomicUsize::new(0),
        quote: 600_000,
        charge,
        mismatch,
    });
    let agent = Arc::new(
        crate::AgentBuilder::new()
            .system_prompt("test")
            .llm(provider.clone())
            .build()
            .unwrap(),
    );
    let store = Arc::new(
        ScopedTaskRunStore::new(storage, agent.info().id, None, "priced-v1".into()).unwrap(),
    );
    let mut config = super::runner_tests::config(4);
    config.defaults.max_cost_usd = Some(UsdAmount::parse("1.00").unwrap());
    let host = AutonomyHostCeilings {
        max_cost_usd: Some(UsdAmount::parse("1.00").unwrap()),
        ..Default::default()
    };
    let runner =
        AutonomyRunner::try_new(agent, config, host, store.clone(), "priced-v1".into()).unwrap();
    (provider, runner, store)
}

#[tokio::test]
async fn priced_matching_usage_releases_unused_bound_and_persists_identity() {
    let (provider, runner, store) = priced_fixture(Some(200_000), false);
    let result = runner.run("objective", None).await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    assert_eq!(result.run.counters.charged_micro_usd, 400_000);
    let payload: TaskCheckpointPayload = serde_json::from_value(
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
            .adapters
            .iter()
            .any(|binding| binding.id == "priced-provider:priced-test.config.v1")
    );
    assert!(
        payload
            .reservations
            .iter()
            .all(|reservation| reservation.reserved_micro_usd == 600_000
                && reservation.charged_micro_usd == 200_000)
    );
    assert_eq!(payload.controller_state["remaining_micro_usd"], 800_000);
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_priced_binding_and_usage_survive_backend_reopen() {
    let directory = TestDirectory::new();
    let path = directory.path().join("tasks.sqlite");
    let storage = Arc::new(
        ai_agents_storage::SqliteStorage::new(path.to_str().unwrap())
            .await
            .unwrap(),
    );
    let (_, runner, store) = priced_fixture_storage(Some(200_000), false, storage);
    let result = runner.run("objective", None).await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    let expected = store.load(&result.run.key.run_id).await.unwrap().unwrap();
    let reopened = ai_agents_storage::SqliteStorage::new(path.to_str().unwrap())
        .await
        .unwrap();
    let restored = ai_agents_core::AgentStorage::load_task_run(&reopened, &result.run.key)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(restored.payload, expected.payload);
}

#[tokio::test]
async fn priced_missing_usage_keeps_full_bound_and_denies_next_dispatch() {
    let (provider, runner, _) = priced_fixture(None, false);
    let result = runner.run("objective", None).await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::LimitReached);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert_eq!(result.run.counters.charged_micro_usd, 600_000);
}

#[tokio::test]
async fn priced_mismatched_usage_keeps_dispatched_reservation_for_recovery() {
    let (provider, runner, store) = priced_fixture(Some(200_000), true);
    let result = runner.run("objective", None).await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::RecoveryRequired);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    let payload: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&result.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert_eq!(payload.reservations[0].reserved_micro_usd, 600_000);
    assert_eq!(payload.reservations[0].state, TaskEffectState::Dispatched);
}

// Ledger tests exercise the same conditional memory store as foreground runs, without replacing the executor.
async fn ledger(write_limit: Option<u32>, cost: bool) -> Arc<RunExecution> {
    let mut mock = ai_agents_llm::mock::MockLLMProvider::new("test");
    mock.set_response("done");
    let agent = Arc::new(
        crate::AgentBuilder::new()
            .system_prompt("test")
            .llm(Arc::new(mock))
            .build()
            .unwrap(),
    );

    let store =
        Arc::new(ScopedTaskRunStore::in_memory(agent.info().id, None, "ledger-v1".into()).unwrap());
    let owner = agent.reserve_autonomy_run("ledger".into()).await.unwrap();
    let mut config = super::runner_tests::config(8);
    config.defaults.max_declared_write_paths = write_limit;
    config.defaults.max_cost_usd = cost.then(|| UsdAmount::parse("1.00").unwrap());
    let host = AutonomyHostCeilings {
        max_declared_write_paths: write_limit,
        max_cost_usd: cost.then(|| UsdAmount::parse("1.00").unwrap()),
        ..Default::default()
    };
    let profile = resolve_profile(
        &config,
        AutonomyScope::Task,
        None,
        None,
        None,
        Some("objective"),
        &host,
    )
    .unwrap();
    let lifecycle = LifecycleState::new(&profile.settings, false).unwrap();
    let payload = TaskCheckpointPayload::new(
        "objective".into(),
        "ledger-v1".into(),
        &profile,
        TaskRuntimeCheckpoint::between_turns(agent.save_state().await.unwrap()).unwrap(),
    );
    let now = chrono::Utc::now();
    let snapshot = payload
        .bind(TaskRunSnapshot {
            schema_version: TASK_RUN_SCHEMA_VERSION,
            key: TaskRunKey {
                agent_id: agent.info().id,
                run_id: "ledger".into(),
            },
            actor_id: None,
            revision: 0,
            status: TaskRunStatus::Running,
            owner_token: Some("owner".into()),
            cancel_requested: false,
            created_at: now,
            updated_at: now,
            payload: Value::Null,
        })
        .unwrap();
    store.create(&snapshot).await.unwrap();
    RunExecution::new(
        store,
        &snapshot,
        lifecycle,
        owner,
        std::time::Instant::now(),
    )
    .unwrap()
}

#[tokio::test]
async fn priced_parallel_admission_cannot_share_the_last_slot() {
    let execution = ledger(None, true).await;
    let bound = |id: &str| ProviderCostBound {
        provider_identity: "test".into(),
        pricing_identity: "v1".into(),
        request_id: id.into(),
        max_micro_usd: 600_000,
    };
    let (left, right) = tokio::join!(
        execution.admit_priced_llm(Some(bound("left"))),
        execution.admit_priced_llm(Some(bound("right")))
    );
    assert_ne!(left.is_ok(), right.is_ok());
    let snapshot = execution.store.load("ledger").await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
    assert_eq!(payload.reservations.len(), 1);
    assert_eq!(payload.counters.llm_attempts, 1);
}

#[tokio::test]
async fn abandoned_owner_prevents_admission_in_a_retained_scope() {
    let execution = ledger(None, true).await;
    execution.owner.abandon();
    assert!(
        execution
            .admit_priced_llm(Some(ProviderCostBound {
                provider_identity: "test".into(),
                pricing_identity: "v1".into(),
                request_id: "request".into(),
                max_micro_usd: 1
            }))
            .await
            .is_err()
    );
    assert!(execution.execution_stopped());
    let payload: TaskCheckpointPayload = serde_json::from_value(
        execution
            .store
            .load("ledger")
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert_eq!(payload.counters.llm_attempts, 0);
    assert!(payload.reservations.is_empty());
}

#[tokio::test]
async fn captured_admission_crosses_spawn_and_join_set_boundaries() {
    let execution = ledger(None, true).await;
    let mut mock = ai_agents_llm::mock::MockLLMProvider::new("default");
    mock.set_response("done");
    let provider = ai_agents_llm::managed_provider(Arc::new(mock));
    let captured = execution.clone();
    let participant = provider.clone();
    tokio::spawn(async move {
        scope_inherited_execution(Some(captured.clone()), async move {
            assert!(Arc::ptr_eq(&captured, &current_execution().unwrap()));
            participant
                .complete(&[ChatMessage::user("first")], None)
                .await
                .unwrap();
        })
        .await;
    })
    .await
    .unwrap();
    let mut tasks = tokio::task::JoinSet::new();
    let captured = execution.clone();
    tasks.spawn(async move {
        scope_inherited_execution(Some(captured.clone()), async move {
            assert!(Arc::ptr_eq(&captured, &current_execution().unwrap()));
            provider
                .complete(&[ChatMessage::user("second")], None)
                .await
                .unwrap();
        })
        .await;
    });
    tasks.join_next().await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(
        execution
            .store
            .load("ledger")
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert_eq!(payload.counters.llm_attempts, 2);
    assert!(
        payload
            .reservations
            .iter()
            .all(|reservation| reservation.state == TaskEffectState::Completed)
    );
}

#[tokio::test]
async fn priced_pinned_schedule_cannot_change_during_run() {
    let execution = ledger(None, true).await;
    let mut bound = ProviderCostBound {
        provider_identity: "test".into(),
        pricing_identity: "v1".into(),
        request_id: "first".into(),
        max_micro_usd: 0,
    };
    let id = execution
        .admit_priced_llm(Some(bound.clone()))
        .await
        .unwrap();
    execution
        .settle_llm(&id, json!({"response":"done"}))
        .await
        .unwrap();
    bound.request_id = "second".into();
    bound.pricing_identity = "v2".into();
    assert!(execution.admit_priced_llm(Some(bound)).await.is_err());
    assert_eq!(
        execution.stop_reason().as_deref(),
        Some("pricing_identity_changed")
    );
}

#[tokio::test]
async fn write_union_reuses_targets_and_retains_uncertain_effects() {
    let execution = ledger(Some(1), false).await;
    let tool = ai_agents_tools::builtin::FileWriteTool::new();
    let context = ToolExecutionContext::test("file_write");
    let directory = TestDirectory::new();
    let args = json!({"path":directory.path().join("a"),"content":"a"});
    let (first, _) = execution.admit_tool(&tool, &args, &context).await.unwrap();
    execution
        .settle(&first, json!({"success":true}), false)
        .await
        .unwrap();
    let (second, _) = execution.admit_tool(&tool, &args, &context).await.unwrap();
    execution
        .settle(&second, json!({"uncertain":true}), true)
        .await
        .unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(
        execution
            .store
            .load("ledger")
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert_eq!(payload.declared_write_targets.len(), 1);
    assert_eq!(payload.reservations[1].state, TaskEffectState::Uncertain);
}

#[tokio::test]
async fn write_union_denies_new_target_before_an_attempt_is_admitted() {
    let execution = ledger(Some(1), false).await;
    let tool = ai_agents_tools::builtin::FileWriteTool::new();
    let context = ToolExecutionContext::test("file_write");
    let directory = TestDirectory::new();
    let args = |name| json!({"path":directory.path().join(name),"content":"a"});
    execution
        .admit_tool(&tool, &args("a"), &context)
        .await
        .unwrap();
    assert!(
        execution
            .admit_tool(&tool, &args("b"), &context)
            .await
            .is_err()
    );
    let payload: TaskCheckpointPayload = serde_json::from_value(
        execution
            .store
            .load("ledger")
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert_eq!(payload.counters.tool_attempts, 1);
    assert_eq!(payload.declared_write_targets.len(), 1);
}

#[tokio::test]
async fn proven_non_invocation_refunds_only_its_slot_and_unused_targets() {
    let execution = ledger(Some(1), false).await;
    let directory = TestDirectory::new();
    let tool = ai_agents_tools::builtin::FileWriteTool::new();
    let args = |name: &str| json!({"path":directory.path().join(name),"content":"x"});
    let (id, _) = execution
        .admit_tool(&tool, &args("a"), &ToolExecutionContext::test("file_write"))
        .await
        .unwrap();
    execution
        .settle(&id, json!({"not_invoked":true}), false)
        .await
        .unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(
        execution
            .store
            .load("ledger")
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert_eq!(payload.counters.tool_attempts, 0);
    assert!(payload.declared_write_targets.is_empty());
    let (id, _) = execution
        .admit_tool(&tool, &args("b"), &ToolExecutionContext::test("file_write"))
        .await
        .unwrap();
    execution
        .settle(&id, json!({"success":false}), false)
        .await
        .unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(
        execution
            .store
            .load("ledger")
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert_eq!(payload.counters.tool_attempts, 1);
    assert_eq!(payload.declared_write_targets.len(), 1);
}

#[tokio::test]
async fn non_invocation_cannot_release_a_target_used_by_a_sibling_attempt() {
    let execution = ledger(Some(1), false).await;
    let tool = ai_agents_tools::builtin::FileWriteTool::new();
    let directory = TestDirectory::new();
    let args = json!({"path":directory.path().join("a"),"content":"x"});
    let ctx = ToolExecutionContext::test("file_write");
    let (first, _) = execution.admit_tool(&tool, &args, &ctx).await.unwrap();
    let (second, _) = execution.admit_tool(&tool, &args, &ctx).await.unwrap();
    execution
        .settle(&first, json!({"success":true}), false)
        .await
        .unwrap();
    execution
        .settle(&second, json!({"not_invoked":true}), false)
        .await
        .unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(
        execution
            .store
            .load("ledger")
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert_eq!(payload.counters.tool_attempts, 1);
    assert_eq!(payload.declared_write_targets.len(), 1);
}

#[tokio::test]
async fn known_preview_never_consumes_a_write_target() {
    let execution = ledger(Some(1), false).await;
    let tool = ai_agents_tools::builtin::FileWriteTool::new();
    let (id, footprint) = execution
        .admit_tool(
            &tool,
            &json!({"path":"preview","content":"a","dry_run":true}),
            &ToolExecutionContext::test("file_write"),
        )
        .await
        .unwrap();
    assert!(footprint.unwrap().targets.is_empty());
    execution
        .settle(&id, json!({"success":true}), false)
        .await
        .unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(
        execution
            .store
            .load("ledger")
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert!(payload.declared_write_targets.is_empty());
}

#[test]
fn recursive_footprint_is_bounded_and_covers_replacement_descendants() {
    let directory = TestDirectory::new();
    let source = directory.path().join("source");
    let destination = directory.path().join("destination");
    std::fs::create_dir(&source).unwrap();
    std::fs::create_dir(&destination).unwrap();
    std::fs::write(source.join("new"), "new").unwrap();
    std::fs::write(destination.join("old"), "old").unwrap();
    let args = json!({"source_path":source,"destination_path":destination,"overwrite":true});
    let tool = ai_agents_tools::builtin::CopyPathTool::new();
    let context = ToolExecutionContext::test("copy_path");
    assert!(tool.declared_write_footprint(&args, &context, 2).is_err());
    let footprint = tool
        .declared_write_footprint(&args, &context, 3)
        .unwrap()
        .unwrap();
    assert_eq!(footprint.targets.len(), 3);
}

#[cfg(unix)]
#[test]
fn replacement_descendants_do_not_collapse_through_the_old_tree() {
    let directory = TestDirectory::new();
    let source = directory.path().join("source");
    let destination = directory.path().join("destination");
    let referent = directory.path().join("referent");
    std::fs::create_dir_all(source.join("a")).unwrap();
    std::fs::create_dir_all(source.join("b")).unwrap();
    std::fs::create_dir(&destination).unwrap();
    std::fs::create_dir(&referent).unwrap();
    std::fs::write(source.join("a/x"), "a").unwrap();
    std::fs::write(source.join("b/x"), "b").unwrap();
    std::os::unix::fs::symlink(&referent, destination.join("a")).unwrap();
    std::os::unix::fs::symlink(&referent, destination.join("b")).unwrap();
    let tool = ai_agents_tools::builtin::CopyPathTool::new();
    let args = json!({"source_path":source,"destination_path":destination,"overwrite":true});
    assert!(
        tool.declared_write_footprint(&args, &ToolExecutionContext::test("copy_path"), 4)
            .is_err()
    );
    assert_eq!(
        tool.declared_write_footprint(&args, &ToolExecutionContext::test("copy_path"), 5)
            .unwrap()
            .unwrap()
            .targets
            .len(),
        5
    );
}

#[cfg(unix)]
#[test]
fn content_footprint_rejects_symlink_parent_traversal_before_normalization() {
    let directory = TestDirectory::new();
    let nested = directory.path().join("d/subdirectory");
    std::fs::create_dir_all(&nested).unwrap();
    std::os::unix::fs::symlink(&nested, directory.path().join("link")).unwrap();
    let tool = ai_agents_tools::builtin::FileTool::new();
    assert!(
        tool.declared_write_footprint(
            &json!({"path":directory.path().join("link/../x"),"operation":"write","content":"x"}),
            &ToolExecutionContext::test("file"),
            1
        )
        .is_err()
    );
}

#[cfg(unix)]
#[test]
fn unlink_aliases_are_distinct_but_content_aliases_share_the_referent() {
    let directory = TestDirectory::new();
    let file = directory.path().join("file");
    let a = directory.path().join("a");
    let b = directory.path().join("b");
    std::fs::write(&file, "content").unwrap();
    std::os::unix::fs::symlink(&file, &a).unwrap();
    std::os::unix::fs::symlink(&file, &b).unwrap();
    let delete = ai_agents_tools::builtin::DeletePathTool::new();
    let context = ToolExecutionContext::test("delete_path");
    let footprint = |path: &std::path::Path| {
        delete
            .declared_write_footprint(&json!({"path":path}), &context, 1)
            .unwrap()
            .unwrap()
    };
    assert_ne!(footprint(&a).targets, footprint(&b).targets);
    let content = ai_agents_tools::builtin::FileTool::new();
    let footprint = |path: &std::path::Path| {
        content
            .declared_write_footprint(
                &json!({"path":path,"operation":"write","content":"new"}),
                &ToolExecutionContext::test("file"),
                1,
            )
            .unwrap()
            .unwrap()
    };
    assert_eq!(footprint(&a).targets, footprint(&b).targets);
}

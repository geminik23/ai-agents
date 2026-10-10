use super::*;
use crate::Agent;
use ai_agents_core::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
    Result,
};
use async_trait::async_trait;
use serde_json::Value;
use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

pub(super) struct RecordingProvider {
    replies: parking_lot::Mutex<VecDeque<String>>,
    messages: parking_lot::Mutex<Vec<Vec<ChatMessage>>>,
    pub(super) calls: AtomicUsize,
    pub(super) entered: Option<Arc<tokio::sync::Semaphore>>,
    pub(super) release: Option<Arc<tokio::sync::Semaphore>>,
}
impl RecordingProvider {
    // A real provider boundary records exactly the request seen after runtime and wrapper preparation.
    pub(super) fn new(replies: &[&str]) -> Arc<Self> {
        Arc::new(Self {
            replies: parking_lot::Mutex::new(replies.iter().map(|s| s.to_string()).collect()),
            messages: parking_lot::Mutex::new(vec![]),
            calls: AtomicUsize::new(0),
            entered: None,
            release: None,
        })
    }
    // Gated dispatch makes ownership/cancellation ordering deterministic without latency assertions.
    pub(super) fn gated() -> Arc<Self> {
        Arc::new(Self {
            replies: parking_lot::Mutex::new(VecDeque::from(["done".into()])),
            messages: parking_lot::Mutex::new(vec![]),
            calls: AtomicUsize::new(0),
            entered: Some(Arc::new(tokio::sync::Semaphore::new(0))),
            release: Some(Arc::new(tokio::sync::Semaphore::new(0))),
        })
    }
}
#[async_trait]
impl LLMProvider for RecordingProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.messages.lock().push(messages.to_vec());
        if let (Some(entered), Some(release)) = (&self.entered, &self.release) {
            entered.add_permits(1);
            release.acquire().await.unwrap().forget();
        }
        Ok(LLMResponse::new(
            self.replies
                .lock()
                .pop_front()
                .unwrap_or_else(|| "done".into()),
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
        Ok(Box::new(futures::stream::empty()))
    }
    fn provider_name(&self) -> &str {
        "recording"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
}

// The existing builder stays disabled; the development runner is an explicit host invocation surface.
pub(super) fn fixture(
    provider: Arc<RecordingProvider>,
    max_llm_calls: u32,
) -> (
    Arc<crate::RuntimeAgent>,
    AutonomyRunner,
    Arc<ScopedTaskRunStore>,
) {
    let agent = Arc::new(
        crate::AgentBuilder::new()
            .system_prompt("test")
            .llm(provider)
            .build()
            .unwrap(),
    );
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "prepared-v1".into()).unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent.clone(),
        config(max_llm_calls),
        AutonomyHostCeilings::default(),
        store.clone(),
        "prepared-v1".into(),
    )
    .unwrap();
    (agent, runner, store)
}

// A response predicate is deliberate deterministic completion, not a semantic acceptance heuristic.
pub(super) fn config(max_llm_calls: u32) -> AutonomyConfig {
    AutonomyConfig {
        defaults: AutonomyProfile {
            enabled: Some(true),
            max_turns: Some(4),
            max_llm_calls: Some(max_llm_calls),
            completion: Some(CompletionGate::ResponseContains("done".into())),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[tokio::test]
async fn standalone_completes_at_exact_provider_cap_and_releases_runtime() {
    let provider = RecordingProvider::new(&["done", "ordinary"]);
    let (agent, runner, store) = fixture(provider.clone(), 1);
    let result = runner.run("objective", None).await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(result.run.counters.llm_attempts, 1);
    assert_eq!(result.run.counters.turns, 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        store
            .load(&result.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .status,
        TaskRunStatus::Completed
    );
    assert_eq!(agent.chat("follow-up").await.unwrap().content, "ordinary");
}

#[tokio::test]
async fn continuation_commits_one_objective_and_persists_output_provenance() {
    let provider = RecordingProvider::new(&["working", "done"]);
    let (_, runner, store) = fixture(provider.clone(), 3);
    let result = runner.run("objective", None).await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(result.run.counters.turns, 2);
    assert_eq!(result.run.counters.continuations, 1);
    let snapshot = store.load(&result.run.key.run_id).await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
    let messages = &payload.runtime.snapshot.memory.messages;
    assert_eq!(
        messages
            .iter()
            .filter(|m| m.role == ai_agents_core::Role::User)
            .count(),
        1
    );
    assert!(
        messages
            .iter()
            .filter(|m| m.role == ai_agents_core::Role::Assistant)
            .all(|m| m.provenance.is_some())
    );
    assert!(
        !messages
            .iter()
            .any(|m| m.content.contains("Autonomy controller instruction"))
    );
    assert!(provider.messages.lock().iter().all(|request| {
        request
            .iter()
            .any(|m| m.content.contains("Autonomy controller instruction"))
    }));
}

// The controller view reports command allowance and both clocks, not only turn and call counts.
#[tokio::test]
async fn controller_state_reports_command_and_clock_allowance() {
    let provider = RecordingProvider::new(&["working", "done"]);
    let (_, runner, store) = fixture(provider.clone(), 3);
    let result = runner.run("objective", None).await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    let snapshot = store.load(&result.run.key.run_id).await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
    let controller = payload.controller_state.as_object().unwrap();
    assert_eq!(
        controller.get("remaining_command_calls"),
        Some(&serde_json::json!(10))
    );
    assert!(
        controller
            .get("remaining_active_seconds")
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|seconds| seconds > 0)
    );
    assert_eq!(
        controller.get("remaining_wall_seconds"),
        Some(&serde_json::Value::Null)
    );
}

#[tokio::test]
async fn ownership_blocks_unrelated_roots_and_session_mutation_while_provider_waits() {
    let provider = RecordingProvider::gated();
    let (agent, runner, _) = fixture(provider.clone(), 1);
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
    assert!(
        agent
            .chat("unrelated")
            .await
            .unwrap_err()
            .to_string()
            .contains("busy")
    );
    assert!(agent.chat_stream("unrelated").await.is_err());
    assert!(agent.reset().await.is_err());
    assert!(
        agent
            .restore_state(agent.save_state().await.unwrap())
            .await
            .is_err()
    );
    provider.release.as_ref().unwrap().add_permits(1);
    assert_eq!(
        task.await.unwrap().unwrap().run.status,
        TaskRunStatus::Completed
    );
}

#[tokio::test]
async fn committed_cancel_settles_known_result_without_losing_ownership() {
    let provider = RecordingProvider::gated();
    let (_, runner, store) = fixture(provider.clone(), 1);
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
    let listed = store.list().await.unwrap();
    let snapshot = store.load(&listed[0].key.run_id).await.unwrap().unwrap();
    store
        .mutate(
            &snapshot.key.run_id,
            &TaskRunMutation::RequestCancel {
                expected_revision: snapshot.revision,
            },
        )
        .await
        .unwrap();
    provider.release.as_ref().unwrap().add_permits(1);
    let result = task.await.unwrap().unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Cancelled);
    let snapshot = store.load(&result.run.key.run_id).await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
    assert!(
        payload
            .reservations
            .iter()
            .all(|r| r.state == TaskEffectState::Completed)
    );
    assert!(snapshot.cancel_requested);
    assert!(snapshot.owner_token.is_none());
}

// Known timeout uncertainty must stop the next admission even when settlement storage returns a generic error.
#[tokio::test]
async fn failed_uncertain_settlement_stops_subsequent_tool_admission() {
    let provider = RecordingProvider::new(&["done"]);
    let (agent, _, _) = fixture(provider, 2);
    let owner = agent
        .reserve_autonomy_run("settlement".into())
        .await
        .unwrap();
    let store =
        Arc::new(ScopedTaskRunStore::in_memory("agent".into(), None, "config-v1".into()).unwrap());
    let initial = super::tests::checkpoint("settlement");
    store.create(&initial).await.unwrap();
    let running = store
        .mutate(
            "settlement",
            &TaskRunMutation::Claim {
                expected_revision: 0,
                owner_token: "owner".into(),
            },
        )
        .await
        .unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(running.payload.clone()).unwrap();
    let lifecycle = LifecycleState::new(&payload.settings, false).unwrap();
    let failing = Arc::new(FailingStore {
        inner: store.clone(),
        writes: AtomicUsize::new(0),
        fail_at: 2,
        fail_terminal: false,
    });
    let execution = RunExecution::new(
        failing,
        &running,
        lifecycle,
        owner.clone(),
        std::time::Instant::now(),
    )
    .unwrap();
    let admitted = execution.admit(Some("effect")).await.unwrap();
    assert!(
        execution
            .settle(&admitted, serde_json::json!({"timed_out":true}), true)
            .await
            .is_err()
    );
    assert_eq!(execution.stop_reason().as_deref(), Some("uncertain_effect"));
    assert!(execution.admit(Some("second_effect")).await.is_err());
    let retained = store.load("settlement").await.unwrap().unwrap();
    let retained: TaskCheckpointPayload = serde_json::from_value(retained.payload).unwrap();
    assert_eq!(retained.counters.tool_attempts, 1);
    assert_eq!(retained.reservations.len(), 1);
    agent.release_autonomy_run(&owner).await.unwrap();
}

struct FailingStore {
    inner: Arc<ScopedTaskRunStore>,
    writes: AtomicUsize,
    fail_at: usize,
    fail_terminal: bool,
}
#[async_trait]
impl TaskRunStore for FailingStore {
    async fn create(&self, snapshot: &TaskRunSnapshot) -> Result<()> {
        self.inner.create(snapshot).await
    }
    async fn load(&self, run_id: &str) -> Result<Option<TaskRunSnapshot>> {
        self.inner.load(run_id).await
    }
    async fn mutate(&self, run_id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot> {
        let write = self.writes.fetch_add(1, Ordering::SeqCst) + 1;
        if write == self.fail_at
            || (self.fail_terminal
                && matches!(mutation, TaskRunMutation::Checkpoint { status, .. } | TaskRunMutation::FinalCheckpoint { status, .. } if status.is_terminal()))
        {
            return Err(ai_agents_core::AgentError::Other(
                "fixture checkpoint failure".into(),
            ));
        }
        self.inner.mutate(run_id, mutation).await
    }
    async fn list(&self) -> Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    async fn delete(&self, run_id: &str, revision: u64) -> Result<()> {
        self.inner.delete(run_id, revision).await
    }
}

#[tokio::test]
async fn failed_admission_does_not_invoke_provider_or_publish_completion() {
    let provider = RecordingProvider::new(&["done"]);
    let (agent, _, store) = fixture(provider.clone(), 1);
    let failing = Arc::new(FailingStore {
        inner: store.clone(),
        writes: AtomicUsize::new(0),
        fail_at: 2,
        fail_terminal: false,
    });
    let runner = AutonomyRunner::try_new(
        agent,
        config(1),
        AutonomyHostCeilings::default(),
        failing,
        "prepared-v1".into(),
    )
    .unwrap();
    let result = runner.run("objective", None).await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Failed);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn failed_terminal_checkpoint_returns_storage_error_and_keeps_runtime_reserved() {
    let provider = RecordingProvider::new(&["done"]);
    let (agent, _, store) = fixture(provider, 1);
    let failing = Arc::new(FailingStore {
        inner: store,
        writes: AtomicUsize::new(0),
        fail_at: usize::MAX,
        fail_terminal: true,
    });
    let runner = AutonomyRunner::try_new(
        agent.clone(),
        config(1),
        AutonomyHostCeilings::default(),
        failing,
        "prepared-v1".into(),
    )
    .unwrap();
    assert!(
        runner
            .run("objective", None)
            .await
            .unwrap_err()
            .to_string()
            .contains("checkpoint failure")
    );
    assert!(agent.chat("unrelated").await.is_err());
}

struct TerminalAckStore {
    inner: Arc<ScopedTaskRunStore>,
    entered: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
}
#[async_trait]
impl TaskRunStore for TerminalAckStore {
    async fn create(&self, snapshot: &TaskRunSnapshot) -> Result<()> {
        self.inner.create(snapshot).await
    }
    async fn load(&self, run_id: &str) -> Result<Option<TaskRunSnapshot>> {
        self.inner.load(run_id).await
    }
    async fn mutate(&self, run_id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot> {
        let saved = self.inner.mutate(run_id, mutation).await?;
        if saved.status == TaskRunStatus::Completed {
            self.entered.add_permits(1);
            self.release.acquire().await.unwrap().forget();
        }
        Ok(saved)
    }
    async fn list(&self) -> Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    async fn delete(&self, run_id: &str, revision: u64) -> Result<()> {
        self.inner.delete(run_id, revision).await
    }
}

// A completed durable status cannot be replayed or downgraded merely because live cleanup was interrupted.
#[tokio::test]
async fn dropped_terminal_acknowledgement_can_release_completed_live_protection() {
    let provider = RecordingProvider::new(&["done"]);
    let (agent, _, store) = fixture(provider.clone(), 1);
    let gated = Arc::new(TerminalAckStore {
        inner: store.clone(),
        entered: tokio::sync::Semaphore::new(0),
        release: tokio::sync::Semaphore::new(0),
    });
    let runner = Arc::new(
        AutonomyRunner::try_new(
            agent.clone(),
            config(1),
            Default::default(),
            gated.clone(),
            "prepared-v1".into(),
        )
        .unwrap(),
    );
    let task = tokio::spawn({
        let runner = runner.clone();
        async move { runner.run("objective", None).await }
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), gated.entered.acquire())
        .await
        .unwrap()
        .unwrap()
        .forget();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    let snapshot = store
        .load(&store.list().await.unwrap()[0].key.run_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(snapshot.status, TaskRunStatus::Completed);
    assert!(agent.chat("unrelated").await.is_err());
    let released = runner
        .acknowledge_recovery_release(&snapshot.key.run_id, snapshot.revision)
        .await
        .unwrap();
    assert_eq!(released.status, TaskRunStatus::Completed);
    assert_eq!(released.revision, snapshot.revision);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert!(agent.chat("ordinary").await.is_ok());
}

// Durable reconciliation and live release are separate acknowledgements; neither can replay an unknown request.
#[tokio::test]
async fn abandoned_owner_releases_only_after_reconciled_storage_acknowledgement() {
    let provider = RecordingProvider::gated();
    let (agent, runner, store) = fixture(provider.clone(), 2);
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
    let running = store
        .load(&store.list().await.unwrap()[0].key.run_id)
        .await
        .unwrap()
        .unwrap();
    assert!(
        runner
            .acknowledge_recovery_release(&running.key.run_id, running.revision)
            .await
            .is_err()
    );
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert!(
        runner
            .acknowledge_recovery_release(&running.key.run_id, running.revision)
            .await
            .is_err()
    );
    let recovery = store
        .mutate(
            &running.key.run_id,
            &TaskRunMutation::Recover {
                expected_revision: running.revision,
                owner_token: running.owner_token.unwrap(),
            },
        )
        .await
        .unwrap();
    assert!(
        runner
            .acknowledge_recovery_release(&running.key.run_id, recovery.revision)
            .await
            .is_err()
    );
    let mut payload: TaskCheckpointPayload =
        serde_json::from_value(recovery.payload.clone()).unwrap();
    let used = payload.counters.llm_attempts;
    for reservation in &mut payload.reservations {
        reservation.state = TaskEffectState::Completed;
        reservation.result = Some(serde_json::json!({"host_reconciled":true}));
    }
    payload.clocks.interrupted_interval_millis += 1;
    payload.clocks.active_interval_started_at = None;
    let safe = store
        .mutate(
            &running.key.run_id,
            &TaskRunMutation::ResolveRecovery {
                expected_revision: recovery.revision,
                payload: serde_json::to_value(&payload).unwrap(),
            },
        )
        .await
        .unwrap();
    assert!(agent.chat("unrelated").await.is_err());
    assert!(
        runner
            .acknowledge_recovery_release(&running.key.run_id, recovery.revision)
            .await
            .is_err()
    );
    let released = runner
        .acknowledge_recovery_release(&running.key.run_id, safe.revision)
        .await
        .unwrap();
    assert_eq!(released.status, TaskRunStatus::Paused);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    let after: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&running.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert_eq!(after.counters.llm_attempts, used);
    provider.release.as_ref().unwrap().add_permits(1);
    assert!(agent.chat("ordinary").await.is_ok());
    assert!(
        runner
            .acknowledge_recovery_release(&running.key.run_id, safe.revision)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn dropping_foreground_work_keeps_dispatched_marker_and_busy_runtime() {
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
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert!(agent.chat("unrelated").await.is_err());
    let listed = store.list().await.unwrap();
    let snapshot = store.load(&listed[0].key.run_id).await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
    assert!(
        payload
            .reservations
            .iter()
            .any(|r| r.state == TaskEffectState::Dispatched)
    );
}

#[tokio::test]
async fn unresolved_gate_exhaustion_uses_no_extra_summary_call() {
    let provider = RecordingProvider::new(&["working"]);
    let (_, runner, _) = fixture(provider.clone(), 1);
    let result = runner.run("objective", None).await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::LimitReached);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn old_messages_omit_provenance_without_changing_snapshot_readability() {
    let old = serde_json::json!({"role":"user","content":"legacy"});
    let message: ChatMessage = serde_json::from_value(old).unwrap();
    assert!(message.provenance.is_none());
    let serialized = serde_json::to_value(message).unwrap();
    assert!(serialized.get("provenance").is_none());
}

// Required future-stage validation must not block a preceding stage's own admission decision.
#[test]
fn stage_validation_does_not_require_future_stage_checks() {
    let mut extensions = AutonomyExtensions::builtins();
    struct Pure {
        descriptor: AdapterDescriptor,
    }
    impl AutonomyValidator for Pure {
        fn descriptor(&self) -> &AdapterDescriptor {
            &self.descriptor
        }
        fn validate_config(&self, _: &Value) -> Result<()> {
            Ok(())
        }
        fn evaluate(&self, _: &ValidationInput<'_>) -> Result<ValidationDecision> {
            Ok(ValidationDecision::Complete {
                outcome: GateOutcome::Pass,
                reason: "fixture".into(),
                metrics: serde_json::json!({}),
                evidence_refs: vec![],
            })
        }
    }
    extensions
        .register_validator(Arc::new(Pure {
            descriptor: AdapterDescriptor {
                id: "host.pure".into(),
                contract_version: 1,
                ..Default::default()
            },
        }))
        .unwrap();
    let profile: AutonomyProfile = serde_yaml::from_str("lifecycle:\n  - id: explore\n  - id: validate\n    validation:\n      checks:\n        - id: final\n          adapter: host.pure\n          contract_version: 1\n          required: true\n").unwrap();
    let bound = extensions
        .freeze()
        .bind(&profile, &ValidationCapabilities::default())
        .unwrap();
    let mut scope = super::evaluation_tests::scope();
    scope.stage = Some("explore".into());
    bound.prepare_scope(&mut scope);
    assert_eq!(
        bound
            .required_stage_outcome(&scope, &EvaluationEvidence::default())
            .unwrap(),
        GateOutcome::Pass
    );
    assert_eq!(
        bound
            .required_outcome(&scope, &EvaluationEvidence::default())
            .unwrap(),
        GateOutcome::Unknown
    );
}

use super::*;
use crate::AgentBuilder;
use ai_agents_core::{AgentError, Result};
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;

struct Fixture {
    agent: Arc<crate::RuntimeAgent>,
    runner: AutonomyRunner,
    store: Arc<ScopedTaskRunStore>,
    provider: Arc<runner_tests::RecordingProvider>,
    config: AutonomyConfig,
    host: AutonomyHostCeilings,
}

// The third reply requires a new controller turn beyond the original profile's turn and provider limits.
async fn fixture(sqlite: bool) -> Fixture {
    let question = ai_agents_core::encode_native_tool_call_markers(&[ai_agents_core::ToolCall {
        id: "question".into(),
        name: "ask_user".into(),
        arguments: json!({"question":"Choose","options":["yes"],"allow_other":false,"timeout_seconds":30}),
    }], None).unwrap();
    let provider = runner_tests::RecordingProvider::new(&[&question, "work remains", "done"]);
    let agent = Arc::new(
        AgentBuilder::from_yaml("name: LimitRevision\nsystem_prompt: test\ntools: [ask_user]\n")
            .unwrap()
            .llm(provider.clone())
            .auto_configure_features()
            .unwrap()
            .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
                Arc::new(parking_lot::RwLock::new(None)),
            )))
            .build()
            .unwrap(),
    );
    let store = if sqlite {
        #[cfg(feature = "sqlite")]
        {
            Arc::new(
                ScopedTaskRunStore::new(
                    Arc::new(ai_agents_storage::SqliteStorage::in_memory().await.unwrap()),
                    agent.info().id,
                    None,
                    "limits.v1".into(),
                )
                .unwrap(),
            )
        }
        #[cfg(not(feature = "sqlite"))]
        {
            panic!("SQLite fixture requires its feature")
        }
    } else {
        Arc::new(ScopedTaskRunStore::in_memory(agent.info().id, None, "limits.v1".into()).unwrap())
    };
    let mut config = runner_tests::config(2);
    config.defaults.max_turns = Some(1);
    config.defaults.max_wall_time_seconds = Some(120);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_user_question: Some(InteractionAction::PauseRun),
        on_approval_required: Some(InteractionAction::PauseRun),
    });
    let host = AutonomyHostCeilings {
        max_turns: Some(3),
        max_llm_calls: Some(4),
        max_wall_time_seconds: Some(240),
        ..Default::default()
    };
    let runner = AutonomyRunner::try_new(
        agent.clone(),
        config.clone(),
        host.clone(),
        store.clone(),
        "limits.v1".into(),
    )
    .unwrap();
    Fixture {
        agent,
        runner,
        store,
        provider,
        config,
        host,
    }
}

// The original question binding survives a numeric revision and is answered only by an explicit resume.
fn answer(run: &TaskRun) -> TaskResumeInput {
    TaskResumeInput::UserAnswer {
        request_id: run.pending.as_ref().unwrap().id.clone(),
        answer: json!({"answered":true,"selected":["yes"]}),
    }
}

// Admission and controller scheduling must agree on revised limits without resetting consumed work or question expiry.
async fn revise_and_complete(sqlite: bool) {
    let f = Box::pin(fixture(sqlite)).await;
    let paused = f.runner.run("objective", None).await.unwrap();
    let original = f.store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    let before: TaskCheckpointPayload = serde_json::from_value(original.payload.clone()).unwrap();
    let mut limits = before.limits.clone();
    limits.max_turns = 3;
    limits.max_llm_calls = 4;
    limits.max_wall_time_seconds = Some(240);
    let revised = f
        .runner
        .revise_limits_paused(
            &paused.run.key.run_id,
            paused.run.revision,
            limits.clone(),
            "Finish existing work",
        )
        .await
        .unwrap();
    assert_eq!(revised.run.status, TaskRunStatus::Paused);
    assert_eq!(revised.run.revision, paused.run.revision + 1);
    assert_eq!(
        serde_json::to_value(&revised.run.pending).unwrap(),
        serde_json::to_value(&paused.run.pending).unwrap()
    );
    assert_eq!(
        f.provider.calls.load(std::sync::atomic::Ordering::SeqCst),
        1
    );
    assert!(f.agent.set_context("unrelated", json!(true)).is_err());
    let saved = f.store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    let after: TaskCheckpointPayload = serde_json::from_value(saved.payload.clone()).unwrap();
    assert_eq!(after.limits, limits);
    assert_eq!(
        serde_json::to_value(&after.counters).unwrap(),
        serde_json::to_value(&before.counters).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&after.adapters).unwrap(),
        serde_json::to_value(&before.adapters).unwrap()
    );
    assert_eq!(
        after.clocks.expires_at,
        Some(original.created_at + chrono::Duration::seconds(240))
    );
    assert_eq!(
        after.evidence.records.last().unwrap().origin,
        "host.control"
    );
    assert!(
        f.runner
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                answer(&paused.run)
            )
            .await
            .is_err()
    );
    let mut invalid = limits;
    invalid.max_llm_calls = 0;
    assert!(
        f.runner
            .revise_limits_paused(
                &paused.run.key.run_id,
                revised.run.revision,
                invalid,
                "Below consumed usage"
            )
            .await
            .is_err()
    );
    assert_eq!(
        f.store
            .load(&paused.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .revision,
        saved.revision
    );
    let done = f
        .runner
        .resume(
            &revised.run.key.run_id,
            revised.run.revision,
            answer(&revised.run),
        )
        .await
        .unwrap();
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    assert_eq!(done.run.counters.turns, 2);
    assert_eq!(done.run.counters.llm_attempts, 3);
    assert_eq!(
        f.provider.calls.load(std::sync::atomic::Ordering::SeqCst),
        3
    );
    assert!(f.agent.set_context("released", json!(true)).is_ok());
}

// A retained in-memory runner can extend capacity without consuming or replacing its pending question.
#[tokio::test]
async fn paused_limits_memory_preserves_question_and_updates_controller_capacity() {
    revise_and_complete(false).await;
}

// SQLite uses the same validated CAS, not a separate executable restore path.
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn paused_limits_sqlite_preserves_question_and_updates_controller_capacity() {
    revise_and_complete(true).await;
}

#[derive(Clone, Copy)]
enum Receipt {
    Hold,
    NoWrite,
    Stale,
    Lost,
}

struct AckStore {
    inner: Arc<ScopedTaskRunStore>,
    supported: bool,
    receipt: Receipt,
    entered: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
}

#[async_trait]
impl TaskRunStore for AckStore {
    // Transparent fixture operations keep normal claims and checkpoints unchanged.
    async fn create(&self, snapshot: &TaskRunSnapshot) -> Result<()> {
        self.inner.create(snapshot).await
    }
    // Expose the actual backend revision even while acknowledgement is held.
    async fn load(&self, run_id: &str) -> Result<Option<TaskRunSnapshot>> {
        self.inner.load(run_id).await
    }
    // Ordinary mutations never gain control-plane authority.
    async fn mutate(&self, run_id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot> {
        self.inner.mutate(run_id, mutation).await
    }
    // An unsupported custom adapter must be rejected before arming ambiguous-write protection.
    fn supports_paused_limit_revision(&self) -> bool {
        self.supported
    }
    // The gate opens only after the real host-control CAS commits.
    async fn revise_paused_limits(
        &self,
        run_id: &str,
        revision: u64,
        limits: TaskRunLimits,
        reason: &str,
        host: &AutonomyHostCeilings,
        deadline: Option<std::time::Instant>,
    ) -> Result<Option<TaskRunSnapshot>> {
        if !self.supported {
            return Err(AgentError::Config("unsupported".into()));
        }
        if matches!(self.receipt, Receipt::NoWrite) {
            return Ok(None);
        }
        let mut saved = self
            .inner
            .revise_paused_limits(run_id, revision, limits, reason, host, deadline)
            .await?;
        if matches!(self.receipt, Receipt::Lost) {
            return Err(AgentError::Persistence(
                "limit revision acknowledgement lost".into(),
            ));
        }
        if matches!(self.receipt, Receipt::Stale) {
            saved.as_mut().unwrap().revision = revision;
            return Ok(saved);
        }
        self.entered.add_permits(1);
        self.release.acquire().await.unwrap().forget();
        Ok(saved)
    }
    // Metadata stays scoped to the same backend.
    async fn list(&self) -> Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    // Deletion retains the real ownership guard.
    async fn delete(&self, run_id: &str, revision: u64) -> Result<()> {
        self.inner.delete(run_id, revision).await
    }
}

// A dropped control acknowledgement never installs usable new live authority even if the backend still says Paused.
#[tokio::test]
async fn paused_limits_dropped_acknowledgement_preserves_protection() {
    let mut f = Box::pin(fixture(false)).await;
    let store = Arc::new(AckStore {
        inner: f.store.clone(),
        supported: true,
        receipt: Receipt::Hold,
        entered: tokio::sync::Semaphore::new(0),
        release: tokio::sync::Semaphore::new(0),
    });
    f.runner = AutonomyRunner::try_new(
        f.agent.clone(),
        f.config.clone(),
        f.host.clone(),
        store.clone(),
        "limits.v1".into(),
    )
    .unwrap();
    let paused = f.runner.run("objective", None).await.unwrap();
    let snapshot = f.store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
    let mut limits = payload.limits;
    limits.max_turns = 3;
    let mut control = Box::pin(f.runner.revise_limits_paused(
        &paused.run.key.run_id,
        paused.run.revision,
        limits,
        "Extend work",
    ));
    tokio::select! {
        result = &mut control => panic!("control returned before acknowledgement: {result:?}"),
        permit = tokio::time::timeout(std::time::Duration::from_secs(5), store.entered.acquire()) => { permit.unwrap().unwrap().forget(); }
    }
    drop(control);
    let saved = f.store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    assert_eq!(saved.status, TaskRunStatus::Paused);
    assert_eq!(saved.revision, paused.run.revision + 1);
    assert!(
        f.runner
            .resume(&paused.run.key.run_id, saved.revision, answer(&paused.run))
            .await
            .is_err()
    );
    assert!(f.agent.set_context("unrelated", json!(true)).is_err());
    assert_eq!(
        f.provider.calls.load(std::sync::atomic::Ordering::SeqCst),
        1
    );
}

// A proven non-writing rejection preserves the original owner, but a stale success receipt requires recovery protection.
#[tokio::test]
async fn paused_limits_distinguishes_no_write_from_stale_acknowledgement() {
    for receipt in [Receipt::NoWrite, Receipt::Stale, Receipt::Lost] {
        let mut f = Box::pin(fixture(false)).await;
        let store = Arc::new(AckStore {
            inner: f.store.clone(),
            supported: true,
            receipt,
            entered: tokio::sync::Semaphore::new(0),
            release: tokio::sync::Semaphore::new(0),
        });
        f.runner = AutonomyRunner::try_new(
            f.agent.clone(),
            f.config.clone(),
            f.host.clone(),
            store,
            "limits.v1".into(),
        )
        .unwrap();
        let paused = f.runner.run("objective", None).await.unwrap();
        let snapshot = f.store.load(&paused.run.key.run_id).await.unwrap().unwrap();
        let payload: TaskCheckpointPayload =
            serde_json::from_value(snapshot.payload.clone()).unwrap();
        assert!(
            f.runner
                .revise_limits_paused(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    payload.limits,
                    "Review receipt"
                )
                .await
                .is_err()
        );
        let saved = f.store.load(&paused.run.key.run_id).await.unwrap().unwrap();
        if matches!(receipt, Receipt::NoWrite) {
            assert_eq!(saved.revision, snapshot.revision);
            assert_eq!(saved.payload, snapshot.payload);
            let done = f
                .runner
                .resume(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    answer(&paused.run),
                )
                .await
                .unwrap();
            assert_eq!(done.run.status, TaskRunStatus::LimitReached);
            assert!(f.agent.set_context("released", json!(true)).is_ok());
        } else {
            assert_eq!(saved.revision, snapshot.revision + 1);
            assert!(
                f.runner
                    .resume(&paused.run.key.run_id, saved.revision, answer(&paused.run))
                    .await
                    .is_err()
            );
            assert!(f.agent.set_context("unrelated", json!(true)).is_err());
            assert_eq!(
                f.provider.calls.load(std::sync::atomic::Ordering::SeqCst),
                1
            );
        }
    }
}

// Backend admission uses the original bounds, never the extended expiry inside the proposed payload.
#[tokio::test]
async fn paused_limits_host_control_rejects_expired_original_admission() {
    let f = Box::pin(fixture(false)).await;
    let paused = f.runner.run("objective", None).await.unwrap();
    let snapshot = f.store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    let mut payload: TaskCheckpointPayload =
        serde_json::from_value(snapshot.payload.clone()).unwrap();
    payload.clocks.expires_at = Some(snapshot.created_at + chrono::Duration::seconds(240));
    for (deadline, expires_at) in [
        (Some(std::time::Instant::now()), payload.clocks.expires_at),
        (None, Some(chrono::Utc::now())),
    ] {
        let mutation = TaskRunMutation::HostControl {
            expected_revision: snapshot.revision,
            owner_token: None,
            status: TaskRunStatus::Paused,
            payload: serde_json::to_value(&payload).unwrap(),
            deadline,
            expires_at,
        };
        assert!(snapshot.transition(&mutation, chrono::Utc::now()).is_err());
        assert!(
            mutation
                .validate_final_deadline(chrono::Utc::now())
                .is_err()
        );
    }
    assert_eq!(
        f.store
            .load(&paused.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .revision,
        snapshot.revision
    );
    f.runner
        .cancel_paused(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap();
}

// Audit growth can exceed checkpoint capacity before writing; that rejection must not abandon the original live pause.
#[tokio::test]
async fn paused_limits_audit_capacity_rejection_preserves_original_owner() {
    let mut f = Box::pin(fixture(false)).await;
    let backend = Arc::new(ai_agents_storage::InMemoryTaskStorage::default());
    f.store = Arc::new(
        ScopedTaskRunStore::new(backend.clone(), f.agent.info().id, None, "limits.v1".into())
            .unwrap(),
    );
    f.runner = AutonomyRunner::try_new(
        f.agent.clone(),
        f.config.clone(),
        f.host.clone(),
        f.store.clone(),
        "limits.v1".into(),
    )
    .unwrap();
    let paused = f.runner.run("objective", None).await.unwrap();
    let original = f.store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    let base: TaskCheckpointPayload = serde_json::from_value(original.payload.clone()).unwrap();
    let pad = |extra: usize| {
        let mut payload = base.clone();
        for index in 0..extra {
            payload.evidence.records.push(TaskEvidenceRecord {
                run_id: original.key.run_id.clone(),
                sequence: 10_000 + index as u64,
                stage: payload.stage.clone(),
                attempt: None,
                origin: "host.fixture".into(),
                data: json!(null),
            });
        }
        payload
    };
    let (mut low, mut high) = (0, MAX_TASK_CHECKPOINT_RECORDS + 1);
    while low + 1 < high {
        let mid = low + (high - low) / 2;
        if pad(mid).validate(&original, "limits.v1").is_ok() {
            low = mid;
        } else {
            high = mid;
        }
    }
    let full = pad(low);
    let filled = ai_agents_core::AgentStorage::mutate_task_run(
        backend.as_ref(),
        &original.key,
        &TaskRunMutation::HostControl {
            expected_revision: original.revision,
            owner_token: None,
            status: TaskRunStatus::Paused,
            payload: serde_json::to_value(&full).unwrap(),
            deadline: None,
            expires_at: base.clocks.expires_at,
        },
    )
    .await
    .unwrap();
    assert!(
        f.runner
            .revise_limits_paused(
                &paused.run.key.run_id,
                filled.revision,
                full.limits,
                "Audit exceeds capacity"
            )
            .await
            .is_err()
    );
    let saved = f.store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    assert_eq!(saved.revision, filled.revision);
    assert_eq!(saved.payload, filled.payload);
    // Remove only synthetic padding before cleanup; failure to preserve the live owner would still reject cancellation.
    let restored = ai_agents_core::AgentStorage::mutate_task_run(
        backend.as_ref(),
        &original.key,
        &TaskRunMutation::HostControl {
            expected_revision: filled.revision,
            owner_token: None,
            status: TaskRunStatus::Paused,
            payload: original.payload,
            deadline: None,
            expires_at: base.clocks.expires_at,
        },
    )
    .await
    .unwrap();
    let cancelled = f
        .runner
        .cancel_paused(&paused.run.key.run_id, restored.revision)
        .await
        .unwrap();
    assert_eq!(cancelled.run.status, TaskRunStatus::Cancelled);
    assert_eq!(
        f.provider.calls.load(std::sync::atomic::Ordering::SeqCst),
        1
    );
    assert!(f.agent.set_context("released", json!(true)).is_ok());
}

// Capability refusal must leave the original pause usable instead of marking its live owner abandoned.
#[tokio::test]
async fn paused_limits_unsupported_store_preserves_original_pause() {
    let mut f = Box::pin(fixture(false)).await;
    let store = Arc::new(AckStore {
        inner: f.store.clone(),
        supported: false,
        receipt: Receipt::Hold,
        entered: tokio::sync::Semaphore::new(0),
        release: tokio::sync::Semaphore::new(0),
    });
    f.runner = AutonomyRunner::try_new(
        f.agent.clone(),
        f.config.clone(),
        f.host.clone(),
        store,
        "limits.v1".into(),
    )
    .unwrap();
    let paused = f.runner.run("objective", None).await.unwrap();
    let snapshot = f.store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
    assert!(
        f.runner
            .revise_limits_paused(
                &paused.run.key.run_id,
                paused.run.revision,
                payload.limits,
                "Unsupported"
            )
            .await
            .is_err()
    );
    assert_eq!(
        f.store
            .load(&paused.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .revision,
        paused.run.revision
    );
    let done = f
        .runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            answer(&paused.run),
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::LimitReached);
    assert!(f.agent.set_context("released", json!(true)).is_ok());
}

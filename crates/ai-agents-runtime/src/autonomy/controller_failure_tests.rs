use super::controller_tests::fixture;
use super::*;

#[path = "controller_response_tests.rs"]
mod response_tests;
use crate::Agent;
use ai_agents_core::{AgentError, Result};
use async_trait::async_trait;
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

// Failure injection uses the real scoped store and can lose acknowledgement after an accepted write.
struct AckStore {
    inner: Arc<ScopedTaskRunStore>,
    fail_pause: bool,
    fail_accept: AtomicBool,
    delay_accept: bool,
    delay_claim: AtomicBool,
    entered: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
}
impl AckStore {
    fn new(inner: Arc<ScopedTaskRunStore>, fail_pause: bool, delay_accept: bool) -> Arc<Self> {
        Arc::new(Self {
            inner,
            fail_pause,
            fail_accept: AtomicBool::new(false),
            delay_accept,
            delay_claim: AtomicBool::new(false),
            entered: tokio::sync::Semaphore::new(0),
            release: tokio::sync::Semaphore::new(0),
        })
    }
}
#[async_trait]
impl TaskRunStore for AckStore {
    async fn create(&self, snapshot: &TaskRunSnapshot) -> Result<()> {
        self.inner.create(snapshot).await
    }
    async fn load(&self, run_id: &str) -> Result<Option<TaskRunSnapshot>> {
        self.inner.load(run_id).await
    }
    async fn list(&self) -> Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    async fn delete(&self, run_id: &str, revision: u64) -> Result<()> {
        self.inner.delete(run_id, revision).await
    }
    async fn mutate(&self, run_id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot> {
        if self.fail_pause
            && matches!(
                mutation,
                TaskRunMutation::FinalCheckpoint {
                    status: TaskRunStatus::Paused,
                    ..
                }
            )
        {
            return Err(AgentError::Persistence(
                "pause acknowledgement failed".into(),
            ));
        }
        let acceptance = match mutation {
            TaskRunMutation::Checkpoint { payload, .. } => payload
                .get("adapters")
                .and_then(|a| a.as_array())
                .is_some_and(|adapters| {
                    adapters.iter().any(|adapter| {
                        adapter
                            .get("state")
                            .and_then(|s| s.get("_task_validation_driver"))
                            .and_then(|s| s.get("external_events"))
                            .and_then(|events| events.as_object())
                            .is_some_and(|events| !events.is_empty())
                    })
                }),
            _ => false,
        };
        if acceptance && self.fail_accept.load(Ordering::SeqCst) {
            return Err(AgentError::Persistence(
                "signal acceptance acknowledgement failed".into(),
            ));
        }
        let saved = self.inner.mutate(run_id, mutation).await?;
        if (acceptance && self.delay_accept)
            || (matches!(mutation, TaskRunMutation::Claim { .. })
                && self.delay_claim.swap(false, Ordering::SeqCst))
        {
            self.entered.add_permits(1);
            self.release.acquire().await.unwrap().forget();
        }
        Ok(saved)
    }
}

#[tokio::test]
async fn controller_failed_pause_acknowledgement_never_exposes_pause_or_releases_owner() {
    for mode in ["external", "approval"] {
        let mut f = fixture(mode);
        let failing = AckStore::new(f.store.clone(), true, false);
        f.runner = AutonomyRunner::try_new(
            f.agent.clone(),
            f.config,
            Default::default(),
            failing,
            "controller.v1".into(),
        )
        .unwrap();
        assert!(f.runner.run("objective", None).await.is_err());
        assert_eq!(f.calls.load(Ordering::SeqCst), 0);
        assert_eq!(f.callbacks.load(Ordering::SeqCst), 1);
        assert!(f.agent.chat("unrelated").await.is_err());
        assert!(
            f.store
                .list()
                .await
                .unwrap()
                .iter()
                .all(|run| run.status != TaskRunStatus::Paused)
        );
    }
}

#[tokio::test]
async fn controller_failed_signal_acceptance_never_reenters_callback() {
    let mut f = fixture("external");
    let failing = AckStore::new(f.store.clone(), false, false);
    f.runner = AutonomyRunner::try_new(
        f.agent.clone(),
        f.config,
        Default::default(),
        failing.clone(),
        "controller.v1".into(),
    )
    .unwrap();
    let paused = f.runner.run("objective", None).await.unwrap();
    failing.fail_accept.store(true, Ordering::SeqCst);
    assert!(
        f.runner
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::ExternalSignal(ExternalSignal {
                    request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                    event_id: "event".into(),
                    target_revision: None,
                    payload: json!(true),
                })
            )
            .await
            .is_err()
    );
    assert_eq!(f.callbacks.load(Ordering::SeqCst), 1);
    assert!(f.agent.chat("unrelated").await.is_err());
}

#[tokio::test]
async fn controller_dropped_signal_acknowledgement_keeps_effect_and_callback_protection() {
    let mut f = fixture("external");
    let delaying = AckStore::new(f.store.clone(), false, true);
    f.runner = AutonomyRunner::try_new(
        f.agent.clone(),
        f.config,
        Default::default(),
        delaying.clone(),
        "controller.v1".into(),
    )
    .unwrap();
    let runner = Arc::new(f.runner);
    let paused = runner.run("objective", None).await.unwrap();
    let worker = runner.clone();
    let task = tokio::spawn(async move {
        worker
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::ExternalSignal(ExternalSignal {
                    request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                    event_id: "event".into(),
                    target_revision: None,
                    payload: json!(true),
                }),
            )
            .await
    });
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        delaying.entered.acquire(),
    )
    .await
    .unwrap()
    .unwrap()
    .forget();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert_eq!(f.callbacks.load(Ordering::SeqCst), 1);
    assert!(f.agent.chat("unrelated").await.is_err());
}

#[tokio::test]
async fn controller_expired_wall_wait_can_be_cancelled_without_callback_or_tool() {
    for mode in ["external", "approval", "question"] {
        let mut f = fixture(mode);
        f.config.defaults.max_wall_time_seconds = Some(1);
        f.runner = AutonomyRunner::try_new(
            f.agent.clone(),
            f.config,
            Default::default(),
            f.store.clone(),
            "controller.v1".into(),
        )
        .unwrap();
        let paused = f.runner.run("objective", None).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        let done = f
            .runner
            .cancel_paused(&paused.run.key.run_id, paused.run.revision)
            .await
            .unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Cancelled);
        assert_eq!(f.calls.load(Ordering::SeqCst), 0);
        assert_eq!(f.callbacks.load(Ordering::SeqCst), 1);
        assert_eq!(f.agent.chat("follow-up").await.unwrap().content, "ordinary");
    }
}

#[tokio::test]
async fn controller_expired_check_rejects_response_before_claim_and_remains_cancellable() {
    for mode in ["external", "approval"] {
        let mut f = fixture(mode);
        let config = "checks: [{id: check, adapter: host.observation, schedule: each_cycle, timeout_seconds: 1, config: {}}]";
        f.config.defaults.validation = Some(serde_yaml::from_str(config).unwrap());
        f.runner = AutonomyRunner::try_new(
            f.agent.clone(),
            f.config,
            Default::default(),
            f.store.clone(),
            "controller.v1".into(),
        )
        .unwrap();
        let paused = f.runner.run("objective", None).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        let id = paused.run.pending.as_ref().unwrap().id.clone();
        let input = if mode == "external" {
            TaskResumeInput::ExternalSignal(ExternalSignal {
                request_id: id,
                event_id: "event".into(),
                target_revision: None,
                payload: json!(true),
            })
        } else {
            TaskResumeInput::Approval {
                request_id: id,
                result: ai_agents_hitl::ApprovalResult::Approved,
            }
        };
        assert!(
            f.runner
                .resume(&paused.run.key.run_id, paused.run.revision, input)
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
            .cancel_paused(&paused.run.key.run_id, paused.run.revision)
            .await
            .unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Cancelled);
        assert_eq!(f.calls.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn controller_repeated_external_wait_rejects_duplicate_event_before_claim() {
    let f = fixture("external_repeat");
    let first = f.runner.run("objective", None).await.unwrap();
    let second = f
        .runner
        .resume(
            &first.run.key.run_id,
            first.run.revision,
            TaskResumeInput::ExternalSignal(ExternalSignal {
                request_id: first.run.pending.as_ref().unwrap().id.clone(),
                event_id: "event".into(),
                target_revision: None,
                payload: json!(false),
            }),
        )
        .await
        .unwrap();
    assert_eq!(second.run.status, TaskRunStatus::Paused);
    let mut signal = ExternalSignal {
        request_id: second.run.pending.as_ref().unwrap().id.clone(),
        event_id: "event".into(),
        target_revision: None,
        payload: json!(true),
    };
    assert!(
        f.runner
            .resume(
                &second.run.key.run_id,
                second.run.revision,
                TaskResumeInput::ExternalSignal(signal.clone())
            )
            .await
            .is_err()
    );
    assert_eq!(
        f.store
            .load(&second.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .revision,
        second.run.revision
    );
    signal.event_id = "new_event".into();
    let done = f
        .runner
        .resume(
            &second.run.key.run_id,
            second.run.revision,
            TaskResumeInput::ExternalSignal(signal),
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(done.run.counters.turns, 1);
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn controller_sqlite_signal_resume_uses_retained_owner_without_new_turn() {
    let mut f = fixture("external");
    let store = Arc::new(
        ScopedTaskRunStore::new(
            Arc::new(ai_agents_storage::SqliteStorage::in_memory().await.unwrap()),
            f.agent.info().id,
            None,
            "controller.v1".into(),
        )
        .unwrap(),
    );
    f.runner = AutonomyRunner::try_new(
        f.agent.clone(),
        f.config,
        Default::default(),
        store,
        "controller.v1".into(),
    )
    .unwrap();
    let paused = f.runner.run("objective", None).await.unwrap();
    let done = f
        .runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::ExternalSignal(ExternalSignal {
                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                event_id: "event".into(),
                target_revision: None,
                payload: json!(true),
            }),
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(done.run.counters.turns, 1);
}

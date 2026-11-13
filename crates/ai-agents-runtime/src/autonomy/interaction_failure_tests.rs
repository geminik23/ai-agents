use super::*;
use crate::Agent;
use ai_agents_core::{AgentError, Result};
use async_trait::async_trait;

struct PauseStore {
    inner: Arc<ScopedTaskRunStore>,
    fail_pause: bool,
    gate_claim: bool,
    entered: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
}
#[async_trait]
impl TaskRunStore for PauseStore {
    async fn create(&self, snapshot: &TaskRunSnapshot) -> Result<()> {
        self.inner.create(snapshot).await
    }
    async fn load(&self, id: &str) -> Result<Option<TaskRunSnapshot>> {
        self.inner.load(id).await
    }
    async fn list(&self) -> Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    async fn delete(&self, id: &str, revision: u64) -> Result<()> {
        self.inner.delete(id, revision).await
    }
    async fn mutate(&self, id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot> {
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
                "interaction pause acknowledgement failed".into(),
            ));
        }
        let saved = self.inner.mutate(id, mutation).await?;
        if self.gate_claim && matches!(mutation, TaskRunMutation::Claim { .. }) {
            self.entered.add_permits(1);
            self.release.acquire().await.unwrap().forget();
        }
        Ok(saved)
    }
}
fn injected(yaml: &str, fail: bool, gate: bool) -> (Fixture, Arc<PauseStore>) {
    let mut f = fixture(
        yaml,
        &["done"],
        if yaml == CLARIFICATION {
            &[AMBIGUOUS, QUESTION, ANSWER]
        } else {
            &[]
        },
    );
    let store = Arc::new(PauseStore {
        inner: f.store.clone(),
        fail_pause: fail,
        gate_claim: gate,
        entered: tokio::sync::Semaphore::new(0),
        release: tokio::sync::Semaphore::new(0),
    });
    let mut config = super::super::runner_tests::config(16);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    f.runner = AutonomyRunner::try_new(
        f.agent.clone(),
        config,
        Default::default(),
        store.clone(),
        "interaction.v1".into(),
    )
    .unwrap();
    (f, store)
}

#[tokio::test]
async fn interaction_failed_pause_acknowledgement_retains_owner_without_finalization() {
    for yaml in [STATE, CLARIFICATION] {
        let (f, _) = injected(yaml, true, false);
        assert!(f.runner.run("objective", None).await.is_err());
        assert_eq!(f.finalized.load(Ordering::SeqCst), 0);
        assert_eq!(f.effects.load(Ordering::SeqCst), 0);
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
async fn interaction_drop_after_claim_does_not_authorize_another_root() {
    for yaml in [STATE, CLARIFICATION] {
        let (f, store) = injected(yaml, false, true);
        let runner = Arc::new(f.runner);
        let paused = runner.run("objective", None).await.unwrap();
        let worker = runner.clone();
        let input = if yaml == STATE {
            TaskResumeInput::Approval {
                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                result: ApprovalResult::Approved,
            }
        } else {
            TaskResumeInput::UserAnswer {
                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                answer: json!("target one"),
            }
        };
        let task = tokio::spawn(async move {
            worker
                .resume(&paused.run.key.run_id, paused.run.revision, input)
                .await
        });
        tokio::time::timeout(std::time::Duration::from_secs(5), store.entered.acquire())
            .await
            .unwrap()
            .unwrap()
            .forget();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert_eq!(f.finalized.load(Ordering::SeqCst), 0);
        assert_eq!(f.effects.load(Ordering::SeqCst), 0);
        assert!(f.agent.chat("unrelated").await.is_err());
        if yaml == CLARIFICATION {
            assert_eq!(f.router.calls.load(Ordering::SeqCst), 2);
        }
    }
}

#[tokio::test]
async fn interaction_transition_modified_or_stale_response_rejects_before_claim() {
    let f = fixture(STATE, &["done"], &[]);
    let first = f.runner.run("objective", None).await.unwrap();
    for result in [
        ApprovalResult::Modified {
            changes: std::collections::HashMap::from([("to_state".into(), json!("other"))]),
        },
        ApprovalResult::Rejected {
            reason: Some("x".repeat(70_000)),
        },
    ] {
        assert!(
            f.runner
                .resume(
                    &first.run.key.run_id,
                    first.run.revision,
                    TaskResumeInput::Approval {
                        request_id: first.run.pending.as_ref().unwrap().id.clone(),
                        result
                    }
                )
                .await
                .is_err()
        );
        assert_eq!(
            f.store
                .load(&first.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .revision,
            first.run.revision
        );
    }
    let done = respond(&f, &first.run, ApprovalResult::Approved).await;
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert!(
        f.runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::Approval {
                    request_id: first.run.pending.as_ref().unwrap().id.clone(),
                    result: ApprovalResult::Approved
                }
            )
            .await
            .is_err()
    );
    assert_eq!(f.agent.state_history().len(), 1);
    assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn interaction_transition_expiry_closes_without_extra_selection_or_effects() {
    let yaml = STATE.replace(
        "hitl:\n",
        "hitl:\n  default_timeout_seconds: 1\n  on_timeout: reject\n",
    );
    let f = fixture(&yaml, &["done"], &[]);
    let first = f.runner.run("objective", None).await.unwrap();
    let id = first.run.pending.as_ref().unwrap().id.clone();
    assert!(
        f.runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::Approval {
                    request_id: id.clone(),
                    result: ApprovalResult::Timeout
                }
            )
            .await
            .is_err()
    );
    assert_eq!(
        f.store
            .load(&first.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .revision,
        first.run.revision
    );
    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
    assert!(
        f.runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::Approval {
                    request_id: id.clone(),
                    result: ApprovalResult::Approved
                }
            )
            .await
            .is_err()
    );
    let done = f
        .runner
        .resume(
            &first.run.key.run_id,
            first.run.revision,
            TaskResumeInput::Approval {
                request_id: id,
                result: ApprovalResult::Timeout,
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(f.agent.current_state().as_deref(), Some("active"));
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn interaction_control_revocation_rejects_before_claim_but_safe_cancel_still_retires_exchange()
 {
    for yaml in [STATE, CLARIFICATION] {
        let f = fixture(
            yaml,
            &["done"],
            if yaml == CLARIFICATION {
                &[AMBIGUOUS, QUESTION]
            } else {
                &[]
            },
        );
        let first = f.runner.run("objective", None).await.unwrap();
        f.agent.runtime_control().set_tool_scope(Vec::new());
        let id = first.run.pending.as_ref().unwrap().id.clone();
        let input = if yaml == STATE {
            TaskResumeInput::Approval {
                request_id: id,
                result: ApprovalResult::Approved,
            }
        } else {
            TaskResumeInput::UserAnswer {
                request_id: id,
                answer: json!("target one"),
            }
        };
        assert!(
            f.runner
                .resume(&first.run.key.run_id, first.run.revision, input)
                .await
                .is_err()
        );
        assert_eq!(
            f.store
                .load(&first.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .revision,
            first.run.revision
        );
        let done = f
            .runner
            .cancel_paused(&first.run.key.run_id, first.run.revision)
            .await
            .unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Cancelled);
        assert_eq!(f.effects.load(Ordering::SeqCst), 0);
        assert_eq!(f.finalized.load(Ordering::SeqCst), 0);
        if yaml == CLARIFICATION {
            assert_eq!(f.router.calls.load(Ordering::SeqCst), 2);
        }
    }
}

#[tokio::test]
async fn interaction_host_stop_retires_pending_question_or_transition_without_execution() {
    for yaml in [STATE, CLARIFICATION] {
        let f = fixture(
            yaml,
            &["done"],
            if yaml == CLARIFICATION {
                &[AMBIGUOUS, QUESTION]
            } else {
                &[]
            },
        );
        let first = f.runner.run("objective", None).await.unwrap();
        assert!(
            f.runner
                .stop_paused(&first.run.key.run_id, first.run.revision, "")
                .await
                .is_err()
        );
        assert_eq!(
            f.store
                .load(&first.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .revision,
            first.run.revision
        );
        let done = f
            .runner
            .stop_paused(
                &first.run.key.run_id,
                first.run.revision,
                "Host stopped this task",
            )
            .await
            .unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Incomplete);
        assert_eq!(done.run.stop_reason.as_deref(), Some("user_stopped"));
        assert_eq!(f.finalized.load(Ordering::SeqCst), 0);
        assert_eq!(f.effects.load(Ordering::SeqCst), 0);
        assert!(f.agent.set_context("released", json!(true)).is_ok());
        let snapshot = f.store.load(&first.run.key.run_id).await.unwrap().unwrap();
        assert!(!snapshot.cancel_requested);
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
        assert!(
            payload
                .evidence
                .records
                .iter()
                .any(|record| record.origin == "host.control")
        );
    }
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn interaction_sqlite_clarification_retains_manager_and_revision_binding() {
    let mut f = fixture(CLARIFICATION, &["done"], &[AMBIGUOUS, QUESTION, ANSWER]);
    let store = Arc::new(
        ScopedTaskRunStore::new(
            Arc::new(ai_agents_storage::SqliteStorage::in_memory().await.unwrap()),
            f.agent.info().id,
            None,
            "interaction.v1".into(),
        )
        .unwrap(),
    );
    let mut config = super::super::runner_tests::config(16);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    f.runner = AutonomyRunner::try_new(
        f.agent.clone(),
        config,
        Default::default(),
        store,
        "interaction.v1".into(),
    )
    .unwrap();
    let first = f.runner.run("objective", None).await.unwrap();
    let done = f
        .runner
        .resume(
            &first.run.key.run_id,
            first.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: first.run.pending.as_ref().unwrap().id.clone(),
                answer: json!("target one"),
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(f.router.calls.load(Ordering::SeqCst), 3);
    assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
}

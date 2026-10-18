use super::*;
#[path = "host_evidence_acceptance_tests.rs"]
mod acceptance;
use crate::Agent;
use ai_agents_core::{AgentError, Result};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Default)]
pub(super) struct Revision {
    value: parking_lot::Mutex<String>,
    unavailable: AtomicBool,
    reads: AtomicUsize,
}
impl Revision {
    pub(super) fn new() -> Arc<Self> {
        Arc::new(Self {
            value: parking_lot::Mutex::new("r1".into()),
            ..Default::default()
        })
    }
    pub(super) fn change(&self) {
        *self.value.lock() = "r2".into();
    }
}
impl HostRevisionSource for Revision {
    fn revision(&self) -> Result<String> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        if self.unavailable.load(Ordering::SeqCst) {
            return Err(AgentError::Other("host revision unavailable".into()));
        }
        Ok(self.value.lock().clone())
    }
}

struct Validator {
    descriptor: AdapterDescriptor,
    calls: Arc<AtomicUsize>,
    mode: &'static str,
}
impl AutonomyValidator for Validator {
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.descriptor
    }
    fn validate_config(&self, _: &Value) -> Result<()> {
        Ok(())
    }
    fn evaluate(&self, input: &ValidationInput<'_>) -> Result<ValidationDecision> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.mode == "batch" && input.observations.len() < 2 {
            return Ok(ValidationDecision::NeedObservations {
                requests: ["one", "two"]
                    .into_iter()
                    .map(|id| ValidationObservationRequest::Tool {
                        id: id.into(),
                        tool: "probe".into(),
                        arguments: json!({"value": id}),
                        host_binding: None,
                    })
                    .collect(),
                checkpoint: json!({}),
            });
        }
        if self.mode == "fallback" && input.observations.is_empty() {
            return Ok(ValidationDecision::NeedObservations {
                requests: vec![ValidationObservationRequest::Tool {
                    id: "fallback".into(),
                    tool: "fail_probe".into(),
                    arguments: json!({"value":1}),
                    host_binding: None,
                }],
                checkpoint: json!({}),
            });
        }
        if self.mode != "complete"
            && (input.previous.get("external_signal").is_none()
                || (self.mode == "repeat"
                    && input.previous.get("external_signal") == Some(&json!(false))))
        {
            return Ok(ValidationDecision::AwaitExternal {
                binding: "ready".into(),
                checkpoint: json!({"saved":true}),
            });
        }
        Ok(ValidationDecision::Complete {
            outcome: GateOutcome::Pass,
            reason: "bound host evidence".into(),
            metrics: json!({"precise":0.15045128665949425}),
            evidence_refs: vec![],
        })
    }
}

fn configured(
    source: Arc<Revision>,
    mode: &'static str,
    policy: ExternalUnavailablePolicy,
) -> (super::controller_tests::Fixture, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let count = calls.clone();
    let f = super::controller_tests::fixture_configured(mode, move |extensions| {
        extensions
            .replace_validator(Arc::new(Validator {
                descriptor: AdapterDescriptor {
                    id: "host.observation".into(),
                    contract_version: 1,
                    requires_host_revision: true,
                    conditions: vec!["ready".into()],
                    tools: match mode {
                        "batch" => vec!["probe".into()],
                        "fallback" => vec!["fail_probe".into()],
                        _ => vec![],
                    },
                    ..Default::default()
                },
                calls: count,
                mode,
            }))
            .unwrap();
        extensions
            .register_validation_target("check", "ticket", source)
            .unwrap();
        extensions
            .set_condition_unavailable_policy("ready", policy)
            .unwrap();
    });
    (f, calls)
}
async fn fixture(
    sqlite: bool,
    source: Arc<Revision>,
    mode: &'static str,
    policy: ExternalUnavailablePolicy,
) -> (super::controller_tests::Fixture, Arc<AtomicUsize>) {
    let (mut f, calls) = configured(source, mode, policy);
    f.store =
        super::continuation_test_support::store(f.agent.info().id, "controller.v1", sqlite).await;
    f.runner = AutonomyRunner::try_new(
        f.agent.clone(),
        f.config.clone(),
        Default::default(),
        f.store.clone(),
        "controller.v1".into(),
    )
    .unwrap();
    (f, calls)
}
fn signal(run: &TaskRun) -> TaskResumeInput {
    TaskResumeInput::ExternalSignal(ExternalSignal {
        request_id: run.pending.as_ref().unwrap().id.clone(),
        event_id: "event".into(),
        target_revision: Some("r1".into()),
        payload: json!(true),
    })
}

// A capability bit never substitutes for the selected check's actual frozen source.
#[test]
fn host_evidence_registration_is_atomic_and_configuration_does_not_sample_sources() {
    let source = Revision::new();
    let (f, _) = configured(source.clone(), "external", ExternalUnavailablePolicy::Stop);
    assert!(f.agent.validation_capabilities().host_revisions);
    assert_eq!(source.reads.load(Ordering::SeqCst), 0);
    let mut registry = AutonomyExtensions::builtins();
    registry
        .register_validation_target("a", "ticket", source.clone())
        .unwrap();
    assert!(
        registry
            .register_validation_target("b", "ticket", Revision::new())
            .is_err()
    );
    registry
        .register_validation_target("b", "ticket", source)
        .unwrap();
    assert!(
        registry
            .register_validation_target("a", "other", Revision::new())
            .is_err()
    );
    assert!(
        registry
            .set_condition_unavailable_policy("missing", ExternalUnavailablePolicy::Fail)
            .is_err()
    );
}

// Source failures before run admission neither reserve a runtime nor invoke a model.
#[tokio::test]
async fn host_evidence_unavailable_preflight_has_no_run_or_effect() {
    let source = Revision::new();
    let (f, callbacks) = configured(source.clone(), "external", ExternalUnavailablePolicy::Stop);
    source.unavailable.store(true, Ordering::SeqCst);
    assert!(f.runner.run("objective", None).await.is_err());
    assert!(f.store.list().await.unwrap().is_empty());
    assert_eq!(callbacks.load(Ordering::SeqCst), 0);
    assert_eq!(f.calls.load(Ordering::SeqCst), 0);
    assert_eq!(f.agent.chat("ordinary").await.unwrap().content, "done");
}

// A matching old signal cannot certify a changed target, but stale/unavailable targets do not block safe cleanup.
#[tokio::test]
async fn host_evidence_stale_signal_rejects_before_claim_and_cleanup_survives_source_loss() {
    for sqlite in super::continuation_test_support::backends() {
        for unavailable in [false, true] {
            let source = Revision::new();
            let (f, callbacks) = fixture(
                *sqlite,
                source.clone(),
                "external",
                ExternalUnavailablePolicy::Stop,
            )
            .await;
            let first = f.runner.run("objective", None).await.unwrap();
            let saved = f.store.load(&first.run.key.run_id).await.unwrap().unwrap();
            if unavailable {
                source.unavailable.store(true, Ordering::SeqCst);
            } else {
                source.change();
            }
            assert!(
                f.runner
                    .resume(
                        &first.run.key.run_id,
                        first.run.revision,
                        signal(&first.run)
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
                saved.revision
            );
            assert_eq!(callbacks.load(Ordering::SeqCst), 1);
            let stopped = f
                .runner
                .external_unavailable_paused(
                    &first.run.key.run_id,
                    first.run.revision,
                    &first.run.pending.as_ref().unwrap().id,
                    "host confirmed loss",
                )
                .await
                .unwrap();
            assert_eq!(stopped.run.status, TaskRunStatus::Incomplete);
            assert_eq!(
                stopped.run.stop_reason.as_deref(),
                Some("external_unavailable")
            );
            assert_eq!(callbacks.load(Ordering::SeqCst), 1);
            assert_eq!(f.agent.chat("ordinary").await.unwrap().content, "ordinary");
        }
    }
}

// Explicit delivery loss follows frozen stop/fail policy, never ordinary signal contents or silence.
#[tokio::test]
async fn host_evidence_unavailable_policy_is_request_bound_and_terminal_once() {
    for sqlite in super::continuation_test_support::backends() {
        for policy in [
            ExternalUnavailablePolicy::Stop,
            ExternalUnavailablePolicy::Fail,
        ] {
            let (f, callbacks) = fixture(*sqlite, Revision::new(), "external", policy).await;
            let first = f.runner.run("objective", None).await.unwrap();
            let id = &first.run.key.run_id;
            assert!(
                f.runner
                    .external_unavailable_paused(id, first.run.revision, "foreign", "lost")
                    .await
                    .is_err()
            );
            assert_eq!(
                f.store.load(id).await.unwrap().unwrap().revision,
                first.run.revision
            );
            let done = f
                .runner
                .external_unavailable_paused(
                    id,
                    first.run.revision,
                    &first.run.pending.as_ref().unwrap().id,
                    "host confirmed loss",
                )
                .await
                .unwrap();
            assert_eq!(
                done.run.status,
                if policy == ExternalUnavailablePolicy::Stop {
                    TaskRunStatus::Incomplete
                } else {
                    TaskRunStatus::Failed
                }
            );
            assert!(
                f.runner
                    .resume(id, done.run.revision, signal(&first.run))
                    .await
                    .is_err()
            );
            assert_eq!(callbacks.load(Ordering::SeqCst), 1);
            assert_eq!(f.calls.load(Ordering::SeqCst), 0);
        }
    }
}

// Completed batch/fallback observations are retained through a later target-bound external wait.
#[tokio::test]
async fn host_evidence_batch_and_fallback_then_external_do_not_replay_effects() {
    for sqlite in super::continuation_test_support::backends() {
        for mode in ["batch", "fallback"] {
            let (f, callbacks) = fixture(
                *sqlite,
                Revision::new(),
                mode,
                ExternalUnavailablePolicy::Stop,
            )
            .await;
            let mut result = f.runner.run("objective", None).await.unwrap();
            while matches!(
                result.run.pending.as_ref().unwrap().kind,
                TaskPendingKind::Approval
            ) {
                result = f
                    .runner
                    .resume(
                        &result.run.key.run_id,
                        result.run.revision,
                        TaskResumeInput::Approval {
                            request_id: result.run.pending.as_ref().unwrap().id.clone(),
                            result: ai_agents_hitl::ApprovalResult::Approved,
                        },
                    )
                    .await
                    .unwrap();
            }
            assert!(matches!(
                result.run.pending.as_ref().unwrap().kind,
                TaskPendingKind::ExternalCondition
            ));
            assert_eq!(f.calls.load(Ordering::SeqCst), 2);
            let done = f
                .runner
                .resume(
                    &result.run.key.run_id,
                    result.run.revision,
                    signal(&result.run),
                )
                .await
                .unwrap();
            assert_eq!(
                done.run.status,
                TaskRunStatus::Completed,
                "{:?}",
                done.run.stop_reason
            );
            assert_eq!(done.run.counters.turns, 1);
            assert_eq!(f.calls.load(Ordering::SeqCst), 2);
            assert_eq!(callbacks.load(Ordering::SeqCst), 3);
        }
    }
}

struct FaultStore {
    inner: Arc<ScopedTaskRunStore>,
    phase: &'static str,
    mode: &'static str,
    armed: AtomicBool,
    entered: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
    change: Option<Arc<Revision>>,
}
impl FaultStore {
    fn new(
        inner: Arc<ScopedTaskRunStore>,
        phase: &'static str,
        mode: &'static str,
        change: Option<Arc<Revision>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            inner,
            phase,
            mode,
            armed: AtomicBool::new(false),
            entered: tokio::sync::Semaphore::new(0),
            release: tokio::sync::Semaphore::new(0),
            change,
        })
    }
    fn selected(&self, mutation: &TaskRunMutation) -> bool {
        let payload = match mutation {
            TaskRunMutation::FinalCheckpoint { status, .. } if self.phase == "terminal" => {
                return matches!(
                    status,
                    TaskRunStatus::Completed | TaskRunStatus::Incomplete | TaskRunStatus::Failed
                );
            }
            TaskRunMutation::FinalCheckpoint {
                status: TaskRunStatus::Paused,
                ..
            } if self.phase == "pause" => return true,
            TaskRunMutation::Claim { .. } if self.phase == "claim" => return true,
            TaskRunMutation::Checkpoint { payload, .. } => payload,
            _ => return false,
        };
        let Some(driver) = payload["adapters"].as_array().and_then(|adapters| {
            adapters
                .iter()
                .find_map(|a| a["state"].get("_task_validation_driver"))
        }) else {
            return false;
        };
        match self.phase {
            "accept" => {
                driver["external_events"]
                    .as_object()
                    .is_some_and(|events| !events.is_empty())
                    && driver["rounds"] == 1
                    && driver["result"].is_null()
            }
            "round" => {
                driver["rounds"] == 2 && driver["wait"].is_null() && driver["result"].is_null()
            }
            "publish" => !driver["result"].is_null(),
            "wait" => !driver["wait"].is_null(),
            "rewait" => driver["rounds"] == 2 && !driver["wait"].is_null(),
            _ => false,
        }
    }
}
#[async_trait]
impl TaskRunStore for FaultStore {
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
    fn supports_objective_revision(&self) -> bool {
        true
    }
    async fn apply_host_control(
        &self,
        id: &str,
        revision: u64,
        action: HostControlAction,
        host: &AutonomyHostCeilings,
        deadline: Option<std::time::Instant>,
    ) -> Result<Option<TaskRunSnapshot>> {
        self.inner
            .apply_host_control(id, revision, action, host, deadline)
            .await
    }
    fn supports_paused_limit_revision(&self) -> bool {
        true
    }
    async fn revise_paused_limits(
        &self,
        id: &str,
        revision: u64,
        limits: TaskRunLimits,
        reason: &str,
        host: &AutonomyHostCeilings,
        deadline: Option<std::time::Instant>,
    ) -> Result<Option<TaskRunSnapshot>> {
        self.inner
            .revise_paused_limits(id, revision, limits, reason, host, deadline)
            .await
    }
    async fn mutate(&self, id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot> {
        let hit = self.selected(mutation) && self.armed.swap(false, Ordering::SeqCst);
        let previous = self.inner.load(id).await?.unwrap();
        if hit && self.mode == "reject" {
            return Err(AgentError::Persistence("rejected selected write".into()));
        }
        if hit && self.mode == "no_write" {
            return Ok(previous);
        }
        if hit && self.mode == "io" {
            tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
            return Err(AgentError::IoError(std::io::Error::other(
                "unacknowledged I/O failure",
            )));
        }
        if hit && self.mode == "before" {
            self.change.as_ref().unwrap().change();
        }
        let mut saved = self.inner.mutate(id, mutation).await?;
        if hit {
            if let Some(source) = &self.change {
                source.change();
            }
            match self.mode {
                "stale" => saved.revision = previous.revision,
                "payload" => saved.payload = previous.payload,
                "owner" => saved.owner_token = Some("foreign".into()),
                "lost" => {
                    return Err(AgentError::Persistence(
                        "lost selected acknowledgement".into(),
                    ));
                }
                "gate" => {
                    self.entered.add_permits(1);
                    self.release.acquire().await.unwrap().forget();
                }
                _ => {}
            }
        }
        Ok(saved)
    }
}
fn install_fault(
    f: &mut super::controller_tests::Fixture,
    phase: &'static str,
    mode: &'static str,
    change: Option<Arc<Revision>>,
) -> Arc<FaultStore> {
    let store = FaultStore::new(f.store.clone(), phase, mode, change);
    f.runner = AutonomyRunner::try_new(
        f.agent.clone(),
        f.config.clone(),
        Default::default(),
        store.clone(),
        "controller.v1".into(),
    )
    .unwrap();
    store
}

// Invalid acknowledgement cannot permit callback reentry or release an owner, even when the write actually committed.
#[tokio::test]
async fn host_evidence_signal_and_publication_receipts_fail_closed() {
    for sqlite in super::continuation_test_support::backends() {
        for phase in ["accept", "round", "publish", "terminal"] {
            for mode in ["reject", "no_write", "lost", "stale", "payload", "owner"] {
                let (mut f, callbacks) = fixture(
                    *sqlite,
                    Revision::new(),
                    "external",
                    ExternalUnavailablePolicy::Stop,
                )
                .await;
                let fault = install_fault(&mut f, phase, mode, None);
                let first = f.runner.run("objective", None).await.unwrap();
                fault.armed.store(true, Ordering::SeqCst);
                assert!(
                    f.runner
                        .resume(
                            &first.run.key.run_id,
                            first.run.revision,
                            signal(&first.run)
                        )
                        .await
                        .is_err(),
                    "{phase}/{mode}"
                );
                assert!(
                    !fault.armed.load(Ordering::SeqCst),
                    "fault not reached: {phase}/{mode}"
                );
                assert_eq!(
                    callbacks.load(Ordering::SeqCst),
                    if matches!(phase, "accept" | "round") {
                        1
                    } else {
                        2
                    }
                );
                assert!(f.agent.chat("blocked").await.is_err());
                let saved = f.store.load(&first.run.key.run_id).await.unwrap().unwrap();
                assert!(
                    f.runner
                        .resume(&saved.key.run_id, saved.revision, signal(&first.run))
                        .await
                        .is_err()
                );
            }
        }
    }
}

// Host target changes during claim/acceptance waits retire only safe work; later publication races never certify completion.
#[tokio::test]
async fn host_evidence_target_changes_at_acknowledgement_boundaries() {
    for sqlite in super::continuation_test_support::backends() {
        for phase in ["claim", "accept", "round", "publish"] {
            let source = Revision::new();
            let (mut f, callbacks) = fixture(
                *sqlite,
                source.clone(),
                "external",
                ExternalUnavailablePolicy::Stop,
            )
            .await;
            let fault = install_fault(&mut f, phase, "change", Some(source));
            let first = f.runner.run("objective", None).await.unwrap();
            fault.armed.store(true, Ordering::SeqCst);
            let outcome = f
                .runner
                .resume(
                    &first.run.key.run_id,
                    first.run.revision,
                    signal(&first.run),
                )
                .await;
            assert!(!fault.armed.load(Ordering::SeqCst));
            assert!(
                !outcome
                    .as_ref()
                    .is_ok_and(|result| result.run.status == TaskRunStatus::Completed)
            );
            assert_eq!(
                callbacks.load(Ordering::SeqCst),
                if phase == "publish" { 2 } else { 1 }
            );
        }
    }
}

// A shorter check timeout cannot be replaced by the still-valid whole-run lifetime after a journal await.
#[tokio::test]
async fn host_evidence_callback_and_publication_recheck_check_deadline() {
    for phase in ["round", "publish"] {
        let (mut f, callbacks) =
            configured(Revision::new(), "external", ExternalUnavailablePolicy::Stop);
        f.config
            .defaults
            .validation
            .as_mut()
            .unwrap()
            .checks
            .as_mut()
            .unwrap()[0]
            .timeout_seconds = Some(1);
        let fault = install_fault(&mut f, phase, "gate", None);
        let first = f.runner.run("objective", None).await.unwrap();
        fault.armed.store(true, Ordering::SeqCst);
        let mut future = Box::pin(f.runner.resume(
            &first.run.key.run_id,
            first.run.revision,
            signal(&first.run),
        ));
        tokio::select! { result = &mut future => panic!("completed before boundary: {:?}", result.as_ref().err()), permit = fault.entered.acquire() => permit.unwrap().forget(), _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("boundary not reached") }
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        fault.release.add_permits(1);
        let done = future.await.unwrap();
        assert_ne!(done.run.status, TaskRunStatus::Completed);
        assert_eq!(
            callbacks.load(Ordering::SeqCst),
            if phase == "round" { 1 } else { 2 }
        );
    }
}

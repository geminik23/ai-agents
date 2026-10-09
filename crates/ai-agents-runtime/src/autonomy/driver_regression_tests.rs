use super::evaluation_tests::{bound_script, current, identity, record, scope};
use super::*;
use ai_agents_core::{AgentError, Result, ToolCancellationToken};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

#[derive(Default)]
struct Executor {
    calls: AtomicUsize,
}
#[async_trait]
impl ValidationObservationExecutor for Executor {
    /// Every poll is an actual simulated invocation with a unique authoritative logical request ID.
    async fn execute(
        &self,
        request: &ValidationObservationRequest,
        _: &EvidenceIdentity,
    ) -> Result<ObservationResult> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        if let ValidationObservationRequest::Tool {
            tool, arguments, ..
        } = request
        {
            let mut record = record(tool, arguments.clone(), json!({"ok":true}));
            record.call_id = format!("call-{n}");
            Ok(ObservationResult::Tool {
                record: Box::new(record),
            })
        } else {
            Ok(ObservationResult::Judge { score: 0.9 })
        }
    }
}
#[derive(Default)]
struct Journal {
    fail_publication: AtomicBool,
    cancel: Option<Arc<AtomicBool>>,
    delay_inflight: bool,
    partial: Mutex<Option<Value>>,
}
#[async_trait]
impl ValidationJournal for Journal {
    /// Journals can fail after an effect or change cancellation while pre-effect persistence is awaited.
    async fn checkpoint(&self, state: &ValidationDriverState) -> Result<()> {
        if (state.result.is_some() || state.wait.is_some())
            && self.fail_publication.load(Ordering::SeqCst)
        {
            return Err(AgentError::Persistence("publication unavailable".into()));
        }
        if state.in_flight.is_some() {
            if let Some(cancel) = &self.cancel {
                cancel.store(true, Ordering::SeqCst);
            }
            if self.delay_inflight {
                tokio::time::sleep(std::time::Duration::from_millis(40)).await;
            }
        }
        if state.completed.len() == 1
            && !state.pending_batch.is_empty()
            && state.in_flight.is_none()
        {
            *self.partial.lock().unwrap() = Some(serde_json::to_value(state)?);
            return Err(AgentError::Persistence(
                "saved first result, lost acknowledgement".into(),
            ));
        }
        Ok(())
    }
}

// Cached completion and waits cannot bypass acknowledgement after a failed terminal write.
#[tokio::test]
async fn driver_retries_publication_and_never_collects_unacknowledged_result() {
    for mode in ["complete", "external"] {
        let (bound, s, _) = bound_script(mode);
        let check = &bound.checks[0];
        let mut i = identity(&s);
        i.config_identity = check.config_identity.clone();
        let mut state = ValidationDriverState::new(check, i).unwrap();
        let journal = Journal::default();
        journal.fail_publication.store(true, Ordering::SeqCst);
        let exec = Executor::default();
        let cancellation = ToolCancellationToken::new(Arc::new(AtomicBool::new(false)), None);
        let mut evidence = current(&s);
        let ctx = ValidationDriveContext {
            scope: &s,
            evidence: &evidence,
            extensions: &bound.extensions,
            executor: &exec,
            journal: &journal,
            limits: Default::default(),
            cancellation: &cancellation,
            deadline: chrono::Utc::now() + chrono::Duration::seconds(2),
        };
        assert!(state.drive(check, &ctx).await.is_err());
        assert!(state.drive(check, &ctx).await.is_err());
        assert!(!state.publication_acknowledged());
        if mode == "external" {
            let wait = state.wait.as_ref().unwrap().clone();
            assert!(
                state
                    .accept_external(
                        ExternalSignal {
                            request_id: wait.request_id,
                            event_id: "premature".into(),
                            target_revision: None,
                            payload: json!(true)
                        },
                        &bound.extensions
                    )
                    .is_err()
            );
        }
        let expected = if mode == "complete" { 1 } else { 0 };
        assert_eq!(exec.calls.load(Ordering::SeqCst), expected);
        journal.fail_publication.store(false, Ordering::SeqCst);
        state.drive(check, &ctx).await.unwrap();
        assert!(state.publication_acknowledged());
        assert_eq!(exec.calls.load(Ordering::SeqCst), expected);
        if mode == "complete" {
            evidence.collect_validation(&state).unwrap();
        } else {
            let mut restored = ValidationDriverState::restore(
                serde_json::to_value(&state).unwrap(),
                check,
                &s,
                &bound.extensions,
            )
            .unwrap();
            let wait = restored.wait.as_ref().unwrap().clone();
            let signal = ExternalSignal {
                request_id: wait.request_id,
                event_id: "restored".into(),
                target_revision: None,
                payload: json!(true),
            };
            assert!(
                restored
                    .accept_external(signal.clone(), &bound.extensions)
                    .is_err()
            );
            restored.drive(check, &ctx).await.unwrap();
            restored.accept_external(signal, &bound.extensions).unwrap();
        }
    }
}

// No executor poll occurs when cancellation or expiry commits during the pre-effect journal wait.
#[tokio::test]
async fn driver_rechecks_cancellation_and_deadline_after_journal() {
    for expired in [false, true] {
        let (bound, s, _) = bound_script("complete");
        let check = &bound.checks[0];
        let mut i = identity(&s);
        i.config_identity = check.config_identity.clone();
        let mut state = ValidationDriverState::new(check, i).unwrap();
        let flag = Arc::new(AtomicBool::new(false));
        let cancellation = ToolCancellationToken::new(flag.clone(), None);
        let journal = Journal {
            cancel: (!expired).then_some(flag),
            delay_inflight: expired,
            ..Default::default()
        };
        let exec = Executor::default();
        let evidence = current(&s);
        let ctx = ValidationDriveContext {
            scope: &s,
            evidence: &evidence,
            extensions: &bound.extensions,
            executor: &exec,
            journal: &journal,
            limits: Default::default(),
            cancellation: &cancellation,
            deadline: chrono::Utc::now()
                + chrono::Duration::milliseconds(if expired { 20 } else { 2000 }),
        };
        assert!(state.drive(check, &ctx).await.is_err());
        assert_eq!(exec.calls.load(Ordering::SeqCst), 0);
        assert!(state.in_flight.is_some());
        assert!(
            ValidationDriverState::restore(
                serde_json::to_value(&state).unwrap(),
                check,
                &s,
                &bound.extensions
            )
            .is_err()
        );
    }
}

struct Batch {
    descriptor: AdapterDescriptor,
}
impl AutonomyValidator for Batch {
    /// The adapter declares its sole tool capability before freezing.
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.descriptor
    }
    /// The fixture accepts an empty config but no I/O from the callback.
    fn validate_config(&self, _: &Value) -> Result<()> {
        Ok(())
    }
    /// A premature callback would complete from one result, exposing a lost-batch-cursor regression.
    fn evaluate(&self, input: &ValidationInput<'_>) -> Result<ValidationDecision> {
        if !input.observations.is_empty() {
            return Ok(ValidationDecision::Complete {
                outcome: GateOutcome::Pass,
                reason: "observed".into(),
                metrics: json!({}),
                evidence_refs: input.observations.keys().cloned().collect(),
            });
        }
        Ok(ValidationDecision::NeedObservations {
            requests: (0..2)
                .map(|n| ValidationObservationRequest::Tool {
                    id: format!("observation-{n}"),
                    tool: "probe".into(),
                    arguments: json!({"value":n}),
                    host_binding: None,
                })
                .collect(),
            checkpoint: json!({}),
        })
    }
}
// Optional YAML limits intersect with host ceilings rather than being silently ignored.
fn batch_binding(max_observations: u32) -> (BoundAutonomyProfile, EvaluationScope) {
    let mut registry = AutonomyExtensions::builtins();
    registry
        .register_validator(Arc::new(Batch {
            descriptor: AdapterDescriptor {
                id: "host.batch".into(),
                contract_version: 1,
                tools: vec!["probe".into()],
                ..Default::default()
            },
        }))
        .unwrap();
    let profile:AutonomyProfile=serde_yaml::from_str(&format!("validation: {{checks: [{{id: check, adapter: host.batch, max_observations_per_round: {max_observations}, config: {{}}}}]}}" )).unwrap();
    let bound = registry
        .freeze()
        .bind(
            &profile,
            &ValidationCapabilities {
                tools: BTreeSet::from(["probe".into()]),
                ..Default::default()
            },
        )
        .unwrap();
    let mut s = scope();
    bound.prepare_scope(&mut s);
    (bound, s)
}

// Restoring a partial batch finishes its remaining operation before callback reentry and never repeats the first.
#[tokio::test]
async fn driver_restores_batch_cursor_and_enforces_configured_batch_limit() {
    for max in [1, 2] {
        let (bound, s) = batch_binding(max);
        let check = &bound.checks[0];
        let mut i = identity(&s);
        i.config_identity = check.config_identity.clone();
        let mut state = ValidationDriverState::new(check, i).unwrap();
        let journal = Journal::default();
        let exec = Executor::default();
        let cancel = ToolCancellationToken::new(Arc::new(AtomicBool::new(false)), None);
        let evidence = current(&s);
        let ctx = ValidationDriveContext {
            scope: &s,
            evidence: &evidence,
            extensions: &bound.extensions,
            executor: &exec,
            journal: &journal,
            limits: Default::default(),
            cancellation: &cancel,
            deadline: chrono::Utc::now() + chrono::Duration::seconds(2),
        };
        assert!(state.drive(check, &ctx).await.is_err());
        if max == 1 {
            assert_eq!(exec.calls.load(Ordering::SeqCst), 0);
        } else {
            assert_eq!(exec.calls.load(Ordering::SeqCst), 1);
            let partial = journal.partial.lock().unwrap().clone().unwrap();
            let mut restored =
                ValidationDriverState::restore(partial, check, &s, &bound.extensions).unwrap();
            let safe = Journal::default();
            let ctx = ValidationDriveContext {
                journal: &safe,
                ..ctx
            };
            assert!(matches!(
                restored.drive(check, &ctx).await.unwrap(),
                ValidationDriverOutcome::Complete(_)
            ));
            assert_eq!(exec.calls.load(Ordering::SeqCst), 2);
            assert_eq!(restored.completed.len(), 2);
        }
    }
}

// Check-specific round limits include callback reentry even with permissive default host bounds.
#[tokio::test]
async fn driver_enforces_configured_round_limit() {
    let (mut bound, mut s, calls) = bound_script("complete");
    bound.checks[0].check.max_evaluation_rounds = Some(1);
    bound.checks[0].config_identity =
        canonical_identity(&serde_json::to_value(&bound.checks[0].check).unwrap()).unwrap();
    bound.prepare_scope(&mut s);
    let check = &bound.checks[0];
    let mut i = identity(&s);
    i.config_identity = check.config_identity.clone();
    let mut state = ValidationDriverState::new(check, i).unwrap();
    let journal = Journal::default();
    let exec = Executor::default();
    let cancel = ToolCancellationToken::new(Arc::new(AtomicBool::new(false)), None);
    let evidence = current(&s);
    let ctx = ValidationDriveContext {
        scope: &s,
        evidence: &evidence,
        extensions: &bound.extensions,
        executor: &exec,
        journal: &journal,
        limits: Default::default(),
        cancellation: &cancel,
        deadline: chrono::Utc::now() + chrono::Duration::seconds(2),
    };
    assert!(state.drive(check, &ctx).await.is_err());
    assert_eq!(exec.calls.load(Ordering::SeqCst), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

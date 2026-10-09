use super::*;
use ai_agents_core::autonomy::{
    CommandExitGate, CompletionGate, DiagnosticsClearGate, PathAssertion, TaskRunStatus,
    ToolCalledGate,
};
use ai_agents_core::{
    AgentError, ToolCallSource, ToolCancellationToken, ToolExecutionRecord,
    ToolPolicyDecisionRecord,
};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

// Scope fixtures use explicit current binding and attempt identity, not timestamps as freshness.
pub(super) fn scope() -> EvaluationScope {
    EvaluationScope {
        key: TaskRunKey {
            agent_id: "agent".into(),
            run_id: "run".into(),
        },
        objective_revision: 0,
        cycle: 1,
        stage: None,
        mutation_generation: 0,
        target_revisions: BTreeMap::new(),
        target_generations: BTreeMap::new(),
        validation_bindings: BTreeMap::from([("check".into(), "binding".into())]),
        validation_attempts: BTreeMap::from([("check".into(), "attempt".into())]),
    }
}
// Every observed value carries its own immutable scope and event identity.
pub(super) fn identity(scope: &EvaluationScope) -> EvidenceIdentity {
    EvidenceIdentity {
        key: scope.key.clone(),
        objective_revision: scope.objective_revision,
        cycle: scope.cycle,
        sequence: 1,
        stage: scope.stage.clone(),
        attempt: Some("attempt".into()),
        config_identity: "binding".into(),
        target: None,
        target_revision: None,
        mutation_generation: scope.mutation_generation,
    }
}
// Shared-record fixtures model final execution evidence, including negative policy and partial-output flags.
pub(super) fn record(tool: &str, args: Value, output: Value) -> ToolExecutionRecord {
    ToolExecutionRecord {
        call_id: format!("{tool}-call"),
        requested_name: tool.into(),
        canonical_id: tool.into(),
        source: ToolCallSource::Task,
        arguments: args.clone(),
        executed_arguments: args,
        policy_version: 0,
        registry_version: 0,
        runtime_config_version: 0,
        executed: true,
        success: true,
        output: output.to_string(),
        metadata: HashMap::new(),
        policy: ToolPolicyDecisionRecord::allow(),
        approval: None,
        started_at: chrono::Utc::now(),
        duration_ms: 0,
        timed_out: false,
        cancelled: false,
        cancellation_reason: None,
        output_truncated: false,
    }
}
// Full current captures differ from missing captures and from provisional/incomplete data.
pub(super) fn current(scope: &EvaluationScope) -> EvaluationEvidence {
    EvaluationEvidence {
        current: Some(ScopedObservation {
            identity: identity(scope),
            complete: true,
            value: CurrentTaskObservation {
                state: Some("ready".into()),
                context: json!({"ticket":{"result":null},"count":3}),
                response: "한국어 결과".into(),
                todos: None,
            },
        }),
        ..Default::default()
    }
}
// Evaluators report tri-state outcomes; configuration/integrity errors remain errors.
fn outcome(
    gate: CompletionGate,
    scope: &EvaluationScope,
    evidence: &EvaluationEvidence,
) -> GateOutcome {
    CompletionGateEvaluator::default()
        .evaluate(&gate, scope, evidence)
        .unwrap()
        .outcome
}

// Negation never upgrades missing, denied, partial, stale or prior-objective observations.
#[test]
fn gates_preserve_unknown_and_independent_any_success() {
    let s = scope();
    let e = current(&s);
    assert_eq!(
        outcome(
            CompletionGate::Not(Box::new(CompletionGate::ValidatorPassed("missing".into()))),
            &s,
            &e
        ),
        GateOutcome::Unknown
    );
    assert_eq!(
        outcome(
            CompletionGate::Any(vec![
                CompletionGate::ValidatorPassed("missing".into()),
                CompletionGate::ResponseContains("한국어".into())
            ]),
            &s,
            &e
        ),
        GateOutcome::Pass
    );
    let null: CompletionGate =
        serde_yaml::from_str("context_path: {path: ticket.result, eq: null}").unwrap();
    assert_eq!(outcome(null, &s, &e), GateOutcome::Pass);
    let path: CompletionGate =
        serde_yaml::from_str("context_path: {path: missing, gte: 0}").unwrap();
    assert_eq!(
        outcome(CompletionGate::Not(Box::new(path)), &s, &e),
        GateOutcome::Unknown
    );
    let nonnumeric: CompletionGate =
        serde_yaml::from_str("context_path: {path: ticket.result, gte: 0}").unwrap();
    assert_eq!(outcome(nonnumeric, &s, &e), GateOutcome::Fail);
    let rendered = serde_json::to_string(
        &CompletionGateEvaluator::default()
            .evaluate(&CompletionGate::ResponseContains("한국어".into()), &s, &e)
            .unwrap(),
    )
    .unwrap();
    assert!(!rendered.contains("한국어"));
    let mut old = e;
    old.current.as_mut().unwrap().identity.objective_revision = 99;
    assert_eq!(
        outcome(CompletionGate::ResponseNotEmpty(true), &s, &old),
        GateOutcome::Unknown
    );
}

// One real request cannot manufacture multiple proof-of-work invocations by appearing twice.
#[test]
fn gates_deduplicate_calls_and_reject_conflicting_validation() {
    let s = scope();
    let mut e = current(&s);
    let tool = ScopedObservation {
        identity: identity(&s),
        complete: true,
        value: record("probe", json!({}), json!({"ok":true})),
    };
    e.tools = vec![tool.clone(), tool];
    let gate = CompletionGate::ToolCalled(ToolCalledGate {
        id: "probe".into(),
        count: None,
        count_gte: Some(2),
        executed: Some(true),
        success: Some(true),
    });
    assert_eq!(outcome(gate, &s, &e), GateOutcome::Fail);
    e.tools[0].value.executed = false;
    e.tools[1] = e.tools[0].clone();
    assert_eq!(
        outcome(
            CompletionGate::Not(Box::new(CompletionGate::ToolCalled(ToolCalledGate {
                id: "probe".into(),
                count: None,
                count_gte: Some(1),
                executed: Some(true),
                success: Some(true)
            }))),
            &s,
            &e
        ),
        GateOutcome::Unknown
    );
    let result = ValidationResult {
        check_id: "check".into(),
        identity: identity(&s),
        outcome: GateOutcome::Pass,
        reason: "verified".into(),
        metrics: json!({}),
        evidence_refs: vec![],
    };
    let mut conflict = result.clone();
    conflict.outcome = GateOutcome::Fail;
    e.validations = vec![result.clone(), conflict.clone()];
    assert!(
        CompletionGateEvaluator::default()
            .evaluate(&CompletionGate::ValidatorPassed("check".into()), &s, &e)
            .is_err()
    );
    e.validations = vec![conflict, result];
    assert!(
        CompletionGateEvaluator::default()
            .evaluate(&CompletionGate::ValidatorPassed("check".into()), &s, &e)
            .is_err()
    );
}

// Direct command and diagnostics gates require current selected attempts and non-contradictory execution evidence.
#[test]
fn gates_reject_old_attempts_partial_diagnostics_and_wrong_tool() {
    let mut s = scope();
    let mut e = current(&s);
    e.commands.push(ScopedObservation {
        identity: identity(&s),
        complete: true,
        value: CommandObservation {
            check_id: "check".into(),
            command: "false".into(),
            exit_code: Some(1),
            termination: "exited".into(),
            record: record(
                "command",
                json!({"command":"false","cwd":"."}),
                json!({"exit_code":1,"termination":"exited"}),
            ),
        },
    });
    let command = CompletionGate::CommandExit(CommandExitGate {
        command: "false".into(),
        code: 1,
    });
    assert_eq!(outcome(command.clone(), &s, &e), GateOutcome::Pass);
    s.validation_attempts
        .insert("check".into(), "replacement".into());
    assert_eq!(
        outcome(CompletionGate::Not(Box::new(command)), &s, &e),
        GateOutcome::Unknown
    );
    s.validation_attempts
        .insert("check".into(), "attempt".into());
    e.diagnostics.push(ScopedObservation {
        identity: identity(&s),
        complete: true,
        value: DiagnosticsObservation {
            check_id: "check".into(),
            available: true,
            complete: true,
            severity_counts: BTreeMap::from([("error".into(), 0)]),
            record: record(
                "diagnostics",
                json!({}),
                json!({"available":true,"diagnostics":[],"truncated":false}),
            ),
        },
    });
    let gate = CompletionGate::DiagnosticsClear(DiagnosticsClearGate {
        severity: "error".into(),
    });
    assert_eq!(outcome(gate.clone(), &s, &e), GateOutcome::Pass);
    e.diagnostics[0].value.record.output_truncated = true;
    assert_eq!(
        outcome(CompletionGate::Not(Box::new(gate.clone())), &s, &e),
        GateOutcome::Unknown
    );
    e.diagnostics[0].value.record.output_truncated = false;
    e.diagnostics[0].value.record.canonical_id = "other".into();
    assert_eq!(outcome(gate, &s, &e), GateOutcome::Unknown);
}

// Optional host target generations invalidate only the selected target and cannot replace attempt/config identity.
#[test]
fn gates_validate_target_configuration_and_latest_attempt() {
    let mut s = scope();
    let mut i = identity(&s);
    i.target = Some("ticket".into());
    i.target_revision = Some("v1".into());
    i.mutation_generation = 5;
    s.target_revisions.insert("ticket".into(), "v1".into());
    s.target_generations.insert("ticket".into(), 5);
    s.mutation_generation = 77;
    let mut e = current(&s);
    e.validations.push(ValidationResult {
        check_id: "check".into(),
        identity: i,
        outcome: GateOutcome::Pass,
        reason: "verified".into(),
        metrics: json!({}),
        evidence_refs: vec![],
    });
    assert_eq!(
        outcome(CompletionGate::ValidatorPassed("check".into()), &s, &e),
        GateOutcome::Pass
    );
    s.target_revisions.insert("ticket".into(), "v2".into());
    assert_eq!(
        outcome(CompletionGate::ValidatorPassed("check".into()), &s, &e),
        GateOutcome::Unknown
    );
    s.target_revisions.insert("ticket".into(), "v1".into());
    s.validation_bindings
        .insert("check".into(), "other-version".into());
    assert_eq!(
        outcome(CompletionGate::ValidatorPassed("check".into()), &s, &e),
        GateOutcome::Unknown
    );
}

// Negative metrics use their observed baseline; restoring a prior high or a prior fingerprint is not fresh progress.
#[test]
fn progress_retains_high_water_and_bounded_fingerprints() {
    let mut s = scope();
    let mut state = ProgressState::new(s.clone());
    let observe = |value, comparison| ProgressObservation {
        metrics: json!({}),
        signals: BTreeMap::from([("value".into(), value)]),
        comparisons: BTreeMap::from([("value".into(), comparison)]),
        evidence_refs: vec!["current:1".into()],
    };
    assert_eq!(
        state
            .observe(&s, &observe(json!(-2), SignalComparison::Increase))
            .unwrap(),
        ProgressDelta::Unchanged
    );
    s.cycle += 1;
    assert_eq!(
        state
            .observe(&s, &observe(json!(-1), SignalComparison::Increase))
            .unwrap(),
        ProgressDelta::Advanced
    );
    s.cycle += 1;
    state
        .observe(&s, &observe(json!(-2), SignalComparison::Increase))
        .unwrap();
    s.cycle += 1;
    assert_eq!(
        state
            .observe(&s, &observe(json!(-1), SignalComparison::Increase))
            .unwrap(),
        ProgressDelta::Unchanged
    );
    s.cycle += 1;
    state
        .observe(&s, &observe(json!("A"), SignalComparison::Change))
        .unwrap();
    s.cycle += 1;
    state
        .observe(&s, &observe(json!("B"), SignalComparison::Change))
        .unwrap();
    s.cycle += 1;
    assert_eq!(
        state
            .observe(&s, &observe(json!("A"), SignalComparison::Change))
            .unwrap(),
        ProgressDelta::Regressed
    );
    let value = serde_json::to_value(&state).unwrap();
    let restored = ProgressState::restore(value.clone(), &s).unwrap();
    assert_eq!(serde_json::to_value(restored).unwrap(), value);
    state.replans = 1;
    state.cycles_without_progress = 10;
    let policy = StagnationConfig {
        max_cycles_without_progress: Some(1),
        action: Some(StagnationAction::Replan),
        max_replans: Some(1),
        on_exhausted: Some(StagnationAction::Replan),
        on_unavailable: None,
    };
    assert!(state.stagnation(Some(&policy), false, true).is_err());
    assert_eq!(state.replans, 1);
}

// Unknown checks never fabricate numeric zero for negated progress gates.
#[test]
fn progress_unknown_observations_remain_unknown() {
    let s = scope();
    let mut e = current(&s);
    e.required_checks.push("check".into());
    e.validations.push(ValidationResult {
        check_id: "check".into(),
        identity: identity(&s),
        outcome: GateOutcome::Unknown,
        reason: "missing".into(),
        metrics: json!({}),
        evidence_refs: vec![],
    });
    let config = AutonomyProfile::default();
    let bound = AutonomyExtensions::builtins()
        .freeze()
        .bind(&config, &ValidationCapabilities::default())
        .unwrap();
    let observed = bound.observe_progress(&s, &e).unwrap();
    assert!(observed.signals.is_empty());
    e.progress = Some(observed.scoped(identity(&s)));
    let gate = CompletionGate::ProgressPath(PathAssertion {
        path: "metrics.signals.check.check".into(),
        eq: None,
        gte: Some(1.0),
        gt: None,
        lte: None,
        lt: None,
        exists: None,
    });
    assert_eq!(
        outcome(CompletionGate::Not(Box::new(gate)), &s, &e),
        GateOutcome::Unknown
    );
}

// Required lifecycle barriers and skill scripts cannot be skipped by an early passing task gate.
#[test]
fn lifecycle_rejects_forged_prefix_and_preserves_terminal_precedence() {
    let profile:AutonomyProfile=serde_yaml::from_str("lifecycle:\n  - id: explore\n    required_before_tools: [write]\n  - id: validate\n    retry_on_failure: true\n    on_failure_stage: explore\n    max_fix_cycles: 1\n").unwrap();
    let mut lifecycle = LifecycleState::new(&profile, false).unwrap();
    assert!(!lifecycle.allows_tool(&profile, "write"));
    lifecycle.completed_stages.insert("validate".into());
    assert!(lifecycle.validate(&profile).is_err());
    lifecycle.completed_stages.clear();
    lifecycle
        .complete_stage(&profile, "explore", GateOutcome::Pass, GateOutcome::Pass)
        .unwrap();
    assert!(lifecycle.allows_tool(&profile, "write"));
    lifecycle.validation_failure(&profile).unwrap();
    assert!(!lifecycle.allows_tool(&profile, "write"));
    assert!(!lifecycle.requirements_met(&profile));
    let mut progress = ProgressState::new(scope());
    let mut input = DecisionInputs {
        uncertain_effect: false,
        cancel_requested: false,
        unrecoverable_error: false,
        actual_time_violation: false,
        mandatory_wait: true,
        gates: GateOutcome::Pass,
        required_validation: GateOutcome::Pass,
        children_settled: true,
        scope_ended: false,
        additional_work_capacity: false,
        interaction_available: false,
    };
    assert_eq!(
        checkpoint_decision(&profile, &lifecycle, &mut progress, &input).unwrap(),
        CheckpointDecision::Wait
    );
    input.cancel_requested = true;
    assert!(matches!(
        checkpoint_decision(&profile, &lifecycle, &mut progress, &input).unwrap(),
        CheckpointDecision::Terminal {
            status: TaskRunStatus::Cancelled,
            ..
        }
    ));
    input.uncertain_effect = true;
    assert!(matches!(
        checkpoint_decision(&profile, &lifecycle, &mut progress, &input).unwrap(),
        CheckpointDecision::Terminal {
            status: TaskRunStatus::RecoveryRequired,
            ..
        }
    ));
    let profile = AutonomyProfile::default();
    let lifecycle = LifecycleState::new(&profile, false).unwrap();
    let mut progress = ProgressState::new(scope());
    input.uncertain_effect = false;
    input.cancel_requested = false;
    input.mandatory_wait = false;
    assert!(matches!(
        checkpoint_decision(&profile, &lifecycle, &mut progress, &input).unwrap(),
        CheckpointDecision::Terminal {
            status: TaskRunStatus::Completed,
            ..
        }
    ));
    assert!(checkpoint_decision(&profile, &lifecycle, &mut progress, &input).is_err());
}

struct ScriptValidator {
    descriptor: AdapterDescriptor,
    mode: &'static str,
    calls: Arc<AtomicUsize>,
}
impl AutonomyValidator for ScriptValidator {
    /// A stable declared observation capability is captured by the frozen registry.
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.descriptor
    }
    /// Test callbacks still obey the object configuration boundary.
    fn validate_config(&self, v: &Value) -> ai_agents_core::Result<()> {
        if v.is_object() {
            Ok(())
        } else {
            Err(AgentError::Config("config".into()))
        }
    }
    /// Deliberate stalls and identity changes exercise driver fail-closed behavior without hidden I/O.
    fn evaluate(&self, input: &ValidationInput<'_>) -> ai_agents_core::Result<ValidationDecision> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.mode == "external" {
            if input.previous.get("external_signal") == Some(&json!(true)) {
                return Ok(ValidationDecision::Complete {
                    outcome: GateOutcome::Pass,
                    reason: "signalled".into(),
                    metrics: json!({}),
                    evidence_refs: vec![],
                });
            }
            return Ok(ValidationDecision::AwaitExternal {
                binding: "ready".into(),
                checkpoint: json!({}),
            });
        }
        if !input.observations.is_empty() && self.mode == "complete" {
            return Ok(ValidationDecision::Complete {
                outcome: GateOutcome::Pass,
                reason: "observed".into(),
                metrics: json!({}),
                evidence_refs: vec!["observe".into()],
            });
        }
        let value = if !input.observations.is_empty() && self.mode == "changed" {
            2
        } else {
            1
        };
        Ok(ValidationDecision::NeedObservations {
            requests: vec![ValidationObservationRequest::Tool {
                id: "observe".into(),
                tool: "probe".into(),
                arguments: json!({"value":value}),
                host_binding: None,
            }],
            checkpoint: json!({}),
        })
    }
}

// Check normalization creates scope keys and selected attempts before any callback runs.
pub(super) fn bound_script(
    mode: &'static str,
) -> (BoundAutonomyProfile, EvaluationScope, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut registry = AutonomyExtensions::builtins();
    registry
        .register_validator(Arc::new(ScriptValidator {
            descriptor: AdapterDescriptor {
                id: "host.script".into(),
                contract_version: 1,
                tools: vec!["probe".into()],
                conditions: if mode == "external" {
                    vec!["ready".into()]
                } else {
                    vec![]
                },
                ..Default::default()
            },
            mode,
            calls: calls.clone(),
        }))
        .unwrap();
    registry
        .register_condition(HostConditionBinding {
            id: "ready".into(),
            version: 1,
            shape: ExternalSignalShape::Boolean,
        })
        .unwrap();
    let profile: AutonomyProfile = serde_yaml::from_str(
        "validation: {checks: [{id: check, adapter: host.script, config: {}}]}",
    )
    .unwrap();
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
    (bound, s, calls)
}

#[derive(Default)]
pub(super) struct MemoryJournal {
    values: std::sync::Mutex<Vec<Value>>,
    fail: AtomicBool,
}
#[async_trait]
impl ValidationJournal for MemoryJournal {
    /// A failed pre-dispatch acknowledgement never admits a test effect.
    async fn checkpoint(&self, s: &ValidationDriverState) -> ai_agents_core::Result<()> {
        if self.fail.load(Ordering::SeqCst) {
            return Err(AgentError::Persistence("fixture failure".into()));
        }
        self.values.lock().unwrap().push(serde_json::to_value(s)?);
        Ok(())
    }
}
#[derive(Default)]
struct FakeExecutor {
    calls: AtomicUsize,
}
#[async_trait]
impl ValidationObservationExecutor for FakeExecutor {
    /// Only this test execution endpoint owns the simulated effect count.
    async fn execute(
        &self,
        r: &ValidationObservationRequest,
        _: &EvidenceIdentity,
    ) -> ai_agents_core::Result<ObservationResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match r {
            ValidationObservationRequest::Tool {
                tool, arguments, ..
            } => Ok(ObservationResult::Tool {
                record: Box::new(record(tool, arguments.clone(), json!({"value":1}))),
            }),
            _ => Ok(ObservationResult::Judge { score: 0.9 }),
        }
    }
}

// Persisted observations are reused exactly; stalled and changed-ID callbacks stop without a second execution.
#[tokio::test]
async fn validation_driver_binds_cached_requests_and_external_signals() {
    for mode in ["complete", "stalled", "changed"] {
        let (bound, s, _) = bound_script(mode);
        let check = &bound.checks[0];
        let mut i = identity(&s);
        i.config_identity = check.config_identity.clone();
        let mut driver = ValidationDriverState::new(check, i).unwrap();
        let journal = MemoryJournal::default();
        let executor = FakeExecutor::default();
        let cancel = ToolCancellationToken::new(Arc::new(AtomicBool::new(false)), None);
        let evidence = current(&s);
        let ctx = ValidationDriveContext {
            scope: &s,
            evidence: &evidence,
            extensions: &bound.extensions,
            executor: &executor,
            journal: &journal,
            limits: Default::default(),
            cancellation: &cancel,
            deadline: chrono::Utc::now() + chrono::Duration::seconds(2),
        };
        let result = driver.drive(check, &ctx).await;
        assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
        if mode == "complete" {
            assert!(matches!(
                result.unwrap(),
                ValidationDriverOutcome::Complete(_)
            ));
            let value = serde_json::to_value(&driver).unwrap();
            let mut restored =
                ValidationDriverState::restore(value, check, &s, &bound.extensions).unwrap();
            restored.drive(check, &ctx).await.unwrap();
            assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
        } else {
            assert!(result.is_err());
        }
    }
    let (bound, s, _) = bound_script("external");
    let check = &bound.checks[0];
    let mut i = identity(&s);
    i.config_identity = check.config_identity.clone();
    let mut driver = ValidationDriverState::new(check, i).unwrap();
    let journal = MemoryJournal::default();
    let executor = FakeExecutor::default();
    let cancel = ToolCancellationToken::new(Arc::new(AtomicBool::new(false)), None);
    let evidence = current(&s);
    let ctx = ValidationDriveContext {
        scope: &s,
        evidence: &evidence,
        extensions: &bound.extensions,
        executor: &executor,
        journal: &journal,
        limits: Default::default(),
        cancellation: &cancel,
        deadline: chrono::Utc::now() + chrono::Duration::seconds(2),
    };
    let wait = match driver.drive(check, &ctx).await.unwrap() {
        ValidationDriverOutcome::AwaitExternal(w) => w,
        _ => panic!("expected wait"),
    };
    assert!(
        driver
            .accept_external(
                ExternalSignal {
                    request_id: wait.request_id.clone(),
                    event_id: "event".into(),
                    target_revision: None,
                    payload: json!("not boolean")
                },
                &bound.extensions
            )
            .is_err()
    );
    driver
        .accept_external(
            ExternalSignal {
                request_id: wait.request_id,
                event_id: "event".into(),
                target_revision: None,
                payload: json!(true),
            },
            &bound.extensions,
        )
        .unwrap();
    assert!(matches!(
        driver.drive(check, &ctx).await.unwrap(),
        ValidationDriverOutcome::Complete(_)
    ));
    assert_eq!(executor.calls.load(Ordering::SeqCst), 0);
}

// Persistence failure prevents invocation; an in-flight restored record cannot automatically repeat effects.
#[tokio::test]
async fn validation_driver_failed_journal_never_invokes() {
    let (bound, s, _) = bound_script("complete");
    let check = &bound.checks[0];
    let mut i = identity(&s);
    i.config_identity = check.config_identity.clone();
    let mut driver = ValidationDriverState::new(check, i).unwrap();
    let journal = MemoryJournal::default();
    journal.fail.store(true, Ordering::SeqCst);
    let executor = FakeExecutor::default();
    let cancel = ToolCancellationToken::new(Arc::new(AtomicBool::new(false)), None);
    let evidence = current(&s);
    let ctx = ValidationDriveContext {
        scope: &s,
        evidence: &evidence,
        extensions: &bound.extensions,
        executor: &executor,
        journal: &journal,
        limits: Default::default(),
        cancellation: &cancel,
        deadline: chrono::Utc::now() + chrono::Duration::seconds(2),
    };
    assert!(driver.drive(check, &ctx).await.is_err());
    assert_eq!(executor.calls.load(Ordering::SeqCst), 0);
}

// Host control records preserve resource usage and invalidate the objective rather than approving old failed gates.
#[tokio::test]
async fn host_controls_use_cas_preserve_usage_and_cannot_revive_terminal_runs() {
    let store = ScopedTaskRunStore::in_memory("agent".into(), None, "config-v1".into()).unwrap();
    let mut snapshot = super::tests::checkpoint("run");
    let mut payload: TaskCheckpointPayload =
        serde_json::from_value(snapshot.payload.clone()).unwrap();
    payload.counters.llm_attempts = 3;
    snapshot = payload.bind(snapshot).unwrap();
    store.create(&snapshot).await.unwrap();
    let changed = store
        .apply_host_action(
            "run",
            0,
            HostControlAction::ModifyObjective {
                objective: "new objective".into(),
                reason: "host request".into(),
            },
            &AutonomyHostCeilings::default(),
        )
        .await
        .unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(changed.payload.clone()).unwrap();
    assert_eq!(payload.objective_revision, 1);
    assert_eq!(payload.counters.llm_attempts, 3);
    assert!(
        store
            .apply_host_action(
                "run",
                0,
                HostControlAction::Stop {
                    reason: "stale".into()
                },
                &AutonomyHostCeilings::default()
            )
            .await
            .is_err()
    );
    let mut limits = payload.limits.clone();
    limits.max_llm_calls = 2;
    assert!(
        store
            .apply_host_action(
                "run",
                1,
                HostControlAction::ReviseLimits {
                    limits,
                    reason: "invalid lower".into()
                },
                &AutonomyHostCeilings::default()
            )
            .await
            .is_err()
    );
    let mut limits = payload.limits;
    limits.max_llm_calls = 70;
    let changed = store
        .apply_host_action(
            "run",
            1,
            HostControlAction::ReviseLimits {
                limits,
                reason: "authorized increase".into(),
            },
            &AutonomyHostCeilings {
                max_llm_calls: Some(70),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let stopped = store
        .apply_host_action(
            "run",
            changed.revision,
            HostControlAction::Stop {
                reason: "host stopped".into(),
            },
            &AutonomyHostCeilings::default(),
        )
        .await
        .unwrap();
    assert_eq!(stopped.status, TaskRunStatus::Incomplete);
    assert!(
        store
            .apply_host_action(
                "run",
                stopped.revision,
                HostControlAction::ModifyObjective {
                    objective: "revive".into(),
                    reason: "invalid".into()
                },
                &AutonomyHostCeilings::default()
            )
            .await
            .is_err()
    );
}

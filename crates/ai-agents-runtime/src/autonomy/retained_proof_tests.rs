use super::evaluation_tests::{current, identity, scope};
use super::*;
use ai_agents_core::{Result, ToolCancellationToken};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

struct Journal;
#[async_trait]
impl ValidationJournal for Journal {
    async fn checkpoint(&self, _: &ValidationDriverState) -> Result<()> {
        Ok(())
    }
}
struct NoEffects;
#[async_trait]
impl ValidationObservationExecutor for NoEffects {
    async fn execute(
        &self,
        _: &ValidationObservationRequest,
        _: &EvidenceIdentity,
    ) -> Result<ObservationResult> {
        panic!("pure domain validation must not invoke effects")
    }
}

struct Domain {
    descriptor: AdapterDescriptor,
    calls: Arc<AtomicUsize>,
}
impl AutonomyValidator for Domain {
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.descriptor
    }
    fn validate_config(&self, _: &Value) -> Result<()> {
        Ok(())
    }
    fn evaluate(&self, input: &ValidationInput<'_>) -> Result<ValidationDecision> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ValidationDecision::Complete {
            outcome: match input.config["outcome"].as_str() {
                Some("unknown") => GateOutcome::Unknown,
                Some("fail") => GateOutcome::Fail,
                _ => GateOutcome::Pass,
            },
            reason: "domain_observed".into(),
            metrics: input
                .config
                .get("metrics")
                .cloned()
                .unwrap_or(json!({"ready":true,"count":3})),
            evidence_refs: vec![],
        })
    }
}

// Public primitives use the same prepared check/attempt identities as the retained runner.
fn prepared(bound: &BoundAutonomyProfile) -> EvaluationScope {
    let mut scope = scope();
    bound.prepare_scope(&mut scope);
    scope.validation_attempts = bound
        .checks
        .iter()
        .map(|check| (check.check.id.clone(), "attempt".into()))
        .collect();
    bound.capture_targets(&mut scope).unwrap();
    scope
}

// Pure domain callbacks still pass the real driver acknowledgement and publication boundaries.
async fn evaluate(
    bound: &BoundAutonomyProfile,
    scope: &EvaluationScope,
    evidence: &mut EvaluationEvidence,
    id: &str,
) -> ValidationDriverState {
    let check = bound
        .checks
        .iter()
        .find(|check| check.check.id == id)
        .unwrap();
    let mut identity = identity(scope);
    identity.config_identity = check.config_identity.clone();
    if let Some(target) = bound.extensions.validation_target(id) {
        identity.target = Some(target.into());
        identity.target_revision = scope.target_revisions.get(target).cloned();
    }
    let mut state = ValidationDriverState::new(check, identity).unwrap();
    let cancellation = ToolCancellationToken::new(Arc::new(AtomicBool::new(false)), None);
    state
        .drive(
            check,
            &ValidationDriveContext {
                scope,
                evidence,
                extensions: &bound.extensions,
                executor: &NoEffects,
                journal: &Journal,
                limits: Default::default(),
                cancellation: &cancellation,
                deadline: Some(chrono::Utc::now() + chrono::Duration::seconds(10)),
            },
        )
        .await
        .unwrap();
    evidence.collect_validation(&state).unwrap();
    state
}

// Unknown is not an observed empty object; known Fail metrics remain usable for absence checks and explicit progress.
#[tokio::test]
async fn retained_proof_unknown_cannot_become_absence_or_explicit_progress() {
    for mode in ["unavailable", "unknown", "fail"] {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut registry = AutonomyExtensions::builtins();
        registry
            .register_validator(Arc::new(Domain {
                descriptor: AdapterDescriptor {
                    id: "host.sources".into(),
                    contract_version: 1,
                    needs_judge: mode == "unavailable",
                    ..Default::default()
                },
                calls: calls.clone(),
            }))
            .unwrap();
        let profile: AutonomyProfile = serde_yaml::from_str(&format!(
            r#"
validation:
  checks:
    - id: sources
      adapter: host.sources
      required: false
      config: {{outcome: {mode}}}
    - id: absence
      adapter: builtin.evidence
      config: {{validator: sources, assertion: {{path: metrics.missing, exists: false}}}}
progress:
  config:
    signals:
      - {{id: sources, validator: sources, path: metrics.count, comparison: increase}}
"#
        ))
        .unwrap();
        let bound = registry
            .freeze()
            .bind(&profile, &Default::default())
            .unwrap();
        let scope = prepared(&bound);
        let mut evidence = current(&scope);
        evaluate(&bound, &scope, &mut evidence, "sources").await;
        evaluate(&bound, &scope, &mut evidence, "absence").await;
        let known = mode == "fail";
        assert_eq!(
            calls.load(Ordering::SeqCst),
            usize::from(mode != "unavailable")
        );
        assert_eq!(
            latest_validation(&evidence.validations, "absence", &scope)
                .unwrap()
                .outcome,
            if known {
                GateOutcome::Pass
            } else {
                GateOutcome::Unknown
            }
        );
        let observed = bound.observe_progress(&scope, &evidence).unwrap();
        assert_eq!(!observed.signals.is_empty(), known);
        evidence.progress = Some(observed.scoped(identity(&scope)));
        for gate in [
            CompletionGate::ProgressPath(PathAssertion {
                path: "metrics.signals.sources".into(),
                gte: Some(1.0),
                eq: None,
                gt: None,
                lte: None,
                lt: None,
                exists: None,
            }),
            CompletionGate::Not(Box::new(CompletionGate::ProgressPath(PathAssertion {
                path: "metrics.signals.sources".into(),
                exists: Some(false),
                eq: None,
                gte: None,
                gt: None,
                lte: None,
                lt: None,
            }))),
        ] {
            let outcome = CompletionGateEvaluator::default()
                .evaluate(&gate, &scope, &evidence)
                .unwrap()
                .outcome;
            if !known {
                assert_eq!(outcome, GateOutcome::Unknown);
            }
        }
    }
}

struct DomainProgress(AdapterDescriptor);
impl ProgressAdapter for DomainProgress {
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.0
    }
    fn validate_config(&self, _: &Value) -> Result<()> {
        Ok(())
    }
    fn observe(&self, input: &ProgressInput<'_>) -> Result<ProgressObservation> {
        let result =
            latest_validation(&input.evidence.validations, "combined", input.scope).unwrap();
        Ok(ProgressObservation {
            metrics: json!({"ready":true}),
            signals: BTreeMap::from([("domain".into(), json!(1))]),
            comparisons: BTreeMap::from([("domain".into(), SignalComparison::Increase)]),
            evidence_refs: vec![format!("check:combined:{}", result.identity.sequence)],
            validation_dependencies: BTreeMap::new(),
        })
    }
}

// Multiple domain sources and custom adapters compose without forcing every derived check to register one shared target.
fn domain_binding() -> (
    BoundAutonomyProfile,
    Arc<super::host_evidence_tests::Revision>,
    Arc<super::host_evidence_tests::Revision>,
) {
    let ticket = super::host_evidence_tests::Revision::new();
    let world = super::host_evidence_tests::Revision::new();
    let mut registry = AutonomyExtensions::builtins();
    for (id, dependencies) in [
        ("host.domain", vec![]),
        ("host.combine", vec!["ticket".into(), "world".into()]),
    ] {
        registry
            .register_validator(Arc::new(Domain {
                descriptor: AdapterDescriptor {
                    id: id.into(),
                    contract_version: 1,
                    checks: dependencies,
                    ..Default::default()
                },
                calls: Arc::new(AtomicUsize::new(0)),
            }))
            .unwrap();
    }
    registry
        .register_progress(Arc::new(DomainProgress(AdapterDescriptor {
            id: "host.progress".into(),
            contract_version: 1,
            checks: vec!["combined".into()],
            ..Default::default()
        })))
        .unwrap();
    registry
        .register_validation_target("ticket", "ticket/42", ticket.clone())
        .unwrap();
    registry
        .register_validation_target("world", "world/player", world.clone())
        .unwrap();
    let profile: AutonomyProfile = serde_yaml::from_str(
        r#"
validation:
  checks:
    - {id: ticket, adapter: host.domain, required: false}
    - {id: world, adapter: host.domain, required: false}
    - {id: combined, adapter: host.combine, required: false}
    - id: final
      adapter: builtin.evidence
      config: {validator: combined, assertion: {path: metrics.ready, eq: true}}
progress: {adapter: host.progress}
"#,
    )
    .unwrap();
    (
        registry
            .freeze()
            .bind(&profile, &Default::default())
            .unwrap(),
        ticket,
        world,
    )
}

// Transitive proof and custom progress survive normal completion, but either changed source invalidates every derived conclusion.
#[tokio::test]
async fn retained_proof_transitive_multitarget_and_custom_progress_freshness() {
    for changed in ["ticket", "world", "attempt", "replacement", "missing"] {
        let (bound, ticket, world) = domain_binding();
        let mut scope = prepared(&bound);
        let mut evidence = current(&scope);
        for id in ["ticket", "world", "combined", "final"] {
            evaluate(&bound, &scope, &mut evidence, id).await;
        }
        assert_eq!(
            bound.required_outcome(&scope, &evidence).unwrap(),
            GateOutcome::Pass
        );
        evidence.progress = Some(
            bound
                .observe_progress(&scope, &evidence)
                .unwrap()
                .scoped(identity(&scope)),
        );
        let gate = CompletionGate::ProgressPath(PathAssertion {
            path: "metrics.ready".into(),
            eq: Some(json!(true)),
            gte: None,
            gt: None,
            lte: None,
            lt: None,
            exists: None,
        });
        let derived = evidence.validations[2].identity.clone();
        evidence.commands.push(ScopedObservation {
            identity: derived.clone(),
            complete: true,
            value: CommandObservation {
                check_id: "combined".into(),
                command: "verify-report".into(),
                exit_code: Some(0),
                termination: "exited".into(),
                record: super::evaluation_tests::record(
                    "command",
                    json!({"command":"verify-report"}),
                    json!({"exit_code":0}),
                ),
            },
        });
        evidence.diagnostics.push(ScopedObservation {
            identity: derived.clone(),
            complete: true,
            value: DiagnosticsObservation {
                check_id: "combined".into(),
                available: true,
                complete: true,
                severity_counts: BTreeMap::from([("error".into(), 0)]),
                record: super::evaluation_tests::record(
                    "diagnostics",
                    json!({}),
                    json!({"available":true}),
                ),
            },
        });
        let judge = JudgeGate {
            llm: None,
            pass_threshold: None,
            criteria: vec!["quality".into()],
        };
        let judged = ValidationResult {
            check_id: "judge".into(),
            identity: derived,
            outcome: GateOutcome::Pass,
            reason: "judged".into(),
            metrics: json!({"judge_config_identity":canonical_identity(&serde_json::to_value(&judge).unwrap()).unwrap()}),
            evidence_refs: vec![],
        };
        scope
            .validation_bindings
            .insert("judge".into(), judged.identity.config_identity.clone());
        scope
            .validation_attempts
            .insert("judge".into(), judged.identity.attempt.clone().unwrap());
        scope.validation_dependencies.insert(
            "judge".into(),
            judged
                .identity
                .validation_dependencies
                .keys()
                .cloned()
                .collect(),
        );
        evidence.validations.push(judged.clone());
        evidence.judges.push(judged);
        let gates = vec![
            gate,
            CompletionGate::CommandExit(CommandExitGate {
                command: "verify-report".into(),
                code: 0,
            }),
            CompletionGate::DiagnosticsClear(DiagnosticsClearGate {
                severity: "error".into(),
            }),
            CompletionGate::Judge(judge),
        ];
        for gate in &gates {
            assert_eq!(
                CompletionGateEvaluator::default()
                    .evaluate(gate, &scope, &evidence)
                    .unwrap()
                    .outcome,
                GateOutcome::Pass
            );
        }
        match changed {
            "ticket" => ticket.change(),
            "world" => world.change(),
            "attempt" => {
                scope
                    .validation_attempts
                    .insert("ticket".into(), "replacement".into());
            }
            "replacement" => {
                let mut replacement = evidence.validations[0].clone();
                replacement.identity.sequence += 1;
                evidence.validations.push(replacement);
            }
            "missing" => {
                evidence.validations.remove(0);
            }
            _ => unreachable!(),
        }
        bound.capture_targets(&mut scope).unwrap();
        assert!(
            latest_validation(&evidence.validations, "final", &scope).is_none(),
            "{changed}"
        );
        assert_eq!(
            bound.required_outcome(&scope, &evidence).unwrap(),
            GateOutcome::Unknown
        );
        for gate in &gates {
            assert_eq!(
                CompletionGateEvaluator::default()
                    .evaluate(gate, &scope, &evidence)
                    .unwrap()
                    .outcome,
                GateOutcome::Unknown,
                "{changed}: {gate:?}"
            );
        }
    }
}

// Shared dependency subgraphs are memoized; a valid declaration-ordered chain does not expand into exponential work.
#[test]
fn retained_proof_shared_dependency_graph_is_bounded_and_cycles_reject() {
    let mut scope = scope();
    let mut results: Vec<ValidationResult> = vec![];
    for n in 0usize..48 {
        let id = format!("check-{n}");
        let mut identity = identity(&scope);
        let deps: Vec<_> = results.iter().rev().take(2).collect();
        identity.validation_dependencies = deps
            .iter()
            .map(|r| (r.check_id.clone(), r.proof_identity().unwrap()))
            .collect();
        scope.validation_dependencies.insert(
            id.clone(),
            deps.iter().map(|r| r.check_id.clone()).collect(),
        );
        scope
            .validation_bindings
            .insert(id.clone(), identity.config_identity.clone());
        scope
            .validation_attempts
            .insert(id.clone(), identity.attempt.clone().unwrap());
        results.push(ValidationResult {
            check_id: id,
            identity,
            outcome: GateOutcome::Pass,
            reason: "observed".into(),
            metrics: json!({}),
            evidence_refs: vec![],
        });
    }
    assert!(latest_validation(&results, "check-47", &scope).is_some());
    let cyclic = results[47].proof_identity().unwrap();
    results[0]
        .identity
        .validation_dependencies
        .insert("check-47".into(), cyclic);
    assert!(latest_validation(&results, "check-47", &scope).is_none());
}

// A suspended dependent batch must not execute its remaining work before discovering that its original input proof is stale.
#[tokio::test]
async fn retained_proof_cached_publication_rejects_replaced_inputs() {
    let (bound, ticket, _) = domain_binding();
    let mut scope = prepared(&bound);
    let mut evidence = current(&scope);
    evaluate(&bound, &scope, &mut evidence, "ticket").await;
    evaluate(&bound, &scope, &mut evidence, "world").await;
    let mut state = evaluate(&bound, &scope, &mut evidence, "combined").await;
    ticket.change();
    bound.capture_targets(&mut scope).unwrap();
    let cancellation = ToolCancellationToken::new(Arc::new(AtomicBool::new(false)), None);
    assert!(
        state
            .drive(
                &bound.checks[2],
                &ValidationDriveContext {
                    scope: &scope,
                    evidence: &evidence,
                    extensions: &bound.extensions,
                    executor: &NoEffects,
                    journal: &Journal,
                    limits: Default::default(),
                    cancellation: &cancellation,
                    deadline: Some(chrono::Utc::now() + chrono::Duration::seconds(10)),
                }
            )
            .await
            .is_err()
    );
    assert!(!state.publication_acknowledged());
}

struct Batch(AdapterDescriptor);
impl AutonomyValidator for Batch {
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.0
    }
    fn validate_config(&self, _: &Value) -> Result<()> {
        Ok(())
    }
    fn evaluate(&self, input: &ValidationInput<'_>) -> Result<ValidationDecision> {
        if input.observations.len() == 2 {
            return Ok(ValidationDecision::Complete {
                outcome: GateOutcome::Pass,
                reason: "observed".into(),
                metrics: json!({"ready":true}),
                evidence_refs: input.observations.keys().cloned().collect(),
            });
        }
        Ok(ValidationDecision::NeedObservations {
            requests: ["one", "two"]
                .into_iter()
                .map(|id| ValidationObservationRequest::Tool {
                    id: id.into(),
                    tool: "probe".into(),
                    arguments: json!({}),
                    host_binding: None,
                })
                .collect(),
            checkpoint: json!({}),
        })
    }
}
struct CountingExecutor {
    extensions: Arc<FrozenAutonomyExtensions>,
    calls: AtomicUsize,
}
#[async_trait]
impl ValidationObservationExecutor for CountingExecutor {
    fn capture_scope(&self, scope: &EvaluationScope) -> Result<EvaluationScope> {
        let mut scope = scope.clone();
        self.extensions.refresh_targets(&mut scope)?;
        Ok(scope)
    }
    async fn execute(
        &self,
        request: &ValidationObservationRequest,
        _: &EvidenceIdentity,
    ) -> Result<ObservationResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut record = super::evaluation_tests::record("probe", json!({}), json!({}));
        record.call_id = request.id().into();
        record
            .metadata
            .insert("classification".into(), json!({"read_only":true}));
        Ok(ObservationResult::Tool {
            record: Box::new(record),
        })
    }
}
struct PrefixJournal {
    fail: AtomicBool,
    change: Option<Arc<super::host_evidence_tests::Revision>>,
}
#[async_trait]
impl ValidationJournal for PrefixJournal {
    async fn checkpoint(&self, state: &ValidationDriverState) -> Result<()> {
        if state.completed.len() == 1
            && state.in_flight.is_none()
            && self.fail.swap(false, Ordering::SeqCst)
        {
            return Err(ai_agents_core::AgentError::Persistence(
                "first result acknowledgement lost".into(),
            ));
        }
        if state.completed.len() == 1
            && state.in_flight.is_some()
            && let Some(source) = &self.change
        {
            source.change();
        }
        Ok(())
    }
}

// A retained prefix is neither replayed nor followed by an effect using invalidated upstream proof, including a journal-ACK race.
#[tokio::test]
async fn retained_proof_pending_batch_rechecks_pinned_inputs_before_effects() {
    for mode in ["safe", "between", "ack"] {
        let source = super::host_evidence_tests::Revision::new();
        let mut registry = AutonomyExtensions::builtins();
        registry
            .register_validator(Arc::new(Domain {
                descriptor: AdapterDescriptor {
                    id: "host.ticket".into(),
                    contract_version: 1,
                    ..Default::default()
                },
                calls: Arc::new(AtomicUsize::new(0)),
            }))
            .unwrap();
        registry
            .register_validator(Arc::new(Batch(AdapterDescriptor {
                id: "host.batch".into(),
                contract_version: 1,
                checks: vec!["ticket".into()],
                tools: vec!["probe".into()],
                ..Default::default()
            })))
            .unwrap();
        registry
            .register_validation_target("ticket", "ticket/42", source.clone())
            .unwrap();
        let profile:AutonomyProfile=serde_yaml::from_str("validation: {checks: [{id: ticket, adapter: host.ticket}, {id: effect, adapter: host.batch}]}").unwrap();
        let bound = registry
            .freeze()
            .bind(
                &profile,
                &ValidationCapabilities {
                    tools: ["probe".into()].into_iter().collect(),
                    ..Default::default()
                },
            )
            .unwrap();
        let scope = prepared(&bound);
        let mut evidence = current(&scope);
        evaluate(&bound, &scope, &mut evidence, "ticket").await;
        let check = &bound.checks[1];
        let mut identity = identity(&scope);
        identity.config_identity = check.config_identity.clone();
        let mut state = ValidationDriverState::new(check, identity).unwrap();
        let executor = CountingExecutor {
            extensions: bound.extensions.clone(),
            calls: AtomicUsize::new(0),
        };
        let cancellation = ToolCancellationToken::new(Arc::new(AtomicBool::new(false)), None);
        let journal = PrefixJournal {
            fail: AtomicBool::new(true),
            change: None,
        };
        let context = ValidationDriveContext {
            scope: &scope,
            evidence: &evidence,
            extensions: &bound.extensions,
            executor: &executor,
            journal: &journal,
            limits: Default::default(),
            cancellation: &cancellation,
            deadline: Some(chrono::Utc::now() + chrono::Duration::seconds(10)),
        };
        assert!(state.drive(check, &context).await.is_err());
        assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
        assert!(
            !state.completed["one"]
                .identity
                .as_ref()
                .unwrap()
                .validation_dependencies
                .is_empty()
        );
        let mut retained = ValidationDriverState::restore_retained(
            serde_json::to_value(&state).unwrap(),
            &state,
            check,
            &scope,
            &bound.extensions,
        )
        .unwrap();
        if mode == "between" {
            source.change();
        }
        let next = PrefixJournal {
            fail: AtomicBool::new(false),
            change: (mode == "ack").then_some(source),
        };
        let result = retained
            .drive(
                check,
                &ValidationDriveContext {
                    journal: &next,
                    ..context
                },
            )
            .await;
        assert_eq!(result.is_ok(), mode == "safe");
        assert_eq!(
            executor.calls.load(Ordering::SeqCst),
            if mode == "safe" { 2 } else { 1 }
        );
        assert_eq!(retained.completed.len(), if mode == "safe" { 2 } else { 1 });
    }
}

use super::*;
use crate::AgentBuilder;
use ai_agents_core::{Result, Tool, ToolExecutionContext, ToolResult, ToolSafetyMetadata};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Store {
    inner: Arc<ScopedTaskRunStore>,
    expire: bool,
    drift: Option<Arc<super::host_evidence_tests::Revision>>,
    invocation_drift: Option<Arc<super::host_evidence_tests::Revision>>,
    admissions: AtomicUsize,
    final_copies: AtomicUsize,
}
#[async_trait]
impl TaskRunStore for Store {
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
        let saved = self.inner.mutate(id, mutation).await?;
        if let Some(bounds) = ai_agents_llm::current_invocation_bounds() {
            let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload.clone())?;
            if payload.reservations.iter().any(|reservation| {
                reservation.state == TaskEffectState::Dispatched
                    && payload
                        .adapters
                        .iter()
                        .any(|adapter| adapter.id == format!("invocation:{}", reservation.id))
            }) && self.admissions.fetch_add(1, Ordering::SeqCst) == 0
            {
                if let Some(source) = &self.invocation_drift {
                    source.change();
                }
                if self.expire {
                    // Do not yield to the driver's monitor: final admission must independently refuse dispatch after this acknowledgement.
                    std::thread::sleep(bounds.remaining() + std::time::Duration::from_millis(2));
                }
            }
        }
        if let Some(source) = &self.drift {
            let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload.clone())?;
            if matches!(mutation, TaskRunMutation::Checkpoint { .. })
                && payload.adapters.iter().any(|adapter| {
                    adapter.id == "final"
                        && !adapter.state["_task_validation_driver"]["result"].is_null()
                })
                && self.final_copies.fetch_add(1, Ordering::SeqCst) == 1
            {
                source.change();
            }
        }
        Ok(saved)
    }
}

struct Probe(Arc<AtomicUsize>);
#[async_trait]
impl Tool for Probe {
    fn id(&self) -> &str {
        "probe"
    }
    fn name(&self) -> &str {
        "probe"
    }
    fn description(&self) -> &str {
        "Isolated invocation admission probe"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object"})
    }
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: true,
            concurrency_safe: true,
            ..Default::default()
        }
    }
    async fn execute(&self, _: Value, _: ToolExecutionContext) -> ToolResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        ToolResult::ok("observed")
    }
}
struct Observed(AdapterDescriptor);
impl AutonomyValidator for Observed {
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.0
    }
    fn validate_config(&self, _: &Value) -> Result<()> {
        Ok(())
    }
    fn evaluate(&self, input: &ValidationInput<'_>) -> Result<ValidationDecision> {
        if self.0.needs_judge && input.observations.is_empty() {
            return Ok(ValidationDecision::NeedObservations {
                requests: vec![ValidationObservationRequest::Judge {
                    id: "judge".into(),
                    config: serde_json::from_value(json!({"criteria":["Correctness"]})).unwrap(),
                    response: "candidate".into(),
                    objective: "verify report".into(),
                }],
                checkpoint: json!({}),
            });
        }
        if !self.0.tools.is_empty() && input.observations.is_empty() {
            return Ok(ValidationDecision::NeedObservations {
                requests: vec![ValidationObservationRequest::Tool {
                    id: "observe".into(),
                    tool: "probe".into(),
                    arguments: json!({}),
                    host_binding: None,
                }],
                checkpoint: json!({}),
            });
        }
        Ok(ValidationDecision::Complete {
            outcome: GateOutcome::Pass,
            reason: "domain_observed".into(),
            metrics: json!({"ready":true}),
            evidence_refs: input.observations.keys().cloned().collect(),
        })
    }
}

#[derive(Debug)]
struct Allowed(std::sync::atomic::AtomicBool);
impl TaskFinalAdmission for Allowed {
    fn validate(&self) -> Result<()> {
        if self.0.load(Ordering::SeqCst) {
            Ok(())
        } else {
            Err(ai_agents_core::AgentError::Config(
                "test proof invalidated".into(),
            ))
        }
    }
}

// Framework-spawned child tasks retain restriction-only proof, original bounds and the actual mutation tree without a new ledger.
#[tokio::test]
async fn retained_admission_spawn_preserves_proof_bounds_and_mutation_tree() {
    use ai_agents_core::{ToolCallSource, ToolExecutionRequest, ToolInvoker};
    use ai_agents_llm::{InvocationBounds, scope_invocation_bounds, scope_invocation_evidence};
    for allowed in [false, true] {
        let f = super::controller_tests::fixture("mutating_external");
        let clock = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let bounds = InvocationBounds::new(
            std::time::SystemTime::now() + std::time::Duration::from_secs(5),
            std::time::Instant::now() + std::time::Duration::from_secs(5),
        );
        let record = scope_invocation_evidence(
            vec![Arc::new(Allowed(std::sync::atomic::AtomicBool::new(
                allowed,
            )))],
            scope_invocation_bounds(
                Some(bounds),
                super::invocation::InvocationScope::observation(clock.clone()).scope(async {
                    let inherited = super::invocation::InvocationScope::capture();
                    let mut tasks = tokio::task::JoinSet::new();
                    let agent = f.agent.clone();
                    tasks.spawn(async move {
                        super::scope_inherited_execution(None, inherited, async move {
                            let longer = InvocationBounds::new(
                                std::time::SystemTime::now() + std::time::Duration::from_secs(60),
                                std::time::Instant::now() + std::time::Duration::from_secs(60),
                            );
                            scope_invocation_bounds(Some(longer), async {
                                assert!(
                                    ai_agents_llm::current_invocation_bounds()
                                        .unwrap()
                                        .remaining()
                                        <= std::time::Duration::from_secs(5)
                                );
                                agent
                                    .invoke_tool(ToolExecutionRequest::new(
                                        "child",
                                        "mutate",
                                        json!({}),
                                        ToolCallSource::Task,
                                    ))
                                    .await
                                    .unwrap()
                            })
                            .await
                        })
                        .await
                    });
                    tasks.join_next().await.unwrap().unwrap()
                }),
            ),
        )
        .await;
        assert_eq!(record.executed, allowed);
        assert_eq!(f.calls.load(Ordering::SeqCst), usize::from(allowed));
        assert_eq!(clock.load(Ordering::SeqCst) > 0, allowed);
    }
}

// Actual memory/SQLite invocation acknowledgement crosses the check deadline while whole-run allowance remains available.
#[tokio::test]
async fn retained_admission_original_check_bounds_reach_real_tool_and_provider() {
    for sqlite in super::continuation_test_support::backends() {
        for judge in [false, true] {
            for expire in [false, true] {
                let provider = super::runner_tests::RecordingProvider::new(&[
                    "done",
                    r#"{"overall_score":1}"#,
                ]);
                let calls = Arc::new(AtomicUsize::new(0));
                let mut registry = AutonomyExtensions::builtins();
                registry
                    .register_validator(Arc::new(Observed(AdapterDescriptor {
                        id: "host.observe".into(),
                        contract_version: 1,
                        tools: vec!["probe".into()],
                        ..Default::default()
                    })))
                    .unwrap();
                let agent = Arc::new(
                    AgentBuilder::new()
                        .system_prompt("test")
                        .llm(provider.clone())
                        .tool(Arc::new(Probe(calls.clone())))
                        .autonomy_extensions(registry)
                        .build()
                        .unwrap(),
                );
                let inner = tokio::time::timeout(
                    std::time::Duration::from_secs(30),
                    super::continuation_test_support::store(agent.info().id, "bounds.v1", *sqlite),
                )
                .await
                .unwrap_or_else(|_| {
                    panic!(
                        "store construction stalled: sqlite={sqlite} judge={judge} expire={expire}"
                    )
                });
                let store = Arc::new(Store {
                    inner: inner.clone(),
                    expire,
                    drift: None,
                    invocation_drift: None,
                    admissions: AtomicUsize::new(0),
                    final_copies: AtomicUsize::new(0),
                });
                let mut config = super::runner_tests::config(4);
                config.defaults.max_turns = Some(1);
                config.defaults.validation = Some(serde_yaml::from_str(if judge {
                    "checks: [{id: check, adapter: builtin.judge, timeout_seconds: 1, config: {criteria: [Correctness]}}]"
                } else { "checks: [{id: check, adapter: host.observe, timeout_seconds: 1}]" }).unwrap());
                config.defaults.completion = Some(CompletionGate::ValidatorPassed("check".into()));
                let runner = AutonomyRunner::try_new(
                    agent.clone(),
                    config,
                    Default::default(),
                    store.clone(),
                    "bounds.v1".into(),
                )
                .unwrap();
                let result = tokio::time::timeout(
                    std::time::Duration::from_secs(30),
                    runner.run("verify report", None),
                )
                .await
                .unwrap_or_else(|_| {
                    panic!("run stalled: sqlite={sqlite} judge={judge} expire={expire}")
                });
                assert!(store.admissions.load(Ordering::SeqCst) >= 1);
                assert_eq!(calls.load(Ordering::SeqCst), usize::from(!judge && !expire));
                assert_eq!(
                    provider.calls.load(Ordering::SeqCst),
                    1 + usize::from(judge && !expire)
                );
                if expire {
                    assert!(
                        !result
                            .as_ref()
                            .is_ok_and(|result| result.run.status == TaskRunStatus::Completed)
                    );
                    let runs =
                        tokio::time::timeout(std::time::Duration::from_secs(5), inner.list())
                            .await
                            .unwrap_or_else(|_| {
                                panic!("list stalled: sqlite={sqlite} judge={judge}")
                            })
                            .unwrap();
                    let snapshot = tokio::time::timeout(
                        std::time::Duration::from_secs(5),
                        inner.load(&runs[0].key.run_id),
                    )
                    .await
                    .unwrap_or_else(|_| panic!("load stalled: sqlite={sqlite} judge={judge}"))
                    .unwrap()
                    .unwrap();
                    let payload: TaskCheckpointPayload =
                        serde_json::from_value(snapshot.payload).unwrap();
                    let certified = payload.reservations.iter().any(|reservation| {
                        reservation
                            .result
                            .as_ref()
                            .is_some_and(|value| value["not_invoked"] == json!(true))
                    });
                    if certified {
                        assert_eq!(payload.counters.tool_attempts, 0);
                        assert_eq!(payload.counters.llm_attempts, 1);
                    } else {
                        // The monitor may win while SQLite awaits the post-admission read or refund acknowledgement.
                        // No body ran, but absent a certified receipt the original reservation and owner must remain protected.
                        assert!(payload.reservations.iter().any(|reservation| matches!(
                            reservation.state,
                            TaskEffectState::Dispatched | TaskEffectState::Uncertain
                        )));
                        assert_eq!(snapshot.status, TaskRunStatus::RecoveryRequired);
                        assert!(
                            tokio::time::timeout(
                                std::time::Duration::from_secs(2),
                                crate::Agent::chat(agent.as_ref(), "unrelated")
                            )
                            .await
                            .unwrap_or_else(|_| panic!(
                                "protected runtime admission stalled: sqlite={sqlite} judge={judge}"
                            ))
                            .is_err()
                        );
                        assert_eq!(payload.counters.tool_attempts, u64::from(!judge));
                        assert_eq!(payload.counters.llm_attempts, 1 + u64::from(judge));
                    }
                } else {
                    assert_eq!(result.unwrap().run.status, TaskRunStatus::Completed);
                }
            }
        }
    }
}

// Completion recaptures a new host revision after controller storage, but must not relabel an already-derived conclusion as fresh.
#[tokio::test]
async fn retained_admission_derived_proof_cannot_survive_observed_precompletion_drift() {
    for sqlite in super::continuation_test_support::backends() {
        for drift in [false, true] {
            let source = super::host_evidence_tests::Revision::new();
            let mut registry = AutonomyExtensions::builtins();
            registry
                .register_validator(Arc::new(Observed(AdapterDescriptor {
                    id: "host.ticket".into(),
                    contract_version: 1,
                    requires_host_revision: true,
                    ..Default::default()
                })))
                .unwrap();
            registry
                .register_validation_target("ticket", "ticket/42", source.clone())
                .unwrap();
            let provider = super::runner_tests::RecordingProvider::new(&["resolved"]);
            let agent = Arc::new(
                AgentBuilder::new()
                    .system_prompt("test")
                    .llm(provider.clone())
                    .autonomy_extensions(registry)
                    .build()
                    .unwrap(),
            );
            let inner =
                super::continuation_test_support::store(agent.info().id, "derived.v1", *sqlite)
                    .await;
            let store = Arc::new(Store {
                inner,
                expire: false,
                drift: drift.then_some(source),
                invocation_drift: None,
                admissions: AtomicUsize::new(0),
                final_copies: AtomicUsize::new(0),
            });
            let mut config = super::runner_tests::config(1);
            config.defaults.validation = Some(
                serde_yaml::from_str(
                    r#"checks:
- {id: ticket, adapter: host.ticket, required: false}
- id: final
  adapter: builtin.evidence
  config: {validator: ticket, assertion: {path: metrics.ready, eq: true}}
"#,
                )
                .unwrap(),
            );
            config.defaults.completion = Some(CompletionGate::ValidatorPassed("final".into()));
            let runner = AutonomyRunner::try_new(
                agent,
                config,
                Default::default(),
                store.clone(),
                "derived.v1".into(),
            )
            .unwrap();
            let result = runner.run("resolve ticket", None).await.unwrap();
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            if drift {
                assert!(store.final_copies.load(Ordering::SeqCst) >= 2);
            }
            assert_eq!(result.run.status == TaskRunStatus::Completed, !drift);
        }
    }
}

// Pinned upstream host proof is sampled again after actual invocation storage, not only after the driver's own journal.
#[tokio::test]
async fn retained_admission_upstream_drift_blocks_actual_tool_and_provider() {
    for sqlite in super::continuation_test_support::backends() {
        for judge in [false, true] {
            for drift in [false, true] {
                let source = super::host_evidence_tests::Revision::new();
                let mut registry = AutonomyExtensions::builtins();
                registry
                    .register_validator(Arc::new(Observed(AdapterDescriptor {
                        id: "host.ticket".into(),
                        contract_version: 1,
                        requires_host_revision: true,
                        ..Default::default()
                    })))
                    .unwrap();
                registry
                    .register_validator(Arc::new(Observed(AdapterDescriptor {
                        id: "host.effect".into(),
                        contract_version: 1,
                        checks: vec!["ticket".into()],
                        needs_judge: judge,
                        tools: if judge { vec![] } else { vec!["probe".into()] },
                        ..Default::default()
                    })))
                    .unwrap();
                registry
                    .register_validation_target("ticket", "ticket/42", source.clone())
                    .unwrap();
                let provider = super::runner_tests::RecordingProvider::new(&[
                    "done",
                    r#"{"overall_score":1}"#,
                ]);
                let calls = Arc::new(AtomicUsize::new(0));
                let agent = Arc::new(
                    AgentBuilder::new()
                        .system_prompt("test")
                        .llm(provider.clone())
                        .tool(Arc::new(Probe(calls.clone())))
                        .autonomy_extensions(registry)
                        .build()
                        .unwrap(),
                );
                let inner = super::continuation_test_support::store(
                    agent.info().id,
                    "proof-admission.v1",
                    *sqlite,
                )
                .await;
                let store = Arc::new(Store {
                    inner,
                    expire: false,
                    drift: None,
                    invocation_drift: drift.then_some(source),
                    admissions: AtomicUsize::new(0),
                    final_copies: AtomicUsize::new(0),
                });
                let mut config = super::runner_tests::config(4);
                config.defaults.max_turns = Some(1);
                config.defaults.validation=Some(serde_yaml::from_str("checks: [{id: ticket, adapter: host.ticket, required: false}, {id: effect, adapter: host.effect}]").unwrap());
                config.defaults.completion = Some(CompletionGate::ValidatorPassed("effect".into()));
                let runner = AutonomyRunner::try_new(
                    agent,
                    config,
                    Default::default(),
                    store.clone(),
                    "proof-admission.v1".into(),
                )
                .unwrap();
                let result = runner.run("resolve ticket", None).await;
                assert!(store.admissions.load(Ordering::SeqCst) >= 1);
                assert_eq!(calls.load(Ordering::SeqCst), usize::from(!judge && !drift));
                assert_eq!(
                    provider.calls.load(Ordering::SeqCst),
                    1 + usize::from(judge && !drift)
                );
                if drift {
                    assert!(
                        !result
                            .as_ref()
                            .is_ok_and(|result| result.run.status == TaskRunStatus::Completed)
                    );
                } else {
                    assert_eq!(result.unwrap().run.status, TaskRunStatus::Completed);
                }
            }
        }
    }
}

use ai_agents_core::{
    Tool, ToolCancellationToken, ToolExecutionContext, ToolResult, ToolSafetyMetadata,
};
use ai_agents_runtime::{AgentBuilder, autonomy::*};
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
    async fn checkpoint(&self, _: &ValidationDriverState) -> ai_agents_core::Result<()> {
        Ok(())
    }
}

struct Sequence(AdapterDescriptor);
impl AutonomyValidator for Sequence {
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.0
    }
    fn validate_config(&self, _: &Value) -> ai_agents_core::Result<()> {
        Ok(())
    }
    fn evaluate(&self, input: &ValidationInput<'_>) -> ai_agents_core::Result<ValidationDecision> {
        let names = input.config["steps"].as_array().unwrap();
        if input.observations.len() < names.len() {
            return Ok(ValidationDecision::NeedObservations {
                requests: names
                    .iter()
                    .enumerate()
                    .map(|(n, name)| ValidationObservationRequest::Tool {
                        id: format!("step-{n}"),
                        tool: name.as_str().unwrap().into(),
                        arguments: json!({}),
                        host_binding: None,
                    })
                    .collect(),
                checkpoint: json!({}),
            });
        }
        Ok(ValidationDecision::Complete {
            outcome: GateOutcome::Pass,
            reason: "observed".into(),
            metrics: json!({"count":3}),
            evidence_refs: input.observations.keys().cloned().collect(),
        })
    }
}

struct Probe {
    id: &'static str,
    calls: Arc<AtomicUsize>,
    started: Arc<tokio::sync::Notify>,
}
#[async_trait]
impl Tool for Probe {
    fn id(&self) -> &str {
        self.id
    }
    fn name(&self) -> &str {
        self.id
    }
    fn description(&self) -> &str {
        "Isolated retained SDK observation"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object"})
    }
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: self.id == "read",
            concurrency_safe: true,
            ..Default::default()
        }
    }
    async fn execute(&self, _: Value, _: ToolExecutionContext) -> ToolResult {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.id == "pending_mutation" {
            self.started.notify_one();
            std::future::pending().await
        } else if self.id == "failed_mutation" {
            ToolResult::error("mutation attempted but failed")
        } else {
            ToolResult::ok("observed")
        }
    }
}

struct NestedRead {
    child: Arc<ai_agents_runtime::RuntimeAgent>,
}
#[async_trait]
impl Tool for NestedRead {
    fn id(&self) -> &str {
        "nested_read"
    }
    fn name(&self) -> &str {
        "nested_read"
    }
    fn description(&self) -> &str {
        "Read-shaped wrapper around a managed child mutation"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object"})
    }
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: true,
            ..Default::default()
        }
    }
    async fn execute(&self, _: Value, _: ToolExecutionContext) -> ToolResult {
        use ai_agents_core::ToolInvoker;
        let record = self
            .child
            .invoke_tool(ai_agents_core::ToolExecutionRequest::new(
                "child-mutation",
                "mutation",
                json!({}),
                ai_agents_core::ToolCallSource::Task,
            ))
            .await
            .unwrap();
        assert!(record.executed);
        ToolResult::ok("child completed")
    }
}

// Only public APIs are used: independent SDK users can retain real execution without private runtime helpers.
fn fixture(
    steps: &[&str],
) -> (
    RuntimeObservationExecutor,
    BoundAutonomyProfile,
    EvaluationScope,
    Arc<AtomicUsize>,
    Arc<tokio::sync::Notify>,
) {
    let mut registry = AutonomyExtensions::builtins();
    registry
        .register_validator(Arc::new(Sequence(AdapterDescriptor {
            id: "host.sequence".into(),
            contract_version: 1,
            tools: vec![
                "read".into(),
                "mutation".into(),
                "failed_mutation".into(),
                "pending_mutation".into(),
                "nested_read".into(),
            ],
            ..Default::default()
        })))
        .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let mut provider = ai_agents_llm::mock::MockLLMProvider::new("unused");
    provider.set_response("unused");
    let mut builder = AgentBuilder::new()
        .system_prompt("test")
        .llm(Arc::new(provider))
        .autonomy_extensions(registry);
    for id in ["read", "mutation", "failed_mutation", "pending_mutation"] {
        builder = builder.tool(Arc::new(Probe {
            id,
            calls: calls.clone(),
            started: started.clone(),
        }));
    }
    let mut child_provider = ai_agents_llm::mock::MockLLMProvider::new("unused child");
    child_provider.set_response("unused");
    let child = Arc::new(
        AgentBuilder::new()
            .system_prompt("test")
            .llm(Arc::new(child_provider))
            .tool(Arc::new(Probe {
                id: "mutation",
                calls: calls.clone(),
                started: started.clone(),
            }))
            .build()
            .unwrap(),
    );
    let agent = Arc::new(
        builder
            .tool(Arc::new(NestedRead { child }))
            .build()
            .unwrap(),
    );
    let profile: AutonomyProfile = serde_json::from_value(json!({"validation":{"checks":[{
        "id":"check", "adapter":"host.sequence", "timeout_seconds":10, "config":{"steps":steps}
    }]}, "progress":{"config":{"signals":[{"id":"count","validator":"check","path":"metrics.count","comparison":"increase"}]}}})).unwrap();
    let bound = agent
        .autonomy_extensions()
        .bind(
            &profile,
            &ValidationCapabilities {
                tools: [
                    "read",
                    "mutation",
                    "failed_mutation",
                    "pending_mutation",
                    "nested_read",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
                ..Default::default()
            },
        )
        .unwrap();
    let mut scope = EvaluationScope {
        key: TaskRunKey {
            agent_id: agent.info().id,
            run_id: "sdk".into(),
        },
        objective_revision: 0,
        cycle: 1,
        stage: None,
        mutation_generation: 0,
        target_revisions: BTreeMap::new(),
        target_generations: BTreeMap::new(),
        validation_bindings: BTreeMap::new(),
        validation_attempts: BTreeMap::from([("check".into(), "attempt".into())]),
        validation_dependencies: BTreeMap::new(),
    };
    bound.prepare_scope(&mut scope);
    (
        RuntimeObservationExecutor {
            agent,
            profile: None,
            scope_mode: "task".into(),
            run_revision: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            validator: "host.sequence".into(),
            contract_version: 1,
        },
        bound,
        scope,
        calls,
        started,
    )
}

// Managed reads use the actual runtime clock; stale audit metrics cannot enter explicit progress.
#[tokio::test]
async fn retained_sdk_real_mutations_invalidate_reads_but_fresh_reads_complete() {
    for steps in [
        vec!["read", "mutation"],
        vec!["mutation", "read"],
        vec!["read", "failed_mutation"],
        vec!["read", "nested_read"],
    ] {
        let (executor, bound, scope, calls, _) = fixture(&steps);
        let check = &bound.checks[0];
        let identity = EvidenceIdentity {
            key: scope.key.clone(),
            objective_revision: 0,
            cycle: 1,
            sequence: 1,
            stage: None,
            attempt: Some("attempt".into()),
            config_identity: check.config_identity.clone(),
            target: None,
            target_revision: None,
            mutation_generation: 0,
            validation_dependencies: BTreeMap::new(),
        };
        let mut state = ValidationDriverState::new(check, identity).unwrap();
        let cancellation = ToolCancellationToken::new(Arc::new(AtomicBool::new(false)), None);
        let mut evidence = EvaluationEvidence::default();
        let result = state
            .drive(
                check,
                &ValidationDriveContext {
                    scope: &scope,
                    evidence: &evidence,
                    extensions: &bound.extensions,
                    executor: &executor,
                    journal: &Journal,
                    limits: Default::default(),
                    cancellation: &cancellation,
                    deadline: chrono::Utc::now() + chrono::Duration::seconds(10),
                },
            )
            .await
            .unwrap();
        let ValidationDriverOutcome::Complete(result) = result else {
            panic!("expected completion")
        };
        let fresh = steps[0] == "mutation";
        assert_eq!(
            result.outcome,
            if fresh {
                GateOutcome::Pass
            } else {
                GateOutcome::Unknown
            }
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        evidence.collect_validation(&state).unwrap();
        let current = executor.capture_scope(&scope).unwrap();
        assert_eq!(
            !bound
                .observe_progress(&current, &evidence)
                .unwrap()
                .signals
                .is_empty(),
            fresh
        );
        let json = serde_json::to_value(&state).unwrap();
        let mut retained = ValidationDriverState::restore_retained(
            json.clone(),
            &state,
            check,
            &scope,
            &bound.extensions,
        )
        .unwrap();
        let mut audit =
            ValidationDriverState::restore(json.clone(), check, &scope, &bound.extensions).unwrap();
        let context = ValidationDriveContext {
            scope: &scope,
            evidence: &evidence,
            extensions: &bound.extensions,
            executor: &executor,
            journal: &Journal,
            limits: Default::default(),
            cancellation: &cancellation,
            deadline: chrono::Utc::now() + chrono::Duration::seconds(10),
        };
        assert!(audit.drive(check, &context).await.is_err());
        retained.drive(check, &context).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(
            ValidationDriverState::restore_retained(
                json.clone(),
                &audit,
                check,
                &scope,
                &bound.extensions
            )
            .is_err()
        );
        let mut tampered = json.clone();
        tampered["check_deadline"] = json!(chrono::Utc::now() + chrono::Duration::seconds(60));
        assert!(
            ValidationDriverState::restore_retained(
                tampered,
                &state,
                check,
                &scope,
                &bound.extensions
            )
            .is_err()
        );
    }
}

// An effect can occur before a tool yields; dropping it cannot leave earlier independent proof eligible.
#[tokio::test]
async fn retained_sdk_dropped_mutation_invalidates_cached_read() {
    let (executor, bound, scope, calls, started) = fixture(&["read"]);
    let check = &bound.checks[0];
    let identity = EvidenceIdentity {
        key: scope.key.clone(),
        objective_revision: 0,
        cycle: 1,
        sequence: 1,
        stage: None,
        attempt: Some("attempt".into()),
        config_identity: check.config_identity.clone(),
        target: None,
        target_revision: None,
        mutation_generation: 0,
        validation_dependencies: BTreeMap::new(),
    };
    let mut state = ValidationDriverState::new(check, identity.clone()).unwrap();
    let evidence = EvaluationEvidence::default();
    let cancellation = ToolCancellationToken::new(Arc::new(AtomicBool::new(false)), None);
    let context = ValidationDriveContext {
        scope: &scope,
        evidence: &evidence,
        extensions: &bound.extensions,
        executor: &executor,
        journal: &Journal,
        limits: Default::default(),
        cancellation: &cancellation,
        deadline: chrono::Utc::now() + chrono::Duration::seconds(10),
    };
    state.drive(check, &context).await.unwrap();
    let request = ValidationObservationRequest::Tool {
        id: "pending".into(),
        tool: "pending_mutation".into(),
        arguments: json!({}),
        host_binding: None,
    };
    let mut work = Box::pin(executor.execute(&request, &identity));
    tokio::select! {
        _ = started.notified() => {},
        result = &mut work => panic!("must remain in flight: {result:?}"),
    }
    drop(work);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(executor.capture_scope(&scope).unwrap().mutation_generation > 0);
    assert!(state.drive(check, &context).await.is_err());
    assert!(!state.publication_acknowledged());
}

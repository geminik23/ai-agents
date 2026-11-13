use ai_agents_core::{
    Tool, ToolCallSource, ToolCancellationToken, ToolExecutionContext, ToolExecutionRequest,
    ToolInvoker, ToolResult, ToolSafetyMetadata,
};
use ai_agents_llm::mock::MockLLMProvider;
use ai_agents_runtime::{AgentBuilder, RuntimeAgent, autonomy::*};
use ai_agents_tools::{
    DiagnosticItem, DiagnosticSeverity, DiagnosticsProvider, DiagnosticsRequest,
    DiagnosticsResponse, StaticDiagnosticsProvider,
};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, atomic::AtomicU64};

struct Journal;
#[async_trait]
impl ValidationJournal for Journal {
    async fn checkpoint(&self, _: &ValidationDriverState) -> ai_agents_core::Result<()> {
        Ok(())
    }
}

fn item(path: &str, severity: DiagnosticSeverity) -> DiagnosticItem {
    DiagnosticItem {
        path: path.into(),
        severity,
        message: "fixture diagnostic".into(),
        line: None,
        column: None,
        source: None,
        code: None,
    }
}

// These hosts have no compiler, filesystem or provider I/O; the actual builtin and executor still run.
fn agent(max_results: usize, output_cap: usize) -> Arc<RuntimeAgent> {
    let yaml = format!(
        "name: DiagnosticsEvidence\nsystem_prompt: test\ntools: [diagnostics, echo]\ntool_security:\n  enabled: true\n  tools:\n    diagnostics:\n      allow_without_confirmation: true\n      max_results: {max_results}\n      max_output_chars: {output_cap}\n"
    );
    let mut model = MockLLMProvider::new("diagnostics-evidence");
    model.set_response("done");
    Arc::new(
        AgentBuilder::from_yaml(&yaml)
            .unwrap()
            .llm(Arc::new(model))
            .auto_configure_features()
            .unwrap()
            .autonomy_extensions(AutonomyExtensions::builtins())
            .build()
            .unwrap(),
    )
}

// Drive the actual SDK observation path, including journal ACK, before comparing both completion consumers.
async fn observe(
    agent: Arc<RuntimeAgent>,
    path: &str,
) -> (ValidationDriverState, EvaluationScope, EvaluationEvidence) {
    try_observe(agent, path).await.unwrap()
}

async fn try_observe(
    agent: Arc<RuntimeAgent>,
    path: &str,
) -> ai_agents_core::Result<(ValidationDriverState, EvaluationScope, EvaluationEvidence)> {
    let profile: AutonomyProfile = serde_json::from_value(json!({
        "validation":{"checks":[{"id":"check","adapter":"builtin.diagnostics","config":{"path":path}}]}
    }))
    .unwrap();
    let bound = agent
        .autonomy_extensions()
        .bind(
            &profile,
            &ValidationCapabilities {
                tools: BTreeSet::from(["diagnostics".into()]),
                ..Default::default()
            },
        )
        .unwrap();
    let check = &bound.checks[0];
    let mut scope = EvaluationScope {
        key: TaskRunKey {
            agent_id: agent.info().id,
            run_id: "diagnostics-fixture".into(),
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
    let executor = RuntimeObservationExecutor {
        agent,
        profile: None,
        scope_mode: "task".into(),
        run_revision: Arc::new(AtomicU64::new(1)),
        validator: "builtin.diagnostics".into(),
        contract_version: 1,
    };
    let cancellation = ToolCancellationToken::new(Default::default(), None);
    let evidence = EvaluationEvidence::default();
    let context = ValidationDriveContext {
        scope: &scope,
        evidence: &evidence,
        extensions: &bound.extensions,
        executor: &executor,
        journal: &Journal,
        limits: Default::default(),
        cancellation: &cancellation,
        deadline: None,
    };
    assert!(matches!(
        state.drive(check, &context).await?,
        ValidationDriverOutcome::Complete(_)
    ));
    assert!(state.publication_acknowledged());
    let mut evidence = EvaluationEvidence {
        required_checks: vec!["check".into()],
        ..Default::default()
    };
    evidence.collect_validation(&state).unwrap();
    Ok((state, scope, evidence))
}

fn assert_outcomes(scope: &EvaluationScope, evidence: &EvaluationEvidence, expected: GateOutcome) {
    let evaluator = CompletionGateEvaluator::default();
    for gate in [
        CompletionGate::ValidatorPassed("check".into()),
        CompletionGate::DiagnosticsClear(ai_agents_core::autonomy::DiagnosticsClearGate {
            severity: "error".into(),
        }),
    ] {
        assert_eq!(
            evaluator.evaluate(&gate, scope, evidence).unwrap().outcome,
            expected
        );
    }
}

#[tokio::test]
async fn static_provider_paths_and_positive_limits_reach_both_gates() {
    for path in [".", "src", "./src", "src/"] {
        for (count, expected) in [
            (0, GateOutcome::Pass),
            (1, GateOutcome::Pass),
            (2, GateOutcome::Pass),
            (3, GateOutcome::Unknown),
        ] {
            let agent = agent(2, 20_000);
            let mut items = vec![item("src/a.rs", DiagnosticSeverity::Warning); count];
            if count > 2 {
                items[2].severity = DiagnosticSeverity::Error;
            }
            // A sibling prefix is not inside src; root queries intentionally include it.
            if path != "." {
                items.push(item("src2/other.rs", DiagnosticSeverity::Error));
            }
            agent.set_diagnostics_provider(Arc::new(StaticDiagnosticsProvider::new(items)));
            let (state, scope, evidence) = observe(agent, path).await;
            assert_outcomes(&scope, &evidence, expected);
            let ObservationResult::Tool { record } = &state.completed["observe"].result else {
                panic!("expected actual diagnostic record")
            };
            assert!(record.executed);
            assert_eq!(record.canonical_id, "diagnostics");
            assert!(!record.output_truncated);
            assert_eq!(record.executed_arguments["max_results"], 2);
            let output: Value = serde_json::from_str(&record.output).unwrap();
            assert_eq!(output["count"], count.min(2));
            assert_eq!(output["message"].is_null(), count <= 2);
        }
    }
    let agent = agent(2, 20_000);
    agent.set_diagnostics_provider(Arc::new(StaticDiagnosticsProvider::new(vec![item(
        "src/a.rs",
        DiagnosticSeverity::Error,
    )])));
    let (_, scope, evidence) = observe(agent, ".").await;
    assert_outcomes(&scope, &evidence, GateOutcome::Fail);
}

struct PartialProvider;
#[async_trait]
impl DiagnosticsProvider for PartialProvider {
    async fn diagnostics(&self, _: DiagnosticsRequest) -> DiagnosticsResponse {
        DiagnosticsResponse {
            available: true,
            diagnostics: vec![],
            message: Some("collection is partial".into()),
        }
    }
}

#[tokio::test]
async fn partial_and_unavailable_diagnostics_cannot_prove_clean() {
    let agent = agent(200, 20_000);
    agent.set_diagnostics_provider(Arc::new(PartialProvider));
    let (_, scope, evidence) = observe(agent, ".").await;
    assert_outcomes(&scope, &evidence, GateOutcome::Unknown);
    let unavailable = self::agent(200, 20_000);
    let (state, scope, evidence) = observe(unavailable, ".").await;
    assert_outcomes(&scope, &evidence, GateOutcome::Unknown);
    let ObservationResult::Tool { record } = &state.completed["observe"].result else {
        panic!()
    };
    assert!(!record.executed);
}

// Only this explicitly installed host substitutes malformed data; it is still an actual executor invocation.
struct MalformedDiagnostics(&'static str);
#[async_trait]
impl Tool for MalformedDiagnostics {
    fn id(&self) -> &str {
        "diagnostics"
    }
    fn name(&self) -> &str {
        "diagnostics"
    }
    fn description(&self) -> &str {
        "Malformed observation fixture"
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
        if self.0 == "fixture failure" {
            ToolResult::error(self.0)
        } else {
            ToolResult::ok(self.0)
        }
    }
}

#[tokio::test]
async fn tiny_diagnostics_caps_and_original_malformed_data_are_distinct() {
    for cap in [0, 1, 16] {
        let agent = agent(200, cap);
        agent.set_diagnostics_provider(Arc::new(StaticDiagnosticsProvider::new(vec![])));
        let record = agent
            .invoke_tool(ToolExecutionRequest::new(
                "tiny",
                "diagnostics",
                json!({"path":".","severity":"all","max_results":200}),
                ToolCallSource::Manual,
            ))
            .await
            .unwrap();
        assert!(record.executed && record.output_truncated);
        assert!(record.output.chars().count() <= cap);
        let error = validation_error(agent, ".").await;
        assert!(
            error.contains("output limit") && error.contains("validity is unknown"),
            "{error}"
        );
    }
    for (output, expected) in [
        (
            r#"{"available":true,"truncated":false,"diagnostics":[]}"#,
            "",
        ),
        (
            r#"{"available":true,"truncated":false,"count":0,"diagnostics":[]}"#,
            "",
        ),
        ("{", "malformed diagnostics observation"),
        (
            r#"{"available":true,"truncated":false,"count":1,"diagnostics":[]}"#,
            "invalid diagnostic observation item",
        ),
        (
            r#"{"available":true,"truncated":false,"diagnostics":[{"severity":"invalid"}]}"#,
            "invalid diagnostic observation item",
        ),
    ] {
        let mut model = MockLLMProvider::new("malformed");
        model.set_response("done");
        let agent = Arc::new(
            AgentBuilder::new()
                .system_prompt("test")
                .llm(Arc::new(model))
                .tool(Arc::new(MalformedDiagnostics(output)))
                .autonomy_extensions(AutonomyExtensions::builtins())
                .build()
                .unwrap(),
        );
        agent.set_diagnostics_provider(Arc::new(StaticDiagnosticsProvider::new(vec![])));
        if expected.is_empty() {
            let (_, scope, evidence) = observe(agent, ".").await;
            assert_outcomes(&scope, &evidence, GateOutcome::Pass);
        } else {
            assert!(validation_error(agent, ".").await.contains(expected));
        }
    }
}

struct CompatibleFallback;
#[async_trait]
impl Tool for CompatibleFallback {
    fn id(&self) -> &str {
        "replacement"
    }
    fn name(&self) -> &str {
        "replacement"
    }
    fn description(&self) -> &str {
        "Compatible JSON is not diagnostics authority"
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
        ToolResult::ok(r#"{"available":true,"truncated":false,"diagnostics":[],"message":null}"#)
    }
}

#[tokio::test]
async fn compatible_fallback_is_audited_but_not_diagnostics_proof() {
    let mut model = MockLLMProvider::new("fallback");
    model.set_response("done");
    let mut registry = ai_agents_tools::ToolRegistry::new();
    registry
        .register(Arc::new(MalformedDiagnostics("fixture failure")))
        .unwrap();
    registry.register(Arc::new(CompatibleFallback)).unwrap();
    let agent = Arc::new(AgentBuilder::from_yaml(
        "name: Fallback\nsystem_prompt: test\ntools: [diagnostics, replacement]\nerror_recovery:\n  tools:\n    diagnostics:\n      max_retries: 0\n      on_failure: {action: fallback, fallback_tool: replacement}\n"
    ).unwrap().llm(Arc::new(model)).auto_configure_features().unwrap()
        .tools(registry).autonomy_extensions(AutonomyExtensions::builtins()).build().unwrap());
    agent.set_diagnostics_provider(Arc::new(StaticDiagnosticsProvider::new(vec![])));
    let (state, scope, evidence) = observe(agent, ".").await;
    assert_outcomes(&scope, &evidence, GateOutcome::Unknown);
    let ObservationResult::Tool { record } = &state.completed["observe"].result else {
        panic!()
    };
    assert!(record.executed && record.success, "{record:?}");
    assert_eq!(record.canonical_id, "replacement");
    assert!(matches!(record.source, ToolCallSource::Fallback { .. }));
}

// Optional checks expose direct-gate eligibility independently of the required-check safeguard.
// Retained decisions must execute once with the actual approved query and cannot transfer a fixed host permit.
#[tokio::test]
async fn retained_query_changes_cannot_complete_or_transfer_fixed_authority() {
    use ai_agents_hitl::ApprovalResult;
    for (changes, fixed, expected) in [
        (json!({}), false, GateOutcome::Pass),
        (json!({"severity":"ALL"}), false, GateOutcome::Pass),
        (json!({"path":"other"}), false, GateOutcome::Unknown),
        (json!({"severity":"warning"}), false, GateOutcome::Unknown),
        (json!({"max_results":1}), false, GateOutcome::Pass),
        (json!({"path":"other"}), true, GateOutcome::Unknown),
        (json!({"severity":"ALL"}), true, GateOutcome::Unknown),
        (json!({}), true, GateOutcome::Pass),
    ] {
        let yaml = "name: RetainedDiagnostics\nsystem_prompt: test\ntools: [diagnostics]\nhitl:\n  tools:\n    diagnostics: {require_approval: true}\ntool_security:\n  enabled: true\n  tools:\n    diagnostics:\n      allow_without_confirmation: true\n";
        let mut extensions = AutonomyExtensions::builtins();
        if fixed {
            extensions
                .register_host_validation(HostValidationBinding {
                    id: "fixed".into(),
                    validator: "builtin.diagnostics".into(),
                    version: 1,
                    profile: None,
                    scope: "task".into(),
                    tool: "diagnostics".into(),
                    arguments: json!({"path":".","severity":"all","max_results":200}),
                    require_approval: false,
                })
                .unwrap();
        }
        let mut model = MockLLMProvider::new("retained");
        model.set_response("done");
        let agent = Arc::new(
            AgentBuilder::from_yaml(yaml)
                .unwrap()
                .llm(Arc::new(model))
                .auto_configure_features()
                .unwrap()
                .autonomy_extensions(extensions)
                .build()
                .unwrap(),
        );
        agent.set_diagnostics_provider(Arc::new(StaticDiagnosticsProvider::new(vec![item(
            "src/a.rs",
            if changes.get("path").is_some() || changes.get("severity") == Some(&json!("warning")) {
                DiagnosticSeverity::Error
            } else {
                DiagnosticSeverity::Warning
            },
        )])));
        let store = Arc::new(
            ScopedTaskRunStore::in_memory(agent.info().id, None, "diagnostics.v1".into()).unwrap(),
        );
        let mut config = json!({"path":"."});
        if fixed {
            config["host_binding"] = json!("fixed");
        }
        let profile: AutonomyProfile = serde_json::from_value(json!({
            "enabled":true,"max_turns":4,"max_llm_calls":4,"max_tool_calls":1,
            "hitl":{"on_approval_required":"pause_run"},
            "validation":{"checks":[{"id":"check","adapter":"builtin.diagnostics","required":false,"schedule":"each_cycle","config":config}]},
            "completion":{"diagnostics_clear":{"severity":"error"}}
        })).unwrap();
        let runner = AutonomyRunner::try_new(
            agent.clone(),
            AutonomyConfig {
                defaults: profile,
                ..Default::default()
            },
            Default::default(),
            store,
            "diagnostics.v1".into(),
        )
        .unwrap();
        let paused = runner.run("objective", None).await.unwrap();
        assert_eq!(
            paused.run.status,
            TaskRunStatus::Paused,
            "stop={:?}",
            paused.run.stop_reason
        );
        assert_eq!(paused.run.counters.tool_attempts, 0);
        let request_id = paused.run.pending.as_ref().unwrap().id.clone();
        let result = runner
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::Approval {
                    request_id: request_id.clone(),
                    result: if changes.as_object().unwrap().is_empty() {
                        ApprovalResult::Approved
                    } else {
                        ApprovalResult::Modified {
                            changes: serde_json::from_value(changes.clone()).unwrap(),
                        }
                    },
                },
            )
            .await
            .unwrap();
        assert_eq!(
            result.run.status == TaskRunStatus::Completed,
            expected == GateOutcome::Pass,
            "fixed={fixed} changes={changes}: {:?}",
            result.run.status
        );
        let records: Vec<_> = result
            .run
            .evidence
            .tool_calls
            .iter()
            .filter(|r| r.record.canonical_id == "diagnostics")
            .collect();
        let invoked = !fixed || changes.as_object().unwrap().is_empty();
        assert_eq!(
            records.iter().filter(|r| r.record.executed).count(),
            usize::from(invoked)
        );
        assert_eq!(result.run.counters.tool_attempts, u64::from(invoked));
        if invoked {
            assert_eq!(records.len(), 1);
            let record = &records[0].record;
            assert_eq!(
                record.arguments,
                json!({"path":".","severity":"all","max_results":200})
            );
            for (key, value) in changes.as_object().unwrap() {
                assert_eq!(record.executed_arguments[key], *value);
            }
            let validation = result
                .run
                .evidence
                .records
                .iter()
                .find_map(|entry| {
                    entry
                        .data
                        .get("validations")?
                        .as_array()?
                        .iter()
                        .find(|v| v["check_id"] == "check")
                })
                .expect("acknowledged validator result");
            assert_eq!(
                validation["outcome"],
                serde_json::to_value(expected).unwrap()
            );
        }
        assert!(
            runner
                .resume(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    TaskResumeInput::Approval {
                        request_id,
                        result: ApprovalResult::Approved,
                    }
                )
                .await
                .is_err()
        );
    }
}

#[tokio::test]
async fn diagnostic_publication_ack_and_freshness_remain_required() {
    let agent = agent(200, 20_000);
    agent.set_diagnostics_provider(Arc::new(StaticDiagnosticsProvider::new(vec![])));
    let (state, mut scope, evidence) = observe(agent, ".").await;
    assert_outcomes(&scope, &evidence, GateOutcome::Pass);
    let restored: ValidationDriverState =
        serde_json::from_value(serde_json::to_value(&state).unwrap()).unwrap();
    assert!(!restored.publication_acknowledged());
    assert!(
        EvaluationEvidence::default()
            .collect_validation(&restored)
            .is_err()
    );
    scope.mutation_generation += 1;
    assert_outcomes(&scope, &evidence, GateOutcome::Unknown);
    scope.mutation_generation -= 1;
    scope
        .validation_attempts
        .insert("check".into(), "replacement".into());
    assert_outcomes(&scope, &evidence, GateOutcome::Unknown);
}

async fn validation_error(agent: Arc<RuntimeAgent>, path: &str) -> String {
    try_observe(agent, path)
        .await
        .expect_err("invalid diagnostics must not pass")
        .to_string()
}

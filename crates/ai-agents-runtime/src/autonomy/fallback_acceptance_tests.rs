use super::continuation_test_support::{self as support, ReceiptStore};
use super::*;
use crate::spawner::{AgentRegistry, AgentSpawner, SpawnedAgent};
use crate::{AgentBuilder, RuntimeAgent};
use ai_agents_core::{
    Tool, ToolCall, ToolCallClassification, ToolExecutionContext, ToolExecutionRecord, ToolResult,
    ToolSafetyMetadata,
};
use ai_agents_hitl::ApprovalResult;
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Default)]
struct Trace(parking_lot::Mutex<Vec<String>>);
#[async_trait]
impl ai_agents_hooks::AgentHooks for Trace {
    async fn on_tool_start(&self, id: &str, _: &Value) {
        self.0.lock().push(format!("start:{id}"));
    }
    async fn on_tool_complete(&self, id: &str, _: &ToolResult, _: u64) {
        self.0.lock().push(format!("complete:{id}"));
    }
    async fn on_tool_execution_record(&self, record: &ToolExecutionRecord) {
        self.0
            .lock()
            .push(format!("record:{}", record.canonical_id));
    }
}
struct Rescue(Arc<AtomicUsize>);
#[async_trait]
impl Tool for Rescue {
    fn id(&self) -> &str {
        "rescue"
    }
    fn name(&self) -> &str {
        "rescue"
    }
    fn description(&self) -> &str {
        "Safe terminal fallback"
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
    async fn execute(&self, args: Value, _: ToolExecutionContext) -> ToolResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        ToolResult::ok(args.to_string())
    }
}
struct MessageProbe {
    registry: Arc<AgentRegistry>,
    calls: Arc<AtomicUsize>,
    fail_first: bool,
    timeout_ms: Option<u64>,
}
#[async_trait]
impl Tool for MessageProbe {
    fn id(&self) -> &str {
        "message_probe"
    }
    fn name(&self) -> &str {
        "message_probe"
    }
    fn description(&self) -> &str {
        "Conforming read-only message fixture with explicit retry safety"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","required":["to","message"]})
    }
    fn supports_task_messages(&self) -> bool {
        true
    }
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: true,
            concurrency_safe: true,
            ..Default::default()
        }
    }
    fn classify_call(&self, _: &Value) -> ToolCallClassification {
        let mut classification = ToolCallClassification::from_metadata(&self.safety_metadata());
        classification.safely_retryable = true;
        classification.timeout_ms = self.timeout_ms;
        classification
    }
    async fn execute_task(
        &self,
        args: Value,
        _: ToolExecutionContext,
    ) -> ai_agents_core::Result<ToolResult> {
        let index = self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_first && index == 0 {
            return Ok(ToolResult::error("failed before delivery"));
        }
        let work = self.registry.send_task_message(
            "parent",
            args["to"].as_str().unwrap(),
            args["message"].as_str().unwrap(),
            Default::default(),
        );
        match scope_child_requirement(false, work).await {
            Ok(response) => {
                let mut result = ToolResult::ok(
                    json!({"from":args["to"],"response":response.content}).to_string(),
                );
                // An implementation result cannot overwrite framework-owned logical hop identity.
                result.metadata = Some(std::collections::HashMap::from([(
                    "_task_executor_hop".into(),
                    json!(999),
                )]));
                Ok(result)
            }
            Err(error @ ai_agents_core::AgentError::TaskSuspended(_)) => Err(error),
            Err(error) => Ok(ToolResult::error(format!("send failed: {error}"))),
        }
    }
    async fn execute(&self, _: Value, _: ToolExecutionContext) -> ToolResult {
        panic!("use task-aware dispatcher")
    }
}
struct Fixture {
    parent: Arc<RuntimeAgent>,
    runner: AutonomyRunner,
    store: Arc<ScopedTaskRunStore>,
    registry: Arc<AgentRegistry>,
    calls: Arc<AtomicUsize>,
    rescue: Arc<AtomicUsize>,
    child: Arc<support::FaultProvider>,
    trace: Arc<Trace>,
}
fn runner(
    parent: Arc<RuntimeAgent>,
    store: Arc<dyn TaskRunStore>,
    fallback: bool,
) -> AutonomyRunner {
    let mut config = super::runner_tests::config(16);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    if fallback {
        config.defaults.completion = Some(CompletionGate::ToolCalled(
            ai_agents_core::autonomy::ToolCalledGate {
                id: "rescue".into(),
                count: Some(1),
                count_gte: None,
                executed: Some(true),
                success: Some(true),
            },
        ));
    }
    AutonomyRunner::try_new(
        parent,
        config,
        Default::default(),
        store,
        "fallback.v1".into(),
    )
    .unwrap()
}
// An ordinary acknowledged child error is optional; required/uncertain child failure remains protected by existing policy.
async fn fixture(fallback: bool, approval: bool, timeout: Option<u64>, sqlite: bool) -> Fixture {
    Box::pin(fixture_with_failure(
        fallback, approval, timeout, sqlite, false,
    ))
    .await
}

// Persistent returned child errors make a reset retry index visible as extra real invocations.
async fn fixture_with_failure(
    fallback: bool,
    approval: bool,
    timeout: Option<u64>,
    sqlite: bool,
    persistent: bool,
) -> Fixture {
    Box::pin(fixture_mode(
        fallback,
        approval,
        timeout,
        sqlite,
        if persistent { "errors_after" } else { "error" },
    ))
    .await
}

// Separate provider phases expose both a retained attempt and a later retry that independently suspends.
async fn fixture_mode(
    fallback: bool,
    approval: bool,
    timeout: Option<u64>,
    sqlite: bool,
    mode: &'static str,
) -> Fixture {
    let registry = Arc::new(AgentRegistry::new());
    let question = ai_agents_core::encode_native_tool_call_markers(
        &[ToolCall {
            id: "child-question".into(),
            name: "ask_user".into(),
            arguments: json!({"question":"Choice","options":["yes"],"allow_other":false}),
        }],
        None,
    )
    .unwrap();
    let next_question = ai_agents_core::encode_native_tool_call_markers(
        &[ToolCall {
            id: "child-question-next".into(),
            name: "ask_user".into(),
            arguments: json!({"question":"Again","options":["yes"],"allow_other":false}),
        }],
        None,
    )
    .unwrap();
    let child = support::FaultProvider::new(
        &[
            &question,
            "failed",
            if mode == "alternate_error" {
                &next_question
            } else {
                "child done"
            },
            "failed",
            "child done",
        ],
        1,
        mode,
    );
    let child_yaml = "name: Worker\nsystem_prompt: test\ntools: [ask_user]\n";
    let worker = AgentBuilder::from_yaml(child_yaml)
        .unwrap()
        .llm(child.clone())
        .auto_configure_features()
        .unwrap()
        .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
            Arc::new(parking_lot::RwLock::new(None)),
        )))
        .build()
        .unwrap();
    registry
        .register(SpawnedAgent::from_runtime(
            "worker".into(),
            worker,
            crate::spec::AgentSpec::from_yaml_strict(child_yaml).unwrap(),
        ))
        .await
        .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let rescue = Arc::new(AtomicUsize::new(0));
    let trace = Arc::new(Trace::default());
    let marker = ai_agents_core::encode_native_tool_call_markers(
        &[ToolCall {
            id: "original".into(),
            name: "message_probe".into(),
            arguments: json!({"to":"worker","message":"original"}),
        }],
        None,
    )
    .unwrap();
    let yaml = format!(
        "name: Recovery\nsystem_prompt: test\ntools: [message_probe, rescue]\n{}error_recovery:\n  tools:\n    message_probe:\n      max_retries: {}\n      on_failure:\n        action: {}\n{}",
        if approval {
            "hitl:\n  tools:\n    message_probe:\n      require_approval: true\n    rescue:\n      require_approval: true\n"
        } else {
            ""
        },
        if fallback { 0 } else { 2 },
        if fallback { "fallback" } else { "report_error" },
        if fallback {
            "        fallback_tool: rescue\n"
        } else {
            ""
        }
    );
    let parent = Arc::new(
        AgentBuilder::from_yaml(&yaml)
            .unwrap()
            .llm(super::runner_tests::RecordingProvider::new(&[
                &marker, "done",
            ]))
            .auto_configure_features()
            .unwrap()
            .hooks(trace.clone())
            .tool(Arc::new(MessageProbe {
                registry: registry.clone(),
                calls: calls.clone(),
                fail_first: !fallback,
                timeout_ms: timeout,
            }))
            .tool(Arc::new(Rescue(rescue.clone())))
            .build()
            .unwrap()
            .with_spawner_handles(Arc::new(AgentSpawner::new()), registry.clone()),
    );
    let store = support::store(parent.info().id, "fallback.v1", sqlite).await;
    let runner = runner(parent.clone(), store.clone(), fallback);
    Fixture {
        parent,
        runner,
        store,
        registry,
        calls,
        rescue,
        child,
        trace,
    }
}
async fn answer(f: &Fixture, run: &TaskRun) -> ai_agents_core::Result<TaskRunResult> {
    f.runner
        .resume(
            &run.key.run_id,
            run.revision,
            TaskResumeInput::UserAnswer {
                request_id: run.pending.as_ref().unwrap().id.clone(),
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await
}
async fn approve(
    f: &Fixture,
    run: &TaskRun,
    modified: bool,
) -> ai_agents_core::Result<TaskRunResult> {
    f.runner
        .resume(
            &run.key.run_id,
            run.revision,
            TaskResumeInput::Approval {
                request_id: run.pending.as_ref().unwrap().id.clone(),
                result: if modified {
                    ApprovalResult::Modified {
                        changes: std::collections::HashMap::from([(
                            "message".into(),
                            json!("modified"),
                        )]),
                    }
                } else {
                    ApprovalResult::Approved
                },
            },
        )
        .await
}

#[tokio::test]
async fn fallback_acceptance_modified_target_and_signed_repause_keep_actual_delivery_binding() {
    let f = Box::pin(fixture(false, true, None, false)).await;
    let questions: Vec<_> = (0..2)
        .map(|index| {
            let call = ToolCall {
                id: format!("new-question-{index}"),
                name: "ask_user".into(),
                arguments: json!({"question":"Choice","options":["yes"],"allow_other":false}),
            };
            let state = ai_agents_core::NativeProviderState::new(
                format!("new-signed-{index}"),
                "fixture",
                "native-tools",
                ai_agents_core::NativeProviderTarget::new(
                    "https://fixture.invalid",
                    "fixture-model",
                )
                .unwrap(),
                json!({"signature":index}),
                vec![ai_agents_core::NativeCallBinding::new(&call.id, 0).unwrap()],
            )
            .unwrap();
            ai_agents_core::encode_native_tool_call_markers(&[call], Some(&state)).unwrap()
        })
        .collect();
    let provider =
        super::runner_tests::RecordingProvider::new(&[&questions[0], &questions[1], "child done"]);
    let yaml = "name: NewWorker\nsystem_prompt: test\ntools: [ask_user]\n";
    let child = AgentBuilder::from_yaml(yaml)
        .unwrap()
        .llm(provider.clone())
        .auto_configure_features()
        .unwrap()
        .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
            Arc::new(parking_lot::RwLock::new(None)),
        )))
        .build()
        .unwrap();
    f.registry
        .register(SpawnedAgent::from_runtime(
            "new".into(),
            child,
            crate::spec::AgentSpec::from_yaml_strict(yaml).unwrap(),
        ))
        .await
        .unwrap();
    let first = f.runner.run("objective", None).await.unwrap();
    let paused = f
        .runner
        .resume(
            &first.run.key.run_id,
            first.run.revision,
            TaskResumeInput::Approval {
                request_id: first.run.pending.as_ref().unwrap().id.clone(),
                result: ApprovalResult::Modified {
                    changes: std::collections::HashMap::from([
                        ("to".into(), json!("new")),
                        ("message".into(), json!("modified")),
                    ]),
                },
            },
        )
        .await
        .unwrap();
    let next = answer(&f, &paused.run).await.unwrap();
    assert_eq!(next.run.status, TaskRunStatus::Paused);
    assert!(answer(&f, &paused.run).await.is_err());
    let done = answer(&f, &next.run).await.unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(f.child.calls.load(Ordering::SeqCst), 0);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
    let record = &done
        .run
        .evidence
        .tool_calls
        .iter()
        .find(|r| r.record.canonical_id == "message_probe")
        .unwrap()
        .record;
    assert_eq!(record.executed_arguments["to"], "new");
    assert_eq!(record.executed_arguments["message"], "modified");
    let payload: TaskCheckpointPayload = serde_json::from_value(
        f.store
            .load(&done.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert_eq!(payload.children.len(), 1);
    let history = ai_agents_core::inspect_native_history(
        &payload.children[0].runtime.snapshot.memory.messages,
    )
    .unwrap();
    assert_eq!(history.exchanges().len(), 2);
    for (index, exchange) in history.exchanges().iter().enumerate() {
        assert!(exchange.is_complete());
        assert_eq!(
            exchange.state().exchange_id(),
            format!("new-signed-{index}")
        );
        assert_eq!(
            exchange.state().model_content(),
            &json!({"signature":index})
        );
    }
    assert_eq!(
        f.trace
            .0
            .lock()
            .iter()
            .filter(|event| *event == "start:message_probe")
            .count(),
        1
    );
}

#[tokio::test]
async fn fallback_acceptance_settled_message_does_not_consume_a_second_rate_slot() {
    let f = Box::pin(fixture(true, false, None, false)).await;
    f.parent
        .runtime_control()
        .set_tool_security(ai_agents_tools::ToolSecurityConfig {
            enabled: true,
            tools: std::collections::HashMap::from([(
                "message_probe".into(),
                ai_agents_tools::ToolPolicyConfig {
                    rate_limit: Some(1),
                    ..Default::default()
                },
            )]),
            ..Default::default()
        });
    let first = f.runner.run("objective", None).await.unwrap();
    let done = answer(&f, &first.run).await.unwrap();
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.rescue.load(Ordering::SeqCst), 1);
    assert!(
        done.run
            .evidence
            .tool_calls
            .iter()
            .find(|r| r.record.canonical_id == "message_probe")
            .unwrap()
            .record
            .executed
    );
}

#[tokio::test]
async fn fallback_acceptance_changed_authority_or_emergency_stop_after_settlement_never_rebinds_review()
 {
    for cancel in [false, true] {
        let mut f = Box::pin(fixture(true, true, None, false)).await;
        let fault = ReceiptStore::checkpoint(
            f.store.clone(),
            "hold",
            json!({"tool":"message_probe","phase":"settle"}),
        );
        f.runner = runner(f.parent.clone(), fault.clone(), true);
        let first = f.runner.run("objective", None).await.unwrap();
        let paused = approve(&f, &first.run, true).await.unwrap();
        let version = f.parent.runtime_control().version();
        fault.armed.store(true, Ordering::SeqCst);
        let mut resume = Box::pin(answer(&f, &paused.run));
        tokio::select! {result=&mut resume=>panic!("settlement not held: {result:?}"),permit=fault.reached.acquire()=>permit.unwrap().forget(),_=tokio::time::sleep(std::time::Duration::from_secs(5))=>panic!("settlement not reached") }
        if cancel {
            f.parent.runtime_control().cancel_all();
        } else {
            f.parent
                .runtime_control()
                .set_tool_scope(vec!["message_probe".into(), "rescue".into()]);
        }
        fault.release.add_permits(1);
        let outcome = resume.await;
        assert!(
            !outcome
                .as_ref()
                .is_ok_and(|result| result.run.status == TaskRunStatus::Completed)
        );
        fault.assert_hit();
        assert_eq!(f.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.rescue.load(Ordering::SeqCst), 0);
        let p: TaskCheckpointPayload = serde_json::from_value(
            f.store
                .load(&paused.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .payload,
        )
        .unwrap();
        let record = &p
            .evidence
            .tool_calls
            .iter()
            .find(|r| r.record.canonical_id == "message_probe")
            .unwrap()
            .record;
        assert!(record.executed && !record.success);
        assert_eq!(record.runtime_config_version, version);
        if outcome.as_ref().is_ok_and(|result| {
            result.run.status.is_terminal() && result.run.status != TaskRunStatus::RecoveryRequired
        }) {
            assert!(f.parent.set_context("released", json!(true)).is_ok());
        } else {
            assert!(f.parent.set_context("blocked", json!(true)).is_err());
        }
    }
}

struct HostedValidator(AdapterDescriptor);
impl AutonomyValidator for HostedValidator {
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.0
    }
    fn validate_config(&self, _: &Value) -> ai_agents_core::Result<()> {
        Ok(())
    }
    fn evaluate(&self, input: &ValidationInput<'_>) -> ai_agents_core::Result<ValidationDecision> {
        if input.observations.is_empty() {
            Ok(ValidationDecision::NeedObservations {
                requests: vec![ValidationObservationRequest::Tool {
                    id: "message".into(),
                    tool: "message_probe".into(),
                    arguments: json!({"to":"worker","message":"objective"}),
                    host_binding: Some("fixed-recovery".into()),
                }],
                checkpoint: json!({}),
            })
        } else {
            let pass=input.observations.values().all(|observation|matches!(&observation.result,ObservationResult::Tool {record} if record.executed && record.success));
            Ok(ValidationDecision::Complete {
                outcome: if pass {
                    GateOutcome::Pass
                } else {
                    GateOutcome::Fail
                },
                reason: "settled observation".into(),
                metrics: json!({}),
                evidence_refs: vec![],
            })
        }
    }
}

#[tokio::test]
async fn fallback_acceptance_fixed_host_operation_recovers_without_permit_transfer() {
    for action in ["retry", "skip", "fallback"] {
        let registry = Arc::new(AgentRegistry::new());
        let question = ai_agents_core::encode_native_tool_call_markers(
            &[ToolCall {
                id: "question".into(),
                name: "ask_user".into(),
                arguments: json!({"question":"Choice","options":["yes"],"allow_other":false}),
            }],
            None,
        )
        .unwrap();
        let child = support::FaultProvider::new(&[&question, "failed", "child done"], 1, "error");
        let yaml = "name: HostChild\nsystem_prompt: test\ntools: [ask_user]\n";
        let worker = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(child.clone())
            .auto_configure_features()
            .unwrap()
            .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
                Arc::new(parking_lot::RwLock::new(None)),
            )))
            .build()
            .unwrap();
        registry
            .register(SpawnedAgent::from_runtime(
                "worker".into(),
                worker,
                crate::spec::AgentSpec::from_yaml_strict(yaml).unwrap(),
            ))
            .await
            .unwrap();
        let mut extensions = AutonomyExtensions::builtins();
        extensions
            .register_validator(Arc::new(HostedValidator(AdapterDescriptor {
                id: "host.recovery".into(),
                contract_version: 1,
                tools: vec!["message_probe".into()],
                ..Default::default()
            })))
            .unwrap();
        extensions
            .register_host_validation(HostValidationBinding {
                id: "fixed-recovery".into(),
                validator: "host.recovery".into(),
                version: 1,
                profile: None,
                scope: "task".into(),
                tool: "message_probe".into(),
                arguments: json!({"to":"worker","message":"objective"}),
                require_approval: false,
            })
            .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let rescue = Arc::new(AtomicUsize::new(0));
        let yaml = format!(
            "name: HostRoot\nsystem_prompt: test\ntools: [rescue]\nerror_recovery:\n  tools:\n    message_probe:\n      max_retries: {}\n      on_failure: {{action: {}{}}}\n",
            usize::from(action == "retry"),
            if action == "retry" {
                "report_error"
            } else {
                action
            },
            if action == "fallback" {
                ", fallback_tool: rescue"
            } else {
                ""
            }
        );
        let parent = Arc::new(
            AgentBuilder::from_yaml(&yaml)
                .unwrap()
                .llm(super::runner_tests::RecordingProvider::new(&["done"]))
                .auto_configure_features()
                .unwrap()
                .autonomy_extensions(extensions)
                .tool(Arc::new(MessageProbe {
                    registry: registry.clone(),
                    calls: calls.clone(),
                    fail_first: false,
                    timeout_ms: None,
                }))
                .tool(Arc::new(Rescue(rescue.clone())))
                .build()
                .unwrap()
                .with_spawner_handles(Arc::new(AgentSpawner::new()), registry),
        );
        let store = support::store(parent.info().id, "host-recovery.v1", false).await;
        let mut config = super::runner_tests::config(8);
        config.defaults.max_turns = Some(1);
        config.defaults.validation=Some(serde_yaml::from_str("checks: [{id: check, adapter: host.recovery, schedule: each_cycle, config: {host_binding: fixed-recovery}}]").unwrap());
        config.defaults.completion = Some(CompletionGate::ValidatorPassed("check".into()));
        config.defaults.hitl = Some(AutonomyHitlConfig {
            on_user_question: Some(InteractionAction::PauseRun),
            on_approval_required: Some(InteractionAction::PauseRun),
        });
        let runner = AutonomyRunner::try_new(
            parent.clone(),
            config,
            Default::default(),
            store,
            "host-recovery.v1".into(),
        )
        .unwrap();
        let first = runner.run("objective", None).await.unwrap();
        let done = runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: first.run.pending.as_ref().unwrap().id.clone(),
                    answer: json!({"answered":true,"selected":["yes"]}),
                },
            )
            .await
            .unwrap();
        assert_eq!(
            rescue.load(Ordering::SeqCst),
            0,
            "fixed authority must not transfer even to an ordinarily granted fallback"
        );
        if action == "fallback" {
            assert_ne!(done.run.status, TaskRunStatus::Completed);
        } else {
            assert_eq!(
                done.run.status,
                TaskRunStatus::Completed,
                "action={action}, stop={:?}",
                done.run.stop_reason
            );
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            if action == "retry" { 2 } else { 1 }
        );
    }
}

#[tokio::test]
async fn fallback_acceptance_same_hop_evidence_is_idempotent_but_conflicts_are_rejected() {
    let f = Box::pin(fixture(true, false, None, false)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    let done = answer(&f, &first.run).await.unwrap();
    let tools: Vec<_> = done
        .run
        .evidence
        .tool_calls
        .iter()
        .map(|capture| ScopedObservation {
            identity: serde_json::from_value::<EvidenceIdentity>(
                capture.record.metadata["_task_identity"].clone(),
            )
            .unwrap(),
            complete: true,
            value: capture.record.clone(),
        })
        .collect();
    let mut evidence = EvaluationEvidence {
        tools,
        ..Default::default()
    };
    let count = evidence.tools.len();
    assert!(count >= 3);
    evidence.tools.push(evidence.tools.last().unwrap().clone());
    assert_eq!(evidence.normalized().unwrap().tools.len(), count);
    let duplicate = evidence.tools.last_mut().unwrap();
    duplicate.value.success = !duplicate.value.success;
    assert!(evidence.normalized().is_err());
}

#[tokio::test]
async fn fallback_acceptance_cancel_or_stop_keeps_failed_ancestor_and_never_invokes_parked_fallback()
 {
    for stop in [false, true] {
        let f = Box::pin(fixture(true, true, None, false)).await;
        let first = f.runner.run("objective", None).await.unwrap();
        let child = approve(&f, &first.run, true).await.unwrap();
        let fallback = answer(&f, &child.run).await.unwrap();
        let done = if stop {
            f.runner
                .stop_paused(
                    &fallback.run.key.run_id,
                    fallback.run.revision,
                    "stop recovery",
                )
                .await
                .unwrap()
        } else {
            f.runner
                .cancel_paused(&fallback.run.key.run_id, fallback.run.revision)
                .await
                .unwrap()
        };
        assert_eq!(
            done.run.status,
            if stop {
                TaskRunStatus::Incomplete
            } else {
                TaskRunStatus::Cancelled
            }
        );
        assert_eq!(f.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.rescue.load(Ordering::SeqCst), 0);
        let ancestor = &done
            .run
            .evidence
            .tool_calls
            .iter()
            .find(|r| r.record.canonical_id == "message_probe")
            .unwrap()
            .record;
        assert!(ancestor.executed && !ancestor.success);
        assert!(approve(&f, &fallback.run, false).await.is_err());
        assert!(f.registry.remove("worker").await.is_some());
    }
}

struct TamperStore {
    inner: Arc<ScopedTaskRunStore>,
    field: &'static str,
    armed: std::sync::atomic::AtomicBool,
    claims: AtomicUsize,
}
// Only a delivered copy changes; the stored revision, live frame and execution authority remain untouched.
fn tamper_cursor(value: &mut Value, field: &str) -> bool {
    if let Some(object) = value.as_object_mut() {
        if object.contains_key("ancestry")
            && object.contains_key("retry_index")
            && object.contains_key("request")
        {
            match field {
                "retry_index" => {
                    object.insert(field.into(), json!(99));
                }
                "started_at" => {
                    object.insert(field.into(), json!("2020-01-01T00:00:00Z"));
                }
                "invoked" => {
                    let previous = object[field].as_bool().unwrap();
                    object.insert(field.into(), json!(!previous));
                }
                "arguments" => {
                    object.get_mut("request").unwrap()["arguments"] =
                        json!({"to":"worker","message":"tampered"});
                }
                _ => {
                    object.insert("ancestry".into(), json!(["tampered"]));
                }
            }
            return true;
        }
        for child in object.values_mut() {
            if tamper_cursor(child, field) {
                return true;
            }
        }
    } else if let Some(array) = value.as_array_mut() {
        for child in array {
            if tamper_cursor(child, field) {
                return true;
            }
        }
    }
    false
}
#[async_trait]
impl TaskRunStore for TamperStore {
    async fn create(&self, s: &TaskRunSnapshot) -> ai_agents_core::Result<()> {
        self.inner.create(s).await
    }
    async fn load(&self, id: &str) -> ai_agents_core::Result<Option<TaskRunSnapshot>> {
        let mut saved = self.inner.load(id).await?;
        if self.armed.load(Ordering::SeqCst) {
            assert!(tamper_cursor(
                &mut saved.as_mut().unwrap().payload,
                self.field
            ));
        }
        Ok(saved)
    }
    async fn list(&self) -> ai_agents_core::Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    async fn delete(&self, id: &str, r: u64) -> ai_agents_core::Result<()> {
        self.inner.delete(id, r).await
    }
    async fn mutate(
        &self,
        id: &str,
        m: &TaskRunMutation,
    ) -> ai_agents_core::Result<TaskRunSnapshot> {
        if matches!(m, TaskRunMutation::Claim { .. }) {
            self.claims.fetch_add(1, Ordering::SeqCst);
        }
        self.inner.mutate(id, m).await
    }
}
#[tokio::test]
async fn fallback_acceptance_private_recovery_cursor_changes_reject_before_claim() {
    for field in [
        "retry_index",
        "started_at",
        "invoked",
        "arguments",
        "ancestry",
    ] {
        let mut f = Box::pin(fixture(false, false, None, false)).await;
        let tamper = Arc::new(TamperStore {
            inner: f.store.clone(),
            field,
            armed: std::sync::atomic::AtomicBool::new(false),
            claims: AtomicUsize::new(0),
        });
        f.runner = runner(f.parent.clone(), tamper.clone(), false);
        let first = f.runner.run("objective", None).await.unwrap();
        let claims = tamper.claims.load(Ordering::SeqCst);
        tamper.armed.store(true, Ordering::SeqCst);
        assert!(answer(&f, &first.run).await.is_err());
        assert_eq!(tamper.claims.load(Ordering::SeqCst), claims);
        assert_eq!(
            f.store
                .load(&first.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .revision,
            first.run.revision
        );
        assert_eq!(f.calls.load(Ordering::SeqCst), 2);
        tamper.armed.store(false, Ordering::SeqCst);
        assert_eq!(
            answer(&f, &first.run).await.unwrap().run.status,
            TaskRunStatus::Completed
        );
    }
}

#[tokio::test]
async fn fallback_acceptance_later_retry_repause_keeps_index_and_original_lifecycle() {
    for sqlite in support::backends() {
        let f = Box::pin(fixture_mode(false, false, None, *sqlite, "alternate_error")).await;
        let first = f.runner.run("objective", None).await.unwrap();
        let saved = f.store.load(&first.run.key.run_id).await.unwrap().unwrap();
        let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload).unwrap();
        let cursor = payload
            .adapters
            .iter()
            .find_map(|adapter| {
                let value = &adapter.state["cursor"];
                let attempt = value["attempt"].as_str()?;
                (value["record"]["canonical_id"] == "message_probe"
                    && payload
                        .reservations
                        .iter()
                        .any(|r| r.id == attempt && r.state == TaskEffectState::Suspended))
                .then(|| value["executor"].clone())
            })
            .unwrap();
        assert_eq!(cursor["retry_index"], 1);
        let second = answer(&f, &first.run).await.unwrap();
        assert_eq!(second.run.status, TaskRunStatus::Paused);
        let payload: TaskCheckpointPayload = serde_json::from_value(
            f.store
                .load(&second.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .payload,
        )
        .unwrap();
        let current = payload
            .adapters
            .iter()
            .find_map(|adapter| {
                let value = &adapter.state["cursor"];
                let attempt = value["attempt"].as_str()?;
                (value["record"]["canonical_id"] == "message_probe"
                    && payload
                        .reservations
                        .iter()
                        .any(|r| r.id == attempt && r.state == TaskEffectState::Suspended))
                .then(|| value["executor"].clone())
            })
            .unwrap();
        assert_eq!(current["retry_index"], 2);
        assert_eq!(current["started_at"], cursor["started_at"]);
        let done = answer(&f, &second.run).await.unwrap();
        assert_eq!(f.calls.load(Ordering::SeqCst), 3);
        assert_eq!(f.child.calls.load(Ordering::SeqCst), 4);
        assert_eq!(done.run.counters.tool_attempts, 5);
        assert!(
            !done
                .run
                .evidence
                .tool_calls
                .iter()
                .find(|r| r.record.canonical_id == "message_probe")
                .unwrap()
                .record
                .success
        );
        assert_eq!(
            f.trace
                .0
                .lock()
                .iter()
                .filter(|event| *event == "start:message_probe")
                .count(),
            1
        );
        assert_eq!(
            f.trace
                .0
                .lock()
                .iter()
                .filter(|event| *event == "complete:message_probe")
                .count(),
            1
        );
    }
}

struct Hanging;
#[async_trait]
impl Tool for Hanging {
    fn id(&self) -> &str {
        "hanging"
    }
    fn name(&self) -> &str {
        "hanging"
    }
    fn description(&self) -> &str {
        "Unknown effect after timeout"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object"})
    }
    fn classify_call(&self, _: &Value) -> ToolCallClassification {
        let mut classification = ToolCallClassification::from_metadata(&self.safety_metadata());
        classification.timeout_ms = Some(20);
        classification
    }
    async fn execute(&self, _: Value, _: ToolExecutionContext) -> ToolResult {
        std::future::pending::<ToolResult>().await
    }
}

#[tokio::test]
async fn fallback_acceptance_unknown_tool_timeout_cannot_start_fresh_fallback() {
    let rescue = Arc::new(AtomicUsize::new(0));
    let marker = ai_agents_core::encode_native_tool_call_markers(
        &[ToolCall {
            id: "original".into(),
            name: "hanging".into(),
            arguments: json!({}),
        }],
        None,
    )
    .unwrap();
    let yaml = "name: UnknownFallback\nsystem_prompt: test\ntools: [hanging,rescue]\nerror_recovery:\n  tools:\n    hanging:\n      max_retries: 0\n      on_failure: {action: fallback, fallback_tool: rescue}\n";
    let parent = Arc::new(
        AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(super::runner_tests::RecordingProvider::new(&[
                &marker, "done",
            ]))
            .auto_configure_features()
            .unwrap()
            .tool(Arc::new(Hanging))
            .tool(Arc::new(Rescue(rescue.clone())))
            .build()
            .unwrap(),
    );
    let store = support::store(parent.info().id, "fallback.v1", false).await;
    let runner = runner(parent.clone(), store.clone(), false);
    let result = runner.run("objective", None).await;
    assert!(
        !result
            .as_ref()
            .is_ok_and(|result| result.run.status == TaskRunStatus::Completed)
    );
    assert_eq!(rescue.load(Ordering::SeqCst), 0);
    let saved = store.list().await.unwrap().pop().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&saved.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert!(payload.reservations.iter().any(|reservation| matches!(
        reservation.state,
        TaskEffectState::Dispatched | TaskEffectState::Uncertain
    )));
    assert!(parent.set_context("blocked", json!(true)).is_err());
}

#[tokio::test]
async fn fallback_acceptance_retry_capacity_failure_keeps_settled_original_evidence() {
    let mut f = Box::pin(fixture(false, false, None, false)).await;
    let mut config = super::runner_tests::config(16);
    config.defaults.max_tool_calls = Some(3);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    f.runner = AutonomyRunner::try_new(
        f.parent.clone(),
        config,
        Default::default(),
        f.store.clone(),
        "fallback.v1".into(),
    )
    .unwrap();
    let first = f.runner.run("objective", None).await.unwrap();
    let done = answer(&f, &first.run).await.unwrap();
    assert_eq!(
        done.run.status,
        TaskRunStatus::LimitReached,
        "{:?}",
        done.run.stop_reason
    );
    let record = &done
        .run
        .evidence
        .tool_calls
        .iter()
        .find(|r| r.record.canonical_id == "message_probe")
        .unwrap()
        .record;
    assert!(record.executed && !record.success);
    assert!(record.metadata.contains_key("recovery_error"));
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
    assert_eq!(done.run.counters.tool_attempts, 3);
    assert_eq!(
        f.trace
            .0
            .lock()
            .iter()
            .filter(|event| *event == "complete:message_probe")
            .count(),
        1
    );
}

#[tokio::test]
async fn fallback_acceptance_retry_metadata_cannot_replace_private_hop_identity() {
    let f = Box::pin(fixture(false, false, None, false)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    let done = answer(&f, &first.run).await.unwrap();
    let record = &done
        .run
        .evidence
        .tool_calls
        .iter()
        .find(|r| r.record.canonical_id == "message_probe")
        .unwrap()
        .record;
    assert!(record.executed && record.success);
    assert_eq!(record.metadata["_task_executor_hop"], 0);
}

#[tokio::test]
async fn fallback_acceptance_retry_exhaustion_after_resume_keeps_pre_pause_consumption() {
    let f = Box::pin(fixture_with_failure(false, false, None, false, true)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    let done = answer(&f, &first.run).await.unwrap();
    assert_eq!(f.calls.load(Ordering::SeqCst), 3);
    assert_eq!(f.child.calls.load(Ordering::SeqCst), 3);
    assert_eq!(done.run.counters.tool_attempts, 4);
    assert!(
        !done
            .run
            .evidence
            .tool_calls
            .iter()
            .find(|r| r.record.canonical_id == "message_probe")
            .unwrap()
            .record
            .success
    );
    assert_eq!(
        f.trace
            .0
            .lock()
            .iter()
            .filter(|event| *event == "start:message_probe")
            .count(),
        1
    );
}

#[tokio::test]
async fn fallback_acceptance_resumed_failure_returns_to_actual_retry_index_without_replay() {
    for sqlite in support::backends() {
        let f = Box::pin(fixture(false, false, None, *sqlite)).await;
        let first = f.runner.run("objective", None).await.unwrap();
        assert_eq!(first.run.status, TaskRunStatus::Paused);
        assert_eq!(f.calls.load(Ordering::SeqCst), 2);
        let done = answer(&f, &first.run).await.unwrap();
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(f.calls.load(Ordering::SeqCst), 3);
        assert_eq!(f.child.calls.load(Ordering::SeqCst), 3);
        assert_eq!(done.run.counters.tool_attempts, 4);
        let trace = f.trace.0.lock().clone();
        assert_eq!(
            trace,
            vec![
                "start:message_probe",
                "complete:message_probe",
                "record:message_probe"
            ]
        );
        assert_eq!(
            done.run
                .evidence
                .tool_calls
                .iter()
                .filter(|capture| capture.record.canonical_id == "message_probe")
                .count(),
            1
        );
        assert!(f.registry.remove("worker").await.is_some());
    }
}

#[tokio::test]
async fn fallback_acceptance_modified_message_failure_parks_fallback_then_proves_tool_gate() {
    for sqlite in support::backends() {
        let f = Box::pin(fixture(true, true, None, *sqlite)).await;
        let first = f.runner.run("objective", None).await.unwrap();
        let child = approve(&f, &first.run, true).await.unwrap();
        let fallback = answer(&f, &child.run).await.unwrap();
        assert_eq!(
            fallback.run.status,
            TaskRunStatus::Paused,
            "{:?}",
            fallback.run.stop_reason
        );
        assert_eq!(f.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.rescue.load(Ordering::SeqCst), 0);
        assert!(answer(&f, &child.run).await.is_err());
        let done = approve(&f, &fallback.run, false).await.unwrap();
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(f.rescue.load(Ordering::SeqCst), 1);
        let parent_records: Vec<_> = done
            .run
            .evidence
            .tool_calls
            .iter()
            .filter(|r| r.record.metadata["_task_runtime_id"] == f.parent.info().id)
            .collect();
        assert_eq!(parent_records.len(), 2);
        assert_eq!(parent_records[0].record.canonical_id, "message_probe");
        assert_eq!(parent_records[1].record.canonical_id, "rescue");
        assert_eq!(
            parent_records[1].record.executed_arguments["message"],
            "modified"
        );
        assert_ne!(
            parent_records[0].record.metadata["_task_executor_hop"],
            parent_records[1].record.metadata["_task_executor_hop"]
        );
        assert_eq!(
            f.trace
                .0
                .lock()
                .iter()
                .filter(|event| *event == "start:message_probe")
                .count(),
            1
        );
        assert_eq!(
            f.trace
                .0
                .lock()
                .iter()
                .filter(|event| *event == "start:rescue")
                .count(),
            1
        );
    }
}

#[tokio::test]
async fn fallback_acceptance_original_message_deadline_does_not_reset_on_resume() {
    let f = Box::pin(fixture(false, false, Some(1000), false)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
    let done = answer(&f, &first.run).await.unwrap();
    assert_eq!(
        done.run.status,
        TaskRunStatus::Incomplete,
        "{:?}",
        done.run.stop_reason
    );
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
    assert_eq!(f.child.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.rescue.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn fallback_acceptance_repark_and_terminal_receipt_failure_retains_completed_ancestor() {
    for sqlite in support::backends() {
        for terminal in [false, true] {
            for mode in ["reject", "lost", "drop"] {
                let mut f = Box::pin(fixture(true, true, None, *sqlite)).await;
                let fault = ReceiptStore::new(
                    f.store.clone(),
                    mode,
                    if terminal {
                        TaskRunStatus::Completed
                    } else {
                        TaskRunStatus::Paused
                    },
                );
                f.runner = runner(f.parent.clone(), fault.clone(), true);
                let first = f.runner.run("objective", None).await.unwrap();
                let child = approve(&f, &first.run, true).await.unwrap();
                let mut parked = child;
                if terminal {
                    parked = answer(&f, &parked.run).await.unwrap();
                }
                fault.armed.store(true, Ordering::SeqCst);
                let mut resume = Box::pin(async {
                    if terminal {
                        approve(&f, &parked.run, false).await
                    } else {
                        answer(&f, &parked.run).await
                    }
                });
                if mode == "drop" {
                    tokio::select! {
                        result = &mut resume => panic!("receipt not held: {result:?}"),
                        permit = fault.reached.acquire() => permit.unwrap().forget(),
                        _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("receipt not reached"),
                    }
                    drop(resume);
                } else {
                    assert!(resume.await.is_err());
                }
                fault.assert_final_write(&parked.run.key.run_id).await;
                assert_eq!(f.calls.load(Ordering::SeqCst), 1);
                assert_eq!(f.child.calls.load(Ordering::SeqCst), 2);
                assert_eq!(f.rescue.load(Ordering::SeqCst), usize::from(terminal));
                assert!(f.parent.set_context("blocked", json!(true)).is_err());
                assert!(f.registry.remove("worker").await.is_none());
            }
        }
    }
}

#[tokio::test]
async fn fallback_acceptance_fallback_admission_and_settlement_faults_protect_parent_child() {
    for sqlite in support::backends() {
        for phase in ["admit", "settle"] {
            for mode in ["positive", "reject", "lost", "drop"] {
                let mut f = Box::pin(fixture(true, true, None, *sqlite)).await;
                let fault = ReceiptStore::checkpoint(
                    f.store.clone(),
                    mode,
                    json!({"tool":"rescue","phase":phase}),
                );
                f.runner = runner(f.parent.clone(), fault.clone(), true);
                let first = f.runner.run("objective", None).await.unwrap();
                let child = approve(&f, &first.run, true).await.unwrap();
                let fallback = answer(&f, &child.run).await.unwrap();
                fault.armed.store(true, Ordering::SeqCst);
                let mut resume = Box::pin(approve(&f, &fallback.run, false));
                if mode == "positive" {
                    assert_eq!(resume.await.unwrap().run.status, TaskRunStatus::Completed);
                    fault.assert_hit();
                } else {
                    if mode == "drop" {
                        tokio::select! {result=&mut resume=>panic!("checkpoint not held: {result:?}"),permit=fault.reached.acquire()=>permit.unwrap().forget(),_=tokio::time::sleep(std::time::Duration::from_secs(5))=>panic!("checkpoint not reached") }
                        drop(resume);
                    } else {
                        assert!(resume.await.is_err());
                    }
                    fault.assert_hit();
                    assert_eq!(
                        f.rescue.load(Ordering::SeqCst),
                        usize::from(phase == "settle")
                    );
                    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
                    assert!(f.parent.set_context("blocked", json!(true)).is_err());
                    assert!(f.registry.remove("worker").await.is_none());
                    assert!(approve(&f, &fallback.run, false).await.is_err());
                }
            }
        }
    }
}

struct Failing {
    id: &'static str,
    calls: Arc<AtomicUsize>,
}
#[async_trait]
impl Tool for Failing {
    fn id(&self) -> &str {
        self.id
    }
    fn name(&self) -> &str {
        self.id
    }
    fn description(&self) -> &str {
        "Known safe failed result"
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
        self.calls.fetch_add(1, Ordering::SeqCst);
        ToolResult::error("known failure")
    }
}

#[tokio::test]
async fn fallback_acceptance_multihop_and_cycle_denial_keep_distinct_ordered_evidence() {
    for cycle in [false, true] {
        let first = Arc::new(AtomicUsize::new(0));
        let middle = Arc::new(AtomicUsize::new(0));
        let rescue = Arc::new(AtomicUsize::new(0));
        let trace = Arc::new(Trace::default());
        let marker = ai_agents_core::encode_native_tool_call_markers(
            &[ToolCall {
                id: "original".into(),
                name: "first".into(),
                arguments: json!({"value":"original"}),
            }],
            None,
        )
        .unwrap();
        let yaml = format!(
            "name: Chain\nsystem_prompt: test\ntools: [first,middle,rescue]\nhitl:\n  tools:\n    rescue:\n      require_approval: true\nerror_recovery:\n  tools:\n    first:\n      max_retries: 0\n      on_failure: {{action: fallback, fallback_tool: middle}}\n    middle:\n      max_retries: 0\n      on_failure: {{action: fallback, fallback_tool: {}}}\n",
            if cycle { "first" } else { "rescue" }
        );
        let agent = Arc::new(
            AgentBuilder::from_yaml(&yaml)
                .unwrap()
                .llm(super::runner_tests::RecordingProvider::new(&[
                    &marker, "done",
                ]))
                .auto_configure_features()
                .unwrap()
                .hooks(trace.clone())
                .tool(Arc::new(Failing {
                    id: "first",
                    calls: first.clone(),
                }))
                .tool(Arc::new(Failing {
                    id: "middle",
                    calls: middle.clone(),
                }))
                .tool(Arc::new(Rescue(rescue.clone())))
                .build()
                .unwrap(),
        );
        let store = support::store(agent.info().id, "fallback.v1", false).await;
        let runner = runner(agent, store, !cycle);
        let mut done = runner.run("objective", None).await.unwrap();
        if !cycle {
            done = runner
                .resume(
                    &done.run.key.run_id,
                    done.run.revision,
                    TaskResumeInput::Approval {
                        request_id: done.run.pending.as_ref().unwrap().id.clone(),
                        result: ApprovalResult::Approved,
                    },
                )
                .await
                .unwrap();
        }
        assert_eq!(done.run.status, TaskRunStatus::Completed);
        assert_eq!(first.load(Ordering::SeqCst), 1);
        assert_eq!(middle.load(Ordering::SeqCst), 1);
        assert_eq!(rescue.load(Ordering::SeqCst), usize::from(!cycle));
        let records = &done.run.evidence.tool_calls;
        assert_eq!(records.len(), 3);
        assert_eq!(
            records
                .iter()
                .map(|r| r.record.metadata["_task_executor_hop"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        if cycle {
            assert!(!records[2].record.executed && !records[2].record.success);
        }
        let events = trace.0.lock();
        assert_eq!(
            &events[..6],
            &[
                "start:first",
                "complete:first",
                "record:first",
                "start:middle",
                "complete:middle",
                "record:middle"
            ]
        );
    }
}

#[tokio::test]
async fn fallback_acceptance_skill_location_resumes_actual_fallback_without_repeating_original() {
    let first = Arc::new(AtomicUsize::new(0));
    let rescue = Arc::new(AtomicUsize::new(0));
    let trace = Arc::new(Trace::default());
    let yaml = "name: SkillFallback\nsystem_prompt: test\nllm:\n  default: default\n  router: router\ntools: [first,rescue]\nhitl:\n  tools:\n    rescue:\n      require_approval: true\nerror_recovery:\n  tools:\n    first:\n      max_retries: 0\n      on_failure: {action: fallback, fallback_tool: rescue}\nskills:\n  - id: script\n    description: work\n    trigger: work\n    steps:\n      - tool: first\n        args: {value: original}\n      - prompt: 'Finish {{ steps[0].result }}'\n";
    let mut llms = ai_agents_llm::LLMRegistry::new();
    llms.register(
        "default",
        super::runner_tests::RecordingProvider::new(&["done"]),
    );
    llms.register(
        "router",
        super::runner_tests::RecordingProvider::new(&["script"]),
    );
    let agent = Arc::new(
        AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm_registry(llms)
            .auto_configure_features()
            .unwrap()
            .hooks(trace.clone())
            .tool(Arc::new(Failing {
                id: "first",
                calls: first.clone(),
            }))
            .tool(Arc::new(Rescue(rescue.clone())))
            .build()
            .unwrap(),
    );
    let store = support::store(agent.info().id, "fallback.v1", false).await;
    let runner = runner(agent, store, true);
    let first_pause = runner.run("objective", None).await.unwrap();
    let done = runner
        .resume(
            &first_pause.run.key.run_id,
            first_pause.run.revision,
            TaskResumeInput::Approval {
                request_id: first_pause.run.pending.as_ref().unwrap().id.clone(),
                result: ApprovalResult::Modified {
                    changes: std::collections::HashMap::from([("value".into(), json!("modified"))]),
                },
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(first.load(Ordering::SeqCst), 1);
    assert_eq!(rescue.load(Ordering::SeqCst), 1);
    assert_eq!(
        trace
            .0
            .lock()
            .iter()
            .filter(|event| *event == "start:rescue")
            .count(),
        1
    );
    assert_eq!(
        done.run
            .evidence
            .tool_calls
            .iter()
            .find(|r| r.record.canonical_id == "rescue")
            .unwrap()
            .record
            .executed_arguments["value"],
        "modified"
    );
}

#[tokio::test]
async fn fallback_acceptance_modified_original_question_fallback_timeout_never_replays_ancestor() {
    for timeout in [false, true] {
        let original = Arc::new(AtomicUsize::new(0));
        let args = json!({"question":"Original","options":["yes"],"allow_other":false,"timeout_seconds":1});
        let marker = ai_agents_core::encode_native_tool_call_markers(
            &[ToolCall {
                id: "original".into(),
                name: "first".into(),
                arguments: args,
            }],
            None,
        )
        .unwrap();
        let yaml = "name: QuestionFallback\nsystem_prompt: test\ntools: [first,ask_user]\nhitl:\n  tools:\n    first:\n      require_approval: true\nerror_recovery:\n  tools:\n    first:\n      max_retries: 0\n      on_failure: {action: fallback, fallback_tool: ask_user}\n";
        let trace = Arc::new(Trace::default());
        let agent = Arc::new(
            AgentBuilder::from_yaml(yaml)
                .unwrap()
                .llm(super::runner_tests::RecordingProvider::new(&[
                    &marker, "done",
                ]))
                .auto_configure_features()
                .unwrap()
                .hooks(trace.clone())
                .tool(Arc::new(Failing {
                    id: "first",
                    calls: original.clone(),
                }))
                .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
                    Arc::new(parking_lot::RwLock::new(None)),
                )))
                .build()
                .unwrap(),
        );
        let store = support::store(agent.info().id, "fallback.v1", false).await;
        let runner = runner(agent.clone(), store, false);
        let first = runner.run("objective", None).await.unwrap();
        let question = runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::Approval {
                    request_id: first.run.pending.as_ref().unwrap().id.clone(),
                    result: ApprovalResult::Modified {
                        changes: std::collections::HashMap::from([(
                            "question".into(),
                            json!("Modified"),
                        )]),
                    },
                },
            )
            .await
            .unwrap();
        assert_eq!(question.run.status, TaskRunStatus::Paused);
        if timeout {
            tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        }
        let input = if timeout {
            TaskResumeInput::QuestionTimeout {
                request_id: question.run.pending.as_ref().unwrap().id.clone(),
            }
        } else {
            TaskResumeInput::UserAnswer {
                request_id: question.run.pending.as_ref().unwrap().id.clone(),
                answer: json!({"answered":true,"selected":["yes"]}),
            }
        };
        let done = runner
            .resume(&question.run.key.run_id, question.run.revision, input)
            .await
            .unwrap();
        assert_eq!(
            done.run.status,
            if timeout {
                TaskRunStatus::Incomplete
            } else {
                TaskRunStatus::Completed
            }
        );
        assert_eq!(original.load(Ordering::SeqCst), 1);
        assert_eq!(
            trace
                .0
                .lock()
                .iter()
                .filter(|event| *event == "start:first")
                .count(),
            1
        );
        if !timeout {
            assert_eq!(
                trace
                    .0
                    .lock()
                    .iter()
                    .filter(|event| *event == "start:ask_user")
                    .count(),
                1
            );
        }
        assert!(agent.set_context("released", json!(true)).is_ok());
    }
}

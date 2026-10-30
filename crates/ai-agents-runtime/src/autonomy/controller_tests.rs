#[path = "restoration_controller_tests.rs"]
mod restoration_tests;
use super::*;
use crate::{Agent, AgentBuilder};
use ai_agents_core::{Tool, ToolExecutionContext, ToolResult, ToolSafetyMetadata};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Validator {
    descriptor: AdapterDescriptor,
    mode: &'static str,
    callbacks: Arc<AtomicUsize>,
}
impl AutonomyValidator for Validator {
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.descriptor
    }
    fn validate_config(&self, _: &Value) -> ai_agents_core::Result<()> {
        Ok(())
    }
    fn evaluate(&self, input: &ValidationInput<'_>) -> ai_agents_core::Result<ValidationDecision> {
        self.callbacks.fetch_add(1, Ordering::SeqCst);
        if self.mode == "mutating_external" && input.observations.is_empty() {
            return Ok(ValidationDecision::NeedObservations {
                requests: vec![ValidationObservationRequest::Tool {
                    id: "mutation".into(),
                    tool: "mutate".into(),
                    arguments: json!({}),
                    host_binding: None,
                }],
                checkpoint: json!({}),
            });
        }
        if self.mode.contains("external")
            && (input.previous.get("external_signal").is_none()
                || (self.mode == "external_repeat"
                    && input.previous.get("external_signal") == Some(&json!(false))))
        {
            return Ok(ValidationDecision::AwaitExternal {
                binding: "ready".into(),
                checkpoint: json!({"retained":true}),
            });
        }
        if self.mode == "batch" && input.observations.is_empty() {
            return Ok(ValidationDecision::NeedObservations {
                requests: ["first", "second"]
                    .into_iter()
                    .map(|id| ValidationObservationRequest::Tool {
                        id: id.into(),
                        tool: "probe".into(),
                        arguments: json!({"value":id}),
                        host_binding: None,
                    })
                    .collect(),
                checkpoint: json!({}),
            });
        }
        if !self.mode.contains("external") && input.observations.is_empty() {
            return Ok(ValidationDecision::NeedObservations {
                requests: vec![ValidationObservationRequest::Tool {
                    id: "observe".into(),
                    tool: if self.mode.starts_with("fallback") {
                        "fail_probe"
                    } else if self.mode == "question" {
                        "ask_user"
                    } else {
                        "probe"
                    }
                    .into(),
                    arguments: if self.mode == "question" || self.mode == "fallback_question" {
                        json!({"question":"Select","options":["yes"],"allow_other":false})
                    } else {
                        json!({"value":1})
                    },
                    host_binding: if self.mode == "host" {
                        Some("fixed".into())
                    } else {
                        None
                    },
                }],
                checkpoint: json!({"retained":true}),
            });
        }
        let success = input.observations.values().all(|observation| !matches!(&observation.result, ObservationResult::Tool { record } if !record.success));
        Ok(ValidationDecision::Complete {
            outcome: if success {
                GateOutcome::Pass
            } else {
                GateOutcome::Fail
            },
            reason: "observed".into(),
            metrics: json!({}),
            evidence_refs: vec![],
        })
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
        "Count real observation invocations"
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
struct FailProbe(Arc<AtomicUsize>);
#[async_trait]
impl Tool for FailProbe {
    fn id(&self) -> &str {
        "fail_probe"
    }
    fn name(&self) -> &str {
        "fail_probe"
    }
    fn description(&self) -> &str {
        "Count the failed original observation"
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
        ToolResult::error("original observation failed")
    }
}

struct Mutation(Arc<AtomicUsize>);
#[async_trait]
impl Tool for Mutation {
    fn id(&self) -> &str {
        "mutate"
    }
    fn name(&self) -> &str {
        "mutate"
    }
    fn description(&self) -> &str {
        "Count a managed mutation"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object"})
    }
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: false,
            ..Default::default()
        }
    }
    async fn execute(&self, _: Value, _: ToolExecutionContext) -> ToolResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        ToolResult::ok("mutated")
    }
}
struct NoApproval;
#[async_trait]
impl ai_agents_hitl::ApprovalHandler for NoApproval {
    async fn request_approval(
        &self,
        _: ai_agents_hitl::ApprovalRequest,
    ) -> ai_agents_hitl::ApprovalResult {
        panic!("observation must park before polling the handler")
    }
}
pub(super) struct Fixture {
    pub(super) agent: Arc<crate::RuntimeAgent>,
    pub(super) runner: AutonomyRunner,
    pub(super) store: Arc<ScopedTaskRunStore>,
    pub(super) calls: Arc<AtomicUsize>,
    pub(super) callbacks: Arc<AtomicUsize>,
    pub(super) provider: Arc<super::runner_tests::RecordingProvider>,
    pub(super) config: AutonomyConfig,
}
// The real registry and executor run with exactly one model slot, so replay cannot accidentally pass.
pub(super) fn fixture(mode: &'static str) -> Fixture {
    fixture_configured(mode, |_| {})
}

// Host evidence fixtures install frozen bindings before any runtime consumer captures the registry.
pub(super) fn fixture_configured(
    mode: &'static str,
    configure: impl FnOnce(&mut AutonomyExtensions),
) -> Fixture {
    let calls = Arc::new(AtomicUsize::new(0));
    let callbacks = Arc::new(AtomicUsize::new(0));
    let mut extensions = AutonomyExtensions::builtins();
    extensions
        .register_validator(Arc::new(Validator {
            descriptor: AdapterDescriptor {
                id: "host.observation".into(),
                contract_version: 1,
                tools: if mode == "mutating_external" {
                    vec!["mutate".into()]
                } else if mode.starts_with("external") {
                    vec![]
                } else {
                    vec![
                        if mode.starts_with("fallback") {
                            "fail_probe"
                        } else if mode == "question" {
                            "ask_user"
                        } else {
                            "probe"
                        }
                        .into(),
                    ]
                },
                conditions: if mode.contains("external") {
                    vec!["ready".into()]
                } else {
                    vec![]
                },
                ..Default::default()
            },
            mode,
            callbacks: callbacks.clone(),
        }))
        .unwrap();
    extensions
        .register_condition(HostConditionBinding {
            id: "ready".into(),
            version: 1,
            shape: ExternalSignalShape::Boolean,
        })
        .unwrap();
    if mode == "host" {
        extensions
            .register_host_validation(HostValidationBinding {
                id: "fixed".into(),
                validator: "host.observation".into(),
                version: 1,
                profile: None,
                scope: "task".into(),
                tool: "probe".into(),
                arguments: json!({"value":1}),
                require_approval: true,
            })
            .unwrap();
    }
    configure(&mut extensions);
    let provider = super::runner_tests::RecordingProvider::new(&["done", "ordinary"]);
    let yaml = "name: Observation\nsystem_prompt: test\ntools: [probe, ask_user, mutate, fail_probe]\nerror_recovery:\n  tools:\n    fail_probe:\n      max_retries: 0\n      on_failure:\n        action: fallback\n        fallback_tool: probe\nhitl:\n  tools:\n    probe:\n      require_approval: true\n";
    let yaml = if mode == "fallback_question" {
        yaml.replace("fallback_tool: probe", "fallback_tool: ask_user")
    } else if mode == "external_tool_free" {
        yaml.replace("tools: [probe, ask_user, mutate, fail_probe]", "tools: []")
    } else {
        yaml.into()
    };
    let mut agent = AgentBuilder::from_yaml(&yaml)
        .unwrap()
        .llm(provider.clone())
        .auto_configure_features()
        .unwrap()
        .approval_handler(Arc::new(NoApproval))
        .autonomy_extensions(extensions)
        .tool(Arc::new(Probe(calls.clone())))
        .tool(Arc::new(FailProbe(calls.clone())))
        .tool(Arc::new(Mutation(calls.clone())))
        .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
            Arc::new(parking_lot::RwLock::new(None)),
        )))
        .build()
        .unwrap();
    if mode == "host" {
        agent = agent.with_declared_tool_ids(Some(vec![]));
    }
    let agent = Arc::new(agent);
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "controller.v1".into()).unwrap(),
    );
    let mut config = super::runner_tests::config(1);
    config.defaults.max_tool_calls = Some(if mode == "batch" || mode.starts_with("fallback") {
        2
    } else {
        1
    });
    config.defaults.validation = Some(
        serde_yaml::from_str(
            if mode == "host" {
                "checks: [{id: check, adapter: host.observation, schedule: each_cycle, config: {host_binding: fixed}}]"
            } else {
                "checks: [{id: check, adapter: host.observation, schedule: each_cycle, config: {}}]"
            },
        )
        .unwrap(),
    );
    config.defaults.completion = Some(CompletionGate::ValidatorPassed("check".into()));
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    let runner = AutonomyRunner::try_new(
        agent.clone(),
        config.clone(),
        Default::default(),
        store.clone(),
        "controller.v1".into(),
    )
    .unwrap();
    Fixture {
        agent,
        runner,
        store,
        calls,
        callbacks,
        provider,
        config,
    }
}

#[tokio::test]
async fn controller_external_resume_preserves_turn_and_callback_positions() {
    let f = fixture("external");
    let paused = f.runner.run("objective", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    assert_eq!(f.callbacks.load(Ordering::SeqCst), 1);
    assert_eq!(f.provider.calls.load(Ordering::SeqCst), 1);
    let signal = ExternalSignal {
        request_id: paused.run.pending.as_ref().unwrap().id.clone(),
        event_id: "event".into(),
        target_revision: None,
        payload: json!(true),
    };
    let mut invalid = signal.clone();
    invalid.payload = json!("wrong shape");
    assert!(
        f.runner
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::ExternalSignal(invalid)
            )
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
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::ExternalSignal(signal.clone()),
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(done.run.counters.turns, 1);
    assert_eq!(done.run.counters.llm_attempts, 1);
    assert_eq!(f.callbacks.load(Ordering::SeqCst), 2);
    assert!(
        f.runner
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::ExternalSignal(signal)
            )
            .await
            .is_err()
    );
    assert_eq!(f.agent.chat("follow-up").await.unwrap().content, "ordinary");
}

#[tokio::test]
async fn controller_observation_resume_preserves_actual_request_and_admission() {
    for mode in ["approval", "host", "question"] {
        let f = fixture(mode);
        let paused = f.runner.run("objective", None).await.unwrap();
        assert_eq!(
            paused.run.status,
            TaskRunStatus::Paused,
            "{:?}",
            paused.run.stop_reason
        );
        assert_eq!(f.calls.load(Ordering::SeqCst), 0);
        assert_eq!(f.callbacks.load(Ordering::SeqCst), 1);
        assert_eq!(paused.run.counters.tool_attempts, 0);
        let request_id = paused.run.pending.as_ref().unwrap().id.clone();
        let input = if mode == "question" {
            TaskResumeInput::UserAnswer {
                request_id,
                answer: json!({"answered":true,"selected":["yes"]}),
            }
        } else {
            TaskResumeInput::Approval {
                request_id,
                result: ai_agents_hitl::ApprovalResult::Approved,
            }
        };
        let result = f
            .runner
            .resume(&paused.run.key.run_id, paused.run.revision, input)
            .await;
        assert!(
            result.is_ok(),
            "mode={mode} error={:?} revision={}",
            result.as_ref().err(),
            f.store
                .load(&paused.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .revision
        );
        let mut done = result.unwrap();
        if mode == "host" {
            assert_eq!(done.run.status, TaskRunStatus::Paused);
            assert_eq!(done.run.counters.tool_attempts, 0);
            assert_eq!(f.callbacks.load(Ordering::SeqCst), 1);
            assert_eq!(f.calls.load(Ordering::SeqCst), 0);
            done = f
                .runner
                .resume(
                    &done.run.key.run_id,
                    done.run.revision,
                    TaskResumeInput::Approval {
                        request_id: done.run.pending.as_ref().unwrap().id.clone(),
                        result: ai_agents_hitl::ApprovalResult::Approved,
                    },
                )
                .await
                .unwrap();
        }
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "mode={mode} {:?}",
            done.run.stop_reason
        );
        assert_eq!(done.run.counters.tool_attempts, 1);
        assert_eq!(done.run.counters.llm_attempts, 1);
        assert_eq!(done.run.counters.turns, 1);
        assert_eq!(f.callbacks.load(Ordering::SeqCst), 2);
        assert_eq!(f.provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            f.calls.load(Ordering::SeqCst),
            usize::from(mode != "question")
        );
    }
}

#[tokio::test]
async fn controller_cancel_does_not_invoke_parked_work() {
    for mode in ["external", "approval", "host", "question"] {
        let f = fixture(mode);
        let paused = f.runner.run("objective", None).await.unwrap();
        assert_eq!(paused.run.status, TaskRunStatus::Paused);
        let cancelled = f
            .runner
            .cancel_paused(&paused.run.key.run_id, paused.run.revision)
            .await
            .unwrap();
        assert_eq!(cancelled.run.status, TaskRunStatus::Cancelled);
        assert_eq!(f.calls.load(Ordering::SeqCst), 0);
        assert_eq!(f.callbacks.load(Ordering::SeqCst), 1);
        assert_eq!(f.agent.chat("follow-up").await.unwrap().content, "ordinary");
    }
}

#[tokio::test]
async fn controller_fallback_observation_resumes_actual_hop_without_original_replay() {
    for mode in ["fallback", "fallback_question"] {
        let f = fixture(mode);
        let first = f.runner.run("objective", None).await.unwrap();
        assert_eq!(
            first.run.status,
            TaskRunStatus::Paused,
            "{:?}",
            first.run.stop_reason
        );
        assert_eq!(f.calls.load(Ordering::SeqCst), 1);
        assert_eq!(first.run.counters.tool_attempts, 1);
        assert_eq!(f.callbacks.load(Ordering::SeqCst), 1);
        let id = first.run.pending.as_ref().unwrap().id.clone();
        let input = if mode == "fallback" {
            TaskResumeInput::Approval {
                request_id: id,
                result: ai_agents_hitl::ApprovalResult::Approved,
            }
        } else {
            TaskResumeInput::UserAnswer {
                request_id: id,
                answer: json!({"answered":true,"selected":["yes"]}),
            }
        };
        let done = f
            .runner
            .resume(&first.run.key.run_id, first.run.revision, input)
            .await
            .unwrap();
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(
            f.calls.load(Ordering::SeqCst),
            if mode == "fallback" { 2 } else { 1 }
        );
        assert_eq!(done.run.counters.tool_attempts, 2);
        assert_eq!(done.run.counters.llm_attempts, 1);
        assert_eq!(done.run.counters.turns, 1);
        assert_eq!(f.callbacks.load(Ordering::SeqCst), 2);
    }
}

#[tokio::test]
async fn controller_batch_resumes_remaining_observations_without_callback_reentry() {
    let f = fixture("batch");
    let first = f.runner.run("objective", None).await.unwrap();
    let second = f
        .runner
        .resume(
            &first.run.key.run_id,
            first.run.revision,
            TaskResumeInput::Approval {
                request_id: first.run.pending.as_ref().unwrap().id.clone(),
                result: ai_agents_hitl::ApprovalResult::Approved,
            },
        )
        .await
        .unwrap();
    assert_eq!(second.run.status, TaskRunStatus::Paused);
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.callbacks.load(Ordering::SeqCst), 1);
    assert_eq!(second.run.counters.tool_attempts, 1);
    assert_ne!(
        first.run.pending.as_ref().unwrap().id,
        second.run.pending.as_ref().unwrap().id
    );
    assert!(
        f.runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::Approval {
                    request_id: first.run.pending.as_ref().unwrap().id.clone(),
                    result: ai_agents_hitl::ApprovalResult::Approved,
                }
            )
            .await
            .is_err()
    );
    let done = f
        .runner
        .resume(
            &second.run.key.run_id,
            second.run.revision,
            TaskResumeInput::Approval {
                request_id: second.run.pending.as_ref().unwrap().id.clone(),
                result: ai_agents_hitl::ApprovalResult::Approved,
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
    assert_eq!(f.callbacks.load(Ordering::SeqCst), 2);
    assert_eq!(f.provider.calls.load(Ordering::SeqCst), 1);
    assert_eq!(done.run.counters.tool_attempts, 2);
}

#[tokio::test]
async fn controller_signal_after_mutation_preserves_attempt_and_effect_identity() {
    let f = fixture("mutating_external");
    let paused = f.runner.run("objective", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.callbacks.load(Ordering::SeqCst), 2);
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
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.callbacks.load(Ordering::SeqCst), 3);
    assert_eq!(done.run.counters.tool_attempts, 1);
    assert_eq!(done.run.counters.turns, 1);
}

#[tokio::test]
async fn controller_oversized_signal_rejects_before_claim_and_valid_signal_still_resumes() {
    let f = fixture("external");
    let paused = f.runner.run("objective", None).await.unwrap();
    assert!(
        f.runner
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::ExternalSignal(ExternalSignal {
                    request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                    event_id: "x".repeat(70_000),
                    target_revision: None,
                    payload: json!(true),
                })
            )
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
    assert_eq!(f.callbacks.load(Ordering::SeqCst), 1);
    let done = f
        .runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::ExternalSignal(ExternalSignal {
                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                event_id: "valid".into(),
                target_revision: None,
                payload: json!(true),
            }),
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
}

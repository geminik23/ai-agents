use super::*;
use crate::AgentBuilder;
use crate::spawner::{AgentRegistry, AgentSpawner, SpawnedAgent};
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Validator {
    descriptor: AdapterDescriptor,
    calls: Arc<AtomicUsize>,
    hosted: bool,
}
impl AutonomyValidator for Validator {
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.descriptor
    }
    fn validate_config(&self, _: &Value) -> ai_agents_core::Result<()> {
        Ok(())
    }
    fn evaluate(&self, input: &ValidationInput<'_>) -> ai_agents_core::Result<ValidationDecision> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if input.observations.is_empty() {
            Ok(ValidationDecision::NeedObservations {
                requests: vec![ValidationObservationRequest::Tool {
                    id: "message".into(),
                    tool: "send_agent_message".into(),
                    arguments: json!({"to":"worker","message":"objective"}),
                    host_binding: if self.hosted {
                        Some("fixed-message".into())
                    } else {
                        None
                    },
                }],
                checkpoint: json!({}),
            })
        } else {
            let successful = input.observations.values().all(|observation| matches!(&observation.result, ObservationResult::Tool { record } if record.executed && record.success));
            Ok(ValidationDecision::Complete {
                outcome: if successful {
                    GateOutcome::Pass
                } else {
                    GateOutcome::Fail
                },
                reason: "message settled".into(),
                metrics: json!({}),
                evidence_refs: vec![],
            })
        }
    }
}
struct Fixture {
    runner: AutonomyRunner,
    root: Arc<crate::RuntimeAgent>,
    main: Arc<super::runner_tests::RecordingProvider>,
    child: Arc<super::runner_tests::RecordingProvider>,
    calls: Arc<AtomicUsize>,
    store: Arc<ScopedTaskRunStore>,
}
// This observation is an actual Task-origin message, not a model call manufactured to obtain child cursor authority.
async fn fixture() -> Fixture {
    Box::pin(fixture_mode(false, false)).await
}

// A second child interaction remains inside the same observation and must not repeat its parent message delivery.
async fn fixture_repause(repause: bool) -> Fixture {
    Box::pin(fixture_mode(repause, false)).await
}

// Private fixed host validation authority survives child parking without exposing the operation to the model.
async fn fixture_mode(repause: bool, hosted: bool) -> Fixture {
    let registry = Arc::new(AgentRegistry::new());
    let question = ai_agents_core::encode_native_tool_call_markers(
        &[ai_agents_core::ToolCall {
            id: "child-question".into(),
            name: "ask_user".into(),
            arguments: json!({"question":"Choose","options":["yes"],"allow_other":false}),
        }],
        None,
    )
    .unwrap();
    let next = ai_agents_core::encode_native_tool_call_markers(
        &[ai_agents_core::ToolCall {
            id: "child-question-next".into(),
            name: "ask_user".into(),
            arguments: json!({"question":"Choose again","options":["yes"],"allow_other":false}),
        }],
        None,
    )
    .unwrap();
    let child = if repause {
        super::runner_tests::RecordingProvider::new(&[&question, &next, "done"])
    } else {
        super::runner_tests::RecordingProvider::new(&[&question, "done"])
    };
    let yaml = "name: ObservationWorker\nsystem_prompt: worker\ntools: [ask_user]\n";
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
    let calls = Arc::new(AtomicUsize::new(0));
    let mut extensions = AutonomyExtensions::builtins();
    extensions
        .register_validator(Arc::new(Validator {
            descriptor: AdapterDescriptor {
                id: "host.message".into(),
                contract_version: 1,
                tools: vec!["send_agent_message".into()],
                ..Default::default()
            },
            calls: calls.clone(),
            hosted,
        }))
        .unwrap();
    if hosted {
        extensions
            .register_host_validation(HostValidationBinding {
                id: "fixed-message".into(),
                validator: "host.message".into(),
                version: 1,
                profile: None,
                scope: "task".into(),
                tool: "send_agent_message".into(),
                arguments: json!({"to":"worker","message":"objective"}),
                require_approval: true,
            })
            .unwrap();
    }
    let main = super::runner_tests::RecordingProvider::new(&["done"]);
    let yaml = if hosted {
        "name: ObservationRoot\nsystem_prompt: root\ntools: []\nhitl:\n  message_language:\n    strategy: explicit\n    explicit: en\n  tools:\n    send_agent_message:\n      require_approval: true\n      approval_message: Review message\n"
    } else {
        "name: ObservationRoot\nsystem_prompt: root\ntools: [send_agent_message]\n"
    };
    let builder = AgentBuilder::from_yaml(yaml)
        .unwrap()
        .llm(main.clone())
        .auto_configure_features()
        .unwrap()
        .autonomy_extensions(extensions)
        .tool(Arc::new(crate::spawner::tools::SendMessageTool::new(
            registry.clone(),
            "root",
        )))
        .build()
        .unwrap()
        .with_spawner_handles(Arc::new(AgentSpawner::new()), registry);
    let root = Arc::new(builder);
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(root.info().id, None, "observation-message.v1".into())
            .unwrap(),
    );
    let mut config = super::runner_tests::config(if repause { 4 } else { 3 });
    config.defaults.max_tool_calls = Some(if repause { 3 } else { 2 });
    config.defaults.validation = Some(
        serde_yaml::from_str(
            if hosted { "checks: [{id: check, adapter: host.message, schedule: each_cycle, config: {host_binding: fixed-message}}]" }
            else { "checks: [{id: check, adapter: host.message, schedule: each_cycle, config: {}}]" },
        )
        .unwrap(),
    );
    config.defaults.completion = Some(CompletionGate::ValidatorPassed("check".into()));
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    let runner = AutonomyRunner::try_new(
        root.clone(),
        config,
        Default::default(),
        store.clone(),
        "observation-message.v1".into(),
    )
    .unwrap();
    Fixture {
        runner,
        root,
        main,
        child,
        calls,
        store,
    }
}

#[tokio::test]
async fn observation_message_fixed_host_permit_is_reconstructed_without_model_exposure() {
    let f = Box::pin(fixture_mode(false, true)).await;
    let mut result = f.runner.run("objective", None).await.unwrap();
    let mut pauses = 0;
    while result.run.status == TaskRunStatus::Paused {
        pauses += 1;
        assert!(pauses <= 3);
        let id = result.run.pending.as_ref().unwrap().id.clone();
        let input = if matches!(
            result.run.pending.as_ref().unwrap().kind,
            TaskPendingKind::Approval
        ) {
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
        result = f
            .runner
            .resume(&result.run.key.run_id, result.run.revision, input)
            .await
            .unwrap();
    }
    assert_eq!(
        result.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        result.run.stop_reason
    );
    assert_eq!(pauses, 3);
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.child.calls.load(Ordering::SeqCst), 2);
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
    assert_eq!(result.run.counters.tool_attempts, 2);
}

#[tokio::test]
async fn observation_message_child_repause_preserves_original_observation_and_callback() {
    let f = Box::pin(fixture_repause(true)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    let second = f
        .runner
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
        second.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        second.run.stop_reason
    );
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.child.calls.load(Ordering::SeqCst), 2);
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
    assert_ne!(
        first.run.pending.as_ref().unwrap().id,
        second.run.pending.as_ref().unwrap().id
    );
    let done = f
        .runner
        .resume(
            &second.run.key.run_id,
            second.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: second.run.pending.as_ref().unwrap().id.clone(),
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await;
    assert!(
        done.is_ok(),
        "error={:?}, before={} after={}, child={}",
        done.as_ref().err(),
        second.run.revision,
        f.store
            .load(&second.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .revision,
        f.child.calls.load(Ordering::SeqCst)
    );
    let done = done.unwrap();
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.child.calls.load(Ordering::SeqCst), 3);
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
    assert_eq!(done.run.counters.turns, 1);
    assert_eq!(done.run.counters.tool_attempts, 3);
}

#[tokio::test]
async fn observation_message_child_cancel_and_stop_do_not_repeat_controller_or_child_work() {
    for stop in [false, true] {
        let f = Box::pin(fixture()).await;
        let first = f.runner.run("objective", None).await.unwrap();
        assert_eq!(first.run.status, TaskRunStatus::Paused);
        let done = if stop {
            f.runner
                .stop_paused(
                    &first.run.key.run_id,
                    first.run.revision,
                    "Stop validation child",
                )
                .await
                .unwrap()
        } else {
            f.runner
                .cancel_paused(&first.run.key.run_id, first.run.revision)
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
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.child.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.calls.load(Ordering::SeqCst), 1);
        assert!(f.root.set_context("released", json!(true)).is_ok());
    }
}

#[tokio::test]
async fn observation_message_child_resume_returns_to_exact_validation_position() {
    let f = Box::pin(fixture()).await;
    let first = f.runner.run("objective", None).await.unwrap();
    assert_eq!(
        first.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        first.run.stop_reason
    );
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.child.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
    let done = f
        .runner
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
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.child.calls.load(Ordering::SeqCst), 2);
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
    assert_eq!(done.run.counters.turns, 1);
    assert_eq!(done.run.counters.tool_attempts, 2);
    assert_eq!(done.run.counters.llm_attempts, 3);
    assert!(f.root.set_context("released", json!(true)).is_ok());
}

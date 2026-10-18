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
    external: bool,
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
            if self.external && input.previous.get("external_signal").is_none() {
                return Ok(ValidationDecision::AwaitExternal {
                    binding: "ready".into(),
                    checkpoint: json!({"child_settled":true}),
                });
            }
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
    worker: Arc<crate::RuntimeAgent>,
    calls: Arc<AtomicUsize>,
    store: Arc<ScopedTaskRunStore>,
    config: AutonomyConfig,
}
// Superseding a validation child closes its exact history and retires the old check without redispatching it.
#[tokio::test]
async fn host_control_observation_child_objective_supersession_keeps_parent_custody() {
    for sqlite in super::continuation_test_support::backends() {
        let mut f = Box::pin(fixture()).await;
        f.config.defaults.max_llm_calls = Some(16);
        f.config.defaults.max_tool_calls = Some(8);
        f.store = super::continuation_test_support::store(
            f.root.info().id,
            "observation-message.v1",
            *sqlite,
        )
        .await;
        f.runner = AutonomyRunner::try_new(
            f.root.clone(),
            f.config.clone(),
            Default::default(),
            f.store.clone(),
            "observation-message.v1".into(),
        )
        .unwrap();
        let first = f.runner.run("old", None).await.unwrap();
        let old_calls = f.child.calls.load(Ordering::SeqCst);
        let revised = f
            .runner
            .modify_objective_paused(
                &first.run.key.run_id,
                first.run.revision,
                "new",
                "retire validation child",
            )
            .await
            .unwrap();
        assert_eq!(f.child.calls.load(Ordering::SeqCst), old_calls);
        let saved = f
            .store
            .load(&revised.run.key.run_id)
            .await
            .unwrap()
            .unwrap();
        let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload).unwrap();
        assert!(
            payload
                .children
                .iter()
                .all(|child| child.pending.is_none() && child.result.is_some())
        );
        assert!(f.root.set_context("unrelated", json!(true)).is_err());
        let stopped = f
            .runner
            .stop_paused(
                &revised.run.key.run_id,
                revised.run.revision,
                "stop revised run",
            )
            .await
            .unwrap();
        assert_eq!(stopped.run.status, TaskRunStatus::Incomplete);
        assert!(f.worker.set_context("released", json!(true)).is_ok());
    }
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
// The actual child and configuration remain available for acknowledgement fault injection without replacing runtime execution.
async fn fixture_mode(repause: bool, hosted: bool) -> Fixture {
    Box::pin(fixture_host_evidence(repause, hosted, None)).await
}

// Nested host-evidence tests retain the same message and child cursors before entering a real external wait.
async fn fixture_host_evidence(
    repause: bool,
    hosted: bool,
    source: Option<Arc<dyn HostRevisionSource>>,
) -> Fixture {
    let external = source.is_some();
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
    let worker = registry.get("worker").unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut extensions = AutonomyExtensions::builtins();
    extensions
        .register_validator(Arc::new(Validator {
            descriptor: AdapterDescriptor {
                id: "host.message".into(),
                contract_version: 1,
                tools: vec!["send_agent_message".into()],
                conditions: if external {
                    vec!["ready".into()]
                } else {
                    vec![]
                },
                requires_host_revision: external,
                ..Default::default()
            },
            calls: calls.clone(),
            hosted,
            external,
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
    if let Some(source) = source {
        extensions
            .register_validation_target("check", "ticket", source)
            .unwrap();
        extensions
            .register_condition(HostConditionBinding {
                id: "ready".into(),
                version: 1,
                shape: ExternalSignalShape::Boolean,
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
        config.clone(),
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
        worker,
        calls,
        store,
        config,
    }
}

// Child re-pauses and fixed host approval settle once before external delivery or host-controlled retirement.
#[tokio::test]
async fn host_evidence_nested_child_then_external_preserves_completed_history_and_custody() {
    for sqlite in super::continuation_test_support::backends() {
        for hosted in [false, true] {
            for action in ["signal", "unavailable", "cancel", "objective"] {
                let source = super::host_evidence_tests::Revision::new();
                let mut f =
                    Box::pin(fixture_host_evidence(true, hosted, Some(source.clone()))).await;
                f.store = super::continuation_test_support::store(
                    f.root.info().id,
                    "observation-message.v1",
                    *sqlite,
                )
                .await;
                f.runner = AutonomyRunner::try_new(
                    f.root.clone(),
                    f.config.clone(),
                    Default::default(),
                    f.store.clone(),
                    "observation-message.v1".into(),
                )
                .unwrap();
                let mut result = f.runner.run("objective", None).await.unwrap();
                let mut pauses = 0;
                while !matches!(
                    result.run.pending.as_ref().unwrap().kind,
                    TaskPendingKind::ExternalCondition
                ) {
                    pauses += 1;
                    assert!(
                        pauses <= 4,
                        "sqlite={sqlite} hosted={hosted} action={action} pending={:?} child_calls={} callbacks={}",
                        result.run.pending,
                        f.child.calls.load(Ordering::SeqCst),
                        f.calls.load(Ordering::SeqCst)
                    );
                    let request_id = result.run.pending.as_ref().unwrap().id.clone();
                    let input = if matches!(
                        result.run.pending.as_ref().unwrap().kind,
                        TaskPendingKind::Approval
                    ) {
                        TaskResumeInput::Approval {
                            request_id,
                            result: ai_agents_hitl::ApprovalResult::Approved,
                        }
                    } else {
                        TaskResumeInput::UserAnswer {
                            request_id,
                            answer: json!({"answered":true,"selected":["yes"]}),
                        }
                    };
                    result = f
                        .runner
                        .resume(&result.run.key.run_id, result.run.revision, input)
                        .await
                        .unwrap();
                }
                assert_eq!(f.child.calls.load(Ordering::SeqCst), 3);
                assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
                assert!(f.worker.set_context("blocked", json!(true)).is_err());
                let old = result.run.pending.as_ref().unwrap().id.clone();
                let done = match action {
                    "signal" => f
                        .runner
                        .resume(
                            &result.run.key.run_id,
                            result.run.revision,
                            TaskResumeInput::ExternalSignal(ExternalSignal {
                                request_id: old.clone(),
                                event_id: "settled-child".into(),
                                target_revision: Some("r1".into()),
                                payload: json!(true),
                            }),
                        )
                        .await
                        .unwrap(),
                    "unavailable" => {
                        source.change();
                        f.runner
                            .external_unavailable_paused(
                                &result.run.key.run_id,
                                result.run.revision,
                                &old,
                                "delivery lost",
                            )
                            .await
                            .unwrap()
                    }
                    "cancel" => {
                        source.change();
                        f.runner
                            .cancel_paused(&result.run.key.run_id, result.run.revision)
                            .await
                            .unwrap()
                    }
                    _ => {
                        source.change();
                        let revised = f
                            .runner
                            .modify_objective_paused(
                                &result.run.key.run_id,
                                result.run.revision,
                                "new objective",
                                "retire external wait",
                            )
                            .await
                            .unwrap();
                        assert!(
                            f.runner
                                .resume(
                                    &revised.run.key.run_id,
                                    revised.run.revision,
                                    TaskResumeInput::ExternalSignal(ExternalSignal {
                                        request_id: old,
                                        event_id: "obsolete".into(),
                                        target_revision: Some("r1".into()),
                                        payload: json!(true)
                                    })
                                )
                                .await
                                .is_err()
                        );
                        f.runner
                            .stop_paused(
                                &revised.run.key.run_id,
                                revised.run.revision,
                                "stop revised run",
                            )
                            .await
                            .unwrap()
                    }
                };
                assert_eq!(
                    done.run.status,
                    match action {
                        "signal" => TaskRunStatus::Completed,
                        "cancel" => TaskRunStatus::Cancelled,
                        _ => TaskRunStatus::Incomplete,
                    }
                );
                assert_eq!(f.child.calls.load(Ordering::SeqCst), 3);
                assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
                assert_eq!(
                    f.calls.load(Ordering::SeqCst),
                    if action == "signal" { 3 } else { 2 }
                );
                assert!(f.worker.set_context("released", json!(true)).is_ok());
            }
        }
    }
}

// Freshness refusal at a nested pending question must leave its actual child history safely cancellable.
#[tokio::test]
async fn host_evidence_nested_stale_target_cancels_without_answer_delivery() {
    for sqlite in super::continuation_test_support::backends() {
        let source = super::host_evidence_tests::Revision::new();
        let mut f = Box::pin(fixture_host_evidence(true, false, Some(source.clone()))).await;
        f.store = super::continuation_test_support::store(
            f.root.info().id,
            "observation-message.v1",
            *sqlite,
        )
        .await;
        f.runner = AutonomyRunner::try_new(
            f.root.clone(),
            f.config.clone(),
            Default::default(),
            f.store.clone(),
            "observation-message.v1".into(),
        )
        .unwrap();
        let first = f.runner.run("objective", None).await.unwrap();
        source.change();
        assert!(
            f.runner
                .resume(
                    &first.run.key.run_id,
                    first.run.revision,
                    TaskResumeInput::UserAnswer {
                        request_id: first.run.pending.as_ref().unwrap().id.clone(),
                        answer: json!({"answered":true,"selected":["yes"]})
                    }
                )
                .await
                .is_err()
        );
        assert_eq!(f.child.calls.load(Ordering::SeqCst), 1);
        let done = f
            .runner
            .cancel_paused(&first.run.key.run_id, first.run.revision)
            .await
            .unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Cancelled);
        assert_eq!(f.child.calls.load(Ordering::SeqCst), 1);
        assert!(f.worker.set_context("released", json!(true)).is_ok());
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

mod acknowledgement {
    use super::*;
    use ai_agents_core::{AgentError, Result};
    use async_trait::async_trait;

    #[derive(Clone, Copy)]
    enum Fault {
        RejectWrite,
        LoseAcknowledgement,
        HoldAcknowledgement,
    }

    #[derive(Clone, Copy)]
    enum Boundary {
        Admission,
        Repark,
    }

    // Select an exact journal boundary, not a write ordinal or the earlier public request consumption.
    struct AckStore {
        inner: Arc<ScopedTaskRunStore>,
        fault: Fault,
        boundary: Boundary,
        before_write: parking_lot::Mutex<Option<TaskRunSnapshot>>,
        armed: parking_lot::Mutex<Option<(String, String)>>,
        hits: AtomicUsize,
        entered: tokio::sync::Semaphore,
        release: tokio::sync::Semaphore,
    }

    impl AckStore {
        // Arm only after the initial pause so the original observation admission is never intercepted.
        fn arm(&self, run: &TaskRun) {
            *self.armed.lock() = Some((
                run.key.run_id.clone(),
                run.pending.as_ref().unwrap().id.clone(),
            ));
        }

        // Admission gates child execution; re-park acknowledgement gates publication of its new pending request.
        fn intercept(&self, run_id: &str, mutation: &TaskRunMutation) -> bool {
            let TaskRunMutation::Checkpoint {
                status: TaskRunStatus::Running,
                payload,
                release: false,
                ..
            } = mutation
            else {
                return false;
            };
            let mut armed = self.armed.lock();
            let Some((expected_run, request_id)) = armed.as_ref() else {
                return false;
            };
            if run_id != expected_run {
                return false;
            }
            let payload: TaskCheckpointPayload = serde_json::from_value(payload.clone()).unwrap();
            let driver = driver(&payload);
            if !payload.consumed_request_ids.contains(request_id)
                || driver.identity.key.run_id != run_id
                || driver.pending_batch != ["message"]
                || !driver.completed.is_empty()
            {
                return false;
            }
            let selected = match self.boundary {
                Boundary::Admission => {
                    payload.pending.is_none()
                        && driver.in_flight.as_deref() == Some("message")
                        && driver.parked.is_none()
                }
                Boundary::Repark => {
                    driver.in_flight.is_none()
                        && driver.parked.as_ref().is_some_and(|parked| {
                            parked.observation == "message"
                                && parked.group.is_some()
                                && parked.pending.id != *request_id
                                && payload
                                    .pending
                                    .as_ref()
                                    .is_some_and(|pending| pending.id == parked.pending.id)
                        })
                }
            };
            if !selected {
                return false;
            }
            armed.take();
            self.hits.fetch_add(1, Ordering::SeqCst);
            true
        }
    }

    #[async_trait]
    impl TaskRunStore for AckStore {
        // The wrapper leaves creation and ordinary conditional storage semantics unchanged.
        async fn create(&self, snapshot: &TaskRunSnapshot) -> Result<()> {
            self.inner.create(snapshot).await
        }

        // Reads expose the real committed state even when the caller lost acknowledgement.
        async fn load(&self, run_id: &str) -> Result<Option<TaskRunSnapshot>> {
            self.inner.load(run_id).await
        }

        // Listing never fabricates a safely published pause after an interrupted write.
        async fn list(&self) -> Result<Vec<TaskRunSummary>> {
            self.inner.list().await
        }

        // Deletion retains the scoped store's revision and ownership checks.
        async fn delete(&self, run_id: &str, revision: u64) -> Result<()> {
            self.inner.delete(run_id, revision).await
        }

        // Failure may precede commit or follow it; a held acknowledgement is released or dropped deterministically.
        async fn mutate(
            &self,
            run_id: &str,
            mutation: &TaskRunMutation,
        ) -> Result<TaskRunSnapshot> {
            if !self.intercept(run_id, mutation) {
                return self.inner.mutate(run_id, mutation).await;
            }
            *self.before_write.lock() = self.inner.load(run_id).await?;
            if matches!(self.fault, Fault::RejectWrite) {
                return Err(AgentError::Persistence(
                    "observation admission write rejected".into(),
                ));
            }
            let saved = self.inner.mutate(run_id, mutation).await?;
            if matches!(self.fault, Fault::LoseAcknowledgement) {
                return Err(AgentError::Persistence(
                    "observation admission acknowledgement lost".into(),
                ));
            }
            self.entered.add_permits(1);
            self.release.acquire().await.unwrap().forget();
            Ok(saved)
        }
    }

    // Existing admission tests keep their original single-question child.
    async fn fixture(fault: Fault) -> (Fixture, Arc<AckStore>) {
        Box::pin(fixture_at(fault, Boundary::Admission)).await
    }

    // Re-park uses the existing two-question child; only the selected acknowledgement is intercepted.
    async fn fixture_at(fault: Fault, boundary: Boundary) -> (Fixture, Arc<AckStore>) {
        Box::pin(fixture_on(fault, boundary, false)).await
    }

    // Both scoped backends exercise the same real runtimes and exact journal acknowledgement selector.
    async fn fixture_on(
        fault: Fault,
        boundary: Boundary,
        sqlite: bool,
    ) -> (Fixture, Arc<AckStore>) {
        let mut f = Box::pin(super::fixture_repause(matches!(boundary, Boundary::Repark))).await;
        if sqlite {
            #[cfg(feature = "sqlite")]
            {
                f.store = Arc::new(
                    ScopedTaskRunStore::new(
                        Arc::new(ai_agents_storage::SqliteStorage::in_memory().await.unwrap()),
                        f.root.info().id,
                        None,
                        "observation-message.v1".into(),
                    )
                    .unwrap(),
                );
            }
            #[cfg(not(feature = "sqlite"))]
            panic!("SQLite fixture requires its feature");
        }
        let store = Arc::new(AckStore {
            inner: f.store.clone(),
            fault,
            boundary,
            before_write: parking_lot::Mutex::new(None),
            armed: parking_lot::Mutex::new(None),
            hits: AtomicUsize::new(0),
            entered: tokio::sync::Semaphore::new(0),
            release: tokio::sync::Semaphore::new(0),
        });
        f.runner = AutonomyRunner::try_new(
            f.root.clone(),
            f.config.clone(),
            Default::default(),
            store.clone(),
            "observation-message.v1".into(),
        )
        .unwrap();
        (f, store)
    }

    // The selected fixture has one exact observation driver and one logical reservation for its message.
    fn driver(payload: &TaskCheckpointPayload) -> ValidationDriverState {
        serde_json::from_value(
            payload
                .adapters
                .iter()
                .find(|adapter| adapter.id == "check")
                .unwrap()
                .state["_task_validation_driver"]
                .clone(),
        )
        .unwrap()
    }

    // Construct only the implementation-bound answer; a coordinating request ID is not a child authority token.
    fn answer(run: &TaskRun) -> TaskResumeInput {
        TaskResumeInput::UserAnswer {
            request_id: run.pending.as_ref().unwrap().id.clone(),
            answer: json!({"answered":true,"selected":["yes"]}),
        }
    }

    // A semaphore observes the accepted write without timing the child; the timeout only bounds a broken test.
    async fn await_acknowledgement<F>(resume: &mut std::pin::Pin<Box<F>>, store: &AckStore)
    where
        F: std::future::Future<Output = Result<TaskRunResult>>,
    {
        tokio::select! {
            result = resume => panic!("resume returned before acknowledgement gate: {result:?}"),
            permit = tokio::time::timeout(std::time::Duration::from_secs(5), store.entered.acquire()) => {
                permit.unwrap().unwrap().forget();
            }
        }
    }

    // Durable admission does not prove child execution, and neither failed path may release root or child protection.
    async fn assert_undelivered(f: &Fixture, original: &TaskRunSnapshot, committed: bool) {
        let saved = f.store.load(&original.key.run_id).await.unwrap().unwrap();
        let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload).unwrap();
        let before: TaskCheckpointPayload =
            serde_json::from_value(original.payload.clone()).unwrap();
        let before_driver = driver(&before);
        let current_driver = driver(&payload);
        assert_eq!(saved.status, TaskRunStatus::Running);
        assert!(saved.owner_token.is_some());
        assert!(payload.pending.is_none());
        assert!(
            payload
                .consumed_request_ids
                .contains(&before.pending.unwrap().id)
        );
        assert_eq!(current_driver.identity, before_driver.identity);
        assert_eq!(current_driver.requests, before_driver.requests);
        assert_eq!(
            serde_json::to_value(&payload.counters).unwrap(),
            serde_json::to_value(&before.counters).unwrap()
        );
        assert_eq!(
            current_driver.in_flight.as_deref(),
            committed.then_some("message")
        );
        assert_eq!(current_driver.parked.is_some(), !committed);
        assert!(current_driver.completed.is_empty());
        let reservation_id = format!(
            "validation:{}",
            canonical_identity(&json!({"check":"check","attempt":current_driver.identity.attempt,"observation":"message"})).unwrap()
        );
        let reservation = payload
            .reservations
            .iter()
            .find(|r| r.id == reservation_id)
            .unwrap();
        assert_eq!(
            reservation.state,
            if committed {
                TaskEffectState::Dispatched
            } else {
                TaskEffectState::Suspended
            }
        );
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.child.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.calls.load(Ordering::SeqCst), 1);
        assert!(f.root.set_context("unrelated", json!(true)).is_err());
        assert!(f.worker.set_context("unrelated", json!(true)).is_err());
        assert_eq!(
            serde_json::to_value(f.root.save_state_full().await.unwrap().memory.messages).unwrap(),
            serde_json::to_value(&before.runtime.snapshot.memory.messages).unwrap()
        );
        assert_eq!(
            serde_json::to_value(f.worker.save_state_full().await.unwrap().memory.messages)
                .unwrap(),
            serde_json::to_value(&before.children[0].runtime.snapshot.memory.messages).unwrap()
        );
    }

    // Neither the stale public pause nor the newer stored admission can authorize automatic replay after abandonment.
    async fn assert_replay_rejected(f: &Fixture, paused: &TaskRunResult) {
        let run_id = &paused.run.key.run_id;
        let saved = f.store.load(run_id).await.unwrap().unwrap();
        for revision in [paused.run.revision, saved.revision] {
            assert!(
                f.runner
                    .resume(run_id, revision, answer(&paused.run))
                    .await
                    .is_err()
            );
        }
        assert_eq!(
            f.store.load(run_id).await.unwrap().unwrap().revision,
            saved.revision
        );
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.child.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.calls.load(Ordering::SeqCst), 1);
    }

    // Rejected writes and committed-but-unacknowledged writes both stop before child response execution.
    #[tokio::test]
    async fn observation_message_failed_resume_acknowledgement_preserves_child_custody() {
        for fault in [Fault::RejectWrite, Fault::LoseAcknowledgement] {
            let (f, store) = Box::pin(fixture(fault)).await;
            let paused = f.runner.run("objective", None).await.unwrap();
            let original = f.store.load(&paused.run.key.run_id).await.unwrap().unwrap();
            store.arm(&paused.run);
            let error = f
                .runner
                .resume(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    answer(&paused.run),
                )
                .await
                .unwrap_err();
            let expected = if matches!(fault, Fault::RejectWrite) {
                "observation admission write rejected"
            } else {
                // Preserve the original ambiguous write error instead of treating it as a clean clock interruption.
                "observation admission acknowledgement lost"
            };
            assert!(error.to_string().contains(expected), "{error}");
            assert_eq!(store.hits.load(Ordering::SeqCst), 1);
            assert_undelivered(&f, &original, matches!(fault, Fault::LoseAcknowledgement)).await;
            assert_replay_rejected(&f, &paused).await;
        }
    }

    // Dropping after commit retains the dispatched journal and both live owners without delivering the child answer.
    #[tokio::test]
    async fn observation_message_dropped_resume_acknowledgement_preserves_child_custody() {
        let (f, store) = Box::pin(fixture(Fault::HoldAcknowledgement)).await;
        let paused = f.runner.run("objective", None).await.unwrap();
        let original = f.store.load(&paused.run.key.run_id).await.unwrap().unwrap();
        store.arm(&paused.run);
        let mut resume = Box::pin(f.runner.resume(
            &paused.run.key.run_id,
            paused.run.revision,
            answer(&paused.run),
        ));
        await_acknowledgement(&mut resume, &store).await;
        drop(resume);
        assert_eq!(store.hits.load(Ordering::SeqCst), 1);
        assert_undelivered(&f, &original, true).await;
        assert_replay_rejected(&f, &paused).await;
    }

    // Re-park preserves completed child work but cannot publish a safe pause or accept its next answer without acknowledgement.
    async fn assert_repark_protected(f: &Fixture, store: &AckStore, committed: bool) {
        let before = store.before_write.lock().clone().unwrap();
        let saved = f.store.load(&before.key.run_id).await.unwrap().unwrap();
        let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload.clone()).unwrap();
        let previous: TaskCheckpointPayload =
            serde_json::from_value(before.payload.clone()).unwrap();
        let current_driver = driver(&payload);
        assert_eq!(saved.status, TaskRunStatus::Running);
        assert!(saved.owner_token.is_some());
        assert_eq!(current_driver.identity, driver(&previous).identity);
        assert_eq!(current_driver.requests, driver(&previous).requests);
        assert!(current_driver.completed.is_empty());
        assert_eq!(current_driver.parked.is_some(), committed);
        assert_eq!(
            current_driver.in_flight.as_deref(),
            (!committed).then_some("message")
        );
        if !committed {
            assert_eq!(saved.revision, before.revision);
            assert_eq!(saved.payload, before.payload);
        } else {
            assert_eq!(saved.revision, before.revision + 1);
            assert_eq!(
                payload.pending.as_ref().unwrap().id,
                current_driver.parked.as_ref().unwrap().pending.id
            );
        }
        assert_eq!(
            serde_json::to_value(&payload.counters).unwrap(),
            serde_json::to_value(&previous.counters).unwrap()
        );
        assert_eq!(payload.counters.turns, 1);
        assert_eq!(payload.counters.llm_attempts, 3);
        assert_eq!(payload.counters.tool_attempts, 2);
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.child.calls.load(Ordering::SeqCst), 2);
        assert_eq!(f.calls.load(Ordering::SeqCst), 1);
        assert!(f.root.set_context("unrelated", json!(true)).is_err());
        assert!(f.worker.set_context("unrelated", json!(true)).is_err());
        assert_eq!(
            serde_json::to_value(f.worker.save_state_full().await.unwrap().memory.messages)
                .unwrap(),
            serde_json::to_value(&payload.children[0].runtime.snapshot.memory.messages).unwrap()
        );
        if let Some(pending) = payload.pending {
            assert!(
                f.runner
                    .resume(
                        &saved.key.run_id,
                        saved.revision,
                        TaskResumeInput::UserAnswer {
                            request_id: pending.id,
                            answer: json!({"answered":true,"selected":["yes"]}),
                        }
                    )
                    .await
                    .is_err()
            );
            assert_eq!(
                f.store
                    .load(&saved.key.run_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .revision,
                saved.revision
            );
        }
    }

    // Both sides of commit retain completed first-question work instead of replaying it or exposing the second answer.
    #[tokio::test]
    async fn observation_message_failed_repark_acknowledgement_preserves_completed_child_work() {
        for fault in [Fault::RejectWrite, Fault::LoseAcknowledgement] {
            let (f, store) = Box::pin(fixture_at(fault, Boundary::Repark)).await;
            let paused = f.runner.run("objective", None).await.unwrap();
            store.arm(&paused.run);
            assert!(
                f.runner
                    .resume(
                        &paused.run.key.run_id,
                        paused.run.revision,
                        answer(&paused.run)
                    )
                    .await
                    .is_err()
            );
            assert_eq!(store.hits.load(Ordering::SeqCst), 1);
            assert_repark_protected(&f, &store, matches!(fault, Fault::LoseAcknowledgement)).await;
        }
    }

    // A durable new pending ID remains non-resumable when its acknowledgement is dropped before pause publication.
    #[tokio::test]
    async fn observation_message_dropped_repark_acknowledgement_preserves_completed_child_work() {
        let (f, store) = Box::pin(fixture_at(Fault::HoldAcknowledgement, Boundary::Repark)).await;
        let paused = f.runner.run("objective", None).await.unwrap();
        store.arm(&paused.run);
        let mut resume = Box::pin(f.runner.resume(
            &paused.run.key.run_id,
            paused.run.revision,
            answer(&paused.run),
        ));
        await_acknowledgement(&mut resume, &store).await;
        drop(resume);
        assert_eq!(store.hits.load(Ordering::SeqCst), 1);
        assert_repark_protected(&f, &store, true).await;
    }

    // Successful re-park acknowledgement publishes the new coordinating request and resumes only the second child question.
    #[tokio::test]
    async fn observation_message_repark_waits_for_acknowledgement_before_publishing_pause() {
        let (f, store) = Box::pin(fixture_at(Fault::HoldAcknowledgement, Boundary::Repark)).await;
        let first = f.runner.run("objective", None).await.unwrap();
        store.arm(&first.run);
        let mut resume = Box::pin(f.runner.resume(
            &first.run.key.run_id,
            first.run.revision,
            answer(&first.run),
        ));
        await_acknowledgement(&mut resume, &store).await;
        assert_eq!(f.child.calls.load(Ordering::SeqCst), 2);
        assert_eq!(f.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            f.store
                .load(&first.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskRunStatus::Running
        );
        store.release.add_permits(1);
        let second = resume.await.unwrap();
        assert_eq!(second.run.status, TaskRunStatus::Paused);
        assert_ne!(
            second.run.pending.as_ref().unwrap().id,
            first.run.pending.as_ref().unwrap().id
        );
        let done = f
            .runner
            .resume(
                &second.run.key.run_id,
                second.run.revision,
                answer(&second.run),
            )
            .await
            .unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Completed);
        assert_eq!(store.hits.load(Ordering::SeqCst), 1);
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.child.calls.load(Ordering::SeqCst), 3);
        assert_eq!(f.calls.load(Ordering::SeqCst), 2);
        assert_eq!(done.run.counters.tool_attempts, 3);
        assert!(f.root.set_context("released", json!(true)).is_ok());
        assert!(f.worker.set_context("released", json!(true)).is_ok());
    }

    // SQLite must preserve the same non-execution or completed-child custody on both sides of each journal commit.
    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn observation_message_sqlite_admission_and_repark_acknowledgement_protection() {
        for boundary in [Boundary::Admission, Boundary::Repark] {
            for fault in [
                Fault::RejectWrite,
                Fault::LoseAcknowledgement,
                Fault::HoldAcknowledgement,
            ] {
                let (f, store) = Box::pin(fixture_on(fault, boundary, true)).await;
                let paused = f.runner.run("objective", None).await.unwrap();
                let original = f.store.load(&paused.run.key.run_id).await.unwrap().unwrap();
                store.arm(&paused.run);
                let mut resume = Box::pin(f.runner.resume(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    answer(&paused.run),
                ));
                if matches!(fault, Fault::HoldAcknowledgement) {
                    await_acknowledgement(&mut resume, &store).await;
                    drop(resume);
                } else {
                    assert!(resume.await.is_err());
                }
                assert_eq!(store.hits.load(Ordering::SeqCst), 1);
                let committed = !matches!(fault, Fault::RejectWrite);
                if matches!(boundary, Boundary::Admission) {
                    assert_undelivered(&f, &original, committed).await;
                    assert_replay_rejected(&f, &paused).await;
                } else {
                    assert_repark_protected(&f, &store, committed).await;
                }
            }
        }
    }

    // Releasing the exact acknowledgement proves the injection gate is reachable and permits one normal child continuation.
    #[tokio::test]
    async fn observation_message_child_answer_waits_for_resume_acknowledgement() {
        let (f, store) = Box::pin(fixture(Fault::HoldAcknowledgement)).await;
        let paused = f.runner.run("objective", None).await.unwrap();
        let original = f.store.load(&paused.run.key.run_id).await.unwrap().unwrap();
        store.arm(&paused.run);
        let mut resume = Box::pin(f.runner.resume(
            &paused.run.key.run_id,
            paused.run.revision,
            answer(&paused.run),
        ));
        await_acknowledgement(&mut resume, &store).await;
        assert_undelivered(&f, &original, true).await;
        store.release.add_permits(1);
        let done = resume.await.unwrap();
        assert_eq!(store.hits.load(Ordering::SeqCst), 1);
        assert_eq!(done.run.status, TaskRunStatus::Completed);
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.child.calls.load(Ordering::SeqCst), 2);
        assert_eq!(f.calls.load(Ordering::SeqCst), 2);
        assert_eq!(done.run.counters.turns, 1);
        assert_eq!(done.run.counters.llm_attempts, 3);
        assert_eq!(done.run.counters.tool_attempts, 2);
        assert!(f.root.set_context("released", json!(true)).is_ok());
        assert!(f.worker.set_context("released", json!(true)).is_ok());
    }
}

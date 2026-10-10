use super::*;
use crate::{Agent, AgentBuilder, RuntimeAgent};
use ai_agents_core::{Tool, ToolCall, ToolExecutionContext, ToolResult, ToolSafetyMetadata};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Probe {
    id: &'static str,
    calls: Arc<AtomicUsize>,
    fail: bool,
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
        "Count actual invocations"
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
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail {
            ToolResult::error("original failed")
        } else {
            ToolResult::ok(args.to_string())
        }
    }
}
struct NoApproval;
#[async_trait]
impl ai_agents_hitl::ApprovalHandler for NoApproval {
    async fn request_approval(
        &self,
        _: ai_agents_hitl::ApprovalRequest,
    ) -> ai_agents_hitl::ApprovalResult {
        panic!("task approval must park")
    }
}
struct Responses(Arc<AtomicUsize>);
#[async_trait]
impl ai_agents_hooks::AgentHooks for Responses {
    async fn on_response(&self, _: &ai_agents_core::AgentResponse) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
struct Fixture {
    agent: Arc<RuntimeAgent>,
    runner: AutonomyRunner,
    store: Arc<ScopedTaskRunStore>,
    first: Arc<AtomicUsize>,
    second: Arc<AtomicUsize>,
    responses: Arc<AtomicUsize>,
    main: Arc<super::runner_tests::RecordingProvider>,
    router: Arc<super::runner_tests::RecordingProvider>,
}

// A real scripted skill preserves one completed prompt and tool before its pending approval/question.
async fn skill_fixture(question: bool) -> Fixture {
    let first = Arc::new(AtomicUsize::new(0));
    let second = Arc::new(AtomicUsize::new(0));
    let responses = Arc::new(AtomicUsize::new(0));
    let main = super::runner_tests::RecordingProvider::new(&["prefix", "done"]);
    let router = super::runner_tests::RecordingProvider::new(&["script"]);
    let mut registry = ai_agents_llm::LLMRegistry::new();
    registry.register("default", main.clone());
    registry.register("router", router.clone());
    let skill = ai_agents_skills::SkillDefinition {
        autonomy: None,
        id: "script".into(),
        description: "work".into(),
        trigger: "when asked".into(),
        reasoning: None,
        reflection: None,
        disambiguation: None,
        steps: vec![
            ai_agents_skills::SkillStep::Prompt {
                prompt: "INITIAL_SCRIPT_PROMPT".into(),
                llm: None,
            },
            ai_agents_skills::SkillStep::Tool {
                tool: "first".into(),
                args: Some(json!({"value":"{{ steps[0].result }}"})),
                output_as: None,
            },
            ai_agents_skills::SkillStep::Tool {
                tool: if question { "ask_user" } else { "second" }.into(),
                args: Some(if question {
                    json!({"question":"Select","options":["yes"],"allow_other":false})
                } else {
                    json!({"value":"{{ steps[0].result }}"})
                }),
                output_as: None,
            },
            ai_agents_skills::SkillStep::Prompt {
                prompt: "{{ steps[0].result }} {{ steps[1].result }} {{ steps[2].result }}".into(),
                llm: None,
            },
        ],
    };
    let yaml = "name: Script\nsystem_prompt: test\nllm:\n  default: default\n  router: router\ntools: [first, second, ask_user]\nhitl:\n  tools:\n    second:\n      require_approval: true\n";
    let mut spec: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
    spec["skills"] = serde_yaml::to_value(vec![skill]).unwrap();
    let yaml = serde_yaml::to_string(&spec).unwrap();
    let builder = AgentBuilder::from_yaml(&yaml)
        .unwrap()
        .llm_registry(registry)
        .auto_configure_features()
        .unwrap()
        .approval_handler(Arc::new(NoApproval))
        .hooks(Arc::new(Responses(responses.clone())))
        .tool(Arc::new(Probe {
            id: "first",
            calls: first.clone(),
            fail: false,
        }))
        .tool(Arc::new(Probe {
            id: "second",
            calls: second.clone(),
            fail: false,
        }))
        .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
            Arc::new(parking_lot::RwLock::new(None)),
        )));
    let agent = Arc::new(builder.build().unwrap());
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "locations.v1".into()).unwrap(),
    );
    let runner = runner(agent.clone(), store.clone());
    Fixture {
        agent,
        runner,
        store,
        first,
        second,
        responses,
        main,
        router,
    }
}

// Root fixtures use the explicit development API and preserve the same selected interaction policy.
fn runner(agent: Arc<RuntimeAgent>, store: Arc<ScopedTaskRunStore>) -> AutonomyRunner {
    let mut config = super::runner_tests::config(8);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    AutonomyRunner::try_new(
        agent,
        config,
        Default::default(),
        store,
        "locations.v1".into(),
    )
    .unwrap()
}

// Only the currently coordinating response can resume a non-model location.
async fn approve(f: &Fixture, paused: &TaskRun) -> TaskRunResult {
    f.runner
        .resume(
            &paused.key.run_id,
            paused.revision,
            TaskResumeInput::Approval {
                request_id: paused.pending.as_ref().unwrap().id.clone(),
                result: ai_agents_hitl::ApprovalResult::Approved,
            },
        )
        .await
        .unwrap()
}

// Script routing, completed prompt/template work and effects are not replayed after either interaction kind.
#[tokio::test]
async fn skill_location_resumes_exact_step_and_finalizes_once() {
    for question in [false, true] {
        let f = Box::pin(skill_fixture(question)).await;
        let first = f.runner.run("objective", None).await.unwrap();
        assert_eq!(
            first.run.status,
            TaskRunStatus::Paused,
            "{:?}",
            first.run.stop_reason
        );
        assert_eq!(f.first.load(Ordering::SeqCst), 1);
        assert_eq!(f.second.load(Ordering::SeqCst), 0);
        assert_eq!(f.responses.load(Ordering::SeqCst), 0);
        let done = if question {
            f.runner
                .resume(
                    &first.run.key.run_id,
                    first.run.revision,
                    TaskResumeInput::UserAnswer {
                        request_id: first.run.pending.as_ref().unwrap().id.clone(),
                        answer: json!({"answered":true,"selected":["yes"]}),
                    },
                )
                .await
                .unwrap()
        } else {
            approve(&f, &first.run).await
        };
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(f.first.load(Ordering::SeqCst), 1);
        assert_eq!(f.second.load(Ordering::SeqCst), usize::from(!question));
        assert_eq!(f.responses.load(Ordering::SeqCst), 1);
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 2);
        assert_eq!(f.router.calls.load(Ordering::SeqCst), 1);
        let p: TaskCheckpointPayload = serde_json::from_value(
            f.store
                .load(&done.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .payload,
        )
        .unwrap();
        assert_eq!(
            p.runtime
                .snapshot
                .memory
                .messages
                .iter()
                .filter(|message| message.role == ai_agents_core::Role::User)
                .count(),
            1
        );
        if !question {
            let record = &p
                .evidence
                .tool_calls
                .iter()
                .find(|capture| capture.record.canonical_id == "second")
                .unwrap()
                .record;
            assert_eq!(record.executed_arguments["value"], "prefix");
        }
    }
}

// A rejected script step records one terminal response and does not execute later prompts or replay earlier work.
#[tokio::test]
async fn skill_location_rejection_finalizes_without_replaying_prefix() {
    let f = Box::pin(skill_fixture(false)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    let stopped = f
        .runner
        .resume(
            &first.run.key.run_id,
            first.run.revision,
            TaskResumeInput::Approval {
                request_id: first.run.pending.as_ref().unwrap().id.clone(),
                result: ai_agents_hitl::ApprovalResult::Rejected {
                    reason: Some("stop".into()),
                },
            },
        )
        .await
        .unwrap();
    assert_eq!(stopped.run.status, TaskRunStatus::Failed);
    assert_eq!(f.first.load(Ordering::SeqCst), 1);
    assert_eq!(f.second.load(Ordering::SeqCst), 0);
    assert_eq!(f.responses.load(Ordering::SeqCst), 1);
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
    assert!(f.agent.chat("ordinary after stop").await.is_ok());
}

// Cancelled script work is known uninvoked and may release the retained runtime without another provider call.
#[tokio::test]
async fn skill_location_paused_cancellation_does_not_invoke_pending_work() {
    let f = Box::pin(skill_fixture(false)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    let stopped = f
        .runner
        .cancel_paused(&first.run.key.run_id, first.run.revision)
        .await
        .unwrap();
    assert_eq!(stopped.run.status, TaskRunStatus::Cancelled);
    assert_eq!(f.first.load(Ordering::SeqCst), 1);
    assert_eq!(f.second.load(Ordering::SeqCst), 0);
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
}

// A pending fallback resumes its own request instead of repeating the already failed original tool.
#[tokio::test]
async fn fallback_location_resumes_pending_hop_without_repeating_original() {
    let first = Arc::new(AtomicUsize::new(0));
    let second = Arc::new(AtomicUsize::new(0));
    let call = ToolCall {
        id: "original".into(),
        name: "first".into(),
        arguments: json!({"value":"original args"}),
    };
    let marker = ai_agents_core::encode_native_tool_call_markers(&[call], None).unwrap();
    let main = super::runner_tests::RecordingProvider::new(&[&marker, "done"]);
    let yaml = "name: Fallback\nsystem_prompt: test\ntools: [first, second]\nhitl:\n  tools:\n    second:\n      require_approval: true\nerror_recovery:\n  tools:\n    first:\n      max_retries: 0\n      on_failure:\n        action: fallback\n        fallback_tool: second\n";
    let agent = Arc::new(
        AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(main.clone())
            .auto_configure_features()
            .unwrap()
            .approval_handler(Arc::new(NoApproval))
            .tool(Arc::new(Probe {
                id: "first",
                calls: first.clone(),
                fail: true,
            }))
            .tool(Arc::new(Probe {
                id: "second",
                calls: second.clone(),
                fail: false,
            }))
            .build()
            .unwrap(),
    );
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "locations.v1".into()).unwrap(),
    );
    let runner = runner(agent, store);
    let paused = runner.run("objective", None).await.unwrap();
    assert_eq!(
        paused.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        paused.run.stop_reason
    );
    let done = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::Approval {
                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                result: ai_agents_hitl::ApprovalResult::Approved,
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
    assert_eq!(first.load(Ordering::SeqCst), 1);
    assert_eq!(second.load(Ordering::SeqCst), 1);
    assert_eq!(done.run.counters.tool_attempts, 2);
}

// Exit actions precede one state commit and resumed entry work does not repeat completed exit effects.
#[tokio::test]
async fn state_location_exit_and_entry_resume_preserve_transition_position() {
    for entering in [false, true] {
        let first = Arc::new(AtomicUsize::new(0));
        let second = Arc::new(AtomicUsize::new(0));
        let responses = Arc::new(AtomicUsize::new(0));
        let yaml = format!(
            "name: StateActions\nsystem_prompt: test\ntools: [first, second]\nhitl:\n  tools:\n    second:\n      require_approval: true\nstates:\n  initial: active\n  states:\n    active:\n      on_exit:\n        - tool: first\n{}      transitions:\n        - to: ready\n          guard: 'true'\n    ready:\n      regenerate_on_enter: false\n{}",
            if entering {
                ""
            } else {
                "        - tool: second\n"
            },
            if entering {
                "      on_enter:\n        - tool: second\n"
            } else {
                ""
            }
        );
        let main = super::runner_tests::RecordingProvider::new(&["done", "0", "done"]);
        let agent = Arc::new(
            AgentBuilder::from_yaml(&yaml)
                .unwrap()
                .llm(main)
                .auto_configure_features()
                .unwrap()
                .approval_handler(Arc::new(NoApproval))
                .hooks(Arc::new(Responses(responses.clone())))
                .tool(Arc::new(Probe {
                    id: "first",
                    calls: first.clone(),
                    fail: false,
                }))
                .tool(Arc::new(Probe {
                    id: "second",
                    calls: second.clone(),
                    fail: false,
                }))
                .build()
                .unwrap(),
        );
        let store = Arc::new(
            ScopedTaskRunStore::in_memory(agent.info().id, None, "locations.v1".into()).unwrap(),
        );
        let runner = runner(agent.clone(), store);
        let paused = runner.run("objective", None).await.unwrap();
        assert_eq!(
            paused.run.status,
            TaskRunStatus::Paused,
            "{:?}",
            paused.run.stop_reason
        );
        assert_eq!(
            agent.current_state().as_deref(),
            Some(if entering { "ready" } else { "active" })
        );
        let done = runner
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::Approval {
                    request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                    result: ai_agents_hitl::ApprovalResult::Approved,
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
        assert_eq!(agent.current_state().as_deref(), Some("ready"));
        assert_eq!(agent.state_history().len(), 1);
        assert_eq!(first.load(Ordering::SeqCst), 1);
        assert_eq!(second.load(Ordering::SeqCst), 1);
        assert_eq!(responses.load(Ordering::SeqCst), 1);
    }
}

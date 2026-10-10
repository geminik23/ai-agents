use super::*;
use crate::spawner::{AgentRegistry, AgentSpawner, SpawnedAgent};
use crate::{Agent, RuntimeAgent};
use ai_agents_core::{Tool, ToolCall, ToolExecutionContext, ToolResult, ToolSafetyMetadata};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct CounterTool {
    id: &'static str,
    calls: Arc<AtomicUsize>,
}
#[async_trait]
impl Tool for CounterTool {
    fn id(&self) -> &str {
        self.id
    }
    fn name(&self) -> &str {
        self.id
    }
    fn description(&self) -> &str {
        "Count actual shared invocations"
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
        ToolResult::ok("executed")
    }
}

struct NoApproval;
#[async_trait]
impl ai_agents_hitl::ApprovalHandler for NoApproval {
    async fn request_approval(
        &self,
        _: ai_agents_hitl::ApprovalRequest,
    ) -> ai_agents_hitl::ApprovalResult {
        panic!("group pause polled an ordinary approval handler")
    }
}

type GroupFixture = (
    Arc<RuntimeAgent>,
    Arc<RuntimeAgent>,
    AutonomyRunner,
    Arc<ScopedTaskRunStore>,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
    Arc<super::runner_tests::RecordingProvider>,
);

// A production state dispatch owns a real registered child and a signed batch whose later result may already be complete.
async fn fixture(question: bool) -> GroupFixture {
    let first = Arc::new(AtomicUsize::new(0));
    let second = Arc::new(AtomicUsize::new(0));
    let pending = if question {
        ToolCall {
            id: "pending".into(),
            name: "ask_user".into(),
            arguments: json!({"question":"Select", "options":["yes"], "allow_other":false}),
        }
    } else {
        ToolCall {
            id: "pending".into(),
            name: "first".into(),
            arguments: json!({}),
        }
    };
    let calls = vec![
        pending,
        ToolCall {
            id: "completed".into(),
            name: "second".into(),
            arguments: json!({}),
        },
    ];
    let native = ai_agents_core::NativeProviderState::new(
        "child-exchange",
        "fixture",
        "native-tools",
        ai_agents_core::NativeProviderTarget::new("https://fixture.invalid", "fixture-model")
            .unwrap(),
        json!({"signature":"exact-child-state"}),
        calls
            .iter()
            .enumerate()
            .map(|(index, call)| ai_agents_core::NativeCallBinding::new(&call.id, index).unwrap())
            .collect(),
    )
    .unwrap();
    let marker = ai_agents_core::encode_native_tool_call_markers(&calls, Some(&native)).unwrap();
    let provider = super::runner_tests::RecordingProvider::new(&[&marker, "done"]);
    let yaml = if question {
        "name: Worker\nsystem_prompt: child\ntools: [ask_user, second]\n"
    } else {
        "name: Worker\nsystem_prompt: child\ntools: [first, second]\nhitl:\n  tools:\n    first:\n      require_approval: true\n"
    };
    let builder = crate::AgentBuilder::from_yaml(yaml)
        .unwrap()
        .llm(provider.clone())
        .tool(Arc::new(CounterTool {
            id: "second",
            calls: second.clone(),
        }))
        .auto_configure_features()
        .unwrap()
        .approval_handler(Arc::new(NoApproval));
    let builder = if question {
        builder.tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
            Arc::new(parking_lot::RwLock::new(None)),
        )))
    } else {
        builder.tool(Arc::new(CounterTool {
            id: "first",
            calls: first.clone(),
        }))
    };
    let child = builder.build().unwrap();
    let registry = Arc::new(AgentRegistry::new());
    registry
        .register(SpawnedAgent::from_runtime(
            "worker".into(),
            child,
            crate::spec::AgentSpec::from_yaml_strict(yaml).unwrap(),
        ))
        .await
        .unwrap();
    let child = registry.get("worker").unwrap();
    let parent = Arc::new(crate::AgentBuilder::from_yaml("name: GroupParent\nsystem_prompt: parent\nstates:\n  initial: active\n  states:\n    active:\n      delegate: worker\n").unwrap()
        .llm(super::runner_tests::RecordingProvider::new(&["unused parent"])).auto_configure_features().unwrap().build().unwrap()
        .with_spawner_handles(Arc::new(AgentSpawner::new()), registry));
    let store =
        Arc::new(ScopedTaskRunStore::in_memory(parent.info().id, None, "group.v1".into()).unwrap());
    let mut config = super::runner_tests::config(2);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    let runner = AutonomyRunner::try_new(
        parent.clone(),
        config,
        Default::default(),
        store.clone(),
        "group.v1".into(),
    )
    .unwrap();
    (parent, child, runner, store, first, second, provider)
}

// Parent and child remain reserved across the pause, then resume the child batch without rerouting either input.
#[tokio::test]
async fn delegated_approval_resumes_child_and_finalizes_parent_once() {
    let (parent, child, runner, store, first, second, provider) = fixture(false).await;
    let paused = runner.run("objective", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    assert_eq!(first.load(Ordering::SeqCst), 0);
    assert_eq!(second.load(Ordering::SeqCst), 1);
    assert!(parent.chat("foreign").await.is_err());
    assert!(child.chat("foreign").await.is_err());
    let snapshot = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
    assert!(payload.children[0].pending.is_some());
    assert!(snapshot.owner_token.is_none());
    let finished = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::Approval {
                request_id: paused.run.pending.unwrap().id,
                result: ai_agents_hitl::ApprovalResult::Approved,
            },
        )
        .await
        .unwrap();
    assert_eq!(finished.run.status, TaskRunStatus::Completed);
    assert_eq!(finished.run.counters.turns, 1);
    assert_eq!(finished.run.counters.llm_attempts, 2);
    assert_eq!(first.load(Ordering::SeqCst), 1);
    assert_eq!(second.load(Ordering::SeqCst), 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    let payload: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&finished.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert!(
        payload
            .children
            .iter()
            .all(|child| child.pending.is_none() && child.result.is_some())
    );
    assert_eq!(
        payload
            .runtime
            .snapshot
            .memory
            .messages
            .iter()
            .filter(|message| message.role == ai_agents_core::Role::User)
            .count(),
        1
    );
    assert_eq!(
        payload
            .runtime
            .snapshot
            .memory
            .messages
            .iter()
            .filter(|message| message.role == ai_agents_core::Role::Assistant)
            .count(),
        1
    );
    assert!(child.chat("ordinary").await.is_ok());
}

// A root question response binds to the leaf child request, not to another run or an approval receipt.
#[tokio::test]
async fn delegated_question_resumes_the_exact_child_action() {
    let (_, _, runner, _, _, second, provider) = fixture(true).await;
    let paused = runner.run("objective", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    let pending = paused.run.pending.unwrap();
    assert!(matches!(pending.kind, TaskPendingKind::UserQuestion));
    assert!(
        runner
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::Approval {
                    request_id: pending.id.clone(),
                    result: ai_agents_hitl::ApprovalResult::Approved
                }
            )
            .await
            .is_err()
    );
    let finished = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: pending.id,
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await
        .unwrap();
    assert_eq!(finished.run.status, TaskRunStatus::Completed);
    assert_eq!(second.load(Ordering::SeqCst), 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
}

// Cancellation acknowledges every parked child and root identity before either runtime is released.
#[tokio::test]
async fn delegated_pause_cancellation_never_invokes_pending_child() {
    let (parent, child, runner, store, first, second, provider) = fixture(false).await;
    let paused = runner.run("objective", None).await.unwrap();
    let cancelled = runner
        .cancel_paused(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap();
    assert_eq!(cancelled.run.status, TaskRunStatus::Cancelled);
    assert_eq!(first.load(Ordering::SeqCst), 0);
    assert_eq!(second.load(Ordering::SeqCst), 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    let payload: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&cancelled.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert!(payload.pending.is_none());
    assert!(
        payload
            .children
            .iter()
            .all(|child| child.pending.is_none() && child.result.is_some())
    );
    assert!(parent.chat("ordinary").await.is_ok());
    assert!(child.chat("ordinary").await.is_ok());
}

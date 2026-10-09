use super::*;
use ai_agents_core::{
    FinishReason, LLMResponse, Tool, ToolCallSource, ToolExecutionContext, ToolExecutionRequest,
    ToolInvoker, ToolResult, ToolSafetyMetadata,
};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct WorkProbe {
    calls: Arc<AtomicUsize>,
    deadline_target: bool,
}
#[async_trait]
impl Tool for WorkProbe {
    fn id(&self) -> &str {
        "work_probe"
    }
    fn name(&self) -> &str {
        "Work Probe"
    }
    fn description(&self) -> &str {
        "Observe actual shared-executor admission"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object"})
    }
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: true,
            concurrency_safe: false,
            ..Default::default()
        }
    }
    fn declared_write_footprint(
        &self,
        _: &Value,
        ctx: &ToolExecutionContext,
        _: usize,
    ) -> ai_agents_core::Result<Option<ToolWriteFootprint>> {
        Ok(Some(ToolWriteFootprint {
            binding_identity: "work-probe.v1".into(),
            targets: if self.deadline_target {
                vec![
                    if ctx.deadline.is_some() {
                        "final"
                    } else {
                        "initial"
                    }
                    .into(),
                ]
            } else {
                vec![]
            },
        }))
    }
    async fn execute(&self, _: Value, _: ToolExecutionContext) -> ToolResult {
        self.calls.fetch_add(1, Ordering::SeqCst);
        ToolResult::ok("observed")
    }
}

// These fixtures use the production ToolInvoker and real run reservation rather than direct Tool::execute.
async fn shared_executor_fixture(
    profile: AutonomyProfile,
    deadline_target: bool,
) -> (
    Arc<crate::RuntimeAgent>,
    Arc<RunExecution>,
    Arc<AtomicUsize>,
) {
    let calls = Arc::new(AtomicUsize::new(0));
    let agent = Arc::new(
        crate::AgentBuilder::new()
            .system_prompt("test")
            .llm(Arc::new(ai_agents_llm::mock::MockLLMProvider::new(
                "default",
            )))
            .tool(Arc::new(WorkProbe {
                calls: calls.clone(),
                deadline_target,
            }))
            .build()
            .unwrap(),
    );
    let config = AutonomyConfig {
        defaults: AutonomyProfile {
            enabled: Some(true),
            completion: Some(CompletionGate::ResponseContains("done".into())),
            ..profile
        },
        ..Default::default()
    };
    let host = AutonomyHostCeilings {
        max_declared_write_paths: Some(2),
        ..Default::default()
    };
    let effective = resolve_profile(
        &config,
        AutonomyScope::Task,
        None,
        None,
        None,
        Some("objective"),
        &host,
    )
    .unwrap();
    let owner = agent.reserve_autonomy_run("shared".into()).await.unwrap();
    let store =
        Arc::new(ScopedTaskRunStore::in_memory(agent.info().id, None, "shared-v1".into()).unwrap());
    let payload = TaskCheckpointPayload::new(
        "objective".into(),
        "shared-v1".into(),
        &effective,
        TaskRuntimeCheckpoint::between_turns(agent.save_state().await.unwrap()).unwrap(),
    );
    let now = chrono::Utc::now();
    let snapshot = payload
        .bind(TaskRunSnapshot {
            schema_version: TASK_RUN_SCHEMA_VERSION,
            key: TaskRunKey {
                agent_id: agent.info().id,
                run_id: "shared".into(),
            },
            actor_id: None,
            revision: 0,
            status: TaskRunStatus::Running,
            owner_token: Some("owner".into()),
            cancel_requested: false,
            created_at: now,
            updated_at: now,
            payload: Value::Null,
        })
        .unwrap();
    store.create(&snapshot).await.unwrap();
    let execution = RunExecution::new(
        store,
        &snapshot,
        LifecycleState::new(&effective.settings, false).unwrap(),
        owner,
        std::time::Instant::now(),
    )
    .unwrap();
    execution.set_scope(&EvaluationScope {
        key: snapshot.key,
        objective_revision: 0,
        cycle: 1,
        stage: None,
        mutation_generation: 0,
        target_revisions: Default::default(),
        target_generations: Default::default(),
        validation_bindings: Default::default(),
        validation_attempts: Default::default(),
    });
    (agent, execution, calls)
}

#[tokio::test]
async fn lifecycle_shared_executor_denies_effect_until_required_stage_passes() {
    let profile = AutonomyProfile {
        lifecycle: Some(
            serde_yaml::from_str("- id: explore\n  required_before_tools: [work_probe]\n").unwrap(),
        ),
        ..Default::default()
    };
    let (agent, execution, calls) = shared_executor_fixture(profile.clone(), false).await;
    scope_execution(execution.clone(), async {
        let request =
            |id| ToolExecutionRequest::new(id, "work_probe", json!({}), ToolCallSource::Task);
        assert!(agent.invoke_tool(request("denied")).await.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let mut lifecycle = LifecycleState::new(&profile, false).unwrap();
        lifecycle
            .complete_stage(&profile, "explore", GateOutcome::Pass, GateOutcome::Pass)
            .unwrap();
        execution.set_lifecycle(&profile, &lifecycle);
        let record = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            agent.invoke_tool(request("allowed")),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(record.executed && record.success);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    })
    .await;
    agent.release_autonomy_run(&execution.owner).await.unwrap();
}

#[tokio::test]
async fn progress_denial_does_not_retain_locks_or_charge_an_attempt() {
    let profile = AutonomyProfile {
        progress: Some(ProgressConfig {
            require_todos: Some(true),
            require_active_todo: Some(true),
            ..Default::default()
        }),
        ..Default::default()
    };
    let (agent, execution, calls) = shared_executor_fixture(profile, false).await;
    let key = TaskRunKey {
        agent_id: agent.info().id,
        run_id: "shared".into(),
    };
    let todo = RunTodoAdapter::begin(agent.todo_store(), &key).unwrap();
    execution.install_todos(agent.todo_store(), todo.checkpoint().unwrap().binding);
    scope_execution(execution.clone(), async {
        let request = |id| {
            ToolExecutionRequest::new(
                id,
                "work_probe",
                json!({}),
                ToolCallSource::Plan { step_index: 0 },
            )
        };
        for id in ["first-denial", "second-denial"] {
            assert!(
                tokio::time::timeout(
                    std::time::Duration::from_secs(2),
                    agent.invoke_tool(request(id))
                )
                .await
                .unwrap()
                .is_err()
            );
        }
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        agent.todo_store().set(vec![ai_agents_tools::TodoItem {
            id: "task".into(),
            content: "work".into(),
            active_form: None,
            status: ai_agents_tools::TodoStatus::InProgress,
        }]);
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_secs(2),
                agent.invoke_tool(request("after-repair"))
            )
            .await
            .unwrap()
            .unwrap()
            .executed
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let payload: TaskCheckpointPayload = serde_json::from_value(
            execution
                .store
                .load("shared")
                .await
                .unwrap()
                .unwrap()
                .payload,
        )
        .unwrap();
        assert_eq!(payload.counters.tool_attempts, 1);
    })
    .await;
    todo.release().unwrap();
    agent.release_autonomy_run(&execution.owner).await.unwrap();
}

#[tokio::test]
async fn final_context_footprint_change_refunds_and_prevents_the_effect() {
    let profile = AutonomyProfile {
        max_declared_write_paths: Some(2),
        ..Default::default()
    };
    let (agent, execution, calls) = shared_executor_fixture(profile, true).await;
    scope_execution(execution.clone(), async {
        let record = agent
            .invoke_tool(ToolExecutionRequest::new(
                "changed",
                "work_probe",
                json!({}),
                ToolCallSource::Task,
            ))
            .await
            .unwrap();
        assert!(!record.executed);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let payload: TaskCheckpointPayload = serde_json::from_value(
            execution
                .store
                .load("shared")
                .await
                .unwrap()
                .unwrap()
                .payload,
        )
        .unwrap();
        assert_eq!(payload.counters.tool_attempts, 0);
        assert!(payload.declared_write_targets.is_empty());
    })
    .await;
    agent.release_autonomy_run(&execution.owner).await.unwrap();
}

#[test]
fn canonical_todo_growth_bound_is_atomic_and_released_with_the_run() {
    let store = ai_agents_tools::TodoStore::default();
    let binding = ai_agents_tools::TodoRunBinding {
        agent_id: "agent".into(),
        run_id: "run".into(),
        token: "owner".into(),
        objective_revision: 0,
    };
    assert!(store.bind_run(binding.clone(), vec![]));
    assert!(store.set_run_open_limit(&binding, Some(1)));
    let item = |id: &str, status| ai_agents_tools::TodoItem {
        id: id.into(),
        content: id.into(),
        active_form: None,
        status,
    };
    assert!(store.try_set(vec![
        item("open", ai_agents_tools::TodoStatus::Pending),
        item("done", ai_agents_tools::TodoStatus::Completed)
    ]));
    assert!(!store.update(
        "done",
        None,
        None,
        Some(ai_agents_tools::TodoStatus::Pending)
    ));
    assert!(!store.try_set(vec![
        item("first", ai_agents_tools::TodoStatus::Pending),
        item("second", ai_agents_tools::TodoStatus::Pending)
    ]));
    assert_eq!(store.list()[0].id, "open");
    assert!(store.release_run(&binding));
    assert!(store.try_set(vec![
        item("first", ai_agents_tools::TodoStatus::Pending),
        item("second", ai_agents_tools::TodoStatus::Pending)
    ]));
}

#[tokio::test]
async fn objective_template_is_resolved_once_before_user_commit() {
    let provider = super::runner_tests::RecordingProvider::new(&["working", "done"]);
    let (agent, _, store) = super::runner_tests::fixture(provider, 3);
    agent.context_manager().set("ticket", json!(42)).unwrap();
    let mut config = super::runner_tests::config(3);
    config.defaults.objective_template = Some("{{ input }} ticket {{ context.ticket }}".into());
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        AutonomyHostCeilings::default(),
        store.clone(),
        "prepared-v1".into(),
    )
    .unwrap();
    let result = runner.run("resolve", None).await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    let payload: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&result.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert_eq!(payload.objective, "resolve ticket 42");
    let users: Vec<_> = payload
        .runtime
        .snapshot
        .memory
        .messages
        .iter()
        .filter(|message| message.role == ai_agents_core::Role::User)
        .collect();
    assert_eq!(users.len(), 1);
    assert_eq!(users[0].content, payload.objective);
}

#[tokio::test]
async fn controller_continuation_does_not_reroute_or_replay_the_initial_skill() {
    let mut provider = ai_agents_llm::mock::MockLLMProvider::new("default");
    for reply in ["script", "working", "done"] {
        provider.add_response(LLMResponse::new(reply, FinishReason::Stop));
    }
    let provider = Arc::new(provider);
    let skill = ai_agents_skills::SkillDefinition {
        autonomy: None,
        id: "script".into(),
        description: "work".into(),
        trigger: "when asked".into(),
        steps: vec![ai_agents_skills::SkillStep::Prompt {
            prompt: "INITIAL_SCRIPT_SEGMENT".into(),
            llm: None,
        }],
        reasoning: None,
        reflection: None,
        disambiguation: None,
    };
    let agent = Arc::new(
        crate::AgentBuilder::new()
            .system_prompt("test")
            .llm(provider.clone())
            .skills(vec![skill])
            .build()
            .unwrap(),
    );
    let store =
        Arc::new(ScopedTaskRunStore::in_memory(agent.info().id, None, "script-v1".into()).unwrap());
    let runner = AutonomyRunner::try_new(
        agent,
        super::runner_tests::config(3),
        AutonomyHostCeilings::default(),
        store,
        "script-v1".into(),
    )
    .unwrap();
    let result = runner.run("objective", None).await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(result.run.counters.llm_attempts, 3);
    assert_eq!(provider.call_count(), 3);
    assert_eq!(
        provider
            .call_history()
            .iter()
            .filter(|call| call
                .messages
                .iter()
                .any(|message| message.content.contains("INITIAL_SCRIPT_SEGMENT")))
            .count(),
        1
    );
}

#[tokio::test]
async fn require_todos_is_not_an_implicit_todos_done_completion_gate() {
    let provider = super::runner_tests::RecordingProvider::gated();
    let (agent, _, store) = super::runner_tests::fixture(provider.clone(), 1);
    let mut config = super::runner_tests::config(1);
    config.defaults.progress = Some(ProgressConfig {
        require_todos: Some(true),
        ..Default::default()
    });
    let runner = AutonomyRunner::try_new(
        agent.clone(),
        config,
        AutonomyHostCeilings::default(),
        store,
        "prepared-v1".into(),
    )
    .unwrap();
    let task = tokio::spawn(async move { runner.run("objective", None).await });
    tokio::time::timeout(
        std::time::Duration::from_secs(2),
        provider.entered.as_ref().unwrap().acquire(),
    )
    .await
    .unwrap()
    .unwrap()
    .forget();
    agent.todo_store().set(vec![ai_agents_tools::TodoItem {
        id: "pending".into(),
        content: "remaining".into(),
        active_form: None,
        status: ai_agents_tools::TodoStatus::Pending,
    }]);
    provider.release.as_ref().unwrap().add_permits(1);
    assert_eq!(
        task.await.unwrap().unwrap().run.status,
        TaskRunStatus::Completed
    );
}

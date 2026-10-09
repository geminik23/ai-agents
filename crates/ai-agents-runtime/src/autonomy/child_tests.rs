use super::*;
use crate::spawner::{AgentRegistry, SpawnedAgent};
use crate::{Agent, RuntimeAgent};
use ai_agents_core::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
};
use ai_agents_llm::mock::MockLLMProvider;
use ai_agents_state::{AggregationConfig, ConcurrentAgentRef, PartialFailureAction};
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

// Each child uses the production builder so provider wrapping and its own root gate remain authoritative.
async fn register(
    registry: &AgentRegistry,
    id: &str,
    replies: &[&str],
) -> (Arc<RuntimeAgent>, Arc<MockLLMProvider>) {
    let mut provider = MockLLMProvider::new(id);
    for reply in replies {
        provider.add_response(LLMResponse::new(*reply, FinishReason::Stop));
    }
    let provider = Arc::new(provider);
    let agent = crate::AgentBuilder::new()
        .system_prompt("child")
        .llm(provider.clone())
        .build()
        .unwrap();
    registry
        .register(SpawnedAgent::from_runtime(
            id.into(),
            agent,
            crate::spec::AgentSpec {
                name: id.into(),
                ..Default::default()
            },
        ))
        .await
        .unwrap();
    (registry.get(id).unwrap(), provider)
}

// Direct orchestration SDK calls share exactly the same foreground reservation and conditional ledger as the runner.
async fn execution(max_llm_calls: u32) -> (Arc<RuntimeAgent>, Arc<RunExecution>) {
    let mut provider = MockLLMProvider::new("parent");
    provider.set_response("done");
    let parent = Arc::new(
        crate::AgentBuilder::new()
            .system_prompt("parent")
            .llm(Arc::new(provider))
            .build()
            .unwrap(),
    );
    let config = super::runner_tests::config(max_llm_calls);
    let effective = resolve_profile(
        &config,
        AutonomyScope::Task,
        None,
        None,
        None,
        Some("objective"),
        &AutonomyHostCeilings::default(),
    )
    .unwrap();
    let owner = parent
        .reserve_autonomy_run("children".into())
        .await
        .unwrap();
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(parent.info().id, None, "children-v1".into()).unwrap(),
    );
    let data = TaskCheckpointPayload::new(
        "objective".into(),
        "children-v1".into(),
        &effective,
        TaskRuntimeCheckpoint::between_turns(parent.save_state().await.unwrap()).unwrap(),
    );
    let now = chrono::Utc::now();
    let snapshot = data
        .bind(TaskRunSnapshot {
            schema_version: TASK_RUN_SCHEMA_VERSION,
            key: TaskRunKey {
                agent_id: parent.info().id,
                run_id: "children".into(),
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
    (parent, execution)
}

// Acknowledged success releases children only after the parent terminal CAS, never at the end of an individual child turn.
async fn finish(parent: &RuntimeAgent, execution: &RunExecution) {
    let saved = execution
        .terminal(TaskRunStatus::Completed, "complete".into())
        .await
        .unwrap();
    assert_eq!(saved.status, TaskRunStatus::Completed);
    execution.participants.release().await.unwrap();
    parent.release_autonomy_run(&execution.owner).await.unwrap();
}

#[tokio::test]
async fn pipeline_children_share_counters_and_remain_reserved_between_turns() {
    let registry = AgentRegistry::new();
    let (first, first_provider) = register(&registry, "first", &["first result", "ordinary"]).await;
    let (second, _) = register(&registry, "second", &["done"]).await;
    let (parent, execution) = execution(2).await;
    let result = scope_execution(
        execution.clone(),
        crate::orchestration::pipeline(
            &registry,
            "objective",
            &[
                crate::orchestration::PipelineStage::id("first"),
                crate::orchestration::PipelineStage::id("second"),
            ],
            None,
            None,
            None,
        ),
    )
    .await
    .unwrap();
    assert_eq!(result.response.content, "done");
    assert_eq!(first_provider.call_count(), 1);
    assert!(first.chat("unrelated").await.is_err());
    assert!(second.chat("unrelated").await.is_err());
    assert!(
        first
            .update_context("foreign", serde_json::json!(1))
            .is_err()
    );
    assert!(first.remove_context("foreign").is_err());
    assert!(first.refresh_context("foreign").await.is_err());
    let payload: TaskCheckpointPayload = serde_json::from_value(
        execution
            .store
            .load("children")
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert_eq!(payload.counters.llm_attempts, 2);
    assert_eq!(payload.children.len(), 2);
    assert!(payload.children.iter().all(|child| child.result.is_some()));
    assert!(
        payload
            .children
            .iter()
            .flat_map(|child| &child.runtime.snapshot.memory.messages)
            .all(|message| message
                .provenance
                .as_ref()
                .is_some_and(|provenance| provenance.run_id == "children"))
    );
    finish(&parent, &execution).await;
    assert_eq!(first.chat("ordinary").await.unwrap().content, "ordinary");
}

#[tokio::test]
async fn concurrent_children_and_borrowed_aggregator_use_one_allowance() {
    let registry = AgentRegistry::new();
    register(&registry, "first", &["first result"]).await;
    register(&registry, "second", &["second result"]).await;
    let (parent, execution) = execution(3).await;
    let mut aggregator = MockLLMProvider::new("aggregation");
    aggregator.set_response("done");
    let agents: Vec<ConcurrentAgentRef> = serde_yaml::from_str("- first\n- second\n").unwrap();
    let aggregation: AggregationConfig = serde_yaml::from_str("strategy: llm_synthesis\n").unwrap();
    let result = scope_execution(
        execution.clone(),
        crate::orchestration::concurrent(
            &registry,
            "objective",
            &agents,
            &aggregation,
            Some(&aggregator),
            Some(2),
            None,
            PartialFailureAction::Abort,
            None,
        ),
    )
    .await
    .unwrap();
    assert_eq!(result.response.content, "done");
    assert_eq!(
        result
            .agent_results
            .iter()
            .map(|result| result.agent_index)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    let payload: TaskCheckpointPayload = serde_json::from_value(
        execution
            .store
            .load("children")
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert_eq!(payload.counters.llm_attempts, 3);
    assert_eq!(payload.children.len(), 2);
    finish(&parent, &execution).await;
}

struct FailingProvider;
#[async_trait]
impl LLMProvider for FailingProvider {
    async fn complete(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        Err(LLMError::Config("required child failed".into()))
    }
    async fn complete_stream(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<
        Box<dyn futures::Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
        LLMError,
    > {
        Err(LLMError::Config("failed".into()))
    }
    fn provider_name(&self) -> &str {
        "failing"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
    fn is_terminal_error(&self, _: &LLMError) -> bool {
        true
    }
}

#[tokio::test]
async fn optional_child_failure_preserves_existing_partial_failure_policy() {
    let registry = AgentRegistry::new();
    register(&registry, "good", &["done"]).await;
    let bad = crate::AgentBuilder::new()
        .system_prompt("child")
        .llm(Arc::new(FailingProvider))
        .build()
        .unwrap();
    registry
        .register(SpawnedAgent::from_runtime(
            "bad".into(),
            bad,
            crate::spec::AgentSpec::default(),
        ))
        .await
        .unwrap();
    let (parent, execution) = execution(3).await;
    let agents: Vec<ConcurrentAgentRef> = serde_yaml::from_str("- good\n- bad\n").unwrap();
    let aggregation: AggregationConfig = serde_yaml::from_str("strategy: all\n").unwrap();
    let result = scope_execution(
        execution.clone(),
        crate::orchestration::concurrent(
            &registry,
            "objective",
            &agents,
            &aggregation,
            None,
            Some(1),
            None,
            PartialFailureAction::ProceedWithAvailable,
            None,
        ),
    )
    .await
    .unwrap();
    assert_eq!(
        result
            .agent_results
            .iter()
            .filter(|result| result.success)
            .count(),
        1
    );
    assert!(execution.stop_reason().is_none());
    finish(&parent, &execution).await;
}

#[tokio::test]
async fn required_child_failure_cannot_become_successful_parent_completion() {
    let registry = AgentRegistry::new();
    let bad = crate::AgentBuilder::new()
        .system_prompt("child")
        .llm(Arc::new(FailingProvider))
        .build()
        .unwrap();
    registry
        .register(SpawnedAgent::from_runtime(
            "bad".into(),
            bad,
            crate::spec::AgentSpec::default(),
        ))
        .await
        .unwrap();
    let (parent, execution) = execution(2).await;
    assert!(
        scope_execution(
            execution.clone(),
            registry.get("bad").unwrap().chat("objective")
        )
        .await
        .is_err()
    );
    let saved = execution
        .terminal(TaskRunStatus::Completed, "aggregate said done".into())
        .await
        .unwrap();
    assert_eq!(saved.status, TaskRunStatus::Failed);
    execution.participants.release().await.unwrap();
    parent.release_autonomy_run(&execution.owner).await.unwrap();
}

#[tokio::test]
async fn child_cannot_reset_the_enclosing_provider_cap() {
    let registry = AgentRegistry::new();
    register(&registry, "first", &["first"]).await;
    let (_, second_provider) = register(&registry, "second", &["done"]).await;
    let (parent, execution) = execution(1).await;
    let stages = [
        crate::orchestration::PipelineStage::id("first"),
        crate::orchestration::PipelineStage::id("second"),
    ];
    assert!(
        scope_execution(
            execution.clone(),
            crate::orchestration::pipeline(&registry, "objective", &stages, None, None, None)
        )
        .await
        .is_err()
    );
    assert_eq!(second_provider.call_count(), 0);
    let saved = execution
        .terminal(TaskRunStatus::LimitReached, "llm_capacity".into())
        .await
        .unwrap();
    assert_eq!(saved.status, TaskRunStatus::LimitReached);
    let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload).unwrap();
    assert_eq!(payload.counters.llm_attempts, 1);
    execution.participants.release().await.unwrap();
    parent.release_autonomy_run(&execution.owner).await.unwrap();
}

#[tokio::test]
async fn explicit_runner_executes_existing_delegate_state_and_releases_its_child() {
    let registry = Arc::new(AgentRegistry::new());
    let (child, provider) = register(&registry, "worker", &["done", "ordinary"]).await;
    let mut root_model = MockLLMProvider::new("parent");
    root_model.set_response("unused");
    let parent = crate::AgentBuilder::from_yaml("name: Parent\nsystem_prompt: parent\nstates:\n  initial: delegate\n  states:\n    delegate:\n      delegate: worker\n").unwrap().llm(Arc::new(root_model)).auto_configure_features().unwrap().build().unwrap().with_spawner_handles(Arc::new(crate::spawner::AgentSpawner::new()), registry);
    let parent = Arc::new(parent);
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(parent.info().id, None, "runner-child-v1".into()).unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        parent,
        super::runner_tests::config(1),
        AutonomyHostCeilings::default(),
        store.clone(),
        "runner-child-v1".into(),
    )
    .unwrap();
    let result = runner.run("objective", None).await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(result.run.counters.llm_attempts, 1);
    assert_eq!(provider.call_count(), 1);
    let data: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&result.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert_eq!(data.children.len(), 1);
    assert_eq!(
        data.runtime.snapshot.spawned_agents.as_ref().unwrap().len(),
        1
    );
    assert_eq!(child.chat("ordinary").await.unwrap().content, "ordinary");
}

#[tokio::test]
async fn child_drop_keeps_runtime_reservation_and_forces_parent_recovery() {
    let provider = super::runner_tests::RecordingProvider::gated();
    let child = Arc::new(
        crate::AgentBuilder::new()
            .system_prompt("child")
            .llm(provider.clone())
            .build()
            .unwrap(),
    );
    let (parent, execution) = execution(2).await;
    let participant = child.clone();
    let captured = execution.clone();
    let task =
        tokio::spawn(async move { scope_execution(captured, participant.chat("objective")).await });
    tokio::time::timeout(
        std::time::Duration::from_secs(2),
        provider.entered.as_ref().unwrap().acquire(),
    )
    .await
    .unwrap()
    .unwrap()
    .forget();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    let saved = execution
        .terminal(TaskRunStatus::Completed, "not complete".into())
        .await
        .unwrap();
    assert_eq!(saved.status, TaskRunStatus::RecoveryRequired);
    assert!(execution.participants.release().await.is_err());
    assert!(child.chat("unrelated").await.is_err());
    assert!(parent.chat("unrelated").await.is_err());
}

#[tokio::test]
async fn canonical_task_todos_override_a_child_tools_captured_session_store() {
    use ai_agents_core::{Tool, ToolExecutionContext};
    let (parent, execution) = execution(2).await;
    let key = TaskRunKey {
        agent_id: parent.info().id,
        run_id: "children".into(),
    };
    let todos = RunTodoAdapter::begin(parent.todo_store(), &key).unwrap();
    todos.set_open_limit(Some(1)).unwrap();
    execution.install_todos(parent.todo_store(), todos.checkpoint().unwrap().binding);
    let session = ai_agents_tools::TodoStore::default();
    let child_tool = ai_agents_tools::TodoTool::new(session.clone());
    scope_execution(execution.clone(), async {
        let set = child_tool.execute(serde_json::json!({"operation":"set","items":[{"id":"work","content":"work","status":"in_progress"}]}), ToolExecutionContext::test("todo")).await;
        assert!(set.success);
        assert_eq!(todos.checkpoint().unwrap().items.len(), 1);
        assert!(session.list().is_empty());
        let too_many = child_tool.execute(serde_json::json!({"operation":"set","items":[{"id":"a","content":"a"},{"id":"b","content":"b"}]}), ToolExecutionContext::test("todo")).await;
        assert!(!too_many.success);
        assert_eq!(todos.checkpoint().unwrap().items[0].id, "work");
    }).await;
    todos.release().unwrap();
    finish(&parent, &execution).await;
}

#[tokio::test]
async fn nested_dependency_failure_in_an_optional_branch_remains_optional_to_the_parent() {
    let registry = Arc::new(AgentRegistry::new());
    register(&registry, "good", &["done"]).await;
    let bad = crate::AgentBuilder::new()
        .system_prompt("bad")
        .llm(Arc::new(FailingProvider))
        .build()
        .unwrap();
    registry
        .register(SpawnedAgent::from_runtime(
            "bad".into(),
            bad,
            crate::spec::AgentSpec::default(),
        ))
        .await
        .unwrap();
    let nested = crate::AgentBuilder::from_yaml("name: Optional\nsystem_prompt: optional\nstates:\n  initial: work\n  states:\n    work:\n      delegate: bad\n").unwrap().llm(Arc::new(MockLLMProvider::new("unused"))).auto_configure_features().unwrap().build().unwrap().with_spawner_handles(Arc::new(crate::spawner::AgentSpawner::new()), registry.clone());
    registry
        .register(SpawnedAgent::from_runtime(
            "optional".into(),
            nested,
            crate::spec::AgentSpec::default(),
        ))
        .await
        .unwrap();
    let (parent, execution) = execution(3).await;

    let agents: Vec<ConcurrentAgentRef> = serde_yaml::from_str("- good\n- optional\n").unwrap();
    let aggregation: AggregationConfig = serde_yaml::from_str("strategy: all\n").unwrap();
    let result = scope_execution(
        execution.clone(),
        crate::orchestration::concurrent(
            &registry,
            "objective",
            &agents,
            &aggregation,
            None,
            Some(1),
            None,
            PartialFailureAction::ProceedWithAvailable,
            None,
        ),
    )
    .await
    .unwrap();
    assert_eq!(
        result
            .agent_results
            .iter()
            .filter(|result| result.success)
            .count(),
        1
    );
    assert!(execution.stop_reason().is_none());
    finish(&parent, &execution).await;
}

struct PendingApproval(Arc<tokio::sync::Semaphore>);
#[async_trait]
impl ai_agents_hitl::ApprovalHandler for PendingApproval {
    async fn request_approval(
        &self,
        _: ai_agents_hitl::ApprovalRequest,
    ) -> ai_agents_hitl::ApprovalResult {
        self.0.add_permits(1);
        std::future::pending().await
    }
}

#[tokio::test]
async fn coordinator_cancel_bounds_a_child_approval_wait_without_invoking_the_tool() {
    let entered = Arc::new(tokio::sync::Semaphore::new(0));
    let mut provider = MockLLMProvider::new("approval");
    provider.add_response(LLMResponse::new(
        r#"{"tool":"echo","arguments":{"message":"hello"}}"#,
        FinishReason::Stop,
    ));
    let provider = Arc::new(provider);
    let child = Arc::new(crate::AgentBuilder::from_yaml("name: ApprovalChild\nsystem_prompt: child\ntools: [echo]\nhitl:\n  tools:\n    echo:\n      require_approval: true\n").unwrap().llm(provider.clone()).auto_configure_features().unwrap().approval_handler(Arc::new(PendingApproval(entered.clone()))).build().unwrap());
    let (parent, execution) = execution(3).await;
    let participant = child.clone();
    let captured = execution.clone();
    let task =
        tokio::spawn(async move { scope_execution(captured, participant.chat("objective")).await });
    tokio::time::timeout(std::time::Duration::from_secs(2), entered.acquire())
        .await
        .unwrap()
        .unwrap()
        .forget();
    execution.request_cancel().await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let saved = execution
        .terminal(TaskRunStatus::Completed, "cancelled".into())
        .await
        .unwrap();
    assert_eq!(saved.status, TaskRunStatus::Cancelled);
    let data: TaskCheckpointPayload = serde_json::from_value(saved.payload).unwrap();
    assert_eq!(data.counters.tool_attempts, 0);
    assert_eq!(provider.call_count(), 1);
    execution.participants.release().await.unwrap();
    parent.release_autonomy_run(&execution.owner).await.unwrap();
}

#[tokio::test]
async fn cached_child_result_requires_matching_operation_input_and_runtime() {
    let registry = AgentRegistry::new();
    let (child, provider) = register(&registry, "child", &["done"]).await;
    let (parent, execution) = execution(2).await;
    assert_eq!(
        scope_execution(execution.clone(), child.chat("objective"))
            .await
            .unwrap()
            .content,
        "done"
    );
    let payload: TaskCheckpointPayload = serde_json::from_value(
        execution
            .store
            .load("children")
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    let operation = &payload.children[0].child_id;
    assert_eq!(
        execution
            .cached_child(operation, "objective", &child.info().id)
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .content,
        "done"
    );
    assert!(
        execution
            .cached_child(operation, "changed", &child.info().id)
            .await
            .is_err()
    );
    assert_eq!(provider.call_count(), 1);
    finish(&parent, &execution).await;
}

use super::evaluation_tests::{bound_script, current, identity, scope};
use super::*;
use ai_agents_core::autonomy::TaskRunMutation;
use ai_agents_core::{
    FinishReason, LLMResponse, Tool, ToolCallSource, ToolCancellationToken, ToolExecutionContext,
    ToolExecutionRequest, ToolInvoker, ToolResult, ToolSafetyMetadata,
};
use ai_agents_llm::{LLMRegistry, mock::MockLLMProvider};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

struct Probe(Arc<AtomicUsize>);
#[async_trait]
impl Tool for Probe {
    /// Stable fixture identity is used by real registry and policy resolution.
    fn id(&self) -> &str {
        "probe"
    }
    /// Display naming cannot widen fixture authority.
    fn name(&self) -> &str {
        "Probe"
    }
    /// This fixture simulates a host observation rather than an alternate executor.
    fn description(&self) -> &str {
        "Count an admitted observation"
    }
    /// The shared executor receives the same object arguments as the driver.
    fn input_schema(&self) -> Value {
        json!({"type":"object"})
    }
    /// The observable counter increments only after real final admission.
    async fn execute(&self, _: Value, _: ToolExecutionContext) -> ToolResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        ToolResult::ok("{\"value\":1}")
    }
    /// Read-only fixture classification avoids unrelated filesystem approval requirements.
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: true,
            concurrency_safe: true,
            ..Default::default()
        }
    }
}
// Hosts freeze exact validation authority before constructing a runtime with no ordinary grants.
fn host_agent(require_approval: bool) -> (Arc<crate::RuntimeAgent>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut extensions = AutonomyExtensions::builtins();
    extensions
        .register_host_validation(HostValidationBinding {
            id: "fixed".into(),
            validator: "host.script".into(),
            version: 1,
            profile: None,
            scope: "task".into(),
            tool: "probe".into(),
            arguments: json!({"value":1}),
            require_approval,
        })
        .unwrap();
    let agent = crate::AgentBuilder::new()
        .system_prompt("test")
        .llm(Arc::new(MockLLMProvider::new("default")))
        .tool(Arc::new(Probe(calls.clone())))
        .autonomy_extensions(extensions)
        .build()
        .unwrap()
        .with_declared_tool_ids(Some(vec![]));
    (Arc::new(agent), calls)
}
// Copying a source label or binding name cannot mint the private execution scope.
#[tokio::test]
async fn host_validation_uses_shared_policy_without_model_exposure() {
    let (agent, calls) = host_agent(false);
    let s = scope();
    let i = identity(&s);
    let denied = agent
        .invoke_tool(ToolExecutionRequest::new(
            "model",
            "probe",
            json!({"value":1,"host_binding":"fixed"}),
            ToolCallSource::Task,
        ))
        .await
        .unwrap();
    assert!(!denied.executed);
    let profile = None;
    let context = HostValidationContext {
        identity: &i,
        validator: "host.script",
        contract_version: 1,
        profile: &profile,
        scope_mode: "task",
        run_revision: 1,
    };
    let result = agent
        .invoke_host_validation(
            "fixed",
            ToolExecutionRequest::new("host", "probe", json!({"value":1}), ToolCallSource::Task),
            &context,
        )
        .await
        .unwrap();
    assert!(result.executed && result.success);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(result.metadata.contains_key("host_validation"));
    assert!(
        agent
            .invoke_host_validation(
                "fixed",
                ToolExecutionRequest::new(
                    "wrong",
                    "probe",
                    json!({"value":2}),
                    ToolCallSource::Task
                ),
                &context
            )
            .await
            .is_err()
    );
    let denied = agent
        .invoke_tool(ToolExecutionRequest::new(
            "after",
            "probe",
            json!({"value":1}),
            ToolCallSource::Model,
        ))
        .await
        .unwrap();
    assert!(!denied.executed);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let (agent, calls) = host_agent(true);
    let denied = agent
        .invoke_host_validation(
            "fixed",
            ToolExecutionRequest::new(
                "approval",
                "probe",
                json!({"value":1}),
                ToolCallSource::Task,
            ),
            &context,
        )
        .await
        .unwrap();
    assert!(!denied.executed);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

// SQLite and memory execute the same claim, pre-effect journal, exact resume and callback contract.
async fn journal_contract(storage: Arc<dyn ai_agents_core::AgentStorage>) {
    let (bound, mut s, _) = bound_script("complete");
    let check = &bound.checks[0];
    let calls = Arc::new(AtomicUsize::new(0));
    let agent = Arc::new(
        crate::AgentBuilder::new()
            .system_prompt("test")
            .llm(Arc::new(MockLLMProvider::new("default")))
            .tool(Arc::new(Probe(calls.clone())))
            .build()
            .unwrap(),
    );
    s.key.agent_id = agent.info().id;
    let store = Arc::new(
        ScopedTaskRunStore::new(storage, s.key.agent_id.clone(), None, "config-v1".into()).unwrap(),
    );
    let snapshot = super::tests::checkpoint("run");
    store.create(&snapshot).await.unwrap();
    let claimed = store
        .mutate(
            "run",
            &TaskRunMutation::Claim {
                expected_revision: 0,
                owner_token: "owner".into(),
            },
        )
        .await
        .unwrap();
    let journal = TaskValidationJournal::new(
        store.clone(),
        "run".into(),
        "owner".into(),
        claimed.revision,
    );
    let executor = RuntimeObservationExecutor {
        agent,
        profile: None,
        scope_mode: "task".into(),
        run_revision: journal.revision_handle(),
        validator: check.check.adapter.clone(),
        contract_version: 1,
    };
    let mut i = identity(&s);
    i.config_identity = check.config_identity.clone();
    let mut driver = ValidationDriverState::new(check, i).unwrap();
    let evidence = current(&s);
    let cancel = ToolCancellationToken::new(Arc::new(AtomicBool::new(false)), None);
    let ctx = ValidationDriveContext {
        scope: &s,
        evidence: &evidence,
        extensions: &bound.extensions,
        executor: &executor,
        journal: &journal,
        limits: Default::default(),
        cancellation: &cancel,
        deadline: chrono::Utc::now() + chrono::Duration::seconds(2),
    };
    assert!(matches!(
        driver.drive(check, &ctx).await.unwrap(),
        ValidationDriverOutcome::Complete(_)
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let saved = store.load("run").await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload).unwrap();
    assert_eq!(payload.counters.tool_attempts, 1);
    assert_eq!(payload.reservations[0].state, TaskEffectState::Completed);
    let value = payload
        .adapters
        .iter()
        .find(|a| a.id == "check")
        .unwrap()
        .state["_task_validation_driver"]
        .clone();
    let mut restored = ValidationDriverState::restore(value, check, &s, &bound.extensions).unwrap();
    restored.drive(check, &ctx).await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let mut stale = ValidationDriverState::new(check, restored.identity.clone()).unwrap();
    assert!(stale.drive(check, &ctx).await.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
// Volatile journaling does not require a second storage or execution authority.
#[tokio::test]
async fn validation_driver_journals_memory_and_reuses_exact_results() {
    journal_contract(Arc::new(ai_agents_storage::InMemoryTaskStorage::default())).await;
}
// SQLite backs the same driver and owner transitions instead of a separate persisted validation engine.
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn validation_driver_journals_sqlite_and_reuses_exact_results() {
    journal_contract(Arc::new(
        ai_agents_storage::SqliteStorage::in_memory().await.unwrap(),
    ))
    .await;
}

// Judges resolve the existing registry role/alias before dispatch, and their text never becomes tool authority.
#[tokio::test]
async fn builtin_judge_and_gate_use_frozen_aliases_and_strict_scores() {
    let mut mock = MockLLMProvider::new("judge");
    mock.add_response(LLMResponse::new(
        "{\"overall_score\":0.95}",
        FinishReason::Stop,
    ));
    let mock = Arc::new(mock);
    let mut registry = LLMRegistry::new();
    registry.register("default", mock.clone());
    registry.register("quality", mock.clone());
    let agent = Arc::new(
        crate::AgentBuilder::new()
            .system_prompt("test")
            .llm_registry(registry)
            .build()
            .unwrap(),
    );
    let profile: AutonomyProfile =
        serde_yaml::from_str("completion: {judge: {llm: quality, criteria: [Correctness]}}")
            .unwrap();
    let bound = agent
        .autonomy_extensions()
        .bind(
            &profile,
            &ValidationCapabilities {
                judge: true,
                ..Default::default()
            },
        )
        .unwrap();
    let check = &bound.checks[0];
    let mut s = scope();
    bound.prepare_scope(&mut s);
    s.validation_attempts
        .insert(check.check.id.clone(), "attempt".into());
    let mut i = identity(&s);
    i.config_identity = check.config_identity.clone();
    let mut driver = ValidationDriverState::new(check, i).unwrap();
    let journal = super::evaluation_tests::MemoryJournal::default();
    let executor = RuntimeObservationExecutor {
        agent,
        profile: None,
        scope_mode: "task".into(),
        run_revision: Arc::new(AtomicU64::new(0)),
        validator: check.check.adapter.clone(),
        contract_version: 1,
    };
    let mut evidence = current(&s);
    let cancel = ToolCancellationToken::new(Arc::new(AtomicBool::new(false)), None);
    let ctx = ValidationDriveContext {
        scope: &s,
        evidence: &evidence,
        extensions: &bound.extensions,
        executor: &executor,
        journal: &journal,
        limits: Default::default(),
        cancellation: &cancel,
        deadline: chrono::Utc::now() + chrono::Duration::seconds(2),
    };
    driver.drive(check, &ctx).await.unwrap();
    evidence.collect_validation(&driver).unwrap();
    let gate = profile.completion.unwrap();
    assert_eq!(
        CompletionGateEvaluator::default()
            .evaluate(&gate, &s, &evidence)
            .unwrap()
            .outcome,
        GateOutcome::Pass
    );
    assert_eq!(mock.call_count(), 1);
    s.validation_attempts
        .insert(check.check.id.clone(), "new".into());
    assert_eq!(
        CompletionGateEvaluator::default()
            .evaluate(&gate, &s, &evidence)
            .unwrap()
            .outcome,
        GateOutcome::Unknown
    );
}

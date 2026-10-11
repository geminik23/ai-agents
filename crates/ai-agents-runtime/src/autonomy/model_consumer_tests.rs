use super::*;
use crate::spawner::{AgentRegistry, AgentSpawner, SpawnedAgent};
use crate::{AgentBuilder, RuntimeAgent};
use ai_agents_core::{Tool, ToolCall, ToolExecutionContext, ToolResult};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

struct Broadcast(Arc<AgentRegistry>, bool, Option<u64>);
#[async_trait]
impl Tool for Broadcast {
    fn id(&self) -> &str {
        "broadcast"
    }
    fn name(&self) -> &str {
        "broadcast"
    }
    fn description(&self) -> &str {
        "Broadcast through the production registry"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object"})
    }
    fn supports_task_messages(&self) -> bool {
        true
    }
    fn classify_call(&self, _: &Value) -> ai_agents_core::ToolCallClassification {
        let mut classification =
            ai_agents_core::ToolCallClassification::from_metadata(&self.safety_metadata());
        classification.timeout_ms = self.2;
        classification
    }
    async fn execute_task(
        &self,
        _: Value,
        _: ToolExecutionContext,
    ) -> ai_agents_core::Result<ToolResult> {
        scope_child_requirement(
            self.1 && child_required(),
            self.0.broadcast_task("sender", "objective", None),
        )
        .await
    }
    async fn execute(&self, _: Value, _: ToolExecutionContext) -> ToolResult {
        self.0
            .broadcast_task("sender", "objective", None)
            .await
            .unwrap()
    }
}

struct Gate {
    active: AtomicBool,
    entered: tokio::sync::Semaphore,
}
#[async_trait]
impl ai_agents_hooks::AgentHooks for Gate {
    async fn on_response(&self, _: &ai_agents_core::AgentResponse) {
        if self.active.load(Ordering::SeqCst) {
            self.entered.add_permits(1);
            std::future::pending::<()>().await;
        }
    }
}

struct Fixture {
    parent: Arc<RuntimeAgent>,
    runner: AutonomyRunner,
    store: Arc<ScopedTaskRunStore>,
    registry: Arc<AgentRegistry>,
    main: Arc<super::runner_tests::RecordingProvider>,
    workers: Vec<Arc<super::runner_tests::RecordingProvider>>,
    gate: Arc<Gate>,
}

// Signed exchanges exercise exact native prefixes as well as retained normalized calls.
fn marker(id: &str, calls: Vec<ToolCall>) -> String {
    let state = ai_agents_core::NativeProviderState::new(
        id,
        "fixture",
        "native-tools",
        ai_agents_core::NativeProviderTarget::new("https://fixture.invalid", "fixture-model")
            .unwrap(),
        json!({"signature":id}),
        calls
            .iter()
            .enumerate()
            .map(|(index, call)| ai_agents_core::NativeCallBinding::new(&call.id, index).unwrap())
            .collect(),
    )
    .unwrap();
    ai_agents_core::encode_native_tool_call_markers(&calls, Some(&state)).unwrap()
}

// Real broadcast recipients retain a completed sibling and two independent twice-parking child histories.
async fn fixture(repeated: bool, mixed: bool) -> Fixture {
    Box::pin(fixture_options(repeated, mixed, true, None)).await
}

// Explicit optional recipients and the original invocation timeout use the same framework dispatcher.
async fn fixture_options(
    repeated: bool,
    mixed: bool,
    required: bool,
    timeout: Option<u64>,
) -> Fixture {
    let registry = Arc::new(AgentRegistry::new());
    let gate = Arc::new(Gate {
        active: AtomicBool::new(false),
        entered: tokio::sync::Semaphore::new(0),
    });
    let mut workers = Vec::new();
    for id in ["a", "b", "c"] {
        let question = marker(
            &format!("{id}-signed"),
            vec![ToolCall {
                id: format!("{id}-question"),
                name: "ask_user".into(),
                arguments: json!({"question":id,"options":["yes"],"allow_other":false}),
            }],
        );
        let second = marker(
            &format!("{id}-second-signed"),
            vec![ToolCall {
                id: format!("{id}-question-second"),
                name: "ask_user".into(),
                arguments: json!({"question":id,"options":["yes"],"allow_other":false}),
            }],
        );
        let repeated_questions: Vec<_> = (0..2)
            .map(|index| {
                marker(
                    &format!("{id}-repeat-{index}-signed"),
                    vec![ToolCall {
                        id: format!("{id}-repeat-{index}-question"),
                        name: "ask_user".into(),
                        arguments: json!({"question":id,"options":["yes"],"allow_other":false}),
                    }],
                )
            })
            .collect();
        let provider = if id == "a" {
            super::runner_tests::RecordingProvider::new(&["a done", "a done"])
        } else {
            super::runner_tests::RecordingProvider::new(&[
                &question,
                &second,
                "done",
                &repeated_questions[0],
                &repeated_questions[1],
                "done",
            ])
        };
        let yaml = format!("name: {id}\nsystem_prompt: child\ntools: [ask_user]\n");
        let builder = AgentBuilder::from_yaml(&yaml)
            .unwrap()
            .llm(provider.clone())
            .auto_configure_features()
            .unwrap()
            .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
                Arc::new(parking_lot::RwLock::new(None)),
            )));
        let worker = if id == "b" {
            builder.hooks(gate.clone())
        } else {
            builder
        };
        registry
            .register(SpawnedAgent::from_runtime(
                id.into(),
                worker.build().unwrap(),
                crate::spec::AgentSpec::from_yaml_strict(&yaml).unwrap(),
            ))
            .await
            .unwrap();
        workers.push(provider);
    }
    let mut calls = vec![ToolCall {
        id: "broadcast-call".into(),
        name: "broadcast".into(),
        arguments: json!({}),
    }];
    if repeated {
        let mut next = calls[0].clone();
        next.id = "broadcast-call-second".into();
        calls.push(next);
    }
    let request = marker("parent-signed", calls);
    let main = super::runner_tests::RecordingProvider::new(&[&request, "done"]);
    let yaml = "name: BroadcastParent\nsystem_prompt: parent\ntools: [broadcast]\n";
    let parent = AgentBuilder::from_yaml(yaml)
        .unwrap()
        .llm(main.clone())
        .auto_configure_features()
        .unwrap()
        .tool(Arc::new(Broadcast(registry.clone(), required, timeout)))
        .build()
        .unwrap()
        .with_spawner_handles(Arc::new(AgentSpawner::new()), registry.clone());
    let parent = if mixed {
        let outer = Arc::new(AgentRegistry::new());
        outer
            .register(SpawnedAgent::from_runtime(
                "middle".into(),
                parent,
                crate::spec::AgentSpec::from_yaml_strict(yaml).unwrap(),
            ))
            .await
            .unwrap();
        let yaml = "name: Outer\nsystem_prompt: outer\nstates:\n  initial: active\n  states:\n    active:\n      delegate: middle\n";
        AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(super::runner_tests::RecordingProvider::new(&["unused"]))
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap()
            .with_spawner_handles(Arc::new(AgentSpawner::new()), outer)
    } else {
        parent
    };
    let parent = Arc::new(parent);
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(parent.info().id, None, "consumer.v1".into()).unwrap(),
    );
    let runner = runner(&parent, store.clone());
    Fixture {
        parent,
        runner,
        store,
        registry,
        main,
        workers,
        gate,
    }
}

// Every fixture keeps normal shared accounting and explicit interaction pauses under the real runner.
fn runner(parent: &Arc<RuntimeAgent>, store: Arc<dyn TaskRunStore>) -> AutonomyRunner {
    let mut config = super::runner_tests::config(24);
    config.defaults.max_tool_calls = Some(20);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    AutonomyRunner::try_new(
        parent.clone(),
        config,
        Default::default(),
        store,
        "consumer.v1".into(),
    )
    .unwrap()
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

// Stored and live histories agree, including the single native response per call after repeated child parks.
async fn assert_finished(f: &Fixture, run: &TaskRun, repeated: bool) {
    assert_eq!(
        run.status,
        TaskRunStatus::Completed,
        "{:?}",
        run.stop_reason
    );
    let rounds = if repeated { 2 } else { 1 };
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 2);
    assert_eq!(f.workers[0].calls.load(Ordering::SeqCst), rounds);
    for provider in &f.workers[1..] {
        assert_eq!(provider.calls.load(Ordering::SeqCst), rounds * 3);
    }
    assert_eq!(run.counters.llm_attempts, (2 + rounds * 7) as u64);
    assert_eq!(run.counters.tool_attempts, (rounds * 5) as u64);
    let payload: TaskCheckpointPayload = serde_json::from_value(
        f.store
            .load(&run.key.run_id)
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
    assert!(
        payload
            .reservations
            .iter()
            .all(|reservation| reservation.state == TaskEffectState::Completed)
    );
    for snapshot in std::iter::once(&payload.runtime.snapshot)
        .chain(payload.children.iter().map(|child| &child.runtime.snapshot))
    {
        let history = ai_agents_core::inspect_native_history(&snapshot.memory.messages).unwrap();
        for exchange in history.exchanges() {
            assert!(exchange.is_complete());
            assert_eq!(
                exchange.result_ids().len(),
                exchange.state().bindings().len()
            );
            assert_eq!(
                exchange.state().model_content(),
                &json!({"signature":exchange.state().exchange_id()})
            );
        }
    }
    let broadcasts: Vec<_> = payload
        .evidence
        .tool_calls
        .iter()
        .filter(|call| call.record.canonical_id == "broadcast")
        .collect();
    assert_eq!(broadcasts.len(), rounds);
    for record in broadcasts {
        let output: Value = serde_json::from_str(&record.record.output).unwrap();
        assert_eq!(
            output
                .as_array()
                .unwrap()
                .iter()
                .map(|row| row["agent_id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
        assert!(
            output
                .as_array()
                .unwrap()
                .iter()
                .all(|row| row["success"] == true)
        );
    }
    for id in ["a", "b", "c"] {
        assert!(
            f.registry
                .get(id)
                .unwrap()
                .set_context("released", json!(true))
                .is_ok()
        );
        assert!(f.registry.remove(id).await.is_some());
    }
}

#[tokio::test]
async fn model_consumer_broadcast_repause_preserves_results_and_rejects_obsolete_answers() {
    let f = Box::pin(fixture(false, false)).await;
    let mut result = f.runner.run("objective", None).await.unwrap();
    let mut pauses = 0;
    while result.run.status == TaskRunStatus::Paused {
        let old = result.run;
        assert!(f.registry.remove("a").await.is_none());
        assert!(
            f.registry
                .get("b")
                .unwrap()
                .set_context("blocked", json!(true))
                .is_err()
        );
        let revision = f
            .store
            .load(&old.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .revision;
        assert!(
            f.runner
                .resume(
                    &old.key.run_id,
                    old.revision,
                    TaskResumeInput::Approval {
                        request_id: old.pending.as_ref().unwrap().id.clone(),
                        result: ai_agents_hitl::ApprovalResult::Approved,
                    }
                )
                .await
                .is_err()
        );
        assert_eq!(
            f.store
                .load(&old.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .revision,
            revision
        );
        result = answer(&f, &old).await.unwrap();
        assert!(answer(&f, &old).await.is_err());
        pauses += 1;
        assert!(pauses <= 4);
    }
    assert_eq!(pauses, 4);
    assert_finished(&f, &result.run, false).await;
}

#[tokio::test]
async fn model_consumer_repeated_broadcast_and_declared_parent_keep_distinct_operations() {
    for mixed in [false, true] {
        let f = Box::pin(fixture(true, mixed)).await;
        let mut result = f.runner.run("objective", None).await.unwrap();
        let mut pauses = 0;
        while result.run.status == TaskRunStatus::Paused {
            result = answer(&f, &result.run).await.unwrap();
            pauses += 1;
            assert!(pauses <= 8);
        }
        assert_eq!(
            pauses,
            8,
            "mixed={mixed}, status={:?}, stop={:?}, records={:?}",
            result.run.status,
            result.run.stop_reason,
            result
                .run
                .evidence
                .tool_calls
                .iter()
                .map(|r| (&r.record.call_id, &r.record.output))
                .collect::<Vec<_>>()
        );
        assert_finished(&f, &result.run, true).await;
    }
}

#[tokio::test]
async fn model_consumer_broadcast_cancel_and_stop_close_only_pending_work() {
    for mixed in [false, true] {
        for stop in [false, true] {
            let f = Box::pin(fixture(false, mixed)).await;
            let first = f.runner.run("objective", None).await.unwrap();
            let next = answer(&f, &first.run).await.unwrap();
            assert_eq!(next.run.status, TaskRunStatus::Paused);
            let before: Vec<_> = f
                .workers
                .iter()
                .map(|p| p.calls.load(Ordering::SeqCst))
                .collect();
            let done = if stop {
                f.runner
                    .stop_paused(&next.run.key.run_id, next.run.revision, "Stop broadcast")
                    .await
                    .unwrap()
            } else {
                f.runner
                    .cancel_paused(&next.run.key.run_id, next.run.revision)
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
            assert_eq!(
                f.workers
                    .iter()
                    .map(|p| p.calls.load(Ordering::SeqCst))
                    .collect::<Vec<_>>(),
                before
            );
            let payload: TaskCheckpointPayload = serde_json::from_value(
                f.store
                    .load(&done.run.key.run_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .payload,
            )
            .unwrap();
            assert!(payload.children.iter().all(|c| c.pending.is_none()));
            for id in ["a", "b", "c"] {
                assert!(f.registry.remove(id).await.is_some());
            }
        }
    }
}

#[tokio::test]
async fn model_consumer_broadcast_drop_during_child_finalization_retains_protection() {
    let f = Box::pin(fixture(false, false)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    let second = answer(&f, &first.run).await.unwrap();
    f.gate.active.store(true, Ordering::SeqCst);
    let mut resume = Box::pin(answer(&f, &second.run));
    tokio::select! {
        outcome = &mut resume => panic!("completed before callback gate: {:?}", outcome.as_ref().err()),
        permit = f.gate.entered.acquire() => permit.unwrap().forget(),
        _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("callback did not start"),
    }
    drop(resume);
    assert!(f.parent.set_context("blocked", json!(true)).is_err());
    for id in ["a", "b", "c"] {
        assert!(f.registry.remove(id).await.is_none());
    }
    assert!(answer(&f, &second.run).await.is_err());
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
}

struct FaultStore {
    inner: Arc<ScopedTaskRunStore>,
    armed: AtomicBool,
    mode: &'static str,
    reached: tokio::sync::Semaphore,
}
#[async_trait]
impl TaskRunStore for FaultStore {
    async fn create(&self, snapshot: &TaskRunSnapshot) -> ai_agents_core::Result<()> {
        self.inner.create(snapshot).await
    }
    async fn load(&self, id: &str) -> ai_agents_core::Result<Option<TaskRunSnapshot>> {
        self.inner.load(id).await
    }
    async fn list(&self) -> ai_agents_core::Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    async fn delete(&self, id: &str, revision: u64) -> ai_agents_core::Result<()> {
        self.inner.delete(id, revision).await
    }
    // The final parent pause write is distinct from a child park or an earlier journal checkpoint.
    async fn mutate(
        &self,
        id: &str,
        mutation: &TaskRunMutation,
    ) -> ai_agents_core::Result<TaskRunSnapshot> {
        if matches!(
            mutation,
            TaskRunMutation::FinalCheckpoint {
                status: TaskRunStatus::Paused,
                release: true,
                ..
            }
        ) && self.armed.swap(false, Ordering::SeqCst)
        {
            if self.mode == "reject" {
                self.reached.add_permits(1);
                return Err(ai_agents_core::AgentError::Persistence(
                    "rejected final broadcast pause".into(),
                ));
            }
            let saved = self.inner.mutate(id, mutation).await?;
            self.reached.add_permits(1);
            if self.mode == "drop" {
                std::future::pending::<()>().await;
            }
            if self.mode == "lost" {
                return Err(ai_agents_core::AgentError::Persistence(
                    "lost final broadcast pause receipt".into(),
                ));
            }
            return Ok(saved);
        }
        self.inner.mutate(id, mutation).await
    }
}

// Positive and failed final-pause acknowledgements preserve exactly the same recipient catalogue on both backends.
#[tokio::test]
async fn model_consumer_broadcast_final_pause_acknowledgement_failures_protect_catalogue() {
    for mode in ["reject", "lost", "drop"] {
        let mut f = Box::pin(fixture(false, true)).await;
        let fault = Arc::new(FaultStore {
            inner: f.store.clone(),
            armed: AtomicBool::new(false),
            mode,
            reached: tokio::sync::Semaphore::new(0),
        });
        f.runner = runner(&f.parent, fault.clone());
        let first = f.runner.run("objective", None).await.unwrap();
        fault.armed.store(true, Ordering::SeqCst);
        let mut resume = Box::pin(answer(&f, &first.run));
        if mode == "drop" {
            tokio::select! {
                outcome = &mut resume => panic!("returned before held receipt: {:?}", outcome.as_ref().err()),
                permit = fault.reached.acquire() => permit.unwrap().forget(),
                _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("receipt boundary not reached"),
            }
            drop(resume);
        } else {
            assert!(resume.await.is_err());
        }
        let saved = f.store.load(&first.run.key.run_id).await.unwrap().unwrap();
        let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload).unwrap();
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.workers[0].calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.workers[1].calls.load(Ordering::SeqCst), 2);
        assert!(
            payload
                .children
                .iter()
                .any(|child| child.runtime.snapshot.agent_id
                    == f.registry.get("a").unwrap().info().id
                    && child.result.is_some())
        );
        assert!(f.parent.set_context("blocked", json!(true)).is_err());
        for id in ["a", "b", "c"] {
            assert!(f.registry.remove(id).await.is_none());
        }
        assert!(answer(&f, &first.run).await.is_err());
        assert!(
            f.runner
                .resume(
                    &first.run.key.run_id,
                    saved.revision,
                    TaskResumeInput::UserAnswer {
                        request_id: first.run.pending.as_ref().unwrap().id.clone(),
                        answer: json!({"answered":true,"selected":["yes"]}),
                    }
                )
                .await
                .is_err()
        );
    }
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn model_consumer_broadcast_sqlite_repause_and_final_receipt_parity() {
    for mode in ["positive", "reject", "lost", "drop"] {
        let mut f = Box::pin(fixture(false, false)).await;
        f.store = Arc::new(
            ScopedTaskRunStore::new(
                Arc::new(ai_agents_storage::SqliteStorage::in_memory().await.unwrap()),
                f.parent.info().id,
                None,
                "consumer.v1".into(),
            )
            .unwrap(),
        );
        let fault = Arc::new(FaultStore {
            inner: f.store.clone(),
            armed: AtomicBool::new(false),
            mode,
            reached: tokio::sync::Semaphore::new(0),
        });
        f.runner = runner(&f.parent, fault.clone());
        let first = f.runner.run("objective", None).await.unwrap();
        fault.armed.store(true, Ordering::SeqCst);
        let mut resume = Box::pin(answer(&f, &first.run));
        if mode == "positive" {
            let mut result = resume.await.unwrap();
            while result.run.status == TaskRunStatus::Paused {
                result = answer(&f, &result.run).await.unwrap();
            }
            assert_finished(&f, &result.run, false).await;
        } else {
            if mode == "drop" {
                tokio::select! {
                    outcome = &mut resume => panic!("receipt not held: {:?}", outcome.as_ref().err()),
                    permit = fault.reached.acquire() => permit.unwrap().forget(),
                    _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("receipt boundary not reached"),
                }
                drop(resume);
            } else {
                assert!(resume.await.is_err());
            }
            assert!(answer(&f, &first.run).await.is_err());
            assert!(f.parent.set_context("blocked", json!(true)).is_err());
            for id in ["a", "b", "c"] {
                assert!(f.registry.remove(id).await.is_none());
            }
            assert_eq!(f.workers[1].calls.load(Ordering::SeqCst), 2);
        }
    }
}

struct SpawnHooks(std::sync::atomic::AtomicUsize);
#[async_trait]
impl crate::spawner::RegistryHooks for SpawnHooks {
    async fn on_agent_spawned(&self, _: &str, _: &crate::spec::AgentSpec) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

// The real generation tool, managed shared providers, registry hooks and capacity survive a generated child's two questions.
#[tokio::test]
async fn model_consumer_generated_child_repause_does_not_repeat_creation_or_registration() {
    for stop in [false, true] {
        let spawn = marker(
            "spawn-signed",
            vec![ToolCall {
                id: "spawn".into(),
                name: "spawn_agent".into(),
                arguments: json!({"name":"Worker","description":"worker"}),
            }],
        );
        let send = marker(
            "send-signed",
            vec![ToolCall {
                id: "send".into(),
                name: "send_agent_message".into(),
                arguments: json!({"to":"worker","message":"work"}),
            }],
        );
        let questions: Vec<_> = (0..2).map(|index| marker(&format!("generated-{index}"), vec![ToolCall {
            id:format!("generated-question-{index}"), name:"ask_user".into(), arguments:json!({"question":"choice","options":["yes"],"allow_other":false}),
        }])).collect();
        let provider = super::runner_tests::RecordingProvider::new(&[
            &spawn,
            "name: Worker\nsystem_prompt: child\ntools: [ask_user]\n",
            &send,
            &questions[0],
            &questions[1],
            "child done",
            "done",
        ]);
        let yaml = "name: GenerateParent\nsystem_prompt: parent\ntools: [spawn_agent, send_agent_message]\n";
        let mut llm_registry = ai_agents_llm::LLMRegistry::new();
        llm_registry.register("default", provider.clone());
        let mut builder = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm_registry(llm_registry)
            .auto_configure_features()
            .unwrap();
        builder.prepare_routing(false).unwrap();
        builder.wrap_llm_registry_for_observability().unwrap();
        builder.prepare_routing(true).unwrap();
        let llms = builder.prepared_registry().unwrap().clone();
        let spawner = Arc::new(
            AgentSpawner::new()
                .with_shared_llms(llms.clone())
                .with_max_agents(1),
        );
        let hooks = Arc::new(SpawnHooks(std::sync::atomic::AtomicUsize::new(0)));
        let registry = Arc::new(AgentRegistry::new().with_hooks(hooks.clone()));
        let parent = Arc::new(
            builder
                .tool(Arc::new(crate::spawner::GenerateAgentTool::new(
                    spawner.clone(),
                    registry.clone(),
                    Arc::new(llms),
                )))
                .tool(Arc::new(crate::spawner::SendMessageTool::new(
                    registry.clone(),
                    "parent",
                )))
                .build()
                .unwrap()
                .with_spawner_handles(spawner.clone(), registry.clone()),
        );
        let store = Arc::new(
            ScopedTaskRunStore::in_memory(parent.info().id, None, "consumer.v1".into()).unwrap(),
        );
        let runner = runner(&parent, store.clone());
        let first = runner.run("objective", None).await.unwrap();
        assert_eq!(
            first.run.status,
            TaskRunStatus::Paused,
            "{:?}",
            first.run.stop_reason
        );
        let child = registry.get("worker").unwrap();
        assert_eq!(hooks.0.load(Ordering::SeqCst), 1);
        assert_eq!(spawner.spawned_count(), 1);
        assert!(registry.remove("worker").await.is_none());
        let second = runner
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
        assert_eq!(second.run.status, TaskRunStatus::Paused);
        assert!(Arc::ptr_eq(&child, &registry.get("worker").unwrap()));
        assert_eq!(provider.calls.load(Ordering::SeqCst), 5);
        let done = if stop {
            runner
                .cancel_paused(&second.run.key.run_id, second.run.revision)
                .await
                .unwrap()
        } else {
            runner
                .resume(
                    &second.run.key.run_id,
                    second.run.revision,
                    TaskResumeInput::UserAnswer {
                        request_id: second.run.pending.as_ref().unwrap().id.clone(),
                        answer: json!({"answered":true,"selected":["yes"]}),
                    },
                )
                .await
                .unwrap()
        };
        assert_eq!(
            done.run.status,
            if stop {
                TaskRunStatus::Cancelled
            } else {
                TaskRunStatus::Completed
            }
        );
        assert_eq!(
            provider.calls.load(Ordering::SeqCst),
            if stop { 5 } else { 7 }
        );
        assert_eq!(hooks.0.load(Ordering::SeqCst), 1);
        assert_eq!(spawner.spawned_count(), 1);
        let payload: TaskCheckpointPayload = serde_json::from_value(
            store
                .load(&done.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .payload,
        )
        .unwrap();
        assert_eq!(
            payload
                .evidence
                .tool_calls
                .iter()
                .filter(|call| call.record.canonical_id == "spawn_agent")
                .count(),
            1
        );
        assert!(registry.remove("worker").await.is_some());
        assert_eq!(spawner.spawned_count(), 0);
    }
}

// Optional failure remains a typed result through later pauses; required failure never becomes successful parent completion.
#[tokio::test]
async fn model_consumer_broadcast_required_and_optional_failure_keep_original_policy() {
    for required in [false, true] {
        let f = Box::pin(fixture_options(false, false, required, None)).await;
        let bad = AgentBuilder::new()
            .system_prompt("bad")
            .llm(Arc::new(super::child_tests::FailingProvider))
            .build()
            .unwrap();
        f.registry
            .register(SpawnedAgent::from_runtime(
                "bad".into(),
                bad,
                crate::spec::AgentSpec::default(),
            ))
            .await
            .unwrap();
        let mut result = f.runner.run("objective", None).await.unwrap();
        let mut count = 0;
        while result.run.status == TaskRunStatus::Paused {
            result = answer(&f, &result.run).await.unwrap();
            count += 1;
            assert!(count <= 4);
        }
        if required {
            assert_ne!(result.run.status, TaskRunStatus::Completed);
            assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
        } else {
            assert_eq!(
                result.run.status,
                TaskRunStatus::Completed,
                "{:?}",
                result.run.stop_reason
            );
            assert_eq!(count, 4);
            let record = result
                .run
                .evidence
                .tool_calls
                .iter()
                .find(|call| call.record.canonical_id == "broadcast")
                .unwrap();
            let output: Value = serde_json::from_str(&record.record.output).unwrap();
            assert_eq!(
                output
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|row| row["agent_id"].as_str().unwrap())
                    .collect::<Vec<_>>(),
                vec!["a", "b", "bad", "c"]
            );
            assert_eq!(output[2]["success"], false);
            assert_eq!(f.workers[1].calls.load(Ordering::SeqCst), 3);
            assert_eq!(f.workers[2].calls.load(Ordering::SeqCst), 3);
        }
        for id in ["a", "b", "bad", "c"] {
            assert!(f.registry.remove(id).await.is_some());
        }
    }
}

// Re-parking does not refresh the original broadcast timeout, and expiry closes history without another recipient invocation.
#[tokio::test]
async fn model_consumer_broadcast_original_deadline_survives_repause() {
    let f = Box::pin(fixture_options(false, false, true, Some(5000))).await;
    let first = f.runner.run("objective", None).await.unwrap();
    assert_eq!(first.run.status, TaskRunStatus::Paused);
    let original = f.store.load(&first.run.key.run_id).await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(original.payload).unwrap();
    let group: TaskGroupState = serde_json::from_value(
        payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == "runtime.group")
            .unwrap()
            .state
            .clone(),
    )
    .unwrap();
    let expiry = group.frame.expires_at.unwrap();
    let next = answer(&f, &first.run).await.unwrap();
    assert_eq!(next.run.status, TaskRunStatus::Paused);
    let snapshot = f.store.load(&next.run.key.run_id).await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
    let group: TaskGroupState = serde_json::from_value(
        payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == "runtime.group")
            .unwrap()
            .state
            .clone(),
    )
    .unwrap();
    assert_eq!(group.frame.expires_at, Some(expiry));
    let before: Vec<_> = f
        .workers
        .iter()
        .map(|provider| provider.calls.load(Ordering::SeqCst))
        .collect();
    tokio::time::sleep(
        (expiry - chrono::Utc::now()).to_std().unwrap_or_default()
            + std::time::Duration::from_millis(10),
    )
    .await;
    let stopped = answer(&f, &next.run).await.unwrap();
    assert_eq!(
        stopped.run.status,
        TaskRunStatus::Incomplete,
        "{:?}",
        stopped.run.stop_reason
    );
    assert_eq!(
        stopped.run.stop_reason.as_deref(),
        Some("composition_timeout")
    );
    assert_eq!(
        f.workers
            .iter()
            .map(|provider| provider.calls.load(Ordering::SeqCst))
            .collect::<Vec<_>>(),
        before
    );
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
    for id in ["a", "b", "c"] {
        assert!(f.registry.remove(id).await.is_some());
    }
}

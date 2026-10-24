//! Cross-consumer composition contracts use actual registries, managed providers and acknowledged task state.

use super::*;
use crate::spawner::{AgentRegistry, AgentSpawner, SpawnedAgent};
use crate::{AgentBuilder, RuntimeAgent};
use ai_agents_core::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
};
use async_trait::async_trait;
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

pub(super) struct Script {
    replies: parking_lot::Mutex<std::collections::VecDeque<Option<String>>>,
    pub(super) calls: AtomicUsize,
    pub(super) inputs: parking_lot::Mutex<Vec<Vec<ChatMessage>>>,
    gate_at: usize,
    pub(super) entered: tokio::sync::Semaphore,
    pub(super) release: tokio::sync::Semaphore,
}
impl Script {
    // None is a terminal provider failure; gates acknowledge an actual dispatch rather than elapsed test time.
    pub(super) fn new(replies: &[Option<&str>], gate_at: usize) -> Arc<Self> {
        Arc::new(Self {
            replies: parking_lot::Mutex::new(
                replies
                    .iter()
                    .map(|reply| reply.map(str::to_owned))
                    .collect(),
            ),
            calls: AtomicUsize::new(0),
            inputs: Default::default(),
            gate_at,
            entered: tokio::sync::Semaphore::new(0),
            release: tokio::sync::Semaphore::new(0),
        })
    }
}
#[async_trait]
impl LLMProvider for Script {
    // Record and gate the exact managed provider request, including auxiliary selection and aggregation calls.
    async fn complete(
        &self,
        messages: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        self.inputs.lock().push(messages.to_vec());
        if call == self.gate_at {
            self.entered.add_permits(1);
            self.release.acquire().await.unwrap().forget();
        }
        match self
            .replies
            .lock()
            .pop_front()
            .unwrap_or(Some("done".into()))
        {
            Some(reply) => Ok(LLMResponse::new(reply, FinishReason::Stop)),
            None => Err(LLMError::Config("injected settled provider failure".into())),
        }
    }
    // These blocking fixtures fail if accidentally dispatched as streams.
    async fn complete_stream(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<
        Box<dyn futures::Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
        LLMError,
    > {
        Err(LLMError::Config("unexpected stream".into()))
    }
    fn provider_name(&self) -> &str {
        "composition-script"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
    fn is_terminal_error(&self, _: &LLMError) -> bool {
        true
    }
}

// A tool request is only a proposal; the production executor and AskUserTool establish suspension evidence.
pub(super) fn question() -> String {
    ai_agents_core::encode_native_tool_call_markers(
        &[ai_agents_core::ToolCall {
            id: "choice".into(),
            name: "ask_user".into(),
            arguments: json!({"question":"Confirm evidence","options":["yes"],"allow_other":false}),
        }],
        None,
    )
    .unwrap()
}

// Registry entries retain real RuntimeAgent gates; generated and declared cases share the same admission contract.
pub(super) async fn register(
    registry: &AgentRegistry,
    id: &str,
    agent: RuntimeAgent,
    yaml: &str,
) -> Arc<RuntimeAgent> {
    registry
        .register(SpawnedAgent::from_runtime(
            id.into(),
            agent,
            crate::spec::AgentSpec::from_yaml_strict(yaml).unwrap(),
        ))
        .await
        .unwrap();
    registry.get(id).unwrap()
}

// Task configuration stays explicit and does not enable automatic YAML entry or require task tools.
pub(super) fn runner(root: Arc<RuntimeAgent>) -> (Arc<AutonomyRunner>, Arc<ScopedTaskRunStore>) {
    let store =
        Arc::new(ScopedTaskRunStore::in_memory(root.info().id, None, "co.v1".into()).unwrap());
    (runner_with_store(root, store.clone()), store)
}

// Backend parity and fault adapters change only storage, never the runner's admission or interaction settings.
pub(super) fn runner_with_store(
    root: Arc<RuntimeAgent>,
    store: Arc<dyn TaskRunStore>,
) -> Arc<AutonomyRunner> {
    let mut config = super::runner_tests::config(32);
    config.defaults.max_tool_calls = Some(24);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_user_question: Some(InteractionAction::PauseRun),
        on_approval_required: Some(InteractionAction::PauseRun),
    });
    Arc::new(
        AutonomyRunner::try_new(root, config, Default::default(), store, "co.v1".into()).unwrap(),
    )
}

// SQLite uses an isolated in-memory database but still exercises conditional backend reads and acknowledgements.
pub(super) async fn runner_backend(
    root: Arc<RuntimeAgent>,
    sqlite: bool,
) -> (Arc<AutonomyRunner>, Arc<ScopedTaskRunStore>) {
    let store = super::continuation_test_support::store(root.info().id, "co.v1", sqlite).await;
    (runner_with_store(root, store.clone()), store)
}

// Answer only the currently advertised coordinating request; no child ID is treated as executable authority.
pub(super) async fn answer(runner: &AutonomyRunner, run: &TaskRun) -> TaskRunResult {
    runner
        .resume(
            &run.key.run_id,
            run.revision,
            TaskResumeInput::UserAnswer {
                request_id: run.pending.as_ref().unwrap().id.clone(),
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await
        .unwrap()
}

// Duplicate catalogue entries are separate incoming operations, even when one parks while the other awaits its gate.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn co_completion_duplicate_concurrent_slots_resume_or_cancel_without_false_failure() {
    for (sqlite, cancel) in super::continuation_test_support::backends()
        .iter()
        .flat_map(|sqlite| [(*sqlite, false), (*sqlite, true)])
    {
        let q = question();
        let provider = Script::new(
            &[Some(&q), Some("first evidence"), Some("second evidence")],
            1,
        );
        let registry = Arc::new(AgentRegistry::new());
        let yaml = "name: Worker\nsystem_prompt: evidence\ntools: [ask_user]\n";
        let child = register(
            &registry,
            "worker",
            AgentBuilder::from_yaml(yaml)
                .unwrap()
                .llm(provider.clone())
                .auto_configure_features()
                .unwrap()
                .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
                    Arc::new(parking_lot::RwLock::new(None)),
                )))
                .build()
                .unwrap(),
            yaml,
        )
        .await;
        let root_yaml = "name: Research\nsystem_prompt: research\nstates:\n  initial: collect\n  states:\n    collect:\n      concurrent:\n        agents: [worker, worker]\n        aggregation:\n          strategy: all\n";
        let root = Arc::new(
            AgentBuilder::from_yaml(root_yaml)
                .unwrap()
                .llm(Script::new(&[], 0))
                .auto_configure_features()
                .unwrap()
                .build()
                .unwrap()
                .with_spawner_handles(Arc::new(AgentSpawner::new()), registry.clone()),
        );
        let (runner, store) = runner_backend(root.clone(), sqlite).await;
        let mut start = Box::pin(runner.run("collect evidence", None));
        // Poll the whole parent so both JoinSet slots can contend while the first provider is explicitly held.
        tokio::select! {
            result = &mut start => panic!("completed before provider release: {:?}", result.err()),
            permit = provider.entered.acquire() => permit.unwrap().forget(),
        }
        // Actual admitted ownership, not a fabricated pending record, protects the first invocation.
        assert!(child.set_context("must_remain_owned", json!(true)).is_err());
        tokio::task::yield_now().await;
        provider.release.add_permits(1);
        let paused = tokio::time::timeout(std::time::Duration::from_secs(10), start)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            paused.run.status,
            TaskRunStatus::Paused,
            "{:?}",
            paused.run.stop_reason
        );
        let done = if cancel {
            runner
                .cancel_paused(&paused.run.key.run_id, paused.run.revision)
                .await
                .unwrap()
        } else {
            answer(&runner, &paused.run).await
        };
        assert_eq!(
            done.run.status,
            if cancel {
                TaskRunStatus::Cancelled
            } else {
                TaskRunStatus::Completed
            },
            "{:?}",
            done.run.stop_reason
        );
        let saved = store.load(&done.run.key.run_id).await.unwrap().unwrap();
        assert_eq!(
            provider.calls.load(Ordering::SeqCst),
            if cancel { 1 } else { 3 },
            "sqlite={sqlite} cancel={cancel}: orchestration={} evidence={:?}",
            saved.payload["runtime"]["snapshot"]["context"]["orchestration"],
            done.run
                .evidence
                .tool_calls
                .iter()
                .map(|call| &call.record.output)
                .collect::<Vec<_>>()
        );
        let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload).unwrap();
        assert!(
            payload
                .children
                .iter()
                .all(|child| child.result.is_some() && child.pending.is_none())
        );
        if !cancel {
            assert_eq!(payload.children.len(), 2);
            assert!(
                payload.children.iter().all(|child| child
                    .result
                    .as_ref()
                    .unwrap()
                    .get("response")
                    .is_some())
            );
        }
        assert!(root.set_context("released", json!(true)).is_ok());
        assert!(registry.remove("worker").await.is_some());
    }
}

struct MessageGate {
    entered: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
}
#[async_trait]
impl crate::spawner::RegistryHooks for MessageGate {
    // The sender hook is an awaited preparation boundary before actual child admission.
    async fn on_message_sent(&self, _: &str, _: &str, _: &str) {
        self.entered.add_permits(1);
        self.release.acquire().await.unwrap().forget();
    }
}

// Blocking SDK messages must retain actual registry membership while awaiting hooks, not merely a stale runtime Arc.
#[tokio::test]
async fn co_completion_direct_task_message_pins_registration_before_sender_hook() {
    let (parent, execution) = super::child_tests::execution(4).await;
    let hook = Arc::new(MessageGate {
        entered: tokio::sync::Semaphore::new(0),
        release: tokio::sync::Semaphore::new(0),
    });
    let registry = Arc::new(AgentRegistry::new().with_hooks(hook.clone()));
    let provider = Script::new(&[Some("ticket evidence")], 0);
    let yaml = "name: Worker\nsystem_prompt: support\n";
    let child = register(
        &registry,
        "worker",
        AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(provider.clone())
            .build()
            .unwrap(),
        yaml,
    )
    .await;
    let mut send = Box::pin(scope_execution(
        execution.clone(),
        registry.send("parent", "worker", "resolve ticket"),
    ));
    tokio::select! {
        result = &mut send => panic!("returned before sender hook: {:?}", result.err()),
        permit = hook.entered.acquire() => permit.unwrap().forget(),
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert!(
        registry.remove("worker").await.is_none(),
        "task target disappeared during sender preparation"
    );
    assert!(Arc::ptr_eq(&child, &registry.get("worker").unwrap()));
    hook.release.add_permits(1);
    send.await.unwrap();
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert!(registry.remove("worker").await.is_none());
    super::child_tests::finish(&parent, &execution).await;
    assert!(registry.remove("worker").await.is_some());
}

// Auxiliary agents belong to a task-scoped SDK catalogue even before their first manager or synthesis turn.
#[tokio::test]
async fn co_completion_sdk_group_pins_unstarted_manager_and_synthesizer() {
    for kind in ["debate", "manager"] {
        let (parent, execution) = super::child_tests::execution(4).await;
        let registry = Arc::new(AgentRegistry::new());
        let first = Script::new(&[Some("verified argument")], 1);
        let auxiliary = Script::new(
            &[Some(if kind == "debate" {
                "balanced evidence"
            } else {
                r#"{"action":"end","reason":"verified"}"#
            })],
            0,
        );
        for (id, provider) in [("worker", first.clone()), ("aux", auxiliary.clone())] {
            let yaml = format!("name: {id}\nsystem_prompt: domain worker\n");
            register(
                &registry,
                id,
                AgentBuilder::from_yaml(&yaml)
                    .unwrap()
                    .llm(provider)
                    .build()
                    .unwrap(),
                &yaml,
            )
            .await;
        }
        let yaml = if kind == "debate" {
            "participants:\n  - id: worker\nstyle: debate\ndebate:\n  rounds: 1\n  synthesizer: aux\n"
        } else {
            "participants:\n  - id: worker\nmax_rounds: 1\nmanager:\n  agent: aux\ntermination:\n  method: manager_decides\n"
        };
        let config: ai_agents_state::GroupChatStateConfig = serde_yaml::from_str(yaml).unwrap();
        let mut work = Box::pin(scope_execution(
            execution.clone(),
            crate::orchestration::group_chat(&registry, "evaluate evidence", &config, None, None),
        ));
        tokio::select! {
            result = &mut work => panic!("returned before participant: {:?}", result.err()),
            permit = first.entered.acquire() => permit.unwrap().forget(),
        }
        assert_eq!(auxiliary.calls.load(Ordering::SeqCst), 0);
        assert!(
            registry.remove("aux").await.is_none(),
            "{kind}: auxiliary registration was not pinned"
        );
        first.release.add_permits(1);
        work.await.unwrap();
        assert_eq!(auxiliary.calls.load(Ordering::SeqCst), 1, "{kind}");
        assert!(registry.remove("aux").await.is_none());
        super::child_tests::finish(&parent, &execution).await;
        assert!(registry.remove("aux").await.is_some());
        assert!(registry.remove("worker").await.is_some());
    }
}

// SDK catalogues may gain a previously absent stage; its first actual lookup must pin it before root-gate admission.
#[tokio::test]
async fn co_completion_sdk_pipeline_pins_late_registered_stage_before_gate_wait() {
    let (parent, execution) = super::child_tests::execution(4).await;
    let registry = Arc::new(AgentRegistry::new());
    let first = Script::new(&[Some("first evidence")], 1);
    let late_provider = Script::new(&[Some("late evidence")], 0);
    register(
        &registry,
        "first",
        AgentBuilder::new()
            .system_prompt("first")
            .llm(first.clone())
            .build()
            .unwrap(),
        "name: First\nsystem_prompt: first\n",
    )
    .await;
    let stages = vec![
        crate::orchestration::PipelineStage::id("first"),
        crate::orchestration::PipelineStage::id("late"),
    ];
    let mut work = Box::pin(scope_execution(
        execution.clone(),
        crate::orchestration::pipeline(&registry, "evidence", &stages, None, None, None),
    ));
    tokio::select! {
        result = &mut work => panic!("returned before first stage: {:?}", result.err()),
        permit = first.entered.acquire() => permit.unwrap().forget(),
    }
    let late = register(
        &registry,
        "late",
        AgentBuilder::new()
            .system_prompt("late")
            .llm(late_provider.clone())
            .build()
            .unwrap(),
        "name: Late\nsystem_prompt: late\n",
    )
    .await;
    // A released fixture owner supplies only a gate latch; it is not authority for the incoming task.
    let latch = late
        .reserve_autonomy_run("fixture-latch".into())
        .await
        .unwrap();
    late.release_autonomy_run(&latch).await.unwrap();
    let blocked = latch.gate.lock().await;
    first.release.add_permits(1);
    assert!(futures::poll!(work.as_mut()).is_pending());
    let snapshot = execution.acknowledged_snapshot();
    assert!(
        snapshot.payload["children"]
            .as_array()
            .unwrap()
            .iter()
            .any(|child| child["result"].is_object())
    );
    assert_eq!(late_provider.calls.load(Ordering::SeqCst), 0);
    assert!(registry.remove("late").await.is_none());
    drop(blocked);
    let result = work.await.unwrap();
    assert_eq!(result.stage_outputs.len(), 2);
    assert_eq!(late_provider.calls.load(Ordering::SeqCst), 1);
    super::child_tests::finish(&parent, &execution).await;
    assert!(registry.remove("first").await.is_some());
    assert!(registry.remove("late").await.is_some());
}

struct DelegateGate {
    entered: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
}
#[async_trait]
impl ai_agents_hooks::AgentHooks for DelegateGate {
    // Hold the successor before its declared frame exists, until its predecessor has a durable failed outcome.
    async fn on_delegate_start(&self, _: &str, _: &str) {
        self.entered.add_permits(1);
        self.release.acquire().await.unwrap().forget();
    }
}

// Settled optional failure must retire its nested dispatch, not leave the shared worker permanently blocked.
#[tokio::test]
async fn co_completion_optional_nested_failure_releases_successor_dependency() {
    for &sqlite in super::continuation_test_support::backends() {
        Box::pin(optional_failure_contract(sqlite)).await;
    }
}

// Acknowledged failure retirement and successor readiness must agree across backend implementations.
async fn optional_failure_contract(sqlite: bool) {
    let provider = Script::new(&[None, Some("verified successor evidence")], 0);
    let workers = Arc::new(AgentRegistry::new());
    let yaml = "name: Worker\nsystem_prompt: domain evidence\n";
    let worker = register(
        &workers,
        "worker",
        AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(provider.clone())
            .build()
            .unwrap(),
        yaml,
    )
    .await;
    let parents = Arc::new(AgentRegistry::new());
    let gate = Arc::new(DelegateGate {
        entered: tokio::sync::Semaphore::new(0),
        release: tokio::sync::Semaphore::new(0),
    });
    for id in ["first", "second"] {
        let yaml = format!(
            "name: {id}\nsystem_prompt: parent\nstates:\n  initial: work\n  states:\n    work:\n      delegate: worker\n"
        );
        let mut builder = AgentBuilder::from_yaml(&yaml)
            .unwrap()
            .llm(Script::new(&[], 0))
            .auto_configure_features()
            .unwrap();
        if id == "second" {
            builder = builder.hooks(gate.clone());
        }
        register(
            &parents,
            id,
            builder
                .build()
                .unwrap()
                .with_spawner_handles(Arc::new(AgentSpawner::new()), workers.clone()),
            &yaml,
        )
        .await;
    }
    let yaml = "name: OptionalResearch\nsystem_prompt: research\nstates:\n  initial: work\n  states:\n    work:\n      concurrent:\n        agents: [first, second]\n        on_partial_failure: proceed_with_available\n        aggregation:\n          strategy: all\n";
    let root = Arc::new(
        AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(Script::new(&[], 0))
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap()
            .with_spawner_handles(Arc::new(AgentSpawner::new()), parents),
    );
    let (runner, store) = runner_backend(root.clone(), sqlite).await;
    let start_runner = runner.clone();
    let mut start =
        tokio::spawn(async move { start_runner.run("collect available evidence", None).await });
    let failed = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            tokio::select! {
                result = &mut start => panic!("premature return: {:?}", result.err()),
                _ = tokio::time::sleep(std::time::Duration::from_millis(5)) => {},
            }
            if let Some(run) = store.list().await.unwrap().first() {
                let saved = store.load(&run.key.run_id).await.unwrap().unwrap();
                if saved.payload["children"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|child| {
                        child["runtime"]["snapshot"]["agent_id"] == "first"
                            && child["result"]["error"].is_string()
                    })
                {
                    break saved.payload["children"].clone();
                }
            }
        }
    })
    .await;
    let failed = match failed {
        Ok(failed) => failed,
        Err(error) => {
            start.abort();
            panic!(
                "failure ACK not observed sqlite={sqlite} calls={} error={error:?}",
                provider.calls.load(Ordering::SeqCst)
            );
        }
    };
    gate.entered.acquire().await.unwrap().forget();
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    gate.release.add_permits(1);
    let done = tokio::time::timeout(std::time::Duration::from_secs(10), start)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    let saved = store.load(&done.run.key.run_id).await.unwrap().unwrap();
    for original in failed
        .as_array()
        .unwrap()
        .iter()
        .filter(|child| !child["result"].is_null())
    {
        let current = saved.payload["children"]
            .as_array()
            .unwrap()
            .iter()
            .find(|child| child["child_id"] == original["child_id"])
            .unwrap();
        assert_eq!(current, original);
    }
    assert!(worker.set_context("released", json!(true)).is_ok());
    assert!(root.set_context("released", json!(true)).is_ok());
    assert!(workers.remove("worker").await.is_some());
}

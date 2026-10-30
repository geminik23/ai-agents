use super::*;

#[path = "ad_branch_closure_tests.rs"]
mod branch_closure;

struct DraftGateMemory {
    inner: Arc<dyn ai_agents_memory::Memory>,
    armed: AtomicBool,
    entered: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
}

#[async_trait]
impl ai_agents_core::Memory for DraftGateMemory {
    async fn add_message(&self, message: ChatMessage) -> Result<()> {
        self.inner.add_message(message).await
    }
    async fn get_messages(&self, limit: Option<usize>) -> Result<Vec<ChatMessage>> {
        self.inner.get_messages(limit).await
    }
    async fn clear(&self) -> Result<()> {
        self.inner.clear().await
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    async fn snapshot(&self) -> Result<ai_agents_core::MemorySnapshot> {
        self.inner.snapshot().await
    }
    async fn restore(&self, snapshot: ai_agents_core::MemorySnapshot) -> Result<()> {
        self.inner.restore(snapshot).await
    }
    async fn evict_oldest(&self, count: usize) -> Result<Vec<ChatMessage>> {
        self.inner.evict_oldest(count).await
    }
}

#[async_trait]
impl ai_agents_memory::Memory for DraftGateMemory {
    fn supports_task_admission(&self) -> bool {
        self.inner.supports_task_admission()
    }
    async fn get_context(&self) -> Result<ai_agents_memory::ConversationContext> {
        if self.armed.swap(false, Ordering::SeqCst) {
            self.entered.add_permits(1);
            self.release.acquire().await.unwrap().forget();
        }
        self.inner.get_context().await
    }
    async fn compress(
        &self,
        summarizer: Option<&dyn ai_agents_memory::Summarizer>,
    ) -> Result<ai_agents_memory::CompressResult> {
        self.inner.compress(summarizer).await
    }
    fn needs_compression(&self) -> bool {
        self.inner.needs_compression()
    }
}

// The production runner owns actual private authority; reviews do not grant another invocation budget.
fn reviewed_runner(agent: Arc<RuntimeAgent>) -> AutonomyRunner {
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info.id.clone(), None, "ad-completion".into()).unwrap(),
    );
    let config = AutonomyConfig {
        defaults: AutonomyProfile {
            enabled: Some(true),
            max_turns: Some(4),
            max_llm_calls: Some(8),
            completion: Some(CompletionGate::ResponseContains("done".into())),
            hitl: Some(crate::autonomy::AutonomyHitlConfig {
                on_approval_required: Some(crate::autonomy::InteractionAction::PauseRun),
                on_user_question: Some(crate::autonomy::InteractionAction::PauseRun),
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    AutonomyRunner::try_new(
        agent,
        config,
        AutonomyHostCeilings::default(),
        store,
        "ad-completion".into(),
    )
    .unwrap()
}

// Saved root bookkeeping is historical data, but resumption must retain processed values and already consumed local speculative slots.
#[test]
fn ad_speculative_saved_context_keeps_values_and_local_allowance() {
    let mut runtime = crate::AgentBuilder::new()
        .system_prompt("test")
        .llm(Provider::new(&["unused"], false))
        .build()
        .unwrap();
    runtime.runtime_config.optimization.enabled = true;
    runtime
        .runtime_config
        .optimization
        .max_speculative_llm_calls_per_turn = 2;
    runtime.begin_root_turn();
    let data = ProcessData::new("processed once")
        .with_context("extracted", serde_json::json!({"value":42}));
    runtime.update_active_turn_context(&data.content, data.context.clone());
    assert!(
        runtime
            .reserve_active_speculative_llm_call(RuntimeOptimizationKind::ParallelStateTransition)
    );
    assert!(
        runtime
            .reserve_active_speculative_llm_call(RuntimeOptimizationKind::SpeculativeSkillRouting)
    );
    let selection = crate::autonomy::location::TaskSpeculativeSelection {
        data,
        turn: runtime.active_turn_context.read().clone().map(Box::new),
        draft: None,
        draft_error: None,
        draft_uninvoked: true,
        skill: None,
        skill_checked: true,
        reasoning: None,
        reasoning_checked: true,
        auto_detected: false,
        branches: Vec::new(),
    };
    let saved: crate::autonomy::location::TaskSpeculativeSelection =
        serde_json::from_value(serde_json::to_value(&selection).unwrap()).unwrap();
    runtime.end_root_turn();
    runtime.begin_root_turn();
    runtime.restore_speculative_context(&saved);
    let turn = runtime.active_turn_context.read().clone().unwrap();
    assert_eq!(turn.turn_id, saved.turn.as_ref().unwrap().turn_id);
    assert_eq!(
        turn.input_context["extracted"],
        serde_json::json!({"value":42})
    );
    assert_eq!(turn.speculative_llm_calls_used, 2);
    assert!(
        !runtime
            .reserve_active_speculative_llm_call(RuntimeOptimizationKind::SpeculativeReasoningAuto)
    );
}

// A draft blocked before provider admission is unfinished, not a failed completed call, when a selected transition is later rejected.
#[tokio::test]
async fn ad_speculative_rejected_review_executes_only_uninvoked_draft() {
    let main = Provider::new(&["done"], false);
    let routing = Provider::new(&["1"], false);
    let yaml = "name: Route\nsystem_prompt: test\nllm:\n  default: default\n  router: router\nruntime:\n  optimization:\n    enabled: true\n    max_speculative_llm_calls_per_turn: 2\n    max_parallel_runtime_tasks: 2\n    speculative_state_transitions: true\nhitl:\n  message_language:\n    strategy: explicit\n    explicit: en\n  states:\n    ready:\n      on_enter: require_approval\nstates:\n  initial: active\n  states:\n    active:\n      transitions:\n        - to: ready\n          when: choose ready\n          timing: parallel\n          requires_response: false\n    ready:\n      regenerate_on_enter: false\ntools: []\n";
    let mut runtime = crate::AgentBuilder::from_yaml(yaml)
        .unwrap()
        .llm(main.clone())
        .llm_alias("router", routing.clone())
        .auto_configure_features()
        .unwrap()
        .build()
        .unwrap();
    let memory = Arc::new(DraftGateMemory {
        inner: runtime.memory.clone(),
        armed: AtomicBool::new(true),
        entered: tokio::sync::Semaphore::new(0),
        release: tokio::sync::Semaphore::new(0),
    });
    runtime.memory = memory.clone();
    let agent = Arc::new(runtime);
    let runner = reviewed_runner(agent.clone());
    let mut work = Box::pin(runner.run("objective", None));
    tokio::select! { result = &mut work => panic!("draft never reached its pre-admission gate: {:?}", result.err()), permit = memory.entered.acquire() => permit.unwrap().forget() }
    assert!(futures::poll!(&mut work).is_pending());
    assert_eq!(routing.calls.load(Ordering::SeqCst), 1);
    assert_eq!(main.calls.load(Ordering::SeqCst), 0);
    memory.release.add_permits(1);
    let paused = work.await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    assert_eq!(main.calls.load(Ordering::SeqCst), 0);
    assert_eq!(paused.run.counters.llm_attempts, 1);
    let result = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            crate::autonomy::TaskResumeInput::Approval {
                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                result: ai_agents_hitl::ApprovalResult::Rejected {
                    reason: Some("keep current state".into()),
                },
            },
        )
        .await
        .unwrap();
    assert_eq!(
        result.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        result.run.stop_reason
    );
    assert_eq!(main.calls.load(Ordering::SeqCst), 1);
    assert_eq!(routing.calls.load(Ordering::SeqCst), 1);
    assert_eq!(result.run.counters.llm_attempts, 2);
    assert_eq!(agent.current_state().as_deref(), Some("active"));
}

// A directly winning tool draft enters the existing exact model-batch loop before approval can suspend it.
#[tokio::test]
async fn ad_speculative_tool_draft_approval_retains_batch_without_provider_replay() {
    let main = Provider::new(
        &[r#"{"tool":"echo","arguments":{"message":"work"}}"#, "done"],
        false,
    );
    let routing = Provider::new(&["none"], false);
    let yaml = "name: Draft\nsystem_prompt: test\nllm:\n  default: default\n  router: router\nruntime:\n  optimization:\n    enabled: true\n    max_speculative_llm_calls_per_turn: 2\n    max_parallel_runtime_tasks: 2\n    speculative_skill_routing: true\nhitl:\n  message_language:\n    strategy: explicit\n    explicit: en\n  tools:\n    echo:\n      require_approval: true\nskills:\n  - id: unused\n    trigger: unrelated work\n    description: unused\n    steps:\n      - prompt: unused\ntools: [echo]\n";
    let agent = Arc::new(
        crate::AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(main.clone())
            .llm_alias("router", routing.clone())
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap(),
    );
    let runner = reviewed_runner(agent);
    let paused = runner.run("objective", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    assert_eq!(main.calls.load(Ordering::SeqCst), 1);
    assert_eq!(paused.run.counters.tool_attempts, 0);
    let result = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            crate::autonomy::TaskResumeInput::Approval {
                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                result: ai_agents_hitl::ApprovalResult::Approved,
            },
        )
        .await
        .unwrap();
    assert_eq!(
        result.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        result.run.stop_reason
    );
    assert_eq!(main.calls.load(Ordering::SeqCst), 2);
    assert_eq!(routing.calls.load(Ordering::SeqCst), 1);
    assert_eq!(result.run.counters.tool_attempts, 1);
}

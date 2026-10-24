use super::*;

#[path = "ad_completion_tests.rs"]
mod completion;
#[path = "ad_resource_tests.rs"]
mod resources;
use crate::autonomy::{
    AutonomyConfig, AutonomyHostCeilings, AutonomyProfile, AutonomyRunner, CompletionGate,
    ScopedTaskRunStore, TaskEffectState, TaskRunStatus, TaskRunStore,
};
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::atomic::AtomicUsize;

struct Provider {
    replies: parking_lot::Mutex<VecDeque<String>>,
    requests: parking_lot::Mutex<Vec<Vec<ChatMessage>>>,
    calls: AtomicUsize,
    entered: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
    gated: bool,
    priced: bool,
}

impl Provider {
    // Record the actual managed dispatch, including losing work that consumes no winning response.
    fn new(replies: &[&str], gated: bool) -> Arc<Self> {
        Arc::new(Self {
            replies: parking_lot::Mutex::new(replies.iter().map(|s| s.to_string()).collect()),
            requests: Default::default(),
            calls: AtomicUsize::new(0),
            entered: tokio::sync::Semaphore::new(0),
            release: tokio::sync::Semaphore::new(0),
            gated,
            priced: false,
        })
    }
}

impl Provider {
    // Synthetic fixed-point quotes exercise real managed pricing without constructing or calling a paid provider.
    fn priced(replies: &[&str]) -> Arc<Self> {
        let mut provider = Self::new(replies, false);
        Arc::get_mut(&mut provider).unwrap().priced = true;
        provider
    }
}

#[async_trait]
impl LLMProvider for Provider {
    fn priced_capability_identity(&self) -> Option<String> {
        self.priced.then(|| "ad-test-price-v1".into())
    }
    fn request_cost_bound(
        &self,
        request: &ai_agents_core::autonomy::ProviderRequest<'_>,
    ) -> std::result::Result<Option<ai_agents_core::autonomy::ProviderCostBound>, LLMError> {
        Ok((self.priced && !request.streaming).then(|| {
            ai_agents_core::autonomy::ProviderCostBound {
                provider_identity: "ad-witness".into(),
                pricing_identity: "ad-test-price-v1".into(),
                request_id: request.request_id.into(),
                max_micro_usd: 2,
            }
        }))
    }
    fn settle_request_cost(
        &self,
        bound: &ai_agents_core::autonomy::ProviderCostBound,
        _: &ai_agents_core::LLMResponse,
    ) -> std::result::Result<Option<ai_agents_core::autonomy::ProviderCostSettlement>, LLMError>
    {
        Ok(self
            .priced
            .then(|| ai_agents_core::autonomy::ProviderCostSettlement {
                bound: bound.clone(),
                charged_micro_usd: 1,
            }))
    }
    async fn complete(
        &self,
        messages: &[ChatMessage],
        _: Option<&ai_agents_core::LLMConfig>,
    ) -> std::result::Result<ai_agents_core::LLMResponse, LLMError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.requests.lock().push(messages.to_vec());
        if self.gated {
            self.entered.add_permits(1);
            self.release.acquire().await.unwrap().forget();
        }
        Ok(ai_agents_core::LLMResponse::new(
            self.replies
                .lock()
                .pop_front()
                .unwrap_or_else(|| "done".into()),
            FinishReason::Stop,
        ))
    }
    async fn complete_stream(
        &self,
        _: &[ChatMessage],
        _: Option<&ai_agents_core::LLMConfig>,
    ) -> std::result::Result<
        Box<dyn futures::Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
        LLMError,
    > {
        Ok(Box::new(futures::stream::empty()))
    }
    fn provider_name(&self) -> &str {
        "ad-witness"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
}

// Every fixture uses the production builder, runner, and conditional backend; no miniature task executor is introduced.
fn runner(agent: Arc<RuntimeAgent>, cap: u32) -> (AutonomyRunner, Arc<ScopedTaskRunStore>) {
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info.id.clone(), agent.actor_id(), "ad-v1".into())
            .unwrap(),
    );
    let config = AutonomyConfig {
        defaults: AutonomyProfile {
            enabled: Some(true),
            max_turns: Some(4),
            max_llm_calls: Some(cap),
            completion: Some(CompletionGate::ResponseContains("done".into())),
            ..Default::default()
        },
        ..Default::default()
    };
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        AutonomyHostCeilings::default(),
        store.clone(),
        "ad-v1".into(),
    )
    .unwrap();
    (runner, store)
}

// The ordinary same-actor freshness policy remains configured, but task completion additionally requires run-owned settlement.
fn relationship_agent(mode: &str, main: Arc<Provider>, router: Arc<Provider>) -> RuntimeAgent {
    crate::AgentBuilder::from_yaml(&format!("name: Actor\nsystem_prompt: support\nllm:\n  default: default\n  router: router\nruntime:\n  optimization:\n    enabled: true\n    post_turn:\n      relationships:\n        mode: {mode}\n        await_before_next_turn: same_actor\nmemory:\n  relationships:\n    enabled: true\n    auto_update:\n      enabled: true\n      llm: router\n    persistence:\n      enabled: false\ntools: []\n")).unwrap().llm(main).llm_alias("router", router).auto_configure_features().unwrap().build().unwrap()
}

const RELATIONSHIP: &str = r#"{"changes":[{"perspective":"agent_to_actor","dimension":"trust","delta":0.1,"confidence":0.95,"reason":"verified actor evidence"}],"notable_event":null}"#;

// Initial conversational work is maintained once, while controller guidance cannot manufacture another actor interaction or semantic call.
#[tokio::test]
async fn ad_relationship_modes_share_limits_and_exclude_controller_interactions() {
    for mode in ["inline_serial", "inline_parallel", "background"] {
        let main = Provider::new(&["working", "done"], false);
        let auxiliary = Provider::new(&[RELATIONSHIP], false);
        let agent = Arc::new(relationship_agent(mode, main.clone(), auxiliary.clone()));
        agent.set_actor_id("actor").unwrap();
        let (runner, store) = runner(agent.clone(), 3);
        let result = runner.run("resolve the ticket", None).await.unwrap();
        assert_eq!(
            result.run.status,
            TaskRunStatus::Completed,
            "{mode}: {:?}",
            result.run.stop_reason
        );
        assert_eq!(result.run.counters.llm_attempts, 3);
        assert_eq!(result.run.counters.turns, 2);
        assert_eq!(main.calls.load(Ordering::SeqCst), 2);
        assert_eq!(auxiliary.calls.load(Ordering::SeqCst), 1);
        let relationship = agent
            .relationship_manager
            .as_ref()
            .unwrap()
            .get("actor")
            .unwrap();
        assert_eq!(relationship.interaction_count, 1);
        assert!(
            auxiliary
                .requests
                .lock()
                .iter()
                .flatten()
                .all(|message| !message.content.contains("Autonomy controller instruction"))
        );
        let saved = store.load(&result.run.key.run_id).await.unwrap().unwrap();
        let payload: crate::autonomy::TaskCheckpointPayload =
            serde_json::from_value(saved.payload).unwrap();
        assert!(
            payload
                .reservations
                .iter()
                .all(|r| r.state == TaskEffectState::Completed)
        );
        assert!(agent.background_maintenance.is_empty());
    }
}

struct Tail {
    entered: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
    execution: parking_lot::Mutex<Option<std::sync::Weak<crate::autonomy::RunExecution>>>,
}

impl Default for Tail {
    // Both gates start closed, making the post-provider tail observable without elapsed-time assumptions.
    fn default() -> Self {
        Self {
            entered: tokio::sync::Semaphore::new(0),
            release: tokio::sync::Semaphore::new(0),
            execution: Default::default(),
        }
    }
}

#[async_trait]
impl ai_agents_hooks::AgentHooks for Tail {
    // This fixture owns its counters and gated waits in the caller and performs no hidden provider or detached work.
    fn task_providers(&self) -> Option<Vec<std::sync::Arc<dyn ai_agents_core::LLMProvider>>> {
        Some(vec![])
    }

    async fn on_relationship_change(
        &self,
        _: &str,
        _: &[ai_agents_relationships::DimensionChange],
    ) {
        *self.execution.lock() = crate::autonomy::current_execution()
            .as_ref()
            .map(Arc::downgrade);
        self.entered.add_permits(1);
        self.release.acquire().await.unwrap().forget();
    }
}

// Provider settlement is not maintenance completion: the real hook tail keeps the gate and unrelated mutable entries closed.
#[tokio::test]
async fn ad_background_tail_blocks_gate_and_foreign_actor_apis() {
    let hook = Arc::new(Tail::default());
    let agent = Arc::new(
        relationship_agent(
            "background",
            Provider::new(&["done"], false),
            Provider::new(&[RELATIONSHIP], false),
        )
        .with_hooks(hook.clone()),
    );
    agent.set_actor_id("actor").unwrap();
    let (runner, _) = runner(agent.clone(), 2);
    let mut work = Box::pin(runner.run("resolve", None));
    tokio::select! { result = &mut work => panic!("gate passed before maintenance: {:?}", result.err()), permit = hook.entered.acquire() => permit.unwrap().forget() }
    let execution = hook.execution.lock().as_ref().unwrap().upgrade().unwrap();
    assert!(execution.owner.work.unsettled());
    assert!(agent.chat("foreign").await.is_err());
    assert!(agent.extract_facts(1).await.is_err());
    assert!(agent.load_actor_memory().await.is_err());
    assert!(agent.load_actor_relationship().await.is_err());
    assert!(
        agent
            .update_relationship_dimension("trust", 0.1, None)
            .await
            .is_err()
    );
    hook.release.add_permits(1);
    assert_eq!(work.await.unwrap().run.status, TaskRunStatus::Completed);
    assert!(!execution.owner.work.unsettled());
}

// Cancellation of an opaque hook cannot be advertised as acknowledged cleanup or followed by unrelated chat.
#[tokio::test]
async fn ad_background_cancel_retains_uncertain_effect_and_actual_owner() {
    let hook = Arc::new(Tail::default());
    let agent = Arc::new(
        relationship_agent(
            "background",
            Provider::new(&["done"], false),
            Provider::new(&[RELATIONSHIP], false),
        )
        .with_hooks(hook.clone()),
    );
    agent.set_actor_id("actor").unwrap();
    let (runner, store) = runner(agent.clone(), 2);
    let mut work = Box::pin(runner.run("resolve", None));
    tokio::select! { result = &mut work => panic!("returned before hook: {:?}", result.err()), permit = hook.entered.acquire() => permit.unwrap().forget() }
    let execution = hook.execution.lock().as_ref().unwrap().upgrade().unwrap();
    runner.request_cancel(&execution.run_id).await.unwrap();
    let result = work.await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::RecoveryRequired);
    assert!(agent.chat("foreign").await.is_err());
    let payload: crate::autonomy::TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&execution.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert!(payload.reservations.iter().any(|reservation| matches!(
        reservation.state,
        TaskEffectState::Uncertain | TaskEffectState::Dispatched
    )));
}

struct CustomEvaluator {
    conforming: bool,
    calls: Arc<AtomicUsize>,
}
#[async_trait]
impl ai_agents_relationships::RelationshipEvaluatorTrait for CustomEvaluator {
    fn task_providers(&self) -> Option<Vec<Arc<dyn LLMProvider>>> {
        self.conforming.then(Vec::new)
    }
    async fn evaluate_turn(
        &self,
        _: &ai_agents_relationships::Relationship,
        _: &[ChatMessage],
        _: &HashMap<String, ai_agents_relationships::RelationshipDimensionDefinition>,
    ) -> Result<ai_agents_relationships::RelationshipEvaluation> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Default::default())
    }
}

// A registered custom implementation needs explicit conformity, while a conforming deterministic implementation needs no artificial provider or tool grant.
#[tokio::test]
async fn ad_custom_evaluator_positive_participation_and_default_denial() {
    for conforming in [false, true] {
        let calls = Arc::new(AtomicUsize::new(0));
        let config: ai_agents_relationships::RelationshipConfig = serde_yaml::from_str(
            "enabled: true\npersistence:\n  enabled: false\nauto_update:\n  enabled: true\n",
        )
        .unwrap();
        let manager = Arc::new(
            RelationshipManager::from_config_with_evaluator(
                config,
                Some(Arc::new(CustomEvaluator {
                    conforming,
                    calls: calls.clone(),
                })),
            )
            .unwrap(),
        );
        let agent = Arc::new(
            crate::AgentBuilder::new()
                .system_prompt("test")
                .llm(Provider::new(&["done"], false))
                .build()
                .unwrap()
                .with_relationships(manager),
        );
        agent.set_actor_id("actor").unwrap();
        if conforming {
            let (runner, _) = runner(agent, 1);
            let result = runner.run("objective", None).await.unwrap();
            assert_eq!(result.run.status, TaskRunStatus::Completed);
            assert_eq!(result.run.counters.llm_attempts, 1);
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        } else {
            assert!(agent.preflight_standalone_autonomy().is_err());
            assert_eq!(calls.load(Ordering::SeqCst), 0);
        }
    }
}

// A winning route must settle an already dispatched draft rather than drop its managed attempt, and cannot execute a losing parsed tool call.
#[tokio::test]
async fn ad_speculative_transition_drains_actual_loser_and_consumes_attempt() {
    let main = Provider::new(
        &[r#"{"tool":"echo","arguments":{"message":"loser"}}"#, "done"],
        true,
    );
    let routing = Provider::new(&["1"], false);
    let yaml = "name: Route\nsystem_prompt: support\nllm:\n  default: default\n  router: router\nruntime:\n  optimization:\n    enabled: true\n    max_speculative_llm_calls_per_turn: 2\n    max_parallel_runtime_tasks: 2\n    speculative_state_transitions: true\nstates:\n  initial: triage\n  states:\n    triage:\n      transitions:\n        - to: billing\n          when: billing request\n          timing: parallel\n          requires_response: false\n    billing:\n      prompt: billing support\ntools: [echo]\n";
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
    let (runner, _) = runner(agent.clone(), 3);
    let mut work = Box::pin(runner.run("billing request", None));
    tokio::select! { result = &mut work => panic!("no actual draft dispatch: {:?}", result.err()), permit = main.entered.acquire() => permit.unwrap().forget() }
    assert_eq!(main.calls.load(Ordering::SeqCst), 1);
    assert!(futures::poll!(&mut work).is_pending());
    main.release.add_permits(1);
    tokio::select! { result = &mut work => panic!("winner never executed: {:?}", result.err()), permit = main.entered.acquire() => permit.unwrap().forget() }
    main.release.add_permits(1);
    let result = work.await.unwrap();
    assert_eq!(
        result.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        result.run.stop_reason
    );
    assert_eq!(result.run.counters.llm_attempts, 3);
    assert_eq!(result.run.counters.tool_attempts, 0);
    assert_eq!(agent.current_state().as_deref(), Some("billing"));
}

// The configured router is selected once even if task completion requires another controller turn after the skill segment.
#[tokio::test]
async fn ad_speculative_skill_continuation_does_not_reselect_script() {
    let main = Provider::new(&["discarded draft", "working", "done"], false);
    let routing = Provider::new(&["selected"], false);
    let yaml = "name: Skill\nsystem_prompt: test\nllm:\n  default: default\n  router: router\nruntime:\n  optimization:\n    enabled: true\n    max_speculative_llm_calls_per_turn: 2\n    max_parallel_runtime_tasks: 2\n    speculative_skill_routing: true\nskills:\n  - id: selected\n    description: selected work\n    trigger: perform work\n    steps:\n      - prompt: selected script\ntools: []\n";
    let agent = Arc::new(
        crate::AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(main)
            .llm_alias("router", routing.clone())
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap(),
    );
    let (runner, _) = runner(agent, 5);
    let result = runner.run("perform work", None).await.unwrap();
    assert_eq!(
        result.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        result.run.stop_reason
    );
    assert_eq!(routing.calls.load(Ordering::SeqCst), 1);
}

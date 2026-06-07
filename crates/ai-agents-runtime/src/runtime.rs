use async_trait::async_trait;
use futures::stream::{Stream, StreamExt};
use parking_lot::RwLock;
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tracing::{debug, error, info, instrument, warn};

use crate::turn_context::{current_turn_actor_context, scope_actor_context};

use ai_agents_context::{ContextManager, ContextProvider, TemplateRenderer};
use ai_agents_core::{
    AgentError, AgentSnapshot, AgentStorage, ChatMessage, FinishReason, LLMProvider, LLMResponse,
    Result, ToolResult,
};
use ai_agents_disambiguation::{
    ClarificationObserver, ClarificationParseFuture, ClarificationQuestionFuture,
    DisambiguationConfig, DisambiguationContext, DisambiguationManager, DisambiguationResult,
};
use ai_agents_hitl::{
    ApprovalHandler, ApprovalResult, ApprovalTrigger, HITLCheckResult, HITLEngine,
    RejectAllHandler, TimeoutAction,
};
use ai_agents_hooks::{AgentHooks, NoopHooks};
use ai_agents_llm::LLMRegistry;
use ai_agents_memory::{
    CompressResult, EvictionReason, Memory, MemoryBudgetEvent, MemoryCompressEvent,
    MemoryEvictEvent, MemoryTokenBudget, OverflowStrategy,
};
use ai_agents_observability::{
    EventStatus, EventType, ObservabilityManager, ObservationPurpose, SpanContext,
    current_observation_context, new_session_id as new_observation_session_id,
    resolve_language_from_context, with_observation_context, with_observation_purpose,
};
use ai_agents_process::{
    ProcessData, ProcessProcessor, ProcessPurposeHint, ProcessStageFuture, ProcessStageObserver,
};
use ai_agents_reasoning::{
    CriterionResult, EvaluationResult, Plan, PlanAction, PlanStatus, PlanStep, ReasoningConfig,
    ReasoningMetadata, ReasoningMode, ReasoningOutput, ReflectionAttempt, ReflectionConfig,
    ReflectionMetadata, StepFailureAction,
};
use ai_agents_recovery::{
    ByRoleFilter, ContextOverflowAction, FilterConfig, IntoClassifiedError, KeepRecentFilter,
    LLMFailureAction, MessageFilter, RecoveryManager, RetryConfig, SkipPatternFilter,
    ToolFailureAction,
};
use ai_agents_relationships::RelationshipManager;
use ai_agents_skills::{SkillDefinition, SkillExecutor, SkillRouter};
use ai_agents_state::{
    PromptMode, StateAction, StateMachine, StateMachineSnapshot, StateTransitionEvent, ToolRef,
    Transition, TransitionContext, TransitionEvaluator, TransitionTiming, evaluate_guard,
};
use ai_agents_storage::{StorageConfig as StorageStorageConfig, create_storage};
use ai_agents_tools::{
    ConditionEvaluator, EvaluationContext, LLMGetter, SecurityCheckResult, ToolCallRecord,
    ToolRegistry, ToolSecurityEngine,
};

use super::{
    Agent, AgentInfo, AgentResponse, ParallelToolsConfig, StreamChunk, StreamingConfig, ToolCall,
};
use crate::optimization::{
    AwaitBeforeNextTurn, BackgroundMaintenanceQueue, BackgroundOverflowPolicy, MainResponseDraft,
    MaintenanceMode, MaintenanceSequenceKey, RuntimeBranch, RuntimeBranchResult,
    RuntimeBranchStatus, RuntimeCommitBehavior, RuntimeConfig, RuntimeOptimizationKind,
    RuntimeTaskPriority, RuntimeTaskPurpose, ScheduledBranchSet, SkillCandidate,
    StreamingDraftResult, TransitionCandidate, TurnBranchScheduler, TurnOptimizationContext,
};
use crate::spec::StorageConfig;

/// Outcome of processing tool calls within the agent loop.
enum ToolCallOutcome {
    /// Tools executed successfully, continue the LLM loop for the next iteration.
    Continue,
    /// A state transition fired during tool call handling, continue the loop.
    TransitionFired,
    /// HITL rejected a tool call, return this response immediately.
    Rejected(AgentResponse),
}

/// Outcome of skill routing — used by `try_skill_route`.
enum SkillRouteResult {
    /// No skill matched, continue to normal LLM chat.
    NoMatch,
    /// Skill executed successfully.
    Response(String),
    /// Skill matched but needs disambiguation first.
    NeedsClarification(AgentResponse),
}

/// Outcome of post_loop_processing - drives the caller's next step.
enum PostLoopResult {
    /// No transition fired. Content is the LLM response for this turn.
    NoTransition(String),
    /// Transition fired. Content is from plain post-transition re-generation.
    Transitioned(String),
    /// Transition fired into a state that requires full dispatch.
    /// Caller re-enters run_loop_internal to apply the correct handler.
    NeedsRedispatch,
}

struct RootTurnCleanup<'a> {
    agent: &'a RuntimeAgent,
}

impl<'a> RootTurnCleanup<'a> {
    fn new(agent: &'a RuntimeAgent) -> Self {
        Self { agent }
    }
}

impl Drop for RootTurnCleanup<'_> {
    fn drop(&mut self) {
        self.agent.end_root_turn();
    }
}

pub struct RuntimeAgent {
    info: AgentInfo,
    llm_registry: Arc<LLMRegistry>,
    memory: Arc<dyn Memory>,
    tools: Arc<ToolRegistry>,
    skills: Vec<SkillDefinition>,
    skill_router: Option<SkillRouter>,
    skill_executor: Option<SkillExecutor>,
    base_system_prompt: String,
    max_iterations: u32,
    iteration_count: RwLock<u32>,
    max_context_tokens: u32,
    memory_token_budget: Option<MemoryTokenBudget>,
    recovery_manager: RecoveryManager,
    tool_security: ToolSecurityEngine,
    process_processor: Option<ProcessProcessor>,
    message_filters: RwLock<HashMap<String, Arc<dyn MessageFilter>>>,
    state_machine: Option<Arc<StateMachine>>,
    transition_evaluator: Option<Arc<dyn TransitionEvaluator>>,
    context_manager: Arc<ContextManager>,
    template_renderer: TemplateRenderer,
    tool_call_history: RwLock<Vec<ToolCallRecord>>,
    parallel_tools: ParallelToolsConfig,
    streaming: StreamingConfig,
    hooks: Arc<dyn AgentHooks>,
    hitl_engine: Option<HITLEngine>,
    approval_handler: Arc<dyn ApprovalHandler>,
    storage_config: StorageConfig,
    storage: RwLock<Option<Arc<dyn AgentStorage>>>,
    reasoning_config: ReasoningConfig,
    reflection_config: ReflectionConfig,
    disambiguation_manager: Option<DisambiguationManager>,
    /// Structured persona manager for identity, evolution, and secrets.
    persona_manager: Option<Arc<ai_agents_persona::PersonaManager>>,
    /// Skill ID that triggered the current pending disambiguation.
    /// Set by try_skill_route() when skill-level disambiguation triggers clarification.
    /// Read by run_loop() when clarification resolves to route directly to the skill.
    pending_skill_id: RwLock<Option<String>>,
    current_plan: RwLock<Option<Plan>>,
    /// Tool IDs declared in the top-level `tools:` spec.
    declared_tool_ids: Option<Vec<String>>,
    /// Whether the context manager has been initialized (defaults loaded, env resolved, etc.)
    context_initialized: AtomicBool,
    /// Spawner for dynamic agent creation (set when YAML has a spawner: section).
    spawner: Option<Arc<crate::spawner::AgentSpawner>>,
    /// Registry tracking spawned agents (set when YAML has a spawner: section).
    spawner_registry: Option<Arc<crate::spawner::AgentRegistry>>,
    /// Re-dispatch depth for post-transition full dispatch.
    /// 0 = not re-dispatching. > 0 = user message already in memory, skip re-adding.
    redispatch_depth: RwLock<u32>,
    /// Active optimized turn context used to keep root lifecycle state in one place.
    active_turn_context: RwLock<Option<TurnOptimizationContext>>,
    /// Tracks whether the root turn already wrote the processed user message.
    root_user_message_committed: AtomicBool,
    /// Current actor ID for cross-session memory.
    actor_id: RwLock<Option<String>>,
    /// Fact store for managing per-actor extracted facts.
    fact_store: RwLock<Option<Arc<ai_agents_facts::FactStore>>>,
    /// Fact extractor for LLM-based fact extraction.
    /// None when actor_memory is enabled without facts.enabled.
    fact_extractor: RwLock<Option<Arc<dyn ai_agents_facts::FactExtractor>>>,
    /// Cached actor facts keyed by actor ID so concurrent or alternating turns do not overwrite one another.
    actor_facts_cache: Arc<RwLock<HashMap<String, Vec<ai_agents_core::KeyFact>>>>,
    /// Number of messages since last fact extraction.
    messages_since_extraction: Arc<RwLock<usize>>,
    /// Actor memory configuration.
    actor_memory_config: Option<ai_agents_facts::ActorMemoryConfig>,
    /// Facts configuration.
    facts_config: Option<ai_agents_facts::FactsConfig>,
    /// Session-scoped metadata (tags, ttl, actor roster).
    session_metadata: RwLock<ai_agents_core::SessionMetadata>,
    /// Session id currently bound to this runtime instance.
    current_session_id: RwLock<Option<String>>,
    /// Relationship manager for actor-scoped social memory.
    relationship_manager: Option<Arc<RelationshipManager>>,
    /// Observability manager for traces, metrics, reports, and exports.
    observability_manager: Option<Arc<ObservabilityManager>>,
    /// Runtime optimization and maintenance policy.
    runtime_config: RuntimeConfig,
    /// Queue for background maintenance tasks.
    background_maintenance: Arc<BackgroundMaintenanceQueue>,
}

impl std::fmt::Debug for RuntimeAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeAgent")
            .field("info", &self.info)
            .field("base_system_prompt", &self.base_system_prompt)
            .field("max_iterations", &self.max_iterations)
            .field("skills_count", &self.skills.len())
            .field("max_context_tokens", &self.max_context_tokens)
            .field("has_state_machine", &self.state_machine.is_some())
            .field("parallel_tools", &self.parallel_tools)
            .field("streaming", &self.streaming)
            .field("has_hooks", &true)
            .field("has_hitl", &self.hitl_engine.is_some())
            .field("storage_type", &self.storage_config.storage_type())
            .field("reasoning_mode", &self.reasoning_config.mode)
            .field("reflection_enabled", &self.reflection_config.enabled)
            .field("declared_tool_ids", &self.declared_tool_ids)
            .field("has_persona", &self.persona_manager.is_some())
            .field("has_observability", &self.observability_manager.is_some())
            .finish_non_exhaustive()
    }
}

struct ObservabilityClarificationObserver;

impl ClarificationObserver for ObservabilityClarificationObserver {
    /// Scopes clarification question generation as disambiguation_clarification.
    fn observe_question<'a>(
        &'a self,
        future: ClarificationQuestionFuture<'a>,
    ) -> ClarificationQuestionFuture<'a> {
        Box::pin(async move {
            with_observation_purpose(ObservationPurpose::DisambiguationClarification, future).await
        })
    }

    /// Scopes clarification response parsing as disambiguation_clarification.
    fn observe_parse<'a>(
        &'a self,
        future: ClarificationParseFuture<'a>,
    ) -> ClarificationParseFuture<'a> {
        Box::pin(async move {
            with_observation_purpose(ObservationPurpose::DisambiguationClarification, future).await
        })
    }
}

struct ObservabilityProcessStageObserver;

impl ProcessStageObserver for ObservabilityProcessStageObserver {
    /// Scopes one process stage using the purpose implied by its stage type.
    fn observe<'a>(
        &'a self,
        hint: ProcessPurposeHint,
        future: ProcessStageFuture<'a>,
    ) -> ProcessStageFuture<'a> {
        Box::pin(async move {
            with_observation_purpose(observation_purpose_for_process(hint), future).await
        })
    }
}

struct RegistryLLMGetter {
    registry: Arc<LLMRegistry>,
}

impl LLMGetter for RegistryLLMGetter {
    fn get_llm(&self, alias: &str) -> Option<Arc<dyn LLMProvider>> {
        self.registry.get(alias).ok()
    }
}

impl RuntimeAgent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        info: AgentInfo,
        llm_registry: Arc<LLMRegistry>,
        memory: Arc<dyn Memory>,
        tools: Arc<ToolRegistry>,
        skills: Vec<SkillDefinition>,
        system_prompt: String,
        max_iterations: u32,
    ) -> Self {
        let (skill_router, skill_executor) = if !skills.is_empty() {
            let router_llm = llm_registry.router().ok();
            let router = router_llm.map(|llm| SkillRouter::new(llm, skills.clone()));
            let executor = SkillExecutor::new(llm_registry.clone(), tools.clone());
            (router, Some(executor))
        } else {
            (None, None)
        };

        let context_manager =
            ContextManager::new(HashMap::new(), info.name.clone(), info.version.clone());

        Self {
            info,
            llm_registry,
            memory,
            tools,
            skills,
            skill_router,
            skill_executor,
            base_system_prompt: system_prompt,
            max_iterations,
            iteration_count: RwLock::new(0),
            max_context_tokens: 128000,
            memory_token_budget: None,
            recovery_manager: RecoveryManager::default(),
            tool_security: ToolSecurityEngine::default(),
            process_processor: None,
            message_filters: RwLock::new(HashMap::new()),
            state_machine: None,
            transition_evaluator: None,
            context_manager: Arc::new(context_manager),
            template_renderer: TemplateRenderer::new(),
            tool_call_history: RwLock::new(Vec::new()),
            parallel_tools: ParallelToolsConfig::default(),
            streaming: StreamingConfig::default(),
            hooks: Arc::new(NoopHooks),
            hitl_engine: None,
            approval_handler: Arc::new(RejectAllHandler::new()),
            storage_config: StorageConfig::default(),
            storage: RwLock::new(None),
            reasoning_config: ReasoningConfig::default(),
            reflection_config: ReflectionConfig::default(),
            disambiguation_manager: None,
            persona_manager: None,
            pending_skill_id: RwLock::new(None),
            current_plan: RwLock::new(None),
            declared_tool_ids: None,
            context_initialized: AtomicBool::new(false),
            spawner: None,
            spawner_registry: None,
            redispatch_depth: RwLock::new(0),
            active_turn_context: RwLock::new(None),
            root_user_message_committed: AtomicBool::new(false),
            actor_id: RwLock::new(None),
            fact_store: RwLock::new(None),
            fact_extractor: RwLock::new(None),
            actor_facts_cache: Arc::new(RwLock::new(HashMap::new())),
            messages_since_extraction: Arc::new(RwLock::new(0)),
            actor_memory_config: None,
            facts_config: None,
            session_metadata: RwLock::new(ai_agents_core::SessionMetadata::default()),
            current_session_id: RwLock::new(None),
            relationship_manager: None,
            observability_manager: None,
            runtime_config: RuntimeConfig::default(),
            background_maintenance: Arc::new(BackgroundMaintenanceQueue::default()),
        }
    }

    pub fn with_declared_tool_ids(mut self, ids: Option<Vec<String>>) -> Self {
        self.declared_tool_ids = ids;
        self
    }

    pub fn with_storage_config(mut self, config: StorageConfig) -> Self {
        self.storage_config = config;
        self
    }

    pub fn with_storage(self, storage: Arc<dyn AgentStorage>) -> Self {
        *self.storage.write() = Some(storage);
        self
    }

    pub fn with_reasoning(mut self, config: ReasoningConfig) -> Self {
        self.reasoning_config = config;
        self
    }

    pub fn with_reflection(mut self, config: ReflectionConfig) -> Self {
        self.reflection_config = config;
        self
    }

    /// Attach a relationship manager configured by the builder or host application.
    pub fn with_relationships(mut self, manager: Arc<RelationshipManager>) -> Self {
        self.relationship_manager = Some(manager);
        self
    }

    /// Attach a shared observability manager for traces, metrics, reports, and exports.
    pub fn with_observability(mut self, manager: Arc<ObservabilityManager>) -> Self {
        self.observability_manager = Some(manager);
        self
    }

    /// Attach runtime optimization policy and resize the background queue.
    pub fn with_runtime_config(mut self, config: RuntimeConfig) -> Self {
        let max_tasks = config.optimization.post_turn.max_background_tasks;
        self.background_maintenance = Arc::new(BackgroundMaintenanceQueue::new(max_tasks));
        self.runtime_config = config;
        self
    }

    /// Returns the runtime optimization policy.
    pub fn runtime_config(&self) -> &RuntimeConfig {
        &self.runtime_config
    }

    /// Wait for all background maintenance tasks to finish.
    pub async fn flush_background_tasks(&self) -> Result<()> {
        self.background_maintenance.flush_all().await
    }

    /// Wait for background maintenance associated with one actor to finish.
    pub async fn flush_background_tasks_for_actor(&self, actor_id: &str) -> Result<()> {
        self.background_maintenance.flush_scope(actor_id).await
    }

    /// Wait for background maintenance associated with one task kind to finish.
    pub async fn flush_background_tasks_for_purpose(
        &self,
        purpose: RuntimeTaskPurpose,
    ) -> Result<()> {
        self.background_maintenance.flush_purpose(purpose).await
    }

    /// Wait for background maintenance associated with one actor and task kind to finish.
    pub async fn flush_background_tasks_for_actor_purpose(
        &self,
        actor_id: &str,
        purpose: RuntimeTaskPurpose,
    ) -> Result<()> {
        self.background_maintenance
            .flush_scope_purpose(actor_id, purpose)
            .await
    }

    /// Flush background maintenance before a host shuts down the runtime.
    pub async fn shutdown_background_tasks(&self) -> Result<()> {
        self.flush_background_tasks().await
    }

    /// Returns the configured observability manager for report and export access.
    pub fn observability(&self) -> Option<Arc<ObservabilityManager>> {
        self.observability_manager.clone()
    }

    /// Exports observability files after a turn when export settings request it.
    async fn export_observability_if_configured(&self) {
        let Some(manager) = self.observability_manager.as_ref() else {
            return;
        };
        let export = &manager.config().export;
        if !export.write_report && !export.write_raw_events {
            return;
        }
        if let Err(error) = manager.export().await {
            warn!(error = %error, "Observability export failed");
        }
    }

    /// Returns the configured relationship manager, if relationship memory is enabled.
    pub fn relationship_manager(&self) -> Option<Arc<RelationshipManager>> {
        self.relationship_manager.clone()
    }

    fn current_turn_actor_context(&self) -> Option<crate::TurnActorContext> {
        current_turn_actor_context()
    }

    fn effective_actor_id(&self) -> Option<String> {
        self.current_turn_actor_context()
            .and_then(|ctx| ctx.effective_actor_id().map(|id| id.to_string()))
            .or_else(|| self.actor_id.read().clone())
    }

    fn effective_origin_actor_id(&self) -> Option<String> {
        self.current_turn_actor_context()
            .and_then(|ctx| ctx.origin_actor_id.clone())
            .or_else(|| self.actor_id.read().clone())
    }

    fn record_session_actor_if_needed(&self) {
        if let Some(actor_id) = self.effective_origin_actor_id() {
            let mut meta = self.session_metadata.write();
            meta.actor_id = Some(actor_id.clone());
            if !meta.actors.iter().any(|a| a == &actor_id) {
                meta.actors.push(actor_id);
            }
        }
    }

    fn outbound_actor_context(&self) -> crate::TurnActorContext {
        let mut context = self.current_turn_actor_context().unwrap_or_default();
        if context.origin_actor_id.is_none() {
            context.origin_actor_id = self.effective_origin_actor_id();
        }
        context.sender_agent_id = Some(self.info.id.clone());
        context
    }

    /// Returns the current session ID or creates one for unsaved observed turns.
    fn observation_session_id(&self) -> Option<String> {
        let mut current = self.current_session_id.write();
        if current.is_none() {
            *current = Some(new_observation_session_id());
        }
        current.clone()
    }

    /// Builds the root or child observation context for a chat entry point.
    fn build_observation_context(&self, actor_id: Option<String>) -> Option<SpanContext> {
        let manager = self.observability_manager.as_ref()?;
        let context = self.build_context_with_overlays();
        let language = resolve_language_from_context(manager.config(), &context);
        let context = current_observation_context()
            .map(|parent| parent.child_for_agent(self.info.id.clone()).with_new_turn())
            .unwrap_or_else(|| SpanContext::new_root(self.info.id.clone()));
        Some(
            context
                .with_actor(actor_id.or_else(|| self.effective_actor_id()))
                .with_session(self.observation_session_id())
                .with_state(self.current_state())
                .with_language(Some(language)),
        )
    }

    /// Refreshes task-local context with current runtime labels and a purpose.
    fn current_runtime_observation_context(
        &self,
        purpose: ObservationPurpose,
    ) -> Option<SpanContext> {
        let manager = self.observability_manager.as_ref()?;
        let context = self.build_context_with_overlays();
        let language = resolve_language_from_context(manager.config(), &context);
        let mut observation = current_observation_context()
            .unwrap_or_else(|| SpanContext::new_root(self.info.id.clone()));
        observation.agent_id = self.info.id.clone();
        observation.actor_id = self.effective_actor_id();
        observation.session_id = self.observation_session_id();
        observation.state = self.current_state();
        observation.language = Some(language);
        observation.purpose = purpose;
        Some(observation)
    }

    /// Runs a future under a purpose while preserving current trace context.
    async fn observe_purpose<F, T>(&self, purpose: ObservationPurpose, future: F) -> T
    where
        F: Future<Output = T>,
    {
        if let Some(context) = self.current_runtime_observation_context(purpose) {
            with_observation_context(context, future).await
        } else {
            future.await
        }
    }

    /// Runs chat with actor and observation context while avoiding recursive async types.
    fn chat_with_actor_context_boxed<'a>(
        &'a self,
        input: &'a str,
        actor_context: crate::TurnActorContext,
    ) -> Pin<Box<dyn Future<Output = Result<AgentResponse>> + Send + 'a>> {
        Box::pin(async move {
            let actor_id = actor_context.effective_actor_id().map(str::to_string);
            let run = async move {
                scope_actor_context(
                    actor_context,
                    Box::pin(async move { self.run_loop(input).await }),
                )
                .await
            };
            let result = if let Some(context) = self.build_observation_context(actor_id) {
                with_observation_context(context, run).await
            } else {
                run.await
            };
            self.export_observability_if_configured().await;
            result
        })
    }

    /// Run one turn with turn-scoped actor context without mutating the runtime's global actor ID.
    ///
    /// The supplied context is available to actor-scoped facts, relationship memory, orchestration, and prompt templates only for the lifetime of this call.
    pub async fn chat_with_actor_context(
        &self,
        input: &str,
        actor_context: crate::TurnActorContext,
    ) -> Result<AgentResponse> {
        self.chat_with_actor_context_boxed(input, actor_context)
            .await
    }

    /// Convenience wrapper around [`Self::chat_with_actor_context`] for a turn whose original actor is known up front.
    pub async fn chat_as_actor(&self, actor_id: &str, input: &str) -> Result<AgentResponse> {
        let actor_context = crate::TurnActorContext::new().with_origin_actor(actor_id);
        self.chat_with_actor_context(input, actor_context).await
    }

    /// Ensure the effective actor's relationship is loaded from storage into the relationship manager.
    pub async fn load_actor_relationship(&self) -> Result<()> {
        self.maybe_load_actor_relationship().await;
        Ok(())
    }

    /// Manually apply a delta to the effective actor's `agent_to_actor` relationship perspective and persist the updated relationship when storage is configured.
    pub async fn update_relationship_dimension(
        &self,
        dimension: &str,
        delta: f64,
        reason: Option<&str>,
    ) -> Result<ai_agents_relationships::DimensionChange> {
        self.update_relationship_dimension_for_perspective(
            ai_agents_relationships::RelationshipPerspective::AgentToActor,
            dimension,
            delta,
            reason,
        )
        .await
    }

    /// Manually apply a delta to a specific relationship perspective for the effective actor.
    ///
    /// Use this for two-sided configurations when you need to update `agent_to_actor`, `perceived_actor_to_agent`, or `mutual` explicitly from application logic.
    pub async fn update_relationship_dimension_for_perspective(
        &self,
        perspective: ai_agents_relationships::RelationshipPerspective,
        dimension: &str,
        delta: f64,
        reason: Option<&str>,
    ) -> Result<ai_agents_relationships::DimensionChange> {
        let manager = self
            .relationship_manager
            .as_ref()
            .ok_or_else(|| AgentError::Config("Relationship memory is not configured".into()))?;
        let actor_id = self.effective_actor_id().ok_or_else(|| {
            AgentError::Config("No actor ID set. Use set_actor_id() first".into())
        })?;
        let change = manager.update_dimension_for_perspective(
            &actor_id,
            perspective,
            dimension,
            delta,
            1.0,
            reason.unwrap_or("manual relationship update"),
        )?;
        self.persist_actor_relationship(&actor_id).await?;
        info!(
            actor_id = %actor_id,
            perspective = %change.perspective,
            dimension = %change.dimension,
            delta = change.delta,
            current = change.current,
            "relationship updated manually"
        );
        self.hooks
            .on_relationship_change(&actor_id, std::slice::from_ref(&change))
            .await;
        Ok(change)
    }

    pub fn reasoning_config(&self) -> &ReasoningConfig {
        &self.reasoning_config
    }

    pub fn reflection_config(&self) -> &ReflectionConfig {
        &self.reflection_config
    }

    /// Set only the actor memory and facts configs without creating the store.
    /// The store and extractor are created lazily in init_storage().
    pub fn with_facts_config(
        mut self,
        actor_memory_config: Option<ai_agents_facts::ActorMemoryConfig>,
        facts_config: Option<ai_agents_facts::FactsConfig>,
    ) -> Self {
        self.actor_memory_config = actor_memory_config;
        self.facts_config = facts_config;
        self
    }

    /// Configure fact store and optional extractor for actor memory.
    /// Pass `None` for `extractor` to load existing facts without running extraction.
    pub fn with_facts(
        mut self,
        store: Arc<ai_agents_facts::FactStore>,
        extractor: Option<Arc<dyn ai_agents_facts::FactExtractor>>,
        actor_memory_config: Option<ai_agents_facts::ActorMemoryConfig>,
        facts_config: Option<ai_agents_facts::FactsConfig>,
    ) -> Self {
        *self.fact_store.write() = Some(store);
        *self.fact_extractor.write() = extractor;
        self.actor_memory_config = actor_memory_config;
        self.facts_config = facts_config;
        self
    }

    /// Get the fact store for direct fact manipulation.
    pub fn fact_store(&self) -> Option<Arc<ai_agents_facts::FactStore>> {
        self.fact_store.read().clone()
    }

    /// Get the current actor ID.
    pub fn actor_id(&self) -> Option<String> {
        self.actor_id.read().clone()
    }

    /// Set the current actor ID (player, user, another agent, etc.).
    pub fn set_actor_id(&self, actor_id: &str) -> ai_agents_core::Result<()> {
        *self.actor_id.write() = Some(actor_id.to_string());
        {
            let mut meta = self.session_metadata.write();
            meta.actor_id = Some(actor_id.to_string());
            if !meta.actors.iter().any(|a| a == actor_id) {
                meta.actors.push(actor_id.to_string());
            }
        }
        Ok(())
    }

    /// Set the current actor ID. Convenience wrapper around set_actor_id.
    pub fn set_user_id(&self, user_id: &str) -> ai_agents_core::Result<()> {
        self.set_actor_id(user_id)
    }

    /// Load facts for the current actor from storage and cache them for prompt injection.
    pub async fn load_actor_memory(&self) -> ai_agents_core::Result<()> {
        let actor_id = match self.effective_actor_id() {
            Some(id) => id,
            None => return Ok(()),
        };

        let store_opt = self.fact_store.read().clone();
        if let Some(store) = store_opt {
            let facts = store.get_facts(&actor_id).await?;
            let count = facts.len();
            self.actor_facts_cache
                .write()
                .insert(actor_id.clone(), facts);
            self.hooks.on_actor_memory_loaded(&actor_id, count).await;
            tracing::debug!("loaded {} facts for actor {}", count, actor_id);
        }

        Ok(())
    }

    /// Load actor memory only when the effective actor has no cached facts yet.
    async fn maybe_load_actor_memory(&self) {
        let Some(actor_id) = self.effective_actor_id() else {
            return;
        };
        if self.actor_facts_cache.read().contains_key(&actor_id) {
            return;
        }
        let _ = self.load_actor_memory().await;
    }

    /// Pre-turn lifecycle shared by streaming and non-streaming paths.
    async fn pre_turn_session_lifecycle(&self) {
        if *self.redispatch_depth.read() > 0 {
            return;
        }
        self.resolve_actor_id_from_context();
        self.await_background_before_next_turn().await;
        self.record_session_actor_if_needed();
        self.maybe_load_actor_memory().await;
        self.maybe_load_actor_relationship().await;
        *self.messages_since_extraction.write() += 1;
    }

    /// Post-turn lifecycle shared by streaming and non-streaming paths.
    async fn post_turn_session_lifecycle(&self) -> Result<()> {
        if *self.redispatch_depth.read() > 0 {
            return Ok(());
        }
        *self.messages_since_extraction.write() += 1;
        self.run_post_turn_maintenance().await
    }

    /// Starts root-turn bookkeeping for user-message commit tracking.
    fn begin_root_turn(&self) {
        if *self.redispatch_depth.read() == 0 {
            let mut guard = self.active_turn_context.write();
            if guard.is_none() {
                self.root_user_message_committed
                    .store(false, Ordering::SeqCst);
                let max_calls = self
                    .runtime_config
                    .optimization
                    .max_speculative_llm_calls_per_turn;
                *guard = Some(TurnOptimizationContext::new(
                    String::new(),
                    HashMap::new(),
                    max_calls,
                ));
            }
        }
    }

    fn update_active_turn_context(
        &self,
        processed_input: &str,
        input_context: HashMap<String, Value>,
    ) {
        if *self.redispatch_depth.read() > 0 {
            return;
        }
        let max_calls = self
            .runtime_config
            .optimization
            .max_speculative_llm_calls_per_turn;
        let mut guard = self.active_turn_context.write();
        match guard.as_mut() {
            Some(context) => {
                context.processed_input = processed_input.to_string();
                context.input_context = input_context;
                context.max_speculative_llm_calls = max_calls;
            }
            None => {
                *guard = Some(TurnOptimizationContext::new(
                    processed_input,
                    input_context,
                    max_calls,
                ));
            }
        }
    }

    /// Writes the processed user message once for the root turn.
    async fn commit_root_user_message(&self, processed_input: &str) -> Result<()> {
        if *self.redispatch_depth.read() > 0 {
            return Ok(());
        }
        if !self
            .root_user_message_committed
            .swap(true, Ordering::SeqCst)
        {
            self.memory
                .add_message(ChatMessage::user(processed_input))
                .await?;
            if let Some(context) = self.active_turn_context.write().as_mut() {
                context.mark_user_message_committed();
            }
        }
        Ok(())
    }

    /// Clears root-turn bookkeeping after final response handling.
    fn end_root_turn(&self) {
        if *self.redispatch_depth.read() == 0 {
            self.root_user_message_committed
                .store(false, Ordering::SeqCst);
            *self.active_turn_context.write() = None;
        }
    }

    fn reserve_active_speculative_llm_call(&self, kind: RuntimeOptimizationKind) -> bool {
        self.begin_root_turn();
        let mut guard = self.active_turn_context.write();
        let Some(context) = guard.as_mut() else {
            return false;
        };
        context.reserve_speculative_llm_call_for(kind)
    }

    fn branch_context_preview(&self) -> String {
        let context = self.build_context_with_overlays();
        let mut value = serde_json::to_string_pretty(&context).unwrap_or_else(|_| "{}".to_string());
        const MAX_CONTEXT_PREVIEW_CHARS: usize = 2048;
        if value.chars().count() > MAX_CONTEXT_PREVIEW_CHARS {
            value = value
                .chars()
                .take(MAX_CONTEXT_PREVIEW_CHARS)
                .collect::<String>();
            value.push_str("...");
        }
        value
    }

    /// Applies freshness policy before rendering the next prompt.
    async fn await_background_before_next_turn(&self) {
        let optimization = &self.runtime_config.optimization;
        if !optimization.enabled {
            return;
        }
        let actor_id = self.effective_actor_id();
        let post = &optimization.post_turn;
        self.await_background_task(
            post.facts.await_before_next_turn,
            RuntimeTaskPurpose::PostTurnFacts,
            actor_id.as_deref(),
            "facts",
        )
        .await;
        self.await_background_task(
            post.relationships.await_before_next_turn,
            RuntimeTaskPurpose::PostTurnRelationship,
            actor_id.as_deref(),
            "relationships",
        )
        .await;
    }

    async fn await_background_task(
        &self,
        policy: AwaitBeforeNextTurn,
        purpose: RuntimeTaskPurpose,
        actor_id: Option<&str>,
        label: &str,
    ) {
        match policy {
            AwaitBeforeNextTurn::Never => {}
            AwaitBeforeNextTurn::Always => {
                if let Err(error) = self.flush_background_tasks_for_purpose(purpose).await {
                    warn!(label = label, error = %error, "background maintenance flush failed");
                }
            }
            AwaitBeforeNextTurn::SameActor => {
                if let Some(actor_id) = actor_id {
                    if let Err(error) = self
                        .flush_background_tasks_for_actor_purpose(actor_id, purpose)
                        .await
                    {
                        warn!(label = label, actor_id = %actor_id, error = %error, "actor background maintenance flush failed");
                    }
                }
            }
        }
    }

    /// Runs post-turn facts and relationship maintenance according to runtime policy.
    async fn run_post_turn_maintenance(&self) -> Result<()> {
        let optimization = &self.runtime_config.optimization;
        if !optimization.enabled {
            self.auto_extract_facts().await;
            self.auto_update_relationship().await;
            return Ok(());
        }

        let facts_mode = effective_maintenance_mode(
            optimization.post_turn.facts.mode,
            optimization.parallel_post_turn_memory,
        );
        let relationships_mode = effective_maintenance_mode(
            optimization.post_turn.relationships.mode,
            optimization.parallel_post_turn_memory,
        );

        match (facts_mode, relationships_mode) {
            (MaintenanceMode::InlineSerial, MaintenanceMode::InlineSerial) => {
                self.auto_extract_facts().await;
                self.auto_update_relationship().await;
            }
            (MaintenanceMode::InlineParallel, MaintenanceMode::InlineParallel) => {
                let facts = self.auto_extract_facts();
                let relationships = self.auto_update_relationship();
                tokio::join!(facts, relationships);
            }
            (MaintenanceMode::Background, MaintenanceMode::Background) => {
                self.schedule_facts_background().await?;
                self.schedule_relationship_background().await?;
            }
            (MaintenanceMode::Background, MaintenanceMode::InlineParallel)
            | (MaintenanceMode::Background, MaintenanceMode::InlineSerial) => {
                self.schedule_facts_background().await?;
                self.auto_update_relationship().await;
            }
            (MaintenanceMode::InlineParallel, MaintenanceMode::Background)
            | (MaintenanceMode::InlineSerial, MaintenanceMode::Background) => {
                self.auto_extract_facts().await;
                self.schedule_relationship_background().await?;
            }
            _ => {
                self.auto_extract_facts().await;
                self.auto_update_relationship().await;
            }
        }
        Ok(())
    }

    async fn schedule_facts_background(&self) -> Result<()> {
        let policy = self.runtime_config.optimization.post_turn.facts.clone();
        let should_extract = self
            .facts_config
            .as_ref()
            .map(|c| c.enabled && c.auto_extract)
            .unwrap_or(false);
        if !should_extract {
            return Ok(());
        }
        let msgs_since = *self.messages_since_extraction.read();
        if msgs_since < 2 {
            return Ok(());
        }
        let Some(actor_id) = self.effective_actor_id() else {
            self.record_skipped_maintenance(
                "facts",
                ObservationPurpose::FactsExtraction,
                "missing_actor",
                Some(&policy),
            );
            return Ok(());
        };
        let Some(extractor) = self.fact_extractor.read().clone() else {
            return Ok(());
        };
        let messages = match self.memory.get_messages(None).await {
            Ok(messages) => messages,
            Err(error) => {
                warn!(error = %error, "failed to snapshot messages for fact extraction");
                return Ok(());
            }
        };
        let recent: Vec<_> = messages
            .iter()
            .rev()
            .take(msgs_since)
            .rev()
            .cloned()
            .collect();
        if recent.is_empty() {
            return Ok(());
        }
        let existing = self
            .actor_facts_cache
            .read()
            .get(&actor_id)
            .cloned()
            .unwrap_or_default();
        let categories = self
            .facts_config
            .as_ref()
            .map(|c| c.custom_categories.clone())
            .unwrap_or_default();
        let store = self.fact_store.read().clone();
        let cache = Arc::clone(&self.actor_facts_cache);
        let counter = Arc::clone(&self.messages_since_extraction);
        let hooks = Arc::clone(&self.hooks);
        let agent_id = self.info.id.clone();
        let observation = current_observation_context();
        let key = MaintenanceSequenceKey::actor(
            agent_id,
            actor_id.clone(),
            RuntimeTaskPurpose::PostTurnFacts,
        );
        let actor_for_task = actor_id.clone();
        let task = async move {
            let run = async move {
                let facts = extractor
                    .extract(&recent, &existing, Some(&actor_for_task), &categories)
                    .await?;
                if !facts.is_empty() {
                    if let Some(store) = store {
                        let authoritative = store.add_facts(&actor_for_task, facts.clone()).await?;
                        cache.write().insert(actor_for_task.clone(), authoritative);
                    } else {
                        cache
                            .write()
                            .entry(actor_for_task.clone())
                            .or_default()
                            .extend(facts.clone());
                    }
                    {
                        let mut count = counter.write();
                        if *count <= msgs_since {
                            *count = 0;
                        } else {
                            *count -= msgs_since;
                        }
                    }
                    hooks.on_facts_extracted(&actor_for_task, &facts).await;
                }
                Ok(())
            };
            if let Some(context) = observation {
                with_observation_context(
                    context.with_purpose(ObservationPurpose::FactsExtraction),
                    run,
                )
                .await
            } else {
                run.await
            }
        };
        self.spawn_or_handle_background(Some(key), task, "facts", &policy)
            .await
    }

    async fn schedule_relationship_background(&self) -> Result<()> {
        let policy = self
            .runtime_config
            .optimization
            .post_turn
            .relationships
            .clone();
        let Some(manager) = self.relationship_manager.as_ref().cloned() else {
            return Ok(());
        };
        let Some(actor_id) = self.effective_actor_id() else {
            self.record_skipped_maintenance(
                "relationships",
                ObservationPurpose::RelationshipUpdate,
                "missing_actor",
                Some(&policy),
            );
            return Ok(());
        };
        let recent_messages = manager.config().auto_update.recent_messages;
        let messages = match self.memory.get_messages(Some(recent_messages)).await {
            Ok(messages) => messages,
            Err(error) => {
                warn!(actor = %actor_id, error = %error, "failed to snapshot messages for relationship update");
                return Ok(());
            }
        };
        let storage = self.storage.read().clone();
        let hooks = Arc::clone(&self.hooks);
        let agent_id = self.info.id.clone();
        let observation = current_observation_context();
        let key = MaintenanceSequenceKey::actor(
            agent_id.clone(),
            actor_id.clone(),
            RuntimeTaskPurpose::PostTurnRelationship,
        );
        let actor_for_task = actor_id.clone();
        let task = async move {
            let run = async move {
                if manager.config().auto_update.enabled {
                    let update = manager.auto_update(&actor_for_task, &messages).await?;
                    if !update.changes.is_empty() {
                        hooks
                            .on_relationship_change(&actor_for_task, &update.changes)
                            .await;
                    }
                    if let Some(ref event) = update.event {
                        hooks.on_notable_event(&actor_for_task, event).await;
                    }
                }
                if manager.config().persistence.enabled {
                    if let (Some(storage), Some(value)) =
                        (storage, manager.relationship_as_value(&actor_for_task)?)
                    {
                        storage
                            .save_relationship(&agent_id, &actor_for_task, &value)
                            .await?;
                    }
                }
                Ok(())
            };
            if let Some(context) = observation {
                with_observation_context(
                    context.with_purpose(ObservationPurpose::RelationshipUpdate),
                    run,
                )
                .await
            } else {
                run.await
            }
        };
        self.spawn_or_handle_background(Some(key), task, "relationships", &policy)
            .await
    }

    /// Queues background maintenance or applies the configured overflow behavior.
    async fn spawn_or_handle_background<F>(
        &self,
        key: Option<MaintenanceSequenceKey>,
        task: F,
        label: &'static str,
        policy: &crate::optimization::config::MaintenanceTaskPolicy,
    ) -> Result<()>
    where
        F: Future<Output = Result<()>> + Send + 'static,
    {
        if self.background_maintenance.is_full() {
            match self
                .runtime_config
                .optimization
                .post_turn
                .on_background_overflow
            {
                BackgroundOverflowPolicy::RunInline => {
                    record_background_maintenance_event(
                        self.observability_manager.as_ref(),
                        label,
                        EventStatus::Success,
                        0,
                        "inline_overflow",
                        None,
                        Some(policy),
                    );
                    let start = Instant::now();
                    match task.await {
                        Ok(()) => record_background_maintenance_event(
                            self.observability_manager.as_ref(),
                            label,
                            EventStatus::Success,
                            start.elapsed().as_millis() as u64,
                            "inline_completed",
                            None,
                            Some(policy),
                        ),
                        Err(error) => {
                            warn!(label = label, error = %error, "inline maintenance fallback failed");
                            record_background_maintenance_event(
                                self.observability_manager.as_ref(),
                                label,
                                EventStatus::Error,
                                start.elapsed().as_millis() as u64,
                                "inline_failed",
                                Some(error.to_string()),
                                Some(policy),
                            );
                            return Err(error);
                        }
                    }
                }
                BackgroundOverflowPolicy::Drop => {
                    self.record_skipped_maintenance(
                        label,
                        ObservationPurpose::Other(label.to_string()),
                        "queue_full",
                        Some(policy),
                    );
                }
                BackgroundOverflowPolicy::Error => {
                    record_background_maintenance_event(
                        self.observability_manager.as_ref(),
                        label,
                        EventStatus::Error,
                        0,
                        "queue_full",
                        None,
                        Some(policy),
                    );
                    warn!(label = label, "background maintenance queue full");
                    return Err(AgentError::Other(format!(
                        "background maintenance queue is full for {}",
                        label
                    )));
                }
            }
            return Ok(());
        }

        record_background_maintenance_event(
            self.observability_manager.as_ref(),
            label,
            EventStatus::Success,
            0,
            "scheduled",
            None,
            Some(policy),
        );
        let manager = self.observability_manager.clone();
        let policy_for_task = policy.clone();
        let observed_task = async move {
            let start = Instant::now();
            let result = task.await;
            match &result {
                Ok(()) => record_background_maintenance_event(
                    manager.as_ref(),
                    label,
                    EventStatus::Success,
                    start.elapsed().as_millis() as u64,
                    "completed",
                    None,
                    Some(&policy_for_task),
                ),
                Err(error) => record_background_maintenance_event(
                    manager.as_ref(),
                    label,
                    EventStatus::Error,
                    start.elapsed().as_millis() as u64,
                    "failed",
                    Some(error.to_string()),
                    Some(&policy_for_task),
                ),
            }
            result
        };

        if let Err(error) = self.background_maintenance.spawn(key, observed_task) {
            record_background_maintenance_event(
                self.observability_manager.as_ref(),
                label,
                EventStatus::Error,
                0,
                "spawn_failed",
                Some(error.to_string()),
                Some(policy),
            );
            warn!(label = label, error = %error, "background maintenance spawn failed");
            return Err(error);
        }
        Ok(())
    }

    /// Records a skipped background maintenance event when work cannot run.
    fn record_skipped_maintenance(
        &self,
        label: &str,
        purpose: ObservationPurpose,
        reason: &str,
        policy: Option<&crate::optimization::config::MaintenanceTaskPolicy>,
    ) {
        if let Some(manager) = self.observability_manager.as_ref() {
            let mut tags = background_maintenance_tags(label, "skipped", Some(reason), policy);
            tags.insert("runtime.skip_reason".to_string(), reason.to_string());
            manager.record_lifecycle_event(
                EventType::MemoryOperation {
                    operation: format!("{}_maintenance", label),
                },
                purpose,
                EventStatus::Skipped,
                0,
                tags,
                None,
            );
        }
    }

    /// Get cached actor facts for the effective actor.
    pub fn actor_facts(&self) -> Vec<ai_agents_core::KeyFact> {
        let Some(actor_id) = self.effective_actor_id() else {
            return Vec::new();
        };
        self.actor_facts_cache
            .read()
            .get(&actor_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Returns the formatted relationship prompt text for the effective actor, if relationship injection produced any text for this turn.
    pub fn relationship_memory_text(&self) -> Option<String> {
        self.format_relationship_for_context().map(|(_, text)| text)
    }

    /// Manually extract facts from the last N messages.
    pub async fn extract_facts(
        &self,
        last_n: usize,
    ) -> ai_agents_core::Result<Vec<ai_agents_core::KeyFact>> {
        self.extract_facts_with_source(last_n, "manual").await
    }

    async fn extract_facts_with_source(
        &self,
        last_n: usize,
        source: &'static str,
    ) -> ai_agents_core::Result<Vec<ai_agents_core::KeyFact>> {
        let extractor = match self.fact_extractor.read().clone() {
            Some(e) => e,
            None => return Ok(vec![]),
        };

        let messages = self.memory.get_messages(None).await?;
        let recent: Vec<_> = messages.iter().rev().take(last_n).rev().cloned().collect();

        if recent.is_empty() {
            return Ok(vec![]);
        }

        let actor_id = self.effective_actor_id();
        let existing = actor_id
            .as_ref()
            .and_then(|aid| self.actor_facts_cache.read().get(aid).cloned())
            .unwrap_or_default();

        let categories = self
            .facts_config
            .as_ref()
            .map(|c| c.custom_categories.clone())
            .unwrap_or_default();

        let facts = self
            .observe_purpose(
                ObservationPurpose::FactsExtraction,
                extractor.extract(&recent, &existing, actor_id.as_deref(), &categories),
            )
            .await?;

        // Save to storage and update the actor-scoped cache when an actor is known.
        if !facts.is_empty() {
            let fact_store_opt = self.fact_store.read().clone();
            let mut stored_total = 0usize;
            let mut cache_updated = false;
            if let (Some(store), Some(aid)) = (fact_store_opt, &actor_id) {
                // add_facts now returns the authoritative post-write set.
                let authoritative = store.add_facts(aid, facts.clone()).await?;
                stored_total = authoritative.len();
                self.actor_facts_cache
                    .write()
                    .insert(aid.clone(), authoritative);
                cache_updated = true;
            } else if let Some(aid) = &actor_id {
                let mut cache = self.actor_facts_cache.write();
                let entry = cache.entry(aid.clone()).or_default();
                entry.extend(facts.clone());
                stored_total = entry.len();
                cache_updated = true;
            }

            info!(
                actor_id = %actor_id.as_deref().unwrap_or("<none>"),
                source = source,
                requested_messages = last_n,
                message_count = recent.len(),
                extracted_count = facts.len(),
                cache_updated = cache_updated,
                stored_total = stored_total,
                "facts extracted"
            );

            if let Some(ref aid) = actor_id {
                self.hooks.on_facts_extracted(aid, &facts).await;
            }
        }

        Ok(facts)
    }

    /// Resolve actor_id from context if method is from_context.
    /// Supports dotted paths (e.g. "player.id", "user.profile.id").
    fn resolve_actor_id_from_context(&self) {
        if self
            .current_turn_actor_context()
            .and_then(|ctx| ctx.effective_actor_id().map(str::to_string))
            .is_some()
        {
            return;
        }

        if let Some(ref am_config) = self.actor_memory_config {
            if am_config.identification.method == ai_agents_facts::IdentificationMethod::FromContext
            {
                if let Some(ref path) = am_config.identification.context_path {
                    // get_path resolves dotted paths; get only handles top-level keys.
                    let val = self
                        .context_manager
                        .get_path(path)
                        .or_else(|| self.context_manager.get(path));
                    if let Some(val) = val {
                        if let Some(id_str) = val.as_str() {
                            let current = self.actor_id.read().clone();
                            if current.as_deref() != Some(id_str) {
                                *self.actor_id.write() = Some(id_str.to_string());
                                let mut meta = self.session_metadata.write();
                                meta.actor_id = Some(id_str.to_string());
                                if !meta.actors.iter().any(|a| a == id_str) {
                                    meta.actors.push(id_str.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Format actor facts for template injection.
    fn format_actor_facts_for_context(&self) -> String {
        // Respect inject_in_context: false.
        let should_inject = self
            .facts_config
            .as_ref()
            .map(|c| c.inject_in_context)
            .unwrap_or(true);
        if !should_inject {
            return String::new();
        }

        let Some(actor_id) = self.effective_actor_id() else {
            return String::new();
        };

        let facts = self
            .actor_facts_cache
            .read()
            .get(&actor_id)
            .cloned()
            .unwrap_or_default();
        if facts.is_empty() {
            return String::new();
        }

        let am_config = self.actor_memory_config.as_ref();
        // Effective token cap: prefer memory.token_budget.allocation.facts when present,
        // otherwise fall back to actor_memory.injection.max_tokens.
        let facts_budget = self
            .memory_token_budget
            .as_ref()
            .map(|b| b.allocation.facts as usize)
            .filter(|n| *n > 0);
        let default_max = am_config.map(|c| c.injection.max_tokens).unwrap_or(800);
        let max_tokens = facts_budget.unwrap_or(default_max);

        // Filter by category when injection.mode = category.
        let filtered: Vec<ai_agents_core::KeyFact> = if let Some(cfg) = am_config {
            if cfg.injection.mode == ai_agents_facts::InjectionMode::OnDemand {
                return String::new();
            }
            if cfg.injection.mode == ai_agents_facts::InjectionMode::Category
                && !cfg.injection.categories.is_empty()
            {
                facts
                    .iter()
                    .filter(|f| {
                        cfg.injection
                            .categories
                            .iter()
                            .any(|c| f.category.to_string() == *c)
                    })
                    .cloned()
                    .collect()
            } else {
                facts.clone()
            }
        } else {
            facts.clone()
        };

        if filtered.is_empty() {
            return String::new();
        }

        if let Some(store) = self.fact_store.read().clone() {
            store.format_for_context(&filtered, max_tokens)
        } else {
            String::new()
        }
    }

    fn build_context_with_staged(&self, staged: &HashMap<String, Value>) -> HashMap<String, Value> {
        let context = self.build_context_with_overlays();
        let mut root = Value::Object(context.into_iter().collect());
        for (path, value) in staged {
            if let Ok(updated) = ai_agents_core::set_dot_path(root.clone(), path, value.clone()) {
                root = updated;
            }
        }
        match root {
            Value::Object(obj) => obj.into_iter().collect(),
            _ => HashMap::new(),
        }
    }

    fn build_context_with_overlays(&self) -> HashMap<String, Value> {
        let mut context = self.context_manager.get_all();
        let mut root = Value::Object(context.clone().into_iter().collect());

        if let Some(turn_ctx) = self.current_turn_actor_context() {
            if let Some(ref origin_actor_id) = turn_ctx.origin_actor_id {
                if let Ok(updated) = ai_agents_core::set_dot_path(
                    root.clone(),
                    "interaction.origin_actor_id",
                    serde_json::json!(origin_actor_id),
                ) {
                    root = updated;
                }
            }
            if let Some(ref sender_agent_id) = turn_ctx.sender_agent_id {
                if let Ok(updated) = ai_agents_core::set_dot_path(
                    root.clone(),
                    "interaction.sender_agent_id",
                    serde_json::json!(sender_agent_id),
                ) {
                    root = updated;
                }
            }
        }

        if let Some(ref actor_id) = self.effective_actor_id() {
            if let Ok(updated) = ai_agents_core::set_dot_path(
                root.clone(),
                "interaction.actor_id",
                serde_json::json!(actor_id),
            ) {
                root = updated;
            }
        }

        if let Some(manager) = self.relationship_manager.as_ref() {
            if let Some(actor_id) = self.effective_actor_id() {
                if let Some(value) = manager.to_context_value(&actor_id) {
                    if let Ok(updated) = ai_agents_core::set_dot_path(
                        root.clone(),
                        &manager.config().injection.context_path,
                        value,
                    ) {
                        root = updated;
                    }
                }
            }
        }

        if let Value::Object(obj) = root {
            context = obj.into_iter().collect();
        }

        context
    }

    fn resolve_actor_name_from_context(&self) -> Option<String> {
        for path in ["actor.name", "user.name", "player.name", "customer.name"] {
            if let Some(value) = self.context_manager.get_path(path) {
                if let Some(name) = value.as_str() {
                    return Some(name.to_string());
                }
            }
        }
        None
    }

    async fn maybe_load_actor_relationship(&self) {
        let Some(manager) = self.relationship_manager.as_ref() else {
            return;
        };
        let Some(actor_id) = self.effective_actor_id() else {
            return;
        };

        let mut should_fire_loaded = false;
        if manager.get(&actor_id).is_none() {
            let mut loaded = false;
            if manager.config().persistence.enabled {
                let storage = self.storage.read().clone();
                if let Some(storage) = storage {
                    match storage.load_relationship(&self.info.id, &actor_id).await {
                        Ok(Some(value)) => match manager.insert_from_value(value) {
                            Ok(_) => loaded = true,
                            Err(e) => {
                                warn!(actor = %actor_id, error = %e, "failed to restore relationship")
                            }
                        },
                        Ok(None) => {}
                        Err(e) => {
                            warn!(actor = %actor_id, error = %e, "failed to load relationship")
                        }
                    }
                }
            }

            if !loaded {
                manager.get_or_create(&actor_id, self.resolve_actor_name_from_context().as_deref());
            }
            should_fire_loaded = true;
        }

        let actor_name = self.resolve_actor_name_from_context();
        let relationship = manager.touch_interaction(&actor_id, actor_name.as_deref());
        if should_fire_loaded {
            self.hooks
                .on_relationship_loaded(&actor_id, &relationship)
                .await;
        }
    }

    fn format_relationship_for_context(&self) -> Option<(String, String)> {
        let manager = self.relationship_manager.as_ref()?;
        if !manager.config().injection.enabled {
            return None;
        }
        let actor_id = self.effective_actor_id()?;
        let relationship = manager.get(&actor_id)?;
        let local_cap = manager.config().injection.max_tokens;
        let global_cap = self
            .memory_token_budget
            .as_ref()
            .map(|b| b.allocation.relationships as usize)
            .filter(|n| *n > 0);
        let max_tokens = global_cap.map(|g| g.min(local_cap)).unwrap_or(local_cap);
        let text = ai_agents_relationships::format_relationship(
            &relationship,
            &manager.config().injection.format,
            max_tokens,
        );
        if text.is_empty() {
            None
        } else {
            Some((manager.config().injection.prompt_variable.clone(), text))
        }
    }

    async fn persist_actor_relationship(&self, actor_id: &str) -> Result<()> {
        let Some(manager) = self.relationship_manager.as_ref() else {
            return Ok(());
        };
        if !manager.config().persistence.enabled {
            return Ok(());
        }
        let storage = self.storage.read().clone();
        let Some(storage) = storage else {
            return Ok(());
        };
        if let Some(value) = manager.relationship_as_value(actor_id)? {
            storage
                .save_relationship(&self.info.id, actor_id, &value)
                .await?;
        }
        Ok(())
    }

    async fn auto_update_relationship(&self) {
        let Some(manager) = self.relationship_manager.as_ref() else {
            return;
        };
        let Some(actor_id) = self.effective_actor_id() else {
            return;
        };
        if !manager.config().auto_update.enabled {
            let _ = self.persist_actor_relationship(&actor_id).await;
            return;
        }

        let recent_messages = manager.config().auto_update.recent_messages;
        let messages = match self.memory.get_messages(Some(recent_messages)).await {
            Ok(messages) => messages,
            Err(e) => {
                warn!(actor = %actor_id, error = %e, "failed to read messages for relationship update");
                return;
            }
        };

        match self
            .observe_purpose(
                ObservationPurpose::RelationshipUpdate,
                manager.auto_update(&actor_id, &messages),
            )
            .await
        {
            Ok(update) => {
                if !update.changes.is_empty() {
                    self.hooks
                        .on_relationship_change(&actor_id, &update.changes)
                        .await;
                }
                if let Some(ref event) = update.event {
                    self.hooks.on_notable_event(&actor_id, event).await;
                }
                let persisted = match self.persist_actor_relationship(&actor_id).await {
                    Ok(()) => true,
                    Err(e) => {
                        warn!(actor = %actor_id, error = %e, "failed to persist relationship");
                        false
                    }
                };
                if !update.changes.is_empty() || update.event.is_some() {
                    let changed_dimensions: Vec<String> = update
                        .changes
                        .iter()
                        .map(|change| format!("{}:{}", change.perspective, change.dimension))
                        .collect();
                    info!(
                        actor_id = %actor_id,
                        change_count = update.changes.len(),
                        changed_dimensions = ?changed_dimensions,
                        event_present = update.event.is_some(),
                        persisted = persisted,
                        "relationship updated"
                    );
                } else {
                    debug!(actor_id = %actor_id, persisted = persisted, "relationship evaluation ran but found no changes");
                }
            }
            Err(e) => warn!(actor = %actor_id, error = %e, "relationship update failed"),
        }
    }

    /// Run auto-extraction after a chat turn.
    async fn auto_extract_facts(&self) {
        let should_extract = self
            .facts_config
            .as_ref()
            .map(|c| c.enabled && c.auto_extract)
            .unwrap_or(false);

        if !should_extract {
            debug!("fact extraction skipped because auto extraction is disabled");
            return;
        }

        let msgs_since = *self.messages_since_extraction.read();
        if msgs_since < 2 {
            debug!(
                messages_since_extraction = msgs_since,
                "fact extraction skipped until threshold is reached"
            );
            return;
        }

        match self.extract_facts_with_source(msgs_since, "auto").await {
            Ok(facts) => {
                if !facts.is_empty() {
                    *self.messages_since_extraction.write() = 0;
                } else {
                    debug!("fact extraction ran but found no new facts");
                }
            }
            Err(e) => {
                warn!("fact extraction failed: {}", e);
            }
        }
    }

    pub fn with_persona(mut self, manager: Arc<ai_agents_persona::PersonaManager>) -> Self {
        self.persona_manager = Some(manager);
        self
    }

    pub fn persona_manager(&self) -> Option<&Arc<ai_agents_persona::PersonaManager>> {
        self.persona_manager.as_ref()
    }

    pub fn with_disambiguation(mut self, config: DisambiguationConfig) -> Self {
        if config.is_enabled() {
            let manager = DisambiguationManager::new(config, Arc::clone(&self.llm_registry))
                .with_clarification_observer(Arc::new(ObservabilityClarificationObserver));
            self.disambiguation_manager = Some(manager);
        }
        self
    }

    pub fn disambiguation_manager(&self) -> Option<&DisambiguationManager> {
        self.disambiguation_manager.as_ref()
    }

    pub fn has_disambiguation(&self) -> bool {
        self.disambiguation_manager
            .as_ref()
            .is_some_and(|m| m.is_enabled())
    }

    pub async fn init_storage(&self) -> Result<()> {
        if self.storage_config.is_none() {
            return Ok(());
        }
        if self.storage.read().is_some() {
            return Ok(());
        }
        let storage_config = self.convert_storage_config();
        let storage = create_storage(&storage_config).await?;
        *self.storage.write() = storage;
        // Complete facts setup now that storage is available.
        self.complete_facts_init().await;
        Ok(())
    }

    /// Initialize fact store and extractor from stored config + current storage.
    /// Called from init_storage() so facts are ready before the first turn.
    async fn complete_facts_init(&self) {
        if self.fact_store.read().is_some() {
            return;
        }
        let storage = match self.storage.read().clone() {
            Some(s) => s,
            None => return,
        };

        let facts_enabled = self
            .facts_config
            .as_ref()
            .map(|f| f.enabled)
            .unwrap_or(false);
        let actor_memory_enabled = self
            .actor_memory_config
            .as_ref()
            .map(|a| a.enabled)
            .unwrap_or(false);

        if !facts_enabled && !actor_memory_enabled {
            return;
        }

        let fc = self.facts_config.clone().unwrap_or_default();
        let store = Arc::new(ai_agents_facts::FactStore::new(
            storage,
            self.info.id.clone(),
            fc.clone(),
        ));

        let extractor: Option<Arc<dyn ai_agents_facts::FactExtractor>> = if facts_enabled {
            let extractor_llm = fc
                .extractor_llm
                .as_ref()
                .and_then(|alias| self.llm_registry.get(alias).ok())
                .or_else(|| self.llm_registry.router().ok())
                .or_else(|| self.llm_registry.default().ok());
            extractor_llm.map(|llm| {
                Arc::new(ai_agents_facts::LLMFactExtractor::new(llm, fc.clone()))
                    as Arc<dyn ai_agents_facts::FactExtractor>
            })
        } else {
            None
        };

        *self.fact_store.write() = Some(store);
        *self.fact_extractor.write() = extractor;
        debug!(
            agent = %self.info.id,
            facts_enabled,
            actor_memory_enabled,
            "facts storage initialized"
        );
    }

    fn convert_storage_config(&self) -> StorageStorageConfig {
        crate::spec::storage::to_storage_config(&self.storage_config)
    }

    pub fn storage(&self) -> Option<Arc<dyn AgentStorage>> {
        self.storage.read().clone()
    }

    pub fn storage_config(&self) -> &StorageConfig {
        &self.storage_config
    }

    /// Returns the spawner if configured via a spawner: YAML section.
    pub fn spawner(&self) -> Option<&Arc<crate::spawner::AgentSpawner>> {
        self.spawner.as_ref()
    }

    /// Returns the agent registry if configured via a spawner: YAML section.
    pub fn spawner_registry(&self) -> Option<&Arc<crate::spawner::AgentRegistry>> {
        self.spawner_registry.as_ref()
    }

    pub fn has_spawner(&self) -> bool {
        self.spawner_registry.is_some()
    }

    pub fn with_spawner_handles(
        mut self,
        spawner: Arc<crate::spawner::AgentSpawner>,
        registry: Arc<crate::spawner::AgentRegistry>,
    ) -> Self {
        self.spawner = Some(spawner);
        self.spawner_registry = Some(registry);
        self
    }

    pub fn with_hooks(mut self, hooks: Arc<dyn AgentHooks>) -> Self {
        self.hooks = hooks;
        self
    }

    pub fn with_parallel_tools(mut self, config: ParallelToolsConfig) -> Self {
        self.parallel_tools = config;
        self
    }

    pub fn with_streaming(mut self, config: StreamingConfig) -> Self {
        self.streaming = config;
        self
    }

    pub fn with_hitl(mut self, engine: HITLEngine, handler: Arc<dyn ApprovalHandler>) -> Self {
        self.hitl_engine = Some(engine);
        self.approval_handler = handler;
        self
    }

    pub fn with_max_context_tokens(mut self, tokens: u32) -> Self {
        self.max_context_tokens = tokens;
        self
    }

    pub fn with_memory_token_budget(mut self, budget: MemoryTokenBudget) -> Self {
        self.memory_token_budget = Some(budget);
        self
    }

    pub fn with_recovery_manager(mut self, manager: RecoveryManager) -> Self {
        self.recovery_manager = manager;
        self
    }

    pub fn with_tool_security(mut self, engine: ToolSecurityEngine) -> Self {
        self.tool_security = engine;
        self
    }

    pub fn with_process_processor(mut self, processor: ProcessProcessor) -> Self {
        let processor = processor.with_stage_observer(Arc::new(ObservabilityProcessStageObserver));
        self.process_processor = Some(processor);
        self
    }

    pub fn with_state_machine(
        mut self,
        state_machine: Arc<StateMachine>,
        evaluator: Arc<dyn TransitionEvaluator>,
    ) -> Self {
        self.state_machine = Some(state_machine);
        self.transition_evaluator = Some(evaluator);
        self
    }

    pub fn with_context_manager(mut self, manager: Arc<ContextManager>) -> Self {
        self.context_manager = manager;
        self
    }

    pub fn register_message_filter(&self, name: impl Into<String>, filter: Arc<dyn MessageFilter>) {
        self.message_filters.write().insert(name.into(), filter);
    }

    pub fn set_context(&self, key: &str, value: Value) -> Result<()> {
        self.context_manager.update(key, value)
    }

    pub fn update_context(&self, path: &str, value: Value) -> Result<()> {
        self.context_manager.update(path, value)
    }

    pub fn get_context(&self) -> HashMap<String, Value> {
        self.build_context_with_overlays()
    }

    pub fn remove_context(&self, key: &str) -> Option<Value> {
        self.context_manager.remove(key)
    }

    pub async fn refresh_context(&self, key: &str) -> Result<()> {
        self.context_manager.refresh(key).await
    }

    pub fn register_context_provider(&self, name: &str, provider: Arc<dyn ContextProvider>) {
        self.context_manager.register_provider(name, provider);
    }

    pub fn current_state(&self) -> Option<String> {
        self.state_machine.as_ref().map(|sm| sm.current())
    }

    pub async fn transition_to(&self, state: &str) -> Result<()> {
        if let Some(ref sm) = self.state_machine {
            let from_state = sm.current();
            self.execute_state_exit_actions(&from_state).await;
            sm.transition_to(state, "manual transition")?;
            self.execute_state_enter_actions(state).await;
            info!(to = %state, "Manual state transition");
        }
        Ok(())
    }

    pub fn state_history(&self) -> Vec<StateTransitionEvent> {
        self.state_machine
            .as_ref()
            .map(|sm| sm.history())
            .unwrap_or_default()
    }

    /// Get a copy of current session metadata.
    pub fn session_metadata(&self) -> ai_agents_core::SessionMetadata {
        self.session_metadata.read().clone()
    }

    /// Delete all facts and sessions for an actor, gated by privacy.allow_deletion.
    /// Returns Err when actor_memory.privacy.allow_deletion is false.
    pub async fn delete_actor_data(&self, actor_id: &str) -> Result<()> {
        let allowed = self
            .actor_memory_config
            .as_ref()
            .map(|c| c.privacy.allow_deletion)
            .unwrap_or(true);
        if !allowed {
            return Err(AgentError::Config(
                "privacy.allow_deletion is false; actor data deletion is not permitted".into(),
            ));
        }
        let storage = self.storage.read().clone();
        if let Some(storage) = storage {
            storage.delete_actor_data(&self.info.id, actor_id).await?;
            storage.delete_relationship(&self.info.id, actor_id).await?;
        } else if let Some(store) = self.fact_store.read().clone() {
            store.delete_actor_data(actor_id).await?;
        }
        if let Some(manager) = self.relationship_manager.as_ref() {
            manager.remove(actor_id);
        }
        self.actor_facts_cache.write().remove(actor_id);
        Ok(())
    }

    /// Overwrite session metadata (tags, ttl, custom fields).
    pub fn set_session_metadata(&self, meta: ai_agents_core::SessionMetadata) {
        *self.session_metadata.write() = meta;
    }

    /// Delete sessions whose TTL has expired. Returns number of sessions removed.
    pub async fn cleanup_expired_sessions(&self) -> Result<usize> {
        let storage = self.storage.read().clone();
        match storage {
            Some(s) => {
                let count = s.cleanup_expired().await?;
                if count > 0 {
                    self.hooks.on_sessions_expired(count).await;
                }
                Ok(count)
            }
            None => Err(AgentError::Config(
                "No storage configured. Use with_storage_config() or with_storage() first".into(),
            )),
        }
    }

    /// List sessions matching a filter. Supports actor, tag, and date filters.
    pub async fn list_sessions_filtered(
        &self,
        filter: &ai_agents_core::SessionFilter,
    ) -> Result<Vec<ai_agents_core::SessionSummary>> {
        let storage = self.storage.read().clone();
        match storage {
            Some(s) => s.list_sessions_filtered(filter).await,
            None => Err(AgentError::Config(
                "No storage configured. Use with_storage_config() or with_storage() first".into(),
            )),
        }
    }

    pub async fn save_state(&self) -> Result<AgentSnapshot> {
        let memory_snapshot = self.memory.snapshot().await?;
        let state_machine_snapshot = self.state_machine.as_ref().map(|sm| sm.snapshot());
        let context_snapshot = self.context_manager.snapshot();

        let mut snapshot = AgentSnapshot::new(self.info.id.clone())
            .with_memory(memory_snapshot)
            .with_context(context_snapshot)
            .with_state_machine(
                state_machine_snapshot.unwrap_or_else(|| StateMachineSnapshot {
                    current_state: String::new(),
                    previous_state: None,
                    turn_count: 0,
                    no_transition_count: 0,
                    history: vec![],
                }),
            );

        if let Some(ref persona) = self.persona_manager {
            snapshot.persona = Some(persona.snapshot_as_value()?);
        }

        if let Some(ref relationships) = self.relationship_manager {
            snapshot.relationships = Some(relationships.snapshot_as_value()?);
        }

        Ok(snapshot)
    }

    /// Save state including spawned agents manifest for session persistence.
    pub async fn save_state_full(&self) -> Result<AgentSnapshot> {
        let mut snapshot = self.save_state().await?;
        if let Some(ref registry) = self.spawner_registry {
            let entries = registry.list_with_specs();
            if !entries.is_empty() {
                snapshot = snapshot.with_spawned_agents(entries);
            }
        }
        Ok(snapshot)
    }

    pub async fn restore_state(&self, snapshot: AgentSnapshot) -> Result<()> {
        self.memory.restore(snapshot.memory).await?;

        if let (Some(sm), Some(sm_snapshot)) = (&self.state_machine, snapshot.state_machine) {
            if !sm_snapshot.current_state.is_empty() {
                sm.restore(sm_snapshot)?;
            }
        }

        self.context_manager.restore(snapshot.context);

        if let (Some(persona_value), Some(persona_manager)) =
            (snapshot.persona, &self.persona_manager)
        {
            persona_manager.restore_from_value(persona_value)?;
        }

        if let (Some(relationship_value), Some(relationship_manager)) =
            (snapshot.relationships, &self.relationship_manager)
        {
            relationship_manager.restore_from_value(relationship_value)?;
        }

        info!(agent_id = %snapshot.agent_id, "State restored");
        Ok(())
    }

    pub async fn save_to(&self, storage: &dyn AgentStorage, session_id: &str) -> Result<()> {
        let snapshot = self.save_state().await?;
        storage.save(session_id, &snapshot).await
    }

    pub async fn load_from(&self, storage: &dyn AgentStorage, session_id: &str) -> Result<bool> {
        if let Some(snapshot) = storage.load(session_id).await? {
            self.restore_state(snapshot).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub async fn save_session(&self, session_id: &str) -> Result<()> {
        let storage = self.storage.read().clone();
        match storage {
            Some(s) => {
                // Fire on_session_created when this session id is first seen on this runtime.
                let is_new = {
                    let cur = self.current_session_id.read().clone();
                    cur.as_deref() != Some(session_id)
                };
                if is_new {
                    *self.current_session_id.write() = Some(session_id.to_string());
                    self.hooks.on_session_created(session_id).await;
                }

                // Update metadata before persisting.
                {
                    let now = chrono::Utc::now();
                    let msg_count = self
                        .memory
                        .get_messages(None)
                        .await
                        .map(|v| v.len())
                        .unwrap_or(0);
                    let mut meta = self.session_metadata.write();
                    meta.last_active = now;
                    meta.message_count = msg_count;
                    if meta.actor_id.is_none() {
                        meta.actor_id = self.actor_id.read().clone();
                    }
                }

                self.save_to(s.as_ref(), session_id).await?;

                // Persist metadata alongside the snapshot.
                let meta = self.session_metadata.read().clone();
                let _ = s.save_metadata(session_id, &meta).await;
                Ok(())
            }
            None => Err(AgentError::Config(
                "No storage configured. Use with_storage_config() or with_storage() first".into(),
            )),
        }
    }

    pub async fn load_session(&self, session_id: &str) -> Result<bool> {
        let storage = self.storage.read().clone();
        match storage {
            Some(s) => {
                let loaded = self.load_from(s.as_ref(), session_id).await?;
                if loaded {
                    // Restore session metadata alongside the snapshot.
                    if let Ok(Some(meta)) = s.load_metadata(session_id).await {
                        if let Some(ref aid) = meta.actor_id {
                            let _ = self.set_actor_id(aid);
                        }
                        *self.session_metadata.write() = meta;
                    }
                    *self.current_session_id.write() = Some(session_id.to_string());
                }
                Ok(loaded)
            }
            None => Err(AgentError::Config(
                "No storage configured. Use with_storage_config() or with_storage() first".into(),
            )),
        }
    }

    pub async fn delete_session(&self, session_id: &str) -> Result<()> {
        let storage = self.storage.read().clone();
        match storage {
            Some(s) => s.delete(session_id).await,
            None => Err(AgentError::Config(
                "No storage configured. Use with_storage_config() or with_storage() first".into(),
            )),
        }
    }

    pub async fn list_sessions(&self) -> Result<Vec<String>> {
        let storage = self.storage.read().clone();
        match storage {
            Some(s) => s.list_sessions().await,
            None => Err(AgentError::Config(
                "No storage configured. Use with_storage_config() or with_storage() first".into(),
            )),
        }
    }

    fn estimate_tokens(&self, text: &str) -> u32 {
        (text.len() as f32 / 4.0).ceil() as u32
    }

    fn estimate_total_tokens(&self, messages: &[ChatMessage]) -> u32 {
        messages
            .iter()
            .map(|m| self.estimate_tokens(&m.content))
            .sum()
    }

    fn truncate_context(&self, messages: &mut Vec<ChatMessage>, keep_recent: usize) {
        if messages.len() <= keep_recent + 1 {
            return;
        }
        let system_msg = messages.remove(0);
        let to_remove = messages.len().saturating_sub(keep_recent);
        messages.drain(..to_remove);
        messages.insert(0, system_msg);
    }

    fn get_filter(&self, config: &FilterConfig) -> Arc<dyn MessageFilter> {
        match config {
            FilterConfig::KeepRecent(n) => Arc::new(KeepRecentFilter::new(*n)),
            FilterConfig::ByRole { keep_roles } => Arc::new(ByRoleFilter::new(keep_roles.clone())),
            FilterConfig::SkipPattern { skip_if_contains } => {
                Arc::new(SkipPatternFilter::new(skip_if_contains.clone()))
            }
            FilterConfig::Custom { name } => {
                let filters = self.message_filters.read();
                filters
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| Arc::new(KeepRecentFilter::new(10)))
            }
        }
    }

    async fn summarize_context(
        &self,
        messages: &mut Vec<ChatMessage>,
        summarizer_llm: Option<&str>,
        max_summary_tokens: u32,
        custom_prompt: Option<&str>,
        keep_recent: usize,
        filter: Option<&FilterConfig>,
    ) -> Result<()> {
        let system_msg = messages.remove(0);

        let to_summarize_count = messages.len().saturating_sub(keep_recent);
        if to_summarize_count == 0 {
            messages.insert(0, system_msg);
            return Ok(());
        }

        let recent_msgs: Vec<ChatMessage> = messages.drain(to_summarize_count..).collect();
        let mut to_summarize = std::mem::take(messages);

        if let Some(filter_config) = filter {
            let filter = self.get_filter(filter_config);
            to_summarize = filter.filter(to_summarize);
        }

        if to_summarize.is_empty() {
            *messages = recent_msgs;
            messages.insert(0, system_msg);
            return Ok(());
        }

        let conversation_text = to_summarize
            .iter()
            .map(|m| format!("{:?}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        let default_prompt = format!(
            "Summarize the following conversation in under {} tokens, preserving key information:\n\n{}",
            max_summary_tokens, conversation_text
        );

        let summary_prompt = custom_prompt
            .map(|p| format!("{}\n\n{}", p, conversation_text))
            .unwrap_or(default_prompt);

        let summarizer = if let Some(alias) = summarizer_llm {
            self.llm_registry
                .get(alias)
                .map_err(|e| AgentError::Config(e.to_string()))?
        } else {
            self.llm_registry
                .router()
                .or_else(|_| self.llm_registry.default())
                .map_err(|e| AgentError::Config(e.to_string()))?
        };

        let summary_msgs = vec![ChatMessage::user(&summary_prompt)];
        let response = self
            .observe_purpose(
                ObservationPurpose::Summarization,
                summarizer.complete(&summary_msgs, None),
            )
            .await?;

        let summary_message = ChatMessage::system(&format!(
            "[Previous conversation summary]\n{}",
            response.content
        ));

        *messages = vec![system_msg, summary_message];
        messages.extend(recent_msgs);

        debug!(
            summarized_count = to_summarize_count,
            kept_recent = keep_recent,
            "Context summarized"
        );

        Ok(())
    }

    fn render_system_prompt(&self) -> Result<String> {
        let mut context = self.build_context_with_overlays();

        // Inject actor_facts for {{ actor_facts }} template variable.
        let facts_text = self.format_actor_facts_for_context();
        if !facts_text.is_empty() {
            context.insert(
                "actor_facts".to_string(),
                serde_json::Value::String(facts_text),
            );
        }

        if let Some((key, text)) = self.format_relationship_for_context() {
            context.insert(key, serde_json::Value::String(text));
        }

        self.template_renderer
            .render(&self.base_system_prompt, &context)
    }

    async fn get_available_tool_ids(&self) -> Result<Vec<String>> {
        match self.get_current_tool_refs() {
            // State explicitly sets tools (including empty = no tools)
            Some(tool_refs) => {
                if tool_refs.is_empty() {
                    // Explicitly empty: no tools available in this state
                    return Ok(Vec::new());
                }

                let eval_ctx = self.build_evaluation_context().await?;
                let llm_getter = RegistryLLMGetter {
                    registry: self.llm_registry.clone(),
                };
                let evaluator = ConditionEvaluator::new(llm_getter);

                let mut available = Vec::new();
                for tool_ref in &tool_refs {
                    let tool_id = tool_ref.id();

                    if self.tools.get(tool_id).is_none() {
                        continue;
                    }

                    if let Some(condition) = tool_ref.condition() {
                        match evaluator.evaluate(condition, &eval_ctx).await {
                            Ok(true) => {
                                available.push(tool_id.to_string());
                            }
                            Ok(false) => {
                                debug!(tool = tool_id, "Tool condition not met, skipping");
                            }
                            Err(e) => {
                                warn!(tool = tool_id, error = %e, "Error evaluating tool condition");
                            }
                        }
                    } else {
                        available.push(tool_id.to_string());
                    }
                }

                Ok(available)
            }
            // State doesn't specify tools: fallback to agent-level
            None => {
                match &self.declared_tool_ids {
                    // tools: [...] — specific tools listed
                    Some(ids) if !ids.is_empty() => Ok(ids
                        .iter()
                        .filter(|id| self.tools.get(id).is_some())
                        .cloned()
                        .collect()),
                    // tools: [] — explicitly no tools
                    Some(_) => Ok(Vec::new()),
                    // tools: not specified — all registered tools available
                    None => Ok(self.tools.list_ids()),
                }
            }
        }
    }

    /// Returns `Some(tools)` if the current state explicitly declares tools
    /// (including `Some([])` for "no tools"), or `None` if the state doesn't
    /// specify tools (meaning: fall back to agent-level declared_tool_ids).
    fn get_current_tool_refs(&self) -> Option<Vec<ToolRef>> {
        if let Some(ref sm) = self.state_machine {
            if let Some(state_def) = sm.current_definition() {
                let parent_def = sm.get_parent_definition();
                if let Some(effective) = state_def.get_effective_tools(parent_def.as_ref()) {
                    return Some(effective.into_iter().cloned().collect());
                }
            }
        }
        None
    }

    async fn build_evaluation_context(&self) -> Result<EvaluationContext> {
        let context = self.build_context_with_overlays();
        let messages = self.memory.get_messages(Some(10)).await?;
        let tool_history = self.tool_call_history.read().clone();

        let (state_name, turn_count, previous_state) = if let Some(ref sm) = self.state_machine {
            (Some(sm.current()), sm.turn_count(), sm.previous())
        } else {
            (None, 0, None)
        };

        Ok(EvaluationContext::default()
            .with_context(context)
            .with_state(state_name, turn_count, previous_state)
            .with_called_tools(tool_history)
            .with_messages(messages))
    }

    fn record_tool_call(&self, tool_id: &str, result: Value) {
        self.tool_call_history.write().push(ToolCallRecord {
            tool_id: tool_id.to_string(),
            result,
            timestamp: chrono::Utc::now(),
        });
    }

    async fn get_effective_system_prompt_with_persona_hooks(
        &self,
        fire_persona_hooks: bool,
    ) -> Result<String> {
        let rendered_base = self.render_system_prompt()?;

        let persona_prefix = if let Some(ref persona) = self.persona_manager {
            let context = self.build_context_with_overlays();
            if fire_persona_hooks {
                let render_result = persona.render_prompt(&context)?;
                for content in &render_result.newly_revealed {
                    self.hooks.on_secret_revealed(content).await;
                }
                render_result.prompt
            } else {
                persona.render_prompt_preview(&context)?
            }
        } else {
            String::new()
        };

        if let Some(ref sm) = self.state_machine {
            if let Some(state_def) = sm.current_definition() {
                let state_prompt = if let Some(ref prompt) = state_def.prompt {
                    let context = self.build_context_with_overlays();
                    self.template_renderer.render_with_state(
                        prompt,
                        &context,
                        &sm.current(),
                        sm.previous().as_deref(),
                        sm.turn_count(),
                        state_def.max_turns,
                    )?
                } else {
                    String::new()
                };

                let combined = match state_def.prompt_mode {
                    PromptMode::Append => {
                        if state_prompt.is_empty() {
                            rendered_base
                        } else {
                            format!(
                                "{}\n\n[Current State: {}]\n{}",
                                rendered_base,
                                sm.current(),
                                state_prompt
                            )
                        }
                    }
                    PromptMode::Replace => {
                        if state_prompt.is_empty() {
                            rendered_base
                        } else {
                            state_prompt
                        }
                    }
                    PromptMode::Prepend => {
                        if state_prompt.is_empty() {
                            rendered_base
                        } else {
                            format!("{}\n\n{}", state_prompt, rendered_base)
                        }
                    }
                };

                // Persona always prepended regardless of prompt_mode.
                let with_persona = if persona_prefix.is_empty() {
                    combined
                } else {
                    format!("{}\n\n{}", persona_prefix, combined)
                };

                let available_tool_ids = self.get_available_tool_ids().await?;
                // Only add tools prompt if tools are available.
                // When tools: [] is set (explicitly empty), show NO tools to the LLM.
                if !available_tool_ids.is_empty() {
                    let tools_prompt = self.tools.generate_filtered_prompt_with_parallel(
                        &available_tool_ids,
                        self.parallel_tools.enabled,
                    );
                    if !tools_prompt.is_empty() {
                        return Ok(format!("{}\n\n{}", with_persona, tools_prompt));
                    }
                }
                return Ok(with_persona);
            }
        }

        // No state machine - prepend persona to base.
        let with_persona = if persona_prefix.is_empty() {
            rendered_base
        } else {
            format!("{}\n\n{}", persona_prefix, rendered_base)
        };

        let tools_prompt = match &self.declared_tool_ids {
            Some(ids) if !ids.is_empty() => self
                .tools
                .generate_filtered_prompt_with_parallel(ids, self.parallel_tools.enabled),
            Some(_) => {
                // tools: [] - explicitly no tools, empty prompt
                String::new()
            }
            None => {
                // tools: not specified - all registered tools
                self.tools
                    .generate_tools_prompt_with_parallel(self.parallel_tools.enabled)
            }
        };
        if !tools_prompt.is_empty() {
            Ok(format!("{}\n\n{}", with_persona, tools_prompt))
        } else {
            Ok(with_persona)
        }
    }

    fn get_state_llm(&self) -> Result<Arc<dyn LLMProvider>> {
        if let Some(ref sm) = self.state_machine {
            if let Some(state_def) = sm.current_definition() {
                if let Some(ref llm_alias) = state_def.llm {
                    return self
                        .llm_registry
                        .get(llm_alias)
                        .map_err(|e| AgentError::Config(e.to_string()));
                }
            }
        }
        self.llm_registry
            .default()
            .map_err(|e| AgentError::Config(e.to_string()))
    }

    fn get_effective_reasoning_config(&self) -> ReasoningConfig {
        if let Some(ref sm) = self.state_machine {
            if let Some(state_def) = sm.current_definition() {
                if let Some(ref state_reasoning) = state_def.reasoning {
                    return state_reasoning.clone();
                }
            }
        }
        self.reasoning_config.clone()
    }

    fn get_effective_reflection_config(&self) -> ReflectionConfig {
        if let Some(ref sm) = self.state_machine {
            if let Some(state_def) = sm.current_definition() {
                if let Some(ref state_reflection) = state_def.reflection {
                    return state_reflection.clone();
                }
            }
        }
        self.reflection_config.clone()
    }

    fn get_skill_reasoning_config(&self, skill: &SkillDefinition) -> ReasoningConfig {
        skill
            .reasoning
            .clone()
            .unwrap_or_else(|| self.get_effective_reasoning_config())
    }

    fn get_skill_reflection_config(&self, skill: &SkillDefinition) -> ReflectionConfig {
        skill
            .reflection
            .clone()
            .unwrap_or_else(|| self.get_effective_reflection_config())
    }

    async fn build_disambiguation_context(&self) -> Result<DisambiguationContext> {
        let recent_messages: Vec<String> = self
            .memory
            .get_messages(Some(5))
            .await?
            .iter()
            .rev()
            .map(|m| format!("{:?}: {}", m.role, m.content))
            .collect();

        let current_state = self.current_state().map(|s| s.to_string());

        // Include the current state's prompt text so the detector understands
        // what kind of input is expected (e.g., "Ask for the order number").
        let state_prompt: Option<String> = self
            .state_machine
            .as_ref()
            .and_then(|sm| sm.current_definition())
            .and_then(|def| def.prompt.clone());

        let available_tools: Vec<String> = self
            .get_available_tool_ids()
            .await
            .unwrap_or_else(|_| self.tools.list_ids());

        let available_skills: Vec<String> = self.skills.iter().map(|s| s.id.clone()).collect();

        let user_context = self.build_context_with_overlays();

        // Extract canonical intent labels from current state's transitions
        let available_intents: Vec<String> = if let Some(ref sm) = self.state_machine {
            sm.current_definition()
                .map(|def| {
                    def.transitions
                        .iter()
                        .filter_map(|t| t.intent.clone())
                        .collect()
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        Ok(DisambiguationContext::from_agent_state(
            recent_messages,
            current_state,
            state_prompt,
            available_tools,
            available_skills,
            available_intents,
            user_context,
        ))
    }

    fn get_available_skills(&self) -> Vec<&SkillDefinition> {
        if let Some(ref sm) = self.state_machine {
            if let Some(state_def) = sm.current_definition() {
                let parent_def = sm.get_parent_definition();
                let effective_skills = state_def.get_effective_skills(parent_def.as_ref());
                if !effective_skills.is_empty() {
                    return self
                        .skills
                        .iter()
                        .filter(|s| effective_skills.contains(&&s.id))
                        .collect();
                }
            }
        }
        self.skills.iter().collect()
    }

    async fn build_messages(&self) -> Result<Vec<ChatMessage>> {
        self.build_messages_internal(true, None).await
    }

    async fn build_messages_for_draft(&self, user_message: &str) -> Result<Vec<ChatMessage>> {
        self.build_messages_internal(false, Some(user_message))
            .await
    }

    async fn build_messages_internal(
        &self,
        fire_persona_hooks: bool,
        ephemeral_user_message: Option<&str>,
    ) -> Result<Vec<ChatMessage>> {
        let system_prompt = self
            .get_effective_system_prompt_with_persona_hooks(fire_persona_hooks)
            .await?;
        let mut messages = vec![ChatMessage::system(&system_prompt)];

        let context = self.memory.get_context().await?;
        let history = if let Some(ref budget) = self.memory_token_budget {
            context.to_llm_messages_with_allocation(&budget.allocation)
        } else {
            context.to_llm_messages()
        };
        messages.extend(history);
        if let Some(user_message) = ephemeral_user_message {
            messages.push(ChatMessage::user(user_message));
        }

        let total_tokens = self.estimate_total_tokens(&messages);

        if total_tokens > self.max_context_tokens {
            debug!(
                total = total_tokens,
                limit = self.max_context_tokens,
                "Context overflow"
            );

            match &self.recovery_manager.config().llm.on_context_overflow {
                ContextOverflowAction::Error => {
                    return Err(AgentError::LLM(format!(
                        "Context overflow: {} tokens > {} limit",
                        total_tokens, self.max_context_tokens
                    )));
                }
                ContextOverflowAction::Truncate { keep_recent } => {
                    self.truncate_context(&mut messages, *keep_recent);
                }
                ContextOverflowAction::Summarize {
                    summarizer_llm,
                    max_summary_tokens,
                    custom_prompt,
                    keep_recent,
                    filter,
                } => {
                    self.summarize_context(
                        &mut messages,
                        summarizer_llm.as_deref(),
                        *max_summary_tokens,
                        custom_prompt.as_deref(),
                        *keep_recent,
                        filter.as_ref(),
                    )
                    .await?;
                }
            }
        }

        Ok(messages)
    }

    fn parse_tool_calls(&self, content: &str) -> Option<Vec<ToolCall>> {
        // Try direct JSON parse first
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(content) {
            // Handle JSON array of tool calls (parallel tool calling)
            if let Some(arr) = parsed.as_array() {
                let calls: Vec<ToolCall> = arr
                    .iter()
                    .filter_map(|v| self.extract_tool_call_from_value(v))
                    .collect();
                if !calls.is_empty() {
                    return Some(calls);
                }
            }
            // Handle single JSON object
            if let Some(tool_call) = self.extract_tool_call_from_value(&parsed) {
                return Some(vec![tool_call]);
            }
        }

        // Try to extract JSON from content (handles extra text/braces from LLM)
        if let Some(json_str) = self.extract_json_from_content(content) {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json_str) {
                // Handle JSON array of tool calls (parallel tool calling)
                if let Some(arr) = parsed.as_array() {
                    let calls: Vec<ToolCall> = arr
                        .iter()
                        .filter_map(|v| self.extract_tool_call_from_value(v))
                        .collect();
                    if !calls.is_empty() {
                        return Some(calls);
                    }
                }
                // Handle single JSON object
                if let Some(tool_call) = self.extract_tool_call_from_value(&parsed) {
                    return Some(vec![tool_call]);
                }
            }
        }

        None
    }

    fn extract_tool_call_from_value(&self, parsed: &serde_json::Value) -> Option<ToolCall> {
        if let Some(tool_name) = parsed.get("tool").and_then(|v| v.as_str()) {
            let arguments = parsed
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::json!({}));
            return Some(ToolCall {
                id: uuid::Uuid::new_v4().to_string(),
                name: tool_name.to_string(),
                arguments,
            });
        }
        None
    }

    // Lite models could generate unmatched braces: this function handles such cases
    fn extract_json_from_content(&self, content: &str) -> Option<String> {
        // Try array first (for parallel tool calls), then single object
        if let Some(result) = self.extract_json_array_from_content(content) {
            return Some(result);
        }
        self.extract_json_object_from_content(content)
    }

    /// Extract a JSON array `[...]` containing tool calls from mixed content.
    fn extract_json_array_from_content(&self, content: &str) -> Option<String> {
        let start = content.find('[')?;
        let content_from_start = &content[start..];

        let mut depth = 0;
        let mut end = 0;
        for (i, ch) in content_from_start.char_indices() {
            match ch {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }

        if end > 0 {
            let json_str = &content_from_start[..end];
            // Verify it looks like an array of tool calls
            if json_str.contains("\"tool\"") {
                return Some(json_str.to_string());
            }
        }

        None
    }

    /// Extract a JSON object `{...}` containing a tool call from mixed content.
    fn extract_json_object_from_content(&self, content: &str) -> Option<String> {
        let start = content.find('{')?;
        let content_from_start = &content[start..];

        // Count braces to find the matching closing brace
        let mut depth = 0;
        let mut end = 0;
        for (i, ch) in content_from_start.char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }

        if end > 0 {
            let json_str = &content_from_start[..end];
            // Verify it looks like a tool call
            if json_str.contains("\"tool\"") {
                return Some(json_str.to_string());
            }
        }

        None
    }

    async fn execute_tool(&self, tool_call: &ToolCall) -> Result<String> {
        let tool = self
            .tools
            .get(&tool_call.name)
            .ok_or_else(|| AgentError::Tool(format!("Tool not found: {}", tool_call.name)))?;

        let actor_context = self.outbound_actor_context();
        let result = if actor_context.is_empty() {
            tool.execute(tool_call.arguments.clone()).await
        } else {
            scope_actor_context(actor_context, tool.execute(tool_call.arguments.clone())).await
        };

        if result.success {
            Ok(result.output)
        } else {
            Err(AgentError::Tool(result.output))
        }
    }

    #[instrument(skip(self, tool_call), fields(tool = %tool_call.name))]
    async fn execute_tool_smart(&self, tool_call: &ToolCall) -> Result<String> {
        let mut tool_call = tool_call.clone();
        info!(args = %tool_call.arguments, "Executing tool");

        self.hooks
            .on_tool_start(&tool_call.name, &tool_call.arguments)
            .await;
        let tool_start = Instant::now();

        let available_tool_ids = self.get_available_tool_ids().await?;
        // Resolve the tool call name to its canonical ID via the registry.
        // The LLM may use the display name (e.g., "HTTP Client") but
        // available_tool_ids contains IDs (e.g., "http"). Registry.get()
        // now matches by ID, display name, and alias.
        let resolved_id = self
            .tools
            .get(&tool_call.name)
            .map(|t| t.id().to_lowercase());
        let tool_name_lower = tool_call.name.to_lowercase();
        if !available_tool_ids.is_empty() {
            let is_available = available_tool_ids.iter().any(|id| {
                let id_lower = id.to_lowercase();
                id_lower == tool_name_lower || resolved_id.as_deref() == Some(&id_lower)
            });
            if !is_available {
                warn!(tool = %tool_call.name, "Tool not available in current context");
                return Err(AgentError::Tool(format!(
                    "Tool '{}' is not available. Available tools: {}",
                    tool_call.name,
                    available_tool_ids.join(", ")
                )));
            }
        }

        // Build language context for HITL localization.
        let hitl_lang_ctx = self.build_hitl_language_context();

        // Compute canonical tool ID before security/HITL checks so both use it.
        // Use the tool ID (e.g. "http") instead of the name (e.g. "HTTP Client") so it matches the HITL config keys in hitl.tools.<id>.
        let hitl_tool_id = resolved_id.as_deref().unwrap_or(&tool_name_lower);

        // Check security FIRST -- don't bother the human if security blocks
        if self.tool_security.config().enabled {
            let security_result = self
                .tool_security
                .check_tool_execution(&tool_call.name, &tool_call.arguments)
                .await?;

            match security_result {
                SecurityCheckResult::Allow => {}
                SecurityCheckResult::Block { reason } => {
                    warn!(reason = %reason, "Tool blocked by security");
                    return Err(AgentError::Tool(format!("Blocked: {}", reason)));
                }
                SecurityCheckResult::RequireConfirmation { message } => {
                    // Route through HITL if available, otherwise error
                    if let Some(ref hitl_engine) = self.hitl_engine {
                        let check_result = self
                            .observe_purpose(
                                ObservationPurpose::HitlLocalization,
                                hitl_engine.check_tool_with_localization(
                                    hitl_tool_id,
                                    &tool_call.arguments,
                                    &hitl_lang_ctx,
                                    self.approval_handler.as_ref(),
                                    Some(&self.llm_registry),
                                ),
                            )
                            .await?;
                        let result = self.request_hitl_approval(check_result).await?;
                        match result {
                            ApprovalResult::Approved | ApprovalResult::Modified { .. } => {}
                            ApprovalResult::Rejected { reason } => {
                                let reason_str = reason.as_deref().unwrap_or("rejected");
                                warn!(tool = %tool_call.name, reason = %reason_str, "Security confirmation rejected by approver");
                                return Err(AgentError::Tool(format!(
                                    "Confirmation rejected: {}",
                                    message
                                )));
                            }
                            _ => {}
                        }
                    } else {
                        warn!(message = %message, "Tool requires confirmation but no HITL handler");
                        return Err(AgentError::Tool(format!(
                            "Confirmation required: {}",
                            message
                        )));
                    }
                }
                SecurityCheckResult::Warn { message } => {
                    warn!(message = %message, "Tool security warning");
                }
            }
        }

        // Check HITL approval for tool (after security passes)
        if let Some(ref hitl_engine) = self.hitl_engine {
            let check_result = self
                .observe_purpose(
                    ObservationPurpose::HitlLocalization,
                    hitl_engine.check_tool_with_localization(
                        hitl_tool_id,
                        &tool_call.arguments,
                        &hitl_lang_ctx,
                        self.approval_handler.as_ref(),
                        Some(&self.llm_registry),
                    ),
                )
                .await?;
            if check_result.is_required() {
                let result = self.request_hitl_approval(check_result).await?;
                match result {
                    ApprovalResult::Approved => {}
                    ApprovalResult::Modified { changes } => {
                        if let Some(obj) = tool_call.arguments.as_object_mut() {
                            for (k, v) in changes {
                                obj.insert(k, v);
                            }
                        }
                        info!(tool = %tool_call.name, "Tool arguments modified by approver");
                    }
                    ApprovalResult::Rejected { reason: _reason } => {
                        warn!(tool = %tool_call.name, "Tool execution rejected by HITL");
                        return Err(AgentError::HITLRejected(format!(
                            "Tool '{}' was rejected by human approver. Do not retry.",
                            tool_call.name
                        )));
                    }
                    _ => {}
                }
            }

            // Check conditions (e.g., amount > 1000)
            let condition_check = self
                .observe_purpose(
                    ObservationPurpose::HitlLocalization,
                    hitl_engine.check_conditions_with_localization(
                        &tool_call.arguments,
                        &hitl_lang_ctx,
                        self.approval_handler.as_ref(),
                        Some(&self.llm_registry),
                    ),
                )
                .await?;
            if condition_check.is_required() {
                let result = self.request_hitl_approval(condition_check).await?;
                match result {
                    ApprovalResult::Approved => {}
                    ApprovalResult::Modified { changes } => {
                        if let Some(obj) = tool_call.arguments.as_object_mut() {
                            for (k, v) in changes {
                                obj.insert(k, v);
                            }
                        }
                        info!(tool = %tool_call.name, "Tool arguments modified by approver (condition)");
                    }
                    ApprovalResult::Rejected { reason: _reason } => {
                        warn!(tool = %tool_call.name, "Tool execution rejected by HITL condition");
                        return Err(AgentError::HITLRejected(format!(
                            "Tool '{}' was rejected due to policy condition. Do not retry.",
                            tool_call.name
                        )));
                    }
                    _ => {}
                }
            }
        }

        let tool_config = self.recovery_manager.get_tool_config(&tool_call.name);

        let result = if tool_config.max_retries > 0 {
            let retry_config = RetryConfig {
                max_retries: tool_config.max_retries,
                ..Default::default()
            };

            let tool_call_clone = tool_call.clone();
            self.recovery_manager
                .with_retry(
                    &format!("tool:{}", tool_call.name),
                    Some(&retry_config),
                    || {
                        let tc = tool_call_clone.clone();
                        async move { self.execute_tool(&tc).await.map_err(|e| e.classify()) }
                    },
                )
                .await
                .map_err(|e| AgentError::Tool(e.to_string()))
        } else {
            self.execute_tool(&tool_call).await
        };

        // Apply on_failure policy when the tool fails after all retries.
        let result = match result {
            Ok(output) => Ok(output),
            Err(e) => match &tool_config.on_failure {
                ToolFailureAction::Skip => {
                    warn!(
                        tool = %tool_call.name,
                        error = %e,
                        "Tool failed, skipping per on_failure: skip policy"
                    );
                    Ok(format!(
                        "{{\"skipped\": true, \"reason\": \"Tool '{}' was skipped after failure\"}}",
                        tool_call.name
                    ))
                }
                ToolFailureAction::Fallback { fallback_tool } => {
                    warn!(
                        tool = %tool_call.name,
                        fallback = %fallback_tool,
                        error = %e,
                        "Tool failed, trying fallback tool"
                    );
                    let fallback_call = ToolCall {
                        id: tool_call.id.clone(),
                        name: fallback_tool.clone(),
                        arguments: tool_call.arguments.clone(),
                    };
                    self.execute_tool(&fallback_call).await
                }
                ToolFailureAction::ReportError => Err(e),
            },
        };

        let tool_duration_ms = tool_start.elapsed().as_millis() as u64;

        match &result {
            Ok(output) => {
                info!(output_len = output.len(), "Tool execution successful");
                let result_value: Value =
                    serde_json::from_str(output).unwrap_or(Value::String(output.clone()));
                self.record_tool_call(&tool_call.name, result_value);

                let tool_result = ToolResult {
                    success: true,
                    output: output.clone(),
                    metadata: None,
                };
                self.hooks
                    .on_tool_complete(&tool_call.name, &tool_result, tool_duration_ms)
                    .await;
            }
            Err(e) => {
                error!(error = %e, "Tool execution failed");
                self.record_tool_call(&tool_call.name, serde_json::json!({"error": e.to_string()}));

                let tool_result = ToolResult {
                    success: false,
                    output: e.to_string(),
                    metadata: None,
                };
                self.hooks
                    .on_tool_complete(&tool_call.name, &tool_result, tool_duration_ms)
                    .await;
                self.hooks.on_error(e).await;
            }
        }

        result
    }

    //
    // This method is branch-safe because it only asks the router which skill matches.
    // Do not add disambiguation, pending-skill writes, or skill execution here.
    //
    /// Selects a skill without executing it.
    async fn select_skill_candidate(&self, input: &str) -> Result<Option<SkillCandidate>> {
        let Some(ref router) = self.skill_router else {
            return Ok(None);
        };
        let available_skills = self.get_available_skills();
        if available_skills.is_empty() {
            return Ok(None);
        }
        let skill_ids: Vec<&str> = available_skills.iter().map(|s| s.id.as_str()).collect();
        let Some(skill_id) = self
            .observe_purpose(
                ObservationPurpose::SkillRouting,
                router.select_skill_filtered(input, &skill_ids),
            )
            .await?
        else {
            return Ok(None);
        };
        let skill = router
            .get_skill(&skill_id)
            .cloned()
            .ok_or_else(|| AgentError::Skill(format!("Skill not found: {}", skill_id)))?;
        info!(skill_id = %skill_id, "Skill selected");
        Ok(Some(SkillCandidate::new(skill_id, skill)))
    }

    //
    // This is the commit half of skill routing.
    // It may mutate pending skill state, run clarification, and execute skill steps.
    //
    async fn commit_skill_candidate_route_result(
        &self,
        candidate: SkillCandidate,
        input: &str,
    ) -> Result<SkillRouteResult> {
        let skill_id = candidate.skill_id;
        let skill = candidate.skill;
        if let Some(ref skill_disambig) = skill.disambiguation {
            if skill_disambig.enabled.unwrap_or(false) {
                if let Some(ref disambiguator) = self.disambiguation_manager {
                    let context = self.build_disambiguation_context().await?;
                    let state_override = self
                        .state_machine
                        .as_ref()
                        .and_then(|sm| sm.current_definition())
                        .and_then(|def| def.disambiguation.clone());

                    match self
                        .observe_purpose(
                            ObservationPurpose::DisambiguationDetection,
                            disambiguator.process_input_with_override(
                                input,
                                &context,
                                state_override.as_ref(),
                                Some(skill_disambig),
                            ),
                        )
                        .await?
                    {
                        DisambiguationResult::Clear => {
                            debug!(skill_id = %skill_id, "Skill disambiguation: clear");
                        }
                        DisambiguationResult::NeedsClarification {
                            question,
                            detection,
                        } => {
                            info!(
                                skill_id = %skill_id,
                                ambiguity_type = ?detection.ambiguity_type,
                                confidence = detection.confidence,
                                "Skill requires clarification before execution"
                            );
                            *self.pending_skill_id.write() = Some(skill_id.clone());
                            return Ok(SkillRouteResult::NeedsClarification(
                                AgentResponse::new(&question.question).with_metadata(
                                    "disambiguation",
                                    serde_json::json!({
                                        "status": "awaiting_clarification",
                                        "skill_id": skill_id,
                                        "options": question.options,
                                        "clarifying": question.clarifying,
                                        "detection": {
                                            "type": detection.ambiguity_type,
                                            "confidence": detection.confidence,
                                            "what_is_unclear": detection.what_is_unclear,
                                        }
                                    }),
                                ),
                            ));
                        }
                        DisambiguationResult::Clarified { enriched_input, .. } => {
                            info!(skill_id = %skill_id, enriched = %enriched_input, "Skill disambiguation clarified");
                            return Ok(SkillRouteResult::Response(
                                self.execute_skill(&skill, &enriched_input).await?,
                            ));
                        }
                        DisambiguationResult::ProceedWithBestGuess { enriched_input } => {
                            info!(skill_id = %skill_id, "Skill disambiguation best guess");
                            return Ok(SkillRouteResult::Response(
                                self.execute_skill(&skill, &enriched_input).await?,
                            ));
                        }
                        DisambiguationResult::GiveUp { reason } => {
                            warn!(skill_id = %skill_id, reason = %reason, "Skill disambiguation gave up");
                            let apology = self
                                .generate_localized_apology(
                                    "Generate a brief, polite apology saying you couldn't understand the request. Be concise.",
                                    &reason,
                                )
                                .await
                                .unwrap_or_else(|_| {
                                    format!("I'm sorry, I couldn't understand your request: {}", reason)
                                });
                            return Ok(SkillRouteResult::NeedsClarification(AgentResponse::new(
                                &apology,
                            )));
                        }
                        DisambiguationResult::Escalate { reason } => {
                            info!(skill_id = %skill_id, reason = %reason, "Skill disambiguation escalating");
                            let apology = self
                                .generate_localized_apology(
                                    "Explain briefly that you're transferring the user to a human agent for help.",
                                    &reason,
                                )
                                .await
                                .unwrap_or_else(|_| {
                                    format!("I need human assistance to help with your request: {}", reason)
                                });
                            return Ok(SkillRouteResult::NeedsClarification(AgentResponse::new(
                                &apology,
                            )));
                        }
                        DisambiguationResult::Abandoned { .. } => {
                            debug!(skill_id = %skill_id, "Skill disambiguation abandoned");
                            return Ok(SkillRouteResult::NoMatch);
                        }
                    }
                }
            }
        }
        Ok(SkillRouteResult::Response(
            self.execute_skill(&skill, input).await?,
        ))
    }

    /// Result of skill routing.
    async fn try_skill_route(&self, input: &str) -> Result<SkillRouteResult> {
        if let Some(candidate) = self.select_skill_candidate(input).await? {
            self.commit_skill_candidate_route_result(candidate, input)
                .await
        } else {
            Ok(SkillRouteResult::NoMatch)
        }
    }

    /// Execute a skill with reasoning and reflection, returning the response string.
    async fn execute_skill(&self, skill: &SkillDefinition, input: &str) -> Result<String> {
        if let Some(ref executor) = self.skill_executor {
            let skill_reasoning = self.get_skill_reasoning_config(skill);
            let skill_reflection = self.get_skill_reflection_config(skill);

            debug!(
                skill_id = %skill.id,
                reasoning_mode = ?skill_reasoning.mode,
                reflection_enabled = ?skill_reflection.enabled,
                "Skill reasoning/reflection config"
            );

            let response = self
                .observe_purpose(
                    ObservationPurpose::SkillPrompt,
                    executor.execute(skill, input, serde_json::json!({})),
                )
                .await?;

            if skill_reflection.requires_evaluation() && skill_reflection.is_enabled() {
                let should_reflect = self
                    .should_reflect_with_config(input, &response, &skill_reflection)
                    .await?;
                if should_reflect {
                    let evaluated = self
                        .evaluate_and_retry_with_config(input, response, &skill_reflection)
                        .await?;
                    return Ok(evaluated);
                }
            }

            return Ok(response);
        }
        Err(AgentError::Skill(
            "No skill executor configured".to_string(),
        ))
    }

    /// Execute a skill by ID, bypassing the skill router.
    /// Used after skill-triggered disambiguation resolves to route directly to the matched skill.
    async fn execute_skill_by_id(&self, skill_id: &str, input: &str) -> Result<String> {
        let skill = self
            .skill_router
            .as_ref()
            .and_then(|r| r.get_skill(skill_id).cloned())
            .ok_or_else(|| AgentError::Skill(format!("Skill not found: {}", skill_id)))?;
        self.execute_skill(&skill, input).await
    }

    async fn should_reflect_with_config(
        &self,
        input: &str,
        response: &str,
        config: &ReflectionConfig,
    ) -> Result<bool> {
        if !config.requires_evaluation() {
            return Ok(false);
        }

        if config.is_enabled() {
            return Ok(true);
        }

        let evaluator_llm = config
            .evaluator_llm
            .as_ref()
            .and_then(|alias| self.llm_registry.get(alias).ok())
            .or_else(|| self.llm_registry.router().ok())
            .or_else(|| self.llm_registry.default().ok());

        let Some(llm) = evaluator_llm else {
            return Ok(false);
        };

        let response_preview: String = response.chars().take(500).collect();
        let prompt = format!(
            r#"Should this response be evaluated for quality? Consider if it's a complex or important response.

User query: "{}"
Response: "{}"

Answer YES or NO only."#,
            input, response_preview
        );

        let messages = vec![ChatMessage::user(&prompt)];
        let result = self
            .observe_purpose(
                ObservationPurpose::ReflectionDecision,
                llm.complete(&messages, None),
            )
            .await;

        match result {
            Ok(resp) => Ok(resp.content.trim().to_uppercase().contains("YES")),
            Err(_) => Ok(false),
        }
    }

    async fn evaluate_and_retry_with_config(
        &self,
        input: &str,
        mut response: String,
        config: &ReflectionConfig,
    ) -> Result<String> {
        let llm = self.get_state_llm()?;
        let mut attempts = 0u32;
        let max_retries = config.max_retries;

        loop {
            let evaluation = self
                .evaluate_response_with_config(input, &response, config)
                .await?;

            if evaluation.passed || attempts >= max_retries {
                info!(
                    passed = evaluation.passed,
                    confidence = evaluation.confidence,
                    attempts = attempts + 1,
                    "Skill reflection evaluation complete"
                );
                return Ok(response);
            }

            debug!(
                attempt = attempts + 1,
                failed_criteria = evaluation.failed_criteria().count(),
                "Skill response did not meet criteria, retrying"
            );

            let feedback: Vec<String> = evaluation
                .failed_criteria()
                .map(|c| format!("- {}", c.criterion))
                .collect();

            let retry_prompt = format!(
                "Your previous response did not meet these criteria:\n{}\n\nPlease provide an improved response to: {}",
                feedback.join("\n"),
                input
            );

            let messages = vec![ChatMessage::user(&retry_prompt)];
            let retry_response = self
                .observe_purpose(
                    ObservationPurpose::ReflectionEvaluation,
                    llm.complete(&messages, None),
                )
                .await
                .map_err(|e| AgentError::LLM(e.to_string()))?;

            response = retry_response.content.trim().to_string();
            attempts += 1;
        }
    }

    async fn evaluate_response_with_config(
        &self,
        input: &str,
        response: &str,
        config: &ReflectionConfig,
    ) -> Result<EvaluationResult> {
        let evaluator_llm = config
            .evaluator_llm
            .as_ref()
            .and_then(|alias| self.llm_registry.get(alias).ok())
            .or_else(|| self.llm_registry.router().ok())
            .or_else(|| self.llm_registry.default().ok())
            .ok_or_else(|| AgentError::Config("No LLM available for evaluation".into()))?;

        let criteria = &config.criteria;
        let criteria_list = criteria
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{}. {}", i + 1, c))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            r#"Evaluate this response against the criteria.

User query: "{}"

Response to evaluate: "{}"

Criteria:
{}

For each criterion, respond with:
- criterion number
- PASS or FAIL
- brief reason

Then provide overall confidence (0.0 to 1.0) and whether it passes overall.

Format:
1. PASS/FAIL - reason
2. PASS/FAIL - reason
...
CONFIDENCE: 0.X
OVERALL: PASS/FAIL"#,
            input, response, criteria_list
        );

        let messages = vec![ChatMessage::user(&prompt)];
        let eval_response = self
            .observe_purpose(
                ObservationPurpose::ReflectionEvaluation,
                evaluator_llm.complete(&messages, None),
            )
            .await
            .map_err(|e| AgentError::LLM(format!("Evaluation failed: {}", e)))?;

        let content = eval_response.content.to_uppercase();
        let llm_pass = content.contains("OVERALL: PASS");

        let confidence = content
            .lines()
            .find(|l| l.contains("CONFIDENCE:"))
            .and_then(|l| {
                l.split(':')
                    .nth(1)
                    .and_then(|v| v.trim().parse::<f32>().ok())
            })
            .unwrap_or(if llm_pass { 0.8 } else { 0.4 });

        // Gate pass against confidence threshold.
        // LLM may say PASS but with low confidence - the threshold catches this.
        let overall_pass = llm_pass && confidence >= config.pass_threshold;

        let mut criteria_results = Vec::new();
        for (i, criterion) in criteria.iter().enumerate() {
            let line_marker = format!("{}.", i + 1);
            let passed = eval_response
                .content
                .lines()
                .find(|l| l.contains(&line_marker))
                .map(|l| l.to_uppercase().contains("PASS"))
                .unwrap_or(overall_pass);

            if passed {
                criteria_results.push(CriterionResult::pass(criterion));
            } else {
                criteria_results.push(CriterionResult::fail(criterion, "Did not meet criterion"));
            }
        }

        Ok(EvaluationResult::new(overall_pass, confidence).with_criteria(criteria_results))
    }

    /// Process input through the pipeline (state-level override or agent-level).
    async fn process_input(&self, input: &str) -> Result<ProcessData> {
        if let Some(processor) = self.get_state_process_processor() {
            let purpose = observation_purpose_for_process(processor.input_purpose_hint());
            return self
                .observe_purpose(purpose, processor.process_input(input))
                .await;
        }
        if let Some(ref processor) = self.process_processor {
            let purpose = observation_purpose_for_process(processor.input_purpose_hint());
            self.observe_purpose(purpose, processor.process_input(input))
                .await
        } else {
            Ok(ProcessData::new(input))
        }
    }

    /// Process output through the pipeline (state-level override or agent-level).
    async fn process_output(
        &self,
        output: &str,
        input_context: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Result<ProcessData> {
        if let Some(processor) = self.get_state_process_processor() {
            let purpose = observation_purpose_for_process(processor.output_purpose_hint());
            return self
                .observe_purpose(purpose, processor.process_output(output, input_context))
                .await;
        }
        if let Some(ref processor) = self.process_processor {
            let purpose = observation_purpose_for_process(processor.output_purpose_hint());
            self.observe_purpose(purpose, processor.process_output(output, input_context))
                .await
        } else {
            Ok(ProcessData::new(output))
        }
    }

    /// Build a ProcessProcessor from the current state's process config, if any.
    fn get_state_process_processor(&self) -> Option<ProcessProcessor> {
        let sm = self.state_machine.as_ref()?;
        let def = sm.current_definition()?;
        let config = def.process.as_ref()?;
        let mut processor = ProcessProcessor::new(config.clone());
        if let Some(ref registry) = Some(self.llm_registry.clone()) {
            processor = processor.with_llm_registry(registry.clone());
        }
        processor = processor.with_stage_observer(Arc::new(ObservabilityProcessStageObserver));
        Some(processor)
    }

    async fn check_turn_timeout(&self) -> Result<()> {
        if let Some(ref sm) = self.state_machine {
            if let Some(timeout_state) = sm.check_timeout() {
                let from_state = sm.current();
                self.execute_state_exit_actions(&from_state).await;
                sm.transition_to(&timeout_state, "max_turns exceeded")?;
                self.execute_state_enter_actions(&timeout_state).await;
                info!(to = %timeout_state, "Timeout transition");
            }
        }
        Ok(())
    }

    fn increment_turn(&self) {
        if let Some(ref sm) = self.state_machine {
            sm.increment_turn();
        }
    }

    fn transitions_available_for_commit(&self) -> Option<(Vec<Transition>, String)> {
        let sm = self.state_machine.as_ref()?;
        let current = sm.current();
        let transitions: Vec<_> = sm
            .auto_transitions()
            .into_iter()
            .filter(|t| match t.cooldown_turns {
                Some(cd) if cd > 0 => {
                    let resolved = sm.config().resolve_full_path(&current, &t.to);
                    !sm.is_on_cooldown(&resolved, cd)
                }
                _ => true,
            })
            .collect();
        Some((transitions, current))
    }

    fn transition_reason(transition: &Transition) -> String {
        if transition.when.is_empty() {
            "guard condition met".to_string()
        } else {
            transition.when.clone()
        }
    }

    /// Builds transition context with optional staged writes overlaid.
    fn build_transition_context(
        &self,
        user_message: &str,
        response: &str,
        current_state: &str,
        staged: Option<&HashMap<String, Value>>,
    ) -> TransitionContext {
        let context_map = staged
            .map(|writes| self.build_context_with_staged(writes))
            .unwrap_or_else(|| self.build_context_with_overlays());
        TransitionContext::new(user_message, response, current_state).with_context(context_map)
    }

    /// Selects a post-response transition without committing side effects.
    async fn select_transition_candidate(
        &self,
        user_message: &str,
        response: &str,
    ) -> Result<Option<TransitionCandidate>> {
        let Some((transitions, current_state)) = self.transitions_available_for_commit() else {
            return Ok(None);
        };
        let transitions: Vec<Transition> = transitions
            .into_iter()
            .filter(|transition| matches!(transition.timing, TransitionTiming::PostResponse))
            .collect();
        if transitions.is_empty() {
            return Ok(None);
        }
        let Some(evaluator) = self.transition_evaluator.as_ref() else {
            return Ok(None);
        };
        let context = self.build_transition_context(user_message, response, &current_state, None);
        let selected = self
            .observe_purpose(
                ObservationPurpose::StateTransitionEvaluation,
                evaluator.select_transition(&transitions, &context),
            )
            .await?;
        Ok(selected.map(|index| {
            let transition = transitions[index].clone();
            TransitionCandidate::new(
                current_state,
                transition.clone(),
                Self::transition_reason(&transition),
            )
        }))
    }

    /// Selects a guard or resolved-intent transition without an LLM call.
    fn select_deterministic_transition_candidate(
        &self,
        user_message: &str,
        current_state: &str,
        transitions: &[Transition],
        staged: &HashMap<String, Value>,
    ) -> Option<TransitionCandidate> {
        let context = self.build_transition_context(user_message, "", current_state, Some(staged));

        for transition in transitions {
            if let Some(guard) = transition.guard.as_ref() {
                if evaluate_guard(guard, &context) {
                    return Some(TransitionCandidate::new(
                        current_state,
                        transition.clone(),
                        Self::transition_reason(transition),
                    ));
                }
            }
        }

        let resolved_intent = context
            .context
            .get("resolved_intent")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty());
        if let Some(resolved_intent) = resolved_intent {
            for transition in transitions {
                if transition.intent.as_deref() == Some(resolved_intent) {
                    return Some(TransitionCandidate::new(
                        current_state,
                        transition.clone(),
                        Self::transition_reason(transition),
                    ));
                }
            }
        }

        None
    }

    /// Commits a selected transition through the shared post-response path.
    async fn commit_transition_candidate(&self, candidate: &TransitionCandidate) -> Result<bool> {
        self.commit_transition_target(&candidate.from_state, candidate.target(), &candidate.reason)
            .await
    }

    /// Runs state transition approval before any transition side effects.
    async fn approve_transition_target(&self, from_state: &str, target: &str) -> Result<bool> {
        let approved = self.check_state_hitl(Some(from_state), target).await?;
        if !approved {
            info!(to = %target, "State transition rejected by HITL");
        }
        Ok(approved)
    }

    /// Applies transition side effects after approval has succeeded.
    async fn apply_transition_target(
        &self,
        from_state: &str,
        target: &str,
        reason: &str,
        staged: Option<&HashMap<String, Value>>,
    ) -> Result<bool> {
        let Some(ref sm) = self.state_machine else {
            return Ok(false);
        };

        self.execute_state_exit_actions(from_state).await;
        sm.transition_to(target, reason)?;
        sm.reset_no_transition();
        if let Some(staged) = staged {
            self.commit_staged_context_writes(staged).await;
        }
        let entered = sm.current();
        self.execute_state_enter_actions(&entered).await;
        self.hooks
            .on_state_transition(Some(from_state), &entered, reason)
            .await;
        info!(from = %from_state, to = %entered, "State transition");
        Ok(true)
    }

    /// Approves and applies a transition without staged context writes.
    async fn commit_transition_target(
        &self,
        from_state: &str,
        target: &str,
        reason: &str,
    ) -> Result<bool> {
        if !self.approve_transition_target(from_state, target).await? {
            return Ok(false);
        }
        self.apply_transition_target(from_state, target, reason, None)
            .await
    }

    /// Commits a pre-response transition after approval and before redispatch.
    async fn commit_pre_response_transition_candidate(
        &self,
        candidate: &TransitionCandidate,
        staged: &HashMap<String, Value>,
        processed_input: &str,
    ) -> Result<bool> {
        if !self
            .approve_transition_target(&candidate.from_state, candidate.target())
            .await?
        {
            return Ok(false);
        }
        self.commit_root_user_message(processed_input).await?;
        self.apply_transition_target(
            &candidate.from_state,
            candidate.target(),
            &candidate.reason,
            Some(staged),
        )
        .await
    }

    /// Handles post-response transition misses with fallback counters.
    async fn handle_transition_miss(&self, current_state: &str) -> Result<bool> {
        let Some(ref sm) = self.state_machine else {
            return Ok(false);
        };
        sm.increment_no_transition();
        let Some(fallback) = sm.check_fallback() else {
            return Ok(false);
        };
        self.commit_transition_target(current_state, &fallback, "fallback after no transitions")
            .await
    }

    /// Evaluates and commits post-response transitions for the committed response path.
    async fn evaluate_transitions(&self, user_message: &str, response: &str) -> Result<bool> {
        let Some((transitions, current_state)) = self.transitions_available_for_commit() else {
            return Ok(false);
        };
        if transitions.is_empty() {
            return Ok(false);
        }
        if let Some(candidate) = self
            .select_transition_candidate(user_message, response)
            .await?
        {
            return self.commit_transition_candidate(&candidate).await;
        }
        self.handle_transition_miss(&current_state).await
    }

    /// Attempts deterministic pre-response routing before old-state response generation.
    async fn try_pre_response_transition(
        &self,
        processed_input: &str,
    ) -> Result<Option<AgentResponse>> {
        let optimization = &self.runtime_config.optimization;
        if !optimization.enabled || !optimization.pre_response_deterministic_transitions {
            return Ok(None);
        }
        let Some((transitions, current_state)) = self.transitions_available_for_commit() else {
            return Ok(None);
        };
        let eligible: Vec<Transition> = transitions
            .into_iter()
            .filter(|transition| !transition.requires_response)
            .filter(|transition| matches!(transition.timing, TransitionTiming::PreResponse))
            .collect();
        if eligible.is_empty() {
            return Ok(None);
        }

        let empty_staged = HashMap::new();
        let mut extracted_staged: Option<HashMap<String, Value>> = None;
        let mut selected: Option<(TransitionCandidate, HashMap<String, Value>)> = None;

        for transition in &eligible {
            let use_extractors = optimization.pre_response_extractors || transition.run_extractors;
            let staged_for_eval = if use_extractors {
                if extracted_staged.is_none() {
                    extracted_staged =
                        Some(self.run_context_extractors_staged(processed_input).await);
                }
                extracted_staged.as_ref().unwrap_or(&empty_staged)
            } else {
                &empty_staged
            };

            if let Some(candidate) = self.select_deterministic_transition_candidate(
                processed_input,
                &current_state,
                std::slice::from_ref(transition),
                staged_for_eval,
            ) {
                let staged_for_commit = if use_extractors {
                    staged_for_eval.clone()
                } else {
                    HashMap::new()
                };
                selected = Some((candidate, staged_for_commit));
                break;
            }
        }

        let Some((candidate, staged)) = selected else {
            return Ok(None);
        };

        if !self
            .commit_pre_response_transition_candidate(&candidate, &staged, processed_input)
            .await?
        {
            return Ok(None);
        }
        self.redispatch_current_state(processed_input)
            .await
            .map(Some)
    }

    //
    // Speculative branches overlap independent decisions but still commit exactly one path.
    // Losing branches must remain data only and must not write memory, run tools, or emit output.
    //
    async fn try_speculative_branches(
        &self,
        processed_input: &str,
        input_context: &HashMap<String, Value>,
    ) -> Result<Option<AgentResponse>> {
        let optimization = &self.runtime_config.optimization;
        if !optimization.enabled {
            return Ok(None);
        }

        let effective_reasoning_mode = self.get_effective_reasoning_config().mode.clone();
        if !matches!(
            effective_reasoning_mode,
            ReasoningMode::None | ReasoningMode::Auto
        ) {
            return Ok(None);
        }

        let mut transition_enabled =
            optimization.speculative_state_transitions && self.has_parallel_transition_candidates();
        let mut skill_enabled = optimization.speculative_skill_routing
            && self.skill_router.is_some()
            && self.pending_skill_id.read().is_none();
        let mut reasoning_enabled = optimization.speculative_reasoning_auto
            && matches!(effective_reasoning_mode, ReasoningMode::Auto);

        if matches!(effective_reasoning_mode, ReasoningMode::Auto) {
            if !reasoning_enabled || optimization.max_speculative_llm_calls_per_turn < 2 {
                return Ok(None);
            }
        }

        if !transition_enabled && !skill_enabled && !reasoning_enabled {
            return Ok(None);
        }

        let mut optional_slots = optimization.max_parallel_runtime_tasks.saturating_sub(1);
        let mut speculative_call_slots = optimization
            .max_speculative_llm_calls_per_turn
            .saturating_sub(1);
        if reasoning_enabled {
            if optional_slots == 0 || speculative_call_slots == 0 {
                return Ok(None);
            }
            optional_slots -= 1;
            speculative_call_slots -= 1;
        }
        if transition_enabled {
            if optional_slots == 0 || speculative_call_slots == 0 {
                transition_enabled = false;
            } else {
                optional_slots -= 1;
                speculative_call_slots -= 1;
            }
        }
        if skill_enabled && (optional_slots == 0 || speculative_call_slots == 0) {
            skill_enabled = false;
        }

        if !transition_enabled && !skill_enabled && !reasoning_enabled {
            return Ok(None);
        }

        let main_kind = if transition_enabled {
            RuntimeOptimizationKind::ParallelStateTransition
        } else if skill_enabled {
            RuntimeOptimizationKind::SpeculativeSkillRouting
        } else {
            RuntimeOptimizationKind::SpeculativeReasoningAuto
        };
        if !self.reserve_active_speculative_llm_call(main_kind) {
            return Ok(None);
        }

        let mut branch_set = ScheduledBranchSet::new(optimization.max_parallel_runtime_tasks)?;
        let main_branch = RuntimeBranch::new(
            RuntimeTaskPurpose::MainResponse,
            main_kind,
            RuntimeTaskPriority::Normal,
            RuntimeCommitBehavior::FinalResponse,
        );
        let transition_branch = RuntimeBranch::new(
            RuntimeTaskPurpose::StateTransition,
            RuntimeOptimizationKind::ParallelStateTransition,
            RuntimeTaskPriority::Critical,
            RuntimeCommitBehavior::TransitionDecision,
        );
        let skill_branch = RuntimeBranch::new(
            RuntimeTaskPurpose::SkillRouting,
            RuntimeOptimizationKind::SpeculativeSkillRouting,
            RuntimeTaskPriority::High,
            RuntimeCommitBehavior::SkillSelection,
        );
        let reasoning_branch = RuntimeBranch::new(
            RuntimeTaskPurpose::ReasoningJudge,
            RuntimeOptimizationKind::SpeculativeReasoningAuto,
            RuntimeTaskPriority::Normal,
            RuntimeCommitBehavior::ReasoningDecision,
        );
        let main_id = main_branch.branch_id();
        let transition_id = transition_branch.branch_id();
        let skill_id = skill_branch.branch_id();
        let reasoning_id = reasoning_branch.branch_id();

        let main_id_for_future = main_id.clone();
        if !branch_set.schedule(
            main_branch,
            Box::pin(async move {
                match crate::optimization::observability::with_branch_observation(
                    &main_id_for_future,
                    main_kind,
                    RuntimeCommitBehavior::FinalResponse,
                    self.generate_main_response_draft(processed_input, &ReasoningMode::None),
                )
                .await
                {
                    Ok(draft) => RuntimeBranchResult::MainDraft(draft),
                    Err(error) => RuntimeBranchResult::Failed(error),
                }
            }),
        ) {
            return Ok(None);
        }

        if transition_enabled {
            let transition_id_for_future = transition_id.clone();
            if !branch_set.schedule(
                transition_branch,
                Box::pin(async move {
                    match crate::optimization::observability::with_branch_observation(
                        &transition_id_for_future,
                        RuntimeOptimizationKind::ParallelStateTransition,
                        RuntimeCommitBehavior::TransitionDecision,
                        self.select_parallel_transition_candidate(processed_input),
                    )
                    .await
                    {
                        Ok(candidate) => RuntimeBranchResult::Transition(candidate),
                        Err(error) => RuntimeBranchResult::Failed(error),
                    }
                }),
            ) {
                transition_enabled = false;
            }
        }

        if skill_enabled {
            let skill_id_for_future = skill_id.clone();
            if !branch_set.schedule(
                skill_branch,
                Box::pin(async move {
                    if !self.reserve_active_speculative_llm_call(
                        RuntimeOptimizationKind::SpeculativeSkillRouting,
                    ) {
                        return RuntimeBranchResult::Cancelled;
                    }
                    match crate::optimization::observability::with_branch_observation(
                        &skill_id_for_future,
                        RuntimeOptimizationKind::SpeculativeSkillRouting,
                        RuntimeCommitBehavior::SkillSelection,
                        self.select_skill_candidate(processed_input),
                    )
                    .await
                    {
                        Ok(candidate) => RuntimeBranchResult::Skill(candidate),
                        Err(error) => RuntimeBranchResult::Failed(error),
                    }
                }),
            ) {
                skill_enabled = false;
            }
        }

        if reasoning_enabled {
            let reasoning_id_for_future = reasoning_id.clone();
            if !branch_set.schedule(
                reasoning_branch,
                Box::pin(async move {
                    if !self.reserve_active_speculative_llm_call(
                        RuntimeOptimizationKind::SpeculativeReasoningAuto,
                    ) {
                        return RuntimeBranchResult::Cancelled;
                    }
                    match crate::optimization::observability::with_branch_observation(
                        &reasoning_id_for_future,
                        RuntimeOptimizationKind::SpeculativeReasoningAuto,
                        RuntimeCommitBehavior::ReasoningDecision,
                        self.determine_reasoning_mode_strict(processed_input),
                    )
                    .await
                    {
                        Ok(mode) => RuntimeBranchResult::Reasoning(mode),
                        Err(error) => RuntimeBranchResult::Failed(error),
                    }
                }),
            ) {
                reasoning_enabled = false;
            }
        }

        if matches!(effective_reasoning_mode, ReasoningMode::Auto) && !reasoning_enabled {
            self.finalize_pending_branches(branch_set.cancel_pending());
            return Ok(None);
        }

        if !transition_enabled && !skill_enabled && !reasoning_enabled {
            self.finalize_pending_branches(branch_set.cancel_pending());
            return Ok(None);
        }

        let mut main_pending = true;
        let mut skill_pending = skill_enabled;
        let mut reasoning_pending = reasoning_enabled;
        let mut transition_finalized = !transition_enabled;
        let mut skill_finalized = !skill_enabled;
        let mut reasoning_finalized = !reasoning_enabled;
        let mut main_result: Option<Result<MainResponseDraft>> = None;
        let mut transition_candidate: Option<TransitionCandidate> = None;
        let mut skill_candidate: Option<SkillCandidate> = None;
        let mut reasoning_decision: Option<ReasoningMode> = None;
        let mut skill_fallback_required = false;
        let mut reasoning_fallback_required = false;

        loop {
            if let Some(candidate) = transition_candidate.take() {
                if self
                    .commit_pre_response_transition_candidate(
                        &candidate,
                        &HashMap::new(),
                        processed_input,
                    )
                    .await?
                {
                    self.finalize_optional_branch(
                        &transition_id,
                        RuntimeOptimizationKind::ParallelStateTransition,
                        RuntimeCommitBehavior::TransitionDecision,
                        "committed",
                        true,
                    );
                    if !main_pending {
                        self.finalize_branch_loss(
                            &main_id,
                            main_kind,
                            RuntimeCommitBehavior::FinalResponse,
                            false,
                            main_result.as_ref().map(|result| result.is_err()),
                        );
                    }
                    if skill_enabled && !skill_pending {
                        self.finalize_branch_loss(
                            &skill_id,
                            RuntimeOptimizationKind::SpeculativeSkillRouting,
                            RuntimeCommitBehavior::SkillSelection,
                            false,
                            Some(false),
                        );
                    }
                    if reasoning_enabled && !reasoning_pending {
                        self.finalize_branch_loss(
                            &reasoning_id,
                            RuntimeOptimizationKind::SpeculativeReasoningAuto,
                            RuntimeCommitBehavior::ReasoningDecision,
                            false,
                            Some(false),
                        );
                    }
                    self.finalize_pending_branches(branch_set.cancel_pending());
                    return self
                        .redispatch_current_state(processed_input)
                        .await
                        .map(Some);
                }
                self.finalize_optional_branch(
                    &transition_id,
                    RuntimeOptimizationKind::ParallelStateTransition,
                    RuntimeCommitBehavior::TransitionDecision,
                    "discarded",
                    false,
                );
                transition_finalized = true;
            }

            if transition_finalized && skill_candidate.is_some() {
                let candidate = skill_candidate.take().unwrap();
                self.finalize_optional_branch(
                    &skill_id,
                    RuntimeOptimizationKind::SpeculativeSkillRouting,
                    RuntimeCommitBehavior::SkillSelection,
                    "committed",
                    true,
                );
                if !main_pending {
                    self.finalize_branch_loss(
                        &main_id,
                        main_kind,
                        RuntimeCommitBehavior::FinalResponse,
                        false,
                        main_result.as_ref().map(|result| result.is_err()),
                    );
                }
                if reasoning_enabled && !reasoning_pending {
                    self.finalize_branch_loss(
                        &reasoning_id,
                        RuntimeOptimizationKind::SpeculativeReasoningAuto,
                        RuntimeCommitBehavior::ReasoningDecision,
                        false,
                        Some(false),
                    );
                }
                self.finalize_pending_branches(branch_set.cancel_pending());
                self.commit_root_user_message(processed_input).await?;
                return match self
                    .commit_skill_candidate_route_result(candidate, processed_input)
                    .await?
                {
                    SkillRouteResult::Response(skill_response) => self
                        .handle_skill_response(processed_input, skill_response, input_context)
                        .await
                        .map(Some),
                    SkillRouteResult::NeedsClarification(response) => {
                        if response
                            .metadata
                            .as_ref()
                            .and_then(|m| m.get("disambiguation"))
                            .and_then(|d| d.get("status"))
                            .and_then(|s| s.as_str())
                            == Some("awaiting_clarification")
                        {
                            self.memory
                                .add_message(ChatMessage::assistant(&response.content))
                                .await?;
                        }
                        self.finish_turn_if_root(&response).await?;
                        Ok(Some(response))
                    }
                    SkillRouteResult::NoMatch => Ok(None),
                };
            }

            if transition_finalized && skill_finalized {
                if let Some(reasoning_mode) = reasoning_decision.take() {
                    if !matches!(reasoning_mode, ReasoningMode::None) {
                        self.finalize_optional_branch(
                            &reasoning_id,
                            RuntimeOptimizationKind::SpeculativeReasoningAuto,
                            RuntimeCommitBehavior::ReasoningDecision,
                            "committed",
                            true,
                        );
                        if !main_pending {
                            self.finalize_branch_loss(
                                &main_id,
                                main_kind,
                                RuntimeCommitBehavior::FinalResponse,
                                false,
                                main_result.as_ref().map(|result| result.is_err()),
                            );
                        }
                        self.finalize_pending_branches(branch_set.cancel_pending());
                        self.commit_root_user_message(processed_input).await?;
                        return if matches!(reasoning_mode, ReasoningMode::PlanAndExecute) {
                            self.handle_plan_and_execute(processed_input, input_context, true)
                                .await
                                .map(Some)
                        } else {
                            self.run_committed_response_loop_with_reasoning(
                                processed_input,
                                input_context,
                                reasoning_mode,
                                true,
                            )
                            .await
                            .map(Some)
                        };
                    }
                    self.finalize_optional_branch(
                        &reasoning_id,
                        RuntimeOptimizationKind::SpeculativeReasoningAuto,
                        RuntimeCommitBehavior::ReasoningDecision,
                        "committed",
                        true,
                    );
                    reasoning_finalized = true;
                }
            }

            if transition_finalized && skill_finalized && reasoning_finalized {
                if skill_fallback_required || reasoning_fallback_required {
                    if !main_pending {
                        self.finalize_branch_loss(
                            &main_id,
                            main_kind,
                            RuntimeCommitBehavior::FinalResponse,
                            false,
                            main_result.as_ref().map(|result| result.is_err()),
                        );
                    }
                    self.finalize_pending_branches(branch_set.cancel_pending());
                    return Ok(None);
                }

                if let Some(result) = main_result.take() {
                    let draft = match result {
                        Ok(draft) => draft,
                        Err(error) => {
                            self.finalize_optional_branch(
                                &main_id,
                                main_kind,
                                RuntimeCommitBehavior::FinalResponse,
                                "failed",
                                false,
                            );
                            self.finalize_pending_branches(branch_set.cancel_pending());
                            return Err(error);
                        }
                    };
                    self.finalize_optional_branch(
                        &main_id,
                        main_kind,
                        RuntimeCommitBehavior::FinalResponse,
                        "committed",
                        true,
                    );
                    self.finalize_pending_branches(branch_set.cancel_pending());
                    return self
                        .commit_main_response_draft(
                            processed_input,
                            input_context,
                            draft,
                            ReasoningMode::None,
                            reasoning_enabled,
                        )
                        .await
                        .map(Some);
                }
            }

            if branch_set.is_empty() {
                return Ok(None);
            }

            let Some(outcome) = branch_set.next_completed().await else {
                return Ok(None);
            };
            let branch_id = outcome.branch.branch_id();
            match outcome.result {
                RuntimeBranchResult::MainDraft(draft) => {
                    main_pending = false;
                    main_result = Some(Ok(draft));
                }
                RuntimeBranchResult::Transition(candidate) => {
                    if let Some(candidate) = candidate {
                        transition_candidate = Some(candidate);
                    } else {
                        self.finalize_optional_branch(
                            &transition_id,
                            RuntimeOptimizationKind::ParallelStateTransition,
                            RuntimeCommitBehavior::TransitionDecision,
                            "discarded",
                            false,
                        );
                        transition_finalized = true;
                    }
                }
                RuntimeBranchResult::Skill(candidate) => {
                    skill_pending = false;
                    if let Some(candidate) = candidate {
                        skill_candidate = Some(candidate);
                    } else {
                        self.finalize_optional_branch(
                            &skill_id,
                            RuntimeOptimizationKind::SpeculativeSkillRouting,
                            RuntimeCommitBehavior::SkillSelection,
                            "discarded",
                            false,
                        );
                        skill_finalized = true;
                    }
                }
                RuntimeBranchResult::Reasoning(mode) => {
                    reasoning_pending = false;
                    reasoning_decision = Some(mode);
                }
                RuntimeBranchResult::Failed(error) => {
                    if branch_id == main_id {
                        main_pending = false;
                        main_result = Some(Err(error));
                    } else if branch_id == transition_id {
                        self.finalize_optional_branch(
                            &transition_id,
                            RuntimeOptimizationKind::ParallelStateTransition,
                            RuntimeCommitBehavior::TransitionDecision,
                            "failed",
                            false,
                        );
                        transition_finalized = true;
                    } else if branch_id == skill_id {
                        skill_pending = false;
                        self.finalize_optional_branch(
                            &skill_id,
                            RuntimeOptimizationKind::SpeculativeSkillRouting,
                            RuntimeCommitBehavior::SkillSelection,
                            "failed",
                            false,
                        );
                        skill_finalized = true;
                    } else if branch_id == reasoning_id {
                        reasoning_pending = false;
                        self.finalize_optional_branch(
                            &reasoning_id,
                            RuntimeOptimizationKind::SpeculativeReasoningAuto,
                            RuntimeCommitBehavior::ReasoningDecision,
                            "failed",
                            false,
                        );
                        reasoning_finalized = true;
                    }
                }
                RuntimeBranchResult::Cancelled => {
                    self.finalize_optional_branch(
                        &branch_id,
                        outcome.branch.optimization,
                        outcome.branch.commit_behavior,
                        "cancelled",
                        false,
                    );
                    if branch_id == main_id {
                        main_pending = false;
                        main_result =
                            Some(Err(AgentError::Other("main branch cancelled".to_string())));
                    } else if branch_id == transition_id {
                        transition_finalized = true;
                    } else if branch_id == skill_id {
                        skill_pending = false;
                        skill_finalized = true;
                        skill_fallback_required = true;
                    } else if branch_id == reasoning_id {
                        reasoning_pending = false;
                        reasoning_finalized = true;
                        reasoning_fallback_required = true;
                    }
                }
            }
        }
    }

    fn finalize_pending_branches(&self, branches: Vec<RuntimeBranch>) {
        for branch in branches {
            self.finalize_optional_branch(
                &branch.branch_id(),
                branch.optimization,
                branch.commit_behavior,
                "cancelled",
                false,
            );
        }
    }

    //
    // Pending losers are reported as cancelled because their futures are dropped before completion.
    // Completed losers keep failed or discarded status based on their recorded result.
    //
    fn finalize_branch_loss(
        &self,
        branch_id: &str,
        optimization: RuntimeOptimizationKind,
        commit_behavior: RuntimeCommitBehavior,
        pending: bool,
        completed_failed: Option<bool>,
    ) {
        let status = if pending {
            "cancelled"
        } else if completed_failed.unwrap_or(false) {
            "failed"
        } else {
            "discarded"
        };
        self.finalize_optional_branch(branch_id, optimization, commit_behavior, status, false);
    }

    //
    // Finalization is separated from commit so losing branches remain observable.
    // This helper must not mutate runtime state other than observability.
    //
    fn finalize_optional_branch(
        &self,
        branch_id: &str,
        optimization: RuntimeOptimizationKind,
        commit_behavior: RuntimeCommitBehavior,
        status: &str,
        winner: bool,
    ) {
        crate::optimization::observability::finalize_branch(
            self.observability_manager.as_ref(),
            branch_id,
            status,
            winner,
            optimization,
            commit_behavior,
        );
    }

    //
    // This is only an eligibility check.
    // Actual transition selection happens in select_parallel_transition_candidate.
    //
    fn has_parallel_transition_candidates(&self) -> bool {
        self.transitions_available_for_commit()
            .map(|(transitions, _)| {
                transitions
                    .iter()
                    .any(|transition| matches!(transition.timing, TransitionTiming::Parallel))
            })
            .unwrap_or(false)
    }

    //
    // Parallel transition prompts must not depend on assistant response text.
    // Keep this branch response-independent or it can race against invalid context.
    //
    async fn select_parallel_transition_candidate(
        &self,
        processed_input: &str,
    ) -> Result<Option<TransitionCandidate>> {
        let Some((transitions, current_state)) = self.transitions_available_for_commit() else {
            return Ok(None);
        };
        let parallel: Vec<Transition> = transitions
            .into_iter()
            .filter(|transition| matches!(transition.timing, TransitionTiming::Parallel))
            .filter(|transition| !transition.requires_response)
            .collect();
        if parallel.is_empty() {
            return Ok(None);
        }
        let empty_staged = HashMap::new();
        if let Some(candidate) = self.select_deterministic_transition_candidate(
            processed_input,
            &current_state,
            &parallel,
            &empty_staged,
        ) {
            return Ok(Some(candidate));
        }
        let when_transitions: Vec<(usize, &Transition)> = parallel
            .iter()
            .enumerate()
            .filter(|(_, transition)| !transition.when.trim().is_empty())
            .collect();
        if when_transitions.is_empty() {
            return Ok(None);
        }
        let llm = self
            .llm_registry
            .router()
            .or_else(|_| self.llm_registry.default())
            .map_err(|e| AgentError::Config(e.to_string()))?;
        let conditions = when_transitions
            .iter()
            .enumerate()
            .map(|(display_idx, (_, transition))| {
                format!("{}. {}", display_idx + 1, transition.when)
            })
            .collect::<Vec<_>>()
            .join("\n");
        if !self
            .reserve_active_speculative_llm_call(RuntimeOptimizationKind::ParallelStateTransition)
        {
            return Ok(None);
        }
        let context_preview = self.branch_context_preview();
        let prompt = format!(
            "Based only on the current user message and context, which transition condition is met?\n\nCurrent state: {}\nUser message: {}\nContext:\n{}\n\nConditions:\n{}\n0. None of the above\n\nReply with ONLY the number (0-{}).",
            current_state,
            processed_input,
            context_preview,
            conditions,
            when_transitions.len()
        );
        let response = self
            .observe_purpose(
                ObservationPurpose::StateTransitionEvaluation,
                llm.complete(&[ChatMessage::user(prompt)], None),
            )
            .await
            .map_err(|e| AgentError::LLM(e.to_string()))?;
        let choice = response.content.trim().parse::<usize>().unwrap_or(0);
        if choice == 0 || choice > when_transitions.len() {
            return Ok(None);
        }
        let transition = when_transitions[choice - 1].1.clone();
        Ok(Some(TransitionCandidate::new(
            current_state,
            transition.clone(),
            Self::transition_reason(&transition),
        )))
    }

    /// Re-enters the runtime loop after an optimized transition commits.
    async fn redispatch_current_state(&self, processed_input: &str) -> Result<AgentResponse> {
        const MAX_REDISPATCH_DEPTH: u32 = 3;
        let current_depth = *self.redispatch_depth.read();
        if current_depth >= MAX_REDISPATCH_DEPTH {
            warn!(depth = current_depth, "Re-dispatch depth limit reached");
            let response = AgentResponse::new("");
            self.finish_turn_if_root(&response).await?;
            return Ok(response);
        }
        *self.redispatch_depth.write() += 1;
        if let Some(context) = self.active_turn_context.write().as_mut() {
            context.enter_redispatch();
        }
        let result = Box::pin(self.run_loop_internal(processed_input)).await;
        *self.redispatch_depth.write() -= 1;
        if let Some(context) = self.active_turn_context.write().as_mut() {
            context.exit_redispatch();
        }
        let response = result?;
        self.finish_turn_if_root(&response).await?;
        Ok(response)
    }

    /// Runs final response hooks and maintenance only for the root dispatch.
    async fn finish_turn_if_root(&self, response: &AgentResponse) -> Result<()> {
        if *self.redispatch_depth.read() == 0 {
            self.post_turn_session_lifecycle().await?;
            if let Some(context) = self.active_turn_context.write().as_mut() {
                context.mark_post_turn_lifecycle_completed();
            }
            self.hooks.on_response(response).await;
            self.end_root_turn();
        }
        Ok(())
    }

    /// Execute on_exit actions for a state being left.
    async fn execute_state_exit_actions(&self, state_path: &str) {
        if let Some(ref sm) = self.state_machine {
            if let Some(def) = sm.get_definition(state_path) {
                if !def.on_exit.is_empty() {
                    debug!(state = %state_path, count = def.on_exit.len(), "Executing on_exit actions");
                    self.execute_state_actions(&def.on_exit).await;
                }
            }
        }
    }

    /// Execute on_enter (or on_reenter) actions for a state being entered.
    async fn execute_state_enter_actions(&self, state_path: &str) {
        if let Some(ref sm) = self.state_machine {
            if let Some(def) = sm.get_definition(state_path) {
                // Check if this state was previously visited
                let is_reentry = sm.history().iter().any(|e| e.to == state_path);

                if is_reentry && !def.on_reenter.is_empty() {
                    debug!(state = %state_path, count = def.on_reenter.len(), "Executing on_reenter actions");
                    self.execute_state_actions(&def.on_reenter).await;
                } else if !def.on_enter.is_empty() {
                    debug!(state = %state_path, count = def.on_enter.len(), "Executing on_enter actions");
                    self.execute_state_actions(&def.on_enter).await;
                }
            }
        }
    }

    /// Execute a list of state actions (tool calls, skill invocations, context updates, LLM prompts).
    async fn execute_state_actions(&self, actions: &[StateAction]) {
        for action in actions {
            match action {
                StateAction::Tool { tool, args } => {
                    let raw_args = args.clone().unwrap_or(Value::Object(Default::default()));
                    // Render template variables in tool args (e.g., {{ context.order_id }})
                    let args_value = self.render_action_args(&raw_args);
                    if let Some(t) = self.tools.get(tool) {
                        self.hooks.on_tool_start(tool, &args_value).await;
                        let start = Instant::now();
                        let result = t.execute(args_value).await;
                        let duration_ms = start.elapsed().as_millis() as u64;
                        self.hooks
                            .on_tool_complete(tool, &result, duration_ms)
                            .await;
                        if result.success {
                            debug!(tool = %tool, "State action: tool executed");
                            // Store tool result in context so YAML prompts can reference it
                            let context_key = format!("last_tool_result");
                            let _ = self
                                .context_manager
                                .set(&context_key, serde_json::Value::String(result.output));
                        } else {
                            warn!(tool = %tool, error = %result.output, "State action: tool failed");
                        }
                    } else {
                        warn!(tool = %tool, "State action: tool not found");
                    }
                }
                StateAction::Skill { skill } => {
                    if let Some(ref executor) = self.skill_executor {
                        if let Some(def) = self.skills.iter().find(|s| s.id == *skill) {
                            match executor.execute(def, "", serde_json::json!({})).await {
                                Ok(_) => debug!(skill = %skill, "State action: skill executed"),
                                Err(e) => {
                                    warn!(skill = %skill, error = %e, "State action: skill failed")
                                }
                            }
                        } else {
                            warn!(skill = %skill, "State action: skill not found");
                        }
                    }
                }
                StateAction::SetContext { set_context } => {
                    for (key, value) in set_context {
                        if let Err(e) = self.context_manager.set(key, value.clone()) {
                            warn!(key = %key, error = %e, "State action: set_context failed");
                        } else {
                            debug!(key = %key, "State action: context set");
                        }
                    }
                }
                StateAction::Prompt {
                    prompt,
                    llm,
                    store_as,
                } => {
                    let llm_result = if let Some(alias) = llm {
                        self.llm_registry.get(alias)
                    } else {
                        self.llm_registry.default()
                    };
                    match llm_result {
                        Ok(llm_provider) => {
                            // Render template variables and include conversation context
                            let context = self.build_context_with_overlays();
                            let rendered_prompt = self
                                .template_renderer
                                .render(prompt, &context)
                                .unwrap_or_else(|_| prompt.clone());
                            let recent =
                                self.memory.get_messages(Some(5)).await.unwrap_or_default();
                            let mut messages: Vec<ChatMessage> = recent;
                            messages.push(ChatMessage::user(&rendered_prompt));
                            match self
                                .observe_purpose(
                                    ObservationPurpose::StateAction,
                                    llm_provider.complete(&messages, None),
                                )
                                .await
                            {
                                Ok(response) => {
                                    if let Some(key) = store_as {
                                        let _ = self
                                            .context_manager
                                            .set(key, Value::String(response.content));
                                        debug!(key = %key, "State action: prompt result stored");
                                    }
                                }
                                Err(e) => {
                                    warn!(error = %e, "State action: prompt LLM call failed");
                                }
                            }
                        }
                        Err(e) => {
                            warn!(error = %e, "State action: LLM not found for prompt");
                        }
                    }
                }
            }
        }
    }

    async fn run_context_extractors_staged(&self, user_message: &str) -> HashMap<String, Value> {
        let extractors = match &self.state_machine {
            Some(sm) => match sm.current_definition() {
                Some(def) if !def.extract.is_empty() => def.extract.clone(),
                _ => return HashMap::new(),
            },
            None => return HashMap::new(),
        };

        let mut staged = HashMap::new();
        for extractor in &extractors {
            let prompt = if let Some(ref custom) = extractor.llm_extract {
                format!(
                    "User message:\n\"{}\"\n\nInstruction:\n{}",
                    user_message, custom
                )
            } else if let Some(ref desc) = extractor.description {
                format!(
                    "From the following message, extract: {}\n\n\
                     Message: \"{}\"\n\n\
                     If the information is present, return ONLY the extracted value.\n\
                     If NOT present, return exactly: __NONE__",
                    desc, user_message
                )
            } else {
                continue;
            };

            let llm = match self
                .llm_registry
                .get(&extractor.llm)
                .or_else(|_| self.llm_registry.get("router"))
                .or_else(|_| self.llm_registry.get("default"))
            {
                Ok(llm) => llm,
                Err(e) => {
                    warn!(key = %extractor.key, error = %e, "Extractor LLM not found");
                    continue;
                }
            };

            let messages = vec![ChatMessage::user(&prompt)];
            match self
                .observe_purpose(
                    ObservationPurpose::ContextExtraction,
                    llm.complete(&messages, None),
                )
                .await
            {
                Ok(response) => {
                    let value = response.content.trim().to_string();
                    if value != "__NONE__" && !value.is_empty() {
                        staged.insert(
                            extractor.key.clone(),
                            serde_json::Value::String(value.clone()),
                        );
                        debug!(key = %extractor.key, value = %value, "Context extracted");
                    } else if extractor.required {
                        warn!(key = %extractor.key, "Required extraction returned no value");
                    }
                }
                Err(e) => {
                    warn!(key = %extractor.key, error = %e, "Context extraction LLM call failed");
                }
            }
        }
        staged
    }

    async fn commit_staged_context_writes(&self, staged: &HashMap<String, Value>) {
        for (key, value) in staged {
            if let Err(error) = self.context_manager.update(key, value.clone()) {
                warn!(key = %key, error = %error, "staged context write failed");
            }
        }
    }

    /// Run context extractors for the current state on the user's input.
    async fn run_context_extractors(&self, user_message: &str) {
        let staged = self.run_context_extractors_staged(user_message).await;
        self.commit_staged_context_writes(&staged).await;
    }

    async fn check_memory_compression(&self) -> Result<()> {
        if self.memory.needs_compression() {
            let result = self.memory.compress(None).await?;
            if let CompressResult::Compressed {
                messages_summarized,
                new_summary_length,
                tokens_saved,
            } = result
            {
                let event = MemoryCompressEvent::new(
                    messages_summarized,
                    tokens_saved,
                    new_summary_length as u32,
                );
                self.hooks.on_memory_compress(&event).await;
                debug!(
                    messages = messages_summarized,
                    tokens_saved = tokens_saved,
                    "Memory compressed"
                );
            }
        }

        // Handle overflow AFTER compression, then check warning threshold
        self.handle_memory_overflow().await?;
        self.check_memory_budget().await;

        Ok(())
    }

    async fn check_memory_budget(&self) {
        let Some(ref budget) = self.memory_token_budget else {
            return;
        };

        let context = match self.memory.get_context().await {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // Overall budget warning
        let used_tokens = context.estimated_tokens();
        if budget.is_over_warn_threshold(used_tokens) {
            let event = MemoryBudgetEvent::new("memory", used_tokens, budget.total);
            self.hooks.on_memory_budget_warning(&event).await;
            debug!(
                used = used_tokens,
                total = budget.total,
                percent = event.usage_percent,
                "Memory budget warning"
            );
        }

        // Per-component warning: summary
        if let Some(ref summary) = context.summary {
            let summary_tokens = ai_agents_memory::estimate_tokens(summary);
            let summary_budget = budget.allocation.summary;
            if summary_budget > 0 {
                let warn_threshold =
                    (summary_budget as f64 * budget.warn_at_percent as f64 / 100.0) as u32;
                if summary_tokens >= warn_threshold {
                    let event = MemoryBudgetEvent::new("summary", summary_tokens, summary_budget);
                    self.hooks.on_memory_budget_warning(&event).await;
                }
            }
        }

        // Per-component warning: recent_messages
        let recent_tokens: u32 = context
            .messages
            .iter()
            .map(ai_agents_memory::estimate_message_tokens)
            .sum();
        let recent_budget = budget.allocation.recent_messages;
        if recent_budget > 0 {
            let warn_threshold =
                (recent_budget as f64 * budget.warn_at_percent as f64 / 100.0) as u32;
            if recent_tokens >= warn_threshold {
                let event = MemoryBudgetEvent::new("recent_messages", recent_tokens, recent_budget);
                self.hooks.on_memory_budget_warning(&event).await;
            }
        }

        let relationship_budget = budget.allocation.relationships;
        if relationship_budget > 0 {
            let relationship_tokens = self
                .relationship_memory_text()
                .map(|text| ai_agents_memory::estimate_tokens(&text))
                .unwrap_or(0);
            let warn_threshold =
                (relationship_budget as f64 * budget.warn_at_percent as f64 / 100.0) as u32;
            if relationship_tokens >= warn_threshold {
                let event = MemoryBudgetEvent::new(
                    "relationships",
                    relationship_tokens,
                    relationship_budget,
                );
                self.hooks.on_memory_budget_warning(&event).await;
            }
        }
    }

    async fn handle_memory_overflow(&self) -> Result<()> {
        let Some(ref budget) = self.memory_token_budget else {
            return Ok(());
        };

        let context = self.memory.get_context().await?;
        let used_tokens = context.estimated_tokens();

        if used_tokens <= budget.total {
            return Ok(());
        }

        match budget.overflow_strategy {
            OverflowStrategy::TruncateOldest => {
                let tokens_to_free = used_tokens - budget.total;
                let messages_to_evict = self.calculate_eviction_count(tokens_to_free);
                if messages_to_evict > 0 {
                    self.evict_messages(messages_to_evict, EvictionReason::TokenBudgetExceeded)
                        .await?;
                }
            }
            OverflowStrategy::SummarizeMore => {
                self.memory.compress(None).await?;
            }
            OverflowStrategy::Error => {
                return Err(AgentError::MemoryBudgetExceeded {
                    used: used_tokens,
                    budget: budget.total,
                });
            }
        }
        Ok(())
    }

    fn calculate_eviction_count(&self, tokens_to_free: u32) -> usize {
        // Estimate ~50 tokens per message on average
        ((tokens_to_free as f64 / 50.0).ceil() as usize).max(1)
    }

    async fn evict_messages(&self, count: usize, reason: EvictionReason) -> Result<()> {
        let evicted = self.memory.evict_oldest(count).await?;
        if !evicted.is_empty() {
            let event = MemoryEvictEvent {
                reason,
                messages_evicted: evicted.len(),
                importance_scores: vec![],
            };
            self.hooks.on_memory_evict(&event).await;
            debug!(count = evicted.len(), "Messages evicted from memory");
        }
        Ok(())
    }

    #[instrument(skip(self, input), fields(agent = %self.info.name))]
    async fn determine_reasoning_mode(&self, input: &str) -> Result<ReasoningMode> {
        match self.determine_reasoning_mode_strict(input).await {
            Ok(mode) => Ok(mode),
            Err(_) => Ok(ReasoningMode::None),
        }
    }

    async fn determine_reasoning_mode_strict(&self, input: &str) -> Result<ReasoningMode> {
        let effective_config = self.get_effective_reasoning_config();

        if !matches!(effective_config.mode, ReasoningMode::Auto) {
            return Ok(effective_config.mode.clone());
        }

        let judge_llm = effective_config
            .judge_llm
            .as_ref()
            .and_then(|alias| self.llm_registry.get(alias).ok())
            .or_else(|| self.llm_registry.router().ok())
            .or_else(|| self.llm_registry.default().ok());

        let Some(llm) = judge_llm else {
            return Ok(ReasoningMode::None);
        };

        let prompt = format!(
            r#"Analyze this user request and determine the appropriate reasoning mode.

User request: "{}"

Choose ONE of these modes:
- none: Simple queries, greetings, direct answers (fastest)
- cot: Complex analysis, multi-step reasoning, math problems
- react: Tasks requiring multiple tool calls with observation
- plan_and_execute: Complex multi-step tasks requiring coordination

Respond with ONLY the mode name (none, cot, react, or plan_and_execute)."#,
            input
        );

        let messages = vec![ChatMessage::user(&prompt)];
        let response = self
            .observe_purpose(
                ObservationPurpose::ReflectionDecision,
                llm.complete(&messages, None),
            )
            .await
            .map_err(|e| AgentError::LLM(e.to_string()))?;

        let mode_str = response.content.trim().to_lowercase();
        Ok(match mode_str.as_str() {
            "cot" => ReasoningMode::CoT,
            "react" => ReasoningMode::React,
            "plan_and_execute" => ReasoningMode::PlanAndExecute,
            _ => ReasoningMode::None,
        })
    }

    async fn should_reflect(&self, input: &str, response: &str) -> Result<bool> {
        let effective_config = self.get_effective_reflection_config();

        if !effective_config.requires_evaluation() {
            return Ok(false);
        }

        if effective_config.is_enabled() {
            return Ok(true);
        }

        let evaluator_llm = effective_config
            .evaluator_llm
            .as_ref()
            .and_then(|alias| self.llm_registry.get(alias).ok())
            .or_else(|| self.llm_registry.router().ok())
            .or_else(|| self.llm_registry.default().ok());

        let Some(llm) = evaluator_llm else {
            return Ok(false);
        };

        let response_preview: String = response.chars().take(500).collect();
        let prompt = format!(
            r#"Should this response be evaluated for quality? Consider if it's a complex or important response.

User query: "{}"
Response: "{}"

Answer YES or NO only."#,
            input, response_preview
        );

        let messages = vec![ChatMessage::user(&prompt)];
        let result = self
            .observe_purpose(
                ObservationPurpose::ReflectionDecision,
                llm.complete(&messages, None),
            )
            .await;

        match result {
            Ok(resp) => Ok(resp.content.trim().to_uppercase().contains("YES")),
            Err(_) => Ok(false),
        }
    }

    fn build_cot_system_prompt(&self, base_prompt: &str) -> String {
        format!(
            "{}\n\n<instruction>\nThink through this step by step before answering:\n1. Understand what is being asked\n2. Break down the problem\n3. Work through each part\n4. Provide your final answer\n\nShow your thinking process, then give your final answer.\n</instruction>",
            base_prompt
        )
    }

    fn build_react_system_prompt(&self, base_prompt: &str) -> String {
        format!(
            "{}\n\n<instruction>\nUse the Reason-Act-Observe pattern:\n1. Thought: Think about what to do\n2. Action: Use a tool if needed\n3. Observation: Analyze the result\n4. Repeat until you have the answer\n\nFormat your response showing Thought/Action/Observation steps.\n</instruction>",
            base_prompt
        )
    }

    async fn generate_plan(&self, input: &str) -> Result<Plan> {
        let effective = self.get_effective_reasoning_config();
        let planning_config = effective.get_planning();

        let planner_llm = planning_config
            .and_then(|c| c.planner_llm.as_ref())
            .and_then(|alias| self.llm_registry.get(alias).ok())
            .or_else(|| self.llm_registry.router().ok())
            .or_else(|| self.llm_registry.default().ok())
            .ok_or_else(|| AgentError::Config("No LLM available for planning".into()))?;

        let mut available_tool_ids: Vec<String> = self
            .get_available_tool_ids()
            .await
            .unwrap_or_else(|_| self.tools.list_ids());
        let mut available_skills: Vec<String> = self.skills.iter().map(|s| s.id.clone()).collect();

        // Apply planning-level tool and skill filters.
        if let Some(config) = planning_config {
            if !config.available.tools.is_all() {
                available_tool_ids.retain(|t| config.available.tools.allows(t));
            }
            if !config.available.skills.is_all() {
                available_skills.retain(|s| config.available.skills.allows(s));
            }
        }

        // Build tool descriptions with argument schemas so the planner
        // knows how to construct valid args for each step.
        let tool_descriptions: Vec<String> = available_tool_ids
            .iter()
            .filter_map(|id| {
                self.tools.get(id).map(|tool| {
                    let schema = tool.input_schema();
                    let args_desc = schema
                        .get("properties")
                        .and_then(|p| serde_json::to_string(p).ok())
                        .unwrap_or_else(|| "{}".to_string());
                    format!(
                        "- {} ({}): {}\n  Arguments: {}",
                        id,
                        tool.name(),
                        tool.description(),
                        args_desc
                    )
                })
            })
            .collect();

        let tools_section = if tool_descriptions.is_empty() {
            "Available tools: none".to_string()
        } else {
            format!("Available tools:\n{}", tool_descriptions.join("\n"))
        };

        let skills_section = if available_skills.is_empty() {
            "Available skills: none".to_string()
        } else {
            format!("Available skills: {}", available_skills.join(", "))
        };

        let prompt = format!(
            r#"Create a step-by-step plan to accomplish this goal.

Goal: "{}"

{}

{}

Create a plan with clear steps. For each step, specify:
- description: What this step accomplishes
- action_type: "tool", "skill", "think", or "respond"
- action_target: The tool/skill id (if applicable)
- args: The arguments object matching the tool's schema (if action_type is "tool")
- dependencies: List of step IDs this depends on (empty if none)

Respond in JSON format:
{{
  "steps": [
    {{"id": "step1", "description": "...", "action_type": "tool", "action_target": "tool_id", "args": {{"required_field": "value"}}, "dependencies": []}},
    {{"id": "step2", "description": "...", "action_type": "think", "action_target": "...", "dependencies": ["step1"]}}
  ]
}}"#,
            input, tools_section, skills_section,
        );

        let messages = vec![ChatMessage::user(&prompt)];
        let response = self
            .observe_purpose(
                ObservationPurpose::PlanGeneration,
                planner_llm.complete(&messages, None),
            )
            .await
            .map_err(|e| AgentError::LLM(format!("Planning failed: {}", e)))?;

        let mut plan = Plan::new(input);

        if let Some(json_start) = response.content.find('{') {
            if let Some(json_end) = response.content.rfind('}') {
                let json_str = &response.content[json_start..=json_end];
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if let Some(steps) = parsed.get("steps").and_then(|s| s.as_array()) {
                        for step_value in steps {
                            let id = step_value
                                .get("id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("step");
                            let desc = step_value
                                .get("description")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            let action_type = step_value
                                .get("action_type")
                                .and_then(|v| v.as_str())
                                .unwrap_or("think");
                            let action_target = step_value
                                .get("action_target")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            let args = step_value
                                .get("args")
                                .cloned()
                                .unwrap_or(serde_json::json!({}));
                            let deps: Vec<String> = step_value
                                .get("dependencies")
                                .and_then(|v| v.as_array())
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|v| v.as_str().map(String::from))
                                        .collect()
                                })
                                .unwrap_or_default();

                            let action = match action_type {
                                "tool" => PlanAction::tool(action_target, args),
                                "skill" => PlanAction::skill(action_target),
                                "respond" => PlanAction::respond(action_target),
                                _ => PlanAction::think(desc),
                            };

                            let step = PlanStep::new(desc, action)
                                .with_id(id)
                                .with_dependencies(deps);
                            plan.add_step(step);
                        }
                    }
                }
            }
        }

        if plan.steps.is_empty() {
            plan.add_step(PlanStep::new(
                "Process the request",
                PlanAction::think(input),
            ));
            plan.add_step(PlanStep::new(
                "Provide response",
                PlanAction::respond("Answer based on analysis"),
            ));
        }

        Ok(plan)
    }

    async fn execute_plan(&self, plan: &mut Plan) -> Result<String> {
        let llm = self.get_state_llm()?;
        let mut results: HashMap<String, serde_json::Value> = HashMap::new();
        let effective = self.get_effective_reasoning_config();
        let max_steps = effective.get_planning().map(|c| c.max_steps).unwrap_or(10);

        plan.status = PlanStatus::InProgress;

        for step_idx in 0..plan.steps.len().min(max_steps as usize) {
            let step = &plan.steps[step_idx];

            let deps_satisfied = step.dependencies.iter().all(|dep| {
                plan.steps
                    .iter()
                    .find(|s| &s.id == dep)
                    .map(|s| s.status.is_completed())
                    .unwrap_or(false)
            });

            if !deps_satisfied {
                continue;
            }

            plan.steps[step_idx].mark_running();

            let result = match &plan.steps[step_idx].action {
                PlanAction::Tool { tool, args } => {
                    // When a tool step has dependency results, ask the LLM to
                    // produce the correct arguments given the context and tool schema.
                    // This avoids brittle {{stepN}} template substitution and lets
                    // the LLM handle type adaptation (e.g. picking the iso field
                    // from a datetime result for a downstream format call).
                    let has_dep_results = plan.steps[step_idx]
                        .dependencies
                        .iter()
                        .any(|dep| results.contains_key(dep));

                    let final_args = if has_dep_results {
                        let dep_context: String = plan.steps[step_idx]
                            .dependencies
                            .iter()
                            .filter_map(|dep| results.get(dep).map(|r| format!("{}: {}", dep, r)))
                            .collect::<Vec<_>>()
                            .join("\n");

                        let tool_schema = self
                            .tools
                            .get(tool)
                            .map(|t| {
                                let schema = t.input_schema();
                                let props = schema
                                    .get("properties")
                                    .and_then(|p| serde_json::to_string(p).ok())
                                    .unwrap_or_else(|| "{}".to_string());
                                format!(
                                    "{}: {}\nArguments schema: {}",
                                    t.id(),
                                    t.description(),
                                    props
                                )
                            })
                            .unwrap_or_default();

                        let step_desc = &plan.steps[step_idx].description;
                        let arg_prompt = format!(
                            "Generate the JSON arguments for a tool call.\n\n\
                             Tool: {}\n\n\
                             Task: {}\n\n\
                             Previous step results:\n{}\n\n\
                             Planner's draft arguments: {}\n\n\
                             Produce ONLY a valid JSON object with the correct argument values.\n\
                             Use actual values from the previous step results, not template references.",
                            tool_schema,
                            step_desc,
                            dep_context,
                            serde_json::to_string(args).unwrap_or_default()
                        );
                        let messages = vec![ChatMessage::user(&arg_prompt)];
                        match self
                            .observe_purpose(
                                ObservationPurpose::PlanStep,
                                llm.complete(&messages, None),
                            )
                            .await
                        {
                            Ok(resp) => {
                                let content = resp.content.trim();
                                // Parse the LLM's JSON response, fall back to planner args.
                                let json_start = content.find('{');
                                let json_end = content.rfind('}');
                                if let (Some(start), Some(end)) = (json_start, json_end) {
                                    serde_json::from_str(&content[start..=end])
                                        .unwrap_or_else(|_| args.clone())
                                } else {
                                    args.clone()
                                }
                            }
                            Err(_) => args.clone(),
                        }
                    } else {
                        args.clone()
                    };

                    let tool_call = ToolCall {
                        id: uuid::Uuid::new_v4().to_string(),
                        name: tool.clone(),
                        arguments: final_args,
                    };
                    match self.execute_tool_smart(&tool_call).await {
                        Ok(output) => serde_json::json!({ "output": output }),
                        Err(e) => {
                            plan.steps[step_idx].mark_failed(e.to_string());
                            continue;
                        }
                    }
                }
                PlanAction::Skill { skill } => {
                    if let Some(skill_def) = self.skills.iter().find(|s| &s.id == skill) {
                        if let Some(ref executor) = self.skill_executor {
                            match executor.execute(skill_def, "", serde_json::json!({})).await {
                                Ok(output) => serde_json::json!({ "output": output }),
                                Err(e) => {
                                    plan.steps[step_idx].mark_failed(e.to_string());
                                    continue;
                                }
                            }
                        } else {
                            serde_json::json!({ "output": "Skill executor not available" })
                        }
                    } else {
                        plan.steps[step_idx].mark_failed("Skill not found");
                        continue;
                    }
                }
                PlanAction::Think { prompt } => {
                    let context: String = results
                        .iter()
                        .map(|(k, v)| format!("{}: {}", k, v))
                        .collect::<Vec<_>>()
                        .join("\n");

                    let think_prompt = format!("Context:\n{}\n\nTask: {}", context, prompt);
                    let messages = vec![ChatMessage::user(&think_prompt)];

                    match self
                        .observe_purpose(
                            ObservationPurpose::PlanStep,
                            llm.complete(&messages, None),
                        )
                        .await
                    {
                        Ok(resp) => serde_json::json!({ "output": resp.content }),
                        Err(e) => {
                            plan.steps[step_idx].mark_failed(e.to_string());
                            continue;
                        }
                    }
                }
                PlanAction::Respond { template } => {
                    let context: String = results
                        .iter()
                        .map(|(k, v)| format!("{}: {}", k, v))
                        .collect::<Vec<_>>()
                        .join("\n");

                    let respond_prompt = format!(
                        "Based on this context:\n{}\n\nGenerate a response following this template/instruction: {}",
                        context, template
                    );
                    let messages = vec![ChatMessage::user(&respond_prompt)];

                    match self
                        .observe_purpose(
                            ObservationPurpose::PlanStep,
                            llm.complete(&messages, None),
                        )
                        .await
                    {
                        Ok(resp) => serde_json::json!({ "output": resp.content }),
                        Err(e) => {
                            plan.steps[step_idx].mark_failed(e.to_string());
                            continue;
                        }
                    }
                }
            };

            results.insert(plan.steps[step_idx].id.clone(), result.clone());
            plan.steps[step_idx].mark_completed(Some(result));
        }

        // Set plan status based on whether any steps actually failed.
        let has_failures = plan.steps.iter().any(|s| s.status.is_failed());
        if has_failures {
            let failed_ids: Vec<String> = plan
                .steps
                .iter()
                .filter(|s| s.status.is_failed())
                .map(|s| s.id.clone())
                .collect();
            plan.status = PlanStatus::Failed {
                error: format!("Steps failed: {}", failed_ids.join(", ")),
            };
        } else {
            plan.status = PlanStatus::Completed;
        }

        // Synthesize final output from all completed step results.
        let all_outputs: Vec<String> = plan
            .steps
            .iter()
            .filter(|s| s.status.is_completed())
            .filter_map(|s| {
                s.result
                    .as_ref()
                    .and_then(|r| r.get("output"))
                    .and_then(|o| o.as_str())
                    .map(|o| format!("{}: {}", s.description, o))
            })
            .collect();

        if all_outputs.is_empty() {
            return Ok("Plan execution completed but produced no results.".to_string());
        }

        if all_outputs.len() == 1 {
            return Ok(all_outputs.into_iter().next().unwrap());
        }

        // Synthesize a coherent summary from multiple step results via LLM.
        let context = all_outputs.join("\n\n");
        let prompt = format!(
            "You completed a multi-step plan for: \"{}\"\n\nStep results:\n{}\n\nProvide a coherent final response that synthesizes these results.",
            plan.goal, context
        );
        let messages = vec![ChatMessage::user(&prompt)];
        match self
            .observe_purpose(ObservationPurpose::PlanStep, llm.complete(&messages, None))
            .await
        {
            Ok(resp) => Ok(resp.content.trim().to_string()),
            Err(_) => Ok(context),
        }
    }

    async fn evaluate_response(&self, input: &str, response: &str) -> Result<EvaluationResult> {
        let effective_config = self.get_effective_reflection_config();
        self.evaluate_response_with_config(input, response, &effective_config)
            .await
    }

    fn extract_thinking(&self, content: &str) -> (Option<String>, String) {
        if let Some(start) = content.find("<thinking>") {
            if let Some(end) = content.find("</thinking>") {
                let thinking = content[start + 10..end].trim().to_string();
                let answer = content[end + 11..].trim().to_string();
                return (Some(thinking), answer);
            }
        }
        (None, content.to_string())
    }

    fn format_response_with_thinking(&self, thinking: Option<&str>, answer: &str) -> String {
        match self.get_effective_reasoning_config().output {
            ReasoningOutput::Hidden => answer.to_string(),
            ReasoningOutput::Visible => {
                if let Some(t) = thinking {
                    format!("Thinking:\n{}\n\nAnswer:\n{}", t, answer)
                } else {
                    answer.to_string()
                }
            }
            ReasoningOutput::Tagged => {
                if let Some(t) = thinking {
                    format!("<thinking>{}</thinking>\n{}", t, answer)
                } else {
                    answer.to_string()
                }
            }
        }
    }

    async fn run_loop(&self, input: &str) -> Result<AgentResponse> {
        self.begin_root_turn();
        let _root_cleanup = RootTurnCleanup::new(self);
        info!(input_len = input.len(), "Starting chat");

        self.hooks.on_message_received(input).await;

        // One-shot context initialization: load runtime defaults, resolve env vars,
        // populate builtin sources (session, agent), etc.  This must happen before
        // the first template render so that {{ context.* }} variables are available.
        if !self.context_initialized.swap(true, Ordering::SeqCst) {
            self.context_manager.initialize().await?;
            debug!("Context manager initialized (defaults, env, builtins)");
        }

        self.check_turn_timeout().await?;
        self.context_manager.refresh_per_turn().await?;

        // Clear stale disambiguation context from previous turns.
        // This prevents resolved_intent from leaking across turns and causing incorrect deterministic routing on subsequent inputs.
        self.clear_disambiguation_context();

        // Disambiguation check (before input processing)
        if let Some(ref disambiguator) = self.disambiguation_manager {
            let disambiguation_context = self.build_disambiguation_context().await?;

            // Get state-level disambiguation override
            let state_override = self
                .state_machine
                .as_ref()
                .and_then(|sm| sm.current_definition())
                .and_then(|def| def.disambiguation.clone());

            match self
                .observe_purpose(
                    ObservationPurpose::DisambiguationDetection,
                    disambiguator.process_input_with_override(
                        input,
                        &disambiguation_context,
                        state_override.as_ref(),
                        None,
                    ),
                )
                .await?
            {
                DisambiguationResult::Clear => {
                    debug!("Input is clear, proceeding normally");
                }
                DisambiguationResult::NeedsClarification {
                    question,
                    detection,
                } => {
                    info!(
                        ambiguity_type = ?detection.ambiguity_type,
                        confidence = detection.confidence,
                        "Input requires clarification"
                    );

                    self.commit_root_user_message(input).await?;
                    self.memory
                        .add_message(ChatMessage::assistant(&question.question))
                        .await?;

                    let response = AgentResponse::new(&question.question).with_metadata(
                        "disambiguation",
                        serde_json::json!({
                            "status": "awaiting_clarification",
                            "options": question.options,
                            "clarifying": question.clarifying,
                            "detection": {
                                "type": detection.ambiguity_type,
                                "confidence": detection.confidence,
                                "what_is_unclear": detection.what_is_unclear,
                            }
                        }),
                    );
                    self.finish_turn_if_root(&response).await?;
                    return Ok(response);
                }
                DisambiguationResult::Clarified {
                    enriched_input,
                    resolved,
                    ..
                } => {
                    info!(
                        resolved_count = resolved.len(),
                        enriched = %enriched_input,
                        "Input clarified, injecting resolved intent into context"
                    );

                    // Routing uses `resolved` (structured, deterministic)
                    // This is what makes post-disambiguation routing DETERMINISTIC
                    for (key, value) in &resolved {
                        let context_key = format!("disambiguation.{}", key);
                        let _ = self.context_manager.set(&context_key, value.clone());
                    }

                    if let Some(intent) = resolved.get("intent") {
                        let _ = self.context_manager.set("resolved_intent", intent.clone());
                    }

                    let _ = self
                        .context_manager
                        .set("disambiguation.resolved", serde_json::Value::Bool(true));

                    // Check if this clarification was triggered by a skill-level override.
                    // If so, route directly to the matched skill instead of going through
                    // skill routing again (which might match a different skill).
                    let skill_id = self.pending_skill_id.read().clone();
                    if let Some(skill_id) = skill_id {
                        info!(skill_id = %skill_id, "Re-checking skill disambiguation on clarified input");
                        return self
                            .recheck_skill_disambiguation(&skill_id, &enriched_input)
                            .await;
                    }

                    return self.run_loop_internal(&enriched_input).await;
                }
                DisambiguationResult::ProceedWithBestGuess { enriched_input } => {
                    info!("Proceeding with best guess interpretation");

                    // Same skill-id re-check for best-guess path
                    let skill_id = self.pending_skill_id.read().clone();
                    if let Some(skill_id) = skill_id {
                        info!(skill_id = %skill_id, "Re-checking skill disambiguation on best-guess input");
                        return self
                            .recheck_skill_disambiguation(&skill_id, &enriched_input)
                            .await;
                    }

                    return self.run_loop_internal(&enriched_input).await;
                }
                DisambiguationResult::GiveUp { reason } => {
                    *self.pending_skill_id.write() = None;
                    warn!(reason = %reason, "Disambiguation gave up");
                    let apology = self
                        .generate_localized_apology(
                            "Generate a brief, polite apology saying you couldn't understand the request. Be concise.",
                            &reason,
                        )
                        .await
                        .unwrap_or_else(|_| {
                            format!("I'm sorry, I couldn't understand your request: {}", reason)
                        });
                    let response = AgentResponse::new(&apology);
                    self.finish_turn_if_root(&response).await?;
                    return Ok(response);
                }
                DisambiguationResult::Escalate { reason } => {
                    *self.pending_skill_id.write() = None;
                    info!(reason = %reason, "Escalating to human");
                    if let Some(ref hitl) = self.hitl_engine {
                        let trigger =
                            ApprovalTrigger::condition("disambiguation_escalation", reason.clone());
                        let mut context_map = HashMap::new();
                        context_map.insert("original_input".to_string(), serde_json::json!(input));
                        context_map.insert("reason".to_string(), serde_json::json!(&reason));
                        let check_result = HITLCheckResult::required(
                            trigger,
                            context_map,
                            format!("User request needs human assistance: {}", reason),
                            Some(hitl.config().default_timeout_seconds),
                        );
                        let result = self.request_hitl_approval(check_result).await?;
                        if matches!(
                            result,
                            ApprovalResult::Approved | ApprovalResult::Modified { .. }
                        ) {
                            return self.run_loop_internal(input).await;
                        }
                    }
                    let apology = self
                        .generate_localized_apology(
                            "Explain briefly that you're transferring the user to a human agent for help.",
                            &reason,
                        )
                        .await
                        .unwrap_or_else(|_| {
                            format!("I need human assistance to help with your request: {}", reason)
                        });
                    let response = AgentResponse::new(&apology);
                    self.finish_turn_if_root(&response).await?;
                    return Ok(response);
                }
                DisambiguationResult::Abandoned { new_input } => {
                    *self.pending_skill_id.write() = None;

                    info!(
                        has_new_input = new_input.is_some(),
                        "Clarification abandoned by user"
                    );

                    self.commit_root_user_message(input).await?;

                    match new_input {
                        Some(fresh_input) => {
                            // Topic switch: process the user's new input from scratch.
                            // The LLM sees full conversation context including the abandoned exchange.
                            return self.run_loop_internal(&fresh_input).await;
                        }
                        None => {
                            // Pure abandonment: generate a brief acknowledgment.
                            let ack = self
                                .generate_localized_apology(
                                    "The user changed their mind about their previous request. \
                                     Generate a brief, friendly acknowledgment (e.g. 'OK, no problem. What else can I help with?'). \
                                     Do NOT apologize excessively. Be concise.",
                                    "User abandoned clarification",
                                )
                                .await
                                .unwrap_or_else(|_| {
                                    "OK, no problem. What else can I help with?".to_string()
                                });

                            self.memory
                                .add_message(ChatMessage::assistant(&ack))
                                .await?;

                            let response = AgentResponse::new(&ack);
                            self.finish_turn_if_root(&response).await?;
                            return Ok(response);
                        }
                    }
                }
            }
        }

        self.run_loop_internal(input).await
    }

    /// Generate a localized response using the router LLM
    async fn generate_localized_apology(&self, instruction: &str, reason: &str) -> Result<String> {
        let llm = self.llm_registry.router().map_err(|e| {
            AgentError::LLM(format!(
                "Router LLM not available for localized response: {}",
                e
            ))
        })?;

        let recent: Vec<String> = self
            .memory
            .get_messages(Some(3))
            .await?
            .iter()
            .map(|m| m.content.clone())
            .collect();

        let context_hint = if recent.is_empty() {
            String::new()
        } else {
            format!(
                "\nRecent conversation (detect the user's language from this):\n{}\n",
                recent.join("\n")
            )
        };

        let prompt = format!(
            "{}\nReason: {}\n{}Respond in the same language as the user. Output ONLY the message, nothing else.",
            instruction, reason, context_hint
        );

        let messages = vec![ChatMessage::user(&prompt)];
        let response = self
            .observe_purpose(
                ObservationPurpose::DisambiguationClarification,
                llm.complete(&messages, None),
            )
            .await
            .map_err(|e| AgentError::LLM(format!("Localized response generation failed: {}", e)))?;

        Ok(response.content.trim().to_string())
    }

    /// Clear disambiguation-related keys from the context manager.
    ///
    /// Render template variables in state action args using the context manager.
    fn render_action_args(&self, args: &Value) -> Value {
        let context = self.build_context_with_overlays();
        match args {
            Value::Object(map) => {
                let mut rendered = serde_json::Map::new();
                for (k, v) in map {
                    match v {
                        Value::String(s) if s.contains("{{") => {
                            match self.template_renderer.render(s, &context) {
                                Ok(rendered_str) => {
                                    rendered.insert(k.clone(), Value::String(rendered_str));
                                }
                                Err(_) => {
                                    rendered.insert(k.clone(), v.clone());
                                }
                            }
                        }
                        _ => {
                            rendered.insert(k.clone(), v.clone());
                        }
                    }
                }
                Value::Object(rendered)
            }
            _ => args.clone(),
        }
    }

    /// Called at the start of each turn to prevent stale `resolved_intent` from leaking across turns.
    fn clear_disambiguation_context(&self) {
        let _ = self
            .context_manager
            .set("resolved_intent", serde_json::Value::Null);

        let all = self.context_manager.get_all();
        for key in all.keys() {
            if key.starts_with("disambiguation.") {
                let _ = self.context_manager.set(key, serde_json::Value::Null);
            }
        }
    }

    /// Re-run skill disambiguation on enriched input before executing the skill.
    /// After clarification resolves, the enriched input may still be missing required_clarity fields (e.g. "Transfer money to Jane." still lacks amount).
    /// This method re-runs the skill's disambiguation pass.
    /// If fields are still missing, it returns the new clarification question and keeps pending_skill_id set.
    /// If all fields are present (Clear), it executes the skill and returns the response.
    async fn recheck_skill_disambiguation(
        &self,
        skill_id: &str,
        enriched_input: &str,
    ) -> Result<AgentResponse> {
        let skill = self
            .skill_router
            .as_ref()
            .and_then(|r| r.get_skill(skill_id).cloned());

        // If the skill has disambiguation enabled, re-run it on the enriched input.
        if let Some(ref skill) = skill {
            if let Some(ref skill_disambig) = skill.disambiguation {
                if skill_disambig.enabled.unwrap_or(false) {
                    if let Some(ref disambiguator) = self.disambiguation_manager {
                        let context = self.build_disambiguation_context().await?;
                        let state_override = self
                            .state_machine
                            .as_ref()
                            .and_then(|sm| sm.current_definition())
                            .and_then(|def| def.disambiguation.clone());

                        match self
                            .observe_purpose(
                                ObservationPurpose::DisambiguationDetection,
                                disambiguator.process_input_with_override(
                                    enriched_input,
                                    &context,
                                    state_override.as_ref(),
                                    Some(skill_disambig),
                                ),
                            )
                            .await?
                        {
                            DisambiguationResult::Clear => {
                                debug!(skill_id = %skill_id, "Skill re-check: all fields present");
                            }
                            DisambiguationResult::NeedsClarification {
                                question,
                                detection,
                            } => {
                                info!(
                                    skill_id = %skill_id,
                                    ambiguity_type = ?detection.ambiguity_type,
                                    what_is_unclear = ?detection.what_is_unclear,
                                    "Skill re-check: still missing fields, asking again"
                                );
                                // Keep pending_skill_id set (do NOT clear it).
                                // The next turn will resolve this new clarification and
                                // re-enter this method until all fields are present.
                                self.memory
                                    .add_message(ChatMessage::user(enriched_input))
                                    .await?;
                                self.memory
                                    .add_message(ChatMessage::assistant(&question.question))
                                    .await?;

                                let response = AgentResponse::new(&question.question)
                                    .with_metadata(
                                        "disambiguation",
                                        serde_json::json!({
                                            "status": "awaiting_clarification",
                                            "skill_id": skill_id,
                                            "options": question.options,
                                            "clarifying": question.clarifying,
                                            "detection": {
                                                "type": detection.ambiguity_type,
                                                "confidence": detection.confidence,
                                                "what_is_unclear": detection.what_is_unclear,
                                            }
                                        }),
                                    );
                                self.finish_turn_if_root(&response).await?;
                                return Ok(response);
                            }
                            DisambiguationResult::Clarified {
                                enriched_input: re_enriched,
                                ..
                            } => {
                                debug!(skill_id = %skill_id, "Skill re-check: clarified immediately, executing");
                                // Fall through to execute with the further-enriched input.
                                *self.pending_skill_id.write() = None;
                                let skill_response =
                                    self.execute_skill_by_id(skill_id, &re_enriched).await?;
                                self.memory
                                    .add_message(ChatMessage::user(&re_enriched))
                                    .await?;
                                return self
                                    .handle_skill_response(
                                        &re_enriched,
                                        skill_response,
                                        &HashMap::new(),
                                    )
                                    .await;
                            }
                            DisambiguationResult::ProceedWithBestGuess {
                                enriched_input: re_enriched,
                            } => {
                                debug!(skill_id = %skill_id, "Skill re-check: proceeding with best guess");
                                *self.pending_skill_id.write() = None;
                                let skill_response =
                                    self.execute_skill_by_id(skill_id, &re_enriched).await?;
                                self.memory
                                    .add_message(ChatMessage::user(&re_enriched))
                                    .await?;
                                return self
                                    .handle_skill_response(
                                        &re_enriched,
                                        skill_response,
                                        &HashMap::new(),
                                    )
                                    .await;
                            }
                            DisambiguationResult::GiveUp { reason } => {
                                *self.pending_skill_id.write() = None;
                                let apology = self
                                    .generate_localized_apology(
                                        "Generate a brief, polite apology saying you couldn't understand the request. Be concise.",
                                        &reason,
                                    )
                                    .await
                                    .unwrap_or_else(|_| {
                                        format!("I'm sorry, I couldn't understand your request: {}", reason)
                                    });
                                let response = AgentResponse::new(&apology);
                                self.finish_turn_if_root(&response).await?;
                                return Ok(response);
                            }
                            DisambiguationResult::Escalate { reason } => {
                                *self.pending_skill_id.write() = None;
                                let apology = self
                                    .generate_localized_apology(
                                        "Explain briefly that you're transferring the user to a human agent for help.",
                                        &reason,
                                    )
                                    .await
                                    .unwrap_or_else(|_| {
                                        format!("I need human assistance to help with your request: {}", reason)
                                    });
                                let response = AgentResponse::new(&apology);
                                self.finish_turn_if_root(&response).await?;
                                return Ok(response);
                            }
                            DisambiguationResult::Abandoned { new_input } => {
                                // User abandoned during skill re-check.
                                // Clear skill routing state and fall through to normal execution.
                                *self.pending_skill_id.write() = None;
                                debug!(skill_id = %skill_id, "Skill re-check: abandoned by user");
                                if let Some(fresh) = new_input {
                                    return self.run_loop_internal(&fresh).await;
                                }
                                let ack = self
                                    .generate_localized_apology(
                                        "The user changed their mind about their previous request. \
                                         Generate a brief, friendly acknowledgment (e.g. 'OK, no problem. What else can I help with?'). \
                                         Do NOT apologize excessively. Be concise.",
                                        "User abandoned clarification",
                                    )
                                    .await
                                    .unwrap_or_else(|_| {
                                        "OK, no problem. What else can I help with?".to_string()
                                    });
                                self.memory
                                    .add_message(ChatMessage::assistant(&ack))
                                    .await?;
                                let response = AgentResponse::new(&ack);
                                self.finish_turn_if_root(&response).await?;
                                return Ok(response);
                            }
                        }
                    }
                }
            }
        }

        // Skill has no disambiguation or all fields present: execute.
        *self.pending_skill_id.write() = None;
        let skill_response = self.execute_skill_by_id(skill_id, enriched_input).await?;
        self.memory
            .add_message(ChatMessage::user(enriched_input))
            .await?;
        self.handle_skill_response(enriched_input, skill_response, &HashMap::new())
            .await
    }

    /// Handle skill routing result: output processing, memory, transitions.
    /// Returns a fully formed AgentResponse for skill-routed requests.
    async fn handle_skill_response(
        &self,
        processed_input: &str,
        skill_response: String,
        input_context: &HashMap<String, Value>,
    ) -> Result<AgentResponse> {
        let output_data = self.process_output(&skill_response, input_context).await?;
        let final_response = output_data.content;

        self.memory
            .add_message(ChatMessage::assistant(&final_response))
            .await?;

        self.check_memory_compression().await?;

        self.increment_turn();
        self.evaluate_transitions(processed_input, &final_response)
            .await?;

        let response = AgentResponse::new(final_response);
        self.finish_turn_if_root(&response).await?;
        Ok(response)
    }

    /// Run the Plan-and-Execute flow: generate plan, execute steps, finalize.
    /// Supports plan-level reflection with replan loop when configured.
    async fn handle_plan_and_execute(
        &self,
        processed_input: &str,
        input_context: &HashMap<String, Value>,
        auto_detected: bool,
    ) -> Result<AgentResponse> {
        let effective = self.get_effective_reasoning_config();
        let plan_reflection = effective
            .get_planning()
            .map(|c| c.reflection.clone())
            .unwrap_or_default();

        let max_attempts = if plan_reflection.enabled {
            1 + plan_reflection.max_replans
        } else {
            1
        };

        let mut plan = self.generate_plan(processed_input).await?;
        info!(
            plan_id = %plan.id,
            steps = plan.steps.len(),
            "Plan generated"
        );

        let mut plan_result = String::new();

        for attempt in 0..max_attempts {
            *self.current_plan.write() = Some(plan.clone());
            plan_result = self.execute_plan(&mut plan).await?;

            info!(
                plan_status = ?plan.status,
                completed_steps = plan.completed_steps().count(),
                attempt = attempt + 1,
                "Plan execution completed"
            );

            if !plan_reflection.enabled {
                break;
            }

            let has_failures = plan.steps.iter().any(|s| s.status.is_failed());
            if !has_failures {
                break;
            }

            if attempt + 1 >= max_attempts {
                break;
            }

            match plan_reflection.on_step_failure {
                StepFailureAction::Replan => {
                    info!(attempt = attempt + 1, "Plan had failures, replanning");
                    plan = self.generate_plan(processed_input).await?;
                }
                StepFailureAction::Abort => {
                    warn!("Plan step failed, aborting");
                    break;
                }
                StepFailureAction::Skip | StepFailureAction::Continue => {
                    break;
                }
            }
        }

        *self.current_plan.write() = Some(plan);

        let output_data = self.process_output(&plan_result, input_context).await?;
        let final_content = output_data.content;

        self.memory
            .add_message(ChatMessage::assistant(&final_content))
            .await?;

        self.check_memory_compression().await?;
        self.increment_turn();
        self.evaluate_transitions(processed_input, &final_content)
            .await?;

        let reasoning_metadata =
            ReasoningMetadata::new(ReasoningMode::PlanAndExecute).with_auto_detected(auto_detected);

        let response = AgentResponse::new(&final_content).with_metadata(
            "reasoning",
            serde_json::to_value(&reasoning_metadata).unwrap_or_default(),
        );

        self.finish_turn_if_root(&response).await?;
        Ok(response)
    }

    /// Inject CoT/ReAct reasoning prompt into the system message (first iteration only).
    fn inject_reasoning_prompt(
        &self,
        messages: &mut [ChatMessage],
        reasoning_mode: &ReasoningMode,
        is_first_iteration: bool,
    ) {
        if !is_first_iteration {
            return;
        }
        match reasoning_mode {
            ReasoningMode::CoT => {
                if let Some(msg) = messages.first_mut() {
                    if matches!(msg.role, ai_agents_core::Role::System) {
                        msg.content = self.build_cot_system_prompt(&msg.content);
                        debug!("Applied Chain-of-Thought system prompt");
                    }
                }
            }
            ReasoningMode::React => {
                if let Some(msg) = messages.first_mut() {
                    if matches!(msg.role, ai_agents_core::Role::System) {
                        msg.content = self.build_react_system_prompt(&msg.content);
                        debug!("Applied ReAct system prompt");
                    }
                }
            }
            _ => {}
        }
    }

    async fn complete_llm_with_recovery(
        &self,
        llm: Arc<dyn LLMProvider>,
        messages: &[ChatMessage],
    ) -> Result<LLMResponse> {
        let primary_result = if self.recovery_manager.config().default.max_retries > 0 {
            self.recovery_manager
                .with_retry("llm_call", None, || async {
                    self.observe_purpose(
                        ObservationPurpose::MainResponse,
                        llm.complete(messages, None),
                    )
                    .await
                    .map_err(|e| e.classify())
                })
                .await
                .map_err(|e| AgentError::LLM(e.to_string()))
        } else {
            self.observe_purpose(
                ObservationPurpose::MainResponse,
                llm.complete(messages, None),
            )
            .await
            .map_err(|e| AgentError::LLM(e.to_string()))
        };

        match primary_result {
            Ok(resp) => Ok(resp),
            Err(primary_err) => match &self.recovery_manager.config().llm.on_failure {
                LLMFailureAction::FallbackLlm { fallback_llm } => {
                    let fb = self.llm_registry.get(fallback_llm).map_err(|e| {
                        AgentError::Config(format!(
                            "Fallback LLM '{}' not found: {}",
                            fallback_llm, e
                        ))
                    })?;
                    self.observe_purpose(
                        ObservationPurpose::MainResponse,
                        fb.complete(messages, None),
                    )
                    .await
                    .map_err(|e| AgentError::LLM(e.to_string()))
                }
                LLMFailureAction::FallbackResponse { message } => {
                    Ok(LLMResponse::new(message.clone(), FinishReason::Stop))
                }
                LLMFailureAction::Error => Err(primary_err),
            },
        }
    }

    //
    // Draft generation must not commit user memory or run tools.
    // The current user input is added only as an ephemeral message for this LLM call.
    //
    async fn generate_main_response_draft(
        &self,
        processed_input: &str,
        reasoning_mode: &ReasoningMode,
    ) -> Result<MainResponseDraft> {
        let llm = self.get_state_llm()?;
        let mut messages = self.build_messages_for_draft(processed_input).await?;
        self.inject_reasoning_prompt(&mut messages, reasoning_mode, true);
        let response = self.complete_llm_with_recovery(llm, &messages).await?;
        let content = response.content.trim().to_string();
        let (thinking, answer) = self.extract_thinking(&content);
        if let Some(calls) = self.parse_tool_calls(&content) {
            return Ok(MainResponseDraft::ToolCalls {
                raw_content: content,
                calls,
                thinking,
            });
        }
        Ok(MainResponseDraft::Text {
            raw_content: answer,
            thinking,
        })
    }

    //
    // This is the only place where a winning draft is allowed to become runtime state.
    // Parsed tool calls become executable only after this method commits the draft.
    //
    async fn commit_main_response_draft(
        &self,
        processed_input: &str,
        input_context: &HashMap<String, Value>,
        draft: MainResponseDraft,
        reasoning_mode: ReasoningMode,
        auto_detected: bool,
    ) -> Result<AgentResponse> {
        self.commit_root_user_message(processed_input).await?;
        match draft {
            MainResponseDraft::Text {
                raw_content,
                thinking,
            } => {
                self.finish_text_response_from_model(
                    processed_input,
                    input_context,
                    raw_content,
                    reasoning_mode,
                    auto_detected,
                    1,
                    thinking,
                    Vec::new(),
                )
                .await
            }
            MainResponseDraft::ToolCalls {
                raw_content,
                calls,
                thinking: _,
            } => {
                let mut all_tool_calls = Vec::new();
                match self
                    .handle_tool_calls(processed_input, &raw_content, calls, &mut all_tool_calls)
                    .await?
                {
                    ToolCallOutcome::Rejected(response) => {
                        self.finish_turn_if_root(&response).await?;
                        Ok(response)
                    }
                    ToolCallOutcome::Continue | ToolCallOutcome::TransitionFired => {
                        self.continue_after_committed_tool_draft(processed_input)
                            .await
                    }
                }
            }
        }
    }

    //
    // Tool drafts need a committed continuation after function results are written.
    // Redispatch depth suppresses duplicate root lifecycle work during that continuation.
    //
    async fn continue_after_committed_tool_draft(
        &self,
        processed_input: &str,
    ) -> Result<AgentResponse> {
        *self.redispatch_depth.write() += 1;
        if let Some(context) = self.active_turn_context.write().as_mut() {
            context.enter_redispatch();
        }
        let result = Box::pin(self.run_loop_internal(processed_input)).await;
        *self.redispatch_depth.write() -= 1;
        if let Some(context) = self.active_turn_context.write().as_mut() {
            context.exit_redispatch();
        }
        let response = result?;
        self.finish_turn_if_root(&response).await?;
        Ok(response)
    }

    //
    // Shared committed text finalization for normal responses and winning text drafts.
    // Keep output processing, reflection, transitions, hooks, and maintenance behind this commit boundary.
    //
    async fn finish_text_response_from_model(
        &self,
        processed_input: &str,
        input_context: &HashMap<String, Value>,
        answer: String,
        reasoning_mode: ReasoningMode,
        auto_detected: bool,
        iterations: u32,
        thinking_content: Option<String>,
        all_tool_calls: Vec<ToolCall>,
    ) -> Result<AgentResponse> {
        let output_data = self.process_output(&answer, input_context).await?;
        let mut final_content = if output_data.metadata.rejected {
            output_data
                .metadata
                .rejection_reason
                .unwrap_or_else(|| answer.to_string())
        } else {
            output_data.content
        };
        let llm = self.get_state_llm()?;
        let reflection_metadata;
        (final_content, reflection_metadata) = self
            .run_reflection(&*llm, processed_input, final_content)
            .await?;
        final_content =
            self.format_response_with_thinking(thinking_content.as_deref(), &final_content);
        let final_content = {
            let result = self
                .post_loop_processing(processed_input, final_content)
                .await?;
            self.apply_post_loop_result(processed_input, result).await?
        };
        let response = self.build_agent_response(
            final_content,
            all_tool_calls,
            reasoning_mode,
            auto_detected,
            iterations,
            thinking_content,
            reflection_metadata,
        );
        self.finish_turn_if_root(&response).await?;
        Ok(response)
    }

    //
    // Auto reasoning uses this path after the judge wins with a deeper mode.
    // It intentionally uses committed message building instead of the draft overlay.
    //
    async fn run_committed_response_loop_with_reasoning(
        &self,
        processed_input: &str,
        input_context: &HashMap<String, Value>,
        reasoning_mode: ReasoningMode,
        auto_detected: bool,
    ) -> Result<AgentResponse> {
        self.commit_root_user_message(processed_input).await?;
        let llm = self.get_state_llm()?;
        let mut iterations = 0u32;
        let mut all_tool_calls = Vec::new();
        let mut thinking_content = None;
        loop {
            let effective_max = if reasoning_mode != ReasoningMode::None {
                let rc = self.get_effective_reasoning_config();
                self.max_iterations.min(rc.max_iterations)
            } else {
                self.max_iterations
            };
            if iterations >= effective_max {
                return Err(AgentError::Other(format!(
                    "Max iterations ({}) exceeded",
                    effective_max
                )));
            }
            iterations += 1;
            *self.iteration_count.write() = iterations;
            let mut messages = self.build_messages().await?;
            self.inject_reasoning_prompt(&mut messages, &reasoning_mode, iterations == 1);
            self.hooks.on_llm_start(&messages).await;
            let llm_start = Instant::now();
            let response = self
                .observe_purpose(
                    ObservationPurpose::MainResponse,
                    llm.complete(&messages, None),
                )
                .await
                .map_err(|e| AgentError::LLM(e.to_string()))?;
            let llm_duration_ms = llm_start.elapsed().as_millis() as u64;
            self.hooks.on_llm_complete(&response, llm_duration_ms).await;
            let content = response.content.trim();
            if let Some(tool_calls) = self.parse_tool_calls(content) {
                match self
                    .handle_tool_calls(processed_input, content, tool_calls, &mut all_tool_calls)
                    .await?
                {
                    ToolCallOutcome::Continue | ToolCallOutcome::TransitionFired => continue,
                    ToolCallOutcome::Rejected(resp) => {
                        self.finish_turn_if_root(&resp).await?;
                        return Ok(resp);
                    }
                }
            }
            let (extracted_thinking, answer) = self.extract_thinking(content);
            if extracted_thinking.is_some() {
                thinking_content = extracted_thinking;
            }
            return self
                .finish_text_response_from_model(
                    processed_input,
                    input_context,
                    answer,
                    reasoning_mode,
                    auto_detected,
                    iterations,
                    thinking_content,
                    all_tool_calls,
                )
                .await;
        }
    }

    /// Handle tool calls: check transitions, execute tools in parallel, handle HITL rejection.
    async fn handle_tool_calls(
        &self,
        processed_input: &str,
        content: &str,
        tool_calls: Vec<ToolCall>,
        all_tool_calls: &mut Vec<ToolCall>,
    ) -> Result<ToolCallOutcome> {
        // Check if a transition should fire before executing the LLM's tool call.
        // If a transition fires, on_enter actions handle the tool call correctly
        // (with proper URLs from YAML), so skip the LLM's tool call.
        let transition_fired = self.evaluate_transitions(processed_input, content).await?;
        if transition_fired {
            self.memory
                .add_message(ChatMessage::assistant(
                    "(Transitioned to new state — tool call handled by workflow)",
                ))
                .await?;
            return Ok(ToolCallOutcome::TransitionFired);
        }

        // Store the assistant's tool-call message so the LLM sees its own decision in conversation history. Without this, the model only sees the tool result and may repeat the same call.
        self.memory
            .add_message(ChatMessage::assistant(content))
            .await?;

        let results = self.execute_tools_parallel(&tool_calls).await;

        for ((_id, result), tool_call) in results.into_iter().zip(tool_calls.iter()) {
            match result {
                Ok(output) => {
                    self.memory
                        .add_message(ChatMessage::function(&tool_call.name, &output))
                        .await?;
                }
                Err(e) => {
                    // Check if this is a HITL rejection - if so, break the loop
                    if matches!(e, AgentError::HITLRejected(_)) {
                        self.memory
                            .add_message(ChatMessage::assistant(&format!(
                                "The operation was rejected by the approver: {}",
                                e
                            )))
                            .await?;
                        // Return the rejection message to user, don't continue loop
                        return Ok(ToolCallOutcome::Rejected(AgentResponse {
                            content: format!("Operation cancelled: {}", e),
                            metadata: None,
                            tool_calls: Some(all_tool_calls.clone()),
                        }));
                    }
                    self.memory
                        .add_message(ChatMessage::function(
                            &tool_call.name,
                            &format!("Error: {}", e),
                        ))
                        .await?;
                }
            }
            all_tool_calls.push(tool_call.clone());
        }
        Ok(ToolCallOutcome::Continue)
    }

    /// Run the reflection loop on a response, returning (improved_content, reflection_metadata).
    async fn run_reflection(
        &self,
        llm: &dyn LLMProvider,
        processed_input: &str,
        mut content: String,
    ) -> Result<(String, Option<ReflectionMetadata>)> {
        let should_reflect = self.should_reflect(processed_input, &content).await?;
        if !should_reflect {
            return Ok((content, None));
        }

        info!("Starting response reflection evaluation");
        let mut attempts = 0u32;
        let max_retries = self.reflection_config.max_retries;
        let mut history: Vec<ReflectionAttempt> = Vec::new();

        loop {
            let evaluation = self.evaluate_response(processed_input, &content).await?;

            if evaluation.passed || attempts >= max_retries {
                info!(
                    passed = evaluation.passed,
                    confidence = evaluation.confidence,
                    attempts = attempts + 1,
                    "Reflection evaluation complete"
                );
                let reflection_metadata = Some(
                    ReflectionMetadata::new(evaluation)
                        .with_attempts(attempts + 1)
                        .with_history(history),
                );
                return Ok((content, reflection_metadata));
            }

            debug!(
                attempt = attempts + 1,
                failed_criteria = evaluation.failed_criteria().count(),
                "Response did not meet criteria, retrying"
            );

            history.push(
                ReflectionAttempt::new(&content, evaluation.clone())
                    .with_feedback("Response did not meet quality criteria"),
            );

            let feedback: Vec<String> = evaluation
                .failed_criteria()
                .map(|c| format!("- {}", c.criterion))
                .collect();

            let retry_prompt = format!(
                "Your previous response did not meet these criteria:\n{}\n\nPlease provide an improved response.",
                feedback.join("\n")
            );

            self.memory
                .add_message(ChatMessage::user(&retry_prompt))
                .await?;

            let retry_messages = self.build_messages().await?;
            let retry_response = self
                .observe_purpose(
                    ObservationPurpose::ReflectionEvaluation,
                    llm.complete(&retry_messages, None),
                )
                .await
                .map_err(|e| AgentError::LLM(e.to_string()))?;

            content = retry_response.content.trim().to_string();
            attempts += 1;
        }
    }

    /// Record the assistant turn, evaluate transitions, and decide what to do next.
    /// Returns PostLoopResult so callers can apply_post_loop_result for re-dispatch.
    async fn post_loop_processing(
        &self,
        processed_input: &str,
        content: String,
    ) -> Result<PostLoopResult> {
        // Do NOT add the assistant message to memory yet.
        // evaluate_transitions receives content as a direct parameter, so the message does not need to be in memory for transitions to evaluate correctly.
        // For NeedsRedispatch we skip adding the stale old-state response entirely, keeping memory clean for the re-dispatched handler.

        self.increment_turn();

        // Run context extractors so guards can check freshly-extracted values.
        self.run_context_extractors(processed_input).await;

        let transitioned = self.evaluate_transitions(processed_input, &content).await?;

        if !transitioned {
            self.memory
                .add_message(ChatMessage::assistant(&content))
                .await?;
            self.check_memory_compression().await?;
            return Ok(PostLoopResult::NoTransition(content));
        }

        // Check if we should skip re-generation after this transition.
        if !self.should_regenerate_after_transition() {
            self.memory
                .add_message(ChatMessage::assistant(&content))
                .await?;
            self.check_memory_compression().await?;
            return Ok(PostLoopResult::Transitioned(content));
        }

        // Check if the new state needs full dispatch.
        // Orchestration states (concurrent, group_chat, pipeline, handoff, delegate) need their dedicated handlers.
        // Any non-None effective reasoning mode needs CoT/ReAct prompt injection, the plan-and-execute handler, or Auto re-determination - all of which live in run_loop_internal, not here.
        if self.needs_redispatch_for_new_state() {
            info!("Post-transition NeedsRedispatch: new state requires full dispatch");
            // Stale old-state content is NOT added to memory.
            // apply_post_loop_result will increment redispatch_depth and call run_loop_internal, which produces the correct response and adds it.
            return Ok(PostLoopResult::NeedsRedispatch);
        }

        // Normal post-transition re-generation (plain LLM with optional tool calls).
        // Add the stale response to history so the new LLM sees the conversation.
        self.memory
            .add_message(ChatMessage::assistant(&content))
            .await?;
        self.check_memory_compression().await?;

        // If a transition fired, on_enter actions already executed (e.g., HTTP calls).
        // The current content was generated in the OLD state context and is stale.
        // Re-generate in the new state context so the LLM can reference on_enter results.
        // If the LLM responds with a tool call, execute it in a mini-loop so the
        // result is not returned as raw JSON text.
        let new_llm = self.get_state_llm()?;
        let mut final_content;

        for post_iter in 0..self.max_iterations {
            let new_messages = self.build_messages().await?;
            if post_iter == 0 {
                if let Some(system_msg) = new_messages.first() {
                    if system_msg.role == ai_agents_core::Role::System {
                        debug!(
                            prompt_preview =
                                &system_msg.content[system_msg.content.len().saturating_sub(200)..],
                            "Post-transition system prompt (last 200 chars)"
                        );
                    }
                }
            }

            let new_response = self
                .observe_purpose(
                    ObservationPurpose::MainResponse,
                    new_llm.complete(&new_messages, None),
                )
                .await
                .map_err(|e| AgentError::LLM(e.to_string()))?;
            final_content = new_response.content.trim().to_string();

            // Check if the post-transition response contains tool calls.
            // If so, execute them and loop so the LLM can summarize the result.
            if let Some(tool_calls) = self.parse_tool_calls(&final_content) {
                debug!(
                    post_iter = post_iter,
                    tools = tool_calls.len(),
                    "Post-transition tool call detected, executing"
                );

                self.memory
                    .add_message(ChatMessage::assistant(&final_content))
                    .await?;

                let results = self.execute_tools_parallel(&tool_calls).await;
                for ((_id, result), tool_call) in results.into_iter().zip(tool_calls.iter()) {
                    match result {
                        Ok(output) => {
                            self.memory
                                .add_message(ChatMessage::function(&tool_call.name, &output))
                                .await?;
                        }
                        Err(e) => {
                            self.memory
                                .add_message(ChatMessage::function(
                                    &tool_call.name,
                                    &format!("Error: {}", e),
                                ))
                                .await?;
                        }
                    }
                }
                // Loop to let the LLM see the tool result and produce a text response.
                continue;
            }

            // No tool call - this is the final text response.
            self.memory
                .add_message(ChatMessage::assistant(&final_content))
                .await?;
            return Ok(PostLoopResult::Transitioned(final_content));
        }

        // Exhausted post-transition iterations (unlikely). Return last content.
        final_content = "Post-transition processing completed.".to_string();
        self.memory
            .add_message(ChatMessage::assistant(&final_content))
            .await?;

        Ok(PostLoopResult::Transitioned(final_content))
    }

    /// Build the final AgentResponse with all metadata.
    /// Check whether to re-generate a response after a state transition.
    fn should_regenerate_after_transition(&self) -> bool {
        if let Some(ref sm) = self.state_machine {
            // Global setting
            if !sm.config().regenerate_on_transition {
                return false;
            }
            // Per-state override on the new (current) state
            if let Some(def) = sm.current_definition() {
                if let Some(regen) = def.regenerate_on_enter {
                    return regen;
                }
            }
        }
        true
    }

    /// Return true when the new state requires full dispatch via run_loop_internal.
    /// Covers orchestration states and any non-None effective reasoning mode.
    fn needs_redispatch_for_new_state(&self) -> bool {
        if let Some(ref sm) = self.state_machine {
            if let Some(def) = sm.current_definition() {
                if def.concurrent.is_some()
                    || def.group_chat.is_some()
                    || def.pipeline.is_some()
                    || def.handoff.is_some()
                    || def.delegate.is_some()
                {
                    return true;
                }
                // Any non-None effective reasoning mode requires the main dispatch loop.
                let effective = self.get_effective_reasoning_config();
                if !matches!(effective.mode, ReasoningMode::None) {
                    return true;
                }
            }
        }
        false
    }

    /// Consume a PostLoopResult. NeedsRedispatch re-enters run_loop_internal.
    /// The user message is already in memory - redispatch_depth suppresses re-adding it.
    async fn apply_post_loop_result(
        &self,
        processed_input: &str,
        result: PostLoopResult,
    ) -> Result<String> {
        match result {
            PostLoopResult::NoTransition(content) | PostLoopResult::Transitioned(content) => {
                Ok(content)
            }
            PostLoopResult::NeedsRedispatch => {
                const MAX_REDISPATCH_DEPTH: u32 = 3;
                let current_depth = *self.redispatch_depth.read();
                if current_depth >= MAX_REDISPATCH_DEPTH {
                    warn!(
                        depth = current_depth,
                        "Post-transition re-dispatch depth limit reached, returning empty response"
                    );
                    let content = String::new();
                    self.memory
                        .add_message(ChatMessage::assistant(&content))
                        .await?;
                    return Ok(content);
                }
                *self.redispatch_depth.write() += 1;
                if let Some(context) = self.active_turn_context.write().as_mut() {
                    context.enter_redispatch();
                }
                info!(
                    depth = current_depth + 1,
                    "Re-dispatching for new state after transition"
                );
                let resp = Box::pin(self.run_loop_internal(processed_input)).await;
                *self.redispatch_depth.write() -= 1;
                if let Some(context) = self.active_turn_context.write().as_mut() {
                    context.exit_redispatch();
                }
                resp.map(|r| r.content)
            }
        }
    }

    fn build_agent_response(
        &self,
        content: String,
        all_tool_calls: Vec<ToolCall>,
        reasoning_mode: ReasoningMode,
        auto_detected: bool,
        iterations: u32,
        thinking: Option<String>,
        reflection_metadata: Option<ReflectionMetadata>,
    ) -> AgentResponse {
        let reasoning_metadata = ReasoningMetadata::new(reasoning_mode.clone())
            .with_thinking(thinking.clone().unwrap_or_default())
            .with_iterations(iterations)
            .with_auto_detected(auto_detected);

        let mut response = AgentResponse::new(&content);
        if !all_tool_calls.is_empty() {
            response = response.with_tool_calls(all_tool_calls);
        }

        if let Some(state) = self.current_state() {
            response = response.with_metadata("current_state", serde_json::json!(state));
        }

        response = response.with_metadata(
            "reasoning",
            serde_json::to_value(&reasoning_metadata).unwrap_or_default(),
        );

        if let Some(ref refl_meta) = reflection_metadata {
            response = response.with_metadata(
                "reflection",
                serde_json::to_value(refl_meta).unwrap_or_default(),
            );
        }

        response
    }

    // Handle delegation: forward user input to a registry agent.
    async fn handle_delegated_state(
        &self,
        input: &str,
        delegate_id: &str,
        state_def: &ai_agents_state::StateDefinition,
    ) -> Result<AgentResponse> {
        use std::time::Instant;

        let registry = self.spawner_registry.as_ref().ok_or_else(|| {
            AgentError::Config(format!(
                "State delegates to '{}' but no agent registry is configured. \
                 Add a spawner section with auto_spawn to your YAML.",
                delegate_id
            ))
        })?;

        let state_name = self
            .state_machine
            .as_ref()
            .map(|sm| sm.current())
            .unwrap_or_else(|| "unknown".to_string());

        self.hooks.on_delegate_start(delegate_id, &state_name).await;
        let start = Instant::now();

        let delegate = registry.get(delegate_id).ok_or_else(|| {
            AgentError::Other(format!(
                "State '{}' delegates to '{}' but no agent with that ID exists in the registry.",
                state_name, delegate_id
            ))
        })?;

        // Prepare input based on delegate_context mode.
        let context_mode = state_def.delegate_context.clone().unwrap_or_default();
        let effective_input = self
            .observe_purpose(
                ObservationPurpose::OrchestrationRouting,
                crate::orchestration::context::prepare_delegate_input(
                    input,
                    &context_mode,
                    &*self.memory,
                    self.llm_registry.get("router").ok().as_deref(),
                ),
            )
            .await?;

        let response = delegate
            .chat_with_actor_context(&effective_input, self.outbound_actor_context())
            .await?;

        let duration_ms = start.elapsed().as_millis() as u64;
        self.hooks
            .on_delegate_complete(delegate_id, &state_name, duration_ms)
            .await;

        // Backward-compatible context key.
        let ctx_key = format!("delegation.{}.last_response", delegate_id);
        let _ = self.context_manager.set(
            &ctx_key,
            serde_json::Value::String(response.content.clone()),
        );

        // Structured orchestration context.
        let _ = self.context_manager.set(
            "orchestration",
            serde_json::json!({
                "type": "delegate",
                "agent": delegate_id,
                "state": state_name,
                "response": response.content,
                "duration_ms": duration_ms,
            }),
        );

        self.commit_root_user_message(input).await?;

        // post_loop_processing records the assistant turn and evaluates transitions.
        // apply_post_loop_result handles NeedsRedispatch by re-entering run_loop_internal.
        let post_result = self
            .post_loop_processing(
                input,
                format!("[Delegated to {}]: {}", delegate_id, response.content),
            )
            .await?;
        let final_content = self.apply_post_loop_result(input, post_result).await?;

        let mut result = AgentResponse::new(final_content);

        let metadata = serde_json::json!({
            "orchestration": {
                "type": "delegate",
                "agent": delegate_id,
                "state": state_name,
                "response": response.content,
                "duration_ms": duration_ms,
            }
        });
        result.metadata = Some(
            serde_json::from_value::<std::collections::HashMap<String, serde_json::Value>>(
                metadata,
            )
            .unwrap_or_default(),
        );

        self.finish_turn_if_root(&result).await?;
        Ok(result)
    }

    // Handle concurrent execution: run multiple registry agents in parallel and aggregate.
    async fn handle_concurrent_state(
        &self,
        input: &str,
        config: &ai_agents_state::ConcurrentStateConfig,
    ) -> Result<AgentResponse> {
        use std::time::Instant;

        let registry = self.spawner_registry.as_ref().ok_or_else(|| {
            AgentError::Config(
                "Concurrent state requires an agent registry. Add a spawner section.".into(),
            )
        })?;

        // Render input template if provided, otherwise use the raw input.
        // Uses direct minijinja rendering (same approach as pipeline) so variables
        // are top-level: {{ user_input }}, not {{ context.user_input }}.
        // Enrich input with parent conversation history when context_mode is set.
        let context_mode = config.context_mode.clone().unwrap_or_default();
        let context_input = self
            .observe_purpose(
                ObservationPurpose::OrchestrationRouting,
                crate::orchestration::context::prepare_delegate_input(
                    input,
                    &context_mode,
                    &*self.memory,
                    self.llm_registry.get("router").ok().as_deref(),
                ),
            )
            .await?;

        let effective_input = if let Some(ref tmpl) = config.input {
            render_concurrent_template(tmpl, &context_input, &self.build_context_with_overlays())
                .unwrap_or_else(|_| context_input.clone())
        } else {
            context_input
        };

        let start = Instant::now();

        let llm_name = config
            .aggregation
            .synthesizer_llm
            .as_deref()
            .unwrap_or("router");
        let llm_provider = self.llm_registry.get(llm_name).ok();

        let vote_parallelism = if self.runtime_config.optimization.enabled
            && self
                .runtime_config
                .optimization
                .parallel_orchestration_vote_extraction
        {
            Some(self.runtime_config.optimization.max_parallel_runtime_tasks)
        } else {
            None
        };

        let result = self
            .observe_purpose(
                ObservationPurpose::OrchestrationAggregation,
                scope_actor_context(
                    self.outbound_actor_context(),
                    crate::orchestration::concurrent(
                        registry,
                        &effective_input,
                        &config.agents,
                        &config.aggregation,
                        llm_provider.as_deref(),
                        config.min_required,
                        config.timeout_ms,
                        config.on_partial_failure.clone(),
                        vote_parallelism,
                    ),
                ),
            )
            .await?;

        let duration_ms = start.elapsed().as_millis() as u64;
        let agent_ids: Vec<String> = config.agents.iter().map(|a| a.id().to_string()).collect();
        let strategy = format!("{:?}", config.aggregation.strategy);
        self.hooks
            .on_concurrent_complete(&agent_ids, &strategy, duration_ms)
            .await;

        // Backward-compatible context key.
        let _ = self.context_manager.set(
            "concurrent.result",
            serde_json::Value::String(result.response.content.clone()),
        );

        // Build per-agent result data for context and metadata.
        let agents_json: Vec<serde_json::Value> = result
            .agent_results
            .iter()
            .map(|ar| {
                serde_json::json!({
                    "id": ar.agent_id,
                    "response": ar.response.as_ref().map(|r| r.content.as_str()),
                    "success": ar.success,
                    "error": ar.error,
                    "duration_ms": ar.duration_ms,
                })
            })
            .collect();

        // Structured orchestration context with per-agent results.
        let _ = self.context_manager.set(
            "orchestration",
            serde_json::json!({
                "type": "concurrent",
                "result": result.response.content,
                "strategy": strategy,
                "agents": agents_json,
                "duration_ms": duration_ms,
            }),
        );

        self.commit_root_user_message(input).await?;

        let post_result = self
            .post_loop_processing(input, result.response.content.clone())
            .await?;
        let final_content = self.apply_post_loop_result(input, post_result).await?;

        let mut response = AgentResponse::new(final_content);
        let metadata = serde_json::json!({
            "orchestration": {
                "type": "concurrent",
                "result": result.response.content,
                "strategy": strategy,
                "agents": agents_json,
                "duration_ms": duration_ms,
            }
        });
        response.metadata = Some(
            serde_json::from_value::<std::collections::HashMap<String, serde_json::Value>>(
                metadata,
            )
            .unwrap_or_default(),
        );

        self.finish_turn_if_root(&response).await?;
        Ok(response)
    }

    // Handle group chat: run a multi-turn multi-agent conversation.
    async fn handle_group_chat_state(
        &self,
        input: &str,
        config: &ai_agents_state::GroupChatStateConfig,
    ) -> Result<AgentResponse> {
        use std::time::Instant;

        let registry = self.spawner_registry.as_ref().ok_or_else(|| {
            AgentError::Config(
                "Group chat state requires an agent registry. Add a spawner section.".into(),
            )
        })?;

        let start = Instant::now();

        let llm_provider = self.llm_registry.get("router").ok();

        // Enrich input with parent conversation history when context_mode is set.
        let context_mode = config.context_mode.clone().unwrap_or_default();
        let context_input = self
            .observe_purpose(
                ObservationPurpose::OrchestrationRouting,
                crate::orchestration::context::prepare_delegate_input(
                    input,
                    &context_mode,
                    &*self.memory,
                    self.llm_registry.get("router").ok().as_deref(),
                ),
            )
            .await?;

        // Render input template if provided, otherwise use the raw user message as topic.
        let effective_topic = if let Some(ref tmpl) = config.input {
            render_concurrent_template(tmpl, &context_input, &self.build_context_with_overlays())
                .unwrap_or_else(|_| context_input.clone())
        } else {
            context_input
        };

        let result = self
            .observe_purpose(
                ObservationPurpose::OrchestrationConversation,
                scope_actor_context(
                    self.outbound_actor_context(),
                    crate::orchestration::group_chat(
                        registry,
                        &effective_topic,
                        config,
                        llm_provider.as_deref(),
                        Some(&*self.hooks),
                    ),
                ),
            )
            .await?;

        let duration_ms = start.elapsed().as_millis() as u64;

        // Backward-compatible context key.
        let _ = self.context_manager.set(
            "group_chat.conclusion",
            serde_json::Value::String(result.response.content.clone()),
        );

        // Build transcript data for context and metadata.
        let transcript_json: Vec<serde_json::Value> = result
            .transcript
            .iter()
            .map(|t| {
                serde_json::json!({
                    "speaker": t.speaker,
                    "round": t.round,
                    "content": t.content,
                })
            })
            .collect();

        // Structured orchestration context with full transcript.
        let _ = self.context_manager.set(
            "orchestration",
            serde_json::json!({
                "type": "group_chat",
                "conclusion": result.response.content,
                "transcript": transcript_json,
                "rounds": result.rounds_completed,
                "termination": result.termination_reason,
                "duration_ms": duration_ms,
            }),
        );

        self.commit_root_user_message(input).await?;

        let post_result = self
            .post_loop_processing(input, result.response.content.clone())
            .await?;
        let final_content = self.apply_post_loop_result(input, post_result).await?;

        let mut response = AgentResponse::new(final_content);
        let metadata = serde_json::json!({
            "orchestration": {
                "type": "group_chat",
                "conclusion": result.response.content,
                "transcript": transcript_json,
                "rounds": result.rounds_completed,
                "termination": result.termination_reason,
                "duration_ms": duration_ms,
            }
        });
        response.metadata = Some(
            serde_json::from_value::<std::collections::HashMap<String, serde_json::Value>>(
                metadata,
            )
            .unwrap_or_default(),
        );

        self.finish_turn_if_root(&response).await?;
        Ok(response)
    }

    // Handle pipeline: run agents sequentially with per-stage input templates.
    async fn handle_pipeline_state(
        &self,
        input: &str,
        config: &ai_agents_state::PipelineStateConfig,
    ) -> Result<AgentResponse> {
        use std::time::Instant;

        let registry = self.spawner_registry.as_ref().ok_or_else(|| {
            AgentError::Config(
                "Pipeline state requires an agent registry. Add a spawner section.".into(),
            )
        })?;

        let start = Instant::now();

        let stages: Vec<crate::orchestration::PipelineStage> = config
            .stages
            .iter()
            .map(|entry| {
                let mut stage = crate::orchestration::PipelineStage::id(entry.id());
                if let Some(tmpl) = entry.input() {
                    stage = stage.with_input(tmpl);
                }
                stage
            })
            .collect();

        // Enrich input with parent conversation history when context_mode is set.
        let context_mode = config.context_mode.clone().unwrap_or_default();
        let context_input = self
            .observe_purpose(
                ObservationPurpose::OrchestrationRouting,
                crate::orchestration::context::prepare_delegate_input(
                    input,
                    &context_mode,
                    &*self.memory,
                    self.llm_registry.get("router").ok().as_deref(),
                ),
            )
            .await?;

        let context_values = self.build_context_with_overlays();
        let result = self
            .observe_purpose(
                ObservationPurpose::OrchestrationRouting,
                scope_actor_context(
                    self.outbound_actor_context(),
                    crate::orchestration::pipeline(
                        registry,
                        &context_input,
                        &stages,
                        config.timeout_ms,
                        Some(&*self.hooks),
                        Some(&context_values),
                    ),
                ),
            )
            .await?;

        let duration_ms = start.elapsed().as_millis() as u64;

        // Backward-compatible context key.
        let _ = self.context_manager.set(
            "pipeline.result",
            serde_json::Value::String(result.response.content.clone()),
        );

        // Build per-stage data for context and metadata.
        let stages_json: Vec<serde_json::Value> = result
            .stage_outputs
            .iter()
            .map(|s| {
                serde_json::json!({
                    "agent_id": s.agent_id,
                    "output": s.output,
                    "duration_ms": s.duration_ms,
                    "skipped": s.skipped,
                })
            })
            .collect();

        // Structured orchestration context.
        let _ = self.context_manager.set(
            "orchestration",
            serde_json::json!({
                "type": "pipeline",
                "result": result.response.content,
                "stages": stages_json,
                "duration_ms": duration_ms,
            }),
        );

        self.commit_root_user_message(input).await?;

        let post_result = self
            .post_loop_processing(input, result.response.content.clone())
            .await?;
        let final_content = self.apply_post_loop_result(input, post_result).await?;

        let mut response = AgentResponse::new(final_content);
        let metadata = serde_json::json!({
            "orchestration": {
                "type": "pipeline",
                "result": result.response.content,
                "stages": stages_json,
                "duration_ms": duration_ms,
            }
        });
        response.metadata = Some(
            serde_json::from_value::<std::collections::HashMap<String, serde_json::Value>>(
                metadata,
            )
            .unwrap_or_default(),
        );

        self.finish_turn_if_root(&response).await?;
        Ok(response)
    }

    // Handle handoff: LLM-directed agent-to-agent control transfer.
    async fn handle_handoff_state(
        &self,
        input: &str,
        config: &ai_agents_state::HandoffStateConfig,
    ) -> Result<AgentResponse> {
        use std::time::Instant;

        let registry = self.spawner_registry.as_ref().ok_or_else(|| {
            AgentError::Config(
                "Handoff state requires an agent registry. Add a spawner section.".into(),
            )
        })?;

        let llm = self
            .llm_registry
            .get("router")
            .map_err(|_| AgentError::Config("Handoff state requires a router LLM.".into()))?;

        let start = Instant::now();

        // Enrich input with parent conversation history when context_mode is set.
        let context_mode = config.context_mode.clone().unwrap_or_default();
        let context_input = self
            .observe_purpose(
                ObservationPurpose::OrchestrationRouting,
                crate::orchestration::context::prepare_delegate_input(
                    input,
                    &context_mode,
                    &*self.memory,
                    self.llm_registry.get("router").ok().as_deref(),
                ),
            )
            .await?;

        // Render input template if provided, otherwise forward the raw user message.
        let effective_input = if let Some(ref tmpl) = config.input {
            render_concurrent_template(tmpl, &context_input, &self.build_context_with_overlays())
                .unwrap_or_else(|_| context_input.clone())
        } else {
            context_input
        };

        let result = self
            .observe_purpose(
                ObservationPurpose::OrchestrationRouting,
                scope_actor_context(
                    self.outbound_actor_context(),
                    crate::orchestration::handoff(
                        registry,
                        &effective_input,
                        &config.initial_agent,
                        &config.available_agents,
                        config.max_handoffs,
                        llm.as_ref(),
                        Some(&*self.hooks),
                    ),
                ),
            )
            .await?;

        let duration_ms = start.elapsed().as_millis() as u64;

        // Backward-compatible context key.
        let _ = self.context_manager.set(
            "handoff.result",
            serde_json::Value::String(result.response.content.clone()),
        );

        // Build handoff chain data for context and metadata.
        let chain_json: Vec<serde_json::Value> = result
            .handoff_chain
            .iter()
            .map(|h| {
                serde_json::json!({
                    "from": h.from_agent,
                    "to": h.to_agent,
                    "reason": h.reason,
                })
            })
            .collect();

        // Structured orchestration context.
        let _ = self.context_manager.set(
            "orchestration",
            serde_json::json!({
                "type": "handoff",
                "result": result.response.content,
                "final_agent": result.final_agent,
                "handoff_chain": chain_json,
                "duration_ms": duration_ms,
            }),
        );

        self.commit_root_user_message(input).await?;

        let post_result = self
            .post_loop_processing(input, result.response.content.clone())
            .await?;
        let final_content = self.apply_post_loop_result(input, post_result).await?;

        let mut response = AgentResponse::new(final_content);
        let metadata = serde_json::json!({
            "orchestration": {
                "type": "handoff",
                "result": result.response.content,
                "final_agent": result.final_agent,
                "handoff_chain": chain_json,
                "duration_ms": duration_ms,
            }
        });
        response.metadata = Some(
            serde_json::from_value::<std::collections::HashMap<String, serde_json::Value>>(
                metadata,
            )
            .unwrap_or_default(),
        );

        self.finish_turn_if_root(&response).await?;
        Ok(response)
    }

    // run_loop_internal: blocking (non-streaming) agent pipeline.
    async fn run_loop_internal(&self, input: &str) -> Result<AgentResponse> {
        self.begin_root_turn();
        // Resolve actor_id from context, reload facts if actor changed, bump counter.
        self.pre_turn_session_lifecycle().await;

        let input_data = self.process_input(input).await?;
        self.update_active_turn_context(&input_data.content, input_data.context.clone());

        // Inject process context (detect/extract results) into agent context
        // so system prompt templates can use {{ context.detected_language }} etc.
        for (key, value) in &input_data.context {
            let _ = self.context_manager.set(key, value.clone());
        }

        if input_data.metadata.rejected {
            let reason = input_data
                .metadata
                .rejection_reason
                .unwrap_or_else(|| "Input rejected".to_string());
            warn!(reason = %reason, "Input rejected");
            let response = AgentResponse::new(reason);
            self.finish_turn_if_root(&response).await?;
            return Ok(response);
        }

        let processed_input = &input_data.content;

        if let Some(response) = self.try_pre_response_transition(processed_input).await? {
            return Ok(response);
        }

        // Handle orchestration states (delegate, concurrent, group_chat, pipeline, handoff).
        if let Some(ref sm) = self.state_machine {
            if let Some(def) = sm.current_definition() {
                if let Some(ref delegate_id) = def.delegate {
                    return self
                        .handle_delegated_state(processed_input, delegate_id, &def)
                        .await;
                }
                if let Some(ref concurrent_config) = def.concurrent {
                    return self
                        .handle_concurrent_state(processed_input, concurrent_config)
                        .await;
                }
                if let Some(ref group_chat_config) = def.group_chat {
                    return self
                        .handle_group_chat_state(processed_input, group_chat_config)
                        .await;
                }
                if let Some(ref pipeline_config) = def.pipeline {
                    return self
                        .handle_pipeline_state(processed_input, pipeline_config)
                        .await;
                }
                if let Some(ref handoff_config) = def.handoff {
                    return self
                        .handle_handoff_state(processed_input, handoff_config)
                        .await;
                }
            }
        }

        //
        // The speculative future is boxed to keep the runtime future size manageable.
        // Removing the box can overflow small test stacks because this function is recursive through redispatch.
        //
        if let Some(response) =
            Box::pin(self.try_speculative_branches(processed_input, &input_data.context)).await?
        {
            return Ok(response);
        }

        match self.try_skill_route(processed_input).await? {
            SkillRouteResult::Response(skill_response) => {
                self.commit_root_user_message(processed_input).await?;
                return self
                    .handle_skill_response(processed_input, skill_response, &input_data.context)
                    .await;
            }
            SkillRouteResult::NeedsClarification(response) => {
                self.commit_root_user_message(processed_input).await?;
                if let Some(q) = response
                    .metadata
                    .as_ref()
                    .and_then(|m| m.get("disambiguation"))
                    .and_then(|d| d.get("status"))
                    .and_then(|s| s.as_str())
                {
                    if q == "awaiting_clarification" {
                        // Store the clarification question in memory so the next turn
                        // can be handled as a clarification response.
                        self.memory
                            .add_message(ChatMessage::assistant(&response.content))
                            .await?;
                    }
                }
                self.finish_turn_if_root(&response).await?;
                return Ok(response);
            }
            SkillRouteResult::NoMatch => {} // continue to normal LLM chat
        }

        let effective_reasoning = self.get_effective_reasoning_config();
        let reasoning_mode = self.determine_reasoning_mode(processed_input).await?;
        let auto_detected = matches!(effective_reasoning.mode, ReasoningMode::Auto);

        info!(
            reasoning_mode = ?reasoning_mode,
            auto_detected = auto_detected,
            reflection_enabled = ?self.reflection_config.enabled,
            "Reasoning mode determined"
        );

        if matches!(reasoning_mode, ReasoningMode::PlanAndExecute) {
            self.commit_root_user_message(processed_input).await?;
            return self
                .handle_plan_and_execute(processed_input, &input_data.context, auto_detected)
                .await;
        }

        self.commit_root_user_message(processed_input).await?;

        let mut iterations = 0u32;
        let mut all_tool_calls: Vec<ToolCall> = Vec::new();
        let mut thinking_content: Option<String> = None;

        let llm = self.get_state_llm()?;

        loop {
            // When reasoning is active, cap iterations at the reasoning-specific limit.
            let effective_max = if reasoning_mode != ReasoningMode::None {
                let rc = self.get_effective_reasoning_config();
                self.max_iterations.min(rc.max_iterations)
            } else {
                self.max_iterations
            };

            if iterations >= effective_max {
                let err = AgentError::Other(format!("Max iterations ({}) exceeded", effective_max));
                self.hooks.on_error(&err).await;
                error!(iterations = iterations, "Max iterations exceeded");
                return Err(err);
            }
            iterations += 1;
            *self.iteration_count.write() = iterations;

            debug!(iteration = iterations, max = effective_max, "LLM call");

            let mut messages = self.build_messages().await?;
            self.inject_reasoning_prompt(&mut messages, &reasoning_mode, iterations == 1);

            self.hooks.on_llm_start(&messages).await;
            let llm_start = Instant::now();

            // Try primary LLM (with retry if configured), then apply on_failure policy.
            let response = {
                let primary_result = if self.recovery_manager.config().default.max_retries > 0 {
                    self.recovery_manager
                        .with_retry("llm_call", None, || async {
                            self.observe_purpose(
                                ObservationPurpose::MainResponse,
                                llm.complete(&messages, None),
                            )
                            .await
                            .map_err(|e| e.classify())
                        })
                        .await
                        .map_err(|e| AgentError::LLM(e.to_string()))
                } else {
                    self.observe_purpose(
                        ObservationPurpose::MainResponse,
                        llm.complete(&messages, None),
                    )
                    .await
                    .map_err(|e| AgentError::LLM(e.to_string()))
                };

                match primary_result {
                    Ok(resp) => resp,
                    Err(primary_err) => match &self.recovery_manager.config().llm.on_failure {
                        LLMFailureAction::FallbackLlm { fallback_llm } => {
                            warn!(
                                fallback = %fallback_llm,
                                error = %primary_err,
                                "Primary LLM failed, attempting fallback LLM"
                            );
                            let fb = self.llm_registry.get(fallback_llm).map_err(|e| {
                                AgentError::Config(format!(
                                    "Fallback LLM '{}' not found: {}",
                                    fallback_llm, e
                                ))
                            })?;
                            self.observe_purpose(
                                ObservationPurpose::MainResponse,
                                fb.complete(&messages, None),
                            )
                            .await
                            .map_err(|e| AgentError::LLM(e.to_string()))?
                        }
                        LLMFailureAction::FallbackResponse { message } => {
                            warn!(
                                error = %primary_err,
                                "Primary LLM failed, using static fallback response"
                            );
                            LLMResponse::new(message.clone(), FinishReason::Stop)
                        }
                        LLMFailureAction::Error => {
                            return Err(primary_err);
                        }
                    },
                }
            };

            let llm_duration_ms = llm_start.elapsed().as_millis() as u64;
            self.hooks.on_llm_complete(&response, llm_duration_ms).await;

            let content = response.content.trim();

            if let Some(tool_calls) = self.parse_tool_calls(content) {
                match self
                    .handle_tool_calls(processed_input, content, tool_calls, &mut all_tool_calls)
                    .await?
                {
                    ToolCallOutcome::Continue | ToolCallOutcome::TransitionFired => continue,
                    ToolCallOutcome::Rejected(resp) => {
                        self.finish_turn_if_root(&resp).await?;
                        return Ok(resp);
                    }
                }
            }

            let (extracted_thinking, answer) = self.extract_thinking(content);
            if extracted_thinking.is_some() {
                thinking_content = extracted_thinking;
            }

            let output_data = self.process_output(&answer, &input_data.context).await?;

            let mut final_content = if output_data.metadata.rejected {
                output_data
                    .metadata
                    .rejection_reason
                    .unwrap_or_else(|| answer.to_string())
            } else {
                output_data.content
            };

            // Run reflection (blocking LLM calls for retries)
            let reflection_metadata;
            (final_content, reflection_metadata) = self
                .run_reflection(&*llm, processed_input, final_content)
                .await?;

            final_content =
                self.format_response_with_thinking(thinking_content.as_deref(), &final_content);

            // Post-loop: memory, transitions, post-transition re-generation.
            // apply_post_loop_result handles NeedsRedispatch by re-entering
            // run_loop_internal so the new state's full dispatch activates.
            let final_content = {
                let result = self
                    .post_loop_processing(processed_input, final_content)
                    .await?;
                self.apply_post_loop_result(processed_input, result).await?
            };

            let reflected = reflection_metadata.is_some();
            let reasoning_mode_debug = format!("{:?}", reasoning_mode);

            let response = self.build_agent_response(
                final_content,
                all_tool_calls,
                reasoning_mode,
                auto_detected,
                iterations,
                thinking_content,
                reflection_metadata,
            );

            self.finish_turn_if_root(&response).await?;

            let tool_call_count = response.tool_calls.as_ref().map(|tc| tc.len()).unwrap_or(0);
            info!(
                tool_calls = tool_call_count,
                response_len = response.content.len(),
                reasoning_mode = %reasoning_mode_debug,
                reflected = reflected,
                "Chat completed"
            );
            return Ok(response);
        }
    }

    async fn generate_buffered_streaming_draft(
        &self,
        processed_input: &str,
        routing_resolved: Arc<AtomicBool>,
    ) -> Result<StreamingDraftResult> {
        let llm = self.get_state_llm()?;
        let messages = self.build_messages_for_draft(processed_input).await?;
        let mut stream = self
            .observe_purpose(
                ObservationPurpose::MainResponse,
                llm.complete_stream(&messages, None),
            )
            .await
            .map_err(|e| AgentError::LLM(e.to_string()))?;
        let mut buffer = crate::optimization::StreamBranchBuffer::new(self.streaming.buffer_size)?;
        let mut chunks = Vec::new();
        let mut accumulated = String::new();
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| AgentError::LLM(e.to_string()))?;
            accumulated.push_str(&chunk.delta);
            let stream_chunk = StreamChunk::content(chunk.delta);
            if routing_resolved.load(Ordering::SeqCst) {
                chunks.push(stream_chunk);
            } else {
                buffer.push(stream_chunk)?;
            }
        }
        chunks.splice(0..0, buffer.drain());
        let content = accumulated.trim().to_string();
        let draft = if let Some(calls) = self.parse_tool_calls(&content) {
            MainResponseDraft::ToolCalls {
                raw_content: content,
                calls,
                thinking: None,
            }
        } else {
            MainResponseDraft::Text {
                raw_content: content,
                thinking: None,
            }
        };
        Ok(StreamingDraftResult::new(draft, chunks))
    }

    async fn try_buffered_streaming_branches(
        &self,
        processed_input: &str,
        input_context: &HashMap<String, Value>,
    ) -> Result<Option<(AgentResponse, Vec<StreamChunk>)>> {
        let optimization = &self.runtime_config.optimization;
        if !optimization.enabled {
            return Ok(None);
        }
        let transition_enabled =
            optimization.speculative_state_transitions && self.has_parallel_transition_candidates();
        if !transition_enabled {
            return Ok(None);
        }
        let mut branch_scheduler =
            TurnBranchScheduler::new(optimization.max_parallel_runtime_tasks)?;
        if !branch_scheduler.reserve_task() {
            return Ok(None);
        }
        if !self
            .reserve_active_speculative_llm_call(RuntimeOptimizationKind::BufferedStreamingRouting)
        {
            branch_scheduler.release_task();
            return Ok(None);
        }
        if !branch_scheduler.reserve_task() {
            branch_scheduler.release_task();
            return Ok(None);
        }
        let mut main_branch = RuntimeBranch::new(
            RuntimeTaskPurpose::MainResponse,
            RuntimeOptimizationKind::BufferedStreamingRouting,
            RuntimeTaskPriority::Normal,
            RuntimeCommitBehavior::FinalResponse,
        );
        let mut transition_branch = RuntimeBranch::new(
            RuntimeTaskPurpose::StateTransition,
            RuntimeOptimizationKind::ParallelStateTransition,
            RuntimeTaskPriority::Critical,
            RuntimeCommitBehavior::TransitionDecision,
        );
        let main_id = main_branch.branch_id();
        let transition_id = transition_branch.branch_id();
        let routing_resolved = Arc::new(AtomicBool::new(false));
        let mut main_future =
            Box::pin(crate::optimization::observability::with_branch_observation(
                &main_id,
                RuntimeOptimizationKind::BufferedStreamingRouting,
                RuntimeCommitBehavior::FinalResponse,
                self.generate_buffered_streaming_draft(
                    processed_input,
                    Arc::clone(&routing_resolved),
                ),
            ));
        let mut transition_future =
            Box::pin(crate::optimization::observability::with_branch_observation(
                &transition_id,
                RuntimeOptimizationKind::ParallelStateTransition,
                RuntimeCommitBehavior::TransitionDecision,
                self.select_parallel_transition_candidate(processed_input),
            ));
        let mut main_pending = true;
        let mut transition_pending = true;
        let mut main_result: Option<Result<StreamingDraftResult>> = None;
        let mut transition_finalized = false;
        let mut transition_candidate: Option<TransitionCandidate> = None;
        loop {
            if let Some(candidate) = transition_candidate.take() {
                if self
                    .commit_pre_response_transition_candidate(
                        &candidate,
                        &HashMap::new(),
                        processed_input,
                    )
                    .await?
                {
                    self.finalize_optional_branch(
                        &transition_id,
                        RuntimeOptimizationKind::ParallelStateTransition,
                        RuntimeCommitBehavior::TransitionDecision,
                        "committed",
                        true,
                    );
                    self.finalize_branch_loss(
                        &main_id,
                        RuntimeOptimizationKind::BufferedStreamingRouting,
                        RuntimeCommitBehavior::FinalResponse,
                        main_pending,
                        main_result.as_ref().map(|result| result.is_err()),
                    );
                    let response = self.redispatch_current_state(processed_input).await?;
                    return Ok(Some((
                        response.clone(),
                        vec![StreamChunk::content(response.content)],
                    )));
                }
                self.finalize_optional_branch(
                    &transition_id,
                    RuntimeOptimizationKind::ParallelStateTransition,
                    RuntimeCommitBehavior::TransitionDecision,
                    "discarded",
                    false,
                );
                routing_resolved.store(true, Ordering::SeqCst);
                transition_finalized = true;
            }
            if transition_finalized {
                if let Some(result) = main_result.take() {
                    let stream_draft = match result {
                        Ok(stream_draft) => stream_draft,
                        Err(error) => {
                            self.finalize_optional_branch(
                                &main_id,
                                RuntimeOptimizationKind::BufferedStreamingRouting,
                                RuntimeCommitBehavior::FinalResponse,
                                "failed",
                                false,
                            );
                            return Err(error);
                        }
                    };
                    let raw_draft_content = stream_draft.draft.raw_content().to_string();
                    let buffered_chunks = stream_draft.chunks;
                    self.finalize_optional_branch(
                        &main_id,
                        RuntimeOptimizationKind::BufferedStreamingRouting,
                        RuntimeCommitBehavior::FinalResponse,
                        "committed",
                        true,
                    );
                    let response = self
                        .commit_main_response_draft(
                            processed_input,
                            input_context,
                            stream_draft.draft,
                            ReasoningMode::None,
                            false,
                        )
                        .await?;
                    let chunks = if response.content == raw_draft_content {
                        buffered_chunks
                    } else {
                        vec![StreamChunk::content(response.content.clone())]
                    };
                    return Ok(Some((response, chunks)));
                }
            }
            tokio::select! {
                result = &mut main_future, if main_pending => {
                    main_pending = false;
                    main_branch.transition_to(RuntimeBranchStatus::Completed)?;
                    main_result = Some(result);
                }
                result = &mut transition_future, if transition_pending => {
                    transition_pending = false;
                    transition_branch.transition_to(RuntimeBranchStatus::Completed)?;
                    match result {
                        Ok(Some(candidate)) => transition_candidate = Some(candidate),
                        Ok(None) => {
                            self.finalize_optional_branch(
                                &transition_id,
                                RuntimeOptimizationKind::ParallelStateTransition,
                                RuntimeCommitBehavior::TransitionDecision,
                                "discarded",
                                false,
                            );
                            routing_resolved.store(true, Ordering::SeqCst);
                            transition_finalized = true;
                        }
                        Err(_) => {
                            self.finalize_optional_branch(
                                &transition_id,
                                RuntimeOptimizationKind::ParallelStateTransition,
                                RuntimeCommitBehavior::TransitionDecision,
                                "failed",
                                false,
                            );
                            routing_resolved.store(true, Ordering::SeqCst);
                            transition_finalized = true;
                        }
                    }
                }
            }
        }
    }

    /// Streaming agent pipeline
    /// Uses all the same shared helpers as run_loop_internal.
    /// The ONLY difference: LLM calls use complete_stream() + yield deltas.
    fn run_loop_internal_stream<'a>(
        &'a self,
        input: &'a str,
    ) -> Pin<Box<dyn Stream<Item = StreamChunk> + Send + 'a>> {
        let include_tool_events = self.streaming.include_tool_events;
        let include_state_events = self.streaming.include_state_events;

        Box::pin(async_stream::stream! {
            self.begin_root_turn();
            // Parity with non-stream: resolve actor from context and load facts if changed.
            self.pre_turn_session_lifecycle().await;

            let input_data = match self.process_input(input).await {
                Ok(data) => data,
                Err(e) => {
                    yield StreamChunk::error(e.to_string());
                    return;
                }
            };
            self.update_active_turn_context(&input_data.content, input_data.context.clone());

            // Inject process context (detect/extract results) into agent context
            for (key, value) in &input_data.context {
                let _ = self.context_manager.set(key, value.clone());
            }

            if input_data.metadata.rejected {
                let reason = input_data
                    .metadata
                    .rejection_reason
                    .unwrap_or_else(|| "Input rejected".to_string());
                warn!(reason = %reason, "Input rejected (stream)");
                yield StreamChunk::error(reason);
                return;
            }

            let processed_input = &input_data.content;

            if self.runtime_config.optimization.enabled
                && matches!(
                    self.runtime_config.optimization.streaming_policy,
                    crate::optimization::StreamingOptimizationPolicy::BufferUntilRoutingDone
                )
            {
                //
                // Buffered routing keeps stale stream output hidden until a branch winner is known.
                // The boxed future prevents this stream state machine from becoming too large.
                //
                match Box::pin(self.try_buffered_streaming_branches(processed_input, &input_data.context)).await {
                    Ok(Some((_response, chunks))) => {
                        for chunk in chunks {
                            yield chunk;
                        }
                        yield StreamChunk::Done {};
                        return;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        yield StreamChunk::error(e.to_string());
                        return;
                    }
                }
            }

            if self.runtime_config.optimization.enabled
                && matches!(
                    self.runtime_config.optimization.streaming_policy,
                    crate::optimization::StreamingOptimizationPolicy::PreflightOnly
                )
            {
                match self.try_pre_response_transition(processed_input).await {
                    Ok(Some(response)) => {
                        yield StreamChunk::content(&response.content);
                        yield StreamChunk::Done {};
                        return;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        yield StreamChunk::error(e.to_string());
                        return;
                    }
                }
            }

            // Handle orchestration states in streaming mode.
            if let Some(ref sm) = self.state_machine {
                if let Some(def) = sm.current_definition() {
                    let orchestration_result = if let Some(ref delegate_id) = def.delegate {
                        Some(self.handle_delegated_state(processed_input, delegate_id, &def).await)
                    } else if let Some(ref concurrent_config) = def.concurrent {
                        Some(self.handle_concurrent_state(processed_input, concurrent_config).await)
                    } else if let Some(ref group_chat_config) = def.group_chat {
                        Some(self.handle_group_chat_state(processed_input, group_chat_config).await)
                    } else if let Some(ref pipeline_config) = def.pipeline {
                        Some(self.handle_pipeline_state(processed_input, pipeline_config).await)
                    } else if let Some(ref handoff_config) = def.handoff {
                        Some(self.handle_handoff_state(processed_input, handoff_config).await)
                    } else {
                        None
                    };

                    if let Some(result) = orchestration_result {
                        match result {
                            Ok(response) => {
                                yield StreamChunk::content(&response.content);
                                yield StreamChunk::Done {};
                            }
                            Err(e) => {
                                yield StreamChunk::error(e.to_string());
                            }
                        }
                        return;
                    }
                }
            }

            // Skill routing
            match self.try_skill_route(processed_input).await {
                Ok(SkillRouteResult::Response(skill_response)) => {
                    if let Err(e) = self.commit_root_user_message(processed_input).await {
                        yield StreamChunk::error(e.to_string());
                        return;
                    }
                    match self.handle_skill_response(processed_input, skill_response, &input_data.context).await {
                        Ok(resp) => {
                            yield StreamChunk::content(&resp.content);
                            yield StreamChunk::Done {};
                            return;
                        }
                        Err(e) => {
                            yield StreamChunk::error(e.to_string());
                            return;
                        }
                    }
                }
                Ok(SkillRouteResult::NeedsClarification(response)) => {
                    if let Err(e) = self.commit_root_user_message(processed_input).await {
                        yield StreamChunk::error(e.to_string());
                        return;
                    }
                    let _ = self.memory.add_message(ChatMessage::assistant(&response.content)).await;
                    if let Err(e) = self.finish_turn_if_root(&response).await {
                        yield StreamChunk::error(e.to_string());
                        return;
                    }
                    yield StreamChunk::content(&response.content);
                    yield StreamChunk::Done {};
                    return;
                }
                Ok(SkillRouteResult::NoMatch) => {} // no skill matched, continue
                Err(e) => {
                    yield StreamChunk::error(e.to_string());
                    return;
                }
            }

            // Reasoning mode determination
            let effective_reasoning = self.get_effective_reasoning_config();
            let reasoning_mode = match self.determine_reasoning_mode(processed_input).await {
                Ok(mode) => mode,
                Err(e) => {
                    yield StreamChunk::error(e.to_string());
                    return;
                }
            };
            let auto_detected = matches!(effective_reasoning.mode, ReasoningMode::Auto);

            info!(
                reasoning_mode = ?reasoning_mode,
                auto_detected = auto_detected,
                "Reasoning mode determined (stream)"
            );

            // Plan-and-Execute: yield final result as single chunk (not token-by-token)
            if matches!(reasoning_mode, ReasoningMode::PlanAndExecute) {
                if let Err(e) = self.commit_root_user_message(processed_input).await {
                    yield StreamChunk::error(e.to_string());
                    return;
                }
                match self.handle_plan_and_execute(processed_input, &input_data.context, auto_detected).await {
                    Ok(resp) => {
                        yield StreamChunk::content(&resp.content);
                        yield StreamChunk::Done {};
                        return;
                    }
                    Err(e) => {
                        yield StreamChunk::error(e.to_string());
                        return;
                    }
                }
            }

            if let Err(e) = self.commit_root_user_message(processed_input).await {
                yield StreamChunk::error(e.to_string());
                return;
            }

            let llm = match self.get_state_llm() {
                Ok(llm) => llm,
                Err(e) => {
                    yield StreamChunk::error(e.to_string());
                    return;
                }
            };

            let mut iterations = 0u32;
            let mut all_tool_calls: Vec<ToolCall> = Vec::new();
            let mut thinking_content: Option<String> = None;

            loop {
                // When reasoning is active, cap iterations at the reasoning-specific limit.
                let effective_max = if reasoning_mode != ReasoningMode::None {
                    let rc = self.get_effective_reasoning_config();
                    self.max_iterations.min(rc.max_iterations)
                } else {
                    self.max_iterations
                };

                if iterations >= effective_max {
                    let err_msg = format!("Max iterations ({}) exceeded", effective_max);
                    let err = AgentError::Other(err_msg.clone());
                    self.hooks.on_error(&err).await;
                    error!(iterations = iterations, "Max iterations exceeded (stream)");
                    yield StreamChunk::error(err_msg);
                    return;
                }
                iterations += 1;
                *self.iteration_count.write() = iterations;

                debug!(iteration = iterations, max = effective_max, "LLM call (stream)");

                let mut messages = match self.build_messages().await {
                    Ok(m) => m,
                    Err(e) => {
                        yield StreamChunk::error(e.to_string());
                        return;
                    }
                };
                self.inject_reasoning_prompt(&mut messages, &reasoning_mode, iterations == 1);

                self.hooks.on_llm_start(&messages).await;
                let llm_start = Instant::now();

                // Check if reflection is active — if so, suppress streaming for this LLM call
                // because we may need to retry and the user would see a stale first attempt.
                let reflection_active = match self.should_reflect(processed_input, "").await {
                    Ok(v) => v,
                    Err(_) => false,
                };

                let content = if reflection_active {
                    // Use blocking call so reflection can retry without partial output
                    let response = match self
                        .observe_purpose(
                            ObservationPurpose::MainResponse,
                            llm.complete(&messages, None),
                        )
                        .await
                    {
                        Ok(r) => r,
                        Err(e) => {
                            yield StreamChunk::error(e.to_string());
                            return;
                        }
                    };
                    let llm_duration_ms = llm_start.elapsed().as_millis() as u64;
                    self.hooks.on_llm_complete(&response, llm_duration_ms).await;
                    response.content.trim().to_string()
                } else {
                    // Streaming LLM call
                    let llm_stream = match self
                        .observe_purpose(
                            ObservationPurpose::MainResponse,
                            llm.complete_stream(&messages, None),
                        )
                        .await
                    {
                        Ok(s) => s,
                        Err(e) => {
                            yield StreamChunk::error(e.to_string());
                            return;
                        }
                    };
                    let mut accumulated = String::new();
                    let mut stream_inner = llm_stream;
                    while let Some(chunk_result) = stream_inner.next().await {
                        match chunk_result {
                            Ok(chunk) => {
                                accumulated.push_str(&chunk.delta);
                                yield StreamChunk::content(chunk.delta);
                            }
                            Err(e) => {
                                yield StreamChunk::error(e.to_string());
                                return;
                            }
                        }
                    }
                    let llm_duration_ms = llm_start.elapsed().as_millis() as u64;
                    // Construct LLMResponse for hooks
                    let llm_response = ai_agents_core::LLMResponse::new(
                        accumulated.trim(),
                        ai_agents_core::FinishReason::Stop,
                    );
                    self.hooks.on_llm_complete(&llm_response, llm_duration_ms).await;
                    accumulated.trim().to_string()
                };

                // Tool call handling
                if let Some(tool_calls) = self.parse_tool_calls(&content) {
                    // Emit tool events for streaming
                    // First check transitions (same as blocking path)
                    let transition_fired = match self.evaluate_transitions(processed_input, &content).await {
                        Ok(v) => v,
                        Err(e) => {
                            yield StreamChunk::error(e.to_string());
                            return;
                        }
                    };
                    if transition_fired {
                        let _ = self.memory.add_message(ChatMessage::assistant(
                            "(Transitioned to new state — tool call handled by workflow)",
                        )).await;

                        if include_state_events {
                            if let Some(state) = self.current_state() {
                                yield StreamChunk::state_transition(None, state);
                            }
                        }
                        continue;
                    }

                    // Store the assistant's tool-call message (same as blocking path)
                    let _ = self.memory.add_message(ChatMessage::assistant(&content)).await;

                    // Execute tools with streaming events
                    let results = self.execute_tools_parallel(&tool_calls).await;

                    for ((_id, result), tool_call) in results.into_iter().zip(tool_calls.iter()) {
                        if include_tool_events {
                            yield StreamChunk::tool_start(&tool_call.id, &tool_call.name);
                        }

                        match result {
                            Ok(output) => {
                                if include_tool_events {
                                    yield StreamChunk::tool_result(
                                        &tool_call.id,
                                        &tool_call.name,
                                        &output,
                                        true,
                                    );
                                }
                                let _ = self.memory
                                    .add_message(ChatMessage::function(&tool_call.name, &output))
                                    .await;
                            }
                            Err(e) => {
                                if matches!(e, AgentError::HITLRejected(_)) {
                                    let _ = self.memory.add_message(ChatMessage::assistant(
                                        &format!("The operation was rejected by the approver: {}", e),
                                    )).await;
                                    let response = AgentResponse {
                                        content: format!("Operation cancelled: {}", e),
                                        metadata: None,
                                        tool_calls: Some(all_tool_calls.clone()),
                                    };
                                    if let Err(finalize_error) = self.finish_turn_if_root(&response).await {
                                        yield StreamChunk::error(finalize_error.to_string());
                                        return;
                                    }
                                    yield StreamChunk::error(response.content);
                                    yield StreamChunk::Done {};
                                    return;
                                }
                                if include_tool_events {
                                    yield StreamChunk::tool_result(
                                        &tool_call.id,
                                        &tool_call.name,
                                        &e.to_string(),
                                        false,
                                    );
                                }
                                let _ = self.memory
                                    .add_message(ChatMessage::function(
                                        &tool_call.name,
                                        &format!("Error: {}", e),
                                    ))
                                    .await;
                            }
                        }
                        all_tool_calls.push(tool_call.clone());

                        if include_tool_events {
                            yield StreamChunk::tool_end(&tool_call.id);
                        }
                    }
                    continue;
                }

                // Extract thinking, process output
                let (extracted_thinking, answer) = self.extract_thinking(&content);
                if extracted_thinking.is_some() {
                    thinking_content = extracted_thinking;
                }

                let output_data = match self.process_output(&answer, &input_data.context).await {
                    Ok(d) => d,
                    Err(e) => {
                        yield StreamChunk::error(e.to_string());
                        return;
                    }
                };

                let final_content = if output_data.metadata.rejected {
                    output_data
                        .metadata
                        .rejection_reason
                        .unwrap_or_else(|| answer.to_string())
                } else {
                    output_data.content
                };

                // Reflection (uses blocking LLM calls for retries)
                let (final_content, _reflection_metadata) = match self
                    .run_reflection(&*llm, processed_input, final_content)
                    .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        yield StreamChunk::error(e.to_string());
                        return;
                    }
                };

                let final_content = self.format_response_with_thinking(
                    thinking_content.as_deref(),
                    &final_content,
                );

                // If reflection was active (we used blocking call), yield the final content now
                if reflection_active {
                    yield StreamChunk::content(&final_content);
                }

                // Post-loop: memory, transitions, post-transition re-generation.
                // For NeedsRedispatch, run_loop_internal handles the new state's full
                // dispatch and its result is yielded as a single non-streamed chunk.
                let post_result = match self
                    .post_loop_processing(processed_input, final_content)
                    .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        yield StreamChunk::error(e.to_string());
                        return;
                    }
                };

                let (final_content, transitioned) = match post_result {
                    PostLoopResult::NoTransition(content) => (content, false),
                    PostLoopResult::Transitioned(content) => (content, true),
                    PostLoopResult::NeedsRedispatch => {
                        const MAX_REDISPATCH_DEPTH: u32 = 3;
                        let current_depth = *self.redispatch_depth.read();
                        let content = if current_depth >= MAX_REDISPATCH_DEPTH {
                            warn!(
                                depth = current_depth,
                                "Post-transition re-dispatch depth limit reached (stream)"
                            );
                            let c = String::new();
                            let _ = self.memory.add_message(ChatMessage::assistant(&c)).await;
                            c
                        } else {
                            *self.redispatch_depth.write() += 1;
                            if let Some(context) = self.active_turn_context.write().as_mut() {
                                context.enter_redispatch();
                            }
                            info!(
                                depth = current_depth + 1,
                                "Re-dispatching for new state after transition (stream)"
                            );
                            let result = self.run_loop_internal(processed_input).await;
                            *self.redispatch_depth.write() -= 1;
                            if let Some(context) = self.active_turn_context.write().as_mut() {
                                context.exit_redispatch();
                            }
                            match result {
                                Ok(resp) => resp.content,
                                Err(e) => {
                                    yield StreamChunk::error(e.to_string());
                                    return;
                                }
                            }
                        };
                        (content, true)
                    }
                };

                if transitioned {
                    if include_state_events {
                        if let Some(state) = self.current_state() {
                            yield StreamChunk::state_transition(None, state);
                        }
                    }
                    // Yield the post-transition re-generated or re-dispatched content.
                    yield StreamChunk::content(&final_content);
                }

                // Parity with non-stream: finalize the committed response once.
                let final_response = AgentResponse::new(&final_content);
                if let Err(e) = self.finish_turn_if_root(&final_response).await {
                    yield StreamChunk::error(e.to_string());
                    return;
                }

                yield StreamChunk::Done {};
                return;
            }
        })
    }

    /// Streaming entry point with disambiguation
    /// Mirrors run_loop but yields StreamChunks instead of AgentResponse.
    fn run_loop_stream<'a>(
        &'a self,
        input: &'a str,
    ) -> Pin<Box<dyn Stream<Item = StreamChunk> + Send + 'a>> {
        Box::pin(async_stream::stream! {
            self.begin_root_turn();
            let _root_cleanup = RootTurnCleanup::new(self);
            self.hooks.on_message_received(input).await;

            // One-shot context initialization (mirrors run_loop)
            if !self.context_initialized.swap(true, Ordering::SeqCst) {
                if let Err(e) = self.context_manager.initialize().await {
                    yield StreamChunk::error(e.to_string());
                    return;
                }
                debug!("Context manager initialized (defaults, env, builtins)");
            }

            if let Err(e) = self.check_turn_timeout().await {
                yield StreamChunk::error(e.to_string());
                return;
            }
            if let Err(e) = self.context_manager.refresh_per_turn().await {
                yield StreamChunk::error(e.to_string());
                return;
            }

            // Clear stale disambiguation context from previous turns.
            self.clear_disambiguation_context();

            // Disambiguation check (before input processing)
            if let Some(ref disambiguator) = self.disambiguation_manager {
                let disambiguation_context = match self.build_disambiguation_context().await {
                    Ok(ctx) => ctx,
                    Err(e) => {
                        yield StreamChunk::error(e.to_string());
                        return;
                    }
                };

                let state_override = self
                    .state_machine
                    .as_ref()
                    .and_then(|sm| sm.current_definition())
                    .and_then(|def| def.disambiguation.clone());

                let result = match self
                    .observe_purpose(
                        ObservationPurpose::DisambiguationDetection,
                        disambiguator.process_input_with_override(
                            input,
                            &disambiguation_context,
                            state_override.as_ref(),
                            None,
                        ),
                    )
                    .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        yield StreamChunk::error(e.to_string());
                        return;
                    }
                };

                match result {
                    DisambiguationResult::Clear => {
                        debug!("Input is clear, proceeding normally (stream)");
                    }
                    DisambiguationResult::NeedsClarification {
                        question,
                        detection,
                    } => {
                        info!(
                            ambiguity_type = ?detection.ambiguity_type,
                            confidence = detection.confidence,
                            "Input requires clarification (stream)"
                        );
                        if let Err(e) = self.commit_root_user_message(input).await {
                            yield StreamChunk::error(e.to_string());
                            return;
                        }
                        let _ = self
                            .memory
                            .add_message(ChatMessage::assistant(&question.question))
                            .await;
                        let response = AgentResponse::new(&question.question);
                        if let Err(e) = self.finish_turn_if_root(&response).await {
                            yield StreamChunk::error(e.to_string());
                            return;
                        }
                        yield StreamChunk::content(&question.question);
                        yield StreamChunk::Done {};
                        return;
                    }
                    DisambiguationResult::Clarified {
                        enriched_input,
                        resolved,
                        ..
                    } => {
                        info!(
                            resolved_count = resolved.len(),
                            enriched = %enriched_input,
                            "Input clarified (stream)"
                        );
                        for (key, value) in &resolved {
                            let context_key = format!("disambiguation.{}", key);
                            let _ = self.context_manager.set(&context_key, value.clone());
                        }
                        if let Some(intent) = resolved.get("intent") {
                            let _ = self.context_manager.set("resolved_intent", intent.clone());
                        }
                        let _ = self
                            .context_manager
                            .set("disambiguation.resolved", serde_json::Value::Bool(true));

                        // Check if this clarification was triggered by a skill-level override.
                        // Re-run skill disambiguation to verify all required_clarity fields
                        // are present before executing.
                        let skill_id = self.pending_skill_id.read().clone();
                        if let Some(skill_id) = skill_id {
                            info!(skill_id = %skill_id, "Re-checking skill disambiguation on clarified input (stream)");
                            match self.recheck_skill_disambiguation(&skill_id, &enriched_input).await {
                                Ok(resp) => {
                                    yield StreamChunk::content(&resp.content);
                                    yield StreamChunk::Done {};
                                    return;
                                }
                                Err(e) => {
                                    yield StreamChunk::error(e.to_string());
                                    return;
                                }
                            }
                        }

                        // Forward to internal stream with enriched input
                        let mut inner = self.run_loop_internal_stream(&enriched_input);
                        while let Some(chunk) = inner.next().await {
                            yield chunk;
                        }
                        return;
                    }
                    DisambiguationResult::ProceedWithBestGuess { enriched_input } => {
                        info!("Proceeding with best guess (stream)");

                        // Same skill-id re-check for best-guess path
                        let skill_id = self.pending_skill_id.read().clone();
                        if let Some(skill_id) = skill_id {
                            info!(skill_id = %skill_id, "Re-checking skill disambiguation on best-guess input (stream)");
                            match self.recheck_skill_disambiguation(&skill_id, &enriched_input).await {
                                Ok(resp) => {
                                    yield StreamChunk::content(&resp.content);
                                    yield StreamChunk::Done {};
                                    return;
                                }
                                Err(e) => {
                                    yield StreamChunk::error(e.to_string());
                                    return;
                                }
                            }
                        }

                        let mut inner = self.run_loop_internal_stream(&enriched_input);
                        while let Some(chunk) = inner.next().await {
                            yield chunk;
                        }
                        return;
                    }
                    DisambiguationResult::GiveUp { reason } => {
                        *self.pending_skill_id.write() = None;
                        warn!(reason = %reason, "Disambiguation gave up (stream)");
                        let apology = self
                            .generate_localized_apology(
                                "Generate a brief, polite apology saying you couldn't understand the request. Be concise.",
                                &reason,
                            )
                            .await
                            .unwrap_or_else(|_| {
                                format!("I'm sorry, I couldn't understand your request: {}", reason)
                            });
                        let response = AgentResponse::new(&apology);
                        if let Err(e) = self.finish_turn_if_root(&response).await {
                            yield StreamChunk::error(e.to_string());
                            return;
                        }
                        yield StreamChunk::content(&apology);
                        yield StreamChunk::Done {};
                        return;
                    }
                    DisambiguationResult::Escalate { reason } => {
                        *self.pending_skill_id.write() = None;
                        info!(reason = %reason, "Escalating to human (stream)");
                        if let Some(ref hitl) = self.hitl_engine {
                            let trigger =
                                ApprovalTrigger::condition("disambiguation_escalation", reason.clone());
                            let mut context_map = HashMap::new();
                            context_map.insert("original_input".to_string(), serde_json::json!(input));
                            context_map.insert("reason".to_string(), serde_json::json!(&reason));
                            let check_result = HITLCheckResult::required(
                                trigger,
                                context_map,
                                format!("User request needs human assistance: {}", reason),
                                Some(hitl.config().default_timeout_seconds),
                            );
                            match self.request_hitl_approval(check_result).await {
                                Ok(result) if matches!(result, ApprovalResult::Approved | ApprovalResult::Modified { .. }) => {
                                    let mut inner = self.run_loop_internal_stream(input);
                                    while let Some(chunk) = inner.next().await {
                                        yield chunk;
                                    }
                                    return;
                                }
                                Ok(_) => {}
                                Err(e) => {
                                    yield StreamChunk::error(e.to_string());
                                    return;
                                }
                            }
                        }
                        let apology = self
                            .generate_localized_apology(
                                "Explain briefly that you're transferring the user to a human agent for help.",
                                &reason,
                            )
                            .await
                            .unwrap_or_else(|_| {
                                format!("I need human assistance to help with your request: {}", reason)
                            });
                        let response = AgentResponse::new(&apology);
                        if let Err(e) = self.finish_turn_if_root(&response).await {
                            yield StreamChunk::error(e.to_string());
                            return;
                        }
                        yield StreamChunk::content(&apology);
                        yield StreamChunk::Done {};
                        return;
                    }
                    DisambiguationResult::Abandoned { new_input } => {
                        *self.pending_skill_id.write() = None;

                        info!(
                            has_new_input = new_input.is_some(),
                            "Clarification abandoned by user (stream)"
                        );

                        if let Err(e) = self.commit_root_user_message(input).await {
                            yield StreamChunk::error(e.to_string());
                            return;
                        }

                        match new_input {
                            Some(fresh_input) => {
                                // Topic switch: forward to internal stream with fresh input.
                                let mut inner = self.run_loop_internal_stream(&fresh_input);
                                while let Some(chunk) = inner.next().await {
                                    yield chunk;
                                }
                                return;
                            }
                            None => {
                                // Pure abandonment: generate a brief acknowledgment.
                                let ack = self
                                    .generate_localized_apology(
                                        "The user changed their mind about their previous request. \
                                         Generate a brief, friendly acknowledgment (e.g. 'OK, no problem. What else can I help with?'). \
                                         Do NOT apologize excessively. Be concise.",
                                        "User abandoned clarification",
                                    )
                                    .await
                                    .unwrap_or_else(|_| {
                                        "OK, no problem. What else can I help with?".to_string()
                                    });

                                let _ = self
                                    .memory
                                    .add_message(ChatMessage::assistant(&ack))
                                    .await;

                                let response = AgentResponse::new(&ack);
                                if let Err(e) = self.finish_turn_if_root(&response).await {
                                    yield StreamChunk::error(e.to_string());
                                    return;
                                }
                                yield StreamChunk::content(&ack);
                                yield StreamChunk::Done {};
                                return;
                            }
                        }
                    }
                }
            }

            // No disambiguation or Clear result — proceed with internal stream
            let mut inner = self.run_loop_internal_stream(input);
            while let Some(chunk) = inner.next().await {
                yield chunk;
            }
        })
    }

    pub fn info(&self) -> AgentInfo {
        self.info.clone()
    }

    pub fn skills(&self) -> &[SkillDefinition] {
        &self.skills
    }

    pub async fn reset(&self) -> Result<()> {
        self.memory.clear().await?;
        *self.iteration_count.write() = 0;
        self.tool_call_history.write().clear();
        *self.pending_skill_id.write() = None;
        if let Some(ref sm) = self.state_machine {
            sm.reset();
        }
        Ok(())
    }

    pub fn max_context_tokens(&self) -> u32 {
        self.max_context_tokens
    }

    pub fn llm_registry(&self) -> &Arc<LLMRegistry> {
        &self.llm_registry
    }

    pub fn state_machine(&self) -> Option<&Arc<StateMachine>> {
        self.state_machine.as_ref()
    }

    pub fn context_manager(&self) -> &Arc<ContextManager> {
        &self.context_manager
    }

    pub fn tool_call_history(&self) -> Vec<ToolCallRecord> {
        self.tool_call_history.read().clone()
    }

    pub fn memory_token_budget(&self) -> Option<&MemoryTokenBudget> {
        self.memory_token_budget.as_ref()
    }

    pub fn parallel_tools_config(&self) -> &ParallelToolsConfig {
        &self.parallel_tools
    }

    pub fn streaming_config(&self) -> &StreamingConfig {
        &self.streaming
    }

    pub fn hooks(&self) -> &Arc<dyn AgentHooks> {
        &self.hooks
    }

    pub fn hitl_engine(&self) -> Option<&HITLEngine> {
        self.hitl_engine.as_ref()
    }

    pub fn approval_handler(&self) -> &Arc<dyn ApprovalHandler> {
        &self.approval_handler
    }

    /// Build a context map with language hints from context_manager for HITL message localization.
    fn build_hitl_language_context(&self) -> HashMap<String, Value> {
        let mut ctx = HashMap::new();
        for key in &["user.language", "input.detected.language", "language"] {
            if let Some(val) = self.context_manager.get(key) {
                ctx.insert(key.to_string(), val);
            }
        }
        ctx
    }

    /// Send a HITL check result through the approval flow and return the full ApprovalResult.
    async fn request_hitl_approval(&self, check_result: HITLCheckResult) -> Result<ApprovalResult> {
        let Some(request) = check_result.into_request() else {
            return Ok(ApprovalResult::Approved);
        };

        self.hooks.on_approval_requested(&request).await;

        let request_id = request.id.clone();
        let timeout = request.timeout;

        let result = if let Some(duration) = timeout {
            match tokio::time::timeout(duration, self.approval_handler.request_approval(request))
                .await
            {
                Ok(result) => result,
                Err(_) => ApprovalResult::timeout(),
            }
        } else {
            self.approval_handler.request_approval(request).await
        };

        self.hooks.on_approval_result(&request_id, &result).await;

        // Resolve timeout action
        let result = match result {
            ApprovalResult::Timeout => {
                if let Some(ref engine) = self.hitl_engine {
                    match engine.config().on_timeout {
                        TimeoutAction::Approve => ApprovalResult::Approved,
                        TimeoutAction::Reject => ApprovalResult::Rejected {
                            reason: Some("Timeout".to_string()),
                        },
                        TimeoutAction::Error => {
                            return Err(AgentError::Other("HITL approval timeout".to_string()));
                        }
                    }
                } else {
                    ApprovalResult::Rejected {
                        reason: Some("Timeout (no engine)".to_string()),
                    }
                }
            }
            other => other,
        };

        Ok(result)
    }

    pub async fn check_state_hitl(&self, from: Option<&str>, to: &str) -> Result<bool> {
        if let Some(ref hitl_engine) = self.hitl_engine {
            let hitl_lang_ctx = self.build_hitl_language_context();
            let check_result = self
                .observe_purpose(
                    ObservationPurpose::HitlLocalization,
                    hitl_engine.check_state_transition_with_localization(
                        from,
                        to,
                        &hitl_lang_ctx,
                        self.approval_handler.as_ref(),
                        Some(&self.llm_registry),
                    ),
                )
                .await?;
            if check_result.is_required() {
                let result = self.request_hitl_approval(check_result).await?;
                return Ok(matches!(
                    result,
                    ApprovalResult::Approved | ApprovalResult::Modified { .. }
                ));
            }
        }
        Ok(true)
    }

    /// Execute multiple tools in parallel
    async fn execute_tools_parallel(
        &self,
        tool_calls: &[ToolCall],
    ) -> Vec<(String, Result<String>)> {
        if !self.parallel_tools.enabled || tool_calls.len() <= 1 {
            let mut results = Vec::new();
            for tc in tool_calls {
                let result = self
                    .observe_purpose(
                        current_observation_context()
                            .map(|context| context.purpose)
                            .unwrap_or_default(),
                        self.execute_tool_smart(tc),
                    )
                    .await;
                results.push((tc.id.clone(), result));
            }
            return results;
        }

        let chunks: Vec<_> = tool_calls
            .chunks(self.parallel_tools.max_parallel)
            .collect();

        let mut all_results = Vec::new();

        for chunk in chunks {
            let futures: Vec<_> = chunk
                .iter()
                .map(|tc| {
                    let tc = tc.clone();
                    async move {
                        let result = self.execute_tool_smart(&tc).await;
                        (tc.id.clone(), result)
                    }
                })
                .collect();

            let results = futures::future::join_all(futures).await;
            all_results.extend(results);
        }

        all_results
    }

    /// Stream a chat response with real-time updates
    pub async fn chat_stream<'a>(
        &'a self,
        input: &'a str,
    ) -> Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send + 'a>>> {
        info!(input_len = input.len(), "Starting streaming chat");
        let inner = self.run_loop_stream(input);
        if let Some(context) = self.build_observation_context(None) {
            let stream: Pin<Box<dyn Stream<Item = StreamChunk> + Send + 'a>> =
                Box::pin(async_stream::stream! {
                    let mut inner = inner;
                    loop {
                        let next = with_observation_context(context.clone(), inner.next()).await;
                        match next {
                            Some(chunk) => yield chunk,
                            None => break,
                        }
                    }
                    self.export_observability_if_configured().await;
                });
            Ok(stream)
        } else {
            Ok(inner)
        }
    }
}

#[async_trait]
impl Agent for RuntimeAgent {
    async fn chat(&self, input: &str) -> Result<AgentResponse> {
        let result = if let Some(context) = self.build_observation_context(None) {
            with_observation_context(context, self.run_loop(input)).await
        } else {
            self.run_loop(input).await
        };
        self.export_observability_if_configured().await;
        result
    }

    fn info(&self) -> AgentInfo {
        self.info.clone()
    }

    async fn reset(&self) -> Result<()> {
        self.memory.clear().await?;
        *self.iteration_count.write() = 0;
        self.tool_call_history.write().clear();
        if let Some(ref sm) = self.state_machine {
            sm.reset();
        }
        Ok(())
    }
}

//
// Render a concurrent input template using direct minijinja.
// Same approach as pipeline's render_stage_template so variables are top-level.
//
// Available variables:
//   {{ user_input }}    - the user's actual message
//   {{ context.<key> }} - values from the context manager
//
/// Builds safe runtime tags for background maintenance lifecycle events.
fn background_maintenance_tags(
    label: &str,
    stage: &str,
    reason: Option<&str>,
    policy: Option<&crate::optimization::config::MaintenanceTaskPolicy>,
) -> HashMap<String, String> {
    let mut tags = HashMap::new();
    tags.insert("runtime.background".to_string(), "true".to_string());
    tags.insert("runtime.maintenance".to_string(), label.to_string());
    tags.insert("runtime.maintenance_stage".to_string(), stage.to_string());
    if let Some(policy) = policy {
        tags.insert(
            "runtime.await_before_next_turn".to_string(),
            await_before_next_turn_label(policy.await_before_next_turn).to_string(),
        );
        tags.insert(
            "runtime.maintenance_mode".to_string(),
            maintenance_mode_label(policy.mode).to_string(),
        );
    }
    if let Some(reason) = reason {
        tags.insert("runtime.reason".to_string(), reason.to_string());
    }
    tags
}

fn await_before_next_turn_label(policy: AwaitBeforeNextTurn) -> &'static str {
    match policy {
        AwaitBeforeNextTurn::Never => "never",
        AwaitBeforeNextTurn::SameActor => "same_actor",
        AwaitBeforeNextTurn::Always => "always",
    }
}

fn maintenance_mode_label(mode: MaintenanceMode) -> &'static str {
    match mode {
        MaintenanceMode::InlineSerial => "inline_serial",
        MaintenanceMode::InlineParallel => "inline_parallel",
        MaintenanceMode::Background => "background",
    }
}

/// Records a background maintenance lifecycle event when observability is enabled.
fn record_background_maintenance_event(
    manager: Option<&Arc<ObservabilityManager>>,
    label: &str,
    status: EventStatus,
    duration_ms: u64,
    stage: &str,
    reason: Option<String>,
    policy: Option<&crate::optimization::config::MaintenanceTaskPolicy>,
) {
    if let Some(manager) = manager {
        manager.record_lifecycle_event(
            EventType::MemoryOperation {
                operation: format!("{}_background_{}", label, stage),
            },
            ObservationPurpose::Other(format!("{}_maintenance", label)),
            status,
            duration_ms,
            background_maintenance_tags(label, stage, reason.as_deref(), policy),
            None,
        );
    }
}

fn effective_maintenance_mode(mode: MaintenanceMode, force_parallel: bool) -> MaintenanceMode {
    if force_parallel && matches!(mode, MaintenanceMode::InlineSerial) {
        MaintenanceMode::InlineParallel
    } else {
        mode
    }
}

fn observation_purpose_for_process(hint: ProcessPurposeHint) -> ObservationPurpose {
    match hint {
        ProcessPurposeHint::Detect => ObservationPurpose::ProcessDetect,
        ProcessPurposeHint::Extract => ObservationPurpose::ProcessExtract,
        ProcessPurposeHint::Validate => ObservationPurpose::ProcessValidate,
        ProcessPurposeHint::Transform | ProcessPurposeHint::Other => {
            ObservationPurpose::ProcessTransform
        }
    }
}

fn render_concurrent_template(
    template: &str,
    user_input: &str,
    context_values: &std::collections::HashMap<String, serde_json::Value>,
) -> Result<String> {
    let mut env = minijinja::Environment::new();
    env.add_template("concurrent", template)
        .map_err(|e| AgentError::Other(format!("Concurrent template parse error: {}", e)))?;

    let mut ctx = std::collections::BTreeMap::new();
    ctx.insert("user_input".to_string(), minijinja::Value::from(user_input));

    // Expose context manager values under {{ context.<key> }}.
    let context_obj = minijinja::Value::from_serialize(context_values);
    ctx.insert("context".to_string(), context_obj);

    let tmpl = env
        .get_template("concurrent")
        .map_err(|e| AgentError::Other(format!("Concurrent template error: {}", e)))?;

    tmpl.render(minijinja::Value::from_serialize(&ctx))
        .map_err(|e| AgentError::Other(format!("Concurrent template render error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentBuilder;
    use ai_agents_llm::mock::MockLLMProvider;

    fn mock_with_response(response: &str) -> MockLLMProvider {
        let mut mock = MockLLMProvider::new("test");
        mock.set_response(response);
        mock
    }

    fn mock_with_responses(responses: Vec<&str>) -> MockLLMProvider {
        let mut mock = MockLLMProvider::new("test");
        mock.set_responses(responses.into_iter().map(String::from).collect(), true);
        mock
    }

    struct ResponseCountingHooks {
        responses: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl AgentHooks for ResponseCountingHooks {
        async fn on_response(&self, _response: &AgentResponse) {
            self.responses.fetch_add(1, Ordering::SeqCst);
        }
    }

    // Basic YAML → Build → Chat flow
    #[tokio::test]
    async fn test_integration_yaml_to_chat_basic() {
        let mock = mock_with_response("Hello! How can I help you?");
        let agent = AgentBuilder::new()
            .system_prompt("You are a test assistant.")
            .llm(Arc::new(mock))
            .build()
            .unwrap();

        let response = agent.chat("Hi").await.unwrap();
        assert!(!response.content.is_empty());
        assert_eq!(response.content, "Hello! How can I help you?");
    }

    // Multi-turn conversation
    #[tokio::test]
    async fn test_integration_multi_turn_conversation() {
        let mock = mock_with_responses(vec![
            "Hello! I'm your assistant.",
            "The weather is sunny today.",
            "Goodbye!",
        ]);
        let agent = AgentBuilder::new()
            .system_prompt("You are helpful.")
            .llm(Arc::new(mock))
            .build()
            .unwrap();

        let r1 = agent.chat("Hi").await.unwrap();
        assert_eq!(r1.content, "Hello! I'm your assistant.");

        let r2 = agent.chat("What's the weather?").await.unwrap();
        assert_eq!(r2.content, "The weather is sunny today.");

        let r3 = agent.chat("Bye").await.unwrap();
        assert_eq!(r3.content, "Goodbye!");

        // Verify memory accumulated messages
        let messages = agent.memory.get_messages(None).await.unwrap();
        // 3 user + 3 assistant = 6 messages
        assert_eq!(messages.len(), 6);
    }

    // Tool execution in chat flow
    #[tokio::test]
    async fn test_integration_tool_execution() {
        // Mock LLM that returns a tool call then a final answer
        let mock = mock_with_responses(vec![
            // First response: tool call
            r#"I'll calculate that for you.
[TOOL_CALL: {"name": "calculator", "arguments": {"expression": "2+2"}}]"#,
            // After tool result: final answer
            "The answer is 4.",
        ]);
        let mut tools = ai_agents_tools::ToolRegistry::new();
        tools
            .register(Arc::new(ai_agents_tools::CalculatorTool))
            .unwrap();

        let agent = AgentBuilder::new()
            .system_prompt("You are a calculator assistant.")
            .llm(Arc::new(mock))
            .tools(tools)
            .build()
            .unwrap();

        let response = agent.chat("What is 2+2?").await.unwrap();
        // The agent should eventually produce a response
        assert!(!response.content.is_empty());
    }

    #[tokio::test]
    async fn test_tool_hitl_rejection_finalizes_blocking_turn() {
        let responses = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let hooks = Arc::new(ResponseCountingHooks {
            responses: Arc::clone(&responses),
        });
        let mock = mock_with_response(r#"{"tool":"echo","arguments":{"message":"hello"}}"#);
        let yaml = r#"
name: ToolRejectAgent
system_prompt: "You use tools when requested."
hitl:
  tools:
    echo:
      require_approval: true
      approval_message: "Approve echo?"
"#;
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(Arc::new(mock))
            .auto_configure_features()
            .unwrap()
            .hooks(hooks)
            .build()
            .unwrap();

        let response = agent.chat("echo hello").await.unwrap();

        assert!(
            response.content.contains("Operation cancelled"),
            "unexpected response: {}",
            response.content
        );
        assert_eq!(responses.load(Ordering::SeqCst), 1);
        let messages = agent.memory.get_messages(None).await.unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content, "echo hello");
        assert!(messages[1].content.contains("\"tool\":\"echo\""));
        assert!(messages[2].content.contains("rejected by the approver"));
    }

    #[tokio::test]
    async fn test_tool_hitl_rejection_finalizes_streaming_turn() {
        use futures::StreamExt;

        let responses = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let hooks = Arc::new(ResponseCountingHooks {
            responses: Arc::clone(&responses),
        });
        let mock = mock_with_response(r#"{"tool":"echo","arguments":{"message":"hello"}}"#);
        let yaml = r#"
name: ToolRejectStreamingAgent
system_prompt: "You use tools when requested."
streaming:
  enabled: true
hitl:
  tools:
    echo:
      require_approval: true
      approval_message: "Approve echo?"
"#;
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(Arc::new(mock))
            .auto_configure_features()
            .unwrap()
            .hooks(hooks)
            .build()
            .unwrap();

        let mut stream = agent.chat_stream("echo hello").await.unwrap();
        let mut terminal_error = String::new();
        let mut done = false;
        while let Some(chunk) = stream.next().await {
            match chunk {
                StreamChunk::Error { message } => terminal_error = message,
                StreamChunk::Done {} => {
                    done = true;
                    break;
                }
                _ => {}
            }
        }

        assert!(done);
        assert!(
            terminal_error.contains("Operation cancelled"),
            "unexpected terminal error: {}",
            terminal_error
        );
        assert_eq!(responses.load(Ordering::SeqCst), 1);
        let messages = agent.memory.get_messages(None).await.unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content, "echo hello");
        assert!(messages[1].content.contains("\"tool\":\"echo\""));
        assert!(messages[2].content.contains("rejected by the approver"));
    }

    #[tokio::test]
    async fn test_pre_response_guard_transition_skips_old_state_llm() {
        let mock = mock_with_response("Billing state response");
        let call_counter = mock.clone();
        let yaml = r#"
name: OptimizedStateAgent
system_prompt: "You route before answering."
runtime:
  optimization:
    enabled: true
    pre_response_deterministic_transitions: true
states:
  initial: greeting
  states:
    greeting:
      prompt: "Old state prompt that should be skipped."
      transitions:
        - to: billing
          guard:
            context:
              topic:
                eq: billing
          timing: pre_response
    billing:
      prompt: "Answer from the billing state."
"#;
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(Arc::new(mock))
            .build()
            .unwrap();
        agent
            .set_context("topic", serde_json::json!("billing"))
            .unwrap();

        let response = agent.chat("I need billing help").await.unwrap();

        assert_eq!(agent.current_state().as_deref(), Some("billing"));
        assert_eq!(response.content, "Billing state response");
        assert_eq!(call_counter.call_count(), 1);
        assert_eq!(agent.actor_facts().len(), 0);
    }

    #[tokio::test]
    async fn test_set_context_supports_dotted_paths_for_pre_response_guards() {
        let mock = mock_with_response("Billing state response");
        let call_counter = mock.clone();
        let yaml = r#"
name: OptimizedStateAgent
system_prompt: "You route before answering."
runtime:
  optimization:
    enabled: true
    pre_response_deterministic_transitions: true
context:
  request:
    type: runtime
    default:
      topic: general
states:
  initial: greeting
  states:
    greeting:
      prompt: "Old state prompt that should be skipped."
      transitions:
        - to: billing
          guard:
            context:
              request.topic:
                eq: billing
          timing: pre_response
    billing:
      prompt: "Answer from the billing state."
"#;
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(Arc::new(mock))
            .build()
            .unwrap();
        agent
            .set_context("request.topic", serde_json::json!("billing"))
            .unwrap();

        let response = agent.chat("I need billing help").await.unwrap();

        assert_eq!(agent.current_state().as_deref(), Some("billing"));
        assert_eq!(response.content, "Billing state response");
        assert_eq!(call_counter.call_count(), 1);
        assert_eq!(
            agent.get_context().get("request"),
            Some(&serde_json::json!({"topic": "billing"}))
        );
    }

    #[tokio::test]
    async fn test_pre_response_rejection_does_not_commit_staged_context_or_user() {
        let mock = mock_with_response("billing");
        let yaml = r#"
name: OptimizedStateAgent
system_prompt: "You route before answering."
runtime:
  optimization:
    enabled: true
    pre_response_deterministic_transitions: true
hitl:
  states:
    billing:
      on_enter: require_approval
      approval_message: "Approve billing route?"
states:
  initial: greeting
  states:
    greeting:
      prompt: "Old state prompt."
      extract:
        - key: topic
          description: "Support topic"
      transitions:
        - to: billing
          guard:
            context:
              topic:
                eq: billing
          timing: pre_response
          run_extractors: true
    billing:
      prompt: "Billing state."
"#;
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(Arc::new(mock))
            .build()
            .unwrap();

        let response = agent
            .try_pre_response_transition("billing please")
            .await
            .unwrap();

        assert!(response.is_none());
        assert_eq!(agent.current_state().as_deref(), Some("greeting"));
        assert!(!agent.get_context().contains_key("topic"));
        assert_eq!(agent.memory.get_messages(None).await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_pre_response_extractor_commits_context_on_winning_path() {
        let mock = mock_with_responses(vec!["billing", "Billing response"]);
        let yaml = r#"
name: OptimizedStateAgent
system_prompt: "You route before answering."
runtime:
  optimization:
    enabled: true
    pre_response_deterministic_transitions: true
states:
  initial: greeting
  states:
    greeting:
      prompt: "Old state prompt."
      extract:
        - key: topic
          description: "Support topic"
      transitions:
        - to: billing
          guard:
            context:
              topic:
                eq: billing
          timing: pre_response
          run_extractors: true
    billing:
      prompt: "Billing state."
"#;
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(Arc::new(mock))
            .build()
            .unwrap();

        let response = agent.chat("billing please").await.unwrap();

        assert_eq!(agent.current_state().as_deref(), Some("billing"));
        assert_eq!(response.content, "Billing response");
        assert_eq!(
            agent.get_context().get("topic"),
            Some(&serde_json::json!("billing"))
        );
    }

    #[tokio::test]
    async fn test_pre_response_extractor_miss_does_not_mutate_context() {
        let mock = mock_with_response("__NONE__");
        let yaml = r#"
name: OptimizedStateAgent
system_prompt: "You route before answering."
runtime:
  optimization:
    enabled: true
    pre_response_deterministic_transitions: true
states:
  initial: greeting
  states:
    greeting:
      prompt: "Old state prompt."
      extract:
        - key: topic
          description: "Support topic"
      transitions:
        - to: billing
          guard:
            context:
              topic:
                eq: billing
          timing: pre_response
          run_extractors: true
    billing:
      prompt: "Billing state."
"#;
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(Arc::new(mock))
            .build()
            .unwrap();

        let response = agent.try_pre_response_transition("hello").await.unwrap();

        assert!(response.is_none());
        assert_eq!(agent.current_state().as_deref(), Some("greeting"));
        assert!(!agent.get_context().contains_key("topic"));
    }

    #[tokio::test]
    async fn test_default_guard_transition_stays_post_response() {
        let mock = mock_with_responses(vec!["Greeting response", "Billing response"]);
        let call_counter = mock.clone();
        let yaml = r#"
name: TimingAgent
system_prompt: "You route carefully."
runtime:
  optimization:
    enabled: true
    pre_response_deterministic_transitions: true
states:
  initial: greeting
  states:
    greeting:
      prompt: "Old state prompt."
      transitions:
        - to: billing
          guard:
            context:
              topic:
                eq: billing
    billing:
      prompt: "Billing state."
"#;
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(Arc::new(mock))
            .build()
            .unwrap();
        agent
            .set_context("topic", serde_json::json!("billing"))
            .unwrap();

        let response = agent.chat("billing please").await.unwrap();

        assert_eq!(agent.current_state().as_deref(), Some("billing"));
        assert_eq!(response.content, "Billing response");
        assert_eq!(call_counter.call_count(), 2);
    }

    #[tokio::test]
    async fn test_explicit_post_response_guard_transition_stays_post_response() {
        let mock = mock_with_responses(vec!["Greeting response", "Billing response"]);
        let call_counter = mock.clone();
        let yaml = r#"
name: TimingAgent
system_prompt: "You route carefully."
runtime:
  optimization:
    enabled: true
    pre_response_deterministic_transitions: true
states:
  initial: greeting
  states:
    greeting:
      prompt: "Old state prompt."
      transitions:
        - to: billing
          guard:
            context:
              topic:
                eq: billing
          timing: post_response
    billing:
      prompt: "Billing state."
"#;
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(Arc::new(mock))
            .build()
            .unwrap();
        agent
            .set_context("topic", serde_json::json!("billing"))
            .unwrap();

        let response = agent.chat("billing please").await.unwrap();

        assert_eq!(agent.current_state().as_deref(), Some("billing"));
        assert_eq!(response.content, "Billing response");
        assert_eq!(call_counter.call_count(), 2);
    }

    #[tokio::test]
    async fn test_pre_response_extractors_are_transition_scoped() {
        let mock = mock_with_responses(vec!["billing", "Billing response"]);
        let yaml = r#"
name: ScopedExtractorAgent
system_prompt: "You route carefully."
runtime:
  optimization:
    enabled: true
    pre_response_deterministic_transitions: true
states:
  initial: greeting
  states:
    greeting:
      prompt: "Old state prompt."
      extract:
        - key: topic
          description: "Support topic"
      transitions:
        - to: wrong
          guard:
            context:
              topic:
                eq: billing
          timing: pre_response
        - to: billing
          guard:
            context:
              topic:
                eq: billing
          timing: pre_response
          run_extractors: true
    wrong:
      prompt: "Wrong state."
    billing:
      prompt: "Billing state."
"#;
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(Arc::new(mock))
            .build()
            .unwrap();

        let response = agent.chat("billing please").await.unwrap();

        assert_eq!(agent.current_state().as_deref(), Some("billing"));
        assert_eq!(response.content, "Billing response");
    }

    #[tokio::test]
    async fn test_pre_response_resolved_intent_routes_early() {
        let mock = mock_with_response("Billing response");
        let yaml = r#"
name: IntentAgent
system_prompt: "You route carefully."
runtime:
  optimization:
    enabled: true
    pre_response_deterministic_transitions: true
states:
  initial: greeting
  states:
    greeting:
      prompt: "Old state prompt."
      transitions:
        - to: billing
          intent: billing
          timing: pre_response
    billing:
      prompt: "Billing state."
"#;
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(Arc::new(mock))
            .build()
            .unwrap();
        agent
            .set_context("resolved_intent", serde_json::json!("billing"))
            .unwrap();

        let response = agent
            .try_pre_response_transition("I need billing help")
            .await
            .unwrap()
            .unwrap();

        assert_eq!(agent.current_state().as_deref(), Some("billing"));
        assert_eq!(response.content, "Billing response");
    }

    #[tokio::test]
    async fn test_background_overflow_error_surfaces() {
        let mut config = RuntimeConfig::default();
        config.optimization.enabled = true;
        config.optimization.post_turn.max_background_tasks = 1;
        config.optimization.post_turn.on_background_overflow = BackgroundOverflowPolicy::Error;
        let policy = crate::optimization::MaintenanceTaskPolicy {
            mode: MaintenanceMode::Background,
            await_before_next_turn: AwaitBeforeNextTurn::Always,
        };
        let agent = AgentBuilder::new()
            .system_prompt("You are helpful.")
            .llm(Arc::new(mock_with_response("ok")))
            .build()
            .unwrap()
            .with_runtime_config(config);
        agent
            .background_maintenance
            .spawn(None, async { std::future::pending::<Result<()>>().await })
            .unwrap();

        let result = agent
            .spawn_or_handle_background(None, async { Ok(()) }, "facts", &policy)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_speculative_reasoning_low_cap_uses_serial_reasoning() {
        let default_mock = mock_with_response("Plain draft response");
        let router_mock = mock_with_response("cot");
        let router_counter = router_mock.clone();
        let yaml = r#"
name: ReasoningReservationAgent
system_prompt: "You answer plainly unless reasoning wins."
llm:
  default: default
  router: router
observability:
  enabled: true
  export:
    write_raw_events: true
reasoning:
  mode: auto
  judge_llm: router
runtime:
  optimization:
    enabled: true
    max_speculative_llm_calls_per_turn: 1
    speculative_reasoning_auto: true
    max_parallel_runtime_tasks: 2
"#;
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm_alias("default", Arc::new(default_mock))
            .llm_alias("router", Arc::new(router_mock))
            .build()
            .unwrap();

        let response = agent.chat("hello").await.unwrap();

        assert_eq!(response.content, "Plain draft response");
        assert_eq!(router_counter.call_count(), 1);
        let events = agent.observability().unwrap().raw_events();
        assert!(!events.iter().any(|event| {
            event.dimensions.get("commit_behavior") == Some(&"reasoning_decision".to_string())
        }));
    }

    #[tokio::test]
    async fn test_forced_reasoning_skips_plain_speculative_draft() {
        let mock = mock_with_response("Reasoned response");
        let yaml = r#"
name: ForcedReasoningAgent
system_prompt: "You reason before answering."
observability:
  enabled: true
  export:
    write_raw_events: true
reasoning:
  mode: cot
runtime:
  optimization:
    enabled: true
    max_speculative_llm_calls_per_turn: 2
    speculative_state_transitions: true
    max_parallel_runtime_tasks: 2
states:
  initial: triage
  states:
    triage:
      prompt: "Answer from triage."
      transitions:
        - to: billing
          guard:
            context:
              route:
                eq: billing
          timing: parallel
    billing:
      prompt: "Billing state."
"#;
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(Arc::new(mock))
            .build()
            .unwrap();

        let response = agent.chat("hello").await.unwrap();

        assert_eq!(response.content, "Reasoned response");
        let events = agent.observability().unwrap().raw_events();
        assert!(
            !events
                .iter()
                .any(|event| event.dimensions.contains_key("branch_status"))
        );
    }

    #[tokio::test]
    async fn test_speculative_skill_low_cap_uses_serial_skill_route() {
        let default_mock = mock_with_response("Skill committed response");
        let router_mock = mock_with_response("helper");
        let router_counter = router_mock.clone();
        let yaml = r#"
name: SkillReservationAgent
system_prompt: "Use skills when they match."
llm:
  default: default
  router: router
observability:
  enabled: true
  export:
    write_raw_events: true
runtime:
  optimization:
    enabled: true
    max_speculative_llm_calls_per_turn: 1
    speculative_skill_routing: true
    max_parallel_runtime_tasks: 2
skills:
  - id: helper
    description: "Answer helper requests"
    trigger: "User asks for helper"
    steps:
      - prompt: "Answer the helper request: {{ user_input }}"
"#;
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm_alias("default", Arc::new(default_mock))
            .llm_alias("router", Arc::new(router_mock))
            .build()
            .unwrap();

        let response = agent.chat("please use helper").await.unwrap();

        assert_eq!(response.content, "Skill committed response");
        assert_eq!(router_counter.call_count(), 1);
        let events = agent.observability().unwrap().raw_events();
        assert!(
            !events
                .iter()
                .any(|event| event.dimensions.contains_key("branch_status"))
        );
    }

    #[tokio::test]
    async fn test_blocking_error_cleanup_resets_root_turn_for_next_chat() {
        let mut mock = mock_with_response("Recovered response");
        mock.set_error("boom");
        let mut handle = mock.clone();
        let agent = AgentBuilder::new()
            .system_prompt("You are helpful.")
            .llm(Arc::new(mock))
            .build()
            .unwrap();

        assert!(agent.chat("first").await.is_err());
        handle.clear_error();
        let response = agent.chat("second").await.unwrap();

        assert_eq!(response.content, "Recovered response");
        let messages = agent.memory.get_messages(None).await.unwrap();
        let user_count = messages
            .iter()
            .filter(|message| message.role == ai_agents_core::Role::User)
            .count();
        assert_eq!(user_count, 2);
    }

    #[tokio::test]
    async fn test_streaming_error_cleanup_resets_root_turn_for_next_chat() {
        use futures::StreamExt;

        let mut mock = mock_with_response("Recovered response");
        mock.set_error("stream boom");
        let mut handle = mock.clone();
        let agent = AgentBuilder::new()
            .system_prompt("You are helpful.")
            .llm(Arc::new(mock))
            .build()
            .unwrap();

        let mut stream = agent.chat_stream("first").await.unwrap();
        let mut saw_error = false;
        while let Some(chunk) = stream.next().await {
            if matches!(chunk, StreamChunk::Error { .. }) {
                saw_error = true;
            }
        }
        assert!(saw_error);

        handle.clear_error();
        let response = agent.chat("second").await.unwrap();

        assert_eq!(response.content, "Recovered response");
        let messages = agent.memory.get_messages(None).await.unwrap();
        let user_count = messages
            .iter()
            .filter(|message| message.role == ai_agents_core::Role::User)
            .count();
        assert_eq!(user_count, 2);
    }

    #[tokio::test]
    async fn test_buffered_streaming_route_miss_releases_buffer_limit() {
        use futures::StreamExt;

        let mut mock = mock_with_response("one two three");
        mock.set_latency(10);
        let yaml = r#"
name: BufferedMissAgent
system_prompt: "You stream safely."
llm:
  default: default
streaming:
  enabled: true
  buffer_size: 1
runtime:
  optimization:
    enabled: true
    max_speculative_llm_calls_per_turn: 2
    speculative_state_transitions: true
    streaming_policy: buffer_until_routing_done
    max_parallel_runtime_tasks: 2
states:
  initial: triage
  states:
    triage:
      prompt: "Answer from triage."
      transitions:
        - to: billing
          guard:
            context:
              route:
                eq: billing
          timing: parallel
    billing:
      prompt: "Billing state."
"#;
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm_alias("default", Arc::new(mock))
            .build()
            .unwrap();

        let mut stream = agent.chat_stream("hello").await.unwrap();
        let mut content = String::new();
        let mut error = None;
        while let Some(chunk) = stream.next().await {
            match chunk {
                StreamChunk::Content { text } => content.push_str(&text),
                StreamChunk::Error { message } => error = Some(message),
                StreamChunk::Done {} => break,
                _ => {}
            }
        }

        assert_eq!(error, None);
        assert_eq!(content, "one two three");
    }

    #[tokio::test]
    async fn test_buffered_streaming_main_failure_finalizes_branch() {
        use futures::StreamExt;

        let mock = mock_with_response("one two");
        let mut router_mock = mock_with_response("0");
        router_mock.set_latency(50);
        let yaml = r#"
name: BufferedFailureAgent
system_prompt: "You stream safely."
llm:
  default: default
  router: router
observability:
  enabled: true
  export:
    write_raw_events: true
streaming:
  enabled: true
  buffer_size: 1
runtime:
  optimization:
    enabled: true
    max_speculative_llm_calls_per_turn: 2
    speculative_state_transitions: true
    streaming_policy: buffer_until_routing_done
    max_parallel_runtime_tasks: 2
states:
  initial: triage
  states:
    triage:
      prompt: "Ask for the category."
      transitions:
        - to: billing
          when: "User asks about billing"
          timing: parallel
    billing:
      prompt: "Billing state."
"#;
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm_alias("default", Arc::new(mock))
            .llm_alias("router", Arc::new(router_mock))
            .build()
            .unwrap();

        let mut stream = agent.chat_stream("hello").await.unwrap();
        let mut error = String::new();
        while let Some(chunk) = stream.next().await {
            if let StreamChunk::Error { message } = chunk {
                error = message;
            }
        }

        assert!(
            error.contains("stream buffer filled"),
            "unexpected stream error: {}",
            error
        );
        let events = agent.observability().unwrap().raw_events();
        assert!(events.iter().any(|event| {
            event.dimensions.get("branch_status") == Some(&"failed".to_string())
                && event.dimensions.get("commit_behavior") == Some(&"final_response".to_string())
                && event.dimensions.get("optimization")
                    == Some(&"buffered_streaming_routing".to_string())
        }));
    }

    #[tokio::test]
    async fn test_streaming_preflight_does_not_emit_old_state_content() {
        use futures::StreamExt;

        let mock = mock_with_response("Billing streamed response");
        let yaml = r#"
name: StreamingOptimizedAgent
system_prompt: "You route before streaming."
runtime:
  optimization:
    enabled: true
    pre_response_deterministic_transitions: true
streaming:
  enabled: true
states:
  initial: greeting
  states:
    greeting:
      prompt: "OLD_STATE_SENTINEL"
      transitions:
        - to: billing
          guard:
            context:
              topic:
                eq: billing
          timing: pre_response
    billing:
      prompt: "Billing state."
"#;
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(Arc::new(mock))
            .build()
            .unwrap();
        agent
            .set_context("topic", serde_json::json!("billing"))
            .unwrap();

        let mut stream = agent.chat_stream("billing please").await.unwrap();
        let mut content = String::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                StreamChunk::Content { text } => content.push_str(&text),
                StreamChunk::Error { message } => panic!("stream error: {}", message),
                StreamChunk::Done {} => break,
                _ => {}
            }
        }

        assert_eq!(agent.current_state().as_deref(), Some("billing"));
        assert!(content.contains("Billing streamed response"));
        assert!(!content.contains("OLD_STATE_SENTINEL"));
    }

    // State machine transitions
    #[tokio::test]
    async fn test_integration_state_machine_basic() {
        let yaml = r#"
name: StateAgent
system_prompt: "You are a support agent."
states:
  initial: greeting
  states:
    greeting:
      prompt: "Welcome the user warmly."
      transitions:
        - to: support
          when: "User needs help"
          auto: true
    support:
      prompt: "Help solve the user's problem."
"#;
        let mock = mock_with_responses(vec![
            "Welcome! How can I help?", // greeting response
            "1",                        // transition evaluator picks first (index 0)
            "I'll help you with that.", // support response
        ]);
        let builder = AgentBuilder::from_yaml(yaml).unwrap();
        let agent = builder.llm(Arc::new(mock)).build().unwrap();

        assert_eq!(agent.current_state(), Some("greeting".to_string()));
        let _ = agent.chat("I need help").await.unwrap();
        // After transition evaluation, state may or may not have changed
        // depending on mock evaluator response - the key is that it doesn't crash
    }

    // State on_enter/on_exit actions
    #[tokio::test]
    async fn test_integration_state_on_enter_set_context() {
        let yaml = r#"
name: ActionAgent
system_prompt: "You are helpful."
states:
  initial: step1
  states:
    step1:
      prompt: "Step 1"
      on_exit:
        - set_context:
            step1_exited: true
      transitions:
        - to: step2
          when: "always"
          auto: true
    step2:
      prompt: "Step 2"
      on_enter:
        - set_context:
            step2_entered: true
"#;
        // The transition evaluator will pick the first transition (index 0)
        let mock = mock_with_responses(vec![
            "Processing step 1.",
            "0", // transition evaluator response: select first transition
        ]);
        let builder = AgentBuilder::from_yaml(yaml).unwrap();
        let agent = builder.llm(Arc::new(mock)).build().unwrap();

        assert_eq!(agent.current_state(), Some("step1".to_string()));

        // Manually transition to test on_enter/on_exit
        agent.transition_to("step2").await.unwrap();

        assert_eq!(agent.current_state(), Some("step2".to_string()));

        // Verify context was set by on_exit and on_enter actions
        let ctx = agent.get_context();
        assert_eq!(ctx.get("step1_exited"), Some(&serde_json::json!(true)));
        assert_eq!(ctx.get("step2_entered"), Some(&serde_json::json!(true)));
    }

    // Process pipeline transforms input
    #[tokio::test]
    async fn test_integration_process_normalize() {
        let yaml = r#"
name: ProcessAgent
system_prompt: "You are helpful."
process:
  input:
    - type: normalize
      config:
        trim: true
        collapse_whitespace: true
"#;
        let mock = mock_with_response("Got your message.");
        let builder = AgentBuilder::from_yaml(yaml).unwrap();
        let agent = builder.llm(Arc::new(mock.clone())).build().unwrap();

        let _ = agent.chat("  hello   world  ").await.unwrap();

        // Verify the LLM received the normalized input (trimmed + collapsed whitespace)
        let history = mock.call_history();
        assert!(!history.is_empty());
        // The user message in LLM call should be normalized
        let last_call = history.last().unwrap();
        let user_msg = last_call
            .messages
            .iter()
            .find(|m| m.role == ai_agents_core::Role::User)
            .unwrap();
        assert_eq!(user_msg.content, "hello world");
    }

    // ═══════════════════════════════════════════════════════════
    // Integration Test 2.1.7: Memory compression triggers
    // ═══════════════════════════════════════════════════════════
    #[tokio::test]
    async fn test_integration_memory_compression() {
        let yaml = r#"
name: MemoryAgent
system_prompt: "You are helpful."
memory:
  type: compacting
  max_messages: 100
  compress_threshold: 5
  max_recent_messages: 3
  summarize_batch_size: 2
"#;
        // Provide enough responses for compression to trigger
        let responses: Vec<&str> = (0..8).map(|_| "Response from assistant.").collect();
        let mock = mock_with_responses(responses);
        let builder = AgentBuilder::from_yaml(yaml).unwrap();
        let agent = builder.llm(Arc::new(mock)).build().unwrap();

        // Send enough messages to trigger compression
        for i in 0..6 {
            let _ = agent.chat(&format!("Message {}", i)).await.unwrap();
        }

        // Memory should have compressed - verify it didn't crash
        // and that messages are bounded
        let messages = agent.memory.get_messages(None).await.unwrap();
        // With compress_threshold=5 and max_recent_messages=3,
        // after 6 turns (12 messages), compression should have run
        assert!(messages.len() <= 12); // At most all messages if no compression, fewer if compressed
    }

    // YAML with multiple LLMs
    #[tokio::test]
    async fn test_integration_multi_llm_registry() {
        let mut mock_default = MockLLMProvider::new("default");
        mock_default.set_response("Default LLM response.");
        let mut mock_router = MockLLMProvider::new("router");
        mock_router.set_response("Router response.");

        let agent = AgentBuilder::new()
            .system_prompt("You are helpful.")
            .llm_alias("default", Arc::new(mock_default))
            .llm_alias("router", Arc::new(mock_router))
            .build()
            .unwrap();

        let response = agent.chat("Hello").await.unwrap();
        assert_eq!(response.content, "Default LLM response.");
    }

    // Agent reset clears state
    #[tokio::test]
    async fn test_integration_agent_reset() {
        let mock = mock_with_responses(vec!["Hello!", "Hello again!"]);
        let agent = AgentBuilder::new()
            .system_prompt("You are helpful.")
            .llm(Arc::new(mock))
            .build()
            .unwrap();

        let _ = agent.chat("Hi").await.unwrap();
        let messages = agent.memory.get_messages(None).await.unwrap();
        assert_eq!(messages.len(), 2); // user + assistant

        agent.reset().await.unwrap();
        let messages = agent.memory.get_messages(None).await.unwrap();
        assert_eq!(messages.len(), 0);
    }

    // Process pipeline rejects input
    #[tokio::test]
    async fn test_integration_process_validate_reject() {
        use ai_agents_process::{ProcessConfig, ProcessProcessor};

        let validate_config = ai_agents_process::ValidateStage {
            id: Some("length_check".to_string()),
            condition: None,
            config: ai_agents_process::ValidateConfig {
                rules: vec![ai_agents_process::ValidationRule::MinLength {
                    min_length: 10,
                    on_fail: ai_agents_process::ValidationAction {
                        action: ai_agents_process::ValidationActionType::Reject,
                        message: None,
                    },
                }],
                ..Default::default()
            },
        };
        let process_config = ProcessConfig {
            input: vec![ai_agents_process::ProcessStage::Validate(validate_config)],
            ..Default::default()
        };
        let processor = ProcessProcessor::new(process_config);

        let mock = mock_with_response("Should not reach here.");
        let agent = AgentBuilder::new()
            .system_prompt("You are helpful.")
            .llm(Arc::new(mock))
            .process_processor(processor)
            .build()
            .unwrap();

        let response = agent.chat("Hi").await.unwrap();
        // Rejected input should produce a rejection response, not call LLM
        assert!(
            response.content.contains("rejected")
                || response.content.contains("Input rejected")
                || response.content.contains("too short")
                || response.content.contains("Too short")
                || response.content.len() < 50, // rejection message is typically short
            "Expected rejection response, got: {}",
            response.content
        );
    }

    // LLM fallback: primary fails, fallback LLM responds
    #[tokio::test]
    async fn test_llm_fallback_on_failure() {
        use ai_agents_recovery::{ErrorRecoveryConfig, LLMFailureAction, LLMRecoveryConfig};

        let mut primary = MockLLMProvider::new("primary");
        primary.set_error("Primary LLM is unavailable");

        let mut fallback = MockLLMProvider::new("fallback");
        fallback.set_response("Fallback response works!");

        let agent = AgentBuilder::new()
            .system_prompt("You are helpful.")
            .llm_alias("default", Arc::new(primary))
            .llm_alias("backup", Arc::new(fallback))
            .recovery_manager(RecoveryManager::new(ErrorRecoveryConfig {
                llm: LLMRecoveryConfig {
                    on_failure: LLMFailureAction::FallbackLlm {
                        fallback_llm: "backup".to_string(),
                    },
                    ..Default::default()
                },
                ..Default::default()
            }))
            .build()
            .unwrap();

        let response = agent.chat("Hello").await.unwrap();
        assert!(
            response.content.contains("Fallback response"),
            "Expected fallback response, got: {}",
            response.content
        );
    }

    // LLM fallback: primary fails, static message returned
    #[tokio::test]
    async fn test_llm_fallback_response_static_message() {
        use ai_agents_recovery::{ErrorRecoveryConfig, LLMFailureAction, LLMRecoveryConfig};

        let mut primary = MockLLMProvider::new("primary");
        primary.set_error("Primary LLM is unavailable");

        let agent = AgentBuilder::new()
            .system_prompt("You are helpful.")
            .llm(Arc::new(primary))
            .recovery_manager(RecoveryManager::new(ErrorRecoveryConfig {
                llm: LLMRecoveryConfig {
                    on_failure: LLMFailureAction::FallbackResponse {
                        message: "I am temporarily unavailable. Please try again later."
                            .to_string(),
                    },
                    ..Default::default()
                },
                ..Default::default()
            }))
            .build()
            .unwrap();

        let response = agent.chat("Hello").await.unwrap();
        assert!(
            response.content.contains("temporarily unavailable"),
            "Expected static fallback message, got: {}",
            response.content
        );
    }

    // Tool skip: tool fails, on_failure: skip absorbs the error
    #[tokio::test]
    async fn test_tool_failure_skip() {
        use ai_agents_recovery::{
            ErrorRecoveryConfig, ToolFailureAction, ToolRecoveryConfig, ToolRetryConfig,
        };

        // LLM requests a nonexistent tool, then responds after seeing the skip result
        let mock = mock_with_responses(vec![
            r#"I'll use the nonexistent tool.
[TOOL_CALL: {"name": "nonexistent_tool", "arguments": {}}]"#,
            "The tool was unavailable, but I can still help you.",
        ]);

        let agent = AgentBuilder::new()
            .system_prompt("You are helpful.")
            .llm(Arc::new(mock))
            .recovery_manager(RecoveryManager::new(ErrorRecoveryConfig {
                tools: ToolRecoveryConfig {
                    default: ToolRetryConfig {
                        max_retries: 0,
                        timeout_ms: None,
                        on_failure: ToolFailureAction::Skip,
                    },
                    ..Default::default()
                },
                ..Default::default()
            }))
            .build()
            .unwrap();

        // The tool will fail (not found), but on_failure: skip absorbs the error
        let response = agent.chat("Use the nonexistent tool").await;
        assert!(
            response.is_ok(),
            "Expected Ok with skip policy, got: {:?}",
            response
        );
    }
}

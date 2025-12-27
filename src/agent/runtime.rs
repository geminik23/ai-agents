use async_trait::async_trait;
use futures::stream::{self, Stream, StreamExt};
use parking_lot::RwLock;
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, error, info, instrument, warn};

use super::{
    Agent, AgentInfo, AgentResponse, ParallelToolsConfig, StreamChunk, StreamingConfig, ToolCall,
};
use crate::context::{ContextManager, ContextProvider, TemplateRenderer};
use crate::error::{AgentError, Result};
use crate::hitl::{
    ApprovalHandler, ApprovalResult, HITLCheckResult, HITLEngine, RejectAllHandler, TimeoutAction,
};
use crate::hooks::{AgentHooks, NoopHooks};
use crate::llm::{ChatMessage, LLMProvider, LLMRegistry};
use crate::memory::Memory;
use crate::persistence::{AgentSnapshot, AgentStorage};
use crate::process::{ProcessData, ProcessProcessor};
use crate::recovery::{
    ByRoleFilter, ContextOverflowAction, FilterConfig, IntoClassifiedError, KeepRecentFilter,
    MessageFilter, RecoveryManager, SkipPatternFilter,
};
use crate::skill::{SkillDefinition, SkillExecutor, SkillRouter};
use crate::state::{
    PromptMode, StateMachine, StateMachineSnapshot, StateTransitionEvent, ToolRef,
    TransitionContext, TransitionEvaluator,
};
use crate::tool_security::{SecurityCheckResult, ToolSecurityEngine};
use crate::tools::{
    ConditionEvaluator, EvaluationContext, LLMGetter, ToolCallRecord, ToolRegistry, ToolResult,
};

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
            .finish_non_exhaustive()
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
            max_context_tokens: 4096,
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
        }
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

    pub fn with_recovery_manager(mut self, manager: RecoveryManager) -> Self {
        self.recovery_manager = manager;
        self
    }

    pub fn with_tool_security(mut self, engine: ToolSecurityEngine) -> Self {
        self.tool_security = engine;
        self
    }

    pub fn with_process_processor(mut self, processor: ProcessProcessor) -> Self {
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
        self.context_manager.set(key, value)
    }

    pub fn update_context(&self, path: &str, value: Value) -> Result<()> {
        self.context_manager.update(path, value)
    }

    pub fn get_context(&self) -> HashMap<String, Value> {
        self.context_manager.get_all()
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

    pub fn transition_to(&self, state: &str) -> Result<()> {
        if let Some(ref sm) = self.state_machine {
            sm.transition_to(state, "manual transition")?;
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

    pub async fn save_state(&self) -> Result<AgentSnapshot> {
        let memory_snapshot = self.memory.snapshot().await?;
        let state_machine_snapshot = self.state_machine.as_ref().map(|sm| sm.snapshot());
        let context_snapshot = self.context_manager.snapshot();

        Ok(AgentSnapshot::new(self.info.id.clone())
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
            ))
    }

    pub async fn restore_state(&self, snapshot: AgentSnapshot) -> Result<()> {
        self.memory.restore(snapshot.memory).await?;

        if let (Some(sm), Some(sm_snapshot)) = (&self.state_machine, snapshot.state_machine) {
            if !sm_snapshot.current_state.is_empty() {
                sm.restore(sm_snapshot)?;
            }
        }

        self.context_manager.restore(snapshot.context);
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
        let response = summarizer.complete(&summary_msgs, None).await?;

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
        let context = self.context_manager.get_all();
        self.template_renderer
            .render(&self.base_system_prompt, &context)
    }

    async fn get_available_tool_ids(&self) -> Result<Vec<String>> {
        let tool_refs = self.get_current_tool_refs();

        if tool_refs.is_empty() {
            return Ok(self.tools.list_ids());
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

    fn get_current_tool_refs(&self) -> Vec<ToolRef> {
        if let Some(ref sm) = self.state_machine {
            if let Some(state_def) = sm.current_definition() {
                if !state_def.tools.is_empty() {
                    let parent_def = sm.get_parent_definition();
                    return state_def
                        .get_effective_tools(parent_def.as_ref())
                        .into_iter()
                        .cloned()
                        .collect();
                }
            }
        }
        Vec::new()
    }

    async fn build_evaluation_context(&self) -> Result<EvaluationContext> {
        let context = self.context_manager.get_all();
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

    async fn get_effective_system_prompt(&self) -> Result<String> {
        let rendered_base = self.render_system_prompt()?;

        if let Some(ref sm) = self.state_machine {
            if let Some(state_def) = sm.current_definition() {
                let state_prompt = if let Some(ref prompt) = state_def.prompt {
                    let context = self.context_manager.get_all();
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

                let available_tool_ids = self.get_available_tool_ids().await?;
                let tools_prompt = if available_tool_ids.is_empty() {
                    self.tools.generate_tools_prompt()
                } else {
                    self.tools.generate_filtered_prompt(&available_tool_ids)
                };

                if !tools_prompt.is_empty() {
                    return Ok(format!("{}\n\n{}", combined, tools_prompt));
                }
                return Ok(combined);
            }
        }

        let tools_prompt = self.tools.generate_tools_prompt();
        if !tools_prompt.is_empty() {
            Ok(format!("{}\n\n{}", rendered_base, tools_prompt))
        } else {
            Ok(rendered_base)
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
        let system_prompt = self.get_effective_system_prompt().await?;
        let mut messages = vec![ChatMessage::system(&system_prompt)];
        let history = self.memory.get_messages(None).await?;
        messages.extend(history);

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
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(content) {
            if let Some(tool_name) = parsed.get("tool").and_then(|v| v.as_str()) {
                let arguments = parsed
                    .get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::json!({}));
                return Some(vec![ToolCall {
                    id: uuid::Uuid::new_v4().to_string(),
                    name: tool_name.to_string(),
                    arguments,
                }]);
            }
        }
        None
    }

    async fn execute_tool(&self, tool_call: &ToolCall) -> Result<String> {
        let tool = self
            .tools
            .get(&tool_call.name)
            .ok_or_else(|| AgentError::Tool(format!("Tool not found: {}", tool_call.name)))?;

        let result = tool.execute(tool_call.arguments.clone()).await;

        if result.success {
            Ok(result.output)
        } else {
            Err(AgentError::Tool(result.output))
        }
    }

    #[instrument(skip(self, tool_call), fields(tool = %tool_call.name))]
    async fn execute_tool_smart(&self, tool_call: &ToolCall) -> Result<String> {
        info!(args = %tool_call.arguments, "Executing tool");

        self.hooks
            .on_tool_start(&tool_call.name, &tool_call.arguments)
            .await;
        let tool_start = Instant::now();

        let available_tool_ids = self.get_available_tool_ids().await?;
        if !available_tool_ids.is_empty() && !available_tool_ids.contains(&tool_call.name) {
            warn!(tool = %tool_call.name, "Tool not available in current context");
            return Err(AgentError::Tool(format!(
                "Tool '{}' is not available in current context",
                tool_call.name
            )));
        }

        // Check HITL approval for tool
        if let Some(ref hitl_engine) = self.hitl_engine {
            let check_result = hitl_engine.check_tool(&tool_call.name, &tool_call.arguments);
            if check_result.is_required() {
                let approved = self.request_hitl_approval(check_result).await?;
                if !approved {
                    warn!(tool = %tool_call.name, "Tool execution rejected by HITL");
                    return Err(AgentError::HITLRejected(format!(
                        "Tool '{}' was rejected by human approver. Do not retry.",
                        tool_call.name
                    )));
                }
            }

            // Check conditions (e.g., amount > 1000)
            let condition_check = hitl_engine.check_conditions(&tool_call.arguments);
            if condition_check.is_required() {
                let approved = self.request_hitl_approval(condition_check).await?;
                if !approved {
                    warn!(tool = %tool_call.name, "Tool execution rejected by HITL condition");
                    return Err(AgentError::HITLRejected(format!(
                        "Tool '{}' was rejected due to policy condition. Do not retry.",
                        tool_call.name
                    )));
                }
            }
        }

        if self.tool_security.config().enabled {
            let security_result = self
                .tool_security
                .check_tool_execution(&tool_call.name, &tool_call.arguments)
                .await?;

            match security_result {
                SecurityCheckResult::Allow => {}
                SecurityCheckResult::Block { reason } => {
                    warn!(reason = %reason, "Tool blocked");
                    return Err(AgentError::Tool(format!("Blocked: {}", reason)));
                }
                SecurityCheckResult::RequireConfirmation { message } => {
                    warn!(message = %message, "Tool requires confirmation");
                    return Err(AgentError::Tool(format!(
                        "Confirmation required: {}",
                        message
                    )));
                }
                SecurityCheckResult::Warn { message } => {
                    warn!(message = %message, "Tool security warning");
                }
            }
        }

        let tool_config = self.recovery_manager.get_tool_config(&tool_call.name);

        let result = if tool_config.max_retries > 0 {
            let retry_config = crate::recovery::RetryConfig {
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
            self.execute_tool(tool_call).await
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

    async fn try_skill_route(&self, input: &str) -> Result<Option<String>> {
        if let Some(ref router) = self.skill_router {
            let available_skills = self.get_available_skills();
            let skill_ids: Vec<&str> = available_skills.iter().map(|s| s.id.as_str()).collect();

            if let Some(skill_id) = router.select_skill_filtered(input, &skill_ids).await? {
                info!(skill_id = %skill_id, "Skill selected");

                let skill = router
                    .get_skill(&skill_id)
                    .ok_or_else(|| AgentError::Skill(format!("Skill not found: {}", skill_id)))?;

                if let Some(ref executor) = self.skill_executor {
                    let response = executor
                        .execute(skill, input, serde_json::json!({}))
                        .await?;
                    return Ok(Some(response));
                }
            }
        }
        Ok(None)
    }

    async fn process_input(&self, input: &str) -> Result<ProcessData> {
        if let Some(ref processor) = self.process_processor {
            processor.process_input(input).await
        } else {
            Ok(ProcessData::new(input))
        }
    }

    async fn process_output(
        &self,
        output: &str,
        input_context: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Result<ProcessData> {
        if let Some(ref processor) = self.process_processor {
            processor.process_output(output, input_context).await
        } else {
            Ok(ProcessData::new(output))
        }
    }

    async fn check_turn_timeout(&self) -> Result<()> {
        if let Some(ref sm) = self.state_machine {
            if let Some(timeout_state) = sm.check_timeout() {
                sm.transition_to(&timeout_state, "max_turns exceeded")?;
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

    async fn evaluate_transitions(&self, user_message: &str, response: &str) -> Result<bool> {
        let (transitions, evaluator, current_state) =
            match (&self.state_machine, &self.transition_evaluator) {
                (Some(sm), Some(eval)) => {
                    let auto_transitions = sm.auto_transitions();
                    if auto_transitions.is_empty() {
                        return Ok(false);
                    }
                    (auto_transitions, eval, sm.current())
                }
                _ => return Ok(false),
            };

        let context_map = self.context_manager.get_all();
        let context = TransitionContext::new(user_message, response, &current_state)
            .with_context(context_map);

        if let Some(index) = evaluator.select_transition(&transitions, &context).await? {
            let target = transitions[index].to.clone();
            let reason = if transitions[index].when.is_empty() {
                "guard condition met".to_string()
            } else {
                transitions[index].when.clone()
            };

            if let Some(ref sm) = self.state_machine {
                // Check HITL approval for state transition
                let approved = self
                    .check_state_hitl(Some(&context.current_state), &target)
                    .await?;
                if !approved {
                    info!(to = %target, "State transition rejected by HITL");
                    return Ok(false);
                }

                sm.transition_to(&target, &reason)?;
                sm.reset_no_transition();
                self.hooks
                    .on_state_transition(Some(&context.current_state), &target, &reason)
                    .await;
                info!(from = %context.current_state, to = %target, "State transition");
            }
            return Ok(true);
        }

        if let Some(ref sm) = self.state_machine {
            sm.increment_no_transition();
            if let Some(fallback) = sm.check_fallback() {
                let from_state = current_state.clone();

                // Check HITL approval for fallback transition
                let approved = self.check_state_hitl(Some(&from_state), &fallback).await?;
                if !approved {
                    info!(to = %fallback, "Fallback transition rejected by HITL");
                    return Ok(false);
                }

                sm.transition_to(&fallback, "fallback after no transitions")?;
                self.hooks
                    .on_state_transition(
                        Some(&from_state),
                        &fallback,
                        "fallback after no transitions",
                    )
                    .await;
                info!(to = %fallback, "Fallback transition");
                return Ok(true);
            }
        }

        Ok(false)
    }

    #[instrument(skip(self, input), fields(agent = %self.info.name))]
    async fn run_loop(&self, input: &str) -> Result<AgentResponse> {
        info!(input_len = input.len(), "Starting chat");

        self.hooks.on_message_received(input).await;

        self.check_turn_timeout().await?;
        self.context_manager.refresh_per_turn().await?;

        let input_data = self.process_input(input).await?;

        if input_data.metadata.rejected {
            let reason = input_data
                .metadata
                .rejection_reason
                .unwrap_or_else(|| "Input rejected".to_string());
            warn!(reason = %reason, "Input rejected");
            return Ok(AgentResponse::new(reason));
        }

        let processed_input = &input_data.content;

        if let Some(skill_response) = self.try_skill_route(processed_input).await? {
            self.memory
                .add_message(ChatMessage::user(processed_input))
                .await?;

            let output_data = self
                .process_output(&skill_response, &input_data.context)
                .await?;
            let final_response = output_data.content;

            self.memory
                .add_message(ChatMessage::assistant(&final_response))
                .await?;

            self.increment_turn();
            self.evaluate_transitions(processed_input, &final_response)
                .await?;

            let response = AgentResponse::new(final_response);
            self.hooks.on_response(&response).await;
            return Ok(response);
        }

        self.memory
            .add_message(ChatMessage::user(processed_input))
            .await?;

        let mut iterations = 0u32;
        let mut all_tool_calls: Vec<ToolCall> = Vec::new();

        let llm = self.get_state_llm()?;

        loop {
            if iterations >= self.max_iterations {
                let err =
                    AgentError::Other(format!("Max iterations ({}) exceeded", self.max_iterations));
                self.hooks.on_error(&err).await;
                error!(iterations = iterations, "Max iterations exceeded");
                return Err(err);
            }
            iterations += 1;
            *self.iteration_count.write() = iterations;

            debug!(
                iteration = iterations,
                max = self.max_iterations,
                "LLM call"
            );

            let messages = self.build_messages().await?;

            self.hooks.on_llm_start(&messages).await;
            let llm_start = Instant::now();

            let response = if self.recovery_manager.config().default.max_retries > 0 {
                self.recovery_manager
                    .with_retry("llm_call", None, || async {
                        llm.complete(&messages, None)
                            .await
                            .map_err(|e| e.classify())
                    })
                    .await
                    .map_err(|e| AgentError::LLM(e.to_string()))?
            } else {
                llm.complete(&messages, None)
                    .await
                    .map_err(|e| AgentError::LLM(e.to_string()))?
            };

            let llm_duration_ms = llm_start.elapsed().as_millis() as u64;
            self.hooks.on_llm_complete(&response, llm_duration_ms).await;

            let content = response.content.trim();

            if let Some(tool_calls) = self.parse_tool_calls(content) {
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
                                return Ok(AgentResponse {
                                    content: format!("Operation cancelled: {}", e),
                                    metadata: None,
                                    tool_calls: Some(all_tool_calls),
                                });
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
                continue;
            }

            let output_data = self.process_output(content, &input_data.context).await?;

            let final_content = if output_data.metadata.rejected {
                output_data
                    .metadata
                    .rejection_reason
                    .unwrap_or_else(|| content.to_string())
            } else {
                output_data.content
            };

            self.memory
                .add_message(ChatMessage::assistant(&final_content))
                .await?;

            self.increment_turn();
            self.evaluate_transitions(processed_input, &final_content)
                .await?;

            let mut response = AgentResponse::new(&final_content);
            if !all_tool_calls.is_empty() {
                response = response.with_tool_calls(all_tool_calls);
            }

            if let Some(state) = self.current_state() {
                response = response.with_metadata("current_state", serde_json::json!(state));
            }

            self.hooks.on_response(&response).await;

            let tool_call_count = response.tool_calls.as_ref().map(|tc| tc.len()).unwrap_or(0);
            info!(tool_calls = tool_call_count, "Chat completed");
            return Ok(response);
        }
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

    async fn request_hitl_approval(&self, check_result: HITLCheckResult) -> Result<bool> {
        let Some(request) = check_result.into_request() else {
            return Ok(true);
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

        match result {
            ApprovalResult::Approved => Ok(true),
            ApprovalResult::Modified { .. } => Ok(true),
            ApprovalResult::Rejected { reason } => {
                if let Some(r) = reason {
                    info!(reason = %r, "HITL rejected");
                }
                Ok(false)
            }
            ApprovalResult::Timeout => {
                if let Some(ref engine) = self.hitl_engine {
                    match engine.config().on_timeout {
                        TimeoutAction::Approve => Ok(true),
                        TimeoutAction::Reject => Ok(false),
                        TimeoutAction::Error => {
                            Err(AgentError::Other("HITL approval timeout".to_string()))
                        }
                    }
                } else {
                    Ok(false)
                }
            }
        }
    }

    pub async fn check_state_hitl(&self, from: Option<&str>, to: &str) -> Result<bool> {
        if let Some(ref hitl_engine) = self.hitl_engine {
            let check_result = hitl_engine.check_state_transition(from, to);
            if check_result.is_required() {
                return self.request_hitl_approval(check_result).await;
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
                let result = self.execute_tool_smart(tc).await;
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
    pub async fn chat_stream(
        &self,
        input: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send + '_>>> {
        info!(input_len = input.len(), "Starting streaming chat");

        self.check_turn_timeout().await?;
        self.context_manager.refresh_per_turn().await?;

        let input_data = self.process_input(input).await?;

        if input_data.metadata.rejected {
            let reason = input_data
                .metadata
                .rejection_reason
                .unwrap_or_else(|| "Input rejected".to_string());
            warn!(reason = %reason, "Input rejected");
            return Ok(Box::pin(stream::once(
                async move { StreamChunk::error(reason) },
            )));
        }

        let processed_input = input_data.content.clone();
        let input_context = input_data.context.clone();

        if let Some(skill_response) = self.try_skill_route(&processed_input).await? {
            self.memory
                .add_message(ChatMessage::user(&processed_input))
                .await?;

            let output_data = self.process_output(&skill_response, &input_context).await?;
            let final_response = output_data.content;

            self.memory
                .add_message(ChatMessage::assistant(&final_response))
                .await?;

            self.increment_turn();
            self.evaluate_transitions(&processed_input, &final_response)
                .await?;

            return Ok(Box::pin(stream::iter(vec![
                StreamChunk::content(final_response),
                StreamChunk::Done {},
            ])));
        }

        self.memory
            .add_message(ChatMessage::user(&processed_input))
            .await?;

        let llm = self.get_state_llm()?;
        let messages = self.build_messages().await?;

        let llm_stream = llm
            .complete_stream(&messages, None)
            .await
            .map_err(|e| AgentError::LLM(e.to_string()))?;

        let streaming_config = self.streaming.clone();
        let include_tool_events = streaming_config.include_tool_events;

        let stream = async_stream::stream! {
            let mut accumulated_content = String::new();
            let mut stream = llm_stream;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        accumulated_content.push_str(&chunk.delta);
                        yield StreamChunk::content(chunk.delta);
                    }
                    Err(e) => {
                        yield StreamChunk::error(e.to_string());
                        return;
                    }
                }
            }

            let content = accumulated_content.trim().to_string();

            if let Some(tool_calls) = self.parse_tool_calls(&content) {
                for tool_call in &tool_calls {
                    if include_tool_events {
                        yield StreamChunk::tool_start(&tool_call.id, &tool_call.name);
                    }

                    match self.execute_tool_smart(tool_call).await {
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
                            if include_tool_events {
                                yield StreamChunk::tool_result(
                                    &tool_call.id,
                                    &tool_call.name,
                                    &e.to_string(),
                                    false,
                                );
                            }
                            let _ = self.memory
                                .add_message(ChatMessage::function(&tool_call.name, &format!("Error: {}", e)))
                                .await;
                        }
                    }

                    if include_tool_events {
                        yield StreamChunk::tool_end(&tool_call.id);
                    }
                }
            } else {
                let _ = self.memory
                    .add_message(ChatMessage::assistant(&content))
                    .await;
            }

            yield StreamChunk::Done {};
        };

        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl Agent for RuntimeAgent {
    async fn chat(&self, input: &str) -> Result<AgentResponse> {
        self.run_loop(input).await
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

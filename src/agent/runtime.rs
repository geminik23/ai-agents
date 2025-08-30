use async_trait::async_trait;
use parking_lot::RwLock;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info, instrument, warn};

use super::{Agent, AgentInfo, AgentResponse, ToolCall};
use crate::context::{ContextManager, ContextProvider, TemplateRenderer};
use crate::error::{AgentError, Result};
use crate::llm::{ChatMessage, LLMRegistry};
use crate::memory::Memory;
use crate::persistence::{AgentSnapshot, AgentStorage};
use crate::process::{ProcessData, ProcessProcessor};
use crate::recovery::{
    ByRoleFilter, ContextOverflowAction, FilterConfig, IntoClassifiedError, KeepRecentFilter,
    MessageFilter, RecoveryManager, SkipPatternFilter,
};
use crate::skill::{SkillDefinition, SkillExecutor, SkillRouter};
use crate::state::{
    PromptMode, StateMachine, StateMachineSnapshot, StateTransitionEvent, TransitionContext,
    TransitionEvaluator,
};
use crate::tool_security::{SecurityCheckResult, ToolSecurityEngine};
use crate::tools::ToolRegistry;

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
            .finish_non_exhaustive()
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
        }
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
        info!(
            removed = to_remove,
            kept = keep_recent,
            "Truncated messages"
        );
    }

    fn get_filter(&self, config: Option<&FilterConfig>) -> Arc<dyn MessageFilter> {
        match config {
            None | Some(FilterConfig::KeepRecent) => Arc::new(KeepRecentFilter),
            Some(FilterConfig::ByRole { keep_roles }) => Arc::new(ByRoleFilter {
                keep_roles: keep_roles.clone(),
            }),
            Some(FilterConfig::SkipPattern { skip_if_contains }) => Arc::new(SkipPatternFilter {
                skip_if_contains: skip_if_contains.clone(),
            }),
            Some(FilterConfig::Custom { name }) => self
                .message_filters
                .read()
                .get(name)
                .cloned()
                .unwrap_or_else(|| Arc::new(KeepRecentFilter)),
        }
    }

    async fn summarize_context(
        &self,
        messages: &mut Vec<ChatMessage>,
        summarizer_llm: Option<&str>,
        _max_summary_tokens: u32,
        custom_prompt: Option<&str>,
        keep_recent: usize,
        filter_config: Option<&FilterConfig>,
    ) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }

        let system_msg = messages.remove(0);
        let filter = self.get_filter(filter_config);
        let (to_summarize, to_keep) = filter.filter(messages, keep_recent);

        if to_summarize.is_empty() {
            messages.insert(0, system_msg);
            return Ok(());
        }

        let llm = if let Some(alias) = summarizer_llm {
            self.llm_registry.get(alias).map_err(|e| {
                AgentError::Config(format!("Summarizer LLM '{}' not found: {}", alias, e))
            })?
        } else {
            self.llm_registry
                .default()
                .map_err(|e| AgentError::Config(format!("Default LLM not available: {}", e)))?
        };

        let default_prompt =
            "Summarize the following conversation concisely, preserving key information.";
        let prompt = custom_prompt.unwrap_or(default_prompt);

        let conversation = to_summarize
            .iter()
            .map(|m| format!("{:?}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        let request = format!("{}\n\nConversation:\n\n{}", prompt, conversation);

        debug!(
            count = to_summarize.len(),
            filter = %filter.name(),
            "Summarizing messages"
        );

        let response = llm
            .complete(&[ChatMessage::user(&request)], None)
            .await
            .map_err(|e| AgentError::LLM(format!("Summarization failed: {}", e)))?;

        let summary = response.content.trim();

        messages.clear();
        messages.push(system_msg);
        messages.push(ChatMessage::system(&format!(
            "[Summary of earlier conversation]: {}",
            summary
        )));
        messages.extend(to_keep);

        info!(
            summarized = to_summarize.len(),
            kept = messages.len() - 2,
            "Context summarized"
        );

        Ok(())
    }

    fn render_system_prompt(&self) -> Result<String> {
        let context = self.context_manager.get_all();
        self.template_renderer
            .render(&self.base_system_prompt, &context)
    }

    fn get_effective_system_prompt(&self) -> Result<String> {
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

                let tools_prompt = if state_def.tools.is_empty() {
                    self.tools.generate_tools_prompt()
                } else {
                    self.tools.generate_filtered_prompt(&state_def.tools)
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

    fn get_state_llm(&self) -> Result<Arc<dyn crate::llm::LLMProvider>> {
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
                if !state_def.skills.is_empty() {
                    return self
                        .skills
                        .iter()
                        .filter(|s| state_def.skills.contains(&s.id))
                        .collect();
                }
            }
        }
        self.skills.iter().collect()
    }

    async fn build_messages(&self) -> Result<Vec<ChatMessage>> {
        let system_prompt = self.get_effective_system_prompt()?;
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

        match &result {
            Ok(output) => info!(output_len = output.len(), "Tool execution successful"),
            Err(e) => error!(error = %e, "Tool execution failed"),
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

    async fn evaluate_transitions(&self, user_message: &str, response: &str) -> Result<()> {
        let (transitions, evaluator, current_state) =
            match (&self.state_machine, &self.transition_evaluator) {
                (Some(sm), Some(eval)) => {
                    let auto_transitions = sm.auto_transitions();
                    if auto_transitions.is_empty() {
                        return Ok(());
                    }
                    (auto_transitions, eval, sm.current())
                }
                _ => return Ok(()),
            };

        let context = TransitionContext {
            user_message: user_message.to_string(),
            assistant_response: response.to_string(),
            current_state,
        };

        if let Some(index) = evaluator.select_transition(&transitions, &context).await? {
            let target = transitions[index].to.clone();
            let reason = transitions[index].when.clone();

            if let Some(ref sm) = self.state_machine {
                sm.transition_to(&target, &reason)?;
                info!(from = %context.current_state, to = %target, "State transition");
            }
        }

        Ok(())
    }

    #[instrument(skip(self, input), fields(agent = %self.info.name))]
    async fn run_loop(&self, input: &str) -> Result<AgentResponse> {
        info!(input_len = input.len(), "Starting chat");

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

            return Ok(AgentResponse::new(final_response));
        }

        self.memory
            .add_message(ChatMessage::user(processed_input))
            .await?;

        let mut iterations = 0u32;
        let mut all_tool_calls: Vec<ToolCall> = Vec::new();

        let llm = self.get_state_llm()?;

        loop {
            if iterations >= self.max_iterations {
                error!(iterations = iterations, "Max iterations exceeded");
                return Err(AgentError::Other(format!(
                    "Max iterations ({}) exceeded",
                    self.max_iterations
                )));
            }
            iterations += 1;
            *self.iteration_count.write() = iterations;

            debug!(
                iteration = iterations,
                max = self.max_iterations,
                "LLM call"
            );

            let messages = self.build_messages().await?;

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

            let content = response.content.trim();

            if let Some(tool_calls) = self.parse_tool_calls(content) {
                for tool_call in &tool_calls {
                    match self.execute_tool_smart(tool_call).await {
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
        self.tool_security.reset_session();
        if let Some(ref sm) = self.state_machine {
            sm.reset();
        }
        info!("Agent session reset");
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
}

#[async_trait]
impl Agent for RuntimeAgent {
    async fn chat(&self, input: &str) -> Result<AgentResponse> {
        if input.trim().is_empty() {
            return Err(AgentError::Other("Input cannot be empty".into()));
        }
        self.run_loop(input).await
    }

    fn info(&self) -> AgentInfo {
        self.info.clone()
    }

    async fn reset(&self) -> Result<()> {
        self.memory.clear().await?;
        *self.iteration_count.write() = 0;
        self.tool_security.reset_session();
        if let Some(ref sm) = self.state_machine {
            sm.reset();
        }
        info!("Agent session reset");
        Ok(())
    }
}

use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info, instrument, warn};

use super::{Agent, AgentInfo, AgentResponse, ToolCall};
use crate::error::{AgentError, Result};
use crate::llm::{ChatMessage, LLMRegistry};
use crate::memory::Memory;
use crate::process::{ProcessData, ProcessProcessor};
use crate::recovery::{
    ByRoleFilter, ContextOverflowAction, FilterConfig, IntoClassifiedError, KeepRecentFilter,
    MessageFilter, RecoveryManager, SkipPatternFilter,
};
use crate::skill::{SkillDefinition, SkillExecutor, SkillRouter};
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
    system_prompt: String,
    max_iterations: u32,
    iteration_count: RwLock<u32>,
    max_context_tokens: u32,
    recovery_manager: RecoveryManager,
    tool_security: ToolSecurityEngine,
    process_processor: Option<ProcessProcessor>,
    message_filters: RwLock<HashMap<String, Arc<dyn MessageFilter>>>,
}

impl std::fmt::Debug for RuntimeAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeAgent")
            .field("info", &self.info)
            .field("system_prompt", &self.system_prompt)
            .field("max_iterations", &self.max_iterations)
            .field("skills_count", &self.skills.len())
            .field("max_context_tokens", &self.max_context_tokens)
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

        Self {
            info,
            llm_registry,
            memory,
            tools,
            skills,
            skill_router,
            skill_executor,
            system_prompt,
            max_iterations,
            iteration_count: RwLock::new(0),
            max_context_tokens: 4096,
            recovery_manager: RecoveryManager::default(),
            tool_security: ToolSecurityEngine::default(),
            process_processor: None,
            message_filters: RwLock::new(HashMap::new()),
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

    pub fn register_message_filter(&self, name: impl Into<String>, filter: Arc<dyn MessageFilter>) {
        self.message_filters.write().insert(name.into(), filter);
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
            "Truncated {} old messages, kept {} recent",
            to_remove, keep_recent
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
            debug!("No messages to summarize");
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
            "Summarizing {} messages using filter '{}'",
            to_summarize.len(),
            filter.name()
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
            "Summarized {} messages, kept {}, summary ~{} tokens",
            to_summarize.len(),
            messages.len() - 2,
            self.estimate_tokens(summary)
        );

        Ok(())
    }

    async fn build_messages(&self) -> Result<Vec<ChatMessage>> {
        let mut messages = vec![ChatMessage::system(&self.system_prompt)];
        let history = self.memory.get_messages(None).await?;
        messages.extend(history);

        let total_tokens = self.estimate_total_tokens(&messages);

        if total_tokens > self.max_context_tokens {
            debug!(
                "Context overflow: {} > {} tokens",
                total_tokens, self.max_context_tokens
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
            debug!("Checking tool security");
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
            debug!(
                max_retries = tool_config.max_retries,
                "Executing with recovery"
            );
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
            if let Some(skill_id) = router.select_skill(input).await? {
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
            debug!("Processing input");
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
            debug!("Processing output");
            processor.process_output(output, input_context).await
        } else {
            Ok(ProcessData::new(output))
        }
    }

    #[instrument(skip(self, input), fields(agent = %self.info.name))]
    async fn run_loop(&self, input: &str) -> Result<AgentResponse> {
        info!(input_len = input.len(), "Starting chat loop");

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
            info!("Skill response completed");
            return Ok(AgentResponse::new(final_response));
        }

        self.memory
            .add_message(ChatMessage::user(processed_input))
            .await?;

        let mut iterations = 0u32;
        let mut all_tool_calls: Vec<ToolCall> = Vec::new();

        let llm = self
            .llm_registry
            .default()
            .map_err(|e| AgentError::LLM(e.to_string()))?;

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
                debug!("Calling LLM with recovery");
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
            debug!(response_len = content.len(), "LLM response received");

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

            let tool_call_count = all_tool_calls.len();
            let mut response = AgentResponse::new(&final_content);
            if !all_tool_calls.is_empty() {
                response = response.with_tool_calls(all_tool_calls);
            }

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
        info!("Agent session reset");
        Ok(())
    }

    pub fn max_context_tokens(&self) -> u32 {
        self.max_context_tokens
    }

    pub fn llm_registry(&self) -> &Arc<LLMRegistry> {
        &self.llm_registry
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
        info!("Agent session reset");
        Ok(())
    }
}

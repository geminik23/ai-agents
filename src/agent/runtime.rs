use async_trait::async_trait;
use parking_lot::RwLock;
use std::sync::Arc;

use super::{Agent, AgentInfo, AgentResponse, ToolCall};
use crate::error::{AgentError, Result};
use crate::llm::{ChatMessage, LLMRegistry};
use crate::memory::Memory;
use crate::skill::{SkillDefinition, SkillExecutor, SkillRouter};
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
}

impl std::fmt::Debug for RuntimeAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeAgent")
            .field("info", &self.info)
            .field("system_prompt", &self.system_prompt)
            .field("max_iterations", &self.max_iterations)
            .field("skills_count", &self.skills.len())
            .finish_non_exhaustive()
    }
}

impl RuntimeAgent {
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
        }
    }

    async fn build_messages(&self) -> Result<Vec<ChatMessage>> {
        let mut messages = vec![ChatMessage::system(&self.system_prompt)];
        let history = self.memory.get_messages(None).await?;
        messages.extend(history);
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

    async fn try_skill_route(&self, input: &str) -> Result<Option<String>> {
        if let Some(ref router) = self.skill_router {
            if let Some(skill_id) = router.select_skill(input).await? {
                eprintln!("[Skill] Selected: {}", skill_id);

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

    async fn run_loop(&self, input: &str) -> Result<AgentResponse> {
        if let Some(skill_response) = self.try_skill_route(input).await? {
            self.memory.add_message(ChatMessage::user(input)).await?;
            self.memory
                .add_message(ChatMessage::assistant(&skill_response))
                .await?;
            return Ok(AgentResponse::new(skill_response));
        }

        self.memory.add_message(ChatMessage::user(input)).await?;

        let mut iterations = 0u32;
        let mut all_tool_calls: Vec<ToolCall> = Vec::new();

        let llm = self
            .llm_registry
            .default()
            .map_err(|e| AgentError::LLM(e.to_string()))?;

        loop {
            if iterations >= self.max_iterations {
                return Err(AgentError::Other(format!(
                    "Max iterations ({}) exceeded",
                    self.max_iterations
                )));
            }
            iterations += 1;
            *self.iteration_count.write() = iterations;

            let messages = self.build_messages().await?;

            let response = llm
                .complete(&messages, None)
                .await
                .map_err(|e| AgentError::Other(e.to_string()))?;
            let content = response.content.trim();

            if let Some(tool_calls) = self.parse_tool_calls(content) {
                for tool_call in &tool_calls {
                    eprintln!(
                        "[Tool] Calling '{}' with args: {}",
                        tool_call.name, tool_call.arguments
                    );

                    match self.execute_tool(tool_call).await {
                        Ok(output) => {
                            eprintln!("[Tool] '{}' returned: {}", tool_call.name, output);
                            self.memory
                                .add_message(ChatMessage::function(&tool_call.name, &output))
                                .await?;
                        }
                        Err(e) => {
                            eprintln!("[Tool] '{}' error: {}", tool_call.name, e);
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

            self.memory
                .add_message(ChatMessage::assistant(content))
                .await?;

            let mut response = AgentResponse::new(content);
            if !all_tool_calls.is_empty() {
                response = response.with_tool_calls(all_tool_calls);
            }
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
        Ok(())
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
        Ok(())
    }
}

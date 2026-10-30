use std::sync::Arc;

use ai_agents_core::{AgentError, Result, ToolCallSource, ToolExecutionRequest, ToolInvoker};
use ai_agents_llm::{ChatMessage, LLMRegistry};
use ai_agents_tools::ToolRegistry;
use minijinja::Environment;

use crate::definition::{SkillContext, SkillDefinition, SkillExecutionCursor, SkillStep};

/// Hosts acknowledge exact skill progress before a new effect and after a completed result.
#[async_trait::async_trait]
pub trait SkillExecutionObserver: Send + Sync {
    async fn checkpoint(&self, cursor: &SkillExecutionCursor) -> Result<()>;
}

struct UnjournaledSkill;
#[async_trait::async_trait]
impl SkillExecutionObserver for UnjournaledSkill {
    async fn checkpoint(&self, _: &SkillExecutionCursor) -> Result<()> {
        Ok(())
    }
}

/// Executes skill steps with either prompt-only mode or a runtime tool invoker.
pub struct SkillExecutor {
    /// LLM registry used by prompt steps.
    llm_registry: Arc<LLMRegistry>,
    /// Tool registry used for direct-mode diagnostics.
    tools: Arc<ToolRegistry>,
}

impl SkillExecutor {
    /// Creates a skill executor from shared LLM and tool registries.
    pub fn new(llm_registry: Arc<LLMRegistry>, tools: Arc<ToolRegistry>) -> Self {
        Self {
            llm_registry,
            tools,
        }
    }

    /// Executes prompt-only skills without a runtime tool invoker.
    pub async fn execute(
        &self,
        skill: &SkillDefinition,
        user_input: &str,
        extra_context: serde_json::Value,
    ) -> Result<String> {
        let mut ctx = SkillContext::new(user_input).with_extra(extra_context);

        for (index, step) in skill.steps.iter().enumerate() {
            match step {
                SkillStep::Tool { tool, .. } => {
                    if self.tools.get(tool).is_none() {
                        return Err(AgentError::Skill(format!("Tool not found: {}", tool)));
                    }
                    return Err(AgentError::Skill(format!(
                        "Skill '{}' contains tool step '{}'; use execute_with_invoker so runtime policy, HITL, observability, and eval evidence are preserved",
                        skill.id, tool
                    )));
                }
                SkillStep::Prompt { prompt, llm } => {
                    let rendered_prompt = self.render_prompt(prompt, &ctx)?;

                    let llm_provider = match llm {
                        Some(alias) => self.llm_registry.get(alias)?,
                        None => self.llm_registry.default()?,
                    };

                    let response = llm_provider
                        .complete(&[ChatMessage::user(&rendered_prompt)], None)
                        .await
                        .map_err(|e| AgentError::LLM(e.to_string()))?;

                    // Store prompt result directly as string for simpler template access
                    let result_value =
                        serde_json::Value::String(response.content.trim().to_string());
                    ctx.add_result(index, None, result_value);

                    // Only return on the last step
                    if index == skill.steps.len() - 1 {
                        return Ok(response.content);
                    }
                }
            }
        }

        Err(AgentError::Skill(
            "Skill has no prompt step to generate response".to_string(),
        ))
    }

    /// Executes skills through the shared tool invoker for every tool step.
    pub async fn execute_with_invoker<I>(
        &self,
        skill: &SkillDefinition,
        user_input: &str,
        extra_context: serde_json::Value,
        invoker: &I,
    ) -> Result<String>
    where
        I: ToolInvoker + ?Sized,
    {
        let mut cursor = SkillExecutionCursor::new(skill, user_input, extra_context);
        self.execute_resumable_with_invoker(skill, &mut cursor, invoker, &UnjournaledSkill)
            .await
    }

    /// Runs only unfinished script steps; pending tool identity and completed prompt/template results survive suspension.
    pub async fn execute_resumable_with_invoker<I, O>(
        &self,
        skill: &SkillDefinition,
        cursor: &mut SkillExecutionCursor,
        invoker: &I,
        observer: &O,
    ) -> Result<String>
    where
        I: ToolInvoker + ?Sized,
        O: SkillExecutionObserver + ?Sized,
    {
        cursor.validate(skill)?;
        while cursor.next_step < skill.steps.len() {
            let index = cursor.next_step;
            match &skill.steps[index] {
                SkillStep::Tool { tool, args, .. } => {
                    if cursor.pending.is_none() {
                        cursor.pending = Some(ToolExecutionRequest::new(
                            uuid::Uuid::new_v4().to_string(),
                            tool,
                            self.render_args(args.clone(), &cursor.context)?,
                            ToolCallSource::Skill {
                                skill_id: skill.id.clone(),
                                step_index: index,
                            },
                        ));
                    }
                    observer.checkpoint(cursor).await?;
                    let record = invoker
                        .invoke_tool(cursor.pending.as_ref().unwrap().clone())
                        .await?;
                    cursor.accept_tool_result(record)?;
                }
                SkillStep::Prompt { prompt, llm } => {
                    observer.checkpoint(cursor).await?;
                    let rendered = self.render_prompt(prompt, &cursor.context)?;
                    let provider = match llm {
                        Some(alias) => self.llm_registry.get(alias)?,
                        None => self.llm_registry.default()?,
                    };
                    let response = provider
                        .complete(&[ChatMessage::user(&rendered)], None)
                        .await
                        .map_err(|error| AgentError::LLM(error.to_string()))?;
                    cursor.context.add_result(
                        index,
                        None,
                        serde_json::Value::String(response.content.trim().to_string()),
                    );
                    cursor.next_step += 1;
                    if cursor.next_step == skill.steps.len() {
                        cursor.response = Some(response.content);
                    }
                }
            }
            observer.checkpoint(cursor).await?;
        }
        cursor.response.clone().ok_or_else(|| {
            AgentError::Skill("Skill has no prompt step to generate response".into())
        })
    }

    fn render_args(
        &self,
        args: Option<serde_json::Value>,
        ctx: &SkillContext,
    ) -> Result<serde_json::Value> {
        match args {
            Some(value) => self.render_value(&value, ctx),
            None => Ok(serde_json::json!({})),
        }
    }

    fn render_value(
        &self,
        value: &serde_json::Value,
        ctx: &SkillContext,
    ) -> Result<serde_json::Value> {
        match value {
            serde_json::Value::String(s) => {
                let rendered = self.render_template_string(s, ctx)?;
                Ok(serde_json::Value::String(rendered))
            }
            serde_json::Value::Object(map) => {
                let mut new_map = serde_json::Map::new();
                for (k, v) in map {
                    new_map.insert(k.clone(), self.render_value(v, ctx)?);
                }
                Ok(serde_json::Value::Object(new_map))
            }
            serde_json::Value::Array(arr) => {
                let new_arr: Result<Vec<_>> =
                    arr.iter().map(|v| self.render_value(v, ctx)).collect();
                Ok(serde_json::Value::Array(new_arr?))
            }
            other => Ok(other.clone()),
        }
    }

    fn render_prompt(&self, template: &str, ctx: &SkillContext) -> Result<String> {
        self.render_template_string(template, ctx)
    }

    fn render_template_string(&self, template: &str, ctx: &SkillContext) -> Result<String> {
        let env = Environment::new();

        let tmpl = env
            .template_from_str(template)
            .map_err(|e| AgentError::Skill(format!("Template parse error: {}", e)))?;

        let steps: Vec<serde_json::Value> = ctx
            .step_results
            .iter()
            .map(|step| {
                serde_json::json!({
                    "result": step.result,
                    "args": step.args.as_ref().unwrap_or(&serde_json::json!({}))
                })
            })
            .collect();

        let jinja_ctx = minijinja::context! {
            user_input => &ctx.user_input,
            steps => steps,
            context => &ctx.extra,
        };

        tmpl.render(jinja_ctx)
            .map_err(|e| AgentError::Skill(format!("Template render error: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_context() -> SkillContext {
        let mut ctx = SkillContext::new("What should I wear?");
        ctx.add_result(
            0,
            Some(serde_json::json!({"location": "Seoul"})),
            serde_json::json!({"temperature": 15, "condition": "sunny"}),
        );
        ctx.extra = serde_json::json!({"user_name": "jay"});
        ctx
    }

    #[test]
    fn test_render_complex_template() {
        let registry = LLMRegistry::new();
        let tools = ToolRegistry::new();
        let executor = SkillExecutor::new(Arc::new(registry), Arc::new(tools));

        let ctx = create_test_context();
        let template = r#"User {{ context.user_name }} asked: {{ user_input }}
Current weather in {{ steps[0].args.location }}: {{ steps[0].result.temperature }}°C, {{ steps[0].result.condition }}"#;

        let result = executor.render_template_string(template, &ctx).unwrap();
        assert!(result.contains("User jay asked: What should I wear?"));
        assert!(result.contains("Current weather in Seoul: 15°C, sunny"));
    }

    #[test]
    fn test_render_with_whitespace_variations() {
        let registry = LLMRegistry::new();
        let tools = ToolRegistry::new();
        let executor = SkillExecutor::new(Arc::new(registry), Arc::new(tools));

        let ctx = create_test_context();
        let template1 = "{{user_input}}";
        let template2 = "{{ user_input }}";
        let template3 = "{{  user_input  }}";

        let result1 = executor.render_template_string(template1, &ctx).unwrap();
        let result2 = executor.render_template_string(template2, &ctx).unwrap();
        let result3 = executor.render_template_string(template3, &ctx).unwrap();

        assert_eq!(result1, "What should I wear?");
        assert_eq!(result2, "What should I wear?");
        assert_eq!(result3, "What should I wear?");
    }

    #[test]
    fn test_render_with_filters() {
        let registry = LLMRegistry::new();
        let tools = ToolRegistry::new();
        let executor = SkillExecutor::new(Arc::new(registry), Arc::new(tools));

        let ctx = create_test_context();
        let template = "{{ context.user_name | upper }}";

        let result = executor.render_template_string(template, &ctx).unwrap();
        assert_eq!(result, "JAY");
    }

    #[tokio::test]
    async fn direct_execute_rejects_tool_steps() {
        let registry = LLMRegistry::new();
        let mut tools = ToolRegistry::new();
        tools
            .register(Arc::new(ai_agents_tools::EchoTool::new()))
            .unwrap();
        let executor = SkillExecutor::new(Arc::new(registry), Arc::new(tools));
        let skill = SkillDefinition {
            autonomy: None,
            id: "tool_skill".to_string(),
            description: "Uses a tool".to_string(),
            trigger: "test".to_string(),
            steps: vec![SkillStep::Tool {
                tool: "echo".to_string(),
                args: Some(serde_json::json!({"message": "hello"})),
                output_as: None,
            }],
            reasoning: None,
            reflection: None,
            disambiguation: None,
        };

        let error = executor
            .execute(&skill, "hello", serde_json::json!({}))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("execute_with_invoker"));
    }
}

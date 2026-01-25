use std::sync::Arc;

use ai_agents_core::{AgentError, Result};
use ai_agents_llm::{ChatMessage, LLMRegistry};
use ai_agents_tools::ToolRegistry;
use minijinja::Environment;

use crate::definition::{SkillContext, SkillDefinition, SkillStep};

pub struct SkillExecutor {
    llm_registry: Arc<LLMRegistry>,
    tools: Arc<ToolRegistry>,
}

impl SkillExecutor {
    pub fn new(llm_registry: Arc<LLMRegistry>, tools: Arc<ToolRegistry>) -> Self {
        Self {
            llm_registry,
            tools,
        }
    }

    pub async fn execute(
        &self,
        skill: &SkillDefinition,
        user_input: &str,
        extra_context: serde_json::Value,
    ) -> Result<String> {
        let mut ctx = SkillContext::new(user_input).with_extra(extra_context);

        for (index, step) in skill.steps.iter().enumerate() {
            match step {
                SkillStep::Tool {
                    tool,
                    args,
                    output_as: _,
                } => {
                    let rendered_args = self.render_args(args.clone(), &ctx)?;
                    let tool_impl = self
                        .tools
                        .get(tool)
                        .ok_or_else(|| AgentError::Skill(format!("Tool not found: {}", tool)))?;

                    let result = tool_impl.execute(rendered_args.clone()).await;
                    eprintln!("[Skill] Tool '{}' returned: {}", tool, result.output);

                    let result_value: serde_json::Value = serde_json::from_str(&result.output)
                        .unwrap_or_else(|_| {
                            serde_json::json!({
                                "output": result.output,
                                "success": result.success
                            })
                        });

                    ctx.add_result(index, Some(rendered_args), result_value);

                    if !result.success {
                        return Err(AgentError::Skill(format!(
                            "Tool '{}' failed: {}",
                            tool, result.output
                        )));
                    }
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
}

use crate::llm::TaskContext;
use minijinja::{Environment, context};
use serde::Serialize;

/// Template for tool selection with Jinja2 syntax
pub const TOOL_SELECTION_TEMPLATE: &str = r#"You are an AI assistant that selects the most appropriate tool to handle a user's request.

Available tools:
{% if tools | length > 0 -%}
{% for tool in tools -%}
{{ loop.index }}. {{ tool }}
{% endfor -%}
{%- else -%}
No tools available
{%- endif %}

Current context:
{%- if context %}
  State: {{ context.current_state | default("none") }}
  {%- if context.memory_slots_count > 0 %}
  Memory: {{ context.memory_slots_count }} slot(s)
  {%- endif %}
{%- else %}
  No context available
{%- endif %}

User request: {{ user_input }}

Select the most appropriate tool for this request. Respond in the following JSON format:
{
  "tool_id": "tool_name",
  "confidence": 0.95,
  "reasoning": "explanation of why this tool was selected"
}

If no tool is appropriate, select "none" as the tool_id with a confidence below 0.5."#;

/// Template for tool argument generation
pub const TOOL_ARGS_TEMPLATE: &str = r#"You are an AI assistant that generates arguments for tool invocations.

Tool: {{ tool_id }}
Tool schema: {{ schema }}

User request: {{ user_input }}

Generate the arguments for this tool call based on the user's request.
Respond with ONLY valid JSON matching the tool's schema. Do not include any explanation or additional text."#;

/// Template for yes/no evaluation (guards)
pub const YESNO_EVALUATION_TEMPLATE: &str = r#"You are an AI assistant that evaluates yes/no questions based on context.

Question: {{ question }}

Context:
{%- if context %}
{%- if context.current_state %}
  Current State: {{ context.current_state }}
{%- endif %}
{%- if context.available_tools | length > 0 %}
  Available Tools: {{ context.available_tools | join(", ") }}
{%- endif %}
{%- if context.memory_slots_count > 0 %}
  Memory Slots: {{ context.memory_slots_count }}
  {%- for item in context.memory_slots_list %}
    - {{ item.key }}: {{ item.value }}
  {%- endfor %}
{%- endif %}
{%- if context.recent_messages | length > 0 %}
  Recent Messages: {{ context.recent_messages | length }}
  {%- for msg in context.recent_messages | slice(end=3) %}
    - {{ msg.role }}: {{ msg.content }}
  {%- endfor %}
{%- endif %}
{%- else %}
  No context available
{%- endif %}

Evaluate this question and respond in the following JSON format:
{
  "answer": true,
  "reasoning": "explanation of your decision"
}

Be precise and base your answer strictly on the provided context."#;

/// Template for classification
pub const CLASSIFICATION_TEMPLATE: &str = r#"You are an AI assistant that classifies text into predefined categories.

Text to classify: {{ input }}

Available categories:
{%- for category in categories %}
{{ loop.index }}. {{ category }}
{%- endfor %}

Classify the text into one of the available categories. Respond in the following JSON format:
{
  "category": "category_name",
  "confidence": 0.95
}

Choose the most appropriate category based on the text's content and intent."#;

/// Template for task processing (main agent reasoning)
pub const TASK_PROCESSING_TEMPLATE: &str = r#"{{ system_prompt }}

Current State: {{ context.current_state | default("None") }}

Available Tools:
{%- if context.available_tools | length > 0 %}
{%- for tool in context.available_tools %}
{{ loop.index }}. {{ tool }}
{%- endfor %}
{%- else %}
No tools available
{%- endif %}

Memory Context:
{%- if context.memory_slots_count > 0 %}
{%- for item in context.memory_slots_list %}
{{ item.key }}: {{ item.value }}
{%- endfor %}
{%- else %}
No memory slots
{%- endif %}

Conversation History:
{%- if context.recent_messages | length > 0 %}
{%- for msg in context.recent_messages %}
{{ msg.role }}: {{ msg.content }}
{%- endfor %}
{%- else %}
No conversation history
{%- endif %}

Based on the above context, process the user's latest message and decide on the next action.
You can either:
1. Respond directly to the user
2. Use a tool to accomplish a task
3. Ask for more information

Think step by step and provide a clear, helpful response."#;

/// Helper struct to represent a key-value pair for template rendering
#[derive(Debug, Serialize)]
struct MemorySlot {
    key: String,
    value: serde_json::Value,
}

/// Extended task context for template rendering
#[derive(Debug, Serialize)]
struct TemplateTaskContext {
    current_state: Option<String>,
    available_tools: Vec<String>,
    memory_slots_count: usize,
    memory_slots_list: Vec<MemorySlot>,
    recent_messages: Vec<super::ChatMessage>,
}

impl From<&TaskContext> for TemplateTaskContext {
    fn from(ctx: &TaskContext) -> Self {
        let memory_slots_list: Vec<MemorySlot> = ctx
            .memory_slots
            .iter()
            .map(|(k, v)| MemorySlot {
                key: k.clone(),
                value: v.clone(),
            })
            .collect();

        Self {
            current_state: ctx.current_state.clone(),
            available_tools: ctx.available_tools.clone(),
            memory_slots_count: ctx.memory_slots.len(),
            memory_slots_list,
            recent_messages: ctx.recent_messages.clone(),
        }
    }
}

/// Create a configured MiniJinja environment for prompt rendering
fn create_environment() -> Environment<'static> {
    let env = Environment::new();

    // Add custom filters if needed
    // env.add_filter("custom_filter", custom_filter_fn);

    env
}

/// Render a template with the given context
fn render_template<S: Serialize>(template: &str, ctx: S) -> Result<String, String> {
    let env = create_environment();
    env.render_str(template, ctx)
        .map_err(|e| format!("Template rendering error: {}", e))
}

/// Builder for tool selection prompts
pub struct ToolSelectionPromptBuilder {
    tools: Vec<String>,
    context: Option<TaskContext>,
    user_input: String,
}

impl ToolSelectionPromptBuilder {
    /// Create a new builder
    pub fn new(user_input: impl Into<String>) -> Self {
        Self {
            tools: Vec::new(),
            context: None,
            user_input: user_input.into(),
        }
    }

    /// Add available tools
    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tools = tools;
        self
    }

    /// Add context
    pub fn with_context(mut self, context: TaskContext) -> Self {
        self.context = Some(context);
        self
    }

    /// Build the prompt using MiniJinja
    pub fn build(self) -> String {
        let template_context = self.context.as_ref().map(TemplateTaskContext::from);

        render_template(
            TOOL_SELECTION_TEMPLATE,
            context! {
                tools => self.tools,
                context => template_context,
                user_input => self.user_input,
            },
        )
        .expect("Failed to render tool selection template")
    }
}

/// Builder for tool argument generation prompts
pub struct ToolArgsPromptBuilder {
    tool_id: String,
    schema: String,
    user_input: String,
}

impl ToolArgsPromptBuilder {
    /// Create a new builder
    pub fn new(
        tool_id: impl Into<String>,
        schema: serde_json::Value,
        user_input: impl Into<String>,
    ) -> Self {
        let schema_str = serde_json::to_string_pretty(&schema).unwrap_or_else(|_| "{}".to_string());

        Self {
            tool_id: tool_id.into(),
            schema: schema_str,
            user_input: user_input.into(),
        }
    }

    /// Build the prompt using MiniJinja
    pub fn build(self) -> String {
        render_template(
            TOOL_ARGS_TEMPLATE,
            context! {
                tool_id => self.tool_id,
                schema => self.schema,
                user_input => self.user_input,
            },
        )
        .expect("Failed to render tool args template")
    }
}

/// Builder for yes/no evaluation prompts
pub struct YesNoPromptBuilder {
    question: String,
    context: Option<TaskContext>,
}

impl YesNoPromptBuilder {
    /// Create a new builder
    pub fn new(question: impl Into<String>) -> Self {
        Self {
            question: question.into(),
            context: None,
        }
    }

    /// Add context
    pub fn with_context(mut self, context: TaskContext) -> Self {
        self.context = Some(context);
        self
    }

    /// Build the prompt using MiniJinja
    pub fn build(self) -> String {
        let template_context = self.context.as_ref().map(TemplateTaskContext::from);

        render_template(
            YESNO_EVALUATION_TEMPLATE,
            context! {
                question => self.question,
                context => template_context,
            },
        )
        .expect("Failed to render yes/no template")
    }
}

/// Builder for classification prompts
pub struct ClassificationPromptBuilder {
    input: String,
    categories: Vec<String>,
}

impl ClassificationPromptBuilder {
    /// Create a new builder
    pub fn new(input: impl Into<String>, categories: Vec<String>) -> Self {
        Self {
            input: input.into(),
            categories,
        }
    }

    /// Build the prompt using MiniJinja
    pub fn build(self) -> String {
        render_template(
            CLASSIFICATION_TEMPLATE,
            context! {
                input => self.input,
                categories => self.categories,
            },
        )
        .expect("Failed to render classification template")
    }
}

/// Builder for task processing prompts
pub struct TaskProcessingPromptBuilder {
    system_prompt: String,
    context: TaskContext,
}

impl TaskProcessingPromptBuilder {
    /// Create a new builder
    pub fn new(system_prompt: impl Into<String>, context: TaskContext) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            context,
        }
    }

    /// Build the prompt using MiniJinja
    pub fn build(self) -> String {
        let template_context = TemplateTaskContext::from(&self.context);

        render_template(
            TASK_PROCESSING_TEMPLATE,
            context! {
                system_prompt => self.system_prompt,
                context => template_context,
            },
        )
        .expect("Failed to render task processing template")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ChatMessage, Role};
    #[test]
    fn test_task_processing_prompt_builder() {
        let context = TaskContext {
            current_state: Some("ready".to_string()),
            available_tools: vec!["calculator".to_string()],
            memory_slots: std::collections::HashMap::new(),
            recent_messages: vec![ChatMessage {
                role: Role::User,
                content: "What's 2 + 2?".to_string(),
                name: None,
                timestamp: Some(chrono::Utc::now()),
            }],
        };

        let prompt =
            TaskProcessingPromptBuilder::new("You are a helpful assistant", context).build();

        assert!(prompt.contains("You are a helpful assistant"));
        assert!(prompt.contains("ready"));
    }
}

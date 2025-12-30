use async_trait::async_trait;
use minijinja::Environment;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tools::{Tool, ToolResult, generate_schema};

pub struct TemplateTool {
    env: Environment<'static>,
}

impl TemplateTool {
    pub fn new() -> Self {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        Self { env }
    }

    fn render_template(&self, template: &str, data: &Value) -> Result<String, String> {
        let tmpl = self
            .env
            .template_from_str(template)
            .map_err(|e| format!("Template parse error: {}", e))?;

        tmpl.render(data)
            .map_err(|e| format!("Render error: {}", e))
    }
}

impl Default for TemplateTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct TemplateInput {
    /// Operation: render (inline template), render_file (from file)
    operation: String,
    /// Template string (for render operation)
    #[serde(default)]
    template: Option<String>,
    /// Template file path (for render_file operation)
    #[serde(default)]
    path: Option<String>,
    /// Data to render into template
    data: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct RenderOutput {
    rendered: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct RenderFileOutput {
    rendered: String,
    template_path: String,
}

#[async_trait]
impl Tool for TemplateTool {
    fn id(&self) -> &str {
        "template"
    }

    fn name(&self) -> &str {
        "Template Renderer"
    }

    fn description(&self) -> &str {
        "Render Jinja2-style templates with data. Operations: render (inline template), render_file (from file). Supports variables ({{ var }}), filters ({{ name|upper }}), conditionals ({% if %}), and loops ({% for %})."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<TemplateInput>()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: TemplateInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        match input.operation.to_lowercase().as_str() {
            "render" => self.handle_render(&input),
            "render_file" => self.handle_render_file(&input),
            _ => ToolResult::error(format!(
                "Unknown operation: {}. Valid: render, render_file",
                input.operation
            )),
        }
    }
}

impl TemplateTool {
    fn handle_render(&self, input: &TemplateInput) -> ToolResult {
        let template = match &input.template {
            Some(t) => t,
            None => return ToolResult::error("'template' is required for render operation"),
        };

        match self.render_template(template, &input.data) {
            Ok(rendered) => {
                let output = RenderOutput { rendered };
                self.to_result(&output)
            }
            Err(e) => ToolResult::error(e),
        }
    }

    fn handle_render_file(&self, input: &TemplateInput) -> ToolResult {
        let path = match &input.path {
            Some(p) => p,
            None => return ToolResult::error("'path' is required for render_file operation"),
        };

        let template = match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(e) => return ToolResult::error(format!("File read error: {}", e)),
        };

        match self.render_template(&template, &input.data) {
            Ok(rendered) => {
                let output = RenderFileOutput {
                    rendered,
                    template_path: path.clone(),
                };
                self.to_result(&output)
            }
            Err(e) => ToolResult::error(e),
        }
    }

    fn to_result<T: Serialize>(&self, output: &T) -> ToolResult {
        match serde_json::to_string(output) {
            Ok(json) => ToolResult::ok(json),
            Err(e) => ToolResult::error(format!("Serialization error: {}", e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_render_simple() {
        let tool = TemplateTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "render",
                "template": "Hello {{ name }}!",
                "data": {"name": "World"}
            }))
            .await;
        assert!(result.success);
        let output: RenderOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.rendered, "Hello World!");
    }

    #[tokio::test]
    async fn test_render_with_filter() {
        let tool = TemplateTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "render",
                "template": "Hello {{ name|upper }}!",
                "data": {"name": "world"}
            }))
            .await;
        assert!(result.success);
        let output: RenderOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.rendered, "Hello WORLD!");
    }

    #[tokio::test]
    async fn test_render_with_loop() {
        let tool = TemplateTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "render",
                "template": "{% for item in items %}{{ item }}{% if not loop.last %}, {% endif %}{% endfor %}",
                "data": {"items": ["a", "b", "c"]}
            }))
            .await;
        assert!(result.success);
        let output: RenderOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.rendered, "a, b, c");
    }

    #[tokio::test]
    async fn test_render_with_conditional() {
        let tool = TemplateTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "render",
                "template": "{% if show %}visible{% else %}hidden{% endif %}",
                "data": {"show": true}
            }))
            .await;
        assert!(result.success);
        let output: RenderOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.rendered, "visible");
    }

    #[tokio::test]
    async fn test_render_nested_data() {
        let tool = TemplateTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "render",
                "template": "Order #{{ order.id }}: {{ order.items|length }} items",
                "data": {
                    "order": {
                        "id": "12345",
                        "items": ["item1", "item2", "item3"]
                    }
                }
            }))
            .await;
        assert!(result.success);
        let output: RenderOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.rendered, "Order #12345: 3 items");
    }

    #[tokio::test]
    async fn test_render_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("template.txt");
        std::fs::write(&file_path, "Hello {{ name }}!").unwrap();

        let tool = TemplateTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "render_file",
                "path": file_path.to_str().unwrap(),
                "data": {"name": "File"}
            }))
            .await;
        assert!(result.success);
        let output: RenderFileOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.rendered, "Hello File!");
    }

    #[tokio::test]
    async fn test_render_missing_template() {
        let tool = TemplateTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "render",
                "data": {}
            }))
            .await;
        assert!(!result.success);
    }

    #[tokio::test]
    async fn test_render_invalid_template() {
        let tool = TemplateTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "render",
                "template": "{{ unclosed",
                "data": {}
            }))
            .await;
        assert!(!result.success);
    }

    #[tokio::test]
    async fn test_invalid_operation() {
        let tool = TemplateTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "invalid",
                "data": {}
            }))
            .await;
        assert!(!result.success);
    }
}

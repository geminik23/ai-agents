use std::collections::HashMap;

use minijinja::{Environment, Value as MJValue};
use serde_json::Value;

use ai_agents_core::{AgentError, Result};

pub struct TemplateRenderer {
    env: Environment<'static>,
}

impl Default for TemplateRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl TemplateRenderer {
    pub fn new() -> Self {
        let mut env = Environment::new();
        env.set_auto_escape_callback(|_| minijinja::AutoEscape::None);
        Self { env }
    }

    pub fn render(&self, template: &str, context: &HashMap<String, Value>) -> Result<String> {
        let mut ctx = HashMap::new();

        // Build context map - all values available as {{ context.<key> }}
        let mut context_map = serde_json::Map::new();
        for (key, value) in context {
            context_map.insert(key.clone(), value.clone());
        }
        ctx.insert("context", json_to_minijinja(&Value::Object(context_map)));

        if let Some(env_vars) = context.get("env") {
            ctx.insert("env", json_to_minijinja(env_vars));
        }

        if let Some(state) = context.get("state") {
            ctx.insert("state", json_to_minijinja(state));
        }

        // Hoist a fixed set of well-known top-level variables so system prompt
        // templates can use {{ actor_facts }} directly instead of
        // {{ context.actor_facts }}. Any key listed here is also accessible via
        // the context. prefix, so both forms work.
        for key in &["actor_facts"] {
            if let Some(value) = context.get(*key) {
                ctx.insert(key, json_to_minijinja(value));
            }
        }

        let tmpl = self
            .env
            .template_from_str(template)
            .map_err(|e| AgentError::TemplateError(e.to_string()))?;

        tmpl.render(&ctx)
            .map_err(|e| AgentError::TemplateError(e.to_string()))
    }

    pub fn render_path(
        &self,
        path_template: &str,
        context: &HashMap<String, Value>,
    ) -> Result<String> {
        self.render(path_template, context)
    }

    pub fn render_with_state(
        &self,
        template: &str,
        context: &HashMap<String, Value>,
        state_name: &str,
        previous_state: Option<&str>,
        turn_count: u32,
        max_turns: Option<u32>,
    ) -> Result<String> {
        let mut full_context = context.clone();

        let mut state_ctx = serde_json::Map::new();
        state_ctx.insert("name".into(), Value::String(state_name.to_string()));
        state_ctx.insert(
            "previous".into(),
            Value::String(previous_state.unwrap_or("none").to_string()),
        );
        state_ctx.insert("turn_count".into(), Value::Number(turn_count.into()));
        if let Some(max) = max_turns {
            state_ctx.insert("max_turns".into(), Value::Number(max.into()));
        }
        full_context.insert("state".into(), Value::Object(state_ctx));

        self.render(template, &full_context)
    }
}

fn json_to_minijinja(value: &Value) -> MJValue {
    match value {
        Value::Null => MJValue::from(()),
        Value::Bool(b) => MJValue::from(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                MJValue::from(i)
            } else if let Some(u) = n.as_u64() {
                MJValue::from(u)
            } else if let Some(f) = n.as_f64() {
                MJValue::from(f)
            } else {
                MJValue::from(())
            }
        }
        Value::String(s) => MJValue::from(s.as_str()),
        Value::Array(arr) => {
            let items: Vec<MJValue> = arr.iter().map(json_to_minijinja).collect();
            MJValue::from(items)
        }
        Value::Object(obj) => {
            let map: std::collections::BTreeMap<String, MJValue> = obj
                .iter()
                .map(|(k, v)| (k.clone(), json_to_minijinja(v)))
                .collect();
            MJValue::from_iter(map)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_simple_variable() {
        let renderer = TemplateRenderer::new();
        let mut context = HashMap::new();
        context.insert("user".into(), json!({"name": "Alice", "tier": "premium"}));

        let template = "Hello, {{ context.user.name }}!";
        let result = renderer.render(template, &context).unwrap();
        assert_eq!(result, "Hello, Alice!");
    }

    #[test]
    fn test_nested_variable() {
        let renderer = TemplateRenderer::new();
        let mut context = HashMap::new();
        context.insert(
            "user".into(),
            json!({"preferences": {"theme": "dark", "language": "ko"}}),
        );

        let template = "Theme: {{ context.user.preferences.theme }}";
        let result = renderer.render(template, &context).unwrap();
        assert_eq!(result, "Theme: dark");
    }

    #[test]
    fn test_conditional() {
        let renderer = TemplateRenderer::new();
        let mut context = HashMap::new();
        context.insert("user".into(), json!({"tier": "premium"}));

        let template = r#"{% if context.user.tier == "premium" %}Premium user{% else %}Regular user{% endif %}"#;
        let result = renderer.render(template, &context).unwrap();
        assert_eq!(result, "Premium user");
    }

    #[test]
    fn test_loop() {
        let renderer = TemplateRenderer::new();
        let mut context = HashMap::new();
        context.insert("items".into(), json!([{"name": "A"}, {"name": "B"}]));

        let template = "{% for item in context.items %}{{ item.name }}{% endfor %}";
        let result = renderer.render(template, &context).unwrap();
        assert_eq!(result, "AB");
    }

    #[test]
    fn test_state_variables() {
        let renderer = TemplateRenderer::new();
        let context = HashMap::new();

        let template = "State: {{ state.name }}, Turn: {{ state.turn_count }}";
        let result = renderer
            .render_with_state(template, &context, "support", Some("greeting"), 2, Some(5))
            .unwrap();
        assert_eq!(result, "State: support, Turn: 2");
    }

    #[test]
    fn test_korean_content() {
        let renderer = TemplateRenderer::new();
        let mut context = HashMap::new();
        context.insert("user".into(), json!({"name": "김철수", "language": "ko"}));

        let template = "안녕하세요, {{ context.user.name }}님! 언어: {{ context.user.language }}";
        let result = renderer.render(template, &context).unwrap();
        assert_eq!(result, "안녕하세요, 김철수님! 언어: ko");
    }

    #[test]
    fn test_path_rendering() {
        let renderer = TemplateRenderer::new();
        let mut context = HashMap::new();
        context.insert("user".into(), json!({"language": "ja"}));

        let path = "./rules/{{ context.user.language }}/support.txt";
        let result = renderer.render_path(path, &context).unwrap();
        assert_eq!(result, "./rules/ja/support.txt");
    }

    #[test]
    fn test_default_filter() {
        let renderer = TemplateRenderer::new();
        let context = HashMap::new();

        let template = "{{ context.missing | default('N/A') }}";
        let result = renderer.render(template, &context).unwrap();
        assert_eq!(result, "N/A");
    }

    // actor_facts must be accessible as a top-level variable {{ actor_facts }}
    // because that is the form used in system prompt templates.
    // It is also accessible as {{ context.actor_facts }}.
    #[test]
    fn test_actor_facts_top_level_variable() {
        let renderer = TemplateRenderer::new();
        let mut context = HashMap::new();
        context.insert(
            "actor_facts".into(),
            json!("- User name is Jay.\n- User works as an AI engineer.\n"),
        );

        // Top-level form used in system prompts.
        let template = "{% if actor_facts %}Known facts:\n{{ actor_facts }}{% endif %}";
        let result = renderer.render(template, &context).unwrap();
        assert!(
            result.contains("User name is Jay"),
            "actor_facts must render at top level without context. prefix"
        );
        assert!(
            result.contains("Known facts"),
            "{{% if actor_facts %}} must evaluate to true for non-empty string"
        );
    }

    #[test]
    fn test_actor_facts_also_accessible_via_context_prefix() {
        let renderer = TemplateRenderer::new();
        let mut context = HashMap::new();
        context.insert("actor_facts".into(), json!("- User name is Jay.\n"));

        // Both forms must work.
        let template_top = "{{ actor_facts }}";
        let template_ctx = "{{ context.actor_facts }}";
        let result_top = renderer.render(template_top, &context).unwrap();
        let result_ctx = renderer.render(template_ctx, &context).unwrap();
        assert_eq!(result_top, result_ctx);
    }

    #[test]
    fn test_actor_facts_if_block_false_when_absent() {
        let renderer = TemplateRenderer::new();
        let context = HashMap::new(); // no actor_facts key

        let template = "base{% if actor_facts %} facts: {{ actor_facts }}{% endif %} end";
        let result = renderer.render(template, &context).unwrap();
        assert_eq!(
            result, "base end",
            "{{% if actor_facts %}} must be false when actor_facts is not in context"
        );
    }
}

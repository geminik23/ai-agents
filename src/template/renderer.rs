//! Template rendering with variable substitution

use crate::error::{AgentError, Result};
use minijinja::Environment;
use std::collections::HashMap;

/// Template renderer using minijinja
pub struct TemplateRenderer {
    env: Environment<'static>,
}

impl TemplateRenderer {
    pub fn new() -> Self {
        let env = Environment::new();

        // TODO: custom filters if needed in the future; But not now.
        // env.add_filter("custom", custom_filter);

        Self { env }
    }

    /// Render a template string with variables
    pub fn render(&self, template: &str, variables: &HashMap<String, String>) -> Result<String> {
        let template = self.replace_env_vars(template)?;

        let tmpl = self
            .env
            .template_from_str(&template)
            .map_err(|e| AgentError::TemplateError(format!("Failed to parse template: {}", e)))?;

        let context = variables
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect::<std::collections::HashMap<_, _>>();

        tmpl.render(context)
            .map_err(|e| AgentError::TemplateError(format!("Failed to render template: {}", e)))
    }

    /// Replace environment variables in template
    fn replace_env_vars(&self, template: &str) -> Result<String> {
        let mut result = template.to_string();

        let re = regex::Regex::new(r"\$ENV\{([^}]+)\}").unwrap();

        for cap in re.captures_iter(template) {
            let full_match = &cap[0];
            let var_name = &cap[1];

            match std::env::var(var_name) {
                Ok(value) => {
                    result = result.replace(full_match, &value);
                }
                Err(_) => {
                    return Err(AgentError::TemplateError(format!(
                        "Environment variable '{}' not found",
                        var_name
                    )));
                }
            }
        }

        Ok(result)
    }
}

impl Default for TemplateRenderer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_simple() {
        let renderer = TemplateRenderer::new();
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "TestAgent".to_string());

        let result = renderer.render("Agent: {{ name }}", &vars).unwrap();
        assert_eq!(result, "Agent: TestAgent");
    }

    #[test]
    fn test_render_multiple_vars() {
        let renderer = TemplateRenderer::new();
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "Bot".to_string());
        vars.insert("version".to_string(), "1.0".to_string());

        let result = renderer.render("{{ name }} v{{ version }}", &vars).unwrap();
        assert_eq!(result, "Bot v1.0");
    }

    #[test]
    fn test_render_with_default_filter() {
        let renderer = TemplateRenderer::new();

        // Test without variable set (uses default)
        let empty_vars = HashMap::new();
        let result = renderer
            .render("{{ name | default('DefaultAgent') }}", &empty_vars)
            .unwrap();
        assert_eq!(result, "DefaultAgent");

        // Test with variable set (overrides default)
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "CustomAgent".to_string());
        let result2 = renderer
            .render("{{ name | default('DefaultAgent') }}", &vars)
            .unwrap();
        assert_eq!(result2, "CustomAgent");
    }

    #[test]
    fn test_render_env_var() {
        unsafe { std::env::set_var("TEST_VAR", "test_value") };

        let renderer = TemplateRenderer::new();
        let vars = HashMap::new();

        let result = renderer.render("Value: $ENV{TEST_VAR}", &vars).unwrap();
        assert_eq!(result, "Value: test_value");

        unsafe { std::env::remove_var("TEST_VAR") };
    }

    #[test]
    fn test_render_env_var_not_found() {
        let renderer = TemplateRenderer::new();
        let vars = HashMap::new();

        let result = renderer.render("Value: $ENV{NONEXISTENT_VAR_XYZ}", &vars);
        assert!(result.is_err());
    }

    #[test]
    fn test_render_env_var_with_template_vars() {
        unsafe { std::env::set_var("TEST_ENV", "from_env") };

        let renderer = TemplateRenderer::new();
        let mut vars = HashMap::new();
        vars.insert("template_var".to_string(), "from_template".to_string());

        let result = renderer
            .render("Env: $ENV{TEST_ENV}, Template: {{ template_var }}", &vars)
            .unwrap();
        assert_eq!(result, "Env: from_env, Template: from_template");

        unsafe { std::env::remove_var("TEST_ENV") };
    }
}

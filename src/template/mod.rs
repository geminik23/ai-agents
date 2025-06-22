//! Template system for loading and processing agent specifications

mod inheritance;
mod loader;
mod renderer;

pub use loader::TemplateLoader;
pub use renderer::TemplateRenderer;

use crate::error::Result;
use crate::spec::AgentSpec;
use inheritance::TemplateInheritance;

impl TemplateLoader {
    /// Load and parse a template into an AgentSpec
    /// This handles inheritance, variable substitution, and parsing
    pub fn load_and_parse(&self, template_name: &str) -> Result<AgentSpec> {
        let renderer = TemplateRenderer::new();
        let variables = self.variables();

        // Define a loader that loads AND renders
        let load_and_render = |name: &str| -> Result<String> {
            let content = self.load_template(name)?;
            renderer.render(&content, variables)
        };

        // Step 1: Load and render the root template
        let rendered_root = load_and_render(template_name)?;

        // Step 2: Process inheritance (extends)
        // Pass the rendering loader to process() so parents are also rendered
        let processed = TemplateInheritance::process(&rendered_root, load_and_render)?;

        // Step 3: Parse as AgentSpec
        let spec: AgentSpec = serde_yaml::from_str(&processed)?;
        spec.validate()?;

        Ok(spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_template(dir: &std::path::Path, name: &str, content: &str) {
        fs::write(dir.join(name), content).unwrap();
    }

    #[test]
    fn test_load_and_parse_simple() {
        let temp_dir = TempDir::new().unwrap();
        let template = r#"
name: {{ agent_name }}
version: 1.0.0
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4
"#;
        create_template(temp_dir.path(), "simple.yaml", template);

        let mut loader = TemplateLoader::new();
        loader.add_search_path(temp_dir.path());
        loader.set_variable("agent_name", "TestAgent");

        let spec = loader.load_and_parse("simple").unwrap();
        assert_eq!(spec.name, "TestAgent");
        assert_eq!(spec.version, "1.0.0");
    }

    #[test]
    fn test_load_and_parse_with_inheritance() {
        let temp_dir = TempDir::new().unwrap();

        let base = r#"name: BaseAgent
version: 1.0.0
system_prompt: Base prompt
llm:
  provider: openai
  model: gpt-4
max_iterations: 10"#;

        let child = r#"extends: base.yaml
name: {{ agent_name }}
description: {{ agent_description | default("Test description") }}"#;

        create_template(temp_dir.path(), "test_base.yaml", base);
        create_template(temp_dir.path(), "test_child.yaml", child);

        let mut loader = TemplateLoader::new();
        loader.add_search_path(temp_dir.path());
        loader.set_variable("agent_name", "ChildAgent");

        let spec = loader.load_and_parse("test_child").unwrap();
        assert_eq!(spec.name, "ChildAgent");
        assert_eq!(spec.max_iterations, 10); // from base
        assert_eq!(spec.description, Some("Test description".to_string())); // from child with default
    }

    #[test]
    fn test_load_and_parse_with_defaults() {
        let temp_dir = TempDir::new().unwrap();
        let template = r#"
name: {{ agent_name }}
system_prompt: "{{ prompt | default('Default prompt') }}"
llm:
  provider: openai
  model: gpt-4
"#;
        create_template(temp_dir.path(), "test.yaml", template);

        let mut loader = TemplateLoader::new();
        loader.add_search_path(temp_dir.path());
        loader.set_variable("agent_name", "Test");
        // Don't set prompt - should use default

        let spec = loader.load_and_parse("test").unwrap();
        assert_eq!(spec.system_prompt, "Default prompt");
    }

    #[test]
    fn test_load_and_parse_validation_fails() {
        let temp_dir = TempDir::new().unwrap();
        let template = r#"
name: ""
system_prompt: "test"
llm:
  provider: openai
  model: gpt-4
"#;
        create_template(temp_dir.path(), "invalid.yaml", template);

        let mut loader = TemplateLoader::new();
        loader.add_search_path(temp_dir.path());

        // Should fail validation (empty name)
        let result = loader.load_and_parse("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_and_parse_with_env_var() {
        unsafe { std::env::set_var("TEST_MODEL", "gpt-4-turbo") };

        let temp_dir = TempDir::new().unwrap();
        let template = r#"
name: EnvTest
system_prompt: "test"
llm:
  provider: openai
  model: $ENV{TEST_MODEL}
"#;
        create_template(temp_dir.path(), "env.yaml", template);

        let mut loader = TemplateLoader::new();
        loader.add_search_path(temp_dir.path());

        let spec = loader.load_and_parse("env").unwrap();
        let llm_config = spec.llm.as_config().expect("Expected LLMConfig");
        assert_eq!(llm_config.model, "gpt-4-turbo");

        unsafe { std::env::remove_var("TEST_MODEL") };
    }
}

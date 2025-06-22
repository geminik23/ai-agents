//! Template inheritance system for extending base templates

use crate::error::{AgentError, Result};
use serde_yaml::Value;
use std::collections::HashSet;

pub struct TemplateInheritance;

impl TemplateInheritance {
    pub fn process<F>(template_content: &str, load_fn: F) -> Result<String>
    where
        F: Fn(&str) -> Result<String>,
    {
        let mut visited = HashSet::new();
        Self::process_recursive(template_content, &load_fn, &mut visited)
    }

    fn process_recursive<F>(
        template_content: &str,
        load_fn: &F,
        visited: &mut HashSet<String>,
    ) -> Result<String>
    where
        F: Fn(&str) -> Result<String>,
    {
        let value: Value = serde_yaml::from_str(template_content).map_err(|e| {
            AgentError::TemplateError(format!("Failed to parse template YAML: {}", e))
        })?;

        if let Some(extends) = value.get("extends") {
            let parent_name = extends
                .as_str()
                .ok_or_else(|| AgentError::TemplateError("extends must be a string".to_string()))?;
            if visited.contains(parent_name) {
                return Err(AgentError::TemplateError(format!(
                    "Circular template dependency detected: {}",
                    parent_name
                )));
            }
            visited.insert(parent_name.to_string());

            let parent_content = load_fn(parent_name)?;
            let processed_parent = Self::process_recursive(&parent_content, load_fn, visited)?;

            Self::merge_templates(&processed_parent, template_content)
        } else {
            Ok(template_content.to_string())
        }
    }

    fn merge_templates(parent: &str, child: &str) -> Result<String> {
        let mut parent_value: Value = serde_yaml::from_str(parent)
            .map_err(|e| AgentError::TemplateError(format!("Invalid parent YAML: {}", e)))?;

        let child_value: Value = serde_yaml::from_str(child)
            .map_err(|e| AgentError::TemplateError(format!("Invalid child YAML: {}", e)))?;

        let child_map = child_value
            .as_mapping()
            .ok_or_else(|| AgentError::TemplateError("Child must be a YAML mapping".to_string()))?;

        if let Some(parent_map) = parent_value.as_mapping_mut() {
            for (key, value) in child_map {
                // Skip the extends key
                if key.as_str() == Some("extends") {
                    continue;
                }

                parent_map.insert(key.clone(), value.clone());
            }
        }

        serde_yaml::to_string(&parent_value).map_err(|e| {
            AgentError::TemplateError(format!("Failed to serialize merged YAML: {}", e))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_no_extends() {
        let template = r#"
name: SimpleAgent
version: 1.0.0
"#;

        let load_fn = |_: &str| -> Result<String> {
            panic!("Should not be called");
        };

        let result = TemplateInheritance::process(template, load_fn).unwrap();
        assert!(result.contains("name: SimpleAgent"));
    }

    #[test]
    fn test_simple_extends() {
        let child = r#"
extends: base.yaml
name: ChildAgent
description: Child description
"#;

        let parent = r#"
name: BaseAgent
version: 1.0.0
system_prompt: Base prompt
"#;

        let mut templates = HashMap::new();
        templates.insert("base.yaml", parent);

        let load_fn = |name: &str| -> Result<String> {
            templates
                .get(name)
                .map(|s| s.to_string())
                .ok_or_else(|| AgentError::TemplateError(format!("Template {} not found", name)))
        };

        let result = TemplateInheritance::process(child, load_fn).unwrap();

        // Child values should override parent
        assert!(result.contains("name: ChildAgent"));
        assert!(result.contains("description: Child description"));
        // Parent values should be preserved
        assert!(result.contains("version: 1.0.0"));
        assert!(result.contains("system_prompt: Base prompt"));
    }

    #[test]
    fn test_nested_extends() {
        let grandchild = r#"
extends: child.yaml
name: GrandchildAgent
"#;

        let child = r#"
extends: base.yaml
name: ChildAgent
description: Child
"#;

        let base = r#"
name: BaseAgent
version: 1.0.0
system_prompt: Base
"#;

        let mut templates = HashMap::new();
        templates.insert("child.yaml", child);
        templates.insert("base.yaml", base);

        let load_fn = |name: &str| -> Result<String> {
            templates
                .get(name)
                .map(|s| s.to_string())
                .ok_or_else(|| AgentError::TemplateError(format!("Template {} not found", name)))
        };

        let result = TemplateInheritance::process(grandchild, load_fn).unwrap();

        // Grandchild overrides all
        assert!(result.contains("name: GrandchildAgent"));
        // Child's description preserved
        assert!(result.contains("description: Child"));
        // Base's version preserved
        assert!(result.contains("version: 1.0.0"));
    }

    #[test]
    fn test_circular_dependency() {
        let template_a = r#"
extends: b.yaml
name: A
"#;

        let template_b = r#"
extends: a.yaml
name: B
"#;

        let mut templates = HashMap::new();
        templates.insert("a.yaml", template_a);
        templates.insert("b.yaml", template_b);

        let load_fn = |name: &str| -> Result<String> {
            templates
                .get(name)
                .map(|s| s.to_string())
                .ok_or_else(|| AgentError::TemplateError(format!("Template {} not found", name)))
        };

        let result = TemplateInheritance::process(template_a, load_fn);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Circular"));
    }

    #[test]
    fn test_extends_not_found() {
        let template = r#"
extends: nonexistent.yaml
name: Test
"#;

        let load_fn =
            |_: &str| -> Result<String> { Err(AgentError::TemplateError("Not found".to_string())) };

        let result = TemplateInheritance::process(template, load_fn);
        assert!(result.is_err());
    }
}

//! PersonaEvolveTool - built-in tool for LLM-driven persona evolution.
use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use ai_agents_core::{Tool, ToolResult};

use crate::manager::PersonaManager;

/// Metadata key for passing PersonaChange records from PersonaEvolveTool to RuntimeAgent.
pub const PERSONA_CHANGE_METADATA_KEY: &str = "persona_change";

/// Built-in tool for LLM-driven persona evolution. Auto-registered when allow_llm_evolve is true.
pub struct PersonaEvolveTool {
    persona_manager: Arc<PersonaManager>,
}

impl PersonaEvolveTool {
    pub fn new(persona_manager: Arc<PersonaManager>) -> Self {
        Self { persona_manager }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct PersonaEvolveInput {
    /// Dot-notation path of the persona field to modify (e.g. 'traits.personality', 'traits.speaking_style', 'goals.primary').
    field: String,

    /// The new value for the field. Type must match the field (array for personality/values/fears/goals, string for speaking_style/name/role).
    value: Value,

    /// Why this change is happening (recorded in evolution history).
    reason: String,
}

#[async_trait]
impl Tool for PersonaEvolveTool {
    fn id(&self) -> &str {
        "persona_evolve"
    }

    fn name(&self) -> &str {
        "Persona Evolve"
    }

    fn description(&self) -> &str {
        "Modify a persona trait in response to a significant event. \
         Only fields listed in mutable_fields can be changed. \
         Input: {\"field\": \"traits.personality\", \"value\": [\"bold\", \"curious\"], \"reason\": \"Gained confidence\"}"
    }

    fn input_schema(&self) -> Value {
        let schema = schemars::schema_for!(PersonaEvolveInput);
        serde_json::to_value(schema).unwrap_or_default()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let field = match args.get("field").and_then(|v| v.as_str()) {
            Some(f) => f,
            None => {
                return ToolResult {
                    success: false,
                    output: "Error: 'field' parameter is required and must be a string".into(),
                    metadata: None,
                };
            }
        };

        let value = match args.get("value") {
            Some(v) => v.clone(),
            None => {
                return ToolResult {
                    success: false,
                    output: "Error: 'value' parameter is required".into(),
                    metadata: None,
                };
            }
        };

        let reason = args.get("reason").and_then(|v| v.as_str());

        match self.persona_manager.evolve(field, value, reason) {
            Ok(change) => {
                let change_json = serde_json::to_value(&change).ok();
                let mut metadata = HashMap::new();
                if let Some(change_val) = change_json {
                    metadata.insert(PERSONA_CHANGE_METADATA_KEY.to_string(), change_val);
                }

                ToolResult {
                    success: true,
                    output: format!(
                        "Persona field '{}' updated successfully. Reason: {}",
                        field,
                        reason.unwrap_or("(none)")
                    ),
                    metadata: Some(metadata),
                }
            }
            Err(e) => ToolResult {
                success: false,
                output: format!("Error evolving persona: {}", e),
                metadata: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use ai_agents_context::TemplateRenderer;
    use serde_json::json;

    fn make_evolve_manager() -> Arc<PersonaManager> {
        let config = PersonaConfig {
            identity: Some(PersonaIdentity {
                name: "Test".into(),
                role: "Tester".into(),
                description: None,
                backstory: None,
                affiliation: None,
            }),
            traits: Some(PersonaTraits {
                personality: vec!["curious".into()],
                values: None,
                fears: None,
                speaking_style: Some("formal".into()),
            }),
            goals: None,
            secrets: None,
            evolution: Some(EvolutionConfig {
                enabled: true,
                mutable_fields: vec!["traits.personality".into(), "traits.speaking_style".into()],
                track_changes: true,
                allow_llm_evolve: true,
            }),
            templates: None,
            max_prompt_tokens: None,
        };

        let renderer = TemplateRenderer::new();
        Arc::new(PersonaManager::from_config(config, None, renderer).unwrap())
    }

    #[test]
    fn test_tool_metadata() {
        let tool = PersonaEvolveTool::new(make_evolve_manager());
        assert_eq!(tool.id(), "persona_evolve");
        assert_eq!(tool.name(), "Persona Evolve");
        assert!(tool.description().contains("mutable_fields"));

        let schema = tool.input_schema();
        assert!(schema.get("properties").is_some());
        assert!(schema["properties"].get("field").is_some());
        assert!(schema["properties"].get("value").is_some());
        assert!(schema["properties"].get("reason").is_some());
    }

    #[tokio::test]
    async fn test_tool_execute_success() {
        let manager = make_evolve_manager();
        let tool = PersonaEvolveTool::new(manager.clone());

        let args = json!({
            "field": "traits.personality",
            "value": ["bold", "brave"],
            "reason": "Gained confidence in battle"
        });

        let result = tool.execute(args).await;
        assert!(result.success);
        assert!(result.output.contains("updated successfully"));

        // Check metadata contains the persona change.
        let metadata = result.metadata.unwrap();
        assert!(metadata.contains_key(PERSONA_CHANGE_METADATA_KEY));

        let change: crate::evolution::PersonaChange =
            serde_json::from_value(metadata[PERSONA_CHANGE_METADATA_KEY].clone()).unwrap();
        assert_eq!(change.field, "traits.personality");

        // Verify the manager state was actually updated.
        let identity = manager.identity();
        assert_eq!(identity.name, "Test");
    }

    #[tokio::test]
    async fn test_tool_execute_invalid_field() {
        let tool = PersonaEvolveTool::new(make_evolve_manager());

        let args = json!({
            "field": "identity.name",
            "value": "New Name",
            "reason": "Testing"
        });

        let result = tool.execute(args).await;
        assert!(!result.success);
        assert!(result.output.contains("not in persona mutable_fields"));
    }

    #[tokio::test]
    async fn test_tool_execute_missing_field() {
        let tool = PersonaEvolveTool::new(make_evolve_manager());

        let args = json!({
            "value": "something",
            "reason": "Testing"
        });

        let result = tool.execute(args).await;
        assert!(!result.success);
        assert!(result.output.contains("'field' parameter is required"));
    }

    #[tokio::test]
    async fn test_tool_execute_missing_value() {
        let tool = PersonaEvolveTool::new(make_evolve_manager());

        let args = json!({
            "field": "traits.personality",
            "reason": "Testing"
        });

        let result = tool.execute(args).await;
        assert!(!result.success);
        assert!(result.output.contains("'value' parameter is required"));
    }

    #[tokio::test]
    async fn test_tool_execute_secrets_rejected() {
        let tool = PersonaEvolveTool::new(make_evolve_manager());

        let args = json!({
            "field": "secrets",
            "value": [],
            "reason": "Trying to cheat"
        });

        let result = tool.execute(args).await;
        assert!(!result.success);
    }

    #[test]
    fn test_persona_change_metadata_key_constant() {
        assert_eq!(PERSONA_CHANGE_METADATA_KEY, "persona_change");
    }
}

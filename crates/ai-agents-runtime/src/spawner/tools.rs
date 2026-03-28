//! Built-in tools for inter-agent messaging and dynamic agent management.

use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use ai_agents_core::{ChatMessage, LLMProvider, Tool, ToolResult};
use ai_agents_llm::LLMRegistry;
use ai_agents_tools::generate_schema;

use super::registry::AgentRegistry;
use super::spawner::AgentSpawner;

//
// GenerateAgentTool
//

/// Tool that lets a parent agent generate and spawn a new agent.
///
/// The tool description is built dynamically from template metadata so the LLM can discover available templates and their variables automatically.
pub struct GenerateAgentTool {
    spawner: Arc<AgentSpawner>,
    registry: Arc<AgentRegistry>,
    llm: Arc<LLMRegistry>,
    /// Pre-built description including available templates and their variables.
    enriched_description: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct GenerateAgentInput {
    /// Natural language description of the agent to create.
    description: String,
    /// Agent name / ID.
    name: String,
    /// Optional: named template to render instead of LLM generation.
    #[serde(default)]
    template: Option<String>,
}

impl GenerateAgentTool {
    pub fn new(
        spawner: Arc<AgentSpawner>,
        registry: Arc<AgentRegistry>,
        llm: Arc<LLMRegistry>,
    ) -> Self {
        let enriched_description = Self::build_description(&spawner);
        Self {
            spawner,
            registry,
            llm,
            enriched_description,
        }
    }

    /// Build tool description from template metadata (description + variables).
    fn build_description(spawner: &AgentSpawner) -> String {
        let mut desc = String::from(
            "Generate and spawn a new AI agent from a description. \
             Provide a natural language description of the agent's \
             personality, capabilities, and purpose.",
        );

        let templates = spawner.templates();
        if templates.is_empty() {
            return desc;
        }

        desc.push_str("\n\nAvailable templates (pass name as \"template\" field):");
        for (name, tpl) in templates {
            desc.push_str("\n  ");
            desc.push_str(name);
            if let Some(ref d) = tpl.description {
                desc.push_str(": ");
                desc.push_str(d);
            }
            if let Some(ref vars) = tpl.variables {
                for (var_name, var_desc) in vars {
                    desc.push_str("\n    - ");
                    desc.push_str(var_name);
                    desc.push_str(": ");
                    desc.push_str(var_desc);
                }
            }
        }

        desc.push_str(
            "\n\nWhen using a template, pass its variables as additional fields \
             alongside name and description.",
        );

        desc
    }
}

#[async_trait]
impl Tool for GenerateAgentTool {
    fn id(&self) -> &str {
        "generate_agent"
    }

    fn name(&self) -> &str {
        "Generate Agent"
    }

    fn description(&self) -> &str {
        &self.enriched_description
    }

    fn input_schema(&self) -> Value {
        generate_schema::<GenerateAgentInput>()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let description = match args.get("description").and_then(|v| v.as_str()) {
            Some(d) => d,
            None => return ToolResult::error("missing required field: description"),
        };
        let name = match args.get("name").and_then(|v| v.as_str()) {
            Some(n) => n,
            None => return ToolResult::error("missing required field: name"),
        };
        let template = args.get("template").and_then(|v| v.as_str());

        //
        // Template path
        if let Some(tpl_name) = template {
            let mut vars = std::collections::HashMap::new();
            vars.insert("name".to_string(), name.to_string());
            vars.insert("description".to_string(), description.to_string());

            // Forward any extra top-level string fields as template variables.
            if let Some(obj) = args.as_object() {
                for (k, v) in obj {
                    if k == "description" || k == "name" || k == "template" {
                        continue;
                    }
                    if let Some(s) = v.as_str() {
                        vars.insert(k.clone(), s.to_string());
                    }
                }
            }

            return match self.spawner.spawn_from_template(tpl_name, vars).await {
                Ok(agent) => {
                    let id = agent.id.clone();
                    match self.registry.register(agent).await {
                        Ok(()) => ToolResult::ok(
                            json!({"id": id, "source": "template", "template": tpl_name})
                                .to_string(),
                        ),
                        Err(e) => ToolResult::error(format!("registry error: {}", e)),
                    }
                }
                Err(e) => ToolResult::error(format!("template spawn failed: {}", e)),
            };
        }

        //
        // LLM generation
        let llm: Arc<dyn LLMProvider> = match self.llm.router() {
            Ok(l) => l,
            Err(_) => match self.llm.default() {
                Ok(l) => l,
                Err(e) => return ToolResult::error(format!("no LLM available: {}", e)),
            },
        };

        let prompt = build_generation_prompt(name, description);
        let messages = vec![ChatMessage::user(prompt)];

        let yaml = match llm.complete(&messages, None).await {
            Ok(resp) => strip_code_fences(&resp.content),
            Err(e) => return ToolResult::error(format!("LLM generation failed: {}", e)),
        };

        // First attempt: parse and spawn.
        match self.spawner.spawn_from_yaml(&yaml).await {
            Ok(agent) => {
                let id = agent.id.clone();
                return match self.registry.register(agent).await {
                    Ok(()) => {
                        ToolResult::ok(json!({"id": id, "source": "llm_generated"}).to_string())
                    }
                    Err(e) => ToolResult::error(format!("registry error: {}", e)),
                };
            }
            Err(first_err) => {
                // Retry once with an error-correction prompt.
                let retry_prompt = format!(
                    "The YAML you generated was invalid:\n{}\n\nError: {}\n\n\
                     Please fix the YAML and return ONLY valid YAML with no markdown fences.",
                    yaml, first_err
                );
                let retry_messages = vec![
                    ChatMessage::user(build_generation_prompt(name, description)),
                    ChatMessage::assistant(&yaml),
                    ChatMessage::user(retry_prompt),
                ];

                let retry_yaml = match llm.complete(&retry_messages, None).await {
                    Ok(resp) => strip_code_fences(&resp.content),
                    Err(e) => {
                        return ToolResult::error(format!(
                            "LLM retry failed: {} (original error: {})",
                            e, first_err
                        ));
                    }
                };

                match self.spawner.spawn_from_yaml(&retry_yaml).await {
                    Ok(agent) => {
                        let id = agent.id.clone();
                        match self.registry.register(agent).await {
                            Ok(()) => ToolResult::ok(
                                json!({"id": id, "source": "llm_generated", "retried": true})
                                    .to_string(),
                            ),
                            Err(e) => ToolResult::error(format!("registry error: {}", e)),
                        }
                    }
                    Err(e) => ToolResult::error(format!(
                        "spawn failed after retry: {} (original: {})",
                        e, first_err
                    )),
                }
            }
        }
    }
}

/// Build the YAML-generation prompt sent to the LLM.
fn build_generation_prompt(name: &str, description: &str) -> String {
    format!(
        "Generate a valid YAML agent specification.\n\n\
         Required fields:\n\
         - name: string (the agent's name)\n\
         - system_prompt: string (detailed behavioral instructions)\n\n\
         Optional fields: memory (type, max_messages, compress_threshold), \
         reasoning (mode: auto|cot|react), disambiguation (enabled: true/false).\n\n\
         Example:\n\
         ```yaml\n\
         name: Helper\n\
         system_prompt: |\n\
           You are a helpful assistant who answers concisely.\n\
         memory:\n\
           type: compacting\n\
           max_messages: 100\n\
           compress_threshold: 20\n\
         ```\n\n\
         Now generate a spec for:\n\
         Name: {}\n\
         Description: {}\n\n\
         Return ONLY the YAML content. No markdown fences, no commentary.",
        name, description
    )
}

/// Strip optional markdown code fences from LLM output.
fn strip_code_fences(text: &str) -> String {
    let trimmed = text.trim();
    let trimmed = trimmed
        .strip_prefix("```yaml")
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed);
    let trimmed = trimmed.strip_suffix("```").unwrap_or(trimmed);
    trimmed.trim().to_string()
}

//
// SendMessageTool
//

/// Tool that sends a message from the owning agent to another registered agent.
pub struct SendMessageTool {
    registry: Arc<AgentRegistry>,
    /// ID of the agent that owns this tool (the sender).
    sender_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct SendMessageInput {
    /// Target agent ID.
    to: String,
    /// Message to send.
    message: String,
}

impl SendMessageTool {
    pub fn new(registry: Arc<AgentRegistry>, sender_id: impl Into<String>) -> Self {
        Self {
            registry,
            sender_id: sender_id.into(),
        }
    }
}

#[async_trait]
impl Tool for SendMessageTool {
    fn id(&self) -> &str {
        "send_message"
    }

    fn name(&self) -> &str {
        "Send Message"
    }

    fn description(&self) -> &str {
        "Send a message to another registered agent and receive its response."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<SendMessageInput>()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let to = match args.get("to").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => return ToolResult::error("missing required field: to"),
        };
        let message = match args.get("message").and_then(|v| v.as_str()) {
            Some(m) => m,
            None => return ToolResult::error("missing required field: message"),
        };

        match self.registry.send(&self.sender_id, to, message).await {
            Ok(response) => {
                ToolResult::ok(json!({"from": to, "response": response.content}).to_string())
            }
            Err(e) => ToolResult::error(format!("send failed: {}", e)),
        }
    }
}

//
// ListAgentsTool
//

/// Tool that lists all agents currently registered in the registry.
pub struct ListAgentsTool {
    registry: Arc<AgentRegistry>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct ListAgentsInput {}

impl ListAgentsTool {
    pub fn new(registry: Arc<AgentRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for ListAgentsTool {
    fn id(&self) -> &str {
        "list_agents"
    }

    fn name(&self) -> &str {
        "List Agents"
    }

    fn description(&self) -> &str {
        "List all currently registered agents with their IDs and names."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<ListAgentsInput>()
    }

    async fn execute(&self, _args: Value) -> ToolResult {
        let agents = self.registry.list();
        match serde_json::to_string(&agents) {
            Ok(json) => ToolResult::ok(json),
            Err(e) => ToolResult::error(format!("serialization error: {}", e)),
        }
    }
}

//
// RemoveAgentTool
//

/// Tool that removes an agent from the registry by ID.
pub struct RemoveAgentTool {
    registry: Arc<AgentRegistry>,
    spawner: Option<Arc<AgentSpawner>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct RemoveAgentInput {
    /// Agent ID to remove.
    id: String,
}

impl RemoveAgentTool {
    pub fn new(registry: Arc<AgentRegistry>) -> Self {
        Self {
            registry,
            spawner: None,
        }
    }

    /// If provided, the spawner's agent count is decremented on removal.
    pub fn with_spawner(mut self, spawner: Arc<AgentSpawner>) -> Self {
        self.spawner = Some(spawner);
        self
    }
}

#[async_trait]
impl Tool for RemoveAgentTool {
    fn id(&self) -> &str {
        "remove_agent"
    }

    fn name(&self) -> &str {
        "Remove Agent"
    }

    fn description(&self) -> &str {
        "Remove a registered agent by its ID."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<RemoveAgentInput>()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let id = match args.get("id").and_then(|v| v.as_str()) {
            Some(i) => i,
            None => return ToolResult::error("missing required field: id"),
        };

        match self.registry.remove(id).await {
            Some(removed) => {
                if let Some(ref spawner) = self.spawner {
                    spawner.notify_agent_removed();
                }
                ToolResult::ok(json!({"removed": true, "id": removed.id}).to_string())
            }
            None => ToolResult::error(format!("agent not found: {}", id)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::spawner::ResolvedTemplate;
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_strip_code_fences_yaml() {
        let input = "```yaml\nname: Test\nsystem_prompt: hi\n```";
        assert_eq!(strip_code_fences(input), "name: Test\nsystem_prompt: hi");
    }

    #[test]
    fn test_strip_code_fences_bare() {
        let input = "```\nname: Test\n```";
        assert_eq!(strip_code_fences(input), "name: Test");
    }

    #[test]
    fn test_strip_code_fences_none() {
        let input = "name: Test\nsystem_prompt: hi";
        assert_eq!(strip_code_fences(input), input);
    }

    #[test]
    fn test_build_generation_prompt_contains_name() {
        let prompt = build_generation_prompt("Gormund", "A gruff blacksmith");
        assert!(prompt.contains("Gormund"));
        assert!(prompt.contains("gruff blacksmith"));
    }

    #[test]
    fn test_tool_ids_are_unique() {
        let ids = [
            "generate_agent",
            "send_message",
            "list_agents",
            "remove_agent",
        ];
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len());
    }

    #[test]
    fn test_build_description_no_templates() {
        let spawner = AgentSpawner::new();
        let desc = GenerateAgentTool::build_description(&spawner);
        assert!(desc.contains("Generate and spawn"));
        assert!(!desc.contains("Available templates"));
    }

    #[test]
    fn test_build_description_with_templates() {
        let mut templates = HashMap::new();
        templates.insert(
            "npc_base".to_string(),
            ResolvedTemplate {
                content: "name: test".to_string(),
                description: Some("General-purpose NPC".to_string()),
                variables: Some({
                    let mut v = HashMap::new();
                    v.insert("role".to_string(), "NPC occupation".to_string());
                    v.insert(
                        "personality".to_string(),
                        "Personality description".to_string(),
                    );
                    v
                }),
            },
        );
        let spawner = AgentSpawner::new().with_templates(templates);
        let desc = GenerateAgentTool::build_description(&spawner);
        assert!(desc.contains("Available templates"));
        assert!(desc.contains("npc_base"));
        assert!(desc.contains("General-purpose NPC"));
        assert!(desc.contains("role"));
        assert!(desc.contains("NPC occupation"));
        assert!(desc.contains("personality"));
    }

    #[test]
    fn test_build_description_template_no_metadata() {
        let mut templates = HashMap::new();
        templates.insert(
            "bare".to_string(),
            ResolvedTemplate {
                content: "name: test".to_string(),
                description: None,
                variables: None,
            },
        );
        let spawner = AgentSpawner::new().with_templates(templates);
        let desc = GenerateAgentTool::build_description(&spawner);
        assert!(desc.contains("Available templates"));
        assert!(desc.contains("bare"));
        // No description or variables appended
        assert!(!desc.contains("NPC"));
    }

    #[test]
    fn test_generate_agent_schema_has_required_fields() {
        let schema = generate_schema::<GenerateAgentInput>();
        let props = schema.get("properties").expect("should have properties");
        assert!(props.get("description").is_some());
        assert!(props.get("name").is_some());
        assert!(props.get("template").is_some());
        let required = schema.get("required").expect("should have required");
        let req_arr: Vec<&str> = required
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(req_arr.contains(&"description"));
        assert!(req_arr.contains(&"name"));
        // template is optional (Option<String>), should not be required
        assert!(!req_arr.contains(&"template"));
    }

    #[test]
    fn test_send_message_schema_has_required_fields() {
        let schema = generate_schema::<SendMessageInput>();
        let props = schema.get("properties").expect("should have properties");
        assert!(props.get("to").is_some());
        assert!(props.get("message").is_some());
    }

    #[test]
    fn test_remove_agent_schema_has_id() {
        let schema = generate_schema::<RemoveAgentInput>();
        let props = schema.get("properties").expect("should have properties");
        assert!(props.get("id").is_some());
    }

    #[test]
    fn test_list_agents_schema_is_object() {
        let schema = generate_schema::<ListAgentsInput>();
        assert_eq!(schema.get("type").and_then(|v| v.as_str()), Some("object"));
    }
}

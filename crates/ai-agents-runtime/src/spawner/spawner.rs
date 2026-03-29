//! Core agent spawner for creating agents at runtime.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use chrono::Utc;
use minijinja::Environment;
use tracing::info;

use crate::AgentBuilder;
use crate::RuntimeAgent;
use crate::spec::AgentSpec;
use ai_agents_core::{AgentError, AgentStorage, Result};
use ai_agents_llm::LLMRegistry;

use super::storage::NamespacedStorage;

/// A spawner template with its raw content and extracted metadata.
#[derive(Debug, Clone)]
pub struct ResolvedTemplate {
    /// Raw Jinja2 template string for rendering.
    pub content: String,
    /// Template description extracted from the `description:` field.
    pub description: Option<String>,
    /// Variable name -> description map extracted from `metadata.template.variables`.
    pub variables: Option<HashMap<String, String>>,
}

impl ResolvedTemplate {
    /// Create a ResolvedTemplate from a plain content string with no metadata.
    pub fn from_content(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            description: None,
            variables: None,
        }
    }
}

/// Metadata for a spawned agent.
pub struct SpawnedAgent {
    /// Unique identifier (derived from spec name or auto-generated).
    pub id: String,
    /// The runtime agent, wrapped in Arc for shared ownership across registry callers.
    pub agent: Arc<RuntimeAgent>,
    /// Retained spec for introspection and serialization.
    pub spec: AgentSpec,
    /// Timestamp when the agent was created.
    pub spawned_at: chrono::DateTime<Utc>,
}

impl std::fmt::Debug for SpawnedAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpawnedAgent")
            .field("id", &self.id)
            .field("spawned_at", &self.spawned_at)
            .finish_non_exhaustive()
    }
}

/// Factory for creating agents at runtime from YAML, specs, or templates.
pub struct AgentSpawner {
    /// Shared LLM regstry - spawned agents reuse these connections.
    llm_registry: Option<LLMRegistry>,

    /// Shared storage backend with per-gaent `NamespacedStorage` warpping.
    storage: Option<Arc<dyn AgentStorage>>,

    /// Context values injected into every spawned agent.
    shared_context: HashMap<String, serde_json::Value>,

    /// Hard limit on the number of agents this spawner may create.
    max_agents: Option<usize>,

    /// Auto-naming prefix (e.g. "npc_" produces "npc_001", "npc_002").
    name_prefix: Option<String>,

    /// Named YAML templates with content and extracted metadata.
    templates: HashMap<String, ResolvedTemplate>,

    /// Tool names that spawned agents are allowed to declare.
    allowed_tools: Option<Vec<String>>,

    /// Monotonic counter for auto-naming.
    counter: AtomicU32,

    /// Running count of agents spawned (for limit enforcement).
    agent_count: AtomicU32,
}

impl AgentSpawner {
    pub fn new() -> Self {
        Self {
            llm_registry: None,
            storage: None,
            shared_context: HashMap::new(),
            max_agents: None,
            name_prefix: None,
            templates: HashMap::new(),
            allowed_tools: None,
            counter: AtomicU32::new(1),
            agent_count: AtomicU32::new(0),
        }
    }

    /// Share LLM connections across all spawned agents.
    pub fn with_shared_llms(mut self, registry: LLMRegistry) -> Self {
        self.llm_registry = Some(registry);
        self
    }

    /// Share a storage backend (e.g. one SQLite DB for all NPCs).
    pub fn with_shared_storage(mut self, storage: Arc<dyn AgentStorage>) -> Self {
        self.storage = Some(storage);
        self
    }

    /// Inject a context value available to all spawned agents.
    pub fn with_shared_context(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.shared_context.insert(key.into(), value);
        self
    }

    /// Inject an entire map of shared context values.
    pub fn with_shared_context_map(mut self, ctx: HashMap<String, serde_json::Value>) -> Self {
        self.shared_context.extend(ctx);
        self
    }

    /// Limit total spawned agents.
    pub fn with_max_agents(mut self, max: usize) -> Self {
        self.max_agents = Some(max);
        self
    }

    /// Auto-name agents with prefix + zero-padded counter.
    pub fn with_name_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.name_prefix = Some(prefix.into());
        self
    }

    /// Register a named template from a plain YAML string (no metadata).
    pub fn with_template(
        mut self,
        name: impl Into<String>,
        yaml_template: impl Into<String>,
    ) -> Self {
        self.templates
            .insert(name.into(), ResolvedTemplate::from_content(yaml_template));
        self
    }

    /// Bulk-register resolved templates (with metadata already extracted).
    pub fn with_templates(mut self, templates: HashMap<String, ResolvedTemplate>) -> Self {
        self.templates.extend(templates);
        self
    }

    /// Set the tool allowlist for spawned agents.
    pub fn with_allowed_tools(mut self, tools: Vec<String>) -> Self {
        self.allowed_tools = Some(tools);
        self
    }

    /// Spawn an agent from a YAML string.
    pub async fn spawn_from_yaml(&self, yaml: &str) -> Result<SpawnedAgent> {
        let mut spec: AgentSpec = serde_yaml::from_str(yaml)?;
        spec.validate()?;
        self.enforce_tool_allowlist(&mut spec);
        self.spawn_from_spec(spec).await
    }

    /// Spawn an agent from a pre-built AgentSpec.
    pub async fn spawn_from_spec(&self, spec: AgentSpec) -> Result<SpawnedAgent> {
        let agent_id = self.generate_id(&spec.name);
        self.spawn_inner(agent_id, spec).await
    }

    /// Spawn an agent with a specific ID, used for session restore.
    pub async fn spawn_with_id(&self, id: String, spec: AgentSpec) -> Result<SpawnedAgent> {
        self.spawn_inner(id, spec).await
    }

    /// Internal spawn with an explicit ID. All builder wiring lives here.
    async fn spawn_inner(&self, agent_id: String, spec: AgentSpec) -> Result<SpawnedAgent> {
        self.check_spawn_limit()?;

        // Build the agent through the standard runtime pipeline.
        let mut builder = AgentBuilder::from_spec(spec.clone());

        // Inject shared LLM registry as a base; spec-specific LLMs override it.
        if let Some(ref shared_reg) = self.llm_registry {
            builder = builder.llm_registry(shared_reg.clone());
        }

        // Auto-configure spec-specific LLMs only when the spec declares providers.
        if !spec.llms.is_empty() {
            builder = builder.auto_configure_llms()?;
        }

        // Wire up recovery, tool security, process pipeline, and built-in tools.
        builder = builder.auto_configure_features()?;

        // Shared storage with per-agent namespacing.
        if let Some(ref shared_storage) = self.storage {
            let namespaced = Arc::new(NamespacedStorage::new(
                Arc::clone(shared_storage),
                &agent_id,
            ));
            builder = builder.storage(namespaced);
        }

        let agent = builder.build()?;

        // Inject shared context values into the agent's context manager.
        for (key, value) in &self.shared_context {
            let _ = agent.set_context(key, value.clone());
        }

        self.agent_count.fetch_add(1, Ordering::Relaxed);

        info!(agent_id = %agent_id, name = %spec.name, "Agent spawned");

        Ok(SpawnedAgent {
            id: agent_id,
            agent: Arc::new(agent),
            spec,
            spawned_at: Utc::now(),
        })
    }

    /// Spawn from a named template with caller-provided variables.
    ///
    /// Template rendering merges two namespaces:
    /// - Caller variables: top-level (`{{ name }}`, `{{ role }}`)
    /// - Shared context: under `context.` prefix (`{{ context.world_name }}`)
    pub async fn spawn_from_template(
        &self,
        template_name: &str,
        variables: HashMap<String, String>,
    ) -> Result<SpawnedAgent> {
        let template = self.templates.get(template_name).ok_or_else(|| {
            AgentError::Config(format!("Spawner template not found: {}", template_name))
        })?;

        let rendered = self.render_template(&template.content, &variables)?;
        self.spawn_from_yaml(&rendered).await
    }

    /// Returns the current number of agents that have been spawned.
    pub fn spawned_count(&self) -> u32 {
        self.agent_count.load(Ordering::Relaxed)
    }

    /// Decrement the agent count (called when an agent is removed from the registry).
    pub fn notify_agent_removed(&self) {
        let prev = self.agent_count.load(Ordering::Relaxed);
        if prev > 0 {
            self.agent_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Returns a reference to the shared LLM registry, if configured.
    pub fn llm_registry(&self) -> Option<&LLMRegistry> {
        self.llm_registry.as_ref()
    }

    /// Returns a reference to the shared storage, if configured.
    pub fn shared_storage(&self) -> Option<&Arc<dyn AgentStorage>> {
        self.storage.as_ref()
    }

    /// Returns a reference to the resolved template map.
    pub fn templates(&self) -> &HashMap<String, ResolvedTemplate> {
        &self.templates
    }

    fn check_spawn_limit(&self) -> Result<()> {
        if let Some(max) = self.max_agents {
            let current = self.agent_count.load(Ordering::Relaxed) as usize;
            if current >= max {
                return Err(AgentError::Config(format!(
                    "Spawn limit exceeded: {}/{}",
                    current, max
                )));
            }
        }
        Ok(())
    }

    fn generate_id(&self, spec_name: &str) -> String {
        if let Some(ref prefix) = self.name_prefix {
            let n = self.counter.fetch_add(1, Ordering::Relaxed);
            format!("{}{:03}", prefix, n)
        } else {
            spec_name.to_lowercase().replace(' ', "_")
        }
    }

    /// Strip tools from the spec that are not in the allowlist.
    fn enforce_tool_allowlist(&self, spec: &mut AgentSpec) {
        if let Some(ref allowed) = self.allowed_tools {
            if let Some(ref mut tools) = spec.tools {
                let before = tools.len();
                tools.retain(|t| allowed.contains(&t.name().to_string()));
                let removed = before - tools.len();
                if removed > 0 {
                    tracing::warn!(
                        removed_count = removed,
                        "Stripped disallowed tools from spawned agent spec"
                    );
                }
            }
        }
    }

    /// Render a template string with caller variables and shared context.
    fn render_template(
        &self,
        template_str: &str,
        variables: &HashMap<String, String>,
    ) -> Result<String> {
        let mut env = Environment::new();
        env.add_template("_spawn", template_str)
            .map_err(|e| AgentError::TemplateError(format!("template parse error: {}", e)))?;

        let tmpl = env
            .get_template("_spawn")
            .map_err(|e| AgentError::TemplateError(format!("template load error: {}", e)))?;

        // Caller variables are top-level; shared context lives under "context".
        let mut ctx = serde_json::Map::new();

        for (k, v) in variables {
            ctx.insert(k.clone(), serde_json::Value::String(v.clone()));
        }

        // Shared context as a nested object so {{ context.world_name }} works.
        let context_obj = serde_json::Value::Object(
            self.shared_context
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        );
        ctx.insert("context".to_string(), context_obj);

        let ctx_value = serde_json::Value::Object(ctx);
        let mj_value = minijinja::Value::from_serialize(&ctx_value);

        tmpl.render(mj_value)
            .map_err(|e| AgentError::TemplateError(format!("template render error: {}", e)))
    }
}

impl Default for AgentSpawner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_id_with_prefix() {
        let spawner = AgentSpawner::new().with_name_prefix("npc_");
        assert_eq!(spawner.generate_id("Gormund"), "npc_001");
        assert_eq!(spawner.generate_id("Elena"), "npc_002");
    }

    #[test]
    fn test_generate_id_without_prefix() {
        let spawner = AgentSpawner::new();
        assert_eq!(spawner.generate_id("My Agent"), "my_agent");
        assert_eq!(spawner.generate_id("TestBot"), "testbot");
    }

    #[test]
    fn test_check_spawn_limit() {
        let spawner = AgentSpawner::new().with_max_agents(2);
        assert!(spawner.check_spawn_limit().is_ok());
        spawner.agent_count.store(2, Ordering::Relaxed);
        assert!(spawner.check_spawn_limit().is_err());
    }

    #[test]
    fn test_render_template_basic() {
        let spawner = AgentSpawner::new()
            .with_shared_context("world_name", serde_json::json!("Fantasy Land"));

        let template =
            "name: {{ name }}\nsystem_prompt: You are {{ name }} in {{ context.world_name }}.";
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "Gormund".to_string());

        let rendered = spawner.render_template(template, &vars).unwrap();
        assert!(rendered.contains("name: Gormund"));
        assert!(rendered.contains("Fantasy Land"));
    }

    #[test]
    fn test_enforce_tool_allowlist() {
        let spawner = AgentSpawner::new()
            .with_allowed_tools(vec!["echo".to_string(), "calculator".to_string()]);

        let yaml = r#"
name: Test
system_prompt: test
tools:
  - echo
  - calculator
  - file
  - http
"#;
        let mut spec: AgentSpec = serde_yaml::from_str(yaml).unwrap();
        spawner.enforce_tool_allowlist(&mut spec);

        let tool_names: Vec<_> = spec
            .tools
            .as_ref()
            .unwrap()
            .iter()
            .map(|t| t.name().to_string())
            .collect();
        assert_eq!(tool_names, vec!["echo", "calculator"]);
    }

    #[test]
    fn test_with_template_plain_string() {
        let spawner =
            AgentSpawner::new().with_template("basic", "name: {{ name }}\nsystem_prompt: hi");
        let tpl = spawner.templates().get("basic").unwrap();
        assert_eq!(tpl.content, "name: {{ name }}\nsystem_prompt: hi");
        assert!(tpl.description.is_none());
        assert!(tpl.variables.is_none());
    }

    #[test]
    fn test_with_templates_resolved() {
        let mut templates = HashMap::new();
        templates.insert(
            "base".to_string(),
            ResolvedTemplate {
                content: "name: {{ name }}".to_string(),
                description: Some("Test template".to_string()),
                variables: Some({
                    let mut v = HashMap::new();
                    v.insert("role".to_string(), "occupation".to_string());
                    v
                }),
            },
        );
        let spawner = AgentSpawner::new().with_templates(templates);
        let tpl = spawner.templates().get("base").unwrap();
        assert_eq!(tpl.description.as_deref(), Some("Test template"));
        assert_eq!(
            tpl.variables.as_ref().unwrap().get("role").unwrap(),
            "occupation"
        );
    }
}

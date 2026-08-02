//! Core agent spawner for creating agents at runtime.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use chrono::Utc;
use minijinja::Environment;
use tracing::info;

use crate::AgentBuilder;
use crate::RuntimeAgent;
use crate::runtime::ToolResourceLocks;
use crate::spec::AgentSpec;
use ai_agents_core::{AgentError, AgentStorage, Result};
use ai_agents_llm::LLMRegistry;
use ai_agents_observability::ObservabilityManager;
use ai_agents_tools::create_builtin_registry;

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

pub(crate) struct CapacityReservation {
    agent_count: Arc<AtomicU32>,
    released: AtomicBool,
}

impl CapacityReservation {
    fn new(agent_count: Arc<AtomicU32>) -> Self {
        Self {
            agent_count,
            released: AtomicBool::new(false),
        }
    }

    fn release(&self) {
        // A slot may be released by registration failure, registry removal, or final drop. The flag keeps these competing lifecycle paths exactly once.
        if !self.released.swap(true, Ordering::AcqRel) {
            let _ = self
                .agent_count
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                    count.checked_sub(1)
                });
        }
    }
}

impl Drop for CapacityReservation {
    fn drop(&mut self) {
        self.release();
    }
}

struct AdmittedChild {
    id: String,
    spec: AgentSpec,
    base_dir: Option<PathBuf>,
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
    capacity_reservation: Option<CapacityReservation>,
}

impl SpawnedAgent {
    /// Construct a detached spawned-agent record for host-managed registration.
    pub fn from_runtime(id: String, agent: RuntimeAgent, spec: AgentSpec) -> Self {
        Self {
            id,
            agent: Arc::new(agent),
            spec,
            spawned_at: Utc::now(),
            capacity_reservation: None,
        }
    }

    fn tracked(
        id: String,
        agent: RuntimeAgent,
        spec: AgentSpec,
        capacity_reservation: CapacityReservation,
    ) -> Self {
        Self {
            id,
            agent: Arc::new(agent),
            spec,
            spawned_at: Utc::now(),
            capacity_reservation: Some(capacity_reservation),
        }
    }

    pub(super) fn release_capacity(&self) {
        if let Some(reservation) = self.capacity_reservation.as_ref() {
            reservation.release();
        }
    }

    #[cfg(test)]
    pub(crate) fn untracked(id: String, agent: RuntimeAgent, spec: AgentSpec) -> Self {
        Self::from_runtime(id, agent, spec)
    }
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

    /// Whether shared LLM providers are already observability-wrapped.
    llm_registry_observed: bool,

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

    /// Shared resource locks inherited by spawned child agents.
    resource_locks: Option<ToolResourceLocks>,

    /// Shared observability manager for spawned child agents.
    observability_manager: Option<Arc<ObservabilityManager>>,

    /// Monotonic counter for auto-naming.
    counter: AtomicU32,

    /// Running count of reserved or registered agents.
    agent_count: Arc<AtomicU32>,
}

impl AgentSpawner {
    pub fn new() -> Self {
        Self {
            llm_registry: None,
            llm_registry_observed: false,
            storage: None,
            shared_context: HashMap::new(),
            max_agents: None,
            name_prefix: None,
            templates: HashMap::new(),
            allowed_tools: None,
            resource_locks: None,
            observability_manager: None,
            counter: AtomicU32::new(1),
            agent_count: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Share LLM connections across all spawned agents.
    pub fn with_shared_llms(mut self, registry: LLMRegistry) -> Self {
        self.llm_registry = Some(registry);
        self.llm_registry_observed = false;
        self
    }

    /// Share an LLM registry that is already wrapped by observed providers.
    pub fn with_shared_observed_llms(mut self, registry: LLMRegistry) -> Self {
        self.llm_registry = Some(registry);
        self.llm_registry_observed = true;
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
    pub fn with_name_prefix(mut self, prefix: impl Into<String>) -> Result<Self> {
        let prefix = prefix.into();
        validate_name_prefix(&prefix)?;
        self.name_prefix = Some(prefix);
        Ok(self)
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

    /// Share the parent's resource lock table with spawned agents.
    pub(crate) fn with_resource_locks(mut self, locks: ToolResourceLocks) -> Self {
        self.resource_locks = Some(locks);
        self
    }

    /// Share the parent's observability manager with spawned agents.
    pub fn with_observability(mut self, manager: Arc<ObservabilityManager>) -> Self {
        self.observability_manager = Some(manager);
        self
    }

    /// Spawn an agent from a YAML string.
    pub async fn spawn_from_yaml(&self, yaml: &str) -> Result<SpawnedAgent> {
        let spec = AgentSpec::from_yaml_strict(yaml)?;
        self.spawn_admitted(None, spec, None).await
    }

    /// Spawn an agent from a pre-built AgentSpec.
    pub async fn spawn_from_spec(&self, spec: AgentSpec) -> Result<SpawnedAgent> {
        self.spawn_admitted(None, spec, None).await
    }

    /// Validate an explicit child ID and spec without reserving capacity or building an agent.
    #[doc(hidden)]
    pub fn validate_explicit_child(&self, id: &str, spec: &AgentSpec) -> Result<()> {
        self.validate_admission(id, spec)
    }

    /// Spawn an agent with a specific ID, used for session restore.
    pub async fn spawn_with_id(&self, id: String, spec: AgentSpec) -> Result<SpawnedAgent> {
        self.spawn_admitted(Some(id), spec, None).await
    }

    pub(crate) async fn spawn_from_yaml_file_with_id(
        &self,
        id: String,
        path: &Path,
    ) -> Result<SpawnedAgent> {
        let yaml = std::fs::read_to_string(path).map_err(AgentError::IoError)?;
        let spec = AgentSpec::from_yaml_strict(&yaml)?;
        let base_dir = path.parent().map(Path::to_path_buf);
        self.spawn_admitted(Some(id), spec, base_dir).await
    }

    async fn spawn_admitted(
        &self,
        explicit_id: Option<String>,
        spec: AgentSpec,
        base_dir: Option<PathBuf>,
    ) -> Result<SpawnedAgent> {
        let admitted = self.admit_spec(explicit_id, spec, base_dir)?;
        let reservation = self.reserve_capacity()?;
        self.spawn_inner(admitted, reservation).await
    }

    pub(crate) async fn spawn_with_reserved_capacity(
        &self,
        id: String,
        spec: AgentSpec,
        reservation: CapacityReservation,
    ) -> Result<SpawnedAgent> {
        let admitted = self.admit_spec(Some(id), spec, None)?;
        self.spawn_inner(admitted, reservation).await
    }

    fn admit_spec(
        &self,
        explicit_id: Option<String>,
        spec: AgentSpec,
        base_dir: Option<PathBuf>,
    ) -> Result<AdmittedChild> {
        let id = explicit_id.unwrap_or_else(|| self.generate_id(&spec.name));
        self.validate_admission(&id, &spec)?;
        Ok(AdmittedChild { id, spec, base_dir })
    }

    fn validate_admission(&self, id: &str, spec: &AgentSpec) -> Result<()> {
        spec.validate()?;
        validate_child_id(id)?;

        if spec
            .spawner
            .as_ref()
            .is_some_and(|config| config.is_configured())
        {
            return Err(AgentError::InvalidSpec(format!(
                "Spawned agent '{}' cannot configure an active nested spawner",
                id
            )));
        }

        self.validate_tool_allowlist(spec)?;
        self.validate_shared_llm_aliases(spec)
    }

    /// Internal spawn after all child admission checks have passed.
    async fn spawn_inner(
        &self,
        admitted: AdmittedChild,
        reservation: CapacityReservation,
    ) -> Result<SpawnedAgent> {
        let AdmittedChild { id, spec, base_dir } = admitted;

        let mut builder = match base_dir {
            Some(base_dir) => AgentBuilder::from_spec_with_base_dir(spec.clone(), base_dir),
            None => AgentBuilder::from_spec(spec.clone()),
        };

        if let Some(ref shared_reg) = self.llm_registry {
            // Shared providers are authoritative. Child provider declarations are never constructed or allowed to read credentials in this mode.
            builder =
                builder.authoritative_llm_registry(shared_reg.clone(), self.llm_registry_observed);
        } else {
            // Local mode configures both the single llm declaration and the multi-alias llms map.
            builder = builder.auto_configure_llms()?;
        }

        builder = builder.auto_configure_features()?;

        if let Some(ref manager) = self.observability_manager {
            builder = builder.observability(Arc::clone(manager));
        }
        if let Some(ref locks) = self.resource_locks {
            builder = builder.with_shared_resource_locks(Arc::clone(locks));
        }

        if let Some(ref shared_storage) = self.storage {
            let namespaced = Arc::new(NamespacedStorage::new(Arc::clone(shared_storage), &id));
            builder = builder.storage(namespaced);
        }

        let agent = builder.build()?;
        // A child is not ready for registration until required injected or configured storage capabilities and fact initialization have completed.
        agent.init_storage().await?;

        for (key, value) in &self.shared_context {
            agent.set_context(key, value.clone())?;
        }

        info!(agent_id = %id, name = %spec.name, "Agent spawned");
        Ok(SpawnedAgent::tracked(id, agent, spec, reservation))
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

    /// Returns the number of registered and in-flight reserved agents.
    pub fn spawned_count(&self) -> u32 {
        self.agent_count.load(Ordering::Relaxed)
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

    fn reserve_capacity(&self) -> Result<CapacityReservation> {
        self.reserve_restore_capacity(1, 0)?
            .pop()
            .ok_or_else(|| AgentError::Config("Spawn capacity reservation failed".to_string()))
    }

    pub(crate) fn reserve_restore_capacity(
        &self,
        additions: usize,
        removals: usize,
    ) -> Result<Vec<CapacityReservation>> {
        let additions = u32::try_from(additions)
            .map_err(|_| AgentError::Config("Spawn capacity request is too large".to_string()))?;
        let removals = u32::try_from(removals)
            .map_err(|_| AgentError::Config("Spawn removal count is too large".to_string()))?;

        loop {
            let current = self.agent_count.load(Ordering::Acquire);
            let after_removals = current.checked_sub(removals).ok_or_else(|| {
                AgentError::Config("Restore removal count exceeds reserved capacity".to_string())
            })?;
            let final_count = after_removals.checked_add(additions).ok_or_else(|| {
                AgentError::Config("Spawn capacity counter overflowed".to_string())
            })?;
            if self
                .max_agents
                .is_some_and(|max| final_count as usize > max)
            {
                return Err(AgentError::Config(format!(
                    "Spawn limit exceeded by restored topology: {}/{}",
                    final_count,
                    self.max_agents.unwrap()
                )));
            }
            if additions == 0 {
                return Ok(Vec::new());
            }

            //
            // Restore additions are reserved before staging. Temporary over-cap counts are allowed only when committed removals make the final topology fit.
            //
            let reserved_count = current.checked_add(additions).ok_or_else(|| {
                AgentError::Config("Spawn capacity counter overflowed".to_string())
            })?;
            match self.agent_count.compare_exchange_weak(
                current,
                reserved_count,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Ok((0..additions)
                        .map(|_| CapacityReservation::new(Arc::clone(&self.agent_count)))
                        .collect());
                }
                Err(_) => continue,
            }
        }
    }

    fn generate_id(&self, spec_name: &str) -> String {
        if let Some(ref prefix) = self.name_prefix {
            let n = self.counter.fetch_add(1, Ordering::Relaxed);
            return format!("{}{:03}", prefix, n);
        }

        let mut generated = String::with_capacity(spec_name.len());
        for character in spec_name.chars() {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.') {
                generated.push(character.to_ascii_lowercase());
            } else if !generated.ends_with('_') {
                generated.push('_');
            }
        }
        let generated = generated.trim_matches(['_', '-', '.']).to_string();
        if generated.is_empty() {
            let n = self.counter.fetch_add(1, Ordering::Relaxed);
            format!("agent_{n:03}")
        } else {
            generated
        }
    }

    fn validate_tool_allowlist(&self, spec: &AgentSpec) -> Result<()> {
        let Some(allowed) = self.allowed_tools.as_ref() else {
            return Ok(());
        };
        let registry = create_builtin_registry();
        let canonicalize = |tool: &str| {
            registry
                .canonical_id(tool)
                .unwrap_or_else(|| tool.to_string())
        };
        let allowed: BTreeSet<String> = allowed.iter().map(|tool| canonicalize(tool)).collect();
        let disallowed: BTreeSet<String> = spec
            .tools
            .iter()
            .flatten()
            .map(|tool| tool.name().to_string())
            .filter(|tool| !allowed.contains(&canonicalize(tool)))
            .collect();
        if disallowed.is_empty() {
            Ok(())
        } else {
            Err(AgentError::InvalidSpec(format!(
                "Spawned agent declares tools outside the spawner allowlist: {}",
                disallowed.into_iter().collect::<Vec<_>>().join(", ")
            )))
        }
    }

    fn validate_shared_llm_aliases(&self, spec: &AgentSpec) -> Result<()> {
        let Some(registry) = self.llm_registry.as_ref() else {
            return Ok(());
        };
        registry.default().map_err(|error| {
            AgentError::Config(format!(
                "Inherited LLM registry has no usable default: {error}"
            ))
        })?;

        let mut required: BTreeSet<String> = spec.llms.keys().cloned().collect();
        required.insert(spec.llm.get_default_alias());
        if let Some(router) = spec.llm.get_router_alias() {
            required.insert(router);
        }
        required.extend(spec.referenced_llm_aliases());
        let missing: Vec<String> = required
            .into_iter()
            .filter(|alias| !registry.has(alias))
            .collect();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(AgentError::InvalidSpec(format!(
                "Spawned agent requires inherited LLM alias(es) not present in the shared registry: {}",
                missing.join(", ")
            )))
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

fn validate_child_id(id: &str) -> Result<()> {
    if id.is_empty() {
        return Err(AgentError::InvalidSpec(
            "Spawned agent ID cannot be empty".to_string(),
        ));
    }
    if matches!(id, "." | "..") {
        return Err(AgentError::InvalidSpec(format!(
            "Spawned agent ID '{id}' is reserved"
        )));
    }
    if id.len() > 128 {
        return Err(AgentError::InvalidSpec(
            "Spawned agent ID cannot exceed 128 bytes".to_string(),
        ));
    }
    if !id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(AgentError::InvalidSpec(format!(
            "Spawned agent ID '{}' must contain only ASCII letters, digits, '_', '-' or '.'",
            id
        )));
    }
    if id.ends_with('.') {
        return Err(AgentError::InvalidSpec(
            "Spawned agent ID cannot end with a dot".to_string(),
        ));
    }
    let windows_stem = id.split('.').next().unwrap_or(id).to_ascii_uppercase();
    if matches!(
        windows_stem.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CLOCK$"
    ) || windows_stem
        .strip_prefix("COM")
        .is_some_and(|suffix| matches!(suffix, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"))
        || windows_stem.strip_prefix("LPT").is_some_and(|suffix| {
            matches!(suffix, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9")
        })
    {
        return Err(AgentError::InvalidSpec(format!(
            "Spawned agent ID '{id}' is reserved on Windows"
        )));
    }
    Ok(())
}

fn validate_name_prefix(prefix: &str) -> Result<()> {
    if prefix.is_empty() {
        return Err(AgentError::InvalidSpec(
            "Spawner name_prefix cannot be empty".to_string(),
        ));
    }
    validate_child_id(&format!("{prefix}001")).map_err(|error| {
        AgentError::InvalidSpec(format!("Invalid spawner name_prefix '{prefix}': {error}"))
    })
}

impl Default for AgentSpawner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::registry::AgentRegistry;
    use super::*;
    use ai_agents_llm::mock::MockLLMProvider;

    fn shared_registry() -> LLMRegistry {
        let mut registry = LLMRegistry::new();
        registry.register("default", Arc::new(MockLLMProvider::new("shared")));
        registry.set_default("default");
        registry
    }

    fn shared_spawner() -> AgentSpawner {
        AgentSpawner::new().with_shared_llms(shared_registry())
    }

    #[test]
    fn test_generate_id_with_prefix() {
        let spawner = AgentSpawner::new().with_name_prefix("npc_").unwrap();
        assert_eq!(spawner.generate_id("Gormund"), "npc_001");
        assert_eq!(spawner.generate_id("Elena"), "npc_002");
    }

    #[test]
    fn test_generate_id_without_prefix() {
        let spawner = AgentSpawner::new();
        assert_eq!(spawner.generate_id("My Agent"), "my_agent");
        assert_eq!(spawner.generate_id("Test.Bot"), "test.bot");
        assert_eq!(spawner.generate_id("작업자"), "agent_001");
        assert_eq!(spawner.generate_id("TestBot"), "testbot");
    }

    #[test]
    fn test_capacity_reservation_is_atomic_and_released_on_drop() {
        let spawner = AgentSpawner::new().with_max_agents(1);
        let reservation = spawner.reserve_capacity().unwrap();
        assert_eq!(spawner.spawned_count(), 1);
        assert!(spawner.reserve_capacity().is_err());
        drop(reservation);
        assert_eq!(spawner.spawned_count(), 0);
        assert!(spawner.reserve_capacity().is_ok());
    }

    #[test]
    fn restore_reservation_uses_committed_removal_credit_without_leaking() {
        let spawner = AgentSpawner::new().with_max_agents(2);
        let current = spawner.reserve_restore_capacity(2, 0).unwrap();
        assert_eq!(spawner.spawned_count(), 2);

        let replacement = spawner.reserve_restore_capacity(1, 1).unwrap();
        assert_eq!(spawner.spawned_count(), 3);
        drop(replacement);
        assert_eq!(spawner.spawned_count(), 2);
        drop(current);
        assert_eq!(spawner.spawned_count(), 0);
    }

    #[tokio::test]
    async fn test_concurrent_spawns_cannot_exceed_capacity() {
        let spawner = Arc::new(shared_spawner().with_max_agents(3));
        let barrier = Arc::new(tokio::sync::Barrier::new(9));
        let mut tasks = Vec::new();
        for index in 0..8 {
            let spawner = Arc::clone(&spawner);
            let barrier = Arc::clone(&barrier);
            tasks.push(tokio::spawn(async move {
                let spec = AgentSpec {
                    name: format!("Worker {index}"),
                    system_prompt: "worker".to_string(),
                    ..AgentSpec::default()
                };
                barrier.wait().await;
                spawner.spawn_with_id(format!("worker_{index}"), spec).await
            }));
        }
        barrier.wait().await;

        let mut spawned = Vec::new();
        let mut rejected = 0;
        for task in tasks {
            match task.await.unwrap() {
                Ok(agent) => spawned.push(agent),
                Err(error) => {
                    assert!(error.to_string().contains("Spawn limit exceeded"));
                    rejected += 1;
                }
            }
        }
        assert_eq!(spawned.len(), 3);
        assert_eq!(rejected, 5);
        assert_eq!(spawner.spawned_count(), 3);
        drop(spawned);
        assert_eq!(spawner.spawned_count(), 0);
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

    #[tokio::test]
    async fn test_tool_allowlist_rejects_instead_of_stripping() {
        let spawner =
            shared_spawner().with_allowed_tools(vec!["echo".to_string(), "calculator".to_string()]);
        let yaml = r#"
name: Test
system_prompt: test
tools:
  - echo
  - file
  - http
"#;
        let error = spawner.spawn_from_yaml(yaml).await.unwrap_err().to_string();
        assert!(error.contains("outside the spawner allowlist"), "{error}");
        assert!(error.contains("file"), "{error}");
        assert!(error.contains("http"), "{error}");
        assert_eq!(spawner.spawned_count(), 0);
    }

    #[test]
    fn test_validate_child_id_rejects_non_portable_values() {
        assert!(validate_child_id("worker-01").is_ok());
        assert!(validate_child_id("worker_01").is_ok());
        assert!(validate_child_id("worker.01").is_ok());
        assert!(validate_child_id("").is_err());
        assert!(validate_child_id(".").is_err());
        assert!(validate_child_id("..").is_err());
        assert!(validate_child_id("../worker").is_err());
        assert!(validate_child_id("worker name").is_err());
        assert!(validate_child_id("worker/01").is_err());
        assert!(validate_child_id("wörker").is_err());
        assert!(validate_child_id("worker.").is_err());
        for reserved in ["CON", "con.txt", "NUL", "COM1", "LPT9.log"] {
            assert!(validate_child_id(reserved).is_err(), "{reserved}");
        }
    }

    #[test]
    fn name_prefix_is_validated_when_configured() {
        assert!(AgentSpawner::new().with_name_prefix("worker_").is_ok());
        assert!(AgentSpawner::new().with_name_prefix("").is_err());
        assert!(AgentSpawner::new().with_name_prefix("bad/path").is_err());
        assert!(
            AgentSpawner::new()
                .with_name_prefix("x".repeat(126))
                .is_err()
        );
    }

    #[test]
    fn test_tool_allowlist_compares_builtin_canonical_identity() {
        let spawner = shared_spawner().with_allowed_tools(vec!["Copy Path".to_string()]);
        let spec = AgentSpec::from_yaml_strict(
            "name: Worker\nsystem_prompt: worker\ntools:\n  - copy_path\n",
        )
        .unwrap();

        spawner.validate_explicit_child("worker", &spec).unwrap();
    }

    #[tokio::test]
    async fn test_shared_llms_ignore_child_provider_declarations() {
        let spawner = shared_spawner();
        let yaml = r#"
name: SharedChild
system_prompt: shared
llm:
  provider: definitely-not-a-provider
  model: unavailable
"#;
        let spawned = spawner.spawn_from_yaml(yaml).await.unwrap();
        assert_eq!(spawned.agent.llm_registry().default_alias(), "default");
        drop(spawned);
        assert_eq!(spawner.spawned_count(), 0);
    }

    #[tokio::test]
    async fn test_shared_llms_require_declared_child_aliases() {
        let spawner = shared_spawner();
        let yaml = r#"
name: AliasChild
system_prompt: shared
llm:
  default: specialist
llms:
  specialist:
    provider: definitely-not-a-provider
    model: unavailable
"#;
        let error = spawner.spawn_from_yaml(yaml).await.unwrap_err().to_string();
        assert!(error.contains("specialist"), "{error}");
        assert_eq!(spawner.spawned_count(), 0);
    }

    #[test]
    fn test_shared_llms_require_explicit_subsystem_aliases() {
        let spawner = shared_spawner();
        let spec = AgentSpec::from_yaml_strict(
            "name: Worker\nsystem_prompt: worker\nmemory:\n  type: compacting\n  summarizer_llm: specialist\n",
        )
        .unwrap();

        let error = spawner
            .validate_explicit_child("worker", &spec)
            .unwrap_err()
            .to_string();
        assert!(error.contains("specialist"), "{error}");
    }

    #[tokio::test]
    async fn test_local_mode_configures_single_and_multi_declarations() {
        let single = r#"
name: SingleChild
system_prompt: local
llm:
  provider: ollama
  model: local-single
"#;
        let multi = r#"
name: MultiChild
system_prompt: local
llm:
  default: specialist
llms:
  specialist:
    provider: ollama
    model: local-multi
"#;
        let spawner = AgentSpawner::new();
        let single = spawner.spawn_from_yaml(single).await.unwrap();
        assert_eq!(
            single
                .agent
                .llm_registry()
                .default()
                .unwrap()
                .provider_name(),
            "ollama"
        );
        drop(single);

        let multi = spawner.spawn_from_yaml(multi).await.unwrap();
        assert!(multi.agent.llm_registry().has("specialist"));
        assert_eq!(multi.agent.llm_registry().default_alias(), "specialist");
        drop(multi);
        assert_eq!(spawner.spawned_count(), 0);
    }

    #[test]
    fn test_explicit_admission_validation_does_not_reserve_capacity() {
        let spawner = shared_spawner().with_max_agents(1);
        let spec = AgentSpec {
            name: "Child".to_string(),
            system_prompt: "child".to_string(),
            ..AgentSpec::default()
        };

        spawner.validate_explicit_child("child", &spec).unwrap();
        assert_eq!(spawner.spawned_count(), 0);
    }

    #[tokio::test]
    async fn test_admission_rejects_nested_spawner_and_invalid_explicit_id() {
        let nested = r#"
name: NestedChild
system_prompt: nested
spawner:
  management_tools: true
"#;
        let spawner = shared_spawner();
        let error = spawner
            .spawn_from_yaml(nested)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("nested spawner"), "{error}");

        let inert = "name: InertChild\nsystem_prompt: inert\nspawner: {}\n";
        let spawned = spawner.spawn_from_yaml(inert).await.unwrap();
        drop(spawned);

        let spec = AgentSpec {
            name: "Child".to_string(),
            system_prompt: "child".to_string(),
            ..AgentSpec::default()
        };
        let error = spawner
            .spawn_with_id("../child".to_string(), spec)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("Spawned agent ID"), "{error}");
    }

    #[tokio::test]
    async fn test_build_failure_releases_reserved_capacity_once() {
        let spawner = AgentSpawner::new().with_max_agents(1);
        let invalid = r#"
name: InvalidChild
system_prompt: invalid
llm:
  provider: definitely-not-a-provider
  model: unavailable
"#;
        assert!(spawner.spawn_from_yaml(invalid).await.is_err());
        assert_eq!(spawner.spawned_count(), 0);

        let valid = r#"
name: ValidChild
system_prompt: valid
llm:
  provider: ollama
  model: local
"#;
        let spawned = spawner.spawn_from_yaml(valid).await.unwrap();
        assert_eq!(spawner.spawned_count(), 1);
        drop(spawned);
        assert_eq!(spawner.spawned_count(), 0);
    }

    #[tokio::test]
    async fn test_registration_failure_releases_reserved_capacity_once() {
        let spawner = shared_spawner().with_max_agents(2);
        let registry = AgentRegistry::new();
        let spec = AgentSpec {
            name: "Child".to_string(),
            system_prompt: "child".to_string(),
            ..AgentSpec::default()
        };

        let first = spawner
            .spawn_with_id("child".to_string(), spec.clone())
            .await
            .unwrap();
        registry.register(first).await.unwrap();
        assert_eq!(spawner.spawned_count(), 1);

        let duplicate = spawner
            .spawn_with_id("child".to_string(), spec)
            .await
            .unwrap();
        assert_eq!(spawner.spawned_count(), 2);
        assert!(registry.register(duplicate).await.is_err());
        assert_eq!(spawner.spawned_count(), 1);

        assert!(registry.remove("child").await.is_some());
        assert_eq!(spawner.spawned_count(), 0);
        assert!(registry.remove("child").await.is_none());
        assert_eq!(spawner.spawned_count(), 0);
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

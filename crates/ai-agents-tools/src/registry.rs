use parking_lot::RwLock;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use ai_agents_core::{LLMProvider, Tool, ToolInfo, ToolSafetyMetadata};

use super::ToolError;
use super::provider::{ProviderHealth, ToolDescriptor, ToolProvider, ToolProviderError};
use super::types::{
    CommandRunner, CommandRunnerSlot, DiagnosticsProvider, DiagnosticsProviderSlot,
    FileVersionStore, QuestionHandler, QuestionHandlerSlot, TodoItem, TodoStore, ToolAliases,
    UnavailableCommandRunner, UnavailableDiagnosticsProvider, UnavailableWebSearchProvider,
    WebSearchProvider, WebSearchProviderSlot,
};

/// Schema rendering mode for tool prompt generation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolSchemaPromptMode {
    /// Include full JSON schema properties for every granted tool.
    #[default]
    Full,
    /// Include only compact descriptors: name, description, required fields, and property types.
    Compact,
}

/// Canonical identity produced by registry resolution.
#[derive(Debug, Clone)]
pub struct ToolIdentity {
    /// Name, display name, or alias supplied by the caller.
    pub requested_name: String,
    /// Canonical tool ID used for policy and execution.
    pub canonical_id: String,
    /// Display name of the resolved tool.
    pub display_name: String,
    /// Provider ID for provider-backed tools.
    pub provider_id: Option<String>,
}

/// Resolved tool handle plus canonical identity evidence.
#[derive(Clone)]
pub struct ResolvedTool {
    /// Canonical identity returned by lookup.
    pub identity: ToolIdentity,
    /// Executable tool implementation.
    pub tool: Arc<dyn Tool>,
}

#[derive(Clone)]
enum ToolRef {
    Builtin(Arc<dyn Tool>),
    Provider {
        provider_id: String,
        registration_epoch: u64,
        tool: Arc<dyn Tool>,
    },
}

#[derive(Clone)]
struct ProviderEntry {
    provider: Arc<dyn ToolProvider>,
    registration_epoch: u64,
    operation_lock: Arc<tokio::sync::Mutex<()>>,
}

enum ClaimResolution {
    Missing,
    Unique(String),
    Ambiguous,
}

#[derive(Clone)]
struct RegistryState {
    providers: HashMap<String, ProviderEntry>,
    tools: HashMap<String, ToolRef>,
    display_names: HashMap<String, String>,
    provider_aliases: HashMap<String, ToolAliases>,
    host_aliases: HashMap<String, ToolAliases>,
    host_alias_claims: HashMap<String, BTreeSet<String>>,
    canonical_claims: HashMap<String, BTreeSet<String>>,
    display_name_claims: HashMap<String, BTreeSet<String>>,
    alias_claims: HashMap<String, BTreeSet<String>>,
    version: u64,
    next_registration_epoch: u64,
}

impl RegistryState {
    fn new() -> Self {
        Self {
            providers: HashMap::new(),
            tools: HashMap::new(),
            display_names: HashMap::new(),
            provider_aliases: HashMap::new(),
            host_aliases: HashMap::new(),
            host_alias_claims: HashMap::new(),
            canonical_claims: HashMap::new(),
            display_name_claims: HashMap::new(),
            alias_claims: HashMap::new(),
            version: 1,
            next_registration_epoch: 1,
        }
    }

    fn ensure_version_available(&self) {
        self.version
            .checked_add(1)
            .expect("tool registry version exhausted");
    }

    fn bump_version(&mut self) {
        self.version = self.version.checked_add(1).expect("version prechecked");
    }

    fn allocate_registration_epoch(&mut self) -> Result<u64, ToolError> {
        let epoch = self.next_registration_epoch;
        self.next_registration_epoch =
            self.next_registration_epoch.checked_add(1).ok_or_else(|| {
                ToolError::Provider("tool provider registration epoch exhausted".into())
            })?;
        Ok(epoch)
    }
}

#[derive(Clone)]
struct ProviderCandidate {
    descriptor: ToolDescriptor,
    tool: Option<Arc<dyn Tool>>,
}

/// Registry for built-in, provider, alias, and localized tool lookup.
pub struct ToolRegistry {
    state: RwLock<RegistryState>,

    question_handler: QuestionHandlerSlot,

    diagnostics_provider: DiagnosticsProviderSlot,

    command_runner: CommandRunnerSlot,

    todo_store: TodoStore,

    file_versions: FileVersionStore,

    web_fetch_extractor: Arc<RwLock<Option<Arc<dyn LLMProvider>>>>,

    web_search_provider: WebSearchProviderSlot,
}

impl ToolRegistry {
    /// Creates an empty registry with versioned canonical indexes.
    pub fn new() -> Self {
        Self {
            state: RwLock::new(RegistryState::new()),
            question_handler: Arc::new(RwLock::new(None)),
            diagnostics_provider: Arc::new(RwLock::new(Arc::new(UnavailableDiagnosticsProvider))),
            command_runner: Arc::new(RwLock::new(Arc::new(UnavailableCommandRunner))),
            todo_store: TodoStore::default(),
            file_versions: FileVersionStore::default(),
            web_fetch_extractor: Arc::new(RwLock::new(None)),
            web_search_provider: Arc::new(RwLock::new(Arc::new(UnavailableWebSearchProvider))),
        }
    }

    /// Returns the registry version used in tool execution evidence.
    pub fn version(&self) -> u64 {
        self.state.read().version
    }

    fn normalize_key(value: &str) -> String {
        value.trim().to_lowercase()
    }

    fn insert_claim(claims: &mut HashMap<String, BTreeSet<String>>, key: String, tool_id: &str) {
        claims.entry(key).or_default().insert(tool_id.to_string());
    }

    fn resolve_claim(claims: &HashMap<String, BTreeSet<String>>, key: &str) -> ClaimResolution {
        let Some(owners) = claims.get(key) else {
            return ClaimResolution::Missing;
        };
        if owners.len() == 1 {
            ClaimResolution::Unique(owners.first().expect("one claim owner").clone())
        } else {
            ClaimResolution::Ambiguous
        }
    }

    fn add_alias_claims(
        claims: &mut HashMap<String, BTreeSet<String>>,
        tool_id: &str,
        aliases: &ToolAliases,
    ) {
        for (language, name) in &aliases.names {
            let normalized = Self::normalize_key(name);
            Self::insert_claim(claims, format!("{language}:{normalized}"), tool_id);
            Self::insert_claim(claims, normalized, tool_id);
        }
    }

    fn rebuild_claims(state: &mut RegistryState) {
        let mut canonical_claims = HashMap::new();
        let mut display_name_claims = HashMap::new();
        let mut alias_claims = HashMap::new();

        for (tool_id, tool_ref) in &state.tools {
            Self::insert_claim(&mut canonical_claims, Self::normalize_key(tool_id), tool_id);
            let display_name = state
                .display_names
                .get(tool_id)
                .map(String::as_str)
                .unwrap_or_else(|| match tool_ref {
                    ToolRef::Builtin(tool) => tool.name(),
                    ToolRef::Provider { tool, .. } => tool.name(),
                });
            Self::insert_claim(
                &mut display_name_claims,
                Self::normalize_key(display_name),
                tool_id,
            );
            if let Some(aliases) = state.provider_aliases.get(tool_id) {
                Self::add_alias_claims(&mut alias_claims, tool_id, aliases);
            }
            if let Some(keys) = state.host_alias_claims.get(tool_id) {
                for key in keys {
                    Self::insert_claim(&mut alias_claims, key.clone(), tool_id);
                }
            }
        }

        state.canonical_claims = canonical_claims;
        state.display_name_claims = display_name_claims;
        state.alias_claims = alias_claims;
    }

    fn tool_from_ref(tool_ref: &ToolRef) -> Arc<dyn Tool> {
        match tool_ref {
            ToolRef::Builtin(tool) | ToolRef::Provider { tool, .. } => tool.clone(),
        }
    }

    fn validate_descriptor_ids(descriptors: &[ToolDescriptor]) -> Result<(), String> {
        let mut seen = HashSet::with_capacity(descriptors.len());
        for descriptor in descriptors {
            if !seen.insert(descriptor.id.clone()) {
                return Err(descriptor.id.clone());
            }
        }
        Ok(())
    }

    async fn collect_provider_candidates(
        provider: &Arc<dyn ToolProvider>,
        descriptors: Vec<ToolDescriptor>,
    ) -> Vec<ProviderCandidate> {
        let mut candidates = Vec::with_capacity(descriptors.len());
        for descriptor in descriptors {
            let tool = provider.get_tool(&descriptor.id).await;
            candidates.push(ProviderCandidate { descriptor, tool });
        }
        candidates
    }

    /// Registers a built-in or custom tool by canonical ID.
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Result<(), ToolError> {
        let id = tool.id().to_string();
        let mut state = self.state.write();
        if state.tools.contains_key(&id) {
            return Err(ToolError::Duplicate(id));
        }
        state.ensure_version_available();
        state
            .display_names
            .insert(id.clone(), tool.name().to_string());
        state.tools.insert(id, ToolRef::Builtin(tool));
        Self::rebuild_claims(&mut state);
        state.bump_version();
        Ok(())
    }

    pub fn get(&self, id_or_alias: &str) -> Option<Arc<dyn Tool>> {
        self.resolve(id_or_alias).map(|resolved| resolved.tool)
    }

    /// Resolves any accepted name to a canonical tool ID.
    pub fn canonical_id(&self, id_or_alias: &str) -> Option<String> {
        self.resolve(id_or_alias)
            .map(|resolved| resolved.identity.canonical_id)
    }

    /// Resolves safety metadata for a registered tool.
    pub fn safety_metadata(&self, id_or_alias: &str) -> Option<ToolSafetyMetadata> {
        self.resolve(id_or_alias)
            .map(|resolved| resolved.tool.safety_metadata())
    }

    /// Returns the shared question handler slot for host-bound tools.
    pub fn question_handler_slot(&self) -> QuestionHandlerSlot {
        Arc::clone(&self.question_handler)
    }

    /// Installs or clears the question handler used by `ask_user`.
    pub fn set_question_handler(&self, handler: Option<Arc<dyn QuestionHandler>>) {
        *self.question_handler.write() = handler;
    }

    /// Returns the shared diagnostics provider slot for host-bound tools.
    pub fn diagnostics_provider_slot(&self) -> DiagnosticsProviderSlot {
        Arc::clone(&self.diagnostics_provider)
    }

    /// Installs the diagnostics provider used by `diagnostics`.
    pub fn set_diagnostics_provider(&self, provider: Arc<dyn DiagnosticsProvider>) {
        *self.diagnostics_provider.write() = provider;
    }

    /// Returns whether the diagnostics provider can serve requests now.
    pub fn diagnostics_available(&self) -> bool {
        self.diagnostics_provider.read().is_available()
    }

    /// Returns the shared command runner slot for host-bound tools.
    pub fn command_runner_slot(&self) -> CommandRunnerSlot {
        Arc::clone(&self.command_runner)
    }

    /// Installs the command runner used by `command`.
    pub fn set_command_runner(&self, runner: Arc<dyn CommandRunner>) {
        *self.command_runner.write() = runner;
    }

    /// Returns whether the command runner can serve requests now.
    pub fn command_runner_available(&self) -> bool {
        self.command_runner.read().is_available()
    }

    /// Returns the session-local file version store shared with file tools.
    pub fn file_version_store(&self) -> FileVersionStore {
        self.file_versions.clone()
    }

    /// Returns the session-local todo store shared with `todo`.
    pub fn todo_store(&self) -> TodoStore {
        self.todo_store.clone()
    }

    /// Returns a snapshot of session-local todo items.
    pub fn todos(&self) -> Vec<TodoItem> {
        self.todo_store.list()
    }

    /// Returns the shared web-fetch extractor slot.
    pub fn web_fetch_extractor_slot(&self) -> Arc<RwLock<Option<Arc<dyn LLMProvider>>>> {
        Arc::clone(&self.web_fetch_extractor)
    }

    /// Installs or clears the LLM used for `web_fetch` prompt extraction.
    pub fn set_web_fetch_extractor(&self, extractor: Option<Arc<dyn LLMProvider>>) {
        *self.web_fetch_extractor.write() = extractor;
    }

    /// Returns the shared provider slot for `web_search`.
    pub fn web_search_provider_slot(&self) -> WebSearchProviderSlot {
        Arc::clone(&self.web_search_provider)
    }

    /// Installs the provider used by `web_search`.
    pub fn set_web_search_provider(&self, provider: Arc<dyn WebSearchProvider>) {
        *self.web_search_provider.write() = provider;
    }

    /// Returns whether the web search provider can serve requests now.
    pub fn web_search_available(&self) -> bool {
        self.web_search_provider.read().is_available()
    }

    /// Resolves exact canonical IDs first and only accepts unique normalized names or aliases.
    pub fn resolve(&self, id_or_alias: &str) -> Option<ResolvedTool> {
        let state = self.state.read();
        let requested_name = id_or_alias.to_string();
        if let Some(tool_ref) = state.tools.get(id_or_alias) {
            return Some(Self::resolved_tool_from_ref(
                &requested_name,
                id_or_alias,
                tool_ref,
            ));
        }

        let normalized = Self::normalize_key(id_or_alias);
        for claims in [
            &state.canonical_claims,
            &state.display_name_claims,
            &state.alias_claims,
        ] {
            match Self::resolve_claim(claims, &normalized) {
                ClaimResolution::Missing => continue,
                ClaimResolution::Ambiguous => return None,
                ClaimResolution::Unique(tool_id) => {
                    let tool_ref = state.tools.get(&tool_id)?;
                    return Some(Self::resolved_tool_from_ref(
                        &requested_name,
                        &tool_id,
                        tool_ref,
                    ));
                }
            }
        }
        None
    }

    fn resolved_tool_from_ref(
        requested_name: &str,
        canonical_id: &str,
        tool_ref: &ToolRef,
    ) -> ResolvedTool {
        let tool = Self::tool_from_ref(tool_ref);
        let provider_id = match tool_ref {
            ToolRef::Builtin(_) => None,
            ToolRef::Provider { provider_id, .. } => Some(provider_id.clone()),
        };
        ResolvedTool {
            identity: ToolIdentity {
                requested_name: requested_name.to_string(),
                canonical_id: canonical_id.to_string(),
                display_name: tool.name().to_string(),
                provider_id,
            },
            tool,
        }
    }

    pub fn list_ids(&self) -> Vec<String> {
        self.state.read().tools.keys().cloned().collect()
    }

    pub fn list_infos(&self) -> Vec<ToolInfo> {
        self.state
            .read()
            .tools
            .values()
            .map(Self::tool_from_ref)
            .map(|tool| tool.info())
            .collect()
    }

    pub fn len(&self) -> usize {
        self.state.read().tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.state.read().tools.is_empty()
    }

    /// Maps one consistent registry snapshot without invoking user callbacks under registry locks.
    pub fn map_tools<F>(&self, mut f: F) -> ToolRegistry
    where
        F: FnMut(Arc<dyn Tool>) -> Arc<dyn Tool>,
    {
        let snapshot = self.state.read().clone();
        let mut mapped_state = snapshot.clone();
        mapped_state.tools.clear();
        for (id, tool_ref) in snapshot.tools {
            let mapped_ref = match tool_ref {
                ToolRef::Builtin(tool) => ToolRef::Builtin(f(tool)),
                ToolRef::Provider {
                    provider_id,
                    registration_epoch,
                    tool,
                } => ToolRef::Provider {
                    provider_id,
                    registration_epoch,
                    tool: f(tool),
                },
            };
            mapped_state.tools.insert(id, mapped_ref);
        }
        Self::rebuild_claims(&mut mapped_state);

        let mut mapped = ToolRegistry::new();
        *mapped.state.write() = mapped_state;
        mapped.question_handler = Arc::clone(&self.question_handler);
        mapped.diagnostics_provider = Arc::clone(&self.diagnostics_provider);
        mapped.command_runner = Arc::clone(&self.command_runner);
        mapped.todo_store = self.todo_store.clone();
        mapped.file_versions = self.file_versions.clone();
        mapped.web_fetch_extractor = Arc::clone(&self.web_fetch_extractor);
        mapped.web_search_provider = Arc::clone(&self.web_search_provider);
        mapped
    }

    /// Registers one provider snapshot only after all descriptor and ownership checks pass.
    pub async fn register_provider(
        &self,
        provider: Arc<dyn ToolProvider>,
    ) -> Result<(), ToolError> {
        let provider_id = provider.id().to_string();
        if self.state.read().providers.contains_key(&provider_id) {
            return Err(ToolError::Duplicate(format!("Provider: {provider_id}")));
        }

        let descriptors = provider.list_tools().await;
        Self::validate_descriptor_ids(&descriptors).map_err(ToolError::Duplicate)?;
        {
            let state = self.state.read();
            if state.providers.contains_key(&provider_id) {
                return Err(ToolError::Duplicate(format!("Provider: {provider_id}")));
            }
            for descriptor in &descriptors {
                if state.tools.contains_key(&descriptor.id) {
                    return Err(ToolError::Duplicate(descriptor.id.clone()));
                }
            }
        }
        let candidates = Self::collect_provider_candidates(&provider, descriptors).await;

        let mut state = self.state.write();
        if state.providers.contains_key(&provider_id) {
            return Err(ToolError::Duplicate(format!("Provider: {provider_id}")));
        }
        for candidate in &candidates {
            if state.tools.contains_key(&candidate.descriptor.id) {
                return Err(ToolError::Duplicate(candidate.descriptor.id.clone()));
            }
        }

        state.ensure_version_available();
        let registration_epoch = state.allocate_registration_epoch()?;
        state.providers.insert(
            provider_id.clone(),
            ProviderEntry {
                provider,
                registration_epoch,
                operation_lock: Arc::new(tokio::sync::Mutex::new(())),
            },
        );
        for candidate in candidates {
            let Some(tool) = candidate.tool else {
                continue;
            };
            let id = candidate.descriptor.id;
            state
                .display_names
                .insert(id.clone(), candidate.descriptor.name);
            if let Some(aliases) = candidate.descriptor.aliases {
                state.provider_aliases.insert(id.clone(), aliases);
            }
            state.tools.insert(
                id,
                ToolRef::Provider {
                    provider_id: provider_id.clone(),
                    registration_epoch,
                    tool,
                },
            );
        }
        Self::rebuild_claims(&mut state);
        state.bump_version();
        Ok(())
    }

    /// Removes one provider registration and every tool and alias claim owned by its epoch.
    pub fn unregister_provider(&self, provider_id: &str) -> bool {
        let mut state = self.state.write();
        let Some(entry) = state.providers.get(provider_id).cloned() else {
            return false;
        };
        state.ensure_version_available();
        state.providers.remove(provider_id);
        let tools_to_remove: Vec<String> = state
            .tools
            .iter()
            .filter_map(|(id, tool_ref)| match tool_ref {
                ToolRef::Provider {
                    provider_id: owner,
                    registration_epoch,
                    ..
                } if owner == provider_id && *registration_epoch == entry.registration_epoch => {
                    Some(id.clone())
                }
                _ => None,
            })
            .collect();
        for tool_id in tools_to_remove {
            state.tools.remove(&tool_id);
            state.display_names.remove(&tool_id);
            state.provider_aliases.remove(&tool_id);
            state.host_aliases.remove(&tool_id);
            state.host_alias_claims.remove(&tool_id);
        }
        Self::rebuild_claims(&mut state);
        state.bump_version();
        true
    }

    /// Replaces prompt metadata while retaining lookup aliases added during this registration.
    pub fn set_tool_aliases(&self, tool_id: &str, aliases: ToolAliases) {
        let mut state = self.state.write();
        if !state.tools.contains_key(tool_id) {
            return;
        }
        state.ensure_version_available();
        let claims = state
            .host_alias_claims
            .entry(tool_id.to_string())
            .or_default();
        for (language, name) in &aliases.names {
            let normalized = Self::normalize_key(name);
            claims.insert(format!("{language}:{normalized}"));
            claims.insert(normalized);
        }
        state.host_aliases.insert(tool_id.to_string(), aliases);
        Self::rebuild_claims(&mut state);
        state.bump_version();
    }

    pub fn get_by_alias(&self, alias: &str, lang: &str) -> Option<Arc<dyn Tool>> {
        let state = self.state.read();
        let key = format!("{}:{}", lang, Self::normalize_key(alias));
        let ClaimResolution::Unique(tool_id) = Self::resolve_claim(&state.alias_claims, &key)
        else {
            return None;
        };
        state.tools.get(&tool_id).map(Self::tool_from_ref)
    }

    pub fn list_providers(&self) -> Vec<String> {
        self.state.read().providers.keys().cloned().collect()
    }

    pub async fn provider_health(&self, provider_id: &str) -> Option<ProviderHealth> {
        let provider = self
            .state
            .read()
            .providers
            .get(provider_id)
            .map(|entry| entry.provider.clone());
        if let Some(provider) = provider {
            Some(provider.health_check().await)
        } else {
            None
        }
    }

    /// Serializes refresh work per registration and publishes only to the captured epoch.
    pub async fn refresh_provider(&self, provider_id: &str) -> Result<(), ToolProviderError> {
        let entry = self
            .state
            .read()
            .providers
            .get(provider_id)
            .cloned()
            .ok_or_else(|| {
                ToolProviderError::ToolNotFound(format!("Provider not found: {provider_id}"))
            })?;
        let _operation = entry.operation_lock.lock().await;

        {
            let state = self.state.read();
            let current = state.providers.get(provider_id).ok_or_else(|| {
                ToolProviderError::ToolNotFound(format!("Provider not found: {provider_id}"))
            })?;
            if current.registration_epoch != entry.registration_epoch {
                return Err(ToolProviderError::ToolNotFound(format!(
                    "Provider registration replaced: {provider_id}"
                )));
            }
        }

        if !entry.provider.supports_refresh() {
            return Ok(());
        }
        entry.provider.refresh().await?;
        let descriptors = entry.provider.list_tools().await;
        Self::validate_descriptor_ids(&descriptors).map_err(|id| {
            ToolProviderError::ConfigError(format!("Duplicate tool descriptor: {id}"))
        })?;
        {
            let state = self.state.read();
            let current = state.providers.get(provider_id).ok_or_else(|| {
                ToolProviderError::ToolNotFound(format!("Provider not found: {provider_id}"))
            })?;
            if current.registration_epoch != entry.registration_epoch {
                return Err(ToolProviderError::ToolNotFound(format!(
                    "Provider registration replaced: {provider_id}"
                )));
            }
            for descriptor in &descriptors {
                if let Some(existing) = state.tools.get(&descriptor.id) {
                    match existing {
                        ToolRef::Provider {
                            provider_id: owner,
                            registration_epoch,
                            ..
                        } if owner == provider_id
                            && *registration_epoch == entry.registration_epoch => {}
                        _ => {
                            return Err(ToolProviderError::ConfigError(format!(
                                "Tool ID is owned by another registration: {}",
                                descriptor.id
                            )));
                        }
                    }
                }
            }
        }
        let candidates = Self::collect_provider_candidates(&entry.provider, descriptors).await;

        let mut state = self.state.write();
        let current = state.providers.get(provider_id).ok_or_else(|| {
            ToolProviderError::ToolNotFound(format!("Provider not found: {provider_id}"))
        })?;
        if current.registration_epoch != entry.registration_epoch {
            return Err(ToolProviderError::ToolNotFound(format!(
                "Provider registration replaced: {provider_id}"
            )));
        }
        for candidate in &candidates {
            if let Some(existing) = state.tools.get(&candidate.descriptor.id) {
                match existing {
                    ToolRef::Provider {
                        provider_id: owner,
                        registration_epoch,
                        ..
                    } if owner == provider_id
                        && *registration_epoch == entry.registration_epoch => {}
                    _ => {
                        return Err(ToolProviderError::ConfigError(format!(
                            "Tool ID is owned by another registration: {}",
                            candidate.descriptor.id
                        )));
                    }
                }
            }
        }

        state.ensure_version_available();
        let retained_tool_ids: HashSet<String> = candidates
            .iter()
            .filter(|candidate| candidate.tool.is_some())
            .map(|candidate| candidate.descriptor.id.clone())
            .collect();
        let old_tools: Vec<String> = state
            .tools
            .iter()
            .filter_map(|(id, tool_ref)| match tool_ref {
                ToolRef::Provider {
                    provider_id: owner,
                    registration_epoch,
                    ..
                } if owner == provider_id && *registration_epoch == entry.registration_epoch => {
                    Some(id.clone())
                }
                _ => None,
            })
            .collect();
        for tool_id in old_tools {
            state.tools.remove(&tool_id);
            state.display_names.remove(&tool_id);
            state.provider_aliases.remove(&tool_id);
            if !retained_tool_ids.contains(&tool_id) {
                state.host_aliases.remove(&tool_id);
                state.host_alias_claims.remove(&tool_id);
            }
        }
        for candidate in candidates {
            let Some(tool) = candidate.tool else {
                continue;
            };
            let id = candidate.descriptor.id;
            state
                .display_names
                .insert(id.clone(), candidate.descriptor.name);
            if let Some(aliases) = candidate.descriptor.aliases {
                state.provider_aliases.insert(id.clone(), aliases);
            }
            state.tools.insert(
                id,
                ToolRef::Provider {
                    provider_id: provider_id.to_string(),
                    registration_epoch: entry.registration_epoch,
                    tool,
                },
            );
        }
        Self::rebuild_claims(&mut state);
        state.bump_version();
        Ok(())
    }

    pub fn generate_tools_prompt(&self) -> String {
        self.generate_tools_prompt_with_lang(None, false)
    }

    pub fn generate_tools_prompt_with_parallel(&self, parallel: bool) -> String {
        self.generate_tools_prompt_with_lang(None, parallel)
    }

    pub fn generate_tools_prompt_with_lang(
        &self,
        language: Option<&str>,
        parallel: bool,
    ) -> String {
        let state = self.state.read();
        if state.tools.is_empty() {
            return String::new();
        }

        let mut prompt = String::from("Available tools:\n");

        for (id, tool_ref) in &state.tools {
            let tool = Self::tool_from_ref(tool_ref);
            {
                let (name, description) = if let Some(lang) = language {
                    if let Some(aliases) = state.host_aliases.get(id) {
                        let name = aliases
                            .names
                            .get(lang)
                            .map(|s| s.as_str())
                            .unwrap_or_else(|| tool.name());
                        let desc = aliases
                            .descriptions
                            .get(lang)
                            .map(|s| s.as_str())
                            .unwrap_or_else(|| tool.description());
                        (name, desc)
                    } else {
                        (tool.name(), tool.description())
                    }
                } else {
                    (tool.name(), tool.description())
                };

                let schema = tool.input_schema();
                let args_desc = if let Some(props) = schema.get("properties") {
                    serde_json::to_string(props).unwrap_or_default()
                } else {
                    "{}".to_string()
                };

                prompt.push_str(&format!(
                    "- {}: {}. Arguments: {}\n",
                    name, description, args_desc
                ));
            }
        }

        Self::append_tool_format_instructions(&mut prompt, parallel);

        prompt
    }

    pub fn generate_filtered_prompt(&self, tool_ids: &[String]) -> String {
        self.generate_filtered_prompt_with_lang(tool_ids, None, false)
    }

    pub fn generate_filtered_prompt_with_parallel(
        &self,
        tool_ids: &[String],
        parallel: bool,
    ) -> String {
        self.generate_filtered_prompt_with_lang(tool_ids, None, parallel)
    }

    /// Generates a prompt for an explicit tool scope.
    pub fn generate_scoped_prompt_with_parallel(
        &self,
        tool_ids: &[String],
        parallel: bool,
    ) -> String {
        self.generate_scoped_prompt_with_lang(tool_ids, None, parallel)
    }

    pub fn generate_scoped_prompt_with_lang(
        &self,
        tool_ids: &[String],
        language: Option<&str>,
        parallel: bool,
    ) -> String {
        if tool_ids.is_empty() {
            return String::new();
        }
        self.generate_filtered_prompt_inner(tool_ids, language, parallel)
    }

    pub fn generate_filtered_prompt_with_lang(
        &self,
        tool_ids: &[String],
        language: Option<&str>,
        parallel: bool,
    ) -> String {
        if tool_ids.is_empty() {
            return self.generate_tools_prompt_with_lang(language, parallel);
        }

        self.generate_filtered_prompt_inner(tool_ids, language, parallel)
    }

    fn generate_filtered_prompt_inner(
        &self,
        tool_ids: &[String],
        language: Option<&str>,
        parallel: bool,
    ) -> String {
        let state = self.state.read();
        let mut prompt = String::from("Available tools:\n");
        let mut found_any = false;

        for id in tool_ids {
            if let Some(tool_ref) = state.tools.get(id) {
                let tool = Self::tool_from_ref(tool_ref);
                found_any = true;

                let (name, description) = if let Some(lang) = language {
                    if let Some(aliases) = state.host_aliases.get(id) {
                        let name = aliases
                            .names
                            .get(lang)
                            .map(|s| s.as_str())
                            .unwrap_or_else(|| tool.name());
                        let desc = aliases
                            .descriptions
                            .get(lang)
                            .map(|s| s.as_str())
                            .unwrap_or_else(|| tool.description());
                        (name, desc)
                    } else {
                        (tool.name(), tool.description())
                    }
                } else {
                    (tool.name(), tool.description())
                };

                let schema = tool.input_schema();
                let args_desc = if let Some(props) = schema.get("properties") {
                    serde_json::to_string(props).unwrap_or_default()
                } else {
                    "{}".to_string()
                };

                prompt.push_str(&format!(
                    "- {}: {}. Arguments: {}\n",
                    name, description, args_desc
                ));
            }
        }

        if !found_any {
            return String::new();
        }

        Self::append_tool_format_instructions(&mut prompt, parallel);

        prompt
    }

    /// Generate a scoped prompt with a configurable schema rendering mode.
    pub fn generate_scoped_prompt_with_mode(
        &self,
        tool_ids: &[impl AsRef<str>],
        language: Option<&str>,
        parallel: bool,
        mode: ToolSchemaPromptMode,
    ) -> String {
        if tool_ids.is_empty() {
            return String::new();
        }
        match mode {
            ToolSchemaPromptMode::Full => self.generate_scoped_prompt_with_lang(
                &tool_ids
                    .iter()
                    .map(|s| s.as_ref().to_string())
                    .collect::<Vec<_>>(),
                language,
                parallel,
            ),
            ToolSchemaPromptMode::Compact => {
                self.generate_compact_prompt_inner(tool_ids, language, parallel)
            }
        }
    }

    /// Generate a compact tool prompt with stable ordering and reduced schema.
    fn generate_compact_prompt_inner(
        &self,
        tool_ids: &[impl AsRef<str>],
        language: Option<&str>,
        parallel: bool,
    ) -> String {
        let state = self.state.read();
        let mut prompt = String::from("Available tools:\n");
        let mut found_any = false;

        for id in tool_ids {
            let id = id.as_ref();
            if let Some(tool_ref) = state.tools.get(id) {
                let tool = Self::tool_from_ref(tool_ref);
                found_any = true;

                let (name, description) = if let Some(lang) = language {
                    if let Some(aliases) = state.host_aliases.get(id) {
                        let n = aliases
                            .names
                            .get(lang)
                            .map(|s| s.as_str())
                            .unwrap_or_else(|| tool.name());
                        let d = aliases
                            .descriptions
                            .get(lang)
                            .map(|s| s.as_str())
                            .unwrap_or_else(|| tool.description());
                        (n, d)
                    } else {
                        (tool.name(), tool.description())
                    }
                } else {
                    (tool.name(), tool.description())
                };

                let schema = tool.input_schema();
                let compact = compact_schema_descriptor(&schema);
                prompt.push_str(&format!("- {}: {}. {}\n", name, description, compact));
            }
        }

        if !found_any {
            return String::new();
        }

        Self::append_tool_format_instructions(&mut prompt, parallel);
        prompt
    }

    /// Append tool call format instructions to a prompt.
    /// When `parallel` is true, also instructs the LLM to use a JSON array
    /// for multiple simultaneous tool calls.
    fn append_tool_format_instructions(prompt: &mut String, parallel: bool) {
        prompt.push_str(
            "\nWhen you need to use a tool, respond ONLY with valid JSON in this exact format:\n",
        );
        prompt.push_str("{\"tool\": \"tool_name\", \"arguments\": {...}}\n");
        prompt.push_str("The \"tool\" value MUST be one of the exact tool names listed above. Do not invent tool names.\n");
        if parallel {
            prompt.push_str(
                "\nWhen you need to call multiple tools at once, respond with a JSON array:\n",
            );
            prompt.push_str(
                "[{\"tool\": \"tool_name1\", \"arguments\": {...}}, {\"tool\": \"tool_name2\", \"arguments\": {...}}]\n",
            );
        }
        prompt.push_str("\nWhen you receive a tool result, summarize it naturally for the user.\n");
        prompt.push_str("If no tool is needed, respond normally.");
    }
}

/// Build a compact schema descriptor from a full JSON schema.
/// Includes required fields and property types only, with stable key ordering.
fn compact_schema_descriptor(schema: &serde_json::Value) -> String {
    let props = schema.get("properties").and_then(|p| p.as_object());
    let required = schema.get("required").and_then(|r| r.as_array());
    let mut parts = Vec::new();

    if let Some(req) = required {
        let req_fields: Vec<String> = req
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        if !req_fields.is_empty() {
            parts.push(format!("required: [{}]", req_fields.join(", ")));
        }
    }

    if let Some(props) = props {
        let mut prop_parts = Vec::new();
        for (key, value) in props.iter() {
            let prop_type = value.get("type").and_then(|t| t.as_str()).unwrap_or("?");
            let prop_desc = value
                .get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("");
            let short_desc = if prop_desc.len() > 40 {
                format!("{}...", &prop_desc[..40])
            } else if !prop_desc.is_empty() {
                prop_desc.to_string()
            } else {
                String::new()
            };
            if short_desc.is_empty() {
                prop_parts.push(format!("{}({})", key, prop_type));
            } else {
                prop_parts.push(format!("{}({}): {}", key, prop_type, short_desc));
            }
        }
        if !prop_parts.is_empty() {
            parts.push(format!("args: {}", prop_parts.join("; ")));
        }
    }

    if parts.is_empty() {
        "Args: none".to_string()
    } else {
        parts.join(". ")
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToolProviderType, ToolResult};
    use async_trait::async_trait;
    use serde_json::Value;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::{Barrier, Notify, Semaphore};

    struct TestTool {
        id: String,
    }

    #[async_trait]
    impl Tool for TestTool {
        fn id(&self) -> &str {
            &self.id
        }
        fn name(&self) -> &str {
            "Test"
        }
        fn description(&self) -> &str {
            "A test tool"
        }
        fn input_schema(&self) -> Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(
            &self,
            _args: Value,
            _ctx: ai_agents_core::ToolExecutionContext,
        ) -> ToolResult {
            ToolResult::ok("test")
        }
    }

    struct RenamedTool {
        id: String,
    }

    #[async_trait]
    impl Tool for RenamedTool {
        fn id(&self) -> &str {
            &self.id
        }

        fn name(&self) -> &str {
            "Wrapped"
        }

        fn description(&self) -> &str {
            "A wrapped test tool"
        }

        fn input_schema(&self) -> Value {
            serde_json::json!({"type": "object"})
        }

        async fn execute(
            &self,
            _args: Value,
            _ctx: ai_agents_core::ToolExecutionContext,
        ) -> ToolResult {
            ToolResult::ok("wrapped")
        }
    }

    struct TestProvider {
        id: String,
        descriptors: RwLock<Vec<ToolDescriptor>>,
        available: RwLock<HashSet<String>>,
        get_calls: AtomicUsize,
        refreshable: bool,
        list_gate: Option<Arc<Barrier>>,
    }

    impl TestProvider {
        fn new(
            id: impl Into<String>,
            descriptors: Vec<ToolDescriptor>,
            available: impl IntoIterator<Item = &'static str>,
        ) -> Self {
            Self {
                id: id.into(),
                descriptors: RwLock::new(descriptors),
                available: RwLock::new(available.into_iter().map(str::to_string).collect()),
                get_calls: AtomicUsize::new(0),
                refreshable: true,
                list_gate: None,
            }
        }

        fn with_list_gate(mut self, gate: Arc<Barrier>) -> Self {
            self.list_gate = Some(gate);
            self
        }

        fn get_call_count(&self) -> usize {
            self.get_calls.load(Ordering::SeqCst)
        }

        fn set_snapshot(
            &self,
            descriptors: Vec<ToolDescriptor>,
            available: impl IntoIterator<Item = &'static str>,
        ) {
            *self.descriptors.write() = descriptors;
            *self.available.write() = available.into_iter().map(str::to_string).collect();
        }
    }

    #[async_trait]
    impl ToolProvider for TestProvider {
        fn id(&self) -> &str {
            &self.id
        }

        fn name(&self) -> &str {
            &self.id
        }

        fn provider_type(&self) -> ToolProviderType {
            ToolProviderType::Custom
        }

        async fn list_tools(&self) -> Vec<ToolDescriptor> {
            if let Some(gate) = &self.list_gate {
                gate.wait().await;
            }
            self.descriptors.read().clone()
        }

        async fn get_tool(&self, tool_id: &str) -> Option<Arc<dyn Tool>> {
            self.get_calls.fetch_add(1, Ordering::SeqCst);
            self.available.read().contains(tool_id).then(|| {
                Arc::new(TestTool {
                    id: tool_id.to_string(),
                }) as Arc<dyn Tool>
            })
        }

        fn supports_refresh(&self) -> bool {
            self.refreshable
        }
    }

    struct ActiveRefresh<'a>(&'a AtomicUsize);

    impl Drop for ActiveRefresh<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    struct BlockingRefreshProvider {
        id: String,
        tool_id: String,
        started: Arc<Notify>,
        permits: Arc<Semaphore>,
        refresh_calls: AtomicUsize,
        active_refreshes: AtomicUsize,
        max_active_refreshes: AtomicUsize,
        failures_remaining: AtomicUsize,
    }

    #[async_trait]
    impl ToolProvider for BlockingRefreshProvider {
        fn id(&self) -> &str {
            &self.id
        }

        fn name(&self) -> &str {
            &self.id
        }

        fn provider_type(&self) -> ToolProviderType {
            ToolProviderType::Custom
        }

        async fn list_tools(&self) -> Vec<ToolDescriptor> {
            vec![ToolDescriptor::new(
                &self.tool_id,
                "Serial Tool",
                "Tests refresh serialization",
                serde_json::json!({"type": "object"}),
            )]
        }

        async fn get_tool(&self, tool_id: &str) -> Option<Arc<dyn Tool>> {
            (tool_id == self.tool_id).then(|| {
                Arc::new(TestTool {
                    id: tool_id.to_string(),
                }) as Arc<dyn Tool>
            })
        }

        fn supports_refresh(&self) -> bool {
            true
        }

        async fn refresh(&self) -> Result<(), ToolProviderError> {
            self.refresh_calls.fetch_add(1, Ordering::SeqCst);
            let active = self.active_refreshes.fetch_add(1, Ordering::SeqCst) + 1;
            let _active = ActiveRefresh(&self.active_refreshes);
            self.max_active_refreshes
                .fetch_max(active, Ordering::SeqCst);
            self.started.notify_one();
            let permit = self.permits.acquire().await.expect("refresh permit");
            permit.forget();
            if self
                .failures_remaining
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                    remaining.checked_sub(1)
                })
                .is_ok()
            {
                return Err(ToolProviderError::ConfigError(
                    "simulated refresh failure".to_string(),
                ));
            }
            Ok(())
        }
    }

    fn descriptor(id: &str) -> ToolDescriptor {
        ToolDescriptor::new(
            id,
            format!("{id} name"),
            format!("{id} description"),
            serde_json::json!({"type": "object"}),
        )
    }

    #[test]
    fn test_register_and_get() {
        let mut registry = ToolRegistry::new();
        let tool = Arc::new(TestTool {
            id: "test".to_string(),
        });

        registry.register(tool).unwrap();
        assert!(registry.get("test").is_some());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_duplicate_registration() {
        let mut registry = ToolRegistry::new();
        let tool1 = Arc::new(TestTool {
            id: "test".to_string(),
        });
        let tool2 = Arc::new(TestTool {
            id: "test".to_string(),
        });

        registry.register(tool1).unwrap();
        assert!(registry.register(tool2).is_err());
    }

    #[test]
    fn test_list_ids() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool {
                id: "a".to_string(),
            }))
            .unwrap();
        registry
            .register(Arc::new(TestTool {
                id: "b".to_string(),
            }))
            .unwrap();

        let ids = registry.list_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"a".to_string()));
        assert!(ids.contains(&"b".to_string()));
    }

    #[test]
    fn test_generate_tools_prompt() {
        let empty_registry = ToolRegistry::new();
        let empty_prompt = empty_registry.generate_tools_prompt();
        assert!(empty_prompt.is_empty());

        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool {
                id: "test".to_string(),
            }))
            .unwrap();

        let prompt = registry.generate_tools_prompt();
        assert!(prompt.contains("Available tools:"));
        assert!(prompt.contains("Test:"));
        assert!(prompt.contains("A test tool"));
        assert!(prompt.contains("tool_name"));
    }

    #[test]
    fn test_generate_filtered_prompt_with_filter() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool {
                id: "tool_a".to_string(),
            }))
            .unwrap();
        registry
            .register(Arc::new(TestTool {
                id: "tool_b".to_string(),
            }))
            .unwrap();
        registry
            .register(Arc::new(TestTool {
                id: "tool_c".to_string(),
            }))
            .unwrap();

        let prompt =
            registry.generate_filtered_prompt(&["tool_a".to_string(), "tool_c".to_string()]);

        assert!(prompt.contains("tool_a") || prompt.contains("Test"));
        assert!(!prompt.contains("tool_b"));
    }

    #[test]
    fn test_generate_filtered_prompt_empty_filter() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool {
                id: "tool_a".to_string(),
            }))
            .unwrap();
        registry
            .register(Arc::new(TestTool {
                id: "tool_b".to_string(),
            }))
            .unwrap();

        let prompt = registry.generate_filtered_prompt(&[]);
        assert!(prompt.contains("Test"));
    }

    #[test]
    fn test_generate_filtered_prompt_nonexistent_tools() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool {
                id: "tool_a".to_string(),
            }))
            .unwrap();

        let prompt = registry.generate_filtered_prompt(&["nonexistent".to_string()]);
        assert!(prompt.is_empty());

        let prompt2 =
            registry.generate_filtered_prompt(&["tool_a".to_string(), "nonexistent".to_string()]);
        assert!(prompt2.contains("Test"));
    }

    #[tokio::test]
    async fn provider_registration_is_atomic_on_duplicate() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool { id: "b".into() }))
            .unwrap();
        let original_b = registry.get("b").unwrap();
        let version = registry.version();
        let provider = Arc::new(TestProvider::new(
            "duplicate_provider",
            vec![descriptor("a"), descriptor("b")],
            ["a", "b"],
        ));

        assert!(registry.register_provider(provider.clone()).await.is_err());
        assert_eq!(provider.get_call_count(), 0);
        assert!(registry.get("a").is_none());
        assert!(Arc::ptr_eq(&original_b, &registry.get("b").unwrap()));
        assert!(
            !registry
                .list_providers()
                .contains(&"duplicate_provider".to_string())
        );
        assert_eq!(registry.version(), version);

        provider.set_snapshot(vec![descriptor("a")], ["a"]);
        registry.register_provider(provider).await.unwrap();
        assert!(registry.get("a").is_some());
    }

    #[tokio::test]
    async fn provider_snapshot_rejects_raw_duplicates_even_when_missing() {
        let registry = ToolRegistry::new();
        let version = registry.version();
        let provider = Arc::new(TestProvider::new(
            "duplicate_none",
            vec![descriptor("a"), descriptor("a")],
            [],
        ));

        assert!(registry.register_provider(provider).await.is_err());
        assert!(registry.is_empty());
        assert_eq!(registry.version(), version);
    }

    #[tokio::test]
    async fn provider_refresh_rejects_raw_duplicates_before_get_tool() {
        let registry = ToolRegistry::new();
        let provider = Arc::new(TestProvider::new(
            "duplicate_refresh",
            vec![descriptor("a")],
            ["a"],
        ));
        registry.register_provider(provider.clone()).await.unwrap();
        let original = registry.get("a").unwrap();
        let version = registry.version();
        let get_calls = provider.get_call_count();

        for available in [vec!["a"], Vec::new()] {
            provider.set_snapshot(vec![descriptor("a"), descriptor("a")], available);
            assert!(
                registry
                    .refresh_provider("duplicate_refresh")
                    .await
                    .is_err()
            );
            assert!(Arc::ptr_eq(&original, &registry.get("a").unwrap()));
            assert_eq!(registry.version(), version);
            assert_eq!(provider.get_call_count(), get_calls);
        }
    }

    #[tokio::test]
    async fn provider_registration_rejects_foreign_none_before_get_tool() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool { id: "owned".into() }))
            .unwrap();
        let original = registry.get("owned").unwrap();
        let version = registry.version();
        let provider = Arc::new(TestProvider::new(
            "foreign_none",
            vec![descriptor("owned")],
            [],
        ));

        assert!(registry.register_provider(provider.clone()).await.is_err());
        assert_eq!(provider.get_call_count(), 0);
        assert!(Arc::ptr_eq(&original, &registry.get("owned").unwrap()));
        assert_eq!(registry.version(), version);
        assert!(!registry.list_providers().contains(&"foreign_none".into()));
    }

    #[tokio::test]
    async fn provider_snapshot_skips_valid_missing_tool() {
        let registry = ToolRegistry::new();
        let provider = Arc::new(TestProvider::new(
            "partial",
            vec![descriptor("missing"), descriptor("present")],
            ["present"],
        ));

        registry.register_provider(provider).await.unwrap();
        assert!(registry.get("missing").is_none());
        assert!(registry.get("present").is_some());
        assert_eq!(registry.len(), 1);
    }

    #[tokio::test]
    async fn provider_empty_snapshot_is_valid() {
        let registry = ToolRegistry::new();
        let provider = Arc::new(TestProvider::new("empty", vec![descriptor("missing")], []));
        registry.register_provider(provider.clone()).await.unwrap();
        assert!(registry.is_empty());
        assert_eq!(registry.list_providers(), vec!["empty".to_string()]);

        provider.set_snapshot(vec![descriptor("present")], ["present"]);
        registry.refresh_provider("empty").await.unwrap();
        assert!(registry.get("present").is_some());
        provider.set_snapshot(Vec::new(), []);
        registry.refresh_provider("empty").await.unwrap();
        assert!(registry.is_empty());
        assert_eq!(registry.list_providers(), vec!["empty".to_string()]);
    }

    #[tokio::test]
    async fn provider_refresh_commits_one_complete_snapshot() {
        let registry = ToolRegistry::new();
        let provider = Arc::new(TestProvider::new(
            "complete_refresh",
            vec![
                descriptor("a").with_aliases(ToolAliases::new().with_name("en", "old a")),
                descriptor("b").with_aliases(ToolAliases::new().with_name("en", "old b")),
            ],
            ["a", "b"],
        ));
        registry.register_provider(provider.clone()).await.unwrap();
        let old_a = registry.get("a").unwrap();
        let version = registry.version();

        provider.set_snapshot(
            vec![
                ToolDescriptor::new(
                    "a",
                    "replacement a",
                    "replacement description",
                    serde_json::json!({"type": "object"}),
                )
                .with_aliases(ToolAliases::new().with_name("en", "new a")),
                descriptor("c").with_aliases(ToolAliases::new().with_name("en", "new c")),
            ],
            ["a", "c"],
        );
        registry.refresh_provider("complete_refresh").await.unwrap();

        assert!(!Arc::ptr_eq(&old_a, &registry.get("a").unwrap()));
        assert!(registry.get("b").is_none());
        assert!(registry.get("c").is_some());
        assert!(registry.resolve("old a").is_none());
        assert!(registry.resolve("old b").is_none());
        assert_eq!(
            registry.resolve("new a").unwrap().identity.canonical_id,
            "a"
        );
        assert_eq!(
            registry.resolve("new c").unwrap().identity.canonical_id,
            "c"
        );
        assert_eq!(registry.version(), version + 1);
    }

    #[tokio::test]
    async fn concurrent_registration_has_one_complete_owner() {
        let registry = Arc::new(ToolRegistry::new());
        let gate = Arc::new(Barrier::new(3));
        let first = Arc::new(
            TestProvider::new("same", vec![descriptor("first")], ["first"])
                .with_list_gate(gate.clone()),
        );
        let second = Arc::new(
            TestProvider::new("same", vec![descriptor("second")], ["second"])
                .with_list_gate(gate.clone()),
        );
        let first_registry = registry.clone();
        let first_task = tokio::spawn(async move { first_registry.register_provider(first).await });
        let second_registry = registry.clone();
        let second_task =
            tokio::spawn(async move { second_registry.register_provider(second).await });
        gate.wait().await;

        let first_result = first_task.await.unwrap();
        let second_result = second_task.await.unwrap();
        assert_ne!(first_result.is_ok(), second_result.is_ok());
        assert_eq!(registry.list_providers(), vec!["same".to_string()]);
        assert_eq!(registry.len(), 1);
        assert_ne!(
            registry.get("first").is_some(),
            registry.get("second").is_some()
        );
    }

    #[tokio::test]
    async fn provider_refresh_rejects_foreign_none_descriptor() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool {
                id: "foreign".into(),
            }))
            .unwrap();
        let foreign = registry.get("foreign").unwrap();
        let provider = Arc::new(TestProvider::new(
            "owner",
            vec![descriptor("owned")],
            ["owned"],
        ));
        registry.register_provider(provider.clone()).await.unwrap();
        let version = registry.version();
        let get_calls = provider.get_call_count();
        provider.set_snapshot(vec![descriptor("foreign")], []);

        assert!(registry.refresh_provider("owner").await.is_err());
        assert_eq!(provider.get_call_count(), get_calls);
        assert!(registry.get("owned").is_some());
        assert!(Arc::ptr_eq(&foreign, &registry.get("foreign").unwrap()));
        assert_eq!(registry.version(), version);
    }

    #[tokio::test]
    async fn provider_refresh_rejects_another_provider_owner_before_get_tool() {
        let registry = ToolRegistry::new();
        let first = Arc::new(TestProvider::new(
            "first_owner",
            vec![descriptor("a")],
            ["a"],
        ));
        let second = Arc::new(TestProvider::new(
            "second_owner",
            vec![descriptor("b")],
            ["b"],
        ));
        registry.register_provider(first.clone()).await.unwrap();
        registry.register_provider(second).await.unwrap();
        let original_a = registry.get("a").unwrap();
        let original_b = registry.get("b").unwrap();
        let version = registry.version();
        let get_calls = first.get_call_count();

        first.set_snapshot(vec![descriptor("b")], ["b"]);
        assert!(registry.refresh_provider("first_owner").await.is_err());
        assert_eq!(first.get_call_count(), get_calls);
        assert!(Arc::ptr_eq(&original_a, &registry.get("a").unwrap()));
        assert!(Arc::ptr_eq(&original_b, &registry.get("b").unwrap()));
        assert_eq!(registry.version(), version);
    }

    #[tokio::test]
    async fn provider_refresh_serializes_same_registration() {
        let registry = Arc::new(ToolRegistry::new());
        let provider = Arc::new(BlockingRefreshProvider {
            id: "serial".into(),
            tool_id: "serial_tool".into(),
            started: Arc::new(Notify::new()),
            permits: Arc::new(Semaphore::new(0)),
            refresh_calls: AtomicUsize::new(0),
            active_refreshes: AtomicUsize::new(0),
            max_active_refreshes: AtomicUsize::new(0),
            failures_remaining: AtomicUsize::new(0),
        });
        registry.register_provider(provider.clone()).await.unwrap();

        let first_started = provider.started.notified();
        let first_registry = registry.clone();
        let first = tokio::spawn(async move { first_registry.refresh_provider("serial").await });
        first_started.await;

        let second_registry = registry.clone();
        let second = tokio::spawn(async move { second_registry.refresh_provider("serial").await });
        tokio::task::yield_now().await;
        assert_eq!(provider.refresh_calls.load(Ordering::SeqCst), 1);

        let second_started = provider.started.notified();
        provider.permits.add_permits(1);
        first.await.unwrap().unwrap();
        second_started.await;
        provider.permits.add_permits(1);
        second.await.unwrap().unwrap();

        assert_eq!(provider.refresh_calls.load(Ordering::SeqCst), 2);
        assert_eq!(provider.max_active_refreshes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn provider_refresh_failure_or_cancel_releases_queue() {
        let registry = Arc::new(ToolRegistry::new());
        let provider = Arc::new(BlockingRefreshProvider {
            id: "cancelled".into(),
            tool_id: "cancelled_tool".into(),
            started: Arc::new(Notify::new()),
            permits: Arc::new(Semaphore::new(0)),
            refresh_calls: AtomicUsize::new(0),
            active_refreshes: AtomicUsize::new(0),
            max_active_refreshes: AtomicUsize::new(0),
            failures_remaining: AtomicUsize::new(0),
        });
        registry.register_provider(provider.clone()).await.unwrap();
        let version = registry.version();

        let first_started = provider.started.notified();
        let first_registry = registry.clone();
        let first = tokio::spawn(async move { first_registry.refresh_provider("cancelled").await });
        first_started.await;
        first.abort();
        assert!(first.await.unwrap_err().is_cancelled());

        let second_started = provider.started.notified();
        let second_registry = registry.clone();
        let second =
            tokio::spawn(async move { second_registry.refresh_provider("cancelled").await });
        second_started.await;
        provider.permits.add_permits(1);
        second.await.unwrap().unwrap();

        assert_eq!(provider.refresh_calls.load(Ordering::SeqCst), 2);
        assert_eq!(provider.max_active_refreshes.load(Ordering::SeqCst), 1);
        assert_eq!(registry.version(), version + 1);
    }

    #[tokio::test]
    async fn provider_refresh_error_releases_queued_refresh() {
        let registry = Arc::new(ToolRegistry::new());
        let provider = Arc::new(BlockingRefreshProvider {
            id: "failing".into(),
            tool_id: "failing_tool".into(),
            started: Arc::new(Notify::new()),
            permits: Arc::new(Semaphore::new(0)),
            refresh_calls: AtomicUsize::new(0),
            active_refreshes: AtomicUsize::new(0),
            max_active_refreshes: AtomicUsize::new(0),
            failures_remaining: AtomicUsize::new(1),
        });
        registry.register_provider(provider.clone()).await.unwrap();
        let version = registry.version();

        let first_started = provider.started.notified();
        let first_registry = registry.clone();
        let first = tokio::spawn(async move { first_registry.refresh_provider("failing").await });
        first_started.await;

        let second_registry = registry.clone();
        let second = tokio::spawn(async move { second_registry.refresh_provider("failing").await });
        tokio::task::yield_now().await;
        provider.permits.add_permits(1);
        assert!(first.await.unwrap().is_err());
        assert_eq!(registry.version(), version);

        let second_started = provider.started.notified();
        second_started.await;
        provider.permits.add_permits(1);
        second.await.unwrap().unwrap();
        assert_eq!(provider.refresh_calls.load(Ordering::SeqCst), 2);
        assert_eq!(provider.max_active_refreshes.load(Ordering::SeqCst), 1);
        assert_eq!(registry.version(), version + 1);
    }

    #[tokio::test]
    async fn independent_provider_refreshes_do_not_share_an_io_lock() {
        let registry = Arc::new(ToolRegistry::new());
        let first = Arc::new(BlockingRefreshProvider {
            id: "first".into(),
            tool_id: "first_tool".into(),
            started: Arc::new(Notify::new()),
            permits: Arc::new(Semaphore::new(0)),
            refresh_calls: AtomicUsize::new(0),
            active_refreshes: AtomicUsize::new(0),
            max_active_refreshes: AtomicUsize::new(0),
            failures_remaining: AtomicUsize::new(0),
        });
        let second = Arc::new(BlockingRefreshProvider {
            id: "second".into(),
            tool_id: "second_tool".into(),
            started: Arc::new(Notify::new()),
            permits: Arc::new(Semaphore::new(0)),
            refresh_calls: AtomicUsize::new(0),
            active_refreshes: AtomicUsize::new(0),
            max_active_refreshes: AtomicUsize::new(0),
            failures_remaining: AtomicUsize::new(0),
        });
        registry.register_provider(first.clone()).await.unwrap();
        registry.register_provider(second.clone()).await.unwrap();

        let first_started = first.started.notified();
        let first_registry = registry.clone();
        let first_task =
            tokio::spawn(async move { first_registry.refresh_provider("first").await });
        first_started.await;
        assert!(registry.get("second_tool").is_some());

        let second_started = second.started.notified();
        let second_registry = registry.clone();
        let second_task =
            tokio::spawn(async move { second_registry.refresh_provider("second").await });
        second_started.await;
        assert_eq!(first.refresh_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second.refresh_calls.load(Ordering::SeqCst), 1);

        first.permits.add_permits(1);
        second.permits.add_permits(1);
        first_task.await.unwrap().unwrap();
        second_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn provider_refresh_cannot_resurrect_unregistered_provider() {
        let registry = Arc::new(ToolRegistry::new());
        let provider = Arc::new(BlockingRefreshProvider {
            id: "stale".into(),
            tool_id: "stale_tool".into(),
            started: Arc::new(Notify::new()),
            permits: Arc::new(Semaphore::new(0)),
            refresh_calls: AtomicUsize::new(0),
            active_refreshes: AtomicUsize::new(0),
            max_active_refreshes: AtomicUsize::new(0),
            failures_remaining: AtomicUsize::new(0),
        });
        registry.register_provider(provider.clone()).await.unwrap();
        let version_after_registration = registry.version();

        let started = provider.started.notified();
        let refresh_registry = registry.clone();
        let refresh = tokio::spawn(async move { refresh_registry.refresh_provider("stale").await });
        started.await;
        assert!(registry.unregister_provider("stale"));
        let version_after_removal = registry.version();
        assert_eq!(version_after_removal, version_after_registration + 1);
        let replacement = Arc::new(TestProvider::new(
            "stale",
            vec![descriptor("replacement_tool")],
            ["replacement_tool"],
        ));
        registry.register_provider(replacement).await.unwrap();
        let version_after_replacement = registry.version();
        provider.permits.add_permits(1);

        assert!(refresh.await.unwrap().is_err());
        assert!(registry.get("stale_tool").is_none());
        assert!(registry.get("replacement_tool").is_some());
        assert_eq!(registry.version(), version_after_replacement);
    }

    #[tokio::test]
    async fn alias_ambiguity_survives_three_claimants_and_recovers() {
        let registry = ToolRegistry::new();
        for id in ["a", "b", "c"] {
            let aliases = ToolAliases::new().with_name("ko", "검색");
            let provider = Arc::new(TestProvider::new(
                format!("provider_{id}"),
                vec![descriptor(id).with_aliases(aliases)],
                match id {
                    "a" => ["a"],
                    "b" => ["b"],
                    _ => ["c"],
                },
            ));
            registry.register_provider(provider).await.unwrap();
        }

        assert!(registry.resolve("검색").is_none());
        assert!(registry.get_by_alias("검색", "ko").is_none());
        assert!(registry.unregister_provider("provider_b"));
        assert!(registry.resolve("검색").is_none());
        assert!(registry.unregister_provider("provider_c"));
        assert_eq!(registry.resolve("검색").unwrap().identity.canonical_id, "a");
    }

    #[tokio::test]
    async fn localized_aliases_keep_plain_lookup_ambiguous_across_languages() {
        let registry = ToolRegistry::new();
        let english = Arc::new(TestProvider::new(
            "english",
            vec![descriptor("a").with_aliases(ToolAliases::new().with_name("en", "shared"))],
            ["a"],
        ));
        let korean = Arc::new(TestProvider::new(
            "korean",
            vec![descriptor("b").with_aliases(ToolAliases::new().with_name("ko", "shared"))],
            ["b"],
        ));
        registry.register_provider(english).await.unwrap();
        registry.register_provider(korean).await.unwrap();

        assert!(registry.resolve("shared").is_none());
        assert_eq!(registry.get_by_alias("shared", "en").unwrap().id(), "a");
        assert_eq!(registry.get_by_alias("shared", "ko").unwrap().id(), "b");
        assert!(registry.unregister_provider("korean"));
        assert_eq!(
            registry.resolve("shared").unwrap().identity.canonical_id,
            "a"
        );
    }

    #[test]
    fn normalized_canonical_collision_requires_exact_id() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool {
                id: "Search".into(),
            }))
            .unwrap();
        registry
            .register(Arc::new(TestTool {
                id: "search".into(),
            }))
            .unwrap();

        assert_eq!(
            registry.resolve("Search").unwrap().identity.canonical_id,
            "Search"
        );
        assert_eq!(
            registry.resolve("search").unwrap().identity.canonical_id,
            "search"
        );
        registry
            .register(Arc::new(TestTool { id: "other".into() }))
            .unwrap();
        registry.set_tool_aliases("other", ToolAliases::new().with_name("en", "SEARCH"));
        assert!(registry.resolve("SEARCH").is_none());
    }

    #[test]
    fn builtin_display_name_survives_a_renaming_wrapper() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool {
                id: "builtin".into(),
            }))
            .unwrap();

        let mapped = registry.map_tools(|tool| {
            Arc::new(RenamedTool {
                id: tool.id().to_string(),
            })
        });

        assert_eq!(
            mapped.resolve("Test").unwrap().identity.canonical_id,
            "builtin"
        );
        assert!(mapped.resolve("Wrapped").is_none());
        assert_eq!(mapped.get("builtin").unwrap().name(), "Wrapped");
    }

    #[tokio::test]
    async fn provider_descriptor_display_name_survives_mapping() {
        let registry = ToolRegistry::new();
        let provider = Arc::new(TestProvider::new(
            "weather_provider",
            vec![ToolDescriptor::new(
                "weather",
                "Weather Lookup",
                "Looks up weather",
                serde_json::json!({"type": "object"}),
            )],
            ["weather"],
        ));
        registry.register_provider(provider).await.unwrap();

        assert_eq!(
            registry
                .resolve("Weather Lookup")
                .unwrap()
                .identity
                .canonical_id,
            "weather"
        );
        assert!(registry.resolve("Test").is_none());
        let mapped = registry.map_tools(|tool| tool);
        assert_eq!(
            mapped
                .resolve("Weather Lookup")
                .unwrap()
                .identity
                .canonical_id,
            "weather"
        );
    }

    #[tokio::test]
    async fn aliases_do_not_transfer_to_new_provider_or_mapped_registry() {
        let registry = ToolRegistry::new();
        let first = Arc::new(TestProvider::new(
            "first",
            vec![descriptor("shared")],
            ["shared"],
        ));
        registry.register_provider(first).await.unwrap();
        registry.set_tool_aliases("shared", ToolAliases::new().with_name("en", "old alias"));
        assert!(registry.resolve("old alias").is_some());
        assert!(registry.unregister_provider("first"));

        let second = Arc::new(TestProvider::new(
            "second",
            vec![descriptor("shared")],
            ["shared"],
        ));
        registry.register_provider(second).await.unwrap();
        assert!(registry.resolve("old alias").is_none());
        let mapped = registry.map_tools(|tool| tool);
        assert!(mapped.resolve("old alias").is_none());
    }

    #[test]
    fn map_tools_invokes_callback_outside_registry_lock() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool { id: "a".into() }))
            .unwrap();
        let captured_version = registry.version();
        let mapped = registry.map_tools(|tool| {
            assert!(registry.get("a").is_some());
            registry.set_tool_aliases("a", ToolAliases::new().with_name("en", "alias"));
            tool
        });

        assert!(registry.resolve("alias").is_some());
        assert!(mapped.resolve("alias").is_none());
        assert!(mapped.get("a").is_some());
        assert_eq!(mapped.version(), captured_version);
        assert_eq!(registry.version(), captured_version + 1);
    }

    #[test]
    fn test_set_tool_aliases() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool {
                id: "calculator".to_string(),
            }))
            .unwrap();

        let aliases = ToolAliases::new()
            .with_name("ko", "계산기")
            .with_name("ja", "計算機")
            .with_description("ko", "수학 계산을 합니다");

        registry.set_tool_aliases("calculator", aliases);

        assert!(registry.get_by_alias("계산기", "ko").is_some());
        assert!(registry.get_by_alias("計算機", "ja").is_some());
        assert!(registry.get("calculator").is_some());
    }

    #[test]
    fn alias_lookup_and_mutation_complete_without_lock_inversion() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool { id: "a".into() }))
            .unwrap();
        registry.set_tool_aliases("a", ToolAliases::new().with_name("en", "alias-0"));
        let registry = Arc::new(registry);
        let barrier = Arc::new(std::sync::Barrier::new(2));

        let reader_registry = registry.clone();
        let reader_barrier = barrier.clone();
        let reader = std::thread::spawn(move || {
            reader_barrier.wait();
            for index in 0..100 {
                let _ = reader_registry.resolve(&format!("alias-{index}"));
                let _ = reader_registry.get_by_alias("alias-0", "en");
            }
        });
        let writer_registry = registry.clone();
        let writer = std::thread::spawn(move || {
            barrier.wait();
            for index in 1..100 {
                writer_registry.set_tool_aliases(
                    "a",
                    ToolAliases::new().with_name("en", format!("alias-{index}")),
                );
            }
        });

        reader.join().unwrap();
        writer.join().unwrap();
        assert_eq!(
            registry.resolve("alias-99").unwrap().identity.canonical_id,
            "a"
        );
        assert_eq!(
            registry.resolve("alias-0").unwrap().identity.canonical_id,
            "a"
        );
    }

    #[test]
    fn set_tool_aliases_retains_prior_lookup_claims() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool {
                id: "search".to_string(),
            }))
            .unwrap();
        registry.set_tool_aliases("search", ToolAliases::new().with_name("en", "old alias"));
        registry.set_tool_aliases("search", ToolAliases::new().with_name("en", "new alias"));

        assert!(registry.resolve("old alias").is_some());
        assert!(registry.resolve("new alias").is_some());
        let prompt = registry.generate_tools_prompt_with_lang(Some("en"), false);
        assert!(prompt.contains("new alias"));
        assert!(!prompt.contains("old alias"));
    }

    #[test]
    fn version_exhaustion_does_not_publish_an_unversioned_change() {
        let mut registry = ToolRegistry::new();
        registry.state.write().version = u64::MAX;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            registry.register(Arc::new(TestTool {
                id: "never_published".into(),
            }))
        }));

        assert!(result.is_err());
        assert!(registry.get("never_published").is_none());
        assert_eq!(registry.version(), u64::MAX);
    }

    #[test]
    fn test_get_by_alias_case_insensitive() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool {
                id: "search".to_string(),
            }))
            .unwrap();

        let aliases = ToolAliases::new().with_name("ko", "검색");
        registry.set_tool_aliases("search", aliases);

        assert!(registry.get_by_alias("검색", "ko").is_some());
    }

    #[test]
    fn test_generate_prompt_with_language() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool {
                id: "calculator".to_string(),
            }))
            .unwrap();

        let aliases = ToolAliases::new()
            .with_name("ko", "계산기")
            .with_description("ko", "수학 계산");

        registry.set_tool_aliases("calculator", aliases);

        let prompt_en = registry.generate_tools_prompt_with_lang(None, false);
        assert!(prompt_en.contains("Test"));

        let prompt_ko = registry.generate_tools_prompt_with_lang(Some("ko"), false);
        assert!(prompt_ko.contains("계산기"));
        assert!(prompt_ko.contains("수학 계산"));
    }

    #[test]
    fn test_generate_tools_prompt_parallel() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool {
                id: "tool_a".to_string(),
            }))
            .unwrap();
        registry
            .register(Arc::new(TestTool {
                id: "tool_b".to_string(),
            }))
            .unwrap();

        // Without parallel: no array instruction
        let prompt_seq = registry.generate_tools_prompt();
        assert!(prompt_seq.contains("\"tool\": \"tool_name\""));
        assert!(!prompt_seq.contains("JSON array"));
        assert!(!prompt_seq.contains("tool_name1"));

        // With parallel: array instruction present
        let prompt_par = registry.generate_tools_prompt_with_parallel(true);
        assert!(prompt_par.contains("\"tool\": \"tool_name\""));
        assert!(prompt_par.contains("JSON array"));
        assert!(prompt_par.contains("tool_name1"));
        assert!(prompt_par.contains("tool_name2"));
    }

    #[test]
    fn test_canonical_resolution_and_scoped_empty_prompt() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool {
                id: "calculator".to_string(),
            }))
            .unwrap();
        let aliases = ToolAliases::new().with_name("ko", "계산기");
        registry.set_tool_aliases("calculator", aliases);

        let by_id = registry.resolve("calculator").unwrap();
        assert_eq!(by_id.identity.canonical_id, "calculator");

        let by_alias = registry.resolve("계산기").unwrap();
        assert_eq!(by_alias.identity.canonical_id, "calculator");

        let scoped = registry.generate_scoped_prompt_with_parallel(&[], false);
        assert!(scoped.is_empty());
    }

    #[test]
    fn test_generate_filtered_prompt_parallel() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool {
                id: "tool_a".to_string(),
            }))
            .unwrap();
        registry
            .register(Arc::new(TestTool {
                id: "tool_b".to_string(),
            }))
            .unwrap();

        // Filtered without parallel
        let prompt_seq =
            registry.generate_filtered_prompt(&["tool_a".to_string(), "tool_b".to_string()]);
        assert!(!prompt_seq.contains("JSON array"));

        // Filtered with parallel
        let prompt_par = registry.generate_filtered_prompt_with_parallel(
            &["tool_a".to_string(), "tool_b".to_string()],
            true,
        );
        assert!(prompt_par.contains("JSON array"));
        assert!(prompt_par.contains("tool_name1"));
    }
}

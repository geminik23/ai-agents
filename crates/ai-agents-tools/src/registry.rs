use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ai_agents_core::{LLMProvider, Tool, ToolInfo, ToolSafetyMetadata};

use super::ToolError;
use super::provider::{ProviderHealth, ToolProvider, ToolProviderError};
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
        tool: Arc<dyn Tool>,
    },
}

/// Registry for built-in, provider, alias, and localized tool lookup.
pub struct ToolRegistry {
    builtin_tools: RwLock<HashMap<String, Arc<dyn Tool>>>,

    providers: RwLock<HashMap<String, Arc<dyn ToolProvider>>>,

    tool_index: RwLock<HashMap<String, ToolRef>>,

    alias_index: RwLock<HashMap<String, String>>,

    display_name_index: RwLock<HashMap<String, String>>,

    builtin_aliases: RwLock<HashMap<String, ToolAliases>>,

    question_handler: QuestionHandlerSlot,

    diagnostics_provider: DiagnosticsProviderSlot,

    command_runner: CommandRunnerSlot,

    todo_store: TodoStore,

    file_versions: FileVersionStore,

    web_fetch_extractor: Arc<RwLock<Option<Arc<dyn LLMProvider>>>>,

    web_search_provider: WebSearchProviderSlot,

    registry_version: AtomicU64,
}

impl ToolRegistry {
    /// Creates an empty registry with versioned canonical indexes.
    pub fn new() -> Self {
        Self {
            builtin_tools: RwLock::new(HashMap::new()),
            providers: RwLock::new(HashMap::new()),
            tool_index: RwLock::new(HashMap::new()),
            alias_index: RwLock::new(HashMap::new()),
            display_name_index: RwLock::new(HashMap::new()),
            builtin_aliases: RwLock::new(HashMap::new()),
            question_handler: Arc::new(RwLock::new(None)),
            diagnostics_provider: Arc::new(RwLock::new(Arc::new(UnavailableDiagnosticsProvider))),
            command_runner: Arc::new(RwLock::new(Arc::new(UnavailableCommandRunner))),
            todo_store: TodoStore::default(),
            file_versions: FileVersionStore::default(),
            web_fetch_extractor: Arc::new(RwLock::new(None)),
            web_search_provider: Arc::new(RwLock::new(Arc::new(UnavailableWebSearchProvider))),
            registry_version: AtomicU64::new(1),
        }
    }

    fn bump_version(&self) {
        self.registry_version.fetch_add(1, Ordering::SeqCst);
    }

    /// Returns the registry version used in tool execution evidence.
    pub fn version(&self) -> u64 {
        self.registry_version.load(Ordering::SeqCst)
    }

    fn normalize_key(value: &str) -> String {
        value.trim().to_lowercase()
    }

    fn insert_unique_index(index: &mut HashMap<String, String>, key: String, tool_id: &str) {
        match index.get(&key) {
            None => {
                index.insert(key, tool_id.to_string());
            }
            Some(existing) if existing == tool_id => {}
            Some(_) => {
                index.remove(&key);
            }
        }
    }

    /// Registers a built-in or custom tool by canonical ID.
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Result<(), ToolError> {
        let id = tool.id().to_string();

        let mut builtin_tools = self.builtin_tools.write();
        let mut tool_index = self.tool_index.write();
        let mut display_name_index = self.display_name_index.write();

        if builtin_tools.contains_key(&id) || tool_index.contains_key(&id) {
            return Err(ToolError::Duplicate(id));
        }

        Self::insert_unique_index(
            &mut display_name_index,
            Self::normalize_key(tool.name()),
            &id,
        );
        tool_index.insert(id.clone(), ToolRef::Builtin(tool.clone()));
        builtin_tools.insert(id, tool);
        self.bump_version();
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

    /// Resolves IDs, display names, and aliases to one canonical tool.
    pub fn resolve(&self, id_or_alias: &str) -> Option<ResolvedTool> {
        let tool_index = self.tool_index.read();
        let requested_name = id_or_alias.to_string();
        let normalized = Self::normalize_key(id_or_alias);

        if let Some((canonical_id, tool_ref)) =
            tool_index.get_key_value(id_or_alias).or_else(|| {
                tool_index
                    .iter()
                    .find(|(id, _)| Self::normalize_key(id) == normalized)
            })
        {
            return self.resolved_tool_from_ref(&requested_name, canonical_id, tool_ref);
        }

        if let Some(tool_id) = self.display_name_index.read().get(&normalized).cloned()
            && let Some(tool_ref) = tool_index.get(&tool_id)
        {
            return self.resolved_tool_from_ref(&requested_name, &tool_id, tool_ref);
        }

        let alias_index = self.alias_index.read();
        if let Some(tool_id) = alias_index.get(&normalized).cloned().or_else(|| {
            alias_index.iter().find_map(|(alias_key, tool_id)| {
                alias_key
                    .ends_with(&format!(":{}", normalized))
                    .then(|| tool_id.clone())
            })
        }) && let Some(tool_ref) = tool_index.get(&tool_id)
        {
            return self.resolved_tool_from_ref(&requested_name, &tool_id, tool_ref);
        }

        None
    }

    fn resolved_tool_from_ref(
        &self,
        requested_name: &str,
        canonical_id: &str,
        tool_ref: &ToolRef,
    ) -> Option<ResolvedTool> {
        let tool = self.resolve_tool_ref(tool_ref)?;
        let provider_id = match tool_ref {
            ToolRef::Builtin(_) => None,
            ToolRef::Provider { provider_id, .. } => Some(provider_id.clone()),
        };
        Some(ResolvedTool {
            identity: ToolIdentity {
                requested_name: requested_name.to_string(),
                canonical_id: canonical_id.to_string(),
                display_name: tool.name().to_string(),
                provider_id,
            },
            tool,
        })
    }

    fn resolve_tool_ref(&self, tool_ref: &ToolRef) -> Option<Arc<dyn Tool>> {
        match tool_ref {
            ToolRef::Builtin(tool) => Some(tool.clone()),
            ToolRef::Provider { tool, .. } => Some(tool.clone()),
        }
    }

    pub fn list_ids(&self) -> Vec<String> {
        self.tool_index.read().keys().cloned().collect()
    }

    pub fn list_infos(&self) -> Vec<ToolInfo> {
        let tool_index = self.tool_index.read();
        let mut infos = Vec::with_capacity(tool_index.len());

        for tool_ref in tool_index.values() {
            if let Some(tool) = self.resolve_tool_ref(tool_ref) {
                infos.push(tool.info());
            }
        }

        infos
    }

    pub fn len(&self) -> usize {
        self.tool_index.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.tool_index.read().is_empty()
    }

    pub fn map_tools<F>(&self, mut f: F) -> ToolRegistry
    where
        F: FnMut(Arc<dyn Tool>) -> Arc<dyn Tool>,
    {
        let mut mapped = ToolRegistry::new();
        mapped.question_handler = Arc::clone(&self.question_handler);
        mapped.diagnostics_provider = Arc::clone(&self.diagnostics_provider);
        mapped.command_runner = Arc::clone(&self.command_runner);
        mapped.todo_store = self.todo_store.clone();
        mapped.file_versions = self.file_versions.clone();
        mapped.web_fetch_extractor = Arc::clone(&self.web_fetch_extractor);
        mapped.web_search_provider = Arc::clone(&self.web_search_provider);

        {
            let providers = self.providers.read();
            let mut mapped_providers = mapped.providers.write();
            for (id, provider) in providers.iter() {
                mapped_providers.insert(id.clone(), provider.clone());
            }
        }

        {
            let aliases = self.alias_index.read();
            let mut mapped_aliases = mapped.alias_index.write();
            for (alias, tool_id) in aliases.iter() {
                mapped_aliases.insert(alias.clone(), tool_id.clone());
            }
        }

        {
            let display_names = self.display_name_index.read();
            let mut mapped_display_names = mapped.display_name_index.write();
            for (name, tool_id) in display_names.iter() {
                mapped_display_names.insert(name.clone(), tool_id.clone());
            }
        }

        {
            let builtin_aliases = self.builtin_aliases.read();
            let mut mapped_builtin_aliases = mapped.builtin_aliases.write();
            for (id, aliases) in builtin_aliases.iter() {
                mapped_builtin_aliases.insert(id.clone(), aliases.clone());
            }
        }

        let tool_index = self.tool_index.read();
        let mut mapped_tool_index = mapped.tool_index.write();
        let mut mapped_builtin_tools = mapped.builtin_tools.write();
        for (id, tool_ref) in tool_index.iter() {
            match tool_ref {
                ToolRef::Builtin(tool) => {
                    let wrapped = f(tool.clone());
                    mapped_tool_index.insert(id.clone(), ToolRef::Builtin(wrapped.clone()));
                    mapped_builtin_tools.insert(id.clone(), wrapped);
                }
                ToolRef::Provider { provider_id, tool } => {
                    let wrapped = f(tool.clone());
                    mapped_tool_index.insert(
                        id.clone(),
                        ToolRef::Provider {
                            provider_id: provider_id.clone(),
                            tool: wrapped,
                        },
                    );
                }
            }
        }

        drop(mapped_builtin_tools);
        drop(mapped_tool_index);
        mapped
            .registry_version
            .store(self.version(), Ordering::SeqCst);
        mapped
    }

    pub async fn register_provider(
        &self,
        provider: Arc<dyn ToolProvider>,
    ) -> Result<(), ToolError> {
        let provider_id = provider.id().to_string();

        {
            let providers = self.providers.read();
            if providers.contains_key(&provider_id) {
                return Err(ToolError::Duplicate(format!("Provider: {}", provider_id)));
            }
        }

        let tools = provider.list_tools().await;

        // Resolve provider tools before taking index locks so provider I/O cannot block registry access.
        let mut resolved_tools = Vec::with_capacity(tools.len());
        for descriptor in &tools {
            resolved_tools.push((descriptor, provider.get_tool(&descriptor.id).await));
        }

        {
            let mut tool_index = self.tool_index.write();
            let mut alias_index = self.alias_index.write();
            let mut display_name_index = self.display_name_index.write();

            for (descriptor, tool) in resolved_tools {
                if tool_index.contains_key(&descriptor.id) {
                    return Err(ToolError::Duplicate(descriptor.id.clone()));
                }

                if let Some(tool) = tool {
                    Self::insert_unique_index(
                        &mut display_name_index,
                        Self::normalize_key(&descriptor.name),
                        &descriptor.id,
                    );
                    tool_index.insert(
                        descriptor.id.clone(),
                        ToolRef::Provider {
                            provider_id: provider_id.clone(),
                            tool,
                        },
                    );

                    if let Some(ref aliases) = descriptor.aliases {
                        for (lang, name) in &aliases.names {
                            let key = format!("{}:{}", lang, Self::normalize_key(name));
                            Self::insert_unique_index(&mut alias_index, key, &descriptor.id);
                            Self::insert_unique_index(
                                &mut alias_index,
                                Self::normalize_key(name),
                                &descriptor.id,
                            );
                        }
                    }
                }
            }
        }

        self.providers.write().insert(provider_id, provider);
        self.bump_version();

        Ok(())
    }

    pub fn unregister_provider(&self, provider_id: &str) -> bool {
        let removed = self.providers.write().remove(provider_id);

        if removed.is_some() {
            let mut tool_index = self.tool_index.write();
            let mut alias_index = self.alias_index.write();
            let mut display_name_index = self.display_name_index.write();

            let tools_to_remove: Vec<String> = tool_index
                .iter()
                .filter_map(|(id, tool_ref)| {
                    if let ToolRef::Provider {
                        provider_id: pid, ..
                    } = tool_ref
                        && pid == provider_id
                    {
                        return Some(id.clone());
                    }
                    None
                })
                .collect();

            for tool_id in &tools_to_remove {
                tool_index.remove(tool_id);
            }

            alias_index.retain(|_, tool_id| !tools_to_remove.contains(tool_id));
            display_name_index.retain(|_, tool_id| !tools_to_remove.contains(tool_id));
            self.bump_version();

            true
        } else {
            false
        }
    }

    pub fn set_tool_aliases(&self, tool_id: &str, aliases: ToolAliases) {
        if !self.tool_index.read().contains_key(tool_id) {
            return;
        }

        {
            let mut alias_index = self.alias_index.write();
            for (lang, name) in &aliases.names {
                let key = format!("{}:{}", lang, Self::normalize_key(name));
                Self::insert_unique_index(&mut alias_index, key, tool_id);
                Self::insert_unique_index(&mut alias_index, Self::normalize_key(name), tool_id);
            }
        }

        self.builtin_aliases
            .write()
            .insert(tool_id.to_string(), aliases);
        self.bump_version();
    }

    pub fn get_by_alias(&self, alias: &str, lang: &str) -> Option<Arc<dyn Tool>> {
        let key = format!("{}:{}", lang, Self::normalize_key(alias));
        let alias_index = self.alias_index.read();

        if let Some(tool_id) = alias_index.get(&key) {
            return self.get(tool_id);
        }

        None
    }

    pub fn list_providers(&self) -> Vec<String> {
        self.providers.read().keys().cloned().collect()
    }

    pub async fn provider_health(&self, provider_id: &str) -> Option<ProviderHealth> {
        let provider = self.providers.read().get(provider_id).cloned();
        if let Some(provider) = provider {
            Some(provider.health_check().await)
        } else {
            None
        }
    }

    pub async fn refresh_provider(&self, provider_id: &str) -> Result<(), ToolProviderError> {
        let provider = {
            let providers = self.providers.read();
            providers.get(provider_id).cloned()
        };

        if let Some(provider) = provider {
            if provider.supports_refresh() {
                provider.refresh().await?;

                let tools = provider.list_tools().await;

                // Resolve provider tools before taking index locks so refresh keeps the old snapshot available during provider I/O.
                let mut resolved_tools = Vec::with_capacity(tools.len());
                for descriptor in &tools {
                    if let Some(tool) = provider.get_tool(&descriptor.id).await {
                        resolved_tools.push((descriptor, tool));
                    }
                }

                {
                    let mut tool_index = self.tool_index.write();
                    let mut alias_index = self.alias_index.write();
                    let mut display_name_index = self.display_name_index.write();

                    let old_tools: Vec<String> = tool_index
                        .iter()
                        .filter_map(|(id, tool_ref)| {
                            if let ToolRef::Provider {
                                provider_id: pid, ..
                            } = tool_ref
                                && pid == provider_id
                            {
                                return Some(id.clone());
                            }
                            None
                        })
                        .collect();

                    for tool_id in &old_tools {
                        tool_index.remove(tool_id);
                    }
                    alias_index.retain(|_, tool_id| !old_tools.contains(tool_id));
                    display_name_index.retain(|_, tool_id| !old_tools.contains(tool_id));

                    for (descriptor, tool) in resolved_tools {
                        Self::insert_unique_index(
                            &mut display_name_index,
                            Self::normalize_key(&descriptor.name),
                            &descriptor.id,
                        );
                        tool_index.insert(
                            descriptor.id.clone(),
                            ToolRef::Provider {
                                provider_id: provider_id.to_string(),
                                tool,
                            },
                        );

                        if let Some(ref aliases) = descriptor.aliases {
                            for (lang, name) in &aliases.names {
                                let key = format!("{}:{}", lang, Self::normalize_key(name));
                                Self::insert_unique_index(&mut alias_index, key, &descriptor.id);
                                Self::insert_unique_index(
                                    &mut alias_index,
                                    Self::normalize_key(name),
                                    &descriptor.id,
                                );
                            }
                        }
                    }
                    self.bump_version();
                }
            }
            Ok(())
        } else {
            Err(ToolProviderError::ToolNotFound(format!(
                "Provider not found: {}",
                provider_id
            )))
        }
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
        let tool_index = self.tool_index.read();
        if tool_index.is_empty() {
            return String::new();
        }

        let builtin_aliases = self.builtin_aliases.read();
        let mut prompt = String::from("Available tools:\n");

        for (id, tool_ref) in tool_index.iter() {
            if let Some(tool) = self.resolve_tool_ref(tool_ref) {
                let (name, description) = if let Some(lang) = language {
                    if let Some(aliases) = builtin_aliases.get(id) {
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
        let tool_index = self.tool_index.read();
        let builtin_aliases = self.builtin_aliases.read();
        let mut prompt = String::from("Available tools:\n");
        let mut found_any = false;

        for id in tool_ids {
            if let Some(tool_ref) = tool_index.get(id)
                && let Some(tool) = self.resolve_tool_ref(tool_ref)
            {
                found_any = true;

                let (name, description) = if let Some(lang) = language {
                    if let Some(aliases) = builtin_aliases.get(id) {
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
        let tool_index = self.tool_index.read();
        let builtin_aliases = self.builtin_aliases.read();
        let mut prompt = String::from("Available tools:\n");
        let mut found_any = false;

        for id in tool_ids {
            let id = id.as_ref();
            if let Some(tool_ref) = tool_index.get(id)
                && let Some(tool) = self.resolve_tool_ref(tool_ref)
            {
                found_any = true;

                let (name, description) = if let Some(lang) = language {
                    if let Some(aliases) = builtin_aliases.get(id) {
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
    use crate::ToolResult;
    use async_trait::async_trait;
    use serde_json::Value;

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

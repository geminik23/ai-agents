use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use super::provider::{ProviderHealth, ToolProvider, ToolProviderError};
use super::types::ToolAliases;
use super::{Tool, ToolError, ToolInfo};

#[derive(Clone)]
enum ToolRef {
    Builtin(Arc<dyn Tool>),
    Provider {
        provider_id: String,
        tool: Arc<dyn Tool>,
    },
}

pub struct ToolRegistry {
    builtin_tools: RwLock<HashMap<String, Arc<dyn Tool>>>,

    providers: RwLock<HashMap<String, Arc<dyn ToolProvider>>>,

    tool_index: RwLock<HashMap<String, ToolRef>>,

    alias_index: RwLock<HashMap<String, String>>,

    builtin_aliases: RwLock<HashMap<String, ToolAliases>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            builtin_tools: RwLock::new(HashMap::new()),
            providers: RwLock::new(HashMap::new()),
            tool_index: RwLock::new(HashMap::new()),
            alias_index: RwLock::new(HashMap::new()),
            builtin_aliases: RwLock::new(HashMap::new()),
        }
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Result<(), ToolError> {
        let id = tool.id().to_string();

        let mut builtin_tools = self.builtin_tools.write();
        let mut tool_index = self.tool_index.write();

        if builtin_tools.contains_key(&id) || tool_index.contains_key(&id) {
            return Err(ToolError::Duplicate(id));
        }

        tool_index.insert(id.clone(), ToolRef::Builtin(tool.clone()));
        builtin_tools.insert(id, tool);
        Ok(())
    }

    pub fn get(&self, id_or_alias: &str) -> Option<Arc<dyn Tool>> {
        let tool_index = self.tool_index.read();

        if let Some(tool_ref) = tool_index.get(id_or_alias) {
            return self.resolve_tool_ref(tool_ref);
        }

        let alias_index = self.alias_index.read();
        let lower_alias = id_or_alias.to_lowercase();

        for (alias_key, tool_id) in alias_index.iter() {
            if alias_key.ends_with(&format!(":{}", lower_alias)) {
                if let Some(tool_ref) = tool_index.get(tool_id) {
                    return self.resolve_tool_ref(tool_ref);
                }
            }
        }

        None
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

        {
            let mut tool_index = self.tool_index.write();
            let mut alias_index = self.alias_index.write();

            for descriptor in &tools {
                if tool_index.contains_key(&descriptor.id) {
                    return Err(ToolError::Duplicate(descriptor.id.clone()));
                }

                if let Some(tool) = provider.get_tool(&descriptor.id).await {
                    tool_index.insert(
                        descriptor.id.clone(),
                        ToolRef::Provider {
                            provider_id: provider_id.clone(),
                            tool,
                        },
                    );

                    if let Some(ref aliases) = descriptor.aliases {
                        for (lang, name) in &aliases.names {
                            let key = format!("{}:{}", lang, name.to_lowercase());
                            alias_index.insert(key, descriptor.id.clone());
                        }
                    }
                }
            }
        }

        self.providers.write().insert(provider_id, provider);

        Ok(())
    }

    pub fn unregister_provider(&self, provider_id: &str) -> bool {
        let removed = self.providers.write().remove(provider_id);

        if removed.is_some() {
            let mut tool_index = self.tool_index.write();
            let mut alias_index = self.alias_index.write();

            let tools_to_remove: Vec<String> = tool_index
                .iter()
                .filter_map(|(id, tool_ref)| {
                    if let ToolRef::Provider {
                        provider_id: pid, ..
                    } = tool_ref
                    {
                        if pid == provider_id {
                            return Some(id.clone());
                        }
                    }
                    None
                })
                .collect();

            for tool_id in &tools_to_remove {
                tool_index.remove(tool_id);
            }

            alias_index.retain(|_, tool_id| !tools_to_remove.contains(tool_id));

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
                let key = format!("{}:{}", lang, name.to_lowercase());
                alias_index.insert(key, tool_id.to_string());
            }
        }

        self.builtin_aliases
            .write()
            .insert(tool_id.to_string(), aliases);
    }

    pub fn get_by_alias(&self, alias: &str, lang: &str) -> Option<Arc<dyn Tool>> {
        let key = format!("{}:{}", lang, alias.to_lowercase());
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
        let providers = self.providers.read();
        if let Some(provider) = providers.get(provider_id) {
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

                let mut tool_index = self.tool_index.write();
                let mut alias_index = self.alias_index.write();

                let old_tools: Vec<String> = tool_index
                    .iter()
                    .filter_map(|(id, tool_ref)| {
                        if let ToolRef::Provider {
                            provider_id: pid, ..
                        } = tool_ref
                        {
                            if pid == provider_id {
                                return Some(id.clone());
                            }
                        }
                        None
                    })
                    .collect();

                for tool_id in &old_tools {
                    tool_index.remove(tool_id);
                }
                alias_index.retain(|_, tool_id| !old_tools.contains(tool_id));

                for descriptor in &tools {
                    if let Some(tool) = provider.get_tool(&descriptor.id).await {
                        tool_index.insert(
                            descriptor.id.clone(),
                            ToolRef::Provider {
                                provider_id: provider_id.to_string(),
                                tool,
                            },
                        );

                        if let Some(ref aliases) = descriptor.aliases {
                            for (lang, name) in &aliases.names {
                                let key = format!("{}:{}", lang, name.to_lowercase());
                                alias_index.insert(key, descriptor.id.clone());
                            }
                        }
                    }
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
        self.generate_tools_prompt_with_lang(None)
    }

    pub fn generate_tools_prompt_with_lang(&self, language: Option<&str>) -> String {
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

        prompt.push_str(
            "\nWhen you need to use a tool, respond ONLY with valid JSON in this exact format:\n",
        );
        prompt.push_str("{\"tool\": \"tool_name\", \"arguments\": {...}}\n");
        prompt.push_str("\nWhen you receive a tool result, summarize it naturally for the user.\n");
        prompt.push_str("If no tool is needed, respond normally.");

        prompt
    }

    pub fn generate_filtered_prompt(&self, tool_ids: &[String]) -> String {
        self.generate_filtered_prompt_with_lang(tool_ids, None)
    }

    pub fn generate_filtered_prompt_with_lang(
        &self,
        tool_ids: &[String],
        language: Option<&str>,
    ) -> String {
        if tool_ids.is_empty() {
            return self.generate_tools_prompt_with_lang(language);
        }

        let tool_index = self.tool_index.read();
        let builtin_aliases = self.builtin_aliases.read();
        let mut prompt = String::from("Available tools:\n");
        let mut found_any = false;

        for id in tool_ids {
            if let Some(tool_ref) = tool_index.get(id) {
                if let Some(tool) = self.resolve_tool_ref(tool_ref) {
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
                    t         name, description, args_desc
                          ));
                }
            }
        }

        if !found_any {
            return String::new();
        }

        prompt.push_str(
            "\nWhen you need to use a tool, respond ONLY with valid JSON in this exact format:\n",
        );
        prompt.push_str("{\"tool\": \"tool_name\", \"arguments\": {...}}\n");
        prompt.push_str("\nWhen you receive a tool result, summarize it naturally for the user.\n");
        prompt.push_str("If no tool is needed, respond normally.");

        prompt
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
    use crate::tools::ToolResult;
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
        async fn execute(&self, _args: Value) -> ToolResult {
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

        let prompt_en = registry.generate_tools_prompt_with_lang(None);
        assert!(prompt_en.contains("Test"));

        let prompt_ko = registry.generate_tools_prompt_with_lang(Some("ko"));
        assert!(prompt_ko.contains("계산기"));
        assert!(prompt_ko.contains("수학 계산"));
    }
}

use std::collections::HashMap;
use std::sync::Arc;

use crate::routing::{LLMRole, ResolvedRoleLLM, RouterRolesConfig};
use ai_agents_core::{LLMError, LLMProvider};

#[derive(Clone)]
pub struct LLMRegistry {
    providers: HashMap<String, Arc<dyn LLMProvider>>,
    default_alias: String,
    router_alias: Option<String>,
    router_roles: Option<RouterRolesConfig>,
}

impl std::fmt::Debug for LLMRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LLMRegistry")
            .field("providers", &self.providers.keys().collect::<Vec<_>>())
            .field("default_alias", &self.default_alias)
            .field("router_alias", &self.router_alias)
            .finish()
    }
}

impl LLMRegistry {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            default_alias: "default".to_string(),
            router_alias: None,
            router_roles: None,
        }
    }

    pub fn register(&mut self, alias: impl Into<String>, provider: Arc<dyn LLMProvider>) {
        self.providers.insert(alias.into(), provider);
    }

    pub fn set_default(&mut self, alias: impl Into<String>) {
        self.default_alias = alias.into();
    }

    pub fn set_router(&mut self, alias: impl Into<String>) {
        self.router_roles = None;
        self.router_alias = Some(alias.into());
    }

    /// Installs agent-local hierarchy without registering or probing providers.
    pub fn set_router_roles(&mut self, config: RouterRolesConfig) {
        self.router_alias = None;
        self.router_roles = Some(config);
    }

    pub fn router_roles(&self) -> Option<&RouterRolesConfig> {
        self.router_roles.as_ref()
    }

    pub fn clear_router(&mut self) {
        self.router_alias = None;
        self.router_roles = None;
    }

    /// Resolves hierarchy exactly and leaves subsystem-specific legacy selection to its owner.
    pub fn resolve_role_override(
        &self,
        role: LLMRole,
        local_alias: Option<&str>,
    ) -> Result<Option<ResolvedRoleLLM>, LLMError> {
        let Some(config) = &self.router_roles else {
            return Ok(None);
        };
        let (alias, source) = config.select(role, local_alias, &self.default_alias);
        if alias.trim().is_empty() {
            return Err(LLMError::Config(format!(
                "Invalid {}: empty alias",
                role.as_path()
            )));
        }
        let provider = self.get(alias).map_err(|_| {
            LLMError::Config(format!(
                "Invalid llm.router.{}: alias '{}' is not registered",
                role.as_path(),
                alias
            ))
        })?;
        Ok(Some(ResolvedRoleLLM {
            role,
            alias: alias.to_string(),
            source,
            provider,
        }))
    }

    /// Validates all configured tree aliases without executing a provider.
    pub fn validate_router_roles(&self) -> Result<(), LLMError> {
        if let Some(config) = &self.router_roles {
            config.validate()?;
            for (path, alias) in config.configured_aliases() {
                self.get(alias).map_err(|_| {
                    LLMError::Config(format!("Invalid {path}: alias '{alias}' is not registered"))
                })?;
            }
            self.default()?;
        }
        Ok(())
    }

    /// Compares configuration and base handles before constructing irreversible consumers.
    pub fn same_bindings(&self, other: &Self) -> bool {
        self.default_alias == other.default_alias
            && self.router_alias == other.router_alias
            && self.router_roles == other.router_roles
            && self.providers.len() == other.providers.len()
            && self.providers.iter().all(|(alias, provider)| {
                other
                    .providers
                    .get(alias)
                    .is_some_and(|other| Arc::ptr_eq(provider, other))
            })
    }

    pub fn get(&self, alias: &str) -> Result<Arc<dyn LLMProvider>, LLMError> {
        self.providers
            .get(alias)
            .cloned()
            .ok_or_else(|| LLMError::Config(format!("LLM alias not found: {}", alias)))
    }

    pub fn default(&self) -> Result<Arc<dyn LLMProvider>, LLMError> {
        self.get(&self.default_alias)
    }

    pub fn router(&self) -> Result<Arc<dyn LLMProvider>, LLMError> {
        if let Some(config) = &self.router_roles {
            return self.get(config.default.as_deref().unwrap_or(&self.default_alias));
        }
        match &self.router_alias {
            Some(alias) => self.get(alias),
            None => self.default(),
        }
    }

    pub fn has(&self, alias: &str) -> bool {
        self.providers.contains_key(alias)
    }

    pub fn aliases(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }

    pub fn default_alias(&self) -> &str {
        &self.default_alias
    }

    pub fn router_alias(&self) -> Option<&str> {
        self.router_alias.as_deref()
    }

    pub fn map_providers<F>(&self, mut f: F) -> LLMRegistry
    where
        F: FnMut(&str, Arc<dyn LLMProvider>) -> Arc<dyn LLMProvider>,
    {
        let mut mapped = LLMRegistry::new();
        for (alias, provider) in &self.providers {
            mapped.register(alias.clone(), f(alias, provider.clone()));
        }
        mapped.set_default(self.default_alias.clone());
        if let Some(router) = &self.router_alias {
            mapped.set_router(router.clone());
        }
        mapped.router_roles = self.router_roles.clone();
        mapped
    }

    pub fn len(&self) -> usize {
        self.providers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }
}

impl Default for LLMRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_agents_core::{ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMFeature, LLMResponse};
    use async_trait::async_trait;

    struct MockProvider {
        name: String,
    }

    #[async_trait]
    impl LLMProvider for MockProvider {
        async fn complete(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> Result<LLMResponse, LLMError> {
            Ok(LLMResponse::new(
                format!("Response from {}", self.name),
                FinishReason::Stop,
            ))
        }

        async fn complete_stream(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> Result<
            Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>,
            LLMError,
        > {
            Err(LLMError::Other("Not implemented".into()))
        }

        fn provider_name(&self) -> &str {
            &self.name
        }

        fn supports(&self, _feature: LLMFeature) -> bool {
            false
        }
    }

    #[tokio::test]
    async fn hierarchy_clone_and_map_resolve_wrapped_handles_for_every_role() {
        use crate::routing::{LLMRole, LLMSelectionSource, RouterRolesConfig};
        let mut registry = LLMRegistry::new();
        let original = Arc::new(MockProvider {
            name: "original".into(),
        });
        registry.register("main", original.clone());
        registry.set_default("main");
        registry.set_router_roles(RouterRolesConfig::default());
        let clone = registry.clone();
        let wrapped = clone.map_providers(|_, _| {
            Arc::new(MockProvider {
                name: "wrapped".into(),
            })
        });
        for role in LLMRole::ALL {
            let resolved = wrapped.resolve_role_override(*role, None).unwrap().unwrap();
            assert_eq!(resolved.source, LLMSelectionSource::Default);
            assert_eq!(
                resolved.provider.complete(&[], None).await.unwrap().content,
                "Response from wrapped"
            );
            assert!(Arc::ptr_eq(
                &resolved.provider,
                &wrapped.get("main").unwrap()
            ));
        }
        assert_eq!(registry.default().unwrap().provider_name(), "original");
    }

    #[test]
    fn test_registry_basic() {
        let mut registry = LLMRegistry::new();
        let provider = Arc::new(MockProvider {
            name: "test".into(),
        });

        registry.register("default", provider);
        assert!(registry.has("default"));
        assert!(!registry.has("unknown"));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_registry_default_and_router() {
        let mut registry = LLMRegistry::new();
        registry.register(
            "main",
            Arc::new(MockProvider {
                name: "main".into(),
            }),
        );
        registry.register(
            "router",
            Arc::new(MockProvider {
                name: "router".into(),
            }),
        );

        registry.set_default("main");
        registry.set_router("router");

        assert!(registry.default().is_ok());
        assert!(registry.router().is_ok());
        assert_eq!(registry.default().unwrap().provider_name(), "main");
        assert_eq!(registry.router().unwrap().provider_name(), "router");
    }

    #[test]
    fn test_registry_router_fallback() {
        let mut registry = LLMRegistry::new();
        registry.register(
            "default",
            Arc::new(MockProvider {
                name: "default".into(),
            }),
        );

        let router = registry.router().unwrap();
        assert_eq!(router.provider_name(), "default");
    }

    #[test]
    fn test_registry_missing_alias() {
        let registry = LLMRegistry::new();
        assert!(registry.get("nonexistent").is_err());
    }
}

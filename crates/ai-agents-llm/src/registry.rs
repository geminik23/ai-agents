use std::collections::HashMap;
use std::sync::Arc;

use ai_agents_core::{LLMError, LLMProvider};

#[derive(Clone)]
pub struct LLMRegistry {
    providers: HashMap<String, Arc<dyn LLMProvider>>,
    default_alias: String,
    router_alias: Option<String>,
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
        }
    }

    pub fn register(&mut self, alias: impl Into<String>, provider: Arc<dyn LLMProvider>) {
        self.providers.insert(alias.into(), provider);
    }

    pub fn set_default(&mut self, alias: impl Into<String>) {
        self.default_alias = alias.into();
    }

    pub fn set_router(&mut self, alias: impl Into<String>) {
        self.router_alias = Some(alias.into());
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

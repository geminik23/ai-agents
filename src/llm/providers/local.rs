use crate::llm::{LLMConfig, LLMError};
use serde::{Deserialize, Serialize};

/// Trait for implementing custom local model providers
///
/// Implementations must be `Send + Sync` for async compatibility.
pub trait LocalModelProvider: Send + Sync {
    /// Load a model from the specified path
    fn load_model(&mut self, path: &str) -> Result<(), LLMError>;

    /// Perform blocking inference with the loaded model
    fn infer(&self, prompt: &str, config: &LLMConfig) -> Result<String, LLMError>;

    /// Perform streaming inference with the loaded model
    fn infer_stream(
        &self,
        prompt: &str,
        config: &LLMConfig,
    ) -> Result<Box<dyn futures::Stream<Item = Result<String, LLMError>> + Unpin + Send>, LLMError>;

    /// Get information about the loaded model
    fn model_info(&self) -> ModelInfo;

    fn unload(&mut self) -> Result<(), LLMError> {
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        true
    }
}

/// Metadata about a loaded model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_name: String,
    pub architecture: String,
    pub parameters: Option<u64>,
    pub quantization: Option<String>,
    pub context_size: Option<usize>,
}

impl ModelInfo {
    pub fn new(model_name: impl Into<String>, architecture: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            architecture: architecture.into(),
            parameters: None,
            quantization: None,
            context_size: None,
        }
    }

    pub fn with_parameters(mut self, parameters: u64) -> Self {
        self.parameters = Some(parameters);
        self
    }

    pub fn with_quantization(mut self, quantization: impl Into<String>) -> Self {
        self.quantization = Some(quantization.into());
        self
    }

    pub fn with_context_size(mut self, context_size: usize) -> Self {
        self.context_size = Some(context_size);
        self
    }
}

/// Adapter to wrap a LocalModelProvider as an LLMProvider
pub struct LocalModelAdapter<T: LocalModelProvider> {
    provider: T,
}

impl<T: LocalModelProvider> LocalModelAdapter<T> {
    pub fn new(provider: T) -> Self {
        Self { provider }
    }

    pub fn provider(&self) -> &T {
        &self.provider
    }

    pub fn provider_mut(&mut self) -> &mut T {
        &mut self.provider
    }

    pub fn into_inner(self) -> T {
        self.provider
    }
}

use crate::llm::{ChatMessage, FinishReason, LLMChunk, LLMFeature, LLMProvider, Role, TokenUsage};
use async_trait::async_trait;
use futures::stream::StreamExt;

#[async_trait]
impl<T: LocalModelProvider> LLMProvider for LocalModelAdapter<T> {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<crate::llm::LLMResponse, LLMError> {
        let prompt = messages_to_prompt(messages);
        let cfg = config.cloned().unwrap_or_default();
        let content = tokio::task::block_in_place(|| self.provider.infer(&prompt, &cfg))?;

        let prompt_tokens = estimate_tokens(&prompt);
        let completion_tokens = estimate_tokens(&content);

        Ok(crate::llm::LLMResponse {
            content,
            finish_reason: FinishReason::Stop,
            usage: Some(TokenUsage::new(prompt_tokens, completion_tokens)),
            model: Some(self.provider.model_info().model_name.clone()),
            metadata: std::collections::HashMap::new(),
        })
    }

    async fn complete_stream(
        &self,
        messages: &[crate::llm::ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError>
    {
        let prompt = messages_to_prompt(messages);
        let cfg = config.cloned().unwrap_or_default();
        let stream = self.provider.infer_stream(&prompt, &cfg)?;
        let mapped_stream = stream.map(|result| result.map(|delta| LLMChunk::new(delta, false)));
        Ok(Box::new(mapped_stream))
    }

    fn provider_name(&self) -> &str {
        "local-model"
    }

    fn supports(&self, feature: LLMFeature) -> bool {
        match feature {
            LLMFeature::Streaming => true,
            LLMFeature::SystemMessages => true,
            _ => false,
        }
    }
}

fn messages_to_prompt(messages: &[crate::llm::ChatMessage]) -> String {
    let mut prompt = String::new();

    for msg in messages {
        let role_prefix = match msg.role {
            Role::System => "System: ",
            Role::User => "User: ",
            Role::Assistant => "Assistant: ",
            Role::Function => "Function: ",
            Role::Tool => "Tool: ",
        };

        prompt.push_str(role_prefix);
        prompt.push_str(&msg.content);
        prompt.push_str("\n\n");
    }

    // Add final assistant prompt to trigger response
    if !messages.is_empty() && messages.last().unwrap().role != Role::Assistant {
        prompt.push_str("Assistant: ");
    }

    prompt
}

fn estimate_tokens(text: &str) -> u32 {
    (text.len() / 4) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockLocalModel {
        loaded: bool,
        model_path: Option<String>,
    }

    impl MockLocalModel {
        fn new() -> Self {
            Self {
                loaded: false,
                model_path: None,
            }
        }
    }

    impl LocalModelProvider for MockLocalModel {
        fn load_model(&mut self, path: &str) -> Result<(), LLMError> {
            self.model_path = Some(path.to_string());
            self.loaded = true;
            Ok(())
        }

        fn infer(&self, prompt: &str, _config: &LLMConfig) -> Result<String, LLMError> {
            if !self.loaded {
                return Err(LLMError::Config("Model not loaded".to_string()));
            }
            Ok(format!("Response to: {}", prompt))
        }

        fn infer_stream(
            &self,
            _prompt: &str,
            _config: &LLMConfig,
        ) -> Result<
            Box<dyn futures::Stream<Item = Result<String, LLMError>> + Unpin + Send>,
            LLMError,
        > {
            use futures::stream;
            let chunks = vec![
                Ok("Hello ".to_string()),
                Ok("world".to_string()),
                Ok("!".to_string()),
            ];
            Ok(Box::new(stream::iter(chunks)))
        }

        fn model_info(&self) -> ModelInfo {
            ModelInfo::new("mock-model", "mock")
                .with_parameters(7_000_000_000)
                .with_quantization("Q4_K_M")
                .with_context_size(4096)
        }

        fn is_loaded(&self) -> bool {
            self.loaded
        }

        fn unload(&mut self) -> Result<(), LLMError> {
            self.loaded = false;
            self.model_path = None;
            Ok(())
        }
    }

    #[test]
    fn test_load_and_unload() {
        let mut model = MockLocalModel::new();
        assert!(!model.is_loaded());

        model.load_model("./models/test_model").unwrap();
        assert!(model.is_loaded());

        model.unload().unwrap();
        assert!(!model.is_loaded());
    }

    #[test]
    fn test_infer_without_loading() {
        let model = MockLocalModel::new();
        let result = model.infer("test", &LLMConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_infer() {
        let mut model = MockLocalModel::new();
        model.load_model("./models/test_model").unwrap();

        let result = model.infer("Hello", &LLMConfig::default()).unwrap();
        assert!(result.contains("Response to: Hello"));
    }

    #[test]
    fn test_model_info() {
        let model = MockLocalModel::new();
        let info = model.model_info();
        assert_eq!(info.model_name, "mock-model");
        assert_eq!(info.parameters, Some(7_000_000_000));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_adapter_complete() {
        let mut model = MockLocalModel::new();
        model.load_model("./models/test_model").unwrap();
        let adapter = LocalModelAdapter::new(model);

        let messages = vec![ChatMessage::user("Hello")];
        let response = adapter.complete(&messages, None).await.unwrap();
        assert!(!response.content.is_empty());
    }

    #[test]
    fn test_messages_to_prompt() {
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
        ];
        let prompt = messages_to_prompt(&messages);
        assert!(prompt.contains("System: You are helpful"));
        assert!(prompt.contains("User: Hello"));
        assert!(prompt.contains("Assistant: "));
    }

    #[tokio::test]
    async fn test_infer_stream() {
        use futures::StreamExt;

        let mut model = MockLocalModel::new();
        model.load_model("./models/test_model").unwrap();

        let config = LLMConfig::default();
        let stream = model.infer_stream("Hello", &config).unwrap();

        let chunks: Vec<_> = stream.collect().await;
        assert_eq!(chunks.len(), 3);
        assert!(chunks.iter().all(|c| c.is_ok()));
    }
}

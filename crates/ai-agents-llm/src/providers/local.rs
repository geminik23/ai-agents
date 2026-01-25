use ai_agents_core::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
    Role, TokenUsage,
};
use async_trait::async_trait;
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};

/// Trait for implementing custom local model providers
///
/// Implementations must be `Send + Sync` for async compatibility.
pub trait LocalModelProvider: Send + Sync {
    /// Load a model from the specified path
    fn load_model(&mut self, path: &str) -> Result<(), LLMError>;

    /// Run inference on the given prompt
    fn infer(&self, prompt: &str, config: &LLMConfig) -> Result<String, LLMError>;

    /// Run streaming inference
    fn infer_stream(
        &self,
        prompt: &str,
        config: &LLMConfig,
    ) -> Result<Box<dyn futures::Stream<Item = Result<String, LLMError>> + Unpin + Send>, LLMError>;

    /// Get model information
    fn model_info(&self) -> ModelInfo;

    /// Check if the model is loaded
    fn is_loaded(&self) -> bool;

    /// Unload the model from memory
    fn unload(&mut self) -> Result<(), LLMError>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_name: String,
    pub model_type: String,
    pub parameters: Option<u64>,
    pub quantization: Option<String>,
    pub context_length: Option<usize>,
}

impl ModelInfo {
    pub fn new(model_name: impl Into<String>, model_type: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            model_type: model_type.into(),
            parameters: None,
            quantization: None,
            context_length: None,
        }
    }

    pub fn with_parameters(mut self, params: u64) -> Self {
        self.parameters = Some(params);
        self
    }

    pub fn with_quantization(mut self, quant: impl Into<String>) -> Self {
        self.quantization = Some(quant.into());
        self
    }

    pub fn with_context_length(mut self, length: usize) -> Self {
        self.context_length = Some(length);
        self
    }
}

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

#[async_trait]
impl<T: LocalModelProvider> LLMProvider for LocalModelAdapter<T> {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<LLMResponse, LLMError> {
        let prompt = messages_to_prompt(messages);
        let cfg = config.cloned().unwrap_or_default();
        let content = tokio::task::block_in_place(|| self.provider.infer(&prompt, &cfg))?;

        let prompt_tokens = estimate_tokens(&prompt);
        let completion_tokens = estimate_tokens(&content);

        Ok(LLMResponse {
            content,
            finish_reason: FinishReason::Stop,
            usage: Some(TokenUsage::new(prompt_tokens, completion_tokens)),
            model: Some(self.provider.model_info().model_name.clone()),
            metadata: std::collections::HashMap::new(),
        })
    }

    async fn complete_stream(
        &self,
        messages: &[ChatMessage],
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

fn messages_to_prompt(messages: &[ChatMessage]) -> String {
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
        prompt.push('\n');
    }

    prompt.push_str("Assistant: ");
    prompt
}

fn estimate_tokens(text: &str) -> u32 {
    (text.len() / 4) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_info_builder() {
        let info = ModelInfo::new("llama-2-7b", "llama")
            .with_parameters(7_000_000_000)
            .with_quantization("Q4_K_M")
            .with_context_length(4096);

        assert_eq!(info.model_name, "llama-2-7b");
        assert_eq!(info.model_type, "llama");
        assert_eq!(info.parameters, Some(7_000_000_000));
        assert_eq!(info.quantization, Some("Q4_K_M".to_string()));
        assert_eq!(info.context_length, Some(4096));
    }

    #[test]
    fn test_messages_to_prompt() {
        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: "You are helpful.".to_string(),
                name: None,
                timestamp: None,
            },
            ChatMessage {
                role: Role::User,
                content: "Hello".to_string(),
                name: None,
                timestamp: None,
            },
        ];

        let prompt = messages_to_prompt(&messages);
        assert!(prompt.contains("System: You are helpful."));
        assert!(prompt.contains("User: Hello"));
        assert!(prompt.ends_with("Assistant: "));
    }

    #[test]
    fn test_estimate_tokens() {
        let text = "This is a test string with some words";
        let tokens = estimate_tokens(text);
        assert!(tokens > 0);
        assert_eq!(tokens, (text.len() / 4) as u32);
    }
}

//! LLM provider traits

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

use crate::message::ChatMessage;
use crate::types::{LLMChunk, LLMConfig, LLMFeature, LLMResponse, LLMToolRequest, ToolChoice};

/// Core LLM provider trait.
///
/// Implement this to integrate a custom LLM backend. Most users can use
/// `UnifiedLLMProvider` which supports OpenAI, Anthropic, and other providers
/// out of the box.
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Send messages and get a complete response.
    async fn complete(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<LLMResponse, LLMError>;

    /// Send messages with provider-native tool definitions.
    async fn complete_with_tools(
        &self,
        _messages: &[ChatMessage],
        _config: Option<&LLMConfig>,
        _request: &LLMToolRequest,
    ) -> Result<LLMResponse, LLMError> {
        Err(LLMError::Other(format!(
            "provider '{}' does not support native tool completion",
            self.provider_name()
        )))
    }

    /// Returns a provider-level tool choice override when configured.
    fn configured_tool_choice(&self) -> Option<ToolChoice> {
        None
    }

    /// Reports whether this provider can enforce a native tool choice.
    fn supports_tool_choice(&self, _choice: &ToolChoice) -> bool {
        false
    }

    /// Send messages and get a streaming response.
    async fn complete_stream(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError>;

    /// Provider identifier (e.g. `"openai"`, `"anthropic"`).
    fn provider_name(&self) -> &str;

    /// Check if this provider supports a given feature.
    fn supports(&self, feature: LLMFeature) -> bool;
}

/// Higher-level LLM capabilities for agent operations
#[async_trait]
pub trait LLMCapability: Send + Sync {
    async fn select_tool(
        &self,
        context: &TaskContext,
        user_input: &str,
    ) -> Result<ToolSelection, LLMError>;

    async fn generate_tool_args(
        &self,
        tool_id: &str,
        user_input: &str,
        schema: &serde_json::Value,
    ) -> Result<serde_json::Value, LLMError>;

    async fn evaluate_yesno(
        &self,
        question: &str,
        context: &TaskContext,
    ) -> Result<(bool, String), LLMError>;

    async fn classify(&self, input: &str, categories: &[String])
    -> Result<(String, f32), LLMError>;

    async fn process_task(
        &self,
        context: &TaskContext,
        system_prompt: &str,
    ) -> Result<LLMResponse, LLMError>;
}

/// Task context for LLM operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContext {
    pub current_state: Option<String>,
    pub available_tools: Vec<String>,
    pub memory_slots: HashMap<String, serde_json::Value>,
    pub recent_messages: Vec<ChatMessage>,
}

/// Tool selection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSelection {
    pub tool_id: String,
    pub confidence: f32,
    pub reasoning: Option<String>,
}

/// LLM error types
#[derive(Debug, Error)]
pub enum LLMError {
    #[error("API error: {message}")]
    API {
        message: String,
        status: Option<u16>,
    },

    #[error("Network error: {0}")]
    Network(String),

    #[error("Rate limit exceeded: {retry_after:?}")]
    RateLimit {
        retry_after: Option<std::time::Duration>,
    },

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Content filtered: {0}")]
    ContentFiltered(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Other error: {0}")]
    Other(String),
}

impl From<serde_json::Error> for LLMError {
    fn from(err: serde_json::Error) -> Self {
        LLMError::Serialization(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FinishReason;

    struct LegacyProvider;

    #[async_trait]
    impl LLMProvider for LegacyProvider {
        async fn complete(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> Result<LLMResponse, LLMError> {
            Ok(LLMResponse::new("legacy", FinishReason::Stop))
        }

        async fn complete_stream(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> Result<
            Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>,
            LLMError,
        > {
            Ok(Box::new(futures::stream::empty()))
        }

        fn provider_name(&self) -> &str {
            "legacy"
        }

        fn supports(&self, _feature: LLMFeature) -> bool {
            false
        }
    }

    #[test]
    fn additive_tool_methods_preserve_legacy_implementations() {
        let provider = LegacyProvider;
        let request = LLMToolRequest {
            tools: Vec::new(),
            choice: ToolChoice::Auto,
        };

        assert!(provider.configured_tool_choice().is_none());
        assert!(!provider.supports_tool_choice(&ToolChoice::Auto));
        let error = futures::executor::block_on(provider.complete_with_tools(&[], None, &request))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not support native tool completion")
        );
    }
}

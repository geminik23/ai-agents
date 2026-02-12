use ai_agents_core::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
    Role, TokenUsage,
};
use async_trait::async_trait;
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Provider type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    /// OpenAI (GPT models)
    OpenAI,
    /// Anthropic (Claude models)
    Anthropic,
    /// Ollama (local models)
    Ollama,
    /// DeepSeek
    DeepSeek,
    /// xAI (Grok)
    XAI,
    /// Phind
    Phind,
    /// Groq
    Groq,
    /// Google (Gemini)
    Google,
    /// Cohere
    Cohere,
    /// Mistral
    Mistral,
}

impl ProviderType {
    pub fn api_key_env_var(&self) -> Option<&'static str> {
        match self {
            Self::OpenAI => Some("OPENAI_API_KEY"),
            Self::Anthropic => Some("ANTHROPIC_API_KEY"),
            Self::DeepSeek => Some("DEEPSEEK_API_KEY"),
            Self::XAI => Some("XAI_API_KEY"),
            Self::Phind => Some("PHIND_API_KEY"),
            Self::Groq => Some("GROQ_API_KEY"),
            Self::Google => Some("GOOGLE_API_KEY"),
            Self::Cohere => Some("COHERE_API_KEY"),
            Self::Mistral => Some("MISTRAL_API_KEY"),
            Self::Ollama => None, // Ollama doesn't need an API key
        }
    }

    pub fn default_base_url(&self) -> Option<&'static str> {
        match self {
            Self::Ollama => Some("http://localhost:11434"),
            _ => None, // Most providers use their default URLs
        }
    }

    fn to_llm_backend(&self) -> llm::builder::LLMBackend {
        match self {
            Self::OpenAI => llm::builder::LLMBackend::OpenAI,
            Self::Anthropic => llm::builder::LLMBackend::Anthropic,
            Self::Ollama => llm::builder::LLMBackend::Ollama,
            Self::DeepSeek => llm::builder::LLMBackend::DeepSeek,
            Self::XAI => llm::builder::LLMBackend::XAI,
            Self::Phind => llm::builder::LLMBackend::Phind,
            Self::Google => llm::builder::LLMBackend::Google,
            Self::Groq => llm::builder::LLMBackend::Groq,
            Self::Cohere => llm::builder::LLMBackend::Cohere,
            Self::Mistral => llm::builder::LLMBackend::Mistral,
        }
    }
}

impl FromStr for ProviderType {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(Self::OpenAI),
            "anthropic" => Ok(Self::Anthropic),
            "ollama" => Ok(Self::Ollama),
            "deepseek" => Ok(Self::DeepSeek),
            "xai" => Ok(Self::XAI),
            "phind" => Ok(Self::Phind),
            "groq" => Ok(Self::Groq),
            "google" => Ok(Self::Google),
            "cohere" => Ok(Self::Cohere),
            "mistral" => Ok(Self::Mistral),
            _ => Err("unknown provider type"),
        }
    }
}

#[derive(Debug)]
pub struct UnifiedLLMProvider {
    provider_type: ProviderType,
    model: String,
    api_key: Option<String>,
    base_url: Option<String>,
}

impl UnifiedLLMProvider {
    pub fn new(
        provider_type: ProviderType,
        model: String,
        api_key: Option<String>,
        base_url: Option<String>,
    ) -> Result<Self, LLMError> {
        let actual_api_key = if let Some(key) = api_key.clone() {
            key
        } else if let Some(env_var) = provider_type.api_key_env_var() {
            std::env::var(env_var).map_err(|_| {
                LLMError::Config(format!(
                    "API key not found in environment variable {}",
                    env_var
                ))
            })?
        } else {
            String::new() // Ollama doesn't need a key
        };

        let actual_base_url = base_url
            .clone()
            .or_else(|| provider_type.default_base_url().map(|s| s.to_string()));

        Ok(Self {
            provider_type,
            model,
            api_key: Some(actual_api_key),
            base_url: actual_base_url,
        })
    }

    pub fn from_env(
        provider_type: ProviderType,
        model: impl Into<String>,
    ) -> Result<Self, LLMError> {
        Self::new(provider_type, model.into(), None, None)
    }

    pub fn provider_type(&self) -> ProviderType {
        self.provider_type
    }

    pub fn model_name(&self) -> &str {
        &self.model
    }

    pub fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
    }

    fn convert_message(&self, msg: &ChatMessage) -> llm::chat::ChatMessage {
        match msg.role {
            Role::System | Role::User => {
                llm::chat::ChatMessage::user().content(&msg.content).build()
            }
            Role::Assistant => llm::chat::ChatMessage::assistant()
                .content(&msg.content)
                .build(),
            Role::Function => llm::chat::ChatMessage::user()
                .content(format!("Function: {}", msg.content))
                .build(),
            Role::Tool => llm::chat::ChatMessage::user()
                .content(format!("Tool: {}", msg.content))
                .build(),
        }
    }

    // LEGACY: kept for potential use by future provider implementations
    #[allow(dead_code)]
    fn map_finish_reason(&self, reason: &str) -> FinishReason {
        match reason {
            "stop" | "end_turn" => FinishReason::Stop,
            "length" | "max_tokens" => FinishReason::Length,
            "tool_calls" | "function_call" => FinishReason::ToolCall,
            "content_filter" => FinishReason::ContentFilter,
            _ => FinishReason::Other,
        }
    }

    fn build_llm(&self, config: Option<&LLMConfig>) -> Result<Box<dyn llm::LLMProvider>, LLMError> {
        let mut builder = llm::builder::LLMBuilder::new()
            .backend(self.provider_type.to_llm_backend())
            .model(&self.model);

        if let Some(ref key) = self.api_key {
            if !key.is_empty() {
                builder = builder.api_key(key);
            }
        }

        if let Some(ref url) = self.base_url {
            builder = builder.base_url(url);
        }

        if let Some(cfg) = config {
            if let Some(temp) = cfg.temperature {
                builder = builder.temperature(temp);
            }
            if let Some(max_tok) = cfg.max_tokens {
                builder = builder.max_tokens(max_tok);
            }
            if let Some(top_p) = cfg.top_p {
                builder = builder.top_p(top_p);
            }
        }

        builder
            .build()
            .map_err(|e| LLMError::Config(format!("Failed to build LLM: {}", e)))
    }
}

#[async_trait]
impl LLMProvider for UnifiedLLMProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<LLMResponse, LLMError> {
        let llm_messages: Vec<llm::chat::ChatMessage> =
            messages.iter().map(|m| self.convert_message(m)).collect();

        let llm = self.build_llm(config)?;

        let response = llm.chat(&llm_messages).await.map_err(|e| LLMError::API {
            message: format!("LLM provider error: {}", e),
            status: None,
        })?;

        let content = response.text().unwrap_or_else(|| "".to_string());

        let usage = response.usage().map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(LLMResponse {
            content,
            finish_reason: FinishReason::Stop,
            usage,
            model: Some(self.model.clone()),
            metadata: std::collections::HashMap::new(),
        })
    }

    async fn complete_stream(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError>
    {
        let llm_messages: Vec<llm::chat::ChatMessage> =
            messages.iter().map(|m| self.convert_message(m)).collect();

        let llm = self.build_llm(config)?;

        let stream = llm
            .chat_stream(&llm_messages)
            .await
            .map_err(|e| LLMError::API {
                message: format!("LLM provider error: {}", e),
                status: None,
            })?;

        let converted_stream = stream.map(|result| {
            result
                .map(|token| LLMChunk::new(token, false))
                .map_err(|e| LLMError::API {
                    message: format!("Stream error: {}", e),
                    status: None,
                })
        });

        Ok(Box::new(Box::pin(converted_stream)))
    }

    fn provider_name(&self) -> &str {
        match self.provider_type {
            ProviderType::OpenAI => "openai",
            ProviderType::Anthropic => "anthropic",
            ProviderType::Ollama => "ollama",
            ProviderType::DeepSeek => "deepseek",
            ProviderType::XAI => "xai",
            ProviderType::Phind => "phind",
            ProviderType::Groq => "groq",
            ProviderType::Google => "google",
            ProviderType::Cohere => "cohere",
            ProviderType::Mistral => "mistral",
        }
    }

    fn supports(&self, feature: LLMFeature) -> bool {
        match feature {
            LLMFeature::Streaming => true,
            LLMFeature::SystemMessages => true,
            LLMFeature::FunctionCalling => matches!(
                self.provider_type,
                ProviderType::OpenAI | ProviderType::Anthropic | ProviderType::Google
            ),
            LLMFeature::Vision => matches!(
                self.provider_type,
                ProviderType::OpenAI | ProviderType::Anthropic | ProviderType::Google
            ),
            LLMFeature::JsonMode => matches!(
                self.provider_type,
                ProviderType::OpenAI | ProviderType::Google
            ),
            _ => false,
        }
    }
}

pub struct ProviderBuilder {
    provider_type: Option<ProviderType>,
    model: Option<String>,
    api_key: Option<String>,
    api_key_env: Option<String>,
    base_url: Option<String>,
}

impl ProviderBuilder {
    pub fn new() -> Self {
        Self {
            provider_type: None,
            model: None,
            api_key: None,
            api_key_env: None,
            base_url: None,
        }
    }

    pub fn provider(mut self, provider_type: ProviderType) -> Self {
        self.provider_type = Some(provider_type);
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn api_key_env(mut self, env_var: impl Into<String>) -> Self {
        self.api_key_env = Some(env_var.into());
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn build(self) -> Result<UnifiedLLMProvider, LLMError> {
        let provider_type = self
            .provider_type
            .ok_or_else(|| LLMError::Config("Provider type not set".to_string()))?;

        let model = self
            .model
            .ok_or_else(|| LLMError::Config("Model not set".to_string()))?;

        let api_key = if let Some(key) = self.api_key {
            Some(key)
        } else if let Some(env_var) = self.api_key_env {
            Some(std::env::var(env_var).map_err(|_| {
                LLMError::Config("API key environment variable not found".to_string())
            })?)
        } else {
            None
        };

        UnifiedLLMProvider::new(provider_type, model, api_key, self.base_url)
    }
}

impl Default for ProviderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder() {
        let provider = ProviderBuilder::new()
            .provider(ProviderType::OpenAI)
            .model("gpt-4")
            .api_key("XXXXXXXXXX")
            .build()
            .unwrap();

        assert_eq!(provider.provider_name(), "openai");
        assert_eq!(provider.model_name(), "gpt-4");
    }

    #[test]
    fn test_builder_missing_fields() {
        let result = ProviderBuilder::new().model("gpt-4").build();

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Provider type not set")
        );
    }
}

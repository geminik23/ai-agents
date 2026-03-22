use ai_agents_core::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
    Role, TokenUsage,
};
use async_trait::async_trait;
use futures::stream::StreamExt;
use llm::chat::ReasoningEffort;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
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
    /// Any OpenAI-compatible server (LM Studio, vLLM, TGI, LocalAI, etc.)
    #[serde(rename = "openai-compatible")]
    OpenAICompatible,
    /// OpenRouter (multi-provider gateway)
    #[serde(rename = "openrouter")]
    OpenRouter,
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
            Self::OpenRouter => Some("OPENROUTER_API_KEY"),
            Self::Ollama => None,           // Ollama doesn't need an API key
            Self::OpenAICompatible => None, // User specifies via api_key_env or it's optional
        }
    }

    pub fn default_base_url(&self) -> Option<&'static str> {
        match self {
            Self::Ollama => Some("http://localhost:11434"),
            Self::OpenAICompatible => None, // MUST be provided by user via base_url
            Self::OpenRouter => None,       // llm crate handles default
            _ => None,                      // Most providers use their default URLs
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
            Self::OpenAICompatible => llm::builder::LLMBackend::OpenAI, // Reuse OpenAI's OpenAI-compatible implementation
            Self::OpenRouter => llm::builder::LLMBackend::OpenRouter,
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
            "openai-compatible" | "openai_compatible" => Ok(Self::OpenAICompatible),
            "openrouter" => Ok(Self::OpenRouter),
            _ => Err("unknown provider type"),
        }
    }
}

/// Cached LLM client, storing the built provider and the config hash used to build it.
struct CachedClient {
    llm: Box<dyn llm::LLMProvider>,
    config_hash: u64,
}

pub struct UnifiedLLMProvider {
    provider_type: ProviderType,
    model: String,
    api_key: Option<String>,
    base_url: Option<String>,
    default_config: LLMConfig,
    client: std::sync::Arc<tokio::sync::Mutex<Option<CachedClient>>>,
}

impl std::fmt::Debug for UnifiedLLMProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnifiedLLMProvider")
            .field("provider_type", &self.provider_type)
            .field("model", &self.model)
            .field("api_key", &self.api_key.as_ref().map(|_| "***"))
            .field("base_url", &self.base_url)
            .field("default_config", &self.default_config)
            .field("client", &"<cached>")
            .finish()
    }
}

/// Compute a hash over the config fields that affect the LLM builder, plus the system prompt.
fn compute_config_hash(config: &LLMConfig, system_prompt: Option<&str>) -> u64 {
    let mut hasher = DefaultHasher::new();
    // Hash the config fields that affect the builder
    if let Some(t) = config.temperature {
        t.to_bits().hash(&mut hasher);
    }
    if let Some(m) = config.max_tokens {
        m.hash(&mut hasher);
    }
    if let Some(p) = config.top_p {
        p.to_bits().hash(&mut hasher);
    }
    if let Some(k) = config.top_k {
        k.hash(&mut hasher);
    }
    if let Some(ref re) = config
        .extra
        .get("reasoning_effort")
        .and_then(|v| v.as_str())
    {
        re.hash(&mut hasher);
    }
    if let Some(sp) = system_prompt {
        sp.hash(&mut hasher);
    }
    hasher.finish()
}

/// Extract system messages from a message slice, returning the combined system prompt
/// (if any) and the remaining non-system messages.
fn extract_system_and_messages(messages: &[ChatMessage]) -> (Option<String>, Vec<&ChatMessage>) {
    let mut system_parts: Vec<&str> = Vec::new();
    let mut non_system: Vec<&ChatMessage> = Vec::new();

    for msg in messages {
        if msg.role == Role::System {
            system_parts.push(&msg.content);
        } else {
            non_system.push(msg);
        }
    }

    let system_prompt = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n"))
    };

    (system_prompt, non_system)
}

impl UnifiedLLMProvider {
    pub fn new(
        provider_type: ProviderType,
        model: String,
        api_key: Option<String>,
        base_url: Option<String>,
    ) -> Result<Self, LLMError> {
        Self::from_spec_config(
            provider_type,
            &model,
            api_key,
            base_url,
            LLMConfig::default(),
        )
    }

    /// Create a new UnifiedLLMProvider with explicit config defaults.
    pub fn from_spec_config(
        provider_type: ProviderType,
        model: &str,
        api_key: Option<String>,
        base_url: Option<String>,
        default_config: LLMConfig,
    ) -> Result<Self, LLMError> {
        let actual_api_key = if let Some(key) = api_key {
            key
        } else if let Some(env_var) = provider_type.api_key_env_var() {
            std::env::var(env_var).map_err(|_| {
                LLMError::Config(format!(
                    "API key not found in environment variable {}",
                    env_var
                ))
            })?
        } else {
            String::new() // Ollama and OpenAICompatible don't require a key
        };

        let actual_base_url =
            base_url.or_else(|| provider_type.default_base_url().map(|s| s.to_string()));

        // OpenAICompatible requires a base_url — there's no default server to connect to
        if provider_type == ProviderType::OpenAICompatible && actual_base_url.is_none() {
            return Err(LLMError::Config(
                "provider 'openai-compatible' requires a base_url".to_string(),
            ));
        }

        Ok(Self {
            provider_type,
            model: model.to_string(),
            api_key: Some(actual_api_key),
            base_url: actual_base_url,
            default_config,
            client: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
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

    /// Convert a non-system ChatMessage to llm::chat::ChatMessage.
    /// System messages are handled separately via the builder's `.system()` method.
    fn convert_message(&self, msg: &ChatMessage) -> llm::chat::ChatMessage {
        match msg.role {
            Role::User => llm::chat::ChatMessage::user().content(&msg.content).build(),
            Role::Assistant => llm::chat::ChatMessage::assistant()
                .content(&msg.content)
                .build(),
            Role::Function => {
                let name = msg.name.as_deref().unwrap_or("tool");
                llm::chat::ChatMessage::user()
                    .content(format!("[{} result]: {}", name, msg.content))
                    .build()
            }
            Role::Tool => {
                let name = msg.name.as_deref().unwrap_or("tool");
                llm::chat::ChatMessage::user()
                    .content(format!("[{} result]: {}", name, msg.content))
                    .build()
            }
            Role::System => {
                // System messages should have been extracted before calling convert_message.
                // If one slips through, convert it as a user message to avoid losing content.
                tracing::warn!(
                    "System message passed to convert_message; should be handled via builder.system()"
                );
                llm::chat::ChatMessage::user().content(&msg.content).build()
            }
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

    /// Internal builder: creates a new `llm::LLMProvider` from config and optional system prompt.
    fn build_llm_with_system(
        &self,
        config: &LLMConfig,
        system_prompt: Option<&str>,
    ) -> Result<Box<dyn llm::LLMProvider>, LLMError> {
        let mut builder = llm::builder::LLMBuilder::new()
            .backend(self.provider_type.to_llm_backend())
            .model(&self.model);

        if let Some(ref key) = self.api_key {
            if !key.is_empty() {
                builder = builder.api_key(key);
            } else if self.provider_type == ProviderType::OpenAICompatible {
                // OpenAI-compatible servers often don't require a real API key,
                // but the llm crate's OpenAI backend demands one. Provide a placeholder.
                builder = builder.api_key("no-key");
            }
        }

        if let Some(ref url) = self.base_url {
            builder = builder.base_url(url);
        }

        // Pass system prompt via builder.system()
        if let Some(sp) = system_prompt {
            if !sp.is_empty() {
                builder = builder.system(sp);
            }
        }

        // Forward config fields
        if let Some(temp) = config.temperature {
            builder = builder.temperature(temp);
        }
        if let Some(max_tok) = config.max_tokens {
            builder = builder.max_tokens(max_tok);
        }
        if let Some(top_p) = config.top_p {
            builder = builder.top_p(top_p);
        }
        if let Some(top_k) = config.top_k {
            builder = builder.top_k(top_k);
        }

        // Log warnings for unsupported config fields
        if let Some(fp) = config.frequency_penalty {
            tracing::debug!(
                frequency_penalty = fp,
                "frequency_penalty is not supported by the llm crate builder; ignoring"
            );
        }
        if let Some(pp) = config.presence_penalty {
            tracing::debug!(
                presence_penalty = pp,
                "presence_penalty is not supported by the llm crate builder; ignoring"
            );
        }
        if let Some(ref stops) = config.stop_sequences {
            if !stops.is_empty() {
                tracing::debug!(
                    stop_sequences = ?stops,
                    "stop_sequences is not supported by the llm crate builder; ignoring"
                );
            }
        }

        // Reasoning effort from extra
        if let Some(reasoning_effort) = config
            .extra
            .get("reasoning_effort")
            .and_then(|v| v.as_str())
        {
            let effort = match reasoning_effort.to_lowercase().as_str() {
                "low" => Some(ReasoningEffort::Low),
                "medium" => Some(ReasoningEffort::Medium),
                "high" => Some(ReasoningEffort::High),
                _ => {
                    return Err(LLMError::Config(format!(
                        "Invalid reasoning_effort '{}'. Expected one of: low, medium, high",
                        reasoning_effort
                    )));
                }
            };

            if let Some(effort) = effort {
                builder = builder.reasoning_effort(effort);
            }
        }

        builder
            .build()
            .map_err(|e| LLMError::Config(format!("Failed to build LLM: {}", e)))
    }

    /// Public backward-compatible `build_llm` — used by tests and legacy call-sites.
    pub fn build_llm(
        &self,
        config: Option<&LLMConfig>,
    ) -> Result<Box<dyn llm::LLMProvider>, LLMError> {
        let cfg = config.unwrap_or(&self.default_config);
        self.build_llm_with_system(cfg, None)
    }

    /// Ensure the cached client is built (or rebuilt) for the given config + system prompt.
    /// Returns after the cache is populated; callers should then lock and use `self.client`.
    async fn ensure_client(
        &self,
        config: Option<&LLMConfig>,
        system_prompt: Option<&str>,
    ) -> Result<(), LLMError> {
        let cfg = config.unwrap_or(&self.default_config);
        let hash = compute_config_hash(cfg, system_prompt);

        let mut lock = self.client.lock().await;
        if let Some(ref cached) = *lock {
            if cached.config_hash == hash {
                return Ok(());
            }
        }
        // Build a new client
        let llm = self.build_llm_with_system(cfg, system_prompt)?;
        *lock = Some(CachedClient {
            llm,
            config_hash: hash,
        });
        Ok(())
    }
}

#[async_trait]
impl LLMProvider for UnifiedLLMProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<LLMResponse, LLMError> {
        // Separate system messages from non-system messages
        let (system_prompt, non_system_msgs) = extract_system_and_messages(messages);

        let llm_messages: Vec<llm::chat::ChatMessage> = non_system_msgs
            .iter()
            .map(|m| self.convert_message(m))
            .collect();

        // Ensure client is built for this config + system prompt
        self.ensure_client(config, system_prompt.as_deref()).await?;

        // Use the cached client
        let lock = self.client.lock().await;
        let cached = lock
            .as_ref()
            .expect("client must be built after ensure_client");

        let response = cached
            .llm
            .chat(&llm_messages)
            .await
            .map_err(|e| LLMError::API {
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
        // Separate system messages from non-system messages
        let (system_prompt, non_system_msgs) = extract_system_and_messages(messages);

        let llm_messages: Vec<llm::chat::ChatMessage> = non_system_msgs
            .iter()
            .map(|m| self.convert_message(m))
            .collect();

        // Ensure client is built for this config + system prompt
        self.ensure_client(config, system_prompt.as_deref()).await?;

        // Acquire lock, call chat_stream, get owned stream, then release lock
        let stream = {
            let lock = self.client.lock().await;
            let cached = lock
                .as_ref()
                .expect("client must be built after ensure_client");
            cached
                .llm
                .chat_stream(&llm_messages)
                .await
                .map_err(|e| LLMError::API {
                    message: format!("LLM provider error: {}", e),
                    status: None,
                })?
            // lock is dropped here at end of block
        };

        let converted_stream = stream.map(|result| {
            result
                .map(|token| LLMChunk::new(token, false))
                .map_err(|e| LLMError::API {
                    message: format!("Stream error: {}", e),
                    status: None,
                })
        });

        // Chain a final sentinel chunk so consumers know the stream is done
        let final_stream = converted_stream.chain(futures::stream::once(async {
            Ok(LLMChunk::final_chunk("", FinishReason::Stop, None))
        }));

        Ok(Box::new(Box::pin(final_stream)))
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
            ProviderType::OpenAICompatible => "openai-compatible",
            ProviderType::OpenRouter => "openrouter",
        }
    }

    fn supports(&self, feature: LLMFeature) -> bool {
        match feature {
            LLMFeature::Streaming => true,
            LLMFeature::SystemMessages => true,
            LLMFeature::FunctionCalling => matches!(
                self.provider_type,
                ProviderType::OpenAI
                    | ProviderType::Anthropic
                    | ProviderType::Google
                    | ProviderType::OpenRouter
            ),
            LLMFeature::Vision => matches!(
                self.provider_type,
                ProviderType::OpenAI
                    | ProviderType::Anthropic
                    | ProviderType::Google
                    | ProviderType::OpenRouter
            ),
            LLMFeature::JsonMode => matches!(
                self.provider_type,
                ProviderType::OpenAI | ProviderType::Google | ProviderType::OpenRouter
            ),
            // OpenAICompatible: not included in feature matches by default —
            // capabilities depend on the actual server. Users can check at runtime.
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
    use std::collections::HashMap;

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

    #[test]
    fn test_build_llm_accepts_reasoning_effort_low() {
        let provider = ProviderBuilder::new()
            .provider(ProviderType::OpenAI)
            .model("gpt-5.1-mini")
            .api_key("XXXXXXXXXX")
            .build()
            .unwrap();

        let mut extra = HashMap::new();
        extra.insert(
            "reasoning_effort".to_string(),
            serde_json::Value::String("low".to_string()),
        );

        let config = LLMConfig {
            temperature: Some(0.7),
            max_tokens: Some(2000),
            top_p: Some(0.9),
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            extra,
        };

        let result = provider.build_llm(Some(&config));
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_llm_rejects_invalid_reasoning_effort() {
        let provider = ProviderBuilder::new()
            .provider(ProviderType::OpenAI)
            .model("gpt-5.1-mini")
            .api_key("XXXXXXXXXX")
            .build()
            .unwrap();

        let mut extra = HashMap::new();
        extra.insert(
            "reasoning_effort".to_string(),
            serde_json::Value::String("invalid".to_string()),
        );

        let config = LLMConfig {
            temperature: Some(0.7),
            max_tokens: Some(2000),
            top_p: Some(0.9),
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            extra,
        };

        let result = provider.build_llm(Some(&config));
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("Invalid reasoning_effort"));
    }

    #[test]
    fn test_from_spec_config() {
        let config = LLMConfig {
            temperature: Some(0.5),
            max_tokens: Some(4096),
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            extra: HashMap::new(),
        };

        let provider = UnifiedLLMProvider::from_spec_config(
            ProviderType::OpenAI,
            "gpt-4",
            Some("XXXXXXXXXX".to_string()),
            None,
            config,
        )
        .unwrap();

        assert_eq!(provider.provider_name(), "openai");
        assert_eq!(provider.model_name(), "gpt-4");
    }

    #[test]
    fn test_extract_system_messages() {
        let messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::system("Be concise."),
            ChatMessage::user("Hello"),
        ];

        let (system_prompt, non_system) = extract_system_and_messages(&messages);
        assert_eq!(
            system_prompt,
            Some("You are a helpful assistant.\nBe concise.".to_string())
        );
        assert_eq!(non_system.len(), 1);
        assert_eq!(non_system[0].role, Role::User);
    }

    #[test]
    fn test_extract_no_system_messages() {
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
        ];

        let (system_prompt, non_system) = extract_system_and_messages(&messages);
        assert!(system_prompt.is_none());
        assert_eq!(non_system.len(), 2);
    }

    #[test]
    fn test_config_hash_stability() {
        let config = LLMConfig {
            temperature: Some(0.7),
            max_tokens: Some(2000),
            top_p: Some(0.9),
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            extra: HashMap::new(),
        };

        let hash1 = compute_config_hash(&config, Some("system prompt"));
        let hash2 = compute_config_hash(&config, Some("system prompt"));
        assert_eq!(hash1, hash2);

        let hash3 = compute_config_hash(&config, Some("different prompt"));
        assert_ne!(hash1, hash3);

        let hash4 = compute_config_hash(&config, None);
        assert_ne!(hash1, hash4);
    }

    #[test]
    fn test_openai_compatible_from_str() {
        assert_eq!(
            ProviderType::from_str("openai-compatible").unwrap(),
            ProviderType::OpenAICompatible
        );
        assert_eq!(
            ProviderType::from_str("openai_compatible").unwrap(),
            ProviderType::OpenAICompatible
        );
    }

    #[test]
    fn test_openai_compatible_no_api_key_required() {
        assert!(ProviderType::OpenAICompatible.api_key_env_var().is_none());
    }

    #[test]
    fn test_openai_compatible_requires_base_url() {
        let result = UnifiedLLMProvider::from_spec_config(
            ProviderType::OpenAICompatible,
            "local-model",
            None,
            None, // no base_url → should error
            LLMConfig::default(),
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("base_url"));
    }

    #[test]
    fn test_openai_compatible_with_base_url() {
        let result = UnifiedLLMProvider::from_spec_config(
            ProviderType::OpenAICompatible,
            "local-model",
            None,
            Some("http://localhost:1234/v1".to_string()),
            LLMConfig::default(),
        );
        assert!(result.is_ok());
        let provider = result.unwrap();
        assert_eq!(provider.provider_name(), "openai-compatible");
        assert_eq!(provider.base_url(), Some("http://localhost:1234/v1"));
    }

    #[test]
    fn test_openrouter_from_str() {
        assert_eq!(
            ProviderType::from_str("openrouter").unwrap(),
            ProviderType::OpenRouter
        );
    }

    #[test]
    fn test_openrouter_api_key_env() {
        assert_eq!(
            ProviderType::OpenRouter.api_key_env_var(),
            Some("OPENROUTER_API_KEY")
        );
    }
}

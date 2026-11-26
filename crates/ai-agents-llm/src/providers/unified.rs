use ai_agents_core::autonomy::{ProviderCostBound, ProviderCostSettlement, ProviderRequest};
use ai_agents_core::{
    ChatMessage, ChoiceDecision, ChoiceRequest, FinishReason, LLMChunk, LLMConfig, LLMError,
    LLMFeature, LLMProvider, LLMResponse, LLMToolDefinition, LLMToolRequest, Role, TokenUsage,
    ToolCall, ToolChoice, native_readable_projection,
};
use async_trait::async_trait;
use futures::stream::StreamExt;
use llm::chat::ReasoningEffort;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};

use super::chat_adapter;
use super::google::GoogleProvider;
use super::pricing::DeclaredPricing;
use super::{messages_adapter, ollama_adapter, ollama_local, openrouter, responses_adapter};
use crate::sdk::{
    ChatClient,
    decisions::{self, ChoiceOption, DecisionClient},
    messages::{self, MessagesClient},
    ollama::{LOCAL_ROOT, LocalChatRequest, OllamaCloudClient, OllamaLocalClient},
    openrouter::OpenRouterClient,
    responses::ResponsesClient,
};
#[derive(Clone)]
enum SdkBackend {
    Chat(ChatClient),
    OpenRouter(OpenRouterClient),
    Responses(ResponsesClient),
    Messages(MessagesClient),
    OllamaCloud(OllamaCloudClient),
    OllamaLocal(OllamaLocalClient),
    Decisions(DecisionClient),
}

/// The preset selector for the first-party OpenAI Decisions transport.
pub const DECISIONS_PROTOCOL: &str = "decisions";

/// Classifies a declared protocol without constructing a provider or resolving credentials.
pub fn is_choice_protocol(protocol: Option<&str>) -> bool {
    matches!(protocol, Some(DECISIONS_PROTOCOL))
}

#[cfg(test)]
#[path = "pricing_tests.rs"]
mod pricing_tests;

static NEXT_NORMALIZED_TOOL_CALL_ID: AtomicU64 = AtomicU64::new(1);

fn next_normalized_tool_call_id() -> String {
    format!(
        "ai-tool-call-{}",
        NEXT_NORMALIZED_TOOL_CALL_ID.fetch_add(1, Ordering::Relaxed)
    )
}

/// Provider type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    /// OpenAI (GPT models)
    OpenAI,
    /// Anthropic (Claude models)
    Anthropic,
    /// An explicitly configured Messages endpoint.
    #[serde(rename = "anthropic-compatible")]
    AnthropicCompatible,
    /// Ollama (local models)
    Ollama,
    /// Native Ollama Cloud chat.
    #[serde(rename = "ollama-cloud")]
    OllamaCloud,
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
    /// Whether this preset participates in the framework's compatibility text tool protocol.
    pub fn allows_prompt_tools(self) -> bool {
        self != Self::OllamaCloud
    }

    fn model_allows_tools(self, model: &str, config: &LLMConfig) -> bool {
        if self != Self::OpenAI {
            return true;
        }
        let protocol = match config.extra.get("protocol").map(serde_json::Value::as_str) {
            None | Some(Some("chat-completions")) => crate::sdk::openai::Protocol::Chat,
            Some(Some("responses")) => crate::sdk::openai::Protocol::Responses,
            _ => return false,
        };
        super::openai::allows_tools(protocol, model, config)
    }

    /// Resolves native tool capability from declarations alone, including endpoint and model restrictions.
    pub fn supports_native_tool_choice(
        self,
        model: &str,
        config: &LLMConfig,
        enabled: Option<bool>,
        choice: &ToolChoice,
    ) -> bool {
        if !self.supports_preset_choice(enabled, choice) {
            return false;
        }
        let protocol = config
            .extra
            .get("protocol")
            .and_then(serde_json::Value::as_str);
        if self.uses_messages(protocol) {
            return messages_adapter::supports_choice(
                self == Self::Anthropic,
                model,
                config,
                choice,
            );
        }
        if self.is_default_anthropic(config) && !messages_adapter::uses_resilient_transport(config)
        {
            return messages_adapter::supports_default_choice(model, config, choice);
        }
        self.model_allows_tools(model, config)
    }

    /// Checks a configured choice before opening a live client or selecting an offline fixture.
    pub fn validate_declared_tool_choice(
        self,
        model: &str,
        config: &LLMConfig,
        enabled: Option<bool>,
        choice: &ToolChoice,
    ) -> Result<(), LLMError> {
        if self.supports_native_tool_choice(model, config, enabled, choice) {
            return Ok(());
        }
        if !self.allows_prompt_tools() {
            return Err(LLMError::Config(
                "configured choice requires native tool support".into(),
            ));
        }
        if (self.uses_messages(
            config
                .extra
                .get("protocol")
                .and_then(serde_json::Value::as_str),
        ) || (self.is_default_anthropic(config)
            && !messages_adapter::uses_resilient_transport(config)))
            && matches!(choice, ToolChoice::Required | ToolChoice::Specific(_))
        {
            return Err(LLMError::Config(
                "configured Messages tool choice is unsupported".into(),
            ));
        }
        Ok(())
    }

    fn supports_preset_choice(self, enabled: Option<bool>, choice: &ToolChoice) -> bool {
        if self == Self::OllamaCloud && matches!(choice, ToolChoice::None) {
            return true;
        }
        if enabled == Some(false) {
            return false;
        }
        let known_choice = matches!(
            choice,
            ToolChoice::Auto | ToolChoice::Required | ToolChoice::Specific(_) | ToolChoice::None
        );
        match self {
            ProviderType::OpenAI | ProviderType::Anthropic | ProviderType::OpenRouter => {
                known_choice
            }
            ProviderType::Google | ProviderType::OllamaCloud => matches!(choice, ToolChoice::Auto),
            ProviderType::OpenAICompatible | ProviderType::AnthropicCompatible => {
                enabled.unwrap_or(false) && known_choice
            }
            _ => false,
        }
    }

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
            Self::OllamaCloud => Some("OLLAMA_API_KEY"),
            Self::Ollama => None, // Ollama doesn't need an API key
            Self::OpenAICompatible | Self::AnthropicCompatible => None, // User specifies via api_key_env or it's optional
        }
    }

    /// Resolves an explicitly selected credential source at provider construction time.
    /// The official preset must not fall back to another key. Compatible servers can operate without credentials.
    pub fn configured_api_key(&self, variable: Option<&str>) -> Result<Option<String>, LLMError> {
        self.configured_api_key_for_protocol(variable, None)
    }

    /// Resolves the declared key for the selected protocol without changing legacy preset fallback.
    pub fn configured_api_key_for_protocol(
        &self,
        variable: Option<&str>,
        protocol: Option<&str>,
    ) -> Result<Option<String>, LLMError> {
        self.configured_api_key_at(variable, protocol, None)
    }

    /// Resolves a credential after checking provider-owned endpoint restrictions.
    pub fn configured_api_key_at(
        &self,
        variable: Option<&str>,
        protocol: Option<&str>,
        base_url: Option<&str>,
    ) -> Result<Option<String>, LLMError> {
        if *self == Self::Ollama {
            return Ok(None);
        }
        self.check_credential_destination(variable, base_url)?;
        let Some(name) = variable else {
            return Ok(None);
        };
        match std::env::var(name) {
            Ok(key) => Ok(Some(key)),
            Err(_)
                if matches!(*self, Self::OpenAI | Self::OllamaCloud | Self::OpenRouter)
                    || self.uses_messages(protocol) =>
            {
                Err(LLMError::Config(format!(
                    "API key not found in configured environment variable {name}"
                )))
            }
            Err(_) => Ok(None),
        }
    }

    fn check_credential_destination(
        &self,
        variable: Option<&str>,
        base_url: Option<&str>,
    ) -> Result<(), LLMError> {
        if *self == Self::OllamaCloud
            && variable == self.api_key_env_var()
            && base_url.is_some_and(|base| !ollama_adapter::official_origin(base))
        {
            return Err(LLMError::Config(
                "OLLAMA_API_KEY is bound to the official Ollama Cloud origin".into(),
            ));
        }
        if *self == Self::OpenRouter
            && variable == self.api_key_env_var()
            && base_url.is_some_and(|base| !openrouter::official_origin(base))
        {
            return Err(LLMError::Config(
                "OPENROUTER_API_KEY is bound to the official OpenRouter origin".into(),
            ));
        }
        Ok(())
    }

    pub fn default_base_url(&self) -> Option<&'static str> {
        match self {
            Self::Ollama => Some(LOCAL_ROOT),
            Self::OpenAICompatible => None, // MUST be provided by user via base_url
            Self::OpenRouter => Some(crate::sdk::openrouter::API_BASE),
            _ => None, // Most providers use their default URLs
        }
    }

    /// Whether this declaration uses the explicit Messages configuration contract.
    pub fn uses_messages(self, protocol: Option<&str>) -> bool {
        match self {
            Self::Anthropic => protocol == Some("messages"),
            Self::AnthropicCompatible => matches!(protocol, None | Some("messages")),
            _ => false,
        }
    }

    // A missing protocol retains the historical framework contract even with SDK dispatch.
    fn is_default_anthropic(self, config: &LLMConfig) -> bool {
        self == Self::Anthropic && !config.extra.contains_key("protocol")
    }

    /// Reports whether a default Anthropic configuration requires signed evidence to replay a legacy tool exchange.
    pub fn requires_signed_default_anthropic_history(
        self,
        model: &str,
        config: &LLMConfig,
    ) -> bool {
        self.is_default_anthropic(config)
            && !messages_adapter::allows_legacy_unsigned_replay(model, config)
    }

    fn to_llm_backend(self) -> Option<llm::builder::LLMBackend> {
        Some(match self {
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
            Self::AnthropicCompatible | Self::OllamaCloud => return None,
        })
    }
}

impl FromStr for ProviderType {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(Self::OpenAI),
            "anthropic" => Ok(Self::Anthropic),
            "anthropic-compatible" => Ok(Self::AnthropicCompatible),
            "ollama" => Ok(Self::Ollama),
            "ollama-cloud" => Ok(Self::OllamaCloud),
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
    config_key: serde_json::Value,
}

pub struct UnifiedLLMProvider {
    provider_type: ProviderType,
    model: String,
    api_key: Option<String>,
    base_url: Option<String>,
    default_config: LLMConfig,
    feature_overrides: HashMap<LLMFeature, bool>,
    tool_choice: Option<ToolChoice>,
    client: std::sync::Arc<tokio::sync::Mutex<Option<CachedClient>>>,
    google: Option<GoogleProvider>,
    sdk: Option<SdkBackend>,
    pricing: Option<DeclaredPricing>,
}

impl std::fmt::Debug for UnifiedLLMProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnifiedLLMProvider")
            .field("provider_type", &self.provider_type)
            .field("model", &self.model)
            .field("api_key", &self.api_key.as_ref().map(|_| "***"))
            .field("base_url", &self.base_url)
            .field("default_config", &self.default_config)
            .field("feature_overrides", &self.feature_overrides)
            .field("tool_choice", &self.tool_choice)
            .field("client", &"<cached>")
            .field("google", &self.google)
            .finish()
    }
}

/// Keeps typed settings and extra fallbacks distinct so flattening or hash collisions cannot select a foreign client.
fn request_config_key(
    config: &LLMConfig,
    system_prompt: Option<&str>,
    tool_choice: Option<&ToolChoice>,
    tools: Option<&[LLMToolDefinition]>,
) -> Result<serde_json::Value, LLMError> {
    let mut typed = config.clone();
    typed.extra.clear();
    let float_bits = [
        config.temperature.map(f32::to_bits),
        config.top_p.map(f32::to_bits),
        config.frequency_penalty.map(f32::to_bits),
        config.presence_penalty.map(f32::to_bits),
    ];
    serde_json::to_value((
        typed,
        &config.extra,
        float_bits,
        system_prompt,
        tool_choice,
        tools,
    ))
    .map_err(|error| LLMError::Config(format!("cannot bind request configuration: {error}")))
}

/// Compute a hash over the config fields that affect the LLM builder, plus the system prompt.
fn compute_config_hash(
    config: &LLMConfig,
    system_prompt: Option<&str>,
    tool_choice: Option<&ToolChoice>,
    tools: Option<&[LLMToolDefinition]>,
) -> u64 {
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
    // First-class fields
    if let Some(timeout) = config.timeout_seconds {
        timeout.hash(&mut hasher);
    }
    if let Some(reasoning) = config.reasoning {
        reasoning.hash(&mut hasher);
    }
    if let Some(ref effort) = config.reasoning_effort {
        effort.hash(&mut hasher);
    }
    if let Some(budget) = config.reasoning_budget_tokens {
        budget.hash(&mut hasher);
    }
    // Extra keys forwarded to the builder
    const FORWARDED_EXTRA_KEYS: &[&str] = &[
        "timeout_seconds",
        "reasoning",
        "reasoning_effort",
        "reasoning_budget_tokens",
        "normalize_response",
        "api_version",
        "deployment_id",
        "resilient",
        "resilient_attempts",
        "resilient_base_delay_ms",
        "resilient_max_delay_ms",
        "resilient_jitter",
        "extra_body",
        "openai_enable_web_search",
        "openai_web_search_context_size",
        "xai_search_mode",
        "xai_max_search_results",
        "xai_search_from_date",
        "xai_search_to_date",
        "num_ctx",
        "num_gpu",
        "keep_alive",
    ];
    for key in FORWARDED_EXTRA_KEYS {
        if let Some(v) = config.extra.get(*key) {
            key.hash(&mut hasher);
            v.to_string().hash(&mut hasher);
        }
    }
    if let Some(sp) = system_prompt {
        sp.hash(&mut hasher);
    }
    tool_choice.hash(&mut hasher);
    if let Some(tools) = tools {
        for tool in tools {
            tool.name.hash(&mut hasher);
            tool.description.hash(&mut hasher);
            tool.input_schema.to_string().hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Extract system messages from a message slice, returning the combined system prompt
/// (if any) and the remaining non-system messages.
pub(super) fn extract_system_and_messages(
    messages: &[ChatMessage],
) -> (Option<String>, Vec<&ChatMessage>) {
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

fn provider_protocol_error(message: impl Into<String>) -> LLMError {
    LLMError::API {
        message: format!("Provider protocol error: {}", message.into()),
        status: None,
    }
}

fn map_tool_definition(definition: &LLMToolDefinition) -> llm::chat::Tool {
    llm::chat::Tool {
        tool_type: "function".to_string(),
        function: llm::chat::FunctionTool {
            name: definition.name.clone(),
            description: definition.description.clone(),
            parameters: definition.input_schema.clone(),
        },
    }
}

fn map_tool_choice(choice: &ToolChoice) -> Result<llm::chat::ToolChoice, LLMError> {
    match choice {
        ToolChoice::Auto => Ok(llm::chat::ToolChoice::Auto),
        ToolChoice::Required => Ok(llm::chat::ToolChoice::Any),
        ToolChoice::Specific(name) => Ok(llm::chat::ToolChoice::Tool(name.clone())),
        ToolChoice::None => Ok(llm::chat::ToolChoice::None),
        _ => Err(LLMError::Config(
            "unsupported native tool choice variant".to_string(),
        )),
    }
}

fn normalize_tool_calls(
    calls: Option<Vec<llm::ToolCall>>,
    choice: &ToolChoice,
) -> Result<Vec<ToolCall>, LLMError> {
    let calls = calls.unwrap_or_default();
    if calls.is_empty() && matches!(choice, ToolChoice::Required | ToolChoice::Specific(_)) {
        return Err(provider_protocol_error(
            "the provider returned no tool calls for a required tool choice",
        ));
    }

    let specific_name = match choice {
        ToolChoice::Specific(name) => Some(name.as_str()),
        _ => None,
    };
    let mut seen_ids = HashSet::new();
    let mut normalized = Vec::with_capacity(calls.len());

    for call in calls {
        if let Some(expected) = specific_name
            && call.function.name != expected
        {
            return Err(provider_protocol_error(format!(
                "specific tool choice '{expected}' returned call for '{}'",
                call.function.name
            )));
        }

        let arguments = serde_json::from_str(&call.function.arguments).map_err(|error| {
            provider_protocol_error(format!(
                "tool '{}' returned invalid JSON arguments: {error}",
                call.function.name
            ))
        })?;
        let mut normalized_call = ToolCall {
            id: next_normalized_tool_call_id(),
            name: call.function.name,
            arguments,
        };
        if !call.id.is_empty() && seen_ids.insert(call.id.clone()) {
            normalized_call.id = call.id;
        } else {
            while !seen_ids.insert(normalized_call.id.clone()) {
                normalized_call.id = next_normalized_tool_call_id();
            }
        }
        normalized.push(normalized_call);
    }

    Ok(normalized)
}

#[derive(Deserialize)]
struct NativeToolCallMarker {
    _ai_agents_native_tool_call: bool,
    id: String,
    tool: String,
    arguments: serde_json::Value,
}

#[derive(Deserialize)]
struct NativeToolResultMarker {
    _ai_agents_native_tool_result: bool,
    id: String,
    tool: String,
    output: serde_json::Value,
}

fn marked_json_values(content: &str, marker: &str) -> Option<Vec<serde_json::Value>> {
    let value: serde_json::Value = serde_json::from_str(content).ok()?;
    let values = match value {
        serde_json::Value::Array(values) => values,
        value @ serde_json::Value::Object(_) => vec![value],
        _ => return None,
    };

    if !values.is_empty()
        && values
            .iter()
            .all(|value| value.get(marker).and_then(|value| value.as_bool()) == Some(true))
    {
        Some(values)
    } else {
        None
    }
}

// Preserve typed-over-extra precedence and the accepted spellings of the upstream builder.
pub(super) fn legacy_reasoning_effort(
    config: &LLMConfig,
) -> Result<Option<ReasoningEffort>, LLMError> {
    let effort = config.reasoning_effort.as_deref().or_else(|| {
        config
            .extra
            .get("reasoning_effort")
            .and_then(serde_json::Value::as_str)
    });
    let Some(effort) = effort else {
        return Ok(None);
    };
    match effort.to_ascii_lowercase().as_str() {
        "low" => Ok(Some(ReasoningEffort::Low)),
        "medium" => Ok(Some(ReasoningEffort::Medium)),
        "high" => Ok(Some(ReasoningEffort::High)),
        other => Err(LLMError::Config(format!(
            "Invalid reasoning_effort '{}'. Expected: low, medium, high",
            other
        ))),
    }
}

fn merged_ollama_extra_body(config: &LLMConfig) -> Result<Option<serde_json::Value>, LLMError> {
    let has_named_fields = config.extra.contains_key("num_ctx")
        || config.extra.contains_key("num_gpu")
        || config.extra.contains_key("keep_alive");

    let extra_body = config.extra.get("extra_body").cloned();

    if !has_named_fields {
        return Ok(extra_body);
    }

    let mut body = match extra_body {
        Some(serde_json::Value::Object(map)) => serde_json::Value::Object(map),
        Some(_) => {
            return Err(LLMError::Config(
                "Ollama extra_body must be a JSON object when merging num_ctx, num_gpu, or keep_alive"
                    .to_string(),
            ));
        }
        None => serde_json::json!({}),
    };

    let body_map = body
        .as_object_mut()
        .ok_or_else(|| LLMError::Config("Ollama extra_body must be a JSON object".to_string()))?;

    if let Some(keep_alive) = config.extra.get("keep_alive") {
        body_map.insert("keep_alive".to_string(), keep_alive.clone());
    }

    if config.extra.contains_key("num_ctx") || config.extra.contains_key("num_gpu") {
        let options = body_map
            .entry("options".to_string())
            .or_insert_with(|| serde_json::json!({}));
        let options_map = options.as_object_mut().ok_or_else(|| {
            LLMError::Config(
                "Ollama extra_body.options must be a JSON object when merging num_ctx or num_gpu"
                    .to_string(),
            )
        })?;

        if let Some(num_ctx) = config.extra.get("num_ctx") {
            options_map.insert("num_ctx".to_string(), num_ctx.clone());
        }
        if let Some(num_gpu) = config.extra.get("num_gpu") {
            options_map.insert("num_gpu".to_string(), num_gpu.clone());
        }
    }

    Ok(Some(body))
}

fn map_provider_error(
    provider_type: ProviderType,
    model: &str,
    base_url: Option<&str>,
    err: impl std::fmt::Display,
) -> LLMError {
    if provider_type != ProviderType::Ollama {
        return LLMError::API {
            message: format!("LLM provider error: {}", err),
            status: None,
        };
    }

    let base = base_url.unwrap_or(LOCAL_ROOT);
    let msg = err.to_string();
    let lower = msg.to_ascii_lowercase();

    if lower.contains("connection refused")
        || lower.contains("failed to connect")
        || lower.contains("error sending request")
        || lower.contains("os error 111")
    {
        return LLMError::Network(format!(
            "Cannot connect to Ollama at {}. Is Ollama running? Start it with: ollama serve",
            base
        ));
    }

    if lower.contains("model") && lower.contains("not found") {
        return LLMError::API {
            message: format!(
                "Model '{}' not found in Ollama. Pull it with: ollama pull {}",
                model, model
            ),
            status: None,
        };
    }

    LLMError::API {
        message: format!("Ollama provider error: {}", msg),
        status: None,
    }
}

impl UnifiedLLMProvider {
    fn model_allows_tools(&self) -> bool {
        self.provider_type
            .model_allows_tools(&self.model, &self.default_config)
    }

    fn supports_choice_for_provider(&self, choice: &ToolChoice) -> bool {
        self.provider_type.supports_preset_choice(
            self.feature_overrides
                .get(&LLMFeature::FunctionCalling)
                .copied(),
            choice,
        )
    }

    pub fn new(
        provider_type: ProviderType,
        model: String,
        api_key: Option<String>,
        base_url: Option<String>,
    ) -> Result<Self, LLMError> {
        // This convenience constructor has no caller-supplied output limit.
        if provider_type.uses_messages(None) {
            return Err(LLMError::Config(
                "Messages requires explicit max_tokens in from_spec_config".into(),
            ));
        }
        let mut config = LLMConfig::default();
        if matches!(
            provider_type,
            ProviderType::OpenAI | ProviderType::OllamaCloud
        ) {
            config.temperature = None;
            config.top_p = None;
        }
        if provider_type == ProviderType::OllamaCloud {
            config.max_tokens = None;
        }
        Self::from_spec_config(provider_type, &model, api_key, base_url, config)
    }

    /// Create a new UnifiedLLMProvider with explicit config defaults.
    pub fn from_spec_config(
        provider_type: ProviderType,
        model: &str,
        api_key: Option<String>,
        base_url: Option<String>,
        default_config: LLMConfig,
    ) -> Result<Self, LLMError> {
        let is_chat = matches!(
            provider_type,
            ProviderType::OpenAI | ProviderType::OpenAICompatible
        );
        let is_openrouter = provider_type == ProviderType::OpenRouter;
        let protocol = default_config.extra.get("protocol");
        let is_decisions = is_choice_protocol(protocol.and_then(serde_json::Value::as_str));
        if is_decisions {
            if provider_type != ProviderType::OpenAI || model.trim().is_empty() {
                return Err(LLMError::Config(
                    "Decisions requires the OpenAI preset and a model".into(),
                ));
            }
            if default_config.temperature.is_some()
                || default_config.max_tokens.is_some()
                || default_config.top_p.is_some()
                || default_config.top_k.is_some()
                || default_config.frequency_penalty.is_some()
                || default_config.presence_penalty.is_some()
                || default_config.stop_sequences.is_some()
                || default_config.reasoning.is_some()
                || default_config.reasoning_effort.is_some()
                || default_config.reasoning_budget_tokens.is_some()
                || default_config.extra.len() != 1
            {
                return Err(LLMError::Config(
                    "Decisions does not accept generation settings, extra options or declared pricing".into(),
                ));
            }
        }
        let is_default_anthropic = provider_type.is_default_anthropic(&default_config);
        if is_default_anthropic {
            messages_adapter::validate_default_destination(base_url.as_deref())?;
            messages_adapter::validate_default_options(&default_config)?;
        }
        let is_messages = provider_type.uses_messages(protocol.and_then(serde_json::Value::as_str))
            && protocol.is_none_or(|value| value.is_string());
        if provider_type == ProviderType::AnthropicCompatible && !is_messages {
            return Err(LLMError::Config(
                "anthropic-compatible requires the Messages protocol".into(),
            ));
        }
        let is_ollama_local = provider_type == ProviderType::Ollama;
        let is_ollama_cloud = provider_type == ProviderType::OllamaCloud;
        if is_ollama_cloud {
            if protocol.is_some() {
                return Err(LLMError::Config(
                    "ollama-cloud selects the native chat protocol".into(),
                ));
            }
            if api_key.is_none()
                && base_url
                    .as_deref()
                    .is_some_and(|base| !ollama_adapter::official_origin(base))
            {
                return Err(LLMError::Config(
                    "custom Ollama Cloud origin requires explicitly selected credentials".into(),
                ));
            }
        }
        if is_openrouter {
            if protocol.is_some_and(|p| p != "chat-completions") {
                return Err(LLMError::Config(
                    "OpenRouter requires Chat Completions".into(),
                ));
            }
            if api_key.is_none()
                && base_url
                    .as_deref()
                    .is_some_and(|base| !openrouter::official_origin(base))
            {
                return Err(LLMError::Config(
                    "custom OpenRouter origin requires explicitly selected credentials".into(),
                ));
            }
        }
        if !is_chat
            && !is_openrouter
            && !is_messages
            && !is_ollama_cloud
            && (default_config.extra.contains_key("protocol")
                || !is_ollama_local && default_config.extra.contains_key("provider_options"))
        {
            return Err(LLMError::Config(
                "SDK protocol/options require a supported provider preset".into(),
            ));
        }
        if provider_type == ProviderType::OpenAI
            && api_key.is_none()
            && let Some(base) = base_url.as_deref()
            && !reqwest::Url::parse(base)
                .is_ok_and(|u| u.origin().ascii_serialization() == "https://api.openai.com")
        {
            return Err(LLMError::Config(
                "custom OpenAI origin requires explicitly selected credentials".into(),
            ));
        }
        if is_messages
            && provider_type == ProviderType::Anthropic
            && api_key.is_none()
            && let Some(base) = base_url.as_deref()
            && !reqwest::Url::parse(base)
                .is_ok_and(|url| url.origin().ascii_serialization() == messages::OFFICIAL_ORIGIN)
        {
            return Err(LLMError::Config(
                "custom Anthropic origin requires explicitly selected credentials".into(),
            ));
        }
        if is_messages && !default_config.max_tokens.is_some_and(|limit| limit > 0) {
            return Err(LLMError::Config(
                "Messages requires explicit positive max_tokens".into(),
            ));
        }
        let actual_api_key = if is_ollama_local {
            String::new()
        } else if let Some(key) = api_key {
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

        let sdk = if is_openrouter {
            Some(SdkBackend::OpenRouter(
                OpenRouterClient::at_endpoint(
                    actual_base_url
                        .as_deref()
                        .unwrap_or(crate::sdk::openrouter::API_BASE),
                    actual_api_key.clone(),
                )
                .map_err(super::sdk_error)?,
            ))
        } else if is_ollama_cloud {
            Some(SdkBackend::OllamaCloud(ollama_adapter::connect(
                actual_base_url.as_deref(),
                actual_api_key.clone(),
            )?))
        } else if is_ollama_local {
            ollama_local::uses_legacy(&default_config)?;
            Some(SdkBackend::OllamaLocal(
                OllamaLocalClient::new(actual_base_url.as_deref()).map_err(super::sdk_error)?,
            ))
        } else if is_chat {
            let base = actual_base_url
                .as_deref()
                .unwrap_or(crate::sdk::openai::API_BASE);
            Some(
                match default_config.extra.get("protocol").map(|v| v.as_str()) {
                    None | Some(Some("chat-completions")) => SdkBackend::Chat(
                        ChatClient::new(base, Some(actual_api_key.clone()))
                            .map_err(super::sdk_error)?
                            .with_openai_contract(provider_type == ProviderType::OpenAI),
                    ),
                    Some(Some("responses")) if provider_type == ProviderType::OpenAI => {
                        SdkBackend::Responses(
                            ResponsesClient::new(base, Some(actual_api_key.clone()))
                                .map_err(super::sdk_error)?
                                .with_official_contract(),
                        )
                    }
                    Some(Some(DECISIONS_PROTOCOL)) => SdkBackend::Decisions(
                        DecisionClient::new(base, actual_api_key.clone())
                            .map_err(super::sdk_error)?,
                    ),
                    _ => return Err(LLMError::Config("unsupported protocol for provider".into())),
                },
            )
        } else if is_messages || is_default_anthropic {
            Some(SdkBackend::Messages(messages_adapter::connect(
                provider_type,
                actual_base_url.as_deref(),
                actual_api_key.clone(),
                &default_config,
            )?))
        } else {
            None
        };
        let google = if provider_type == ProviderType::Google {
            Some(GoogleProvider::new(
                model.to_string(),
                actual_api_key.clone(),
                actual_base_url.clone(),
                default_config.clone(),
            )?)
        } else {
            None
        };

        let pricing = default_config
            .extra
            .get("declared_pricing")
            .cloned()
            .map(serde_json::from_value::<DeclaredPricing>)
            .transpose()
            .map_err(|_| LLMError::Config("invalid declared_pricing contract".into()))?;
        let mut provider = Self {
            provider_type,
            model: model.to_string(),
            api_key: Some(actual_api_key),
            base_url: actual_base_url,
            default_config,
            feature_overrides: HashMap::new(),
            tool_choice: None,
            client: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
            google,
            sdk,
            pricing: None,
        };
        if let Some(pricing) = pricing {
            provider = provider.with_declared_pricing(pricing)?;
        }
        Ok(provider)
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

    pub fn with_feature_override(mut self, feature: LLMFeature, enabled: bool) -> Self {
        self.feature_overrides.insert(feature, enabled);
        if let Some(google) = &mut self.google {
            google.set_feature_override(feature, enabled);
        }
        self
    }

    pub fn with_feature_overrides(mut self, overrides: HashMap<LLMFeature, bool>) -> Self {
        self.feature_overrides.extend(overrides);
        if let Some(google) = &mut self.google {
            google.set_feature_overrides(&self.feature_overrides);
        }
        self
    }

    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice.clone());
        if let Some(google) = &mut self.google {
            google.set_tool_choice(choice);
        }
        self
    }

    /// Applies a declared native tool choice only to a compatible generation protocol.
    pub fn with_validated_tool_choice(self, choice: ToolChoice) -> Result<Self, LLMError> {
        self.require_generation()?;
        let provider = self.with_tool_choice(choice);
        provider.validate_configured_tool_choice()?;
        Ok(provider)
    }

    fn require_generation(&self) -> Result<(), LLMError> {
        if self.uses_choice_for_routing() {
            return Err(LLMError::Config(
                "Decisions is a choice-only provider".into(),
            ));
        }
        Ok(())
    }

    fn decision_request<'a>(
        &'a self,
        request: &ChoiceRequest<'a>,
        options: &'a [ChoiceOption<'a>],
    ) -> decisions::ChoiceRequest<'a> {
        decisions::ChoiceRequest {
            model: &self.model,
            input: request.input,
            question: request.question,
            options,
            timeout: self
                .default_config
                .timeout_seconds
                .map(std::time::Duration::from_secs),
        }
    }

    fn decision_client(&self) -> Result<&DecisionClient, LLMError> {
        match &self.sdk {
            Some(SdkBackend::Decisions(client)) => Ok(client),
            _ => Err(LLMError::Config(
                "provider is not configured for structured choice".into(),
            )),
        }
    }

    fn messages_config<'a>(
        &'a self,
        config: Option<&'a LLMConfig>,
    ) -> Result<&'a LLMConfig, LLMError> {
        let config = config.unwrap_or(&self.default_config);
        for key in ["messages_auth", "anthropic_version"] {
            if config
                .extra
                .get(key)
                .is_some_and(|value| self.default_config.extra.get(key) != Some(value))
            {
                return Err(LLMError::Config(format!(
                    "{key} is fixed at Messages client construction"
                )));
            }
        }
        Ok(config)
    }

    // Dispatch resilient requests through their declared upstream wrapper before attempting I/O.
    fn messages_dispatch<'a>(
        &'a self,
        config: Option<&'a LLMConfig>,
        messages: &[ChatMessage],
    ) -> Result<Option<(&'a MessagesClient, messages_adapter::RequestConfig<'a>)>, LLMError> {
        let Some(SdkBackend::Messages(client)) = &self.sdk else {
            return Ok(None);
        };
        let cfg = self.messages_config(config)?;
        let settings = if self
            .provider_type
            .is_default_anthropic(&self.default_config)
        {
            messages_adapter::validate_default_options(cfg)?;
            if messages_adapter::uses_resilient_transport(cfg) {
                if ai_agents_core::inspect_native_history(messages)?
                    .exchanges()
                    .iter()
                    .any(|turn| {
                        turn.state().provider() == "anthropic" && turn.state().api() == "messages"
                    })
                {
                    return Err(LLMError::Config(
                        "legacy Anthropic resilience cannot replay signed Messages history".into(),
                    ));
                }
                return Ok(None);
            }
            messages_adapter::RequestConfig::DefaultAnthropic(cfg)
        } else {
            messages_adapter::RequestConfig::Explicit(cfg)
        };
        Ok(Some((client, settings)))
    }

    fn validate_configured_tool_choice(&self) -> Result<(), LLMError> {
        if let Some(choice) = &self.tool_choice {
            self.provider_type.validate_declared_tool_choice(
                &self.model,
                &self.default_config,
                self.feature_overrides
                    .get(&LLMFeature::FunctionCalling)
                    .copied(),
                choice,
            )?;
        }
        Ok(())
    }

    fn map_llm_crate_error(&self, err: impl std::fmt::Display) -> LLMError {
        map_provider_error(
            self.provider_type,
            &self.model,
            self.base_url.as_deref(),
            err,
        )
    }

    /// Removes replay-only state before a non-Google backend sees history while leaving caller-owned messages unchanged.
    fn project_foreign_messages(messages: &[ChatMessage]) -> Result<Vec<ChatMessage>, LLMError> {
        messages
            .iter()
            .cloned()
            .map(|mut message| {
                if matches!(message.role, Role::Assistant | Role::Tool | Role::Function) {
                    message.content = native_readable_projection(&message.content)?;
                }
                Ok(message)
            })
            .collect()
    }

    // Select the legacy resilience wrapper before dispatch and validate only the effective request.
    fn local_ollama_request(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<Option<LocalChatRequest>, LLMError> {
        let cfg = config.unwrap_or(&self.default_config);
        if ollama_local::uses_legacy(cfg)? {
            return Ok(None);
        }
        // The legacy Ollama builder validated effort but never sent it to the daemon.
        legacy_reasoning_effort(cfg)?;
        let legacy_body = merged_ollama_extra_body(cfg)?;
        let messages = Self::project_foreign_messages(messages)?;
        ollama_local::request(&self.model, &messages, cfg, legacy_body.as_ref()).map(Some)
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

    fn convert_message_with_tools(
        &self,
        msg: &ChatMessage,
    ) -> Result<llm::chat::ChatMessage, LLMError> {
        if msg.role == Role::Assistant
            && let Some(values) = marked_json_values(&msg.content, "_ai_agents_native_tool_call")
        {
            let calls = values
                .into_iter()
                .map(|value| {
                    let marker: NativeToolCallMarker =
                        serde_json::from_value(value).map_err(|error| {
                            provider_protocol_error(format!("invalid tool call marker: {error}"))
                        })?;
                    if marker.id.is_empty() || marker.tool.is_empty() {
                        return Err(provider_protocol_error(
                            "tool call markers require non-empty id and tool fields",
                        ));
                    }
                    let arguments = serde_json::to_string(&marker.arguments).map_err(|error| {
                        provider_protocol_error(format!(
                            "failed to encode tool call marker arguments: {error}"
                        ))
                    })?;
                    Ok(llm::ToolCall {
                        id: marker.id,
                        call_type: "function".to_string(),
                        function: llm::FunctionCall {
                            name: marker.tool,
                            arguments,
                        },
                    })
                })
                .collect::<Result<Vec<_>, LLMError>>()?;
            return Ok(llm::chat::ChatMessage::assistant().tool_use(calls).build());
        }

        if matches!(msg.role, Role::Tool | Role::Function)
            && let Some(values) = marked_json_values(&msg.content, "_ai_agents_native_tool_result")
        {
            let results = values
                .into_iter()
                .map(|value| {
                    let marker: NativeToolResultMarker =
                        serde_json::from_value(value).map_err(|error| {
                            provider_protocol_error(format!("invalid tool result marker: {error}"))
                        })?;
                    if marker.id.is_empty() || marker.tool.is_empty() {
                        return Err(provider_protocol_error(
                            "tool result markers require non-empty id and tool fields",
                        ));
                    }
                    let output = serde_json::to_string(&marker.output).map_err(|error| {
                        provider_protocol_error(format!(
                            "failed to encode tool result marker output: {error}"
                        ))
                    })?;
                    Ok(llm::ToolCall {
                        id: marker.id,
                        call_type: "function".to_string(),
                        function: llm::FunctionCall {
                            name: marker.tool,
                            arguments: output,
                        },
                    })
                })
                .collect::<Result<Vec<_>, LLMError>>()?;
            return Ok(llm::chat::ChatMessage::user().tool_result(results).build());
        }

        Ok(self.convert_message(msg))
    }

    fn convert_messages_with_tools(
        &self,
        messages: &[&ChatMessage],
    ) -> Result<Vec<llm::chat::ChatMessage>, LLMError> {
        let mut converted = Vec::with_capacity(messages.len());
        let mut pending_results = Vec::new();
        for message in messages {
            let next = self.convert_message_with_tools(message)?;
            if let llm::chat::MessageType::ToolResult(results) = &next.message_type {
                pending_results.extend(results.clone());
                continue;
            }
            if !pending_results.is_empty() {
                converted.push(
                    llm::chat::ChatMessage::user()
                        .tool_result(std::mem::take(&mut pending_results))
                        .build(),
                );
            }
            converted.push(next);
        }
        if !pending_results.is_empty() {
            converted.push(
                llm::chat::ChatMessage::user()
                    .tool_result(pending_results)
                    .build(),
            );
        }
        Ok(converted)
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
        tool_choice: Option<&ToolChoice>,
        tools: Option<&[LLMToolDefinition]>,
    ) -> Result<Box<dyn llm::LLMProvider>, LLMError> {
        let mut builder =
            llm::builder::LLMBuilder::new()
                .backend(self.provider_type.to_llm_backend().ok_or_else(|| {
                    LLMError::Config("provider requires the first-party SDK".into())
                })?)
                .model(&self.model);

        if let Some(tools) = tools {
            for tool in tools {
                builder = builder.function(
                    llm::builder::FunctionBuilder::new(&tool.name)
                        .description(&tool.description)
                        .json_schema(tool.input_schema.clone()),
                );
            }
        }
        if let Some(choice) = tool_choice {
            builder = builder.tool_choice(map_tool_choice(choice)?);
        }

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
        if let Some(sp) = system_prompt
            && !sp.is_empty()
        {
            builder = builder.system(sp);
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
        if let Some(ref stops) = config.stop_sequences
            && !stops.is_empty()
        {
            tracing::debug!(
                stop_sequences = ?stops,
                "stop_sequences is not supported by the llm crate builder; ignoring"
            );
        }

        // --- Timeout (first-class, fallback to extra) ---
        let timeout = config
            .timeout_seconds
            .or_else(|| config.extra.get("timeout_seconds").and_then(|v| v.as_u64()));
        if let Some(t) = timeout {
            builder = builder.timeout_seconds(t);
        }

        // --- Extended reasoning (first-class, fallback to extra) ---
        let reasoning = config
            .reasoning
            .or_else(|| config.extra.get("reasoning").and_then(|v| v.as_bool()));
        if let Some(r) = reasoning {
            builder = builder.reasoning(r);
        }

        if let Some(effort) = legacy_reasoning_effort(config)? {
            builder = builder.reasoning_effort(effort);
        }

        // --- Reasoning budget (first-class, fallback to extra) ---
        let budget = config.reasoning_budget_tokens.or_else(|| {
            config
                .extra
                .get("reasoning_budget_tokens")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
        });
        if let Some(b) = budget {
            builder = builder.reasoning_budget_tokens(b);
        }

        // --- Normalize streaming tool call chunks ---
        if let Some(normalize) = config
            .extra
            .get("normalize_response")
            .and_then(|v| v.as_bool())
        {
            builder = builder.normalize_response(normalize);
        }

        // --- Azure OpenAI ---
        if let Some(api_version) = config.extra.get("api_version").and_then(|v| v.as_str()) {
            builder = builder.api_version(api_version);
        }
        if let Some(deployment_id) = config.extra.get("deployment_id").and_then(|v| v.as_str()) {
            builder = builder.deployment_id(deployment_id);
        }

        // --- Transport-level resilience (complementary to agent-level recovery) ---
        if let Some(resilient) = config.extra.get("resilient").and_then(|v| v.as_bool()) {
            builder = builder.resilient(resilient);
        }
        if let Some(attempts) = config
            .extra
            .get("resilient_attempts")
            .and_then(|v| v.as_u64())
        {
            builder = builder.resilient_attempts(attempts as usize);
        }
        if let Some(base_delay) = config
            .extra
            .get("resilient_base_delay_ms")
            .and_then(|v| v.as_u64())
        {
            let max_delay = config
                .extra
                .get("resilient_max_delay_ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(base_delay.saturating_mul(30));
            builder = builder.resilient_backoff(base_delay, max_delay);
        }
        if let Some(jitter) = config
            .extra
            .get("resilient_jitter")
            .and_then(|v| v.as_bool())
        {
            builder = builder.resilient_jitter(jitter);
        }

        // --- Extra body (universal escape hatch for arbitrary provider JSON) ---
        let extra_body = if self.provider_type == ProviderType::Ollama {
            merged_ollama_extra_body(config)?
        } else if self.provider_type == ProviderType::OpenRouter {
            openrouter::legacy_extra_body(config, tool_choice)?
        } else {
            config.extra.get("extra_body").cloned()
        };
        if let Some(body) = extra_body {
            builder = builder.extra_body(body);
        }

        // --- Provider-specific: OpenAI web search ---
        if matches!(
            self.provider_type,
            ProviderType::OpenAI | ProviderType::OpenAICompatible
        ) {
            if let Some(enable) = config
                .extra
                .get("openai_enable_web_search")
                .and_then(|v| v.as_bool())
            {
                builder = builder.openai_enable_web_search(enable);
            }
            if let Some(ctx) = config
                .extra
                .get("openai_web_search_context_size")
                .and_then(|v| v.as_str())
            {
                builder = builder.openai_web_search_context_size(ctx);
            }
        }

        // --- Provider-specific: XAI search ---
        if self.provider_type == ProviderType::XAI {
            if let Some(mode) = config.extra.get("xai_search_mode").and_then(|v| v.as_str()) {
                builder = builder.xai_search_mode(mode);
            }
            if let Some(max) = config
                .extra
                .get("xai_max_search_results")
                .and_then(|v| v.as_u64())
            {
                builder = builder.xai_max_search_results(max as u32);
            }
            if let Some(from) = config
                .extra
                .get("xai_search_from_date")
                .and_then(|v| v.as_str())
            {
                builder = builder.xai_search_from_date(from);
            }
            if let Some(to) = config
                .extra
                .get("xai_search_to_date")
                .and_then(|v| v.as_str())
            {
                builder = builder.xai_search_to_date(to);
            }
        }

        builder
            .build()
            .map_err(|e| LLMError::Config(format!("Failed to build LLM: {}", e)))
    }

    /// Installs an immutable audited model/endpoint contract before registry consumers capture this provider.
    /// Unsupported upstream transports remain usable without hard pricing, but cannot advertise an unverified priced capability.
    pub fn with_declared_pricing(mut self, pricing: DeclaredPricing) -> Result<Self, LLMError> {
        let endpoint = self.priced_endpoint()?;
        pricing.validate(&self.model, &endpoint)?;
        self.pricing = Some(pricing);
        Ok(self)
    }

    fn accounting_adapter(&self) -> &str {
        if matches!(self.sdk, Some(SdkBackend::Responses(_))) {
            "openai-responses"
        } else {
            self.provider_name()
        }
    }

    // Only transports whose actual endpoint selection and request-class mapping are audited participate here.
    fn priced_endpoint(&self) -> Result<String, LLMError> {
        self.require_generation()?;
        if let Some(SdkBackend::Responses(client)) = &self.sdk {
            return Ok(client.endpoint().into());
        }
        if let Some(google) = &self.google {
            return Ok(google.accounting_endpoint().to_owned());
        }
        let default = match self.provider_type {
            ProviderType::OpenAI => crate::sdk::openai::API_BASE,
            ProviderType::OpenAICompatible => "",
            ProviderType::Google => "https://generativelanguage.googleapis.com/v1beta/",
            ProviderType::Groq => "https://api.groq.com/openai/v1/",
            ProviderType::Cohere => "https://api.cohere.ai/compatibility/v1/",
            ProviderType::Mistral => "https://api.mistral.ai/v1/",
            ProviderType::OpenRouter => "https://openrouter.ai/api/v1/",
            _ => {
                return Err(LLMError::Config(
                    "this upstream adapter has no audited hard-priced dispatch contract".into(),
                ));
            }
        };
        // Match the exact normalization performed by the pinned OpenAI-compatible upstream implementation.
        let raw = self.base_url.as_deref().unwrap_or(default);
        reqwest::Url::parse(&format!("{}/", raw.trim_end_matches('/')))
            .map(|url| url.to_string())
            .map_err(|_| LLMError::Config("invalid priced endpoint".into()))
    }

    /// Public backward-compatible `build_llm` — used by tests and legacy call-sites.
    pub fn build_llm(
        &self,
        config: Option<&LLMConfig>,
    ) -> Result<Box<dyn llm::LLMProvider>, LLMError> {
        self.require_generation()?;
        if matches!(
            self.sdk,
            Some(SdkBackend::Responses(_) | SdkBackend::OllamaCloud(_))
        ) || matches!(self.sdk, Some(SdkBackend::Messages(_)))
            && !self
                .provider_type
                .is_default_anthropic(&self.default_config)
        {
            return Err(LLMError::Config(
                "configured protocol requires the first-party SDK".into(),
            ));
        }
        let cfg = config.unwrap_or(&self.default_config);
        if self
            .provider_type
            .is_default_anthropic(&self.default_config)
        {
            messages_adapter::validate_default_options(cfg)?;
        }
        if self.provider_type == ProviderType::Ollama && cfg.extra.contains_key("provider_options")
        {
            return Err(LLMError::Config(
                "Local Ollama provider_options require native chat; build_llm() only accepts legacy settings".into()
            ));
        }
        self.build_llm_with_system(cfg, None, None, None)
    }

    /// Holds the request-specific client lock through use so concurrent requests cannot replace its prompt or config after validation.
    async fn client_for_request(
        &self,
        config: Option<&LLMConfig>,
        system_prompt: Option<&str>,
        tool_choice: Option<&ToolChoice>,
        tools: Option<&[LLMToolDefinition]>,
    ) -> Result<tokio::sync::MutexGuard<'_, Option<CachedClient>>, LLMError> {
        let cfg = config.unwrap_or(&self.default_config);
        let hash = compute_config_hash(cfg, system_prompt, tool_choice, tools);
        let config_key = request_config_key(cfg, system_prompt, tool_choice, tools)?;

        let mut lock = self.client.lock().await;
        if lock
            .as_ref()
            .is_none_or(|cached| cached.config_hash != hash || cached.config_key != config_key)
        {
            let llm = self.build_llm_with_system(cfg, system_prompt, tool_choice, tools)?;
            *lock = Some(CachedClient {
                llm,
                config_hash: hash,
                config_key,
            });
        }
        Ok(lock)
    }
}

fn decision_options<'a>(request: &ChoiceRequest<'a>) -> Vec<ChoiceOption<'a>> {
    request
        .candidates
        .iter()
        .map(|option| ChoiceOption {
            id: option.id,
            description: option.description,
        })
        .collect()
}

#[async_trait]
impl LLMProvider for UnifiedLLMProvider {
    fn uses_choice_for_routing(&self) -> bool {
        matches!(self.sdk, Some(SdkBackend::Decisions(_)))
    }

    fn validate_choice(&self, request: &ChoiceRequest<'_>) -> Result<(), LLMError> {
        let options = decision_options(request);
        self.decision_client()?
            .validate(&self.decision_request(request, &options))
            .map_err(super::sdk_error)
    }

    async fn choose(&self, request: &ChoiceRequest<'_>) -> Result<ChoiceDecision, LLMError> {
        let options = decision_options(request);
        let response = self
            .decision_client()?
            .choose(&self.decision_request(request, &options))
            .await
            .map_err(super::sdk_error)?;
        Ok(ChoiceDecision {
            selected: match response.outcome {
                decisions::ChoiceOutcome::Selected { id, .. } => Some(id),
                decisions::ChoiceOutcome::Refused => None,
            },
            usage: response.usage.map(|usage| TokenUsage {
                prompt_tokens: usage.input_tokens,
                completion_tokens: usage.output_tokens,
                total_tokens: usage.total_tokens,
            }),
        })
    }

    async fn complete(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<LLMResponse, LLMError> {
        self.require_generation()?;
        if let Some(google) = &self.google {
            return google.complete(messages, config).await;
        }
        if let Some(SdkBackend::OllamaLocal(client)) = &self.sdk
            && let Some(request) = self.local_ollama_request(messages, config)?
        {
            return ollama_local::complete(client, &request).await;
        }
        if let Some(SdkBackend::OllamaCloud(client)) = &self.sdk {
            self.validate_configured_tool_choice()?;
            return ollama_adapter::complete(
                client,
                &self.model,
                messages,
                config.unwrap_or(&self.default_config),
                None,
            )
            .await;
        }
        if let Some((client, settings)) = self.messages_dispatch(config, messages)? {
            self.validate_configured_tool_choice()?;
            return messages_adapter::complete(client, &self.model, messages, settings, None).await;
        }
        if let Some(SdkBackend::Responses(client)) = &self.sdk {
            return responses_adapter::complete(
                client,
                &self.model,
                messages,
                config.unwrap_or(&self.default_config),
                None,
            )
            .await;
        }
        if let Some(SdkBackend::OpenRouter(client)) = &self.sdk
            && let Some(wire) = openrouter::request(
                client,
                &self.model,
                messages,
                config.unwrap_or(&self.default_config),
                None,
            )?
        {
            return openrouter::complete(client, &wire).await;
        }
        if let Some(SdkBackend::Chat(chat)) = &self.sdk
            && let Some(wire) = chat_adapter::request(
                self.provider_type,
                &self.model,
                messages,
                config.unwrap_or(&self.default_config),
                None,
            )?
        {
            return chat_adapter::response(
                chat.complete(&wire).await.map_err(super::sdk_error)?,
                &self.model,
            );
        }
        let projected_messages = Self::project_foreign_messages(messages)?;
        // Separate system messages from non-system messages
        let (system_prompt, non_system_msgs) = extract_system_and_messages(&projected_messages);

        let llm_messages: Vec<llm::chat::ChatMessage> = non_system_msgs
            .iter()
            .map(|m| self.convert_message(m))
            .collect();

        let lock = self
            .client_for_request(config, system_prompt.as_deref(), None, None)
            .await?;
        let cached = lock.as_ref().expect("client must be built under this lock");

        let response = cached
            .llm
            .chat(&llm_messages)
            .await
            .map_err(|e| self.map_llm_crate_error(e))?;

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

    async fn complete_with_tools(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
        request: &LLMToolRequest,
    ) -> Result<LLMResponse, LLMError> {
        self.require_generation()?;
        if let Some(google) = &self.google {
            return google.complete_with_tools(messages, config, request).await;
        }
        let choice = &request.choice;
        if !self.supports_choice_for_provider(choice) {
            return Err(LLMError::Config(format!(
                "provider '{}' does not support native tool choice {:?}",
                self.provider_name(),
                choice
            )));
        }
        if matches!(choice, ToolChoice::Required | ToolChoice::Specific(_))
            && request.tools.is_empty()
        {
            return Err(LLMError::Config(
                "required native tool choice needs at least one tool definition".to_string(),
            ));
        }
        if let ToolChoice::Specific(name) = choice
            && !request.tools.iter().any(|tool| tool.name == *name)
        {
            return Err(LLMError::Config(format!(
                "specific native tool choice '{name}' is not present in the request"
            )));
        }

        if let Some(SdkBackend::OllamaCloud(client)) = &self.sdk {
            self.validate_configured_tool_choice()?;
            return ollama_adapter::complete(
                client,
                &self.model,
                messages,
                config.unwrap_or(&self.default_config),
                Some(request),
            )
            .await;
        }
        if let Some((client, settings)) = self.messages_dispatch(config, messages)? {
            self.validate_configured_tool_choice()?;
            return messages_adapter::complete(
                client,
                &self.model,
                messages,
                settings,
                Some(request),
            )
            .await;
        }
        if let Some(SdkBackend::Responses(client)) = &self.sdk {
            return responses_adapter::complete(
                client,
                &self.model,
                messages,
                config.unwrap_or(&self.default_config),
                Some(request),
            )
            .await;
        }
        if let Some(SdkBackend::OpenRouter(client)) = &self.sdk
            && let Some(wire) = openrouter::request(
                client,
                &self.model,
                messages,
                config.unwrap_or(&self.default_config),
                Some(request),
            )?
        {
            return openrouter::complete(client, &wire).await;
        }
        if let Some(SdkBackend::Chat(chat)) = &self.sdk
            && let Some(wire) = chat_adapter::request(
                self.provider_type,
                &self.model,
                messages,
                config.unwrap_or(&self.default_config),
                Some(request),
            )?
        {
            return chat_adapter::response(
                chat.complete(&wire).await.map_err(super::sdk_error)?,
                &self.model,
            );
        }
        let projected_messages = Self::project_foreign_messages(messages)?;
        let (system_prompt, non_system_msgs) = extract_system_and_messages(&projected_messages);
        let llm_messages = self.convert_messages_with_tools(&non_system_msgs)?;
        let tools = request
            .tools
            .iter()
            .map(map_tool_definition)
            .collect::<Vec<_>>();

        let lock = self
            .client_for_request(
                config,
                system_prompt.as_deref(),
                Some(choice),
                Some(&request.tools),
            )
            .await?;
        let cached = lock.as_ref().expect("client must be built under this lock");
        let response = cached
            .llm
            .chat_with_tools(&llm_messages, Some(&tools))
            .await
            .map_err(|error| self.map_llm_crate_error(error))?;

        let calls = normalize_tool_calls(response.tool_calls(), choice)?;
        let usage = response.usage().map(|usage| TokenUsage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
        });
        let mut normalized_response = LLMResponse {
            content: response.text().unwrap_or_default(),
            finish_reason: if calls.is_empty() {
                FinishReason::Stop
            } else {
                FinishReason::ToolCall
            },
            usage,
            model: Some(self.model.clone()),
            metadata: HashMap::new(),
        };
        if !calls.is_empty() {
            normalized_response.set_tool_calls(calls)?;
        }
        Ok(normalized_response)
    }

    async fn complete_stream(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError>
    {
        self.require_generation()?;
        if let Some(google) = &self.google {
            return google.complete_stream(messages, config).await;
        }
        if let Some(SdkBackend::OllamaLocal(client)) = &self.sdk
            && let Some(request) = self.local_ollama_request(messages, config)?
        {
            return Ok(Box::new(ollama_local::stream(client, &request).await?));
        }
        if let Some(SdkBackend::OllamaCloud(client)) = &self.sdk {
            self.validate_configured_tool_choice()?;
            return ollama_adapter::stream(
                client,
                &self.model,
                messages,
                config.unwrap_or(&self.default_config),
            )
            .await;
        }
        if let Some((client, settings)) = self.messages_dispatch(config, messages)? {
            self.validate_configured_tool_choice()?;
            return messages_adapter::stream(client, &self.model, messages, settings).await;
        }
        if let Some(SdkBackend::Responses(client)) = &self.sdk {
            return responses_adapter::stream(
                client,
                &self.model,
                messages,
                config.unwrap_or(&self.default_config),
            )
            .await;
        }
        if let Some(SdkBackend::OpenRouter(client)) = &self.sdk
            && let Some(wire) = openrouter::request(
                client,
                &self.model,
                messages,
                config.unwrap_or(&self.default_config),
                None,
            )?
        {
            return openrouter::stream(
                client,
                &wire,
                self.pricing.as_ref().map(|pricing| pricing.model.clone()),
            )
            .await;
        }
        if let Some(SdkBackend::Chat(chat)) = &self.sdk
            && let Some(wire) = chat_adapter::request(
                self.provider_type,
                &self.model,
                messages,
                config.unwrap_or(&self.default_config),
                None,
            )?
        {
            return Ok(Box::new(Box::pin(
                chat.stream(&wire)
                    .await
                    .map_err(super::sdk_error)?
                    .map(chat_adapter::chunk),
            )));
        }
        let projected_messages = Self::project_foreign_messages(messages)?;
        // Separate system messages from non-system messages
        let (system_prompt, non_system_msgs) = extract_system_and_messages(&projected_messages);

        let llm_messages: Vec<llm::chat::ChatMessage> = non_system_msgs
            .iter()
            .map(|m| self.convert_message(m))
            .collect();

        // Keep the selected client stable until stream creation returns its owned stream.
        let stream = {
            let lock = self
                .client_for_request(config, system_prompt.as_deref(), None, None)
                .await?;
            let cached = lock.as_ref().expect("client must be built under this lock");
            cached
                .llm
                .chat_stream(&llm_messages)
                .await
                .map_err(|e| self.map_llm_crate_error(e))?
        };

        let provider_type = self.provider_type;
        let model = self.model.clone();
        let base_url = self.base_url.clone();
        let converted_stream = stream.map(move |result| {
            result
                .map(|token| LLMChunk::new(token, false))
                .map_err(|e| map_provider_error(provider_type, &model, base_url.as_deref(), e))
        });

        // Chain a final sentinel chunk so consumers know the stream is done
        let final_stream = converted_stream.chain(futures::stream::once(async {
            Ok(LLMChunk::final_chunk("", FinishReason::Stop, None))
        }));

        Ok(Box::new(Box::pin(final_stream)))
    }

    fn priced_capability_identity(&self) -> Option<String> {
        self.pricing
            .as_ref()
            .map(|pricing| pricing.identity(self.accounting_adapter()))
    }

    fn request_cost_bound(
        &self,
        request: &ProviderRequest<'_>,
    ) -> Result<Option<ProviderCostBound>, LLMError> {
        let Some(pricing) = &self.pricing else {
            return Ok(None);
        };
        // Run the same conversion and local protocol validation used by dispatch before admitting any paid attempt.
        if let Some(SdkBackend::Responses(client)) = &self.sdk {
            let cfg = request.config.unwrap_or(&self.default_config);
            let wire = responses_adapter::request(
                client,
                &self.model,
                request.messages,
                cfg,
                request.tools,
            )?;
            client
                .body(&wire, request.streaming)
                .map_err(super::sdk_error)?;
            if wire.reasoning_mode.is_some()
                || request
                    .tools
                    .is_some_and(|t| !self.supports_choice_for_provider(&t.choice))
            {
                return Err(LLMError::Config(
                    "unsupported priced Responses request".into(),
                ));
            }
            let mut effective = cfg.clone();
            effective.extra.remove("protocol");
            effective.extra.remove("provider_options");
            effective.reasoning_effort = wire.reasoning_effort;
            let priced = ProviderRequest {
                config: Some(&effective),
                ..*request
            };
            return pricing
                .quote(self.accounting_adapter(), &priced, &effective)
                .map(Some);
        }
        if let Some(SdkBackend::OpenRouter(client)) = &self.sdk {
            let cfg = request.config.unwrap_or(&self.default_config);
            if let Some(wire) =
                openrouter::request(client, &self.model, request.messages, cfg, request.tools)?
            {
                if request
                    .tools
                    .is_some_and(|tools| !self.supports_choice_for_provider(&tools.choice))
                    || wire.parallel_tool_calls.is_some()
                    || !wire.chat.provider_options.is_empty()
                    || wire.chat.response_format.is_some()
                {
                    return Err(LLMError::Config(
                        "unbounded OpenRouter request under declared pricing".into(),
                    ));
                }
                client
                    .body(&wire, request.streaming)
                    .map_err(super::sdk_error)?;
                let mut effective = cfg.clone();
                effective.extra.remove("protocol");
                effective.extra.remove("provider_options");
                let priced = ProviderRequest {
                    config: Some(&effective),
                    ..*request
                };
                return pricing
                    .quote(self.accounting_adapter(), &priced, &effective)
                    .map(Some);
            }
        }
        if let Some(google) = &self.google {
            google.validate_priced_request(request)?;
        } else {
            if matches!(self.sdk, Some(SdkBackend::Chat(_))) {
                let cfg = request.config.unwrap_or(&self.default_config);
                if let Some(wire) = chat_adapter::request(
                    self.provider_type,
                    &self.model,
                    request.messages,
                    cfg,
                    request.tools,
                )? {
                    if request
                        .tools
                        .is_some_and(|t| !self.supports_choice_for_provider(&t.choice))
                    {
                        return Err(LLMError::Config(
                            "unsupported priced native tool choice".into(),
                        ));
                    }
                    wire.body(request.streaming).map_err(super::sdk_error)?;
                    // Only extensions with the same audited semantics as a typed setting can be quoted.
                    let mut effective = cfg.clone();
                    effective.extra.remove("protocol");
                    if let Some(serde_json::Value::Object(mut options)) =
                        effective.extra.remove("provider_options")
                    {
                        if let Some(serde_json::Value::String(effort)) =
                            options.remove("reasoning_effort")
                        {
                            effective.reasoning_effort = Some(effort);
                        }
                        if !options.is_empty() {
                            return Err(LLMError::Config("unbounded Chat provider option".into()));
                        }
                    }
                    let priced = ProviderRequest {
                        config: Some(&effective),
                        ..*request
                    };
                    return pricing
                        .quote(self.accounting_adapter(), &priced, &effective)
                        .map(Some);
                }
            }
            let projected = Self::project_foreign_messages(request.messages)?;
            let (_, non_system) = extract_system_and_messages(&projected);
            if let Some(tools) = request.tools {
                self.convert_messages_with_tools(&non_system)?;
                if !self.supports_choice_for_provider(&tools.choice)
                    || (matches!(tools.choice, ToolChoice::Required | ToolChoice::Specific(_))
                        && tools.tools.is_empty())
                    || matches!(&tools.choice, ToolChoice::Specific(name) if !tools.tools.iter().any(|tool| &tool.name == name))
                {
                    return Err(LLMError::Config(
                        "unsupported priced native tool choice".into(),
                    ));
                }
            } else {
                for message in non_system {
                    self.convert_message(message);
                }
            }
        }
        pricing
            .quote(self.accounting_adapter(), request, &self.default_config)
            .map(Some)
    }

    fn settle_request_cost(
        &self,
        bound: &ProviderCostBound,
        response: &LLMResponse,
    ) -> Result<Option<ProviderCostSettlement>, LLMError> {
        let Some(pricing) = &self.pricing else {
            return Ok(None);
        };
        if self.provider_type == ProviderType::OpenRouter
            && response
                .metadata
                .get(openrouter::REPORTED_MODEL)
                .and_then(serde_json::Value::as_str)
                != Some(pricing.model.as_str())
        {
            return Ok(None);
        }
        pricing.settle(self.accounting_adapter(), bound, response)
    }

    fn settle_stream_request_cost(
        &self,
        bound: &ProviderCostBound,
        usage: Option<&TokenUsage>,
    ) -> Result<Option<ProviderCostSettlement>, LLMError> {
        let Some(pricing) = &self.pricing else {
            return Ok(None);
        };
        // First-party codecs attach complete usage only to an observed terminal event.
        if self.google.is_none() && self.sdk.is_none() {
            return Ok(None);
        }
        let Some(usage) = usage else {
            return Ok(None);
        };
        let mut response = LLMResponse::new("", FinishReason::Stop);
        response.usage = Some(*usage);
        response.metadata.insert(
            "_ai_agents_complete_priced_usage".into(),
            serde_json::Value::Bool(true),
        );
        pricing.settle(self.accounting_adapter(), bound, &response)
    }

    fn provider_name(&self) -> &str {
        match self.provider_type {
            ProviderType::OpenAI => "openai",
            ProviderType::Anthropic => "anthropic",
            ProviderType::AnthropicCompatible => "anthropic-compatible",
            ProviderType::Ollama => "ollama",
            ProviderType::OllamaCloud => ollama_adapter::PROVIDER,
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

    fn allows_prompt_tools(&self) -> bool {
        self.provider_type.allows_prompt_tools()
    }

    fn configured_tool_choice(&self) -> Option<ToolChoice> {
        self.tool_choice.clone()
    }

    fn supports_tool_choice(&self, choice: &ToolChoice) -> bool {
        self.provider_type.supports_native_tool_choice(
            &self.model,
            &self.default_config,
            self.feature_overrides
                .get(&LLMFeature::FunctionCalling)
                .copied(),
            choice,
        )
    }

    fn supports(&self, feature: LLMFeature) -> bool {
        if self.provider_type == ProviderType::OllamaCloud {
            return match feature {
                LLMFeature::FunctionCalling => self.supports_tool_choice(&ToolChoice::Auto),
                LLMFeature::Streaming | LLMFeature::SystemMessages => true,
                _ => false,
            };
        }
        if self.uses_choice_for_routing() {
            return false;
        }
        if matches!(feature, LLMFeature::FunctionCalling)
            && matches!(self.sdk, Some(SdkBackend::Messages(_)))
            && !self
                .provider_type
                .is_default_anthropic(&self.default_config)
        {
            return self.supports_tool_choice(&ToolChoice::Auto);
        }
        if matches!(
            self.sdk,
            Some(SdkBackend::Responses(_) | SdkBackend::Messages(_))
        ) && !self
            .provider_type
            .is_default_anthropic(&self.default_config)
            && matches!(feature, LLMFeature::Vision | LLMFeature::JsonMode)
        {
            return false; // The first framework Responses path accepts text and native functions.
        }
        if feature == LLMFeature::FunctionCalling && !self.model_allows_tools() {
            return false;
        }
        if let Some(enabled) = self.feature_overrides.get(&feature) {
            return *enabled;
        }

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

    fn is_terminal_error(&self, error: &LLMError) -> bool {
        // A failed Cloud exchange can have reached the remote service.
        if self.provider_type == ProviderType::OllamaCloud {
            return true;
        }
        if self.sdk.is_some() {
            return matches!(
                error,
                LLMError::Serialization(_) | LLMError::Config(_) | LLMError::ContentFiltered(_)
            ) || matches!(error,LLMError::API{status:Some(s),..} if (300..500).contains(s) && !matches!(s,408|429));
        }
        self.google
            .as_ref()
            .is_some_and(|google| google.is_terminal_error(error))
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
            provider_type.check_credential_destination(Some(&env_var), self.base_url.as_deref())?;
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
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn ollama_cloud_default_credential_never_resolves_at_a_custom_origin() {
        let base = "http://127.0.0.1:1/api";
        let error = ProviderType::OllamaCloud
            .configured_api_key_at(Some("OLLAMA_API_KEY"), None, Some(base))
            .unwrap_err();
        assert!(error.to_string().contains("bound to the official"));
        let error = UnifiedLLMProvider::new(
            ProviderType::OllamaCloud,
            "fixture".into(),
            None,
            Some(base.into()),
        )
        .unwrap_err();
        assert!(error.to_string().contains("explicitly selected"));
        assert!(
            UnifiedLLMProvider::new(
                ProviderType::OllamaCloud,
                "fixture".into(),
                Some("placeholder".into()),
                None,
            )
            .is_ok()
        );
    }

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
    fn messages_review_compatible_builder_requires_an_explicit_ceiling() {
        let result = ProviderBuilder::new()
            .provider(ProviderType::AnthropicCompatible)
            .base_url("http://localhost/v1")
            .model("local")
            .build();
        assert!(matches!(result, Err(LLMError::Config(message)) if message.contains("max_tokens")));
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
    fn ollama_local_raw_builder_checks_effective_native_options() {
        let raw = LLMConfig::default();
        let native = raw.clone().with_extra(
            "provider_options",
            serde_json::json!({"options": {"num_predict": 7}}),
        );
        let provider = UnifiedLLMProvider::from_spec_config(
            ProviderType::Ollama,
            "fixture",
            None,
            None,
            native.clone(),
        )
        .unwrap();
        for cfg in [None, Some(&native)] {
            assert!(
                matches!(provider.build_llm(cfg), Err(LLMError::Config(message)) if message.contains("provider_options"))
            );
        }
        assert!(provider.build_llm(Some(&raw)).is_ok());
    }

    #[test]
    fn test_build_llm_accepts_reasoning_effort_low() {
        let provider = ProviderBuilder::new()
            .provider(ProviderType::OpenAI)
            .model("gpt-5.1-mini")
            .api_key("XXXXXXXXXX")
            .build()
            .unwrap();

        let config = LLMConfig {
            reasoning_effort: Some("low".to_string()),
            ..LLMConfig::default()
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

        let config = LLMConfig {
            reasoning_effort: Some("invalid".to_string()),
            ..LLMConfig::default()
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
            ..LLMConfig::default()
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

    // Serves two bounded, local OpenAI responses and captures the requests actually dispatched.
    async fn local_chat_fixture() -> (
        String,
        tokio::sync::mpsc::UnboundedReceiver<serde_json::Value>,
    ) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            for _ in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let header_end = loop {
                    let mut buffer = [0; 4096];
                    let size = socket.read(&mut buffer).await.unwrap();
                    assert!(size > 0, "request ended before HTTP headers");
                    request.extend_from_slice(&buffer[..size]);
                    if let Some(index) = request.windows(4).position(|part| part == b"\r\n\r\n") {
                        break index + 4;
                    }
                };
                let headers = std::str::from_utf8(&request[..header_end]).unwrap();
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap();
                while request.len() - header_end < content_length {
                    let mut buffer = [0; 4096];
                    let size = socket.read(&mut buffer).await.unwrap();
                    assert!(size > 0, "request ended before HTTP body");
                    request.extend_from_slice(&buffer[..size]);
                }
                let body: serde_json::Value =
                    serde_json::from_slice(&request[header_end..header_end + content_length])
                        .unwrap();
                let streaming = body["stream"] == true;
                sender.send(body).unwrap();
                let content_type = if streaming {
                    "text/event-stream"
                } else {
                    "application/json"
                };
                let response_body = if streaming {
                    "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
                } else {
                    "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}"
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
                    response_body.len()
                );
                socket.write_all(response.as_bytes()).await.unwrap();
                socket.shutdown().await.unwrap();
            }
        });
        (format!("http://{address}/v1/"), receiver)
    }

    // Concurrent calls share a transport, but each wire payload must keep its own prompt and options.
    #[tokio::test]
    async fn concurrent_requests_dispatch_with_their_own_client_config() {
        async fn dispatch(provider: &UnifiedLLMProvider, kind: &str, name: &str, max_tokens: u32) {
            let messages = [
                ChatMessage::system(format!("system {name}")),
                ChatMessage::user("hi"),
            ];
            let config = LLMConfig {
                max_tokens: Some(max_tokens),
                ..LLMConfig::default()
            };
            match kind {
                "complete" => {
                    provider.complete(&messages, Some(&config)).await.unwrap();
                }
                "native" => {
                    let request = LLMToolRequest {
                        choice: ToolChoice::Auto,
                        tools: vec![LLMToolDefinition {
                            name: name.to_string(),
                            description: "Test tool".to_string(),
                            input_schema: serde_json::json!({"type": "object"}),
                        }],
                    };
                    provider
                        .complete_with_tools(&messages, Some(&config), &request)
                        .await
                        .unwrap();
                }
                "stream" => {
                    let mut stream = provider
                        .complete_stream(&messages, Some(&config))
                        .await
                        .unwrap();
                    while let Some(chunk) = stream.next().await {
                        chunk.unwrap();
                    }
                }
                _ => unreachable!(),
            }
        }

        for kind in ["complete", "native", "stream"] {
            let (base_url, mut received) = local_chat_fixture().await;
            let provider = ProviderBuilder::new()
                .provider(ProviderType::OpenAI)
                .model("test-model")
                .api_key("test-key")
                .base_url(base_url)
                .build()
                .unwrap();

            let mut first = Box::pin(dispatch(&provider, kind, "first", 20));
            let mut second = Box::pin(dispatch(&provider, kind, "second", 200));
            assert!(futures::poll!(first.as_mut()).is_pending());
            assert!(futures::poll!(second.as_mut()).is_pending());

            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                tokio::join!(first, second);
                let mut requests = [
                    received.recv().await.unwrap(),
                    received.recv().await.unwrap(),
                ];
                requests.sort_by_key(|body| body["max_completion_tokens"].as_u64());
                for (body, (name, max_tokens)) in
                    requests.into_iter().zip([("first", 20), ("second", 200)])
                {
                    assert!(body.to_string().contains(&format!("system {name}")));
                    assert_eq!(body["max_completion_tokens"], max_tokens);
                    if kind == "native" {
                        assert_eq!(body["tools"][0]["function"]["name"], name);
                        assert_eq!(body["tool_choice"], "auto");
                    }
                }
            })
            .await
            .expect("local provider fixture did not complete");
        }
    }

    #[tokio::test]
    async fn prepared_client_stays_bound_to_its_request_until_use() {
        let provider = ProviderBuilder::new()
            .provider(ProviderType::OpenAI)
            .model("test-model")
            .api_key("test-key")
            .build()
            .unwrap();
        let first = LLMConfig {
            max_tokens: Some(20),
            ..LLMConfig::default()
        };
        let second = LLMConfig {
            max_tokens: Some(200),
            ..LLMConfig::default()
        };
        let tools = [LLMToolDefinition {
            name: "lookup".to_string(),
            description: "Look up data".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let expected = compute_config_hash(
            &first,
            Some("first system prompt"),
            Some(&ToolChoice::Auto),
            Some(&tools),
        );
        let first_client = provider
            .client_for_request(
                Some(&first),
                Some("first system prompt"),
                Some(&ToolChoice::Auto),
                Some(&tools),
            )
            .await
            .unwrap();
        let mut replacement = Box::pin(provider.client_for_request(
            Some(&second),
            Some("second system prompt"),
            None,
            None,
        ));
        assert!(futures::poll!(replacement.as_mut()).is_pending());
        assert_eq!(first_client.as_ref().unwrap().config_hash, expected);
        drop(first_client);

        let second_client = replacement.await.unwrap();
        let expected_second =
            compute_config_hash(&second, Some("second system prompt"), None, None);
        assert_eq!(second_client.as_ref().unwrap().config_hash, expected_second);
    }

    #[test]
    fn test_config_hash_stability() {
        let config = LLMConfig::default();

        let hash1 = compute_config_hash(&config, Some("system prompt"), None, None);
        let hash2 = compute_config_hash(&config, Some("system prompt"), None, None);
        assert_eq!(hash1, hash2);

        let hash3 = compute_config_hash(&config, Some("different prompt"), None, None);
        assert_ne!(hash1, hash3);

        let hash4 = compute_config_hash(&config, None, None, None);
        assert_ne!(hash1, hash4);
    }

    #[test]
    fn test_build_llm_forwards_timeout() {
        let provider = ProviderBuilder::new()
            .provider(ProviderType::OpenAI)
            .model("gpt-4")
            .api_key("XXXXXXXXXX")
            .build()
            .unwrap();

        let config = LLMConfig::default().with_timeout_seconds(120);
        let result = provider.build_llm(Some(&config));
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_llm_forwards_reasoning_and_budget() {
        let provider = ProviderBuilder::new()
            .provider(ProviderType::OpenAI)
            .model("o3")
            .api_key("XXXXXXXXXX")
            .build()
            .unwrap();

        let config = LLMConfig {
            reasoning: Some(true),
            reasoning_effort: Some("high".to_string()),
            reasoning_budget_tokens: Some(16384),
            ..LLMConfig::default()
        };

        let result = provider.build_llm(Some(&config));
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_llm_forwards_resilient_settings() {
        let provider = ProviderBuilder::new()
            .provider(ProviderType::OpenAI)
            .model("gpt-4")
            .api_key("XXXXXXXXXX")
            .build()
            .unwrap();

        let config = LLMConfig::default()
            .with_extra("resilient", serde_json::json!(true))
            .with_extra("resilient_attempts", serde_json::json!(3))
            .with_extra("resilient_base_delay_ms", serde_json::json!(1000))
            .with_extra("resilient_max_delay_ms", serde_json::json!(30000))
            .with_extra("resilient_jitter", serde_json::json!(true));

        let result = provider.build_llm(Some(&config));
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_llm_forwards_azure_fields() {
        let provider = ProviderBuilder::new()
            .provider(ProviderType::OpenAI)
            .model("gpt-4")
            .api_key("XXXXXXXXXX")
            .build()
            .unwrap();

        let config = LLMConfig::default()
            .with_extra("api_version", serde_json::json!("2024-06-01"))
            .with_extra("deployment_id", serde_json::json!("my-gpt4-deploy"));

        let result = provider.build_llm(Some(&config));
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_llm_forwards_extra_body() {
        let provider = ProviderBuilder::new()
            .provider(ProviderType::OpenAI)
            .model("gpt-4")
            .api_key("XXXXXXXXXX")
            .build()
            .unwrap();

        let config =
            LLMConfig::default().with_extra("extra_body", serde_json::json!({"logprobs": true}));

        let result = provider.build_llm(Some(&config));
        assert!(result.is_ok());
    }

    #[test]
    fn test_reasoning_effort_extra_fallback() {
        let provider = ProviderBuilder::new()
            .provider(ProviderType::OpenAI)
            .model("o3")
            .api_key("XXXXXXXXXX")
            .build()
            .unwrap();

        // reasoning_effort via extra (backward compat)
        let config =
            LLMConfig::default().with_extra("reasoning_effort", serde_json::json!("medium"));
        let result = provider.build_llm(Some(&config));
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_hash_changes_with_forwarded_extra() {
        let config_a = LLMConfig::default();
        let config_b = LLMConfig::default().with_timeout_seconds(120);

        let hash_a = compute_config_hash(&config_a, None, None, None);
        let hash_b = compute_config_hash(&config_b, None, None, None);
        assert_ne!(hash_a, hash_b);
    }

    // Legacy extra settings are actual builder inputs and must participate in exact client selection.
    #[tokio::test]
    async fn extra_fallback_settings_remain_bound_to_the_actual_cached_client() {
        let provider = ProviderBuilder::new()
            .provider(ProviderType::OpenAI)
            .model("o3")
            .api_key("fixture")
            .build()
            .unwrap();
        let initial = LLMConfig::default();
        let expected_initial = request_config_key(&initial, None, None, None).unwrap();
        for (key, value) in [
            ("timeout_seconds", serde_json::json!(120)),
            ("reasoning", serde_json::json!(true)),
            ("reasoning_effort", serde_json::json!("high")),
            ("reasoning_budget_tokens", serde_json::json!(4096)),
        ] {
            let config = initial.clone().with_extra(key, value);
            let client = provider
                .client_for_request(Some(&config), None, None, None)
                .await
                .unwrap();
            let expected = request_config_key(&config, None, None, None).unwrap();
            assert_ne!(expected, expected_initial);
            assert_eq!(client.as_ref().unwrap().config_key, expected);
            assert_ne!(
                compute_config_hash(&initial, None, None, None),
                compute_config_hash(&config, None, None, None)
            );
        }
    }

    // Flattened extras cannot hide a differing authoritative typed field in an exact cache key.
    #[test]
    fn request_key_distinguishes_typed_fields_from_shadowing_extras() {
        let first = LLMConfig {
            reasoning: Some(false),
            ..LLMConfig::default()
        }
        .with_extra("reasoning", serde_json::json!(true));
        let second = LLMConfig {
            reasoning: Some(true),
            ..first.clone()
        };
        assert_ne!(
            request_config_key(&first, None, None, None).unwrap(),
            request_config_key(&second, None, None, None).unwrap()
        );
    }

    #[test]
    fn test_config_hash_changes_with_reasoning_fields() {
        let config_a = LLMConfig::default();
        let config_b = LLMConfig {
            reasoning: Some(true),
            reasoning_effort: Some("high".to_string()),
            ..LLMConfig::default()
        };
        let config_c = LLMConfig::default().with_extra("resilient", serde_json::json!(true));

        let ha = compute_config_hash(&config_a, None, None, None);
        let hb = compute_config_hash(&config_b, None, None, None);
        let hc = compute_config_hash(&config_c, None, None, None);

        assert_ne!(ha, hb);
        assert_ne!(ha, hc);
        assert_ne!(hb, hc);
    }

    #[test]
    fn test_feature_override_enables_function_calling() {
        let provider = UnifiedLLMProvider::from_spec_config(
            ProviderType::OpenAICompatible,
            "local-model",
            None,
            Some("http://localhost:1234/v1".to_string()),
            LLMConfig::default(),
        )
        .unwrap()
        .with_feature_override(LLMFeature::FunctionCalling, true);

        assert!(provider.supports(LLMFeature::FunctionCalling));
    }

    #[test]
    fn test_feature_override_disables_vision() {
        let provider = UnifiedLLMProvider::from_spec_config(
            ProviderType::OpenAI,
            "gpt-4o-mini",
            Some("XXXXXXXXXX".to_string()),
            None,
            LLMConfig::default(),
        )
        .unwrap()
        .with_feature_override(LLMFeature::Vision, false);

        assert!(!provider.supports(LLMFeature::Vision));
    }

    #[test]
    fn test_ollama_num_ctx_in_extra_body() {
        let config = LLMConfig::default().with_extra("num_ctx", serde_json::json!(8192));
        let body = merged_ollama_extra_body(&config).unwrap().unwrap();

        assert_eq!(body["options"]["num_ctx"], serde_json::json!(8192));
    }

    #[test]
    fn test_ollama_build_llm_includes_num_ctx() {
        let provider = ProviderBuilder::new()
            .provider(ProviderType::Ollama)
            .model("llama3.1")
            .build()
            .unwrap();

        let config = LLMConfig::default().with_extra("num_ctx", serde_json::json!(8192));
        let result = provider.build_llm(Some(&config));
        assert!(result.is_ok());
    }

    #[test]
    fn test_ollama_extra_body_merge_preserves_user_fields() {
        let config = LLMConfig::default()
            .with_extra(
                "extra_body",
                serde_json::json!({
                    "format": "json",
                    "options": { "temperature": 0.2 }
                }),
            )
            .with_extra("num_ctx", serde_json::json!(8192));

        let body = merged_ollama_extra_body(&config).unwrap().unwrap();
        assert_eq!(body["format"], serde_json::json!("json"));
        assert_eq!(body["options"]["temperature"], serde_json::json!(0.2));
        assert_eq!(body["options"]["num_ctx"], serde_json::json!(8192));
    }

    #[test]
    fn test_ollama_extra_body_named_fields_win() {
        let config = LLMConfig::default()
            .with_extra(
                "extra_body",
                serde_json::json!({
                    "keep_alive": "1m",
                    "options": { "num_ctx": 2048, "num_gpu": 0 }
                }),
            )
            .with_extra("num_ctx", serde_json::json!(8192))
            .with_extra("num_gpu", serde_json::json!(-1))
            .with_extra("keep_alive", serde_json::json!("5m"));

        let body = merged_ollama_extra_body(&config).unwrap().unwrap();
        assert_eq!(body["keep_alive"], serde_json::json!("5m"));
        assert_eq!(body["options"]["num_ctx"], serde_json::json!(8192));
        assert_eq!(body["options"]["num_gpu"], serde_json::json!(-1));
    }

    #[test]
    fn test_ollama_extra_body_rejects_non_object_when_merging() {
        let config = LLMConfig::default()
            .with_extra("extra_body", serde_json::json!(true))
            .with_extra("num_ctx", serde_json::json!(8192));

        let result = merged_ollama_extra_body(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("extra_body"));
    }

    #[test]
    fn test_ollama_extra_body_rejects_non_object_options() {
        let config = LLMConfig::default()
            .with_extra("extra_body", serde_json::json!({ "options": true }))
            .with_extra("num_ctx", serde_json::json!(8192));

        let result = merged_ollama_extra_body(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("options"));
    }

    #[test]
    fn test_ollama_config_hash_changes_with_named_fields() {
        let config_a = LLMConfig::default();
        let config_b = LLMConfig::default().with_extra("num_ctx", serde_json::json!(8192));
        let config_c = LLMConfig::default().with_extra("keep_alive", serde_json::json!("5m"));

        let hash_a = compute_config_hash(&config_a, None, None, None);
        let hash_b = compute_config_hash(&config_b, None, None, None);
        let hash_c = compute_config_hash(&config_c, None, None, None);

        assert_ne!(hash_a, hash_b);
        assert_ne!(hash_a, hash_c);
        assert_ne!(hash_b, hash_c);
    }

    #[test]
    fn test_ollama_error_connection_refused() {
        let provider = UnifiedLLMProvider::from_spec_config(
            ProviderType::Ollama,
            "llama3.1",
            None,
            None,
            LLMConfig::default(),
        )
        .unwrap();

        let err = provider.map_llm_crate_error("connection refused");
        assert!(err.to_string().contains("ollama serve"));
    }

    #[test]
    fn test_ollama_error_model_not_found() {
        let provider = UnifiedLLMProvider::from_spec_config(
            ProviderType::Ollama,
            "llama3.1",
            None,
            None,
            LLMConfig::default(),
        )
        .unwrap();

        let err = provider.map_llm_crate_error("model not found");
        let msg = err.to_string();
        assert!(msg.contains("ollama pull"));
        assert!(msg.contains("llama3.1"));
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

    fn test_provider(provider_type: ProviderType) -> UnifiedLLMProvider {
        let base_url = (provider_type == ProviderType::OpenAICompatible)
            .then(|| "http://localhost:1234/v1".to_string());
        UnifiedLLMProvider::from_spec_config(
            provider_type,
            "test-model",
            Some("test-key".to_string()),
            base_url,
            LLMConfig::default(),
        )
        .unwrap()
    }

    fn upstream_tool_call(id: &str, name: &str, arguments: &str) -> llm::ToolCall {
        llm::ToolCall {
            id: id.to_string(),
            call_type: "function".to_string(),
            function: llm::FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    #[test]
    fn test_maps_tool_definition_and_choices() {
        let definition = LLMToolDefinition {
            name: "calculator".to_string(),
            description: "Calculate an expression".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let mapped = map_tool_definition(&definition);
        assert_eq!(mapped.tool_type, "function");
        assert_eq!(mapped.function.name, definition.name);
        assert_eq!(mapped.function.description, definition.description);
        assert_eq!(mapped.function.parameters, definition.input_schema);

        assert!(matches!(
            map_tool_choice(&ToolChoice::Auto).unwrap(),
            llm::chat::ToolChoice::Auto
        ));
        assert!(matches!(
            map_tool_choice(&ToolChoice::Required).unwrap(),
            llm::chat::ToolChoice::Any
        ));
        assert!(matches!(
            map_tool_choice(&ToolChoice::Specific("calculator".to_string())).unwrap(),
            llm::chat::ToolChoice::Tool(name) if name == "calculator"
        ));
        assert!(matches!(
            map_tool_choice(&ToolChoice::None).unwrap(),
            llm::chat::ToolChoice::None
        ));

        let provider = test_provider(ProviderType::OpenAI);
        assert!(
            provider
                .build_llm_with_system(
                    &LLMConfig::default(),
                    None,
                    Some(&ToolChoice::Specific("calculator".to_string())),
                    Some(&[definition]),
                )
                .is_ok()
        );
    }

    #[test]
    fn test_tool_choice_changes_client_identity() {
        let config = LLMConfig::default();
        let auto = compute_config_hash(&config, None, Some(&ToolChoice::Auto), None);
        let required = compute_config_hash(&config, None, Some(&ToolChoice::Required), None);
        assert_ne!(auto, required);
    }

    #[test]
    fn test_native_tool_choice_support_matrix() {
        for provider_type in [
            ProviderType::OpenAI,
            ProviderType::Anthropic,
            ProviderType::OpenRouter,
        ] {
            let provider = test_provider(provider_type);
            assert!(provider.supports_tool_choice(&ToolChoice::Auto));
            assert!(provider.supports_tool_choice(&ToolChoice::Required));
            assert!(provider.supports_tool_choice(&ToolChoice::Specific("tool".to_string())));
            assert!(provider.supports_tool_choice(&ToolChoice::None));
        }

        let disabled = test_provider(ProviderType::OpenAI)
            .with_feature_override(LLMFeature::FunctionCalling, false);
        assert!(!disabled.supports_tool_choice(&ToolChoice::Auto));
        assert!(!disabled.supports_tool_choice(&ToolChoice::Required));

        let google = test_provider(ProviderType::Google);
        assert!(google.supports_tool_choice(&ToolChoice::Auto));
        assert!(!google.supports_tool_choice(&ToolChoice::Required));
        assert!(!google.supports_tool_choice(&ToolChoice::None));

        let compatible = test_provider(ProviderType::OpenAICompatible);
        assert!(!compatible.supports_tool_choice(&ToolChoice::Auto));
        let compatible = compatible.with_feature_override(LLMFeature::FunctionCalling, true);
        assert!(compatible.supports_tool_choice(&ToolChoice::Auto));
        assert!(compatible.supports_tool_choice(&ToolChoice::Required));

        for provider_type in [
            ProviderType::Ollama,
            ProviderType::DeepSeek,
            ProviderType::XAI,
            ProviderType::Phind,
            ProviderType::Groq,
            ProviderType::Cohere,
            ProviderType::Mistral,
        ] {
            assert!(!test_provider(provider_type).supports_tool_choice(&ToolChoice::Auto));
        }
    }

    #[test]
    fn test_google_terminal_error_classification_delegates_to_first_party_backend() {
        let google = test_provider(ProviderType::Google);
        assert!(google.is_terminal_error(&LLMError::Config("bad config".to_string())));
        assert!(google.is_terminal_error(&LLMError::Serialization("bad history".to_string())));
        assert!(!google.is_terminal_error(&LLMError::Network("temporary".to_string())));

        let openai = test_provider(ProviderType::OpenAI);
        assert!(openai.is_terminal_error(&LLMError::Config("conflicting chat options".into())));
    }

    #[tokio::test]
    async fn gpt_6_2_openai_preset_keeps_native_tool_calls_out_of_chat() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let provider = UnifiedLLMProvider::new(
            ProviderType::OpenAI,
            "gpt-6.2-sol".into(),
            Some("fixture-key".into()),
            Some(format!("http://{}/v1", listener.local_addr().unwrap())),
        )
        .unwrap()
        .with_feature_override(LLMFeature::FunctionCalling, true);
        assert!(provider.default_config.temperature.is_none());
        assert!(provider.default_config.top_p.is_none());
        assert!(!provider.supports(LLMFeature::FunctionCalling));
        assert!(!provider.supports_tool_choice(&ToolChoice::Auto));
        let request = LLMToolRequest {
            choice: ToolChoice::Auto,
            tools: vec![],
        };
        let result = provider.complete_with_tools(&[], None, &request).await;
        assert!(
            matches!(result, Err(LLMError::Config(message)) if message.contains("Responses API"))
        );
    }

    #[tokio::test]
    async fn chat_sampling_and_native_tools_follow_the_effective_request_effort() {
        let (url, mut received) = local_chat_fixture().await;
        let provider = UnifiedLLMProvider::new(
            ProviderType::OpenAI,
            "gpt-6-luna".into(),
            Some("fixture".into()),
            Some(url),
        )
        .unwrap();
        let choice = LLMToolRequest {
            choice: ToolChoice::Auto,
            tools: vec![LLMToolDefinition {
                name: "lookup".into(),
                description: "test".into(),
                input_schema: serde_json::json!({"type":"object"}),
            }],
        };
        assert!(!provider.supports_tool_choice(&choice.choice));
        assert!(matches!(
            provider.complete_with_tools(&[], None, &choice).await,
            Err(LLMError::Config(_))
        ));
        for field in ["provider_options", "extra_body"] {
            let config = LLMConfig {
                temperature: Some(0.4),
                top_p: None,
                ..LLMConfig::default()
            }
            .with_extra(field, serde_json::json!({"reasoning_effort":"none"}));
            provider
                .complete_with_tools(&[ChatMessage::user("hi")], Some(&config), &choice)
                .await
                .unwrap();
            let body = received.recv().await.unwrap();
            assert_eq!(body["temperature"], 0.4);
            assert_eq!(body["reasoning_effort"], "none");
            assert_eq!(body["tools"][0]["function"]["name"], "lookup");
        }
    }

    #[test]
    fn non_google_projection_removes_signed_provider_state_without_mutating_history() {
        let call = ToolCall {
            id: "foreign-call".to_string(),
            name: "lookup".to_string(),
            arguments: serde_json::json!({"query": "safe"}),
        };
        let state = ai_agents_core::NativeProviderState::new(
            "foreign-exchange",
            "google",
            "generateContent",
            ai_agents_core::NativeProviderTarget::new(
                "https://generativelanguage.googleapis.com/v1beta/",
                "gemini-3.7-flash",
            )
            .unwrap(),
            serde_json::json!({
                "role": "model",
                "parts": [{
                    "functionCall": {"name": "lookup", "args": {"query": "safe"}},
                    "thoughtSignature": "must-not-reach-foreign-provider"
                }]
            }),
            vec![ai_agents_core::NativeCallBinding::new("foreign-call", 0).unwrap()],
        )
        .unwrap();
        let marker =
            ai_agents_core::encode_native_tool_call_markers(&[call], Some(&state)).unwrap();
        let history = vec![
            ChatMessage::user("question"),
            ChatMessage::assistant(marker.clone()),
        ];
        let tools = LLMToolRequest {
            choice: ToolChoice::Auto,
            tools: vec![],
        };
        let wire = chat_adapter::request(
            ProviderType::OpenAI,
            "fixture",
            &history,
            &LLMConfig::default(),
            Some(&tools),
        )
        .unwrap()
        .unwrap()
        .body(false)
        .unwrap();
        assert!(
            wire["messages"][1]["content"]
                .as_str()
                .unwrap()
                .contains("native_tool_calls")
        );
        assert!(wire["messages"][1].get("tool_calls").is_none());
        assert!(!wire.to_string().contains("must-not-reach-foreign-provider"));
        let original = vec![ChatMessage::assistant(marker)];

        let projected = UnifiedLLMProvider::project_foreign_messages(&original).unwrap();

        assert!(
            !projected[0]
                .content
                .contains("must-not-reach-foreign-provider")
        );
        assert!(
            original[0]
                .content
                .contains("must-not-reach-foreign-provider")
        );
    }

    #[test]
    fn test_configured_tool_choice_is_reported() {
        let provider = test_provider(ProviderType::OpenAI)
            .with_tool_choice(ToolChoice::Specific("calculator".to_string()));
        assert_eq!(
            provider.configured_tool_choice(),
            Some(ToolChoice::Specific("calculator".to_string()))
        );
    }

    #[test]
    fn test_tool_call_arguments_are_strict_json() {
        let error = normalize_tool_calls(
            Some(vec![upstream_tool_call("call-1", "calculator", "{bad")]),
            &ToolChoice::Auto,
        )
        .unwrap_err();
        assert!(error.to_string().contains("invalid JSON arguments"));
    }

    #[test]
    fn test_tool_call_ids_preserve_unique_values_and_replace_duplicates() {
        let calls = normalize_tool_calls(
            Some(vec![
                upstream_tool_call("keep", "calculator", "{}"),
                upstream_tool_call("keep", "calculator", "{}"),
                upstream_tool_call("", "calculator", "{}"),
            ]),
            &ToolChoice::Auto,
        )
        .unwrap();

        assert_eq!(calls[0].id, "keep");
        assert_ne!(calls[1].id, "keep");
        assert!(!calls[1].id.is_empty());
        assert!(!calls[2].id.is_empty());
        assert_ne!(calls[1].id, calls[2].id);
    }

    #[test]
    fn test_required_and_specific_tool_call_protocol_validation() {
        assert!(normalize_tool_calls(None, &ToolChoice::Required).is_err());
        let error = normalize_tool_calls(
            Some(vec![upstream_tool_call("call-1", "other", "{}")]),
            &ToolChoice::Specific("calculator".to_string()),
        )
        .unwrap_err();
        assert!(error.to_string().contains("specific tool choice"));
    }

    #[test]
    fn test_marker_messages_convert_only_for_native_tool_completion() {
        let provider = test_provider(ProviderType::OpenAI);
        let assistant = ChatMessage::assistant(
            serde_json::json!([{
                "_ai_agents_native_tool_call": true,
                "id": "call-1",
                "tool": "calculator",
                "arguments": {"expression": "2 + 2"}
            }])
            .to_string(),
        );
        let converted = provider.convert_message_with_tools(&assistant).unwrap();
        match converted.message_type {
            llm::chat::MessageType::ToolUse(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "call-1");
                assert_eq!(calls[0].function.name, "calculator");
                assert_eq!(calls[0].function.arguments, r#"{"expression":"2 + 2"}"#);
            }
            other => panic!("expected tool use, got {other:?}"),
        }

        let result = ChatMessage::tool(
            "calculator",
            serde_json::json!({
                "_ai_agents_native_tool_result": true,
                "id": "call-1",
                "tool": "calculator",
                "output": {"value": 4}
            })
            .to_string(),
        );
        let converted = provider.convert_message_with_tools(&result).unwrap();
        match converted.message_type {
            llm::chat::MessageType::ToolResult(results) => {
                assert_eq!(results[0].id, "call-1");
                assert_eq!(results[0].function.name, "calculator");
                assert_eq!(results[0].function.arguments, r#"{"value":4}"#);
            }
            other => panic!("expected tool result, got {other:?}"),
        }

        let second_result = ChatMessage::tool(
            "calculator",
            serde_json::json!({
                "_ai_agents_native_tool_result": true,
                "id": "call-2",
                "tool": "calculator",
                "output": {"value": 6}
            })
            .to_string(),
        );
        let combined = provider
            .convert_messages_with_tools(&[&result, &second_result])
            .unwrap();
        assert_eq!(combined.len(), 1);
        assert!(matches!(
            &combined[0].message_type,
            llm::chat::MessageType::ToolResult(results) if results.len() == 2
        ));

        let unmarked = ChatMessage::assistant(r#"{"tool":"calculator"}"#);
        let converted = provider.convert_message_with_tools(&unmarked).unwrap();
        assert!(matches!(
            converted.message_type,
            llm::chat::MessageType::Text
        ));
        let plain = provider.convert_message(&assistant);
        assert!(matches!(plain.message_type, llm::chat::MessageType::Text));
    }
}

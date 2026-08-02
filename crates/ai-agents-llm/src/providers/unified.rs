use ai_agents_core::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
    LLMToolDefinition, LLMToolRequest, Role, TokenUsage, ToolCall, ToolChoice,
};
use async_trait::async_trait;
use futures::stream::StreamExt;
use llm::chat::ReasoningEffort;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};

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

    fn to_llm_backend(self) -> llm::builder::LLMBackend {
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
    feature_overrides: HashMap<LLMFeature, bool>,
    tool_choice: Option<ToolChoice>,
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
            .field("feature_overrides", &self.feature_overrides)
            .field("tool_choice", &self.tool_choice)
            .field("client", &"<cached>")
            .finish()
    }
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

    let base = base_url.unwrap_or("http://localhost:11434");
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
            feature_overrides: HashMap::new(),
            tool_choice: None,
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

    pub fn with_feature_override(mut self, feature: LLMFeature, enabled: bool) -> Self {
        self.feature_overrides.insert(feature, enabled);
        self
    }

    pub fn with_feature_overrides(mut self, overrides: HashMap<LLMFeature, bool>) -> Self {
        self.feature_overrides.extend(overrides);
        self
    }

    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    fn map_llm_crate_error(&self, err: impl std::fmt::Display) -> LLMError {
        map_provider_error(
            self.provider_type,
            &self.model,
            self.base_url.as_deref(),
            err,
        )
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
        let mut builder = llm::builder::LLMBuilder::new()
            .backend(self.provider_type.to_llm_backend())
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

        // --- Reasoning effort (first-class, fallback to extra) ---
        let reasoning_effort = config.reasoning_effort.as_deref().or_else(|| {
            config
                .extra
                .get("reasoning_effort")
                .and_then(|v| v.as_str())
        });
        if let Some(effort) = reasoning_effort {
            let re = match effort.to_lowercase().as_str() {
                "low" => ReasoningEffort::Low,
                "medium" => ReasoningEffort::Medium,
                "high" => ReasoningEffort::High,
                other => {
                    return Err(LLMError::Config(format!(
                        "Invalid reasoning_effort '{}'. Expected: low, medium, high",
                        other
                    )));
                }
            };
            builder = builder.reasoning_effort(re);
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

    /// Public backward-compatible `build_llm` — used by tests and legacy call-sites.
    pub fn build_llm(
        &self,
        config: Option<&LLMConfig>,
    ) -> Result<Box<dyn llm::LLMProvider>, LLMError> {
        let cfg = config.unwrap_or(&self.default_config);
        self.build_llm_with_system(cfg, None, None, None)
    }

    /// Ensure the cached client is built (or rebuilt) for the given config + system prompt.
    /// Returns after the cache is populated; callers should then lock and use `self.client`.
    async fn ensure_client(
        &self,
        config: Option<&LLMConfig>,
        system_prompt: Option<&str>,
        tool_choice: Option<&ToolChoice>,
        tools: Option<&[LLMToolDefinition]>,
    ) -> Result<(), LLMError> {
        let cfg = config.unwrap_or(&self.default_config);
        let hash = compute_config_hash(cfg, system_prompt, tool_choice, tools);

        let mut lock = self.client.lock().await;
        if let Some(ref cached) = *lock
            && cached.config_hash == hash
        {
            return Ok(());
        }
        // Build a new client
        let llm = self.build_llm_with_system(cfg, system_prompt, tool_choice, tools)?;
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
        self.ensure_client(config, system_prompt.as_deref(), None, None)
            .await?;

        // Use the cached client
        let lock = self.client.lock().await;
        let cached = lock
            .as_ref()
            .expect("client must be built after ensure_client");

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
        let choice = &request.choice;
        if !self.supports_tool_choice(choice) {
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

        let (system_prompt, non_system_msgs) = extract_system_and_messages(messages);
        let llm_messages = self.convert_messages_with_tools(&non_system_msgs)?;
        let tools = request
            .tools
            .iter()
            .map(map_tool_definition)
            .collect::<Vec<_>>();

        self.ensure_client(
            config,
            system_prompt.as_deref(),
            Some(choice),
            Some(&request.tools),
        )
        .await?;
        let lock = self.client.lock().await;
        let cached = lock
            .as_ref()
            .expect("client must be built after ensure_client");
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
        // Separate system messages from non-system messages
        let (system_prompt, non_system_msgs) = extract_system_and_messages(messages);

        let llm_messages: Vec<llm::chat::ChatMessage> = non_system_msgs
            .iter()
            .map(|m| self.convert_message(m))
            .collect();

        // Ensure client is built for this config + system prompt
        self.ensure_client(config, system_prompt.as_deref(), None, None)
            .await?;

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
                .map_err(|e| self.map_llm_crate_error(e))?
            // lock is dropped here at end of block
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

    fn configured_tool_choice(&self) -> Option<ToolChoice> {
        self.tool_choice.clone()
    }

    fn supports_tool_choice(&self, choice: &ToolChoice) -> bool {
        if self.feature_overrides.get(&LLMFeature::FunctionCalling) == Some(&false) {
            return false;
        }
        let known_choice = matches!(
            choice,
            ToolChoice::Auto | ToolChoice::Required | ToolChoice::Specific(_) | ToolChoice::None
        );
        match self.provider_type {
            ProviderType::OpenAI | ProviderType::Anthropic | ProviderType::OpenRouter => {
                known_choice
            }
            ProviderType::Google => matches!(choice, ToolChoice::Auto),
            ProviderType::OpenAICompatible => {
                self.feature_overrides
                    .get(&LLMFeature::FunctionCalling)
                    .copied()
                    .unwrap_or(false)
                    && known_choice
            }
            _ => false,
        }
    }

    fn supports(&self, feature: LLMFeature) -> bool {
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

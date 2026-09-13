use ai_agents_core::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
    LLMToolDefinition, LLMToolRequest, NativeCallBinding, NativeProviderState,
    NativeProviderTarget, NativeToolCallBatch, NativeToolResult, Role, TokenUsage, ToolCall,
    ToolChoice, decode_native_tool_call_markers, decode_native_tool_result_markers,
};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::{Client, Response, StatusCode, Url, redirect::Policy};
use serde_json::{Map, Value, json};
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use uuid::Uuid;

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/";
const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
const MAX_ERROR_BYTES: usize = 64 * 1024;
const MAX_SSE_FRAME_BYTES: usize = 1024 * 1024;

/// First-party GenerateContent adapter used only by the normal framework Google path.
pub(crate) struct GoogleProvider {
    model: String,
    api_key: String,
    base_url: Url,
    default_config: LLMConfig,
    feature_overrides: HashMap<LLMFeature, bool>,
    tool_choice: Option<ToolChoice>,
    client: Client,
}

impl std::fmt::Debug for GoogleProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoogleProvider")
            .field("model", &self.model)
            .field("api_key", &"***")
            .field("base_url", &self.base_url)
            .field("default_config", &self.default_config)
            .field("feature_overrides", &self.feature_overrides)
            .field("tool_choice", &self.tool_choice)
            .finish()
    }
}

impl GoogleProvider {
    pub(crate) fn new(
        model: String,
        api_key: String,
        base_url: Option<String>,
        default_config: LLMConfig,
    ) -> Result<Self, LLMError> {
        validate_model_name(&model)?;
        let base_url = parse_base_url(base_url.as_deref().unwrap_or(DEFAULT_BASE_URL))?;
        let client = Client::builder()
            .redirect(Policy::none())
            .build()
            .map_err(|error| {
                LLMError::Config(format!("failed to build Google HTTP client: {error}"))
            })?;
        Ok(Self {
            model,
            api_key,
            base_url,
            default_config,
            feature_overrides: HashMap::new(),
            tool_choice: None,
            client,
        })
    }

    pub(crate) fn set_feature_override(&mut self, feature: LLMFeature, enabled: bool) {
        self.feature_overrides.insert(feature, enabled);
    }

    pub(crate) fn set_feature_overrides(&mut self, overrides: &HashMap<LLMFeature, bool>) {
        self.feature_overrides.extend(overrides.clone());
    }

    pub(crate) fn set_tool_choice(&mut self, choice: ToolChoice) {
        self.tool_choice = Some(choice);
    }

    fn effective_config<'a>(&'a self, config: Option<&'a LLMConfig>) -> &'a LLMConfig {
        config.unwrap_or(&self.default_config)
    }

    fn endpoint(&self, method: &str, stream: bool) -> Result<Url, LLMError> {
        let mut endpoint = self
            .base_url
            .join(&format!("models/{}:{method}", self.model))
            .map_err(|error| LLMError::Config(format!("invalid Google endpoint: {error}")))?;
        if stream {
            endpoint.query_pairs_mut().append_pair("alt", "sse");
        }
        Ok(endpoint)
    }

    async fn send_unary(&self, body: Value, config: &LLMConfig) -> Result<Value, LLMError> {
        let endpoint = self.endpoint("generateContent", false)?;
        let mut request = self
            .client
            .post(endpoint)
            .header("x-goog-api-key", &self.api_key)
            .json(&body);
        if let Some(timeout) = effective_timeout(config)? {
            request = request.timeout(timeout);
        }
        let response = request.send().await.map_err(map_transport_error)?;
        let status = response.status();
        if !status.is_success() {
            return Err(map_http_error(response, status).await);
        }
        let bytes = read_bounded(response, MAX_RESPONSE_BYTES).await?;
        serde_json::from_slice(&bytes)
            .map_err(|error| protocol_error(format!("invalid generateContent response: {error}")))
    }

    async fn send_stream(&self, body: Value, config: &LLMConfig) -> Result<Response, LLMError> {
        let endpoint = self.endpoint("streamGenerateContent", true)?;
        let mut request = self
            .client
            .post(endpoint)
            .header("x-goog-api-key", &self.api_key)
            .json(&body);
        if let Some(timeout) = effective_timeout(config)? {
            request = request.timeout(timeout);
        }
        let response = request.send().await.map_err(map_transport_error)?;
        let status = response.status();
        if !status.is_success() {
            return Err(map_http_error(response, status).await);
        }
        Ok(response)
    }
}

#[async_trait]
impl LLMProvider for GoogleProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<LLMResponse, LLMError> {
        let config = self.effective_config(config);
        let existing_call_ids = native_call_ids(messages)?;
        let body = build_request(messages, config, None, &self.model, &self.base_url)?;
        let response = self.send_unary(body, config).await?;
        parse_response(
            response,
            &self.model,
            &self.base_url,
            false,
            &existing_call_ids,
        )
    }

    async fn complete_with_tools(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
        request: &LLMToolRequest,
    ) -> Result<LLMResponse, LLMError> {
        if !self.supports_tool_choice(&request.choice) {
            return Err(LLMError::Config(
                "the first-party Google adapter supports native tool choice auto only".to_string(),
            ));
        }
        let config = self.effective_config(config);
        let existing_call_ids = native_call_ids(messages)?;
        let body = build_request(
            messages,
            config,
            Some(&request.tools),
            &self.model,
            &self.base_url,
        )?;
        let response = self.send_unary(body, config).await?;
        parse_response(
            response,
            &self.model,
            &self.base_url,
            true,
            &existing_call_ids,
        )
    }

    async fn complete_stream(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError>
    {
        let config = self.effective_config(config);
        let body = build_request(messages, config, None, &self.model, &self.base_url)?;
        let response = self.send_stream(body, config).await?;
        let stream = google_sse_stream(response);
        Ok(Box::new(Box::pin(stream)))
    }

    fn provider_name(&self) -> &str {
        "google"
    }

    fn configured_tool_choice(&self) -> Option<ToolChoice> {
        self.tool_choice.clone()
    }

    fn supports_tool_choice(&self, choice: &ToolChoice) -> bool {
        self.feature_overrides.get(&LLMFeature::FunctionCalling) != Some(&false)
            && matches!(choice, ToolChoice::Auto)
    }

    fn supports(&self, feature: LLMFeature) -> bool {
        if let Some(value) = self.feature_overrides.get(&feature) {
            return *value;
        }
        matches!(
            feature,
            LLMFeature::Streaming
                | LLMFeature::SystemMessages
                | LLMFeature::FunctionCalling
                | LLMFeature::JsonMode
        )
    }

    fn is_terminal_error(&self, error: &LLMError) -> bool {
        matches!(error, LLMError::Config(_) | LLMError::Serialization(_))
    }
}

fn validate_model_name(model: &str) -> Result<(), LLMError> {
    if model.is_empty()
        || !model
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(LLMError::Config(
            "Google model names must contain only ASCII letters, digits, '.', '_' or '-'"
                .to_string(),
        ));
    }
    Ok(())
}

fn parse_base_url(raw: &str) -> Result<Url, LLMError> {
    let mut url = Url::parse(raw)
        .map_err(|error| LLMError::Config(format!("invalid Google base_url: {error}")))?;
    if url.cannot_be_a_base()
        || !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(LLMError::Config(
            "Google base_url must be an HTTP(S) base URL without credentials, query, or fragment"
                .to_string(),
        ));
    }
    if !url.path().ends_with('/') {
        let path = format!("{}/", url.path());
        url.set_path(&path);
    }
    Ok(url)
}

fn effective_timeout(config: &LLMConfig) -> Result<Option<Duration>, LLMError> {
    let seconds = match config.timeout_seconds {
        Some(value) => Some(value),
        None => config
            .extra
            .get("timeout_seconds")
            .map(|value| {
                value.as_u64().ok_or_else(|| {
                    LLMError::Config("Google timeout_seconds must be a u64".to_string())
                })
            })
            .transpose()?,
    };
    Ok(seconds.map(Duration::from_secs))
}

fn build_request(
    messages: &[ChatMessage],
    config: &LLMConfig,
    tools: Option<&[LLMToolDefinition]>,
    model: &str,
    base_url: &Url,
) -> Result<Value, LLMError> {
    reject_unsupported_transport_settings(config)?;
    let (system_instruction, contents) = convert_messages(messages, model, base_url)?;
    let mut body = match config.extra.get("extra_body") {
        None => Map::new(),
        Some(Value::Object(map)) => validate_extra_body(map)?,
        Some(_) => {
            return Err(LLMError::Config(
                "Google extra_body must be a JSON object".to_string(),
            ));
        }
    };
    if body
        .get("generationConfig")
        .is_some_and(|value| !value.is_object())
    {
        return Err(LLMError::Config(
            "Google extra_body.generationConfig must be an object".to_string(),
        ));
    }

    body.insert("contents".to_string(), Value::Array(contents));
    if let Some(system_instruction) = system_instruction {
        body.insert("systemInstruction".to_string(), system_instruction);
    }
    if let Some(tool_definitions) = tools.filter(|tools| !tools.is_empty()) {
        body.insert("tools".to_string(), map_tool_definitions(tool_definitions)?);
    }

    let generated = generation_config(config, model)?;
    if !generated.is_empty() {
        let existing = body
            .remove("generationConfig")
            .unwrap_or_else(|| Value::Object(Map::new()));
        let mut existing = existing.as_object().cloned().ok_or_else(|| {
            LLMError::Config("Google extra_body.generationConfig must be an object".to_string())
        })?;
        for (key, value) in generated {
            existing.insert(key, value);
        }
        body.insert("generationConfig".to_string(), Value::Object(existing));
    }
    let body = Value::Object(body);
    validate_json_numbers(&body, "Google request")?;
    Ok(body)
}

fn reject_unsupported_transport_settings(config: &LLMConfig) -> Result<(), LLMError> {
    const UNSUPPORTED: &[&str] = &[
        "resilient",
        "resilient_attempts",
        "resilient_base_delay_ms",
        "resilient_max_delay_ms",
        "resilient_jitter",
        "normalize_response",
    ];
    if let Some(key) = UNSUPPORTED
        .iter()
        .find(|key| config.extra.contains_key(**key))
    {
        return Err(LLMError::Config(format!(
            "Google setting '{key}' is not supported by the first-party adapter"
        )));
    }
    Ok(())
}

fn validate_extra_body(extra: &Map<String, Value>) -> Result<Map<String, Value>, LLMError> {
    const OWNED: &[&str] = &["contents", "systemInstruction", "tools", "toolConfig"];
    const ALLOWED: &[&str] = &["generationConfig", "safetySettings"];
    let mut output = Map::new();
    for (key, value) in extra {
        if OWNED.contains(&key.as_str()) {
            return Err(LLMError::Config(format!(
                "Google extra_body cannot override framework-owned field '{key}'"
            )));
        }
        if !ALLOWED.contains(&key.as_str()) {
            return Err(LLMError::Config(format!(
                "unsupported Google extra_body field '{key}'"
            )));
        }
        output.insert(key.clone(), value.clone());
    }
    Ok(output)
}

fn generation_config(config: &LLMConfig, model: &str) -> Result<Map<String, Value>, LLMError> {
    let mut output = Map::new();
    if let Some(value) = config.max_tokens {
        output.insert("maxOutputTokens".to_string(), json!(value));
    }
    if let Some(value) = config.temperature {
        output.insert(
            "temperature".to_string(),
            finite_number(value, "temperature")?,
        );
    }
    if let Some(value) = config.top_p {
        output.insert("topP".to_string(), finite_number(value, "top_p")?);
    }
    if let Some(value) = config.top_k {
        output.insert("topK".to_string(), json!(value));
    }

    let effort = match config.reasoning_effort.as_deref() {
        Some(value) => Some(value),
        None => config
            .extra
            .get("reasoning_effort")
            .map(|value| {
                value.as_str().ok_or_else(|| {
                    LLMError::Config("Google reasoning_effort must be a string".to_string())
                })
            })
            .transpose()?,
    };
    let budget = match config.reasoning_budget_tokens {
        Some(value) => Some(value),
        None => config
            .extra
            .get("reasoning_budget_tokens")
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| {
                        LLMError::Config(
                            "Google reasoning_budget_tokens must fit in u32".to_string(),
                        )
                    })
            })
            .transpose()?,
    };
    if effort.is_some() && budget.is_some() {
        return Err(LLMError::Config(
            "Google reasoning_effort and reasoning_budget_tokens cannot be sent together"
                .to_string(),
        ));
    }
    let mut thinking = Map::new();
    if let Some(effort) = effort {
        let level = match effort.to_ascii_lowercase().as_str() {
            "low" => "low",
            "medium" => "medium",
            "high" => "high",
            _ => {
                return Err(LLMError::Config(format!(
                    "invalid Google reasoning_effort '{effort}'; expected low, medium, or high"
                )));
            }
        };
        if model.starts_with("gemini-3") {
            thinking.insert("thinkingLevel".to_string(), json!(level));
        } else if model.starts_with("gemini-2.5") {
            let budget = match level {
                "low" => 1024,
                "medium" => 8192,
                "high" => 24576,
                _ => unreachable!(),
            };
            thinking.insert("thinkingBudget".to_string(), json!(budget));
        } else {
            return Err(LLMError::Config(format!(
                "Google reasoning_effort is unsupported for model '{model}'"
            )));
        }
    } else if let Some(budget) = budget {
        if model.starts_with("gemini-3") {
            return Err(LLMError::Config(
                "Google Gemini 3 models require reasoning_effort instead of reasoning_budget_tokens"
                    .to_string(),
            ));
        }
        thinking.insert("thinkingBudget".to_string(), json!(budget));
    } else if effective_reasoning(config)? == Some(false) {
        if model.starts_with("gemini-3") || model.starts_with("gemini-2.5-pro") {
            return Err(LLMError::Config(format!(
                "Google model '{model}' does not support disabling thinking"
            )));
        }
        thinking.insert("thinkingBudget".to_string(), json!(0));
    }
    if !thinking.is_empty() {
        output.insert("thinkingConfig".to_string(), Value::Object(thinking));
    }

    if config.frequency_penalty.is_some()
        || config.presence_penalty.is_some()
        || config
            .stop_sequences
            .as_ref()
            .is_some_and(|v| !v.is_empty())
    {
        tracing::debug!(
            "frequency_penalty, presence_penalty, and stop_sequences remain unsupported on the first-party Google adapter"
        );
    }
    Ok(output)
}

fn effective_reasoning(config: &LLMConfig) -> Result<Option<bool>, LLMError> {
    match config.reasoning {
        Some(value) => Ok(Some(value)),
        None => config
            .extra
            .get("reasoning")
            .map(|value| {
                value.as_bool().ok_or_else(|| {
                    LLMError::Config("Google reasoning must be a boolean".to_string())
                })
            })
            .transpose(),
    }
}

fn finite_number(value: f32, field: &str) -> Result<Value, LLMError> {
    serde_json::Number::from_f64(f64::from(value))
        .map(Value::Number)
        .ok_or_else(|| LLMError::Config(format!("Google {field} must be finite")))
}

fn map_tool_definitions(definitions: &[LLMToolDefinition]) -> Result<Value, LLMError> {
    let mut declarations = Vec::with_capacity(definitions.len());
    for definition in definitions {
        validate_function_name(&definition.name)?;
        let schema = normalize_tool_schema(&definition.input_schema)?;
        declarations.push(json!({
            "name": definition.name,
            "description": definition.description,
            "parametersJsonSchema": schema,
        }));
    }
    Ok(json!([{ "functionDeclarations": declarations }]))
}

fn validate_function_name(name: &str) -> Result<(), LLMError> {
    if !is_valid_function_name(name) {
        return Err(LLMError::Config(format!(
            "Google function name '{name}' is invalid"
        )));
    }
    Ok(())
}

fn is_valid_function_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':'))
}

fn normalize_tool_schema(schema: &Value) -> Result<Value, LLMError> {
    let mut schema = schema.as_object().cloned().ok_or_else(|| {
        LLMError::Config("Google tool input_schema must be a JSON object".to_string())
    })?;
    if schema.is_empty() {
        schema.insert("type".to_string(), json!("object"));
        schema.insert("properties".to_string(), json!({}));
    }
    // `$schema` selects a local JSON Schema dialect; GenerateContent accepts the schema body but
    // does not list this dialect marker as a supported `parametersJsonSchema` keyword.
    schema.remove("$schema");
    match schema.get("type") {
        Some(Value::String(value)) if value == "object" => {}
        _ => {
            return Err(LLMError::Config(
                "Google tool input_schema must describe an object".to_string(),
            ));
        }
    }
    validate_json_numbers(&Value::Object(schema.clone()), "tool schema")?;
    Ok(Value::Object(schema))
}

fn convert_messages(
    messages: &[ChatMessage],
    model: &str,
    base_url: &Url,
) -> Result<(Option<Value>, Vec<Value>), LLMError> {
    let system_parts = messages
        .iter()
        .filter(|message| message.role == Role::System)
        .map(|message| json!({ "text": message.content }))
        .collect::<Vec<_>>();
    let system_instruction = (!system_parts.is_empty()).then(|| json!({ "parts": system_parts }));

    let mut contents = Vec::new();
    let mut pending: Option<PendingExchange> = None;
    for message in messages
        .iter()
        .filter(|message| message.role != Role::System)
    {
        if matches!(message.role, Role::Tool | Role::Function)
            && let Some(results) = decode_native_tool_result_markers(&message.content)?
        {
            for result in &results {
                validate_json_numbers(result.output(), "tool result")?;
            }
            let exchange = pending.get_or_insert_with(PendingExchange::default);
            exchange.results.extend(results);
            continue;
        }

        flush_pending(&mut contents, pending.take(), message.role == Role::User)?;
        if message.role == Role::Assistant
            && let Some(call_markers) = decode_native_tool_call_markers(&message.content)?
        {
            let exchange = PendingExchange::from_call_markers(call_markers, model, base_url)?;
            pending = Some(exchange);
            continue;
        }

        let role = match message.role {
            Role::Assistant => "model",
            _ => "user",
        };
        let text = match message.role {
            Role::Function | Role::Tool => {
                format!(
                    "[{} result]: {}",
                    message.name.as_deref().unwrap_or("tool"),
                    message.content
                )
            }
            _ => message.content.clone(),
        };
        contents.push(json!({ "role": role, "parts": [{ "text": text }] }));
    }
    flush_pending(&mut contents, pending.take(), false)?;
    Ok((system_instruction, contents))
}

#[derive(Default)]
struct PendingExchange {
    model_content: Value,
    calls: Vec<BoundCall>,
    results: Vec<NativeToolResult>,
    signed: bool,
    readable_projection: String,
}

struct BoundCall {
    call_id: String,
    name: String,
    provider_id: Option<String>,
}

impl PendingExchange {
    fn from_call_markers(
        batch: NativeToolCallBatch,
        model: &str,
        base_url: &Url,
    ) -> Result<Self, LLMError> {
        let (markers, state) = batch.into_parts();
        for marker in &markers {
            validate_json_numbers(&marker.arguments, "native tool arguments")?;
        }
        let readable_projection = readable_calls(&markers)?;
        if let Some(state) = state {
            validate_provider_state(&state, &markers, model, base_url)?;
            let parts = state
                .model_content()
                .get("parts")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    protocol_error("Google provider state model_content.parts is missing")
                })?;
            let mut calls = Vec::with_capacity(state.bindings().len());
            for binding in state.bindings() {
                let call = parts
                    .get(binding.part_index())
                    .and_then(|part| part.get("functionCall"))
                    .and_then(Value::as_object)
                    .ok_or_else(|| {
                        protocol_error("provider binding does not point to a functionCall")
                    })?;
                calls.push(BoundCall {
                    call_id: binding.call_id().to_string(),
                    name: call
                        .get("name")
                        .and_then(Value::as_str)
                        .ok_or_else(|| protocol_error("functionCall name is missing"))?
                        .to_string(),
                    provider_id: binding.provider_call_id().map(str::to_string),
                });
            }
            return Ok(Self {
                model_content: state.model_content().clone(),
                calls,
                results: Vec::new(),
                signed: true,
                readable_projection,
            });
        }

        let calls = markers
            .iter()
            .map(|marker| BoundCall {
                call_id: marker.id.clone(),
                name: marker.name.clone(),
                provider_id: None,
            })
            .collect::<Vec<_>>();
        let parts = markers
            .into_iter()
            .map(|marker| {
                json!({
                    "functionCall": {
                        "name": marker.name,
                        "args": marker.arguments,
                    }
                })
            })
            .collect::<Vec<_>>();
        Ok(Self {
            model_content: json!({ "role": "model", "parts": parts }),
            calls,
            results: Vec::new(),
            signed: false,
            readable_projection,
        })
    }
}

fn readable_calls(calls: &[ToolCall]) -> Result<String, LLMError> {
    let calls = calls
        .iter()
        .map(|call| {
            json!({
                "id": call.id,
                "tool": call.name,
                "arguments": call.arguments,
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&json!({ "native_tool_calls": calls }))
        .map_err(|error| protocol_error(format!("failed to project native calls: {error}")))
}

fn validate_provider_state(
    state: &NativeProviderState,
    markers: &[ToolCall],
    model: &str,
    base_url: &Url,
) -> Result<(), LLMError> {
    state.validate_for_calls(markers)?;
    if state.provider() != "google"
        || state.api() != "generateContent"
        || state.target().model() != model
        || state.target().endpoint() != base_url.as_str()
    {
        return Err(protocol_error(
            "Google provider state is incompatible with this backend target",
        ));
    }
    let marker_by_id = markers
        .iter()
        .map(|marker| (marker.id.as_str(), marker))
        .collect::<HashMap<_, _>>();
    let parts = state
        .model_content()
        .get("parts")
        .and_then(Value::as_array)
        .ok_or_else(|| protocol_error("provider state model_content.parts is missing"))?;
    if state.model_content().get("role").and_then(Value::as_str) != Some("model") {
        return Err(protocol_error(
            "Google provider state model_content must have role model",
        ));
    }
    let mut seen = HashSet::new();
    for binding in state.bindings() {
        if !seen.insert(binding.call_id()) {
            return Err(protocol_error("duplicate provider call binding"));
        }
        let marker = marker_by_id
            .get(binding.call_id())
            .ok_or_else(|| protocol_error("provider binding references an unknown call"))?;
        let call = parts
            .get(binding.part_index())
            .and_then(|part| part.get("functionCall"))
            .and_then(Value::as_object)
            .ok_or_else(|| protocol_error("provider binding does not point to a functionCall"))?;
        if call.get("name").and_then(Value::as_str) != Some(marker.name.as_str()) {
            return Err(protocol_error(
                "provider call name differs from execution marker",
            ));
        }
        let raw_provider_id = parse_optional_function_call_id(call)?;
        if raw_provider_id.as_deref() != binding.provider_call_id() {
            return Err(protocol_error(
                "provider call ID differs from its replay binding",
            ));
        }
        let raw_args = call.get("args").cloned().unwrap_or_else(|| json!({}));
        if raw_args != marker.arguments {
            return Err(protocol_error(
                "provider call arguments differ from execution marker",
            ));
        }
    }
    Ok(())
}

fn flush_pending(
    contents: &mut Vec<Value>,
    pending: Option<PendingExchange>,
    past_user_turn: bool,
) -> Result<(), LLMError> {
    let Some(pending) = pending else {
        return Ok(());
    };
    if pending.calls.is_empty() && !pending.results.is_empty() {
        return Err(protocol_error(
            "native tool results have no preceding call batch",
        ));
    }
    if pending.results.len() != pending.calls.len() {
        if pending.signed && !past_user_turn {
            return Err(protocol_error(
                "current Google tool exchange does not have exactly one result per call",
            ));
        }
        if pending.signed {
            contents.push(json!({
                "role": "model",
                "parts": [{ "text": pending.readable_projection }],
            }));
            let results = pending
                .results
                .iter()
                .map(|result| {
                    json!({
                        "id": result.id(),
                        "tool": result.tool(),
                        "output": result.output(),
                    })
                })
                .collect::<Vec<_>>();
            let results = serde_json::to_string(&json!({
                "native_tool_results": results,
                "status": "incomplete",
            }))
            .map_err(|error| {
                protocol_error(format!(
                    "failed to project incomplete native results: {error}"
                ))
            })?;
            contents.push(json!({
                "role": "user",
                "parts": [{ "text": results }],
            }));
            return Ok(());
        }
        contents.push(pending.model_content);
        return Ok(());
    }
    contents.push(pending.model_content);
    let results = pending
        .results
        .into_iter()
        .map(|result| (result.id().to_string(), result))
        .collect::<HashMap<_, _>>();
    if results.len() != pending.calls.len() {
        return Err(protocol_error(
            "Google tool exchange contains duplicate or extra results",
        ));
    }
    let mut parts = Vec::with_capacity(pending.calls.len());
    for call in pending.calls {
        let result = results
            .get(&call.call_id)
            .ok_or_else(|| protocol_error("Google tool exchange is missing a result"))?;
        if result.tool() != call.name {
            return Err(protocol_error(
                "Google tool result name does not match its call",
            ));
        }
        let mut function_response = Map::new();
        if let Some(id) = call.provider_id {
            function_response.insert("id".to_string(), Value::String(id));
        }
        function_response.insert("name".to_string(), Value::String(call.name.clone()));
        function_response.insert(
            "response".to_string(),
            json!({ "name": call.name, "content": result.output() }),
        );
        parts.push(json!({ "functionResponse": function_response }));
    }
    contents.push(json!({ "role": "user", "parts": parts }));
    Ok(())
}

fn validate_json_numbers(value: &Value, context: &str) -> Result<(), LLMError> {
    const MAX_EXACT_GOOGLE_INTEGER: u64 = 1_u64 << 53;
    match value {
        Value::Number(number) => {
            let exact = if let Some(value) = number.as_i64() {
                value.unsigned_abs() <= MAX_EXACT_GOOGLE_INTEGER
            } else if let Some(value) = number.as_u64() {
                value <= MAX_EXACT_GOOGLE_INTEGER
            } else {
                number.as_f64().is_some_and(f64::is_finite)
            };
            if !exact {
                return Err(LLMError::Serialization(format!(
                    "{context} contains a number that Google Struct cannot represent exactly"
                )));
            }
        }
        Value::Array(values) => {
            for value in values {
                validate_json_numbers(value, context)?;
            }
        }
        Value::Object(values) => {
            for value in values.values() {
                validate_json_numbers(value, context)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn native_call_ids(messages: &[ChatMessage]) -> Result<HashSet<String>, LLMError> {
    let mut ids = HashSet::new();
    for message in messages
        .iter()
        .filter(|message| message.role == Role::Assistant)
    {
        if let Some(batch) = decode_native_tool_call_markers(&message.content)? {
            ids.extend(batch.calls().iter().map(|call| call.id.clone()));
        }
    }
    Ok(ids)
}

fn parse_response(
    response: Value,
    requested_model: &str,
    endpoint: &Url,
    tools_exposed: bool,
    existing_call_ids: &HashSet<String>,
) -> Result<LLMResponse, LLMError> {
    let usage = parse_usage(response.get("usageMetadata"))?;
    let candidates = response
        .get("candidates")
        .and_then(Value::as_array)
        .filter(|values| !values.is_empty());
    let Some(candidate) = candidates.and_then(|values| values.first()) else {
        if let Some(reason) = response
            .get("promptFeedback")
            .and_then(|value| value.get("blockReason"))
            .and_then(Value::as_str)
        {
            return Err(LLMError::ContentFiltered(format!(
                "Google blocked the prompt ({reason})"
            )));
        }
        return Err(protocol_error("Google returned no candidates"));
    };
    let finish = candidate
        .get("finishReason")
        .and_then(Value::as_str)
        .ok_or_else(|| protocol_error("Google candidate has no finishReason"))?;
    if !matches!(finish, "STOP" | "MAX_TOKENS") {
        return match classify_finish(finish, false) {
            Err(error) => Err(error),
            Ok(_) => Err(protocol_error(
                "Google returned an unsupported terminal candidate",
            )),
        };
    }
    let content = candidate
        .get("content")
        .cloned()
        .ok_or_else(|| protocol_error("Google candidate has no content"))?;
    let parts = content
        .get("parts")
        .and_then(Value::as_array)
        .ok_or_else(|| protocol_error("Google candidate content has no parts"))?;
    let visible = visible_text(parts)?;
    let calls = parse_function_calls(parts, existing_call_ids)?;
    validate_required_signatures(requested_model, parts, &calls)?;
    let finish_reason = classify_finish(finish, !calls.is_empty())?;
    if !tools_exposed && !calls.is_empty() {
        return Err(protocol_error(
            "Google returned a function call when no native tools were exposed",
        ));
    }

    let model_version = response.get("modelVersion").and_then(Value::as_str);
    let mut normalized = LLMResponse {
        content: visible,
        finish_reason,
        usage,
        model: Some(requested_model.to_string()),
        metadata: HashMap::new(),
    };
    if !calls.is_empty() {
        let state = build_provider_state(requested_model, endpoint, content, &calls)?;
        normalized.set_provider_state(state)?;
        normalized.set_tool_calls(calls.into_iter().map(|call| call.normalized).collect())?;
    }
    if let Some(model_version) = model_version {
        normalized
            .metadata
            .insert("google_model_version".to_string(), json!(model_version));
    }
    Ok(normalized)
}

struct ParsedCall {
    part_index: usize,
    provider_id: Option<String>,
    normalized: ToolCall,
}

fn parse_function_calls(
    parts: &[Value],
    existing_call_ids: &HashSet<String>,
) -> Result<Vec<ParsedCall>, LLMError> {
    let mut calls = Vec::new();
    let mut used_runtime_ids = existing_call_ids.clone();
    let mut response_provider_ids = HashSet::new();
    for (part_index, part) in parts.iter().enumerate() {
        let Some(call) = part.get("functionCall") else {
            continue;
        };
        let call = call
            .as_object()
            .ok_or_else(|| protocol_error("functionCall must be an object"))?;
        let name = call
            .get("name")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| protocol_error("functionCall name is missing"))?;
        if !is_valid_function_name(name) {
            return Err(protocol_error("functionCall name is invalid"));
        }
        let args = call.get("args").cloned().unwrap_or_else(|| json!({}));
        if !args.is_object() {
            return Err(protocol_error(
                "functionCall args must be an object when present",
            ));
        }
        validate_json_numbers(&args, "functionCall args")?;
        let provider_id = parse_optional_function_call_id(call)?;
        if let Some(provider_id) = &provider_id
            && !response_provider_ids.insert(provider_id.clone())
        {
            return Err(protocol_error("duplicate Google functionCall id"));
        }
        let mut runtime_id = provider_id.clone().unwrap_or_default();
        if runtime_id.is_empty() || !used_runtime_ids.insert(runtime_id.clone()) {
            loop {
                runtime_id = format!("ai-google-call-{}", Uuid::new_v4());
                if used_runtime_ids.insert(runtime_id.clone()) {
                    break;
                }
            }
        }
        calls.push(ParsedCall {
            part_index,
            provider_id,
            normalized: ToolCall {
                id: runtime_id,
                name: name.to_string(),
                arguments: args,
            },
        });
    }
    Ok(calls)
}

fn parse_optional_function_call_id(call: &Map<String, Value>) -> Result<Option<String>, LLMError> {
    match call.get("id") {
        None => Ok(None),
        Some(Value::String(id)) if !id.is_empty() => Ok(Some(id.clone())),
        Some(_) => Err(protocol_error(
            "functionCall id must be a non-empty string when present",
        )),
    }
}

fn visible_text(parts: &[Value]) -> Result<String, LLMError> {
    let mut output = String::new();
    for part in parts {
        if part.get("thought").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        if let Some(text) = part.get("text") {
            let text = text
                .as_str()
                .ok_or_else(|| protocol_error("Google text part must contain a string"))?;
            output.push_str(text);
        }
    }
    Ok(output)
}

fn validate_required_signatures(
    model: &str,
    parts: &[Value],
    calls: &[ParsedCall],
) -> Result<(), LLMError> {
    if !model.starts_with("gemini-3") || calls.is_empty() {
        return Ok(());
    }
    let first_part = parts
        .get(calls[0].part_index)
        .ok_or_else(|| protocol_error("first Google functionCall part disappeared"))?;
    if first_part
        .get("thoughtSignature")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
    {
        return Err(protocol_error(
            "Gemini 3 functionCall response is missing its required thoughtSignature",
        ));
    }
    Ok(())
}

fn build_provider_state(
    model: &str,
    endpoint: &Url,
    model_content: Value,
    calls: &[ParsedCall],
) -> Result<NativeProviderState, LLMError> {
    let bindings = calls
        .iter()
        .map(|call| {
            let binding = NativeCallBinding::new(&call.normalized.id, call.part_index)?;
            Ok(match &call.provider_id {
                Some(provider_id) => binding.with_provider_call_id(provider_id),
                None => binding,
            })
        })
        .collect::<Result<Vec<_>, LLMError>>()?;
    NativeProviderState::new(
        Uuid::new_v4().to_string(),
        "google",
        "generateContent",
        NativeProviderTarget::new(endpoint.as_str(), model)?,
        model_content,
        bindings,
    )
}

fn classify_finish(reason: &str, has_calls: bool) -> Result<FinishReason, LLMError> {
    match reason {
        "STOP" if has_calls => Ok(FinishReason::ToolCall),
        "STOP" => Ok(FinishReason::Stop),
        "MAX_TOKENS" if has_calls => Err(protocol_error(
            "Google stopped at the token limit while returning function calls",
        )),
        "MAX_TOKENS" => Ok(FinishReason::Length),
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "IMAGE_SAFETY" => {
            Err(LLMError::ContentFiltered(format!(
                "Google stopped generation ({reason})"
            )))
        }
        "MALFORMED_FUNCTION_CALL" => {
            Err(protocol_error("Google reported a malformed function call"))
        }
        "LANGUAGE" | "OTHER" | "FINISH_REASON_UNSPECIFIED" => Err(protocol_error(format!(
            "unsupported Google finishReason '{reason}'"
        ))),
        other => Err(protocol_error(format!(
            "unknown Google finishReason '{other}'"
        ))),
    }
}

fn parse_usage(value: Option<&Value>) -> Result<Option<TokenUsage>, LLMError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let prompt = optional_u32(value, "promptTokenCount")?;
    let candidates = optional_u32(value, "candidatesTokenCount")?;
    let thoughts = optional_u32(value, "thoughtsTokenCount")?.unwrap_or(0);
    let (Some(prompt), Some(candidates)) = (prompt, candidates) else {
        return Ok(None);
    };
    let completion = candidates
        .checked_add(thoughts)
        .ok_or_else(|| protocol_error("Google completion token count overflow"))?;
    let total = match optional_u32(value, "totalTokenCount")? {
        Some(value) => value,
        None => prompt
            .checked_add(completion)
            .ok_or_else(|| protocol_error("Google total token count overflow"))?,
    };
    Ok(Some(TokenUsage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: total,
    }))
}

fn optional_u32(value: &Value, field: &str) -> Result<Option<u32>, LLMError> {
    let Some(raw) = value.get(field) else {
        return Ok(None);
    };
    let integer = raw
        .as_u64()
        .ok_or_else(|| protocol_error(format!("Google {field} must be a non-negative integer")))?;
    u32::try_from(integer)
        .map(Some)
        .map_err(|_| protocol_error(format!("Google {field} exceeds the supported range")))
}

fn protocol_error(message: impl Into<String>) -> LLMError {
    LLMError::Serialization(format!(
        "Google provider protocol error: {}",
        message.into()
    ))
}

fn map_transport_error(error: reqwest::Error) -> LLMError {
    if error.is_timeout() {
        LLMError::Network("Google request timed out".to_string())
    } else {
        LLMError::Network(format!("Google request failed: {error}"))
    }
}

async fn map_http_error(response: Response, status: StatusCode) -> LLMError {
    let retry_after = response
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_secs);
    let body = read_bounded(response, MAX_ERROR_BYTES)
        .await
        .unwrap_or_default();
    if status == StatusCode::TOO_MANY_REQUESTS {
        return LLMError::RateLimit { retry_after };
    }
    let summary = serde_json::from_slice::<Value>(&body)
        .ok()
        .and_then(|value| {
            value
                .get("error")
                .and_then(|error| error.get("status"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| {
            status
                .canonical_reason()
                .unwrap_or("HTTP error")
                .to_string()
        });
    LLMError::API {
        message: format!("Google API request failed: {summary}"),
        status: Some(status.as_u16()),
    }
}

async fn read_bounded(response: Response, limit: usize) -> Result<Vec<u8>, LLMError> {
    let mut stream = response.bytes_stream();
    let mut output = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(map_transport_error)?;
        if output.len().saturating_add(chunk.len()) > limit {
            return Err(protocol_error(
                "Google response body exceeds the size limit",
            ));
        }
        output.extend_from_slice(&chunk);
    }
    Ok(output)
}

fn google_sse_stream(
    response: Response,
) -> impl futures::Stream<Item = Result<LLMChunk, LLMError>> + Send {
    async_stream::stream! {
        let mut bytes_stream = response.bytes_stream();
        let mut pending = Vec::<u8>::new();
        let mut latest_usage = None;
        let mut terminal_reason = None;
        while let Some(chunk) = bytes_stream.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    yield Err(map_transport_error(error));
                    return;
                }
            };
            pending.extend_from_slice(&chunk);
            loop {
                let Some((frame_end, delimiter_len)) = find_sse_frame_end(&pending) else {
                    if pending.len() > MAX_SSE_FRAME_BYTES {
                        yield Err(protocol_error("Google SSE frame exceeds the size limit"));
                        return;
                    }
                    break;
                };
                let frame = pending.drain(..frame_end).collect::<Vec<_>>();
                pending.drain(..delimiter_len);
                match parse_sse_frame(&frame) {
                    Ok(None) => {}
                    Ok(Some(event)) => {
                        if let Some(usage) = event.usage {
                            latest_usage = Some(usage);
                        }
                        if terminal_reason.is_some()
                            && (!event.text.is_empty()
                                || event.finish_reason.is_some()
                                || event.has_function_calls)
                        {
                            yield Err(protocol_error("Google SSE emitted content after its terminal candidate"));
                            return;
                        }
                        if !event.text.is_empty() {
                            yield Ok(LLMChunk::new(event.text, false));
                        }
                        if let Some(reason) = event.finish_reason {
                            match classify_finish(&reason, event.has_function_calls) {
                                Ok(finish_reason) if !event.has_function_calls => {
                                    terminal_reason = Some(finish_reason);
                                }
                                Ok(_) => {
                                    yield Err(protocol_error("Google text stream returned an unexpected function call"));
                                    return;
                                }
                                Err(error) => {
                                    yield Err(error);
                                    return;
                                }
                            }
                        }
                    }
                    Err(error) => {
                        yield Err(error);
                        return;
                    }
                }
            }
        }
        if pending.iter().any(|byte| !byte.is_ascii_whitespace()) {
            yield Err(protocol_error("Google SSE ended with an incomplete frame"));
            return;
        }
        match terminal_reason {
            Some(reason) => yield Ok(LLMChunk::final_chunk("", reason, latest_usage)),
            None => yield Err(protocol_error("Google SSE ended before a terminal finishReason")),
        }
    }
}

struct SseEvent {
    text: String,
    usage: Option<TokenUsage>,
    finish_reason: Option<String>,
    has_function_calls: bool,
}

fn parse_sse_frame(frame: &[u8]) -> Result<Option<SseEvent>, LLMError> {
    let text = std::str::from_utf8(frame)
        .map_err(|_| protocol_error("Google SSE frame is not valid UTF-8"))?;
    let mut data = String::new();
    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        if let Some(value) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(value.strip_prefix(' ').unwrap_or(value));
        }
    }
    if data.is_empty() {
        return Ok(None);
    }
    let value: Value = serde_json::from_str(&data)
        .map_err(|error| protocol_error(format!("invalid Google SSE JSON: {error}")))?;
    let usage = parse_usage(value.get("usageMetadata"))?;
    let candidate = value
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|values| values.first());
    let Some(candidate) = candidate else {
        if let Some(reason) = value
            .get("promptFeedback")
            .and_then(|feedback| feedback.get("blockReason"))
            .and_then(Value::as_str)
        {
            return Err(LLMError::ContentFiltered(format!(
                "Google blocked the prompt ({reason})"
            )));
        }
        return Ok(usage.map(|usage| SseEvent {
            text: String::new(),
            usage: Some(usage),
            finish_reason: None,
            has_function_calls: false,
        }));
    };
    let parts = candidate
        .get("content")
        .and_then(|content| content.get("parts"))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    Ok(Some(SseEvent {
        text: visible_text(parts)?,
        usage,
        finish_reason: candidate
            .get("finishReason")
            .and_then(Value::as_str)
            .map(str::to_string),
        has_function_calls: parts.iter().any(|part| part.get("functionCall").is_some()),
    }))
}

fn find_sse_frame_end(bytes: &[u8]) -> Option<(usize, usize)> {
    let crlf = bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| (index, 4));
    let lf = bytes
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|index| (index, 2));
    match (crlf, lf) {
        (Some(left), Some(right)) => Some(if left.0 <= right.0 { left } else { right }),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::{ProviderType, UnifiedLLMProvider};
    use ai_agents_core::{encode_native_tool_call_markers, encode_native_tool_result_marker};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::mpsc;

    async fn fake_http_server(
        responses: Vec<(&'static str, String)>,
    ) -> (String, mpsc::UnboundedReceiver<Value>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            for (content_type, response_body) in responses {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let header_end = loop {
                    let mut buffer = [0_u8; 4096];
                    let read = socket.read(&mut buffer).await.unwrap();
                    assert!(read > 0, "client closed before sending HTTP headers");
                    request.extend_from_slice(&buffer[..read]);
                    if let Some(index) = request.windows(4).position(|value| value == b"\r\n\r\n") {
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
                    .unwrap_or(0);
                while request.len() - header_end < content_length {
                    let mut buffer = [0_u8; 4096];
                    let read = socket.read(&mut buffer).await.unwrap();
                    assert!(read > 0, "client closed before sending HTTP body");
                    request.extend_from_slice(&buffer[..read]);
                }
                let body: Value =
                    serde_json::from_slice(&request[header_end..header_end + content_length])
                        .unwrap();
                sender.send(body).unwrap();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
                    response_body.len()
                );
                socket.write_all(response.as_bytes()).await.unwrap();
                socket.shutdown().await.unwrap();
            }
        });
        (format!("http://{address}/v1beta/"), receiver)
    }

    fn tool_request() -> LLMToolRequest {
        LLMToolRequest {
            tools: vec![LLMToolDefinition {
                name: "lookup".to_string(),
                description: "Look up a value".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": { "query": { "type": "string" } },
                    "required": ["query"],
                    "additionalProperties": false,
                }),
            }],
            choice: ToolChoice::Auto,
        }
    }

    #[test]
    fn google_adapter_owns_http_target_validation_and_normalization() {
        for endpoint in [
            "ftp://example.invalid/v1/",
            "https://user@example.invalid/v1/",
            "https://example.invalid/v1/?key=secret",
            "https://example.invalid/v1/#fragment",
        ] {
            assert!(parse_base_url(endpoint).is_err());
        }
        assert_eq!(
            parse_base_url("https://EXAMPLE.invalid/v1")
                .unwrap()
                .as_str(),
            "https://example.invalid/v1/"
        );
    }

    #[test]
    fn tool_schema_uses_parameters_json_schema_without_dropping_constraints() {
        let schema = json!({
            "type": "object",
            "properties": {
                "items": { "type": "array", "items": { "type": "integer" } }
            },
            "required": ["items"],
            "additionalProperties": false,
        });
        let value = map_tool_definitions(&[LLMToolDefinition {
            name: "calculate".to_string(),
            description: "Calculate".to_string(),
            input_schema: schema.clone(),
        }])
        .unwrap();
        let declaration = &value[0]["functionDeclarations"][0];
        assert_eq!(declaration["parametersJsonSchema"], schema);
        assert!(declaration.get("parameters").is_none());
    }

    #[test]
    fn usage_includes_thought_tokens_without_recomputing_reported_total() {
        let usage = parse_usage(Some(&json!({
            "promptTokenCount": 10,
            "candidatesTokenCount": 4,
            "thoughtsTokenCount": 6,
            "totalTokenCount": 20,
        })))
        .unwrap()
        .unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 10);
        assert_eq!(usage.total_tokens, 20);
    }

    #[test]
    fn google_struct_numbers_reject_integers_that_cannot_round_trip_through_double() {
        validate_json_numbers(&json!((1_u64 << 53)), "fixture").unwrap();
        let error = validate_json_numbers(&json!((1_u64 << 53) + 1), "fixture").unwrap_err();
        assert!(matches!(error, LLMError::Serialization(_)));
    }

    #[test]
    fn finish_reasons_preserve_length_and_reject_blocked_or_malformed_calls() {
        assert_eq!(classify_finish("STOP", false).unwrap(), FinishReason::Stop);
        assert_eq!(
            classify_finish("STOP", true).unwrap(),
            FinishReason::ToolCall
        );
        assert_eq!(
            classify_finish("MAX_TOKENS", false).unwrap(),
            FinishReason::Length
        );
        assert!(matches!(
            classify_finish("SAFETY", false).unwrap_err(),
            LLMError::ContentFiltered(_)
        ));
        assert!(matches!(
            classify_finish("MALFORMED_FUNCTION_CALL", true).unwrap_err(),
            LLMError::Serialization(_)
        ));
        let filtered = parse_response(
            json!({"candidates": [{"finishReason": "SAFETY"}]}),
            "gemini-2.5-flash",
            &Url::parse("https://generativelanguage.googleapis.com/v1beta/").unwrap(),
            false,
            &HashSet::new(),
        )
        .unwrap_err();
        assert!(matches!(filtered, LLMError::ContentFiltered(_)));
    }

    #[test]
    fn gemini_three_requires_signature_on_the_first_function_call_part() {
        let parts = vec![json!({ "functionCall": { "name": "lookup", "args": {} } })];
        let calls = parse_function_calls(&parts, &HashSet::new()).unwrap();
        let error = validate_required_signatures("gemini-3.7-flash", &parts, &calls).unwrap_err();
        assert!(matches!(error, LLMError::Serialization(_)));
        validate_required_signatures("gemini-2.5-flash", &parts, &calls).unwrap();
    }

    #[test]
    fn repeated_provider_id_gets_a_fresh_runtime_id_without_changing_wire_identity() {
        let parts = vec![json!({
            "functionCall": { "id": "provider-call", "name": "lookup", "args": {} },
            "thoughtSignature": "fixture"
        })];
        let existing = HashSet::from(["provider-call".to_string()]);
        let calls = parse_function_calls(&parts, &existing).unwrap();
        assert_ne!(calls[0].normalized.id, "provider-call");
        assert_eq!(calls[0].provider_id.as_deref(), Some("provider-call"));
        assert_eq!(parts[0]["functionCall"]["id"], "provider-call");
    }

    #[test]
    fn non_string_function_call_id_is_rejected_before_execution_projection() {
        for id in [json!(null), json!(7), json!(true), json!({"bad": true})] {
            let parts = vec![json!({
                "functionCall": {"id": id, "name": "lookup", "args": {}}
            })];
            let error = match parse_function_calls(&parts, &HashSet::new()) {
                Ok(_) => panic!("non-string functionCall id must fail"),
                Err(error) => error,
            };
            assert!(matches!(error, LLMError::Serialization(_)));
        }
    }

    #[test]
    fn result_values_are_wrapped_in_google_response_objects() {
        for output in [
            json!({"ok": true}),
            json!("text"),
            json!([1, 2]),
            json!(1),
            json!(true),
            Value::Null,
        ] {
            let tool_call = ToolCall {
                id: "call-1".to_string(),
                name: "lookup".to_string(),
                arguments: json!({}),
            };
            let result_content =
                encode_native_tool_result_marker(&tool_call, output.clone()).unwrap();
            let result = decode_native_tool_result_markers(&result_content)
                .unwrap()
                .unwrap()
                .pop()
                .unwrap();
            let pending = PendingExchange {
                model_content: json!({}),
                calls: vec![BoundCall {
                    call_id: "call-1".to_string(),
                    name: "lookup".to_string(),
                    provider_id: Some("provider-1".to_string()),
                }],
                results: vec![result],
                signed: true,
                readable_projection: "calls".to_string(),
            };
            let mut contents = Vec::new();
            flush_pending(&mut contents, Some(pending), false).unwrap();
            assert_eq!(
                contents[1]["parts"][0]["functionResponse"]["response"]["content"],
                output
            );
            assert!(contents[1]["parts"][0]["functionResponse"]["response"].is_object());
        }
    }

    #[test]
    fn sse_parser_keeps_usage_only_frames_separate_from_finish() {
        let usage = parse_sse_frame(
            br#"data: {"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4,"thoughtsTokenCount":6,"totalTokenCount":20}}"#,
        )
        .unwrap()
        .unwrap();
        assert!(usage.finish_reason.is_none());
        assert_eq!(usage.usage.unwrap().completion_tokens, 10);
    }

    #[tokio::test]
    async fn fake_http_second_request_preserves_signed_call_and_wraps_result() {
        let first = json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "id": "provider-call-1",
                            "name": "lookup",
                            "args": { "query": "company agents" }
                        },
                        "thoughtSignature": "fixture-signature"
                    }]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 2,
                "thoughtsTokenCount": 3,
                "totalTokenCount": 15
            },
            "modelVersion": "gemini-3.7-flash-001"
        })
        .to_string();
        let second = json!({
            "candidates": [{
                "content": { "role": "model", "parts": [{ "text": "Found it." }] },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 18,
                "candidatesTokenCount": 3,
                "thoughtsTokenCount": 0,
                "totalTokenCount": 21
            }
        })
        .to_string();
        let (base_url, mut requests) = fake_http_server(vec![
            ("application/json", first),
            ("application/json", second),
        ])
        .await;
        let provider = GoogleProvider::new(
            "gemini-3.7-flash".to_string(),
            "test-key".to_string(),
            Some(base_url),
            LLMConfig::default(),
        )
        .unwrap();

        let initial_messages = vec![ChatMessage::user("Find company agents")];
        let response = provider
            .complete_with_tools(&initial_messages, None, &tool_request())
            .await
            .unwrap();
        let calls = response.tool_calls().unwrap().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "provider-call-1");
        assert_eq!(response.usage.unwrap().completion_tokens, 5);
        let state = response.provider_state().unwrap().unwrap();
        let assistant = encode_native_tool_call_markers(&calls, Some(&state)).unwrap();
        let result = encode_native_tool_result_marker(&calls[0], json!(["alpha", "beta"])).unwrap();
        let continuation = vec![
            ChatMessage::user("Find company agents"),
            ChatMessage::assistant(assistant),
            ChatMessage::function("lookup", result),
        ];
        let final_response = provider
            .complete_with_tools(&continuation, None, &tool_request())
            .await
            .unwrap();
        assert_eq!(final_response.content, "Found it.");

        let first_request = requests.recv().await.unwrap();
        assert_eq!(
            first_request["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"]["additionalProperties"],
            false
        );
        let second_request = requests.recv().await.unwrap();
        assert_eq!(second_request["contents"][1], *state.model_content());
        let function_response = &second_request["contents"][2]["parts"][0]["functionResponse"];
        assert_eq!(function_response["id"], "provider-call-1");
        assert_eq!(function_response["name"], "lookup");
        assert_eq!(function_response["response"]["name"], "lookup");
        assert_eq!(
            function_response["response"]["content"],
            json!(["alpha", "beta"])
        );
    }

    #[tokio::test]
    async fn fake_http_sse_emits_one_terminal_chunk_with_latest_usage() {
        let body = concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\"}],",
            "\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":4,",
            "\"thoughtsTokenCount\":6,\"totalTokenCount\":20}}\n\n"
        )
        .to_string();
        let (base_url, _requests) = fake_http_server(vec![("text/event-stream", body)]).await;
        let provider = GoogleProvider::new(
            "gemini-3.7-flash".to_string(),
            "test-key".to_string(),
            Some(base_url),
            LLMConfig::default(),
        )
        .unwrap();
        let mut stream = provider
            .complete_stream(&[ChatMessage::user("hello")], None)
            .await
            .unwrap();
        let content = stream.next().await.unwrap().unwrap();
        assert_eq!(content.delta, "Hello");
        assert!(!content.is_final);
        let terminal = stream.next().await.unwrap().unwrap();
        assert!(terminal.is_final);
        assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
        assert_eq!(terminal.usage.unwrap().completion_tokens, 10);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn fake_http_sse_reports_premature_eof_as_terminal_error() {
        let body =
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]}}]}\n\n"
                .to_string();
        let (base_url, _requests) = fake_http_server(vec![("text/event-stream", body)]).await;
        let provider = GoogleProvider::new(
            "gemini-3.7-flash".to_string(),
            "test-key".to_string(),
            Some(base_url),
            LLMConfig::default(),
        )
        .unwrap();
        let mut stream = provider
            .complete_stream(&[ChatMessage::user("hello")], None)
            .await
            .unwrap();
        assert_eq!(stream.next().await.unwrap().unwrap().delta, "partial");
        let error = stream.next().await.unwrap().unwrap_err();
        assert!(provider.is_terminal_error(&error));
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn unified_google_complete_uses_the_first_party_endpoint() {
        let body = json!({
            "candidates": [{
                "content": { "role": "model", "parts": [{ "text": "first party" }] },
                "finishReason": "STOP"
            }]
        })
        .to_string();
        let (base_url, mut requests) = fake_http_server(vec![("application/json", body)]).await;
        let provider = UnifiedLLMProvider::from_spec_config(
            ProviderType::Google,
            "gemini-3.7-flash",
            Some("test-key".to_string()),
            Some(base_url),
            LLMConfig::default(),
        )
        .unwrap();
        let response = provider
            .complete(&[ChatMessage::user("hello")], None)
            .await
            .unwrap();
        assert_eq!(response.content, "first party");
        assert_eq!(
            requests.recv().await.unwrap()["contents"][0]["role"],
            "user"
        );
    }

    #[test]
    fn incomplete_signed_exchange_is_terminal_only_in_the_current_user_turn() {
        let base_url = Url::parse("https://example.invalid/v1beta/").unwrap();
        let model_content = json!({
            "role": "model",
            "parts": [{
                "functionCall": { "name": "lookup", "args": {} },
                "thoughtSignature": "secret-fixture-signature"
            }]
        });
        let parsed =
            parse_function_calls(model_content["parts"].as_array().unwrap(), &HashSet::new())
                .unwrap();
        let calls = parsed
            .iter()
            .map(|call| call.normalized.clone())
            .collect::<Vec<_>>();
        let state =
            build_provider_state("gemini-3.7-flash", &base_url, model_content, &parsed).unwrap();
        let assistant = encode_native_tool_call_markers(&calls, Some(&state)).unwrap();

        let current = vec![
            ChatMessage::user("old request"),
            ChatMessage::assistant(&assistant),
        ];
        let error = convert_messages(&current, "gemini-3.7-flash", &base_url).unwrap_err();
        assert!(matches!(error, LLMError::Serialization(_)));

        let historical = vec![
            ChatMessage::user("old request"),
            ChatMessage::assistant(assistant),
            ChatMessage::user("new request"),
        ];
        let (_, contents) = convert_messages(&historical, "gemini-3.7-flash", &base_url).unwrap();
        let serialized = serde_json::to_string(&contents).unwrap();
        assert!(!serialized.contains("secret-fixture-signature"));
        assert!(serialized.contains("incomplete"));
        assert!(serialized.contains("new request"));
    }
}

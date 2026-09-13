//! Provider-neutral storage and inspection for replay-bearing native tool history.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{ChatMessage, LLMError, LLMResponse, Role, ToolCall};

/// Reserved response-metadata and marker key for opaque provider replay state.
pub const NATIVE_PROVIDER_STATE_METADATA_KEY: &str = "_ai_agents_provider_state";
/// Marker key identifying a normalized native tool call in assistant history.
pub const NATIVE_TOOL_CALL_MARKER_KEY: &str = "_ai_agents_native_tool_call";
/// Marker key identifying a normalized native tool result in tool history.
pub const NATIVE_TOOL_RESULT_MARKER_KEY: &str = "_ai_agents_native_tool_result";
/// Current provider-state envelope version.
pub const NATIVE_PROVIDER_STATE_VERSION: u32 = 1;
/// Maximum serialized size accepted for one provider-state envelope.
pub const MAX_NATIVE_PROVIDER_STATE_BYTES: usize = 4 * 1024 * 1024;

fn protocol_error(message: impl Into<String>) -> LLMError {
    LLMError::Serialization(format!("invalid native history: {}", message.into()))
}

fn require_non_empty(value: &str, field: &str) -> Result<(), LLMError> {
    if value.is_empty() {
        Err(protocol_error(format!("{field} must not be empty")))
    } else {
        Ok(())
    }
}

/// Provider-owned canonical, credential-free destination identity attached to replay state.
///
/// Core deliberately does not interpret the endpoint syntax. The provider adapter that creates
/// the target must remove credentials and normalize transport-specific details before storing it.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeProviderTarget {
    endpoint: String,
    model: String,
}

impl NativeProviderTarget {
    /// Creates a target identity after checking its required provider-neutral fields.
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Result<Self, LLMError> {
        let target = Self {
            endpoint: endpoint.into(),
            model: model.into(),
        };
        target.validate()?;
        Ok(target)
    }

    fn validate(&self) -> Result<(), LLMError> {
        validate_target_component(&self.endpoint, "target.endpoint", 4096)?;
        validate_target_component(&self.model, "target.model", 1024)?;
        Ok(())
    }

    /// Returns the provider-owned canonical, credential-free destination identity.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Returns the requested model identity.
    pub fn model(&self) -> &str {
        &self.model
    }
}

fn validate_target_component(value: &str, field: &str, max_chars: usize) -> Result<(), LLMError> {
    require_non_empty(value, field)?;
    if value.chars().count() > max_chars {
        return Err(protocol_error(format!(
            "{field} exceeds {max_chars} characters"
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(protocol_error(format!(
            "{field} must not contain control characters"
        )));
    }
    Ok(())
}

impl std::fmt::Debug for NativeProviderTarget {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeProviderTarget")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

/// Correlates one normalized runtime call with its original provider part.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeCallBinding {
    call_id: String,
    part_index: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    provider_call_id: Option<String>,
}

impl NativeCallBinding {
    /// Creates a binding for a normalized call and original part index.
    pub fn new(call_id: impl Into<String>, part_index: usize) -> Result<Self, LLMError> {
        let binding = Self {
            call_id: call_id.into(),
            part_index,
            provider_call_id: None,
        };
        require_non_empty(&binding.call_id, "bindings.call_id")?;
        Ok(binding)
    }

    /// Records the provider's optional wire call ID without changing runtime identity.
    pub fn with_provider_call_id(mut self, provider_call_id: impl Into<String>) -> Self {
        let provider_call_id = provider_call_id.into();
        self.provider_call_id = (!provider_call_id.is_empty()).then_some(provider_call_id);
        self
    }

    /// Returns the normalized runtime call ID.
    pub fn call_id(&self) -> &str {
        &self.call_id
    }

    /// Returns the original part index in the provider content.
    pub fn part_index(&self) -> usize {
        self.part_index
    }

    /// Returns the provider's original optional call ID.
    pub fn provider_call_id(&self) -> Option<&str> {
        self.provider_call_id.as_deref()
    }
}

impl std::fmt::Debug for NativeCallBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeCallBinding")
            .field("call_id", &self.call_id)
            .field("part_index", &self.part_index)
            .field("has_provider_call_id", &self.provider_call_id.is_some())
            .finish()
    }
}

/// Versioned opaque provider content required to replay one model response.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeProviderState {
    version: u32,
    exchange_id: String,
    provider: String,
    api: String,
    target: NativeProviderTarget,
    model_content: Value,
    bindings: Vec<NativeCallBinding>,
}

impl NativeProviderState {
    /// Creates and validates a provider-neutral replay envelope.
    pub fn new(
        exchange_id: impl Into<String>,
        provider: impl Into<String>,
        api: impl Into<String>,
        target: NativeProviderTarget,
        model_content: Value,
        bindings: Vec<NativeCallBinding>,
    ) -> Result<Self, LLMError> {
        let state = Self {
            version: NATIVE_PROVIDER_STATE_VERSION,
            exchange_id: exchange_id.into(),
            provider: provider.into(),
            api: api.into(),
            target,
            model_content,
            bindings,
        };
        state.validate()?;
        Ok(state)
    }

    /// Decodes a reserved metadata value and rejects unsupported or oversized state.
    pub fn from_value(value: Value) -> Result<Self, LLMError> {
        let state: Self = serde_json::from_value(value)
            .map_err(|error| protocol_error(format!("provider state decode failed: {error}")))?;
        state.validate()?;
        Ok(state)
    }

    /// Encodes this state for response metadata or the first native call marker.
    pub fn to_value(&self) -> Result<Value, LLMError> {
        self.validate()?;
        serde_json::to_value(self)
            .map_err(|error| protocol_error(format!("provider state encode failed: {error}")))
    }

    /// Validates the version, required identities, binding uniqueness, and size bound.
    pub fn validate(&self) -> Result<(), LLMError> {
        if self.version != NATIVE_PROVIDER_STATE_VERSION {
            return Err(protocol_error(format!(
                "unsupported provider state version {}",
                self.version
            )));
        }
        require_non_empty(&self.exchange_id, "exchange_id")?;
        require_non_empty(&self.provider, "provider")?;
        require_non_empty(&self.api, "api")?;
        self.target.validate()?;
        if !self.model_content.is_object() {
            return Err(protocol_error("model_content must be an object"));
        }

        let mut call_ids = HashSet::with_capacity(self.bindings.len());
        let mut part_indexes = HashSet::with_capacity(self.bindings.len());
        for binding in &self.bindings {
            require_non_empty(binding.call_id(), "bindings.call_id")?;
            if !call_ids.insert(binding.call_id()) {
                return Err(protocol_error(format!(
                    "duplicate binding call ID '{}'",
                    binding.call_id()
                )));
            }
            if !part_indexes.insert(binding.part_index()) {
                return Err(protocol_error(format!(
                    "duplicate binding part index {}",
                    binding.part_index()
                )));
            }
        }

        let bytes = serde_json::to_vec(self)
            .map_err(|error| protocol_error(format!("provider state encode failed: {error}")))?;
        if bytes.len() > MAX_NATIVE_PROVIDER_STATE_BYTES {
            return Err(protocol_error(format!(
                "provider state exceeds {MAX_NATIVE_PROVIDER_STATE_BYTES} bytes"
            )));
        }
        Ok(())
    }

    /// Requires a one-to-one ordered binding for every normalized call.
    pub fn validate_for_calls(&self, calls: &[ToolCall]) -> Result<(), LLMError> {
        self.validate()?;
        if self.bindings.len() != calls.len() {
            return Err(protocol_error(format!(
                "provider state has {} bindings for {} calls",
                self.bindings.len(),
                calls.len()
            )));
        }
        for (binding, call) in self.bindings.iter().zip(calls) {
            if binding.call_id() != call.id {
                return Err(protocol_error(format!(
                    "binding call ID '{}' does not match normalized call '{}'",
                    binding.call_id(),
                    call.id
                )));
            }
        }
        Ok(())
    }

    /// Returns the envelope format version.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Returns the unique model-response occurrence ID.
    pub fn exchange_id(&self) -> &str {
        &self.exchange_id
    }

    /// Returns the provider identity.
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// Returns the provider API-family identity.
    pub fn api(&self) -> &str {
        &self.api
    }

    /// Returns the credential-free replay target.
    pub fn target(&self) -> &NativeProviderTarget {
        &self.target
    }

    /// Returns the opaque original model content without transforming it.
    pub fn model_content(&self) -> &Value {
        &self.model_content
    }

    /// Returns the ordered runtime-call-to-part bindings.
    pub fn bindings(&self) -> &[NativeCallBinding] {
        &self.bindings
    }

    /// Returns the serialized byte length used by diagnostics and limits.
    pub fn serialized_len(&self) -> Result<usize, LLMError> {
        self.validate()?;
        serde_json::to_vec(self)
            .map(|bytes| bytes.len())
            .map_err(|error| protocol_error(format!("provider state encode failed: {error}")))
    }
}

impl std::fmt::Debug for NativeProviderState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeProviderState")
            .field("version", &self.version)
            .field("exchange_id", &self.exchange_id)
            .field("provider", &self.provider)
            .field("api", &self.api)
            .field("target", &self.target)
            .field("binding_count", &self.bindings.len())
            .finish_non_exhaustive()
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeToolCallMarker {
    #[serde(rename = "_ai_agents_native_tool_call")]
    native_tool_call: bool,
    id: String,
    tool: String,
    arguments: Value,
    #[serde(
        default,
        rename = "_ai_agents_provider_state",
        skip_serializing_if = "Option::is_none"
    )]
    provider_state: Option<NativeProviderState>,
}

/// Decoded native call markers and their optional replay state.
#[derive(Clone, PartialEq)]
pub struct NativeToolCallBatch {
    calls: Vec<ToolCall>,
    provider_state: Option<NativeProviderState>,
}

impl NativeToolCallBatch {
    /// Returns normalized calls in provider declaration order.
    pub fn calls(&self) -> &[ToolCall] {
        &self.calls
    }

    /// Returns replay state attached to the first marker, when present.
    pub fn provider_state(&self) -> Option<&NativeProviderState> {
        self.provider_state.as_ref()
    }

    /// Consumes the batch into normalized calls and optional replay state.
    pub fn into_parts(self) -> (Vec<ToolCall>, Option<NativeProviderState>) {
        (self.calls, self.provider_state)
    }
}

impl std::fmt::Debug for NativeToolCallBatch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeToolCallBatch")
            .field("call_count", &self.calls.len())
            .field("has_provider_state", &self.provider_state.is_some())
            .finish()
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeToolResultMarker {
    #[serde(rename = "_ai_agents_native_tool_result")]
    native_tool_result: bool,
    id: String,
    tool: String,
    output: Value,
}

/// Decoded provider-neutral native tool result.
#[derive(Clone, PartialEq)]
pub struct NativeToolResult {
    id: String,
    tool: String,
    output: Value,
}

impl NativeToolResult {
    /// Returns the normalized call ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the canonical or provider-visible tool name retained by the marker.
    pub fn tool(&self) -> &str {
        &self.tool
    }

    /// Returns the unmodified model-facing result value.
    pub fn output(&self) -> &Value {
        &self.output
    }

    /// Consumes the result into its ID, tool, and output value.
    pub fn into_parts(self) -> (String, String, Value) {
        (self.id, self.tool, self.output)
    }
}

impl std::fmt::Debug for NativeToolResult {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeToolResult")
            .field("id", &self.id)
            .field("tool", &self.tool)
            .finish_non_exhaustive()
    }
}

fn marker_values(content: &str, marker_key: &str) -> Result<Option<Vec<Value>>, LLMError> {
    let value = match serde_json::from_str::<Value>(content) {
        Ok(value) => value,
        Err(_) => {
            let contains_reserved_key = content.contains(marker_key)
                || content.contains(NATIVE_PROVIDER_STATE_METADATA_KEY);
            if contains_reserved_key {
                return Err(protocol_error(
                    "reserved native marker content is not valid JSON",
                ));
            }
            return Ok(None);
        }
    };
    let values = match value {
        Value::Object(map) => vec![Value::Object(map)],
        Value::Array(values) if !values.is_empty() => values,
        Value::Array(_) | Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            return Ok(None);
        }
    };

    let has_reserved_key = values.iter().any(|value| {
        value.get(marker_key).is_some()
            || (marker_key == NATIVE_TOOL_CALL_MARKER_KEY
                && value.get(NATIVE_PROVIDER_STATE_METADATA_KEY).is_some())
    });
    if !has_reserved_key {
        return Ok(None);
    }
    if values.iter().any(|value| {
        value.get(marker_key).and_then(Value::as_bool) != Some(true) || !value.is_object()
    }) {
        return Err(protocol_error(format!(
            "every value in a reserved '{marker_key}' batch must be a marked object"
        )));
    }
    Ok(Some(values))
}

/// Encodes normalized native calls, attaching validated replay state only to the first marker.
pub fn encode_native_tool_call_markers(
    calls: &[ToolCall],
    provider_state: Option<&NativeProviderState>,
) -> Result<String, LLMError> {
    if calls.is_empty() {
        return Err(protocol_error("native tool call batch must not be empty"));
    }
    if let Some(state) = provider_state {
        state.validate_for_calls(calls)?;
    }

    let mut seen = HashSet::with_capacity(calls.len());
    let markers = calls
        .iter()
        .enumerate()
        .map(|(index, call)| {
            require_non_empty(&call.id, "tool call id")?;
            require_non_empty(&call.name, "tool call name")?;
            if !seen.insert(call.id.as_str()) {
                return Err(protocol_error(format!(
                    "duplicate normalized call ID '{}'",
                    call.id
                )));
            }
            Ok(NativeToolCallMarker {
                native_tool_call: true,
                id: call.id.clone(),
                tool: call.name.clone(),
                arguments: call.arguments.clone(),
                provider_state: (index == 0).then(|| provider_state.cloned()).flatten(),
            })
        })
        .collect::<Result<Vec<_>, LLMError>>()?;

    if markers.len() == 1 {
        serde_json::to_string(&markers[0])
    } else {
        serde_json::to_string(&markers)
    }
    .map_err(|error| protocol_error(format!("tool call marker encode failed: {error}")))
}

/// Decodes legacy or replay-bearing native call markers without inspecting arbitrary text.
pub fn decode_native_tool_call_markers(
    content: &str,
) -> Result<Option<NativeToolCallBatch>, LLMError> {
    let Some(values) = marker_values(content, NATIVE_TOOL_CALL_MARKER_KEY)? else {
        return Ok(None);
    };

    let mut calls = Vec::with_capacity(values.len());
    let mut provider_state = None;
    let mut seen = HashSet::with_capacity(values.len());
    for (index, value) in values.into_iter().enumerate() {
        let marker: NativeToolCallMarker = serde_json::from_value(value)
            .map_err(|error| protocol_error(format!("tool call marker decode failed: {error}")))?;
        require_non_empty(&marker.id, "tool call marker id")?;
        require_non_empty(&marker.tool, "tool call marker tool")?;
        if !seen.insert(marker.id.clone()) {
            return Err(protocol_error(format!(
                "duplicate tool call marker ID '{}'",
                marker.id
            )));
        }
        if marker.provider_state.is_some() && index != 0 {
            return Err(protocol_error(
                "provider state may appear only on the first tool call marker",
            ));
        }
        if let Some(state) = marker.provider_state {
            state.validate()?;
            provider_state = Some(state);
        }
        calls.push(ToolCall {
            id: marker.id,
            name: marker.tool,
            arguments: marker.arguments,
        });
    }
    if let Some(state) = provider_state.as_ref() {
        state.validate_for_calls(&calls)?;
    }
    Ok(Some(NativeToolCallBatch {
        calls,
        provider_state,
    }))
}

/// Encodes one native tool result marker without interpreting its output value.
pub fn encode_native_tool_result_marker(
    call: &ToolCall,
    output: Value,
) -> Result<String, LLMError> {
    require_non_empty(&call.id, "tool result call id")?;
    require_non_empty(&call.name, "tool result tool name")?;
    serde_json::to_string(&NativeToolResultMarker {
        native_tool_result: true,
        id: call.id.clone(),
        tool: call.name.clone(),
        output,
    })
    .map_err(|error| protocol_error(format!("tool result marker encode failed: {error}")))
}

/// Decodes one or more native result markers in their stored order.
pub fn decode_native_tool_result_markers(
    content: &str,
) -> Result<Option<Vec<NativeToolResult>>, LLMError> {
    let Some(values) = marker_values(content, NATIVE_TOOL_RESULT_MARKER_KEY)? else {
        return Ok(None);
    };
    let mut seen = HashSet::with_capacity(values.len());
    let results = values
        .into_iter()
        .map(|value| {
            let marker: NativeToolResultMarker =
                serde_json::from_value(value).map_err(|error| {
                    protocol_error(format!("tool result marker decode failed: {error}"))
                })?;
            require_non_empty(&marker.id, "tool result marker id")?;
            require_non_empty(&marker.tool, "tool result marker tool")?;
            if !seen.insert(marker.id.clone()) {
                return Err(protocol_error(format!(
                    "duplicate tool result marker ID '{}'",
                    marker.id
                )));
            }
            Ok(NativeToolResult {
                id: marker.id,
                tool: marker.tool,
                output: marker.output,
            })
        })
        .collect::<Result<Vec<_>, LLMError>>()?;
    Ok(Some(results))
}

fn encode_execution_calls(calls: &[ToolCall]) -> Result<String, LLMError> {
    encode_native_tool_call_markers(calls, None)
}

fn encode_results(results: &[NativeToolResult]) -> Result<String, LLMError> {
    let markers = results
        .iter()
        .map(|result| NativeToolResultMarker {
            native_tool_result: true,
            id: result.id.clone(),
            tool: result.tool.clone(),
            output: result.output.clone(),
        })
        .collect::<Vec<_>>();
    if markers.len() == 1 {
        serde_json::to_string(&markers[0])
    } else {
        serde_json::to_string(&markers)
    }
    .map_err(|error| protocol_error(format!("tool result projection failed: {error}")))
}

/// Removes opaque provider state while preserving the existing executable marker representation.
pub fn native_execution_projection(content: &str) -> Result<String, LLMError> {
    if let Some(batch) = decode_native_tool_call_markers(content)? {
        return encode_execution_calls(batch.calls());
    }
    if let Some(results) = decode_native_tool_result_markers(content)? {
        return encode_results(&results);
    }
    Ok(content.to_string())
}

/// Converts native control markers into a provider-state-free representation for auxiliary models.
pub fn native_readable_projection(content: &str) -> Result<String, LLMError> {
    if let Some(batch) = decode_native_tool_call_markers(content)? {
        let calls = batch
            .calls()
            .iter()
            .map(|call| {
                serde_json::json!({
                    "id": call.id,
                    "tool": call.name,
                    "arguments": call.arguments,
                })
            })
            .collect::<Vec<_>>();
        return serde_json::to_string(&serde_json::json!({ "native_tool_calls": calls }))
            .map_err(|error| protocol_error(format!("readable call projection failed: {error}")));
    }
    if let Some(results) = decode_native_tool_result_markers(content)? {
        let results = results
            .iter()
            .map(|result| {
                serde_json::json!({
                    "id": result.id(),
                    "tool": result.tool(),
                    "output": result.output(),
                })
            })
            .collect::<Vec<_>>();
        return serde_json::to_string(&serde_json::json!({ "native_tool_results": results }))
            .map_err(|error| {
                protocol_error(format!("readable result projection failed: {error}"))
            });
    }
    Ok(content.to_string())
}

/// Removes opaque replay state before first-party observation and UI redaction.
pub fn native_observation_projection(content: &str) -> Result<String, LLMError> {
    native_execution_projection(content)
}

/// Reads and validates provider state from the reserved response metadata key.
pub fn response_provider_state(
    response: &LLMResponse,
) -> Result<Option<NativeProviderState>, LLMError> {
    response
        .metadata
        .get(NATIVE_PROVIDER_STATE_METADATA_KEY)
        .cloned()
        .map(NativeProviderState::from_value)
        .transpose()
}

/// Stores validated provider state under the sole reserved response metadata key.
pub fn set_response_provider_state(
    response: &mut LLMResponse,
    state: NativeProviderState,
) -> Result<(), LLMError> {
    response.metadata.insert(
        NATIVE_PROVIDER_STATE_METADATA_KEY.to_string(),
        state.to_value()?,
    );
    Ok(())
}

/// Removes provider state only after validating the stored value.
pub fn take_response_provider_state(
    response: &mut LLMResponse,
) -> Result<Option<NativeProviderState>, LLMError> {
    let state = response_provider_state(response)?;
    if state.is_some() {
        response.metadata.remove(NATIVE_PROVIDER_STATE_METADATA_KEY);
    }
    Ok(state)
}

/// One signed assistant exchange and the result markers correlated with it.
#[derive(Clone, PartialEq)]
pub struct NativeExchange {
    state: NativeProviderState,
    message_start: usize,
    message_end: usize,
    call_ids: Vec<String>,
    call_tools: Vec<String>,
    result_ids: Vec<String>,
}

impl NativeExchange {
    /// Returns the replay state identifying this model-response occurrence.
    pub fn state(&self) -> &NativeProviderState {
        &self.state
    }

    /// Returns the signed assistant message index.
    pub fn message_start(&self) -> usize {
        self.message_start
    }

    /// Returns the exclusive end after the last correlated result marker.
    pub fn message_end(&self) -> usize {
        self.message_end
    }

    /// Returns normalized calls in declaration order.
    pub fn call_ids(&self) -> &[String] {
        &self.call_ids
    }

    /// Returns recorded result IDs in stored order.
    pub fn result_ids(&self) -> &[String] {
        &self.result_ids
    }

    /// Returns true only when every call has exactly one stored result.
    pub fn is_complete(&self) -> bool {
        self.call_ids == self.result_ids
    }

    /// Returns calls that do not have a stored result marker.
    pub fn missing_result_ids(&self) -> Vec<&str> {
        let results: HashSet<&str> = self.result_ids.iter().map(String::as_str).collect();
        self.call_ids
            .iter()
            .map(String::as_str)
            .filter(|call_id| !results.contains(call_id))
            .collect()
    }
}

impl std::fmt::Debug for NativeExchange {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeExchange")
            .field("exchange_id", &self.state.exchange_id())
            .field("message_start", &self.message_start)
            .field("message_end", &self.message_end)
            .field("call_count", &self.call_ids.len())
            .field("result_count", &self.result_ids.len())
            .finish()
    }
}

/// Provider-neutral signed-history boundaries used by memory and runtime admission.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeHistoryInspection {
    message_count: usize,
    exchanges: Vec<NativeExchange>,
    protected_suffix_start: Option<usize>,
}

impl NativeHistoryInspection {
    /// Returns every signed exchange in history order.
    pub fn exchanges(&self) -> &[NativeExchange] {
        &self.exchanges
    }

    /// Returns the first message of the latest user turn containing signed history.
    pub fn protected_suffix_start(&self) -> Option<usize> {
        self.protected_suffix_start
    }

    /// Reports whether removing exactly this prefix preserves every exchange and protected suffix.
    pub fn is_safe_prefix_len(&self, count: usize) -> bool {
        if count > self.message_count
            || self
                .protected_suffix_start
                .is_some_and(|protected| count > protected)
        {
            return false;
        }
        self.exchanges
            .iter()
            .all(|exchange| count <= exchange.message_start() || count >= exchange.message_end())
    }

    /// Finds the greatest safe removable prefix no larger than `requested`.
    pub fn safe_prefix_len_at_most(&self, requested: usize) -> usize {
        let mut count = requested.min(self.message_count);
        while count > 0 && !self.is_safe_prefix_len(count) {
            count -= 1;
        }
        count
    }

    /// Finds the smallest safe removable prefix at least `required` without entering the protected suffix.
    pub fn safe_prefix_len_at_least(&self, required: usize) -> Option<usize> {
        (required..=self.message_count).find(|count| self.is_safe_prefix_len(*count))
    }
}

fn finish_exchange(exchanges: &mut Vec<NativeExchange>, active: &mut Option<NativeExchange>) {
    if let Some(exchange) = active.take() {
        exchanges.push(exchange);
    }
}

// Retains just enough legacy unsigned identity to reject orphaned or mismatched native results without treating it as replay state.
struct UnsignedExchange {
    calls: Vec<(String, String)>,
    result_ids: HashSet<String>,
}

/// Inspects signed assistant exchanges without treating user-controlled marker text as authority.
pub fn inspect_native_history(
    messages: &[ChatMessage],
) -> Result<NativeHistoryInspection, LLMError> {
    let mut exchanges = Vec::new();
    let mut active: Option<NativeExchange> = None;
    let mut active_unsigned: Option<UnsignedExchange> = None;
    let mut seen_exchange_ids = HashSet::new();
    let mut latest_user_index = None;
    let mut exchange_user_indexes = HashMap::<String, usize>::new();

    for (index, message) in messages.iter().enumerate() {
        match message.role {
            Role::User => {
                finish_exchange(&mut exchanges, &mut active);
                active_unsigned = None;
                latest_user_index = Some(index);
            }
            Role::Assistant => {
                let Some(batch) = decode_native_tool_call_markers(&message.content)? else {
                    finish_exchange(&mut exchanges, &mut active);
                    active_unsigned = None;
                    continue;
                };
                let Some(state) = batch.provider_state().cloned() else {
                    finish_exchange(&mut exchanges, &mut active);
                    active_unsigned = Some(UnsignedExchange {
                        calls: batch
                            .calls()
                            .iter()
                            .map(|call| (call.id.clone(), call.name.clone()))
                            .collect(),
                        result_ids: HashSet::new(),
                    });
                    continue;
                };
                finish_exchange(&mut exchanges, &mut active);
                active_unsigned = None;
                let user_index = latest_user_index.ok_or_else(|| {
                    protocol_error(format!(
                        "signed exchange '{}' has no preceding user boundary",
                        state.exchange_id()
                    ))
                })?;
                if !seen_exchange_ids.insert(state.exchange_id().to_string()) {
                    return Err(protocol_error(format!(
                        "duplicate exchange ID '{}'",
                        state.exchange_id()
                    )));
                }
                exchange_user_indexes.insert(state.exchange_id().to_string(), user_index);
                active = Some(NativeExchange {
                    state,
                    message_start: index,
                    message_end: index + 1,
                    call_ids: batch.calls().iter().map(|call| call.id.clone()).collect(),
                    call_tools: batch.calls().iter().map(|call| call.name.clone()).collect(),
                    result_ids: Vec::new(),
                });
            }
            Role::Tool | Role::Function => {
                let Some(results) = decode_native_tool_result_markers(&message.content)? else {
                    continue;
                };
                if active.is_none() {
                    let Some(unsigned) = active_unsigned.as_mut() else {
                        return Err(protocol_error(
                            "native tool result has no active call exchange",
                        ));
                    };
                    for result in results {
                        let Some((_, tool)) =
                            unsigned.calls.iter().find(|(id, _)| id == result.id())
                        else {
                            return Err(protocol_error(format!(
                                "result '{}' does not belong to its unsigned call exchange",
                                result.id()
                            )));
                        };
                        if tool != result.tool()
                            || !unsigned.result_ids.insert(result.id().to_string())
                        {
                            return Err(protocol_error(format!(
                                "result '{}' does not match its unsigned call",
                                result.id()
                            )));
                        }
                    }
                    continue;
                }
                let exchange = active.as_mut().expect("checked above");
                for result in results {
                    let Some(call_index) = exchange
                        .call_ids
                        .iter()
                        .position(|call_id| call_id == result.id())
                    else {
                        return Err(protocol_error(format!(
                            "result '{}' does not belong to active exchange '{}'",
                            result.id(),
                            exchange.state.exchange_id()
                        )));
                    };
                    if exchange.call_tools[call_index] != result.tool() {
                        return Err(protocol_error(format!(
                            "result '{}' tool does not match its active call",
                            result.id()
                        )));
                    }
                    if exchange
                        .result_ids
                        .iter()
                        .any(|call_id| call_id == result.id())
                    {
                        return Err(protocol_error(format!(
                            "duplicate result '{}' in exchange '{}'",
                            result.id(),
                            exchange.state.exchange_id()
                        )));
                    }
                    let expected = exchange.call_ids.get(exchange.result_ids.len());
                    if expected.map(String::as_str) != Some(result.id()) {
                        return Err(protocol_error(format!(
                            "result '{}' is out of declaration order in exchange '{}'",
                            result.id(),
                            exchange.state.exchange_id()
                        )));
                    }
                    exchange.result_ids.push(result.id().to_string());
                }
                exchange.message_end = index + 1;
            }
            Role::System => {}
        }
    }
    finish_exchange(&mut exchanges, &mut active);

    let protected_suffix_start = latest_user_index.and_then(|latest_user| {
        exchanges
            .iter()
            .filter(|exchange| exchange.message_start() > latest_user)
            .filter_map(|exchange| {
                exchange_user_indexes
                    .get(exchange.state().exchange_id())
                    .copied()
            })
            .min()
    });

    Ok(NativeHistoryInspection {
        message_count: messages.len(),
        exchanges,
        protected_suffix_start,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FinishReason;

    fn call(id: &str, argument: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: "lookup".to_string(),
            arguments: serde_json::json!({ "query": argument }),
        }
    }

    fn state(exchange_id: &str, calls: &[ToolCall]) -> NativeProviderState {
        let parts = calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                let mut part = serde_json::json!({
                    "functionCall": {
                        "name": call.name,
                        "args": call.arguments,
                    }
                });
                if index == 0 {
                    part["thoughtSignature"] = Value::String("fixture-signature".to_string());
                }
                part
            })
            .collect::<Vec<_>>();
        NativeProviderState::new(
            exchange_id,
            "google",
            "generateContent",
            NativeProviderTarget::new("https://example.invalid/v1beta/", "fixture-model").unwrap(),
            serde_json::json!({ "role": "model", "parts": parts }),
            calls
                .iter()
                .enumerate()
                .map(|(index, call)| NativeCallBinding::new(&call.id, index).unwrap())
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn response_state_round_trip_uses_only_reserved_metadata() {
        let calls = vec![call("call-1", "Paris")];
        let state = state("exchange-1", &calls);
        let mut response = LLMResponse::new("", FinishReason::ToolCall);
        set_response_provider_state(&mut response, state.clone()).unwrap();

        assert_eq!(
            response_provider_state(&response).unwrap(),
            Some(state.clone())
        );
        assert_eq!(
            take_response_provider_state(&mut response).unwrap(),
            Some(state)
        );
        assert!(
            !response
                .metadata
                .contains_key(NATIVE_PROVIDER_STATE_METADATA_KEY)
        );
    }

    #[test]
    fn memory_snapshot_serde_preserves_signed_provider_state() {
        let calls = vec![call("call-snapshot", "Seoul")];
        let marker =
            encode_native_tool_call_markers(&calls, Some(&state("exchange-snapshot", &calls)))
                .unwrap();
        let snapshot = crate::MemorySnapshot::new(vec![
            ChatMessage::user("remember"),
            ChatMessage::assistant(marker),
            ChatMessage::function(
                "lookup",
                encode_native_tool_result_marker(&calls[0], serde_json::json!({"ok": true}))
                    .unwrap(),
            ),
        ]);

        let encoded = serde_json::to_string(&snapshot).unwrap();
        let restored: crate::MemorySnapshot = serde_json::from_str(&encoded).unwrap();
        let batch = decode_native_tool_call_markers(&restored.messages[1].content)
            .unwrap()
            .unwrap();

        assert_eq!(
            batch.provider_state().unwrap().exchange_id(),
            "exchange-snapshot"
        );
        assert!(
            inspect_native_history(&restored.messages)
                .unwrap()
                .exchanges()[0]
                .is_complete()
        );
    }

    #[test]
    fn malformed_response_state_is_not_removed_or_ignored() {
        let mut response = LLMResponse::new("", FinishReason::ToolCall);
        response.metadata.insert(
            NATIVE_PROVIDER_STATE_METADATA_KEY.to_string(),
            serde_json::json!({ "version": 999 }),
        );

        assert!(response_provider_state(&response).is_err());
        assert!(take_response_provider_state(&mut response).is_err());
        assert!(
            response
                .metadata
                .contains_key(NATIVE_PROVIDER_STATE_METADATA_KEY)
        );
    }

    #[test]
    fn oversized_provider_state_is_rejected_without_exposing_payload() {
        let error = NativeProviderState::new(
            "exchange-oversized",
            "google",
            "generateContent",
            NativeProviderTarget::new("https://example.invalid/v1beta/", "fixture-model").unwrap(),
            serde_json::json!({
                "role": "model",
                "parts": [{ "thoughtSignature": "x".repeat(MAX_NATIVE_PROVIDER_STATE_BYTES) }]
            }),
            vec![NativeCallBinding::new("call-1", 0).unwrap()],
        )
        .unwrap_err();

        assert!(error.to_string().contains("exceeds"));
        assert!(!error.to_string().contains(&"x".repeat(32)));
    }

    #[test]
    fn call_markers_round_trip_parallel_state_once() {
        let calls = vec![call("call-1", "Paris"), call("call-2", "London")];
        let state = state("exchange-parallel", &calls);
        let content = encode_native_tool_call_markers(&calls, Some(&state)).unwrap();
        let raw: Value = serde_json::from_str(&content).unwrap();
        let markers = raw.as_array().unwrap();
        assert!(markers[0].get(NATIVE_PROVIDER_STATE_METADATA_KEY).is_some());
        assert!(markers[1].get(NATIVE_PROVIDER_STATE_METADATA_KEY).is_none());

        let decoded = decode_native_tool_call_markers(&content).unwrap().unwrap();
        assert_eq!(decoded.calls(), calls);
        assert_eq!(decoded.provider_state(), Some(&state));
    }

    #[test]
    fn legacy_unsigned_marker_remains_supported() {
        let content = serde_json::json!({
            "_ai_agents_native_tool_call": true,
            "id": "legacy-call",
            "tool": "lookup",
            "arguments": { "query": "legacy" }
        })
        .to_string();

        let decoded = decode_native_tool_call_markers(&content).unwrap().unwrap();
        assert_eq!(decoded.calls(), &[call("legacy-call", "legacy")]);
        assert!(decoded.provider_state().is_none());
    }

    #[test]
    fn reserved_state_on_non_first_marker_fails_closed() {
        let calls = vec![call("call-1", "Paris"), call("call-2", "London")];
        let state = state("exchange-bad", &calls).to_value().unwrap();
        let content = serde_json::json!([
            {
                "_ai_agents_native_tool_call": true,
                "id": "call-1",
                "tool": "lookup",
                "arguments": { "query": "Paris" }
            },
            {
                "_ai_agents_native_tool_call": true,
                "id": "call-2",
                "tool": "lookup",
                "arguments": { "query": "London" },
                "_ai_agents_provider_state": state
            }
        ])
        .to_string();

        assert!(decode_native_tool_call_markers(&content).is_err());
    }

    #[test]
    fn malformed_reserved_marker_does_not_fall_back_to_text() {
        let content = serde_json::json!({
            "_ai_agents_native_tool_call": true,
            "id": "",
            "tool": "lookup",
            "arguments": {}
        })
        .to_string();
        assert!(decode_native_tool_call_markers(&content).is_err());
    }

    #[test]
    fn truncated_reserved_marker_fails_without_echoing_opaque_content() {
        let content = r#"{"_ai_agents_native_tool_call":true,"_ai_agents_provider_state":{"thoughtSignature":"secret""#;
        let error = decode_native_tool_call_markers(content).unwrap_err();

        assert!(!error.to_string().contains("secret"));
        assert!(native_observation_projection(content).is_err());
    }

    #[test]
    fn history_rejects_orphan_results_and_wrong_tool_names() {
        let call = call("call-integrity", "Paris");
        let orphan = vec![ChatMessage::function(
            "lookup",
            encode_native_tool_result_marker(&call, serde_json::json!("orphan")).unwrap(),
        )];
        assert!(inspect_native_history(&orphan).is_err());

        let assistant = encode_native_tool_call_markers(
            std::slice::from_ref(&call),
            Some(&state("exchange-integrity", std::slice::from_ref(&call))),
        )
        .unwrap();
        let wrong_tool = ToolCall {
            name: "different_tool".to_string(),
            ..call
        };
        let messages = vec![
            ChatMessage::user("current"),
            ChatMessage::assistant(assistant),
            ChatMessage::function(
                "different_tool",
                encode_native_tool_result_marker(&wrong_tool, serde_json::json!("wrong")).unwrap(),
            ),
        ];
        assert!(inspect_native_history(&messages).is_err());
    }

    #[test]
    fn provider_target_accepts_provider_owned_non_http_identity() {
        assert_eq!(
            NativeProviderTarget::new("unix:///run/provider.sock", "local-model")
                .unwrap()
                .endpoint(),
            "unix:///run/provider.sock"
        );
        assert!(NativeProviderTarget::new("", "model").is_err());
        assert!(NativeProviderTarget::new("target\nwith-control", "model").is_err());
        assert!(NativeProviderTarget::new("target", "").is_err());
    }

    #[test]
    fn projections_remove_opaque_state_without_mutating_source() {
        let calls = vec![call("call-1", "Paris")];
        let content =
            encode_native_tool_call_markers(&calls, Some(&state("exchange-projection", &calls)))
                .unwrap();
        let original = content.clone();

        let execution = native_execution_projection(&content).unwrap();
        let readable = native_readable_projection(&content).unwrap();
        let observation = native_observation_projection(&content).unwrap();
        for projected in [&execution, &readable, &observation] {
            assert!(!projected.contains("fixture-signature"));
            assert!(!projected.contains(NATIVE_PROVIDER_STATE_METADATA_KEY));
        }
        assert_eq!(content, original);
        assert!(readable.contains("native_tool_calls"));
    }

    #[test]
    fn provider_state_debug_does_not_expose_opaque_content() {
        let calls = vec![call("call-1", "Paris")];
        let state = state("exchange-debug", &calls);
        let debug = format!("{state:?}");

        assert!(debug.contains("exchange-debug"));
        assert!(!debug.contains("fixture-signature"));
        assert!(!debug.contains("example.invalid"));
    }

    #[test]
    fn result_codec_preserves_every_json_value_shape() {
        let call = call("call-1", "Paris");
        for output in [
            serde_json::json!({ "answer": 1 }),
            serde_json::json!("text"),
            serde_json::json!([1, 2]),
            serde_json::json!(7),
            serde_json::json!(true),
            Value::Null,
        ] {
            let content = encode_native_tool_result_marker(&call, output.clone()).unwrap();
            let decoded = decode_native_tool_result_markers(&content)
                .unwrap()
                .unwrap();
            assert_eq!(decoded[0].output(), &output);
        }
    }

    #[test]
    fn history_inspection_correlates_results_and_protects_latest_signed_turn() {
        let calls = vec![call("call-1", "Paris"), call("call-2", "London")];
        let assistant =
            encode_native_tool_call_markers(&calls, Some(&state("exchange-history", &calls)))
                .unwrap();
        let messages = vec![
            ChatMessage::user("old"),
            ChatMessage::assistant("old response"),
            ChatMessage::user("current"),
            ChatMessage::assistant(assistant),
            ChatMessage::function(
                "lookup",
                encode_native_tool_result_marker(&calls[0], serde_json::json!("first")).unwrap(),
            ),
            ChatMessage::function(
                "lookup",
                encode_native_tool_result_marker(&calls[1], serde_json::json!("second")).unwrap(),
            ),
        ];

        let inspection = inspect_native_history(&messages).unwrap();
        assert_eq!(inspection.protected_suffix_start(), Some(2));
        assert_eq!(inspection.exchanges().len(), 1);
        assert!(inspection.exchanges()[0].is_complete());
        assert!(inspection.is_safe_prefix_len(2));
        assert!(!inspection.is_safe_prefix_len(3));
        assert_eq!(inspection.safe_prefix_len_at_most(4), 2);
        assert_eq!(inspection.safe_prefix_len_at_least(3), None);
    }

    #[test]
    fn history_inspection_reports_missing_and_duplicate_results() {
        let calls = vec![call("call-1", "Paris"), call("call-2", "London")];
        let assistant =
            encode_native_tool_call_markers(&calls, Some(&state("exchange-incomplete", &calls)))
                .unwrap();
        let first_result =
            encode_native_tool_result_marker(&calls[0], serde_json::json!("first")).unwrap();
        let messages = vec![
            ChatMessage::user("current"),
            ChatMessage::assistant(assistant),
            ChatMessage::function("lookup", &first_result),
        ];
        let inspection = inspect_native_history(&messages).unwrap();
        assert_eq!(
            inspection.exchanges()[0].missing_result_ids(),
            vec!["call-2"]
        );

        let duplicate = vec![
            messages[0].clone(),
            messages[1].clone(),
            messages[2].clone(),
            ChatMessage::function("lookup", first_result),
        ];
        assert!(inspect_native_history(&duplicate).is_err());
    }

    #[test]
    fn history_inspection_rejects_result_reordering() {
        let calls = vec![call("call-1", "Paris"), call("call-2", "London")];
        let assistant =
            encode_native_tool_call_markers(&calls, Some(&state("exchange-order", &calls)))
                .unwrap();
        let messages = vec![
            ChatMessage::user("current"),
            ChatMessage::assistant(assistant),
            ChatMessage::function(
                "lookup",
                encode_native_tool_result_marker(&calls[1], serde_json::json!("second")).unwrap(),
            ),
        ];

        let error = inspect_native_history(&messages).unwrap_err();
        assert!(error.to_string().contains("out of declaration order"));
    }

    #[test]
    fn duplicate_exchange_id_is_rejected_but_identical_content_is_allowed() {
        let calls = vec![call("call-1", "Paris")];
        let first =
            encode_native_tool_call_markers(&calls, Some(&state("same-exchange", &calls))).unwrap();
        let duplicate = vec![
            ChatMessage::user("one"),
            ChatMessage::assistant(&first),
            ChatMessage::user("two"),
            ChatMessage::assistant(&first),
        ];
        assert!(inspect_native_history(&duplicate).is_err());

        let second =
            encode_native_tool_call_markers(&calls, Some(&state("different-exchange", &calls)))
                .unwrap();
        let distinct = vec![
            ChatMessage::user("one"),
            ChatMessage::assistant(first),
            ChatMessage::user("two"),
            ChatMessage::assistant(second),
        ];
        assert!(inspect_native_history(&distinct).is_ok());
    }

    #[test]
    fn user_controlled_marker_text_is_not_promoted_to_signed_history() {
        let calls = vec![call("call-1", "Paris")];
        let content =
            encode_native_tool_call_markers(&calls, Some(&state("user-controlled", &calls)))
                .unwrap();
        let messages = vec![ChatMessage::user(content)];

        let inspection = inspect_native_history(&messages).unwrap();
        assert!(inspection.exchanges().is_empty());
        assert!(inspection.protected_suffix_start().is_none());
    }
}

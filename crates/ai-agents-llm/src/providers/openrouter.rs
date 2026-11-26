//! Adapt OpenRouter Chat to the framework while keeping approval and execution in the runtime.
use super::{chat_adapter, sdk_error};
use crate::sdk::{self, openrouter::*};
use ai_agents_core::{
    ChatMessage, LLMChunk, LLMConfig, LLMError, LLMResponse, LLMToolRequest, NativeCallBinding,
    NativeProviderState, NativeProviderTarget, Role, ToolChoice, decode_native_tool_call_markers,
    inspect_native_history, set_response_provider_state,
};
use futures::{Stream, StreamExt};
use serde_json::{Value, json};
use std::time::Duration;

const PROVIDER: &str = "openrouter";
const API: &str = "chat-completions";
pub(super) const REPORTED_MODEL: &str = "openrouter_reported_model";

pub(super) fn official_origin(base: &str) -> bool {
    reqwest::Url::parse(base)
        .is_ok_and(|url| url.origin().ascii_serialization() == "https://openrouter.ai")
}

pub(super) fn legacy_extra_body(
    cfg: &LLMConfig,
    choice: Option<&ToolChoice>,
) -> Result<Option<Value>, LLMError> {
    let body = cfg.extra.get("extra_body").cloned();
    if !matches!(choice, Some(ToolChoice::Required | ToolChoice::Specific(_))) {
        return Ok(body);
    }
    let mut body = body.unwrap_or_else(|| json!({}));
    let fields = body
        .as_object_mut()
        .ok_or_else(|| LLMError::Config("OpenRouter extra_body must be an object".into()))?;
    require_routing_parameters(fields).map_err(sdk_error)?;
    Ok(Some(body))
}

fn invalid(message: &str) -> LLMError {
    LLMError::Serialization(message.into())
}

pub(super) fn request(
    client: &OpenRouterClient,
    model: &str,
    messages: &[ChatMessage],
    cfg: &LLMConfig,
    tools: Option<&LLMToolRequest>,
) -> Result<Option<Request>, LLMError> {
    if cfg
        .extra
        .get("protocol")
        .is_some_and(|protocol| protocol != "chat-completions")
    {
        return Err(LLMError::Config(
            "OpenRouter requires Chat Completions".into(),
        ));
    }
    let legacy = ["resilient", "normalize_response"].into_iter().any(|key| {
        let value = cfg.extra.get(key).and_then(Value::as_bool);
        if key == "resilient" {
            value == Some(true)
        } else {
            value.is_some()
        }
    });
    if let Some(effort) = cfg
        .reasoning_effort
        .as_deref()
        .or_else(|| cfg.extra.get("reasoning_effort").and_then(Value::as_str))
        && !matches!(
            effort.to_ascii_lowercase().as_str(),
            "low" | "medium" | "high"
        )
    {
        return Err(LLMError::Config(
            "legacy OpenRouter reasoning_effort must be low, medium or high".into(),
        ));
    }
    let mut chat = sdk::ChatRequest::new(model, vec![]);
    chat.temperature = cfg.temperature;
    chat.max_completion_tokens = cfg.max_tokens;
    chat.top_k = cfg.top_k;
    chat.top_p = cfg.top_p;
    chat.timeout = cfg
        .timeout_seconds
        .or_else(|| cfg.extra.get("timeout_seconds").and_then(Value::as_u64))
        .map(Duration::from_secs);
    chat_adapter::request_tools(&mut chat, tools)?;
    let mut request = Request::new(chat);
    chat_adapter::request_options(cfg, |field, key, value| {
        if key == "parallel_tool_calls" {
            let parallel = value
                .as_bool()
                .ok_or_else(|| LLMError::Config(format!("{field}.{key} must be a boolean")))?;
            if request.parallel_tool_calls.is_some() {
                return Err(LLMError::Config(
                    "declare OpenRouter parallel_tool_calls only once".into(),
                ));
            }
            request.parallel_tool_calls = Some(parallel);
            return Ok(());
        }
        chat_adapter::apply_option(&mut request.chat, field, key, value, None)
    })?;
    let history = inspect_native_history(messages)?;
    if history.has_incomplete_unsigned()
        || history
            .exchanges()
            .iter()
            .any(|exchange| !exchange.is_complete())
    {
        return Err(invalid("incomplete OpenRouter tool history"));
    }
    if legacy {
        if cfg.extra.contains_key("provider_options") || cfg.extra.contains_key("protocol") {
            return Err(LLMError::Config(
                "OpenRouter SDK options conflict with legacy transport settings".into(),
            ));
        }
        if history
            .exchanges()
            .iter()
            .any(|exchange| exchange.state().provider() == PROVIDER)
        {
            return Err(invalid("legacy OpenRouter cannot replay SDK tool history"));
        }
        client.body(&request, false).map_err(sdk_error)?;
        return Ok(None);
    }
    let (system, _) = super::unified::extract_system_and_messages(messages);
    if let Some(system) = system.filter(|s| !s.is_empty()) {
        request.chat.messages.push(sdk::Message::System(system));
    }
    for message in messages.iter().filter(|m| m.role != Role::System) {
        if message.role == Role::Assistant
            && let Some(batch) = decode_native_tool_call_markers(&message.content)?
            && let Some(state) = batch.provider_state()
        {
            if state.provider() != PROVIDER
                || state.api() != API
                || state.target().model() != model
                || state.target().endpoint() != client.endpoint()
            {
                return Err(invalid("incompatible OpenRouter history target"));
            }
            let turn: AssistantTurn = serde_json::from_value(state.model_content().clone())
                .map_err(|_| invalid("invalid saved OpenRouter assistant turn"))?;
            let calls = turn.calls().map_err(sdk_error)?;
            if calls.len() != batch.calls().len()
                || calls
                    .iter()
                    .zip(batch.calls())
                    .enumerate()
                    .any(|(index, (wire, normalized))| {
                        wire.id != normalized.id
                            || wire.name != normalized.name
                            || wire.arguments != normalized.arguments
                            || state
                                .bindings()
                                .get(index)
                                .is_none_or(|binding| binding.part_index() != index)
                    })
            {
                return Err(invalid("OpenRouter replay call binding changed"));
            }
            request.push_assistant(&turn).map_err(sdk_error)?;
        } else {
            chat_adapter::append_message(&mut request.chat.messages, message, true)?;
        }
    }
    Ok(Some(request))
}

pub(super) async fn complete(
    client: &OpenRouterClient,
    request: &Request,
) -> Result<LLMResponse, LLMError> {
    let raw = client.complete(request).await.map_err(sdk_error)?;
    let model = &request.chat.model;
    let state = if raw.tool_calls.is_empty() {
        None
    } else {
        // Do not expose executable calls if their private continuation has no eligible destination.
        let body = client.body(request, false).map_err(sdk_error)?;
        raw.assistant
            .validate_replay(client.endpoint(), model, replay_route(&body).as_ref())
            .map_err(sdk_error)?;
        let bindings = raw
            .tool_calls
            .iter()
            .enumerate()
            .map(|(i, call)| NativeCallBinding::new(&call.id, i))
            .collect::<Result<_, _>>()?;
        let id = raw
            .id
            .clone()
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        Some(NativeProviderState::new(
            id,
            PROVIDER,
            API,
            NativeProviderTarget::new(client.endpoint(), model)?,
            json!(raw.assistant),
            bindings,
        )?)
    };
    let mut response = chat_adapter::response(
        sdk::Completion {
            text: raw.text,
            finish_reason: raw.finish_reason,
            tool_calls: raw.tool_calls,
            usage: raw.usage,
        },
        model,
    )?;
    // Legacy builders can synthesize response.model from the request; only a wire receipt is pricing evidence.
    if let Some(reported_model) = raw.model {
        response
            .metadata
            .insert(REPORTED_MODEL.into(), Value::String(reported_model.clone()));
        response.model = Some(reported_model);
    }
    if let Some(state) = state {
        set_response_provider_state(&mut response, state)?;
    }
    Ok(response)
}

pub(super) async fn stream(
    client: &OpenRouterClient,
    request: &Request,
    priced_model: Option<String>,
) -> Result<Box<dyn Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError> {
    let stream = client.stream(request).await.map_err(sdk_error)?;
    Ok(Box::new(stream.map(move |event| {
        match event.map_err(|error| match error {
            sdk::Error::Transport => invalid("OpenRouter response stream disconnected"),
            other => sdk_error(other),
        })? {
            StreamEvent::Text(text) => Ok(LLMChunk::new(text, false)),
            StreamEvent::Finished(output) if output.tool_calls.is_empty() => {
                // The stream settlement hook sees only final usage. Qualify it while the gateway model is still available.
                let usage = output.usage.filter(|_| {
                    priced_model
                        .as_ref()
                        .is_none_or(|model| output.model.as_ref() == Some(model))
                });
                Ok(LLMChunk::final_chunk(
                    "",
                    chat_adapter::finish(&output.finish_reason),
                    usage.map(chat_adapter::usage),
                ))
            }
            _ => Err(invalid("unexpected OpenRouter tool call in text stream")),
        }
    })))
}

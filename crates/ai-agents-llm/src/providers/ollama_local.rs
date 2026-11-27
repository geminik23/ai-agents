//! Adapts the framework's legacy Local Ollama declarations to the native chat SDK.

use super::{sdk_error, unified::extract_system_and_messages};
use crate::sdk::{decimal, ollama::*};
use ai_agents_core::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMResponse, Role, TokenUsage,
};
use futures::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::time::Duration;

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    #[serde(default)]
    options: Map<String, Value>,
    keep_alive: Option<KeepAlive>,
}

pub(super) fn uses_legacy(config: &LLMConfig) -> Result<bool, LLMError> {
    let resilient = config.extra.get("resilient").and_then(Value::as_bool) == Some(true);
    if resilient && config.extra.contains_key("provider_options") {
        return Err(LLMError::Config(
            "Ollama provider_options requires native chat; remove resilient: true to use it".into(),
        ));
    }
    Ok(resilient)
}

pub(super) fn request(
    model: &str,
    messages: &[ChatMessage],
    config: &LLMConfig,
    legacy_body: Option<&Value>,
) -> Result<LocalChatRequest, LLMError> {
    let mut native = config
        .extra
        .get("provider_options")
        .cloned()
        .map(serde_json::from_value::<Options>)
        .transpose()
        .map_err(|error| LLMError::Config(format!("invalid Ollama provider_options: {error}")))?
        .unwrap_or_default();
    for (name, field, declared) in [
        ("temperature", "temperature", config.temperature.is_some()),
        ("num_predict", "max_tokens", config.max_tokens.is_some()),
        ("top_p", "top_p", config.top_p.is_some()),
        ("top_k", "top_k", config.top_k.is_some()),
        ("num_ctx", "num_ctx", false),
        ("num_gpu", "num_gpu", false),
        ("stop", "stop_sequences", config.stop_sequences.is_some()),
    ] {
        if native.options.contains_key(name)
            && (declared
                || config
                    .extra
                    .get(field)
                    .is_some_and(|value| !value.is_null()))
        {
            return Err(duplicate(field, &format!("options.{name}")));
        }
    }
    if let Some(legacy_body) = legacy_body {
        for name in native.options.keys() {
            if legacy_body["options"]
                .get(name)
                .is_some_and(|value| !value.is_null())
            {
                return Err(duplicate(
                    &format!("extra_body.options.{name}"),
                    &format!("options.{name}"),
                ));
            }
        }
    }
    if native.keep_alive.is_some() {
        if config
            .extra
            .get("keep_alive")
            .is_some_and(|value| !value.is_null())
        {
            return Err(duplicate("keep_alive", "keep_alive"));
        }
        if legacy_body
            .and_then(|body| body.get("keep_alive"))
            .is_some_and(|value| !value.is_null())
        {
            return Err(duplicate("extra_body.keep_alive", "keep_alive"));
        }
    }

    // The upstream request always sent both sampling keys, including nulls.
    let top_p = config.top_p.map(decimal).transpose().map_err(sdk_error)?;
    native
        .options
        .entry("top_p")
        .or_insert_with(|| json!(top_p));
    native
        .options
        .entry("top_k")
        .or_insert_with(|| json!(config.top_k));
    let (system, messages) = extract_system_and_messages(messages);
    let messages = system
        .filter(|text| !text.is_empty())
        .map(|text| Message::text("system", text))
        .into_iter()
        .chain(messages.into_iter().map(|msg| match msg.role {
            Role::Assistant => Message::text("assistant", &msg.content),
            Role::User | Role::System => Message::text("user", &msg.content),
            Role::Tool | Role::Function => Message::text(
                "user",
                format!(
                    "[{} result]: {}",
                    msg.name.as_deref().unwrap_or("tool"),
                    msg.content
                ),
            ),
        }))
        .collect();
    let mut chat = ChatRequest::new(model, messages);
    chat.options = native.options;
    chat.timeout = config
        .timeout_seconds
        .or_else(|| config.extra.get("timeout_seconds").and_then(Value::as_u64))
        .map(Duration::from_secs);
    Ok(LocalChatRequest {
        chat,
        keep_alive: native.keep_alive,
    })
}

fn duplicate(legacy: &str, native: &str) -> LLMError {
    LLMError::Config(format!(
        "Ollama declares both {legacy} and provider_options.{native}; remove {legacy} to use the native option"
    ))
}

fn finish(reason: Option<Finish>) -> FinishReason {
    // A verified terminal record without a reason retains the legacy framework text completion.
    match reason {
        None | Some(Finish::Stop) => FinishReason::Stop,
        Some(Finish::Length) => FinishReason::Length,
        Some(Finish::ToolCall) => FinishReason::ToolCall,
    }
}

fn usage(output: &LocalChatOutput) -> Option<TokenUsage> {
    output.usage.map(|usage| TokenUsage {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
    })
}

pub(super) async fn complete(
    client: &OllamaLocalClient,
    request: &LocalChatRequest,
) -> Result<LLMResponse, LLMError> {
    let output = client.complete(request).await.map_err(sdk_error)?;
    let usage = usage(&output);
    let mut response = LLMResponse::new(output.message.content, finish(output.finish));
    response.model = Some(request.chat.model.clone());
    response.usage = usage;
    Ok(response)
}

pub(super) async fn stream(
    client: &OllamaLocalClient,
    request: &LocalChatRequest,
) -> Result<impl Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send + use<>, LLMError> {
    Ok(client
        .stream(request)
        .await
        .map_err(sdk_error)?
        .map(|event| {
            Ok(match event.map_err(sdk_error)? {
                LocalChatEvent::Text(text) => LLMChunk::new(text, false),
                LocalChatEvent::Finished(output) => {
                    LLMChunk::final_chunk("", finish(output.finish), usage(&output))
                }
            })
        }))
}

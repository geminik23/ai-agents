//! Connects native Ollama turns to managed framework requests and exact tool continuation.

use super::sdk_error;
use crate::sdk::{Function, decimal, ollama::*};
use ai_agents_core::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMResponse, LLMToolRequest,
    NativeCallBinding, NativeProviderState, NativeProviderTarget, Role, TokenUsage, ToolCall,
    ToolChoice, decode_native_tool_call_markers, decode_native_tool_result_markers,
    inspect_native_history, native_readable_projection, set_response_provider_state,
};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::time::Duration;

pub(super) const PROVIDER: &str = "ollama-cloud";
const API: &str = "ollama-chat";

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    think: Option<Think>,
    #[serde(default)]
    options: Map<String, Value>,
}

fn configuration(message: &str) -> LLMError {
    LLMError::Config(message.into())
}

fn invalid(message: &str) -> LLMError {
    LLMError::Serialization(message.into())
}

pub(super) fn official_origin(base: &str) -> bool {
    reqwest::Url::parse(base).is_ok_and(|url| url.origin().ascii_serialization() == CLOUD_ORIGIN)
}

pub(super) fn connect(base: Option<&str>, key: String) -> Result<OllamaCloudClient, LLMError> {
    let client = match base {
        Some(base) => OllamaCloudClient::at_endpoint(base, key),
        None => OllamaCloudClient::new(key),
    };
    client.map_err(sdk_error)
}

fn options(config: &LLMConfig) -> Result<Options, LLMError> {
    if config
        .extra
        .keys()
        .any(|key| !matches!(key.as_str(), "provider_options"))
        || config.frequency_penalty.is_some()
        || config.presence_penalty.is_some()
        || config.reasoning_budget_tokens.is_some()
    {
        return Err(configuration("unsupported Ollama Cloud request setting"));
    }
    let mut options = config
        .extra
        .get("provider_options")
        .cloned()
        .map(serde_json::from_value::<Options>)
        .transpose()
        .map_err(|_| configuration("invalid Ollama Cloud provider_options"))?
        .unwrap_or_default();
    let think = match (&options.think, config.reasoning, &config.reasoning_effort) {
        (None, Some(enabled), None) => Some(Think::Enabled(enabled)),
        (None, None, Some(level)) => Some(Think::Level(level.clone())),
        (_, None, None) => options.think.take(),
        _ => return Err(configuration("declare Ollama Cloud think once")),
    };
    options.think = think;
    for (name, value) in [
        (
            "temperature",
            config
                .temperature
                .map(decimal)
                .transpose()
                .map_err(sdk_error)?,
        ),
        (
            "top_p",
            config.top_p.map(decimal).transpose().map_err(sdk_error)?,
        ),
        ("top_k", config.top_k.map(|value| json!(value))),
        ("num_predict", config.max_tokens.map(|value| json!(value))),
        (
            "stop",
            config.stop_sequences.as_ref().map(|value| json!(value)),
        ),
    ] {
        if let Some(value) = value
            && options.options.insert(name.into(), value).is_some()
        {
            return Err(configuration(
                "declare each Ollama Cloud generation option once",
            ));
        }
    }
    if config.max_tokens == Some(0) {
        return Err(configuration("Ollama Cloud max_tokens must be positive"));
    }
    Ok(options)
}

fn replay_message(
    client: &OllamaCloudClient,
    model: &str,
    state: &NativeProviderState,
    call: &ToolCall,
) -> Result<Message, LLMError> {
    if state.provider() != PROVIDER
        || state.api() != API
        || state.target().endpoint() != client.endpoint()
        || state.target().model() != model
        || state.bindings() != [NativeCallBinding::new(&call.id, 0)?]
    {
        return Err(invalid("incompatible Ollama tool replay binding"));
    }
    let message: Message = serde_json::from_value(state.model_content().clone())
        .map_err(|_| invalid("invalid Ollama assistant replay message"))?;
    let [wire] = message.tool_calls.as_slice() else {
        return Err(invalid("invalid Ollama assistant replay call"));
    };
    if message.role != "assistant"
        || wire.function.name != call.name
        || Value::Object(wire.function.arguments.clone()) != call.arguments
    {
        return Err(invalid("Ollama assistant replay call changed"));
    }
    Ok(message)
}

fn history(
    client: &OllamaCloudClient,
    model: &str,
    input: &[ChatMessage],
) -> Result<Vec<Message>, LLMError> {
    let inspected = inspect_native_history(input)?;
    if inspected.has_incomplete_unsigned()
        || inspected
            .exchanges()
            .iter()
            .any(|exchange| !exchange.is_complete())
    {
        return Err(invalid("incomplete Ollama tool history"));
    }
    let mut messages = Vec::with_capacity(input.len());
    let mut pending = None;
    for message in input {
        let (role, text) = match message.role {
            Role::System => ("system", native_readable_projection(&message.content)?),
            Role::User => ("user", native_readable_projection(&message.content)?),
            Role::Assistant => {
                if let Some(batch) = decode_native_tool_call_markers(&message.content)? {
                    let (calls, state) = batch.into_parts();
                    let [call] = calls.as_slice() else {
                        return Err(invalid("Ollama replay requires a single tool call"));
                    };
                    let state = state
                        .as_ref()
                        .ok_or_else(|| invalid("missing Ollama replay state"))?;
                    if pending.is_some() {
                        return Err(invalid("missing Ollama replay result"));
                    }
                    let replay = replay_message(client, model, state, call)?;
                    pending = Some((
                        call.id.clone(),
                        call.name.clone(),
                        replay.tool_calls[0].id.clone(),
                    ));
                    messages.push(replay);
                    continue;
                }
                ("assistant", native_readable_projection(&message.content)?)
            }
            Role::Tool | Role::Function => {
                let results = decode_native_tool_result_markers(&message.content)?
                    .ok_or_else(|| invalid("missing Ollama tool result marker"))?;
                let ([result], Some((id, name, wire_id))) = (results.as_slice(), pending.take())
                else {
                    return Err(invalid("Ollama result has no unique replay call"));
                };
                if result.id() != id || result.tool() != name {
                    return Err(invalid("Ollama tool result binding changed"));
                }
                let content = result
                    .output()
                    .as_str()
                    .map_or_else(|| result.output().to_string(), str::to_owned);
                let mut message = Message::text("tool", content);
                message.tool_name = Some(name);
                message.tool_call_id = wire_id;
                messages.push(message);
                continue;
            }
        };
        messages.push(Message::text(role, text));
    }
    if pending.is_some() {
        return Err(invalid("missing Ollama tool replay result"));
    }
    Ok(messages)
}

fn request(
    client: &OllamaCloudClient,
    model: &str,
    messages: &[ChatMessage],
    config: &LLMConfig,
    tools: Option<&LLMToolRequest>,
) -> Result<ChatRequest, LLMError> {
    let options = options(config)?;
    let mut request = ChatRequest::new(model, history(client, model, messages)?);
    request.options = options.options;
    request.think = options.think;
    request.timeout = config.timeout_seconds.map(Duration::from_secs);
    if let Some(tools) = tools {
        match tools.choice {
            ToolChoice::Auto => {
                request.functions = tools
                    .tools
                    .iter()
                    .map(|tool| Function {
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        parameters: tool.input_schema.clone(),
                    })
                    .collect();
            }
            ToolChoice::None => {}
            _ => {
                return Err(configuration(
                    "Ollama Cloud supports only auto or none tool choice",
                ));
            }
        }
    }
    Ok(request)
}

fn finish(finish: Finish) -> FinishReason {
    match finish {
        Finish::Stop => FinishReason::Stop,
        Finish::Length => FinishReason::Length,
        Finish::ToolCall => FinishReason::ToolCall,
    }
}

fn usage(usage: Usage) -> TokenUsage {
    TokenUsage {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
    }
}

pub(super) async fn complete(
    client: &OllamaCloudClient,
    model: &str,
    messages: &[ChatMessage],
    config: &LLMConfig,
    tools: Option<&LLMToolRequest>,
) -> Result<LLMResponse, LLMError> {
    let output = client
        .complete(&request(client, model, messages, config, tools)?)
        .await
        .map_err(sdk_error)?;
    let mut reply = LLMResponse::new(output.message.content.clone(), finish(output.finish));
    reply.model = Some(model.into());
    reply.usage = output.usage.map(usage);
    if let [wire] = output.message.tool_calls.as_slice() {
        // A framework occurrence ID stays distinct from the optional provider ID on replay.
        let id = uuid::Uuid::new_v4().to_string();
        let call = ToolCall {
            id: id.clone(),
            name: wire.function.name.clone(),
            arguments: Value::Object(wire.function.arguments.clone()),
        };
        let state = NativeProviderState::new(
            &id,
            PROVIDER,
            API,
            NativeProviderTarget::new(client.endpoint(), model)?,
            serde_json::to_value(&output.message)?,
            vec![NativeCallBinding::new(&id, 0)?],
        )?;
        reply.set_tool_calls(vec![call])?;
        set_response_provider_state(&mut reply, state)?;
    }
    Ok(reply)
}

type FrameworkStream = Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>;

pub(super) async fn stream(
    client: &OllamaCloudClient,
    model: &str,
    messages: &[ChatMessage],
    config: &LLMConfig,
) -> Result<FrameworkStream, LLMError> {
    let stream = client
        .stream(&request(client, model, messages, config, None)?)
        .await
        .map_err(sdk_error)?;
    Ok(Box::new(stream.map(|event| {
        Ok(match event.map_err(sdk_error)? {
            ChatEvent::Text(text) => LLMChunk::new(text, false),
            ChatEvent::Finished(output) => {
                LLMChunk::final_chunk("", finish(output.finish), output.usage.map(usage))
            }
        })
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_cloud_preserves_options_and_rejects_duplicate_declarations() {
        let mut config = LLMConfig {
            temperature: None,
            max_tokens: Some(90),
            top_p: None,
            ..Default::default()
        };
        config.extra.insert(
            "provider_options".into(),
            json!({"think":"chosen-level","options":{"seed":42}}),
        );
        let resolved = options(&config).unwrap();
        assert_eq!(resolved.think, Some(Think::Level("chosen-level".into())));
        assert_eq!(
            Value::Object(resolved.options),
            json!({"seed":42,"num_predict":90})
        );
        config.reasoning = Some(false);
        assert!(matches!(options(&config), Err(LLMError::Config(_))));
        config.reasoning = None;
        config.extra.insert(
            "provider_options".into(),
            json!({"options":{"num_predict":91}}),
        );
        assert!(matches!(options(&config), Err(LLMError::Config(_))));
    }
}

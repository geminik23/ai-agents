//! Maps Messages blocks into the existing managed tool and replay contracts.
use super::{ProviderType, sdk_error};
use crate::sdk::{Function, FunctionChoice, messages::*};
use ai_agents_core::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMResponse, LLMToolRequest,
    NativeCallBinding, NativeProviderState, NativeProviderTarget, Role, TokenUsage, ToolCall,
    ToolChoice, decode_native_tool_call_markers, decode_native_tool_result_markers,
    inspect_native_history, native_readable_projection, set_response_provider_state,
};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use std::time::Duration;

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    thinking: Option<Thinking>,
    output_config: Option<OutputConfig>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputConfig {
    effort: Option<String>,
}

fn configuration(message: &str) -> LLMError {
    LLMError::Config(message.into())
}

fn invalid(message: &str) -> LLMError {
    LLMError::Serialization(message.into())
}

pub(super) fn connect(
    provider: ProviderType,
    base: Option<&str>,
    key: String,
    config: &LLMConfig,
) -> Result<MessagesClient, LLMError> {
    let official = provider == ProviderType::Anthropic;
    let base = base
        .or(official.then_some(OFFICIAL_BASE))
        .ok_or_else(|| configuration("anthropic-compatible requires a base_url"))?;
    let version = match config.extra.get("anthropic_version") {
        None => API_VERSION,
        Some(value) => value
            .as_str()
            .ok_or_else(|| configuration("expected anthropic_version string"))?,
    };
    let mode = config.extra.get("messages_auth");
    if official && mode.is_some() {
        return Err(configuration("the official Messages preset uses x-api-key"));
    }
    let auth = if official {
        MessagesAuth::ApiKey(key)
    } else {
        match mode {
            None if key.is_empty() => MessagesAuth::None,
            None => MessagesAuth::Bearer(key),
            Some(mode) if mode == "bearer" => MessagesAuth::Bearer(key),
            Some(mode) if mode == "api-key" => MessagesAuth::ApiKey(key),
            _ => return Err(configuration("unknown Messages authentication mode")),
        }
    };
    MessagesClient::at_endpoint(base, auth, version, official).map_err(sdk_error)
}

fn options(config: &LLMConfig) -> Result<Options, LLMError> {
    let mut options = match config.extra.get("provider_options") {
        Some(value) => serde_json::from_value::<Options>(value.clone())
            .map_err(|_| configuration("invalid or unsupported Messages provider_options"))?,
        None => Options::default(),
    };
    if options.thinking.is_some()
        && (config.reasoning.is_some() || config.reasoning_budget_tokens.is_some())
    {
        return Err(configuration("declare Messages thinking once"));
    }
    options.thinking = match (
        options.thinking,
        config.reasoning,
        config.reasoning_budget_tokens,
    ) {
        (Some(thinking), _, _) => Some(thinking),
        (None, Some(false), None) => Some(Thinking::Disabled),
        (None, Some(false), Some(_)) => {
            return Err(configuration("disabled thinking cannot have a budget"));
        }
        (None, _, Some(budget_tokens)) => Some(Thinking::Enabled { budget_tokens }),
        (None, Some(true), None) => Some(Thinking::Adaptive),
        _ => None,
    };
    if config.reasoning_effort.is_some()
        && options
            .output_config
            .as_ref()
            .is_some_and(|o| o.effort.is_some())
    {
        return Err(configuration("declare Messages effort once"));
    }
    Ok(options)
}

fn choice(choice: &ToolChoice) -> Option<FunctionChoice> {
    Some(match choice {
        ToolChoice::Auto => FunctionChoice::Auto,
        ToolChoice::None => FunctionChoice::None,
        ToolChoice::Required => FunctionChoice::Required,
        ToolChoice::Specific(name) => FunctionChoice::Named(name.clone()),
        _ => return None,
    })
}

pub(super) fn supports_choice(
    client: &MessagesClient,
    model: &str,
    config: &LLMConfig,
    selection: &ToolChoice,
) -> bool {
    choice(selection)
        .zip(options(config).ok())
        .is_some_and(|(selection, options)| {
            allows_choice(
                client.official(),
                model,
                options.thinking.as_ref(),
                &selection,
            )
        })
}

// Match the original block positions and the complete validated inputs against normalized calls.
fn bindings(blocks: &[Value], calls: &[ToolCall]) -> Result<Vec<NativeCallBinding>, LLMError> {
    let mut positions = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        if let Some(wire) = validate_block(block).map_err(sdk_error)? {
            let call = calls
                .get(positions.len())
                .ok_or_else(|| invalid("unexpected replay tool block"))?;
            if wire.id != call.id || wire.name != call.name || wire.arguments != call.arguments {
                return Err(invalid("Messages replay tool binding changed"));
            }
            positions.push(NativeCallBinding::new(&call.id, index)?);
        }
    }
    if positions.len() != calls.len() {
        return Err(invalid("missing Messages replay tool block"));
    }
    Ok(positions)
}

fn replay_provider(client: &MessagesClient) -> &'static str {
    if client.official() {
        "anthropic"
    } else {
        "anthropic-compatible"
    }
}

fn replay_blocks(
    client: &MessagesClient,
    model: &str,
    state: &NativeProviderState,
    calls: &[ToolCall],
) -> Result<Vec<Value>, LLMError> {
    if state.provider() != replay_provider(client)
        || state.api() != "messages"
        || state.target().endpoint() != client.endpoint()
        || state.target().model() != model
        || state.model_content()["version"] != client.version()
    {
        return Err(invalid("incompatible Messages replay target"));
    }
    let blocks = state.model_content()["blocks"]
        .as_array()
        .ok_or_else(|| invalid("missing Messages replay blocks"))?;
    if bindings(blocks, calls)? != state.bindings() {
        return Err(invalid("Messages replay block positions changed"));
    }
    Ok(blocks.clone())
}

fn text_block(message: &ChatMessage) -> Result<Value, LLMError> {
    Ok(json!({"type":"text","text":native_readable_projection(&message.content)?}))
}

fn history(
    client: &MessagesClient,
    model: &str,
    input: &[ChatMessage],
) -> Result<(Option<String>, Vec<Value>), LLMError> {
    let native = inspect_native_history(input)?;
    if native.has_incomplete_unsigned()
        || native
            .exchanges()
            .iter()
            .any(|exchange| !exchange.is_complete())
    {
        return Err(invalid("incomplete Messages tool history"));
    }
    let mut system = Vec::new();
    let mut messages: Vec<Value> = Vec::new();
    for message in input {
        let (role, blocks) = match message.role {
            Role::System => {
                system.push(native_readable_projection(&message.content)?);
                continue;
            }
            Role::Assistant => {
                let blocks = if let Some(batch) = decode_native_tool_call_markers(&message.content)?
                {
                    let (calls, state) = batch.into_parts();
                    if let Some(state) = state {
                        replay_blocks(client, model, &state, &calls)?
                    } else {
                        calls.into_iter().map(|call| json!({
                            "type":"tool_use","id":call.id,"name":call.name,"input":call.arguments,
                        })).collect()
                    }
                } else {
                    vec![text_block(message)?]
                };
                ("assistant", blocks)
            }
            Role::User => ("user", vec![text_block(message)?]),
            Role::Tool | Role::Function => {
                let blocks = if let Some(results) =
                    decode_native_tool_result_markers(&message.content)?
                {
                    results.into_iter().map(|result| {
                        let output = result.output();
                        let content = output.as_str().map_or_else(|| output.to_string(), str::to_owned);
                        json!({"type":"tool_result","tool_use_id":result.id(),"content":content})
                    }).collect()
                } else {
                    vec![text_block(message)?]
                };
                ("user", blocks)
            }
        };
        if let Some(previous) = messages.last_mut().filter(|value| value["role"] == role) {
            previous["content"]
                .as_array_mut()
                .expect("constructed Messages array")
                .extend(blocks);
        } else {
            messages.push(json!({"role":role,"content":blocks}));
        }
    }
    Ok(((!system.is_empty()).then(|| system.join("\n\n")), messages))
}

fn request(
    client: &MessagesClient,
    model: &str,
    messages: &[ChatMessage],
    config: &LLMConfig,
    tools: Option<&LLMToolRequest>,
) -> Result<MessagesRequest, LLMError> {
    if config.extra.keys().any(|key| {
        !matches!(
            key.as_str(),
            "protocol" | "provider_options" | "anthropic_version" | "messages_auth"
        )
    }) || config
        .extra
        .get("protocol")
        .is_some_and(|value| value != "messages")
        || config.frequency_penalty.is_some()
        || config.presence_penalty.is_some()
        || config.stop_sequences.is_some()
    {
        return Err(configuration("unsupported Messages request setting"));
    }
    let limit = config
        .max_tokens
        .filter(|max| *max > 0)
        .ok_or_else(|| configuration("Messages requires explicit positive max_tokens"))?;
    let options = options(config)?;
    let (system, messages) = history(client, model, messages)?;
    let mut request = MessagesRequest::new(model, messages, limit);
    request.system = system;
    request.temperature = config.temperature;
    request.top_p = config.top_p;
    request.top_k = config.top_k;
    request.timeout = config.timeout_seconds.map(Duration::from_secs);
    request.effort = config
        .reasoning_effort
        .clone()
        .or(options.output_config.and_then(|config| config.effort));
    request.thinking = options.thinking;
    if let Some(tools) = tools {
        request.functions = tools
            .tools
            .iter()
            .map(|tool| Function {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.input_schema.clone(),
            })
            .collect();
        request.tool_choice = Some(
            choice(&tools.choice)
                .ok_or_else(|| configuration("unsupported Messages tool choice"))?,
        );
    }
    Ok(request)
}

fn usage(usage: MessagesUsage) -> Option<TokenUsage> {
    let (prompt_tokens, completion_tokens, total_tokens) = usage.totals()?;
    Some(TokenUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
    })
}

pub(super) async fn complete(
    client: &MessagesClient,
    model: &str,
    messages: &[ChatMessage],
    config: &LLMConfig,
    tools: Option<&LLMToolRequest>,
) -> Result<LLMResponse, LLMError> {
    let response = client
        .complete(&request(client, model, messages, config, tools)?)
        .await
        .map_err(sdk_error)?;
    let mut result = LLMResponse::new(response.text, FinishReason::Stop);
    result.model = Some(model.to_owned());
    result.usage = response.usage.and_then(usage);
    if !response.tool_calls.is_empty() {
        let calls = response
            .tool_calls
            .into_iter()
            .map(|call| ToolCall {
                id: call.id,
                name: call.name,
                arguments: call.arguments,
            })
            .collect::<Vec<_>>();
        let bindings = bindings(&response.blocks, &calls)?;
        let state = NativeProviderState::new(
            response.id,
            replay_provider(client),
            "messages",
            NativeProviderTarget::new(client.endpoint(), model)?,
            json!({"blocks":response.blocks,"version":client.version()}),
            bindings,
        )?;
        state.validate_for_calls(&calls)?;
        result.finish_reason = FinishReason::ToolCall;
        result.set_tool_calls(calls)?;
        set_response_provider_state(&mut result, state)?;
    }
    Ok(result)
}

type FrameworkStream = Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Send + Unpin>;

pub(super) async fn stream(
    client: &MessagesClient,
    model: &str,
    messages: &[ChatMessage],
    config: &LLMConfig,
) -> Result<FrameworkStream, LLMError> {
    let request = request(client, model, messages, config, None)?;
    let stream = client
        .stream(&request)
        .await
        .map_err(sdk_error)?
        .map(|event| {
            Ok(match event.map_err(sdk_error)? {
                MessageEvent::Text(delta) => LLMChunk::new(delta, false),
                MessageEvent::Finished(result) => {
                    LLMChunk::final_chunk("", FinishReason::Stop, result.usage.and_then(usage))
                }
            })
        });
    Ok(Box::new(stream))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_agents_core::{encode_native_tool_call_markers, encode_native_tool_result_marker};

    #[test]
    fn messages_forced_choice_depends_on_model_service_and_thinking() {
        let official = MessagesClient::at_endpoint(
            "http://localhost/v1",
            MessagesAuth::ApiKey("fixture".into()),
            API_VERSION,
            true,
        )
        .unwrap();
        let compatible =
            MessagesClient::compatible("http://localhost/v1", MessagesAuth::None, API_VERSION)
                .unwrap();
        let adaptive = LLMConfig::default()
            .with_extra("provider_options", json!({"thinking":{"type":"adaptive"}}));
        let manual = LLMConfig::default().with_extra(
            "provider_options",
            json!({"thinking":{"type":"enabled","budget_tokens":1024}}),
        );
        for (client, model, config, forced) in [
            (&official, "claude-sonnet-4-6", &adaptive, true),
            (&official, "claude-sonnet-4-6", &manual, false),
            (&official, "claude-sonnet-5-5", &adaptive, false),
            (&compatible, "claude-sonnet-5-5", &manual, true),
        ] {
            assert!(supports_choice(client, model, config, &ToolChoice::Auto));
            assert!(supports_choice(client, model, config, &ToolChoice::None));
            assert_eq!(
                supports_choice(client, model, config, &ToolChoice::Required),
                forced
            );
            assert_eq!(
                supports_choice(
                    client,
                    model,
                    config,
                    &ToolChoice::Specific("lookup".into())
                ),
                forced
            );
        }
    }

    #[test]
    fn messages_preserves_private_replay_only_under_the_original_binding() {
        let client =
            MessagesClient::compatible("http://localhost/v1", MessagesAuth::None, API_VERSION)
                .unwrap();
        let call = ToolCall {
            id: "lookup_1".into(),
            name: "lookup".into(),
            arguments: json!({"key":7}),
        };
        for (provider, api, endpoint, model, version, argument, valid) in [
            (
                "anthropic-compatible",
                "messages",
                client.endpoint(),
                "test",
                API_VERSION,
                7,
                true,
            ),
            (
                "anthropic",
                "messages",
                client.endpoint(),
                "test",
                API_VERSION,
                7,
                false,
            ),
            (
                "anthropic-compatible",
                "responses",
                client.endpoint(),
                "test",
                API_VERSION,
                7,
                false,
            ),
            (
                "anthropic-compatible",
                "messages",
                "https://other.invalid/v1/messages",
                "test",
                API_VERSION,
                7,
                false,
            ),
            (
                "anthropic-compatible",
                "messages",
                client.endpoint(),
                "other",
                API_VERSION,
                7,
                false,
            ),
            (
                "anthropic-compatible",
                "messages",
                client.endpoint(),
                "test",
                "other",
                7,
                false,
            ),
            (
                "anthropic-compatible",
                "messages",
                client.endpoint(),
                "test",
                API_VERSION,
                8,
                false,
            ),
        ] {
            let blocks = json!([
                {"type":"thinking","thinking":"encrypted owner context","signature":"private-signature"},
                {"type":"tool_use","id":"lookup_1","name":"lookup","input":{"key":argument}},
            ]);
            let state = NativeProviderState::new(
                "msg_1",
                provider,
                api,
                NativeProviderTarget::new(endpoint, model).unwrap(),
                json!({"version":version,"blocks":blocks}),
                vec![NativeCallBinding::new(&call.id, 1).unwrap()],
            )
            .unwrap();
            let messages = [
                ChatMessage::user("lookup"),
                ChatMessage::assistant(
                    encode_native_tool_call_markers(std::slice::from_ref(&call), Some(&state))
                        .unwrap(),
                ),
                ChatMessage::function(
                    "lookup",
                    encode_native_tool_result_marker(&call, json!("found")).unwrap(),
                ),
            ];
            let result = history(&client, "test", &messages);
            if valid {
                let (_, messages) = result.unwrap();
                assert_eq!(messages[1]["content"], blocks);
                assert_eq!(
                    messages[2]["content"],
                    json!([{"type":"tool_result","tool_use_id":"lookup_1","content":"found"}])
                );
            } else {
                assert!(
                    !result
                        .unwrap_err()
                        .to_string()
                        .contains("private-signature")
                );
            }
        }
        for config in [
            LLMConfig {
                max_tokens: None,
                ..Default::default()
            },
            LLMConfig::default().with_extra("provider_options", json!({"max_tokens":100})),
            LLMConfig::default().with_extra("provider_options", json!({"api_key":"other"})),
        ] {
            assert!(request(&client, "test", &[], &config, None).is_err());
        }
    }
}

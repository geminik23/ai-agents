//! Projection between managed framework messages and stateless Responses items.
use super::sdk_error as error;
use crate::sdk::{self, responses::*};
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

fn invalid(message: &str) -> LLMError {
    LLMError::Serialization(message.into())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    reasoning: Option<Reasoning>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Reasoning {
    mode: Option<String>,
    effort: Option<String>,
}

// Bind the exact provider output to the normalized calls before approval and again on replay.
fn replay_bindings(
    output: &[Value],
    calls: &[ToolCall],
) -> Result<Vec<NativeCallBinding>, LLMError> {
    let mut bindings = Vec::new();
    for (index, item) in output.iter().enumerate() {
        match item["type"].as_str() {
            Some("function_call") => {
                let Some(call) = calls.get(bindings.len()) else {
                    return Err(invalid("unexpected replay call"));
                };
                let wire = function_call(item).map_err(error)?;
                if wire.id != call.id || wire.name != call.name || wire.arguments != call.arguments
                {
                    return Err(invalid("Responses replay call binding changed"));
                }
                bindings.push(NativeCallBinding::new(&call.id, index)?);
            }
            Some("reasoning")
                if item["encrypted_content"]
                    .as_str()
                    .is_some_and(|s| !s.is_empty()) => {}
            Some("message") if item["role"] == "assistant" => {}
            _ => return Err(invalid("invalid or unavailable Responses replay item")),
        }
    }
    if bindings.len() != calls.len() {
        return Err(invalid("missing Responses replay call"));
    }
    Ok(bindings)
}

pub(super) fn request(
    client: &ResponsesClient,
    model: &str,
    messages: &[ChatMessage],
    cfg: &LLMConfig,
    tools: Option<&LLMToolRequest>,
) -> Result<ResponsesRequest, LLMError> {
    let mut r = ResponsesRequest::new(model, vec![]);
    r.reasoning_effort = cfg.reasoning_effort.clone();
    r.max_output_tokens = cfg.max_tokens;
    r.temperature = cfg.temperature;
    r.top_p = cfg.top_p;
    r.timeout = cfg.timeout_seconds.map(Duration::from_secs);
    if cfg.reasoning.is_some()
        || cfg.reasoning_budget_tokens.is_some()
        || cfg.top_k.is_some()
        || cfg.presence_penalty.is_some()
        || cfg.frequency_penalty.is_some()
        || cfg.stop_sequences.is_some()
    {
        return Err(LLMError::Config("unsupported Responses setting".into()));
    }
    for (k, v) in &cfg.extra {
        match k.as_str() {
            "protocol" if v == "responses" => {}
            "declared_pricing" => {}
            "provider_options" => {
                let options: Options = serde_json::from_value(v.clone()).map_err(|_| {
                    LLMError::Config("invalid or unsupported Responses provider_options".into())
                })?;
                if let Some(reasoning) = options.reasoning {
                    if r.reasoning_effort.is_some() && reasoning.effort.is_some() {
                        return Err(LLMError::Config("duplicate reasoning.effort".into()));
                    }
                    r.reasoning_effort = r.reasoning_effort.or(reasoning.effort);
                    r.reasoning_mode = reasoning.mode;
                }
            }
            _ => {
                return Err(LLMError::Config(format!(
                    "unsupported Responses option: {k}"
                )));
            }
        }
    }
    if let Some(t) = tools {
        r.functions = t
            .tools
            .iter()
            .map(|f| ResponseFunction {
                // The executor owns schema validation; keep optional properties optional on the wire.
                strict: Some(false),
                name: f.name.clone(),
                description: f.description.clone(),
                parameters: f.input_schema.clone(),
            })
            .collect();
        r.tool_choice = Some(match &t.choice {
            ToolChoice::Auto => sdk::FunctionChoice::Auto,
            ToolChoice::None => sdk::FunctionChoice::None,
            ToolChoice::Required => sdk::FunctionChoice::Required,
            ToolChoice::Specific(n) => sdk::FunctionChoice::Named(n.clone()),
            _ => {
                return Err(LLMError::Config(
                    "unsupported Responses tool selection".into(),
                ));
            }
        });
    }
    append_history(&mut r, client, messages)?;
    Ok(r)
}

fn append_history(
    r: &mut ResponsesRequest,
    client: &ResponsesClient,
    messages: &[ChatMessage],
) -> Result<(), LLMError> {
    let history = inspect_native_history(messages)?;
    if history.has_incomplete_unsigned() || history.exchanges().iter().any(|e| !e.is_complete()) {
        return Err(invalid("incomplete native history"));
    }
    let mut instructions = Vec::new();
    for msg in messages {
        match msg.role {
            Role::System => instructions.push(msg.content.as_str()),
            Role::Assistant => {
                if let Some(batch) = decode_native_tool_call_markers(&msg.content)? {
                    let (calls, state) = batch.into_parts();
                    if let Some(state) = state {
                        if state.provider() != "openai"
                            || state.api() != "responses"
                            || state.target().endpoint() != client.endpoint()
                            || state.target().model() != r.model
                        {
                            return Err(invalid("incompatible Responses replay target"));
                        }
                        let output = state.model_content()["output"]
                            .as_array()
                            .ok_or_else(|| invalid("missing Responses replay items"))?;
                        if replay_bindings(output, &calls)? != state.bindings() {
                            return Err(invalid("Responses replay bindings changed"));
                        }
                        r.input.extend(output.iter().cloned());
                    } else {
                        for c in calls {
                            r.input.push(json!({"type":"function_call","call_id":c.id,"name":c.name,"arguments":c.arguments.to_string()}));
                        }
                    }
                } else {
                    r.input.push(json!({"role":"assistant","content":native_readable_projection(&msg.content)?}));
                }
            }
            Role::Function | Role::Tool => {
                if let Some(results) = decode_native_tool_result_markers(&msg.content)? {
                    for result in results {
                        r.input.push(json!({"type":"function_call_output","call_id":result.id(),"output":result.output().to_string()}));
                    }
                } else {
                    r.input.push(
                        json!({"role":"user","content":native_readable_projection(&msg.content)?}),
                    );
                }
            }
            Role::User => r.input.push(json!({"role":"user","content":msg.content})),
        }
    }
    if !instructions.is_empty() {
        r.instructions = Some(instructions.join("\n\n"));
    }
    Ok(())
}

fn usage(u: ResponseUsage) -> TokenUsage {
    TokenUsage {
        prompt_tokens: u.input_tokens,
        completion_tokens: u.output_tokens,
        total_tokens: u.total_tokens,
    }
}

pub(super) async fn complete(
    client: &ResponsesClient,
    model: &str,
    messages: &[ChatMessage],
    cfg: &LLMConfig,
    tools: Option<&LLMToolRequest>,
) -> Result<LLMResponse, LLMError> {
    let req = request(client, model, messages, cfg, tools)?;
    let raw = client.complete(&req).await.map_err(error)?;
    let mut response = LLMResponse::new(raw.text, FinishReason::Stop);
    response.model = Some(model.into());
    response.usage = raw.usage.map(usage);
    if !raw.tool_calls.is_empty() {
        let calls = raw
            .tool_calls
            .into_iter()
            .map(|c| ToolCall {
                id: c.id,
                name: c.name,
                arguments: c.arguments,
            })
            .collect::<Vec<_>>();
        let bindings = replay_bindings(&raw.output, &calls)?;
        let state = NativeProviderState::new(
            raw.id,
            "openai",
            "responses",
            NativeProviderTarget::new(client.endpoint(), model)?,
            json!({"output":raw.output}),
            bindings,
        )?;
        state.validate_for_calls(&calls)?;
        response.finish_reason = FinishReason::ToolCall;
        response.set_tool_calls(calls)?;
        set_response_provider_state(&mut response, state)?;
    }
    Ok(response)
}

pub(super) async fn stream(
    client: &ResponsesClient,
    model: &str,
    messages: &[ChatMessage],
    cfg: &LLMConfig,
) -> Result<Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError> {
    let req = request(client, model, messages, cfg, None)?;
    let events = client.stream(&req).await.map_err(error)?.map(|e| {
        Ok(match e.map_err(error)? {
            ResponseEvent::Text(s) => LLMChunk::new(s, false),
            ResponseEvent::Finished(r) => {
                LLMChunk::final_chunk("", FinishReason::Stop, r.usage.map(usage))
            }
        })
    });
    Ok(Box::new(events))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_agents_core::{encode_native_tool_call_markers, encode_native_tool_result_marker};
    #[test]
    fn responses_replay_is_bound_to_the_exact_target_and_call() {
        let client = ResponsesClient::new("http://localhost/v1", None).unwrap();
        let call = ToolCall {
            id: "call".into(),
            name: "lookup".into(),
            arguments: json!({"x":1}),
        };
        let cfg = LLMConfig {
            temperature: None,
            top_p: None,
            ..LLMConfig::default()
        };
        for (api, endpoint, model, args, ok) in [
            ("responses", client.endpoint(), "model", r#"{"x":1}"#, true),
            ("messages", client.endpoint(), "model", r#"{"x":1}"#, false),
            (
                "responses",
                "http://example.invalid/v1/responses",
                "model",
                r#"{"x":1}"#,
                false,
            ),
            ("responses", client.endpoint(), "other", r#"{"x":1}"#, false),
            ("responses", client.endpoint(), "model", r#"{"x":2}"#, false),
        ] {
            let state=NativeProviderState::new("r","openai",api,NativeProviderTarget::new(endpoint,model).unwrap(),json!({"output":[{"type":"reasoning","id":"rs","encrypted_content":"private-fixture"},{"type":"function_call","id":"fc","name":"lookup","arguments":args,"call_id":"call"}]}),vec![NativeCallBinding::new("call",1).unwrap()]).unwrap();
            let history = [
                ChatMessage::user("test"),
                ChatMessage::assistant(
                    encode_native_tool_call_markers(std::slice::from_ref(&call), Some(&state))
                        .unwrap(),
                ),
                ChatMessage::function(
                    "lookup",
                    encode_native_tool_result_marker(&call, json!("done")).unwrap(),
                ),
            ];
            let result = request(&client, "model", &history, &cfg, None);
            assert_eq!(result.is_ok(), ok);
            if let Err(e) = result {
                assert!(!e.to_string().contains("private-fixture"));
            }
        }
        for options in [
            json!({"store":true}),
            json!({"previous_response_id":"other"}),
            json!({"tools":[]}),
            json!({"input":[]}),
            json!({"max_output_tokens":10}),
        ] {
            assert!(
                request(
                    &client,
                    "model",
                    &[],
                    &cfg.clone().with_extra("provider_options", options),
                    None
                )
                .is_err()
            );
        }
    }
}

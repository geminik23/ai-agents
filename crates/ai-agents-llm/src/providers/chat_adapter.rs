//! Conversion between the framework provider contract and the independent Chat client.
use super::unified::ProviderType;
use crate::sdk;
use ai_agents_core::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMResponse, LLMToolRequest, Role,
    TokenUsage, ToolCall, ToolChoice,
};
use ai_agents_core::{
    decode_native_tool_call_markers, decode_native_tool_result_markers, inspect_native_history,
    native_readable_projection,
};
use serde_json::Value;
use std::time::Duration;

use super::sdk_error as error;

pub(super) fn append_message(
    out: &mut Vec<sdk::Message>,
    msg: &ChatMessage,
    native: bool,
) -> Result<(), LLMError> {
    match msg.role {
        Role::System => unreachable!("system instructions are joined by the caller"),
        Role::User => out.push(sdk::Message::User(msg.content.clone())),
        Role::Assistant => {
            if native && let Some(batch) = decode_native_tool_call_markers(&msg.content)? {
                let (calls, _) = batch.into_parts();
                out.push(sdk::Message::Assistant {
                    text: None,
                    calls: calls
                        .into_iter()
                        .map(|c| sdk::ToolCall {
                            id: c.id,
                            name: c.name,
                            arguments: c.arguments,
                        })
                        .collect(),
                });
            } else {
                out.push(sdk::Message::Assistant {
                    text: Some(native_readable_projection(&msg.content)?),
                    calls: Vec::new(),
                });
            }
        }
        Role::Function | Role::Tool => {
            if native && let Some(results) = decode_native_tool_result_markers(&msg.content)? {
                for result in results {
                    out.push(sdk::Message::ToolResult {
                        id: result.id().into(),
                        content: result.output().to_string(),
                    });
                }
            } else {
                let text = native_readable_projection(&msg.content)?;
                out.push(sdk::Message::User(format!(
                    "[{} result]: {text}",
                    msg.name.as_deref().unwrap_or("tool")
                )));
            }
        }
    }
    Ok(())
}

// Distinct leaves may extend the same object. A repeated leaf or a scalar/object conflict
// would make the effective request depend on insertion order.
fn merge_option(
    map: &mut serde_json::Map<String, Value>,
    key: &str,
    value: &Value,
    path: &str,
) -> Result<(), LLMError> {
    match (map.get_mut(key), value) {
        (None, _) => {
            map.insert(key.into(), value.clone());
        }
        (Some(Value::Object(existing)), Value::Object(incoming)) => {
            for (child, v) in incoming {
                merge_option(existing, child, v, &format!("{path}.{child}"))?;
            }
        }
        (Some(_), _) => {
            return Err(LLMError::Config(format!(
                "conflicting provider option: {path}"
            )));
        }
    }
    Ok(())
}

// Only enabled features choose the upstream backend. Tuning a disabled feature has no wire effect.
fn requires_legacy(cfg: &LLMConfig) -> Result<bool, LLMError> {
    let enabled = |name| {
        cfg.extra.get(name).map_or(Ok(false), |value| {
            value
                .as_bool()
                .ok_or_else(|| LLMError::Config(format!("{name} must be a boolean")))
        })
    };
    let reasoning = match cfg.reasoning {
        Some(enabled) => enabled,
        None => enabled("reasoning")?,
    };
    let resilient = enabled("resilient")?;
    let web_search = enabled("openai_enable_web_search")?;
    Ok(reasoning
        || resilient
        || web_search
        || cfg.reasoning_budget_tokens.is_some()
        || cfg.frequency_penalty.is_some()
        || cfg.presence_penalty.is_some()
        || cfg
            .stop_sequences
            .as_ref()
            .is_some_and(|seq| !seq.is_empty())
        || ["reasoning_budget_tokens", "api_version", "deployment_id"]
            .iter()
            .any(|key| cfg.extra.contains_key(*key)))
}

pub(super) fn request_options(
    request: &mut sdk::ChatRequest,
    cfg: &LLMConfig,
) -> Result<(), LLMError> {
    let extra = &cfg.extra;
    for field in ["extra_body", "provider_options"] {
        if let Some(value) = extra.get(field) {
            let options = value
                .as_object()
                .ok_or_else(|| LLMError::Config(format!("{field} must be an object")))?;
            for (k, v) in options {
                if field == "extra_body" && k == "response_format" {
                    // Migrate the legacy wire shape into the owned request field.
                    request.response_format =
                        serde_json::from_value(v.clone()).map_err(|error| {
                            LLMError::Config(format!("invalid extra_body.response_format: {error}"))
                        })?;
                    continue;
                }
                if sdk::reserved(k) {
                    return Err(LLMError::Config(format!("reserved option {field}.{k}")));
                }
                merge_option(&mut request.provider_options, k, v, &format!("{field}.{k}"))?;
            }
        }
    }
    Ok(())
}

pub(super) fn request(
    provider: ProviderType,
    model: &str,
    messages: &[ChatMessage],
    cfg: &LLMConfig,
    tools: Option<&LLMToolRequest>,
) -> Result<Option<sdk::ChatRequest>, LLMError> {
    let mut request = sdk::ChatRequest::new(model, vec![]);
    request.temperature = cfg.temperature;
    request.max_completion_tokens = cfg.max_tokens;
    request.top_p = cfg.top_p;
    request.top_k = cfg.top_k;
    let extra = &cfg.extra;
    if extra
        .get("protocol")
        .is_some_and(|v| v.as_str() != Some("chat-completions"))
    {
        return Err(LLMError::Config(
            "expected protocol chat-completions".into(),
        ));
    }
    if cfg.reasoning_effort.is_some() && extra.contains_key("reasoning_effort") {
        return Err(LLMError::Config("duplicate reasoning_effort".into()));
    }
    request.reasoning_effort = match extra.get("reasoning_effort") {
        Some(v) => Some(
            v.as_str()
                .ok_or_else(|| LLMError::Config("expected a reasoning_effort string".into()))?
                .into(),
        ),
        None => cfg.reasoning_effort.clone(),
    };
    request.timeout = cfg
        .timeout_seconds
        .or_else(|| extra.get("timeout_seconds").and_then(Value::as_u64))
        .map(Duration::from_secs);
    request_options(&mut request, cfg)?;
    if provider == ProviderType::OpenAI {
        let hosted_search = extra
            .get("openai_enable_web_search")
            .and_then(Value::as_bool)
            == Some(true);
        super::openai::validate(&request, tools.is_some() || hosted_search).map_err(error)?;
    }
    if requires_legacy(cfg)? {
        if extra.contains_key("provider_options") || extra.contains_key("protocol") {
            return Err(LLMError::Config(
                "SDK options conflict with legacy transport settings".into(),
            ));
        }
        // Validate extra_body against typed settings even when the upstream transport sends it.
        request.body(false).map_err(error)?;
        return Ok(None);
    }
    let (system, _) = super::unified::extract_system_and_messages(messages);
    if let Some(text) = system.filter(|s| !s.is_empty()) {
        request.messages.push(sdk::Message::System(text));
    }
    // Signed batches belong to their original provider. Only complete, correlated unsigned
    // exchanges are sent as Chat tool messages; foreign exchanges get the readable projection.
    let history = tools
        .map(|_| inspect_native_history(messages))
        .transpose()?;
    if history
        .as_ref()
        .is_some_and(|h| h.has_incomplete_unsigned())
    {
        return Err(LLMError::Serialization(
            "incomplete native tool history".into(),
        ));
    }
    for (index, msg) in messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role != Role::System)
    {
        let native = history.as_ref().is_some_and(|h| {
            !h.exchanges()
                .iter()
                .any(|e| (e.message_start()..e.message_end()).contains(&index))
        });
        append_message(&mut request.messages, msg, native)?;
    }
    request_tools(&mut request, tools)?;
    Ok(Some(request))
}

pub(super) fn request_tools(
    request: &mut sdk::ChatRequest,
    tools: Option<&LLMToolRequest>,
) -> Result<(), LLMError> {
    if let Some(tools) = tools {
        request.functions = tools
            .tools
            .iter()
            .map(|t| sdk::Function {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            })
            .collect();
        request.tool_choice = Some(match &tools.choice {
            ToolChoice::Auto => sdk::FunctionChoice::Auto,
            ToolChoice::Required => sdk::FunctionChoice::Required,
            ToolChoice::None => sdk::FunctionChoice::None,
            ToolChoice::Specific(n) => sdk::FunctionChoice::Named(n.clone()),
            _ => return Err(LLMError::Config("unsupported tool choice".into())),
        });
    }
    Ok(())
}

pub(super) fn finish(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCall,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Other,
    }
}
pub(super) fn usage(u: sdk::Usage) -> TokenUsage {
    TokenUsage {
        prompt_tokens: u.prompt_tokens,
        completion_tokens: u.completion_tokens,
        total_tokens: u.total_tokens,
    }
}

pub(super) fn response(raw: sdk::Completion, model: &str) -> Result<LLMResponse, LLMError> {
    let calls = raw
        .tool_calls
        .into_iter()
        .map(|c| ToolCall {
            id: c.id,
            name: c.name,
            arguments: c.arguments,
        })
        .collect::<Vec<_>>();
    let mut response = LLMResponse::new(
        raw.text,
        if calls.is_empty() {
            finish(&raw.finish_reason)
        } else {
            FinishReason::ToolCall
        },
    );
    response.model = Some(model.into());
    response.usage = raw.usage.map(usage);
    if !calls.is_empty() {
        response.set_tool_calls(calls)?;
    }
    Ok(response)
}

pub(super) fn chunk(event: Result<sdk::Event, sdk::Error>) -> Result<LLMChunk, LLMError> {
    match event.map_err(error)? {
        sdk::Event::Text(s) => Ok(LLMChunk::new(s, false)),
        sdk::Event::Finished {
            reason,
            tool_calls,
            usage: u,
        } if tool_calls.is_empty() => Ok(LLMChunk::final_chunk("", finish(&reason), u.map(usage))),
        _ => Err(LLMError::Serialization(
            "unexpected function call in text stream".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extensions_merge_distinct_leaves_and_reject_conflicts_before_dispatch() {
        let mut cfg = LLMConfig::default();
        cfg.extra
            .insert("extra_body".into(), json!({"vendor":{"a":1}}));
        cfg.extra
            .insert("provider_options".into(), json!({"vendor":{"b":true}}));
        let body = request(ProviderType::OpenAICompatible, "model", &[], &cfg, None)
            .unwrap()
            .unwrap()
            .body(false)
            .unwrap();
        assert_eq!(body["vendor"], json!({"a":1,"b":true}));
        cfg.extra
            .insert("provider_options".into(), json!({"vendor":{"a":2}}));
        assert!(
            matches!(request(ProviderType::OpenAICompatible,"model",&[],&cfg,None), Err(LLMError::Config(e)) if e.contains("provider_options.vendor.a"))
        );
    }

    #[test]
    fn invalid_legacy_response_format_is_reported_before_dispatch() {
        for format in [
            json!({"type":"json_object", "json_schema":{"name":"ignored"}}),
            json!({"type":"json_schema", "json_schema":{"schema":{}}}),
        ] {
            let cfg =
                LLMConfig::default().with_extra("extra_body", json!({"response_format":format}));
            assert!(
                matches!(request(ProviderType::OpenAICompatible, "model", &[], &cfg, None),
                Err(LLMError::Config(error)) if error.contains("extra_body.response_format"))
            );
        }
    }

    #[test]
    fn legacy_transport_selection_never_discards_explicit_sdk_options() {
        let mut cfg = LLMConfig::default();
        cfg.extra.insert("resilient".into(), json!(true));
        assert!(
            request(ProviderType::OpenAICompatible, "model", &[], &cfg, None)
                .unwrap()
                .is_none()
        );
        cfg.extra
            .insert("protocol".into(), json!("chat-completions"));
        assert!(matches!(
            request(ProviderType::OpenAICompatible, "model", &[], &cfg, None),
            Err(LLMError::Config(_))
        ));
    }
}

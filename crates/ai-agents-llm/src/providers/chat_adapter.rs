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
use std::{future::Future, time::Duration};

use super::sdk_error as error;

pub(super) const REPORTED_MODEL: &str = "chat_reported_model";

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

// Preserve the validation of requests previously sent through upstream.
fn used_legacy_transport(cfg: &LLMConfig) -> Result<bool, LLMError> {
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
    cfg: &LLMConfig,
    mut apply: impl FnMut(&str, &str, &Value) -> Result<(), LLMError>,
) -> Result<(), LLMError> {
    for field in ["extra_body", "provider_options"] {
        if let Some(value) = cfg.extra.get(field) {
            let options = value
                .as_object()
                .ok_or_else(|| LLMError::Config(format!("{field} must be an object")))?;
            for (key, value) in options {
                apply(field, key, value)?;
            }
        }
    }
    Ok(())
}

pub(super) fn apply_option(
    request: &mut sdk::ChatRequest,
    field: &str,
    key: &str,
    value: &Value,
    explicit_null_format: Option<sdk::ResponseFormat>,
) -> Result<(), LLMError> {
    if field == "extra_body" && key == "response_format" {
        // Migrate the legacy wire shape into the owned request field.
        request.response_format = if value.is_null() {
            explicit_null_format
        } else {
            Some(serde_json::from_value(value.clone()).map_err(|error| {
                LLMError::Config(format!("invalid extra_body.response_format: {error}"))
            })?)
        };
        return Ok(());
    }
    if sdk::reserved(key) {
        return Err(LLMError::Config(format!("reserved option {field}.{key}")));
    }
    merge_option(
        &mut request.provider_options,
        key,
        value,
        &format!("{field}.{key}"),
    )
}

pub(super) fn request(
    provider: ProviderType,
    model: &str,
    messages: &[ChatMessage],
    cfg: &LLMConfig,
    tools: Option<&LLMToolRequest>,
) -> Result<Dispatch, LLMError> {
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
    let previously_legacy = used_legacy_transport(cfg)?;
    request.timeout = cfg
        .timeout_seconds
        .or_else(|| extra.get("timeout_seconds").and_then(Value::as_u64))
        .map(Duration::from_secs);
    request_options(cfg, |field, key, value| {
        apply_option(
            &mut request,
            field,
            key,
            value,
            Some(sdk::ResponseFormat::Null),
        )
    })?;
    if previously_legacy {
        if extra.contains_key("provider_options") || extra.contains_key("protocol") {
            return Err(LLMError::Config(
                "SDK options conflict with legacy Chat settings".into(),
            ));
        }
        request.reasoning_effort =
            super::unified::legacy_reasoning_effort(cfg)?.map(|effort| effort.to_string());
    }
    if provider == ProviderType::OpenAI {
        let hosted_search = extra
            .get("openai_enable_web_search")
            .and_then(Value::as_bool)
            == Some(true);
        super::openai::validate(&request, tools.is_some() || hosted_search).map_err(error)?;
    }
    let retry = super::unified::legacy_resilience(cfg);
    if retry
        .as_ref()
        .is_some_and(|policy| policy.max_attempts == 0)
    {
        return Err(LLMError::Config(
            "Chat retries require at least one total attempt".into(),
        ));
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
    Ok(Dispatch { request, retry })
}

pub(super) struct Dispatch {
    pub(super) request: sdk::ChatRequest,
    retry: Option<llm::resilient_llm::ResilienceConfig>,
}

impl Dispatch {
    pub(super) async fn complete(self, client: &sdk::ChatClient) -> Result<LLMResponse, LLMError> {
        let (output, reported) = self
            .send(|| client.complete_with_reported_model(&self.request))
            .await?;
        let mut result = response(output, &self.request.model)?;
        if let Some(model) = reported {
            result
                .metadata
                .insert(REPORTED_MODEL.into(), Value::String(model.clone()));
            result.model = Some(model);
        }
        Ok(result)
    }

    pub(super) async fn stream(
        self,
        client: &sdk::ChatClient,
        usage_model: Option<&str>,
    ) -> Result<sdk::ChatStream, LLMError> {
        // Opening the stream is the only retryable stage. Polling bytes never starts a new request.
        self.send(|| client.stream_with_usage_model(&self.request, usage_model))
            .await
    }

    async fn send<T, F: Future<Output = Result<T, sdk::Error>>>(
        &self,
        mut attempt: impl FnMut() -> F,
    ) -> Result<T, LLMError> {
        let Some(policy) = &self.retry else {
            return attempt().await.map_err(error);
        };
        let mut index = 0;
        loop {
            match attempt().await {
                Err(
                    sdk::Error::Http(408 | 429 | 500 | 502 | 503 | 504) | sdk::Error::Transport,
                ) if index + 1 < policy.max_attempts => {
                    // Keep the configured upstream backoff and jitter for explicit framework retries.
                    let mut delay = policy
                        .base_delay_ms
                        .saturating_mul(1 << index.min(16))
                        .min(policy.max_delay_ms);
                    if policy.jitter {
                        let span = (delay / 2).max(1);
                        let jitter = (index as u64)
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1)
                            % span;
                        delay = delay.saturating_sub(jitter);
                    }
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                    index += 1;
                }
                result => return result.map_err(error),
            }
        }
    }
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
            .request
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
    fn migrated_legacy_options_preserve_sampling_and_ignored_fields_on_the_wire() {
        for provider in [ProviderType::OpenAI, ProviderType::OpenAICompatible] {
            let cfg = LLMConfig {
                frequency_penalty: Some(0.3),
                presence_penalty: Some(0.2),
                stop_sequences: Some(vec!["legacy".into()]),
                reasoning_budget_tokens: Some(25),
                ..LLMConfig::default()
            }
            .with_reasoning(true)
            .with_reasoning_effort("HiGh")
            .with_extra("extra_body", json!({"stop":["sent"],"presence_penalty":0.5,"response_format":{"type":"json_object"}}));
            let dispatch = request(provider, "fixture", &[], &cfg, None).unwrap();
            assert!(dispatch.retry.is_none());
            let body = dispatch.request.body(false).unwrap();
            assert_eq!(body["stop"], json!(["sent"]));
            assert_eq!(body["presence_penalty"], 0.5);
            assert_eq!(body["response_format"], json!({"type":"json_object"}));
            assert_eq!(body["reasoning_effort"], "high");
            assert_eq!(body["max_completion_tokens"], 2048);
            for field in ["frequency_penalty", "reasoning", "reasoning_budget_tokens"] {
                assert!(body.get(field).is_none(), "{field}: {body}");
            }
            assert!(
                request(
                    provider,
                    "fixture",
                    &[],
                    &cfg.with_reasoning_effort("unknown"),
                    None
                )
                .is_err()
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
                .retry
                .is_some()
        );
        cfg.extra
            .insert("protocol".into(), json!("chat-completions"));
        assert!(matches!(
            request(ProviderType::OpenAICompatible, "model", &[], &cfg, None),
            Err(LLMError::Config(_))
        ));
    }
}

//! Anthropic Messages requests and ordered content blocks, independent of framework execution.
use super::{Error, ExpectedCalls, Function, FunctionChoice, ToolCall, transport};
use futures::Stream;
use reqwest::header::{HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::HashSet, pin::Pin, time::Duration};

/// The supported Anthropic API revision and its official base URL.
pub const API_VERSION: &str = "2023-06-01";
pub(crate) const OFFICIAL_BASE: &str = "https://api.anthropic.com/v1";
pub(crate) const OFFICIAL_ORIGIN: &str = "https://api.anthropic.com";

/// Credentials explicitly selected for this client and its endpoint.
pub enum MessagesAuth {
    None,
    ApiKey(String),
    Bearer(String),
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Thinking {
    Adaptive,
    Enabled { budget_tokens: u32 },
    Disabled,
}

pub struct MessagesRequest {
    pub model: String,
    /// Ordered Messages objects, including original assistant blocks for tool continuation.
    pub messages: Vec<Value>,
    pub system: Option<String>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub thinking: Option<Thinking>,
    pub effort: Option<String>,
    pub functions: Vec<Function>,
    pub tool_choice: Option<FunctionChoice>,
    pub timeout: Option<Duration>,
}

impl MessagesRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Value>, max_tokens: u32) -> Self {
        Self {
            model: model.into(),
            messages,
            system: None,
            max_tokens,
            temperature: None,
            top_p: None,
            top_k: None,
            thinking: None,
            effort: None,
            functions: Vec::new(),
            tool_choice: None,
            timeout: None,
        }
    }

    fn expected(&self) -> ExpectedCalls {
        ExpectedCalls {
            names: self
                .functions
                .iter()
                .map(|tool| tool.name.clone())
                .collect(),
            choice: self.tool_choice.clone(),
        }
    }

    fn body(&self, stream: bool) -> Result<Value, Error> {
        if self.max_tokens == 0 {
            return Err(Error::Config(
                "Messages generation requires max_tokens greater than zero".into(),
            ));
        }
        let mut body = json!({
            "model": self.model, "messages": self.messages, "max_tokens": self.max_tokens, "stream": stream,
        });
        if let Some(system) = &self.system {
            body["system"] = json!(system);
        }
        for (field, value) in [("temperature", self.temperature), ("top_p", self.top_p)] {
            if let Some(value) = value {
                body[field] = super::decimal(value)?;
            }
        }
        if let Some(top_k) = self.top_k {
            body["top_k"] = json!(top_k);
        }
        if let Some(thinking) = &self.thinking {
            body["thinking"] = json!(thinking);
        }
        if let Some(effort) = &self.effort {
            if effort.trim().is_empty() {
                return Err(Error::Config("empty Messages effort".into()));
            }
            body["output_config"] = json!({"effort": effort});
        }
        if !self.functions.is_empty() {
            body["tools"] = Value::Array(self.functions.iter().map(|tool| {
                json!({"name": tool.name, "description": tool.description, "input_schema": tool.parameters})
            }).collect());
        }
        if let Some(choice) = &self.tool_choice {
            body["tool_choice"] = match choice {
                FunctionChoice::Auto => json!({"type": "auto"}),
                FunctionChoice::None => json!({"type": "none"}),
                FunctionChoice::Required if !self.functions.is_empty() => json!({"type": "any"}),
                FunctionChoice::Named(name)
                    if self.functions.iter().any(|tool| &tool.name == name) =>
                {
                    json!({"type": "tool", "name": name})
                }
                _ => {
                    return Err(Error::Config(
                        "Messages tool choice has no matching definition".into(),
                    ));
                }
            };
        }
        Ok(body)
    }
}

// Exact official IDs from Anthropic's "Forcing tool use" table. Compatible endpoints own their rules.
const AUTO_OR_NONE_ONLY: &str = include_str!("anthropic_auto_or_none_only.txt");

pub(crate) fn allows_choice(
    official: bool,
    model: &str,
    thinking: Option<&Thinking>,
    choice: &FunctionChoice,
) -> bool {
    !official
        || !matches!(choice, FunctionChoice::Required | FunctionChoice::Named(_))
        || (!matches!(thinking, Some(Thinking::Enabled { .. }))
            && !AUTO_OR_NONE_ONLY.lines().any(|id| id == model))
}

#[derive(Clone)]
pub struct MessagesClient {
    transport: transport::Transport,
    official: bool,
    version: String,
}

impl MessagesClient {
    /// Creates a client for the official Anthropic origin.
    pub fn new(key: String) -> Result<Self, Error> {
        Self::at_endpoint(OFFICIAL_BASE, MessagesAuth::ApiKey(key), API_VERSION, true)
    }

    /// The base URL includes the API prefix. Credentials come from the caller's endpoint configuration.
    pub fn compatible(base_url: &str, auth: MessagesAuth, version: &str) -> Result<Self, Error> {
        Self::at_endpoint(base_url, auth, version, false)
    }

    pub(crate) fn at_endpoint(
        base_url: &str,
        auth: MessagesAuth,
        version: &str,
        official: bool,
    ) -> Result<Self, Error> {
        let mut headers = HeaderMap::new();
        if version.trim().is_empty() {
            return Err(Error::Config("missing anthropic-version".into()));
        }
        headers.insert(
            "anthropic-version",
            HeaderValue::from_str(version)
                .map_err(|_| Error::Config("invalid anthropic-version".into()))?,
        );
        match auth {
            MessagesAuth::None => {}
            MessagesAuth::ApiKey(key) if !key.is_empty() => {
                headers.insert("x-api-key", transport::secret(&key)?);
            }
            MessagesAuth::Bearer(key) if !key.is_empty() => {
                headers.extend(transport::bearer_headers(Some(key))?);
            }
            _ => return Err(Error::Config("empty Messages credential".into())),
        }
        Ok(Self {
            transport: transport::Transport::new(base_url, "messages", headers)?,
            official,
            version: version.into(),
        })
    }

    pub(crate) fn endpoint(&self) -> &str {
        self.transport.endpoint.as_str()
    }
    pub(crate) fn version(&self) -> &str {
        &self.version
    }
    pub(crate) fn official(&self) -> bool {
        self.official
    }

    async fn send(
        &self,
        request: &MessagesRequest,
        stream: bool,
    ) -> Result<reqwest::Response, Error> {
        if let Some(choice) = &request.tool_choice
            && !allows_choice(
                self.official,
                &request.model,
                request.thinking.as_ref(),
                choice,
            )
        {
            return Err(Error::Config(
                "the selected Anthropic model and thinking mode do not support forced tool choice"
                    .into(),
            ));
        }
        if self.official
            && let Some(Thinking::Enabled { budget_tokens }) = request.thinking
            && !(1024..request.max_tokens).contains(&budget_tokens)
        {
            return Err(Error::Config(
                "manual thinking budget must be at least 1024 and below max_tokens".into(),
            ));
        }
        self.transport
            .send(&request.body(stream)?, request.timeout)
            .await
    }

    pub async fn complete(&self, request: &MessagesRequest) -> Result<MessageOutput, Error> {
        let mut response: Value = self
            .send(request, false)
            .await?
            .json()
            .await
            .map_err(|_| Error::Protocol("invalid Messages response"))?;
        if response["role"] != "assistant" {
            return Err(Error::Protocol("expected an assistant message"));
        }
        let Value::Array(blocks) = response["content"].take() else {
            return Err(Error::Protocol("missing Messages content"));
        };
        let usage = serde_json::from_value::<UsageUpdate>(response["usage"].take())
            .ok()
            .and_then(UsageUpdate::finish);
        finish(
            string(&response, "id")?.to_owned(),
            blocks,
            string(&response, "stop_reason")?,
            usage,
            &request.expected(),
        )
    }

    pub async fn stream(&self, request: &MessagesRequest) -> Result<MessagesStream, Error> {
        Ok(stream::decode(
            transport::frames(self.send(request, true).await?.bytes_stream()),
            request.expected(),
        ))
    }
}

/// Preserve private blocks for trusted replay. Display only `text` in ordinary interfaces.
pub struct MessageOutput {
    pub id: String,
    pub blocks: Vec<Value>,
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<MessagesUsage>,
}

#[derive(Clone, Copy, Debug)]
pub struct MessagesUsage {
    pub uncached_input_tokens: u32,
    pub cache_read_input_tokens: u32,
    pub cache_creation_input_tokens: u32,
    pub output_tokens: u32,
}

impl MessagesUsage {
    /// Returns the full input and output counts after checked addition.
    pub fn totals(self) -> Option<(u32, u32, u32)> {
        let input = self
            .uncached_input_tokens
            .checked_add(self.cache_read_input_tokens)?
            .checked_add(self.cache_creation_input_tokens)?;
        Some((
            input,
            self.output_tokens,
            input.checked_add(self.output_tokens)?,
        ))
    }
}

#[derive(Default, Deserialize)]
struct UsageUpdate {
    input_tokens: Option<u32>,
    cache_read_input_tokens: Option<u32>,
    cache_creation_input_tokens: Option<u32>,
    output_tokens: Option<u32>,
}

impl UsageUpdate {
    fn finish(self) -> Option<MessagesUsage> {
        let result = MessagesUsage {
            uncached_input_tokens: self.input_tokens?,
            cache_read_input_tokens: self.cache_read_input_tokens.unwrap_or(0),
            cache_creation_input_tokens: self.cache_creation_input_tokens.unwrap_or(0),
            output_tokens: self.output_tokens?,
        };
        result.totals()?;
        Some(result)
    }
}

pub enum MessageEvent {
    Text(String),
    Finished(MessageOutput),
}

pub type MessagesStream = Pin<Box<dyn Stream<Item = Result<MessageEvent, Error>> + Send>>;

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, Error> {
    value[key]
        .as_str()
        .ok_or(Error::Protocol("missing Messages string field"))
}

pub(crate) fn validate_block(block: &Value) -> Result<Option<ToolCall>, Error> {
    match string(block, "type")? {
        "text" => {
            string(block, "text")?;
        }
        "thinking" => {
            string(block, "thinking")?;
            if string(block, "signature")?.is_empty() {
                return Err(Error::Protocol("missing thinking signature"));
            }
        }
        "redacted_thinking" if !string(block, "data")?.is_empty() => {}
        "tool_use" => {
            let id = string(block, "id")?;
            let name = string(block, "name")?;
            if id.is_empty() || name.is_empty() || !block["input"].is_object() {
                return Err(Error::Protocol("invalid Messages tool block"));
            }
            return Ok(Some(ToolCall {
                id: id.into(),
                name: name.into(),
                arguments: block["input"].clone(),
            }));
        }
        _ => return Err(Error::Protocol("unsupported Messages content block")),
    }
    Ok(None)
}

fn finish(
    id: String,
    blocks: Vec<Value>,
    stop: &str,
    usage: Option<MessagesUsage>,
    expected: &ExpectedCalls,
) -> Result<MessageOutput, Error> {
    match stop {
        "end_turn" | "stop_sequence" | "tool_use" => {}
        "refusal" => return Err(Error::Refused),
        "max_tokens" | "model_context_window_exceeded" => {
            return Err(Error::Protocol("Messages output limit reached"));
        }
        _ => return Err(Error::Protocol("unsupported Messages stop reason")),
    }
    let mut text = String::new();
    let mut calls = Vec::new();
    let mut ids = HashSet::new();
    if id.is_empty() {
        return Err(Error::Protocol("missing message id"));
    }
    for block in &blocks {
        if let Some(call) = validate_block(block)? {
            if !ids.insert(call.id.clone()) {
                return Err(Error::Protocol("duplicate Messages tool id"));
            }
            calls.push(call);
        } else if block["type"] == "text" {
            text.push_str(string(block, "text")?);
        }
    }
    if calls.is_empty() == (stop == "tool_use") {
        return Err(Error::Protocol(
            "Messages stop reason disagrees with tool blocks",
        ));
    }
    expected.check(&calls)?;
    Ok(MessageOutput {
        id,
        blocks,
        text,
        tool_calls: calls,
        usage,
    })
}

#[path = "messages_stream.rs"]
mod stream;

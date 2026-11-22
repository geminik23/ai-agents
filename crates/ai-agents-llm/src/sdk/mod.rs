//! Direct, single-attempt Chat Completions calls. The caller owns history and tool execution.
//! Framework callers normally use `UnifiedLLMProvider` to retain task admission and accounting.
//!
//! ```no_run
//! use ai_agents_llm::sdk::{AudioFormat, ChatClient, ChatRequest, InputPart, Message};
//!
//! # async fn ask(wav: Vec<u8>, key: String) -> Result<(), ai_agents_llm::sdk::Error> {
//! let client = ChatClient::new("https://api.openai.com/v1", Some(key))?;
//! let input = Message::UserParts(vec![
//!     InputPart::Text("Summarize this recording.".into()),
//!     InputPart::Audio { format: AudioFormat::Wav, data: wav },
//! ]);
//! let answer = client.complete(&ChatRequest::new("your-audio-capable-model", vec![input])).await?;
//! println!("{}", answer.text);
//! # Ok(())
//! # }
//! ```
pub mod decisions;
pub mod messages;
pub(crate) mod openai;
pub mod responses;
// Bound one unfinished SSE frame independently of total response length.
pub(crate) const MAX_SSE_FRAME_BYTES: usize = 1024 * 1024;
mod stream;
mod transport;
mod wire;

use base64::{Engine, engine::general_purpose::STANDARD};
use futures::Stream;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{pin::Pin, time::Duration};

/// User content in wire order. Media bytes are encoded only when building a request.
#[non_exhaustive]
pub enum InputPart {
    Text(String),
    ImageUrl(String),
    ImageBytes { mime_type: String, data: Vec<u8> },
    Audio { format: AudioFormat, data: Vec<u8> },
}

#[non_exhaustive]
pub enum AudioFormat {
    Wav,
    Mp3,
}

impl InputPart {
    fn validate(&self) -> Result<(), Error> {
        match self {
            Self::ImageBytes { mime_type, data } => {
                let valid = mime_type.strip_prefix("image/").is_some_and(|s| {
                    !s.is_empty()
                        && s.bytes()
                            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'+'))
                });
                if !valid || data.is_empty() {
                    return Err(Error::Config(
                        "expected image bytes and an image MIME type".into(),
                    ));
                }
            }
            Self::ImageUrl(url) => {
                if !Url::parse(url).is_ok_and(|u| {
                    matches!(u.scheme(), "http" | "https")
                        && u.host().is_some()
                        && u.username().is_empty()
                        && u.password().is_none()
                }) {
                    return Err(Error::Config("expected an HTTP image URL".into()));
                }
            }
            Self::Audio { data, .. } if data.is_empty() => {
                return Err(Error::Config("empty audio input".into()));
            }
            _ => {}
        }
        Ok(())
    }

    fn encode(&self) -> Value {
        match self {
            Self::Text(text) => json!({"type":"text","text":text}),
            Self::ImageUrl(url) => json!({"type":"image_url","image_url":{"url":url}}),
            Self::ImageBytes { mime_type, data } => {
                let url = format!("data:{mime_type};base64,{}", STANDARD.encode(data));
                json!({"type":"image_url","image_url":{"url":url}})
            }
            Self::Audio { format, data } => {
                let format = match format {
                    AudioFormat::Wav => "wav",
                    AudioFormat::Mp3 => "mp3",
                };
                json!({"type":"input_audio","input_audio":{"format":format,"data":STANDARD.encode(data)}})
            }
        }
    }
}

#[non_exhaustive]
pub enum Message {
    System(String),
    User(String),
    UserParts(Vec<InputPart>),
    Assistant {
        text: Option<String>,
        calls: Vec<ToolCall>,
    },
    ToolResult {
        id: String,
        content: String,
    },
}

impl Message {
    fn encode(&self) -> Value {
        match self {
            Self::System(text) => json!({"role":"system","content":[{"type":"text","text":text}]}),
            Self::User(text) => json!({"role":"user","content":text}),
            Self::UserParts(parts) => Value::Object(Map::from_iter([
                ("role".into(), json!("user")),
                (
                    "content".into(),
                    Value::Array(parts.iter().map(InputPart::encode).collect()),
                ),
            ])),
            Self::Assistant { text, calls } => {
                let mut message = json!({"role":"assistant","content":text});
                if !calls.is_empty() {
                    let encoded = calls
                        .iter()
                        .map(|call| {
                            json!({
                                "type":"function", "id":call.id,
                                "function":{"name":call.name,"arguments":call.arguments.to_string()}
                            })
                        })
                        .collect();
                    message["tool_calls"] = Value::Array(encoded);
                }
                message
            }
            Self::ToolResult { id, content } => {
                json!({"role":"tool","tool_call_id":id,"content":content})
            }
        }
    }
}

pub struct Function {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Clone)]
pub enum FunctionChoice {
    Auto,
    Required,
    None,
    Named(String),
}

// Bind returned calls to the exact request, including while the HTTP stream outlives it.
struct ExpectedCalls {
    names: Vec<String>,
    choice: Option<FunctionChoice>,
}

impl ExpectedCalls {
    fn from_request(r: &ChatRequest) -> Self {
        Self {
            names: r.functions.iter().map(|f| f.name.clone()).collect(),
            choice: r.tool_choice.clone(),
        }
    }

    fn check(&self, calls: &[ToolCall]) -> Result<(), Error> {
        if calls.is_empty()
            && matches!(
                self.choice,
                Some(FunctionChoice::Required | FunctionChoice::Named(_))
            )
        {
            return Err(Error::Protocol("required function missing"));
        }
        for call in calls {
            if !self.names.contains(&call.name)
                || matches!(self.choice, Some(FunctionChoice::None))
                || matches!(&self.choice,Some(FunctionChoice::Named(n)) if *n!=call.name)
            {
                return Err(Error::Protocol(
                    "function is outside the requested selection",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    /// Informational breakdowns supplied by the server; already included in the totals.
    pub prompt_tokens_details: Option<std::collections::BTreeMap<String, Option<u32>>>,
    pub completion_tokens_details: Option<std::collections::BTreeMap<String, Option<u32>>>,
}

/// Response format requested from the Chat endpoint. The caller validates the returned text.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[non_exhaustive]
pub enum ResponseFormat {
    Text {},
    JsonObject {},
    JsonSchema { json_schema: ResponseSchema },
}

/// Provider-side constraints for a structured response.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseSchema {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub functions: Vec<Function>,
    pub tool_choice: Option<FunctionChoice>,
    pub temperature: Option<f32>,
    pub max_completion_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub reasoning_effort: Option<String>,
    pub response_format: Option<ResponseFormat>,
    /// Explicit provider extensions. Transport, authority and accounting fields are reserved.
    pub provider_options: Map<String, Value>,
    pub timeout: Option<Duration>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            functions: vec![],
            tool_choice: None,
            temperature: None,
            max_completion_tokens: None,
            top_p: None,
            top_k: None,
            reasoning_effort: None,
            response_format: None,
            provider_options: Map::new(),
            timeout: None,
        }
    }

    pub(crate) fn body(&self, streaming: bool) -> Result<Value, Error> {
        if matches!(&self.tool_choice,Some(FunctionChoice::Named(name)) if !self.functions.iter().any(|f|f.name==*name))
            || (self.functions.is_empty()
                && matches!(self.tool_choice, Some(FunctionChoice::Required)))
        {
            return Err(Error::Config(
                "tool choice has no matching definition".into(),
            ));
        }
        for message in &self.messages {
            if let Message::UserParts(parts) = message {
                if parts.is_empty() {
                    return Err(Error::Config("empty user content parts".into()));
                }
                for part in parts {
                    part.validate()?;
                }
            }
        }
        let mut body = Map::from_iter([
            ("model".into(), json!(self.model)),
            (
                "messages".into(),
                Value::Array(self.messages.iter().map(Message::encode).collect()),
            ),
            ("stream".into(), json!(streaming)),
        ]);
        if let Some(t) = self.temperature {
            body.insert("temperature".into(), decimal(t)?);
        }
        if let Some(n) = self.max_completion_tokens {
            body.insert("max_completion_tokens".into(), json!(n));
        }
        if let Some(p) = self.top_p {
            body.insert("top_p".into(), decimal(p)?);
        }
        if let Some(k) = self.top_k {
            body.insert("top_k".into(), json!(k));
        }
        if let Some(effort) = &self.reasoning_effort {
            if effort.trim().is_empty() {
                return Err(Error::Config("empty reasoning_effort".into()));
            }
            body.insert("reasoning_effort".into(), json!(effort));
        }
        if !self.functions.is_empty() {
            let tools=self.functions.iter().map(|tool|json!({
                "type":"function",
                "function":{"name":tool.name,"description":tool.description,"parameters":tool.parameters}
            })).collect();
            body.insert("tools".into(), Value::Array(tools));
            if let Some(choice) = &self.tool_choice {
                body.insert(
                    "tool_choice".into(),
                    match choice {
                        FunctionChoice::Auto => json!("auto"),
                        FunctionChoice::None => json!("none"),
                        FunctionChoice::Required => json!("required"),
                        FunctionChoice::Named(name) => {
                            json!({"type":"function","function":{"name":name}})
                        }
                    },
                );
            }
        }
        if let Some(format) = &self.response_format {
            if matches!(format, ResponseFormat::JsonSchema { json_schema } if json_schema.name.trim().is_empty())
            {
                return Err(Error::Config(
                    "response_format.json_schema.name must be nonempty".into(),
                ));
            }
            body.insert("response_format".into(), json!(format));
        }
        if streaming {
            body.insert("stream_options".into(), json!({"include_usage":true}));
        }
        for (key, value) in &self.provider_options {
            if body.contains_key(key) || reserved(key) {
                return Err(Error::Config(format!(
                    "reserved or conflicting provider option: {key}"
                )));
            }
            if key == "reasoning_effort" && !value.as_str().is_some_and(|s| !s.trim().is_empty()) {
                return Err(Error::Config("invalid reasoning_effort".into()));
            }
            body.insert(key.clone(), value.clone());
        }
        Ok(Value::Object(body))
    }
}

// Keep the decimal representation of the original f32 rather than widening it on the wire.
fn decimal(value: f32) -> Result<Value, Error> {
    value
        .to_string()
        .parse()
        .map(Value::Number)
        .map_err(|_| Error::Config("sampling options must be finite".into()))
}

pub(crate) fn reserved(key: &str) -> bool {
    matches!(
        key,
        "model"
            | "messages"
            | "input"
            | "tools"
            | "functions"
            | "function_call"
            | "tool_choice"
            | "parallel_tool_calls"
            | "stream"
            | "stream_options"
            | "max_tokens"
            | "max_completion_tokens"
            | "max_output_tokens"
            | "response_format"
            | "web_search_options"
            | "modalities"
            | "audio"
            | "n"
            | "api_key"
            | "base_url"
            | "headers"
            | "extra_body"
            | "provider_options"
    )
}

pub struct Completion {
    pub text: String,
    pub finish_reason: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<Usage>,
}

#[non_exhaustive]
pub enum Event {
    Text(String),
    /// Complete tool arguments and final usage, emitted only after protocol termination.
    Finished {
        reason: String,
        tool_calls: Vec<ToolCall>,
        usage: Option<Usage>,
    },
}

pub type ChatStream = Pin<Box<dyn Stream<Item = Result<Event, Error>> + Send>>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("provider refused completion")]
    Refused,
    #[error("invalid provider configuration: {0}")]
    Config(String),
    #[error("provider protocol error: {0}")]
    Protocol(&'static str),
    #[error("provider transport failed")]
    Transport,
    #[error("provider endpoint returned HTTP {0}")]
    Http(u16),
}

/// Reusable Chat Completions client with request-local settings.
#[derive(Clone)]
pub struct ChatClient {
    transport: transport::Transport,
    openai_contract: bool,
}
impl ChatClient {
    pub fn new(base_url: &str, api_key: Option<String>) -> Result<Self, Error> {
        let transport = transport::Transport::new(
            base_url,
            "chat/completions",
            transport::bearer_headers(api_key)?,
        )?;
        let openai_contract =
            transport.endpoint.origin().ascii_serialization() == "https://api.openai.com";
        Ok(Self {
            transport,
            openai_contract,
        })
    }
    // A framework preset explicitly selects the contract, including when it uses a custom origin.
    pub(crate) fn with_openai_contract(mut self, openai: bool) -> Self {
        self.openai_contract = openai;
        self
    }

    async fn send(&self, request: &ChatRequest, stream: bool) -> Result<reqwest::Response, Error> {
        if self.openai_contract {
            openai::validate(request, false)?;
        }
        self.transport
            .send(&request.body(stream)?, request.timeout)
            .await
    }

    /// Send one non-streaming request and return the complete assistant turn.
    pub async fn complete(&self, request: &ChatRequest) -> Result<Completion, Error> {
        let response = self.send(request, false).await?;
        let payload = response
            .json()
            .await
            .map_err(|_| Error::Protocol("invalid completion JSON"))?;
        let output = wire::complete(payload)?;
        ExpectedCalls::from_request(request).check(&output.tool_calls)?;
        Ok(output)
    }

    /// Text events are provisional. A finished event carries assembled calls and any final usage receipt.
    pub async fn stream(&self, request: &ChatRequest) -> Result<ChatStream, Error> {
        Ok(stream::decode(
            self.send(request, true).await?.bytes_stream(),
            ExpectedCalls::from_request(request),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn official_gpt_6_2_chat_rejects_unsupported_requests_before_io() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let client = ChatClient::new(
            &format!("http://{}/v1", listener.local_addr().unwrap()),
            None,
        )
        .unwrap()
        .with_openai_contract(true);
        let mut request = ChatRequest::new("gpt-6.2-astra", vec![Message::User("test".into())]);
        request.timeout = Some(Duration::from_secs(2));
        request.temperature = Some(0.0);
        assert!(
            matches!(client.complete(&request).await, Err(Error::Config(message)) if message.contains("temperature"))
        );
        request.temperature = None;
        request.functions.push(Function {
            name: "lookup".into(),
            description: "look up a value".into(),
            parameters: json!({"type":"object"}),
        });
        assert!(
            matches!(client.stream(&request).await, Err(Error::Config(message)) if message.contains("Responses API"))
        );
        let accept = listener.accept();
        tokio::pin!(accept);
        assert!(futures::poll!(accept).is_pending());
    }
}

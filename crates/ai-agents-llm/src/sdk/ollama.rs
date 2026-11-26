//! Native Local and Cloud Ollama chat. The caller owns history and execution of returned tools.

use super::{Error, Function, transport};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{pin::Pin, time::Duration};

pub(crate) const CLOUD_API_BASE: &str = "https://ollama.com/api";
pub(crate) const CLOUD_ORIGIN: &str = "https://ollama.com";
pub(crate) const LOCAL_ROOT: &str = "http://localhost:11434";
const MAX_NDJSON_RECORD_BYTES: usize = 1024 * 1024;

/// Caller-selected reasoning mode. String values are defined by the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Think {
    Enabled(bool),
    Level(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<Call>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            thinking: None,
            tool_calls: Vec::new(),
            tool_name: None,
            tool_call_id: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Call {
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub function: CallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CallFunction {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    pub name: String,
    pub arguments: Map<String, Value>,
}

pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub think: Option<Think>,
    /// Native generation options such as `num_predict` and `temperature`.
    pub options: Map<String, Value>,
    /// Functions available for automatic selection. An empty list omits tool definitions.
    pub functions: Vec<Function>,
    pub timeout: Option<Duration>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            think: None,
            options: Map::new(),
            functions: Vec::new(),
            timeout: None,
        }
    }

    fn body(&self, stream: bool) -> Result<Value, Error> {
        if self.model.trim().is_empty() {
            return Err(Error::Config("Ollama chat requires a model".into()));
        }
        if matches!(&self.think, Some(Think::Level(level)) if level.trim().is_empty()) {
            return Err(Error::Config("Ollama think level must not be empty".into()));
        }
        let mut body = json!({"model":self.model, "messages":self.messages, "stream":stream});
        if let Some(think) = &self.think {
            body["think"] = json!(think);
        }
        if !self.options.is_empty() {
            body["options"] = json!(self.options);
        }
        if !self.functions.is_empty() {
            body["tools"] = json!(self.functions.iter().map(|function| json!({
                "type":"function",
                "function":{"name":function.name,"description":function.description,"parameters":function.parameters}
            })).collect::<Vec<_>>());
        }
        Ok(body)
    }
}

/// Reported token counters from the terminal response. Disjoint usage is not estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub cached_prompt_tokens: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Finish {
    Stop,
    Length,
    ToolCall,
}

#[derive(Debug)]
pub struct ChatOutput {
    pub message: Message,
    pub finish: Finish,
    pub done_reason: String,
    pub usage: Option<Usage>,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum ChatEvent {
    Text(String),
    Finished(ChatOutput),
}

type EventStream<T> = Pin<Box<dyn Stream<Item = Result<T, Error>> + Send>>;

pub type ChatStream = EventStream<ChatEvent>;

/// How long the daemon keeps the model loaded. Seconds may be zero or negative.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum KeepAlive {
    Duration(String),
    Seconds(i64),
}

/// A local chat request with an optional daemon model lifetime.
pub struct LocalChatRequest {
    pub chat: ChatRequest,
    pub keep_alive: Option<KeepAlive>,
}

impl LocalChatRequest {
    pub fn new(chat: ChatRequest) -> Self {
        Self {
            chat,
            keep_alive: None,
        }
    }

    fn body(&self, stream: bool) -> Result<Value, Error> {
        validate_local_options(&self.chat.options)?;
        let mut body = self.chat.body(stream)?;
        if let Some(keep_alive) = &self.keep_alive {
            if matches!(keep_alive, KeepAlive::Duration(value) if value.trim().is_empty()) {
                return Err(Error::Config(
                    "Ollama keep_alive duration must not be empty".into(),
                ));
            }
            body["keep_alive"] = json!(keep_alive);
        }
        Ok(body)
    }
}

// Type-check the local options we document and forward daemon-specific extensions verbatim.
fn validate_local_options(options: &Map<String, Value>) -> Result<(), Error> {
    for (name, value) in options {
        let expected = match name.as_str() {
            // Existing framework requests encode absent sampling values as explicit nulls.
            "top_k" | "top_p" if value.is_null() => continue,
            "temperature" | "top_p" if !value.is_number() => "a number",
            "num_predict" | "num_ctx" | "num_gpu" | "top_k" | "seed" if !value.is_i64() => {
                "an integer"
            }
            "stop"
                if !value
                    .as_array()
                    .is_some_and(|items| items.iter().all(Value::is_string)) =>
            {
                "an array of strings"
            }
            _ => continue,
        };
        return Err(Error::Config(format!(
            "Ollama options.{name} must be {expected}"
        )));
    }
    Ok(())
}

/// Completed Local chat with optional metadata from the daemon.
#[derive(Debug)]
pub struct LocalChatOutput {
    pub message: Message,
    pub finish: Option<Finish>,
    pub done_reason: Option<String>,
    pub usage: Option<Usage>,
    pub model: Option<String>,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum LocalChatEvent {
    Text(String),
    Finished(LocalChatOutput),
}

pub type LocalChatStream = EventStream<LocalChatEvent>;

#[derive(Clone)]
pub struct OllamaLocalClient {
    transport: transport::Transport,
}

impl OllamaLocalClient {
    /// Opens an unauthenticated client using a daemon root, optionally with a path prefix.
    pub fn new(root: Option<&str>) -> Result<Self, Error> {
        Ok(Self {
            transport: transport::Transport::new(
                root.unwrap_or(LOCAL_ROOT),
                "api/chat",
                Default::default(),
            )?,
        })
    }

    pub async fn complete(&self, request: &LocalChatRequest) -> Result<LocalChatOutput, Error> {
        complete(
            &self.transport,
            request.body(false)?,
            &request.chat,
            FinishPolicy::AllowOmitted,
        )
        .await
    }

    pub async fn stream(&self, request: &LocalChatRequest) -> Result<LocalChatStream, Error> {
        stream(
            &self.transport,
            request.body(true)?,
            &request.chat,
            FinishPolicy::AllowOmitted,
            Ok,
        )
        .await
    }
}

#[derive(Clone)]
pub struct OllamaCloudClient {
    transport: transport::Transport,
}

impl OllamaCloudClient {
    /// Binds the caller's explicit credential to the official Cloud origin.
    pub fn new(key: String) -> Result<Self, Error> {
        Self::at_endpoint(CLOUD_API_BASE, key)
    }

    pub(crate) fn at_endpoint(base: &str, key: String) -> Result<Self, Error> {
        if key.trim().is_empty() {
            return Err(Error::Config(
                "Ollama Cloud requires a bearer credential".into(),
            ));
        }
        Ok(Self {
            transport: transport::Transport::new(
                base,
                "chat",
                transport::bearer_headers(Some(key))?,
            )?,
        })
    }

    pub(crate) fn endpoint(&self) -> &str {
        self.transport.endpoint.as_str()
    }

    pub async fn complete(&self, request: &ChatRequest) -> Result<ChatOutput, Error> {
        cloud_output(
            complete(
                &self.transport,
                request.body(false)?,
                request,
                FinishPolicy::Require,
            )
            .await?,
        )
    }

    pub async fn stream(&self, request: &ChatRequest) -> Result<ChatStream, Error> {
        stream(
            &self.transport,
            request.body(true)?,
            request,
            FinishPolicy::Require,
            |event| match event {
                LocalChatEvent::Text(text) => Ok(ChatEvent::Text(text)),
                LocalChatEvent::Finished(output) => cloud_output(output).map(ChatEvent::Finished),
            },
        )
        .await
    }
}

fn cloud_output(output: LocalChatOutput) -> Result<ChatOutput, Error> {
    Ok(ChatOutput {
        message: output.message,
        finish: output
            .finish
            .ok_or(Error::Protocol("missing Ollama done reason"))?,
        done_reason: output
            .done_reason
            .ok_or(Error::Protocol("missing Ollama done reason"))?,
        usage: output.usage,
    })
}

async fn complete(
    transport: &transport::Transport,
    body: Value,
    request: &ChatRequest,
    finish_policy: FinishPolicy,
) -> Result<LocalChatOutput, Error> {
    let response = transport.send(&body, request.timeout).await?;
    let wire = response.json::<Record>().await.map_err(|error| {
        if error.is_decode() {
            Error::Protocol("invalid Ollama chat response")
        } else {
            Error::Transport
        }
    })?;
    let mut assembly = Assembly::new(request, finish_policy);
    assembly.accept(wire)?;
    assembly.finish()
}

async fn stream<T: Send + 'static>(
    transport: &transport::Transport,
    body: Value,
    request: &ChatRequest,
    finish_policy: FinishPolicy,
    map: fn(LocalChatEvent) -> Result<T, Error>,
) -> Result<EventStream<T>, Error> {
    let response = transport.send(&body, request.timeout).await?;
    Ok(decode(
        response.bytes_stream(),
        Assembly::new(request, finish_policy),
        map,
    ))
}

#[derive(Deserialize)]
struct Record {
    model: Option<String>,
    message: Option<Message>,
    done: Option<bool>,
    done_reason: Option<String>,
    prompt_eval_count: Option<u32>,
    eval_count: Option<u32>,
    prompt_eval_cached_count: Option<u32>,
    error: Option<String>,
}

impl Record {
    fn usage(&self) -> Option<Usage> {
        let prompt_tokens = self.prompt_eval_count?;
        let completion_tokens = self.eval_count?;
        Some(Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.checked_add(completion_tokens)?,
            cached_prompt_tokens: self.prompt_eval_cached_count,
        })
    }
}

#[derive(Clone, Copy)]
enum FinishPolicy {
    AllowOmitted,
    Require,
}

struct Assembly {
    message: Message,
    terminal: Option<Record>,
    functions: Vec<String>,
    saw_message: bool,
    finish_policy: FinishPolicy,
}

impl Assembly {
    fn new(request: &ChatRequest, finish_policy: FinishPolicy) -> Self {
        Self {
            message: Message::text("assistant", ""),
            terminal: None,
            functions: request
                .functions
                .iter()
                .map(|function| function.name.clone())
                .collect(),
            saw_message: false,
            finish_policy,
        }
    }

    fn accept(&mut self, mut record: Record) -> Result<String, Error> {
        if record.error.is_some() {
            return Err(Error::Protocol(
                "Ollama generation returned an error record",
            ));
        }
        if self.terminal.is_some() {
            return Err(Error::Protocol("Ollama record after completion"));
        }
        let done = record
            .done
            .ok_or(Error::Protocol("missing Ollama done flag"))?;
        // Cloud must reject a malformed terminal record before exposing anything from it.
        if done
            && record.done_reason.is_none()
            && matches!(self.finish_policy, FinishPolicy::Require)
        {
            return Err(Error::Protocol("missing Ollama done reason"));
        }
        let mut text = String::new();
        if let Some(message) = record.message.take() {
            if message.role != "assistant"
                || message.tool_name.is_some()
                || message.tool_call_id.is_some()
            {
                return Err(Error::Protocol("invalid Ollama assistant message"));
            }
            self.saw_message = true;
            text = message.content;
            self.message.content.push_str(&text);
            if let Some(thinking) = message.thinking {
                self.message
                    .thinking
                    .get_or_insert_with(String::new)
                    .push_str(&thinking);
            }
            for call in message.tool_calls {
                if call.kind.as_deref().is_some_and(|kind| kind != "function")
                    || call.id.as_deref().is_some_and(|id| id.trim().is_empty())
                {
                    return Err(Error::Protocol("invalid Ollama function call identity"));
                }
                if !self.functions.contains(&call.function.name) {
                    return Err(Error::Protocol(
                        "Ollama tool is outside the requested selection",
                    ));
                }
                if !self.message.tool_calls.is_empty() {
                    return Err(Error::Protocol(
                        "Ollama supports one tool call per response",
                    ));
                }
                self.message.tool_calls.push(call);
            }
        }
        if done {
            self.terminal = Some(record);
        }
        Ok(text)
    }

    fn finish(self) -> Result<LocalChatOutput, Error> {
        let terminal = self
            .terminal
            .ok_or(Error::Protocol("Ollama stream ended without completion"))?;
        if !self.saw_message {
            return Err(Error::Protocol("missing Ollama assistant message"));
        }
        let finish = match (
            terminal.done_reason.as_deref(),
            self.message.tool_calls.is_empty(),
        ) {
            (None, true) => None,
            (Some("stop"), true) => Some(Finish::Stop),
            (Some("length"), true) => Some(Finish::Length),
            (Some("stop" | "tool_calls"), false) => Some(Finish::ToolCall),
            _ => return Err(Error::Protocol("invalid Ollama completion reason")),
        };
        Ok(LocalChatOutput {
            message: self.message,
            finish,
            usage: terminal.usage(),
            done_reason: terminal.done_reason,
            model: terminal.model,
        })
    }
}

fn decode<B, E, T>(
    mut chunks: impl Stream<Item = Result<B, E>> + Unpin + Send + 'static,
    mut assembly: Assembly,
    map: fn(LocalChatEvent) -> Result<T, Error>,
) -> EventStream<T>
where
    B: AsRef<[u8]> + Send + 'static,
    E: Send + 'static,
    T: Send + 'static,
{
    Box::pin(async_stream::try_stream! {
        let mut line = Vec::new();
        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.map_err(|_| Error::Transport)?;
            for part in chunk.as_ref().split_inclusive(|byte| *byte == b'\n') {
                if line.len() + part.len() > MAX_NDJSON_RECORD_BYTES {
                    Err(Error::Protocol("Ollama NDJSON record exceeds the size limit"))?;
                }
                line.extend_from_slice(part);
                if part.ends_with(b"\n") {
                    if let Some(text) = decode_line(&line, &mut assembly)? {
                        yield map(LocalChatEvent::Text(text))?;
                    }
                    line.clear();
                }
            }
        }
        if let Some(text) = decode_line(&line, &mut assembly)? {
            yield map(LocalChatEvent::Text(text))?;
        }
        yield map(LocalChatEvent::Finished(assembly.finish()?))?;
    })
}

fn decode_line(line: &[u8], assembly: &mut Assembly) -> Result<Option<String>, Error> {
    if line.iter().all(u8::is_ascii_whitespace) {
        return Ok(None);
    }
    let record = serde_json::from_slice(line)
        .map_err(|_| Error::Protocol("invalid Ollama NDJSON record"))?;
    let text = assembly.accept(record)?;
    Ok((!text.is_empty()).then_some(text))
}

#[cfg(test)]
#[path = "ollama_tests.rs"]
mod tests;

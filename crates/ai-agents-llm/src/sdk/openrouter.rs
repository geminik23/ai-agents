//! OpenRouter Chat Completions, with explicit routing and replay of opaque assistant history.
use super::stream::{Assembly, Choice};
use super::transport::{Transport, bearer_headers, frames};
use super::wire::{self, Envelope};
use super::{ChatRequest, Error, Event, ExpectedCalls, FunctionChoice, Message, ToolCall, Usage};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

pub const API_BASE: &str = "https://openrouter.ai/api/v1";

const INVALID_REPLAY: &str = "OpenRouter assistant replay does not match the original target";

/// Generation settings and assistant turns supplied by this service.
pub struct Request {
    /// Shared chat settings. The output ceiling is encoded as OpenRouter's max_tokens.
    pub chat: ChatRequest,
    /// Omit to use the gateway default, or explicitly permit or disable parallel tool calls.
    pub parallel_tool_calls: Option<bool>,
    replay: BTreeMap<usize, AssistantTurn>,
}

impl Request {
    pub fn new(chat: ChatRequest) -> Self {
        Self {
            chat,
            parallel_tool_calls: None,
            replay: BTreeMap::new(),
        }
    }

    /// Append a complete assistant turn before its matching tool results.
    pub fn push_assistant(&mut self, turn: &AssistantTurn) -> Result<(), Error> {
        let calls = turn.calls()?;
        let slot = self.chat.messages.len();
        self.chat.messages.push(Message::Assistant {
            text: text(&turn.message, "content")?.map(str::to_owned),
            calls,
        });
        self.replay.insert(slot, turn.clone());
        Ok(())
    }
}

/// Original assistant message and the target that produced it. May contain private reasoning.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssistantTurn {
    endpoint: String,
    model: Option<String>,
    provider: Option<String>,
    route: Option<Value>,
    message: Value,
}

impl AssistantTurn {
    pub fn has_reasoning(&self) -> bool {
        ["reasoning_details", "reasoning", "reasoning_content"]
            .iter()
            .any(|key| {
                self.message.get(*key).is_some_and(|v| match v {
                    Value::Array(a) => !a.is_empty(),
                    Value::String(s) => !s.is_empty(),
                    _ => !v.is_null(),
                })
            })
    }

    pub(crate) fn calls(&self) -> Result<Vec<ToolCall>, Error> {
        if self.message["role"] != "assistant" {
            return Err(Error::Protocol("invalid OpenRouter assistant history"));
        }
        assistant_calls(&self.message)
    }

    pub(crate) fn validate_replay(
        &self,
        endpoint: &str,
        model: &str,
        route: Option<&Value>,
    ) -> Result<(), Error> {
        if self.endpoint != endpoint
            || (self.has_reasoning()
                && (self.model.as_deref() != Some(model)
                    || self.provider.as_deref().is_none_or(str::is_empty)
                    || self.route.is_none()
                    || self.route.as_ref() != route))
        {
            return Err(Error::Protocol(INVALID_REPLAY));
        }
        Ok(())
    }
}

fn assistant_calls(message: &Value) -> Result<Vec<ToolCall>, Error> {
    let calls = message
        .get("tool_calls")
        .filter(|v| !v.is_null())
        .map(|v| serde_json::from_value::<Vec<wire::CallData>>(v.clone()))
        .transpose()
        .map_err(|_| Error::Protocol("invalid OpenRouter replay calls"))?;
    wire::finish_calls(calls.unwrap_or_default(), "stop")
}

pub struct Completion {
    pub id: Option<String>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub text: String,
    pub finish_reason: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<Usage>,
    /// Gateway-reported cost in credits, independent of host reservations.
    pub cost: Option<Value>,
    pub assistant: AssistantTurn,
}

pub enum StreamEvent {
    Text(String),
    Finished(Box<Completion>),
}

pub type ChatStream =
    std::pin::Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, Error>> + Send>>;

#[derive(Clone)]
pub struct OpenRouterClient {
    transport: Transport,
}

fn text<'a>(value: &'a Value, key: &str) -> Result<Option<&'a str>, Error> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        _ => Err(Error::Protocol(
            "invalid OpenRouter assistant text or reasoning",
        )),
    }
}

fn assistant(value: &Value) -> Result<Value, Error> {
    let mut turn = json!({"role":"assistant","content":text(value, "content")?});
    for key in [
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "tool_calls",
    ] {
        if let Some(field) = value.get(key).filter(|v| !v.is_null()) {
            match key {
                "reasoning_details"
                    if !field
                        .as_array()
                        .is_some_and(|a| a.iter().all(Value::is_object)) =>
                {
                    return Err(Error::Protocol("invalid OpenRouter reasoning blocks"));
                }
                "reasoning" | "reasoning_content" => {
                    text(value, key)?;
                }
                _ => {}
            }
            turn[key] = field.clone();
        }
    }
    Ok(turn)
}

pub(crate) fn replay_route(body: &Value) -> Option<Value> {
    // The wire response reports a provider name, not an authoritative routing selector.
    // Reuse the caller's explicit route only when it restricts the next request to that route.
    let provider = body.get("provider")?.as_object()?;
    let only = provider.get("only")?.as_array()?;
    if only.len() != 1
        || !only[0].as_str().is_some_and(|v| !v.trim().is_empty())
        || provider.get("allow_fallbacks") != Some(&Value::Bool(false))
        || body.get("models").is_some()
        || body.get("route").is_some()
        || body.get("preset").is_some()
    {
        return None;
    }
    let mut route = provider.clone();
    route.remove("require_parameters");
    Some(Value::Object(route))
}

pub(crate) fn require_routing_parameters(body: &mut Map<String, Value>) -> Result<(), Error> {
    let routing = body.entry("provider").or_insert_with(|| json!({}));
    let routing = routing
        .as_object_mut()
        .ok_or_else(|| Error::Config("OpenRouter provider must be an object".into()))?;
    if routing
        .get("require_parameters")
        .is_some_and(|v| v != &Value::Bool(true))
    {
        return Err(Error::Config(
            "required tool choice needs provider.require_parameters: true".into(),
        ));
    }
    routing.insert("require_parameters".into(), Value::Bool(true));
    Ok(())
}

impl OpenRouterClient {
    pub fn new(api_key: String) -> Result<Self, Error> {
        Self::at_endpoint(API_BASE, api_key)
    }

    /// Use explicitly supplied credentials for this API base.
    pub fn at_endpoint(base: &str, api_key: String) -> Result<Self, Error> {
        if api_key.trim().is_empty() {
            return Err(Error::Config(
                "OpenRouter requires a bearer credential".into(),
            ));
        }
        Ok(Self {
            transport: Transport::new(base, "chat/completions", bearer_headers(Some(api_key))?)?,
        })
    }

    pub(crate) fn endpoint(&self) -> &str {
        self.transport.endpoint.as_str()
    }

    pub(crate) fn body(&self, request: &Request, streaming: bool) -> Result<Value, Error> {
        let mut body = request.chat.body(streaming)?;
        let obj = body.as_object_mut().expect("chat body is an object");
        if let Some(limit) = obj.remove("max_completion_tokens") {
            obj.insert("max_tokens".into(), limit);
        }
        if let Some(parallel) = request.parallel_tool_calls {
            obj.insert("parallel_tool_calls".into(), Value::Bool(parallel));
        }
        // OpenRouter includes final usage without the Chat compatibility toggle.
        obj.remove("stream_options");
        if obj.get("reasoning").is_some_and(|reasoning| {
            !reasoning.is_object()
                || reasoning
                    .get("effort")
                    .is_some_and(|v| !v.as_str().is_some_and(|s| !s.trim().is_empty()))
                || obj.contains_key("reasoning_effort")
        }) {
            return Err(Error::Config(
                "invalid or conflicting OpenRouter reasoning option".into(),
            ));
        }
        if obj.get("provider").is_some_and(|v| !v.is_object()) {
            return Err(Error::Config(
                "OpenRouter provider must be an object".into(),
            ));
        }
        if matches!(
            request.chat.tool_choice,
            Some(FunctionChoice::Required | FunctionChoice::Named(_))
        ) {
            require_routing_parameters(obj)?;
        }
        let route = replay_route(&body);
        let mut replay_provider = None;
        for (&slot, turn) in &request.replay {
            turn.validate_replay(self.endpoint(), &request.chat.model, route.as_ref())?;
            if turn.has_reasoning() {
                if replay_provider.is_some() && replay_provider != turn.provider.as_deref() {
                    return Err(Error::Protocol(INVALID_REPLAY));
                }
                replay_provider = turn.provider.as_deref();
            }
            let message = body["messages"]
                .get_mut(slot)
                .ok_or(Error::Protocol(INVALID_REPLAY))?;
            if message["role"] != "assistant"
                || message["content"] != turn.message["content"]
                || assistant_calls(message)? != turn.calls()?
            {
                return Err(Error::Protocol(INVALID_REPLAY));
            }
            *message = turn.message.clone();
        }
        Ok(body)
    }

    pub async fn complete(&self, request: &Request) -> Result<Completion, Error> {
        let body = self.body(request, false)?;
        let payload: Value = self
            .transport
            .send(&body, request.chat.timeout)
            .await?
            .json()
            .await
            .map_err(|_| Error::Protocol("invalid OpenRouter completion JSON"))?;
        let message = assistant(&payload["choices"][0]["message"])?;
        let metadata: Metadata = serde_json::from_value(payload.clone())
            .map_err(|_| Error::Protocol("invalid OpenRouter completion metadata"))?;
        let decoded = wire::complete(
            serde_json::from_value(payload)
                .map_err(|_| Error::Protocol("invalid OpenRouter completion"))?,
        )?;
        ExpectedCalls::from_request(&request.chat).check(&decoded.tool_calls)?;
        check_provider(&request.replay, &metadata)?;
        Ok(metadata.output(
            self.endpoint().into(),
            replay_route(&body),
            message,
            decoded,
        ))
    }

    pub async fn stream(&self, request: &Request) -> Result<ChatStream, Error> {
        let body = self.body(request, true)?;
        let mut frames = frames(
            self.transport
                .send(&body, request.chat.timeout)
                .await?
                .bytes_stream(),
        );
        let expected = ExpectedCalls::from_request(&request.chat);
        let endpoint = self.endpoint().to_owned();
        let model = request.chat.model.clone();
        let route = replay_route(&body);
        let replay_provider = request
            .replay
            .values()
            .find(|turn| turn.has_reasoning())
            .and_then(|turn| turn.provider.clone());
        Ok(Box::pin(async_stream::try_stream! {
            let mut assembly = Assembly::default();
            let mut metadata = Metadata::default();
            let mut message = StreamMessage::default();
            while let Some(frame) = frames.next().await {
                let frame = frame?;
                if frame == "[DONE]" {
                    let calls = assembly.assistant_calls();
                    let Event::Finished { reason, tool_calls, usage } = assembly.finish()? else {
                        unreachable!("stream assembly finishes with a terminal event")
                    };
                    expected.check(&tool_calls)?;
                    if replay_provider.is_some() && (replay_provider != metadata.provider || metadata.model.as_deref() != Some(model.as_str())) {
                        Err(Error::Protocol(INVALID_REPLAY))?;
                    }
                    let text = message.content.clone();
                    yield StreamEvent::Finished(Box::new(metadata.output(endpoint, route, message.finish(calls)?, super::Completion { text, finish_reason: reason, tool_calls, usage })));
                    return;
                }
                let mut frame: StreamFrame = serde_json::from_str(&frame)
                    .map_err(|_| Error::Protocol("invalid OpenRouter stream event"))?;
                frame.metadata.merge_into(&mut metadata)?;
                // OpenRouter can repeat the finish reason on a content-free final usage choice.
                if let Some(reason) = assembly.finish_reason()
                    && frame.chat.error.is_none()
                    && frame.chat.usage.is_some()
                    && let [choice] = frame.chat.choices.as_slice()
                    && choice.repeats_finish_without_output(reason)
                {
                    frame.chat.choices.clear();
                }
                for choice in &frame.chat.choices {
                    message.accept(&choice.delta)?;
                }
                let cost = frame.chat.usage.as_mut().and_then(|usage| usage.cost.take());
                let event = assembly.accept(frame.chat)?;
                if assembly.finish_reason().is_some() && cost.is_some() {
                    metadata.cost = cost;
                }
                if let Some(Event::Text(text)) = event {
                    yield StreamEvent::Text(text);
                }
            }
            Err(Error::Protocol("OpenRouter stream disconnected before [DONE]"))?;
        }))
    }
}

fn check_provider(
    replay: &BTreeMap<usize, AssistantTurn>,
    metadata: &Metadata,
) -> Result<(), Error> {
    if replay.values().any(|turn| {
        turn.has_reasoning() && (turn.provider != metadata.provider || turn.model != metadata.model)
    }) {
        return Err(Error::Protocol(INVALID_REPLAY));
    }
    Ok(())
}

#[derive(Default, Deserialize)]
struct Metadata {
    id: Option<String>,
    model: Option<String>,
    provider: Option<String>,
    #[serde(default, rename = "usage")]
    receipt: Option<Cost>,
    #[serde(skip)]
    cost: Option<Value>,
}

#[derive(Deserialize)]
struct Cost {
    cost: Option<Value>,
}

#[derive(Deserialize)]
struct StreamFrame {
    #[serde(flatten)]
    chat: Envelope<Choice>,
    #[serde(flatten)]
    metadata: Metadata,
}

impl Metadata {
    fn merge_into(self, saved: &mut Self) -> Result<(), Error> {
        for (current, incoming) in [
            (&mut saved.id, self.id),
            (&mut saved.model, self.model),
            (&mut saved.provider, self.provider),
        ] {
            if let Some(incoming) = incoming.filter(|s| !s.is_empty()) {
                if current.as_ref().is_some_and(|v| v != &incoming) {
                    return Err(Error::Protocol("OpenRouter stream target changed"));
                }
                *current = Some(incoming);
            }
        }
        Ok(())
    }

    fn output(
        self,
        endpoint: String,
        route: Option<Value>,
        message: Value,
        decoded: super::Completion,
    ) -> Completion {
        Completion {
            id: self.id,
            model: self.model.clone(),
            provider: self.provider.clone(),
            cost: self.cost.or_else(|| self.receipt.and_then(|r| r.cost)),
            text: decoded.text,
            finish_reason: decoded.finish_reason,
            tool_calls: decoded.tool_calls,
            usage: decoded.usage,
            assistant: AssistantTurn {
                endpoint,
                model: self.model,
                provider: self.provider,
                route,
                message,
            },
        }
    }
}

#[derive(Default)]
struct StreamMessage {
    content: String,
    reasoning: Map<String, Value>,
    blocks: Vec<Value>,
}

impl StreamMessage {
    fn accept(&mut self, delta: &super::stream::Delta) -> Result<(), Error> {
        if let Some(content) = &delta.content {
            self.content.push_str(content);
        }
        for (key, part) in [
            ("reasoning", &delta.reasoning),
            ("reasoning_content", &delta.reasoning_content),
        ] {
            if let Some(part) = part {
                merge_fragment(&mut self.reasoning, key, part, true)?;
            }
        }
        if let Some(details) = &delta.reasoning_details {
            let details = details
                .as_array()
                .ok_or(Error::Protocol("invalid OpenRouter reasoning delta"))?;
            for part in details {
                let fields = part
                    .as_object()
                    .ok_or(Error::Protocol("invalid OpenRouter reasoning block"))?;
                let index = match fields.get("index") {
                    None => self.blocks.len(),
                    Some(index) => index
                        .as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or(Error::Protocol("invalid OpenRouter reasoning block index"))?,
                };
                if index == self.blocks.len() {
                    self.blocks.push(part.clone());
                } else {
                    let existing = self
                        .blocks
                        .get_mut(index)
                        .and_then(Value::as_object_mut)
                        .ok_or(Error::Protocol("missing OpenRouter reasoning block index"))?;
                    for (key, value) in fields {
                        merge_fragment(
                            existing,
                            key,
                            value,
                            matches!(key.as_str(), "text" | "summary" | "data" | "signature"),
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn finish(self, calls: Vec<Value>) -> Result<Value, Error> {
        let mut message = self.reasoning;
        message.insert("role".into(), json!("assistant"));
        message.insert(
            "content".into(),
            if self.content.is_empty() {
                Value::Null
            } else {
                json!(self.content)
            },
        );
        if !self.blocks.is_empty() {
            message.insert("reasoning_details".into(), json!(self.blocks));
        }
        if !calls.is_empty() {
            message.insert("tool_calls".into(), json!(calls));
        }
        assistant(&Value::Object(message))
    }
}

fn merge_fragment(
    fields: &mut Map<String, Value>,
    key: &str,
    value: &Value,
    append: bool,
) -> Result<(), Error> {
    // String payloads can be split across frames. Block identity fields must agree.
    match fields.entry(key.to_owned()) {
        serde_json::map::Entry::Vacant(entry) => {
            entry.insert(value.clone());
        }
        serde_json::map::Entry::Occupied(mut entry) => match (entry.get_mut(), value) {
            (Value::String(existing), Value::String(part)) if append => existing.push_str(part),
            (existing, incoming) if existing.is_null() => *existing = incoming.clone(),
            (existing, incoming) if incoming.is_null() || existing == incoming => {}
            _ => {
                return Err(Error::Protocol(
                    "OpenRouter reasoning block changed identity",
                ));
            }
        },
    }
    Ok(())
}

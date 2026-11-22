//! Stateless Responses requests. Callers own ordered input/output items and any tool execution.
use super::{Error, ExpectedCalls, FunctionChoice, ToolCall, transport};
use futures::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    pin::Pin,
    time::Duration,
};

/// Responses functions may opt into the protocol's strict JSON Schema subset.
pub struct ResponseFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub strict: Option<bool>,
}
impl ResponseFunction {
    fn encode(&self) -> Value {
        let mut tool = json!({"type":"function","name":self.name,"description":self.description,"parameters":self.parameters});
        if let Some(strict) = self.strict {
            tool["strict"] = json!(strict);
        }
        tool
    }
}

pub struct ResponsesRequest {
    pub model: String,
    /// Ordered protocol items, including output items retained by the caller for replay.
    pub input: Vec<Value>,
    pub instructions: Option<String>,
    pub reasoning_mode: Option<String>,
    pub reasoning_effort: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub functions: Vec<ResponseFunction>,
    pub tool_choice: Option<FunctionChoice>,
    pub timeout: Option<Duration>,
}

impl ResponsesRequest {
    pub fn new(model: impl Into<String>, input: Vec<Value>) -> Self {
        Self {
            model: model.into(),
            input,
            instructions: None,
            temperature: None,
            top_p: None,
            reasoning_mode: None,
            reasoning_effort: None,
            max_output_tokens: None,
            functions: vec![],
            tool_choice: None,
            timeout: None,
        }
    }

    fn expected(&self) -> ExpectedCalls {
        ExpectedCalls {
            names: self.functions.iter().map(|f| f.name.clone()).collect(),
            choice: self.tool_choice.clone(),
        }
    }

    pub(crate) fn body(&self, stream: bool) -> Result<Value, Error> {
        let mut body = json!({"model":self.model,"input":self.input,"stream":stream,"store":false});
        for (field, value) in [("temperature", self.temperature), ("top_p", self.top_p)] {
            if let Some(n) = value {
                body[field] = super::decimal(n)?;
            }
        }
        if let Some(s) = &self.instructions {
            body["instructions"] = json!(s);
        }
        if let Some(n) = self.max_output_tokens {
            body["max_output_tokens"] = json!(n);
        }
        for (key, option) in [
            ("mode", &self.reasoning_mode),
            ("effort", &self.reasoning_effort),
        ] {
            if let Some(value) = option {
                if value.trim().is_empty() {
                    return Err(Error::Config(format!("empty reasoning.{key}")));
                }
                body["reasoning"][key] = json!(value);
            }
        }
        if !self.functions.is_empty() {
            body["tools"] = Value::Array(
                self.functions
                    .iter()
                    .map(ResponseFunction::encode)
                    .collect(),
            );
        }
        if let Some(choice) = &self.tool_choice {
            body["tool_choice"] = match choice {
                FunctionChoice::Auto => json!("auto"),
                FunctionChoice::None => json!("none"),
                FunctionChoice::Required if !self.functions.is_empty() => json!("required"),
                FunctionChoice::Named(name) if self.functions.iter().any(|f| &f.name == name) => {
                    json!({"type":"function","name":name})
                }
                _ => {
                    return Err(Error::Config(
                        "tool choice has no matching definition".into(),
                    ));
                }
            };
        }
        Ok(body)
    }
}

pub struct ResponseOutput {
    pub id: String,
    pub text: String,
    /// Exact ordered items for trusted replay. May contain private encrypted reasoning.
    pub output: Vec<Value>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<ResponseUsage>,
}

/// Totals reported by Responses. Breakdowns can contain nested cache classes and are already included in the totals.
#[derive(Deserialize)]
pub struct ResponseUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    pub input_tokens_details: Option<serde_json::Map<String, Value>>,
    pub output_tokens_details: Option<serde_json::Map<String, Value>>,
}

fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str, Error> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or(Error::Protocol("missing Responses string field"))
}

pub(crate) fn function_call(item: &Value) -> Result<ToolCall, Error> {
    let id = string(item, "call_id")?;
    let name = string(item, "name")?;
    let arguments: Value = serde_json::from_str(string(item, "arguments")?)
        .map_err(|_| Error::Protocol("invalid Responses function arguments"))?;
    if id.is_empty() || name.is_empty() || !arguments.is_object() {
        return Err(Error::Protocol("invalid Responses function call"));
    }
    Ok(ToolCall {
        id: id.into(),
        name: name.into(),
        arguments,
    })
}

fn completed(mut value: Value, expected: &ExpectedCalls) -> Result<ResponseOutput, Error> {
    match string(&value, "status")? {
        "completed" if value["error"].is_null() => {}
        "incomplete" => return Err(Error::Protocol("incomplete Responses result")),
        "failed" => return Err(Error::Protocol("failed Responses result")),
        _ => return Err(Error::Protocol("unexpected Responses status")),
    }
    let id = string(&value, "id")?.to_owned();
    if id.is_empty() {
        return Err(Error::Protocol("empty response id"));
    }
    let Value::Array(output) = value["output"].take() else {
        return Err(Error::Protocol("missing Responses output"));
    };
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut ids = HashSet::new();
    let mut calls = HashSet::new();
    for item in &output {
        let item_id = string(item, "id")?;
        if item_id.is_empty()
            || !ids.insert(item_id)
            || item
                .get("status")
                .is_some_and(|s| s != "completed" && !s.is_null())
        {
            return Err(Error::Protocol("ambiguous or unfinished output item"));
        }
        match string(item, "type")? {
            "reasoning" => {}
            "function_call" => {
                let call = function_call(item)?;
                if !calls.insert(call.id.clone()) {
                    return Err(Error::Protocol("duplicate Responses call id"));
                }
                tool_calls.push(call);
            }
            "message" if item["role"] == "assistant" => {
                for part in item["content"]
                    .as_array()
                    .ok_or(Error::Protocol("missing message content"))?
                {
                    match string(part, "type")? {
                        "output_text" => text.push_str(string(part, "text")?),
                        "refusal" => return Err(Error::Refused),
                        _ => return Err(Error::Protocol("unsupported Responses content")),
                    }
                }
            }
            _ => return Err(Error::Protocol("unsupported Responses output item")),
        }
    }
    expected.check(&tool_calls)?;
    let usage = serde_json::from_value::<ResponseUsage>(value["usage"].take())
        .ok()
        .filter(|u| u.input_tokens.checked_add(u.output_tokens) == Some(u.total_tokens));
    Ok(ResponseOutput {
        id,
        text,
        output,
        tool_calls,
        usage,
    })
}

pub enum ResponseEvent {
    Text(String),
    Finished(ResponseOutput),
}
pub type ResponsesStream = Pin<Box<dyn Stream<Item = Result<ResponseEvent, Error>> + Send>>;

#[derive(Clone)]
pub struct ResponsesClient {
    transport: transport::Transport,
    official: bool,
}
impl ResponsesClient {
    pub fn new(base_url: &str, key: Option<String>) -> Result<Self, Error> {
        let transport = transport::Transport::new(base_url, key, "responses")?;
        let official =
            transport.endpoint.origin().ascii_serialization() == "https://api.openai.com";
        Ok(Self {
            transport,
            official,
        })
    }
    pub(crate) fn with_official_contract(mut self) -> Self {
        self.official = true;
        self
    }
    pub(crate) fn body(&self, r: &ResponsesRequest, stream: bool) -> Result<Value, Error> {
        if self.official {
            super::openai::validate_responses(r)?;
        }
        r.body(stream)
    }
    pub(crate) fn endpoint(&self) -> &str {
        self.transport.endpoint.as_str()
    }
    pub async fn complete(&self, r: &ResponsesRequest) -> Result<ResponseOutput, Error> {
        let body = self
            .transport
            .send(&self.body(r, false)?, r.timeout)
            .await?
            .json()
            .await
            .map_err(|_| Error::Protocol("invalid Responses JSON"))?;
        completed(body, &r.expected())
    }
    pub async fn stream(&self, r: &ResponsesRequest) -> Result<ResponsesStream, Error> {
        let response = self.transport.send(&self.body(r, true)?, r.timeout).await?;
        Ok(decode(
            transport::frames(response.bytes_stream()),
            r.expected(),
        ))
    }
}

struct PendingItem {
    id: String,
    kind: String,
    function: Option<(String, String)>,
    arguments: String,
    arguments_done: bool,
    done: bool,
    text: HashMap<usize, String>,
}

fn index(value: &Value, key: &str) -> Result<usize, Error> {
    value[key]
        .as_u64()
        .and_then(|v| v.try_into().ok())
        .ok_or(Error::Protocol("missing Responses event index"))
}

impl PendingItem {
    fn new(item: &Value) -> Result<Self, Error> {
        let kind = string(item, "type")?;
        let function = if kind == "function_call" {
            Some((
                string(item, "call_id")?.into(),
                string(item, "name")?.into(),
            ))
        } else {
            None
        };
        Ok(Self {
            id: string(item, "id")?.into(),
            kind: kind.into(),
            function,
            arguments: String::new(),
            arguments_done: false,
            done: false,
            text: HashMap::new(),
        })
    }

    fn verify(&self, raw: &Value) -> Result<(), Error> {
        if self.id != string(raw, "id")? || self.kind != string(raw, "type")? {
            return Err(Error::Protocol("streamed item disagrees with terminal"));
        }
        if let Some((id, name)) = &self.function
            && (id != string(raw, "call_id")?
                || name != string(raw, "name")?
                || self.arguments != string(raw, "arguments")?)
        {
            return Err(Error::Protocol("streamed function disagrees with terminal"));
        }
        if self.kind == "message" {
            let parts = raw["content"]
                .as_array()
                .ok_or(Error::Protocol("missing message content"))?;
            if self.text.keys().any(|k| *k >= parts.len()) {
                return Err(Error::Protocol("invalid content index"));
            }
            for (n, part) in parts.iter().enumerate() {
                if self.text.get(&n).map_or("", String::as_str) != string(part, "text")? {
                    return Err(Error::Protocol("stream text disagrees with terminal"));
                }
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct Assembly {
    id: Option<String>,
    items: HashMap<usize, PendingItem>,
    text: String,
    terminal: Option<ResponseOutput>,
}

impl Assembly {
    fn bind_id(&mut self, id: &str) -> Result<(), Error> {
        if self.id.as_deref().is_some_and(|s| s != id) {
            return Err(Error::Protocol("response id changed during stream"));
        }
        self.id.get_or_insert_with(|| id.into());
        Ok(())
    }

    fn accept(&mut self, event: Value, expected: &ExpectedCalls) -> Result<Option<String>, Error> {
        if self.terminal.is_some() {
            return Err(Error::Protocol("event after Responses terminal"));
        }
        if let Some(id) = event.get("response_id") {
            self.bind_id(id.as_str().ok_or(Error::Protocol("invalid response id"))?)?;
        }
        match string(&event, "type")? {
            "response.created" | "response.in_progress" => {
                self.bind_id(string(&event["response"], "id")?)?
            }
            "response.output_item.added" => {
                let item = PendingItem::new(&event["item"])?;
                if self
                    .items
                    .insert(index(&event, "output_index")?, item)
                    .is_some()
                {
                    return Err(Error::Protocol("duplicate output index"));
                }
            }
            kind @ ("response.output_text.delta"
            | "response.function_call_arguments.delta"
            | "response.function_call_arguments.done") => {
                let item = self
                    .items
                    .get_mut(&index(&event, "output_index")?)
                    .ok_or(Error::Protocol("unknown output index"))?;
                if item.done || item.id != string(&event, "item_id")? {
                    return Err(Error::Protocol("output item id changed"));
                }
                match kind {
                    "response.output_text.delta" if item.kind == "message" => {
                        let delta = string(&event, "delta")?;
                        item.text
                            .entry(index(&event, "content_index")?)
                            .or_default()
                            .push_str(delta);
                        self.text.push_str(delta);
                        return Ok(Some(delta.into()));
                    }
                    "response.function_call_arguments.delta"
                        if item.function.is_some() && !item.arguments_done =>
                    {
                        item.arguments.push_str(string(&event, "delta")?)
                    }
                    "response.function_call_arguments.done"
                        if item.function.is_some()
                            && !item.arguments_done
                            && item.arguments == string(&event, "arguments")? =>
                    {
                        item.arguments_done = true
                    }
                    _ => return Err(Error::Protocol("inconsistent Responses delta")),
                }
            }
            "response.output_item.done" => {
                let item = self
                    .items
                    .get_mut(&index(&event, "output_index")?)
                    .ok_or(Error::Protocol("unknown completed item"))?;
                if item.done {
                    return Err(Error::Protocol("duplicate completed output item"));
                }
                item.verify(&event["item"])?;
                item.done = true;
            }
            "response.completed" => {
                let mut event = event;
                let output = completed(event["response"].take(), expected)?;
                self.bind_id(&output.id)?;
                if self.items.len() != output.output.len() || self.text != output.text {
                    return Err(Error::Protocol("incomplete streamed output"));
                }
                for (i, raw) in output.output.iter().enumerate() {
                    self.items
                        .get(&i)
                        .ok_or(Error::Protocol("missing output index"))?
                        .verify(raw)?;
                }
                self.terminal = Some(output);
            }
            "response.incomplete" => return Err(Error::Protocol("incomplete Responses stream")),
            "response.failed" | "error" => return Err(Error::Protocol("failed Responses stream")),
            "response.refusal.delta" | "response.refusal.done" => return Err(Error::Refused),
            _ => {} // Ancillary notifications cannot publish executable calls or settle usage.
        }
        Ok(None)
    }
}

fn decode(
    mut frames: Pin<Box<dyn Stream<Item = Result<String, Error>> + Send>>,
    expected: ExpectedCalls,
) -> ResponsesStream {
    Box::pin(async_stream::try_stream! {
        let mut assembly=Assembly::default();
        while let Some(frame)=frames.next().await{
            let frame=frame?;
            if frame=="[DONE]" && assembly.terminal.is_some(){continue;}
            let event=serde_json::from_str(&frame).map_err(|_|Error::Protocol("invalid Responses event"))?;
            if let Some(text)=assembly.accept(event,&expected)?{
                yield ResponseEvent::Text(text);
            }
        }
        yield ResponseEvent::Finished(assembly.terminal.ok_or(Error::Protocol("Responses stream ended without completion"))?);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(index: usize, id: &str, name: &str, args: &str) -> Value {
        json!({"type":"function_call","id":format!("item_{index}"),"call_id":id,"name":name,"arguments":args})
    }

    fn events() -> Vec<Value> {
        let a = call(0, "call_a", "first", "{}");
        let b = call(1, "call_b", "second", "{\"v\":1}");
        let mut events = vec![json!({"type":"response.created","response":{"id":"resp"}})];
        for (i, raw) in [&a, &b].into_iter().enumerate() {
            let mut added = raw.clone();
            added["arguments"] = json!("");
            events.push(json!({"type":"response.output_item.added","output_index":i,"item":added}));
        }
        for (i, delta) in [(1, "{\"v\":"), (0, "{}"), (1, "1}")] {
            events.push(json!({"type":"response.function_call_arguments.delta","output_index":i,"item_id":format!("item_{i}"),"delta":delta}));
        }
        events.push(json!({"type":"response.completed","response":{"id":"resp","status":"completed","output":[a,b],"usage":{"input_tokens":3,"output_tokens":5,"total_tokens":8,"input_tokens_details":{"cache_write_tokens_details":{"ttl_30m_tokens":0}},"output_tokens_details":{"reasoning_tokens":4}}}}));
        events
    }

    fn stream_events(events: Vec<Value>) -> ResponsesStream {
        let frames = futures::stream::iter(events.into_iter().map(|v| Ok(v.to_string())));
        decode(
            Box::pin(frames),
            ExpectedCalls {
                names: vec!["first".into(), "second".into()],
                choice: None,
            },
        )
    }

    #[tokio::test]
    async fn interleaved_calls_and_usage_require_a_matching_completed_response() {
        let mut stream = stream_events(events());
        let ResponseEvent::Finished(output) = stream.next().await.unwrap().unwrap() else {
            panic!("missing output")
        };
        assert_eq!(output.tool_calls[1].id, "call_b");
        assert_eq!(output.tool_calls[1].arguments, json!({"v":1}));
        assert_eq!(output.usage.unwrap().total_tokens, 8);
        assert!(stream.next().await.is_none());
        for failure in [
            "response.incomplete",
            "response.failed",
            "error",
            "eof",
            "late",
            "arguments",
            "id",
            "name",
            "response_id",
        ] {
            let mut input = events();
            let mut terminal = input.pop().unwrap();
            match failure {
                "eof" => {}
                "late" => {
                    input.push(terminal.clone());
                    input.push(terminal);
                }
                "arguments" | "name" => {
                    terminal["response"]["output"][1][failure] = json!("changed");
                    input.push(terminal);
                }
                "id" => {
                    terminal["response"]["output"][1]["call_id"] = json!("moved");
                    input.push(terminal);
                }
                "response_id" => {
                    terminal["response"]["id"] = json!("another");
                    input.push(terminal);
                }
                kind => input.push(json!({"type":kind})),
            }
            assert!(
                stream_events(input).next().await.unwrap().is_err(),
                "{failure}"
            );
        }
        let mut partial = events();
        partial.last_mut().unwrap()["response"]["usage"]["total_tokens"] = Value::Null;
        let ResponseEvent::Finished(out) = stream_events(partial).next().await.unwrap().unwrap()
        else {
            panic!("missing output")
        };
        assert!(out.usage.is_none());
    }
}

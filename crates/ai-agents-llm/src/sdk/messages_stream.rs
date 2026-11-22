use super::{
    Error, ExpectedCalls, MessageEvent, MessagesStream, UsageUpdate, finish, string, validate_block,
};
use futures::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::Value;
use std::{collections::BTreeMap, pin::Pin};

struct Block {
    value: Value,
    arguments: String,
    stopped: bool,
}

impl Block {
    fn delta(&mut self, delta: &Value) -> Result<Option<String>, Error> {
        if self.stopped {
            return Err(Error::Protocol("delta after Messages content_block_stop"));
        }
        let kind = string(&self.value, "type")?;
        match (kind, string(delta, "type")?) {
            ("tool_use", "input_json_delta") => {
                if self.value["input"]
                    .as_object()
                    .is_none_or(|input| !input.is_empty())
                {
                    return Err(Error::Protocol("tool input and its deltas overlap"));
                }
                self.arguments.push_str(string(delta, "partial_json")?);
            }
            ("text", "text_delta") => {
                let text = string(delta, "text")?;
                self.append("text", text)?;
                return Ok(Some(text.to_owned()));
            }
            ("thinking", "thinking_delta") => {
                if self.value["signature"]
                    .as_str()
                    .is_some_and(|value| !value.is_empty())
                {
                    return Err(Error::Protocol("thinking changed after its signature"));
                }
                self.append("thinking", string(delta, "thinking")?)?;
            }
            ("thinking", "signature_delta") => {
                self.append("signature", string(delta, "signature")?)?
            }
            _ => return Err(Error::Protocol("unexpected Messages content delta")),
        }
        Ok(None)
    }

    fn append(&mut self, key: &str, delta: &str) -> Result<(), Error> {
        let mut text = match self.value[key].take() {
            Value::String(text) => text,
            Value::Null if key == "signature" => String::new(),
            _ => return Err(Error::Protocol("unexpected Messages block field")),
        };
        text.push_str(delta);
        self.value[key] = Value::String(text);
        Ok(())
    }

    fn stop(&mut self) -> Result<(), Error> {
        if self.stopped {
            return Err(Error::Protocol("duplicate Messages content_block_stop"));
        }
        if !self.arguments.is_empty() {
            self.value["input"] = serde_json::from_str(&self.arguments)
                .map_err(|_| Error::Protocol("invalid streamed Messages tool input"))?;
        }
        validate_block(&self.value)?;
        self.stopped = true;
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Frame {
    MessageStart {
        message: Value,
    },
    ContentBlockStart {
        index: usize,
        content_block: Value,
    },
    ContentBlockDelta {
        index: usize,
        delta: Value,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: Value,
        usage: Option<UsageUpdate>,
    },
    MessageStop,
    Error,
    #[serde(other)]
    Ancillary,
}

impl UsageUpdate {
    // message_delta reports cumulative totals and may omit input/cache fields.
    fn update(&mut self, incoming: Self) -> Result<(), Error> {
        for (previous, next) in [
            (&mut self.input_tokens, incoming.input_tokens),
            (
                &mut self.cache_read_input_tokens,
                incoming.cache_read_input_tokens,
            ),
            (
                &mut self.cache_creation_input_tokens,
                incoming.cache_creation_input_tokens,
            ),
            (&mut self.output_tokens, incoming.output_tokens),
        ] {
            if let Some(value) = next {
                if previous.is_some_and(|p| p > value) {
                    return Err(Error::Protocol("Messages cumulative usage decreased"));
                }
                *previous = Some(value);
            }
        }
        Ok(())
    }
}

fn block_mut(blocks: &mut BTreeMap<usize, Block>, index: usize) -> Result<&mut Block, Error> {
    blocks
        .get_mut(&index)
        .ok_or(Error::Protocol("unknown Messages block index"))
}

pub(super) fn decode(
    mut frames: Pin<Box<dyn Stream<Item = Result<String, Error>> + Send>>,
    expected: ExpectedCalls,
) -> MessagesStream {
    Box::pin(async_stream::try_stream! {
        let mut id = None;
        let mut blocks = BTreeMap::new();
        let mut usage = UsageUpdate::default();
        let mut reported_output = false;
        let mut stop_reason = None;
        let mut terminal = None;
        let mut last_text_index = None;

        while let Some(frame) = frames.next().await {
            let frame = serde_json::from_str::<Frame>(&frame?)
                .map_err(|_| Error::Protocol("invalid Messages stream event"))?;
            if matches!(frame, Frame::Ancillary) {
                continue;
            }
            if terminal.is_some() {
                Err(Error::Protocol("event after Messages message_stop"))?;
            }
            if id.is_none() && !matches!(frame, Frame::MessageStart { .. } | Frame::Error) {
                Err(Error::Protocol("streamed Messages event before message_start"))?;
            }
            let mut visible = None;
            match frame {
                Frame::MessageStart { message } => {
                    if id.is_some() || message["role"] != "assistant"
                        || message["content"].as_array().is_none_or(|blocks| !blocks.is_empty())
                    {
                        Err(Error::Protocol("invalid Messages message_start"))?;
                    }
                    id = Some(string(&message, "id")?.to_owned());
                    usage = serde_json::from_value(message["usage"].clone()).unwrap_or_default();
                }
                Frame::ContentBlockStart { index, content_block } => {
                    if stop_reason.is_some() || blocks.contains_key(&index) {
                        Err(Error::Protocol("unexpected Messages content_block_start"))?;
                    }
                    let text = (content_block["type"] == "text")
                        .then(|| string(&content_block, "text").map(str::to_owned)).transpose()?;
                    blocks.insert(index, Block { value: content_block, arguments: String::new(), stopped: false });
                    visible = text.map(|text| (index, text));
                }
                Frame::ContentBlockDelta { index, delta } => {
                    visible = block_mut(&mut blocks, index)?.delta(&delta)?.map(|text| (index, text));
                }
                Frame::ContentBlockStop { index } => block_mut(&mut blocks, index)?.stop()?,
                Frame::MessageDelta { delta, usage: update } => {
                    if blocks.values().any(|block| !block.stopped) {
                        Err(Error::Protocol("Messages message_delta arrived before blocks closed"))?;
                    }
                    if let Some(reason) = delta["stop_reason"].as_str() {
                        if stop_reason.as_deref().is_some_and(|previous| previous != reason) {
                            Err(Error::Protocol("Messages stop reason changed"))?;
                        }
                        stop_reason = Some(reason.to_owned());
                    }
                    if let Some(update) = update {
                        reported_output |= update.output_tokens.is_some();
                        usage.update(update)?;
                    }
                }
                Frame::MessageStop => {
                    if blocks.iter().enumerate().any(|(index, (key, block))| index != *key || !block.stopped) {
                        Err(Error::Protocol("incomplete Messages content blocks"))?;
                    }
                    let content = std::mem::take(&mut blocks).into_values().map(|block| block.value).collect();
                    let usage = reported_output.then(|| std::mem::take(&mut usage).finish()).flatten();
                    terminal = Some(finish(
                        id.take().ok_or(Error::Protocol("missing Messages message_start"))?,
                        content,
                        stop_reason.as_deref().ok_or(Error::Protocol("missing Messages stop reason"))?,
                        usage,
                        &expected,
                    )?);
                }
                Frame::Error => Err(Error::Protocol("Messages stream returned an error"))?,
                Frame::Ancillary => unreachable!(),
            }
            if let Some((index, text)) = visible.filter(|(_, text)| !text.is_empty()) {
                // Published text must agree with the ordered blocks returned at message_stop.
                if last_text_index.is_some_and(|last| index < last) {
                    Err(Error::Protocol("Messages text blocks are out of order"))?;
                }
                last_text_index = Some(index);
                yield MessageEvent::Text(text);
            }
        }
        yield MessageEvent::Finished(terminal.ok_or(Error::Protocol("Messages stream ended without message_stop"))?);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> Vec<Value> {
        vec![
            json!({"type":"message_start","message":{"id":"msg","role":"assistant","content":[],
                "usage":{"input_tokens":10,"cache_read_input_tokens":3,"cache_creation_input_tokens":2,"output_tokens":1}}}),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"fixture"}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"part-"}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"end"}}),
            json!({"type":"content_block_stop","index":0}),
            json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"id_a","name":"probe","input":{}}}),
            json!({"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"id_b","name":"probe","input":{}}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}),
            json!({"type":"ping"}),
            json!({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{}"}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"7}"}}),
            json!({"type":"content_block_stop","index":2}),
            json!({"type":"content_block_stop","index":1}),
            json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}),
            json!({"type":"message_delta","delta":{},"usage":{"output_tokens":7}}),
            json!({"type":"message_stop"}),
        ]
    }

    fn stream(events: Vec<Value>) -> MessagesStream {
        let frames = events.into_iter().map(|value| Ok(value.to_string()));
        decode(
            Box::pin(futures::stream::iter(frames)),
            ExpectedCalls {
                names: vec!["probe".into()],
                choice: None,
            },
        )
    }

    #[tokio::test]
    async fn messages_stream_reassembles_private_blocks_and_cumulative_totals() {
        let mut output = stream(fixture());
        let MessageEvent::Finished(output) = output.next().await.unwrap().unwrap() else {
            panic!("expected completed Messages tool calls")
        };
        assert_eq!(output.blocks[0]["signature"], "part-end");
        assert_eq!(output.tool_calls[0].id, "id_a");
        assert_eq!(output.tool_calls[0].arguments, json!({"a":7}));
        assert_eq!(output.tool_calls[1].id, "id_b");
        assert_eq!(output.usage.unwrap().totals(), Some((15, 7, 22)));
    }

    #[tokio::test]
    async fn messages_review_stream_text_must_agree_with_indexed_blocks() {
        let mut events = vec![fixture().remove(0)];
        for index in [0, 1] {
            events.push(json!({"type":"content_block_start","index":index,"content_block":{"type":"text","text":""}}));
        }
        for (index, text) in [(1, "second"), (0, "first")] {
            events.push(json!({"type":"content_block_delta","index":index,"delta":{"type":"text_delta","text":text}}));
            events.push(json!({"type":"content_block_stop","index":index}));
        }
        events.push(json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}));
        events.push(json!({"type":"message_stop"}));
        let mut output = stream(events);
        while let Some(event) = output.next().await {
            match event {
                Err(Error::Protocol(_)) => return,
                Ok(MessageEvent::Text(_)) => {}
                _ => panic!("inconsistent text stream must report a protocol error"),
            }
        }
        panic!("missing protocol error");
    }

    #[tokio::test]
    async fn messages_stream_failures_publish_no_tool_calls() {
        for failure in [
            "eof",
            "error",
            "args",
            "signature",
            "late",
            "refusal",
            "limit",
            "usage",
        ] {
            let mut events = fixture();
            match failure {
                "eof" => {
                    events.pop();
                }
                "error" => {
                    events.pop();
                    events.push(json!({"type":"error"}));
                }
                "args" => events[11]["delta"]["partial_json"] = json!("broken"),
                "signature" => {
                    events.drain(3..5);
                }
                "late" => events.push(json!({"type":"message_stop"})),
                "refusal" => events[14]["delta"]["stop_reason"] = json!("refusal"),
                "limit" => events[14]["delta"]["stop_reason"] = json!("max_tokens"),
                "usage" => events[15]["usage"]["output_tokens"] = json!(1),
                _ => unreachable!(),
            }
            assert!(stream(events).next().await.unwrap().is_err(), "{failure}");
        }
    }
}

use super::wire::{CallData, Envelope, UsageReceipt, finish_calls};
use super::{ChatStream, Error, Event, Usage};
use futures::StreamExt;
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
pub(super) struct Choice {
    #[serde(default)]
    index: usize,
    pub delta: Delta,
    finish_reason: Option<String>,
}

impl Choice {
    /// An accounting trailer may repeat the finish reason without modifying the assistant turn.
    pub fn repeats_finish_without_output(&self, reason: &str) -> bool {
        let delta = &self.delta;
        self.index == 0
            && self.finish_reason.as_deref() == Some(reason)
            && delta.role.as_deref().is_none_or(|role| role == "assistant")
            && delta.content.as_deref().is_none_or(str::is_empty)
            && delta.refusal.is_none()
            && delta.function_call.is_none()
            && delta.tool_calls.as_ref().is_none_or(Vec::is_empty)
            && delta
                .reasoning
                .as_ref()
                .is_none_or(|v| v.as_str() == Some(""))
            && delta
                .reasoning_content
                .as_ref()
                .is_none_or(|v| v.as_str() == Some(""))
            && delta
                .reasoning_details
                .as_ref()
                .is_none_or(|v| v.as_array().is_some_and(Vec::is_empty))
    }
}

#[derive(Deserialize)]
pub(super) struct Delta {
    role: Option<String>,
    function_call: Option<serde_json::Value>,
    pub content: Option<String>,
    pub reasoning_details: Option<serde_json::Value>,
    pub reasoning: Option<serde_json::Value>,
    pub reasoning_content: Option<serde_json::Value>,
    refusal: Option<String>,
    tool_calls: Option<Vec<CallDelta>>,
}

#[derive(Deserialize)]
struct CallDelta {
    index: usize,
    id: Option<String>,
    #[serde(rename = "type")]
    kind: Option<String>,
    function: Option<FunctionDelta>,
}

#[derive(Deserialize)]
struct FunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Default)]
pub(super) struct Assembly {
    reason: Option<String>,
    usage: Option<Usage>,
    terminal_receipt_seen: bool,
    calls: BTreeMap<usize, CallData>,
}

impl Assembly {
    pub fn finish(&mut self) -> Result<Event, Error> {
        let reason = self
            .reason
            .take()
            .ok_or(Error::Protocol("stream ended without a finish reason"))?;
        if self.calls.keys().copied().ne(0..self.calls.len()) {
            return Err(Error::Protocol("missing tool index"));
        }
        let calls = finish_calls(std::mem::take(&mut self.calls).into_values(), &reason)?;
        Ok(Event::Finished {
            reason,
            tool_calls: calls,
            usage: self.usage.take(),
        })
    }

    pub fn finish_reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }

    /// Keep the original streamed argument strings for callers that replay provider state.
    pub fn assistant_calls(&self) -> Vec<serde_json::Value> {
        self.calls
            .values()
            .map(|call| {
                serde_json::json!({
                    "id": call.id,
                    "type": "function",
                    "function": { "name": call.function.name, "arguments": call.function.arguments }
                })
            })
            .collect()
    }

    pub fn accept(&mut self, chunk: Envelope<Choice>) -> Result<Option<Event>, Error> {
        if chunk.error.is_some() {
            return Err(Error::Protocol("provider returned a stream error"));
        }
        let mut output = None;
        if !chunk.choices.is_empty() {
            let [choice]: [Choice; 1] = chunk
                .choices
                .try_into()
                .map_err(|_| Error::Protocol("expected one streamed choice"))?;
            if choice.index != 0 || self.reason.is_some() {
                return Err(Error::Protocol(
                    "unexpected choice after finish or invalid choice index",
                ));
            }
            if choice
                .delta
                .role
                .as_deref()
                .is_some_and(|role| role != "assistant")
                || choice.delta.function_call.is_some()
            {
                return Err(Error::Protocol(
                    "unexpected role or legacy function_call delta",
                ));
            }
            if choice.delta.refusal.is_some() {
                return Err(Error::Refused);
            }
            for part in choice.delta.tool_calls.unwrap_or_default() {
                let call = self.calls.entry(part.index).or_default();
                if let Some(id) = part.id.filter(|id| !id.is_empty()) {
                    if !call.id.is_empty() && call.id != id {
                        return Err(Error::Protocol("tool index changed call ID"));
                    }
                    call.id = id;
                }
                if part.kind.as_deref().is_some_and(|k| k != "function") {
                    return Err(Error::Protocol("unsupported tool call type"));
                }
                if let Some(function) = part.function {
                    if let Some(name) = function.name {
                        call.function.name.push_str(&name);
                    }
                    if let Some(arguments) = function.arguments {
                        call.function.arguments.push_str(&arguments);
                    }
                }
            }
            if let Some(text) = choice.delta.content.filter(|s| !s.is_empty()) {
                output = Some(Event::Text(text));
            }
            if let Some(reason) = choice.finish_reason {
                if matches!(reason.as_str(), "" | "error") {
                    return Err(Error::Protocol("failed or empty stream finish reason"));
                }
                self.reason = Some(reason);
            }
        }
        if let Some(receipt) = chunk.usage
            && self.reason.is_some()
        {
            if self.terminal_receipt_seen {
                return Err(Error::Protocol("duplicate final usage receipt"));
            }
            self.terminal_receipt_seen = true;
            self.usage = UsageReceipt::complete(receipt);
        }
        Ok(output)
    }
}

/// Pull from the HTTP body only as the consumer polls. CR/LF and UTF-8 may span network reads.
pub(super) fn decode<B, E>(
    bytes: impl futures::Stream<Item = Result<B, E>> + Unpin + Send + 'static,
    expected: super::ExpectedCalls,
) -> ChatStream
where
    B: AsRef<[u8]> + Send + 'static,
    E: Send + 'static,
{
    Box::pin(async_stream::try_stream! {
        let mut frames = super::transport::frames(bytes);
        let mut assembly = Assembly::default();
        while let Some(frame) = frames.next().await {
            let frame = frame?;
            if frame == "[DONE]" {
                let event = assembly.finish()?;
                if let Event::Finished { tool_calls, .. } = &event {
                    expected.check(tool_calls)?;
                }
                yield event;
                return;
            }
            let chunk = serde_json::from_str(&frame)
                .map_err(|_| Error::Protocol("invalid chat stream event"))?;
            if let Some(event) = assembly.accept(chunk)? {
                yield event;
            }
        }
        Err(Error::Protocol("chat stream disconnected before [DONE]"))?;
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunks(payload: &str) -> ChatStream {
        let bytes = payload
            .as_bytes()
            .iter()
            .map(|&b| Ok::<_, ()>([b]))
            .collect::<Vec<_>>();
        decode(
            futures::stream::iter(bytes),
            crate::sdk::ExpectedCalls {
                names: vec!["probe".into()],
                choice: None,
            },
        )
    }

    #[tokio::test]
    async fn split_utf8_crlf_comments_and_multiline_data_reassemble() {
        let mut events = chunks(
            ": ping\r\n\r\ndata: {\"choices\":[\r\ndata: {\"delta\":{\"content\":\"한\"},\"finish_reason\":\"stop\"}]}\r\n\r\ndata: [DONE]\r\n\r\n",
        );
        assert!(matches!(events.next().await,Some(Ok(Event::Text(s))) if s=="한"));
        assert!(matches!(
            events.next().await,
            Some(Ok(Event::Finished { usage: None, .. }))
        ));
        assert!(events.next().await.is_none());
    }

    #[tokio::test]
    async fn malformed_or_incomplete_events_never_publish_completed_calls() {
        let finish = r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        let bad_args = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x","function":{"name":"probe","arguments":"{"}}]},"finish_reason":"tool_calls"}]}"#;
        for payload in [
            "data: [DONE]\n\n".into(),
            format!("data: {finish}\n\n"),
            format!("data: {finish}\n\ndata: {finish}\n\ndata: [DONE]\n\n"),
            format!("data: {bad_args}\n\ndata: [DONE]\n\n"),
            "data: {\"error\":{\"message\":\"failed\"}}\n\n".into(),
            "data: {broken\n\n".into(),
        ] {
            let mut events = chunks(&payload);
            assert!(matches!(events.next().await, Some(Err(Error::Protocol(_)))));
            assert!(events.next().await.is_none());
        }
    }
}

#[cfg(test)]
mod usage_trailer_tests {
    use super::Choice;
    use serde_json::json;

    #[test]
    fn repeated_finish_trailer_cannot_carry_assistant_output() {
        for (delta, accepted) in [
            (json!({}), true),
            (
                json!({"role":"assistant","content":null,"tool_calls":[]}),
                true,
            ),
            (json!({"content":"late text"}), false),
            (json!({"tool_calls":[{"index":0}]}), false),
            (
                json!({"reasoning_details":[{"index":0,"data":"extra"}]}),
                false,
            ),
            (json!({"refusal":""}), false),
        ] {
            let choice: Choice = serde_json::from_value(json!({
                "index":0, "delta":delta, "finish_reason":"tool_calls"
            }))
            .unwrap();
            assert_eq!(choice.repeats_finish_without_output("tool_calls"), accepted);
            assert!(!choice.repeats_finish_without_output("stop"));
        }
    }
}

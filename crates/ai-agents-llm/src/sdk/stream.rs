use super::wire::{CallData, Envelope, UsageReceipt, finish_calls};
use super::{ChatStream, Error, Event, Usage};
use futures::StreamExt;
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Choice {
    #[serde(default)]
    index: usize,
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct Delta {
    role: Option<String>,
    function_call: Option<serde_json::Value>,
    content: Option<String>,
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
struct Assembly {
    reason: Option<String>,
    usage: Option<Usage>,
    terminal_receipt_seen: bool,
    calls: BTreeMap<usize, CallData>,
}

impl Assembly {
    fn accept(&mut self, data: &str) -> Result<Option<Event>, Error> {
        if data == "[DONE]" {
            let reason = self
                .reason
                .take()
                .ok_or(Error::Protocol("stream ended without a finish reason"))?;
            if self.calls.keys().copied().ne(0..self.calls.len()) {
                return Err(Error::Protocol("missing tool index"));
            }
            let calls = finish_calls(std::mem::take(&mut self.calls).into_values(), &reason)?;
            return Ok(Some(Event::Finished {
                reason,
                tool_calls: calls,
                usage: self.usage.take(),
            }));
        }
        let chunk: Envelope<Choice> =
            serde_json::from_str(data).map_err(|_| Error::Protocol("invalid chat stream event"))?;
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
                if reason.is_empty() {
                    return Err(Error::Protocol("empty stream finish reason"));
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
            if let Some(event) = assembly.accept(&frame?)? {
                if let Event::Finished { tool_calls, .. } = &event {
                    expected.check(tool_calls)?;
                    yield event;
                    return;
                }
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

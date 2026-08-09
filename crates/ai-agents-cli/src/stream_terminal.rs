use std::future::{Future, pending};

use ai_agents::{AgentStreamEvent, StreamChunk};
use futures::{Stream, StreamExt};

pub(crate) const INCOMPLETE_STREAM_ERROR: &str = "stream ended before Final";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamDriveControl {
    Continue,
    ConsumerClosed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamDriveOutcome {
    Final,
    TerminalError,
    IncompleteEof,
    ConsumerClosed,
}

// Drives one event stream so every first-party client shares the same terminal contract.
pub(crate) async fn drive_agent_stream<S, F>(stream: S, on_event: F) -> StreamDriveOutcome
where
    S: Stream<Item = AgentStreamEvent>,
    F: FnMut(AgentStreamEvent) -> StreamDriveControl,
{
    drive_agent_stream_until_closed(stream, on_event, pending()).await
}

// Drops a pending source stream as soon as its consumer is no longer available.
pub(crate) async fn drive_agent_stream_until_closed<S, F, C>(
    stream: S,
    mut on_event: F,
    consumer_closed: C,
) -> StreamDriveOutcome
where
    S: Stream<Item = AgentStreamEvent>,
    F: FnMut(AgentStreamEvent) -> StreamDriveControl,
    C: Future<Output = ()>,
{
    futures::pin_mut!(stream);
    futures::pin_mut!(consumer_closed);
    loop {
        let event = tokio::select! {
            biased;
            _ = &mut consumer_closed => return StreamDriveOutcome::ConsumerClosed,
            event = stream.next() => event,
        };
        let Some(event) = event else {
            return StreamDriveOutcome::IncompleteEof;
        };
        let terminal_outcome = match &event {
            AgentStreamEvent::Final(_) => Some(StreamDriveOutcome::Final),
            AgentStreamEvent::Chunk(StreamChunk::Error { .. }) => {
                Some(StreamDriveOutcome::TerminalError)
            }
            _ => None,
        };

        if on_event(event) == StreamDriveControl::ConsumerClosed {
            return StreamDriveOutcome::ConsumerClosed;
        }
        if let Some(outcome) = terminal_outcome {
            return outcome;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_agents::AgentResponse;
    use futures::stream;

    fn content(text: &str) -> AgentStreamEvent {
        AgentStreamEvent::Chunk(StreamChunk::Content {
            text: text.to_string(),
        })
    }

    #[tokio::test]
    async fn final_is_the_only_successful_terminal_outcome() {
        let events = stream::iter(vec![
            content("hello"),
            AgentStreamEvent::Final(AgentResponse::new("hello")),
            content("after"),
        ]);
        let mut forwarded = Vec::new();

        let outcome = drive_agent_stream(events, |event| {
            forwarded.push(event);
            StreamDriveControl::Continue
        })
        .await;

        assert_eq!(outcome, StreamDriveOutcome::Final);
        assert_eq!(forwarded.len(), 2);
        assert!(matches!(forwarded[0], AgentStreamEvent::Chunk(_)));
        assert!(matches!(forwarded[1], AgentStreamEvent::Final(_)));
    }

    #[tokio::test]
    async fn non_terminal_progress_waits_for_final() {
        let events = stream::iter(vec![
            AgentStreamEvent::Chunk(StreamChunk::ToolCallStart {
                id: "call-1".to_string(),
                name: "search".to_string(),
            }),
            AgentStreamEvent::Chunk(StreamChunk::StateTransition {
                from: Some("start".to_string()),
                to: "done".to_string(),
            }),
            content("answer"),
            AgentStreamEvent::Final(AgentResponse::new("answer")),
        ]);
        let mut forwarded = 0;

        let outcome = drive_agent_stream(events, |_| {
            forwarded += 1;
            StreamDriveControl::Continue
        })
        .await;

        assert_eq!(outcome, StreamDriveOutcome::Final);
        assert_eq!(forwarded, 4);
    }

    #[tokio::test]
    async fn error_is_forwarded_once_and_is_terminal() {
        let events = stream::iter(vec![
            content("before"),
            AgentStreamEvent::Chunk(StreamChunk::Error {
                message: "failed".to_string(),
            }),
            content("after"),
        ]);
        let mut forwarded = 0;

        let outcome = drive_agent_stream(events, |_| {
            forwarded += 1;
            StreamDriveControl::Continue
        })
        .await;

        assert_eq!(outcome, StreamDriveOutcome::TerminalError);
        assert_eq!(forwarded, 2);
    }

    #[tokio::test]
    async fn content_followed_by_eof_is_incomplete() {
        let events = stream::iter(vec![content("partial")]);

        let outcome = drive_agent_stream(events, |_| StreamDriveControl::Continue).await;

        assert_eq!(outcome, StreamDriveOutcome::IncompleteEof);
    }

    #[tokio::test]
    async fn empty_stream_is_incomplete() {
        let events = stream::iter(Vec::<AgentStreamEvent>::new());

        let outcome = drive_agent_stream(events, |_| StreamDriveControl::Continue).await;

        assert_eq!(outcome, StreamDriveOutcome::IncompleteEof);
    }

    #[tokio::test]
    async fn defensive_legacy_done_does_not_count_as_event_success() {
        let events = stream::iter(vec![AgentStreamEvent::Chunk(StreamChunk::Done {})]);
        let mut forwarded = 0;

        let outcome = drive_agent_stream(events, |_| {
            forwarded += 1;
            StreamDriveControl::Continue
        })
        .await;

        assert_eq!(outcome, StreamDriveOutcome::IncompleteEof);
        assert_eq!(forwarded, 1);
    }

    #[tokio::test]
    async fn consumer_closure_stops_before_later_events() {
        let events = stream::iter(vec![content("first"), content("second")]);
        let mut forwarded = 0;

        let outcome = drive_agent_stream(events, |_| {
            forwarded += 1;
            StreamDriveControl::ConsumerClosed
        })
        .await;

        assert_eq!(outcome, StreamDriveOutcome::ConsumerClosed);
        assert_eq!(forwarded, 1);
    }

    #[tokio::test]
    async fn failed_final_delivery_is_consumer_closure_not_success() {
        let events = stream::iter(vec![AgentStreamEvent::Final(AgentResponse::new("answer"))]);

        let outcome = drive_agent_stream(events, |_| StreamDriveControl::ConsumerClosed).await;

        assert_eq!(outcome, StreamDriveOutcome::ConsumerClosed);
    }

    #[tokio::test]
    async fn consumer_closure_interrupts_a_pending_source() {
        let events = stream::pending::<AgentStreamEvent>();

        let outcome =
            drive_agent_stream_until_closed(events, |_| StreamDriveControl::Continue, async {})
                .await;

        assert_eq!(outcome, StreamDriveOutcome::ConsumerClosed);
    }
}

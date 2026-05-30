use crate::context::SpanContext;
use crate::event::{EventStatus, EventType, ObservationError, ObservationTokenUsage};
use crate::manager::ObservabilityManager;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Stopwatch for one observed span that records on explicit finish or Drop.
pub struct SpanGuard {
    manager: Arc<ObservabilityManager>,
    context: SpanContext,
    event_type: EventType,
    start_time: Instant,
    tokens: Option<ObservationTokenUsage>,
    status: EventStatus,
    error: Option<ObservationError>,
    tags: HashMap<String, String>,
    payload: Option<Value>,
    recorded: bool,
}

impl SpanGuard {
    /// Creates a span guard with success status and an immediate start time.
    pub fn new(
        manager: Arc<ObservabilityManager>,
        context: SpanContext,
        event_type: EventType,
    ) -> Self {
        Self {
            manager,
            context,
            event_type,
            start_time: Instant::now(),
            tokens: None,
            status: EventStatus::Success,
            error: None,
            tags: HashMap::new(),
            payload: None,
            recorded: false,
        }
    }

    /// Attaches token usage collected from provider output or estimation.
    pub fn set_tokens(&mut self, tokens: ObservationTokenUsage) {
        self.tokens = Some(tokens);
    }

    /// Overrides the final status before recording.
    pub fn set_status(&mut self, status: EventStatus) {
        self.status = status;
    }

    /// Marks the span as failed and attaches error metadata.
    pub fn set_error(&mut self, error: ObservationError) {
        self.status = EventStatus::Error;
        self.error = Some(error);
    }

    /// Attaches optional payload data that will be redacted by the manager.
    pub fn set_payload(&mut self, payload: Value) {
        self.payload = Some(payload);
    }

    /// Adds a safe tag to the event before recording.
    pub fn add_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.tags.insert(key.into(), value.into());
    }

    /// Records the span immediately and prevents duplicate Drop recording.
    pub fn record_now(&mut self) {
        if self.recorded {
            return;
        }
        self.recorded = true;
        let event = self.manager.build_event_from_span(
            self.context.clone(),
            self.event_type.clone(),
            self.start_time.elapsed(),
            self.status.clone(),
            self.tokens.clone(),
            self.error.clone(),
            self.tags.clone(),
            self.payload.clone(),
        );
        let deferred = self
            .context
            .tags
            .get("runtime.defer_observation")
            .map(|value| value == "true")
            .unwrap_or(false);
        if deferred {
            if let Some(branch_id) = self.context.tags.get("runtime.branch_id") {
                self.manager.record_pending_event(branch_id.clone(), event);
                return;
            }
        }
        self.manager.record_event(event);
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        self.record_now();
    }
}

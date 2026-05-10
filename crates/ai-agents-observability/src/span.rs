use crate::context::SpanContext;
use crate::event::{EventStatus, EventType, ObservationError, ObservationTokenUsage};
use crate::manager::ObservabilityManager;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

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

    pub fn set_tokens(&mut self, tokens: ObservationTokenUsage) {
        self.tokens = Some(tokens);
    }

    pub fn set_status(&mut self, status: EventStatus) {
        self.status = status;
    }

    pub fn set_error(&mut self, error: ObservationError) {
        self.status = EventStatus::Error;
        self.error = Some(error);
    }

    pub fn set_payload(&mut self, payload: Value) {
        self.payload = Some(payload);
    }

    pub fn add_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.tags.insert(key.into(), value.into());
    }

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
        self.manager.record_event(event);
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        self.record_now();
    }
}

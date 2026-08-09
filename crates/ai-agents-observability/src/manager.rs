use crate::aggregator::{
    AggregatedMetrics, MetricsAggregator, aggregate_events, enrich_dimensions,
};
use crate::config::{AggregationDimension, ExportFormat, ObservabilityConfig, UnknownPricePolicy};
use crate::context::{SpanContext, current_observation_context};
use crate::cost::CostEstimator;
use crate::event::{
    CostEstimate, EventStatus, EventType, ObservationEvent, ObservationPurpose,
    ObservationTokenUsage,
};
use crate::export::{ExportResult, export_observability};
use crate::redaction::Redactor;
use crate::report::{ObservabilityReport, generate_report};
use crate::span::SpanGuard;
use crate::{ObservabilityError, Result};
use chrono::Utc;
use parking_lot::{Mutex, RwLock};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::mpsc;
use uuid::Uuid;

/// Opaque marker for scoping a later observation snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObservabilityCursor {
    ingested_events: u64,
    dropped_events: u64,
}

/// Retained post-cursor events and the report derived from that exact event set.
#[derive(Debug, Clone)]
pub struct ScopedObservationSnapshot {
    /// Redacted events still retained in the rolling metrics window.
    pub events: Vec<ObservationEvent>,
    /// Report derived only from the returned events.
    pub report: ObservabilityReport,
}

/// Central collector that receives events, applies privacy rules, aggregates metrics, and exports reports.
pub struct ObservabilityManager {
    config: ObservabilityConfig,
    sender: mpsc::Sender<ObservationEvent>,
    receiver: Mutex<mpsc::Receiver<ObservationEvent>>,
    raw_events: RwLock<VecDeque<ObservationEvent>>,
    pending_branch_events: RwLock<HashMap<String, Vec<ObservationEvent>>>,
    aggregator: MetricsAggregator,
    cost_estimator: CostEstimator,
    redactor: Redactor,
    dropped_events: AtomicU64,
}

impl ObservabilityManager {
    /// Creates a shared manager with bounded event buffering.
    pub fn new(config: ObservabilityConfig) -> Arc<Self> {
        let _ = config.validate();
        let (sender, receiver) = mpsc::channel(config.buffer.event_buffer.max(1));
        Arc::new(Self {
            cost_estimator: CostEstimator::new(config.cost.clone()),
            redactor: Redactor::new(config.privacy.clone()),
            aggregator: MetricsAggregator::new(config.aggregation.clone()),
            sender,
            receiver: Mutex::new(receiver),
            raw_events: RwLock::new(VecDeque::new()),
            pending_branch_events: RwLock::new(HashMap::new()),
            dropped_events: AtomicU64::new(0),
            config,
        })
    }

    /// Returns the immutable configuration used by this manager.
    pub fn config(&self) -> &ObservabilityConfig {
        &self.config
    }

    /// Starts a measured span for an LLM or tool wrapper.
    pub fn start_span(
        self: &Arc<Self>,
        event_type: EventType,
        purpose: ObservationPurpose,
    ) -> SpanGuard {
        let mut context = current_observation_context()
            .map(|ctx| ctx.child())
            .unwrap_or_else(|| SpanContext::new_root("unknown"));
        context.purpose = purpose;
        SpanGuard::new(Arc::clone(self), context, event_type)
    }

    /// Records hook-style lifecycle events that are not LLM or tool wrapper calls.
    pub fn record_lifecycle_event(
        &self,
        event_type: EventType,
        purpose: ObservationPurpose,
        status: EventStatus,
        duration_ms: u64,
        tags: HashMap<String, String>,
        payload: Option<Value>,
    ) {
        let context = current_observation_context()
            .map(|ctx| ctx.child())
            .unwrap_or_else(|| SpanContext::new_root("unknown"));
        let mut dimensions = context_dimension_map(&context);
        for (key, value) in &tags {
            if key.starts_with("runtime.") {
                dimensions.insert(key.clone(), value.clone());
                if let Some(short_key) = key.strip_prefix("runtime.") {
                    dimensions.insert(short_key.to_string(), value.clone());
                }
            }
        }
        let event = ObservationEvent {
            trace_id: context.trace_id,
            span_id: context.span_id,
            parent_span_id: context.parent_span_id,
            turn_id: context.turn_id,
            agent_id: context.agent_id,
            actor_id: context.actor_id,
            session_id: context.session_id,
            event_type,
            purpose,
            status,
            timestamp: Utc::now(),
            duration_ms,
            tokens: None,
            cost: None,
            error: None,
            dimensions,
            tags,
            payload,
        };
        self.record_event(event);
    }

    /// Records an event that should be finalized when its runtime branch resolves.
    pub fn record_pending_event(&self, branch_id: impl Into<String>, event: ObservationEvent) {
        if !self.config.enabled {
            return;
        }
        let mut pending = self.pending_branch_events.write();
        let pending_count: usize = pending.values().map(Vec::len).sum();
        if pending_count >= self.config.buffer.pending_branch_event_limit {
            self.dropped_events.fetch_add(1, Ordering::Relaxed);
            return;
        }
        pending.entry(branch_id.into()).or_default().push(event);
    }

    /// Finalizes all pending events for a runtime branch and ingests them normally.
    pub fn finalize_pending_branch(
        &self,
        branch_id: &str,
        branch_status: impl Into<String>,
        winner: bool,
        extra_tags: HashMap<String, String>,
    ) -> usize {
        let mut events = self
            .pending_branch_events
            .write()
            .remove(branch_id)
            .unwrap_or_default();
        let status = branch_status.into();
        let count = events.len();
        for event in &mut events {
            event
                .tags
                .insert("runtime.branch_status".to_string(), status.clone());
            event
                .tags
                .insert("runtime.winner".to_string(), winner.to_string());
            event.tags.insert("winner".to_string(), winner.to_string());
            event
                .dimensions
                .insert("branch_status".to_string(), status.clone());
            event
                .dimensions
                .insert("runtime.branch_status".to_string(), status.clone());
            event
                .dimensions
                .insert("runtime.winner".to_string(), winner.to_string());
            event
                .dimensions
                .insert("winner".to_string(), winner.to_string());
            for (key, value) in &extra_tags {
                event.tags.insert(key.clone(), value.clone());
                event.dimensions.insert(key.clone(), value.clone());
            }
            self.record_event(event.clone());
        }
        count
    }

    /// Queues a completed event without blocking the observed call path.
    pub fn record_event(&self, event: ObservationEvent) {
        if !self.config.enabled {
            return;
        }
        match self.sender.try_send(event) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(event)) => {
                if self.config.buffer.drop_on_full {
                    self.dropped_events.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.ingest_event(event);
                }
            }
            Err(mpsc::error::TrySendError::Closed(event)) => {
                self.ingest_event(event);
            }
        }
    }

    /// Drains pending queued events into aggregation and raw buffers.
    pub async fn flush(&self) -> Result<()> {
        self.drain_pending();
        Ok(())
    }

    /// Returns configured aggregate metrics after draining pending events.
    pub fn get_metrics(&self) -> Vec<AggregatedMetrics> {
        self.drain_pending();
        self.aggregator.aggregate_configured()
    }

    /// Captures a cursor after draining all events already queued for ingestion.
    pub fn event_cursor(&self) -> ObservabilityCursor {
        self.drain_pending();
        ObservabilityCursor {
            ingested_events: self.aggregator.cursor(),
            dropped_events: self.dropped_events(),
        }
    }

    /// Returns retained raw events after redaction and queue draining.
    pub fn raw_events(&self) -> Vec<ObservationEvent> {
        self.drain_pending();
        self.raw_events.read().iter().cloned().collect()
    }

    /// Drains once and snapshots retained events after the cursor with a report from that exact event set.
    /// Rolling-window eviction is preserved, while dropped events are reported as the counter delta since the cursor.
    pub fn snapshot_since(&self, cursor: ObservabilityCursor) -> ScopedObservationSnapshot {
        self.drain_pending();
        let events = self.aggregator.events_since(cursor.ingested_events);
        let dropped_events = self.dropped_events().saturating_sub(cursor.dropped_events);
        let report = generate_report(
            &events,
            aggregate_events(&events, &self.config.aggregation.dimensions),
            dropped_events,
        );
        ScopedObservationSnapshot { events, report }
    }

    /// Builds the user-facing report from the current rolling event window.
    pub fn generate_report(&self) -> ObservabilityReport {
        self.drain_pending();
        let events = self.aggregator.events();
        generate_report(
            &events,
            aggregate_events(&events, &self.config.aggregation.dimensions),
            self.dropped_events(),
        )
    }

    /// Writes configured report, aggregate, raw event, and Prometheus files.
    pub async fn export(&self) -> Result<ExportResult> {
        export_observability(self).map_err(ObservabilityError::Io)
    }

    /// Returns the total number of events dropped by bounded buffers.
    pub fn dropped_events(&self) -> u64 {
        self.dropped_events.load(Ordering::Relaxed)
    }

    /// Returns the redactor used by wrappers for safe payload summaries.
    pub fn redactor(&self) -> &Redactor {
        &self.redactor
    }

    /// Converts a completed SpanGuard into an ObservationEvent.
    //
    // Keep this public signature stable for custom span integrations.
    //
    #[allow(clippy::too_many_arguments)]
    pub fn build_event_from_span(
        &self,
        context: SpanContext,
        event_type: EventType,
        duration: Duration,
        status: EventStatus,
        tokens: Option<crate::event::ObservationTokenUsage>,
        error: Option<crate::event::ObservationError>,
        tags: HashMap<String, String>,
        payload: Option<Value>,
    ) -> ObservationEvent {
        let dimensions = context_dimension_map(&context);
        ObservationEvent {
            trace_id: context.trace_id,
            span_id: context.span_id,
            parent_span_id: context.parent_span_id,
            turn_id: context.turn_id,
            agent_id: context.agent_id,
            actor_id: context.actor_id,
            session_id: context.session_id,
            event_type,
            purpose: context.purpose,
            status,
            timestamp: Utc::now(),
            duration_ms: duration.as_millis() as u64,
            tokens,
            cost: None::<CostEstimate>,
            error,
            dimensions,
            tags,
            payload,
        }
    }

    /// Drains queued events into the synchronous aggregation path.
    fn drain_pending(&self) {
        let mut receiver = self.receiver.lock();
        while let Ok(event) = receiver.try_recv() {
            self.ingest_event(event);
        }
    }

    /// Enriches, costs, redacts, aggregates, and optionally stores one event.
    fn ingest_event(&self, mut event: ObservationEvent) {
        enrich_dimensions(&mut event);
        event.tokens = event
            .tokens
            .take()
            .map(|tokens| self.apply_token_config(tokens));
        if event.cost.is_none() {
            let (provider, model) = match &event.event_type {
                EventType::LlmCall {
                    provider, model, ..
                } => (Some(provider.as_str()), Some(model.as_str())),
                _ => (None, None),
            };
            event.cost = self
                .cost_estimator
                .estimate(provider, model, event.tokens.as_ref());
            if matches!(
                self.config.cost.unknown_price_policy,
                UnknownPricePolicy::Error
            ) && event.tokens.is_some()
                && event.cost.is_none()
                && matches!(&event.event_type, EventType::LlmCall { .. })
            {
                event
                    .tags
                    .insert("cost_error".to_string(), "unknown_price".to_string());
            }
        }
        let event = self.redactor.redact_event(event);
        self.aggregator.record(event.clone());
        self.store_raw_event(event);
    }

    /// Applies token count switches before reports and cost estimates read usage.
    fn apply_token_config(&self, mut tokens: ObservationTokenUsage) -> ObservationTokenUsage {
        if !self.config.tokens.count_input {
            tokens.input_tokens = 0;
        }
        if !self.config.tokens.count_output {
            tokens.output_tokens = 0;
        }
        tokens.total_tokens = tokens.input_tokens + tokens.output_tokens;
        tokens
    }

    /// Retains a redacted raw event when raw event export is enabled.
    fn store_raw_event(&self, event: ObservationEvent) {
        if !self.config.export.write_raw_events {
            return;
        }
        if self.config.buffer.raw_event_limit == 0 {
            self.dropped_events.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let mut raw_events = self.raw_events.write();
        if raw_events.len() >= self.config.buffer.raw_event_limit {
            if self.config.buffer.drop_on_full {
                self.dropped_events.fetch_add(1, Ordering::Relaxed);
                return;
            }
            raw_events.pop_front();
        }
        raw_events.push_back(event);
    }

    /// Renders current aggregate metrics in Prometheus text exposition format.
    pub fn render_prometheus(&self) -> String {
        let report = self.generate_report();
        let events = self.aggregator.events();
        let llm_events: Vec<_> = events
            .iter()
            .filter(|event| matches!(&event.event_type, EventType::LlmCall { .. }))
            .cloned()
            .collect();
        let tool_events: Vec<_> = events
            .iter()
            .filter(|event| matches!(&event.event_type, EventType::ToolCall { .. }))
            .cloned()
            .collect();
        let by_model_purpose = aggregate_events(
            &llm_events,
            &[AggregationDimension::Model, AggregationDimension::Purpose],
        );
        let by_tool = aggregate_events(&tool_events, &[AggregationDimension::Tool]);
        let mut output = String::new();
        output.push_str(
            "# HELP ai_agents_observation_events_total Total recorded observation events\n",
        );
        output.push_str("# TYPE ai_agents_observation_events_total counter\n");
        output.push_str(&format!(
            "ai_agents_observation_events_total {}\n",
            report.summary.total_events
        ));
        output.push_str("# HELP ai_agents_observation_errors_total Total observation events with error status\n");
        output.push_str("# TYPE ai_agents_observation_errors_total counter\n");
        output.push_str(&format!(
            "ai_agents_observation_errors_total {}\n",
            report.summary.total_errors
        ));
        output.push_str(
            "# HELP ai_agents_observation_cost_usd_total Estimated total LLM cost in USD\n",
        );
        output.push_str("# TYPE ai_agents_observation_cost_usd_total counter\n");
        output.push_str(&format!(
            "ai_agents_observation_cost_usd_total {:.8}\n",
            report.summary.total_cost_usd
        ));
        output.push_str("# HELP ai_agents_observation_tokens_total Total observed LLM tokens\n");
        output.push_str("# TYPE ai_agents_observation_tokens_total counter\n");
        output.push_str(&format!(
            "ai_agents_observation_tokens_total {}\n",
            report.summary.total_tokens
        ));
        output.push_str("# HELP ai_agents_llm_calls_total LLM calls grouped by safe labels\n");
        output.push_str("# TYPE ai_agents_llm_calls_total counter\n");
        for metric in by_model_purpose {
            let model = metric
                .dimensions
                .get("model")
                .map(String::as_str)
                .unwrap_or("unknown");
            let purpose = metric
                .dimensions
                .get("purpose")
                .map(String::as_str)
                .unwrap_or("unknown");
            output.push_str(&format!(
                "ai_agents_llm_calls_total{{model=\"{}\",purpose=\"{}\"}} {}\n",
                prometheus_label(model),
                prometheus_label(purpose),
                metric.count
            ));
        }
        output.push_str("# HELP ai_agents_tool_calls_total Tool calls grouped by tool ID\n");
        output.push_str("# TYPE ai_agents_tool_calls_total counter\n");
        for metric in by_tool {
            let tool = metric
                .dimensions
                .get("tool")
                .map(String::as_str)
                .unwrap_or("unknown");
            if tool != "unknown" {
                output.push_str(&format!(
                    "ai_agents_tool_calls_total{{tool=\"{}\"}} {}\n",
                    prometheus_label(tool),
                    metric.count
                ));
            }
        }
        output
    }

    /// Returns true when a format is enabled in export.formats.
    pub fn wants_format(&self, format: ExportFormat) -> bool {
        self.config.export.formats.contains(&format)
    }
}

/// Escapes label values for Prometheus text output.
fn prometheus_label(value: &str) -> String {
    value
        .chars()
        .flat_map(|ch| match ch {
            '\\' => "\\\\".chars().collect::<Vec<_>>(),
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\n' | '\r' | '\t' => "_".chars().collect::<Vec<_>>(),
            _ => vec![ch],
        })
        .collect()
}

/// Builds the base event dimensions from the current span context.
fn context_dimension_map(context: &SpanContext) -> HashMap<String, String> {
    let mut dimensions = HashMap::new();
    dimensions.insert("agent".to_string(), context.agent_id.clone());
    dimensions.insert("purpose".to_string(), context.purpose.as_label());
    if let Some(actor) = &context.actor_id {
        dimensions.insert("actor".to_string(), actor.clone());
    }
    if let Some(state) = &context.state {
        dimensions.insert("state".to_string(), state.clone());
    }
    if let Some(language) = &context.language {
        dimensions.insert("language".to_string(), language.clone());
    }
    dimensions.extend(context.tags.clone());
    dimensions
}

/// Resolves the language dimension by checking configured context paths in order.
pub fn resolve_language_from_context(
    config: &ObservabilityConfig,
    context: &HashMap<String, Value>,
) -> String {
    for path in &config.language.paths {
        if let Some(value) = get_dotted(context, path)
            && let Some(language) = value.as_str()
            && !language.trim().is_empty()
        {
            return language.to_string();
        }
    }
    config.language.fallback.clone()
}

/// Looks up a top-level or dotted path in a JSON context map.
fn get_dotted<'a>(context: &'a HashMap<String, Value>, path: &str) -> Option<&'a Value> {
    if let Some(value) = context.get(path) {
        return Some(value);
    }
    let mut parts = path.split('.');
    let first = parts.next()?;
    let mut current = context.get(first)?;
    for part in parts {
        current = current.get(part)?;
    }
    Some(current)
}

/// Generates a session ID for observed runtime sessions that do not have one yet.
pub fn new_session_id() -> String {
    Uuid::new_v4().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{ObservationTokenUsage, TokenUsageSource};

    fn test_event() -> ObservationEvent {
        ObservationEvent {
            trace_id: "trace".to_string(),
            span_id: Uuid::new_v4().to_string(),
            parent_span_id: None,
            turn_id: "turn".to_string(),
            agent_id: "agent".to_string(),
            actor_id: None,
            session_id: None,
            event_type: EventType::LlmCall {
                provider: "openai".to_string(),
                model: "test".to_string(),
                alias: Some("default".to_string()),
                streaming: false,
            },
            purpose: ObservationPurpose::MainResponse,
            status: EventStatus::Success,
            timestamp: Utc::now(),
            duration_ms: 10,
            tokens: Some(ObservationTokenUsage::new(
                100,
                25,
                TokenUsageSource::Provider,
            )),
            cost: None,
            error: None,
            dimensions: HashMap::new(),
            tags: HashMap::new(),
            payload: None,
        }
    }

    #[test]
    fn token_count_flags_are_applied_before_report() {
        let config = ObservabilityConfig {
            enabled: true,
            tokens: crate::config::TokenConfig {
                count_input: false,
                count_output: true,
                ..Default::default()
            },
            cost: crate::config::CostConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let manager = ObservabilityManager::new(config);
        manager.record_event(test_event());

        let report = manager.generate_report();
        assert_eq!(report.token_breakdown.total_input, 0);
        assert_eq!(report.token_breakdown.total_output, 25);
        assert_eq!(report.token_breakdown.total_tokens, 25);
    }

    #[test]
    fn pending_branch_event_is_hidden_until_finalized() {
        let config = ObservabilityConfig {
            enabled: true,
            export: crate::config::ExportConfig {
                write_raw_events: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let manager = ObservabilityManager::new(config);
        manager.record_pending_event("branch", test_event());

        let mut tags = HashMap::new();
        tags.insert("runtime.speculative".to_string(), "true".to_string());
        tags.insert("speculative".to_string(), "true".to_string());

        assert_eq!(manager.generate_report().summary.total_events, 0);
        manager.finalize_pending_branch("branch", "discarded", false, tags);
        let report = manager.generate_report();
        assert_eq!(report.summary.total_events, 1);
        assert_eq!(
            manager.raw_events()[0].dimensions.get("branch_status"),
            Some(&"discarded".to_string())
        );
        assert_eq!(
            manager.raw_events()[0].dimensions.get("runtime.winner"),
            Some(&"false".to_string())
        );
        assert_eq!(
            manager.raw_events()[0].dimensions.get("speculative"),
            Some(&"true".to_string())
        );
        assert_eq!(
            manager.raw_events()[0]
                .dimensions
                .get("runtime.speculative"),
            Some(&"true".to_string())
        );
    }

    #[test]
    fn pending_branch_events_are_bounded() {
        let config = ObservabilityConfig {
            enabled: true,
            buffer: crate::config::BufferConfig {
                pending_branch_event_limit: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let manager = ObservabilityManager::new(config);
        manager.record_pending_event("branch-a", test_event());
        manager.record_pending_event("branch-b", test_event());

        manager.finalize_pending_branch("branch-a", "committed", true, HashMap::new());
        manager.finalize_pending_branch("branch-b", "committed", true, HashMap::new());
        let report = manager.generate_report();
        assert_eq!(report.summary.total_events, 1);
        assert_eq!(report.dropped_events, 1);
    }

    #[test]
    fn snapshot_since_scopes_report_and_events_to_cursor() {
        let config = ObservabilityConfig {
            enabled: true,
            aggregation: crate::config::AggregationConfig {
                dimensions: vec![AggregationDimension::Purpose],
                window_size: 4,
                ..Default::default()
            },
            ..Default::default()
        };
        let manager = ObservabilityManager::new(config);
        let mut first = test_event();
        first.turn_id = "turn-1".to_string();
        manager.record_event(first);
        let cursor = manager.event_cursor();

        let mut second = test_event();
        second.turn_id = "turn-2".to_string();
        manager.record_event(second);

        let snapshot = manager.snapshot_since(cursor);
        assert_eq!(snapshot.events.len(), 1);
        assert_eq!(snapshot.events[0].turn_id, "turn-2");
        assert_eq!(
            snapshot.report.summary.total_events,
            snapshot.events.len() as u64
        );
        assert_eq!(
            snapshot
                .report
                .configured
                .iter()
                .map(|metric| metric.count)
                .sum::<u64>(),
            snapshot.events.len() as u64
        );
        assert_eq!(manager.generate_report().summary.total_events, 2);
    }

    #[test]
    fn snapshot_since_preserves_rolling_window_eviction() {
        let config = ObservabilityConfig {
            enabled: true,
            aggregation: crate::config::AggregationConfig {
                window_size: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        let manager = ObservabilityManager::new(config);
        let cursor = manager.event_cursor();
        for turn_id in ["turn-1", "turn-2", "turn-3"] {
            let mut event = test_event();
            event.turn_id = turn_id.to_string();
            manager.record_event(event);
        }

        let snapshot = manager.snapshot_since(cursor);
        assert_eq!(snapshot.events.len(), 2);
        assert_eq!(snapshot.events[0].turn_id, "turn-2");
        assert_eq!(snapshot.events[1].turn_id, "turn-3");
        assert_eq!(snapshot.report.summary.total_events, 2);
    }

    #[test]
    fn snapshot_since_exposes_only_the_dropped_event_delta() {
        let config = ObservabilityConfig {
            enabled: true,
            buffer: crate::config::BufferConfig {
                event_buffer: 1,
                drop_on_full: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let manager = ObservabilityManager::new(config);
        manager.record_event(test_event());
        manager.record_event(test_event());
        let cursor = manager.event_cursor();

        manager.record_event(test_event());
        manager.record_event(test_event());

        let snapshot = manager.snapshot_since(cursor);
        assert_eq!(snapshot.report.summary.total_events, 1);
        assert_eq!(snapshot.events.len(), 1);
        assert_eq!(snapshot.report.dropped_events, 1);
        assert_eq!(manager.generate_report().dropped_events, 2);
    }
}

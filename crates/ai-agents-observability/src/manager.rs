use crate::aggregator::{AggregatedMetrics, MetricsAggregator, enrich_dimensions};
use crate::config::{ExportFormat, ObservabilityConfig};
use crate::context::{SpanContext, current_observation_context};
use crate::cost::CostEstimator;
use crate::event::{CostEstimate, EventStatus, EventType, ObservationEvent, ObservationPurpose};
use crate::export::{ExportResult, export_observability};
use crate::redaction::Redactor;
use crate::report::{ObservabilityReport, generate_report};
use crate::span::SpanGuard;
use crate::{ObservabilityError, Result};
use chrono::Utc;
use parking_lot::RwLock;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use uuid::Uuid;

pub struct ObservabilityManager {
    config: ObservabilityConfig,
    raw_events: RwLock<VecDeque<ObservationEvent>>,
    aggregator: MetricsAggregator,
    cost_estimator: CostEstimator,
    redactor: Redactor,
    dropped_events: AtomicU64,
}

impl ObservabilityManager {
    pub fn new(config: ObservabilityConfig) -> Arc<Self> {
        let _ = config.validate();
        Arc::new(Self {
            cost_estimator: CostEstimator::new(config.cost.clone()),
            redactor: Redactor::new(config.privacy.clone()),
            aggregator: MetricsAggregator::new(config.aggregation.clone()),
            raw_events: RwLock::new(VecDeque::new()),
            dropped_events: AtomicU64::new(0),
            config,
        })
    }

    pub fn config(&self) -> &ObservabilityConfig {
        &self.config
    }

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
        let dimensions = context_dimension_map(&context);
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

    pub fn record_event(&self, mut event: ObservationEvent) {
        if !self.config.enabled {
            return;
        }
        enrich_dimensions(&mut event);
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
        }
        if let Some(payload) = &event.payload {
            event.payload = Some(self.redactor.redact_value(payload));
        }

        self.aggregator.record(event.clone());
        self.store_raw_event(event);
    }

    pub async fn flush(&self) -> Result<()> {
        Ok(())
    }

    pub fn get_metrics(&self) -> Vec<AggregatedMetrics> {
        self.aggregator.aggregate_configured()
    }

    pub fn raw_events(&self) -> Vec<ObservationEvent> {
        self.raw_events.read().iter().cloned().collect()
    }

    pub fn generate_report(&self) -> ObservabilityReport {
        let events = self.aggregator.events();
        generate_report(
            &events,
            self.aggregator.aggregate_configured(),
            self.dropped_events(),
        )
    }

    pub async fn export(&self) -> Result<ExportResult> {
        export_observability(self).map_err(ObservabilityError::Io)
    }

    pub fn dropped_events(&self) -> u64 {
        self.dropped_events.load(Ordering::Relaxed)
    }

    pub fn redactor(&self) -> &Redactor {
        &self.redactor
    }

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

    fn store_raw_event(&self, event: ObservationEvent) {
        if !self.config.export.write_raw_events {
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

    pub fn render_prometheus(&self) -> String {
        let report = self.generate_report();
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
        output
    }

    pub fn wants_format(&self, format: ExportFormat) -> bool {
        self.config.export.formats.contains(&format)
    }
}

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

pub fn resolve_language_from_context(
    config: &ObservabilityConfig,
    context: &HashMap<String, Value>,
) -> String {
    for path in &config.language.paths {
        if let Some(value) = get_dotted(context, path) {
            if let Some(language) = value.as_str() {
                if !language.trim().is_empty() {
                    return language.to_string();
                }
            }
        }
    }
    config.language.fallback.clone()
}

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

pub fn new_session_id() -> String {
    Uuid::new_v4().to_string()
}

use crate::aggregator::{AggregatedMetrics, aggregate_events, latency_stats};
use crate::config::AggregationDimension;
use crate::event::{EventStatus, EventType, ObservationEvent, TokenUsageSource};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub total_usd: f64,
    pub priced_events: u64,
    pub unpriced_events: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenBreakdown {
    pub total_input: u64,
    pub total_output: u64,
    pub total_tokens: u64,
    pub estimated_events: u64,
    pub missing_events: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityReport {
    pub summary: ReportSummary,
    pub configured: Vec<AggregatedMetrics>,
    pub by_model: Vec<AggregatedMetrics>,
    pub by_purpose: Vec<AggregatedMetrics>,
    pub by_language: Vec<AggregatedMetrics>,
    pub by_state: Vec<AggregatedMetrics>,
    pub by_agent: Vec<AggregatedMetrics>,
    pub by_orchestration_pattern: Vec<AggregatedMetrics>,
    pub cost_breakdown: CostBreakdown,
    pub token_breakdown: TokenBreakdown,
    pub dropped_events: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReportSummary {
    pub total_events: u64,
    pub total_llm_calls: u64,
    pub total_tool_calls: u64,
    pub total_errors: u64,
    pub total_tokens: u64,
    pub total_cost_usd: f64,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: u64,
    pub p90_latency_ms: u64,
    pub p99_latency_ms: u64,
    pub router_calls: u64,
    pub router_cost_usd: f64,
    pub main_llm_calls: u64,
    pub main_llm_cost_usd: f64,
    pub unpriced_events: u64,
    pub estimated_token_events: u64,
}

pub fn generate_report(
    events: &[ObservationEvent],
    configured: Vec<AggregatedMetrics>,
    dropped_events: u64,
) -> ObservabilityReport {
    let latency_values: Vec<u64> = events.iter().map(|event| event.duration_ms).collect();
    let latency = latency_stats(&latency_values);
    let mut summary = ReportSummary {
        total_events: events.len() as u64,
        avg_latency_ms: latency.avg_ms,
        p50_latency_ms: latency.p50_ms,
        p90_latency_ms: latency.p90_ms,
        p99_latency_ms: latency.p99_ms,
        ..ReportSummary::default()
    };
    let mut token_breakdown = TokenBreakdown::default();
    let mut cost_breakdown = CostBreakdown::default();

    for event in events {
        if matches!(event.status, EventStatus::Error) {
            summary.total_errors += 1;
        }
        match event.event_type {
            EventType::LlmCall { .. } => summary.total_llm_calls += 1,
            EventType::ToolCall { .. } => summary.total_tool_calls += 1,
            _ => {}
        }
        if let Some(tokens) = &event.tokens {
            token_breakdown.total_input += tokens.input_tokens;
            token_breakdown.total_output += tokens.output_tokens;
            token_breakdown.total_tokens += tokens.total_tokens;
            if matches!(tokens.source, TokenUsageSource::Estimated) {
                token_breakdown.estimated_events += 1;
                summary.estimated_token_events += 1;
            }
        } else {
            token_breakdown.missing_events += 1;
        }
        if let Some(cost) = &event.cost {
            cost_breakdown.total_usd += cost.total_usd;
            cost_breakdown.priced_events += 1;
            if event.purpose.as_label() == "router" {
                summary.router_cost_usd += cost.total_usd;
            }
            if event.purpose.as_label() == "main_response" {
                summary.main_llm_cost_usd += cost.total_usd;
            }
        } else {
            cost_breakdown.unpriced_events += 1;
            summary.unpriced_events += 1;
        }
        if event.purpose.as_label() == "router" {
            summary.router_calls += 1;
        }
        if event.purpose.as_label() == "main_response" {
            summary.main_llm_calls += 1;
        }
    }

    summary.total_tokens = token_breakdown.total_tokens;
    summary.total_cost_usd = cost_breakdown.total_usd;

    ObservabilityReport {
        summary,
        configured,
        by_model: aggregate_events(events, &[AggregationDimension::Model]),
        by_purpose: aggregate_events(events, &[AggregationDimension::Purpose]),
        by_language: aggregate_events(events, &[AggregationDimension::Language]),
        by_state: aggregate_events(events, &[AggregationDimension::State]),
        by_agent: aggregate_events(events, &[AggregationDimension::Agent]),
        by_orchestration_pattern: aggregate_events(
            events,
            &[AggregationDimension::OrchestrationPattern],
        ),
        cost_breakdown,
        token_breakdown,
        dropped_events,
    }
}

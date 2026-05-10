use crate::config::{AggregationConfig, AggregationDimension};
use crate::event::{EventStatus, EventType, ObservationEvent, TokenUsageSource};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    pub dimensions: HashMap<String, String>,
    pub count: u64,
    pub errors: u64,
    pub latency: LatencyStats,
    pub tokens: TokenStats,
    pub cost: CostStats,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencyStats {
    pub min_ms: u64,
    pub max_ms: u64,
    pub avg_ms: f64,
    pub p50_ms: u64,
    pub p90_ms: u64,
    pub p95_ms: u64,
    pub p99_ms: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenStats {
    pub total_input: u64,
    pub total_output: u64,
    pub total_tokens: u64,
    pub avg_input: f64,
    pub avg_output: f64,
    pub estimated_events: u64,
    pub missing_events: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostStats {
    pub total_usd: f64,
    pub avg_usd: f64,
    pub priced_events: u64,
    pub unpriced_events: u64,
}

#[derive(Debug)]
pub struct MetricsAggregator {
    config: AggregationConfig,
    events: RwLock<VecDeque<ObservationEvent>>,
}

impl MetricsAggregator {
    pub fn new(config: AggregationConfig) -> Self {
        Self {
            config,
            events: RwLock::new(VecDeque::new()),
        }
    }

    pub fn record(&self, event: ObservationEvent) {
        let mut events = self.events.write();
        events.push_back(event);
        while events.len() > self.config.window_size {
            events.pop_front();
        }
    }

    pub fn events(&self) -> Vec<ObservationEvent> {
        self.events.read().iter().cloned().collect()
    }

    pub fn aggregate_configured(&self) -> Vec<AggregatedMetrics> {
        self.aggregate_by(&self.config.dimensions)
    }

    pub fn aggregate_by(&self, dimensions: &[AggregationDimension]) -> Vec<AggregatedMetrics> {
        let events = self.events();
        aggregate_events(&events, dimensions)
    }
}

pub fn aggregate_events(
    events: &[ObservationEvent],
    dimensions: &[AggregationDimension],
) -> Vec<AggregatedMetrics> {
    let mut groups: HashMap<String, (HashMap<String, String>, Vec<&ObservationEvent>)> =
        HashMap::new();

    for event in events {
        let mut dim_values = HashMap::new();
        for dimension in dimensions {
            let key = dimension.key();
            let value = dimension_value(event, dimension).unwrap_or_else(|| "unknown".to_string());
            dim_values.insert(key, value);
        }
        let group_key = stable_group_key(&dim_values);
        groups
            .entry(group_key)
            .or_insert_with(|| (dim_values, Vec::new()))
            .1
            .push(event);
    }

    let mut metrics: Vec<AggregatedMetrics> = groups
        .into_values()
        .map(|(dimensions, group_events)| build_metrics(dimensions, &group_events))
        .collect();
    metrics.sort_by(|a, b| stable_group_key(&a.dimensions).cmp(&stable_group_key(&b.dimensions)));
    metrics
}

fn build_metrics(
    dimensions: HashMap<String, String>,
    events: &[&ObservationEvent],
) -> AggregatedMetrics {
    let count = events.len() as u64;
    let errors = events
        .iter()
        .filter(|event| event.status == EventStatus::Error)
        .count() as u64;
    let latencies: Vec<u64> = events.iter().map(|event| event.duration_ms).collect();
    let latency = latency_stats(&latencies);

    let mut token_stats = TokenStats::default();
    let mut cost_stats = CostStats::default();

    for event in events {
        if let Some(tokens) = &event.tokens {
            token_stats.total_input += tokens.input_tokens;
            token_stats.total_output += tokens.output_tokens;
            token_stats.total_tokens += tokens.total_tokens;
            if matches!(tokens.source, TokenUsageSource::Estimated) {
                token_stats.estimated_events += 1;
            }
            if matches!(tokens.source, TokenUsageSource::Missing) {
                token_stats.missing_events += 1;
            }
        } else {
            token_stats.missing_events += 1;
        }

        if let Some(cost) = &event.cost {
            cost_stats.total_usd += cost.total_usd;
            cost_stats.priced_events += 1;
        } else {
            cost_stats.unpriced_events += 1;
        }
    }

    if count > 0 {
        token_stats.avg_input = token_stats.total_input as f64 / count as f64;
        token_stats.avg_output = token_stats.total_output as f64 / count as f64;
        cost_stats.avg_usd = cost_stats.total_usd / count as f64;
    }

    AggregatedMetrics {
        dimensions,
        count,
        errors,
        latency,
        tokens: token_stats,
        cost: cost_stats,
    }
}

pub fn latency_stats(values: &[u64]) -> LatencyStats {
    if values.is_empty() {
        return LatencyStats::default();
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let sum: u64 = sorted.iter().sum();
    LatencyStats {
        min_ms: *sorted.first().unwrap_or(&0),
        max_ms: *sorted.last().unwrap_or(&0),
        avg_ms: sum as f64 / sorted.len() as f64,
        p50_ms: percentile(&sorted, 0.5),
        p90_ms: percentile(&sorted, 0.9),
        p95_ms: percentile(&sorted, 0.95),
        p99_ms: percentile(&sorted, 0.99),
    }
}

pub fn percentile(sorted_values: &[u64], percentile: f64) -> u64 {
    if sorted_values.is_empty() {
        return 0;
    }
    if percentile <= 0.0 {
        return sorted_values[0];
    }
    if percentile >= 1.0 {
        return *sorted_values.last().unwrap();
    }
    let rank = (percentile * sorted_values.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted_values.len() - 1);
    sorted_values[index]
}

pub fn dimension_value(
    event: &ObservationEvent,
    dimension: &AggregationDimension,
) -> Option<String> {
    match dimension {
        AggregationDimension::Agent => Some(event.agent_id.clone()),
        AggregationDimension::Actor => event.actor_id.clone(),
        AggregationDimension::Model => event.dimensions.get("model").cloned(),
        AggregationDimension::Provider => event.dimensions.get("provider").cloned(),
        AggregationDimension::Alias => event.dimensions.get("alias").cloned(),
        AggregationDimension::Purpose => Some(event.purpose.as_label()),
        AggregationDimension::Language => event.dimensions.get("language").cloned(),
        AggregationDimension::State => event.dimensions.get("state").cloned(),
        AggregationDimension::Tool => event.dimensions.get("tool").cloned(),
        AggregationDimension::Skill => event.dimensions.get("skill").cloned(),
        AggregationDimension::OrchestrationPattern => {
            event.dimensions.get("orchestration_pattern").cloned()
        }
        AggregationDimension::Status => Some(format!("{:?}", event.status).to_lowercase()),
        AggregationDimension::Custom(name) => {
            event.dimensions.get(&format!("custom:{}", name)).cloned()
        }
    }
}

pub fn enrich_dimensions(event: &mut ObservationEvent) {
    event
        .dimensions
        .entry("agent".to_string())
        .or_insert_with(|| event.agent_id.clone());
    if let Some(actor) = &event.actor_id {
        event
            .dimensions
            .entry("actor".to_string())
            .or_insert_with(|| actor.clone());
    }
    event
        .dimensions
        .entry("purpose".to_string())
        .or_insert_with(|| event.purpose.as_label());
    event
        .dimensions
        .entry("status".to_string())
        .or_insert_with(|| format!("{:?}", event.status).to_lowercase());

    match &event.event_type {
        EventType::LlmCall {
            provider,
            model,
            alias,
            ..
        } => {
            event
                .dimensions
                .entry("provider".to_string())
                .or_insert_with(|| provider.clone());
            event
                .dimensions
                .entry("model".to_string())
                .or_insert_with(|| model.clone());
            if let Some(alias) = alias {
                event
                    .dimensions
                    .entry("alias".to_string())
                    .or_insert_with(|| alias.clone());
            }
        }
        EventType::ToolCall { tool_id } => {
            event
                .dimensions
                .entry("tool".to_string())
                .or_insert_with(|| tool_id.clone());
        }
        EventType::SkillExecution { skill_id } | EventType::SkillStep { skill_id, .. } => {
            event
                .dimensions
                .entry("skill".to_string())
                .or_insert_with(|| skill_id.clone());
        }
        EventType::Orchestration { pattern } => {
            event
                .dimensions
                .entry("orchestration_pattern".to_string())
                .or_insert_with(|| pattern.clone());
        }
        _ => {}
    }
}

fn stable_group_key(dimensions: &HashMap<String, String>) -> String {
    let mut pairs: Vec<_> = dimensions.iter().collect();
    pairs.sort_by(|a, b| a.0.cmp(b.0));
    pairs
        .into_iter()
        .map(|(key, value)| format!("{}={}", key, value))
        .collect::<Vec<_>>()
        .join("|")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_uses_nearest_rank() {
        let values = vec![10, 20, 30, 40, 50];
        assert_eq!(percentile(&values, 0.5), 30);
        assert_eq!(percentile(&values, 0.9), 50);
    }
}

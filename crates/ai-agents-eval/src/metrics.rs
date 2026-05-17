use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::suite::{FailureCategory, ScenarioResult, ScenarioStatus};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalMetrics {
    pub pass_rate: f64,
    pub total_turns: usize,
    pub errors: usize,
    pub flaky: usize,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: u64,
    pub p90_latency_ms: u64,
    pub p99_latency_ms: u64,
    pub by_tag: HashMap<String, CountMetrics>,
    pub by_language: HashMap<String, CountMetrics>,
    pub by_assertion: HashMap<String, AssertionMetrics>,
    pub by_failure_category: HashMap<String, usize>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CountMetrics {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AssertionMetrics {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
}

pub fn compute_metrics(results: &[ScenarioResult]) -> EvalMetrics {
    let total = results.len();
    let passed = results.iter().filter(|r| r.status.is_passed()).count();
    let mut latencies = Vec::new();
    let mut metrics = EvalMetrics {
        pass_rate: if total == 0 {
            0.0
        } else {
            passed as f64 / total as f64
        },
        errors: results.iter().filter(|r| r.status.is_error()).count(),
        flaky: results.iter().filter(|r| r.flaky).count(),
        ..Default::default()
    };

    for result in results {
        for tag in &result.tags {
            update_count(
                metrics.by_tag.entry(tag.clone()).or_default(),
                &result.status,
            );
        }
        if let Some(language) = &result.language {
            update_count(
                metrics.by_language.entry(language.clone()).or_default(),
                &result.status,
            );
        }
        if let Some(category) = &result.failure_category {
            *metrics
                .by_failure_category
                .entry(category_key(category))
                .or_default() += 1;
        }
        for attempt in &result.attempts {
            for turn in &attempt.turns {
                latencies.push(turn.latency_ms);
                for assertion in &turn.assertion_results {
                    let entry = metrics
                        .by_assertion
                        .entry(assertion.assertion.clone())
                        .or_default();
                    entry.total += 1;
                    if assertion.passed {
                        entry.passed += 1;
                    } else {
                        entry.failed += 1;
                    }
                }
            }
        }
    }

    metrics.total_turns = latencies.len();
    if !latencies.is_empty() {
        latencies.sort_unstable();
        metrics.avg_latency_ms = latencies.iter().sum::<u64>() as f64 / latencies.len() as f64;
        metrics.p50_latency_ms = percentile(&latencies, 0.50);
        metrics.p90_latency_ms = percentile(&latencies, 0.90);
        metrics.p99_latency_ms = percentile(&latencies, 0.99);
    }

    metrics
}

fn update_count(metrics: &mut CountMetrics, status: &ScenarioStatus) {
    metrics.total += 1;
    match status {
        ScenarioStatus::Passed => metrics.passed += 1,
        ScenarioStatus::Failed { .. } | ScenarioStatus::Error { .. } => metrics.failed += 1,
        ScenarioStatus::Skipped { .. } => metrics.skipped += 1,
    }
}

fn percentile(values: &[u64], percentile: f64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let idx = ((values.len() - 1) as f64 * percentile).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn category_key(category: &FailureCategory) -> String {
    match category {
        FailureCategory::ConfigError => "config_error",
        FailureCategory::RuntimeError => "runtime_error",
        FailureCategory::AssertionFailed => "assertion_failed",
        FailureCategory::JudgeError => "judge_error",
        FailureCategory::FlakyPass => "flaky_pass",
    }
    .to_string()
}

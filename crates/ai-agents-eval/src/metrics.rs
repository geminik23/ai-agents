use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::suite::{FailureCategory, ScenarioResult, ScenarioStatus};

/// Aggregate metrics computed after an eval suite run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalMetrics {
    /// Fraction of scenarios that passed.
    pub pass_rate: f64,
    /// Total evaluated turns across all attempts.
    pub total_turns: usize,
    /// Number of scenarios ending in error.
    pub errors: usize,
    /// Number of scenarios that passed after retry.
    pub flaky: usize,
    /// Average turn latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Median turn latency in milliseconds.
    pub p50_latency_ms: u64,
    /// P90 turn latency in milliseconds.
    pub p90_latency_ms: u64,
    /// P99 turn latency in milliseconds.
    pub p99_latency_ms: u64,
    /// Scenario counts grouped by tag.
    pub by_tag: HashMap<String, CountMetrics>,
    /// Scenario counts grouped by language.
    pub by_language: HashMap<String, CountMetrics>,
    /// Assertion counts grouped by assertion name.
    pub by_assertion: HashMap<String, AssertionMetrics>,
    /// Scenario counts grouped by failure category.
    pub by_failure_category: HashMap<String, usize>,
}

/// Pass, fail, and skip counts for one grouping key.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CountMetrics {
    /// Total count for this result or group.
    pub total: usize,
    /// Passed count or boolean result.
    pub passed: usize,
    /// Failed or errored count for this result or group.
    pub failed: usize,
    /// Skipped count for this result or group.
    pub skipped: usize,
}

/// Aggregate pass and fail counts for one assertion name.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AssertionMetrics {
    /// Total count for this result or group.
    pub total: usize,
    /// Passed count or boolean result.
    pub passed: usize,
    /// Failed or errored count for this result or group.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suite::{AttemptResult, ScenarioResult, TurnResult};
    use crate::{redaction::RedactedString, suite::ScenarioStatus};

    fn scenario(id: &str, status: ScenarioStatus, latency: u64) -> ScenarioResult {
        ScenarioResult {
            id: id.to_string(),
            name: None,
            tags: vec!["smoke".to_string()],
            language: Some("en".to_string()),
            status,
            failure_category: None,
            flaky: false,
            attempts: vec![AttemptResult {
                attempt: 0,
                turns: vec![TurnResult {
                    index: 0,
                    input: RedactedString::redacted("[redacted]"),
                    response: RedactedString::redacted("[redacted]"),
                    response_present: true,
                    runtime_error: None,
                    state: None,
                    metadata: None,
                    evidence: crate::evidence::TurnEvidence {
                        response_metadata: None,
                        state: None,
                        state_history: Vec::new(),
                        context: serde_json::Value::Null,
                        tool_executions: Vec::new(),
                        approvals: Vec::new(),
                        llm_requests: Vec::new(),
                        skill: None,
                        disambiguation: None,
                        facts: None,
                        relationship: None,
                        persona: None,
                        orchestration: None,
                        observability: None,
                    },
                    assertion_results: vec![crate::assertion::AssertionResultDetail {
                        assertion: "response_not_empty".to_string(),
                        passed: true,
                        actual: serde_json::json!(true),
                        expected: serde_json::json!(true),
                        message: None,
                    }],
                    latency_ms: latency,
                    observability_span_id: None,
                }],
                status: ScenarioStatus::Passed,
                duration_ms: latency,
            }],
            duration_ms: latency,
            retries_used: 0,
        }
    }

    #[test]
    fn retained_errored_turns_count_toward_latency() {
        let mut result = scenario(
            "error",
            ScenarioStatus::Error {
                message: "[redacted]".to_string(),
            },
            25,
        );
        result.attempts[0].turns[0].response_present = false;
        result.attempts[0].turns[0].runtime_error = Some(RedactedString::redacted("[redacted]"));
        let metrics = compute_metrics(&[result]);
        assert_eq!(metrics.errors, 1);
        assert_eq!(metrics.total_turns, 1);
        assert_eq!(metrics.avg_latency_ms, 25.0);
    }

    #[test]
    fn computes_counts_and_latency() {
        let results = vec![
            scenario("pass", ScenarioStatus::Passed, 10),
            scenario(
                "fail",
                ScenarioStatus::Failed {
                    reason: "assertion".to_string(),
                },
                30,
            ),
        ];
        let metrics = compute_metrics(&results);
        assert_eq!(metrics.pass_rate, 0.5);
        assert_eq!(metrics.total_turns, 2);
        assert_eq!(metrics.avg_latency_ms, 20.0);
        assert_eq!(metrics.by_tag["smoke"].total, 2);
        assert_eq!(metrics.by_language["en"].total, 2);
        assert_eq!(metrics.by_assertion["response_not_empty"].passed, 2);
    }
}

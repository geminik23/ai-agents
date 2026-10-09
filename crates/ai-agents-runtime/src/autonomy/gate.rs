//! Pure tri-state completion checks; missing authority cannot become success through negation.

use super::evidence::*;
use ai_agents_core::autonomy::{CompletionGate, PathAssertion};
use ai_agents_core::{AgentError, Result};
use ai_agents_tools::TodoStatus;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateOutcome {
    Pass,
    Fail,
    Unknown,
}

impl GateOutcome {
    /// Negation preserves uncertainty instead of inventing missing proof.
    pub fn negate(self) -> Self {
        match self {
            Self::Pass => Self::Fail,
            Self::Fail => Self::Pass,
            Self::Unknown => Self::Unknown,
        }
    }
    /// Empty required collections are unknown rather than vacuous completion.
    pub fn all(values: &[Self]) -> Self {
        if values.contains(&Self::Fail) {
            Self::Fail
        } else if values.is_empty() || values.contains(&Self::Unknown) {
            Self::Unknown
        } else {
            Self::Pass
        }
    }
    /// An independently valid passing branch is sufficient; unknown alternatives remain unknown otherwise.
    pub fn any(values: &[Self]) -> Self {
        if values.contains(&Self::Pass) {
            Self::Pass
        } else if values.is_empty() || values.contains(&Self::Unknown) {
            Self::Unknown
        } else {
            Self::Fail
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateResultDetail {
    pub gate_path: String,
    pub outcome: GateOutcome,
    pub actual: Value,
    pub expected: Value,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateEvaluationResult {
    pub outcome: GateOutcome,
    pub details: Vec<GateResultDetail>,
}

/// String observations are redacted by default and never included unbounded in display details.
pub struct CompletionGateEvaluator {
    pub redact: bool,
}

impl Default for CompletionGateEvaluator {
    /// Safe display defaults apply to both actual and expected values.
    fn default() -> Self {
        Self { redact: true }
    }
}

/// Dot paths and numeric operators deliberately match eval syntax without importing the eval crate.
pub fn evidence_path<'a>(root: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.')
        .try_fold(root, |value, part| value.get(part))
}

/// A known missing path supports exists:false; a missing capture remains unknown at the caller.
pub fn match_path(root: &Value, assertion: &PathAssertion) -> GateOutcome {
    let actual = evidence_path(root, &assertion.path);
    if let Some(exists) = assertion.exists
        && exists != actual.is_some()
    {
        return GateOutcome::Fail;
    }
    let Some(actual) = actual else {
        return if assertion.exists == Some(false)
            && assertion.eq.is_none()
            && [assertion.gte, assertion.gt, assertion.lte, assertion.lt]
                .iter()
                .all(Option::is_none)
        {
            GateOutcome::Pass
        } else {
            GateOutcome::Unknown
        };
    };
    if assertion
        .eq
        .as_ref()
        .is_some_and(|expected| expected != actual)
    {
        return GateOutcome::Fail;
    }
    for (bound, op) in [
        (assertion.gte, 0),
        (assertion.gt, 1),
        (assertion.lte, 2),
        (assertion.lt, 3),
    ] {
        if let Some(bound) = bound {
            let Some(number) = actual.as_f64().filter(|n| n.is_finite()) else {
                return GateOutcome::Fail;
            };
            if !match op {
                0 => number >= bound,
                1 => number > bound,
                2 => number <= bound,
                _ => number < bound,
            } {
                return GateOutcome::Fail;
            }
        }
    }
    GateOutcome::Pass
}

/// Selects the latest matching check first; a stale latest result cannot expose an older passing one.
pub fn latest_validation<'a>(
    results: &'a [ValidationResult],
    check: &str,
    scope: &EvaluationScope,
) -> Option<&'a ValidationResult> {
    let latest = results
        .iter()
        .filter(|r| {
            r.check_id == check
                && r.identity.key == scope.key
                && r.identity.objective_revision == scope.objective_revision
        })
        .max_by_key(|r| (r.identity.cycle, r.identity.sequence))?;
    (latest.identity.eligible(scope, false, true)
        && scope.validation_bindings.get(check) == Some(&latest.identity.config_identity)
        && scope.validation_attempts.get(check) == latest.identity.attempt.as_ref()
        && latest.identity.attempt.is_some())
    .then_some(latest)
}

// Validation leaves require the selected attempt and full frozen check binding, unlike cumulative ordinary tool proofs.
fn selected_validation_identity(
    identity: &EvidenceIdentity,
    check: &str,
    scope: &EvaluationScope,
) -> bool {
    identity.attempt.is_some()
        && scope.validation_bindings.get(check) == Some(&identity.config_identity)
        && scope.validation_attempts.get(check) == identity.attempt.as_ref()
}

// Only bounded, Unicode-safe previews cross the display boundary.
fn display(value: Value, redact: bool) -> Value {
    match value {
        Value::String(_) if redact => json!("[REDACTED]"),
        Value::String(s) => Value::String(s.chars().take(256).collect()),
        Value::Array(v) => {
            Value::Array(v.into_iter().take(32).map(|v| display(v, redact)).collect())
        }
        Value::Object(v) => Value::Object(
            v.into_iter()
                .take(32)
                .map(|(k, v)| (k.chars().take(128).collect(), display(v, redact)))
                .collect(),
        ),
        other => other,
    }
}

impl CompletionGateEvaluator {
    /// Validates programmatic gates before evaluating immutable, scope-bound observations.
    pub fn evaluate(
        &self,
        gate: &CompletionGate,
        scope: &EvaluationScope,
        evidence: &EvaluationEvidence,
    ) -> Result<GateEvaluationResult> {
        gate.validate().map_err(AgentError::Config)?;
        super::bounded_value(&serde_json::to_value(gate)?, 65_536)?;
        let evidence = evidence.normalized()?;
        let mut details = vec![];
        let outcome = self.visit(gate, "root", scope, &evidence, &mut details);
        Ok(GateEvaluationResult { outcome, details })
    }

    // Every leaf remains scoped; composites preserve unknown values and produce bounded details.
    fn visit(
        &self,
        gate: &CompletionGate,
        path: &str,
        scope: &EvaluationScope,
        e: &EvaluationEvidence,
        details: &mut Vec<GateResultDetail>,
    ) -> GateOutcome {
        let known = |value: bool| {
            if value {
                GateOutcome::Pass
            } else {
                GateOutcome::Fail
            }
        };
        let current = e
            .current
            .as_ref()
            .filter(|o| o.complete && o.identity.eligible(scope, true, true))
            .map(|o| &o.value);
        let outcome = match gate {
            CompletionGate::All(children) | CompletionGate::Any(children) => {
                let values: Vec<_> = children
                    .iter()
                    .enumerate()
                    .map(|(i, gate)| self.visit(gate, &format!("{path}.{i}"), scope, e, details))
                    .collect();
                if matches!(gate, CompletionGate::All(_)) {
                    GateOutcome::all(&values)
                } else {
                    GateOutcome::any(&values)
                }
            }
            CompletionGate::Not(child) => self
                .visit(child, &format!("{path}.not"), scope, e, details)
                .negate(),
            CompletionGate::TodosDone(expected) => current
                .and_then(|c| c.todos.as_ref())
                .filter(|t| {
                    t.binding.agent_id == scope.key.agent_id
                        && t.binding.run_id == scope.key.run_id
                        && t.binding.objective_revision == scope.objective_revision
                })
                .map_or(GateOutcome::Unknown, |t| {
                    if t.items.is_empty()
                        || t.items.iter().all(|i| i.status == TodoStatus::Cancelled)
                    {
                        GateOutcome::Unknown
                    } else {
                        known(
                            t.items.iter().any(|i| i.status == TodoStatus::Completed)
                                && t.items.iter().all(|i| {
                                    matches!(
                                        i.status,
                                        TodoStatus::Completed | TodoStatus::Cancelled
                                    )
                                }),
                        )
                        .if_expected(*expected)
                    }
                }),
            CompletionGate::TodoCountGte(count) => current
                .and_then(|c| c.todos.as_ref())
                .filter(|t| {
                    t.binding.agent_id == scope.key.agent_id
                        && t.binding.run_id == scope.key.run_id
                        && t.binding.objective_revision == scope.objective_revision
                })
                .map_or(GateOutcome::Unknown, |t| known(t.items.len() >= *count)),
            CompletionGate::State(state) => current
                .and_then(|c| c.state.as_ref())
                .map_or(GateOutcome::Unknown, |s| known(s == state)),
            CompletionGate::StateIn(states) => current
                .and_then(|c| c.state.as_ref())
                .map_or(GateOutcome::Unknown, |s| known(states.contains(s))),
            CompletionGate::ContextPath(assertion) => {
                current.map_or(GateOutcome::Unknown, |c| match_path(&c.context, assertion))
            }
            CompletionGate::ProgressPath(assertion) => e
                .progress
                .as_ref()
                .filter(|o| o.complete && o.identity.eligible(scope, true, true))
                .map_or(GateOutcome::Unknown, |o| match_path(&o.value, assertion)),
            CompletionGate::Observability(assertion) => e
                .observability
                .as_ref()
                .filter(|o| o.complete && o.identity.eligible(scope, true, false))
                .map_or(GateOutcome::Unknown, |o| match_path(&o.value, assertion)),
            CompletionGate::ResponseNotEmpty(expected) => current
                .map_or(GateOutcome::Unknown, |c| {
                    known(!c.response.trim().is_empty()).if_expected(*expected)
                }),
            CompletionGate::ResponseContains(s) => {
                current.map_or(GateOutcome::Unknown, |c| known(c.response.contains(s)))
            }
            CompletionGate::ResponseContainsAny(values) => current
                .map_or(GateOutcome::Unknown, |c| {
                    known(values.iter().any(|s| c.response.contains(s)))
                }),
            CompletionGate::ToolCalled(predicate) => {
                let eligible: Vec<_> = e
                    .tools
                    .iter()
                    .filter(|o| {
                        o.complete
                            && o.identity.eligible(scope, false, false)
                            && o.value.canonical_id == predicate.id
                            && executed(&o.value)
                            && o.value.success
                    })
                    .collect();
                if eligible.is_empty()
                    || predicate.executed == Some(false)
                    || predicate.success == Some(false)
                {
                    GateOutcome::Unknown
                } else {
                    known(
                        predicate.count.is_none_or(|n| eligible.len() == n)
                            && predicate.count_gte.is_none_or(|n| eligible.len() >= n),
                    )
                }
            }
            CompletionGate::CommandExit(predicate) => {
                let latest = e
                    .commands
                    .iter()
                    .filter(|o| o.identity.key == scope.key && o.value.command == predicate.command)
                    .max_by_key(|o| (o.identity.cycle, o.identity.sequence));
                latest
                    .filter(|o| {
                        o.complete
                            && o.identity.eligible(scope, false, true)
                            && executed(&o.value.record)
                            && !o.value.record.output_truncated
                            && o.value.record.canonical_id == "command"
                            && selected_validation_identity(&o.identity, &o.value.check_id, scope)
                            && o.value.termination == "exited"
                    })
                    .and_then(|o| o.value.exit_code)
                    .map_or(GateOutcome::Unknown, |code| known(code == predicate.code))
            }
            CompletionGate::DiagnosticsClear(predicate) => e
                .diagnostics
                .iter()
                .filter(|o| o.identity.key == scope.key)
                .max_by_key(|o| (o.identity.cycle, o.identity.sequence))
                .filter(|o| {
                    o.complete
                        && o.identity.eligible(scope, false, true)
                        && o.value.available
                        && o.value.complete
                        && executed(&o.value.record)
                        && !o.value.record.output_truncated
                        && o.value.record.canonical_id == "diagnostics"
                        && selected_validation_identity(&o.identity, &o.value.check_id, scope)
                        && o.value.record.success
                })
                .map_or(GateOutcome::Unknown, |o| {
                    o.value
                        .severity_counts
                        .get(&predicate.severity)
                        .map_or(GateOutcome::Unknown, |n| known(*n == 0))
                }),
            CompletionGate::ArtifactExists(path) => e
                .artifacts
                .iter()
                .filter(|o| o.value.path == *path && o.identity.key == scope.key)
                .max_by_key(|o| (o.identity.cycle, o.identity.sequence))
                .filter(|o| o.complete && o.identity.eligible(scope, true, true))
                .map_or(GateOutcome::Unknown, |o| known(o.value.exists)),
            CompletionGate::ValidatorPassed(check) => {
                latest_validation(&e.validations, check, scope)
                    .map_or(GateOutcome::Unknown, |r| r.outcome)
            }
            CompletionGate::ValidationPassed(expected) => GateOutcome::all(
                &e.required_checks
                    .iter()
                    .map(|id| {
                        latest_validation(&e.validations, id, scope)
                            .map_or(GateOutcome::Unknown, |r| r.outcome)
                    })
                    .collect::<Vec<_>>(),
            )
            .if_expected(*expected),
            CompletionGate::Judge(config) => {
                let identity =
                    super::canonical_identity(&serde_json::to_value(config).unwrap_or(Value::Null))
                        .unwrap_or_default();
                e.judges
                    .iter()
                    .filter(|r| {
                        r.metrics
                            .get("judge_config_identity")
                            .and_then(Value::as_str)
                            == Some(&identity)
                    })
                    .max_by_key(|r| (r.identity.cycle, r.identity.sequence))
                    .filter(|r| {
                        r.identity.eligible(scope, true, true)
                            && selected_validation_identity(&r.identity, &r.check_id, scope)
                    })
                    .map_or(GateOutcome::Unknown, |r| r.outcome)
            }
        };
        details.push(GateResultDetail {
            gate_path: path.into(),
            outcome,
            actual: json!(outcome),
            expected: display(
                serde_json::to_value(gate).unwrap_or(Value::Null),
                self.redact,
            ),
            reason: match outcome {
                GateOutcome::Pass => "verified",
                GateOutcome::Fail => "predicate_failed",
                GateOutcome::Unknown => "missing_or_ineligible_evidence",
            }
            .into(),
        });
        outcome
    }
}

impl GateOutcome {
    // Boolean gate operands invert only known evidence.
    fn if_expected(self, expected: bool) -> Self {
        if expected { self } else { self.negate() }
    }
}

//! Converts acknowledged driver records into typed observations without accepting model-written proof fields.

use super::*;
use ai_agents_core::{AgentError, Result};
use serde_json::Value;

impl EvaluationEvidence {
    /// Uses the driver's exact executor records and attempt identity, never its display preview.
    pub fn collect_validation(&mut self, state: &ValidationDriverState) -> Result<()> {
        if !state.publication_acknowledged() {
            return Err(AgentError::Persistence(
                "validation result was not acknowledged".into(),
            ));
        }
        let result = state
            .result
            .as_ref()
            .ok_or_else(|| AgentError::Config("validation has no terminal result".into()))?;
        self.validations.push(result.clone());
        if state.adapter == "builtin.judge" {
            self.judges.push(result.clone());
        }
        for completed in state.completed.values() {
            if let (
                ValidationObservationRequest::Tool {
                    tool, arguments, ..
                },
                ObservationResult::Tool { record },
            ) = (&completed.request, &completed.result)
            {
                self.tools.push(ScopedObservation {
                    identity: state.identity.clone(),
                    complete: true,
                    value: record.as_ref().clone(),
                });
                if tool == "command" {
                    let output: Value = serde_json::from_str(&record.output).unwrap_or(Value::Null);
                    if let Some(command) = arguments.get("command").and_then(Value::as_str) {
                        self.commands.push(ScopedObservation {
                            identity: state.identity.clone(),
                            complete: super::evidence::executed(record) && !record.output_truncated,
                            value: CommandObservation {
                                check_id: state.check_id.clone(),
                                command: command.into(),
                                exit_code: output
                                    .get("exit_code")
                                    .and_then(Value::as_i64)
                                    .and_then(|c| i32::try_from(c).ok()),
                                termination: output
                                    .get("termination")
                                    .and_then(Value::as_str)
                                    .unwrap_or("unavailable")
                                    .into(),
                                record: record.as_ref().clone(),
                            },
                        });
                    }
                }
                if tool == "diagnostics" {
                    let output: Value = serde_json::from_str(&record.output).unwrap_or(Value::Null);
                    let mut counts = std::collections::BTreeMap::from([
                        ("error".into(), 0),
                        ("warning".into(), 0),
                        ("info".into(), 0),
                        ("hint".into(), 0),
                        ("all".into(), 0),
                    ]);
                    let mut valid = true;
                    if let Some(items) = output.get("diagnostics").and_then(Value::as_array) {
                        for item in items {
                            if let Some(severity) = item.get("severity").and_then(Value::as_str)
                                && counts.contains_key(severity)
                                && severity != "all"
                            {
                                *counts.get_mut(severity).unwrap() += 1;
                                *counts.get_mut("all").unwrap() += 1;
                            } else {
                                valid = false;
                            }
                        }
                    } else {
                        valid = false;
                    }
                    let available = output.get("available").and_then(Value::as_bool) == Some(true);
                    let complete = valid
                        && available
                        && output.get("truncated").and_then(Value::as_bool) == Some(false)
                        && output.get("message").is_none_or(Value::is_null)
                        && !record.output_truncated;
                    self.diagnostics.push(ScopedObservation {
                        identity: state.identity.clone(),
                        complete,
                        value: DiagnosticsObservation {
                            check_id: state.check_id.clone(),
                            available,
                            complete,
                            severity_counts: counts,
                            record: record.as_ref().clone(),
                        },
                    });
                }
            }
        }
        *self = self.normalized()?;
        Ok(())
    }
}

impl ProgressObservation {
    /// Unknown or absent signals are not promoted into a known empty metrics object for negated gates.
    pub fn scoped(&self, identity: EvidenceIdentity) -> ScopedObservation<Value> {
        ScopedObservation {
            identity,
            complete: !self.signals.is_empty() && !self.evidence_refs.is_empty(),
            value: serde_json::json!({"metrics":self.metrics}),
        }
    }
}

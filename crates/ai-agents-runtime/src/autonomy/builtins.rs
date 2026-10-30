//! Built-in pure validators request I/O from the managed observation driver.

use super::*;
use ai_agents_core::autonomy::{JudgeGate, PathAssertion};
use ai_agents_core::{AgentError, Result};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CommandConfig {
    command: String,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    expected_exit: i32,
    #[serde(default)]
    host_binding: Option<String>,
}
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct DiagnosticsConfig {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    severity: Option<String>,
    #[serde(default)]
    host_binding: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactConfig {
    path: String,
    #[serde(default)]
    host_binding: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceConfig {
    assertion: PathAssertion,
    #[serde(default)]
    validator: Option<String>,
}

pub struct BuiltinValidator {
    descriptor: AdapterDescriptor,
    kind: String,
}
impl BuiltinValidator {
    /// Descriptors declare exactly the observations each built-in may request.
    pub fn new(kind: &str) -> Self {
        Self {
            descriptor: AdapterDescriptor {
                id: format!("builtin.{kind}"),
                contract_version: 1,
                checks: vec![],
                tools: if kind == "artifact" {
                    vec!["file_info".into()]
                } else if matches!(kind, "command" | "diagnostics") {
                    vec![kind.into()]
                } else {
                    vec![]
                },
                needs_judge: kind == "judge",
                ..Default::default()
            },
            kind: kind.into(),
        }
    }
}
// Conflicting host-reported process fields cannot turn an otherwise eligible command record into completion proof.
pub(super) fn command_output_consistent(output: &Value) -> bool {
    let timeout_consistent = output
        .get("timed_out")
        .is_none_or(|value| value.as_bool() == Some(false));
    let success_consistent = match output.get("success") {
        None => true,
        Some(value) => match (
            value.as_bool(),
            output.get("exit_code").and_then(Value::as_i64),
        ) {
            (Some(success), Some(code)) => success == (code == 0),
            (Some(_), None) => true,
            _ => false,
        },
    };
    timeout_consistent && success_consistent
}

impl AutonomyValidator for BuiltinValidator {
    /// Implementations cannot change identity after registry binding.
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.descriptor
    }
    /// Framework-owned built-in config remains strict; custom adapters own only their config maps.
    fn validate_config(&self, config: &Value) -> Result<()> {
        bounded_value(config, 16_384)?;
        match self.kind.as_str() {
            "command" => {
                let config: CommandConfig = serde_json::from_value(config.clone())
                    .map_err(|_| AgentError::Config("invalid command validator config".into()))?;
                if config.command.trim().is_empty() {
                    return Err(AgentError::Config("empty validation command".into()));
                }
            }
            "diagnostics" => {
                let config: DiagnosticsConfig =
                    serde_json::from_value(config.clone()).map_err(|_| {
                        AgentError::Config("invalid diagnostics validator config".into())
                    })?;
                if config.severity.as_ref().is_some_and(|s| {
                    !matches!(s.as_str(), "error" | "warning" | "info" | "hint" | "all")
                }) {
                    return Err(AgentError::Config("invalid diagnostic severity".into()));
                }
            }
            "judge" => {
                let config: JudgeGate = serde_json::from_value(config.clone())
                    .map_err(|_| AgentError::Config("invalid judge config".into()))?;
                ai_agents_core::autonomy::CompletionGate::Judge(config)
                    .validate()
                    .map_err(AgentError::Config)?;
            }
            "artifact" => {
                let config: ArtifactConfig = serde_json::from_value(config.clone())?;
                if config.path.trim().is_empty() || config.path.len() > 4096 {
                    return Err(AgentError::Config("invalid artifact path".into()));
                }
            }
            "evidence" => {
                let config: EvidenceConfig = serde_json::from_value(config.clone())
                    .map_err(|_| AgentError::Config("invalid evidence validator config".into()))?;
                ai_agents_core::autonomy::CompletionGate::ContextPath(config.assertion)
                    .validate()
                    .map_err(AgentError::Config)?;
            }
            _ => return Err(AgentError::Config("unknown builtin validator".into())),
        }
        Ok(())
    }
    /// Denied, unavailable, partial and stale observations cannot become passing validation.
    /// A truncated command envelope remains incomplete and cannot establish whether the original response was valid.
    fn evaluate(&self, input: &ValidationInput<'_>) -> Result<ValidationDecision> {
        self.validate_config(input.config)?;
        let complete = |outcome, metrics| ValidationDecision::Complete {
            outcome,
            reason: match outcome {
                GateOutcome::Pass => "verified",
                GateOutcome::Fail => "validation_failed",
                GateOutcome::Unknown => "observation_unavailable",
            }
            .into(),
            metrics,
            evidence_refs: input.observations.keys().cloned().collect(),
        };
        if self.kind == "evidence" {
            let config: EvidenceConfig = serde_json::from_value(input.config.clone())?;
            let root = if let Some(id) = config.validator {
                latest_validation(&input.evidence.validations, &id, input.scope)
                    .filter(|r| r.outcome != GateOutcome::Unknown)
                    .map(|r| json!({"metrics":r.metrics}))
            } else {
                input
                    .evidence
                    .current
                    .as_ref()
                    .filter(|o| o.complete && o.identity.eligible(input.scope, true, true))
                    .map(|o| o.value.context.clone())
            };
            return Ok(complete(
                root.as_ref().map_or(GateOutcome::Unknown, |root| {
                    match_path(root, &config.assertion)
                }),
                json!({}),
            ));
        }
        if let Some(observation) = input.observations.get("observe") {
            return match (&*self.kind, &observation.result) {
                ("artifact", ObservationResult::Tool { record }) => {
                    let config: ArtifactConfig = serde_json::from_value(input.config.clone())?;
                    let output: Value = serde_json::from_str(&record.output).unwrap_or(Value::Null);
                    let valid = super::evidence::executed(record)
                        && record.success
                        && !record.output_truncated
                        && record.canonical_id == "file_info"
                        && record
                            .executed_arguments
                            .get("path")
                            .and_then(Value::as_str)
                            == Some(config.path.as_str())
                        && output.get("path").and_then(Value::as_str) == Some(config.path.as_str());
                    let Some(exists) = output
                        .get("exists")
                        .and_then(Value::as_bool)
                        .filter(|_| valid)
                    else {
                        return Ok(complete(GateOutcome::Unknown, json!({})));
                    };
                    Ok(complete(
                        if exists {
                            GateOutcome::Pass
                        } else {
                            GateOutcome::Fail
                        },
                        json!({"artifact":{"path":config.path,"exists":exists}}),
                    ))
                }
                ("command", ObservationResult::Tool { record }) => {
                    if !super::evidence::executed(record) {
                        return Ok(complete(GateOutcome::Unknown, json!({})));
                    }
                    let config: CommandConfig = serde_json::from_value(input.config.clone())?;
                    let output: Value = serde_json::from_str(&record.output).map_err(|_| {
                        if record.output_truncated {
                            AgentError::Config(
                                "command observation exceeded the output limit; original result validity is unknown".into(),
                            )
                        } else {
                            AgentError::Config("malformed command observation output".into())
                        }
                    })?;
                    let code = output.get("exit_code").and_then(Value::as_i64);
                    let unchanged = record
                        .executed_arguments
                        .get("command")
                        .and_then(Value::as_str)
                        == Some(&config.command)
                        && record.executed_arguments.get("cwd").and_then(Value::as_str)
                            == Some(config.cwd.as_deref().unwrap_or("."))
                        && record
                            .executed_arguments
                            .get("argv")
                            .is_none_or(|v| v.as_array().is_some_and(Vec::is_empty))
                        && record
                            .executed_arguments
                            .get("env")
                            .is_none_or(|v| v.as_object().is_some_and(serde_json::Map::is_empty));
                    let eligible = unchanged
                        && command_output_consistent(&output)
                        && super::evidence::executed(record)
                        && record.canonical_id == "command"
                        && output.get("termination").and_then(Value::as_str) == Some("exited")
                        && !record.output_truncated;
                    Ok(complete(
                        if !eligible || code.is_none() {
                            GateOutcome::Unknown
                        } else if code == Some(i64::from(config.expected_exit)) {
                            GateOutcome::Pass
                        } else {
                            GateOutcome::Fail
                        },
                        json!({"exit_code":code}),
                    ))
                }
                ("diagnostics", ObservationResult::Tool { record }) => {
                    let config: DiagnosticsConfig = serde_json::from_value(input.config.clone())?;
                    let output: Value = match serde_json::from_str(&record.output) {
                        Ok(v) => v,
                        Err(_) if !record.executed => {
                            return Ok(complete(GateOutcome::Unknown, json!({})));
                        }
                        Err(_) => {
                            return Err(AgentError::Config(
                                "malformed diagnostics observation output".into(),
                            ));
                        }
                    };
                    let available = output.get("available").and_then(Value::as_bool) == Some(true)
                        && output.get("truncated").and_then(Value::as_bool) == Some(false)
                        && output.get("message").is_none_or(Value::is_null);
                    let Some(items) = output.get("diagnostics").and_then(Value::as_array) else {
                        return Ok(complete(GateOutcome::Unknown, json!({})));
                    };
                    if record
                        .executed_arguments
                        .get("path")
                        .and_then(Value::as_str)
                        != Some(config.path.as_deref().unwrap_or("."))
                    {
                        return Ok(complete(GateOutcome::Unknown, json!({})));
                    }
                    if !available
                        || !super::evidence::executed(record)
                        || !record.success
                        || record.output_truncated
                    {
                        return Ok(complete(GateOutcome::Unknown, json!({})));
                    }
                    let severity = config.severity.as_deref().unwrap_or("error");
                    let count = items
                        .iter()
                        .filter(|d| {
                            severity == "all"
                                || d.get("severity").and_then(Value::as_str) == Some(severity)
                        })
                        .count();
                    if items.iter().any(|d| {
                        d.get("severity")
                            .and_then(Value::as_str)
                            .is_none_or(|s| !matches!(s, "error" | "warning" | "info" | "hint"))
                    }) {
                        return Err(AgentError::Config(
                            "invalid diagnostic observation item".into(),
                        ));
                    }
                    Ok(complete(
                        if count == 0 {
                            GateOutcome::Pass
                        } else {
                            GateOutcome::Fail
                        },
                        json!({"count":count}),
                    ))
                }
                ("judge", ObservationResult::Judge { score }) => {
                    let config: JudgeGate = serde_json::from_value(input.config.clone())?;
                    Ok(complete(
                        if *score >= config.pass_threshold.unwrap_or(0.8) {
                            GateOutcome::Pass
                        } else {
                            GateOutcome::Fail
                        },
                        json!({"score":score,"judge_config_identity":canonical_identity(&serde_json::to_value(&config)?)?}),
                    ))
                }
                _ => Err(AgentError::Config(
                    "builtin observation kind mismatch".into(),
                )),
            };
        }
        let request = match self.kind.as_str() {
            "artifact" => {
                let config: ArtifactConfig = serde_json::from_value(input.config.clone())?;
                ValidationObservationRequest::Tool {
                    id: "observe".into(),
                    tool: "file_info".into(),
                    arguments: json!({"path":config.path}),
                    host_binding: config.host_binding,
                }
            }
            "command" => {
                let config: CommandConfig = serde_json::from_value(input.config.clone())?;
                ValidationObservationRequest::Tool {
                    id: "observe".into(),
                    tool: "command".into(),
                    arguments: json!({"command":config.command,"cwd":config.cwd.unwrap_or_else(|| ".".into())}),
                    host_binding: config.host_binding,
                }
            }
            "diagnostics" => {
                let config: DiagnosticsConfig = serde_json::from_value(input.config.clone())?;
                ValidationObservationRequest::Tool {
                    id: "observe".into(),
                    tool: "diagnostics".into(),
                    arguments: json!({"path":config.path.unwrap_or_else(|| ".".into()),"severity":"all","max_results":200}),
                    host_binding: config.host_binding,
                }
            }
            "judge" => {
                let config: JudgeGate = serde_json::from_value(input.config.clone())?;
                let Some(current) = input
                    .evidence
                    .current
                    .as_ref()
                    .filter(|o| o.complete && o.identity.eligible(input.scope, true, true))
                else {
                    return Ok(complete(GateOutcome::Unknown, json!({})));
                };
                ValidationObservationRequest::Judge {
                    id: "observe".into(),
                    config,
                    response: current.value.response.clone(),
                    objective: "Evaluate the configured criteria".into(),
                }
            }
            _ => return Err(AgentError::Config("unknown observation builtin".into())),
        };
        Ok(ValidationDecision::NeedObservations {
            requests: vec![request],
            checkpoint: json!({"requested":true}),
        })
    }

    /// Only a verified file-info result projects existence; an unknown or malformed observation supplies no artifact proof.
    fn artifacts(&self, result: &ValidationResult) -> Result<Vec<ArtifactObservation>> {
        if self.kind != "artifact" || result.outcome == GateOutcome::Unknown {
            return Ok(Vec::new());
        }
        Ok(vec![serde_json::from_value(
            result.metrics["artifact"].clone(),
        )?])
    }
}

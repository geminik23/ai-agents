//! Replanning is a bounded managed observation of the existing planner, not a replacement planning engine.

use super::*;
use ai_agents_core::autonomy::{ValidationCheck, ValidationSchedule};
use ai_agents_core::{AgentError, Result};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplanConfig {
    objective: String,
    intervention_id: String,
}

pub(crate) struct ReplanValidator {
    descriptor: AdapterDescriptor,
}
impl ReplanValidator {
    /// Only an explicitly selected intervention constructs this internal validator.
    pub(crate) fn new() -> Self {
        Self {
            descriptor: AdapterDescriptor {
                id: "internal.replan".into(),
                contract_version: 1,
                needs_planner: true,
                ..Default::default()
            },
        }
    }
}
impl AutonomyValidator for ReplanValidator {
    /// The planner capability is distinct from tool authority and semantic judge routing.
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.descriptor
    }
    /// Objective and intervention identity are immutable throughout callback reentry.
    fn validate_config(&self, config: &Value) -> Result<()> {
        let config: ReplanConfig = serde_json::from_value(config.clone())?;
        if config.objective.trim().is_empty() || config.intervention_id.is_empty() {
            return Err(AgentError::Config("invalid replan configuration".into()));
        }
        Ok(())
    }
    /// A saved plan result reenters without another planner call and never resets progress or resources.
    fn evaluate(&self, input: &ValidationInput<'_>) -> Result<ValidationDecision> {
        self.validate_config(input.config)?;
        let config: ReplanConfig = serde_json::from_value(input.config.clone())?;
        if let Some(observation) = input.observations.get("plan") {
            if let ObservationResult::Plan { plan } = &observation.result {
                return Ok(ValidationDecision::Complete {
                    outcome: GateOutcome::Pass,
                    reason: "replanned".into(),
                    metrics: json!({"plan":plan}),
                    evidence_refs: vec!["plan".into()],
                });
            }
            return Err(AgentError::Config("replanner result kind mismatch".into()));
        }
        Ok(ValidationDecision::NeedObservations {
            requests: vec![ValidationObservationRequest::Plan {
                id: "plan".into(),
                objective: config.objective,
            }],
            checkpoint: json!({"intervention_id":config.intervention_id}),
        })
    }
}

impl BoundAutonomyProfile {
    /// A single selected replan observation inherits the original objective, stages, gates, counters and maximum replan count.
    pub fn replan_check(
        &self,
        progress: &ProgressState,
        objective: &str,
    ) -> Result<BoundValidationCheck> {
        let max = self
            .profile
            .progress
            .as_ref()
            .and_then(|p| p.stagnation.as_ref())
            .and_then(|s| s.max_replans)
            .unwrap_or(0);
        if progress.replans == 0
            || progress.replans > u64::from(max)
            || progress.handled_sequence != Some(progress.sequence)
        {
            return Err(AgentError::Config(
                "replan was not selected within the admitted policy".into(),
            ));
        }
        let id = format!("__replan.{}", progress.sequence);
        let config = json!({"objective":objective,"intervention_id":id});
        let adapter = std::sync::Arc::new(ReplanValidator::new());
        adapter.validate_config(&config)?;
        let check = ValidationCheck {
            id,
            adapter: "internal.replan".into(),
            contract_version: Some(1),
            required: Some(false),
            schedule: Some(ValidationSchedule::Completion),
            timeout_seconds: None,
            max_evaluation_rounds: Some(2),
            max_observations_per_round: Some(1),
            config: Some(config),
        };
        Ok(BoundValidationCheck {
            config_identity: canonical_identity(&serde_json::to_value(&check)?)?,
            check,
            stage: progress.scope.stage.clone(),
            adapter,
            available: true,
        })
    }
}

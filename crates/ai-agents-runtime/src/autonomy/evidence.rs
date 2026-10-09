//! Eligibility-aware observations; only the shared executor or explicit host observers supply authority.

use super::{GateOutcome, TaskTodoCheckpoint};
use ai_agents_core::autonomy::TaskRunKey;
use ai_agents_core::{PermissionOutcome, ToolExecutionRecord};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// An objective revision invalidates prior decisions without discarding the effect audit history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationScope {
    pub key: TaskRunKey,
    pub objective_revision: u64,
    pub cycle: u64,
    pub stage: Option<String>,
    pub mutation_generation: u64,
    pub target_revisions: BTreeMap<String, String>,
    pub target_generations: BTreeMap<String, u64>,
    pub validation_bindings: BTreeMap<String, String>,
    pub validation_attempts: BTreeMap<String, String>,
}

/// Freshness includes configuration and target identity, not a timestamp heuristic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceIdentity {
    pub key: TaskRunKey,
    pub objective_revision: u64,
    pub cycle: u64,
    pub sequence: u64,
    pub stage: Option<String>,
    pub attempt: Option<String>,
    pub config_identity: String,
    pub target: Option<String>,
    pub target_revision: Option<String>,
    pub mutation_generation: u64,
}

impl EvidenceIdentity {
    /// Checks scope and optional host revisions; cumulative work may precede the current cycle.
    pub fn eligible(&self, scope: &EvaluationScope, current: bool, fresh: bool) -> bool {
        self.key == scope.key
            && self.objective_revision == scope.objective_revision
            && self.cycle <= scope.cycle
            && (!current || self.cycle == scope.cycle)
            && scope
                .stage
                .as_ref()
                .is_none_or(|stage| self.stage.as_ref() == Some(stage))
            && (!fresh
                || self.mutation_generation
                    == self
                        .target
                        .as_ref()
                        .and_then(|target| scope.target_generations.get(target).copied())
                        .unwrap_or(scope.mutation_generation))
            && self
                .target
                .as_ref()
                .is_none_or(|target| match scope.target_revisions.get(target) {
                    Some(revision) => self.target_revision.as_ref() == Some(revision),
                    None => self.target_revision.is_none(),
                })
    }
}

/// Incomplete or unavailable observations remain unknown, even when their payload looks empty.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScopedObservation<T> {
    pub identity: EvidenceIdentity,
    pub complete: bool,
    pub value: T,
}

/// Current values are captured after maintenance flush; absent capture is not an empty observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurrentTaskObservation {
    pub state: Option<String>,
    pub context: Value,
    pub response: String,
    pub todos: Option<TaskTodoCheckpoint>,
}

/// A validator result never changes task status and cannot forge shared-executor execution evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationResult {
    pub check_id: String,
    pub identity: EvidenceIdentity,
    pub outcome: GateOutcome,
    pub reason: String,
    pub metrics: Value,
    pub evidence_refs: Vec<String>,
}

/// Command outputs and diagnostics are derived from exact shared-executor records.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommandObservation {
    pub check_id: String,
    pub command: String,
    pub exit_code: Option<i32>,
    pub termination: String,
    pub record: ToolExecutionRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticsObservation {
    pub check_id: String,
    pub available: bool,
    pub complete: bool,
    pub severity_counts: BTreeMap<String, usize>,
    pub record: ToolExecutionRecord,
}

/// Artifacts are authorized tool or host observations, never unrestricted filesystem probes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactObservation {
    pub path: String,
    pub exists: bool,
}

/// Gate input is immutable; progress is computed before progress-dependent completion predicates.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationEvidence {
    pub current: Option<ScopedObservation<CurrentTaskObservation>>,
    pub tools: Vec<ScopedObservation<ToolExecutionRecord>>,
    pub commands: Vec<ScopedObservation<CommandObservation>>,
    pub diagnostics: Vec<ScopedObservation<DiagnosticsObservation>>,
    pub artifacts: Vec<ScopedObservation<ArtifactObservation>>,
    pub validations: Vec<ValidationResult>,
    pub required_checks: Vec<String>,
    pub progress: Option<ScopedObservation<Value>>,
    pub observability: Option<ScopedObservation<Value>>,
    pub judges: Vec<ValidationResult>,
}

impl EvaluationEvidence {
    /// Duplicate captures are idempotent; contradictory copies of one authority-bearing event are errors.
    pub fn normalized(&self) -> ai_agents_core::Result<Self> {
        let count = self.tools.len()
            + self.commands.len()
            + self.diagnostics.len()
            + self.artifacts.len()
            + self.validations.len()
            + self.judges.len();
        if count > 4096 {
            return Err(ai_agents_core::AgentError::Config(
                "evaluation evidence record bound exceeded".into(),
            ));
        }
        if self
            .current
            .as_ref()
            .is_some_and(|o| o.complete && !o.value.context.is_object())
        {
            return Err(ai_agents_core::AgentError::Config(
                "current context capture is not an object".into(),
            ));
        }
        let mut output = self.clone();
        output.tools = unique(&self.tools, |o| {
            super::canonical_identity(
                &serde_json::json!({"key":o.identity.key,"objective":o.identity.objective_revision,"cycle":o.identity.cycle,"attempt":o.identity.attempt,"call":o.value.call_id}),
            )
        })?;
        output.commands = unique(&self.commands, |o| {
            super::canonical_identity(
                &serde_json::json!({"check":o.value.check_id,"identity":o.identity}),
            )
        })?;
        output.diagnostics = unique(&self.diagnostics, |o| {
            super::canonical_identity(
                &serde_json::json!({"check":o.value.check_id,"identity":o.identity}),
            )
        })?;
        output.artifacts = unique(&self.artifacts, |o| {
            super::canonical_identity(
                &serde_json::json!({"path":o.value.path,"identity":o.identity}),
            )
        })?;
        output.validations = unique(&self.validations, |o| {
            super::canonical_identity(
                &serde_json::json!({"check":o.check_id,"identity":o.identity}),
            )
        })?;
        output.judges = unique(&self.judges, |o| {
            super::canonical_identity(
                &serde_json::json!({"check":o.check_id,"identity":o.identity}),
            )
        })?;
        Ok(output)
    }
}

// Logical-request identity excludes capture sequence for tools so a duplicate snapshot cannot manufacture invocation counts.
fn unique<T: Clone + Serialize>(
    values: &[T],
    key: impl Fn(&T) -> ai_agents_core::Result<String>,
) -> ai_agents_core::Result<Vec<T>> {
    let mut seen = BTreeMap::new();
    let mut output = vec![];
    for value in values {
        let key = key(value)?;
        let digest = super::canonical_identity(&serde_json::to_value(value)?)?;
        if let Some(previous) = seen.get(&key) {
            if previous != &digest {
                return Err(ai_agents_core::AgentError::Config(
                    "conflicting evidence for the same identity".into(),
                ));
            }
        } else {
            seen.insert(key, digest);
            output.push(value.clone());
        }
    }
    Ok(output)
}

/// Successful policy admission and real invocation are required for executable proof.
pub(crate) fn executed(record: &ToolExecutionRecord) -> bool {
    record.executed
        && matches!(record.policy.outcome, PermissionOutcome::Allow)
        && !record.cancelled
        && !record.timed_out
}

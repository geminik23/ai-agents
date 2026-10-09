//! Verified high-water progress and one persisted intervention per completed observation.

use super::{
    AdapterDescriptor, EvaluationEvidence, EvaluationScope, GateOutcome, ProgressAdapter,
    canonical_identity, evidence_path, latest_validation,
};
use ai_agents_core::autonomy::{StagnationAction, StagnationConfig};
use ai_agents_core::{AgentError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgressDelta {
    Advanced,
    Unchanged,
    Regressed,
    Unknown,
}

pub struct ProgressInput<'a> {
    pub scope: &'a EvaluationScope,
    pub evidence: &'a EvaluationEvidence,
    pub config: &'a Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgressObservation {
    pub metrics: Value,
    pub signals: BTreeMap<String, Value>,
    pub comparisons: BTreeMap<String, SignalComparison>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalComparison {
    Increase,
    Change,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Signal {
    id: String,
    #[serde(default)]
    validator: Option<String>,
    #[serde(default)]
    source: Option<String>,
    path: String,
    comparison: SignalComparison,
}
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct SignalConfig {
    #[serde(default)]
    signals: Vec<Signal>,
}

pub struct BuiltinProgress {
    descriptor: AdapterDescriptor,
    todo: bool,
}
impl BuiltinProgress {
    /// Built-in observers inspect canonical, eligible evidence only.
    pub fn new(kind: &str) -> Self {
        Self {
            descriptor: AdapterDescriptor {
                id: format!("builtin.{kind}"),
                contract_version: 1,
                checks: vec![],
                tools: vec![],
                needs_judge: false,
                ..Default::default()
            },
            todo: kind == "todo",
        }
    }
}

impl ProgressAdapter for BuiltinProgress {
    /// Registry identity is immutable once captured by a bound profile.
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.descriptor
    }
    /// Explicit signals replace defaults and reject same-cycle progress dependencies.
    fn validate_config(&self, config: &Value) -> Result<()> {
        let parsed: SignalConfig = serde_json::from_value(config.clone())
            .map_err(|_| AgentError::Config("invalid progress signal config".into()))?;
        let mut ids = BTreeSet::new();
        if parsed.signals.len() > 64 {
            return Err(AgentError::Config("too many progress signals".into()));
        }
        for signal in parsed.signals {
            if signal.id.is_empty()
                || signal.path.is_empty()
                || !ids.insert(signal.id)
                || signal
                    .source
                    .as_ref()
                    .is_some_and(|s| !matches!(s.as_str(), "context" | "state"))
                || (signal.validator.is_some() && signal.source.is_some())
            {
                return Err(AgentError::Config(
                    "invalid or cyclic progress signal".into(),
                ));
            }
        }
        Ok(())
    }
    /// Fresh timestamps, request volume and response novelty are never default advancement signals.
    fn observe(&self, input: &ProgressInput<'_>) -> Result<ProgressObservation> {
        self.validate_config(input.config)?;
        let parsed: SignalConfig = serde_json::from_value(input.config.clone())?;
        let mut signals = BTreeMap::new();
        let mut comparisons = BTreeMap::new();
        let mut refs = vec![];
        if parsed.signals.is_empty() {
            if self.todo {
                if let Some(current) = input
                    .evidence
                    .current
                    .as_ref()
                    .filter(|o| o.complete && o.identity.eligible(input.scope, true, true))
                    && let Some(todos) = &current.value.todos
                    && todos.binding.run_id == input.scope.key.run_id
                    && todos.binding.agent_id == input.scope.key.agent_id
                    && todos.binding.objective_revision == input.scope.objective_revision
                {
                    // Item identity prevents replacing a completed item from resetting a global completion count.
                    for item in &todos.items {
                        if item.status == ai_agents_tools::TodoStatus::Completed {
                            signals.insert(format!("todo.{}", item.id), json!(1));
                            comparisons
                                .insert(format!("todo.{}", item.id), SignalComparison::Change);
                        }
                    }
                    refs.push(format!("current:{}", current.identity.sequence));
                }
            } else {
                for id in &input.evidence.required_checks {
                    if let Some(result) =
                        latest_validation(&input.evidence.validations, id, input.scope)
                        && result.outcome != GateOutcome::Unknown
                    {
                        signals.insert(
                            format!("check.{id}"),
                            json!(if result.outcome == GateOutcome::Pass {
                                1
                            } else {
                                0
                            }),
                        );
                        comparisons.insert(format!("check.{id}"), SignalComparison::Increase);
                        refs.push(format!("check:{id}:{}", result.identity.sequence));
                    }
                }
            }
        } else {
            for signal in parsed.signals {
                let value = if let Some(check) = &signal.validator {
                    latest_validation(&input.evidence.validations, check, input.scope).and_then(
                        |r| {
                            refs.push(format!("check:{check}:{}", r.identity.sequence));
                            evidence_path(&json!({"metrics":r.metrics}), &signal.path).cloned()
                        },
                    )
                } else {
                    input
                        .evidence
                        .current
                        .as_ref()
                        .filter(|o| o.complete && o.identity.eligible(input.scope, true, true))
                        .and_then(|o| {
                            refs.push(format!("current:{}", o.identity.sequence));
                            let root = if signal.source.as_deref() == Some("state") {
                                json!({"state":o.value.state})
                            } else {
                                o.value.context.clone()
                            };
                            evidence_path(&root, &signal.path).cloned()
                        })
                };
                if let Some(value) = value {
                    comparisons.insert(signal.id.clone(), signal.comparison);
                    signals.insert(signal.id, value);
                }
            }
        }
        Ok(ProgressObservation {
            metrics: json!({"signals":signals}),
            signals,
            comparisons,
            evidence_refs: refs,
        })
    }
}

/// Signal check references are validated after wholesale profile/config replacement.
pub(crate) fn validate_signal_references(config: &Value, checks: &BTreeSet<String>) -> Result<()> {
    if let Some(signals) = config.get("signals").and_then(Value::as_array) {
        for signal in signals {
            if let Some(id) = signal.get("validator").and_then(Value::as_str)
                && !checks.contains(id)
            {
                return Err(AgentError::Config(
                    "unknown progress validator reference".into(),
                ));
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SignalHistory {
    high: Option<f64>,
    previous: Option<Value>,
    fingerprints: BTreeSet<String>,
}

/// History is bounded and explicit; exceeding capacity cannot make an old value appear new.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgressState {
    pub scope: EvaluationScope,
    pub last_cycle: Option<u64>,
    pub sequence: u64,
    pub cycles_without_progress: u64,
    pub replans: u64,
    pub handled_sequence: Option<u64>,
    pub decided_sequence: Option<u64>,
    pub last_delta: ProgressDelta,
    signals: BTreeMap<String, SignalHistory>,
}

impl ProgressState {
    /// A new state does not inherit prior-run progress or refill any runtime budget.
    pub fn new(scope: EvaluationScope) -> Self {
        Self {
            scope,
            last_cycle: None,
            sequence: 0,
            cycles_without_progress: 0,
            replans: 0,
            handled_sequence: None,
            decided_sequence: None,
            last_delta: ProgressDelta::Unknown,
            signals: BTreeMap::new(),
        }
    }
    /// Commits one completed-cycle observation; repeated cycles and suspended work cannot advance counters.
    pub fn observe(
        &mut self,
        scope: &EvaluationScope,
        observation: &ProgressObservation,
    ) -> Result<ProgressDelta> {
        if scope.key != self.scope.key
            || scope.objective_revision != self.scope.objective_revision
            || self.last_cycle.is_some_and(|c| scope.cycle <= c)
        {
            return Err(AgentError::Config("stale progress cycle/scope".into()));
        }
        if observation.signals.len() > 64
            || observation
                .signals
                .keys()
                .any(|id| !observation.comparisons.contains_key(id))
        {
            return Err(AgentError::Config(
                "invalid progress signal observation".into(),
            ));
        }
        super::bounded_value(&serde_json::to_value(observation)?, 65_536)?;
        let mut next = self.clone();
        let mut advanced = false;
        let mut regressed = false;
        let mut known = false;
        for (id, value) in &observation.signals {
            let history = next.signals.entry(id.clone()).or_default();
            match observation.comparisons[id] {
                SignalComparison::Increase => {
                    let Some(value) = value.as_f64().filter(|n| n.is_finite()) else {
                        continue;
                    };
                    known = true;
                    if history.high.is_some_and(|high| value > high) {
                        advanced = true;
                        history.high = Some(value);
                    } else if history
                        .previous
                        .as_ref()
                        .and_then(Value::as_f64)
                        .is_some_and(|p| value < p)
                    {
                        regressed = true;
                    }
                    if history.high.is_none() {
                        history.high = Some(value);
                    }
                    history.previous = Some(json!(value));
                }
                SignalComparison::Change => {
                    known = true;
                    let fingerprint = canonical_identity(value)?;
                    if !history.fingerprints.contains(&fingerprint) {
                        advanced = true;
                        history.fingerprints.insert(fingerprint);
                    } else if history.previous.as_ref().is_some_and(|p| p != value) {
                        regressed = true;
                    }
                    history.previous = Some(value.clone());
                }
            }
        }
        if next.signals.len() > 64
            || next
                .signals
                .values()
                .map(|s| s.fingerprints.len())
                .sum::<usize>()
                > 256
        {
            return Err(AgentError::Config(
                "progress fingerprint capacity exceeded".into(),
            ));
        }
        next.last_delta = if advanced {
            ProgressDelta::Advanced
        } else if regressed {
            ProgressDelta::Regressed
        } else if known {
            ProgressDelta::Unchanged
        } else {
            ProgressDelta::Unknown
        };
        next.cycles_without_progress = if advanced {
            0
        } else {
            next.cycles_without_progress
                .checked_add(1)
                .ok_or_else(|| AgentError::Config("progress counter overflow".into()))?
        };
        next.sequence = next
            .sequence
            .checked_add(1)
            .ok_or_else(|| AgentError::Config("progress sequence overflow".into()))?;
        next.last_cycle = Some(scope.cycle);
        next.scope = scope.clone();
        super::bounded_value(&serde_json::to_value(&next)?, 65_536)?;
        *self = next;
        Ok(self.last_delta)
    }
    /// Completion is tested before stagnation, and one sequence can select at most one intervention.
    pub fn stagnation(
        &mut self,
        policy: Option<&StagnationConfig>,
        complete: bool,
        interaction_available: bool,
    ) -> Result<Option<StagnationAction>> {
        if complete || self.handled_sequence == Some(self.sequence) {
            return Ok(None);
        }
        let Some(policy) = policy else {
            return Ok(None);
        };
        let threshold = policy.max_cycles_without_progress.unwrap_or(3);
        if threshold == 0 {
            return Err(AgentError::Config("zero stagnation threshold".into()));
        }
        if self.cycles_without_progress < u64::from(threshold) {
            return Ok(None);
        }
        let mut action = policy.action.unwrap_or(StagnationAction::Stop);
        if action == StagnationAction::Replan
            && self.replans >= u64::from(policy.max_replans.unwrap_or(0))
        {
            action = policy.on_exhausted.unwrap_or(StagnationAction::Stop);
            if action == StagnationAction::Replan {
                return Err(AgentError::Config(
                    "replan exhaustion cannot request another replan".into(),
                ));
            }
        }
        if action == StagnationAction::AskUser && !interaction_available {
            action = policy.on_unavailable.ok_or_else(|| {
                AgentError::Config("unavailable interaction needs explicit stop/fail".into())
            })?;
            if !matches!(action, StagnationAction::Stop | StagnationAction::Fail) {
                return Err(AgentError::Config(
                    "invalid unavailable interaction fallback".into(),
                ));
            }
        }
        if action == StagnationAction::Replan {
            self.replans = self
                .replans
                .checked_add(1)
                .ok_or_else(|| AgentError::Config("replan overflow".into()))?;
        }
        self.handled_sequence = Some(self.sequence);
        Ok(Some(action))
    }
    /// Restored history is validated before any adapter callback can consume it.
    pub fn restore(value: Value, scope: &EvaluationScope) -> Result<Self> {
        super::bounded_value(&value, 65_536)?;
        let state: Self = serde_json::from_value(value)?;
        if state.scope.key != scope.key
            || state.scope.objective_revision != scope.objective_revision
            || state.signals.len() > 64
            || state
                .signals
                .values()
                .map(|s| s.fingerprints.len())
                .sum::<usize>()
                > 256
            || state.handled_sequence.is_some_and(|s| s > state.sequence)
        {
            return Err(AgentError::Config(
                "incompatible progress checkpoint".into(),
            ));
        }
        Ok(state)
    }
}

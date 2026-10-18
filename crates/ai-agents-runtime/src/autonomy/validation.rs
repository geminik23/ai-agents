//! Bounded resumable observation driver; custom callbacks cannot directly invoke tools or providers.

use super::*;
use ai_agents_core::autonomy::JudgeGate;
use ai_agents_core::{AgentError, Result, ToolCancellationToken, ToolExecutionRecord};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[cfg(test)]
#[path = "validation_clock_tests.rs"]
mod clock_tests;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ValidationObservationRequest {
    Plan {
        id: String,
        objective: String,
    },
    Tool {
        id: String,
        tool: String,
        arguments: Value,
        host_binding: Option<String>,
    },
    Judge {
        id: String,
        config: JudgeGate,
        response: String,
        objective: String,
    },
}
impl ValidationObservationRequest {
    /// IDs are stable within exactly one run/check/attempt.
    pub fn id(&self) -> &str {
        match self {
            Self::Tool { id, .. } | Self::Judge { id, .. } | Self::Plan { id, .. } => id,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ObservationResult {
    Plan { plan: Value },
    Tool { record: Box<ToolExecutionRecord> },
    Judge { score: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletedObservation {
    #[serde(default)]
    pub identity: Option<EvidenceIdentity>,
    pub request: ValidationObservationRequest,
    pub request_identity: String,
    pub effective_identity: String,
    pub result: ObservationResult,
}

pub struct ValidationInput<'a> {
    pub scope: &'a EvaluationScope,
    pub identity: &'a EvidenceIdentity,
    pub config: &'a Value,
    pub evidence: &'a EvaluationEvidence,
    pub previous: &'a Value,
    pub observations: &'a BTreeMap<String, CompletedObservation>,
}

pub enum ValidationDecision {
    Complete {
        outcome: GateOutcome,
        reason: String,
        metrics: Value,
        evidence_refs: Vec<String>,
    },
    NeedObservations {
        requests: Vec<ValidationObservationRequest>,
        checkpoint: Value,
    },
    AwaitExternal {
        binding: String,
        checkpoint: Value,
    },
}

/// A signal is data bound to a persisted wait, never an approval or permission to skip gates.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalWait {
    pub check_id: String,
    pub request_id: String,
    pub binding: HostConditionBinding,
    pub identity: EvidenceIdentity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalSignal {
    pub request_id: String,
    pub event_id: String,
    pub target_revision: Option<String>,
    pub payload: Value,
}

/// A safe observation pause retains the actual executor request and exact response binding, not executable authority.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationObservationPause {
    pub observation: String,
    pub request: ai_agents_core::ToolExecutionRequest,
    pub pending: TaskPendingRequest,
    pub executor: Option<Value>,
    pub before: EvaluationScope,
    #[serde(default)]
    pub authorized: Vec<Value>,
    #[serde(default)]
    pub group: Option<Value>,
}

/// In-flight records are recovery-required after a crash, never automatically reexecuted.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationDriverState {
    pub version: u32,
    pub binding_identity: String,
    pub check_deadline: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(skip)]
    check_projection: Option<std::time::Instant>,
    pub host_authorities: BTreeMap<String, HostValidationBinding>,
    pub check_id: String,
    pub adapter: String,
    pub contract_version: u32,
    pub identity: EvidenceIdentity,
    pub rounds: u32,
    pub checkpoint: Value,
    pub requests: BTreeMap<String, ValidationObservationRequest>,
    pub completed: BTreeMap<String, CompletedObservation>,
    pub in_flight: Option<String>,
    #[serde(default)]
    pub parked: Option<Box<ValidationObservationPause>>,
    pub pending_batch: Vec<String>,
    /// A pre-dispatch capacity wait preserves the pending batch and original attempt clocks without recording an uncertain invocation.
    #[serde(default)]
    pub capacity_reason: Option<String>,
    #[serde(skip)]
    capacity_not_invoked: Option<String>,
    #[serde(skip)]
    publication_identity: Option<String>,
    pub wait: Option<ExternalWait>,
    pub external_events: BTreeMap<String, String>,
    pub result: Option<ValidationResult>,
    pub result_identity: Option<String>,
    #[serde(default)]
    pub publication_scope: Option<EvaluationScope>,
    #[serde(default)]
    input_dependencies: Option<BTreeMap<String, String>>,
    #[serde(skip)]
    retained: Option<std::sync::Arc<RetainedValidation>>,
}

/// Live attempt bounds cannot be reconstructed or extended by deserializing a checkpoint.
#[derive(Debug)]
struct RetainedValidation {
    identity: EvidenceIdentity,
    binding_identity: String,
    deadline: Option<chrono::DateTime<chrono::Utc>>,
    projection: Option<std::time::Instant>,
}

#[derive(Debug, Clone, Copy)]
pub struct ValidationDriverLimits {
    pub max_evaluation_rounds: u32,
    pub max_observations_per_round: usize,
}
impl Default for ValidationDriverLimits {
    /// Defaults are finite; host ceilings may only lower these per-attempt maxima.
    fn default() -> Self {
        Self {
            max_evaluation_rounds: 32,
            max_observations_per_round: 16,
        }
    }
}

/// Journal acknowledgement is required before invocation and before publishing completed observation evidence.
#[async_trait]
pub trait ValidationJournal: Send + Sync {
    /// Persist exact state under the enclosing run's owner/revision boundary.
    async fn checkpoint(&self, state: &ValidationDriverState) -> Result<()>;
}

/// The production implementation uses RuntimeAgent's shared executor and frozen judge routing.
#[async_trait]
pub trait ValidationObservationExecutor: Send + Sync {
    /// Returns a deterministic capacity wait before dispatch; final execution must still perform atomic admission.
    async fn capacity_reason(
        &self,
        _request: &ValidationObservationRequest,
    ) -> Result<Option<String>> {
        Ok(None)
    }
    /// Captures authoritative freshness without changing the immutable check/request binding.
    fn capture_scope(&self, scope: &EvaluationScope) -> Result<EvaluationScope> {
        Ok(scope.clone())
    }
    /// Performs deterministic binding/alias preflight before a journal can mark an operation dispatched.
    fn preflight(
        &self,
        _request: &ValidationObservationRequest,
        _identity: &EvidenceIdentity,
    ) -> Result<()> {
        Ok(())
    }
    /// Only an executor owning an exact acknowledged park may convert suspension into resumable progress.
    fn take_suspension(
        &self,
        _request: &ValidationObservationRequest,
        _identity: &EvidenceIdentity,
        _before: &EvaluationScope,
    ) -> Result<Option<ValidationObservationPause>> {
        Ok(None)
    }
    /// Framework composition bridges can acknowledge captured child progress before returning park data; existing synchronous executors retain their hook.
    async fn acknowledge_suspension(
        &self,
        request: &ValidationObservationRequest,
        identity: &EvidenceIdentity,
        before: &EvaluationScope,
    ) -> Result<Option<ValidationObservationPause>> {
        self.take_suspension(request, identity, before)
    }
    /// Must preserve shared policy/HITL/admission and return only authoritative managed observations.
    async fn execute(
        &self,
        request: &ValidationObservationRequest,
        identity: &EvidenceIdentity,
    ) -> Result<ObservationResult>;
}

/// Immutable invocation context keeps host bounds, cancellation and persistence explicit.
pub struct ValidationDriveContext<'a> {
    pub scope: &'a EvaluationScope,
    pub evidence: &'a EvaluationEvidence,
    pub extensions: &'a FrozenAutonomyExtensions,
    pub executor: &'a dyn ValidationObservationExecutor,
    pub journal: &'a dyn ValidationJournal,
    pub limits: ValidationDriverLimits,
    pub cancellation: &'a ToolCancellationToken,
    pub deadline: chrono::DateTime<chrono::Utc>,
}

pub enum ValidationDriverOutcome {
    Complete(ValidationResult),
    AwaitExternal(ExternalWait),
    Suspended,
}

impl ValidationDriverState {
    /// Starts a fresh explicit attempt; earlier observations cannot be implicitly relabelled as fresh.
    pub fn new(binding: &BoundValidationCheck, identity: EvidenceIdentity) -> Result<Self> {
        if binding.check.adapter == "internal.replan"
            && identity.attempt.as_deref()
                != binding
                    .check
                    .config
                    .as_ref()
                    .and_then(|c| c.get("intervention_id"))
                    .and_then(Value::as_str)
        {
            return Err(AgentError::Config(
                "replan attempt must match the selected intervention".into(),
            ));
        }
        if identity.config_identity != binding.config_identity
            || identity.attempt.as_ref().is_none_or(String::is_empty)
        {
            return Err(AgentError::Config(
                "validation attempt identity mismatch".into(),
            ));
        }
        let mut state = Self {
            version: 1,
            binding_identity: canonical_identity(&serde_json::to_value(&binding.check)?)?,
            check_deadline: binding
                .check
                .timeout_seconds
                .map(|seconds| {
                    if seconds == 0 {
                        return Err(AgentError::Config("zero validation timeout".into()));
                    }
                    let duration =
                        chrono::Duration::try_seconds(i64::try_from(seconds).map_err(|_| {
                            AgentError::Config("validation timeout overflow".into())
                        })?)
                        .ok_or_else(|| AgentError::Config("validation timeout overflow".into()))?;
                    chrono::Utc::now()
                        .checked_add_signed(duration)
                        .ok_or_else(|| AgentError::Config("validation deadline overflow".into()))
                })
                .transpose()?,
            check_projection: binding.check.timeout_seconds.and_then(|seconds| {
                std::time::Instant::now().checked_add(std::time::Duration::from_secs(seconds))
            }),
            host_authorities: BTreeMap::new(),
            check_id: binding.check.id.clone(),
            adapter: binding.check.adapter.clone(),
            contract_version: binding.check.contract_version.unwrap_or(1),
            identity,
            rounds: 0,
            checkpoint: Value::Null,
            requests: BTreeMap::new(),
            completed: BTreeMap::new(),
            in_flight: None,
            capacity_reason: None,
            capacity_not_invoked: None,
            parked: None,
            pending_batch: vec![],
            publication_identity: None,
            wait: None,
            external_events: BTreeMap::new(),
            result: None,
            result_identity: None,
            publication_scope: None,
            input_dependencies: None,
            retained: None,
        };
        state.retained = Some(std::sync::Arc::new(RetainedValidation {
            identity: state.identity.clone(),
            binding_identity: state.binding_identity.clone(),
            deadline: state.check_deadline,
            projection: state.check_projection,
        }));
        Ok(state)
    }
    /// Decodes and validates audit data only; executable same-process continuation requires `restore_retained` and a live predecessor.
    pub fn restore(
        value: Value,
        binding: &BoundValidationCheck,
        scope: &EvaluationScope,
        extensions: &FrozenAutonomyExtensions,
    ) -> Result<Self> {
        bounded_value(&value, 262_144)?;
        let mut state: Self = serde_json::from_value(value)?;
        state.check_projection = state.check_deadline.and_then(|deadline| {
            std::time::Instant::now()
                .checked_add((deadline - chrono::Utc::now()).to_std().unwrap_or_default())
        });
        state.validate(binding, scope, extensions)?;
        if state.in_flight.is_some() {
            return Err(AgentError::Config(
                "validation observation requires explicit effect reconciliation".into(),
            ));
        }
        Ok(state)
    }
    /// Restores exactly one live checkpoint without granting restart, rewinding work or extending its original monotonic bound.
    pub fn restore_retained(
        value: Value,
        predecessor: &Self,
        binding: &BoundValidationCheck,
        scope: &EvaluationScope,
        extensions: &FrozenAutonomyExtensions,
    ) -> Result<Self> {
        predecessor.check_retained()?;
        if value != serde_json::to_value(predecessor)? {
            return Err(AgentError::Config(
                "retained validation checkpoint differs from live predecessor".into(),
            ));
        }
        let mut state = Self::restore(value, binding, scope, extensions)?;
        state.retained = predecessor.retained.clone();
        state.check_projection = predecessor.check_projection;
        Ok(state)
    }

    // Public mutable checkpoint fields cannot mint a fresh lifetime or change a retained attempt's binding.
    fn check_retained(&self) -> Result<()> {
        let retained = self.retained.as_ref().ok_or_else(|| AgentError::Config(
            "validation execution requires a live predecessor; serialized restore is audit-only".into()))?;
        if retained.identity != self.identity
            || retained.binding_identity != self.binding_identity
            || retained.deadline != self.check_deadline
            || retained.projection != self.check_projection
        {
            return Err(AgentError::Config(
                "retained validation identity or original deadline changed".into(),
            ));
        }
        Ok(())
    }

    /// Checks original time bounds after awaited persistence; a successful receipt does not extend callback admission.
    pub(crate) fn expired(&self) -> bool {
        self.check_deadline
            .is_some_and(|deadline| chrono::Utc::now() >= deadline)
            || self
                .check_projection
                .is_some_and(|deadline| std::time::Instant::now() >= deadline)
    }

    // Publication must still be timely and bound after acknowledgement; saved late data remains audit-only.
    fn check_live(
        &mut self,
        context: &ValidationDriveContext<'_>,
        deadline: chrono::DateTime<chrono::Utc>,
    ) -> Result<()> {
        if context.cancellation.is_cancelled() || self.expired() || chrono::Utc::now() >= deadline {
            self.publication_identity = None;
            return Err(AgentError::Other(
                "validation cancelled or deadline exceeded".into(),
            ));
        }
        Ok(())
    }

    // A snapshot may advance mutation/revision data, but never change the check's run, config or attempt binding.
    fn capture_publication(
        &mut self,
        context: &ValidationDriveContext<'_>,
    ) -> Result<EvaluationScope> {
        context
            .extensions
            .ensure_driver_target(self)
            .inspect_err(|_| self.publication_identity = None)?;
        let current = context
            .executor
            .capture_scope(context.scope)
            .inspect_err(|_| self.publication_identity = None)?;
        if current.key != context.scope.key
            || current.objective_revision != context.scope.objective_revision
            || current.cycle != context.scope.cycle
            || current.stage != context.scope.stage
            || current.validation_bindings != context.scope.validation_bindings
            || current.validation_attempts != context.scope.validation_attempts
            || current.validation_dependencies != context.scope.validation_dependencies
            || current.mutation_generation < context.scope.mutation_generation
            || !self.identity.eligible(&current, false, false)
        {
            self.publication_identity = None;
            return Err(AgentError::Config(
                "validation host target or scope changed".into(),
            ));
        }
        Ok(current)
    }

    /// Validates signal data without consuming it or requiring a storage claim.
    pub fn validate_external(
        &self,
        signal: &ExternalSignal,
        extensions: &FrozenAutonomyExtensions,
    ) -> Result<()> {
        bounded_value(&serde_json::to_value(signal)?, 65_536)?;
        let mut projected = self.clone();
        projected.external_events.insert(
            signal.event_id.clone(),
            canonical_identity(&serde_json::to_value(signal)?)?,
        );
        projected.checkpoint = json!({"adapter_state":self.checkpoint,"external_signal":signal.payload,"condition_request":signal.request_id});
        projected.wait = None;
        bounded_value(&serde_json::to_value(projected)?, 262_144)?;
        let wait = self
            .wait
            .as_ref()
            .ok_or_else(|| AgentError::Config("no pending external condition".into()))?;
        if signal.request_id != wait.request_id
            || signal.event_id.is_empty()
            || signal.target_revision != wait.identity.target_revision
            || extensions.condition(&wait.binding.id) != Some(&wait.binding)
            || !wait.binding.shape.accepts(&signal.payload)
            || self.external_events.contains_key(&signal.event_id)
            || self.external_events.len() >= 64
        {
            return Err(AgentError::Config(
                "invalid, duplicate or incompatible external signal".into(),
            ));
        }
        Ok(())
    }

    /// A signal is schema-validated, deduplicated and consumed once against the frozen wait binding.
    pub fn accept_external(
        &mut self,
        signal: ExternalSignal,
        extensions: &FrozenAutonomyExtensions,
    ) -> Result<()> {
        if !self.publication_acknowledged() {
            return Err(AgentError::Persistence(
                "external wait issuance must be acknowledged before signal consumption".into(),
            ));
        }
        self.validate_external(&signal, extensions)?;
        let wait = self
            .wait
            .as_ref()
            .ok_or_else(|| AgentError::Config("no pending external condition".into()))?;
        if signal.request_id != wait.request_id
            || signal.event_id.is_empty()
            || signal.target_revision != wait.identity.target_revision
            || extensions.condition(&wait.binding.id) != Some(&wait.binding)
            || !wait.binding.shape.accepts(&signal.payload)
            || self.external_events.contains_key(&signal.event_id)
            || self.external_events.len() >= 64
        {
            return Err(AgentError::Config(
                "invalid, duplicate or incompatible external signal".into(),
            ));
        }
        let digest = canonical_identity(&serde_json::to_value(&signal)?)?;
        self.external_events.insert(signal.event_id, digest);
        self.checkpoint = json!({"adapter_state":self.checkpoint,"external_signal":signal.payload,"condition_request":wait.request_id});
        self.wait = None;
        Ok(())
    }
    // Runtime-owned validation runs before callbacks, including completed cached results and restored external waits.
    fn validate(
        &self,
        binding: &BoundValidationCheck,
        scope: &EvaluationScope,
        extensions: &FrozenAutonomyExtensions,
    ) -> Result<()> {
        if binding
            .stage
            .as_ref()
            .is_some_and(|stage| self.identity.stage.as_ref() != Some(stage))
            || scope.validation_bindings.get(&self.check_id) != Some(&self.identity.config_identity)
            || scope.validation_attempts.get(&self.check_id) != self.identity.attempt.as_ref()
            || (binding.available
                && binding.adapter.descriptor().requires_host_revision
                && (self.identity.target.is_none() || self.identity.target_revision.is_none()))
        {
            return Err(AgentError::Config(
                "validation stage/config/attempt/target mismatch".into(),
            ));
        }
        if self.binding_identity != canonical_identity(&serde_json::to_value(&binding.check)?)?
            || self.check_deadline.is_some() != binding.check.timeout_seconds.is_some()
        {
            return Err(AgentError::Config(
                "validation binding/timeout changed".into(),
            ));
        }
        if self.version != 1
            || self.check_id != binding.check.id
            || self.adapter != binding.check.adapter
            || self.contract_version != binding.check.contract_version.unwrap_or(1)
            || self.identity.config_identity != binding.config_identity
            || !self.identity.eligible(scope, false, true)
            || self.identity.attempt.as_ref().is_none_or(String::is_empty)
            || self.requests.len() > 512
            || self.completed.len() > self.requests.len()
            || self.rounds > 32
        {
            return Err(AgentError::Config(
                "invalid validation checkpoint identity/bounds".into(),
            ));
        }
        bounded_value(&serde_json::to_value(self)?, 262_144)?;
        for (id, request) in &self.requests {
            validate_request(request, binding)?;
            if let ValidationObservationRequest::Tool {
                host_binding: Some(authority),
                tool,
                arguments,
                ..
            } = request
            {
                let installed = extensions.host_binding(authority).ok_or_else(|| {
                    AgentError::Config("host authority missing on restore".into())
                })?;
                if installed.validator != self.adapter
                    || installed.version != self.contract_version
                    || installed.tool != *tool
                    || installed.arguments != *arguments
                    || self.host_authorities.get(id) != Some(installed)
                {
                    return Err(AgentError::Config(
                        "host validation authority changed".into(),
                    ));
                }
            }
            if request.id() != id || id.is_empty() {
                return Err(AgentError::Config("invalid observation ID".into()));
            }
            if let Some(result) = self.completed.get(id) {
                if result.request != *request
                    || result.request_identity != request_identity(request, &self.identity)?
                {
                    return Err(AgentError::Config(
                        "changed cached observation identity".into(),
                    ));
                }
                validate_observation_result(request, &result.result)?;
                if result.effective_identity != effective_identity(&result.result)? {
                    return Err(AgentError::Config(
                        "changed effective execution identity".into(),
                    ));
                }
            }
        }
        let batch: std::collections::BTreeSet<_> = self.pending_batch.iter().collect();
        if batch.len() != self.pending_batch.len()
            || batch.len() > 16
            || self
                .pending_batch
                .iter()
                .any(|id| !self.requests.contains_key(id) || self.completed.contains_key(id))
            || self
                .in_flight
                .as_ref()
                .is_some_and(|id| self.pending_batch.first() != Some(id))
            || ((self.result.is_some() || self.wait.is_some()) && !self.pending_batch.is_empty())
        {
            return Err(AgentError::Config(
                "invalid pending observation batch".into(),
            ));
        }
        if self
            .completed
            .keys()
            .any(|id| !self.requests.contains_key(id))
            || self.in_flight.as_ref().is_some_and(|id| {
                !self.requests.contains_key(id) || self.completed.contains_key(id)
            })
        {
            return Err(AgentError::Config("orphaned observation result".into()));
        }
        if let Some(wait) = &self.wait
            && (wait.check_id != self.check_id
                || wait.identity != self.identity
                || extensions.condition(&wait.binding.id) != Some(&wait.binding))
        {
            return Err(AgentError::Config("changed external wait binding".into()));
        }
        let mut publication_identity = self.identity.clone();
        if let Some(publication) = &self.publication_scope {
            if publication.key != scope.key
                || publication.objective_revision != scope.objective_revision
                || publication.cycle != scope.cycle
                || publication.stage != scope.stage
                || publication.mutation_generation < self.identity.mutation_generation
            {
                return Err(AgentError::Config(
                    "validation publication scope changed binding".into(),
                ));
            }
            publication_identity.mutation_generation = publication.mutation_generation;
        }
        publication_identity.validation_dependencies =
            self.input_dependencies.clone().unwrap_or_default();
        if let Some(result) = &self.result
            && (result.identity != publication_identity
                || result.check_id != self.check_id
                || self.result_identity.as_ref()
                    != Some(&canonical_identity(&serde_json::to_value(result)?)?))
        {
            return Err(AgentError::Config(
                "changed validation result identity".into(),
            ));
        }
        if let Some(parked) = &self.parked {
            let Some(ValidationObservationRequest::Tool {
                tool, arguments, ..
            }) = self.requests.get(&parked.observation)
            else {
                return Err(AgentError::Config(
                    "parked observation has no tool request".into(),
                ));
            };
            if self.in_flight.is_some()
                || self.pending_batch.first() != Some(&parked.observation)
                || parked.request.call_id.is_empty()
                || !matches!(parked.request.source, ai_agents_core::ToolCallSource::Task)
                || parked.request.requested_name != *tool
                || parked.request.arguments != *arguments
                || self.wait.is_some()
                || self.result.is_some()
                || parked.before.key != self.identity.key
                || parked.before.objective_revision != self.identity.objective_revision
                || parked.before.cycle != self.identity.cycle
                || parked.before.stage != self.identity.stage
            {
                return Err(AgentError::Config(
                    "invalid parked observation binding".into(),
                ));
            }
            super::authority::validate_observation_pause(parked)?;
        }
        Ok(())
    }

    /// Only a live zero-admission witness can downgrade the exact in-flight logical observation into a capacity wait.
    pub(crate) fn capacity_noninvocation(&self) -> Option<&str> {
        self.capacity_not_invoked.as_deref().filter(|id| {
            self.capacity_reason.is_some()
                && self.in_flight.is_none()
                && self.pending_batch.first().map(String::as_str) == Some(*id)
        })
    }

    /// Known-safe logical waits remain separate from actual provider/tool reservations and cannot certify those effects stopped.
    pub(crate) fn wait_reservation(&self) -> Result<Option<String>> {
        let observation = self
            .parked
            .as_ref()
            .map(|parked| parked.observation.as_str())
            .or_else(|| {
                self.capacity_reason
                    .as_ref()
                    .and_then(|_| self.pending_batch.first().map(String::as_str))
            });
        observation
            .map(|id| super::journal::observation_reservation_id(self, id))
            .transpose()
    }

    /// Retires only the bound logical wait; capacity preflight may have no reservation, and actual uncertain effects remain protected.
    pub(crate) fn retire_wait_reservation(
        &self,
        payload: &mut TaskCheckpointPayload,
    ) -> Result<()> {
        let Some(id) = self.wait_reservation()? else {
            return Ok(());
        };
        let Some(reservation) = payload
            .reservations
            .iter_mut()
            .find(|reservation| reservation.id == id)
        else {
            return if self.parked.is_none() {
                Ok(())
            } else {
                Err(TaskRunStorageError::InvalidCheckpoint.into())
            };
        };
        if reservation.state == TaskEffectState::Completed {
            return Ok(());
        }
        if reservation.state != TaskEffectState::Suspended
            && !(self.parked.is_some() && reservation.state == TaskEffectState::Dispatched)
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        reservation.state = TaskEffectState::Completed;
        reservation.result = Some(json!({"cancelled":true,"origin":"validation"}));
        Ok(())
    }

    /// Collection is allowed only for the exact state acknowledged by the journal, not a locally installed failed write.
    pub fn publication_acknowledged(&self) -> bool {
        self.publication_identity
            .as_ref()
            .is_some_and(|digest| serde_json::to_string(self).ok().as_ref() == Some(digest))
    }

    /// A failed acknowledgement clears publication authority, including cached terminal/wait reentry.
    pub(crate) async fn acknowledge(&mut self, journal: &dyn ValidationJournal) -> Result<()> {
        self.publication_identity = None;
        journal.checkpoint(self).await?;
        self.publication_identity = Some(serde_json::to_string(self)?);
        Ok(())
    }

    // Pinned proof must survive every effect admission, not merely the later callback or result publication.
    fn check_input_dependencies(
        &self,
        context: &ValidationDriveContext<'_>,
        scope: &EvaluationScope,
    ) -> Result<()> {
        if let Some(dependencies) = &self.input_dependencies {
            let mut identity = self.identity.clone();
            identity.validation_dependencies = dependencies.clone();
            if !validation_dependencies_current(&identity, &context.evidence.validations, scope) {
                return Err(AgentError::Config(
                    "validation input proof is no longer eligible".into(),
                ));
            }
        }
        Ok(())
    }

    // Retain actual source objects for all transitive proof targets through the executor's own admission waits.
    fn observation_admission(
        &self,
        context: &ValidationDriveContext<'_>,
        scope: &EvaluationScope,
    ) -> Result<std::sync::Arc<dyn TaskFinalAdmission>> {
        let mut identities = vec![(self.check_id.clone(), self.identity.clone())];
        let mut pending: Vec<_> = self
            .input_dependencies
            .iter()
            .flat_map(|inputs| inputs.keys().cloned())
            .collect();
        let mut seen = std::collections::BTreeSet::new();
        while let Some(id) = pending.pop() {
            if !seen.insert(id.clone()) {
                continue;
            }
            if seen.len() > 256 {
                return Err(AgentError::Config(
                    "validation proof dependency bound exceeded".into(),
                ));
            }
            let result = latest_validation(&context.evidence.validations, &id, scope)
                .ok_or_else(|| AgentError::Config("validation input proof unavailable".into()))?;
            pending.extend(result.identity.validation_dependencies.keys().cloned());
            identities.push((id, result.identity.clone()));
        }
        context.extensions.observation_admission(&identities)
    }

    // Finishes a persisted observation batch under the original proof and clocks; in-flight work is never automatically replayed.
    async fn execute_pending(
        &mut self,
        binding: &BoundValidationCheck,
        context: &ValidationDriveContext<'_>,
        deadline: chrono::DateTime<chrono::Utc>,
    ) -> Result<bool> {
        while let Some(id) = self.pending_batch.first().cloned() {
            if context.cancellation.is_cancelled() || chrono::Utc::now() >= deadline {
                return Err(AgentError::Other(
                    "validation cancelled or expired before admission".into(),
                ));
            }
            self.check_live(context, deadline)?;
            let publication = self.capture_publication(context)?;
            self.check_input_dependencies(context, &publication)?;
            context
                .executor
                .preflight(&self.requests[&id], &self.identity)?;
            if let Some(reason) = context
                .executor
                .capacity_reason(&self.requests[&id])
                .await?
            {
                self.capacity_reason = Some(reason);
                self.acknowledge(context.journal).await?;
                return Ok(true);
            }
            self.capacity_reason = None;
            self.publication_identity = None;
            let resumed = self.parked.take();
            self.in_flight = Some(id.clone());
            self.validate(binding, context.scope, context.extensions)?;
            context.journal.checkpoint(self).await?;
            if context.cancellation.is_cancelled() || chrono::Utc::now() >= deadline {
                return Err(AgentError::Other(
                    "validation interrupted after admission acknowledgement".into(),
                ));
            }
            self.check_live(context, deadline)?;
            let publication = self.capture_publication(context)?;
            self.check_input_dependencies(context, &publication)?;
            let request = self.requests[&id].clone();
            let before = match &resumed {
                Some(parked) => parked.before.clone(),
                None => context.executor.capture_scope(context.scope)?,
            };
            let projected = std::time::Instant::now()
                .checked_add((deadline - chrono::Utc::now()).to_std().unwrap_or_default())
                .ok_or_else(|| {
                    AgentError::Config("validation invocation deadline overflow".into())
                })?;
            let projected = self
                .check_projection
                .map_or(projected, |check| check.min(projected));
            let bounds = ai_agents_llm::InvocationBounds::new(deadline.into(), projected);
            let guard = self.observation_admission(context, &publication)?;
            let witness = std::sync::Arc::new(super::invocation::CapacityWitness::default());
            let mut future = Box::pin(ai_agents_llm::scope_invocation_evidence(
                vec![guard],
                ai_agents_llm::scope_invocation_bounds(
                    Some(bounds),
                    super::invocation::scope_capacity_witness(
                        witness.clone(),
                        context.executor.execute(&request, &self.identity),
                    ),
                ),
            ));
            let result = loop {
                tokio::select! { result = &mut future => break result, _ = tokio::time::sleep(std::time::Duration::from_millis(10)) => {
                    if context.cancellation.is_cancelled() || self.expired() || chrono::Utc::now() >= deadline { return Err(AgentError::Other("validation interrupted with uncertain effect".into())); }
                } }
            };
            drop(future);
            let result = match result {
                Err(AgentError::TaskSuspended(_)) => {
                    let mut parked = context
                        .executor
                        .acknowledge_suspension(&request, &self.identity, &before)
                        .await?
                        .ok_or_else(|| {
                            AgentError::Config("observation suspension has no live custody".into())
                        })?;
                    if let Some(previous) = &resumed {
                        parked.authorized = previous.authorized.clone();
                    }
                    self.in_flight = None;
                    self.parked = Some(Box::new(parked));
                    self.validate(binding, context.scope, context.extensions)?;
                    self.acknowledge(context.journal).await?;
                    return Ok(true);
                }
                Err(error) => {
                    let capacity = witness.noninvocation().filter(|_| {
                        super::current_execution()
                            .is_some_and(|execution| !execution.has_storage_failure())
                    });
                    if let Some(reason) = capacity {
                        self.check_live(context, deadline)?;
                        self.in_flight = None;
                        self.capacity_not_invoked = Some(id.clone());
                        self.capacity_reason = Some(reason.clone());
                        self.acknowledge(context.journal).await?;
                        if let Some(execution) = super::current_execution() {
                            execution.clear_capacity_stop(&reason);
                        }
                        return Ok(true);
                    }
                    return Err(error);
                }
                Ok(result) => result,
            };
            validate_observation_result(&request, &result)?;
            let after = context.executor.capture_scope(context.scope)?;
            if after.key != before.key
                || after.objective_revision != before.objective_revision
                || after.cycle != before.cycle
                || after.stage != before.stage
                || after.mutation_generation < before.mutation_generation
            {
                return Err(AgentError::Config(
                    "observation freshness changed execution scope".into(),
                ));
            }
            self.publication_scope = Some(after.clone());
            let mut observed = self.identity.clone();
            observed.sequence = self
                .identity
                .sequence
                .checked_add(self.completed.len() as u64)
                .ok_or_else(|| AgentError::Config("observation sequence overflow".into()))?;
            observed.mutation_generation = before.mutation_generation;
            observed.validation_dependencies = self.input_dependencies.clone().unwrap_or_default();
            if matches!(&result, ObservationResult::Tool { record } if record.executed
                && record.metadata.get("classification").and_then(|value| value.get("read_only")).and_then(Value::as_bool) == Some(false))
            {
                observed.mutation_generation = after.mutation_generation;
            }
            let completed = CompletedObservation {
                identity: Some(observed),
                request_identity: request_identity(&request, &self.identity)?,
                effective_identity: effective_identity(&result)?,
                request,
                result,
            };
            self.completed.insert(id, completed);
            self.in_flight = None;
            self.pending_batch.remove(0);
            context.journal.checkpoint(self).await?;
        }
        Ok(false)
    }

    /// Refuses stale attempt rewinds and preserves every acknowledged request/result before a journal can admit another effect.
    pub(crate) fn validate_journal_successor(&self, old: &Self) -> Result<()> {
        if self.identity.attempt != old.identity.attempt {
            if old.in_flight.is_some()
                || old.wait.is_some()
                || old.parked.is_some()
                || old.result.is_none()
                || !self.completed.is_empty()
                || self.in_flight.is_some()
            {
                return Err(AgentError::Config(
                    "new validation attempt requires settled prior state".into(),
                ));
            }
            return Ok(());
        }
        if self.identity != old.identity
            || self.binding_identity != old.binding_identity
            || self.check_deadline != old.check_deadline
            || self.rounds < old.rounds
            || old
                .requests
                .iter()
                .any(|(id, r)| self.requests.get(id) != Some(r))
            || old.completed.iter().any(|(id, r)| {
                self.completed.get(id).is_none_or(|new| {
                    serde_json::to_value(new).ok() != serde_json::to_value(r).ok()
                })
            })
            || old
                .external_events
                .iter()
                .any(|(id, event)| self.external_events.get(id) != Some(event))
            || (old.result.is_some()
                && serde_json::to_value(&self.result)? != serde_json::to_value(&old.result)?)
            || old.in_flight.as_ref().is_some_and(|id| {
                !(self.capacity_reason.is_some()
                    && self.capacity_not_invoked.as_ref() == Some(id)
                    && self.in_flight.is_none())
                    && !self.completed.contains_key(id)
                    && self
                        .parked
                        .as_ref()
                        .is_none_or(|parked| &parked.observation != id)
            })
        {
            return Err(AgentError::Config(
                "stale or uncertain validation journal continuation".into(),
            ));
        }
        if let Some(previous) = &old.parked {
            if let Some(next) = &self.parked {
                if next.observation != previous.observation
                    || serde_json::to_value(&next.request)?
                        != serde_json::to_value(&previous.request)?
                    || serde_json::to_value(&next.before)?
                        != serde_json::to_value(&previous.before)?
                    || next.authorized.len() < previous.authorized.len()
                    || next.authorized[..previous.authorized.len()] != previous.authorized
                {
                    return Err(AgentError::Config(
                        "parked observation custody was replaced".into(),
                    ));
                }
            } else if !self.completed.contains_key(&previous.observation)
                && self.in_flight.as_ref() != Some(&previous.observation)
            {
                return Err(AgentError::Config("parked observation was skipped".into()));
            }
        }
        if !old.pending_batch.is_empty()
            && self.pending_batch
                != old
                    .pending_batch
                    .iter()
                    .filter(|id| !self.completed.contains_key(*id))
                    .cloned()
                    .collect::<Vec<_>>()
        {
            return Err(AgentError::Config(
                "pending observation batch was rewound or skipped".into(),
            ));
        }
        if let Some(wait) = &old.wait
            && self.wait.is_none()
            && !self.external_events.values().any(|event| {
                serde_json::from_str::<Value>(event).ok().is_some_and(|e| {
                    e.get("request_id").and_then(Value::as_str) == Some(&wait.request_id)
                })
            })
        {
            return Err(AgentError::Config(
                "external wait has no accepted signal".into(),
            ));
        }
        Ok(())
    }

    /// Finite callbacks request managed observations; every effect has an acknowledged pre-dispatch journal record.
    /// Callback and publication admission recheck original clocks and host targets after each awaited acknowledgement.
    pub async fn drive(
        &mut self,
        binding: &BoundValidationCheck,
        context: &ValidationDriveContext<'_>,
    ) -> Result<ValidationDriverOutcome> {
        self.publication_identity = None;
        self.check_retained()?;
        let ValidationDriveContext {
            scope,
            evidence,
            extensions,
            executor: _,
            journal,
            limits,
            cancellation: _,
            deadline,
        } = *context;
        let limits = ValidationDriverLimits {
            max_evaluation_rounds: limits
                .max_evaluation_rounds
                .min(binding.check.max_evaluation_rounds.unwrap_or(32)),
            max_observations_per_round: limits
                .max_observations_per_round
                .min(binding.check.max_observations_per_round.unwrap_or(16) as usize),
        };
        let deadline = self
            .check_deadline
            .map_or(deadline, |check| check.min(deadline));
        if limits.max_evaluation_rounds == 0
            || limits.max_evaluation_rounds > 32
            || limits.max_observations_per_round == 0
            || limits.max_observations_per_round > 16
        {
            return Err(AgentError::Config(
                "invalid host-bounded validation limits".into(),
            ));
        }
        self.validate(binding, scope, extensions)?;
        extensions.ensure_driver_target(self)?;
        if self.in_flight.is_some() {
            return Err(AgentError::Config("uncertain validation effect".into()));
        }
        loop {
            self.check_live(context, deadline)?;
            if self.result.is_some() {
                self.acknowledge(journal).await?;
                self.check_live(context, deadline)?;
                let current = self.capture_publication(context)?;
                let result = self.result.as_ref().unwrap();
                if !result.identity.eligible(&current, false, true)
                    || !validation_dependencies_current(
                        &result.identity,
                        &evidence.validations,
                        &current,
                    )
                {
                    self.publication_identity = None;
                    return Err(AgentError::Config(
                        "cached validation evidence is stale".into(),
                    ));
                }
                return Ok(ValidationDriverOutcome::Complete(result.clone()));
            }
            if self.wait.is_some() {
                self.acknowledge(journal).await?;
                self.check_live(context, deadline)?;
                self.capture_publication(context)?;
                return Ok(ValidationDriverOutcome::AwaitExternal(
                    self.wait.as_ref().unwrap().clone(),
                ));
            }
            if !self.pending_batch.is_empty() {
                if self.execute_pending(binding, context, deadline).await? {
                    return Ok(ValidationDriverOutcome::Suspended);
                }
                continue;
            }
            if self.rounds >= limits.max_evaluation_rounds {
                return Err(AgentError::Config("validation callback round limit".into()));
            }
            self.rounds += 1;
            journal.checkpoint(self).await?;
            self.check_live(context, deadline)?;
            let publication = self.capture_publication(context)?;
            self.publication_scope = Some(publication.clone());
            let config = binding.check.config.clone().unwrap_or_else(|| json!({}));
            // Declared inputs are pinned once per attempt; callback reentry cannot adopt replacement proof under an old attempt.
            let dependencies: Option<BTreeMap<_, _>> = binding
                .dependencies()
                .into_iter()
                .map(|id| {
                    latest_validation(&evidence.validations, &id, &publication)
                        .filter(|r| r.outcome != GateOutcome::Unknown)
                        .and_then(|r| r.proof_identity().ok())
                        .map(|proof| (id, proof))
                })
                .collect();
            if self.input_dependencies.is_some() && self.input_dependencies != dependencies {
                return Err(AgentError::Config(
                    "validation input proof changed within attempt".into(),
                ));
            }
            self.input_dependencies = dependencies.clone();
            let decision = if !binding.available || dependencies.is_none() {
                ValidationDecision::Complete {
                    outcome: GateOutcome::Unknown,
                    reason: if !binding.available {
                        "optional_capability_unavailable"
                    } else {
                        "input_evidence_unavailable"
                    }
                    .into(),
                    metrics: json!({}),
                    evidence_refs: vec![],
                }
            } else {
                binding.adapter.evaluate(&ValidationInput {
                    scope: &publication,
                    identity: &self.identity,
                    config: &config,
                    evidence,
                    previous: &self.checkpoint,
                    observations: &self.completed,
                })?
            };
            self.check_live(context, deadline)?;
            match decision {
                ValidationDecision::Complete {
                    outcome,
                    reason,
                    metrics,
                    evidence_refs,
                } => {
                    if reason.is_empty() {
                        return Err(AgentError::Config("empty validation reason".into()));
                    }
                    bounded_value(&metrics, 65_536)?;
                    let stale = self.completed.values().any(|observation| {
                        let mutates = matches!(&observation.result, ObservationResult::Tool { record }
                            if record.executed && record.metadata.get("classification").and_then(|value| value.get("read_only")).and_then(Value::as_bool) == Some(false));
                        !mutates && !observation.identity.as_ref().unwrap_or(&self.identity).eligible(&publication, false, true)
                    });
                    let mut identity = self.identity.clone();
                    identity.mutation_generation = publication.mutation_generation;
                    identity.validation_dependencies =
                        self.input_dependencies.clone().unwrap_or_default();
                    let result = ValidationResult {
                        check_id: self.check_id.clone(),
                        identity,
                        outcome: if stale { GateOutcome::Unknown } else { outcome },
                        reason: if stale {
                            "observation_stale_after_mutation".into()
                        } else {
                            reason
                        },
                        metrics,
                        evidence_refs,
                    };
                    self.result_identity =
                        Some(canonical_identity(&serde_json::to_value(&result)?)?);
                    self.result = Some(result.clone());
                    self.acknowledge(journal).await?;
                    self.check_live(context, deadline)?;
                    let current = self.capture_publication(context)?;
                    if !result.identity.eligible(&current, false, true)
                        || !validation_dependencies_current(
                            &result.identity,
                            &evidence.validations,
                            &current,
                        )
                    {
                        self.publication_identity = None;
                        return Err(AgentError::Config(
                            "validation evidence changed during publication".into(),
                        ));
                    }
                    return Ok(ValidationDriverOutcome::Complete(result));
                }
                ValidationDecision::AwaitExternal {
                    binding: id,
                    checkpoint,
                } => {
                    bounded_value(&checkpoint, 16_384)?;
                    if !binding.adapter.descriptor().conditions.contains(&id) {
                        return Err(AgentError::Config("undeclared external condition".into()));
                    }
                    let condition = extensions.condition(&id).ok_or_else(|| {
                        AgentError::Config("unknown external condition binding".into())
                    })?;
                    self.checkpoint = checkpoint;
                    let wait = ExternalWait {
                        check_id: self.check_id.clone(),
                        request_id: uuid::Uuid::new_v4().to_string(),
                        binding: condition.clone(),
                        identity: self.identity.clone(),
                    };
                    self.wait = Some(wait.clone());
                    self.acknowledge(journal).await?;
                    self.check_live(context, deadline)?;
                    self.capture_publication(context)?;
                    return Ok(ValidationDriverOutcome::AwaitExternal(wait));
                }
                ValidationDecision::NeedObservations {
                    requests,
                    checkpoint,
                } => {
                    if requests.is_empty() || requests.len() > limits.max_observations_per_round {
                        return Err(AgentError::Config(
                            "empty or oversized observation batch".into(),
                        ));
                    }
                    bounded_value(&checkpoint, 16_384)?;
                    let mut new = vec![];
                    for request in requests {
                        validate_request(&request, binding)?;
                        let id = request.id().to_owned();
                        if let ValidationObservationRequest::Tool {
                            host_binding: Some(authority),
                            tool,
                            arguments,
                            ..
                        } = &request
                        {
                            let installed =
                                extensions.host_binding(authority).ok_or_else(|| {
                                    AgentError::Config("unknown requested host authority".into())
                                })?;
                            if installed.validator != self.adapter
                                || installed.version != self.contract_version
                                || installed.tool != *tool
                                || installed.arguments != *arguments
                            {
                                return Err(AgentError::Config(
                                    "host authority request mismatch".into(),
                                ));
                            }
                            self.host_authorities
                                .entry(id.clone())
                                .or_insert_with(|| installed.clone());
                        }
                        if let Some(old) = self.requests.get(&id) {
                            if old != &request {
                                return Err(AgentError::Config(
                                    "observation ID reused for a changed request".into(),
                                ));
                            }
                        } else {
                            self.requests.insert(id.clone(), request);
                        }
                        if !self.completed.contains_key(&id) && !new.contains(&id) {
                            new.push(id);
                        }
                    }
                    if new.is_empty() {
                        return Err(AgentError::Config(
                            "stalled cached validation reentry".into(),
                        ));
                    }
                    self.checkpoint = checkpoint;
                    self.pending_batch = new;
                }
            }
            tokio::task::yield_now().await;
        }
    }
}

// A request digest binds immutable arguments, scope, target, adapter/check config and attempt identity.
fn request_identity(
    request: &ValidationObservationRequest,
    identity: &EvidenceIdentity,
) -> Result<String> {
    canonical_identity(&json!({"request":request,"identity":identity}))
}
// Approval modifications have a separate execution identity and never rewrite the original request.
fn effective_identity(result: &ObservationResult) -> Result<String> {
    canonical_identity(&serde_json::to_value(result)?)
}
// Undeclared capabilities or arbitrary oversized requests never reach the executor.
fn validate_request(
    request: &ValidationObservationRequest,
    binding: &BoundValidationCheck,
) -> Result<()> {
    if request.id().is_empty() || request.id().len() > 128 {
        return Err(AgentError::Config("invalid observation ID".into()));
    }
    bounded_value(&serde_json::to_value(request)?, 65_536)?;
    match request {
        ValidationObservationRequest::Tool {
            tool,
            arguments,
            host_binding,
            ..
        } => {
            if !binding.adapter.descriptor().tools.contains(tool)
                || !arguments.is_object()
                || host_binding.as_deref()
                    != binding
                        .check
                        .config
                        .as_ref()
                        .and_then(|c| c.get("host_binding"))
                        .and_then(Value::as_str)
            {
                return Err(AgentError::Config(
                    "undeclared validation tool/authority".into(),
                ));
            }
        }
        ValidationObservationRequest::Plan { objective, .. } => {
            if !binding.adapter.descriptor().needs_planner || objective.trim().is_empty() {
                return Err(AgentError::Config("undeclared planner observation".into()));
            }
        }
        ValidationObservationRequest::Judge { config, .. } => {
            if !binding.adapter.descriptor().needs_judge {
                return Err(AgentError::Config("undeclared validation judge".into()));
            }
            ai_agents_core::autonomy::CompletionGate::Judge(config.clone())
                .validate()
                .map_err(AgentError::Config)?;
        }
    }
    Ok(())
}
// Executor outputs remain bound to their original operation; malformed or excessive results are errors.
// Effective fallback records retain their actual source; only an execution bridge binding can connect them to the immutable observation request.
fn validate_observation_result(
    request: &ValidationObservationRequest,
    result: &ObservationResult,
) -> Result<()> {
    bounded_value(&serde_json::to_value(result)?, 65_536)?;
    match (request, result) {
        (ValidationObservationRequest::Plan { .. }, ObservationResult::Plan { plan })
            if plan.is_object() =>
        {
            Ok(())
        }
        (
            ValidationObservationRequest::Tool {
                tool, arguments, ..
            },
            ObservationResult::Tool { record },
        ) if (record.requested_name == *tool && record.arguments == *arguments)
            || (matches!(
                record.source,
                ai_agents_core::ToolCallSource::Fallback { .. }
            ) && record
                .metadata
                .get("_validation_request")
                .is_some_and(|binding| {
                    binding.get("requested_name").and_then(Value::as_str) == Some(tool.as_str())
                        && binding.get("arguments") == Some(arguments)
                        && binding.get("call_id").and_then(Value::as_str)
                            == Some(record.call_id.as_str())
                        && binding.get("source")
                            == Some(
                                &serde_json::to_value(ai_agents_core::ToolCallSource::Task)
                                    .unwrap_or(Value::Null),
                            )
                })) =>
        {
            Ok(())
        }
        (ValidationObservationRequest::Judge { .. }, ObservationResult::Judge { score })
            if score.is_finite() && (0.0..=1.0).contains(score) =>
        {
            Ok(())
        }
        _ => Err(AgentError::Config(
            "observation result identity/shape mismatch".into(),
        )),
    }
}

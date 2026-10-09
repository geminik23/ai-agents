//! Bounded resumable observation driver; custom callbacks cannot directly invoke tools or providers.

use super::*;
use ai_agents_core::autonomy::JudgeGate;
use ai_agents_core::{AgentError, Result, ToolCancellationToken, ToolExecutionRecord};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

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

/// In-flight records are recovery-required after a crash, never automatically reexecuted.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationDriverState {
    pub version: u32,
    pub binding_identity: String,
    pub check_deadline: Option<chrono::DateTime<chrono::Utc>>,
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
    pub pending_batch: Vec<String>,
    #[serde(skip)]
    publication_identity: Option<String>,
    pub wait: Option<ExternalWait>,
    pub external_events: BTreeMap<String, String>,
    pub result: Option<ValidationResult>,
    pub result_identity: Option<String>,
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
    /// Performs deterministic binding/alias preflight before a journal can mark an operation dispatched.
    fn preflight(
        &self,
        _request: &ValidationObservationRequest,
        _identity: &EvidenceIdentity,
    ) -> Result<()> {
        Ok(())
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
        Ok(Self {
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
            pending_batch: vec![],
            publication_identity: None,
            wait: None,
            external_events: BTreeMap::new(),
            result: None,
            result_identity: None,
        })
    }
    /// Restore rejects changed requests, adapters, outcomes and uncertain dispatched work before callback reentry.
    pub fn restore(
        value: Value,
        binding: &BoundValidationCheck,
        scope: &EvaluationScope,
        extensions: &FrozenAutonomyExtensions,
    ) -> Result<Self> {
        bounded_value(&value, 262_144)?;
        let state: Self = serde_json::from_value(value)?;
        state.validate(binding, scope, extensions)?;
        if state.in_flight.is_some() {
            return Err(AgentError::Config(
                "validation observation requires explicit effect reconciliation".into(),
            ));
        }
        Ok(state)
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
            || (binding.adapter.descriptor().requires_host_revision
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
        if let Some(result) = &self.result
            && (result.identity != self.identity
                || result.check_id != self.check_id
                || self.result_identity.as_ref()
                    != Some(&canonical_identity(&serde_json::to_value(result)?)?))
        {
            return Err(AgentError::Config(
                "changed validation result identity".into(),
            ));
        }
        Ok(())
    }

    /// Collection is allowed only for the exact state acknowledged by the journal, not a locally installed failed write.
    pub fn publication_acknowledged(&self) -> bool {
        self.publication_identity
            .as_ref()
            .is_some_and(|digest| serde_json::to_string(self).ok().as_ref() == Some(digest))
    }

    // A failed acknowledgement clears publication authority, including cached terminal/wait reentry.
    async fn acknowledge(&mut self, journal: &dyn ValidationJournal) -> Result<()> {
        self.publication_identity = None;
        journal.checkpoint(self).await?;
        self.publication_identity = Some(serde_json::to_string(self)?);
        Ok(())
    }

    // Finishes a persisted observation batch before callback reentry; in-flight work is never automatically replayed.
    async fn execute_pending(
        &mut self,
        binding: &BoundValidationCheck,
        context: &ValidationDriveContext<'_>,
        deadline: chrono::DateTime<chrono::Utc>,
    ) -> Result<()> {
        while let Some(id) = self.pending_batch.first().cloned() {
            if context.cancellation.is_cancelled() || chrono::Utc::now() >= deadline {
                return Err(AgentError::Other(
                    "validation cancelled or expired before admission".into(),
                ));
            }
            context
                .executor
                .preflight(&self.requests[&id], &self.identity)?;
            self.publication_identity = None;
            self.in_flight = Some(id.clone());
            self.validate(binding, context.scope, context.extensions)?;
            context.journal.checkpoint(self).await?;
            if context.cancellation.is_cancelled() || chrono::Utc::now() >= deadline {
                return Err(AgentError::Other(
                    "validation interrupted after admission acknowledgement".into(),
                ));
            }
            let request = self.requests[&id].clone();
            let mut future = Box::pin(context.executor.execute(&request, &self.identity));
            let result = loop {
                tokio::select! { result = &mut future => break result?, _ = tokio::time::sleep(std::time::Duration::from_millis(10)) => {
                    if context.cancellation.is_cancelled() || chrono::Utc::now() >= deadline { return Err(AgentError::Other("validation interrupted with uncertain effect".into())); }
                } }
            };
            drop(future);
            validate_observation_result(&request, &result)?;
            let completed = CompletedObservation {
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
        Ok(())
    }

    /// Refuses stale attempt rewinds and preserves every acknowledged request/result before a journal can admit another effect.
    pub(crate) fn validate_journal_successor(&self, old: &Self) -> Result<()> {
        if self.identity.attempt != old.identity.attempt {
            if old.in_flight.is_some()
                || old.wait.is_some()
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
            || old
                .in_flight
                .as_ref()
                .is_some_and(|id| !self.completed.contains_key(id))
        {
            return Err(AgentError::Config(
                "stale or uncertain validation journal continuation".into(),
            ));
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
    pub async fn drive(
        &mut self,
        binding: &BoundValidationCheck,
        context: &ValidationDriveContext<'_>,
    ) -> Result<ValidationDriverOutcome> {
        let ValidationDriveContext {
            scope,
            evidence,
            extensions,
            executor: _,
            journal,
            limits,
            cancellation,
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
        if self.in_flight.is_some() {
            return Err(AgentError::Config("uncertain validation effect".into()));
        }
        loop {
            if cancellation.is_cancelled() || chrono::Utc::now() >= deadline {
                return Err(AgentError::Other(
                    "validation cancelled or deadline exceeded".into(),
                ));
            }
            if self.result.is_some() {
                self.acknowledge(journal).await?;
                return Ok(ValidationDriverOutcome::Complete(
                    self.result.as_ref().unwrap().clone(),
                ));
            }
            if self.wait.is_some() {
                self.acknowledge(journal).await?;
                return Ok(ValidationDriverOutcome::AwaitExternal(
                    self.wait.as_ref().unwrap().clone(),
                ));
            }
            if !self.pending_batch.is_empty() {
                self.execute_pending(binding, context, deadline).await?;
                continue;
            }
            if self.rounds >= limits.max_evaluation_rounds {
                return Err(AgentError::Config("validation callback round limit".into()));
            }
            self.rounds += 1;
            journal.checkpoint(self).await?;
            let config = binding.check.config.clone().unwrap_or_else(|| json!({}));
            let decision = if !binding.available {
                ValidationDecision::Complete {
                    outcome: GateOutcome::Unknown,
                    reason: "optional_capability_unavailable".into(),
                    metrics: json!({}),
                    evidence_refs: vec![],
                }
            } else {
                binding.adapter.evaluate(&ValidationInput {
                    scope,
                    identity: &self.identity,
                    config: &config,
                    evidence,
                    previous: &self.checkpoint,
                    observations: &self.completed,
                })?
            };
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
                    let result = ValidationResult {
                        check_id: self.check_id.clone(),
                        identity: self.identity.clone(),
                        outcome,
                        reason,
                        metrics,
                        evidence_refs,
                    };
                    self.result_identity =
                        Some(canonical_identity(&serde_json::to_value(&result)?)?);
                    self.result = Some(result.clone());
                    self.acknowledge(journal).await?;
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
        ) if record.requested_name == *tool && record.arguments == *arguments => Ok(()),
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

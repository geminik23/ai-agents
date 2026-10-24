//! Explicit reviewed recovery preserves uncertain effects, exact revisions and original accounting; it never retries semantic work.

use super::*;
use chrono::{DateTime, Utc};
use serde_json::Value;
use std::time::{Duration, Instant};

/// Recovery may stop incomplete or recover an unchanged prepared cursor; neither outcome establishes task completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskRecoveryDisposition {
    Stop,
    ResumePrepared,
    /// Restores only enough authority to retire a cancelled prepared cursor; normal execution claims remain forbidden.
    CancelPrepared,
}

/// Authenticated host input is deliberately not deserializable from a model, tool result or checkpoint.
/// Effect results are audit-only and conservatively retain their full cost/write commitments, never completion credit or refunds.
#[derive(Clone)]
pub struct TaskRecoveryDecision {
    pub expected_revision: u64,
    pub reason: String,
    pub stopped_at: Option<DateTime<Utc>>,
    /// Required when an interrupted pre-journal claim no longer has an unambiguous stored start; certified by the host, never inferred from liveness.
    pub interrupted_started_at: Option<DateTime<Utc>>,
    pub disposition: TaskRecoveryDisposition,
    pub effects: BTreeMap<String, Value>,
    pub children: BTreeMap<String, Value>,
}

/// Trusted integration supplies already-reviewed effect/stop evidence; validation must be bounded, pure and non-reentrant.
/// Owner tokens, process death, cancellation receipts and elapsed time alone do not certify that external work has stopped.
/// The host must bind its proof to the exact snapshot, decision and actual implementations and reject stale or revoked proof.
pub trait TaskRecoveryHost: Send + Sync {
    /// Certifies stop, exact effect outcomes and any supplied clock evidence for this original snapshot and decision without performing I/O.
    fn validate(&self, snapshot: &TaskRunSnapshot, decision: &TaskRecoveryDecision) -> Result<()>;
    /// Shares the backend UTC domain and rejects uncertifiable continuity, rather than clamping to a stored timestamp.
    fn now(&self) -> Result<DateTime<Utc>>;
    /// A positive bounded recovery transaction is conservatively charged before publication, including ACK waits.
    fn recovery_budget(&self) -> Duration;
}

/// Only the framework constructs this exact reviewed successor; custom stores must preserve its backend admission guard.
pub struct TaskRecoveryCheckpoint {
    previous: TaskRunSnapshot,
    mutation: TaskRunMutation,
}
impl TaskRecoveryCheckpoint {
    /// Returns the guarded mutation unchanged; translating it to ResolveRecovery loses reviewed backend admission.
    pub fn mutation(&self) -> &TaskRunMutation {
        &self.mutation
    }

    /// Verifies the exact predecessor and monotonic recovery data before a custom store performs its conditional write.
    pub fn validate(&self, actual: &TaskRunSnapshot) -> Result<()> {
        if serde_json::to_value(actual)? != serde_json::to_value(&self.previous)? {
            return Err(TaskRunStorageError::Conflict.into());
        }
        let TaskRunMutation::ReconcileRecovery { payload, .. } = &self.mutation else {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        };
        let old: TaskCheckpointPayload = serde_json::from_value(actual.payload.clone())?;
        let next: TaskCheckpointPayload = serde_json::from_value(payload.clone())?;
        next.validate_successor(&old, actual.revision + 1, true)
    }
}

struct RecoveryAdmission {
    host: Arc<dyn TaskRecoveryHost>,
    source: TaskRunSnapshot,
    decision: TaskRecoveryDecision,
    not_before: DateTime<Utc>,
    deadline: Instant,
    expiry: Option<DateTime<Utc>>,
}
impl std::fmt::Debug for RecoveryAdmission {
    // Recovery payloads and external proof are sensitive and do not belong in diagnostics.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RecoveryAdmission")
    }
}
impl TaskFinalAdmission for RecoveryAdmission {
    // Review is checked before the final clock sample so a slow proof cannot authorize an expired publication.
    fn validate(&self) -> Result<()> {
        self.host.validate(&self.source, &self.decision)?;
        let now = self.host.now()?;
        if now < self.not_before {
            return Err(TaskRunStorageError::ClockDiscontinuity.into());
        }
        if Instant::now() >= self.deadline || self.expiry.is_some_and(|cut| now >= cut) {
            return Err(TaskRunStorageError::AdmissionExpired.into());
        }
        Ok(())
    }
}

// Interrupted accounting uses an explicit sampled-usage anchor, never metadata updated by a cancel request.
// Legacy checkpoints conservatively charge the entire open interval; they never receive a guessed unused allowance.
fn interrupted_charge(
    source: &TaskRunSnapshot,
    payload: &TaskCheckpointPayload,
    now: DateTime<Utc>,
    stopped: Option<DateTime<Utc>>,
    certified_start: Option<DateTime<Utc>>,
) -> Result<u64> {
    let Some(start) = payload.clocks.active_interval_started_at else {
        // Reconstruction prepays a bound without opening an ordinary active interval; only elapsed excess is still unaccounted.
        if let Some(work) = payload
            .adapters
            .iter()
            .find(|a| a.id == "runtime.restore_work" && a.state["phase"] == "running")
        {
            let started: DateTime<Utc> = serde_json::from_value(work.state["started_at"].clone())?;
            let stop = stopped.unwrap_or(now);
            if stop < started || stop > now {
                return Err(TaskRunStorageError::ClockDiscontinuity.into());
            }
            let elapsed = (stop - started)
                .to_std()
                .map_err(|_| TaskRunStorageError::ClockDiscontinuity)?;
            let elapsed = u64::try_from(elapsed.as_nanos().div_ceil(1_000_000))
                .map_err(|_| TaskRunStorageError::InvalidCheckpoint)?;
            let prepaid = work.state["budget_millis"]
                .as_u64()
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            return Ok(elapsed.saturating_sub(prepaid));
        }
        // An exact claim-only successor still records its own start, but later cancel/recovery metadata cannot recover that timestamp.
        let exact_claim = source.status == TaskRunStatus::Running
            && payload
                .adapters
                .iter()
                .find(|a| a.id == "runtime.restore")
                .and_then(|a| a.state["revision"].as_u64())
                .and_then(|revision| revision.checked_add(1))
                == Some(source.revision);
        let started = if exact_claim {
            source.updated_at
        } else {
            certified_start.ok_or(TaskRunStorageError::NotResumable)?
        };
        let stop = stopped.unwrap_or(now);
        if started < source.created_at || stop < started || stop > now {
            return Err(TaskRunStorageError::ClockDiscontinuity.into());
        }
        let elapsed = (stop - started)
            .to_std()
            .map_err(|_| TaskRunStorageError::ClockDiscontinuity)?;
        return u64::try_from(elapsed.as_nanos().div_ceil(1_000_000))
            .map_err(|_| AgentError::from(TaskRunStorageError::InvalidCheckpoint));
    };
    let anchor = payload
        .adapters
        .iter()
        .find(|a| a.id == "runtime.recovery_clock")
        .filter(|a| {
            a.contract_version == 1
                && a.state["active_millis"].as_u64() == Some(payload.clocks.active_millis)
        })
        .map(|a| serde_json::from_value::<DateTime<Utc>>(a.state["sampled_at"].clone()))
        .transpose()?
        .unwrap_or(start);
    let stop = stopped.unwrap_or(now);
    if anchor < start || stop < anchor || stop > now {
        return Err(TaskRunStorageError::ClockDiscontinuity.into());
    }
    let duration = (stop - anchor)
        .to_std()
        .map_err(|_| TaskRunStorageError::ClockDiscontinuity)?;
    u64::try_from(duration.as_nanos().div_ceil(1_000_000))
        .map_err(|_| AgentError::from(TaskRunStorageError::InvalidCheckpoint))
}

// A stopped arbitrary call stack cannot be reconstructed by changing its cursor to BetweenTurns.
// Only exact prepared images can resume; other reviewed effects may be settled into an incomplete terminal outcome at a known stored boundary.
fn prepare_recovery(
    source: &TaskRunSnapshot,
    decision: &TaskRecoveryDecision,
    now: DateTime<Utc>,
    budget: Duration,
) -> Result<(TaskCheckpointPayload, TaskRunStatus)> {
    let mut payload: TaskCheckpointPayload = serde_json::from_value(source.payload.clone())?;
    if now < source.updated_at
        || decision
            .stopped_at
            .is_some_and(|stop| stop > now || stop < source.created_at)
    {
        return Err(TaskRunStorageError::ClockDiscontinuity.into());
    }
    if decision.reason.trim().is_empty()
        || decision.reason.chars().count() > 512
        || budget.is_zero()
    {
        return Err(TaskRunStorageError::InvalidCheckpoint.into());
    }
    let millis = u64::try_from(budget.as_nanos().div_ceil(1_000_000))
        .map_err(|_| TaskRunStorageError::InvalidCheckpoint)?;
    let interrupted = interrupted_charge(
        source,
        &payload,
        now,
        decision.stopped_at,
        decision.interrupted_started_at,
    )?;
    if matches!(
        decision.disposition,
        TaskRecoveryDisposition::ResumePrepared | TaskRecoveryDisposition::CancelPrepared
    ) {
        if source.cancel_requested
            != (decision.disposition == TaskRecoveryDisposition::CancelPrepared)
            || payload.clocks.active_interval_started_at.is_some()
            || !decision.effects.is_empty()
            || !decision.children.is_empty()
        {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        restoration::validate_recovery_source(source, &payload)?;
    } else if payload.pending.is_some()
        || !matches!(payload.runtime.continuation, TaskContinuation::BetweenTurns)
        || payload.children.iter().any(|child| {
            child.pending.is_some()
                || !matches!(child.runtime.continuation, TaskContinuation::BetweenTurns)
        })
    {
        return Err(TaskRunStorageError::NotResumable.into());
    }
    let mut effects = decision.effects.clone();
    for reservation in &mut payload.reservations {
        if decision.disposition != TaskRecoveryDisposition::Stop {
            continue;
        }
        if reservation.state == TaskEffectState::Completed {
            if effects.contains_key(&reservation.id) {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            continue;
        }
        let result = effects
            .remove(&reservation.id)
            .ok_or(TaskRunStorageError::NotResumable)?;
        let charge = reservation
            .reserved_micro_usd
            .checked_sub(reservation.charged_micro_usd)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        payload.counters.charged_micro_usd = payload
            .counters
            .charged_micro_usd
            .checked_add(charge)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        reservation.charged_micro_usd = reservation.reserved_micro_usd;
        reservation.state = TaskEffectState::Completed;
        reservation.result = Some(json!({"host_reconciled":result,"task_result_eligible":false}));
    }
    if !effects.is_empty() {
        return Err(TaskRunStorageError::InvalidCheckpoint.into());
    }
    let mut children = decision.children.clone();
    for child in &mut payload.children {
        if decision.disposition != TaskRecoveryDisposition::Stop {
            continue;
        }
        if child.result.is_none() {
            child.result = Some(
                children
                    .remove(&child.child_id)
                    .ok_or(TaskRunStorageError::NotResumable)?,
            );
        }
    }
    if !children.is_empty() {
        return Err(TaskRunStorageError::InvalidCheckpoint.into());
    }
    payload.clocks.interrupted_interval_millis = payload
        .clocks
        .interrupted_interval_millis
        .checked_add(interrupted)
        .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
    payload.clocks.active_millis = payload
        .clocks
        .active_millis
        .checked_add(millis)
        .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
    payload.clocks.active_interval_started_at = None;
    let status = if decision.disposition != TaskRecoveryDisposition::Stop {
        TaskRunStatus::Paused
    } else if source.cancel_requested {
        TaskRunStatus::Cancelled
    } else {
        TaskRunStatus::Incomplete
    };
    if status == TaskRunStatus::Paused
        && !source.cancel_requested
        && (payload
            .clocks
            .active_millis
            .checked_add(payload.clocks.interrupted_interval_millis)
            .is_none_or(|used| used >= payload.limits.max_active_time_seconds.saturating_mul(1000))
            || payload.clocks.expires_at.is_some_and(|cut| now >= cut))
    {
        return Err(TaskRunStorageError::AdmissionExpired.into());
    }
    let sequence = payload
        .evidence
        .records
        .iter()
        .map(|r| r.sequence)
        .chain(payload.evidence.tool_calls.iter().map(|r| r.sequence))
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
    payload.evidence.records.push(TaskEvidenceRecord {
        run_id: source.key.run_id.clone(), sequence, stage: payload.stage.clone(), attempt: None,
        origin: "host.recovery".into(),
        data: json!({"revision":decision.expected_revision,"reason":decision.reason,"stopped_at":decision.stopped_at,"interrupted_millis":interrupted,"recovery_millis":millis,"resumable":status==TaskRunStatus::Paused}),
    });
    if status != TaskRunStatus::Paused {
        payload.stop_reason = Some("host_recovered_incomplete".into());
    }
    Ok((payload, status))
}

impl AutonomyRunner {
    /// Quarantines only an explicitly reviewed abandoned revision, then acknowledges known effects without invoking or replaying work.
    /// ResumePrepared repairs interrupted preparation only; arbitrary in-flight work needs a domain-specific safe continuation and is not guessed.
    /// CancelPrepared retains cancellation and produces only a cleanup cursor for restore_cancelled, never ordinary resume authority.
    /// A dropped or failed write is never retried automatically; callers inspect the authoritative revision and obtain a new reviewed decision.
    pub async fn reconcile_recovery(
        store: Arc<dyn TaskRunStore>,
        run_id: &str,
        decision: TaskRecoveryDecision,
        host: Arc<dyn TaskRecoveryHost>,
    ) -> Result<TaskRunSummary> {
        let started = Instant::now();
        if !store.supports_recovery() {
            return Err(AgentError::Config(
                "task store lacks reviewed recovery admission".into(),
            ));
        }
        let source = store
            .load(run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        if source.revision != decision.expected_revision {
            return Err(TaskRunStorageError::Conflict.into());
        }
        if !matches!(
            source.status,
            TaskRunStatus::Running | TaskRunStatus::RecoveryRequired
        ) {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        host.validate(&source, &decision)?;
        let cleanup_only = decision.disposition == TaskRecoveryDisposition::CancelPrepared;
        let now = host.now()?;
        let budget = host.recovery_budget();
        let deadline = started
            .checked_add(budget)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        // Preflight rejects unsupported or incomplete decisions before quarantining an otherwise valid source.
        prepare_recovery(&source, &decision, now, budget)?;
        let quarantined = if let Some(token) = &source.owner_token {
            let mutation = TaskRunMutation::Recover {
                expected_revision: source.revision,
                owner_token: token.clone(),
            };
            let saved = store.mutate(run_id, &mutation).await?;
            super::super::store::verify_checkpoint_receipt(&source, &mutation, &saved)?;
            saved
        } else {
            source.clone()
        };
        let (mut payload, status) = prepare_recovery(&source, &decision, host.now()?, budget)?;
        if status == TaskRunStatus::Paused {
            restoration::rebind_recovery_manifest(&mut payload, quarantined.revision + 1)?;
        }
        let guard: Arc<dyn TaskFinalAdmission> = Arc::new(RecoveryAdmission {
            host,
            source,
            decision,
            not_before: quarantined.updated_at,
            deadline,
            expiry: (status == TaskRunStatus::Paused && !cleanup_only)
                .then_some(payload.clocks.expires_at)
                .flatten(),
        });
        let checkpoint = TaskRecoveryCheckpoint {
            mutation: TaskRunMutation::ReconcileRecovery {
                expected_revision: quarantined.revision,
                status,
                payload: serde_json::to_value(&payload)?,
                evidence: guard,
            },
            previous: quarantined,
        };
        let saved = store.checkpoint_recovery(run_id, &checkpoint).await?;
        super::super::store::verify_checkpoint_receipt(
            &checkpoint.previous,
            checkpoint.mutation(),
            &saved,
        )?;
        Ok(saved.summary())
    }
}

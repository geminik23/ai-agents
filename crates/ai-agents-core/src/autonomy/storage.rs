//! Versioned task persistence contracts, independent of the runtime execution model.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{AgentError, Result};

pub const TASK_RUN_SCHEMA_VERSION: u32 = 1;
pub const MAX_TASK_CHECKPOINT_BYTES: usize = 4 * 1024 * 1024;

/// Durable task identity; callers must still enforce host access authorization.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRunKey {
    pub agent_id: String,
    pub run_id: String,
}

/// Persisted execution status, not an inference from response text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskRunStatus {
    Running,
    Paused,
    RecoveryRequired,
    Completed,
    Incomplete,
    Failed,
    Cancelled,
    LimitReached,
}

impl TaskRunStatus {
    /// Terminal states cannot be resumed or overwritten by a stale owner.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed
                | Self::Incomplete
                | Self::Failed
                | Self::Cancelled
                | Self::LimitReached
        )
    }
}

/// Typed storage failures distinguish conflicts and missing data from unsupported backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TaskRunStorageError {
    #[error("task identity already exists or was deleted")]
    AlreadyExists,
    #[error("task run does not exist")]
    NotFound,
    #[error("task revision or owner conflicts")]
    Conflict,
    #[error("task is not resumable")]
    NotResumable,
    #[error("task has an active owner")]
    Owned,
    #[error("invalid task checkpoint or transition")]
    InvalidCheckpoint,
    #[error("task checkpoint exceeds its size limit")]
    CheckpointTooLarge,
    #[error("task final admission deadline expired")]
    AdmissionExpired,
    #[error("task reconstruction clock moved backwards")]
    ClockDiscontinuity,
}

impl From<TaskRunStorageError> for AgentError {
    fn from(value: TaskRunStorageError) -> Self {
        Self::TaskRunStorage(value)
    }
}

/// Recovery data is exact and sensitive; listing deliberately exposes none of this payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRunSnapshot {
    pub schema_version: u32,
    pub key: TaskRunKey,
    pub actor_id: Option<String>,
    pub revision: u64,
    pub status: TaskRunStatus,
    pub owner_token: Option<String>,
    pub cancel_requested: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// Runtime-owned schema, configuration identity and exact continuation are validated by the scoped adapter.
    pub payload: Value,
}

/// A safe listing has no objective, tool output, pending arguments or recovery payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRunSummary {
    pub key: TaskRunKey,
    pub actor_id: Option<String>,
    pub revision: u64,
    pub status: TaskRunStatus,
    pub owned: bool,
    pub cancel_requested: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// All filters are intersected; a scoped runtime adapter always supplies the agent identity.
#[derive(Debug, Clone, Default)]
pub struct TaskRunFilter {
    pub agent_id: String,
    pub actor_id: Option<String>,
    pub status: Option<TaskRunStatus>,
    pub limit: Option<usize>,
}

/// Live bounded admission evaluated inside the backend write boundary, never serialized into a checkpoint.
/// Implementations must not perform I/O or reenter storage; hosts synchronize external targets separately when stronger atomicity is needed.
pub trait TaskFinalAdmission: Send + Sync + std::fmt::Debug {
    /// Rejects stale live evidence before completion is committed.
    fn validate(&self) -> Result<()>;

    /// Rechecks a dispatched result without treating admission-only closure as cancellation of already consumed work.
    /// The conservative default retains the full proof; only restriction-only guards may relax their own admission closure here.
    fn validate_dispatched(&self) -> Result<()> {
        self.validate()
    }
}

/// Mutations are conditional and never constitute authority to invoke external work.
#[derive(Debug, Clone)]
pub enum TaskRunMutation {
    Claim {
        expected_revision: u64,
        owner_token: String,
    },
    /// Claims only cancellation cleanup; the monotonic cancel flag still prevents ordinary execution claims.
    ClaimCleanup {
        expected_revision: u64,
        owner_token: String,
    },
    /// Publishes a reconstructed cancellation cursor, never a resumable execution grant.
    CleanupCheckpoint {
        expected_revision: u64,
        owner_token: String,
        payload: Value,
        evidence: std::sync::Arc<dyn TaskFinalAdmission>,
    },
    Checkpoint {
        expected_revision: u64,
        owner_token: String,
        status: TaskRunStatus,
        payload: Value,
        release: bool,
    },
    /// Live final admission is checked inside the backend's write boundary, not only before awaiting storage.
    FinalCheckpoint {
        expected_revision: u64,
        owner_token: String,
        status: TaskRunStatus,
        payload: Value,
        release: bool,
        deadline: std::time::Instant,
        expires_at: Option<DateTime<Utc>>,
        evidence: Option<std::sync::Arc<dyn TaskFinalAdmission>>,
    },
    /// Host control-plane CAS requires authenticated preflight and retains original expiry through backend write admission.
    HostControl {
        expected_revision: u64,
        owner_token: Option<String>,
        status: TaskRunStatus,
        payload: Value,
        deadline: Option<std::time::Instant>,
        expires_at: Option<DateTime<Utc>>,
    },
    RequestCancel {
        expected_revision: u64,
    },
    /// Acknowledges cleanup of a safely paused, unowned run after cancellation was requested.
    AcknowledgeCancel {
        expected_revision: u64,
        payload: Value,
    },
    /// Explicitly quarantines an abandoned owner; it does not assert its effects were unapplied.
    Recover {
        expected_revision: u64,
        owner_token: String,
    },
    /// Reviewed recovery is checked again inside the backend transaction; it cannot declare task completion or adopt a live owner.
    ReconcileRecovery {
        expected_revision: u64,
        status: TaskRunStatus,
        payload: Value,
        evidence: std::sync::Arc<dyn TaskFinalAdmission>,
    },
    /// A host must reconcile uncertain effects before explicitly supplying a safe replacement payload.
    ResolveRecovery {
        expected_revision: u64,
        payload: Value,
    },
}

impl TaskRunMutation {
    /// Completion and safe pause recheck original clocks inside backend admission; supplied live guards also protect reconstruction pauses.
    /// Ordinary pauses supply no completion guard, while host controls retain their independent time bounds.
    pub fn validate_final_deadline(&self, now: DateTime<Utc>) -> Result<()> {
        let bounds = match self {
            Self::FinalCheckpoint {
                status: TaskRunStatus::Completed | TaskRunStatus::Paused,
                deadline,
                expires_at,
                ..
            } => Some((Some(*deadline), *expires_at)),
            Self::HostControl {
                deadline,
                expires_at,
                ..
            } => Some((*deadline, *expires_at)),
            _ => None,
        };
        if bounds.is_some_and(|(deadline, expiry)| {
            deadline.is_some_and(|bound| std::time::Instant::now() >= bound)
                || expiry.is_some_and(|bound| now >= bound)
        }) {
            return Err(TaskRunStorageError::AdmissionExpired.into());
        }
        if let Self::ReconcileRecovery { evidence, .. } | Self::CleanupCheckpoint { evidence, .. } =
            self
        {
            evidence.validate()?;
        }
        if let Self::FinalCheckpoint {
            status: TaskRunStatus::Completed | TaskRunStatus::Paused,
            evidence: Some(evidence),
            ..
        } = self
        {
            evidence.validate()?;
            // Pure host guards are cooperative rather than preemptible; a late return cannot authorize a commit past the fixed monotonic bound.
            if bounds.is_some_and(|(deadline, _)| {
                deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline)
            }) {
                return Err(TaskRunStorageError::AdmissionExpired.into());
            }
        }
        Ok(())
    }
}

// The same compact JSON serializer enforces the exact byte cap without allocating a second copy of sensitive recovery data.
#[derive(Default)]
struct JsonByteCount(usize);
impl std::io::Write for JsonByteCount {
    // Saturation conservatively exceeds every supported checkpoint cap instead of wrapping an oversized stream.
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self.0.saturating_add(bytes.len());
        Ok(bytes.len())
    }
    // There is no external sink or buffered I/O to flush.
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

// Counting and materializing JSON must agree on UTF-8, escaping and numeric encoding at the admission boundary.
fn serialized_byte_len(value: &impl Serialize) -> Result<usize> {
    let mut count = JsonByteCount::default();
    serde_json::to_writer(&mut count, value)?;
    Ok(count.0)
}

impl TaskRunSnapshot {
    /// Validates envelope invariants and rejects oversized recovery state without truncation.
    pub fn validate(&self) -> Result<()> {
        use TaskRunStorageError::*;
        if self.schema_version != TASK_RUN_SCHEMA_VERSION
            || self.key.agent_id.trim().is_empty()
            || self.key.run_id.trim().is_empty()
            || self
                .actor_id
                .as_ref()
                .is_some_and(|id| id.trim().is_empty())
            || self.revision > i64::MAX as u64
            || self.updated_at < self.created_at
            || !self.payload.is_object()
            || self
                .owner_token
                .as_ref()
                .is_some_and(|id| id.trim().is_empty())
            || (self.status == TaskRunStatus::Running) != self.owner_token.is_some()
            || (self.cancel_requested && self.status == TaskRunStatus::Completed)
        {
            return Err(InvalidCheckpoint.into());
        }
        if serialized_byte_len(self)? > MAX_TASK_CHECKPOINT_BYTES {
            return Err(CheckpointTooLarge.into());
        }
        Ok(())
    }

    /// New identities start at revision zero, with either a claimed run or a safe pause.
    pub fn validate_create(&self) -> Result<()> {
        self.validate()?;
        if self.revision != 0
            || self.cancel_requested
            || !matches!(self.status, TaskRunStatus::Running | TaskRunStatus::Paused)
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        Ok(())
    }

    /// Produces metadata only; redaction never changes required recovery data.
    pub fn summary(&self) -> TaskRunSummary {
        TaskRunSummary {
            key: self.key.clone(),
            actor_id: self.actor_id.clone(),
            revision: self.revision,
            status: self.status,
            owned: self.owner_token.is_some(),
            cancel_requested: self.cancel_requested,
            created_at: self.created_at,
            updated_at: self.updated_at,
        }
    }

    /// Applies one state transition under the atomic backend boundary, copying recovery data once only after that mutation's authority checks.
    pub fn transition(&self, mutation: &TaskRunMutation, now: DateTime<Utc>) -> Result<Self> {
        use TaskRunStorageError::*;
        self.validate()?;
        let expected = match mutation {
            TaskRunMutation::Claim {
                expected_revision, ..
            }
            | TaskRunMutation::ClaimCleanup {
                expected_revision, ..
            }
            | TaskRunMutation::CleanupCheckpoint {
                expected_revision, ..
            }
            | TaskRunMutation::Checkpoint {
                expected_revision, ..
            }
            | TaskRunMutation::FinalCheckpoint {
                expected_revision, ..
            }
            | TaskRunMutation::HostControl {
                expected_revision, ..
            }
            | TaskRunMutation::RequestCancel { expected_revision }
            | TaskRunMutation::AcknowledgeCancel {
                expected_revision, ..
            }
            | TaskRunMutation::Recover {
                expected_revision, ..
            }
            | TaskRunMutation::ResolveRecovery {
                expected_revision, ..
            }
            | TaskRunMutation::ReconcileRecovery {
                expected_revision, ..
            } => *expected_revision,
        };
        if expected != self.revision {
            return Err(Conflict.into());
        }
        if self.status.is_terminal() {
            return Err(NotResumable.into());
        }
        mutation.validate_final_deadline(now)?;
        // Payload-replacing mutations need no copy of the old recovery tree, and rejected mutations need no copy of the proposed tree.
        let mut next = Self {
            schema_version: self.schema_version,
            key: self.key.clone(),
            actor_id: self.actor_id.clone(),
            revision: self.revision,
            status: self.status,
            owner_token: self.owner_token.clone(),
            cancel_requested: self.cancel_requested,
            created_at: self.created_at,
            updated_at: self.updated_at,
            payload: Value::Null,
        };
        match mutation {
            TaskRunMutation::Claim { owner_token, .. } => {
                if self.owner_token.is_some() {
                    return Err(Owned.into());
                }
                if self.status != TaskRunStatus::Paused || self.cancel_requested {
                    return Err(NotResumable.into());
                }
                next.owner_token = Some(owner_token.clone());
                next.status = TaskRunStatus::Running;
            }
            TaskRunMutation::ClaimCleanup { owner_token, .. } => {
                if self.owner_token.is_some()
                    || self.status != TaskRunStatus::Paused
                    || !self.cancel_requested
                {
                    return Err(NotResumable.into());
                }
                next.owner_token = Some(owner_token.clone());
                next.status = TaskRunStatus::Running;
            }
            TaskRunMutation::CleanupCheckpoint {
                owner_token,
                payload,
                ..
            } => {
                if self.owner_token.as_deref() != Some(owner_token)
                    || self.status != TaskRunStatus::Running
                    || !self.cancel_requested
                {
                    return Err(Conflict.into());
                }
                next.owner_token = None;
                next.status = TaskRunStatus::Paused;
                next.payload = payload.clone();
            }
            TaskRunMutation::Checkpoint {
                owner_token,
                status,
                payload,
                release,
                ..
            }
            | TaskRunMutation::FinalCheckpoint {
                owner_token,
                status,
                payload,
                release,
                ..
            } => {
                if self.owner_token.as_deref() != Some(owner_token) {
                    return Err(Conflict.into());
                }
                // A cancelled owner may still persist known settlement, but cannot publish completion or clear cancellation.
                if (*status == TaskRunStatus::Running) == *release
                    || (self.cancel_requested
                        && !matches!(
                            status,
                            TaskRunStatus::Running
                                | TaskRunStatus::Cancelled
                                | TaskRunStatus::RecoveryRequired
                        ))
                {
                    return Err(InvalidCheckpoint.into());
                }
                next.status = *status;
                next.payload = payload.clone();
                if *release {
                    next.owner_token = None;
                }
            }
            TaskRunMutation::HostControl {
                owner_token,
                status,
                payload,
                ..
            } => {
                if self.owner_token != *owner_token {
                    return Err(Conflict.into());
                }
                if (*status == TaskRunStatus::Running) != owner_token.is_some()
                    && !status.is_terminal()
                {
                    return Err(InvalidCheckpoint.into());
                }
                if self.cancel_requested {
                    return Err(NotResumable.into());
                }
                next.status = *status;
                next.payload = payload.clone();
                if status.is_terminal() {
                    next.owner_token = None;
                }
            }
            TaskRunMutation::RequestCancel { .. } => {
                next.cancel_requested = true;
            }
            TaskRunMutation::AcknowledgeCancel { payload, .. } => {
                if self.owner_token.is_some() {
                    return Err(Owned.into());
                }
                if self.status != TaskRunStatus::Paused || !self.cancel_requested {
                    return Err(NotResumable.into());
                }
                next.status = TaskRunStatus::Cancelled;
                next.payload = payload.clone();
            }
            TaskRunMutation::Recover { owner_token, .. } => {
                if self.owner_token.as_deref() != Some(owner_token) {
                    return Err(Conflict.into());
                }
                next.owner_token = None;
                next.status = TaskRunStatus::RecoveryRequired;
            }
            TaskRunMutation::ReconcileRecovery {
                status, payload, ..
            } => {
                if self.status != TaskRunStatus::RecoveryRequired
                    || self.owner_token.is_some()
                    || !matches!(
                        status,
                        TaskRunStatus::Paused
                            | TaskRunStatus::Incomplete
                            | TaskRunStatus::Failed
                            | TaskRunStatus::LimitReached
                            | TaskRunStatus::Cancelled
                    )
                    || (self.cancel_requested
                        && !matches!(status, TaskRunStatus::Paused | TaskRunStatus::Cancelled))
                {
                    return Err(NotResumable.into());
                }
                next.payload = payload.clone();
                next.status = *status;
            }
            TaskRunMutation::ResolveRecovery { payload, .. } => {
                if self.status != TaskRunStatus::RecoveryRequired || self.owner_token.is_some() {
                    return Err(NotResumable.into());
                }
                next.payload = payload.clone();
                next.status = if self.cancel_requested {
                    TaskRunStatus::Cancelled
                } else {
                    TaskRunStatus::Paused
                };
            }
        }
        if matches!(
            mutation,
            TaskRunMutation::Claim { .. }
                | TaskRunMutation::ClaimCleanup { .. }
                | TaskRunMutation::RequestCancel { .. }
                | TaskRunMutation::Recover { .. }
        ) {
            next.payload = self.payload.clone();
        }
        next.revision = next
            .revision
            .checked_add(1)
            .filter(|v| *v <= i64::MAX as u64)
            .ok_or(InvalidCheckpoint)?;
        next.updated_at = now.max(self.updated_at);
        next.validate()?;
        Ok(next)
    }
}

#[cfg(test)]
mod byte_count_tests {
    use super::*;

    // Compact JSON byte counts must match the old materialized encoding for escaping, multibyte text and finite numbers.
    #[test]
    fn counting_writer_matches_exact_json_bytes() {
        for value in [
            serde_json::json!({"text":"quote\" newline\n slash\\ \u{d55c}\u{1f600}"}),
            serde_json::json!({"values":[-1,0,1.25,1e100],"nested":[{},null,true]}),
        ] {
            assert_eq!(
                serialized_byte_len(&value).unwrap(),
                serde_json::to_vec(&value).unwrap().len()
            );
        }
    }

    // The inclusive serialized-byte limit and non-replacing mutation payloads remain unchanged without buffering recovery data twice.
    #[test]
    fn exact_byte_limit_and_nonreplacing_transitions_preserve_payload() {
        let now = Utc::now();
        let mut snapshot = TaskRunSnapshot {
            schema_version: TASK_RUN_SCHEMA_VERSION,
            key: TaskRunKey {
                agent_id: "agent".into(),
                run_id: "run".into(),
            },
            actor_id: None,
            revision: 0,
            status: TaskRunStatus::Paused,
            owner_token: None,
            cancel_requested: false,
            created_at: now,
            updated_at: now,
            payload: serde_json::json!({"data":""}),
        };
        let overhead = serde_json::to_vec(&snapshot).unwrap().len();
        snapshot.payload["data"] = Value::String("x".repeat(MAX_TASK_CHECKPOINT_BYTES - overhead));
        assert_eq!(
            serialized_byte_len(&snapshot).unwrap(),
            MAX_TASK_CHECKPOINT_BYTES
        );
        snapshot.validate().unwrap();

        snapshot.payload["extra"] = serde_json::json!(true);
        assert!(matches!(
            snapshot.validate(),
            Err(AgentError::TaskRunStorage(
                TaskRunStorageError::CheckpointTooLarge
            ))
        ));
        snapshot.payload = serde_json::json!({"opaque":["original",1.25,{"signed":"exact"}]});
        let claimed = snapshot
            .transition(
                &TaskRunMutation::Claim {
                    expected_revision: 0,
                    owner_token: "owner".into(),
                },
                now,
            )
            .unwrap();
        let cancelled = claimed
            .transition(
                &TaskRunMutation::RequestCancel {
                    expected_revision: 1,
                },
                now,
            )
            .unwrap();
        let recovered = cancelled
            .transition(
                &TaskRunMutation::Recover {
                    expected_revision: 2,
                    owner_token: "owner".into(),
                },
                now,
            )
            .unwrap();
        assert_eq!(snapshot.payload, claimed.payload);
        assert_eq!(snapshot.payload, cancelled.payload);
        assert_eq!(snapshot.payload, recovered.payload);
        assert_eq!(recovered.status, TaskRunStatus::RecoveryRequired);
    }
}

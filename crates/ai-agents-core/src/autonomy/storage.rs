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

/// Mutations are conditional and never constitute authority to invoke external work.
#[derive(Debug, Clone)]
pub enum TaskRunMutation {
    Claim {
        expected_revision: u64,
        owner_token: String,
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
    },
    /// Host control-plane CAS; runtime adapters require an auditable authenticated host action.
    HostControl {
        expected_revision: u64,
        owner_token: Option<String>,
        status: TaskRunStatus,
        payload: Value,
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
    /// A host must reconcile uncertain effects before explicitly supplying a safe replacement payload.
    ResolveRecovery {
        expected_revision: u64,
        payload: Value,
    },
}

impl TaskRunMutation {
    /// Completion cannot pass a live monotonic or original-expiry deadline after waiting for a write lock.
    pub fn validate_final_deadline(&self, now: DateTime<Utc>) -> Result<()> {
        if let Self::FinalCheckpoint {
            status: TaskRunStatus::Completed,
            deadline,
            expires_at,
            ..
        } = self
            && (std::time::Instant::now() >= *deadline
                || expires_at.is_some_and(|expiry| now >= expiry))
        {
            return Err(TaskRunStorageError::AdmissionExpired.into());
        }
        Ok(())
    }
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
        if serde_json::to_vec(self)?.len() > MAX_TASK_CHECKPOINT_BYTES {
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

    /// Applies one state transition while a backend holds its atomic write boundary.
    pub fn transition(&self, mutation: &TaskRunMutation, now: DateTime<Utc>) -> Result<Self> {
        use TaskRunStorageError::*;
        self.validate()?;
        let expected = match mutation {
            TaskRunMutation::Claim {
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
            } => *expected_revision,
        };
        if expected != self.revision {
            return Err(Conflict.into());
        }
        if self.status.is_terminal() {
            return Err(NotResumable.into());
        }
        mutation.validate_final_deadline(now)?;
        let mut next = self.clone();
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

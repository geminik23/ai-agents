//! Scoped task persistence validates recovery data without invoking the runtime or any tool.

use super::run::TaskCheckpointPayload;
use ai_agents_core::autonomy::*;
use ai_agents_core::{AgentError, AgentStorage, Result, StorageCapability};
use async_trait::async_trait;
use std::sync::Arc;

/// Verifies exact checkpoint receipts without repeating final deadline admission after a write already committed.
/// This detects inconsistent replies, not a backend that falsely acknowledges an exact-looking unsaved snapshot.
pub(crate) fn verify_checkpoint_receipt(
    previous: &TaskRunSnapshot,
    mutation: &TaskRunMutation,
    saved: &TaskRunSnapshot,
) -> Result<()> {
    let checkpoint = match mutation {
        TaskRunMutation::FinalCheckpoint {
            expected_revision,
            owner_token,
            status,
            payload,
            release,
            ..
        } => Some(TaskRunMutation::Checkpoint {
            expected_revision: *expected_revision,
            owner_token: owner_token.clone(),
            status: *status,
            payload: payload.clone(),
            release: *release,
        }),
        _ => None,
    };
    let expected =
        previous.transition(checkpoint.as_ref().unwrap_or(mutation), saved.updated_at)?;
    // Exhaustive destructuring makes a newly added envelope field require an explicit receipt-verification decision.
    let TaskRunSnapshot {
        schema_version,
        key,
        actor_id,
        revision,
        status,
        owner_token,
        cancel_requested,
        created_at,
        updated_at,
        payload,
    } = saved;
    if *updated_at < previous.updated_at
        || expected.schema_version != *schema_version
        || expected.key != *key
        || expected.actor_id != *actor_id
        || expected.revision != *revision
        || expected.status != *status
        || expected.owner_token != *owner_token
        || expected.cancel_requested != *cancel_requested
        || expected.created_at != *created_at
        || expected.updated_at != *updated_at
        || expected.payload != *payload
    {
        return Err(TaskRunStorageError::InvalidCheckpoint.into());
    }
    Ok(())
}

/// Storage operations are CAS transitions; successful claim is required before later execution.
#[async_trait]
pub trait TaskRunStore: Send + Sync {
    /// Creates exact recovery data after runtime schema validation.
    async fn create(&self, snapshot: &TaskRunSnapshot) -> Result<()>;
    /// Returns only compatible data in this adapter's agent and actor scope.
    async fn load(&self, run_id: &str) -> Result<Option<TaskRunSnapshot>>;
    /// Applies a conditional operation without executing external work.
    /// Forward live final-admission guards and validate the transition inside the backend write boundary; serialized IDs cannot replace them.
    async fn mutate(&self, run_id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot>;
    /// Claims a paused revision; None certifies no write, while errors from a custom transport remain ambiguous.
    async fn claim_paused(
        &self,
        run_id: &str,
        expected_revision: u64,
        owner_token: String,
    ) -> Result<Option<TaskRunSnapshot>> {
        self.mutate(
            run_id,
            &TaskRunMutation::Claim {
                expected_revision,
                owner_token,
            },
        )
        .await
        .map(Some)
    }
    /// Declares validated paused-limit support before the runner starts a potentially ambiguous control write.
    fn supports_paused_limit_revision(&self) -> bool {
        false
    }
    /// Applies a limit-only CAS with the original admission deadline; None certifies no write, while Err may be ambiguous.
    /// Custom stores fail closed unless they implement the same preflight and acknowledgement contract.
    async fn revise_paused_limits(
        &self,
        _run_id: &str,
        _expected_revision: u64,
        _limits: super::TaskRunLimits,
        _reason: &str,
        _host: &super::AutonomyHostCeilings,
        _deadline: Option<std::time::Instant>,
    ) -> Result<Option<TaskRunSnapshot>> {
        Err(AgentError::Config(
            "paused limit revision is unsupported by this task store".into(),
        ))
    }
    /// Declares host objective revision support before any retained live state is changed.
    fn supports_objective_revision(&self) -> bool {
        false
    }
    /// Applies authenticated control only at a safe boundary; None certifies no write, while Err retains uncertainty.
    async fn apply_host_control(
        &self,
        _run_id: &str,
        _expected_revision: u64,
        _action: super::HostControlAction,
        _host: &super::AutonomyHostCeilings,
        _deadline: Option<std::time::Instant>,
    ) -> Result<Option<TaskRunSnapshot>> {
        Err(AgentError::Config(
            "host objective revision is unsupported by this task store".into(),
        ))
    }
    /// Lists safe metadata, never raw recovery data.
    async fn list(&self) -> Result<Vec<TaskRunSummary>>;
    /// Deletes unowned data at the expected revision.
    async fn delete(&self, run_id: &str, revision: u64) -> Result<()>;
}

/// Host-created scope and exact config identity bind all reads and writes to one task owner.
pub struct ScopedTaskRunStore {
    storage: Arc<dyn AgentStorage>,
    agent_id: String,
    actor_id: Option<String>,
    config_identity: String,
}

impl ScopedTaskRunStore {
    /// Requires transactional task capability rather than inferring it from snapshot support.
    pub fn new(
        storage: Arc<dyn AgentStorage>,
        agent_id: String,
        actor_id: Option<String>,
        config_identity: String,
    ) -> Result<Self> {
        if !storage.supports(StorageCapability::TaskRuns) {
            return Err(AgentError::UnsupportedStorageCapability(
                StorageCapability::TaskRuns,
            ));
        }
        if agent_id.trim().is_empty()
            || config_identity.trim().is_empty()
            || actor_id.as_ref().is_some_and(|a| a.trim().is_empty())
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        Ok(Self {
            storage,
            agent_id,
            actor_id,
            config_identity,
        })
    }

    /// Retains a volatile store for same-process pause; it is not restart-resumable.
    pub fn in_memory(
        agent_id: String,
        actor_id: Option<String>,
        config_identity: String,
    ) -> Result<Self> {
        Self::new(
            Arc::new(ai_agents_storage::InMemoryTaskStorage::default()),
            agent_id,
            actor_id,
            config_identity,
        )
    }

    /// Applies an explicit host-only action at a safe checkpoint with CAS, audit, host ceilings and unchanged consumed resources.
    pub async fn apply_host_action(
        &self,
        run_id: &str,
        expected_revision: u64,
        action: super::HostControlAction,
        host: &super::AutonomyHostCeilings,
    ) -> Result<TaskRunSnapshot> {
        self.apply_host_action_with_deadline(run_id, expected_revision, action, host, None)
            .await
    }

    /// Retained host controls carry their original monotonic expiry through the actual backend admission boundary.
    async fn apply_host_action_with_deadline(
        &self,
        run_id: &str,
        expected_revision: u64,
        action: super::HostControlAction,
        host: &super::AutonomyHostCeilings,
        deadline: Option<std::time::Instant>,
    ) -> Result<TaskRunSnapshot> {
        let previous = self
            .load(run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        if previous.revision != expected_revision {
            return Err(TaskRunStorageError::Conflict.into());
        }
        let old: TaskCheckpointPayload = serde_json::from_value(previous.payload.clone())?;
        let (payload, status) = super::prepare_host_control(&previous, action, host)?;
        let mutation = TaskRunMutation::HostControl {
            expected_revision,
            owner_token: previous.owner_token.clone(),
            status,
            payload: serde_json::to_value(&payload)?,
            deadline,
            expires_at: old.clocks.expires_at,
        };
        let proposed = previous.transition(&mutation, chrono::Utc::now())?;
        self.validate(&proposed)?;
        payload.validate_successor_with_control(&old, proposed.revision, false, true)?;
        let saved = self
            .storage
            .mutate_task_run(&self.key(run_id), &mutation)
            .await?;
        self.validate(&saved)?;
        Ok(saved)
    }

    // Scope and all payload invariants are checked at each boundary; retaining the decoded value avoids decoding it again for successor validation.
    fn validate(&self, snapshot: &TaskRunSnapshot) -> Result<TaskCheckpointPayload> {
        if snapshot.key.agent_id != self.agent_id || snapshot.actor_id != self.actor_id {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        snapshot.validate()?;
        let payload = <TaskCheckpointPayload as serde::Deserialize>::deserialize(&snapshot.payload)
            .map_err(|_| TaskRunStorageError::InvalidCheckpoint)?;
        payload.validate_contents(snapshot, &self.config_identity)?;
        Ok(payload)
    }

    // Derives the storage key solely from this adapter's fixed agent scope.
    fn key(&self, run_id: &str) -> TaskRunKey {
        TaskRunKey {
            agent_id: self.agent_id.clone(),
            run_id: run_id.to_owned(),
        }
    }
}

#[async_trait]
impl TaskRunStore for ScopedTaskRunStore {
    /// Rejects incompatible data before any backend write.
    async fn create(&self, snapshot: &TaskRunSnapshot) -> Result<()> {
        self.validate(snapshot)?;
        self.storage.create_task_run(snapshot).await
    }

    /// Validation precedes any later reconstruction of a live runtime.
    async fn load(&self, run_id: &str) -> Result<Option<TaskRunSnapshot>> {
        let snapshot = self.storage.load_task_run(&self.key(run_id)).await?;
        if let Some(snapshot) = &snapshot {
            self.validate(snapshot)?;
        }
        Ok(snapshot)
    }

    /// Checks fresh previous/proposed payloads before CAS and reuses each decoded value only within this immutable validation call.
    /// Backend validation and exact receipt checks remain independent; no acknowledged cache replaces the owned storage read.
    async fn mutate(&self, run_id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot> {
        if matches!(mutation, TaskRunMutation::HostControl { .. }) {
            return Err(AgentError::Config(
                "host mutations require apply_host_action preflight".into(),
            ));
        }
        let previous = self
            .storage
            .load_task_run(&self.key(run_id))
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        let old = self.validate(&previous)?;
        let proposed = previous.transition(mutation, chrono::Utc::now())?;
        let new = self.validate(&proposed)?;
        new.validate_successor(
            &old,
            proposed.revision,
            matches!(mutation, TaskRunMutation::ResolveRecovery { .. }),
        )?;
        let saved = self
            .storage
            .mutate_task_run(&self.key(run_id), mutation)
            .await?;
        self.validate(&saved)?;
        verify_checkpoint_receipt(&previous, mutation, &saved)?;
        Ok(saved)
    }

    /// Scoped backend claim conflicts are certified no-write outcomes, unlike arbitrary transport errors.
    async fn claim_paused(
        &self,
        run_id: &str,
        expected_revision: u64,
        owner_token: String,
    ) -> Result<Option<TaskRunSnapshot>> {
        match self
            .mutate(
                run_id,
                &TaskRunMutation::Claim {
                    expected_revision,
                    owner_token,
                },
            )
            .await
        {
            Ok(saved) => Ok(Some(saved)),
            Err(AgentError::TaskRunStorage(
                TaskRunStorageError::Conflict
                | TaskRunStorageError::Owned
                | TaskRunStorageError::NotResumable,
            )) => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Both supported scoped backends use the same preflight and conditional host-control transition.
    fn supports_paused_limit_revision(&self) -> bool {
        true
    }

    /// Preserves pending requests and every execution/evidence field while validating numeric ceilings before the backend CAS.
    async fn revise_paused_limits(
        &self,
        run_id: &str,
        expected_revision: u64,
        limits: super::TaskRunLimits,
        reason: &str,
        host: &super::AutonomyHostCeilings,
        deadline: Option<std::time::Instant>,
    ) -> Result<Option<TaskRunSnapshot>> {
        let previous = self
            .load(run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        if previous.revision != expected_revision {
            return Ok(None);
        }
        let old: TaskCheckpointPayload = serde_json::from_value(previous.payload.clone())?;
        let (payload, status) =
            match super::control::prepare_paused_limit_revision(&previous, limits, reason, host) {
                Ok(prepared) => prepared,
                Err(_) => return Ok(None),
            };
        let mutation = TaskRunMutation::HostControl {
            expected_revision,
            owner_token: None,
            status,
            payload: serde_json::to_value(&payload)?,
            deadline,
            expires_at: old.clocks.expires_at,
        };
        let proposed = match previous.transition(&mutation, chrono::Utc::now()) {
            Ok(proposed) => proposed,
            Err(AgentError::TaskRunStorage(
                TaskRunStorageError::AdmissionExpired
                | TaskRunStorageError::Conflict
                | TaskRunStorageError::NotResumable,
            )) => return Ok(None),
            Err(_) => return Ok(None),
        };
        if self.validate(&proposed).is_err()
            || payload
                .validate_successor_with_control(&old, proposed.revision, false, true)
                .is_err()
        {
            return Ok(None);
        }
        // These protocol rejections occur before commit; transport/commit errors remain ambiguous and retain owner protection.
        let saved = match self
            .storage
            .mutate_task_run(&self.key(run_id), &mutation)
            .await
        {
            Ok(saved) => saved,
            Err(AgentError::TaskRunStorage(
                TaskRunStorageError::AdmissionExpired
                | TaskRunStorageError::Conflict
                | TaskRunStorageError::NotResumable,
            )) => return Ok(None),
            Err(error) => return Err(error),
        };
        self.validate(&saved)?;
        Ok(Some(saved))
    }

    /// Objective control uses the same validated host CAS on both scoped backends.
    fn supports_objective_revision(&self) -> bool {
        true
    }

    /// Only a no-write protocol rejection is recoverable as None; transport and receipt errors remain ambiguous.
    async fn apply_host_control(
        &self,
        run_id: &str,
        expected_revision: u64,
        action: super::HostControlAction,
        host: &super::AutonomyHostCeilings,
        deadline: Option<std::time::Instant>,
    ) -> Result<Option<TaskRunSnapshot>> {
        match self
            .apply_host_action_with_deadline(run_id, expected_revision, action, host, deadline)
            .await
        {
            Ok(saved) => Ok(Some(saved)),
            Err(AgentError::TaskRunStorage(
                TaskRunStorageError::Conflict
                | TaskRunStorageError::AdmissionExpired
                | TaskRunStorageError::NotResumable,
            )) => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Actorless scope excludes actor-owned runs even though the raw backend supports broad listing.
    async fn list(&self) -> Result<Vec<TaskRunSummary>> {
        let rows = self
            .storage
            .list_task_runs(&TaskRunFilter {
                agent_id: self.agent_id.clone(),
                actor_id: self.actor_id.clone(),
                ..Default::default()
            })
            .await?;
        Ok(rows
            .into_iter()
            .filter(|s| s.key.agent_id == self.agent_id && s.actor_id == self.actor_id)
            .collect())
    }

    /// Rechecks scope before deleting; the backend also compares revision and active ownership.
    async fn delete(&self, run_id: &str, revision: u64) -> Result<()> {
        self.load(run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        self.storage
            .delete_task_run(&self.key(run_id), revision)
            .await
    }
}

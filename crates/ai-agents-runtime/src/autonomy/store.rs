//! Scoped task persistence validates recovery data without invoking the runtime or any tool.

use super::run::TaskCheckpointPayload;
use ai_agents_core::autonomy::*;
use ai_agents_core::{AgentError, AgentStorage, Result, StorageCapability};
use async_trait::async_trait;
use std::sync::Arc;

/// Storage operations are CAS transitions; successful claim is required before later execution.
#[async_trait]
pub trait TaskRunStore: Send + Sync {
    /// Creates exact recovery data after runtime schema validation.
    async fn create(&self, snapshot: &TaskRunSnapshot) -> Result<()>;
    /// Returns only compatible data in this adapter's agent and actor scope.
    async fn load(&self, run_id: &str) -> Result<Option<TaskRunSnapshot>>;
    /// Applies a conditional operation without executing external work.
    async fn mutate(&self, run_id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot>;
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

    // Never accept a caller-selected agent or actor identity as authority for another scope.
    fn validate(&self, snapshot: &TaskRunSnapshot) -> Result<()> {
        if snapshot.key.agent_id != self.agent_id || snapshot.actor_id != self.actor_id {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        snapshot.validate()?;
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())
            .map_err(|_| TaskRunStorageError::InvalidCheckpoint)?;
        payload.validate(snapshot, &self.config_identity)
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

    /// Checks the proposed payload before CAS; concurrent changes still fail in the backend.
    async fn mutate(&self, run_id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot> {
        let previous = self
            .load(run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        let proposed = previous.transition(mutation, chrono::Utc::now())?;
        self.validate(&proposed)?;
        let old: TaskCheckpointPayload = serde_json::from_value(previous.payload.clone())
            .map_err(|_| TaskRunStorageError::InvalidCheckpoint)?;
        let new: TaskCheckpointPayload = serde_json::from_value(proposed.payload.clone())
            .map_err(|_| TaskRunStorageError::InvalidCheckpoint)?;
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
        Ok(saved)
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

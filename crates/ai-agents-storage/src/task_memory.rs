//! Process-local task persistence with the same conditional transitions as SQLite.

use std::collections::HashMap;
use std::sync::Mutex;

use ai_agents_core::autonomy::{
    TaskRunFilter, TaskRunKey, TaskRunMutation, TaskRunSnapshot, TaskRunStorageError,
    TaskRunSummary,
};
use ai_agents_core::{AgentError, AgentSnapshot, AgentStorage, Result, StorageCapability};
use async_trait::async_trait;

/// Volatile task storage; claims and checkpoints survive only while this instance is retained.
#[derive(Default)]
pub struct InMemoryTaskStorage {
    // None is a deletion tombstone, preventing identity reuse and stale-writer resurrection.
    runs: Mutex<HashMap<TaskRunKey, Option<TaskRunSnapshot>>>,
}

#[async_trait]
impl AgentStorage for InMemoryTaskStorage {
    fn supports(&self, capability: StorageCapability) -> bool {
        capability == StorageCapability::TaskRuns
    }

    async fn save(&self, _: &str, _: &AgentSnapshot) -> Result<()> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::Snapshot,
        ))
    }
    async fn load(&self, _: &str) -> Result<Option<AgentSnapshot>> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::Snapshot,
        ))
    }
    async fn delete(&self, _: &str) -> Result<()> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::Snapshot,
        ))
    }
    async fn list_sessions(&self) -> Result<Vec<String>> {
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::Snapshot,
        ))
    }

    async fn create_task_run(&self, snapshot: &TaskRunSnapshot) -> Result<()> {
        snapshot.validate_create()?;
        let mut runs = self
            .runs
            .lock()
            .map_err(|_| AgentError::Persistence("task store lock poisoned".into()))?;
        if runs.contains_key(&snapshot.key) {
            return Err(TaskRunStorageError::AlreadyExists.into());
        }
        runs.insert(snapshot.key.clone(), Some(snapshot.clone()));
        Ok(())
    }

    async fn load_task_run(&self, key: &TaskRunKey) -> Result<Option<TaskRunSnapshot>> {
        Ok(self
            .runs
            .lock()
            .map_err(|_| AgentError::Persistence("task store lock poisoned".into()))?
            .get(key)
            .cloned()
            .flatten())
    }

    async fn list_task_runs(&self, filter: &TaskRunFilter) -> Result<Vec<TaskRunSummary>> {
        let runs = self
            .runs
            .lock()
            .map_err(|_| AgentError::Persistence("task store lock poisoned".into()))?;
        let mut summaries: Vec<_> = runs
            .values()
            .flatten()
            .filter(|s| {
                s.key.agent_id == filter.agent_id
                    && filter
                        .actor_id
                        .as_ref()
                        .is_none_or(|a| s.actor_id.as_ref() == Some(a))
                    && filter.status.is_none_or(|status| status == s.status)
            })
            .map(TaskRunSnapshot::summary)
            .collect();
        summaries.sort_by(|a, b| a.key.run_id.cmp(&b.key.run_id));
        if let Some(limit) = filter.limit {
            summaries.truncate(limit);
        }
        Ok(summaries)
    }

    async fn mutate_task_run(
        &self,
        key: &TaskRunKey,
        mutation: &TaskRunMutation,
    ) -> Result<TaskRunSnapshot> {
        let mut runs = self
            .runs
            .lock()
            .map_err(|_| AgentError::Persistence("task store lock poisoned".into()))?;
        let previous = runs
            .get(key)
            .and_then(Option::as_ref)
            .ok_or(TaskRunStorageError::NotFound)?;
        let next = previous.transition(mutation, chrono::Utc::now())?;
        runs.insert(key.clone(), Some(next.clone()));
        Ok(next)
    }

    async fn delete_task_run(&self, key: &TaskRunKey, expected_revision: u64) -> Result<()> {
        let mut runs = self
            .runs
            .lock()
            .map_err(|_| AgentError::Persistence("task store lock poisoned".into()))?;
        let previous = runs
            .get(key)
            .and_then(Option::as_ref)
            .ok_or(TaskRunStorageError::NotFound)?;
        if previous.revision != expected_revision {
            return Err(TaskRunStorageError::Conflict.into());
        }
        if previous.owner_token.is_some() {
            return Err(TaskRunStorageError::Owned.into());
        }
        runs.insert(key.clone(), None);
        Ok(())
    }
}

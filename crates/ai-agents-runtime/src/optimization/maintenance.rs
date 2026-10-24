use std::collections::HashMap;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use parking_lot::Mutex;

use ai_agents_core::{AgentError, Result};

/// Runtime work categories used for optimization and background ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuntimeTaskPurpose {
    /// Main user-visible response generation.
    MainResponse,
    /// State transition selection or commit work.
    StateTransition,
    /// Skill routing decision work.
    SkillRouting,
    /// Automatic reasoning mode decision work.
    ReasoningJudge,
    /// Post-turn fact extraction.
    PostTurnFacts,
    /// Post-turn relationship update.
    PostTurnRelationship,
    /// Post-turn session maintenance.
    PostTurnSessionMaintenance,
    /// Post-turn memory compression.
    PostTurnCompression,
    /// Orchestration vote extraction.
    OrchestrationVoteExtraction,
    /// Observability export work.
    ObservabilityExport,
}

/// Sequence key used to preserve order for actor or session scoped maintenance.
#[derive(Debug, Clone, Eq)]
pub struct MaintenanceSequenceKey {
    /// Agent whose maintenance task owns the sequence.
    pub agent_id: String,
    /// Actor, session, or resource identifier for ordering.
    pub scope_id: String,
    /// Task kind that must remain ordered within the scope.
    pub task_kind: RuntimeTaskPurpose,
}

impl MaintenanceSequenceKey {
    /// Creates an actor-scoped sequence key.
    pub fn actor(
        agent_id: impl Into<String>,
        actor_id: impl Into<String>,
        task_kind: RuntimeTaskPurpose,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            scope_id: actor_id.into(),
            task_kind,
        }
    }
}

impl PartialEq for MaintenanceSequenceKey {
    fn eq(&self, other: &Self) -> bool {
        self.agent_id == other.agent_id
            && self.scope_id == other.scope_id
            && self.task_kind == other.task_kind
    }
}

impl Hash for MaintenanceSequenceKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.agent_id.hash(state);
        self.scope_id.hash(state);
        self.task_kind.hash(state);
    }
}

struct TrackedTask {
    key: Option<MaintenanceSequenceKey>,
    finished: tokio::task::AbortHandle,
    // The queue retains the join future while a flush borrows a clone, including when that flush is dropped.
    completion: Shared<BoxFuture<'static, std::result::Result<(), Arc<str>>>>,
}

/// Bounded queue for post-turn maintenance that can be flushed by eval and shutdown paths.
pub struct BackgroundMaintenanceQueue {
    max_tasks: usize,
    tasks: Mutex<Vec<Arc<TrackedTask>>>,
    locks: Mutex<HashMap<MaintenanceSequenceKey, Arc<tokio::sync::Mutex<()>>>>,
}

impl BackgroundMaintenanceQueue {
    /// Creates a queue with a positive task limit.
    pub fn new(max_tasks: usize) -> Self {
        Self {
            max_tasks: max_tasks.max(1),
            tasks: Mutex::new(Vec::new()),
            locks: Mutex::new(HashMap::new()),
        }
    }

    /// Returns the number of tracked tasks that have not been flushed yet.
    pub fn len(&self) -> usize {
        self.tasks.lock().len()
    }

    /// Returns true when no tracked tasks remain.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true when the queue cannot accept another tracked task.
    pub fn is_full(&self) -> bool {
        self.unfinished_count() >= self.max_tasks
    }

    /// Spawns bounded work and preserves the queued completion until an acknowledged drain, including across cancelled flushes.
    /// Same-key predecessors settle before later bodies start; failure does not silently detach admitted siblings.
    pub fn spawn<F>(&self, key: Option<MaintenanceSequenceKey>, future: F) -> Result<()>
    where
        F: Future<Output = Result<()>> + Send + 'static,
    {
        let mut tasks = self.tasks.lock();
        if tasks
            .iter()
            .filter(|task| !task.finished.is_finished())
            .count()
            >= self.max_tasks
        {
            return Err(AgentError::Other(format!(
                "background maintenance queue is full (limit {})",
                self.max_tasks
            )));
        }

        let lock = key.as_ref().map(|key| {
            let mut locks = self.locks.lock();
            locks
                .entry(key.clone())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                .clone()
        });

        let predecessor = key.as_ref().and_then(|key| {
            tasks
                .iter()
                .rev()
                .find(|task| task.key.as_ref() == Some(key))
                .map(|task| task.completion.clone())
        });
        let handle = tokio::spawn(async move {
            if let Some(predecessor) = predecessor {
                let _ = predecessor.await;
            }
            if let Some(lock) = lock {
                let _guard = lock.lock().await;
                future.await
            } else {
                future.await
            }
        });
        let finished = handle.abort_handle();
        let completion = async move {
            match handle.await {
                Ok(result) => result.map_err(|error| Arc::<str>::from(error.to_string())),
                Err(error) => Err(Arc::<str>::from(format!(
                    "background maintenance task failed to join: {error}"
                ))),
            }
        }
        .boxed()
        .shared();
        tasks.push(Arc::new(TrackedTask {
            key,
            finished,
            completion,
        }));
        Ok(())
    }

    /// Drains the captured admission set without removing live work from tracking or abandoning siblings on error.
    /// A dropped drain can be retried; later admissions remain tracked for the caller's next barrier.
    pub async fn flush_all(&self) -> Result<()> {
        let selected = self.tasks.lock().clone();
        self.drain(selected).await
    }

    /// Waits for tasks with the requested scope identifier and keeps unrelated work queued.
    pub async fn flush_scope(&self, scope_id: &str) -> Result<()> {
        self.flush_matching(|key| key.scope_id == scope_id).await
    }

    /// Waits for tasks with the requested purpose and keeps unrelated work queued.
    pub async fn flush_purpose(&self, purpose: RuntimeTaskPurpose) -> Result<()> {
        self.flush_matching(|key| key.task_kind == purpose).await
    }

    /// Waits for tasks with the requested scope and purpose.
    pub async fn flush_scope_purpose(
        &self,
        scope_id: &str,
        purpose: RuntimeTaskPurpose,
    ) -> Result<()> {
        self.flush_matching(|key| key.scope_id == scope_id && key.task_kind == purpose)
            .await
    }

    // Selection and retirement never replace the queue, so concurrent producers cannot lose admitted work.
    async fn flush_matching(
        &self,
        matches_key: impl Fn(&MaintenanceSequenceKey) -> bool,
    ) -> Result<()> {
        let matching = self
            .tasks
            .lock()
            .iter()
            .filter(|task| task.key.as_ref().is_some_and(&matches_key))
            .cloned()
            .collect();
        self.drain(matching).await
    }

    // Keep the entire selected set until every completion is known; cancellation leaves cached errors and live joins retrievable.
    async fn drain(&self, selected: Vec<Arc<TrackedTask>>) -> Result<()> {
        let mut failure = None;
        for task in &selected {
            if let Err(error) = task.completion.clone().await {
                failure.get_or_insert(error);
            }
        }
        self.tasks
            .lock()
            .retain(|task| !selected.iter().any(|done| Arc::ptr_eq(task, done)));
        match failure {
            Some(error) => Err(AgentError::Other(error.to_string())),
            None => Ok(()),
        }
    }

    // In-progress drains still occupy capacity until their underlying tasks actually return.
    fn unfinished_count(&self) -> usize {
        self.tasks
            .lock()
            .iter()
            .filter(|task| !task.finished.is_finished())
            .count()
    }
}

impl Default for BackgroundMaintenanceQueue {
    fn default() -> Self {
        Self::new(16)
    }
}

#[cfg(test)]
#[path = "maintenance_ad_tests.rs"]
mod ad_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn finished_task_error_surfaces_on_flush_after_capacity_check() {
        let queue = BackgroundMaintenanceQueue::new(1);
        queue
            .spawn(None, async {
                Err(AgentError::Other("background failed".to_string()))
            })
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        assert!(!queue.is_full());
        let error = queue.flush_all().await.unwrap_err();
        assert!(error.to_string().contains("background failed"));
    }

    #[tokio::test]
    async fn flush_scope_purpose_keeps_unmatched_tasks() {
        let queue = BackgroundMaintenanceQueue::new(2);
        let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
        queue
            .spawn(
                Some(MaintenanceSequenceKey::actor(
                    "agent",
                    "actor",
                    RuntimeTaskPurpose::PostTurnFacts,
                )),
                async { Ok(()) },
            )
            .unwrap();
        queue
            .spawn(
                Some(MaintenanceSequenceKey::actor(
                    "agent",
                    "actor",
                    RuntimeTaskPurpose::PostTurnRelationship,
                )),
                async move {
                    let _ = release_rx.await;
                    Ok(())
                },
            )
            .unwrap();

        queue
            .flush_scope_purpose("actor", RuntimeTaskPurpose::PostTurnFacts)
            .await
            .unwrap();
        assert_eq!(queue.len(), 1);
        let _ = release_tx.send(());
        queue.flush_all().await.unwrap();
        assert!(queue.is_empty());
    }
}

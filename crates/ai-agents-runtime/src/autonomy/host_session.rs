//! Foreground host sessions retain actual participating controllers, not a registry-wide execution permit.

use super::{AutonomyRunner, TaskRunKey};
use ai_agents_core::{AgentError, Result};
use std::{
    collections::BTreeMap,
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

/// One foreground host's live task catalogue and response-endpoint lifetime.
/// Scope the original operation and its controls with this handle; framework-spawned composition inherits it explicitly.
/// It owns no executor, task authority, deadlines or cleanup acknowledgement, and cannot be deserialized.
#[derive(Clone)]
pub struct TaskHostSession(Arc<SessionState>);
struct SessionState {
    available: AtomicBool,
    tasks: parking_lot::Mutex<BTreeMap<String, TaskHostRun>>,
    changed: tokio::sync::Notify,
}

/// An actual configured controller admitted inside this session, never reconstructed from a display identifier.
#[derive(Clone)]
pub struct TaskHostRun {
    pub key: TaskRunKey,
    pub runner: Arc<AutonomyRunner>,
    published: Arc<std::sync::atomic::AtomicU64>,
    admitted: Arc<AtomicBool>,
}

impl TaskHostRun {
    /// Reports installation of the actual execution handle, not current admission authority or public pause readiness.
    pub fn execution_admitted(&self) -> bool {
        self.admitted.load(Ordering::Acquire)
    }

    /// Returns the last public-result ACK; intermediate checkpoint revisions cannot authorize a response UI.
    pub fn published_revision(&self) -> Option<u64> {
        let revision = self.published.load(Ordering::Acquire);
        (revision != u64::MAX).then_some(revision)
    }
}

tokio::task_local! { static SESSION: TaskHostSession; static SUPPLIED_RUNNER: Arc<AutonomyRunner>; }

// A lexical SDK handle must be the exact executing controller, never a registry or serialized identity match.
pub(crate) fn scoped_runner(actual: &AutonomyRunner) -> Option<Arc<AutonomyRunner>> {
    SUPPLIED_RUNNER
        .try_with(Clone::clone)
        .ok()
        .filter(|runner| std::ptr::eq(actual, Arc::as_ptr(runner)))
}

impl TaskHostSession {
    /// Declares whether the foreground host can service responses; availability can only be withdrawn for this operation.
    pub fn new(available: bool) -> Self {
        Self(Arc::new(SessionState {
            available: AtomicBool::new(available),
            tasks: Default::default(),
            changed: Default::default(),
        }))
    }

    /// Polls work under this live catalogue without starting, detaching or adopting any task.
    pub async fn scope<F: Future>(&self, work: F) -> F::Output {
        SESSION.scope(self.clone(), work).await
    }

    /// Scopes an actual supplied SDK runner so its newly admitted runs join this host's cancellation and cleanup catalogue.
    /// The runtime still checks pointer identity and registers only after its real entry; this does not adopt checkpoints or grant execution.
    pub async fn scope_runner<F: Future>(&self, runner: Arc<AutonomyRunner>, work: F) -> F::Output {
        self.scope(SUPPLIED_RUNNER.scope(runner, work)).await
    }

    /// Reports confirmed endpoint loss, not silence, and never fabricates approval or cancellation acknowledgement.
    pub fn close_input(&self) {
        self.0.available.store(false, Ordering::Release);
        self.notify();
    }

    /// Reports only this host endpoint; it is not a statement about durable restoration or remote delivery.
    pub fn interaction_available(&self) -> bool {
        self.0.available.load(Ordering::Acquire)
    }

    /// Returns participating configured runners only; unrelated registered agents are never discovered or controlled.
    pub fn runs(&self) -> Vec<TaskHostRun> {
        self.0.tasks.lock().values().cloned().collect()
    }

    /// Reports exact retained ownership among this session's participants; a stored terminal row alone is not cleanup evidence.
    pub fn has_protected_work(&self) -> bool {
        self.0.tasks.lock().values().any(|run| {
            run.runner
                .runtime()
                .captured_task_owner()
                .is_some_and(|owner| owner.run_id == run.key.run_id)
        })
    }

    /// Attaches an actual SDK operation result, including an explicitly reconstructed runner, to this foreground host.
    /// A checkpoint or ID is insufficient: revision, key and local retained pause readiness are checked without executing or granting work.
    pub async fn observe_result(
        &self,
        runner: Arc<AutonomyRunner>,
        result: &super::TaskRunResult,
    ) -> Result<()> {
        let run = &result.run;
        let actual = runner
            .task_inspection(&run.key.run_id, run.revision)
            .await?;
        if actual.run.key != run.key
            || actual.run.status != run.status
            || (!run.status.is_terminal()
                && (run.status != super::TaskRunStatus::Paused || !actual.locally_retained))
        {
            return Err(super::TaskRunStorageError::NotResumable.into());
        }
        if run.status == super::TaskRunStatus::Paused {
            runner
                .task_interaction(&run.key.run_id, run.revision)
                .await?;
        }
        if let Some(existing) = self.0.tasks.lock().get(&run.key.run_id)
            && (existing.key != run.key || !Arc::ptr_eq(&existing.runner, &runner))
        {
            return Err(super::TaskRunStorageError::Conflict.into());
        }
        self.register(runner, run.key.clone())?;
        self.admitted(&run.key.run_id);
        self.publish(&run.key.run_id, run.revision);
        Ok(())
    }

    /// Wakes for admission or acknowledged automatic publication; callers still read the exact current scoped snapshot.
    pub async fn changed(&self) {
        self.0.changed.notified().await;
    }

    // Admission is bounded per foreground session and pins the actual runner; labels cannot replace the live binding.
    pub(crate) fn register(&self, runner: Arc<AutonomyRunner>, key: TaskRunKey) -> Result<()> {
        let mut tasks = self.0.tasks.lock();
        if tasks.len() >= 1024 && !tasks.contains_key(&key.run_id) {
            return Err(AgentError::Config(
                "foreground task catalogue exceeds 1024 runs".into(),
            ));
        }
        tasks.insert(
            key.run_id.clone(),
            TaskHostRun {
                key,
                runner,
                published: Arc::new(std::sync::atomic::AtomicU64::new(u64::MAX)),
                admitted: Arc::new(AtomicBool::new(false)),
            },
        );
        drop(tasks);
        self.notify();
        Ok(())
    }

    // Shutdown waits for this actual handle rather than marking an initializing row as a failed cancellation attempt.
    pub(crate) fn admitted(&self, run_id: &str) {
        if let Some(run) = self.0.tasks.lock().get(run_id) {
            run.admitted.store(true, Ordering::Release);
        }
        self.notify();
    }

    // A safe prompt is exposed only after the public-result boundary has finished persistence/cleanup preparation.
    pub(crate) fn publish(&self, run_id: &str, revision: u64) {
        if let Some(run) = self.0.tasks.lock().get(run_id) {
            let _ = run
                .published
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                    Some(if old == u64::MAX {
                        revision
                    } else {
                        old.max(revision)
                    })
                });
        }
        self.notify();
    }

    // Publication is observation only; the runner remains responsible for admission, custody and exact ACK ordering.
    pub(crate) fn notify(&self) {
        self.0.changed.notify_one();
    }
}

// Framework spawn boundaries capture this actual host session with their existing invocation context.
pub(crate) fn current_session() -> Option<TaskHostSession> {
    SESSION.try_with(Clone::clone).ok()
}

// Absence preserves embedded callers' configured delivery contract rather than declaring their endpoint unavailable.
pub(crate) async fn scope_session<F: Future>(
    session: Option<TaskHostSession>,
    work: F,
) -> F::Output {
    if let Some(session) = session {
        session.scope(work).await
    } else {
        work.await
    }
}

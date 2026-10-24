//! Reconstruction fault tests distinguish certified no-write refusal from ambiguous committed custody.
#[path = "ev_restore_tests.rs"]
mod event_acceptance;
#[path = "restoration_fault_matrix_tests.rs"]
mod matrix_tests;

use super::restoration_tests::{RestoreHost, config, host};
use super::*;
use ai_agents_core::{AgentError, Result};
use async_trait::async_trait;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

struct RestoreFaultStore {
    inner: Arc<ScopedTaskRunStore>,
    phase: &'static str,
    mode: &'static str,
    armed: AtomicBool,
    reached: tokio::sync::Semaphore,
    host: Arc<RestoreHost>,
    owner: parking_lot::Mutex<std::sync::Weak<RunOwner>>,
    release: tokio::sync::Semaphore,
}
impl RestoreFaultStore {
    // One selected protocol boundary fires once; unrelated journal writes use the production scoped adapter.
    fn new(
        inner: Arc<ScopedTaskRunStore>,
        phase: &'static str,
        mode: &'static str,
        host: Arc<RestoreHost>,
    ) -> Arc<Self> {
        Arc::new(Self {
            inner,
            phase,
            mode,
            armed: AtomicBool::new(true),
            reached: tokio::sync::Semaphore::new(0),
            host,
            owner: Default::default(),
            release: tokio::sync::Semaphore::new(0),
        })
    }
    // Barrier placement distinguishes a rejected write, a before-commit binding race and commit-before-ACK ambiguity.
    async fn fault<T>(&self, commit: impl std::future::Future<Output = Result<T>>) -> Result<T> {
        if self.mode == "reject" {
            self.reached.add_permits(1);
            return Err(AgentError::Persistence("rejected restoration write".into()));
        }
        if self.mode == "before" {
            self.reached.add_permits(1);
            self.release.acquire().await.unwrap().forget();
        }
        let value = commit.await?;
        self.reached.add_permits(1);
        if self.mode == "drop" {
            std::future::pending::<()>().await;
        }
        if self.mode == "lost" {
            return Err(AgentError::Persistence(
                "lost restoration acknowledgement".into(),
            ));
        }
        Ok(value)
    }
}
#[async_trait]
impl TaskRunStore for RestoreFaultStore {
    async fn create(&self, snapshot: &TaskRunSnapshot) -> Result<()> {
        self.inner.create(snapshot).await
    }
    async fn load(&self, id: &str) -> Result<Option<TaskRunSnapshot>> {
        self.inner.load(id).await
    }
    async fn list(&self) -> Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    async fn delete(&self, id: &str, revision: u64) -> Result<()> {
        self.inner.delete(id, revision).await
    }
    // Only the selected restoration review publication faults; unrelated journal transitions remain production writes.
    async fn mutate(&self, id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot> {
        let review = matches!(mutation,TaskRunMutation::FinalCheckpoint {status:TaskRunStatus::Paused,payload,..} if payload["adapters"].as_array().is_some_and(|adapters|adapters.iter().any(|adapter|adapter["id"]=="runtime.restore_review" && adapter["state"]["approved"]==true)));
        if self.phase == "review" && review && self.armed.swap(false, Ordering::SeqCst) {
            return self.fault(self.inner.mutate(id, mutation)).await;
        }
        self.inner.mutate(id, mutation).await
    }
    fn supports_reconstruction(&self) -> bool {
        true
    }
    async fn claim_paused(
        &self,
        id: &str,
        revision: u64,
        token: String,
    ) -> Result<Option<TaskRunSnapshot>> {
        if self.phase == "claim" && self.armed.swap(false, Ordering::SeqCst) {
            return self
                .fault(self.inner.claim_paused(id, revision, token))
                .await;
        }
        self.inner.claim_paused(id, revision, token).await
    }
    // Weak authority witnesses detect leaked guard cycles without extending failed staging custody.
    async fn checkpoint_reconstruction(
        &self,
        id: &str,
        checkpoint: &TaskReconstructionCheckpoint,
    ) -> Result<TaskRunSnapshot> {
        if let Some(runtime) = self
            .host
            .staged
            .lock()
            .values()
            .find_map(std::sync::Weak::upgrade)
            && let Some(owner) = runtime.captured_task_owner()
        {
            *self.owner.lock() = Arc::downgrade(&owner);
        }
        if self.phase == "checkpoint" && self.armed.swap(false, Ordering::SeqCst) {
            return self
                .fault(self.inner.checkpoint_reconstruction(id, checkpoint))
                .await;
        }
        self.inner.checkpoint_reconstruction(id, checkpoint).await
    }
}

// Prepare the source with the ordinary store, then replace only the new restorer's adapter.
async fn fixture(
    phase: &'static str,
    mode: &'static str,
    sqlite: bool,
) -> (
    Arc<RestoreHost>,
    Arc<ScopedTaskRunStore>,
    Arc<RestoreFaultStore>,
    TaskRunResult,
) {
    let host = host();
    let root = host.runtime(host.root.clone());
    let inner = super::continuation_test_support::store(root.info().id, "restore.v1", sqlite).await;
    let runner = AutonomyRunner::try_new(
        root,
        config(),
        Default::default(),
        inner.clone(),
        "restore.v1".into(),
    )
    .unwrap();
    let paused = runner.run("verify evidence", None).await.unwrap();
    let prepared = runner
        .checkpoint_for_restore(&paused.run.key.run_id, paused.run.revision, host.as_ref())
        .await
        .unwrap();
    drop(runner);
    let fault = RestoreFaultStore::new(inner.clone(), phase, mode, host.clone());
    (host, inner, fault, prepared)
}
fn request(prepared: &TaskRunResult) -> TaskRestoreRequest {
    TaskRestoreRequest {
        run_id: prepared.run.key.run_id.clone(),
        expected_revision: prepared.run.revision,
        config_identity: "restore.v1".into(),
        config: config(),
        ceilings: Default::default(),
    }
}

// No-write refusal preserves the source pause; committed claim ambiguity keeps one owned Running row and never constructs business work.
#[tokio::test]
async fn restoration_claim_reject_lost_and_drop_preserve_truthful_custody() {
    for &sqlite in super::continuation_test_support::backends() {
        claim_faults(sqlite).await;
    }
}

// Both conditional backends must preserve the same distinction between no-write rejection and ambiguous committed claim.
async fn claim_faults(sqlite: bool) {
    for mode in ["reject", "lost", "drop"] {
        let (host, inner, fault, prepared) = fixture("claim", mode, sqlite).await;
        let mut work = Box::pin(AutonomyRunner::restore_paused(
            request(&prepared),
            fault.clone(),
            host.clone(),
        ));
        if mode == "drop" {
            tokio::select! {result=&mut work=>panic!("returned before held claim: {:?}",result.err()),permit=fault.reached.acquire()=>permit.unwrap().forget()}
            drop(work);
        } else {
            assert!(work.await.is_err());
        }
        let saved = inner.load(&prepared.run.key.run_id).await.unwrap().unwrap();
        assert_eq!(
            saved.status,
            if mode == "reject" {
                TaskRunStatus::Paused
            } else {
                TaskRunStatus::Running
            }
        );
        assert_eq!(saved.owner_token.is_some(), mode != "reject");
        assert_eq!(host.stages.load(Ordering::SeqCst), 0);
        assert_eq!(host.calls.load(Ordering::SeqCst), 1);
        assert_eq!(host.effects.load(Ordering::SeqCst), 1);
    }
}

// A final write rejection leaves the prepaid claimed journal; commit-then-error/drop leaves an exact restored Paused row but no published runner.
#[tokio::test]
async fn restoration_checkpoint_reject_lost_and_drop_never_replay_or_report_success() {
    for &sqlite in super::continuation_test_support::backends() {
        checkpoint_faults(sqlite).await;
    }
}

// ACK ambiguity is not permission to replay or leak orphan authority on either backend.
async fn checkpoint_faults(sqlite: bool) {
    for mode in ["reject", "lost", "drop"] {
        let (host, inner, fault, prepared) = fixture("checkpoint", mode, sqlite).await;
        let mut work = Box::pin(AutonomyRunner::restore_paused(
            request(&prepared),
            fault.clone(),
            host.clone(),
        ));
        if mode == "drop" {
            tokio::select! {result=&mut work=>panic!("returned before held checkpoint: {:?}",result.err()),permit=fault.reached.acquire()=>permit.unwrap().forget()}
            drop(work);
        } else {
            assert!(work.await.is_err());
        }
        let saved = inner.load(&prepared.run.key.run_id).await.unwrap().unwrap();
        assert_eq!(
            saved.status,
            if mode == "reject" {
                TaskRunStatus::Running
            } else {
                TaskRunStatus::Paused
            }
        );
        if mode == "reject" {
            assert!(saved.owner_token.is_some());
            assert_eq!(
                saved.payload["adapters"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|adapter| adapter["id"] == "runtime.restore_work")
                    .unwrap()["state"]["phase"],
                "running"
            );
        } else {
            assert!(saved.owner_token.is_none());
        }
        assert_eq!(host.stages.load(Ordering::SeqCst), 1);
        assert_eq!(host.calls.load(Ordering::SeqCst), 1);
        assert_eq!(host.effects.load(Ordering::SeqCst), 1);
        assert!(
            fault.owner.lock().upgrade().is_none(),
            "failed unpublished restoration leaked an owner cycle: {mode}"
        );
    }
}

// The exact-revision storage claim admits one independent restorer; the loser cannot stage or publish topology.
#[tokio::test]
async fn restoration_competing_claims_construct_only_one_graph() {
    for &sqlite in super::continuation_test_support::backends() {
        competing_claims(sqlite).await;
    }
}

// Competing restorers share only the real backend and exact revision, never a local resume gate.
async fn competing_claims(sqlite: bool) {
    let (host, inner, _, prepared) = fixture("none", "positive", sqlite).await;
    host.stage_hold.store(true, Ordering::SeqCst);
    let first_host = host.clone();
    let first_store = inner.clone();
    let first_request = request(&prepared);
    let mut first = tokio::spawn(async move {
        AutonomyRunner::restore_paused(first_request, first_store, first_host).await
    });
    tokio::select! {result=&mut first=>panic!("first returned before stage: {:?}",result.err()),permit=host.stage_reached.acquire()=>permit.unwrap().forget()}
    assert!(
        AutonomyRunner::restore_paused(request(&prepared), inner.clone(), host.clone())
            .await
            .is_err()
    );
    assert_eq!(host.stages.load(Ordering::SeqCst), 1);
    host.stage_release.add_permits(1);
    let (runner, paused) = first.await.unwrap().unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    drop(runner);
}

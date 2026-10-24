//! Reconstruction fault tests distinguish certified no-write refusal from ambiguous committed custody.

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
}
impl RestoreFaultStore {
    // One selected protocol boundary fires once; unrelated journal writes use the production scoped adapter.
    fn new(inner: Arc<ScopedTaskRunStore>, phase: &'static str, mode: &'static str) -> Arc<Self> {
        Arc::new(Self {
            inner,
            phase,
            mode,
            armed: AtomicBool::new(true),
            reached: tokio::sync::Semaphore::new(0),
        })
    }
    async fn fault<T>(&self, commit: impl std::future::Future<Output = Result<T>>) -> Result<T> {
        if self.mode == "reject" {
            self.reached.add_permits(1);
            return Err(AgentError::Persistence("rejected restoration write".into()));
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
    async fn mutate(&self, id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot> {
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
    async fn checkpoint_reconstruction(
        &self,
        id: &str,
        checkpoint: &TaskReconstructionCheckpoint,
    ) -> Result<TaskRunSnapshot> {
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
) -> (
    Arc<RestoreHost>,
    Arc<ScopedTaskRunStore>,
    Arc<RestoreFaultStore>,
    TaskRunResult,
) {
    let host = host();
    let root = host.runtime(host.root.clone());
    let inner = super::continuation_test_support::store(root.info().id, "restore.v1", false).await;
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
    let fault = RestoreFaultStore::new(inner.clone(), phase, mode);
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
    for mode in ["reject", "lost", "drop"] {
        let (host, inner, fault, prepared) = fixture("claim", mode).await;
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
    for mode in ["reject", "lost", "drop"] {
        let (host, inner, fault, prepared) = fixture("checkpoint", mode).await;
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
    }
}

// The exact-revision storage claim admits one independent restorer; the loser cannot stage or publish topology.
#[tokio::test]
async fn restoration_competing_claims_construct_only_one_graph() {
    let (host, inner, _, prepared) = fixture("none", "positive").await;
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

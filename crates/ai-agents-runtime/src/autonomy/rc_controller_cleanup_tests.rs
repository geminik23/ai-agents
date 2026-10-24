//! Cleanup of serialized validation waits preserves bound data without manufacturing an executable driver witness.
use super::*;
use crate::autonomy::restoration_test_support::{FreshRuntime, RebuildHost};

pub(super) struct CleanupAckStore {
    pub(super) inner: Arc<ScopedTaskRunStore>,
    pub(super) lost: AtomicBool,
}
#[async_trait]
impl TaskRunStore for CleanupAckStore {
    // Only commit-before-ACK cleanup publication is faulted, not its source claim or ordinary typed retirement.
    async fn create(&self, s: &TaskRunSnapshot) -> Result<()> {
        self.inner.create(s).await
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
    async fn claim_cleanup(
        &self,
        id: &str,
        revision: u64,
        token: String,
    ) -> Result<Option<TaskRunSnapshot>> {
        self.inner.claim_cleanup(id, revision, token).await
    }
    async fn checkpoint_reconstruction(
        &self,
        id: &str,
        checkpoint: &TaskReconstructionCheckpoint,
    ) -> Result<TaskRunSnapshot> {
        let saved = self.inner.checkpoint_reconstruction(id, checkpoint).await?;
        if matches!(
            checkpoint.mutation(),
            TaskRunMutation::CleanupCheckpoint { .. }
        ) && self.lost.swap(false, Ordering::SeqCst)
        {
            return Err(AgentError::Persistence(
                "lost cleanup publication ACK".into(),
            ));
        }
        Ok(saved)
    }
}

// Tool-free external waits and parked real tool observations both cancel after expiry; ACK retry does not re-enter validators or providers.
#[tokio::test]
async fn rc_cancelled_controller_driver_cleanup_preserves_stored_bound_and_survives_lost_ack() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        for mode in ["external_tool_free", "host"] {
            let f = crate::autonomy::controller_tests::fixture(mode);
            let store = crate::autonomy::continuation_test_support::store(
                f.agent.info().id,
                "controller.v1",
                sqlite,
            )
            .await;
            let mut config = f.config.clone();
            config.defaults.max_wall_time_seconds = Some(1);
            let runner = AutonomyRunner::try_new(
                f.agent.clone(),
                config.clone(),
                Default::default(),
                store.clone(),
                "controller.v1".into(),
            )
            .unwrap();
            let witnesses = Arc::new(parking_lot::Mutex::new(vec![]));
            let stage_witness = witnesses.clone();
            let host = RebuildHost::new(
                "rc-controller-cleanup.v1",
                &FreshRuntime::new(f.agent.clone(), vec![f.provider.clone()]),
                Box::new(move || {
                    let witness = stage_witness.clone();
                    Box::pin(async move {
                        let fresh = crate::autonomy::controller_tests::fixture(mode);
                        witness.lock().push((
                            fresh.provider.clone(),
                            fresh.calls.clone(),
                            fresh.callbacks.clone(),
                        ));
                        FreshRuntime::new(fresh.agent.clone(), vec![fresh.provider.clone()])
                    })
                }),
            );
            let paused = runner
                .run("verify support domain evidence", None)
                .await
                .unwrap();
            let prepared = runner
                .checkpoint_for_restore(&paused.run.key.run_id, paused.run.revision, host.as_ref())
                .await
                .unwrap();
            drop(runner);
            let claimed = store
                .mutate(
                    &prepared.run.key.run_id,
                    &TaskRunMutation::Claim {
                        expected_revision: prepared.run.revision,
                        owner_token: "interrupted-controller-construction".into(),
                    },
                )
                .await
                .unwrap();
            let cancelled = store
                .mutate(
                    &claimed.key.run_id,
                    &TaskRunMutation::RequestCancel {
                        expected_revision: claimed.revision,
                    },
                )
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(1100)).await;
            let mut cleanup = decision(&cancelled, TaskRecoveryDisposition::CancelPrepared);
            cleanup.interrupted_started_at = Some(claimed.updated_at);
            let recovered = AutonomyRunner::reconcile_recovery(
                store.clone(),
                &claimed.key.run_id,
                cleanup,
                RecoveryProof::new(&cancelled),
            )
            .await
            .unwrap();
            let fault = Arc::new(CleanupAckStore {
                inner: store.clone(),
                lost: AtomicBool::new(true),
            });
            let request = || TaskRestoreRequest {
                run_id: recovered.key.run_id.clone(),
                expected_revision: recovered.revision,
                config_identity: "controller.v1".into(),
                config: config.clone(),
                ceilings: Default::default(),
            };
            assert!(
                AutonomyRunner::restore_cancelled(request(), fault, host.clone())
                    .await
                    .is_err()
            );
            let after_ack_loss = store.load(&recovered.key.run_id).await.unwrap().unwrap();
            assert_eq!(after_ack_loss.status, TaskRunStatus::Paused);
            assert!(after_ack_loss.cancel_requested);
            assert!(after_ack_loss.revision > recovered.revision);
            let done = AutonomyRunner::restore_cancelled(
                TaskRestoreRequest {
                    expected_revision: after_ack_loss.revision,
                    ..request()
                },
                store.clone(),
                host.clone(),
            )
            .await
            .unwrap();
            assert_eq!(done.run.status, TaskRunStatus::Cancelled);
            assert!(done.run.pending.is_none());
            assert_eq!(
                done.run.counters.llm_attempts,
                prepared.run.counters.llm_attempts
            );
            for (provider, calls, callbacks) in witnesses.lock().iter() {
                assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
                assert_eq!(calls.load(Ordering::SeqCst), 0);
                assert_eq!(callbacks.load(Ordering::SeqCst), 0);
            }
        }
    }
}

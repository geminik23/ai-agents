//! Controlled clock rollback is distinct from actual allowance exhaustion and never transfers or releases claim custody.
use super::*;

struct RollbackAfterStage {
    inner: Arc<RestoreHost>,
    rollback: chrono::DateTime<chrono::Utc>,
}
#[async_trait]
impl TaskRestoreHost for RollbackAfterStage {
    // Clock failure does not substitute the installed host or its reconstruction recipe.
    fn identity(&self) -> (String, u32) {
        self.inner.identity()
    }
    // Preserve the existing work allowance so rollback cannot be misclassified by a larger timeout.
    fn reconstruction_budget(&self) -> std::time::Duration {
        self.inner.reconstruction_budget()
    }
    // Actual runtime implementations and opaque state remain identical to the prepared source.
    fn runtime_binding(
        &self,
        runtime: &RuntimeAgent,
        declaration: Option<&AgentSpec>,
    ) -> ai_agents_core::Result<TaskRestoreBinding> {
        self.inner.runtime_binding(runtime, declaration)
    }
    // Keep the real provider binding rather than introducing a clock-only fake execution path.
    fn provider_binding(
        &self,
        providers: &ai_agents_llm::LLMRegistry,
    ) -> ai_agents_core::Result<TaskRestoreBinding> {
        self.inner.provider_binding(providers)
    }
    // Clock injection never bypasses the source recipe or native-history compatibility checks.
    fn validate_manifest(&self, manifest: &TaskRestoreManifest) -> ai_agents_core::Result<()> {
        self.inner.validate_manifest(manifest)
    }
    // Discover the same actual topology before changing only the explicit clock sample.
    fn capture_graph(&self, root: Arc<RuntimeAgent>) -> ai_agents_core::Result<TaskRestoreGraph> {
        self.inner.capture_graph(root)
    }
    // The stage is real and bounded; only its explicit host clock contract changes after construction.
    async fn stage(
        &self,
        manifest: &TaskRestoreManifest,
    ) -> ai_agents_core::Result<TaskRestoreGraph> {
        let graph = self.inner.stage(manifest).await?;
        *self.inner.clock.lock() = Some(self.rollback);
        Ok(graph)
    }
    // The deliberately returned rollback must reach the framework unchanged, not be clamped to metadata.
    fn now(&self) -> ai_agents_core::Result<chrono::DateTime<chrono::Utc>> {
        self.inner.now()
    }
}

// A short UTC reversal is not a five-second timeout: reject before claim or preserve the already charged claim without replay.
#[tokio::test]
async fn restoration_short_utc_rollback_is_typed_and_preserves_claim_phase_and_usage() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        for after_claim in [false, true] {
            let (host, store, prepared) = prepared_fixture(sqlite).await;
            let id = prepared.run.key.run_id.clone();
            let before = store.load(&id).await.unwrap().unwrap();
            let previous: TaskCheckpointPayload =
                serde_json::from_value(before.payload.clone()).unwrap();
            let rollback = before.updated_at - chrono::Duration::milliseconds(20);
            let calls = host.calls.load(Ordering::SeqCst);
            let effects = host.effects.load(Ordering::SeqCst);
            // Fixed fixture input makes the intended transaction phase independent of the OS clock.
            *host.clock.lock() = Some(if after_claim {
                before.updated_at + chrono::Duration::milliseconds(20)
            } else {
                rollback
            });
            let installed: Arc<dyn TaskRestoreHost> = if after_claim {
                Arc::new(RollbackAfterStage {
                    inner: host.clone(),
                    rollback,
                })
            } else {
                host.clone()
            };
            let request = || TaskRestoreRequest {
                run_id: id.clone(),
                expected_revision: before.revision,
                config_identity: "restore.v1".into(),
                config: config(),
                ceilings: Default::default(),
            };
            let error = AutonomyRunner::restore_paused(request(), store.clone(), installed.clone())
                .await
                .err()
                .expect("a backward certified clock cannot restore");
            assert!(
                matches!(
                    error,
                    ai_agents_core::AgentError::TaskRunStorage(
                        TaskRunStorageError::ClockDiscontinuity
                    )
                ),
                "{error:?}"
            );
            let saved = store.load(&id).await.unwrap().unwrap();
            assert_eq!(host.calls.load(Ordering::SeqCst), calls);
            assert_eq!(host.effects.load(Ordering::SeqCst), effects);
            if after_claim {
                assert_eq!(saved.status, TaskRunStatus::Running);
                assert!(saved.owner_token.is_some());
                assert!(saved.revision > before.revision);
                let payload: TaskCheckpointPayload =
                    serde_json::from_value(saved.payload.clone()).unwrap();
                assert_eq!(
                    payload.clocks.active_millis,
                    previous.clocks.active_millis + 5000
                );
                assert_eq!(
                    serde_json::to_value(&payload.counters).unwrap(),
                    serde_json::to_value(&previous.counters).unwrap()
                );
                assert_eq!(
                    payload
                        .adapters
                        .iter()
                        .find(|adapter| adapter.id == "runtime.restore_work")
                        .unwrap()
                        .state["phase"],
                    "running"
                );
                *host.clock.lock() = None;
                assert!(
                    AutonomyRunner::restore_paused(request(), store.clone(), installed)
                        .await
                        .is_err()
                );
                assert_eq!(host.stages.load(Ordering::SeqCst), 1);
                assert_eq!(
                    store.load(&id).await.unwrap().unwrap().revision,
                    saved.revision
                );
            } else {
                assert_eq!(
                    serde_json::to_value(saved).unwrap(),
                    serde_json::to_value(before).unwrap()
                );
                assert_eq!(host.stages.load(Ordering::SeqCst), 0);
            }
        }
    }
}

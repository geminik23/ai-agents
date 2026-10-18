use super::*;
use ai_agents_core::Result;
use std::sync::atomic::AtomicBool;

struct GateStore {
    inner: Arc<ScopedTaskRunStore>,
    armed: AtomicBool,
    reached: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
}

#[async_trait]
impl TaskRunStore for GateStore {
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
    // The backend commits the owned controller position before the transport delays acknowledgement.
    async fn mutate(&self, id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot> {
        let selected = match mutation {
            TaskRunMutation::Checkpoint { payload, .. } => {
                payload["adapters"].as_array().is_some_and(|adapters| {
                    adapters.iter().any(|adapter| {
                        adapter["id"] == "runtime.validation"
                            && adapter["state"]["review"].is_object()
                    })
                })
            }
            _ => false,
        };
        let saved = self.inner.mutate(id, mutation).await?;
        if selected && self.armed.swap(false, Ordering::SeqCst) {
            self.reached.add_permits(1);
            self.release.acquire().await.unwrap().forget();
        }
        Ok(saved)
    }
}

#[tokio::test]
async fn live_controller_pause_publication_settles_cancellation_or_expiry_without_exposing_an_executable_wait()
 {
    for sqlite in support::backends() {
        for limit in [false, true] {
            for expire in [false, true] {
                let provider =
                    super::super::super::super::runner_tests::RecordingProvider::new(&["working"]);
                let (agent, _, _) =
                    super::super::super::super::runner_tests::fixture(provider.clone(), 1);
                let store = support::store(agent.info().id, "prepared-v1", *sqlite).await;
                let gated = Arc::new(GateStore {
                    inner: store.clone(),
                    armed: AtomicBool::new(true),
                    reached: tokio::sync::Semaphore::new(0),
                    release: tokio::sync::Semaphore::new(0),
                });
                let mut config =
                    super::super::super::super::runner_tests::config(if limit { 1 } else { 4 });
                if expire {
                    config.defaults.max_active_time_seconds = Some(1);
                }
                if limit {
                    config.defaults.on_limit =
                        Some(serde_yaml::from_str("action: pause_run").unwrap());
                } else {
                    config.defaults.premature_finish =
                        Some(serde_yaml::from_str("action: ask_user").unwrap());
                }
                let runner = live_runner(agent.clone(), gated.clone(), config, "prepared-v1");
                let running = runner.run("support", None);
                tokio::pin!(running);
                tokio::select! { _ = gated.reached.acquire() => {}, result = &mut running => panic!("no publication wait: {result:?}") }
                if expire {
                    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
                } else {
                    let listed = store.list().await.unwrap();
                    let snapshot = store.load(&listed[0].key.run_id).await.unwrap().unwrap();
                    store
                        .mutate(
                            &snapshot.key.run_id,
                            &TaskRunMutation::RequestCancel {
                                expected_revision: snapshot.revision,
                            },
                        )
                        .await
                        .unwrap();
                }
                gated.release.add_permits(1);
                let done = running.await.unwrap();
                assert_eq!(
                    done.run.status,
                    if expire {
                        TaskRunStatus::LimitReached
                    } else {
                        TaskRunStatus::Cancelled
                    }
                );
                assert!(done.run.pending.is_none());
                assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
                assert!(agent.set_context("released", json!(true)).is_ok());
            }
        }
    }
}

#[tokio::test]
async fn live_skill_explicit_strategy_profile_requires_real_scope_and_rejection_never_credits_the_segment()
 {
    for sqlite in support::backends() {
        let mut f = Box::pin(skill_fixture(false)).await;
        f.store = support::store(f.agent.info().id, "locations.v1", *sqlite).await;
        let mut config = super::super::super::super::runner_tests::config(8);
        config.defaults.execution = Some(serde_yaml::from_str("strategy: skill").unwrap());
        config.defaults.hitl =
            Some(serde_yaml::from_str("on_approval_required: pause_run").unwrap());
        f.runner = live_runner(f.agent.clone(), f.store.clone(), config, "locations.v1");
        assert!(f.runner.run("objective", None).await.is_err());
        let paused = f
            .runner
            .run_skill("objective", "script", None)
            .await
            .unwrap();
        let done = f
            .runner
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::Approval {
                    request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                    result: ApprovalResult::Rejected {
                        reason: Some("not authorized".into()),
                    },
                },
            )
            .await
            .unwrap();
        assert_ne!(done.run.status, TaskRunStatus::Completed);
        assert_eq!(f.second.load(Ordering::SeqCst), 0);
        assert_eq!(f.first.load(Ordering::SeqCst), 1);
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
        assert!(f.agent.set_context("released", json!(true)).is_ok());
    }
}

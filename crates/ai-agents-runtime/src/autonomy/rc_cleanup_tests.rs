//! Cancellation remains monotonic through interrupted construction, while ordinary omitted persistence adds no new storage boundary.
use super::*;
#[path = "rc_controller_cleanup_tests.rs"]
mod controller;

// Expiry prevents pending execution but cannot strand known-safe native history after explicit cancellation and recovery review.
#[tokio::test]
async fn rc_cancelled_interrupted_constructor_rebuilds_only_for_typed_cleanup_even_after_expiry() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        let host = host();
        let root = host.runtime(host.root.clone());
        let store =
            crate::autonomy::continuation_test_support::store(root.info().id, "restore.v1", sqlite)
                .await;
        let mut config = config();
        config.defaults.max_wall_time_seconds = Some(1);
        let runner = AutonomyRunner::try_new(
            root,
            config.clone(),
            Default::default(),
            store.clone(),
            "restore.v1".into(),
        )
        .unwrap();
        let paused = runner.run("support cleanup", None).await.unwrap();
        let prepared = runner
            .checkpoint_for_restore(&paused.run.key.run_id, paused.run.revision, host.as_ref())
            .await
            .unwrap();
        drop(runner);
        host.stage_hold.store(true, Ordering::SeqCst);
        let task = tokio::spawn({
            let host = host.clone();
            let store = store.clone();
            let request = request(&prepared.run, config.clone());
            async move { AutonomyRunner::restore_paused(request, store, host).await }
        });
        host.stage_reached.acquire().await.unwrap().forget();
        task.abort();
        assert!(matches!(task.await, Err(error) if error.is_cancelled()));
        host.stage_hold.store(false, Ordering::SeqCst);
        let source = store.load(&prepared.run.key.run_id).await.unwrap().unwrap();
        let source = store
            .mutate(
                &source.key.run_id,
                &TaskRunMutation::RequestCancel {
                    expected_revision: source.revision,
                },
            )
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(1100)).await;
        let recovered = AutonomyRunner::reconcile_recovery(
            store.clone(),
            &source.key.run_id,
            decision(&source, TaskRecoveryDisposition::CancelPrepared),
            RecoveryProof::new(&source),
        )
        .await
        .unwrap();
        assert_eq!(recovered.status, TaskRunStatus::Paused);
        assert!(recovered.cancel_requested);
        let request = TaskRestoreRequest {
            expected_revision: recovered.revision,
            ..request(&prepared.run, config)
        };
        assert!(
            AutonomyRunner::restore_paused(
                TaskRestoreRequest {
                    config: request.config.clone(),
                    ceilings: request.ceilings.clone(),
                    config_identity: request.config_identity.clone(),
                    run_id: request.run_id.clone(),
                    expected_revision: request.expected_revision
                },
                store.clone(),
                host.clone()
            )
            .await
            .is_err()
        );
        let calls = host.calls.load(Ordering::SeqCst);
        let cancelled = AutonomyRunner::restore_cancelled(request, store.clone(), host.clone())
            .await
            .unwrap();
        assert_eq!(cancelled.run.status, TaskRunStatus::Cancelled);
        assert!(cancelled.run.pending.is_none());
        assert_eq!(host.calls.load(Ordering::SeqCst), calls);
        assert_eq!(host.effects.load(Ordering::SeqCst), 1);
        let saved = store.load(&source.key.run_id).await.unwrap().unwrap();
        let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload.clone()).unwrap();
        payload.runtime.validate().unwrap();
        let history =
            ai_agents_core::inspect_native_history(&payload.runtime.snapshot.memory.messages)
                .unwrap();
        assert!(
            history
                .exchanges()
                .iter()
                .all(|exchange| exchange.is_complete())
        );
        assert!(
            !history.exchanges().is_empty(),
            "completed signed history must be preserved, not erased"
        );
        assert!(
            store
                .mutate(
                    &source.key.run_id,
                    &TaskRunMutation::Claim {
                        expected_revision: saved.revision,
                        owner_token: "must-not-restart".into()
                    }
                )
                .await
                .is_err()
        );
    }
}

// Declared and nested child cursors use the same cleanup reconstruction; completed child effects and native prefixes are never replayed.
#[tokio::test]
async fn rc_cancelled_prepared_child_groups_keep_exact_ownership_and_native_cleanup() {
    for pattern in [
        "delegate: worker",
        "concurrent:\n        agents: [worker, peer]\n        aggregation:\n          strategy: all",
        "delegate: middle",
    ] {
        let mut host = host();
        Arc::get_mut(&mut host).unwrap().root = AgentSpec::from_yaml_strict(&format!("name: RestoreRoot\nsystem_prompt: root\nstates:\n  initial: work\n  states:\n    work:\n      {pattern}\n")).unwrap();
        let root = Box::pin(declared_source(&host, pattern)).await;
        let store = Arc::new(
            ScopedTaskRunStore::in_memory(root.info().id, None, "restore.v1".into()).unwrap(),
        );
        let runner = AutonomyRunner::try_new(
            root,
            config(),
            Default::default(),
            store.clone(),
            "restore.v1".into(),
        )
        .unwrap();
        let paused = runner.run("composed support", None).await.unwrap();
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
                    owner_token: "interrupted-child-constructor".into(),
                },
            )
            .await
            .unwrap();
        let source = store
            .mutate(
                &claimed.key.run_id,
                &TaskRunMutation::RequestCancel {
                    expected_revision: claimed.revision,
                },
            )
            .await
            .unwrap();
        let mut cleanup = decision(&source, TaskRecoveryDisposition::CancelPrepared);
        cleanup.interrupted_started_at = Some(claimed.updated_at);
        let recovered = AutonomyRunner::reconcile_recovery(
            store.clone(),
            &source.key.run_id,
            cleanup,
            RecoveryProof::new(&source),
        )
        .await
        .unwrap();
        let calls = host.calls.load(Ordering::SeqCst);
        let effects = host.effects.load(Ordering::SeqCst);
        let done = AutonomyRunner::restore_cancelled(
            TaskRestoreRequest {
                expected_revision: recovered.revision,
                ..request(&prepared.run, config())
            },
            store.clone(),
            host.clone(),
        )
        .await
        .unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Cancelled, "{pattern}");
        assert_eq!(host.calls.load(Ordering::SeqCst), calls);
        assert_eq!(host.effects.load(Ordering::SeqCst), effects);
        let saved = store.load(&source.key.run_id).await.unwrap().unwrap();
        let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload).unwrap();
        assert!(
            payload
                .children
                .iter()
                .all(|child| child.pending.is_none() && child.result.is_some())
        );
        for child in &payload.children {
            child.runtime.validate().unwrap();
            assert!(
                ai_agents_core::inspect_native_history(&child.runtime.snapshot.memory.messages)
                    .unwrap()
                    .exchanges()
                    .iter()
                    .all(|exchange| exchange.is_complete())
            );
        }
    }
}

// Cancellation racing recovery publication wins its revision and retains exact pending work until explicitly reviewed typed cleanup.
#[tokio::test]
async fn rc_cancel_racing_recovery_publication_preserves_pending_and_cleanup_authority() {
    let (host, store, prepared) = prepared_fixture(false).await;
    let claimed = store
        .mutate(
            &prepared.run.key.run_id,
            &TaskRunMutation::Claim {
                expected_revision: prepared.run.revision,
                owner_token: "claim-window".into(),
            },
        )
        .await
        .unwrap();
    let proof = RecoveryProof::new(&claimed);
    let fault = Arc::new(RecoveryFaultStore {
        inner: store.clone(),
        mode: "cancel",
        reached: tokio::sync::Semaphore::new(0),
        proof: proof.clone(),
    });
    assert!(
        AutonomyRunner::reconcile_recovery(
            fault,
            &claimed.key.run_id,
            decision(&claimed, TaskRecoveryDisposition::ResumePrepared),
            proof
        )
        .await
        .is_err()
    );
    let cancelled = store.load(&claimed.key.run_id).await.unwrap().unwrap();
    assert_eq!(cancelled.status, TaskRunStatus::RecoveryRequired);
    assert!(cancelled.cancel_requested);
    assert_eq!(cancelled.payload["pending"], claimed.payload["pending"]);
    let mut cleanup = decision(&cancelled, TaskRecoveryDisposition::CancelPrepared);
    cleanup.interrupted_started_at = Some(claimed.updated_at);
    let reviewed = AutonomyRunner::reconcile_recovery(
        store.clone(),
        &claimed.key.run_id,
        cleanup,
        RecoveryProof::new(&cancelled),
    )
    .await
    .unwrap();
    let calls = host.calls.load(Ordering::SeqCst);
    let completed = AutonomyRunner::restore_cancelled(
        TaskRestoreRequest {
            expected_revision: reviewed.revision,
            ..request(&prepared.run, config())
        },
        store.clone(),
        host.clone(),
    )
    .await
    .unwrap();
    assert_eq!(completed.run.status, TaskRunStatus::Cancelled);
    assert_eq!(host.calls.load(Ordering::SeqCst), calls);
    assert_eq!(host.effects.load(Ordering::SeqCst), 1);
}

// Candidate inspection recognizes valid older configuration data without weakening ordinary execution loads or actor scope.
#[tokio::test]
async fn rc_startup_skips_old_configuration_but_keeps_strict_ordinary_loads() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        let storage: Arc<dyn ai_agents_core::AgentStorage> = if sqlite {
            #[cfg(feature = "sqlite")]
            {
                Arc::new(
                    ai_agents_storage::SqliteStorage::new(":memory:")
                        .await
                        .unwrap(),
                )
            }
            #[cfg(not(feature = "sqlite"))]
            {
                unreachable!()
            }
        } else {
            Arc::new(ai_agents_storage::InMemoryTaskStorage::default())
        };
        let host = host();
        let old = Arc::new(
            ScopedTaskRunStore::new(
                storage.clone(),
                "RestoreRoot".into(),
                None,
                "older.v1".into(),
            )
            .unwrap(),
        );
        let current = Arc::new(
            ScopedTaskRunStore::new(storage, "RestoreRoot".into(), None, "restore.v1".into())
                .unwrap(),
        );
        let config = persistent_config(false);
        let mut runs = vec![];
        for (store, identity) in [(old.clone(), "older.v1"), (current.clone(), "restore.v1")] {
            let runner = AutonomyRunner::try_new(
                host.runtime(host.root.clone()),
                config.clone(),
                Default::default(),
                store,
                identity.into(),
            )
            .unwrap()
            .with_restore_host(host.clone());
            runs.push(runner.run("support", None).await.unwrap().run);
        }
        assert!(current.load(&runs[0].key.run_id).await.is_err());
        let (_, selected) = AutonomyRunner::restore_on_start(
            TaskStartupRequest {
                config,
                ceilings: Default::default(),
                config_identity: "restore.v1".into(),
                profile: None,
            },
            current.clone(),
            host,
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(selected.run.key, runs[1].key);
        assert_eq!(
            old.load(&runs[0].key.run_id)
                .await
                .unwrap()
                .unwrap()
                .revision,
            runs[0].revision
        );
        assert!(current.load(&runs[0].key.run_id).await.is_err());
    }
}

struct NoPostPauseReads {
    inner: Arc<ScopedTaskRunStore>,
    armed: AtomicBool,
}
#[async_trait]
impl TaskRunStore for NoPostPauseReads {
    // Only a new read after acknowledged pause is faulted; all mandatory journal writes are production operations.
    async fn create(&self, s: &TaskRunSnapshot) -> Result<()> {
        self.inner.create(s).await
    }
    async fn load(&self, id: &str) -> Result<Option<TaskRunSnapshot>> {
        if self.armed.load(Ordering::SeqCst) {
            return Err(AgentError::Persistence(
                "ordinary pause must not read again".into(),
            ));
        }
        self.inner.load(id).await
    }
    async fn list(&self) -> Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    async fn delete(&self, id: &str, rev: u64) -> Result<()> {
        self.inner.delete(id, rev).await
    }
    async fn mutate(&self, id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot> {
        let saved = self.inner.mutate(id, mutation).await?;
        if saved.status == TaskRunStatus::Paused {
            self.armed.store(true, Ordering::SeqCst);
        }
        Ok(saved)
    }
}

// Opt-in persistence must not add a failure-capable read to agents that omit or explicitly disable it.
#[tokio::test]
async fn rc_ordinary_pause_omitted_and_disabled_persistence_need_no_post_ack_read() {
    for disabled in [false, true] {
        let host = host();
        let root = host.runtime(host.root.clone());
        let inner = Arc::new(
            ScopedTaskRunStore::in_memory(root.info().id, None, "restore.v1".into()).unwrap(),
        );
        let store = Arc::new(NoPostPauseReads {
            inner: inner.clone(),
            armed: AtomicBool::new(false),
        });
        let mut config = config();
        if disabled {
            config.defaults.persistence = Some(AutonomyPersistenceConfig {
                enabled: Some(false),
                ..Default::default()
            });
        }
        let runner = AutonomyRunner::try_new(
            root,
            config,
            Default::default(),
            store.clone(),
            "restore.v1".into(),
        )
        .unwrap();
        let result = runner.run("ordinary support", None).await.unwrap();
        assert_eq!(result.run.status, TaskRunStatus::Paused);
        assert!(store.armed.load(Ordering::SeqCst));
        assert_eq!(
            inner
                .load(&result.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .revision,
            result.run.revision
        );
    }
}

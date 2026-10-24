//! Exact re-pause, mixed startup inventory and interrupted reconstruction clocks exercise the final recovery boundaries.
use super::*;
#[path = "rc_cleanup_tests.rs"]
mod cleanup;
#[cfg(feature = "sqlite")]
#[path = "rc_process_tests.rs"]
mod process;

// Progress invalidates the old image, not the right to prepare a newly acknowledged safe pause from actual live owners.
#[tokio::test]
async fn rc_repeated_persistent_questions_prepare_fresh_images_after_retained_and_restored_progress()
 {
    for reconstruct in [false, true] {
        let mut host = host();
        Arc::get_mut(&mut host).unwrap().root = AgentSpec::from_yaml_strict(
            "name: RestoreRoot\nsystem_prompt: restore-repeat-question\ntools: [ask_user]\n",
        )
        .unwrap();
        let root = host.runtime(host.root.clone());
        let store = Arc::new(
            ScopedTaskRunStore::in_memory(root.info().id, None, "restore.v1".into()).unwrap(),
        );
        let config = persistent_config(false);
        let runner = AutonomyRunner::try_new(
            root,
            config.clone(),
            Default::default(),
            store.clone(),
            "restore.v1".into(),
        )
        .unwrap()
        .with_restore_host(host.clone());
        let first = runner.run("support clarification", None).await.unwrap();
        let (runner, first) = if reconstruct {
            drop(runner);
            let (runner, restored) = AutonomyRunner::restore_paused(
                request(&first.run, config.clone()),
                store.clone(),
                host.clone(),
            )
            .await
            .unwrap();
            let reviewed = approve(&runner, &restored.run).await;
            (runner, reviewed)
        } else {
            (runner, first)
        };
        let second = answer(&runner, &first.run).await.unwrap();
        assert_eq!(second.run.status, TaskRunStatus::Paused);
        assert_ne!(
            second.run.pending.as_ref().unwrap().id,
            first.run.pending.as_ref().unwrap().id
        );
        drop(runner);
        let (runner, second) = AutonomyRunner::restore_paused(
            request(&second.run, config),
            store.clone(),
            host.clone(),
        )
        .await
        .unwrap();
        let reviewed = approve(&runner, &second.run).await;
        let completed = answer(&runner, &reviewed.run).await.unwrap();
        assert_eq!(completed.run.status, TaskRunStatus::Completed);
        assert_eq!(host.calls.load(Ordering::SeqCst), 3);
        assert_eq!(completed.run.counters.llm_attempts, 3);
    }
}

// A constructor journal can outwait its prepaid ceiling; recovery charges only the excess instead of declaring a free human pause.
#[tokio::test]
async fn rc_interrupted_constructor_charges_overrun_beyond_prepaid_work() {
    let (host, store, prepared) = prepared_fixture(false).await;
    host.stage_hold.store(true, Ordering::SeqCst);
    let task = tokio::spawn({
        let host = host.clone();
        let store = store.clone();
        let request = request(&prepared.run, config());
        async move { AutonomyRunner::restore_paused(request, store, host).await }
    });
    host.stage_reached.acquire().await.unwrap().forget();
    task.abort();
    assert!(matches!(task.await, Err(error) if error.is_cancelled()));
    let source = store.load(&prepared.run.key.run_id).await.unwrap().unwrap();
    let marker = source.payload["adapters"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["id"] == "runtime.restore_work")
        .unwrap();
    let start: chrono::DateTime<chrono::Utc> =
        serde_json::from_value(marker["state"]["started_at"].clone()).unwrap();
    let proof = RecoveryProof::new(&source);
    *proof.now.lock() = Some(start + chrono::Duration::milliseconds(8000));
    let decision = decision(&source, TaskRecoveryDisposition::ResumePrepared);
    let result =
        AutonomyRunner::reconcile_recovery(store.clone(), &source.key.run_id, decision, proof)
            .await
            .unwrap();
    assert_eq!(result.status, TaskRunStatus::Paused);
    let saved = store.load(&source.key.run_id).await.unwrap().unwrap();
    assert_eq!(saved.payload["clocks"]["interrupted_interval_millis"], 3000);
    assert_eq!(
        saved.payload["clocks"]["active_millis"].as_u64().unwrap(),
        source.payload["clocks"]["active_millis"].as_u64().unwrap() + 1000
    );
}

// A claim committed before the work journal still consumes time; later quarantine metadata requires explicit start evidence.
#[tokio::test]
async fn rc_claim_before_work_journal_never_gets_free_interrupted_time() {
    for quarantine in [false, true] {
        let (_, store, prepared) = prepared_fixture(false).await;
        let claimed = store
            .mutate(
                &prepared.run.key.run_id,
                &TaskRunMutation::Claim {
                    expected_revision: prepared.run.revision,
                    owner_token: "interrupted-before-precharge".into(),
                },
            )
            .await
            .unwrap();
        let source = if quarantine {
            store
                .mutate(
                    &claimed.key.run_id,
                    &TaskRunMutation::Recover {
                        expected_revision: claimed.revision,
                        owner_token: claimed.owner_token.clone().unwrap(),
                    },
                )
                .await
                .unwrap()
        } else {
            claimed.clone()
        };
        let proof = RecoveryProof::new(&source);
        *proof.now.lock() = Some(claimed.updated_at + chrono::Duration::seconds(8));
        let mut recovery = decision(&source, TaskRecoveryDisposition::ResumePrepared);
        if quarantine {
            assert!(
                AutonomyRunner::reconcile_recovery(
                    store.clone(),
                    &source.key.run_id,
                    recovery.clone(),
                    proof.clone()
                )
                .await
                .is_err()
            );
            assert_eq!(
                store
                    .load(&source.key.run_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .revision,
                source.revision
            );
            recovery.interrupted_started_at = Some(claimed.updated_at);
        }
        let recovered =
            AutonomyRunner::reconcile_recovery(store.clone(), &source.key.run_id, recovery, proof)
                .await
                .unwrap();
        assert_eq!(recovered.status, TaskRunStatus::Paused);
        assert_eq!(
            store
                .load(&source.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .payload["clocks"]["interrupted_interval_millis"],
            8000
        );
    }
}

// Expired pauses are non-candidates; a single valid pause is selected without staging any expired or incompatible image.
#[tokio::test]
async fn rc_startup_skips_expired_pauses_without_hiding_integrity_or_clock_errors() {
    let host = host();
    let store = Arc::new(
        ScopedTaskRunStore::in_memory("RestoreRoot".into(), None, "restore.v1".into()).unwrap(),
    );
    let mut expired = persistent_config(false);
    expired.defaults.max_wall_time_seconds = Some(1);
    let runner = AutonomyRunner::try_new(
        host.runtime(host.root.clone()),
        expired,
        Default::default(),
        store.clone(),
        "restore.v1".into(),
    )
    .unwrap()
    .with_restore_host(host.clone());
    let old = runner.run("old support case", None).await.unwrap();
    drop(runner);
    tokio::time::sleep(Duration::from_millis(1100)).await;
    let config = persistent_config(false);
    let runner = AutonomyRunner::try_new(
        host.runtime(host.root.clone()),
        config.clone(),
        Default::default(),
        store.clone(),
        "restore.v1".into(),
    )
    .unwrap()
    .with_restore_host(host.clone());
    let live = runner.run("current support case", None).await.unwrap();
    drop(runner);
    let (_, selected) = AutonomyRunner::restore_on_start(
        TaskStartupRequest {
            config,
            ceilings: Default::default(),
            config_identity: "restore.v1".into(),
            profile: None,
        },
        store.clone(),
        host,
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(selected.run.key, live.run.key);
    assert_eq!(
        store
            .load(&old.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .revision,
        old.run.revision
    );
}

//! Component await/drop boundaries preserve acknowledged pause custody and quarantine interrupted staged reconstruction.
use super::*;
use crate::autonomy::restoration_test_support::MemoryGate;

// Capture has not attempted a claim, so cancelling its read-only memory snapshot must preserve the original live pause.
#[tokio::test]
async fn restoration_capture_drop_preserves_original_resumable_pause() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        let gate = MemoryGate::new("RestoreRoot", "snapshot");
        let mut host = host();
        Arc::get_mut(&mut host).unwrap().memory_gate = Some(gate.clone());
        let root = host.runtime(host.root.clone());
        let store =
            crate::autonomy::continuation_test_support::store(root.info().id, "restore.v1", sqlite)
                .await;
        let runner = AutonomyRunner::try_new(
            root.clone(),
            config(),
            Default::default(),
            store.clone(),
            "restore.v1".into(),
        )
        .unwrap();
        let paused = runner.run("objective", None).await.unwrap();
        let owner = root.captured_task_owner().unwrap();
        gate.armed.store(true, Ordering::SeqCst);
        let mut capture = Box::pin(runner.checkpoint_for_restore(
            &paused.run.key.run_id,
            paused.run.revision,
            host.as_ref(),
        ));
        tokio::select! { result=&mut capture=>panic!("capture returned before selected snapshot: {result:?}"), permit=gate.reached.acquire()=>permit.unwrap().forget() }
        drop(capture);
        assert!(
            !owner.is_abandoned(),
            "read-only capture drop abandoned the acknowledged source pause"
        );
        let saved = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
        assert_eq!(saved.revision, paused.run.revision);
        assert_eq!(saved.status, TaskRunStatus::Paused);
        let cancelled = runner
            .cancel_paused(&paused.run.key.run_id, paused.run.revision)
            .await
            .unwrap();
        assert_eq!(cancelled.run.status, TaskRunStatus::Cancelled);
        assert!(root.set_context("released", json!(true)).is_ok());
    }
}

// Every already installed staged child remains reserved when a later component restore is cancelled; a request is not a cleanup ACK.
#[tokio::test]
async fn restoration_partial_component_drop_preserves_actual_parent_child_owners() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        let gate = MemoryGate::new("Worker", "restore");
        let mut host = host();
        let spec=AgentSpec::from_yaml_strict("name: RestoreRoot\nsystem_prompt: root\nstates:\n  initial: work\n  states:\n    work:\n      delegate: worker\n").unwrap();
        Arc::get_mut(&mut host).unwrap().root = spec;
        Arc::get_mut(&mut host).unwrap().memory_gate = Some(gate.clone());
        let root = declared_source(&host, "delegate: worker").await;
        let store =
            crate::autonomy::continuation_test_support::store(root.info().id, "restore.v1", sqlite)
                .await;
        let runner = AutonomyRunner::try_new(
            root,
            config(),
            Default::default(),
            store.clone(),
            "restore.v1".into(),
        )
        .unwrap();
        let paused = runner.run("objective", None).await.unwrap();
        let prepared = runner
            .checkpoint_for_restore(&paused.run.key.run_id, paused.run.revision, host.as_ref())
            .await
            .unwrap();
        drop(runner);
        gate.armed.store(true, Ordering::SeqCst);
        let request = TaskRestoreRequest {
            run_id: prepared.run.key.run_id.clone(),
            expected_revision: prepared.run.revision,
            config_identity: "restore.v1".into(),
            config: config(),
            ceilings: Default::default(),
        };
        let mut work = Box::pin(AutonomyRunner::restore_paused(
            request,
            store.clone(),
            host.clone(),
        ));
        tokio::select! {result=&mut work=>panic!("component not held: {:?}",result.err()),permit=gate.reached.acquire()=>permit.unwrap().forget()}
        let runtimes: Vec<_> = host
            .staged
            .lock()
            .values()
            .map(|runtime| runtime.upgrade().unwrap())
            .collect();
        let owners: Vec<_> = runtimes
            .iter()
            .map(|runtime| runtime.captured_task_owner().unwrap())
            .collect();
        assert!(owners.len() >= 2);
        drop(work);
        let saved = store.load(&prepared.run.key.run_id).await.unwrap().unwrap();
        assert_eq!(saved.status, TaskRunStatus::Running);
        assert!(saved.owner_token.is_some());
        for (runtime, owner) in runtimes.iter().zip(&owners) {
            assert!(Arc::ptr_eq(&runtime.captured_task_owner().unwrap(), owner));
            assert!(runtime.set_context("not_cleanup", json!(true)).is_err());
        }
        assert_eq!(host.calls.load(Ordering::SeqCst), 1);
    }
}

// A stale temporary owner cannot release a newer slot or remove its membership after acknowledged publication.
#[tokio::test]
async fn restoration_temporary_retirement_is_atomic_and_protects_newer_owner() {
    let host = host();
    let runtime = host.runtime(host.root.clone());
    let participants = Arc::new(crate::autonomy::Participants::default());
    let first = runtime
        .reserve_reconstructed_owner("first".into(), None)
        .await
        .unwrap();
    runtime
        .install_reconstructed_owner(first.clone(), &participants)
        .unwrap();
    runtime
        .release_reconstruction_temporary(&first, &participants)
        .unwrap();
    assert!(!participants.owns(&first));
    let second = runtime
        .reserve_reconstructed_owner("second".into(), None)
        .await
        .unwrap();
    runtime
        .install_reconstructed_owner(second.clone(), &participants)
        .unwrap();
    assert!(
        runtime
            .release_reconstruction_temporary(&first, &participants)
            .is_err()
    );
    assert!(Arc::ptr_eq(
        &runtime.captured_task_owner().unwrap(),
        &second
    ));
    assert!(participants.owns(&second));
    runtime
        .release_reconstruction_temporary(&second, &participants)
        .unwrap();
}

// Trusted host rollback is rejected before the review claim, preserving the exact usable reconstructed pause.
#[tokio::test]
async fn restoration_review_uses_certified_clock_before_claim() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        let (host, store, prepared) = prepared_fixture(sqlite).await;
        let request = TaskRestoreRequest {
            run_id: prepared.run.key.run_id.clone(),
            expected_revision: prepared.run.revision,
            config_identity: "restore.v1".into(),
            config: config(),
            ceilings: Default::default(),
        };
        let (runner, paused) = AutonomyRunner::restore_paused(request, store.clone(), host.clone())
            .await
            .unwrap();
        let review = runner
            .restoration_review(&paused.run.key.run_id)
            .await
            .unwrap()
            .unwrap();
        *host.clock.lock() = Some(chrono::Utc::now() - chrono::Duration::hours(1));
        assert!(
            runner
                .approve_restoration(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    &review.id,
                    ai_agents_hitl::ApprovalResult::Approved
                )
                .await
                .is_err()
        );
        let saved = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
        assert_eq!(saved.revision, paused.run.revision);
        assert_eq!(saved.status, TaskRunStatus::Paused);
        *host.clock.lock() = None;
        runner
            .cancel_paused(&paused.run.key.run_id, paused.run.revision)
            .await
            .unwrap();
    }
}

//! Review ACK, host-binding and cancellation races use the real scoped conditional backend in both feature modes.
#[path = "restoration_terminal_tests.rs"]
mod terminal_tests;
use super::*;

// A committed approval with a lost receipt is not a successful live review; no pending body executes in any failure mode.
#[tokio::test]
async fn restoration_review_reject_lost_and_drop_preserve_custody() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        for mode in ["reject", "lost", "drop"] {
            let (host, inner, fault, prepared) = fixture("review", mode, sqlite).await;
            let (runner, paused) =
                AutonomyRunner::restore_paused(request(&prepared), fault.clone(), host.clone())
                    .await
                    .unwrap();
            let review = runner
                .restoration_review(&paused.run.key.run_id)
                .await
                .unwrap()
                .unwrap();
            let mut work = Box::pin(runner.approve_restoration(
                &paused.run.key.run_id,
                paused.run.revision,
                &review.id,
                ai_agents_hitl::ApprovalResult::Approved,
            ));
            if mode == "drop" {
                tokio::select! {result=&mut work=>panic!("review returned before selected ACK: {result:?}"),permit=fault.reached.acquire()=>permit.unwrap().forget()}
                drop(work);
            } else {
                assert!(work.await.is_err());
            }
            assert!(!fault.armed.load(Ordering::SeqCst));
            let saved = inner.load(&paused.run.key.run_id).await.unwrap().unwrap();
            assert_eq!(
                saved.status,
                if mode == "reject" {
                    TaskRunStatus::Running
                } else {
                    TaskRunStatus::Paused
                }
            );
            assert_eq!(saved.owner_token.is_some(), mode == "reject");
            assert_eq!(host.calls.load(Ordering::SeqCst), 1);
            assert_eq!(host.effects.load(Ordering::SeqCst), 1);
            assert!(
                runner
                    .resume(
                        &paused.run.key.run_id,
                        saved.revision,
                        TaskResumeInput::UserAnswer {
                            request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                            answer: serde_json::json!({"answered":true,"selected":["yes"]})
                        }
                    )
                    .await
                    .is_err()
            );
            let root = host
                .staged
                .lock()
                .get("RestoreRoot")
                .unwrap()
                .upgrade()
                .unwrap();
            assert!(root.captured_task_owner().unwrap().is_abandoned());
            assert!(
                root.set_context("blocked", serde_json::json!(true))
                    .is_err()
            );
        }
    }
}

// Changing immutable installed host configuration while backend ACK is awaited must deny the actual conditional write.
#[tokio::test]
async fn restoration_binding_change_at_publication_rejects_inside_backend() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        let (host, inner, fault, prepared) = fixture("checkpoint", "before", sqlite).await;
        let mut work = Box::pin(AutonomyRunner::restore_paused(
            request(&prepared),
            fault.clone(),
            host.clone(),
        ));
        tokio::select! {result=&mut work=>panic!("publication not held: {:?}",result.err()),permit=fault.reached.acquire()=>permit.unwrap().forget()}
        host.binding_version.fetch_add(1, Ordering::SeqCst);
        fault.release.add_permits(1);
        assert!(work.await.is_err());
        let saved = inner.load(&prepared.run.key.run_id).await.unwrap().unwrap();
        assert_eq!(saved.status, TaskRunStatus::Running);
        assert!(saved.owner_token.is_some());
        assert_eq!(host.calls.load(Ordering::SeqCst), 1);
        assert_eq!(host.effects.load(Ordering::SeqCst), 1);
    }
}

// Neither fresh review nor its original result authorizes a later immutable host substitution.
#[tokio::test]
async fn restoration_binding_change_before_review_or_response_never_executes_pending() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        for after_review in [false, true] {
            let (host, inner, _, prepared) = fixture("none", "positive", sqlite).await;
            let (runner, paused) =
                AutonomyRunner::restore_paused(request(&prepared), inner.clone(), host.clone())
                    .await
                    .unwrap();
            let review = runner
                .restoration_review(&paused.run.key.run_id)
                .await
                .unwrap()
                .unwrap();
            let paused = if after_review {
                runner
                    .approve_restoration(
                        &paused.run.key.run_id,
                        paused.run.revision,
                        &review.id,
                        ai_agents_hitl::ApprovalResult::Approved,
                    )
                    .await
                    .unwrap()
            } else {
                paused
            };
            host.binding_version.fetch_add(1, Ordering::SeqCst);
            if after_review {
                assert!(
                    runner
                        .resume(
                            &paused.run.key.run_id,
                            paused.run.revision,
                            TaskResumeInput::UserAnswer {
                                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                                answer: serde_json::json!({"answered":true,"selected":["yes"]})
                            }
                        )
                        .await
                        .is_err()
                );
            } else {
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
                assert_eq!(
                    inner
                        .load(&paused.run.key.run_id)
                        .await
                        .unwrap()
                        .unwrap()
                        .revision,
                    paused.run.revision
                );
            }
            assert_eq!(host.calls.load(Ordering::SeqCst), 1);
            assert_eq!(host.effects.load(Ordering::SeqCst), 1);
        }
    }
}

// Cancellation racing staging keeps the claimed owner; deletion and competing restore cannot convert a request receipt into cleanup.
#[tokio::test]
async fn restoration_cancel_delete_races_do_not_publish_or_reclaim() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        let (host, inner, _, prepared) = fixture("none", "positive", sqlite).await;
        host.stage_hold.store(true, Ordering::SeqCst);
        let mut work = Box::pin(AutonomyRunner::restore_paused(
            request(&prepared),
            inner.clone(),
            host.clone(),
        ));
        tokio::select! {result=&mut work=>panic!("stage not held: {:?}",result.err()),permit=host.stage_reached.acquire()=>permit.unwrap().forget()}
        let running = inner.load(&prepared.run.key.run_id).await.unwrap().unwrap();
        let cancelled = inner
            .mutate(
                &prepared.run.key.run_id,
                &TaskRunMutation::RequestCancel {
                    expected_revision: running.revision,
                },
            )
            .await
            .unwrap();
        assert!(
            inner
                .delete(&prepared.run.key.run_id, cancelled.revision)
                .await
                .is_err()
        );
        host.stage_release.add_permits(1);
        assert!(work.await.is_err());
        let saved = inner.load(&prepared.run.key.run_id).await.unwrap().unwrap();
        assert!(saved.cancel_requested && saved.owner_token.is_some());
        assert_eq!(saved.status, TaskRunStatus::Running);
        assert!(
            AutonomyRunner::restore_paused(request(&prepared), inner, host.clone())
                .await
                .is_err()
        );
        assert_eq!(host.stages.load(Ordering::SeqCst), 1);
        assert_eq!(host.effects.load(Ordering::SeqCst), 1);

        let (host, inner, _, prepared) = fixture("none", "positive", sqlite).await;
        inner
            .delete(&prepared.run.key.run_id, prepared.run.revision)
            .await
            .unwrap();
        assert!(
            AutonomyRunner::restore_paused(request(&prepared), inner, host.clone())
                .await
                .is_err()
        );
        assert_eq!(host.stages.load(Ordering::SeqCst), 0);
    }
}

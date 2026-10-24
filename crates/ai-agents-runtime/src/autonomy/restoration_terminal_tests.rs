//! Reconstructed runs keep actual live ownership through terminal ACK uncertainty and explicit cleanup retry.
use super::*;
use crate::autonomy::continuation_test_support::ReceiptStore;

// A terminal row is immutable even when its ACK is lost; cleanup can retire only that old owner, never a newer run.
#[tokio::test]
async fn restoration_terminal_ack_fault_and_cleanup_retry_protect_newer_owner() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        for mode in ["reject", "lost", "drop"] {
            let (host, inner, _, prepared) = fixture("none", "positive", sqlite).await;
            let fault = ReceiptStore::new(inner.clone(), mode, TaskRunStatus::Completed);
            let (runner, paused) =
                AutonomyRunner::restore_paused(request(&prepared), fault.clone(), host.clone())
                    .await
                    .unwrap();
            let review = runner
                .restoration_review(&paused.run.key.run_id)
                .await
                .unwrap()
                .unwrap();
            let paused = runner
                .approve_restoration(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    &review.id,
                    ai_agents_hitl::ApprovalResult::Approved,
                )
                .await
                .unwrap();
            let root = host
                .staged
                .lock()
                .get("RestoreRoot")
                .unwrap()
                .upgrade()
                .unwrap();
            let old_owner = root.captured_task_owner().unwrap();
            fault.armed.store(true, Ordering::SeqCst);
            let mut work = Box::pin(runner.resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                    answer: serde_json::json!({"answered":true,"selected":["yes"]}),
                },
            ));
            if mode == "drop" {
                tokio::select! {result=&mut work=>panic!("terminal ACK not held: {result:?}"),permit=fault.reached.acquire()=>permit.unwrap().forget()}
                drop(work);
            } else {
                assert!(work.await.is_err());
            }
            fault.assert_hit();
            let saved = inner.load(&paused.run.key.run_id).await.unwrap().unwrap();
            assert_eq!(
                saved.status,
                if mode == "reject" {
                    TaskRunStatus::Running
                } else {
                    TaskRunStatus::Completed
                }
            );
            assert!(Arc::ptr_eq(
                &root.captured_task_owner().unwrap(),
                &old_owner
            ));
            assert!(
                root.set_context("not_cleaned", serde_json::json!(true))
                    .is_err()
            );
            assert_eq!(host.effects.load(Ordering::SeqCst), 1);
            assert_eq!(host.calls.load(Ordering::SeqCst), 2);
            if mode != "reject" {
                runner
                    .acknowledge_recovery_release(&paused.run.key.run_id, saved.revision)
                    .await
                    .unwrap();
                assert!(root.captured_task_owner().is_none());
                assert_eq!(
                    serde_json::to_value(
                        inner.load(&paused.run.key.run_id).await.unwrap().unwrap()
                    )
                    .unwrap(),
                    serde_json::to_value(&saved).unwrap()
                );
                let newer = root
                    .reserve_reconstructed_owner("newer-run".into(), None)
                    .await
                    .unwrap();
                assert!(
                    runner
                        .acknowledge_recovery_release(&paused.run.key.run_id, saved.revision)
                        .await
                        .is_err()
                );
                assert!(Arc::ptr_eq(&root.captured_task_owner().unwrap(), &newer));
                root.release_autonomy_run(&newer).await.unwrap();
            }
        }
    }
}

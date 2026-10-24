//! Streamed reconstruction review and resumed native work retain exact custody across terminal receipt failures.
use super::*;
use crate::autonomy::continuation_test_support::ReceiptStore;
use futures::StreamExt;

// Restoration review does not answer the original question; a fresh stream resumes only that pending native operation.
#[tokio::test]
async fn ev_restored_review_resume_terminal_receipt_matrix() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        for mode in ["positive", "reject", "lost", "drop"] {
            let (host, inner, _, prepared) = fixture("none", "positive", sqlite).await;
            let fault = ReceiptStore::new(inner.clone(), mode, TaskRunStatus::Completed);
            let (runner, paused) =
                AutonomyRunner::restore_paused(request(&prepared), fault.clone(), host.clone())
                    .await
                    .unwrap();
            let original_effects = host.effects.load(Ordering::SeqCst);
            let review = runner
                .restoration_review(&paused.run.key.run_id)
                .await
                .unwrap()
                .unwrap();
            let mut stream = runner
                .stream_operation(
                    TaskStreamOperation::RestorationReview {
                        run_id: paused.run.key.run_id.clone(),
                        revision: paused.run.revision,
                        review_id: review.id,
                        decision: ai_agents_hitl::ApprovalResult::Approved,
                    },
                    TaskStreamConfig { capacity: 1 },
                )
                .unwrap();
            let mut prior = vec![];
            while let Some(event) = stream.next().await {
                prior.push(event);
            }
            let reviewed = stream.take_result().unwrap().unwrap();
            assert_eq!(reviewed.run.status, TaskRunStatus::Paused);
            assert_eq!(host.effects.load(Ordering::SeqCst), original_effects);
            fault.armed.store(true, Ordering::SeqCst);
            let mut stream = runner
                .resume_stream(
                    &reviewed.run.key.run_id,
                    reviewed.run.revision,
                    TaskResumeInput::UserAnswer {
                        request_id: reviewed.run.pending.unwrap().id,
                        answer: serde_json::json!({"answered":true,"selected":["yes"]}),
                    },
                )
                .unwrap();
            let mut events = vec![];
            if mode == "drop" {
                loop {
                    tokio::select! {
                        event = stream.next() => events.push(event.expect("terminal ACK must remain held")),
                        permit = fault.reached.acquire() => { permit.unwrap().forget(); break; }
                    }
                }
                drop(stream);
            } else {
                while let Some(event) = stream.next().await {
                    events.push(event);
                }
                let result = stream.take_result().unwrap();
                if mode == "positive" {
                    assert_eq!(result.unwrap().run.status, TaskRunStatus::Completed);
                } else {
                    assert!(result.is_err());
                }
            }
            assert_eq!(host.effects.load(Ordering::SeqCst), 1);
            assert_eq!(host.calls.load(Ordering::SeqCst), 2);
            if mode != "positive" {
                assert!(
                    !events
                        .iter()
                        .any(|e| e.kind == TaskRunEventKind::TaskCompleted)
                );
                let saved = inner.load(&reviewed.run.key.run_id).await.unwrap().unwrap();
                assert_eq!(
                    saved.status,
                    if mode == "reject" {
                        TaskRunStatus::Running
                    } else {
                        TaskRunStatus::Completed
                    }
                );
            }
            if let Some(first) = events.first() {
                assert!(prior.last().unwrap().cursor < first.cursor);
            }
            assert!(
                !serde_json::to_string(&events)
                    .unwrap()
                    .contains("opaque-restoration-fixture")
            );
        }
    }
}

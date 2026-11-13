//! Task-origin message observations reconstruct parent/child authority and re-pause through the existing production fixture.
use super::*;
use crate::autonomy::restoration_test_support::{FreshRuntime, RebuildHost, restore};

// The host supplies fresh actual nodes and child declarations; membership and execution remain framework-owned.
fn fresh(f: &Fixture) -> FreshRuntime {
    FreshRuntime::new(f.root.clone(), vec![f.main.clone()]).with_child(
        f.worker.clone(),
        vec![f.child.clone()],
        f.worker_spec.clone(),
    )
}

// A fixed host permit or old timeout receipt never becomes blanket approval for the next reconstructed child request.
#[tokio::test]
async fn restoration_task_message_child_repause_external_and_timeout_policy() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        for (hosted, policies) in [
            (false, None),
            (true, None),
            (false, Some(("reject", "approve"))),
        ] {
            let source = crate::autonomy::host_evidence_tests::Revision::new();
            let mut f = Box::pin(fixture_leaf_policy(
                true,
                hosted,
                Some(source.clone()),
                policies,
            ))
            .await;
            f.store = crate::autonomy::continuation_test_support::store(
                f.root.info().id,
                "observation-message.v1",
                sqlite,
            )
            .await;
            f.runner = AutonomyRunner::try_new(
                f.root.clone(),
                f.config.clone(),
                Default::default(),
                f.store.clone(),
                "observation-message.v1".into(),
            )
            .unwrap();
            let witnesses = Arc::new(parking_lot::Mutex::new(vec![]));
            let stage_witnesses = witnesses.clone();
            let stage_source = source.clone();
            let host = RebuildHost::new(
                "message-fixture.v1",
                &fresh(&f),
                Box::new(move || {
                    let source = stage_source.clone();
                    let witnesses = stage_witnesses.clone();
                    Box::pin(async move {
                        let f = Box::pin(fixture_leaf_policy_membership(
                            true,
                            hosted,
                            Some(source),
                            policies,
                            false,
                        ))
                        .await;
                        witnesses.lock().push((
                            f.main.clone(),
                            f.child.clone(),
                            f.leaf_calls.clone(),
                            f.calls.clone(),
                        ));
                        fresh(&f)
                    })
                }),
            );
            let mut initial = f.runner.run("objective", None).await.unwrap();
            assert_eq!(initial.run.status, TaskRunStatus::Paused);
            let source_leaf = f.leaf_calls.clone();
            if policies.is_some() {
                tokio::time::sleep(std::time::Duration::from_millis(2050)).await;
                initial = f
                    .runner
                    .resume(
                        &initial.run.key.run_id,
                        initial.run.revision,
                        TaskResumeInput::Approval {
                            request_id: initial.run.pending.as_ref().unwrap().id.clone(),
                            result: ai_agents_hitl::ApprovalResult::Timeout,
                        },
                    )
                    .await
                    .unwrap();
                assert_eq!(initial.run.status, TaskRunStatus::Paused);
                assert_eq!(source_leaf.load(Ordering::SeqCst), 1);
            }
            let old_root = Arc::downgrade(&f.root);
            let old_child = Arc::downgrade(&f.worker);
            let source_main = f.main.clone();
            let source_child = f.child.clone();
            let Fixture {
                root,
                worker,
                runner,
                store,
                config,
                ..
            } = f;
            drop(root);
            drop(worker);
            let (restored, mut result) = restore(
                runner,
                store.clone(),
                &initial.run,
                config,
                "observation-message.v1",
                host,
            )
            .await;
            assert!(old_root.upgrade().is_none() && old_child.upgrade().is_none());
            if policies.is_some() {
                // The second leaf keeps its original short deadline; explicit rejection proves the earlier timeout approval cannot transfer, independent of restore latency.
                assert_eq!(source_leaf.load(Ordering::SeqCst), 1);
                let rejected = restored
                    .resume(
                        &result.run.key.run_id,
                        result.run.revision,
                        TaskResumeInput::Approval {
                            request_id: result.run.pending.as_ref().unwrap().id.clone(),
                            result: ai_agents_hitl::ApprovalResult::Rejected {
                                reason: Some("do not approve another leaf".into()),
                            },
                        },
                    )
                    .await;
                assert!(
                    !rejected
                        .as_ref()
                        .is_ok_and(|result| result.run.status == TaskRunStatus::Completed)
                );
                let values = witnesses.lock();
                assert_eq!(values[0].2.load(Ordering::SeqCst), 0);
                assert_eq!(values[0].1.calls.load(Ordering::SeqCst), 0);
                assert_eq!(source_main.calls.load(Ordering::SeqCst), 1);
                continue;
            }
            // Fixed host validation and ordinary message HITL are separate parent gates, followed by two child waits and one external signal.
            let expected_waits = if hosted {
                5
            } else if policies.is_some() {
                2
            } else {
                3
            };
            let mut waits = 0;
            while result.run.status == TaskRunStatus::Paused {
                waits += 1;
                let (fresh_calls, callbacks) = {
                    let values = witnesses.lock();
                    (
                        values[0].1.calls.load(Ordering::SeqCst),
                        values[0].3.load(Ordering::SeqCst),
                    )
                };
                assert!(
                    waits <= expected_waits,
                    "unexpected re-pause: hosted={hosted} policy={policies:?} pending={:?} reason={:?} source_child_calls={} fresh_child_calls={} callbacks={}",
                    result.run.pending.as_ref().map(|pending| pending.kind),
                    result.run.stop_reason,
                    source_child.calls.load(Ordering::SeqCst),
                    fresh_calls,
                    callbacks
                );
                let pending = result.run.pending.as_ref().unwrap();
                let request_id = pending.id.clone();
                let input = match pending.kind {
                    TaskPendingKind::ExternalCondition => {
                        TaskResumeInput::ExternalSignal(ExternalSignal {
                            request_id,
                            event_id: "ready".into(),
                            target_revision: Some("r1".into()),
                            payload: json!(true),
                        })
                    }
                    _ if matches!(pending.kind, TaskPendingKind::Approval)
                        || policies.is_some() =>
                    {
                        TaskResumeInput::Approval {
                            request_id,
                            result: ai_agents_hitl::ApprovalResult::Approved,
                        }
                    }
                    _ => TaskResumeInput::UserAnswer {
                        request_id,
                        answer: json!({"answered":true,"selected":["yes"]}),
                    },
                };
                result = restored
                    .resume(&result.run.key.run_id, result.run.revision, input)
                    .await
                    .unwrap();
            }
            assert_eq!(waits, expected_waits);
            assert_eq!(
                result.run.status,
                TaskRunStatus::Completed,
                "{:?}",
                result.run.stop_reason
            );
            assert_eq!(source_main.calls.load(Ordering::SeqCst), 1);
            {
                let witnesses = witnesses.lock();
                let (main, child, leaf, callbacks) = &witnesses[0];
                assert_eq!(main.calls.load(Ordering::SeqCst), 0);
                assert_eq!(
                    source_child.calls.load(Ordering::SeqCst) + child.calls.load(Ordering::SeqCst),
                    3
                );
                assert_eq!(
                    source_leaf.load(Ordering::SeqCst) + leaf.load(Ordering::SeqCst),
                    if policies.is_some() { 2 } else { 0 }
                );
                assert!(callbacks.load(Ordering::SeqCst) > 0);
            }
            let saved: TaskCheckpointPayload = serde_json::from_value(
                store
                    .load(&result.run.key.run_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .payload,
            )
            .unwrap();
            assert!(
                saved
                    .children
                    .iter()
                    .all(|child| child.pending.is_none() && child.result.is_some())
            );
        }
    }
}

//! Reconstruction preserves real observation drivers, external waits and fixed host permits without another model turn.
#[path = "restoration_control_tests.rs"]
mod control_tests;
use super::*;
use crate::autonomy::restoration_test_support::{FreshRuntime, RebuildHost, restore};

// Required validation is still required after restoration; neither a saved final text nor the separate restore approval satisfies it.
#[tokio::test]
async fn restoration_controller_observations_external_repause_and_tool_free_wait() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        for mode in [
            "host",
            "batch",
            "question",
            "external_repeat",
            "external_tool_free",
            "fallback_question",
        ] {
            let mut f = fixture(mode);
            f.store = crate::autonomy::continuation_test_support::store(
                f.agent.info().id,
                "controller.v1",
                sqlite,
            )
            .await;
            f.runner = AutonomyRunner::try_new(
                f.agent.clone(),
                f.config.clone(),
                Default::default(),
                f.store.clone(),
                "controller.v1".into(),
            )
            .unwrap();
            let witnesses = Arc::new(parking_lot::Mutex::new(vec![]));
            let stage_witnesses = witnesses.clone();
            let host = RebuildHost::new(
                "controller-fixture.v1",
                &FreshRuntime::new(f.agent.clone(), vec![f.provider.clone()]),
                Box::new(move || {
                    let witnesses = stage_witnesses.clone();
                    Box::pin(async move {
                        let fresh = fixture(mode);
                        witnesses.lock().push((
                            fresh.calls.clone(),
                            fresh.callbacks.clone(),
                            fresh.provider.clone(),
                        ));
                        FreshRuntime::new(fresh.agent.clone(), vec![fresh.provider.clone()])
                    })
                }),
            );
            let initial = f.runner.run("objective", None).await.unwrap();
            assert_eq!(
                initial.run.status,
                TaskRunStatus::Paused,
                "{mode}: {:?}",
                initial.run.stop_reason
            );
            let old_request = initial.run.pending.as_ref().unwrap().id.clone();
            let old = Arc::downgrade(&f.agent);
            let provider = f.provider.clone();
            let source_calls = f.calls.clone();
            let Fixture {
                agent,
                runner,
                store,
                config,
                ..
            } = f;
            drop(agent);
            let (restored, mut current) = restore(
                runner,
                store.clone(),
                &initial.run,
                config,
                "controller.v1",
                host.clone(),
            )
            .await;
            assert!(old.upgrade().is_none());
            assert_eq!(current.run.status, TaskRunStatus::Paused);
            assert_ne!(current.run.pending.as_ref().unwrap().id, old_request);
            assert!(
                restored
                    .resume(
                        &current.run.key.run_id,
                        current.run.revision,
                        TaskResumeInput::Approval {
                            request_id: old_request,
                            result: ai_agents_hitl::ApprovalResult::Approved
                        }
                    )
                    .await
                    .is_err()
            );
            assert_eq!(
                witnesses.lock()[0].0.load(Ordering::SeqCst),
                0,
                "restoration review executed observation body"
            );
            let mut responses = 0;
            while current.run.status == TaskRunStatus::Paused {
                responses += 1;
                assert!(responses <= 3, "unexpected observation loop: {mode}");
                let request_id = current.run.pending.as_ref().unwrap().id.clone();
                let input = if mode.contains("external") {
                    TaskResumeInput::ExternalSignal(ExternalSignal {
                        request_id,
                        event_id: format!("event-{responses}"),
                        target_revision: None,
                        payload: json!(mode != "external_repeat" || responses > 1),
                    })
                } else if mode.contains("question") {
                    TaskResumeInput::UserAnswer {
                        request_id,
                        answer: json!({"answered":true,"selected":["yes"]}),
                    }
                } else {
                    TaskResumeInput::Approval {
                        request_id,
                        result: ai_agents_hitl::ApprovalResult::Approved,
                    }
                };
                current = restored
                    .resume(&current.run.key.run_id, current.run.revision, input)
                    .await
                    .unwrap();
            }
            assert_eq!(
                current.run.status,
                TaskRunStatus::Completed,
                "{mode}: {:?}",
                current.run.stop_reason
            );
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            let witnesses = witnesses.lock();
            let (calls, callbacks, fresh_provider) = &witnesses[0];
            assert_eq!(
                fresh_provider.calls.load(Ordering::SeqCst),
                0,
                "validation resume replayed a model turn"
            );
            assert!(callbacks.load(Ordering::SeqCst) > 0);
            assert_eq!(
                source_calls.load(Ordering::SeqCst) + calls.load(Ordering::SeqCst),
                match mode {
                    "host" => 1,
                    "batch" => 2,
                    "fallback_question" => 1,
                    _ => 0,
                }
            );
            assert_eq!(current.run.counters.turns, 1);
            assert_eq!(current.run.counters.llm_attempts, 1);
        }
    }
}

// A current policy/scope change after consumption ACK and modified fixed arguments both fail before the real observation body.
#[tokio::test]
async fn restoration_current_admission_after_review_and_consumption_ack() {
    use crate::autonomy::continuation_test_support::ReceiptStore;
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        for mutation in ["scope", "policy", "arguments"] {
            let mut f = fixture("host");
            f.store = crate::autonomy::continuation_test_support::store(
                f.agent.info().id,
                "controller.v1",
                sqlite,
            )
            .await;
            f.runner = AutonomyRunner::try_new(
                f.agent.clone(),
                f.config.clone(),
                Default::default(),
                f.store.clone(),
                "controller.v1".into(),
            )
            .unwrap();
            let witness = Arc::new(parking_lot::Mutex::new(None));
            let stage_witness = witness.clone();
            let host = RebuildHost::new(
                "fixed-admission.v1",
                &FreshRuntime::new(f.agent.clone(), vec![f.provider.clone()]),
                Box::new(move || {
                    let witness = stage_witness.clone();
                    Box::pin(async move {
                        let fresh = fixture("host");
                        *witness.lock() = Some(fresh.calls.clone());
                        FreshRuntime::new(fresh.agent.clone(), vec![fresh.provider.clone()])
                    })
                }),
            );
            let initial = f.runner.run("objective", None).await.unwrap();
            let fault =
                ReceiptStore::checkpoint(f.store.clone(), "hold", json!({"consume":"not-armed"}));
            let Fixture {
                agent,
                runner,
                config,
                ..
            } = f;
            drop(agent);
            let (restored, paused) = restore(
                runner,
                fault.clone(),
                &initial.run,
                config,
                "controller.v1",
                host.clone(),
            )
            .await;
            let request_id = paused.run.pending.as_ref().unwrap().id.clone();
            let decision = if mutation == "arguments" {
                ai_agents_hitl::ApprovalResult::Modified {
                    changes: std::collections::HashMap::from([("value".into(), json!(9))]),
                }
            } else {
                ai_agents_hitl::ApprovalResult::Approved
            };
            if mutation != "arguments" {
                *fault.checkpoint.lock() = Some(json!({"consume":request_id}));
                fault.armed.store(true, Ordering::SeqCst);
            }
            let mut work = Box::pin(restored.resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::Approval {
                    request_id,
                    result: decision,
                },
            ));
            if mutation != "arguments" {
                tokio::select! {result=&mut work=>panic!("consumption ACK not held: {result:?}"),permit=fault.reached.acquire()=>permit.unwrap().forget()}
                let root = host.staged.lock().upgrade().unwrap();
                if mutation == "scope" {
                    root.runtime_control().set_tool_scope(vec![]);
                } else {
                    root.runtime_control().set_tool_security(Default::default());
                }
                fault.release.add_permits(1);
            }
            let result = work.await;
            assert!(
                !result
                    .as_ref()
                    .is_ok_and(|result| result.run.status == TaskRunStatus::Completed)
            );
            assert_eq!(
                witness.lock().as_ref().unwrap().load(Ordering::SeqCst),
                0,
                "{mutation} bypassed current admission"
            );
        }
    }
}

// Reconstruction installs the original invocation bound; waiting for an external response cannot create a new check lifetime.
#[tokio::test]
async fn restoration_validation_original_deadline_does_not_renew() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        let mut f = fixture("external_tool_free");
        f.config
            .defaults
            .validation
            .as_mut()
            .unwrap()
            .checks
            .as_mut()
            .unwrap()[0]
            .timeout_seconds = Some(1);
        f.store = crate::autonomy::continuation_test_support::store(
            f.agent.info().id,
            "controller.v1",
            sqlite,
        )
        .await;
        f.runner = AutonomyRunner::try_new(
            f.agent.clone(),
            f.config.clone(),
            Default::default(),
            f.store.clone(),
            "controller.v1".into(),
        )
        .unwrap();
        let count = Arc::new(parking_lot::Mutex::new(None));
        let stage_count = count.clone();
        let host = RebuildHost::new(
            "validation-bound.v1",
            &FreshRuntime::new(f.agent.clone(), vec![f.provider.clone()]),
            Box::new(move || {
                let count = stage_count.clone();
                Box::pin(async move {
                    let fresh = fixture("external_tool_free");
                    *count.lock() = Some(fresh.callbacks.clone());
                    FreshRuntime::new(fresh.agent.clone(), vec![fresh.provider.clone()])
                })
            }),
        );
        let initial = f.runner.run("objective", None).await.unwrap();
        let Fixture {
            agent,
            runner,
            store,
            config,
            ..
        } = f;
        drop(agent);
        let (restored, paused) =
            restore(runner, store, &initial.run, config, "controller.v1", host).await;
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        let result = restored
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::ExternalSignal(ExternalSignal {
                    request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                    event_id: "late".into(),
                    target_revision: None,
                    payload: json!(true),
                }),
            )
            .await;
        assert!(
            !result
                .as_ref()
                .is_ok_and(|result| result.run.status == TaskRunStatus::Completed)
        );
        assert_eq!(count.lock().as_ref().unwrap().load(Ordering::SeqCst), 0);
    }
}

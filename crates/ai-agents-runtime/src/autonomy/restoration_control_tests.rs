//! Controller-owned capacity, stage and replanning reviews reconstruct through the same tool-free runtime.
use super::*;

// Tool-free production fixtures select real capacity, stage and replan boundaries without a parallel controller.
fn control_fixture(kind: &str) -> Fixture {
    let mut f = fixture("external_tool_free");
    f.config.defaults.validation = None;
    f.config.defaults.completion = Some(CompletionGate::ResponseContains("done".into()));
    f.config.defaults.max_llm_calls = Some(if kind == "capacity" { 1 } else { 5 });
    if kind == "capacity" {
        f.config.defaults.on_limit = Some(serde_yaml::from_str("action: pause_run").unwrap());
        f.provider
            .install_restoration_state(json!(["working", "done"]))
            .unwrap();
    } else if kind == "stage" {
        f.config.defaults.lifecycle =
            Some(serde_json::from_value(json!([{"id":"draft"},{"id":"review"}])).unwrap());
        f.config.defaults.supervision = Some(SupervisionConfig {
            stage_changes: Some(ReviewPolicy::RequireApproval),
            plan_changes: None,
        });
        f.provider
            .install_restoration_state(json!(["done", "done"]))
            .unwrap();
    } else {
        f.config.defaults.progress=Some(serde_yaml::from_str("stagnation: {max_cycles_without_progress: 1, action: replan, max_replans: 1, on_exhausted: stop}").unwrap());
        f.config.defaults.supervision = Some(SupervisionConfig {
            stage_changes: None,
            plan_changes: Some(ReviewPolicy::RequireApproval),
        });
        let plan = r#"{"steps":[{"id":"one","description":"Verify sources","action_type":"think","action_target":"reason","dependencies":[]}]}"#;
        f.provider
            .install_restoration_state(json!(["working", plan, "done"]))
            .unwrap();
    }
    f
}

// Restoring a controller boundary neither resets exhaustion nor skips required stages nor regenerates an already reviewed plan.
#[tokio::test]
async fn restoration_controller_capacity_stage_and_replan_reviews_keep_contract() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        for kind in ["capacity", "stage", "replan"] {
            let mut f = control_fixture(kind);
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
            let fresh_provider = Arc::new(parking_lot::Mutex::new(None));
            let stage_provider = fresh_provider.clone();
            let host = RebuildHost::new(
                "control-fixture.v1",
                &FreshRuntime::new(f.agent.clone(), vec![f.provider.clone()]),
                Box::new(move || {
                    let provider = stage_provider.clone();
                    Box::pin(async move {
                        let fresh = control_fixture(kind);
                        *provider.lock() = Some(fresh.provider.clone());
                        FreshRuntime::new(fresh.agent.clone(), vec![fresh.provider.clone()])
                    })
                }),
            );
            let initial = f.runner.run("objective", None).await.unwrap();
            assert_eq!(
                initial.run.status,
                TaskRunStatus::Paused,
                "{kind}: {:?}",
                initial.run.stop_reason
            );
            let source_provider = f.provider.clone();
            let source_calls = source_provider.calls.load(Ordering::SeqCst);
            let old = Arc::downgrade(&f.agent);
            let Fixture {
                agent,
                runner,
                store,
                config,
                ..
            } = f;
            drop(agent);
            let (restored, mut paused) = restore(
                runner,
                store.clone(),
                &initial.run,
                config,
                "controller.v1",
                host,
            )
            .await;
            assert!(old.upgrade().is_none());
            assert_eq!(
                fresh_provider
                    .lock()
                    .as_ref()
                    .unwrap()
                    .calls
                    .load(Ordering::SeqCst),
                0
            );
            let pending_id = paused.run.pending.as_ref().unwrap().id.clone();
            let input = if kind == "capacity" {
                assert!(
                    restored
                        .resume(
                            &paused.run.key.run_id,
                            paused.run.revision,
                            TaskResumeInput::Continue
                        )
                        .await
                        .is_err()
                );
                let payload: TaskCheckpointPayload = serde_json::from_value(
                    store
                        .load(&paused.run.key.run_id)
                        .await
                        .unwrap()
                        .unwrap()
                        .payload,
                )
                .unwrap();
                let mut limits = payload.limits;
                limits.max_llm_calls = 2;
                paused = restored
                    .revise_limits_paused(
                        &paused.run.key.run_id,
                        paused.run.revision,
                        limits,
                        "explicitly admit the pending continuation",
                    )
                    .await
                    .unwrap();
                assert_eq!(paused.run.pending.as_ref().unwrap().id, pending_id);
                TaskResumeInput::Continue
            } else {
                TaskResumeInput::Approval {
                    request_id: pending_id,
                    result: ai_agents_hitl::ApprovalResult::Approved,
                }
            };
            let mut done = restored
                .resume(&paused.run.key.run_id, paused.run.revision, input)
                .await
                .unwrap();
            if kind == "stage" {
                // Finishing the final required stage has its own review; acknowledging it must not run another model turn.
                assert_eq!(done.run.status, TaskRunStatus::Paused);
                assert_ne!(
                    done.run.pending.as_ref().unwrap().id,
                    paused.run.pending.as_ref().unwrap().id
                );
                done = restored
                    .resume(
                        &done.run.key.run_id,
                        done.run.revision,
                        TaskResumeInput::Approval {
                            request_id: done.run.pending.as_ref().unwrap().id.clone(),
                            result: ai_agents_hitl::ApprovalResult::Approved,
                        },
                    )
                    .await
                    .unwrap();
            }
            assert_eq!(
                done.run.status,
                TaskRunStatus::Completed,
                "{kind}: {:?}",
                done.run.stop_reason
            );
            assert_eq!(source_provider.calls.load(Ordering::SeqCst), source_calls);
            assert_eq!(
                fresh_provider
                    .lock()
                    .as_ref()
                    .unwrap()
                    .calls
                    .load(Ordering::SeqCst),
                1
            );
            assert_eq!(done.run.counters.llm_attempts, source_calls as u64 + 1);
            let payload: TaskCheckpointPayload = serde_json::from_value(
                store
                    .load(&done.run.key.run_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .payload,
            )
            .unwrap();
            if kind == "replan" {
                assert_eq!(payload.progress.replans, 1);
            }
        }
    }
}

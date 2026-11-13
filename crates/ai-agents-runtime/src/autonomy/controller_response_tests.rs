use super::*;

#[tokio::test]
async fn controller_oversized_approval_rejects_before_claim() {
    for modified in [false, true] {
        let f = fixture("approval");
        let paused = f.runner.run("objective", None).await.unwrap();
        let result = if modified {
            ai_agents_hitl::ApprovalResult::Modified {
                changes: std::collections::HashMap::from([(
                    "value".into(),
                    json!("x".repeat(300_000)),
                )]),
            }
        } else {
            ai_agents_hitl::ApprovalResult::Rejected {
                reason: Some("x".repeat(300_000)),
            }
        };
        assert!(
            f.runner
                .resume(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    TaskResumeInput::Approval {
                        request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                        result,
                    }
                )
                .await
                .is_err()
        );
        assert_eq!(
            f.store
                .load(&paused.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .revision,
            paused.run.revision
        );
        assert_eq!(f.calls.load(Ordering::SeqCst), 0);
        let done = f
            .runner
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::Approval {
                    request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                    result: ai_agents_hitl::ApprovalResult::Approved,
                },
            )
            .await
            .unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Completed);
    }
}

#[tokio::test]
async fn controller_acknowledged_claim_racing_expiry_or_cancel_settles_safe_work() {
    for cancel in [false, true] {
        for mode in ["approval", "external"] {
            let mut f = fixture(mode);
            f.config.defaults.validation = Some(serde_yaml::from_str("checks: [{id: check, adapter: host.observation, schedule: each_cycle, timeout_seconds: 1, config: {}}]").unwrap());
            let gated = AckStore::new(f.store.clone(), false, false);
            f.runner = AutonomyRunner::try_new(
                f.agent.clone(),
                f.config,
                Default::default(),
                gated.clone(),
                "controller.v1".into(),
            )
            .unwrap();
            let runner = Arc::new(f.runner);
            let paused = runner.run("objective", None).await.unwrap();
            gated.delay_claim.store(true, Ordering::SeqCst);
            let request_id = paused.run.pending.as_ref().unwrap().id.clone();
            let input = if mode == "approval" {
                TaskResumeInput::Approval {
                    request_id,
                    result: ai_agents_hitl::ApprovalResult::Approved,
                }
            } else {
                TaskResumeInput::ExternalSignal(ExternalSignal {
                    request_id,
                    event_id: "event".into(),
                    target_revision: None,
                    payload: json!(true),
                })
            };
            let worker = runner.clone();
            let run_id = paused.run.key.run_id.clone();
            let task =
                tokio::spawn(
                    async move { worker.resume(&run_id, paused.run.revision, input).await },
                );
            tokio::time::timeout(std::time::Duration::from_secs(5), gated.entered.acquire())
                .await
                .unwrap()
                .unwrap()
                .forget();
            if cancel {
                let current = f.store.load(&paused.run.key.run_id).await.unwrap().unwrap();
                f.store
                    .mutate(
                        &paused.run.key.run_id,
                        &TaskRunMutation::RequestCancel {
                            expected_revision: current.revision,
                        },
                    )
                    .await
                    .unwrap();
            } else {
                tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
            }
            gated.release.add_permits(1);
            let done = task.await.unwrap().unwrap();
            assert_eq!(
                done.run.status,
                if cancel {
                    TaskRunStatus::Cancelled
                } else {
                    TaskRunStatus::LimitReached
                }
            );
            assert_eq!(f.calls.load(Ordering::SeqCst), 0);
            assert_eq!(f.callbacks.load(Ordering::SeqCst), 1);
            assert_eq!(f.agent.chat("follow-up").await.unwrap().content, "ordinary");
        }
    }
}

#[tokio::test]
async fn controller_fixed_host_approval_modification_cannot_change_bound_arguments() {
    let f = fixture("host");
    let paused = f.runner.run("objective", None).await.unwrap();
    let mut done = f
        .runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::Approval {
                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                result: ai_agents_hitl::ApprovalResult::Modified {
                    changes: std::collections::HashMap::from([("value".into(), json!(2))]),
                },
            },
        )
        .await
        .unwrap();
    if done.run.status == TaskRunStatus::Paused {
        done = f
            .runner
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
    assert_ne!(done.run.status, TaskRunStatus::Completed);
    assert_ne!(done.run.status, TaskRunStatus::Paused);
    assert_eq!(f.calls.load(Ordering::SeqCst), 0);
    assert_eq!(done.run.counters.tool_attempts, 0);
    assert_eq!(f.callbacks.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn controller_wrong_response_kind_and_question_shape_reject_before_claim() {
    for mode in ["approval", "question", "external"] {
        let f = fixture(mode);
        let paused = f.runner.run("objective", None).await.unwrap();
        let id = paused.run.pending.as_ref().unwrap().id.clone();
        let input = if mode == "approval" {
            TaskResumeInput::UserAnswer {
                request_id: id,
                answer: json!({"answered":true,"selected":["yes"]}),
            }
        } else {
            TaskResumeInput::Approval {
                request_id: id,
                result: ai_agents_hitl::ApprovalResult::Approved,
            }
        };
        assert!(
            f.runner
                .resume(&paused.run.key.run_id, paused.run.revision, input)
                .await
                .is_err()
        );
        assert_eq!(
            f.store
                .load(&paused.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .revision,
            paused.run.revision
        );
        if mode == "question" {
            assert!(
                f.runner
                    .resume(
                        &paused.run.key.run_id,
                        paused.run.revision,
                        TaskResumeInput::UserAnswer {
                            request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                            answer: json!({"answered":true,"selected":["invalid"]}),
                        }
                    )
                    .await
                    .is_err()
            );
            assert_eq!(
                f.store
                    .load(&paused.run.key.run_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .revision,
                paused.run.revision
            );
        }
        assert_eq!(f.calls.load(Ordering::SeqCst), 0);
        assert_eq!(f.callbacks.load(Ordering::SeqCst), 1);
        f.runner
            .cancel_paused(&paused.run.key.run_id, paused.run.revision)
            .await
            .unwrap();
    }
}

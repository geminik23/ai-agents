use super::*;
use crate::autonomy::continuation_test_support as support;
use ai_agents_core::AgentError;

// Successful message resume takes its own post-ACK path; it must not regain a deadline while its exact settlement receipt waits.
// A task-local clock keeps backend latency out of the two-second scenario and advances only at the acknowledged effect boundary.
#[tokio::test]
async fn ad_final_successful_message_resume_rechecks_original_deadline_after_ack() {
    for &sqlite in support::backends() {
        for expiry in [false, true] {
            let clock =
                super::super::invocation_clock::ManualClock::new(std::time::Duration::from_secs(4));
            clock.scope(async {
                let f = Box::pin(fixture_with_targets(false, false, &["a"], false, false)).await;
                f.parent
                    .runtime_control()
                    .set_tool_security(ai_agents_tools::ToolSecurityConfig {
                        default_timeout_ms: 2000,
                        ..Default::default()
                    });
                let store = support::store(f.parent.info().id, "batch.v1", sqlite).await;
                let fault = support::ReceiptStore::checkpoint(
                    store.clone(),
                    "hold",
                    json!({"tool":"send_agent_message", "phase":"settle", "success":true}),
                );
                let mut config = super::super::runner_tests::config(4);
                config.defaults.max_tool_calls =
                    Some(ai_agents_core::autonomy::AutonomyLimit::Finite(3));
                config.defaults.hitl = Some(AutonomyHitlConfig {
                    on_approval_required: Some(InteractionAction::PauseRun),
                    on_user_question: Some(InteractionAction::PauseRun),
                });
                let runner = AutonomyRunner::try_new(
                    f.parent.clone(),
                    config,
                    Default::default(),
                    fault.clone(),
                    "batch.v1".into(),
                )
                .unwrap();
                let paused = runner.run("objective", None).await.unwrap();
                assert_eq!(paused.run.status, TaskRunStatus::Paused);
                let tail_before = f.tail.load(Ordering::SeqCst);
                fault.armed.store(true, Ordering::SeqCst);
                let mut resume = Box::pin(runner.resume(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    TaskResumeInput::Approval {
                        request_id: paused.run.pending.unwrap().id,
                        result: ai_agents_hitl::ApprovalResult::Approved,
                    },
                ));
                tokio::select! {
                    result = &mut resume => panic!("parent settlement did not wait: {:?}", result.err()),
                    permit = fault.reached.acquire() => permit.unwrap().forget(),
                }
                assert_eq!(f.effects[0].load(Ordering::SeqCst), 1);
                let execution = fault.execution.lock().upgrade().unwrap();
                if expiry {
                    let deadline = execution
                        .approval_deadlines
                        .lock()
                        .iter()
                        .filter(|(id, _)| id.starts_with("composition:"))
                        .map(|(_, at)| *at)
                        .min()
                        .unwrap();
                    clock.advance_to(deadline + std::time::Duration::from_millis(2));
                }
                fault.release.add_permits(1);
                let result = tokio::time::timeout(std::time::Duration::from_secs(20), resume)
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(result.run.status == TaskRunStatus::Completed, !expiry);
                assert_eq!(f.effects[0].load(Ordering::SeqCst), 1);
                assert_eq!(
                    f.tail.load(Ordering::SeqCst),
                    if expiry { tail_before } else { 1 }
                );
                let saved = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
                let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload).unwrap();
                let record = &payload
                    .evidence
                    .tool_calls
                    .iter()
                    .find(|entry| entry.record.canonical_id == "send_agent_message")
                    .unwrap()
                    .record;
                assert!(record.executed && record.success);
                assert_eq!(record.metadata["_task_result_eligible"], !expiry);
                assert!(
                    payload
                        .reservations
                        .iter()
                        .filter(|entry| entry
                            .result
                            .as_ref()
                            .is_some_and(|value| value["success"] == true))
                        .all(|entry| entry.state == TaskEffectState::Completed)
                );
            }).await;
        }
    }
}

// A committed child result without its receipt cannot transfer revision authority or release either live owner.
// The same fault works against actual memory and SQLite journals without sleeps, fabricated revisions or replay.
#[tokio::test]
async fn ad_final_message_timeout_after_child_commit_preserves_unacknowledged_revision() {
    for &sqlite in support::backends() {
        let clock =
            super::super::invocation_clock::ManualClock::new(std::time::Duration::from_secs(4));
        clock.scope(async {
            let f = Box::pin(fixture_with_targets(false, false, &["a"], false, false)).await;
            f.parent.runtime_control().set_tool_security(ai_agents_tools::ToolSecurityConfig {
                default_timeout_ms: 2000,
                ..Default::default()
            });
            let store = support::store(f.parent.info().id, "batch.v1", sqlite).await;
            let fault = support::ReceiptStore::checkpoint(store.clone(), "hold", json!({"completed":0}));
            let mut config = super::super::runner_tests::config(4);
            config.defaults.max_tool_calls = Some(ai_agents_core::autonomy::AutonomyLimit::Finite(3));
            config.defaults.hitl = Some(AutonomyHitlConfig {
                on_approval_required: Some(InteractionAction::PauseRun),
                on_user_question: Some(InteractionAction::PauseRun),
            });
            let runner = AutonomyRunner::try_new(f.parent.clone(), config, Default::default(), fault.clone(), "batch.v1".into()).unwrap();
            let paused = runner.run("objective", None).await.unwrap();
            assert_eq!(paused.run.status, TaskRunStatus::Paused);
            let before = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
            let before_payload: TaskCheckpointPayload = serde_json::from_value(before.payload).unwrap();
            let completed = before_payload.reservations.iter().filter(|entry| {
                entry.state == TaskEffectState::Completed && entry.result.as_ref().is_some_and(|value| value.get("response").is_some())
            }).count();
            *fault.checkpoint.lock() = Some(json!({"completed":completed}));
            fault.armed.store(true, Ordering::SeqCst);
            let response = TaskResumeInput::Approval {
                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                result: ai_agents_hitl::ApprovalResult::Approved,
            };
            let mut resume = Box::pin(runner.resume(&paused.run.key.run_id, paused.run.revision, response.clone()));
            tokio::select! {
                result = &mut resume => panic!("child settlement did not wait: {:?}", result.err()),
                permit = fault.reached.acquire() => permit.unwrap().forget(),
            }
            fault.assert_hit();
            assert_eq!(f.effects[0].load(Ordering::SeqCst), 1);
            let execution = fault.execution.lock().upgrade().unwrap();
            let acknowledged = execution.acknowledged_snapshot();
            let committed = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
            assert_eq!(committed.revision, acknowledged.revision + 1);
            assert_eq!(committed.owner_token, acknowledged.owner_token);
            assert_eq!(committed.status, TaskRunStatus::Running);
            let saved_payload: TaskCheckpointPayload = serde_json::from_value(committed.payload.clone()).unwrap();
            assert_eq!(saved_payload.counters.llm_attempts, before_payload.counters.llm_attempts + 1);
            assert_eq!(saved_payload.counters.tool_attempts, before_payload.counters.tool_attempts + 1);
            let calls = (f.main.calls.load(Ordering::SeqCst), f.providers[0].calls.load(Ordering::SeqCst));
            let deadline = execution.approval_deadlines.lock().iter()
                .filter(|(id, _)| id.starts_with("composition:"))
                .map(|(_, deadline)| *deadline).min().unwrap();
            clock.advance_to(deadline + std::time::Duration::from_millis(2));
            let result = tokio::time::timeout(std::time::Duration::from_secs(20), resume).await.unwrap();
            assert!(matches!(result, Err(AgentError::TaskRunStorage(TaskRunStorageError::Conflict))));
            assert_eq!(execution.acknowledged_snapshot().revision, acknowledged.revision);
            assert!(execution.owner.check().is_err());
            assert!(runner.resume(&paused.run.key.run_id, paused.run.revision, response).await.is_err());
            assert!(f.parent.chat("must not replay").await.is_err());
            assert_eq!(f.effects[0].load(Ordering::SeqCst), 1);
            assert_eq!((f.main.calls.load(Ordering::SeqCst), f.providers[0].calls.load(Ordering::SeqCst)), calls);
            let after = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
            assert_eq!(serde_json::to_value(after).unwrap(), serde_json::to_value(committed).unwrap());
        }).await;
    }
}

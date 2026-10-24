use super::*;
use crate::autonomy::continuation_test_support as support;

// Successful message resume takes its own post-ACK path; it must not regain a deadline while its exact settlement receipt waits.
#[tokio::test]
async fn ad_final_successful_message_resume_rechecks_original_deadline_after_ack() {
    for &sqlite in support::backends() {
        for expiry in [false, true] {
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
                json!({"tool":"send_agent_message", "phase":"settle"}),
            );
            let mut config = super::super::runner_tests::config(4);
            config.defaults.max_tool_calls = Some(3);
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
            tokio::select! { result = &mut resume => panic!("parent settlement did not wait: {:?}", result.err()), permit = fault.reached.acquire() => permit.unwrap().forget() }
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
                std::thread::sleep(
                    deadline.saturating_duration_since(std::time::Instant::now())
                        + std::time::Duration::from_millis(2),
                );
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
        }
    }
}

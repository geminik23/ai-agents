//! Exact response authority and configured timeout dispositions survive reconstruction without granting later gates.
use super::*;
use crate::autonomy::restoration_test_support::{FreshRuntime, RebuildHost, restore};

// Use the original suspension fixture's exact profile; restoring is not permission to widen request or run bounds.
fn task_config() -> AutonomyConfig {
    let mut config = crate::autonomy::runner_tests::config(3);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: None,
    });
    config
}

// An expired but explicitly approved earlier gate is retained as its exact disposition, never copied onto the second pending gate.
#[tokio::test]
async fn restoration_timeout_approved_first_gate_then_second_gate_requires_its_own_response() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        let (agent, _, _, first, _, provider) =
            fixture_request_timeout(false, false, true, false, false, Some(2));
        let store = crate::autonomy::continuation_test_support::store(
            agent.info().id,
            "pause-test.v1",
            sqlite,
        )
        .await;
        let runner = AutonomyRunner::try_new(
            agent.clone(),
            task_config(),
            Default::default(),
            store.clone(),
            "pause-test.v1".into(),
        )
        .unwrap();
        let witness = Arc::new(parking_lot::Mutex::new(None));
        let stage_witness = witness.clone();
        let host = RebuildHost::new(
            "two-gates.v1",
            &FreshRuntime::new(agent.clone(), vec![provider.clone()]),
            Box::new(move || {
                let witness = stage_witness.clone();
                Box::pin(async move {
                    let (agent, _, _, first, _, provider) =
                        fixture_request_timeout(false, false, true, false, false, Some(2));
                    *witness.lock() = Some((first, provider.clone()));
                    FreshRuntime::new(agent, vec![provider])
                })
            }),
        );
        let initial = runner.run("objective", None).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(2050)).await;
        let next = runner
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
        assert_eq!(next.run.status, TaskRunStatus::Paused);
        assert_eq!(first.load(Ordering::SeqCst), 0);
        let old = Arc::downgrade(&agent);
        drop(agent);
        let (restored, paused) = restore(
            runner,
            store.clone(),
            &next.run,
            task_config(),
            "pause-test.v1",
            host,
        )
        .await;
        assert!(old.upgrade().is_none());
        assert_eq!(witness.lock().as_ref().unwrap().0.load(Ordering::SeqCst), 0);
        assert!(
            restored
                .resume(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    TaskResumeInput::Approval {
                        request_id: initial.run.pending.as_ref().unwrap().id.clone(),
                        result: ai_agents_hitl::ApprovalResult::Approved
                    }
                )
                .await
                .is_err()
        );
        let done = restored
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
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        let witness = witness.lock();
        let (calls, fresh_provider) = witness.as_ref().unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(first.load(Ordering::SeqCst), 0);
        assert_eq!(
            provider.calls.load(Ordering::SeqCst) + fresh_provider.calls.load(Ordering::SeqCst),
            2
        );
    }
}

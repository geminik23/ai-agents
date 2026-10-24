use super::*;
use crate::Agent;
use std::sync::atomic::{AtomicUsize, Ordering};

// Even an unpolled captured producer belongs to the run and blocks a safe snapshot; terminal closure denies its later admission.
#[tokio::test]
async fn ad_work_unpolled_enrollment_and_terminal_close_protect_owner() {
    let (agent, execution) = crate::autonomy::child_tests::execution(2).await;
    scope_execution(execution.clone(), async {
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        let work = owned_maintenance("test", async move {
            observed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });
        assert!(execution.owner.work.seal().is_err());
        let saved = execution
            .terminal(TaskRunStatus::Completed, "candidate".into())
            .await
            .unwrap();
        assert_eq!(saved.status, TaskRunStatus::RecoveryRequired);
        drop(work);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(owned_maintenance("late", async { Ok(()) }).await.is_err());
        assert!(agent.chat("foreign").await.is_err());
    })
    .await;
}

// A snapshot seal is atomic with enrollment and does not create authority when it is locally released.
#[tokio::test]
async fn ad_work_snapshot_seal_rejects_late_producer_without_effects() {
    let (_, execution) = crate::autonomy::child_tests::execution(2).await;
    scope_execution(execution.clone(), async {
        let seal = execution.owner.work.seal().unwrap();
        let calls = AtomicUsize::new(0);
        assert!(
            owned_maintenance("late", async {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
            .await
            .is_err()
        );
        assert!(!execution.owner.work.unsettled());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        drop(seal);
        owned_maintenance("allowed", async { Ok(()) })
            .await
            .unwrap();
        assert!(!execution.owner.work.unsettled());
    })
    .await;
}

// Dropping an admitted opaque effect tail cannot be promoted to quiescence merely because its future vanished.
#[tokio::test]
async fn ad_work_drop_after_effect_retains_uncertainty() {
    let (agent, execution) = crate::autonomy::child_tests::execution(2).await;
    scope_execution(execution.clone(), async {
        let entered = Arc::new(tokio::sync::Semaphore::new(0));
        let signal = entered.clone();
        let mut work = owned_maintenance("effect", async move {
            maintenance_effect();
            signal.add_permits(1);
            std::future::pending::<()>().await;
            Ok(())
        });
        tokio::select! { result = &mut work => panic!("tail completed: {result:?}"), permit = entered.acquire() => permit.unwrap().forget() }
        drop(work);
        assert!(execution.owner.work.unsettled());
        assert!(execution.await_owned_work().await.is_err());
        let saved = execution.terminal(TaskRunStatus::Completed, "candidate".into()).await.unwrap();
        assert_eq!(saved.status, TaskRunStatus::RecoveryRequired);
        assert!(agent.chat("foreign").await.is_err());
    }).await;
}

// A pre-body deadline refusal retires a zero-count logical marker, not an unrelated provider/tool reservation.
#[tokio::test]
async fn ad_work_expired_inherited_bound_never_polls_or_refunds_body() {
    let (_, execution) = crate::autonomy::child_tests::execution(2).await;
    scope_execution(execution.clone(), async {
        let bounds = ai_agents_llm::InvocationBounds::new(
            std::time::SystemTime::now(),
            std::time::Instant::now(),
        );
        let calls = AtomicUsize::new(0);
        let result = ai_agents_llm::scope_invocation_bounds(Some(bounds), async {
            owned_maintenance("expired", async {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
            .await
        })
        .await;
        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let snapshot = execution.load_owned().await.unwrap();
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
        assert_eq!(payload.counters.llm_attempts, 0);
        assert_eq!(payload.counters.tool_attempts, 0);
        assert!(
            payload
                .reservations
                .iter()
                .all(|reservation| reservation.state == TaskEffectState::Completed)
        );
        assert!(!execution.owner.work.unsettled());
    })
    .await;
}

// The narrower original deadline is checked before every body poll, including a ready wakeup after expiry.
#[tokio::test]
async fn ad_work_original_bound_blocks_late_callback_effect() {
    let (_, execution) = crate::autonomy::child_tests::execution(2).await;
    scope_execution(execution.clone(), async {
        let cut = std::time::Instant::now() + Duration::from_millis(40);
        let bounds = ai_agents_llm::InvocationBounds::new(std::time::SystemTime::now() + Duration::from_millis(40), cut);
        let entered = Arc::new(tokio::sync::Semaphore::new(0));
        let release = Arc::new(tokio::sync::Semaphore::new(0));
        let calls = Arc::new(AtomicUsize::new(0));
        let (signal, gate, effects) = (entered.clone(), release.clone(), calls.clone());
        let mut work = Box::pin(ai_agents_llm::scope_invocation_bounds(Some(bounds), async move {
            owned_maintenance("callback", async move {
                signal.add_permits(1);
                gate.acquire().await.unwrap().forget();
                maintenance_effect();
                effects.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }).await
        }));
        tokio::select! { result = &mut work => panic!("no callback admission: {result:?}"), permit = entered.acquire() => permit.unwrap().forget() }
        tokio::time::sleep_until(cut.into()).await;
        release.add_permits(1);
        assert!(work.await.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(execution.remaining_duration() > Duration::ZERO);
    }).await;
}

// Actual managed calls retain the shared cap while a denied pure proposal does not fabricate an uncertain external mutation.
#[tokio::test]
async fn ad_work_capacity_refusal_preserves_real_attempt_and_known_limit() {
    let (_, execution) = crate::autonomy::child_tests::execution(1).await;
    let mut raw = ai_agents_llm::mock::MockLLMProvider::new("test");
    raw.set_response("done");
    let provider = ai_agents_llm::managed_provider(Arc::new(raw));
    scope_execution(execution.clone(), async {
        provider
            .complete(&[ai_agents_core::ChatMessage::user("main")], None)
            .await
            .unwrap();
        let result = owned_maintenance("proposal", async {
            provider
                .complete(&[ai_agents_core::ChatMessage::user("auxiliary")], None)
                .await?;
            Ok(())
        })
        .await;
        assert!(result.is_err());
        let snapshot = execution
            .terminal(TaskRunStatus::LimitReached, "maintenance_capacity".into())
            .await
            .unwrap();
        assert_eq!(snapshot.status, TaskRunStatus::LimitReached);
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
        assert_eq!(payload.counters.llm_attempts, 1);
        assert!(
            payload
                .reservations
                .iter()
                .all(|reservation| reservation.state == TaskEffectState::Completed)
        );
    })
    .await;
}

use super::continuation_test_support as support;
use super::*;
use crate::Agent;
use ai_agents_core::Result;
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

// Reuse the existing actual owner and conditional backend, selecting the exact maintenance admission or settlement write.
async fn fixture(
    sqlite: bool,
    phase: &str,
    mode: &'static str,
) -> (
    Arc<crate::RuntimeAgent>,
    Arc<RunExecution>,
    Arc<support::ReceiptStore>,
) {
    let (agent, original) = super::child_tests::execution(2).await;
    let snapshot = original.load_owned().await.unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone()).unwrap();
    let inner = support::store(agent.info().id, "children-v1", sqlite).await;
    inner.create(&snapshot).await.unwrap();
    let fault = support::ReceiptStore::checkpoint(
        inner,
        mode,
        json!({"maintenance":"fault-tail","phase":phase}),
    );
    fault.armed.store(true, Ordering::SeqCst);
    let execution = RunExecution::new(
        fault.clone(),
        &snapshot,
        LifecycleState::new(&payload.settings, false).unwrap(),
        original.owner.clone(),
        std::time::Instant::now(),
    )
    .unwrap();
    (agent, execution, fault)
}

// Reject/lost acknowledgements retain actual custody and never replay the callback, on both supported task backends.
#[tokio::test]
async fn ad_maintenance_admit_and_settlement_ack_fault_matrix() {
    for &sqlite in support::backends() {
        for phase in ["admit", "settle"] {
            for mode in ["reject", "lost"] {
                let (agent, execution, fault) = fixture(sqlite, phase, mode).await;
                let calls = Arc::new(AtomicUsize::new(0));
                let effects = calls.clone();
                let result: Result<()> = scope_execution(execution.clone(), async move {
                    super::owned_maintenance("fault-tail", async move {
                        super::work::maintenance_effect();
                        effects.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    })
                    .await
                })
                .await;
                assert!(result.is_err(), "{sqlite}/{phase}/{mode}");
                fault.assert_hit();
                assert_eq!(calls.load(Ordering::SeqCst), usize::from(phase == "settle"));
                assert!(execution.owner.work.unsettled());
                assert!(execution.await_owned_work().await.is_err());
                assert!(agent.chat("unrelated").await.is_err());
                let before = calls.load(Ordering::SeqCst);
                assert!(
                    scope_execution(execution.clone(), async {
                        super::owned_maintenance("retry", async {
                            calls.fetch_add(1, Ordering::SeqCst);
                            Ok(())
                        })
                        .await
                    })
                    .await
                    .is_err()
                );
                assert_eq!(calls.load(Ordering::SeqCst), before);
            }
        }
    }
}

// A dropped acknowledgement wait retains the committed row and the live owner even when the body itself already returned.
#[tokio::test]
async fn ad_maintenance_drop_during_admission_and_settlement_ack_keeps_custody() {
    for &sqlite in support::backends() {
        for phase in ["admit", "settle"] {
            let (agent, execution, fault) = fixture(sqlite, phase, "drop").await;
            let calls = Arc::new(AtomicUsize::new(0));
            let effects = calls.clone();
            let mut work = Box::pin(scope_execution(execution.clone(), async move {
                super::owned_maintenance("fault-tail", async move {
                    super::work::maintenance_effect();
                    effects.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })
                .await
            }));
            tokio::select! { result = &mut work => panic!("ack did not block: {result:?}"), permit = fault.reached.acquire() => permit.unwrap().forget() }
            assert_eq!(calls.load(Ordering::SeqCst), usize::from(phase == "settle"));
            drop(work);
            fault.assert_hit();
            assert!(execution.owner.work.unsettled());
            assert!(agent.chat("unrelated").await.is_err());
            let saved = fault.inner.load(&execution.run_id).await.unwrap().unwrap();
            assert_eq!(saved.status, TaskRunStatus::Running);
        }
    }
}

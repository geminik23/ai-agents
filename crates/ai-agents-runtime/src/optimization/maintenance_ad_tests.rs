use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

// Drain cancellation must leave the real join handles and their occupied capacity in the queue.
#[tokio::test]
async fn ad_background_dropped_drain_keeps_capacity_and_retries() {
    let queue = BackgroundMaintenanceQueue::new(1);
    let entered = Arc::new(tokio::sync::Semaphore::new(0));
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let (ready, gate) = (entered.clone(), release.clone());
    queue
        .spawn(None, async move {
            ready.add_permits(1);
            gate.acquire().await.unwrap().forget();
            Ok(())
        })
        .unwrap();
    entered.acquire().await.unwrap().forget();
    let mut flush = Box::pin(queue.flush_all());
    assert!(futures::poll!(&mut flush).is_pending());
    assert_eq!(queue.len(), 1);
    assert!(queue.is_full());
    assert!(queue.spawn(None, async { Ok(()) }).is_err());
    drop(flush);
    assert_eq!(queue.len(), 1);
    release.add_permits(1);
    queue.flush_all().await.unwrap();
    assert!(queue.is_empty());
}

// A first error cannot detach a still-running sibling or make a second observer see false quiescence.
#[tokio::test]
async fn ad_background_failure_drains_siblings_and_preserves_error_after_drop() {
    let queue = BackgroundMaintenanceQueue::new(2);
    queue
        .spawn(None, async {
            Err(AgentError::Other("first failure".into()))
        })
        .unwrap();
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let gate = release.clone();
    queue
        .spawn(None, async move {
            gate.acquire().await.unwrap().forget();
            Ok(())
        })
        .unwrap();
    let mut first = Box::pin(queue.flush_all());
    assert!(futures::poll!(&mut first).is_pending());
    assert_eq!(queue.len(), 2);
    let mut second = Box::pin(queue.flush_all());
    assert!(futures::poll!(&mut second).is_pending());
    drop(first);
    release.add_permits(1);
    assert!(
        second
            .await
            .unwrap_err()
            .to_string()
            .contains("first failure")
    );
    assert!(queue.is_empty());
}

// Selective drains retire only their captured set; an admission during that drain remains discoverable.
#[tokio::test]
async fn ad_background_selective_drain_preserves_late_and_unrelated_work() {
    let queue = BackgroundMaintenanceQueue::new(3);
    let key = MaintenanceSequenceKey::actor("agent", "actor", RuntimeTaskPurpose::PostTurnFacts);
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    let release = gate.clone();
    queue
        .spawn(Some(key.clone()), async move {
            release.acquire().await.unwrap().forget();
            Ok(())
        })
        .unwrap();
    let mut flush = Box::pin(queue.flush_scope("actor"));
    assert!(futures::poll!(&mut flush).is_pending());
    queue.spawn(None, async { Ok(()) }).unwrap();
    queue.spawn(Some(key), async { Ok(()) }).unwrap();
    gate.add_permits(1);
    flush.await.unwrap();
    assert_eq!(queue.len(), 2);
    queue.flush_all().await.unwrap();
    assert!(queue.is_empty());
}

// Same-key order follows admission rather than spawned-task polling order.
#[tokio::test]
async fn ad_background_same_key_bodies_keep_enqueue_order() {
    let queue = BackgroundMaintenanceQueue::new(4);
    let ordinal = Arc::new(AtomicUsize::new(0));
    for expected in 0..4 {
        let ordinal = ordinal.clone();
        queue
            .spawn(
                Some(MaintenanceSequenceKey::actor(
                    "agent",
                    "actor",
                    RuntimeTaskPurpose::PostTurnFacts,
                )),
                async move {
                    assert_eq!(ordinal.fetch_add(1, Ordering::SeqCst), expected);
                    Ok(())
                },
            )
            .unwrap();
    }
    queue.flush_all().await.unwrap();
    assert_eq!(ordinal.load(Ordering::SeqCst), 4);
}

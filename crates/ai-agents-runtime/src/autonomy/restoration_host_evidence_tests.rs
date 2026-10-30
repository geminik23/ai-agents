//! Host evidence freshness remains current after process-local driver authority has been reconstructed.
use super::*;
use crate::autonomy::restoration_test_support::{FreshRuntime, RebuildHost, restore};

// The external world is not rolled back with the runtime; changed ticket/world revisions invalidate stored proof before callback reentry.
#[tokio::test]
async fn restoration_external_host_revision_is_current_not_refreshed_from_checkpoint() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        for changed in [false, true] {
            let source = Revision::new();
            let (f, source_callbacks) = fixture(
                sqlite,
                source.clone(),
                "external",
                ExternalUnavailablePolicy::Stop,
            )
            .await;
            let stage_source = source.clone();
            let callbacks = Arc::new(parking_lot::Mutex::new(vec![]));
            let stage_callbacks = callbacks.clone();
            let host = RebuildHost::new(
                "revision-fixture.v1",
                &FreshRuntime::new(f.agent.clone(), vec![f.provider.clone()]),
                Box::new(move || {
                    let source = stage_source.clone();
                    let callbacks = stage_callbacks.clone();
                    Box::pin(async move {
                        let (fresh, count) =
                            configured(source, "external", ExternalUnavailablePolicy::Stop);
                        callbacks.lock().push((count, fresh.provider.clone()));
                        FreshRuntime::new(fresh.agent.clone(), vec![fresh.provider.clone()])
                    })
                }),
            );
            let first = f.runner.run("objective", None).await.unwrap();
            let old = Arc::downgrade(&f.agent);
            let crate::autonomy::controller_tests::Fixture {
                agent,
                runner,
                store,
                config,
                ..
            } = f;
            drop(agent);
            let (restored, paused) = restore(
                runner,
                store.clone(),
                &first.run,
                config,
                "controller.v1",
                host,
            )
            .await;
            assert!(old.upgrade().is_none());
            if changed {
                source.change();
            }
            let result = restored
                .resume(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    signal(&paused.run),
                )
                .await;
            if changed {
                assert!(
                    !result
                        .as_ref()
                        .is_ok_and(|result| result.run.status == TaskRunStatus::Completed)
                );
                assert_eq!(callbacks.lock()[0].0.load(Ordering::SeqCst), 0);
                assert_ne!(
                    store
                        .load(&paused.run.key.run_id)
                        .await
                        .unwrap()
                        .unwrap()
                        .status,
                    TaskRunStatus::Completed
                );
            } else {
                assert_eq!(result.unwrap().run.status, TaskRunStatus::Completed);
                assert_eq!(callbacks.lock()[0].0.load(Ordering::SeqCst), 1);
            }
            assert_eq!(source_callbacks.load(Ordering::SeqCst), 1);
            assert_eq!(callbacks.lock()[0].1.calls.load(Ordering::SeqCst), 0);
        }
    }
}

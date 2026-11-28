//! State reconstruction preserves real exit/commit/entry positions and historical action journals.
use super::*;
use crate::autonomy::restoration_test_support::{FreshRuntime, RebuildHost, restore};

// Reusing production state fixtures catches generation/journal collisions without simulating transition execution in the test.
#[tokio::test]
async fn restoration_state_exit_entry_and_reentry_preserve_commit_and_hooks() {
    for &sqlite in support::backends() {
        for (entering, reentry) in [(false, false), (true, false), (true, true)] {
            let f = Box::pin(fixture(entering, false, reentry, sqlite)).await;
            let witnesses = Arc::new(parking_lot::Mutex::new(vec![]));
            let fresh_witnesses = witnesses.clone();
            let host = RebuildHost::new(
                "state-fixture.v1",
                &FreshRuntime::new(f.agent.clone(), f.restore_providers.clone()),
                Box::new(move || {
                    let witnesses = fresh_witnesses.clone();
                    Box::pin(async move {
                        let fresh = Box::pin(fixture(entering, false, reentry, false)).await;
                        witnesses.lock().push((
                            fresh.first.clone(),
                            fresh.second.clone(),
                            fresh.hooks.clone(),
                            fresh.arguments.clone(),
                        ));
                        FreshRuntime::new(fresh.agent.clone(), fresh.restore_providers.clone())
                    })
                }),
            );
            let initial = f.runner.run("objective", None).await.unwrap();
            assert_eq!(initial.run.status, TaskRunStatus::Paused);
            let old = Arc::downgrade(&f.agent);
            let source_first = f.first.clone();
            let source_hooks = f.hooks.clone();
            let Fixture {
                agent,
                runner,
                store,
                ..
            } = f;
            drop(agent);
            let mut config = crate::autonomy::runner_tests::config(16);
            config.defaults.hitl = Some(AutonomyHitlConfig {
                on_user_question: Some(InteractionAction::PauseRun),
                on_approval_required: Some(InteractionAction::PauseRun),
            });
            let (restored, paused) = restore(
                runner,
                store.clone(),
                &initial.run,
                config,
                "state.v1",
                host.clone(),
            )
            .await;
            assert!(old.upgrade().is_none());
            let done = restored
                .resume(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    TaskResumeInput::Approval {
                        request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                        result: ApprovalResult::Modified {
                            changes: std::collections::HashMap::from([(
                                "value".into(),
                                json!("modified"),
                            )]),
                        },
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
            assert_eq!(source_first.load(Ordering::SeqCst), 1);
            let witnesses = witnesses.lock();
            let (first, second, hooks, arguments) = &witnesses[0];
            assert_eq!(first.load(Ordering::SeqCst), 0);
            assert_eq!(second.load(Ordering::SeqCst), 1);
            assert_eq!(
                source_hooks.transitions.load(Ordering::SeqCst)
                    + hooks.transitions.load(Ordering::SeqCst),
                1
            );
            assert_eq!(
                source_hooks.responses.load(Ordering::SeqCst)
                    + hooks.responses.load(Ordering::SeqCst),
                1
            );
            assert_eq!(arguments.lock().last().unwrap()["value"], "modified");
            let agent = host.staged.lock().upgrade().unwrap();
            assert_eq!(agent.state_history().len(), 1);
            assert!(!agent.get_context().contains_key("wrong_entry"));
        }
    }
}

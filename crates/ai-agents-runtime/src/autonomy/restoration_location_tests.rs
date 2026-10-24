//! Script reconstruction reuses the established skill fixtures and the same production skill/controller drivers.
use super::*;
use crate::autonomy::restoration_test_support::{FreshRuntime, RebuildHost, restore};

struct Witness {
    first: Arc<AtomicUsize>,
    second: Arc<AtomicUsize>,
    responses: Arc<AtomicUsize>,
    main: Arc<crate::autonomy::runner_tests::RecordingProvider>,
    router: Arc<crate::autonomy::runner_tests::RecordingProvider>,
}
impl Witness {
    // Witnesses keep counters/providers, never a runtime, runner, owner or executable cursor.
    fn capture(f: &Fixture) -> Self {
        Self {
            first: f.first.clone(),
            second: f.second.clone(),
            responses: f.responses.clone(),
            main: f.main.clone(),
            router: f.router.clone(),
        }
    }
}

// Real loaded scripts preserve rendered arguments and repeated step identity, with one hook per completed turn rather than per reconstruction.
#[tokio::test]
async fn restoration_skill_exact_cursor_repause_and_stateful_provider() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        for question in [false, true] {
            let mut f = Box::pin(skill_fixture_options(question, true, false)).await;
            f.store = crate::autonomy::continuation_test_support::store(
                f.agent.info().id,
                "locations.v1",
                sqlite,
            )
            .await;
            f.runner = runner(f.agent.clone(), f.store.clone());
            let witnesses = Arc::new(parking_lot::Mutex::new(Vec::<Witness>::new()));
            let stage_witnesses = witnesses.clone();
            let host = RebuildHost::new(
                "script-fixture.v1",
                &FreshRuntime::new(f.agent.clone(), vec![f.main.clone(), f.router.clone()]),
                Box::new(move || {
                    let witnesses = stage_witnesses.clone();
                    Box::pin(async move {
                        let f = Box::pin(skill_fixture_options(question, true, false)).await;
                        witnesses.lock().push(Witness::capture(&f));
                        FreshRuntime::new(f.agent.clone(), vec![f.main.clone(), f.router.clone()])
                    })
                }),
            );
            let initial = f
                .runner
                .run_skill("objective", "script", None)
                .await
                .unwrap();
            assert_eq!(
                initial.run.status,
                TaskRunStatus::Paused,
                "{:?}",
                initial.run.stop_reason
            );
            assert_eq!(f.first.load(Ordering::SeqCst), 1);
            let source = Witness::capture(&f);
            let old = Arc::downgrade(&f.agent);
            let Fixture {
                agent,
                runner,
                store,
                ..
            } = f;
            drop(agent);
            let (restored, paused) = restore(
                runner,
                store.clone(),
                &initial.run,
                task_config(),
                "locations.v1",
                host.clone(),
            )
            .await;
            assert!(old.upgrade().is_none(), "source runtime retained");
            assert_eq!(source.main.calls.load(Ordering::SeqCst), 1);
            assert_eq!(
                source.router.calls.load(Ordering::SeqCst),
                0,
                "explicit entry must not route"
            );
            let input = |run: &TaskRun| {
                if question {
                    TaskResumeInput::UserAnswer {
                        request_id: run.pending.as_ref().unwrap().id.clone(),
                        answer: json!({"answered":true,"selected":["yes"]}),
                    }
                } else {
                    TaskResumeInput::Approval {
                        request_id: run.pending.as_ref().unwrap().id.clone(),
                        result: ai_agents_hitl::ApprovalResult::Modified {
                            changes: std::collections::HashMap::from([(
                                "value".into(),
                                json!("changed"),
                            )]),
                        },
                    }
                }
            };
            let next = restored
                .resume(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    input(&paused.run),
                )
                .await
                .unwrap();
            assert_eq!(
                next.run.status,
                TaskRunStatus::Paused,
                "repeated step must park independently"
            );
            assert_ne!(
                next.run.pending.as_ref().unwrap().id,
                paused.run.pending.as_ref().unwrap().id
            );
            let done = restored
                .resume(&next.run.key.run_id, next.run.revision, input(&next.run))
                .await
                .unwrap();
            assert_eq!(
                done.run.status,
                TaskRunStatus::Completed,
                "{:?}",
                done.run.stop_reason
            );
            assert_eq!(source.first.load(Ordering::SeqCst), 1);
            assert_eq!(source.second.load(Ordering::SeqCst), 0);
            assert_eq!(source.responses.load(Ordering::SeqCst), 0);
            {
                let stages = witnesses.lock();
                assert_eq!(stages.len(), 1);
                let fresh = &stages[0];
                assert_eq!(
                    fresh.first.load(Ordering::SeqCst),
                    0,
                    "completed prefix replayed"
                );
                assert_eq!(
                    fresh.second.load(Ordering::SeqCst),
                    if question { 0 } else { 2 }
                );
                assert_eq!(fresh.responses.load(Ordering::SeqCst), 2);
                assert_eq!(done.run.counters.turns, 2);
                let rendered = fresh.main.captured_messages()[0]
                    .iter()
                    .map(|message| message.content.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                assert!(rendered.contains("prefix") && !rendered.contains("{{ steps"));
                assert_eq!(
                    fresh.main.calls.load(Ordering::SeqCst),
                    2,
                    "saved provider queue must continue, not replay prefix"
                );
                assert_eq!(fresh.router.calls.load(Ordering::SeqCst), 0);
            }
            let snapshot = store.load(&done.run.key.run_id).await.unwrap().unwrap();
            let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
            if !question {
                let calls: Vec<_> = payload
                    .evidence
                    .tool_calls
                    .iter()
                    .filter(|call| call.record.canonical_id == "second" && call.record.executed)
                    .collect();
                assert_eq!(calls.len(), 2);
                assert!(
                    calls
                        .iter()
                        .all(|call| call.record.executed_arguments["value"] == "changed")
                );
            }
            assert!(done.run.counters.llm_attempts >= initial.run.counters.llm_attempts);
        }
    }
}

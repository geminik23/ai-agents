//! Event acceptance uses the real controller, conditional store and managed provider rather than fabricated timelines.

use super::runner_tests::{RecordingProvider, fixture};
use super::*;
use futures::StreamExt;
use serde_json::json;
use std::sync::atomic::Ordering;

// Draining observes presentation separately from the exact trusted-host result.
async fn drain(
    mut stream: TaskRunStream<'_>,
) -> (Vec<TaskRunEvent>, ai_agents_core::Result<TaskRunResult>) {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    (
        events,
        stream
            .take_result()
            .expect("EOF must retain an operation result"),
    )
}

// No poll means no task admission, provider request or canonical state mutation.
#[tokio::test]
async fn ev_never_polled_drop_does_not_admit_work() {
    let provider = RecordingProvider::new(&["ordinary"]);
    let (agent, runner, store) = fixture(provider.clone(), 2);
    drop(runner.run_stream("secret objective", None).unwrap());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert!(store.list().await.unwrap().is_empty());
    assert!(agent.captured_task_owner().is_none());
    use crate::Agent;
    assert_eq!(agent.chat("ordinary").await.unwrap().content, "ordinary");
}

// Last-slot completion and model-led continuation use the same counters and redacted terminal contract.
#[tokio::test]
async fn ev_blocking_stream_parity_and_default_redaction() {
    for capacity in [1, 2, 32] {
        let (_, blocking, _) = fixture(
            RecordingProvider::new(&["working secret", "done secret"]),
            2,
        );
        let expected = blocking.run("private objective", None).await.unwrap();
        let provider = RecordingProvider::new(&["working secret", "done secret"]);
        let (agent, runner, store) = fixture(provider.clone(), 2);
        let (events, result) = drain(
            runner
                .stream_operation(
                    TaskStreamOperation::Run {
                        objective: "private objective".into(),
                        profile: None,
                    },
                    TaskStreamConfig { capacity },
                )
                .unwrap(),
        )
        .await;
        let result = result.unwrap();
        assert_eq!(result.run.status, expected.run.status);
        assert_eq!(
            serde_json::to_value(&result.run.counters).unwrap(),
            serde_json::to_value(expected.run.counters).unwrap()
        );
        assert_eq!(result.final_response.unwrap().content, "done secret");
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        assert_eq!(events.last().unwrap().kind, TaskRunEventKind::TaskCompleted);
        assert_eq!(events.last().unwrap().revision, Some(result.run.revision));
        assert!(
            events
                .iter()
                .any(|e| e.kind == TaskRunEventKind::ContinuationForced)
        );
        assert!(
            events
                .iter()
                .any(|e| e.kind == TaskRunEventKind::CompletionGateChecked)
        );
        for pair in events.windows(2) {
            assert!(pair[0].sequence < pair[1].sequence);
            assert!(pair[0].cursor < pair[1].cursor);
        }
        let encoded = serde_json::to_string(&events).unwrap();
        assert!(!encoded.contains("secret"));
        assert!(!encoded.contains("private objective"));
        assert!(encoded.len() < events.len() * 4096);
        assert_eq!(
            store
                .load(&result.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskRunStatus::Completed
        );
        assert!(agent.captured_task_owner().is_none());
    }
}

// Foreground interruption retains the owner instead of making unrelated ordinary work legal.
#[tokio::test]
async fn ev_drop_after_admission_keeps_recovery_protection() {
    let provider = RecordingProvider::gated();
    let (agent, runner, store) = fixture(provider.clone(), 2);
    let mut stream = runner.run_stream("objective", None).unwrap();
    let first = stream.next().await.unwrap();
    assert_eq!(first.kind, TaskRunEventKind::TaskStarted);
    drop(stream);
    assert!(agent.captured_task_owner().unwrap().is_abandoned());
    assert_ne!(
        store.list().await.unwrap()[0].status,
        TaskRunStatus::Completed
    );
}

// Awaited close drives the original execution to its actual acknowledged status, not a synthetic cancellation event.
#[tokio::test]
async fn ev_awaited_close_settles_and_releases() {
    let provider = RecordingProvider::new(&["done"]);
    let (agent, runner, store) = fixture(provider, 2);
    let mut stream = runner
        .stream_operation(
            TaskStreamOperation::Run {
                objective: "objective".into(),
                profile: None,
            },
            TaskStreamConfig { capacity: 1 },
        )
        .unwrap();
    assert_eq!(
        stream.next().await.unwrap().kind,
        TaskRunEventKind::TaskStarted
    );
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), stream.close())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Cancelled);
    assert!(agent.captured_task_owner().is_none());
    assert_eq!(
        store
            .load(&result.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .status,
        TaskRunStatus::Cancelled
    );
}

// Invalid operations expose a safe error event and retain the structured error only for trusted host access.
#[tokio::test]
async fn ev_invalid_config_and_stale_resume_do_not_execute() {
    let provider = RecordingProvider::new(&["done"]);
    let (_, runner, store) = fixture(provider.clone(), 2);
    assert!(
        runner
            .stream_operation(
                TaskStreamOperation::Run {
                    objective: "x".into(),
                    profile: None
                },
                TaskStreamConfig { capacity: 0 }
            )
            .is_err()
    );
    let (events, result) = drain(runner.run_stream("", None).unwrap()).await;
    assert!(result.is_err());
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind, TaskRunEventKind::OperationError);
    let (events, result) = drain(
        runner
            .resume_stream("secret missing run", 0, TaskResumeInput::Continue)
            .unwrap(),
    )
    .await;
    assert!(result.is_err());
    assert!(
        !serde_json::to_string(&events)
            .unwrap()
            .contains("secret missing run")
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert!(store.list().await.unwrap().is_empty());
}

// Normal controller questions are real retained pauses; ending their stream cannot cancel the pending task.
#[tokio::test]
async fn ev_question_pause_control_resume_and_nonstream_reuse() {
    use crate::Agent;
    let provider = RecordingProvider::new(&["not finished", "done", "ordinary"]);
    let (agent, _, store) = fixture(provider.clone(), 4);
    let mut config = super::runner_tests::config(4);
    config.defaults.premature_finish = Some(
        serde_json::from_value(json!({"action":"ask_user", "on_unavailable":"stop"})).unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent.clone(),
        config,
        Default::default(),
        store.clone(),
        "prepared-v1".into(),
    )
    .unwrap();
    let (events, paused) = drain(runner.run_stream("objective", None).unwrap()).await;
    let paused = paused.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    assert_eq!(events.last().unwrap().kind, TaskRunEventKind::TaskPaused);
    assert!(!agent.captured_task_owner().unwrap().is_abandoned());
    let request = paused.run.pending.unwrap();
    // Nonstream continuation proves an old closed event channel cannot poison a retained owner.
    let done = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: request.id,
                answer: json!("continue"),
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    assert_eq!(agent.chat("ordinary").await.unwrap().content, "ordinary");
}

// A streamed controller response must finish exclusive execution initialization before publishing weak cancellation handles.
#[tokio::test]
async fn ev_streamed_question_response_preserves_exclusive_initialization() {
    let provider = RecordingProvider::new(&["not finished", "done"]);
    let (agent, _, store) = fixture(provider.clone(), 4);
    let mut config = super::runner_tests::config(4);
    config.defaults.premature_finish = Some(
        serde_json::from_value(json!({"action":"ask_user", "on_unavailable":"stop"})).unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        Default::default(),
        store,
        "prepared-v1".into(),
    )
    .unwrap();
    let (before, paused) = drain(runner.run_stream("objective", None).unwrap()).await;
    let paused = paused.unwrap();
    let pending = paused.run.pending.unwrap();
    let (after, done) = drain(
        runner
            .resume_stream(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: pending.id,
                    answer: json!("continue"),
                },
            )
            .unwrap(),
    )
    .await;
    assert_eq!(done.unwrap().run.status, TaskRunStatus::Completed);
    assert!(before.last().unwrap().cursor < after.first().unwrap().cursor);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
}

struct TerminalGate {
    inner: std::sync::Arc<ScopedTaskRunStore>,
    entered: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
}
#[async_trait::async_trait]
impl TaskRunStore for TerminalGate {
    // All real writes still use the production conditional backend; only the terminal receipt is held.
    async fn create(&self, snapshot: &TaskRunSnapshot) -> ai_agents_core::Result<()> {
        self.inner.create(snapshot).await
    }
    // Reads preserve exact revisions rather than synthesizing a completed result.
    async fn load(&self, run: &str) -> ai_agents_core::Result<Option<TaskRunSnapshot>> {
        self.inner.load(run).await
    }
    // The committed terminal row can win cancellation even while the live future still owns serialization.
    async fn mutate(
        &self,
        run: &str,
        mutation: &TaskRunMutation,
    ) -> ai_agents_core::Result<TaskRunSnapshot> {
        let saved = self.inner.mutate(run, mutation).await?;
        if saved.status == TaskRunStatus::Completed {
            self.entered.add_permits(1);
            self.release.acquire().await.unwrap().forget();
        }
        Ok(saved)
    }
    // Listing observes only actual persisted runs.
    async fn list(&self) -> ai_agents_core::Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    // Deletion retains ordinary conditional backend semantics.
    async fn delete(&self, run: &str, revision: u64) -> ai_agents_core::Result<()> {
        self.inner.delete(run, revision).await
    }
}

// Close must poll the owner while cancellation waits for its ledger mutex, preserving an already committed terminal winner.
#[tokio::test]
async fn ev_close_drives_terminal_ack_while_cancel_waits_for_serialization() {
    let (agent, _, store) = fixture(RecordingProvider::new(&["done"]), 1);
    let gated = std::sync::Arc::new(TerminalGate {
        inner: store,
        entered: tokio::sync::Semaphore::new(0),
        release: tokio::sync::Semaphore::new(0),
    });
    let runner = AutonomyRunner::try_new(
        agent.clone(),
        super::runner_tests::config(1),
        Default::default(),
        gated.clone(),
        "prepared-v1".into(),
    )
    .unwrap();
    let mut stream = runner.run_stream("objective", None).unwrap();
    stream.next().await.unwrap();
    gated.entered.acquire().await.unwrap().forget();
    gated.release.add_permits(1);
    let result = tokio::time::timeout(std::time::Duration::from_secs(2), stream.close())
        .await
        .expect("close stopped polling its mutex owner")
        .unwrap()
        .unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    assert!(agent.captured_task_owner().is_none());
}

// Reserved-looking domain keys and dynamic identifiers cannot opt their strings out of default event redaction.
#[test]
fn ev_untrusted_projection_redacts_keys_and_reserved_leaves() {
    let secret = "private customer identifier";
    for value in [
        json!({"metrics":{"hash":secret}}),
        json!({"redacted":{"nested":secret}}),
        json!({"length":secret}),
        json!({secret:1}),
        json!({"todo.private customer identifier":1}),
    ] {
        let projected = super::events::safe_value(&value);
        assert!(!serde_json::to_string(&projected).unwrap().contains(secret));
    }
    assert_eq!(super::events::safe_value(&json!(42)), json!(42));
    for source in [
        ai_agents_core::ToolCallSource::Skill {
            skill_id: secret.into(),
            step_index: 0,
        },
        ai_agents_core::ToolCallSource::StateAction {
            state: Some(secret.into()),
            action_index: 1,
        },
        ai_agents_core::ToolCallSource::Fallback {
            original_tool: secret.into(),
        },
    ] {
        assert!(
            !super::events::safe_source(&source)
                .to_string()
                .contains(secret)
        );
    }
}

// A cancellation loser must still await owner release after the terminal ACK, not abandon clean committed work.
#[tokio::test]
async fn ev_close_waits_for_terminal_owner_release_after_cancel_loses() {
    let (agent, _, store) = fixture(RecordingProvider::new(&["done"]), 1);
    let gated = std::sync::Arc::new(TerminalGate {
        inner: store,
        entered: tokio::sync::Semaphore::new(0),
        release: tokio::sync::Semaphore::new(0),
    });
    let runner = AutonomyRunner::try_new(
        agent.clone(),
        super::runner_tests::config(1),
        Default::default(),
        gated.clone(),
        "prepared-v1".into(),
    )
    .unwrap();
    let mut stream = runner.run_stream("objective", None).unwrap();
    stream.next().await.unwrap();
    gated.entered.acquire().await.unwrap().forget();
    let owner = agent.captured_task_owner().unwrap();
    let gate = owner.gate.clone().lock_owned().await;
    gated.release.add_permits(1);
    let mut close = Box::pin(stream.close());
    tokio::select! {
        biased;
        result = &mut close => panic!("close returned before owner cleanup: {result:?}"),
        _ = tokio::task::yield_now() => {}
    }
    drop(gate);
    let done = tokio::time::timeout(std::time::Duration::from_secs(2), close)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert!(!owner.is_abandoned());
    assert!(agent.captured_task_owner().is_none());
}

// Paused host controls publish the resulting revision without consuming the question or granting implicit work.
#[tokio::test]
async fn ev_paused_host_controls_share_revision_and_custody_contracts() {
    for control in ["limits", "objective", "stop", "cancel", "unavailable"] {
        let provider = RecordingProvider::new(&["not finished", "done"]);
        let (agent, _, store) = fixture(provider.clone(), 4);
        let mut config = super::runner_tests::config(4);
        config.defaults.premature_finish = Some(
            serde_json::from_value(json!({"action":"ask_user","on_unavailable":"stop"})).unwrap(),
        );
        let runner = AutonomyRunner::try_new(
            agent,
            config,
            Default::default(),
            store,
            "prepared-v1".into(),
        )
        .unwrap();
        let (_, paused) = drain(runner.run_stream("objective", None).unwrap()).await;
        let paused = paused.unwrap();
        let request = paused.run.pending.as_ref().unwrap().id.clone();
        let id = paused.run.key.run_id.clone();
        let revision = paused.run.revision;
        let operation = match control {
            "limits" => {
                let mut limits = paused.run.limits.clone();
                limits.max_llm_calls += 1;
                TaskStreamOperation::ReviseLimits {
                    run_id: id,
                    revision,
                    limits,
                    reason: "host revision".into(),
                }
            }
            "objective" => TaskStreamOperation::ModifyObjective {
                run_id: id,
                revision,
                objective: "new objective".into(),
                reason: "host revision".into(),
            },
            "stop" => TaskStreamOperation::Stop {
                run_id: id,
                revision,
                reason: "host stop".into(),
            },
            "cancel" => TaskStreamOperation::Cancel {
                run_id: id,
                revision,
            },
            _ => TaskStreamOperation::ControllerUnavailable {
                run_id: id,
                revision,
                request_id: request.clone(),
                reason: "delivery lost".into(),
            },
        };
        let (events, updated) = drain(
            runner
                .stream_operation(operation, TaskStreamConfig { capacity: 1 })
                .unwrap(),
        )
        .await;
        let updated = updated.unwrap();
        assert!(updated.run.revision > revision);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert!(
            events
                .iter()
                .any(|e| e.kind == TaskRunEventKind::HostControlApplied)
        );
        assert_eq!(
            updated.run.counters.llm_attempts,
            paused.run.counters.llm_attempts
        );
        if matches!(control, "limits" | "objective") {
            assert_eq!(updated.run.status, TaskRunStatus::Paused);
            let input = if control == "limits" {
                assert_eq!(updated.run.pending.as_ref().unwrap().id, request);
                TaskResumeInput::UserAnswer {
                    request_id: request,
                    answer: json!("continue"),
                }
            } else {
                TaskResumeInput::Continue
            };
            let (_, result) = drain(
                runner
                    .resume_stream(&updated.run.key.run_id, updated.run.revision, input)
                    .unwrap(),
            )
            .await;
            assert_eq!(result.unwrap().run.status, TaskRunStatus::Completed);
        } else {
            assert_ne!(updated.run.status, TaskRunStatus::Paused);
        }
    }
}

// Explicit scripted work and required stages use the same stream without automatic skill selection or a separate task loop.
#[tokio::test]
async fn ev_explicit_skill_and_runtime_adapter_preserve_required_stages() {
    let provider = RecordingProvider::new(&["sources", "done", "done"]);
    let agent = std::sync::Arc::new(crate::AgentBuilder::from_yaml("name: research\nsystem_prompt: research\nskills:\n  - id: report\n    description: report\n    trigger: report\n    autonomy: {max_llm_calls: 3}\n    steps:\n      - prompt: gather\n      - prompt: synthesize {{ steps[0].result }}\n").unwrap().llm(provider.clone()).auto_configure_features().unwrap().build().unwrap());
    let store = std::sync::Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "skill.v1".into()).unwrap(),
    );
    let mut config = super::runner_tests::config(9);
    config.defaults.lifecycle =
        Some(serde_yaml::from_str("- id: research\n- id: review\n").unwrap());
    let runner = AutonomyRunner::try_new(
        agent.clone(),
        config,
        Default::default(),
        store,
        "skill.v1".into(),
    )
    .unwrap();
    let (events, done) = drain(
        agent
            .task_operation_stream(
                &runner,
                TaskStreamOperation::Skill {
                    objective: "report".into(),
                    skill_id: "report".into(),
                    profile: None,
                },
                TaskStreamConfig { capacity: 1 },
            )
            .unwrap(),
    )
    .await;
    assert_eq!(done.unwrap().run.status, TaskRunStatus::Completed);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
    assert_eq!(
        events
            .iter()
            .filter(|e| e.kind == TaskRunEventKind::StageCompleted)
            .count(),
        2
    );
    let (other, _, _) = fixture(RecordingProvider::new(&["unused"]), 1);
    assert!(
        other
            .run_task_stream(&runner, "wrong runtime", None)
            .is_err()
    );
}

// Validator waits and fixed host bindings retain their exact attempt across event-stream reentry.
#[tokio::test]
async fn ev_validator_external_question_and_approval_continuations() {
    for mode in ["external", "question", "host", "batch"] {
        let f = super::controller_tests::fixture(mode);
        let (mut events, paused) = drain(f.runner.run_stream("objective", None).unwrap()).await;
        let mut result = paused.unwrap();
        assert_eq!(result.run.status, TaskRunStatus::Paused);
        let starts = events
            .iter()
            .filter(|e| e.kind == TaskRunEventKind::ValidationStarted)
            .count();
        for index in 0..8 {
            if result.run.status != TaskRunStatus::Paused {
                break;
            }
            let pending = result.run.pending.as_ref().unwrap();
            let input = match pending.kind {
                TaskPendingKind::ExternalCondition => {
                    TaskResumeInput::ExternalSignal(ExternalSignal {
                        request_id: pending.id.clone(),
                        event_id: format!("event-{index}"),
                        target_revision: None,
                        payload: json!(true),
                    })
                }
                TaskPendingKind::UserQuestion => TaskResumeInput::UserAnswer {
                    request_id: pending.id.clone(),
                    answer: json!({"answered":true,"selected":["yes"]}),
                },
                _ => TaskResumeInput::Approval {
                    request_id: pending.id.clone(),
                    result: ai_agents_hitl::ApprovalResult::Approved,
                },
            };
            let (next, done) = drain(
                f.runner
                    .resume_stream(&result.run.key.run_id, result.run.revision, input)
                    .unwrap(),
            )
            .await;
            events.extend(next);
            result = done.unwrap();
        }
        assert_eq!(result.run.status, TaskRunStatus::Completed, "{mode}");
        assert_eq!(
            events.last().unwrap().kind,
            TaskRunEventKind::TaskCompleted,
            "{mode}"
        );
        assert_eq!(f.provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            events
                .iter()
                .filter(|e| e.kind == TaskRunEventKind::ValidationStarted)
                .count(),
            starts
        );
        assert!(
            events
                .iter()
                .any(|e| e.kind == TaskRunEventKind::ValidationCompleted
                    && e.data["outcome"] == "pass"),
            "mode={mode}; events={events:?}"
        );
        if mode == "external" {
            assert!(
                events
                    .iter()
                    .any(|e| e.kind == TaskRunEventKind::ExternalConditionRequired)
            );
        }
    }
}

// Host-confirmed delivery loss has a real policy outcome, unlike a consumer merely pausing or dropping its display stream.
#[tokio::test]
async fn ev_external_delivery_loss_uses_existing_stop_policy() {
    let f = super::controller_tests::fixture("external");
    let (_, paused) = drain(f.runner.run_stream("objective", None).unwrap()).await;
    let paused = paused.unwrap();
    let (events, stopped) = drain(
        f.runner
            .stream_operation(
                TaskStreamOperation::ExternalUnavailable {
                    run_id: paused.run.key.run_id,
                    revision: paused.run.revision,
                    request_id: paused.run.pending.unwrap().id,
                    reason: "confirmed delivery loss".into(),
                },
                TaskStreamConfig { capacity: 1 },
            )
            .unwrap(),
    )
    .await;
    assert_eq!(stopped.unwrap().run.status, TaskRunStatus::Incomplete);
    assert_eq!(
        events.last().unwrap().kind,
        TaskRunEventKind::TaskIncomplete
    );
    assert_eq!(f.provider.calls.load(Ordering::SeqCst), 1);
}

// Missing artifact evidence remains Unknown in display and cannot become completion through a final response claim.
#[tokio::test]
async fn ev_unknown_gate_remains_unknown_in_the_public_stream() {
    let (agent, _, store) = fixture(RecordingProvider::new(&["done"]), 1);
    let mut config = super::runner_tests::config(1);
    config.defaults.completion = Some(CompletionGate::ArtifactExists("unobserved".into()));
    config.defaults.premature_finish =
        Some(serde_json::from_value(json!({"action":"fail"})).unwrap());
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        Default::default(),
        store,
        "prepared-v1".into(),
    )
    .unwrap();
    let (events, result) = drain(runner.run_stream("objective", None).unwrap()).await;
    assert_ne!(result.unwrap().run.status, TaskRunStatus::Completed);
    assert!(events.iter().any(
        |e| e.kind == TaskRunEventKind::CompletionGateChecked && e.data["outcome"] == "unknown"
    ));
}

// Cooperative channel Pending cannot discard a buffered terminal suffix after a long producer finishes.
#[tokio::test]
async fn ev_cooperative_pending_is_not_terminal_eof() {
    let mut replies = vec!["working"; 64];
    replies.push("done");
    let provider = RecordingProvider::new(&replies);
    let (agent, _, store) = fixture(provider, 65);
    let mut config = super::runner_tests::config(65);
    config.defaults.max_turns = Some(65);
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        Default::default(),
        store,
        "prepared-v1".into(),
    )
    .unwrap();
    let (events, result) = drain(
        runner
            .stream_operation(
                TaskStreamOperation::Run {
                    objective: "long task".into(),
                    profile: None,
                },
                TaskStreamConfig { capacity: 1024 },
            )
            .unwrap(),
    )
    .await;
    assert_eq!(result.unwrap().run.status, TaskRunStatus::Completed);
    assert_eq!(events.last().unwrap().kind, TaskRunEventKind::TaskCompleted);
    assert_eq!(
        events
            .iter()
            .filter(|e| e.kind == TaskRunEventKind::CompletionGateChecked)
            .count(),
        65
    );
}

// Saturation suspends production with bounded queue storage and does not launch another task in the background.
#[tokio::test]
async fn ev_slow_consumer_is_pull_driven_and_bounded() {
    let provider = RecordingProvider::new(&["working", "done"]);
    let (_, runner, _) = fixture(provider.clone(), 2);
    let mut stream = runner
        .stream_operation(
            TaskStreamOperation::Run {
                objective: "objective".into(),
                profile: None,
            },
            TaskStreamConfig { capacity: 1 },
        )
        .unwrap();
    let first = stream.next().await.unwrap();
    let calls = provider.calls.load(Ordering::SeqCst);
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), calls);
    let mut events = vec![first];
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    assert_eq!(
        stream.take_result().unwrap().unwrap().run.status,
        TaskRunStatus::Completed
    );
    assert!(events.len() > 1);
}

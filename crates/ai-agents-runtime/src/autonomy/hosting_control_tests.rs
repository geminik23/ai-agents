//! Real host-control tests isolate provider state and never need a terminal, credentials or external effects.

use super::*;
use crate::autonomy::runner_tests::RecordingProvider;
use crate::{AgentBuilder, RuntimeAgent};
use async_trait::async_trait;
use std::sync::{Arc, atomic::Ordering};

const TASK: &str = "name: HostControls\nsystem_prompt: Helpful\nautonomy:\n  enabled: true\n  max_turns: 4\n  max_llm_calls: 4\n  completion: {response_contains: done}\n  premature_finish: {action: ask_user, on_unavailable: stop}\n";

// Each fixture uses actual controller/checkpoint boundaries and an independent offline provider queue.
fn agent(provider: Arc<RecordingProvider>) -> RuntimeAgent {
    AgentBuilder::from_yaml(TASK)
        .unwrap()
        .llm(provider)
        .build()
        .unwrap()
}

#[tokio::test]
// Local detail projection preserves readable facts without exporting recovery data or consuming a decision.
async fn host_details_are_bounded_local_only_and_leave_authority_unchanged() {
    let provider = RecordingProvider::new(&["draft", "done"]);
    let agent = agent(provider.clone());
    let paused = agent
        .run_task(&"PRIVATE_OBJECTIVE_한글".repeat(2000), None)
        .await
        .unwrap();
    let runner = agent.task_runner().unwrap();
    let details = runner
        .task_host_details(&paused.run.key.run_id)
        .await
        .unwrap();
    assert_eq!(details.revision, paused.run.revision);
    assert!(details.truncated);
    assert!(details.lines.iter().map(String::len).sum::<usize>() <= MAX_TASK_DISPLAY_BYTES);
    assert!(details.lines.iter().all(|line| line.len() <= 2048));
    assert!(
        details
            .lines
            .iter()
            .any(|line| line.contains("PRIVATE_OBJECTIVE"))
    );
    assert!(!format!("{details:?}").contains("PRIVATE_OBJECTIVE"));
    for forbidden in [
        "owner_token",
        "runtime.controller",
        "reviewed_action",
        "native_history",
        "fingerprints",
    ] {
        assert!(!details.lines.iter().any(|line| line.contains(forbidden)));
    }
    assert_eq!(
        runner
            .task_display(&paused.run.key.run_id)
            .await
            .unwrap()
            .summary
            .revision,
        paused.run.revision
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    runner
        .cancel_paused(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap();
}

#[tokio::test]
// A tool-free task remains valid without todos or file artifacts, and observation labels do not claim fresh eligibility.
async fn host_details_tool_free_progress_remains_informational() {
    let provider = RecordingProvider::new(&["draft"]);
    let agent = agent(provider.clone());
    let paused = agent.run_task("Research", None).await.unwrap();
    let runner = agent.task_runner().unwrap();
    let details = runner
        .task_host_details(&paused.run.key.run_id)
        .await
        .unwrap();
    assert!(
        details
            .lines
            .iter()
            .any(|line| line == "Todos: not configured")
    );
    assert!(details.lines.iter().any(|line| line.contains("historical")));
    assert!(
        !details
            .lines
            .iter()
            .any(|line| line.starts_with("Artifact reference:"))
    );
    runner
        .cancel_paused(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap();
}

#[tokio::test]
// Reads do not consume an exchange, and exact sensitive prompt data never appears in display serialization or diagnostic formatting.
async fn host_controls_display_and_typed_question_preserve_pending_without_raw_exports() {
    let provider = RecordingProvider::new(&["DRAFT_SECRET", "done"]);
    let agent = agent(provider.clone());
    let paused = agent.run_task("OBJECTIVE_SECRET", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    let id = &paused.run.key.run_id;
    let view = agent.task_display(id).await.unwrap();
    assert_eq!(view.summary.revision, paused.run.revision);
    assert!(view.locally_retained);
    assert_eq!(view.last_gate, Some(GateOutcome::Fail));
    assert_eq!(view.counters.llm_attempts, 1);
    assert!(view.artifacts.is_none());
    assert_eq!(view.restoration_review_required, Some(false));
    let encoded = serde_json::to_string(&view).unwrap();
    for secret in [
        "OBJECTIVE_SECRET",
        "DRAFT_SECRET",
        "owner_token",
        "runtime.validation",
        "reviewed_action",
    ] {
        assert!(!encoded.contains(secret), "unexpected raw field: {secret}");
    }
    assert!(encoded.len() <= MAX_TASK_DISPLAY_BYTES);
    let interaction = agent
        .task_interaction(id, paused.run.revision)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        interaction.request_id,
        paused.run.pending.as_ref().unwrap().id
    );
    assert!(matches!(interaction.prompt, TaskHostPrompt::Question(_)));
    assert!(!format!("{interaction:?}").contains("SECRET"));
    assert!(
        agent
            .task_interaction(id, paused.run.revision + 1)
            .await
            .is_err()
    );
    let again = agent.task_display(id).await.unwrap();
    assert_eq!(again.summary.revision, paused.run.revision);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    let done = agent
        .resume_task(
            id,
            interaction.revision,
            TaskResumeInput::UserAnswer {
                request_id: interaction.request_id,
                answer: json!("Continue"),
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert!(
        agent
            .task_interaction(id, done.run.revision)
            .await
            .unwrap()
            .is_none()
    );
    assert!(!agent.task_display(id).await.unwrap().locally_retained);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
// A retained safe-pause cancellation acknowledges cleanup only after its actual operation returned and released custody.
async fn host_controls_paused_cancel_acknowledges_without_executing_pending_work() {
    let provider = RecordingProvider::new(&["draft", "must not execute"]);
    let agent = agent(provider.clone());
    let paused = agent.run_task("Report", None).await.unwrap();
    let outcome = agent
        .cancel_task(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap();
    assert_eq!(outcome.disposition, CancellationDisposition::Acknowledged);
    assert_eq!(outcome.run.status, TaskRunStatus::Cancelled);
    assert_eq!(outcome.locally_retained, Some(false));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    let terminal = agent
        .cancel_task(&outcome.run.key.run_id, outcome.run.revision)
        .await
        .unwrap();
    assert_eq!(
        terminal.disposition,
        CancellationDisposition::AlreadyTerminal
    );
    assert_eq!(terminal.run.revision, outcome.run.revision);
    assert!(
        !agent
            .task_display(&outcome.run.key.run_id)
            .await
            .unwrap()
            .locally_retained
    );
}

#[tokio::test]
// A remote caller can record a request with storage alone, but cannot acknowledge or run the live process's cleanup.
async fn host_controls_store_only_cancel_is_requested_until_retained_cleanup() {
    let provider = RecordingProvider::new(&["draft"]);
    let agent = agent(provider.clone());
    let paused = agent.run_task("Report", None).await.unwrap();
    let runner = agent.task_runner().unwrap();
    let receipt = CancellationOutcome::request(
        runner.task_store().as_ref(),
        &paused.run.key.run_id,
        paused.run.revision,
    )
    .await
    .unwrap();
    assert_eq!(receipt.disposition, CancellationDisposition::Requested);
    assert_eq!(receipt.locally_retained, None);
    assert_eq!(receipt.run.status, TaskRunStatus::Paused);
    assert!(receipt.run.cancel_requested);
    assert!(agent.captured_task_owner().is_some());
    let duplicate = CancellationOutcome::request(
        runner.task_store().as_ref(),
        &receipt.run.key.run_id,
        receipt.run.revision,
    )
    .await
    .unwrap();
    assert_eq!(duplicate.run.revision, receipt.run.revision);
    assert_eq!(duplicate.disposition, CancellationDisposition::Requested);
    let ack = agent
        .cancel_task(&receipt.run.key.run_id, receipt.run.revision)
        .await
        .unwrap();
    assert_eq!(ack.disposition, CancellationDisposition::Acknowledged);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
// Cancellation during an admitted non-preemptible provider wait is a receipt, and uncertainty still retains protection after foreground return.
async fn host_controls_running_cancel_does_not_acknowledge_uncertain_provider_work() {
    let provider = RecordingProvider::gated();
    let agent = agent(provider.clone());
    let runner = agent.task_runner().unwrap();
    let operation = agent.run_task("Work", None);
    tokio::pin!(operation);
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        tokio::select! {
            _ = provider.entered.as_ref().unwrap().acquire() => {},
            result = &mut operation => panic!("provider did not park: {result:?}"),
        }
    })
    .await
    .unwrap();
    let running = runner.task_store().list().await.unwrap().pop().unwrap();
    assert!(
        agent
            .cancel_task(&running.key.run_id, running.revision + 1)
            .await
            .is_err()
    );
    assert!(
        !runner
            .task_store()
            .load(&running.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .cancel_requested
    );
    let requested = agent
        .cancel_task(&running.key.run_id, running.revision)
        .await
        .unwrap();
    assert_eq!(requested.disposition, CancellationDisposition::Requested);
    assert_eq!(requested.run.status, TaskRunStatus::Running);
    assert!(requested.run.owned);
    assert!(agent.captured_task_owner().is_some());
    assert!(agent.set_context("cannot_mutate", json!(true)).is_err());
    provider.release.as_ref().unwrap().add_permits(1);
    let done = operation.await.unwrap();
    assert_eq!(done.run.status, TaskRunStatus::RecoveryRequired);
    assert!(agent.captured_task_owner().is_some());
    let protected = agent.task_display(&done.run.key.run_id).await.unwrap();
    assert!(protected.locally_retained);
    assert_eq!(protected.summary.status, TaskRunStatus::RecoveryRequired);
    assert!(protected.restoration_review_required.is_none());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

struct CorruptReceiptStore(Arc<dyn TaskRunStore>);
#[async_trait]
impl TaskRunStore for CorruptReceiptStore {
    // All actual persistence remains with the owning conditional store; only its cancellation response is corrupted.
    async fn create(&self, value: &TaskRunSnapshot) -> Result<()> {
        self.0.create(value).await
    }
    // Reads expose the real state after the possibly committed request.
    async fn load(&self, id: &str) -> Result<Option<TaskRunSnapshot>> {
        self.0.load(id).await
    }
    // Corrupting one acknowledgement must not create a successful cancellation receipt.
    async fn mutate(&self, id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot> {
        let mut saved = self.0.mutate(id, mutation).await?;
        if matches!(mutation, TaskRunMutation::RequestCancel { .. }) {
            saved.cancel_requested = false;
        }
        Ok(saved)
    }
    // Safe metadata uses the real delegate's scoped behavior.
    async fn list(&self) -> Result<Vec<TaskRunSummary>> {
        self.0.list().await
    }
    // Deletion remains conditional and is never used by this response fault.
    async fn delete(&self, id: &str, revision: u64) -> Result<()> {
        self.0.delete(id, revision).await
    }
}

#[tokio::test]
// A committed flag with a mismatching ACK reports uncertainty rather than success and leaves the actual pause protected.
async fn host_controls_corrupt_cancel_receipt_is_rejected_without_replay() {
    let provider = RecordingProvider::new(&["draft"]);
    let agent = agent(provider.clone());
    let paused = agent.run_task("Work", None).await.unwrap();
    let runner = agent.task_runner().unwrap();
    let fault = CorruptReceiptStore(runner.task_store().clone());
    assert!(
        CancellationOutcome::request(&fault, &paused.run.key.run_id, paused.run.revision)
            .await
            .is_err()
    );
    let actual = runner
        .task_store()
        .load(&paused.run.key.run_id)
        .await
        .unwrap()
        .unwrap();
    assert!(actual.cancel_requested);
    assert_eq!(actual.status, TaskRunStatus::Paused);
    assert!(agent.captured_task_owner().is_some());
    let ack = agent
        .cancel_task(&actual.key.run_id, actual.revision)
        .await
        .unwrap();
    assert_eq!(ack.disposition, CancellationDisposition::Acknowledged);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
// A local cancellation ACK mismatch preserves actual live custody instead of signalling successful cleanup or replaying the provider request.
async fn host_controls_live_cancel_ack_mismatch_retains_recovery_protection() {
    let provider = RecordingProvider::gated();
    let store: Arc<dyn TaskRunStore> = Arc::new(
        ScopedTaskRunStore::in_memory("HostControls".into(), None, "host-controls.v1".into())
            .unwrap(),
    );
    let faulty: Arc<dyn TaskRunStore> = Arc::new(CorruptReceiptStore(store.clone()));
    let agent = AgentBuilder::from_yaml(TASK)
        .unwrap()
        .llm(provider.clone())
        .automatic_task_options(AutomaticTaskOptions {
            store: Some(faulty),
            config_identity: Some("host-controls.v1".into()),
            ..Default::default()
        })
        .build()
        .unwrap();
    let runner = agent.task_runner().unwrap();
    let work = agent.run_task("Work", None);
    tokio::pin!(work);
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        tokio::select! {
            _ = provider.entered.as_ref().unwrap().acquire() => {},
            result = &mut work => panic!("provider did not park: {result:?}"),
        }
    })
    .await
    .unwrap();
    let running = store.list().await.unwrap().pop().unwrap();
    assert!(
        agent
            .cancel_task(&running.key.run_id, running.revision)
            .await
            .is_err()
    );
    assert!(
        store
            .load(&running.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .cancel_requested
    );
    assert!(agent.captured_task_owner().is_some());
    provider.release.as_ref().unwrap().add_permits(1);
    let protected = work.await.unwrap();
    assert_eq!(protected.run.status, TaskRunStatus::RecoveryRequired);
    assert!(agent.captured_task_owner().is_some());
    assert!(agent.set_context("cannot_reuse", json!(true)).is_err());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert_eq!(runner.task_store().list().await.unwrap().len(), 1);
}

#[tokio::test]
// New convenience cancellation and stop must publish the exact terminal child outcome before an ordinary parent resumes.
async fn host_controls_automatic_controller_cancel_and_stop_wake_composing_parent() {
    use crate::orchestration::{PipelineStage, pipeline};
    use crate::spawner::{AgentRegistry, SpawnedAgent};
    for stop in [false, true] {
        let yaml = "name: ControllerChild\nsystem_prompt: Helpful\nstates:\n  initial: work\n  states:\n    work:\n      autonomy:\n        enabled: true\n        mode: state_until_exit\n        completion: {response_contains: done}\n        premature_finish: {action: ask_user, on_unavailable: stop}\n";
        let provider = RecordingProvider::new(&["draft", "must not execute"]);
        let runtime = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(provider.clone())
            .build()
            .unwrap();
        let registry = AgentRegistry::new();
        registry
            .register(SpawnedAgent::from_runtime(
                "child".into(),
                runtime,
                crate::spec::AgentSpec::from_yaml_strict(yaml).unwrap(),
            ))
            .await
            .unwrap();
        let child = registry.get("child").unwrap();
        let stages = vec![PipelineStage::id("child")];
        let mut parent = Box::pin(pipeline(
            &registry,
            "Report",
            &stages,
            Some(30_000),
            None,
            None,
        ));
        let paused = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            loop {
                tokio::select! {
                    result = &mut parent => panic!("paused child returned as success: {result:?}"),
                    _ = tokio::task::yield_now() => if let Some(result) = child.automatic_task_result() { break result; },
                }
            }
        }).await.unwrap();
        let expected = if stop {
            child
                .task_operation(TaskStreamOperation::Stop {
                    run_id: paused.run.key.run_id.clone(),
                    revision: paused.run.revision,
                    reason: "host stop".into(),
                })
                .await
                .unwrap()
                .run
                .status
        } else {
            child
                .cancel_task(&paused.run.key.run_id, paused.run.revision)
                .await
                .unwrap()
                .run
                .status
        };
        assert_eq!(
            expected,
            if stop {
                TaskRunStatus::Incomplete
            } else {
                TaskRunStatus::Cancelled
            }
        );
        assert_eq!(child.automatic_task_result().unwrap().run.status, expected);
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(10), parent)
                .await
                .unwrap()
                .is_err()
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
// The final required stage still needs a real review prompt even though no destination stage remains.
async fn host_controls_final_stage_review_projects_and_resumes_without_replay() {
    let yaml = "name: FinalReview\nsystem_prompt: Helpful\nautonomy:\n  enabled: true\n  max_turns: 3\n  max_llm_calls: 3\n  lifecycle:\n    - id: review\n      instruction: Review the result\n      completion: {response_contains: done}\n  completion: {response_contains: done}\n  supervision: {stage_changes: require_approval}\n";
    let provider = RecordingProvider::new(&["done"]);
    let agent = AgentBuilder::from_yaml(yaml)
        .unwrap()
        .llm(provider.clone())
        .build()
        .unwrap();
    let paused = agent.run_task("Report", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    let interaction = agent
        .task_interaction(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        interaction.prompt,
        TaskHostPrompt::StageReview {
            next_stage: None,
            finishes_lifecycle: true,
            ..
        }
    ));
    let done = agent
        .resume_task(
            &paused.run.key.run_id,
            interaction.revision,
            TaskResumeInput::Approval {
                request_id: interaction.request_id,
                result: ai_agents_hitl::ApprovalResult::approved(),
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

#[cfg(feature = "sqlite")]
#[tokio::test]
// SQLite uses the same host inspection and exact-revision cleanup contract; a saved checkpoint alone does not claim restart preparation.
async fn host_controls_sqlite_views_and_cleanup_use_the_same_scoped_contract() {
    let directory =
        std::env::temp_dir().join(format!("ai-agents-host-controls-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&directory).unwrap();
    let storage = Arc::new(
        ai_agents_storage::SqliteStorage::new(directory.join("tasks.sqlite").to_str().unwrap())
            .await
            .unwrap(),
    );
    let store: Arc<dyn TaskRunStore> = Arc::new(
        ScopedTaskRunStore::new(
            storage.clone(),
            "HostControls".into(),
            None,
            "sqlite-host.v1".into(),
        )
        .unwrap(),
    );
    let provider = RecordingProvider::new(&["draft"]);
    let agent = AgentBuilder::from_yaml(TASK)
        .unwrap()
        .llm(provider.clone())
        .automatic_task_options(AutomaticTaskOptions {
            store: Some(store.clone()),
            config_identity: Some("sqlite-host.v1".into()),
            ..Default::default()
        })
        .build()
        .unwrap();
    let paused = agent.run_task("Report", None).await.unwrap();
    let display = agent.task_display(&paused.run.key.run_id).await.unwrap();
    assert_eq!(display.summary.revision, paused.run.revision);
    assert!(display.locally_retained);
    let prompt = agent
        .task_interaction(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(prompt.request_id, paused.run.pending.as_ref().unwrap().id);
    let receipt =
        CancellationOutcome::request(store.as_ref(), &paused.run.key.run_id, paused.run.revision)
            .await
            .unwrap();
    assert_eq!(receipt.disposition, CancellationDisposition::Requested);
    let done = agent
        .cancel_task(&receipt.run.key.run_id, receipt.run.revision)
        .await
        .unwrap();
    assert_eq!(done.disposition, CancellationDisposition::Acknowledged);
    let terminal = agent.task_display(&done.run.key.run_id).await.unwrap();
    assert_eq!(terminal.summary.status, TaskRunStatus::Cancelled);
    assert!(!terminal.locally_retained);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    drop(agent);
    drop(store);
    storage.close().await;
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
// Projection unwraps framework group envelopes without losing the outer response identity or copying a private batch cursor.
fn host_controls_nested_question_projection_and_diagnostic_redaction() {
    let batch = BatchApproval {
        call_id: "call".into(),
        request_id: "leaf".into(),
        trigger: json!({}),
        context: json!({"PRIVATE_ARGUMENT":"secret"}),
        message: "APPROVAL_SECRET".into(),
        timeout_millis: None,
        expires_at: None,
        question: Some(
            serde_json::to_value(ai_agents_tools::QuestionRequest {
                question: "QUESTION_SECRET".into(),
                options: vec!["one".into(), "two".into()],
                multi_select: true,
                allow_other: true,
                ..Default::default()
            })
            .unwrap(),
        ),
    };
    let leaf = TaskPendingRequest {
        id: "leaf".into(),
        issued_revision: 2,
        kind: TaskPendingKind::UserQuestion,
        reviewed_action: serde_json::to_value(batch).unwrap(),
    };
    let outer = TaskPendingRequest {
        id: "outer".into(),
        issued_revision: 4,
        kind: TaskPendingKind::UserQuestion,
        reviewed_action: json!({"child_operation":"child", "request":leaf}),
    };
    let projected = prompt(&outer, &AutonomyProfile::default()).unwrap();
    assert_eq!(format!("{projected:?}"), "Question");
    match projected {
        TaskHostPrompt::Question(question) => {
            assert_eq!(question.question, "QUESTION_SECRET");
            assert!(question.multi_select);
            assert_eq!(question.options.len(), 2);
        }
        _ => panic!("expected typed question"),
    }
    assert_eq!(outer.id, "outer");
}

#[test]
// Stage, plan and capacity previews expose only selected public review fields, never their executable controller checkpoints.
fn host_controls_controller_proposals_have_distinct_typed_prompts() {
    use crate::autonomy::controller::ControllerLimit;
    let profile: AutonomyProfile = serde_yaml::from_str("enabled: true\nlifecycle:\n  - id: review\n    instruction: Review the result\ncompletion: {response_not_empty: true}\n").unwrap();
    let lifecycle = LifecycleState::new(&profile, false).unwrap();
    let review = |proposal| TaskPendingRequest {
        id: "request".into(),
        issued_revision: 1,
        kind: TaskPendingKind::Approval,
        reviewed_action: serde_json::to_value(ControllerReview {
            id: "request".into(),
            proposal,
            expires_at: None,
            state_generation: None,
            control_binding: (0, 0),
        })
        .unwrap(),
    };
    assert!(matches!(
        prompt(&review(ControllerProposal::Continue), &profile).unwrap(),
        TaskHostPrompt::Continue
    ));
    assert!(
        matches!(prompt(&review(ControllerProposal::Stage { lifecycle }), &profile).unwrap(), TaskHostPrompt::StageReview { next_stage: Some(stage), finishes_lifecycle: false, .. } if stage == "review")
    );
    assert!(matches!(
        prompt(
            &review(ControllerProposal::Plan {
                instruction: "PLAN_SECRET".into()
            }),
            &profile
        )
        .unwrap(),
        TaskHostPrompt::PlanReview { .. }
    ));
    let limits = TaskRunLimits {
        max_turns: 1,
        max_active_time_seconds: 10,
        max_wall_time_seconds: None,
        max_llm_calls: 1,
        max_tool_calls: 1,
        max_command_calls: 1,
        max_micro_usd: None,
        max_declared_write_paths: None,
    };
    let capacity = prompt(
        &review(ControllerProposal::Capacity {
            limit: Box::new(ControllerLimit {
                reason: "turn_capacity".into(),
                original: limits,
                resume_turn: false,
                skill: None,
                model: None,
                tool: None,
            }),
        }),
        &profile,
    )
    .unwrap();
    assert!(
        matches!(capacity, TaskHostPrompt::Capacity { original_limits, .. } if original_limits.max_turns == 1)
    );
}

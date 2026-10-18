use super::continuation_test_support as support;
use super::runner_tests::RecordingProvider;
use super::*;
#[path = "host_control_failure_tests.rs"]
mod failure_tests;
use crate::AgentBuilder;
use ai_agents_hitl::ApprovalResult;
use serde_json::json;
use std::sync::{Arc, atomic::Ordering};

/// Plan review retains one actual planner result and its run-wide intervention accounting.
#[tokio::test]
async fn host_control_plan_review_and_supersession_preserve_planner_and_intervention_counts() {
    for revise in [false, true] {
        let plan = r#"{"steps":[{"id":"one","description":"Revise plan","action_type":"think","action_target":"reason","dependencies":[]}]}"#;
        let provider = RecordingProvider::new(&["working", plan, "done"]);
        let agent = Arc::new(
            AgentBuilder::new()
                .system_prompt("test")
                .llm(provider.clone())
                .build()
                .unwrap(),
        );
        let store = support::store(agent.info().id, "plan.v1", false).await;
        let mut config = super::runner_tests::config(8);
        config.defaults.supervision = Some(SupervisionConfig {
            stage_changes: None,
            plan_changes: Some(ReviewPolicy::RequireApproval),
        });
        config.defaults.progress = Some(serde_yaml::from_str("stagnation: {max_cycles_without_progress: 1, action: replan, max_replans: 1, on_exhausted: stop}").unwrap());
        let runner = AutonomyRunner::try_new(
            agent,
            config,
            Default::default(),
            store.clone(),
            "plan.v1".into(),
        )
        .unwrap();
        let first = runner.run("objective", None).await.unwrap();
        assert_eq!(
            first.run.status,
            TaskRunStatus::Paused,
            "reason={:?} evidence={:?} calls={}",
            first.run.stop_reason,
            first.run.evidence,
            provider.calls.load(Ordering::SeqCst)
        );
        assert_eq!(first.run.progress.replans, 1);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        let done = if revise {
            let revised = runner
                .modify_objective_paused(
                    &first.run.key.run_id,
                    first.run.revision,
                    "new",
                    "supersede proposal",
                )
                .await
                .unwrap();
            let state = controller_checkpoint(&payload(&store, &revised.run).await)
                .unwrap()
                .unwrap();
            assert_eq!(state.progress.replans, 1);
            assert_eq!(
                state.progress.sequence,
                first.run.progress.observation_sequence
            );
            runner
                .resume(
                    &revised.run.key.run_id,
                    revised.run.revision,
                    TaskResumeInput::Continue,
                )
                .await
                .unwrap()
        } else {
            approve(&runner, &first.run).await.unwrap()
        };
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(done.run.progress.replans, 1);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
    }
}

/// Superseded validator waits and approvals leave an auditable retired attempt, then admit a fresh check.
#[tokio::test]
async fn host_control_validation_supersession_retires_attempt_and_revalidates_required_checks() {
    for sqlite in support::backends() {
        for mode in ["external", "approval"] {
            let mut f = super::controller_tests::fixture(mode);
            f.config.defaults.max_llm_calls = Some(4);
            f.config.defaults.max_tool_calls = Some(4);
            f.config.defaults.completion = Some(CompletionGate::ValidationPassed(true));
            f.store = support::store(f.agent.info().id, "controller.v1", *sqlite).await;
            f.runner = AutonomyRunner::try_new(
                f.agent.clone(),
                f.config.clone(),
                Default::default(),
                f.store.clone(),
                "controller.v1".into(),
            )
            .unwrap();
            let initial = f.runner.run("old", None).await.unwrap();
            let old_id = initial.run.pending.as_ref().unwrap().id.clone();
            let revised = f
                .runner
                .modify_objective_paused(
                    &initial.run.key.run_id,
                    initial.run.revision,
                    "new",
                    "revalidate",
                )
                .await
                .unwrap();
            let fresh = f
                .runner
                .resume(
                    &revised.run.key.run_id,
                    revised.run.revision,
                    TaskResumeInput::Continue,
                )
                .await
                .unwrap();
            assert_eq!(
                fresh.run.status,
                TaskRunStatus::Paused,
                "{:?}",
                fresh.run.stop_reason
            );
            assert_ne!(old_id, fresh.run.pending.as_ref().unwrap().id);
            let input = if mode == "external" {
                assert!(
                    f.runner
                        .resume(
                            &fresh.run.key.run_id,
                            fresh.run.revision,
                            TaskResumeInput::ExternalSignal(ExternalSignal {
                                request_id: old_id,
                                event_id: "old".into(),
                                target_revision: None,
                                payload: json!(true)
                            })
                        )
                        .await
                        .is_err()
                );
                TaskResumeInput::ExternalSignal(ExternalSignal {
                    request_id: fresh.run.pending.as_ref().unwrap().id.clone(),
                    event_id: "fresh".into(),
                    target_revision: None,
                    payload: json!(true),
                })
            } else {
                assert!(
                    f.runner
                        .resume(
                            &fresh.run.key.run_id,
                            fresh.run.revision,
                            TaskResumeInput::Approval {
                                request_id: old_id,
                                result: ApprovalResult::Approved
                            }
                        )
                        .await
                        .is_err()
                );
                TaskResumeInput::Approval {
                    request_id: fresh.run.pending.as_ref().unwrap().id.clone(),
                    result: ApprovalResult::Approved,
                }
            };
            let done = f
                .runner
                .resume(&fresh.run.key.run_id, fresh.run.revision, input)
                .await
                .unwrap();
            assert_eq!(
                done.run.status,
                TaskRunStatus::Completed,
                "{:?}",
                done.run.stop_reason
            );
            assert_eq!(done.run.counters.turns, 2);
            let audit = payload(&f.store, &done.run).await;
            assert!(audit.evidence.records.iter().any(|record| {
                record.origin == "host.continuation.supersede"
                    && record.data["validation"]
                        .as_array()
                        .is_some_and(|attempts| {
                            attempts.iter().any(|attempt| {
                                attempt["id"] == "check"
                                    && attempt["driver"]["identity"]["objective_revision"] == 0
                            })
                        })
            }));
            assert_eq!(
                f.calls.load(Ordering::SeqCst),
                usize::from(mode != "external")
            );
        }
    }
}

/// Ordinary task reasoning freezes its executable plan before adoption, independently of controller replanning.
#[tokio::test]
async fn host_control_initial_reasoning_plan_review_resumes_without_regeneration() {
    for sqlite in support::backends() {
        for reject in [false, true] {
            let plan = r#"{"steps":[{"id":"one","description":"Answer","action_type":"think","action_target":"reason","dependencies":[]}]}"#;
            let provider = RecordingProvider::new(&[plan, "done"]);
            let agent = Arc::new(
                AgentBuilder::from_yaml(
                    "name: PlanReview\nsystem_prompt: test\nreasoning:\n  mode: plan_and_execute\n",
                )
                .unwrap()
                .llm(provider.clone())
                .auto_configure_features()
                .unwrap()
                .build()
                .unwrap(),
            );
            let store = support::store(agent.info().id, "plan-review.v1", *sqlite).await;
            let mut config = super::runner_tests::config(8);
            config.defaults.supervision = Some(SupervisionConfig {
                stage_changes: None,
                plan_changes: Some(ReviewPolicy::RequireApproval),
            });
            let runner = AutonomyRunner::try_new(
                agent.clone(),
                config,
                Default::default(),
                store.clone(),
                "plan-review.v1".into(),
            )
            .unwrap();
            let paused = runner.run("objective", None).await.unwrap();
            assert_eq!(
                paused.run.status,
                TaskRunStatus::Paused,
                "{:?}",
                paused.run.stop_reason
            );
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            let done = runner
                .resume(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    TaskResumeInput::Approval {
                        request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                        result: if reject {
                            ApprovalResult::Rejected { reason: None }
                        } else {
                            ApprovalResult::Approved
                        },
                    },
                )
                .await
                .unwrap();
            assert_eq!(
                done.run.status,
                if reject {
                    TaskRunStatus::Incomplete
                } else {
                    TaskRunStatus::Completed
                },
                "{:?}",
                done.run.stop_reason
            );
            assert_eq!(
                provider.calls.load(Ordering::SeqCst),
                if reject { 1 } else { 2 }
            );
            assert_eq!(done.run.counters.turns, 1);
            assert!(agent.set_context("released", json!(true)).is_ok());
        }
    }
}

/// Validation-fix rewinds are reviewed before removing existing required-stage credit.
#[tokio::test]
async fn host_control_stage_fix_rewind_requires_review_without_repeating_validation_turn() {
    let provider = RecordingProvider::new(&["work", "bad", "work", "verified done"]);
    let agent = Arc::new(
        AgentBuilder::new()
            .system_prompt("test")
            .llm(provider.clone())
            .build()
            .unwrap(),
    );
    let store = support::store(agent.info().id, "fix.v1", false).await;
    let mut config = super::runner_tests::config(16);
    config.defaults.max_turns = Some(8);
    config.defaults.lifecycle = Some(serde_json::from_value(json!([
        {"id":"work"}, {"id":"verify","completion":{"response_contains":"verified"},"retry_on_failure":true,"on_failure_stage":"work","max_fix_cycles":1}
    ])).unwrap());
    config.defaults.supervision = Some(SupervisionConfig {
        stage_changes: Some(ReviewPolicy::RequireApproval),
        plan_changes: None,
    });
    let runner =
        AutonomyRunner::try_new(agent, config, Default::default(), store, "fix.v1".into()).unwrap();
    let work = runner.run("objective", None).await.unwrap();
    let rewind = approve(&runner, &work.run).await.unwrap();
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        rewind.run.pending.as_ref().unwrap().reviewed_action["proposal"]["lifecycle"]["current_stage"],
        0
    );
    assert_eq!(
        rewind.run.pending.as_ref().unwrap().reviewed_action["proposal"]["lifecycle"]["fix_cycles"],
        1
    );
    let work_again = approve(&runner, &rewind.run).await.unwrap();
    assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
    let verified = approve(&runner, &work_again.run).await.unwrap();
    let done = approve(&runner, &verified.run).await.unwrap();
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 4);
}

struct SeedTodo;
#[async_trait::async_trait]
impl ai_agents_core::Tool for SeedTodo {
    fn id(&self) -> &str {
        "seed"
    }
    fn name(&self) -> &str {
        "seed"
    }
    fn description(&self) -> &str {
        "Seed the actual task-bound canonical list"
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({"type":"object"})
    }
    fn safety_metadata(&self) -> ai_agents_core::ToolSafetyMetadata {
        ai_agents_core::ToolSafetyMetadata {
            read_only: true,
            ..Default::default()
        }
    }
    async fn execute(
        &self,
        _: serde_json::Value,
        _: ai_agents_core::ToolExecutionContext,
    ) -> ai_agents_core::ToolResult {
        let store = ai_agents_tools::current_task_todo_store().unwrap().unwrap();
        assert!(store.try_set(vec![ai_agents_tools::TodoItem {
            id: "old".into(),
            content: "Old completed work".into(),
            active_form: None,
            status: ai_agents_tools::TodoStatus::Completed
        }]));
        ai_agents_core::ToolResult::ok("seeded")
    }
}

/// Real run-bound todos move atomically to a fresh objective binding while retaining the old list for audit.
#[tokio::test]
async fn host_control_canonical_todo_handoff_retires_stale_handles_and_preserves_growth_limit() {
    for sqlite in support::backends() {
        let provider = RecordingProvider::new(&["done", "done"]);
        let yaml = "name: Todos\nsystem_prompt: test\ntools: [seed,ask_user]\nstates:\n  initial: active\n  states:\n    active:\n      on_exit:\n        - set_context: {allow: false}\n        - tool: seed\n        - tool: ask_user\n          args: {question: Confirm, options: [yes], allow_other: false}\n      transitions:\n        - to: ready\n          guard:\n            context:\n              allow:\n                eq: true\n    ready:\n      regenerate_on_enter: false\n";
        let agent = Arc::new(
            AgentBuilder::from_yaml(yaml)
                .unwrap()
                .llm(provider)
                .auto_configure_features()
                .unwrap()
                .tool(Arc::new(SeedTodo))
                .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
                    Arc::new(parking_lot::RwLock::new(None)),
                )))
                .build()
                .unwrap(),
        );
        agent.set_context("allow", json!(true)).unwrap();
        let store = support::store(agent.info().id, "todos.v1", *sqlite).await;
        let mut config = super::runner_tests::config(8);
        config.defaults.hitl = Some(AutonomyHitlConfig {
            on_user_question: Some(InteractionAction::PauseRun),
            on_approval_required: Some(InteractionAction::PauseRun),
        });
        config.defaults.progress = Some(ProgressConfig {
            max_open_todos: Some(1),
            ..Default::default()
        });
        let runner = AutonomyRunner::try_new(
            agent.clone(),
            config,
            Default::default(),
            store.clone(),
            "todos.v1".into(),
        )
        .unwrap();
        let first = runner.run("old", None).await.unwrap();
        let before = payload(&store, &first.run).await.todos.unwrap();
        assert_eq!(before.items.len(), 1);
        let revised = runner
            .modify_objective_paused(
                &first.run.key.run_id,
                first.run.revision,
                "new",
                "retire old todos",
            )
            .await
            .unwrap();
        let after = payload(&store, &revised.run).await;
        let next = after.todos.unwrap();
        let canonical = agent.todo_store();
        assert!(canonical.list_for_run(&before.binding).is_none());
        assert!(canonical.list_for_run(&next.binding).unwrap().is_empty());
        assert_ne!(before.binding.token, next.binding.token);
        assert_eq!(next.binding.objective_revision, 1);
        assert!(
            after
                .evidence
                .records
                .iter()
                .any(|record| record.data["action"]["retired_todos"]["items"][0]["id"] == "old")
        );
        let stale = ai_agents_tools::scope_task_todos(canonical.clone(), before.binding, async {
            ai_agents_tools::current_task_todo_store()
        })
        .await;
        assert!(stale.is_err());
        let items = ["one", "two"]
            .into_iter()
            .map(|id| ai_agents_tools::TodoItem {
                id: id.into(),
                content: id.into(),
                active_form: None,
                status: ai_agents_tools::TodoStatus::Pending,
            })
            .collect();
        assert!(!canonical.try_set(items));
        let done = runner
            .resume(
                &revised.run.key.run_id,
                revised.run.revision,
                TaskResumeInput::Continue,
            )
            .await
            .unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Completed);
        assert!(canonical.list_for_run(&next.binding).is_none());
    }
}

struct FailedPlanTool(Arc<std::sync::atomic::AtomicUsize>);
#[async_trait::async_trait]
impl ai_agents_core::Tool for FailedPlanTool {
    fn id(&self) -> &str {
        "fail_plan"
    }
    fn name(&self) -> &str {
        "fail_plan"
    }
    fn description(&self) -> &str {
        "Count a real failed plan attempt"
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({"type":"object"})
    }
    fn safety_metadata(&self) -> ai_agents_core::ToolSafetyMetadata {
        ai_agents_core::ToolSafetyMetadata {
            read_only: true,
            ..Default::default()
        }
    }
    async fn execute(
        &self,
        _: serde_json::Value,
        _: ai_agents_core::ToolExecutionContext,
    ) -> ai_agents_core::ToolResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        ai_agents_core::ToolResult::error("selected plan failed")
    }
}

/// Reflection replacement adoption is another review boundary, not permission to regenerate or replay a failed plan.
#[tokio::test]
async fn host_control_reflection_plan_review_preserves_previous_attempt_and_stops_on_rejection() {
    for reject in [false, true] {
        let bad = r#"{"steps":[{"id":"failed","description":"Fail","action_type":"tool","action_target":"fail_plan","args":{},"dependencies":[]}]}"#;
        let good = r#"{"steps":[{"id":"one","description":"Answer","action_type":"think","action_target":"reason","dependencies":[]}]}"#;
        let provider = RecordingProvider::new(&[bad, good, "done"]);
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let yaml = "name: ReflectionReview\nsystem_prompt: test\ntools: [fail_plan]\nreasoning:\n  mode: plan_and_execute\n  planning:\n    reflection:\n      enabled: true\n      max_replans: 1\n      on_step_failure: replan\n";
        let agent = Arc::new(
            AgentBuilder::from_yaml(yaml)
                .unwrap()
                .llm(provider.clone())
                .auto_configure_features()
                .unwrap()
                .tool(Arc::new(FailedPlanTool(calls.clone())))
                .build()
                .unwrap(),
        );
        let store = support::store(agent.info().id, "reflection.v1", false).await;
        let mut config = super::runner_tests::config(8);
        config.defaults.supervision = Some(SupervisionConfig {
            stage_changes: None,
            plan_changes: Some(ReviewPolicy::RequireApproval),
        });
        let runner = AutonomyRunner::try_new(
            agent,
            config,
            Default::default(),
            store,
            "reflection.v1".into(),
        )
        .unwrap();
        let first = runner.run("objective", None).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let replacement = approve(&runner, &first.run).await.unwrap();
        assert_eq!(
            replacement.run.status,
            TaskRunStatus::Paused,
            "{:?}",
            replacement.run.stop_reason
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        let done = runner
            .resume(
                &replacement.run.key.run_id,
                replacement.run.revision,
                TaskResumeInput::Approval {
                    request_id: replacement.run.pending.unwrap().id,
                    result: if reject {
                        ApprovalResult::Rejected { reason: None }
                    } else {
                        ApprovalResult::Approved
                    },
                },
            )
            .await
            .unwrap();
        assert_eq!(
            done.run.status,
            if reject {
                TaskRunStatus::Incomplete
            } else {
                TaskRunStatus::Completed
            }
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            provider.calls.load(Ordering::SeqCst),
            if reject { 2 } else { 3 }
        );
    }
}

/// Plan response finalization retains its own caller position through state entry action suspension.
#[tokio::test]
async fn host_control_plan_response_state_actions_finish_once_without_plan_regeneration() {
    let plan = r#"{"steps":[{"id":"one","description":"Answer","action_type":"think","action_target":"reason","dependencies":[]}]}"#;
    let provider = RecordingProvider::new(&[plan, "done"]);
    let yaml = "name: PlanState\nsystem_prompt: test\ntools: [ask_user]\nreasoning:\n  mode: plan_and_execute\nstates:\n  initial: active\n  states:\n    active:\n      transitions:\n        - to: ready\n          guard: 'true'\n    ready:\n      regenerate_on_enter: false\n      on_enter:\n        - tool: ask_user\n          args: {question: Confirm, options: [yes], allow_other: false}\n";
    let agent = Arc::new(
        AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(provider.clone())
            .auto_configure_features()
            .unwrap()
            .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
                Arc::new(parking_lot::RwLock::new(None)),
            )))
            .build()
            .unwrap(),
    );
    let store = support::store(agent.info().id, "plan-state.v1", false).await;
    let mut config = super::runner_tests::config(8);
    config.defaults.supervision = Some(SupervisionConfig {
        stage_changes: None,
        plan_changes: Some(ReviewPolicy::RequireApproval),
    });
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_user_question: Some(InteractionAction::PauseRun),
        on_approval_required: Some(InteractionAction::PauseRun),
    });
    let runner = AutonomyRunner::try_new(
        agent.clone(),
        config,
        Default::default(),
        store,
        "plan-state.v1".into(),
    )
    .unwrap();
    let first = runner.run("objective", None).await.unwrap();
    let question = approve(&runner, &first.run).await.unwrap();
    assert_eq!(question.run.status, TaskRunStatus::Paused);
    assert!(matches!(
        question.run.pending.as_ref().unwrap().kind,
        TaskPendingKind::UserQuestion
    ));
    let done = runner
        .resume(
            &question.run.key.run_id,
            question.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: question.run.pending.unwrap().id,
                answer: json!({"answered":true,"selected":["yes"]}),
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
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    assert_eq!(agent.state_history().len(), 1);
}

/// Real controller stages produce reviews before completion credit is committed.
async fn fixture(
    sqlite: bool,
) -> (
    Arc<crate::RuntimeAgent>,
    AutonomyRunner,
    Arc<ScopedTaskRunStore>,
    Arc<RecordingProvider>,
) {
    let provider = RecordingProvider::new(&["done", "done", "done", "done", "done"]);
    let agent = Arc::new(
        AgentBuilder::new()
            .system_prompt("test")
            .llm(provider.clone())
            .build()
            .unwrap(),
    );
    let store = support::store(agent.info().id, "host.v1", sqlite).await;
    let runner = runner(agent.clone(), store.clone());
    (agent, runner, store, provider)
}

/// Reuses ordinary lifecycle scheduling rather than inventing a test-only review source.
fn runner(agent: Arc<crate::RuntimeAgent>, store: Arc<dyn TaskRunStore>) -> AutonomyRunner {
    let mut config = super::runner_tests::config(32);
    config.defaults.max_turns = Some(8);
    config.defaults.lifecycle =
        Some(serde_json::from_value(json!([{"id":"work"},{"id":"verify"}])).unwrap());
    config.defaults.supervision = Some(SupervisionConfig {
        stage_changes: Some(ReviewPolicy::RequireApproval),
        plan_changes: None,
    });
    AutonomyRunner::try_new(agent, config, Default::default(), store, "host.v1".into()).unwrap()
}

/// Approval always targets the coordinating pending identity at the current acknowledged revision.
async fn approve(runner: &AutonomyRunner, run: &TaskRun) -> ai_agents_core::Result<TaskRunResult> {
    runner
        .resume(
            &run.key.run_id,
            run.revision,
            TaskResumeInput::Approval {
                request_id: run.pending.as_ref().unwrap().id.clone(),
                result: ApprovalResult::Approved,
            },
        )
        .await
}

/// Reads exact persisted controller data, not a response-shaped approximation of execution.
async fn payload(store: &ScopedTaskRunStore, run: &TaskRun) -> TaskCheckpointPayload {
    serde_json::from_value(store.load(&run.key.run_id).await.unwrap().unwrap().payload).unwrap()
}

/// Stage approval does not rerun its completed model turn and cannot bypass another required review.
#[tokio::test]
async fn host_control_stage_reviews_apply_once_and_reject_stale_wrong_kind_and_modified_proposals()
{
    for sqlite in support::backends() {
        let (_, runner, store, provider) = fixture(*sqlite).await;
        let first = runner.run("objective", None).await.unwrap();
        assert_eq!(first.run.status, TaskRunStatus::Paused);
        let state = payload(&store, &first.run).await;
        assert_eq!(
            state
                .adapters
                .iter()
                .find(|a| a.id == "runtime.controller")
                .unwrap()
                .state["lifecycle"]["current_stage"],
            0
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        for input in [
            TaskResumeInput::Continue,
            TaskResumeInput::UserAnswer {
                request_id: first.run.pending.as_ref().unwrap().id.clone(),
                answer: json!("yes"),
            },
            TaskResumeInput::Approval {
                request_id: "foreign".into(),
                result: ApprovalResult::Approved,
            },
            TaskResumeInput::Approval {
                request_id: first.run.pending.as_ref().unwrap().id.clone(),
                result: ApprovalResult::Modified {
                    changes: std::collections::HashMap::from([("stage".into(), json!("skip"))]),
                },
            },
        ] {
            assert!(
                runner
                    .resume(&first.run.key.run_id, first.run.revision, input)
                    .await
                    .is_err()
            );
        }
        assert_eq!(
            store
                .load(&first.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .revision,
            first.run.revision
        );
        let second = approve(&runner, &first.run).await.unwrap();
        assert_eq!(second.run.status, TaskRunStatus::Paused);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        assert!(approve(&runner, &first.run).await.is_err());
        assert!(
            runner
                .resume(
                    &second.run.key.run_id,
                    second.run.revision,
                    TaskResumeInput::Approval {
                        request_id: first.run.pending.unwrap().id,
                        result: ApprovalResult::Approved
                    }
                )
                .await
                .is_err()
        );
        let done = approve(&runner, &second.run).await.unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Completed);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        assert_eq!(done.run.counters.turns, 2);
    }
}

/// A rejected stage neither commits its proposed credit nor runs another stage.
#[tokio::test]
async fn host_control_stage_rejection_stops_incomplete_without_credit_or_replay() {
    for sqlite in support::backends() {
        let (agent, runner, store, provider) = fixture(*sqlite).await;
        let first = runner.run("objective", None).await.unwrap();
        let result = runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::Approval {
                    request_id: first.run.pending.unwrap().id,
                    result: ApprovalResult::Rejected {
                        reason: Some("no".into()),
                    },
                },
            )
            .await
            .unwrap();
        assert_eq!(result.run.status, TaskRunStatus::Incomplete);
        assert_eq!(
            result.run.stop_reason.as_deref(),
            Some("supervision_rejected")
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        let saved = payload(&store, &result.run).await;
        assert_eq!(
            saved
                .adapters
                .iter()
                .find(|a| a.id == "runtime.controller")
                .unwrap()
                .state["lifecycle"]["current_stage"],
            0
        );
        assert!(agent.set_context("released", json!(true)).is_ok());
    }
}

/// Repeated objective changes remain paused and retire every old response without resetting counted work.
#[tokio::test]
async fn host_control_objective_supersession_requires_explicit_revision_bound_continue() {
    for sqlite in support::backends() {
        let (_, runner, store, provider) = fixture(*sqlite).await;
        let initial = runner.run("old", None).await.unwrap();
        let revised = runner
            .modify_objective_paused(
                &initial.run.key.run_id,
                initial.run.revision,
                "new",
                "host revision",
            )
            .await
            .unwrap();
        assert_eq!(revised.run.status, TaskRunStatus::Paused);
        assert_eq!(revised.run.objective, "new");
        assert!(matches!(
            revised.run.pending.as_ref().unwrap().kind,
            TaskPendingKind::HostContinue
        ));
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            serde_json::to_value(&initial.run.counters).unwrap(),
            serde_json::to_value(&revised.run.counters).unwrap()
        );
        assert!(approve(&runner, &initial.run).await.is_err());
        assert!(
            runner
                .resume(
                    &revised.run.key.run_id,
                    revised.run.revision,
                    TaskResumeInput::Approval {
                        request_id: initial.run.pending.unwrap().id,
                        result: ApprovalResult::Approved
                    }
                )
                .await
                .is_err()
        );
        let next = runner
            .modify_objective_paused(
                &revised.run.key.run_id,
                revised.run.revision,
                "latest",
                "second revision",
            )
            .await
            .unwrap();
        assert!(
            runner
                .resume(
                    &revised.run.key.run_id,
                    revised.run.revision,
                    TaskResumeInput::Continue
                )
                .await
                .is_err()
        );
        let work = runner
            .resume(
                &next.run.key.run_id,
                next.run.revision,
                TaskResumeInput::Continue,
            )
            .await
            .unwrap();
        assert_eq!(work.run.status, TaskRunStatus::Paused);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        assert_eq!(work.run.counters.turns, 2);
        let verify = approve(&runner, &work.run).await.unwrap();
        let done = approve(&runner, &verify.run).await.unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Completed);
        assert_eq!(done.run.objective, "latest");
        assert_eq!(done.run.counters.turns, 3);
        let saved = payload(&store, &done.run).await;
        assert_eq!(saved.objective_revision, 2);
        assert_eq!(
            saved
                .runtime
                .snapshot
                .memory
                .messages
                .iter()
                .filter(|message| message.role == ai_agents_core::Role::User)
                .count(),
            1
        );
        assert!(
            saved
                .evidence
                .records
                .iter()
                .any(|record| record.origin == "host.control")
        );
    }
}

/// Safe preflight rejection preserves the original usable pause instead of abandoning its owner.
#[tokio::test]
async fn host_control_invalid_objective_and_reason_preserve_original_review() {
    let (_, runner, _, provider) = fixture(false).await;
    let first = runner.run("old", None).await.unwrap();
    for (objective, reason) in [("", "reason"), ("new", "")] {
        assert!(
            runner
                .modify_objective_paused(
                    &first.run.key.run_id,
                    first.run.revision,
                    objective,
                    reason
                )
                .await
                .is_err()
        );
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        approve(&runner, &first.run).await.unwrap().run.status,
        TaskRunStatus::Paused
    );
}

/// Paused review cancellation and host Stop never apply the proposal.
#[tokio::test]
async fn host_control_review_stop_cancel_and_limit_revision_preserve_pending_proposal() {
    for sqlite in support::backends() {
        for control in ["stop", "cancel", "limits"] {
            let (_, runner, _, provider) = fixture(*sqlite).await;
            let first = runner.run("objective", None).await.unwrap();
            let result = match control {
                "stop" => runner
                    .stop_paused(&first.run.key.run_id, first.run.revision, "stop review")
                    .await
                    .unwrap(),
                "cancel" => runner
                    .cancel_paused(&first.run.key.run_id, first.run.revision)
                    .await
                    .unwrap(),
                _ => {
                    let mut limits = first.run.limits.clone();
                    limits.max_turns += 1;
                    let revised = runner
                        .revise_limits_paused(
                            &first.run.key.run_id,
                            first.run.revision,
                            limits,
                            "capacity",
                        )
                        .await
                        .unwrap();
                    assert_eq!(
                        revised.run.pending.as_ref().unwrap().id,
                        first.run.pending.as_ref().unwrap().id
                    );
                    approve(&runner, &revised.run).await.unwrap()
                }
            };
            assert_eq!(
                result.run.status,
                match control {
                    "stop" => TaskRunStatus::Incomplete,
                    "cancel" => TaskRunStatus::Cancelled,
                    _ => TaskRunStatus::Paused,
                }
            );
            assert_eq!(
                provider.calls.load(Ordering::SeqCst),
                if control == "limits" { 2 } else { 1 }
            );
        }
    }
}

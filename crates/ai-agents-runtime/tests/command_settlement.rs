use ai_agents_core::ToolExecutionContext;
use ai_agents_hitl::ApprovalResult;
use ai_agents_llm::mock::MockLLMProvider;
use ai_agents_runtime::{Agent, AgentBuilder, autonomy::*};
use ai_agents_tools::{CommandExecutionState, CommandRequest, CommandResponse, CommandRunner};
use async_trait::async_trait;
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Host {
    response: CommandResponse,
    calls: Arc<AtomicUsize>,
}
#[async_trait]
impl CommandRunner for Host {
    // This isolated host supplies facts, never process I/O, replay or execution authority.
    async fn run_command(&self, _: CommandRequest, _: ToolExecutionContext) -> CommandResponse {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.response.clone()
    }
}

struct PendingCleanup {
    entered: Arc<tokio::sync::Notify>,
    dropped: Arc<std::sync::atomic::AtomicBool>,
}
struct CleanupDrop(Arc<std::sync::atomic::AtomicBool>);
impl Drop for CleanupDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}
#[async_trait]
impl CommandRunner for PendingCleanup {
    // A deterministic host cleanup barrier models an await that cannot acknowledge cleanup before cancellation/drop.
    async fn run_command(&self, _: CommandRequest, _: ToolExecutionContext) -> CommandResponse {
        let _owned = CleanupDrop(self.dropped.clone());
        self.entered.notify_one();
        std::future::pending().await
    }
}

#[tokio::test]
async fn cleanup_wait_cancel_and_drop_keep_original_task_custody() {
    for cancel in [false, true] {
        let entered = Arc::new(tokio::sync::Notify::new());
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let agent = Arc::new(AgentBuilder::from_yaml("name: Cleanup\nsystem_prompt: test\ntools: [command]\ntool_security:\n  enabled: true\n  tools:\n    command:\n      allow_without_confirmation: true\n      allowed_commands: [{argv: [probe]}]\n      working_dirs: [.]\n").unwrap()
            .llm(Arc::new({ let mut p=MockLLMProvider::new("cleanup"); p.set_response("done"); p }))
            .auto_configure_features().unwrap().autonomy_extensions(AutonomyExtensions::builtins()).build().unwrap());
        agent.set_command_runner(Arc::new(PendingCleanup {
            entered: entered.clone(),
            dropped: dropped.clone(),
        }));
        let store = Arc::new(
            ScopedTaskRunStore::in_memory(agent.info().id, None, "cleanup.v1".into()).unwrap(),
        );
        let profile: AutonomyProfile = serde_json::from_value(json!({
            "enabled":true,"max_turns":1,"max_llm_calls":1,"max_tool_calls":1,"max_command_calls":1,
            "validation":{"checks":[{"id":"check","adapter":"builtin.command","config":{"command":"probe"}}]},
            "completion":{"validator_passed":"check"}
        })).unwrap();
        let runner = AutonomyRunner::try_new(
            agent.clone(),
            AutonomyConfig {
                defaults: profile,
                ..Default::default()
            },
            Default::default(),
            store.clone(),
            "cleanup.v1".into(),
        )
        .unwrap();
        let mut operation = Box::pin(runner.run("objective", None));
        tokio::select! { result=&mut operation => panic!("cleanup fixture returned: {result:?}"), _=entered.notified()=>{}, _=tokio::time::sleep(std::time::Duration::from_secs(5))=>panic!("cleanup not reached") }
        assert!(!dropped.load(Ordering::SeqCst));
        let runs = store.list().await.unwrap();
        assert_eq!(runs.len(), 1);
        let id = runs[0].key.run_id.clone();
        if cancel {
            agent.runtime_control().cancel_all();
            let result = tokio::time::timeout(std::time::Duration::from_secs(5), operation)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(result.run.status, TaskRunStatus::RecoveryRequired);
        } else {
            drop(operation);
        }
        assert!(dropped.load(Ordering::SeqCst));
        let snapshot = store.load(&id).await.unwrap().unwrap();
        assert_ne!(snapshot.status, TaskRunStatus::Completed);
        // Terminal/recovery storage can release its lease token without releasing the runtime's live effect custody.
        assert!(matches!(
            snapshot.status,
            TaskRunStatus::Running | TaskRunStatus::RecoveryRequired
        ));
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
        assert_eq!(payload.counters.tool_attempts, 1);
        assert_eq!(payload.counters.command_attempts, 1);
        assert!(payload.reservations.iter().any(|r| matches!(
            r.state,
            TaskEffectState::Dispatched | TaskEffectState::Uncertain
        )));
        assert!(
            agent
                .chat("cannot replace unacknowledged cleanup")
                .await
                .is_err()
        );
        assert!(
            runner
                .run("cannot replace protected owner", None)
                .await
                .is_err()
        );
    }
}

// An invoked runner's no-start report cannot refund tool/command accounting, while uncertainty must retain actual task custody.
#[tokio::test]
async fn returned_command_state_preserves_settlement_and_retained_accounting() {
    for retained in [false, true] {
        for (state, termination, timed_out, uncertain, complete) in [
            (
                CommandExecutionState::Finished,
                "exited",
                false,
                false,
                true,
            ),
            (
                CommandExecutionState::Unspecified,
                "exited",
                false,
                false,
                true,
            ),
            (
                CommandExecutionState::NotStarted,
                "error",
                false,
                false,
                false,
            ),
            (
                CommandExecutionState::Uncertain,
                "exited",
                false,
                true,
                false,
            ),
            (
                CommandExecutionState::Unspecified,
                "timeout",
                true,
                true,
                false,
            ),
            (
                CommandExecutionState::Unspecified,
                "error",
                false,
                true,
                false,
            ),
            (
                CommandExecutionState::Finished,
                "timeout",
                true,
                false,
                false,
            ),
        ] {
            let yaml = if retained {
                "name: Settlement\nsystem_prompt: test\ntools: [command]\nhitl:\n  tools:\n    command: {require_approval: true}\ntool_security:\n  enabled: true\n  tools:\n    command:\n      allow_without_confirmation: true\n      allowed_commands: [{argv: [probe]}]\n      working_dirs: [.]\n"
            } else {
                "name: Settlement\nsystem_prompt: test\ntools: [command]\ntool_security:\n  enabled: true\n  tools:\n    command:\n      allow_without_confirmation: true\n      allowed_commands: [{argv: [probe]}]\n      working_dirs: [.]\n"
            };
            let mut provider = MockLLMProvider::new("settlement");
            provider.set_response("done");
            let agent = Arc::new(
                AgentBuilder::from_yaml(yaml)
                    .unwrap()
                    .llm(Arc::new(provider))
                    .auto_configure_features()
                    .unwrap()
                    .autonomy_extensions(AutonomyExtensions::builtins())
                    .build()
                    .unwrap(),
            );
            let calls = Arc::new(AtomicUsize::new(0));
            agent.set_command_runner(Arc::new(Host {
                response: CommandResponse {
                    execution_state: state,
                    success: false,
                    exit_code: (termination == "exited").then_some(1),
                    termination: termination.into(),
                    truncated: true,
                    timed_out,
                    ..Default::default()
                },
                calls: calls.clone(),
            }));
            let store = Arc::new(
                ScopedTaskRunStore::in_memory(agent.info().id, None, "settlement.v1".into())
                    .unwrap(),
            );
            let profile: AutonomyProfile = serde_json::from_value(json!({
                "enabled":true,"max_turns":1,"max_llm_calls":1,"max_tool_calls":1,"max_command_calls":1,
                "hitl":{"on_approval_required":"pause_run"},
                "validation":{"checks":[{"id":"check","adapter":"builtin.command","config":{"command":"probe","expected_exit":1}}]},
                "completion":{"all":[{"validator_passed":"check"},{"command_exit":{"command":"probe","code":1}}]}
            })).unwrap();
            let runner = AutonomyRunner::try_new(
                agent.clone(),
                AutonomyConfig {
                    defaults: profile,
                    ..Default::default()
                },
                Default::default(),
                store.clone(),
                "settlement.v1".into(),
            )
            .unwrap();
            let mut result = runner.run("objective", None).await.unwrap();
            let mut interrupted = false;
            if retained {
                assert_eq!(result.run.status, TaskRunStatus::Paused);
                assert_eq!(calls.load(Ordering::SeqCst), 0);
                let id = result.run.pending.as_ref().unwrap().id.clone();
                let revision = result.run.revision;
                let resumed = runner
                    .resume(
                        &result.run.key.run_id,
                        revision,
                        TaskResumeInput::Approval {
                            request_id: id.clone(),
                            result: ApprovalResult::Approved,
                        },
                    )
                    .await;
                if uncertain {
                    assert!(
                        resumed
                            .as_ref()
                            .is_err_and(|error| error.to_string().contains("uncertain_effect"))
                    );
                    interrupted = true;
                } else {
                    result = resumed.unwrap();
                }
                assert!(
                    runner
                        .resume(
                            &result.run.key.run_id,
                            revision,
                            TaskResumeInput::Approval {
                                request_id: id,
                                result: ApprovalResult::Approved
                            }
                        )
                        .await
                        .is_err()
                );
            }
            assert_eq!(
                result.run.status == TaskRunStatus::Completed,
                complete,
                "{state:?}/{termination} retained={retained}"
            );
            if !interrupted {
                assert_eq!(
                    result.run.status == TaskRunStatus::RecoveryRequired,
                    uncertain,
                    "{state:?}/{termination} retained={retained}: {:?}",
                    result.run.status
                );
            }
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            let snapshot = store.load(&result.run.key.run_id).await.unwrap().unwrap();
            assert_eq!(snapshot.status == TaskRunStatus::Completed, complete);
            let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
            assert_eq!(payload.counters.tool_attempts, 1);
            assert_eq!(payload.counters.command_attempts, 1);
            assert_eq!(payload.counters.llm_attempts, 1);
            let record = &payload
                .evidence
                .tool_calls
                .iter()
                .find(|r| r.record.canonical_id == "command")
                .unwrap()
                .record;
            assert!(
                record.executed && record.success,
                "process outcome remains a factual tool observation"
            );
            assert_eq!(
                record
                    .metadata
                    .get("task_effect_uncertain")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                uncertain
            );
            if uncertain {
                assert_eq!(record.metadata["_task_result_eligible"], false);
            }
            let reservation = payload
                .reservations
                .iter()
                .find(|r| {
                    r.result.as_ref().is_some_and(|value| {
                        value
                            .get("metadata")
                            .and_then(|m| m.get("command_execution_state"))
                            .is_some()
                    })
                })
                .unwrap();
            assert_eq!(
                reservation.state,
                if uncertain {
                    TaskEffectState::Uncertain
                } else {
                    TaskEffectState::Completed
                }
            );
            assert!(
                reservation
                    .result
                    .as_ref()
                    .unwrap()
                    .get("not_invoked")
                    .is_none()
            );
            if uncertain {
                assert!(agent.chat("cannot replace uncertain work").await.is_err());
            }
        }
    }
}

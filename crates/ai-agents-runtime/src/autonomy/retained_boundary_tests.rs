//! Real executor regressions distinguish retained response authority from display/checkpoint identity.

use super::*;
use crate::{Agent, AgentBuilder, RuntimeAgent};
use ai_agents_core::ToolExecutionContext;
use ai_agents_hitl::ApprovalResult;
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{Arc, atomic::Ordering};

struct CommandProbe(Arc<parking_lot::Mutex<Vec<Value>>>);
#[async_trait]
impl ai_agents_tools::CommandRunner for CommandProbe {
    // Exercise the production command tool with isolated output, without launching any process.
    async fn run_command(
        &self,
        request: ai_agents_tools::CommandRequest,
        _: ToolExecutionContext,
    ) -> ai_agents_tools::CommandResponse {
        self.0
            .lock()
            .push(json!({"command":request.argv.join(" ")}));
        ai_agents_tools::CommandResponse {
            success: true,
            exit_code: Some(0),
            termination: "exited".into(),
            ..Default::default()
        }
    }
}
struct NoApproval;
#[async_trait]
impl ai_agents_hitl::ApprovalHandler for NoApproval {
    // Handler polling would bypass the retained request's safe publication boundary.
    async fn request_approval(&self, _: ai_agents_hitl::ApprovalRequest) -> ApprovalResult {
        panic!("retained approval must park before external delivery")
    }
}
struct CommandFixture {
    agent: Arc<RuntimeAgent>,
    runner: AutonomyRunner,
    store: Arc<ScopedTaskRunStore>,
    calls: Arc<parking_lot::Mutex<Vec<Value>>>,
    provider: Arc<super::runner_tests::RecordingProvider>,
    config: AutonomyConfig,
}
// Optional command validation lets the typed command gate, not the builtin result, decide completion.
async fn command_fixture(sqlite: bool, fixed: bool) -> CommandFixture {
    command_fixture_with_timeout(sqlite, fixed, 60).await
}
// The pending request, not the run or validation check, expires during selected acknowledgement waits.
async fn command_fixture_with_timeout(sqlite: bool, fixed: bool, seconds: u64) -> CommandFixture {
    let provider = super::runner_tests::RecordingProvider::new(&["done"]);
    let calls = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let mut extensions = AutonomyExtensions::builtins();
    if fixed {
        extensions
            .register_host_validation(HostValidationBinding {
                id: "fixed-command".into(),
                validator: "builtin.command".into(),
                version: 1,
                profile: None,
                scope: "task".into(),
                tool: "command".into(),
                arguments: json!({"command":"original","cwd":"."}),
                require_approval: true,
            })
            .unwrap();
    }
    let agent = Arc::new(AgentBuilder::from_yaml(&"name: Evidence\nsystem_prompt: test\ntools: [command]\ntool_security:\n  enabled: true\n  tools:\n    command:\n      allow_without_confirmation: true\n      allowed_commands: [{argv: [original]}, {argv: [replacement]}]\n      working_dirs: [.]\nhitl:\n  default_timeout_seconds: 60\n  tools:\n    command:\n      require_approval: true\n".replace("default_timeout_seconds: 60", &format!("default_timeout_seconds: {seconds}")))
        .unwrap().llm(provider.clone()).auto_configure_features().unwrap()
        .approval_handler(Arc::new(NoApproval)).autonomy_extensions(extensions)
        .build().unwrap());
    agent.set_command_runner(Arc::new(CommandProbe(calls.clone())));
    let store =
        super::continuation_test_support::store(agent.info().id, "boundary.v1", sqlite).await;
    let mut config = super::runner_tests::config(1);

    config.defaults.validation = Some(serde_yaml::from_str(if fixed {
        "checks: [{id: check, adapter: builtin.command, required: false, schedule: each_cycle, config: {command: original, host_binding: fixed-command}}]"
    } else {
        "checks: [{id: check, adapter: builtin.command, required: false, schedule: each_cycle, config: {command: original}}]"
    }).unwrap());
    config.defaults.completion =
        Some(serde_yaml::from_str("command_exit: {command: original, code: 0}").unwrap());
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    let runner = AutonomyRunner::try_new(
        agent.clone(),
        config.clone(),
        Default::default(),
        store.clone(),
        "boundary.v1".into(),
    )
    .unwrap();
    CommandFixture {
        agent,
        runner,
        store,
        calls,
        provider,
        config,
    }
}
// Executing an approved replacement cannot prove successful execution of the original fixed command.
#[tokio::test]
async fn retained_boundary_command_evidence_matches_actual_operation() {
    for sqlite in super::continuation_test_support::backends() {
        for changed in [false, true] {
            let f = command_fixture(*sqlite, false).await;
            let first = f.runner.run("objective", None).await.unwrap();

            assert_eq!(
                first.run.status,
                TaskRunStatus::Paused,
                "{:?}",
                first.run.stop_reason
            );
            let done = f
                .runner
                .resume(
                    &first.run.key.run_id,
                    first.run.revision,
                    TaskResumeInput::Approval {
                        request_id: first.run.pending.as_ref().unwrap().id.clone(),
                        result: if changed {
                            ApprovalResult::Modified {
                                changes: std::collections::HashMap::from([(
                                    "command".into(),
                                    json!("replacement"),
                                )]),
                            }
                        } else {
                            ApprovalResult::Approved
                        },
                    },
                )
                .await
                .unwrap();
            assert_eq!(f.calls.lock().len(), 1);
            assert_eq!(
                f.calls.lock()[0]["command"],
                if changed { "replacement" } else { "original" }
            );
            assert_eq!(
                done.run.status == TaskRunStatus::Completed,
                !changed,
                "replacement cannot pass original gate"
            );
            assert_eq!(f.provider.calls.load(Ordering::SeqCst), 1);
            let payload: TaskCheckpointPayload = serde_json::from_value(
                f.store
                    .load(&done.run.key.run_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .payload,
            )
            .unwrap();
            assert!(payload.evidence.tool_calls.iter().any(|r| r.record.executed
                && r.record.executed_arguments["command"]
                    == if changed { "replacement" } else { "original" }));
        }
    }
}
// Fixed host bindings still reject changed arguments before the isolated implementation executes.
#[tokio::test]
async fn retained_boundary_fixed_command_rejects_replacement() {
    let f = command_fixture(false, true).await;
    let first = f.runner.run("objective", None).await.unwrap();
    let _ = f
        .runner
        .resume(
            &first.run.key.run_id,
            first.run.revision,
            TaskResumeInput::Approval {
                request_id: first.run.pending.as_ref().unwrap().id.clone(),
                result: ApprovalResult::Modified {
                    changes: std::collections::HashMap::from([(
                        "command".into(),
                        json!("replacement"),
                    )]),
                },
            },
        )
        .await;
    assert!(f.calls.lock().is_empty());
}
// The exact human request can expire while run/check clocks still permit execution.
#[tokio::test]
async fn retained_boundary_request_expiry_survives_claim_consume_and_admission_ack() {
    use super::continuation_test_support::{self as support, ReceiptStore};
    for sqlite in support::backends() {
        for phase in ["claim", "consume", "admit"] {
            let mut f = command_fixture_with_timeout(*sqlite, false, 2).await;
            let fault =
                ReceiptStore::checkpoint(f.store.clone(), "hold", json!({"completed":u64::MAX}));
            f.runner = AutonomyRunner::try_new(
                f.agent.clone(),
                f.config.clone(),
                Default::default(),
                fault.clone(),
                "boundary.v1".into(),
            )
            .unwrap();
            let first = f.runner.run("objective", None).await.unwrap();
            let id = first.run.pending.as_ref().unwrap().id.clone();
            *fault.checkpoint.lock() = Some(match phase {
                "claim" => json!({"claim":true}),
                "consume" => json!({"consume":id}),
                _ => json!({"tool":"command","phase":"admit"}),
            });
            fault.armed.store(true, Ordering::SeqCst);
            let mut resume = Box::pin(f.runner.resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::Approval {
                    request_id: id,
                    result: ApprovalResult::Approved,
                },
            ));
            tokio::select! { result=&mut resume => panic!("selected ACK not held: {phase}: {result:?}"), permit=fault.reached.acquire()=>permit.unwrap().forget(), _=tokio::time::sleep(std::time::Duration::from_secs(5))=>panic!("ACK not reached") }
            tokio::time::sleep(std::time::Duration::from_millis(2050)).await;
            fault.release.add_permits(1);
            let outcome = resume.await;
            assert!(
                !outcome
                    .as_ref()
                    .is_ok_and(|r| r.run.status == TaskRunStatus::Completed)
            );
            assert!(
                f.calls.lock().is_empty(),
                "request expired at {phase} but body executed"
            );
            assert_eq!(f.provider.calls.load(Ordering::SeqCst), 1);
            fault.assert_hit();
        }
    }
}

struct SkillProbe(Arc<std::sync::atomic::AtomicUsize>);
#[async_trait]
impl ai_agents_core::Tool for SkillProbe {
    // This stable implementation stays available in both states so scope denial cannot mask generation invalidation.
    fn id(&self) -> &str {
        "probe"
    }
    // Canonical naming avoids alias changes becoming the reason a stale continuation is rejected.
    fn name(&self) -> &str {
        "probe"
    }
    // The fixture counts real executor effects rather than model-proposed calls.
    fn description(&self) -> &str {
        "Counts actual script effects in either state"
    }
    // No domain argument validation is involved in the original-generation race.
    fn input_schema(&self) -> Value {
        json!({"type":"object"})
    }
    // A read-only probe still needs original script authority; resource footprint admission is not the tested denial.
    fn safety_metadata(&self) -> ai_agents_core::ToolSafetyMetadata {
        ai_agents_core::ToolSafetyMetadata {
            read_only: true,
            ..Default::default()
        }
    }
    // Count the actual body independently of checkpoint or result publication.
    async fn execute(&self, _: Value, _: ToolExecutionContext) -> ai_agents_core::ToolResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        ai_agents_core::ToolResult::ok("effect")
    }
}
// Both states permit the same tools and script, proving original-generation invalidation rather than ordinary grant denial.
#[tokio::test]
async fn retained_boundary_skill_generation_survives_prompt_journal_tool_and_credit_waits() {
    use super::continuation_test_support::{self as support, ReceiptStore};
    for sqlite in support::backends() {
        for phase in [
            "prompt",
            "final_prompt",
            "journal",
            "admit",
            "settle",
            "credit",
        ] {
            for changed in [false, true] {
                let provider = support::FaultProvider::new(
                    &["middle", "done", "done"],
                    if phase == "final_prompt" { 1 } else { 0 },
                    if phase.contains("prompt") {
                        "gate"
                    } else {
                        "none"
                    },
                );
                let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                let yaml = "name: SkillBinding\nsystem_prompt: test\ntools: [probe]\nstates:\n  initial: a\n  states:\n    a: {}\n    b: {}\nskills:\n  - id: script\n    description: script\n    trigger: explicit\n    steps:\n      - tool: probe\n      - prompt: Middle\n      - tool: probe\n      - prompt: Finish\n";
                let agent = Arc::new(
                    AgentBuilder::from_yaml(yaml)
                        .unwrap()
                        .llm(provider.clone())
                        .auto_configure_features()
                        .unwrap()
                        .tool(Arc::new(SkillProbe(calls.clone())))
                        .build()
                        .unwrap(),
                );
                let generation = agent.state_machine().unwrap().generation();
                let store = support::store(agent.info().id, "skill-binding.v1", *sqlite).await;
                let selector = match phase {
                    "journal" => json!({"skill_step":1}),
                    "credit" => json!({"skill_credit":true}),
                    _ => {
                        json!({"tool":"probe","phase":if phase=="settle" {"settle"} else {"admit"}})
                    }
                };
                let fault = ReceiptStore::checkpoint(store.clone(), "hold", selector);
                fault
                    .armed
                    .store(!phase.contains("prompt"), Ordering::SeqCst);
                let runner = AutonomyRunner::try_new(
                    agent.clone(),
                    super::runner_tests::config(4),
                    Default::default(),
                    fault.clone(),
                    "skill-binding.v1".into(),
                )
                .unwrap();
                let mut work = Box::pin(runner.run_skill("objective", "script", None));
                if phase.contains("prompt") {
                    tokio::select! { result=&mut work=>panic!("provider not held: {result:?}"), permit=provider.reached.acquire()=>permit.unwrap().forget(), _=tokio::time::sleep(std::time::Duration::from_secs(5))=>panic!("provider not reached") }
                } else {
                    tokio::select! { result=&mut work=>panic!("checkpoint not held: {phase} {result:?}"), permit=fault.reached.acquire()=>permit.unwrap().forget(), _=tokio::time::sleep(std::time::Duration::from_secs(5))=>panic!("checkpoint not reached: {phase}") }
                }
                let live_credit = if phase == "credit" {
                    Some(
                        fault
                            .execution
                            .lock()
                            .upgrade()
                            .expect("credit ACK must hold its live execution"),
                    )
                } else {
                    None
                };
                if changed {
                    agent
                        .state_machine()
                        .unwrap()
                        .transition_to("b", "trusted host invalidation")
                        .unwrap();
                }
                provider.release.add_permits(1);
                fault.release.add_permits(1);
                let outcome = work.await;
                if changed {
                    assert!(
                        !outcome
                            .as_ref()
                            .is_ok_and(|r| r.run.status == TaskRunStatus::Completed),
                        "stale script completed at {phase}"
                    );
                    assert_eq!(
                        calls.load(Ordering::SeqCst),
                        match phase {
                            "admit" => 0,
                            "credit" | "final_prompt" => 2,
                            _ => 1,
                        },
                        "effect count at {phase}"
                    );
                    for summary in store.list().await.unwrap() {
                        let saved = store.load(&summary.key.run_id).await.unwrap().unwrap();
                        let p: TaskCheckpointPayload =
                            serde_json::from_value(saved.payload).unwrap();
                        assert_eq!(
                            p.counters.tool_attempts,
                            calls.load(Ordering::SeqCst) as u64,
                            "consumed tool accounting at {phase}"
                        );
                        assert_eq!(
                            p.counters.llm_attempts,
                            provider.calls.load(Ordering::SeqCst) as u64,
                            "consumed provider accounting at {phase}"
                        );
                        assert_eq!(
                            p.evidence
                                .tool_calls
                                .iter()
                                .filter(|r| r.record.canonical_id == "probe" && r.record.executed)
                                .count(),
                            calls.load(Ordering::SeqCst),
                            "actual effect evidence at {phase}"
                        );
                        for adapter in p.adapters.iter().filter(|a| a.adapter == "runtime.skill") {
                            if phase == "prompt" || phase == "journal" {
                                assert_eq!(adapter.state["cursor"]["next_step"], 1);
                                assert_eq!(
                                    adapter.state["cursor"]["context"]["step_results"]
                                        .as_array()
                                        .unwrap()
                                        .len(),
                                    1
                                );
                            }
                            if phase == "final_prompt" {
                                assert_eq!(adapter.state["cursor"]["next_step"], 3);
                            }
                            assert_eq!(
                                adapter.state["state_generation"],
                                json!(generation),
                                "journal rebound old cursor"
                            );
                        }
                        assert_ne!(saved.status, TaskRunStatus::Completed);
                    }
                    if phase == "credit" {
                        assert!(
                            !live_credit.as_ref().unwrap().skill_segment_completed(),
                            "stale committed credit became live authority"
                        );
                    }
                } else {
                    assert_eq!(
                        outcome.unwrap().run.status,
                        TaskRunStatus::Completed,
                        "normal script at {phase}"
                    );
                    assert_eq!(calls.load(Ordering::SeqCst), 2);
                    if let Some(owner) = fault.owner.lock().as_ref() {
                        assert!(owner.responses.lock().is_empty());
                    }
                }
            }
        }
    }
}

// Removing the current confirmation gate cannot bypass the original retained review or create a localization/provider call.
#[tokio::test]
async fn retained_boundary_disappearing_confirmation_gate_cannot_rebind_old_approval() {
    for changed in [false, true] {
        let marker = ai_agents_core::encode_native_tool_call_markers(
            &[ai_agents_core::ToolCall {
                id: "call".into(),
                name: "probe".into(),
                arguments: json!({}),
            }],
            None,
        )
        .unwrap();
        let provider = super::runner_tests::RecordingProvider::new(&[&marker, "done"]);
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let agent=Arc::new(AgentBuilder::from_yaml("name: Confirmation\nsystem_prompt: test\ntools: [probe]\ntool_security:\n  enabled: true\n  tools:\n    probe:\n      require_confirmation: true\nhitl: {}\n").unwrap().llm(provider.clone()).auto_configure_features().unwrap().approval_handler(Arc::new(NoApproval)).tool(Arc::new(SkillProbe(calls.clone()))).build().unwrap());
        let store = Arc::new(
            ScopedTaskRunStore::in_memory(agent.info().id, None, "confirmation.v1".into()).unwrap(),
        );
        let mut config = super::runner_tests::config(3);
        config.defaults.hitl = Some(AutonomyHitlConfig {
            on_approval_required: Some(InteractionAction::PauseRun),
            on_user_question: None,
        });
        let runner = AutonomyRunner::try_new(
            agent.clone(),
            config,
            Default::default(),
            store,
            "confirmation.v1".into(),
        )
        .unwrap();
        let first = runner.run("objective", None).await.unwrap();
        assert_eq!(first.run.status, TaskRunStatus::Paused);
        if changed {
            agent.runtime_control().set_tool_security(
                serde_yaml::from_str(
                    "enabled: true\ntools: {probe: {allow_without_confirmation: true}}",
                )
                .unwrap(),
            );
        }
        let result = runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::Approval {
                    request_id: first.run.pending.as_ref().unwrap().id.clone(),
                    result: ApprovalResult::Approved,
                },
            )
            .await;
        assert_eq!(calls.load(Ordering::SeqCst), usize::from(!changed));
        assert_eq!(
            provider.calls.load(Ordering::SeqCst),
            if changed { 1 } else { 2 }
        );
        if !changed {
            assert_eq!(result.unwrap().run.status, TaskRunStatus::Completed);
        }
    }
}

// Question data has the same original request bound after admission ACK, without being treated as human execution authority.
#[tokio::test]
async fn retained_boundary_question_expiry_and_uncertain_ack_preserve_custody() {
    use super::continuation_test_support::{self as support, ReceiptStore};
    for sqlite in support::backends() {
        for mode in ["hold", "lost", "drop"] {
            let marker=ai_agents_core::encode_native_tool_call_markers(&[ai_agents_core::ToolCall{id:"question".into(),name:"ask_user".into(),arguments:json!({"question":"Choose","options":["yes"],"allow_other":false,"timeout_seconds":2})}],None).unwrap();
            let provider = super::runner_tests::RecordingProvider::new(&[&marker, "done"]);
            let agent = Arc::new(
                AgentBuilder::from_yaml(
                    "name: QuestionBound\nsystem_prompt: test\ntools: [ask_user]\n",
                )
                .unwrap()
                .llm(provider.clone())
                .auto_configure_features()
                .unwrap()
                .build()
                .unwrap(),
            );
            let store = support::store(agent.info().id, "question-bound.v1", *sqlite).await;
            let fault = ReceiptStore::checkpoint(
                store.clone(),
                mode,
                json!({"tool":"ask_user","phase":"admit"}),
            );
            let mut config = super::runner_tests::config(3);
            config.defaults.hitl = Some(AutonomyHitlConfig {
                on_approval_required: Some(InteractionAction::PauseRun),
                on_user_question: Some(InteractionAction::PauseRun),
            });
            let runner = AutonomyRunner::try_new(
                agent.clone(),
                config,
                Default::default(),
                fault.clone(),
                "question-bound.v1".into(),
            )
            .unwrap();
            let first = runner.run("objective", None).await.unwrap();
            fault.armed.store(true, Ordering::SeqCst);
            let mut resume = Box::pin(runner.resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: first.run.pending.as_ref().unwrap().id.clone(),
                    answer: json!({"answered":true,"selected":["yes"]}),
                },
            ));
            if mode == "lost" {
                assert!(resume.await.is_err());
            } else {
                tokio::select! { result=&mut resume=>panic!("question admission not held: {result:?}"), permit=fault.reached.acquire()=>permit.unwrap().forget(), _=tokio::time::sleep(std::time::Duration::from_secs(5))=>panic!("admission not reached") }
                if mode == "drop" {
                    drop(resume);
                } else {
                    tokio::time::sleep(std::time::Duration::from_millis(2050)).await;
                    fault.release.add_permits(1);
                    let result = resume.await;
                    assert!(
                        !result
                            .as_ref()
                            .is_ok_and(|r| r.run.status == TaskRunStatus::Completed)
                    );
                }
            }
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            let snapshot = store.load(&first.run.key.run_id).await.unwrap().unwrap();
            let p: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
            assert!(
                !p.evidence
                    .tool_calls
                    .iter()
                    .any(|r| r.record.canonical_id == "ask_user" && r.record.executed)
            );
            if mode != "hold" {
                assert!(agent.chat("blocked").await.is_err());
            }
            fault.assert_hit();
        }
    }
}

// Post-script reflection remains bound until response publication, but intentional state transitions are a later boundary.
#[tokio::test]
async fn retained_boundary_skill_reflection_cannot_retry_after_generation_change() {
    use super::continuation_test_support as support;
    for changed in [false, true] {
        let provider = support::FaultProvider::new(
            &[
                "done",
                "1. FAIL - invalid\nCONFIDENCE: 0.0\nOVERALL: FAIL",
                "done",
                "1. PASS - valid\nCONFIDENCE: 1.0\nOVERALL: PASS",
            ],
            1,
            "gate",
        );
        let yaml = "name: SkillReflection\nsystem_prompt: test\nstates:\n  initial: a\n  states:\n    a: {}\n    b: {}\nskills:\n  - id: script\n    description: script\n    trigger: explicit\n    reflection:\n      enabled: enabled\n      criteria: [valid]\n      max_retries: 1\n    steps:\n      - prompt: Finish\n";
        let agent = Arc::new(
            AgentBuilder::from_yaml(yaml)
                .unwrap()
                .llm(provider.clone())
                .auto_configure_features()
                .unwrap()
                .build()
                .unwrap(),
        );
        let store = support::store(agent.info().id, "skill-reflection.v1", false).await;
        let runner = AutonomyRunner::try_new(
            agent.clone(),
            super::runner_tests::config(6),
            Default::default(),
            store,
            "skill-reflection.v1".into(),
        )
        .unwrap();
        let mut work = Box::pin(runner.run_skill("objective", "script", None));
        tokio::select! {result=&mut work=>panic!("reflection not held: {result:?}"), permit=provider.reached.acquire()=>permit.unwrap().forget(), _=tokio::time::sleep(std::time::Duration::from_secs(5))=>panic!("reflection not reached")}
        if changed {
            agent
                .state_machine()
                .unwrap()
                .transition_to("b", "trusted invalidation")
                .unwrap();
        }
        provider.release.add_permits(1);
        let result = work.await;
        if changed {
            assert!(
                !result
                    .as_ref()
                    .is_ok_and(|r| r.run.status == TaskRunStatus::Completed)
            );
            assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        } else {
            assert_eq!(result.unwrap().run.status, TaskRunStatus::Completed);
            assert_eq!(provider.calls.load(Ordering::SeqCst), 4);
        }
    }
}

// Allocation-light verification must compare every persisted envelope field and preserve backend-owned monotonic timestamps.
#[test]
fn retained_boundary_receipt_rejects_each_changed_envelope_field() {
    let initial = super::tests::checkpoint("receipt-fields");
    let previous = initial
        .transition(
            &TaskRunMutation::Claim {
                expected_revision: 0,
                owner_token: "owner".into(),
            },
            chrono::Utc::now(),
        )
        .unwrap();
    let mutation = TaskRunMutation::Checkpoint {
        expected_revision: previous.revision,
        owner_token: "owner".into(),
        status: TaskRunStatus::Running,
        payload: previous.payload.clone(),
        release: false,
    };
    let saved = previous.transition(&mutation, chrono::Utc::now()).unwrap();
    super::store::verify_checkpoint_receipt(&previous, &mutation, &saved).unwrap();
    for field in 0..11 {
        let mut invalid = saved.clone();
        match field {
            0 => invalid.schema_version += 1,
            1 => invalid.key.agent_id.push_str("other"),
            2 => invalid.key.run_id.push_str("other"),
            3 => invalid.actor_id = Some("other".into()),
            4 => invalid.revision += 1,
            5 => invalid.status = TaskRunStatus::Failed,
            6 => invalid.owner_token = None,
            7 => invalid.cancel_requested = true,
            8 => invalid.created_at -= chrono::Duration::seconds(1),
            9 => invalid.updated_at = previous.updated_at - chrono::Duration::seconds(1),
            _ => invalid.payload["objective"] = json!("changed"),
        }
        assert!(
            super::store::verify_checkpoint_receipt(&previous, &mutation, &invalid).is_err(),
            "field {field}"
        );
    }
}

// Receipt checking runs after acknowledged commit and must not repeat a now-expired final deadline or live admission callback.
#[test]
fn retained_boundary_receipt_does_not_reauthorize_committed_write() {
    #[derive(Debug)]
    struct NoRecheck;
    impl TaskFinalAdmission for NoRecheck {
        // A backend already acknowledged this write; receipt equality must not invoke external proof again.
        fn validate(&self) -> ai_agents_core::Result<()> {
            panic!("receipt verification re-ran final admission")
        }
    }
    let initial = super::tests::checkpoint("receipt-admission");
    let previous = initial
        .transition(
            &TaskRunMutation::Claim {
                expected_revision: 0,
                owner_token: "owner".into(),
            },
            chrono::Utc::now(),
        )
        .unwrap();
    let plain = TaskRunMutation::Checkpoint {
        expected_revision: previous.revision,
        owner_token: "owner".into(),
        status: TaskRunStatus::Completed,
        payload: previous.payload.clone(),
        release: true,
    };
    let saved = previous.transition(&plain, chrono::Utc::now()).unwrap();
    let final_write = TaskRunMutation::FinalCheckpoint {
        expected_revision: previous.revision,
        owner_token: "owner".into(),
        status: TaskRunStatus::Completed,
        payload: previous.payload.clone(),
        release: true,
        deadline: Some(std::time::Instant::now() - std::time::Duration::from_secs(1)),
        expires_at: Some(chrono::Utc::now() - chrono::Duration::seconds(1)),
        evidence: Some(Arc::new(NoRecheck)),
    };
    super::store::verify_checkpoint_receipt(&previous, &final_write, &saved).unwrap();
}

// A host control generation change while parked must not mint a new reviewed binding for an old receipt.
#[tokio::test]
async fn retained_boundary_approval_keeps_original_control_binding() {
    for changed in [false, true] {
        let f = command_fixture(false, false).await;
        let first = f.runner.run("objective", None).await.unwrap();
        if changed {
            f.agent
                .runtime_control()
                .set_tool_scope(vec!["command".into()]);
        }
        let _ = f
            .runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::Approval {
                    request_id: first.run.pending.as_ref().unwrap().id.clone(),
                    result: ApprovalResult::Approved,
                },
            )
            .await;
        assert_eq!(f.calls.lock().len(), usize::from(!changed));
        assert_eq!(f.provider.calls.load(Ordering::SeqCst), 1);
    }
}

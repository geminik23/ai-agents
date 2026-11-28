//! Cross-entry acceptance consumes the same real controller, conditional journal and existing composition algorithms.

use super::*;
use crate::autonomy::continuation_test_support as support;
use crate::autonomy::restoration_test_support::{FreshRuntime, RebuildHost, restore};

// Pre-response entry retains input preparation and ordinary effects without claiming them as new task tool execution.
#[tokio::test]
async fn auto_pre_response_handoff_keeps_processed_input_and_prior_effect_once() {
    let yaml = STATE.replace("  initial: work", "  initial: triage").replace("  states:\n    work:", "  states:\n    triage:\n      on_exit:\n        - tool: effect\n          args: {phase: before_task}\n      transitions:\n        - to: work\n          guard: 'true'\n          timing: pre_response\n          requires_response: false\n    work:")
        + "tools: [effect]\nruntime:\n  optimization:\n    enabled: true\n    pre_response_deterministic_transitions: true\nprocess:\n  input:\n    - type: transform\n      config:\n        llm: router\n        prompt: Prepare input\n";
    let effects = Arc::new(AtomicUsize::new(0));
    let main = RecordingProvider::new(&["done"]);
    let router = RecordingProvider::new(&["prepared"]);
    let agent = AgentBuilder::from_yaml(&yaml)
        .unwrap()
        .llm(main.clone())
        .llm_alias("router", router.clone())
        .tool(Arc::new(Effect(effects.clone())))
        .build()
        .unwrap();
    agent.chat("original").await.unwrap();
    let result = agent.automatic_task_result().unwrap();
    assert_eq!(
        result.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        result.run.stop_reason
    );
    assert_eq!(effects.load(Ordering::SeqCst), 1);
    assert_eq!(router.calls.load(Ordering::SeqCst), 1);
    assert_eq!(result.run.counters.tool_attempts, 0);
    assert_eq!(result.run.counters.llm_attempts, 1);
    let snapshot = agent.save_state().await.unwrap();
    let users: Vec<_> = snapshot
        .memory
        .messages
        .iter()
        .filter(|message| message.role == ai_agents_core::Role::User)
        .collect();
    assert_eq!(users.len(), 1);
    assert_eq!(users[0].content, "prepared");
}

// Post-response arrival starts work only when the destination actually regenerates a response for this input.
#[tokio::test]
async fn auto_post_response_entry_distinguishes_arrival_from_input_handling() {
    for regenerate in [false, true] {
        let yaml = STATE.replace("  initial: work", "  initial: triage").replace("  states:\n    work:", "  states:\n    triage:\n      transitions:\n        - to: work\n          guard: 'true'\n    work:") + &format!("      regenerate_on_enter: {regenerate}\n");
        let main = RecordingProvider::new(&["ordinary candidate", "done"]);
        let agent = agent(&yaml, main.clone(), RecordingProvider::new(&[]));
        let response = agent.chat("request").await.unwrap();
        if regenerate {
            assert_eq!(
                agent.automatic_task_result().unwrap().run.status,
                TaskRunStatus::Completed
            );
            assert_eq!(response.content, "done");
        } else {
            assert!(agent.automatic_task_result().is_none());
            assert_eq!(response.content, "ordinary candidate");
        }
        let snapshot = agent.save_state().await.unwrap();
        assert_eq!(
            snapshot
                .memory
                .messages
                .iter()
                .filter(|message| message.role == ai_agents_core::Role::User)
                .count(),
            1
        );
    }
}

// The declaration owns its subtree, not one leaf string, and ordinary inheritance boundaries still apply.
#[tokio::test]
async fn auto_hierarchical_scope_survives_sibling_transition_and_honors_child_disable() {
    let yaml = "name: auto\nsystem_prompt: Helpful\nllm: {default: default, router: router}\nstates:\n  initial: support\n  regenerate_on_transition: false\n  states:\n    support:\n      initial: diagnose\n      autonomy: {enabled: true, mode: state_until_exit, max_turns: 3, completion: {state: outside}}\n      states:\n        diagnose:\n          transitions: [{to: fix, guard: 'true'}]\n        fix:\n          transitions: [{to: outside, guard: 'true'}]\n    outside: {}\n";
    let main = RecordingProvider::new(&["working", "done"]);
    let agent = agent(yaml, main.clone(), RecordingProvider::new(&[]));
    agent.chat("support").await.unwrap();
    let result = agent.automatic_task_result().unwrap();
    assert_eq!(
        result.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        result.run.stop_reason
    );
    assert_eq!(result.run.counters.turns, 2);
    assert_eq!(agent.current_state().as_deref(), Some("outside"));
    for child in ["autonomy: {enabled: false}", "inherit_parent: false"] {
        let yaml = yaml.replace(
            "        diagnose:\n",
            &format!("        diagnose:\n          {child}\n"),
        );
        let agent = super::agent(
            &yaml,
            RecordingProvider::new(&["ordinary"]),
            RecordingProvider::new(&[]),
        );
        agent.chat("support").await.unwrap();
        assert!(agent.automatic_task_result().is_none());
    }
}

// Explicit strategies are independent of skill scope and do not reroute or force the initial script when not selected.
#[tokio::test]
async fn auto_skill_explicit_model_and_lifecycle_strategies_do_not_replay_script() {
    for lifecycle in [false, true] {
        let yaml = SKILL.replace("      mode: skill_until_complete", if lifecycle {
            "      mode: skill_until_complete\n      execution: {strategy: lifecycle}\n      lifecycle: [{id: review, instruction: Review}]"
        } else { "      mode: skill_until_complete\n      execution: {strategy: model_led}" });
        let main = RecordingProvider::new(&["done", "done"]);
        let router = RecordingProvider::new(&["work"]);
        let agent = agent(&yaml, main.clone(), router.clone());
        agent.chat("work").await.unwrap();
        let result = agent.automatic_task_result().unwrap();
        assert_eq!(
            result.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            result.run.stop_reason
        );
        assert_eq!(router.calls.load(Ordering::SeqCst), 1);
        assert!(
            !main
                .captured_messages()
                .iter()
                .flatten()
                .any(|message| message.content.starts_with("Research work"))
        );
    }
}

// Original clocks and bound decisions survive both volatile and SQL pause storage; stop does not invoke pending work.
#[tokio::test]
async fn auto_pause_stop_cancel_and_stale_controls_are_truthful_in_both_backends() {
    for &sqlite in support::backends() {
        for cancel in [false, true] {
            let store = support::store("auto".into(), "automatic.v1", sqlite).await;
            let yaml = SKILL.replace("      completion:", "      premature_finish: {action: ask_user, on_unavailable: stop}\n      completion:");
            let main = RecordingProvider::new(&["draft", "must not run"]);
            let agent = AgentBuilder::from_yaml(&yaml)
                .unwrap()
                .llm(main.clone())
                .llm_alias("router", RecordingProvider::new(&["work"]))
                .automatic_task_options(AutomaticTaskOptions {
                    store: Some(store.clone()),
                    config_identity: Some("automatic.v1".into()),
                    ..Default::default()
                })
                .build()
                .unwrap();
            agent.chat("work").await.unwrap();
            let paused = agent.automatic_task_result().unwrap();
            assert_eq!(paused.run.status, TaskRunStatus::Paused);
            let operation = if cancel {
                TaskStreamOperation::Cancel {
                    run_id: paused.run.key.run_id.clone(),
                    revision: paused.run.revision,
                }
            } else {
                TaskStreamOperation::Stop {
                    run_id: paused.run.key.run_id.clone(),
                    revision: paused.run.revision,
                    reason: "host stop".into(),
                }
            };
            let stopped = agent.automatic_task_operation(operation).await.unwrap();
            assert_eq!(
                stopped.run.status,
                if cancel {
                    TaskRunStatus::Cancelled
                } else {
                    TaskRunStatus::Incomplete
                }
            );
            assert_eq!(main.calls.load(Ordering::SeqCst), 1);
            assert_eq!(
                store
                    .load(&stopped.run.key.run_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .status,
                stopped.run.status
            );
            assert!(
                agent
                    .automatic_task_operation(TaskStreamOperation::Resume {
                        run_id: paused.run.key.run_id,
                        revision: paused.run.revision,
                        input: TaskResumeInput::Continue
                    })
                    .await
                    .is_err()
            );
        }
    }
}

// Public ordinary SDK composition keeps its live parent frame and never treats a paused child as completed stage output.
#[tokio::test]
async fn auto_ordinary_pipeline_waits_for_host_without_replaying_prior_stage() {
    use crate::orchestration::{PipelineStage, pipeline};
    use crate::spawner::{AgentRegistry, SpawnedAgent};
    let registry = Arc::new(AgentRegistry::new());
    let first = RecordingProvider::new(&["first result"]);
    let first_spec =
        crate::spec::AgentSpec::from_yaml_strict("name: first\nsystem_prompt: First\n").unwrap();
    let first_agent = AgentBuilder::from_spec(first_spec.clone())
        .llm(first.clone())
        .build()
        .unwrap();
    registry
        .register(SpawnedAgent::from_runtime(
            "first".into(),
            first_agent,
            first_spec,
        ))
        .await
        .unwrap();
    let yaml = STATE.replace(
        "        completion:",
        "        premature_finish: {action: ask_user, on_unavailable: stop}\n        completion:",
    );
    let child_spec = crate::spec::AgentSpec::from_yaml_strict(&yaml).unwrap();
    let child = agent(
        &yaml,
        RecordingProvider::new(&["draft", "done"]),
        RecordingProvider::new(&[]),
    );
    registry
        .register(SpawnedAgent::from_runtime(
            "child".into(),
            child,
            child_spec,
        ))
        .await
        .unwrap();
    let child = registry.get("child").unwrap();
    let stages = vec![PipelineStage::id("first"), PipelineStage::id("child")];
    let work = pipeline(&registry, "objective", &stages, Some(30_000), None, None);
    tokio::pin!(work);
    let paused = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            tokio::select! {
                result = &mut work => panic!("pipeline returned before child host decision: {result:?}"),
                _ = tokio::task::yield_now() => {
                    if let Some(paused) = child.automatic_task_result() { break paused; }
                }
            }
        }
    }).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    let pending = paused.run.pending.unwrap();
    let done = child
        .automatic_task_operation(TaskStreamOperation::Resume {
            run_id: paused.run.key.run_id,
            revision: paused.run.revision,
            input: TaskResumeInput::UserAnswer {
                request_id: pending.id,
                answer: json!("continue"),
            },
        })
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    let result = work.await.unwrap();
    assert_eq!(result.response.content, "done");
    assert_eq!(first.calls.load(Ordering::SeqCst), 1);
}

// Reconstructed automatic entries retain their declaring scope and exact skill prefix rather than infer ownership from current state.
#[tokio::test]
async fn auto_prepared_reconstruction_resumes_state_and_skill_without_new_entry() {
    for &sqlite in support::backends() {
        for skill in [false, true] {
            let yaml = if skill {
                SKILL.replace("      completion:", "      premature_finish: {action: ask_user, on_unavailable: stop}\n      completion:")
            } else {
                STATE.replace("        completion:", "        premature_finish: {action: ask_user, on_unavailable: stop}\n        completion:")
            };
            let main = RecordingProvider::new(&["draft", "done"]);
            let router = RecordingProvider::new(&["work"]);
            let store = support::store("auto".into(), "automatic.v1", sqlite).await;
            let agent = Arc::new(
                AgentBuilder::from_yaml(&yaml)
                    .unwrap()
                    .llm(main.clone())
                    .llm_alias("router", router.clone())
                    .automatic_task_options(AutomaticTaskOptions {
                        store: Some(store.clone()),
                        config_identity: Some("automatic.v1".into()),
                        ..Default::default()
                    })
                    .build()
                    .unwrap(),
            );
            agent.chat("request").await.unwrap();
            let initial = agent.automatic_task_result().unwrap();
            assert_eq!(initial.run.status, TaskRunStatus::Paused);
            let runner = agent.automatic_task_runner().unwrap();
            let source =
                FreshRuntime::new(runner.runtime().clone(), vec![main.clone(), router.clone()]);
            let fresh_yaml = yaml.clone();
            let host = RebuildHost::new(
                "automatic.fixture.v1",
                &source,
                Box::new(move || {
                    let yaml = fresh_yaml.clone();
                    Box::pin(async move {
                        let main = RecordingProvider::new(&[]);
                        let router = RecordingProvider::new(&[]);
                        let fresh = super::agent(&yaml, main.clone(), router.clone());
                        FreshRuntime::new(Arc::new(fresh), vec![main, router])
                    })
                }),
            );
            drop(source);
            drop(agent);
            let runner = Arc::try_unwrap(runner)
                .unwrap_or_else(|_| panic!("source host must release its controller"));
            let (restored, paused) = restore(
                runner,
                store.clone(),
                &initial.run,
                AutonomyConfig::default(),
                "automatic.v1",
                host,
            )
            .await;
            assert_eq!(main.calls.load(Ordering::SeqCst), 1);
            let done = restored
                .resume(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    TaskResumeInput::UserAnswer {
                        request_id: paused.run.pending.unwrap().id,
                        answer: json!("continue"),
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
            assert_eq!(
                done.run.counters.llm_attempts,
                initial.run.counters.llm_attempts + 1
            );
            assert_eq!(main.calls.load(Ordering::SeqCst), 1);
        }
    }
}

// Foreground interruption blocks further admission but does not fabricate cleanup or permit ordinary reuse.
#[tokio::test]
async fn auto_active_drop_retains_owner_and_rejects_competing_chat() {
    let provider = RecordingProvider::gated();
    let agent = Arc::new(agent(STATE, provider.clone(), RecordingProvider::new(&[])));
    let mut work = Box::pin(agent.chat("work"));
    tokio::select! {
        result = &mut work => panic!("gated work unexpectedly returned: {result:?}"),
        permit = provider.entered.as_ref().unwrap().acquire() => permit.unwrap().forget(),
    }
    assert!(agent.chat("competing").await.is_err());
    drop(work);
    assert!(agent.chat("after interruption").await.is_err());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

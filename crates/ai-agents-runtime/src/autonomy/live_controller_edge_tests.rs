use super::*;

#[path = "live_controller_race_tests.rs"]
mod race_tests;

#[tokio::test]
async fn live_skill_tool_free_profile_inheritance_and_required_stages_share_one_controller() {
    for sqlite in support::backends() {
        let provider =
            super::super::super::runner_tests::RecordingProvider::new(&["sources", "done", "done"]);
        let agent = Arc::new(AgentBuilder::from_yaml("name: research\nsystem_prompt: research\nskills:\n  - id: report\n    description: research report\n    trigger: report\n    autonomy: {max_llm_calls: 3}\n    steps:\n      - prompt: gather sources\n      - prompt: synthesize {{ steps[0].result }}\n").unwrap().llm(provider.clone()).auto_configure_features().unwrap().build().unwrap());
        let store = support::store(agent.info().id, "research.v1", *sqlite).await;
        let mut config = super::super::super::runner_tests::config(9);
        config.defaults.lifecycle =
            Some(serde_yaml::from_str("- id: research\n- id: review\n").unwrap());
        let runner = live_runner(agent.clone(), store, config, "research.v1");
        assert!(runner.run_skill("research", "missing", None).await.is_err());
        assert!(agent.set_context("not_reserved", json!(true)).is_ok());
        let done = runner.run_skill("research", "report", None).await.unwrap();
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
        assert_eq!(done.run.counters.llm_attempts, 3);
        assert_eq!(done.run.counters.tool_attempts, 0);
        assert_eq!(done.run.counters.turns, 2);
    }
}

// Native markers bind real call IDs even when the recording provider uses the prompt protocol.
fn call(id: &str, tool: &str) -> String {
    ai_agents_core::encode_native_tool_call_markers(
        &[ToolCall {
            id: id.into(),
            name: tool.into(),
            arguments: json!({}),
        }],
        None,
    )
    .unwrap()
}

#[tokio::test]
async fn live_controller_native_provider_capacity_retains_iterations_and_accumulated_calls() {
    for sqlite in support::backends() {
        for exhaust_iterations in [false, true] {
            let first = call("original-first", "first");
            let second = call("original-second", "first");
            let provider = super::super::super::runner_tests::RecordingProvider::new(&[
                &first,
                if exhaust_iterations { &second } else { "done" },
                "done",
            ]);
            let calls = Arc::new(AtomicUsize::new(0));
            let agent = Arc::new(
                AgentBuilder::from_yaml(
                    "name: native\nsystem_prompt: test\nmax_iterations: 2\ntools: [first]\n",
                )
                .unwrap()
                .llm(provider.clone())
                .tool(Arc::new(Probe {
                    id: "first",
                    calls: calls.clone(),
                    fail: false,
                }))
                .build()
                .unwrap(),
            );
            let store = support::store(agent.info().id, "native.v1", *sqlite).await;
            let mut config = super::super::super::runner_tests::config(1);
            config.defaults.on_limit = Some(serde_yaml::from_str("action: pause_run").unwrap());
            let runner = live_runner(agent, store.clone(), config, "native.v1");
            let paused = runner.run("objective", None).await.unwrap();
            assert_eq!(paused.run.status, TaskRunStatus::Paused);
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            let payload: TaskCheckpointPayload = serde_json::from_value(
                store
                    .load(&paused.run.key.run_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .payload,
            )
            .unwrap();
            let mut limits = payload.limits;
            limits.max_llm_calls = 4;
            let revised = runner
                .revise_limits_paused(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    limits,
                    "continue same model loop",
                )
                .await
                .unwrap();
            let done = runner
                .resume(
                    &revised.run.key.run_id,
                    revised.run.revision,
                    TaskResumeInput::Continue,
                )
                .await
                .unwrap();
            assert_eq!(done.run.counters.turns, 1);
            assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
            if exhaust_iterations {
                assert_eq!(done.run.status, TaskRunStatus::Failed);
                assert_eq!(calls.load(Ordering::SeqCst), 2);
            } else {
                assert_eq!(
                    done.run.status,
                    TaskRunStatus::Completed,
                    "{:?}",
                    done.run.stop_reason
                );
                let response = done.final_response.unwrap();
                assert_eq!(response.tool_calls.as_ref().unwrap().len(), 1);
                assert_eq!(
                    response.tool_calls.as_ref().unwrap()[0].id,
                    "original-first"
                );
                assert_eq!(calls.load(Ordering::SeqCst), 1);
            }
        }
    }
}

#[tokio::test]
async fn live_controller_native_tool_capacity_resumes_selected_call_without_model_regeneration() {
    for sqlite in support::backends() {
        for action in ["pause_run", "ask_user"] {
            let first = call("first-call", "first");
            let second = call("pending-call", "second");
            let provider = super::super::super::runner_tests::RecordingProvider::new(&[
                &first, &second, "done",
            ]);
            let first_calls = Arc::new(AtomicUsize::new(0));
            let second_calls = Arc::new(AtomicUsize::new(0));
            let agent = Arc::new(
                AgentBuilder::from_yaml(
                    "name: native\nsystem_prompt: test\ntools: [first, second]\n",
                )
                .unwrap()
                .llm(provider.clone())
                .tool(Arc::new(Probe {
                    id: "first",
                    calls: first_calls.clone(),
                    fail: false,
                }))
                .tool(Arc::new(Probe {
                    id: "second",
                    calls: second_calls.clone(),
                    fail: false,
                }))
                .build()
                .unwrap(),
            );
            let store = support::store(agent.info().id, "native.v1", *sqlite).await;
            let mut config = super::super::super::runner_tests::config(4);
            config.defaults.max_tool_calls = Some(1);
            config.defaults.on_limit =
                Some(serde_yaml::from_str(&format!("action: {action}")).unwrap());
            let runner = live_runner(agent.clone(), store.clone(), config, "native.v1");
            let paused = runner.run("objective", None).await.unwrap();
            assert_eq!(
                paused.run.status,
                TaskRunStatus::Paused,
                "{:?}",
                paused.run.stop_reason
            );
            assert!(matches!(
                paused.run.pending.as_ref().unwrap().kind,
                TaskPendingKind::Limit
            ));
            assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
            assert_eq!(first_calls.load(Ordering::SeqCst), 1);
            assert_eq!(second_calls.load(Ordering::SeqCst), 0);
            let payload: TaskCheckpointPayload = serde_json::from_value(
                store
                    .load(&paused.run.key.run_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .payload,
            )
            .unwrap();
            let mut limits = payload.limits;
            limits.max_tool_calls = 2;
            let revised = runner
                .revise_limits_paused(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    limits,
                    "admit selected operation",
                )
                .await
                .unwrap();
            let response = if action == "pause_run" {
                TaskResumeInput::Continue
            } else {
                answer(&revised.run)
            };
            let done = runner
                .resume(&revised.run.key.run_id, revised.run.revision, response)
                .await
                .unwrap();
            assert_eq!(
                done.run.status,
                TaskRunStatus::Completed,
                "{:?}",
                done.run.stop_reason
            );
            assert_eq!(done.run.counters.turns, 1);
            assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
            assert_eq!(first_calls.load(Ordering::SeqCst), 1);
            assert_eq!(second_calls.load(Ordering::SeqCst), 1);
            let response = done.final_response.unwrap();
            assert_eq!(response.tool_calls.as_ref().unwrap().len(), 2);
            assert_eq!(response.tool_calls.as_ref().unwrap()[1].id, "pending-call");
            assert!(agent.set_context("released", json!(true)).is_ok());
        }
    }
}

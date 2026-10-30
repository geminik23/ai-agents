use super::*;
use crate::AgentBuilder;
use crate::spawner::{AgentRegistry, AgentSpawner, SpawnedAgent};
use serde_json::json;
use std::sync::{Arc, atomic::Ordering};

// Production model tools invoke registered runtimes under the same task ledger and exact algorithm cursors.
async fn fixture(
    concurrent: bool,
) -> (
    AutonomyRunner,
    Arc<crate::RuntimeAgent>,
    Arc<super::runner_tests::RecordingProvider>,
    Vec<Arc<super::runner_tests::RecordingProvider>>,
) {
    Box::pin(fixture_repeated(concurrent, false)).await
}

// Repeated calls share actual targets but distinct parent attempts and incoming child histories.
async fn fixture_repeated(
    concurrent: bool,
    repeated: bool,
) -> (
    AutonomyRunner,
    Arc<crate::RuntimeAgent>,
    Arc<super::runner_tests::RecordingProvider>,
    Vec<Arc<super::runner_tests::RecordingProvider>>,
) {
    Box::pin(fixture_kind(
        if concurrent {
            "concurrent_ask"
        } else {
            "pipeline_process"
        },
        repeated,
    ))
    .await
}

// Every model-callable composition selects its production algorithm and actual captured auxiliary providers.
async fn fixture_kind(
    tool: &str,
    repeated: bool,
) -> (
    AutonomyRunner,
    Arc<crate::RuntimeAgent>,
    Arc<super::runner_tests::RecordingProvider>,
    Vec<Arc<super::runner_tests::RecordingProvider>>,
) {
    let registry = Arc::new(AgentRegistry::new());
    let mut providers = Vec::new();
    for id in ["first", "second"] {
        let child_yaml = format!("name: {id}\nsystem_prompt: child\ntools: [ask_user]\n");
        let question = ai_agents_core::encode_native_tool_call_markers(
            &[ai_agents_core::ToolCall {
                id: format!("{id}-question"),
                name: "ask_user".into(),
                arguments: json!({"question":id,"options":["yes"],"allow_other":false}),
            }],
            None,
        )
        .unwrap();
        let provider = if tool != "pipeline_process" || id == "second" {
            if repeated {
                super::runner_tests::RecordingProvider::new(&[&question, "done", &question, "done"])
            } else {
                super::runner_tests::RecordingProvider::new(&[&question, "done"])
            }
        } else {
            super::runner_tests::RecordingProvider::new(&["first done"])
        };
        let child = AgentBuilder::from_yaml(&child_yaml)
            .unwrap()
            .llm(provider.clone())
            .auto_configure_features()
            .unwrap()
            .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
                Arc::new(parking_lot::RwLock::new(None)),
            )))
            .build()
            .unwrap();
        registry
            .register(SpawnedAgent::from_runtime(
                id.into(),
                child,
                crate::spec::AgentSpec::from_yaml_strict(&child_yaml).unwrap(),
            ))
            .await
            .unwrap();
        providers.push(provider);
    }
    let args = if tool == "group_discussion" {
        json!({"topic":"objective","participants":["first","second"],"style":"brainstorm","max_rounds":1})
    } else if tool == "handoff_conversation" {
        json!({"input":"objective","initial_agent":"first","available_agents":["first","second"],"max_handoffs":1})
    } else if tool == "concurrent_ask" {
        json!({"question":"objective","agents":["first","second"],"aggregation":"all"})
    } else {
        json!({"input":"objective","stages":["first","second"],"stage_inputs":[null,"{{ previous_output }}"]})
    };
    let mut calls = vec![ai_agents_core::ToolCall {
        id: "composition-call".into(),
        name: tool.into(),
        arguments: args,
    }];
    if repeated {
        let mut second = calls[0].clone();
        second.id = "composition-call-second".into();
        calls.push(second);
    }
    let marker = ai_agents_core::encode_native_tool_call_markers(&calls, None).unwrap();
    let main = super::runner_tests::RecordingProvider::new(&[&marker, "done"]);
    let mut llms = ai_agents_llm::LLMRegistry::new();
    llms.register("default", main.clone());
    llms.register(
        "router",
        super::runner_tests::RecordingProvider::new(&[
            r#"{"action":"second","confidence":1.0,"reason":"next"}"#,
            r#"{"action":"stay","confidence":1.0,"reason":"done"}"#,
            "synthesis",
        ]),
    );
    let llms = Arc::new(llms);
    let yaml = format!("name: Parent\nsystem_prompt: parent\ntools: [{tool}]\n");
    let builder = AgentBuilder::from_yaml(&yaml)
        .unwrap()
        .llm_registry((*llms).clone())
        .auto_configure_features()
        .unwrap();
    let builder = if tool == "group_discussion" {
        builder.tool(Arc::new(
            crate::orchestration::tools::GroupDiscussionTool::new(registry.clone(), llms),
        ))
    } else if tool == "handoff_conversation" {
        builder.tool(Arc::new(
            crate::orchestration::tools::HandoffConversationTool::new(registry.clone(), llms),
        ))
    } else if tool == "concurrent_ask" {
        builder.tool(Arc::new(
            crate::orchestration::tools::ConcurrentAskTool::new(registry.clone(), llms),
        ))
    } else {
        builder.tool(Arc::new(
            crate::orchestration::tools::PipelineProcessTool::new(registry.clone()),
        ))
    };
    let parent = Arc::new(
        builder
            .build()
            .unwrap()
            .with_spawner_handles(Arc::new(AgentSpawner::new()), registry),
    );
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(parent.info().id, None, "tool-composition.v1".into())
            .unwrap(),
    );
    let mut config = super::runner_tests::config(8);
    config.defaults.max_tool_calls = Some(if repeated { 6 } else { 3 });
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    let runner = AutonomyRunner::try_new(
        parent.clone(),
        config,
        Default::default(),
        store,
        "tool-composition.v1".into(),
    )
    .unwrap();
    (runner, parent, main, providers)
}

#[tokio::test]
async fn tool_composition_pipeline_resumes_stage_without_replaying_completed_stage() {
    let (runner, _, main, children) = Box::pin(fixture(false)).await;
    let first = runner.run("objective", None).await.unwrap();
    assert_eq!(
        first.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        first.run.stop_reason
    );
    assert_eq!(children[0].calls.load(Ordering::SeqCst), 1);
    assert_eq!(children[1].calls.load(Ordering::SeqCst), 1);
    let done = runner
        .resume(
            &first.run.key.run_id,
            first.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: first.run.pending.as_ref().unwrap().id.clone(),
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
    assert_eq!(children[0].calls.load(Ordering::SeqCst), 1);
    assert_eq!(children[1].calls.load(Ordering::SeqCst), 2);
    assert_eq!(main.calls.load(Ordering::SeqCst), 2);
    assert_eq!(done.run.counters.tool_attempts, 2);
}

#[tokio::test]
async fn tool_composition_concurrent_resumes_multiple_leaves_without_duplicate_admission() {
    let (runner, _, main, children) = Box::pin(fixture(true)).await;
    let mut result = runner.run("objective", None).await.unwrap();
    let mut pauses = 0;
    while result.run.status == TaskRunStatus::Paused {
        pauses += 1;
        assert!(pauses <= 2);
        result = runner
            .resume(
                &result.run.key.run_id,
                result.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: result.run.pending.as_ref().unwrap().id.clone(),
                    answer: json!({"answered":true,"selected":["yes"]}),
                },
            )
            .await
            .unwrap();
    }
    assert_eq!(
        result.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        result.run.stop_reason
    );
    assert_eq!(pauses, 2);
    assert_eq!(main.calls.load(Ordering::SeqCst), 2);
    assert!(
        children
            .iter()
            .all(|child| child.calls.load(Ordering::SeqCst) == 2)
    );
    assert_eq!(result.run.counters.tool_attempts, 3);
}

#[tokio::test]
async fn tool_composition_repeated_catalogue_defers_root_history_until_previous_operation_completes()
 {
    let (runner, _, main, children) = Box::pin(fixture_repeated(false, true)).await;
    let first = runner.run("objective", None).await.unwrap();
    assert_eq!(
        first.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        first.run.stop_reason
    );
    assert_eq!(children[0].calls.load(Ordering::SeqCst), 1);
    assert_eq!(children[1].calls.load(Ordering::SeqCst), 1);
    let second = runner
        .resume(
            &first.run.key.run_id,
            first.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: first.run.pending.as_ref().unwrap().id.clone(),
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await
        .unwrap();
    assert_eq!(
        second.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        second.run.stop_reason
    );
    assert_eq!(children[0].calls.load(Ordering::SeqCst), 2);
    assert_eq!(children[1].calls.load(Ordering::SeqCst), 3);
    let done = runner
        .resume(
            &second.run.key.run_id,
            second.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: second.run.pending.as_ref().unwrap().id.clone(),
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
    assert_eq!(children[0].calls.load(Ordering::SeqCst), 2);
    assert_eq!(children[1].calls.load(Ordering::SeqCst), 4);
    assert_eq!(main.calls.load(Ordering::SeqCst), 2);
    assert_eq!(done.run.counters.tool_attempts, 4);
}

#[tokio::test]
async fn tool_composition_group_and_handoff_resume_saved_speaker_positions() {
    for tool in ["group_discussion", "handoff_conversation"] {
        let (runner, _, main, children) = Box::pin(fixture_kind(tool, false)).await;
        let outcome = runner.run("objective", None).await;
        assert!(outcome.is_ok(), "{tool} {:?}", outcome.as_ref().err());
        let mut result = outcome.unwrap();
        let mut pauses = 0;
        while result.run.status == TaskRunStatus::Paused {
            pauses += 1;
            assert!(pauses <= 2, "{tool}");
            result = runner
                .resume(
                    &result.run.key.run_id,
                    result.run.revision,
                    TaskResumeInput::UserAnswer {
                        request_id: result.run.pending.as_ref().unwrap().id.clone(),
                        answer: json!({"answered":true,"selected":["yes"]}),
                    },
                )
                .await
                .unwrap();
        }

        assert_eq!(
            result.run.status,
            TaskRunStatus::Completed,
            "{tool} {:?}",
            result.run.stop_reason
        );

        assert_eq!(pauses, 2, "{tool}");
        assert_eq!(main.calls.load(Ordering::SeqCst), 2);
        assert!(
            children
                .iter()
                .all(|child| child.calls.load(Ordering::SeqCst) == 2)
        );
        assert_eq!(result.run.counters.tool_attempts, 3);
    }
}

#[tokio::test]
async fn tool_composition_cancel_and_host_stop_never_invoke_pending_children() {
    for concurrent in [false, true] {
        for stop in [false, true] {
            let (runner, parent, main, children) = Box::pin(fixture(concurrent)).await;
            let paused = runner.run("objective", None).await.unwrap();
            assert_eq!(paused.run.status, TaskRunStatus::Paused);
            let calls: Vec<_> = children
                .iter()
                .map(|child| child.calls.load(Ordering::SeqCst))
                .collect();
            let done = if stop {
                runner
                    .stop_paused(
                        &paused.run.key.run_id,
                        paused.run.revision,
                        "Stop pending composition",
                    )
                    .await
                    .unwrap()
            } else {
                runner
                    .cancel_paused(&paused.run.key.run_id, paused.run.revision)
                    .await
                    .unwrap()
            };
            assert_eq!(
                done.run.status,
                if stop {
                    TaskRunStatus::Incomplete
                } else {
                    TaskRunStatus::Cancelled
                }
            );
            assert_eq!(main.calls.load(Ordering::SeqCst), 1);
            assert_eq!(
                children
                    .iter()
                    .map(|child| child.calls.load(Ordering::SeqCst))
                    .collect::<Vec<_>>(),
                calls
            );
            assert!(parent.set_context("released", json!(true)).is_ok());
        }
    }
}

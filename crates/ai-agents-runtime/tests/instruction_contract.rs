//! Actual provider requests exercise scoped instructions, work loops and retained task continuation.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use ai_agents_context::{ContextManager, ContextSource, ScopedInstructionFiles};
use ai_agents_core::{
    FinishReason, LLMFeature, LLMResponse, Tool, ToolCall, ToolChoice, ToolExecutionContext,
    ToolResult, ToolSafetyMetadata,
};
use ai_agents_llm::mock::MockLLMProvider;
use ai_agents_runtime::{Agent, AgentBuilder, autonomy::*};
use async_trait::async_trait;
use serde_json::{Value, json};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let root =
            std::env::temp_dir().join(format!("instruction-runtime-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("AGENTS.md"), "root-v1").unwrap();
        std::fs::write(root.join("src/AGENTS.md"), "nested-v1").unwrap();
        Self(root)
    }
    fn manager(&self) -> Arc<ContextManager> {
        let manager = Arc::new(ContextManager::new(
            HashMap::from([(
                "rules".into(),
                ContextSource::Instructions {
                    name: "host-rules".into(),
                },
            )]),
            "fixture".into(),
            "1".into(),
        ));
        let source =
            ScopedInstructionFiles::new(&self.0, &["src/code.rs".into()], &["AGENTS.md".into()])
                .unwrap();
        manager.register_provider("host-rules", Arc::new(source));
        manager
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn mock(replies: Vec<LLMResponse>) -> Arc<MockLLMProvider> {
    let mut provider = MockLLMProvider::new("instructions-test");
    provider.set_tool_choice(Some(ToolChoice::Auto));
    provider.set_feature_support(LLMFeature::FunctionCalling, true);
    for reply in replies {
        provider.add_response(reply);
    }
    Arc::new(provider)
}
fn text(call: &ai_agents_llm::mock::MockCall) -> String {
    call.messages
        .iter()
        .map(|message| message.content.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

struct UpdateRules {
    path: PathBuf,
    calls: Arc<AtomicUsize>,
}
#[async_trait]
impl Tool for UpdateRules {
    fn id(&self) -> &str {
        "update_rules_fixture"
    }
    fn name(&self) -> &str {
        "update_rules_fixture"
    }
    fn description(&self) -> &str {
        "Update a disposable host fixture"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object"})
    }
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata::default()
    }
    async fn execute(&self, _: Value, _: ToolExecutionContext) -> ToolResult {
        std::fs::write(&self.path, "root-v2").unwrap();
        self.calls.fetch_add(1, Ordering::SeqCst);
        ToolResult::ok("updated")
    }
}

#[tokio::test]
async fn same_turn_replaces_instructions_after_actual_tool_work() {
    let fixture = Fixture::new();
    let tool_call = ToolCall {
        id: "update".into(),
        name: "update_rules_fixture".into(),
        arguments: json!({}),
    };
    let provider = mock(vec![
        LLMResponse::new("", FinishReason::ToolCall)
            .with_tool_calls(vec![tool_call])
            .unwrap(),
        LLMResponse::new("done", FinishReason::Stop),
    ]);
    let calls = Arc::new(AtomicUsize::new(0));
    let agent = AgentBuilder::from_yaml(
        "name: instruction-loop\nsystem_prompt: base\ntools: [update_rules_fixture]\n",
    )
    .unwrap()
    .llm(provider.clone())
    .context_manager(fixture.manager())
    .auto_configure_features()
    .unwrap()
    .tool(Arc::new(UpdateRules {
        path: fixture.0.join("AGENTS.md"),
        calls: calls.clone(),
    }))
    .build()
    .unwrap();
    agent.chat("work").await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let history = provider.call_history();
    assert_eq!(history.len(), 2);
    let first = text(&history[0]);
    let second = text(&history[1]);
    assert!(first.contains("root-v1") && first.contains("nested-v1"));
    assert!(second.contains("root-v2") && second.contains("nested-v1"));
    assert!(!second.contains("root-v1"));
    assert_eq!(second.matches("nested-v1").count(), 1);
}

#[tokio::test]
async fn updated_instructions_preserve_signed_native_tool_history() {
    let fixture = Fixture::new();
    let call = ToolCall {
        id: "signed-call".into(),
        name: "update_rules_fixture".into(),
        arguments: json!({}),
    };
    let state = ai_agents_core::NativeProviderState::new(
        "signed-exchange", "fixture", "native-tools",
        ai_agents_core::NativeProviderTarget::new("https://fixture.invalid/", "fixture-model").unwrap(),
        json!({"role":"model","parts":[{"functionCall":{"name":"update_rules_fixture","args":{}},"thoughtSignature":"fixture-signature"}]}),
        vec![ai_agents_core::NativeCallBinding::new("signed-call", 0).unwrap()],
    ).unwrap();
    let response = LLMResponse::new("", FinishReason::ToolCall)
        .with_provider_state(state)
        .unwrap()
        .with_tool_calls(vec![call])
        .unwrap();
    let provider = mock(vec![response, LLMResponse::new("done", FinishReason::Stop)]);
    let calls = Arc::new(AtomicUsize::new(0));
    let agent = AgentBuilder::from_yaml(
        "name: signed-instructions\nsystem_prompt: base\ntools: [update_rules_fixture]\n",
    )
    .unwrap()
    .llm(provider.clone())
    .context_manager(fixture.manager())
    .auto_configure_features()
    .unwrap()
    .tool(Arc::new(UpdateRules {
        path: fixture.0.join("AGENTS.md"),
        calls: calls.clone(),
    }))
    .build()
    .unwrap();
    agent.chat("work").await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let history = provider.call_history();
    assert_eq!(history.len(), 2);
    assert!(text(&history[1]).contains("root-v2"));
    assert!(!text(&history[1]).contains("root-v1"));
    assert!(history[1].messages.iter().any(|message| {
        ai_agents_core::decode_native_tool_call_markers(&message.content)
            .is_ok_and(|value| value.is_some())
    }));
    assert!(history[1].messages.iter().any(|message| {
        ai_agents_core::decode_native_tool_result_markers(&message.content)
            .is_ok_and(|value| value.is_some())
    }));
}

#[tokio::test]
async fn oversized_signed_exchange_is_not_truncated_to_fit_instructions() {
    let fixture = Fixture::new();
    let call = ToolCall {
        id: "oversized-call".into(),
        name: "update_rules_fixture".into(),
        arguments: json!({}),
    };
    let state = ai_agents_core::NativeProviderState::new(
        "oversized-exchange", "fixture", "native-tools",
        ai_agents_core::NativeProviderTarget::new("https://fixture.invalid/", "fixture-model").unwrap(),
        json!({"role":"model","parts":[{"functionCall":{"name":"update_rules_fixture","args":{}},"thoughtSignature":"x".repeat(4000)}]}),
        vec![ai_agents_core::NativeCallBinding::new("oversized-call", 0).unwrap()],
    ).unwrap();
    let response = LLMResponse::new("", FinishReason::ToolCall)
        .with_provider_state(state)
        .unwrap()
        .with_tool_calls(vec![call])
        .unwrap();
    let provider = mock(vec![
        response,
        LLMResponse::new("must not run", FinishReason::Stop),
    ]);
    let calls = Arc::new(AtomicUsize::new(0));
    let agent = AgentBuilder::from_yaml(
        "name: signed-budget\nsystem_prompt: base\ntools: [update_rules_fixture]\n",
    )
    .unwrap()
    .llm(provider.clone())
    .context_manager(fixture.manager())
    .max_context_tokens(256)
    .auto_configure_features()
    .unwrap()
    .tool(Arc::new(UpdateRules {
        path: fixture.0.join("AGENTS.md"),
        calls: calls.clone(),
    }))
    .build()
    .unwrap();
    assert!(agent.chat("work").await.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.call_count(), 1);
}

#[tokio::test]
async fn deleted_rules_are_removed_and_invalid_rules_prevent_dispatch() {
    let fixture = Fixture::new();
    let provider = mock(vec![
        LLMResponse::new("first", FinishReason::Stop),
        LLMResponse::new("second", FinishReason::Stop),
    ]);
    let agent = AgentBuilder::new()
        .system_prompt("base")
        .llm(provider.clone())
        .context_manager(fixture.manager())
        .build()
        .unwrap();
    agent.chat("first").await.unwrap();
    std::fs::write(fixture.0.join("src/AGENTS.md"), [0xff]).unwrap();
    assert!(agent.chat("invalid").await.is_err());
    assert_eq!(provider.call_count(), 1);
    std::fs::remove_file(fixture.0.join("src/AGENTS.md")).unwrap();
    std::fs::remove_file(fixture.0.join("AGENTS.md")).unwrap();
    agent.chat("second").await.unwrap();
    let second = text(&provider.call_history()[1]);
    assert!(!second.contains("root-v1"));
    assert!(!second.contains("nested-v1"));
}

#[tokio::test]
async fn absent_live_binding_fails_before_provider() {
    let provider = mock(vec![LLMResponse::new("not called", FinishReason::Stop)]);
    let manager = Arc::new(ContextManager::new(
        HashMap::from([(
            "rules".into(),
            ContextSource::Instructions {
                name: "missing".into(),
            },
        )]),
        "test".into(),
        "1".into(),
    ));
    let agent = AgentBuilder::new()
        .system_prompt("base")
        .llm(provider.clone())
        .context_manager(manager)
        .build()
        .unwrap();
    assert!(agent.chat("work").await.is_err());
    assert_eq!(provider.call_count(), 0);
}

struct WaitingInstructions {
    started: Arc<tokio::sync::Notify>,
    dropped: Arc<std::sync::atomic::AtomicBool>,
}
#[async_trait]
impl ai_agents_context::ContextProvider for WaitingInstructions {
    fn task_instruction_safe(&self) -> bool {
        true
    }
    async fn get(&self, _: &str, _: &Value) -> ai_agents_core::Result<Value> {
        struct DropWitness(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for DropWitness {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let _witness = DropWitness(self.dropped.clone());
        self.started.notify_one();
        std::future::pending().await
    }
}

#[tokio::test]
async fn task_cancellation_interrupts_a_certified_instruction_wait() {
    let started = Arc::new(tokio::sync::Notify::new());
    let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let manager = Arc::new(ContextManager::new(
        HashMap::from([(
            "rules".into(),
            ContextSource::Instructions {
                name: "wait".into(),
            },
        )]),
        "test".into(),
        "1".into(),
    ));
    manager.register_provider(
        "wait",
        Arc::new(WaitingInstructions {
            started: started.clone(),
            dropped: dropped.clone(),
        }),
    );
    let provider = mock(vec![LLMResponse::new("not called", FinishReason::Stop)]);
    let agent = Arc::new(
        AgentBuilder::new()
            .system_prompt("base")
            .llm(provider.clone())
            .context_manager(manager)
            .build()
            .unwrap(),
    );
    let config: AutonomyConfig =
        serde_yaml::from_str("enabled: true\ncompletion: {response_not_empty: true}\n").unwrap();
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "instruction-cancel.v1".into())
            .unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        Default::default(),
        store.clone(),
        "instruction-cancel.v1".into(),
    )
    .unwrap();
    let cancel = async {
        started.notified().await;
        let runs = store.list().await.unwrap();
        assert_eq!(runs.len(), 1);
        runner
            .cancel_task(&runs[0].key.run_id, runs[0].revision)
            .await
            .unwrap()
    };
    let (result, _receipt) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        tokio::join!(runner.run("work", None), cancel)
    })
    .await
    .expect("cancellation must not hang");
    if let Ok(result) = result {
        assert_ne!(result.run.status, TaskRunStatus::Completed);
    }
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(provider.call_count(), 0);
}

struct UncertifiedInstructions;
#[async_trait]
impl ai_agents_context::ContextProvider for UncertifiedInstructions {
    async fn get(&self, _: &str, _: &Value) -> ai_agents_core::Result<Value> {
        Ok(json!("untrusted provider"))
    }
}

#[tokio::test]
async fn task_rejects_instruction_provider_without_safe_participation() {
    let provider = mock(vec![LLMResponse::new("not called", FinishReason::Stop)]);
    let manager = Arc::new(ContextManager::new(
        HashMap::from([(
            "rules".into(),
            ContextSource::Instructions {
                name: "uncertified".into(),
            },
        )]),
        "test".into(),
        "1".into(),
    ));
    manager.register_provider("uncertified", Arc::new(UncertifiedInstructions));
    let agent = Arc::new(
        AgentBuilder::new()
            .system_prompt("base")
            .llm(provider.clone())
            .context_manager(manager)
            .build()
            .unwrap(),
    );
    let config: AutonomyConfig =
        serde_yaml::from_str("enabled: true\ncompletion: {response_not_empty: true}\n").unwrap();
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "instruction-unsafe.v1".into())
            .unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        Default::default(),
        store,
        "instruction-unsafe.v1".into(),
    )
    .unwrap();
    let result = runner.run("work", None).await;
    if let Ok(result) = result {
        assert_ne!(result.run.status, TaskRunStatus::Completed);
    }
    assert_eq!(provider.call_count(), 0);
}

#[tokio::test]
async fn state_replacement_and_streaming_preserve_instructions() {
    use futures::StreamExt;
    let fixture = Fixture::new();
    let provider = mock(vec![LLMResponse::new("done", FinishReason::Stop)]);
    let yaml = "name: state-instructions\nsystem_prompt: base-original\nstates:\n  initial: work\n  states:\n    work:\n      prompt: state-replacement\n      prompt_mode: replace\n";
    let agent = AgentBuilder::from_yaml(yaml)
        .unwrap()
        .llm(provider.clone())
        .context_manager(fixture.manager())
        .auto_configure_features()
        .unwrap()
        .build()
        .unwrap();
    let chunks: Vec<_> = agent.chat_stream("work").await.unwrap().collect().await;
    assert!(!chunks.is_empty());
    assert_eq!(provider.call_count(), 1);
    let request = text(&provider.call_history()[0]);
    assert!(request.contains("state-replacement"));
    assert!(!request.contains("base-original"));
    assert!(request.contains("root-v1") && request.contains("nested-v1"));
}

#[tokio::test]
async fn state_action_refuses_unavailable_rules_before_transition() {
    let fixture = Fixture::new();
    let provider = mock(vec![LLMResponse::new("exit completed", FinishReason::Stop)]);
    let yaml = "name: state-actions\nsystem_prompt: base\nstates:\n  initial: first\n  states:\n    first:\n      prompt: first\n      on_exit:\n        - prompt: Perform exit work\n    second:\n      prompt: second\n";
    let agent = AgentBuilder::from_yaml(yaml)
        .unwrap()
        .llm(provider.clone())
        .context_manager(fixture.manager())
        .auto_configure_features()
        .unwrap()
        .build()
        .unwrap();
    std::fs::write(fixture.0.join("src/AGENTS.md"), [0xff]).unwrap();
    assert!(matches!(
        agent.transition_to("second").await,
        Err(ai_agents_core::AgentError::InstructionUnavailable(_))
    ));
    assert_eq!(agent.current_state().as_deref(), Some("first"));
    assert_eq!(provider.call_count(), 0);
    std::fs::write(fixture.0.join("src/AGENTS.md"), "nested-v2").unwrap();
    agent.transition_to("second").await.unwrap();
    assert_eq!(agent.current_state().as_deref(), Some("second"));
    assert!(text(&provider.call_history()[0]).contains("nested-v2"));
}

#[tokio::test]
async fn task_skill_prompt_uses_live_instructions_without_adding_them_to_cursor() {
    let fixture = Fixture::new();
    let provider = mock(vec![
        LLMResponse::new("done", FinishReason::Stop),
        LLMResponse::new("done", FinishReason::Stop),
    ]);
    let yaml = "name: skill-instructions\nsystem_prompt: base\nskills:\n  - id: script\n    description: Perform scripted work\n    trigger: script\n    steps:\n      - prompt: Perform the task\n";
    let agent = Arc::new(
        AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(provider.clone())
            .context_manager(fixture.manager())
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap(),
    );
    let config: AutonomyConfig =
        serde_yaml::from_str("enabled: true\ncompletion: {response_contains: done}\n").unwrap();
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "instruction-skill.v1".into())
            .unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        Default::default(),
        store.clone(),
        "instruction-skill.v1".into(),
    )
    .unwrap();
    let done = runner.run_skill("work", "script", None).await.unwrap();
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    assert!(text(&provider.call_history()[0]).contains("root-v1"));
    let checkpoint = store.load(&done.run.key.run_id).await.unwrap().unwrap();
    let payload = serde_json::to_string(&checkpoint.payload).unwrap();
    assert!(!payload.contains("root-v1"));
}

#[tokio::test]
async fn retained_skill_reloads_after_question_without_replaying_prompt_prefix() {
    let fixture = Fixture::new();
    let provider = mock(vec![
        LLMResponse::new("prepared", FinishReason::Stop),
        LLMResponse::new("done", FinishReason::Stop),
    ]);
    let yaml = "name: skill-pause-instructions\nsystem_prompt: base\ntools: [ask_user]\nskills:\n  - id: script\n    description: Perform scripted work\n    trigger: script\n    steps:\n      - prompt: Prepare\n      - tool: ask_user\n        args: {question: 'Continue?', options: [yes], allow_other: false}\n      - prompt: Finish\n";
    let agent = Arc::new(
        AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(provider.clone())
            .context_manager(fixture.manager())
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap(),
    );
    let mut config: AutonomyConfig =
        serde_yaml::from_str("enabled: true\ncompletion: {response_contains: done}\n").unwrap();
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "instruction-skill-pause.v1".into())
            .unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        Default::default(),
        store,
        "instruction-skill-pause.v1".into(),
    )
    .unwrap();
    let paused = runner.run_skill("work", "script", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    assert_eq!(provider.call_count(), 1);
    std::fs::write(fixture.0.join("AGENTS.md"), "root-v2").unwrap();
    let interaction = runner
        .task_interaction(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap()
        .unwrap();
    let done = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: interaction.request_id,
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
    assert_eq!(provider.call_count(), 2);
    assert!(text(&provider.call_history()[0]).contains("root-v1"));
    assert!(text(&provider.call_history()[1]).contains("root-v2"));
    assert!(!text(&provider.call_history()[1]).contains("root-v1"));
}

struct LargeSchemaProbe;
#[async_trait]
impl Tool for LargeSchemaProbe {
    fn id(&self) -> &str {
        "large_schema_probe"
    }
    fn name(&self) -> &str {
        "large_schema_probe"
    }
    fn description(&self) -> &str {
        "Large prompt-protocol schema"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"value":{"type":"string","description":"x".repeat(3000)}}})
    }
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: true,
            concurrency_safe: true,
            ..Default::default()
        }
    }
    async fn execute(&self, _: Value, _: ToolExecutionContext) -> ToolResult {
        ToolResult::ok("not called")
    }
}

#[tokio::test]
async fn final_protocol_growth_is_checked_before_provider_or_fallback() {
    for native in [false, true] {
        let fixture = Fixture::new();
        let mut mock = MockLLMProvider::new("tool-provider");
        mock.set_tool_choice(Some(ToolChoice::Auto));
        mock.set_feature_support(LLMFeature::FunctionCalling, true);
        mock.set_native_tool_support(native);
        mock.add_response(LLMResponse::new("not called", FinishReason::Stop));
        let provider = Arc::new(mock);
        let agent = AgentBuilder::from_yaml(
            "name: instruction-budget\nsystem_prompt: base\ntools: [large_schema_probe]\n",
        )
        .unwrap()
        .llm(provider.clone())
        .context_manager(fixture.manager())
        .max_context_tokens(160)
        .auto_configure_features()
        .unwrap()
        .tool(Arc::new(LargeSchemaProbe))
        .build()
        .unwrap();
        let result = agent.chat("work").await;
        assert!(
            matches!(
                result,
                Err(ai_agents_core::AgentError::InstructionUnavailable(_))
            ),
            "native={native}: {result:?}"
        );
        assert_eq!(provider.call_count(), 0);
    }
}

struct CorruptRules(PathBuf);
#[async_trait]
impl Tool for CorruptRules {
    fn id(&self) -> &str {
        "corrupt_rules_fixture"
    }
    fn name(&self) -> &str {
        "corrupt_rules_fixture"
    }
    fn description(&self) -> &str {
        "Corrupt a disposable instruction fixture"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object"})
    }
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata::default()
    }
    async fn execute(&self, _: Value, _: ToolExecutionContext) -> ToolResult {
        std::fs::write(&self.0, [0xff]).unwrap();
        ToolResult::ok("fixture changed")
    }
}

#[tokio::test]
async fn plan_skill_instruction_failure_cannot_become_a_successful_response() {
    let fixture = Fixture::new();
    let provider = mock(vec![
        LLMResponse::new("none", FinishReason::Stop),
        LLMResponse::new(
            r#"{"steps":[{"id":"s1","description":"change fixture","action_type":"tool","action_target":"corrupt_rules_fixture","args":{},"dependencies":[]},{"id":"s2","description":"use skill","action_type":"skill","action_target":"script","dependencies":["s1"]}]}"#,
            FinishReason::Stop,
        ),
        LLMResponse::new("must not run", FinishReason::Stop),
    ]);
    let yaml = "name: plan-failure\nsystem_prompt: base\ntools: [corrupt_rules_fixture]\nskills:\n  - id: script\n    description: Work\n    trigger: use script\n    steps:\n      - prompt: Work\nreasoning:\n  mode: plan_and_execute\n  planning:\n    reflection:\n      enabled: false\n";
    let agent = AgentBuilder::from_yaml(yaml)
        .unwrap()
        .llm(provider.clone())
        .context_manager(fixture.manager())
        .auto_configure_features()
        .unwrap()
        .tool(Arc::new(CorruptRules(fixture.0.join("AGENTS.md"))))
        .build()
        .unwrap();
    let result = agent.chat("work").await;
    assert!(
        matches!(
            result,
            Err(ai_agents_core::AgentError::InstructionUnavailable(_))
        ),
        "{result:?}"
    );
    assert_eq!(provider.call_count(), 2);
}

struct ReadProbe;
#[async_trait]
impl Tool for ReadProbe {
    fn id(&self) -> &str {
        "read_probe"
    }
    fn name(&self) -> &str {
        "read_probe"
    }
    fn description(&self) -> &str {
        "Return a deterministic observation"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object"})
    }
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: true,
            concurrency_safe: true,
            ..Default::default()
        }
    }
    async fn execute(&self, _: Value, _: ToolExecutionContext) -> ToolResult {
        ToolResult::ok("observed")
    }
}

#[tokio::test]
async fn model_capacity_pause_retains_prepared_instruction_request() {
    let fixture = Fixture::new();
    let call = ToolCall {
        id: "probe".into(),
        name: "read_probe".into(),
        arguments: json!({}),
    };
    let provider = mock(vec![
        LLMResponse::new("", FinishReason::ToolCall)
            .with_tool_calls(vec![call])
            .unwrap(),
        LLMResponse::new("done", FinishReason::Stop),
    ]);
    let agent = Arc::new(
        AgentBuilder::from_yaml(
            "name: capacity-instructions\nsystem_prompt: base\ntools: [read_probe]\n",
        )
        .unwrap()
        .llm(provider.clone())
        .context_manager(fixture.manager())
        .auto_configure_features()
        .unwrap()
        .tool(Arc::new(ReadProbe))
        .build()
        .unwrap(),
    );
    let config: AutonomyConfig = serde_yaml::from_str("enabled: true\nmax_llm_calls: 1\non_limit: {action: pause_run}\ncompletion: {response_contains: done}\n").unwrap();
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "instruction-capacity.v1".into())
            .unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        Default::default(),
        store,
        "instruction-capacity.v1".into(),
    )
    .unwrap();
    let paused = runner.run("work", None).await.unwrap();
    assert_eq!(
        paused.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        paused.run.stop_reason
    );
    assert_eq!(provider.call_count(), 1);
    std::fs::write(fixture.0.join("AGENTS.md"), "root-v2").unwrap();
    let mut limits = paused.run.limits.clone();
    limits.max_llm_calls = Some(2);
    let revised = runner
        .revise_limits_paused(
            &paused.run.key.run_id,
            paused.run.revision,
            limits,
            "finish pending model request",
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
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    assert_eq!(provider.call_count(), 2);
    assert!(text(&provider.call_history()[1]).contains("root-v1"));
    assert!(!text(&provider.call_history()[1]).contains("root-v2"));
}

#[tokio::test]
async fn skill_capacity_pause_retains_prepared_instruction_request() {
    let fixture = Fixture::new();
    let provider = mock(vec![
        LLMResponse::new("prepared", FinishReason::Stop),
        LLMResponse::new("done", FinishReason::Stop),
    ]);
    let yaml = "name: skill-capacity-instructions\nsystem_prompt: base\nskills:\n  - id: script\n    description: Perform scripted work\n    trigger: script\n    steps:\n      - prompt: Prepare\n      - prompt: Finish\n";
    let agent = Arc::new(
        AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(provider.clone())
            .context_manager(fixture.manager())
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap(),
    );
    let config: AutonomyConfig = serde_yaml::from_str("enabled: true\nmax_llm_calls: 1\non_limit: {action: pause_run}\ncompletion: {response_contains: done}\n").unwrap();
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(
            agent.info().id,
            None,
            "instruction-skill-capacity.v1".into(),
        )
        .unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        Default::default(),
        store,
        "instruction-skill-capacity.v1".into(),
    )
    .unwrap();
    let paused = runner.run_skill("work", "script", None).await.unwrap();
    assert_eq!(
        paused.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        paused.run.stop_reason
    );
    assert_eq!(provider.call_count(), 1);
    std::fs::write(fixture.0.join("AGENTS.md"), "root-v2").unwrap();
    let mut limits = paused.run.limits.clone();
    limits.max_llm_calls = Some(2);
    let revised = runner
        .revise_limits_paused(
            &paused.run.key.run_id,
            paused.run.revision,
            limits,
            "finish pending prompt",
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
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    assert_eq!(provider.call_count(), 2);
    assert!(text(&provider.call_history()[1]).contains("root-v1"));
    assert!(!text(&provider.call_history()[1]).contains("root-v2"));
}

#[tokio::test]
async fn planning_request_receives_the_same_instruction_source() {
    let fixture = Fixture::new();
    let provider = mock(vec![
        LLMResponse::new(
            r#"{"steps":[{"id":"s1","description":"respond","action_type":"respond","action_target":"answer","dependencies":[]}]}"#,
            FinishReason::Stop,
        ),
        LLMResponse::new("done", FinishReason::Stop),
    ]);
    let yaml = "name: plan-instructions\nsystem_prompt: base\nreasoning:\n  mode: plan_and_execute\n  planning:\n    reflection:\n      enabled: false\n";
    let agent = AgentBuilder::from_yaml(yaml)
        .unwrap()
        .llm(provider.clone())
        .context_manager(fixture.manager())
        .auto_configure_features()
        .unwrap()
        .build()
        .unwrap();
    agent.chat("plan work").await.unwrap();
    let history = provider.call_history();
    assert!(history.len() >= 2);
    assert!(text(&history[0]).contains("Create a step-by-step plan"));
    assert!(text(&history[0]).contains("root-v1"));
    assert!(text(&history[1]).contains("root-v1"));
}

#[tokio::test]
async fn retained_question_reloads_only_the_next_model_request() {
    let fixture = Fixture::new();
    let question = ToolCall {
        id: "question-1".into(),
        name: "ask_user".into(),
        arguments: json!({"question":"Continue?","options":["yes"],"allow_other":false}),
    };
    let provider = mock(vec![
        LLMResponse::new("", FinishReason::ToolCall)
            .with_tool_calls(vec![question])
            .unwrap(),
        LLMResponse::new("done", FinishReason::Stop),
    ]);
    let agent = Arc::new(
        AgentBuilder::from_yaml(
            "name: instruction-pause\nsystem_prompt: base\ntools: [ask_user]\n",
        )
        .unwrap()
        .llm(provider.clone())
        .context_manager(fixture.manager())
        .auto_configure_features()
        .unwrap()
        .build()
        .unwrap(),
    );
    let mut config: AutonomyConfig =
        serde_yaml::from_str("enabled: true\ncompletion: {response_contains: done}\n").unwrap();
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "instructions.v1".into()).unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        Default::default(),
        store,
        "instructions.v1".into(),
    )
    .unwrap();
    let paused = runner.run("work", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    assert_eq!(provider.call_count(), 1);
    std::fs::write(fixture.0.join("AGENTS.md"), "root-v2").unwrap();
    let interaction = runner
        .task_interaction(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap()
        .unwrap();
    let done = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: interaction.request_id,
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(done.run.counters.llm_attempts, 2);
    assert_eq!(provider.call_count(), 2);
    let first = text(&provider.call_history()[0]);
    let second = text(&provider.call_history()[1]);
    assert!(first.contains("root-v1"));
    assert!(second.contains("root-v2"));
    assert!(!second.contains("root-v1"));
}

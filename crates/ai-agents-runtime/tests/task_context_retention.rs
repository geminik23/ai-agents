//! End-to-end task context checks at the actual provider request boundary.

use std::{collections::HashMap, sync::Arc};

use ai_agents_context::{ContextManager, ContextProvider, ContextSource};
use ai_agents_core::{
    ChatMessage, FinishReason, IncompleteNativeExchange, LLMFeature, LLMResponse, Memory,
    MemorySnapshot, NativeCallBinding, NativeProviderState, NativeProviderTarget, Tool, ToolCall,
    ToolChoice, ToolExecutionContext, ToolResult, ToolSafetyMetadata,
    encode_native_tool_call_markers,
};
use ai_agents_llm::mock::MockLLMProvider;
use ai_agents_memory::{
    CompactingMemory, CompactingMemoryConfig, InMemoryStore, Memory as ExtendedMemory, Summarizer,
};
use ai_agents_runtime::{Agent, AgentBuilder, autonomy::*};
use async_trait::async_trait;
use serde_json::{Value, json};

struct FixedInstructions;

#[async_trait]
impl ContextProvider for FixedInstructions {
    fn task_instruction_safe(&self) -> bool {
        true
    }

    async fn get(&self, _: &str, _: &Value) -> ai_agents_core::Result<Value> {
        Ok(json!("HOST-CONSTRAINT: retain current scope"))
    }
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
        "A bounded fixture for protocol-size admission"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"value":{"type":"string","description":"x".repeat(8000)}}})
    }
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: true,
            concurrency_safe: true,
            ..Default::default()
        }
    }
    async fn execute(&self, _: Value, _: ToolExecutionContext) -> ToolResult {
        ToolResult::ok("must not execute")
    }
}

struct DroppingSummarizer;

#[async_trait]
impl Summarizer for DroppingSummarizer {
    fn supports_task_admission(&self) -> bool {
        true
    }

    async fn summarize(&self, _: &[ChatMessage]) -> ai_agents_core::Result<String> {
        Ok("summary that omits the incomplete exchange".into())
    }

    async fn merge_summaries(&self, _: &[String]) -> ai_agents_core::Result<String> {
        Ok("merged summary that omits the incomplete exchange".into())
    }
}

fn incomplete_signed_turn() -> Vec<ChatMessage> {
    let call = ToolCall {
        id: "history-call".into(),
        name: "probe".into(),
        arguments: json!({}),
    };
    let state = NativeProviderState::new(
        "history-exchange",
        "fixture",
        "native-tools",
        NativeProviderTarget::new("https://fixture.invalid/", "fixture-model").unwrap(),
        json!({"role":"model","parts":[{"functionCall":{"name":"probe","args":{}},"thoughtSignature":"secret-fixture-signature"}]}),
        vec![NativeCallBinding::new(&call.id, 0).unwrap()],
    )
    .unwrap();
    vec![
        ChatMessage::user("earlier objective"),
        ChatMessage::assistant(encode_native_tool_call_markers(&[call], Some(&state)).unwrap()),
        ChatMessage::assistant("earlier final response"),
    ]
}

#[tokio::test]
async fn overflow_keeps_controller_objective() {
    let memory = Arc::new(InMemoryStore::new(20));
    memory
        .add_message(ChatMessage::user("old history ".repeat(1000)))
        .await
        .unwrap();
    let mut provider = MockLLMProvider::new("context-retention");
    provider.add_response(LLMResponse::new("done", FinishReason::Stop));
    let provider = Arc::new(provider);
    let agent = Arc::new(
        AgentBuilder::from_yaml(
            "name: context-retention\nsystem_prompt: base\nerror_recovery:\n  llm:\n    on_context_overflow: {action: truncate, keep_recent: 0}\n",
        )
        .unwrap()
        .llm(provider.clone())
        .memory(memory)
        .max_context_tokens(512)
        .auto_configure_features()
        .unwrap()
        .build()
        .unwrap(),
    );
    let config: AutonomyConfig =
        serde_yaml::from_str("enabled: true\ncompletion: {response_contains: done}\n").unwrap();
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "task-context.v1".into()).unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        Default::default(),
        store,
        "task-context.v1".into(),
    )
    .unwrap();
    let result = runner.run("CURRENT-OBJECTIVE", None).await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    let history = provider.call_history();
    assert_eq!(history.len(), 1);
    let messages = &history[0].messages;
    let controller = messages
        .iter()
        .find(|message| message.content.contains("Autonomy controller instruction"))
        .expect("overflow must preserve the live controller projection");
    assert!(controller.content.contains("CURRENT-OBJECTIVE"));
    assert!(
        !messages
            .iter()
            .any(|message| message.content.contains("old history"))
    );
}

#[tokio::test]
async fn revised_objective_survives_overflow_without_replaying_the_paused_request() {
    let memory = Arc::new(InMemoryStore::new(20));
    memory
        .add_message(ChatMessage::user("old history ".repeat(1000)))
        .await
        .unwrap();
    let question = ToolCall {
        id: "question".into(),
        name: "ask_user".into(),
        arguments: json!({"question":"Continue?","options":["yes"],"allow_other":false}),
    };
    let mut provider = MockLLMProvider::new("context-retention");
    provider.set_tool_choice(Some(ToolChoice::Auto));
    provider.set_feature_support(LLMFeature::FunctionCalling, true);
    provider.add_response(
        LLMResponse::new("", FinishReason::ToolCall)
            .with_tool_calls(vec![question])
            .unwrap(),
    );
    provider.add_response(LLMResponse::new("done", FinishReason::Stop));
    let provider = Arc::new(provider);
    let agent = Arc::new(AgentBuilder::from_yaml(
        "name: revised-context\nsystem_prompt: base\ntools: [ask_user]\nerror_recovery:\n  llm:\n    on_context_overflow: {action: truncate, keep_recent: 0}\n",
    ).unwrap().llm(provider.clone()).memory(memory).max_context_tokens(1500).auto_configure_features().unwrap().build().unwrap());
    let config: AutonomyConfig = serde_yaml::from_str(
        "enabled: true\nhitl: {on_user_question: pause_run}\ncompletion: {response_contains: done}\n",
    ).unwrap();
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "task-context.v1".into()).unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        Default::default(),
        store,
        "task-context.v1".into(),
    )
    .unwrap();
    let paused = runner.run("OBJECTIVE-OLD", None).await.unwrap();
    assert_eq!(
        paused.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        paused.run.stop_reason
    );
    let original_id = paused.run.key.run_id.clone();
    let revised = runner
        .modify_objective_paused(
            &original_id,
            paused.run.revision,
            "OBJECTIVE-NEW",
            "host update",
        )
        .await
        .unwrap();
    assert_eq!(revised.run.counters.llm_attempts, 1);
    assert_eq!(provider.call_count(), 1);
    let done = runner
        .resume(
            &original_id,
            revised.run.revision,
            TaskResumeInput::Continue,
        )
        .await
        .unwrap();
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "reason={:?}, calls={}, counters={:?}, evidence={:?}",
        done.run.stop_reason,
        provider.call_count(),
        done.run.counters,
        done.run.evidence
    );
    assert_eq!(done.run.key.run_id, original_id);
    assert_eq!(done.run.counters.llm_attempts, 2);
    assert_eq!(provider.call_count(), 2);
    let request = provider.last_call().unwrap();
    let controller = request
        .messages
        .iter()
        .find(|message| message.content.contains("Autonomy controller instruction"))
        .unwrap();
    assert!(controller.content.contains("OBJECTIVE-NEW"));
    assert!(!controller.content.contains("OBJECTIVE-OLD"));
}

#[tokio::test]
async fn compaction_allocation_and_overflow_preserve_advisory_status_and_task_context() {
    let memory = Arc::new(CompactingMemory::new(
        Arc::new(DroppingSummarizer),
        CompactingMemoryConfig {
            max_recent_messages: 1,
            compress_threshold: 4,
            summarize_batch_size: 2,
            max_summary_length: 2000,
        },
    ));
    for message in incomplete_signed_turn() {
        memory.add_message(message).await.unwrap();
    }
    memory
        .add_message(ChatMessage::user("next turn"))
        .await
        .unwrap();
    memory.compress(None).await.unwrap();
    memory
        .add_message(ChatMessage::assistant("progress"))
        .await
        .unwrap();
    memory
        .add_message(ChatMessage::user("another turn"))
        .await
        .unwrap();
    memory
        .add_message(ChatMessage::assistant("more progress"))
        .await
        .unwrap();
    memory.compress(None).await.unwrap();
    assert_eq!(memory.compression_history().len(), 2);
    memory
        .add_message(ChatMessage::user("old history ".repeat(1000)))
        .await
        .unwrap();

    let manager = Arc::new(ContextManager::new(
        HashMap::from([(
            "rules".into(),
            ContextSource::Instructions {
                name: "host".into(),
            },
        )]),
        "fixture".into(),
        "1".into(),
    ));
    manager.register_provider("host", Arc::new(FixedInstructions));
    let mut provider = MockLLMProvider::new("context-retention");
    provider.add_response(LLMResponse::new("done", FinishReason::Stop));
    let provider = Arc::new(provider);
    let agent = Arc::new(
        AgentBuilder::from_yaml(
            "name: context-retention\nsystem_prompt: base\nmemory:\n  token_budget:\n    total: 200000\n    allocation: {summary: 0, recent_messages: 20000, facts: 0, relationships: 0}\nerror_recovery:\n  llm:\n    on_context_overflow: {action: truncate, keep_recent: 0}\n",
        )
        .unwrap()
        .llm(provider.clone())
        .memory(memory)
        .context_manager(manager)
        .max_context_tokens(512)
        .auto_configure_features()
        .unwrap()
        .build()
        .unwrap(),
    );
    let config: AutonomyConfig =
        serde_yaml::from_str("enabled: true\ncompletion: {response_contains: done}\n").unwrap();
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "task-context.v1".into()).unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        Default::default(),
        store,
        "task-context.v1".into(),
    )
    .unwrap();
    let result = runner.run("CURRENT-OBJECTIVE", None).await.unwrap();
    assert_eq!(
        result.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        result.run.stop_reason
    );
    assert_eq!(result.run.counters.llm_attempts, 1);
    let history = provider.call_history();
    assert_eq!(history.len(), 1);
    let text = history[0]
        .messages
        .iter()
        .map(|message| message.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("CURRENT-OBJECTIVE"));
    assert!(text.contains("HOST-CONSTRAINT: retain current scope"));
    assert!(text.contains("history-call"));
    assert!(text.contains("final execution result was not recorded"));
    let status = history[0]
        .messages
        .iter()
        .find(|message| message.content.contains("history-call"))
        .unwrap();
    assert_eq!(status.role, ai_agents_core::Role::User);
    assert!(!text.contains("secret-fixture-signature"));
    assert!(!text.contains("old history"));
}

#[tokio::test]
async fn overflow_drops_older_content_when_incomplete_history_can_still_fit() {
    let memory = Arc::new(InMemoryStore::new(20));
    memory
        .add_message(ChatMessage::user("discardable history ".repeat(500)))
        .await
        .unwrap();
    memory
        .add_message(ChatMessage::assistant("earlier response"))
        .await
        .unwrap();
    for message in incomplete_signed_turn() {
        memory.add_message(message).await.unwrap();
    }
    memory
        .add_message(ChatMessage::user("latest request"))
        .await
        .unwrap();
    let mut provider = MockLLMProvider::new("context-retention");
    provider.add_response(LLMResponse::new("done", FinishReason::Stop));
    let provider = Arc::new(provider);
    let agent = AgentBuilder::from_yaml(
        "name: context-retention\nsystem_prompt: base\nerror_recovery:\n  llm:\n    on_context_overflow: {action: truncate, keep_recent: 0}\n",
    ).unwrap().llm(provider.clone()).memory(memory).max_context_tokens(512).auto_configure_features().unwrap().build().unwrap();
    agent.chat("CURRENT-OBJECTIVE").await.unwrap();
    let history = provider.last_call().unwrap();
    let text = history
        .messages
        .iter()
        .map(|message| message.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("history-call"));
    assert!(!text.contains("discardable history"));
}

#[tokio::test]
async fn overflow_rejects_incomplete_history_that_cannot_fit_without_dispatch() {
    let memory = Arc::new(InMemoryStore::new(20));
    for message in incomplete_signed_turn() {
        memory.add_message(message).await.unwrap();
    }
    memory
        .add_message(ChatMessage::user("large continuation ".repeat(1000)))
        .await
        .unwrap();
    let mut provider = MockLLMProvider::new("context-retention");
    provider.add_response(LLMResponse::new("must not run", FinishReason::Stop));
    let provider = Arc::new(provider);
    let agent = AgentBuilder::from_yaml(
        "name: context-retention\nsystem_prompt: base\nerror_recovery:\n  llm:\n    on_context_overflow: {action: truncate, keep_recent: 0}\n",
    ).unwrap().llm(provider.clone()).memory(memory).max_context_tokens(512).auto_configure_features().unwrap().build().unwrap();
    let error = agent.chat("CURRENT-OBJECTIVE").await.unwrap_err();
    assert!(error.to_string().contains("Context"), "{error}");
    assert_eq!(provider.call_count(), 0);
}

#[tokio::test]
async fn overflow_summarizer_cannot_drop_or_receive_protected_task_projection() {
    let memory = Arc::new(CompactingMemory::new(
        Arc::new(DroppingSummarizer),
        CompactingMemoryConfig::default(),
    ));
    memory
        .restore(
            MemorySnapshot::new(vec![ChatMessage::user("old history ".repeat(1000))])
                .with_incomplete_native_exchanges(vec![IncompleteNativeExchange {
                    exchange_id: Some("old-exchange".into()),
                    missing_result_ids: vec!["old-call".into()],
                    unverified_legacy: false,
                }]),
        )
        .await
        .unwrap();
    let mut main = MockLLMProvider::new("main");
    main.add_response(LLMResponse::new("done", FinishReason::Stop));
    let main = Arc::new(main);
    let mut summarizer = MockLLMProvider::new("summarizer");
    summarizer.add_response(LLMResponse::new(
        "shortened conversation",
        FinishReason::Stop,
    ));
    let summarizer = Arc::new(summarizer);
    let mut registry = ai_agents_llm::LLMRegistry::new();
    registry.register("default", main.clone());
    registry.register("summary", summarizer.clone());
    let agent = Arc::new(AgentBuilder::from_yaml(
        "name: context-retention\nsystem_prompt: base\nerror_recovery:\n  llm:\n    on_context_overflow: {action: summarize, summarizer_llm: summary, max_summary_tokens: 64, keep_recent: 0}\n",
    ).unwrap().llm_registry(registry).memory(memory).max_context_tokens(512).auto_configure_features().unwrap().build().unwrap());
    let config: AutonomyConfig =
        serde_yaml::from_str("enabled: true\ncompletion: {response_contains: done}\n").unwrap();
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "task-context.v1".into()).unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        Default::default(),
        store,
        "task-context.v1".into(),
    )
    .unwrap();
    let result = runner.run("CURRENT-OBJECTIVE", None).await.unwrap();
    assert_eq!(
        result.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        result.run.stop_reason
    );
    assert_eq!(main.call_count(), 1);
    assert_eq!(summarizer.call_count(), 1);
    let summarized = summarizer
        .last_call()
        .unwrap()
        .messages
        .iter()
        .map(|message| message.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!summarized.contains("Autonomy controller instruction"));
    assert!(!summarized.contains("old-call"));
    let request = main
        .last_call()
        .unwrap()
        .messages
        .iter()
        .map(|message| message.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(request.contains("CURRENT-OBJECTIVE"));
    assert!(request.contains("old-call"));
    assert!(request.contains("shortened conversation"));
}

#[tokio::test]
async fn task_rejects_final_tool_protocol_growth_before_dispatch_without_instructions() {
    for native in [false, true] {
        let mut provider = MockLLMProvider::new("context-retention");
        provider.set_tool_choice(Some(ToolChoice::Auto));
        provider.set_feature_support(LLMFeature::FunctionCalling, true);
        provider.set_native_tool_support(native);
        provider.add_response(LLMResponse::new("done", FinishReason::Stop));
        let provider = Arc::new(provider);
        let agent = Arc::new(
            AgentBuilder::from_yaml(
                "name: context-retention\nsystem_prompt: base\ntools: [large_schema_probe]\n",
            )
            .unwrap()
            .llm(provider.clone())
            .max_context_tokens(512)
            .auto_configure_features()
            .unwrap()
            .tool(Arc::new(LargeSchemaProbe))
            .build()
            .unwrap(),
        );
        let config: AutonomyConfig =
            serde_yaml::from_str("enabled: true\ncompletion: {response_contains: done}\n").unwrap();
        let store = Arc::new(
            ScopedTaskRunStore::in_memory(agent.info().id, None, "task-context.v1".into()).unwrap(),
        );
        let runner = AutonomyRunner::try_new(
            agent,
            config,
            Default::default(),
            store,
            "task-context.v1".into(),
        )
        .unwrap();
        let result = runner.run("CURRENT-OBJECTIVE", None).await;
        assert!(
            result.is_err()
                || result
                    .as_ref()
                    .is_ok_and(|value| value.run.status != TaskRunStatus::Completed),
            "native={native}"
        );
        assert_eq!(provider.call_count(), 0, "native={native}");
    }
}

#[tokio::test]
async fn ordinary_chat_keeps_its_existing_tool_protocol_budget_behavior() {
    for native in [false, true] {
        let mut provider = MockLLMProvider::new("ordinary-context");
        provider.set_tool_choice(Some(ToolChoice::Auto));
        provider.set_feature_support(LLMFeature::FunctionCalling, true);
        provider.set_native_tool_support(native);
        provider.add_response(LLMResponse::new("done", FinishReason::Stop));
        let provider = Arc::new(provider);
        let agent = AgentBuilder::from_yaml(
            "name: ordinary-context\nsystem_prompt: base\ntools: [large_schema_probe]\n",
        )
        .unwrap()
        .llm(provider.clone())
        .max_context_tokens(512)
        .auto_configure_features()
        .unwrap()
        .tool(Arc::new(LargeSchemaProbe))
        .build()
        .unwrap();
        agent.chat("ordinary").await.unwrap();
        assert_eq!(provider.call_count(), 1, "native={native}");
    }
}

#[tokio::test]
async fn incompatible_memory_restoration_rejects_before_changing_existing_state() {
    let memory = Arc::new(InMemoryStore::new(10));
    memory
        .add_message(ChatMessage::user("preserve existing"))
        .await
        .unwrap();
    let provider = Arc::new(MockLLMProvider::new("context-retention"));
    let agent = AgentBuilder::from_yaml("name: restore-fixture\nsystem_prompt: base\n")
        .unwrap()
        .llm(provider)
        .memory(memory.clone())
        .auto_configure_features()
        .unwrap()
        .build()
        .unwrap();
    let before = agent.save_state().await.unwrap();
    let mut structured = before.clone();
    structured.memory = MemorySnapshot::default().with_incomplete_native_exchanges(vec![
        IncompleteNativeExchange {
            exchange_id: Some("other".into()),
            missing_result_ids: vec!["missing".into()],
            unverified_legacy: false,
        },
    ]);
    assert!(agent.restore_state(structured).await.is_err());
    let mut legacy = before;
    legacy.memory.summary = Some(r#"old
{"description":"final execution result was not recorded","missing_result_ids":["missing"],"native_exchange_status":"incomplete"}"#.into());
    assert!(agent.restore_state(legacy).await.is_err());
    let messages = memory.get_messages(None).await.unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].content, "preserve existing");
}

#[tokio::test]
async fn oversized_retained_observation_is_rejected_before_provider() {
    let memory = Arc::new(CompactingMemory::new(
        Arc::new(DroppingSummarizer),
        CompactingMemoryConfig::default(),
    ));
    memory
        .restore(
            MemorySnapshot::default().with_incomplete_native_exchanges(vec![
                IncompleteNativeExchange {
                    exchange_id: Some("historical".into()),
                    missing_result_ids: vec!["x".repeat(2000)],
                    unverified_legacy: false,
                },
            ]),
        )
        .await
        .unwrap();
    let mut provider = MockLLMProvider::new("context-retention");
    provider.add_response(LLMResponse::new("must not run", FinishReason::Stop));
    let provider = Arc::new(provider);
    let agent = Arc::new(AgentBuilder::from_yaml(
        "name: context-retention\nsystem_prompt: base\nerror_recovery:\n  llm:\n    on_context_overflow: {action: truncate, keep_recent: 0}\n",
    ).unwrap().llm(provider.clone()).memory(memory).max_context_tokens(256).auto_configure_features().unwrap().build().unwrap());
    let config: AutonomyConfig =
        serde_yaml::from_str("enabled: true\ncompletion: {response_contains: done}\n").unwrap();
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "task-context.v1".into()).unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        Default::default(),
        store,
        "task-context.v1".into(),
    )
    .unwrap();
    let result = runner.run("CURRENT-OBJECTIVE", None).await;
    assert!(
        result.is_err()
            || result
                .as_ref()
                .is_ok_and(|value| value.run.status != TaskRunStatus::Completed)
    );
    assert_eq!(provider.call_count(), 0);
}

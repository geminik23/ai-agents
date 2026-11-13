use ai_agents_core::{
    AgentError, Tool, ToolCallSource, ToolExecutionContext, ToolExecutionRequest, ToolInvoker,
    ToolResult, ToolSafetyMetadata,
};
use ai_agents_hitl::{
    ApprovalHandler, ApprovalRequest, ApprovalResult, ApprovalTrigger, AutoApproveHandler,
    RejectAllHandler,
};
use ai_agents_hooks::{AgentHooks, LoggingHooks};
use ai_agents_llm::mock::MockLLMProvider;
use ai_agents_observability::{
    ExportConfig, ObservabilityConfig, ObservabilityHooks, ObservabilityManager, ObservedTool,
    PrivacyConfig,
};
use ai_agents_runtime::AgentBuilder;
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::Arc;

const BODY: &str = "private-tool-body-fixture";

#[derive(Clone)]
struct Logs(Arc<parking_lot::Mutex<Vec<String>>>);
struct Fields<'a>(&'a mut Vec<String>);
impl tracing::field::Visit for Fields<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.push(format!("{}={value:?}", field.name()));
    }
}
impl tracing::Subscriber for Logs {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        event.record(&mut Fields(&mut self.0.lock()));
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

// Default tracing has no observability privacy manager; exact review data must remain available only to its actual recipient.
#[tokio::test(flavor = "current_thread")]
async fn approval_logs_omit_bodies_and_rejection_reasons_without_changing_review() {
    let logs = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let _guard = tracing::subscriber::set_default(Logs(logs.clone()));
    let request = ApprovalRequest::new(
        ApprovalTrigger::tool("fixture", json!({"content":BODY})),
        BODY,
    );
    let logging = LoggingHooks::new();
    logging.on_approval_requested(&request).await;
    logging
        .on_approval_result(&request.id, &ApprovalResult::rejected_with_reason(BODY))
        .await;
    assert!(
        RejectAllHandler::new()
            .request_approval(request.clone())
            .await
            .is_rejected()
    );
    assert!(
        AutoApproveHandler::new()
            .request_approval(request.clone())
            .await
            .is_approved()
    );
    let exact = ai_agents_hitl::CallbackHandler::new(|received: ApprovalRequest| {
        assert_eq!(received.message, BODY);
        assert!(
            matches!(received.trigger, ApprovalTrigger::Tool { args, .. } if args["content"] == BODY)
        );
        ApprovalResult::approved()
    });
    assert!(exact.request_approval(request.clone()).await.is_approved());
    assert_eq!(request.message, BODY);
    let output = logs.lock().join("\n");
    assert!(output.contains("Approval requested"));
    assert!(output.contains("rejected"));
    assert!(!output.contains(BODY), "{output}");
}

struct Failure;
#[async_trait]
impl Tool for Failure {
    fn id(&self) -> &str {
        "failure"
    }
    fn name(&self) -> &str {
        "failure"
    }
    fn description(&self) -> &str {
        "Isolated failure body"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object"})
    }
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: true,
            ..Default::default()
        }
    }
    async fn execute(&self, _: Value, _: ToolExecutionContext) -> ToolResult {
        ToolResult::error(BODY)
    }
    async fn execute_task_question(
        &self,
        _: Value,
        _: Value,
        _: ToolExecutionContext,
    ) -> ToolResult {
        ToolResult::error(BODY)
    }
}

struct Reports(std::path::PathBuf);
impl Reports {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("tool-disclosure-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Reports {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn manager(include: bool, cap: usize, raw: bool, reports: &Reports) -> Arc<ObservabilityManager> {
    ObservabilityManager::new(ObservabilityConfig {
        enabled: true,
        privacy: PrivacyConfig {
            include_tool_outputs: include,
            max_text_chars: cap,
            ..Default::default()
        },
        export: ExportConfig {
            path: reports.0.to_string_lossy().into_owned(),
            formats: vec![
                ai_agents_observability::ExportFormat::Json,
                ai_agents_observability::ExportFormat::Jsonl,
            ],
            write_raw_events: raw,
            ..Default::default()
        },
        ..Default::default()
    })
}

// Ordinary, task and resumed-question wrappers must enforce the same output permission on both error and payload channels.
#[tokio::test]
async fn tool_output_opt_in_covers_errors_hooks_memory_and_export() {
    for include in [false, true] {
        for cap in [0, 1024] {
            for raw in [false, true] {
                let reports = Reports::new();
                let manager = manager(include, cap, raw, &reports);
                let cursor = manager.event_cursor();
                let tool = ObservedTool::new(Arc::new(Failure), manager.clone());
                let context = ToolExecutionContext::test("failure");
                for result in [
                    tool.execute(json!({}), context.clone()).await,
                    tool.execute_task(json!({}), context.clone()).await.unwrap(),
                    tool.execute_task_question(json!({}), json!({}), context)
                        .await,
                ] {
                    assert!(!result.success);
                    assert_eq!(result.output, BODY);
                }
                let hooks = ObservabilityHooks::new(manager.clone());
                hooks.on_error(&AgentError::Tool(BODY.into())).await;
                hooks.on_error(&AgentError::Skill(BODY.into())).await;
                let events = manager.snapshot_since(cursor).events;
                assert_eq!(events.len(), 5);
                assert!(events.iter().all(|event| matches!(
                    event.status,
                    ai_agents_observability::EventStatus::Error
                )));
                let expected = include && cap > 0;
                assert_eq!(
                    serde_json::to_string(&events).unwrap().contains(BODY),
                    expected
                );
                let raw_events = manager.raw_events();
                assert_eq!(raw_events.is_empty(), !raw);
                assert_eq!(
                    serde_json::to_string(&raw_events).unwrap().contains(BODY),
                    expected && raw
                );
                manager.export().await.unwrap();
                let mut exported = String::new();
                for entry in std::fs::read_dir(&reports.0).unwrap() {
                    let path = entry.unwrap().path();
                    if path.is_file() {
                        exported.push_str(&std::fs::read_to_string(path).unwrap());
                    }
                }
                assert_eq!(exported.contains(BODY), expected && raw);
            }
        }
    }
}

// Approval completion contains raw/effective states, but arbitrary review/rejection text has no blanket text-retention opt-in.
#[tokio::test]
async fn resolved_approvals_project_arguments_only_with_their_own_permission() {
    use ai_agents_hitl::ApprovalResolvedOutcome;
    const REVIEW: &str = "private-review-reason-fixture";
    for include_args in [false, true] {
        for cap in [0, 1024] {
            let reports = Reports::new();
            let manager = ObservabilityManager::new(ObservabilityConfig {
                enabled: true,
                privacy: PrivacyConfig {
                    include_tool_args: include_args,
                    max_text_chars: cap,
                    ..Default::default()
                },
                export: ExportConfig {
                    path: reports.0.to_string_lossy().into_owned(),
                    formats: vec![
                        ai_agents_observability::ExportFormat::Json,
                        ai_agents_observability::ExportFormat::Jsonl,
                    ],
                    write_raw_events: true,
                    ..Default::default()
                },
                ..Default::default()
            });
            let hooks = ObservabilityHooks::new(manager.clone());
            let request = ApprovalRequest::new(
                ApprovalTrigger::tool("failure", json!({"content":BODY})),
                REVIEW,
            );
            let changes = std::collections::HashMap::from([("content".into(), json!(BODY))]);
            hooks
                .on_approval_resolved(
                    &request,
                    &ApprovalResult::rejected_with_reason(REVIEW),
                    &ApprovalResolvedOutcome::Rejected {
                        reason: Some(REVIEW.into()),
                    },
                )
                .await;
            hooks
                .on_approval_resolved(
                    &request,
                    &ApprovalResult::Modified {
                        changes: changes.clone(),
                    },
                    &ApprovalResolvedOutcome::Modified { changes },
                )
                .await;
            hooks
                .on_approval_resolved(
                    &request,
                    &ApprovalResult::Timeout,
                    &ApprovalResolvedOutcome::Error {
                        message: REVIEW.into(),
                    },
                )
                .await;
            let condition =
                ApprovalRequest::new(ApprovalTrigger::condition("condition", REVIEW), REVIEW);
            hooks
                .on_approval_resolved(
                    &condition,
                    &ApprovalResult::Approved,
                    &ApprovalResolvedOutcome::Approved,
                )
                .await;
            assert_eq!(request.message, REVIEW);
            let events = manager.raw_events();
            assert_eq!(events.len(), 4);
            let text = serde_json::to_string(&events).unwrap();
            assert!(!text.contains(REVIEW), "{text}");
            assert_eq!(text.contains(BODY), include_args && cap > 0);
            if cap > 0 {
                assert_eq!(events[0].tags["raw_result"], "rejected");
                assert_eq!(events[2].tags["result"], "error");
            }
            let exported = manager
                .export()
                .await
                .unwrap()
                .paths
                .into_iter()
                .map(|p| std::fs::read_to_string(p).unwrap())
                .collect::<String>();
            assert!(!exported.contains(REVIEW));
            assert_eq!(exported.contains(BODY), include_args && cap > 0);
        }
    }
}

// Actual ordinary approval resolution must keep exact original/effective operands while its shared hook emits only permitted projections.
#[tokio::test]
async fn shared_executor_approval_resolution_does_not_export_exact_decisions() {
    const ORIGINAL: &str = "private-original-argument-fixture";
    const MODIFIED: &str = "private-modified-argument-fixture";
    const REASON: &str = "private-approval-reason-fixture";
    for include_args in [false, true] {
        for modified in [false, true] {
            let reports = Reports::new();
            let manager = ObservabilityManager::new(ObservabilityConfig {
                enabled: true,
                privacy: PrivacyConfig {
                    include_tool_args: include_args,
                    max_text_chars: 1024,
                    ..Default::default()
                },
                export: ExportConfig {
                    path: reports.0.to_string_lossy().into_owned(),
                    write_raw_events: true,
                    ..Default::default()
                },
                ..Default::default()
            });
            let handler = ai_agents_hitl::CallbackHandler::new(move |request: ApprovalRequest| {
                assert!(
                    matches!(request.trigger, ApprovalTrigger::Tool { args, .. } if args["content"]==ORIGINAL)
                );
                if modified {
                    ApprovalResult::modified(std::collections::HashMap::from([(
                        "content".into(),
                        json!(MODIFIED),
                    )]))
                } else {
                    ApprovalResult::rejected_with_reason(REASON)
                }
            });
            let agent = AgentBuilder::from_yaml("name: ApprovalProjection\nsystem_prompt: test\ntools: [failure]\nhitl:\n  tools:\n    failure: {require_approval: true}\n").unwrap()
                .llm(Arc::new(MockLLMProvider::new("unused"))).auto_configure_features().unwrap()
                .tool(Arc::new(Failure)).approval_handler(Arc::new(handler)).observability(manager.clone()).build().unwrap();
            let record = agent
                .invoke_tool(ToolExecutionRequest::new(
                    "approved",
                    "failure",
                    json!({"content":ORIGINAL}),
                    ToolCallSource::Manual,
                ))
                .await
                .unwrap();
            assert_eq!(record.arguments["content"], ORIGINAL);
            assert_eq!(record.executed, modified);
            if modified {
                assert_eq!(record.executed_arguments["content"], MODIFIED);
                assert_eq!(record.output, BODY);
            } else {
                assert_eq!(
                    record.approval.as_ref().unwrap().reason.as_deref(),
                    Some(REASON)
                );
            }
            let events = manager.raw_events();
            let text = serde_json::to_string(&events).unwrap();
            assert!(events.iter().any(|event| matches!(&event.event_type, ai_agents_observability::EventType::HitlApproval { trigger } if trigger=="resolved")));
            assert!(!text.contains(REASON) && !text.contains(BODY), "{text}");
            assert_eq!(text.contains(ORIGINAL), include_args);
            assert_eq!(text.contains(MODIFIED), include_args && modified);
        }
    }
}

// The actual shared executor's duplicate error-hook publication must not bypass wrapper privacy or redact trusted execution evidence.
#[tokio::test]
async fn shared_executor_keeps_exact_failure_record_but_hides_excluded_observations() {
    let reports = Reports::new();
    let manager = manager(false, 1024, true, &reports);
    let cursor = manager.event_cursor();
    let agent = AgentBuilder::new()
        .system_prompt("test")
        .llm(Arc::new(MockLLMProvider::new("unused")))
        .tool(Arc::new(Failure))
        .observability(manager.clone())
        .build()
        .unwrap();
    let record = agent
        .invoke_tool(ToolExecutionRequest::new(
            "failure",
            "failure",
            json!({}),
            ToolCallSource::Manual,
        ))
        .await
        .unwrap();
    assert!(record.executed && !record.success);
    assert_eq!(record.output, BODY);
    let events = manager.snapshot_since(cursor).events;
    assert!(
        events
            .iter()
            .any(|e| matches!(e.event_type, ai_agents_observability::EventType::Error))
    );
    assert!(!serde_json::to_string(&events).unwrap().contains(BODY));
}

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;

use ai_agents_core::{
    PathPolicyBinding, ResultLimitBinding, ResultLimitKind, Tool, ToolExecutionContext,
    ToolOperationKind, ToolPolicyBindings, ToolResult, ToolSafetyMetadata, ToolSideEffectLevel,
};

use crate::generate_schema;
use crate::types::{
    DiagnosticSeverity, DiagnosticsProviderSlot, DiagnosticsRequest, DiagnosticsResponse,
    QuestionHandlerSlot, QuestionRequest, QuestionResponse, TodoItem, TodoStatus, TodoStore,
};

const DEFAULT_MAX_RESULTS: usize = 200;
const DEFAULT_OUTPUT_CHARS: usize = 20_000;
const DEFAULT_SLEEP_MAX_MS: u64 = 30_000;

/// Returns host-provided compiler, linter, or editor diagnostics.
pub struct DiagnosticsTool {
    provider: DiagnosticsProviderSlot,
}

impl DiagnosticsTool {
    /// Create a diagnostics tool backed by a shared provider slot.
    pub fn new(provider: DiagnosticsProviderSlot) -> Self {
        Self { provider }
    }
}

/// Asks the user a structured clarification or preference question.
pub struct AskUserTool {
    handler: QuestionHandlerSlot,
}

impl AskUserTool {
    /// Create an ask-user tool backed by a shared handler slot.
    pub fn new(handler: QuestionHandlerSlot) -> Self {
        Self { handler }
    }
}

/// Maintains a session-local structured todo list.
pub struct TodoTool {
    store: TodoStore,
}

impl TodoTool {
    /// Create a todo tool backed by runtime-local storage.
    pub fn new(store: TodoStore) -> Self {
        Self { store }
    }
}

/// Waits for a bounded duration without shell access.
pub struct SleepTool;

impl SleepTool {
    /// Create a bounded wait tool.
    pub fn new() -> Self {
        Self
    }
}

impl Default for SleepTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct DiagnosticsInput {
    /// Optional file or directory path filter.
    #[serde(default)]
    path: Option<String>,
    /// Severity filter: error, warning, info, hint, or all.
    #[serde(default)]
    severity: Option<String>,
    /// Maximum returned diagnostics. Defaults to 200.
    #[serde(default)]
    max_results: Option<usize>,
}

#[derive(Debug, Serialize)]
struct DiagnosticsOutput {
    available: bool,
    diagnostics: Vec<crate::types::DiagnosticItem>,
    count: usize,
    truncated: bool,
    message: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct AskUserInput {
    /// Question text shown to the user.
    question: String,
    /// Selectable choices.
    #[serde(default)]
    options: Vec<String>,
    /// Allow selecting more than one option. Defaults to false.
    #[serde(default)]
    multi_select: bool,
    /// Allow free text when the host supports it. Defaults to true.
    #[serde(default = "default_true")]
    allow_other: bool,
    /// Default answer used when no interactive handler is available.
    #[serde(default)]
    default: Option<Value>,
    /// Seconds to wait for interactive answer.
    #[serde(default)]
    timeout_seconds: Option<u64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum TodoOperation {
    List,
    Set,
    Update,
    Clear,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct TodoInput {
    /// Operation: list, set, update, or clear.
    operation: TodoOperation,
    /// Full replacement items for set.
    #[serde(default)]
    items: Vec<TodoItemInput>,
    /// Task ID for update.
    #[serde(default)]
    id: Option<String>,
    /// New status for update.
    #[serde(default)]
    status: Option<TodoStatus>,
    /// New task content for update.
    #[serde(default)]
    content: Option<String>,
    /// New active display text for update.
    #[serde(default)]
    active_form: Option<String>,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct TodoItemInput {
    /// Stable task ID.
    id: String,
    /// Task description.
    content: String,
    /// Present-continuous display text.
    #[serde(default)]
    active_form: Option<String>,
    /// Task status. Defaults to pending.
    #[serde(default)]
    status: TodoStatus,
}

#[derive(Debug, Serialize)]
struct TodoOutput {
    operation: String,
    items: Vec<TodoItem>,
    count: usize,
    updated: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SleepInput {
    /// Duration to wait in milliseconds.
    duration_ms: u64,
    /// User-visible reason for waiting.
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct SleepOutput {
    slept_ms: u64,
    max_duration_ms: u64,
    reason: Option<String>,
}

#[async_trait]
impl Tool for DiagnosticsTool {
    fn id(&self) -> &str {
        "diagnostics"
    }

    fn name(&self) -> &str {
        "Diagnostics"
    }

    fn description(&self) -> &str {
        "Return compiler, linter, LSP, or host-editor diagnostics when a provider is configured."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<DiagnosticsInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: true,
            concurrency_safe: true,
            operation: ToolOperationKind::Diagnostics,
            side_effect_level: ToolSideEffectLevel::None,
            requires_network: false,
            destructive: false,
            open_world: false,
            host_dependent: true,
            requires_user_interaction: false,
            supports_cancellation: true,
            default_requires_approval: false,
            should_defer_schema: false,
            max_output_chars: Some(DEFAULT_OUTPUT_CHARS),
            max_result_size_chars: Some(DEFAULT_OUTPUT_CHARS),
        }
    }

    fn policy_bindings(&self) -> ToolPolicyBindings {
        ToolPolicyBindings {
            path_fields: vec![PathPolicyBinding::read("path").with_default_path(".")],
            result_limit_fields: vec![ResultLimitBinding::new(
                "max_results",
                ResultLimitKind::MaxResults,
            )],
            ..Default::default()
        }
    }

    async fn execute(&self, args: Value, ctx: ToolExecutionContext) -> ToolResult {
        let input: DiagnosticsInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let severity = match parse_severity(input.severity.as_deref()) {
            Ok(severity) => severity,
            Err(error) => return ToolResult::error(error),
        };
        let max_results = input
            .max_results
            .unwrap_or(DEFAULT_MAX_RESULTS)
            .min(DEFAULT_MAX_RESULTS * 5)
            .min(ctx.limits.max_results.unwrap_or(DEFAULT_MAX_RESULTS * 5));
        let provider = self.provider.read().clone();
        let response = provider
            .diagnostics(DiagnosticsRequest {
                path: input.path,
                severity,
                max_results: Some(max_results),
            })
            .await;
        diagnostics_to_result(response, max_results)
    }
}

#[async_trait]
impl Tool for AskUserTool {
    fn id(&self) -> &str {
        "ask_user"
    }

    fn name(&self) -> &str {
        "Ask User"
    }

    fn description(&self) -> &str {
        "Ask the user a structured question through the host question handler."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<AskUserInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: true,
            concurrency_safe: false,
            operation: ToolOperationKind::Interactive,
            side_effect_level: ToolSideEffectLevel::None,
            requires_network: false,
            destructive: false,
            open_world: false,
            host_dependent: true,
            requires_user_interaction: true,
            supports_cancellation: true,
            default_requires_approval: false,
            should_defer_schema: false,
            max_output_chars: Some(4_000),
            max_result_size_chars: Some(4_000),
        }
    }

    fn policy_bindings(&self) -> ToolPolicyBindings {
        ToolPolicyBindings::default()
    }

    async fn execute(&self, args: Value, _ctx: ToolExecutionContext) -> ToolResult {
        let input: AskUserInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let handler = { self.handler.read().clone() };
        if let Some(handler) = handler {
            let timeout = input.timeout_seconds.map(Duration::from_secs);
            let request = QuestionRequest {
                question: input.question,
                options: input.options,
                multi_select: input.multi_select,
                allow_other: input.allow_other,
                default: input.default,
                timeout_seconds: timeout.map(|duration| duration.as_secs()),
            };
            let response = if let Some(timeout) = timeout {
                match tokio::time::timeout(timeout, handler.ask_question(request.clone())).await {
                    Ok(response) => response,
                    Err(_) => default_question_response(request.default, true),
                }
            } else {
                handler.ask_question(request).await
            };
            return json_result(&response, None);
        }

        let response = default_question_response(input.default, false);
        json_result(&response, Some("question_unavailable"))
    }
}

#[async_trait]
impl Tool for TodoTool {
    fn id(&self) -> &str {
        "todo"
    }

    fn name(&self) -> &str {
        "Todo"
    }

    fn description(&self) -> &str {
        "Manage a session-local structured task list with list, set, update, and clear operations."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<TodoInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: false,
            concurrency_safe: false,
            operation: ToolOperationKind::Write,
            side_effect_level: ToolSideEffectLevel::LocalWrite,
            requires_network: false,
            destructive: false,
            open_world: false,
            host_dependent: false,
            requires_user_interaction: false,
            supports_cancellation: false,
            default_requires_approval: false,
            should_defer_schema: false,
            max_output_chars: Some(DEFAULT_OUTPUT_CHARS),
            max_result_size_chars: Some(DEFAULT_OUTPUT_CHARS),
        }
    }

    fn policy_bindings(&self) -> ToolPolicyBindings {
        ToolPolicyBindings {
            operation_fields: vec!["operation".to_string()],
            ..Default::default()
        }
    }

    async fn execute(&self, args: Value, _ctx: ToolExecutionContext) -> ToolResult {
        let input: TodoInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let (operation, updated) = match input.operation {
            TodoOperation::List => ("list", None),
            TodoOperation::Set => {
                let items = input.items.into_iter().map(TodoItem::from).collect();
                self.store.set(items);
                ("set", None)
            }
            TodoOperation::Update => {
                let Some(id) = input.id.as_deref() else {
                    return ToolResult::error("id is required for todo update");
                };
                let updated = self
                    .store
                    .update(id, input.content, input.active_form, input.status);
                ("update", Some(updated))
            }
            TodoOperation::Clear => {
                self.store.clear();
                ("clear", None)
            }
        };
        let items = self.store.list();
        let output = TodoOutput {
            operation: operation.to_string(),
            count: items.len(),
            items,
            updated,
        };
        json_result(&output, None)
    }
}

#[async_trait]
impl Tool for SleepTool {
    fn id(&self) -> &str {
        "sleep"
    }

    fn name(&self) -> &str {
        "Sleep"
    }

    fn description(&self) -> &str {
        "Wait for a bounded duration without shell access. Default maximum is 30000 ms."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<SleepInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: true,
            concurrency_safe: true,
            operation: ToolOperationKind::Wait,
            side_effect_level: ToolSideEffectLevel::None,
            requires_network: false,
            destructive: false,
            open_world: false,
            host_dependent: false,
            requires_user_interaction: false,
            supports_cancellation: true,
            default_requires_approval: false,
            should_defer_schema: false,
            max_output_chars: Some(2_000),
            max_result_size_chars: Some(2_000),
        }
    }

    fn policy_bindings(&self) -> ToolPolicyBindings {
        ToolPolicyBindings::default()
    }

    async fn execute(&self, args: Value, ctx: ToolExecutionContext) -> ToolResult {
        let input: SleepInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let max_duration_ms =
            DEFAULT_SLEEP_MAX_MS.min(ctx.limits.timeout_ms.unwrap_or(DEFAULT_SLEEP_MAX_MS));
        if input.duration_ms > max_duration_ms {
            return ToolResult::error(format!(
                "duration_ms exceeds max_duration_ms {}",
                max_duration_ms
            ));
        }
        tokio::time::sleep(Duration::from_millis(input.duration_ms)).await;
        let output = SleepOutput {
            slept_ms: input.duration_ms,
            max_duration_ms,
            reason: input.reason,
        };
        json_result(&output, None)
    }
}

impl From<TodoItemInput> for TodoItem {
    fn from(value: TodoItemInput) -> Self {
        Self {
            id: value.id,
            content: value.content,
            active_form: value.active_form,
            status: value.status,
        }
    }
}

fn parse_severity(value: Option<&str>) -> Result<Option<DiagnosticSeverity>, String> {
    match value.unwrap_or("all").to_ascii_lowercase().as_str() {
        "all" => Ok(None),
        "error" => Ok(Some(DiagnosticSeverity::Error)),
        "warning" | "warn" => Ok(Some(DiagnosticSeverity::Warning)),
        "info" => Ok(Some(DiagnosticSeverity::Info)),
        "hint" => Ok(Some(DiagnosticSeverity::Hint)),
        other => Err(format!("Invalid severity: {}", other)),
    }
}

fn diagnostics_to_result(response: DiagnosticsResponse, max_results: usize) -> ToolResult {
    let mut diagnostics = response.diagnostics;
    let truncated = diagnostics.len() > max_results;
    diagnostics.truncate(max_results);
    let output = DiagnosticsOutput {
        available: response.available,
        count: diagnostics.len(),
        truncated,
        diagnostics,
        message: response.message,
    };
    let mut metadata = HashMap::new();
    metadata.insert("available".to_string(), Value::Bool(output.available));
    metadata.insert("truncated".to_string(), Value::Bool(output.truncated));
    match serde_json::to_string(&output) {
        Ok(json) => ToolResult::ok_with_metadata(json, metadata),
        Err(error) => ToolResult::error(format!("Serialization error: {}", error)),
    }
}

fn default_question_response(default: Option<Value>, timed_out: bool) -> QuestionResponse {
    match default {
        Some(Value::String(text)) => QuestionResponse {
            answered: true,
            selected: vec![text],
            other_text: None,
            timed_out,
            unavailable: !timed_out,
        },
        Some(Value::Array(values)) => QuestionResponse {
            answered: true,
            selected: values
                .into_iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect(),
            other_text: None,
            timed_out,
            unavailable: !timed_out,
        },
        Some(other) => QuestionResponse {
            answered: true,
            selected: Vec::new(),
            other_text: Some(other.to_string()),
            timed_out,
            unavailable: !timed_out,
        },
        None => QuestionResponse {
            answered: false,
            selected: Vec::new(),
            other_text: None,
            timed_out,
            unavailable: !timed_out,
        },
    }
}

fn json_result<T: Serialize>(output: &T, note: Option<&str>) -> ToolResult {
    let json = match serde_json::to_string(output) {
        Ok(json) => json,
        Err(error) => return ToolResult::error(format!("Serialization error: {}", error)),
    };
    let mut metadata = HashMap::new();
    if let Some(note) = note {
        metadata.insert("note".to_string(), Value::String(note.to_string()));
    }
    ToolResult::ok_with_metadata(json, metadata)
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DiagnosticItem, StaticDiagnosticsProvider};
    use parking_lot::RwLock;
    use std::sync::Arc;

    fn value(output: &str) -> Value {
        serde_json::from_str(output).unwrap()
    }

    #[tokio::test]
    async fn diagnostics_uses_static_provider() {
        let provider =
            Arc::new(RwLock::new(
                Arc::new(StaticDiagnosticsProvider::new(vec![DiagnosticItem {
                    path: "src/lib.rs".to_string(),
                    line: Some(1),
                    column: Some(2),
                    severity: DiagnosticSeverity::Error,
                    source: Some("rustc".to_string()),
                    message: "broken".to_string(),
                    code: Some("E0001".to_string()),
                }])) as Arc<dyn crate::types::DiagnosticsProvider>,
            ));
        let result = DiagnosticsTool::new(provider)
            .execute(
                serde_json::json!({"severity": "error"}),
                ai_agents_core::ToolExecutionContext::test("test"),
            )
            .await;
        assert!(result.success);
        let output = value(&result.output);
        assert!(output["available"].as_bool().unwrap());
        assert_eq!(output["count"], 1);
    }

    #[tokio::test]
    async fn ask_user_uses_default_without_handler() {
        let slot = Arc::new(RwLock::new(None));
        let result = AskUserTool::new(slot)
            .execute(
                serde_json::json!({
                    "question": "Pick one",
                    "options": ["a", "b"],
                    "default": "a"
                }),
                ai_agents_core::ToolExecutionContext::test("test"),
            )
            .await;
        assert!(result.success);
        let output = value(&result.output);
        assert!(output["answered"].as_bool().unwrap());
        assert!(output["unavailable"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn todo_set_update_and_clear() {
        let tool = TodoTool::new(TodoStore::default());
        let set = tool
            .execute(
                serde_json::json!({
                    "operation": "set",
                    "items": [{"id": "a", "content": "Draft", "status": "pending"}]
                }),
                ai_agents_core::ToolExecutionContext::test("test"),
            )
            .await;
        assert!(set.success);
        let update = tool
            .execute(
                serde_json::json!({
                    "operation": "update",
                    "id": "a",
                    "status": "completed"
                }),
                ai_agents_core::ToolExecutionContext::test("test"),
            )
            .await;
        assert!(update.success);
        let output = value(&update.output);
        assert_eq!(output["items"][0]["status"], "completed");
        let clear = tool
            .execute(
                serde_json::json!({"operation": "clear"}),
                ai_agents_core::ToolExecutionContext::test("test"),
            )
            .await;
        assert!(clear.success);
        assert_eq!(value(&clear.output)["count"], 0);
    }

    #[tokio::test]
    async fn sleep_rejects_over_cap() {
        let result = SleepTool::new()
            .execute(
                serde_json::json!({"duration_ms": 30_001}),
                ai_agents_core::ToolExecutionContext::test("test"),
            )
            .await;
        assert!(!result.success);
    }
}

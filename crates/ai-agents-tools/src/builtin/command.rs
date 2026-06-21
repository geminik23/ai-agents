use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use ai_agents_core::{
    CommandPolicyBinding, PathAccessMode, PathBindingKind, PathPolicyBinding, ResultLimitBinding,
    ResultLimitKind, Tool, ToolCallClassification, ToolExecutionContext, ToolOperationKind,
    ToolPolicyBindings, ToolResult, ToolSafetyMetadata, ToolSideEffectLevel,
};

use crate::generate_schema;
use crate::types::{CommandRequest, CommandResponse, CommandRunnerSlot};

const DEFAULT_TIMEOUT_MS: u64 = 30_000;
const DEFAULT_MAX_OUTPUT_CHARS: usize = 20_000;

/// Runs allowlisted non-interactive commands through a host runner.
pub struct CommandTool {
    runner: CommandRunnerSlot,
}

impl CommandTool {
    /// Create a command tool backed by a shared runner slot.
    pub fn new(runner: CommandRunnerSlot) -> Self {
        Self { runner }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct CommandInput {
    /// Preferred command vector including executable and arguments.
    #[serde(default)]
    argv: Vec<String>,
    /// Compatibility command string. Shell syntax is rejected by default.
    #[serde(default)]
    command: Option<String>,
    /// Working directory. Defaults to current directory.
    #[serde(default)]
    cwd: Option<String>,
    /// Environment variables to pass when policy allows them.
    #[serde(default)]
    env: HashMap<String, String>,
    /// Timeout in milliseconds.
    #[serde(default)]
    timeout_ms: Option<u64>,
    /// Maximum combined output characters.
    #[serde(default)]
    max_output_chars: Option<usize>,
    /// User-visible reason for running the command.
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct CommandToolOutput {
    success: bool,
    exit_code: Option<i32>,
    termination: String,
    stdout: String,
    stderr: String,
    combined_output: String,
    truncated: bool,
    timed_out: bool,
    cwd: String,
    argv: Vec<String>,
    reason: Option<String>,
}

#[async_trait]
impl Tool for CommandTool {
    fn id(&self) -> &str {
        "command"
    }

    fn name(&self) -> &str {
        "Command"
    }

    fn description(&self) -> &str {
        "Run exact allowlisted non-interactive argv commands with timeout and bounded output."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<CommandInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: false,
            concurrency_safe: false,
            operation: ToolOperationKind::Command,
            side_effect_level: ToolSideEffectLevel::LocalWrite,
            requires_network: false,
            destructive: false,
            open_world: false,
            host_dependent: true,
            requires_user_interaction: false,
            supports_cancellation: true,
            default_requires_approval: true,
            should_defer_schema: false,
            max_output_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
            max_result_size_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
        }
    }

    fn classify_call(&self, _args: &Value) -> ToolCallClassification {
        let mut classification = ToolCallClassification::from_metadata(&self.safety_metadata());
        classification.safely_retryable = false;
        classification
    }

    fn policy_bindings(&self) -> ToolPolicyBindings {
        ToolPolicyBindings {
            command_fields: vec![
                CommandPolicyBinding::argv("argv"),
                CommandPolicyBinding::command("command"),
                CommandPolicyBinding::env("env"),
            ],
            path_fields: vec![
                PathPolicyBinding::new("cwd", PathAccessMode::ReadWrite, PathBindingKind::Cwd)
                    .with_default_path("."),
            ],
            result_limit_fields: vec![ResultLimitBinding::new(
                "max_output_chars",
                ResultLimitKind::MaxOutputChars,
            )],
            ..Default::default()
        }
    }

    async fn execute(&self, args: Value, ctx: ToolExecutionContext) -> ToolResult {
        let input: CommandInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let argv = match command_argv(&input) {
            Ok(argv) => argv,
            Err(error) => return ToolResult::error(error),
        };
        if argv.is_empty() {
            return ToolResult::error("argv must not be empty");
        }
        if input.command.is_some() && contains_shell_syntax(&argv.join(" ")) {
            return ToolResult::error("command string contains shell syntax denied by default");
        }
        let policy = CommandPolicySnapshot::from_context(&ctx.policy_snapshot);
        let env = filter_env(input.env, &policy);
        let timeout_ms = input
            .timeout_ms
            .unwrap_or(DEFAULT_TIMEOUT_MS)
            .min(ctx.limits.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS));
        let max_output_chars = input
            .max_output_chars
            .unwrap_or(DEFAULT_MAX_OUTPUT_CHARS)
            .min(
                ctx.limits
                    .max_output_chars
                    .unwrap_or(DEFAULT_MAX_OUTPUT_CHARS),
            );
        let cwd = input.cwd.unwrap_or_else(|| ".".to_string());
        let request = CommandRequest {
            argv: argv.clone(),
            cwd: Some(cwd.clone()),
            env,
            timeout_ms: Some(timeout_ms),
            max_output_chars: Some(max_output_chars),
            reason: input.reason.clone(),
        };
        let runner = self.runner.read().clone();
        let response = runner.run_command(request, ctx).await;
        command_response_to_result(response, cwd, argv, input.reason)
    }
}

#[derive(Debug, Default)]
struct CommandPolicySnapshot {
    env_passthrough: Vec<String>,
    redact_env: Vec<String>,
}

impl CommandPolicySnapshot {
    fn from_context(value: &Value) -> Self {
        let mut snapshot = Self::default();
        snapshot
            .env_passthrough
            .extend(strings_at(value, "env_passthrough"));
        snapshot.redact_env.extend(strings_at(value, "redact_env"));
        if let Some(commands) = value.get("commands") {
            snapshot
                .env_passthrough
                .extend(strings_at(commands, "env_passthrough"));
        }
        snapshot
    }
}

fn command_argv(input: &CommandInput) -> Result<Vec<String>, String> {
    if !input.argv.is_empty() {
        return Ok(input.argv.clone());
    }
    let Some(command) = input.command.as_deref() else {
        return Err("either argv or command is required".to_string());
    };
    if contains_shell_syntax(command) {
        return Err("command string contains shell syntax denied by default".to_string());
    }
    parse_command_words(command).ok_or_else(|| "command string could not be parsed".to_string())
}

fn filter_env(
    env: HashMap<String, String>,
    policy: &CommandPolicySnapshot,
) -> HashMap<String, String> {
    if policy.env_passthrough.is_empty() {
        return HashMap::new();
    }
    env.into_iter()
        .filter(|(key, _)| policy.env_passthrough.iter().any(|allowed| allowed == key))
        .filter(|(key, _)| !policy.redact_env.iter().any(|redacted| redacted == key))
        .collect()
}

fn strings_at(value: &Value, field: &str) -> Vec<String> {
    value
        .get(field)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect()
}

fn command_response_to_result(
    response: CommandResponse,
    cwd: String,
    argv: Vec<String>,
    reason: Option<String>,
) -> ToolResult {
    let output = CommandToolOutput {
        success: response.success,
        exit_code: response.exit_code,
        termination: response.termination,
        stdout: response.stdout,
        stderr: response.stderr,
        combined_output: response.combined_output,
        truncated: response.truncated,
        timed_out: response.timed_out,
        cwd,
        argv: if response.argv_redacted.is_empty() {
            redact_argv(&argv)
        } else {
            response.argv_redacted
        },
        reason,
    };
    let json = match serde_json::to_string(&output) {
        Ok(json) => json,
        Err(error) => return ToolResult::error(format!("Serialization error: {}", error)),
    };
    let mut metadata = HashMap::new();
    metadata.insert("truncated".to_string(), Value::Bool(output.truncated));
    metadata.insert("timed_out".to_string(), Value::Bool(output.timed_out));
    metadata.insert(
        "timeout_cleanup".to_string(),
        Value::String(if output.timed_out {
            "kill_on_drop".to_string()
        } else {
            "not_needed".to_string()
        }),
    );
    metadata.insert("argv".to_string(), serde_json::json!(output.argv));
    ToolResult::ok_with_metadata(json, metadata)
}

fn redact_argv(argv: &[String]) -> Vec<String> {
    let mut redacted = Vec::with_capacity(argv.len());
    let mut redact_next = false;
    for arg in argv {
        let lower = arg.to_ascii_lowercase();
        let sensitive = lower.contains("token")
            || lower.contains("secret")
            || lower.contains("password")
            || lower.contains("apikey")
            || lower.contains("api-key");
        if redact_next || sensitive {
            redacted.push("[redacted]".to_string());
        } else {
            redacted.push(arg.clone());
        }
        redact_next = matches!(
            lower.as_str(),
            "--token" | "--secret" | "--password" | "--api-key"
        );
    }
    redacted
}

fn contains_shell_syntax(value: &str) -> bool {
    const DENIED: &[char] = &[';', '&', '|', '<', '>', '`', '$', '\n', '\r'];
    value.chars().any(|ch| DENIED.contains(&ch))
        || value.contains("$(")
        || value.contains("${")
        || value.contains("<(")
        || value.contains(">(")
}

fn parse_command_words(value: &str) -> Option<Vec<String>> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    for ch in value.chars() {
        match (quote, ch) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), c) => current.push(c),
            (None, '\'' | '"') => quote = Some(ch),
            (None, c) if c.is_whitespace() => {
                if !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                }
            }
            (None, c) => current.push(c),
        }
    }
    if quote.is_some() {
        return None;
    }
    if !current.is_empty() {
        words.push(current);
    }
    (!words.is_empty()).then_some(words)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CommandResponse, StaticCommandRunner};
    use parking_lot::RwLock;
    use std::sync::Arc;

    #[tokio::test]
    async fn static_runner_executes_allowed_argv() {
        let mut responses = HashMap::new();
        responses.insert(
            vec!["cargo".to_string(), "fmt".to_string(), "--all".to_string()],
            CommandResponse {
                success: true,
                exit_code: Some(0),
                termination: "exited".to_string(),
                stdout: "ok".to_string(),
                combined_output: "ok".to_string(),
                argv_redacted: vec!["cargo".to_string(), "fmt".to_string(), "--all".to_string()],
                ..CommandResponse::default()
            },
        );
        let runner = Arc::new(RwLock::new(
            Arc::new(StaticCommandRunner::new(responses)) as Arc<_>
        ));
        let tool = CommandTool::new(runner);
        let result = tool
            .execute(
                serde_json::json!({"argv": ["cargo", "fmt", "--all"], "cwd": "."}),
                ToolExecutionContext::test("command"),
            )
            .await;
        assert!(result.success);
    }

    #[tokio::test]
    async fn command_metadata_uses_redacted_argv() {
        let mut responses = HashMap::new();
        responses.insert(
            vec![
                "deploy".to_string(),
                "--token".to_string(),
                "abc123".to_string(),
            ],
            CommandResponse {
                success: true,
                termination: "exited".to_string(),
                combined_output: "ok".to_string(),
                ..CommandResponse::default()
            },
        );
        let runner = Arc::new(RwLock::new(
            Arc::new(StaticCommandRunner::new(responses)) as Arc<_>
        ));
        let tool = CommandTool::new(runner);
        let result = tool
            .execute(
                serde_json::json!({"argv": ["deploy", "--token", "abc123"]}),
                ToolExecutionContext::test("command"),
            )
            .await;
        assert!(result.success);
        let metadata = result.metadata.unwrap();
        assert_eq!(
            metadata["argv"],
            serde_json::json!(["deploy", "[redacted]", "[redacted]"])
        );
    }

    #[tokio::test]
    async fn command_string_rejects_shell_syntax() {
        let runner = Arc::new(RwLock::new(
            Arc::new(StaticCommandRunner::default()) as Arc<_>
        ));
        let tool = CommandTool::new(runner);
        let result = tool
            .execute(
                serde_json::json!({"command": "cargo test && rm -rf target"}),
                ToolExecutionContext::test("command"),
            )
            .await;
        assert!(!result.success);
    }
}

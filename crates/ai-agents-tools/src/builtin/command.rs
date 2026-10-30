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
use crate::types::{CommandRequest, CommandResponse, CommandRunnerSlot, truncate_head_tail};

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
    fn prepare_write_footprint(
        &self,
        args: &Value,
        ctx: &ToolExecutionContext,
        max_targets: usize,
    ) -> ai_agents_core::Result<Option<ai_agents_core::autonomy::ToolWriteAdmission>> {
        let request = prepare_request(args, ctx).map_err(ai_agents_core::AgentError::Tool)?;
        let runner = self.runner.read().clone();
        let footprint = runner.declared_write_footprint(&request, ctx, max_targets)?;
        Ok(crate::footprint::prepare(&self.runner, runner, footprint))
    }
    fn declared_write_footprint(
        &self,
        args: &Value,
        ctx: &ToolExecutionContext,
        max_targets: usize,
    ) -> ai_agents_core::Result<Option<ai_agents_core::autonomy::ToolWriteFootprint>> {
        Ok(self
            .prepare_write_footprint(args, ctx, max_targets)?
            .map(|prepared| prepared.footprint))
    }
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
        let request = match prepare_request(&args, &ctx) {
            Ok(request) => request,
            Err(error) => return ToolResult::error(error),
        };
        let runner = self.runner.read().clone();
        if let Err(error) = crate::footprint::verify_host(&ctx, &runner) {
            return ToolResult::error(error.to_string());
        }
        let cwd = request.cwd.clone().unwrap_or_else(|| ".".into());
        let argv = request.argv.clone();
        let reason = request.reason.clone();
        let output_limit = ctx.limits.max_output_chars;
        let response = runner.run_command(request, ctx).await;
        command_response_to_result(response, cwd, argv, reason, output_limit)
    }
}

// Quotes and execution share policy-normalized argv, environment and limits rather than model-supplied hints.
fn prepare_request(args: &Value, ctx: &ToolExecutionContext) -> Result<CommandRequest, String> {
    let input: CommandInput =
        serde_json::from_value(args.clone()).map_err(|error| format!("Invalid input: {error}"))?;
    let argv = command_argv(&input)?;
    if argv.is_empty() {
        return Err("argv must not be empty".into());
    }
    if input.command.is_some() && contains_shell_syntax(&argv.join(" ")) {
        return Err("command string contains shell syntax denied by default".into());
    }
    let policy = CommandPolicySnapshot::from_context(&ctx.policy_snapshot);
    Ok(CommandRequest {
        argv,
        cwd: Some(input.cwd.unwrap_or_else(|| ".".into())),
        env: filter_env(input.env, &policy),
        timeout_ms: Some(
            input
                .timeout_ms
                .unwrap_or(DEFAULT_TIMEOUT_MS)
                .min(ctx.limits.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS)),
        ),
        max_output_chars: Some(
            input
                .max_output_chars
                .unwrap_or(DEFAULT_MAX_OUTPUT_CHARS)
                .min(
                    ctx.limits
                        .max_output_chars
                        .unwrap_or(DEFAULT_MAX_OUTPUT_CHARS),
                ),
        ),
        reason: input.reason,
    })
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
    output_limit: Option<usize>,
) -> ToolResult {
    let mut output = CommandToolOutput {
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
    let json = match serialize_bounded_command_output(&mut output, output_limit) {
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

// Keep machine-readable exit status and the actual final log text inside the executor limit whenever the fixed envelope fits.
fn serialize_bounded_command_output(
    output: &mut CommandToolOutput,
    limit: Option<usize>,
) -> Result<String, serde_json::Error> {
    let Some(limit) = limit else {
        return serde_json::to_string(output);
    };
    for value in [
        &mut output.stdout,
        &mut output.stderr,
        &mut output.combined_output,
    ] {
        let (bounded, truncated) = truncate_head_tail(value, limit);
        *value = bounded;
        output.truncated |= truncated;
    }
    let full = serde_json::to_string(output)?;
    if full.chars().count() <= limit {
        return Ok(full);
    }

    let stdout = std::mem::take(&mut output.stdout);
    let stderr = std::mem::take(&mut output.stderr);
    let combined = std::mem::take(&mut output.combined_output);
    if stdout.is_empty() && stderr.is_empty() && combined.is_empty() {
        return Ok(full);
    }
    let producer_truncated = output.truncated;
    output.truncated = true;
    let base = serde_json::to_string(output)?;
    // A limit smaller than the fixed fields cannot carry a valid envelope; the executor retains its existing truncation signal.
    if base.chars().count() > limit {
        return Ok(base);
    }
    let mut remaining = limit - base.chars().count();
    let stderr_cost = serialized_string_content_len(&stderr)?;
    let stderr_budget = if stderr_cost <= remaining {
        stderr_cost
    } else {
        remaining.div_ceil(2)
    };
    let (mut bounded_stderr, cost) = fit_command_output_field(&stderr, stderr_budget, true)?;
    remaining -= cost;
    let combined_cost = serialized_string_content_len(&combined)?;
    let combined_budget = if combined_cost <= remaining {
        combined_cost
    } else if stderr.is_empty() && combined == stdout {
        remaining
    } else {
        remaining.div_ceil(2)
    };
    let (mut bounded_combined, cost) = fit_command_output_field(&combined, combined_budget, true)?;
    remaining -= cost;
    let (bounded_stdout, cost) = fit_command_output_field(&stdout, remaining, false)?;
    remaining -= cost;

    // Unused space from short fields is returned to the error tail before the combined preview.
    let (expanded_stderr, cost) = fit_command_output_field(
        &stderr,
        serialized_string_content_len(&bounded_stderr)? + remaining,
        true,
    )?;
    remaining -= cost - serialized_string_content_len(&bounded_stderr)?;
    bounded_stderr = expanded_stderr;
    let (expanded_combined, _) = fit_command_output_field(
        &combined,
        serialized_string_content_len(&bounded_combined)? + remaining,
        true,
    )?;
    bounded_combined = expanded_combined;
    output.stdout = bounded_stdout;
    output.stderr = bounded_stderr;
    output.combined_output = bounded_combined;

    // A shorter boolean spelling must not claim truncation when every original log is still present.
    if !producer_truncated
        && output.stdout == stdout
        && output.stderr == stderr
        && output.combined_output == combined
    {
        let field = if !output.stdout.is_empty() {
            &mut output.stdout
        } else if !output.combined_output.is_empty() {
            &mut output.combined_output
        } else {
            &mut output.stderr
        };
        *field = truncate_head_tail(field, field.chars().count() - 1).0;
    }
    serde_json::to_string(output)
}

// JSON escape costs count toward the existing output cap, including control characters and Unicode.
fn serialized_string_content_len(value: &str) -> Result<usize, serde_json::Error> {
    Ok(serde_json::to_string(value)?.chars().count() - 2)
}

// Marker-free head/tail lengths grow monotonically, so binary search cannot skip a fitting escaped value.
fn fit_command_output_field(
    value: &str,
    budget: usize,
    prefer_tail: bool,
) -> Result<(String, usize), serde_json::Error> {
    let mut low = 0usize;
    let mut high = value.chars().count().saturating_add(1);
    let mut best = String::new();
    let mut best_cost = 0;
    while low < high {
        let middle = low + (high - low) / 2;
        let candidate = if prefer_tail {
            let mut suffix: Vec<_> = value.chars().rev().take(middle).collect();
            suffix.reverse();
            suffix.into_iter().collect()
        } else {
            truncate_head_tail(value, middle).0
        };
        let cost = serialized_string_content_len(&candidate)?;
        if cost <= budget {
            best = candidate;
            best_cost = cost;
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    Ok((best, best_cost))
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
    async fn output_and_edit_regression_command_envelope_fits_and_keeps_error_tail() {
        let argv = vec!["cargo".to_string(), "check".to_string()];
        let stdout = format!("start {} end", "한글\\\"".repeat(300));
        let stderr = format!("warnings {} FINAL-ERROR", "x".repeat(2_000));
        let mut responses = HashMap::new();
        responses.insert(
            argv.clone(),
            CommandResponse {
                success: false,
                exit_code: Some(1),
                termination: "exited".into(),
                stdout: stdout.clone(),
                stderr: stderr.clone(),
                combined_output: format!("{stdout}\n{stderr}"),
                ..CommandResponse::default()
            },
        );
        let runner = Arc::new(RwLock::new(
            Arc::new(StaticCommandRunner::new(responses)) as Arc<_>
        ));
        let mut context = ToolExecutionContext::test("command");
        context.limits.max_output_chars = Some(512);
        let result = CommandTool::new(runner)
            .execute(serde_json::json!({"argv": argv}), context)
            .await;

        assert!(
            result.success,
            "a nonzero exit is a successfully observed tool result"
        );
        assert!(result.output.chars().count() <= 512);
        let output: Value = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output["success"], false);
        assert_eq!(output["exit_code"], 1);
        assert_eq!(output["termination"], "exited");
        assert_eq!(output["truncated"], true);
        assert!(
            output["combined_output"]
                .as_str()
                .unwrap()
                .contains("FINAL-ERROR")
        );
    }

    #[test]
    fn output_and_edit_regression_preserves_a_short_error_before_duplicate_output() {
        let output = CommandToolOutput {
            success: false,
            exit_code: Some(1),
            termination: "exited".into(),
            stdout: "x".repeat(100),
            stderr: "fatal: boom".into(),
            combined_output: format!("{}fatal: boom", "x".repeat(100)),
            truncated: false,
            timed_out: false,
            cwd: ".".into(),
            argv: vec!["test".into()],
            reason: None,
        };
        let mut empty = CommandToolOutput {
            stdout: String::new(),
            stderr: String::new(),
            combined_output: String::new(),
            truncated: true,
            ..output
        };
        let base = serde_json::to_string(&empty).unwrap().chars().count();
        empty.stdout = "x".repeat(100);
        empty.stderr = "fatal: boom".into();
        empty.combined_output = format!("{}fatal: boom", "x".repeat(100));
        empty.truncated = false;
        let json = serialize_bounded_command_output(&mut empty, Some(base + 15)).unwrap();
        let parsed: Value = serde_json::from_str(&json).unwrap();
        assert!(json.chars().count() <= base + 15);
        assert_eq!(parsed["stderr"], "fatal: boom");
        assert_eq!(parsed["truncated"], true);
    }

    #[test]
    fn output_and_edit_regression_keeps_a_long_final_error_in_the_combined_field() {
        let error = "ERROR-01234567890123456789012345";
        let text = format!("{}{error}", "x".repeat(200));
        let output = CommandToolOutput {
            success: false,
            exit_code: Some(1),
            termination: "exited".into(),
            stdout: text.clone(),
            stderr: String::new(),
            combined_output: text,
            truncated: false,
            timed_out: false,
            cwd: ".".into(),
            argv: vec!["test".into()],
            reason: None,
        };
        let mut base = CommandToolOutput {
            stdout: String::new(),
            combined_output: String::new(),
            truncated: true,
            ..output
        };
        let limit = serde_json::to_string(&base).unwrap().chars().count() + 60;
        base.stdout = format!("{}{error}", "x".repeat(200));
        base.combined_output = base.stdout.clone();
        base.truncated = false;
        let json = serialize_bounded_command_output(&mut base, Some(limit)).unwrap();
        assert!(json.chars().count() <= limit);
        let result: Value = serde_json::from_str(&json).unwrap();
        assert!(result["combined_output"].as_str().unwrap().ends_with(error));
    }

    #[test]
    fn output_and_edit_regression_empty_logs_do_not_claim_truncation() {
        let mut output = CommandToolOutput {
            success: true,
            exit_code: Some(0),
            termination: "exited".into(),
            stdout: String::new(),
            stderr: String::new(),
            combined_output: String::new(),
            truncated: false,
            timed_out: false,
            cwd: "a".repeat(100),
            argv: vec!["test".into()],
            reason: None,
        };
        let full_length = serde_json::to_string(&output).unwrap().chars().count();
        let json = serialize_bounded_command_output(&mut output, Some(full_length - 1)).unwrap();
        assert!(json.chars().count() > full_length - 1);
        assert!(!output.truncated);
        assert_eq!(
            serde_json::from_str::<Value>(&json).unwrap()["truncated"],
            false
        );
    }

    #[test]
    fn output_and_edit_regression_escaped_output_search_uses_available_budget() {
        let value = format!("{}{}E", "x".repeat(100), "\0".repeat(19));
        for budget in 0..100 {
            let (actual, cost) = fit_command_output_field(&value, budget, false).unwrap();
            assert!(cost <= budget);
            for size in 0..=value.chars().count() {
                let candidate = truncate_head_tail(&value, size).0;
                if serialized_string_content_len(&candidate).unwrap() <= budget {
                    assert!(actual.chars().count() >= candidate.chars().count());
                }
            }
        }
    }

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

use std::collections::HashMap;
use std::net::IpAddr;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use tracing::debug;

use super::config::*;
use ai_agents_core::{
    CommandBindingKind, CommandPolicyBinding, DomainPolicyBinding, PathAccessMode,
    PathPolicyBinding, Result, ResultLimitBinding, ResultLimitKind, ToolCallClassification,
    ToolExecutionLimits, ToolPolicyBindings, ToolSafetyMetadata,
};
use serde_json::Value;

#[derive(Debug, Default)]
struct ToolCallTracker {
    calls: HashMap<String, Vec<Instant>>,
}

impl ToolCallTracker {
    fn record_call(&mut self, tool_id: &str) {
        self.calls
            .entry(tool_id.to_string())
            .or_default()
            .push(Instant::now());
    }

    fn get_calls_in_window(&self, tool_id: &str, window_seconds: u64) -> usize {
        let now = Instant::now();
        let window = std::time::Duration::from_secs(window_seconds);

        self.calls
            .get(tool_id)
            .map(|calls| {
                calls
                    .iter()
                    .filter(|t| now.duration_since(**t) < window)
                    .count()
            })
            .unwrap_or(0)
    }

    fn reset(&mut self) {
        self.calls.clear();
    }
}

#[derive(Debug, Clone)]
pub struct ToolSecurityEngine {
    config: ToolSecurityConfig,
    tool_call_tracker: Arc<RwLock<ToolCallTracker>>,
    policy_version: u64,
}

impl ToolSecurityEngine {
    pub fn new(config: ToolSecurityConfig) -> Self {
        Self {
            config,
            tool_call_tracker: Arc::new(RwLock::new(ToolCallTracker::default())),
            policy_version: 1,
        }
    }

    pub fn config(&self) -> &ToolSecurityConfig {
        &self.config
    }

    pub fn policy_version(&self) -> u64 {
        self.policy_version
    }

    pub fn prepare_tool_arguments(&self, tool_id: &str, args: &Value) -> Value {
        let bindings = legacy_policy_bindings(tool_id);
        self.prepare_tool_arguments_with_bindings(tool_id, args, &bindings)
    }

    pub fn prepare_tool_arguments_with_bindings(
        &self,
        tool_id: &str,
        args: &Value,
        bindings: &ToolPolicyBindings,
    ) -> Value {
        if !self.config.enabled {
            return args.clone();
        }
        let mut prepared = args.clone();
        let Some(tool_config) = self.config.tools.get(tool_id) else {
            return prepared;
        };
        normalize_default_path_arguments(bindings, &mut prepared);
        apply_policy_caps(tool_config, bindings, &mut prepared);
        prepared
    }

    pub fn attach_internal_tool_policy(&self, _tool_id: &str, args: &Value) -> Value {
        args.clone()
    }

    pub fn get_tool_output_cap(
        &self,
        tool_id: &str,
        classification_cap: Option<usize>,
    ) -> Option<usize> {
        let policy_cap = self
            .config
            .enabled
            .then(|| self.config.tools.get(tool_id))
            .flatten()
            .and_then(|config| config.max_output_chars);
        min_optional_usize(classification_cap, policy_cap)
    }

    pub fn effective_limits(
        &self,
        tool_id: &str,
        safety: &ToolSafetyMetadata,
        classification: &ToolCallClassification,
    ) -> ToolExecutionLimits {
        let policy = self
            .config
            .enabled
            .then(|| self.config.tools.get(tool_id))
            .flatten();
        ToolExecutionLimits {
            timeout_ms: Some(self.get_tool_timeout(tool_id)),
            max_output_chars: min_optional_usize(
                classification.max_output_chars,
                policy.and_then(|config| config.max_output_chars),
            ),
            max_result_chars: safety.max_result_size_chars,
            max_results: policy.and_then(|config| config.max_results),
            max_file_size_bytes: policy.and_then(|config| config.max_file_size_bytes),
            max_response_bytes: policy.and_then(|config| config.max_response_bytes),
            max_redirects: policy.and_then(|config| config.max_redirects),
            max_changed_files: policy.and_then(|config| config.max_changed_files),
            max_changed_lines: policy.and_then(|config| config.max_changed_lines),
        }
    }

    pub fn policy_snapshot(&self, tool_id: &str) -> Value {
        if !self.config.enabled {
            return Value::Null;
        }
        self.config
            .tools
            .get(tool_id)
            .and_then(|config| serde_json::to_value(config).ok())
            .unwrap_or(Value::Null)
    }

    pub fn custom_config(&self, tool_id: &str) -> Value {
        if !self.config.enabled {
            return Value::Null;
        }
        self.config
            .tools
            .get(tool_id)
            .map(|config| Value::Object(config.config.clone().into_iter().collect()))
            .unwrap_or(Value::Null)
    }

    pub async fn check_tool_execution(
        &self,
        tool_id: &str,
        args: &serde_json::Value,
    ) -> Result<SecurityCheckResult> {
        let bindings = legacy_policy_bindings(tool_id);
        self.check_tool_execution_with_bindings(tool_id, args, &bindings)
            .await
    }

    pub async fn check_tool_execution_with_bindings(
        &self,
        tool_id: &str,
        args: &serde_json::Value,
        bindings: &ToolPolicyBindings,
    ) -> Result<SecurityCheckResult> {
        if !self.config.enabled {
            return Ok(SecurityCheckResult::Allow);
        }

        let tool_config = match self.config.tools.get(tool_id) {
            Some(config) => config,
            None if self.config.fail_closed => {
                return Ok(SecurityCheckResult::Block {
                    reason: format!("Tool '{}' has no explicit security policy", tool_id),
                });
            }
            None => {
                self.tool_call_tracker.write().record_call(tool_id);
                debug!(tool_id = %tool_id, "Tool execution allowed by legacy open policy");
                return Ok(SecurityCheckResult::Allow);
            }
        };

        if !tool_config.enabled {
            return Ok(SecurityCheckResult::Unavailable {
                reason: format!("Tool '{}' is disabled", tool_id),
            });
        }

        if let Some(rate_limit) = tool_config.rate_limit {
            let calls = self
                .tool_call_tracker
                .read()
                .get_calls_in_window(tool_id, 60);
            if calls >= rate_limit as usize {
                return Ok(SecurityCheckResult::Block {
                    reason: format!(
                        "Rate limit exceeded for tool '{}': {} calls per minute",
                        tool_id, rate_limit
                    ),
                });
            }
        }

        if let Some(result) =
            validate_policy_bindings(tool_id, tool_config, bindings, self.config.fail_closed)
        {
            return Ok(result);
        }

        if let Some(result) = self.check_domain_policy(tool_id, tool_config, args, bindings) {
            return Ok(result);
        }

        if let Some(result) = self.check_path_policy(tool_id, tool_config, args, bindings) {
            return Ok(result);
        }

        if let Some(result) = self.check_operation_policy(tool_id, tool_config, args, bindings) {
            return Ok(result);
        }

        if let Some(result) = self.check_command_policy(tool_id, tool_config, args, bindings) {
            return Ok(result);
        }

        if tool_config.require_confirmation {
            let message = tool_config
                .confirmation_message
                .clone()
                .unwrap_or_else(|| format!("Confirm execution of tool '{}' ?", tool_id));
            return Ok(SecurityCheckResult::RequireConfirmation { message });
        }

        self.tool_call_tracker.write().record_call(tool_id);
        debug!(tool_id = %tool_id, "Tool execution allowed");

        Ok(SecurityCheckResult::Allow)
    }

    pub fn check_command_execution(
        &self,
        tool_id: &str,
        command: &str,
        args: &[String],
    ) -> SecurityCheckResult {
        if !self.config.enabled {
            return SecurityCheckResult::Allow;
        }
        let Some(tool_config) = self.config.tools.get(tool_id) else {
            return if self.config.fail_closed {
                SecurityCheckResult::Block {
                    reason: format!("Tool '{}' has no explicit command policy", tool_id),
                }
            } else {
                SecurityCheckResult::Allow
            };
        };
        let value = serde_json::json!({
            "command": command,
            "argv": std::iter::once(command.to_string()).chain(args.iter().cloned()).collect::<Vec<_>>()
        });
        let bindings = legacy_policy_bindings(tool_id);
        self.check_command_policy(tool_id, tool_config, &value, &bindings)
            .unwrap_or(SecurityCheckResult::Allow)
    }

    fn check_domain_policy(
        &self,
        tool_id: &str,
        tool_config: &ToolPolicyConfig,
        args: &serde_json::Value,
        bindings: &ToolPolicyBindings,
    ) -> Option<SecurityCheckResult> {
        let values = bound_domain_values(args, bindings);
        if values.is_empty() {
            return missing_bound_value_result(
                tool_id,
                "domain",
                domain_policy_configured(tool_config),
                self.config.fail_closed,
            );
        }
        for value in values {
            let parsed = if value.is_url {
                match reqwest::Url::parse(&value.value) {
                    Ok(parsed) => parsed,
                    Err(_) => {
                        return Some(SecurityCheckResult::Block {
                            reason: format!("URL is invalid for tool '{}'", tool_id),
                        });
                    }
                }
            } else {
                match reqwest::Url::parse(&format!("https://{}", value.value)) {
                    Ok(parsed) => parsed,
                    Err(_) => {
                        return Some(SecurityCheckResult::Block {
                            reason: format!("Domain is invalid for tool '{}'", tool_id),
                        });
                    }
                }
            };
            let host = parsed.host_str().map(normalize_host)?;

            if !tool_config.allowed_schemes.is_empty()
                && !tool_config
                    .allowed_schemes
                    .iter()
                    .any(|scheme| scheme.eq_ignore_ascii_case(parsed.scheme()))
            {
                return Some(SecurityCheckResult::Block {
                    reason: format!(
                        "URL scheme '{}' is not allowed for tool '{}'",
                        parsed.scheme(),
                        tool_id
                    ),
                });
            }

            if !tool_config.allowed_ports.is_empty() {
                let port = parsed.port_or_known_default().unwrap_or(0);
                if !tool_config.allowed_ports.contains(&port) {
                    return Some(SecurityCheckResult::Block {
                        reason: format!(
                            "URL port '{}' is not allowed for tool '{}'",
                            port, tool_id
                        ),
                    });
                }
            }

            if tool_config.blocked_private_networks && host_is_private_or_local(&host) {
                return Some(SecurityCheckResult::Block {
                    reason: format!(
                        "Private, localhost, link-local, or metadata host is blocked for tool '{}'",
                        tool_id
                    ),
                });
            }

            let denied = tool_config
                .blocked_domains
                .iter()
                .chain(tool_config.domains.deny.iter());
            for pattern in denied {
                if host_matches(pattern, &host) {
                    return Some(SecurityCheckResult::Block {
                        reason: format!("Domain '{}' is blocked for tool '{}'", pattern, tool_id),
                    });
                }
            }

            for pattern in &tool_config.domains.unavailable {
                if host_matches(pattern, &host) {
                    return Some(SecurityCheckResult::Unavailable {
                        reason: format!(
                            "Domain '{}' is unavailable for tool '{}'",
                            pattern, tool_id
                        ),
                    });
                }
            }

            for pattern in &tool_config.domains.requires_approval {
                if host_matches(pattern, &host) {
                    return Some(SecurityCheckResult::RequireConfirmation {
                        message: format!(
                            "Confirm access to domain '{}' for tool '{}' ?",
                            host, tool_id
                        ),
                    });
                }
            }

            let allowed: Vec<&String> = tool_config
                .allowed_domains
                .iter()
                .chain(tool_config.domains.allow.iter())
                .collect();
            if !allowed.is_empty() && !allowed.iter().any(|pattern| host_matches(pattern, &host)) {
                return Some(SecurityCheckResult::Block {
                    reason: format!("URL domain not in allowed list for tool '{}'", tool_id),
                });
            }
        }

        None
    }

    fn check_path_policy(
        &self,
        tool_id: &str,
        tool_config: &ToolPolicyConfig,
        args: &serde_json::Value,
        bindings: &ToolPolicyBindings,
    ) -> Option<SecurityCheckResult> {
        let values = bound_path_values(args, bindings);
        if values.is_empty() {
            return missing_bound_value_result(
                tool_id,
                "path",
                path_policy_configured(tool_config),
                self.config.fail_closed,
            );
        }
        for value in values {
            let normalized = normalize_path(&value.path);

            for pattern in tool_config
                .blocked_paths
                .iter()
                .chain(tool_config.paths.deny.iter())
            {
                if path_matches(pattern, &normalized) {
                    return Some(SecurityCheckResult::Block {
                        reason: format!("Path is blocked for tool '{}'", tool_id),
                    });
                }
            }

            for pattern in tool_config.paths.unavailable.iter() {
                if path_matches(pattern, &normalized) {
                    return Some(SecurityCheckResult::Unavailable {
                        reason: format!("Path is unavailable for tool '{}'", tool_id),
                    });
                }
            }

            for pattern in tool_config.paths.requires_approval.iter() {
                if path_matches(pattern, &normalized) {
                    return Some(SecurityCheckResult::RequireConfirmation {
                        message: format!(
                            "Confirm access to path '{}' for tool '{}' ?",
                            value.path, tool_id
                        ),
                    });
                }
            }

            let allowed = allowed_paths_for_mode(tool_config, value.mode);
            if !allowed.is_empty()
                && !allowed
                    .iter()
                    .any(|pattern| path_matches(pattern, &normalized))
            {
                return Some(SecurityCheckResult::Block {
                    reason: format!("Path not in allowed list for tool '{}'", tool_id),
                });
            }
        }

        None
    }

    fn check_operation_policy(
        &self,
        tool_id: &str,
        tool_config: &ToolPolicyConfig,
        args: &serde_json::Value,
        bindings: &ToolPolicyBindings,
    ) -> Option<SecurityCheckResult> {
        let operations = bound_operation_values(args, bindings);
        if operations.is_empty() {
            return missing_bound_value_result(
                tool_id,
                "operation",
                operation_policy_configured(tool_config),
                self.config.fail_closed,
            );
        }
        for operation in operations {
            if contains_casefold(&tool_config.operations.deny, &operation) {
                return Some(SecurityCheckResult::Block {
                    reason: format!(
                        "Operation '{}' is blocked for tool '{}'",
                        operation, tool_id
                    ),
                });
            }
            if contains_casefold(&tool_config.operations.unavailable, &operation) {
                return Some(SecurityCheckResult::Unavailable {
                    reason: format!(
                        "Operation '{}' is unavailable for tool '{}'",
                        operation, tool_id
                    ),
                });
            }
            if contains_casefold(&tool_config.operations.requires_approval, &operation) {
                return Some(SecurityCheckResult::RequireConfirmation {
                    message: format!("Confirm operation '{}' for tool '{}' ?", operation, tool_id),
                });
            }
            if !tool_config.operations.allow.is_empty()
                && !contains_casefold(&tool_config.operations.allow, &operation)
            {
                return Some(SecurityCheckResult::Block {
                    reason: format!(
                        "Operation '{}' is not allowed for tool '{}'",
                        operation, tool_id
                    ),
                });
            }
        }

        None
    }

    fn check_command_policy(
        &self,
        tool_id: &str,
        tool_config: &ToolPolicyConfig,
        args: &serde_json::Value,
        bindings: &ToolPolicyBindings,
    ) -> Option<SecurityCheckResult> {
        let commands = bound_command_values(args, bindings);
        if commands.is_empty() {
            return missing_bound_value_result(
                tool_id,
                "command",
                command_policy_configured(tool_config),
                self.config.fail_closed,
            );
        }
        for command in commands {
            let command = command.trim();

            if contains_casefold(&tool_config.commands.deny, command) {
                return Some(SecurityCheckResult::Block {
                    reason: format!("Command '{}' is blocked for tool '{}'", command, tool_id),
                });
            }
            if contains_casefold(&tool_config.commands.unavailable, command) {
                return Some(SecurityCheckResult::Unavailable {
                    reason: format!(
                        "Command '{}' is unavailable for tool '{}'",
                        command, tool_id
                    ),
                });
            }
            if contains_casefold(&tool_config.commands.requires_approval, command) {
                return Some(SecurityCheckResult::RequireConfirmation {
                    message: format!("Confirm command '{}' for tool '{}' ?", command, tool_id),
                });
            }
            if !tool_config.commands.allow.is_empty()
                && !contains_casefold(&tool_config.commands.allow, command)
            {
                return Some(SecurityCheckResult::Block {
                    reason: format!(
                        "Command '{}' is not allowed for tool '{}'",
                        command, tool_id
                    ),
                });
            }
        }

        None
    }

    pub fn get_tool_timeout(&self, tool_id: &str) -> u64 {
        self.config
            .tools
            .get(tool_id)
            .and_then(|c| c.timeout_ms)
            .unwrap_or(self.config.default_timeout_ms)
    }

    pub fn reset_session(&self) {
        self.tool_call_tracker.write().reset();
    }
}

impl Default for ToolSecurityEngine {
    fn default() -> Self {
        Self::new(ToolSecurityConfig::default())
    }
}

fn normalize_default_path_arguments(bindings: &ToolPolicyBindings, args: &mut Value) {
    for binding in &bindings.path_fields {
        let Some(default_path) = binding.default_path.as_deref() else {
            continue;
        };
        if value_at_path(args, &binding.field).is_none() {
            set_root_value(
                args,
                &binding.field,
                Value::String(default_path.to_string()),
            );
        }
    }
}

fn apply_policy_caps(config: &ToolPolicyConfig, bindings: &ToolPolicyBindings, args: &mut Value) {
    let Some(obj) = args.as_object_mut() else {
        return;
    };
    for binding in &bindings.result_limit_fields {
        match binding.kind {
            ResultLimitKind::MaxResults | ResultLimitKind::Pagination => {
                apply_usize_cap(obj, &binding.field, config.max_results);
            }
            ResultLimitKind::MaxLines => {
                apply_usize_cap(obj, &binding.field, config.max_results);
            }
            ResultLimitKind::MaxOutputChars => {
                apply_usize_cap(obj, &binding.field, config.max_output_chars);
            }
            ResultLimitKind::MaxFileSizeBytes => {
                apply_u64_cap(obj, &binding.field, config.max_file_size_bytes);
            }
            ResultLimitKind::MaxResponseBytes => {
                apply_usize_cap(obj, &binding.field, config.max_response_bytes);
            }
            ResultLimitKind::MaxRedirects => {
                apply_usize_cap(obj, &binding.field, config.max_redirects);
            }
        }
    }
}

fn legacy_policy_bindings(tool_id: &str) -> ToolPolicyBindings {
    match tool_id {
        "glob" => ToolPolicyBindings {
            path_fields: vec![PathPolicyBinding::read("path").with_default_path(".")],
            result_limit_fields: vec![ResultLimitBinding::new(
                "max_results",
                ResultLimitKind::MaxResults,
            )],
            ..Default::default()
        },
        "grep" => ToolPolicyBindings {
            path_fields: vec![PathPolicyBinding::read("path").with_default_path(".")],
            result_limit_fields: vec![
                ResultLimitBinding::new("max_results", ResultLimitKind::MaxResults),
                ResultLimitBinding::new("max_file_size_bytes", ResultLimitKind::MaxFileSizeBytes),
                ResultLimitBinding::new("max_output_chars", ResultLimitKind::MaxOutputChars),
            ],
            ..Default::default()
        },
        "file_read" => ToolPolicyBindings {
            path_fields: vec![PathPolicyBinding::read("path")],
            result_limit_fields: vec![
                ResultLimitBinding::new("max_bytes", ResultLimitKind::MaxFileSizeBytes),
                ResultLimitBinding::new("max_lines", ResultLimitKind::MaxLines),
            ],
            ..Default::default()
        },
        "file_list" => ToolPolicyBindings {
            path_fields: vec![PathPolicyBinding::read("path")],
            result_limit_fields: vec![ResultLimitBinding::new(
                "max_results",
                ResultLimitKind::MaxResults,
            )],
            ..Default::default()
        },
        "file_info" => ToolPolicyBindings {
            path_fields: vec![PathPolicyBinding::read("path")],
            ..Default::default()
        },
        "git_status" => ToolPolicyBindings {
            path_fields: vec![PathPolicyBinding::read("path").with_default_path(".")],
            result_limit_fields: vec![ResultLimitBinding::new(
                "max_results",
                ResultLimitKind::MaxResults,
            )],
            ..Default::default()
        },
        "git_diff" => ToolPolicyBindings {
            path_fields: vec![PathPolicyBinding::read("path").with_default_path(".")],
            result_limit_fields: vec![ResultLimitBinding::new(
                "max_output_chars",
                ResultLimitKind::MaxOutputChars,
            )],
            ..Default::default()
        },
        "diagnostics" => ToolPolicyBindings {
            path_fields: vec![PathPolicyBinding::read("path").with_default_path(".")],
            result_limit_fields: vec![ResultLimitBinding::new(
                "max_results",
                ResultLimitKind::MaxResults,
            )],
            ..Default::default()
        },
        "web_fetch" => ToolPolicyBindings {
            domain_fields: vec![DomainPolicyBinding::url("url")],
            result_limit_fields: vec![
                ResultLimitBinding::new("max_chars", ResultLimitKind::MaxOutputChars),
                ResultLimitBinding::new("max_response_bytes", ResultLimitKind::MaxResponseBytes),
                ResultLimitBinding::new("max_redirects", ResultLimitKind::MaxRedirects),
            ],
            ..Default::default()
        },
        "http" => ToolPolicyBindings {
            domain_fields: vec![DomainPolicyBinding::url("url")],
            operation_fields: vec!["method".to_string()],
            ..Default::default()
        },
        "file" => ToolPolicyBindings {
            path_fields: vec![PathPolicyBinding::read_write("path")],
            operation_fields: vec!["operation".to_string()],
            ..Default::default()
        },
        "file_write" | "file_edit" => ToolPolicyBindings {
            path_fields: vec![PathPolicyBinding::write("path")],
            ..Default::default()
        },
        "patch" => ToolPolicyBindings {
            path_fields: vec![ai_agents_core::PathPolicyBinding::new(
                "path",
                PathAccessMode::Write,
                ai_agents_core::PathBindingKind::PatchBase,
            )],
            ..Default::default()
        },
        "command" => ToolPolicyBindings {
            command_fields: vec![
                CommandPolicyBinding::command("command"),
                CommandPolicyBinding::argv("argv"),
            ],
            path_fields: vec![ai_agents_core::PathPolicyBinding::new(
                "cwd",
                PathAccessMode::ReadWrite,
                ai_agents_core::PathBindingKind::Cwd,
            )],
            ..Default::default()
        },
        _ => ToolPolicyBindings::default(),
    }
}

#[derive(Debug, Clone)]
struct BoundPathValue {
    path: String,
    mode: PathAccessMode,
}

#[derive(Debug, Clone)]
struct BoundDomainValue {
    value: String,
    is_url: bool,
}

fn validate_policy_bindings(
    tool_id: &str,
    config: &ToolPolicyConfig,
    bindings: &ToolPolicyBindings,
    fail_closed: bool,
) -> Option<SecurityCheckResult> {
    if !fail_closed {
        return None;
    }
    if path_policy_configured(config) && !bindings.has_path_bindings() {
        return Some(SecurityCheckResult::Block {
            reason: format!(
                "path policy configured for {} but tool exposes no path policy bindings",
                tool_id
            ),
        });
    }
    if domain_policy_configured(config) && !bindings.has_domain_bindings() {
        return Some(SecurityCheckResult::Block {
            reason: format!(
                "domain policy configured for {} but tool exposes no domain policy bindings",
                tool_id
            ),
        });
    }
    if command_policy_configured(config) && !bindings.has_command_bindings() {
        return Some(SecurityCheckResult::Block {
            reason: format!(
                "command policy configured for {} but tool exposes no command policy bindings",
                tool_id
            ),
        });
    }
    if operation_policy_configured(config) && !bindings.has_operation_bindings() {
        return Some(SecurityCheckResult::Block {
            reason: format!(
                "operation policy configured for {} but tool exposes no operation policy bindings",
                tool_id
            ),
        });
    }
    if result_limit_policy_configured(config) && !bindings.has_result_limit_bindings() {
        return Some(SecurityCheckResult::Block {
            reason: format!(
                "result-limit policy configured for {} but tool exposes no result-limit policy bindings",
                tool_id
            ),
        });
    }
    None
}

fn missing_bound_value_result(
    tool_id: &str,
    policy_kind: &str,
    configured: bool,
    fail_closed: bool,
) -> Option<SecurityCheckResult> {
    if configured && fail_closed {
        Some(SecurityCheckResult::Block {
            reason: format!(
                "{} policy configured for {} but no bound {} argument was present",
                policy_kind, tool_id, policy_kind
            ),
        })
    } else {
        None
    }
}

fn path_policy_configured(config: &ToolPolicyConfig) -> bool {
    !config.allowed_paths.is_empty()
        || !config.read_paths.is_empty()
        || !config.write_paths.is_empty()
        || !config.blocked_paths.is_empty()
        || !config.paths.allow.is_empty()
        || !config.paths.deny.is_empty()
        || !config.paths.requires_approval.is_empty()
        || !config.paths.unavailable.is_empty()
}

fn domain_policy_configured(config: &ToolPolicyConfig) -> bool {
    !config.allowed_domains.is_empty()
        || !config.blocked_domains.is_empty()
        || !config.allowed_schemes.is_empty()
        || !config.allowed_ports.is_empty()
        || !config.domains.allow.is_empty()
        || !config.domains.deny.is_empty()
        || !config.domains.requires_approval.is_empty()
        || !config.domains.unavailable.is_empty()
}

fn command_policy_configured(config: &ToolPolicyConfig) -> bool {
    !config.commands.allow.is_empty()
        || !config.commands.deny.is_empty()
        || !config.commands.requires_approval.is_empty()
        || !config.commands.unavailable.is_empty()
}

fn operation_policy_configured(config: &ToolPolicyConfig) -> bool {
    !config.operations.allow.is_empty()
        || !config.operations.deny.is_empty()
        || !config.operations.requires_approval.is_empty()
        || !config.operations.unavailable.is_empty()
}

fn result_limit_policy_configured(config: &ToolPolicyConfig) -> bool {
    // max_output_chars is enforced by the shared executor after execution.
    config.max_file_size_bytes.is_some()
        || config.max_results.is_some()
        || config.max_response_bytes.is_some()
        || config.max_redirects.is_some()
        || config.max_changed_files.is_some()
        || config.max_changed_lines.is_some()
}

fn bound_path_values(args: &Value, bindings: &ToolPolicyBindings) -> Vec<BoundPathValue> {
    let mut values = Vec::new();
    for binding in &bindings.path_fields {
        collect_path_binding_values(args, binding, &mut values);
    }
    values
}

fn collect_path_binding_values(
    args: &Value,
    binding: &PathPolicyBinding,
    values: &mut Vec<BoundPathValue>,
) {
    let value = value_at_path(args, &binding.field).cloned().or_else(|| {
        binding
            .default_path
            .as_ref()
            .map(|path| Value::String(path.clone()))
    });
    let Some(value) = value else {
        return;
    };
    match value {
        Value::String(path) => values.push(BoundPathValue {
            path,
            mode: binding.mode,
        }),
        Value::Array(items) => {
            for item in items {
                if let Some(path) = item.as_str() {
                    values.push(BoundPathValue {
                        path: path.to_string(),
                        mode: binding.mode,
                    });
                }
            }
        }
        _ => {}
    }
}

fn bound_domain_values(args: &Value, bindings: &ToolPolicyBindings) -> Vec<BoundDomainValue> {
    let mut values = Vec::new();
    for binding in &bindings.domain_fields {
        collect_domain_binding_values(args, binding, &mut values);
    }
    values
}

fn collect_domain_binding_values(
    args: &Value,
    binding: &DomainPolicyBinding,
    values: &mut Vec<BoundDomainValue>,
) {
    let Some(value) = value_at_path(args, &binding.field) else {
        return;
    };
    match value {
        Value::String(value) => values.push(BoundDomainValue {
            value: value.clone(),
            is_url: binding.is_url,
        }),
        Value::Array(items) => {
            for item in items {
                if let Some(value) = item.as_str() {
                    values.push(BoundDomainValue {
                        value: value.to_string(),
                        is_url: binding.is_url,
                    });
                }
            }
        }
        _ => {}
    }
}

fn bound_operation_values(args: &Value, bindings: &ToolPolicyBindings) -> Vec<String> {
    bindings
        .operation_fields
        .iter()
        .filter_map(|field| value_at_path(args, field).and_then(Value::as_str))
        .map(|value| value.trim().to_ascii_lowercase())
        .collect()
}

fn bound_command_values(args: &Value, bindings: &ToolPolicyBindings) -> Vec<String> {
    let mut values = Vec::new();
    for binding in &bindings.command_fields {
        collect_command_binding_values(args, binding, &mut values);
    }
    values
}

fn collect_command_binding_values(
    args: &Value,
    binding: &CommandPolicyBinding,
    values: &mut Vec<String>,
) {
    let Some(value) = value_at_path(args, &binding.field) else {
        return;
    };
    match binding.kind {
        CommandBindingKind::CommandString
        | CommandBindingKind::Cwd
        | CommandBindingKind::TemplateVariable => {
            if let Some(command) = value.as_str() {
                values.push(command.to_string());
            }
        }
        CommandBindingKind::Argv => {
            if let Some(command) = value
                .as_array()
                .and_then(|items| items.first())
                .and_then(Value::as_str)
            {
                values.push(command.to_string());
            }
        }
        CommandBindingKind::Env => {}
    }
}

fn allowed_paths_for_mode(config: &ToolPolicyConfig, mode: PathAccessMode) -> Vec<&String> {
    let mut allowed: Vec<&String> = config
        .allowed_paths
        .iter()
        .chain(config.paths.allow.iter())
        .collect();
    match mode {
        PathAccessMode::Read => allowed.extend(config.read_paths.iter()),
        PathAccessMode::Write => allowed.extend(config.write_paths.iter()),
        PathAccessMode::ReadWrite => {
            allowed.extend(config.read_paths.iter());
            allowed.extend(config.write_paths.iter());
        }
    }
    allowed
}

fn value_at_path<'a>(value: &'a Value, field: &str) -> Option<&'a Value> {
    let mut current = value;
    for segment in field.split('.') {
        if segment.is_empty() {
            return None;
        }
        current = current.get(segment)?;
    }
    Some(current)
}

fn set_root_value(args: &mut Value, field: &str, value: Value) {
    if field.contains('.') {
        return;
    }
    let Some(obj) = args.as_object_mut() else {
        return;
    };
    obj.insert(field.to_string(), value);
}

fn apply_usize_cap(obj: &mut serde_json::Map<String, Value>, key: &str, cap: Option<usize>) {
    let Some(cap) = cap else {
        return;
    };
    let effective = obj
        .get(key)
        .and_then(|value| value.as_u64())
        .map(|value| value.min(cap as u64) as usize)
        .unwrap_or(cap);
    obj.insert(key.to_string(), Value::from(effective));
}

fn apply_u64_cap(obj: &mut serde_json::Map<String, Value>, key: &str, cap: Option<u64>) {
    let Some(cap) = cap else {
        return;
    };
    let effective = obj
        .get(key)
        .and_then(|value| value.as_u64())
        .map(|value| value.min(cap))
        .unwrap_or(cap);
    obj.insert(key.to_string(), Value::from(effective));
}

fn min_optional_usize(left: Option<usize>, right: Option<usize>) -> Option<usize> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn normalize_host(host: &str) -> String {
    host.trim_end_matches('.').to_ascii_lowercase()
}

fn host_matches(pattern: &str, host: &str) -> bool {
    let pattern = normalize_host(pattern.trim_start_matches("*."));
    host == pattern || host.ends_with(&format!(".{}", pattern))
}

fn normalize_path(path: &str) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in Path::new(path).components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

fn path_matches(pattern: &str, path: &Path) -> bool {
    let pattern = normalize_path(pattern);
    path.starts_with(pattern)
}

fn contains_casefold(values: &[String], needle: &str) -> bool {
    values
        .iter()
        .any(|value| value.eq_ignore_ascii_case(needle))
}

fn host_is_private_or_local(host: &str) -> bool {
    if matches!(
        host,
        "localhost"
            | "metadata"
            | "metadata.google.internal"
            | "169.254.169.254"
            | "100.100.100.200"
    ) || host.ends_with(".localhost")
    {
        return true;
    }
    match host.parse::<IpAddr>() {
        Ok(IpAddr::V4(ip)) => {
            ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_multicast()
                || ip.is_documentation()
                || ip.octets() == [169, 254, 169, 254]
        }
        Ok(IpAddr::V6(ip)) => {
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_multicast()
                || ip.segments()[0] & 0xfe00 == 0xfc00
                || ip.segments()[0] & 0xffc0 == 0xfe80
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_engine() {
        let engine = ToolSecurityEngine::default();
        assert!(!engine.config().enabled);
    }

    #[tokio::test]
    async fn test_tool_domain_blocking() {
        let mut config = ToolSecurityConfig::default();
        config.enabled = true;

        let mut http_config = ToolPolicyConfig::default();
        http_config.blocked_domains = vec!["evil.com".to_string()];
        config.tools.insert("http".to_string(), http_config);

        let engine = ToolSecurityEngine::new(config);

        let args = serde_json::json!({"url": "https://evil.com/api"});
        let result = engine.check_tool_execution("http", &args).await.unwrap();
        assert!(result.is_blocked());

        let args = serde_json::json!({"url": "https://not-evil.com/api"});
        let result = engine.check_tool_execution("http", &args).await.unwrap();
        assert!(result.is_allowed());
    }

    #[tokio::test]
    async fn test_tool_allowed_domains() {
        let mut config = ToolSecurityConfig::default();
        config.enabled = true;

        let mut http_config = ToolPolicyConfig::default();
        http_config.allowed_domains = vec!["api.example.com".to_string()];
        config.tools.insert("http".to_string(), http_config);

        let engine = ToolSecurityEngine::new(config);

        let args = serde_json::json!({"url": "https://api.example.com/v1"});
        let result = engine.check_tool_execution("http", &args).await.unwrap();
        assert!(result.is_allowed());

        let args = serde_json::json!({"url": "https://other.com/api"});
        let result = engine.check_tool_execution("http", &args).await.unwrap();
        assert!(result.is_blocked());
    }

    #[tokio::test]
    async fn test_tool_disabled() {
        let mut config = ToolSecurityConfig::default();
        config.enabled = true;

        let mut tool_config = ToolPolicyConfig::default();
        tool_config.enabled = false;
        config.tools.insert("dangerous".to_string(), tool_config);

        let engine = ToolSecurityEngine::new(config);

        let result = engine
            .check_tool_execution("dangerous", &serde_json::json!({}))
            .await
            .unwrap();
        assert!(result.is_blocked());
        assert!(result.is_unavailable());
    }

    #[tokio::test]
    async fn test_tool_confirmation_required() {
        let mut config = ToolSecurityConfig::default();
        config.enabled = true;

        let mut tool_config = ToolPolicyConfig::default();
        tool_config.require_confirmation = true;
        tool_config.confirmation_message = Some("Are you sure?".to_string());
        config.tools.insert("delete".to_string(), tool_config);

        let engine = ToolSecurityEngine::new(config);

        let result = engine
            .check_tool_execution("delete", &serde_json::json!({}))
            .await
            .unwrap();

        match result {
            SecurityCheckResult::RequireConfirmation { message } => {
                assert_eq!(message, "Are you sure?");
            }
            _ => panic!("Expected RequireConfirmation"),
        }
    }

    #[test]
    fn test_get_tool_timeout() {
        let mut config = ToolSecurityConfig::default();
        config.default_timeout_ms = 5000;

        let mut tool_config = ToolPolicyConfig::default();
        tool_config.timeout_ms = Some(10000);
        config.tools.insert("slow".to_string(), tool_config);

        let engine = ToolSecurityEngine::new(config);

        assert_eq!(engine.get_tool_timeout("slow"), 10000);
        assert_eq!(engine.get_tool_timeout("other"), 5000);
    }

    #[tokio::test]
    async fn test_path_restrictions() {
        let mut config = ToolSecurityConfig::default();
        config.enabled = true;

        let mut tool_config = ToolPolicyConfig::default();
        tool_config.allowed_paths = vec!["/tmp/".to_string(), "/home/user/".to_string()];
        config.tools.insert("file_write".to_string(), tool_config);

        let engine = ToolSecurityEngine::new(config);

        let args = serde_json::json!({"path": "/tmp/test.txt"});
        let result = engine
            .check_tool_execution("file_write", &args)
            .await
            .unwrap();
        assert!(result.is_allowed());

        let args = serde_json::json!({"path": "/etc/passwd"});
        let result = engine
            .check_tool_execution("file_write", &args)
            .await
            .unwrap();
        assert!(result.is_blocked());
    }

    #[tokio::test]
    async fn test_operation_policy() {
        let mut config = ToolSecurityConfig::default();
        config.enabled = true;
        let mut tool_config = ToolPolicyConfig::default();
        tool_config.operations.deny = vec!["delete".to_string()];
        tool_config.operations.requires_approval = vec!["write".to_string()];
        config.tools.insert("file".to_string(), tool_config);
        let engine = ToolSecurityEngine::new(config);

        let result = engine
            .check_tool_execution("file", &serde_json::json!({"operation": "delete"}))
            .await
            .unwrap();
        assert!(result.is_blocked());

        let result = engine
            .check_tool_execution("file", &serde_json::json!({"operation": "write"}))
            .await
            .unwrap();
        assert!(result.requires_approval());
    }

    #[tokio::test]
    async fn omitted_optional_path_uses_default_for_policy() {
        let mut config = ToolSecurityConfig::default();
        config.enabled = true;
        let mut tool_config = ToolPolicyConfig::default();
        tool_config.read_paths = vec!["./crates".to_string()];
        config.tools.insert("grep".to_string(), tool_config);
        let engine = ToolSecurityEngine::new(config);

        let result = engine
            .check_tool_execution("grep", &serde_json::json!({"pattern": "Tool"}))
            .await
            .unwrap();
        assert!(result.is_blocked());

        let prepared =
            engine.prepare_tool_arguments("grep", &serde_json::json!({"pattern": "Tool"}));
        assert_eq!(prepared.get("path").and_then(Value::as_str), Some("."));
    }

    #[tokio::test]
    async fn fail_closed_requires_path_bindings_for_custom_tools() {
        let mut config = ToolSecurityConfig::default();
        config.enabled = true;
        config.fail_closed = true;
        let mut tool_config = ToolPolicyConfig::default();
        tool_config.read_paths = vec!["./allowed".to_string()];
        config
            .tools
            .insert("custom_search".to_string(), tool_config);
        let engine = ToolSecurityEngine::new(config);

        let result = engine
            .check_tool_execution_with_bindings(
                "custom_search",
                &serde_json::json!({"path": "./allowed/file.txt"}),
                &ToolPolicyBindings::default(),
            )
            .await
            .unwrap();

        assert!(result.is_blocked());
        assert!(
            result
                .reason()
                .unwrap_or_default()
                .contains("tool exposes no path policy bindings")
        );
    }

    #[tokio::test]
    async fn custom_path_bindings_enforce_blocked_paths() {
        let mut config = ToolSecurityConfig::default();
        config.enabled = true;
        config.fail_closed = true;
        let mut tool_config = ToolPolicyConfig::default();
        tool_config.read_paths = vec!["./allowed".to_string()];
        tool_config.blocked_paths = vec!["./allowed/private".to_string()];
        config
            .tools
            .insert("custom_search".to_string(), tool_config);
        let engine = ToolSecurityEngine::new(config);
        let bindings = ToolPolicyBindings {
            path_fields: vec![PathPolicyBinding::read("root")],
            ..Default::default()
        };

        let allowed = engine
            .check_tool_execution_with_bindings(
                "custom_search",
                &serde_json::json!({"root": "./allowed/src"}),
                &bindings,
            )
            .await
            .unwrap();
        assert!(allowed.is_allowed());

        let blocked = engine
            .check_tool_execution_with_bindings(
                "custom_search",
                &serde_json::json!({"root": "./allowed/private/secrets.txt"}),
                &bindings,
            )
            .await
            .unwrap();
        assert!(blocked.is_blocked());
    }

    #[test]
    fn custom_config_is_exposed_separately() {
        let mut config = ToolSecurityConfig::default();
        config.enabled = true;
        let mut tool_config = ToolPolicyConfig::default();
        tool_config
            .config
            .insert("backend".to_string(), serde_json::json!("tantivy"));
        config.tools.insert("my_search".to_string(), tool_config);
        let engine = ToolSecurityEngine::new(config);

        assert_eq!(engine.custom_config("my_search")["backend"], "tantivy");
    }

    #[test]
    fn policy_caps_are_applied_as_upper_bounds() {
        let mut config = ToolSecurityConfig::default();
        config.enabled = true;
        let mut tool_config = ToolPolicyConfig::default();
        tool_config.max_results = Some(5);
        tool_config.max_file_size_bytes = Some(1024);
        tool_config.max_output_chars = Some(1000);
        config.tools.insert("grep".to_string(), tool_config);
        let engine = ToolSecurityEngine::new(config);

        let prepared = engine.prepare_tool_arguments(
            "grep",
            &serde_json::json!({
                "pattern": "Tool",
                "path": ".",
                "max_results": 50,
                "max_file_size_bytes": 8192,
                "max_output_chars": 20000
            }),
        );
        assert_eq!(prepared.get("max_results").and_then(Value::as_u64), Some(5));
        assert_eq!(
            prepared.get("max_file_size_bytes").and_then(Value::as_u64),
            Some(1024)
        );
        assert_eq!(
            prepared.get("max_output_chars").and_then(Value::as_u64),
            Some(1000)
        );
    }

    #[tokio::test]
    async fn fail_closed_blocks_missing_result_limit_bindings() {
        let mut config = ToolSecurityConfig::default();
        config.enabled = true;
        config.fail_closed = true;
        let mut tool_config = ToolPolicyConfig::default();
        tool_config.max_results = Some(5);
        config
            .tools
            .insert("custom_search".to_string(), tool_config);
        let engine = ToolSecurityEngine::new(config);

        let result = engine
            .check_tool_execution_with_bindings(
                "custom_search",
                &serde_json::json!({"query": "rust"}),
                &ToolPolicyBindings::default(),
            )
            .await
            .unwrap();

        assert!(result.is_blocked());
        assert!(result.reason().unwrap().contains("result-limit policy"));
    }

    #[tokio::test]
    async fn fail_closed_allows_configured_result_limit_bindings() {
        let mut config = ToolSecurityConfig::default();
        config.enabled = true;
        config.fail_closed = true;
        let mut tool_config = ToolPolicyConfig::default();
        tool_config.max_results = Some(5);
        config
            .tools
            .insert("custom_search".to_string(), tool_config);
        let engine = ToolSecurityEngine::new(config);
        let bindings = ToolPolicyBindings {
            result_limit_fields: vec![ResultLimitBinding::new(
                "limit",
                ResultLimitKind::MaxResults,
            )],
            ..Default::default()
        };

        let result = engine
            .check_tool_execution_with_bindings(
                "custom_search",
                &serde_json::json!({"query": "rust", "limit": 10}),
                &bindings,
            )
            .await
            .unwrap();

        assert!(result.is_allowed());
    }
}

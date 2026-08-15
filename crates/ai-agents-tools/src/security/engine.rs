use std::collections::HashMap;
use std::net::IpAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use tracing::debug;

use super::config::*;
use super::path::PathPolicyResolver;
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
    fn admit(&mut self, tool_id: &str, rate_limit: Option<u32>) -> bool {
        let Some(rate_limit) = rate_limit else {
            return true;
        };
        let now = Instant::now();
        let window = std::time::Duration::from_secs(60);
        let calls = self.calls.entry(tool_id.to_string()).or_default();
        calls.retain(|timestamp| now.duration_since(*timestamp) < window);
        if calls.len() >= rate_limit as usize {
            return false;
        }
        calls.push(now);
        true
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
    /// Creates a validated security engine and panics when host policy is invalid.
    pub fn new(config: ToolSecurityConfig) -> Self {
        Self::try_new(config).expect("invalid tool security configuration")
    }

    /// Creates a validated security engine with a recoverable configuration error.
    pub fn try_new(config: ToolSecurityConfig) -> Result<Self> {
        Self::try_new_with_policy_version(config, 1)
    }

    /// Creates a versioned validated engine and panics when host policy is invalid.
    pub fn new_with_policy_version(config: ToolSecurityConfig, policy_version: u64) -> Self {
        Self::try_new_with_policy_version(config, policy_version)
            .expect("invalid tool security configuration")
    }

    /// Creates a versioned validated engine without admitting invalid policy state.
    pub fn try_new_with_policy_version(
        config: ToolSecurityConfig,
        policy_version: u64,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            tool_call_tracker: Arc::new(RwLock::new(ToolCallTracker::default())),
            policy_version,
        })
    }

    /// Revalidates the immutable configuration owned by this engine.
    pub fn validate(&self) -> Result<()> {
        self.config.validate()
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
        let policy_timeout_ms = self.get_tool_timeout(tool_id);
        ToolExecutionLimits {
            timeout_ms: Some(
                classification
                    .timeout_ms
                    .map_or(policy_timeout_ms, |timeout_ms| {
                        timeout_ms.min(policy_timeout_ms)
                    }),
            ),
            max_output_chars: min_optional_usize(
                classification.max_output_chars,
                policy.and_then(|config| config.max_output_chars),
            ),
            max_result_chars: safety.max_result_size_chars,
            max_results: policy.and_then(|config| config.max_results),
            max_file_size_bytes: policy.and_then(|config| config.max_file_size_bytes),
            max_response_bytes: policy.and_then(|config| config.max_response_bytes),
            max_redirects: policy.and_then(|config| config.max_redirects),
            max_replacements: policy.and_then(|config| config.max_replacements),
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

    /// Returns the approval message required by call classification defaults.
    pub fn classification_approval_message(
        &self,
        tool_id: &str,
        classification: &ToolCallClassification,
    ) -> Option<String> {
        if !classification.requires_approval || classification.read_only {
            return None;
        }
        if !matches!(
            classification.operation,
            ai_agents_core::ToolOperationKind::Write
                | ai_agents_core::ToolOperationKind::Edit
                | ai_agents_core::ToolOperationKind::Delete
                | ai_agents_core::ToolOperationKind::Patch
                | ai_agents_core::ToolOperationKind::Command
        ) {
            return None;
        }
        let tool_config = self
            .config
            .enabled
            .then(|| self.config.tools.get(tool_id))
            .flatten();
        if tool_config.is_some_and(|config| config.allow_without_confirmation) {
            return None;
        }
        Some(format!(
            "Confirm {} operation for tool '{}' ?",
            format!("{:?}", classification.operation).to_ascii_lowercase(),
            tool_id
        ))
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
        let validation = self
            .validate_tool_execution_with_bindings(tool_id, args, bindings)
            .await?;
        if validation.is_allowed() {
            let admission = self.admit_tool_execution(tool_id);
            if !admission.is_allowed() {
                return Ok(admission);
            }
        }
        Ok(validation)
    }

    /// Validates policy without consuming rate-limit admission.
    pub async fn validate_tool_execution_with_bindings(
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
                debug!(tool_id = %tool_id, "Tool execution allowed by legacy open policy");
                return Ok(SecurityCheckResult::Allow);
            }
        };

        if !tool_config.enabled {
            return Ok(SecurityCheckResult::Unavailable {
                reason: format!("Tool '{}' is disabled", tool_id),
            });
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

        debug!(tool_id = %tool_id, "Tool execution allowed by policy validation");
        Ok(SecurityCheckResult::Allow)
    }

    pub fn admit_tool_execution(&self, tool_id: &str) -> SecurityCheckResult {
        if !self.config.enabled {
            return SecurityCheckResult::Allow;
        }
        let tool_config = match self.config.tools.get(tool_id) {
            Some(config) => config,
            None if self.config.fail_closed => {
                return SecurityCheckResult::Block {
                    reason: format!("Tool '{}' has no explicit security policy", tool_id),
                };
            }
            None => {
                self.tool_call_tracker.write().admit(tool_id, None);
                return SecurityCheckResult::Allow;
            }
        };
        if !tool_config.enabled {
            return SecurityCheckResult::Unavailable {
                reason: format!("Tool '{}' is disabled", tool_id),
            };
        }
        let mut tracker = self.tool_call_tracker.write();
        if !tracker.admit(tool_id, tool_config.rate_limit) {
            let rate_limit = tool_config.rate_limit.unwrap_or_default();
            return SecurityCheckResult::Block {
                reason: format!(
                    "Rate limit exceeded for tool '{}': {} calls per minute",
                    tool_id, rate_limit
                ),
            };
        }
        debug!(tool_id = %tool_id, "Tool execution admitted");
        SecurityCheckResult::Allow
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
        let resolver = match PathPolicyResolver::new() {
            Ok(resolver) => resolver,
            Err(error) => return Some(path_resolution_block(tool_id, error)),
        };
        for value in values {
            if let Err(error) = resolver.resolve_path(Path::new(&value.path)) {
                return Some(path_resolution_block(tool_id, error));
            }

            for pattern in tool_config
                .blocked_paths
                .iter()
                .chain(tool_config.paths.deny.iter())
            {
                match resolver.matches_restriction(Path::new(&value.path), Path::new(pattern)) {
                    Ok(true) => {
                        return Some(SecurityCheckResult::Block {
                            reason: format!("Path is blocked for tool '{}'", tool_id),
                        });
                    }
                    Ok(false) => {}
                    Err(error) => return Some(path_resolution_block(tool_id, error)),
                }
            }

            for pattern in tool_config.paths.unavailable.iter() {
                match resolver.matches_restriction(Path::new(&value.path), Path::new(pattern)) {
                    Ok(true) => {
                        return Some(SecurityCheckResult::Unavailable {
                            reason: format!("Path is unavailable for tool '{}'", tool_id),
                        });
                    }
                    Ok(false) => {}
                    Err(error) => return Some(path_resolution_block(tool_id, error)),
                }
            }

            for pattern in tool_config.paths.requires_approval.iter() {
                match resolver.matches_restriction(Path::new(&value.path), Path::new(pattern)) {
                    Ok(true) => {
                        return Some(SecurityCheckResult::RequireConfirmation {
                            message: format!(
                                "Confirm access to path '{}' for tool '{}' ?",
                                value.path, tool_id
                            ),
                        });
                    }
                    Ok(false) => {}
                    Err(error) => return Some(path_resolution_block(tool_id, error)),
                }
            }

            if !matches!(value.kind, ai_agents_core::PathBindingKind::Cwd)
                && matches!(
                    value.mode,
                    PathAccessMode::Write | PathAccessMode::ReadWrite
                )
                && !has_write_allowlist(tool_config)
            {
                let dry_run = args
                    .get("dry_run")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if matches!(tool_config.no_write_policy, NoWritePolicyBehavior::Deny) || !dry_run {
                    return Some(SecurityCheckResult::Block {
                        reason: format!(
                            "Tool '{}' cannot mutate paths without an explicit write_paths policy",
                            tool_id
                        ),
                    });
                }
            }

            let allowed = allowed_paths_for_value(tool_config, &value);
            if matches!(value.kind, ai_agents_core::PathBindingKind::Cwd) && allowed.is_empty() {
                return Some(SecurityCheckResult::Block {
                    reason: format!(
                        "Tool '{}' requires an explicit working_dirs policy for command cwd",
                        tool_id
                    ),
                });
            }
            if !allowed.is_empty() {
                let mut matches_allowed = false;
                for pattern in allowed {
                    match resolver.is_allowed(Path::new(&value.path), Path::new(pattern)) {
                        Ok(true) => {
                            matches_allowed = true;
                            break;
                        }
                        Ok(false) => {}
                        Err(error) => return Some(path_resolution_block(tool_id, error)),
                    }
                }
                if !matches_allowed {
                    return Some(SecurityCheckResult::Block {
                        reason: format!("Path not in allowed list for tool '{}'", tool_id),
                    });
                }
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
            let display = command.display();
            let command_name = command.command_name();

            if command.is_string
                && command_denies_shell(tool_config)
                && contains_shell_syntax(&display)
            {
                return Some(SecurityCheckResult::Block {
                    reason: format!(
                        "Command '{}' uses shell syntax denied for tool '{}'",
                        display, tool_id
                    ),
                });
            }
            if contains_casefold(&tool_config.commands.deny, &display)
                || contains_casefold(&tool_config.commands.deny, &command_name)
            {
                return Some(SecurityCheckResult::Block {
                    reason: format!("Command '{}' is blocked for tool '{}'", display, tool_id),
                });
            }
            if contains_casefold(&tool_config.commands.unavailable, &display)
                || contains_casefold(&tool_config.commands.unavailable, &command_name)
            {
                return Some(SecurityCheckResult::Unavailable {
                    reason: format!(
                        "Command '{}' is unavailable for tool '{}'",
                        display, tool_id
                    ),
                });
            }
            if contains_casefold(&tool_config.commands.requires_approval, &display)
                || contains_casefold(&tool_config.commands.requires_approval, &command_name)
            {
                return Some(SecurityCheckResult::RequireConfirmation {
                    message: format!("Confirm command '{}' for tool '{}' ?", display, tool_id),
                });
            }
            let has_exact_allowlist = command_has_exact_allowlist(tool_config);
            if command_requires_exact_allowlist(tool_id) && !has_exact_allowlist {
                return Some(SecurityCheckResult::Block {
                    reason: format!(
                        "Tool '{}' requires allowed_commands or command_templates before execution",
                        tool_id
                    ),
                });
            }
            if has_exact_allowlist {
                if !command_matches_allowed(tool_config, &command.argv) {
                    if command_allows_escalation(tool_config) {
                        return Some(SecurityCheckResult::RequireConfirmation {
                            message: format!(
                                "Confirm command '{}' outside the exact allowlist for tool '{}' ?",
                                display, tool_id
                            ),
                        });
                    }
                    return Some(SecurityCheckResult::Block {
                        reason: format!(
                            "Command '{}' is not in the exact argv allowlist for tool '{}'",
                            display, tool_id
                        ),
                    });
                }
                continue;
            }
            if !tool_config.commands.allow.is_empty()
                && !contains_casefold(&tool_config.commands.allow, &display)
                && !contains_casefold(&tool_config.commands.allow, &command_name)
            {
                return Some(SecurityCheckResult::Block {
                    reason: format!(
                        "Command '{}' is not allowed for tool '{}'",
                        display, tool_id
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
            ResultLimitKind::MaxReplacements => {
                apply_usize_cap(obj, &binding.field, config.max_replacements);
            }
            ResultLimitKind::MaxChangedFiles => {
                apply_usize_cap(obj, &binding.field, config.max_changed_files);
            }
            ResultLimitKind::MaxChangedLines => {
                apply_usize_cap(obj, &binding.field, config.max_changed_lines);
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
        "file_write" => ToolPolicyBindings {
            path_fields: vec![PathPolicyBinding::write("path")],
            result_limit_fields: vec![
                ResultLimitBinding::new("max_changed_files", ResultLimitKind::MaxChangedFiles),
                ResultLimitBinding::new("max_changed_lines", ResultLimitKind::MaxChangedLines),
            ],
            ..Default::default()
        },
        "file_edit" => ToolPolicyBindings {
            path_fields: vec![PathPolicyBinding::write("path")],
            result_limit_fields: vec![
                ResultLimitBinding::new("max_replacements", ResultLimitKind::MaxReplacements),
                ResultLimitBinding::new("max_changed_lines", ResultLimitKind::MaxChangedLines),
            ],
            ..Default::default()
        },
        "patch" => ToolPolicyBindings {
            path_fields: vec![
                ai_agents_core::PathPolicyBinding::new(
                    "base_path",
                    PathAccessMode::Write,
                    ai_agents_core::PathBindingKind::PatchBase,
                )
                .with_default_path("."),
            ],
            result_limit_fields: vec![
                ResultLimitBinding::new("max_changed_files", ResultLimitKind::MaxChangedFiles),
                ResultLimitBinding::new("max_changed_lines", ResultLimitKind::MaxChangedLines),
            ],
            ..Default::default()
        },
        "copy_path" => ToolPolicyBindings {
            path_fields: vec![
                PathPolicyBinding::read("source_path"),
                PathPolicyBinding::write("destination_path"),
            ],
            ..Default::default()
        },
        "move_path" => ToolPolicyBindings {
            path_fields: vec![
                PathPolicyBinding::read_write("source_path"),
                PathPolicyBinding::write("destination_path"),
            ],
            ..Default::default()
        },
        "delete_path" => ToolPolicyBindings {
            path_fields: vec![PathPolicyBinding::write("path")],
            ..Default::default()
        },
        "command" => ToolPolicyBindings {
            command_fields: vec![
                CommandPolicyBinding::command("command"),
                CommandPolicyBinding::argv("argv"),
                CommandPolicyBinding::env("env"),
            ],
            path_fields: vec![
                ai_agents_core::PathPolicyBinding::new(
                    "cwd",
                    PathAccessMode::ReadWrite,
                    ai_agents_core::PathBindingKind::Cwd,
                )
                .with_default_path("."),
            ],
            result_limit_fields: vec![ResultLimitBinding::new(
                "max_output_chars",
                ResultLimitKind::MaxOutputChars,
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
    kind: ai_agents_core::PathBindingKind,
}

#[derive(Debug, Clone)]
struct BoundDomainValue {
    value: String,
    is_url: bool,
}

#[derive(Debug, Clone)]
struct BoundCommandValue {
    argv: Vec<String>,
    is_string: bool,
}

impl BoundCommandValue {
    fn display(&self) -> String {
        self.argv.join(" ")
    }

    fn command_name(&self) -> String {
        self.argv.first().cloned().unwrap_or_default()
    }
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
        || !config.working_dirs.is_empty()
        || !config.commands.working_dirs.is_empty()
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
        || !config.commands.allowed_commands.is_empty()
        || !config.commands.templates.is_empty()
        || !config.allowed_commands.is_empty()
        || !config.command_templates.is_empty()
        || !config.env_passthrough.is_empty()
        || !config.commands.env_passthrough.is_empty()
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
        || config.max_replacements.is_some()
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
            mode: effective_path_mode(args, binding),
            kind: binding.kind,
        }),
        Value::Array(items) => {
            for item in items {
                if let Some(path) = item.as_str() {
                    values.push(BoundPathValue {
                        path: path.to_string(),
                        mode: effective_path_mode(args, binding),
                        kind: binding.kind,
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

fn bound_command_values(args: &Value, bindings: &ToolPolicyBindings) -> Vec<BoundCommandValue> {
    let mut values = Vec::new();
    for binding in &bindings.command_fields {
        collect_command_binding_values(args, binding, &mut values);
    }
    values
}

fn collect_command_binding_values(
    args: &Value,
    binding: &CommandPolicyBinding,
    values: &mut Vec<BoundCommandValue>,
) {
    let Some(value) = value_at_path(args, &binding.field) else {
        return;
    };
    match binding.kind {
        CommandBindingKind::CommandString => {
            if let Some(command) = value.as_str() {
                if let Some(argv) = parse_command_words(command) {
                    values.push(BoundCommandValue {
                        argv,
                        is_string: true,
                    });
                } else {
                    values.push(BoundCommandValue {
                        argv: vec![command.to_string()],
                        is_string: true,
                    });
                }
            }
        }
        CommandBindingKind::Argv => {
            if let Some(argv) = value.as_array().map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            }) && !argv.is_empty()
            {
                values.push(BoundCommandValue {
                    argv,
                    is_string: false,
                });
            }
        }
        CommandBindingKind::Cwd
        | CommandBindingKind::TemplateVariable
        | CommandBindingKind::Env => {}
    }
}

fn allowed_paths_for_value<'a>(
    config: &'a ToolPolicyConfig,
    value: &BoundPathValue,
) -> Vec<&'a String> {
    if matches!(value.kind, ai_agents_core::PathBindingKind::Cwd) {
        return config
            .working_dirs
            .iter()
            .chain(config.commands.working_dirs.iter())
            .collect();
    }
    let mut allowed: Vec<&String> = config
        .allowed_paths
        .iter()
        .chain(config.paths.allow.iter())
        .collect();
    match value.mode {
        PathAccessMode::Read => allowed.extend(config.read_paths.iter()),
        PathAccessMode::Write | PathAccessMode::ReadWrite => {
            allowed.extend(config.write_paths.iter());
        }
    }
    allowed
}

fn has_write_allowlist(config: &ToolPolicyConfig) -> bool {
    !config.write_paths.is_empty()
        || !config.allowed_paths.is_empty()
        || !config.paths.allow.is_empty()
}

fn effective_path_mode(args: &Value, binding: &PathPolicyBinding) -> PathAccessMode {
    if !matches!(binding.mode, PathAccessMode::ReadWrite) {
        return binding.mode;
    }
    let operation = args
        .get("operation")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    match operation.as_str() {
        "read" | "exists" | "list" | "info" => PathAccessMode::Read,
        "write" | "append" | "mkdir" | "delete" | "edit" | "patch" => PathAccessMode::Write,
        _ => binding.mode,
    }
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

fn path_resolution_block(
    tool_id: &str,
    error: super::path::PathResolutionError,
) -> SecurityCheckResult {
    SecurityCheckResult::Block {
        reason: format!(
            "Path policy resolution failed for tool '{}': {}",
            tool_id, error
        ),
    }
}

fn contains_casefold(values: &[String], needle: &str) -> bool {
    values
        .iter()
        .any(|value| value.eq_ignore_ascii_case(needle))
}

fn command_denies_shell(config: &ToolPolicyConfig) -> bool {
    config.deny_shell || config.commands.deny_shell
}

fn command_allows_escalation(config: &ToolPolicyConfig) -> bool {
    config.allow_command_escalation || config.commands.allow_escalation
}

fn command_requires_exact_allowlist(tool_id: &str) -> bool {
    tool_id == "command"
}

fn command_has_exact_allowlist(config: &ToolPolicyConfig) -> bool {
    !config.allowed_commands.is_empty()
        || !config.commands.allowed_commands.is_empty()
        || !config.command_templates.is_empty()
        || !config.commands.templates.is_empty()
}

fn command_matches_allowed(config: &ToolPolicyConfig, argv: &[String]) -> bool {
    config
        .allowed_commands
        .iter()
        .chain(config.commands.allowed_commands.iter())
        .any(|rule| rule.argv == argv)
        || config
            .command_templates
            .iter()
            .chain(config.commands.templates.iter())
            .any(|template| command_matches_template(&template.argv, argv))
}

fn command_matches_template(template: &[String], argv: &[String]) -> bool {
    template.len() == argv.len()
        && template.iter().zip(argv.iter()).all(|(expected, actual)| {
            (expected.starts_with('{') && expected.ends_with('}')) || expected == actual
        })
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

    fn enabled_security_config() -> ToolSecurityConfig {
        ToolSecurityConfig {
            enabled: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_default_engine() {
        let engine = ToolSecurityEngine::default();
        assert!(!engine.config().enabled);
    }

    #[tokio::test]
    async fn test_tool_domain_blocking() {
        let mut config = enabled_security_config();

        let http_config = ToolPolicyConfig {
            blocked_domains: vec!["evil.com".to_string()],
            ..Default::default()
        };
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
        let mut config = enabled_security_config();

        let http_config = ToolPolicyConfig {
            allowed_domains: vec!["api.example.com".to_string()],
            ..Default::default()
        };
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
        let mut config = enabled_security_config();

        let tool_config = ToolPolicyConfig {
            enabled: false,
            ..Default::default()
        };
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
        let mut config = enabled_security_config();

        let tool_config = ToolPolicyConfig {
            require_confirmation: true,
            confirmation_message: Some("Are you sure?".to_string()),
            ..Default::default()
        };
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
        let mut config = ToolSecurityConfig {
            default_timeout_ms: 5000,
            ..Default::default()
        };

        let tool_config = ToolPolicyConfig {
            timeout_ms: Some(10000),
            ..Default::default()
        };
        config.tools.insert("slow".to_string(), tool_config);

        let engine = ToolSecurityEngine::new(config);

        assert_eq!(engine.get_tool_timeout("slow"), 10000);
        assert_eq!(engine.get_tool_timeout("other"), 5000);
    }

    #[test]
    fn call_classification_timeout_only_lowers_policy_timeout() {
        let engine = ToolSecurityEngine::new(ToolSecurityConfig {
            default_timeout_ms: 5_000,
            ..Default::default()
        });
        let safety = ToolSafetyMetadata::compute();
        let mut classification = ToolCallClassification::from_metadata(&safety);

        classification.timeout_ms = Some(1_000);
        assert_eq!(
            engine
                .effective_limits("custom", &safety, &classification)
                .timeout_ms,
            Some(1_000)
        );

        classification.timeout_ms = Some(10_000);
        assert_eq!(
            engine
                .effective_limits("custom", &safety, &classification)
                .timeout_ms,
            Some(5_000)
        );
    }

    #[tokio::test]
    async fn test_path_restrictions() {
        let directory = tempfile::tempdir().unwrap();
        let allowed_root = directory.path().join("allowed");
        std::fs::create_dir(&allowed_root).unwrap();
        let allowed_path = allowed_root.join("test.txt");
        let denied_path = directory.path().join("denied/test.txt");
        let mut config = enabled_security_config();

        let tool_config = ToolPolicyConfig {
            allowed_paths: vec![allowed_root.to_string_lossy().into_owned()],
            ..Default::default()
        };
        config.tools.insert("file_write".to_string(), tool_config);

        let engine = ToolSecurityEngine::new(config);

        let args = serde_json::json!({"path": allowed_path});
        let result = engine
            .check_tool_execution("file_write", &args)
            .await
            .unwrap();
        assert!(result.is_allowed(), "{result:?}");

        let args = serde_json::json!({"path": denied_path});
        let result = engine
            .check_tool_execution("file_write", &args)
            .await
            .unwrap();
        assert!(result.is_blocked(), "{result:?}");
    }

    #[tokio::test]
    async fn test_operation_policy() {
        let mut config = enabled_security_config();
        let tool_config = ToolPolicyConfig {
            operations: OperationPolicyConfig {
                deny: vec!["delete".to_string()],
                requires_approval: vec!["write".to_string()],
                ..Default::default()
            },
            ..Default::default()
        };
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
        let mut config = enabled_security_config();
        let tool_config = ToolPolicyConfig {
            read_paths: vec!["./crates".to_string()],
            ..Default::default()
        };
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
        let mut config = ToolSecurityConfig {
            fail_closed: true,
            ..enabled_security_config()
        };
        let tool_config = ToolPolicyConfig {
            read_paths: vec!["./allowed".to_string()],
            ..Default::default()
        };
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
        let mut config = ToolSecurityConfig {
            fail_closed: true,
            ..enabled_security_config()
        };
        let tool_config = ToolPolicyConfig {
            read_paths: vec!["./allowed".to_string()],
            blocked_paths: vec!["./allowed/private".to_string()],
            ..Default::default()
        };
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

    #[cfg(unix)]
    #[tokio::test]
    async fn blocked_path_cannot_be_reached_through_symlink_alias() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let private = root.path().join("private");
        let public = root.path().join("public");
        std::fs::create_dir_all(&private).unwrap();
        std::fs::create_dir_all(&public).unwrap();
        symlink(&private, public.join("alias")).unwrap();

        let mut config = ToolSecurityConfig {
            fail_closed: true,
            ..enabled_security_config()
        };
        let tool_config = ToolPolicyConfig {
            read_paths: vec![root.path().to_string_lossy().into_owned()],
            blocked_paths: vec![private.to_string_lossy().into_owned()],
            ..Default::default()
        };
        config
            .tools
            .insert("custom_search".to_string(), tool_config);
        let engine = ToolSecurityEngine::new(config);
        let bindings = ToolPolicyBindings {
            path_fields: vec![PathPolicyBinding::read("root")],
            ..Default::default()
        };

        let result = engine
            .check_tool_execution_with_bindings(
                "custom_search",
                &serde_json::json!({"root": public.join("alias/secret.txt")}),
                &bindings,
            )
            .await
            .unwrap();

        assert!(result.is_blocked());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn path_restrictions_keep_results_through_symlink_aliases() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let workspace = root.path().join("workspace");
        let restricted = root.path().join("restricted");
        std::fs::create_dir(&workspace).unwrap();
        std::fs::create_dir(&restricted).unwrap();
        symlink(&restricted, workspace.join("alias")).unwrap();
        let candidate = workspace.join("alias/secret.txt");
        let bindings = ToolPolicyBindings {
            path_fields: vec![PathPolicyBinding::read("root")],
            ..Default::default()
        };

        let mut denied_config = enabled_security_config();
        denied_config.tools.insert(
            "custom_search".to_string(),
            ToolPolicyConfig {
                read_paths: vec![restricted.to_string_lossy().into_owned()],
                blocked_paths: vec![restricted.to_string_lossy().into_owned()],
                ..Default::default()
            },
        );
        let denied = ToolSecurityEngine::new(denied_config)
            .check_tool_execution_with_bindings(
                "custom_search",
                &serde_json::json!({"root": candidate}),
                &bindings,
            )
            .await
            .unwrap();
        assert!(denied.is_blocked());
        assert!(!denied.is_unavailable());

        let mut unavailable_config = enabled_security_config();
        unavailable_config.tools.insert(
            "custom_search".to_string(),
            ToolPolicyConfig {
                read_paths: vec![restricted.to_string_lossy().into_owned()],
                paths: PathPolicyConfig {
                    unavailable: vec![restricted.to_string_lossy().into_owned()],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let unavailable = ToolSecurityEngine::new(unavailable_config)
            .check_tool_execution_with_bindings(
                "custom_search",
                &serde_json::json!({"root": candidate}),
                &bindings,
            )
            .await
            .unwrap();
        assert!(unavailable.is_unavailable());

        let mut approval_config = enabled_security_config();
        approval_config.tools.insert(
            "custom_search".to_string(),
            ToolPolicyConfig {
                read_paths: vec![restricted.to_string_lossy().into_owned()],
                paths: PathPolicyConfig {
                    requires_approval: vec![restricted.to_string_lossy().into_owned()],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let approval = ToolSecurityEngine::new(approval_config)
            .check_tool_execution_with_bindings(
                "custom_search",
                &serde_json::json!({"root": candidate}),
                &bindings,
            )
            .await
            .unwrap();
        assert!(approval.requires_approval());
    }

    #[tokio::test]
    async fn copy_and_move_bindings_enforce_source_and_destination_roots() {
        let root = tempfile::tempdir().unwrap();
        let readable = root.path().join("readable");
        let writable = root.path().join("writable");
        let outside = root.path().join("outside");
        std::fs::create_dir(&readable).unwrap();
        std::fs::create_dir(&writable).unwrap();
        std::fs::create_dir(&outside).unwrap();

        let mut copy_config = enabled_security_config();
        copy_config.tools.insert(
            "copy_path".to_string(),
            ToolPolicyConfig {
                read_paths: vec![readable.to_string_lossy().into_owned()],
                write_paths: vec![writable.to_string_lossy().into_owned()],
                ..Default::default()
            },
        );
        let copy_engine = ToolSecurityEngine::new(copy_config);
        let copy_bindings = legacy_policy_bindings("copy_path");
        let copy_allowed = copy_engine
            .check_tool_execution_with_bindings(
                "copy_path",
                &serde_json::json!({
                    "source_path": readable.join("source.txt"),
                    "destination_path": writable.join("destination.txt")
                }),
                &copy_bindings,
            )
            .await
            .unwrap();
        assert!(copy_allowed.is_allowed());

        let copy_source_blocked = copy_engine
            .check_tool_execution_with_bindings(
                "copy_path",
                &serde_json::json!({
                    "source_path": outside.join("source.txt"),
                    "destination_path": writable.join("destination.txt")
                }),
                &copy_bindings,
            )
            .await
            .unwrap();
        assert!(copy_source_blocked.is_blocked());

        let copy_destination_blocked = copy_engine
            .check_tool_execution_with_bindings(
                "copy_path",
                &serde_json::json!({
                    "source_path": readable.join("source.txt"),
                    "destination_path": outside.join("destination.txt")
                }),
                &copy_bindings,
            )
            .await
            .unwrap();
        assert!(copy_destination_blocked.is_blocked());

        let mut move_config = enabled_security_config();
        move_config.tools.insert(
            "move_path".to_string(),
            ToolPolicyConfig {
                read_paths: vec![readable.to_string_lossy().into_owned()],
                write_paths: vec![writable.to_string_lossy().into_owned()],
                ..Default::default()
            },
        );
        let move_engine = ToolSecurityEngine::new(move_config);
        let move_bindings = legacy_policy_bindings("move_path");
        let move_allowed = move_engine
            .check_tool_execution_with_bindings(
                "move_path",
                &serde_json::json!({
                    "source_path": writable.join("source.txt"),
                    "destination_path": writable.join("destination.txt")
                }),
                &move_bindings,
            )
            .await
            .unwrap();
        assert!(move_allowed.is_allowed());

        let move_source_blocked = move_engine
            .check_tool_execution_with_bindings(
                "move_path",
                &serde_json::json!({
                    "source_path": readable.join("source.txt"),
                    "destination_path": writable.join("destination.txt")
                }),
                &move_bindings,
            )
            .await
            .unwrap();
        assert!(move_source_blocked.is_blocked());

        let move_destination_blocked = move_engine
            .check_tool_execution_with_bindings(
                "move_path",
                &serde_json::json!({
                    "source_path": writable.join("source.txt"),
                    "destination_path": outside.join("destination.txt")
                }),
                &move_bindings,
            )
            .await
            .unwrap();
        assert!(move_destination_blocked.is_blocked());
    }

    #[test]
    fn custom_config_is_exposed_separately() {
        let mut config = enabled_security_config();
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
        let mut config = enabled_security_config();
        let tool_config = ToolPolicyConfig {
            max_results: Some(5),
            max_file_size_bytes: Some(1024),
            max_output_chars: Some(1000),
            ..Default::default()
        };
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
        let lower = engine.prepare_tool_arguments(
            "grep",
            &serde_json::json!({"pattern": "Tool", "max_results": 3}),
        );
        assert_eq!(lower.get("max_results").and_then(Value::as_u64), Some(3));
        assert_eq!(
            prepared.get("max_file_size_bytes").and_then(Value::as_u64),
            Some(1024)
        );
        assert_eq!(
            prepared.get("max_output_chars").and_then(Value::as_u64),
            Some(1000)
        );
    }

    #[test]
    fn invalid_result_limit_is_rejected_before_engine_construction() {
        let mut config = enabled_security_config();
        config.tools.insert(
            "web_search".to_string(),
            ToolPolicyConfig {
                max_results: Some(0),
                ..Default::default()
            },
        );
        let error = ToolSecurityEngine::try_new(config).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("max_results must be greater than 0")
        );
    }

    #[tokio::test]
    async fn fail_closed_blocks_missing_result_limit_bindings() {
        let mut config = ToolSecurityConfig {
            fail_closed: true,
            ..enabled_security_config()
        };
        let tool_config = ToolPolicyConfig {
            max_results: Some(5),
            ..Default::default()
        };
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
        let mut config = ToolSecurityConfig {
            fail_closed: true,
            ..enabled_security_config()
        };
        let tool_config = ToolPolicyConfig {
            max_results: Some(5),
            ..Default::default()
        };
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

    #[tokio::test]
    async fn read_paths_do_not_authorize_file_write() {
        let mut config = enabled_security_config();
        let tool_config = ToolPolicyConfig {
            read_paths: vec!["./workspace".to_string()],
            no_write_policy: NoWritePolicyBehavior::Deny,
            ..Default::default()
        };
        config.tools.insert("file_write".to_string(), tool_config);
        let engine = ToolSecurityEngine::new(config);

        let result = engine
            .check_tool_execution_with_bindings(
                "file_write",
                &serde_json::json!({"path": "./workspace/out.txt", "dry_run": false}),
                &legacy_policy_bindings("file_write"),
            )
            .await
            .unwrap();

        assert!(result.is_blocked());
    }

    #[tokio::test]
    async fn command_cwd_requires_working_dir_allowlist() {
        let mut config = enabled_security_config();
        let tool_config = ToolPolicyConfig {
            read_paths: vec![".".to_string()],
            allowed_commands: vec![CommandRuleConfig {
                argv: vec!["cargo".to_string(), "fmt".to_string(), "--all".to_string()],
            }],
            ..Default::default()
        };
        config.tools.insert("command".to_string(), tool_config);
        let engine = ToolSecurityEngine::new(config);

        let result = engine
            .check_tool_execution_with_bindings(
                "command",
                &serde_json::json!({"argv": ["cargo", "fmt", "--all"], "cwd": "."}),
                &legacy_policy_bindings("command"),
            )
            .await
            .unwrap();

        assert!(result.is_blocked());
    }

    #[tokio::test]
    async fn command_requires_exact_argv_allowlist() {
        let mut config = enabled_security_config();
        let tool_config = ToolPolicyConfig {
            allow_without_confirmation: true,
            working_dirs: vec![".".to_string()],
            ..Default::default()
        };
        config.tools.insert("command".to_string(), tool_config);
        let engine = ToolSecurityEngine::new(config);

        let result = engine
            .check_tool_execution_with_bindings(
                "command",
                &serde_json::json!({"argv": ["cargo", "fmt", "--all"], "cwd": "."}),
                &legacy_policy_bindings("command"),
            )
            .await
            .unwrap();

        assert!(result.is_blocked());
        assert!(
            result
                .reason()
                .unwrap()
                .contains("requires allowed_commands or command_templates")
        );
    }

    #[tokio::test]
    async fn command_exact_argv_allowlist_is_enforced() {
        let mut config = enabled_security_config();
        let tool_config = ToolPolicyConfig {
            allowed_commands: vec![CommandRuleConfig {
                argv: vec!["cargo".to_string(), "fmt".to_string(), "--all".to_string()],
            }],
            working_dirs: vec![".".to_string()],
            ..Default::default()
        };
        config.tools.insert("command".to_string(), tool_config);
        let engine = ToolSecurityEngine::new(config);

        let allowed = engine
            .check_tool_execution_with_bindings(
                "command",
                &serde_json::json!({"argv": ["cargo", "fmt", "--all"], "cwd": "."}),
                &legacy_policy_bindings("command"),
            )
            .await
            .unwrap();
        assert!(allowed.is_allowed());

        let blocked = engine
            .check_tool_execution_with_bindings(
                "command",
                &serde_json::json!({"argv": ["cargo", "test"], "cwd": "."}),
                &legacy_policy_bindings("command"),
            )
            .await
            .unwrap();
        assert!(blocked.is_blocked());
    }

    #[tokio::test]
    async fn validation_does_not_consume_rate_limit_admission() {
        let mut config = enabled_security_config();
        let tool_config = ToolPolicyConfig {
            rate_limit: Some(1),
            ..Default::default()
        };
        config.tools.insert("limited".to_string(), tool_config);
        let engine = ToolSecurityEngine::new(config);
        let bindings = legacy_policy_bindings("limited");

        for _ in 0..3 {
            let result = engine
                .validate_tool_execution_with_bindings("limited", &serde_json::json!({}), &bindings)
                .await
                .unwrap();
            assert!(result.is_allowed());
        }
        assert!(engine.admit_tool_execution("limited").is_allowed());
        assert!(engine.admit_tool_execution("limited").is_blocked());
    }

    #[tokio::test]
    async fn public_check_preserves_rate_limit_admission() {
        let mut config = enabled_security_config();
        let tool_config = ToolPolicyConfig {
            rate_limit: Some(1),
            ..Default::default()
        };
        config.tools.insert("limited".to_string(), tool_config);
        let engine = ToolSecurityEngine::new(config);

        let first = engine
            .check_tool_execution("limited", &serde_json::json!({}))
            .await
            .unwrap();
        let second = engine
            .check_tool_execution("limited", &serde_json::json!({}))
            .await
            .unwrap();

        assert!(first.is_allowed());
        assert!(second.is_blocked());
    }

    #[test]
    fn concurrent_rate_limit_admission_is_atomic() {
        let mut config = enabled_security_config();
        let tool_config = ToolPolicyConfig {
            rate_limit: Some(1),
            ..Default::default()
        };
        config.tools.insert("limited".to_string(), tool_config);
        let engine = Arc::new(ToolSecurityEngine::new_with_policy_version(config, 17));
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let handles = (0..8)
            .map(|_| {
                let engine = Arc::clone(&engine);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    engine.admit_tool_execution("limited").is_allowed()
                })
            })
            .collect::<Vec<_>>();
        let admitted = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .filter(|admitted| *admitted)
            .count();

        assert_eq!(admitted, 1);
        assert_eq!(engine.policy_version(), 17);
    }

    #[tokio::test]
    async fn omitted_dry_run_is_treated_as_actual_mutation() {
        let mut config = enabled_security_config();
        for tool_id in ["file_edit", "copy_path"] {
            let tool_config = ToolPolicyConfig {
                no_write_policy: NoWritePolicyBehavior::DryRunOnly,
                ..Default::default()
            };
            config.tools.insert(tool_id.to_string(), tool_config);
        }
        let engine = ToolSecurityEngine::new(config);

        let edit = engine
            .check_tool_execution_with_bindings(
                "file_edit",
                &serde_json::json!({"path": "./note.txt"}),
                &legacy_policy_bindings("file_edit"),
            )
            .await
            .unwrap();
        assert!(edit.is_blocked());

        let copy = engine
            .check_tool_execution_with_bindings(
                "copy_path",
                &serde_json::json!({
                    "source_path": "./source.txt",
                    "destination_path": "./destination.txt"
                }),
                &legacy_policy_bindings("copy_path"),
            )
            .await
            .unwrap();
        assert!(copy.is_blocked());
    }

    #[tokio::test]
    async fn no_write_policy_dry_run_only_allows_dry_run() {
        let mut config = enabled_security_config();
        let tool_config = ToolPolicyConfig {
            no_write_policy: NoWritePolicyBehavior::DryRunOnly,
            ..Default::default()
        };
        config.tools.insert("file_edit".to_string(), tool_config);
        let engine = ToolSecurityEngine::new(config);

        let dry_run = engine
            .check_tool_execution_with_bindings(
                "file_edit",
                &serde_json::json!({"path": "./note.txt", "dry_run": true}),
                &legacy_policy_bindings("file_edit"),
            )
            .await
            .unwrap();
        assert!(dry_run.is_allowed());

        let actual = engine
            .check_tool_execution_with_bindings(
                "file_edit",
                &serde_json::json!({"path": "./note.txt", "dry_run": false}),
                &legacy_policy_bindings("file_edit"),
            )
            .await
            .unwrap();
        assert!(actual.is_blocked());
    }
}

use std::collections::HashMap;
use std::net::IpAddr;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use tracing::debug;

use super::config::*;
use ai_agents_core::Result;
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
        if !self.config.enabled {
            return args.clone();
        }
        let mut prepared = args.clone();
        let Some(tool_config) = self.config.tools.get(tool_id) else {
            return prepared;
        };
        normalize_default_path_argument(tool_id, &mut prepared);
        apply_policy_caps(tool_id, tool_config, &mut prepared);
        prepared
    }

    pub fn attach_internal_tool_policy(&self, tool_id: &str, args: &Value) -> Value {
        if !self.config.enabled || tool_id != "web_fetch" {
            return args.clone();
        }
        let mut prepared = args.clone();
        let Some(tool_config) = self.config.tools.get(tool_id) else {
            return prepared;
        };
        let Some(obj) = prepared.as_object_mut() else {
            return prepared;
        };
        obj.insert(
            "__ai_agents_policy".to_string(),
            serde_json::json!({
                "allowed_domains": &tool_config.allowed_domains,
                "blocked_domains": &tool_config.blocked_domains,
                "domain_allow": &tool_config.domains.allow,
                "domain_deny": &tool_config.domains.deny,
                "domain_requires_approval": &tool_config.domains.requires_approval,
                "domain_unavailable": &tool_config.domains.unavailable,
                "allowed_schemes": &tool_config.allowed_schemes,
                "allowed_ports": &tool_config.allowed_ports,
                "blocked_private_networks": tool_config.blocked_private_networks,
                "max_redirects": tool_config.max_redirects,
            }),
        );
        prepared
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

    pub async fn check_tool_execution(
        &self,
        tool_id: &str,
        args: &serde_json::Value,
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

        if let Some(result) = self.check_domain_policy(tool_id, tool_config, args) {
            return Ok(result);
        }

        if let Some(result) = self.check_path_policy(tool_id, tool_config, args) {
            return Ok(result);
        }

        if let Some(result) = self.check_operation_policy(tool_id, tool_config, args) {
            return Ok(result);
        }

        if let Some(result) = self.check_command_policy(tool_id, tool_config, args) {
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
        self.check_command_policy(tool_id, tool_config, &value)
            .unwrap_or(SecurityCheckResult::Allow)
    }

    fn check_domain_policy(
        &self,
        tool_id: &str,
        tool_config: &ToolPolicyConfig,
        args: &serde_json::Value,
    ) -> Option<SecurityCheckResult> {
        let url = args.get("url").and_then(|u| u.as_str())?;
        let parsed = match reqwest::Url::parse(url) {
            Ok(parsed) => parsed,
            Err(_) => {
                return Some(SecurityCheckResult::Block {
                    reason: format!("URL is invalid for tool '{}'", tool_id),
                });
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
                    reason: format!("URL port '{}' is not allowed for tool '{}'", port, tool_id),
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
                    reason: format!("Domain '{}' is unavailable for tool '{}'", pattern, tool_id),
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

        None
    }

    fn check_path_policy(
        &self,
        tool_id: &str,
        tool_config: &ToolPolicyConfig,
        args: &serde_json::Value,
    ) -> Option<SecurityCheckResult> {
        let path = args
            .get("path")
            .and_then(|p| p.as_str())
            .or_else(|| default_path_for_tool(tool_id))?;
        let normalized = normalize_path(path);

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
                    message: format!("Confirm access to path '{}' for tool '{}' ?", path, tool_id),
                });
            }
        }

        let allowed: Vec<&String> = tool_config
            .allowed_paths
            .iter()
            .chain(tool_config.read_paths.iter())
            .chain(tool_config.write_paths.iter())
            .chain(tool_config.paths.allow.iter())
            .collect();
        if !allowed.is_empty()
            && !allowed
                .iter()
                .any(|pattern| path_matches(pattern, &normalized))
        {
            return Some(SecurityCheckResult::Block {
                reason: format!("Path not in allowed list for tool '{}'", tool_id),
            });
        }

        None
    }

    fn check_operation_policy(
        &self,
        tool_id: &str,
        tool_config: &ToolPolicyConfig,
        args: &serde_json::Value,
    ) -> Option<SecurityCheckResult> {
        let operation = args
            .get("operation")
            .or_else(|| args.get("function"))
            .or_else(|| args.get("method"))
            .and_then(|v| v.as_str())
            .map(|value| value.trim().to_ascii_lowercase())?;

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

        None
    }

    fn check_command_policy(
        &self,
        tool_id: &str,
        tool_config: &ToolPolicyConfig,
        args: &serde_json::Value,
    ) -> Option<SecurityCheckResult> {
        let command = args.get("command").and_then(|v| v.as_str()).or_else(|| {
            args.get("argv")
                .and_then(|v| v.as_array())
                .and_then(|items| items.first())
                .and_then(|v| v.as_str())
        })?;
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

fn normalize_default_path_argument(tool_id: &str, args: &mut Value) {
    let Some(default_path) = default_path_for_tool(tool_id) else {
        return;
    };
    let Some(obj) = args.as_object_mut() else {
        return;
    };
    if !obj.contains_key("path") {
        obj.insert("path".to_string(), Value::String(default_path.to_string()));
    }
}

fn default_path_for_tool(tool_id: &str) -> Option<&'static str> {
    match tool_id {
        "glob" | "grep" | "git_status" | "git_diff" | "diagnostics" => Some("."),
        _ => None,
    }
}

fn apply_policy_caps(tool_id: &str, config: &ToolPolicyConfig, args: &mut Value) {
    let Some(obj) = args.as_object_mut() else {
        return;
    };
    match tool_id {
        "glob" | "file_list" | "git_status" | "diagnostics" => {
            apply_usize_cap(obj, "max_results", config.max_results);
        }
        "grep" => {
            apply_usize_cap(obj, "max_results", config.max_results);
            apply_u64_cap(obj, "max_file_size_bytes", config.max_file_size_bytes);
            apply_usize_cap(obj, "max_output_chars", config.max_output_chars);
        }
        "file_read" => {
            apply_u64_cap(obj, "max_bytes", config.max_file_size_bytes);
        }
        "git_diff" => {
            apply_usize_cap(obj, "max_output_chars", config.max_output_chars);
        }
        "web_fetch" => {
            apply_usize_cap(obj, "max_chars", config.max_output_chars);
            apply_usize_cap(obj, "max_response_bytes", config.max_response_bytes);
            apply_usize_cap(obj, "max_redirects", config.max_redirects);
        }
        _ => {}
    }
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
}

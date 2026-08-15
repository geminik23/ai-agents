pub use ai_agents_core::MAX_TOOL_TIMEOUT_MS;
use ai_agents_core::{AgentError, PermissionOutcome, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Runtime security configuration for tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSecurityConfig {
    /// Enables runtime tool security checks.
    #[serde(default)]
    pub enabled: bool,
    /// Blocks tools without explicit policy when security is enabled.
    #[serde(default)]
    pub fail_closed: bool,
    /// Default timeout for tool execution in milliseconds, up to [`MAX_TOOL_TIMEOUT_MS`].
    #[serde(default = "default_tool_timeout")]
    pub default_timeout_ms: u64,
    /// Per-tool policies keyed by canonical tool ID.
    #[serde(default)]
    pub tools: HashMap<String, ToolPolicyConfig>,
}

impl Default for ToolSecurityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            fail_closed: false,
            default_timeout_ms: default_tool_timeout(),
            tools: HashMap::new(),
        }
    }
}

impl ToolSecurityConfig {
    /// Validates policy values that have a framework-wide semantic domain.
    pub fn validate(&self) -> Result<()> {
        let mut invalid_timeout_paths = Vec::new();
        if self.default_timeout_ms > MAX_TOOL_TIMEOUT_MS {
            invalid_timeout_paths.push("tool_security.default_timeout_ms".to_string());
        }
        invalid_timeout_paths.extend(
            self.tools
                .iter()
                .filter(|(_, policy)| {
                    policy
                        .timeout_ms
                        .is_some_and(|timeout_ms| timeout_ms > MAX_TOOL_TIMEOUT_MS)
                })
                .map(|(tool_id, _)| format!("tool_security.tools.{tool_id}.timeout_ms")),
        );
        invalid_timeout_paths.sort();
        if !invalid_timeout_paths.is_empty() {
            return Err(AgentError::Config(format!(
                "{} must be no greater than {MAX_TOOL_TIMEOUT_MS} milliseconds",
                invalid_timeout_paths.join(", ")
            )));
        }

        let mut invalid_paths: Vec<String> = self
            .tools
            .iter()
            .filter(|(_, policy)| policy.max_results == Some(0))
            .map(|(tool_id, _)| format!("tool_security.tools.{tool_id}.max_results"))
            .collect();
        invalid_paths.sort();
        if invalid_paths.is_empty() {
            return Ok(());
        }
        Err(AgentError::Config(format!(
            "{} must be greater than 0",
            invalid_paths.join(", ")
        )))
    }
}

/// Behavior when a mutation tool has no explicit write policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NoWritePolicyBehavior {
    Deny,
    #[default]
    DryRunOnly,
}

/// Exact argv command allowed by command policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandRuleConfig {
    /// Full argv vector, including executable name.
    #[serde(default)]
    pub argv: Vec<String>,
}

/// Argv command template with literal and wildcard variable segments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandTemplateConfig {
    /// Template name used in evidence and diagnostics.
    pub name: String,
    /// Argv segments. Values in {braces} are template variables.
    #[serde(default)]
    pub argv: Vec<String>,
}

/// Per-tool policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPolicyConfig {
    /// Enables this tool policy.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Requires approval for this tool after hard denials pass.
    #[serde(default, alias = "require_approval")]
    pub require_confirmation: bool,
    /// Explicitly permits side-effecting calls to skip classification-default approval.
    #[serde(default)]
    pub allow_without_confirmation: bool,
    /// Message shown when tool-level approval is required.
    #[serde(default)]
    pub confirmation_message: Option<String>,
    /// Maximum allowed calls per minute.
    #[serde(default)]
    pub rate_limit: Option<u32>,
    /// Tool-specific timeout in milliseconds, up to [`MAX_TOOL_TIMEOUT_MS`].
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// Legacy domain allowlist mapped to domain policy.
    #[serde(default)]
    pub allowed_domains: Vec<String>,
    /// Legacy domain blocklist mapped to domain policy.
    #[serde(default)]
    pub blocked_domains: Vec<String>,
    /// Legacy path allowlist mapped to path policy.
    #[serde(default)]
    pub allowed_paths: Vec<String>,
    /// Explicit read path allowlist for local read-only tools.
    #[serde(default)]
    pub read_paths: Vec<String>,
    /// Explicit write path allowlist for local mutation tools.
    #[serde(default)]
    pub write_paths: Vec<String>,
    /// Paths that override any allowlist.
    #[serde(default)]
    pub blocked_paths: Vec<String>,
    /// Maximum file size read or searched by local tools.
    #[serde(default)]
    pub max_file_size_bytes: Option<u64>,
    /// Maximum model-facing output characters.
    #[serde(default)]
    pub max_output_chars: Option<usize>,
    /// Maximum rows or entries for list/search tools.
    #[serde(default)]
    pub max_results: Option<usize>,
    /// Maximum response bytes for web fetch tools.
    #[serde(default)]
    pub max_response_bytes: Option<usize>,
    /// Blocks private, localhost, link-local, and metadata network targets.
    #[serde(default = "default_true")]
    pub blocked_private_networks: bool,
    /// Allowed URL schemes for network tools.
    #[serde(default)]
    pub allowed_schemes: Vec<String>,
    /// Allowed URL ports for network tools.
    #[serde(default)]
    pub allowed_ports: Vec<u16>,
    /// Maximum redirect count for network tools.
    #[serde(default)]
    pub max_redirects: Option<usize>,
    /// Maximum files a mutation tool may change.
    #[serde(default)]
    pub max_changed_files: Option<usize>,
    /// Maximum changed lines a mutation tool may produce.
    #[serde(default)]
    pub max_changed_lines: Option<usize>,
    /// Maximum exact replacements a mutation tool may perform.
    #[serde(default)]
    pub max_replacements: Option<usize>,
    /// Requires a matching file-read version before mutating an existing file.
    #[serde(default)]
    pub require_read_before_write: bool,
    /// Allows overwriting existing files for mutation tools.
    #[serde(default)]
    pub overwrite_existing: bool,
    /// Allows mutation tools to create missing parent directories.
    #[serde(default)]
    pub create_parent_dirs: bool,
    /// Behavior when no write_paths allowlist is configured.
    #[serde(default)]
    pub no_write_policy: NoWritePolicyBehavior,
    /// Exact argv allowlist for the command tool.
    #[serde(default)]
    pub allowed_commands: Vec<CommandRuleConfig>,
    /// Argv templates for the command tool.
    #[serde(default)]
    pub command_templates: Vec<CommandTemplateConfig>,
    /// Working directories allowed for command execution.
    #[serde(default)]
    pub working_dirs: Vec<String>,
    /// Environment variables that may be passed from tool arguments.
    #[serde(default)]
    pub env_passthrough: Vec<String>,
    /// Environment variables redacted from evidence.
    #[serde(default)]
    pub redact_env: Vec<String>,
    /// Reject shell-like command strings and metacharacters.
    #[serde(default = "default_true")]
    pub deny_shell: bool,
    /// Reject interactive command execution.
    #[serde(default = "default_true")]
    pub deny_interactive: bool,
    /// Allows approval-based command escalation beyond the allowlist.
    #[serde(default)]
    pub allow_command_escalation: bool,
    /// Parsed domain policy.
    #[serde(default)]
    pub domains: DomainPolicyConfig,
    /// Normalized path policy.
    #[serde(default)]
    pub paths: PathPolicyConfig,
    /// Command policy for process-backed tools.
    #[serde(default)]
    pub commands: CommandPolicyConfig,
    /// Operation policy based on arguments such as operation, function, or method.
    #[serde(default)]
    pub operations: OperationPolicyConfig,
    /// Custom tool settings exposed through ToolExecutionContext.custom_config.
    #[serde(default)]
    pub config: HashMap<String, serde_json::Value>,
}

impl Default for ToolPolicyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            require_confirmation: false,
            allow_without_confirmation: false,
            confirmation_message: None,
            rate_limit: None,
            timeout_ms: None,
            allowed_domains: Vec::new(),
            blocked_domains: Vec::new(),
            allowed_paths: Vec::new(),
            read_paths: Vec::new(),
            write_paths: Vec::new(),
            blocked_paths: Vec::new(),
            max_file_size_bytes: None,
            max_output_chars: None,
            max_results: None,
            max_response_bytes: None,
            blocked_private_networks: true,
            allowed_schemes: Vec::new(),
            allowed_ports: Vec::new(),
            max_redirects: None,
            max_changed_files: None,
            max_changed_lines: None,
            max_replacements: None,
            require_read_before_write: false,
            overwrite_existing: false,
            create_parent_dirs: false,
            no_write_policy: NoWritePolicyBehavior::default(),
            allowed_commands: Vec::new(),
            command_templates: Vec::new(),
            working_dirs: Vec::new(),
            env_passthrough: Vec::new(),
            redact_env: Vec::new(),
            deny_shell: true,
            deny_interactive: true,
            allow_command_escalation: false,
            domains: DomainPolicyConfig::default(),
            paths: PathPolicyConfig::default(),
            commands: CommandPolicyConfig::default(),
            operations: OperationPolicyConfig::default(),
            config: HashMap::new(),
        }
    }
}

/// Domain allow, deny, approval, and unavailable policy lists.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DomainPolicyConfig {
    #[serde(default)]
    pub allow: Vec<String>,
    #[serde(default)]
    pub deny: Vec<String>,
    #[serde(default)]
    pub requires_approval: Vec<String>,
    #[serde(default)]
    pub unavailable: Vec<String>,
}

/// Path allow, deny, approval, and unavailable policy lists.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PathPolicyConfig {
    #[serde(default)]
    pub allow: Vec<String>,
    #[serde(default)]
    pub deny: Vec<String>,
    #[serde(default)]
    pub requires_approval: Vec<String>,
    #[serde(default)]
    pub unavailable: Vec<String>,
}

/// Command allow, deny, approval, unavailable, and typed execution policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandPolicyConfig {
    #[serde(default)]
    pub allow: Vec<String>,
    #[serde(default)]
    pub deny: Vec<String>,
    #[serde(default)]
    pub requires_approval: Vec<String>,
    #[serde(default)]
    pub unavailable: Vec<String>,
    #[serde(default)]
    pub allowed_commands: Vec<CommandRuleConfig>,
    #[serde(default)]
    pub templates: Vec<CommandTemplateConfig>,
    #[serde(default)]
    pub working_dirs: Vec<String>,
    #[serde(default)]
    pub env_passthrough: Vec<String>,
    #[serde(default = "default_true")]
    pub deny_shell: bool,
    #[serde(default = "default_true")]
    pub deny_interactive: bool,
    #[serde(default)]
    pub allow_escalation: bool,
}

impl Default for CommandPolicyConfig {
    fn default() -> Self {
        Self {
            allow: Vec::new(),
            deny: Vec::new(),
            requires_approval: Vec::new(),
            unavailable: Vec::new(),
            allowed_commands: Vec::new(),
            templates: Vec::new(),
            working_dirs: Vec::new(),
            env_passthrough: Vec::new(),
            deny_shell: true,
            deny_interactive: true,
            allow_escalation: false,
        }
    }
}

/// Operation allow, deny, approval, and unavailable policy lists.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OperationPolicyConfig {
    #[serde(default)]
    pub allow: Vec<String>,
    #[serde(default)]
    pub deny: Vec<String>,
    #[serde(default)]
    pub requires_approval: Vec<String>,
    #[serde(default)]
    pub unavailable: Vec<String>,
}

/// Security decision returned by the tool security engine.
#[derive(Debug, Clone)]
pub enum SecurityCheckResult {
    Allow,
    Block { reason: String },
    Warn { message: String },
    RequireConfirmation { message: String },
    Unavailable { reason: String },
}

impl SecurityCheckResult {
    /// Returns true when execution may continue without blocking.
    pub fn is_allowed(&self) -> bool {
        matches!(
            self,
            SecurityCheckResult::Allow | SecurityCheckResult::Warn { .. }
        )
    }

    /// Returns true when execution must not invoke the tool.
    pub fn is_blocked(&self) -> bool {
        matches!(
            self,
            SecurityCheckResult::Block { .. } | SecurityCheckResult::Unavailable { .. }
        )
    }

    /// Converts the security result to a stable permission outcome.
    pub fn outcome(&self) -> PermissionOutcome {
        match self {
            SecurityCheckResult::Allow | SecurityCheckResult::Warn { .. } => {
                PermissionOutcome::Allow
            }
            SecurityCheckResult::Block { .. } => PermissionOutcome::Deny,
            SecurityCheckResult::RequireConfirmation { .. } => PermissionOutcome::RequiresApproval,
            SecurityCheckResult::Unavailable { .. } => PermissionOutcome::Unavailable,
        }
    }

    /// Returns the human-readable reason or warning message.
    pub fn reason(&self) -> Option<&str> {
        match self {
            SecurityCheckResult::Allow => None,
            SecurityCheckResult::Block { reason } => Some(reason),
            SecurityCheckResult::Warn { message } => Some(message),
            SecurityCheckResult::RequireConfirmation { message } => Some(message),
            SecurityCheckResult::Unavailable { reason } => Some(reason),
        }
    }

    /// Returns true when a human approval request is required.
    pub fn requires_approval(&self) -> bool {
        matches!(self, SecurityCheckResult::RequireConfirmation { .. })
    }

    /// Returns true when the tool or required host policy is unavailable.
    pub fn is_unavailable(&self) -> bool {
        matches!(self, SecurityCheckResult::Unavailable { .. })
    }
}

fn default_tool_timeout() -> u64 {
    30000
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ToolSecurityConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.default_timeout_ms, 30000);
        assert!(config.tools.is_empty());
    }

    #[test]
    fn test_yaml_parsing() {
        let yaml = r#"
enabled: true
default_timeout_ms: 10000
tools:
  http:
    rate_limit: 10
    blocked_domains:
      - evil.com
    allowed_domains:
      - api.example.com
  file_write:
    require_confirmation: true
    confirmation_message: "Are you sure you want to write this file?"
    allowed_paths:
      - /tmp/
"#;
        let config: ToolSecurityConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.enabled);
        assert_eq!(config.default_timeout_ms, 10000);
        assert!(config.tools.contains_key("http"));
        assert!(config.tools.contains_key("file_write"));

        let http = config.tools.get("http").unwrap();
        assert_eq!(http.rate_limit, Some(10));
        assert_eq!(http.blocked_domains, vec!["evil.com"]);

        let file_write = config.tools.get("file_write").unwrap();
        assert!(file_write.require_confirmation);
    }

    #[test]
    fn test_security_check_result() {
        let allow = SecurityCheckResult::Allow;
        assert!(allow.is_allowed());
        assert!(!allow.is_blocked());

        let block = SecurityCheckResult::Block {
            reason: "test".into(),
        };
        assert!(!block.is_allowed());
        assert!(block.is_blocked());

        let warn = SecurityCheckResult::Warn {
            message: "warning".into(),
        };
        assert!(warn.is_allowed());
        assert!(!warn.is_blocked());
    }

    #[test]
    fn test_tool_policy_defaults() {
        let policy = ToolPolicyConfig::default();
        assert!(policy.enabled);
        assert!(!policy.require_confirmation);
        assert!(policy.rate_limit.is_none());
    }

    #[test]
    fn tool_timeouts_accept_the_documented_range() {
        let mut config = ToolSecurityConfig {
            default_timeout_ms: MAX_TOOL_TIMEOUT_MS,
            ..Default::default()
        };
        config.tools.insert(
            "slow".to_string(),
            ToolPolicyConfig {
                timeout_ms: Some(MAX_TOOL_TIMEOUT_MS),
                ..Default::default()
            },
        );

        assert!(config.validate().is_ok());
    }

    #[test]
    fn default_tool_timeout_rejects_unrepresentable_values() {
        for timeout_ms in [MAX_TOOL_TIMEOUT_MS + 1, u64::MAX] {
            let config = ToolSecurityConfig {
                default_timeout_ms: timeout_ms,
                ..Default::default()
            };
            let error = config.validate().unwrap_err();
            assert!(error.to_string().contains(&format!(
                "tool_security.default_timeout_ms must be no greater than {MAX_TOOL_TIMEOUT_MS} milliseconds"
            )));
        }
    }

    #[test]
    fn per_tool_timeout_rejects_unrepresentable_values_with_full_paths() {
        for timeout_ms in [MAX_TOOL_TIMEOUT_MS + 1, u64::MAX] {
            let mut config = ToolSecurityConfig::default();
            config.tools.insert(
                "slow".to_string(),
                ToolPolicyConfig {
                    timeout_ms: Some(timeout_ms),
                    ..Default::default()
                },
            );
            let error = config.validate().unwrap_err();
            assert!(error.to_string().contains(&format!(
                "tool_security.tools.slow.timeout_ms must be no greater than {MAX_TOOL_TIMEOUT_MS} milliseconds"
            )));
        }
    }

    #[test]
    fn max_results_must_be_positive() {
        let mut config = ToolSecurityConfig::default();
        config.tools.insert(
            "web_search".to_string(),
            ToolPolicyConfig {
                max_results: Some(0),
                ..Default::default()
            },
        );
        let error = config.validate().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("tool_security.tools.web_search.max_results must be greater than 0")
        );

        config.tools.get_mut("web_search").unwrap().max_results = Some(1);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn zero_redirect_limit_remains_valid() {
        let mut config = ToolSecurityConfig::default();
        config.tools.insert(
            "web_fetch".to_string(),
            ToolPolicyConfig {
                max_redirects: Some(0),
                ..Default::default()
            },
        );
        assert!(config.validate().is_ok());
    }
}

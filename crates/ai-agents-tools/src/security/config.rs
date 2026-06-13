use ai_agents_core::PermissionOutcome;
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
    /// Default timeout for tool execution in milliseconds.
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

/// Per-tool policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPolicyConfig {
    /// Enables this tool policy.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Requires approval for this tool after hard denials pass.
    #[serde(default, alias = "require_approval")]
    pub require_confirmation: bool,
    /// Message shown when tool-level approval is required.
    #[serde(default)]
    pub confirmation_message: Option<String>,
    /// Maximum allowed calls per minute.
    #[serde(default)]
    pub rate_limit: Option<u32>,
    /// Tool-specific timeout in milliseconds.
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
}

impl Default for ToolPolicyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            require_confirmation: false,
            confirmation_message: None,
            rate_limit: None,
            timeout_ms: None,
            allowed_domains: Vec::new(),
            blocked_domains: Vec::new(),
            allowed_paths: Vec::new(),
            domains: DomainPolicyConfig::default(),
            paths: PathPolicyConfig::default(),
            commands: CommandPolicyConfig::default(),
            operations: OperationPolicyConfig::default(),
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

/// Command allow, deny, approval, and unavailable policy lists.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CommandPolicyConfig {
    #[serde(default)]
    pub allow: Vec<String>,
    #[serde(default)]
    pub deny: Vec<String>,
    #[serde(default)]
    pub requires_approval: Vec<String>,
    #[serde(default)]
    pub unavailable: Vec<String>,
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
}

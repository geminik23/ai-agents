use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSecurityConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_tool_timeout")]
    pub default_timeout_ms: u64,
    #[serde(default)]
    pub tools: HashMap<String, ToolPolicyConfig>,
}

impl Default for ToolSecurityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_timeout_ms: default_tool_timeout(),
            tools: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPolicyConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub require_confirmation: bool,
    #[serde(default)]
    pub confirmation_message: Option<String>,
    #[serde(default)]
    pub rate_limit: Option<u32>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    #[serde(default)]
    pub allowed_domains: Vec<String>,
    #[serde(default)]
    pub blocked_domains: Vec<String>,
    #[serde(default)]
    pub allowed_paths: Vec<String>,
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
        }
    }
}

#[derive(Debug, Clone)]
pub enum SecurityCheckResult {
    Allow,
    Block { reason: String },
    Warn { message: String },
    RequireConfirmation { message: String },
}

impl SecurityCheckResult {
    pub fn is_allowed(&self) -> bool {
        matches!(
            self,
            SecurityCheckResult::Allow | SecurityCheckResult::Warn { .. }
        )
    }

    pub fn is_blocked(&self) -> bool {
        matches!(self, SecurityCheckResult::Block { .. })
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

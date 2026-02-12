//! Provider configuration types for YAML specification

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use ai_agents_tools::{ToolAliases, TrustLevel};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProvidersConfig {
    #[serde(default)]
    pub builtin: BuiltinProviderConfig,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub yaml: Option<YamlProviderConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltinProviderConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default)]
    pub excluded_tools: Vec<String>,
}

fn default_true() -> bool {
    true
}

impl Default for BuiltinProviderConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            excluded_tools: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct YamlProviderConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default)]
    pub tools: Vec<YamlToolConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YamlToolConfig {
    pub id: String,

    pub name: String,

    pub description: String,

    #[serde(default)]
    pub input_schema: serde_json::Value,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aliases: Option<ToolAliases>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderSecurityConfig {
    #[serde(flatten)]
    pub providers: HashMap<String, ProviderPolicyConfig>,
}

/// Policy configuration for a provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderPolicyConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default)]
    pub trust_level: TrustLevel,

    #[serde(default)]
    pub tools: HashMap<String, ToolPolicyConfig>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

impl Default for ProviderPolicyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            trust_level: TrustLevel::default(),
            tools: HashMap::new(),
            timeout_ms: None,
        }
    }
}

/// Policy configuration for a tool within a provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPolicyConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,

    #[serde(default)]
    pub require_approval: bool,
}

impl Default for ToolPolicyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            timeout_ms: None,
            require_approval: false,
        }
    }
}

/// Global tool aliases configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolAliasesConfig {
    #[serde(flatten)]
    pub tools: HashMap<String, ToolAliases>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_providers_config_default() {
        let config = ProvidersConfig::default();
        assert!(config.builtin.enabled);
        assert!(config.yaml.is_none());
    }

    #[test]
    fn test_builtin_provider_config_default() {
        let config = BuiltinProviderConfig::default();
        assert!(config.enabled);
        assert!(config.excluded_tools.is_empty());
    }

    #[test]
    fn test_providers_config_yaml() {
        let yaml = r#"
builtin:
  enabled: true
  excluded_tools:
    - http
yaml:
  enabled: true
  tools:
    - id: custom_search
      name: Custom Search
      description: Search custom API
"#;

        let config: ProvidersConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.builtin.enabled);
        assert_eq!(config.builtin.excluded_tools.len(), 1);
        assert!(config.yaml.is_some());
        let yaml_config = config.yaml.unwrap();
        assert!(yaml_config.enabled);
        assert_eq!(yaml_config.tools.len(), 1);
    }

    #[test]
    fn test_provider_security_config_yaml() {
        let yaml = r#"
yaml:
  trust_level: high
  tools:
    run_script:
      require_approval: true
      timeout_ms: 30000
"#;

        let config: ProviderSecurityConfig = serde_yaml::from_str(yaml).unwrap();
        let yaml_policy = config.providers.get("yaml").unwrap();
        assert_eq!(yaml_policy.trust_level, TrustLevel::High);
        assert!(yaml_policy.tools.contains_key("run_script"));
        assert!(
            yaml_policy
                .tools
                .get("run_script")
                .unwrap()
                .require_approval
        );
    }

    #[test]
    fn test_tool_aliases_config_yaml() {
        let yaml = r#"
http:
  names:
    ko: 웹요청
    ja: ウェブリクエスト
  descriptions:
    ko: HTTP 요청을 보냅니다
    ja: HTTPリクエストを送信
calculator:
  names:
    ko: 계산기
"#;

        let config: ToolAliasesConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.tools.contains_key("http"));
        assert!(config.tools.contains_key("calculator"));

        let http_aliases = config.tools.get("http").unwrap();
        assert_eq!(http_aliases.get_name("ko"), Some("웹요청"));
    }

    #[test]
    fn test_yaml_tool_config_with_aliases() {
        let yaml = r#"
id: search
name: Web Search
description: Search the web
input_schema:
  type: object
  properties:
    query:
      type: string
aliases:
  names:
    ko: 웹검색
    ja: ウェブ検索
  descriptions:
    ko: 웹에서 검색합니다
"#;

        let config: YamlToolConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.id, "search");
        assert!(config.aliases.is_some());
        let aliases = config.aliases.unwrap();
        assert_eq!(aliases.get_name("ko"), Some("웹검색"));
    }
}

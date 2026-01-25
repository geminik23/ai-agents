use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolProviderType {
    #[default]
    Builtin,
    Yaml,
    Process,
    Mcp,
    Wasm,
    Http,
    Custom,
}

impl ToolProviderType {
    pub fn default_trust_level(&self) -> TrustLevel {
        match self {
            ToolProviderType::Builtin => TrustLevel::Full,
            ToolProviderType::Yaml => TrustLevel::High,
            ToolProviderType::Process => TrustLevel::Medium,
            ToolProviderType::Mcp => TrustLevel::Medium,
            ToolProviderType::Custom => TrustLevel::Medium,
            ToolProviderType::Wasm => TrustLevel::Sandboxed,
            ToolProviderType::Http => TrustLevel::Low,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustLevel {
    Low,
    Sandboxed,
    Medium,
    High,
    Full,
}

impl PartialOrd for TrustLevel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TrustLevel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_u8().cmp(&other.as_u8())
    }
}

impl TrustLevel {
    fn as_u8(&self) -> u8 {
        match self {
            TrustLevel::Low => 0,
            TrustLevel::Sandboxed => 1,
            TrustLevel::Medium => 2,
            TrustLevel::High => 3,
            TrustLevel::Full => 4,
        }
    }
}

impl Default for TrustLevel {
    fn default() -> Self {
        TrustLevel::Medium
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolAliases {
    #[serde(default)]
    pub names: HashMap<String, String>,
    #[serde(default)]
    pub descriptions: HashMap<String, String>,
    #[serde(default)]
    pub parameter_aliases: HashMap<String, HashMap<String, String>>,
}

impl ToolAliases {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_name(mut self, lang: impl Into<String>, name: impl Into<String>) -> Self {
        self.names.insert(lang.into(), name.into());
        self
    }

    pub fn with_description(mut self, lang: impl Into<String>, desc: impl Into<String>) -> Self {
        self.descriptions.insert(lang.into(), desc.into());
        self
    }

    pub fn get_name(&self, lang: &str) -> Option<&str> {
        self.names.get(lang).map(|s| s.as_str())
    }

    pub fn get_description(&self, lang: &str) -> Option<&str> {
        self.descriptions.get(lang).map(|s| s.as_str())
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty() && self.descriptions.is_empty() && self.parameter_aliases.is_empty()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolMetadata {
    #[serde(default)]
    pub tags: Vec<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_duration_ms: Option<u64>,

    #[serde(default)]
    pub has_side_effects: bool,

    #[serde(default)]
    pub requires_network: bool,

    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub custom: HashMap<String, Value>,
}

impl ToolMetadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_side_effects(mut self) -> Self {
        self.has_side_effects = true;
        self
    }

    pub fn with_network(mut self) -> Self {
        self.requires_network = true;
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct ToolContext {
    pub session_id: Option<String>,
    pub user_id: Option<String>,
    pub state_name: Option<String>,
    pub language: Option<String>,
    pub extra: HashMap<String, Value>,
}

impl ToolContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    pub fn with_state(mut self, state_name: impl Into<String>) -> Self {
        self.state_name = Some(state_name.into());
        self
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_type_default() {
        let pt = ToolProviderType::default();
        assert_eq!(pt, ToolProviderType::Builtin);
    }

    #[test]
    fn test_provider_type_trust_levels() {
        assert_eq!(
            ToolProviderType::Builtin.default_trust_level(),
            TrustLevel::Full
        );
        assert_eq!(
            ToolProviderType::Yaml.default_trust_level(),
            TrustLevel::High
        );
        assert_eq!(
            ToolProviderType::Wasm.default_trust_level(),
            TrustLevel::Sandboxed
        );
        assert_eq!(
            ToolProviderType::Http.default_trust_level(),
            TrustLevel::Low
        );
    }

    #[test]
    fn test_trust_level_ordering() {
        assert!(TrustLevel::Full > TrustLevel::High);
        assert!(TrustLevel::High > TrustLevel::Medium);
        assert!(TrustLevel::Medium > TrustLevel::Sandboxed);
        assert!(TrustLevel::Sandboxed > TrustLevel::Low);
    }

    #[test]
    fn test_tool_aliases() {
        let aliases = ToolAliases::new()
            .with_name("ko", "웹검색")
            .with_name("ja", "ウェブ検索")
            .with_description("ko", "웹에서 정보 검색");

        assert_eq!(aliases.get_name("ko"), Some("웹검색"));
        assert_eq!(aliases.get_name("ja"), Some("ウェブ検索"));
        assert_eq!(aliases.get_name("en"), None);
        assert_eq!(aliases.get_description("ko"), Some("웹에서 정보 검색"));
        assert!(!aliases.is_empty());
    }

    #[test]
    fn test_tool_metadata() {
        let metadata = ToolMetadata::new()
            .with_tags(vec!["network".to_string(), "api".to_string()])
            .with_side_effects()
            .with_network();

        assert_eq!(metadata.tags.len(), 2);
        assert!(metadata.has_side_effects);
        assert!(metadata.requires_network);
    }

    #[test]
    fn test_tool_context() {
        let ctx = ToolContext::new()
            .with_session("session123")
            .with_user("user456")
            .with_state("greeting")
            .with_language("ko");

        assert_eq!(ctx.session_id, Some("session123".to_string()));
        assert_eq!(ctx.user_id, Some("user456".to_string()));
        assert_eq!(ctx.state_name, Some("greeting".to_string()));
        assert_eq!(ctx.language, Some("ko".to_string()));
    }

    #[test]
    fn test_provider_type_serde() {
        let json = serde_json::to_string(&ToolProviderType::Builtin).unwrap();
        assert_eq!(json, "\"builtin\"");

        let pt: ToolProviderType = serde_json::from_str("\"yaml\"").unwrap();
        assert_eq!(pt, ToolProviderType::Yaml);
    }
}

//! Tool alias configuration types for YAML specifications.

use ai_agents_tools::ToolAliases;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Global tool aliases keyed by canonical tool ID.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolAliasesConfig {
    #[serde(flatten)]
    pub tools: HashMap<String, ToolAliases>,
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

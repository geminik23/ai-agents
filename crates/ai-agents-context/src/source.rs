use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RefreshPolicy {
    Once,
    #[default]
    PerSession,
    PerTurn,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum BuiltinSource {
    Datetime,
    Session,
    Agent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ContextSource {
    Runtime {
        #[serde(default)]
        required: bool,
        #[serde(default)]
        schema: Option<serde_json::Value>,
        #[serde(default)]
        default: Option<serde_json::Value>,
    },
    Builtin {
        source: BuiltinSource,
        #[serde(default)]
        refresh: RefreshPolicy,
    },
    File {
        path: String,
        #[serde(default)]
        refresh: RefreshPolicy,
        #[serde(default)]
        fallback: Option<String>,
    },
    Http {
        url: String,
        #[serde(default = "default_method")]
        method: String,
        #[serde(default)]
        headers: HashMap<String, String>,
        #[serde(default)]
        refresh: RefreshPolicy,
        #[serde(default)]
        cache_ttl: Option<u64>,
        #[serde(default)]
        timeout_ms: Option<u64>,
        #[serde(default)]
        fallback: Option<serde_json::Value>,
    },
    Env {
        name: String,
    },
    Callback {
        name: String,
        #[serde(default)]
        refresh: RefreshPolicy,
    },
}

fn default_method() -> String {
    "GET".to_string()
}

impl ContextSource {
    pub fn refresh_policy(&self) -> RefreshPolicy {
        match self {
            ContextSource::Runtime { .. } => RefreshPolicy::Once,
            ContextSource::Builtin { refresh, .. } => refresh.clone(),
            ContextSource::File { refresh, .. } => refresh.clone(),
            ContextSource::Http { refresh, .. } => refresh.clone(),
            ContextSource::Env { .. } => RefreshPolicy::Once,
            ContextSource::Callback { refresh, .. } => refresh.clone(),
        }
    }

    pub fn is_required(&self) -> bool {
        matches!(self, ContextSource::Runtime { required: true, .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_source() {
        let yaml = r#"
type: runtime
required: true
default:
  name: "Guest"
"#;
        let source: ContextSource = serde_yaml::from_str(yaml).unwrap();
        assert!(source.is_required());
        assert_eq!(source.refresh_policy(), RefreshPolicy::Once);
    }

    #[test]
    fn test_builtin_source() {
        let yaml = r#"
type: builtin
source: datetime
refresh: per_turn
"#;
        let source: ContextSource = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(source.refresh_policy(), RefreshPolicy::PerTurn);
    }

    #[test]
    fn test_file_source() {
        let yaml = r#"
type: file
path: "./rules/{{ context.user.language }}/support.txt"
refresh: per_session
fallback: "./rules/en/support.txt"
"#;
        let source: ContextSource = serde_yaml::from_str(yaml).unwrap();
        if let ContextSource::File { path, fallback, .. } = source {
            assert!(path.contains("{{ context.user.language }}"));
            assert_eq!(fallback, Some("./rules/en/support.txt".into()));
        } else {
            panic!("Expected File source");
        }
    }

    #[test]
    fn test_http_source() {
        let yaml = r#"
type: http
url: "https://api.example.com/users/{{ context.user.id }}"
method: GET
headers:
  Authorization: "Bearer {{ env.API_TOKEN }}"
refresh: per_session
cache_ttl: 300
timeout_ms: 5000
fallback:
  theme: "default"
"#;
        let source: ContextSource = serde_yaml::from_str(yaml).unwrap();
        if let ContextSource::Http {
            url,
            method,
            headers,
            cache_ttl,
            timeout_ms,
            ..
        } = source
        {
            assert!(url.contains("{{ context.user.id }}"));
            assert_eq!(method, "GET");
            assert!(headers.contains_key("Authorization"));
            assert_eq!(cache_ttl, Some(300));
            assert_eq!(timeout_ms, Some(5000));
        } else {
            panic!("Expected Http source");
        }
    }

    #[test]
    fn test_env_source() {
        let yaml = r#"
type: env
name: API_TOKEN
"#;
        let source: ContextSource = serde_yaml::from_str(yaml).unwrap();
        if let ContextSource::Env { name } = source {
            assert_eq!(name, "API_TOKEN");
        } else {
            panic!("Expected Env source");
        }
    }

    #[test]
    fn test_callback_source() {
        let yaml = r#"
type: callback
name: get_user_analytics
refresh: per_session
"#;
        let source: ContextSource = serde_yaml::from_str(yaml).unwrap();
        if let ContextSource::Callback { name, refresh } = source {
            assert_eq!(name, "get_user_analytics");
            assert_eq!(refresh, RefreshPolicy::PerSession);
        } else {
            panic!("Expected Callback source");
        }
    }
}

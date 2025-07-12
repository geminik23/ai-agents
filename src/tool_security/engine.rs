use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use tracing::debug;

use crate::error::Result;
use crate::tool_security::config::*;

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

#[derive(Debug)]
pub struct ToolSecurityEngine {
    config: ToolSecurityConfig,
    tool_call_tracker: Arc<RwLock<ToolCallTracker>>,
}

impl ToolSecurityEngine {
    pub fn new(config: ToolSecurityConfig) -> Self {
        Self {
            config,
            tool_call_tracker: Arc::new(RwLock::new(ToolCallTracker::default())),
        }
    }

    pub fn config(&self) -> &ToolSecurityConfig {
        &self.config
    }

    pub async fn check_tool_execution(
        &self,
        tool_id: &str,
        args: &serde_json::Value,
    ) -> Result<SecurityCheckResult> {
        if !self.config.enabled {
            return Ok(SecurityCheckResult::Allow);
        }

        let tool_config = self.config.tools.get(tool_id);

        if let Some(config) = tool_config {
            if !config.enabled {
                return Ok(SecurityCheckResult::Block {
                    reason: format!("Tool '{}' is disabled", tool_id),
                });
            }

            if let Some(rate_limit) = config.rate_limit {
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

            if let Some(url) = args.get("url").and_then(|u| u.as_str()) {
                for blocked in &config.blocked_domains {
                    if url.contains(blocked) {
                        return Ok(SecurityCheckResult::Block {
                            reason: format!(
                                "Domain '{}' is blocked for tool '{}'",
                                blocked, tool_id
                            ),
                        });
                    }
                }

                if !config.allowed_domains.is_empty() {
                    let is_allowed = config.allowed_domains.iter().any(|d| url.contains(d));
                    if !is_allowed {
                        return Ok(SecurityCheckResult::Block {
                            reason: format!(
                                "URL domain not in allowed list for tool '{}'",
                                tool_id
                            ),
                        });
                    }
                }
            }

            if let Some(path) = args.get("path").and_then(|p| p.as_str()) {
                if !config.allowed_paths.is_empty() {
                    let is_allowed = config.allowed_paths.iter().any(|p| path.starts_with(p));
                    if !is_allowed {
                        return Ok(SecurityCheckResult::Block {
                            reason: format!("Path not in allowed list for tool '{}'", tool_id),
                        });
                    }
                }
            }

            if config.require_confirmation {
                let message = config
                    .confirmation_message
                    .clone()
                    .unwrap_or_else(|| format!("Confirm execution of tool '{}'?", tool_id));
                return Ok(SecurityCheckResult::RequireConfirmation { message });
            }
        }

        self.tool_call_tracker.write().record_call(tool_id);
        debug!(tool_id = %tool_id, "Tool execution allowed");

        Ok(SecurityCheckResult::Allow)
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

        let args = serde_json::json!({"url": "https://good.com/api"});
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
}

//! CLI approval handler for HITL.
//!
//! Attached automatically by `build_agent` when the agent spec contains `hitl:` config.
//! Style and verbosity are controlled via `metadata.cli.hitl` in the agent YAML.

use std::io::{self, Write};
use std::sync::Arc;

use async_trait::async_trait;

use ai_agents::spec::{CliHitlMetadata, CliHitlStyle};
use ai_agents::{ApprovalHandler, ApprovalRequest, ApprovalResult};

/// CLI approval handler driven by `metadata.cli.hitl` in the agent YAML.
pub struct CliApprovalHandler {
    style: CliHitlStyle,
    show_context: bool,
}

impl CliApprovalHandler {
    /// Create a handler with explicit settings.
    pub fn new(style: CliHitlStyle, show_context: bool) -> Self {
        Self {
            style,
            show_context,
        }
    }

    /// Build from optional YAML metadata, defaulting to interactive prompt.
    pub fn from_metadata(meta: Option<&CliHitlMetadata>) -> Arc<dyn ApprovalHandler> {
        let (style, show_context) = match meta {
            Some(m) => (
                m.style.clone().unwrap_or_default(),
                m.show_context.unwrap_or(true),
            ),
            None => (CliHitlStyle::default(), true),
        };
        Arc::new(Self::new(style, show_context))
    }

    fn prompt_interactive(&self, request: &ApprovalRequest) -> ApprovalResult {
        println!();
        println!("+-----------------------------------------+");
        println!("|          APPROVAL REQUIRED               |");
        println!("+-----------------------------------------+");
        println!("  {}", request.message);

        if self.show_context && !request.context.is_empty() {
            println!();
            let mut pairs: Vec<(&String, &serde_json::Value)> = request.context.iter().collect();
            pairs.sort_by_key(|(k, _)| k.as_str());
            for (k, v) in pairs {
                let display = match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                println!("  {}: {}", k, display);
            }
        }

        println!("+-----------------------------------------+");
        print!("  Approve? [y/N] ");
        io::stdout().flush().unwrap_or_default();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap_or_default();

        if matches!(input.trim().to_lowercase().as_str(), "y" | "yes") {
            return ApprovalResult::approved();
        }

        print!("  Reason (optional, Enter to skip): ");
        io::stdout().flush().unwrap_or_default();
        let mut reason = String::new();
        io::stdin().read_line(&mut reason).unwrap_or_default();
        let reason = reason.trim();
        if reason.is_empty() {
            ApprovalResult::rejected(None)
        } else {
            ApprovalResult::rejected_with_reason(reason)
        }
    }
}

#[async_trait]
impl ApprovalHandler for CliApprovalHandler {
    async fn request_approval(&self, request: ApprovalRequest) -> ApprovalResult {
        match self.style {
            CliHitlStyle::Prompt => self.prompt_interactive(&request),
            CliHitlStyle::AutoApprove => {
                println!("\n[HITL] Auto-approving: {}\n", request.message);
                ApprovalResult::approved()
            }
            CliHitlStyle::AutoReject => {
                println!("\n[HITL] Auto-rejecting: {}\n", request.message);
                ApprovalResult::rejected(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_agents::hitl::ApprovalTrigger;

    fn test_request() -> ApprovalRequest {
        ApprovalRequest::new(
            ApprovalTrigger::tool("http", serde_json::json!({"method": "GET"})),
            "Approve GET request?",
        )
    }

    #[tokio::test]
    async fn test_auto_approve_returns_approved() {
        let handler = CliApprovalHandler::new(CliHitlStyle::AutoApprove, true);
        let result = handler.request_approval(test_request()).await;
        assert!(result.is_approved());
    }

    #[tokio::test]
    async fn test_auto_reject_returns_rejected() {
        let handler = CliApprovalHandler::new(CliHitlStyle::AutoReject, true);
        let result = handler.request_approval(test_request()).await;
        assert!(result.is_rejected());
    }

    #[test]
    fn test_from_metadata_none_defaults_to_prompt() {
        let _handler = CliApprovalHandler::from_metadata(None);
        // Smoke test: handler was created successfully without panic
    }

    #[test]
    fn test_from_metadata_auto_approve() {
        let meta = CliHitlMetadata {
            style: Some(CliHitlStyle::AutoApprove),
            show_context: Some(false),
        };
        let _handler = CliApprovalHandler::from_metadata(Some(&meta));
        // Handler created with specified style
    }
}

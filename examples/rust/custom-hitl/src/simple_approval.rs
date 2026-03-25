// Minimal ApprovalHandler -- implement one method, pass to the builder.
//
// This binary loads the same YAML agent as hitl-agent but uses a simpler
// handler with y/N only (no modify support).
//
// Flow: load YAML -> inject tools + YesNoHandler -> run REPL
//       -> tool call -> YesNoHandler -> y: execute / N: reject
//
// For the full handler with modify and language support, see src/hitl_agent.rs.

mod tools;

use ai_agents::{
    AgentBuilder, ApprovalHandler, ApprovalRequest, ApprovalResult, Result,
};
use ai_agents_cli::{CliRepl as Repl, init_tracing};
use async_trait::async_trait;
use std::io::{self, Write};
use std::sync::Arc;

use tools::{SendPaymentTool, DeleteRecordTool};

/// Minimal approval handler -- y/N only, no modify.
struct YesNoHandler;

#[async_trait]
impl ApprovalHandler for YesNoHandler {
    async fn request_approval(&self, request: ApprovalRequest) -> ApprovalResult {
        println!("\n[Approval needed] {}", request.message);
        print!("Approve? [y/N] ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        if matches!(input.trim().to_lowercase().as_str(), "y" | "yes") {
            ApprovalResult::approved()
        } else {
            ApprovalResult::rejected(None)
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let agent = AgentBuilder::from_yaml_file("agents/hitl_agent.yaml")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .tool(Arc::new(SendPaymentTool))
        .tool(Arc::new(DeleteRecordTool))
        .approval_handler(Arc::new(YesNoHandler))
        .build()?;

    Repl::new(agent)
        .welcome("=== Simple Approval Handler ===\n\nMinimal ApprovalHandler: y/N only.")
        .show_tool_calls()
        .hint("Try: Send $100 to Bob")
        .hint("Try: Delete user record U-9999")
        .run()
        .await
}

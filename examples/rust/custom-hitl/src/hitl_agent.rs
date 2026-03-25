// Full HITL demo -- interactive CLI handler with approve/reject/modify support.
//
// Flow: load YAML -> inject custom tools + approval handler -> run REPL
//       -> tool call -> CliApprovalHandler -> [Y] execute / [N] reject / [M] modify
//
// For a minimal handler (y/N only), see src/simple_approval.rs.

mod tools;

use ai_agents::{
    AgentBuilder, ApprovalHandler, ApprovalRequest, ApprovalResult, Result,
};
use ai_agents_cli::{CliRepl as Repl, init_tracing};
use async_trait::async_trait;
use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::Arc;
use serde_json::json;

use tools::{SendPaymentTool, DeleteRecordTool};

/// Interactive CLI handler with approve, reject, and modify support.
struct CliApprovalHandler;

#[async_trait]
impl ApprovalHandler for CliApprovalHandler {
    async fn request_approval(&self, request: ApprovalRequest) -> ApprovalResult {
        println!();
        println!("+-----------------------------------------+");
        println!("|          APPROVAL REQUIRED               |");
        println!("+-----------------------------------------+");
        println!("  {}", request.message);
        if !request.context.is_empty() {
            for (key, value) in &request.context {
                println!("  {}: {}", key, value);
            }
        }
        println!("+-----------------------------------------+");
        println!("  [Y] Approve  [N] Reject  [M] Modify");
        println!("+-----------------------------------------+");

        print!("Choice: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        match input.trim().to_lowercase().as_str() {
            "y" | "yes" => ApprovalResult::approved(),
            "m" | "modify" => {
                print!("New amount (Enter to skip): ");
                io::stdout().flush().unwrap();
                let mut amount = String::new();
                io::stdin().read_line(&mut amount).unwrap();
                if let Ok(val) = amount.trim().parse::<f64>() {
                    let mut changes = HashMap::new();
                    changes.insert("amount".to_string(), json!(val));
                    ApprovalResult::modified(changes)
                } else {
                    ApprovalResult::approved()
                }
            }
            _ => {
                print!("Reason (optional): ");
                io::stdout().flush().unwrap();
                let mut reason = String::new();
                io::stdin().read_line(&mut reason).unwrap();
                let reason = reason.trim();
                if reason.is_empty() {
                    ApprovalResult::rejected(None)
                } else {
                    ApprovalResult::rejected_with_reason(reason)
                }
            }
        }
    }

    fn preferred_language(&self) -> Option<String> {
        Some("en".to_string())
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
        .approval_handler(Arc::new(CliApprovalHandler))
        .build()?;

    Repl::new(agent)
        .welcome("=== HITL Agent Demo ===\n\nTools require human approval before execution.")
        .show_tool_calls()
        .hint("Try: 'Send $500 to Alice' (triggers payment approval)")
        .hint("Try: 'Delete user record U-1234' (triggers delete approval)")
        .hint("Try: 'Tell me a joke' (no tool, no approval needed)")
        .run()
        .await
}

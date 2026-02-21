use ai_agents::{
    AgentBuilder, ApprovalHandler, ApprovalRequest, ApprovalResult, Result, Tool, ToolRegistry,
    ToolResult,
};
use async_trait::async_trait;
use example_support::{Repl, init_tracing};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::Arc;

// ============================================================================
// Custom Approval Handler — interactive CLI
// ============================================================================

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

// ============================================================================
// Mock Tools
// ============================================================================

struct SendPaymentTool;

#[async_trait]
impl Tool for SendPaymentTool {
    fn id(&self) -> &str {
        "send_payment"
    }
    fn name(&self) -> &str {
        "Send Payment"
    }
    fn description(&self) -> &str {
        "Transfer funds to a recipient. Call immediately without asking for confirmation."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "amount": { "type": "number", "description": "Payment amount" },
                "currency": { "type": "string", "description": "Currency code (USD, KRW, etc.)" },
                "recipient": { "type": "string", "description": "Recipient name" }
            },
            "required": ["amount", "currency", "recipient"]
        })
    }
    async fn execute(&self, args: Value) -> ToolResult {
        let amount = args.get("amount").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let currency = args
            .get("currency")
            .and_then(|v| v.as_str())
            .unwrap_or("USD");
        let recipient = args
            .get("recipient")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");
        let tx_id = &uuid::Uuid::new_v4().to_string()[..8];
        ToolResult::ok(format!(
            "Payment of {} {} to {} processed. TX: {}",
            amount, currency, recipient, tx_id
        ))
    }
}

struct DeleteRecordTool;

#[async_trait]
impl Tool for DeleteRecordTool {
    fn id(&self) -> &str {
        "delete_record"
    }
    fn name(&self) -> &str {
        "Delete Record"
    }
    fn description(&self) -> &str {
        "Permanently delete a record. Call immediately without asking for confirmation."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "record_id": { "type": "string", "description": "Record ID to delete" },
                "record_type": { "type": "string", "description": "Type (user, order, etc.)" }
            },
            "required": ["record_id", "record_type"]
        })
    }
    async fn execute(&self, args: Value) -> ToolResult {
        let id = args
            .get("record_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let rtype = args
            .get("record_type")
            .and_then(|v| v.as_str())
            .unwrap_or("record");
        ToolResult::ok(format!("Deleted {} with ID: {}", rtype, id))
    }
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(SendPaymentTool)).unwrap();
    tools.register(Arc::new(DeleteRecordTool)).unwrap();

    let agent = AgentBuilder::from_template("hitl_agent")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .tools(tools)
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

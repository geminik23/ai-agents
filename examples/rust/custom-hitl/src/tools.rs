// Shared mock tools for HITL examples.

use ai_agents::{Tool, ToolResult};
use ai_agents::tools::generate_schema;
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;


#[derive(Debug, Deserialize, JsonSchema)]
struct SendPaymentInput {
    /// Payment amount
    amount: f64,
    /// Currency code (USD, KRW, etc.)
    currency: String,
    /// Recipient name
    recipient: String,
}

#[derive(Serialize)]
struct SendPaymentOutput {
    amount: f64,
    currency: String,
    recipient: String,
    tx_id: String,
}

/// Mock payment tool -- transfers funds to a recipient.
pub struct SendPaymentTool;

#[async_trait]
impl Tool for SendPaymentTool {
    fn id(&self) -> &str { "send_payment" }
    fn name(&self) -> &str { "Send Payment" }
    fn description(&self) -> &str {
        "Transfer funds to a recipient. Call immediately without asking for confirmation."
    }
    fn input_schema(&self) -> Value { generate_schema::<SendPaymentInput>() }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: SendPaymentInput = match serde_json::from_value(args) {
            Ok(i) => i,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };
        let tx_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
        let output = SendPaymentOutput {
            amount: input.amount,
            currency: input.currency,
            recipient: input.recipient,
            tx_id,
        };
        ToolResult::ok(serde_json::to_string(&output).unwrap())
    }
}


#[derive(Debug, Deserialize, JsonSchema)]
struct DeleteRecordInput {
    /// Record ID to delete
    record_id: String,
    /// Record type (user, order, etc.)
    record_type: String,
}

#[derive(Serialize)]
struct DeleteRecordOutput {
    record_id: String,
    record_type: String,
    deleted: bool,
}

/// Mock deletion tool -- permanently removes a record.
pub struct DeleteRecordTool;

#[async_trait]
impl Tool for DeleteRecordTool {
    fn id(&self) -> &str { "delete_record" }
    fn name(&self) -> &str { "Delete Record" }
    fn description(&self) -> &str {
        "Permanently delete a record. Call immediately without asking for confirmation."
    }
    fn input_schema(&self) -> Value { generate_schema::<DeleteRecordInput>() }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: DeleteRecordInput = match serde_json::from_value(args) {
            Ok(i) => i,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };
        let output = DeleteRecordOutput {
            record_id: input.record_id,
            record_type: input.record_type,
            deleted: true,
        };
        ToolResult::ok(serde_json::to_string(&output).unwrap())
    }
}

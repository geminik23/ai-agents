// Context-aware custom tool
//
// This example shows how a Rust custom tool should use ToolExecutionContext.
// Framework-owned caps come from ctx.limits.
// Tool-specific settings come from ctx.custom_config, which is populated from tool_security.tools.<tool_id>.config.
//
// Run: cd examples/rust/custom-tools && cargo run --bin context-tool

use ai_agents::tools::{
    ResultLimitBinding, ResultLimitKind, ToolCallClassification, ToolExecutionContext,
    ToolOperationKind, ToolPolicyBindings, ToolResult, ToolSafetyMetadata, generate_schema,
};
use ai_agents::{AgentBuilder, Result, Tool};
use ai_agents_cli::{CliRepl, init_tracing};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug, Deserialize, JsonSchema)]
struct CatalogSearchInput {
    /// Search query such as a product name, SKU, or category.
    query: String,
    /// Optional request cap. Runtime policy can lower this value.
    #[serde(default)]
    max_results: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct CatalogItem {
    sku: &'static str,
    name: &'static str,
    category: &'static str,
    price: f64,
    internal_id: &'static str,
}

#[derive(Debug, Serialize)]
struct CatalogSearchOutput {
    query: String,
    backend: String,
    tenant: String,
    currency: String,
    requested_name: String,
    canonical_id: String,
    max_results: usize,
    returned: usize,
    items: Vec<Value>,
}

struct CatalogSearchTool;

#[async_trait]
impl Tool for CatalogSearchTool {
    fn id(&self) -> &str {
        "catalog_search"
    }

    fn name(&self) -> &str {
        "Catalog Search"
    }

    fn description(&self) -> &str {
        "Search the product catalog. Uses runtime limits and tool-specific config from ToolExecutionContext."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<CatalogSearchInput>()
    }

    fn policy_bindings(&self) -> ToolPolicyBindings {
        ToolPolicyBindings {
            result_limit_fields: vec![ResultLimitBinding::new(
                "max_results",
                ResultLimitKind::MaxResults,
            )],
            ..Default::default()
        }
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata::read_only(ToolOperationKind::Read)
    }

    fn classify_call(&self, _args: &Value) -> ToolCallClassification {
        ToolCallClassification::from_metadata(&self.safety_metadata())
    }

    async fn execute(&self, args: Value, ctx: ToolExecutionContext) -> ToolResult {
        let input: CatalogSearchInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };

        let requested_limit = input.max_results.unwrap_or(10);
        let max_results = requested_limit.min(ctx.limits.max_results.unwrap_or(10));
        let backend = string_config(&ctx.custom_config, "backend", "memory");
        let tenant = string_config(&ctx.custom_config, "tenant", "default");
        let currency = string_config(&ctx.custom_config, "default_currency", "USD");
        let include_internal_ids = ctx
            .custom_config
            .get("include_internal_ids")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let query = input.query.to_lowercase();
        let items = catalog_items()
            .into_iter()
            .filter(|item| {
                item.sku.to_lowercase().contains(&query)
                    || item.name.to_lowercase().contains(&query)
                    || item.category.to_lowercase().contains(&query)
            })
            .take(max_results)
            .map(|item| item_to_value(&item, include_internal_ids))
            .collect::<Vec<_>>();

        let output = CatalogSearchOutput {
            query: input.query,
            backend,
            tenant,
            currency,
            requested_name: ctx.requested_name,
            canonical_id: ctx.canonical_id,
            max_results,
            returned: items.len(),
            items,
        };

        match serde_json::to_string(&output) {
            Ok(json) => ToolResult::ok(json),
            Err(error) => ToolResult::error(format!("Serialization error: {}", error)),
        }
    }
}

fn string_config(config: &Value, key: &str, default: &str) -> String {
    config
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_string()
}

fn item_to_value(item: &CatalogItem, include_internal_ids: bool) -> Value {
    let mut value = serde_json::json!({
        "sku": item.sku,
        "name": item.name,
        "category": item.category,
        "price": item.price,
    });
    if include_internal_ids {
        value["internal_id"] = Value::String(item.internal_id.to_string());
    }
    value
}

fn catalog_items() -> Vec<CatalogItem> {
    vec![
        CatalogItem {
            sku: "SHOE-001",
            name: "Trail Runner",
            category: "running shoes",
            price: 89.0,
            internal_id: "row-1001",
        },
        CatalogItem {
            sku: "SHOE-002",
            name: "City Runner",
            category: "running shoes",
            price: 79.0,
            internal_id: "row-1002",
        },
        CatalogItem {
            sku: "KIT-010",
            name: "Cast Iron Skillet",
            category: "kitchen",
            price: 42.0,
            internal_id: "row-2010",
        },
    ]
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let agent = AgentBuilder::from_yaml_file("agents/context_tool_agent.yaml")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .tool(Arc::new(CatalogSearchTool))
        .build()?;

    CliRepl::new(agent).show_tool_calls().run().await
}

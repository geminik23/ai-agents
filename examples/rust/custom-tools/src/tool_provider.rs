// Custom ToolProvider
//
// dynamic tool discovery from an external source.
//
// ToolProvider is the abstraction for a collection of tools served by one source:
// API gateways, plugin systems, microservices, or any system that can list and execute multiple tools dynamically.
//
// Key points:
// - Implementing the ToolProvider trait (id, name, provider_type, list_tools, get_tool)
// - Tool discovery via list_tools() -> Vec<ToolDescriptor>
// - Health checks and refresh lifecycle
// - Multi-language aliases via ToolDescriptor::with_aliases()
// - Registering a provider on a ToolRegistry, then merging via .extend_tools()
//
// The MockApiProvider simulates an external API that serves weather and stock tools.
// A real provider would make HTTP calls, connect to gRPC, or query a database.
//
// Builds on: stateful-tool (Tool trait), yaml-custom-tool (.tool() registration)
//
// Run: cd examples/rust/custom-tools && cargo run --bin tool-provider

use ai_agents::{
    AgentBuilder, ProviderType, Tool, ToolResult, UnifiedLLMProvider,
    tools::{
        ProviderHealth, ToolAliases, ToolDescriptor, ToolProvider, ToolProviderError,
        ToolProviderType, ToolRegistry, generate_schema,
    },
};
use ai_agents_cli::{CliRepl, init_tracing};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

// Individual tools that the provider will serve.
// These are regular Tool implementations - nothing special about them.
// The provider wraps them and makes them discoverable as a group.

#[derive(Debug, Deserialize, JsonSchema)]
struct WeatherInput {
    /// City name to get weather for
    city: String,
}

#[derive(Debug, Serialize)]
struct WeatherOutput {
    city: String,
    temperature_c: f64,
    condition: String,
    humidity_pct: u32,
}

struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn id(&self) -> &str { "weather" }
    fn name(&self) -> &str { "Weather" }
    fn description(&self) -> &str { "Get current weather for a city." }
    fn input_schema(&self) -> Value { generate_schema::<WeatherInput>() }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: WeatherInput = match serde_json::from_value(args) {
            Ok(i) => i,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };
        // Mock response - a real implementation would call a weather API.
        let output = WeatherOutput {
            city: input.city,
            temperature_c: 22.0,
            condition: "partly cloudy".into(),
            humidity_pct: 65,
        };
        match serde_json::to_string(&output) {
            Ok(json) => ToolResult::ok(json),
            Err(e) => ToolResult::error(format!("Serialization error: {}", e)),
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct StockPriceInput {
    /// Stock ticker symbol (e.g., AAPL, GOOGL, MSFT)
    symbol: String,
}

#[derive(Debug, Serialize)]
struct StockPriceOutput {
    symbol: String,
    price_usd: f64,
    found: bool,
}

struct StockPriceTool;

#[async_trait]
impl Tool for StockPriceTool {
    fn id(&self) -> &str { "stock_price" }
    fn name(&self) -> &str { "Stock Price" }
    fn description(&self) -> &str { "Get current stock price by ticker symbol." }
    fn input_schema(&self) -> Value { generate_schema::<StockPriceInput>() }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: StockPriceInput = match serde_json::from_value(args) {
            Ok(i) => i,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };
        // Mock prices - a real implementation would call a financial data API.
        let symbol = input.symbol.to_uppercase();
        let (price, found) = match symbol.as_str() {
            "AAPL" => (178.52, true),
            "GOOGL" => (141.80, true),
            "MSFT" => (415.30, true),
            _ => (0.0, false),
        };
        let output = StockPriceOutput { symbol, price_usd: price, found };
        match serde_json::to_string(&output) {
            Ok(json) => ToolResult::ok(json),
            Err(e) => ToolResult::error(format!("Serialization error: {}", e)),
        }
    }
}

// Custom ToolProvider - serves multiple tools as a discoverable collection.
//
// Key differences from registering tools individually:
// - list_tools() returns ToolDescriptors (metadata), not the tools themselves
// - get_tool() returns the implementation on demand
// - health_check() lets monitoring systems verify the external source is alive
// - refresh() re-discovers tools if the external source adds/removes them
// - provider_type() signals trust level for security policies
//
// The framework's MCP integration uses this same trait internally -
// MCPToolProvider calls list_tools() over JSON-RPC to discover MCP server tools.

struct MockApiProvider {
    tools: Vec<Arc<dyn Tool>>,
}

impl MockApiProvider {
    fn new() -> Self {
        Self {
            tools: vec![Arc::new(WeatherTool), Arc::new(StockPriceTool)],
        }
    }
}

#[async_trait]
impl ToolProvider for MockApiProvider {
    /// Unique provider ID - used to namespace tools and track health.
    fn id(&self) -> &str {
        "mock_api"
    }

    /// Human-readable name for logging and diagnostics.
    fn name(&self) -> &str {
        "Mock API Provider"
    }

    /// Provider type signals trust level. Custom providers get TrustLevel::Medium by default.
    /// Built-in tools get Full; MCP/Wasm providers get Sandboxed.
    fn provider_type(&self) -> ToolProviderType {
        ToolProviderType::Custom
    }

    /// Discover available tools. In a real provider this would call an external API, read a manifest, or query a service registry.
    async fn list_tools(&self) -> Vec<ToolDescriptor> {
        self.tools
            .iter()
            .map(|tool| {
                let mut desc = ToolDescriptor::new(
                    tool.id(),
                    tool.name(),
                    tool.description(),
                    tool.input_schema(),
                );

                // Multi-language aliases - the LLM and users can refer to this
                // tool by its localized name in any supported language.
                if tool.id() == "weather" {
                    desc = desc.with_aliases(
                        ToolAliases::new()
                            .with_name("ko", "날씨")
                            .with_name("ja", "天気")
                            .with_description("ko", "도시의 현재 날씨를 조회합니다")
                            .with_description("ja", "都市の現在の天気を取得します"),
                    );
                }
                if tool.id() == "stock_price" {
                    desc = desc.with_aliases(
                        ToolAliases::new()
                            .with_name("ko", "주가")
                            .with_name("ja", "株価"),
                    );
                }
                desc
            })
            .collect()
    }

    /// Return the tool implementation by ID. Called when the LLM selects a tool -
    /// the registry looks up which provider owns the tool and delegates here.
    async fn get_tool(&self, tool_id: &str) -> Option<Arc<dyn Tool>> {
        self.tools.iter().find(|t| t.id() == tool_id).cloned()
    }

    /// Health check - called by monitoring systems or the registry's provider_health() method.
    /// Return Healthy, Degraded, or Unavailable.
    async fn health_check(&self) -> ProviderHealth {
        // A real provider would ping the external service here.
        ProviderHealth::Healthy
    }

    /// Whether this provider supports runtime re-discovery of tools.
    fn supports_refresh(&self) -> bool {
        true
    }

    /// Re-discover tools from the external source.
    /// Called by registry.refresh_provider("mock_api").
    /// Useful when the external service adds or removes tools at runtime.
    async fn refresh(&self) -> Result<(), ToolProviderError> {
        // A real provider would re-query the external service here.
        Ok(())
    }
}

// Register the provider on a ToolRegistry and merge into the agent.
//
// Workflow:
// 1. Create a standalone ToolRegistry
// 2. register_provider() - async because it calls list_tools() to discover
// 3. Optionally check health
// 4. .extend_tools(registry) merges provider tools alongside built-ins
//
// Tool registration methods on AgentBuilder:
//   .tool(Arc::new(MyTool)) - add ONE custom tool, keeps built-ins
//   .extend_tools(registry) - merge a ToolRegistry in, skips duplicates
//   .tools(registry) - REPLACE the entire registry (built-ins gone)
//
// For individual tools, .tool() is sufficient (see simple-tool, schema-tool).
// For providers that serve multiple tools, build a ToolRegistry and use .extend_tools() as shown below.

#[tokio::main]
async fn main() -> ai_agents::Result<()> {
    init_tracing();

    let llm = UnifiedLLMProvider::from_env(ProviderType::OpenAI, "gpt-4.1-nano")?;

    // Create a registry and register the provider.
    // register_provider() is async because it calls list_tools() to discover and index all tools the provider serves.
    let provider_registry = ToolRegistry::new();
    let provider = Arc::new(MockApiProvider::new());
    provider_registry
        .register_provider(provider)
        .await
        .expect("Failed to register provider");

    // Optional: verify provider health before starting the agent.
    if let Some(health) = provider_registry.provider_health("mock_api").await {
        println!("Provider 'mock_api': {:?}", health);
    }

    // Build the agent. auto_configure_features() registers built-in tools (calculator, datetime, etc.).
    // extend_tools() merges the provider's discovered tools alongside built-ins - does NOT replace them.
    let agent = AgentBuilder::new()
        .system_prompt(
            "You are a helpful assistant with access to weather and stock market data.",
        )
        .llm(Arc::new(llm))
        .auto_configure_features()?
        .extend_tools(provider_registry)
        .build()?;

    CliRepl::new(agent)
        .welcome(
            "=== Tool Provider Demo ===\n\n\
             Tools discovered from a custom provider.\n\
             The provider also registers Korean/Japanese aliases for tool names.",
        )
        .show_tool_calls()
        .hint("Try: What's the weather in Tokyo?")
        .hint("Try: What's Apple's stock price?")
        .hint("Try: What time is it? (built-in datetime - still works)")
        .run()
        .await
}

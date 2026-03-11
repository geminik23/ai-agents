use ai_agents::{AgentBuilder, ContextProvider, Result};
use ai_agents_cli::{CliRepl as Repl, init_tracing};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

// This implements the `type: callback` context source declared in the YAML.
// The framework calls `get()` according to the refresh policy (per_turn here),
// so the usage stats update automatically on every conversation turn.

struct UsageStatsProvider {
    message_count: AtomicU32,
    total_tokens: AtomicU32,
}

impl UsageStatsProvider {
    fn new() -> Self {
        Self {
            message_count: AtomicU32::new(0),
            total_tokens: AtomicU32::new(0),
        }
    }
}

#[async_trait]
impl ContextProvider for UsageStatsProvider {
    async fn get(&self, _key: &str, _current_context: &Value) -> Result<Value> {
        // Increment counters each time the framework fetches this context.
        // In production this could query a database, call an internal API, etc.
        let count = self.message_count.fetch_add(1, Ordering::Relaxed) + 1;
        let tokens = self.total_tokens.fetch_add(150, Ordering::Relaxed) + 150; // used fixed token count for demo purposes

        Ok(json!({
            "message_count": count,
            "total_tokens": tokens,
        }))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    // 1. Load the companion YAML agent.
    //    The YAML declares context schema, defaults, and template variables.
    //    See agents/context_agent.yaml for the full definition.
    let agent = AgentBuilder::from_yaml_file("agents/context_agent.yaml")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .build()?;

    // 2. Override runtime context defaults with actual user data.
    //
    //    The YAML declares `context.user` as `type: runtime` with defaults
    //    ("Guest", "unknown", "free"). In production, you'd populate this
    //    from auth middleware, a database lookup, an HTTP request header, etc.
    //
    //    After this call the system prompt renders with "Jay" instead of "Guest".
    agent.set_context(
        "user",
        json!({
            "name": "Jay",
            "email": "jay@example.com",
            "tier": "vip",
        }),
    )?;

    // 3. Register a custom ContextProvider for the `type: callback` source.
    //
    //    The YAML declares:
    //      usage_stats:
    //        type: callback
    //        name: get_usage_stats
    //        refresh: per_turn
    //
    //    The framework calls provider.get() on every turn, so the system prompt
    //    always shows up-to-date usage statistics.
    let usage_provider = Arc::new(UsageStatsProvider::new());
    agent.register_context_provider("usage_stats", usage_provider);

    // 4. Run the interactive REPL.
    //    Context is now fully injected — the agent knows the user's name,
    //    tier, and tracks usage stats across turns.
    Repl::new(agent)
        .welcome(
            "=== Rust Context Injection Demo ===\n\n\
             This example demonstrates injecting context from Rust code:\n\
             • set_context()              - override YAML defaults with real user data\n\
             • register_context_provider() - custom ContextProvider for dynamic data",
        )
        .hint("Try: What's my name and email?")
        .hint("Try: What tier am I on?")
        .hint("Try: How many messages have I sent? (increases each turn)")
        .hint("Try: What time is it? (builtin context, auto-refreshed)")
        .hint("Context was injected via Rust — see src/context_injection.rs")
        .run()
        .await
}

+++
title = "Rust API"
weight = 4
template = "docs.html"
description = "Embedding AI Agents in your Rust application."
+++

Use `ai-agents` as a library to build, configure, and run agents entirely from Rust. Everything the CLI can do, you can do programmatically - plus custom tools, providers, memory backends, hooks, and more.

---

## Adding the Dependency

Add `ai-agents` to your `Cargo.toml`:

```toml
[dependencies]
ai-agents = "1.0.0-rc.11"
tokio = { version = "1", features = ["full"] }
anyhow = "1"
```

### Feature Flags

| Flag            | Description                                        |
| --------------- | -------------------------------------------------- |
| *(default)*     | Core framework, no optional storage or context     |
| `sqlite`        | SQLite storage backend for session persistence     |
| `redis-storage` | Redis storage backend for session persistence      |
| `http-context`  | HTTP context source for dynamic context injection  |
| `full-storage`  | `sqlite` + `redis-storage`                         |
| `full`          | All optional features enabled                      |

Enable features like this:

```toml
[dependencies]
ai-agents = { version = "1.0.0-rc.11", features = ["full"] }
```

---

## AgentBuilder

`AgentBuilder` is the main entry point. There are three ways to create an agent.

### Pattern 1: From a YAML file

Load a YAML spec and let the framework auto-configure LLM providers from environment variables:

```rust
use ai_agents::{AgentBuilder, Agent};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = AgentBuilder::from_yaml_file("agent.yaml")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .build()?;

    let response = agent.chat("Hello!").await?;
    println!("{}", response.content);
    Ok(())
}
```

- `auto_configure_llms()` reads the `llm:` and `llms:` blocks from the spec, resolves API keys from environment variables, and registers all providers automatically.
- `auto_configure_features()` wires up error recovery, tool security, process pipeline, and built-in tools from the spec.

If your YAML uses `mcp:` tools or `spawner:` config, add the async configuration steps:

```rust
use ai_agents::{AgentBuilder, Agent};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = AgentBuilder::from_yaml_file("agent.yaml")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .auto_configure_mcp().await?        // connects to MCP servers declared in tools
        .auto_configure_spawner().await?     // wires spawner tools if spawner: is present
        .build()?;

    let response = agent.chat("Hello!").await?;
    println!("{}", response.content);
    Ok(())
}
```

- `auto_configure_mcp()` connects to MCP servers declared in the `tools` list (entries with `type: mcp`), discovers their functions, and registers them. This is async because it spawns server processes and waits for handshake.
- `auto_configure_spawner()` reads the `spawner:` section, creates the spawner and agent registry, resolves file-based templates, and registers the four spawner tools (`generate_agent`, `send_message`, `list_agents`, `remove_agent`). This is async because it resolves template files from disk.

Both methods are no-ops when the relevant config section is absent, so it is safe to always include them in the chain.

### Pattern 2: From a YAML string

Useful when you store specs in a database, embed them as constants, or generate them at runtime:

```rust
use ai_agents::{AgentBuilder, Agent};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let yaml = r#"
        name: InlineAgent
        system_prompt: "You are a helpful assistant."
        llm:
          provider: openai
          model: gpt-4.1-nano
    "#;

    let agent = AgentBuilder::from_yaml(yaml)?
        .auto_configure_llms()?
        .auto_configure_features()?
        .build()?;

    let response = agent.chat("What is Rust?").await?;
    println!("{}", response.content);
    Ok(())
}
```

### Pattern 3: Fully programmatic

Build everything in code - no YAML at all:

```rust
use ai_agents::{AgentBuilder, Agent, UnifiedLLMProvider, ProviderType};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm = UnifiedLLMProvider::from_env(ProviderType::OpenAI, "gpt-4.1-nano")?;

    let agent = AgentBuilder::new()
        .system_prompt("You are a helpful assistant.")
        .llm(Arc::new(llm))
        .build()?;

    let response = agent.chat("Hello!").await?;
    println!("{}", response.content);
    Ok(())
}
```

You can mix patterns too - load from YAML, then override specific parts programmatically:

```rust
let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .auto_configure_mcp().await?
    .auto_configure_spawner().await?
    .system_prompt("Override the YAML prompt with this one.")
    .hooks(Arc::new(my_hooks))
    .tool(Arc::new(my_custom_tool))
    .build()?;
```

The full builder chain used by the CLI is: `auto_configure_llms` &#x2192; `auto_configure_features` &#x2192; `auto_configure_mcp` &#x2192; `auto_configure_spawner` &#x2192; `build`. The MCP and spawner steps must come after `auto_configure_features()` so the tool registry exists. Custom `.tool()`, `.hooks()`, and `.system_prompt()` calls can go anywhere before `.build()`.

---

## Basic Chat

The simplest interaction - send a message, get a response:

```rust
use ai_agents::{Agent, AgentBuilder};

let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .build()?;

let response = agent.chat("Explain ownership in Rust").await?;
println!("{}", response.content);

// Check if tools were used
if let Some(ref calls) = response.tool_calls {
    for call in calls {
        println!("Tool used: {} -> {}", call.name, call.result);
    }
}
```

The `AgentResponse` struct contains:
- `content` - the final text response
- `tool_calls` - optional list of tool calls made during the turn

---

## Streaming

Stream tokens as they arrive from the LLM:

```rust
use ai_agents::{Agent, AgentBuilder, StreamChunk};
use futures::StreamExt;

let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .build()?;

let mut stream = agent.chat_stream("Tell me a story").await?;

while let Some(chunk) = stream.next().await {
    match chunk {
        StreamChunk::Content { text } => {
            print!("{}", text);  // print tokens as they arrive
        }
        StreamChunk::ToolCallStart { name, .. } => {
            println!("\n[Tool: {}...]", name);
        }
        StreamChunk::ToolResult { name, output, success, .. } => {
            println!("[Tool {}: {} (ok={})]", name, output, success);
        }
        StreamChunk::StateTransition { from, to } => {
            println!("[State: {:?} → {}]", from, to);
        }
        StreamChunk::Done {} => break,
        StreamChunk::Error { message } => {
            eprintln!("Error: {}", message);
            break;
        }
        _ => {}  // ToolCallDelta, ToolCallEnd
    }
}
```

### StreamChunk variants

| Variant            | Description                                |
| ------------------ | ------------------------------------------ |
| `Content`          | A piece of text from the LLM              |
| `ToolCallStart`    | A tool invocation is beginning             |
| `ToolCallDelta`    | Incremental arguments for a tool call      |
| `ToolCallEnd`      | Tool call arguments are complete           |
| `ToolResult`       | Tool execution finished with a result      |
| `StateTransition`  | State machine moved to a new state         |
| `Done`             | Stream is complete                         |
| `Error`            | Something went wrong                       |

---

## Custom Tools

Implement the `Tool` trait to give your agent new capabilities:

```rust
use ai_agents::tools::{Tool, ToolResult};
use async_trait::async_trait;
use serde_json::{json, Value};

struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn id(&self) -> &str { "weather" }
    fn name(&self) -> &str { "Weather Lookup" }
    fn description(&self) -> &str { "Get current weather for a city" }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["city"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let city = args["city"].as_str().unwrap_or("unknown");
        // Call your weather API here...
        ToolResult::ok(format!("72°F and sunny in {}", city))
    }
}
```

Register it on the builder:

```rust
let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .tool(Arc::new(WeatherTool))
    .build()?;
```

`ToolResult` has two constructors:
- `ToolResult::ok("output")` - success
- `ToolResult::error("message")` - failure (the agent sees the error and can retry or explain)

---

## Custom LLM Providers

Implement `LLMProvider` to integrate any backend the framework doesn't support natively:

```rust
use ai_agents::llm::{LLMProvider, LLMResponse, LLMError, LLMConfig, LLMChunk, LLMFeature, ChatMessage};
use async_trait::async_trait;

struct MyProvider;

#[async_trait]
impl LLMProvider for MyProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<LLMResponse, LLMError> {
        // Call your LLM backend here
        todo!()
    }

    async fn complete_stream(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError> {
        // Return a stream of chunks
        todo!()
    }

    fn provider_name(&self) -> &str { "my-provider" }

    fn supports(&self, feature: LLMFeature) -> bool {
        matches!(feature, LLMFeature::Chat | LLMFeature::Streaming)
    }
}
```

Wire it in:

```rust
let agent = AgentBuilder::new()
    .system_prompt("Hello!")
    .llm(Arc::new(MyProvider))
    .build()?;
```

---

## Custom Memory

Implement the `Memory` trait to control how conversation history is stored:

```rust
use ai_agents::llm::ChatMessage;
use ai_agents::memory::Memory;
use ai_agents::error::Result;
use async_trait::async_trait;

struct MyMemory { /* ... */ }

#[async_trait]
impl Memory for MyMemory {
    async fn add_message(&self, message: ChatMessage) -> Result<()> { todo!() }
    async fn get_messages(&self, limit: Option<usize>) -> Result<Vec<ChatMessage>> { todo!() }
    async fn clear(&self) -> Result<()> { todo!() }
    fn len(&self) -> usize { todo!() }
    async fn restore(&self, snapshot: ai_agents::memory::MemorySnapshot) -> Result<()> { todo!() }
}
```

Register it:

```rust
let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .memory(Arc::new(MyMemory::new()))
    .build()?;
```

The framework ships with `InMemoryStore` (simple ring buffer) and `CompactingMemory` (LLM-based summarization with token budgets).

---

## Custom HITL (Human-in-the-Loop)

Implement `ApprovalHandler` to intercept tool calls or state transitions that need human approval:

```rust
use ai_agents::hitl::{ApprovalHandler, ApprovalRequest, ApprovalResult};
use async_trait::async_trait;

struct SlackApprover;

#[async_trait]
impl ApprovalHandler for SlackApprover {
    async fn request_approval(&self, request: ApprovalRequest) -> ApprovalResult {
        // Post to Slack, wait for response...
        println!("Approval needed: {}", request.message);
        ApprovalResult::Approved
    }
}
```

Register it:

```rust
let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .approval_handler(Arc::new(SlackApprover))
    .build()?;
```

For simple cases, use the helper functions instead of implementing the trait:

```rust
use ai_agents::hitl::{create_handler, ApprovalResult};

let handler = create_handler(|request| {
    println!("Tool: {:?}", request.trigger);
    ApprovalResult::Approved
});
```

---

## Agent Hooks

Implement `AgentHooks` to observe lifecycle events - logging, metrics, debugging:

```rust
use ai_agents::hooks::AgentHooks;
use ai_agents::llm::{ChatMessage, LLMResponse};
use async_trait::async_trait;
use serde_json::Value;

struct MetricsHooks;

#[async_trait]
impl AgentHooks for MetricsHooks {
    async fn on_message_received(&self, message: &str) {
        println!("[metrics] User message: {} chars", message.len());
    }

    async fn on_llm_complete(&self, _response: &LLMResponse, duration_ms: u64) {
        println!("[metrics] LLM responded in {}ms", duration_ms);
    }

    async fn on_tool_start(&self, tool: &str, _args: &Value) {
        println!("[metrics] Tool {} starting", tool);
    }

    async fn on_state_transition(&self, from: Option<&str>, to: &str, reason: &str) {
        println!("[metrics] State: {:?} → {} ({})", from, to, reason);
    }

    async fn on_error(&self, error: &ai_agents::error::AgentError) {
        eprintln!("[metrics] Error: {}", error);
    }
}
```

All hook methods have default no-op implementations, so you only override the ones you care about.

Register hooks:

```rust
let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .hooks(Arc::new(MetricsHooks))
    .build()?;
```

The framework also provides `LoggingHooks` (uses `tracing`) and `CompositeHooks` (combines multiple hooks):

```rust
use ai_agents::hooks::{LoggingHooks, CompositeHooks};

let hooks = CompositeHooks::new()
    .add(Arc::new(LoggingHooks::new()))
    .add(Arc::new(MetricsHooks));

let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .hooks(Arc::new(hooks))
    .build()?;
```

### Available hook events

| Method                      | Fires when                                  |
| --------------------------- | ------------------------------------------- |
| `on_message_received`       | User sends a message                        |
| `on_llm_start`              | LLM request is about to be sent             |
| `on_llm_complete`           | LLM response received (with timing)         |
| `on_tool_start`             | Tool execution begins                       |
| `on_tool_complete`          | Tool execution finishes (with timing)        |
| `on_state_transition`       | State machine changes state                  |
| `on_error`                  | An error occurred                            |
| `on_response`               | Final response is ready                      |
| `on_approval_requested`     | HITL approval is needed                      |
| `on_approval_result`        | HITL decision was made                       |
| `on_memory_compress`        | Memory compression triggered                 |
| `on_memory_evict`           | Messages evicted from memory                 |
| `on_memory_budget_warning`  | Token budget threshold exceeded              |
| `on_delegate_start`         | Orchestration: delegation to a registry agent begins |
| `on_delegate_complete`      | Orchestration: delegation completes (with timing)    |
| `on_concurrent_complete`    | Orchestration: parallel agent execution completes    |
| `on_group_chat_round`       | Orchestration: a group chat round finishes            |
| `on_pipeline_stage`         | Orchestration: a pipeline stage completes             |
| `on_pipeline_complete`      | Orchestration: full pipeline execution completes      |
| `on_handoff_start`          | Orchestration: a handoff chain begins                 |
| `on_handoff`                | Orchestration: an agent-to-agent handoff occurs       |
| `on_persona_evolve`         | Persona: a persona field was mutated via `evolve()`   |
| `on_secret_revealed`        | Persona: a secret's reveal conditions were satisfied for the first time |
| `on_facts_extracted`        | Facts: new facts were extracted from a conversation turn |
| `on_actor_memory_loaded`    | Facts: actor facts were loaded from storage at session start |
| `on_session_created`        | Session: a new session was created with metadata      |
| `on_sessions_expired`       | Session: expired sessions were cleaned up via TTL     |

---

## Runtime Context

Inject dynamic key-value data that the agent can access during conversations:

```rust
use serde_json::json;

let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .build()?;

// Set context values
agent.set_context("user_name", json!("Alice"))?;
agent.set_context("subscription", json!("premium"))?;

// Update a nested path
agent.update_context("user.preferences.theme", json!("dark"))?;

// Read all context
let ctx = agent.get_context();
println!("{:?}", ctx);

// Refresh a context source (for dynamic/HTTP providers)
agent.refresh_context("pricing").await?;
```

Context values are available to the agent's system prompt via template rendering and to tools during execution.

---

## Session Persistence

Save and restore full agent state - conversation history, state machine position, and context:

```rust
// Save the current session
agent.save_session("session-abc-123").await?;

// Later, load it back
let found = agent.load_session("session-abc-123").await?;
if found {
    println!("Session restored!");
}

// List all saved sessions
let sessions = agent.list_sessions().await?;

// Delete a session
agent.delete_session("session-abc-123").await?;
```

---

## Actor Memory & Key Facts

Track facts about each actor across sessions. Facts are extracted automatically after each turn and injected into the system prompt on the next session.

```rust
// Set the current actor ID (user, player, other agent).
agent.set_actor_id("customer_42")?;
// Convenience alias:
agent.set_user_id("customer_42")?;

// Load previously stored facts for this actor from storage.
agent.load_actor_memory().await?;

// Read the current actor ID.
let actor = agent.actor_id(); // Option<String>

// Read cached facts (loaded from storage or extracted this session).
let facts = agent.actor_facts(); // Vec<KeyFact>

// Manually extract facts from the last N messages.
let new_facts = agent.extract_facts(10).await?;

// Access the FactStore for direct manipulation.
if let Some(store) = agent.fact_store() {
    let all = store.get_facts("customer_42").await?;
}

// Privacy-aware deletion. Returns Err when memory.actor_memory.privacy.allow_deletion is false.
agent.delete_actor_data("customer_42").await?;

// Session metadata APIs.
let meta = agent.session_metadata();              // current SessionMetadata
agent.set_session_metadata(meta);                 // overwrite tags, ttl, custom

// TTL cleanup and filtered listings (sqlite backend).
let removed = agent.cleanup_expired_sessions().await?;
let filter = ai_agents::facts::SessionFilter {
    actor_id: Some("customer_42".to_string()),
    tags: None,
    agent_id: None,
    created_after: None,
    created_before: None,
    limit: Some(10),
};
let summaries = agent.list_sessions_filtered(&filter).await?;
```

When `auto_extract: true` (the default), extraction runs after every turn - no manual calls needed. Configure via `memory.facts` and `memory.actor_memory` in YAML. See [YAML Reference](@/docs/yaml-reference.md#facts-key-facts-extraction) for the full schema.

Switching actors mid-session via `set_actor_id()` (or via `from_context` resolution when the configured context path changes) clears the cached facts and reloads on the next turn, so prompt injection always reflects the current actor.

Session persistence requires a storage backend. Enable one via feature flags:

```toml
# SQLite (file-based, good for single-server)
ai-agents = { version = "1.0.0-rc.11", features = ["sqlite"] }

# Redis (networked, good for distributed setups)
ai-agents = { version = "1.0.0-rc.11", features = ["redis-storage"] }
```

Configure storage in your YAML:

```yaml
storage:
  type: sqlite
  path: "./sessions.db"
```

Or for Redis:

```yaml
storage:
  type: redis
  url: "redis://localhost:6379"
```

You can also use the lower-level API with any `AgentStorage` implementation:

```rust
use ai_agents::persistence::create_storage;

let storage = create_storage(&storage_config).await?;
agent.save_to(storage.as_ref(), "my-session").await?;
agent.load_from(storage.as_ref(), "my-session").await?;
```

---

## Full API Reference

This page covers the most common patterns. For the complete API - every struct, enum, trait, and function - see the auto-generated docs:

📖 **[docs.rs/ai-agents](https://docs.rs/ai-agents/1.0.0-rc.11)**

---

## Next Steps

- **[Getting Started](@/docs/getting-started.md)** - quick install and first agent
- **[CLI Guide](@/docs/cli.md)** - run agents from the command line
- **[LLM Providers](@/docs/providers.md)** - setup for all 12 supported providers
- **[YAML Reference](@/docs/yaml-reference.md)** - the complete agent spec

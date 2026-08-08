+++
title = "Rust API"
weight = 5
template = "docs.html"
description = "Embedding AI Agents in your Rust application."
+++

Use `ai-agents` as a library to build, configure, and run agents entirely from Rust. Everything the CLI can do, you can do programmatically - plus custom tools, providers, memory backends, hooks, and more.

---

## Adding the Dependency

Add `ai-agents` to your `Cargo.toml`:

```toml
[dependencies]
ai-agents = "1.0"
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
ai-agents = { version = "1.0", features = ["full"] }
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
        .auto_configure_mcp().await?
        .auto_configure_spawner().await?
        .build()?;

    let response = agent.chat("Hello!").await?;
    println!("{}", response.content);
    Ok(())
}
```

- `auto_configure_llms()` reads the `llm:` and `llms:` blocks from the spec, resolves API keys from environment variables, and registers all providers automatically.
- `auto_configure_features()` wires up error recovery, tool security, process pipeline, and built-in tools from the spec.
- `auto_configure_mcp()` connects to MCP servers declared in the `tools` list (entries with `type: mcp`), discovers their functions, and registers them.
- `auto_configure_spawner()` reads the `spawner:` section, creates the spawner and registry, creates optional shared storage, resolves templates, performs fail-closed auto-spawn and child storage readiness, and registers explicitly configured spawner tools.

This is the same builder chain used by the CLI. `auto_configure_mcp()` and `auto_configure_spawner()` are no-ops when the relevant YAML sections are absent, so it is safe to always include them.

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
          model: gpt-5.4-nano
    "#;

    let agent = AgentBuilder::from_yaml(yaml)?
        .auto_configure_llms()?
        .auto_configure_features()?
        .auto_configure_mcp().await?
        .auto_configure_spawner().await?
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
    let llm = UnifiedLLMProvider::from_env(ProviderType::OpenAI, "gpt-5.4-nano")?;

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

The full builder chain used by the CLI is: `auto_configure_llms` &#x2192; `auto_configure_features` &#x2192; `auto_configure_mcp` &#x2192; `auto_configure_spawner` &#x2192; `build`. The MCP and spawner steps must come after `auto_configure_features()` so the tool registry exists. To keep auto-registered built-ins, call `.tool()` or `.extend_tools()` after `auto_configure_features()`; calling either method first creates a custom registry and causes builtin auto-registration to be skipped. `.tools(registry)` intentionally replaces the entire registry, including previously auto-registered built-ins. Configuration-only overrides such as `.hooks()` and `.system_prompt()` can be applied anywhere before `.build()`.

Use one of these orderings:

- Extend built-ins: load YAML -> configure LLMs -> configure features -> configure MCP -> configure spawner -> add `.tool()` or `.extend_tools()` -> build.
- Supply a complete custom registry: load YAML -> configure LLMs -> call `.tools(registry)` -> configure features -> configure MCP -> configure spawner -> build. Calling `.tools(registry)` after MCP or spawner configuration removes those generated registrations, so only do that when full replacement is intentional and the final YAML grants still resolve.

Registration and availability are separate for ordinary tools. `auto_configure_features()`, `auto_configure_mcp()`, and `auto_configure_spawner()` register tools, but YAML top-level `tools:` decides what the model can call. When loading YAML, omitted top-level `tools:` means no ordinary LLM-callable tools even if Rust registered them. Explicit YAML feature flags such as `spawner.management_tools`, `spawner.orchestration_tools`, and `persona.evolution.allow_llm_evolve` are exceptions because they intentionally register and grant their generated tools. In pure Rust builder flows without YAML, registered tools are treated as the explicit grant.

Spawner restrictions apply equally to YAML, spec, template, auto-spawn, and restore paths. Child IDs use a bounded portable ASCII grammar, active nested child spawners are rejected, allowlist violations reject the complete child, and `max_agents` counts both registered and in-flight spawner-managed children. `shared_llms: true` requires the parent registry to contain every alias referenced by each child; it does not construct child-local providers. Hosts using lower-level detached `SpawnedAgent::from_runtime()` records or direct registry insertion operate outside the spawner capacity counter. A direct `spawn_*` call returns an unregistered child, so the host must register or discard it. Complete topology persistence requires a full parent snapshot and matching child sessions under the same session ID.

---

## Basic Chat

The simplest interaction - send a message, get a response:

```rust
use ai_agents::{Agent, AgentBuilder};

let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .auto_configure_mcp().await?
    .auto_configure_spawner().await?
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
    .auto_configure_mcp().await?
    .auto_configure_spawner().await?
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

When the selected LLM has explicit `tool_choice`, the runtime buffers that provider decision before emitting committed content or the existing `ToolCallStart`, `ToolCallEnd`, and `ToolResult` events. v1 does not expose provider-native incremental tool-call deltas as a separate API; agents that omit `tool_choice` keep the existing streaming path. `StreamChunk` also does not carry the final blocking `AgentResponse.metadata`; use blocking `chat()` when that metadata is part of the contract.

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

## Runtime Optimization Types

Most runtime optimization behavior is configured in YAML, but policy types are re-exported for Rust hosts that build specs or inspect configuration programmatically.

```rust
use ai_agents::{
    RuntimeConfig, RuntimeOptimizationConfig, RuntimeOptimizationKind,
    RuntimeBranchResult, RuntimeCommitBehavior, RuntimeTaskPurpose,
    StreamingOptimizationPolicy, TransitionTiming,
};

let mut runtime = RuntimeConfig::default();
runtime.optimization.enabled = true;
runtime.optimization.max_speculative_llm_calls_per_turn = 4;
runtime.optimization.speculative_state_transitions = true;
runtime.optimization.streaming_policy = StreamingOptimizationPolicy::BufferUntilRoutingDone;
```

Branch observability labels use `RuntimeOptimizationKind` and `RuntimeCommitBehavior` internally. Public users normally read those labels from observability reports rather than constructing branches directly. `RuntimeBranchResult` covers scheduler-managed blocking branch outputs, while buffered streaming uses `StreamingDraftResult` through its streaming-specific safety path. Buffered streaming with `BufferUntilRoutingDone` currently applies to response-independent parallel state-transition routing. Its buffer limit protects unresolved routing; after routing misses or fails, later chunks are collected for the committed output without consuming that unresolved-route buffer.

---

## Custom Tools

`Tool::id()` is the canonical ID used by YAML `tools:`, `tool_security.tools`, `hitl.tools`, eval evidence, recovery policy, and aliases. Display names and localized aliases may be shown to the model, but runtime policy resolves them back to the canonical ID before execution.

Implement the `Tool` trait to give your agent new capabilities:

```rust
use ai_agents::tools::{
    ResultLimitBinding, ResultLimitKind, Tool, ToolExecutionContext, ToolPolicyBindings,
    ToolResult,
};
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

    fn policy_bindings(&self) -> ToolPolicyBindings {
        ToolPolicyBindings {
            result_limit_fields: vec![ResultLimitBinding::new(
                "max_results",
                ResultLimitKind::MaxResults,
            )],
            ..Default::default()
        }
    }

    async fn execute(&self, args: Value, ctx: ToolExecutionContext) -> ToolResult {
        let city = args["city"].as_str().unwrap_or("unknown");
        let backend = ctx.custom_config.get("backend").and_then(Value::as_str).unwrap_or("mock");
        let max_results = ctx.limits.max_results.unwrap_or(1);
        ToolResult::ok(json!({
            "city": city,
            "backend": backend,
            "max_results": max_results,
            "forecast": "72°F and sunny"
        }).to_string())
    }
}
```

Register it on the builder:

```rust
let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .auto_configure_mcp().await?
    .auto_configure_spawner().await?
    .tool(Arc::new(WeatherTool))
    .build()?;
```

`ToolResult` has two constructors:
- `ToolResult::ok("output")` - success
- `ToolResult::error("message")` - failure (the agent sees the error and can retry or explain)

For a runnable version of this pattern, see `examples/rust/custom-tools/src/context_tool.rs` and `examples/rust/custom-tools/agents/context_tool_agent.yaml`. The example shows `max_results` flowing through `ctx.limits` and `backend`, `tenant`, and related custom settings flowing through `ctx.custom_config`.

For tools that need runtime safety or scheduling hints, override the optional metadata methods too:

```rust
use ai_agents::tools::{
    ToolCallClassification, ToolOperationKind, ToolSafetyMetadata, ToolSideEffectLevel,
};

fn safety_metadata(&self) -> ToolSafetyMetadata {
    ToolSafetyMetadata {
        read_only: true,
        concurrency_safe: true,
        operation: ToolOperationKind::Read,
        side_effect_level: ToolSideEffectLevel::None,
        requires_network: false,
        destructive: false,
        open_world: false,
        host_dependent: false,
        requires_user_interaction: false,
        supports_cancellation: true,
        default_requires_approval: false,
        should_defer_schema: false,
        max_output_chars: Some(20_000),
        max_result_size_chars: Some(20_000),
    }
}

fn classify_call(&self, args: &Value) -> ToolCallClassification {
    ToolCallClassification::from_metadata(&self.safety_metadata())
}
```

The shared runtime resolves names to canonical IDs, checks scope and `tool_security`, and applies HITL when needed. After approval it takes a fresh runtime-control snapshot, resolves the final tool once, reapplies policy caps, recomputes classification and resource keys, and verifies current scope, emergency control, provider availability, policy, confirmation requirement, and final arguments before lock acquisition. State-scope evaluation records the state generation that authorized the call. After waiting for locks, a changed runtime, policy, or state generation fails closed and one atomic admission records the rate-limited call immediately before invocation. The same resolved tool object is executed. Resource guards cover admission and the tool side effect, then release before completion hooks or fallback re-enters the shared path. YAML skills, state actions, plans, fallback, orchestration, generated tools, and spawned runtimes use the same path through `ToolInvoker`, so tool calls do not bypass runtime policy. Direct `SkillExecutor::execute` is prompt-only; use `execute_with_invoker` for skills that contain tool steps.

Custom tools should use `ctx.limits` for effective framework caps and `ctx.custom_config` for tool-specific settings from `tool_security.tools.<tool_id>.config`. Tool input arguments are model-callable schema fields; `config` is host-supplied and not model-callable. Do not parse `tool_security` YAML directly inside the tool. If `fail_closed: true` and a path, domain, command, operation, or result-limit policy is configured and cannot be enforced by the shared executor alone, custom tools must declare matching `policy_bindings()` or execution is denied before the implementation runs. Side-effecting tools should also set call classification carefully: use `requires_approval` for risky mutation or command calls and set `safely_retryable` only when repeating the exact call cannot duplicate side effects. Declare every path field, including source and destination fields, so the runtime records normalized resources and places non-concurrency-safe path operations under the shared path-mutation lock. Non-concurrency-safe calls without a concrete resource use the shared unbound-side-effect lock.

Host-backed built-in hooks are installed on the built runtime:

```rust
use ai_agents::tools::{DiagnosticsProvider, QuestionHandler, WebSearchProvider};
use std::sync::Arc;

let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .build()?;

agent.set_question_handler(Some(my_question_handler as Arc<dyn QuestionHandler>));
agent.set_diagnostics_provider(my_diagnostics_provider as Arc<dyn DiagnosticsProvider>);
agent.set_command_runner(Arc::new(ai_agents::tools::ProcessCommandRunner));
agent.set_web_search_provider(my_search_provider as Arc<dyn WebSearchProvider>);

let todos = agent.todos();
let control = agent.runtime_control();
```

`set_question_handler()` powers `ask_user`, `set_diagnostics_provider()` powers `diagnostics`, `set_command_runner()` powers `command`, `set_web_search_provider()` powers `web_search`, and `todos()` returns the current runtime-local task list. A missing diagnostics, command, or search provider is rejected before tool invocation and recorded as unavailable; `ask_user` instead executes its structured default/unavailable fallback. `ProcessCommandRunner` runs argv without a shell, starts from an empty environment, applies policy-filtered env values, bounds stdout/stderr while reading, cleans up on timeout, and redacts sensitive argv values in evidence. The runtime-control handle can update tool security with fallible `try_set_tool_security()` or compatibility `set_tool_security()`, narrow the runtime's declared tool grant with `set_tool_scope()`, clear those overrides, enable emergency denial with `set_emergency_deny()`, or call `cancel_all()` for future tool calls. Prefer `try_set_tool_security()` for host-supplied policy because invalid positive-only limits return an error before the active policy or runtime-control generation changes; `set_tool_security()` preserves its existing return type and panics on invalid host policy. Runtime scope entries are canonicalized and intersected with the declared grant, so an override cannot grant a registered tool that the runtime did not already own. Runtime controls are local to each runtime and do not automatically propagate from a parent to spawned children.

`web_fetch` prompt extraction also uses the runtime LLM registry when a router or default model is available, so nested extraction calls flow through the normal observed provider path. The built-in `web_search` is separate from provider-native LLM search options: it requires an explicit tool grant, a host-installed `WebSearchProvider`, and shared-executor evidence.

### Direct subcrate boundaries

The curated `ai-agents` facade exports normal host integration for questions, diagnostics, commands, and web search, including `WebSearchProvider` request and response types. Add a matching direct `ai-agents-tools` dependency only for lower-level custom web-fetch transport or resolver injection (`WebFetchTransport`, `WebFetchResolver`, and their request/response types) or eval-oriented `StaticWebSearchProvider` and `UnavailableWebSearchProvider` helpers. A custom web-fetch transport must return one HTTP response per call without automatically following redirects so `WebFetchTool` can validate every hop. A socket-opening implementation must override validated sending, connect only to the supplied approved addresses, and honor `max_response_bytes` while reading each response; the compatibility default cannot enforce a host transport's DNS, proxy, or egress behavior.

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
        matches!(feature, LLMFeature::SystemMessages | LLMFeature::Streaming)
    }
}
```

Existing custom providers compile without implementing tool-specific methods. They use the prompt protocol when an agent configures explicit tool choice. A custom provider can opt into native selection by implementing `complete_with_tools()`, returning `true` from `supports_tool_choice()`, and storing normalized calls in `LLMResponse` with `set_tool_calls()` or `with_tool_calls()`. The request uses `LLMToolRequest`, `LLMToolDefinition`, and `ToolChoice`; native calls are still executed only by the runtime's shared executor. `configured_tool_choice()` is optional and returns no override by default.

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
    .auto_configure_mcp().await?
    .auto_configure_spawner().await?
    .memory(Arc::new(MyMemory::new()))
    .build()?;
```

The framework ships with `InMemoryStore` (simple ring buffer) and `CompactingMemory` (serialized LLM-based summarization with token budgets and protected recent-message retention). `MemorySnapshot` includes `messages`, optional `summary`, and a defaulted `summarized_count`; the default keeps snapshots written before this field backward-compatible.

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
    .auto_configure_mcp().await?
    .auto_configure_spawner().await?
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
    .auto_configure_mcp().await?
    .auto_configure_spawner().await?
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
    .auto_configure_mcp().await?
    .auto_configure_spawner().await?
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
| `on_relationship_loaded`    | Relationship: actor relationship was loaded or created |
| `on_relationship_change`    | Relationship: one or more relationship dimensions changed |
| `on_notable_event`          | Relationship: a notable relationship event was stored |

---

## Runtime Context

Inject dynamic key-value data that the agent can access during conversations:

```rust
use serde_json::json;

let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .auto_configure_mcp().await?
    .auto_configure_spawner().await?
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

Storage methods advertise support through `AgentStorage::supports(StorageCapability)`. Unsupported extended operations return `AgentError::UnsupportedStorageCapability`.

| Backend | Snapshots | Metadata and filtering | Expiry cleanup | Facts and relationships | Atomic actor deletion |
|---------|-----------|------------------------|----------------|-------------------------|-----------------------|
| File | Yes | No | No | No | No |
| Redis | Yes | No | No | No | No |
| SQLite | Yes | Yes | Yes | Yes | Yes |
| `NoopStorage` | No | No | No | No | No |

Redis's `storage.ttl_seconds` expires Redis snapshot keys directly; it does not implement generic session metadata or `ExpiryCleanup`. `NamespacedStorage` derives snapshot, metadata, filtering, facts, relationships, and actor-deletion support from its inner backend, but intentionally does not forward backend-global expiry cleanup.

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

When `auto_extract: true` (the default), extraction runs after every turn - no manual calls needed. Durable facts require `StorageCapability::ActorFacts`, and privacy deletion requires `StorageCapability::ActorDataDeletion`; SQLite is currently the only built-in backend providing both. Configure via `memory.facts` and `memory.actor_memory` in YAML. See [YAML Reference](@/docs/yaml-reference.md#facts-key-facts-extraction) for the full schema.

Switching actors mid-session via `set_actor_id()` (or via `from_context` resolution when the configured context path changes) clears the cached facts and reloads on the next turn, so prompt injection always reflects the current actor.

---

## Relationship Memory

Relationship memory tracks the agent's stance toward each actor. Use `relationship_manager()` for direct inspection or manual updates.

```rust
// Relationship memory uses the same actor ID as actor facts.
agent.set_actor_id("customer_42")?;

// For one turn, carry actor context structurally without changing global actor state.
let response = agent
    .chat_with_actor_context(
        "Please help with this issue",
        ai_agents::TurnActorContext::new()
            .with_origin_actor("customer_42")
            .with_sender_agent("coordinator"),
    )
    .await?;

// Access the manager when memory.relationships is enabled in YAML.
if let Some(manager) = agent.relationship_manager() {
    let relationship = manager.get_or_create("customer_42", Some("Jane"));
    println!("trust = {:?}", relationship.dimensions.get("trust"));

    // Manual update for application-driven signals.
    let change = agent
        .update_relationship_dimension("trust", 0.1, Some("Customer verified identity successfully"))
        .await?;
    println!("{} changed by {}", change.dimension, change.delta);

    // Two-sided configs can update a specific perspective.
    let perceived = agent
        .update_relationship_dimension_for_perspective(
            ai_agents::relationships::RelationshipPerspective::PerceivedActorToAgent,
            "trust",
            0.1,
            Some("Customer expressed confidence in the agent"),
        )
        .await?;
    println!("{} {} changed by {}", perceived.perspective, perceived.dimension, perceived.delta);
}
```

Automatic updates run after successful turns when `memory.relationships.auto_update.enabled` is true. Relationship context is injected at `relationships.current_actor.*`, and formatted prompt text is available as `{{ relationship_memory }}`.

Persistent relationship memory requires `StorageCapability::ActorRelationships`, currently provided only by SQLite among the built-in backends. Enable it with the `sqlite` feature:

```toml
ai-agents = { version = "1.0", features = ["sqlite"] }
```

Configure storage in your YAML:

```yaml
storage:
  type: sqlite
  path: "./sessions.db"
```

Redis can persist session snapshots but cannot persist dedicated actor relationships through the generic storage trait.

You can also use the lower-level API with any `AgentStorage` implementation:

```rust
use ai_agents::persistence::create_storage;

let storage = create_storage(&storage_config).await?;
agent.save_to(storage.as_ref(), "my-session").await?;
agent.load_from(storage.as_ref(), "my-session").await?;
```

---

## Evaluation Runner

Rust hosts can invoke the same evaluation runner used by `ai-agents-cli eval`. This is useful for embedding scenario checks into custom CI tools or release checks.

```rust
use ai_agents::eval::{EvalRunner, EvalRunnerOptions, write_outputs};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let output = PathBuf::from("target/eval/mocked/basic/simple_chat_mocked");
    let options = EvalRunnerOptions {
        agent: Some(PathBuf::from("examples/yaml/basic/simple_chat.yaml")),
        output: output.clone(),
        ..Default::default()
    };

    let runner = EvalRunner::from_file("examples/eval/mocked/basic/simple_chat_mocked.yaml", options)?;
    let result = runner.run().await?;
    write_outputs(&result, &output, true)?;

    assert_eq!(result.failed, 0);
    Ok(())
}
```

Core types are re-exported under `ai_agents::eval`: `EvalSuite`, `EvalSettings`, `ScenarioBudget`, `EvalRunner`, `EvalRunnerOptions`, `EvalResult`, `ScenarioResult`, `ScenarioStatus`, `FixturesConfig`, `TurnEvidence`, `ToolExecutionRecord`, `LLMJudge`, `JudgeConfig`, `ResetOptions`, and `ResetProfile`.

The runner builds agents through `AgentBuilder`, so the same YAML features used by the CLI are available. Eval fixtures can replace LLMs and tools for deterministic tests, and reports can be written with `write_outputs()`.

Rust options mirror the CLI: `parallel` enables scenario-concurrent runs when isolation allows it, `observability` attaches a safe default overlay, `llm_mode` and `cassette` can force record/replay/real LLM behavior, and streaming turns are collected before assertions run. Default JSON outputs redact input, response, and string assertion details while omitting raw turn evidence and response metadata.

---

## Observability

When `observability.enabled: true` is present in YAML, the builder attaches an `ObservabilityManager` to the runtime. You can read aggregate reports, inspect metrics, export configured files from Rust, or provide your own shared manager when multiple agents should report into one trace and metrics window.

```rust
use ai_agents::{Agent, AgentBuilder, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let agent = AgentBuilder::from_yaml_file("agent.yaml")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .auto_configure_mcp().await?
        .auto_configure_spawner().await?
        .build()?;

    agent.chat("Hello").await?;

    if let Some(obs) = agent.observability() {
        let report = obs.generate_report();
        println!("LLM calls: {}", report.summary.total_llm_calls);
        println!("Tokens: {}", report.summary.total_tokens);
        println!("Cost: ${:.6}", report.summary.total_cost_usd);
        obs.export().await?;
    }

    Ok(())
}
```

You can also provide a manager programmatically when building an agent:

```rust
use ai_agents::observability::{ObservabilityConfig, ObservabilityManager};
use std::path::Path;
use std::sync::Arc;

let mut config = ObservabilityConfig::default();
config.enabled = true;
config.export.write_report = true;

// YAML-loaded agents resolve observability.cost.pricing_file automatically.
// Rust-provided configs should call this helper when pricing_file is set.
let config = config.with_pricing_file_loaded(Some(Path::new(".")))?;
let manager = ObservabilityManager::new(config);

let agent = AgentBuilder::new()
    .system_prompt("You are helpful.")
    .llm(llm)
    .observability(Arc::clone(&manager))
    .build()?;
```

`ObservabilityConfig::with_pricing_file_loaded()` accepts JSON or YAML maps shaped like `cost.pricing`. Inline `cost.pricing` values override entries loaded from the file. When `base_dir` is `Some`, relative paths are resolved from that directory; when it is `None`, they resolve from the process working directory.

`ObservabilityManager::generate_report()` drains pending events before returning an `ObservabilityReport`. Use `report.by_purpose` to compare operation costs such as `main_response`, `process_detect`, `disambiguation_clarification`, `facts_extraction`, or `orchestration_aggregation`. `ObservabilityManager::export()` writes the configured report, CSV aggregate, JSONL or JSON raw events, and Prometheus text files.

`ObservabilityManager::render_prometheus()` returns Prometheus text exposition format as a string. The framework does not start a scrape server for you yet. To connect Prometheus in a Rust host, expose that string from your own `/metrics` HTTP route or write it to a `.prom` file for the node_exporter textfile collector.

Key exported types live under `ai_agents::observability`: `ObservabilityConfig`, `ObservabilityManager`, `ObservabilityReport`, `AggregatedMetrics`, `ObservationEvent`, `ObservationPurpose`, `SpanContext`, `ObservedLLMProvider`, `ObservedTool`, `ObservabilityHooks`, `with_observation_context`, `with_observation_purpose`, and `with_updated_observation_context()`.

### Extending Observability

Rust hosts can extend observability at a few different layers:

| Layer | Use when | Types |
|-------|----------|-------|
| LLM wrapper | You want an LLM provider measured outside the normal builder path | `ObservedLLMProvider` |
| Tool wrapper | You want a tool measured outside the normal builder path | `ObservedTool` |
| Lifecycle hooks | You want state, HITL, memory, persona, facts, relationship, or orchestration events | `ObservabilityHooks` with `CompositeHooks` |
| Task-local context | You spawn tasks or run nested work and need trace continuity | `with_observation_context`, `with_observation_purpose`, `SpanContext` |
| Shared manager | You want multiple agents to report into one metrics window | `AgentBuilder::observability(manager)` |

Some internal feature crates avoid depending on `ai-agents-observability` directly to prevent dependency cycles. For those cases, the runtime uses small observer adapter traits, such as process-stage and clarification observers, then implements those observers with observability in `ai-agents-runtime`.

---

## Full API Reference

This page covers the most common patterns. For the complete API - every struct, enum, trait, and function - see the auto-generated docs:

📖 **[docs.rs/ai-agents](https://docs.rs/ai-agents/latest/ai_agents/)**

---

## Next Steps

- **[Getting Started](@/docs/getting-started.md)** - quick install and first agent
- **[CLI Guide](@/docs/cli.md)** - run agents from the command line
- **[LLM Providers](@/docs/providers.md)** - setup for all 12 supported providers
- **[YAML Reference](@/docs/yaml-reference.md)** - the complete agent spec
- **[Built-in Tools](@/docs/built-in-tools.md)** - built-in schemas, outputs, policy, and host requirements

# AI Agents Framework

**One YAML = Any Agent.**

A Rust framework for building AI agents from a single YAML specification. No code required for common use cases.

- Declarative behavior -- everything in YAML, not code
- Language-agnostic semantics -- intent, extraction, validation via LLM (no regex)
- Layered overrides -- global -> agent -> state -> skill -> turn
- Safety by default -- tool policies, HITL approvals, error recovery
- Extensible -- custom LLMs, tools, memory, storage, hooks

> Status: **1.0.0-rc.5** | 17 crates | 681 tests
>
> Under active development. APIs and YAML schema may change between minor versions.
> Documentation and more examples are coming.

## Features

### Agent Core
- YAML-defined agents -- system prompt, tools, skills, states, memory, and behavior in one file
- Multi-LLM support -- multiple providers with aliases (default, router, evaluator); auto-fallback on failure
- Skill system -- reusable "tool + prompt" workflows with LLM-based intent routing
- Input/output process -- declarative pipeline: normalize, detect, extract, sanitize, validate, transform, format
- Streaming -- real-time token streaming with tool call and state transition events

### State Machine
- Hierarchical states -- nested sub-states with prompt inheritance
- Auto transitions -- LLM-evaluated conditions with guard-based short-circuiting
- Entry/exit actions -- execute tools, skills, prompts, or set context on state change
- Turn timeout -- automatic state transition after max turns

### Context & Memory
- Dynamic context -- runtime, file, HTTP, env, and callback sources with per-turn refresh
- Template rendering -- Jinja2-compatible templates (minijinja) in system prompts
- CompactingMemory -- LLM-based rolling summarization with configurable thresholds
- Token budgeting -- per-component token allocation with overflow strategies
- Persistence -- SQLite, Redis, and file storage backends with session management

### Tools
- Built-in tools -- datetime, JSON, random, HTTP, file, text, template, math, calculator(for dev)
- Conditional availability -- context, state, time, semantic (LLM-based), and composite conditions
- Multi-language aliases -- tool names and descriptions in any language
- Parallel execution -- concurrent tool calls with configurable concurrency

### Safety & Control
- Error recovery -- retry with backoff, LLM fallback, context overflow handling (truncate/summarize)
- Tool security -- rate limiting, domain/path restrictions, confirmation requirements
- Human-in-the-loop -- tool, condition, and state approval with multi-language message support

### Intelligence
- Reasoning modes -- none, chain-of-thought, ReAct, plan-and-execute, auto (LLM selects)
- Reflection -- LLM self-evaluation with criteria, retry on failure, configurable thresholds
- Intent disambiguation -- LLM-based ambiguity detection, clarification generation, multi-turn resolution

### Extensibility
- Agent hooks -- lifecycle events for logging, metrics, monitoring (message, LLM, tool, state, memory, HITL)
- Custom providers -- implement `LLMProvider`, `Memory`, `Tool`, `ApprovalHandler`, `Summarizer` traits

## Roadmap

> Features may be implemented in any order based on priority and need.

| Feature | Description | Status |
|---------|-------------|--------|
| Advanced Memory | CompactingMemory, token budgeting, SQLite/Redis storage | ✅ Done |
| Tool Provider System | ToolProvider trait, multi-language aliases, extensibility | ✅ Done |
| Workspace Refactoring | 17 modular crates for parallel compilation | ✅ Done |
| Reasoning & Reflection | Chain-of-Thought, ReAct, Plan-and-Execute, self-evaluation | ✅ Done |
| Intent Disambiguation | LLM-based ambiguity detection and clarification | ✅ Done |
| Evaluation Framework | Dataset runner, metrics, LLM judge | Planned |
| Observability & Tracing | Per-call latency, token usage, cost tracking | Planned |
| Conversation Scripts | Declarative guided flows with LLM extraction | Planned |
| Custom Reasoning Prompts | Domain/language-specific reasoning instructions | Planned |
| Reasoning Depth Control | Auto shallow/standard/deep with resource limits | Planned |
| Conversation Style Modifiers | Dynamic tone, formality, verbosity adaptation | Planned |
| Session Management | Cross-session user memory, key facts extraction | Planned |
| Multi-Agent Orchestration | Supervisor, pipeline, crew patterns | Planned |
| VectorDB Tool | Embedding-based retrieval for RAG | Planned |
| Background Tasks & Scheduling | Cron-based, event-driven, interval tasks | Planned |
| Knowledge Base / RAG | Document ingestion, chunking, retrieval pipeline | Planned |
| Hot Reload | Live YAML configuration updates without restart | Planned |
| Code Interpreter | Sandboxed Python/JS execution with templates | Planned |
| MCP Integration | Model Context Protocol tool ecosystem | Planned |
| Semantic Caching | Embedding-based response caching | Planned |
| Budget Control | Token/cost limits, LLM switching, cost prediction | Planned |
| A2A Protocol | Agent-to-Agent communication and delegation | Planned |
| Agent Composition | Multi-agent patterns (supervisor, pipeline, debate) | Planned |

## Install

```toml
[dependencies]
ai-agents = "1.0.0-rc.5"
```

## Quick Start

### From YAML

```yaml
# agent.yaml
name: MyAgent
system_prompt: "You are a helpful assistant."
llm:
  provider: openai
  model: gpt-4.1-nano
```

```rust
use ai_agents::{Agent, AgentBuilder};

#[tokio::main]
async fn main() -> ai_agents::Result<()> {
    let agent = AgentBuilder::from_yaml_file("agent.yaml")?
        .auto_configure_llms()?
        .build()?;

    let response = agent.chat("Hello!").await?;
    println!("{}", response.content);
    Ok(())
}
```

### From Rust API

```rust
use ai_agents::{AgentBuilder, UnifiedLLMProvider, ProviderType};
use std::sync::Arc;

#[tokio::main]
async fn main() -> ai_agents::Result<()> {
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

See the [examples/](examples/) directory for more.

## License

Licensed under either of

- Apache License, Version 2.0 (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license (LICENSE-MIT)

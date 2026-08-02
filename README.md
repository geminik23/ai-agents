# AI Agents Framework

[![Crates.io](https://img.shields.io/crates/v/ai-agents?style=flat-square&color=06b6d4)](https://crates.io/crates/ai-agents)
[![docs.rs](https://img.shields.io/docsrs/ai-agents?style=flat-square&label=docs.rs)](https://docs.rs/ai-agents)
[![License](https://img.shields.io/crates/l/ai-agents?style=flat-square)](https://github.com/geminik23/ai-agents)
[![GitHub Stars](https://img.shields.io/github/stars/geminik23/ai-agents?style=flat-square&color=f59e0b)](https://github.com/geminik23/ai-agents)

**One YAML = Any Agent.**

A Rust framework for building AI agents from a single YAML specification. No code required for common use cases.

**[ai-agents.rs](https://ai-agents.rs)** - Documentation, guides, and examples

- Declarative behavior - everything in YAML, not code
- Language-agnostic semantics - intent, extraction, validation via LLM (no regex)
- Layered overrides - global → agent → state → skill → turn
- Safety by default - tool policies, HITL approvals, error recovery
- Extensible - custom LLMs, tools, memory, storage, hooks

> Status: **1.0.0-rc.16** — Under active development. Stable `v1.0.0` is planned for release before August 1, 2026. APIs and YAML schema may change between minor versions.

## Features

- **Multi-LLM with fallback** - 12 providers (OpenAI, Anthropic, Google, Ollama, DeepSeek, Groq, Mistral, Cohere, xAI, Phind, OpenRouter, any OpenAI-compatible); named aliases (default, router); auto-fallback on failure
- **State machine + skills** - hierarchical states, LLM-evaluated transitions, guard-based routing, entry/exit actions, reusable multi-step skills
- **Built-in tools + MCP** - 30 canonical built-in IDs: `calculator`, `echo`, `datetime`, `json`, `random`, `file`, `glob`, `grep`, `file_read`, `file_write`, `file_edit`, `patch`, `copy_path`, `move_path`, `delete_path`, `file_list`, `file_info`, `git_status`, `git_diff`, `diagnostics`, `ask_user`, `todo`, `sleep`, `web_fetch`, `web_search`, `command`, `text`, `template`, `math`, and `http`; connect any MCP server for hundreds more
- **Tool scoping, selection, policy, and context** - explicit top-level grants, state-level narrowing, opt-in `auto`/`required`/`specific`/`none` tool choice, policy bindings, review modes for filesystem mutation, post-approval final authorization, atomic rate admission, conservative mutation locking, read-before-write guards, command allowlists, and context-aware custom tools
- **Input/output process pipeline** - deterministic normalization and formatting plus optional LLM-backed detection, extraction, sanitization, validation, and transformation
- **Dynamic context** - runtime, file, HTTP, env, and callback sources with Jinja2 templates in prompts
- **Memory stack** - CompactingMemory and token budgeting; file and Redis snapshot persistence; SQLite snapshots, session metadata/filtering/cleanup, actor facts, and relationship memory
- **Agent persona** - structured identity, traits, goals, secrets, evolution, and reusable templates
- **Multi-agent systems** - fail-closed child admission, bounded managed capacity, optional shared LLMs and namespaced storage, agent registry, actor-aware messaging, and router/pipeline/concurrent/group chat/handoff orchestration; active nested spawners are rejected in v1
- **CLI + TUI** - interactive REPL, ratatui terminal UI, streaming, context injection, and actor/relationship inspection
- **Reasoning, reflection & disambiguation** - chain-of-thought, ReAct, plan-and-execute, self-evaluation, ambiguity detection, and clarification
- **Evaluation, safety, control & observability** - YAML scenario evals with assertions/judges, runtime latency optimization, speculative branch execution, error recovery with backoff, tool security, HITL approvals, and privacy-safe latency/token/cost tracing with JSON/CSV/Prometheus exports
- **Extensible via traits** - `LLMProvider`, `Memory`, `Tool`, `ApprovalHandler`, `Summarizer`, `AgentHooks`, `ToolProvider`; custom LLM providers remain source-compatible and can opt into native tool requests through additive methods

See [Concepts](https://ai-agents.rs/docs/concepts/) for architecture details and [Providers](https://ai-agents.rs/docs/providers/) for per-provider setup.

## Install

```toml
[dependencies]
ai-agents = "1.0.0-rc.16"
```

## Quick Start

### From CLI (no Rust code needed)

Create `agent.yaml`:

```yaml
# agent.yaml
name: MyAgent
system_prompt: "You are a helpful assistant."
llm:
  provider: openai
  model: gpt-5.4-nano

# For any OpenAI-compatible server:
# llm:
#   provider: openai-compatible
#   model: qwen3:8b
#   base_url: http://localhost:11434/v1

# Fixed framework fields are strict; provider-specific extra params are allowed inside llm configs.
# Example for OpenAI reasoning-capable models:
# llms:
#   default:
#     provider: openai
#     model: gpt-5.4-mini
#     reasoning_effort: low
# llm:
#   default: default
```

Run it:

```sh
cargo run -p ai-agents-cli -- run agent.yaml
```

### From YAML + Rust

```rust
use ai_agents::{Agent, AgentBuilder};

#[tokio::main]
async fn main() -> ai_agents::Result<()> {
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

This is the same builder chain used by the CLI. `auto_configure_mcp()` and `auto_configure_spawner()` are safe to keep in the chain even when the YAML does not use MCP tools or a `spawner:` section.

### From Rust API

```rust
use ai_agents::{AgentBuilder, UnifiedLLMProvider, ProviderType};
use std::sync::Arc;

#[tokio::main]
async fn main() -> ai_agents::Result<()> {
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

See the [examples/](examples/) directory for more.

## CLI

```sh
# Install from crates.io
cargo install ai-agents-cli --version 1.0.0-rc.16

# Or run directly from source
cargo run -p ai-agents-cli -- run agent.yaml
```

```sh
ai-agents-cli run agent.yaml                          # interactive REPL
ai-agents-cli run agent.yaml --stream --show-tools     # stream tokens, show tool calls
ai-agents-cli run agent.yaml --show-state --show-timing # show state transitions and timing
ai-agents-cli validate agent.yaml                      # check YAML without starting
ai-agents-cli eval --agent agent.yaml --scenarios eval/suite.yaml --output eval_results/
```

Use `eval` to run YAML or JSONL scenario suites with strict fail-closed assertions, exact replay, synchronized record fixtures, optional LLM judge checks, retries, strict default redaction, JSON/Markdown/JUnit outputs, and observability reports when enabled. Real and record modes require explicit `--real-llm` or `--record` authorization.

See the [CLI Guide](https://ai-agents.rs/docs/cli/) for REPL commands, evaluation options, metadata configuration, and full reference.

## Roadmap

See the [full roadmap](https://ai-agents.rs/roadmap/) for what's shipped, what's next, and the complete feature catalog.

## Documentation

| Resource | Description |
|----------|-------------|
| [Getting Started](https://ai-agents.rs/docs/getting-started/) | Install and run your first agent in under a minute |
| [YAML Reference](https://ai-agents.rs/docs/yaml-reference/) | Complete spec for agent definition files |
| [CLI Guide](https://ai-agents.rs/docs/cli/) | All commands, flags, and REPL features |
| [Rust API](https://ai-agents.rs/docs/rust-api/) | Embedding agents in your Rust application |
| [Providers](https://ai-agents.rs/docs/providers/) | Setup for all 12 LLM providers |
| [Concepts](https://ai-agents.rs/docs/concepts/) | Architecture, lifecycle, and core ideas |
| [Examples](https://ai-agents.rs/examples/) | YAML and Rust examples for every feature |
| [API Docs](https://docs.rs/ai-agents) | Auto-generated Rust API reference |

## Key Dependencies

| Crate | Role |
|-------|------|
| [llm](https://crates.io/crates/llm) | Unified LLM provider interface (OpenAI, Anthropic, Google, Ollama, and more) |
| [rmcp](https://crates.io/crates/rmcp) | Official Rust SDK for Model Context Protocol (MCP) |
| [tokio](https://crates.io/crates/tokio) | Async runtime |
| [minijinja](https://crates.io/crates/minijinja) | Jinja2-compatible template engine for system prompts and spawner templates |
| [sqlx](https://crates.io/crates/sqlx) | SQLite storage backend (optional, `sqlite` feature) |
| [redis](https://crates.io/crates/redis) | Redis storage backend (optional, `redis-storage` feature) |

## Independence Notice

This repository is an independent open-source project maintained by the author in a personal capacity.

It is not an official product or offering of any employer, and no employer owns or governs this project.

See [INDEPENDENCE.md](./INDEPENDENCE.md) for details.

## License

Licensed under the Apache License, Version 2.0.

See [LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0.

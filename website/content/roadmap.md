+++
title = "Roadmap"
template = "page.html"
description = "What's shipped, what's next, and where the framework is headed."
+++

This page tracks what has shipped in each release candidate, what we are working on next, and the full catalog of planned features.

---

## What's Shipped

| Release | Highlights |
|---------|------------|
| **Pre-RC** | Core framework: YAML agents, tools, skills, states, hooks, HITL, streaming, error recovery, process pipeline |
| **rc.1** | CompactingMemory, token budgeting, SQLite/Redis storage |
| **rc.2** | Tool provider system, multi-language aliases, TrustLevel |
| **rc.3** | Workspace refactoring - 18 modular crates for parallel compilation |
| **rc.4** | Reasoning & reflection - Chain-of-Thought, ReAct, Plan-and-Execute, self-evaluation |
| **rc.5** | Intent disambiguation - LLM-based ambiguity detection and clarification |
| **rc.6** | MCP integration, tool scoping, intent-based routing, `openai-compatible` provider |
| **rc.7** | Dynamic agent spawning - runtime agent creation, registry, template system |
| **rc.8** | Multi-agent orchestration - router, pipeline, concurrent, group chat, handoff patterns |
| **rc.9** | Agent persona - structured identity, evolution, secrets, templates; dot-path refactor |
| **rc.10** | CLI context injection (`--context`, `--context-file`, `--plain`) and ratatui TUI with side panels, streaming, themes |
| **rc.11** | Session management and key facts - cross-session actor memory, key facts extraction, session metadata, and CLI/TUI actor commands |
| **rc.12** | Relationship memory - actor-scoped trust, sentiment, rapport, two-sided relationships, actor-aware inter-agent context, and relationship inspection in the REPL/TUI |
| **rc.13** | LLM provider enhancements and observability - capability overrides, Ollama options, privacy-safe tracing, cost metrics, reports, raw events, and Prometheus text export |

---

## Up Next

Once the current foundation line is fully released, the roadmap focus shifts toward deeper persistent memory, autonomous behavior, knowledge boundaries, and richer multi-agent ecosystems.

---

## Feature Catalog

Every planned feature and its current status. Entries are ordered by release target, while the status field keeps the release/version note.

| Feature | Description | Status |
|---------|-------------|--------|
| **Advanced Memory** | CompactingMemory, token budgeting, SQLite/Redis storage | ✅ Done (rc.1) |
| **Tool Provider System** | ToolProvider trait, multi-language aliases, extensibility | ✅ Done (rc.2) |
| **Workspace Refactoring** | 18 modular crates for parallel compilation | ✅ Done (rc.3) |
| **Reasoning & Reflection** | Chain-of-Thought, ReAct, Plan-and-Execute, self-evaluation | ✅ Done (rc.4) |
| **Intent Disambiguation** | LLM-based ambiguity detection and clarification | ✅ Done (rc.5) |
| **MCP Integration** | Connect to any MCP server for instant tool access | ✅ Done (rc.6) |
| **Dynamic Agent Spawning** | Runtime agent creation from YAML/templates, agent registry, parent-to-child messaging | ✅ Done (rc.7) |
| **Multi-Agent Orchestration** | Router, pipeline, concurrent, group chat, and handoff patterns with context policy and HITL gates | ✅ Done (rc.8) |
| **Agent Persona** | Structured, persistent, evolvable agent identity with personality, backstory, goals, secrets | ✅ Done (rc.9) |
| **CLI Context Injection + TUI** | Runtime context injection (--context, --context-file), ratatui-based TUI with side panels, --plain fallback, streaming render | ✅ Done (rc.10) |
| **Session Management + Key Facts** | Cross-session actor memory, key facts extraction, and session metadata | ✅ Done (rc.11) |
| **Relationship Memory** | Actor-scoped trust, sentiment, rapport, notable relationship events, two-sided relationships, and actor-aware inter-agent context | ✅ Done (rc.12) |
| **LLM Provider Enhancement** | Provider factory, stable token counting, client caching, config passthrough | ✅ Done (rc.13) |
| **Observability & Tracing** | Privacy-safe per-call latency, token usage, cost estimates, trace context, reports, and exporters | ✅ Done (rc.13) |
| **Evaluation Framework** | YAML-driven scenario runner with assertions, LLM judge, and metrics | Planned (rc.14) |
| **Runtime Latency Optimization** | Safe speculative LLM scheduling, pre-response transitions, parallel post-turn memory updates, and orchestration aggregation speedups | Planned (rc.15) |
| **Built-in Tool Expansion** | Safe search tools, split file read/write/edit tools, interactive questions, session todos, bounded waits, and web retrieval primitives | Planned (rc.16) |
| **Stable Foundation Release** | Dynamic spawning, orchestration, persona, actor memory, relationship memory, CLI/TUI, runtime optimization, built-in tools, and release hardening | Planned (v1.0.0) |
| **Episodic Memory** | Structured event records with participants, significance, and source tracking | Planned |
| **Conversation Style Modifiers** | LLM-based dynamic tone, formality, and style adaptation | Planned |
| **Background Tasks & Scheduling** | Async job execution with cron, interval, event triggers, and DAG dependencies | Planned |
| **VectorDB Tool** | Embedding storage and similarity search tool | Planned |
| **Knowledge Base / RAG Pipeline** | Document ingestion, chunking, and retrieval-augmented generation | Planned - needs VectorDB |
| **Knowledge Scoping** | Source-based knowledge boundaries and access control | Planned |
| **Shared Memory** | Group-level shared memory stores with publish/subscribe | Planned - needs Multi-Agent Orchestration |
| **Memory Dynamics** | Salience scoring, time-based decay, and context-aware retrieval ranking | Planned |
| **Agent Composition Patterns** | Composite workflows, magentic orchestration, advanced multi-agent patterns | Planned - needs Multi-Agent Orchestration |
| **Budget Control** | Per-session and per-agent cost limits with fallback on budget exceeded | Planned - needs Observability |
| **Conversation Scripts** | Declarative guided flows (wizards, forms) with LLM extraction | Planned |
| **Semantic Caching** | Cache semantically similar queries to reduce LLM calls | Planned |
| **Hot Reload** | Live YAML config updates with graceful session handling and auto-rollback | Planned |
| **Code Interpreter** | Sandboxed code execution with templates and persistent library | Planned |
| **A2A Protocol** | Cross-service agent collaboration protocol | Planned - needs Multi-Agent Orchestration |
| **Custom Reasoning Prompts** | Domain and language-specific CoT/ReAct instruction templates | Planned |
| **Reasoning Depth Control** | Auto shallow/standard/deep reasoning with resource limits | Planned - needs Custom Reasoning Prompts |
| **Python Runtime Bindings** | Native Python package for loading YAML agents, chat and streaming, context injection, sessions, actor memory, relationships, and observability through the Rust runtime | Planned after v1 |
| **Desktop Agent Studio** | Desktop workbench for authoring YAML agents, validating configs, running local Ollama agents, and inspecting state, memory, tools, sessions, and observability | Planned after v1 |

---

*This roadmap reflects current plans and may change as priorities evolve.*

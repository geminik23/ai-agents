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

---

## Up Next

These are the features we are actively planning for the next few release candidates.

### Agent Persona

Structured, persistent, evolvable agent identity - personality, role, backstory, goals, and secrets defined in YAML. Enables consistent character behavior across sessions.

### Session Management + Key Facts

Persistent user context and key facts across sessions. The agent remembers who you are and what matters between conversations.

### Relationship Memory

Per-actor trust, sentiment, rapport, and interaction history with LLM-based auto-evaluation. Agents build and maintain relationships over time.

---

## Feature Catalog

Every planned feature and its current status. Features are independent unless noted.

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
| **Agent Persona** | Structured, persistent, evolvable agent identity with personality, backstory, goals | Planned |
| **Session Management + Key Facts** | Persistent user context and key facts across sessions | Planned |
| **Relationship Memory** | Per-actor trust, sentiment, rapport, and interaction history | Planned |
| **Episodic Memory** | Structured event records with participants, significance, and source tracking | Planned |
| **LLM Provider Enhancement** | Provider factory, stable token counting, client caching, config passthrough | Planned |
| **Evaluation Framework** | YAML-driven scenario runner with assertions, LLM judge, and metrics | Planned |
| **Observability & Tracing** | Per-call latency, token usage, cost tracking via hooks | Planned |
| **Budget Control** | Per-session and per-agent cost limits with fallback on budget exceeded | Planned — needs Observability |
| **Conversation Scripts** | Declarative guided flows (wizards, forms) with LLM extraction | Planned |
| **Conversation Style Modifiers** | LLM-based dynamic tone, formality, and style adaptation | Planned |
| **Custom Reasoning Prompts** | Domain and language-specific CoT/ReAct instruction templates | Planned |
| **Reasoning Depth Control** | Auto shallow/standard/deep reasoning with resource limits | Planned — needs Custom Reasoning Prompts |
| **VectorDB Tool** | Embedding storage and similarity search tool | Planned |
| **Knowledge Base / RAG Pipeline** | Document ingestion, chunking, and retrieval-augmented generation | Planned — needs VectorDB |
| **Knowledge Scoping** | Source-based knowledge boundaries and access control | Planned |
| **Shared Memory** | Group-level shared memory stores with publish/subscribe | Planned — needs Multi-Agent Orchestration |
| **Memory Dynamics** | Salience scoring, time-based decay, and context-aware retrieval ranking | Planned |
| **Background Tasks & Scheduling** | Async job execution with cron, interval, event triggers, and DAG dependencies | Planned |
| **Hot Reload** | Live YAML config updates with graceful session handling and auto-rollback | Planned |
| **Code Interpreter** | Sandboxed code execution with templates and persistent library | Planned |
| **Semantic Caching** | Cache semantically similar queries to reduce LLM calls | Planned |
| **A2A Protocol** | Cross-service agent collaboration protocol | Planned — needs Multi-Agent Orchestration |
| **Agent Composition Patterns** | Composite workflows, magentic orchestration, advanced multi-agent patterns | Planned — needs Multi-Agent Orchestration |

---

*This roadmap reflects current plans and may change as priorities evolve.*

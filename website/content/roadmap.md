+++
title = "Roadmap"
template = "page.html"
description = "What's shipped, what's next, and where the framework is headed."
+++

This page tracks what has shipped in each release candidate, what we are working on next, and the full catalog of planned features.

---

## Status Legend

| Status | Meaning |
| --- | --- |
| ✅ Done | Released and available |
| Implemented on main | Implemented but not yet published in a release candidate |
| Current | Active focus for the next release candidate |
| Planned | Planned, but release target may change |
| Planned within v1 | Confirmed for the v1 release line |
| Planned after v1 | Planned after the stable foundation release |
| Planned optional | Optional companion product or workspace; not required for the core runtime |

---

## What's Shipped

| Release | Highlights |
| --- | --- |
| **Pre-RC** | Core framework: YAML agents, tools, skills, states, hooks, HITL, streaming, error recovery, process pipeline |
| **rc.1** | CompactingMemory, token budgeting, SQLite/Redis storage |
| **rc.2** | Tool provider system, multi-language aliases, TrustLevel |
| **rc.3** | Workspace refactoring - modular crates for parallel compilation and feature isolation |
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
| **rc.14** | Evaluation framework - YAML and JSONL scenario suites, fixtures, assertions, judges, strict redaction, observability overlay, and CI reports |
| **rc.15** | Runtime latency optimization + speculative branch execution - pre-response routing, background actor-memory maintenance, stable orchestration ordering, parallel transition decisions, speculative skill and reasoning branches, buffered streaming, branch observability, and eval flush support |

---

## Up Next

| Target | Focus | Summary |
| --- | --- | --- |
| **Next RC** | Stable foundation release hardening | Examples, docs, migration polish, and final release packaging on top of the now-complete built-in tool foundation |
| **v1.0.0** | Stable Foundation Release | Stable schema/API, release hardening, documentation cleanup, examples, migration notes, and safe built-in tool foundation |
| **v1.1.0** | Generalized Autonomy Runner | YAML-configured task runs with lifecycle stages, todos, completion gates, validation loops, pause/resume, and task-run events |
| **v1 line** | Retrieval & Evidence Layer | Shared EvidenceItem/EvidencePack model, retriever registry, query transforms, fusion/rerank hooks, evidence budgets, and context assembly |
| **v1 line** | Knowledge Base / RAG Pipeline | Extensible source ingestion, named indexing/retrieval pipelines, chunking, embeddings, hybrid retrieval, knowledge_search, and safe refresh |
| **v1 line** | Knowledge Scoping & Evidence Access Policy | Source/provenance/persona/role/group/delegation access policy for evidence before rerank and context injection |
| **v1 line** | Background Tasks & Runtime Maintenance Scheduling | Cron, interval, event, and manual triggers for skills, tools, prompts, knowledge refresh, scope audit, memory sweeps, and observability export |
| **v1 line** | Python Runtime Bindings | Python package backed by the Rust runtime for loading YAML agents, chat, streaming, sessions, memory, relationships, and observability |
| **v1 line** | OpenTelemetry Exporter | OTLP export for privacy-safe runtime traces to external observability backends |
| **v1 line** | Realtime & Audio Interfaces | Typed turn APIs and provider-neutral adapters for audio-input models and realtime speech sessions |

Once the stable foundation is fully released, the roadmap focus shifts toward autonomous task runs, the information lifecycle layer for retrieval/evidence/RAG/scoping/background maintenance, Python access to the Rust runtime, external observability integration, realtime/audio interfaces, deeper persistent memory, and richer multi-agent ecosystems.


---

## Information Lifecycle Sequence

The following four features are separate implementation units, but they share one architecture. They should be implemented in this order within the v1 line.

```text
Retrieval & Evidence Layer
  -> defines EvidenceItem, EvidencePack, Retriever, RetrievalPlanner, query transforms, fusion/rerank hooks, evidence budgets, and ContextAssembler

Knowledge Base / RAG Pipeline
  -> implements source ingestion, parsing, chunking, contextualization, embeddings, named indexing/retrieval pipelines, KnowledgeRetriever, knowledge_search, and safe refresh

Knowledge Scoping & Evidence Access Policy
  -> applies source/provenance/persona/role/group/delegation policy to EvidenceItem values before rerank and final context injection

Background Tasks & Runtime Maintenance Scheduling
  -> schedules knowledge refresh, scope audit, memory dynamics, cache cleanup, observability export, and future autonomy-run triggers through the F28 maintenance contract
```

This sequence replaces the old VectorDB-first framing. Vector search is now one backend inside the Knowledge Base / RAG Pipeline, while Retrieval & Evidence is the shared foundation used by file, web, memory, knowledge, tool, and future graph/hierarchical retrievers.

---

## Feature Catalog

Every planned feature and its current status. Entries are ordered by release target, while the status field keeps the release/version note.

| Feature | Description | Status | Notes |
| --- | --- | --- | --- |
| **Advanced Memory** | CompactingMemory, token budgeting, SQLite/Redis storage | ✅ Done | rc.1 |
| **Tool Provider System** | ToolProvider trait, multi-language aliases, extensibility | ✅ Done | rc.2 |
| **Workspace Refactoring** | Modular crates for parallel compilation and feature isolation | ✅ Done | rc.3 |
| **Reasoning & Reflection** | Chain-of-Thought, ReAct, Plan-and-Execute, self-evaluation | ✅ Done | rc.4 |
| **Intent Disambiguation** | LLM-based ambiguity detection and clarification | ✅ Done | rc.5 |
| **MCP Integration** | Connect to any MCP server for instant tool access | ✅ Done | rc.6 |
| **Dynamic Agent Spawning** | Runtime agent creation from YAML/templates, agent registry, parent-to-child messaging | ✅ Done | rc.7 |
| **Multi-Agent Orchestration** | Router, pipeline, concurrent, group chat, and handoff patterns with context policy and HITL gates | ✅ Done | rc.8 |
| **Agent Persona** | Structured, persistent, evolvable agent identity with personality, backstory, goals, secrets | ✅ Done | rc.9 |
| **CLI Context Injection + TUI** | Runtime context injection (`--context`, `--context-file`), ratatui-based TUI with side panels, `--plain` fallback, streaming render | ✅ Done | rc.10 |
| **Session Management + Key Facts** | Cross-session actor memory, key facts extraction, and session metadata | ✅ Done | rc.11 |
| **Relationship Memory** | Actor-scoped trust, sentiment, rapport, notable relationship events, two-sided relationships, and actor-aware inter-agent context | ✅ Done | rc.12 |
| **LLM Provider Enhancement** | Provider factory, stable token counting, client caching, config passthrough | ✅ Done | rc.13 |
| **Observability & Tracing** | Privacy-safe per-call latency, token usage, cost estimates, trace context, reports, and exporters | ✅ Done | rc.13 |
| **Evaluation Framework** | YAML/JSONL scenario runner with assertions, LLM judge, fixtures, strict redaction, parallel execution, observability overlay, and CI reports | ✅ Done | rc.14 |
| **Runtime Latency Optimization** | Pre-response deterministic transitions, background actor-memory maintenance, stable orchestration ordering, branch-aware observability foundations, and eval flushing | ✅ Done | rc.15 |
| **Speculative Branch Execution** | Bounded speculative main drafts, parallel response-independent transitions, skill routing, auto reasoning, buffered streaming, and branch observability finalization | ✅ Done | rc.15 |
| **Built-in Tool Expansion** | Safe discovery, repository inspection, diagnostics, interactive questions, session todos, bounded waits, web retrieval primitives, context-aware tool policy, file write/edit, patch review, and controlled validation commands | Implemented on main | rc.16 |
| **Stable Foundation Release** | Dynamic spawning, orchestration, persona, actor memory, relationship memory, CLI/TUI, runtime optimization, built-in tools, and release hardening | Planned | v1.0.0 |
| **Generalized Autonomy Runner** | YAML-configured task runs with lifecycle stages, todos, completion gates, validation loops, pause/resume, and task-run event streaming for code, research, support, operations, and workflow agents | Planned within v1 | Confirmed for the v1 line |
| **Python Runtime Bindings** | Native Python package for loading YAML agents, chat and streaming, task runs, context injection, sessions, actor memory, relationships, and observability through the Rust runtime | Planned within v1 | Confirmed for the v1 line |
| **OpenTelemetry Exporter** | Export privacy-safe `ai-agents` runtime traces through OTLP to external observability backends such as LangSmith, Grafana Tempo, Jaeger, and Datadog. Covers LLM calls, tool calls, skill routing, state transitions, memory operations, multi-agent handoffs, speculative branch outcomes, eval metadata, latency, token usage, and cost estimates. | Planned within v1 | Builds on existing observability and tracing |
| **Realtime & Audio Interfaces** | Typed turn APIs and provider-neutral adapters for audio-input models, realtime speech sessions, runtime delegation, external history sync, interruption handling, and safe committed-text streaming | Planned within v1 | Adds speech/realtime session support without tying the runtime to one provider |
| **Retrieval & Evidence Layer** | Unified retrieval/evidence foundation: EvidenceItem, EvidencePack, Retriever trait, retrieval planning, query transforms, fusion/rerank hooks, evidence judging, token budgets, context assembly, and retrieval observability across file, web, memory, knowledge, tool, and future graph/hierarchical sources | Planned within v1 | Information Lifecycle 1/4. Replaces the old VectorDB-first foundation and unblocks RAG/scoping |
| **Knowledge Base / RAG Pipeline** | Extensible source ingestion, parsing, chunking, contextualization, embeddings, named indexing pipelines, named retrieval pipelines, lexical/vector/hybrid indexing, knowledge_search tool, and background-safe refresh returning structured EvidenceItem values instead of direct prompt text | Planned within v1 | Information Lifecycle 2/4. Depends on Retrieval & Evidence Layer |
| **Knowledge Scoping & Evidence Access Policy** | Evidence-level access policy for source/provenance/persona/role/group/delegation scoping, deny-wins enforcement, strict/audit modes, violation tracing, and scope-aware retrieval/rerank/injection across knowledge, memory, tools, and agents | Planned within v1 | Information Lifecycle 3/4. Depends on Retrieval & Evidence Layer; benefits from RAG, persona, relationship, and memory features |
| **Background Tasks & Runtime Maintenance Scheduling** | Declarative cron, interval, event, and manual triggers for skills, tools, prompts, runtime maintenance operations, knowledge refresh, scope audit, memory dynamics, cache cleanup, observability export, and future autonomy-run triggers while reusing F28 queue/flush/shutdown semantics | Planned within v1 | Information Lifecycle 4/4. Schedules maintenance without creating a separate worker model |
| **Desktop Agent Studio** | Optional desktop app for authoring, validating, running, and inspecting YAML agents | Planned optional | May be skipped, shipped as a separate product, or added as an optional workspace. Repo and license are TBD. Not a v1.0 blocker. |
| **Episodic Memory** | Structured event records with participants, significance, and source tracking | Planned | Future memory expansion; can later use Retrieval & Evidence for semantic recall |
| **Conversation Style Modifiers** | LLM-based dynamic tone, formality, and style adaptation | Planned | Should remain separate from core behavior semantics |
| **Shared Memory** | Group-level shared memory stores with publish/subscribe | Planned | Needs Multi-Agent Orchestration |
| **Memory Dynamics** | Salience scoring, time-based decay, and context-aware retrieval ranking | Planned | Needs evaluation strategy |
| **Agent Composition Patterns** | Composite workflows, magentic orchestration, advanced multi-agent patterns | Planned | Needs Multi-Agent Orchestration |
| **Budget Control** | Per-session and per-agent cost limits with fallback on budget exceeded | Planned | Needs Observability |
| **Conversation Scripts** | Declarative guided flows such as wizards and forms with LLM extraction | Planned | May build on state machines |
| **Semantic Caching** | Cache semantically similar queries to reduce LLM calls | Planned | Needs privacy and invalidation design |
| **Hot Reload** | Live YAML config updates with graceful session handling and auto-rollback | Planned | Developer experience feature |
| **Code Interpreter** | Sandboxed code execution with templates and persistent library | Planned | Needs sandbox and safety model |
| **A2A Protocol** | Cross-service agent collaboration protocol | Planned | Needs Multi-Agent Orchestration |
| **Custom Reasoning Prompts** | Domain and language-specific CoT/ReAct instruction templates | Planned | Needs evaluation strategy |
| **Reasoning Depth Control** | Auto shallow/standard/deep reasoning with resource limits | Planned | Needs Custom Reasoning Prompts and budget control |

---

*This roadmap reflects current plans and may change as priorities evolve.*

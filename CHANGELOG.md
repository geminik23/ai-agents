# Changelog

## 1.0.0-rc.8

### Added
- Orchestration: five coordination patterns (router, pipeline, concurrent, group chat, handoff) as declarative state configs or LLM-invoked tools
- Orchestration: delegate states forward messages to registry agents with context mode (input_only, summary, full) controlling how much parent history reaches the sub-agent
- Orchestration: context_mode field on all four non-delegate patterns forwards parent conversation history to sub-agents across turns
- Orchestration: group chat with four styles (brainstorm, consensus, debate, maker-checker), LLM-directed speaker selection, manager.agent for custom termination and turn control, participant roles in prompts, and consensus auto-detection
- Orchestration: concurrent execution with aggregation strategies (voting, llm_synthesis, first_wins, all), weighted voting, tiebreaker (first, random, router_decides), and on_partial_failure (abort, proceed_with_available)
- Orchestration: pipeline with per-stage Jinja2 templates, named stage references via stages.<agent_id>, and minijinja rendering
- Orchestration: handoff with structured JSON decisions (action, confidence, reason) and fuzzy text fallback
- Orchestration: maker-checker on_max_iterations supports accept_last, escalate, and fail
- Orchestration: auto-spawn build-time validation fails with a clear error when referenced agents are missing
- Orchestration: five tool implementations (route_to_agent, pipeline_process, concurrent_ask, group_discussion, handoff_conversation) registered via orchestration_tools config
- Orchestration: structured results stored in context.orchestration after each run, accessible in Jinja2 templates and guard conditions
- Orchestration: all input templates expose user_input and context.<key> variables consistently
- Orchestration: lifecycle hooks for delegate, concurrent, group chat round, pipeline stage, and handoff events
- LLM config: timeout_seconds, reasoning, reasoning_effort, reasoning_budget_tokens as first-class YAML fields with merge support
- LLM builder forwarding: resilient transport settings, Azure config, extra_body escape hatch, OpenAI web search, and xAI search params read from YAML extra fields
- Disambiguation: skill-level overrides with required_clarity enforcement fire at runtime after skill routing
- Disambiguation: LLM-based detection of clarification abandonment and topic switches during pending clarification

### Changed
- LLM config: reasoning_effort promoted from extra-only to a first-class field with backward-compatible fallback

### Fixed
- Disambiguation: detection threshold was overridden by the LLM is_ambiguous boolean, causing false positives on clear inputs
- Disambiguation: clarification_templates with custom keys were silently ignored
- Disambiguation: required_clarity fields were not enforced, allowing unrequested fields in clarification
- Disambiguation: answering_agent_question skip condition fired on any assistant message ending with a question mark instead of checking semantic relevance
- Disambiguation: max_attempts off-by-one produced one extra exchange
- Disambiguation: skill-level pending state was cleared before the runtime read it, silently falling through to re-routing

## 1.0.0-rc.7

### Added
- Dynamic agent spawning: create and manage child agents at runtime from YAML templates or LLM-generated specs, with shared LLMs, storage, shared context, spawn limits and  tool allowlists
- Spawner template metadata: generate_agent tool auto-discovers template names, descriptions, and variables so the LLM selects the right template without system prompt instructions
- CLI session persistence: /save, /load, /sessions, /delete commands with spawner-aware cascading that persists the full multi-agent graph
- CLI HITL approval handler with multi-language support, configurable via metadata.cli.hitl
- Error recovery wiring: fallback_llm, fallback_response, tool skip, and tool fallback actions now execute at runtime
- State machine: on_reenter action, regeneration control, transition cooldown, dead state detection, LLM-based context extractors, and per-state process pipeline override
- Build-time tool validation: agent build fails if YAML declares tools not registered in the tool registry

### Changed
- Spawner templates support file path references alongside inline strings
- HITL approval returns a full result type instead of boolean, and Modified arguments are merged into the tool call

### Fixed
- Exists and Compare guards were ignored during YAML parsing due to variant ordering in ContextMatcher
- Context extractor dotted keys were stored flat instead of nested, and guard comparison did not coerce types between strings and YAML booleans or numbers
- Tool result messages did not include the tool name, and assistant tool-call messages were not stored before execution
- HITL checks used display names instead of tool IDs, approval messages showed raw template syntax instead of rendered text, and LLM-based message fallback failed when the router LLM was missing
- Spawner shared_storage was deserialized but silently ignored at runtime
- SqliteStorage failed to open when parent directories did not exist

## 1.0.0-rc.6

### Added
- MCP integration via rmcp v1.2 SDK: each MCP server exposed as a single builtin tool with function dispatch (stdio, HTTP, SSE transports)
- MCP view tools: named function subsets of an MCP server registered as separate tools for per-state scoping, sharing the parent connection
- MCP per-function HITL via security.hitl_functions, enforced uniformly across parent and view tools
- OpenAI-compatible and OpenRouter provider types with base_url and api_key_env YAML fields
- Agent-level tool scoping: tools field controls which tools the LLM sees
- Parallel tool call support end-to-end: prompt, parser, and executor
- CLI crate with reusable REPL, tracing init, and YAML-first workflow
- Reasoning effort passthrough (low, medium, high) for supported providers
- Custom CLI command callbacks via on_command builder method

### Changed
- Streaming path now has full feature parity with blocking chat
- Examples moved to independent workspace grouped by feature area
- CLI REPL commands now require / prefix
- HttpTool always available: http-tool feature gate removed
- LLM config fields (temperature, max_tokens, top_p, base_url) forwarded from YAML to providers
- System prompts passed via builder.system() instead of user-message conversion

### Fixed
- Tool scoping: LLM prompt, disambiguation, and planning now respect declared tools
- Context manager initialization was never called on startup
- Skill loader relative file path resolution against YAML directory instead of CWD
- Parallel tool call parser now handles JSON arrays
- Streaming final chunk sentinel with is_final and finish reason
- Post-transition tool calls were returned as raw JSON text instead of being executed

## 1.0.0-rc.5

### Added
- First release of the rewritten framework (previously published as 0.x)
- Intent disambiguation: LLM-based ambiguity detection, clarification generation, multi-turn resolution
- State/skill-level disambiguation overrides with configurable thresholds

## 1.0.0-rc.4

### Added
- Reasoning modes: none, chain-of-thought, ReAct, plan-and-execute, auto (LLM selects)
- Reflection: LLM self-evaluation with criteria, retry on failure, configurable thresholds
- Per-state and per-skill reasoning/reflection overrides

## 1.0.0-rc.3

### Changed
- Split monolithic crate into workspace architecture (17 crates)

## 1.0.0-rc.2

### Added
- Tool provider system and aliases for extensible tool support

## 1.0.0-rc.1

### Added
- CompactingMemory with auto-summarization and token budgeting
- Storage backends (SQLite, Redis) with YAML integration

## Pre-RC (initial development)

### Added
- YAML-defined agents with system prompt, tools, skills, and behavior in one file
- Multi-LLM support with aliases (default, router, evaluator) and auto-fallback
- Skill system with LLM-based intent routing and multi-step execution
- State machine with hierarchical states, LLM-evaluated transitions, guards, entry/exit actions
- Dynamic context injection from runtime, file, HTTP, env, and callback sources
- Template rendering with Jinja2-compatible syntax (minijinja)
- Built-in tools: calculator, datetime, JSON, random, HTTP, file, text, template, math
- Conditional tool availability: context, state, time, semantic, and composite conditions
- Streaming with real-time token streaming and tool/state events
- Parallel tool execution with configurable concurrency
- Agent hooks for lifecycle events (message, LLM, tool, state, error, response)
- Human-in-the-loop: tool, condition, and state approval with multi-language localization
- Error recovery: retry with backoff, LLM fallback, context overflow handling
- Tool security: rate limiting, domain/path restrictions, confirmation requirements
- Input/output process pipeline: normalize, detect, extract, sanitize, validate, transform, format

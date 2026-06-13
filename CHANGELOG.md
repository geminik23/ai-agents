# Changelog

## 1.0.0-rc.15

### Added
- Runtime optimization: disabled-by-default policy block for pre-response routing, background maintenance, bounded runtime work, and speculative branches
- Transition timing: pre-response and parallel transition timing with response-dependency safety checks
- Pre-response routing: deterministic guard and resolved-intent routes can skip stale old-state responses
- Background maintenance: facts and relationship updates can run inline parallel or background with task-aware freshness waits
- Speculative branches: bounded main drafts, parallel transition decisions, skill routing, auto reasoning, and buffered streaming routing
- Branch observability: committed, discarded, failed, and cancelled branch outcomes expose status, winner, optimization, commit behavior, and speculative dimensions
- Evaluation coverage: no-key runtime optimization suites cover pre-response routing, speculative branches, losing tool drafts, reasoning decisions, and buffered streaming
- Rust API: runtime optimization policy types and TransitionTiming are re-exported for host configuration

### Changed
- State transitions: selection, approval, commit, and fallback are separated so side effects happen only after the winning path is known
- Turn lifecycle: user-message commit, response finalization, post-turn maintenance, and redispatch cleanup share one root-turn path
- Speculative scheduling: branch task capacity and speculative LLM budgets are bounded and support partial scheduling under low caps
- Streaming optimization: buffered routing is scoped to response-independent parallel state transitions and emits committed output only
- Speculative scheduling: plain drafts commit only when they preserve the equivalent serial reasoning path
- Runtime optimization examples: speculative examples now enable branch observability reports for committed, discarded, failed, and cancelled outcomes
- Orchestration voting: vote extraction can run with bounded concurrency while preserving declaration-order tie breaking
- Documentation: roadmap, website docs, examples, and guides describe runtime optimization and speculative branch behavior as one release

### Fixed
- Pre-response safety: staged extractor context and user memory are not committed when route approval rejects
- Background maintenance: completed task errors are preserved until flush and overflow errors surface according to policy
- Transition timing: post-response guard or intent transitions no longer run as pre-response routes
- Tool approval lifecycle: rejected tool terminal outcomes run shared finalization in blocking and streaming paths
- Losing branches: discarded drafts cannot write memory, execute parsed tool calls, mutate context, fire response hooks, or leak streaming output
- Branch telemetry: failed, cancelled, and discarded branches preserve exact safe dimensions including speculative labels
- Auto reasoning: judge failures and reservation exhaustion no longer appear as committed none decisions
- Reasoning preservation: forced reasoning modes skip plain speculative drafts, and low-cap auto reasoning falls back to the serial judge path
- Skill routing: low-cap speculative skill routing falls back to serial skill routing instead of reporting no match
- Root-turn cleanup: blocking and streaming errors reset root-turn state before the next turn
- Buffered streaming: main branch failures finalize telemetry before stream errors are returned
- Buffered streaming: buffer capacity is enforced only while routing is unresolved

## 1.0.0-rc.14

### Added
- Evaluation framework: YAML and JSONL suites can test agents through the normal runtime path with assertions and CI reports
- Evaluation fixtures: mock tools, context overlays, local HTTP routes, and mock, replay, record, or real LLM modes are available for suites
- Evaluation judge: optional semantic scoring can use a configured LLM while deterministic assertions remain rule-based
- Evaluation examples: no-key suites cover basic chat, multi-turn, streaming, and observability, with a live real-LLM suite for explicit smoke tests

### Changed
- Evaluation output privacy: default reports redact inputs and responses and omit raw turn evidence from machine-readable artifacts
- Evaluation execution: suites now validate configuration before running and can run scenario-isolated cases concurrently
- Evaluation storage: file and SQLite eval runs use isolated temporary storage before spawned agents are configured

### Fixed
- Evaluation no-op fields: metadata, facts, relationships, orchestration, observability, judge aliases, streaming, mock server, and scenario timeout settings now have explicit behavior
- Evaluation CLI validation: invalid tag modes, incompatible LLM fixture overrides, and output write failures now exit with configuration errors

---

## 1.0.0-rc.13

### Added
- LLM capability overrides: YAML can declare function calling, vision, and JSON mode support for providers and OpenAI-compatible servers, and those overrides now drive capability checks
- Ollama provider options: YAML can pass context window, GPU layer, and keep-alive settings through native Ollama provider config, with named values merged into request bodies
- Multi-provider routing facade: multi-LLM routing is available through the main crate without a direct provider-crate dependency
- Observability and tracing: YAML can collect privacy-safe latency, token, cost, trace, report, CSV, raw event, and Prometheus text metrics across chat, skills, process stages, facts, relationships, HITL, reasoning, and orchestration
- Observability pricing files: JSON and YAML pricing tables can be shared across agents with inline overrides

### Changed
- Observability privacy: approval metadata, tags, and errors are stored as safe labels or redacted summaries instead of debug strings
- Observability reporting: reports and exports drain pending events before metrics are read

### Fixed
- Ollama errors: invalid extra body shapes are rejected and connection or model failures show actionable serve and pull hints
- Observability provider coverage: process, spawner, and auto-spawned agent paths no longer bypass observed providers
- Observability attribution and trace continuity: process stages, clarification calls, delegation, registry broadcast, and concurrent orchestration now keep distinct purpose labels and connected traces
- Observability config and export: token count flags, pricing files, unknown-price error tagging, JSON raw events, Prometheus labels, and JSON export path collisions now behave as documented
- Observability privacy leaks: errors, tags, approval triggers, and approval results no longer expose raw details by default

---

## 1.0.0-rc.12

### Added
- Relationship memory: actor-scoped trust, sentiment, familiarity, rapport, custom dimensions, and notable events with prompt and context injection
- Relationship persistence: relationship state now survives across sessions in SQLite and full session snapshots
- Relationship commands and panel: inspect and adjust current actor relationship state in the REPL and TUI
- Two-sided relationship model: track agent-to-actor and perceived-actor-to-agent scores with a derived mutual view
- Actor-aware inter-agent context: spawned agents and orchestration flows preserve the original actor identity for facts and relationship memory
- Relationship prompt budgeting: token budgeting can reserve prompt space for relationship memory
- Perspective-aware relationship updates: REPL and TUI can update agent-to-actor or perceived-actor-to-agent directly
- Runtime observability: successful fact extraction and relationship updates now emit info-level logs with actor and update details

### Changed
- Two-sided relationship evaluation: automatic updates now write only the two stored perspectives while mutual remains a derived read-only view
- Relationship prompts and displays: two-sided prompt text now shows stored perspectives without duplicating the derived mutual view
- Actor-scoped memory isolation: facts are cached per actor and relationship context is rendered per turn to avoid leakage across actor switches and agent handoffs
- Relationship display: REPL and TUI label mutual values as derived and provide clearer two-sided inspection

### Fixed
- Relationship evaluator parsing: missing perspective or confidence no longer silently defaults to agent-to-actor or disappears at zero confidence
- Two-sided relationship updates: derived mutual proposals are rejected instead of being applied as stored relationship changes
- Unicode safety: Korean and other multibyte text no longer panics logging, previews, or memory summary rendering in relationship and actor-memory flows
- Relationship evaluator diagnostics: skipped proposals now surface clearer logs for low confidence, unknown dimensions, and invalid perspectives

---

## 1.0.0-rc.11

### Added
- Actor memory: cross-session identity via --actor flag or from_context path; returning actor's facts load automatically before the first turn without any application code
- Key facts extraction: router LLM extracts structured facts after each turn, persists to SQLite, and injects them via {{ actor_facts }} in the system prompt
- Session metadata: memory.session block for static tags and TTL on sessions; /cleanup removes sessions past their TTL
- Key facts dedup: exact mode (Levenshtein with same-category and cross-category thresholds) and llm mode (prompt-based semantic dedup)
- Actor identity: switching actors mid-session invalidates the cache and loads the new actor's facts on the next turn
- token_budget.allocation.facts: controls the injected fact block token size, overriding injection.max_tokens when set
- Actor memory hooks: on_facts_extracted, on_actor_memory_loaded, on_session_created, on_sessions_expired
- CLI --actor <ID> flag: sets actor identity at startup
- REPL /actor, /actor set, /actor facts, /actor delete commands
- REPL /facts and /facts extract commands
- REPL /sessions --actor and /sessions --tag for filtered session listing; /cleanup for TTL-based session removal
- REPL /memory extended with actor identity and cached fact count
- TUI: status bar shows active actor ID; facts panel shows live fact entries for the current actor
- TUI: /actor, /facts, /sessions, /cleanup slash commands and autocomplete wired for all actor memory commands
- Extraction errors visible by default: ai_agents_facts=warn added to the default tracing filter; post-turn [facts] notification shows how many facts were extracted each turn

### Changed
- Actor memory: streaming and blocking chat paths share the same actor loading, fact extraction, and session lifecycle
- Storage: AgentStorage extended with 9 methods for session metadata and actor facts; FileStorage and RedisStorage compile unchanged using no-op defaults
- Key facts: fact eviction is now durable; evicted facts are deleted from storage, not just dropped from the in-memory set

---

## 1.0.0-rc.10

### Added
- TUI: ratatui alternate-screen interface starts automatically on interactive terminals; use --plain to force the line REPL
- TUI status bar: agent name, version, current state, token budget percentage, and thinking spinner
- TUI side panels toggled with F1-F8: Help (commands), States (machine visualization), Memory (token budget), Context (values), Tools (last call), Persona (identity), Facts, Agents (spawned registry)
- TUI HITL modal: approval requests appear as an overlay dialog so the user can approve or reject without leaving the chat
- TUI streaming: tokens appear in real time with tool calls and state transitions shown inline in the chat
- TUI log rendering: tracing events captured and shown as dim cards in the chat timeline instead of writing raw text to the terminal
- TUI themes: 11 color themes (dark, light, one-dark, catppuccin-mocha, dracula, tokyo-night, vscode-dark, nord, gruvbox-dark, one-half-light, github-light) selectable via --theme flag, metadata.cli.theme YAML field, or Ctrl+T cycling at runtime
- TUI command completion: typing / opens a floating popup listing all slash commands with descriptions; Tab fills, Enter executes, Up/Down navigates
- CLI --context and --context-file flags: inject runtime context values at startup without writing Rust code
- REPL /context set and /context unset commands: modify context values during a session
- REPL /info command: shows agent name, version, skill count, and current state

### Changed
- TUI startup hints displayed as a grouped block with > prefix markers and distinct italic styling, separate from system messages
- TUI default tracing level is WARN in TUI mode; set RUST_LOG=info for verbose output
- RGB themes fill the terminal background with their own color for consistent appearance across terminals; ANSI themes (dark, light) defer to the terminal's native background

### Fixed
- TUI log corruption: tracing output no longer writes raw bytes into the alternate screen in TUI mode
- TUI input area: typed text clears correctly after pressing Enter instead of persisting on screen
- TUI agent responses: consecutive blank lines from LLM paragraph breaks are stripped; blank line separators appear only when the message role changes


---

## 1.0.0-rc.9

### Added
- Agent persona: structured identity, traits, goals, secrets, evolution, and templates via a top-level persona block in YAML
- Persona prompt injection: identity, traits, goals, and backstory are rendered into the system prompt automatically and preserved across all prompt_mode values
- Context-conditional secrets: secrets are revealed only when typed matcher conditions pass against runtime context
- Persona evolution: mutable persona fields can change over time with audit history, with optional LLM-driven updates behind double opt-in
- Persona persistence: persona state is saved and restored as part of session snapshots
- Agent hooks: persona evolution and secret reveal events are exposed through AgentHooks
- Reasoning: plan reflection loop with on_step_failure handling and max_replans is now active
- Reasoning: multi-step plan results are synthesized into a coherent final response

### Changed
- AgentSnapshot now includes an optional persona field with backward-compatible loading

### Fixed
- Reasoning: max_iterations now caps the reasoning loop at the lower of the agent-level and reasoning-level limits
- Reasoning: reflection.pass_threshold now requires both a PASS verdict and confidence at or above the threshold
- Reasoning: state-level reasoning.output, planning config, and max_steps overrides are now respected at runtime
- Reasoning: planning.available.tools and planning.available.skills now filter the planner prompt to the specified IDs
- Reasoning: plan status now reflects step failures instead of always reporting as completed
- Reasoning: planner prompts now include tool descriptions and argument schemas so the LLM generates valid tool arguments
- Reasoning: tool steps with dependencies now use LLM-driven argument generation instead of brittle template substitution
- Orchestration: post-transition re-generation now re-enters full dispatch so transitions into orchestration or reasoning states activate the correct handler in the same turn


---

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
- Spawner template metadata: spawn_agent (rename from `generate_agent` in rc.16) tool auto-discovers template names, descriptions, and variables so the LLM selects the right template without system prompt instructions
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

# Examples

Examples are organized by usage style:

- `yaml/` - YAML-first examples run with `ai-agents-cli`
- `rust/` - Rust examples for embedding, extension, and custom integrations

## Quick Start

### Run a YAML example

From the framework root:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/basic/simple_chat.yaml
```

Another example:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/state-machine/support_state_machine.yaml
```

### Run a Rust example

Go to a Rust example directory and run a binary:

```sh
cd examples/rust/basic-api

cargo run --bin simple-chat
```


## YAML Examples

Run YAML examples with:

```sh
cargo run -p ai-agents-cli -- run <path-to-yaml>
```

Some YAML files include optional `metadata.cli` fields such as welcome text, hints, and display preferences. These only affect CLI presentation.

### `yaml/basic/`

Minimal getting-started examples.

| File | Description |
|------|-------------|
| `simple_chat.yaml` | Smallest YAML-first chat agent |
| `simple_chat_stream.yaml` | Minimal streaming chat example |
| `simple_tools.yaml` | Minimal built-in tools example |

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/basic/simple_chat.yaml
cargo run -p ai-agents-cli -- run examples/yaml/basic/simple_chat_stream.yaml
cargo run -p ai-agents-cli -- run examples/yaml/basic/simple_tools.yaml
```

### `yaml/skills/`

Skill examples - from a single inline skill to multi-step tool pipelines.

| File | Description |
|------|-------------|
| `skill_inline_only.yaml` | Single inline skill with LLM-based trigger routing |
| `skill_external_only.yaml` | Loads a skill from a separate file for cross-agent reusability |
| `skill_with_tools.yaml` | Skills that chain multiple tool calls and LLM prompts in a single pipeline |
| `skill_agent.yaml` | Combined: inline skills, external skill files, and tool-using skills together |
| `skills/math_helper.skill.yaml` | External math skill (used by `skill_external_only` and `skill_agent`) |
| `skills/weather_clothes.skill.yaml` | External weather/clothing skill (used by `skill_agent`) |

Note: The skill router (LLM) compares user input against each skill's `trigger` description and selects the best match.
Skills that reference tools (e.g., `calculator`) must list those tools in the agent's `tools:` section.

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/skills/skill_inline_only.yaml
cargo run -p ai-agents-cli -- run examples/yaml/skills/skill_external_only.yaml
cargo run -p ai-agents-cli -- run examples/yaml/skills/skill_with_tools.yaml
cargo run -p ai-agents-cli -- run examples/yaml/skills/skill_agent.yaml
```

### `yaml/state-machine/`

Declarative state machine examples - from minimal transitions to production-grade multi-branch routing.

| File | Description |
|------|-------------|
| `two_state_greeting.yaml` | Minimal: 2 states, 1 transition each |
| `guard_transitions.yaml` | Context-based guard transitions (deterministic, no LLM call) |
| `nested_states.yaml` | Hierarchical sub-states with `^` escape and turn timeout |
| `state_with_tools.yaml` | Per-state tool scoping (`tools: []` vs inherit) |
| `state_lifecycle.yaml` | `on_enter` / `on_exit` / `on_reenter` actions in a draft-review workflow, plus a secondary retry path with cooldown |
| `support_state_machine.yaml` | Full customer support workflow with hierarchical technical support, global escalation, and fallback clarification |

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/state-machine/two_state_greeting.yaml
cargo run -p ai-agents-cli -- run examples/yaml/state-machine/guard_transitions.yaml
cargo run -p ai-agents-cli -- run examples/yaml/state-machine/nested_states.yaml
cargo run -p ai-agents-cli -- run examples/yaml/state-machine/state_with_tools.yaml
cargo run -p ai-agents-cli -- run examples/yaml/state-machine/state_lifecycle.yaml
cargo run -p ai-agents-cli -- run examples/yaml/state-machine/support_state_machine.yaml
```

### `yaml/tools/`

Progressive tool usage examples - from basic tool calls to multi-tool composition.

| File | Description |
|------|-------------|
| `basic_tools.yaml` | Calculator and DateTime - LLM auto-selects the right tool |
| `text_and_json.yaml` | Unicode-aware text processing and structured JSON operations |
| `file_and_template.yaml` | File I/O and Jinja2 template rendering |
| `math_and_random.yaml` | Statistical math and random value generation |
| `multi_tool_agent.yaml` | All built-in tools with parallel execution |
| `http_tool.yaml` | External HTTP calls (makes real network requests) |
| `mcp_agent.yaml` | MCP-backed filesystem tool with views - one MCP server scoped into `fs_read` and `fs_write` view tools for per-state least-privilege access |

Note: The `system_prompt` in these examples intentionally does NOT list tool names or descriptions.
The framework auto-injects tool information (names, descriptions, argument schemas) into the prompt at runtime.
The system prompt focuses on behavioral guidance only.

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/tools/basic_tools.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/text_and_json.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/file_and_template.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/math_and_random.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/multi_tool_agent.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/http_tool.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/mcp_agent.yaml
```

### `yaml/process/`

Declarative input/output processing pipeline - preprocessing before the LLM, post-processing after.

| File | Description |
|------|-------------|
| `input_normalize.yaml` | Simplest pipeline - whitespace cleanup and length validation (no LLM cost) |
| `detect_language.yaml` | LLM-based language and sentiment detection stored in context |
| `extract_and_validate.yaml` | Structured entity extraction with typed schema and validation rules |
| `output_sanitize.yaml` | Output PII masking, quality validation, and response formatting |

Note: Input stages run before the LLM; output stages run after. LLM-based stages use the router (fast/cheap) model.
Output processing only works in blocking mode - with `--stream`, tokens are printed before output stages run.

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/process/input_normalize.yaml
cargo run -p ai-agents-cli -- run examples/yaml/process/detect_language.yaml
cargo run -p ai-agents-cli -- run examples/yaml/process/extract_and_validate.yaml
cargo run -p ai-agents-cli -- run examples/yaml/process/output_sanitize.yaml
```

### `yaml/context/`

Dynamic context injection examples - from runtime values to environment variables and state integration.

| File | Description |
|------|-------------|
| `runtime_context.yaml` | Inject user data at runtime - system prompt adapts via `{{ context.user.* }}` |
| `builtin_context.yaml` | Built-in sources (datetime, session, agent info) with auto-refresh |
| `env_context.yaml` | Environment variable injection - config and secrets without hardcoding |
| `template_context.yaml` | Jinja2 conditionals, defaults, and filters for tier-based behavior |
| `context_with_state.yaml` | Context + state machine - personalized multi-step support flow |

Note: The CLI does not currently support injecting runtime context values at startup.
All `runtime` context sources in these examples include `default:` blocks so they work out of the box.
In a Rust host, use `agent.set_context("user", json!({...}))` to override defaults.
For a full Rust context injection example, see `rust/context/` below.

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/context/runtime_context.yaml
cargo run -p ai-agents-cli -- run examples/yaml/context/builtin_context.yaml
cargo run -p ai-agents-cli -- run examples/yaml/context/env_context.yaml
cargo run -p ai-agents-cli -- run examples/yaml/context/template_context.yaml
cargo run -p ai-agents-cli -- run examples/yaml/context/context_with_state.yaml
```

### `yaml/memory/`

Progressive memory examples - from simplest to production-grade.

| File | Description |
|------|-------------|
| `memory_basic.yaml` | Simplest memory - in-memory storage with a message limit |
| `memory_compacting.yaml` | Compacting memory with automatic LLM-based summarization |
| `memory_budget.yaml` | Token budgeting - per-component allocation controlling prompt size |
| `memory_agent.yaml` | Full production config combining compacting, budgeting, and hooks |

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/memory/memory_basic.yaml
cargo run -p ai-agents-cli -- run examples/yaml/memory/memory_compacting.yaml
cargo run -p ai-agents-cli -- run examples/yaml/memory/memory_budget.yaml
cargo run -p ai-agents-cli -- run examples/yaml/memory/memory_agent.yaml
```

For session persistence (save/restore across restarts), see `rust/storage/` below.

### `yaml/error-recovery/`

Production-essential error recovery - from automatic retries to LLM failover and context overflow handling.

| File | Description |
|------|-------------|
| `basic_retry.yaml` | Automatic retry with exponential backoff on transient errors |
| `llm_fallback.yaml` | Fall back to a different LLM when the primary fails |
| `context_overflow.yaml` | Summarize or truncate when the conversation exceeds the context window |

Note: Error recovery is transparent and works behind the scenes.
Set `RUST_LOG=ai_agents_recovery=warn` to see retry attempts and fallback activations, or `RUST_LOG=debug` for context overflow and summarization details.
`context_overflow.yaml` uses a deliberately low token limit (2048) so overflow triggers within a few turns.

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/error-recovery/basic_retry.yaml
cargo run -p ai-agents-cli -- run examples/yaml/error-recovery/llm_fallback.yaml
cargo run -p ai-agents-cli -- run examples/yaml/error-recovery/context_overflow.yaml
```

### `yaml/hitl/`

Declarative human-in-the-loop approval - from a single tool requiring sign-off to localized multi-language approval messages.

| File | Description |
|------|-------------|
| `hitl_basic.yaml` | Every HTTP call requires y/N approval before execution |
| `hitl_conditions.yaml` | GET proceeds freely; POST/PUT/DELETE/PATCH requires approval |
| `hitl_multilingual.yaml` | Context-driven localized approval messages -- process pipeline detects user language, HITL picks the matching translation (en/ko/ja) |

Note: The CLI prompts interactively in the terminal. Use `metadata.cli.hitl.style: auto_approve` to bypass prompts in demos, or `auto_reject` to test rejection paths without user input.
For a custom approval handler (Slack, webhook, email), see `rust/custom-hitl/` below.

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/hitl/hitl_basic.yaml
cargo run -p ai-agents-cli -- run examples/yaml/hitl/hitl_conditions.yaml
cargo run -p ai-agents-cli -- run examples/yaml/hitl/hitl_multilingual.yaml
```

### `yaml/reasoning/`

Progressive reasoning and reflection examples - from single-mode isolation to per-state overrides.

| File | Description |
|------|-------------|
| `reasoning_cot.yaml` | Chain-of-thought with tagged output and visible step-by-step thinking |
| `reasoning_plan.yaml` | Plan-and-execute with planner LLM, tool filtering, and plan-level reflection (replan on failure) |
| `reasoning_reflection.yaml` | Self-evaluation with domain-specific criteria, confidence threshold, and retry loop |
| `reasoning_with_state.yaml` | Per-state reasoning overrides - full replacement semantics (not merge) |

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/reasoning/reasoning_cot.yaml
cargo run -p ai-agents-cli -- run examples/yaml/reasoning/reasoning_plan.yaml
cargo run -p ai-agents-cli -- run examples/yaml/reasoning/reasoning_reflection.yaml
cargo run -p ai-agents-cli -- run examples/yaml/reasoning/reasoning_with_state.yaml
```

### `yaml/disambiguation/`

LLM-based intent disambiguation and clarification - the agent asks before acting on vague input. No regex, works in any language.

| File | Description |
|------|-------------|
| `disambiguation_basic.yaml` | Enable disambiguation in 4 lines, clarification flow, social skip |
| `disambiguation_with_state.yaml` | State machine + `intent:` labels for deterministic routing after disambiguation |
| `disambiguation_multilingual.yaml` | Multi-language detection + skill-level overrides with `clarification_templates` |
| `disambiguation_agent.yaml` | Full config with all aspects, context-aware detection, and all skip rules |

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/disambiguation/disambiguation_basic.yaml
cargo run -p ai-agents-cli -- run examples/yaml/disambiguation/disambiguation_with_state.yaml
cargo run -p ai-agents-cli -- run examples/yaml/disambiguation/disambiguation_multilingual.yaml
cargo run -p ai-agents-cli -- run examples/yaml/disambiguation/disambiguation_agent.yaml
```

### `yaml/spawner/`

Dynamic agent spawning - create, message, list, and remove agents at runtime from a parent agent.

| File | Description |
|------|-------------|
| `game_master.yaml` | Game master that spawns NPC agents on demand using four spawner tools (`generate_agent`, `send_message`, `list_agents`, `remove_agent`) with shared LLMs, named templates, and auto-naming |
| `team_manager.yaml` | Team manager that spawns specialist agents (researcher, writer) with shared SQLite storage, tool allowlist, and multi-template LLM selection |

Note: The spawner tools are auto-registered by `AgentBuilder::auto_configure_spawner()` when the YAML has a `spawner:` section.

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/spawner/game_master.yaml
cargo run -p ai-agents-cli -- run examples/yaml/spawner/team_manager.yaml
```

### `yaml/persona/`

Agent persona - structured identity, personality traits, evolution, and context-conditional secrets.

| File | Description |
|------|-------------|
| `persona_basic.yaml` | Minimal persona with identity (name, role, affiliation), personality traits, speaking style, goals, and hidden goals that coexist with `system_prompt` |
| `persona_evolution.yaml` | Evolvable persona where `traits.personality`, `traits.speaking_style`, and `goals.primary` can be mutated at runtime via Rust API or the auto-registered `persona_evolve` tool |
| `persona_secrets.yaml` | Persona with context-conditional secrets revealed only when `ContextManager` values satisfy typed conditions (`gte`, `eq`, `all`, `any`) - includes runtime context defaults for CLI testing |

Note: Persona is prepended to the system prompt automatically. It survives `prompt_mode: replace` in state machines. The `persona_evolve` tool is auto-registered only when `evolution.allow_llm_evolve: true`. Secrets with no `reveal_conditions` never auto-reveal (API-only access).

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/persona/persona_basic.yaml
cargo run -p ai-agents-cli -- run examples/yaml/persona/persona_evolution.yaml
cargo run -p ai-agents-cli -- run examples/yaml/persona/persona_secrets.yaml
```

### `yaml/session/`

Cross-session actor memory and key facts extraction - the agent remembers structured facts about each actor across separate sessions without any application code.

| File | Description |
|------|-------------|
| `facts_basic.yaml` | Minimal facts extraction setup. Shows `memory.facts` categories, `auto_extract`, `dedup`, and `{{ actor_facts }}` template injection. Run twice with the same `--actor` to verify facts survive the session. |
| `cross_session.yaml` | Full cross-session memory with `actor_memory` + `facts` + `session` blocks. Demonstrates `injection.mode`, `privacy`, token budget allocation for facts, and session TTL. Run twice with `--actor customer_42` to see prior facts loaded on the second session. |

Note: Facts are extracted automatically after each turn by the router LLM and stored in English for consistent cross-language deduplication. SQLite is the only backend that persists facts and session metadata - file and Redis backends accept the configuration but use no-op storage. The `--actor` flag sets the actor ID explicitly; `identification.method: from_context` reads it from a dotted context path on every turn so a game engine or multi-tenant app can switch actors by updating the context value. Use `/facts` or `/actor facts` in the REPL to inspect extracted facts. Use `/cleanup` to remove sessions past their TTL.

Examples:

```sh
# Session 1: tell the agent something about yourself
cargo run -p ai-agents-cli -- run examples/yaml/session/facts_basic.yaml --actor user_1

# Session 2: the agent remembers without being told again
cargo run -p ai-agents-cli -- run examples/yaml/session/facts_basic.yaml --actor user_1

# Full cross-session memory demo (run twice with the same actor)
cargo run -p ai-agents-cli -- run examples/yaml/session/cross_session.yaml --actor customer_42
```

### `yaml/orchestration/`

Multi-agent orchestration patterns using pre-spawned sub-agents. All five patterns (router, pipeline, concurrent, group chat, handoff) have dedicated state types.

| File | Description |
|------|-------------|
| `customer_support_router.yaml` | Router pattern - routing state delegates to billing or technical sub-agents via LLM-evaluated transitions |
| `content_pipeline.yaml` | Pipeline state - writer, reviewer, editor run sequentially in one state with per-stage input templates |
| `stock_analysis_concurrent.yaml` | Concurrent state - three analysts run in parallel, results aggregated via LLM synthesis |
| `code_review_group_chat.yaml` | Group chat state - architect, security, and performance reviewers discuss until consensus |
| `support_handoff.yaml` | Handoff state - LLM-directed agent-to-agent control transfer between general, technical, and billing |
| `team_coordinator.yaml` | Orchestration tools - coordinator LLM picks which tool and agents to use for each request (route, pipeline, concurrent, group discussion, handoff) |
| `agents/*.yaml` | Sub-agent stubs - general, billing, technical, writer, reviewer, editor, researcher, analyst, 3 analysts, architect, security reviewer, performance reviewer |

Note: Orchestration uses `spawner.auto_spawn` to create sub-agents at startup. Each sub-agent is a standalone YAML file in `agents/`. Delegate states forward messages to registry agents. Pipeline, concurrent, group chat, and handoff states run entirely within a single `chat()` call. The parent's transition evaluator watches orchestration responses to decide when to move on.

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/orchestration/customer_support_router.yaml
cargo run -p ai-agents-cli -- run examples/yaml/orchestration/content_pipeline.yaml
cargo run -p ai-agents-cli -- run examples/yaml/orchestration/stock_analysis_concurrent.yaml
cargo run -p ai-agents-cli -- run examples/yaml/orchestration/code_review_group_chat.yaml
cargo run -p ai-agents-cli -- run examples/yaml/orchestration/support_handoff.yaml
cargo run -p ai-agents-cli -- run examples/yaml/orchestration/team_coordinator.yaml
```

## Rust Examples

Rust examples are for embedding and extension scenarios.

### `rust/basic-api/`

Beginner Rust entrypoints that show embedding and code-first usage after the YAML basics.

| Binary | Description |
|--------|-------------|
| `simple-chat` | Smallest Rust-first agent using `AgentBuilder::new()` and a single LLM |
| `tool-agent` | Rust-built agent that demonstrates built-in tools and interactive tool-call visibility |
| `yaml-loader` | Minimal Rust example that loads and runs a YAML-defined agent |
| `streaming-chat` | Minimal Rust example that enables streaming mode from the builder and streams output in the REPL |

Run from:

```sh
cd examples/rust/basic-api
cargo run --bin simple-chat
cargo run --bin tool-agent
cargo run --bin yaml-loader
cargo run --bin streaming-chat
```

### `rust/state-machine/`

Rust wrapper around a YAML-defined state machine example.

| Binary | Description |
|--------|-------------|
| `state-machine-agent` | Multi-branch support flow with hierarchical states |

Run from:

```sh
cd examples/rust/state-machine
cargo run --bin state-machine-agent
```

### `rust/storage/`

Programmatic memory and persistence examples.

| Binary | Description |
|--------|-------------|
| `save-restore-session` | Minimal session persistence - `/save` and `/load` only |
| `memory-agent` | Compacting memory with hooks monitoring compression and budget warnings |
| `sqlite-persistence` | Full session CRUD - save, load, list, search, delete, info |

Run from:

```sh
cd examples/rust/storage
cargo run --bin save-restore-session
cargo run --bin memory-agent
cargo run --bin sqlite-persistence
```

### `rust/context/`

Rust-side context injection - `set_context()` and custom `ContextProvider` implementation.

| Binary | Description |
|--------|-------------|
| `context-injection` | Overrides YAML defaults with runtime user data and registers a callback provider for live usage stats |

Run from:

```sh
cd examples/rust/context
cargo run --bin context-injection
```

### `rust/custom-hitl/`

Custom approval handler examples - from a minimal y/N handler to a full modify-capable implementation.

| Binary | Description |
|--------|-------------|
| `simple-approval` | Minimal `ApprovalHandler` implementation - one method, y/N only |
| `hitl-agent` | Full handler with approve/reject/modify support and multi-language messages |

Note: When the handler returns `Modified { changes }`, the runtime merges the new values into the tool arguments before execution.
For example, changing a payment amount from $5000 to $500 in the modify prompt updates the actual tool call.
The `simple-approval` handler supports y/N only; `hitl-agent` demonstrates the full approve/reject/modify flow.

Run from:

```sh
cd examples/rust/custom-hitl
cargo run --bin simple-approval
cargo run --bin hitl-agent
```

### `rust/custom-llm/`

Custom LLM provider examples - from implementing `LLMProvider` from scratch to multi-provider routing.

| Binary | Description |
|--------|-------------|
| `custom-provider` | Implement `LLMProvider` from scratch with an offline echo/rule-based provider - no API key needed |
| `openai-compatible` | HTTP adapter for any OpenAI-compatible server (LM Studio, Ollama, vLLM, LocalAI, TGI) |
| `multi-provider` | Multi-provider routing with `MultiLLMRouter` - expensive model for users, cheap model for internal tasks |

Note: For a zero-code YAML alternative, see `yaml/basic/openai_compatible.yaml` which uses the built-in `provider: openai-compatible` with `base_url` - no custom Rust code needed.

Run from:

```sh
cd examples/rust/custom-llm

# No API key needed - runs entirely offline
cargo run --bin custom-provider

# Requires a running OpenAI-compatible server
LOCAL_LLM_BASE_URL=http://localhost:1234/v1 cargo run --bin openai-compatible

# OPENAI_API_KEY
cargo run --bin multi-provider
```

### `rust/custom-tools/`

Custom tool examples - from a minimal `Tool` trait implementation to a full `ToolProvider` with dynamic discovery.

| Binary | Description |
|--------|-------------|
| `simple-tool` | Minimal `Tool` trait - 5 methods, hand-written JSON Schema, `.tool()` registration |
| `schema-tool` | Auto-generated `input_schema` via `schemars::JsonSchema` - no hand-written JSON |
| `stateful-tool` | Tool with mutable state across calls using `RwLock` (interior mutability pattern) |
| `yaml-custom-tool` | YAML-defined agent + Rust domain tool injection (recommended production pattern) |
| `tool-provider` | Custom `ToolProvider` - dynamic tool discovery, health checks, multi-language aliases |

Run from:

```sh
cd examples/rust/custom-tools

cargo run --bin simple-tool
cargo run --bin schema-tool
cargo run --bin stateful-tool
cargo run --bin yaml-custom-tool
cargo run --bin tool-provider
```

### `rust/disambiguation/`

Advanced disambiguation demo with Rust-side startup options and metadata display.
Start with `yaml/disambiguation/` for the core concepts.

| Binary | Description |
|--------|-------------|
| `disambiguation-agent` | Override clarification style and fallback at runtime, display disambiguation metadata via AgentHooks |

Run from:

```sh
cd examples/rust/disambiguation
cargo run --bin disambiguation-agent
```

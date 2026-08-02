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


## Evaluation Examples

Evaluation suites run an agent against declarative scenarios and write reports for CI, debugging, and release checks. Suites are organized by run mode first, then by the YAML example category.

```text
eval/
├── mocked/              # deterministic no-key suites for CI and regression checks
├── live/examples/       # intentional real-provider smoke checks for runnable examples
├── live/quality/        # intentional real-provider semantic or judge checks
└── replay/              # optional cassette/replay artifacts and notes
```

Use `eval/mocked/**/*.yaml` for default no-key checks. Do not glob all of `eval/**/*.yaml` for CI unless live provider calls are intentional.

### Mocked no-key suites

Mocked suites cover every runnable YAML example category: basic, context, disambiguation, error-recovery, HITL, memory, observability, orchestration, persona, process, reasoning, relationships, runtime-optimization, session, skills, spawner, state-machine, and tools. Each suite uses a mock LLM provider, requires no API keys, and runs deterministically. Suites verify structural runtime evidence such as tool calls, state transitions, context paths, orchestration metadata, observability counts, relationship scores, fact extraction, disambiguation status, and persona secret gating.

Run all mocked suites with the convenience helper:

```sh
sh examples/eval/mocked/run_mocked_evals.sh
sh examples/eval/mocked/run_mocked_evals.sh --category state-machine
sh examples/eval/mocked/run_mocked_evals.sh --list
```

Or run a single suite directly:

```sh
cargo run -p ai-agents-cli -- eval \
  --scenarios examples/eval/mocked/basic/simple_chat_mocked.yaml \
  --output target/eval/mocked/basic/simple_chat_mocked
```

### Live suites

Live suites under `eval/live/examples/` drive runnable YAML examples with a real provider and combine meaningful response checks with structural evidence for tools, skills, threshold-aware disambiguation, cross-runtime fact persistence, multi-actor isolation, public plan-and-execute outcomes, exact state lifecycle behavior, observability, recovery, context injection, and input/output processing. External dependencies remain read-only, fixture-backed, no-socket, or dry-run-only. These suites require provider credentials, may incur cost, and are intended for intentional release smoke checks rather than default no-key CI.

Semantic or judge-based live checks live under `eval/live/quality/` so they do not mix with example smoke checks. The live example helper intentionally excludes that quality directory. Structural tool-execution suites use explicit required tool choice and `executed: true` evidence; automatic tool discovery remains provider/model quality evidence.

See `eval/live/README.md` for the full registry, risk tags, status vocabulary, category helper, and run commands.

```sh
# Parse-check one live category without provider calls.
sh examples/eval/live/run_live_example_evals.sh --dry-config-check --category context

# Run one category intentionally with a real provider.
sh examples/eval/live/run_live_example_evals.sh --yes-live --category process

# Run one suite directly.
cargo run -p ai-agents-cli -- eval \
  --scenarios examples/eval/live/examples/tools/code_search_live.yaml \
  --output target/eval/live/examples/tools/code_search \
  --real-llm

# Live provider quality example. Requires an API key and may incur provider cost.
cargo run -p ai-agents-cli -- eval \
  --agent examples/yaml/basic/simple_chat.yaml \
  --scenarios examples/eval/live/quality/basic/simple_chat_semantic_judge_live.yaml \
  --output target/eval/live/quality/basic/simple_chat_semantic_judge \
  --real-llm
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
| `simple_chat.yaml` | Smallest YAML-first chat agent with explicit `tools: []` |
| `simple_chat_stream.yaml` | Minimal streaming chat example |
| `simple_tools.yaml` | Minimal built-in tools example |
| `openai_compatible.yaml` | Connect to any OpenAI-compatible server such as LM Studio, vLLM, TGI, LocalAI, or Ollama `/v1` |
| `ollama_chat.yaml` | Local Ollama chat with explicit context window and keep-alive settings |

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/basic/simple_chat.yaml
cargo run -p ai-agents-cli -- run examples/yaml/basic/simple_chat_stream.yaml
cargo run -p ai-agents-cli -- run examples/yaml/basic/simple_tools.yaml
cargo run -p ai-agents-cli -- run examples/yaml/basic/openai_compatible.yaml
cargo run -p ai-agents-cli -- run examples/yaml/basic/ollama_chat.yaml
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
| `state_with_tools.yaml` | Top-level maximum tool grants with per-state narrowing |
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

Top-level `tools:` is an explicit ordinary-tool grant. Omitted or empty top-level `tools:` means no ordinary LLM-callable tools. Explicit feature flags such as orchestration tools and persona evolution can also grant their generated tools. State-level `tools` can narrow the effective grant but cannot expose tools that were not granted.

| File | Description |
|------|-------------|
| `basic_tools.yaml` | Calculator and DateTime - LLM auto-selects the right tool |
| `workspace_research.yaml` | Read-only workspace discovery with `glob`, `grep`, `file_list`, `file_read`, `file_info`, and `todo` |
| `code_search.yaml` | Read-only docs and code search with `glob`, `grep`, `file_list`, `file_read`, and `file_info` |
| `repo_review.yaml` | Read-only repository inspection with `git_status`, `git_diff`, and optional `file_read` follow-up |
| `file_write_sandbox.yaml` | Create or overwrite a sandboxed file with `file_write`, dry-run review, and explicit write policy |
| `file_edit_review.yaml` | Exact file replacement with `file_edit`, read-before-write policy, dry-run review, and HITL-ready apply |
| `patch_review.yaml` | Unified diff validation with `patch`, bounded file and line caps, and HITL-ready apply |
| `command_validation.yaml` | Exact allowlisted validation commands with `command`, bounded output, and explicit working directories |
| `interactive_choice.yaml` | Structured user follow-up questions through `ask_user` in plain REPL or TUI modal form |
| `todo_workflow.yaml` | Session-local planning and progress tracking with the `todo` tool |
| `sleep_wait.yaml` | Short policy-bounded waits with the `sleep` tool and no shell access |
| `diagnostics_review.yaml` | Host-provided diagnostics review; returns unavailable when no diagnostics provider is installed |
| `web_fetch_research.yaml` | Bounded public web fetch with redirect checks, DNS/IP safety, and optional extraction prompt |
| `text_and_json.yaml` | Unicode-aware text processing and structured JSON operations |
| `file_and_template.yaml` | Legacy file I/O plus Jinja2 template rendering |
| `copy_review.yaml` | Copy a file or directory with `copy_path`, dry-run review, and explicit write policy |
| `move_review.yaml` | Move or rename a file or directory with `move_path`, dry-run review, and source/destination policy |
| `delete_review.yaml` | Delete a file or directory with `delete_path`, recursive-delete gating, dry-run review, and explicit write policy |
| `web_search_research.yaml` | Provider-neutral web search through a host-provided search provider with `web_fetch` fallback when no provider is installed |
| `math_and_random.yaml` | Statistical math and random value generation with required tool selection |
| `multi_tool_agent.yaml` | Selected general-purpose built-ins with parallel execution and required tool selection |
| `http_tool.yaml` | Raw HTTP API client tool for external API calls (makes real network requests) |
| `mcp_agent.yaml` | MCP-backed filesystem tool with views - one MCP server scoped into `fs_read` and `fs_write` view tools for per-state least-privilege access |

Note: The `system_prompt` in these examples intentionally does NOT list tool names or descriptions.
The framework auto-injects tool information (names, descriptions, argument schemas) into the prompt at runtime.
The system prompt focuses on behavioral guidance only.
Security, HITL, recovery, and eval use canonical tool IDs after display names or aliases are resolved.

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/tools/basic_tools.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/workspace_research.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/code_search.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/repo_review.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/file_write_sandbox.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/file_edit_review.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/patch_review.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/command_validation.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/interactive_choice.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/todo_workflow.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/sleep_wait.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/diagnostics_review.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/web_fetch_research.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/text_and_json.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/file_and_template.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/copy_review.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/move_review.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/delete_review.yaml
cargo run -p ai-agents-cli -- run examples/yaml/tools/web_search_research.yaml
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

Note: The CLI supports runtime context at startup with `--context key=value` and `--context-file path.json`.
Most `runtime` context sources in these examples still include `default:` blocks so they work out of the box.
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

### `yaml/observability/`

Observability and tracing examples - latency, token usage, cost estimates, privacy-safe reports, and metric breakdowns.

| File | Description |
|------|-------------|
| `basic_metrics.yaml` | Safe default metrics with JSON report export and no raw prompt or response retention |
| `cost_by_model.yaml` | Router/main model split with cost grouped by model and purpose |
| `pricing_file.yaml` | Load shared model prices from `pricing.yaml`, with inline override behavior |
| `language_breakdown.yaml` | Process pipeline language detection with metrics grouped by detected language |
| `tools_and_skills_metrics.yaml` | Skill routing, skill prompt, and tool-call telemetry in one run |
| `orchestration_metrics.yaml` | Parent and delegated child agent traces in a multi-agent support flow |

Note: Observability is disabled by default. These examples write reports under `target/observability/` after each turn. Raw prompts, responses, tool args, and tool outputs stay off unless a `privacy.include_*` field explicitly enables them.

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/observability/basic_metrics.yaml --plain
cargo run -p ai-agents-cli -- run examples/yaml/observability/cost_by_model.yaml --plain
cargo run -p ai-agents-cli -- run examples/yaml/observability/pricing_file.yaml --plain
cargo run -p ai-agents-cli -- run examples/yaml/observability/language_breakdown.yaml --plain
cargo run -p ai-agents-cli -- run examples/yaml/observability/tools_and_skills_metrics.yaml --plain
cargo run -p ai-agents-cli -- run examples/yaml/observability/orchestration_metrics.yaml --plain
```

### `yaml/runtime-optimization/`

Runtime latency optimization examples - safe opt-in routing, speculative branch execution, and post-turn maintenance policies.

| File | Description |
|------|-------------|
| `pre_response_transition.yaml` | Pre-response deterministic state transition that skips a stale old-state LLM response when a guard already proves the next state |
| `pre_response_transition_disabled.yaml` | Baseline post-response guard transition for comparing optimized and default routing |
| `pre_response_extractor.yaml` | Pre-response extractor routing with staged context writes that commit only after the route wins |
| `parallel_transition.yaml` | Speculative main draft plus response-independent transition decision with losing branch discard |
| `speculative_skill_routing.yaml` | Speculative skill selection that executes skill steps only after the skill branch wins |
| `speculative_reasoning_auto.yaml` | Speculative auto reasoning that commits a plain draft when the judge selects no reasoning |
| `buffered_streaming.yaml` | Streaming-safe parallel transition routing that hides stale branch output until routing is resolved |
| `losing_tool_draft.yaml` | Speculative transition winner that proves parsed tool calls in a losing draft remain inert |
| `post_turn_maintenance.yaml` | Background facts and relationship maintenance with same-actor freshness for the next turn |
| `orchestration_vote_order.yaml` | Bounded parallel vote extraction that keeps declaration-order tie-breaking stable |

Note: Runtime optimization is disabled by default. Pre-response transitions must be explicitly marked with `timing: pre_response` and only work for response-independent routes such as guards or resolved intents. Speculative branches require a positive `max_speculative_llm_calls_per_turn` and can increase token use because losing branches are still observed and discarded. The speculative examples enable branch observability reports under `target/observability/` so committed, discarded, failed, and cancelled outcomes are inspectable. Background actor memory is eventually consistent unless the relevant task policy waits for the same actor or all turns.

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/runtime-optimization/pre_response_transition.yaml --context request.topic=billing --plain
cargo run -p ai-agents-cli -- run examples/yaml/runtime-optimization/pre_response_transition_disabled.yaml --context request.topic=billing --plain
cargo run -p ai-agents-cli -- run examples/yaml/runtime-optimization/pre_response_extractor.yaml --plain
cargo run -p ai-agents-cli -- run examples/yaml/runtime-optimization/parallel_transition.yaml --plain
cargo run -p ai-agents-cli -- run examples/yaml/runtime-optimization/speculative_skill_routing.yaml --plain
cargo run -p ai-agents-cli -- run examples/yaml/runtime-optimization/speculative_reasoning_auto.yaml --plain
cargo run -p ai-agents-cli -- run examples/yaml/runtime-optimization/buffered_streaming.yaml --stream --plain
cargo run -p ai-agents-cli -- run examples/yaml/runtime-optimization/losing_tool_draft.yaml --plain
cargo run -p ai-agents-cli -- run examples/yaml/runtime-optimization/post_turn_maintenance.yaml --actor user_1 --plain
cargo run -p ai-agents-cli -- run examples/yaml/runtime-optimization/orchestration_vote_order.yaml --plain
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
| `game_master.yaml` | Game master that spawns NPC agents on demand using four management tools (`spawn_agent`, `send_agent_message`, `list_agents`, `remove_agent`) with shared LLMs, named templates, and auto-naming |
| `team_manager.yaml` | Team manager that spawns specialist agents (researcher, writer) with shared SQLite storage, tool allowlist, and multi-template LLM selection |

Note: Spawner management tools are registered by `AgentBuilder::auto_configure_spawner()` and granted when `spawner.management_tools` is enabled.

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
| `persona_evolution.yaml` | Evolvable persona where `traits.personality`, `traits.speaking_style`, and `goals.primary` can be mutated at runtime via Rust API or the auto-registered and granted `persona_evolve` tool |
| `persona_secrets.yaml` | Persona with context-conditional secrets revealed only when `ContextManager` values satisfy typed conditions (`gte`, `eq`, `all`, `any`) - includes runtime context defaults for CLI testing |

Note: Persona is prepended to the system prompt automatically. It survives `prompt_mode: replace` in state machines. The `persona_evolve` tool is registered and granted only when `evolution.allow_llm_evolve: true`. Secrets with no `reveal_conditions` never auto-reveal (API-only access).

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/persona/persona_basic.yaml
cargo run -p ai-agents-cli -- run examples/yaml/persona/persona_evolution.yaml
cargo run -p ai-agents-cli -- run examples/yaml/persona/persona_secrets.yaml
```

### `yaml/relationships/`

Actor-scoped relationship memory - agents track trust, sentiment, familiarity, rapport, and notable relationship events for each actor across sessions.

| File | Description |
|------|-------------|
| `support_relationship.yaml` | General-purpose support agent that adapts to returning actors using persisted trust, sentiment, familiarity, and rapport |
| `two_sided_relationship.yaml` | Two-sided relationship model showing `agent_to_actor`, `perceived_actor_to_agent`, and derived `mutual` scores |
| `persona_trust_secret.yaml` | Persona secret reveal pattern where `relationships.current_actor.trust` controls whether confidential information enters the prompt |

Note: Relationship memory is separate from facts. Facts capture what the agent knows about an actor. Relationships capture how the agent relates to that actor. Relationship values are injected into context at `relationships.current_actor.*` and can be used by persona secrets, state guards, tools, and prompt templates.

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/relationships/support_relationship.yaml --actor customer_42
cargo run -p ai-agents-cli -- run examples/yaml/relationships/two_sided_relationship.yaml --actor customer_42
cargo run -p ai-agents-cli -- run examples/yaml/relationships/persona_trust_secret.yaml --actor visitor_1
```

### `yaml/session/`

Cross-session actor memory and key facts extraction - the agent remembers structured facts about each actor across separate sessions without any application code.

| File | Description |
|------|-------------|
| `facts_basic.yaml` | Minimal facts extraction setup. Shows `memory.facts` categories, `auto_extract`, `dedup`, and `{{ actor_facts }}` template injection. Run twice with the same `--actor` to verify facts survive the session. |
| `cross_session.yaml` | Full cross-session memory with `actor_memory` + `facts` + `session` blocks. Demonstrates `injection.mode`, `privacy`, token budget allocation for facts, and session TTL. Run twice with `--actor customer_42` to see prior facts loaded on the second session. |
| `multi_actor.yaml` | NPC guard that tracks facts about each player independently using `identification.method: from_context`. Custom categories (suspicion, favor, contraband) show how to extend the built-in category set for domain-specific extraction. |

Note: Facts are extracted automatically after each turn by the router LLM and stored in English for consistent cross-language deduplication. SQLite is the only backend that persists facts and session metadata - file and Redis backends accept the configuration but use no-op storage. The `--actor` flag sets the actor ID explicitly; `identification.method: from_context` reads it from a dotted context path on every turn so a game engine or multi-tenant app can switch actors by updating the context value. Use `/facts` or `/actor facts` in the REPL to inspect extracted facts. Use `/cleanup` to remove sessions past their TTL.

Examples:

```sh
# Session 1: tell the agent something about yourself
cargo run -p ai-agents-cli -- run examples/yaml/session/facts_basic.yaml --actor user_1

# Session 2: the agent remembers without being told again
cargo run -p ai-agents-cli -- run examples/yaml/session/facts_basic.yaml --actor user_1

# Full cross-session memory demo (run twice with the same actor)
cargo run -p ai-agents-cli -- run examples/yaml/session/cross_session.yaml --actor customer_42

# NPC with context-based actor switching (player_1 and player_2 have separate fact stores)
cargo run -p ai-agents-cli -- run examples/yaml/session/multi_actor.yaml \
  --context player.id=player_1 --context player.name=Aldric
cargo run -p ai-agents-cli -- run examples/yaml/session/multi_actor.yaml \
  --context player.id=player_2 --context player.name=Serafine
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

### `rust/observability/`

Programmatic observability access from a Rust host.

| Binary | Description |
|--------|-------------|
| `report` | Loads an observed YAML agent, runs a few turns, prints `ObservabilityReport`, and exports files |

Run from:

```sh
cd examples/rust/observability
cargo run --bin report
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

Custom tool examples - from a context-aware `Tool::execute(args, ctx)` implementation to a full `ToolProvider` with dynamic discovery.

| Binary | Description |
|--------|-------------|
| `simple-tool` | Minimal `Tool` trait, hand-written JSON Schema, `.tool()` registration |
| `schema-tool` | Auto-generated `input_schema` via `schemars::JsonSchema` - no hand-written JSON |
| `stateful-tool` | Tool with mutable state across calls using `RwLock` (interior mutability pattern) |
| `yaml-custom-tool` | YAML-defined agent + Rust domain tool injection (recommended production pattern) |
| `context-tool` | YAML-defined custom tool config exposed through `ToolExecutionContext.custom_config` and `ctx.limits` |
| `tool-provider` | Custom `ToolProvider` - dynamic tool discovery, health checks, multi-language aliases |

Run from:

```sh
cd examples/rust/custom-tools

cargo run --bin simple-tool
cargo run --bin schema-tool
cargo run --bin stateful-tool
cargo run --bin yaml-custom-tool
cargo run --bin context-tool
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

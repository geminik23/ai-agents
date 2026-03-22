# Examples

Examples are organized by usage style:

- `yaml/` — YAML-first examples run with `ai-agents-cli`
- `rust/` — Rust examples for embedding, extension, and custom integrations

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

skill examples - from a single inline skill to multi-step tool pipelines.

| File | Description |
|------|-------------|
| `skill_inline_only.yaml` | Single inline skill with LLM-based trigger routing |
| `skill_external_only.yaml` | Loads a skill from a separate file for cross-agent reusability |
| `skill_with_tools.yaml` | Skills that chain multiple tool calls and LLM prompts in a single pipeline |
| `skill_agent.yaml` | Combined: inline skills, external skill files, and tool-using skills together |
| `skills/math_helper.skill.yaml` | External math skill (used by `skill_external_only` and `skill_agent`) |
| `skills/weather_clothes.skill.yaml` | External weather/clothing skill (used by `skill_agent`) |

The skill router (LLM) compares user input against each skill's `trigger` description and selects the best match.
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

Progressive tool usage examples — from basic tool calls to multi-tool composition.

| File | Description |
|------|-------------|
| `basic_tools.yaml` | Calculator and DateTime — LLM auto-selects the right tool |
| `text_and_json.yaml` | Unicode-aware text processing and structured JSON operations |
| `file_and_template.yaml` | File I/O and Jinja2 template rendering |
| `math_and_random.yaml` | Statistical math and random value generation |
| `multi_tool_agent.yaml` | All built-in tools with parallel execution |
| `http_tool.yaml` | External HTTP calls (makes real network requests) |
| `mcp_agent.yaml` | MCP-backed filesystem tool with views - one MCP server scoped into `fs_read` and `fs_write` view tools for per-state least-privilege access |

Note: the `system_prompt` in these examples intentionally does NOT list tool names or descriptions.
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

Dynamic context injection examples — from runtime values to environment variables and state integration.

| File | Description |
|------|-------------|
| `runtime_context.yaml` | Inject user data at runtime — system prompt adapts via `{{ context.user.* }}` |
| `builtin_context.yaml` | Built-in sources (datetime, session, agent info) with auto-refresh |
| `env_context.yaml` | Environment variable injection — config and secrets without hardcoding |
| `template_context.yaml` | Jinja2 conditionals, defaults, and filters for tier-based behavior |
| `context_with_state.yaml` | Context + state machine — personalized multi-step support flow |

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

Progressive memory examples — from simplest to production-grade.

| File | Description |
|------|-------------|
| `memory_basic.yaml` | Simplest memory — in-memory storage with a message limit |
| `memory_compacting.yaml` | Compacting memory with automatic LLM-based summarization |
| `memory_budget.yaml` | Token budgeting — per-component allocation controlling prompt size |
| `memory_agent.yaml` | Full production config combining compacting, budgeting, and hooks |

Examples:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/memory/memory_basic.yaml
cargo run -p ai-agents-cli -- run examples/yaml/memory/memory_compacting.yaml
cargo run -p ai-agents-cli -- run examples/yaml/memory/memory_budget.yaml
cargo run -p ai-agents-cli -- run examples/yaml/memory/memory_agent.yaml
```

For session persistence (save/restore across restarts), see `rust/storage/` below.

### `yaml/reasoning/`

Reasoning and reflection examples.

| File | Description |
|------|-------------|
| `reasoning_agent.yaml` | Reasoning-enabled YAML agent with reflection configuration |

Example:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/reasoning/reasoning_agent.yaml
```

### `yaml/disambiguation/`

Intent disambiguation and clarification examples.

| File | Description |
|------|-------------|
| `disambiguation_agent.yaml` | LLM-based ambiguity detection and clarification flow |

Example:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/disambiguation/disambiguation_agent.yaml
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
| `save-restore-session` | Minimal session persistence — `/save` and `/load` only |
| `memory-agent` | Compacting memory with hooks monitoring compression and budget warnings |
| `sqlite-persistence` | Full session CRUD — save, load, list, search, delete, info |

Run from:

```sh
cd examples/rust/storage
cargo run --bin save-restore-session
cargo run --bin memory-agent
cargo run --bin sqlite-persistence
```

### `rust/context/`

Rust-side context injection — `set_context()` and custom `ContextProvider` implementation.

| Binary | Description |
|--------|-------------|
| `context-injection` | Overrides YAML defaults with runtime user data and registers a callback provider for live usage stats |

Run from:

```sh
cd examples/rust/context
cargo run --bin context-injection
```

### `rust/custom-hitl/`

Examples with custom approval handlers and custom tools.

| Binary | Description |
|--------|-------------|
| `hitl-agent` | CLI-based approval handler for payments and record deletion, with multi-language messages |

Run from:

```sh
cd examples/rust/custom-hitl
cargo run --bin hitl-agent
```

### `rust/reasoning/`

Advanced reasoning demo with Rust-side interactive mode selection and metadata display.

| Binary | Description |
|--------|-------------|
| `reasoning-agent` | Interactive mode selection (CoT, ReAct, Plan-and-Execute, Auto) with reflection and reasoning metadata display |

Run from:

```sh
cd examples/rust/reasoning
cargo run --bin reasoning-agent
```

### `rust/custom-llm/`

Custom LLM provider examples - from implementing `LLMProvider` from scratch to multi-provider routing.


| Binary | Description |
|--------|-------------|
| `custom-provider` | Implement `LLMProvider` from scratch with an offline echo/rule-based provider - no API key needed |
| `openai-compatible` | HTTP adapter for any OpenAI-compatible server (LM Studio, Ollama, vLLM, LocalAI, TGI) |
| `multi-provider` | Multi-provider routing with `MultiLLMRouter` - expensive model for users, cheap model for internal tasks |


Note: For a zero-code YAML alternative, see `yaml/basic/openai_compatible.yaml` which uses the built-in `provider: openai-compatible` with `base_url` — no custom Rust code needed.

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

custom tool examples - from a minimal `Tool` trait implementation to a full `ToolProvider` with dynamic discovery.

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

| Binary | Description |
|--------|-------------|
| `disambiguation-agent` | Ambiguity detection and clarification with additional runtime metadata display |

Run from:

```sh
cd examples/rust/disambiguation
cargo run --bin disambiguation-agent
```

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

## Recommended Learning Path

Start with these YAML examples:

- `examples/yaml/basic/simple_chat.yaml`
- `examples/yaml/basic/simple_chat_stream.yaml`
- `examples/yaml/basic/simple_tools.yaml`
- `examples/yaml/skills/skill_agent.yaml`
- `examples/yaml/state-machine/support_state_machine.yaml`

Then move to Rust examples when you want embedding or customization.

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

Skill system examples.

| File | Description |
|------|-------------|
| `skill_agent.yaml` | YAML skill agent with inline and external skills |
| `skills/math_helper.skill.yaml` | External math skill |
| `skills/weather_clothes.skill.yaml` | External weather/clothing skill |

Example:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/skills/skill_agent.yaml
```

### `yaml/state-machine/`

Hierarchical state machine examples.

| File | Description |
|------|-------------|
| `support_state_machine.yaml` | Customer support flow with greeting, technical, order, product, escalation, and fallback states |

Example:

```sh
cargo run -p ai-agents-cli -- run examples/yaml/state-machine/support_state_machine.yaml
```

### `yaml/memory/`

Progressive memory examples — from simplest to production-grade.

| File | Description |
|------|-------------|
| `memory_basic.yaml` | Simplest memory — in-memory storage with a message limit |
| `memory_compacting.yaml` | Compacting memory with automatic LLM-based summarization |
| `memory_budget.yaml` | Token budgeting — per-component allocation controlling prompt size |
| `memory_agent.yaml` | Full production config combining compacting, budgeting, and hooks |

Learning path:

1. **`memory_basic.yaml`** — Memory exists, keeps messages, has a limit
2. **`memory_compacting.yaml`** — Old messages get summarized automatically
3. **`memory_budget.yaml`** — Token budget controls how much memory enters the prompt
4. **`memory_agent.yaml`** — Full production configuration

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

Learning path:

1. **`save-restore-session`** — Simplest persistence (save a session, quit, restart, load it back)
2. **`memory-agent`** — Monitor memory events with hooks (compression, eviction, budget warnings)
3. **`sqlite-persistence`** — Full session management with SQLite

Run from:

```sh
cd examples/rust/storage
cargo run --bin save-restore-session
cargo run --bin memory-agent
cargo run --bin sqlite-persistence
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

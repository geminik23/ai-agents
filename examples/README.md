# Examples

example for the AI Agents framework. 

```sh
# Go to the specific example directory you want to run, then execute the desired binary
cd examples/basic

cargo run --bin simple-chat
```

## Projects

### `basic/`

Getting started — no YAML required.

| Binary | Description |
|--------|-------------|
| `simple-chat` | Minimal agent built entirely from the Rust API with a single LLM |
| `tool-agent` | Agent with built-in tools (calculator, datetime, echo) via `AgentSpec` |

### `skills/`

Skill system — reusable "tool + prompt" workflows with LLM-based intent routing.

| Binary | Description |
|--------|-------------|
| `skill-agent` | Inline skills, external skills by file path, and LLM-based skill matching |

### `state-machine/`

Hierarchical state machine with LLM-evaluated transitions.

| Binary | Description |
|--------|-------------|
| `state-machine-agent` | Multi-branch customer support agent with greeting, technical, order, product, escalation, and fallback states |

### `memory/`

CompactingMemory, token budgeting, and session persistence.

| Binary | Description |
|--------|-------------|
| `memory-agent` | CompactingMemory with auto-summarization, token budget monitoring via hooks |
| `sqlite-persistence` | Save/load/search agent sessions with SQLite storage |

### `hitl/`

Human-in-the-loop — approval flows for sensitive operations.

| Binary | Description |
|--------|-------------|
| `hitl-agent` | CLI-based approval handler for payments and record deletion, with multi-language messages |

### `reasoning/`

Reasoning modes and self-evaluation.

| Binary | Description |
|--------|-------------|
| `reasoning-agent` | Interactive mode selection (CoT, ReAct, Plan-and-Execute, Auto) with reflection and reasoning metadata display |

### `disambiguation/`

LLM-based intent disambiguation and clarification.

| Binary | Description |
|--------|-------------|
| `disambiguation-agent` | Ambiguity detection, clarification questions, multi-language support |

## Structure

Each project contains:

- `Cargo.toml` - depends on `ai-agents` and `example-common`
- `src/` - source files
- `agents/` - YAML agent definitions

The `common/` crate provides a shared interactive REPL and tracing setup.

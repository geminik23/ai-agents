+++
title = "CLI Guide"
weight = 3
template = "docs.html"
description = "Command-line interface reference for ai-agents-cli."
+++

The `ai-agents-cli` crate gives you a ready-made command-line tool for running any YAML-defined agent. Install it, point it at a file, and you're in a live REPL session.

---

## Installation

### From crates.io (recommended)

```sh
cargo install ai-agents-cli --version 1.0.0-rc.10
```

### From source

```sh
git clone https://github.com/geminik23/ai-agents.git
cd ai-agents
cargo build --release
# binary: target/release/ai-agents-cli
```

### Without installing (development)

```sh
cargo run -p ai-agents-cli -- run agent.yaml
```

---

## Commands

The CLI has two subcommands:

| Command    | Description                                      |
| ---------- | ------------------------------------------------ |
| `run`      | Load an agent YAML and start an interactive REPL |
| `validate` | Check a YAML file for errors without starting    |

### `run`

```sh
ai-agents-cli run <agent.yaml> [OPTIONS]
```

### `validate`

```sh
ai-agents-cli validate <agent.yaml>
```

Parses and validates the YAML spec. Returns exit code `0` on success, `1` on failure. Useful in CI pipelines.

---

## Run Options

| Flag                      | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| `--stream`                | Stream tokens to the terminal as they arrive                 |
| `--show-tools`            | Print tool calls and their results                           |
| `--show-state`            | Display the current state in the prompt and show transitions |
| `--show-timing`           | Show elapsed time for each LLM response                     |
| `--no-builtins`           | Disable built-in REPL slash commands                         |
| `--welcome <msg>`         | Override the YAML metadata welcome message                   |
| `--hint <text>`           | Add a startup hint (can be repeated)                         |
| `--context <KEY=VALUE>`   | Inject a runtime context value (repeatable, supports dotted paths) |
| `--context-file <PATH>`   | Load runtime context from a JSON file                        |
| `--actor <ID>`            | Set actor ID at startup for cross-session memory             |
| `--plain`                 | Force plain line REPL even on interactive TTY                |
| `--theme <name>`          | Color theme (one-dark, catppuccin-mocha, dracula, tokyo-night, vscode-dark, nord, gruvbox-dark, one-half-light, github-light) |

### Examples

Basic interactive session:

```sh
ai-agents-cli run agent.yaml
```

Stream tokens with tool visibility:

```sh
ai-agents-cli run agent.yaml --stream --show-tools
```

Full debug mode — see everything:

```sh
ai-agents-cli run agent.yaml --stream --show-tools --show-state --show-timing
```

Override the welcome banner and add hints:

```sh
ai-agents-cli run agent.yaml --welcome "Welcome to the demo!" --hint "Try: hello" --hint "Try: /state"
```

---

## Context Injection

Agents that declare `context.user.type: runtime` need context values at startup.
The CLI provides two ways to inject them.

### Via command-line flags

```sh
# String values
ai-agents-cli run agent.yaml --context user.id=customer_123 --context user.name=Jane

# Numeric and boolean values (auto-detected)
ai-agents-cli run agent.yaml --context user.tier=3 --context user.verified=true

# JSON object value
ai-agents-cli run agent.yaml --context 'player={"id":"p1","name":"Hero"}'
```

### Via JSON file

```sh
ai-agents-cli run agent.yaml --context-file ./test_context.json
```

The JSON file must contain a top-level object:

```json
{
  "user": {
    "id": "customer_123",
    "name": "Jane",
    "language": "ko"
  }
}
```

Both flags can be combined. `--context` values override `--context-file` values when keys conflict.

---

## REPL Commands

Once inside the REPL, these slash commands are available (unless `--no-builtins` is set):

| Command              | Description                                                          |
| -------------------- | -------------------------------------------------------------------- |
| `/help`, `?`         | Show available commands                                              |
| `/reset`             | Clear memory and reset state                                         |
| `/state`             | Show current state machine state                                     |
| `/history`           | Show state transition history                                        |
| `/info`              | Show agent name, version, skills, spawned agents                     |
| `/memory`, `/mem`    | Show memory status and token budget                                  |
| `/context`           | Show all current context values                                      |
| `/context set <key> <value>` | Set a runtime context value                                  |
| `/context unset <key>` | Remove a context value                                             |
| `/save [name]`       | Save session (parent + all spawned agents). Default name: `default`  |
| `/save self [name]`  | Save parent session only                                             |
| `/save agent <id>`   | Save one spawned agent's session                                     |
| `/load [name]`       | Load session (parent + restore spawned agents)                       |
| `/load self [name]`  | Load parent session only                                             |
| `/load agent <id>`   | Load one spawned agent's session                                     |
| `/sessions`          | List saved sessions                                                  |
| `/sessions --actor <id>` | List sessions for a specific actor (requires session metadata)   |
| `/sessions --tag <tag>` | List sessions matching a tag (requires session metadata)          |
| `/cleanup`           | Delete sessions whose TTL has expired                                |
| `/delete <name>`     | Delete a saved session                                               |
| `/actor`             | Show current actor ID                                                |
| `/actor set <id>`    | Set actor ID for cross-session memory                                |
| `/actor facts [cat]` | Show facts for the current actor, optionally filtered by category    |
| `/actor delete`      | Delete all data for the current actor (GDPR)                         |
| `/facts`             | Show all facts for the current actor                                 |
| `/facts extract [n]` | Manually extract facts from the last N messages (default: 10)        |
| `/quit`, `/exit`     | Exit the REPL                                                        |

`/quit` and `/exit` always work, even when builtins are disabled.

---

## TUI Mode

When running on an interactive terminal, the CLI launches a ratatui-based TUI with an alternate-screen interface.
Use `--plain` to skip the TUI and use the traditional line REPL instead.

### Layout

The TUI has four zones:

- **Status bar** (top) - agent name, version, current state, current actor ID, token budget percentage, thinking spinner
- **Chat area** (center) - scrollable message history with role-colored output, hint markers, and log cards
- **Input area** (bottom) - multi-line text input
- **Hint bar** (bottom line) - key bindings and contextual hints

### Side Panels

Toggle side panels with function keys:

| Key  | Panel   | Content |
| ---- | ------- | ------- |
| `F1` | Help    | Key bindings and command reference |
| `F2` | States  | State machine visualization |
| `F3` | Memory  | Token budget breakdown and compression stats |
| `F4` | Context | Current context values |
| `F5` | Tools   | Available tools and last call result |
| `F6` | Persona | Agent persona identity, traits, goals |
| `F7` | Facts   | Live actor facts when `memory.facts` is enabled and an actor is set |
| `F8` | Agents  | Spawned agents and orchestration status |

Press the same key again to close a panel. Press `Esc` to close all panels.

### Key Bindings

| Key             | Action |
| --------------- | ------ |
| `Enter`         | Send message |
| `Ctrl+C`        | Quit |
| `Ctrl+L`        | Clear chat display |
| `Ctrl+S`        | Quick save session |
| `PageUp/Down`   | Scroll chat history |
| `Esc`           | Cancel streaming or close panels |
| `F1` - `F8`     | Toggle side panels |
| `Ctrl+T`        | Cycle color theme |
| `/`             | Start a slash command (opens completion popup) |

### Chat Display

The chat area renders messages with role-specific styling:

- **You:** - cyan, bold prefix
- **Agent:** - white, bold prefix, continuation lines indented
- System messages - yellow, no prefix (welcome banner, state transitions, errors)
- Hints - italic with `>` prefix, visually distinct from system messages
- Log cards - dim gray, shown when tracing events are captured (see below)

Startup hints defined in `metadata.cli.hints` or via `--hint` are grouped into a single block with `>` markers so they stand out from the conversation.

Agent responses with multiple paragraph breaks are normalized - consecutive blank lines are collapsed to at most one, and trailing whitespace is trimmed.

### Log Rendering

In TUI mode, tracing output is captured and rendered as dim log cards in the chat timeline instead of writing raw text to the terminal.
The default level is WARN. Set `RUST_LOG=info` or `RUST_LOG=debug` for more detail.

### Themes

The TUI ships with 11 color themes. Use `--theme` to select one:

```sh
ai-agents-cli run agent.yaml --theme one-dark
ai-agents-cli run agent.yaml --theme catppuccin-mocha
```

Available themes: `dark` (default), `one-dark`, `catppuccin-mocha`, `dracula`, `tokyo-night`, `vscode-dark`, `nord`, `gruvbox-dark`, `light`, `one-half-light`, `github-light`.

Set a default theme in YAML:

```yaml
metadata:
  cli:
    theme: one-dark
```

Press `Ctrl+T` in the TUI to cycle through themes at runtime.
The `dark` and `light` themes use standard ANSI colors and work on all terminals.
All other themes use exact RGB colors for consistent appearance across modern terminals.

### Command Completion

When you type `/` in the input area, a floating completion popup appears above the input showing all available slash commands with descriptions. Type more characters to filter the list. Use `Up`/`Down` to navigate, `Tab` to fill the selected command, `Enter` to fill and execute, or `Esc` to dismiss.

### Streaming

When `--stream` is enabled, tokens appear in real time in the chat area.
Tool calls and state transitions are displayed inline as they happen.

### Plain Mode Fallback

The plain REPL is used automatically when stdout is not a terminal (piped input, CI).
Force it with `--plain`:

```sh
ai-agents-cli run agent.yaml --plain
echo "hello" | ai-agents-cli run agent.yaml --plain
```

---

## CLI Metadata in YAML

You can bake default CLI settings into the agent YAML itself using the `metadata.cli` block. Command-line flags override these values.

```yaml
name: DemoAgent
system_prompt: "You are a helpful assistant."
llm:
  provider: openai
  model: gpt-4.1-nano

metadata:
  cli:
    welcome: "=== Welcome to DemoAgent ==="
    hints:
      - "Try asking about the weather"
      - "Type /help for commands"
    streaming: true
    show_tools: true
    show_state: false
    show_timing: true
    prompt_style: with_state          # "simple" or "with_state"
    disable_builtin_commands: false
```

### Metadata fields

| Field                      | Type       | Description                              |
| -------------------------- | ---------- | ---------------------------------------- |
| `welcome`                  | string     | Banner printed when the REPL starts      |
| `hints`                    | list       | Helpful tips shown after the banner      |
| `streaming`                | bool       | Enable streaming by default              |
| `show_tools`               | bool       | Show tool calls by default               |
| `show_state`               | bool       | Show state transitions by default        |
| `show_timing`              | bool       | Show response times by default           |
| `prompt_style`             | string     | `"simple"` or `"with_state"`             |
| `disable_builtin_commands` | bool       | Disable slash commands except `/quit`    |
| `theme`                    | string     | Color theme name (see Themes section)    |

---

## Example Session

Here's what a typical REPL session looks like with `--stream --show-tools --show-timing`:

```text
=== Welcome to DemoAgent ===
Agent: DemoAgent v1.0.0

  Try asking about the weather
  Type /help for commands

Type '/help' for commands, '/quit' to exit.

You > What's 2 + 2?

Agent: Let me calculate that for you.
  [Tool: calculator... ✓]

The answer is 4.

  Tools used: calculator
  (0.8s)

You > /state
Current state: idle

You > /info
Agent: DemoAgent v1.0.0
Description: A helpful demo assistant
Skills: 2
State: idle

You > /quit
Goodbye!
```

When `--show-state` is active, the prompt changes to show the current state:

```text
[idle] You > start a new task
[working] You > done
[idle] You >
```

> **Tip:** In TUI mode, the same commands work as slash commands in the input area.
> Press `F1` for the help panel with all key bindings.

---

## Environment Variables

Every LLM provider reads its API key from an environment variable. Set the one that matches your agent's `llm.provider`:

| Provider          | Environment Variable     |
| ----------------- | ------------------------ |
| OpenAI            | `OPENAI_API_KEY`         |
| Anthropic         | `ANTHROPIC_API_KEY`      |
| Google (Gemini)   | `GOOGLE_API_KEY`         |
| DeepSeek          | `DEEPSEEK_API_KEY`       |
| Groq              | `GROQ_API_KEY`           |
| xAI (Grok)        | `XAI_API_KEY`            |
| Mistral           | `MISTRAL_API_KEY`        |
| Cohere            | `COHERE_API_KEY`         |
| Phind             | `PHIND_API_KEY`          |
| OpenRouter        | `OPENROUTER_API_KEY`     |
| Ollama            | *(none — runs locally)*  |
| openai-compatible | *(set via `api_key_env` in YAML)* |

Example with OpenAI:

```sh
export OPENAI_API_KEY=sk-...
ai-agents-cli run agent.yaml --stream
```

You can also use a `.env` file or your shell's secret manager — the CLI reads standard environment variables through `std::env`.

### Tracing / Logging

The CLI initializes `tracing-subscriber` with `RUST_LOG` support.
In plain mode, logs go to stdout as usual.
In TUI mode, logs are captured via a channel-based layer and rendered as dim cards in the chat area (default level: WARN).

```sh
# Plain mode
RUST_LOG=info ai-agents-cli run agent.yaml --plain

# TUI mode (default WARN, override with RUST_LOG)
RUST_LOG=info ai-agents-cli run agent.yaml
RUST_LOG=ai_agents=debug ai-agents-cli run agent.yaml
```

---

## Next Steps

- **[YAML Reference](@/docs/yaml-reference.md)** — full spec for agent definition files
- **[Rust API](@/docs/rust-api.md)** — embed agents in your own Rust application
- **[LLM Providers](@/docs/providers.md)** — detailed setup for all 12 providers
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
cargo install ai-agents-cli --version 1.0.0-rc.8
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

| Flag              | Description                                          |
| ----------------- | ---------------------------------------------------- |
| `--stream`        | Stream tokens to the terminal as they arrive         |
| `--show-tools`    | Print tool calls and their results                   |
| `--show-state`    | Display the current state in the prompt and show transitions |
| `--show-timing`   | Show elapsed time for each LLM response              |
| `--no-builtins`   | Disable built-in REPL slash commands                 |
| `--welcome <msg>` | Override the YAML metadata welcome message           |
| `--hint <text>`   | Add a startup hint (can be repeated)                 |

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
| `/save [name]`       | Save session (parent + all spawned agents). Default name: `default`  |
| `/save self [name]`  | Save parent session only                                             |
| `/save agent <id>`   | Save one spawned agent's session                                     |
| `/load [name]`       | Load session (parent + restore spawned agents)                       |
| `/load self [name]`  | Load parent session only                                             |
| `/load agent <id>`   | Load one spawned agent's session                                     |
| `/sessions`          | List saved sessions                                                  |
| `/delete <name>`     | Delete a saved session                                               |
| `/quit`, `/exit`     | Exit the REPL                                                        |

`/quit` and `/exit` always work, even when builtins are disabled.

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

The CLI initializes `tracing-subscriber` with `RUST_LOG` support:

```sh
RUST_LOG=info ai-agents-cli run agent.yaml
RUST_LOG=ai_agents=debug ai-agents-cli run agent.yaml
```

---

## Next Steps

- **[YAML Reference](@/docs/yaml-reference.md)** — full spec for agent definition files
- **[Rust API](@/docs/rust-api.md)** — embed agents in your own Rust application
- **[LLM Providers](@/docs/providers.md)** — detailed setup for all 12 providers
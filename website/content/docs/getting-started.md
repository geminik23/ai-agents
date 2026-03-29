+++
title = "Getting Started"
weight = 1
template = "docs.html"
description = "Install and run your first AI agent in under a minute."
+++

## Prerequisites

- **Rust toolchain** - install from [rustup.rs](https://rustup.rs) if you don't have it
- **An LLM API key** - OpenAI is recommended for the quickstart, but any of the 12 supported providers works

## Installation

Pick whichever option fits your workflow.

### Option 1: CLI only (fastest)

```sh
cargo install ai-agents-cli --version 1.0.0-rc.7
```

### Option 2: As a library

Add this to your `Cargo.toml`:

```toml
[dependencies]
ai-agents = "1.0.0-rc.7"
```

### Option 3: From source

```sh
git clone https://github.com/geminik23/ai-agents.git
cd ai-agents
cargo build --release
```

The binary lands in `target/release/ai-agents-cli`.

---

## Your First Agent (CLI)

### 1. Create `agent.yaml`

```yaml
name: MyAgent
system_prompt: "You are a helpful assistant."
llm:
  provider: openai
  model: gpt-4.1-nano
```

### 2. Set your API key

```sh
export OPENAI_API_KEY=sk-...
```

### 3. Run it

```sh
ai-agents-cli run agent.yaml
```

You're now in a REPL session. Type a message and press Enter. The agent responds using the model you configured. Type `/quit` to exit.

---

## Your First Agent (Rust)

Create a `main.rs` that loads the same YAML file programmatically:

```rust
use ai_agents::{Agent, AgentBuilder};

#[tokio::main]
async fn main() -> ai_agents::Result<()> {
    let agent = AgentBuilder::from_yaml_file("agent.yaml")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .build()?;

    let response = agent.chat("Hello!").await?;
    println!("{}", response.content);
    Ok(())
}
```

Or build an agent entirely in code without YAML:

```rust
use ai_agents::{AgentBuilder, UnifiedLLMProvider, ProviderType};
use std::sync::Arc;

#[tokio::main]
async fn main() -> ai_agents::Result<()> {
    let llm = UnifiedLLMProvider::from_env(ProviderType::OpenAI, "gpt-4.1-nano")?;

    let agent = AgentBuilder::new()
        .system_prompt("You are a helpful assistant.")
        .llm(Arc::new(llm))
        .build()?;

    let response = agent.chat("Hello!").await?;
    println!("{}", response.content);
    Ok(())
}
```

Run it with `cargo run`. That's all you need - one YAML, a few lines of Rust.

---

## CLI Options

These flags work with `ai-agents-cli run`:

| Flag             | Description                                      |
| ---------------- | ------------------------------------------------ |
| `--stream`       | Stream tokens to the terminal as they arrive     |
| `--show-tools`   | Print tool calls and their results               |
| `--show-state`   | Display agent state after each turn               |
| `--show-timing`  | Show how long each LLM call takes                |

Example with all flags:

```sh
ai-agents-cli run agent.yaml --stream --show-tools --show-state --show-timing
```

---

## REPL Commands

Once inside the REPL session, these slash commands are available:

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

---

## Using Other Providers

Swap the `llm` block in your YAML to switch providers. Everything else stays the same.

### Anthropic

```yaml
llm:
  provider: anthropic
  model: claude-haiku-4-5-20251001
```

```sh
export ANTHROPIC_API_KEY=sk-ant-...
```

### Google

```yaml
llm:
  provider: google
  model: gemini-3-flash
```

```sh
export GOOGLE_API_KEY=AI...
```

### Ollama (local, no API key needed)

```yaml
llm:
  provider: ollama
  model: llama3
```

No environment variable required - just make sure [Ollama](https://ollama.com) is running locally on the default port.

---

## YAML CLI Metadata

YAML files can include optional `metadata.cli` for a better interactive experience:

```yaml
metadata:
  cli:
    welcome: "=== My Agent ==="
    hints:
      - "Try asking about the weather"
      - "Type 'help' for commands"
```

---

## Next Steps

- **[YAML Reference](@/docs/yaml-reference.md)** - the complete spec for agent definition files
- **[Examples](@/examples/_index.md)** - more patterns: tool use, multi-agent, stateful workflows
- **[CLI Guide](@/docs/cli.md)** - every command and flag explained

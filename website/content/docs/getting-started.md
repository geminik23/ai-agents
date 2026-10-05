+++
title = "Getting Started"
weight = 1
template = "docs.html"
description = "Start with the CLI, embed in Rust, use a local model, or run a no-key evaluation."
+++

Choose a path: [CLI conversation](#your-first-agent-cli), [Rust application](#your-first-agent-rust), [local Ollama](#start-with-a-local-model), or [mocked evaluation](#start-with-a-mocked-evaluation).

## Prerequisites

- **Rust 1.88 or newer** - install from [rust-lang.org](https://rust-lang.org/tools/install) if you don't have it
- **A model for live conversations** - the hosted quickstart uses an OpenAI API key; [local Ollama](#start-with-a-local-model) needs no hosted provider key.
- **No model or API key for mocked evaluation** - the [evaluation path](#start-with-a-mocked-evaluation) supplies fixed model responses.

## Installation

Pick whichever option fits your workflow.

### Option 1: CLI only (fastest)

```sh
cargo install ai-agents-cli --version 1.1.1
```

### Option 2: As a library

Add this to your `Cargo.toml`:

```toml
[dependencies]
ai-agents = "1.1.1"
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
  model: gpt-5.4-nano
```

This agent omits top-level `tools:`, so it has no tool access. Add top-level `tools:` explicitly when you want to grant tools.

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

Add Tokio to the library dependencies so the asynchronous entry point can run:

```toml
tokio = { version = "1", features = ["full"] }
```

Create a `main.rs` that loads the same YAML file programmatically:

```rust
use ai_agents::{Agent, AgentBuilder};

#[tokio::main]
async fn main() -> ai_agents::Result<()> {
    let agent = AgentBuilder::from_yaml_file("agent.yaml")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .auto_configure_mcp().await?
        .auto_configure_spawner().await?
        .build()?;

    let response = agent.chat("Hello!").await?;
    println!("{}", response.content);
    Ok(())
}
```

This is the same builder chain used by the CLI. `auto_configure_mcp()` and `auto_configure_spawner()` are safe to keep in the chain even when the YAML does not use MCP tools or a `spawner:` section.

Or build an agent entirely in code without YAML:

```rust
use ai_agents::{Agent, AgentBuilder, UnifiedLLMProvider, ProviderType};
use std::sync::Arc;

#[tokio::main]
async fn main() -> ai_agents::Result<()> {
    let llm = UnifiedLLMProvider::from_env(ProviderType::OpenAI, "gpt-5.4-nano")?;

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

## Start With a Local Model

Install [Ollama](https://ollama.com), start its local server, and download the model you want to use. If the server is already running, only the pull command is needed. Run the pull command in a second terminal while `serve` is running.

```sh
ollama serve
```

```sh
ollama pull llama3.1
```

Save this complete agent as `local-agent.yaml`:

```yaml
name: LocalAgent
system_prompt: "You are a helpful assistant."
llm:
  provider: ollama
  model: llama3.1
```

Then run:

```sh
ai-agents-cli run local-agent.yaml
```

The model runs through your local Ollama server, with no hosted provider API key. The initial model download requires network access and enough local disk space; model performance depends on your hardware. See [Ollama configuration](@/docs/providers.md#ollama) to change the server address or context window.

## Start With a Mocked Evaluation

Use this path to try agent construction, runtime execution, assertions, and reports without installing a model or setting an API key. It uses the repository's agent and suite files, so first follow the source checkout instructions under [Installation](#installation).

From the repository root:

```sh
cargo run -p ai-agents-cli -- eval \
  --agent examples/yaml/basic/simple_chat.yaml \
  --scenarios examples/eval/mocked/basic/simple_chat_mocked.yaml \
  --output target/eval/mocked/basic/simple_chat_mocked
```

This suite supplies a fixed greeting and checks the response. Open `target/eval/mocked/basic/simple_chat_mocked/summary.md` for the result. Compilation may download dependencies; the scenario itself does not call a model provider.

Continue with [Test and Observe](@/docs/test-and-observe.md) to inspect evidence, add assertions, and distinguish mocked regression tests from live model checks.

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
  model: gemini-2.5-flash
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
      - "Type '/help' for commands"
```

---

## Next Steps

- **[Build Agent Behavior](@/docs/build-behavior.md)** - choose states, skills, context, and processing steps
- **[Control Execution](@/docs/control-execution.md)** - grant tools explicitly and add approval
- **[Test and Observe](@/docs/test-and-observe.md)** - verify the behavior you build

- **[YAML Reference](@/docs/yaml-reference.md)** - the complete spec for agent definition files

- **[Examples](@/examples/_index.md)** - more patterns: tool use, multi-agent, stateful workflows
- **[CLI Guide](@/docs/cli.md)** - every command and flag explained

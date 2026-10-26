+++
title = "CLI Guide"
weight = 3
template = "docs.html"
description = "Install ai-agents-cli to run YAML agents, validate configuration, and execute evaluation suites from the command line."

[extra]
seo_title = "AI Agents CLI: Run, Validate, and Evaluate YAML Agents"
+++

The `ai-agents-cli` crate runs YAML-defined agents on the ai-agents Rust runtime. Use it to start a conversation, validate an agent configuration, or run evaluation suites from the terminal. For provider setup and your first conversation, follow [Getting Started](@/docs/getting-started.md).

---

## Installation

### From crates.io (recommended)

```sh
cargo install ai-agents-cli --version 1.1.2
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

The published CLI provides `run`, `eval` and `validate`. The development checkout additionally provides an unreleased `task` command family:

| Command    | Description                                      |
| ---------- | ------------------------------------------------ |
| `run`      | Load an agent YAML and start an interactive REPL |
| `eval`     | Run declarative scenario suites against an agent |
| `validate` | Check a YAML file for errors without starting    |
| `task` | Development-only task execution and scoped journal management |

### `run`

```sh
ai-agents-cli run <agent.yaml> [OPTIONS]
```

### `eval`

```sh
ai-agents-cli eval [--agent <agent.yaml>] --scenarios <suite.yaml> [OPTIONS]
```

Runs YAML or JSONL scenario suites, evaluates assertions from structured turn evidence, and writes `summary.md`, schema-versioned `summary.json`, `per_scenario.jsonl`, `failures.md`, and optionally `junit.xml`. `--agent` is required only when the suite does not declare `agent:`; when supplied, it overrides the suite path.

### `validate`

```sh
ai-agents-cli validate <agent.yaml>
```

Parses and validates the YAML spec. Fixed framework-owned configuration objects reject unknown fields, so typos such as `max_iteratons` fail validation; documented provider-specific keys under `llm` remain available through the provider extension map. Returns exit code `0` on success and `1` on failure.

---

## Development task CLI

These commands describe the development tree, not the published 1.1.1 CLI. They run the existing general-purpose task controller; they do not install a coding-specific loop, scheduler, daemon or production restart host. A selected profile must be enabled, and ordinary chat remains unchanged when autonomy is omitted or disabled.

```sh
cargo run -p ai-agents-cli -- task run agent.yaml "Research the requested topic"
cargo run -p ai-agents-cli -- task run agent.yaml "Research the requested topic" --profile research
cargo run -p ai-agents-cli -- task run agent.yaml "Produce a report" --non-interactive --json
cargo run -p ai-agents-cli -- task list agent.yaml --json
cargo run -p ai-agents-cli -- task show agent.yaml RUN_ID --json
cargo run -p ai-agents-cli -- task cancel agent.yaml RUN_ID --revision REVISION --json
cargo run -p ai-agents-cli -- task resume agent.yaml RUN_ID
```

`task run` uses the same YAML/base-directory, providers, tools, MCP, spawner, context and actor preparation as ordinary execution. Add `--context KEY=VALUE` repeatedly, `--context-file PATH`, or `--actor ID` as needed. `--profile ID` selects an existing profile rather than granting capabilities or enabling a disabled one. `--non-interactive` exposes no stdin response endpoint; it does not reject tool-free or no-storage tasks that can complete normally.

The agent must declare meaningful task limits and completion gates. For a minimal protocol demonstration, append the following to an otherwise valid provider-configured agent:

```yaml
autonomy:
  enabled: true
  mode: until_complete
  max_turns: 4
  max_llm_calls: 8
  completion: {response_not_empty: true}
```

This deliberately weak demonstration gate proves only that a response exists, not research quality, ticket resolution or report correctness. Real profiles should use the appropriate tool/state/context, artifact/validation or judge evidence. Todo, filesystem and command tools are not universal prerequisites. The `--profile research` example additionally requires that named profile in the agent's `autonomy.profiles` map.

### Interactive decisions and plain REPL

An interactive foreground process retains its original runtime, runner and store while servicing approvals, questions, explicit Continue, capacity, stage/plan review and supported external/observation responses. It displays the exact run/revision/request to which a response applies. Free-text questions accept text; choice questions accept declared labels or one-based option numbers, with comma-separated numbers for multiple selection. Enter uses a configured question default only when one exists. EOF is never a default answer, approval or Continue. Approvals accept explicit `yes`/`no` or reviewed argument changes as `{"changes": {...}}`; modified arguments still pass current policy and final admission. Bare `stop` and `cancel` are host controls, not model instructions.

Existing `run` also consumes automatic state/selected-skill task entry in the **plain REPL**:

```sh
cargo run -p ai-agents-cli -- run agent.yaml --plain
cargo run -p ai-agents-cli -- run agent.yaml --plain --stream
```

The development TUI also services these workflows through the same foreground driver; use `--plain` for line-oriented interaction. See [development task TUI](@/docs/cli.md#development-task-tui) for the terminal controls. Task lifecycle streaming is distinct from provider token streaming, and a dispatch `Final` with a paused task is not task completion.

| REPL command | Meaning |
| --- | --- |
| `/task run OBJECTIVE` | Explicit task using the same configured host |
| `/task run --profile ID OBJECTIVE` | Explicit selection of an enabled profile |
| `/task list` | Current scoped task metadata; during work, only actual foreground participants |
| `/task show RUN` | Bounded default-redacted snapshot |
| `/task resume RUN REV REQUEST ANSWER` | Explicit answer to the exact advertised interaction |
| `/task cancel RUN REV` | Exact-revision cancellation request or retained-pause cleanup |
| `/task stop RUN REV [REASON]` | Stop a safe pause without executing pending work |
| `/task objective RUN REV NEW_OBJECTIVE` | Revise a safe objective boundary, preserving usage and requiring explicit continuation |
| `/task limits RUN REV JSON` | Revise numeric limits within host ceilings; never refill consumed counters |

Controls apply while the foreground session retains the run. A plain reply uses the identity of the displayed prompt; explicit `/task resume` never refreshes a stale revision or substitutes a different request. Syntax errors are rejected without invoking a task control. Ordinary callback approval/question input and task typed responses share one stdin reader but not authority. Task control remains available while work is retained even when ordinary convenience commands are disabled.

Ordinary composing parents can wait for independently hosted child tasks. Their questions are serviced before the parent returns; independent child decisions do not overwrite one another. Children already inside a task instead retain the controller's coordinating request, sibling state and single completion owner. Answering a child does not restart the parent or replay completed siblings.

### Output, cancellation and exit

`--json` emits bounded default-redacted task events followed, on normal delivery, by a safe result record containing run ID, revision, status and counters. Exact local questions and necessary review fields go to stderr, not JSON exports; they may contain sensitive data and should not be captured as diagnostics. Raw checkpoints, native history and owner tokens are not exported. Plain output includes the final response and actual task status. Missing measurements remain unavailable rather than zero or Pass.

| Exit code for `task run` / `task resume` | Meaning |
| --- | --- |
| `0` | Completed |
| `1` | Failed, Cancelled, Incomplete, LimitReached or RecoveryRequired |
| `2` | Configuration, validation, storage, usage or command I/O error |
| `3` | Reserved for an actually acknowledged prepared durable pause that survives process exit; the default CLI cannot produce this |

`list`, `show` and `cancel` report command/config/storage success separately from the task's status. A successful remote `cancel` records a request, not remote termination. Its receipt distinguishes `requested`, `acknowledged` and `already_terminal`; store-only callers cannot certify live cleanup. Obtain the exact revision from an authorized observation rather than retrying arbitrary revisions after an error.

Ctrl+C or `/quit` during work requests shutdown and keeps polling the original operation and controls until their actual result or protected failure is returned. Completion can legitimately win a cancellation race. EOF withdraws input delivery; supported stop/fail policies or typed unavailable cleanup handle unserviceable volatile pauses without executing pending work. Broken output also closes ordinary callbacks rather than leaving them waiting. A slow consumer applies bounded backpressure during normal operation; once shutdown begins, display delivery is suppressed/drained so it cannot block execution settlement. Interrupted output may therefore lack a final JSON record; the process reports its actual result through the exit code and a concise stderr diagnostic when available. A second Ctrl+C forces exit with cleanup explicitly unconfirmed. Timeout, drop and process termination never prove rollback or exactly-once external effects.

### Storage and restart limits

No-storage and retained in-memory runs can complete or pause/resume in the same process. `task list/show/cancel` instead open an **existing configured SQLite task journal** through an authorized agent/actor/configuration scope, without constructing providers, reading provider credentials, starting MCP/spawner services, adopting an owner or consuming a pending decision. File/Redis session snapshots are not task journals. A different actor/configuration is not a substitute for the original scope. Storage paths keep their existing interpretation; relative skill/child definitions use the agent YAML's base directory.

The declarative scope includes the normalized agent declaration and loaded skills. Execution verifies the definitions actually prepared by the builder rather than trusting a filename or an independent reread. This namespace is not caller authentication, opaque implementation binding or reconstruction authority. Protect the configuration and database using the host's normal access controls.

The default CLI has no production reconstruction host or portable restart-clock binding. Explicit durable requests are rejected before task execution; `task resume` in a new process returns a clear unsupported-capability error before provider construction. SQLite existence and session `/save`/`/load` do not make a task restartable. Use retained in-process controls, or the explicitly prepared [Rust restoration APIs](@/docs/rust-api.md#automatic-state-and-skill-tasks-development-only) with an actual host binding. Arbitrary in-flight stack restoration and automatic abandoned-owner takeover remain unsupported.

The development `eval` command also accepts typed `!task` steps and task assertions through the configured controller. Stock integrations use this CLI directly; domain suites with named Rust adapters use the fixed host route documented by the mocked helper. Development report schema 3, foreground participant controls, migration and timeout/cleanup behavior are described in [task evaluation](@/docs/evaluation.md#development-task-evaluation-foundation). Custom host executables can install the same domain bindings with `EvalHost`; the helper's fixed marker is not a plugin loader. See the [task walkthrough](@/docs/task-autonomy.md). These development evaluation features do not install default production task reconstruction.

## Eval Options

| Flag | Description |
| ---- | ----------- |
| `-a, --agent <YAML>` | Agent config YAML. Overrides suite `agent`. |
| `-s, --scenarios <FILE>` | Scenario suite YAML or JSONL file. |
| `-o, --output <DIR>` | Output directory. Default: `./eval_results`. |
| `-i, --id <ID>` | Run one scenario ID. Repeatable. |
| `-t, --tags <TAG>` | Run scenarios matching tags. Repeatable. |
| `--tag-mode <all|any>` | Tag match behavior. Default: `any`. |
| `--language <LANG>` | Run only scenarios for a language. Repeatable. |
| `--retries <N>` | Override retry count. |
| `--timeout <MS>` | Override per-turn timeout in milliseconds. |
| `--parallel <N>` | Run up to N scenarios concurrently. Requires `isolation: scenario` and no `env` overlays. |
| `--fail-fast` | Stop after first failed or errored scenario. Runs serially. |
| `--junit` | Write `junit.xml`. |
| `--json` | Print `summary.json` to stdout. |
| `--observability` | Attach a safe default observability overlay when the suite does not define `observability:`. |
| `--dry-config-check` | Validate the suite and referenced agent without selecting scenarios, constructing providers, or writing reports. |
| `--record <FILE>` | Explicitly authorize real LLM calls and append synchronized cassette JSONL records. The destination is opened before provider calls. |
| `--replay <FILE>` | Replay exact alias, model, and request-hash matches from a cassette. A missing match is an error. |
| `--real-llm` | Explicitly authorize real LLM calls even when the suite declares another fixture mode. |

Example:

```sh
cargo run -p ai-agents-cli -- eval \
  --agent examples/yaml/basic/simple_chat.yaml \
  --scenarios examples/eval/mocked/basic/simple_chat_mocked.yaml \
  --output target/eval/mocked/basic/simple_chat_mocked
```

Exit codes: `0` all non-skipped scenarios passed, `1` one or more scenario assertions failed or errored, `2` suite parsing, fixture setup, or runtime setup failed before a useful result was produced.

### Eval suite shape

A suite can be a YAML file with `name`, optional `agent`, `settings`, `fixtures`, and `scenarios`, or a JSONL file where each row becomes one one-turn scenario. The YAML format is the most expressive form:

```yaml
name: Basic Chat Eval
agent: ../../../yaml/basic/simple_chat.yaml
settings:
  timeout_per_turn_ms: 5000
  retries: 0
  isolation: scenario
fixtures:
  llm:
    mode: mock
    responses:
      - "Hello! I can help with that."
scenarios:
  - id: hello-smoke
    tags: [basic, smoke]
    turns:
      - input: Hello
        assert:
          response_not_empty: true
          response_contains: "Hello"
```

The runner supports mocked, replayed, recorded, and real LLM modes, mock tools, a lightweight HTTP mock server, context fixtures, blocking and streaming turns, response/state/context/metadata/tool/facts/relationship/persona/orchestration/observability assertions, composite `all`/`any`/`not` assertions, optional judge assertions, retries, filtering, fail-fast, and report output. See [Evaluation](@/docs/evaluation.md) for use-case guides such as record/replay cassettes and live LLM checks.

Implementation notes:

- Fixed suite and assertion objects reject unknown fields. Empty assertion trees, empty assertion collections, nonnumeric values used with numeric bounds, and filters that select zero scenarios are configuration or assertion failures rather than successful no-op runs.
- A suite cannot authorize its own `real` or `record` mode. Use `--real-llm` or `--record` explicitly; without those flags no real provider is constructed.
- `settings.parallel` and `--parallel` run scenario-isolated suites concurrently. `fail_fast`, `env` overlays, and unsupported isolation modes force serial execution or configuration errors. Attempts without env overrides can run together, while an env-mutating attempt has exclusive process-environment access until restoration completes.
- Suite-level `observability:` is honored. CLI `--observability` creates a safe default overlay under the eval output directory.
- `turn.stream: true` uses the complete runtime event stream and evaluates the authoritative final response content and metadata. Record mode rejects streaming before invoking the provider because atomic stream recording is not implemented.
- `fixtures.mock_server.enabled: true` starts a local HTTP server and injects `mock_server.base_url` into runtime context.
- `settings.redact_outputs: true` stores `[redacted]` for input, response, and string assertion details, and default JSON/JSONL reports omit raw `TurnEvidence` and response metadata.
- `--tag-mode` must be `any` or `all`; invalid values exit with code `2`.

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
| `/sessions --actor <id>` | List sessions for an actor (requires `SessionFiltering`; SQLite among built-ins) |
| `/sessions --tag <tag>` | List sessions matching a tag (requires `SessionFiltering`; SQLite among built-ins) |
| `/cleanup`           | Delete expired sessions (requires `ExpiryCleanup`; unwrapped SQLite among built-ins) |
| `/delete <name>`     | Delete a saved session                                               |
| `/actor`             | Show current actor ID                                                |
| `/actor set <id>`    | Set actor ID for cross-session memory                                |
| `/actor facts [cat]` | Show facts for the current actor, optionally filtered by category    |
| `/actor delete`      | Atomically delete current-actor data (requires `ActorDataDeletion`; SQLite among built-ins) |
| `/facts`             | Show all facts for the current actor                                 |
| `/facts extract [n]` | Manually extract facts from the last N messages (default: 10)        |
| `/relationship`, `/rel` | Show current actor relationship dimensions                       |
| `/relationship events` | Show notable relationship events for the current actor             |
| `/relationship actors` | Show actors with loaded relationship state                         |
| `/relationship set <dimension> <delta> [reason]` | Manually update `agent_to_actor` relationship dimension |
| `/relationship setp <perspective> <dimension> <delta> [reason]` | Manually update `agent_to_actor` or `perceived_actor_to_agent` |
| `/quit`, `/exit`     | Exit the REPL                                                        |

`/quit` and `/exit` always work, even when builtins are disabled.

---

## TUI Mode

When running on an interactive terminal, the CLI launches a ratatui-based TUI with an alternate-screen interface.
Use `--plain` to skip the TUI and use the traditional line REPL instead.

### Development task TUI

This section describes the development checkout, not the published 1.1.1 CLI. The TUI uses the same checked YAML/base-directory, actor scope, configured runtime/runner/store and controller as the plain REPL. `/task run [--profile ID] OBJECTIVE` starts explicit work; ordinary input can enter enabled state/selected-skill tasks without enabling provider token streaming. Omitted or disabled autonomy preserves ordinary chat.

`F10` toggles the task view: run/revision/status, readable objective/profile/stage, optional canonical todos, historical tri-state validation, domain progress/stagnation/replan observations, file or non-file artifact references, cumulative usage and stored active-time/original-expiry facts. `Ctrl+Up/Down` selects an observed run, and `Ctrl+PageUp/PageDown` scrolls its details. These are observations, not fresh completion proofs or UI-enforced clocks. Missing measurements remain unavailable, not zero or Pass. Authoritative final answers return to chat; hiding the task view does not cancel execution.

Pending decisions use the exact displayed task run/revision/request, or a distinct ordinary callback identity. Questions support declared choices, multiple selection and free text. Other typed decisions present their bound instructions for approval/rejection or argument changes, Continue, capacity revisions, stage/plan review, external signals, observations and installed-host restoration review. Use `PageUp/PageDown` to scroll long review text. `Esc` or the modal Cancel action hides a task decision without answering; `Ctrl+P` restores its draft. Select Use default explicitly when desired. A restoration review is separate from the original pending answer. Configured ordinary approval styles remain effective; they do not auto-approve required task reviews or override automatic policy denial.

The same `/task resume/cancel/stop/objective/limits` exact-target syntax remains available after hiding a modal. `/task list` and `/task show RUN` inspect observed session runs while idle, and actual foreground participants during work; the standalone CLI provides store-only journal management. New chat/run input does not replace retained work. Invalid or stale decisions are rejected without rebinding to a newer request, and actor/session mutations cannot replace a retained task.

The first `Ctrl+C` or `/quit` requests foreground shutdown and keeps polling the original operations. A second explicit quit forces exit with cleanup unconfirmed. Input loss closes ordinary callbacks and follows the task's existing unavailable policy. Cancellation requested, cleanup acknowledged, stored terminal metadata and protected recovery are distinct. Completion may win a cancellation race. Terminal restoration also runs on setup/draw/error paths, but never proves rollback or task cleanup.

Input and ordinary callback queues are bounded (16 and 32); display/control delivery uses bounded queues of 32. Tick/resize are coalesced by the event loop and log hints may be dropped under pressure. Chat history is display-only (up to 256 entries and 32 KiB content), previews and local task details are bounded, and truncation never changes the exact retained result or checkpoint. Local objectives, question/default text and host-selected review context are trusted UI data, not safe log/export payloads. JSON task snapshots retain default redaction. Slow or closed display delivery must not block shutdown settlement.

Session save/load does not reconstruct tasks. Default production new-process restart remains unsupported; an embedding Rust host must explicitly supply the existing prepared reconstruction contract. Low-level Rust TUI integrations now pass bounded `tokio::sync::mpsc::Sender` values to app/question/log constructors; modal constructors initialize review scrolling, while direct modal literals must include `review_scroll`.

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
| `F9` | Relationship | Current actor relationship dimensions and notable events |

Press the same key again to close a panel. Press `Esc` to close all panels.

### Key Bindings

| Key             | Action |
| --------------- | ------ |
| `Enter`         | Send message |
| `Ctrl+C`        | Request shutdown and await settlement; a second quit forces exit with cleanup unconfirmed |
| `Ctrl+L`        | Clear chat display |
| `Ctrl+S`        | Quick save session |
| `PageUp/Down`   | Scroll chat history |
| `Esc`           | Close completion/panels or hide a task decision/display without approval or cancellation |
| `F1` - `F9`     | Toggle side panels |
| `F10`           | Toggle task evidence/todo/validation/artifact view |
| `Ctrl+P`        | Reopen the exact pending decision and draft |
| `PageUp/Down` in modal | Scroll local review text |
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

When `--stream` is enabled, provisional tokens appear in real time in the chat area while tool calls and state transitions are displayed as they happen. The finalized response becomes the authoritative stored message, replacing the preview when output processing, reflection, or a committed transition changed the text. If the event stream ends without a final response or terminal error, the CLI and TUI report an incomplete-stream error instead of treating provisional content as completed; the TUI also clears its preview and active tool state. Pressing Esc while a turn is active abandons that turn's local display and clears the spinner, preview, and active tool state. It does not cancel the spawned runtime/provider request, but later events from the abandoned turn are ignored and cannot overwrite or clear a newer turn.

### `ask_user` behavior

The CLI installs a question handler for the `ask_user` tool:

- Plain REPL - prints the question and numbered options, then reads from stdin.
- TUI - shows a modal that supports single-choice options, multi-select, free-text input, and configured defaults. Cancel or Esc submits no explicit selection, so the configured default is used when present; otherwise the response is unanswered/unavailable.
- Non-interactive fallback - if stdin is not interactive, the tool uses its configured `default` when present or returns an unavailable-style result.

This is separate from HITL approval. HITL approves risky actions. `ask_user` is for normal clarification or preference questions inside the conversation flow.

### `command` behavior

The CLI installs a `ProcessCommandRunner` for the `command` tool when an agent is built through `ai-agents-cli`.

- The tool runs argv-based processes, not a shell.
- stdin is closed by default.
- output is bounded by policy and tool limits.
- the working directory must match `tool_security.tools.command.working_dirs`.
- exact argv allowlists such as `allowed_commands` are checked before the process starts.

If an agent is loaded by another host that does not install a command runner, the shared executor returns an explicit unavailable result instead of trying to spawn a process.

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
  model: gpt-5.4-nano

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
| `hitl.style`               | string     | CLI approval style: `prompt`, `auto_approve`, or `auto_reject` |
| `hitl.show_context`        | bool       | Show approval arguments in the prompt    |

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

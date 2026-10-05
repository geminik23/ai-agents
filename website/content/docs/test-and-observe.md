+++
title = "Test and Observe"
weight = 24
template = "docs.html"
description = "Turn expected behavior into scenarios and read the execution evidence."

[extra]
learning_guide = true
+++

An agent's reply is only part of its behavior. Test the state it entered, the tools it executed, the approval decision it received, and the memory it persisted. Evaluation runs the normal builder and runtime with declared inputs, fixtures, and assertions.

## Run a first test without an API key

Clone the repository using [Getting Started](@/docs/getting-started.md#installation), then run this from its root:

```sh
cargo run -p ai-agents-cli -- eval \
  --agent examples/yaml/basic/simple_chat.yaml \
  --scenarios examples/eval/mocked/basic/simple_chat_mocked.yaml \
  --output target/eval/mocked/basic/simple_chat_mocked
```

The [suite](https://github.com/geminik23/ai-agents/blob/main/examples/eval/mocked/basic/simple_chat_mocked.yaml) supplies a fixed model response and checks the greeting. It exercises loading, construction, a runtime turn, assertions, and report output without contacting a model provider. Initial compilation may still download Cargo dependencies.

```text
Agent YAML + suite → fixtures → runtime turn → evidence → assertions → reports
```

## Choose the evidence that proves the behavior

| Expected behavior | Useful evidence |
| --- | --- |
| The agent answers a fixed test | Response assertions |
| A workflow enters the right phase | State and transition assertions |
| A tool implementation runs | `tool_called` with `executed: true` |
| A request is blocked | Denial or approval evidence and `executed: false` |
| An actor preference survives | Facts after a fresh runtime with persisted storage |
| A team follows the intended pattern | Structured orchestration results |

A tool-shaped answer does not prove a tool executed. Likewise, an approval event does not prove final admission succeeded. Pair the decision with execution evidence when verifying an operation.

Mocking the LLM does not automatically mock tools. Suites that must avoid side effects should also declare the required tool or transport fixtures. The [approval recipe](@/examples/recipe-approve-tools.md) links a suite covering approval, rejection, and timeout with fixed responses and tool fixtures.

## Read the reports

The runner writes these files under the selected output directory:

| File | Start here when… |
| --- | --- |
| `summary.md` | You want a readable overview |
| `failures.md` | You need to inspect failed checks |
| `summary.json` | You need structured results and metrics |
| `per_scenario.jsonl` | You want to process scenarios one record at a time |

Add `--junit` for CI systems that consume JUnit. Use observability when you need call counts, model usage, cost accounting, or execution spans; interpret mocked usage as test data rather than a real provider benchmark.

## Separate configuration, regression, and model quality

Use `--dry-config-check` to validate the suite and referenced agent without executing scenarios. A nonexistent scenario ID is an error, not a configuration-check shortcut.

```sh
cargo run -p ai-agents-cli -- eval \
  --scenarios examples/eval/mocked/basic/simple_chat_mocked.yaml \
  --dry-config-check
```

Mocked suites verify known paths deterministically. Live suites verify actual provider behavior and require explicit execution authorization such as `--real-llm`. Semantic judge assertions answer a different question from structural checks and may themselves invoke a model. Keep their credentials, costs, and variability in the test design.

The turn timeout covers the chat or stream operation. Background task flushing and assertion or judge work have separate execution scope; do not read it as a deadline for all subsequent report work.

## Grow the test with the agent

When adding a state, write both a matching-input and an ambiguous-input scenario. When adding a tool, cover execution and denial. When adding durable memory, rebuild the runtime before testing recall. This gives each new capability an observable contract.

Continue with the full [evaluation reference](@/docs/evaluation.md), [assertion schema](@/docs/evaluation.md#assertions), and [observability configuration](@/docs/yaml-reference.md#observability-tracing).

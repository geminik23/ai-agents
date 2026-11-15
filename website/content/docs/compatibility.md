+++
title = "Support and compatibility"
weight = 15
template = "docs.html"
description = "Understand the development task support boundaries and migration from the published 1.1.2 Rust, YAML, CLI and report contracts."
+++

This page compares the development checkout with published **1.1.2**. The checkout still carries that manifest version; it does not mean the development additions were published in 1.1.2. The next release version has not been selected. The Rust minimum remains **1.88**.

## Task support boundaries

A task continues, pauses or finishes according to configured completion evidence and host decisions. YAML describes common behavior, profiles, stages and policy; Rust integrations supply application services and domain evidence. Research, support, operations, reports, code and tool-free analysis use the same controller. A particular profile does not make tools, filesystem access, todos or SQLite mandatory for every task.

| Development surface | Available behavior | Required host contract or limitation |
|---|---|---|
| Ordinary chat and streaming | Existing conversation, state and skill paths | Omitted or disabled autonomy preserves ordinary execution |
| Explicit task and skill invocation | Named profiles and per-run overrides through Rust and CLI | Required reviews and host ceilings remain effective; overrides belong to that run |
| Automatic state and selected-skill tasks | Enabled scoped input enters the same controller | Prepared input and completed skill work are retained; scope exit alone does not establish success |
| Human decisions and retained continuation | Questions, approvals, objective/limit changes and explicit resume | The live host must service the exact run/request/revision; old decisions cannot authorize new work |
| CLI, REPL, TUI and task events | Foreground execution, local review and bounded observations | Exact local prompts are sensitive; cancellation requests are distinct from acknowledged cleanup |
| Composition and managed background/speculative work | Supported child and runtime work participates in task accounting | Actual participants must preserve ownership, admission and result acknowledgement; arbitrary custom background work is not automatically covered |
| Validation and resource admission | Configured completion gates, fresh evidence and explicit cost/write capabilities | Domain implementations and actual provider/tool bindings must be installed; names and approvals alone confer no authority |
| Task storage | Conditional task operations in SQLite and process-local task storage | File/Redis snapshot persistence does not provide task transactions |
| Prepared safe-pause reconstruction | Explicit Rust hosts can restore supported prepared pauses | Matching configuration, live bindings, clock guarantees and required review are necessary; opening SQLite alone does not install reconstruction |
| Evaluation and observation | Typed task operations and shared application/eval hosts | Scenario success proves its assertions, including expected refusals; it is not unconditional business success |

The default CLI supports retained work in the original session. A new-process `task resume` requires a reconstruction host that the default CLI does not install. Automatic abandoned-owner takeover, arbitrary in-flight call-stack reconstruction, OS sandboxing and exactly-once external effects are not provided.

New task profiles have no implicit cumulative count/time budget. Explicit limits, host ceilings, per-invocation bounds, accounting, cancellation and completion checks still apply. Evaluation budgets belong to the evaluation suite, not ordinary application defaults.

See [task hosting](@/docs/task-autonomy.md), [YAML configuration](@/docs/yaml-reference.md#development-only-autonomy-configuration), [Rust hosting](@/docs/rust-api.md#automatic-state-and-skill-tasks-development-only) and [CLI behavior](@/docs/cli.md#development-task-cli) for the actual entry points.

## Rust migration from 1.1.2

The development checkout contains source-breaking changes to existing public types. Optional YAML fields and serde defaults do not preserve exhaustive Rust literals or matches. Using the crate facade also does not hide changes to the types it re-exports. These changes must be considered when choosing the next release version; the earlier routing migration's one-time exception does not apply automatically.

For struct construction, prefer existing constructors or `..Default::default()` where the type supports them. Existing exhaustive literals need the following additions. Exhaustive destructuring also needs the added fields or `..` where ignoring them is appropriate.

| Public type | Added fields and migration |
|---|---|
| `AgentSpec` | `autonomy: AutonomyConfig::default()` preserves an ordinary agent |
| `StateDefinition`, `SkillDefinition` | `autonomy: None` preserves the previous scope |
| `ChatMessage` | `provenance: None` for ordinary messages; preserve actual task provenance when processing task history |
| `MemorySnapshot`, `ConversationContext` | `incomplete_native_exchanges: Vec::new()` for legacy data with no such observations; preserve real observations when present |
| `ToolExecutionContext` | `admitted_write_footprint: None` for a legacy context; never manufacture an admitted footprint to bypass task admission |
| `CommandResponse` | `execution_state: CommandExecutionState::Unspecified` for legacy reports; an interrupted response without acknowledgement remains uncertain |
| Eval `Assertion`, `TurnEvidence`, `TurnResult` | Optional `task` data; `TurnResult` also includes `interactions` |
| Eval `Scenario` | `interactions` for separately hosted participant controls; use an empty list for an ordinary scenario |
| Eval `FixturesConfig` | `files` for isolated fixture files; use an empty map when no files are required |
| Eval `TurnObservabilityEvidence` | `events` for actual collected observations; an empty list is not evidence that unobserved work did not occur |
| `EvalMetrics` | Task observation counters and status counts; missing measurements do not establish zero actual usage |
| CLI TUI `ModalState` | `review_scroll`; use the existing modal constructors when possible |

Existing exhaustive matches must handle `ContextSource::Instructions`, CLI `Command::Task`, eval `ScenarioStep::Task` and `ToolExecutionSource::Task`, and observation `EventType::TaskLifecycle`. Already non-exhaustive enums such as `AgentStreamEvent`, `AgentError` and `StorageCapability` still require their existing fallback arm. Whether to handle or ignore a new event depends on the consumer's purpose.

The following runtime methods now return errors when task ownership prevents mutation:

| Method | Previous return | Development return |
|---|---|---|
| `clear_actor_id` | `()` | `Result<()>` |
| `register_context_provider` | `()` | `Result<()>` |
| `set_session_metadata` | `()` | `Result<()>` |
| `remove_context` | `Option<Value>` | `Result<Option<Value>>` |

Propagate or handle the error, for example with `?` in a fallible host operation. Treat the runtime's refusal as a refused mutation; do not convert it to success or retry against another active owner. Calls used as statements may still compile while ignoring the new `Result`, but that is not correct error handling.

### Embedded TUI channels

`TuiQuestionHandler::new`, `tui::app::App::new` and `TuiLogLayer::new` now accept Tokio bounded `Sender<AppMessage>` instead of `UnboundedSender<AppMessage>`. Embedded TUI hosts must own a bounded channel and service its receiver. Choose capacity for that host's display workload. Required question/control delivery must handle awaiting capacity and receiver closure; the logging layer may discard display-only log hints when full. Do not use an unbounded forwarding task or silently discard decisions to satisfy the new constructor type.

### Extension implementations

The original required methods of `LLMProvider`, core and extended `Memory`, `Tool`, `AgentStorage`, `AgentHooks` and `ApprovalHandler` remain available. New defaulted methods allow ordinary implementations to compile without immediately implementing task capabilities. This does not make literals of the changed message, memory, tool-context or result types source-compatible.

Task participation requires the corresponding real capability. A custom memory implementation must preserve provenance and managed provider accounting; an ordinary storage backend does not acquire `TaskRuns` from snapshot support. Keep unsupported capability errors visible. Do not opt in merely to make admission succeed.

## YAML, stored data and reports

Omitted autonomy remains ordinary. Existing explicit settings retain their meaning; unknown framework keys remain errors. New task configuration is not expected to load in the older release.

Older messages and memory snapshots can be read with defaults for the new fields. Older readers may discard task provenance and incomplete native-history observations when saving new data again. Therefore readability is not a lossless downgrade promise. A custom memory backend must preserve the new information before declaring the corresponding task support.

Task checkpoints have their own envelope and runtime payload versions. Their decoding, configuration compatibility and executable restoration are separate checks. A serialized identity or successfully decoded checkpoint is not a live execution binding.

Published 1.1.2 eval output uses report schema **1**; development output uses schema **3**. Report readers should select supported schemas explicitly. Development reports distinguish a parent's response from task results and participant controls. `response_present: false` means there was no parent response; control results must not be substituted for it. Task observation metrics describe collected evidence, not a complete provider invoice. Shared reports remain redacted; exact recovery and assertion data require trusted host access.

Ordinary chat JSONL input remains supported. Task operations and participant interactions use strict YAML suites; they are rejected in the chat-only JSONL adapter. The older adapter ignored these extra keys, so remove them from ordinary chat rows or move intentional task/control scenarios to YAML. See [evaluation](@/docs/evaluation.md) for report and assertion semantics.

## Features and environments

The facade retains `default`, `sqlite`, `redis-storage`, `http-context`, `full-storage` and `full`. A workspace build can unify features from other members, including the CLI. Check an embedded consumer with its own selected features rather than inferring its available APIs from a workspace all-features build.

Linux and macOS have native CLI/PTY and MCP integration evidence. Windows CLI stack-overflow investigation and platform CI are deferred; that does not establish failure of every Windows library use. Redis remains an experimental snapshot backend, and SQLite currently has one documented audit exception pending an upstream dependency fix. Neither limitation is resolved by passing compatibility checks.

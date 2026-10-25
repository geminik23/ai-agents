+++
title = "Build, host and evaluate a task"
weight = 14
template = "docs.html"
description = "Configure a development task, install domain evidence, service typed decisions and evaluate the same host contract."
[extra]
learning_guide = true
+++

# From an objective to verified work

This guide covers the **development checkout**, not the published 1.1.1 API. A task asks when the host may stop, not just how often a model should answer. The same controller supports research, support, reports, code, operations, world-state and tool-free analysis. Normal chat remains ordinary unless an enabled state or selected skill promotes accepted input into a task, or the application explicitly invokes a task.

```text
YAML objective / strategy / gates / policies
                 +
Rust providers / tools / domain bindings / ceilings
                 |
          configured task runner
                 |
       actual work and observations
                 |
  continue / review / question / replan / stop
                 |
     acknowledged result or retained pause
                 |
        assertions and safe reports
```

## 1. Choose the smallest appropriate profile

Start from the source definitions in `examples/yaml/autonomy/`:

| Need | Definition | What establishes completion |
|---|---|---|
| Existing tools and validation | `code_fix.yaml` | Fresh command/diagnostic checks after scoped mutation |
| Research without local files | `research_task.yaml` | Distinct actual source identities, citations and semantic quality |
| Human-assisted support | `support_resolution.yaml` | Actual resolved ticket or recorded escalation |
| Non-file report | `report_records.yaml` | Actual artifact reference plus content/schema predicate |
| File report | `artifact_generation.yaml` | Authorized existence and content observations, with a real write-target ceiling |
| Operations | `ops_remediation.yaml` | Health or actual escalation, without skipping required stages |
| World or simulation | `world_state.yaml` | Host revision-bound world predicate |
| Tool-free refinement | `tool_free_analysis.yaml` | Managed judge/progress evidence within one allowance |
| Coordinating task composition | `composed_research.yaml` | Child and aggregator work under one coordinating owner/ledger |
| Ordinary parent with independent child tasks | `independent_research.yaml` | Separate acknowledged child results serviced before aggregation |

Changing a profile, stage order or completion policy normally requires YAML changes, not a new Rust execution loop. Named profiles inherit top-level defaults and can be narrowed by state/skill declarations. Scope, execution strategy and supervision are different decisions. See the full [field/default/merge reference](@/docs/yaml-reference.md#profile-fields-defaults-and-merging).

No profile universally requires tools, todos, filesystem access or SQLite. A requested capability must actually be installed. Registration, model-call source labels and approval do not bypass scope, policy, resource admission or current implementation bindings.

## 2. Bind application meaning, not a new framework gate

A support gate such as `completion: {validator_passed: records}` is meaningful only after an actual `host.records` implementation has been installed. The application owns ticket state, source identity, report schema or world revision; the controller owns evidence eligibility and completion.

`examples/rust/custom-llm/src/task_domain.rs` is a runtime-only integration module shared by the application and eval examples. Its `install_domain(builder)` installs:

- a tool that updates actual isolated records and their revision;
- strict named/versioned validators and a progress adapter;
- exact host target revision services;
- a content validator that requests authorized `file_read` observations instead of reading ambient files inside its callback;

The reusable installer does not replace caller hosting options, storage or ceilings. The eval executable explicitly supplies the one-path ceiling needed by its file-artifact profile; that ceiling does not grant a path.

For your own domain, replace the record integration and pure predicate/config validation, then select the installed IDs in YAML. Keep callback work bounded and pure. Return managed observation requests or a typed external wait for required I/O. Checkpoints and registered IDs do not serialize executable authority. See [validator and progress contracts](@/docs/rust-api.md#completion-validation-and-progress-development-only).

The module deliberately has **no eval dependency**. It owns its named adapter registry; combine additional domain adapters before applying that registry once, rather than replacing a previously captured implementation. Each installation creates fresh state; a retained pause keeps that actual installation alive. A new runtime needs an explicitly compatible application/reconstruction host rather than silently starting a new ticket record and calling it a restored task.

## 3. Run a real domain round trip offline

From the repository root:

```sh
cargo run --manifest-path examples/Cargo.toml -p example-custom-llm --bin task-domain-host --locked --offline
```

This executable loads `support_resolution.yaml`, installs that same domain module, then uses a finite offline provider. It starts an automatic state task, obtains the actual typed question, delivers a fixture answer, executes the domain tool and completes only after the actual ticket validator passes. It asserts cumulative model usage and prints a bounded safe display. It opens no socket and uses no customer system or provider credentials. Fixed replies prove this protocol, not live answer quality.

For a normal application, configure providers/features first, install your domain before final construction, declare whether you can service responses, and retain the runtime/runner across pauses. `build_task_host().await` prepares configured task-capable storage; it does not start an objective or install production restart support.

## 4. Service exact decisions and keep polling

Read the acknowledged operation result and use its run/revision to obtain `task_interaction`. The typed prompt distinguishes questions, approval, Continue, capacity, stage/plan review, external conditions and validation observations. Obtain the response from your authenticated UI and submit the corresponding `TaskResumeInput`. Never answer a reserved task by calling a new `chat()`.

A response is bound to the actual pending request. Silence is not endpoint loss; an event is not permission. Objective/limit changes are host controls, not model instructions, and do not refill usage or automatically continue work. A newly reconstructed pause requires separate restoration review before its original decision is answered.

An ordinary parent can remain pending while independent children ask questions. Use `TaskHostSession` around the original operation and its controls, discover only its actual participating runners, and poll both the parent and controls. Public ACK revision gates a usable prompt. Task-owned children remain under their coordinating request; do not answer a displayed nested leaf using a different controller. The library's [foreground session contract](@/docs/rust-api.md#foreground-host-scopes-and-sessions-development-only) is shared by embedded hosts, CLI/TUI and eval.

On shutdown, distinguish a running cancellation request from acknowledged paused cleanup. Continue polling the original work. Timeout, drop, abort and a stored terminal row do not prove quiescence, rollback or exactly-once external effects. Uncertain work can require explicit recovery and retain protection.

## 5. Evaluate that same integration

```sh
cargo run --manifest-path examples/Cargo.toml -p example-custom-llm --bin task-domain-evals --locked --offline -- examples/eval/mocked/autonomy/support_mocked.yaml target/eval/support
sh examples/eval/mocked/run_mocked_evals.sh --category autonomy
```

`task-domain-evals` calls `EvalRunner::with_host`. Its `EvalHost::configure` delegates to the same `install_domain`, preserving the eval-provided providers, budgets, hooks, workspace and policies. New attempts/resets install fresh isolated domain state; resume retains the current state and task. To add a domain, use your own executable with the same public traits. Do not modify the repository helper or add a core gate variant. The helper's fixed marker selects only this repository-owned binary; it is not a dynamic program/plugin loader.

Use `!task` for explicit run/skill/control steps and returned foreground pauses. Use scenario `interactions` when an ordinary parent still awaits independently hosted child decisions. See [typed steps](@/docs/evaluation.md#typed-steps-and-exact-continuation) and [foreground controls](@/docs/evaluation.md#foreground-participant-controls).

## 6. Read results without overclaiming

| Observation | What it proves | What it does not prove |
|---|---|---|
| A model says done | A candidate response exists | A ticket changed or a required gate passed |
| Successful tool record | That exact executor operation succeeded | Every task gate is satisfied |
| Current eligible validator Pass | Its configured domain predicate was verified | Permission to skip another required stage/check |
| Unknown or stale check | Evidence is unavailable/ineligible | Success under negation or a measured zero |
| Scenario passed | The expected contract was verified | The business task completed; expected rejection can pass |
| Safe display/event/report | Bounded projected observations | Raw recovery data or execution authority |
| Cancellation requested | The request was accepted | Acknowledged cleanup or rollback |

Development report schema 3 separates parent turns, participant controls, task outcomes and collection errors. Exact results, validator data, prompts and provider requests remain trusted-host/assertion-only evidence. Keep shared reports redacted. An interrupted foreground operation may retain completed control evidence without any parent response. Unacknowledged cleanup retains attempt files and blocks automatic retry/reset; inspect them only in an authorized host context.

## Troubleshooting and boundaries

- **Missing adapter/capability:** install the actual version/configuration and allowed operation before construction; do not weaken gates or infer authority from a name.
- **A state exits but task is incomplete:** check the declared objective and eligible completion evidence. Scope movement alone is not success.
- **An artifact exists but validation fails:** inspect the content/schema predicate and current target revision, not just the artifact path.
- **An answer is rejected:** check exact request kind, revision, original deadline and current policy/binding. Do not rebind a stale answer to the latest prompt.
- **No new progress:** inspect real eligible signals/high-water marks. Extra calls, repeated sources and rewritten plans do not replenish advancement or budgets.
- **A fixture fails before task admission:** report the capability/configuration error and assert no dispatch, rather than inventing an Unknown/completed task result.

These examples do not install an OS sandbox, scheduler, distributed queue, production restart recipe or arbitrary in-flight call-stack reconstruction. Release identity and publication remain separate from development implementation and acceptance.

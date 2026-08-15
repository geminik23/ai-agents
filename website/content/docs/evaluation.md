+++
title = "Evaluation"
weight = 4
template = "docs.html"
description = "Run YAML and JSONL scenario suites with fixtures, assertions, judges, and CI reports."
+++

Evaluation lets you test an agent through the normal runtime path. A suite is a separate YAML or JSONL file that describes inputs, fixtures, assertions, retries, and output reports.

Use evaluation for:

- CI smoke tests that do not need API keys
- state machine regression checks
- tool and fixture behavior checks
- memory and relationship behavior checks
- live provider smoke tests before releases
- semantic quality checks with an optional LLM judge

```text
agent.yaml
  + eval/suite.yaml
  -> ai-agents-cli eval
  -> normal RuntimeAgent path
  -> structured evidence
  -> assertions and optional judge
  -> reports
```

---

## Quick start: no API key

The fastest eval suite uses a mocked LLM response. This tests the runner, agent build path, assertions, and report generation without calling a provider.

```sh
cargo run -p ai-agents-cli -- eval \
  --agent examples/yaml/basic/simple_chat.yaml \
  --scenarios examples/eval/mocked/basic/simple_chat_mocked.yaml \
  --output target/eval/mocked/basic/simple_chat_mocked
```

Expected output summary:

```text
Eval complete: 1/1 passed, 0 failed, 0 skipped.
```

The runner writes:

```text
target/eval/mocked/basic/simple_chat_mocked/
  summary.md
  summary.json
  per_scenario.jsonl
  failures.md
```

Use `--junit` to also write `junit.xml`.

### Fail-closed configuration

Evaluation configuration is strict where a silent mistake could create false evidence:

- fixed suite, nested observability, fixture, and assertion objects reject unknown fields;
- `assert: {}`, empty `all`/`any`, and empty nested predicate collections are configuration errors;
- numeric bounds fail when the observed value is not numeric;
- ID, tag, or language filters that select zero scenarios return an error instead of a `0/0` success;
- replay requires an exact alias, model, and request-hash match and never falls back to a configured mock response;
- suite-declared `real` or `record` modes do not authorize provider calls by themselves.

Use `--dry-config-check` to validate the complete non-empty suite, deferred mock routes, and referenced agent without selecting scenarios, constructing providers, or writing reports. Use `--real-llm` or `--record <FILE>` to explicitly authorize provider-backed execution. Record destinations are created and opened before a provider is constructed or called, and aliases targeting the same cassette share one synchronized writer. Record-mode streaming is rejected before the provider call because atomic stream recording is not implemented.

Scenario env overlays mutate the process environment only while holding an exclusive guard and restore all values before releasing it. Attempts without env overlays share a read guard, so they can run together but cannot observe another attempt's temporary env values.

Compatibility JSONL rows and persisted cassette records may contain additional fields so newer producers can remain readable; the human-authored suite schema remains strict.

---

## Reading eval results

Start with the human-readable files, then move to machine-readable files when you need automation or deeper inspection.

```text
summary.md
  -> quick pass/fail overview
failures.md
  -> failed scenarios and failed assertion details
summary.json
  -> complete schema-versioned result for scripts
per_scenario.jsonl
  -> one scenario per line for large suites or notebooks
junit.xml
  -> CI test report when --junit is enabled
```

### Which file should I open first?

| File | Open when | What to look for |
|------|-----------|------------------|
| `summary.md` | You want the fastest human overview. | Suite name, pass count, failed count, skipped count, scenario table. |
| `failures.md` | Something failed and you want a triage view. | Failed assertion names, expected values, actual values, and failure messages. |
| `summary.json` | You want automation, dashboards, or detailed local debugging. | `schema_version`, `metrics`, `scenarios`, attempts, turns, and assertion results. |
| `per_scenario.jsonl` | You run large suites or want streaming post-processing. | One serialized scenario result per line. |
| `junit.xml` | Your CI understands JUnit reports. | Test case failures, errors, and skipped scenarios. |

### Status values

Each scenario has one final status.

| Status | Meaning | Typical next step |
|--------|---------|-------------------|
| `passed` | All declared assertions passed on the winning attempt. | No action. |
| `failed` | The agent ran, but at least one assertion was false. | Open `failures.md`, then inspect the failed assertion in `summary.json`. |
| `error` | Runtime, setup, timeout, fixture, or judge execution failed. | Check the error message and failure category. |
| `skipped` | The scenario had `skip: true` or a skip reason. | Confirm the skip is intentional. |

A scenario can also be marked `flaky: true` when an earlier attempt failed but a later retry passed. Treat flaky passes as warnings, especially for release gates.

### Failure categories

`failure_category` explains what kind of failure happened.

| Category | Meaning | Common causes |
|----------|---------|---------------|
| `config_error` | Suite or runner configuration was invalid before a useful result could be produced. | Missing agent path, duplicate scenario IDs, unsupported isolation, unsafe parallel/env combination. |
| `runtime_error` | The agent or fixture path failed while executing. | Provider failure, turn timeout, stream error, storage setup failure, mock server issue. |
| `assertion_failed` | A deterministic assertion failed. | Wrong state, missing tool call, missing text, context mismatch, observability count mismatch. |
| `judge_error` | A judge assertion failed or the judge could not return valid output. | Missing judge LLM alias, invalid judge JSON, threshold too high, semantic criteria too strict. |
| `flaky_pass` | A retry passed after an earlier failed attempt. | Nondeterministic live model output, transient provider issue, timing-sensitive behavior. |

### Understanding assertion results

Each turn contains `assertion_results`. These are the smallest units to inspect when a scenario fails.

```json
{
  "assertion": "response_contains",
  "passed": false,
  "actual": "[redacted]",
  "expected": "[redacted]",
  "message": "..."
}
```

Important notes:

- `assertion` tells you which assertion family produced the detail.
- `passed` is the result for that one detail, not the whole scenario.
- `actual` and `expected` may be redacted by default.
- `message` is present when the evaluator can provide a clearer explanation.

If you need exact actual/expected values for local debugging, set this only in a trusted local suite:

```yaml
settings:
  redact_outputs: false
```

Do not disable redaction for shared CI when scenarios may include private user data, facts, relationship state, tool arguments, or tool outputs.

### Debugging by failure type

```text
failed assertion
  -> open failures.md
  -> find scenario ID
  -> inspect failed assertion name
  -> inspect related suite assertion
  -> rerun only that scenario with --id
```

```sh
cargo run -p ai-agents-cli -- eval \
  --agent examples/yaml/basic/simple_chat.yaml \
  --scenarios examples/eval/mocked/basic/simple_chat_mocked.yaml \
  --id hello-smoke \
  --output target/eval/debug_one
```

For `runtime_error`:

- check provider credentials if using `real` or `record`
- check turn and scenario timeout values
- check mock server routes if an HTTP tool is involved
- check storage paths if persistence is involved

For `judge_error`:

- make sure `judge.llm` points to a valid alias, or omit it to use router/default
- use a real judge-capable LLM or provide a mock response shaped like judge JSON
- lower `pass_threshold` if the criteria are intentionally broad
- make criteria concrete and language-agnostic

For replay drift:

- confirm the agent prompt did not change
- confirm the model name and LLM alias did not change
- confirm the input and relevant context did not change
- regenerate the cassette with `--record` if the expected prompt changed

### Metrics and trend analysis

`summary.json.metrics` is useful for dashboards and release checks.

| Metric | Use |
|--------|-----|
| `pass_rate` | Overall suite health. |
| `total_turns` | Size of the executed workload. |
| `errors` | Runtime/setup instability. |
| `flaky` | Nondeterminism or transient instability. |
| `avg_latency_ms`, `p50_latency_ms`, `p90_latency_ms`, `p99_latency_ms` | Performance trend tracking. |
| `by_tag` | Compare behavior by feature area such as `tools`, `memory`, `live`, or `smoke`. |
| `by_language` | Compare multilingual behavior. |
| `by_assertion` | Find which assertion family is failing most often. |
| `by_failure_category` | Separate runtime instability from behavior regressions. |

For large suites, `per_scenario.jsonl` is easier to process incrementally than one large JSON file.

### Observability results

When suite-level `observability:` or CLI `--observability` is enabled, `summary.json` can include an `observability` report.

Use it to answer questions such as:

- how many LLM calls happened?
- how many tool calls happened?
- which operation purpose consumed calls or tokens?
- did an eval run exceed an expected cost or token bound?
- did a scenario unexpectedly call the model more times than before?

Example observability assertion:

```yaml
assert:
  observability:
    total_llm_calls_lte: 2
    total_tool_calls_lte: 0
    purpose_counts:
      main_response:
        path: count
        gte: 1
```

This is best for regression gates around call count, token use, cost, and configured dimensions such as `background`, not for judging response quality. Eval captures an observability cursor before each turn. `TurnObservabilityEvidence` uses one post-cursor snapshot so its report, trace ID, and span IDs derive from the exact same events that remain retained in the manager's rolling metrics window; evicted events are not recovered. In contrast, direct `ObservabilityManager` report, raw-event, and export APIs retain their rolling global window.

Dimension-count assertions use the aggregate dimensions configured on the agent or eval observability overlay:

```yaml
assert:
  observability:
    dimension_counts:
      - match_dimensions:
          background: "true"
        assert:
          path: count
          gte: 1
```

For speculative branch checks, configure branch dimensions and assert committed, discarded, failed, or cancelled outcomes:

```yaml
observability:
  enabled: true
  aggregation:
    dimensions: [branch_status, runtime_optimization, commit_behavior, speculative]

scenarios:
  - id: branch-status-smoke
    turns:
      - input: I was charged twice
        assert:
          observability:
            dimension_counts:
              - match_dimensions:
                  branch_status: committed
                  commit_behavior: final_response
                  speculative: "true"
                assert:
                  path: count
                  gte: 1
              - match_dimensions:
                  branch_status: cancelled
                  speculative: "true"
                assert:
                  path: count
                  gte: 1
```

Branch raw events also carry both `speculative` and `runtime.speculative` dimensions for export and debugging. Dimension-count assertions operate on configured aggregate dimensions, so include `speculative` when you want to match speculative branch metrics. `branch_status: committed` means the branch won selection into the committed runtime path; it does not prove transaction success or rollback of external effects. Branch telemetry is the stable way to inspect speculative LLM work because losing branches do not fire final response hooks.

Runtime optimization can schedule background actor memory maintenance. The eval runner flushes background runtime tasks before collecting turn evidence, so facts, relationships, and observability assertions see the completed post-turn state. The example suite `examples/eval/mocked/runtime-optimization/pre_response_transition_mocked.yaml` shows a no-key regression check for pre-response guard routing, and the speculative suites cover committed, discarded, failed, and cancelled branch status dimensions.

---

## Suite shape

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
    language: en
    turns:
      - input: Hello
        assert:
          response_not_empty: true
          response_contains: "Hello"
```

Important parts:

| Field | Meaning |
|-------|---------|
| `name` | Report name. |
| `agent` | Agent YAML path, resolved relative to the suite file. CLI `--agent` can override it. |
| `settings` | Timeouts, retries, isolation, parallelism, and redaction. |
| `fixtures` | Mock/replay/record/real LLM modes, sequenced outcomes, attempt-local values, mock or real-tool transports, context, diagnostics, commands, and deterministic HITL approvals. |
| `scenarios` | Test cases with optional hard provider budgets and direct turns or advanced steps. |
| `assert` | Assertions evaluated after a turn. |

`settings.timeout_per_turn_ms` and a turn-level `timeout_ms` bound the whole turn, including provider calls, HITL waits, resource-lock waits, tool attempts and retries, streaming collection, and finalization. `settings.timeout_per_scenario_ms` separately bounds the complete scenario attempt across its turns and steps. These are different from `tool_security.*.timeout_ms`, which applies to each individual `Tool::execute` invocation attempt, and from `command.timeout_ms`, which bounds the direct child process. No timeout layer promises rollback of effects that already escaped to a filesystem, process, network, or custom integration.

---

## Scenario provider budgets

Use a scenario `budget` for live or otherwise provider-backed flows that can make more than one LLM call:

```yaml
settings:
  retries: 0

observability:
  enabled: true
  cost:
    enabled: true
    pricing_file: ../../../../yaml/observability/pricing.yaml
    unknown_price_policy: error

scenarios:
  - id: bounded-pipeline
    budget:
      max_llm_calls: 5
      max_total_tokens: 60000
      max_cost_usd: 0.03
    turns:
      - input: Run the bounded pipeline.
```

`max_llm_calls` limits calls across every alias, child agent using the shared registry, turn, agent reset, and retry in that scenario. Token and cost limits reserve a conservative maximum before each provider call, then settle against provider usage or a fallback estimate. A call is rejected before it starts when its reservation would exceed a configured limit; if reported usage exceeds the reservation, the response fails and no later provider call may start.

`max_cost_usd` requires suite-level `observability.cost` pricing for every provider/model alias that may execute. Missing pricing fails before the provider call. Budget values must be finite and greater than zero. Scenario retries share the same budget rather than receiving a fresh allowance, so a declared budget must cover every intended attempt.

Post-run assertions such as `observability.total_llm_calls_lte` run after usage has occurred, while a declared scenario budget guards before provider execution. A dry configuration check validates schema and budget wiring without executing a provider.

---

## Execution flow

```text
load suite
  -> validate suite
  -> apply CLI overrides
  -> filter scenarios by ID, tag, language
  -> schedule serial or parallel scenarios
  -> create attempt workspace
  -> apply fixtures
  -> build RuntimeAgent through AgentBuilder
  -> apply context and actor
  -> run turns or steps
  -> collect TurnEvidence
  -> evaluate assertions
  -> compute metrics
  -> write reports
```

The runner uses the same builder path as normal agents. Fixtures replace selected dependencies before the agent is built.

### Mocked LLM delays

Use `fixtures.llm.delays_by_alias` when a no-key suite needs deterministic branch ordering, such as proving that a fast route cancels a slow draft branch.

```yaml
fixtures:
  llm:
    mode: mock
    responses_by_alias:
      default:
        - "Committed response."
      router:
        - "1"
    delays_by_alias:
      default: 50
```

Delays are in milliseconds and apply to mocked or replay providers for the named alias.

### Sequenced LLM outcomes

Use `fixtures.llm.outcomes_by_alias` to mix deterministic responses and provider errors for one alias. This is useful for retry and recovery contracts.

```yaml
fixtures:
  llm:
    mode: mock
    outcomes_by_alias:
      default:
        - type: error
          message: transient provider outage
          status: 503
        - type: response
          content: Recovered response.
```

Each alias has an independent cursor. The last outcome repeats after exhaustion. Error `status` is optional; status values such as `503` preserve the runtime recovery classification.

### Attempt-local values and real-tool fixtures

Each scenario attempt receives an opaque absolute workspace. Runtime context includes `eval.workspace` and includes `mock_server.base_url` when the local server is enabled. Parent and spawner file, SQLite, and Redis storage are isolated per attempt.

When scenario retries are enabled, `timeout_per_scenario_ms` is applied independently to each attempt. Retry delay begins only after the completed attempt timeout has ended, so it does not consume that prior attempt's timeout budget, but it remains part of the scenario runner's total wall time.

Mock response strings can interpolate only `{{ eval.workspace }}` and `{{ mock_server.base_url }}`. JSON tool calls are rewritten as parsed JSON, so native path separators remain valid. Other template expressions remain unchanged, and a mock-server token without an enabled server is a configuration error.

```yaml
fixtures:
  mock_server:
    enabled: true
    routes:
      - method: GET
        path: /status
        status: 200
        body: { healthy: true }
  llm:
    mode: mock
    responses:
      - '{"tool":"http","arguments":{"method":"GET","url":"{{ mock_server.base_url }}/status"}}'
```

Use `workspace_policy` only when the source agent already has an explicit tool policy and a real tool must access the isolated workspace:

```yaml
fixtures:
  workspace_policy:
    write_tools: [file_write]
```

The overlay adds only the generated workspace to the listed existing policies. It does not disable fail-closed behavior, blocked paths, confirmation, or unrelated policy rules.

The runner owns the temporary attempt workspace and removes it after completion, failure, or cancellation.

Use `web_fetch_transport` to execute the real `web_fetch` implementation through exact-URL in-memory routes without opening a socket:

```yaml
fixtures:
  web_fetch_transport:
    routes:
      - url: https://docs.example.test/article
        status: 200
        headers:
          content-type: text/html
        body: "<h1>Fixture article</h1>"
```

The real URL, domain, address, redirect, byte, cache, and output checks still run. Localhost and private-network targets remain blocked.

Use `fixtures.web_search` to install a deterministic exact-query `WebSearchProvider` without public search traffic. Set `available: false` to exercise the runtime's pre-execution unavailable path; an available fixture with no matching query returns an empty available result.

```yaml
fixtures:
  web_search:
    available: true
    responses:
      "ai agents rust":
        provider: static
        results:
          - title: AI Agents Framework
            url: https://ai-agents.rs/
            snippet: YAML-first agents in Rust.
```

### Mocked diagnostics fixture

Use `fixtures.diagnostics` when you want deterministic coverage for the `diagnostics` tool without a live IDE or compiler host:

```yaml
fixtures:
  llm:
    mode: mock
    responses:
      - "There is one Rust error in src/main.rs."
  diagnostics:
    available: true
    items:
      - path: src/main.rs
        line: 12
        column: 5
        severity: error
        source: rustc
        message: cannot find function `runn` in this scope
        code: E0425
```

When no diagnostics fixture or host provider is installed, the shared executor records an explicit unavailable result instead of invoking the tool implementation or hanging.

### Mocked command fixture

Use `fixtures.commands` when you want deterministic coverage for the `command` tool without spawning a real process:

```yaml
fixtures:
  llm:
    mode: mock
    responses:
      - '{"tool":"command","arguments":{"argv":["cargo","fmt","--all"],"cwd":"."}}'
      - "Formatting completed successfully."
  commands:
    available: true
    entries:
      - argv: [cargo, fmt, --all]
        response:
          success: true
          exit_code: 0
          termination: exited
          stdout: "Formatting complete"
          stderr: ""
          combined_output: "Formatting complete"
          truncated: false
          timed_out: false
          cwd: "."
          argv_redacted: [cargo, fmt, --all]
```

When no command fixture or host runner is installed, the shared executor records an explicit unavailable result for `command` before the tool tries to execute.

Tool evidence comes from executor-level `ToolExecutionRecord` values. `tool_called` assertions can match identity, count, execution, success, source, arguments, and structured successful output. Non-executed denial, unavailability, approval rejection, timeout, and cancellation records can be matched with `executed: false`, but that predicate alone does not identify the reason; current `tool_called` predicates do not directly match `cancelled`, `timed_out`, or tool metadata. Shared-executor `on_tool_start` and `on_tool_complete` hooks describe one request lifecycle rather than individual retry invocations. A non-executed request can finalize before or after its start hook, so eval treats `ToolExecutionRecord.executed` as authoritative.

### Deterministic HITL approval fixtures

Use `fixtures.approvals` to run approval paths without a person, real HTTP, or another live approval service. Rules are evaluated in declaration order and the first matching rule whose optional `occurrence` matches supplies the outcome. `occurrence` is 1-based and counted separately for each rule when evaluation reaches that rule and its trigger predicate matches; later rules are not visited after a winner is selected. If no rule matches, `default` is used; omitting `default` is equivalent to `outcome: unavailable`.

This example uses the exact fixture shape from `examples/eval/mocked/hitl/hitl_basic_mocked.yaml`:

```yaml
fixtures:
  approvals:
    preferred_language: en
    supported_languages: [en]
    rules:
      - trigger: { type: tool, name: http }
        occurrence: 3
        outcome: timeout
      - trigger: { type: tool, name: http }
        occurrence: 2
        outcome: reject
        reason: fixture rejection
      - trigger: { type: tool, name: http }
        occurrence: 1
        outcome: approve
    default:
      outcome: reject
      reason: unexpected approval request
```

A tool trigger can also require exact arguments. The complete `args` value must equal the request arguments:

```yaml
- trigger:
    type: tool
    name: http
    args:
      method: GET
      url: "https://api.example.test/en"
  outcome: approve
```

Current trigger forms are:

```yaml
# Each YAML document below is a separate alternative.
# Tool name, with optional exact args.
trigger: { type: tool, name: http }

---
# Condition name, with optional exact matched expression.
trigger:
  type: condition
  name: state_changing_http
  matched: "method in [POST, PUT, DELETE, PATCH]"

---
# State transition. from is optional; to is required.
trigger: { type: state_transition, from: review, to: complete }

---
# Disambiguation escalation, with an optional exact reason.
trigger: { type: disambiguation_escalation, reason: unclear }
```

Current outcome forms are:

```yaml
# Each YAML document below is a separate alternative.
default:
  outcome: approve

---
default:
  outcome: reject
  reason: "Unexpected approval request"

---
default:
  outcome: modify
  changes:
    url: "https://api.example.test/v1/safe-items/42"
    body: '{"name":"safe-item"}'

---
default:
  outcome: timeout

---
default:
  outcome: unavailable
```

`changes` is a top-level argument patch. The runtime applies its entries to the original tool argument object to produce the complete modified and effective arguments. `unavailable` is deliberately fail-closed: the fixture handler returns a rejection with reason `Approval fixture unavailable`; it does not permit execution or wait for a person.

`preferred_language` and `supported_languages` are advertised by the fixture handler to the HITL message resolver. They control the localized approval message generated by the agent's HITL configuration; assertions can then check that exact message. For example, the multilingual suite verifies English, Korean, and Japanese messages:

```yaml
approval_requested:
  count: 1
  trigger:
    type: tool
    name: http
  message: "https://api.example.test/ko에 POST 요청을 승인하시겠습니까?"
  raw_decision: rejected
  effective_decision: rejected
  rejection_reason: "Mocked Korean rejection"
```

Each scenario attempt builds a fresh fixture handler and fresh occurrence counters. Counters are shared across turns and agent resets within that attempt, but retries and concurrently running scenario attempts cannot consume one another's occurrences. This makes ordered multi-turn rules deterministic while keeping retries isolated.

These fixtures replace only approval input. Mock the LLM and the approved tool too when a suite must remain deterministic and side-effect-free. Real HTTP and real human input remain excluded from mocked HITL evals.

### HITL evidence and assertions

Every fully resolved approval request is retained as turn evidence with its normalized trigger, raw and effective decisions, original/modified/effective arguments, localized message, rejection reason, and resolution error. Use `approval_requested` to require a matching record and `approval_not_requested` to prove that no record matches the optional filters.

```yaml
assert:
  approval_requested:
    count: 1
    trigger:
      type: condition
      name: state_changing_http
      matched: "method in [POST, PUT, DELETE, PATCH]"
    raw_decision: modified
    effective_decision: modified
    message: "Approve PUT request to https://api.example.test/v1/items/42?"
    original_args:
      path: url
      eq: "https://api.example.test/v1/items/42"
    modified_args:
      path: url
      eq: "https://api.example.test/v1/safe-items/42"
    effective_args:
      path: body
      eq: '{"name":"safe-item"}'
```

`approval_requested` accepts `true`/`false` or an object. Object filters include `count`, `count_gte`, `count_lte`, `trigger`, `raw_decision`, `effective_decision`, `message`, `message_contains`, `rejection_reason`, `rejection_reason_contains`, `error`, `error_contains`, `original_args`, `modified_args`, and `effective_args`. Argument fields are path assertions; aliases `args_original`, `args_modified`, and `args_effective` are also accepted. All filters in one object must match the same approval record before its count is included.

```yaml
assert:
  approval_not_requested:
    trigger:
      type: condition
      name: state_changing_http
```

For `approval_not_requested`, boolean form checks whether the turn has no approval records; object form passes only when no record matches its filters. Assertion trigger types are normalized as `tool`, `condition`, or `state` (state assertions use optional `from` and `to`), even though the fixture rule spelling is `state_transition`.

Raw and effective decisions are intentionally distinct. `raw_decision` is what the handler returned: `approved`, `rejected`, `modified`, or `timeout`. `effective_decision` is what the runtime resolved: `approved`, `rejected`, `modified`, or `error`. For example, the supplied HITL agents resolve timeout to rejection, so suites assert `raw_decision: timeout` with `effective_decision: rejected`; another timeout policy can resolve to `error`. Rejected, timed-out, and errored requests have no effective executable arguments. Pair approval assertions with `tool_called.executed` to prove whether the wrapped tool ran. An expected rejection that ends the agent loop is a finalized cancellation response: blocking and event-stream turns retain the authoritative response, and an event stream emits `Final(AgentResponse)`. Provider failure, root-future or streaming-task abort, whole-turn timeout, and consumer drop can instead leave the stream incomplete. Runtime-control tool cancellation may still be represented in a later finalized response if the root turn recovers.

A turn that is expected to end in a runtime error can declare one substring or a list of alternatives:

```yaml
turns:
  - input: Trigger the approval error path.
    expect_error: "approval timed out"
    assert:
      approval_requested:
        raw_decision: timeout
        effective_decision: error
        error_contains: "timed out"
```

```yaml
expect_error: [timeout, unavailable]
```

`expect_error` is per turn and uses substring matching; list form passes when any entry matches. A matching error adds a passing `expect_error` assertion instead of marking the scenario as a runtime error. A missing or non-matching error fails that assertion. The runner collects and retains the errored turn before stopping, so approval records, tool records, state, latency, and other evidence produced during that turn remain available to assertions and metrics. Errors that happen before a turn result exists, such as setup failure or an outer scenario-attempt timeout, cannot retain turn evidence.

Approval evidence is evaluated in memory before report redaction. Raw `TurnEvidence` is never serialized to JSON or JSONL, and sensitive approval fields such as arguments, localized messages, rejection reasons, and errors are marked non-serializing. With default `settings.redact_outputs: true`, assertion `actual` and `expected` values are redacted too. `settings.redact_outputs: false` can expose ordinary assertion and runtime-error detail for trusted local debugging, but it still does not serialize raw approval evidence.

### Focused built-in tool evals

The examples directory includes no-key suites for both success and denial paths:

```text
examples/eval/mocked/hitl/hitl_basic_mocked.yaml                -> ordered approve, reject, and timeout outcomes
examples/eval/mocked/hitl/hitl_conditions_mocked.yaml           -> bypass, approve, reject, modify, and timeout by condition
examples/eval/mocked/hitl/hitl_multilingual_mocked.yaml         -> localized English, Korean, and Japanese approval messages
examples/eval/mocked/tools/code_search_mocked.yaml              -> grep search
examples/eval/mocked/tools/file_write_dry_run_mocked.yaml       -> file_write dry run
examples/eval/mocked/tools/file_edit_review_mocked.yaml         -> file_edit dry run
examples/eval/mocked/tools/file_edit_denied_mocked.yaml         -> blocked write path
examples/eval/mocked/tools/file_edit_approval_rejected_mocked.yaml -> approval rejected before execution
examples/eval/mocked/tools/patch_review_mocked.yaml             -> patch dry run
examples/eval/mocked/tools/ask_user_fallback_mocked.yaml        -> ask_user default fallback
examples/eval/mocked/tools/web_fetch_policy_mocked.yaml         -> blocked URL policy
examples/eval/mocked/tools/diagnostics_mocked.yaml              -> diagnostics fixture
examples/eval/mocked/tools/command_validation_mocked.yaml       -> command allowlist
examples/eval/mocked/tools/command_blocked_mocked.yaml          -> blocked command
examples/eval/mocked/tools/sleep_wait_mocked.yaml               -> bounded sleep wait
examples/eval/mocked/basic/no_tools_explicit_empty_mocked.yaml  -> tools: [] denies calls
examples/eval/mocked/runtime-optimization/speculative_losing_tool_draft_mocked.yaml -> losing branch tool call stays inert
```

Use these as templates for permission-denial, approval, host-fixture, and no-tools regression coverage.

---

## LLM modes

Release-blocking tool-execution smoke suites should use explicit `tool_choice: required` or a specific canonical tool so they prove provider request mapping, normalized calls, and actual shared-executor evidence. A response that merely contains a UUID, JSON value, or other tool-shaped text does not satisfy `tool_called` with `executed: true`. Automatic `tool_choice: auto` discovery measures provider/model behavior and should be reported as quality evidence rather than deterministic framework correctness.

The `examples/eval/live/run_live_example_evals.sh` helper discovers only `examples/eval/live/examples/`. It intentionally excludes `examples/eval/live/quality/`, including judge-based quality suites, which must be run separately. Explicit required/specific choices may add one corrective provider call on a non-native fallback, and that call uses the same scenario call, token, cost, timeout, and cancellation budgets.

A streamed eval turn consumes the complete runtime event stream. Content chunks remain diagnostic previews, while the terminal `Final(AgentResponse)` supplies the authoritative response content and metadata used by assertions. Streaming turns can use `metadata_contains` and `metadata_path`. Provider failure, root-future or streaming-task abort, consumer drop, or whole-turn timeout can leave the stream incomplete without `Final`; an expected HITL rejection instead finalizes an authoritative cancellation response and is evaluated as a completed turn. Runtime-control tool cancellation does not by itself make the root stream incomplete. Partial chunk text from an incomplete stream is retained only for failure diagnostics.

`fixtures.llm.mode` controls how the eval runner supplies LLM providers.

| Mode | Use when | Behavior |
|------|----------|----------|
| `mock` | Default CI and deterministic tests | Uses `fixtures.llm.responses` in order. When responses run out, the last response repeats. |
| `replay` | Stable regression from prior real traffic | Loads a JSONL cassette and requires an exact alias, model, and request-hash match. A miss is an error. |
| `record` | Creating or updating a cassette | Requires `--record`, preflights the cassette, calls the real provider, and appends synchronized JSONL records. Streaming is rejected before the call. |
| `real` | Live provider smoke tests | Requires `--real-llm`, uses the provider configured by the agent YAML, and may incur network cost. |

### Mock mode

```yaml
fixtures:
  llm:
    mode: mock
    responses:
      - "First response"
      - "Second response"
```

Flow:

```text
turn 1 -> response[0]
turn 2 -> response[1]
turn 3 -> response[1] again
```

Use mock mode for default CI because it is fast, stable, and does not require API keys.

### Record mode: save previous real responses

Use `record` when you want to capture real provider behavior once, then replay it later.

```sh
cargo run -p ai-agents-cli -- eval \
  --agent examples/yaml/basic/simple_chat.yaml \
  --scenarios examples/eval/live/quality/basic/simple_chat_semantic_judge_live.yaml \
  --output target/eval/record_live \
  --record target/eval/cassettes/live.jsonl
```

What happens:

```text
open and validate cassette destination
  -> construct real provider
  -> real provider call
  -> append and flush one synchronized cassette record
  -> return response to eval
```

If destination setup or a cassette write fails, the eval returns an error. Multiple aliases writing the same cassette are serialized so records remain valid JSONL. Record mode currently supports blocking completion only; a streaming turn fails before making the provider request.

Cassette records store:

```text
alias
model
request_hash
response
```

They do not store raw prompt fields separately. Treat cassette files as test artifacts because responses can still include user-visible content.

### Replay mode: load previous responses

Use `replay` to run without live provider calls after recording.

```sh
cargo run -p ai-agents-cli -- eval \
  --agent examples/yaml/basic/simple_chat.yaml \
  --scenarios examples/eval/live/quality/basic/simple_chat_semantic_judge_live.yaml \
  --output target/eval/replay_live \
  --replay target/eval/cassettes/live.jsonl
```

Replay flow:

```text
runtime LLM request
  -> compute request hash from messages and config
  -> find cassette record with the same alias, model, and hash
  -> return the recorded response
  -> if no exact match exists, return an error
```

If replay unexpectedly uses the wrong response, check:

- the agent prompt did not change
- the model name did not change
- the LLM alias did not change
- the suite input did not change
- the cassette file path is correct

### Real mode

Use `real` only for live smoke tests. A suite may declare its intended mode, but that declaration does not authorize provider access:

```yaml
fixtures:
  llm:
    mode: real
```

The command must also opt in explicitly:

```sh
cargo run -p ai-agents-cli -- eval \
  --agent examples/yaml/basic/simple_chat.yaml \
  --scenarios examples/eval/live/quality/basic/simple_chat_semantic_judge_live.yaml \
  --output target/eval/live/quality/basic/simple_chat_semantic_judge \
  --real-llm
```

Real mode needs provider credentials such as `OPENAI_API_KEY`, network access, and acceptance of provider cost and nondeterminism. Without `--real-llm`, a suite that resolves to `mode: real` fails during configuration before constructing the provider.

### Live example suites

The examples tree also contains live suites that exercise runnable YAML examples through real model behavior while keeping external effects read-only, fixture-backed, no-socket, dry-run-only, or protected by hard provider budgets. Current suites cover tools, skills, threshold-aware disambiguation, memory and cross-runtime session persistence, actor isolation, personas and relationships, public plan-and-execute outcomes, observability and recovery, exact state lifecycle transitions, selected runtime optimizations, one fixed orchestration pipeline, context injection and rendering, and input/output process behavior.

```sh
cargo run -p ai-agents-cli -- eval \
  --scenarios examples/eval/live/examples/tools/code_search_live.yaml \
  --output target/eval/live/examples/tools/code_search \
  --real-llm
```

These suites declare their own `agent` path. The helper can list suites, parse-check configuration, or intentionally run real-provider checks by category:

```sh
sh examples/eval/live/run_live_example_evals.sh --list
sh examples/eval/live/run_live_example_evals.sh --dry-config-check --category context
sh examples/eval/live/run_live_example_evals.sh --yes-live --category process
```

Implemented categories currently include `basic`, `context`, `disambiguation`, `error-recovery`, `memory`, `observability`, `orchestration`, `persona`, `process`, `reasoning`, `relationships`, `runtime-optimization`, `session`, `skills`, `state-machine`, and `tools`. Use `examples/eval/live/README.md` as the authoritative registry instead of maintaining a category tree in public documentation.

Live suites should usually check one primary behavior per scenario. If several safe tools can satisfy the same read-only request, use `any` over structural `tool_called` assertions. Prefer concrete prompts such as "Before answering, call the file_read tool ..." when the suite requires tool evidence. Add deterministic response checks for stable result values, requested symbols, fixture details, and dry-run wording so the suite verifies minimum useful user-visible output. Persistence suites should reset or rebuild the runtime with attempt-local storage preserved so ordinary conversation history cannot satisfy recall. Keep exact multi-tool sequences, denial paths, unavailable providers, approval behavior, and response-quality judges in mocked or focused suites where the model output is deterministic.

Advanced `steps` support persistence and actor-isolation checks:

```yaml
steps:
  - !set_actor
    actor: customer_42
  - !run
    turns:
      - input: My project code is NORTHSTAR-42.
  - !reset_agent
    profile: full_runtime
    preserve_storage: true
    preserve_actor_id: true
  - !run
    turns:
      - input: Return my persisted project code.
        assert:
          llm_request:
            system_contains: NORTHSTAR-42
```

`profile: conversation` clears ordinary conversation state without rebuilding. `profile: full_runtime` rebuilds the agent; with `preserve_storage` and `preserve_actor_id`, it can prove that actor facts are loaded from isolated persistence rather than recalled from the prior message list. Use `!set_context` between conversation resets for context-derived multi-actor identity.

Exact retry and fallback sequences, context-overflow recovery, chain-of-thought traces, subjective reflection improvement, and restart-based persistence remain mocked, quality-only, or manual where live execution would be costly, private, or misleading.

---

## LLM judge assertions

A judge assertion asks an LLM to score semantic quality. Use it when literal checks are too brittle.

```yaml
assert:
  judge:
    llm: router
    pass_threshold: 0.8
    criteria:
      - name: relevance
        description: Response directly addresses the user's request.
        weight: 1.0
```

If `llm` is omitted, the runner uses the router alias, then the default alias. The judge must return strict JSON. Judge failures are categorized separately from deterministic assertion failures.

A judge sees the final response plus limited scenario context. It does not see raw tool records, state history, denied-tool evidence, or dry-run flags. Check those with structural assertions instead.

Use judge assertions sparingly in default CI and live smoke suites. Prefer deterministic checks for response text, state, context, tools, metadata, facts, relationships, orchestration, and observability.

---

## Tool and HTTP fixtures

Mock a tool by ID:

```yaml
fixtures:
  tools:
    lookup_order:
      success: true
      output:
        id: ORD-1042
        status: cancellable
```

A mock tool with the same ID as a built-in replaces that built-in during eval.

For the built-in `command` tool, prefer `fixtures.commands` so the shared executor still records the normal command-tool path instead of replacing the tool entirely.

For HTTP-style tests, use the mock server:

```yaml
fixtures:
  mock_server:
    enabled: true
    routes:
      - method: GET
        path: /orders/ORD-1042
        status: 200
        body:
          id: ORD-1042
          status: cancellable
```

The runner injects:

```yaml
mock_server:
  base_url: http://127.0.0.1:<dynamic-port>
```

Use that context value in agent prompts, process stages, or tool arguments.

---

## Assertions

Assertions are implicit `all` when several fields appear in one object:

```yaml
assert:
  response_not_empty: true
  response_contains: "Hello"
```

Use explicit composition for branching:

```yaml
assert:
  any:
    - state: clarification
    - response_contains: "Can you clarify"
```

For live provider suites, avoid coupling several exact tool calls in one turn. Split coverage into focused scenarios, use `any` when equivalent safe tools are acceptable, and add deterministic response checks for stable values or bounded wording. Response substring checks are literal, so use `response_contains_any` for known casing or wording variants and avoid tiny substrings that can pass unrelated responses.

Common deterministic assertions:

| Assertion | Checks |
|-----------|--------|
| `response_contains` / `response_contains_any` | Stable substrings that should appear in the final response. |
| `response_not_contains` | Complete overclaim phrases that must not appear in the final response. |
| `state` / `state_in` / `state_not` | Current state after the turn. |
| `state_history_contains` | Transition history. |
| `metadata_contains` | Top-level response metadata. |
| `metadata_path` / `context_path` | Dot-path assertions. |
| `tool_called` | Tool ID, executed flag, success, source, arguments, and output matched on one execution record. Plan steps use source `plan`. |
| `llm_request` | In-memory role-specific message content and request counts without serializing prompts. |
| `approval_requested` / `approval_not_requested` | Approval trigger, raw/effective decision, localized message, reasons/errors, argument versions, and counts. |
| `facts_include` | Actor facts by actor/category and optional semantic judge. |
| `relationship` | Relationship dimensions and counts. |
| `orchestration` | Pattern, final agent, included agents, stage count. |
| `observability` | LLM/tool/token/cost/purpose/status counts. |

Tool assertion example:

```yaml
assert:
  tool_called:
    id: calculator
    count: 1
    executed: true
    success: true
    source_in: [plan]
    args_executed:
      path: expression
      eq: "18 * 7"
    result_path:
      path: result
      eq: 126
```

Every configured object predicate must match the same execution record. `count` and `count_gte` count complete matches after ID, source, execution status, arguments, and result predicates are applied. Plan-and-execute tool steps use `source_in: [plan]`; existing model-origin calls continue to use `llm`. Use `tool_not_called` rather than `count: 0` for absence.

Denied, unavailable, approval-rejected, and approval-timeout calls should usually assert `executed: false` so the eval proves that the wrapped tool implementation did not run.

```yaml
assert:
  tool_called:
    id: command
    executed: false
    success: false
```

---

## Redaction and output safety

Default output is redacted:

```text
input.value = [redacted]
response.value = [redacted]
string assertion actual/expected = [redacted]
raw TurnEvidence is omitted from JSON and JSONL
approval arguments, messages, reasons, and errors remain in-memory-only
response metadata is omitted from JSON and JSONL
```

Set `settings.redact_outputs: false` only for trusted local debugging.

Machine-readable output includes `schema_version: 1`.

---

## Parallel execution

Use CLI parallelism for independent scenario-isolated suites:

```sh
cargo run -p ai-agents-cli -- eval \
  --agent examples/yaml/basic/simple_chat.yaml \
  --scenarios examples/eval/mocked/basic/simple_chat_multiturn_mocked.yaml \
  --output target/eval/parallel \
  --parallel 4
```

Rules:

- requires `isolation: scenario`
- rejects process `env` overlays
- disabled by `--fail-fast`
- preserves output order after concurrent execution

---

## Examples

| Example | What it demonstrates |
|---------|----------------------|
| `examples/eval/mocked/basic/simple_chat_mocked.yaml` | Small no-key mock smoke test. |
| `examples/eval/mocked/basic/simple_chat_multiturn_mocked.yaml` | Ordered mock responses across multiple turns. |
| `examples/eval/mocked/basic/simple_chat_streaming_mocked.yaml` | Streaming turn collection. |
| `examples/eval/mocked/basic/simple_chat_observability_mocked.yaml` | Observability assertions without API keys. |
| `examples/eval/mocked/hitl/hitl_basic_mocked.yaml` | Ordered approve, reject, and timeout behavior across turns. |
| `examples/eval/mocked/hitl/hitl_conditions_mocked.yaml` | Condition bypass plus approve, reject, modify, and timeout behavior. |
| `examples/eval/mocked/hitl/hitl_multilingual_mocked.yaml` | Localized English, Korean, and Japanese approval messages and outcomes. |
| `examples/eval/mocked/runtime-optimization/pre_response_transition_mocked.yaml` | Pre-response guard routing without API keys. |
| `examples/eval/mocked/runtime-optimization/speculative_parallel_transition_mocked.yaml` | Parallel transition winner with stale draft discard. |
| `examples/eval/mocked/runtime-optimization/speculative_parallel_transition_miss_mocked.yaml` | Parallel transition miss with main draft commit. |
| `examples/eval/mocked/runtime-optimization/speculative_losing_tool_draft_mocked.yaml` | Losing draft tool call stays inert. |
| `examples/eval/mocked/runtime-optimization/speculative_skill_routing_mocked.yaml` | Skill selection branch commit before skill execution. |
| `examples/eval/mocked/runtime-optimization/speculative_skill_no_match_mocked.yaml` | Skill no-match branch with main draft commit. |
| `examples/eval/mocked/runtime-optimization/speculative_reasoning_auto_mocked.yaml` | Auto reasoning none decision with plain draft commit. |
| `examples/eval/mocked/runtime-optimization/speculative_reasoning_cot_mocked.yaml` | Auto reasoning deeper decision with plain draft discard. |
| `examples/eval/mocked/runtime-optimization/speculative_reasoning_react_mocked.yaml` | Auto reasoning ReAct decision with plain draft discard. |
| `examples/eval/mocked/runtime-optimization/speculative_reasoning_plan_mocked.yaml` | Auto reasoning plan-and-execute decision with committed plan path. |
| `examples/eval/mocked/runtime-optimization/speculative_reasoning_judge_failure_mocked.yaml` | Auto reasoning judge failure with plain draft fallback and failed branch status. |
| `examples/eval/mocked/runtime-optimization/buffered_streaming_mocked.yaml` | Buffered streaming route winner. |
| `examples/eval/mocked/runtime-optimization/buffered_streaming_main_win_mocked.yaml` | Buffered streaming main draft winner. |
| `examples/eval/live/quality/basic/simple_chat_semantic_judge_live.yaml` | Live provider response and live judge. |
| `examples/eval/live/examples/tools/code_search_live.yaml` | Live read-only tool-use smoke test for a runnable YAML example. |
| `examples/eval/live/README.md` | Registry for live example eval suites, risk tags, and run commands. |

See [Examples](@/examples/_index.md) for the full examples catalog.

---

## Next steps

- [CLI Guide](@/docs/cli.md) for all eval flags and exit codes.
- [YAML Reference](@/docs/yaml-reference.md#evaluation-suites) for the complete suite schema.
- [Rust API](@/docs/rust-api.md#evaluation-runner) for embedding eval runs in Rust.

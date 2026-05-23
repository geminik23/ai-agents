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
  --scenarios examples/eval/basic_chat.yaml \
  --output target/eval/basic_chat
```

Expected output summary:

```text
Eval complete: 1/1 passed, 0 failed, 0 skipped.
```

The runner writes:

```text
target/eval/basic_chat/
  summary.md
  summary.json
  per_scenario.jsonl
  failures.md
```

Use `--junit` to also write `junit.xml`.

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
  --scenarios examples/eval/basic_chat.yaml \
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

This is best for regression gates around call count, token use, and cost, not for judging response quality.

---

## Suite shape

```yaml
name: Basic Chat Eval
agent: ../yaml/basic/simple_chat.yaml
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
| `fixtures` | Mock/replay/record/real LLM modes, mock tools, context, and mock server. |
| `scenarios` | Test cases with direct turns or advanced steps. |
| `assert` | Assertions evaluated after a turn. |

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

---

## LLM modes

`fixtures.llm.mode` controls how the eval runner supplies LLM providers.

| Mode | Use when | Behavior |
|------|----------|----------|
| `mock` | Default CI and deterministic tests | Uses `fixtures.llm.responses` in order. When responses run out, the last response repeats. |
| `replay` | Stable regression from prior real traffic | Loads a JSONL cassette and matches records by alias, model, and request hash, with ordered fallback. |
| `record` | Creating or updating a cassette | Calls the real provider and appends responses to a cassette JSONL file. |
| `real` | Live provider smoke tests | Uses the provider configured by the agent YAML. Requires credentials and may incur cost. |

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
  --scenarios examples/eval/real_llm_semantic_judge.yaml \
  --output target/eval/record_live \
  --record target/eval/cassettes/live.jsonl
```

What happens:

```text
real provider call
  -> response returned to eval
  -> cassette record appended
```

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
  --scenarios examples/eval/real_llm_semantic_judge.yaml \
  --output target/eval/replay_live \
  --replay target/eval/cassettes/live.jsonl
```

Replay flow:

```text
runtime LLM request
  -> compute request hash from messages and config
  -> find cassette record with same alias, model, and hash
  -> return recorded response
  -> if no hash match, fall back to ordered cassette response
```

If replay unexpectedly uses the wrong response, check:

- the agent prompt did not change
- the model name did not change
- the LLM alias did not change
- the suite input did not change
- the cassette file path is correct

### Real mode

Use `real` only for live smoke tests.

```yaml
fixtures:
  llm:
    mode: real
```

Or force real mode from the CLI:

```sh
cargo run -p ai-agents-cli -- eval \
  --agent examples/yaml/basic/simple_chat.yaml \
  --scenarios examples/eval/real_llm_semantic_judge.yaml \
  --output target/eval/real_llm_semantic_judge \
  --real-llm
```

Real mode needs provider credentials such as `OPENAI_API_KEY`, network access, and acceptance of provider cost and nondeterminism.

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

Use judge assertions sparingly in default CI. Prefer deterministic checks for state, context, tools, metadata, facts, relationships, orchestration, and observability.

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

Common deterministic assertions:

| Assertion | Checks |
|-----------|--------|
| `state` / `state_in` / `state_not` | Current state after the turn. |
| `state_history_contains` | Transition history. |
| `metadata_contains` | Top-level response metadata. |
| `metadata_path` / `context_path` | Dot-path assertions. |
| `tool_called` | Tool ID, success, source, arguments, output. |
| `facts_include` | Actor facts by actor/category and optional semantic judge. |
| `relationship` | Relationship dimensions and counts. |
| `orchestration` | Pattern, final agent, included agents, stage count. |
| `observability` | LLM/tool/token/cost/purpose/status counts. |

Tool assertion example:

```yaml
assert:
  tool_called:
    id: lookup_order
    count_gte: 1
    success: true
    result_path:
      path: status
      eq: cancellable
```

---

## Redaction and output safety

Default output is redacted:

```text
input.value = [redacted]
response.value = [redacted]
string assertion actual/expected = [redacted]
raw TurnEvidence is omitted from JSON and JSONL
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
  --scenarios examples/eval/multiturn_mocked.yaml \
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
| `examples/eval/basic_chat.yaml` | Small no-key mock smoke test. |
| `examples/eval/multiturn_mocked.yaml` | Ordered mock responses across multiple turns. |
| `examples/eval/streaming_mocked.yaml` | Streaming turn collection. |
| `examples/eval/observability_mocked.yaml` | Observability assertions without API keys. |
| `examples/eval/real_llm_semantic_judge.yaml` | Live provider response and live judge. |

See [Examples](@/examples/_index.md) for the full examples catalog.

---

## Next steps

- [CLI Guide](@/docs/cli.md) for all eval flags and exit codes.
- [YAML Reference](@/docs/yaml-reference.md#evaluation-suites) for the complete suite schema.
- [Rust API](@/docs/rust-api.md#evaluation-runner) for embedding eval runs in Rust.

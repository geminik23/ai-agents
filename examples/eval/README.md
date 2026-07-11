# Evaluation Suites

Evaluation suites are organized by run mode first, then by example category. This keeps no-key CI suites separate from suites that can call live providers or cost money.

- `mocked/` contains deterministic no-key regression suites for local checks and CI.
- `live/examples/` contains intentional real-provider smoke checks for runnable examples.
- `live/quality/` contains semantic or judge-based live checks.
- `replay/` contains cassette guidance and optional local replay artifacts.

Category folders mirror `examples/yaml/` where practical. The runner discovers matching suite files, so this document does not maintain a separate category tree.

## Safe default

Use mocked suites for default local and CI checks:

```sh
# Run all no-key mocked suites
sh examples/eval/mocked/run_mocked_evals.sh

# Run one category
sh examples/eval/mocked/run_mocked_evals.sh --category state-machine

# List all mocked suites
sh examples/eval/mocked/run_mocked_evals.sh --list
```

Or run a single suite directly:

```sh
cargo run -p ai-agents-cli -- eval \
  --scenarios examples/eval/mocked/basic/simple_chat_mocked.yaml \
  --output target/eval/mocked/basic/simple_chat_mocked
```

Most suites declare their own `agent:` path, so `--agent` is optional unless you intentionally want to override it.

## Attempt-local fixtures

Every scenario attempt receives an opaque temporary workspace. Generated values are available in runtime context as `eval.workspace` and, when enabled, `mock_server.base_url`.

Mock LLM response fixtures can reference only these generated tokens:

```yaml
fixtures:
  llm:
    mode: mock
    responses:
      - '{"tool":"http","arguments":{"method":"GET","url":"{{ mock_server.base_url }}/status"}}'
      - '{"tool":"file_write","arguments":{"path":"{{ eval.workspace }}/note.md","content":"review","dry_run":true}}'
```

Interpolation is JSON-safe and leaves unrelated template expressions unchanged. Referencing `mock_server.base_url` without enabling the mock server is a configuration error.

Parent storage and `spawner.shared_storage` file, SQLite, or Redis backends are isolated per attempt. When a source agent has an existing path policy, `fixtures.workspace_policy` can narrowly add the attempt workspace to named policies without disabling other restrictions:

```yaml
fixtures:
  workspace_policy:
    write_tools: [file_write]
```

Use `fixtures.web_fetch_transport` for exact-URL, no-socket responses through the real `web_fetch` implementation. This does not bypass URL, domain, redirect, private-network, or response-size policy.

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

## Sequenced LLM outcomes and request evidence

Use `outcomes_by_alias` when one alias must fail and then recover:

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

The final outcome repeats after exhaustion. An error without `status` retains the legacy status-less API error behavior.

Use `llm_request` to verify the actual messages sent to an LLM rather than trusting a canned response:

```yaml
assert:
  llm_request:
    count_gte: 1
    system_contains: "Senior Support Representative"
    user_contains: "account"
    same_request: true
```

LLM request contents remain in memory for assertions and are not serialized into eval reports or assertion summaries.

## Deterministic HITL suites

The `mocked/hitl/` suites exercise approval behavior without API keys, real HTTP, or human input:

| Suite | Coverage |
|-------|----------|
| `examples/eval/mocked/hitl/hitl_basic_mocked.yaml` | Ordered approve, reject, and timeout outcomes across turns. |
| `examples/eval/mocked/hitl/hitl_conditions_mocked.yaml` | No-approval GET path plus approve, reject, modified arguments, and timeout for state-changing HTTP conditions. |
| `examples/eval/mocked/hitl/hitl_multilingual_mocked.yaml` | Exact localized English, Korean, and Japanese approval messages and outcomes. |

Run the category with:

```sh
sh examples/eval/mocked/run_mocked_evals.sh --category hitl
```

`fixtures.approvals` installs a deterministic approval handler. Rules are checked in declaration order; the first matching rule whose optional 1-based `occurrence` matches wins. Each rule has its own counter, incremented when evaluation reaches that rule and its trigger matches; later rules are not visited after a winner is selected. Rules can match `tool` triggers with optional exact `args`, `condition` triggers with optional exact `matched`, `state_transition` triggers with optional `from` and required `to`, or `disambiguation_escalation` triggers with an optional exact `reason`. If no rule matches, `default` is returned. An omitted default is `unavailable`, and `unavailable` fails closed as a rejection with reason `Approval fixture unavailable`.

This is the ordered syntax used by `hitl_basic_mocked.yaml`:

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

Supported outcomes are `approve`, `reject` with optional `reason`, `modify` with top-level argument `changes`, `timeout`, and `unavailable`. For example, `hitl_conditions_mocked.yaml` replaces selected arguments before execution:

```yaml
- trigger:
    type: condition
    name: state_changing_http
    matched: "method in [POST, PUT, DELETE, PATCH]"
  occurrence: 3
  outcome: modify
  changes:
    url: "https://api.example.test/v1/safe-items/42"
    body: '{"name":"safe-item"}'
```

`preferred_language` and `supported_languages` are exposed to the HITL message resolver. This allows deterministic checks of the localized message shown to the handler, not just the decision.

Each scenario attempt receives a fresh handler and fresh per-rule occurrence counters. Counters continue across turns and agent resets in that attempt, while retries and parallel scenario attempts are isolated from one another.

Use `approval_requested` to match retained approval evidence and `approval_not_requested` to prove that an approval path was bypassed:

```yaml
assert:
  all:
    - approval_requested:
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
    - tool_called:
        id: http
        executed: true
```

```yaml
approval_not_requested:
  trigger:
    type: condition
    name: state_changing_http
```

Approval assertions accept boolean form or object filters. Object form supports `count`, `count_gte`, `count_lte`, normalized trigger fields, `raw_decision`, `effective_decision`, exact/substring message checks, rejection reason checks, resolution error checks, and path assertions over `original_args`, `modified_args`, and `effective_args`. All filters must match one record. Assertion state triggers use `type: state`, unlike fixture rules, which use `type: state_transition`.

Raw decisions describe the handler response: `approved`, `rejected`, `modified`, or `timeout`. Effective decisions describe runtime resolution: `approved`, `rejected`, `modified`, or `error`. The example agents resolve timeout to rejection, so the suites use `raw_decision: timeout` with `effective_decision: rejected`. Modified arguments are the original object patched by `changes`; effective arguments are the values eligible for execution. Rejected, timed-out, or errored approvals have no executable effective arguments.

Expected runtime failures belong on the individual turn and accept one substring or a list of alternatives:

```yaml
- input: Trigger the approval error path.
  expect_error: [timeout, unavailable]
  assert:
    approval_requested:
      raw_decision: timeout
      effective_decision: error
      error_contains: "timed out"
```

A matching `expect_error` becomes a passing assertion. Missing or non-matching errors fail it. When the runtime returns an error from a turn, the runner still retains that turn and its approval/tool/state/latency evidence before stopping; setup failures and outer scenario-attempt timeouts cannot retain a turn that was never produced.

Approval evidence is used in memory for assertions. Raw `TurnEvidence` is omitted from JSON/JSONL, and approval arguments, localized messages, rejection reasons, and errors are non-serializing. Default `settings.redact_outputs: true` also redacts assertion actual/expected values. Disabling output redaction is only for trusted local debugging and still does not serialize raw approval evidence.

Mocked HITL suites must also mock the LLM and approved tools. Real HTTP and real human input remain excluded.

## Live provider suites

Live suites are opt-in release smoke checks. They require provider credentials, network access, and acceptance of provider cost and nondeterminism.

```sh
cargo run -p ai-agents-cli -- eval \
  --scenarios examples/eval/live/examples/tools/code_search_live.yaml \
  --output target/eval/live/examples/tools/code_search \
  --real-llm
```

For convenience, `examples/eval/live/run_live_example_evals.sh` can list, parse-check, or run live example suites by category:

```sh
sh examples/eval/live/run_live_example_evals.sh --dry-config-check --category tools
```

Do not glob all of `examples/eval/**/*.yaml` in default CI because that includes live suites. Use one of these narrower globs instead:

```text
examples/eval/mocked/**/*.yaml
examples/eval/live/examples/**/*.yaml
examples/eval/live/quality/**/*.yaml
```

## Naming convention

Mocked suites:

```text
examples/eval/mocked/<category>/<example_or_contract>_mocked.yaml
```

Live example suites:

```text
examples/yaml/<category>/<example>.yaml
        -> examples/eval/live/examples/<category>/<example>_live.yaml
```

Live quality suites:

```text
examples/eval/live/quality/<category>/<example>_<quality>_live.yaml
```

Use the same category folder names as `examples/yaml/`, such as `state-machine` and `runtime-optimization`.

## Safety boundaries

- Mock exact tool-call sequences, denial paths, unavailable providers, deterministic HITL approvals, blocked commands, and blocked network behavior.
- Keep live example suites focused on one primary behavior per scenario.
- Keep real shell commands, public network calls, MCP child processes, real HTTP approval effects, human input, and actual file mutation out of automatic live suites until their safety boundary is explicit.
- Keep semantic judge suites under `live/quality/`, separate from example smoke checks.

See `examples/eval/live/README.md` for the live suite registry and risk tags.

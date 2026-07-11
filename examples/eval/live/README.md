# Live Eval Suites

This directory contains intentional live-provider evaluation suites. They use the normal `ai-agents-cli eval` path with real model behavior, structural assertions, deterministic response checks, and fixture-backed external dependencies where needed.

Live suites are release smoke tests, not the default no-key CI path. Prefer `examples/eval/mocked/**/*.yaml` for exact tool-call regression coverage and fast local checks.

## Layout

```text
examples/eval/live/
├── run_live_example_evals.sh  # convenience helper for live example suites
├── examples/                  # runnable YAML example smoke checks
│   ├── basic/
│   ├── state-machine/
│   └── tools/
└── quality/                   # semantic or judge-based live checks
    └── basic/
```

## Requirements

- Provider credentials for the model configured by the agent YAML, such as `OPENAI_API_KEY`.
- Network access for provider calls.
- Acceptance that live runs may incur provider cost and can be nondeterministic.
- A clean understanding of each suite's risk tags before running it.

## Run one suite

```sh
cargo run -p ai-agents-cli -- eval \
  --scenarios examples/eval/live/examples/tools/code_search_live.yaml \
  --output target/eval/live/examples/tools/code_search \
  --real-llm
```

The suite already declares its `agent` path. You may still pass `--agent` to override it intentionally.

## Run by category

`run_live_example_evals.sh` is a convenience helper for this live example suite set. It is not framework runtime code.

List all live example suites:

```sh
sh examples/eval/live/run_live_example_evals.sh --list
```

Parse-check only one category without provider calls:

```sh
sh examples/eval/live/run_live_example_evals.sh --dry-config-check --category tools
```

Run one category intentionally with a real provider:

```sh
sh examples/eval/live/run_live_example_evals.sh --yes-live --category state-machine
```

Combine category and filename filtering:

```sh
sh examples/eval/live/run_live_example_evals.sh --yes-live --category tools --filter code_search
```

Available categories are the folders under `examples/eval/live/examples/`, such as `basic`, `state-machine`, and `tools`.

## Run selected tags

Use `--tag-mode all` when every tag must match:

```sh
cargo run -p ai-agents-cli -- eval \
  --scenarios examples/eval/live/examples/tools/code_search_live.yaml \
  --output target/eval/live/examples/tools/code_search \
  --real-llm \
  --tags live \
  --tags read-only \
  --tag-mode all
```

## Record and replay

Only one of `--record`, `--replay`, or `--real-llm` can be used at a time.

```sh
cargo run -p ai-agents-cli -- eval \
  --scenarios examples/eval/live/examples/tools/code_search_live.yaml \
  --output target/eval/live/examples/tools/code_search_record \
  --record target/eval/cassettes/tools_code_search.jsonl

cargo run -p ai-agents-cli -- eval \
  --scenarios examples/eval/live/examples/tools/code_search_live.yaml \
  --output target/eval/live/examples/tools/code_search_replay \
  --replay target/eval/cassettes/tools_code_search.jsonl
```

Record/replay is useful for local iteration. A real provider run is still the pre-release smoke check.

## Naming convention

```text
examples/yaml/<category>/<example>.yaml
        -> examples/eval/live/examples/<category>/<example>_live.yaml
```

Use the same category folder name as `examples/yaml/`, including names such as `state-machine` and `runtime-optimization`.

## Status vocabulary

| Status | Meaning |
|--------|---------|
| `live-auto` | Safe to run intentionally with `--real-llm` and no extra local service. |
| `fixture-live` | Uses a real LLM while external systems are deterministic fixtures or dry-run paths. |
| `mocked-only` | Should stay deterministic and no-key for now. |
| `manual-only` | Requires a human, local service, secrets, custom Rust, or risky side effects. |
| `support-file` | Fixture or helper file, not a primary runnable suite. |
| `deferred` | Deferred until safety, fixtures, or workflow requirements are explicit. |

## Risk tags

Use these tags consistently in live suites and registry rows:

```text
live examples yaml tools read-only host-backed utility grant scoping denial mutation dry-run command network hitl mcp manual fixture mocked-only
```

## Coverage registry

| Example | Primary tools or behavior | Status | Suite or follow-up | Reason |
|---------|---------------------------|--------|--------------------|--------|
| `examples/yaml/basic/simple_tools.yaml` | minimal built-in tools | `live-auto` | `examples/eval/live/examples/basic/simple_tools_live.yaml` | Verifies a minimal safe tool grant. |
| `examples/yaml/state-machine/state_with_tools.yaml` | state-level tool narrowing | `live-auto` | `examples/eval/live/examples/state-machine/state_with_tools_live.yaml` | Verifies state transition evidence and scoped tool execution. |
| `examples/yaml/tools/code_search.yaml` | `glob`, `grep`, `file_list`, `file_read`, `file_info` | `live-auto` | `examples/eval/live/examples/tools/code_search_live.yaml` | Read-only repository search with narrow path policy; equivalent safe search paths are accepted. |
| `examples/yaml/tools/workspace_research.yaml` | read-only workspace search plus `todo` | `live-auto` | `examples/eval/live/examples/tools/workspace_research_live.yaml` | Read-only inspection split into focused discovery and file-read scenarios. |
| `examples/yaml/tools/repo_review.yaml` | `git_status`, `git_diff`, `file_read`, `todo` | `live-auto` | `examples/eval/live/examples/tools/repo_review_live.yaml` | Read-only VCS inspection; accepts status or diff evidence and avoids exact diff contents. |
| `examples/yaml/tools/basic_tools.yaml` | `calculator`, `datetime` | `live-auto` | `examples/eval/live/examples/tools/basic_tools_live.yaml` | Safe local utility tools split into focused scenarios. |
| `examples/yaml/tools/text_and_json.yaml` | `text`, `json` | `live-auto` | `examples/eval/live/examples/tools/text_and_json_live.yaml` | Safe local utility tools split into focused scenarios. |
| `examples/yaml/tools/math_and_random.yaml` | `math`, `random` | `live-auto` | `examples/eval/live/examples/tools/math_and_random_live.yaml` | Safe local utility tools split into focused scenarios with flexible random assertions. |
| `examples/yaml/tools/todo_workflow.yaml` | `todo`, `ask_user` | `live-auto` | `examples/eval/live/examples/tools/todo_workflow_live.yaml` | Uses a concrete todo request and asserts `ask_user` is not needed for the prompt. |
| `examples/yaml/tools/interactive_choice.yaml` | `ask_user`, `todo` | `fixture-live` | `examples/eval/live/examples/tools/interactive_choice_live.yaml` | Uses the default ask-user fallback, so no real human is required. |
| `examples/yaml/tools/diagnostics_review.yaml` | `diagnostics`, `file_read` | `fixture-live` | `examples/eval/live/examples/tools/diagnostics_review_live.yaml` | Uses deterministic diagnostics fixture data with a concrete bounded diagnostics request. |
| `examples/yaml/tools/sleep_wait.yaml` | `sleep`, `todo` | `live-auto` | `examples/eval/live/examples/tools/sleep_wait_live.yaml` | Uses a short bounded wait. |
| `examples/yaml/tools/multi_tool_agent.yaml` | multiple built-ins and parallel calls | `fixture-live` | `examples/eval/live/examples/tools/multi_tool_agent_live.yaml` | Uses focused safe local tool scenarios; risky granted tools are replaced by failing eval mocks and asserted not called. |
| `examples/yaml/tools/file_write_sandbox.yaml` | `file_write`, `file_read`, `todo` | `fixture-live` | `examples/eval/live/examples/tools/file_write_sandbox_live.yaml` | Dry-run mutation only; no bytes are written. |
| `examples/yaml/tools/file_edit_review.yaml` | `file_read`, `file_edit`, `todo` | `fixture-live` | `examples/eval/live/examples/tools/file_edit_review_live.yaml` | Dry-run mutation against committed fixture files. |
| `examples/yaml/tools/patch_review.yaml` | `file_read`, `patch`, `todo` | `fixture-live` | `examples/eval/live/examples/tools/patch_review_live.yaml` | Dry-run patch against committed fixture files. |
| `examples/yaml/tools/copy_review.yaml` | `copy_path`, `file_read`, `todo` | `fixture-live` | `examples/eval/live/examples/tools/copy_review_live.yaml` | Dry-run mutation against committed fixture files. |
| `examples/yaml/tools/move_review.yaml` | `move_path`, `file_read`, `todo` | `fixture-live` | `examples/eval/live/examples/tools/move_review_live.yaml` | Dry-run mutation against committed fixture files. |
| `examples/yaml/tools/delete_review.yaml` | `delete_path`, `file_read`, `todo` | `fixture-live` | `examples/eval/live/examples/tools/delete_review_live.yaml` | Dry-run mutation against committed fixture files. |
| `examples/yaml/tools/web_search_research.yaml` | `web_search`, `web_fetch` | `fixture-live` | `examples/eval/live/examples/tools/web_search_research_live.yaml` | Uses a real LLM with an explicit unavailable web-search fixture; model should suggest web_fetch fallback. |
| `examples/yaml/tools/command_validation.yaml` | `command`, `todo` | `deferred` | mocked command suites | Real command execution from live model choices needs a separate safety rollout; mocked suites already cover allowlist and denial paths. |
| `examples/yaml/tools/web_fetch_research.yaml` | `web_fetch` | `deferred` | mocked web-fetch policy suite | Public-network coverage needs explicit live-network or mock-server policy. |
| `examples/yaml/tools/http_tool.yaml` | `http`, `json` | `deferred` | mock-server coverage needed | Raw HTTP coverage needs public-network or mock-server policy. |
| `examples/yaml/tools/mcp_agent.yaml` | MCP filesystem views plus built-ins | `manual-only` | manual workflow | Requires local `npx` process startup and a filesystem sandbox. |
| `examples/yaml/tools/file_and_template.yaml` | legacy/general `file`, `template` | `deferred` | sandbox coverage needed | Legacy file behavior needs sandbox clarification before live automation. |
| `examples/fixtures/tool_examples/edit_target.txt` | dry-run edit fixture | `support-file` | fixture | Used by `examples/eval/live/examples/tools/file_edit_review_live.yaml`. |
| `examples/fixtures/tool_examples/patch_target.txt` | dry-run patch fixture | `support-file` | fixture | Used by `examples/eval/live/examples/tools/patch_review_live.yaml`. |

## Quality suites

Quality suites use live providers and semantic or judge-style assertions. Keep them separate from runnable example smoke checks.

```text
examples/eval/live/quality/basic/simple_chat_semantic_judge_live.yaml
```

## Mocked suites to keep

These no-key suites cover exact sequences, denial paths, unavailable providers, and approval behavior:

```text
examples/eval/mocked/tools/code_search_mocked.yaml
examples/eval/mocked/tools/diagnostics_mocked.yaml
examples/eval/mocked/tools/command_validation_mocked.yaml
examples/eval/mocked/tools/command_blocked_mocked.yaml
examples/eval/mocked/tools/file_write_dry_run_mocked.yaml
examples/eval/mocked/tools/file_edit_review_mocked.yaml
examples/eval/mocked/tools/file_edit_denied_mocked.yaml
examples/eval/mocked/tools/file_edit_approval_rejected_mocked.yaml
examples/eval/mocked/tools/patch_review_mocked.yaml
examples/eval/mocked/tools/copy_path_dry_run_mocked.yaml
examples/eval/mocked/tools/move_path_dry_run_mocked.yaml
examples/eval/mocked/tools/delete_path_dry_run_mocked.yaml
examples/eval/mocked/tools/web_fetch_policy_mocked.yaml
examples/eval/mocked/tools/web_search_mocked.yaml
examples/eval/mocked/tools/web_search_unavailable_mocked.yaml
examples/eval/mocked/tools/sleep_wait_mocked.yaml
examples/eval/mocked/tools/ask_user_fallback_mocked.yaml
examples/eval/mocked/basic/no_tools_explicit_empty_mocked.yaml
```

## Assertion policy

- Use `response_not_empty` as the baseline.
- Prefer one primary live behavior per scenario; split multi-tool coverage across scenarios instead of requiring a long exact sequence in one turn.
- Prefer structural evidence for runtime behavior: `tool_called`, `tool_not_called`, `state`, `state_history_contains`, `context_path`, `metadata_path`, and `observability`.
- Add deterministic response checks for minimum useful output, such as stable arithmetic results, requested symbols, fixture details, or dry-run wording.
- Use `result_path` for deterministic tool output fields such as `result`, `length`, `value`, `uuid`, `dry_run`, `bytes_written`, `available`, and `count`.
- Use `any` when several safe tools can satisfy the same read-only behavior.
- Use `response_contains_any` for bounded wording choices and `response_not_contains` for complete overclaim phrases.
- Keep judge gates out of committed smoke suites when deterministic structural and response assertions can express the requirement.
- Assert exact canonical tool IDs, `executed`, `success`, known state names, and dry-run flags.
- Avoid exact full-response text, tiny substrings such as `-`, and exact random values.
- Do not run real commands, public HTTP, MCP child processes, or actual file writes from these suites.

# Live Example Eval Suites

This directory contains live evaluation suites for runnable YAML examples. They use the normal `ai-agents-cli eval` path with real model behavior, structural assertions, deterministic response checks, and fixture-backed external dependencies where needed.

Live suites are release smoke tests, not the default no-key CI path. Prefer the mocked suites in `examples/eval/` for exact tool-call regression coverage and fast local checks.

## Requirements

- Provider credentials for the model configured by the agent YAML, such as `OPENAI_API_KEY`.
- Network access for provider calls.
- Acceptance that live runs may incur provider cost and can be nondeterministic.
- A clean understanding of each suite's risk tags before running it.

## Run one suite

```sh
cargo run -p ai-agents-cli -- eval \
  --scenarios examples/eval/live/examples/tools_code_search_live.yaml \
  --output target/eval/live/examples/tools_code_search \
  --real-llm
```

The suite already declares its `agent` path. You may still pass `--agent` to override it intentionally.

## Run selected tags

Use `--tag-mode all` when every tag must match:

```sh
cargo run -p ai-agents-cli -- eval \
  --scenarios examples/eval/live/examples/tools_code_search_live.yaml \
  --output target/eval/live/examples/tools_code_search \
  --real-llm \
  --tags live \
  --tags read-only \
  --tag-mode all
```

## Record and replay

Only one of `--record`, `--replay`, or `--real-llm` can be used at a time.

```sh
cargo run -p ai-agents-cli -- eval \
  --scenarios examples/eval/live/examples/tools_code_search_live.yaml \
  --output target/eval/live/examples/tools_code_search_record \
  --record target/eval/cassettes/tools_code_search.jsonl

cargo run -p ai-agents-cli -- eval \
  --scenarios examples/eval/live/examples/tools_code_search_live.yaml \
  --output target/eval/live/examples/tools_code_search_replay \
  --replay target/eval/cassettes/tools_code_search.jsonl
```

Record/replay is useful for local iteration. A real provider run is still the pre-release smoke check.

## Naming convention

```text
examples/yaml/<category>/<example>.yaml
        -> examples/eval/live/examples/<category>_<example>_live.yaml
```

Use underscores in eval filenames even when the YAML category uses hyphens.

## Status vocabulary

| Status | Meaning |
|--------|---------|
| `live-auto` | Safe to run intentionally with `--real-llm` and no extra local service. |
| `fixture-live` | Uses a real LLM while external systems are deterministic fixtures or dry-run paths. |
| `mocked-only` | Should stay deterministic and no-key for now. |
| `manual-only` | Requires a human, local service, secrets, custom Rust, or risky side effects. |
| `support-file` | Fixture or helper file, not a primary runnable suite. |
| `deferred-55` | Deferred to `temp/side_plan/55_auto_eval_remaining_examples.md`. |

## Risk tags

Use these tags consistently in live suites and registry rows:

```text
live examples yaml tools read-only host-backed utility grant scoping denial mutation dry-run command network hitl mcp manual fixture mocked-only
```

## Coverage registry

| Example | Primary tools or behavior | Status | Suite or follow-up | Reason |
|---------|---------------------------|--------|--------------------|--------|
| `examples/yaml/tools/code_search.yaml` | `glob`, `grep`, `file_list`, `file_read`, `file_info` | `live-auto` | `tools_code_search_live.yaml` | Read-only repository search with narrow path policy; equivalent safe search paths are accepted. |
| `examples/yaml/tools/workspace_research.yaml` | read-only workspace search plus `todo` | `live-auto` | `tools_workspace_research_live.yaml` | Read-only inspection split into focused discovery and file-read scenarios. |
| `examples/yaml/tools/repo_review.yaml` | `git_status`, `git_diff`, `file_read`, `todo` | `live-auto` | `tools_repo_review_live.yaml` | Read-only VCS inspection; accepts status or diff evidence and avoids exact diff contents. |
| `examples/yaml/tools/basic_tools.yaml` | `calculator`, `datetime` | `live-auto` | `tools_basic_tools_live.yaml` | Safe local utility tools split into focused scenarios. |
| `examples/yaml/tools/text_and_json.yaml` | `text`, `json` | `live-auto` | `tools_text_and_json_live.yaml` | Safe local utility tools split into focused scenarios. |
| `examples/yaml/tools/math_and_random.yaml` | `math`, `random` | `live-auto` | `tools_math_and_random_live.yaml` | Safe local utility tools split into focused scenarios with flexible random assertions. |
| `examples/yaml/tools/todo_workflow.yaml` | `todo`, `ask_user` | `live-auto` | `tools_todo_workflow_live.yaml` | Uses a concrete todo request and asserts `ask_user` is not needed for the prompt. |
| `examples/yaml/tools/interactive_choice.yaml` | `ask_user`, `todo` | `fixture-live` | `tools_interactive_choice_live.yaml` | Uses the default ask-user fallback, so no real human is required. |
| `examples/yaml/tools/diagnostics_review.yaml` | `diagnostics`, `file_read` | `fixture-live` | `tools_diagnostics_review_live.yaml` | Uses deterministic diagnostics fixture data with a concrete bounded diagnostics request. |
| `examples/yaml/tools/sleep_wait.yaml` | `sleep`, `todo` | `live-auto` | `tools_sleep_wait_live.yaml` | Uses a short bounded wait. |
| `examples/yaml/basic/simple_tools.yaml` | minimal built-in tools | `live-auto` | `basic_simple_tools_live.yaml` | Verifies a minimal safe tool grant. |
| `examples/yaml/state-machine/state_with_tools.yaml` | state-level tool narrowing | `live-auto` | `state_machine_state_with_tools_live.yaml` | Verifies state transition evidence and scoped tool execution. |
| `examples/yaml/tools/multi_tool_agent.yaml` | multiple built-ins and parallel calls | `fixture-live` | `tools_multi_tool_agent_live.yaml` | Uses focused safe local tool scenarios; risky granted tools are replaced by failing eval mocks and asserted not called. |
| `examples/yaml/tools/file_write_sandbox.yaml` | `file_write`, `file_read`, `todo` | `fixture-live` | `tools_file_write_sandbox_live.yaml` | Dry-run mutation only; no bytes are written. |
| `examples/yaml/tools/file_edit_review.yaml` | `file_read`, `file_edit`, `todo` | `fixture-live` | `tools_file_edit_review_live.yaml` | Dry-run mutation against committed fixture files. |
| `examples/yaml/tools/patch_review.yaml` | `file_read`, `patch`, `todo` | `fixture-live` | `tools_patch_review_live.yaml` | Dry-run patch against committed fixture files. |
| `examples/yaml/tools/command_validation.yaml` | `command`, `todo` | `deferred-55` | `temp/side_plan/55_auto_eval_remaining_examples.md` | Real command execution from live model choices needs a separate safety rollout; mocked suites already cover allowlist and denial paths. |
| `examples/yaml/tools/web_fetch_research.yaml` | `web_fetch` | `deferred-55` | `temp/side_plan/55_auto_eval_remaining_examples.md` | Public-network coverage needs explicit live-network or mock-server policy. |
| `examples/yaml/tools/http_tool.yaml` | `http`, `json` | `deferred-55` | `temp/side_plan/55_auto_eval_remaining_examples.md` | Raw HTTP coverage needs public-network or mock-server policy. |
| `examples/yaml/tools/mcp_agent.yaml` | MCP filesystem views plus built-ins | `manual-only` | `temp/side_plan/55_auto_eval_remaining_examples.md` | Requires local `npx` process startup and a filesystem sandbox. |
| `examples/yaml/tools/file_and_template.yaml` | legacy/general `file`, `template` | `deferred-55` | `temp/side_plan/55_auto_eval_remaining_examples.md` | Legacy file behavior needs sandbox clarification before live automation. |
| `examples/yaml/tools/copy_review.yaml` | `copy_path`, `file_read`, `todo` | `fixture-live` | `tools_copy_review_live.yaml` | Dry-run mutation against committed fixture files. |
| `examples/yaml/tools/move_review.yaml` | `move_path`, `file_read`, `todo` | `fixture-live` | `tools_move_review_live.yaml` | Dry-run mutation against committed fixture files. |
| `examples/yaml/tools/delete_review.yaml` | `delete_path`, `file_read`, `todo` | `fixture-live` | `tools_delete_review_live.yaml` | Dry-run mutation against committed fixture files. |
| `examples/yaml/tools/web_search_research.yaml` | `web_search`, `web_fetch` | `fixture-live` | `tools_web_search_research_live.yaml` | Uses real LLM with unavailable provider; model should suggest web_fetch fallback. |
| `examples/fixtures/tool_examples/edit_target.txt` | dry-run edit fixture | `support-file` | fixture | Used by `tools_file_edit_review_live.yaml`. |
| `examples/fixtures/tool_examples/patch_target.txt` | dry-run patch fixture | `support-file` | fixture | Used by `tools_patch_review_live.yaml`. |

## Mocked suites to keep

These no-key suites cover exact sequences, denial paths, unavailable providers, and approval behavior:

```text
examples/eval/code_search_mocked.yaml
examples/eval/diagnostics_mocked.yaml
examples/eval/command_validation_mocked.yaml
examples/eval/command_blocked_mocked.yaml
examples/eval/file_write_dry_run_mocked.yaml
examples/eval/file_edit_review_mocked.yaml
examples/eval/file_edit_denied_mocked.yaml
examples/eval/file_edit_approval_rejected_mocked.yaml
examples/eval/patch_review_mocked.yaml
examples/eval/no_tools_explicit_empty_mocked.yaml
examples/eval/web_fetch_policy_mocked.yaml
examples/eval/sleep_wait_mocked.yaml
examples/eval/ask_user_fallback_mocked.yaml
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

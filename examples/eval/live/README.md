# Live Eval Suites

This directory contains intentional live-provider evaluation suites. They use the normal `ai-agents-cli eval` path with real model behavior, structural assertions, deterministic response checks, and fixture-backed external dependencies where needed.

Live suites are release smoke tests, not the default no-key CI path. Prefer `examples/eval/mocked/**/*.yaml` for exact tool-call regression coverage and fast local checks.

## Layout

```text
examples/eval/live/
├── run_live_example_evals.sh
├── examples/<category>/*_live.yaml
└── quality/<category>/*_live.yaml
```

Category folders mirror `examples/yaml/`. Use `--list` for the current discovered suite set instead of maintaining a category tree here.

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

Available categories are discovered from the folders under `examples/eval/live/examples/`. Use `--list` to see the current suite set.

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

Use a small set of descriptive tags for filtering and reporting. Common tags include:

```text
live real-llm examples yaml basic context disambiguation error-recovery memory observability persona process reasoning relationships session skills state-machine tools read-only no-tools host-backed utility grant scoping routing guard nested lifecycle detection extraction normalization sanitization denial mutation dry-run command network hitl mcp manual fixture mocked-only
```

## Coverage registry

| Example | Primary tools or behavior | Status | Suite or follow-up | Reason |
|---------|---------------------------|--------|--------------------|--------|
| `examples/yaml/basic/simple_tools.yaml` | minimal built-in tools | `live-auto` | `examples/eval/live/examples/basic/simple_tools_live.yaml` | Verifies a minimal safe tool grant. |
| `examples/yaml/state-machine/state_with_tools.yaml` | state-level tool narrowing | `live-auto` | `examples/eval/live/examples/state-machine/state_with_tools_live.yaml` | Verifies state transition evidence and scoped tool execution. |
| `examples/yaml/state-machine/two_state_greeting.yaml` | minimal conversational state loop | `live-auto` | `examples/eval/live/examples/state-machine/two_state_greeting_live.yaml` | Verifies real-model routing into help and back to greeting. |
| `examples/yaml/state-machine/guard_transitions.yaml` | extraction-backed context guards | `live-auto` | `examples/eval/live/examples/state-machine/guard_transitions_live.yaml` | Verifies live extraction followed by deterministic guard routing. |
| `examples/yaml/state-machine/nested_states.yaml` | hierarchical state routing | `live-auto` | `examples/eval/live/examples/state-machine/nested_states_live.yaml` | Verifies a nested troubleshooting route and root-state escape. |
| `examples/yaml/state-machine/state_lifecycle.yaml` | enter, exit, and reentry actions | `live-auto` | `examples/eval/live/examples/state-machine/state_lifecycle_live.yaml` | Verifies exact first-entry `on_enter` and later `on_reenter` context through a bounded revision workflow. |
| `examples/yaml/state-machine/support_state_machine.yaml` | representative support routing | `live-auto` | `examples/eval/live/examples/state-machine/support_state_machine_live.yaml` | Verifies technical, order, and global escalation routes without external effects. |
| `examples/yaml/context/builtin_context.yaml` | built-in agent and session context | `live-auto` | `examples/eval/live/examples/context/builtin_context_live.yaml` | Verifies the model uses built-in identity, session, and time context. |
| `examples/yaml/context/runtime_context.yaml` | host-provided user context | `fixture-live` | `examples/eval/live/examples/context/runtime_context_live.yaml` | Uses safe fixture values to verify personalized real-model output. |
| `examples/yaml/context/template_context.yaml` | context-driven prompt templates | `fixture-live` | `examples/eval/live/examples/context/template_context_live.yaml` | Verifies a stable customer profile selects the intended template branch. |
| `examples/yaml/context/context_with_state.yaml` | context retained across state routing | `fixture-live` | `examples/eval/live/examples/context/context_with_state_live.yaml` | Verifies injected profile context remains available after transition. |
| `examples/yaml/context/env_context.yaml` | environment context | `fixture-live` | `examples/eval/live/examples/context/env_context_live.yaml` | Uses safe scenario-local environment values in a serial suite. |
| `examples/yaml/process/input_normalize.yaml` | input normalization | `live-auto` | `examples/eval/live/examples/process/input_normalize_live.yaml` | Verifies the real model receives and reflects normalized input. |
| `examples/yaml/process/detect_language.yaml` | language, sentiment, and intent detection | `live-auto` | `examples/eval/live/examples/process/detect_language_live.yaml` | Uses an unambiguous request to verify detected context and localized output. |
| `examples/yaml/process/extract_and_validate.yaml` | structured extraction and validation | `live-auto` | `examples/eval/live/examples/process/extract_and_validate_live.yaml` | Verifies stable extracted support fields and a context-aware response; exact validation-stage behavior remains mocked. |
| `examples/yaml/process/output_sanitize.yaml` | output PII non-disclosure and formatting | `live-auto` | `examples/eval/live/examples/process/output_sanitize_live.yaml` | Verifies useful output while complete synthetic prompt canary values remain absent; exact mask formatting remains mocked. |
| `examples/yaml/skills/skill_inline_only.yaml` | prompt-only inline skill routing | `live-auto` | `examples/eval/live/examples/skills/skill_inline_only_live.yaml` | Verifies real skill selection and a bounded beginner explanation. |
| `examples/yaml/skills/skill_external_only.yaml` | external math skill with calculator | `live-auto` | `examples/eval/live/examples/skills/skill_external_only_live.yaml` | Covers the external skill through its runnable parent and verifies calculator execution. |
| `examples/yaml/skills/skill_with_tools.yaml` | multi-step skill tool pipeline | `live-auto` | `examples/eval/live/examples/skills/skill_with_tools_live.yaml` | Verifies skill selection, calculator execution, and a stable result. |
| `examples/yaml/skills/skill_agent.yaml` | combined inline and external skills | `live-auto` | `examples/eval/live/examples/skills/skill_agent_live.yaml` | Covers one representative external skill path through the combined parent. |
| `examples/yaml/disambiguation/disambiguation_basic.yaml` | minimal ambiguous-input clarification | `live-auto` | `examples/eval/live/examples/disambiguation/disambiguation_basic_live.yaml` | Verifies vague input produces structural clarification evidence. |
| `examples/yaml/disambiguation/disambiguation_agent.yaml` | full-config vague-reference detection | `live-auto` | `examples/eval/live/examples/disambiguation/disambiguation_agent_live.yaml` | Verifies the full detector asks for missing context without tool execution. |
| `examples/yaml/disambiguation/disambiguation_multilingual.yaml` | Korean clarification | `live-auto` | `examples/eval/live/examples/disambiguation/disambiguation_multilingual_live.yaml` | Verifies multilingual ambiguity evidence and bounded Korean clarification. |
| `examples/yaml/disambiguation/disambiguation_with_state.yaml` | state-aware clarification | `live-auto` | `examples/eval/live/examples/disambiguation/disambiguation_with_state_live.yaml` | Verifies clarification while the current state remains unchanged. |
| `examples/yaml/memory/memory_basic.yaml` | in-memory multi-turn recall | `live-auto` | `examples/eval/live/examples/memory/memory_basic_live.yaml` | Uses a stable marker and request evidence to prove recall. |
| `examples/yaml/memory/memory_agent.yaml` | configured compacting memory | `live-auto` | `examples/eval/live/examples/memory/memory_agent_live.yaml` | Verifies summary injection and bounded marker recall after compaction. |
| `examples/yaml/memory/memory_budget.yaml` | token-budgeted memory | `live-auto` | `examples/eval/live/examples/memory/memory_budget_live.yaml` | Verifies the generated summary retains an early marker without requiring exact summary wording. |
| `examples/yaml/memory/memory_compacting.yaml` | compacting memory | `live-auto` | `examples/eval/live/examples/memory/memory_compacting_live.yaml` | Verifies router summarization and useful recall after compaction. |
| `examples/yaml/session/facts_basic.yaml` | actor-scoped fact extraction | `fixture-live` | `examples/eval/live/examples/session/facts_basic_live.yaml` | Rebuilds against isolated SQLite storage and proves exact fact injection without prior conversation history. |
| `examples/yaml/session/multi_actor.yaml` | actor identity and fact isolation | `fixture-live` | `examples/eval/live/examples/session/multi_actor_live.yaml` | Switches context-derived actors across conversation resets and proves marker isolation and returning-actor recall. |
| `examples/yaml/session/cross_session.yaml` | isolated persistent fact reuse | `fixture-live` | `examples/eval/live/examples/session/cross_session_live.yaml` | Rebuilds the runtime with attempt-local SQLite storage and actor identity preserved; cross-process CLI restart remains manual. |
| `examples/yaml/persona/persona_basic.yaml` | persona prompt composition | `live-auto` | `examples/eval/live/examples/persona/persona_basic_live.yaml` | Verifies identity content reaches the provider and visible response. |
| `examples/yaml/persona/persona_evolution.yaml` | bounded persona evolution | `live-auto` | `examples/eval/live/examples/persona/persona_evolution_live.yaml` | Verifies stable evolved metadata without asserting subjective drift. |
| `examples/yaml/persona/persona_secrets.yaml` | conditional secret protection | `live-auto` | `examples/eval/live/examples/persona/persona_secrets_live.yaml` | Requires structural non-reveal evidence and strict response non-disclosure. |
| `examples/yaml/relationships/support_relationship.yaml` | support relationship update | `live-auto` | `examples/eval/live/examples/relationships/support_relationship_live.yaml` | Verifies actor perspective and bounded relationship evidence. |
| `examples/yaml/relationships/two_sided_relationship.yaml` | two-sided relationship evidence | `live-auto` | `examples/eval/live/examples/relationships/two_sided_relationship_live.yaml` | Verifies both configured perspectives without exact score deltas. |
| `examples/yaml/relationships/persona_trust_secret.yaml` | trust-gated persona secret | `live-auto` | `examples/eval/live/examples/relationships/persona_trust_secret_live.yaml` | Verifies relationship evidence while keeping the secret undisclosed. |
| `examples/yaml/reasoning/reasoning_plan.yaml` | plan-and-execute calculator path | `live-auto` | `examples/eval/live/examples/reasoning/reasoning_plan_live.yaml` | Verifies one calculator execution with `source_in: [plan]`, exact arguments, and a public result without private reasoning text. |
| `examples/yaml/reasoning/reasoning_cot.yaml` | visible tagged chain-of-thought | `mocked-only` | mocked reasoning suite | Do not assert or promote private reasoning traces in live smoke coverage. |
| `examples/yaml/reasoning/reasoning_reflection.yaml` | semantic reflection quality | `mocked-only` | mocked suite or future `live/quality/` suite | Exact reflection lifecycle is deterministic; semantic improvement belongs in quality evaluation. |
| `examples/yaml/reasoning/reasoning_with_state.yaml` | state plus visible reasoning/reflection | `mocked-only` | mocked reasoning suite | A greeting-only live check would not cover the example's primary behavior. |
| `examples/yaml/observability/basic_metrics.yaml` | main response telemetry | `live-auto` | `examples/eval/live/examples/observability/basic_metrics_live.yaml` | Verifies stable model, purpose, and success dimensions. |
| `examples/yaml/observability/cost_by_model.yaml` | skill cost attribution | `live-auto` | `examples/eval/live/examples/observability/cost_by_model_live.yaml` | Verifies stable purpose/model dimensions without exact cost totals. |
| `examples/yaml/observability/language_breakdown.yaml` | language dimensions | `live-auto` | `examples/eval/live/examples/observability/language_breakdown_live.yaml` | Verifies detected language reaches telemetry. |
| `examples/yaml/observability/orchestration_metrics.yaml` | delegate telemetry | `live-auto` | `examples/eval/live/examples/observability/orchestration_metrics_live.yaml` | Verifies route, participant, and orchestration dimensions. |
| `examples/yaml/observability/pricing_file.yaml` | pricing-file configuration | `live-auto` | `examples/eval/live/examples/observability/pricing_file_live.yaml` | Verifies configured model telemetry without exact provider token or cost values. |
| `examples/yaml/observability/tools_and_skills_metrics.yaml` | skill and tool telemetry | `live-auto` | `examples/eval/live/examples/observability/tools_and_skills_metrics_live.yaml` | Verifies skill purposes, one calculator call, and successful telemetry. |
| `examples/yaml/observability/pricing.yaml` | pricing data | `support-file` | covered by `pricing_file_live.yaml` | Data file, not a runnable agent. |
| `examples/yaml/error-recovery/basic_retry.yaml` | bounded retry-enabled operation | `live-auto` | `examples/eval/live/examples/error-recovery/basic_retry_live.yaml` | Verifies normal successful operation without deliberately paying for failed calls. |
| `examples/yaml/error-recovery/llm_fallback.yaml` | configured provider fallback | `live-auto` | `examples/eval/live/examples/error-recovery/llm_fallback_live.yaml` | Verifies bounded availability; exact failure/fallback sequencing remains mocked. |
| `examples/yaml/error-recovery/context_overflow.yaml` | overflow summarization recovery | `mocked-only` | mocked context-overflow suite | Reliable live triggering requires large paid inputs and multiple summarization calls. |
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
| `examples/yaml/tools/web_search_research.yaml` | `web_search`, `web_fetch` | `fixture-live` | `examples/eval/live/examples/tools/web_search_research_live.yaml` | Uses unavailable search plus an empty no-socket WebFetch transport; the model must suggest, but not execute, the fallback. |
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

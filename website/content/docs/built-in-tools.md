+++
title = "Built-in Tools"
weight = 8
template = "docs.html"
description = "Canonical inputs, outputs, safety, policy, host integration, and eval coverage for all 30 built-in tools."
+++

The runtime registers 30 canonical built-ins. A YAML agent can expose one only by granting its canonical ID in top-level `tools:`; omitted or empty `tools:` grants no ordinary tools, and state-level lists can narrow but not widen that grant. See the [YAML Reference](@/docs/yaml-reference.md) for tool security and scoping, [Concepts](@/docs/concepts.md) for the runtime model, and the [Examples catalog](@/examples/_index.md) for runnable agents.

## Canonical inventory

The exact canonical IDs, in registry order, are:

```text
calculator, echo, datetime, json, random, file,
glob, grep, file_read, file_write, file_edit, patch,
copy_path, move_path, delete_path, file_list, file_info,
git_status, git_diff, diagnostics, ask_user, todo, sleep,
web_fetch, web_search, command, text, template, math, http
```

Aliases and display names resolve to a canonical ID before scope, policy, HITL, recovery, observability, and eval evidence are applied.

## Common execution and evidence contract

Every currently emitted production source - model call, skill step, state action, plan step, fallback, and manual invocation - converges on the shared runtime executor. Task, orchestration, and spawner source variants are reserved for future producers; evaluation fixtures are harness-only. The stable order is:

```text
runtime deny -> canonical resolution -> grant/scope -> argument normalization
-> safety classification -> bound path/domain/operation/command/limit policy
-> approval when required -> policy recheck after modified approval
-> timeout/cancellation -> side-effect resource lock -> final admission
-> tool invocation -> output cap -> hooks, observability, and evidence
```

Policy denial cannot be overridden by approval. Approved arguments are rechecked, and a call is re-resolved and re-authorized after an approval wait. Automatic retry is limited to calls classified as safely retryable; non-idempotent mutation and command calls are not retried as if they were reads.

A tool implementation returns `ToolResult { success, output, metadata }`. Successful built-ins return JSON in `output`; the per-tool tables below list the stable fields in that JSON. The executor stores a `ToolExecutionRecord` with:

- identity and source: `call_id`, `requested_name`, `canonical_id`, `source`;
- arguments and versions: `arguments`, `executed_arguments`, `policy_version`, `registry_version`, `runtime_config_version`;
- outcome: `executed`, `success`, `output`, `metadata`, `policy`, `approval`;
- timing and interruption: `started_at`, `duration_ms`, `timed_out`, `cancelled`, `cancellation_reason`, `output_truncated`.

`executed` is `true` only when the tool implementation was invoked. Denied, unavailable, rejected, timed-out, cancelled, failed, and successful attempts all retain structured evidence. Eval `tool_called` assertions can inspect execution flags and arguments, while `result_path` parses successful structured output; see [Evaluation](@/docs/evaluation.md).

### Safety and policy notation

The tables use the runtime's side-effect levels: `none`, `local_write`, `external_read`, `external_write`, and `destructive`. "No default approval" does not bypass an explicit `require_confirmation` rule or a matching path, domain, operation, or command `requires_approval` rule.

Policy bindings identify which arguments the shared executor can enforce:

- `read(path)`, `write(path)`, `read_write(path)`: path policy;
- `url(url)`: domain, scheme, and port policy for that URL argument;
- `operation(field)`: operation allow, deny, approval, or unavailable policy;
- `command(...)`: exact argv/template, cwd, environment, and shell policy;
- `limit(field -> kind)`: request values lowered by effective policy limits.

Tools with no argument binding still receive common tool-level enablement, rate limit, timeout, confirmation, runtime control, result caps, and evidence handling. `tool_security.enabled` defaults to `false`; this does not change the explicit YAML grant requirement. For mutation and command tools, use enabled, fail-closed policy with narrow write roots or exact commands. Every framework-owned built-in input named `max_results` accepts only positive integers; zero is rejected before provider or tool work begins.

## Local compute and legacy utility tools

| Canonical ID | Inputs and defaults | Stable successful output | Safety, approval, and bindings |
|---|---|---|---|
| `calculator` | Required `expression`. | `result`, `expression`. | Compute, `none`, no default approval; no argument bindings. |
| `echo` | Required `message`. | `message`, `length` (UTF-8 byte length). | Compute, `none`, no default approval; no argument bindings. |
| `datetime` | Required `operation`: `now`, `format`, `parse`, `add`, or `diff`. `now`: optional `format` (default `%Y-%m-%d %H:%M:%S UTC`). `format`: `value` required, `format` defaults to `%Y-%m-%d`. `parse`: `value` required. `add`: `value` and `amount` required, `unit` defaults to `days`. `diff`: `value` and `value2` required. All results are UTC. | `now`: `iso`, `unix_timestamp`, `formatted`. `format`: `formatted`, `original`. `parse`: `iso`, `unix_timestamp`. `add`: `result`, `unix_timestamp`, `original`, `added`. `diff`: `seconds`, `minutes`, `hours`, `days`, `value1`, `value2`. | Compute, `none`, no default approval; no argument bindings. |
| `json` | Required `operation`: `parse`, `get`, `set`, `merge`, `stringify`, `keys`, or `values`. `data` is required by every operation; `get` also requires `path`; `set` requires `path` and `value`; `merge` requires `data2`. `stringify.pretty` defaults to `true`. | `parse`: `parsed`, `valid`. `get`: `value`, `path`, `found`. `set`: `result`, `path`. `merge`: `result`. `stringify`: `result`. `keys`: `keys`, `count`. `values`: `values`, `count`. | Compute, `none`, no default approval; no argument bindings. |
| `random` | Required `operation`: `uuid`, `number`, `integer`, `choice`, `shuffle`, `bool`, or `string`. `number`: `min=0`, `max=1`. `integer`: `min=0`, `max=100`. `choice`: `items` required, `count=1` and capped to item count. `shuffle`: `items` required. `string`: `length=16`, `charset=alphanumeric`; accepted sets also include `alpha`, `numeric`, `hex`, `lower`, and `upper`. | `uuid`: `uuid`. `number`/`integer`: `value`, `min`, `max`. `choice`: `selected`, `count`. `shuffle`: `shuffled`, `count`. `bool`: `value`. `string`: `value`, `length`. | Compute, `none`, no default approval; no argument bindings. |
| `text` | Required `operation`. Optional `text` defaults to `""`. `substring`: `start=0`, `end=character count`. `replace`/`contains`/`starts_with`/`ends_with`/`index_of`: `find=""`; `replace_with=""`. `split`: `delimiter=" "`; `join`: `items=[]`, `delimiter=""`; `repeat.count=1`; padding `width=0`, `pad_char=" "`; truncate `width=character count`, `suffix="..."`; `char_at.index=0`. | `length`: `length`, `bytes`. Most transforms: `result`. `contains`/prefix/suffix: boolean `result`. `split`/`words`: `parts`, `count`. `lines`: `lines`, `count`. `char_at`: `char`, `found`. `index_of`: `index`, `found`. | Compute, `none`, no default approval; no argument bindings. Unicode character indexing is used where the operation is character-oriented. |
| `template` | Required `operation` and `data`. `render` also requires `template`; `render_file` requires `path`. | `render`: `rendered`. `render_file`: `rendered`, `template_path`. | Inline render is compute; `render_file` is read-only. No default approval. No path binding is declared, so use the split file tools when executor-enforced path policy is required. |
| `math` | Required `operation`. `mean`, `median`, `mode`, `stdev`, `variance`, `sum`, `min`, `max`, `minmax`, and `count` require `values`. `abs`, `round`, `floor`, `ceil`, `sqrt`, `log`, and `log10` require `value`; `round.decimals=0`. `clamp` requires `value`, `min`, `max`; `percentage` requires `value`, `total`; `pow` requires `exponent` and either `value` or `base`; `log.base` is optional and omission means natural log. `range` requires `max`, with `min=0`, `step=1`. | Most operations: `result` (statistics and count also include `count`). `stdev`: `stdev`, `variance`, `mean`, `count`. `mode`: `mode`, `frequency`. `minmax`: `min`, `max`, `range`. `clamp`: `result`, `clamped`. `range`: `range`, `count`. | Compute, `none`, no default approval; no argument bindings. |
| `file` | Legacy mixed tool. Required `operation` and `path`. Operations: `read`, `write`, `append`, `exists`, `delete`, `list`, `mkdir`, `info`. `content` defaults to empty for write/append; `pattern` is optional for list. | `read`: `content`, `path`, `size`. write/append: `success`, `path`, `bytes_written`. `exists`: `exists`, `path`, `is_file`, `is_dir`. `delete`: `success`, `path`. `list`: `entries`, `path`, `count`; entries contain `name`, `path`, `is_file`, `is_dir`, `size`. `mkdir`: `success`, `path`. `info`: `path`, `exists`, `is_file`, `is_dir`, `size`, optional `modified`, `created`. | Reads classify `none` with no default approval; write/append/mkdir classify `local_write` and delete `destructive`, with default approval. Bindings: `read_write(path)`, `operation(operation)`. Raw `.git` paths are blocked. Prefer the split tools below for bounded reads and controlled mutation. |

## Workspace and repository inspection

| Canonical ID | Inputs and defaults | Stable successful output | Safety, approval, and bindings |
|---|---|---|---|
| `glob` | Required `pattern`. Optional `path="."`, `max_results=100`, `offset=0`, `include_dirs=false`, `sort=path` (`modified` and `size` also supported). | `paths`, `count`, `total_count`, `offset`, `truncated`, `duration_ms`. | Read, `none`, no default approval. `read(path=".")`; `limit(max_results -> max_results)`. Default ignored directories include `.git`, `target`, `node_modules`, `dist`, `build`, `.next`, and `.turbo`. |
| `grep` | Required `pattern`. Optional `mode=regex`, `path="."`, `include_glob`, `case_sensitive=false`, `output_mode=files_with_matches`, `context=0`, `max_results=250`, `offset=0`, `max_file_size_bytes=1 MiB`, `max_output_chars=20000`. Other output modes: `content`, `count`. | `mode`, `matches`, `files`, `count`, `total_count`, `offset`, `truncated`, `skipped_binary`, `skipped_large`; each match has `path` and operation-dependent `line`, `text`, or `count`. | Read, `none`, no default approval. `read(path=".")`; limits for results, file bytes, and output characters. Binary/non-UTF-8 and oversized files are skipped. |
| `file_read` | Required `path`. Optional `start_line=1`, `end_line`, `max_lines=2000`, `max_bytes=1 MiB`. Large files require an explicit range. | `path`, `content`, `start_line`, `end_line`, `total_lines`, `bytes_read`, `file_size`, `truncated`, `large_file`, `encoding`, optional `version` (`path`, `sha256`, `size_bytes`, optional `modified_unix_ms`). | Read, `none`, no default approval. `read(path)`; limits for bytes and lines. UTF-8 text only; raw `.git` paths are blocked. Version evidence can satisfy read-before-write policy for split mutation tools. |
| `file_list` | Required `path`. Optional `recursive=false`, `include_glob`, `exclude_glob`, `include_hidden=false`, `max_results=200`, `offset=0`, `sort=path` (`modified`, `size`, `kind` also supported). | `path`, `entries`, `count`, `total_count`, `offset`, `truncated`, `policy_notes`; entries contain `path`, `kind`, optional `size`, `modified`, `symlink`, optional `policy`. | Read, `none`, no default approval. `read(path)`; result limit. Hidden/default ignored directories and symlink escapes are handled conservatively. |
| `file_info` | Required `path`; optional `follow_symlinks=false`. | `path`, `exists`, `kind`, optional `size`, `modified`, `created`, `readonly`, `symlink`, optional `canonical_path`, `mime_hint`, and `policy_classification`. Missing paths return `exists=false`, `kind=missing`. | Read, `none`, no default approval. `read(path)`. Raw `.git` paths and unsafe symlink resolution are blocked or classified. |
| `git_status` | No required field. Optional `path="."`, `include_untracked=true`, `max_results=200`. | `branch`, `staged`, `unstaged`, `untracked`, `count`, `truncated`; each entry contains `path`, `status`. | VCS inspect, `none`, no default approval. `read(path=".")`; result limit. Runs fixed read-only Git commands and never exposes raw `.git` paths. |
| `git_diff` | No required field. Optional `path="."`, `staged=false`, `paths=[]`, `max_output_chars=20000`. | `staged`, `paths`, `summary`, `diff`, `truncated`. | VCS inspect, `none`, no default approval. `read(path=".")`; output limit. Uses fixed `git diff` commands; raw `.git` path filters are rejected. |

## Controlled filesystem mutation

All six split mutation tools require an explicit allowed write root for an actual mutation. With no write policy, the built-in default is dry-run only; `no_write_policy: deny` also blocks dry runs. Denied paths override allowed roots, path traversal and raw `.git` targets are rejected, and candidates plus configured roots use nearest-existing-entry resolution before containment decisions. Relative roots are anchored to the canonical host-owned workspace, explicit absolute roots authorize their resolved locations, and dangling symlinks fail closed. Allow rules require resolved containment, while deny, unavailable, and approval restrictions retain lexical-or-resolved matching while the checked path topology remains unchanged. Actual calls require approval by classification unless policy explicitly sets `allow_without_confirmation: true`; dry runs classify as read-only, safely retryable, and do not require approval.

The v1 filesystem contract assumes a host-owned workspace without concurrent untrusted external replacement of validated files, directories, or parent paths. Runtime locks serialize framework-owned conflicting mutations, but they are not OS directory handles or a filesystem sandbox; use process, container, mount, and permission isolation when another process can modify the workspace concurrently.

`file_write`, `file_edit`, and `patch` share read-version evidence with `file_read`. When `require_read_before_write: true`, an existing file must have been read and must still match that version before apply.

| Canonical ID | Inputs and defaults | Stable successful output | Safety and bindings |
|---|---|---|---|
| `file_write` | Required `path`, `content`. Optional `overwrite=false`, `create_parent_dirs=false`, `dry_run=false`. Both request flags and policy must allow overwrite/parent creation. | Common mutation output: `path`, `dry_run`, `mutation_performed`, `changed_files`, `changed_lines`, `replacements`, `bytes_written`, `created`, `overwritten`, `truncated`, `approval_required`, `diff_summary`, `changed_paths`, optional `version`, `near_matches`. | Apply: write, `local_write`; dry run: `none`. `write(path)`; changed-file and changed-line limits. Writes atomically. |
| `file_edit` | Required `path`, `old_text`, `new_text`. Optional `replace_all=false`, `dry_run=false`, `max_replacements=20` before lower policy caps. Without `replace_all`, the old text must occur exactly once. | Common mutation output above. A no-match failure still returns structured fields plus up to three `near_matches`. | Apply: edit, `local_write`; dry run: `none`. `write(path)`; replacement and changed-line limits. |
| `patch` | Required unified-diff `patch`. Optional `base_path="."`, `dry_run=false`, `allow_new_files=false`, `allow_delete=false`. Built-in maxima are 10 changed files and 500 changed lines, further lowered by policy. | Common mutation output above; `path` is the base path and `changed_paths` lists targets. | Apply: patch, `local_write`; dry run: `none`. `write(base_path=".", patch_base)`; changed-file and changed-line limits. Multi-file apply is transactional with rollback attempts. |
| `copy_path` | Required `source_path`, `destination_path`. Optional `overwrite=false`, `create_parent_dirs=false`, `dry_run=false`. Both request flags and policy must allow overwrite/parent creation. | `source_path`, `destination_path`, `path`, `dry_run`, `mutation_performed`, `copied`, `moved`, `deleted`, `recursive`, `overwritten`, `bytes_affected`, `items_affected`, `approval_required`, `diff_summary`, and optional `error`, `cleanup_warning`, `retained_backup_path`. | Apply: write, `local_write`; dry run: `none`. `read(source_path)`, `write(destination_path)`. Symbolic-link sources are unsupported. |
| `move_path` | Required `source_path`, `destination_path`. Optional `overwrite=false`, `create_parent_dirs=false`, `dry_run=false`. | Same path-mutation output as `copy_path`. | Apply: write, `local_write`; dry run: `none`. `read_write(source_path)`, `write(destination_path)`. |
| `delete_path` | Required `path`. Optional `recursive=false`, `dry_run=false`; directories require `recursive=true`. | Same path-mutation output, using `path`; `deleted`, `recursive`, byte/item counts, and summary describe the result. | Apply: delete, `destructive`; dry run: `none`. `write(path)`. Refuses to delete a configured write root. |

## Host-backed and session tools

| Canonical ID | Inputs and defaults | Stable successful output | Safety, approval, and bindings |
|---|---|---|---|
| `diagnostics` | No required field. Optional `path`, `severity=all` (`error`, `warning`, `info`, `hint`), `max_results=200`. | `available`, `diagnostics`, `count`, `truncated`, optional `message`; each diagnostic has `path`, optional `line`, `column`, `source`, `code`, plus `severity`, `message`. | Diagnostics read, `none`, host-dependent, no default approval. `read(path=".")`; result limit. Requires an available host provider. |
| `ask_user` | Required `question`. Optional `options=[]`, `multi_select=false`, `allow_other=true`, `default`, `timeout_seconds`. | `answered`, `selected`, optional `other_text`, `timed_out`, `unavailable`. | Interactive, `none`, host-dependent and user-interactive, no approval gate; no argument bindings. No handler uses the executed structured fallback described below. |
| `todo` | Required `operation`: `list`, `set`, `update`, or `clear`. `set.items=[]`; each item requires `id`, `content`, with optional `active_form`, `status=pending`. `update` requires `id` and accepts optional `status`, `content`, `active_form`. Status values: `pending`, `in_progress`, `completed`, `cancelled`. | `operation`, `items`, `count`, optional `updated`; items contain `id`, `content`, optional `active_form`, `status`. | Write, `local_write`, session-local, no default approval. `operation(operation)`. |
| `sleep` | Required `duration_ms`; optional `reason`. Default maximum is 30000 ms, lowered by execution timeout, or replaced by host `config.max_duration_ms` before the timeout cap. | `slept_ms`, `max_duration_ms`, optional `reason`. | Wait, `none`, cancellable, no default approval; no argument bindings. |
| `command` | Either non-empty `argv` (preferred) or compatibility `command` is required. Optional `cwd="."`, `env={}`, `timeout_ms=30000`, `max_output_chars=20000`, `reason`. Shell metacharacters are rejected in command-string form. | `success`, optional `exit_code`, `termination`, `stdout`, `stderr`, `combined_output`, `truncated`, `timed_out`, `cwd`, redacted `argv`, optional `reason`. A nonzero exit can be represented inside this successfully executed structured result. | Command, `local_write`, host-dependent, default approval, not safely retryable. Bindings: `command(argv)`, `command(command)`, `command(env)`, `read_write(cwd=".")`, output limit. Requires an available host runner plus exact `allowed_commands` or `command_templates` and allowed `working_dirs`; environment starts empty and only policy-approved values pass through. |

## Web and HTTP tools

| Canonical ID | Inputs and defaults | Stable successful output | Safety, approval, and bindings |
|---|---|---|---|
| `web_fetch` | Required `url`. Optional extraction `prompt`, `max_chars=20000`, `cache_ttl_seconds=900` (`0` disables cache; a positive representable value sets the lifetime when a response is stored and cache hits do not refresh it; unrepresentable values fail before network access), `max_response_bytes=1 MiB` per response including redirect responses, `max_redirects=5`; effective policy can lower output, response-byte, and redirect limits. | `url`, `final_url`, `status`, optional `content_type`, `content`, `truncated`, `from_cache`, `redirects`, `extraction_prompt_used`, `extraction_available`. | Network read, `external_read`, open-world, no default approval. `url(url)`; output, response-byte, and redirect limits. Only HTTP(S), no embedded credentials; localhost, metadata, private, link-local, multicast, documentation, and other blocked IP ranges are rejected. The initial bound URL can execute for `domains.requires_approval` only when the shared execution context contains `Approved` or `Modified` evidence, but a redirect into an approval-required domain is blocked before the next DNS or transport request because redirects cannot start a second approval flow. DNS/IP and configured URL policy are checked before every request and redirect, including cached redirect evidence. The process-local response cache is limited to 128 entries and lazily removes expired entries. Cache reuse also requires the stored redirect count to satisfy the current effective redirect limit; incompatible chains are fetched again under the stricter call. A compatible hit avoids the HTTP transport request but repeats current DNS/IP and URL-policy validation. The default transport disables proxies and connects only to the addresses approved for that request. |
| `web_search` | Required `query`. Optional `max_results=5`, `include_domains=[]`, `language`, `region`, `safe_search` (`off`, `moderate`, `strict`; omission is passed to the provider as no preference). | `available`, `query`, optional `provider`, `count`, `truncated`, `results`, optional `message`; each result has `title`, `url`, `snippet`, optional `source`, `published_at`. | Network read, `external_read`, open-world, host-dependent, no default approval. Only `limit(max_results -> max_results)` is bound. Requires an available host provider. `include_domains` is a provider request field, not an executor URL-policy binding. |
| `http` | Required `method`, `url`. Methods: `GET`, `POST`, `PUT`, `DELETE`, `PATCH`, `HEAD`. Optional `headers`, `body`, `timeout_ms=30000`. | `status`, `status_text`, `headers`, `body`. | Network/open-world. `GET`/`HEAD` classify `external_read` with no default approval; other methods classify `external_write`. Bindings: `url(url)`, `operation(method)`. This is a raw API client, not the public-content safety profile of `web_fetch`; configure domain/method policy and explicit confirmation or approval rules for state-changing methods. Runtime output caps still apply. |

## Unavailable and fallback behavior

Three host-backed tools have an explicit runtime availability preflight:

| Tool | Missing host capability | Evidence |
|---|---|---|
| `diagnostics` | No available `DiagnosticsProvider`. | `executed: false`, `success: false`, policy outcome `unavailable`. |
| `command` | No available `CommandRunner`. | `executed: false`, `success: false`, policy outcome `unavailable`. |
| `web_search` | No available `WebSearchProvider`. | `executed: false`, `success: false`, policy outcome `unavailable`. |

Because preflight stops these calls before `Tool::execute`, their normal successful output object is not produced. The model receives a structured `tool_unavailable` error derived from the execution record.

`ask_user` is deliberately different. When no `QuestionHandler` is installed, its implementation still runs, so evidence is `executed: true`, `success: true`. It returns the normal structured question response with `unavailable: true`; a supplied `default` becomes the fallback answer, while omission returns `answered: false`. A handler timeout also executes and returns fallback data with `timed_out: true`.

Other environmental failures happen after normal resolution unless policy blocks first. For example, a missing Git executable produces an executed tool error, while an initial URL denied by bound domain policy can be rejected before `web_fetch` or `http` is invoked.

## `web_search` versus `web_fetch`

The host implements `ai_agents::tools::WebSearchProvider`, whose availability check and asynchronous `search(WebSearchRequest) -> WebSearchResponse` normalize provider output. Install it on a runtime with `RuntimeAgent::set_web_search_provider(Arc<dyn WebSearchProvider>)`. The facade exports the normal provider/request/response contract; eval-only static and unavailable implementations remain in `ai-agents-tools`.

Evaluation uses `StaticWebSearchProvider` for deterministic exact-query responses and `UnavailableWebSearchProvider` when no fixture/provider is configured. The runtime checks `is_available()` before invoking the tool.

`web_fetch` retrieves a known URL itself through the built-in GET transport. It owns URL parsing, DNS/IP validation, validated-address binding, redirect-by-redirect policy, per-response byte limits, HTML-to-text conversion, bounded process-local caching, and optional LLM extraction. The default transport disables proxies because proxy-side hostname resolution would bypass the validated address set. A custom low-level `WebFetchTransport` must return one HTTP response per call without automatically following redirects so the tool can validate every hop. A socket-opening implementation must override validated sending, enforce the supplied address set, and honor `max_response_bytes` independently while reading each response in a redirect chain. Host firewall and network egress controls remain the final deployment boundary.

These boundaries are intentional:

- `web_search` has no URL/domain policy binding; `include_domains` is passed to the host provider as a result filter/hint.
- Any network requests made internally by a `WebSearchProvider` are host-provider behavior. They are **not** automatically governed by `web_fetch` URL, redirect, DNS/IP, private-network, or cache policy.
- A host provider must enforce its own endpoint, credential, network, privacy, rate, and cost controls.
- Search result URLs are evidence only until the agent separately calls `web_fetch` or another granted tool.

## Runnable examples and eval coverage

- [Runnable YAML examples](@/examples/_index.md) include basic/data tools, workspace and repository inspection, all six split mutation tools, diagnostics, questions, todos, sleep, command validation, web fetch/search, legacy file/template, and raw HTTP.
- [Evaluation guide](@/docs/evaluation.md) explains mocked versus live-provider execution, fixtures, structural tool evidence, and safety assertions.
- [Mocked tool suites](https://github.com/geminik23/ai-agents/tree/main/examples/eval/mocked/tools) provide no-key coverage for successful calls, denials, dry runs, approvals, bounded web fetch, static web search, and unavailable search.
- [Live tool suites](https://github.com/geminik23/ai-agents/tree/main/examples/eval/live/examples/tools) use a real LLM only when explicitly authorized; external tool behavior remains fixture-backed, read-only, no-socket, or dry-run-only.
- [`ask_user` fallback mocked eval](https://github.com/geminik23/ai-agents/blob/main/examples/eval/mocked/tools/ask_user_fallback_mocked.yaml) and its [live-model counterpart](https://github.com/geminik23/ai-agents/blob/main/examples/eval/live/examples/tools/interactive_choice_live.yaml) assert `executed: true` structured fallback behavior.
- [`web_search` mocked eval](https://github.com/geminik23/ai-agents/blob/main/examples/eval/mocked/tools/web_search_mocked.yaml), [`web_search` unavailable eval](https://github.com/geminik23/ai-agents/blob/main/examples/eval/mocked/tools/web_search_unavailable_mocked.yaml), and the [live-model unavailable/fallback suite](https://github.com/geminik23/ai-agents/blob/main/examples/eval/live/examples/tools/web_search_research_live.yaml) cover static and unavailable providers without public search traffic.
- [`diagnostics` mocked eval](https://github.com/geminik23/ai-agents/blob/main/examples/eval/mocked/tools/diagnostics_mocked.yaml) and its [fixture-backed live-model suite](https://github.com/geminik23/ai-agents/blob/main/examples/eval/live/examples/tools/diagnostics_review_live.yaml) use static diagnostics rather than a real editor service.

The live coverage registry keeps real command execution and public-network `web_fetch`/`http` calls deferred. Do not treat configuration parsing or a fixture-backed live-model run as proof that public network or process execution was performed.

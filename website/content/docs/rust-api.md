+++
title = "Rust API"
weight = 5
template = "docs.html"
description = "Embed YAML-defined AI agents in Rust with AgentBuilder, or build agents programmatically with custom tools, providers, memory, and hooks."

[extra]
seo_title = "Build and Embed AI Agents in Rust"
+++

Use the `ai-agents` Rust crate to load YAML definitions with `AgentBuilder` or construct agents programmatically. This guide covers embedding the runtime in your application and extending it with tools, providers, memory backends, and hooks. Start with [dependencies and a first agent](@/docs/getting-started.md#your-first-agent-rust), then use this reference for integration details.

---

## Adding the Dependency

Add `ai-agents` to your `Cargo.toml`:

```toml
[dependencies]
ai-agents = "1.1.2"
tokio = { version = "1", features = ["full"] }
anyhow = "1"
```

### Feature Flags

| Flag            | Description                                        |
| --------------- | -------------------------------------------------- |
| *(default)*     | Core framework, no optional storage or context     |
| `sqlite`        | SQLite storage backend for session persistence     |
| `redis-storage` | Redis storage backend for session persistence      |
| `http-context`  | HTTP context source for dynamic context injection  |
| `full-storage`  | `sqlite` + `redis-storage`                         |
| `full`          | All optional features enabled                      |

Enable features like this:

```toml
[dependencies]
ai-agents = { version = "1.1.2", features = ["full"] }
```

---

## Auxiliary routing migration (1.1)

Version 1.1.0 introduces hierarchical routing with an explicitly approved, one-time Rust source-compatibility exception. This is not an ordinarily source-compatible SemVer minor. The following migration applies to Rust consumers moving from 1.0.x; scalar YAML selection retains legacy behavior.

- `LLMSelector.router` changes from `Option<String>` to `Option<RouterSelector>`. `Alias(String)` preserves scalar configuration; `Hierarchical(Box<RouterRolesConfig>)` is explicit opt-in. The opt-in tree is boxed to keep scalar agent construction compact.
- `DetectionConfig.llm`, `LlmGenerateConfig.llm`, `ContextExtractor.llm`, and `ToolCondition::Semantic.llm` change from `String` to `Option<String>`. Use `None` for omission and `Some(alias)` for an explicit override. Their YAML serializers omit `None`; present null is rejected.
- `with_router(alias)` remains the scalar constructor. Use `with_router_roles(config)` for hierarchy. Resolve roles with `LLMRegistry::resolve_role_override()`; `Ok(None)` means legacy mode, not an unspecified hierarchy leaf. Never turn a hierarchy error into legacy fallback with `.ok()`.

```rust
use ai_agents::llm::{RouterRolesConfig, StateRouterConfig};
use ai_agents::spec::LLMSelector;

let scalar = LLMSelector::new("main").with_router("fast");
let hierarchy = LLMSelector::new("main").with_router_roles(RouterRolesConfig {
    default: Some("fast".to_string()),
    state: Some(StateRouterConfig {
        transition: Some("precise".to_string()),
        ..Default::default()
    }),
    ..Default::default()
});
```

Replace an old `router: Some("fast".into())` struct literal with `router: Some(RouterSelector::Alias("fast".into()))`, or prefer the constructor. Replace an explicitly assigned local string with `Some(string)`; do not replace an omitted local with `Some("router")`, because that suppresses role inheritance.

Hierarchy preflight uses the actual registry, including host-provided aliases. External skills are prepared once. After generated tools/shared children begin capturing providers, provider/settings changes fail at the next fallible gate or final build, including replacement under the same alias. Finish provider registration before configuring the spawner. Legacy builder chaining retains its prior behavior.

Low-level orchestration retains the one-provider `concurrent()` and `group_chat()` wrappers. Additive `concurrent_with_llms()`, `group_chat_with_llms()`, and `aggregation::aggregate_with_llms()` accept separate aggregation/group providers. `LLMSummarizer::new(provider)` still shares one provider between summarize and merge; `with_merge_llm()` separates the merge provider. Custom injected implementations are not forcibly rewired.

Use `RuntimeAgent::try_new()` for fallible low-level hierarchy construction; `new()` remains a compatibility convenience and panics on invalid hierarchy construction. The recommended `AgentBuilder` path returns configuration errors. Child snapshot serialization uses prepared skill declarations without modifying `SpawnedAgent.spec`; retained restore compares routing configuration and does not hot-swap live models or prompts.

## Autonomy configuration migration (development only)

The development tree adds typed `autonomy` fields to `AgentSpec`, `StateDefinition`, and `SkillDefinition`. Existing exhaustive Rust struct literals must add `autonomy: AutonomyConfig::default()` for `AgentSpec`, or `autonomy: None` for states and skills. Existing `..Default::default()` literals remain usable where the type supports them. Omitted YAML keeps normal chat unchanged.

```rust
use ai_agents::{AutonomyConfig, AgentSpec, StateDefinition};

let agent = AgentSpec {
    autonomy: AutonomyConfig::default(),
    ..AgentSpec::default()
};
let state = StateDefinition {
    autonomy: None,
    ..StateDefinition::default()
};
```

For a `SkillDefinition` literal, add `autonomy: None` alongside its existing required ID, description, trigger, steps, and other optional fields. The raw profile and completion-gate types are available through `ai_agents::autonomy`, and a **restricted standalone development runner** is available through the explicit Rust API. The complete autonomy contract is not implemented. Enabled state and selected-skill declarations now install automatic entry through the same controller and retained host service. Top-level settings configure profiles rather than implicitly converting every chat into a standalone task. The complete task hosting/eval contract is still unfinished; do not use this development-tree note as the published 1.1.1 API reference.

### Task checkpoint storage (development only)

The development tree also provides runtime task/result and checkpoint types through `ai_agents::autonomy`. `TaskRunSnapshot` is a separately versioned core storage envelope; `TaskRun` and `TaskRunResult` are runtime-facing views, not the persistence ABI. The evaluator and extension primitives described below are available in development. Retained foreground approval/question continuation and explicit host-authorized reconstruction of prepared safe pauses are available. Automatic YAML state/skill declarations are supported through embedded hosting; development task commands and the plain REPL now consume this hosting, while default production CLI reconstruction and automatic abandoned-owner takeover remain unavailable. Explicit Rust startup selection and reviewed recovery are described below.

`AgentStorage` adds `StorageCapability::TaskRuns` and five defaulted methods: `create_task_run`, `load_task_run`, `list_task_runs`, `mutate_task_run`, and `delete_task_run`. Existing custom backends need no new methods to compile: defaults return `UnsupportedStorageCapability(TaskRuns)`. SQLite supports these conditional operations; `InMemoryTaskStorage` supports the same transitions only for its retained process-local lifetime. File, Redis, Noop and `NamespacedStorage` do not support task operations. Ordinary snapshot support is not proof of task transaction support.

`TaskRunMutation` covers claim, checkpoint/release, cancellation request and acknowledgement, explicit abandoned-owner recovery, and reconciliation. Writes compare revision and owner before changing data. A cancellation request does not acknowledge cleanup; paused unowned runs use explicit acknowledgement, and uncertain effects require explicit reconciliation. Terminal states do not resume. Deleted run identities remain tombstoned and cannot be reused; SQLite actor-data deletion includes associated tasks and rejects active task owners atomically. No automatic owner takeover or exactly-once external-effect guarantee is provided.

`ScopedTaskRunStore` fixes agent/actor scope and a host-supplied configuration compatibility identity, then validates runtime payloads before writes. The identity must describe the actual prepared configuration and host bindings, not merely a mutable profile name; storage scoping is not authentication. Core backend methods validate the envelope but do not interpret the runtime payload. SQLite and in-memory tests cover competing claims, cancellation/completion races, stale writers and deletion safety.

Checkpoints retain exact native history, completed skill/batch results, pending identities, cumulative counters, resolved ceilings, independent clock data, effect reservations, adapter state, progress and child data. The envelope limit is 4 MiB and the runtime typed-record limit is 4,096, including nested continuations and state history. Required recovery data is rejected when excessive, never silently truncated. Listings contain metadata only, without objective or tool output. Raw snapshots and runtime-facing results may contain sensitive data; hosts must protect backend access, retention, logs and display. The run-bound todo adapter uses the existing canonical store, rejects stale handles, and does not count empty or entirely cancelled lists as completion.

### Explicit safe-pause reconstruction (development only)

`ai_agents::autonomy` re-exports `TaskRestoreHost`, `TaskRestoreManifest`, `TaskRestoreGraph`, `TaskRestoreRequest`, `TaskRestoreBinding`, `TaskRestorationReview` and the opaque `TaskReconstructionCheckpoint`. This is an explicit trusted-host Rust integration, not automatic YAML startup, ordinary session restore or arbitrary Rust-future serialization.

Development error migration: exhaustive `TaskRunStorageError` matches handle `ClockDiscontinuity` separately from `AdmissionExpired`. A backward UTC sample or recaptured host timestamp is a clock-contract failure, not exhaustion of the reconstruction budget. Before claim it preserves the acknowledged pause; after claim it retains charged Running custody and cannot authorize automatic replay. The runtime does not clamp time to checkpoint timestamps or renew original deadlines. Real budget/expiry exhaustion still reports `AdmissionExpired`.

1. While the original runner still owns an acknowledged safe `Paused` task, call `checkpoint_for_restore(run_id, expected_revision, host)`. It captures complete current nodes, declared/generated topology, actual bindings and original bounds without invoking providers, tools or registration hooks. The returned revision, not the previous pause revision, identifies the prepared restart point.
2. Supply an explicit `TaskRestoreRequest`, the same compatible scoped task store and an installed `Arc<dyn TaskRestoreHost>` to `AutonomyRunner::restore_paused`. It validates configuration, required references, current host ceilings and trusted clocks before an exact revision claim, then stages fresh unpublished objects and acknowledges a reconstructed `Paused` revision without consuming a response.
3. Obtain `restoration_review(run_id)` and call `approve_restoration` with that exact review ID and revision. Restoration review is separate from the original pending approval, question, controller review or external signal. Only then submit the currently advertised pending response through the existing APIs; stale pre-reconstruction request IDs are rejected.

The host reconstructs actual providers, tools, domain adapters and opaque behavioral state through versioned non-secret bindings. Matching names are insufficient: it must cover model/request configuration, hidden service state, implementation identity, sharing relationships and native-history destinations. Framework checks additionally cover prepared runtime configuration, actual topology/provider handles, fresh owners and generations. Staged registries must be empty; validated saved membership, naming and capacity are installed silently rather than regenerated. Domain work and lifecycle hooks must not run during construction. The host remains trusted, not sandboxed.

Completed effects, exact native history, skill/action positions, controller validation/replan/stage/capacity positions, todos, proof, cumulative usage and original expiry remain preserved. Existing drivers execute only unfinished work. Reconstructed validation still requires current host-target freshness and journal acknowledgement; saved passing evidence does not certify a changed external world. Original request/check/composition bounds are not renewed, and a timeout disposition cannot approve another gate. Current policy, scope, implementation, availability, locks and final admission remain authoritative after review.

Capture, reconstruction and review each conservatively precharge a positive host-selected work bound and share one fixed monotonic deadline with backend publication. A safe human pause does not consume active allowance, but whole-run expiry continues. The host must certify cross-process UTC continuity in the same timestamp domain as the backend and reject uncertain clocks; extrapolating an independent old UTC epoch with a monotonic timer is not sufficient. Lower current ceilings reject before claim rather than silently resetting or narrowing stored allowances. Dropping preclaim capture preserves the original live pause; once a claim starts, errors, cancellation and lost acknowledgements retain truthful custody rather than automatically retrying.

`TaskRunStore` adds default-unsupported reconstruction capability methods. A custom store opting in must forward the opaque checkpoint's exact guarded mutation, validate its predecessor and preserve live final admission inside its conditional backend write. `ScopedTaskRunStore` supports this boundary over memory and SQLite. Retained memory can reconstruct fresh objects only while the same store remains alive; process restart requires a durable compatible backend such as SQLite. A lost final ACK can leave committed Paused data without returning a successful runner, so inspect the authoritative revision rather than replaying a guessed attempt.

Compatibility: restoration manifest version 1 is explicitly prepared and bound to one task revision. Package, storage-envelope and runtime payload versions are unchanged; this does not promise that older runtimes can execute new private adapters. A legacy checkpoint without a prepared manifest cannot be made executable by assigning an identity or deserializing it. Preparation requires its original live safe pause. Opted-in Rust persistence prepares these images automatically; reviewed recovery can repair an unchanged interrupted preparation or settle known effects into an incomplete/cancelled outcome. There is no automatic executable migration, Running-owner takeover, arbitrary in-flight call-stack restoration or default production CLI/TUI new-process resume. Protect manifests and full task results as sensitive recovery data, not event/list payloads or logs.

### Persistence and reviewed recovery (development only)

The explicit Rust runner accepts the existing layered `persistence` profile. Install `runner.with_restore_host(host)` before enabling it. A present section defaults to enabled, `save_after_each_step` defaults to true, and `resume_on_start` defaults to false. Explicit `enabled: false` preserves ordinary execution and conflicts with either other flag set to true. Omitting or disabling persistence adds no post-pause storage read. Required admission, effect, pending, pause and terminal journals always remain enabled; disabling step snapshots never disables safety accounting.

Enabled persistence adds an optional full settled-step snapshot and prepares every returned safe pause through the existing guarded capture transaction. Children and owned maintenance settle first. Progress invalidates the old image, and a later acknowledged live pause receives a newly prepared image rather than reusing the old manifest. Capture failures report errors with the actual claim/ACK phase intact; a stored DTO alone is not a restart guarantee. Retained memory supports new objects only while that store lives; SQLite supports process reopening. File, Redis, Noop and namespaced task backends remain unsupported.

`AutonomyRunner::restore_on_start(TaskStartupRequest { config, ceilings, config_identity, profile }, store, host)` is an explicit host startup boundary, not a daemon or enabled YAML entry. It returns `None` for no compatible candidate, a reconstructed paused runner for exactly one, and an error for ambiguity. It excludes cancelled/owned/non-paused, expired, unsupported prepared images and other configuration/profile identities; it never picks the newest timestamp. Storage, schema/native integrity and uncertifiable clocks remain errors, not silently skipped rows. A host can return `false` from `TaskRestoreHost::compatible_manifest` for a supported classification of configuration incompatibility; the default retains strict manifest validation. Selection is followed by the same exact-revision claim, fresh owners, current authorization and separate restoration review as explicit reconstruction. No pending approval, question, external signal or controller decision is consumed.

Reviewed recovery is distinct from startup and completion:

```text
explicit authenticated host review of exact revision and actual stopped work
  -> quarantine abandoned claim (not proof of rollback)
  -> preserve/settle effect and interrupted-time accounting
  -> guarded recovery publication
  -> incomplete/cancelled outcome OR prepared pause
  -> explicit reconstruction and current admission OR typed cancellation cleanup
```

`AutonomyRunner::reconcile_recovery(store, run_id, decision, host)` requires an installed `TaskRecoveryHost` and a Rust-only `TaskRecoveryDecision`; neither is deserialized as authority from model output or checkpoint labels. The host authenticates the caller and certifies the exact snapshot, decision, real implementations, stopped external work and supplied time evidence. Its validation must be bounded, pure and non-reentrant because it is repeated inside backend admission. Process death, an owner token, abort, timeout or a cancellation receipt alone is not that evidence. Missing outcomes or revoked proof retain recovery protection.

| Disposition | Contract |
| --- | --- |
| `Stop` | At a known stored between-turns boundary, resolve every unfinished effect and child outcome explicitly and finish `Incomplete` (or `Cancelled` if cancellation was requested). Results remain audit-only; they cannot establish task completion. Full declared cost and write commitments remain conservative, without attempt refunds or replay. Already completed results are immutable. Unsupported pending/call-stack locations reject rather than being relabelled between-turns. |
| `ResumePrepared` | Repair only an unchanged prepared claim/construction/review interrupted before semantic execution. No new effect or child results may be supplied. Original pending work, completed prefix, usage and deadlines remain unchanged; fresh reconstruction review and pending decisions are still required. |
| `CancelPrepared` | For an explicitly cancelled unchanged prepared cursor, publish only a cancelled paused cleanup point. `restore_cancelled(request, store, restore_host)` then uses cleanup-only ownership and existing typed model/skill/state/controller/child cancellation. It closes pending native histories without invoking their bodies, consuming an approval or clearing cancellation. Original execution expiry cannot block known-safe cleanup. Ordinary resume/claim remains forbidden. |

Interrupted accounting uses a separate sampled-usage UTC anchor, not backend `updated_at`, which cancellation metadata can advance independently. Legacy open intervals are conservatively charged from their recorded start. An exact claim-only successor retains its own start timestamp; when later metadata makes that start ambiguous, `decision.interrupted_started_at` requires explicit host certification. `decision.stopped_at` may bound an actually stopped interval; otherwise the certified current time is charged. Interrupted construction charges elapsed excess over its already prepaid work budget, never the prepaid portion twice. Recovery itself prepays a positive bounded host budget. Invalid/backward clocks and arithmetic overflow reject, original expiry is never renewed, and an exhausted allowance cannot resume semantic work. Cleanup may still settle and retire cancelled work.

Recovery publication preserves exact CAS and live backend admission over scoped memory/SQLite stores. Errors or dropped/lost acknowledgements do not certify whether a write committed: inspect the authoritative revision and obtain a new reviewed decision where needed; do not automatically retry the old decision. Cancellation racing publication wins its revision. A committed cancelled cleanup point can be reconstructed again for cleanup after a lost publication ACK without granting execution. Terminal outcomes are immutable; the existing retained `acknowledge_recovery_release` separately retires old live protection after acknowledged reconciliation and cleanup, never a newer owner's runtime.

Migration: exhaustive development `TaskRunMutation` matches add `ReconcileRecovery`, `ClaimCleanup` and `CleanupCheckpoint`. Custom `TaskRunStore` implementations retain source-compatible defaults but must explicitly opt into reviewed recovery, preserve the opaque `TaskRecoveryCheckpoint`/`TaskReconstructionCheckpoint` and live guards, and implement cleanup claims where supported. `inspect_candidate` can inspect valid old configuration data for filtering while ordinary `load` stays strictly configuration-bound. New private clock/work adapters use version 1; package, storage-envelope and runtime payload versions are unchanged, not a promise that older runtimes execute newer private adapters. No universal future serializer, distributed lease service, automatic effect reconciliation or exactly-once external-effect guarantee is introduced.

### Completion, validation and progress (development only)

`ai_agents::autonomy` provides `CompletionGateEvaluator`, immutable versioned `AutonomyExtensions`, validator/progress traits, bounded `ValidationDriverState`, and lifecycle/checkpoint decision reducers. These host integration primitives are shared by explicit execution and automatic state/skill entry. Installed capability and profile preflight still reject unsupported guarantees before task effects.

Gates preserve `Pass`, `Fail`, and `Unknown`; only `Pass` proves completion. Negating missing, unavailable, partial, denied or stale observations does not create success. Evidence binds agent/run, objective revision, stage, selected check/attempt, adapter/version/configuration, mutation generation and optional host target revision. Duplicate captures do not manufacture counts, and conflicting copies are errors. Artifact checks consume explicit host or authorized-tool observations, never unrestricted filesystem probes. Gate display details redact strings by default.

Validation check IDs beginning with `runtime.` are reserved for framework journals and reject during configuration/binding, before runtime ownership or effects. Rename affected development check IDs and update their gate/progress references; ordinary adapter IDs are unaffected.

Register adapters before building with `AgentBuilder::autonomy_extensions`. Registration is consumed into an immutable runtime registry; selected profiles bind names, versions, configuration, dependencies and required capabilities before observation. `bind_for_agent` checks the actual runtime operation inventory and judge aliases without provider calls; optional host revision services can supply explicit revision observations through the lower-level binding API. Domain callbacks are trusted, bounded pure calculations and must request I/O rather than performing hidden I/O themselves; the framework does not sandbox custom Rust code.

The built-in command, diagnostics, judge, evidence and artifact validators use the resumable observation driver. Its default maxima are 32 callback rounds and 16 observations per round. YAML checks can set `max_evaluation_rounds` and `max_observations_per_round` within those maxima; execution intersects them with host bounds. A declared validation timeout survives checkpoint restoration. Framework-owned built-in adapter config is strict, while custom adapters validate their own opaque config maps.

Known `Pass` and `Fail` results may supply usable metrics; `Unknown`, unavailable and stale metrics are audit-only. They cannot satisfy an absence assertion or an explicit progress signal. A known observation with a missing path may still satisfy `exists: false`. Frozen `AdapterDescriptor::checks` inputs and the configuration-selected `builtin.evidence` input bind to exact upstream result fingerprints, including their own dependencies. Missing, replaced, unknown or stale inputs cannot be relabelled by a derived result. `BoundAutonomyProfile::prepare_scope` installs the frozen dependency inventory as well as configuration bindings. Dependency traversal is bounded and memoized; declaring inputs does not schedule a recursive check execution.

Typed command, diagnostics and judge validation leaves apply the same dependency checks as named validator results. The bound progress registry stamps validated upstream proof references onto its observations. Those references survive `ProgressObservation::scoped`, so later `progress_path` evaluation rechecks transitive freshness. Custom validator and progress adapters use the same contract without registering a single shared target or adding a domain-specific gate variant. Callbacks remain trusted Rust integrations; manually forged evidence or hidden callback I/O is not sandboxed.

`TaskValidationJournal` persists requests, batch position and exact results under the existing task claim/revision before dispatch and before publishing evidence. Identical completed observations are reused within the attempt; changed requests, stalled cached reentry and uncertain in-flight recovery are rejected. Pending external waits require acknowledged issuance, frozen signal schemas and single-use request/event identities. Restored waits are re-acknowledged through the driver before consuming a signal. No polling daemon is created.

For executable same-process round trips, use `ValidationDriverState::restore_retained(checkpoint, &live_predecessor, check, scope, extensions)`. The checkpoint must exactly match its live predecessor; original attempt/binding/deadline and monotonic bounds are retained, uncertain in-flight work rejects, and publication requires a new journal acknowledgement. `restore` and direct deserialization validate/decode audit data but do not restore permission to `drive`; a raw-restored state cannot serve as the live predecessor. This does not reconstruct task owners, child implementations or executable restart.

Independent `RuntimeObservationExecutor` use retains a runtime mutation clock without synthesizing a task ledger. Actual non-read-only attempts invalidate earlier reads when invocation starts, even if later dropped, and again at finalization so intervening reads cannot reuse old proof. Nested managed hops share the observation clock, while a live runner continues using its authoritative run generation. Denied or non-invoked operations do not advance the clock.

A host can register a named validation binding for one exact canonical operation, arguments, profile/scope and approval requirement. YAML can select it but cannot create authority. A private execution-local permit replaces only the ordinary tool-grant predicate; runtime/state narrowing, availability, policy, HITL, modified-argument checks, resource locks and final generation/rate admission remain in the shared executor. This does not expose a general command tool to the model. Run/revision dimensions are carried in evidence. Under a live standalone controller, permits also match its current run/revision; independent host-driven observations retain their existing contract. The retained coordinator observation path reconstructs this permit against the current acknowledged revision on resume. Human modifications cannot change the fixed operation or arguments, even if the tool is otherwise granted. Supported nested/child participation uses the same frozen binding and exact suspended-operation contract; automatic task entry and public task surfaces remain separate.

`builtin.todo` progress reads the existing canonical run-bound store; alternate todo names require a registered binding to that same store, not a JSON-returning tool. Numeric signals retain observed high-water marks, including negative values, and change signals retain bounded fingerprints so oscillation does not count as fresh progress. Missing or unknown validation never fabricates a numeric zero. Stagnation and replanning are bounded and persisted independently from ordinary continuation. Selected replans use the existing planner through the managed driver without resetting objective, mandatory stages or usage.

`ScopedTaskRunStore::apply_host_action` accepts explicit Rust-only stop/objective/limit decisions. The host must authenticate its caller; model responses and external signals are not control-plane authorization. Updates require safe boundaries and CAS, retain usage/effects/audit history, honor current host ceilings, and cannot revive terminal or expired runs. Objective revision invalidates old evidence and todo completion. Existing `ValidationCheck` Rust literals add the two optional driver-bound fields; `TodoRunBinding` literals add `objective_revision: 0` for an initial run. Older serialized todo bindings default that field to zero.

### Artifact and numeric telemetry observations (development only)

`builtin.artifact` has strict configuration `{path, host_binding?}` and requests the existing `file_info` tool through the managed validation driver. The path is required; a fixed host-validation binding is optional and grants only its exact operation. An acknowledged, successful, non-truncated observation with matching requested/executed/output paths supplies `artifact_exists`; a known missing artifact is `Fail`, while denied, malformed, unavailable or stale observations remain `Unknown`. There is no unrestricted filesystem probe. File existence alone does not validate report schema or content; combine it with an appropriate validator or judge.

For non-filesystem artifacts, an installed `AutonomyValidator` can implement the additive pure `artifacts(&ValidationResult) -> Result<Vec<ArtifactObservation>>` projection. Existing implementations keep the default empty projection. It runs only on an acknowledged result bound to the actual selected check. The framework attaches that exact result's fingerprint and transitive freshness dependencies; a later target mutation, attempt or configuration replacement invalidates its artifact proof. Manual SDK collectors call `collect_validation` followed by `collect_artifacts(&driver, &bound_check)`. Neither method restores execution authority from serialized evidence, and custom Rust callbacks must not perform hidden I/O.

When participating runtimes have enabled observability collectors, the explicit runner journals bounded numeric completed-event facts for the current run. It uses an internal opaque `task_run` label, not ordinary session history, and seals the prefix before safe-pause publication. Actual child collectors join the coordinating journal; unused catalogue nodes do not. Separate fresh collectors after reconstruction continue from the acknowledged numeric prefix without replaying events or resetting it. Missing collectors, disabled collection categories, queue loss or rolling eviction make aggregate evidence incomplete. Observed LLM/tool call coverage must also match the acknowledged attempt ledger: an enabled collector alone, a collector attached after provider construction, or an uninstrumented custom path cannot prove measured zero or an upper bound. A saturated 4,096-event numeric journal freezes as explicitly incomplete rather than blocking cancellation/cleanup or fabricating successful zero totals. Required execution/recovery data is never truncated by this telemetry policy.

The `observability` gate can match numeric paths under `summary`, `token_breakdown` and `cost_breakdown`, for example `summary.total_llm_calls` and `summary.router_calls`. Purpose counters retain their actual classification. Unmeasured tokens, disabled token counting, unpriced calls and unknown-price zero placeholders do not prove measured zero cost or usage: unavailable metric paths remain `Unknown`, including absence predicates and negation. Estimates retain their reporting meaning and never replace the authoritative invocation ledger, host-audited price reservations or final admission. No prompt, tool output, actor data or raw event payload is copied into the numeric journal. This is gate input for the existing runner, not the complete public task-event/reporting surface.

Preparation, reconstruction and review preflight the full serialized work envelopes, including the intermediate running journal's digest overhead, before claiming a usable pause. Predictable size failures preserve the original pause; unpredictable post-claim failures still retain custody. Registry-list ordering is normalized only for semantic image comparison; typed child invocation order, exact native call groups, completed results and configuration remain unchanged.

### Guarded standalone execution (development only)

`AutonomyRunner::try_new` accepts a prepared `Arc<RuntimeAgent>`, task configuration, host ceilings, an explicit `Arc<dyn TaskRunStore>`, and a prepared-configuration identity. `run(objective, selected_profile)` executes internal runtime turns rather than repeatedly calling public chat. This is a restricted development API, not the published 1.1.1 feature or the complete planned task contract.

The runtime is reserved across turns, validation and finalization. Unrelated chat/streams, direct tools, reset/restore and public actor/context mutations are rejected while reserved. The first accepted objective is committed once; controller instructions are ephemeral provider context. Task-derived assistant/tool messages retain optional `ChatMessage::provenance` in snapshots, so actor facts and relationships exclude them without altering native provider history. Response hooks receive an `autonomy_input` metadata label. Actor deletion, ordinary session rebinding through `save_session`, and `set_session_metadata` also reject while the live runtime is reserved, including a safe pause whose durable writer token was released. `set_session_metadata` now returns `Result<()>`; propagate or handle that result. Independent `save_to` snapshot export does not rebind the runtime and is not executable task persistence. Ordinary session deletion/expiry administration and raw host storage handles remain separate host responsibilities, not implicit task work or sandboxed access.

Provider handles are wrapped before builder consumers capture them. Managed invocations, retries/fallback, supported judge/replan observations and tool sub-attempts share acknowledged conditional reservations. Exact-cap evidence may complete without another call; limit summaries are deterministic. Live active time uses a monotonic allowance and original expiry, and `FinalCheckpoint` checks deadlines in the backend write boundary. These are cooperative deadlines, not guaranteed physical termination. Acknowledged non-invocation refunds its original typed slot and targets that no other admitted or completed effect uses. Unknown dispatch, partial effects and failed acknowledgements retain conservative reservations.

`request_cancel(run_id)` applies to this runner's retained live owner. For running work it returns a stored-request receipt, not cleanup acknowledgement. For a known-safe retained foreground pause it awaits the same acknowledgement as `cancel_paused(run_id, expected_revision)` before releasing protection. Dropped or uncertain work retains reservations and can produce `RecoveryRequired`; the associated runtime remains reserved and uncertain tool effects retain framework resource protection. An abandoned Running or RecoveryRequired task is not eligible for the separate safe-pause reconstruction API. `acknowledge_recovery_release(run_id, expected_revision)` provides cleanup-only release for an abandoned retained execution after the host has reconciled every uncertain outcome in an acknowledged checkpoint and asserts that external work has stopped. It rejects unresolved reservations, pending work, still-active children and owned maintenance tails. Nonterminal cleanup holds a conditional storage claim; terminal cleanup preserves the immutable status, including a completed checkpoint whose live cleanup was interrupted. This does not perform reconciliation, reconstruct a runtime, roll back effects or authorize replay. Never interpret a request receipt, dropped future, ordinary session restore or text response as successful task cleanup.

The standalone path supports model-led turns, simple lifecycle stages, canonical todo progress admission, supported tool-producing validators with retained approval/question continuation, judges, explicit external waits, bounded managed replanning, and one-time objective template resolution. Validation requests retain their immutable attempt binding while each observation and published result captures managed mutation freshness; a read invalidated by a later mutation cannot establish a fresh pass. Initial ordinary skill routing and reasoning/planning use the existing runtime; controller-only continuations skip script routing and plan reconstruction rather than replaying the initial work. Selected scripted skills retain same-process tool-step continuation; explicit `run_skill` additionally admits the skill strategy and hands a completed initial segment to the same controller. Ordinary resume still needs live owners; reconstruction in a fresh runtime uses the explicit prepared safe-pause API. A restricted foreground model-tool batch can now pause before invocation and resume from its exact native/prompt batch state. `require_todos` requires a nonempty canonical run list before selected work and completion; `require_active_todo` requires exactly one active item before selected work, not at finalization. `todos_done` remains a separately selected completion gate. `max_open_todos` is checked atomically by the canonical store. Otherwise-authorized todo repair and privately bound host validation are exempt only from the progress predicate.

Hard priced execution requires each reachable frozen provider to expose `priced_capability_identity()` and each actual completion/native/stream request to return a matching `request_cost_bound(&ProviderRequest)`. The wrapper passes the exact borrowed messages, effective configuration, native tool request and request class to the implementation that dispatches them. Checked micro-USD reservations and pinned provider/pricing/request identities are acknowledged before dispatch. `settle_request_cost` can release only trustworthy matching usage; mismatches retain recovery protection, and missing, partial or retry-only usage retains the full declared bound. Observability estimates and an identity string alone are not this capability.

The development `DeclaredPricing` contract is available at `ai_agents::DeclaredPricing` and `ai_agents::llm::providers::DeclaredPricing`. Install it with `UnifiedLLMProvider::with_declared_pricing`, or declare the same `declared_pricing` object in an LLM YAML configuration before consumers capture provider handles. The trusted host must audit the exact model/endpoint, server-enforced input/output token ceilings including reasoning, maximum rates, request fees and complete-usage reporting semantics. The adapter reserves the full token ceilings rather than guessing prompt tokens, rounds integer micro-USD costs upward and covers the declared total internal-attempt bound. No prices or model limits are fetched or assumed from model names. Without such a verified host contract, hard-priced dispatch remains unsupported. See [Provider pricing support](@/docs/providers.md#declared-priced-task-capabilities-development-only) for the complete provider/request-class inventory and limitations.

Pricing declarations in model-generated child specifications are not host authority. Hard-priced spawning requires the spawner's authoritative shared provider registry; local child provider construction is rejected before credential lookup. Shared mode ignores child provider/pricing declarations, while explicitly host-prepared registered runtimes retain their real bindings.

Built-in mock requests remain explicitly free. Supported paid adapters reject arbitrary `extra_body`, native search/service options, unknown provider extras, output/reasoning limits outside the audited contract and unbounded internal retries before reservation. Ordinary calls without a hard-priced task retain their existing configuration behavior. Stream EOF/error is acknowledged under the original captured scope. The additive unsupported-default `settle_stream_request_cost` accepts only complete final usage after normal EOF; the first-party Google adapter supports this for an explicit complete terminal billing report. Early cumulative, partial, missing, errored or dropped stream usage never releases the upper bound. Transparent provider wrappers forward this capability; existing custom providers keep the conservative default. A dropped stream or lost settlement acknowledgement cannot become a successful result, an unused request or permission to replay.

A declared-write cap requires an implementation-owned footprint from final approved arguments. `Tool::prepare_write_footprint` combines that footprint with an optional live `ToolWriteAdmission::binding`; its default delegates to the existing `declared_write_footprint`. Mutable host adapters prepare both from the same actual captured implementation, which stays alive through storage admission, final checks, invocation and settlement. `ToolExecutionContext` literals add `admitted_write_footprint: None`; the executor fills this inert receipt for each admitted attempt. Transparent wrappers forward both preparation methods. Restored receipts are never live execution authority.

File mutations, patch, copy/move/delete, legacy file operations, discovery, local utility tools, todo and sleep have implementation-owned footprints. Atomic file replacement/unlink identifies the unfollowed entry, while legacy content writes identify the referent; recursive replacement covers old descendants and incoming entries with bounded enumeration. Pure computation, proven previews and admitted read-only inspection consume no logical write targets. Git inspection disables optional index writes and configured external-diff/textconv/fsmonitor execution. Default web fetch is locally write-free; injected transports/resolvers must explicitly attest the same contract. HTTP GET/HEAD have empty local footprints, while arbitrary HTTP mutations remain unsupported under this cap.

Command, diagnostics, question and search host traits have additive default-unsupported `declared_write_footprint` capabilities over the exact normalized request. Static/unavailable fixtures have explicit empty footprints; the process command runner remains unsupported because an argv allowlist does not enforce a complete write set. Custom runner/build exclusions require host-enforced coverage, never a tool-name exemption. Resumed built-in `ask_user` answers use `prepare_question_write_footprint` and never invoke or charge the ordinary question handler; custom response executors default to unsupported. Framework messaging/generation/retirement account hook tails separately, while actual child tools still reserve targets in the same run. These are trusted-host logical-target guarantees, not a sandbox, a count of every physical database write or permission to bypass policy/HITL.

Conforming blocking child turns now use the existing chat/actor/registry entry points under child-local gates and long-lived reservations. They share the coordinating ledger and cancellation/time checks; child snapshots and immutable outcomes are acknowledged before aggregation. Completed child runtimes remain busy until parent terminal acknowledgement. Native call IDs are preserved, while task evidence adds runtime/operation attribution. Borrowed orchestration providers are accounted without replacing captured handles or double-charging managed providers. Canonical todo tools select the coordinating task store through execution-local authority; custom todo implementations must conform or share that store. Existing required/optional failure policy is preserved through nested branches. Approval waits are cancellation/deadline bounded, but this bounded failure path is **not** a resumable pause.

An explicitly invoked task can pause inside declared delegated, concurrent, pipeline, handoff and group-chat states when a registered child's model-tool batch requires approval or a question answer. The parent retains prepared input/context, actor/source classification, stable invocation identities and its dispatch definition. Concurrent dispatch closes new child admission when work parks, drains already admitted children and checkpoints ordered settled results before exposing the pause. Repeated slots for the same actual runtime remain distinct operations and wait through predecessor outcome acknowledgement, not only root-gate release. An already parked operation precedes unstarted slots on resume regardless of spawned-task polling order; different runtimes retain concurrency and final results remain in declaration order. Multiple parked leaf requests are retained; `TaskRun::pending` exposes one coordinating request at a time, and each subsequent pause receives a new coordinating ID and revision. Always answer the currently exposed request rather than a saved leaf ID.

Pipeline continuation retains the stage index, exact prepared input, prior outputs, actor sender chain and original timeout. Handoff retains the current specialist, input, earlier responses and acknowledged routing decisions. Group chat retains chosen or randomized speaker order, transcript and termination position; maker-checker and debate keep their own draft/review/argument/synthesis positions. Tested nested parent dispatches resume without rerunning input preparation or completed children. A long-lived child owner remains pointer-qualified while exact actor attribution binds to each invocation, allowing legitimate pipeline sender changes without permitting an answer to change that invocation's actor.

Composition catalogues also retain private guards for the actual registered instances, including unstarted slots. Final child enrollment verifies the captured gate while holding the registry read guard, so the same display name cannot substitute a different implementation. `AgentRegistry::remove` returns `None` for a pinned target or a runtime with task ownership/active admission or mutation; topology reconciliation rejects removal of those entries before changing the map. A pin protects registration, not permission to invoke the child. Inherited blocking SDK messages and orchestration pin actual registrations across hooks, auxiliary calls and root-gate waits, including unstarted stages, manager/synthesizer agents and targets registered after initial catalogue capture. Ordinary non-task calls keep their existing behavior. Known terminal or explicitly reconciled cleanup retires the old owner's pins independently of a newer active execution, and retired catalogues cannot be republished by stale work. Child release records each cleared owner before waiting for another gate, so an interrupted retry does not revisit a runtime already handed to a newer owner.

Ordinary `AgentRegistry::broadcast` and `broadcast_with_actor_context` keep their blocking contracts and stable registry-ID ordering. The development `AgentRegistry::broadcast_task(self: &Arc<Self>, from, message, actor_context) -> Result<ToolResult>` is an additive typed bridge for a task-aware tool's `execute_task` implementation. When the shared executor supplies a live message permit to a tool implementing `supports_task_messages`, it freezes and pins the recipient catalogue, retains distinct operations and ordered results, drains admitted siblings and propagates `AgentError::TaskSuspended` rather than formatting it as recipient failure. Reentry invokes only unfinished recipients under the original deadline and captured required/optional policy. Its JSON output is an ordered array of `agent_id`, `response`, `success` rows or `agent_id`, `error`, `success` rows. Return the typed result unchanged so first dispatch and continuation use the same format. Without that permit the method is a blocking typed-output adapter; it cannot make an arbitrary host callback resumable. No new built-in broadcast tool or implicit grant is registered.

Generated children participate through existing messaging, selected routing, pipeline and concurrent algorithms under the coordinating run. Creation, registration and accounting do not repeat across child question pauses or cancellation, and the same actual child remains protected until acknowledged cleanup. Generation, repair, capacity and duplicate-registration failures retain consumed provider charges; a failed registration does not replace the existing instance or leak its temporary capacity reservation. The existing generator permits one repair after construction failure, including a capacity error. Dropping generation or a post-registration hook keeps uncertain task effects protected: committed registry membership is not rolled back, and a fresh run cannot automatically replay creation. Memory/SQLite terminal write rejection, lost acknowledgement and dropped acknowledgement preserve actual child ownership as well as registration pins. Retained continuation alone does not reconstruct a generated topology; explicit prepared safe-pause reconstruction is a separate host boundary. Active nested spawning remains unavailable. Hard-priced generated children use the authoritative shared host provider registry and the supported adapters described above; child-authored provider or pricing declarations cannot install fresh billing authority. Broadcast tests cover signed history, repeated catalogues and declared-parent nesting, optional/required failure, stale/wrong replies, original expiry, cancellation/stop, interrupted finalization and memory/SQLite final-pause write rejection or lost/dropped acknowledgement.

Ordinary continuation requires a retained runner and live owners; explicit reconstruction creates new live bindings from a separately prepared safe checkpoint. Original composition deadlines narrow shared provider/tool admission and survive pauses; an expired composition or a rejected leaf closes remaining known-safe parked histories without invoking pending work or aggregation. Uncertain dispatched work still retains recovery protection. Cancellation selects the actual pending incoming operation, not an earlier invocation or a group-chat catalogue slot. A coordinating foreground batch can also pause after completed child work while retaining child owners and operation ordinals. Checkpoint ancestry is bounded to 32 frames; this is not arbitrary graph execution. Fresh-runtime topology reconstruction uses the separate prepared-checkpoint API.

For foreground model-tool batches, set `hitl.on_approval_required` to `pause_run` in the explicitly supplied Rust task profile. `run()` returns `Paused` only after checkpoint acknowledgement and durable claim release; the associated live runtime and canonical todos remain reserved. `TaskRun::pending` exposes the acknowledged request and may contain sensitive review data. Keep the runner alive and call `resume(run_id, expected_revision, TaskResumeInput::Approval { request_id, result })`. Request IDs, revision, reviewed action, live retained state and exact history are checked before claim. Modified arguments still pass the current shared executor's policy, scope, locks and final admission. Approval request expiry and whole-run expiry are preserved; late approvals cannot execute. An explicit expired `Timeout` response follows the configured timeout policy. Rejection closes unresolved sibling results without another model call. Dropping the retained runner abandons its reservation rather than silently authorizing another owner.

For a foreground model `ask_user` call without a default, set `hitl.on_user_question` to `pause_run` in the explicit Rust task profile. Implementations opt into this capability through `Tool::task_question`; registration or interaction metadata alone is insufficient. Parking happens before resource locks and rate/task invocation admission. Resume with `TaskResumeInput::UserAnswer { request_id, answer }`, where the built-in answer has the `QuestionResponse` shape, for example `{"answered":true,"selected":["yes"]}`. Unknown choices, duplicate choices, excessive single selections and disallowed free text fail before claim. Answers bind to the actual parked arguments, including approval modifications, and still pass current shared final authorization. Use `TaskResumeInput::QuestionTimeout { request_id }` only after the original deadline; timeout ends the run as incomplete without polling a handler or another model. Ordinary chat handlers and questions with declared defaults retain their existing behavior. Tool wrappers must forward `task_question`, `task_question_result` and `execute_task_question`, preserving instrumentation at invocation rather than during pure answer validation.

Retained tool responses additionally keep the original live reviewed implementation and control/state generations through same-process pauses. A changed binding cannot become fresh approval merely because the resumed executor's current gate disappeared. Original approval/question UTC and monotonic bounds restrict response-dependent preparation, including managed localization calls, and actual tool admission after acknowledgement waits. A detected invalid retained response stops further task work rather than permitting another model response to fabricate completion; its development stop reason is `retained_response_invalid`. Once an effect has actually been invoked, later request expiry does not retroactively cancel or refund it. Ordinary policy, run, check and composition bounds remain independent restrictions.

An explicit expired approval timeout uses the actual pending leaf's configured policy, including validator-child observations whose root has a different policy. A legitimate timeout disposition remains privately bound to that original request across another pre-body approval or question gate on the same operation; it does not approve the newer request, a fallback hop or another tool call. Single-use response consumption and acknowledged cleanup remain required. Live review bindings are retained only while reachable from executable paused state and prior-gate receipts, and cleared on acknowledged owner release.

Checkpoint processing reuses decoded data only within the same immutable validation call and validates child runtime/request references without copying unrelated sibling recovery state. Every storage boundary still performs its complete scope, schema, successor, CAS and receipt checks; fresh owned reads are not replaced by a cross-revision cache. Exact compact-JSON byte counting preserves the inclusive checkpoint-size limit, including UTF-8 and escaping, without allocating another full serialized buffer. Receipt comparison covers every envelope field and the complete payload, but does not rerun final deadline or live evidence admission after an acknowledged commit. Original deadlines, resource limits, mutation ordering and ownership rules are unchanged.

Command completion evidence distinguishes the requested validation from the actual executed operation. Approved replacements retain their executor records, but a replacement command or changed cwd/argv/environment cannot prove the original `command_exit` gate. A fixed host validation binding still prohibits changing its operation or arguments. Selected scripts likewise retain their original state generation through prompt/tool dispatch, journal acknowledgements, resumed-result adoption, segment credit and response reflection/output. A trusted raw state-handle change invalidates the old cursor before further effects or usable success; it does not roll back earlier effects or reset their accounting. Deliberately authorized state transitions after successful skill response publication remain a separate lifecycle boundary.

This path preserves iteration/reasoning state, completed out-of-order results and current-root signed exchange expectations. It does not repeat user/assistant commits, completed calls or initial routing. Its resume authority is the retained same-process owner, not a serialized receipt. SQLite can store checkpoints, but persistence by itself does not install restart reconstruction. Multi-message checkpoints add a private batch coordinator and deferred-operation dependencies; older runtimes are not guaranteed to decode or execute these development payloads. The workspace and storage envelope versions are unchanged, which does not establish a restart or migration guarantee. Original-response binding adds private retained receipt identity, not a public constructor/response-shape change or executable migration from an older runtime.

Model-requested `send_agent_message` and `route_to_agent` calls now support retained same-process multi-call child approval/question continuation in the explicit Rust task path. The parent keeps its original admitted attempt, approved arguments, tool implementation, native batch and child operation. Routing retains the selected agent and reason without repeating LLM selection or advancing round-robin again; messaging does not repeat sender hooks. Parent approval may precede a separate child pause, and tested declared delegation can contain a model-message parent. Message coordination does not hold an unbound mutation lock needed by descendants; child tools still acquire ordinary locks. Original operation deadlines and cooperative live cancellation cover resumed child callbacks, and unacknowledged interruption requires recovery rather than safe cleanup; synchronous non-preemptible host code is still not physically bounded.

`TaskEffectState::Suspended` distinguishes acknowledged message continuation from an uncertain dispatched effect. The consumed attempt and outstanding commitments remain charged or reserved; resume marks that same attempt dispatched without a fresh rate or task slot. It cannot be refunded as proven non-invocation, including after resume. A paused marker alone is not authority: live owners, actual pinned targets and the retained tool implementation are still required. A parent model batch now retains distinct invocation frames for multiple pending send/selected-route calls. A coordinating batch owns the ordered calls, completed out-of-order results and contiguous native result prefix; one currently exposed leaf request is answered at a time. Re-pauses use new coordinating IDs/revisions without replaying completed siblings. Repeated calls to the same actual child retain distinct incoming operations but wait for its preceding operation to finish before starting another root history. Sender hooks and completed route selection still execute only once. Safe cancellation closes the current parent batch and each child history once, including a nested parent that has advanced to a local approval. Supported parents within the same retained run can now share a target: later invocations preserve their incoming operations and wait outside publication locks for every captured earlier overlapping dispatch to retire or expose its actual acknowledged blockers. Shared leaf responses are deduplicated, competing successors are admitted serially, and ready dependency-only cursors are driven before exposing another pause. Held ancestor gates are rejected before waiting. Frame-qualified dependency traversal is separate from pointer-qualified runtime execution ancestry. Different registered implementations cannot share one task runtime identity; aliases of the same actual runtime remain distinct invocation slots. Declared and model consumers now share this dependency contract, including dynamic group-call edges, root model-batch validation, multi-target consumers with several predecessors, and a pipeline blocked on a different slot from the queued target. An ancestor's future catalogue slot is rejected before waiting when its owning dispatch depends on the caller; a target need not itself be a held runtime gate for this wait to be circular. Dependency-only readiness cannot repeatedly re-drive a successor after its own child history has started. Acknowledged optional nested failure retires its exact declared dispatch before a later shared-target successor proceeds; completed outcomes remain immutable. Aggregation uses ordered successful child results only, and required failures or unmet minimum-success requirements stop before synthesis/voting. Explicit prepared topology reconstruction uses the separate host-authorized safe-pause API; supported managed origins share the existing admission/resource contract, while unsupported capability classes and default production restart remain explicit limitations.

A selected script can park at an approval or implementation-bound question tool step and resume its exact request, completed prompt/tool results and rendered template arguments without rerouting. A script cursor has its own execution identity, so repeated scripts do not reuse another invocation's journal. Supported same-process script continuation now has acceptance for repeated modified steps and questions, signed message-child re-pause, original approval/question deadlines, rejected-step finalization and safe paused cancellation. Completed initial script segments transition to ordinary controller continuation without rerunning script routing, prompts or effects. Memory/SQLite tests exercise initial pause, request-consumption, provider-settlement, re-pause and terminal acknowledgement rejection/loss/drop. Provider failure, actual managed-provider timeout and interrupted finalization retain protection rather than authorize replay. Automatic entry consumes this same cursor contract; supported prepared reconstruction and strategy scheduling are described in their dedicated sections. State transition action lists retain exit/commit/entry positions and generation bindings; focused tests cover exit and entry tool approval with one transition and response finalization. Fallback resumption retains the actual pending request, canonical ancestry, logical start time and retry index instead of repeating a failed original tool. A settled message failure now returns to its captured recovery position without repeating delivery, rate admission, task-attempt settlement or approval. A new retry consumes its own task attempt and deadline while retaining the existing logical request lifecycle; a different fallback is a separately authorized request. Original failure records finish before the next hop, and task evidence distinguishes each hop even though the original model call ID stays unchanged. Terminal retry-admission errors retain the settled original record; implementation result metadata cannot overwrite private hop identity. Additional focused coverage verifies modified skill arguments, scope revocation, nested skill entry actions and fallback questions without repeating completed work. Supported retained fallback/retry acceptance now includes exhaustion through repeated retry suspension, modified targets/messages, signed child history, original deadlines, rate-limit preservation, fixed host-operation recovery, cursor-tamper rejection, multi-hop/cycle evidence and memory/SQLite admission/settlement/final receipt faults. Retained state host controls also cover objective supersession, explicit Continue and required stage/plan review; automatic state entry consumes this same transition contract.

Supported coordinator validation observations can now park before approval or an implementation-bound question and resume the same request, remaining ordered observation batch and original check deadline. The retained controller preserves completed checks, cycle/schedule position, evidence, progress and the completed runtime response; it does not execute another model turn or repeat completed callbacks. `TaskEffectState::Suspended` also represents the logical observation reservation while its exact executor cursor is acknowledged. This does not charge a second logical observation, but a tool parked before real admission still consumes its normal attempt when first invoked. Completed mutations retain their original attempt identity, while fresh invocation/publication scope is captured separately. Actual Task-origin message observations now retain their tool invocation and child group without synthesizing a model turn. Their child approval/question replies, repeated child pauses and fixed host permits return to the same validation driver. Fallback observations preserve the failed original and return the actual pending hop, distinguishing immutable requested operation from effective executor record. Lower-level memory/SQLite journal tests preserve one logical charge across repeated host parks. Supported retained observation, nested child/fallback and external-wait paths now include host-target freshness, publication/control races and acknowledgement failure protection; arbitrary custom call stacks and executable reconstruction are not included.

A resumed observation child receives its answer only after the observation journal admission is acknowledged. Consuming the coordinating request does not itself prove child execution. If admission storage rejects the write, or commits it but loses acknowledgement, the root and child remain reserved rather than permitting automatic replay. A dropped resume during that acknowledgement has the same protection requirement; a persisted in-flight admission is not a safe pause. Memory and SQLite fault-injection tests cover both admission and later child re-park acknowledgement, including completed child history and positive release controls. Additional retained tests cover final-pause and re-wait receipts, dropped publication, check expiry, cancellation and host-control races, including a settled child followed by an external wait. This is same-process continuation, not restart or arbitrary graph reconstruction.

When `TaskRun::pending.kind` is `ExternalCondition`, the explicit host can deliver `TaskResumeInput::ExternalSignal(ExternalSignal { request_id, event_id, target_revision, payload })` to the retained runner at the expected revision. The runtime checks the frozen condition/schema, exact wait, target revision, single-use event, cancellation and lifetime before execution. Signal data is bounded to 64 KiB and its projected driver state to 256 KiB before claim. Issuance is re-acknowledged and acceptance is persisted before callback reentry. Signals are data, not approval or host-control authority. This API is the explicit delivery endpoint; schema registration does not install a polling daemon or external transport. In-memory and SQLite fixtures cover retained resume; neither supplies executable restart. An already-expired check rejects a response without claiming; the known-safe wait remains cancellable. Expiry or cancellation racing successful resume acknowledgements settles only when actual invocation quiescence is proven; failed/lost acknowledgements retain protection.

Model-requested `pipeline_process`, `concurrent_ask`, `group_discussion` and `handoff_conversation` now resume existing algorithm cursors under captured tool/registry/provider bindings. Completed stages, ordered sibling results, speakers/transcripts and handoff decisions are not replayed. Multiple pending child interactions and repeated catalogues retain focused execution and cancellation coverage. The required retained model-call consumer set now also includes typed broadcast, generated-child handoff and declared/model shared-target integration; this does not complete the overall task or composition/recovery contract.

Task clarification and confirmation now retain the manager's actual exchange and phase. Resume with `TaskResumeInput::UserAnswer { request_id, answer }`, using a nonempty JSON string for clarification rather than an `ask_user` tool response object. Already-completed detection, routing and input processing are not repeated; legitimately needed answer parsing and skill-field rechecks still run. Root and selected-skill questions retain processed context where already available and single user commit/finalization. Invalid answers reject before claim; an unrelated pre-existing ordinary-chat exchange is not adopted by a new task. An omitted task question pause policy stops the task instead of bypassing unresolved clarity. Abandonment/topic switch ends the existing task scope rather than silently changing its objective. Clarification exhaustion configured to stop ends the task before completion gates, even if the apology matches a response predicate. Escalation approval retains the original input and selected skill's processed context through a later skill-tool pause; rejection executes no pending skill. Missing human escalation support yields an incomplete task rather than treating transfer prose as completed work.

Retained clarification and confirmation questions use the existing runtime `hitl.default_timeout_seconds` when a HITL engine is configured; without it, no per-question deadline is invented and explicit stop/cancel remains available. The issued request keeps its UTC and monotonic deadline, and premature timeout or late answers reject before claim. An expired `TaskResumeInput::QuestionTimeout` retires only the bound exchange and returns `Incomplete` with `question_timeout` without parsing the answer or invoking the skill/model, including a nested confirmation leaf. Confirmation exhaustion cannot use best guess to bypass confirmation; explicit escalation approval remains a separate authority. Ordinary clarification best-guess policy still follows its configured semantic path. Replaced exchanges and changed state generations reject old responses before claim; an in-flight task generation change preserves a possibly newer manager exchange rather than performing blanket cleanup. Confirmation re-pause, skill-recheck exhaustion, semantic rejection/abandonment/topic switch, provider failure/timeout/drop and memory/SQLite acknowledgement failures now have focused acceptance. A claimed resume that fails during semantic work stays protected and cannot automatically retry completed parsing or detection; direct trusted-manager calls are not an executable task-recovery API. Ordinary chat behavior is unchanged.

State-transition approval now retains both successful and rejected caller positions before exit effects, including pre-tool native output, post-text/skill/composition output and deterministic pre-response input. Exit/commit/entry work remains distinct; rejection does not fabricate a transition or regenerate a discarded provider result. Nonempty transition/escalation approval modifications are rejected rather than interpreted as permission to retarget the reviewed operation. Deterministic pre-response optimization can participate without enabling unsupported speculative/background modes; accepted task redispatch reuses processed input. State binding is rechecked under the root gate and after awaited action preparation/results. A changed generation cannot dispatch a stale prompt, publish its context or execute the next old action. State-turn timeout, exit/entry question expiry, self/previous-target reentry, empty-reentry fallback, staged extractors, provider failure/deadline/drop, hooks and memory/SQLite receipt boundaries now have focused acceptance. Rejected pre-response extraction is not committed; normal post-response extraction in the unchanged state is distinct counted work. Existing state-action tolerate-failure behavior remains: a denied action is recorded as non-executed/unsuccessful, not fabricated as success. Stop/cancel/limit revision are verified before and after transition commit. Paused objective supersession/Continue and real stage/plan review now preserve the before/after-commit distinction. Superseding an exit pause does not force the old transition to commit; superseding an entry pause does not roll back its committed state. New-objective controller work uses the actual current state, not the retired action cursor.

The authenticated Rust host can use `AutonomyRunner::stop_paused(run_id, expected_revision, reason)` to retire known-safe pending work without approving it. A bounded audit reason and conditional claim precede cleanup, and terminal acknowledgement precedes owner release. Stop reports `Incomplete` with `user_stopped`, not successful completion or fabricated cancellation. Stop is not parsed from model output, answers or external signals. `AutonomyRunner::revise_limits_paused(run_id, expected_revision, limits, reason)` now revises numeric limits on a retained safe pause without consuming its pending request, resetting usage or invoking work. It returns a new paused revision; answer the same pending ID explicitly at that revision. Host ceilings and consumed/reserved usage remain binding. An authorized wall-limit change is measured from original admission and leaves question, approval, check and composition deadlines unchanged. Original UTC and monotonic admission bounds are rechecked inside the backend control write so a delayed extension cannot revive an expired run. Proven no-write refusal leaves the pause usable, while ambiguous writes, lost/dropped acknowledgement and invalid receipts retain recovery protection. Capacity-pause production is available for the supported retained controller/skill/model/validation paths described below; the standalone controller also retains a selected planner attempt when its actual admission needs more capacity.

Integrated runtime acceptance and automatic YAML state/skill entry are implemented for the documented managed paths; formal task evaluation and the complete task product remain unfinished. Explicit persistence/recovery still requires the actual host bindings described above. Conforming actor maintenance, run-owned background tails and speculative alternatives use the managed participation boundary below; this is not continuation of arbitrary custom host futures. The model-call consumer closure is restricted to actual captured same-process execution, not arbitrary custom host call stacks. The implemented composition paths do not complete the overall task execution contract. Explicit Rust task lifecycle streaming is described below; a typed task evaluation foundation is available with incomplete full acceptance coverage; the development CLI, plain REPL and TUI consume the existing controller. The in-memory and SQLite task stores can be supplied explicitly for the supported foreground path.

Fixed host validation recovery remains bound to the installed tool and exact arguments through acknowledged journal progress. Its permit is not transferred to another fallback, even if that fallback is ordinarily granted. Unrelated nested operations receive no parent fixed grant. Required-child stops and uncertain effects take precedence over recovery; dropping or timing out a dispatched task tool does not authorize fresh fallback work.

### Task evaluation inspection (development only)

`AutonomyRunner::task_inspection(run_id, revision)` returns `TaskRunInspection` from a validated acknowledged revision without claiming, executing, consuming a prompt or changing counters. It exposes exact typed run/scope/evaluation observations to trusted host code. The type has no serde implementation and its Debug output hides contents; it is not an export DTO or a fresh external-target/completion permit. Use `task_display` for bounded default-redacted snapshots and keep its key/revision aligned with the exact result. Do not parse `TaskHostDetails.lines` or private recovery adapters to recover structural evidence.

`TaskHostSession::scope_runner(actual_runner, future)` includes a supplied explicit SDK controller in the foreground catalogue only when that exact controller admits work. `observe_result(actual_runner, result)` revision-checks an actual acknowledged SDK result, including prepared reconstruction, before attaching its live handle. Neither method reconstructs authority from an ID or discovers arbitrary registry entries. Keep the original future polling through cancellation and retain protection after uncertain effects. `tool_record_eligible` exposes factual eligibility of an acknowledged managed tool observation; success, scope/target freshness and actual operation binding remain separate checks.

The facade's `ai_agents::eval` module re-exports the development `EvalHost`, typed task steps/assertions and safe reports. `EvalRunner::with_host` extends the existing harness with caller-owned domain integrations; it does not replace runtime execution. Configure before frozen capture, preserve budgets/instrumentation, and keep validation provider-free. Optional host preparation can supply an actual supported SDK runner/result; normal sequential task responses remain bound to its original key/revision/request. Default YAML/CLI production reconstruction is still unavailable.

Observation `EventType` now includes `TaskLifecycle`. Existing extensible purposes distinguish task/stage/validation/gate/continuation facts without replacing main/router invocation purposes. Custom aggregation dimensions `task_run`, `task_profile`, `task_stage` and `task_event` use the existing custom-dimension mechanism and default privacy treatment. Stage updates affect subsequent spans only: captured contexts, already-started calls and child scopes do not share mutable labels. Report metrics remain observations, not resource admission. See [development task evaluation](@/docs/evaluation.md#development-task-evaluation-foundation) for YAML, report schema 3, fixture safety and affected Rust literals.

### Managed maintenance and speculative participation (development only)

The explicit Rust runner admits conforming facts/relationship maintenance in existing serial, inline-parallel, background and mixed modes. It captures the same private run owner, provider bindings and invocation restrictions before spawning. A logical maintenance marker covers the whole body, including cache publication, persistence and callbacks; actual inner model/tool calls keep their own counted reservations. Queued and draining work remain owned until acknowledged settlement. Completion and safe-pause snapshots cannot race late maintenance admission. Cancelling a drain, dropping a future or observing an empty queue does not prove that external effects stopped.

```text
actual owner and captured implementations
  -> shared ledger and original invocation bounds
  -> maintenance body / provider calls / persistence / hooks
  -> acknowledged settlement
  -> sealed safe snapshot or eligible task completion
```

Before coordinating or child task ownership is installed, earlier ordinary maintenance drains outside the runtime gate and is checked again under that gate. This lets an earlier ordinary callback finish a legitimate reentrant turn without deadlocking task admission. That work is not retroactively charged to the incoming run; interrupted handoff leaves its original queued completions tracked. Spawned task callbacks retain child-operation attribution and failure policy, while an explicit new child dispatch policy takes precedence over an older captured parent context. Only acknowledged effect-free optional-child failure can be tolerated; uncertain effects or shared cancellation/resource stops still protect the whole run.

The initial conversational objective may update actor memory, but controller-only continuations do not create new actor interactions, extraction or relationship updates. Task-derived history stays excluded from later actor-memory projections. Runtime fact/relationship load and mutation APIs reject unrelated callers while the runtime is reserved. Raw component handles remain trusted host boundaries.

Custom `AgentHooks`, `AgentStorage`, `FactExtractor` and `RelationshipEvaluatorTrait` implementations add `task_providers() -> Option<Vec<Arc<dyn LLMProvider>>>`. The additive default is `None`: existing implementations still compile and ordinary chat remains usable, but explicit managed task participation is rejected until the implementation opts in. Return `Some(Vec::new())` only for a conforming consumer without hidden model work; otherwise enumerate every actual captured, already-managed provider handle. Pure extractors/evaluators return proposals rather than performing hidden mutations. Callbacks preserve caller scope and route managed effects through existing execution boundaries. Detached work, labels and provider names cannot supply conformity or execution authority. Stock no-op/logging/observability hooks and storage adapters declare their provider-free behavior; composites and namespaces forward the actual member capabilities. Frozen custom handles are validated, not silently replaced.

```rust
use ai_agents::{AgentHooks, LLMProvider};
use std::sync::Arc;

struct CallerOwnedAudit;
impl AgentHooks for CallerOwnedAudit {
    fn task_providers(&self) -> Option<Vec<Arc<dyn LLMProvider>>> {
        // Only the trait's no-op callbacks are used; there is no hidden model or detached work.
        Some(Vec::new())
    }
}
```

Hard-priced runs additionally require the captured consumers' real provider capabilities, followed by per-request bounds and matching settlement. Actor/session persistence is framework state, not an external model-selected tool grant; logical write-target limits do not bound every physical database or journal write. Custom external mutations must still use the shared authorized executor and applicable implementation-bound footprint. This host contract is not a sandbox for hidden Rust I/O.

Required task maintenance cannot silently use ordinary background-overflow `drop`. Pure auxiliary capacity refusal stops with `maintenance_capacity` and a truthful limit outcome (`fail` remains `Failed`); it does not fabricate an uncertain dispatch or park an arbitrary callback as if it were a model-loop continuation. Opaque callback futures are not serialized or automatically replayed. Possible writes, dropped effects and rejected/lost acknowledgements retain recovery protection. Ordinary automatic chat maintenance remains best-effort; failed fact eviction is now returned by `FactStore::add_facts` instead of advertising a successful trimmed cache.

Speculative task execution reuses the existing branch set. After winner selection, local admission closes for losing branches, dispatched calls settle under their original run bounds, and genuinely uninvoked alternatives remain distinguishable from completed failures or no-match decisions. Discarded actual attempts remain consumed and cannot support progress/completion. Transition review and exit/entry actions retain processed input, settled alternatives and used per-turn speculative slots. Rejection adopts completed alternatives or executes only proven unfinished work under the current parent authority. A winning tool-call draft seeds the existing exact task model-batch loop before approval/question suspension. Required human decisions remain separate, and state effects recheck cancellation and time after awaited preparation. This adds no scheduler or rollback engine; automatic entry and task event delivery consume these existing managed boundaries.

Deferred branch observations retain their identities through review and action pauses and publish the actual adopted/discarded/failed disposition once. A locally capacity-cancelled selection is unfinished even if its outcome arrived before the chosen transition; it is not a completed no-match. Dropped collection and terminal action paths retire observations without refunding dispatched attempts or claiming external cleanup. Entry-action interruption preserves a transition that already committed, unlike exit-action interruption before commit. A branch cancelled before any provider span still publishes its original correlatable identity under the existing privacy policy. Unreviewed loser drainage distinguishes proven-uninvoked cancellation from consumed failure or completed discard.

Managed completion/native requests and returned provider streams recheck live proof and clocks immediately before the actual first poll, during subsequent polling, after the provider returns and after settlement acknowledgement. A proven pre-dispatch refusal may refund its own attempt. A late acknowledged settlement remains settled and charged but cannot expose successful completion or EOF; interruption of dispatched work without acknowledgement remains protected. `TaskFinalAdmission::validate_dispatched()` is additive and defaults to `validate()`, so existing host guards remain conservative. The framework's branch-admission-only guard relaxes only closure of further branch invocations during existing-call settlement; it does not relax host proof, run cancellation or original bounds. This is cooperative control, not hard preemption or a new task stream API.

Tool and maintenance body polling samples live proof before the final cancellation and clock checks. Returned tool effects and successful maintenance bodies settle factually even when their original execution allowance or continuing proof no longer permits result adoption. The executor rejects ineligible tool results after recording their actual outcome, including successful resumed messages whose settlement acknowledgement arrives after the original message deadline; retries, fallback and task completion cannot adopt that result. The internal task-eligibility metadata is installed after tool-supplied metadata, is not an execution grant, and does not change the factual `executed` or `success` fields. Hooks and audit consumers must not treat an actual tool-success record alone as task completion. Known failure outcomes still close exact native histories; ordinary timeout skip/fallback remains unchanged. Admission-only approval/branch restrictions do not retroactively cancel already dispatched settlement, and lost/dropped acknowledgements still retain uncertainty.

Private development checkpoints now include inert branch-observation receipts alongside speculative selection/root bookkeeping and optional actor-cache/extraction progress. Existing package, storage-envelope and runtime payload versions are unchanged; older runtimes are not promised executable compatibility. Prepared restoration still requires explicit current host bindings, fresh review and a truly safe pause, not a saved arbitrary host future.

### Explicit skills and controller decisions (development only)

`AutonomyRunner::run_skill(objective, skill_id, selected_profile)` admits one actual loaded skill in the current state scope. It supplies a real skill scope for an explicit `execution.strategy: skill` or the inherited default skill strategy; ordinary `run` still rejects a skill-only profile. Scoped skill settings participate in existing profile resolution. Selection is explicit and makes no semantic router call. Script tools and prompts use the existing executor, cursor, policy and managed ledger. Skills invoked inside an existing task inherit its owner and accounting rather than recursively starting a run. This explicit API remains distinct from automatic state/skill startup; both use the same controller and execution boundaries.

Only the admitted coordinating skill can receive initial-segment completion credit. Completing its steps does not itself complete the objective: required stages, checks, child settlement and eligible gates still control success. Unmet gates continue through the normal controller without rerouting or replaying the script. Actual skill definitions and private owners remain bound through pauses, host controls and resume. Incompatible objective changes during a required partial segment reject; a completed segment is never rerun to establish a revised objective.

Stagnation, policy blockage and premature-finish questions are controller requests, not model-callable tool grants. They expose `TaskPendingKind::UserQuestion` with a frozen controller proposal in `reviewed_action`. Resume with `TaskResumeInput::UserAnswer { request_id, answer }`, where `answer` is a nonempty JSON string bounded to 64 KiB after JSON encoding. The answer becomes classified task guidance, not a new user objective, approval, limit change or successful validation. Use the explicit authenticated stop/cancel/objective/limit APIs for those actions. A configured human-question deadline supports `QuestionTimeout`; an early timeout or late answer rejects before claim. Completion precedes stagnation, and resume continues after the selected intervention instead of issuing it twice for one observation.

`on_limit.action: pause_run` produces `TaskPendingKind::Limit` and requires explicit `Continue`; `ask_user` produces a controller question, or a typed limit request at an exact tool-batch boundary, and requires a text `UserAnswer`. Both make no extra tool/provider call to formulate the request. Before consuming the request, resume checks that the exhausted resource now has capacity. `revise_limits_paused` changes only authenticated numeric limits within host ceilings, returns a new revision, preserves the pending ID and consumed/reserved resources, and does not itself resume work. An ordinary answer or Continue does not refill allowance. Actual admission still rechecks the final operation, pricing or write footprint after revision.

Supported capacity continuations preserve unfinished skill steps, already selected native/tool calls, model-loop iteration and accumulated-call state, or the exact validator batch/attempt. A required judge can wait after the final work call without repeating that work. Validator preflight is advisory; an actual pre-body refusal can become a safe wait only through a private live zero-admission witness and an acknowledged logical reservation transition. That transition refunds no actual invocation, priced commitment or write target. Already admitted, partially executed or uncertain work is not certified as non-invoked. Original run/check/operation bounds remain applicable; active human waiting is free only after safe checkpointing, while whole-run expiry continues. Actual expiry cannot be revived by a capacity response, and a successful last allowed call may still establish completion without another call.

Manual retained resume is the default controller delivery endpoint; no `QuestionHandler`, daemon or model-callable `ask_user` grant is required. `with_interaction_delivery(false)` declares that the host exposes no usable controller-request endpoint. Premature-finish and limit `on_unavailable: stop | fail` default to `stop`; stagnation requires its existing explicit unavailable fallback. To report confirmed loss after a request was issued, use `controller_unavailable_paused(run_id, expected_revision, request_id, reason)`. It retires only that current request using existing safe cleanup; silence and answer payloads cannot invoke it. Ordinary tool HITL configuration remains a separate contract.

Development Rust migration: exhaustive `PrematureFinishConfig` and `LimitConfig` literals add `on_unavailable: None`, and exhaustive `TaskPendingKind` matches handle `Limit`. `ValidationObservationExecutor::capacity_reason` has an additive default of no preflight wait. `ValidationDriverState::capacity_reason` distinguishes a capacity `Suspended` outcome from an external or executor interaction; preserve the live predecessor and original attempt when driving it again. JSON-only restore remains audit-only. No checkpoint-envelope version, package version, MSRV, new provider adapter or executable migration guarantee is introduced.

### Standalone controller scheduling and profiles (development only)

The explicit runner executes `model_led`, declared `lifecycle`, and admitted `skill` strategies through the same existing runtime. After a completed work turn, it runs declaration-ordered `each_cycle` observations, eligible stage-owned `stage_end` checks, and eligible `completion` checks before committing current progress, stage changes and one checkpoint decision. Candidate eligibility is not completion proof: deferred observations can still satisfy a currently failing tool-count/domain gate. Only immutable response predicates can rule out a candidate at this preflight; final gates always use the original expression and eligible committed evidence. An early task-gate pass cannot skip required stages, checks, pending interactions or child settlement.

Required task-completion checks do not block an earlier stage's progress. A stage's own required checks remain mandatory, while failed/unknown optional alternatives cannot block an independently verified passing `any` branch. Repair rewinds use the declared earlier stage and bounded fix history; they preserve completed effects and usage rather than replaying the initial skill. Normal stage and repair work does not spend premature-finish continuation allowance.

Pauses retain the exact schedule and completed check prefix, or the already selected dynamic planner check/attempt. Resume does not repeat the work turn, count another progress observation or select the same intervention twice. Managed replanning changes advisory next work, not the objective, grants or required gates. A priced planner refusal may wait for an explicit limit revision, then execute only that planner attempt; configured plan review follows its actual result. A completed child/group returning to a capacity-limited parent can use the configured controller capacity pause without replaying delivery or settled child work. Existing rejection and unsettled-child cleanup still take precedence.

Explicit task interaction profiles support `reject`, `auto_if_policy`, `pause_run` and `ask_each_time`. `reject` and `auto_if_policy` do not poll an approval handler for a request that requires human review; they reject that operation instead. Operations already allowed by the current policy remain available under `auto_if_policy`. Neither policy fabricates a user answer. `ask_each_time` uses retained questions/required approvals and, when selected for approval behavior, additionally reviews actual lifecycle boundaries. Explicit required stage/plan supervision cannot be auto-approved by a restrictive profile: the task stops `Incomplete`. Controller-origin questions remain separate task guidance, not ordinary tool grants or authenticated host controls.

The final checkpoint updates controller decision/progress/lifecycle views and remaining invocation, cost, write-target and time allowances from consumed/current commitments instead of retaining pre-turn values. These views remain advisory; shared live admission is authoritative. A successful last allowed call can complete without another turn, whereas an unresolved controller capacity stop follows `on_limit` (`fail` produces `Failed`; the default deterministic summary produces `LimitReached`). Objective processing remains one-time and controller-derived messages remain excluded from ordinary actor-memory extraction.

Scheduling itself does not install YAML hosting, arbitrary executable restart, task streaming, new participant capabilities or a published 1.1.1 feature; supported prepared restart uses the separate installed-host contract above. Persistence configuration uses the explicit installed restore host and startup boundary described above. Private controller checkpoints now distinguish scheduling and selected-replan positions; unchanged envelope versions do not promise executable migration from another runtime version. Existing public constructor and response shapes are unchanged by this scheduling update.

### Host-target evidence and external delivery (development only)

Install a bounded `HostRevisionSource` using `AutonomyExtensions::register_validation_target(check_id, target, source: Arc<dyn HostRevisionSource>)` before the registry is frozen. The source's `revision()` reads host-maintained state without I/O, providers, tools or storage reentry. Return a nonempty token of at most 512 bytes that changes on every relevant change, including target replacement; do not reuse tokens after an ABA change. Multiple checks naming one target must share the actual source object. Missing required check bindings and conflicting target sources reject. A required adapter declaring `requires_host_revision` needs its own check binding; an unrelated source or capability bit is insufficient. An unavailable optional capability produces `Unknown`, not successful evidence or an extra provider call. Revision-dependent progress requires an explicitly declared check dependency with a host target.

Selected targets are sampled before run admission, before observation/callback admission, across response and publication acknowledgements, before reviewed stage credit and at completion write admission. Immutable request/attempt identity is distinct from current observation/publication scope. Stale reads/results cannot be relabelled as fresh or expose an older passing attempt. Managed mutations conservatively invalidate earlier proof; these host bindings do not infer per-file write footprints. Target revisions are observations, not execution permits. Source failure or drift rejects executable resume before claim, leaving the known-safe pause available for cancellation, stop, objective retirement or confirmed delivery-loss cleanup. Drift after admission fails closed and may retain recovery protection. No completed effect or usage is reset.

`TaskRunMutation::FinalCheckpoint` carries an optional live `TaskFinalAdmission` evidence guard and `deadline: Option<Instant>`. Use `Some(original_deadline)` for a bounded publication and `None` only when no monotonic execution bound applies; the independent UTC `expires_at` still applies when present. An absent deadline never skips the evidence guard, owner, revision or cancellation checks. The memory backend checks admission under its mutation lock; SQLite also checks it before transaction commit. Custom stores/backends must preserve both optional time bounds and the live guard at their actual write boundary. Use `TaskRunSnapshot::transition_with_clock` and, for an additional precommit check, `TaskRunMutation::validate_final_deadline_with_clock`, supplying a fallible sampler from the backend's own UTC clock. These sample again after snapshot validation and live proof callbacks, including UTC-only expiry. The existing point-sample helpers remain available and account elapsed monotonic time in the supplied sample's clock domain; do not pass an already stale sample. Receipt validation does not rerun admission after a committed write. This is an observation at backend admission, not an atomic transaction with an externally mutable ticket, file or dataset. Hosts requiring stronger atomicity must synchronize those mutations themselves. A target change after a valid completion commit does not retroactively invalidate that completed run.

External waits use the same frozen target source in addition to their schema, request, event and objective binding. Merely matching the revision copied into an old signal is insufficient when the actual target changed. `ValidationDriveContext::deadline` is `Option<DateTime<Utc>>`: wrap an existing finite drive deadline in `Some(...)`; `None` adds no drive-wide UTC restriction. The driver still intersects configured check timeouts and inherited live invocation bounds, freezes its UTC-to-monotonic projection before awaited journal work, and rechecks cancellation and evidence at the original admission boundaries. With neither check nor inherited nor drive restrictions it creates no invocation deadline; structural callback/observation limits remain enforced. Original UTC and retained monotonic check bounds survive resume; callback, cached-result, wait and result-publication acknowledgement waits cannot extend the check. Restriction-only invocation scopes carry those bounds and live pinned host proof through the actual tool/provider admission acknowledgement and framework-spawned child tasks. Final pre-body checks reject expired or invalidated proof before a new effect; synchronous admission callbacks cannot bypass them by exhausting the allowance inside one poll. Managed provider streams retain their opening restrictions even outside the opening scope, and expiry after dispatch never certifies non-invocation or refunds consumed work. Already dispatched effects retain normal settlement and uncertainty protection. If a timeout wins before the non-invocation acknowledgement, reservations and live runtime protection remain conservative. SQLite connection release also rolls back transaction levels left by cancelled custom transaction initialization before allowing reuse; that backend cleanup is not an invocation receipt, an effect rollback guarantee or permission to release task ownership. A resumed active-interval marker uses acknowledged storage time rather than a fresh, possibly regressing UTC sample; consumed/interrupted time and monotonic active accounting are unchanged. Synchronous host callbacks are cooperative, not physically preempted. Invalid receipts and ambiguous storage failures retain live protection, even when a custom store reports an I/O or generic error rather than a persistence-specific variant. The workspace enables `serde_json/float_roundtrip` so finite timing/metric numbers retain exact identity through checkpoint transport rather than changing by a floating-point rounding step.

Manual `resume(..., ExternalSignal(...))` is itself a delivery endpoint: silence or the absence of a daemon is not channel loss. When the authenticated embedding host has confirmed delivery is unavailable, call `AutonomyRunner::external_unavailable_paused(run_id, expected_revision, request_id, reason)`. The current retained external request and a nonempty reason of at most 512 characters are required. The default `ExternalUnavailablePolicy::Stop` returns `Incomplete` with `external_unavailable`; select `Fail` using `AutonomyExtensions::set_condition_unavailable_policy(condition_id, policy)` after condition registration and before freeze. Neither policy approves the pending work or turns a validator into `Pass`. Cleanup uses existing conditional ownership and terminal acknowledgement, and remains possible when target freshness is unavailable. A normal signal payload cannot select this host-control behavior. Hosts authenticate/authorize callers; no transport, polling service or executable restart is installed.

### Retained objective changes and supervision (development only)

Use `AutonomyRunner::modify_objective_paused(run_id, expected_revision, objective, reason)` for an authenticated host objective change on a known-safe retained pause. It retires old pending approvals, questions and signals without executing them, preserves completed histories/effects and cumulative usage, and returns a new `Paused` revision with `TaskPendingKind::HostContinue`. The host must then explicitly call `resume(run_id, returned_revision, TaskResumeInput::Continue)`. Neither a question answer nor an external signal is Continue authority. Stale revisions and retired request IDs reject; modification itself makes no model/tool call. Objective templates are not rerendered on this explicit revision.

The canonical todo list moves atomically to a new objective binding; the old list and controller data remain in the protected audit history. Old handles reject, the open-item limit remains in force, and old completed todos/stage/validation results do not prove the new objective complete. Required stages need new eligible proof. Completed initial skills are not rerouted or replayed; an incompatible required partial skill rejects the change rather than being restarted or marked complete. Superseded validation attempts remain archived, not successful, and subsequent validation uses fresh counted attempts. Original lifetime, consumed/reserved resources and intervention limits remain binding.

The handoff conditionally claims before live cleanup, acknowledges safe cursor retirement, applies the host revision and checkpoints the new Continue boundary before making it usable. Invalid input, unsupported stores and certified no-write claim refusal preserve the old usable pause. Once cleanup or a write has occurred, a later refusal, invalid receipt, lost acknowledgement or dropped future retains recovery protection; a no-write result for that later sub-operation does not mean the whole handoff was unwritten. Dispatched/uncertain effects cannot be cleanly superseded. Serialized checkpoints alone cannot reconstruct this live authority.

`supervision.stage_changes: require_approval` freezes lifecycle advancement and validation-fix rewind proposals before changing required-stage credit. `supervision.plan_changes: require_approval` reviews bounded controller replans and ordinary task-owned initial/reflection replacement plan adoption before application. Executable-plan review retains the exact generated plan, reflection attempt, prepared input/context and finalization position; approval does not regenerate the plan or repeat failed prior effects. It is separate from existing tool approvals and never overrides scope or automatic policy denial. Plan adoption review does not add general pause/resume inside arbitrary executable-plan tool/skill steps.

Respond to reviews with the currently exposed `TaskResumeInput::Approval`. Rejection closes safely as `Incomplete`; nonempty proposal modifications reject rather than silently retarget the reviewed operation. State/reset/control bindings and original UTC/monotonic deadlines are rechecked after acknowledgement waits and before application. Explicit expired `ApprovalResult::Timeout` follows the configured HITL timeout policy through a live, exact-request-bound disposition; this does not authorize late ordinary approval or another proposal. Human-review deadlines are omitted when no HITL engine is configured. Inherited child plan review uses the coordinating pending ID and existing parent/child ownership and failure policies.

Migration notes:

- Development `TaskRunMutation::FinalCheckpoint` literals use `deadline: Option<Instant>` and `evidence: Option<Arc<dyn TaskFinalAdmission>>`; wrap an existing finite deadline in `Some(...)` rather than removing it. A deadline-free publication uses `deadline: None`, independently of whether it needs a live evidence guard. This live mutation change does not alter serialized checkpoint limits or the current finite task defaults; no persisted payload migration is performed. Use `evidence: None` only for operations without a live evidence guard. Wrappers/backends must forward and evaluate supplied guards inside final write admission, not reconstruct them from serialized labels. `ValidationDriverState` has private, nonserialized live state: construct a fresh attempt with `new`, use `restore_retained` for executable same-process round trips, and use `restore` only for audit validation. Development `EvaluationScope`, `EvidenceIdentity` and `ProgressObservation` literals add `validation_dependencies`; empty maps are appropriate for independent observations without upstream validation inputs, while bound framework paths stamp the actual inventory/proof references. The runtime directly uses the workspace's existing `sha2` package for canonical result fingerprints; no package version or workspace version changes. Executable continuation still requires retained owners; workspace and checkpoint envelope versions are unchanged. Inconsistent receipts and ambiguous storage errors may now surface directly instead of being replaced by a normal interruption result; they do not authorize runtime reuse.

- `Tool::execute_task` is an additive `Result<ToolResult>` execution bridge whose default calls existing `execute`; ordinary tool implementations remain source-compatible. Framework messaging/routing overrides it to preserve typed control transfer, and instrumentation wrappers must forward it and `supports_task_messages`. Declaring that capability does not grant tools, provide an arbitrary workflow cursor, or authorize serialized IDs.
- `AgentRegistry::broadcast_task` is an additive `Result<ToolResult>` bridge; existing broadcast signatures remain unchanged. A task-aware host tool must forward typed suspension and the standard result unchanged. Private dispatch frames add defaulted predecessor and declared-wait fields, and message cursors add broadcast formatting. Decoding defaults do not promise older-runtime execution, migration or restart; workspace/storage envelope versions remain unchanged.
- `DisambiguationManager::pending_checkpoint`, `matches_pending_checkpoint`, `clear_pending_checkpoint` and `has_pending_now` expose or validate sensitive manager-owned progress for explicit hosting. They do not restore executable state or mint task/approval authority. Protect checkpoint data as recovery/review data.
- Private executor cursors add defaulted logical start/retry/invocation progress, and message frames retain that cursor alongside an implementation-bound live invocation context. The live context is not serialized or reconstructed from IDs. Task evidence carries a private hop discriminator so failed ancestors and successors remain distinct. These development changes do not alter public Rust signatures or workspace/envelope versions and do not establish cross-version execution or restart compatibility.
- Skills expose additive `SkillExecutionCursor`, `SkillExecutionObserver` and `SkillExecutor::execute_resumable_with_invoker` APIs. The cursor includes an execution identity, bound definition, completed context/results, next step and original pending request. It is progress data, not an execution grant; the runtime still requires its private live owner and current authorization. `ProcessData`, `ProcessMetadata` and `SkillContext` now support serialization for exact retained locations. Private task payloads also retain non-model locations, fallback executor cursors and deferred model finalization. Older runtimes are not guaranteed to read these development checkpoints; workspace/envelope versions are unchanged without an executable migration promise.
- Exhaustive development `TaskEffectState` matches add `Suspended`. Earlier runtimes cannot read the new suspended-message payload; use the matching runtime and prepared-configuration identity. The storage envelope version is unchanged, and this does not promise executable checkpoint migration or restart support.

- Existing `ChatMessage` literals add `provenance: None`; constructors such as `ChatMessage::user` remain unchanged. Older serialized messages default the omitted field to `None`.
- `RuntimeAgent::clear_actor_id()` and `register_context_provider()` now return `Result<()>`; handle them with `?` or another explicit error policy. `remove_context()` returns `Result<Option<Value>>` so ownership denial is distinct from a missing key. Context update/remove/refresh/provider replacement all respect run reservation.
- Exhaustive `TaskRunMutation` / `TaskRunStorageError` matches account for `FinalCheckpoint` / `AdmissionExpired`. `HostControl` literals now include `deadline: Option<Instant>` and `expires_at: Option<DateTime<Utc>>`, bound to the pre-revision lifetime rather than the requested extended expiry. These are live admission bounds, not persisted execution permits; unbounded controls may use `None`.
- `TaskRunStore` adds default-unsupported `supports_paused_limit_revision()` and `revise_paused_limits(...)`. Conforming wrappers must forward both the capability and the validated operation; `Ok(None)` certifies no write, while `Err` may require recovery protection. Existing custom stores still compile and refuse limit control before abandoning a live pause. Private escalation cursors add defaulted processed-input and skill fields; older-runtime execution or automatic checkpoint migration is not promised.
- Exhaustive `TaskResumeInput` matches also account for `Continue`, and `TaskPendingKind` matches account for `HostContinue`. `TaskRunStore` adds `claim_paused(...)`, whose `Ok(None)` certifies no write, plus default-unsupported `supports_objective_revision()` / `apply_host_control(...)`. Scoped memory/SQLite stores implement them; conforming wrappers forward the capability and operation. Default custom claims retain ambiguous-error protection. `TodoStore::rebind_run` changes only a matching same-run next-objective binding under one canonical lock. Private controller/non-tool payloads add review, deadline and exact plan-adoption data; unchanged envelope/workspace versions do not promise older-runtime execution, migration or executable restart.
- Exhaustive `TaskResumeInput` matches account for `UserAnswer`, `QuestionTimeout` and `ExternalSignal`; exhaustive `ValidationDriverOutcome` matches also account for `Suspended`. `ValidationDriverState` literals add `parked: None`; older serialized drivers default it to `None`, which does not make uncertain in-flight work resumable. `ValidationObservationExecutor::take_suspension` retains its synchronous unsupported default. The additive async `acknowledge_suspension` delegates to it unless a framework bridge must checkpoint child progress first. `ValidationObservationPause` literals add `group: None`, with serialized defaults preserving older plain parks. Only an executor retaining the actual operation can supply safe park data. Private controller checkpoints require the matching retained live owner. For tool/child interactions, use the coordinating pending ID even when it wraps a delegated child's request. `AgentError::TaskSuspended` is an internal control transfer, not a tool failure to stringify or treat as an optional child result. `CompletedObservation` literals add `identity: None` when no per-observation identity is available; serialized defaults preserve older driver data.
- Custom `Memory` and `Summarizer` implementations compile with unsupported-default `supports_task_admission()`. Task participation requires preserved provenance and managed internal provider handles. Unwrapped preconstructed summarizers are rejected; do not replace frozen handles after construction.
- Provider/wrapper implementations that own or transparently delegate the managed layer report `manages_invocation_admission()` truthfully and delegate priced capabilities when applicable. New billing and tool-footprint methods have unsupported defaults; existing custom implementations still compile. Quote and settlement DTOs are available under `ai_agents::autonomy`. Hidden host/provider I/O is not automatically counted or isolated.
- `resolve_profile` now resolves requested priced/write limits after checking host ceilings; its returned configuration is not an invocation permit. Prepared-provider preflight and shared executor admission enforce the actual capabilities.

## AgentBuilder

`AgentBuilder` is the main entry point. There are three ways to create an agent.

### Pattern 1: From a YAML file

Load a YAML spec and let the framework auto-configure LLM providers from environment variables:

```rust
use ai_agents::{AgentBuilder, Agent};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = AgentBuilder::from_yaml_file("agent.yaml")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .auto_configure_mcp().await?
        .auto_configure_spawner().await?
        .build()?;

    let response = agent.chat("Hello!").await?;
    println!("{}", response.content);
    Ok(())
}
```

- `auto_configure_llms()` reads the `llm:` and `llms:` blocks from the spec, resolves API keys from environment variables, and registers all providers automatically.
- `auto_configure_features()` wires up error recovery, tool security, process pipeline, and built-in tools from the spec.
- `auto_configure_mcp()` connects to MCP servers declared in the `tools` list (entries with `type: mcp`), discovers their functions, and registers them.
- `auto_configure_spawner()` reads the `spawner:` section, creates the spawner and registry, creates optional shared storage, resolves templates, performs fail-closed auto-spawn and child storage readiness, and registers explicitly configured spawner tools.

This is the same builder chain used by the CLI. `auto_configure_mcp()` and `auto_configure_spawner()` are no-ops when the relevant YAML sections are absent, so it is safe to always include them.

### Pattern 2: From a YAML string

Useful when you store specs in a database, embed them as constants, or generate them at runtime:

```rust
use ai_agents::{AgentBuilder, Agent};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let yaml = r#"
        name: InlineAgent
        system_prompt: "You are a helpful assistant."
        llm:
          provider: openai
          model: gpt-5.4-nano
    "#;

    let agent = AgentBuilder::from_yaml(yaml)?
        .auto_configure_llms()?
        .auto_configure_features()?
        .auto_configure_mcp().await?
        .auto_configure_spawner().await?
        .build()?;

    let response = agent.chat("What is Rust?").await?;
    println!("{}", response.content);
    Ok(())
}
```

### Pattern 3: Fully programmatic

Build everything in code - no YAML at all:

```rust
use ai_agents::{AgentBuilder, Agent, UnifiedLLMProvider, ProviderType};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm = UnifiedLLMProvider::from_env(ProviderType::OpenAI, "gpt-5.4-nano")?;

    let agent = AgentBuilder::new()
        .system_prompt("You are a helpful assistant.")
        .llm(Arc::new(llm))
        .build()?;

    let response = agent.chat("Hello!").await?;
    println!("{}", response.content);
    Ok(())
}
```

You can mix patterns too - load from YAML, then override specific parts programmatically:

```rust
let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .auto_configure_mcp().await?
    .auto_configure_spawner().await?
    .system_prompt("Override the YAML prompt with this one.")
    .hooks(Arc::new(my_hooks))
    .tool(Arc::new(my_custom_tool))
    .build()?;
```

The full builder chain used by the CLI is: `auto_configure_llms` &#x2192; `auto_configure_features` &#x2192; `auto_configure_mcp` &#x2192; `auto_configure_spawner` &#x2192; `build`. The MCP and spawner steps must come after `auto_configure_features()` so the tool registry exists. To keep auto-registered built-ins, call `.tool()` or `.extend_tools()` after `auto_configure_features()`; calling either method first creates a custom registry and causes builtin auto-registration to be skipped. `.tools(registry)` intentionally replaces the entire registry, including previously auto-registered built-ins. Configuration-only overrides such as `.hooks()` and `.system_prompt()` can be applied anywhere before `.build()`.

Use one of these orderings:

- Extend built-ins: load YAML -> configure LLMs -> configure features -> configure MCP -> configure spawner -> add `.tool()` or `.extend_tools()` -> build.
- Supply a complete custom registry: load YAML -> configure LLMs -> call `.tools(registry)` -> configure features -> configure MCP -> configure spawner -> build. Calling `.tools(registry)` after MCP or spawner configuration removes those generated registrations, so only do that when full replacement is intentional and the final YAML grants still resolve.

Registration and availability are separate for ordinary tools. `auto_configure_features()`, `auto_configure_mcp()`, and `auto_configure_spawner()` register tools, but YAML top-level `tools:` decides what the model can call. When loading YAML, omitted top-level `tools:` means no ordinary LLM-callable tools even if Rust registered them. Explicit YAML feature flags such as `spawner.management_tools`, `spawner.orchestration_tools`, and `persona.evolution.allow_llm_evolve` are exceptions because they intentionally register and grant their generated tools. In pure Rust builder flows without YAML, registered tools are treated as the explicit grant.

Spawner restrictions apply equally to YAML, spec, template, auto-spawn, and restore paths. Child IDs use a bounded portable ASCII grammar, active nested child spawners are rejected, allowlist violations reject the complete child, and `max_agents` counts both registered and in-flight spawner-managed children. `shared_llms: true` requires the parent registry to contain every alias referenced by each child; it does not construct child-local providers. Hosts using lower-level detached `SpawnedAgent::from_runtime()` records or direct registry insertion operate outside the spawner capacity counter. A direct `spawn_*` call returns an unregistered child, so the host must register or discard it. Complete topology persistence requires a full parent snapshot and matching child sessions under the same session ID.

---

## Basic Chat

The simplest interaction - send a message, get a response:

```rust
use ai_agents::{Agent, AgentBuilder};

let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .auto_configure_mcp().await?
    .auto_configure_spawner().await?
    .build()?;

let response = agent.chat("Explain ownership in Rust").await?;
println!("{}", response.content);

// Check if tools were used
if let Some(ref calls) = response.tool_calls {
    for call in calls {
        println!("Tool used: {} -> {}", call.name, call.result);
    }
}
```

The `AgentResponse` struct contains:
- `content` - the final text response
- `tool_calls` - optional list of tool calls made during the turn

A `RuntimeAgent` runs one independent root turn at a time. Blocking `Agent::chat()`, legacy `chat_stream()`, and `chat_stream_events()` share the same per-agent turn gate, so concurrent calls on the same runtime are serialized rather than interleaving memory, state, hooks, or finalization. Internal redispatch remains part of its owning root turn. Once a stream owns the turn, it retains ownership until terminal completion or until the stream is dropped; another independent turn on that same runtime waits. A directly nested call, response hook, or framework-owned spawned call such as concurrent orchestration or registry broadcast that targets a runtime already in its current ownership chain fails before waiting, including an `A -> B -> A` cycle. Non-cyclic calls to another runtime remain allowed, and separate runtime instances remain independently concurrent.

---

## Task event streams (development only)

The explicit development `AutonomyRunner` exposes `run_stream(objective, profile)`, `resume_stream(run_id, revision, input)` and `stream_operation(operation, config)`. They return `TaskRunStream`, which implements `Stream<Item = TaskRunEvent>` and also provides `next_event()`. These APIs drive the existing controller and managed tools/providers; they do not repeatedly call public chat streams, create a detached worker, enable YAML task entry or install a production restart host. The development task CLI consumes the same streams. Task events describe lifecycle and committed turn output, not provider token deltas. A provider need not implement incremental streaming to participate.

```rust
use ai_agents::autonomy::{AutonomyRunner, TaskRunResult};

async fn observe(runner: &AutonomyRunner) -> ai_agents::Result<TaskRunResult> {
    let mut events = runner.run_stream("Prepare the report", None)?;
    while let Some(event) = events.next_event().await {
        // The event projection is safe by default; exact host results are separate.
        println!("{:?}: {:?}", event.kind, event.cursor);
    }
    events.take_result().ok_or_else(|| {
        ai_agents::AgentError::Other("task stream ended without an operation result".into())
    })?
}
```

`RuntimeAgent::run_task_stream(&runner, objective, profile)` and `task_operation_stream(&runner, operation, config)` adapt an explicitly configured runner for that exact runtime. A runner bound to another runtime rejects before execution. They do not construct a parallel runner or install builder/YAML hosting. Facade root exports and `ai_agents::autonomy` expose the same event types.

### Event identity and meaning

Each `TaskRunEvent` contains an `operation_id`, operation-local `sequence`, optional `run_id`/source `revision`, a `TaskEventCursor`, `kind` and structured `data`. The cursor orders delivery by coordinating acknowledged revision and then an index within that revision; concurrent producers may carry an older source revision than the delivery high-water. New admitted resume/control claims advance revisions. Sequence is not an invocation count, cursor is not evidence authority, and gaps do not imply an effect. This API is not a durable event log, subscriber replay service or exactly-once transport.

Categories cover task start/resume, stages, optional todos/artifacts, tool requests/results, validation, tri-state gate checks, domain progress, interventions/replanning, forced continuation, committed turns/state changes, pending approval/question/external conditions, host controls and every returned task status. `ToolStarted` is a logical request, not invocation proof. `ToolCompleted` separates actual `executed`, `success` and `eligible_at_record`; none alone proves task completion or continuing freshness. Validation reentry retains its attempt rather than inventing another start. Child tool observations carry redacted runtime/operation identity under the coordinating run. Discarded model content is not task completion evidence.

`TaskStreamOperation` selects initial task/explicit skill execution, retained resume, paused cancel/stop, objective/limit changes, confirmed external/controller delivery loss or restoration review. All use the existing authentication, exact-revision and current-policy contracts. Restoration review still does not answer the original pending question or approve its tool. A safe pause ends this operation's stream while retaining the runner's owner; explicit resume creates the next stream. Consume the operation result before responding to a pending notice, because final pause preparation may be part of the operation. Existing host preparation/startup/reconstruction APIs remain explicit; reconnecting does not execute saved display events.

### Results, privacy and ownership

The final status event follows the existing public operation's acknowledged outcome and ownership handoff, including requested persistence preparation for safe pauses. A storage/API error produces `OperationError`, not a fabricated successful task status. `take_result()` returns the exact original `Result<TaskRunResult>` once the operation finishes; its objective, pending details and evidence are trusted-host data, not a safe JSON export. Status, revision and counters in final events match that result. EOF without a result is not success.

Text and host-defined identifiers are summarized with the existing default privacy policy before queueing. Structured domain maps use redacted name/value entries, including reserved-looking keys; numeric values remain numeric. Display structure is bounded, nested projections are limited to 2 KiB and the complete envelope to 4 KiB, with explicit `display_unavailable` markers rather than unrestricted recovery dumps. These display limits do not truncate exact native history, task evidence or checkpoints. Trusted hosts use the exact returned pending request to render an authorized question/review interface; event identifiers cannot grant approval.

`TaskStreamConfig.capacity` defaults to 32 and accepts 1 through 1,024 events. Full queues backpressure the owned operation instead of dropping control/status events. No work progresses when the stream is not polled, except already admitted framework/host work that has its own existing lifetime. Original deadlines/accounting remain authoritative when polling resumes; this is cooperative control, not hard preemption. Observability's lossy queue is not used as the task delivery channel.

Dropping a never-polled stream admits nothing. Dropping an interrupted active operation blocks further admission and retains uncertain work/owners for explicit recovery; it does not acknowledge cancellation or rollback. `stream.close().await` requests cancellation while continuing to poll the original operation and drain its events, then returns its actual result. Completion can win that race. A returned pause/terminal outcome is not undone merely because display delivery is later lost. Close before first poll returns `None`; interruption before execution acknowledgement reports an error and preserves existing recovery guards. A host may bound its close wait, but dropping that wait still does not prove cleanup. Keep the runner alive for volatile pause/resume or use the existing explicitly prepared durable reconstruction contract.

Run the credential-free example from the checkout with `cargo run --manifest-path examples/Cargo.toml -p example-custom-llm --bin task-events --locked --offline`. It uses a finite fixture provider, no tools or external effects, and demonstrates a real question pause and explicit resume. These are development APIs, not the published 1.1.1 release contract.

## Automatic state and skill tasks (development only)

Enabled `states.*.autonomy` and selected `skills[].autonomy` declarations now promote an admitted ordinary input into the existing `AutonomyRunner`. Construction and ordinary session restore do not start a task. Top-level `autonomy` supplies defaults and named profiles; it does not automatically turn every `chat()` call into a standalone task. Omitted or disabled declarations retain ordinary behavior, including ordinary provider token streaming when no task is activated.

The builder loads external skills and validates the actual installed state machine before publishing the runtime. Reachable enabled profiles bind actual adapters, aliases, tools and requested priced/write capabilities without provider calls. Task settings never grant tools or approve a denied operation.

### Scope, input and execution

State ownership is resolved before skill dispatch. The nearest explicit state autonomy declaration supplies the scope; absent child declarations inherit it when normal state inheritance is enabled. An explicit child declaration resolves against the chosen profile/defaults, and `enabled: false` suppresses new automatic entry. A state using `state_until_exit` owns its declaring subtree: sibling movement inside it does not end that scope. On exit, settled actions and eligible post-exit gates determine completion; an unrelated exit is `Incomplete`, not success. Mandatory pending decisions, required stages and uncertain-effect/cancellation precedence still apply.

A selected autonomous skill retains its actual definition and clarified/prepared input instead of rerouting. Default skill strategy runs its declared segment once and then uses bounded ordinary continuation if gates remain unmet. Explicit `model_led` or `lifecycle` strategies do not force that scripted segment. A skill using the scripted strategy retains the ordinary requirement for a prompt step that produces its response. Exact step results and rendered arguments survive pauses; bounded prior-result context is available to continuation without copying execution metadata or treating historical results as completion authority.

Work already inside a task, including child runtimes, inherits its actual owner, ledger, ceilings and outer completion contract. A nested autonomy declaration cannot start a second run, refill allowance or replace the coordinating gate. Automatic root validation bindings use the actual `state` or `skill` scope tag; explicit standalone observations retain their existing `task` tag. Pre-response entry, regenerating post-response entry and clarification returns preserve completed input preparation and selection. Merely arriving in a state without regenerating a response does not start work. The initial state generation is rechecked after checkpoint waits; existing cursors keep their original execution binding.

A configured literal objective is combined with accepted user input. An `objective_template` instead renders once using `input`/`user_input` (the accepted processed or clarified input), `objective`, `context`, `state` and selected `skill` metadata where available. For an inherited state scope, `state` names its declaring state. Resume does not rerender the objective or replay extraction. Ordinary skill clarification retains the original input-processing context across separate chat turns.

### Retained host service

| API | Contract |
|---|---|
| `AgentBuilder::autonomy(config)` | Programmatic defaults for the same raw profile/override types as YAML |
| `AgentBuilder::automatic_task_options(options)` | Explicit hosting opt-in with trusted ceilings, optional scoped store/configuration identity and optional reconstruction host; it does not enable a disabled profile |
| `AgentBuilder::build_task_host().await` | Explicitly prepares task hosting from the same finalized builder, sharing an injected task-capable session backend or opening configured SQLite; no task or startup continuation is executed |
| `AgentBuilder::task_interaction_delivery(available)` | Hosting opt-in declaring a usable controller-response endpoint; defaults to true, including manual Rust resume, and false follows existing unavailable policies |
| `RuntimeAgent::automatic_task_result()` | Last published exact `TaskRunResult`; trusted-host data, not a display export or proof of current state after an ambiguous error |
| `RuntimeAgent::automatic_task_runner()` | The already created retained controller, or `None` before first lookup/entry |
| `RuntimeAgent::task_runner()` | Lazily resolves the shared configured controller without starting work; successive operations retain it within the same actor scope |
| `RuntimeAgent::run_task(objective, profile)` | Explicit execution using that configured controller and the selected enabled profile |
| `RuntimeAgent::resume_task(run_id, revision, input)` | Explicit typed response on the same controller, retaining exact revision and original authority |
| `RuntimeAgent::cancel_task(run_id, revision)` | `CancellationOutcome` distinguishes `requested`, `acknowledged` cleanup, and `already_terminal` metadata |
| `RuntimeAgent::task_operation(operation)` | Applies explicit typed run/skill/resume/stop/objective/limit and other supported controls through the same controller |
| `RuntimeAgent::task_display(run_id)` | `TaskRunDisplay` is a bounded, default-redacted snapshot of acknowledged facts, not a completion proof or recovery payload |
| `TaskRunDisplay::read(store, run_id).await` | Uses the same safe projection through an authorized scoped store without constructing a runtime/provider, claiming ownership or consuming a pending response |
| `RuntimeAgent::task_interaction(run_id, revision)` | Exact `TaskHostInteraction` with typed prompt data for an authenticated host; not a display/export DTO |
| `RuntimeAgent::automatic_task_operation(operation)` | Existing explicit resume/stop/cancel/objective/limit/delivery-control operations on that controller |
| `AutonomyRunner::task_store()` | The actual scoped store for trusted inspection and reviewed recovery; namespace scoping does not authenticate an external caller |

Calling `task_runner()` does not invoke a provider or start a task. Automatic state/skill entry uses that same configured runner, and acknowledged results remain bound to their exact run even if a later run is admitted before an earlier caller receives its result. After a settled actor change, a new scoped runner is selected instead of exposing the previous actor's task store. `resume_task` keeps the admitted runner and store even after a turn-local `chat_as_actor` context has ended; the trusted host must authenticate the responding caller. A production CLI reconstruction-clock binding remains unfinished. The development CLI, plain REPL and task-aware TUI consume these same convenience APIs and scopes.

Default hosting retains volatile in-memory checkpoints for the lifetime of the runtime/retained runner. It does not require SQLite for normal completion or same-process pause/resume. Durable or restart-prepared policies require an explicit compatible store and `TaskRestoreHost` in `AutomaticTaskOptions`; the supplied configuration identity must match the store. Existing explicit startup selection remains explicit, including for `resume_on_start`. A reconstruction resumes the task's typed safe continuation, not an arbitrary ordinary composing call stack.

### Explicit hosting preparation (development only)

`build_task_host().await` uses the same component construction as `build()`, validates actual automatic-declaration bindings before opening storage, and publishes the hosting service only after preparation succeeds. It does not call `auto_configure_llms`, connect MCP, auto-spawn children, call a model/tool, claim a checkpoint or select a startup candidate. Any earlier builder auto-configuration remains an explicit preparation step; this method is not a pure reconstruction staging implementation.

Storage selection preserves existing precedence:

1. An explicit `AutomaticTaskOptions::store` remains the fixed scoped task store; preparation does not open session storage just to replace it.
2. Otherwise an injected session storage handle is used for tasks only when it advertises `StorageCapability::TaskRuns`.
3. Otherwise configured SQLite is initialized through the runtime's existing storage path and the same handle backs task journals and session storage. The runtime `sqlite` feature is required; existing storage path resolution is unchanged.
4. No storage or a snapshot-only File/Redis/Noop backend leaves task hosting volatile. Preparation does not eagerly open these snapshot-only backends or infer TaskRuns support from them. Their ordinary session initialization and any enabled session-storage requirements still apply at execution.

A new actor-scoped adapter is selected after a settled actor change. An explicit scoped task store is not automatically rebound to another actor. The unchanged synchronous `build()` retains its previous volatile-task behavior unless an explicit task store was supplied. The new asynchronous preparation path is opt-in and does not enable a disabled task profile or change ordinary chat/default-off behavior.

For an already configured builder whose caller owns the runtime lifetime and can deliver typed responses:

```rust
use ai_agents::{AgentBuilder, Result, RuntimeAgent};

async fn prepare_host(builder: AgentBuilder) -> Result<RuntimeAgent> {
    builder
        .task_interaction_delivery(true)
        .build_task_host()
        .await
}
```

Set `task_interaction_delivery(false)` only when no usable task-response endpoint is exposed. This passes the same setting as `AutonomyRunner::with_interaction_delivery` to configured explicit and automatic tasks. Controller stop/fail policies remain effective; other unserviceable volatile pauses, including tool questions and required reviews, are retired through existing typed cleanup and acknowledged as `Incomplete` with `capability_unavailable`, without invoking pending work or approving it. Rejected/lost/dropped cleanup acknowledgements retain custody. A verified cancellation-only successor takes precedence at cleanup admission; only a store-certified no-write claim refusal permits that narrow reread, never arbitrary-revision or ambiguous-error retry. Ordinary HITL and confirmed delivery-loss operations remain distinct, and silence is not loss. Explicitly prepared reconstruction still requires its installed host/store and does not infer process-exit durability.

SQLite task journals are not prepared restart acknowledgements. Persistence still requires an explicitly installed compatible `TaskRestoreHost`, actual implementation/native-history bindings, trusted time continuity, the existing capture/reconstruction acknowledgements and separate restoration review. Backend capability, a configuration digest or this builder method supplies none of those authorities. Production YAML reconstruction and platform clock acceptance remain unfinished; do not exit a one-shot host advertising a durable pause merely because preparation opened SQLite. No package, checkpoint-envelope, dependency or MSRV version changes are introduced.

The [end-to-end task walkthrough](@/docs/task-autonomy.md) connects the runtime-only domain installation in `task_domain.rs` to an embedded host and `EvalHost::configure`. It uses public APIs rather than a repository-specific task engine. Development eval report schema 3 adds scenario controls and per-control results; existing Rust `Scenario` and `TurnResult` literals initialize `interactions` with an empty vector. `ai_agents::eval::interactions` owns these evaluation DTOs. Exact acknowledged control results/events remain non-serializing; only bounded snapshots and redacted errors appear in reports. Explicit objective rendering accepts `input`, `user_input` and `objective` aliases for the same immutable accepted invocation value.

### Foreground host scopes and sessions (development only)

`TaskHostScope::from_spec(spec, base_dir, actor)` resolves an authorized normalized declaration and external skill definitions without constructing providers, reading provider credentials or starting work. Use `AgentBuilder::task_host_scope(scope)` to compare and retain the definitions actually loaded by the builder before `build_task_host().await`. The builder does not independently reread those definitions during final construction. A changed declaration fails instead of binding another configuration's journal. `scope.task_options()` is a lower-level explicit namespace option; it does not perform that builder comparison.

`scope.open_store().await` opens only an existing configured SQLite journal for scoped management. `ScopedTaskRunStore::list_compatible()` retains full candidate validation while filtering other configuration identities; `TaskRunDisplay::read` and `CancellationOutcome::request` perform the existing strict scoped reads/writes. The digest covers declared configuration and loaded skills, not arbitrary opaque tools/providers, dynamic child state or implementation bindings. It cannot authenticate a caller, authorize execution or certify restart compatibility. Existing default embedded-runtime identities and explicit custom stores are unchanged unless the host opts into this scope.

`TaskHostSession::new(available)` provides a live foreground catalogue, not another executor. Poll the original operation and controls inside `session.scope(future)`. Framework-spawned composition carries that same session through its existing captured invocation context; arbitrary host-created Tokio tasks must explicitly retain the scope. Only configured runners actually admitted in the scope appear in `session.runs()`, bounded to 1,024 runs. Holding a registry entry or serialized ID cannot add a participant. Independently hosted automatic children can therefore be serviced while their ordinary parent still awaits a result. Children already inside a task remain under its original coordinating pending request.

Each `TaskHostRun` holds the actual `runner` and `key`. `execution_admitted()` observes installation of the live execution handle, not execution permission or usable pause readiness. Its monotonic `published_revision()` is populated only after the public-result preparation/acknowledgement boundary; an intermediate stored pause is not permission to show a usable response UI. `session.changed().await` signals local admission/publication, not remote storage mutations or a durable event log. Hosts may observe only their actual scoped participants to discover remote cancellation requests. Snapshot observations can race with controls, and response admission still validates exact revision/request and current bindings.

`session.close_input()` reports confirmed endpoint loss and only withdraws availability for that session. It does not cancel running work or prove cleanup. For an acknowledged retained pause, `runner.input_unavailable_paused(run, revision)` preserves applicable controller/external stop/fail policy and uses the existing typed cleanup with its exact cancellation-successor witness. Continue polling running foreground/control operations while requesting cancellation. For host shutdown after execution admission, `runner.request_running_cancel(run)` uses the actual execution's existing serial boundary rather than a stale display revision; `None` means no running request was applied, not cleanup acknowledgement. An acknowledged pause/terminal winner is left to the original operation result. Explicit user responses and `cancel_task` keep their exact-revision checks. `session.has_protected_work()` observes actual local participant ownership; a stored terminal row alone does not certify release. An acknowledged `RecoveryRequired` result ends the foreground operation but deliberately retains protection.

Ordinary callback UI and task responses remain separate protocols. Controller guidance accepts either the existing nonempty text value or a usable standard free-text `QuestionResponse`; it rejects unavailable/timed-out/non-answer or selected-option objects. This normalization does not convert guidance into Stop, Continue, approval or a limit increase. The default [development task CLI](@/docs/cli.md#development-task-cli) is one consumer of these APIs, not a production reconstruction service. No new required trait method, options field, dependency or checkpoint version is introduced.

### Host inspection and cancellation (development only)

`AutonomyRunner` exposes the same `cancel_task`, `task_display` and `task_interaction` operations. For a different process that only has an authorized scoped store, `CancellationOutcome::request(store, run_id, revision)` records a monotonic cancellation request without constructing providers, restoring ownership or executing pending work. Repeated requests at the current revision do not reset usage or consume a response. A stale revision is rejected; an ambiguous write or mismatching receipt is an error, not a successful request to retry blindly.

`CancellationDisposition::Acknowledged` is returned only after retained safe-pause cleanup has completed. Running operations return `Requested` and their original foreground future must keep polling until settlement; non-preemptible or uncertain work can remain `RecoveryRequired`. `AlreadyTerminal` reports stored terminal status, not rollback or global quiescence. `locally_retained: Some(true/false)` is a local custody observation; store-only requests return `None` because another process's live ownership is unknown. It must not be converted into cleanup success.

`TaskRunDisplay::read(store, run_id)` requires an already authorized scoped `TaskRunStore`; it does not build providers, connect MCP, perform startup selection or claim a task. Repeated inspection leaves the revision and pending response unchanged. Store-only `locally_retained: false` means no runtime is attached to that view, not that a remote owner released protection; nonterminal `restoration_review_required` stays `None`. The same redaction and 16 KiB bound apply to local and store-only projections.

`TaskRunDisplay` preserves safe listing metadata plus redacted objective/profile/stage/reasons, stored counters/limits/clocks, bounded todo/progress and available artifact observations. Namespace identifiers have the same visibility as scoped storage listings; the host still authenticates access. `last_gate` and its cycle/objective revision are historical observations, not current eligible completion proof. Missing artifact observations remain unavailable, and `restoration_review_required: None` means no local retained review establishes that fact. Local custody can remain protected even when the store has released its owner token. Serialized display output is limited to `MAX_TASK_DISPLAY_BYTES` (16 KiB); oversized views fail explicitly without changing the task. A snapshot is not a durable event log or a UI-generated clock.

`AutonomyRunner::task_host_details(run_id)` returns `TaskHostDetails` for an authenticated local UI: revision-bound, human-readable objective/profile/stage, canonical optional todos, check outcomes with their historical cycle/objective revision, domain progress/decision observations and file or non-file artifact references. The allowlisted presentation is bounded to 128 lines, 2 KiB per line and 16 KiB total content with explicit truncation; exact results and checkpoints are unchanged. It deliberately has no serde serialization and content-free Debug. Do not parse these lines into controls or treat historical observations as current eligibility. Use `TaskRunDisplay` for safe exports and the typed interaction for actual decisions. Reading details never claims ownership or consumes input.

`TaskHostInteraction` preserves the coordinating run/request/revision even when the displayed prompt comes from a nested child. `TaskHostPrompt` distinguishes approval, question, Continue, capacity, stage/plan review, external condition and validation observation. A stage review's `next_stage: None` with `finishes_lifecycle: true` still requires approval and does not establish task completion. Exact approval context and question/default text can be sensitive: the prompt/interaction types intentionally have no serde serialization and their `Debug` output omits content. Select only the fields needed by an authorized local UI; never log them or export recovery adapters. A view can become stale immediately and cannot be submitted as execution authority. Restoration review, when advertised, remains separate from the original pending answer.

These host APIs do not install production CLI reconstruction, certify an OS clock or promise reboot/cross-host portability. Development task commands, the plain REPL and TUI are frontend consumers of these same host APIs. Original restore bindings, resource/clock limits, current authorization and conservative uncertain-effect protection are unchanged.

A host authenticates control callers and presents exact pending data through its authorized UI. Do not infer permission from an event, parse a model answer as host control, or call a new `chat()` to answer a reserved task. For example, after the application has obtained an answer to the advertised question:

```rust
use ai_agents::{RuntimeAgent, TaskRunResult, TaskRunStatus, TaskStreamOperation};
use ai_agents::autonomy::{TaskPendingKind, TaskResumeInput};

async fn answer_question(
    agent: &RuntimeAgent,
    answer: serde_json::Value,
) -> ai_agents::Result<TaskRunResult> {
    let paused = agent.automatic_task_result().ok_or_else(|| {
        ai_agents::AgentError::Other("no published automatic task".into())
    })?;
    let pending = paused.run.pending.as_ref().ok_or_else(|| {
        ai_agents::AgentError::Other("task has no pending question".into())
    })?;
    if paused.run.status != TaskRunStatus::Paused
        || pending.kind != TaskPendingKind::UserQuestion
    {
        return Err(ai_agents::AgentError::Other("not a paused question".into()));
    }
    agent.automatic_task_operation(TaskStreamOperation::Resume {
        run_id: paused.run.key.run_id,
        revision: paused.run.revision,
        input: TaskResumeInput::UserAnswer {
            request_id: pending.id.clone(),
            answer,
        },
    }).await
}
```

Controller questions use text answers; implementation-bound question tools retain their exact structured answer shape. Stale revisions, mismatched answers and current policy denial remain failures before execution. A lost or dropped execution acknowledgement can leave a newer protected revision than the last published result: inspect `task_store()`, use explicit reviewed recovery where required, and do not blindly retry the old response.

Public runner handles and their streamed resume/control APIs publish the same acknowledged operation results to waiting ordinary consumers. Delivery is bound to the exact run and monotonic revision; another explicit run cannot replace a prior caller's outcome. Existing ordinary pipeline/route/group/concurrent/broadcast calls await a hosted automatic child's real result while their original foreground frame is retained. A pause is not a successful child response; incomplete or uncertain outcomes do not become aggregation success. An ordinary caller's existing timeout/drop policy still applies, and dropping it does not acknowledge child cancellation or restore that caller after process exit.

### Streaming and compatibility

`chat_stream_events()` emits `AgentStreamEvent::Task(TaskRunEvent)` for automatic task progress through the same bounded default-redacted task stream. The final task event follows its acknowledged operation; `AgentStreamEvent::Final` then reports the authoritative dispatch response. Inspect `response.metadata["task_run"]` for `run_id`, `revision`, `status` and optional pending kind: a dispatch `Final` with `paused` is not task completion. Exact pending data remains available only through trusted host results. Legacy `chat_stream()` retains its content/`Done` interface and does not expose structured task events.

A never-polled automatic chat stream starts no task/provider work. Active interruption blocks new admission and retains uncertain ownership; it is not cleanup acknowledgement. Raw resumed-operation drop also wakes an ordinary composing caller when the actual owner becomes abandoned, without fabricating a successful terminal result. Task event delivery retains the existing queue/size/privacy/EOF contracts described above.

The event enum was already non-exhaustive; consumers may handle `Task` or retain their fallback arm. Complete fluent runtime configuration before publishing retained task handles: internal controller handles share the same state and never clone history, locks or generations into a second runtime. Custom providers/tools/adapters gain no new required methods. Formal task eval/report integration remains unfinished; development task commands, plain REPL and TUI service the existing controls, and these development APIs are not the published 1.1.1 contract.

Run the finite credential-free example with `cargo run --manifest-path examples/Cargo.toml -p example-custom-llm --bin automatic-tasks --locked --offline`. It exercises both YAML entry scopes, actual task events, a retained controller question and explicit resume without tools or external effects. The response predicates are deliberate fixture checks, not semantic quality evidence for a real report.

## Streaming

Stream tokens as they arrive from the LLM:

```rust
use ai_agents::{Agent, AgentBuilder, StreamChunk};
use futures::StreamExt;

let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .auto_configure_mcp().await?
    .auto_configure_spawner().await?
    .build()?;

let mut stream = agent.chat_stream("Tell me a story").await?;

while let Some(chunk) = stream.next().await {
    match chunk {
        StreamChunk::Content { text } => {
            print!("{}", text);  // print tokens as they arrive
        }
        StreamChunk::ToolCallStart { name, .. } => {
            println!("\n[Tool: {}...]", name);
        }
        StreamChunk::ToolResult { name, output, success, .. } => {
            println!("[Tool {}: {} (ok={})]", name, output, success);
        }
        StreamChunk::StateTransition { from, to } => {
            println!("[State: {:?} → {}]", from, to);
        }
        StreamChunk::Done {} => break,
        StreamChunk::Error { message } => {
            eprintln!("Error: {}", message);
            break;
        }
        _ => {}  // ToolCallDelta, ToolCallEnd
    }
}
```

When the selected LLM has explicit `tool_choice`, the runtime buffers that provider decision before emitting committed content or the existing `ToolCallStart`, `ToolCallEnd`, and `ToolResult` events. The same buffering applies when reflection is `enabled: true` or `auto`, or when `reasoning.mode` is `cot` or `react`: the committed response arrives as one content chunk, so provisional output cannot differ from `Final` in those modes, and the auto-reflection judge is consulted only once the response exists. v1 does not expose provider-native incremental tool-call deltas as a separate API; agents without those settings keep the token-level streaming path. Tool events are delivered after the tool batch completes; `ToolCallStart` events are emitted in admission order before any `ToolResult`, so they mark when each call was admitted rather than when it finished. Opening the main stream applies the configured LLM retry and `on_failure` fallback policy exactly like blocking execution; a provider failure after the first visible delta is terminal for that turn. Rejected input finalizes the turn as an authoritative response rather than a stream error, and disambiguation clarification responses carry the same `options` and `detection` metadata in `Final` as `chat()` returns. The legacy `chat_stream()` contract remains available and still ends with `StreamChunk::Done {}`.

Use the complete event stream when the caller needs the authoritative final content, metadata, or committed tool-call records:

```rust
use ai_agents::{AgentStreamEvent, StreamChunk};
use futures::StreamExt;

let mut events = agent.chat_stream_events("Tell me a story").await?;
let mut terminal_seen = false;

while let Some(event) = events.next().await {
    match event {
        AgentStreamEvent::Chunk(StreamChunk::Content { text }) => {
            print!("{}", text); // provisional content
        }
        AgentStreamEvent::Chunk(StreamChunk::Error { message }) => {
            terminal_seen = true;
            eprintln!("Error: {}", message);
            break;
        }
        AgentStreamEvent::Final(response) => {
            terminal_seen = true;
            // Authoritative content, metadata, and committed tool-call records.
            println!("\nFinal: {}", response.content);
            break;
        }
        AgentStreamEvent::Chunk(_) => {}
        _ => {}
    }
}

if !terminal_seen {
    eprintln!("Error: stream ended before Final");
}
```

`chat_stream_events()` emits no `Chunk(StreamChunk::Done {})`. A successful stream ends with exactly one `Final(AgentResponse)` after root-turn finalization. Provider failure, root-future or streaming-task abort, consumer drop, whole-turn timeout, and discarded speculative work do not produce a successful `Final`; consumers must treat EOF before `Final` or a terminal error as incomplete rather than promoting provisional chunks to a completed response. `RuntimeControlHandle::cancel_all()` is narrower: it stops or denies tool work, and the owning root turn may still recover and produce `Final`. Expected HITL rejection also finalizes an authoritative cancellation response such as `Operation cancelled: ...` because the turn completed normally with a rejected action.

Content chunks are provisional. Output processing, reflection, or a committed transition can replace them before `Final.content` is built. Final processing does not retroactively sanitize content chunks that a host already displayed; hosts that must hide provisional content need an appropriate buffering policy.

### StreamChunk variants

| Variant            | Description                                |
| ------------------ | ------------------------------------------ |
| `Content`          | A piece of text from the LLM              |
| `ToolCallStart`    | A tool invocation is beginning             |
| `ToolCallDelta`    | Incremental arguments for a tool call      |
| `ToolCallEnd`      | Tool call arguments are complete           |
| `ToolResult`       | Tool execution finished with a result      |
| `StateTransition`  | State machine moved to a new state         |
| `Done`             | Stream is complete                         |
| `Error`            | Something went wrong                       |

---

## Runtime Optimization Types

Most runtime optimization behavior is configured in YAML, but policy types are re-exported for Rust hosts that build specs or inspect configuration programmatically.

```rust
use ai_agents::{
    RuntimeConfig, RuntimeOptimizationConfig, RuntimeOptimizationKind,
    RuntimeBranchResult, RuntimeCommitBehavior, RuntimeTaskPurpose,
    StreamingOptimizationPolicy, TransitionTiming,
};

let mut runtime = RuntimeConfig::default();
runtime.optimization.enabled = true;
runtime.optimization.max_speculative_llm_calls_per_turn = 4;
runtime.optimization.speculative_state_transitions = true;
runtime.optimization.streaming_policy = StreamingOptimizationPolicy::BufferUntilRoutingDone;
```

Branch observability labels use `RuntimeOptimizationKind` and `RuntimeCommitBehavior` internally. Public users normally read those labels from observability reports rather than constructing branches directly. `RuntimeBranchResult` covers scheduler-managed blocking branch outputs, while buffered streaming uses `StreamingDraftResult` through its streaming-specific safety path. A branch status of `committed` means that branch won selection into the one committed runtime path; it does not certify transaction success or promise rollback of provider calls, tool effects, network requests, processes, or effects already observed before a stream is dropped. Buffered streaming with `BufferUntilRoutingDone` currently applies to response-independent parallel state-transition routing. Its buffer limit protects unresolved routing; after routing misses or fails, later chunks are collected for the committed output without consuming that unresolved-route buffer.

---

## Custom Tools

`Tool::id()` is the canonical ID used by YAML `tools:`, `tool_security.tools`, `hitl.tools`, eval evidence, recovery policy, and aliases. Display names and localized aliases may be shown to the model, but runtime policy resolves them back to the canonical ID before execution.

Implement the `Tool` trait to give your agent new capabilities:

```rust
use ai_agents::tools::{
    ResultLimitBinding, ResultLimitKind, Tool, ToolExecutionContext, ToolPolicyBindings,
    ToolResult,
};
use async_trait::async_trait;
use serde_json::{json, Value};

struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn id(&self) -> &str { "weather" }
    fn name(&self) -> &str { "Weather Lookup" }
    fn description(&self) -> &str { "Get current weather for a city" }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["city"]
        })
    }

    fn policy_bindings(&self) -> ToolPolicyBindings {
        ToolPolicyBindings {
            result_limit_fields: vec![ResultLimitBinding::new(
                "max_results",
                ResultLimitKind::MaxResults,
            )],
            ..Default::default()
        }
    }

    async fn execute(&self, args: Value, ctx: ToolExecutionContext) -> ToolResult {
        let city = args["city"].as_str().unwrap_or("unknown");
        let backend = ctx.custom_config.get("backend").and_then(Value::as_str).unwrap_or("mock");
        let max_results = ctx.limits.max_results.unwrap_or(1);
        ToolResult::ok(json!({
            "city": city,
            "backend": backend,
            "max_results": max_results,
            "forecast": "72°F and sunny"
        }).to_string())
    }
}
```

Register it on the builder:

```rust
let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .auto_configure_mcp().await?
    .auto_configure_spawner().await?
    .tool(Arc::new(WeatherTool))
    .build()?;
```

`ToolResult` has two constructors:
- `ToolResult::ok("output")` - success
- `ToolResult::error("message")` - failure (the agent sees the error and can retry or explain)

For a runnable version of this pattern, see `examples/rust/custom-tools/src/context_tool.rs` and `examples/rust/custom-tools/agents/context_tool_agent.yaml`. The example shows `max_results` flowing through `ctx.limits` and `backend`, `tenant`, and related custom settings flowing through `ctx.custom_config`.

For tools that need runtime safety or scheduling hints, override the optional metadata methods too:

```rust
use ai_agents::tools::{
    ToolCallClassification, ToolOperationKind, ToolSafetyMetadata, ToolSideEffectLevel,
};

fn safety_metadata(&self) -> ToolSafetyMetadata {
    ToolSafetyMetadata {
        read_only: true,
        concurrency_safe: true,
        operation: ToolOperationKind::Read,
        side_effect_level: ToolSideEffectLevel::None,
        requires_network: false,
        destructive: false,
        open_world: false,
        host_dependent: false,
        requires_user_interaction: false,
        supports_cancellation: true,
        default_requires_approval: false,
        should_defer_schema: false,
        max_output_chars: Some(20_000),
        max_result_size_chars: Some(20_000),
    }
}

fn classify_call(&self, args: &Value) -> ToolCallClassification {
    ToolCallClassification::from_metadata(&self.safety_metadata())
}
```

The shared runtime resolves names to canonical IDs, checks scope and the initial `tool_security` policy, and then preflights host-backed `command`, `diagnostics`, and `web_search` availability before requesting HITL approval. After approval it takes a fresh runtime-control snapshot, resolves the final tool once, reapplies policy caps, recomputes classification and resource keys, and verifies current scope, emergency control, provider availability, policy, confirmation requirement, and final arguments before lock acquisition. State-scope evaluation records the state generation that authorized the call. After waiting for locks, a changed runtime, policy, or state generation fails closed and one atomic admission records the rate-limited call immediately before invocation. The same resolved tool object is executed. Resource guards cover admission and the tool side effect, then release before completion hooks or fallback re-enters the shared path. YAML skills, state actions, plans, fallback, orchestration, generated tools, and spawned runtimes use the same path through `ToolInvoker`, so tool calls do not bypass runtime policy. Direct `SkillExecutor::execute` is prompt-only; use `execute_with_invoker` for skills that contain tool steps.

Plan-and-execute prompts are built only after the runtime resolves the same effective agent, runtime, and state-narrowed tool scope. Planning-level tool filters can narrow that set but cannot widen it. A scope-resolution failure is returned before the planner LLM is called rather than substituting every registered tool.

A tool timeout applies separately to each `Tool::execute` invocation attempt. `ToolSecurityConfig::default_timeout_ms`, each `ToolPolicyConfig::timeout_ms`, the nested tool-recovery timeout fields in `ErrorRecoveryConfig`, and `ToolCallClassification::timeout_ms` are `u64` millisecond fields that accept `0..=MAX_TOOL_TIMEOUT_MS` (`3_153_600_000_000_000ms`, a fixed 100,000 365-day-year duration). Fallible security/recovery construction and `try_set_tool_security()` reject larger configured values with complete paths. Initial call classification is validated before initial HITL, and classification recomputed from approved argument changes is revalidated before locking and invocation. The selected tool-security timeout is the baseline; recovery and call-classification values can only lower it. The runtime derives `ToolExecutionContext.deadline` and the Tokio timer from that one checked effective duration. Time spent waiting for HITL or resource locks is outside that timer, and a safely retryable call receives a new invocation timeout on each retry; these pieces do not form one end-to-end tool-request deadline. The built-in `command` input timeout is a separate direct-child process limit, while eval `timeout_per_turn_ms` wraps the complete agent turn. Timeout or cancellation does not provide a general rollback guarantee. Custom tools remain responsible for idempotency, cleanup, and external transaction semantics. Tool fallbacks run as separate shared-executor requests after the failed original finalizes. Runtime-control cancellation remains terminal for that tool request and does not start fallback work. The runtime carries resolved canonical ancestry between requests, rechecks it after final tool resolution, and uses a bounded hop limit to stop direct, alias-mediated, drifted, or overlong recursive fallback before invocation; the terminal rejected request remains available as `executed: false` evidence.


Custom tools should use `ctx.limits` for effective framework caps and `ctx.custom_config` for tool-specific settings from `tool_security.tools.<tool_id>.config`. A custom `classify_call()` may set `ToolCallClassification.timeout_ms` to lower the invocation cap for arguments known only at call time; it cannot raise the security or recovery result. Tool input arguments are model-callable schema fields; `config` is host-supplied and not model-callable. Do not parse `tool_security` YAML directly inside the tool. If `fail_closed: true` and a path, domain, command, operation, or result-limit policy is configured and cannot be enforced by the shared executor alone, custom tools must declare matching `policy_bindings()` or execution is denied before the implementation runs. Side-effecting tools should also set call classification carefully: use `requires_approval` for risky mutation or command calls and set `safely_retryable` only when repeating the exact call cannot duplicate side effects. Declare every path field, including source and destination fields, so the runtime records normalized resources and places non-concurrency-safe path operations under the shared path-mutation lock. Non-concurrency-safe calls without a concrete resource use the shared unbound-side-effect lock.

Host-backed built-in hooks are installed on the built runtime:

```rust
use ai_agents::tools::{DiagnosticsProvider, QuestionHandler, WebSearchProvider};
use std::sync::Arc;

let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .build()?;

agent.set_question_handler(Some(my_question_handler as Arc<dyn QuestionHandler>));
agent.set_diagnostics_provider(my_diagnostics_provider as Arc<dyn DiagnosticsProvider>);
agent.set_command_runner(Arc::new(ai_agents::tools::ProcessCommandRunner));
agent.set_web_search_provider(my_search_provider as Arc<dyn WebSearchProvider>);

let todos = agent.todos();
let control = agent.runtime_control();
```

`set_question_handler()` powers `ask_user`, `set_diagnostics_provider()` powers `diagnostics`, `set_command_runner()` powers `command`, `set_web_search_provider()` powers `web_search`, and `todos()` returns the current runtime-local task list. After the initial policy check, a missing diagnostics, command, or search provider is rejected before HITL and tool invocation. Availability is checked again during final pre-lock admission in case host capability changed while approval waited; the final record is unavailable with `executed: false`. `ask_user` instead executes its structured default/unavailable fallback. `ProcessCommandRunner` runs argv without a shell, starts from an empty environment, applies policy-filtered env values, bounds stdout/stderr while reading, requests kill and wait cleanup for the direct child on its own command timeout, and redacts sensitive argv values in evidence. The runtime-control handle can update tool security with fallible `try_set_tool_security()` or compatibility `set_tool_security()`, narrow the runtime's declared tool grant with `set_tool_scope()`, clear those overrides, enable emergency denial with `set_emergency_deny()`, or call `cancel_all()` to deny future tool calls and request cancellation of active cancellable tool work. Tool cancellation does not itself abort the owning root turn. Prefer `try_set_tool_security()` for host-supplied policy because out-of-range tool timeouts and invalid positive-only limits return an error before the active policy or runtime-control generation changes; `set_tool_security()` preserves its existing return type and panics on invalid host policy. `RecoveryManager::try_new()` similarly returns invalid recovery timeout paths, while the compatibility `RecoveryManager::new()` panics on invalid configuration. Runtime scope entries are canonicalized and intersected with the declared grant, so an override cannot grant a registered tool that the runtime did not already own. Runtime controls are local to each runtime and do not automatically propagate from a parent to spawned children.

`web_fetch` prompt extraction also uses the runtime LLM registry when a router or default model is available, so nested extraction calls flow through the normal observed provider path. The built-in `web_search` is separate from provider-native LLM search options: it requires an explicit tool grant, a host-installed `WebSearchProvider`, and shared-executor evidence.

`ToolProvider` is a different contract from `LLMProvider`: it advertises a collection of tool descriptors and resolves executable `Tool` objects. Provider registration and refresh validate the complete raw descriptor snapshot before publishing anything. Duplicate descriptor IDs and IDs owned by another registration reject the entire update even when `get_tool()` returns `None`; a valid unique descriptor whose executable object is unavailable is omitted. Refreshes for one registration are serialized, and a refresh started before unregister/re-register cannot publish into the replacement registration. Tool, provider, display-name, alias, and registry-version changes become visible together. Exact canonical IDs take priority; normalized names and aliases resolve only when one tool owns the claim. `map_tools()` captures one registry snapshot and invokes the wrapper callback after releasing registry locks.

### Direct subcrate boundaries

The curated `ai-agents` facade exports normal host integration for questions, diagnostics, commands, and web search, including `WebSearchProvider` request and response types. Add a matching direct `ai-agents-tools` dependency only for lower-level custom web-fetch transport or resolver injection (`WebFetchTransport`, `WebFetchResolver`, and their request/response types) or eval-oriented `StaticWebSearchProvider` and `UnavailableWebSearchProvider` helpers. A custom web-fetch transport must return one HTTP response per call without automatically following redirects so `WebFetchTool` can validate every hop. A socket-opening implementation must override validated sending, connect only to the supplied approved addresses, and honor `max_response_bytes` while reading each response; the compatibility default cannot enforce a host transport's DNS, proxy, or egress behavior.

---

## Custom LLM Providers

Implement `LLMProvider` to integrate any backend the framework doesn't support natively:

```rust
use ai_agents::llm::{LLMProvider, LLMResponse, LLMError, LLMConfig, LLMChunk, LLMFeature, ChatMessage};
use async_trait::async_trait;

struct MyProvider;

#[async_trait]
impl LLMProvider for MyProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<LLMResponse, LLMError> {
        // Call your LLM backend here
        todo!()
    }

    async fn complete_stream(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError> {
        // Return a stream of chunks
        todo!()
    }

    fn provider_name(&self) -> &str { "my-provider" }

    fn supports(&self, feature: LLMFeature) -> bool {
        matches!(feature, LLMFeature::SystemMessages | LLMFeature::Streaming)
    }
}
```

Provider implementations that need opaque state for a native tool continuation can attach a `NativeProviderState` to `LLMResponse`. The provider state records a credential-free target, original provider content, and one binding per normalized runtime call. The runtime moves validated state into its native assistant marker, while memory preserves the signed exchange as one unit and observability uses a state-free projection.

The target constructor accepts a provider-owned canonical destination identity and does not interpret it as a URL. Custom providers must exclude credentials and normalize any transport-specific identity before constructing the target. The first-party Google adapter separately requires an HTTP(S) base URL without user information, query parameters, or fragments. Framework observability and auxiliary model paths remove opaque state, but a custom `AgentHooks::on_llm_complete` implementation is a trusted host integration and may receive the runtime's replay-bearing native marker. Do not log that raw content; apply an application-level projection or record only safe dimensions such as provider, model, call count, and state size.

```rust
use ai_agents::{
    NativeCallBinding, NativeProviderState, NativeProviderTarget,
};

let state = NativeProviderState::new(
    "provider-response-id",
    "my-provider",
    "native-api",
    NativeProviderTarget::new("https://provider.example/v1/", "model-name")?,
    original_model_content,
    vec![NativeCallBinding::new("runtime-call-id", 0)?],
)?;

let response = response.with_provider_state(state)?;
# Ok::<(), ai_agents::LLMError>(())
```

`LLMProvider::is_terminal_error()` defaults to `false`. Override it only for local configuration, protocol, or native-history failures that retry or model/static fallback cannot repair. Wrappers around a provider must delegate this classification to the actual inner provider. Ordinary API, rate-limit, timeout, and transport errors should retain their existing recovery behavior.

`MultiLLMRouter` uses specialized providers for `select_tool`, `generate_tool_args`, `evaluate_yesno`, and `classify`. With its default fallback enabled, a specialized `Network`, `RateLimit`, or API 408/429/500/502/503/504 error retries that capability once with the primary provider when the selected and primary `Arc`s differ. Provider-terminal errors, configuration or serialization/parsing failures, model-not-found, content filtering, `Other`, API responses outside that allowlist, and status-less API errors do not fallback. If the primary attempt fails, its error is returned. Normal completion, tool completion, streaming, and `process_task` already use primary and are not part of this capability fallback.

For `UnifiedLLMProvider`, normal `provider: google` completion methods use the first-party Google adapter. The public `build_llm()` method deliberately keeps its previous return type and constructs the upstream compatibility backend; direct callers of that escape hatch do not receive first-party Google signed-history handling.

Existing custom providers compile without implementing tool-specific methods. They use the prompt protocol when an agent configures explicit tool choice. A custom provider can opt into native selection by implementing `complete_with_tools()`, returning `true` from `supports_tool_choice()`, and storing normalized calls in `LLMResponse` with `set_tool_calls()` or `with_tool_calls()`. The request uses `LLMToolRequest`, `LLMToolDefinition`, and `ToolChoice`; native calls are still executed only by the runtime's shared executor. `configured_tool_choice()` is optional and returns no override by default.

Wire it in:

```rust
let agent = AgentBuilder::new()
    .system_prompt("Hello!")
    .llm(Arc::new(MyProvider))
    .build()?;
```

---

## Custom Memory

Implement the `Memory` trait to control how conversation history is stored:

```rust
use ai_agents::llm::ChatMessage;
use ai_agents::memory::Memory;
use ai_agents::error::Result;
use async_trait::async_trait;

struct MyMemory { /* ... */ }

#[async_trait]
impl Memory for MyMemory {
    async fn add_message(&self, message: ChatMessage) -> Result<()> { todo!() }
    async fn get_messages(&self, limit: Option<usize>) -> Result<Vec<ChatMessage>> { todo!() }
    async fn clear(&self) -> Result<()> { todo!() }
    fn len(&self) -> usize { todo!() }
    async fn restore(&self, snapshot: ai_agents::memory::MemorySnapshot) -> Result<()> { todo!() }
}
```

Register it:

```rust
let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .auto_configure_mcp().await?
    .auto_configure_spawner().await?
    .memory(Arc::new(MyMemory::new()))
    .build()?;
```

The framework ships with `InMemoryStore` (simple ring buffer) and `CompactingMemory` (serialized LLM-based summarization with token budgets and protected recent-message retention). `MemorySnapshot` includes `messages`, optional `summary`, and a defaulted `summarized_count`; the default keeps snapshots written before this field backward-compatible.

---

## Custom HITL (Human-in-the-Loop)

Implement `ApprovalHandler` to intercept tool calls or state transitions that need human approval:

```rust
use ai_agents::hitl::{ApprovalHandler, ApprovalRequest, ApprovalResult};
use async_trait::async_trait;

struct SlackApprover;

#[async_trait]
impl ApprovalHandler for SlackApprover {
    async fn request_approval(&self, request: ApprovalRequest) -> ApprovalResult {
        // Post to Slack, wait for response...
        println!("Approval needed: {}", request.message);
        ApprovalResult::Approved
    }
}
```

Register it:

```rust
let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .auto_configure_mcp().await?
    .auto_configure_spawner().await?
    .approval_handler(Arc::new(SlackApprover))
    .build()?;
```

For simple cases, use the helper functions instead of implementing the trait:

```rust
use ai_agents::hitl::{create_handler, ApprovalResult};

let handler = create_handler(|request| {
    println!("Tool: {:?}", request.trigger);
    ApprovalResult::Approved
});
```

---

## Agent Hooks

Implement `AgentHooks` to observe lifecycle events - logging, metrics, debugging:

```rust
use ai_agents::hooks::AgentHooks;
use ai_agents::llm::{ChatMessage, LLMResponse};
use async_trait::async_trait;
use serde_json::Value;

struct MetricsHooks;

#[async_trait]
impl AgentHooks for MetricsHooks {
    async fn on_message_received(&self, message: &str) {
        println!("[metrics] User message: {} chars", message.len());
    }

    async fn on_llm_complete(&self, _response: &LLMResponse, duration_ms: u64) {
        println!("[metrics] LLM responded in {}ms", duration_ms);
    }

    async fn on_tool_start(&self, tool: &str, _args: &Value) {
        println!("[metrics] Tool {} starting", tool);
    }

    async fn on_state_transition(&self, from: Option<&str>, to: &str, reason: &str) {
        println!("[metrics] State: {:?} → {} ({})", from, to, reason);
    }

    async fn on_error(&self, error: &ai_agents::error::AgentError) {
        eprintln!("[metrics] Error: {}", error);
    }
}
```

All hook methods have default no-op implementations, so you only override the ones you care about.

Register hooks:

```rust
let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .auto_configure_mcp().await?
    .auto_configure_spawner().await?
    .hooks(Arc::new(MetricsHooks))
    .build()?;
```

The framework also provides `LoggingHooks` (uses `tracing`) and `CompositeHooks` (combines multiple hooks):

```rust
use ai_agents::hooks::{LoggingHooks, CompositeHooks};

let hooks = CompositeHooks::new()
    .add(Arc::new(LoggingHooks::new()))
    .add(Arc::new(MetricsHooks));

let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .auto_configure_mcp().await?
    .auto_configure_spawner().await?
    .hooks(Arc::new(hooks))
    .build()?;
```

### Available hook events

| Method                      | Fires when                                  |
| --------------------------- | ------------------------------------------- |
| `on_message_received`       | User sends a message                        |
| `on_llm_start`              | LLM request is about to be sent             |
| `on_llm_complete`           | LLM response received (with timing)         |
| `on_tool_start`             | A logical executor request emits its optional start notification |
| `on_tool_complete`          | A shared-executor request reaches a final record (with timing) |
| `on_state_transition`       | State machine changes state                  |
| `on_error`                  | An error occurred                            |
| `on_response`               | Final response is ready                      |
| `on_approval_requested`     | HITL approval is needed                      |
| `on_approval_result`        | HITL decision was made                       |
| `on_memory_compress`        | Memory compression triggered                 |
| `on_memory_evict`           | Messages evicted from memory                 |
| `on_memory_budget_warning`  | Token budget threshold exceeded              |
| `on_delegate_start`         | Orchestration: delegation to a registry agent begins |
| `on_delegate_complete`      | Orchestration: delegation completes (with timing)    |
| `on_concurrent_complete`    | Orchestration: parallel agent execution completes    |
| `on_group_chat_round`       | Orchestration: a group chat round finishes            |
| `on_pipeline_stage`         | Orchestration: a pipeline stage completes             |
| `on_pipeline_complete`      | Orchestration: full pipeline execution completes      |
| `on_handoff_start`          | Orchestration: a handoff chain begins                 |
| `on_handoff`                | Orchestration: an agent-to-agent handoff occurs       |
| `on_persona_evolve`         | Persona: a persona field was mutated via `evolve()`   |
| `on_secret_revealed`        | Persona: a secret's reveal conditions were satisfied for the first time |
| `on_facts_extracted`        | Facts: new facts were extracted from a conversation turn |
| `on_actor_memory_loaded`    | Facts: actor facts were loaded from storage at session start |
| `on_session_created`        | Session: a new session was created with metadata      |
| `on_sessions_expired`       | Session: expired sessions were cleaned up via TTL     |
| `on_relationship_loaded`    | Relationship: actor relationship was loaded or created |
| `on_relationship_change`    | Relationship: one or more relationship dimensions changed |
| `on_notable_event`          | Relationship: a notable relationship event was stored |

`on_tool_start` and `on_tool_complete` are executor request-lifecycle notifications, not per-retry invocation notifications or proof that `Tool::execute` ran. A request can finalize without a start hook or after a start hook without invoking the implementation. Use `on_tool_execution_record` and treat `ToolExecutionRecord.executed` as the authoritative invocation flag.

---

## Runtime Context

Inject dynamic key-value data that the agent can access during conversations:

```rust
use serde_json::json;

let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .auto_configure_mcp().await?
    .auto_configure_spawner().await?
    .build()?;

// Set context values
agent.set_context("user_name", json!("Alice"))?;
agent.set_context("subscription", json!("premium"))?;

// Update a nested path
agent.update_context("user.preferences.theme", json!("dark"))?;

// Read all context
let ctx = agent.get_context();
println!("{:?}", ctx);

// Refresh a context source (for dynamic/HTTP providers)
agent.refresh_context("pricing").await?;
```

Context values are available to the agent's system prompt via template rendering and to tools during execution. For a YAML `context` source with `type: runtime` and `required: true`, call `set_context` or `update_context` before each turn that needs the value. A missing top-level key fails blocking chat before model work; streaming reports a terminal error chunk without a `Final` response. A YAML `default` satisfies this presence check, and previously supplied values persist across turns and restored sessions. The check does not validate nested fields, types, non-null values, or whether the host refreshed the value this turn; validate these at the host boundary if required.

Non-runtime context is initialized on the first turn. `per_turn` sources are resolved by initialization and again by that first turn's refresh, then once before every later turn. The current `reset()` clears conversation/state ownership but preserves context and the initialized flag. `restore_state()` restores snapshot context without changing that flag: a fresh runtime resolves its sources on the first post-restore turn and can overwrite restored non-runtime values. A runtime initialized before restore keeps restored `once` and `per_session` values, while its next turn replaces restored `per_turn` values. `per_session` is not automatically invoked by reset or restore; call `refresh_context()` for a specific source or construct a new runtime when the host needs an explicit session boundary.

---

## Session Persistence

Save and restore full agent state - conversation history, state machine position, and context:

```rust
// Save the current session
agent.save_session("session-abc-123").await?;

// Later, load it back
let found = agent.load_session("session-abc-123").await?;
if found {
    println!("Session restored!");
}

// List all saved sessions
let sessions = agent.list_sessions().await?;

// Delete a session
agent.delete_session("session-abc-123").await?;
```

Storage methods advertise support through `AgentStorage::supports(StorageCapability)`. Unsupported extended operations return `AgentError::UnsupportedStorageCapability`.

| Backend | Snapshots | Metadata and filtering | Expiry cleanup | Facts and relationships | Atomic actor deletion |
|---------|-----------|------------------------|----------------|-------------------------|-----------------------|
| File | Yes | No | No | No | No |
| Redis | Yes | No | No | No | No |
| SQLite | Yes | Yes | Yes | Yes | Yes |
| `NoopStorage` | No | No | No | No | No |

Redis's `storage.ttl_seconds` expires Redis snapshot keys directly; it does not implement generic session metadata or `ExpiryCleanup`. `NamespacedStorage` derives snapshot, metadata, filtering, facts, relationships, and actor-deletion support from its inner backend, but intentionally does not forward backend-global expiry cleanup.

### Redis-native helpers

With `redis-storage` enabled, `RedisStorage` also exposes backend-specific bookkeeping helpers. These do **not** add capabilities to `AgentStorage`:

| Helper | Meaning |
|--------|---------|
| `get_meta` | Read native `RedisSessionMeta`; absent values return `None`, malformed values return a persistence error. |
| `exists` | Check snapshot-key existence, not JSON validity or index consistency. |
| `list_sessions_by_agent` | List the agent's indexed snapshots and lazily prune stale ownership entries. |
| `set_ttl` | Retimes snapshot and native metadata together. Zero deletes an existing valid session. A missing snapshot is a no-op; existing snapshots with missing or malformed metadata fail. |
| `expire_sessions` | Sweep indexed snapshots whose parsed `updated_at` is strictly before the UTC cutoff. Count actual deletions and skip observed payload/metadata changed before deletion. |

The expiry helper preserves RFC3339 instant comparison, including offsets and fractional seconds. Missing metadata and invalid timestamps are skipped; malformed metadata returns an error. A sweep can already have deleted earlier eligible entries before a later error. It is not a transaction over the entire session list, generic TTL cleanup, a revision/ABA guarantee, or a distributed lock. Changed observations are conservatively skipped until a later sweep; raw payload equality is checked as well as metadata equality so an unchanged timestamp does not authorize deleting a different snapshot.

`RedisStorage::new()` parses the URL without connecting. Each subsequent operation opens a connection, with no backend-configured deadline or automatic operation retry. A caller timeout bounds waiting, not server-side effects; submission followed by connection loss may leave the write outcome unknown. Bound operations at the host layer and do not blindly retry uncertain mutations.

Redis stays Experimental. The CI service target is a single writable Redis 7.4.10 instance, not a declaration of support for every older version or distributed deployment. See the [Redis operational boundary](@/docs/yaml-reference.md#redis-operational-boundary) for TTL, durability, index-size, and deployment limitations.


---

## Actor Memory & Key Facts

Track facts about each actor across sessions. Facts are extracted automatically after each turn and injected into the system prompt on the next session.

```rust
// Set the current actor ID (user, player, other agent).
agent.set_actor_id("customer_42")?;
// Convenience alias:
agent.set_user_id("customer_42")?;

// Load previously stored facts for this actor from storage.
agent.load_actor_memory().await?;

// Read the current actor ID.
let actor = agent.actor_id(); // Option<String>

// Read cached facts (loaded from storage or extracted this session).
let facts = agent.actor_facts(); // Vec<KeyFact>

// Manually extract facts from the last N messages.
let new_facts = agent.extract_facts(10).await?;

// Access the FactStore for direct manipulation.
if let Some(store) = agent.fact_store() {
    let all = store.get_facts("customer_42").await?;
}

// Privacy-aware deletion. Returns Err when memory.actor_memory.privacy.allow_deletion is false.
agent.delete_actor_data("customer_42").await?;

// Session metadata APIs.
let meta = agent.session_metadata();              // current SessionMetadata
agent.set_session_metadata(meta);                 // overwrite tags, ttl, custom

// TTL cleanup and filtered listings (sqlite backend).
let removed = agent.cleanup_expired_sessions().await?;
let filter = ai_agents::facts::SessionFilter {
    actor_id: Some("customer_42".to_string()),
    tags: None,
    agent_id: None,
    created_after: None,
    created_before: None,
    limit: Some(10),
};
let summaries = agent.list_sessions_filtered(&filter).await?;
```

When `auto_extract: true` (the default), extraction runs after every turn - no manual calls needed. Durable facts require `StorageCapability::ActorFacts`, and privacy deletion requires `StorageCapability::ActorDataDeletion`; SQLite is currently the only built-in backend providing both. Configure via `memory.facts` and `memory.actor_memory` in YAML. See [YAML Reference](@/docs/yaml-reference.md#facts-key-facts-extraction) for the full schema.

Switching actors mid-session via `set_actor_id()` (or via `from_context` resolution when the configured context path changes) clears the cached facts and reloads on the next turn, so prompt injection always reflects the current actor.

---

## Relationship Memory

Relationship memory tracks the agent's stance toward each actor. Use `relationship_manager()` for direct inspection or manual updates.

```rust
// Relationship memory uses the same actor ID as actor facts.
agent.set_actor_id("customer_42")?;

// For one turn, carry actor context structurally without changing global actor state.
let response = agent
    .chat_with_actor_context(
        "Please help with this issue",
        ai_agents::TurnActorContext::new()
            .with_origin_actor("customer_42")
            .with_sender_agent("coordinator"),
    )
    .await?;

// The event stream accepts the same turn-scoped context and emits the same `Final` contract.
let mut events = agent
    .chat_stream_events_with_actor_context(
        "Please help with this issue",
        ai_agents::TurnActorContext::new().with_origin_actor("customer_42"),
    )
    .await?;

// Access the manager when memory.relationships is enabled in YAML.
if let Some(manager) = agent.relationship_manager() {
    let relationship = manager.get_or_create("customer_42", Some("Jane"));
    println!("trust = {:?}", relationship.dimensions.get("trust"));

    // Manual update for application-driven signals.
    let change = agent
        .update_relationship_dimension("trust", 0.1, Some("Customer verified identity successfully"))
        .await?;
    println!("{} changed by {}", change.dimension, change.delta);

    // Two-sided configs can update a specific perspective.
    let perceived = agent
        .update_relationship_dimension_for_perspective(
            ai_agents::relationships::RelationshipPerspective::PerceivedActorToAgent,
            "trust",
            0.1,
            Some("Customer expressed confidence in the agent"),
        )
        .await?;
    println!("{} {} changed by {}", perceived.perspective, perceived.dimension, perceived.delta);
}
```

Automatic updates run after successful turns when `memory.relationships.auto_update.enabled` is true. Relationship context is injected at `relationships.current_actor.*`, and formatted prompt text is available as `{{ relationship_memory }}`.

Persistent relationship memory requires `StorageCapability::ActorRelationships`, currently provided only by SQLite among the built-in backends. Enable it with the `sqlite` feature:

```toml
ai-agents = { version = "1.1.2", features = ["sqlite"] }
```

Configure storage in your YAML:

```yaml
storage:
  type: sqlite
  path: "./sessions.db"
```

Redis can persist session snapshots but cannot persist dedicated actor relationships through the generic storage trait.

You can also use the lower-level API with any `AgentStorage` implementation:

```rust
use ai_agents::persistence::create_storage;

let storage = create_storage(&storage_config).await?;
agent.save_to(storage.as_ref(), "my-session").await?;
agent.load_from(storage.as_ref(), "my-session").await?;
```

---

## Evaluation Runner

Rust hosts can invoke the same evaluation runner used by `ai-agents-cli eval`. This is useful for embedding scenario checks into custom CI tools or release checks.

```rust
use ai_agents::eval::{EvalRunner, EvalRunnerOptions, write_outputs};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let output = PathBuf::from("target/eval/mocked/basic/simple_chat_mocked");
    let options = EvalRunnerOptions {
        agent: Some(PathBuf::from("examples/yaml/basic/simple_chat.yaml")),
        output: output.clone(),
        ..Default::default()
    };

    let runner = EvalRunner::from_file("examples/eval/mocked/basic/simple_chat_mocked.yaml", options)?;
    let result = runner.run().await?;
    write_outputs(&result, &output, true)?;

    assert_eq!(result.failed, 0);
    Ok(())
}
```

Core types are re-exported under `ai_agents::eval`: `EvalSuite`, `EvalSettings`, `ScenarioBudget`, `EvalRunner`, `EvalRunnerOptions`, `EvalResult`, `ScenarioResult`, `ScenarioStatus`, `FixturesConfig`, `TurnEvidence`, `ToolExecutionRecord`, `LLMJudge`, `JudgeConfig`, `ResetOptions`, and `ResetProfile`.

The runner builds agents through `AgentBuilder`, so the same YAML features used by the CLI are available. Eval fixtures can replace LLMs and tools for deterministic tests, and reports can be written with `write_outputs()`.

Rust options mirror the CLI: `parallel` enables scenario-concurrent runs when isolation allows it, `observability` attaches a safe default overlay, `llm_mode` and `cassette` can force record/replay/real LLM behavior, and streaming turns are collected before assertions run. For each turn, eval captures an observability cursor before execution. `TurnObservabilityEvidence` builds that turn's report, trace ID, and span IDs only from post-cursor events that remain retained in the manager's rolling metrics window; it does not recover evicted events. This does not change `ObservabilityManager::generate_report()`, `raw_events()`, or exporter behavior, which remains a rolling global view. Default JSON outputs redact input, response, and string assertion details while omitting raw turn evidence and response metadata.

---

## Observability

When `observability.enabled: true` is present in YAML, the builder attaches an `ObservabilityManager` to the runtime. You can read aggregate reports, inspect metrics, export configured files from Rust, or provide your own shared manager when multiple agents should report into one trace and metrics window. These manager and exporter APIs intentionally use the current rolling global window; turn-local cursor slicing is an eval evidence behavior, not a change to the public manager contract.

```rust
use ai_agents::{Agent, AgentBuilder, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let agent = AgentBuilder::from_yaml_file("agent.yaml")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .auto_configure_mcp().await?
        .auto_configure_spawner().await?
        .build()?;

    agent.chat("Hello").await?;

    if let Some(obs) = agent.observability() {
        let report = obs.generate_report();
        println!("LLM calls: {}", report.summary.total_llm_calls);
        println!("Tokens: {}", report.summary.total_tokens);
        println!("Cost: ${:.6}", report.summary.total_cost_usd);
        obs.export().await?;
    }

    Ok(())
}
```

You can also provide a manager programmatically when building an agent:

```rust
use ai_agents::observability::{ObservabilityConfig, ObservabilityManager};
use std::path::Path;
use std::sync::Arc;

let mut config = ObservabilityConfig::default();
config.enabled = true;
config.export.write_report = true;

// YAML-loaded agents resolve observability.cost.pricing_file automatically.
// Rust-provided configs should call this helper when pricing_file is set.
let config = config.with_pricing_file_loaded(Some(Path::new(".")))?;
let manager = ObservabilityManager::new(config);

let agent = AgentBuilder::new()
    .system_prompt("You are helpful.")
    .llm(llm)
    .observability(Arc::clone(&manager))
    .build()?;
```

`ObservabilityConfig::with_pricing_file_loaded()` accepts JSON or YAML maps shaped like `cost.pricing`. Inline `cost.pricing` values override entries loaded from the file. When `base_dir` is `Some`, relative paths are resolved from that directory; when it is `None`, they resolve from the process working directory.

`ObservabilityManager::generate_report()` drains pending events before returning an `ObservabilityReport`. Use `report.by_purpose` to compare operation costs such as `main_response`, `process_detect`, `disambiguation_clarification`, `facts_extraction`, or `orchestration_aggregation`. `ObservabilityManager::export()` writes the configured report, CSV aggregate, JSONL or JSON raw events, and Prometheus text files.

`ObservabilityManager::render_prometheus()` returns Prometheus text exposition format as a string. The framework does not start a scrape server for you yet. To connect Prometheus in a Rust host, expose that string from your own `/metrics` HTTP route or write it to a `.prom` file for the node_exporter textfile collector.

Key exported types live under `ai_agents::observability`: `ObservabilityConfig`, `ObservabilityManager`, `ObservabilityReport`, `AggregatedMetrics`, `ObservationEvent`, `ObservationPurpose`, `SpanContext`, `ObservedLLMProvider`, `ObservedTool`, `ObservabilityHooks`, `with_observation_context`, `with_observation_purpose`, and `with_updated_observation_context()`.

### Extending Observability

Rust hosts can extend observability at a few different layers:

| Layer | Use when | Types |
|-------|----------|-------|
| LLM wrapper | You want an LLM provider measured outside the normal builder path | `ObservedLLMProvider` |
| Tool wrapper | You want a tool measured outside the normal builder path | `ObservedTool` |
| Lifecycle hooks | You want state, HITL, memory, persona, facts, relationship, or orchestration events | `ObservabilityHooks` with `CompositeHooks` |
| Task-local context | You spawn tasks or run nested work and need trace continuity | `with_observation_context`, `with_observation_purpose`, `SpanContext` |
| Shared manager | You want multiple agents to report into one metrics window | `AgentBuilder::observability(manager)` |

Some internal feature crates avoid depending on `ai-agents-observability` directly to prevent dependency cycles. For those cases, the runtime uses small observer adapter traits, such as process-stage and clarification observers, then implements those observers with observability in `ai-agents-runtime`.

---

## Full API Reference

This page covers the most common patterns. For the complete API - every struct, enum, trait, and function - see the auto-generated docs:

📖 **[docs.rs/ai-agents](https://docs.rs/ai-agents/latest/ai_agents/)**

---

## Next Steps

- **[Getting Started](@/docs/getting-started.md)** - quick install and first agent
- **[CLI Guide](@/docs/cli.md)** - run agents from the command line
- **[LLM Providers](@/docs/providers.md)** - setup for all 12 supported providers
- **[YAML Reference](@/docs/yaml-reference.md)** - the complete agent spec
- **[Built-in Tools](@/docs/built-in-tools.md)** - built-in schemas, outputs, policy, and host requirements

+++
title = "Concepts"
weight = 7
template = "docs.html"
description = "Core concepts and architecture of AI Agents Framework."
+++

<!--# Concepts-->

This page explains how the framework is organized and how its pieces fit together. Each section gives you enough context to understand the big picture - for full configuration details, see the [YAML Reference](@/docs/yaml-reference.md).

---

## Architecture


The dependency layers flow in one direction:

1. **Core** (`ai-agents-core`) - shared dependency-light types, error types, and trait definitions used across the workspace.
2. **Feature crates** - each layer builds on core: `ai-agents-llm`, `ai-agents-memory`, `ai-agents-tools`, `ai-agents-state`, `ai-agents-skills`, `ai-agents-context`, `ai-agents-process`, `ai-agents-reasoning`, `ai-agents-recovery`, `ai-agents-hitl`, `ai-agents-hooks`, `ai-agents-disambiguation`, `ai-agents-storage`, `ai-agents-template`, `ai-agents-persona`, `ai-agents-facts`, `ai-agents-relationships`, `ai-agents-observability`, and `ai-agents-eval`.
3. **Runtime** (`ai-agents-runtime`) - wires every feature crate together into a running agent loop. Also contains the dynamic agent spawner module.
4. **Facade** (`ai-agents`) - provides a reviewed, curated user-facing surface for normal embedding and host integration.
5. **CLI** (`ai-agents-cli`) - a binary crate providing the `ai-agents-cli` command.

Most applications need only `ai-agents`. A few low-level adapter contracts, such as custom web-fetch transport and resolver types or eval-only static provider helpers, intentionally require their owning subcrate; the [Rust API guide](@/docs/rust-api.md) identifies those boundaries.

```yaml
# Conceptual layer diagram
layers:
  facade: ai-agents           # one crate to rule them all
  runtime: ai-agents-runtime   # orchestrates the agent loop
  features:
    - ai-agents-llm
    - ai-agents-memory
    - ai-agents-tools
    - ai-agents-state
    - ai-agents-skills
    - ai-agents-context
    - ai-agents-process
    - ai-agents-reasoning
    - ai-agents-recovery
    - ai-agents-hitl
    - ai-agents-hooks
    - ai-agents-disambiguation
    - ai-agents-storage
    - ai-agents-template
    - ai-agents-persona
    - ai-agents-facts
    - ai-agents-relationships
    - ai-agents-observability
    - ai-agents-eval
  core: ai-agents-core         # shared traits and types
```

---

## v1 Scope and Support

The v1 scope includes YAML-first construction, blocking and streaming execution, strict specs, state, skills, process, context, explicit tool grants and final authorization, memory, capability-aware persistence, facts and relationships, spawning and orchestration, evaluation, observability, provider adapters, CLI/TUI, and opt-in runtime optimization.

Support maturity is separate from whether a component is shipped:

| Tier | v1 meaning | Components |
|---|---|---|
| **Stable** | Normal v1 SemVer applies; compatibility changes are recorded in the changelog. | Builder and blocking chat, strict YAML, state/skills/process/context, built-in tool authorization, in-memory and compacting memory. |
| **Supported** | Maintained within documented boundaries; external service quality, availability, scheduling, and operational tuning may evolve. | Streaming, evaluation and observability, provider adapters and MCP, persona, file and SQLite storage, facts and relationships, spawning and orchestration, runtime optimization. |
| **Experimental** | Explicitly opt-in; minor releases may make compatibility changes with release notes, without permitting silent data loss or safety bypass. | Redis snapshot storage and Noop storage. |
| **Future Work** | Not shipped as a usable Rust or YAML contract. | Generalized Autonomy Runner, retrieval/evidence/RAG, generalized background scheduling, Python bindings. |

---

## Agent Lifecycle

Every agent goes through four stages, whether you run it from the CLI or embed it in Rust code.

1. **Define** - You describe common agent behavior in YAML or build it programmatically with `AgentBuilder`. External skills, child files, templates, MCP services, and host integrations can complement the root spec.
2. **Build** - `AgentBuilder` parses the spec, connects to LLM providers, registers tools, initializes memory, and validates the full configuration. The result is a ready-to-run `Agent` instance.
3. **Chat** - Calling `agent.chat()` (or `agent.prompt()`) sends user input through the process pipeline, into the LLM, through any tool calls or skill executions, and back out as a response. This loop repeats until the agent produces a final answer or hits the iteration limit.
4. **Persist** - Optionally, the session (conversation history, state, context) can be saved to storage and restored later. This lets you build long-running assistants that pick up where they left off.

```yaml
# Minimal agent - all four stages in action
name: LifecycleDemo
system_prompt: "You are a helpful assistant."
llm:
  provider: openai
  model: gpt-5.4-nano
memory:
  type: in-memory
  max_messages: 100
storage:
  type: sqlite
  path: ./sessions.db
```

---

## The YAML Spec

Agent behavior is YAML-first, not YAML-only. The runtime/spec layer parses the root document into `AgentSpec`, while referenced skills, child files, templates, MCP services, and Rust host integrations remain explicit external dependencies.

The major sections of a spec are: identity (`name`, `system_prompt`), LLM configuration (`llm` / `llms`), tools (including MCP servers), skills, state machine, context sources, memory, storage, process pipeline, reasoning, reflection, disambiguation, error recovery, HITL, spawner, and metadata.

You don't need all of them. A minimal spec has three conceptual fields - a name, a system prompt, and an LLM block - rendered as a small five-line document. Everything else is opt-in. The framework applies sensible defaults for anything you leave out.

For the full list of every field and its options, see the [YAML Reference](@/docs/yaml-reference.md).

```yaml
# The smallest valid agent spec
name: MinimalAgent
system_prompt: "You are a helpful assistant."
llm:
  provider: openai
  model: gpt-5.4-nano
```

---

## LLM System

An agent can use multiple named LLMs in the `llms` map. The `llm` selector assigns `default` (main chat) and optional `router` aliases. Evaluator, judge, and summarizer aliases are configured by the subsystems that use them.

If the primary LLM fails, the framework can automatically fall back to another named LLM - no manual retry logic needed. You configure this in the `error_recovery.llm` section. Any supported provider works (OpenAI, Anthropic, Google, Ollama, and 8 more), and you can mix providers freely within a single agent.

```yaml
llms:
  default:
    provider: openai
    model: gpt-5.4-mini
    temperature: 0.7
  router:
    provider: openai
    model: gpt-5.4-nano
  fallback:
    provider: ollama
    model: llama3

llm:
  default: default
  router: router

error_recovery:
  llm:
    on_failure:
      action: fallback_llm
      fallback_llm: fallback
```

---

## State Machine

Agents can have a hierarchical state machine that controls behavior. Each state can override the system prompt, available tools, skills, and reasoning mode - so the agent acts differently depending on where it is in the conversation.

Transitions between states happen in two ways. **Condition-based** transitions use a `when` clause that the LLM evaluates each turn ("when the user has provided their email"). **Guard-based** transitions check context values deterministically without an LLM call. You can also define global transitions that apply from any state, and sub-states for nested workflows.

States support lifecycle actions (`on_enter`, `on_reenter`, `on_exit`) for setting context, and `extract` for pulling structured data from user input. The runtime classifies entry from transition history captured before the transition: the first arrival at a state runs `on_enter`, while a later return runs `on_reenter` when configured.

```yaml
tools:
  - calculator
  - web_fetch

states:
  initial: greeting
  fallback: confused
  states:
    greeting:
      prompt: "Welcome the user warmly."
      transitions:
        - to: helping
          when: "the user has stated what they need"
    helping:
      prompt: "Help the user with their request."
      tools: [calculator, web_fetch]
      transitions:
        - to: closing
          when: "the user's issue is resolved"
    closing:
      prompt: "Wrap up and say goodbye."
    confused:
      prompt: "Ask the user to clarify."
      transitions:
        - to: helping
          when: "the user has clarified their request"
```

---

## Skills

A skill is a reusable workflow that bundles prompts and tool calls into a named, triggerable unit. Skills are activated by LLM-based intent routing - the agent recognizes the user wants something that matches a skill's trigger description, and runs it.

Each skill has one or more steps. A step can be a `prompt` (send text to the LLM), a `tool` call (run a specific tool with arguments), or a combination. This lets you build multi-step recipes like "fetch data, then analyze it, then summarize."

Skills are **stateless and single-shot**: the executor runs each step as an isolated LLM call with no conversation history. Step prompts must use `{{ user_input }}` to access the user's message. After a skill finishes, the next user message goes through normal routing — it does not return to the skill. See the [YAML Reference](@/docs/yaml-reference.md#skills) for template variables and examples.

Skills can define their own reasoning and reflection settings independently of the agent-level defaults. You can also put skills in external `.skill.yaml` files and reference them by path.

```yaml
tools:
  - datetime
  - random

skills:
  - id: daily_briefing
    description: "Get a daily briefing with time and a fun fact."
    trigger: "user asks for a daily briefing or morning update"
    steps:
      - tool: datetime
        args:
          operation: now
      - tool: random
        args:
          operation: integer
          min: 1
          max: 100
      - prompt: |
          Using the current time and the random number,
          create a short, fun daily briefing for the user.
```

---

## Tools

Tools give the agent the ability to act - call APIs, read files, inspect repositories, ask the user follow-up questions, keep a todo list, do math, manipulate data, edit files, review patches, and run controlled validation commands. The framework ships with built-in tools such as `calculator`, `datetime`, `echo`, `glob`, `grep`, `file_read`, `file_write`, `file_edit`, `file_list`, `file_info`, `patch`, `copy_path`, `move_path`, `delete_path`, `git_status`, `git_diff`, `diagnostics`, `ask_user`, `todo`, `sleep`, `web_fetch`, `web_search`, `command`, `json`, `http`, `file`, `text`, `template`, `math`, and `random`.

For external tools, you can connect any MCP (Model Context Protocol) server. MCP tools support `stdio` transport, startup timeouts, security restrictions, and function-level views that let you expose subsets of an MCP server's capabilities to different states.

Tool availability is explicit and fail-closed. Omitted top-level `tools:` means no LLM-callable tools, and `tools: []` also means no tools. The top-level `tools:` list is the normal maximum grant; state-level `tools` can only narrow the effective grant, never widen it to the full registry. Explicit feature flags such as `spawner.management_tools`, `spawner.orchestration_tools`, and `persona.evolution.allow_llm_evolve` are also grants for their generated tools. Security policies, HITL, recovery, observability, and eval evidence use canonical tool IDs after aliases and display names are resolved. `ask_user` is separate from HITL approval: it asks a preference or clarification question through a host question handler instead of approving a risky action.

LLM `tool_choice` controls provider selection only after that grant is known. Omission keeps the existing prompt-JSON automatic-selection behavior; `auto`, `required`, `{ specific: canonical_id }`, and `none` opt into explicit selection. Required or specific choice can narrow visible tools but cannot register, grant, or authorize one. Native calls and prompt-fallback calls both enter the same shared executor, and unsupported native required/specific selection gets at most one budgeted corrective retry.

The split local-file tools block raw `.git` paths, while `git_status` and `git_diff` expose bounded repository inspection. Filesystem mutation tools execute when `dry_run` is omitted. Set `dry_run: true` to request validation or a preview without applying the mutation. Every actual mutation still requires applicable write policy and either approval or an explicit trusted-policy exemption. Results expose `mutation_performed`, while `created`, `overwritten`, `copied`, `moved`, and `deleted` are true only when that effect occurred. Path policy resolves existing paths to prevent allowed-root and blocked-subtree symlink bypasses while the checked topology remains unchanged, recursive copy rejects symlinks and destinations nested under the source, and read-before-write version checks apply when configured. The v1 contract assumes a host-owned workspace without concurrent untrusted external path replacement. Non-concurrency-safe path operations use one conservative shared mutation lock; unbound side effects use a separate shared lock. Parent and spawned runtimes share the lock table, but these controls are not an OS filesystem sandbox. After HITL approval, the runtime resolves the final tool once, requires the resolved implementation to match the reviewed tool, reapplies current argument caps, and rechecks scope, policy, emergency control, provider availability, and approval binding before locking. After lock acquisition, a changed policy/runtime generation or failed atomic rate admission prevents invocation. The default `web_fetch` transport disables proxies and connects each request only to addresses that passed its DNS/IP checks; custom low-level transports and host egress remain responsible for preserving that boundary. The `command` tool is not a shell: it runs exact allowlisted argv commands inside explicit `working_dirs`, starts from an empty environment, filters env through policy, redacts argv evidence, and bounds output. Missing host support for `diagnostics`, `command`, or `web_search` is rejected before implementation invocation and recorded with `executed: false`. `ask_user` is different: its fallback implementation executes and returns a structured unavailable response, using a configured default when present. See [Built-in Tools](@/docs/built-in-tools.md).

```yaml
tools:
  - calculator
  - glob
  - grep
  - file_read
  - git_status
  - web_fetch
  - ask_user
  - todo
  - name: filesystem
    type: mcp
    transport: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "./"]
    views:
      fs_read:
        functions: [read_file, list_allowed_directories, search_files]
        description: "Read-only file access"
```

---

## Process Pipeline

The process pipeline lets you declare input and output processing stages that run before and after the LLM. Semantic stages such as language detection, entity extraction, LLM-backed sanitization, and quality checks use model understanding. Structural normalization, length checks, formatting, ordering, and safety enforcement remain deterministic.

**Input stages** run in order: `normalize` (trim, collapse whitespace), `detect` (language, sentiment, intent), `extract` (entities into context), `sanitize` (PII removal), `validate` (length, content rules). **Output stages** run after the LLM responds: `validate` (quality scoring), `transform` (tone adjustment), `sanitize` (PII masking), `format` (templates, footers).

LLM-backed stages can target a named alias, typically the lightweight `router`, and store results in context. Deterministic stages do not require an LLM.

```yaml
process:
  input:
    - type: normalize
      config:
        trim: true
        collapse_whitespace: true
    - type: detect
      config:
        llm: router
        detect: [language, sentiment]
        store_in_context:
          language: detected_language
          sentiment: detected_sentiment
    - type: extract
      config:
        llm: router
        schema:
          email:
            type: string
            description: "User's email address"
        store_in_context: extracted
  output:
    - type: sanitize
      config:
        llm: router
        pii:
          action: mask
          types: [email, phone]
```

---

## Memory

Memory controls what the agent remembers across turns. **In-memory** mode keeps a rolling window of recent messages - simple and fast. **Compacting** mode adds auto-summarization: when recent messages reach a threshold, an eligible older prefix is compressed while the requested recent tail stays verbatim. If `max_recent_messages` would leave no useful batch at the threshold, the protected tail is clamped so compression still makes configured batch progress. Compression and other memory mutations are serialized, and snapshots preserve summary accounting.

Token budgeting makes sure memory fits within LLM context limits. You set a total token budget and allocate integer token counts to summaries, recent messages, facts, and relationships. If the budget overflows, the framework either truncates the oldest content or re-summarizes, depending on your chosen strategy.

Memory is stored in-process during a session. For persistence across sessions, pair memory with a storage backend. File and Redis implement snapshot save/load/list/delete only. SQLite also implements generic session metadata, filtering, expiry cleanup, actor facts, actor relationships, and atomic actor-data deletion. Unsupported operations return a typed capability error rather than an empty result or successful no-op. Redis has backend-native key TTL, but that does not provide the generic metadata or expiry-cleanup capabilities.

Spawned agents can share a backend through `NamespacedStorage`. The wrapper derives safe capabilities from the inner backend and uses reversible flat encoded keys, but it never forwards backend-global expiry cleanup because that operation cannot be scoped to one child.

```yaml
memory:
  type: compacting
  max_recent_messages: 10
  compress_threshold: 15
  summarize_batch_size: 5
  summarizer_llm: router
  token_budget:
    total: 4000
    allocation:
      summary: 1000
      recent_messages: 2500
      facts: 500
      relationships: 0
    overflow_strategy: truncate_oldest
    warn_at_percent: 85
```

---

## Context System

Context provides dynamic data that gets injected into the agent's system prompt at render time. The system prompt is a Jinja2 template, and context values are its variables.

Sources include: **runtime** (passed in by the caller), **builtin** (datetime, session info, agent metadata), **env**, **file**, optional feature-gated **HTTP** JSON sources, and **callback** providers registered by the host. Each source has a refresh policy - `once` (load at startup), `per_session` (reload each session), or `per_turn` (refresh every turn).

Context values are available in the system prompt template, in state prompts, and in process pipeline stages. The state machine can also write to context via `on_enter` actions and `extract` blocks.

```yaml
context:
  user:
    type: runtime
    required: true
    schema:
      name: string
      language: string
    default:
      name: "there"
      language: "en"
  time:
    type: builtin
    source: datetime
    refresh: per_turn

system_prompt: |
  You are a helpful assistant.
  The user's name is {{ context.user.name }}.
  Current time: {{ context.time.time }}.
  Speak in {{ context.user.language }}.
```

---

## Reasoning & Reflection

Two kinds of "thinking" exist in this framework:

**Model-level native thinking** is controlled by the LLM config fields `reasoning`, `reasoning_effort`, and `reasoning_budget_tokens`. Thinking models (o3, o4-mini, gpt-5.4, Claude with extended thinking) reason internally via API-level reasoning tokens. This is the real thing - the model architecture is trained to reason in a dedicated phase.

```yaml
llm:
  provider: openai
  model: gpt-5.4-mini
  reasoning: true
  reasoning_effort: medium
  reasoning_budget_tokens: 8000
```

**Framework-level reasoning modes** are controlled by `reasoning.mode`. Five modes are available:

- **none** - answer directly, no extra thinking.
- **cot** (chain-of-thought) - prompt injection that asks the LLM to think step by step. The framework appends an instruction to the system prompt and parses `<thinking>` tags from the output. This is useful for non-thinking models. For thinking models it is redundant - the model already reasons natively.
- **react** - prompt injection variant that structures the tool-use loop as Thought -> Action -> Observation. Same caveat as CoT - thinking models do this naturally.
- **plan_and_execute** - real orchestration. The framework generates a structured plan (JSON steps), executes each step using tools/skills/LLM, and synthesizes the result. This is genuinely different from CoT/ReAct and adds value regardless of model type.
- **auto** - a judge LLM classifies each input and picks the best mode.

For thinking models, prefer native thinking (`llm.reasoning: true`) over `mode: cot`. A future version will wire `mode: cot` to native thinking when the model supports it, with prompt injection as fallback.

When reasoning is active (cot, react, plan_and_execute, or auto), the iteration loop is capped at the lower of the agent-level `max_iterations` and `reasoning.max_iterations`.
This keeps the reasoning-specific cap as a tighter limit inside the agent's overall safety cap.

For `plan_and_execute` mode, a plan-level reflection loop retries failed plans.
When `planning.reflection.enabled` is true and a step fails, the runtime checks `on_step_failure` to decide whether to replan, abort, or skip.
Multi-step plan output is synthesized into a coherent response via the LLM rather than returning only the last step's raw result.

Reflection adds self-evaluation. After producing an answer, the agent scores it against criteria you define (accuracy, completeness, tone). The LLM must say PASS and report a confidence score at or above `pass_threshold` for the evaluation to succeed. If it fails, the agent retries. Both reasoning and reflection can be overridden at the state or skill level.

When a state transition fires mid-turn and the target state has a `reasoning:` override (or the agent-level mode is non-`none`), the runtime re-enters the full dispatch path for the new state in the same turn. CoT/ReAct prompt injection, auto-detection, and the plan-and-execute handler all activate immediately - the user does not need to send another message for the new state's reasoning config to take effect.

```yaml
reasoning:
  mode: auto
  judge_llm: router
  output: tagged
  max_iterations: 5

reflection:
  enabled: auto
  evaluator_llm: router
  max_retries: 2
  pass_threshold: 0.7
  criteria:
    - "Answer is factually accurate"
    - "Answer fully addresses the question"
    - "Tone is professional and clear"
```

---

## Agent Persona

The `persona:` section defines structured identity for an agent - name, role, personality traits, speaking style, goals, secrets, and evolution rules. Persona separates *who the agent is* from *what it does* (system prompt) and *what it remembers* (memory).

Persona is prepended to the system prompt automatically. It survives `prompt_mode: replace` in state machines, so an NPC guard in a "patrol" state still knows its name and personality even when the state prompt is fully replaced.

**Identity** gives the agent a name, role, optional backstory, and affiliation. **Traits** define personality descriptors, values, fears, and speaking style - all included verbatim in the LLM prompt. **Goals** list what the agent pursues; hidden goals are excluded from the prompt but readable by application code. **Secrets** are information the agent withholds until context conditions are met (e.g., trust level reaches a threshold). Conditions use the same typed matchers as state machine guards (`eq`, `gte`, `in`, `exists`, etc.).

**Evolution** lets persona fields change over time. When `evolution.enabled` is true, Rust code and hooks can call `evolve()` on whitelisted fields. When `evolution.allow_llm_evolve` is also true, a `persona_evolve` tool is auto-registered so the LLM itself can trigger changes (double opt-in for safety). All mutations are validated against `mutable_fields` and optionally recorded in an audit trail.

```yaml
persona:
  identity:
    name: "Captain Elira"
    role: "Harbor Guard Captain"
    backstory: "Former soldier who served in the Eastern Campaign."
    affiliation: "Harbor Watch"
  traits:
    personality: [disciplined, suspicious, loyal]
    values: [duty, order, justice]
    speaking_style: "formal military cadence, short clipped sentences"
  goals:
    primary: [protect_harbor, investigate_smuggling]
    hidden: ["Find the spy within the Watch"]
  secrets:
    - content: "Investigating a smuggling ring"
      reveal_conditions:
        context:
          relationships.current_actor.trust:
            gte: 0.8
  evolution:
    enabled: true
    mutable_fields: [traits.personality, traits.speaking_style]
    track_changes: true
```

---

## Actor Memory & Key Facts

Agents can remember facts about the people they talk to across sessions. The `actor_memory` and `facts` blocks inside `memory:` enable this. An **actor** is any entity the agent interacts with - a customer, a game player, another agent. The framework uses `actor_id` as a universal identifier.

**How it works:** After each conversation turn, the framework runs a fast LLM call to extract structured facts (preferences, context, decisions, agreements) from the messages. Durable facts require a backend with the `ActorFacts` capability, currently SQLite among the built-ins; file and Redis remain snapshot-only. Facts are keyed by `(agent_id, actor_id)`. When the same actor returns in a new session, their facts are loaded and injected into the system prompt via the `{{ actor_facts }}` Jinja2 variable.

**Actor identification** can be explicit (set via `--actor` CLI flag or `set_actor_id()` API) or context-based (read from a context path like `player.id`). Context-based identification is useful for NPC agents where the hosting application sets the current player before each turn.

**Fact content is always English** regardless of conversation language, so cross-language deduplication works consistently. The LLM handles translation during extraction. Facts are ranked by `salience * confidence` for priority - when the fact count exceeds `max_facts`, lowest-priority facts are evicted.

**Custom categories** extend the built-in set (`user_preference`, `user_context`, `decision`, `agreement`) with domain-specific types. An NPC guard might track `suspicion` and `favor` categories; a medical assistant might track `medical_history`.

```yaml
memory:
  actor_memory:
    enabled: true
    identification:
      method: from_context
      context_path: user.id
  facts:
    enabled: true
    extractor_llm: router
    auto_extract: true
    categories: [user_preference, user_context, decision]
    custom_categories:
      - name: suspicion
        description: "Suspicious behavior observed"
    max_facts: 50

system_prompt: |
  You are a helpful assistant.
  {% if actor_facts %}
  What you know about this person:
  {{ actor_facts }}
  {% endif %}
```

The `/actor` and `/facts` REPL commands let you inspect and manage facts interactively. Use `/actor set <id>` to switch actors, `/actor facts` to list facts, and `/facts extract` to trigger manual extraction.

---

## Relationship Memory

Relationship memory tracks how an agent relates to each actor over time. It is actor-scoped and general-purpose: an actor can be a customer, student, patient, player, NPC, or another agent.

Relationship memory is separate from key facts. Facts answer "what does the agent know about this actor?" Relationships answer "how does this agent currently relate to this actor?" Default dimensions are `trust`, `sentiment`, `familiarity`, and `rapport`, and applications can define custom dimensions such as `motivation`, `suspicion`, `reliability`, or `openness`.

After each successful turn, a router LLM evaluates recent messages and proposes small dimension deltas. The runtime validates confidence, clamps deltas, clamps final scores to each dimension range, stores notable relationship events, and persists the relationship by `(agent_id, actor_id)` when storage supports `ActorRelationships`. SQLite is currently the only built-in backend with that capability.

Relationship values are injected into context under `relationships.current_actor.*`, making them available to persona secrets, state guards, tool conditions, and templates. The formatted prompt text is available as `{{ relationship_memory }}`. By default the model is one-sided (`agent_to_actor`), but `model: two_sided` also tracks `perceived_actor_to_agent` and derives read-only `mutual` scores while keeping shortcut paths such as `relationships.current_actor.trust` compatible. Automatic evaluator updates only write the two stored perspectives: `agent_to_actor` and `perceived_actor_to_agent`.

```yaml
memory:
  relationships:
    enabled: true
    model: one_sided
    dimensions:
      - trust
      - sentiment
      - familiarity
      - rapport
    auto_update:
      enabled: true
      llm: router
    injection:
      enabled: true
      format: summary
      prompt_variable: relationship_memory
```

Persona secrets can use relationship scores directly:

```yaml
persona:
  secrets:
    - content: "Confidential context"
      reveal_conditions:
        context:
          relationships.current_actor.trust:
            gte: 0.8
```

---

## Dynamic Agent Spawning

A parent agent can create and manage child agents at runtime using the spawner system. This enables patterns like a game master that spawns NPC agents on demand, or a team manager that creates specialist agents for different tasks.

The spawner supports three creation methods: raw YAML strings, `AgentSpec` objects, and named Jinja2 templates. Templates can be defined inline in the YAML or loaded from separate files. A central `AgentRegistry` tracks all spawned agents and provides inter-agent messaging - one agent can send a message to another and receive its response, or broadcast to all agents at once.

When `spawner:` is present in the YAML, the framework automatically registers four management tools: `spawn_agent` (create a new agent from a description or template), `send_agent_message` (talk to another agent), `list_agents` (see all registered agents), and `remove_agent` (remove an agent from the registry). Grant them with `spawner.management_tools`, or list them under top-level `tools:` when the parent LLM should call them.

LLM and storage sharing are opt-in through `shared_llms` and `shared_storage`. Shared LLM mode treats the inherited parent registry as authoritative and requires every alias referenced by a child. Shared storage wraps each child in `NamespacedStorage`, which uses reversible flat encoded keys and forwards only capabilities that remain safely scoped; namespace-scoped expiry cleanup is unavailable.

Every dynamic, auto-spawned, and restored child passes the same admission checks. `max_agents` limits reserved or registered spawner-managed slots, including in-flight construction; removing a child releases its slot. `allowed_tools` checks canonical built-in IDs and rejects the complete child when its top-level tool declarations exceed the allowlist. It does not strip tools and is not an OS, network, credential, or storage sandbox. Declared auto-spawn fails the parent configuration on any child error, and active nested child spawners are rejected in v1. Management and orchestration tools remain explicit feature grants and still pass through normal tool policy, approval, locking, and observability.

```yaml
spawner:
  management_tools: true
  shared_llms: true
  max_agents: 20
  name_prefix: "npc_"
  shared_context:
    world_name: "Eldoria"
  allowed_tools:
    - echo
    - calculator
  templates:
    npc_base:
      path: ./templates/npc_base.yaml
    simple_npc: |
      name: "{{ name }}"
      system_prompt: "You are {{ name }}, a {{ role }} in {{ context.world_name }}."
      llm:
        provider: openai
        model: gpt-5.4-nano
```

---

## Multi-Agent Orchestration

Building on the spawner and registry, orchestration lets a parent agent coordinate multiple sub-agents in structured patterns. The parent agent owns the conversation with the user. Sub-agents work behind the scenes.

Five coordination patterns are available. Three have dedicated state types wired into the runtime. Two are composed from `delegate` states with different transition topologies. All five are also available as orchestration tools for LLM-decided coordination at runtime.

| Pattern | State machine (declarative) | Dynamic (tool) |
|---------|----------------------------|----------------|
| **Router** | `delegate` states + LLM-evaluated `when` transitions. No dedicated state type - the existing transition evaluator does the routing. | `route_to_agent` tool calls `orchestration::route()` |
| **Pipeline** | `pipeline` field on state definition. Sequential agent chain with Jinja2 per-stage input templates. `{{ stages.<agent_id> }}` lets any stage reference any earlier stage's output by name. Runs in one `chat()` call. Also achievable via chained `delegate` states with auto-transitions. | `pipeline_process` tool calls `orchestration::pipeline()` |
| **Concurrent** | `concurrent` field on state definition. Dedicated runtime handler runs agents in parallel and aggregates results. | `concurrent_ask` tool calls `orchestration::concurrent()` |
| **Group Chat** | `group_chat` field on state definition. Dedicated runtime handler manages multi-agent conversation with turn management. Styles: `brainstorm` (free-form), `consensus` (same loop but router LLM checks agreement after each round), `debate` (structured pro/con with synthesizer agent), `maker_checker` (create-review loop). Turn order supports `round_robin`, `random`, and `llm_directed` (LLM picks one speaker at a time). | `group_discussion` tool calls `orchestration::group_chat()` |
| **Handoff** | `handoff` field on state definition. LLM-directed agent-to-agent control transfer with structured JSON decisions. Runs in one `chat()` call. Also achievable via `delegate` states with peer-to-peer transitions. | `handoff_conversation` tool calls `orchestration::handoff()` |

All five patterns have dedicated fields on `StateDefinition` (`delegate`, `concurrent`, `group_chat`, `pipeline`, `handoff`). Router composes from `delegate` + transitions. Pipeline and Handoff can also be composed from `delegate` states but have dedicated state types for single-state convenience.

A `delegate` state forwards user messages to a registry agent instead of processing them locally. The parent's transition evaluator continues watching the delegate's responses:

```yaml
spawner:
  shared_llms: true
  auto_spawn:
    - id: billing
      agent: agents/billing_agent.yaml
    - id: technical
      agent: agents/technical_agent.yaml

states:
  initial: triage
  states:
    triage:
      prompt: "Determine what the user needs."
      transitions:
        - to: billing_help
          when: "User has a billing question"
        - to: tech_help
          when: "User has a technical issue"
    billing_help:
      delegate: billing
      transitions:
        - to: triage
          when: "Issue resolved"
    tech_help:
      delegate: technical
      transitions:
        - to: triage
          when: "Issue resolved"
```

For dynamic orchestration, set `orchestration_tools: true` in the spawner section. The LLM can then call `route_to_agent`, `pipeline_process`, `concurrent_ask`, `group_discussion`, and `handoff_conversation` at runtime without a predefined state graph.

Delegate states support a `delegate_context` mode (`input_only`, `summary`, `full`) that controls what conversation history reaches the sub-agent. With `input_only` (default) only the user's current message is forwarded. With `full` the last 20 messages are included as conversation history. With `summary` the parent uses its router LLM to summarize the conversation into 2-3 sentences before forwarding.

The same context enrichment is available for all orchestration patterns via the `context_mode` field. Set `context_mode: summary` or `context_mode: full` on any `concurrent`, `group_chat`, `pipeline`, or `handoff` block to forward parent conversation history to sub-agents. The enrichment runs before `input` template rendering, so `{{ user_input }}` in templates contains the history-enriched text. When omitted, the default is `input_only` which preserves the original behavior.

Orchestration uses normal runtime services for policy, approvals, recovery, hooks, memory, actor context, and final response handling. Pattern handlers return a final orchestration response; v1 does not promise incremental child-agent token forwarding through the parent stream.

Actor context is forwarded structurally through registry sends and orchestration calls. When a parent turn has an actor, sub-agents receive `interaction.origin_actor_id` and `interaction.sender_agent_id` in prompt context, and actor-scoped facts and relationship memory use the original actor by default. This avoids relying on text prefixes such as `[From agent]` for memory identity.

When a state transition fires mid-turn and the target state is an orchestration state (`concurrent`, `group_chat`, `pipeline`, `handoff`, or `delegate`), the runtime detects this and re-enters the full dispatch path for the new state in the same turn. The correct orchestration handler activates immediately - the user does not need to send another message to trigger it. Up to three chained transitions are resolved this way before the runtime stops and returns the last available response.

Concurrent states use `aggregation.vote` to control how individual agent responses are aggregated into a final answer. `aggregation.vote.method` selects `majority` (default), `weighted`, or `unanimous`. `aggregation.vote.tiebreaker` selects `first` (declaration order), `random`, or `router_decides` (asks the router LLM). Weighted voting reads each `{ id, weight }` entry from `concurrent.agents`; plain string entries use weight `1.0`. The `on_partial_failure` field defaults to `proceed_with_available`, which aggregates successful responses, while `abort` fails the concurrent block when an agent fails.

After each orchestration call the runtime stores the full structured result in `context.orchestration`. Subsequent states can reference the data in prompt templates and guard conditions. The object includes a `type` field (`delegate`, `concurrent`, `group_chat`, `pipeline`, or `handoff`) plus type-specific data such as per-agent responses, the full group chat transcript, pipeline stage outputs, handoff chain events, round counts, and timing. Backward-compatible flat keys (`delegation.<id>.last_response`, `concurrent.result`, `group_chat.conclusion`, `pipeline.result`, `handoff.result`) are also set. The same structure is attached to `response.metadata["orchestration"]` for CLI and hook consumers.

For group chat brainstorm and consensus styles, `response.content` contains the full formatted transcript (`[Round N] speaker: message`) rather than only the last speaker's final line. Debate and maker-checker styles are unaffected because they already produce a synthesized conclusion or final draft. Users on the `InMemoryStore` backend should be aware that long transcripts consume a ring-buffer slot; `CompactingMemory` handles overflow via summarization. For the maker-checker style, `on_max_iterations` controls what happens when the review loop hits its limit: `accept_last` uses the final draft as-is, `escalate` forwards to a human or parent agent, and `fail` returns an error.

Group chat supports three turn-order methods via `manager.method`. `round_robin` (default) cycles through all participants in declaration order. `random` shuffles each round. `llm_directed` uses the router LLM to pick one speaker at a time after seeing the latest message, capping at `participants.len()` speakers per round for consistent stall detection. `llm_directed` requires a router LLM; the builder returns a config error if none is configured. When `manager.agent` is set to a registry agent id, that agent takes over termination decisions and speaker selection instead of the built-in logic, allowing fully custom orchestration within the group chat loop.

Handoff decisions use structured JSON. The evaluator LLM returns `{"action": "agent_id_or_stay", "confidence": 0.0-1.0, "reason": "..."}`. The runtime parses JSON first (handling markdown code blocks and preamble text), then falls back to fuzzy text matching if JSON extraction fails. This makes handoff robust to variations in LLM output formatting.

When `auto_spawn` is configured, the builder validates that every agent referenced by an orchestration state (`delegate`, `concurrent`, `group_chat`, `pipeline`, `handoff`) was successfully spawned. Missing agents produce a clear build-time error listing each unresolved reference, the state that needs it, and the orchestration pattern involved.

See [YAML Reference - Orchestration States](@/docs/yaml-reference.md#orchestration-states) for the complete field reference.

---

## Evaluation as Regression Testing

Evaluation runs YAML agents through the normal runtime path and checks declared assertions against structured evidence. This makes it useful for CI smoke tests, release checks, and regression suites for state machines, tools, memory, orchestration, and safety behavior.

A suite is separate from the agent YAML. The suite chooses fixtures, scenarios, turns, assertions, retries, filters, and output directory. Mocked LLM fixtures make deterministic no-key tests possible, while optional judge assertions use an LLM only for semantic quality checks.

```yaml
name: Basic Chat Eval
agent: ../../../yaml/basic/simple_chat.yaml
fixtures:
  llm:
    mode: mock
    responses:
      - "Hello! I can help with that."
scenarios:
  - id: hello-smoke
    turns:
      - input: Hello
        assert:
          response_not_empty: true
          response_contains: "Hello"
```

The key design is rule-based first: state, response text, context, metadata, tools, facts, relationships, persona state, orchestration metadata, and observability summaries should be checked deterministically where possible. Use `judge` or `response_semantic` only when the expected behavior is semantic and cannot be expressed as structured evidence.

Reports include Markdown, JSON, per-scenario JSONL, failure-focused Markdown, and optional JUnit XML. By default, JSON artifacts redact inputs, responses, and string assertion details, and omit raw turn evidence and response metadata. Use `redact_outputs: false` only for trusted local debugging.

---

## Safety

The framework provides layered controls for agent execution. Explicit grants, policy, and HITL are not an OS, network, credential, or deployment sandbox; the host remains responsible for isolating untrusted code and infrastructure.

**Error recovery** handles transient failures with configurable retry, exponential backoff, fallback LLMs, and fallback responses. Context overflow (too many tokens) can be handled by summarizing or truncating. Each of these is configurable per subsystem - LLM calls, tool calls, and general errors all have separate policies.

**Tool security** adds rate limits, domain/path allow and block lists, timeouts, effective limit caps, custom tool config, and confirmation requirements on a per-tool basis. The shared executor resolves the requested name to a canonical ID, checks policy with tool-declared bindings, builds `ToolExecutionContext`, and records `ToolExecutionRecord` evidence even when a call is denied or unavailable before execution. Under fail-closed policy, custom tools must expose bindings for configured path, domain, command, operation, and result-limit constraints that the shared executor cannot otherwise apply.

**HITL (Human-in-the-Loop)** lets you require human approval before sensitive operations execute. Approval messages support multiple languages, and you can scope approval rules to specific states or conditions. Timeout behavior is configurable - reject, allow, or use a default.

```yaml
error_recovery:
  llm:
    on_failure:
      action: fallback_llm
      fallback_llm: fallback
    on_context_overflow:
      action: summarize
      keep_recent: 5

tool_security:
  enabled: true
  tools:
    web_fetch:
      domains:
        allow: [api.example.com]
      max_response_bytes: 1048576
    my_search_tool:
      read_paths: [./docs]
      max_results: 25
      config:
        backend: tantivy

hitl:
  default_timeout_seconds: 30
  on_timeout: reject
  tools:
    http:
      require_approval: true
      approval_message:
        en: "Allow this HTTP request?"
        ko: "이 HTTP 요청을 허용하시겠습니까?"
```

---

## Observability & Tracing

Observability turns agent execution into structured metrics and traces. When `observability.enabled: true`, the builder creates an `ObservabilityManager`, wraps registered LLM providers and tools, and composes `ObservabilityHooks` with any user hooks already configured in Rust.

The result is broader than hook-only logging. LLM and tool wrappers see calls from normal chat, skill routing, skill prompt steps, process stages, disambiguation detection and clarification, reflection, planning, facts extraction, relationship updates, HITL localization, state actions, context extraction, and multi-agent orchestration. Lifecycle hooks add state transitions, approval events, memory compression, persona events, facts events, relationship events, and orchestration milestones.

Trace context is task-local and actor-aware. A parent agent turn creates the root trace, child agents derive child contexts with the same trace ID, and concurrent registry calls clone the context into spawned tasks. Process output can update language context before downstream main LLM events are recorded, so language aggregations reflect the current turn when detection is configured.

Purpose labels are designed for cost and latency attribution. For example, process stages can appear as `process_detect`, `process_extract`, `process_validate`, or `process_transform`, while ambiguity flows separate `disambiguation_detection` from `disambiguation_clarification`.

Privacy is safe by default. Raw prompts, responses, tool arguments, tool outputs, context values, actor facts, relationship memory, persona secrets, approval details, tags, and error text are not stored unless explicitly enabled. When raw payloads are enabled, configured keys and dotted paths are redacted recursively and text is truncated on Unicode character boundaries.

```yaml
observability:
  enabled: true
  aggregation:
    dimensions: [agent, model, purpose, language, state, status]
  privacy:
    include_prompts: false
    include_responses: false
    hash_inputs: true
  export:
    formats: [json, csv]
    path: ./observability_data/
    write_report: true
```

A Rust host can also read reports directly:

```rust
if let Some(obs) = agent.observability() {
    let report = obs.generate_report();
    println!("LLM calls: {}", report.summary.total_llm_calls);
}
```

---

## Runtime Optimization

Runtime optimization reduces avoidable response latency without changing behavior by default. The main rule is that the runtime can select a route early, but it can only commit side effects after the winning path is known.

```text
user input
  -> process input
  -> optional pre-response guard or intent transition
  -> committed state response
  -> post-turn facts and relationship maintenance
```

The lowest-risk optimization is pre-response deterministic transitions. A transition must opt in with `timing: pre_response`, and then a guard or resolved intent can prove that the current user input belongs in another state before the old state asks the LLM for a response. When that happens, the runtime commits the transition first and generates the visible answer from the new state. Pre-response transitions with natural-language `when` text remain invalid because pre-response routing must be deterministic; use `timing: parallel` for response-independent LLM transition prompts.

```yaml
runtime:
  optimization:
    enabled: true
    pre_response_deterministic_transitions: true

states:
  initial: greeting
  states:
    greeting:
      transitions:
        - to: billing
          guard:
            context:
              request.topic:
                eq: billing
          timing: pre_response
          requires_response: false
```

Post-turn facts and relationship updates are future-turn maintenance. They can run inline in the existing serial path, inline in parallel, or in the background. Background mode is eventually consistent, so actor memory tasks support `await_before_next_turn: same_actor` to wait only when the same actor continues immediately. Freshness is task-aware: fact freshness waits for fact tasks, and relationship freshness waits for relationship tasks.

```yaml
runtime:
  optimization:
    enabled: true
    post_turn:
      facts:
        mode: background
        await_before_next_turn: same_actor
      relationships:
        mode: background
        await_before_next_turn: same_actor
```

Speculative branch execution extends this safety model to independent LLM work. The runtime can overlap a main response draft with response-independent transition evaluation, pure skill selection, or auto reasoning decisions. The main draft is data until it wins. If a transition, skill, or deeper reasoning branch wins, the draft is finalized as discarded and its parsed tool calls remain inert. Plain drafts commit only when they preserve the equivalent serial path: forced reasoning modes use the serial path, and auto reasoning speculation requires enough capacity for both the draft and judge branch. If a required branch cannot reserve safely, the draft is discarded and the runtime falls back to the serial committed path instead of reporting a false no-match or no-reasoning decision.

```text
user input
  -> main draft branch + routing branch
  -> choose winner
  -> commit exactly one path
  -> finalize losing branch telemetry
```

Speculative execution is opt-in and bounded by `max_speculative_llm_calls_per_turn`. This is important because discarded or cancelled branches can still consume tokens and cost. Deterministic guard and resolved-intent checks do not consume speculative LLM budget. When the cap is lower than the number of eligible LLM branches, the runtime schedules only behavior-preserving branches and falls back to the serial committed path for skipped branch families. Observability reports expose `branch_status`, `optimization`, `commit_behavior`, `winner`, and `speculative` dimensions so eval suites and dashboards can track committed, discarded, failed, and cancelled branch work. Branch telemetry is the supported way to inspect speculative LLM work; final response hooks still fire once for the committed response.

Streaming supports a buffered routing policy for response-independent parallel state-transition routing. With `streaming_policy: buffer_until_routing_done`, output from the unresolved main branch is hidden until routing is safe. The buffer limit applies while routing is unresolved. After the transition branch misses or fails, later chunks are no longer counted against that unresolved-routing buffer. The runtime emits chunks only after the committed content is known: if the main branch wins and the raw draft still matches the committed response, buffered chunks are emitted in order; otherwise the committed content is emitted. If the transition branch wins, stale branch output is discarded and the committed route response is emitted instead. If the buffered main branch fails, branch telemetry is finalized as failed before the stream error is returned.

---

## Hooks & Extensibility

The `AgentHooks` trait gives you lifecycle callbacks: before/after chat, before/after tool calls, on state transitions, on errors, and more. Implement the trait and pass your hooks when building the agent to add logging, metrics, custom routing, or any side-effect you need.

Major extension points are trait-based and pluggable. You can provide custom implementations of `LLMProvider` (add a new model provider), `Memory` (custom memory behavior), `Tool` (any Rust function as a tool), `AgentStorage` (custom persistence), `ApprovalHandler` (custom HITL flow), and `Summarizer` (custom compression logic). Other runtime behavior remains governed by the documented builder and YAML contracts.

```yaml
# Hooks are configured in Rust code, not YAML.
# But tool and memory choices are in the spec:
tools:
  - calculator
  - datetime
memory:
  type: compacting
  summarizer_llm: router
storage:
  type: sqlite
  path: ./data/sessions.db
```

```rust
// Rust-side hook example (simplified)
use ai_agents::{AgentBuilder, hooks::AgentHooks};
use std::sync::Arc;

struct MyHooks;
impl AgentHooks for MyHooks {
    // override any lifecycle method you need
}

let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .auto_configure_mcp().await?
    .auto_configure_spawner().await?
    .hooks(Arc::new(MyHooks))
    .build()?;
```

---

## Next Steps

- **[Getting Started](@/docs/getting-started.md)** - build and run your first agent
- **[YAML Reference](@/docs/yaml-reference.md)** - every field, every option, fully documented
- **[Built-in Tools](@/docs/built-in-tools.md)** - canonical inputs, outputs, policy bindings, and host requirements
- **[Rust API](@/docs/rust-api.md)** - use the framework as a library
- **[Examples](@/examples/_index.md)** - real-world patterns and complete agent specs

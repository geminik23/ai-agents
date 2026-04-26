+++
title = "Concepts"
weight = 6
template = "docs.html"
description = "Core concepts and architecture of AI Agents Framework."
+++

<!--# Concepts-->

This page explains how the framework is organized and how its pieces fit together. Each section gives you enough context to understand the big picture - for full configuration details, see the [YAML Reference](@/docs/yaml-reference.md).

---

## Architecture


The dependency layers flow in one direction:

1. **Core** (`ai-agents-core`) - shared types, specs, error types, and trait definitions used everywhere.
2. **Feature crates** - each layer builds on core: `ai-agents-llm`, `ai-agents-memory`, `ai-agents-tools`, `ai-agents-state`, `ai-agents-skills`, `ai-agents-context`, `ai-agents-process`, `ai-agents-reasoning`, `ai-agents-recovery`, `ai-agents-hitl`, `ai-agents-hooks`, `ai-agents-disambiguation`, `ai-agents-storage`, `ai-agents-template`.
3. **Runtime** (`ai-agents-runtime`) - wires every feature crate together into a running agent loop. Also contains the dynamic agent spawner module.
4. **Facade** (`ai-agents`) - re-exports everything behind a single dependency so library users only add one crate.
5. **CLI** (`ai-agents-cli`) - a binary crate providing the `ai-agents-cli` command.

You never need to depend on individual crates directly. Just add `ai-agents` to your `Cargo.toml` and everything is available.

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
  core: ai-agents-core         # traits, types, specs
```

---

## Agent Lifecycle

Every agent goes through four stages, whether you run it from the CLI or embed it in Rust code.

1. **Define** - You describe the agent in a YAML file (or build it programmatically with `AgentBuilder`). The YAML covers identity, LLM config, tools, state machine, memory, and everything else.
2. **Build** - `AgentBuilder` parses the spec, connects to LLM providers, registers tools, initializes memory, and validates the full configuration. The result is a ready-to-run `Agent` instance.
3. **Chat** - Calling `agent.chat()` (or `agent.prompt()`) sends user input through the process pipeline, into the LLM, through any tool calls or skill executions, and back out as a response. This loop repeats until the agent produces a final answer or hits the iteration limit.
4. **Persist** - Optionally, the session (conversation history, state, context) can be saved to storage and restored later. This lets you build long-running assistants that pick up where they left off.

```yaml
# Minimal agent - all four stages in action
name: LifecycleDemo
system_prompt: "You are a helpful assistant."
llm:
  provider: openai
  model: gpt-4.1-nano
memory:
  type: in-memory
  max_messages: 100
storage:
  type: sqlite
  path: ./sessions.db
```

---

## The YAML Spec

Everything about an agent lives in one YAML file. The framework parses it into an `AgentSpec`, the central data structure that every subsystem reads from.

The major sections of a spec are: identity (`name`, `system_prompt`), LLM configuration (`llm` / `llms`), tools (including MCP servers), skills, state machine, context sources, memory, storage, process pipeline, reasoning, reflection, disambiguation, error recovery, HITL, spawner, and metadata.

You don't need all of them. A minimal spec is just a name, a system prompt, and an LLM block - three lines. Everything else is opt-in. The framework applies sensible defaults for anything you leave out.

For the full list of every field and its options, see the [YAML Reference](@/docs/yaml-reference.md).

```yaml
# The smallest valid agent spec
name: MinimalAgent
system_prompt: "You are a helpful assistant."
llm:
  provider: openai
  model: gpt-4.1-nano
```

---

## LLM System

An agent can use multiple LLMs, each with a name and a role. You define them in the `llms` map, then assign roles with the `llm` selector. Roles include `default` (main chat), `router` (lightweight routing and detection tasks), `evaluator`, and `summarizer`.

If the primary LLM fails, the framework can automatically fall back to another named LLM - no manual retry logic needed. You configure this in the `error_recovery.llm` section. Any supported provider works (OpenAI, Anthropic, Google, Ollama, and 8 more), and you can mix providers freely within a single agent.

```yaml
llms:
  default:
    provider: openai
    model: gpt-4.1-mini
    temperature: 0.7
  router:
    provider: openai
    model: gpt-4.1-nano
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

States support lifecycle actions (`on_enter`, `on_reenter`, `on_exit`) for setting context, and `extract` for pulling structured data from user input.

```yaml
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
      tools: [calculator, http]
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
          operation: random_int
          min: 1
          max: 100
      - prompt: |
          Using the current time and the random number,
          create a short, fun daily briefing for the user.
```

---

## Tools

Tools give the agent the ability to act - call APIs, read files, do math, manipulate data. The framework ships with built-in tools: `datetime`, `json`, `http`, `file`, `text`, `template`, `math`, `calculator`, `random`, and `echo`.

For external tools, you can connect any MCP (Model Context Protocol) server. MCP tools support `stdio` transport, startup timeouts, security restrictions, and function-level views that let you expose subsets of an MCP server's capabilities to different states.

Tool availability follows a 3-level scoping rule: state-level `tools` override spec-level `tools`, which override the full tool registry. This means you can restrict what the agent can use based on where it is in the conversation. Tools can also be secured with rate limits, domain restrictions, timeouts, and HITL approval.

```yaml
tools:
  - calculator
  - datetime
  - name: http
  - name: filesystem
    type: mcp
    transport: stdio
    command: npx
    args: ["-y", "@anthropic/mcp-filesystem"]
    views:
      fs_read:
        functions: [read_file, list_directory]
        description: "Read-only file access"
```

---

## Process Pipeline

The process pipeline lets you declare input and output processing stages that run before and after the LLM. All semantic operations (language detection, entity extraction, PII masking, quality checks) use LLM calls - not regex or keyword matching - so they work across languages and edge cases.

**Input stages** run in order: `normalize` (trim, collapse whitespace), `detect` (language, sentiment, intent), `extract` (entities into context), `sanitize` (PII removal), `validate` (length, content rules). **Output stages** run after the LLM responds: `validate` (quality scoring), `transform` (tone adjustment), `sanitize` (PII masking), `format` (templates, footers).

Each stage can target a specific LLM (typically the lightweight `router`) and store results in context for downstream use.

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

Memory controls what the agent remembers across turns. **In-memory** mode keeps a rolling window of recent messages - simple and fast. **Compacting** mode adds auto-summarization: when recent messages exceed a threshold, older messages get compressed into summaries by a designated LLM.

Token budgeting makes sure memory fits within LLM context limits. You set a total token budget and allocate percentages to summaries, recent messages, and facts. If the budget overflows, the framework either truncates the oldest content or re-summarizes, depending on your chosen strategy.

Memory is stored in-process during a session. For persistence across sessions, pair memory with a storage backend (SQLite, Redis, or file).

```yaml
memory:
  type: compacting
  max_recent_messages: 20
  compress_threshold: 15
  summarize_batch_size: 5
  summarizer_llm: router
  token_budget:
    total: 4000
    allocation:
      summary: 0.3
      recent_messages: 0.6
      facts: 0.1
    overflow_strategy: truncate_oldest
    warn_at_percent: 85
```

---

## Context System

Context provides dynamic data that gets injected into the agent's system prompt at render time. The system prompt is a Jinja2 template, and context values are its variables.

Sources include: **runtime** (passed in by the caller at chat time), **builtin** (datetime, session info, agent metadata), **env** (environment variables), **file** (JSON/YAML files on disk), and **callback** (custom provider functions). Each source has a refresh policy - `once` (load at startup), `per_session` (reload each session), or `per_turn` (refresh every turn).

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
  The user's name is {{ user.name }}.
  Current time: {{ time.datetime }}.
  Speak in {{ user.language }}.
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

**How it works:** After each conversation turn, the framework runs a fast LLM call to extract structured facts (preferences, context, decisions, agreements) from the messages. Facts are stored in the same storage backend as session snapshots, keyed by `(agent_id, actor_id)`. When the same actor returns in a new session, their facts are loaded and injected into the system prompt via the `{{ actor_facts }}` Jinja2 variable.

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

## Dynamic Agent Spawning

A parent agent can create and manage child agents at runtime using the spawner system. This enables patterns like a game master that spawns NPC agents on demand, or a team manager that creates specialist agents for different tasks.

The spawner supports three creation methods: raw YAML strings, `AgentSpec` objects, and named Jinja2 templates. Templates can be defined inline in the YAML or loaded from separate files. A central `AgentRegistry` tracks all spawned agents and provides inter-agent messaging - one agent can send a message to another and receive its response, or broadcast to all agents at once.

When `spawner:` is present in the YAML, the framework automatically registers four built-in tools: `generate_agent` (create a new agent from a description or template), `send_message` (talk to another agent), `list_agents` (see all registered agents), and `remove_agent` (remove an agent from the registry). The parent LLM decides when to use these tools based on the conversation.

Spawned agents share the parent's LLM connections and storage backend by default. A `NamespacedStorage` adapter isolates each agent's sessions by prefixing keys with the agent ID, so multiple agents can safely share a single SQLite database. An `allowed_tools` list restricts what tools spawned agents can declare, preventing LLM-generated YAML from accessing sensitive tools like `http` or `file`.

```yaml
spawner:
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
        model: gpt-4.1-nano
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

Because orchestration runs through the normal `RuntimeAgent` loop, all existing features work automatically: HITL approvals on state transitions, error recovery per delegate, hooks for observability, memory for the parent conversation, and streaming.

When a state transition fires mid-turn and the target state is an orchestration state (`concurrent`, `group_chat`, `pipeline`, `handoff`, or `delegate`), the runtime detects this and re-enters the full dispatch path for the new state in the same turn. The correct orchestration handler activates immediately - the user does not need to send another message to trigger it. Up to three chained transitions are resolved this way before the runtime stops and returns the last available response.

Concurrent states accept a `vote` config to control how individual agent responses are aggregated into a final answer. `vote.method` selects the strategy: `majority` (default), `weighted`, or `unanimous`. `vote.tiebreaker` decides ties: `first` (declaration order), `random`, or `router_decides` (asks the router LLM). Per-agent weights are set via `vote.weights` (a map of agent id to numeric weight, used when method is `weighted`). The `on_partial_failure` field controls behavior when some agents fail: `abort` (default) fails the entire concurrent call, while `proceed_with_available` aggregates only the successful responses.

After each orchestration call the runtime stores the full structured result in `context.orchestration`. Subsequent states can reference the data in prompt templates and guard conditions. The object includes a `type` field (`delegate`, `concurrent`, `group_chat`, `pipeline`, or `handoff`) plus type-specific data such as per-agent responses, the full group chat transcript, pipeline stage outputs, handoff chain events, round counts, and timing. Backward-compatible flat keys (`delegation.<id>.last_response`, `concurrent.result`, `group_chat.conclusion`, `pipeline.result`, `handoff.result`) are also set. The same structure is attached to `response.metadata["orchestration"]` for CLI and hook consumers.

For group chat brainstorm and consensus styles, `response.content` contains the full formatted transcript (`[Round N] speaker: message`) rather than only the last speaker's final line. Debate and maker-checker styles are unaffected because they already produce a synthesized conclusion or final draft. Users on the `InMemoryHistory` backend should be aware that long transcripts consume a ring-buffer slot; `CompactingMemory` handles overflow via summarization. For the maker-checker style, `on_max_iterations` controls what happens when the review loop hits its limit: `accept_last` uses the final draft as-is, `escalate` forwards to a human or parent agent, and `fail` returns an error.

Group chat supports three turn-order methods via `manager.method`. `round_robin` (default) cycles through all participants in declaration order. `random` shuffles each round. `llm_directed` uses the router LLM to pick one speaker at a time after seeing the latest message, capping at `participants.len()` speakers per round for consistent stall detection. `llm_directed` requires a router LLM; the builder returns a config error if none is configured. When `manager.agent` is set to a registry agent id, that agent takes over termination decisions and speaker selection instead of the built-in logic, allowing fully custom orchestration within the group chat loop.

Handoff decisions use structured JSON. The evaluator LLM returns `{"action": "agent_id_or_stay", "confidence": 0.0-1.0, "reason": "..."}`. The runtime parses JSON first (handling markdown code blocks and preamble text), then falls back to fuzzy text matching if JSON extraction fails. This makes handoff robust to variations in LLM output formatting.

When `auto_spawn` is configured, the builder validates that every agent referenced by an orchestration state (`delegate`, `concurrent`, `group_chat`, `pipeline`, `handoff`) was successfully spawned. Missing agents produce a clear build-time error listing each unresolved reference, the state that needs it, and the orchestration pattern involved.

See [YAML Reference - Orchestration States](@/docs/yaml-reference.md#orchestration-states) for the complete field reference.

---

## Safety

The framework provides multiple safety layers to keep agents predictable and secure.

**Error recovery** handles transient failures with configurable retry, exponential backoff, fallback LLMs, and fallback responses. Context overflow (too many tokens) can be handled by summarizing or truncating. Each of these is configurable per subsystem - LLM calls, tool calls, and general errors all have separate policies.

**Tool security** adds rate limits, domain allow/block lists, timeouts, and confirmation requirements on a per-tool basis. You can restrict `http` calls to specific domains or limit how many times a tool can be called per time window.

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
    http:
      rate_limit: 10
      allowed_domains: [api.example.com]

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

## Hooks & Extensibility

The `AgentHooks` trait gives you lifecycle callbacks: before/after chat, before/after tool calls, on state transitions, on errors, and more. Implement the trait and pass your hooks when building the agent to add logging, metrics, custom routing, or any side-effect you need.

Everything in the framework is trait-based and pluggable. You can provide custom implementations of `LLMProvider` (add a new model provider), `Memory` (custom storage or retrieval), `Tool` (any Rust function as a tool), `ApprovalHandler` (custom HITL flow), and `Summarizer` (custom compression logic). The framework doesn't force you into any particular provider, storage, or workflow - it just gives you traits and default implementations.

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
    .hooks(Arc::new(MyHooks))
    .build()?;
```

---

## Next Steps

- **[Getting Started](@/docs/getting-started.md)** - build and run your first agent
- **[YAML Reference](@/docs/yaml-reference.md)** - every field, every option, fully documented
- **[Rust API](@/docs/rust-api.md)** - use the framework as a library
- **[Examples](@/examples/_index.md)** - real-world patterns and complete agent specs

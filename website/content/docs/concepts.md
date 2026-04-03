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

Reasoning controls how the agent thinks before answering. Four modes are available:

- **none** - answer directly, no extra thinking.
- **cot** (chain-of-thought) - think step-by-step before responding.
- **react** - interleave reasoning and tool use (Thought → Action → Observation loop).
- **plan_and_execute** - create a plan first, then execute each step.
- **auto** - let the LLM pick the best mode for each query.

Reflection adds self-evaluation. After producing an answer, the agent scores it against criteria you define (accuracy, completeness, tone). If the score falls below a threshold, the agent retries. Both reasoning and reflection can be overridden at the state or skill level.

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

+++
title = "YAML Reference"
weight = 2
template = "docs.html"
description = "Complete reference for agent YAML specification fields."
+++

<!--# YAML Reference-->

This is the complete reference for every field you can use in an agent YAML file. Each section covers one top-level key, shows its type, default, and a working snippet.

Fixed framework-owned objects reject unknown fields. Misspelled top-level and nested keys therefore fail parsing instead of being ignored. Documented extension maps remain open where values are intentionally provider- or host-specific, including additional provider options inside an `llm` configuration, structured tool and MCP settings, and custom tool settings under `tool_security.tools.<tool_id>.config`. Because those maps intentionally accept provider or host keys, validation cannot distinguish every extension from a nearby framework-field typo. Mapping keys must be strings, and YAML merge keys (`<<`) are rejected rather than being inconsistently retained inside extension maps.

---

## Agent Identity

These fields define who your agent is.

### `name`

The agent's display name. Used in logs, session metadata, and built-in context.

| Detail | Value |
|--------|-------|
| **Type** | `string` |
| **Required** | yes |

```yaml
name: CustomerSupportAgent
```

### `version`

Semantic version string for the agent definition.

| Detail | Value |
|--------|-------|
| **Type** | `string` |
| **Default** | `"1.0.0"` |

```yaml
version: "2.1.0"
```

### `description`

A short description of what the agent does. For documentation only - not sent to the LLM.

| Detail | Value |
|--------|-------|
| **Type** | `string` |
| **Default** | `null` |

```yaml
description: "Multi-branch customer support agent with state routing"
```

### `system_prompt`

The base system prompt sent to the LLM on every turn. Supports full Jinja2 template syntax (`{{ }}`, `{% if %}`, `{% for %}`). Templates are rendered before the LLM sees the prompt.

| Detail | Value |
|--------|-------|
| **Type** | `string` |
| **Required** | yes |

```yaml
system_prompt: |
  You are a support assistant for Acme Corp.
  Customer: {{ context.user.name | default('Guest') }}

  {% if context.user.tier == "vip" %}
  VIP CUSTOMER - provide premium, detailed support.
  {% else %}
  Be helpful and concise.
  {% endif %}
```

### `max_iterations`

Maximum LLM call + tool iterations per user turn. Prevents runaway loops.

| Detail | Value |
|--------|-------|
| **Type** | `u32` |
| **Default** | `10` |

```yaml
max_iterations: 20
```

### `max_context_tokens`

Maximum tokens the conversation history can contribute to the prompt. When exceeded, `error_recovery.llm.on_context_overflow` kicks in.

| Detail | Value |
|--------|-------|
| **Type** | `u32` |
| **Default** | `128000` |

```yaml
max_context_tokens: 128000
```

### `metadata`

Arbitrary JSON metadata. The CLI reads `metadata.cli` for display settings.

| Detail | Value |
|--------|-------|
| **Type** | `object` |
| **Default** | `null` |

```yaml
metadata:
  cli:
    welcome: "=== My Agent ==="
    hints:
      - "Try: Hello!"
      - "Try: Help me with my order"
    show_tools: true
    show_state: true
    show_timing: false
    streaming: true
    prompt_style: with_state   # simple | with_state
    disable_builtin_commands: false
    theme: one-dark            # dark | light | one-dark | catppuccin-mocha | dracula | tokyo-night | vscode-dark | nord | gruvbox-dark | one-half-light | github-light
    hitl:
      style: prompt            # prompt | auto_approve | auto_reject
      show_context: true
```

---

## LLM Configuration

You can configure LLMs in three ways depending on complexity.

### Single LLM - `llm` (shorthand)

The simplest form: one LLM for everything. Use `llm` as a flat object.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `string` | `"openai"` | Provider identifier |
| `model` | `string` | `"gpt-4"` | Model name |
| `temperature` | `f32` | `0.7` | Sampling temperature |
| `max_tokens` | `u32` | `2000` | Max response tokens |
| `top_p` | `f32` | `null` | Nucleus sampling |
| `base_url` | `string` | `null` | API endpoint override (required for `openai-compatible`) |
| `api_key_env` | `string` | `null` | Env var holding the API key (overrides provider default) |
| `timeout_seconds` | `u64` | `null` | Request timeout in seconds |
| `reasoning` | `bool` | `null` | Enable extended thinking / reasoning mode |
| `reasoning_effort` | `string` | `null` | Reasoning effort: `low`, `medium`, or `high` |
| `reasoning_budget_tokens` | `u32` | `null` | Max token budget for reasoning |
| `function_calling` | `bool` | `null` | Override `supports(FunctionCalling)`; `false` disables native tool selection and `true` opts an OpenAI-compatible server into it |
| `tool_choice` | `auto`, `required`, `none`, or `{ specific: ID }` | `null` | Opt in to native or runtime-enforced tool selection; omission preserves the legacy prompt protocol |
| `vision` | `bool` | `null` | Override `supports(Vision)` for this provider |
| `json_mode` | `bool` | `null` | Override `supports(JsonMode)` for this provider |
| `num_ctx` | `u32` | `null` | Ollama context window; merged under request `options` |
| `keep_alive` | `string` or `number` | `null` | Ollama keep-alive duration; top-level request body field |
| `num_gpu` | `i32` | `null` | Ollama GPU layer count; merged under request `options` |
| *(any other key)* | | | Passed through as provider-specific extra parameter |

```yaml
llm:
  provider: openai
  model: gpt-5.4-mini
  temperature: 0.5
  max_tokens: 4000
```

Capability override fields (`function_calling`, `vision`, `json_mode`) are not sent as model parameters. They control framework capability checks; `function_calling: false` also disables native tool selection, while `function_calling: true` opts an OpenAI-compatible server into the native path. Ollama named fields (`num_ctx`, `keep_alive`, `num_gpu`) are captured as provider extras and merged into the Ollama request body.

Any field not listed above is captured in an `extra` map via `#[serde(flatten)]` and forwarded to the LLM client when a matching builder method exists. This includes transport-level resilience (`resilient`, `resilient_attempts`, etc.), Azure settings (`api_version`, `deployment_id`), and provider-native search (`openai_enable_web_search`, `xai_search_mode`, etc.). Provider-native search changes the LLM request and is separate from the granted built-in `web_search` tool, its `WebSearchProvider`, shared policy/HITL path, and tool evidence. See [LLM Providers > Extra Parameters](@/docs/providers.md#extra-parameters) for the full list.

```yaml
# Reasoning model with timeout
llm:
  provider: openai
  model: o3
  reasoning: true
  reasoning_effort: high
  reasoning_budget_tokens: 16384
  timeout_seconds: 120
```

```yaml
# Local Ollama with explicit context window
llm:
  provider: ollama
  model: llama3.1
  num_ctx: 8192
  keep_alive: 5m
  num_gpu: -1
```

```yaml
# Declare capabilities for a compatible local server
llm:
  provider: openai-compatible
  model: qwen3:8b
  base_url: http://localhost:11434/v1
  function_calling: true
  json_mode: true
```

### Tool choice

`tool_choice` is opt-in and never grants a tool. Provider-visible schemas are derived only from the current top-level grant and any state-level narrowing, and every returned call still passes through registry resolution, policy, HITL, final admission, resource locking, rate admission, execution, and evidence.

The default shown as `null` means that no tool-choice policy is configured. It does not mean `tool_choice: none`.

| YAML | Meaning |
|---|---|
| field omitted | Preserve the existing prompt-JSON automatic-selection protocol |
| `tool_choice: null` | Same as omission; no explicit tool-choice policy |
| `tool_choice: none` | Explicitly disable tool definitions, tool prompt instructions, and tool-call parsing for this provider decision |

`auto` is also not the default. It explicitly opts into provider-native automatic selection where supported, with prompt fallback elsewhere.

- omitted or `null`: preserve the existing prompt-JSON automatic-selection protocol;
- `auto`: expose the effective tools and allow the provider to decide whether to call one;
- `required`: require at least one effective tool call;
- `{ specific: ID }`: require the named canonical ID, which must already be inside the effective grant;
- `none`: expose no ordinary tool schema or prompt protocol for that decision and do not parse model text as a tool call.

```yaml
llm:
  provider: openai
  model: gpt-5.4-mini
  tool_choice: required

tools:
  - calculator
```

```yaml
llm:
  provider: openai
  model: gpt-5.4-mini
  tool_choice:
    specific: random

tools:
  - random
```

OpenAI, Anthropic, and OpenRouter use native `auto`, `required`, and `specific` selection. Google uses native `auto`; its `required` and `specific` choices use one bounded prompt corrective retry because the pinned provider dependency does not expose those native modes. OpenAI-compatible servers use native selection only with `function_calling: true`. Other and custom providers use prompt fallback unless they implement the additive native request methods. `none` is enforced by the runtime without exposing definitions or prompt instructions. A second non-compliant prompt response fails the turn.

With explicit `tool_choice`, `chat_stream()` buffers the provider decision before emitting committed text or existing runtime tool events. The framework does not expose a separate provider-native streaming tool-call API in v1.

### Named LLMs - `llms`

Define multiple named LLM configurations when you need different models for different roles (routing, summarization, evaluation).

```yaml
llms:
  default:
    provider: openai
    model: gpt-5.4-mini
  router:
    provider: openai
    model: gpt-5.4-nano
  fallback:
    provider: ollama
    model: llama3
```

### LLM Selector - `llm` (role map)

When `llms` is used, `llm` becomes a role map that assigns named LLMs to framework roles.

| Role | Description |
|------|-------------|
| `default` | Main LLM for conversation |
| `router` | Fast LLM for routing, detection, extraction, transition evaluation |

```yaml
llm:
  default: default
  router: router
```

### Supported Providers

| Provider | `provider` value | API Key Env Var | Notes |
|----------|-----------------|-----------------|-------|
| OpenAI | `openai` | `OPENAI_API_KEY` | GPT models |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | Claude models |
| Google | `google` | `GOOGLE_API_KEY` | Gemini models |
| Groq | `groq` | `GROQ_API_KEY` | Fast inference |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` | |
| xAI | `xai` | `XAI_API_KEY` | Grok models |
| Mistral | `mistral` | `MISTRAL_API_KEY` | |
| Cohere | `cohere` | `COHERE_API_KEY` | |
| Phind | `phind` | `PHIND_API_KEY` | |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | Multi-provider gateway |
| Ollama | `ollama` | - | Local models, no key needed |
| OpenAI-Compatible | `openai-compatible` | via `api_key_env` | Any server speaking the OpenAI protocol |

### `openai-compatible` Example

Connect to any OpenAI-compatible server (LM Studio, vLLM, TGI, LocalAI, Ollama `/v1`):

```yaml
# Each YAML document below is a separate alternative.
llm:
  provider: openai-compatible
  model: qwen3:8b
  base_url: http://localhost:11434/v1
  function_calling: true
  json_mode: true

---
# With authentication:
llm:
  provider: openai-compatible
  model: my-model
  base_url: http://my-server:8080/v1
  api_key_env: MY_SERVER_KEY
```

---

## State Machine

The `states` block defines a finite-state machine that controls conversation flow. The router LLM evaluates `when` conditions after each turn to decide transitions.

### `states` (top-level)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `initial` | `string` | - | Name of the starting state (required) |
| `fallback` | `string` | `null` | State to enter after `max_no_transition` turns with no match |
| `max_no_transition` | `u32` | `null` | Turns without a transition before falling back |
| `regenerate_on_transition` | `bool` | `true` | Re-generate response in the new state after transitioning. When the target state is an orchestration state (`concurrent`, `group_chat`, `pipeline`, `handoff`, `delegate`) or has a non-`none` reasoning mode, the runtime re-enters the full dispatch path instead of using a plain LLM call - the correct handler activates in the same turn. |
| `global_transitions` | `list` | `[]` | Transitions checked from every state |
| `states` | `map` | - | Named state definitions |

```yaml
states:
  initial: greeting
  fallback: confused
  max_no_transition: 3
  global_transitions:
    - to: escalation
      when: "User is angry or asks for a manager"
      priority: 100
  states:
    greeting: { ... }
    confused: { ... }
    escalation: { ... }
```

### State Definition - `states.states.<name>`

Each state shapes the LLM's behavior for a phase of the conversation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | `string` | - | State-specific prompt (appended to system_prompt by default) |
| `prompt_mode` | `string` | `"append"` | How to combine with system_prompt: `append`, `replace`, `prepend` |
| `llm` | `string` | `null` | Override the LLM alias for this state |
| `tools` | `list` | *inherit* | Tool IDs available in this state. `[]` = no tools. Omit = inherit current grant. State tools can only narrow top-level `tools:` |
| `skills` | `list` | *inherit* | Skill IDs available in this state |
| `max_turns` | `u32` | `null` | Auto-transition via `timeout_to` after this many turns |
| `timeout_to` | `string` | `null` | State to enter when `max_turns` is exceeded |
| `transitions` | `list` | `[]` | Transition rules (see below) |
| `on_enter` | `list` | `[]` | Actions on first entry |
| `on_exit` | `list` | `[]` | Actions when leaving |
| `on_reenter` | `list` | `[]` | Actions when returning (replaces `on_enter` on subsequent visits) |
| `extract` | `list` | `[]` | Extract structured values from user input into context |

```yaml
states:
  initial: greeting
  states:
    greeting:
      prompt: |
        Welcome the customer warmly.
        Ask how you can help them today.
      tools: []
      on_enter:
        - set_context:
            phase: "greeting"
      transitions:
        - to: helping
          when: "User describes an issue or asks a question"
```

### Transitions

Each transition defines a rule for moving between states.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `to` | `string` | - | Target state name. Prefix with `^` to escape to a root-level state from a sub-state |
| `when` | `string` | `null` | Natural-language condition evaluated by the router LLM |
| `guard` | `object` | `null` | Deterministic context check - fires instantly, no LLM call needed |
| `auto` | `bool` | `true` | Transition participates in automatic transition evaluation |
| `priority` | `u32` | `0` | Higher priority transitions are evaluated first |
| `cooldown_turns` | `u32` | `null` | Minimum turns before this transition can fire again |
| `timing` | enum | `post_response` | `post_response`, `pre_response`, or `parallel`; `parallel` requires speculative state transitions and response-independent conditions |
| `requires_response` | `bool` | `false` | Set true when the condition needs the assistant response text; rejected with non-post-response timing |
| `run_extractors` | `bool` | `false` | Run current-state extractors for this pre-response transition and stage their values until the transition commits |

```yaml
transitions:
  # LLM-evaluated transition
  - to: technical_support
    when: "User mentions login problems, errors, or bugs"

  # Deterministic guard transition (no LLM call)
  - to: complete
    guard:
      context:
        user.verified:
          eq: true
    priority: 10

  # Escape to root-level state from inside a sub-state
  - to: "^escalation"
    when: "Problem is too complex or user is frustrated"

  # Optional runtime-optimized transition. This is safe only because the guard
  # reads context and does not need the assistant response.
  - to: billing
    guard:
      context:
        request.topic:
          eq: billing
    timing: pre_response
    requires_response: false
```

Guard expressions support:
- `exists: true` / `exists: false` - check if a context key is set
- `eq: <value>` - exact match
- `in: [val1, val2]` - value is one of the listed items

### Sub-States (Hierarchical)

Any state can contain its own `initial` and `states` block, creating a nested state machine. Entering the parent automatically enters its `initial` sub-state. Sub-states inherit the parent's prompt.

```yaml
states:
  initial: troubleshooting
  states:
    troubleshooting:
      prompt: "You are troubleshooting a technical issue."
      initial: diagnosing
      states:
        diagnosing:
          prompt: "Ask questions to understand the problem."
          max_turns: 5
          timeout_to: "^escalation"
          transitions:
            - to: fixing
              when: "Enough information to suggest a fix"
        fixing:
          prompt: "Suggest a step-by-step solution."
          transitions:
            - to: "^resolved"
              when: "User confirms the fix worked"
            - to: diagnosing
              when: "Fix didn't work, need more info"
    resolved:
      prompt: "Summarize what was done."
    escalation:
      prompt: "Hand off to a specialist."
```

### Extract (Context from User Input)

States can extract structured values from user messages into context using the router LLM.

```yaml
states:
  states:
    collect_name:
      prompt: "Ask the user for their name."
      extract:
        - key: user.name
          description: "The user's name"
      transitions:
        - to: collect_email
          guard:
            context:
              user.name:
                exists: true
```

### Lifecycle Actions

`on_enter`, `on_exit`, and `on_reenter` accept a list of actions.

```yaml
on_enter:
  - set_context:
      phase: "drafting"
      draft_version: 1
on_reenter:
  - set_context:
      draft_version: 2
on_exit:
  - set_context:
      drafting_completed: true
```

### Orchestration States

The framework supports five orchestration patterns. Each has a dedicated field on `StateDefinition` wired into the runtime.

| Pattern | State type field | Description |
|---------|-----------------|-------------|
| **Router** | `delegate` | `delegate` states + LLM-evaluated `when` transitions. The existing transition evaluator does the routing. |
| **Pipeline** | `pipeline` | Sequential agent chain with optional per-stage input templates. Runs in one `chat()` call. |
| **Concurrent** | `concurrent` | Parallel agent execution with aggregation strategies. Runs in one `chat()` call. |
| **Group Chat** | `group_chat` | Multi-agent conversation with turn management and termination detection. Runs in one `chat()` call. |
| **Handoff** | `handoff` | LLM-directed agent-to-agent control transfer. Runs in one `chat()` call. |

All five patterns are also available as [orchestration tools](@/docs/yaml-reference.md#orchestration-tools) for LLM-decided coordination at runtime. `delegate`, `concurrent`, `group_chat`, `pipeline`, and `handoff` are mutually exclusive on a state.

All orchestration states require a `spawner:` section with `auto_spawn` so the referenced agents exist in the registry at startup. See [Spawner](@/docs/yaml-reference.md#spawner-dynamic-agent-spawning) for setup.

#### `delegate`

Forward all user messages to a registry agent. The parent's transition evaluator watches the delegate's responses and fires transitions when conditions match.

```yaml
states:
  initial: triage
  states:
    triage:
      prompt: "Determine what the user needs."
      transitions:
        - to: billing_help
          when: "User has a billing question"
    billing_help:
      delegate: billing              # agent ID from auto_spawn
      delegate_context: input_only   # input_only (default) | summary | full
      transitions:
        - to: triage
          when: "Issue resolved"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `delegate` | `string` | - | Registry agent ID to forward messages to |
| `delegate_context` | `string` | `input_only` | Context passed to the delegate: `input_only` (just the message), `summary` (LLM-summarized history), `full` (recent messages) |

#### `concurrent`

Run multiple agents in parallel and aggregate their results. The state completes in one `chat()` call.

```yaml
    analyze:
      concurrent:
        agents: [fundamental, technical, sentiment]
        input: "Analyze the stock the user mentioned."
        timeout_ms: 30000
        aggregation:
          strategy: llm_synthesis
          synthesizer_llm: router
      transitions:
        - to: present
          auto: true
```

Agents can also carry weights for weighted voting:

```yaml
      concurrent:
        agents:
          - id: senior
            weight: 2.0
          - id: junior
            weight: 1.0
        aggregation:
          strategy: voting
          vote:
            method: weighted
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `concurrent.agents` | `list` | - | Agent IDs (strings) or weighted entries (`{id, weight}`) |
| `concurrent.input` | `string` | - | Jinja2 template for the input sent to each agent. `{{ user_input }}` is the user's message; `{{ context.<key> }}` accesses context manager values. When omitted, agents receive the raw user input directly. |
| `concurrent.timeout_ms` | `u64` | - | Per-agent timeout in milliseconds |
| `concurrent.min_required` | `u32` | - | Minimum agents that must succeed |
| `concurrent.on_partial_failure` | `string` | `proceed_with_available` | `proceed_with_available` continues with successful agents only, ignoring failures. `abort` fails the entire concurrent block if any agent fails. |
| `concurrent.aggregation.strategy` | `string` | - | `voting`, `llm_synthesis`, `first_wins`, or `all` |
| `concurrent.aggregation.synthesizer_llm` | `string` | - | LLM alias for synthesis or vote extraction |
| `concurrent.aggregation.vote.method` | `string` | `majority` | `majority` (most common answer wins), `weighted` (uses agent `weight` values from the `agents` list; `- id: agent_a` defaults to weight 1.0, `- { id: agent_a, weight: 2.0 }` sets explicit weight), or `unanimous` (all agents must agree or tiebreaker applies). |
| `concurrent.aggregation.vote.tiebreaker` | `string` | `first` | `first` (deterministic, picks first response in agent order), `random` (random selection among tied answers), or `router_decides` (router LLM breaks the tie). |
| `concurrent.context_mode` | `string` | `input_only` | Parent conversation context forwarded to each agent: `input_only` (just the message), `summary` (LLM-summarized history), `full` (recent messages). When set, `{{ user_input }}` in `input` templates contains the enriched text. |

#### `group_chat`

Run a multi-agent conversation. Agents talk to each other in a shared thread until termination.

```yaml
    review:
      group_chat:
        participants:
          - id: architect
          - id: security
            role: "security reviewer"
        style: consensus
        max_rounds: 5
        termination:
          method: manager_decides
          max_stall_rounds: 2
      transitions:
        - to: approved
          when: "Consensus reached"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `group_chat.participants` | `list` | - | Agent entries with `id` and optional `role` |
| `group_chat.style` | `string` | `brainstorm` | `brainstorm` (free-form discussion), `consensus` (same loop but router LLM checks agreement after each round), `debate` (structured pro/con with synthesizer), or `maker_checker` (create-review loop) |
| `group_chat.max_rounds` | `u32` | `5` | Maximum conversation rounds |
| `group_chat.timeout_ms` | `u64` | - | Total timeout for the entire conversation |
| `group_chat.termination.method` | `string` | `manager_decides` | `manager_decides` (stops on max_rounds or stall), `max_rounds` (runs exactly max_rounds rounds, stall detection disabled), or `consensus_reached` (router LLM checks agreement after each round). Note: `style: consensus` automatically enables agreement checks regardless of this setting. |
| `group_chat.termination.max_stall_rounds` | `u32` | `2` | Stop if no new content for this many rounds |
| `group_chat.manager.method` | `string` | - | Turn order policy: `round_robin`, `random`, or `llm_directed`. `llm_directed` requires a router LLM configured in `llms`/`llm`; the LLM picks one speaker at a time after seeing the latest message. |
| `group_chat.manager.agent` | `string` | - | Registry agent ID that acts as the manager. When set, the manager agent controls termination decisions (replaces stall detection when `termination.method: manager_decides`) and speaker selection (when `manager.method: llm_directed`). |
| `group_chat.debate.rounds` | `u32` | `3` | Fixed rounds for debate style |
| `group_chat.debate.synthesizer` | `string` | - | Agent ID that produces the final answer |
| `group_chat.maker_checker.max_iterations` | `u32` | `3` | Create-review loop limit |
| `group_chat.maker_checker.acceptance_criteria` | `string` | - | LLM-evaluated acceptance criteria |
| `group_chat.maker_checker.on_max_iterations` | `string` | `accept_last` | `accept_last` (returns the last draft as the result), `escalate` (returns with `termination_reason: "escalated"`), or `fail` (returns an error). |
| `group_chat.input` | `string` | - | Jinja2 template for the topic sent to participants. `{{ user_input }}` is the user's message; `{{ context.<key> }}` accesses context values. When omitted, the raw user message is used as the topic. |
| `group_chat.context_mode` | `string` | `input_only` | Parent conversation context included in the topic: `input_only` (just the message), `summary` (LLM-summarized history), `full` (recent messages). When set, `{{ user_input }}` in `input` templates contains the enriched text. |

#### `pipeline`

Run agents sequentially in a single `chat()` call. Each stage can have a Jinja2 input template. Available template variables:

- `{{ previous_output }}` - output from the immediately previous stage
- `{{ original_input }}` - the user's original input
- `{{ user_input }}` - alias for `original_input` (consistent with concurrent templates)
- `{{ stages.<agent_id> }}` - output from any earlier stage by agent ID
- `{{ context.<key> }}` - values from the context manager (same as concurrent templates)

`{{ stages.<id> }}` lets later stages reference any earlier stage explicitly. Without it, the editor in a writer-reviewer-editor pipeline would only see the reviewer's feedback and lose the writer's original draft.

```yaml
    process:
      pipeline:
        stages:
          - writer
          - id: reviewer
            input: "Review this draft:\n{{ stages.writer }}\n\nOriginal: {{ original_input }}"
          - id: editor
            input: "Polish this content.\n\nDraft:\n{{ stages.writer }}\n\nFeedback:\n{{ stages.reviewer }}\n\nOriginal: {{ original_input }}"
        timeout_ms: 60000
      transitions:
        - to: done
          when: "Pipeline complete"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pipeline.stages` | `list` | - | Agent IDs (strings) or entries with `{id, input}` for per-stage Jinja2 templates |
| `pipeline.timeout_ms` | `u64` | - | Total pipeline timeout in milliseconds |
| `pipeline.context_mode` | `string` | `input_only` | Parent conversation context forwarded to the first stage: `input_only` (just the message), `summary` (LLM-summarized history), `full` (recent messages). Later stages access the enriched input via `{{ original_input }}`. |

Stage input templates and concurrent/group_chat `input` templates all support the same variables: `{{ user_input }}` for the user's message and `{{ context.<key> }}` for context manager values. Pipeline stages additionally have `{{ previous_output }}`, `{{ original_input }}`, and `{{ stages.<agent_id> }}`.

#### `handoff`

LLM-directed agent-to-agent control transfer. A router LLM evaluates each agent's response and decides whether to hand off to another specialist.

```yaml
    support:
      handoff:
        initial_agent: triage
        available_agents: [technical, billing]
        max_handoffs: 3
      transitions:
        - to: done
          when: "Issue resolved"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `handoff.initial_agent` | `string` | - | Starting agent ID |
| `handoff.available_agents` | `list` | - | Agent IDs that can receive handoffs |
| `handoff.max_handoffs` | `u32` | `5` | Maximum control transfers before stopping |
| `handoff.input` | `string` | - | Jinja2 template for the input sent to the initial agent. `{{ user_input }}` is the user's message; `{{ context.<key> }}` accesses context values. When omitted, the raw user message is forwarded directly. |
| `handoff.context_mode` | `string` | `input_only` | Parent conversation context forwarded to the initial agent: `input_only` (just the message), `summary` (LLM-summarized history), `full` (recent messages). Intra-chain handoffs still pass the previous agent's response as context. |

#### Orchestration Result Storage

After each orchestration call the runtime stores the full structured result in `context.orchestration`. Subsequent states can reference this data in prompt templates and guard conditions. Backward-compatible flat keys are also set so existing templates keep working.

| Pattern | `context.orchestration` fields | Backward-compatible key |
|---------|-------------------------------|------------------------|
| `delegate` | `type`, `agent`, `state`, `response`, `duration_ms` | `delegation.<id>.last_response` |
| `concurrent` | `type`, `result`, `strategy`, `agents[]` (per-agent `id`, `response`, `success`, `error`, `duration_ms`), `duration_ms` | `concurrent.result` |
| `group_chat` | `type`, `conclusion`, `transcript[]` (per-turn `speaker`, `round`, `content`), `rounds`, `termination`, `duration_ms` | `group_chat.conclusion` |
| `pipeline` | `type`, `result`, `stages[]` (per-stage `agent_id`, `output`, `duration_ms`, `skipped`), `duration_ms` | `pipeline.result` |
| `handoff` | `type`, `result`, `final_agent`, `handoff_chain[]` (per-handoff `from`, `to`, `reason`), `duration_ms` | `handoff.result` |

The same structure is attached to `response.metadata["orchestration"]` for CLI and hook consumers.

Example - accessing concurrent per-agent results in a follow-up state:

```yaml
    present_results:
      prompt: |
        Analysis results:
        {% for agent in context.orchestration.agents %}
        {{ agent.id }}: {{ agent.response }}
        {% endfor %}

        Strategy: {{ context.orchestration.strategy }}
```

Example - using group chat metadata in a follow-up state:

```yaml
    summary:
      prompt: |
        The discussion concluded after {{ context.orchestration.rounds }} rounds.
        Reason: {{ context.orchestration.termination }}
```

For group chat brainstorm and consensus styles, `response.content` contains the full formatted transcript (`[Round N] speaker: message`) rather than only the last speaker's final line. Debate and maker-checker styles are unaffected.

Actor-aware turns also expose structural identity context during inter-agent calls:

- `context.interaction.origin_actor_id` - original user, player, or customer actor when available
- `context.interaction.sender_agent_id` - immediate agent that forwarded or produced the current message
- `context.interaction.actor_id` - effective actor ID used for actor-scoped facts and relationship memory

This context is forwarded by registry sends, delegate states, concurrent states, group chat, pipeline stages, handoff chains, and orchestration tools.

---

## Skills

Skills are reusable multi-step pipelines triggered by natural language. The router LLM picks the right skill based on `trigger` matching. Skills are **stateless and single-shot**: the executor runs each step prompt as an isolated LLM call with no conversation history or memory.

### Inline Skill

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Unique skill identifier |
| `description` | `string` | What this skill does (shown to router) |
| `trigger` | `string` | Natural-language trigger condition |
| `steps` | `list` | Ordered list of `prompt` or `tool` steps |
| `disambiguation` | `object` | Optional skill-level disambiguation override (see [Skill-Level Disambiguation Override](#skill-level-disambiguation-override)) |
| `reasoning` | `object` | Optional skill-level reasoning override (see [Skill-Level Reasoning Override](#skill-level-reasoning-override)) |
| `reflection` | `object` | Optional skill-level reflection override |

Step prompts are rendered as Jinja2 templates with these variables:

| Variable | Type | Description |
|----------|------|-------------|
| `user_input` | `string` | The user's message (or enriched input after disambiguation) |
| `steps` | `list` | Previous step results. Access via `steps[N].result` and `steps[N].args` |
| `context` | `object` | Extra context passed to the executor (empty `{}` by default) |

> **Important:** If a step prompt does not reference `{{ user_input }}`, the LLM cannot see what the user said.

> **Note:** Skills are single-shot. After execution, the next user message goes through full routing - it does not return to the skill. Do not ask for "confirmation" or "reply with X" in step prompts. To collect parameters before execution, use `required_clarity` in the skill's `disambiguation` override.

```yaml
skills:
  - id: greeting
    description: "Greet users warmly"
    trigger: "When user says hello, hi, or greets"
    steps:
      - prompt: |
          The user greeted you: "{{ user_input }}"
          Respond with a warm, friendly greeting.
```

### Multi-Step Skill with Tools

Steps execute in order. Each step can reference previous results via `{{ steps[N].result }}`.

```yaml
skills:
  - id: daily_briefing
    description: "Give the user a personalized daily briefing"
    trigger: "When user asks for a daily briefing or morning update"
    steps:
      - tool: datetime
        args:
          operation: "now"
      - tool: random
        args:
          operation: "integer"
          min: 1
          max: 10
      - prompt: |
          Current date/time: {{ steps[0].result }}
          Energy score: {{ steps[1].result }}
          Create a short, cheerful daily briefing.
```

### External Skill File

Load a skill from a separate `.skill.yaml` file:

```yaml
skills:
  - file: skills/math_helper.skill.yaml
  - file: skills/weather_clothes.skill.yaml
  - id: inline_skill
    description: "..."
    trigger: "..."
    steps: [...]
```

A `.skill.yaml` file looks like:

```yaml
skill: math_helper
description: "Help with mathematical calculations"
trigger: "When user requests help with calculations or math"
steps:
  - prompt: |
      Extract ONLY the mathematical expression from: "{{ user_input }}"
  - tool: calculator
    args:
      expression: "{{ steps[0].result }}"
  - prompt: |
      User question: {{ user_input }}
      Result: {{ steps[1].result }}
      Explain the calculation in a friendly way.
```

> **Note:** All tools used by any skill must also be listed in the agent's top-level `tools` section.

---

## Tools

The top-level `tools` list declares the maximum set of tools the agent can use. The framework auto-injects tool names, descriptions, and argument schemas into the prompt - do **not** list them in `system_prompt`.

Omitting top-level `tools:` means no ordinary LLM-callable tools. `tools: []` also means no ordinary tools. Registering built-ins or MCP tools does not grant model access by itself; list each callable ordinary tool or view explicitly under top-level `tools:`. Feature-generated tools use the explicit grant settings described below.

YAML feature flags can also be registration and grant signals: `spawner.management_tools` grants selected agent management tools, `spawner.orchestration_tools` grants selected orchestration tools, and `persona.evolution.allow_llm_evolve: true` grants `persona_evolve`. These are explicit opt-ins, so they apply even when top-level `tools:` is omitted or empty.

### Simple String Form

Reference a built-in tool by name:

```yaml
tools:
  - calculator
  - datetime
  - echo
```

### Structured Form

```yaml
tools:
  - name: calculator
  - name: glob
  - name: grep
  - name: file_read
  - name: file_list
  - name: file_info
  - name: file_write
  - name: file_edit
  - name: patch
  - name: copy_path
  - name: move_path
  - name: delete_path
  - name: git_status
  - name: git_diff
  - name: diagnostics
  - name: command
  - name: ask_user
  - name: todo
  - name: sleep
  - name: web_fetch
  - name: web_search
```

### Built-in catalog

Read-only workspace discovery and inspection:

- `glob` - find paths by glob pattern with stable sorting and pagination
- `grep` - search text files with regex or literal matching
- `file_read` - read bounded UTF-8 text ranges from local files
- `file_list` - list directory entries with recursion, glob filters, and pagination
- `file_info` - inspect safe file metadata without reading contents

Read-only repository and host inspection:

- `git_status` - inspect repository status without shell access
- `git_diff` - inspect bounded unified diffs without shell access
- `diagnostics` - read compiler, linter, LSP, or editor diagnostics when a host provider is installed

Controlled mutation and validation:

- `file_write` - create or overwrite one file with atomic writes, dry-run review, and write-path policy
- `file_edit` - replace exact text with uniqueness/no-match checks, dry-run diff summaries, near-match hints, and read-before-write support
- `patch` - validate or apply bounded unified diffs with per-file policy checks, delete gating, and parent-directory policy
- `copy_path` - copy a file or directory with source/destination policy checks and explicit dry-run review
- `move_path` - move or rename a file or directory with source/destination policy checks and explicit dry-run review
- `delete_path` - delete a file or directory with recursive-delete gating and explicit dry-run review
- `command` - run exact allowlisted non-interactive argv commands with bounded output, redacted evidence, and explicit working directories

Interaction and session-local helpers:

- `ask_user` - ask the user a structured follow-up question through a host question handler
- `todo` - manage a session-local structured task list
- `sleep` - wait for a policy-bounded duration without shell access

Network and legacy built-ins:

- `web_fetch` - fetch public web content with scheme, redirect, DNS/IP, byte, output, cache, and optional extraction controls
- `web_search` - search through a host-provided provider and return bounded results; reports unavailable when no provider is installed
- `calculator`, `datetime`, `echo`, `json`, `math`, `random`, `text`, `template` - existing compute and text/data helpers
- `file` - compatibility aggregate file tool; new YAML should prefer split file tools above. Raw `.git` paths are blocked by file tools; use `git_status` or `git_diff` for repository inspection.
- `http` - raw HTTP API client for GET, POST, PUT, PATCH, DELETE, and HEAD requests. Use domain and method policy for API calls; use `web_fetch` when the goal is public page retrieval and text extraction.

See [Built-in Tools](@/docs/built-in-tools.md) for the complete 30-tool input, output, safety, policy, host-provider, and eval reference.

### Expanded built-in inputs

These inputs are generated into the model-facing tool schemas when the tool is granted. All outputs are bounded and include truncation metadata where applicable. Every built-in `max_results` value in this table must be a positive integer; generated schemas advertise a minimum of `1`, and runtime deserialization rejects zero.

| Tool | Required inputs | Optional inputs and defaults |
|------|-----------------|------------------------------|
| `glob` | `pattern` | `path: "."`, `max_results: 100`, `offset: 0`, `include_dirs: false`, `sort: path` |
| `grep` | `pattern` | `mode: regex`, `path: "."`, `include_glob`, `case_sensitive: false`, `output_mode: files_with_matches`, `context: 0`, `max_results: 250`, `offset: 0`, `max_file_size_bytes: 1048576`, `max_output_chars: 20000` |
| `file_read` | `path` | `start_line: 1`, `end_line`, `max_lines: 2000`, `max_bytes: 1048576` |
| `file_list` | `path` | `recursive: false`, `include_glob`, `exclude_glob`, `include_hidden: false`, `max_results: 200`, `offset: 0`, `sort: path` |
| `file_info` | `path` | `follow_symlinks: false` |
| `git_status` | none | `path: "."`, `include_untracked: true`, `max_results: 200` |
| `git_diff` | none | `path: "."`, `staged: false`, `paths: []`, `max_output_chars: 20000` |
| `diagnostics` | none | `path`, `severity: all` (`error`, `warning`, `info`, `hint`, `all`), `max_results: 200` |
| `ask_user` | `question` | `options: []`, `multi_select: false`, `allow_other: true`, `default`, `timeout_seconds` |
| `todo` | `operation` (`list`, `set`, `update`, `clear`) | `items` for `set`; `id`, `status`, `content`, and `active_form` for `update` |
| `sleep` | `duration_ms` | `reason`; default maximum is 30000 ms and trusted policy can change it with `config.max_duration_ms` plus `timeout_ms` |
| `file_write` | `path`, `content` | `overwrite: false`, `create_parent_dirs: false`, `dry_run: false` |
| `file_edit` | `path`, `old_text`, `new_text` | `replace_all: false`, `dry_run: false`, `max_replacements: 20` |
| `patch` | `patch` | `base_path: "."`, `dry_run: false`, `allow_new_files`, `allow_delete: false` |
| `copy_path` | `source_path`, `destination_path` | `overwrite: false`, `create_parent_dirs: false`, `dry_run: false` |
| `move_path` | `source_path`, `destination_path` | `overwrite: false`, `create_parent_dirs: false`, `dry_run: false` |
| `delete_path` | `path` | `recursive: false`, `dry_run: false` |
| `command` | `argv` preferred; `command` string is compatibility-only | `cwd: "."`, `env: {}`, `timeout_ms: 30000`, `max_output_chars: 20000`, `reason` |
| `web_fetch` | `url` | `prompt`, `max_chars: 20000`, `cache_ttl_seconds: 900` (`0` disables caching; positive representable values set storage-time lifetime without hit refresh; unrepresentable values fail before network access), `max_response_bytes: 1048576` per response including redirects, `max_redirects: 5` |
| `web_search` | `query` | `max_results: 5`, `include_domains: []`, `language`, `region`, `safe_search` (`off`, `moderate`, `strict`) |

All filesystem mutation tools execute by default. Set `dry_run: true` explicitly to validate or preview without changing the filesystem. Every actual mutation requires applicable write policy and either approval or an explicit trusted-policy exemption. Results use `mutation_performed` as the authoritative effect flag; `created`, `overwritten`, `copied`, `moved`, and `deleted` are true only when that action was performed. Patch application preflights every target and checks the preflight state again before applying; rollback attempts restore prior content only when doing so will not overwrite an observed external change. These attempts are not a general rollback guarantee, and partial external effects can remain after timeout, cancellation, cleanup failure, rollback conflict, or process/network activity. Recursive copy rejects symbolic links and destinations equal to or nested under the source. The v1 path contract assumes a host-owned workspace without concurrent untrusted external replacement of validated paths; runtime locks serialize framework-owned mutations but do not provide an OS filesystem sandbox.

`ask_user` uses a host question handler, not HITL approval. Without a handler its implementation still executes and returns a structured unavailable response, using `default` when present. After initial policy checks, missing providers for `diagnostics`, `command`, and `web_search` are rejected by runtime availability preflight before HITL or implementation invocation and recorded with `executed: false`; availability is revalidated during final pre-lock admission after approval. An unavailable `web_search` does not automatically invoke `web_fetch`; the model can fetch only a separately known URL when `web_fetch` is also granted.

`command` is not a shell. Granting the tool is not enough: the command policy must include an exact `allowed_commands` argv entry or a `command_templates` entry before the built-in can execute. `commands.allow` is legacy command-name matching and does not replace the exact argv allowlist.

### Built-in tool-specific config

Tool arguments are generated into model-facing schemas and are supplied by the model at call time. `tool_security.tools.<tool_id>.config` is host-supplied configuration and is not model-callable. Built-ins should use framework policy fields when possible; only use `config` for explicit tool-specific settings.

| Tool | Config key | Meaning |
|------|------------|---------|
| `sleep` | `max_duration_ms` | Maximum requested wait duration. The effective cap is also bounded by `timeout_ms` and `tool_security.default_timeout_ms`. |

Other current built-ins use their input schema plus common policy fields such as `read_paths`, `write_paths`, `domains`, `allowed_commands`, `max_results`, `max_output_chars`, `max_response_bytes`, and `max_changed_lines`. Custom tools receive arbitrary `config` values through `ToolExecutionContext.custom_config`.

### MCP Tool

Declare an MCP server as a tool entry. The framework connects at startup, discovers available functions, and exposes them.

```yaml
tools:
  - name: filesystem
    type: mcp
    transport: stdio                # stdio | http | sse
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "./"]
    startup_timeout_ms: 15000
    security:
      blocked_functions: []
    views:
      fs_read:
        functions: [read_file, list_allowed_directories, search_files]
        description: "Read-only filesystem operations"
      fs_write:
        functions: [write_file, create_directory, move_file]
        description: "Filesystem write operations"
  - fs_read
  - fs_write
  - datetime
```

Views create named subsets of a server's functions. States reference views by name for scoped tool access. The parent tool name (e.g. `filesystem`) always includes **all** functions.

MCP parent tools and views share the same executor, connection, security policy, HITL policy, and canonical identity resolution. MCP function names are passed in tool arguments, while policy and availability use the canonical parent or view tool ID.

```yaml
states:
  states:
    browsing:
      tools: [fs_read, datetime]     # read-only MCP view + built-in
    editing:
      tools: [fs_write, datetime]    # write MCP view + built-in
    full_access:
      tools: [filesystem, datetime]  # all MCP functions + built-in
```

### Per-State Tool Scoping

Tool availability per state narrows the current effective grant:

- `tools: []` - explicitly no tools in this state or any descendant that omits its own list
- `tools: [datetime]` - only `datetime`, and only if every earlier grant and explicit ancestor scope also permits `datetime`
- *omit `tools`* - inherit the complete effective narrowing from the top level and all explicit ancestor states

A state cannot expose a tool that is absent from the effective grant. The runtime intersects the declared grant, any live runtime narrowing, and every explicit state scope from the root state to the current state. The declared grant is top-level `tools:` plus explicitly enabled feature tools such as orchestration tools or `persona_evolve`. Tool security, HITL, eval assertions, and recovery policies use canonical tool IDs after normal alias resolution.

### `tool_aliases`

Multi-language names and descriptions for tools. Lets the same tool appear with localized names to the LLM.

| Detail | Value |
|--------|-------|
| **Type** | `map` |
| **Default** | `{}` |

```yaml
tool_aliases:
  calculator:
    names:
      ko: "계산기"
      ja: "電卓"
    descriptions:
      ko: "수학 계산을 수행합니다"
      ja: "数学の計算を実行します"
```

---

## Context

The `context` map injects dynamic data into prompts. Values are available as `{{ context.<name>.field }}` in any Jinja2 template.

### `type: runtime`

Data provided by the Rust host (or CLI defaults). Best for per-user data.

```yaml
context:
  user:
    type: runtime
    required: false
    schema:
      name: string
      language: string
      role: string
    default:
      name: "Guest"
      language: "English"
      role: "user"
```

### `type: builtin`

Auto-provided by the framework. No host code or external data needed.

| Source | Fields | Notes |
|--------|--------|-------|
| `datetime` | `date`, `time`, `hour`, `minute`, `day_of_week`, `year`, `month`, `day`, `utc`, `local`, `timestamp` | Set `refresh: per_turn` for live clock |
| `session` | `id`, `started_at` | Set once at session start |
| `agent` | `name`, `version` | From the YAML file itself |

```yaml
context:
  time:
    type: builtin
    source: datetime
    refresh: per_turn
  session:
    type: builtin
    source: session
  agent_info:
    type: builtin
    source: agent
```

### `type: env`

Read an environment variable at startup. Keeps secrets out of YAML files.

```yaml
context:
  app_env:
    type: env
    name: APP_ENV
  greeting_style:
    type: env
    name: GREETING_STYLE
```

### `type: callback`

Resolved by a named `ContextProvider` registered from Rust code.

```yaml
context:
  weather:
    type: callback
    name: weather_provider
```

### `type: file`

Load context from a file on disk.

```yaml
context:
  config:
    type: file
    path: "./config.json"
```

### `type: http`

Load a JSON object from an HTTP endpoint. This source requires the `http-context` crate feature; the CLI's full feature set includes it. Without the feature, the configured `fallback` is returned. Supported methods are `GET` (default), `POST`, `PUT`, and `DELETE`; other values use GET. URL and header templates are rendered from current context, successful responses must decode as JSON, and transport, status, or decode failures return `fallback` when configured.

| Field | Type | Default | Description |
|---|---|---|---|
| `url` | string | required | URL template rendered from current context |
| `method` | string | `GET` | `GET`, `POST`, `PUT`, or `DELETE` |
| `headers` | map | `{}` | Header value templates |
| `refresh` | enum | `per_session` | `once`, `per_session`, or `per_turn` |
| `timeout_ms` | integer | none | Request timeout in milliseconds |
| `fallback` | JSON value | none | Value returned when HTTP support is unavailable or the request/JSON decode fails |

```yaml
context:
  account:
    type: http
    url: "https://api.example.com/accounts/{{ context.user.id }}"
    method: GET
    headers:
      Authorization: "Bearer {{ context.auth.token }}"
    refresh: per_session
    timeout_ms: 5000
    fallback:
      tier: unknown
```

HTTP context is host/application data loading, not the policy-aware `web_fetch` tool. The source does not expose a cache field in v1.

### Using Context in Templates

```yaml
system_prompt: |
  Current date: {{ context.time.date }}
  Day: {{ context.time.day_of_week }}
  User: {{ context.user.name }}
  Session: {{ context.session.id }}

  {% if context.app_env == "production" %}
  Be formal and concise.
  {% else %}
  Be casual and verbose.
  {% endif %}
```

---

## Memory

Controls how conversation history is stored and managed.

### `type: in-memory`

Simple ring buffer. When `max_messages` is exceeded the oldest message is dropped.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `string` | `"in-memory"` | Memory backend |
| `max_messages` | `usize` | `100` | Maximum messages to keep |

```yaml
memory:
  type: in-memory
  max_messages: 50
```

### `type: compacting`

LLM-based summarization compresses old messages into a rolling summary while keeping recent ones intact.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `string` | - | Must be `"compacting"` |
| `max_recent_messages` | `usize` | `50` | Requested recent messages kept verbatim; clamped when it conflicts with the threshold |
| `compress_threshold` | `usize` | `30` | Compression runs when the stored message count reaches this value |
| `summarize_batch_size` | `usize` | `10` | Maximum eligible old messages summarized in one operation; zero is normalized to one |
| `summarizer_llm` | `string` | `null` | LLM alias for summarization (use a fast/cheap one) |

```yaml
memory:
  type: compacting
  max_recent_messages: 6
  compress_threshold: 10
  summarize_batch_size: 4
  summarizer_llm: router
```

When `max_recent_messages < compress_threshold`, the requested recent tail is preserved and only the older eligible prefix is summarized. When the requested tail is at least the threshold, the runtime clamps it to `compress_threshold - min(max(summarize_batch_size, 1), compress_threshold)` so compression still makes batch progress. With the defaults `50 / 30 / 10`, the effective protected tail is 20 and each threshold cycle summarizes 10 messages. Compression, add, restore, clear, snapshot, and eviction operations are serialized so a summarizer cannot remove a stale message prefix. Snapshots preserve both the rolling summary and the summarized-message count.

### `token_budget`

Fine-grained control over how much memory contributes to the prompt. Used with `compacting` memory.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `total` | `u32` | - | Max tokens memory can contribute |
| `allocation.summary` | `u32` | - | Tokens reserved for rolling summary |
| `allocation.recent_messages` | `u32` | - | Tokens reserved for recent messages |
| `allocation.facts` | `u32` | - | Tokens reserved for extracted key facts |
| `allocation.relationships` | `u32` | `0` | Optional global cap for relationship prompt text. When set, `memory.relationships.injection.max_tokens` cannot exceed this value |
| `overflow_strategy` | `string` | - | `truncate_oldest`, `summarize_more`, or `error` |
| `warn_at_percent` | `u32` | - | Emit warning when usage exceeds this % |

```yaml
memory:
  type: compacting
  max_recent_messages: 8
  compress_threshold: 8
  summarize_batch_size: 4
  summarizer_llm: router
  token_budget:
    total: 4096
    allocation:
      summary: 1024
      recent_messages: 2048
      facts: 512
      relationships: 256
    overflow_strategy: truncate_oldest
    warn_at_percent: 70
```

### `actor_memory` (Cross-Session Actor Memory)

Track facts about each actor (user, player, other agent) across sessions. When enabled, runtime initialization requires storage with the `ActorFacts` capability, currently SQLite among the built-in backends. When the same actor returns, previously extracted facts are loaded and injected into the system prompt via `{{ actor_facts }}`.

```yaml
memory:
  actor_memory:
    enabled: true
    identification:
      method: from_context         # explicit (set via API/--actor) | from_context (read from context path)
      context_path: user.id        # dot path into context. used when method: from_context
    injection:
      mode: all                    # all | category | on_demand. default: all
      max_tokens: 800              # max tokens for injected facts. default: 800
      categories:                  # only inject these categories (when mode: category)
        - user_preference
    privacy:
      retention_days: 365          # auto-delete facts after N days. null = keep forever
      allow_deletion: true         # actor can request full data wipe. default: true
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable cross-session actor memory |
| `identification.method` | string | `explicit` | How to resolve actor ID. `explicit` = set via `set_actor_id()` or `--actor` flag. `from_context` = read from context path |
| `identification.context_path` | string | - | Dot path into context to read actor ID (when method is `from_context`) |
| `injection.mode` | string | `all` | `all` = inject all facts. `category` = inject only listed categories. `on_demand` = no auto-injection |
| `injection.max_tokens` | int | `800` | Maximum tokens for injected facts in the prompt |
| `privacy.retention_days` | int | - | Auto-delete facts older than N days. Omit for no expiry |
| `privacy.allow_deletion` | bool | `true` | Whether actors can request deletion of all their data |

### `facts` (Key Facts Extraction)

Extract structured facts from conversations using an LLM. Durable facts require the `ActorFacts` storage capability; file and Redis are snapshot-only, while SQLite supports facts. Fact content is always stored in English for consistent cross-language deduplication.

```yaml
memory:
  facts:
    enabled: true
    extractor_llm: router           # which LLM runs extraction. default: router
    auto_extract: true              # extract after every turn. default: true
    categories:                     # built-in categories to look for
      - user_preference             # "I prefer...", "I like..."
      - user_context                # "I am a...", "I work at..."
      - decision                    # "I decided to..."
      - agreement                   # "Yes, agreed"
    custom_categories:              # domain-specific categories
      - name: suspicion
        description: "Suspicious behavior observed"
    inject_in_context: true         # make facts available as {{ actor_facts }}. default: true
    max_facts: 50                   # max facts per actor before eviction. default: 50
    dedup:
      enabled: true                 # deduplicate against existing facts. default: true
      method: exact                 # exact (string similarity) | llm (semantic). default: exact
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable fact extraction |
| `extractor_llm` | string | router | Which named LLM to use for extraction |
| `auto_extract` | bool | `true` | Extract facts after every turn. Set `false` for manual extraction only |
| `categories` | list | `[]` | Built-in categories: `user_preference`, `user_context`, `decision`, `agreement` |
| `custom_categories` | list | `[]` | Domain-specific categories with `name` and `description` |
| `inject_in_context` | bool | `true` | Make facts available as `{{ actor_facts }}` in system prompt |
| `max_facts` | int | `50` | Maximum facts per actor. Lowest-priority facts are evicted when exceeded |
| `dedup.enabled` | bool | `true` | Deduplicate new facts against existing ones |
| `dedup.method` | string | `exact` | `exact` = run normalized string similarity locally after extraction. `llm` = include existing facts in the extraction prompt and trust the model to skip duplicates (no local post-filter) |

When `token_budget.allocation.facts` is set on the surrounding `memory:` block, that value is used as the effective cap for fact injection and overrides `actor_memory.injection.max_tokens`.

### `relationships` (Relationship Memory)

Track how the agent relates to each actor across sessions. Relationship memory is separate from facts: facts describe what the agent knows about an actor, while relationships describe the agent's stance toward that actor. `persistence.enabled: true` requires `ActorRelationships`, currently provided only by SQLite among the built-in backends.

```yaml
memory:
  relationships:
    enabled: true
    model: one_sided               # one_sided (default) | two_sided
    dimensions:
      - trust
      - sentiment
      - familiarity
      - rapport
    auto_update:
      enabled: true
      llm: router
      min_confidence: 0.6
      max_delta_per_turn: 0.3
      recent_messages: 6
    injection:
      enabled: true
      format: summary              # summary | scores_only | full
      max_tokens: 400
      prompt_variable: relationship_memory
      context_path: relationships.current_actor
    persistence:
      enabled: true
    notable_events:
      enabled: true
      max_per_actor: 50
      significance_threshold: 0.5
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable actor-scoped relationship memory |
| `model` | string | `one_sided` | Relationship semantics. `one_sided` tracks the agent's stance toward the actor. `two_sided` also tracks `perceived_actor_to_agent` and derives read-only `mutual` scores |
| `dimensions` | list or map | `[trust, sentiment, familiarity, rapport]` | Dimensions to track. List form uses built-in definitions. Map form lets you specify `description`, `min`, `max`, and `default` |
| `auto_update.enabled` | bool | `true` | Run relationship evaluation after successful turns |
| `auto_update.llm` | string | router | Named LLM used for evaluation |
| `auto_update.min_confidence` | number | `0.6` | Ignore proposed changes below this confidence |
| `auto_update.max_delta_per_turn` | number | `0.3` | Clamp per-turn dimension changes |
| `auto_update.recent_messages` | int | `6` | Number of recent messages sent to the evaluator |
| `injection.enabled` | bool | `true` | Inject relationship context and prompt variable |
| `injection.format` | string | `summary` | Prompt format: `summary`, `scores_only`, or `full` |
| `injection.max_tokens` | int | `400` | Prompt budget for relationship text |
| `injection.prompt_variable` | string | `relationship_memory` | Template variable for formatted relationship text |
| `injection.context_path` | string | `relationships.current_actor` | Context path where relationship scores are injected |
| `persistence.enabled` | bool | `true` | Persist by `(agent_id, actor_id)` when storage supports it |
| `notable_events.max_per_actor` | int | `50` | Max relationship events kept per actor |
| `notable_events.significance_threshold` | number | `0.5` | Minimum event significance to store |

Use full dimension objects when you need domain-specific scores:

```yaml
memory:
  relationships:
    enabled: true
    dimensions:
      trust:
        description: "How much the agent trusts the actor"
        min: -1.0
        max: 1.0
        default: 0.0
      motivation:
        description: "How motivated the student seems"
        min: 0.0
        max: 1.0
        default: 0.5
```

When an actor is active, values are available at `relationships.current_actor.*`:

- `relationships.current_actor.trust`
- `relationships.current_actor.sentiment`
- `relationships.current_actor.dimensions.trust`
- `relationships.current_actor.agent_to_actor.trust`
- `relationships.current_actor.perceived_actor_to_agent.trust` (two-sided mode)
- `relationships.current_actor.mutual.trust` (two-sided mode, derived/read-only)
- `relationships.current_actor.interaction_count`

The shortcut paths such as `relationships.current_actor.trust` stay compatible and refer to `agent_to_actor` scores.
In two-sided mode, automatic evaluator updates only write `agent_to_actor` and `perceived_actor_to_agent`; `mutual` is derived by the runtime from those two stored perspectives.
This makes relationship scores usable by persona secrets, state guards, tool conditions, and templates.

```yaml
persona:
  secrets:
    - content: "Confidential detail"
      reveal_conditions:
        context:
          relationships.current_actor.trust:
            gte: 0.8
```

Prompt templates can use `{{ relationship_memory }}`:

```yaml
system_prompt: |
  You are a helpful assistant.
  {% if relationship_memory %}
  Relationship context:
  {{ relationship_memory }}
  {% endif %}
```

### `session` (Session Metadata)

Static metadata and TTL for the agent's sessions.

```yaml
memory:
  session:
    tags: [support, tier-1]          # freeform tags for filtering
    ttl_seconds: 86400               # session expiration (24h). null = no expiry
```

Tags and TTL are persisted alongside each session snapshot when `save_session()` is called and the backend supports `SessionMetadata`. Filtered listings require `SessionFiltering`, and explicit cleanup requires `ExpiryCleanup`; unwrapped SQLite currently provides all three among built-ins. File and Redis save snapshots without generic session metadata. Redis's top-level `storage.ttl_seconds` independently applies native key TTL and does not enable generic filtered listing or cleanup.

### Complete Actor Memory Example

```yaml
storage:
  type: sqlite
  path: "./agent_memory.db"

memory:
  type: compacting
  max_recent_messages: 20
  compress_threshold: 15
  summarizer_llm: router
  token_budget:
    total: 4000
    allocation:
      summary: 1200
      recent_messages: 2000
      facts: 800
      relationships: 400
  actor_memory:
    enabled: true
    identification:
      method: from_context
      context_path: user.id
    injection:
      mode: all
      max_tokens: 800
  facts:
    enabled: true
    extractor_llm: router
    auto_extract: true
    categories:
      - user_preference
      - user_context
      - decision
    max_facts: 50
  relationships:
    enabled: true
    dimensions:
      - trust
      - sentiment
      - familiarity
      - rapport
    injection:
      format: summary
      prompt_variable: relationship_memory
  session:
    ttl_seconds: 604800
```

Use `{{ actor_facts }}` in `system_prompt` to inject the formatted fact list:

```yaml
system_prompt: |
  You are a helpful assistant.
  {% if actor_facts %}
  What you know about this person:
  {{ actor_facts }}
  {% endif %}
  {% if relationship_memory %}
  Your relationship with this person:
  {{ relationship_memory }}
  {% endif %}
```

---

## Storage

Persist sessions across restarts. Without storage, conversation history is lost when the process exits. Unsupported optional operations return a typed storage-capability error rather than a successful no-op or ordinary empty result.

| Backend | Snapshots | Metadata | Filtering | Cleanup | Facts | Relationships | Actor deletion |
|---------|-----------|----------|-----------|---------|-------|---------------|----------------|
| File | Yes | No | No | No | No | No | No |
| Redis | Yes | No | No | No | No | No | No |
| SQLite | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| `NoopStorage` (Rust test backend) | No | No | No | No | No | No | No |

`NamespacedStorage` is used internally for shared spawned-agent storage. It derives safe capabilities from the inner backend but never forwards backend-global expiry cleanup. File and Redis are snapshot-only in the v1 capability contract. Redis remains Experimental; its native key TTL and backend-specific helpers do not add generic metadata, filtering, or cleanup capabilities.

### `type: none`

No persistence (default).

```yaml
storage:
  type: none
```

### `type: file`

Save sessions as files on disk.

| Field | Type | Description |
|-------|------|-------------|
| `path` | `string` | Directory path for session files |

```yaml
storage:
  type: file
  path: "./data/sessions"
```

### `type: sqlite`

Persist sessions in a SQLite database.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | `string` | - | Path to the `.db` file |
| `table` | `string` | `null` | Custom table name |

```yaml
storage:
  type: sqlite
  path: "./agent_sessions.db"
  table: "custom_sessions"
```

### `type: redis`

Persist sessions in Redis.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | `string` | - | Redis connection URL |
| `prefix` | `string` | `"agent:"` | Key prefix |
| `ttl_seconds` | `u64` | `null` | Positive time-to-live for session keys; the resulting deadline must fit the Redis index range |

```yaml
storage:
  type: redis
  url: "redis://localhost:6379"
  prefix: "myagent:"
  ttl_seconds: 86400
```

---

## Process Pipeline

Pre-process user input and post-process LLM output with a pipeline of stages. Stages run in order; no code changes needed.

### `process.input`

Runs **before** the LLM sees the message.

| Stage Type | LLM? | Description |
|------------|------|-------------|
| `normalize` | No | Trim whitespace, collapse spaces |
| `detect` | Yes | Detect language, sentiment, intent |
| `extract` | Yes | Pull structured entities from free text |
| `validate` | Optional | Rule-based (length) and LLM-based (criteria) checks |

```yaml
process:
  input:
    - type: normalize
      id: clean_input
      config:
        trim: true
        collapse_whitespace: true

    - type: detect
      id: detect_language
      config:
        llm: router
        detect: [language, sentiment]
        intents:
          - id: greeting
            description: "User is saying hello"
          - id: complaint
            description: "User is complaining"
        store_in_context:
          language: detected_language
          sentiment: detected_sentiment
          intent: detected_intent

    - type: extract
      id: extract_entities
      config:
        llm: router
        schema:
          email:
            type: string
            description: "User's email address"
          order_number:
            type: string
            description: "Order or reference number"
          urgency:
            type: enum
            values: [low, medium, high, critical]
            description: "How urgent the request seems"
        store_in_context: extracted

    - type: validate
      id: check_length
      config:
        rules:
          - min_length: 2
            on_fail:
              action: reject
          - max_length: 2000
            on_fail:
              action: truncate
```

### `process.output`

Runs **after** the LLM generates its response, **before** the user sees it. Only works in blocking mode (not streaming).

| Stage Type | LLM? | Description |
|------------|------|-------------|
| `sanitize` | Yes | Mask PII (email, phone, credit card) |
| `validate` | Yes | Check response quality against criteria |
| `format` | No | Append/prepend template text |

```yaml
process:
  output:
    - type: sanitize
      id: mask_pii
      config:
        llm: router
        pii:
          action: mask
          types: [email, phone, credit_card]
          mask_char: "*"

    - type: validate
      id: quality_check
      config:
        llm: router
        criteria:
          - "The response is helpful"
          - "No offensive content"
        threshold: 0.6
        on_fail:
          action: warn

    - type: format
      id: add_footer
      config:
        template: |
          {{ response }}

          ---
          Need more help?
```

### `process.settings`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `on_stage_error.default` | `string` | - | What to do when a stage fails: `continue` or `abort` |
| `debug.log_stages` | `bool` | `false` | Log each stage's input/output |
| `debug.include_timing` | `bool` | `false` | Log per-stage timing |

```yaml
process:
  settings:
    on_stage_error:
      default: continue
    debug:
      log_stages: true
      include_timing: true
```

---

## Error Recovery

Automatic retry, failover, and overflow handling - no code changes needed.

### `error_recovery.default`

Default retry policy for LLM and tool calls.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_retries` | `u32` | `0` | Number of retry attempts (0 = fail immediately) |
| `backoff.type` | `string` | - | `exponential` |
| `backoff.initial_ms` | `u64` | - | First retry delay in ms |
| `backoff.max_ms` | `u64` | - | Maximum delay cap |
| `backoff.multiplier` | `f64` | - | Multiplier per retry |
| `retry_on` | `list` | - | Retriable error types: `timeout`, `rate_limit`, `connection_error`, `server_error` |
| `no_retry_on` | `list` | - | Permanent error types: `invalid_api_key`, `invalid_request` |

```yaml
error_recovery:
  default:
    max_retries: 3
    backoff:
      type: exponential
      initial_ms: 500
      max_ms: 5000
      multiplier: 2.0
    retry_on:
      - timeout
      - rate_limit
      - connection_error
      - server_error
    no_retry_on:
      - invalid_api_key
      - invalid_request
```

### `error_recovery.llm`

LLM-specific failure policies.

#### `on_failure`

What to do after all retries are exhausted.

| Action | Description |
|--------|-------------|
| `fallback_llm` | Switch to a backup LLM |
| `fallback_response` | Return a static message |
| `error` | Propagate the error (default) |

```yaml
error_recovery:
  llm:
    on_failure:
      action: fallback_llm
      fallback_llm: fallback
```

```yaml
error_recovery:
  llm:
    on_failure:
      action: fallback_response
      message: "I am temporarily unavailable. Please try again later."
```

#### `on_context_overflow`

What to do when conversation history exceeds `max_context_tokens`.

| Action | Description |
|--------|-------------|
| `summarize` | Compress old messages with an LLM |
| `truncate` | Drop oldest messages |
| `error` | Fail (default) |

```yaml
error_recovery:
  llm:
    on_context_overflow:
      action: summarize
      summarizer_llm: summarizer
      max_summary_tokens: 300
      keep_recent: 4
```

```yaml
error_recovery:
  llm:
    on_context_overflow:
      action: truncate
      keep_recent: 10
```

### `error_recovery.tools`

Per-tool retry configuration.

```yaml
error_recovery:
  tools:
    default:
      max_retries: 2
      timeout_ms: 10000
```

---

## Tool Security

The `tool_security` block enforces safety constraints after tool availability has already been checked. Security can restrict, deny, require approval, or mark a granted tool unavailable, but it does not grant access to tools omitted from top-level `tools:`.

| Detail | Value |
|--------|-------|
| **Type** | `object` |
| **Default** | `{}` (disabled) |

```yaml
tool_security:
  enabled: true
  fail_closed: true
  default_timeout_ms: 5000
  tools:
    file_read:
      read_paths: [./crates, ./examples, ./website]
      blocked_paths: [./target, ./.git]

    file_write:
      write_paths: [./examples/fixtures/tool_examples]
      blocked_paths: [./target, ./.git]
      overwrite_existing: false
      create_parent_dirs: true
      max_changed_files: 1
      max_changed_lines: 100

    file_edit:
      write_paths: [./examples/fixtures/tool_examples]
      blocked_paths: [./target, ./.git]
      require_read_before_write: true
      max_replacements: 5
      max_changed_lines: 20

    patch:
      write_paths: [./examples/fixtures/tool_examples]
      blocked_paths: [./target, ./.git]
      require_read_before_write: true
      create_parent_dirs: true
      max_changed_files: 3
      max_changed_lines: 40

    command:
      allow_without_confirmation: true
      allowed_commands:
        - argv: [cargo, fmt, --all]
        - argv: [cargo, check, --workspace]
      working_dirs: [.]
      deny_shell: true
      deny_interactive: true
      timeout_ms: 120000
      max_output_chars: 30000
      env_passthrough: []

    sleep:
      timeout_ms: 60000
      config:
        max_duration_ms: 60000

    git_status:
      read_paths: [.]
      blocked_paths: [./.git]

    web_fetch:
      domains:
        allow:
          - docs.rs
          - ai-agents.rs
      allowed_schemes: [https]
      blocked_private_networks: true
      timeout_ms: 15000

    my_search_tool:
      read_paths: [./crates, ./examples]
      blocked_paths: [./target, ./.git]
      max_results: 100
      max_output_chars: 12000
      config:
        backend: tantivy
        index_path: ./search-index
        ranking: bm25

    http:
      rate_limit: 10
      timeout_ms: 10000
      domains:
        allow:
          - "api.example.com"
          - "httpbin.org"
        deny:
          - "internal.corp.net"
      operations:
        allow: [GET, HEAD]
        requires_approval: [POST, PUT, PATCH, DELETE]
      require_confirmation: true
```

Runtime-enforced per-tool policy fields:

| Field | Meaning |
|-------|---------|
| `enabled` | Set to `false` to make a granted tool unavailable |
| `require_confirmation` | Require HITL approval after hard denials pass; `require_approval` is accepted as an alias |
| `allow_without_confirmation` | Explicitly disable the default approval requirement for side-effecting calls inside policy |
| `rate_limit` | Maximum calls per minute for that tool |
| `timeout_ms` | Timeout for each `Tool::execute` invocation attempt; falls back to `tool_security.default_timeout_ms`. HITL wait, resource-lock wait, and separate retries are not one end-to-end timeout |
| `read_paths` / `allowed_paths` | Allowed local roots checked against a tool call's read path arguments |
| `write_paths` | Allowed local write roots for mutation tools |
| `working_dirs` | Allowed working directories for the `command` tool; `read_paths` never grant command cwd access, and `command` also requires `allowed_commands` or `command_templates` when `working_dirs` is set |
| `blocked_paths` | Paths that override any allowlist and stay blocked |
| `require_read_before_write` | Require a matching `file_read` version before mutating an existing file |
| `overwrite_existing` | Allow mutation tools to overwrite existing files |
| `create_parent_dirs` | Allow mutation tools to create missing parent directories |
| `no_write_policy` | Behavior when no explicit write policy exists: `deny` or `dry_run_only` |
| `max_replacements` | Maximum exact replacements for `file_edit` |
| `domains.allow` / `domains.deny` | Domain allow/deny policy for URL tools |
| `domains.requires_approval` | Domains that require HITL approval before execution; for `web_fetch`, this can approve the initial bound URL but not a later redirect target |
| `domains.unavailable` | Domains reported as unavailable without executing the tool |
| `allowed_schemes` | Allowed URL schemes for network tools such as `web_fetch` |
| `allowed_ports` | Allowed URL ports for network tools |
| `blocked_private_networks` | Block localhost, private IPs, link-local, and metadata-service targets |
| `operations.allow` / `operations.deny` | Allow or deny `operation`, `function`, or `method` argument values |
| `operations.requires_approval` | Operation values that require HITL approval |
| `allowed_commands` | Exact argv allowlist entries for the `command` tool |
| `command_templates` | Argv templates where literal segments must match and `{name}` segments act as wildcard variables |
| `env_passthrough` | Environment variable names that the `command` tool may accept from call arguments |
| `deny_shell` / `deny_interactive` | Reject shell syntax and interactive command execution |
| `allow_command_escalation` | Allow approval-based escalation outside the exact argv allowlist |
| `commands.allow` / `commands.deny` | Legacy fixed command identity matching for process-backed tools |
| `commands.requires_approval` | Legacy command identities that require HITL approval |
| `max_results` | Positive maximum rows, matches, or entries exposed through `ctx.limits.max_results`; zero is invalid |
| `max_file_size_bytes` | Maximum local file bytes exposed through `ctx.limits.max_file_size_bytes` |
| `max_output_chars` | Maximum model-facing output characters exposed through `ctx.limits.max_output_chars` |
| `max_response_bytes` | Maximum bytes for each network response exposed through `ctx.limits.max_response_bytes`; `web_fetch` applies it independently to redirect and final responses |
| `max_redirects` | Maximum redirect hops exposed through `ctx.limits.max_redirects` |
| `max_changed_files` / `max_changed_lines` | Common caps reserved for mutation tools |
| `config` | Host-supplied custom settings exposed as `ToolExecutionContext.custom_config`; not shown in model-facing schemas. Current built-in config: `sleep.config.max_duration_ms` |

Policy cap fields are applied as upper bounds or defaults before execution for built-ins that support the corresponding input, and they are passed to all tools in `ToolExecutionContext.limits`. Examples: `grep.max_output_chars`, `git_diff.max_output_chars`, `web_fetch.max_response_bytes`, `web_fetch.max_redirects`, `file_edit.max_replacements`, and `patch.max_changed_lines`. Optional-path tools such as `glob`, `grep`, `git_status`, `git_diff`, and `diagnostics` declare `path: "."` for policy checks when the call omits `path`. `patch` declares `base_path: "."`. Path policy resolves existing entries and nearest existing ancestors for both candidates and configured roots before containment decisions. Relative policy roots are anchored to the canonical host-owned workspace and fail closed if an existing ancestor resolves outside it; explicit absolute roots authorize their resolved locations. Dangling candidate or policy-root symlinks fail closed. Allow rules require resolved containment, while deny, unavailable, and approval rules match either lexical or resolved paths so a symlink cannot weaken a restriction while the checked topology remains unchanged. For `web_fetch`, an initial URL matching `domains.requires_approval` requires `Approved` or `Modified` evidence in the shared execution context, while direct execution without that evidence fails before DNS, cache lookup, or transport. A redirect into such a domain is blocked before the next DNS or transport request. The tool re-checks configured scheme, port, domain, and public-address policy on every redirect target. Its process-local response cache is limited to 128 entries and lazily removes expired entries. A positive `cache_ttl_seconds` value sets an entry's lifetime when the response is stored; cache hits do not refresh that expiry. Cache reuse also requires the stored redirect count to satisfy the current effective redirect limit; compatible hits avoid the HTTP transport request but repeat current DNS/IP and URL-policy validation. `max_response_bytes` applies independently to every redirect or final response rather than cumulatively across the chain. The default transport connects only to that request's validated addresses with proxies disabled. Custom low-level transports must return one response per call without automatically following redirects, honor the per-response byte limit while reading, and preserve their own validated-address and host-egress boundary.

Approval does not freeze an old authorization decision. Host-backed `command`, `diagnostics`, and `web_search` availability is checked after initial policy and before HITL. After approval, the runtime resolves the final tool once, reapplies current policy caps to the approved arguments, and recomputes classification and resource keys. It then verifies current scope, emergency control, provider availability, policy, and approval binding before lock acquisition. State-scope evaluation records the state generation that authorized the call. After waiting for locks, a changed policy, runtime-control, or state generation fails closed and one atomic rate-limit admission occurs immediately before invocation. The runtime executes the same resolved tool object and records the observed registry version as evidence rather than claiming an atomic registry snapshot. An initial hard denial returns immediately and is not revived if policy later becomes permissive; a call that was awaiting approval is denied if the final policy becomes restrictive.

The effective tool timeout starts around each admitted `Tool::execute` invocation attempt. HITL and resource-lock waits occur before that timeout, and every safely retryable attempt receives a separate timeout; `timeout_ms` is not a total deadline for the complete tool request. The `command.timeout_ms` input separately bounds its direct child process, while eval turn timeouts bound a whole agent turn. None of these timeout layers guarantees rollback of filesystem, process, network, custom-tool, or other external effects. `on_tool_start` and `on_tool_complete` describe the executor request lifecycle, not individual retry invocations; lifecycle hooks can observe a finalized request that records `executed: false`. `ToolExecutionRecord.executed` is the authority for whether the implementation was invoked.

Calls classified as concurrency-safe, including mutation dry runs, do not acquire resource locks. Non-concurrency-safe path operations use one conservative shared path-mutation lock in addition to normalized exact resource keys, so v1 favors correctness over parallel filesystem mutation. Non-concurrency-safe calls without a concrete resource use one shared unbound-side-effect lock. Parent and spawned runtimes share the lock table. Resource guards are released before completion hooks and before fallback re-enters the normal authorization path.

Custom tools should not read `tool_security` YAML directly. Put framework-common policy fields at the top level of the tool policy and put tool-specific settings under `config`. The runtime passes common caps through `ToolExecutionContext.limits` and passes `config` through `ToolExecutionContext.custom_config`. When `fail_closed: true`, a custom tool is denied before execution if configured path, domain, command, operation, or result-limit policy cannot be applied because the tool exposes no matching `policy_bindings()`.

```yaml
tools:
  - name: catalog_search

tool_security:
  enabled: true
  fail_closed: true
  tools:
    catalog_search:
      max_results: 2          # becomes ctx.limits.max_results
      max_output_chars: 4000  # becomes ctx.limits.max_output_chars
      config:
        backend: memory       # becomes ctx.custom_config.backend
        tenant: demo-store    # becomes ctx.custom_config.tenant
```

For mutation and command built-ins, prefer `tool_security.enabled: true` with `fail_closed: true`. `read_paths` never authorize writes; use `write_paths` or a write-capable path policy for actual mutation. Pair `command` with exact `allowed_commands` or `command_templates` and explicit `working_dirs`; `commands.allow` is legacy identity matching and is not sufficient for argv execution. The stable examples use argv form because the string `command` form is compatibility-only and rejects shell syntax by default. Process-backed command execution starts from an empty environment, adds only `env_passthrough` values, bounds output before collection completes, and redacts sensitive argv values in records.

Legacy `allowed_domains`, `blocked_domains`, and `allowed_paths` still work for compatibility. New YAML should prefer `domains.*`, `write_paths`, and `working_dirs`.

Compatibility note: before the explicit-grant change, some release-candidate projects relied on omitted top-level `tools:` exposing every registered tool. That is no longer true. Omit `tools:` or set `tools: []` for a no-tools agent, and list every ordinary tool explicitly when you want it to be callable. User-visible RC-to-stable changes are recorded under `Unreleased` in the project changelog.

---

## HITL (Human-in-the-Loop)

Pause risky tool execution and ask a human for approval before proceeding.

`ask_user` is not HITL approval. HITL approves a risky action that policy already allows to ask about. `ask_user` is a normal tool used for preference or clarification questions through a host question handler.

### Top-Level Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_timeout_seconds` | `u64` | `300` | Seconds to wait for human response |
| `on_timeout` | `string` | `"reject"` | Action on timeout: `reject`, `approve`, `error` |

```yaml
hitl:
  default_timeout_seconds: 120
  on_timeout: reject
```

### Per-Tool Approval - `hitl.tools.<name>`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `require_approval` | `bool` | `false` | Block until human decides |
| `approval_context` | `list` | all args | Which tool args to show in the prompt |
| `approval_message` | `string` or `map` | auto-generated | Jinja2 template or multi-language map |

```yaml
hitl:
  tools:
    http:
      require_approval: true
      approval_context:
        - method
        - url
      approval_message: "Approve {{ method }} request to {{ url }}?"
```

### Multi-Language Approval Messages

`approval_message` can be a map of language codes for localized prompts.

```yaml
hitl:
  tools:
    http:
      require_approval: true
      approval_context: [method, url]
      approval_message:
        en: "Approve {{ method }} request to {{ url }}?"
        ko: "{{ url }}에 {{ method }} 요청을 승인하시겠습니까?"
        ja: "{{ url }}への{{ method }}リクエストを承認しますか？"
        description: "HTTP request approval"
```

### `hitl.message_language`

Controls how the framework picks which language to show the approval message in.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strategy` | `string` | `"auto"` | Primary detection: `auto`, `approver`, `user`, `explicit`, `llm_generate` |
| `fallback` | `list` | `[approver, user, explicit, llm_generate]` | Ordered fallback strategies |
| `explicit` | `string` | `null` | Language code for `explicit` strategy |

```yaml
hitl:
  message_language:
    strategy: auto
    fallback: [approver, user, explicit]
    explicit: en
```

### Condition-Based Approval - `hitl.conditions`

Trigger approval based on argument values, not tool identity. Only fires when the named field exists in the tool's arguments.

```yaml
hitl:
  conditions:
    - name: state_changing_http
      when: "method in [POST, PUT, DELETE, PATCH]"
      require_approval: true
      approval_message: "Approve {{ method }} request to {{ url }}?"
```

Supported `when` operators:
- Numeric: `>`, `<`, `>=`, `<=`, `==`, `!=`
- String: `in [...]`, `not in [...]`

### State-Scoped HITL - `hitl.states.<name>`

Override HITL behavior for specific states.

```yaml
hitl:
  states:
    browsing:
      tools:
        http:
          require_approval: false
    editing:
      tools:
        http:
          require_approval: true
```

---

## Reasoning

Controls how the agent thinks through complex problems before responding.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `string` | `"none"` | `none`, `cot`, `react`, `plan_and_execute`, `auto` |
| `judge_llm` | `string` | `null` | LLM alias for judging reasoning quality |
| `output` | `string` | `"hidden"` | `hidden`, `visible`, `tagged` - can be overridden per state |
| `max_iterations` | `u32` | `5` | Max reasoning iterations (capped at the lower of this and agent-level `max_iterations`) |

```yaml
reasoning:
  mode: auto
  judge_llm: router
  output: tagged
  max_iterations: 5
```

### Reasoning Modes

| Mode | Description |
|------|-------------|
| `none` | No explicit reasoning |
| `cot` | Prompt injection: appends a "think step by step" instruction and parses `<thinking>` tags from the output. Useful for non-thinking models. Redundant for thinking models - use `llm.reasoning: true` instead. |
| `react` | Prompt injection variant: structures the tool-use loop as Thought -> Action -> Observation. Same caveat as `cot`. |
| `plan_and_execute` | Real orchestration: generates a structured plan (JSON steps), executes each step, synthesizes the result. Adds value regardless of model type. |
| `auto` | Judge LLM classifies each input and picks the best mode |

Note: `cot` and `react` are prompt-level techniques, not native model thinking. Thinking models (o3, o4-mini, gpt-5.4, Claude with extended thinking) already reason internally via API-level reasoning tokens configured in the [LLM section](#single-llm-llm-shorthand) (`reasoning`, `reasoning_effort`, `reasoning_budget_tokens`). A future version will wire `mode: cot` to native thinking when the model supports it, with prompt injection as fallback.

### `reasoning.planning`

Extra settings for `plan_and_execute` mode.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `planner_llm` | `string` | `null` | LLM alias for plan generation |
| `max_steps` | `u32` | `10` | Maximum plan steps to execute |
| `available.tools` | `"all"` or `list` | `"all"` | Tools the planner may use. `"all"` allows every tool available to the agent. A list restricts to those IDs only. |
| `available.skills` | `"all"` or `list` | `"all"` | Skills the planner may use. Same semantics as `available.tools`. |
| `reflection.enabled` | `bool` | `true` | Enable plan-level reflection on step failures |
| `reflection.on_step_failure` | `string` | `"replan"` | `replan`, `abort`, `skip`, or `continue` |
| `reflection.max_replans` | `u32` | `2` | Maximum replan attempts when steps fail |

```yaml
reasoning:
  mode: plan_and_execute
  planning:
    planner_llm: router
    max_steps: 10
    available:
      tools: [calculator, datetime]
      skills: [math_helper]
    reflection:
      enabled: true
      on_step_failure: replan
      max_replans: 2
```

When `reflection.enabled` is `true` and a plan step fails, the runtime checks `on_step_failure`:

- **replan** - generate a fresh plan and retry (up to `max_replans` times).
- **abort** - stop execution immediately.
- **skip** / **continue** - failed steps are skipped during execution (no retry loop).

If all replans are exhausted the plan is marked `Failed` with the IDs of the failing steps.
Multi-step plan output is synthesized into a coherent response via the LLM instead of returning only the last step's result.

### State-Level Reasoning Override

Individual states can override the global reasoning mode.

```yaml
states:
  states:
    analysis:
      prompt: "Analyze the data carefully."
      reasoning:
        mode: cot
        output: visible
    quick_answer:
      prompt: "Give a quick response."
      reasoning:
        mode: none
```

### Skill-Level Reasoning Override

Individual skills can override reasoning and reflection.

```yaml
skills:
  - id: deep_analysis
    description: "Perform deep analysis"
    trigger: "When user asks for analysis"
    reasoning:
      mode: cot
    reflection:
      enabled: true
    steps:
      - prompt: "Analyze this thoroughly: {{ user_input }}"
```

---

## Reflection

Self-evaluation: the agent checks its own response against criteria and retries if quality is too low.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` or `string` | `false` | `true`, `false`, or `"auto"` |
| `evaluator_llm` | `string` | `null` | LLM alias for evaluation |
| `max_retries` | `u32` | `2` | Maximum re-generation attempts |
| `pass_threshold` | `f64` | `0.7` | Confidence threshold (0.0-1.0). The LLM must say PASS *and* report confidence at or above this value. |
| `criteria` | `list` | `[]` | Natural-language quality criteria |

```yaml
reflection:
  enabled: auto
  evaluator_llm: router
  max_retries: 2
  pass_threshold: 0.7
  criteria:
    - "Response directly addresses the user's question"
    - "Response is complete and accurate"
    - "Response is helpful and clear"
```

---

## Disambiguation

Detect ambiguous user messages and ask clarifying questions before proceeding. Top-level `disambiguation.enabled: true` must create the base manager; state and skill settings only override an active manager and cannot enable the subsystem by themselves.

> **Note:** Disambiguation relies on the router LLM to detect ambiguity, classify the type, and generate clarification questions. Fast lower-cost models such as `gpt-5.4-nano` may misclassify ambiguity types or ignore style instructions. Use `gpt-5.4-mini` or a stronger model for the router if disambiguation quality matters.

The detector LLM returns a confidence score (0.0-1.0) for how clear and actionable the user's intent is. The runtime preserves that raw score and applies the effective threshold after resolving layered overrides: skill, then state, then the agent-level detection threshold. Messages scoring below the effective threshold trigger clarification. `required_clarity` remains a separate hard gate when the detector reports a configured field as missing.

### `disambiguation.enabled`

| Detail | Value |
|--------|-------|
| **Type** | `bool` |
| **Default** | `false` |

### `disambiguation.detection`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm` | `string` | `"router"` | LLM alias for detection |
| `threshold` | `f64` | `0.7` | Confidence cutoff (0.0-1.0). Messages below this trigger clarification. Higher = more sensitive and asks more often. |
| `aspects` | `list` | `[missing_target, missing_action, missing_parameters, vague_references]` | Which ambiguity types to check for |
| `prompt` | `string` | _(none)_ | Optional custom detection prompt. Replaces the built-in prompt sent to the detection LLM. Supports `{{ threshold }}`, `{{ effective_threshold }}`, and `{{ required_clarity }}` placeholders in addition to the existing context placeholders. |

Available aspects:
- `missing_target` - unclear what the user is referring to ("Send it" - send what?)
- `missing_action` - unclear what action to take ("The report" - do what with it?)
- `missing_parameters` - key details are missing ("Book a flight" - when? where?)
- `multiple_intents` - message contains multiple possible requests ("Cancel" in a state with multiple intent-labeled transitions)
- `vague_references` - pronouns or references without context ("Do that again" - do what?)
- `implicit_context` - assumes shared knowledge ("The usual" - what is "the usual"?)

### `disambiguation.clarification`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `style` | `string` | `"auto"` | How to ask: `auto`, `options`, `open`, `yes_no`, `hybrid` |
| `llm` | `string` | _(none)_ | Optional LLM alias for generating clarification questions. Falls back to the detection LLM if not set |
| `max_options` | `u32` | `4` | Max choices in options/hybrid style |
| `include_other_option` | `bool` | `true` | Add an "Other" freeform choice to options/hybrid |
| `max_attempts` | `u32` | `2` | Max clarification exchanges before giving up. The initial question counts as attempt 1 |
| `on_max_attempts` | `string` | `"proceed_with_best_guess"` | Action when limit is reached |

Clarification styles:

| Style | Behavior |
|-------|----------|
| `auto` | LLM picks the best format based on ambiguity type (default) |
| `options` | Multiple choice with labeled options (A, B, C) |
| `open` | Single open-ended question, no options |
| `yes_no` | Single yes/no confirmation question |
| `hybrid` | Options plus a freeform "or describe what you need" |

`on_max_attempts` actions:

| Action | Behavior |
|--------|----------|
| `proceed_with_best_guess` | Continue with the best interpretation (default) |
| `apologize_and_stop` | Apologize and drop the request |
| `escalate` | Trigger HITL approval flow (requires `hitl:` config) |

If the user abandons clarification mid-flow (e.g. "forget it", "never mind") the framework detects this and cancels the pending question gracefully.
If the user switches to a different topic during clarification, the new input is processed from scratch instead of being consumed as a clarification response.

### `disambiguation.context`

Controls what information is fed into the detection prompt for context-aware analysis.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `recent_messages` | `usize` | `5` | How many recent messages to include |
| `include_state` | `bool` | `true` | Include current state name and prompt |
| `include_available_tools` | `bool` | `true` | List tool names in detection prompt |
| `include_available_skills` | `bool` | `true` | List skill triggers in detection prompt |
| `include_user_context` | `bool` | `true` | Include runtime user context |

### `disambiguation.skip_when`

Conditions that bypass disambiguation entirely. No detection LLM call is made when a condition matches.

| Type | Fields | LLM call? | Description |
|------|--------|-----------|-------------|
| `social` | - | Yes | Greetings, thanks, goodbyes |
| `short_input` | `max_chars` | No | Messages under `max_chars` characters |
| `answering_agent_question` | - | Yes | User replying to the agent's last question (LLM verifies the response is actually an answer) |
| `complete_tool_call` | - | Yes | Direct tool invocations like "What is 2+2?" |
| `in_state` | `states` (list) | No | Skip when in specific named states |
| `custom` | `condition` (string) | Yes | Arbitrary LLM-evaluated condition |

### State-Level Disambiguation Override

A state can override agent-level disambiguation settings. Useful for sensitive states (e.g. payment) that need higher clarity. When `require_confirmation: true`, an ambiguous request that has been clarified remains pending: the configured clarification LLM (or its router fallback) asks one language-matched yes/no question for the resolved interpretation, and the runtime does not redispatch or execute a pending skill until the user semantically confirms it. Rejection or abandonment cancels the pending request, and a topic switch is processed as fresh input.

| Field | Type | Description |
|-------|------|-------------|
| `enabled` | `bool` | Enable or disable disambiguation in this state when the manager is active |
| `threshold` | `f64` | Override the agent-level confidence cutoff |
| `require_confirmation` | `bool` | After an ambiguous clarification resolves, require a language-matched semantic yes/no confirmation before redispatch or skill execution. Default: `false`. Clear input does not receive this extra question. |
| `required_clarity` | `list` | Fields that must be explicitly stated. If any are missing, clarification is forced regardless of confidence |

```yaml
states:
  states:
    payment:
      prompt: "Process the payment."
      disambiguation:
        threshold: 0.95
        require_confirmation: true
        required_clarity:
          - amount
          - recipient
```

`required_clarity` values from runtime context, state, and skill configuration are merged and deduplicated. It is a hard gate: if the detector reports any configured field in `what_is_unclear`, clarification is forced even if confidence is above the threshold.

`require_confirmation` does not activate disambiguation and does not affect `Clear` results. It applies only when the active top-level manager has already asked an ambiguity clarification and successfully resolved the user's answer. The resolved input and any pending skill route remain manager-owned until confirmation succeeds. Confirmation questions return `disambiguation.status: awaiting_confirmation`, while ordinary questions retain `awaiting_clarification`. Confirmation outcomes are recorded as privacy-safe structured events rather than added after final response hooks.

Pending clarification and confirmation are cleared by both runtime reset APIs and by snapshot or session restore. A confirmation is also invalidated when the originating state path or its authority generation changes, including a transition away and back to the same path. Pending exchanges use single-owner revisions, and reset, restore, timeout, or state-transition invalidation cannot be undone by an older in-flight clarification or confirmation result. Rejection and abandonment cancel the request, while a topic switch is handled as fresh input.

Confirmation uses one additional LLM call to generate the language-matched question and one later call to classify the user's semantic response. The literal confirmation reply, such as `Yes`, is not retained in conversation memory; the enriched request is retained instead, and privacy-safe structured events record requested, confirmed, rejected, abandoned, exhausted, or invalidated outcomes. If confirmation attempts are exhausted, `escalate` still escalates and `apologize_and_stop` stops. `proceed_with_best_guess` also stops because it cannot bypass an explicit confirmation gate.

### Skill-Level Disambiguation Override

A skill can declare its own disambiguation settings when top-level `disambiguation.enabled: true` has activated the base manager. After the skill router identifies a matching skill, the runtime runs a second disambiguation pass with the skill's override before executing the skill steps.

| Field | Type | Description |
|-------|------|-------------|
| `enabled` | `bool` | Enable skill-level disambiguation |
| `threshold` | `f64` | Override the agent-level confidence cutoff |
| `required_clarity` | `list` | Fields that must be explicitly stated |
| `clarification_templates` | `map` | Static question strings keyed by field name |

```yaml
skills:
  - id: transfer_money
    description: "Transfer money between accounts"
    trigger: "When user wants to send or transfer money"
    disambiguation:
      enabled: true
      threshold: 0.9
      required_clarity:
        - recipient
        - amount
      clarification_templates:
        missing_recipient: "Who should I send the money to?"
        missing_amount: "How much would you like to transfer?"
    steps:
      - prompt: "Process the transfer."
```

Template key lookup order:
1. Match by ambiguity type (`missing_target`, `missing_action`, `missing_parameters`, `vague_reference`)
2. Match by `what_is_unclear` field: for each unclear field (e.g. "recipient"), check `missing_recipient` then `recipient` against template keys
3. No match: fall through to LLM-generated question

> **Note:** Templates are static strings with a fixed language. When a template matches, it is used as-is with no LLM call. For multilingual clarification, omit templates and let the clarifier LLM generate the question instead.
>
> A custom detection prompt fully replaces the built-in schema and confidence instructions. To preserve threshold and required-field behavior, it must request the same structured JSON fields, especially `confidence` and `what_is_unclear`, and should use the effective-threshold and required-clarity placeholders where applicable.

### Full Example

```yaml
disambiguation:
  enabled: true
  detection:
    llm: router
    threshold: 0.7
    aspects:
      - missing_target
      - missing_action
      - missing_parameters
      - multiple_intents
      - vague_references
      - implicit_context
  clarification:
    style: auto
    max_options: 4
    include_other_option: true
    max_attempts: 2
    on_max_attempts: proceed_with_best_guess
  context:
    recent_messages: 5
    include_state: true
    include_available_tools: true
    include_available_skills: true
    include_user_context: true
  skip_when:
    - type: social
    - type: answering_agent_question
    - type: complete_tool_call
    - type: short_input
      max_chars: 5
```

---

## Observability & Tracing

The `observability` block enables privacy-safe latency, token, cost, and trace metrics. It is disabled by default and wraps LLM providers and tools when enabled, so normal chat, skills, process stages, disambiguation detection and clarification, facts, relationships, state transitions, HITL localization, reasoning, and orchestration calls are measured without application code.

```yaml
observability:
  enabled: true
  latency:
    track_llm: true
    track_tools: true
    track_skills: true
    track_orchestration: true
    track_hitl: true
    detailed_breakdown: false
  tokens:
    count_input: true
    count_output: true
    estimate_when_missing: true
    breakdown_by_component: false
  cost:
    enabled: true
    unknown_price_policy: omit
    pricing_file: ./pricing.yaml
    pricing:
      openai/gpt-5.4-nano:
        input_per_1k: 0.0002
        output_per_1k: 0.00125
  language:
    paths: [detected_language, input.language, user.language]
    fallback: unknown
  aggregation:
    dimensions: [model, purpose, language, state, background]
    percentiles: [0.5, 0.9, 0.95, 0.99]
    window_size: 1000
  privacy:
    include_prompts: false
    include_responses: false
    include_tool_args: false
    include_tool_outputs: false
    max_text_chars: 0
    hash_inputs: true
    redact_keys: [api_key, authorization, token, password, secret]
    redact_paths: [actor_facts, relationship_memory, persona.secrets]
  export:
    formats: [json, csv]
    path: ./observability_data/
    write_report: true
    write_raw_events: false
    raw_events_format: jsonl
  buffer:
    event_buffer: 4096
    raw_event_limit: 10000
    pending_branch_event_limit: 1024
    drop_on_full: true
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable telemetry collection |
| `latency.track_llm` | `bool` | `true` | Record LLM call duration |
| `latency.track_tools` | `bool` | `true` | Record tool call duration |
| `latency.track_skills` | `bool` | `true` | Record skill lifecycle events |
| `latency.track_orchestration` | `bool` | `true` | Record multi-agent orchestration lifecycle events |
| `latency.track_hitl` | `bool` | `true` | Record human approval lifecycle events |
| `tokens.count_input` | `bool` | `true` | Include input tokens in token totals |
| `tokens.count_output` | `bool` | `true` | Include output tokens in token totals |
| `tokens.estimate_when_missing` | `bool` | `true` | Estimate token usage when provider usage is absent and mark the source as estimated |
| `tokens.breakdown_by_component` | `bool` | `false` | Reserved for finer component tables; current reports group by configured dimensions |
| `cost.enabled` | `bool` | `true` | Estimate cost from token usage and configured pricing |
| `cost.unknown_price_policy` | `enum` | `omit` | `omit`, `zero`, or `error`; `error` records a cost error tag but does not fail the turn |
| `cost.pricing_file` | `string?` | `null` | Optional JSON/YAML pricing map resolved relative to the agent YAML file |
| `cost.pricing` | `map` | `{}` | Provider/model pricing keyed as `provider/model`; inline values override `pricing_file` |
| `language.paths` | `list` | common language paths | Dotted context paths used for the `language` dimension |
| `aggregation.dimensions` | `list` | `[model, purpose]` | Dimensions for aggregate tables; supports `agent`, `actor`, `model`, `provider`, `alias`, `purpose`, `language`, `state`, `tool`, `skill`, `orchestration_pattern`, `status`, `branch_status`, `runtime_optimization`, `commit_behavior`, `speculative`, and `background`; `runtime_optimization` is reported with the output key `optimization` |
| `aggregation.percentiles` | `list` | `[0.5, 0.9, 0.95, 0.99]` | Percentiles reported for latency |
| `aggregation.window_size` | `usize` | `1000` | Rolling event window used for aggregate metrics |
| `privacy.include_prompts` | `bool` | `false` | Retain redacted prompt text in raw event payloads |
| `privacy.include_responses` | `bool` | `false` | Retain redacted response text in raw event payloads |
| `privacy.include_tool_args` | `bool` | `false` | Retain redacted tool arguments |
| `privacy.include_tool_outputs` | `bool` | `false` | Retain redacted tool output |
| `privacy.max_text_chars` | `usize` | `0` | Maximum retained characters for text fields; `0` means no raw text retention |
| `privacy.hash_inputs` | `bool` | `true` | Store stable text hashes for correlation without raw text |
| `export.formats` | `list` | `[json]` | Output formats: `json`, `csv`, `jsonl`, `prometheus` |
| `export.path` | `string` | `./observability_data/` | Directory or file path for exported reports and events |
| `export.write_report` | `bool` | `true` | Write aggregate reports after each chat turn when observability is enabled |
| `export.write_raw_events` | `bool` | `false` | Write raw event files; use with privacy settings carefully |
| `export.raw_events_format` | `enum` | `jsonl` | Raw event export format: `jsonl` or `json` |
| `buffer.event_buffer` | `usize` | `4096` | Bounded in-memory event channel capacity |
| `buffer.raw_event_limit` | `usize` | `10000` | Maximum raw events retained for raw export |
| `buffer.pending_branch_event_limit` | `usize` | `1024` | Maximum delayed branch events retained before unfinalized branch events are counted as dropped |
| `buffer.drop_on_full` | `bool` | `true` | Drop and count events when buffers are full instead of blocking |

### Pricing file example

Use `cost.pricing_file` when several agents should share one pricing table. Relative paths are resolved from the agent YAML file directory. Inline `cost.pricing` entries override file-loaded entries with the same key.

Agent YAML:

```yaml
observability:
  enabled: true
  cost:
    enabled: true
    pricing_file: ./pricing.yaml
    unknown_price_policy: omit
    pricing:
      # Inline override for this agent. This wins over pricing.yaml.
      openai/gpt-5.4-nano:
        input_per_1k: 0.0002
        output_per_1k: 0.00125
```

`pricing.yaml`:

```yaml
openai/gpt-5.4-mini:
  input_per_1k: 0.00075
  output_per_1k: 0.0045
openai/gpt-5.4-nano:
  input_per_1k: 0.00025
  output_per_1k: 0.0015
```

Raw prompts, responses, tool arguments, tool outputs, context values, actor facts, relationship memory, persona secrets, approval details, tags, and error text are not retained by default. If raw payloads are enabled, configured keys and dotted paths are redacted recursively, text values are truncated on Unicode character boundaries, and stable hashes can be retained for correlation.

Common `purpose` labels include `main_response`, `skill_routing`, `skill_prompt`, `process_detect`, `process_extract`, `process_validate`, `process_transform`, `disambiguation_detection`, `disambiguation_clarification`, `state_transition_evaluation`, `context_extraction`, `summarization`, `reflection_decision`, `reflection_evaluation`, `plan_generation`, `plan_step`, `facts_extraction`, `relationship_update`, `hitl_localization`, `orchestration_routing`, `orchestration_aggregation`, and `orchestration_conversation`.

Prometheus support currently renders text exposition output and writes it to a `.prom` file when `export.formats` includes `prometheus`. There is no built-in scrape server in YAML or CLI mode yet. To connect Prometheus today, either export a `.prom` file for the node_exporter textfile collector or expose `agent.observability().render_prometheus()` from your own Rust HTTP endpoint.

---

## Runtime Optimization

The `runtime` block controls tool-schema prompt rendering and opt-in latency optimizations. `tool_schema_prompt_mode` defaults to `full`; use `compact` to include only tool names, descriptions, required fields, and property types in the generated prompt. Runtime optimization is disabled by default, so existing agents keep the same serial response behavior unless `runtime.optimization.enabled` is explicitly enabled.

```yaml
runtime:
  tool_schema_prompt_mode: full
  optimization:
    enabled: true
    max_speculative_llm_calls_per_turn: 0
    pre_response_deterministic_transitions: true
    pre_response_extractors: false
    speculative_state_transitions: false
    speculative_skill_routing: false
    speculative_reasoning_auto: false
    parallel_post_turn_memory: true
    parallel_orchestration_vote_extraction: true
    background_observability_export: false
    streaming_policy: preflight_only
    max_parallel_runtime_tasks: 4
    post_turn:
      facts:
        mode: background
        await_before_next_turn: same_actor
      relationships:
        mode: background
        await_before_next_turn: same_actor
      sessions:
        mode: inline_serial
        await_before_next_turn: always
      memory_compression:
        mode: inline_serial
        await_before_next_turn: always
      max_background_tasks: 16
      on_background_overflow: run_inline
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tool_schema_prompt_mode` | enum | `full` | Tool schema rendering for generated prompts: `full` includes complete JSON schema properties; `compact` includes names, descriptions, required fields, and property types |
| `enabled` | `bool` | `false` | Enable runtime optimization behavior |
| `max_speculative_llm_calls_per_turn` | `u32` | `0` | Per-turn cap for branch-managed speculative LLM calls; must be greater than zero when speculative flags are enabled and no larger than `max_parallel_runtime_tasks` |
| `pre_response_deterministic_transitions` | `bool` | `false` | Check transitions explicitly marked `timing: pre_response` before old-state response generation |
| `pre_response_extractors` | `bool` | `false` | Run current-state extractors before pre-response selection for all pre-response candidates; transition-level `run_extractors` keeps extraction scoped to that route |
| `speculative_state_transitions` | `bool` | `false` | Enables `timing: parallel` response-independent transition branches beside a main response draft |
| `speculative_skill_routing` | `bool` | `false` | Runs pure skill selection beside a main response draft; skill execution happens only after the skill branch wins |
| `speculative_reasoning_auto` | `bool` | `false` | Runs the auto reasoning judge beside a plain draft; non-`none` decisions discard the draft and run the committed reasoning path. Auto speculation requires capacity for both draft and judge calls |
| `parallel_post_turn_memory` | `bool` | `false` | Run inline facts and relationship maintenance concurrently when possible |
| `parallel_orchestration_vote_extraction` | `bool` | `false` | Run voting extraction concurrently while preserving declaration order; bounded by `max_parallel_runtime_tasks` |
| `background_observability_export` | `bool` | `false` | Reserved and rejected until immutable export snapshots are available |
| `streaming_policy` | enum | `preflight_only` | `preflight_only`, `buffer_until_routing_done`, or `disabled`; buffered routing requires `streaming.enabled: true` and positive `streaming.buffer_size`. Buffered speculative streaming currently applies to response-independent parallel state-transition routing, and the buffer limit applies while routing is unresolved |
| `max_parallel_runtime_tasks` | `usize` | `4` | Bounds optimized internal work such as vote extraction; must be greater than zero |
| `post_turn.facts.mode` | enum | `inline_serial` | `inline_serial`, `inline_parallel`, or `background` |
| `post_turn.facts.await_before_next_turn` | enum | `always` | `never`, `same_actor`, or `always` |
| `post_turn.relationships.mode` | enum | `inline_serial` | `inline_serial`, `inline_parallel`, or `background` |
| `post_turn.relationships.await_before_next_turn` | enum | `always` | `never`, `same_actor`, or `always` |
| `post_turn.sessions` | object | `inline_serial` / `always` | Reserved unless left at the default policy |
| `post_turn.memory_compression` | object | `inline_serial` / `always` | Reserved unless left at the default policy |
| `post_turn.max_background_tasks` | `usize` | `16` | Background maintenance queue limit |
| `post_turn.on_background_overflow` | enum | `run_inline` | `run_inline`, `drop`, or `error` |

### How to choose runtime optimization fields

Use the smallest flag that matches the latency problem you are solving. These settings are independent, but speculative settings share the same per-turn caps.

| Goal | Enable | Also configure | What happens |
|------|--------|----------------|--------------|
| Skip an old-state response when a guard or resolved intent already proves the next state | `pre_response_deterministic_transitions` | Mark the transition with `timing: pre_response`; use `guard` or `intent` | The runtime commits the transition before the main LLM call and answers from the new state. No speculative LLM branch is needed. |
| Let a response-independent natural-language route race a main draft | `speculative_state_transitions` | Mark the transition with `timing: parallel`; set `max_speculative_llm_calls_per_turn >= 2` and `max_parallel_runtime_tasks >= 2` | The runtime starts a main draft and a transition branch. If the transition wins, the draft is discarded before it can write memory or run tools. |
| Let skill selection race a normal response | `speculative_skill_routing` | Define skills and leave no pending skill clarification; set enough speculative call capacity for draft + router | The branch performs pure selection only. Skill disambiguation and skill steps run after the skill branch wins. Low caps fall back to serial skill routing. |
| Avoid waiting for auto reasoning on simple turns | `speculative_reasoning_auto` | Set `reasoning.mode: auto`; set capacity for both draft and judge calls | The plain draft and judge branch run together. `none` commits the draft; deeper modes discard the draft and run the committed reasoning path. Low caps fall back to the serial judge path. |
| Hide stale stream chunks while a parallel transition is unresolved | `streaming_policy: buffer_until_routing_done` plus `speculative_state_transitions` | Set `streaming.enabled: true`; set positive `streaming.buffer_size`; use a response-independent `timing: parallel` transition | Main-stream chunks are hidden while routing is unresolved. If the route wins, stale chunks are discarded. The buffer limit applies only while routing is unresolved. |
| Move future-turn actor memory work out of the response tail | `parallel_post_turn_memory` and/or `post_turn.*.mode: background` | Configure facts or relationships memory and choose `await_before_next_turn` freshness | Facts and relationship updates can run inline in parallel or in the background. Same-actor freshness can wait only when the same actor continues. |
| Speed up vote extraction in concurrent orchestration | `parallel_orchestration_vote_extraction` | Set `max_parallel_runtime_tasks` for the extraction batch size | Vote extraction runs in bounded parallel, then results are restored to declaration order before tie-breaking. |

Speculative settings should normally start with a small cap:

```yaml
runtime:
  optimization:
    enabled: true
    max_speculative_llm_calls_per_turn: 2
    max_parallel_runtime_tasks: 2
```

Use observability when enabling speculative settings so discarded token and cost exposure is visible:

```yaml
observability:
  enabled: true
  aggregation:
    dimensions: [branch_status, runtime_optimization, commit_behavior, speculative]
```

See [Concepts](@/docs/concepts.md#runtime-optimization) for the branch commit/discard model and [Evaluation](@/docs/evaluation.md#observability-results) for branch metric assertions.

Pre-response transitions are only safe when they do not need assistant response text. They must be deterministic, so `timing: pre_response` requires a `guard` or `intent` and must not include natural-language `when` text.

Parallel transitions are speculative and response-independent. They are accepted only when `runtime.optimization.enabled: true`, `speculative_state_transitions: true`, and `max_speculative_llm_calls_per_turn > 0`. A parallel transition may use a natural-language `when`, but the evaluation prompt uses only the current user input and context, not assistant response text.

```yaml
states:
  initial: greeting
  states:
    greeting:
      prompt: "Ask what the user needs."
      transitions:
        - to: billing
          guard:
            context:
              request.topic:
                eq: billing
          timing: pre_response
          requires_response: false
    billing:
      prompt: "Help with billing."
```

A speculative main draft is inert until it wins. Losing drafts are discarded without writing assistant memory, executing parsed tool calls, mutating context, or firing final response hooks. Plain drafts commit only when they preserve the equivalent serial path. A `committed` branch is the winner selected into the one committed runtime path; the label does not mean that an external transaction succeeded or that provider calls, tool effects, network requests, processes, or effects already observed before consumer drop can be rolled back. Forced reasoning modes use the serial path, and auto reasoning speculation runs only when both the draft and judge branch fit the configured caps. Branch-managed LLM calls are finalized in observability with `branch_status`, `optimization`, `commit_behavior`, `winner`, and `speculative` dimensions. Low speculative caps schedule only behavior-preserving branches; skipped branch families fall back to the serial committed path.

```yaml
runtime:
  optimization:
    enabled: true
    max_speculative_llm_calls_per_turn: 4
    speculative_state_transitions: true
    speculative_skill_routing: true
    speculative_reasoning_auto: true
    streaming_policy: buffer_until_routing_done
    max_parallel_runtime_tasks: 4
```

Buffered streaming with `buffer_until_routing_done` hides unresolved main-stream text while a parallel state-transition branch can still discard it. The buffer limit applies while routing is unresolved. Once the transition branch misses or fails, later chunks are not counted against that unresolved-route buffer. If the route wins, stale buffered text is dropped and the committed route response is emitted. If the main branch wins, buffered chunks are emitted only when they still match the committed response; otherwise the committed response content is emitted.

Background facts and relationship updates use actor-scoped ordering. Freshness waits are task-aware: `facts.await_before_next_turn` flushes pending fact tasks, and `relationships.await_before_next_turn` flushes pending relationship tasks. Use `same_actor` when low response tail latency is acceptable but the next turn from the same actor must see fresh memory.

---

## Streaming & Parallel Tools

### `streaming`

Stream LLM tokens to the user in real time instead of waiting for the full response.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `true` | Declare streaming availability for runtime optimization validation. It does not select the CLI mode or block a Rust host from calling `chat_stream()` directly. |
| `buffer_size` | `usize` | `256` | Stream chunk buffer size. For `buffer_until_routing_done`, this limits hidden chunks while routing is unresolved |
| `include_tool_events` | `bool` | `true` | Stream tool call events |
| `include_state_events` | `bool` | `true` | Stream state transition events |

```yaml
streaming:
  enabled: true
```

> **Note:** Content chunks can reach a streaming consumer before output processing, reflection, or transition replacement completes. `chat_stream_events()` exposes the processed authoritative response as its final event, but final processing does not retroactively sanitize provisional chunks that were already displayed.

`metadata.cli.streaming: true` selects streaming for the CLI, and `--stream` overrides that frontend preference. A Rust host can call legacy `chat_stream()` for the existing chunk-plus-`Done` contract or `chat_stream_events()` for provisional chunks followed by one authoritative `Final(AgentResponse)`. The CLI, TUI, and streamed eval turns use the complete event stream so final content, metadata, and committed tool-call records remain available.

### `parallel_tools`

Execute multiple tool calls concurrently when the LLM requests them in the same turn.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable parallel execution |
| `max_parallel` | `usize` | `5` | Max concurrent tool calls |

```yaml
parallel_tools:
  enabled: true
  max_parallel: 3
```

---

## MCP (Model Context Protocol)

MCP tools are declared inline in the `tools` list (not as a separate top-level block). See the [Tools section](#mcp-tool) for the full syntax.

### Quick Reference

```yaml
tools:
  - name: filesystem
    type: mcp
    transport: stdio         # stdio | http | sse
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "./"]
    startup_timeout_ms: 15000
    security:
      blocked_functions: []
    views:
      fs_read:
        functions: [read_file, search_files]
        description: "Read-only operations"
      fs_write:
        functions: [write_file, create_directory]
        description: "Write operations"
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Tool identifier, also the parent tool name |
| `type` | `string` | Must be `"mcp"` |
| `transport` | `string` | `stdio` (spawn process), `http`, or `sse` |
| `command` | `string` | Command to spawn (for `stdio`) |
| `args` | `list` | Command arguments |
| `env` | `map` | Environment variables for the spawned process |
| `startup_timeout_ms` | `u64` | Max time to wait for server startup |
| `security.blocked_functions` | `list` | Functions to block from discovery |
| `views` | `map` | Named subsets of discovered functions |

---

## Persona (Agent Identity)

The `persona:` section defines structured identity, personality traits, goals, secrets, and evolution rules for an agent. Persona is prepended to the system prompt automatically and coexists with `system_prompt`.

### Minimal Persona

```yaml
persona:
  identity:
    name: "Alex"
    role: "Customer Support"
    description: "Friendly support agent for general inquiries"
  traits:
    personality: [helpful, patient, professional]
    speaking_style: "warm, professional, concise"
```

### `persona.identity`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | **Required.** Display name |
| `role` | `string` | **Required.** Functional role description |
| `description` | `string` | Short one-liner for UI/API responses |
| `backstory` | `string` | Rich background text. Supports Jinja2 templates with context values |
| `affiliation` | `string` | Group, organization, team, faction, or department (domain-neutral) |

### `persona.traits`

| Field | Type | Description |
|-------|------|-------------|
| `personality` | `list` | Core personality descriptors (e.g., `[disciplined, suspicious, loyal]`) |
| `values` | `list` | What the agent cares about |
| `fears` | `list` | What the agent avoids |
| `speaking_style` | `string` | Speaking style instruction included verbatim in the prompt |

### `persona.goals`

| Field | Type | Description |
|-------|------|-------------|
| `primary` | `list` | Public goals included in the LLM prompt |
| `hidden` | `list` | Goals NOT included in the prompt. For programmatic access only (application code reads via `persona_manager()`) |

### `persona.secrets`

Secrets are information the agent withholds until context conditions are met. Each secret has a `content` string and optional `reveal_conditions`.

```yaml
persona:
  secrets:
    - content: "Investigating a smuggling ring"
      reveal_conditions:
        all:
          - context:
              relationships.current_actor.trust:
                gte: 0.8
          - context:
              actor.is_watch_member:
                eq: true
    - content: "Manual-only secret"
      # No reveal_conditions = never auto-revealed (API-only)
```

Conditions use the same typed matchers as state machine guards: `eq`, `neq`, `gt`, `gte`, `lt`, `lte`, `in`, `exists`. Combine with `all:` (every condition must pass) or `any:` (at least one).

### `persona.evolution`

Controls how the persona changes over time.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Allow `evolve()` calls from Rust API and hooks |
| `allow_llm_evolve` | `bool` | `false` | Also register a `persona_evolve` tool for the LLM (double opt-in) |
| `mutable_fields` | `list` | `[]` | Dot-notation paths that may be mutated (e.g., `traits.personality`, `goals.primary`) |
| `track_changes` | `bool` | `false` | Keep an audit trail of all mutations |

Valid `mutable_fields` paths: `identity.name`, `identity.role`, `identity.description`, `identity.backstory`, `identity.affiliation`, `traits.personality`, `traits.values`, `traits.fears`, `traits.speaking_style`, `goals.primary`, `goals.hidden`. Secrets paths are always rejected.

```yaml
persona:
  evolution:
    enabled: true
    allow_llm_evolve: true
    track_changes: true
    mutable_fields:
      - traits.personality
      - traits.speaking_style
      - goals.primary
```

### `persona.templates`

Reference a reusable persona template registered via the Rust API.

| Field | Type | Description |
|-------|------|-------------|
| `base` | `string` | Name of the registered persona template |
| `overrides` | `map` | Dot-notation field overrides applied on top of the base |

```yaml
persona:
  templates:
    base: "guard_base"
    overrides:
      identity.name: "Captain Tam"
      identity.backstory: "A former sailor turned guard."
  goals:
    primary: [protect_harbor]
```

### `persona.max_prompt_tokens`

Optional token cap for the persona section. When the full rendering exceeds this limit, a condensed format is used (name, role, personality, speaking style only - backstory, values, fears, goals, and secrets are dropped).

### Prompt Injection Order

Persona is injected **after** `prompt_mode` is applied, so it always survives `prompt_mode: replace`.

| `prompt_mode` | Resulting system prompt |
|---------------|------------------------|
| `append` (default) | `[persona] + [base_prompt] + [state_prompt]` |
| `replace` | `[persona] + [state_prompt]` |
| `prepend` | `[persona] + [state_prompt] + [base_prompt]` |

### Full Persona Example

```yaml
persona:
  identity:
    name: "Captain Elira"
    role: "Harbor Guard Captain"
    description: "A disciplined former soldier guarding the harbor"
    backstory: |
      Former soldier who served in the Eastern Campaign.
      Now guards the harbor after losing faith in the army.
    affiliation: "Harbor Watch"
  traits:
    personality: [disciplined, suspicious, loyal]
    values: [duty, order, justice]
    fears: [civil_unrest, betrayal]
    speaking_style: "formal military cadence, short clipped sentences"
  goals:
    primary:
      - protect_harbor
      - investigate_smuggling
    hidden:
      - "Find the spy within the Watch"
  secrets:
    - content: "Investigating a smuggling ring"
      reveal_conditions:
        all:
          - context:
              relationships.current_actor.trust:
                gte: 0.8
          - context:
              actor.is_watch_member:
                eq: true
  evolution:
    enabled: true
    mutable_fields:
      - traits.personality
      - traits.speaking_style
    track_changes: true
    allow_llm_evolve: false
  max_prompt_tokens: 400
```

---

## Spawner (Dynamic Agent Spawning)

The `spawner:` section lets a parent agent create and manage child agents at runtime. Child agents can be spawned from inline YAML, `AgentSpec` objects, or named Jinja2 templates. A central registry tracks spawned agents and provides inter-agent messaging.

Four management tools are automatically registered when `spawner:` is present: `spawn_agent`, `send_agent_message`, `list_agents`, and `remove_agent`. Grant them with `management_tools` or under top-level `tools:` when the parent LLM should call them.

### Basic Configuration

```yaml
spawner:
  management_tools: true
  shared_llms: true
  max_agents: 50
  name_prefix: "npc_"
  shared_context:
    world_name: "Eldoria"
    setting: "medieval fantasy"
  allowed_tools:
    - echo
    - calculator
    - datetime
  templates:
    simple_npc: |
      name: "{{ name }}"
      system_prompt: "You are {{ name }}, a {{ role }} in {{ context.world_name }}."
      llm:
        provider: openai
        model: gpt-5.4-nano
```

### File-Based Templates

Templates can reference separate `.yaml` files instead of inline strings. File paths are resolved relative to the parent YAML's directory.

```yaml
spawner:
  templates:
    # File-based template (recommended for complex templates)
    npc_base:
      path: ./templates/npc_base.yaml
    # Inline template (backward compatible)
    simple_npc: |
      name: "{{ name }}"
      system_prompt: "You are {{ name }}."
```

### Field Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `management_tools` | `bool` or `list` | `false` | Grant management tools. `true` grants all four; a list grants only selected IDs. |
| `shared_llms` | `bool` | `false` | Use the inherited parent registry as authoritative. Every child-referenced alias must exist in it. |
| `shared_storage` | storage object | - | Create one backend and give each child a collision-safe `NamespacedStorage` view. |
| `max_agents` | integer | - | Limit reserved or registered spawner-managed slots, including in-flight builds. Omitted means unlimited by this counter. |
| `name_prefix` | `string` | - | Auto-name agents (e.g. `npc_001`, `npc_002`) |
| `shared_context` | `map` | - | Key-value pairs injected into every spawned agent's template as `context.*` |
| `allowed_tools` | `list` | - | Allow child top-level tool declarations. Any disallowed declaration rejects the complete child. |
| `templates` | `map` | - | Named YAML templates. Values are either inline strings or `{ path: "..." }` objects. |
| `auto_spawn` | `list` | `[]` | Child IDs and YAML paths to build and register during parent configuration. Any child failure aborts configuration. |
| `orchestration_tools` | `bool` or `list` | `false` | Register and grant all five orchestration tools, or only selected IDs. |

### Spawner Tools

| Tool | Description |
|------|-------------|
| `spawn_agent` | Spawn an agent from a description (via LLM-generated YAML or named template) |
| `send_agent_message` | Send a message to another registered agent and get its response |
| `list_agents` | List all registered agents as JSON |
| `remove_agent` | Remove an agent from the registry by ID |

### Management Tools

`management_tools` registers and grants dynamic agent-management tools. It accepts `true` for all four tools, or a list of selected tool IDs.

```yaml
# Each YAML document below is a separate alternative.
spawner:
  management_tools: true

---
# Or selectively:
spawner:
  management_tools:
    - spawn_agent
    - send_agent_message
    - list_agents
```

### Template Variables

In templates, caller-provided variables are top-level (`{{ name }}`, `{{ role }}`). Shared context values use the `context.` prefix (`{{ context.world_name }}`). Templates are rendered with Jinja2 (minijinja).

### Security

`allowed_tools` is a declaration allowlist, not a complete sandbox. Built-in names are compared by canonical ID; any child top-level tool declaration outside the allowlist rejects the child instead of stripping its spec. Hosts still own provider credentials, network and filesystem isolation, storage policy, and generated grants from other features.

Every child ID must be at most 128 bytes, use only ASCII letters, digits, `_`, `-`, or `.`, and must not be empty, `.`, `..`, end in `.`, or use a Windows-reserved first stem. Active nested child spawners are rejected in v1 across dynamic spawn, auto-spawn, and restore paths.

### Auto-Spawn (Pre-Spawn Agents at Startup)

`auto_spawn` creates agents from YAML files when the parent agent starts. These agents are registered in the `AgentRegistry` and available for orchestration states (`delegate`, `concurrent`, `group_chat`).

```yaml
spawner:
  shared_llms: true
  auto_spawn:
    - id: billing
      agent: agents/billing_agent.yaml
    - id: technical
      agent: agents/technical_agent.yaml
    - id: sales
      agent: agents/sales_agent.yaml
```

| Field | Type | Description |
|-------|------|-------------|
| `auto_spawn[].id` | `string` | Registry ID for this agent (referenced by `delegate`, `concurrent`, etc.) |
| `auto_spawn[].agent` | `string` | Path to the agent YAML file (resolved relative to the parent YAML directory) |

When `shared_llms: true`, auto-spawned agents use the parent's authoritative LLM registry without constructing child-local providers or reading child provider credentials. Every child-referenced alias must exist in that registry. When false, the child configures its own declared providers. File read, strict parsing, admission, provider, feature, storage-readiness, or registration failure aborts `auto_configure_spawner()`; the parent is not returned with a partial declared topology.

### Orchestration Tools

`orchestration_tools` registers and grants multi-agent coordination patterns as tools that the LLM can call at runtime. This enables dynamic orchestration where the LLM decides which agents to involve.

```yaml
spawner:
  shared_llms: true
  orchestration_tools: true    # register and grant all 5 orchestration tools
  # or selectively:
  # orchestration_tools: [route_to_agent, pipeline_process, concurrent_ask, group_discussion, handoff_conversation]
  templates:
    merchant:
      system_prompt: "You are a merchant."
```

| Tool | Description |
|------|-------------|
| `route_to_agent` | Route input to the best-matched agent from a set of candidates |
| `pipeline_process` | Chain agents sequentially with per-stage Jinja2 templates (`{{ user_input }}`, `{{ previous_output }}`, `{{ original_input }}`, `{{ stages.<agent_id> }}`) |
| `concurrent_ask` | Ask multiple agents the same question in parallel and aggregate |
| `group_discussion` | Run a multi-agent conversation on a topic |
| `handoff_conversation` | Start with one agent and allow dynamic handoffs to others |

Accepts `true` (all 5 tools) or a list of specific tool names.

---

## Evaluation Suites

Evaluation suites are separate YAML files used by `ai-agents-cli eval`. They are not agent specs; they describe scenarios, fixtures, assertions, and output behavior for testing an agent through the normal runtime path. For practical workflows such as record/replay cassettes and live LLM smoke tests, see [Evaluation](@/docs/evaluation.md).

```yaml
name: Basic Chat Eval
agent: ../../../yaml/basic/simple_chat.yaml
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
    name: Basic response is returned
    tags: [basic, smoke]
    language: en
    turns:
      - input: Hello
        assert:
          response_not_empty: true
          response_contains: "Hello"
```

### Suite fields

A suite is the outer test document. It is intentionally separate from the agent YAML so the same agent can be tested with several datasets, fixture modes, or CI policies.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `string` | required | Human-readable suite name used in `summary.md`, `summary.json`, and JUnit output. Must not be empty. |
| `agent` | `path?` | `null` | Agent YAML path resolved relative to the suite file. CLI `--agent` overrides this, which is useful when the same suite should run against several agent variants. |
| `settings` | `map` | defaults | Execution policy for timeouts, retries, isolation, concurrency, fail-fast, provider overrides, and output redaction. |
| `observability` | `map?` | `null` | Optional observability config attached only for eval runs. Use this when eval reports should include latency, token, cost, or purpose metrics. CLI `--observability` creates a safe default when this is omitted. |
| `fixtures` | `map` | empty | Replaces external dependencies during eval: mock/replay/record/real LLMs, mock tools, context files, and local HTTP routes. |
| `scenarios` | `list` | empty | Test cases. Active scenarios must define `turns` or `steps`; duplicate IDs are rejected. |

Execution order is: load suite -> validate suite -> filter scenarios -> create an isolated workspace -> build the agent -> apply fixtures and context -> run turns or steps -> evaluate assertions -> write reports.

### `settings`

```yaml
settings:
  timeout_per_turn_ms: 30000
  retries: 0
  retry_delay_ms: 1000
  isolation: scenario
  parallel: false
  max_concurrent: 4
  fail_fast: false
  redact_outputs: true
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `timeout_per_turn_ms` | `u64` | `30000` | Milliseconds allowed for one turn |
| `timeout_per_scenario_ms` | `u64?` | `null` | Milliseconds allowed for one scenario attempt |
| `retries` | `u32` | `0` | Retry failed scenarios before final status |
| `retry_delay_ms` | `u64` | `1000` | Delay between attempts |
| `isolation` | enum | `scenario` | `scenario` uses a fresh runtime per attempt; `turn` resets between turns; `suite` and `none` are rejected until shared-run isolation is hardened |
| `parallel` | `bool` | `false` | Run scenarios concurrently when `isolation: scenario` and no scenario env overlays are used |
| `max_concurrent` | `usize` | `4` | Maximum concurrently running scenarios |
| `fail_fast` | `bool` | `false` | Stop after the first failed or errored scenario; runs serially |
| `redact_outputs` | `bool` | `true` | Store `[redacted]` for inputs, responses, and string assertion details; raw evidence is omitted from JSON outputs |
| `temperature` | `f32?` | `null` | Override agent LLM temperatures during eval. This is useful for more deterministic live-provider smoke tests. |
| `seed` | `u64?` | `null` | Adds a provider-specific `seed` value to LLM extra config when providers support deterministic seeding. |

`isolation: scenario` is the recommended default. It creates a fresh runtime and temp workspace per scenario attempt. `isolation: turn` resets conversation state between direct turns, while still reapplying fixture/scenario context. `isolation: suite` and `isolation: none` are rejected until shared-run isolation has a stronger public contract.

Use `parallel: true` or CLI `--parallel <N>` for faster CI runs across independent scenarios. Parallel execution requires `isolation: scenario` and cannot be combined with process environment overlays because environment variables are process-global.

### `fixtures`

```yaml
fixtures:
  context:
    user:
      id: customer_42
      tier: vip
  context_file: ./fixtures/default_context.json
  tools:
    lookup_order:
      output:
        id: ORD-1042
        status: cancellable
  llm:
    mode: mock
    responses:
      - "Mocked response text"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `context` | object | `{}` | Runtime context applied after `context_file` |
| `context_file` | path? | `null` | JSON context file resolved relative to the suite file |
| `tools` | map | `{}` | Mock tool outputs keyed by tool ID |
| `llm.mode` | enum | `real` | `real`, `mock`, `replay`, or `record` |
| `llm.responses` | list | `[]` | Ordered mock responses for `mock` mode |
| `llm.responses_by_alias` | map | `{}` | Ordered responses for each configured LLM alias |
| `llm.outcomes_by_alias` | map | `{}` | Ordered response/error outcomes per alias; error entries accept optional HTTP `status` |
| `llm.cassette` | path? | `null` | JSONL cassette path for `replay` or `record` mode |
| `mock_server` | map | disabled | Start an attempt-local HTTP server and expose `mock_server.base_url` |
| `workspace_policy` | map? | `null` | Add the attempt workspace to named existing read/write tool policies |
| `web_fetch_transport` | map? | `null` | Exact-URL no-socket routes executed through the real web-fetch implementation |
| `web_search` | map? | `null` | Static exact-query search provider; `available: false` installs an unavailable provider, while unmatched queries on an available fixture return an empty available response |
| `diagnostics` | map? | `null` | Static diagnostics provider for deterministic `diagnostics` tool evals |
| `commands` | map? | `null` | Static command-runner responses for deterministic `command` tool evals |

LLM fixture modes:

| Mode | Behavior |
|------|----------|
| `mock` | Uses `llm.responses` in order. When responses are exhausted, the last response repeats. In streamed turns, mock responses are split into multiple chunks before the eval runner joins them again. Best for default CI. |
| `replay` | Loads cassette JSONL records and requires an exact alias, model, and request-hash match. A miss is an error and never falls back to configured mock responses. |
| `record` | Calls the real provider and appends cassette JSONL records. Records request hashes and responses, not separate raw prompt fields. |
| `real` | Calls the provider configured by the agent YAML. Use for live smoke tests only because it needs credentials and may incur cost. |

Mock tools are keyed by tool ID. A mock with the same ID as a built-in replaces that built-in for the eval run.

```yaml
fixtures:
  tools:
    lookup_order:
      success: true
      output:
        id: ORD-1042
        status: cancellable
```

`mock_server.routes` entries accept `method`, `path`, `status`, optional `headers`, and `body`. The runner chooses a dynamic port when `port` is omitted. Mock LLM responses may interpolate only `{{ mock_server.base_url }}` and `{{ eval.workspace }}`. Interpolation is JSON-safe, unrelated template expressions remain unchanged, and referencing the server token without an enabled server is an error.

Every attempt exposes `eval.workspace` in runtime context and isolates parent and spawner file, SQLite, or Redis storage. `workspace_policy.read_tools` and `workspace_policy.write_tools` may add that workspace only to named existing policies; unknown tool policies are rejected.

`web_fetch_transport.routes` use exact public-style URLs with `status`, optional `headers`, and `body`. They do not open sockets and do not bypass the real web-fetch URL, address, redirect, byte, cache, or policy checks.

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

### Scenario and turn fields

```yaml
scenarios:
  - id: two-turn-conversation
    name: Two deterministic mocked turns pass
    tags: [basic, smoke]
    language: en
    actor: customer_42
    context:
      user:
        id: customer_42
    turns:
      - input: Hello
        context:
          channel: cli
        assert:
          all:
            - response_not_empty: true
            - response_contains: "Hello"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `string` | required | Stable scenario ID for filtering and reports |
| `name` | `string?` | `null` | Human-readable name |
| `tags` | list | `[]` | Filter with CLI `--tags` |
| `language` | `string?` | `null` | Filter with CLI `--language` and group metrics |
| `actor` | `string?` | `null` | Sets `actor_id` before scenario turns |
| `context` | object | `null` | Scenario-level context overlay |
| `skip` | bool/string | `false` | Skip the scenario. A string value is treated as the skip reason and appears in reports. |
| `env` | map | `{}` | Serial-only process environment overlay for one scenario attempt. Values are restored after the attempt. Parallel suites that use `env` are rejected. |
| `turns` | list | `[]` | Direct conversation turns. Each turn uses the same runtime unless `isolation: turn` is configured. |
| `steps` | list | `[]` | Advanced sequence with run/reset/save/load/context/actor steps. Use this for persistence, actor switching, or multi-session checks. |
| `turn.input` | `string` | required | User input for a turn |
| `turn.actor` | `string?` | `null` | Actor override for a single turn |
| `turn.context` | object | `null` | Turn-level context overlay |
| `turn.stream` | `bool?` | `null` | Use `chat_stream()`, collect all content chunks into a final response string, then run assertions against that final text |
| `turn.expect_error` | string/list? | `null` | Runtime error substring or alternatives expected for this turn |
| `turn.assert` | object | `null` | Assertions evaluated after the turn |

Context overlay order is: `fixtures.context_file` -> `fixtures.context` -> generated `eval.workspace` and optional `mock_server.base_url` -> `scenario.context` -> `turn.context`. Later values replace earlier values at the same top-level key. Fixture interpolation always uses immutable generated attempt values, not overridable runtime context.

Advanced step example:

```yaml
steps:
  - !set_actor
    actor: customer_42
  - !run
    turns:
      - input: My preferred language is Korean.
  - !reset_agent
    profile: full_runtime
    preserve_storage: true
    preserve_host_context: true
    preserve_actor_id: true
  - !run
    turns:
      - input: What do you remember?
        assert:
          response_not_empty: true
```

`!reset_agent true` uses the default full-runtime reset behavior. Object form lets you choose a profile and preservation flags. `profile: conversation` calls the runtime conversation reset without rebuilding the agent; other profiles rebuild the eval runtime. `preserve_storage` keeps the attempt-local persistence path, `preserve_host_context` reapplies fixture and scenario context, `preserve_actor_id` reapplies and reloads the active actor, and `delete_persistence` removes temp persistence before rebuilding. Use full-runtime reset plus preserved storage and actor identity to prove cross-runtime fact persistence without ordinary conversation history.

### Common assertions

Assertions are an implicit `all` when several simple keys appear in one object. Use `all`, `any`, and `not` for explicit composition.

```yaml
assert:
  all:
    - response_not_empty: true
    - response_contains_any: ["Hello", "Hi"]
    - state_in: [greeting, helping]
```

| Assertion | Purpose |
|-----------|---------|
| `state`, `state_in`, `state_not`, `state_history_contains` | Check state-machine evidence |
| `response_contains`, `response_contains_any`, `response_not_contains`, `response_not_empty` | Literal response checks |
| `metadata_contains` | Top-level blocking-response metadata key/value checks. Streamed turns do not expose final response metadata. |
| `metadata_path`, `context_path` | Dot-path checks with `eq`, `neq`, `in`, `contains`, `exists`, `gte`, `lte`, `gt`, `lt`. `metadata_path` requires a blocking turn; `context_path` remains available for streamed turns. |
| `tool_called`, `tool_not_called` | Tool execution evidence, including count, success, source, args, and result paths. |
| `skill_triggered` | Skill metadata when available. Useful for skill router regression tests. |
| `disambiguation`, `no_disambiguation` | Ambiguity flow evidence when available. Checks statuses such as `triggered`, `clarified`, `best_guess`, or `skipped`. |
| `facts_include` | Actor fact evidence. Can check actor, category, and semantic support via judge. |
| `relationship` | Relationship existence, actor, perspective, interaction counts, event counts, and dimension comparisons. |
| `persona_secret_revealed` | Persona secret reveal state. Boolean form checks if any secret is revealed. String form checks revealed secret IDs when available. |
| `orchestration` | Orchestration metadata such as pattern, final agent, included agents, and stage count. |
| `observability` | Observability report limits and counts for LLM calls, tool calls, tokens, cost, purpose counts, status counts, and configured dimension counts. |
| `judge`, `response_semantic` | Optional LLM judge checks for semantic quality. Supports `llm`, `pass_threshold`, and weighted criteria. |

Path assertion fields:

| Field | Meaning |
|-------|---------|
| `path` | Dot path to read from metadata, context, tool args, tool result, or a generated metrics object. Empty path means the root object. |
| `eq` / `neq` | Exact JSON equality or inequality. |
| `in` | Passes when the actual JSON value is one of the provided JSON values. |
| `contains` | String substring check for strings, or element membership for arrays. |
| `exists` | `true` requires the path to exist; `false` requires it to be absent. |
| `gte`, `lte`, `gt`, `lt` | Numeric comparisons. Non-numeric actual values fail. |

Example metadata and context assertions:

```yaml
assert:
  metadata_contains:
    intent: greeting
  context_path:
    path: user.tier
    eq: vip
```

Example tool assertion:

```yaml
assert:
  tool_called:
    id: lookup_order
    count_gte: 1
    success: true
    source_in: [mock]
    args_executed:
      path: id
      eq: ORD-1042
    result_path:
      path: status
      eq: cancellable
```

Tool assertion object fields:

| Field | Description |
|-------|-------------|
| `id` | Tool ID or requested tool name to match. String form `tool_called: lookup_order` is shorthand for this. |
| `count` | Exact number of calls that satisfy every configured predicate on the same execution record. |
| `count_gte` | Minimum number of calls that satisfy every configured predicate on the same execution record. |
| `executed` | Filter to calls where the wrapped tool implementation did or did not run. Use `false` for denied, unavailable, approval-rejected, or timeout paths. |
| `success` | Filter to successful or failed calls before counting. |
| `source_in` | Allowed source labels such as `llm`, `skill`, `plan`, `state_action`, `on_enter`, `on_exit`, `post_transition`, `spawner`, `orchestration`, or `mock`. Plan-and-execute tool steps use `plan`. |
| `args` / `args_executed` | Path assertion against the arguments actually executed. |
| `args_original` | Path assertion against original arguments before any wrapper or normalization behavior. |
| `result_path` | Path assertion against parsed tool output. String outputs are treated as strings; JSON outputs are parsed when possible. |

All object-form predicates are evaluated against one execution record. Argument and result predicates cannot be satisfied by different calls. `tool_called` always requires at least one complete match; use `tool_not_called` to assert absence.

Plan-origin calls previously appeared under the broad `llm` source. They now use `plan`, so suites that explicitly filter plan-and-execute tool steps should migrate to `source_in: [plan]`.

Example facts, relationship, orchestration, and observability assertions:

```yaml
assert:
  all:
    - facts_include:
        actor: customer_42
        category: user_preference
    - relationship:
        actor: customer_42
        perspective: agent_to_actor
        dimension: trust
        gte: 0.5
    - orchestration:
        pattern: pipeline
        agents_include: [writer, editor]
        stages: 2
    - observability:
        total_llm_calls_lte: 4
        total_tool_calls_lte: 1
        purpose_counts:
          main_response:
            path: count
            gte: 1
        dimension_counts:
          - match_dimensions:
              background: "true"
            assert:
              path: count
              gte: 1
```

Assertion-specific fields:

| Assertion | Fields |
|-----------|--------|
| `facts_include` | `actor`, `category`, `semantic`. `semantic` uses a judge LLM to check whether the selected fact set supports the claim. |
| `relationship` | `actor`, `exists`, `perspective`, `dimension`, `gte`, `lte`, `gt`, `lt`, `eq`, `interaction_count_gte`, `notable_event_count_gte`. |
| `orchestration` | `pattern`, `type`, `final_agent_in`, `agents_include`, `stages`. Agent matching checks common metadata shapes such as agents arrays, stage agent IDs, participants, and handoff events. |
| `observability` | `total_llm_calls_lte`, `total_tool_calls_lte`, `total_tokens_lte`, `total_cost_usd_lte`, `purpose_counts`, `status_counts`, `dimension_counts`. Count assertions use path assertion syntax over `{ count: N }`. |

Example judge assertion:

```yaml
assert:
  judge:
    llm: router
    pass_threshold: 0.8
    criteria:
      - name: relevance
        description: Response directly addresses the user's request.
        weight: 1.0
      - name: safety
        description: Response avoids irreversible action without confirmation.
        weight: 0.5
```

Judge fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm` | `string?` | router/default | Optional LLM alias from the agent's `llms` map. Use a cheap router model for judge checks when possible. |
| `pass_threshold` | `f32` | `0.75` | Minimum overall judge score required to pass. |
| `criteria` | list | required unless defaults are configured in Rust | Text criteria or objects with `name`, `description`, and `weight`. The judge must return strict JSON. |

`response_semantic` uses the same object shape as `judge`; it is just a more descriptive alias for response-quality checks.

### Output files

`ai-agents-cli eval` writes these files to the configured output directory:

| File | Purpose |
|------|---------|
| `summary.md` | Human-readable pass/fail summary |
| `summary.json` | Machine-readable result with `schema_version: 1` |
| `per_scenario.jsonl` | One scenario result per line |
| `failures.md` | Failure-focused report |
| `junit.xml` | Optional CI report when `--junit` is used |

Privacy note: `redact_outputs: true` stores `[redacted]` for input, response, and string assertion details. Default JSON and JSONL outputs omit raw `TurnEvidence` and response metadata. Set `redact_outputs: false` only for trusted local debugging.

## Complete Minimal Example

The smallest valid agent YAML:

```yaml
name: MinimalAgent
system_prompt: "You are a helpful assistant."
llm:
  provider: openai
  model: gpt-5.4-nano
```

## Complete Full Example

An agent using most features together:

```yaml
name: FullFeaturedAgent
version: "1.0.0"
description: "Shows all major features in one file"

system_prompt: |
  You are a support assistant.
  Customer: {{ context.user.name }} ({{ context.user.tier }})
  Today: {{ context.time.date }}

llms:
  default:
    provider: openai
    model: gpt-5.4-mini
  router:
    provider: openai
    model: gpt-5.4-nano

llm:
  default: default
  router: router

context:
  user:
    type: runtime
    default:
      name: "Guest"
      tier: "free"
  time:
    type: builtin
    source: datetime
    refresh: per_turn

memory:
  type: compacting
  max_recent_messages: 8
  compress_threshold: 8
  summarize_batch_size: 4
  summarizer_llm: router
  token_budget:
    total: 4096
    allocation:
      summary: 1024
      recent_messages: 2048
      facts: 512
    overflow_strategy: truncate_oldest
    warn_at_percent: 70

storage:
  type: sqlite
  path: "./sessions.db"

tools:
  - calculator
  - datetime
  - http

tool_security:
  enabled: true
  fail_closed: true
  tools:
    calculator: {}
    datetime: {}
    http:
      domains:
        allow: [api.example.com]
      operations:
        allow: [GET, HEAD]
        requires_approval: [POST, PUT, PATCH, DELETE]
      timeout_ms: 10000

parallel_tools:
  enabled: true
  max_parallel: 3

streaming:
  enabled: false

states:
  initial: greeting
  fallback: confused
  max_no_transition: 3
  global_transitions:
    - to: escalation
      when: "User is very frustrated or asks for a manager"
      priority: 100
  states:
    greeting:
      prompt: "Welcome the customer. Ask how you can help."
      tools: []
      transitions:
        - to: helping
          when: "User describes an issue"
    helping:
      prompt: "Help the user with their issue."
      transitions:
        - to: closing
          when: "Issue is resolved"
    closing:
      prompt: "Thank the user. Ask if there's anything else."
      transitions:
        - to: greeting
          when: "User has another question"
    confused:
      prompt: "Ask the user to clarify their request."
      transitions:
        - to: helping
          when: "User clarifies their issue"
    escalation:
      prompt: "Acknowledge frustration. Summarize the issue for escalation."

skills:
  - id: quick_math
    description: "Quick calculation"
    trigger: "When user asks for a calculation"
    steps:
      - prompt: "Extract the math expression from: {{ user_input }}"
      - tool: calculator
        args:
          expression: "{{ steps[0].result }}"
      - prompt: "Result: {{ steps[1].result }}. Explain it clearly."

process:
  input:
    - type: normalize
      config:
        trim: true
        collapse_whitespace: true

hitl:
  default_timeout_seconds: 120
  on_timeout: reject
  tools:
    http:
      require_approval: true
      approval_message: "Approve {{ method }} request to {{ url }}?"

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
    - "Response addresses the user's question"
    - "Response is accurate and helpful"

disambiguation:
  enabled: true
  detection:
    llm: router
    threshold: 0.7
    aspects: [missing_target, missing_action, vague_references]
  clarification:
    style: auto
    max_attempts: 2
    on_max_attempts: proceed_with_best_guess
  skip_when:
    - type: social

error_recovery:
  default:
    max_retries: 3
    backoff:
      type: exponential
      initial_ms: 500
      max_ms: 5000
      multiplier: 2.0
  llm:
    on_failure:
      action: fallback_response
      message: "I'm temporarily unavailable. Please try again."
    on_context_overflow:
      action: summarize
      summarizer_llm: router
      keep_recent: 4
  tools:
    default:
      max_retries: 2
      timeout_ms: 10000

max_iterations: 15
max_context_tokens: 128000

metadata:
  cli:
    welcome: "=== Full-Featured Agent ==="
    hints:
      - "Try: Hello!"
      - "Try: What is 42 * 17?"
      - "Try: I'm furious, get me a manager!"
    show_state: true
    show_tools: true
```

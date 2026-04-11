+++
title = "YAML Reference"
weight = 2
template = "docs.html"
description = "Complete reference for agent YAML specification fields."
+++

<!--# YAML Reference-->

This is the complete reference for every field you can use in an agent YAML file. Each section covers one top-level key, shows its type, default, and a working snippet.

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
| **Default** | `4096` |

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
| *(any other key)* | | | Passed through as provider-specific extra parameter |

```yaml
llm:
  provider: openai
  model: gpt-4.1-mini
  temperature: 0.5
  max_tokens: 4000
```

Any field not listed above is captured in an `extra` map via `#[serde(flatten)]` and forwarded to the LLM client when a matching builder method exists. This includes transport-level resilience (`resilient`, `resilient_attempts`, etc.), Azure settings (`api_version`, `deployment_id`), and provider-specific search (`openai_enable_web_search`, `xai_search_mode`, etc.). See [LLM Providers > Extra Parameters](@/docs/providers.md#extra-parameters) for the full list.

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

### Named LLMs - `llms`

Define multiple named LLM configurations when you need different models for different roles (routing, summarization, evaluation).

```yaml
llms:
  default:
    provider: openai
    model: gpt-4.1-mini
  router:
    provider: openai
    model: gpt-4.1-nano
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
llm:
  provider: openai-compatible
  model: qwen3:8b
  base_url: http://localhost:11434/v1

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
| `regenerate_on_transition` | `bool` | `true` | Re-generate response in the new state after transitioning |
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
| `tools` | `list` | *inherit* | Tool IDs available in this state. `[]` = no tools. Omit = inherit agent-level tools |
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
| `auto` | `bool` | `false` | Transition without re-evaluating (used when `when` is structural) |
| `priority` | `u32` | `0` | Higher priority transitions are evaluated first |
| `cooldown_turns` | `u32` | `null` | Minimum turns before this transition can fire again |

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

The `tools` list declares which tools the agent can use. The framework auto-injects tool names, descriptions, and argument schemas into the prompt - do **not** list them in `system_prompt`.

### Simple String Form

Reference a built-in tool by name:

```yaml
tools:
  - calculator
  - datetime
  - http
```

### Structured Form

```yaml
tools:
  - name: calculator
  - name: datetime
  - name: text
  - name: json
  - name: math
  - name: random
  - name: file
  - name: template
  - name: http
  - name: echo
```

### MCP Tool

Declare an MCP server as a tool entry. The framework connects at startup, discovers available functions, and exposes them.

```yaml
tools:
  - name: filesystem
    type: mcp
    transport: stdio                # stdio | sse
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
```

Views create named subsets of a server's functions. States reference views by name for scoped tool access. The parent tool name (e.g. `filesystem`) always includes **all** functions.

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

Tool availability per state uses 3-level filtering:

- `tools: []` - explicitly no tools (even if agent declares them)
- `tools: [datetime]` - only those tools
- *omit `tools`* - inherit from agent-level `tools` list

### `tool_aliases`

Multi-language names and descriptions for tools. Lets the same tool appear with localized names to the LLM.

| Detail | Value |
|--------|-------|
| **Type** | `map` |
| **Default** | `{}` |

```yaml
tool_aliases:
  calculator:
    ko:
      name: "계산기"
      description: "수학 계산을 수행합니다"
    ja:
      name: "電卓"
      description: "数学の計算を実行します"
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
| `max_recent_messages` | `usize` | `50` | Recent messages always kept in full |
| `compress_threshold` | `usize` | `30` | Compression kicks in after this many messages |
| `summarize_batch_size` | `usize` | `10` | How many old messages to summarize at once |
| `summarizer_llm` | `string` | `null` | LLM alias for summarization (use a fast/cheap one) |

```yaml
memory:
  type: compacting
  max_recent_messages: 6
  compress_threshold: 8
  summarize_batch_size: 4
  summarizer_llm: router
```

### `token_budget`

Fine-grained control over how much memory contributes to the prompt. Used with `compacting` memory.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `total` | `u32` | - | Max tokens memory can contribute |
| `allocation.summary` | `u32` | - | Tokens reserved for rolling summary |
| `allocation.recent_messages` | `u32` | - | Tokens reserved for recent messages |
| `allocation.facts` | `u32` | - | Tokens reserved for extracted key facts |
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
    overflow_strategy: truncate_oldest
    warn_at_percent: 70
```

---

## Storage

Persist sessions across restarts. Without storage, conversation history is lost when the process exits.

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
| `ttl_seconds` | `u64` | `null` | Time-to-live for session keys |

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

The `tool_security` block enforces safety constraints on tool execution.

| Detail | Value |
|--------|-------|
| **Type** | `object` |
| **Default** | `{}` (disabled) |

```yaml
tool_security:
  enabled: true
  default_timeout_ms: 5000
  tools:
    http:
      rate_limit: 10
      timeout_ms: 10000
      allowed_domains:
        - "api.example.com"
        - "httpbin.org"
      blocked_domains:
        - "internal.corp.net"
      require_confirmation: true
```

---

## HITL (Human-in-the-Loop)

Pause tool execution and ask a human for approval before proceeding.

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
| `output` | `string` | `"hidden"` | `hidden`, `visible`, `tagged` |
| `max_iterations` | `u32` | `5` | Max reasoning iterations |

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
| `cot` | Chain-of-thought: think step by step before answering |
| `react` | ReAct: interleave reasoning and tool use |
| `plan_and_execute` | Create a plan first, then execute each step |
| `auto` | Framework picks the best mode based on the query |

### `reasoning.planning`

Extra settings for `plan_and_execute` mode.

```yaml
reasoning:
  mode: plan_and_execute
  planning:
    planner_llm: router
    max_steps: 10
    available:
      tools: [calculator, datetime]
      skills: [math_helper]
    reflection: true
```

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
| `pass_threshold` | `f64` | `0.7` | Score threshold (0.0–1.0) to pass |
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

Detect ambiguous user messages and ask clarifying questions before proceeding.

> **Note:** Disambiguation relies on the router LLM to detect ambiguity, classify the type, and generate clarification questions. Very small models (e.g. gpt-4.1-nano) may misclassify ambiguity types or ignore style instructions. Use at least a mid-tier model or latest model (e.g. gpt-4.1-mini or gpt-5.4-nano) for the router if disambiguation quality matters.

The threshold is the sole decision for ambiguity. The detector LLM returns a confidence score (0.0-1.0) for how clear the user's intent is. Messages scoring below the threshold trigger clarification.

### `disambiguation.enabled`

| Detail | Value |
|--------|-------|
| **Type** | `bool` |
| **Default** | `false` |

### `disambiguation.detection`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm` | `string` | `"router"` | LLM alias for detection |
| `threshold` | `f64` | `0.7` | Confidence cutoff (0.0-1.0). Messages below this trigger clarification. Lower = more sensitive |
| `aspects` | `list` | `[missing_target, missing_action, missing_parameters, vague_references]` | Which ambiguity types to check for |
| `prompt` | `string` | _(none)_ | Optional custom detection prompt. Replaces the built-in prompt sent to the detection LLM. Omit to use the default |

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

A state can override agent-level disambiguation settings. Useful for sensitive states (e.g. payment) that need higher clarity.

| Field | Type | Description |
|-------|------|-------------|
| `threshold` | `f64` | Override the agent-level confidence cutoff |
| `require_confirmation` | `bool` | Ask "Did you mean X?" even after clarification resolves |
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

`required_clarity` is a hard gate: if the detector reports any of these fields in `what_is_unclear`, clarification is forced even if confidence is above the threshold.

### Skill-Level Disambiguation Override

A skill can declare its own disambiguation settings. After the skill router identifies a matching skill, the runtime runs a second disambiguation pass with the skill's override before executing the skill steps.

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

## Streaming & Parallel Tools

### `streaming`

Stream LLM tokens to the user in real time instead of waiting for the full response.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable streaming mode |
| `buffer_size` | `usize` | `32` | Token buffer size |
| `include_tool_events` | `bool` | `true` | Stream tool call events |
| `include_state_events` | `bool` | `true` | Stream state transition events |

```yaml
streaming:
  enabled: true
```

> **Note:** Output process pipeline stages (sanitize, format) only run in blocking mode. They are skipped when streaming is enabled.

Also set `metadata.cli.streaming: true` to tell the CLI to use streaming mode.

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
    transport: stdio         # stdio | sse
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
| `transport` | `string` | `stdio` (spawn process) or `sse` (HTTP server) |
| `command` | `string` | Command to spawn (for `stdio`) |
| `args` | `list` | Command arguments |
| `env` | `map` | Environment variables for the spawned process |
| `startup_timeout_ms` | `u64` | Max time to wait for server startup |
| `security.blocked_functions` | `list` | Functions to block from discovery |
| `views` | `map` | Named subsets of discovered functions |

---

## Spawner (Dynamic Agent Spawning)

The `spawner:` section lets a parent agent create and manage child agents at runtime. Child agents can be spawned from inline YAML, `AgentSpec` objects, or named Jinja2 templates. A central registry tracks spawned agents and provides inter-agent messaging.

Four built-in tools are automatically registered when `spawner:` is present: `generate_agent`, `send_message`, `list_agents`, and `remove_agent`.

### Basic Configuration

```yaml
spawner:
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
        model: gpt-4.1-nano
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
| `shared_llms` | `bool` | `false` | Reuse parent's LLM connections for spawned agents |
| `max_agents` | `u32` | - | Hard limit on total spawned agents |
| `name_prefix` | `string` | - | Auto-name agents (e.g. `npc_001`, `npc_002`) |
| `shared_context` | `map` | - | Key-value pairs injected into every spawned agent's template as `context.*` |
| `allowed_tools` | `list` | - | Restrict which tools spawned agents may declare. Unlisted tools are stripped. |
| `templates` | `map` | - | Named YAML templates. Values are either inline strings or `{ path: "..." }` objects. |

### Spawner Tools

| Tool | Description |
|------|-------------|
| `generate_agent` | Spawn an agent from a description (via LLM-generated YAML or named template) |
| `send_message` | Send a message to another registered agent and get its response |
| `list_agents` | List all registered agents as JSON |
| `remove_agent` | Remove an agent from the registry by ID |

### Template Variables

In templates, caller-provided variables are top-level (`{{ name }}`, `{{ role }}`). Shared context values use the `context.` prefix (`{{ context.world_name }}`). Templates are rendered with Jinja2 (minijinja).

### Security

The `allowed_tools` list prevents spawned agents from accessing sensitive tools. `generate_agent` is never injected into spawned agents by default, preventing recursive spawning. LLM-generated YAML that references tools outside the allowlist is stripped before the agent is built.

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

When `shared_llms: true`, auto-spawned agents inherit the parent's LLM connections. Each agent is built through the standard `AgentBuilder` pipeline with `auto_configure_llms()` and `auto_configure_features()`.

### Orchestration Tools

`orchestration_tools` registers multi-agent coordination patterns as tools that the LLM can call at runtime. This enables dynamic orchestration where the LLM decides which agents to involve.

```yaml
spawner:
  shared_llms: true
  orchestration_tools: true    # register all 5 orchestration tools
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

## Complete Minimal Example

The smallest valid agent YAML:

```yaml
name: MinimalAgent
system_prompt: "You are a helpful assistant."
llm:
  provider: openai
  model: gpt-4.1-nano
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
    model: gpt-4.1-mini
  router:
    provider: openai
    model: gpt-4.1-nano

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

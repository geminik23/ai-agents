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

---

## Skills

Skills are reusable multi-step pipelines triggered by natural language. The router LLM picks the right skill based on `trigger` matching.

### Inline Skill

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Unique skill identifier |
| `description` | `string` | What this skill does (shown to router) |
| `trigger` | `string` | Natural-language trigger condition |
| `steps` | `list` | Ordered list of `prompt` or `tool` steps |

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

### `disambiguation.enabled`

| Detail | Value |
|--------|-------|
| **Type** | `bool` |
| **Default** | `false` |

### `disambiguation.detection`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm` | `string` | `null` | LLM alias for detection |
| `threshold` | `f64` | `0.7` | Ambiguity score threshold to trigger clarification |
| `aspects` | `list` | `[]` | What to check for |

Available aspects:
- `missing_target` - unclear what the user is referring to
- `missing_action` - unclear what action to take
- `missing_parameters` - key details are missing
- `multiple_intents` - message contains multiple requests
- `vague_references` - pronouns or references without context
- `implicit_context` - assumes shared knowledge

### `disambiguation.clarification`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `style` | `string` | `"auto"` | How to ask: `auto`, `options`, `open_ended` |
| `max_attempts` | `u32` | `2` | Max clarification rounds |
| `on_max_attempts` | `string` | `"proceed_with_best_guess"` | Action when limit is reached |

### `disambiguation.skip_when`

Skip disambiguation for certain message types.

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
    max_attempts: 2
    on_max_attempts: proceed_with_best_guess
  skip_when:
    - type: social
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

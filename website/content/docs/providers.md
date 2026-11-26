+++
title = "LLM Providers"
weight = 6
template = "docs.html"
description = "Setup guide for all supported LLM providers."
+++

The development checkout offers 14 provider presets - from cloud APIs like OpenAI and Anthropic to local servers like Ollama and any OpenAI-compatible endpoint. This page covers setup for each one with copy-paste-ready YAML snippets.

---

## Overview

| Provider | YAML `provider:` value | Environment Variable | Example Models |
| --- | --- | --- | --- |
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-5.4-nano`, `gpt-5.4-mini`, `gpt-5.4` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6`, `claude-haiku-4-5-20251001` |
| Messages-compatible | `anthropic-compatible` | Optional explicit `api_key_env` | Server-supplied model IDs |
| Google Gemini (first-party adapter) | `google` | `GOOGLE_API_KEY` | `gemini-2.5-flash`, `gemini-2.5-pro` |
| Ollama | `ollama` | *(none)* | `llama3.1`, `qwen3:8b`, `mistral` |
| Ollama Cloud | `ollama-cloud` | `OLLAMA_API_KEY` | Explicit Cloud model ID |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-chat`, `deepseek-reasoner` |
| Groq | `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| Mistral | `mistral` | `MISTRAL_API_KEY` | `mistral-large-2512`, `mistral-small-2503` |
| Cohere | `cohere` | `COHERE_API_KEY` | `command-r-plus-08-2024`, `command-r-08-2024` |
| xAI (Grok) | `xai` | `XAI_API_KEY` | `grok-4.20`, `grok-3-mini` |
| Phind | `phind` | `PHIND_API_KEY` | *(legacy - verify endpoint availability)* |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | `openai/gpt-5.4-mini`, `anthropic/claude-sonnet-4.6` |
| OpenAI-compatible | `openai-compatible` | *(set via `api_key_env`)* | *(depends on server)* |

---

The default upstream Anthropic, DeepSeek, xAI and Phind clients use fixed service endpoints for chat, even if the configuration supplies a `base_url`. For a custom Anthropic Messages server, select `anthropic-compatible` and supply its API base explicitly.

## First-party Chat Completions (development)

The development checkout sends ordinary `openai` and `openai-compatible` requests to the Chat Completions endpoint through `ai-agents-llm`. The base URL includes the API prefix. A compatible server can accept unauthenticated requests when its optional key is absent. An explicitly selected key variable for the official OpenAI preset must be available. Select credentials explicitly for an OpenAI preset with a custom origin. The direct client disables redirects.

```yaml
llm:
  provider: openai-compatible
  base_url: http://localhost:8000/v1
  model: your-model
  protocol: chat-completions
  reasoning_effort: xhigh
  provider_options:
    logprobs: true
```

`protocol` is optional. The first-party codec sends an ordinary request's nonempty effort string verbatim and omits it when absent. Choose a value and extensions supported by your server. The adapter combines explicit `provider_options` and legacy `extra_body` on distinct JSON paths. Conflicting values and overrides of messages, tools, model, stream controls, output ceilings, credentials or other owned fields fail before sending a request. Hard-priced calls require the audited option set. An existing `extra_body.response_format` with `text`, `json_object`, or `json_schema` maps to the owned Chat response format. Direct SDK callers can set `ChatRequest::response_format`.

For `provider: openai`, an omitted temperature stays absent from the HTTP request. Existing legacy presets and `openai-compatible` resolve an omitted temperature to their historical `0.7` at provider construction. First-party Messages preserves omission. Both shorthand and named YAML configurations preserve omission on save. The output token default is `2000`.

The direct client on the official OpenAI origin and the framework OpenAI preset apply the same documented Chat restrictions to explicit sampling options and native tools. For example, GPT-6.2 Astra/Sol and GPT-6 Astra/6.1 Sol require Responses for native tools and reject explicit sampling. GPT-6 Sol/Luna allow Chat native tools and sampling with effort `none`. GPT-5.2/5.4 allow sampling with effort `none`, their default. Unknown model IDs pass through to the endpoint. A compatible server owns its own model rules. [GPT-6.2 guide](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6.2-astra) · [GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra) · [GPT-5.4 guide](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.4)

Legacy transport settings such as resilience, deployment/API version and hosted web search select the upstream backend before dispatch. The same applies to legacy reasoning/budget and unsupported typed sampling overrides. Enabled legacy transport settings conflict with explicit SDK options. Disabled resilience and web search, including their inactive tuning fields, keep the SDK path. The public legacy `build_llm()` continues to use the upstream backend. The OpenRouter preset has its own Chat contract.

The SDK reads text as it arrives. A successful stream requires a finish reason and a complete `[DONE]` event. A disconnect after visible text reports an error. The framework also reports explicit non-success text finishes such as `length` and `content_filter` as terminal errors. Previously published deltas remain provisional and cannot trigger tools or a successful final response. Native tool calls require distinct, nonempty IDs. Arguments are assembled by call index, parsed as JSON objects and checked against the requested function selection before delivery. Direct SDK consumers validate their application schema before acting on a call. Framework tool approval and execution use the existing executor. Incomplete calls never become executable. The direct client exposes the provider finish reason and complete token receipt. The framework continues to publish its own processed agent response.

Rust applications can also send ordered text, image URL or image bytes, and WAV/MP3 audio bytes through the [direct Chat SDK](@/docs/rust-api.md#direct-chat-completions-sdk-development). Media support depends on the chosen server/model. The framework's YAML/CLI message input remains text-based.

## OpenAI Responses (development)

Select the Responses endpoint explicitly for text or native functions on the OpenAI preset. The base URL includes the API prefix. Existing Chat configurations retain their selected endpoint.

```yaml
llm:
  provider: openai
  protocol: responses
  model: gpt-6.2-sol
  reasoning_effort: low
  tool_choice: auto
```

The adapter sends ordered input items with `store: false`. An explicit YAML `max_tokens` sets `max_output_tokens`. Omission leaves the output ceiling to the endpoint. Token-limited or cost-limited evaluations require a declared output ceiling. An explicit `provider_options.reasoning.mode` or `effort` is passed through. Specify effort once. The adapter rejects unsupported options, including server conversation handles, legacy transport settings and overrides of tools, storage or output ceilings. GPT-6.2 Astra and Sol accept native functions here. Omitted sampling options stay absent. The framework uses its existing tool grants, approval and executor.

The direct SDK returns ordered output items, including private encrypted reasoning when supplied. A trusted caller can append these items and its `function_call_output` with the original `call_id` to the next input. Managed tool turns retain the original items and call bindings across supported task continuation. Replay requires the same provider, API, endpoint and model. Ordinary tool-free framework turns continue from readable messages. Applications requiring encrypted reasoning continuity across tool-free turns must manage those items through the direct SDK.

Text deltas are provisional. Function calls become available only on verified completion. Incomplete, failed or disconnected streams surface errors. Shared task accounting settles only complete matching final usage after consumption. Reasoning tokens are included in total output tokens. A declared pricing contract for Responses must name the actual endpoint ending in `/responses`. Reasoning-mode overrides are outside the audited hard-priced class. [Direct Rust API](@/docs/rust-api.md#direct-responses-sdk-development)

## SDK model catalogs (development)

The [direct Rust SDK](@/docs/rust-api.md#direct-model-catalog-sdk-development) lists models from the official OpenAI, Anthropic, and Ollama Cloud endpoints. Applications can use the reported IDs and metadata in a configuration interface or an inventory tool. YAML uses the explicitly configured model and runs independently of catalog availability.

## OpenAI Decisions (development)

Use a choice-only preset for the orchestration routing role alongside a separate generation preset. This excerpt assumes `primary` is declared under `llms` for the main agent.

```yaml
llms:
  dispatch:
    provider: openai
    protocol: decisions
    model: gpt-6-luna
llm:
  default: primary
  router:
    orchestration:
      routing: dispatch
```

Grant `route_to_agent` through the normal orchestration and tool configuration. The tool passes its input and candidate IDs to the selected provider. With a single registered candidate it can select directly. With multiple candidates it requires an exact ID from a complete answer. A refusal or invalid answer produces no child dispatch. The runtime retains the actual child binding across supported task resume and restoration. A tool call may omit `method` or use `llm` in this mode.

With `base_url` omitted, the OpenAI preset sends a single text classification question to `https://api.openai.com/v1/decisions`. It accepts a configured model, timeout, credential and API base URL. A custom origin requires an explicit credential. Generation settings, native tool requests, streaming and declared-priced task calls are unsupported for this preset. A scenario can declare an unused Decisions alias without its API key. If routing needs a model decision, the eval returns an unsupported-operation error before sending a request or using a text fixture. Use the [direct Rust SDK](@/docs/rust-api.md#direct-decisions-sdk-development) to classify text independently of an agent.

## Declared-priced task capabilities (development only)

Ordinary provider support is not hard-priced task support. The development standalone Rust task runner can use an immutable host-audited `DeclaredPricing` contract installed before provider handles are captured. The same object can be declared as `llm.declared_pricing` or on a named `llms` entry; it is optional and is not forwarded as provider JSON. It does not activate YAML task entry or change the published 1.1.2 task contract.

| Adapter | Priced request classes with an audited contract | Explicit limits |
|---|---|---|
| OpenAI Chat Completions | Completion, native tools, stream | Native choice follows the Chat model and effort contract |
| OpenAI Responses | Completion, native functions, text stream | Audit the full Responses endpoint separately. Complete terminal usage is required for final settlement. |
| OpenAI-compatible | Completion, stream; native tools with explicit `function_calling: true` | The host must audit the actual server, model and reporting semantics; compatibility is not a price proof |
| Google GenerateContent | Completion, native auto tool choice, stream | Signed native history and the normal first-party request builder are retained |
| OpenRouter | Completion, native tools, stream | Routing and model-tier prices require a complete host contract; unknown native service options are rejected |
| Groq, Cohere, Mistral | Completion, stream | Native tool requests retain their existing unsupported status |
| Anthropic, Messages-compatible, DeepSeek, xAI, Phind, Ollama | Unsupported for the built-in hard-priced capability | The current upstream-specific dispatch/accounting contracts are not audited by this adapter; ordinary use is unchanged |
| Custom/local Rust providers | Explicit host implementation of request bounds and settlement | Neither local execution nor a provider label implies free or conforming execution |

`DeclaredPricing` binds `schedule_id`, exact `model`, normalized HTTP(S) `endpoint`, `server_input_token_limit`, `server_output_token_limit`, integer `input_micro_usd_per_million` and `output_micro_usd_per_million`. Optional fields are `request_fee_micro_usd` (default `0`), `max_internal_attempts` (default `1`, including the initial upstream call) and `complete_usage` (default `false`). Token ceilings are audited server-enforced maxima, not desired client limits or token estimates. Rates must cover the highest applicable input/output tiers, cached input and reasoning classes. A false host declaration is not made true by parsing it; hosts are responsible for verifying that external contract.

Model-generated child YAML cannot install an audited pricing contract. Inside a hard-priced task, a spawner without a host-installed authoritative shared provider registry rejects fresh child construction before provider/credential setup. Shared mode ignores child provider and pricing declarations and uses the actual host-installed handles; hosts can also register already prepared child runtimes explicitly. A generated model name, zero-rate table or copied identity cannot mint a new priced capability.

The full input/output ceilings plus request fee are reserved for every possible internal attempt, with checked integer arithmetic and upward rounding. This intentionally trades utilization for a conservative bound without introducing an unproven generic tokenizer. Effective per-call output and reasoning settings must stay inside the contract. Unknown extras, arbitrary body overrides and native search/service fees without a complete adapter bound are rejected under hard pricing; no fallback may bypass that denial. Enabled internal resilience needs an explicit total-attempt count no larger than the installed contract. Multi-attempt results retain the full reservation because upstream may report only the final attempt.

Only audited complete provider usage may release unused reservation. Google additionally requires an explicitly reported total rather than relying on a display total synthesized by normalization. Missing/partial usage retains the complete upper bound. Google streams can settle a complete explicitly reported terminal usage receipt only after normal EOF; early cumulative or later partial reports are not final billing evidence. First-party Chat Completions streams can also settle after a verified finish, a complete matching token receipt and the end marker. Streams through legacy adapters retain the full bound when the upstream stream supplies no billing receipt. Errors, cancellation after possible dispatch, drop and lost acknowledgements never imply a refund or cleanup. The guarantee covers declared prices for managed conforming model requests, not actual invoices, unrelated tool-service fees, arbitrary host I/O or exactly-once effects.

See [Rust task APIs](@/docs/rust-api.md#guarded-standalone-execution-development-only) and [YAML pricing fields](@/docs/yaml-reference.md#declared-pricing-development-only). Offline fixture acceptance is separate from live provider/billing validation; no live paid run is needed to use the tests.

## OpenAI

Set the API key and pick a model:

```sh
export OPENAI_API_KEY=sk-...
```

```yaml
llm:
  provider: openai
  model: gpt-5.4-nano
```

Other model options:

```yaml
# Each YAML document below is a separate alternative.
# Good balance of speed and quality
llm:
  provider: openai
  model: gpt-5.4-mini

---
# Most capable
llm:
  provider: openai
  model: gpt-5.4

---
# Reasoning-capable model with effort control
llms:
  default:
    provider: openai
    model: gpt-5.4-mini
    reasoning_effort: low    # Values depend on the endpoint and model
llm:
  default: default
```

The `reasoning_effort` parameter controls how much reasoning the model applies before answering. See [Reasoning Parameters](#reasoning-parameters) below.

---

## Anthropic

```sh
export ANTHROPIC_API_KEY=sk-ant-...
```

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-6
```

Other models:

```yaml
# Fast and affordable
llm:
  provider: anthropic
  model: claude-haiku-4-5-20251001
```


### First-party Messages (development)

Select the native Messages endpoint explicitly. The official preset uses the Anthropic API key and version headers. The SDK sends the configured model ID as supplied.

```yaml
llm:
  provider: anthropic
  protocol: messages
  model: claude-sonnet-4-6
  max_tokens: 2048
  tool_choice: auto
  provider_options:
    thinking: {type: adaptive}
    output_config: {effort: low}
```

Messages requires an explicit positive `max_tokens` for generation. An omitted temperature stays absent. For API-supported manual thinking, use `thinking: {type: enabled, budget_tokens: 1024}` with a budget below `max_tokens` on the official endpoint. Declare thinking and effort once. Forced selection is checked against the selected official model and thinking mode. An unsupported configured `required` or `specific` choice returns a configuration error. Omitting `protocol` on the official preset continues to select the legacy adapter and its existing defaults.

Use `provider: anthropic-compatible` with an explicit `base_url` that includes the API prefix, a server-specific model ID and `max_tokens`. A compatible endpoint can run without credentials or use a server-specific `api_key_env`. With a key, the compatible preset uses Bearer authentication by default. Set `messages_auth: api-key` for servers expecting `x-api-key`. Set `anthropic_version` when a server requires a different API version. Native choices on a compatible endpoint require `function_calling: true`.

The framework displays text and manages complete native tool calls through its existing tool grants, approval and execution path. Explicit tool choice uses the buffered framework path. The direct SDK can stream a native function response and returns complete calls on verified termination. It retains ordered private thinking blocks for the active tool exchange and replays them only to the matching provider, endpoint, API version and model. Direct SDK consumers retain tool-free thinking themselves. Stream text stays provisional until `message_stop` and successful completion. The SDK accounts for cumulative output usage and records cache input separately. Hard-priced task admission rejects Messages before dispatch. [Direct Rust API](@/docs/rust-api.md#direct-messages-sdk-development)

The official endpoint was tested with Claude Sonnet 4.6 for signed thinking replay, native functions and streaming. The compatible codec was tested against a local HTTP fixture. vLLM has no verified server/version/model combination in this checkout.

---

## Google Gemini

```sh
export GOOGLE_API_KEY=AI...
```

```yaml
llm:
  provider: google
  model: gemini-2.5-flash
```

Other models:

```yaml
llm:
  provider: google
  model: gemini-2.5-pro
```

The normal `provider: google` path uses the framework's first-party Google Developer API adapter. It preserves the original model `Content.parts`, function-call IDs, and Gemini `thoughtSignature` values across native tool continuations. Native selection remains opt-in with `tool_choice: auto`; omitted choice uses the framework's prompt-JSON protocol, and `required` or `specific` uses the existing bounded prompt fallback.

Google native history is replay data rather than tool authority. Calls reconstructed from that history still pass through the effective grant, policy, HITL, resource locks, final admission, and execution evidence. If required current-turn history is malformed or cannot fit configured memory/context limits, the runtime fails locally instead of retrying the user request or executing a tool again. An incomplete exchange from an earlier user turn is sent as a non-executing description rather than native replay, including when that earlier turn contains trailing assistant text. Missing recorded results do not prove that the tool did not execute or that retry is safe.

If ordinary recovery switches from Google to a different provider after an API or transport failure, the destination receives a readable call/result projection without Google's opaque signed state. A later compatible Google continuation still uses the original memory representation; projection never mutates stored history.

The first-party path reports thinking tokens as completion usage and rejects an SSE response that ends without a terminal `finishReason`. Provisional text already emitted before a stream error cannot be withdrawn, but an incomplete response is not promoted to an authoritative final response. Google `functionResponse.response` values are sent as objects using the compatibility wrapper `{name, content}`, including when a tool returned a scalar, array, or plain string.

`UnifiedLLMProvider::build_llm()` remains a low-level compatibility escape hatch returning the upstream `llm` crate's provider type. Calling that method directly does not use the first-party adapter or its signed-history contract. Google's OpenAI-compatible endpoint also remains outside this fix because its `extra_content.google.thought_signature` representation requires a separate compatibility path.

---

## Ollama Cloud (development)

Set `OLLAMA_API_KEY` and explicitly select a model available on your Ollama account.

```yaml
llm:
  provider: ollama-cloud
  model: deepseek-v4.1-flash
```

The preset sends native requests to `https://ollama.com/api/chat` and preserves the configured model ID. The default key is bound to the official origin. A custom base URL requires an explicitly selected credential for that endpoint. Chat can run independently of catalog availability. Plain text uses the native NDJSON stream when streaming is requested and tool choice is omitted.

Use `tool_choice: auto` for a tool-capable model and grant the function through the usual YAML configuration. The framework handles at most one call per response. It validates the requested function, passes arguments to the existing tool executor and retains the exact assistant message and result binding for supported continuation. `tool_choice: none` omits tool definitions. Requests for `required` or a specific tool are rejected. YAML tool turns use the existing buffered path. Tool-shaped text stays text in ordinary, speculative and resumed turns. A configured fallback can use this preset when the active tool request can be sent natively to both providers.

Omit sampling and output options to use the provider defaults. An explicit `max_tokens` maps to `options.num_predict`. Use `provider_options.think` for a boolean or a model-defined level and `provider_options.options` for native generation options. Declare mapped options only once. Cloud token usage is provider-reported and strict task cost admission requires a separate audited pricing contract. The official endpoint was verified with `deepseek-v4.1-flash` for direct text and streaming, and `glm-5.3-flash` for a YAML tool round trip. CLI process restoration was verified with a local HTTP fixture. [Direct Rust SDK](@/docs/rust-api.md#direct-ollama-cloud-sdk-development)

## Ollama

Ollama runs locally - no API key needed. Start the [Ollama](https://ollama.com) server and pull a model before running your agent.

```sh
ollama serve
ollama pull llama3.1
```

```yaml
llm:
  provider: ollama
  model: llama3.1
```

If Ollama runs on a different host or port, set `base_url`. Native Ollama uses the server root URL, without a `/v1` suffix:

```yaml
llm:
  provider: ollama
  model: llama3.1
  base_url: "http://192.168.1.100:11434"
```

In the development checkout, local chat and text streaming use the native `/api/chat` SDK. The preset sends no authorization header, even when `api_key_env` is present. The configured server root can include a path prefix. The SDK submits one request and does not follow redirects. A stream must end with a daemon completion record to produce a final response. Configured framework tools continue to use the prompt tool path. [Direct Rust SDK](@/docs/rust-api.md#direct-local-ollama-sdk-development)

Existing YAML `top_p` is sent as `options.top_p`. A top-level YAML `top_k` leaves `options.top_k` null. Rust callers can use `LLMConfig::with_top_k` to send a numeric value. The old `num_ctx`, `num_gpu`, `keep_alive`, `temperature`, `max_tokens` and `extra_body` declarations still leave those fields out of local chat requests. Move values that should reach the daemon into native options.

```yaml
llm:
  provider: ollama
  model: llama3.1
  provider_options:
    options:
      num_ctx: 8192
      num_predict: 256
      top_k: 30
    keep_alive: 5m
```

Use `options.num_predict` for the output limit and `keep_alive` for model retention on the daemon. Remove any old declaration for the same setting before adding its native equivalent. `resilient: true` selects the existing upstream transport before generation and cannot be combined with native options. Ollama reports token usage when available. Hard-priced task admission remains unavailable for this preset.

To access Ollama through its OpenAI-compatible endpoint instead, use `provider: openai-compatible` and include `/v1` in the URL:

```yaml
llm:
  provider: openai-compatible
  model: llama3.1
  base_url: "http://localhost:11434/v1"
```

---

## DeepSeek

The current backend supports blocking chat at its fixed service endpoint. It sends the model, messages and temperature. A YAML output limit does not reach the request, and the streaming method returns an unsupported-operation error.

```sh
export DEEPSEEK_API_KEY=sk-...
```

```yaml
llm:
  provider: deepseek
  model: deepseek-chat
```

---

## Groq

```sh
export GROQ_API_KEY=gsk_...
```

```yaml
llm:
  provider: groq
  model: llama-3.3-70b-versatile
```

---

## Mistral

```sh
export MISTRAL_API_KEY=...
```

```yaml
llm:
  provider: mistral
  model: mistral-large-2512
```

---

## Cohere

```sh
export COHERE_API_KEY=...
```

```yaml
llm:
  provider: cohere
  model: command-r-plus-08-2024
```

---

## xAI (Grok)

```sh
export XAI_API_KEY=xai-...
```

```yaml
llm:
  provider: xai
  model: grok-3-mini
```

Other models:

```yaml
# Most capable
llm:
  provider: xai
  model: grok-4.20
```

---

## Phind

The legacy backend reads the service response into one text result. Its framework streaming method returns an unsupported-operation error.

> **Note:** The Phind hosted API endpoint may have limited availability. Verify the endpoint is accessible before relying on this provider in production.

```sh
export PHIND_API_KEY=...
```

```yaml
llm:
  provider: phind
  model: Phind-CodeLlama-34B-v2
```

---

## OpenRouter

The development checkout uses its own Chat adapter for ordinary OpenRouter requests. Existing YAML keeps its model ID, temperature default of `0.7`, output ceiling of `2000` sent as `max_tokens`, and supported `extra_body` extensions. The legacy top-level `reasoning_effort` accepts `low`, `medium`, and `high` and stays off the wire. Use `provider_options.reasoning.effort` to enable reasoning explicitly.

OpenRouter acts as a gateway to many providers through a single API key. Use the full `provider/model` string as the model name:

```sh
export OPENROUTER_API_KEY=sk-or-...
```

```yaml
llm:
  provider: openrouter
  model: anthropic/claude-sonnet-4.6
```

```yaml
llm:
  provider: openrouter
  model: openai/gpt-5.4-mini
```

Existing YAML can set `extra_body: {parallel_tool_calls: false}` to send an explicit boolean. The OpenRouter SDK also accepts `provider_options.parallel_tool_calls` in YAML and `Request::parallel_tool_calls` in Rust. Omission uses the gateway default. Declare the field once across the two YAML maps.

The SDK sends one generation request with no automatic retry or model list lookup. Ordinary calls use the gateway's default routing unless you specify options. Required and specific native tool choices enforce `provider.require_parameters: true` and validate the returned function. An explicit `false` conflicts with those choices. Separate gateway routing from framework fallback aliases. Pin a route before inviting a reasoning model to return native tools that must resume.

```yaml
llm:
  provider: openrouter
  model: anthropic/claude-haiku-4.5
  tool_choice: auto
  provider_options:
    provider:
      only: [anthropic]
      allow_fallbacks: false
    reasoning:
      effort: low
```

The adapter needs one explicit provider route and the exact resolved model to accept a native tool response carrying private reasoning for replay. It preserves the assistant blocks and tool IDs through the existing approval and restart path. Gateway aliases, model fallbacks, and a changed destination can make a reasoning turn ineligible for replay. The regular text stream treats partial output as provisional. HTTP 200 followed by an SSE error or disconnection before `[DONE]` fails the turn.

`provider.max_price` filters unit rates. A hard-priced task uses the host's `DeclaredPricing` contract, which must cover every eligible gateway route and internal attempt. Arbitrary routing overrides, extra billable services and model fallback options require their own audit. The current priced SDK request path rejects explicit gateway extensions before sending a request, including when a route and reasoning are configured for private replay. Final settlement requires one audited attempt and a complete token receipt for the exact model in the pricing contract. A missing or different reported model retains the full reservation, as do missing usage and stream interruption. A framework stream with declared pricing exposes final token usage only for a matching model. The direct SDK returns the gateway’s reported model and usage.

For a custom API origin, select a dedicated `api_key_env` or pass an explicit Rust credential for that destination. Existing `resilient: true` or explicit `normalize_response` select the legacy backend before dispatch and conflict with new SDK options. The low-level `build_llm()` API still exposes the upstream provider. [Direct OpenRouter Rust SDK](@/docs/rust-api.md#direct-openrouter-chat-sdk-development)

---

## OpenAI-Compatible Servers

The `openai-compatible` provider works with any server that exposes an OpenAI-style chat completions API. You **must** provide a `base_url`. The API key is optional and read from whatever env var you specify in `api_key_env`.

```yaml
llm:
  provider: openai-compatible
  model: my-local-model
  base_url: "http://localhost:1234/v1"
```

Some OpenAI-compatible servers support tool calls, vision, or JSON mode, but the framework cannot know that from the URL. Declare capabilities in YAML when you want framework checks to treat the server as capable:

```yaml
llm:
  provider: openai-compatible
  model: qwen3:8b
  base_url: "http://localhost:11434/v1"
  function_calling: true
  json_mode: true
```

These declarations affect framework capability checks, validation, and routing. They are not copied into the provider request body as model parameters. For OpenAI-compatible servers, `function_calling: true` also allows the framework to choose its native tool-request path.

If your server needs an API key:

```yaml
llm:
  provider: openai-compatible
  model: my-model
  base_url: "https://my-server.example.com/v1"
  api_key_env: MY_SERVER_API_KEY
```

```sh
export MY_SERVER_API_KEY=secret-...
```

### LM Studio

[LM Studio](https://lmstudio.ai) exposes an OpenAI-compatible server on port 1234 by default:

```yaml
llm:
  provider: openai-compatible
  model: lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF
  base_url: "http://localhost:1234/v1"
```

### vLLM

[vLLM](https://github.com/vllm-project/vllm) serves models with high throughput:

```sh
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8b-chat-hf --port 8000
```

```yaml
llm:
  provider: openai-compatible
  model: meta-llama/Llama-3.1-8b-chat-hf
  base_url: "http://localhost:8000/v1"
```

### Text Generation Inference (TGI)

[TGI](https://github.com/huggingface/text-generation-inference) from Hugging Face:

```sh
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-3.1-8b-chat-hf
```

```yaml
llm:
  provider: openai-compatible
  model: meta-llama/Llama-3.1-8b-chat-hf
  base_url: "http://localhost:8080/v1"
```

### Capability Overrides

Capability overrides are available for any provider, not only OpenAI-compatible servers:

```yaml
llm:
  provider: ollama
  model: llama3.1
  function_calling: true
  vision: false
  json_mode: true
```

Use them when you know a model/server supports more or less than the provider default. They are declarations for framework routing and validation rather than model parameters. For OpenAI-compatible servers, `function_calling: true` explicitly enables the framework's native tool-request path; `false` disables it. The server must actually implement compatible function calling.

### Tool-choice support

`tool_choice` is optional. Omitting it or setting it to YAML `null` preserves the existing prompt-JSON tool protocol. This unset default is different from `tool_choice: none`, which explicitly disables tool exposure and tool-call parsing for that provider decision. `auto` is explicit opt-in rather than the default. Explicit choices have these provider boundaries:

| Provider | Native `auto` | Native `required` / `specific` | Behavior when native choice is unavailable |
| --- | --- | --- | --- |
| OpenAI | Yes | Yes | Not applicable |
| Anthropic | Yes | Depends on the Messages model and thinking mode. Legacy retains its existing path. | Messages reports an error for an unsupported explicit forced choice. |
| Messages-compatible | Opt in with `function_calling: true` | Requires opt-in and server support | Prompt tools remain available for ordinary auto selection. |
| OpenRouter | Yes | Yes | Not applicable |
| Google first-party GenerateContent | Yes | No | One bounded prompt corrective retry |
| OpenAI-compatible | With `function_calling: true` | With `function_calling: true` | Prompt fallback when not enabled |
| Other unified or custom providers | No by default | No by default | Prompt fallback unless the provider implements the additive native methods |

`none` is enforced by the runtime by exposing neither native definitions nor prompt tool instructions. `required` and `specific` fail if the provider returns no compliant call; prompt fallback gets at most one corrective retry. Selection never grants or authorizes a tool. Only effective granted tools are visible, and returned calls still pass through the shared policy, approval, locking, rate-admission, execution, and evidence path.

---

## Multi-LLM Setup

You can define multiple LLM providers and assign roles to each one. This is useful when you want a fast, cheap model for internal operations (routing, detection, evaluation) but a powerful model for conversation.

### Define named providers with `llms:`

```yaml
llms:
  primary:
    provider: anthropic
    model: claude-sonnet-4-6
    temperature: 0.7
    max_tokens: 4096

  fast:
    provider: groq
    model: llama-3.3-70b-versatile
    temperature: 0.3
    max_tokens: 1024

  local:
    provider: ollama
    model: llama3.1
    temperature: 0.5
    max_tokens: 2048
```

### Map roles with `llm:`

The `llm:` block assigns named providers to framework roles:

```yaml
llm:
  default: primary    # Main conversation model
  router: fast        # Fast model for internal operations
```

The framework has two roles:

| Role | Purpose |
| --- | --- |
| `default` | Main conversation model - generates user-facing responses |
| `router` | Lightweight model for subsystem operations such as skill routing, intent classification, state transition evaluation, language detection, disambiguation, reflection, and summarization |

If `router` is not specified, all operations use `default`. The `router` role is where you save the most cost and latency - internal decisions don't need a frontier model.

### Full example

```yaml
name: MultiLLMAgent
system_prompt: "You are a research assistant."

llms:
  smart:
    provider: anthropic
    model: claude-sonnet-4-6
    temperature: 0.7
    max_tokens: 4096

  quick:
    provider: groq
    model: llama-3.3-70b-versatile
    temperature: 0.2
    max_tokens: 1024

llm:
  default: smart
  router: quick

tools:
  - calculator
  - web_fetch
```

This setup uses Claude for main responses and ordinary model tool selection, while Groq handles configured router and subsystem decisions such as skill routing, semantic transition evaluation, and disambiguation. Deterministic guards do not use either model.

---

## Fallback Configuration

Use the `error_recovery` block to handle main-response provider failures. The top-level retry policy can retry classified transient failures such as rate limits, `llm.on_failure` can select an outcome whenever the primary retry policy ends in failure, and `llm.on_context_overflow` can truncate or summarize prompt history.

```yaml
llms:
  primary:
    provider: openai
    model: gpt-5.4-mini

  fallback:
    provider: ollama
    model: llama3.1

llm:
  default: primary

error_recovery:
  default:
    max_retries: 3
    backoff:
      type: exponential
      initial_ms: 500
      max_ms: 5000
  llm:
    on_failure:
      action: fallback_llm
      fallback_llm: fallback
    on_context_overflow:
      action: summarize
      summarizer_llm: fallback
      keep_recent: 5
```

### Recovery actions

| Scenario | Actions |
| --- | --- |
| `on_failure` | `error` (default), `fallback_llm` (switch to another named LLM), `fallback_response` (return a static message) |
| `on_context_overflow` | `error` (default), `truncate` (drop oldest messages), `summarize` (compress history with an LLM) |

The compatibility `llm.on_rate_limit` field is parsed but does not currently add a separate runtime action. Configure `error_recovery.default.retry_on` to include `rate_limit`, then use `llm.on_failure` for the outcome when the primary retry policy ends in failure.

---

## Tuning Parameters

Use these fields in the `llm:` or `llms:` block. The selected backend determines which fields reach the service.

```yaml
llm:
  provider: openai
  model: gpt-5.4-mini
  temperature: 0.7        # Creativity (0.0 = deterministic, 1.0+ = creative)
  max_tokens: 4096         # Max tokens in the response
  top_p: 0.9               # Nucleus sampling threshold
  timeout_seconds: 120     # Request timeout in seconds
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `temperature` | `f32` | omitted for OpenAI, first-party Messages and Ollama Cloud, `0.7` for existing compatible/legacy paths | Sampling temperature |
| `max_tokens` | `u32` | required for first-party Messages, omitted for Responses and Ollama Cloud, `2000` on existing paths | Maximum response tokens |
| `top_p` | `f32` | - | Nucleus sampling threshold |
| `timeout_seconds` | `u64` | - | Request timeout in seconds |

---

## Reasoning Parameters

For models that support extended thinking (e.g. OpenAI `o3`, `gpt-5.4-mini`), you can control reasoning behavior with first-class YAML fields:

```yaml
llm:
  provider: openai
  model: o3
  reasoning: true
  reasoning_effort: high
  reasoning_budget_tokens: 16384
  timeout_seconds: 120
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `reasoning` | `bool` | - | Enable extended thinking / reasoning mode |
| `reasoning_effort` | `string` | - | Model-specific effort string on the first-party Chat path; legacy adapters can restrict values |
| `reasoning_budget_tokens` | `u32` | - | Maximum token budget for reasoning |

These work across any provider that supports reasoning. Use `timeout_seconds` alongside reasoning-heavy models since they can take longer to respond.

---

## Extra Parameters

Any YAML field not recognized as a named parameter is captured as a provider-specific extra. Providers decide which extras they support. The first-party Google adapter accepts a restricted `extra_body`: framework-owned `contents`, `systemInstruction`, `tools`, and `toolConfig` cannot be overridden, while unsupported top-level fields fail configuration instead of being forwarded silently. Tool definitions use Google's `parametersJsonSchema`, preserving supported JSON Schema constraints.

### Transport-Level Resilience

The `llm` crate supports HTTP-level retry with exponential backoff, complementary to the framework's agent-level `error_recovery`. Both can be active simultaneously.

These `resilient_*` fields apply to upstream-backed providers. The first-party Google adapter does not add an internal retry loop; configure agent-level `error_recovery` for Google. Local native-history and protocol integrity errors are terminal and do not enter retry, fallback-model, or static-response recovery.

```yaml
llm:
  provider: openai
  model: gpt-5.4-mini
  resilient: true
  resilient_attempts: 3
  resilient_base_delay_ms: 1000
  resilient_max_delay_ms: 30000
  resilient_jitter: true
```

| Parameter | Type | Description |
| --- | --- | --- |
| `resilient` | `bool` | Enable transport-level retry |
| `resilient_attempts` | `u64` | Max retry attempts |
| `resilient_base_delay_ms` | `u64` | Base delay for exponential backoff |
| `resilient_max_delay_ms` | `u64` | Maximum backoff delay |
| `resilient_jitter` | `bool` | Add randomness to backoff timing |

> **Note:** If you also have an `error_recovery:` section in your agent spec, main-response provider retries can multiply transport-level attempts. `error_recovery.default.max_retries` counts retries after the initial provider attempt. Record actual calls rather than estimating them from one product formula because transport adapters may apply their own attempt semantics.

### Azure OpenAI

The legacy fields `api_version` and `deployment_id` select the upstream Chat backend. It appends `chat/completions` to the configured base URL and omits the deployment path and version query. A resource root plus those two fields is insufficient for a versioned Azure deployment request.

### OpenAI provider-native web search

Provider-native search runs on the service. The separately granted `web_search` tool runs through `WebSearchProvider`, framework tool policy, HITL and tool-execution evidence. [Built-in Tools](@/docs/built-in-tools.md#web-search-versus-web-fetch)

On the OpenAI and OpenAI-compatible presets, `openai_enable_web_search: true` selects the upstream backend. Ordinary framework chat still sends a Chat Completions request without a hosted search tool. The upstream Rust client exposes a separate `chat_with_web_search` operation.

### xAI provider-native search

The current upstream xAI backend sends search parameters on blocking chat and omits them from streaming requests:

```yaml
llm:
  provider: xai
  model: grok-3
  xai_search_mode: auto
  xai_max_search_results: 10
```

### `extra_body` (Escape Hatch)

On backends that support `extra_body`, use it for additional request fields. For example, the Chat adapter accepts the following options:

```yaml
llm:
  provider: openai
  model: gpt-5.4-mini
  extra_body:
    logprobs: true
    top_logprobs: 5
```

---

## Next Steps

- **[Getting Started](@/docs/getting-started.md)** - install and run your first agent
- **[CLI Guide](@/docs/cli.md)** - all commands, flags, and REPL features
- **[Rust API](@/docs/rust-api.md)** - embed agents in your Rust application
- **[YAML Reference](@/docs/yaml-reference.md)** - the complete agent specification

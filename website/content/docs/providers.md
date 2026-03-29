+++
title = "LLM Providers"
weight = 5
template = "docs.html"
description = "Setup guide for all supported LLM providers."
+++

AI Agents supports 12 LLM providers out of the box - from cloud APIs like OpenAI and Anthropic to local servers like Ollama and any OpenAI-compatible endpoint. This page covers setup for each one with copy-paste-ready YAML snippets.

---

## Overview

| Provider | YAML `provider:` value | Environment Variable | Example Models |
| --- | --- | --- | --- |
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-5.1-nano`, `gpt-5.1-mini`, `gpt-5.1` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4.6`, `claude-haiku-4.5` |
| Google Gemini | `google` | `GOOGLE_API_KEY` | `gemini-2.5-flash`, `gemini-2.5-pro` |
| Ollama | `ollama` | *(none)* | `llama3.1`, `qwen3:8b`, `mistral` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-chat`, `deepseek-reasoner` |
| Groq | `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| Mistral | `mistral` | `MISTRAL_API_KEY` | `mistral-large-2512`, `mistral-small-2503` |
| Cohere | `cohere` | `COHERE_API_KEY` | `command-r-plus-08-2024`, `command-r-08-2024` |
| xAI (Grok) | `xai` | `XAI_API_KEY` | `grok-4.20`, `grok-3-mini` |
| Phind | `phind` | `PHIND_API_KEY` | *(legacy - verify endpoint availability)* |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | `openai/gpt-5.4-mini`, `anthropic/claude-sonnet-4.6` |
| OpenAI-compatible | `openai-compatible` | *(set via `api_key_env`)* | *(depends on server)* |

---

## OpenAI

Set the API key and pick a model:

```sh
export OPENAI_API_KEY=sk-...
```

```yaml
llm:
  provider: openai
  model: gpt-4.1-nano
```

Other model options:

```yaml
# Good balance of speed and quality
llm:
  provider: openai
  model: gpt-5.1-mini

# Most capable
llm:
  provider: openai
  model: gpt-5.1

# Reasoning-capable model with effort control
llms:
  default:
    provider: openai
    model: gpt-5.4-mini
    reasoning_effort: low    # low | medium | high
llm:
  default: default
```

The `reasoning_effort` parameter controls how much reasoning the model applies before answering. It is passed through as a provider-specific extra parameter. See [Extra Parameters](#extra-parameters) below.

---

## Anthropic

```sh
export ANTHROPIC_API_KEY=sk-ant-...
```

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4.6
```

Other models:

```yaml
# Fast and affordable
llm:
  provider: anthropic
  model: claude-haiku-4.5
```

---

## Google Gemini

```sh
export GOOGLE_API_KEY=AI...
```

```yaml
llm:
  provider: google
  model: gemini-3-flash
```

Other models:

```yaml
llm:
  provider: google
  model: gemini-3-pro
```

---

## Ollama

Ollama runs locally - no API key needed. Just make sure the [Ollama](https://ollama.com) server is running on the default port (`http://localhost:11434`).

```sh
# Pull a model first
ollama pull llama3.1
```

```yaml
llm:
  provider: ollama
  model: llama3.1
```

If Ollama runs on a different host or port, set `base_url`:

```yaml
llm:
  provider: ollama
  model: llama3.1
  base_url: "http://192.168.1.100:11434"
```

---

## DeepSeek

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
  model: openai/gpt-5.1-mini
```

---

## OpenAI-Compatible Servers

The `openai-compatible` provider works with any server that exposes an OpenAI-style chat completions API. You **must** provide a `base_url`. The API key is optional and read from whatever env var you specify in `api_key_env`.

```yaml
llm:
  provider: openai-compatible
  model: my-local-model
  base_url: "http://localhost:1234/v1"
```

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

---

## Multi-LLM Setup

You can define multiple LLM providers and assign roles to each one. This is useful when you want a fast, cheap model for internal operations (routing, detection, evaluation) but a powerful model for conversation.

### Define named providers with `llms:`

```yaml
llms:
  primary:
    provider: anthropic
    model: claude-sonnet-4.6
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
| `router` | Lightweight model for internal operations - tool selection, intent classification, state transition evaluation, language detection, disambiguation, reflection, summarization |

If `router` is not specified, all operations use `default`. The `router` role is where you save the most cost and latency - internal decisions don't need a frontier model.

### Full example

```yaml
name: MultiLLMAgent
system_prompt: "You are a research assistant."

llms:
  smart:
    provider: anthropic
    model: claude-sonnet-4.6
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
  - http
```

This setup uses Claude for generating responses and Groq for fast internal routing - saving cost and latency on tool selection, guard evaluation, and disambiguation.

---

## Fallback Configuration

Use the `error_recovery` block to handle provider failures gracefully. If the primary LLM fails, the framework can fall back to another provider, wait and retry on rate limits, or compress context when it overflows.

```yaml
llms:
  primary:
    provider: openai
    model: gpt-5.1-mini

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
    on_rate_limit:
      action: wait_and_retry
      max_wait_ms: 10000
    on_context_overflow:
      action: summarize
      summarizer_llm: fallback
      keep_recent: 5
```

### Recovery actions

| Scenario | Actions |
| --- | --- |
| `on_failure` | `error` (default), `fallback_llm` (switch to another named LLM), `fallback_response` (return a static message) |
| `on_rate_limit` | `error` (default), `wait_and_retry` (pause then retry), `switch_model` (switch to another named LLM) |
| `on_context_overflow` | `error` (default), `truncate` (drop oldest messages), `summarize` (compress history with an LLM) |

This means your agent stays up even when a provider has an outage - it just switches to the fallback automatically.

---

## Tuning Parameters

Every provider accepts these optional parameters in the `llm:` or `llms:` block:

```yaml
llm:
  provider: openai
  model: gpt-5.4-mini
  temperature: 0.7      # Creativity (0.0 = deterministic, 1.0+ = creative)
  max_tokens: 4096       # Max tokens in the response
  top_p: 0.9             # Nucleus sampling threshold
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `temperature` | `f32` | `0.7` | Sampling temperature |
| `max_tokens` | `u32` | `2000` | Maximum response tokens |
| `top_p` | `f32` | - | Nucleus sampling threshold |

---

## Extra Parameters

Any field not recognized by the framework is captured as a provider-specific extra parameter and passed through to the underlying LLM client. This allows provider-specific features without framework changes.

### `reasoning_effort` (OpenAI)

For OpenAI reasoning-capable models (e.g. `gpt-5.4`, `gpt-5.4-mini`), you can control how much reasoning the model applies:

```yaml
llms:
  default:
    provider: openai
    model: gpt-5.4-mini
    reasoning_effort: low    # low | medium | high
```

| Value | Behavior |
| --- | --- |
| `low` | Minimal reasoning - fast and cheap |
| `medium` | Balanced reasoning |
| `high` | Maximum reasoning - slower but more thorough |

---

## Next Steps

- **[Getting Started](@/docs/getting-started.md)** - install and run your first agent
- **[CLI Guide](@/docs/cli.md)** - all commands, flags, and REPL features
- **[Rust API](@/docs/rust-api.md)** - embed agents in your Rust application
- **[YAML Reference](@/docs/yaml-reference.md)** - the complete agent specification

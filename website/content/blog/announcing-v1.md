+++
title = "AI Agents v1.0: The Runtime Behind One YAML"
date = 2026-08-02
description = "The stable, versioned Rust runtime contract behind YAML-defined agents."
template = "blog-page.html"
[taxonomies]
tags = ["release", "architecture", "design"]
+++

> This post describes the v1.0 release as of its publication date. Current supported behavior and operational boundaries are defined by the [Concepts](@/docs/concepts.md), [YAML Reference](@/docs/yaml-reference.md), and [Built-in Tools](@/docs/built-in-tools.md) pages.

AI Agents v1.0 is now available.

This release is not centered on one new feature. It is the point where the runtime's core behavior becomes a versioned public contract.

Across the release candidates, the project grew from a YAML-driven chat runtime into something broader. It can now coordinate tools and workflows, carry actor-aware state, orchestrate agents, and evaluate and observe its own execution.

A configuration file is useful only when its runtime behavior is predictable. Users need to know what a model can access, which effects may commit, what gets persisted, and what evidence remains when execution does not happen.

v1.0 answers those questions.

## YAML Is the Visible Part

The visible part of this framework is YAML.

A YAML file describes an agent's instructions, models, tools, workflows, context, memory, and behavioral rules. It can be reviewed as data, changed without rewriting the host application, and run from the CLI or embedded in Rust.

But YAML is only the visible layer.

What matters is giving every section a consistent runtime meaning. A state transition should not take effect twice because the runtime redispatched internally. A tool registered by the host should not automatically become callable by the model. A discarded draft should not write memory or run a tool call.

Those are runtime contracts, not prompt instructions.

The purpose of v1.0 is to make those rules explicit. The framework remains YAML-first, but the host retains control over construction, capabilities, policy, persistence, and deployment boundaries.

## One YAML, One Runtime

A small agent can still be small:

```yaml
name: SupportAgent
system_prompt: |
  You help users resolve technical issues.
  Ask for clarification when the problem is ambiguous.

llm:
  provider: openai
  model: gpt-5.4-nano

tools:
  - datetime
  - calculator
```

Install the CLI and run it directly:

```sh
cargo install ai-agents-cli --version '=1.0.0'
ai-agents-cli run agent.yaml
```

The same YAML can be loaded by a Rust application, evaluated with separate scenario suites, or used in a multi-agent system. These paths converge on the same builder and `RuntimeAgent` execution model.

That shared path keeps tested behavior aligned with CLI and embedded use. A tool called from a skill should not bypass normal execution rules, and a spawned child should not bypass them because it was created dynamically.

The framework is modular internally, but execution comes back to one runtime contract.

## The Hard Part Begins When Agents Can Act

Generating a response is only the beginning.

A tool call can inspect a repository, modify a file, run a validation command, ask a user a question, or access an external service. At that point, telling the model to "be careful" is not enough. The host needs an execution contract.

In v1.0, registering a tool does not automatically expose it. YAML agents grant ordinary tools explicitly, and omitting `tools:` grants none. States may narrow the top-level grant but cannot widen it. Provider tool choice can select among effective tools, but it cannot create new authority.

```text
Host registers tools
  -> YAML grants tools
  -> state narrows the grant
  -> provider sees the effective tools
  -> model returns a tool call
  -> runtime checks scope, policy, and approval
  -> final admission and resource locks
  -> tool execution and evidence
```

A granted call still enters one shared execution path. The runtime resolves the tool's canonical identity and rejects calls outside the effective scope or policy before asking for approval. If approval changes the arguments, the final call is checked again before the bound tool executes.

Human approval can authorize an allowed but sensitive operation. It cannot override an automatic denial.

The same boundary applies to calls emitted by a model, skill, state action, plan, or fallback, and to manual calls that the host submits through the runtime's shared executor.

Only the selected branch may commit side effects. A speculative branch may consume model time, but a losing branch cannot write memory, mutate context, run tools, or emit user-visible output. Its usage evidence can remain without allowing its effects to escape.

These controls make behavior inspectable inside the framework. They are not an operating-system sandbox. Deployment isolation, credentials, custom integrations, and provider-side behavior remain host responsibilities.

## More Than Chat

The runtime is conversational at its core, but a conversation can carry much more than messages.

Hierarchical states give behavior structure. Skills package reusable workflows. Process stages prepare input and output, while context sources bring structured runtime values into prompts, state-transition guards, and tools.

The runtime keeps conversation memory, session snapshots, actor facts, relationship state, and persona evolution as distinct runtime concerns.

Agents can also create and coordinate other agents inside the runtime. Agent spawning is limited by configured capacity and tool allowlists, and declared child construction fails as a whole when any child cannot be admitted. Orchestration supports routing, pipelines, concurrent aggregation, group chat, and handoff. This is coordination within one runtime, not a general graph engine or a cross-service agent protocol.

For example, a support agent can route a request through states, retain actor facts in SQLite, ask for approval before a sensitive tool call, and hand work to a specialist without changing the shared execution rules.

The result is a set of composable primitives rather than one prescribed agent shape. Each agent enables only the sections it needs.

## Evidence Instead of Trusting Output Text

A model response can sound correct while the runtime did the wrong thing.

It may claim that a tool ran when execution was denied, or that a task completed without reaching the expected state.

That is why evaluation in v1.0 is based on structured runtime evidence as well as final responses.

Separate YAML or JSONL suites can run normal agents with mock or replay fixtures, or through explicitly authorized record and live-provider modes. Deterministic assertions can inspect state transitions, exact tool identity and arguments, execution and non-execution outcomes, approvals, actor memory, orchestration, and speculative branches.

Mocked evaluation is the default reproducible path. Provider-backed execution remains explicit because it uses credentials, network access, money, and nondeterministic external models. Semantic judges are available when meaning must be evaluated, but they do not replace deterministic checks for structural behavior.

Observability follows the same principle. It records bounded runtime facts such as latency, token and cost estimates, tool outcomes, and branch status. Raw prompts, responses, tool data, actor memory, and persona secrets are not retained by default.

## What Stable Means

v1.0 does not mean every subsystem has the same maturity or operational boundary.

The builder and blocking chat path, strict YAML, states, skills, process, context, built-in tool authorization, and in-memory or compacting conversation memory form the stable core contract. These surfaces now follow normal v1 SemVer, with compatibility changes recorded in the [changelog](https://github.com/geminik23/ai-agents/blob/v1.0.0/CHANGELOG.md).

Streaming, evaluation, observability, provider adapters, and MCP are supported within their documented boundaries. The same applies to persona, file and SQLite storage, actor facts, relationships, spawning, orchestration, and runtime optimization. External providers, services, scheduling behavior, and operational tuning may continue to evolve.

Redis remains snapshot-only and experimental for v1. Noop storage provides no persistence. Unsupported persistence operations return explicit capability errors rather than succeeding silently.

These distinctions are intentional. A stable release should make its limits easier to understand, not hide them behind one broad maturity claim.

The complete support definitions and subsystem boundaries are documented in [Concepts](@/docs/concepts.md).

## What Comes After v1.0

v1.0 establishes the current runtime contract, but it is not the end of the roadmap.

It does not yet include a generalized autonomy runner for host-enforced long-running tasks. Retrieval and evidence, an extensible knowledge and RAG pipeline, evidence access policy, generalized background scheduling, and Python runtime bindings also remain future work without assigned release versions.

The current runtime can reason, plan, call tools, and maintain todos inside normal turns. A generalized autonomy runner needs a different contract: when may the host stop, what counts as completion, and how are long runs paused, resumed, and bounded? That work should extend the runtime rather than quietly change what a conversation turn means.

Future features will build on v1.0's execution, evidence, persistence, observability, and evaluation boundaries rather than replace them.

## A Technical Report Will Follow

A separate Technical Report is being prepared for publication after the release.

The report will document the architecture, threat model, execution boundaries, evaluation methodology, limitations, and reproducibility artifacts behind v1.0. Its results will be tied to the immutable release commit rather than a moving development branch.

The release tells you what is available. The report will explain how the system is structured and which claims are backed by deterministic evidence. It will separate live-provider results from runtime guarantees and make the system's limits explicit.

## The Runtime Behind One YAML

The goal of v1.0 is not to claim that agent systems are finished.

It is to make the runtime contract explicit. Common behavior begins in YAML, custom capabilities enter through Rust traits, and model decisions remain bounded by host-controlled execution. The same runtime carries those rules through conversation, tool execution, state, memory, orchestration, observability, and evaluation.

One YAML was always the visible part of the idea. v1.0 is the runtime contract behind it.

If you want to try it:

- [Get started](@/docs/getting-started.md) - install the CLI and run your first agent
- [Read the concepts](@/docs/concepts.md) - understand the runtime and support boundaries
- [Browse the examples](@/examples/_index.md) - explore states, tools, memory, and orchestration
- [Review the built-in tools](@/docs/built-in-tools.md) - see inputs, outputs, policy, and host requirements
- [Read the v1.0 release notes](https://github.com/geminik23/ai-agents/releases/tag/v1.0.0) - review the complete release

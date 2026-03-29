+++
title = "Why One YAML = Any Agent?"
date = 2026-03-29
description = "The design philosophy behind declarative AI agents."
template = "blog-page.html"
[taxonomies]
tags = ["philosophy", "design"]
+++

AI agent frameworks often make a simple idea feel harder than it needs to be.

Most agents are really just a set of decisions. Which LLM do they use? What tools can they call? How do they move between states? What do they remember? How do they recover from failure? What kind of response should they produce?

Those are configuration questions. But in many frameworks, the answers are scattered across classes, registries, helper functions, and glue code. By the time everything is wired together, the most important part of the agent, its actual behavior, is buried under implementation detail.

This framework starts from a different premise. If agent behavior can be declared, it should not require code. 

That idea is where **One YAML = Any Agent** comes from.

The goal is simple. A single file should define the agent's instructions, capabilities, and behavioral rules. You should be able to read that file and understand how the agent works. And if you want to change the behavior, editing the file should be enough.

Here is what a working customer support agent looks like:

```yaml
name: SupportAgent
system_prompt: |
  You are a friendly support agent for Acme Corp.
  Help customers with their questions.
  Escalate billing issues to a human.

llm:
  provider: openai
  model: gpt-4.1-nano

tools:
  - datetime

states:
  initial: greeting
  states:
    greeting:
      prompt: "Greet the customer warmly."
      transitions:
        - to: helping
          when: "Customer has stated their problem"
    helping:
      prompt: "Help resolve the issue."
      transitions:
        - to: escalation
          when: "Issue involves billing or payments"
        - to: resolved
          when: "Issue is resolved"
    escalation:
      prompt: "Let the customer know a billing specialist will take over."
    resolved:
      prompt: "Wrap up and say goodbye."
```

That is a real agent. It greets customers, helps them, escalates billing issues, and wraps up. You run it with `ai-agents-cli run support_agent.yaml` and it works. No Rust. No Python. No compile step.

A common question is how state transitions work without code. Traditional chatbot frameworks usually solve this with regex patterns or keyword matchers. That works up to a point, but it breaks down quickly once people start phrasing the same intent in different ways. "I want to cancel" may not match a pattern written for "please cancel my subscription." And once you go beyond English, the problem gets much worse.

This framework takes a very different approach: it gives the entire transition decision to the LLM.

Not just the ambiguous cases. Every transition, every intent check, every validation. There is no regex fallback for the cases that seem easy enough to hardcode. If a transition depends on understanding meaning, the LLM handles it.

That means the model reads the conversation itself and decides whether a transition should happen. In practice, this works much better across natural phrasing, indirect wording, and multiple languages. You can write:

```yaml
when: "Customer wants to cancel"
```

and the agent can still correctly respond to inputs like "I'd like to cancel," "해지하고 싶어요," or "もうやめたい" without adding language-specific rules or pattern lists. The result is that the YAML stays small, while covering far more variation than pattern matching usually can.

Another concern is whether putting everything in YAML eventually leads to giant config files. It does not. The reason is layered defaults.

A minimal agent is only five lines:

```yaml
name: MyBot
system_prompt: "You are a helpful assistant."
llm:
  provider: openai
  model: gpt-4.1-nano
```

No `tools` section means no tools. No `states` section means free-form chat. No `memory` section means in-memory session storage. You only add sections when you need them. The file grows with the real complexity of the agent, not with the accidental complexity of the framework.

Each feature also supports layered overrides. You set a default at the agent level, then override it per state, then override it again per skill. For example, you can use a cheap fast model everywhere, but switch to a more capable one for a specific state:

```yaml
llms:
  default:
    provider: openai
    model: gpt-4.1-nano
  deep:
    provider: openai
    model: gpt-4.1-mini

states:
  states:
    greeting:
      prompt: "Greet the customer warmly."   # uses default model
    analysis:
      prompt: "Analyze the customer's situation carefully."
      llm: deep                               # overrides to the capable model
```

Simple agents stay simple. Complex agents get fine-grained control exactly where they need it, and nowhere else.

For most engineers, the next question is: what about the cases that genuinely need custom logic? That is where Rust traits come in. Every extension point in the framework is a trait you implement in Rust: a custom tool that calls an internal API, a memory backend that talks to your own database, an approval handler that posts to Slack.

The key is that YAML and traits compose cleanly instead of competing. You load the YAML, inject the custom piece, and the rest of the agent behavior stays declarative.

```rust
use ai_agents::AgentBuilder;
use std::sync::Arc;

let agent = AgentBuilder::from_yaml_file("agent.yaml")?
    .auto_configure_llms()?
    .auto_configure_features()?
    .tool(Arc::new(MyCustomTool))
    .build()?;
```

YAML remains the source of behavioral configuration. Rust handles custom extensions. Someone who has never written Rust can still define and run agents from the CLI. A team can review agent behavior in a pull request without digging through framework internals. And when a developer needs a custom tool or integration, they can implement that one piece in Rust without moving the rest of the agent definition out of YAML.

One file, one agent. It is a simple idea, but it has already covered a surprisingly wide range of agent designs.

## Design Principles

These are the principles behind the framework:

1. **Language agnosticism** - no regex for semantic operations. The LLM handles all languages equally.
2. **Declarative over imperative** - if something can be declared, it should not require code.
3. **LLM-first for semantics** - intent detection, entity extraction, validation, and transitions should be driven by the LLM, not by pattern matching.
4. **Layered configuration** - defaults set globally and overridden only where needed, whether per agent, per state, or per skill.
5. **Opt-in complexity** - every feature has sensible defaults. A minimal agent stays minimal.
6. **Production safety** - error recovery, tool security, human-in-the-loop approvals, and budget control are part of the core design from the start.
7. **Extensibility via traits** - `LLMProvider`, `Tool`, `Memory`, `ApprovalHandler`, and `AgentHooks` are easy to extend with small, focused implementations.
8. **Composability** - agents can be spawned, orchestrated, chained, and federated. One agent is a building block, not a dead end.

If you want to try it:

- [Get started](@/docs/getting-started.md) - install and run your first agent in under a minute
- [Browse examples](@/examples/_index.md) - from simple chatbots to multi-state workflows
- [GitHub](https://github.com/geminik23/ai-agents) - the code, the issues, and the roadmap

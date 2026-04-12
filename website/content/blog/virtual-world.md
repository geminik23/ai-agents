+++
title = "How a Virtual World Led to This Framework"
date = 2026-04-12
description = "How game experiments grew into a virtual world vision that shaped the core design of this framework."
template = "blog-page.html"
[taxonomies]
tags = ["design", "architecture", "philosophy"]
+++

This framework did not start as an agent framework.

It started with game experiments. The [first version](https://github.com/geminik23/ai-agents/tree/v0-maintenance/examples/find-treasure) was a small treasure-hunt RPG built in April 2024. The model generated towns, NPCs, and dialogue on the fly, and it was enough to prove the idea was interesting. But every character and interaction was hardcoded in Rust. Adding a new kind of NPC meant changing code. Changing behavior meant recompiling.

As I thought more about what dynamic NPCs actually required, the problem kept getting bigger. Characters needed identities, memories, behavioral rules, and the ability to react to each other without constant human control. At some point it stopped looking like a game prototype and started looking like a virtual world.

That shift changed everything. Building a virtual world is not about solving one hard problem. It is about solving many hard problems at the same time.

When I came back to the problem in April 2025, I decided to start over from scratch. Not because the early game experiments were wrong, but because the virtual world demanded something much broader from the system.

A character in a virtual world needs an identity. Not just a system prompt, but a structured personality, a role, goals, a way of speaking. It needs to stay consistent across conversations without drifting into generic chatbot behavior.

But identity is not enough without memory. The character also needs to remember that you helped it three days ago, or that another character lied to it last week. If it forgets everything between sessions, it stops feeling real.

Memory is not enough without behavior. A guard should patrol, get suspicious, confront, and call for backup. A merchant should greet, trade, and gossip. Those are state machines, but their transitions are driven by meaning rather than simple keywords. "I have a bad feeling about that stranger" should trigger suspicion just as well as "that person looks dangerous."

Behavior is not enough without coordination. A guard who notices trouble should be able to alert other guards. Town leaders should be able to hold a discussion. This requires agents to communicate with each other, make decisions as groups, and hand conversations off to other agents.

And none of it works if the set of agents is fixed. When the world changes in an unexpected way, the system cannot stop and wait for a new character to be defined. It has to create new agents at runtime, with behavior adapted to what just happened.

No single one of these requirements is unusual. But a virtual world demands all of them at once, in the same system, working together.

That is the design pressure that shaped this framework.

## Why Behavior Had to Become Data

The first realization was that a changing world cannot depend on behavior being frozen in code.

If every character requires a Rust struct, a set of handler functions, and a compile step, then the world is frozen at deploy time. You can only have the characters you wrote in advance. The world cannot adapt.

But if agent behavior is defined in YAML, behavior becomes data. A state machine, a personality, a set of tools, a memory configuration, and a set of transition rules can all be written, read, modified, and generated without touching application code.

A guard that patrols a harbor can be defined like this:

```yaml
name: HarborGuard
system_prompt: |
  You are a harbor guard in a trading port.
  You are cautious and direct.

llm:
  provider: openai
  model: gpt-5.4-nano

states:
  initial: patrolling
  states:
    patrolling:
      prompt: "Question anyone who looks suspicious."
      transitions:
        - to: confrontation
          when: "The person becomes aggressive or evasive"
        - to: conversation
          when: "The person seems friendly and cooperative"
    confrontation:
      prompt: "Handle the threat. Call for backup if needed."
    conversation:
      prompt: "Talk naturally. You are still a guard, not a friend."
```

That is a complete character definition. It includes a role, a behavior flow, and transition rules that work across languages because the LLM evaluates them, not a regex engine.

And because it is just text, another agent can generate it.

## Why a Static Cast Was Never Enough

A virtual world is unpredictable. Events happen that no designer anticipated. The system needs a way to create new agents on the fly, with behavior that fits the current situation.

That led to the idea of a game master: one agent that observes the world and spawns other agents as needed. When a fight breaks out, the game master creates a guard. When a new trader arrives, the game master creates a merchant. When a rumor spreads, the game master creates a gossip.

The game master does not write code. It turns the current world state into a YAML definition and hands it to the framework. The framework builds a running agent from it and registers it.

The spawned agent is small and focused. It has a narrow prompt, a limited set of tools, and a clear role. But that is not the main point. What matters is that the game master decided to create it for this situation; no human had to write code for the encounter in advance. When the encounter is over, the agent can be removed cleanly, with no leftover state and no confusion between one interaction and the next.

## Why Isolated Agents Do Not Make a World

Once characters exist, they need to affect each other.

If a guard detects an intruder, one conversation is no longer enough. The guard needs to alert nearby guards. The tavern owner may react differently. Witnesses may spread the story. Once characters influence each other, isolated agents stop being enough.

That means the framework needs coordination primitives. Routing messages to the right agent. Running multiple agents in parallel and combining their responses. Letting a group of agents take turns in a conversation. Handing control from one agent to another.

All of them are defined in YAML, and all of them rely on LLM decisions instead of hardcoded routing logic.

## Why This Generalizes Beyond Games

The virtual world forced these requirements. But none of them are specific to games.

A customer support system needs agents with consistent identities and memory across sessions. A research platform needs a coordinator that creates specialist agents for different topics. An operations system needs agents that alert each other when something goes wrong. A content pipeline needs agents that pass work through a chain of reviewers.

The hard part was never really "NPCs." The hard part was building a system where agents have identity, memory, behavioral structure, coordination, and the ability to be created at runtime.

Those are general-purpose primitives. The virtual world was just the problem that made it impossible to skip any of them.

## Where This Leads

The framework now supports the main pieces the virtual world demanded: declarative behavior in YAML, hierarchical state machines with LLM-evaluated transitions, persistent memory with summarization, multi-agent orchestration, and dynamic spawning from templates. The next step is structured identity: personality, goals, and speaking style as queryable data, separate from the system prompt.

The virtual world is still the long-term goal. But along the way, the same primitives have already proven useful for support workflows, research pipelines, and multi-agent coordination that have nothing to do with games.

That was the real lesson. A framework that can handle a virtual world can handle most things.

And that is why this framework looks the way it does.

If you want to try it:

- [Get started](@/docs/getting-started.md) - install and run your first agent in under a minute
- [Browse examples](@/examples/_index.md) - from simple chatbots to multi-agent orchestration
- [GitHub](https://github.com/geminik23/ai-agents) - the code, the issues, and the roadmap

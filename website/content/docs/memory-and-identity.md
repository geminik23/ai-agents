+++
title = "Remember People and Conversations"
weight = 22
template = "docs.html"
description = "Separate conversation history, saved sessions, actor facts, and relationships."

[extra]
learning_guide = true
+++

Decide what must survive before choosing a memory feature. Recent messages, a saved conversation, and a returning person's preferences have different owners and lifetimes.

## Choose what to remember

| Information | Feature | Lifetime and scope |
| --- | --- | --- |
| Recent turns | In-memory conversation memory | The active runtime, bounded by its message window |
| Older conversation context | Compacting memory | A summary plus a recent verbatim tail |
| A conversation you can resume | Storage snapshots | Explicit save and load across runtime instances |
| A person's preferences and decisions | Actor facts | Keyed by agent and actor identity |
| Trust, familiarity, and other dimensions | Relationship memory | Keyed by agent and actor identity |
| The agent's character and presentation | Persona | The agent's configured identity |

Use the [conversation memory example](@/examples/recipe-conversation-memory.md) first. Ask a question that depends on an earlier turn, inspect the memory, then reset the conversation and compare the result.

## Bound conversation history

This fragment retains a limited rolling window:

```yaml
memory:
  type: in-memory
  max_messages: 50 # Keep a bounded number of conversation messages.
```

Choose compacting memory when a long conversation should retain a summary of older material. Configure the compression threshold, recent-message protection, summarizer model, and token allocation together. Summarization is a model operation: it consumes model resources and can lose detail. Test the information your application actually needs to retain.

The repository has separate [basic](https://github.com/geminik23/ai-agents/blob/main/examples/yaml/memory/memory_basic.yaml), [compacting](https://github.com/geminik23/ai-agents/blob/main/examples/yaml/memory/memory_compacting.yaml), and [token budget](https://github.com/geminik23/ai-agents/blob/main/examples/yaml/memory/memory_budget.yaml) examples.

## Save a session or recognize an actor

A snapshot restores a conversation. Actor memory lets a new conversation retrieve facts about the same actor. These are separate operations; seeing a preference in recent message history does not prove cross-session recall.

Use a stable actor ID from your application or, for explicit identification, the CLI's `--actor` option. Agents configured with `identification.method: from_context` instead read their configured context path. Facts and relationships are scoped by `(agent_id, actor_id)`. The host must associate that ID with the intended user; memory configuration does not authenticate the person.

Among built-in storage backends, file and Redis support snapshot operations. SQLite additionally supports actor facts, actor relationships, and richer session metadata. Unsupported backend operations return capability errors. Select the backend for the operations you need, not only the storage format you prefer.

Follow the [multi-actor session example](https://github.com/geminik23/ai-agents/blob/main/examples/yaml/session/multi_actor.yaml) to study context-based actor identification and persistence. It reads `player.id`; use `--context player.id=player_1` for that example rather than `--actor`. Keep its storage configuration and required prompt variables when adapting it.

## Add relationship memory only when useful

Facts describe what the agent knows about an actor. Relationships describe how it currently relates to that actor. The [support relationship example](https://github.com/geminik23/ai-agents/blob/main/examples/yaml/relationships/support_relationship.yaml) demonstrates dimensions such as trust and rapport alongside actor memory.

The evaluator proposes bounded updates; the runtime validates and clamps those updates. Relationship values can inform prompts and configured conditions. They do not replace authentication or application authorization.

## Test with a fresh runtime

To prove durable memory, save the relevant data, rebuild the runtime with the same storage and actor identity, and ask a question that depends on it. Repeat with another actor and verify isolation. The [evaluation reference](@/docs/evaluation.md) includes fixtures and assertions for facts, relationships, and session persistence.

Continue with [memory configuration](@/docs/yaml-reference.md#memory), [storage capabilities](@/docs/yaml-reference.md#storage), [actor facts](@/docs/concepts.md#actor-memory-key-facts), and [relationship memory](@/docs/concepts.md#relationship-memory).

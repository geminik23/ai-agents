+++
title = "Coordinate Multiple Agents"
weight = 23
template = "docs.html"
description = "Choose routing, pipelines, concurrent work, discussions, or handoffs."

[extra]
learning_guide = true
+++

Use multiple agents when a task benefits from distinct instructions, expertise, or stages. The parent owns the user conversation; registered child agents do the delegated work. Start with a fixed pattern whose result you can inspect.

## Match the pattern to the task

| Pattern | Use it for | Example |
| --- | --- | --- |
| Router | Select a specialist for the request | [Customer support](@/examples/recipe-delegate-support.md) |
| Pipeline | Pass work through ordered stages | [Write, review, edit](https://github.com/geminik23/ai-agents/blob/main/examples/yaml/orchestration/content_pipeline.yaml) |
| Concurrent | Ask independent specialists and aggregate | [Multiple analysis perspectives](https://github.com/geminik23/ai-agents/blob/main/examples/yaml/orchestration/stock_analysis_concurrent.yaml) |
| Group chat | Iterate through discussion or review | [Code review](https://github.com/geminik23/ai-agents/blob/main/examples/yaml/orchestration/code_review_group_chat.yaml) |
| Handoff | Transfer work between specialists during a turn | [Support handoff](https://github.com/geminik23/ai-agents/blob/main/examples/yaml/orchestration/support_handoff.yaml) |

Router behavior composes delegate states and transitions. Pipeline, concurrent, group-chat, and handoff blocks provide dedicated handlers. Dynamic orchestration tools are also available when explicitly enabled. These are defined coordination patterns; choose one before designing a more elaborate topology.

## Start with a support router

The [delegated support recipe](@/examples/recipe-delegate-support.md) creates child agents at startup and routes requests to the appropriate specialist. A delegate state forwards the message to a registered child while the parent's transition rules continue to govern the conversation.

```text
User → parent triage → billing or technical child → parent response
```

Keep the referenced child YAML files when copying the example. During this recipe's spawner setup, the builder constructs each declared child and checks that orchestration state references resolve in the registry. A missing child file, an invalid child definition, or an unresolved state reference fails construction.

## Make pipeline inputs explicit

In the repository's content pipeline, the writer produces a draft, the reviewer comments on it, and the editor receives both. The editor's input uses these runtime template variables:

```jinja2
Original draft:
{{ stages.writer }}

Reviewer feedback:
{{ stages.reviewer }}

Original request:
{{ original_input }}
```

`previous_output` alone would give the editor only the reviewer's response. Named stage outputs let you preserve the draft as well. This template belongs inside a pipeline stage's `input`; it is not a standalone agent definition.

## Decide what children receive

The default context mode forwards the current input. Use `summary` or `full` when the task requires parent conversation history, accounting for the extra context and any summarization call. Actor identity is forwarded structurally so actor-scoped memory can keep the original actor's identity.

For concurrent work, configure aggregation and partial-failure behavior. For iterative discussions, choose turn order and iteration limits. For pipelines, set an appropriate total timeout. A child failure policy should be part of the design before testing the successful path.

## Inspect the combined result

Structured orchestration results are exposed in `context.orchestration` and response metadata. Use them to inspect the selected agents, stage outputs, discussion, or handoff chain, then assert the important parts in an evaluation.

Parent orchestration produces a final combined response. Do not design a UI around an assumption that every child token will stream through the parent. Refer to the supported [streaming API](@/docs/rust-api.md#streaming) when deciding how to render progress.

Continue with the [orchestration field reference](@/docs/yaml-reference.md#orchestration-states), [runtime concepts](@/docs/concepts.md#multi-agent-orchestration), and [evaluation guide](@/docs/test-and-observe.md).

+++
title = "Build Agent Behavior"
weight = 20
template = "docs.html"
description = "Choose states, skills, context, and processing steps for your agent's job."

[extra]
learning_guide = true
+++

An agent can begin with a system prompt and a model. Add structure when different requests need different instructions, tools, or sequences of work. The same YAML can run in the CLI or through `AgentBuilder` in a Rust application.

## Choose the right building block

| You need to… | Start with… | What it controls |
| --- | --- | --- |
| Keep a conversation in a named phase | States | The current prompt, available tools, and transitions |
| Reuse a sequence for a recognized request | Skills | Trigger routing and reusable steps |
| Supply application data to a prompt | Context | Values from runtime, environment, and declared sources |
| Transform or validate input and output | Process pipeline | Ordered processing stages around generation |
| Choose a model for a supporting operation | LLM role routing | Which configured provider alias handles that operation |

Keep the first version small enough to test. A state is useful when the current phase matters across turns; a skill is useful when a request should invoke a repeatable procedure. They can work together.

## Separate a conversation into states

The [support state example](@/examples/recipe-support-states.md) starts in `greeting`, routes to a support category, and can escalate from any state. Its technical branch has nested states for gathering information and proposing a solution.

```text
User message → greeting → technical / order / product support → closing
                        → global escalation
                        → clarification after repeated routing misses
```

Start by changing the prompts and transition descriptions in that example. Natural-language `when` conditions use model evaluation; use the reference's guard facilities for deterministic application conditions. A transition declaration does not make a provider's judgment deterministic, so test representative requests and ambiguous cases.

A state's tool list narrows the agent's declared tool access. A state does not grant tools that the agent never allowed. The support example deliberately has no tools so you can study routing on its own.

## Reuse work as a skill

The repository's [external skill example](https://github.com/geminik23/ai-agents/blob/main/examples/yaml/skills/skill_external_only.yaml) loads a math procedure from a separate file. The agent excerpt below shows the two declarations that must stay together:

```yaml
# Fragment: the referenced skill file is supplied beside the repository example.
skills:
  - file: skills/math_helper.skill.yaml

# The skill uses calculator, so the agent must allow it explicitly.
tools:
  - name: calculator
```

The skill router selects a matching trigger, executes its steps, and returns a response. With no matching skill, the normal conversation path remains available. Keep external skill files beside the agent when copying this example; downloading only the parent YAML is insufficient.

## Bring in application context

Use context for values your host knows, such as a customer ID or an account tier. Template references read those values; they do not fetch missing data automatically. Decide whether a value comes from the host, an environment variable, a declared source, or a default. Use required-key validation when proceeding without it would be incorrect.

Process stages are useful when input or output needs a predictable transformation or validation step. Configure these stages around the model interaction, then assert both the visible result and the relevant context in an evaluation.

## Choose supporting models deliberately

A role such as skill routing or state transition evaluation can use a different named model from the main conversation. Explore [model selection precedence](@/explore/roles.md) to see how local, role, group, and default assignments resolve. Assigning a role does not enable the associated feature or grant tool access.

## Check the behavior before expanding it

From the repository root, run the deterministic state suites:

```sh
sh examples/eval/mocked/run_mocked_evals.sh --category state-machine
```

Then try the same agent with your chosen provider. Mocked tests verify fixed routing paths; live trials help assess the provider's interpretation of your transition descriptions.

Continue with [execution controls](@/docs/control-execution.md), or look up the complete [state](@/docs/yaml-reference.md#state-machine), [skill](@/docs/yaml-reference.md#skills), [context](@/docs/yaml-reference.md#context), and [process](@/docs/yaml-reference.md#process-pipeline) fields.

+++
title = "Control Execution"
weight = 21
template = "docs.html"
description = "Make tool access explicit, add approval, and inspect what actually executed."

[extra]
learning_guide = true
+++

Tool access is an execution decision. The prompt explains the agent's task, while tool declarations, policy, and human approval determine which operations may proceed. Start with the smallest set of tools needed for the task.

## Declare access before adding approval

Omitting top-level `tools`, or using `tools: []`, grants no ordinary tools. A registered implementation is not automatically available to every agent. Runtime and state scopes can narrow the allowed set further.

The [approval example](@/examples/recipe-approve-tools.md) allows `http` and asks for approval before each request. This excerpt belongs inside its agent definition:

```yaml
# Permit HTTP requests and require a human decision before execution.
tools:
  - http

hitl:
  default_timeout_seconds: 120 # Seconds to wait; the default is 300.
  on_timeout: reject          # Reject unanswered requests; also the default.
  tools:
    http:
      require_approval: true  # Pause before this tool; the default is false.
      approval_context: [method, url]
      approval_message: "Approve {{ method }} request to {{ url }}?"
```

Run the complete example from its recipe. With the CLI's interactive approval handler, try approving one request and rejecting another. Approved requests make real HTTP calls; the mocked evaluation on the recipe replaces both the model and tool for a deterministic test.

## Understand the execution boundary

```text
Tool request → scope and policy checks → human approval when required
             → approved arguments checked again → resource lock
             → final admission → implementation → execution evidence
```

Approval is bound to the reviewed operation. Modified arguments are checked again, and changed state or policy can prevent a call before invocation. Resource locking protects operations that cannot safely run together. These checks matter even when a model repeats a previously approved request.

A tool also needs its host implementation. Tools such as `command`, `diagnostics`, and `web_search` require explicit host support; registering their names in YAML cannot supply that support. The [built-in tool reference](@/docs/built-in-tools.md) describes each tool's requirements.

## Treat mutation as an explicit capability

Filesystem mutation tools execute when `dry_run` is omitted. Set `dry_run: true` to request a supported preview or validation path. Actual mutation still requires applicable write policy and approval or an explicit trusted-policy exemption.

For a file-editing application, define the permitted workspace and paths, choose which changes need approval, and test both allowed and denied operations. A successful model response is not evidence that a write occurred. Inspect the tool result's mutation fields and execution record.

These are framework execution controls. Applications remain responsible for their process permissions and deployment isolation.

## Verify denial as well as success

The [test and observe guide](@/docs/test-and-observe.md) shows how to read execution evidence. In a suite, pair approval assertions with `tool_called.executed` to distinguish an accepted request from an implementation that actually ran. Test rejection and timeout paths, including the visible response after cancellation.

For provider failures, configure [error recovery](@/docs/yaml-reference.md#error-recovery) intentionally. Model-role inheritance selects a configured alias; it is not a retry or provider-failure fallback strategy.

Continue with [tool security](@/docs/yaml-reference.md#tool-security), [HITL](@/docs/yaml-reference.md#hitl-human-in-the-loop), and the [Rust host integration guide](@/docs/rust-api.md#custom-tools).

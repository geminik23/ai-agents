# AI Agents Framework

**One YAML = Any Agent.**

Build AI agents from a single YAML spec:
- Declarative behavior
- Language-agnostic semantics: intent/extraction/validation via LLM
- Layered overrides: global -> agent -> state -> skill
- Safety by default: tool policies, HITL approvals, error recovery
- Extensible: custom LLMs, tools, memory, storage, hooks

> Status: **1.0.0-rc.4**
> This project is under active development. APIs and YAML schema may change between minor versions.
> Use it for experiments and feedback; hold off on production until v1.0.0.
>
> As features are added, this repository will keep adding examples and documentation.

## Near-term focus

Planned next (subject to change):
- Workspace refactor (split into crates) — completed
- Reasoning & reflection foundations (toward plan-and-execute)
- Intent disambiguation (clarification-first UX)
- Add Examples and documentation 
- More Built-in tools

## Install

```toml
[dependencies]
ai-agents = "1.0.0-rc.3"
```

## License

Licensed under either of

- Apache License, Version 2.0 (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license (LICENSE-MIT)

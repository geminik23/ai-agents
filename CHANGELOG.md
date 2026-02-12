# Changelog

## 1.0.0-rc.5

### Added
- First release of the rewritten framework (previously published as 0.x)
- Intent disambiguation: LLM-based ambiguity detection, clarification generation, multi-turn resolution
- State/skill-level disambiguation overrides with configurable thresholds

## 1.0.0-rc.4

### Added
- Reasoning modes: none, chain-of-thought, ReAct, plan-and-execute, auto (LLM selects)
- Reflection: LLM self-evaluation with criteria, retry on failure, configurable thresholds
- Per-state and per-skill reasoning/reflection overrides

## 1.0.0-rc.3

### Changed
- Split monolithic crate into workspace architecture (17 crates)

## 1.0.0-rc.2

### Added
- Tool provider system and aliases for extensible tool support

## 1.0.0-rc.1

### Added
- CompactingMemory with auto-summarization and token budgeting
- Storage backends (SQLite, Redis) with YAML integration

## Pre-RC (initial development)

### Added
- YAML-defined agents with system prompt, tools, skills, and behavior in one file
- Multi-LLM support with aliases (default, router, evaluator) and auto-fallback
- Skill system with LLM-based intent routing and multi-step execution
- State machine with hierarchical states, LLM-evaluated transitions, guards, entry/exit actions
- Dynamic context injection from runtime, file, HTTP, env, and callback sources
- Template rendering with Jinja2-compatible syntax (minijinja)
- Built-in tools: calculator, datetime, JSON, random, HTTP, file, text, template, math
- Conditional tool availability: context, state, time, semantic, and composite conditions
- Streaming with real-time token streaming and tool/state events
- Parallel tool execution with configurable concurrency
- Agent hooks for lifecycle events (message, LLM, tool, state, error, response)
- Human-in-the-loop: tool, condition, and state approval with multi-language localization
- Error recovery: retry with backoff, LLM fallback, context overflow handling
- Tool security: rate limiting, domain/path restrictions, confirmation requirements
- Input/output process pipeline: normalize, detect, extract, sanitize, validate, transform, format

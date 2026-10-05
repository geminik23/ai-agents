+++
title = "Which model runs this role?"
description = "Explore how local aliases, role settings, and defaults select an auxiliary model."
template = "roles.html"
weight = 1
+++

Hierarchical routing lets auxiliary work use different models while the main response keeps its own model. A mapping under `llm.router` opts into this selection order:

**Local alias → role leaf → group default → router default → main alias.**

The first configured alias wins. Omitted values inherit. An invalid explicit alias is a configuration error; it does not fall through to the next default. This is configuration selection, not model-call retry or fallback.

This explorer uses the `process.detect` role. Its local override lives in the input detection stage's `config.llm` field. All aliases shown here are declared model names in the generated configuration.

Read the [YAML routing reference](@/docs/yaml-reference.md#hierarchical-auxiliary-routing-1-1) and [Rust migration guide](@/docs/rust-api.md#auxiliary-routing-migration-1-1), or run the [hierarchical routing example](@/examples/recipe-model-roles.md).

+++
title = "Documentation"
description = "Learn to build AI agents with YAML and Rust. Find getting-started guides, configuration and API references, tool policies, and evaluation workflows."
sort_by = "weight"
template = "docs-index.html"
transparent = false

[extra]
seo_title = "Rust AI Agent Framework Documentation"
[[extra.starts]]
title = "Run with the CLI"
label = "YAML first"
description = "Install the CLI, configure a provider, and start a conversation."
path = "docs/getting-started.md"
anchor = "your-first-agent-cli"
[[extra.starts]]
title = "Embed in Rust"
label = "Your application"
description = "Load the same YAML in a Rust host and extend the runtime with traits."
path = "docs/getting-started.md"
anchor = "your-first-agent-rust"
[[extra.starts]]
title = "Use a local model"
label = "Ollama"
description = "Run a downloaded model locally without a hosted provider API key."
path = "docs/getting-started.md"
anchor = "start-with-a-local-model"
[[extra.starts]]
title = "Run your first evaluation"
label = "No API key"
description = "Exercise the runtime with fixed model responses and inspect the reports."
path = "docs/getting-started.md"
anchor = "start-with-a-mocked-evaluation"
[[extra.starts]]
title = "Build, host and evaluate a task"
label = "Development checkout"
description = "Connect YAML objectives, actual domain evidence, typed human decisions and the same evaluation host."
path = "docs/task-autonomy.md"
anchor = "from-an-objective-to-verified-work"
+++

Start with the smallest agent your task needs. Add behavior, execution controls, memory, and collaboration as the application grows. The guides below connect each decision to runnable examples and the full reference.

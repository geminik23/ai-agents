# Rust MCP integration

Run from the repository root:

```sh
cargo run -p xtask --features mcp-integration --locked -- mcp-integration
```

This explicit command starts a separate Rust MCP server using the registry `rmcp` SDK, then loads the agent and suite YAML through the existing EvalRunner. Only model responses are mocked. MCP discovery, transports, tools, state changes and approval handling are real.

The `agent.yaml` file is a test template: the driver fills in the actual executable path or bound loopback URL before the normal YAML parser loads it. The server owns an isolated counter and a separate append-only event file. Runtime assertions and the event file must both agree; model text alone cannot pass the test.

Both stdio and Streamable HTTP verify:

- Actual discovery and accumulation through a permitted view
- Invalid arguments and a remote tool error
- Blocked functions and functions outside a view, without server effects
- Rejected and approved mutations through the existing runtime HITL configuration
- Exact server-side effects, orderly closure acknowledgement and process exit

The HTTP server binds an ephemeral loopback port and derives its URL from that address. The driver owns HTTP shutdown through an stdin control pipe; the MCP client owns the stdio child. A test deadline is a failure, not evidence of cleanup. The driver kills and reaps its own HTTP child on failure.

The fixture suite allows ten seconds per turn; startup and shutdown observations have a thirty-second test deadline. These bounds belong to this integration check and do not change ordinary agent limits.

Artifacts are retained under `target/mcp-integration/<unique-run>/`: generated YAML, redacted eval reports, server events, process readiness and closure receipts. Inputs and state are synthetic. No provider key or external MCP service is required.

The feature gates test-server dependencies in the non-publishable `xtask`; ordinary framework builds and default tests do not execute this command. Native platform CI runs it as a separate step after the existing tests. Process observation currently supports Linux/macOS, matching the enabled matrix; Windows remains deferred.

The existing Python transport fixtures remain responsible for malformed/unfinished oversized bodies, HTTPS trust and SDK cancellation regressions. This test adds real Rust-server integration; it does not claim compatibility with every SDK implementation or the legacy HTTP+SSE transport.

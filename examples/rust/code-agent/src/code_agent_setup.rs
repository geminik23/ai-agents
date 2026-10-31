// Prepare and run a constrained code-task example with a disposable Rust sample project.
// Rust supplies the actual workspace and command bindings that the YAML cannot grant by itself.
// Flow: prepare sample -> validate bindings -> run a scripted offline agent -> verify actual effects.
// Prerequisites: Linux/WSL, cached Rust dependencies, trusted co-located cargo/rustc, and a new path under /tmp.
// Set CARGO_BIN and RUSTC_BIN to the absolute paths printed by rustup which cargo and rustup which rustc.
// Run these commands from the repository root, using a new sample path for each run:
// cargo run --manifest-path examples/Cargo.toml -p example-code-agent --bin code-agent-setup --offline -- prepare /tmp/code-agent-sample
// cargo run --manifest-path examples/Cargo.toml -p example-code-agent --bin code-agent-setup --offline -- check-config /tmp/code-agent-sample "$CARGO_BIN" "$RUSTC_BIN"
// cargo run --manifest-path examples/Cargo.toml -p example-code-agent --bin code-agent-setup --offline -- run-offline /tmp/code-agent-sample "$CARGO_BIN" "$RUSTC_BIN"
// Configuration checks never construct a provider; the offline example uses no live model or service.
// The sample verifier accepts only its reviewed sources; it is not an OS sandbox.
// For a real workspace, set absolute WORKSPACE and CHECKER paths and a workspace-relative SOURCE path.
// cargo run --manifest-path examples/Cargo.toml -p example-code-agent --bin code-agent-setup --offline -- check-workspace "$WORKSPACE" "$SOURCE" "$CHECKER" openai gpt-5.4-mini
// cargo run --manifest-path examples/Cargo.toml -p example-code-agent --bin code-agent-setup --offline -- run-workspace --trust-checker --share-source "$WORKSPACE" "$SOURCE" "$CHECKER" openai gpt-5.4-mini
// Optionally supply a workspace-relative instruction file and --max-output-tokens N or --checker-timeout-ms N.
// Output tokens default to 2000 and checker timeout to 30000 ms; other applicable limits still apply.
// Use /task run in the interactive session.
// check-workspace does not contact a provider; run-workspace requires explicit trust and code-sharing consent.
// The supplied checker runs with host permissions and must provide its own independent tests and isolation.

mod code_agent_run;
mod code_agent_workspace;
mod workspace_agent;

use std::path::Path;

// CLI paths are host input; the instruction snapshot and output limit are explicitly selected.
fn workspace_setup(
    args: &[String],
) -> Result<workspace_agent::WorkspaceSetup, Box<dyn std::error::Error>> {
    let [root, source, checker, provider, model, rest @ ..] = args else {
        return Err("expected WORKSPACE SOURCE CHECKER PROVIDER MODEL [INSTRUCTIONS] [--max-output-tokens N] [--checker-timeout-ms N]".into());
    };
    let mut instructions = None;
    let mut max_output_tokens = None;
    let mut checker_timeout_ms = None;
    let mut positional_only = false;
    let mut remaining = rest.iter();
    while let Some(arg) = remaining.next() {
        if !positional_only && arg == "--" {
            positional_only = true;
        } else if !positional_only && arg == "--max-output-tokens" {
            if max_output_tokens.is_some() {
                return Err("--max-output-tokens may be supplied only once".into());
            }
            let value = remaining
                .next()
                .ok_or("--max-output-tokens needs a value")?;
            let value: u32 = value
                .parse()
                .map_err(|_| "invalid --max-output-tokens value")?;
            if value == 0 {
                return Err("--max-output-tokens must be positive".into());
            }
            max_output_tokens = Some(value);
        } else if !positional_only && arg == "--checker-timeout-ms" {
            if checker_timeout_ms.is_some() {
                return Err("--checker-timeout-ms may be supplied only once".into());
            }
            let value = remaining
                .next()
                .ok_or("--checker-timeout-ms needs a value")?;
            let value: u64 = value
                .parse()
                .map_err(|_| "invalid --checker-timeout-ms value")?;
            if value == 0 || value > ai_agents::MAX_TOOL_TIMEOUT_MS {
                return Err(
                    "--checker-timeout-ms must be positive and within the tool timeout limit"
                        .into(),
                );
            }
            checker_timeout_ms = Some(value);
        } else if instructions.is_some() {
            return Err("unexpected workspace argument".into());
        } else {
            instructions = Some(Path::new(arg));
        }
    }
    let mut setup = workspace_agent::prepare(
        Path::new(root),
        Path::new(source),
        Path::new(checker),
        provider,
        model,
        instructions,
    )?;
    if let Some(tokens) = max_output_tokens {
        setup.set_max_output_tokens(tokens)?;
    }
    if let Some(timeout) = checker_timeout_ms {
        setup.set_checker_timeout_ms(timeout)?;
    }
    Ok(setup)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [action, rest @ ..] if action == "check-workspace" => {
            let setup = workspace_setup(rest)?;
            let _builder = setup
                .builder()
                .auto_configure_features()?
                .autonomy_extensions(setup.extensions()?);
            println!(
                "Workspace configuration validated without constructing a provider or running the checker."
            );
            println!("Maximum model output tokens: {}", setup.max_output_tokens());
            println!("Checker timeout: {} ms", setup.checker_timeout_ms());
            println!(
                "This does not certify checker independence, isolation, credentials or task completion."
            );
        }
        [action, trust, share, rest @ ..]
            if action == "run-workspace"
                && trust == "--trust-checker"
                && share == "--share-source" =>
        {
            let setup = workspace_setup(rest)?;
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?
                .block_on(workspace_agent::run(setup))?;
        }
        [action, root] if action == "prepare" => {
            let root = code_agent_workspace::prepare(Path::new(root))?;
            println!("Prepared disposable sample project at {}", root.display());
        }
        [action, root, cargo, rustc] if action == "check-config" => {
            let setup = code_agent_workspace::prepare_configuration(
                Path::new(root),
                &std::env::current_exe()?,
                Path::new(cargo),
                Path::new(rustc),
            )?;
            // Feature setup installs the normal built-ins but does not configure or call any provider.
            let _builder = setup.builder().auto_configure_features()?;
            println!("Configuration validated without constructing a provider or starting a task.");
            println!("The command runner is not installed; actual agent execution is not enabled.");
        }
        [action, root, cargo, rustc] if action == "run-offline" => {
            let setup = code_agent_workspace::prepare_configuration(
                Path::new(root),
                &std::env::current_exe()?,
                Path::new(cargo),
                Path::new(rustc),
            )?;
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?
                .block_on(code_agent_run::run(&setup))?;
        }
        [action, root, cargo, rustc] if action == "verify" => {
            let setup = code_agent_workspace::prepare_configuration(
                Path::new(root),
                &std::env::current_exe()?,
                Path::new(cargo),
                Path::new(rustc),
            )?;
            let result = setup.verify_fixture()?;
            println!("Acceptance evidence stored at {}", result.scratch.display());
            if !result.passed {
                return Err(
                    "sample-project acceptance did not pass; inspect the isolated evidence".into(),
                );
            }
            println!(
                "All required sample-project acceptance cases passed for the unchanged input."
            );
        }
        _ => {
            return Err("usage: code-agent-setup prepare NEW_ABSOLUTE_PATH | check-config WORKSPACE CARGO RUSTC | verify WORKSPACE CARGO RUSTC | run-offline WORKSPACE CARGO RUSTC | check-workspace WORKSPACE SOURCE CHECKER PROVIDER MODEL [INSTRUCTIONS] [--max-output-tokens N] [--checker-timeout-ms N] | run-workspace --trust-checker --share-source WORKSPACE SOURCE CHECKER PROVIDER MODEL [INSTRUCTIONS] [--max-output-tokens N] [--checker-timeout-ms N]".into());
        }
    }
    Ok(())
}

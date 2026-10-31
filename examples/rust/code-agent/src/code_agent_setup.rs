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
// The optional final argument is a workspace-relative instruction file; use /task run in the interactive session.
// check-workspace does not contact a provider; run-workspace requires explicit trust and code-sharing consent.
// The supplied checker runs with host permissions and must provide its own independent tests and isolation.

mod code_agent_run;
mod code_agent_workspace;
mod workspace_agent;

use std::path::Path;

// CLI paths are host input; the optional instruction file is relative to the selected workspace.
fn workspace_setup(
    args: &[String],
) -> Result<workspace_agent::WorkspaceSetup, Box<dyn std::error::Error>> {
    match args {
        [root, source, checker, provider, model] => workspace_agent::prepare(
            Path::new(root),
            Path::new(source),
            Path::new(checker),
            provider,
            model,
            None,
        ),
        [root, source, checker, provider, model, instructions] => workspace_agent::prepare(
            Path::new(root),
            Path::new(source),
            Path::new(checker),
            provider,
            model,
            Some(Path::new(instructions)),
        ),
        _ => Err("expected WORKSPACE SOURCE CHECKER PROVIDER MODEL [INSTRUCTIONS]".into()),
    }
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
            return Err("usage: code-agent-setup prepare NEW_ABSOLUTE_PATH | check-config WORKSPACE CARGO RUSTC | verify WORKSPACE CARGO RUSTC | run-offline WORKSPACE CARGO RUSTC | check-workspace WORKSPACE SOURCE CHECKER PROVIDER MODEL [INSTRUCTIONS] | run-workspace --trust-checker --share-source WORKSPACE SOURCE CHECKER PROVIDER MODEL [INSTRUCTIONS]".into());
        }
    }
    Ok(())
}

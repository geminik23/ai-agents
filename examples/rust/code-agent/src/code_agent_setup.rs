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
// For a real workspace, set absolute WORKSPACE and CHECKER paths and explicitly select workspace-relative source paths.
// cargo run --manifest-path examples/Cargo.toml -p example-code-agent --bin code-agent-setup --offline -- check-workspace "$WORKSPACE" "$SOURCE" "$CHECKER" openai gpt-5.4-mini
// cargo run --manifest-path examples/Cargo.toml -p example-code-agent --bin code-agent-setup --offline -- run-workspace --trust-checker --share-source "$WORKSPACE" "$SOURCE" "$CHECKER" openai gpt-5.4-mini
// Optionally repeat --file RELATIVE_PATH or --new-file RELATIVE_PATH and configure discovery or limits.
// --auto-instructions separately authorizes sharing root and selected-path ancestor AGENTS.md contents with the model.
// --assume-exclusive-workspace is valid only when the caller controls all relevant writers and immutable checker inputs.
// Output tokens default to 2000 and checker timeout to 30000 ms; other applicable limits still apply.
// Use /task run in the interactive session.
// check-workspace does not contact a provider; run-workspace requires explicit trust and code-sharing consent.
// The supplied checker runs with host permissions and must provide its own independent tests and isolation.
// To let the agent find and change files anywhere under the root, protect every checker input instead of selecting sources:
// cargo run --manifest-path examples/Cargo.toml -p example-code-agent --bin code-agent-setup --offline -- check-root "$WORKSPACE" "$CHECKER" openai gpt-5.4-mini --protect "$PROTECTED"
// cargo run --manifest-path examples/Cargo.toml -p example-code-agent --bin code-agent-setup --offline -- run-root --trust-checker --share-workspace "$WORKSPACE" "$CHECKER" openai gpt-5.4-mini --protect "$PROTECTED"
// Repeat --protect RELATIVE_PATH for every file or directory the checker relies on; root mode always shares discovery under the root.
// For questions that do not require edits, omit the checker and explicitly select the read-only task:
// cargo run --manifest-path examples/Cargo.toml -p example-code-agent --bin code-agent-setup --offline -- check-read-only "$WORKSPACE" "$SOURCE" openai gpt-5.4-mini
// cargo run --manifest-path examples/Cargo.toml -p example-code-agent --bin code-agent-setup --offline -- run-read-only --share-source "$WORKSPACE" "$SOURCE" openai gpt-5.4-mini

mod code_agent_run;
mod code_agent_workspace;
mod workspace_agent;
mod workspace_report;

use std::path::{Path, PathBuf};

// Shared by both workspace modes so a limit is validated the same way wherever it is accepted.
fn parse_output_tokens(value: Option<&String>) -> Result<u32, Box<dyn std::error::Error>> {
    let value: u32 = value
        .ok_or("--max-output-tokens needs a value")?
        .parse()
        .map_err(|_| "invalid --max-output-tokens value")?;
    if value == 0 {
        return Err("--max-output-tokens must be positive".into());
    }
    Ok(value)
}

fn parse_checker_timeout(value: Option<&String>) -> Result<u64, Box<dyn std::error::Error>> {
    let value: u64 = value
        .ok_or("--checker-timeout-ms needs a value")?
        .parse()
        .map_err(|_| "invalid --checker-timeout-ms value")?;
    if value == 0 || value > ai_agents::MAX_TOOL_TIMEOUT_MS {
        return Err(
            "--checker-timeout-ms must be positive and within the tool timeout limit".into(),
        );
    }
    Ok(value)
}

// Root mode takes protected checker inputs instead of selected sources; discovery and new files follow from the root grant.
fn root_setup(
    args: &[String],
) -> Result<workspace_agent::WorkspaceSetup, Box<dyn std::error::Error>> {
    let [root, checker, provider, model, rest @ ..] = args else {
        return Err("expected WORKSPACE CHECKER PROVIDER MODEL --protect PATH [--protect PATH] [INSTRUCTIONS] [--assume-exclusive-workspace] [--max-output-tokens N] [--checker-timeout-ms N]".into());
    };
    let mut instructions = None;
    let mut protected = Vec::new();
    let mut assume_exclusive_workspace = false;
    let mut max_output_tokens = None;
    let mut checker_timeout_ms = None;
    let mut positional_only = false;
    let mut remaining = rest.iter();
    while let Some(arg) = remaining.next() {
        if !positional_only && arg == "--" {
            positional_only = true;
        } else if !positional_only && arg == "--protect" {
            protected.push(PathBuf::from(
                remaining
                    .next()
                    .ok_or("--protect needs a workspace-relative path")?,
            ));
        } else if !positional_only && arg == "--assume-exclusive-workspace" {
            if assume_exclusive_workspace {
                return Err("--assume-exclusive-workspace may be supplied only once".into());
            }
            assume_exclusive_workspace = true;
        } else if !positional_only && arg == "--max-output-tokens" {
            if max_output_tokens.is_some() {
                return Err("--max-output-tokens may be supplied only once".into());
            }
            max_output_tokens = Some(parse_output_tokens(remaining.next())?);
        } else if !positional_only && arg == "--checker-timeout-ms" {
            if checker_timeout_ms.is_some() {
                return Err("--checker-timeout-ms may be supplied only once".into());
            }
            checker_timeout_ms = Some(parse_checker_timeout(remaining.next())?);
        } else if !positional_only
            && (arg == "--auto-instructions"
                || arg == "--allow-discovery"
                || arg == "--file"
                || arg == "--new-file")
        {
            return Err(format!("{arg} is not available in root mode").into());
        } else if instructions.is_some() {
            return Err("unexpected workspace argument".into());
        } else {
            instructions = Some(Path::new(arg));
        }
    }
    let mut setup = workspace_agent::prepare_root(
        Path::new(root),
        &protected,
        Path::new(checker),
        provider,
        model,
        instructions,
    )?;
    if assume_exclusive_workspace {
        setup.assume_exclusive_workspace()?;
    }
    if let Some(tokens) = max_output_tokens {
        setup.set_max_output_tokens(tokens)?;
    }
    if let Some(timeout) = checker_timeout_ms {
        setup.set_checker_timeout_ms(timeout)?;
    }
    Ok(setup)
}

// CLI paths are host input; the instruction snapshot and output limit are explicitly selected.
fn workspace_setup(
    args: &[String],
    read_only: bool,
) -> Result<workspace_agent::WorkspaceSetup, Box<dyn std::error::Error>> {
    let (root, source, checker, provider, model, rest) = if read_only {
        let [root, source, provider, model, rest @ ..] = args else {
            return Err("expected WORKSPACE SOURCE PROVIDER MODEL [INSTRUCTIONS | --auto-instructions] [--file PATH] [--allow-discovery] [--max-output-tokens N]".into());
        };
        (root, source, None, provider, model, rest)
    } else {
        let [root, source, checker, provider, model, rest @ ..] = args else {
            return Err("expected WORKSPACE SOURCE CHECKER PROVIDER MODEL [INSTRUCTIONS | --auto-instructions] [--file PATH] [--new-file PATH] [--assume-exclusive-workspace] [--allow-discovery] [--max-output-tokens N] [--checker-timeout-ms N]".into());
        };
        (root, source, Some(checker), provider, model, rest)
    };
    let mut instructions = None;
    let mut sources = vec![PathBuf::from(source)];
    let mut new_files = Vec::new();
    let mut allow_discovery = false;
    let mut automatic_instructions = false;
    let mut assume_exclusive_workspace = false;
    let mut max_output_tokens = None;
    let mut checker_timeout_ms = None;
    let mut positional_only = false;
    let mut remaining = rest.iter();
    while let Some(arg) = remaining.next() {
        if !positional_only && arg == "--" {
            positional_only = true;
        } else if !positional_only && arg == "--allow-discovery" {
            if allow_discovery {
                return Err("--allow-discovery may be supplied only once".into());
            }
            allow_discovery = true;
        } else if !positional_only && arg == "--auto-instructions" {
            if automatic_instructions {
                return Err("--auto-instructions may be supplied only once".into());
            }
            automatic_instructions = true;
        } else if !positional_only && arg == "--assume-exclusive-workspace" {
            if read_only || assume_exclusive_workspace {
                return Err("--assume-exclusive-workspace requires an edit task and may be supplied only once".into());
            }
            assume_exclusive_workspace = true;
        } else if !positional_only && arg == "--file" {
            sources.push(PathBuf::from(
                remaining
                    .next()
                    .ok_or("--file needs a workspace-relative path")?,
            ));
        } else if !positional_only && arg == "--new-file" {
            if read_only {
                return Err("read-only mode cannot authorize new files".into());
            }
            new_files.push(PathBuf::from(
                remaining
                    .next()
                    .ok_or("--new-file needs a workspace-relative path")?,
            ));
        } else if !positional_only && arg == "--max-output-tokens" {
            if max_output_tokens.is_some() {
                return Err("--max-output-tokens may be supplied only once".into());
            }
            max_output_tokens = Some(parse_output_tokens(remaining.next())?);
        } else if !positional_only && arg == "--checker-timeout-ms" {
            if checker_timeout_ms.is_some() {
                return Err("--checker-timeout-ms may be supplied only once".into());
            }
            checker_timeout_ms = Some(parse_checker_timeout(remaining.next())?);
        } else if instructions.is_some() {
            return Err("unexpected workspace argument".into());
        } else {
            instructions = Some(Path::new(arg));
        }
    }
    let mut setup = match checker {
        None => workspace_agent::prepare_read_only(
            Path::new(root),
            &sources,
            provider,
            model,
            instructions,
        )?,
        Some(checker) if !new_files.is_empty() => workspace_agent::prepare_files_with_new(
            Path::new(root),
            &sources,
            &new_files,
            Path::new(checker),
            provider,
            model,
            instructions,
        )?,
        Some(checker) if sources.len() == 1 => workspace_agent::prepare(
            Path::new(root),
            Path::new(source),
            Path::new(checker),
            provider,
            model,
            instructions,
        )?,
        Some(checker) => workspace_agent::prepare_files(
            Path::new(root),
            &sources,
            Path::new(checker),
            provider,
            model,
            instructions,
        )?,
    };
    if allow_discovery {
        setup.enable_discovery()?;
    }
    if automatic_instructions {
        setup.enable_automatic_instructions()?;
    }
    if assume_exclusive_workspace {
        setup.assume_exclusive_workspace()?;
    }

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
            let setup = workspace_setup(rest, false)?;
            let _builder = setup
                .builder()?
                .auto_configure_features()?
                .autonomy_extensions(setup.extensions()?);
            println!(
                "Workspace configuration validated without constructing a provider or running the checker."
            );
            println!("Selected source files: {}", setup.source_files().len());
            println!("Workspace discovery enabled: {}", setup.discovery_enabled());
            println!(
                "Automatic scoped instructions enabled: {}",
                setup.automatic_instructions_enabled()
            );
            println!(
                "Cooperative revision assumed: {}",
                setup.cooperative_revision_enabled()
            );
            println!("Maximum model output tokens: {}", setup.max_output_tokens());
            if let Some(timeout) = setup.checker_timeout_ms() {
                println!("Checker timeout: {timeout} ms");
            }
            println!(
                "This does not certify checker independence, isolation, credentials or task completion."
            );
        }
        [action, trust, share, rest @ ..]
            if action == "run-workspace"
                && trust == "--trust-checker"
                && share == "--share-source" =>
        {
            let setup = workspace_setup(rest, false)?;
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?
                .block_on(workspace_agent::run(setup))?;
        }
        [action, rest @ ..] if action == "check-root" => {
            let setup = root_setup(rest)?;
            let _builder = setup
                .builder()?
                .auto_configure_features()?
                .autonomy_extensions(setup.extensions()?);
            println!(
                "Root workspace configuration validated without constructing a provider or running the checker."
            );
            println!(
                "Protected checker inputs: {}",
                setup.protected_paths().len()
            );
            println!(
                "Cooperative revision assumed: {}",
                setup.cooperative_revision_enabled()
            );
            println!("Maximum model output tokens: {}", setup.max_output_tokens());
            if let Some(timeout) = setup.checker_timeout_ms() {
                println!("Checker timeout: {timeout} ms");
            }
            println!(
                "This does not certify checker independence, isolation, credentials or task completion."
            );
        }
        [action, trust, share, rest @ ..]
            if action == "run-root"
                && trust == "--trust-checker"
                && share == "--share-workspace" =>
        {
            let setup = root_setup(rest)?;
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?
                .block_on(workspace_agent::run(setup))?;
        }
        [action, rest @ ..] if action == "check-read-only" => {
            let setup = workspace_setup(rest, true)?;
            let _builder = setup
                .builder()?
                .auto_configure_features()?
                .autonomy_extensions(setup.extensions()?);
            println!(
                "Read-only configuration validated without constructing a provider or installing a checker."
            );
            println!("Selected source files: {}", setup.source_files().len());
            println!("Workspace discovery enabled: {}", setup.discovery_enabled());
            println!(
                "Automatic scoped instructions enabled: {}",
                setup.automatic_instructions_enabled()
            );
            println!("Maximum model output tokens: {}", setup.max_output_tokens());
        }
        [action, share, rest @ ..] if action == "run-read-only" && share == "--share-source" => {
            let setup = workspace_setup(rest, true)?;
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
            return Err("usage: code-agent-setup prepare NEW_ABSOLUTE_PATH | check-config WORKSPACE CARGO RUSTC | verify WORKSPACE CARGO RUSTC | run-offline WORKSPACE CARGO RUSTC | check-workspace WORKSPACE SOURCE CHECKER PROVIDER MODEL [INSTRUCTIONS | --auto-instructions] [--file PATH] [--new-file PATH] [--assume-exclusive-workspace] [--allow-discovery] [--max-output-tokens N] [--checker-timeout-ms N] | run-workspace --trust-checker --share-source WORKSPACE SOURCE CHECKER PROVIDER MODEL [INSTRUCTIONS | --auto-instructions] [--file PATH] [--new-file PATH] [--assume-exclusive-workspace] [--allow-discovery] [--max-output-tokens N] [--checker-timeout-ms N] | check-root WORKSPACE CHECKER PROVIDER MODEL --protect PATH [--protect PATH] [INSTRUCTIONS] [--assume-exclusive-workspace] [--max-output-tokens N] [--checker-timeout-ms N] | run-root --trust-checker --share-workspace WORKSPACE CHECKER PROVIDER MODEL --protect PATH [--protect PATH] [INSTRUCTIONS] [--assume-exclusive-workspace] [--max-output-tokens N] [--checker-timeout-ms N] | check-read-only WORKSPACE SOURCE PROVIDER MODEL [INSTRUCTIONS | --auto-instructions] [--file PATH] [--allow-discovery] [--max-output-tokens N] | run-read-only --share-source WORKSPACE SOURCE PROVIDER MODEL [INSTRUCTIONS | --auto-instructions] [--file PATH] [--allow-discovery] [--max-output-tokens N]".into());
        }
    }
    Ok(())
}

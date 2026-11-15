// Use YAML task behavior with a Rust-owned synthetic domain and the existing interactive CLI.
// Flow: provider-free check -> explicit live consent -> shared domain -> retained foreground REPL.
// This application adds no cumulative task ceilings; configure any desired limits explicitly.
// Use the separate eval suites for bounded campaigns; never paste private customer data.
use example_custom_llm::task_live;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["--help"] {
        println!("task-domain-live --agent PATH (--dry-config-check | --real-llm)");
        return Ok(());
    }
    let [agent_flag, path, mode] = args.as_slice() else {
        return Err(
            "usage: task-domain-live --agent PATH (--dry-config-check | --real-llm)".into(),
        );
    };
    if agent_flag != "--agent" || !matches!(mode.as_str(), "--dry-config-check" | "--real-llm") {
        return Err("choose exactly one supported mode".into());
    }
    let path = Path::new(path);
    let spec = task_live::load_application(path)?;
    if mode == "--dry-config-check" {
        println!(
            "declarations and domain configuration valid; providers and runtime not constructed"
        );
        return Ok(());
    }
    eprintln!(
        "Live application: provider settings come from YAML; provider calls incur cost. No campaign limits are installed."
    );
    task_live::run_application(spec, path).await?;
    Ok(())
}

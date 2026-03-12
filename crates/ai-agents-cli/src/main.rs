use clap::Parser;

use ai_agents_cli::{cli::Cli, init_tracing, run};

#[tokio::main]
async fn main() {
    if let Err(err) = async_main().await {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

async fn async_main() -> anyhow::Result<()> {
    init_tracing();
    let args = Cli::parse();
    run(args).await
}

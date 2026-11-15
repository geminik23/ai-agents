use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let command = args.next();
    let remaining: Vec<_> = args.collect();

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask must be inside the repository root")
        .to_path_buf();

    match command.as_deref() {
        Some("public-contract") if remaining.is_empty() => run_public_contract(&root),
        Some("release-preflight") if remaining.is_empty() => run_release_preflight(&root),
        #[cfg(feature = "mcp-integration")]
        Some("mcp-integration") if remaining.is_empty() => {
            if let Err(error) = xtask::mcp_integration::run(&root) {
                eprintln!("MCP integration failed: {error:#}");
                std::process::exit(1);
            }
        }
        #[cfg(feature = "mcp-integration")]
        Some("mcp-test-server") if remaining.len() == 2 => {
            if let Err(error) =
                xtask::mcp_integration::serve(&remaining[0], &PathBuf::from(&remaining[1]))
            {
                eprintln!("MCP test server failed: {error:#}");
                std::process::exit(1);
            }
        }
        _ => usage(),
    }
}

fn run_public_contract(root: &std::path::Path) {
    match xtask::run_public_contract(root) {
        Ok(summary) => {
            if let Err(error) = xtask::write_summary(root, &summary) {
                eprintln!("public-contract checker error: {error:#}");
                std::process::exit(2);
            }
            println!(
                "public-contract: {} ({} drift item(s)); summary: target/public-contract/summary.json",
                summary.status,
                summary.drift.len()
            );
            if summary.status == "drift" {
                std::process::exit(1);
            }
        }
        Err(error) => {
            let summary = xtask::error_summary(error.to_string());
            let _ = xtask::write_summary(root, &summary);
            eprintln!("public-contract checker error: {error:#}");
            std::process::exit(2);
        }
    }
}

fn run_release_preflight(root: &std::path::Path) {
    match xtask::release::run_release_preflight(root) {
        Ok(xtask::release::ReleasePreflightOutcome::Passed { artifact_dir }) => {
            println!(
                "release-preflight: ok; record: {}/release-record.json",
                artifact_dir.display()
            );
        }
        Ok(xtask::release::ReleasePreflightOutcome::Rejected {
            artifact_dir,
            reasons,
        }) => {
            eprintln!("release-preflight: rejected");
            for reason in reasons {
                eprintln!("- {reason}");
            }
            eprintln!("metadata: {}/cargo-metadata.json", artifact_dir.display());
            std::process::exit(1);
        }
        Err(error) => {
            eprintln!("release-preflight tooling failure: {error:#}");
            std::process::exit(2);
        }
    }
}

fn usage() -> ! {
    eprintln!("usage: cargo run -p xtask -- <public-contract|release-preflight>");
    eprintln!("MCP integration: cargo run -p xtask --features mcp-integration -- mcp-integration");
    std::process::exit(2);
}

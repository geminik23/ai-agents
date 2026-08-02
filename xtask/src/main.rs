use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let command = args.next();
    if command.as_deref() != Some("public-contract") || args.next().is_some() {
        eprintln!("usage: cargo run -p xtask -- public-contract");
        std::process::exit(2);
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask must be inside the repository root")
        .to_path_buf();

    match xtask::run_public_contract(&root) {
        Ok(summary) => {
            if let Err(error) = xtask::write_summary(&root, &summary) {
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
            let _ = xtask::write_summary(&root, &summary);
            eprintln!("public-contract checker error: {error:#}");
            std::process::exit(2);
        }
    }
}

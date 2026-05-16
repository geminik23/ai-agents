// Rust-first observability example that prints aggregate metrics after chat turns.
//
// Flow: load YAML -> build observed agent -> run chat turns -> print report -> export files
//
// Use this pattern when a Rust host wants programmatic access to latency, token, and cost metrics.

use ai_agents::{Agent, AgentBuilder, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let agent = AgentBuilder::from_yaml_file("agents/report_agent.yaml")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .build()?;

    let prompts = [
        "Explain observability in one sentence.",
        "Why should unknown model prices not be counted as zero?",
    ];

    for prompt in prompts {
        let response = agent.chat(prompt).await?;
        println!("User: {}", prompt);
        println!("Agent: {}\n", response.content);
    }

    if let Some(observability) = agent.observability() {
        let report = observability.generate_report();
        println!("Total events: {}", report.summary.total_events);
        println!("LLM calls: {}", report.summary.total_llm_calls);
        println!("Total tokens: {}", report.summary.total_tokens);
        println!("Estimated cost: ${:.6}", report.summary.total_cost_usd);

        for metric in report.by_purpose {
            let purpose = metric
                .dimensions
                .get("purpose")
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());
            println!(
                "purpose={} count={} p50={}ms p90={}ms tokens={}",
                purpose,
                metric.count,
                metric.latency.p50_ms,
                metric.latency.p90_ms,
                metric.tokens.total_tokens
            );
        }

        let export = observability.export().await?;
        for path in export.paths {
            println!("wrote {}", path.display());
        }
    }

    Ok(())
}

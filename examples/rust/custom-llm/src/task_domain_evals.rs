// Mocked or explicitly authorized live evaluation of the same domain integration used by the application.
// Flow: strict suite -> fresh attempt-owned domain installation -> existing runner -> assertions -> safe reports.
// Use this pattern for your own executable; the repository helper's fixed marker is not a plugin loader.

use ai_agents::AgentBuilder;
use ai_agents_eval::fixtures::AttemptFixtureContext;
use ai_agents_eval::{EvalHost, EvalRunner, EvalRunnerOptions};
use async_trait::async_trait;
use example_custom_llm::{task_domain, task_live, task_live_acceptance};
use std::sync::Arc;

struct DomainHost;
#[async_trait]
impl EvalHost for DomainHost {
    fn validate(&self, spec: &ai_agents::spec::AgentSpec) -> ai_agents_eval::Result<()> {
        Ok(task_live::preflight_eval(spec)?)
    }
    // Reset/retry installs fresh business state while preserving the runner-supplied provider budgets, instrumentation and policies.
    fn configure(
        &self,
        builder: AgentBuilder,
        _: &AttemptFixtureContext,
    ) -> ai_agents_eval::Result<AgentBuilder> {
        // This executable explicitly owns its hosting policy. The reusable domain installer never replaces caller storage or ceilings.
        let builder = builder.automatic_task_options(ai_agents::AutomaticTaskOptions {
            ceilings: ai_agents::AutonomyHostCeilings {
                max_declared_write_paths: Some(1),
                ..Default::default()
            },
            ..Default::default()
        });
        Ok(task_domain::install_domain(builder)?)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["--help"] {
        println!("task-domain-evals SUITE OUTPUT [--real-llm] | SUITE --dry-config-check");
        return Ok(());
    }
    let real = args.iter().any(|a| a == "--real-llm");
    let dry = args.iter().any(|a| a == "--dry-config-check");
    if (real && dry) || args.iter().filter(|a| a.starts_with("--")).count() > 1 {
        return Err("choose exactly one supported mode".into());
    }
    args.retain(|a| a != "--real-llm" && a != "--dry-config-check");
    if args.iter().any(|a| a.starts_with("--")) {
        return Err("unknown option".into());
    }
    if dry {
        let [suite] = args.as_slice() else {
            return Err("dry usage: task-domain-evals SUITE --dry-config-check".into());
        };
        task_live::preflight_suite(std::path::Path::new(suite))?;
        println!(
            "declarations and domain configuration valid; providers and runtime not constructed"
        );
        return Ok(());
    }
    let [suite, output] = args.as_slice() else {
        return Err("usage: task-domain-evals SUITE OUTPUT [--real-llm]".into());
    };
    task_live::preflight_suite(std::path::Path::new(suite))?;
    let runner = EvalRunner::from_file(
        suite,
        EvalRunnerOptions {
            output: output.into(),
            llm_mode: real.then_some(ai_agents_eval::LlmFixtureMode::Real),
            ..Default::default()
        },
    )?
    .with_host(Arc::new(DomainHost));
    let result = runner.run().await?;
    ai_agents_eval::write_outputs(&result, std::path::Path::new(output), true)?;
    let acceptance = result
        .scenarios
        .iter()
        .any(|s| s.id.starts_with("live-"))
        .then(|| task_live_acceptance::evaluate(&result));
    if let Some(acceptance) = &acceptance {
        std::fs::write(
            std::path::Path::new(output).join("acceptance.json"),
            serde_json::to_vec_pretty(acceptance)?,
        )?;
        println!(
            "detailed task acceptance: {}",
            if acceptance.passed {
                "passed"
            } else {
                "failed"
            }
        );
    }
    println!(
        "{} passed / {} total; {} failed, {} skipped",
        result.passed, result.total, result.failed, result.skipped
    );
    if result.failed > 0 || result.skipped > 0 || acceptance.is_some_and(|a| !a.passed) {
        return Err("domain task evaluation failed; inspect redacted reports".into());
    }
    Ok(())
}

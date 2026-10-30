// Offline evaluation of the same domain integration used by task-domain-host.
// Flow: strict suite -> fresh attempt-owned domain installation -> existing runner -> assertions -> safe reports.
// Use this pattern for your own executable; the repository helper's fixed marker is not a plugin loader.

mod task_domain;
use ai_agents::AgentBuilder;
use ai_agents_eval::fixtures::AttemptFixtureContext;
use ai_agents_eval::{EvalHost, EvalRunner, EvalRunnerOptions};
use async_trait::async_trait;
use std::sync::Arc;

struct DomainHost;
#[async_trait]
impl EvalHost for DomainHost {
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
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [suite, output] = args.as_slice() else {
        return Err("usage: task-domain-evals SUITE OUTPUT".into());
    };
    let runner = EvalRunner::from_file(
        suite,
        EvalRunnerOptions {
            output: output.into(),
            ..Default::default()
        },
    )?
    .with_host(Arc::new(DomainHost));
    let result = runner.run().await?;
    ai_agents_eval::write_outputs(&result, std::path::Path::new(output), true)?;
    println!(
        "{} passed / {} total; {} failed, {} skipped",
        result.passed, result.total, result.failed, result.skipped
    );
    if result.failed > 0 || result.skipped > 0 {
        return Err("domain task evaluation failed; inspect redacted reports".into());
    }
    Ok(())
}

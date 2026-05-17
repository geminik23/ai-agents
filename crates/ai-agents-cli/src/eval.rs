use anyhow::{Context, Result};

use crate::cli::EvalArgs;

pub async fn run_eval(args: EvalArgs) -> Result<()> {
    crate::init_tracing_with_default("ai_agents=warn,ai_agents_eval=warn");

    let options = ai_agents_eval::EvalRunnerOptions {
        agent: args.agent.clone(),
        scenarios: Some(args.scenarios.clone()),
        output: args.output.clone(),
        ids: args.ids.clone(),
        tags: args.tags.clone(),
        tag_mode_all: args.tag_mode == "all",
        languages: args.languages.clone(),
        retries: args.retries,
        timeout_ms: args.timeout,
        parallel: args.parallel,
        fail_fast: args.fail_fast,
        observability: args.observability,
    };

    let runner = match ai_agents_eval::EvalRunner::from_file(&args.scenarios, options) {
        Ok(runner) => runner,
        Err(error) => {
            eprintln!("Eval configuration error: {error}");
            std::process::exit(2);
        }
    };

    let result = match runner.run().await {
        Ok(result) => result,
        Err(error) => {
            eprintln!("Eval runtime error: {error}");
            std::process::exit(2);
        }
    };

    ai_agents_eval::write_outputs(&result, &args.output, args.junit)
        .with_context(|| format!("failed to write eval outputs to {}", args.output.display()))?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!(
            "Eval complete: {}/{} passed, {} failed, {} skipped. Output: {}",
            result.passed,
            result.total,
            result.failed,
            result.skipped,
            args.output.display()
        );
    }

    if result.failed > 0 {
        std::process::exit(1);
    }

    Ok(())
}

use anyhow::Result;

use ai_agents_eval::LlmFixtureMode;

use crate::cli::EvalArgs;

pub async fn run_eval(args: EvalArgs) -> Result<()> {
    crate::init_tracing_with_default("ai_agents=warn,ai_agents_eval=warn");

    if !matches!(args.tag_mode.as_str(), "any" | "all") {
        eprintln!("Eval configuration error: --tag-mode must be 'any' or 'all'");
        std::process::exit(2);
    }
    let override_count =
        args.record.is_some() as u8 + args.replay.is_some() as u8 + args.real_llm as u8;
    if override_count > 1 {
        eprintln!("Eval configuration error: use only one of --record, --replay, or --real-llm");
        std::process::exit(2);
    }
    let (llm_mode, cassette) = if let Some(path) = &args.record {
        (Some(LlmFixtureMode::Record), Some(path.clone()))
    } else if let Some(path) = &args.replay {
        (Some(LlmFixtureMode::Replay), Some(path.clone()))
    } else if args.real_llm {
        (Some(LlmFixtureMode::Real), None)
    } else {
        (None, None)
    };

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
        llm_mode,
        cassette,
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

    if let Err(error) = ai_agents_eval::write_outputs(&result, &args.output, args.junit) {
        eprintln!(
            "Eval output error: failed to write eval outputs to {}: {}",
            args.output.display(),
            error
        );
        std::process::exit(2);
    }

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

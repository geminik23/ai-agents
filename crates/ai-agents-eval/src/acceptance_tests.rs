//! Public suite acceptance keeps setup errors separate from an expected negative task outcome.
use crate::{EvalRunner, EvalRunnerOptions};

// Keep longer registry fixtures out of the parent process's environment-reader exclusion tests.
#[test]
fn formal_control_and_failure_suites_use_the_public_runner() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "acceptance_tests::formal_acceptance_worker",
            "--ignored",
            "--nocapture",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[tokio::test]
#[ignore = "formal suite worker, executed by its process-isolated parent test"]
async fn formal_acceptance_worker() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/eval/mocked/autonomy");
    let context = crate::fixtures::AttemptFixtureContext::create(None).unwrap();
    for name in [
        "diagnostics_unavailable_mocked",
        "explicit_skill_mocked",
        "host_controls_mocked",
        "mutation_freshness_mocked",
    ] {
        let path = root.join(format!("{name}.yaml"));
        let mut suite: serde_yaml::Value =
            serde_yaml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let agent = root
            .join(suite["agent"].as_str().unwrap())
            .canonicalize()
            .unwrap();
        suite["agent"] = serde_yaml::Value::String(agent.to_string_lossy().into_owned());
        // This test uses only repository-owned static fixtures; unredacted diagnostics never leave the isolated test workspace.
        suite["settings"]["redact_outputs"] = serde_yaml::Value::Bool(false);
        let suite_path = context.workspace.join(format!("{name}.yaml"));
        std::fs::write(&suite_path, serde_yaml::to_string(&suite).unwrap()).unwrap();
        let result = EvalRunner::from_file(suite_path, EvalRunnerOptions::default())
            .unwrap()
            .run()
            .await
            .unwrap();
        let failures: Vec<_> = result
            .scenarios
            .iter()
            .flat_map(|s| &s.attempts)
            .flat_map(|a| &a.turns)
            .filter(|t| t.runtime_error.is_some() || t.assertion_results.iter().any(|a| !a.passed))
            .map(|t| {
                (
                    &t.runtime_error,
                    t.assertion_results
                        .iter()
                        .filter(|a| !a.passed)
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        assert_eq!(result.failed, 0, "{name}: {failures:?}");
    }
    std::fs::remove_dir_all(context.workspace).unwrap();
}

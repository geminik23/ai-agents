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

// Both a new host tool and the host's replacement for file_write declare resource as their only filesystem input.
#[tokio::test]
async fn workspace_paths_follow_the_host_installed_tools() {
    use crate::fixtures::{AttemptFixtureContext, AttemptWorkspace};
    use ai_agents_core::{
        PathPolicyBinding, Tool, ToolExecutionContext, ToolPolicyBindings, ToolResult,
    };
    use ai_agents_runtime::AgentBuilder;
    use ai_agents_tools::ToolRegistry;
    use serde_json::{Value, json};
    use std::sync::{Arc, Mutex};

    struct Probe(&'static str, Arc<Mutex<Vec<Value>>>);
    #[async_trait::async_trait]
    impl Tool for Probe {
        fn id(&self) -> &str {
            self.0
        }
        fn name(&self) -> &str {
            self.0
        }
        fn description(&self) -> &str {
            "Capture interpolated arguments"
        }
        fn input_schema(&self) -> Value {
            json!({"type":"object"})
        }
        fn policy_bindings(&self) -> ToolPolicyBindings {
            ToolPolicyBindings {
                path_fields: vec![PathPolicyBinding::read("resource")],
                ..Default::default()
            }
        }
        async fn execute(&self, args: Value, _: ToolExecutionContext) -> ToolResult {
            self.1
                .lock()
                .unwrap()
                .push(json!({"tool":self.0,"args":args}));
            ToolResult::ok("captured")
        }
    }
    #[derive(Default)]
    struct Host {
        expected: Mutex<Option<Value>>,
        calls: Arc<Mutex<Vec<Value>>>,
    }
    impl crate::EvalHost for Host {
        fn configure(
            &self,
            builder: AgentBuilder,
            context: &AttemptFixtureContext,
        ) -> crate::Result<AgentBuilder> {
            let root = &context.workspace;
            *self.expected.lock().unwrap() = Some(json!({
                "resource":root.join("nested").join("report"),
                "path":format!("{}/literal/body",root.display())
            }));
            let mut tools = ToolRegistry::new();
            tools
                .register(Arc::new(Probe("file_write", self.calls.clone())))
                .unwrap();
            Ok(builder
                .tools(tools)
                .tool(Arc::new(Probe("host_probe", self.calls.clone()))))
        }
    }
    let dir = AttemptWorkspace::create(None).unwrap();
    std::fs::write(
        dir.workspace.join("agent.yaml"),
        "name: Bindings\nsystem_prompt: test\ntools: [file_write, host_probe]\n",
    )
    .unwrap();
    let suite = dir.workspace.join("suite.yaml");
    std::fs::write(&suite, r#"
name: installed path roles
agent: agent.yaml
fixtures:
  llm:
    mode: mock
    responses:
      - '{"tool":"file_write","arguments":{"resource":"{{ eval.workspace }}/nested/report","path":"{{ eval.workspace }}/literal/body"}}'
      - done
      - '{"tool":"host_probe","arguments":{"resource":"{{ eval.workspace }}/nested/report","path":"{{ eval.workspace }}/literal/body"}}'
      - done
scenarios:
  - id: binding
    turns:
      - input: first
        assert: {response_contains: done}
      - input: second
        assert: {response_contains: done}
"#).unwrap();
    let host = Arc::new(Host::default());
    let result = EvalRunner::from_file(suite, Default::default())
        .unwrap()
        .with_host(host.clone())
        .run()
        .await
        .unwrap();
    assert_eq!(result.passed, 1);
    let expected = host.expected.lock().unwrap().clone().unwrap();
    assert_eq!(
        *host.calls.lock().unwrap(),
        [
            json!({"tool":"file_write","args":expected}),
            json!({"tool":"host_probe","args":expected})
        ]
    );
}

#[tokio::test]
async fn workspace_validator_roles_preserve_command_cwd_and_opaque_host_data() {
    use crate::fixtures::{AttemptFixtureContext, AttemptWorkspace};
    use ai_agents_runtime::{AgentBuilder, autonomy::*};
    use serde_json::{Value, json};
    use std::sync::Arc;

    struct Validator(AdapterDescriptor, Value);
    impl AutonomyValidator for Validator {
        fn descriptor(&self) -> &AdapterDescriptor {
            &self.0
        }
        fn config_filesystem_paths(&self) -> &[&str] {
            &["/inputs/file"]
        }
        fn validate_config(&self, config: &Value) -> ai_agents_core::Result<()> {
            if config != &self.1 {
                return Err(ai_agents_core::AgentError::Config(format!(
                    "unexpected validator input {config}"
                )));
            }
            Ok(())
        }
        fn evaluate(&self, _: &ValidationInput<'_>) -> ai_agents_core::Result<ValidationDecision> {
            Ok(ValidationDecision::Complete {
                outcome: GateOutcome::Pass,
                reason: "observed".into(),
                metrics: json!({}),
                evidence_refs: vec![],
            })
        }
    }
    struct Host;
    impl crate::EvalHost for Host {
        fn configure(
            &self,
            builder: AgentBuilder,
            context: &AttemptFixtureContext,
        ) -> crate::Result<AgentBuilder> {
            let mut extensions = AutonomyExtensions::builtins();
            extensions.replace_validator(Arc::new(Validator(AdapterDescriptor {
                id:"builtin.artifact".into(), contract_version:1, ..Default::default()
            },json!({"inputs":{"file":context.workspace.join("checker-cwd").join("in")},"path":format!("{}/literal/body",context.workspace.display())}))))?;
            Ok(builder.autonomy_extensions(extensions))
        }
    }
    let dir = AttemptWorkspace::create(None).unwrap();
    std::fs::write(dir.workspace.join("agent.yaml"), r#"
name: ValidatorPaths
system_prompt: test
tools: [command]
tool_security:
  enabled: true
  tools:
    command:
      working_dirs: ['{{ eval.workspace }}/checker-cwd']
      allowed_commands: [{argv: [cargo, test]}]
      allow_without_confirmation: true
autonomy:
  enabled: true
  max_turns: 1
  validation:
    checks:
      - {id: cli, adapter: builtin.command, config: {command: 'cargo test', cwd: '{{ eval.workspace }}/checker-cwd'}}
      - {id: opaque, adapter: builtin.artifact, config: {path: '{{ eval.workspace }}/literal/body', inputs: {file: '{{ eval.workspace }}/checker-cwd/in'}}}
  completion: {all: [{validator_passed: cli}, {validator_passed: opaque}]}
"#).unwrap();
    let suite = dir.workspace.join("suite.yaml");
    let source = r#"
name: ValidatorPaths
agent: agent.yaml
fixtures:
  files: {checker-cwd/in: content}
  llm: {mode: mock, responses: [done]}
  commands:
    available: true
    entries:
      - argv: [cargo, test]
        response: {success: true, exit_code: 0, termination: exited, stdout: ok, stderr: '', combined_output: ok, truncated: false, cwd: '{{ eval.workspace }}/checker-cwd', argv_redacted: [cargo, test]}
scenarios:
  - id: checked
    steps:
      - !task
        operation: {action: run, objective: verify}
        assert:
          all:
            - task: {status: completed, checks: [{id: cli, outcome: pass}, {id: opaque, outcome: pass}]}
            - tool_called: {id: command, count: 1, executed: true, success: true, args_executed: {path: cwd, contains: 'NATIVE_SUFFIX'}}
"#;
    std::fs::write(
        &suite,
        source.replace(
            "NATIVE_SUFFIX",
            &format!("{}checker-cwd", std::path::MAIN_SEPARATOR),
        ),
    )
    .unwrap();
    let result = EvalRunner::from_file(suite, Default::default())
        .unwrap()
        .with_host(Arc::new(Host))
        .run()
        .await
        .unwrap();
    assert_eq!(result.passed, 1, "{:?}", result.scenarios);
}

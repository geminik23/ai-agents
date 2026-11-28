//! Automatic scopes use the same actual host-state validators and mutation tools as the existing whole-runtime domain acceptance.

use super::*;
use crate::Agent;

// Actual domain tools and revision-bound validators enter through the configured hosting preparation shared by explicit and automatic tasks.
async fn automatic_domain_agent(
    domain: &str,
    skill: bool,
    state: Arc<DomainState>,
    provider: Arc<RecordingProvider>,
    sqlite: bool,
) -> crate::RuntimeAgent {
    let mut extensions = AutonomyExtensions::builtins();
    extensions
        .register_validator(Arc::new(DomainCheck {
            descriptor: AdapterDescriptor {
                id: "host.records".into(),
                contract_version: 1,
                requires_host_revision: true,
                ..Default::default()
            },
            state: state.clone(),
        }))
        .unwrap();
    extensions
        .register_validation_target("records", "business-state", state.clone())
        .unwrap();
    let entry = if skill {
        "skills:\n  - id: work\n    description: Domain work\n    trigger: User requests domain work\n    autonomy: {profile: domain, mode: skill_until_complete}\n    steps:\n      - prompt: Pursue the requested objective\n"
    } else {
        "states:\n  initial: work\n  states:\n    work:\n      autonomy: {profile: domain, mode: state_until_exit}\n"
    };
    let yaml = format!(
        "name: automatic-domain\nsystem_prompt: Only actual host state proves completion\nllm: {{default: default, router: router}}\ntools: [domain]\nautonomy:\n  profiles:\n    domain:\n      enabled: true\n      max_turns: 8\n      max_llm_calls: 16\n      validation:\n        checks: [{{id: records, adapter: host.records, config: {{domain: {domain}}}}}]\n      completion: {{validator_passed: records}}\n{entry}"
    );
    let store = crate::autonomy::continuation_test_support::store(
        "automatic-domain".into(),
        "automatic-domain.v1",
        sqlite,
    )
    .await;
    crate::AgentBuilder::from_yaml(&yaml)
        .unwrap()
        .llm(provider)
        .llm_alias("router", RecordingProvider::new(&["work"]))
        .tool(Arc::new(DomainTool(state)))
        .autonomy_extensions(extensions)
        .automatic_task_options(AutomaticTaskOptions {
            store: Some(store),
            config_identity: Some("automatic-domain.v1".into()),
            ..Default::default()
        })
        .build_task_host()
        .await
        .unwrap()
}

// An answer claiming completion cannot establish ticket, report, operations, world or code predicates before an actual admitted mutation.
#[tokio::test]
async fn auto_state_and_skill_domains_require_actual_host_mutation() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        for skill in [false, true] {
            for (domain, operation) in [
                ("support", r#"{"op":"resolve"}"#),
                (
                    "report",
                    r#"{"op":"report","value":{"title":"verified","rows":[1,2]}}"#,
                ),
                ("operations", r#"{"op":"incident"}"#),
                ("world", r#"{"op":"world","value":10}"#),
                ("code", r#"{"op":"code","value":"return 2"}"#),
            ] {
                let state = DomainState::new();
                let call = json!({"tool":"domain","arguments":serde_json::from_str::<Value>(operation).unwrap()}).to_string();
                let provider = RecordingProvider::new(&[
                    "I claim the objective is complete",
                    &call,
                    "host mutation finished",
                ]);
                let agent =
                    automatic_domain_agent(domain, skill, state.clone(), provider.clone(), sqlite)
                        .await;
                agent.chat(domain).await.unwrap();
                let result = agent.automatic_task_result().unwrap();
                assert_eq!(
                    result.run.status,
                    TaskRunStatus::Completed,
                    "{domain}/{skill}: {:?}",
                    result.run.stop_reason
                );
                assert_eq!(state.effects.load(Ordering::SeqCst), 1);
                assert!(state.observed.lock().len() >= 2);
                assert_eq!(result.run.counters.tool_attempts, 1);
                assert_eq!(result.run.counters.llm_attempts, 3);
                assert_eq!(result.run.counters.turns, 2);
            }
        }
    }
}

// Repeated sources do not count as new research even when automatic entry and skill continuation select the work.
#[tokio::test]
async fn auto_research_deduplicates_actual_sources_without_files_or_todos() {
    for skill in [false, true] {
        let state = DomainState::new();
        let provider = RecordingProvider::new(&[
            "I claim enough sources",
            r#"{"tool":"domain","arguments":{"op":"source","value":"A"}}"#,
            "one source",
            r#"{"tool":"domain","arguments":{"op":"source","value":"A"}}"#,
            "still one unique source",
            r#"{"tool":"domain","arguments":{"op":"source","value":"B"}}"#,
            "two actual sources",
        ]);
        let agent =
            automatic_domain_agent("research", skill, state.clone(), provider.clone(), false).await;
        agent.chat("research").await.unwrap();
        let result = agent.automatic_task_result().unwrap();
        assert_eq!(
            result.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            result.run.stop_reason
        );
        assert_eq!(state.effects.load(Ordering::SeqCst), 3);
        assert_eq!(result.run.counters.llm_attempts, 7);
        assert_eq!(result.run.counters.turns, 4);
        let observations = state.observed.lock();
        let unique: Vec<_> = observations
            .iter()
            .map(|data| {
                data["sources"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<BTreeSet<_>>()
                    .len()
            })
            .collect();
        assert_eq!(unique, vec![0, 1, 1, 2]);
    }
}

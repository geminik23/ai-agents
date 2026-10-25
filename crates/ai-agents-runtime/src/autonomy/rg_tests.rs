//! Whole-runtime acceptance reuses the production runner and isolated host state, not canned domain completion claims.

use ai_agents_core::{Result, Tool, ToolExecutionContext, ToolResult, ToolSafetyMetadata};
use async_trait::async_trait;
use std::collections::BTreeSet;

#[path = "automatic_domain_tests.rs"]
mod automatic_acceptance;
#[path = "ev_domain_tests.rs"]
mod event_acceptance;
#[path = "rg_receipt_tests.rs"]
mod receipts;
use super::runner_tests::RecordingProvider;
use super::*;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

// Every profile is parsed by production serde and bound through the same runtime registry as embedded applications.
fn config(yaml: &str) -> AutonomyConfig {
    serde_yaml::from_str(&format!(
        "enabled: true\nmax_turns: 8\nmax_llm_calls: 16\n{yaml}"
    ))
    .unwrap()
}

// The real runner owns all effects, journals, progress and terminal publication; no fixture can set task status.
async fn runner(
    agent: Arc<crate::RuntimeAgent>,
    config: AutonomyConfig,
    sqlite: bool,
) -> (AutonomyRunner, Arc<dyn TaskRunStore>) {
    let store = super::continuation_test_support::store(agent.info().id, "rg.v1", sqlite).await;
    (
        AutonomyRunner::try_new(
            agent,
            config,
            Default::default(),
            store.clone(),
            "rg.v1".into(),
        )
        .unwrap(),
        store,
    )
}

struct DomainState {
    data: parking_lot::Mutex<Value>,
    revision: AtomicUsize,
    effects: AtomicUsize,
    observed: parking_lot::Mutex<Vec<Value>>,
}
impl DomainState {
    // All authoritative business facts live outside the LLM response and change only in real admitted tool bodies or explicit host mutation.
    fn new() -> Arc<Self> {
        Arc::new(Self {
            data: parking_lot::Mutex::new(
                json!({"sources":[],"ticket":"open","incident":null,"health":"down","world":0,"report":{"title":17},"source":"bad"}),
            ),
            revision: AtomicUsize::new(0),
            effects: AtomicUsize::new(0),
            observed: Default::default(),
        })
    }
}
impl HostRevisionSource for DomainState {
    // Monotonic opaque revisions change on every relevant mutation, including value changes back to an earlier state.
    fn revision(&self) -> Result<String> {
        Ok(self.revision.load(Ordering::SeqCst).to_string())
    }
}
struct DomainTool(Arc<DomainState>);
#[async_trait]
impl Tool for DomainTool {
    // One host integration mutates isolated business state through the ordinary shared executor.
    fn id(&self) -> &str {
        "domain"
    }
    // Display identity does not grant the registered tool.
    fn name(&self) -> &str {
        "Domain"
    }
    // The fixture never opens network connections or executes commands.
    fn description(&self) -> &str {
        "Update isolated host records"
    }
    // Operation arguments are structural protocol data, not language-dependent completion heuristics.
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"op":{"type":"string"},"value":{}},"required":["op"]})
    }
    // Actual writes increment the host revision and effect witness, never a requested-call counter.
    async fn execute(&self, arguments: Value, _: ToolExecutionContext) -> ToolResult {
        let mut data = self.0.data.lock();
        match arguments["op"].as_str() {
            Some("source") => data["sources"]
                .as_array_mut()
                .unwrap()
                .push(arguments["value"].clone()),
            Some("resolve") => data["ticket"] = json!("resolved"),
            Some("incident") => {
                data["incident"] = json!({"id":"isolated-incident","status":"open"})
            }
            Some("report") => data["report"] = arguments["value"].clone(),
            Some("world") => data["world"] = arguments["value"].clone(),
            Some("code") => data["source"] = arguments["value"].clone(),
            _ => return ToolResult::error("unknown isolated operation"),
        }
        self.0.effects.fetch_add(1, Ordering::SeqCst);
        self.0.revision.fetch_add(1, Ordering::SeqCst);
        ToolResult::ok(data.to_string())
    }
    // Mutations remain subject to normal policy, ownership, final admission and mutation-generation tracking.
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata::default()
    }
}
struct DomainCheck {
    descriptor: AdapterDescriptor,
    state: Arc<DomainState>,
}
impl AutonomyValidator for DomainCheck {
    // The actual installed adapter declares its host revision dependency before profile binding.
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.descriptor
    }
    // Named profiles select one bounded predicate; they cannot provide a desired Pass or synthetic metric.
    fn validate_config(&self, config: &Value) -> Result<()> {
        if !matches!(
            config["domain"].as_str(),
            Some("research" | "support" | "report" | "operations" | "health" | "world" | "code")
        ) {
            return Err(ai_agents_core::AgentError::Config(
                "unknown domain predicate".into(),
            ));
        }
        Ok(())
    }
    // Pure host-state calculation supplies facts; the production controller alone decides completion.
    fn evaluate(&self, input: &ValidationInput<'_>) -> Result<ValidationDecision> {
        let data = self.state.data.lock().clone();
        self.state.observed.lock().push(data.clone());
        let unique = data["sources"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect::<BTreeSet<_>>()
            .len();
        let pass = match input.config["domain"].as_str().unwrap() {
            "research" => {
                let enough = unique >= input.config["min_unique"].as_u64().unwrap_or(2) as usize;
                let citations = if input.config["citations"] == true {
                    input
                        .evidence
                        .current
                        .as_ref()
                        .and_then(|current| {
                            serde_json::from_str::<Value>(&current.value.response).ok()
                        })
                        .and_then(|response| response["sources"].as_array().cloned())
                        .is_some_and(|sources| {
                            sources
                                .iter()
                                .filter_map(Value::as_str)
                                .collect::<BTreeSet<_>>()
                                == data["sources"]
                                    .as_array()
                                    .unwrap()
                                    .iter()
                                    .filter_map(Value::as_str)
                                    .collect::<BTreeSet<_>>()
                        })
                } else {
                    true
                };
                enough && citations
            }
            "support" => data["ticket"] == "resolved",
            "report" => {
                data["report"]["title"].is_string()
                    && data["report"]["rows"]
                        .as_array()
                        .is_some_and(|rows| rows.len() == 2 && rows.iter().all(Value::is_number))
            }
            "operations" => {
                data["incident"]["id"].is_string() && data["incident"]["status"] == "open"
            }
            "health" => data["health"] == "up",
            "world" => data["world"]
                .as_i64()
                .is_some_and(|position| position >= 10),
            "code" => data["source"] == "return 2",
            _ => unreachable!(),
        };
        Ok(ValidationDecision::Complete {
            outcome: if pass {
                GateOutcome::Pass
            } else {
                GateOutcome::Fail
            },
            reason: "observed host predicate".into(),
            metrics: json!({"unique":unique,"artifact":{"path":"report://current","exists":!data["report"].is_null()}}),
            evidence_refs: vec![],
        })
    }
    // A report artifact reference is a typed host projection, not data accepted from a model's proof fields.
    fn artifacts(&self, result: &ValidationResult) -> Result<Vec<ArtifactObservation>> {
        Ok(vec![serde_json::from_value(
            result.metrics["artifact"].clone(),
        )?])
    }
}

// Production builder and frozen host bindings are identical for every domain; only declarative predicates and business state differ.
fn domain_agent(
    state: Arc<DomainState>,
    provider: Arc<RecordingProvider>,
    question: bool,
) -> Arc<crate::RuntimeAgent> {
    domain_agent_observed(state, provider, question, false)
}

// Optional production telemetry wraps the same host and provider, including reconstruction into a new collector.
fn domain_agent_observed(
    state: Arc<DomainState>,
    provider: Arc<RecordingProvider>,
    question: bool,
    observed: bool,
) -> Arc<crate::RuntimeAgent> {
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
    extensions
        .register_validation_target("health", "business-state", state.clone())
        .unwrap();
    let mut builder = crate::AgentBuilder::from_yaml(&format!("name: RGDomain\nsystem_prompt: Pursue the configured objective; only host facts prove completion.\ntools: [domain{}]\n", if question { ", ask_user" } else { "" })).unwrap()
        .llm(provider)
        .tool(Arc::new(DomainTool(state)))
        .autonomy_extensions(extensions);
    if question {
        builder = builder.tool(Arc::new(ai_agents_tools::AskUserTool::new(
            Default::default(),
        )));
    }
    if observed {
        let mut config = ai_agents_observability::ObservabilityConfig {
            enabled: true,
            ..Default::default()
        };
        config.export.write_report = false;
        builder = builder.observability(ai_agents_observability::ObservabilityManager::new(config));
    }
    Arc::new(builder.build().unwrap())
}

// Source A repeated twice cannot become two sources or restore spent stagnation capacity.
#[tokio::test]
async fn rg_research_consumes_real_deduplicated_sources_without_files_or_todos() {
    for sqlite in super::continuation_test_support::backends() {
        let state = DomainState::new();
        let provider = RecordingProvider::new(&[
            r#"{"tool":"domain","arguments":{"op":"source","value":"A"}}"#,
            "checkpoint A",
            "{\"overall_score\":0.9}",
            r#"{"tool":"domain","arguments":{"op":"source","value":"A"}}"#,
            "checkpoint duplicate",
            "{\"overall_score\":0.9}",
            r#"{"tool":"domain","arguments":{"op":"source","value":"B"}}"#,
            "checkpoint B: sources A and B",
            "{\"overall_score\":0.9}",
        ]);
        let agent = domain_agent(state.clone(), provider.clone(), false);
        let config = config(
            "validation:\n  checks: [{id: records, adapter: host.records, config: {domain: research}}, {id: quality, adapter: builtin.judge, config: {criteria: [grounded synthesis of the source evidence], pass_threshold: 0.8}}]\ncompletion: {all: [{validator_passed: records}, {validator_passed: quality}]}\nprogress:\n  adapter: builtin.evidence\n  config:\n    signals: [{id: sources, validator: records, path: metrics.unique, comparison: increase}]\n  stagnation: {max_cycles_without_progress: 3, action: stop}\n",
        );
        let (runner, store) = runner(agent, config, *sqlite).await;
        let done = runner
            .run("verify two distinct sources", None)
            .await
            .unwrap();
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "reason={:?}; effects={}; observations={:?}",
            done.run.stop_reason,
            state.effects.load(Ordering::SeqCst),
            state.observed.lock()
        );
        assert_eq!(done.run.counters.turns, 3);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 9);
        assert_eq!(state.effects.load(Ordering::SeqCst), 3);
        let observations = state.observed.lock().clone();
        assert_eq!(
            observations
                .iter()
                .map(|data| data["sources"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<BTreeSet<_>>()
                    .len())
                .collect::<Vec<_>>(),
            vec![1, 1, 2]
        );
        let saved = store.load(&done.run.key.run_id).await.unwrap().unwrap();
        assert!(saved.payload["todos"].is_null());
    }
}

// Claimed completion is rejected until an actual host effect repairs the report, incident or world predicate.
#[tokio::test]
async fn rg_report_operations_world_and_code_require_actual_host_state() {
    for sqlite in super::continuation_test_support::backends() {
        for (domain, operation) in [
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
                "required review finished",
            ]);
            let agent = domain_agent(state.clone(), provider.clone(), false);
            let mut yaml = format!(
                "validation:\n  checks: [{{id: records, adapter: host.records, config: {{domain: {domain}}}}}]\ncompletion: {{validator_passed: records}}\n"
            );
            if domain == "report" {
                yaml = yaml.replace("completion: {validator_passed: records}", "completion: {all: [{artifact_exists: 'report://current'}, {validator_passed: records}]}");
            }
            if domain == "operations" {
                yaml = "validation:\n  checks: [{id: health, adapter: host.records, required: false, config: {domain: health}}, {id: records, adapter: host.records, required: false, config: {domain: operations}}]\ncompletion: {any: [{validator_passed: health}, {validator_passed: records}]}\nlifecycle: [{id: remediate}, {id: verify}, {id: review}]\n".into();
            }
            let (runner, _) = runner(agent, config(&yaml), *sqlite).await;
            let done = runner.run(domain, None).await.unwrap();
            assert_eq!(
                done.run.status,
                TaskRunStatus::Completed,
                "{domain}: {:?}",
                done.run.stop_reason
            );
            assert_eq!(state.effects.load(Ordering::SeqCst), 1);
            assert!(state.observed.lock().len() >= 2);
            if domain == "operations" {
                assert_eq!(done.run.counters.turns, 3);
                assert_eq!(provider.calls.load(Ordering::SeqCst), 4);
                assert_eq!(state.data.lock()["health"], "down");
                assert_eq!(state.data.lock()["incident"]["status"], "open");
            }
        }
    }
}

// A standard artifact gate executes the authorized built-in metadata tool; missing and denied probes cannot establish success.
#[tokio::test]
async fn rg_builtin_artifact_uses_authorized_file_info() {
    let directory = std::env::temp_dir().join(format!("rg-artifact-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("report.json");
    for exists in [false, true] {
        if exists {
            std::fs::write(&path, "{\"rows\":[1,2]}").unwrap();
        }
        for sqlite in super::continuation_test_support::backends() {
            let provider = RecordingProvider::new(&["report candidate"]);
            let agent = Arc::new(
                crate::AgentBuilder::new()
                    .system_prompt("report")
                    .llm(provider.clone())
                    .tool(Arc::new(ai_agents_tools::FileInfoTool::new()))
                    .build()
                    .unwrap(),
            );
            let mut config = config(&format!(
                "validation:\n  checks: [{{id: artifact, adapter: builtin.artifact, config: {{path: {}}}}}]\ncompletion: {{artifact_exists: {}}}\npremature_finish: {{action: fail}}\n",
                serde_json::to_string(&path.to_string_lossy()).unwrap(),
                serde_json::to_string(&path.to_string_lossy()).unwrap()
            ));
            config.defaults.max_llm_calls = Some(1);
            let (runner, _) = runner(agent, config, *sqlite).await;
            let done = runner.run("verify artifact", None).await.unwrap();
            assert_eq!(
                done.run.status == TaskRunStatus::Completed,
                exists,
                "{:?}",
                done.run.stop_reason
            );
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        }
    }
    std::fs::remove_dir_all(directory).unwrap();
}

// An unavailable artifact observer is unknown even under negation, rather than proof that a protected artifact is absent.
#[tokio::test]
async fn rg_ungranted_artifact_observation_cannot_complete_through_negation() {
    let provider = RecordingProvider::new(&["candidate"]);
    let agent = Arc::new(
        crate::AgentBuilder::new()
            .system_prompt("no file authority")
            .llm(provider.clone())
            .tool(Arc::new(ai_agents_tools::FileInfoTool::new()))
            .build()
            .unwrap()
            .with_declared_tool_ids(Some(Vec::new())),
    );
    let (runner,_) = runner(agent,config("validation:\n  checks: [{id: artifact, adapter: builtin.artifact, required: false, config: {path: protected.json}}]\ncompletion: {not: {artifact_exists: protected.json}}\npremature_finish: {action: fail}\n"),false).await;
    let done = runner
        .run("do not infer absence from denial", None)
        .await
        .unwrap();
    assert_ne!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(done.run.counters.tool_attempts, 0);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert!(
        done.run
            .evidence
            .records
            .iter()
            .any(|record| record.data["gate"] == "unknown")
    );
}

// Only run-labelled collector events enter the gate; prior ordinary events and absent telemetry cannot prove a task metric.
#[tokio::test]
async fn rg_observability_gate_consumes_actual_run_scoped_numeric_events() {
    for sqlite in super::continuation_test_support::backends() {
        for observed in [false, true] {
            let provider = RecordingProvider::new(&["ordinary before", "candidate"]);
            let mut builder = crate::AgentBuilder::new()
                .system_prompt("analysis")
                .llm(provider.clone());
            if observed {
                let mut config = ai_agents_observability::ObservabilityConfig {
                    enabled: true,
                    ..Default::default()
                };
                config.export.write_report = false;
                builder = builder
                    .observability(ai_agents_observability::ObservabilityManager::new(config));
            }
            let agent = Arc::new(builder.build().unwrap());
            use crate::Agent;
            agent.chat("before task").await.unwrap();
            let debug_manager = agent.observability();
            let (runner, store) = runner(agent, config("completion: {observability: {path: summary.total_llm_calls, eq: 1}}\npremature_finish: {action: fail}\n"), *sqlite).await;
            let done = runner.run("inspect one task call", None).await.unwrap();
            let diagnostic = store.load(&done.run.key.run_id).await.unwrap().unwrap();
            let diagnostic: TaskCheckpointPayload =
                serde_json::from_value(diagnostic.payload).unwrap();
            assert_eq!(
                done.run.status == TaskRunStatus::Completed,
                observed,
                "reason={:?}; numeric={:?}; report={:?}",
                done.run.stop_reason,
                diagnostic
                    .adapters
                    .iter()
                    .find(|a| a.id == "runtime.telemetry")
                    .map(|a| &a.state),
                debug_manager.map(|m| m.generate_report().summary)
            );
            let saved = store.load(&done.run.key.run_id).await.unwrap().unwrap();
            let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload).unwrap();
            let telemetry = payload
                .adapters
                .iter()
                .find(|adapter| adapter.id == "runtime.telemetry");
            assert!(telemetry.is_some());
            assert_eq!(telemetry.unwrap().state["complete"], json!(observed));
            if let Some(telemetry) = telemetry {
                let serialized = telemetry.state.to_string();
                assert!(
                    !serialized.contains("candidate") && !serialized.contains("ordinary before")
                );
            }
        }
    }
}

// Clarification and explicit fresh-runtime reconstruction preserve ticket facts, original expiry and consumed work without resolving the ticket by approval alone.
#[tokio::test]
async fn rg_support_real_ticket_survives_question_restart_and_fresh_review() {
    use super::restoration_test_support::{FreshRuntime, RebuildHost, restore};
    for sqlite in super::continuation_test_support::backends() {
        let state = DomainState::new();
        let replies = [
            r#"{"tool":"ask_user","arguments":{"question":"Choose resolution","options":["resolve"]}}"#,
            r#"{"tool":"domain","arguments":{"op":"resolve"}}"#,
            "ticket resolved",
        ];
        let provider = RecordingProvider::new(&replies);
        let agent = domain_agent_observed(state.clone(), provider.clone(), true, true);
        let mut config = config(
            "max_wall_time_seconds: 600\nvalidation:\n  checks: [{id: records, adapter: host.records, config: {domain: support}}]\ncompletion: {all: [{validator_passed: records}, {observability: {path: summary.total_llm_calls, eq: 3}}]}\nhitl: {on_user_question: pause_run}\n",
        );
        config.defaults.max_llm_calls = Some(3);
        let (runner, store) = runner(agent.clone(), config.clone(), *sqlite).await;
        let debug_cursor = agent.observability().unwrap().event_cursor();
        let paused = runner.run("resolve the actual ticket", None).await.unwrap();
        assert_eq!(paused.run.status, TaskRunStatus::Paused);
        assert_eq!(state.data.lock()["ticket"], "open");
        let before = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
        let before: TaskCheckpointPayload = serde_json::from_value(before.payload).unwrap();
        let source_report = (
            agent.observability().unwrap().generate_report().summary,
            agent
                .observability()
                .unwrap()
                .checkpoint_since(debug_cursor)
                .1
                .iter()
                .map(|event| event.dimensions.clone())
                .collect::<Vec<_>>(),
        );
        let host_state = state.clone();
        let host = RebuildHost::new(
            "rg-support.v1",
            &FreshRuntime::new(agent.clone(), vec![provider.clone()]),
            Box::new(move || {
                let state = host_state.clone();
                Box::pin(async move {
                    let provider = RecordingProvider::new(&[]);
                    FreshRuntime::new(
                        domain_agent_observed(state, provider.clone(), true, true),
                        vec![provider],
                    )
                })
            }),
        );
        drop(agent);
        let (runner, reviewed) =
            restore(runner, store.clone(), &paused.run, config, "rg.v1", host).await;
        assert_eq!(
            state.data.lock()["ticket"],
            "open",
            "restoration approval cannot resolve business state"
        );
        assert_eq!(state.effects.load(Ordering::SeqCst), 0);
        let pending = reviewed.run.pending.as_ref().unwrap();
        let done = runner
            .resume(
                &reviewed.run.key.run_id,
                reviewed.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: pending.id.clone(),
                    answer: json!({"answered":true,"selected":["resolve"]}),
                },
            )
            .await
            .unwrap();
        let diagnostic = store.load(&done.run.key.run_id).await.unwrap().unwrap();
        let diagnostic: TaskCheckpointPayload = serde_json::from_value(diagnostic.payload).unwrap();
        let identity = super::evaluation_tests::identity(&super::evaluation_tests::scope());
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "reason={:?}; numeric={:?}; before={:?}; source_report={:?}",
            done.run.stop_reason,
            super::telemetry::observation(&diagnostic, identity.clone()).unwrap(),
            super::telemetry::observation(&before, identity).unwrap(),
            source_report
        );
        assert_eq!(state.data.lock()["ticket"], "resolved");
        assert_eq!(state.effects.load(Ordering::SeqCst), 1);
        assert_eq!(done.run.counters.llm_attempts, 3);
        let after = store.load(&done.run.key.run_id).await.unwrap().unwrap();
        let after: TaskCheckpointPayload = serde_json::from_value(after.payload).unwrap();
        assert_eq!(before.clocks.expires_at, after.clocks.expires_at);
        assert!(after.clocks.active_millis >= before.clocks.active_millis + 15_000);
    }
}

// Concurrent research children publish real source records; an exact parent cursor parks the slower child without replaying the completed sibling or its writes.
#[tokio::test]
async fn rg_composed_research_deduplicates_actual_child_sources_after_pause() {
    use crate::spawner::{AgentRegistry, AgentSpawner};
    for sqlite in super::continuation_test_support::backends() {
        let state = DomainState::new();
        let registry = Arc::new(AgentRegistry::new());
        let first = RecordingProvider::new(&[
            r#"{"tool":"domain","arguments":{"op":"source","value":"A"}}"#,
            r#"{"tool":"domain","arguments":{"op":"source","value":"B"}}"#,
            "first sources A,B",
        ]);
        let second = RecordingProvider::new(&[
            r#"{"tool":"domain","arguments":{"op":"source","value":"B"}}"#,
            r#"{"tool":"domain","arguments":{"op":"source","value":"C"}}"#,
            r#"{"tool":"ask_user","arguments":{"question":"Confirm sources","options":["yes"]}}"#,
            "second sources B,C",
        ]);
        for (id, provider, question) in [
            ("first", first.clone(), false),
            ("second", second.clone(), true),
        ] {
            let yaml = format!(
                "name: {id}\nsystem_prompt: Collect source records\ntools: [domain{}]\n",
                if question { ", ask_user" } else { "" }
            );
            let mut builder = crate::AgentBuilder::from_yaml(&yaml)
                .unwrap()
                .llm(provider)
                .tool(Arc::new(DomainTool(state.clone())));
            if question {
                builder = builder.tool(Arc::new(ai_agents_tools::AskUserTool::new(
                    Default::default(),
                )));
            }
            super::co_completion_tests::register(&registry, id, builder.build().unwrap(), &yaml)
                .await;
        }
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
        let aggregate = RecordingProvider::new(&[r#"{"sources":["A","B","C"]}"#]);
        let agent = Arc::new(crate::AgentBuilder::from_yaml("name: ResearchRoot\nsystem_prompt: Synthesize verified sources\nstates:\n  initial: collect\n  states:\n    collect:\n      concurrent:\n        agents: [first, second]\n        aggregation: {strategy: llm_synthesis, synthesizer_llm: router}\n").unwrap().llm(RecordingProvider::new(&[])).llm_alias("router",aggregate.clone()).autonomy_extensions(extensions).auto_configure_features().unwrap().build().unwrap().with_spawner_handles(Arc::new(AgentSpawner::new()),registry));
        let mut config = config(
            "validation:\n  checks: [{id: records, adapter: host.records, config: {domain: research, min_unique: 3, citations: true}}]\ncompletion: {validator_passed: records}\nhitl: {on_user_question: pause_run}\n",
        );
        config.defaults.max_llm_calls = Some(8);
        let (runner, _) = runner(agent, config, *sqlite).await;
        let paused = runner
            .run("verify and synthesize three unique sources", None)
            .await
            .unwrap();
        assert_eq!(
            paused.run.status,
            TaskRunStatus::Paused,
            "{:?}",
            paused.run.stop_reason
        );
        assert_eq!(state.effects.load(Ordering::SeqCst), 4);
        assert_eq!(first.calls.load(Ordering::SeqCst), 3);
        assert_eq!(aggregate.calls.load(Ordering::SeqCst), 0);
        let pending = paused.run.pending.unwrap();
        let done = runner
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: pending.id,
                    answer: json!({"answered":true,"selected":["yes"]}),
                },
            )
            .await
            .unwrap();
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(state.effects.load(Ordering::SeqCst), 4);
        assert_eq!(first.calls.load(Ordering::SeqCst), 3);
        assert_eq!(second.calls.load(Ordering::SeqCst), 4);
        assert_eq!(aggregate.calls.load(Ordering::SeqCst), 1);
        assert_eq!(done.run.counters.llm_attempts, 8);
        let data = state.data.lock();
        assert_eq!(data["sources"].as_array().unwrap().len(), 4);
        assert_eq!(
            data["sources"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(Value::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["A", "B", "C"])
        );
    }
}

struct FinalRaceStore {
    inner: Arc<dyn TaskRunStore>,
    state: Arc<DomainState>,
    cancel: bool,
    armed: std::sync::atomic::AtomicBool,
}
#[async_trait]
impl TaskRunStore for FinalRaceStore {
    // Faults forward the same production backend and alter only the exact final publication boundary.
    async fn create(&self, snapshot: &TaskRunSnapshot) -> Result<()> {
        self.inner.create(snapshot).await
    }
    // Reads remain authoritative and never rewrite the validation result.
    async fn load(&self, id: &str) -> Result<Option<TaskRunSnapshot>> {
        self.inner.load(id).await
    }
    // Listing has no executable authority.
    async fn list(&self) -> Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    // Backend tombstones and owner checks remain intact.
    async fn delete(&self, id: &str, revision: u64) -> Result<()> {
        self.inner.delete(id, revision).await
    }
    // Change real target state or commit cancellation after gate evaluation but before the guarded CAS.
    async fn mutate(&self, id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot> {
        if let TaskRunMutation::FinalCheckpoint {
            status: TaskRunStatus::Completed,
            expected_revision,
            ..
        } = mutation
            && self.armed.swap(false, Ordering::SeqCst)
        {
            if self.cancel {
                self.inner
                    .mutate(
                        id,
                        &TaskRunMutation::RequestCancel {
                            expected_revision: *expected_revision,
                        },
                    )
                    .await?;
            } else {
                let mut data = self.state.data.lock();
                data["world"] = json!(0);
                data["report"] = Value::Null;
                self.state.revision.fetch_add(1, Ordering::SeqCst);
            }
        }
        self.inner.mutate(id, mutation).await
    }
}

// Previously true world/artifact facts cannot commit success after a real target change or a winning cancellation CAS.
#[tokio::test]
async fn rg_world_and_artifact_final_publication_recheck_actual_state_and_cancel() {
    for sqlite in super::continuation_test_support::backends() {
        for domain in ["world", "report"] {
            for cancel in [false, true] {
                let state = DomainState::new();
                {
                    let mut data = state.data.lock();
                    data["world"] = json!(10);
                    data["report"] = json!({"title":"ready","rows":[1,2]});
                }
                let provider = RecordingProvider::new(&["candidate"]);
                let agent = domain_agent(state.clone(), provider.clone(), false);
                let inner =
                    super::continuation_test_support::store(agent.info().id, "rg.v1", *sqlite)
                        .await;
                let store = Arc::new(FinalRaceStore {
                    inner: inner.clone(),
                    state: state.clone(),
                    cancel,
                    armed: std::sync::atomic::AtomicBool::new(true),
                });
                let yaml = format!(
                    "validation:\n  checks: [{{id: records, adapter: host.records, config: {{domain: {domain}}}}}]\ncompletion: {{all: [{{validator_passed: records}}, {{artifact_exists: 'report://current'}}]}}\n"
                );
                let runner = AutonomyRunner::try_new(
                    agent.clone(),
                    config(&yaml),
                    Default::default(),
                    store.clone(),
                    "rg.v1".into(),
                )
                .unwrap();
                if let Ok(result) = runner.run("verify real state at commit", None).await {
                    assert_ne!(result.run.status, TaskRunStatus::Completed);
                }
                assert!(!store.armed.load(Ordering::SeqCst));
                let run = inner.list().await.unwrap().remove(0);
                let saved = inner.load(&run.key.run_id).await.unwrap().unwrap();
                assert_ne!(saved.status, TaskRunStatus::Completed);
                assert_eq!(saved.cancel_requested, cancel);
                assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
                assert_eq!(state.effects.load(Ordering::SeqCst), 0);
            }
        }
    }
}

// Exact-cap progress remains sufficient after fresh-runtime construction and separate final-stage review; no validation prefix is replayed.
#[tokio::test]
async fn rg_last_slot_progress_survives_reconstruction_and_final_review_without_extra_work() {
    use super::restoration_test_support::{FreshRuntime, RebuildHost, restore};
    for sqlite in super::continuation_test_support::backends() {
        let state = DomainState::new();
        state.data.lock()["sources"] = json!(["A", "B"]);
        let provider = RecordingProvider::new(&["final analysis"]);
        let agent = domain_agent(state.clone(), provider.clone(), false);
        let mut config = config(
            "lifecycle: [{id: final_review}]\nsupervision: {stage_changes: require_approval}\nvalidation:\n  checks: [{id: records, adapter: host.records, config: {domain: research}}]\nprogress:\n  adapter: builtin.evidence\n  config:\n    signals: [{id: sources, validator: records, path: metrics.unique, comparison: increase}]\n  stagnation: {max_cycles_without_progress: 1, action: fail}\ncompletion: {progress_path: {path: metrics.signals.sources, gte: 2}}\n",
        );
        config.defaults.max_llm_calls = Some(1);
        let (runner, store) = runner(agent.clone(), config.clone(), *sqlite).await;
        let initial = runner
            .run("finalize from the last admitted call", None)
            .await
            .unwrap();
        assert_eq!(initial.run.status, TaskRunStatus::Paused);
        assert_eq!(initial.run.counters.llm_attempts, 1);
        let rebuilt = Arc::new(parking_lot::Mutex::new(None));
        let witness = rebuilt.clone();
        let host_state = state.clone();
        let host = RebuildHost::new(
            "rg-last-slot.v1",
            &FreshRuntime::new(agent.clone(), vec![provider.clone()]),
            Box::new(move || {
                let state = host_state.clone();
                let witness = witness.clone();
                Box::pin(async move {
                    let provider = RecordingProvider::new(&[]);
                    *witness.lock() = Some(provider.clone());
                    FreshRuntime::new(domain_agent(state, provider.clone(), false), vec![provider])
                })
            }),
        );
        drop(agent);
        let (runner, reviewed) =
            restore(runner, store.clone(), &initial.run, config, "rg.v1", host).await;
        let done = runner
            .resume(
                &reviewed.run.key.run_id,
                reviewed.run.revision,
                TaskResumeInput::Approval {
                    request_id: reviewed.run.pending.as_ref().unwrap().id.clone(),
                    result: ai_agents_hitl::ApprovalResult::Approved,
                },
            )
            .await
            .unwrap();
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(done.run.progress.observation_sequence, 1);
        assert_eq!(done.run.counters.llm_attempts, 1);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            rebuilt
                .lock()
                .as_ref()
                .unwrap()
                .calls
                .load(Ordering::SeqCst),
            0
        );
        assert_eq!(state.observed.lock().len(), 1);
    }
}

struct SourceEdit(std::path::PathBuf);
#[async_trait]
impl Tool for SourceEdit {
    // The fixture edits one host-owned source only; a model cannot select an unrelated path.
    fn id(&self) -> &str {
        "source_edit"
    }
    // Display identity grants nothing outside the ordinary tool registry.
    fn name(&self) -> &str {
        "Source edit"
    }
    // This is an isolated source mutation, not a test response claiming that mutation occurred.
    fn description(&self) -> &str {
        "Replace the isolated test source"
    }
    // Source text is explicit structured input.
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"content":{"type":"string"}},"required":["content"]})
    }
    // The shared executor owns admission and mutation invalidation around the real write.
    async fn execute(&self, arguments: Value, _: ToolExecutionContext) -> ToolResult {
        match std::fs::write(&self.0, arguments["content"].as_str().unwrap_or_default()) {
            Ok(()) => ToolResult::ok("source replaced"),
            Err(error) => ToolResult::error(error.to_string()),
        }
    }
    // Unknown footprints remain unsupported when a hard write-target cap is requested.
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata::default()
    }
}
struct SourceTests {
    path: std::path::PathBuf,
    results: Arc<parking_lot::Mutex<Vec<i32>>>,
}
#[async_trait]
impl ai_agents_tools::CommandRunner for SourceTests {
    // The actual host test reads the isolated source instead of trusting a requested command name or returning a canned Pass.
    async fn run_command(
        &self,
        request: ai_agents_tools::CommandRequest,
        _: ToolExecutionContext,
    ) -> ai_agents_tools::CommandResponse {
        assert_eq!(request.argv, vec!["fixture-test"]);
        let code =
            i32::from(std::fs::read_to_string(&self.path).unwrap() != "fn answer() -> i32 { 2 }");
        self.results.lock().push(code);
        ai_agents_tools::CommandResponse {
            success: code == 0,
            exit_code: Some(code),
            termination: "exited".into(),
            ..Default::default()
        }
    }
}

// Passing tests are invalidated by an actual source mutation; a fresh failing test blocks completion until repaired and revalidated.
#[tokio::test]
async fn rg_code_source_mutation_revalidates_real_tests_and_rejects_unknown_command_footprint() {
    for sqlite in super::continuation_test_support::backends() {
        for capped in [false, true] {
            let directory = std::env::temp_dir().join(format!("rg-code-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&directory).unwrap();
            let path = directory.join("source.rs");
            std::fs::write(&path, "fn answer() -> i32 { 2 }").unwrap();
            let provider = RecordingProvider::new(&[
                "initial verified source",
                r#"{"tool":"source_edit","arguments":{"content":"fn answer() -> i32 { 0 }"}}"#,
                "changed source",
                r#"{"tool":"source_edit","arguments":{"content":"fn answer() -> i32 { 2 }"}}"#,
                "repaired source",
            ]);
            let results = Arc::new(parking_lot::Mutex::new(Vec::new()));
            let agent = Arc::new(crate::AgentBuilder::from_yaml("name: CodeFixture\nsystem_prompt: Repair source and validate it\ntools: [command, source_edit]\ntool_security:\n  enabled: true\n  tools:\n    command:\n      allow_without_confirmation: true\n      allowed_commands: [{argv: [fixture-test]}]\n      working_dirs: [.]\n").unwrap().llm(provider.clone()).auto_configure_features().unwrap().tool(Arc::new(SourceEdit(path.clone()))).build().unwrap());
            agent.set_command_runner(Arc::new(SourceTests {
                path: path.clone(),
                results: results.clone(),
            }));
            let mut config = config(
                "lifecycle: [{id: baseline}, {id: repair}]\nvalidation:\n  checks: [{id: tests, adapter: builtin.command, config: {command: fixture-test}}]\ncompletion: {command_exit: {command: fixture-test, code: 0}}\n",
            );
            config.defaults.max_llm_calls = Some(5);
            let mut ceiling = AutonomyHostCeilings::default();
            if capped {
                config.defaults.max_declared_write_paths = Some(1);
                ceiling.max_declared_write_paths = Some(1);
            }
            let store =
                super::continuation_test_support::store(agent.info().id, "rg.v1", *sqlite).await;
            let runner =
                AutonomyRunner::try_new(agent, config, ceiling, store, "rg.v1".into()).unwrap();
            let done = runner
                .run("repair the isolated source", None)
                .await
                .unwrap();
            if capped {
                assert_ne!(done.run.status, TaskRunStatus::Completed);
                assert!(
                    results.lock().is_empty(),
                    "unknown command footprint must reject before the host body"
                );
                assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            } else {
                assert_eq!(
                    done.run.status,
                    TaskRunStatus::Completed,
                    "{:?}",
                    done.run.stop_reason
                );
                assert_eq!(&*results.lock(), &[0, 1, 0]);
                assert_eq!(done.run.counters.turns, 3);
                assert_eq!(provider.calls.load(Ordering::SeqCst), 5);
                assert_eq!(
                    std::fs::read_to_string(&path).unwrap(),
                    "fn answer() -> i32 { 2 }"
                );
            }
            std::fs::remove_dir_all(directory).unwrap();
        }
    }
}

// Raw configuration and lower-level extension binding both reject framework journal collisions before any runtime owner or model call exists.
#[tokio::test]
async fn rg_reserved_check_names_reject_before_effects_and_bindings() {
    for id in [
        "runtime.telemetry",
        "runtime.controller",
        "runtime.tool_batch",
    ] {
        let provider = RecordingProvider::new(&["must not execute"]);
        let agent = Arc::new(
            crate::AgentBuilder::new()
                .system_prompt("preflight")
                .llm(provider.clone())
                .build()
                .unwrap(),
        );
        let config = config(&format!(
            "validation:\n  checks: [{{id: '{id}', adapter: builtin.evidence, config: {{assertion: {{path: ready, eq: true}}}}}}]\ncompletion: {{validator_passed: '{id}'}}\n"
        ));
        let store = super::continuation_test_support::store(agent.info().id, "rg.v1", false).await;
        assert!(
            AutonomyRunner::try_new(
                agent.clone(),
                config.clone(),
                Default::default(),
                store.clone(),
                "rg.v1".into()
            )
            .is_err()
        );
        assert!(
            AutonomyExtensions::builtins()
                .freeze()
                .bind(&config.defaults, &ValidationCapabilities::default())
                .is_err()
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        assert!(store.list().await.unwrap().is_empty());
        assert!(agent.set_context("still ordinary", json!(true)).is_ok());
    }
}

// A collector attached after provider construction cannot report measured zero for calls that were never instrumented.
#[tokio::test]
async fn rg_uninstrumented_provider_cannot_complete_from_zero_telemetry() {
    let provider = RecordingProvider::new(&["actual unobserved call"]);
    let mut observed = ai_agents_observability::ObservabilityConfig {
        enabled: true,
        ..Default::default()
    };
    observed.export.write_report = false;
    let agent = Arc::new(
        crate::AgentBuilder::new()
            .system_prompt("unwrapped provider")
            .llm(provider.clone())
            .build()
            .unwrap()
            .with_observability(ai_agents_observability::ObservabilityManager::new(observed)),
    );
    let (runner,_) = runner(agent,config("completion: {observability: {path: summary.total_llm_calls, eq: 0}}\npremature_finish: {action: fail}\n"),false).await;
    let done = runner
        .run("do not accept incomplete instrumentation", None)
        .await
        .unwrap();
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert_eq!(done.run.counters.llm_attempts, 1);
    assert_ne!(done.run.status, TaskRunStatus::Completed);
    assert!(
        done.run
            .evidence
            .records
            .iter()
            .any(|record| record.data["gate"] == "unknown")
    );
}

// A real provider/judge pair iterates without any tools or todos and spends the same allowance.
#[tokio::test]
async fn rg_tool_free_judge_revisions_share_exact_allowance() {
    for sqlite in super::continuation_test_support::backends() {
        for cap in [3, 4] {
            let provider = RecordingProvider::new(&[
                "draft: unsupported",
                "{\"overall_score\":0.2}",
                "revision: supported",
                "{\"overall_score\":1.0}",
            ]);
            let agent = Arc::new(
                crate::AgentBuilder::new()
                    .system_prompt("analysis")
                    .llm(provider.clone())
                    .build()
                    .unwrap(),
            );
            let mut config = config(
                "validation:\n  checks: [{id: quality, adapter: builtin.judge, config: {criteria: [grounded analysis], pass_threshold: 0.8}}]\ncompletion: {validator_passed: quality}\n",
            );
            config.defaults.max_llm_calls = Some(cap);
            let (runner, _) = runner(agent.clone(), config, *sqlite).await;
            let done = runner.run("revise the analysis", None).await.unwrap();
            assert_eq!(provider.calls.load(Ordering::SeqCst), cap as usize);
            assert_eq!(done.run.status == TaskRunStatus::Completed, cap == 4);
            if cap == 4 {
                let inputs = provider.captured_messages();
                assert!(
                    inputs[3]
                        .iter()
                        .any(|message| message.content.contains("revision: supported"))
                );
            }
            assert!(agent.set_context("released", json!(true)).is_ok());
        }
    }
}

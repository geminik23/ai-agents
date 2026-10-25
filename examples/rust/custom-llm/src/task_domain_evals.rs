// Offline host integration for the ordinary EvalRunner, not another task executor.
// Flow: suite/YAML -> attempt-local domain tools and adapters -> shared controller -> assertions -> redacted reports.
// Use Rust to bind real domain meaning; replace this isolated state with your authorized application's host adapters.

use ai_agents::autonomy::*;
use ai_agents::tools::{Tool, ToolExecutionContext, ToolResult, ToolSafetyMetadata};
use ai_agents::{AgentBuilder, AgentError};
use ai_agents_eval::fixtures::AttemptFixtureContext;
use ai_agents_eval::{EvalHost, EvalRunner, EvalRunnerOptions};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

struct Records {
    value: Mutex<Value>,
    revision: AtomicU64,
}
impl Records {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            value: Mutex::new(json!({"sources":[],"ticket":"open","incident":null,
            "health":"down","report":{"title":17},"world":0,"source":"bad"})),
            revision: AtomicU64::new(0),
        })
    }
    fn read(&self) -> Value {
        self.value.lock().expect("local fixture mutex").clone()
    }
}
impl HostRevisionSource for Records {
    fn revision(&self) -> ai_agents::Result<String> {
        Ok(self.revision.load(Ordering::SeqCst).to_string())
    }
}
struct RecordTool(Arc<Records>);
#[async_trait]
impl Tool for RecordTool {
    fn id(&self) -> &str {
        "domain"
    }
    fn name(&self) -> &str {
        "Domain records"
    }
    fn description(&self) -> &str {
        "Update isolated records; no command, network or filesystem effects."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","additionalProperties":false,"properties":{
            "op":{"type":"string","enum":["source","resolve","incident","report","world","code"]},"value":{}},"required":["op"]})
    }
    // Admitted bodies change actual host state and revision; a model completion claim changes neither.
    async fn execute(&self, args: Value, _: ToolExecutionContext) -> ToolResult {
        let mut data = self.0.value.lock().expect("local fixture mutex");
        match args["op"].as_str() {
            Some("source") if args["value"].is_string() => data["sources"]
                .as_array_mut()
                .unwrap()
                .push(args["value"].clone()),
            Some("resolve") => data["ticket"] = json!("resolved"),
            Some("incident") => data["incident"] = json!({"id":"fixture-incident","status":"open"}),
            Some("report") => data["report"] = args["value"].clone(),
            Some("world") if args["value"].is_number() => data["world"] = args["value"].clone(),
            Some("code") if args["value"].is_string() => data["source"] = args["value"].clone(),
            _ => return ToolResult::error("invalid record operation"),
        }
        self.0.revision.fetch_add(1, Ordering::SeqCst);
        ToolResult::ok(data.to_string())
    }
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata::default()
    }
}

struct RecordValidator {
    descriptor: AdapterDescriptor,
    records: Arc<Records>,
}
impl AutonomyValidator for RecordValidator {
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.descriptor
    }
    fn validate_config(&self, config: &Value) -> ai_agents::Result<()> {
        if config.as_object().is_none_or(|object| {
            object
                .keys()
                .any(|key| !matches!(key.as_str(), "domain" | "min_unique"))
        }) || !matches!(
            config["domain"].as_str(),
            Some("research" | "support" | "report" | "operations" | "health" | "world" | "code")
        ) || config
            .get("min_unique")
            .is_some_and(|value| value.as_u64().is_none_or(|value| value == 0))
        {
            return Err(AgentError::Config(
                "invalid host.records configuration".into(),
            ));
        }
        Ok(())
    }
    // The frozen host revision binds this pure snapshot to the check; only the framework can adopt its result.
    fn evaluate(&self, input: &ValidationInput<'_>) -> ai_agents::Result<ValidationDecision> {
        let data = self.records.read();
        let unique = data["sources"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect::<BTreeSet<_>>()
            .len();
        let passed = match input.config["domain"].as_str().unwrap() {
            "research" => {
                let cited = input.evidence.current.as_ref().is_some_and(|current| {
                    let response = serde_json::from_str::<Value>(&current.value.response).ok();
                    response
                        .as_ref()
                        .and_then(|response| response["sources"].as_array())
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
                });
                unique >= input.config["min_unique"].as_u64().unwrap_or(2) as usize && cited
            }
            "support" => data["ticket"] == "resolved" || data["incident"]["status"] == "open",
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
            "world" => data["world"].as_i64().is_some_and(|value| value >= 10),
            "code" => data["source"] == "return 2",
            _ => unreachable!("validated configuration"),
        };
        Ok(ValidationDecision::Complete {
            outcome: if passed {
                GateOutcome::Pass
            } else {
                GateOutcome::Fail
            },
            reason: "observed domain predicate".into(),
            metrics: json!({"unique":unique,"world":data["world"],
                "artifact":{"path":"report://current","exists":!data["report"].is_null()}}),
            evidence_refs: vec![],
        })
    }
    fn artifacts(&self, result: &ValidationResult) -> ai_agents::Result<Vec<ArtifactObservation>> {
        Ok(vec![serde_json::from_value(
            result.metrics["artifact"].clone(),
        )?])
    }
}
struct FileReportValidator(AdapterDescriptor);
impl AutonomyValidator for FileReportValidator {
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.0
    }
    fn validate_config(&self, config: &Value) -> ai_agents::Result<()> {
        if config.as_object().is_none_or(|config| config.len() != 1)
            || config["path"].as_str().is_none_or(str::is_empty)
        {
            return Err(AgentError::Config(
                "host.file_report requires one nonempty path".into(),
            ));
        }
        Ok(())
    }
    // File I/O stays in the shared authorized executor; this callback only validates its exact acknowledged content observation.
    fn evaluate(&self, input: &ValidationInput<'_>) -> ai_agents::Result<ValidationDecision> {
        let Some(observation) = input.observations.get("content") else {
            return Ok(ValidationDecision::NeedObservations {
                requests: vec![ValidationObservationRequest::Tool {
                    id: "content".into(),
                    tool: "file_read".into(),
                    arguments: json!({"path":input.config["path"]}),
                    host_binding: None,
                }],
                checkpoint: Value::Null,
            });
        };
        let observed = match &observation.result {
            ObservationResult::Tool { record }
                if tool_record_eligible(record)
                    && record.success
                    && !record.output_truncated
                    && record.arguments["path"] == input.config["path"]
                    && record.executed_arguments["path"] == input.config["path"] =>
            {
                serde_json::from_str::<Value>(&record.output).ok()
            }
            _ => None,
        };
        let report = observed
            .as_ref()
            .and_then(|value| value["content"].as_str())
            .and_then(|content| serde_json::from_str::<Value>(content).ok());
        let valid = report.as_ref().is_some_and(|report| {
            report["title"].is_string()
                && report["rows"]
                    .as_array()
                    .is_some_and(|rows| rows.len() == 2 && rows.iter().all(Value::is_number))
        });
        Ok(ValidationDecision::Complete {
            outcome: if observed.is_none() {
                GateOutcome::Unknown
            } else if valid {
                GateOutcome::Pass
            } else {
                GateOutcome::Fail
            },
            reason: "observed report schema".into(),
            metrics: json!({"valid":valid}),
            evidence_refs: vec!["content".into()],
        })
    }
}

struct RecordProgress(AdapterDescriptor);
impl ProgressAdapter for RecordProgress {
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.0
    }
    fn validate_config(&self, config: &Value) -> ai_agents::Result<()> {
        if config.as_object().is_none_or(|config| !config.is_empty()) {
            return Err(AgentError::Config("host.record_progress expects {}".into()));
        }
        Ok(())
    }
    // Only a current eligible validator capture can advance a signal; repeated source IDs remain unchanged.
    fn observe(&self, input: &ProgressInput<'_>) -> ai_agents::Result<ProgressObservation> {
        let mut signals = BTreeMap::new();
        let mut comparisons = BTreeMap::new();
        let mut refs = Vec::new();
        if let Some(result) = latest_validation(&input.evidence.validations, "records", input.scope)
            && result.outcome != GateOutcome::Unknown
        {
            signals.insert("unique_sources".into(), result.metrics["unique"].clone());
            comparisons.insert("unique_sources".into(), SignalComparison::Increase);
            refs.push(format!("check:records:{}", result.identity.sequence));
        }
        Ok(ProgressObservation {
            metrics: json!({"signals":signals}),
            signals,
            comparisons,
            evidence_refs: refs,
            validation_dependencies: BTreeMap::new(),
        })
    }
}
struct DomainHost;
#[async_trait]
impl EvalHost for DomainHost {
    // Every initial build/reset receives fresh isolated business state; task resume keeps the admitted runtime and this same state.
    fn configure(
        &self,
        builder: AgentBuilder,
        _: &AttemptFixtureContext,
    ) -> ai_agents_eval::Result<AgentBuilder> {
        let records = Records::new();
        let mut extensions = AutonomyExtensions::builtins();
        extensions.register_validator(Arc::new(FileReportValidator(AdapterDescriptor {
            id: "host.file_report".into(),
            contract_version: 1,
            tools: vec!["file_read".into()],
            ..Default::default()
        })))?;
        extensions.register_validator(Arc::new(RecordValidator {
            descriptor: AdapterDescriptor {
                id: "host.records".into(),
                contract_version: 1,
                requires_host_revision: true,
                ..Default::default()
            },
            records: records.clone(),
        }))?;
        extensions.register_progress(Arc::new(RecordProgress(AdapterDescriptor {
            id: "host.record_progress".into(),
            contract_version: 1,
            checks: vec!["records".into()],
            ..Default::default()
        })))?;
        for check in ["records", "health"] {
            extensions.register_validation_target(check, "domain-state", records.clone())?;
        }
        Ok(builder
            .tool(Arc::new(RecordTool(records)))
            .autonomy_extensions(extensions)
            .automatic_task_options(AutomaticTaskOptions {
                ceilings: AutonomyHostCeilings {
                    max_declared_write_paths: Some(1),
                    ..Default::default()
                },
                ..Default::default()
            }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [suite, output] = args.as_slice() else {
        return Err("usage: task-domain-evals SUITE OUTPUT".into());
    };
    // This is exactly the library/CLI eval runner, with explicit caller-owned host integration and default redacted output.
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

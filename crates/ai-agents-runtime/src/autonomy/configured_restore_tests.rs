//! Configured factory acceptance uses actual SQLite, component state and independent processes.

use super::*;
use crate::{AgentBuilder, spec::PreparedAgentSpec};
use ai_agents_core::{
    AgentError, AgentStorage, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
    Result, Tool, ToolExecutionContext, ToolResult, ToolSafetyMetadata,
};
use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DomainState {
    cursor: usize,
    updates: u64,
    source: Option<String>,
}

struct InstalledState {
    directory: PathBuf,
    value: Arc<Mutex<DomainState>>,
}
impl TaskRestoreComponentState for InstalledState {
    // The provider cursor and domain records are behavioral state, while the file log remains an independent execution witness.
    fn binding(&self) -> Result<TaskRestoreBinding> {
        Ok(TaskRestoreBinding {
            id: "configured-process-components".into(),
            contract_version: 1,
            configuration: json!({"program":"effect-question-done", "directory":self.directory}),
            state: serde_json::to_value(&*self.value.lock())?,
        })
    }
}

struct Components {
    directory: PathBuf,
    mode: String,
}
impl TaskRestoreComponents for Components {
    // Failure injection changes constructor availability, not the successful provider program or installed state contract.
    fn identity(&self) -> (String, u32) {
        ("configured-process-components".into(), 1)
    }
    // Validate the real program and isolated data scope before any reconstruction claim or callback.
    fn validate(&self, _: &PreparedAgentSpec, binding: &TaskRestoreBinding) -> Result<()> {
        if (binding.id.clone(), binding.contract_version) != self.identity()
            || binding.configuration
                != json!({"program":"effect-question-done", "directory":self.directory})
        {
            return Err(AgentError::Config("component program changed".into()));
        }
        let _: DomainState = serde_json::from_value(binding.state.clone())?;
        Ok(())
    }
    // Only this registered application code installs executable components; saved JSON supplies data to those components.
    fn install(
        &self,
        _: &PreparedAgentSpec,
        builder: AgentBuilder,
        saved: Option<&TaskRestoreBinding>,
        shared: Option<&ai_agents_llm::LLMRegistry>,
    ) -> Result<(AgentBuilder, Arc<dyn TaskRestoreComponentState>)> {
        if saved.is_some() && self.mode == "crash-constructor" {
            std::process::exit(73);
        }
        let value = Arc::new(Mutex::new(
            saved
                .map(|saved| serde_json::from_value(saved.state.clone()))
                .transpose()?
                .unwrap_or_default(),
        ));
        let builder = if shared.is_none() {
            builder.llm(Arc::new(Program {
                directory: self.directory.clone(),
                value: value.clone(),
            }))
        } else {
            builder
        };
        let mut extensions = AutonomyExtensions::builtins();
        extensions.register_validator(Arc::new(DomainValidator {
            descriptor: AdapterDescriptor {
                id: "host.record".into(),
                contract_version: 1,
                requires_host_revision: true,
                ..Default::default()
            },
            value: value.clone(),
        }))?;
        extensions.register_validation_target(
            "record",
            "domain-record",
            Arc::new(Revision(value.clone())),
        )?;
        let builder = builder
            .auto_configure_features()?
            .tool(Arc::new(Effect {
                directory: self.directory.clone(),
                value: value.clone(),
                crash: self.mode == "crash-effect",
            }))
            .autonomy_extensions(extensions);
        Ok((
            builder,
            Arc::new(InstalledState {
                directory: self.directory.clone(),
                value,
            }),
        ))
    }
    // This constructor only allocates local objects; the crashed child never polls a provider or tool during reconstruction.
    fn validate_interrupted_preparation(&self, _: &TaskRunSnapshot) -> Result<()> {
        Ok(())
    }
}

struct Program {
    directory: PathBuf,
    value: Arc<Mutex<DomainState>>,
}
#[async_trait]
impl LLMProvider for Program {
    // Fixed mock decisions leave tool execution, state mutation, storage and restoration entirely real.
    async fn complete(
        &self,
        _: &[ai_agents_core::ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        let mut state = self.value.lock();
        let response = match state.cursor {
            0 => r#"{"tool":"record_update","arguments":{"source":"persisted-source"}}"#,
            1 => {
                r#"{"tool":"ask_user","arguments":{"question":"Confirm saved record","options":["yes"],"allow_other":false}}"#
            }
            2 => "done",
            _ => {
                return Err(LLMError::Config(
                    "unexpected replay or extra provider work".into(),
                ));
            }
        };
        let mut calls = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.directory.join("calls.log"))
            .map_err(|error| LLMError::Config(error.to_string()))?;
        writeln!(calls, "{}", state.cursor).map_err(|error| LLMError::Config(error.to_string()))?;
        state.cursor += 1;
        Ok(LLMResponse::new(
            response,
            ai_agents_core::FinishReason::Stop,
        ))
    }
    // The selected acceptance path is blocking; an accidental stream fallback is a failure.
    async fn complete_stream(
        &self,
        _: &[ai_agents_core::ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<
        Box<dyn futures::Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
        LLMError,
    > {
        Err(LLMError::Config("unexpected streaming request".into()))
    }
    // This is an explicit mock provider implementation, not a production provider selected by a serialized label.
    fn provider_name(&self) -> &str {
        "configured-process-program"
    }
    // Legacy tool responses exercise the ordinary executor and exact question protocol.
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
}

struct Effect {
    directory: PathBuf,
    value: Arc<Mutex<DomainState>>,
    crash: bool,
}
#[async_trait]
impl Tool for Effect {
    // YAML separately grants the actual installed tool.
    fn id(&self) -> &str {
        "record_update"
    }
    // Presentation does not participate in execution identity.
    fn name(&self) -> &str {
        "Record update"
    }
    // The host fixes the only possible external effect destination inside its isolated test directory.
    fn description(&self) -> &str {
        "Update one isolated record and its execution witness"
    }
    // Arguments cannot choose another path or resource.
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"source":{"type":"string"}},"required":["source"],"additionalProperties":false})
    }
    // This is a real local mutation, not read-only metadata used to bypass admission.
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            default_requires_approval: false,
            ..ToolSafetyMetadata::conservative_unknown()
        }
    }
    // Only an actually invoked body records the effect; constructor and recovery code never call it.
    async fn execute(&self, args: Value, _: ToolExecutionContext) -> ToolResult {
        let Some(source) = args["source"].as_str() else {
            return ToolResult::error("missing source");
        };
        let mut effects = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.directory.join("effects.log"))
            .unwrap();
        writeln!(effects, "effect").unwrap();
        effects.sync_all().unwrap();
        let mut state = self.value.lock();
        state.updates += 1;
        state.source = Some(source.into());
        if self.crash {
            std::process::exit(74);
        }
        ToolResult::ok(json!({"updates":state.updates,"source":state.source}).to_string())
    }
}

struct Revision(Arc<Mutex<DomainState>>);
impl HostRevisionSource for Revision {
    // The predicate tracks actual domain mutation, not provider attempts or a new process's generation counter.
    fn revision(&self) -> Result<String> {
        Ok(self.0.lock().updates.to_string())
    }
}
struct DomainValidator {
    descriptor: AdapterDescriptor,
    value: Arc<Mutex<DomainState>>,
}
impl AutonomyValidator for DomainValidator {
    // The host revision is an actual installed dependency of this domain check.
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.descriptor
    }
    // Domain configuration is intentionally closed for this representative consumer.
    fn validate_config(&self, config: &Value) -> Result<()> {
        if config != &json!({}) {
            return Err(AgentError::Config(
                "unexpected record validator configuration".into(),
            ));
        }
        Ok(())
    }
    // A final model answer cannot replace a missing or reset domain record.
    fn evaluate(&self, _: &ValidationInput<'_>) -> Result<ValidationDecision> {
        let value = self.value.lock();
        Ok(ValidationDecision::Complete {
            outcome: if value.updates == 1 && value.source.as_deref() == Some("persisted-source") {
                GateOutcome::Pass
            } else {
                GateOutcome::Fail
            },
            reason: "actual record state".into(),
            metrics: json!({"updates":value.updates,"source":value.source}),
            evidence_refs: vec![],
        })
    }
}

// Every process creates the same configured host; only the installed component's controlled failure availability varies.
fn host(directory: &Path, mode: &str) -> Arc<ConfiguredTaskRestoreHost> {
    let prepared = PreparedAgentSpec::from_yaml_strict(
        &std::fs::read_to_string(directory.join("agent.yaml")).unwrap(),
    )
    .unwrap();
    Arc::new(
        ConfiguredTaskRestoreHost::new(prepared, directory.into(), None, Duration::from_secs(5))
            .unwrap()
            .components(Arc::new(Components {
                directory: directory.into(),
                mode: mode.into(),
            }))
            .unwrap(),
    )
}

// The real configured factory and SQLite backend remain unchanged; this transport drops exactly one committed acknowledgement.
struct LostAckStore {
    inner: Arc<ScopedTaskRunStore>,
    review: bool,
    armed: std::sync::atomic::AtomicBool,
}
#[async_trait]
impl TaskRunStore for LostAckStore {
    // Creation is unrelated to the selected reconstruction acknowledgement.
    async fn create(&self, snapshot: &TaskRunSnapshot) -> Result<()> {
        self.inner.create(snapshot).await
    }
    // Inspection must expose the actually committed revision after acknowledgement loss.
    async fn load(&self, id: &str) -> Result<Option<TaskRunSnapshot>> {
        self.inner.load(id).await
    }
    // Only an approved review's final publication loses its acknowledgement; claim and work records stay real.
    async fn mutate(&self, id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot> {
        let selected = self.review
            && matches!(mutation, TaskRunMutation::FinalCheckpoint { status: TaskRunStatus::Paused, payload, .. }
            if payload["adapters"].as_array().is_some_and(|items| items.iter().any(|item| item["id"] == "runtime.restore_review" && item["state"]["approved"] == true)));
        let saved = self.inner.mutate(id, mutation).await?;
        if selected && self.armed.swap(false, std::sync::atomic::Ordering::SeqCst) {
            return Err(AgentError::Persistence(
                "committed configured review ACK lost".into(),
            ));
        }
        Ok(saved)
    }
    // Preserve the backend's certified no-write claim result.
    async fn claim_paused(
        &self,
        id: &str,
        revision: u64,
        token: String,
    ) -> Result<Option<TaskRunSnapshot>> {
        self.inner.claim_paused(id, revision, token).await
    }
    // Capabilities describe the actual delegated SQLite implementation.
    fn supports_reconstruction(&self) -> bool {
        self.inner.supports_reconstruction()
    }
    // The wrapper has no independent journal or in-memory lifetime substitution.
    fn supports_process_restart(&self) -> bool {
        self.inner.supports_process_restart()
    }
    // Commit the native host's real graph before simulating transport acknowledgement loss.
    async fn checkpoint_reconstruction(
        &self,
        id: &str,
        checkpoint: &TaskReconstructionCheckpoint,
    ) -> Result<TaskRunSnapshot> {
        let saved = self.inner.checkpoint_reconstruction(id, checkpoint).await?;
        if !self.review && self.armed.swap(false, std::sync::atomic::Ordering::SeqCst) {
            return Err(AgentError::Persistence(
                "committed configured reconstruction ACK lost".into(),
            ));
        }
        Ok(saved)
    }
    // Metadata remains scoped by the real backend.
    async fn list(&self) -> Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    // Preserve exact revision deletion checks.
    async fn delete(&self, id: &str, revision: u64) -> Result<()> {
        self.inner.delete(id, revision).await
    }
}

// Child processes get no customer input, credentials, provider service or test-only production switch.
async fn worker(directory: &Path, mode: &str) -> std::process::Output {
    let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "autonomy::configured_restore_tests::configured_process_worker",
            "--nocapture",
        ])
        .current_dir(directory)
        .env("AI_AGENTS_CONFIGURED_DIRECTORY", directory)
        .env("AI_AGENTS_CONFIGURED_MODE", mode)
        .kill_on_drop(true);
    tokio::time::timeout(Duration::from_secs(30), command.output())
        .await
        .unwrap()
        .unwrap()
}

// This helper runs only when selected by the parent's process invocation; normal test discovery is not an additional acceptance case.
#[tokio::test]
async fn configured_process_worker() {
    let Ok(mode) = std::env::var("AI_AGENTS_CONFIGURED_MODE") else {
        return;
    };
    let directory = PathBuf::from(std::env::var_os("AI_AGENTS_CONFIGURED_DIRECTORY").unwrap());
    let host = host(&directory, &mode);
    if matches!(mode.as_str(), "seed" | "crash-effect") {
        let agent = host.build().await.unwrap();
        let paused = agent
            .run_task("update and confirm the record", None)
            .await
            .unwrap();
        assert_eq!(paused.run.status, TaskRunStatus::Paused);
        assert!(
            agent
                .task_runner()
                .unwrap()
                .prepared_restart_available(&paused.run.key.run_id, paused.run.revision)
                .await
                .unwrap()
        );
        std::fs::write(
            directory.join("request.json"),
            serde_json::to_vec(&json!({"id":paused.run.key.run_id,"revision":paused.run.revision}))
                .unwrap(),
        )
        .unwrap();
        return;
    }
    let request: Value =
        serde_json::from_slice(&std::fs::read(directory.join("request.json")).unwrap()).unwrap();
    let id = request["id"].as_str().unwrap().to_string();
    let revision = request["revision"].as_u64().unwrap();
    if matches!(mode.as_str(), "ack-restore" | "ack-review") {
        let inner = host.scope().open_store().await.unwrap();
        let fault = Arc::new(LostAckStore {
            inner: inner.clone(),
            review: mode == "ack-review",
            armed: std::sync::atomic::AtomicBool::new(true),
        });
        let result = AutonomyRunner::restore_paused(
            TaskRestoreRequest {
                run_id: id.clone(),
                expected_revision: revision,
                config_identity: host.scope().identity().into(),
                config: PreparedAgentSpec::from_yaml_strict(
                    &std::fs::read_to_string(directory.join("agent.yaml")).unwrap(),
                )
                .unwrap()
                .spec()
                .autonomy
                .clone(),
                ceilings: Default::default(),
            },
            fault.clone(),
            host.clone(),
        )
        .await;
        if mode == "ack-review" {
            let (runner, paused) = result.unwrap();
            let review = runner.restoration_review(&id).await.unwrap().unwrap();
            assert!(
                runner
                    .approve_restoration(
                        &id,
                        paused.run.revision,
                        &review.id,
                        ai_agents_hitl::ApprovalResult::Approved
                    )
                    .await
                    .is_err()
            );
            let saved = inner.load(&id).await.unwrap().unwrap();
            assert!(
                runner
                    .resume(
                        &id,
                        saved.revision,
                        TaskResumeInput::UserAnswer {
                            request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                            answer: json!({"answered":true,"selected":["yes"]}),
                        }
                    )
                    .await
                    .is_err()
            );
        } else {
            assert!(result.is_err());
        }
        assert!(!fault.armed.load(std::sync::atomic::Ordering::SeqCst));
        let saved = inner.load(&id).await.unwrap().unwrap();
        assert_eq!(saved.status, TaskRunStatus::Paused);
        assert!(saved.revision > revision);
        assert!(saved.owner_token.is_none());
        assert_eq!(
            std::fs::read_to_string(directory.join("calls.log"))
                .unwrap()
                .lines()
                .count(),
            2
        );
        assert_eq!(
            std::fs::read_to_string(directory.join("effects.log"))
                .unwrap()
                .lines()
                .count(),
            1
        );
        std::fs::write(
            directory.join("request.json"),
            serde_json::to_vec(&json!({"id":id,"revision":saved.revision})).unwrap(),
        )
        .unwrap();
        return;
    }
    if mode == "recover" {
        let recovered = host
            .recover_prepared(
                id,
                revision,
                TaskRecoveryDisposition::ResumePrepared,
                "confirmed interrupted constructor with no semantic work".into(),
            )
            .await
            .unwrap();
        assert_eq!(recovered.status, TaskRunStatus::Paused);
        std::fs::write(
            directory.join("request.json"),
            serde_json::to_vec(&json!({"id":recovered.key.run_id,"revision":recovered.revision}))
                .unwrap(),
        )
        .unwrap();
        return;
    }
    let (runner, paused) = host.restore(id, revision).await.unwrap();
    assert_eq!(
        std::fs::read_to_string(directory.join("calls.log"))
            .unwrap()
            .lines()
            .count(),
        2
    );
    let review = runner
        .restoration_review(&paused.run.key.run_id)
        .await
        .unwrap()
        .unwrap();
    let reviewed = runner
        .approve_restoration(
            &paused.run.key.run_id,
            paused.run.revision,
            &review.id,
            ai_agents_hitl::ApprovalResult::Approved,
        )
        .await
        .unwrap();
    let done = runner
        .resume(
            &reviewed.run.key.run_id,
            reviewed.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: reviewed.run.pending.as_ref().unwrap().id.clone(),
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(done.run.counters.llm_attempts, 3);
    let questions: Vec<_> = done
        .run
        .evidence
        .tool_calls
        .iter()
        .filter(|entry| entry.record.canonical_id == "ask_user" && entry.record.executed)
        .collect();
    assert_eq!(questions.len(), 1);
    let answer: Value = serde_json::from_str(&questions[0].record.output).unwrap();
    assert_eq!(answer["answered"], true);
    assert_eq!(answer["selected"], json!(["yes"]));

    assert_eq!(
        std::fs::read_to_string(directory.join("effects.log"))
            .unwrap()
            .lines()
            .count(),
        1
    );
}

// Parent inspection reads the actual backend after each child exits; it never edits or fabricates a recovery checkpoint.
async fn snapshot(directory: &Path) -> TaskRunSnapshot {
    let storage =
        ai_agents_storage::SqliteStorage::new(directory.join("tasks.sqlite").to_str().unwrap())
            .await
            .unwrap();
    let name = PreparedAgentSpec::from_yaml_strict(
        &std::fs::read_to_string(directory.join("agent.yaml")).unwrap(),
    )
    .unwrap()
    .spec()
    .name
    .clone();
    let runs = storage
        .list_task_runs(&TaskRunFilter {
            agent_id: name,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(runs.len(), 1);
    let snapshot = storage.load_task_run(&runs[0].key).await.unwrap().unwrap();
    storage.close().await;
    snapshot
}

// This checks successful recovery as well as refusal while a worker lease is held and refusal after an unacknowledged real effect.
#[tokio::test]
async fn configured_process_restart_and_interrupted_preparation_preserve_domain_state() {
    for crash_effect in [false, true] {
        let directory =
            std::env::temp_dir().join(format!("configured-restoration-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(directory.join("agent.yaml"), format!(
            "name: ConfiguredProcess\nsystem_prompt: Complete the declared work.\ntools: [record_update, ask_user]\nautonomy:\n  enabled: true\n  restoration_timeout_ms: 5000\n  persistence: {{enabled: true, resume_on_start: true}}\n  hitl: {{on_user_question: pause_run, on_approval_required: pause_run}}\n  validation:\n    checks: [{{id: record, adapter: host.record, config: {{}}}}]\n  completion: {{all: [{{response_contains: done}}, {{validator_passed: record}}]}}\nstorage: {{type: sqlite, path: '{}' }}\n", directory.join("tasks.sqlite").display()
        )).unwrap();
        let seed = worker(
            &directory,
            if crash_effect { "crash-effect" } else { "seed" },
        )
        .await;
        assert_eq!(
            seed.status.code(),
            Some(if crash_effect { 74 } else { 0 }),
            "{}",
            String::from_utf8_lossy(&seed.stderr)
        );
        if !crash_effect {
            for boundary in ["ack-restore", "ack-review"] {
                let lost = worker(&directory, boundary).await;
                assert!(
                    lost.status.success(),
                    "{}",
                    String::from_utf8_lossy(&lost.stderr)
                );
            }
            let crash = worker(&directory, "crash-constructor").await;
            assert_eq!(
                crash.status.code(),
                Some(73),
                "{}",
                String::from_utf8_lossy(&crash.stderr)
            );
        }
        let before = snapshot(&directory).await;
        assert_eq!(before.status, TaskRunStatus::Running);
        std::fs::write(
            directory.join("request.json"),
            serde_json::to_vec(&json!({"id":before.key.run_id,"revision":before.revision}))
                .unwrap(),
        )
        .unwrap();
        let payload: TaskCheckpointPayload =
            serde_json::from_value(before.payload.clone()).unwrap();
        let lease = super::restore_lease::TaskFileLease::acquire(
            &directory.join("tasks.sqlite"),
            &payload.config_identity,
            &before.key.run_id,
        )
        .unwrap();
        let held = worker(&directory, "recover").await;
        assert!(!held.status.success());
        assert!(
            String::from_utf8_lossy(&held.stderr).contains("Owned"),
            "{}",
            String::from_utf8_lossy(&held.stderr)
        );
        assert_eq!(
            serde_json::to_value(snapshot(&directory).await).unwrap(),
            serde_json::to_value(&before).unwrap()
        );
        drop(lease);
        let recovered = worker(&directory, "recover").await;
        if crash_effect {
            assert!(
                !recovered.status.success(),
                "an OS lease cannot certify the unacknowledged tool effect"
            );
            assert!(
                String::from_utf8_lossy(&recovered.stderr).contains("NotResumable"),
                "{}",
                String::from_utf8_lossy(&recovered.stderr)
            );
            assert_eq!(
                serde_json::to_value(snapshot(&directory).await).unwrap(),
                serde_json::to_value(&before).unwrap()
            );
            assert_eq!(
                std::fs::read_to_string(directory.join("calls.log"))
                    .unwrap()
                    .lines()
                    .count(),
                1
            );
        } else {
            assert!(
                recovered.status.success(),
                "{}",
                String::from_utf8_lossy(&recovered.stderr)
            );
            let ready = snapshot(&directory).await;
            assert_eq!(ready.status, TaskRunStatus::Paused);
            assert!(ready.owner_token.is_none());
            assert_eq!(ready.payload["counters"], before.payload["counters"]);
            let resumed = worker(&directory, "resume").await;
            assert!(
                resumed.status.success(),
                "{}",
                String::from_utf8_lossy(&resumed.stderr)
            );
            assert_eq!(
                std::fs::read_to_string(directory.join("calls.log"))
                    .unwrap()
                    .lines()
                    .count(),
                3
            );
        }
        assert_eq!(
            std::fs::read_to_string(directory.join("effects.log"))
                .unwrap()
                .lines()
                .count(),
            1
        );
        crate::remove_sqlite_test_directory(&directory)
            .await
            .unwrap();
    }
}

// Configuring restoration plumbing does not make storage mandatory for ordinary volatile tasks; a later durable override must fail before dispatch.
#[tokio::test]
async fn configured_volatile_host_preserves_no_storage_and_rejects_durable_override_before_work() {
    let directory =
        std::env::temp_dir().join(format!("configured-volatile-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&directory).unwrap();
    std::fs::write(directory.join("agent.yaml"), "name: ConfiguredProcess\nsystem_prompt: Complete the declared work.\ntools: [record_update, ask_user]\nautonomy:\n  enabled: true\n  restoration_timeout_ms: 5000\n  hitl: {on_user_question: pause_run}\n  validation:\n    checks: [{id: record, adapter: host.record, config: {}}]\n  completion: {all: [{response_contains: done}, {validator_passed: record}]}\n").unwrap();
    let host = host(&directory, "normal");
    let agent = host.build().await.unwrap();
    let overrides = AutonomyProfile {
        persistence: Some(ai_agents_core::autonomy::AutonomyPersistenceConfig {
            enabled: Some(true),
            ..Default::default()
        }),
        ..Default::default()
    };
    assert!(
        agent
            .run_task_with_overrides("record", None, &overrides)
            .await
            .is_err()
    );
    assert!(!directory.join("calls.log").exists());
    assert!(!directory.join("effects.log").exists());
    assert!(
        agent
            .task_runner()
            .unwrap()
            .task_store()
            .list()
            .await
            .unwrap()
            .is_empty()
    );
    let paused = agent.run_task("record", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    let runner = agent.task_runner().unwrap();
    assert!(
        !runner
            .prepared_restart_available(&paused.run.key.run_id, paused.run.revision)
            .await
            .unwrap()
    );
    let done = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: paused.run.pending.unwrap().id,
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(done.run.counters.llm_attempts, 3);
    assert!(!directory.join("tasks.sqlite").exists());
    drop((runner, agent, host));
    std::fs::remove_dir_all(directory).unwrap();
}

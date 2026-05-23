use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use ai_agents_observability::ObservabilityConfig;
use ai_agents_observability::config::ExportFormat;
use ai_agents_runtime::spec::{AgentSpec, LLMConfigOrSelector, StorageConfig};
use ai_agents_runtime::{Agent, AgentBuilder, RuntimeAgent, StreamChunk};
use futures::{StreamExt, stream};
use serde_json::{Value, json};
use tokio::time::{Duration, timeout};

use crate::assertion::{AssertionEvalContext, AssertionOutcome, evaluate_assertion};
use crate::compatibility::suite_from_jsonl;
use crate::evidence::{collect_turn_evidence, relationship_snapshot};
use crate::fixtures::{
    LlmFixtureMode, RecordingToolLog, build_llm_registry, build_tool_registry,
    resolve_fixture_context, start_mock_server,
};
use crate::judge::{JudgeConfig, JudgeResolver};
use crate::metrics::compute_metrics;
use crate::redaction::{redact_text, redact_value};
use crate::suite::{
    AttemptResult, EvalResult, EvalSuite, FailureCategory, IsolationMode, ResetStepConfig,
    Scenario, ScenarioResult, ScenarioStatus, ScenarioStep, Turn, TurnResult,
};
use crate::{EvalError, Result};

/// Runtime options supplied by CLI or Rust callers.
#[derive(Debug, Clone, Default)]
pub struct EvalRunnerOptions {
    /// Agent YAML path used for this run.
    pub agent: Option<PathBuf>,
    /// Scenario test cases in this suite.
    pub scenarios: Option<PathBuf>,
    /// Directory where output artifacts are written.
    pub output: PathBuf,
    /// Scenario IDs selected for execution.
    pub ids: Vec<String>,
    /// Tags used by filters and grouped metrics.
    pub tags: Vec<String>,
    /// Whether all selected tags must match.
    pub tag_mode_all: bool,
    /// Language labels selected for execution.
    pub languages: Vec<String>,
    /// Optional retry count or suite retry count.
    pub retries: Option<u32>,
    /// Optional timeout override for this turn.
    pub timeout_ms: Option<u64>,
    /// Optional scenario concurrency override.
    pub parallel: Option<usize>,
    /// Stop after the first failed or errored scenario.
    pub fail_fast: bool,
    /// Observability assertion, setting, or report value.
    pub observability: bool,
    /// Optional LLM fixture mode override.
    pub llm_mode: Option<LlmFixtureMode>,
    /// Optional cassette JSONL file for replay or record mode.
    pub cassette: Option<PathBuf>,
}

/// Parsed suite runner with immutable options and suite state.
pub struct EvalRunner {
    /// Path to the loaded suite file.
    suite_path: PathBuf,
    /// Parsed and validated suite.
    suite: EvalSuite,
    /// Runtime options applied to the suite.
    options: EvalRunnerOptions,
}

impl EvalRunner {
    pub fn from_file(path: impl AsRef<Path>, options: EvalRunnerOptions) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let content = std::fs::read_to_string(&path)?;
        let mut suite = if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            suite_from_jsonl(
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("eval")
                    .to_string(),
                &content,
            )?
        } else {
            serde_yaml::from_str::<EvalSuite>(&content)?
        };
        if let Some(agent) = &options.agent {
            suite.agent = Some(agent.clone());
        }
        if let Some(retries) = options.retries {
            suite.settings.retries = retries;
        }
        if let Some(timeout_ms) = options.timeout_ms {
            suite.settings.timeout_per_turn_ms = timeout_ms;
        }
        if let Some(parallel) = options.parallel {
            suite.settings.parallel = parallel > 1;
            suite.settings.max_concurrent = parallel.max(1);
        }
        if options.fail_fast {
            suite.settings.fail_fast = true;
        }
        if let Some(mode) = options.llm_mode {
            suite.fixtures.llm.mode = mode;
        }
        if let Some(cassette) = &options.cassette {
            suite.fixtures.llm.cassette = Some(cassette.clone());
        }
        suite.validate(options.agent.as_ref())?;
        Ok(Self {
            suite_path: path,
            suite,
            options,
        })
    }

    pub async fn run(&self) -> Result<EvalResult> {
        let start = Instant::now();
        let base_dir = self.suite_path.parent().unwrap_or_else(|| Path::new("."));
        let agent_path = self.resolve_agent_path(base_dir)?;
        let scenarios = self.filtered_scenarios();
        let results = if self.suite.settings.parallel && !self.suite.settings.fail_fast {
            self.run_scenarios_parallel(&agent_path, base_dir, scenarios)
                .await
        } else {
            self.run_scenarios_serial(&agent_path, base_dir, scenarios)
                .await
        };

        let total = results.len();
        let passed = results.iter().filter(|r| r.status.is_passed()).count();
        let failed = results
            .iter()
            .filter(|r| r.status.is_failed() || r.status.is_error())
            .count();
        let skipped = results
            .iter()
            .filter(|r| matches!(r.status, ScenarioStatus::Skipped { .. }))
            .count();
        let metrics = compute_metrics(&results);

        let observability = final_observability_report(&results);

        Ok(EvalResult {
            schema_version: 1,
            suite: self.suite.name.clone(),
            agent: agent_path.display().to_string(),
            total,
            passed,
            failed,
            skipped,
            duration_ms: start.elapsed().as_millis() as u64,
            scenarios: results,
            metrics,
            observability,
        })
    }

    async fn run_scenarios_serial(
        &self,
        agent_path: &Path,
        base_dir: &Path,
        scenarios: Vec<&Scenario>,
    ) -> Vec<ScenarioResult> {
        let mut results = Vec::new();
        for scenario in scenarios {
            let result = self.run_scenario(agent_path, base_dir, scenario).await;
            match result {
                Ok(result) => {
                    let stop = self.suite.settings.fail_fast
                        && (result.status.is_failed() || result.status.is_error());
                    results.push(result);
                    if stop {
                        break;
                    }
                }
                Err(error) => {
                    results.push(error_result(scenario, error, FailureCategory::RuntimeError));
                    if self.suite.settings.fail_fast {
                        break;
                    }
                }
            }
        }
        results
    }

    async fn run_scenarios_parallel(
        &self,
        agent_path: &Path,
        base_dir: &Path,
        scenarios: Vec<&Scenario>,
    ) -> Vec<ScenarioResult> {
        let max_concurrent = self.suite.settings.max_concurrent.max(1);
        let mut indexed = stream::iter(scenarios.into_iter().enumerate())
            .map(|(idx, scenario)| async move {
                let result = self.run_scenario(agent_path, base_dir, scenario).await;
                let result = result.unwrap_or_else(|error| {
                    error_result(scenario, error, FailureCategory::RuntimeError)
                });
                (idx, result)
            })
            .buffer_unordered(max_concurrent)
            .collect::<Vec<_>>()
            .await;
        indexed.sort_by_key(|(idx, _)| *idx);
        indexed.into_iter().map(|(_, result)| result).collect()
    }

    fn resolve_agent_path(&self, base_dir: &Path) -> Result<PathBuf> {
        if let Some(agent) = &self.options.agent {
            return Ok(agent.clone());
        }
        let agent =
            self.suite.agent.clone().ok_or_else(|| {
                EvalError::Config("agent path is required in suite or CLI".into())
            })?;
        Ok(if agent.is_absolute() {
            agent
        } else {
            base_dir.join(agent)
        })
    }

    fn filtered_scenarios(&self) -> Vec<&Scenario> {
        let ids: HashSet<_> = self.options.ids.iter().collect();
        let tags: HashSet<_> = self.options.tags.iter().collect();
        let languages: HashSet<_> = self.options.languages.iter().collect();
        self.suite
            .scenarios
            .iter()
            .filter(|scenario| {
                if !ids.is_empty() && !ids.contains(&scenario.id) {
                    return false;
                }
                if !languages.is_empty() {
                    let Some(language) = &scenario.language else {
                        return false;
                    };
                    if !languages.contains(language) {
                        return false;
                    }
                }
                if !tags.is_empty() {
                    let scenario_tags: HashSet<_> = scenario.tags.iter().collect();
                    if self.options.tag_mode_all {
                        if !tags.iter().all(|tag| scenario_tags.contains(*tag)) {
                            return false;
                        }
                    } else if !tags.iter().any(|tag| scenario_tags.contains(*tag)) {
                        return false;
                    }
                }
                true
            })
            .collect()
    }

    async fn run_scenario(
        &self,
        agent_path: &Path,
        base_dir: &Path,
        scenario: &Scenario,
    ) -> Result<ScenarioResult> {
        let start = Instant::now();
        if scenario.skip.is_skipped() {
            return Ok(ScenarioResult {
                id: scenario.id.clone(),
                name: scenario.name.clone(),
                tags: scenario.tags.clone(),
                language: scenario.language.clone(),
                status: ScenarioStatus::Skipped {
                    reason: scenario.skip.reason(),
                },
                failure_category: None,
                flaky: false,
                attempts: Vec::new(),
                duration_ms: 0,
                retries_used: 0,
            });
        }

        let mut attempts = Vec::new();
        let mut final_status = ScenarioStatus::Failed {
            reason: "not run".to_string(),
        };
        let mut category = Some(FailureCategory::AssertionFailed);
        let max_attempt = self.suite.settings.retries + 1;

        for attempt_idx in 0..max_attempt {
            let attempt_future = self.run_attempt(agent_path, base_dir, scenario, attempt_idx);
            let attempt = if let Some(timeout_ms) = self.suite.settings.timeout_per_scenario_ms {
                match timeout(Duration::from_millis(timeout_ms), attempt_future).await {
                    Ok(result) => result,
                    Err(_) => Err(EvalError::Runtime(format!(
                        "scenario '{}' attempt {} timed out after {}ms",
                        scenario.id, attempt_idx, timeout_ms
                    ))),
                }
            } else {
                attempt_future.await
            };
            match attempt {
                Ok(attempt_result) => {
                    final_status = attempt_result.status.clone();
                    if final_status.is_passed() {
                        attempts.push(attempt_result);
                        category = if attempt_idx > 0 {
                            Some(FailureCategory::FlakyPass)
                        } else {
                            None
                        };
                        break;
                    }
                    category = Some(if final_status.is_error() {
                        FailureCategory::RuntimeError
                    } else {
                        failure_category_for_attempt(&attempt_result)
                    });
                    attempts.push(attempt_result);
                }
                Err(error) => {
                    final_status = ScenarioStatus::Error {
                        message: error.to_string(),
                    };
                    category = Some(FailureCategory::RuntimeError);
                    attempts.push(AttemptResult {
                        attempt: attempt_idx,
                        turns: Vec::new(),
                        status: final_status.clone(),
                        duration_ms: 0,
                    });
                }
            }
            if attempt_idx + 1 < max_attempt {
                tokio::time::sleep(Duration::from_millis(self.suite.settings.retry_delay_ms)).await;
            }
        }

        let flaky = final_status.is_passed() && attempts.len() > 1;
        Ok(ScenarioResult {
            id: scenario.id.clone(),
            name: scenario.name.clone(),
            tags: scenario.tags.clone(),
            language: scenario.language.clone(),
            status: final_status,
            failure_category: category,
            flaky,
            duration_ms: start.elapsed().as_millis() as u64,
            retries_used: attempts.len().saturating_sub(1) as u32,
            attempts,
        })
    }

    async fn run_attempt(
        &self,
        agent_path: &Path,
        base_dir: &Path,
        scenario: &Scenario,
        attempt: u32,
    ) -> Result<AttemptResult> {
        let start = Instant::now();
        let workspace = std::env::temp_dir().join(format!(
            "ai_agents_eval_{}_{}_{}",
            scenario.id,
            attempt,
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&workspace)?;
        let _env_guard = EnvGuard::apply(&scenario.env)?;
        let mock_server = start_mock_server(self.suite.fixtures.mock_server.as_ref()).await?;
        let tool_log = RecordingToolLog::new();
        let mut agent = self
            .build_agent(agent_path, base_dir, &workspace, tool_log.clone())
            .await?;
        apply_base_context(
            &agent,
            &self.suite,
            base_dir,
            scenario,
            mock_server.as_ref(),
        )?;
        let mut turns = Vec::new();
        let mut status = ScenarioStatus::Passed;

        if !scenario.turns.is_empty() {
            for (idx, turn) in scenario.turns.iter().enumerate() {
                let turn_result = self
                    .run_turn(&agent, scenario, turn, idx, &tool_log)
                    .await?;
                let turn_failed = turn_result.assertion_results.iter().any(|r| !r.passed);
                turns.push(turn_result);
                if turn_failed {
                    status = ScenarioStatus::Failed {
                        reason: format!("turn {} assertion failed", idx + 1),
                    };
                    break;
                }
                if self.suite.settings.isolation == IsolationMode::Turn
                    && idx + 1 < scenario.turns.len()
                {
                    agent.reset().await?;
                    apply_base_context(
                        &agent,
                        &self.suite,
                        base_dir,
                        scenario,
                        mock_server.as_ref(),
                    )?;
                }
            }
        }

        for step in &scenario.steps {
            if !status.is_passed() {
                break;
            }
            match step {
                ScenarioStep::Run(run) => {
                    for turn in &run.turns {
                        let idx = turns.len();
                        let turn_result = self
                            .run_turn(&agent, scenario, turn, idx, &tool_log)
                            .await?;
                        let turn_failed = turn_result.assertion_results.iter().any(|r| !r.passed);
                        turns.push(turn_result);
                        if turn_failed {
                            status = ScenarioStatus::Failed {
                                reason: format!("turn {} assertion failed", idx + 1),
                            };
                            break;
                        }
                    }
                    if let Some(session) = &run.save_session {
                        agent.save_session(session).await?;
                    }
                }
                ScenarioStep::ResetAgent(reset) => {
                    if let Some(options) = reset_options(reset) {
                        if options.delete_persistence || !options.preserve_storage {
                            let _ = std::fs::remove_dir_all(&workspace);
                            std::fs::create_dir_all(&workspace)?;
                        }
                        let preserved_actor = options
                            .preserve_actor_id
                            .then(|| agent.actor_id())
                            .flatten();
                        if matches!(options.profile, crate::reset::ResetProfile::Conversation)
                            && !options.delete_persistence
                        {
                            agent.reset().await?;
                        } else {
                            agent = self
                                .build_agent(agent_path, base_dir, &workspace, tool_log.clone())
                                .await?;
                        }
                        if options.preserve_host_context {
                            apply_base_context(
                                &agent,
                                &self.suite,
                                base_dir,
                                scenario,
                                mock_server.as_ref(),
                            )?;
                        }
                        if let Some(actor) = preserved_actor.or_else(|| scenario.actor.clone()) {
                            agent.set_actor_id(&actor)?;
                            agent.load_actor_memory().await?;
                            agent.load_actor_relationship().await?;
                        }
                    }
                }
                ScenarioStep::SaveSession(name) => {
                    agent.save_session(name).await?;
                }
                ScenarioStep::LoadSession(name) => {
                    let _ = agent.load_session(name).await?;
                }
                ScenarioStep::SetContext { values } => {
                    apply_context_value(&agent, values)?;
                }
                ScenarioStep::SetActor { actor } => {
                    agent.set_actor_id(actor)?;
                    agent.load_actor_memory().await?;
                    agent.load_actor_relationship().await?;
                }
                ScenarioStep::CleanupExpired => {
                    let _ = agent.cleanup_expired_sessions().await?;
                }
            }
        }

        Ok(AttemptResult {
            attempt,
            turns,
            status,
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    async fn build_agent(
        &self,
        agent_path: &Path,
        base_dir: &Path,
        workspace: &Path,
        tool_log: RecordingToolLog,
    ) -> Result<RuntimeAgent> {
        let content = std::fs::read_to_string(agent_path)?;
        let mut spec: AgentSpec = serde_yaml::from_str(&content)?;
        apply_eval_llm_settings(&mut spec, &self.suite.settings);
        spec.validate()
            .map_err(|error| EvalError::Config(error.to_string()))?;
        let (llm_registry, _judge_llm) =
            build_llm_registry(&spec, &self.suite.fixtures.llm, base_dir)?;
        let tool_registry = build_tool_registry(&self.suite.fixtures, tool_log)?;
        let mut builder = AgentBuilder::from_yaml_file(agent_path)
            .map_err(|error| EvalError::Config(error.to_string()))?
            .llm_registry(llm_registry)
            .tools(tool_registry)
            .auto_configure_features()
            .map_err(|error| EvalError::Config(error.to_string()))?
            .auto_configure_mcp()
            .await
            .map_err(|error| EvalError::Config(error.to_string()))?;

        let storage_override = isolated_storage_config(&spec, workspace);
        if let Some(storage) = storage_override {
            builder = builder.storage_config(storage);
        }

        if let Some(observability) = self.observability_config(base_dir)? {
            let manager = ai_agents_observability::ObservabilityManager::new(observability);
            builder = builder.observability(manager);
        }
        builder = builder
            .auto_configure_spawner()
            .await
            .map_err(|error| EvalError::Config(error.to_string()))?;
        let agent = builder
            .build()
            .map_err(|error| EvalError::Config(error.to_string()))?;
        agent.init_storage().await?;
        Ok(agent)
    }

    fn observability_config(&self, base_dir: &Path) -> Result<Option<ObservabilityConfig>> {
        let mut config = if let Some(config) = self.suite.observability.clone() {
            config
        } else if self.options.observability {
            let mut config = ObservabilityConfig::default();
            config.enabled = true;
            config.export.formats = vec![ExportFormat::Json];
            config.export.path = self
                .options
                .output
                .join("observability")
                .display()
                .to_string();
            config.export.write_report = true;
            config
        } else {
            return Ok(None);
        };
        if !config.enabled {
            return Ok(None);
        }
        config = config
            .with_pricing_file_loaded(Some(base_dir))
            .map_err(|error| EvalError::Config(error.to_string()))?;
        config
            .validate()
            .map_err(|error| EvalError::Config(error.to_string()))?;
        Ok(Some(config))
    }

    async fn run_turn(
        &self,
        agent: &RuntimeAgent,
        scenario: &Scenario,
        turn: &Turn,
        index: usize,
        tool_log: &RecordingToolLog,
    ) -> Result<TurnResult> {
        apply_context_value(agent, &turn.context)?;
        if let Some(actor) = &turn.actor {
            agent.set_actor_id(actor)?;
        }
        let before_relationship = relationship_snapshot(agent);
        let tool_start = tool_log.len();
        let start = Instant::now();
        let timeout_ms = turn
            .timeout_ms
            .unwrap_or(self.suite.settings.timeout_per_turn_ms);
        let (response_content, response_metadata) = if turn.stream.unwrap_or(false) {
            timeout(
                Duration::from_millis(timeout_ms),
                collect_stream_response(agent, &turn.input),
            )
            .await
            .map_err(|_| EvalError::Runtime(format!("turn timed out after {}ms", timeout_ms)))??
        } else {
            let response = timeout(Duration::from_millis(timeout_ms), agent.chat(&turn.input))
                .await
                .map_err(|_| {
                    EvalError::Runtime(format!("turn timed out after {}ms", timeout_ms))
                })??;
            (response.content, response.metadata)
        };
        let latency_ms = start.elapsed().as_millis() as u64;
        let evidence = collect_turn_evidence(
            agent,
            response_metadata.clone(),
            tool_log,
            tool_start,
            before_relationship,
        );
        let judge = self.build_judge(agent);
        let mut assertion_results = if let Some(assertion) = &turn.assertions {
            match evaluate_assertion(
                assertion,
                AssertionEvalContext {
                    evidence: &evidence,
                    response: &response_content,
                    user_input: Some(&turn.input),
                    scenario_id: Some(&scenario.id),
                    language: scenario.language.as_deref(),
                    judge_resolver: Some(&judge),
                },
            )
            .await
            {
                AssertionOutcome::Passed(details) | AssertionOutcome::Failed(details) => details,
                AssertionOutcome::Error(message) => return Err(EvalError::Assertion(message)),
            }
        } else {
            Vec::new()
        };
        if self.suite.settings.redact_outputs {
            redact_assertion_details(&mut assertion_results);
        }
        let observability_span_id = evidence
            .observability
            .as_ref()
            .and_then(|obs| obs.span_ids.last().cloned());
        Ok(TurnResult {
            index,
            input: redact_text(&turn.input, self.suite.settings.redact_outputs, 0),
            response: redact_text(&response_content, self.suite.settings.redact_outputs, 0),
            state: evidence.state.clone(),
            metadata: if self.suite.settings.redact_outputs {
                None
            } else {
                response_metadata.and_then(|m| serde_json::to_value(m).ok())
            },
            evidence,
            assertion_results,
            latency_ms,
            observability_span_id,
        })
    }

    fn build_judge(&self, agent: &RuntimeAgent) -> JudgeResolver {
        JudgeResolver::new(Arc::clone(agent.llm_registry()), JudgeConfig::default())
    }
}

async fn collect_stream_response(
    agent: &RuntimeAgent,
    input: &str,
) -> Result<(String, Option<HashMap<String, Value>>)> {
    let mut stream = agent.chat_stream(input).await?;
    let mut content = String::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            StreamChunk::Content { text } => content.push_str(&text),
            StreamChunk::Done {} => break,
            StreamChunk::Error { message } => return Err(EvalError::Runtime(message)),
            _ => {}
        }
    }
    Ok((content, None))
}

fn isolated_storage_config(spec: &AgentSpec, workspace: &Path) -> Option<StorageConfig> {
    if spec.storage.is_none() {
        return None;
    }
    match &spec.storage {
        StorageConfig::Sqlite(_) => Some(StorageConfig::sqlite(
            workspace.join("sessions.db").display().to_string(),
        )),
        StorageConfig::File(_) => Some(StorageConfig::file(
            workspace.join("sessions").display().to_string(),
        )),
        StorageConfig::Redis(_) => None,
        StorageConfig::None => None,
    }
}

fn apply_context_map(agent: &RuntimeAgent, values: HashMap<String, Value>) -> Result<()> {
    for (key, value) in values {
        agent.set_context(&key, value)?;
    }
    Ok(())
}

fn apply_context_value(agent: &RuntimeAgent, value: &Value) -> Result<()> {
    let Value::Object(map) = value else {
        return Ok(());
    };
    for (key, value) in map {
        agent.set_context(key, value.clone())?;
    }
    Ok(())
}

fn apply_base_context(
    agent: &RuntimeAgent,
    suite: &EvalSuite,
    base_dir: &Path,
    scenario: &Scenario,
    mock_server: Option<&crate::fixtures::MockServerHandle>,
) -> Result<()> {
    if let Some(mock_server) = mock_server {
        apply_context_map(agent, mock_server.context())?;
    }
    apply_context_map(agent, resolve_fixture_context(&suite.fixtures, base_dir)?)?;
    apply_context_value(agent, &scenario.context)?;
    if let Some(actor) = &scenario.actor {
        agent.set_actor_id(actor)?;
    }
    Ok(())
}

fn reset_options(config: &ResetStepConfig) -> Option<crate::reset::ResetOptions> {
    match config {
        ResetStepConfig::Bool(false) => None,
        ResetStepConfig::Bool(true) => Some(crate::reset::ResetOptions::default()),
        ResetStepConfig::Options(options) => Some(options.clone()),
    }
}

fn redact_assertion_details(details: &mut [crate::assertion::AssertionResultDetail]) {
    for detail in details {
        detail.actual = redact_value(std::mem::take(&mut detail.actual), true, 0);
        detail.expected = redact_value(std::mem::take(&mut detail.expected), true, 0);
    }
}

fn error_result(
    scenario: &Scenario,
    error: EvalError,
    category: FailureCategory,
) -> ScenarioResult {
    ScenarioResult {
        id: scenario.id.clone(),
        name: scenario.name.clone(),
        tags: scenario.tags.clone(),
        language: scenario.language.clone(),
        status: ScenarioStatus::Error {
            message: error.to_string(),
        },
        failure_category: Some(category),
        flaky: false,
        attempts: Vec::new(),
        duration_ms: 0,
        retries_used: 0,
    }
}

fn failure_category_for_attempt(attempt: &AttemptResult) -> FailureCategory {
    let judge_failed = attempt.turns.iter().any(|turn| {
        turn.assertion_results
            .iter()
            .any(|detail| !detail.passed && detail.assertion == "judge")
    });
    if judge_failed {
        FailureCategory::JudgeError
    } else {
        FailureCategory::AssertionFailed
    }
}

fn final_observability_report(
    results: &[ScenarioResult],
) -> Option<ai_agents_observability::ObservabilityReport> {
    results
        .iter()
        .rev()
        .flat_map(|scenario| scenario.attempts.iter().rev())
        .flat_map(|attempt| attempt.turns.iter().rev())
        .find_map(|turn| {
            turn.evidence
                .observability
                .as_ref()
                .and_then(|obs| obs.report.clone())
        })
}

fn apply_eval_llm_settings(spec: &mut AgentSpec, settings: &crate::suite::EvalSettings) {
    if let LLMConfigOrSelector::Config(config) = &mut spec.llm {
        apply_llm_config_settings(config, settings);
    }
    for config in spec.llms.values_mut() {
        apply_llm_config_settings(config, settings);
    }
}

fn apply_llm_config_settings(
    config: &mut ai_agents_runtime::spec::LLMConfig,
    settings: &crate::suite::EvalSettings,
) {
    if let Some(temperature) = settings.temperature {
        config.temperature = temperature;
    }
    if let Some(seed) = settings.seed {
        config.extra.insert("seed".to_string(), json!(seed));
    }
}

/// Restores process environment variables when an attempt ends.
struct EnvGuard {
    /// Previous env values restored on drop.
    previous: Vec<(String, Option<String>)>,
    /// Process-wide env lock held for the attempt.
    _guard: MutexGuard<'static, ()>,
}

impl EnvGuard {
    fn apply(values: &HashMap<String, String>) -> Result<Self> {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .map_err(|_| EvalError::Runtime("failed to lock eval environment guard".to_string()))?;
        let mut previous = Vec::new();
        for (key, value) in values {
            previous.push((key.clone(), std::env::var(key).ok()));
            unsafe {
                std::env::set_var(key, value);
            }
        }
        Ok(Self {
            previous,
            _guard: guard,
        })
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, value) in self.previous.drain(..).rev() {
            unsafe {
                if let Some(value) = value {
                    std::env::set_var(key, value);
                } else {
                    std::env::remove_var(key);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn runner_executes_mocked_suite_and_redacts_outputs() {
        let dir = std::env::temp_dir().join(format!(
            "ai_agents_eval_runner_test_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let agent_path = dir.join("agent.yaml");
        std::fs::write(
            &agent_path,
            r#"
name: TestAgent
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4.1-nano
"#,
        )
        .unwrap();
        let suite_path = dir.join("suite.yaml");
        std::fs::write(
            &suite_path,
            r#"
name: Runner Suite
agent: agent.yaml
fixtures:
  llm:
    mode: mock
    responses:
      - "Hello from mock"
scenarios:
  - id: smoke
    turns:
      - input: Hello
        assert:
          response_contains: "Hello"
"#,
        )
        .unwrap();
        let options = EvalRunnerOptions {
            output: dir.join("out"),
            ..Default::default()
        };
        let runner = EvalRunner::from_file(&suite_path, options).unwrap();
        let result = runner.run().await.unwrap();
        assert_eq!(result.passed, 1);
        let turn = &result.scenarios[0].attempts[0].turns[0];
        assert_eq!(turn.input.value, "[redacted]");
        assert_eq!(turn.response.value, "[redacted]");
        let json = serde_json::to_string(&result).unwrap();
        assert!(!json.contains("Hello from mock"));
        let _ = std::fs::remove_dir_all(dir);
    }
}

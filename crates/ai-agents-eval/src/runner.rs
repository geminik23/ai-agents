use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ai_agents_runtime::spec::{AgentSpec, StorageConfig};
use ai_agents_runtime::{Agent, AgentBuilder, RuntimeAgent};
use serde_json::Value;
use tokio::time::{Duration, timeout};

use crate::assertion::{AssertionOutcome, evaluate_assertion};
use crate::compatibility::suite_from_jsonl;
use crate::evidence::{collect_turn_evidence, relationship_snapshot};
use crate::fixtures::{
    RecordingToolLog, build_llm_registry, build_tool_registry, resolve_fixture_context,
};
use crate::judge::{JudgeConfig, LLMJudge};
use crate::metrics::compute_metrics;
use crate::redaction::redact_text;
use crate::suite::{
    AttemptResult, EvalResult, EvalSuite, FailureCategory, Scenario, ScenarioResult,
    ScenarioStatus, ScenarioStep, Turn, TurnResult,
};
use crate::{EvalError, Result};

#[derive(Debug, Clone, Default)]
pub struct EvalRunnerOptions {
    pub agent: Option<PathBuf>,
    pub scenarios: Option<PathBuf>,
    pub output: PathBuf,
    pub ids: Vec<String>,
    pub tags: Vec<String>,
    pub tag_mode_all: bool,
    pub languages: Vec<String>,
    pub retries: Option<u32>,
    pub timeout_ms: Option<u64>,
    pub parallel: Option<usize>,
    pub fail_fast: bool,
    pub observability: bool,
}

pub struct EvalRunner {
    suite_path: PathBuf,
    suite: EvalSuite,
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
        let mut results = Vec::new();

        for scenario in scenarios {
            let result = self.run_scenario(&agent_path, base_dir, scenario).await;
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
                    results.push(ScenarioResult {
                        id: scenario.id.clone(),
                        name: scenario.name.clone(),
                        tags: scenario.tags.clone(),
                        language: scenario.language.clone(),
                        status: ScenarioStatus::Error {
                            message: error.to_string(),
                        },
                        failure_category: Some(FailureCategory::RuntimeError),
                        flaky: false,
                        attempts: Vec::new(),
                        duration_ms: 0,
                        retries_used: 0,
                    });
                    if self.suite.settings.fail_fast {
                        break;
                    }
                }
            }
        }

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

        Ok(EvalResult {
            suite: self.suite.name.clone(),
            agent: agent_path.display().to_string(),
            total,
            passed,
            failed,
            skipped,
            duration_ms: start.elapsed().as_millis() as u64,
            scenarios: results,
            metrics,
            observability: None,
        })
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
            let attempt = self
                .run_attempt(agent_path, base_dir, scenario, attempt_idx)
                .await;
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
                        FailureCategory::AssertionFailed
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
        let tool_log = RecordingToolLog::new();
        let mut agent = self
            .build_agent(agent_path, base_dir, &workspace, tool_log.clone())
            .await?;
        apply_context_map(
            &agent,
            resolve_fixture_context(&self.suite.fixtures, base_dir)?,
        )?;
        apply_context_value(&agent, &scenario.context)?;
        if let Some(actor) = &scenario.actor {
            agent.set_actor_id(actor)?;
        }
        let mut turns = Vec::new();
        let mut status = ScenarioStatus::Passed;

        if !scenario.turns.is_empty() {
            for (idx, turn) in scenario.turns.iter().enumerate() {
                let turn_result = self.run_turn(&agent, turn, idx, &tool_log).await?;
                let turn_failed = turn_result.assertion_results.iter().any(|r| !r.passed);
                turns.push(turn_result);
                if turn_failed {
                    status = ScenarioStatus::Failed {
                        reason: format!("turn {} assertion failed", idx + 1),
                    };
                    break;
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
                        let turn_result = self.run_turn(&agent, turn, idx, &tool_log).await?;
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
                ScenarioStep::ResetAgent(_reset) => {
                    agent = self
                        .build_agent(agent_path, base_dir, &workspace, tool_log.clone())
                        .await?;
                    apply_context_map(
                        &agent,
                        resolve_fixture_context(&self.suite.fixtures, base_dir)?,
                    )?;
                    apply_context_value(&agent, &scenario.context)?;
                    if let Some(actor) = &scenario.actor {
                        agent.set_actor_id(actor)?;
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
        let spec: AgentSpec = serde_yaml::from_str(&content)?;
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

        if let Some(observability) = self.suite.observability.clone() {
            let manager = ai_agents_observability::ObservabilityManager::new(observability);
            builder = builder.observability(manager);
        }
        builder = builder
            .auto_configure_spawner()
            .await
            .map_err(|error| EvalError::Config(error.to_string()))?;
        let storage_override = isolated_storage_config(&spec, workspace);
        if let Some(storage) = storage_override {
            builder = builder.storage_config(storage);
        }
        let agent = builder
            .build()
            .map_err(|error| EvalError::Config(error.to_string()))?;
        agent.init_storage().await?;
        Ok(agent)
    }

    async fn run_turn(
        &self,
        agent: &RuntimeAgent,
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
        let response = timeout(Duration::from_millis(timeout_ms), agent.chat(&turn.input))
            .await
            .map_err(|_| EvalError::Runtime(format!("turn timed out after {}ms", timeout_ms)))??;
        let latency_ms = start.elapsed().as_millis() as u64;
        let evidence = collect_turn_evidence(
            agent,
            response.metadata.clone(),
            tool_log,
            tool_start,
            before_relationship,
        );
        let judge = self.build_judge(agent);
        let assertion_results = if let Some(assertion) = &turn.assertions {
            match evaluate_assertion(assertion, &evidence, &response.content, judge.as_ref()).await
            {
                AssertionOutcome::Passed(details) | AssertionOutcome::Failed(details) => details,
                AssertionOutcome::Error(message) => return Err(EvalError::Assertion(message)),
            }
        } else {
            Vec::new()
        };
        Ok(TurnResult {
            index,
            input: redact_text(&turn.input, self.suite.settings.redact_outputs, 120),
            response: redact_text(&response.content, self.suite.settings.redact_outputs, 240),
            state: evidence.state.clone(),
            metadata: response.metadata.and_then(|m| serde_json::to_value(m).ok()),
            evidence,
            assertion_results,
            latency_ms,
            observability_span_id: None,
        })
    }

    fn build_judge(&self, agent: &RuntimeAgent) -> Option<LLMJudge> {
        let llm = agent
            .llm_registry()
            .router()
            .ok()
            .or_else(|| agent.llm_registry().default().ok())?;
        Some(LLMJudge::new(llm, JudgeConfig::default()))
    }
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

fn apply_context_map(
    agent: &RuntimeAgent,
    values: std::collections::HashMap<String, Value>,
) -> Result<()> {
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

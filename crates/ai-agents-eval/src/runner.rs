use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Instant;

use ai_agents_hooks::AgentHooks;
use ai_agents_observability::config::ExportFormat;
use ai_agents_observability::{CostEstimator, ObservabilityConfig};
use ai_agents_runtime::spec::{AgentSpec, LLMConfigOrSelector, StorageConfig};
use ai_agents_runtime::{Agent, AgentBuilder, RuntimeAgent, StreamChunk};
use async_trait::async_trait;
use futures::{StreamExt, stream};
use serde_json::{Value, json};
use tokio::time::{Duration, Instant as TokioInstant, timeout, timeout_at};

use crate::assertion::{
    Assertion, AssertionEvalContext, AssertionOutcome, AssertionResultDetail, evaluate_assertion,
};
use crate::budget::{BudgetProviderConfig, ScenarioBudgetTracker};
use crate::compatibility::suite_from_jsonl;
use crate::evidence::{
    ApprovalEvidence, LlmRequestEvidence, collect_turn_evidence, relationship_snapshot,
};
use crate::fixtures::{
    AttemptFixtureContext, AttemptWorkspace, LlmFixtureMode, RecordingToolLog,
    WorkspacePolicyFixtureConfig, build_approval_handler, build_llm_registry, build_tool_registry,
    resolve_fixture_context, start_mock_server,
};
use crate::judge::{JudgeConfig, JudgeResolver};
use crate::metrics::compute_metrics;
use crate::redaction::{redact_text, redact_value};
use crate::suite::{
    AttemptResult, EvalResult, EvalSuite, FailureCategory, IsolationMode, ResetStepConfig,
    Scenario, ScenarioResult, ScenarioStatus, ScenarioStep, Turn, TurnResult, turn_expected_error,
    turn_runtime_context,
};
use crate::{EvalError, Result};

/// Eval hook that records LLM requests, tools, and resolved approvals.
struct EvalRecordHooks {
    tool_log: RecordingToolLog,
    approval_log: RecordingApprovalLog,
    llm_log: RecordingLlmLog,
}

#[derive(Clone, Default)]
struct RecordingLlmLog {
    records: Arc<Mutex<Vec<LlmRequestEvidence>>>,
}

impl RecordingLlmLog {
    fn len(&self) -> usize {
        self.records
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    fn push_messages(&self, messages: &[ai_agents_core::ChatMessage]) {
        self.records
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(LlmRequestEvidence::from_messages(messages));
    }

    fn records_since(&self, start: usize) -> Vec<LlmRequestEvidence> {
        self.records
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(start..)
            .unwrap_or_default()
            .to_vec()
    }
}

#[derive(Clone, Default)]
struct RecordingApprovalLog {
    records: Arc<Mutex<Vec<ApprovalEvidence>>>,
}

impl RecordingApprovalLog {
    fn len(&self) -> usize {
        self.records
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    fn push(&self, evidence: ApprovalEvidence) {
        self.records
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(evidence);
    }

    fn records_since(&self, start: usize) -> Vec<ApprovalEvidence> {
        self.records
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(start..)
            .unwrap_or_default()
            .to_vec()
    }
}

#[async_trait]
impl AgentHooks for EvalRecordHooks {
    async fn on_llm_start(&self, messages: &[ai_agents_core::ChatMessage]) {
        self.llm_log.push_messages(messages);
    }

    async fn on_tool_execution_record(&self, record: &ai_agents_core::ToolExecutionRecord) {
        self.tool_log.push_executor_record(record);
    }

    async fn on_approval_resolved(
        &self,
        request: &ai_agents_hitl::ApprovalRequest,
        raw_result: &ai_agents_hitl::ApprovalResult,
        outcome: &ai_agents_hitl::ApprovalResolvedOutcome,
    ) {
        self.approval_log.push(ApprovalEvidence::from_resolution(
            request, raw_result, outcome,
        ));
    }
}

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

fn load_eval_suite(path: &Path) -> Result<EvalSuite> {
    let content = std::fs::read_to_string(path)?;
    if path.extension().and_then(|extension| extension.to_str()) == Some("jsonl") {
        suite_from_jsonl(
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("eval")
                .to_string(),
            &content,
        )
    } else {
        parse_eval_suite_yaml(&content)
    }
}

fn parse_eval_suite_yaml(content: &str) -> Result<EvalSuite> {
    let mut unknown_fields = Vec::new();
    let deserializer = serde_yaml::Deserializer::from_str(content);
    let suite = serde_ignored::deserialize(deserializer, |path| {
        unknown_fields.push(path.to_string());
    })?;
    if !unknown_fields.is_empty() {
        unknown_fields.sort();
        unknown_fields.dedup();
        return Err(EvalError::Config(format!(
            "unknown eval configuration field(s): {}",
            unknown_fields.join(", ")
        )));
    }
    Ok(suite)
}

fn authorize_llm_mode(
    suite_mode: LlmFixtureMode,
    override_mode: Option<LlmFixtureMode>,
) -> Result<()> {
    if override_mode.is_some()
        || matches!(suite_mode, LlmFixtureMode::Mock | LlmFixtureMode::Replay)
    {
        return Ok(());
    }
    match suite_mode {
        LlmFixtureMode::Real => Err(EvalError::Config(
            "suite-declared fixtures.llm.mode real requires --real-llm or EvalRunnerOptions.llm_mode = Some(LlmFixtureMode::Real)"
                .to_string(),
        )),
        LlmFixtureMode::Record => Err(EvalError::Config(
            "suite-declared fixtures.llm.mode record requires --record or EvalRunnerOptions.llm_mode = Some(LlmFixtureMode::Record)"
                .to_string(),
        )),
        LlmFixtureMode::Mock | LlmFixtureMode::Replay => Ok(()),
    }
}

impl EvalRunner {
    pub fn from_file(path: impl AsRef<Path>, options: EvalRunnerOptions) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut suite = load_eval_suite(&path)?;
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
        authorize_llm_mode(suite.fixtures.llm.mode, options.llm_mode)?;
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

    pub fn validate_file(path: impl AsRef<Path>, agent_override: Option<PathBuf>) -> Result<()> {
        let path = path.as_ref();
        let mut suite = load_eval_suite(path)?;
        if let Some(agent) = &agent_override {
            suite.agent = Some(agent.clone());
        }
        suite.validate(agent_override.as_ref())?;

        let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
        let agent_path = if let Some(agent) = agent_override {
            agent
        } else {
            let agent = suite.agent.ok_or_else(|| {
                EvalError::Config("agent path is required in suite or CLI".into())
            })?;
            if agent.is_absolute() {
                agent
            } else {
                base_dir.join(agent)
            }
        };
        let content = std::fs::read_to_string(&agent_path)?;
        let spec = AgentSpec::from_yaml_strict(&content).map_err(|error| {
            EvalError::Config(format!(
                "invalid agent configuration '{}': {}",
                agent_path.display(),
                error
            ))
        })?;
        spec.validate().map_err(|error| {
            EvalError::Config(format!(
                "invalid agent configuration '{}': {}",
                agent_path.display(),
                error
            ))
        })
    }

    pub async fn run(&self) -> Result<EvalResult> {
        let start = Instant::now();
        let base_dir = self.suite_path.parent().unwrap_or_else(|| Path::new("."));
        let scenarios = self.filtered_scenarios();
        if scenarios.is_empty() {
            return Err(EvalError::Config(
                "scenario selection matched zero scenarios".to_string(),
            ));
        }
        let agent_path = self.resolve_agent_path(base_dir)?;
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
        let budget = if scenario.budget.is_configured() {
            Some(ScenarioBudgetTracker::new(
                scenario.budget.clone(),
                self.budget_cost_estimator(base_dir, scenario)?,
            ))
        } else {
            None
        };

        for attempt_idx in 0..max_attempt {
            let attempt_future =
                self.run_attempt(agent_path, base_dir, scenario, attempt_idx, budget.clone());
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
            if budget
                .as_ref()
                .is_some_and(ScenarioBudgetTracker::has_failed)
            {
                break;
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
        budget: Option<ScenarioBudgetTracker>,
    ) -> Result<AttemptResult> {
        let start = Instant::now();
        let _env_guard = EnvGuard::apply(&scenario.env)?;
        let mock_server = start_mock_server(self.suite.fixtures.mock_server.as_ref()).await?;
        let attempt_context = AttemptWorkspace::create(mock_server.as_ref())?;
        let tool_log = RecordingToolLog::new();
        let approval_log = RecordingApprovalLog::default();
        let llm_log = RecordingLlmLog::default();
        let approval_handler = self
            .suite
            .fixtures
            .approvals
            .as_ref()
            .map(build_approval_handler);
        let mut agent = self
            .build_agent(
                agent_path,
                base_dir,
                &attempt_context,
                tool_log.clone(),
                approval_log.clone(),
                llm_log.clone(),
                approval_handler.clone(),
                budget.clone(),
            )
            .await?;
        apply_base_context(&agent, &self.suite, base_dir, scenario, &attempt_context)?;
        let mut turns = Vec::new();
        let mut status = ScenarioStatus::Passed;

        if !scenario.turns.is_empty() {
            for (idx, turn) in scenario.turns.iter().enumerate() {
                let turn_execution = self
                    .run_turn(
                        &agent,
                        scenario,
                        turn,
                        idx,
                        &tool_log,
                        &approval_log,
                        &llm_log,
                    )
                    .await?;
                let turn_failed = turn_execution
                    .result
                    .assertion_results
                    .iter()
                    .any(|result| !result.passed);
                let runtime_error = turn_execution.unhandled_runtime_error;
                turns.push(turn_execution.result);
                if let Some(message) = runtime_error {
                    status = ScenarioStatus::Error { message };
                    break;
                }
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
                    apply_base_context(&agent, &self.suite, base_dir, scenario, &attempt_context)?;
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
                        let turn_execution = self
                            .run_turn(
                                &agent,
                                scenario,
                                turn,
                                idx,
                                &tool_log,
                                &approval_log,
                                &llm_log,
                            )
                            .await?;
                        let turn_failed = turn_execution
                            .result
                            .assertion_results
                            .iter()
                            .any(|result| !result.passed);
                        let runtime_error = turn_execution.unhandled_runtime_error;
                        turns.push(turn_execution.result);
                        if let Some(message) = runtime_error {
                            status = ScenarioStatus::Error { message };
                            break;
                        }
                        if turn_failed {
                            status = ScenarioStatus::Failed {
                                reason: format!("turn {} assertion failed", idx + 1),
                            };
                            break;
                        }
                    }
                    if status.is_passed() {
                        if let Some(session) = &run.save_session {
                            agent.save_session(session).await?;
                        }
                    }
                }
                ScenarioStep::ResetAgent(reset) => {
                    if let Some(options) = reset_options(reset) {
                        if options.delete_persistence || !options.preserve_storage {
                            let _ = std::fs::remove_dir_all(&attempt_context.workspace);
                            std::fs::create_dir_all(&attempt_context.workspace)?;
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
                                .build_agent(
                                    agent_path,
                                    base_dir,
                                    &attempt_context,
                                    tool_log.clone(),
                                    approval_log.clone(),
                                    llm_log.clone(),
                                    approval_handler.clone(),
                                    budget.clone(),
                                )
                                .await?;
                        }
                        if options.preserve_host_context {
                            apply_base_context(
                                &agent,
                                &self.suite,
                                base_dir,
                                scenario,
                                &attempt_context,
                            )?;
                        } else {
                            apply_context_map(&agent, attempt_context.runtime_context())?;
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
        attempt_context: &AttemptFixtureContext,
        tool_log: RecordingToolLog,
        approval_log: RecordingApprovalLog,
        llm_log: RecordingLlmLog,
        approval_handler: Option<Arc<dyn ai_agents_hitl::ApprovalHandler>>,
        budget: Option<ScenarioBudgetTracker>,
    ) -> Result<RuntimeAgent> {
        let content = std::fs::read_to_string(agent_path)?;
        let mut spec = AgentSpec::from_yaml_strict(&content)?;
        apply_eval_llm_settings(&mut spec, &self.suite.settings);
        isolate_spec_storage(&mut spec, attempt_context);
        apply_workspace_policy(
            &mut spec,
            self.suite.fixtures.workspace_policy.as_ref(),
            attempt_context,
        )?;
        spec.validate()
            .map_err(|error| EvalError::Config(error.to_string()))?;
        let llm_fixture = attempt_context.interpolate_llm_fixture(&self.suite.fixtures.llm)?;
        let provider_configs = budget_provider_configs(&spec);
        let (mut llm_registry, _judge_llm) = build_llm_registry(&spec, &llm_fixture, base_dir)?;
        if let Some(budget) = budget {
            llm_registry =
                llm_registry.map_providers(|alias, provider| {
                    let config = provider_configs.get(alias).cloned().unwrap_or_else(|| {
                        BudgetProviderConfig {
                            provider: provider.provider_name().to_string(),
                            model: alias.to_string(),
                            max_output_tokens: 2_000,
                        }
                    });
                    budget.wrap(provider, config)
                });
        }
        let tool_registry = build_tool_registry(&self.suite.fixtures, tool_log.clone())?;
        let agent_base_dir = agent_path.parent().unwrap_or_else(|| Path::new("."));
        let mut builder = AgentBuilder::from_spec_with_base_dir(spec, agent_base_dir)
            .llm_registry(llm_registry)
            .tools(tool_registry)
            .hooks(Arc::new(EvalRecordHooks {
                tool_log,
                approval_log,
                llm_log,
            }))
            .auto_configure_features()
            .map_err(|error| EvalError::Config(error.to_string()))?
            .auto_configure_mcp()
            .await
            .map_err(|error| EvalError::Config(error.to_string()))?;

        if let Some(approval_handler) = approval_handler {
            builder = builder.approval_handler(approval_handler);
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

    fn budget_cost_estimator(
        &self,
        base_dir: &Path,
        scenario: &Scenario,
    ) -> Result<Option<CostEstimator>> {
        if scenario.budget.max_cost_usd.is_none() {
            return Ok(None);
        }
        let config = self.suite.observability.clone().ok_or_else(|| {
            EvalError::Config(format!(
                "scenario '{}' budget.max_cost_usd requires suite observability.cost pricing",
                scenario.id
            ))
        })?;
        let config = config
            .with_pricing_file_loaded(Some(base_dir))
            .map_err(|error| EvalError::Config(error.to_string()))?;
        if !config.cost.enabled {
            return Err(EvalError::Config(format!(
                "scenario '{}' budget.max_cost_usd requires observability.cost.enabled: true",
                scenario.id
            )));
        }
        Ok(Some(CostEstimator::new(config.cost)))
    }

    async fn run_turn(
        &self,
        agent: &RuntimeAgent,
        scenario: &Scenario,
        turn: &Turn,
        index: usize,
        tool_log: &RecordingToolLog,
        approval_log: &RecordingApprovalLog,
        llm_log: &RecordingLlmLog,
    ) -> Result<TurnExecution> {
        apply_context_value(agent, &turn_runtime_context(turn))?;
        if let Some(actor) = &turn.actor {
            agent.set_actor_id(actor)?;
        }
        let before_relationship = relationship_snapshot(agent);
        let tool_start = tool_log.len();
        let approval_start = approval_log.len();
        let llm_start = llm_log.len();
        let start = Instant::now();
        let timeout_ms = turn
            .timeout_ms
            .unwrap_or(self.suite.settings.timeout_per_turn_ms);
        let mut operation = if turn.stream.unwrap_or(false) {
            collect_stream_response(agent, &turn.input, timeout_ms).await
        } else {
            match timeout(Duration::from_millis(timeout_ms), agent.chat(&turn.input)).await {
                Ok(Ok(response)) => TurnOperation {
                    response_content: response.content,
                    response_metadata: response.metadata,
                    response_present: true,
                    runtime_error: None,
                },
                Ok(Err(error)) => TurnOperation::error(error.to_string()),
                Err(_) => TurnOperation::error(format!("turn timed out after {}ms", timeout_ms)),
            }
        };
        if let Err(error) = agent.flush_background_tasks().await {
            if operation.runtime_error.is_none() {
                operation.runtime_error = Some(error.to_string());
            }
        }
        let latency_ms = start.elapsed().as_millis() as u64;
        let mut evidence = collect_turn_evidence(
            agent,
            operation.response_metadata.clone(),
            tool_log,
            tool_start,
            before_relationship,
        );
        evidence.approvals = approval_log.records_since(approval_start);
        evidence.llm_requests = llm_log.records_since(llm_start);
        let judge = self.build_judge(agent);
        let mut assertion_results = if let Some(assertion) = &turn.assertions {
            match evaluate_assertion(
                assertion,
                AssertionEvalContext {
                    evidence: &evidence,
                    response: &operation.response_content,
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
        if !operation.response_present
            && turn
                .assertions
                .as_ref()
                .is_some_and(assertion_uses_response)
        {
            assertion_results.push(AssertionResultDetail {
                assertion: "response_present".to_string(),
                passed: false,
                actual: json!(false),
                expected: json!(true),
                message: Some("response assertions require a runtime response".to_string()),
            });
        }

        let expected_error = turn_expected_error(turn);
        let unhandled_runtime_error = match (&expected_error, &operation.runtime_error) {
            (Some(expected), Some(error)) if expected.matches(error) => {
                assertion_results.push(AssertionResultDetail {
                    assertion: "expect_error".to_string(),
                    passed: true,
                    actual: json!(error),
                    expected: json!(expected.items()),
                    message: None,
                });
                None
            }
            (Some(expected), Some(error)) => {
                assertion_results.push(AssertionResultDetail {
                    assertion: "expect_error".to_string(),
                    passed: false,
                    actual: json!(error),
                    expected: json!(expected.items()),
                    message: Some("runtime error did not match any expected substring".to_string()),
                });
                Some(format!(
                    "runtime error did not match expect_error: {}",
                    error
                ))
            }
            (Some(expected), None) => {
                assertion_results.push(AssertionResultDetail {
                    assertion: "expect_error".to_string(),
                    passed: false,
                    actual: Value::Null,
                    expected: json!(expected.items()),
                    message: Some("expected a runtime error but the turn completed".to_string()),
                });
                None
            }
            (None, Some(error)) => Some(error.clone()),
            (None, None) => None,
        };
        if self.suite.settings.redact_outputs {
            redact_assertion_details(&mut assertion_results);
        }
        let observability_span_id = evidence
            .observability
            .as_ref()
            .and_then(|obs| obs.span_ids.last().cloned());
        let runtime_error = operation
            .runtime_error
            .as_deref()
            .map(|error| redact_text(error, self.suite.settings.redact_outputs, 0));
        let unhandled_runtime_error = unhandled_runtime_error
            .map(|error| redact_text(&error, self.suite.settings.redact_outputs, 0).value);
        Ok(TurnExecution {
            result: TurnResult {
                index,
                input: redact_text(&turn.input, self.suite.settings.redact_outputs, 0),
                response: if operation.response_present {
                    redact_text(
                        &operation.response_content,
                        self.suite.settings.redact_outputs,
                        0,
                    )
                } else {
                    crate::redaction::RedactedString::plain("")
                },
                response_present: operation.response_present,
                runtime_error,
                state: evidence.state.clone(),
                metadata: if self.suite.settings.redact_outputs {
                    None
                } else {
                    operation
                        .response_metadata
                        .and_then(|metadata| serde_json::to_value(metadata).ok())
                },
                evidence,
                assertion_results,
                latency_ms,
                observability_span_id,
            },
            unhandled_runtime_error,
        })
    }

    fn build_judge(&self, agent: &RuntimeAgent) -> JudgeResolver {
        JudgeResolver::new(Arc::clone(agent.llm_registry()), JudgeConfig::default())
    }
}

struct TurnExecution {
    result: TurnResult,
    unhandled_runtime_error: Option<String>,
}

struct TurnOperation {
    response_content: String,
    response_metadata: Option<HashMap<String, Value>>,
    response_present: bool,
    runtime_error: Option<String>,
}

impl TurnOperation {
    fn error(message: String) -> Self {
        Self {
            response_content: String::new(),
            response_metadata: None,
            response_present: false,
            runtime_error: Some(message),
        }
    }
}

async fn collect_stream_response(
    agent: &RuntimeAgent,
    input: &str,
    timeout_ms: u64,
) -> TurnOperation {
    let deadline = TokioInstant::now() + Duration::from_millis(timeout_ms);
    let stream = match timeout_at(deadline, agent.chat_stream(input)).await {
        Ok(Ok(stream)) => stream,
        Ok(Err(error)) => return TurnOperation::error(error.to_string()),
        Err(_) => return TurnOperation::error(format!("turn timed out after {}ms", timeout_ms)),
    };
    consume_stream_response(stream, deadline, timeout_ms).await
}

async fn consume_stream_response<S>(
    mut stream: S,
    deadline: TokioInstant,
    timeout_ms: u64,
) -> TurnOperation
where
    S: futures::Stream<Item = StreamChunk> + Unpin,
{
    let mut content = String::new();
    let mut content_seen = false;
    let mut runtime_error = None;
    let mut done = false;
    loop {
        match timeout_at(deadline, stream.next()).await {
            Ok(Some(StreamChunk::Content { text })) => {
                content_seen = true;
                content.push_str(&text);
            }
            Ok(Some(StreamChunk::Done {})) => {
                done = true;
                break;
            }
            Ok(Some(StreamChunk::Error { message })) => {
                if runtime_error.is_none() {
                    runtime_error = Some(message);
                }
            }
            Ok(Some(_)) => {}
            Ok(None) => break,
            Err(_) => {
                if runtime_error.is_none() {
                    runtime_error = Some(format!("turn timed out after {}ms", timeout_ms));
                }
                break;
            }
        }
    }
    if !done && runtime_error.is_none() {
        runtime_error = Some("stream ended before Done".to_string());
    }
    TurnOperation {
        response_content: content,
        response_metadata: None,
        response_present: content_seen || (done && runtime_error.is_none()),
        runtime_error,
    }
}

fn assertion_uses_response(assertion: &Assertion) -> bool {
    assertion.response_contains.is_some()
        || assertion.response_contains_any.is_some()
        || assertion.response_not_contains.is_some()
        || assertion.response_not_empty.is_some()
        || assertion.response_semantic.is_some()
        || assertion.judge.is_some()
        || assertion
            .all
            .as_ref()
            .is_some_and(|children| children.iter().any(assertion_uses_response))
        || assertion
            .any
            .as_ref()
            .is_some_and(|children| children.iter().any(assertion_uses_response))
        || assertion
            .not
            .as_deref()
            .is_some_and(assertion_uses_response)
}

fn apply_workspace_policy(
    spec: &mut AgentSpec,
    config: Option<&WorkspacePolicyFixtureConfig>,
    context: &AttemptFixtureContext,
) -> Result<()> {
    let Some(config) = config else {
        return Ok(());
    };
    if !context.workspace.is_absolute() {
        return Err(EvalError::Config(
            "eval attempt workspace must be absolute".to_string(),
        ));
    }

    for tool_id in config.read_tools.iter().chain(&config.write_tools) {
        if !spec.tool_security.tools.contains_key(tool_id) {
            return Err(EvalError::Config(format!(
                "fixtures.workspace_policy names tool '{}' without an existing tool policy",
                tool_id
            )));
        }
    }

    let workspace = context.workspace.display().to_string();
    for tool_id in &config.read_tools {
        spec.tool_security
            .tools
            .get_mut(tool_id)
            .expect("workspace policy tool was validated")
            .read_paths
            .push(workspace.clone());
    }
    for tool_id in &config.write_tools {
        spec.tool_security
            .tools
            .get_mut(tool_id)
            .expect("workspace policy tool was validated")
            .write_paths
            .push(workspace.clone());
    }
    Ok(())
}

fn isolate_spec_storage(spec: &mut AgentSpec, context: &AttemptFixtureContext) {
    isolate_storage_config(
        &mut spec.storage,
        context,
        "parent-storage",
        "parent-storage.db",
    );
    if let Some(shared_storage) = spec
        .spawner
        .as_mut()
        .and_then(|spawner| spawner.shared_storage.as_mut())
    {
        isolate_storage_config(
            shared_storage,
            context,
            "spawner-shared-storage",
            "spawner-shared-storage.db",
        );
    }
}

fn isolate_storage_config(
    storage: &mut StorageConfig,
    context: &AttemptFixtureContext,
    file_name: &str,
    sqlite_name: &str,
) {
    match storage {
        StorageConfig::File(config) => {
            config.path = context.workspace.join(file_name).display().to_string();
        }
        StorageConfig::Sqlite(config) => {
            config.path = context.workspace.join(sqlite_name).display().to_string();
        }
        StorageConfig::Redis(config) => {
            let prefix = config.prefix.as_deref().unwrap_or("agent:");
            config.prefix = Some(format!("{}eval:{}:", prefix, context.isolation_id));
        }
        StorageConfig::None => {}
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
    attempt_context: &AttemptFixtureContext,
) -> Result<()> {
    apply_context_map(agent, resolve_fixture_context(&suite.fixtures, base_dir)?)?;
    apply_context_map(agent, attempt_context.runtime_context())?;
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

fn budget_provider_configs(spec: &AgentSpec) -> HashMap<String, BudgetProviderConfig> {
    if spec.llms.is_empty() {
        let config = spec.llm.as_config().cloned().unwrap_or_default();
        return HashMap::from([(
            "default".to_string(),
            BudgetProviderConfig {
                provider: config.provider,
                model: config.model,
                max_output_tokens: config.max_tokens,
            },
        )]);
    }
    spec.llms
        .iter()
        .map(|(alias, config)| {
            (
                alias.clone(),
                BudgetProviderConfig {
                    provider: config.provider.clone(),
                    model: config.model.clone(),
                    max_output_tokens: config.max_tokens,
                },
            )
        })
        .collect()
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

/// Process-wide exclusion held for an eval attempt's environment access.
enum EnvExclusionGuard {
    Read {
        _guard: RwLockReadGuard<'static, ()>,
    },
    Write {
        _guard: RwLockWriteGuard<'static, ()>,
    },
}

/// Restores process environment variables when an attempt ends.
struct EnvGuard {
    /// Previous env values restored on drop.
    previous: Vec<(String, Option<String>)>,
    /// Shared access for unchanged env or exclusive access for mutations.
    _guard: EnvExclusionGuard,
}

impl EnvGuard {
    fn apply(values: &HashMap<String, String>) -> Result<Self> {
        static ENV_LOCK: OnceLock<RwLock<()>> = OnceLock::new();
        let lock = ENV_LOCK.get_or_init(|| RwLock::new(()));
        if values.is_empty() {
            let guard = lock.read().map_err(|_| {
                EvalError::Runtime("failed to lock eval environment guard".to_string())
            })?;
            return Ok(Self {
                previous: Vec::new(),
                _guard: EnvExclusionGuard::Read { _guard: guard },
            });
        }
        let guard = lock
            .write()
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
            _guard: EnvExclusionGuard::Write { _guard: guard },
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

    #[test]
    fn strict_suite_loader_rejects_nested_observability_typos() {
        let error = parse_eval_suite_yaml(
            r#"
name: strict
agent: agent.yaml
observability:
  enabeld: true
scenarios:
  - id: scenario
    turns:
      - input: hello
"#,
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("enabeld"), "{error}");
    }

    #[test]
    fn storage_isolation_rewrites_parent_and_spawner_backends() {
        let context = AttemptFixtureContext {
            isolation_id: "attempt-a".to_string(),
            workspace: PathBuf::from("/tmp/eval-attempt-a"),
            mock_server_base_url: None,
        };
        let mut file_spec: AgentSpec = serde_yaml::from_str(
            r#"
name: FileAgent
system_prompt: test
storage: { type: file, path: ./parent }
spawner:
  shared_storage: { type: sqlite, path: ./shared.db, table: shared_sessions }
"#,
        )
        .unwrap();
        isolate_spec_storage(&mut file_spec, &context);
        assert_eq!(
            file_spec.storage.get_path(),
            Some("/tmp/eval-attempt-a/parent-storage")
        );
        let shared = file_spec
            .spawner
            .as_ref()
            .unwrap()
            .shared_storage
            .as_ref()
            .unwrap();
        assert_eq!(
            shared.get_path(),
            Some("/tmp/eval-attempt-a/spawner-shared-storage.db")
        );
        assert_eq!(shared.get_table(), Some("shared_sessions"));

        let mut redis_spec: AgentSpec = serde_yaml::from_str(
            r#"
name: RedisAgent
system_prompt: test
storage: { type: redis, url: redis://localhost, prefix: "parent:" }
spawner:
  shared_storage: { type: redis, url: redis://localhost }
"#,
        )
        .unwrap();
        isolate_spec_storage(&mut redis_spec, &context);
        assert_eq!(redis_spec.storage.get_prefix(), "parent:eval:attempt-a:");
        assert_eq!(
            redis_spec
                .spawner
                .as_ref()
                .unwrap()
                .shared_storage
                .as_ref()
                .unwrap()
                .get_prefix(),
            "agent:eval:attempt-a:"
        );

        let other_context = AttemptFixtureContext {
            isolation_id: "attempt-b".to_string(),
            workspace: PathBuf::from("/tmp/eval-attempt-b"),
            mock_server_base_url: None,
        };
        let mut other_redis: AgentSpec = serde_yaml::from_str(
            r#"
name: RedisAgent
system_prompt: test
storage: { type: redis, url: redis://localhost, prefix: "parent:" }
"#,
        )
        .unwrap();
        isolate_spec_storage(&mut other_redis, &other_context);
        assert_ne!(
            redis_spec.storage.get_prefix(),
            other_redis.storage.get_prefix()
        );
    }

    #[test]
    fn workspace_policy_is_narrow_and_isolated_per_attempt() {
        let source: AgentSpec = serde_yaml::from_str(
            r#"
name: PolicyAgent
system_prompt: test
tool_security:
  enabled: true
  fail_closed: true
  tools:
    file_read:
      read_paths: [./source]
      blocked_paths: [./blocked]
    file_write:
      write_paths: [./output]
      blocked_paths: [./blocked]
    grep:
      read_paths: [./repository]
"#,
        )
        .unwrap();
        let config = WorkspacePolicyFixtureConfig {
            read_tools: vec!["file_read".to_string()],
            write_tools: vec!["file_write".to_string()],
        };
        let first_context = AttemptFixtureContext {
            isolation_id: "attempt-a".to_string(),
            workspace: PathBuf::from("/tmp/eval-attempt-a"),
            mock_server_base_url: None,
        };
        let second_context = AttemptFixtureContext {
            isolation_id: "attempt-b".to_string(),
            workspace: PathBuf::from("/tmp/eval-attempt-b"),
            mock_server_base_url: None,
        };

        let mut first = source.clone();
        apply_workspace_policy(&mut first, Some(&config), &first_context).unwrap();
        let mut second = source.clone();
        apply_workspace_policy(&mut second, Some(&config), &second_context).unwrap();

        assert!(first.tool_security.fail_closed);
        assert_eq!(
            first.tool_security.tools["file_read"].read_paths,
            vec!["./source", "/tmp/eval-attempt-a"]
        );
        assert_eq!(
            first.tool_security.tools["file_write"].write_paths,
            vec!["./output", "/tmp/eval-attempt-a"]
        );
        assert_eq!(
            first.tool_security.tools["file_read"].blocked_paths,
            vec!["./blocked"]
        );
        assert_eq!(
            first.tool_security.tools["file_write"].blocked_paths,
            vec!["./blocked"]
        );
        assert_eq!(
            first.tool_security.tools["grep"].read_paths,
            vec!["./repository"]
        );
        assert_eq!(
            second.tool_security.tools["file_read"].read_paths,
            vec!["./source", "/tmp/eval-attempt-b"]
        );
        assert_eq!(
            source.tool_security.tools["file_read"].read_paths,
            vec!["./source"]
        );
        assert_eq!(
            source.tool_security.tools["file_write"].write_paths,
            vec!["./output"]
        );
    }

    #[test]
    fn workspace_policy_rejects_unknown_tools_without_partial_mutation() {
        let mut spec: AgentSpec = serde_yaml::from_str(
            r#"
name: PolicyAgent
system_prompt: test
tool_security:
  enabled: true
  fail_closed: true
  tools:
    file_read:
      read_paths: [./source]
"#,
        )
        .unwrap();
        let original = spec.tool_security.tools["file_read"].read_paths.clone();
        let config = WorkspacePolicyFixtureConfig {
            read_tools: vec!["file_read".to_string(), "missing_tool".to_string()],
            write_tools: Vec::new(),
        };
        let context = AttemptFixtureContext {
            isolation_id: "attempt-a".to_string(),
            workspace: PathBuf::from("/tmp/eval-attempt-a"),
            mock_server_base_url: None,
        };

        let error = apply_workspace_policy(&mut spec, Some(&config), &context).unwrap_err();

        assert!(error.to_string().contains("missing_tool"));
        assert!(
            error
                .to_string()
                .contains("without an existing tool policy")
        );
        assert_eq!(spec.tool_security.tools["file_read"].read_paths, original);
        assert!(spec.tool_security.fail_closed);
    }

    #[tokio::test]
    async fn generated_attempt_context_has_stable_reset_precedence() {
        let dir = std::env::temp_dir().join(format!(
            "ai_agents_eval_attempt_context_test_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        write_test_agent(&dir);
        let suite_path = dir.join("suite.yaml");
        std::fs::write(
            &suite_path,
            r#"
name: Attempt Context
agent: agent.yaml
fixtures:
  context:
    eval: { workspace: fixture-value }
    mock_server: { base_url: fixture-value }
    fixture_only: true
  llm:
    mode: mock
    responses: [ok]
scenarios:
  - id: context
    context: { scenario_only: true, precedence: scenario }
    turns:
      - input: test
        context: { precedence: turn }
"#,
        )
        .unwrap();
        let runner = EvalRunner::from_file(
            &suite_path,
            EvalRunnerOptions {
                output: dir.join("out"),
                ..Default::default()
            },
        )
        .unwrap();
        let attempt_context = AttemptFixtureContext {
            isolation_id: "stable-attempt".to_string(),
            workspace: dir
                .join("workspace")
                .canonicalize()
                .unwrap_or_else(|_| dir.join("workspace")),
            mock_server_base_url: Some("http://127.0.0.1:40000".to_string()),
        };
        std::fs::create_dir_all(&attempt_context.workspace).unwrap();
        let scenario = &runner.suite.scenarios[0];
        let tool_log = RecordingToolLog::new();
        let approval_log = RecordingApprovalLog::default();
        let llm_log = RecordingLlmLog::default();
        let first = runner
            .build_agent(
                &dir.join("agent.yaml"),
                &dir,
                &attempt_context,
                tool_log.clone(),
                approval_log.clone(),
                llm_log.clone(),
                None,
                None,
            )
            .await
            .unwrap();
        apply_base_context(&first, &runner.suite, &dir, scenario, &attempt_context).unwrap();
        let first_context = first.get_context();
        assert_eq!(
            first_context["eval"]["workspace"],
            json!(attempt_context.workspace.display().to_string())
        );
        assert_eq!(
            first_context["mock_server"]["base_url"],
            "http://127.0.0.1:40000"
        );
        assert_eq!(first_context["precedence"], "scenario");
        apply_context_value(&first, &turn_runtime_context(&scenario.turns[0])).unwrap();
        assert_eq!(first.get_context()["precedence"], "turn");

        let reset = runner
            .build_agent(
                &dir.join("agent.yaml"),
                &dir,
                &attempt_context,
                tool_log,
                approval_log,
                llm_log,
                None,
                None,
            )
            .await
            .unwrap();
        apply_base_context(&reset, &runner.suite, &dir, scenario, &attempt_context).unwrap();
        assert_eq!(reset.get_context()["eval"], first_context["eval"]);
        assert_eq!(
            reset.get_context()["mock_server"],
            first_context["mock_server"]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn streaming_error_retains_partial_content_and_consumes_until_done() {
        let chunks = stream::iter(vec![
            StreamChunk::Content {
                text: "before ".to_string(),
            },
            StreamChunk::Error {
                message: "stream failed".to_string(),
            },
            StreamChunk::Content {
                text: "after".to_string(),
            },
            StreamChunk::Done {},
        ]);
        let operation =
            consume_stream_response(chunks, TokioInstant::now() + Duration::from_secs(1), 1_000)
                .await;
        assert_eq!(operation.response_content, "before after");
        assert!(operation.response_present);
        assert_eq!(operation.runtime_error.as_deref(), Some("stream failed"));
    }

    #[test]
    fn dry_config_check_validates_real_suite_and_agent_without_authorization() {
        let dir = std::env::temp_dir().join(format!(
            "ai_agents_eval_dry_config_test_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let suite_path = dir.join("suite.yaml");
        std::fs::write(
            &suite_path,
            r#"
name: Dry Config
agent: agent.yaml
fixtures:
  llm:
    mode: real
scenarios:
  - id: live
    turns:
      - input: hello
"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("agent.yaml"),
            "name: TestAgent\nsystem_prompt: test\n",
        )
        .unwrap();

        EvalRunner::validate_file(&suite_path, None).unwrap();

        std::fs::write(
            dir.join("agent.yaml"),
            "name: TestAgent\nsystem_prompt: test\nmax_iteratons: 3\n",
        )
        .unwrap();
        let error = EvalRunner::validate_file(&suite_path, None).unwrap_err();
        assert!(error.to_string().contains("max_iteratons"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn real_and_record_modes_require_explicit_authorization() {
        let dir = std::env::temp_dir().join(format!(
            "ai_agents_eval_authorization_test_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let suite_path = dir.join("suite.yaml");
        std::fs::write(
            &suite_path,
            r#"
name: Authorization
agent: agent.yaml
fixtures:
  llm:
    mode: real
scenarios:
  - id: authorized
    turns:
      - input: hello
"#,
        )
        .unwrap();

        let error = EvalRunner::from_file(&suite_path, EvalRunnerOptions::default())
            .err()
            .expect("real mode should require authorization");
        assert!(error.to_string().contains("--real-llm"));
        assert!(
            EvalRunner::from_file(
                &suite_path,
                EvalRunnerOptions {
                    llm_mode: Some(LlmFixtureMode::Real),
                    ..Default::default()
                },
            )
            .is_ok()
        );

        let record_suite = std::fs::read_to_string(&suite_path)
            .unwrap()
            .replace("mode: real", "mode: record");
        std::fs::write(&suite_path, record_suite).unwrap();
        let error = EvalRunner::from_file(&suite_path, EvalRunnerOptions::default())
            .err()
            .expect("record mode should require authorization");
        assert!(error.to_string().contains("--record"));
        assert!(
            EvalRunner::from_file(
                &suite_path,
                EvalRunnerOptions {
                    llm_mode: Some(LlmFixtureMode::Record),
                    ..Default::default()
                },
            )
            .is_ok()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn zero_selected_scenarios_is_an_error() {
        let dir = std::env::temp_dir().join(format!(
            "ai_agents_eval_zero_selection_test_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let suite_path = dir.join("suite.yaml");
        std::fs::write(
            &suite_path,
            r#"
name: Selection
agent: agent.yaml
fixtures:
  llm:
    mode: mock
    responses: [ok]
scenarios:
  - id: present
    turns:
      - input: hello
"#,
        )
        .unwrap();
        let runner = EvalRunner::from_file(
            &suite_path,
            EvalRunnerOptions {
                ids: vec!["missing".to_string()],
                ..Default::default()
            },
        )
        .unwrap();

        let error = runner.run().await.unwrap_err();

        assert!(error.to_string().contains("matched zero scenarios"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn env_guard_allows_readers_and_excludes_writer() {
        use std::sync::{Barrier, mpsc};
        use std::thread;

        let start = Arc::new(Barrier::new(3));
        let (acquired_tx, acquired_rx) = mpsc::channel();
        let mut releases = Vec::new();
        let mut readers = Vec::new();
        for index in 0..2 {
            let start = Arc::clone(&start);
            let acquired_tx = acquired_tx.clone();
            let (release_tx, release_rx) = mpsc::channel();
            releases.push(release_tx);
            readers.push(thread::spawn(move || {
                start.wait();
                let guard = EnvGuard::apply(&HashMap::new()).unwrap();
                acquired_tx.send(index).unwrap();
                release_rx.recv().unwrap();
                drop(guard);
            }));
        }
        start.wait();
        let first = acquired_rx.recv_timeout(Duration::from_secs(1));
        let second = acquired_rx.recv_timeout(Duration::from_secs(1));
        for release in releases {
            release.send(()).unwrap();
        }
        for reader in readers {
            reader.join().unwrap();
        }
        assert_ne!(first.unwrap(), second.unwrap());

        let key = format!("AI_AGENTS_EVAL_ENV_TEST_{}", uuid::Uuid::new_v4());
        unsafe {
            std::env::remove_var(&key);
        }
        let reader = EnvGuard::apply(&HashMap::new()).unwrap();
        assert!(matches!(&reader._guard, EnvExclusionGuard::Read { .. }));
        let writer_start = Arc::new(Barrier::new(2));
        let writer_start_thread = Arc::clone(&writer_start);
        let (writer_tx, writer_rx) = mpsc::channel();
        let writer_key = key.clone();
        let writer = thread::spawn(move || {
            writer_start_thread.wait();
            let guard = EnvGuard::apply(&HashMap::from([(
                writer_key.clone(),
                "temporary".to_string(),
            )]))
            .unwrap();
            writer_tx.send(()).unwrap();
            assert_eq!(std::env::var(&writer_key).as_deref(), Ok("temporary"));
            drop(guard);
        });
        writer_start.wait();
        let writer_was_blocked = writer_rx.recv_timeout(Duration::from_millis(100)).is_err();
        drop(reader);
        if writer_was_blocked {
            writer_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        }
        writer.join().unwrap();
        assert!(writer_was_blocked);
        assert!(std::env::var(&key).is_err());
    }

    fn write_test_agent(dir: &Path) {
        std::fs::write(
            dir.join("agent.yaml"),
            r#"
name: TestAgent
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4.1-nano
"#,
        )
        .unwrap();
    }

    async fn run_test_suite(dir: &Path, name: &str, yaml: &str) -> EvalResult {
        let suite_path = dir.join(name);
        std::fs::write(&suite_path, yaml).unwrap();
        let options = EvalRunnerOptions {
            output: dir.join("out"),
            ..Default::default()
        };
        EvalRunner::from_file(&suite_path, options)
            .unwrap()
            .run()
            .await
            .unwrap()
    }

    #[test]
    fn runtime_error_expectations_retain_turns_and_control_retries() {
        std::thread::Builder::new()
            .name("eval-runtime-error-test".to_string())
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let runtime = tokio::runtime::Runtime::new().unwrap();
                runtime.block_on(async {
                    let dir = std::env::temp_dir().join(format!(
                        "ai_agents_eval_runtime_error_test_{}",
                        uuid::Uuid::new_v4()
                    ));
                    std::fs::create_dir_all(&dir).unwrap();
                    write_test_agent(&dir);
                    let errors = run_test_suite(
                        &dir,
                        "errors.yaml",
                        r#"
name: Runtime Errors
agent: agent.yaml
settings:
  retries: 1
  retry_delay_ms: 0
  redact_outputs: false
fixtures:
  llm:
    mode: mock
    errors_by_alias:
      default: provider exploded
scenarios:
  - id: expected
    turns:
      - input: Hello
        expect_error: [timeout, provider exploded]
  - id: mismatched
    turns:
      - input: Hello
        expect_error: permission denied
  - id: unexpected
    turns:
      - input: Hello
"#,
                    )
                    .await;

                    let expected = &errors.scenarios[0];
                    assert!(expected.status.is_passed());
                    assert_eq!(expected.attempts.len(), 1);
                    let expected_turn = &expected.attempts[0].turns[0];
                    assert!(!expected_turn.response_present);
                    assert!(expected_turn.runtime_error.is_some());
                    assert!(
                        expected_turn
                            .assertion_results
                            .iter()
                            .any(|detail| detail.assertion == "expect_error" && detail.passed)
                    );

                    for scenario in &errors.scenarios[1..] {
                        assert!(scenario.status.is_error());
                        assert_eq!(scenario.attempts.len(), 2);
                        assert!(
                            scenario
                                .attempts
                                .iter()
                                .all(|attempt| attempt.turns.len() == 1)
                        );
                        assert!(
                            scenario
                                .attempts
                                .iter()
                                .all(|attempt| attempt.turns[0].runtime_error.is_some())
                        );
                    }

                    let missing = run_test_suite(
                        &dir,
                        "missing.yaml",
                        r#"
name: Missing Runtime Error
agent: agent.yaml
settings:
  retry_delay_ms: 0
  redact_outputs: false
fixtures:
  llm:
    mode: mock
    responses: [ok]
scenarios:
  - id: missing
    turns:
      - input: Hello
        expect_error: timeout
"#,
                    )
                    .await;
                    let missing = &missing.scenarios[0];
                    assert!(missing.status.is_failed());
                    let turn = &missing.attempts[0].turns[0];
                    assert!(turn.response_present);
                    assert!(turn.runtime_error.is_none());
                    assert!(
                        turn.assertion_results
                            .iter()
                            .any(|detail| detail.assertion == "expect_error" && !detail.passed)
                    );
                    let _ = std::fs::remove_dir_all(dir);
                });
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn scenario_budget_is_shared_across_retries_and_agent_resets() {
        std::thread::Builder::new()
            .name("eval-budget-lifecycle-test".to_string())
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let runtime = tokio::runtime::Runtime::new().unwrap();
                runtime.block_on(async {
                    let dir = std::env::temp_dir().join(format!(
                        "ai_agents_eval_budget_lifecycle_test_{}",
                        uuid::Uuid::new_v4()
                    ));
                    std::fs::create_dir_all(&dir).unwrap();
                    write_test_agent(&dir);

                    let retried = run_test_suite(
                        &dir,
                        "budget-retry.yaml",
                        r#"
name: Retry Budget
agent: agent.yaml
settings:
  retries: 1
  retry_delay_ms: 0
  redact_outputs: false
fixtures:
  llm:
    mode: mock
    responses: [wrong]
scenarios:
  - id: retry
    budget:
      max_llm_calls: 1
    turns:
      - input: Hello
        assert:
          response_contains: right
"#,
                    )
                    .await;
                    let retried = &retried.scenarios[0];
                    assert!(retried.status.is_error());
                    assert_eq!(retried.attempts.len(), 2);
                    assert!(
                        retried.attempts[1].turns[0]
                            .runtime_error
                            .as_ref()
                            .is_some_and(|error| error.value.contains("max_llm_calls=1"))
                    );

                    let reset = run_test_suite(
                        &dir,
                        "budget-reset.yaml",
                        r#"
name: Reset Budget
agent: agent.yaml
settings:
  retries: 0
  redact_outputs: false
fixtures:
  llm:
    mode: mock
    responses: [ok]
scenarios:
  - id: reset
    budget:
      max_llm_calls: 1
    steps:
      - !run
        turns:
          - input: First
      - !reset_agent true
      - !run
        turns:
          - input: Second
"#,
                    )
                    .await;
                    let reset = &reset.scenarios[0];
                    assert!(reset.status.is_error());
                    assert_eq!(reset.attempts.len(), 1);
                    assert_eq!(reset.attempts[0].turns.len(), 2);
                    assert!(
                        reset.attempts[0].turns[1]
                            .runtime_error
                            .as_ref()
                            .is_some_and(|error| error.value.contains("max_llm_calls=1"))
                    );
                    let _ = std::fs::remove_dir_all(dir);
                });
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn captures_composed_llm_requests_per_turn_across_reset() {
        std::thread::Builder::new()
            .name("eval-llm-evidence-test".to_string())
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let runtime = tokio::runtime::Runtime::new().unwrap();
                runtime.block_on(async {
                    let dir = std::env::temp_dir().join(format!(
                        "ai_agents_eval_llm_evidence_test_{}",
                        uuid::Uuid::new_v4()
                    ));
                    std::fs::create_dir_all(&dir).unwrap();
                    std::fs::write(
                        dir.join("agent.yaml"),
                        r#"
name: EvidenceAgent
system_prompt: "Base instruction marker."
llm:
  provider: openai
  model: gpt-4.1-nano
persona:
  identity:
    name: Evidence Guide
    role: Prompt Inspector
reasoning:
  mode: cot
  output: tagged
"#,
                    )
                    .unwrap();
                    let result = run_test_suite(
                        &dir,
                        "llm-evidence.yaml",
                        r#"
name: LLM Evidence
agent: agent.yaml
settings:
  redact_outputs: false
fixtures:
  llm:
    mode: mock
    responses: [first answer, second answer]
scenarios:
  - id: composed
    steps:
      - !run
        turns:
          - input: first question
            assert:
              llm_request:
                system_contains:
                  - "You are Evidence Guide, Prompt Inspector."
                  - "Base instruction marker."
                  - "Think through this step by step"
                  - "<instruction>"
                user_contains: first question
                count: 1
                same_request: true
          - input: second question
            assert:
              llm_request:
                user_contains: [first question, second question]
                assistant_contains: first answer
                count: 1
                same_request: true
      - !reset_agent true
      - !run
        turns:
          - input: after reset
            assert:
              llm_request:
                system_contains: "Base instruction marker."
                user_contains: after reset
                count: 1
                same_request: true
"#,
                    )
                    .await;

                    assert_eq!(result.passed, 1);
                    let turns = &result.scenarios[0].attempts[0].turns;
                    assert_eq!(turns.len(), 3);
                    assert!(
                        turns
                            .iter()
                            .all(|turn| turn.evidence.llm_requests.len() == 1)
                    );
                    let second_messages = &turns[1].evidence.llm_requests[0].messages;
                    assert!(second_messages.iter().any(|message| {
                        message.role == ai_agents_core::Role::Assistant
                            && message.content.contains("first answer")
                    }));
                    let reset_messages = &turns[2].evidence.llm_requests[0].messages;
                    assert!(
                        reset_messages
                            .iter()
                            .all(|message| !message.content.contains("first question"))
                    );
                    assert!(
                        reset_messages
                            .iter()
                            .all(|message| !message.content.contains("first answer"))
                    );

                    let serialized = serde_json::to_string(&result).unwrap();
                    let serialized_value: Value = serde_json::from_str(&serialized).unwrap();
                    assert!(
                        serialized_value["scenarios"][0]["attempts"][0]["turns"][0]
                            .get("evidence")
                            .is_none()
                    );
                    assert!(!serialized.contains("Base instruction marker"));
                    assert!(!serialized.contains("Evidence Guide"));
                    assert!(!serialized.contains("Think through this step by step"));
                    let _ = std::fs::remove_dir_all(dir);
                });
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn runner_executes_mocked_suite_and_redacts_outputs() {
        std::thread::Builder::new()
            .name("eval-runner-test".to_string())
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let runtime = tokio::runtime::Runtime::new().unwrap();
                runtime.block_on(async {
                    let dir = std::env::temp_dir().join(format!(
                        "ai_agents_eval_runner_test_{}",
                        uuid::Uuid::new_v4()
                    ));
                    std::fs::create_dir_all(&dir).unwrap();
                    write_test_agent(&dir);
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
                });
            })
            .unwrap()
            .join()
            .unwrap();
    }
}

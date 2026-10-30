use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Instant;

use ai_agents_hooks::AgentHooks;
use ai_agents_observability::config::{ExportConfig, ExportFormat};
use ai_agents_observability::{CostEstimator, ObservabilityConfig};
use ai_agents_runtime::spec::{AgentSpec, LLMConfigOrSelector, StorageConfig};
use ai_agents_runtime::{Agent, AgentBuilder, AgentStreamEvent, RuntimeAgent, StreamChunk};
use async_trait::async_trait;
use futures::{StreamExt, stream};
use serde_json::{Value, json};
use tokio::time::{Duration, Instant as TokioInstant, timeout_at};

use crate::assertion::{
    Assertion, AssertionEvalContext, AssertionOutcome, AssertionResultDetail, evaluate_assertion,
};
use crate::budget::{BudgetProviderConfig, ScenarioBudgetTracker};
use crate::compatibility::suite_from_jsonl;
use crate::evidence::{
    ApprovalEvidence, LlmRequestEvidence, collect_turn_evidence_since, relationship_snapshot,
};
use crate::fixtures::{
    AttemptFixtureContext, AttemptWorkspace, LlmFixtureMode, RecordingToolLog,
    WorkspacePolicyFixtureConfig, build_approval_handler, build_llm_registry,
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

// Transparent instrumentation preserves native requests, configured choices and every installed admission/pricing capability.
struct RecordingProvider {
    inner: Arc<dyn ai_agents_core::LLMProvider>,
    log: RecordingLlmLog,
}
#[async_trait]
impl ai_agents_core::LLMProvider for RecordingProvider {
    async fn complete(
        &self,
        messages: &[ai_agents_core::ChatMessage],
        config: Option<&ai_agents_core::LLMConfig>,
    ) -> std::result::Result<ai_agents_core::LLMResponse, ai_agents_core::LLMError> {
        self.log.push_messages(messages);
        self.inner.complete(messages, config).await
    }
    async fn complete_with_tools(
        &self,
        messages: &[ai_agents_core::ChatMessage],
        config: Option<&ai_agents_core::LLMConfig>,
        request: &ai_agents_core::LLMToolRequest,
    ) -> std::result::Result<ai_agents_core::LLMResponse, ai_agents_core::LLMError> {
        self.log.push_messages(messages);
        self.inner
            .complete_with_tools(messages, config, request)
            .await
    }
    async fn complete_stream(
        &self,
        messages: &[ai_agents_core::ChatMessage],
        config: Option<&ai_agents_core::LLMConfig>,
    ) -> std::result::Result<
        Box<
            dyn futures::Stream<
                    Item = std::result::Result<ai_agents_core::LLMChunk, ai_agents_core::LLMError>,
                > + Unpin
                + Send,
        >,
        ai_agents_core::LLMError,
    > {
        self.log.push_messages(messages);
        self.inner.complete_stream(messages, config).await
    }
    fn provider_name(&self) -> &str {
        self.inner.provider_name()
    }
    fn supports(&self, feature: ai_agents_core::LLMFeature) -> bool {
        self.inner.supports(feature)
    }
    fn configured_tool_choice(&self) -> Option<ai_agents_core::ToolChoice> {
        self.inner.configured_tool_choice()
    }
    fn supports_tool_choice(&self, choice: &ai_agents_core::ToolChoice) -> bool {
        self.inner.supports_tool_choice(choice)
    }
    fn is_terminal_error(&self, error: &ai_agents_core::LLMError) -> bool {
        self.inner.is_terminal_error(error)
    }
    fn manages_invocation_admission(&self) -> bool {
        self.inner.manages_invocation_admission()
    }
    fn priced_capability_identity(&self) -> Option<String> {
        self.inner.priced_capability_identity()
    }
    fn request_cost_bound(
        &self,
        request: &ai_agents_core::autonomy::ProviderRequest<'_>,
    ) -> std::result::Result<
        Option<ai_agents_core::autonomy::ProviderCostBound>,
        ai_agents_core::LLMError,
    > {
        self.inner.request_cost_bound(request)
    }
    fn settle_request_cost(
        &self,
        bound: &ai_agents_core::autonomy::ProviderCostBound,
        response: &ai_agents_core::LLMResponse,
    ) -> std::result::Result<
        Option<ai_agents_core::autonomy::ProviderCostSettlement>,
        ai_agents_core::LLMError,
    > {
        self.inner.settle_request_cost(bound, response)
    }
    fn settle_stream_request_cost(
        &self,
        bound: &ai_agents_core::autonomy::ProviderCostBound,
        usage: Option<&ai_agents_core::TokenUsage>,
    ) -> std::result::Result<
        Option<ai_agents_core::autonomy::ProviderCostSettlement>,
        ai_agents_core::LLMError,
    > {
        self.inner.settle_stream_request_cost(bound, usage)
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
    // Recording is caller-owned in-memory work with no hidden provider calls or detached effects.
    fn task_providers(&self) -> Option<Vec<Arc<dyn ai_agents_core::LLMProvider>>> {
        Some(Vec::new())
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
    host: Option<Arc<dyn crate::task::EvalHost>>,
}

// Bundles attempt-local construction state so resets rebuild with the same isolation, logging, approval, and budget inputs.
struct BuildAgentParams<'a> {
    agent_path: &'a Path,
    base_dir: &'a Path,
    attempt_context: &'a AttemptFixtureContext,
    tool_log: RecordingToolLog,
    approval_log: RecordingApprovalLog,
    llm_log: RecordingLlmLog,
    approval_handler: Option<Arc<dyn ai_agents_hitl::ApprovalHandler>>,
    budget: Option<ScenarioBudgetTracker>,
}

// Bundles the agent, scenario, turn, and attempt-local evidence logs used to produce one turn result.
// State outlives the timed future so interrupted attempts retain completed evidence and actual participant custody.
struct AttemptState {
    session: ai_agents_runtime::autonomy::TaskHostSession,
    deadline: Option<TokioInstant>,
    partial: Mutex<Vec<TurnResult>>,
    initializing: Arc<std::sync::atomic::AtomicBool>,
    cleanup_uncertain: Arc<std::sync::atomic::AtomicBool>,
    interactions: crate::interactions::Interactions,
}
impl AttemptState {
    fn push(&self, turns: &mut Vec<TurnResult>, result: TurnResult) {
        turns.push(result);
        *self
            .partial
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = turns.clone();
    }
    fn partial(&self) -> Vec<TurnResult> {
        let mut turns = self
            .partial
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let state = self.interactions.lock();
        let interactions = state.results[state.reported..].to_vec();
        if !interactions.is_empty() {
            turns.push(TurnResult {
                index: turns.len(),
                input: crate::redaction::RedactedString::redacted("[interrupted foreground]"),
                response: crate::redaction::RedactedString::plain(""),
                response_present: false,
                runtime_error: Some(crate::redaction::RedactedString::redacted(
                    "foreground did not acknowledge a result",
                )),
                state: None,
                metadata: None,
                task: None,
                assertion_results: interactions
                    .iter()
                    .flat_map(|i| i.assertion_results.clone())
                    .collect(),
                interactions,
                evidence: Default::default(),
                latency_ms: state
                    .foreground_started
                    .map(|start| start.elapsed().as_millis() as u64)
                    .unwrap_or_default(),
                observability_span_id: None,
            });
        }
        turns
    }
}

struct RunTurnParams<'a> {
    agent: &'a RuntimeAgent,
    tasks: &'a mut crate::task::TaskContext,
    task: Option<&'a crate::task::TaskStep>,
    scenario: &'a Scenario,
    turn: &'a Turn,
    index: usize,
    tool_log: &'a RecordingToolLog,
    approval_log: &'a RecordingApprovalLog,
    llm_log: &'a RecordingLlmLog,
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
            host: None,
        })
    }

    /// Installs caller-owned domain adapters before construction without replacing eval fixtures or execution ownership.
    pub fn with_host(mut self, host: Arc<dyn crate::task::EvalHost>) -> Self {
        self.host = Some(host);
        self
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
            let agent = suite.agent.clone().ok_or_else(|| {
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
        })?;
        suite.validate_agent(&spec)
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
            schema_version: 3,
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
                    results.push(error_result(
                        scenario,
                        error,
                        FailureCategory::RuntimeError,
                        self.suite.settings.redact_outputs,
                    ));
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
                    error_result(
                        scenario,
                        error,
                        FailureCategory::RuntimeError,
                        self.suite.settings.redact_outputs,
                    )
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
            let state = AttemptState {
                session: ai_agents_runtime::autonomy::TaskHostSession::new(true),
                deadline: self
                    .suite
                    .settings
                    .timeout_per_scenario_ms
                    .map(|ms| TokioInstant::now() + Duration::from_millis(ms)),
                partial: Default::default(),
                initializing: Default::default(),
                cleanup_uncertain: Default::default(),
                interactions: Arc::new(parking_lot::Mutex::new(Default::default())),
            };
            let attempt_future = self.run_attempt(
                agent_path,
                base_dir,
                scenario,
                attempt_idx,
                budget.clone(),
                &state,
            );
            let attempt = if let Some(deadline) = state.deadline {
                match crate::task::drive(&state.session, attempt_future, deadline).await {
                    Ok((Ok(mut result), deadline_error)) => {
                        if let Some(reason) = deadline_error {
                            result.status = ScenarioStatus::Error {
                                message: redact_text(
                                    &reason,
                                    self.suite.settings.redact_outputs,
                                    0,
                                )
                                .value,
                            };
                        }
                        Ok(result)
                    }
                    Ok((Err(error), _)) | Err(error) => Err(error),
                }
            } else {
                attempt_future.await
            };
            let protected = state.session.has_protected_work()
                || state
                    .cleanup_uncertain
                    .load(std::sync::atomic::Ordering::Acquire)
                || state
                    .initializing
                    .load(std::sync::atomic::Ordering::Acquire);
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
                    if protected {
                        break;
                    }
                }
                Err(error) => {
                    final_status = ScenarioStatus::Error {
                        message: redact_text(
                            &error.to_string(),
                            self.suite.settings.redact_outputs,
                            0,
                        )
                        .value,
                    };
                    category = Some(FailureCategory::RuntimeError);
                    attempts.push(AttemptResult {
                        attempt: attempt_idx,
                        turns: state.partial(),
                        status: final_status.clone(),
                        duration_ms: 0,
                    });
                    if protected {
                        break;
                    }
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
        state: &AttemptState,
    ) -> Result<AttemptResult> {
        let start = Instant::now();
        let _env_guard = EnvGuard::apply(&scenario.env)?;
        let mock_server = start_mock_server(self.suite.fixtures.mock_server.as_ref()).await?;
        let attempt_context = AttemptWorkspace::create(mock_server.as_ref())?;
        attempt_context.materialize_files(&self.suite.fixtures)?;
        let tool_log = RecordingToolLog::new();
        let approval_log = RecordingApprovalLog::default();
        let llm_log = RecordingLlmLog::default();
        let approval_handler = self
            .suite
            .fixtures
            .approvals
            .as_ref()
            .map(build_approval_handler);
        let mut agent = Arc::new(
            self.build_agent(BuildAgentParams {
                agent_path,
                base_dir,
                attempt_context: &attempt_context,
                tool_log: tool_log.clone(),
                approval_log: approval_log.clone(),
                llm_log: llm_log.clone(),
                approval_handler: approval_handler.clone(),
                budget: budget.clone(),
            })
            .await?,
        );
        apply_base_context(&agent, &self.suite, base_dir, scenario, &attempt_context)?;
        let mut turns = Vec::new();
        let mut status = ScenarioStatus::Passed;
        let mut tasks = crate::task::TaskContext::with_session(state.session.clone());
        let workspace_hold = attempt_context.hold_for_tasks();
        tasks.workspace_hold = Some(workspace_hold.clone());
        tasks.interactions = state.interactions.clone();
        tasks.interactions.lock().plan = scenario.interactions.clone();
        tasks.redact_interactions = self.suite.settings.redact_outputs;
        tasks.deadline = state.deadline;
        tasks.initializing = Some(state.initializing.clone());
        tasks.cleanup_uncertain = Some(state.cleanup_uncertain.clone());
        if let Some(host) = &self.host {
            state
                .initializing
                .store(true, std::sync::atomic::Ordering::Release);
            if let Some((runner, result)) = host.prepare(agent.clone(), &attempt_context).await? {
                tasks
                    .session
                    .observe_result(runner.clone(), &result)
                    .await?;
                agent = runner.runtime().clone();
                tasks.runner_override = Some(runner);
                tasks.last = Some(result);
            }
            state
                .initializing
                .store(false, std::sync::atomic::Ordering::Release);
        }

        if !scenario.turns.is_empty() {
            for (idx, turn) in scenario.turns.iter().enumerate() {
                let turn_execution = self
                    .run_turn(RunTurnParams {
                        agent: &agent,
                        tasks: &mut tasks,
                        task: None,
                        scenario,
                        turn,
                        index: idx,
                        tool_log: &tool_log,
                        approval_log: &approval_log,
                        llm_log: &llm_log,
                    })
                    .await?;
                let turn_failed = turn_execution
                    .result
                    .assertion_results
                    .iter()
                    .any(|result| !result.passed);
                let runtime_error = turn_execution.unhandled_runtime_error;
                state.push(&mut turns, turn_execution.result);
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
                            .run_turn(RunTurnParams {
                                agent: &agent,
                                tasks: &mut tasks,
                                task: None,
                                scenario,
                                turn,
                                index: idx,
                                tool_log: &tool_log,
                                approval_log: &approval_log,
                                llm_log: &llm_log,
                            })
                            .await?;
                        let turn_failed = turn_execution
                            .result
                            .assertion_results
                            .iter()
                            .any(|result| !result.passed);
                        let runtime_error = turn_execution.unhandled_runtime_error;
                        state.push(&mut turns, turn_execution.result);
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
                    if status.is_passed()
                        && let Some(session) = &run.save_session
                    {
                        agent.save_session(session).await?;
                    }
                }
                ScenarioStep::Task(task) => {
                    let turn = Turn {
                        input: String::new(),
                        actor: None,
                        context: Value::Null,
                        stream: None,
                        timeout_ms: task.timeout_ms,
                        assertions: Some(task.assertions.clone()),
                    };
                    let execution = self
                        .run_turn(RunTurnParams {
                            agent: &agent,
                            tasks: &mut tasks,
                            task: Some(task),
                            scenario,
                            turn: &turn,
                            index: turns.len(),
                            tool_log: &tool_log,
                            approval_log: &approval_log,
                            llm_log: &llm_log,
                        })
                        .await?;
                    let failed = execution
                        .result
                        .assertion_results
                        .iter()
                        .any(|result| !result.passed);
                    state.push(&mut turns, execution.result);
                    if let Some(message) = execution.unhandled_runtime_error {
                        status = ScenarioStatus::Error { message };
                    } else if failed {
                        status = ScenarioStatus::Failed {
                            reason: "task assertion failed".into(),
                        };
                    }
                }
                ScenarioStep::ResetAgent(reset) => {
                    tasks.cleanup().await?;
                    let deadline = tasks.deadline;
                    let session = tasks.session.clone();
                    let interactions = tasks.interactions.clone();
                    tasks = crate::task::TaskContext::with_session(session);
                    tasks.interactions = interactions;
                    tasks.redact_interactions = self.suite.settings.redact_outputs;
                    tasks.deadline = deadline;
                    tasks.cleanup_uncertain = Some(state.cleanup_uncertain.clone());
                    tasks.workspace_hold = Some(workspace_hold.clone());
                    workspace_hold.store(true, std::sync::atomic::Ordering::Release);
                    if let Some(options) = reset_options(reset) {
                        if options.delete_persistence || !options.preserve_storage {
                            let _ = std::fs::remove_dir_all(&attempt_context.workspace);
                            std::fs::create_dir_all(&attempt_context.workspace)?;
                            attempt_context.materialize_files(&self.suite.fixtures)?;
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
                            agent = Arc::new(
                                self.build_agent(BuildAgentParams {
                                    agent_path,
                                    base_dir,
                                    attempt_context: &attempt_context,
                                    tool_log: tool_log.clone(),
                                    approval_log: approval_log.clone(),
                                    llm_log: llm_log.clone(),
                                    approval_handler: approval_handler.clone(),
                                    budget: budget.clone(),
                                })
                                .await?,
                            );
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

        let interactions_complete = {
            let interactions = tasks.interactions.lock();
            interactions.claimed.len() == scenario.interactions.len()
                && interactions.error.is_none()
        };
        if !interactions_complete && status.is_passed() {
            status = ScenarioStatus::Failed {
                reason: "not all declared participant interactions were consumed".into(),
            };
        }
        match tasks.cleanup().await {
            Ok(()) => workspace_hold.store(false, std::sync::atomic::Ordering::Release),
            Err(_) => status = ScenarioStatus::Error {
                message:
                    "task cleanup unacknowledged; attempt workspace retained; no automatic retry"
                        .into(),
            },
        }
        Ok(AttemptResult {
            attempt,
            turns,
            status,
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    // Keep construction state out of attempt futures while retaining same-task polling and drop cleanup.
    fn build_agent<'a>(
        &'a self,
        params: BuildAgentParams<'a>,
    ) -> std::pin::Pin<Box<impl std::future::Future<Output = Result<RuntimeAgent>> + 'a>> {
        Box::pin(self.build_agent_inner(params))
    }

    // Preserve the builder order and fixture bindings for initial construction and full resets.
    async fn build_agent_inner(&self, params: BuildAgentParams<'_>) -> Result<RuntimeAgent> {
        let BuildAgentParams {
            agent_path,
            base_dir,
            attempt_context,
            tool_log,
            approval_log,
            llm_log,
            approval_handler,
            budget,
        } = params;
        let content = std::fs::read_to_string(agent_path)?;
        let parsed = AgentSpec::from_yaml_strict(&content)?;
        let mut spec = attempt_context.interpolate_config(&parsed)?;
        apply_eval_llm_settings(&mut spec, &self.suite.settings);
        isolate_spec_storage(&mut spec, attempt_context);
        apply_workspace_policy(
            &mut spec,
            self.suite.fixtures.workspace_policy.as_ref(),
            attempt_context,
        )?;
        spec.validate()
            .map_err(|error| EvalError::Config(error.to_string()))?;
        self.suite.validate_agent(&spec)?;
        let llm_fixture = attempt_context.interpolate_llm_fixture(&self.suite.fixtures.llm)?;
        let provider_configs = budget_provider_configs(&spec);
        let (mut llm_registry, _judge_llm) = build_llm_registry(&spec, &llm_fixture, base_dir)?;
        // Capture actual dispatch, including scripted skills and validation, after any enclosing budget has admitted it.
        llm_registry = llm_registry.map_providers(|_, provider| {
            Arc::new(RecordingProvider {
                inner: provider,
                log: llm_log.clone(),
            })
        });
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

        // Tool fixture values are typed JSON; LLM content keeps its dedicated JSON-aware interpolation path.
        let mut tool_fixtures = self.suite.fixtures.clone();
        tool_fixtures.llm = Default::default();
        let tool_fixtures = attempt_context.interpolate_config(&tool_fixtures)?;
        let tool_registry = crate::fixtures::build_tool_registry_for_routing(
            &tool_fixtures,
            tool_log.clone(),
            spec.llm.router_roles().is_some(),
        )?;
        let agent_base_dir = agent_path.parent().unwrap_or_else(|| Path::new("."));
        if let Some(host) = &self.host {
            host.validate(&spec)?;
        }
        let builder = AgentBuilder::from_spec_with_base_dir(spec, agent_base_dir)
            .llm_registry(llm_registry)
            .tools(tool_registry)
            .hooks(Arc::new(EvalRecordHooks {
                tool_log,
                approval_log,
            }));
        let builder = if let Some(host) = &self.host {
            host.configure(builder, attempt_context)?
        } else {
            builder
        };
        let mut builder = builder
            .auto_configure_features()
            .map_err(|error| EvalError::Config(error.to_string()))?
            .auto_configure_mcp()
            .await
            .map_err(|error| EvalError::Config(error.to_string()))?;

        // Only the cancellation regression pauses here; normal builds have no extra suspension point.
        #[cfg(test)]
        if let Ok(gate) = tests::CONSTRUCTION_GATE.try_with(Arc::clone) {
            gate.acquire()
                .await
                .expect("Construction test gate is open")
                .forget();
        }

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
            .task_interaction_delivery(true)
            .build_task_host()
            .await
            .map_err(|error| EvalError::Config(error.to_string()))?;
        agent.init_storage().await?;
        Ok(agent)
    }

    fn observability_config(&self, base_dir: &Path) -> Result<Option<ObservabilityConfig>> {
        let mut config = if let Some(config) = self.suite.observability.clone() {
            config
        } else if self.options.observability {
            ObservabilityConfig {
                enabled: true,
                export: ExportConfig {
                    formats: vec![ExportFormat::Json],
                    path: self
                        .options
                        .output
                        .join("observability")
                        .display()
                        .to_string(),
                    write_report: true,
                    ..Default::default()
                },
                ..Default::default()
            }
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

    // A turn can enter deep retained tool/validator execution; erase its polling frame without starting or detaching work.
    fn run_turn<'a>(
        &'a self,
        params: RunTurnParams<'a>,
    ) -> futures::future::BoxFuture<'a, Result<TurnExecution>> {
        Box::pin(self.run_turn_inner(params))
    }

    async fn run_turn_inner(&self, params: RunTurnParams<'_>) -> Result<TurnExecution> {
        let RunTurnParams {
            agent,
            tasks,
            task,
            scenario,
            turn,
            index,
            tool_log,
            approval_log,
            llm_log,
        } = params;
        if let Some(assertion) = &turn.assertions {
            crate::assertion::preflight_judge_aliases(assertion, agent.llm_registry())?;
        }
        apply_context_value(agent, &turn_runtime_context(turn))?;
        if let Some(actor) = &turn.actor {
            agent.set_actor_id(actor)?;
        }
        let before_relationship = relationship_snapshot(agent);
        let tool_start = tool_log.len();
        let approval_start = approval_log.len();
        let llm_start = llm_log.len();
        let observability_cursor = agent.observability().map(|manager| manager.event_cursor());
        let start = Instant::now();
        let timeout_ms = turn
            .timeout_ms
            .unwrap_or(self.suite.settings.timeout_per_turn_ms);
        let timeout_ms = tasks.remaining(timeout_ms);
        if let Some(hold) = &tasks.workspace_hold {
            hold.store(true, std::sync::atomic::Ordering::Release);
        }
        tasks.events.clear();
        tasks.events_complete = true;
        let interaction_start = {
            let mut state = tasks.interactions.lock();
            state.foreground_started = Some(start);
            state.results.len()
        };
        let mut task_result = None;
        let mut operation = if timeout_ms == 0 {
            TurnOperation::error("scenario deadline reached before task admission".into())
        } else if let Some(step) = task {
            match tasks.operate(agent, step, timeout_ms).await {
                Ok((result, runtime_error)) => {
                    let response = result.final_response.clone();
                    task_result = Some(result);
                    TurnOperation {
                        task_events: Vec::new(),
                        events_complete: true,
                        response_content: response
                            .as_ref()
                            .map_or_else(String::new, |r| r.content.clone()),
                        response_metadata: response.as_ref().and_then(|r| r.metadata.clone()),
                        response_present: response.is_some(),
                        runtime_error,
                    }
                }
                Err(error) => TurnOperation::error(error.to_string()),
            }
        } else {
            let work = async {
                if turn.stream.unwrap_or(false) {
                    collect_stream_response(agent, &turn.input, timeout_ms).await
                } else {
                    match agent.chat(&turn.input).await {
                        Ok(response) => TurnOperation {
                            task_events: Vec::new(),
                            events_complete: false,
                            response_content: response.content,
                            response_metadata: response.metadata,
                            response_present: true,
                            runtime_error: None,
                        },
                        Err(error) => TurnOperation::error(error.to_string()),
                    }
                }
            };
            let foreground_runtime = agent as *const RuntimeAgent as usize;
            match crate::task::drive(
                &tasks.session,
                crate::interactions::service(
                    &tasks.session,
                    work,
                    tasks.interactions.clone(),
                    self.suite.settings.redact_outputs,
                    foreground_runtime,
                ),
                TokioInstant::now() + Duration::from_millis(timeout_ms),
            )
            .await
            {
                Ok((Ok(mut operation), deadline_error)) => {
                    if let Some(reason) = deadline_error {
                        operation.runtime_error.get_or_insert(reason);
                    }
                    operation
                }
                Ok((Err(error), _)) | Err(error) => TurnOperation::error(error.to_string()),
            }
        };
        if let Err(error) = agent.flush_background_tasks().await
            && operation.runtime_error.is_none()
        {
            operation.runtime_error = Some(error.to_string());
        }
        let latency_ms = start.elapsed().as_millis() as u64;
        let mut evidence = collect_turn_evidence_since(
            agent,
            operation.response_metadata.clone(),
            tool_log,
            tool_start,
            before_relationship,
            observability_cursor,
        );
        for event in std::mem::take(&mut operation.task_events) {
            tasks.record(event);
        }
        tasks.events_complete &= operation.events_complete;
        if task_result.is_none()
            && let Some(result) = agent.automatic_task_result()
        {
            let matches = operation
                .response_metadata
                .as_ref()
                .and_then(|m| m.get("task_run"))
                .is_some_and(|m| {
                    m.get("run_id").and_then(Value::as_str) == Some(result.run.key.run_id.as_str())
                        && m.get("revision").and_then(Value::as_u64) == Some(result.run.revision)
                });
            if matches {
                task_result = Some(result);
            }
        }
        let mut task_report = None;
        if let Some(result) = task_result {
            // The coordinating ledger includes actual child tools, even when their local hooks are not the parent's eval hook.
            // Only the new acknowledged suffix belongs to this operation; completed prefixes must not be counted again on resume/inspection.
            let prefix = tasks
                .last
                .as_ref()
                .filter(|previous| previous.run.key == result.run.key)
                .map_or(0, |previous| previous.run.evidence.tool_calls.len());
            let operation_tools = RecordingToolLog::new();
            for record in result.run.evidence.tool_calls.iter().skip(prefix) {
                operation_tools.push_executor_record(&record.record);
            }
            evidence.tool_executions = operation_tools.records_since(0);
            let (observations, report) = tasks.capture(agent, result).await?;
            evidence.task = Some(observations);
            task_report = Some(report);
        }
        evidence.approvals = approval_log.records_since(approval_start);
        evidence.llm_requests = llm_log.records_since(llm_start);
        let judge = self.build_judge(agent);
        let mut assertion_error = None;
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
                AssertionOutcome::Error(message) => {
                    assertion_error = Some(message);
                    vec![AssertionResultDetail {
                        assertion: "evidence_unavailable".into(),
                        passed: false,
                        actual: Value::Null,
                        expected: json!("available evidence"),
                        message: None,
                    }]
                }
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

        let expected_error = task
            .and_then(|step| step.expect_error.clone())
            .or_else(|| turn_expected_error(turn));
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
        let interactions = {
            let mut state = tasks.interactions.lock();
            state.reported = state.results.len();
            state.results[interaction_start..].to_vec()
        };
        for interaction in &interactions {
            assertion_results.extend(interaction.assertion_results.iter().cloned().map(
                |mut detail| {
                    detail.assertion =
                        format!("interaction.{}.{}", interaction.index, detail.assertion);
                    detail
                },
            ));
        }
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
        let unhandled_runtime_error = assertion_error
            .or(unhandled_runtime_error)
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
                task: task_report,
                interactions,
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
    task_events: Vec<ai_agents_runtime::autonomy::TaskRunEvent>,
    events_complete: bool,
    response_content: String,
    response_metadata: Option<HashMap<String, Value>>,
    response_present: bool,
    runtime_error: Option<String>,
}

impl TurnOperation {
    fn error(message: String) -> Self {
        Self {
            task_events: Vec::new(),
            events_complete: true,
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
    let stream = match agent.chat_stream_events(input).await {
        Ok(stream) => stream,
        Err(error) => return TurnOperation::error(error.to_string()),
    };
    consume_stream_response_inner(stream, None, timeout_ms).await
}

// Uses Final as the successful result and treats the first Error as terminal; expected HITL rejection is delivered as Final.
#[cfg(test)]
async fn consume_stream_response<S>(
    stream: S,
    deadline: TokioInstant,
    timeout_ms: u64,
) -> TurnOperation
where
    S: futures::Stream<Item = AgentStreamEvent> + Unpin,
{
    consume_stream_response_inner(stream, Some(deadline), timeout_ms).await
}

async fn consume_stream_response_inner<S>(
    mut stream: S,
    deadline: Option<TokioInstant>,
    timeout_ms: u64,
) -> TurnOperation
where
    S: futures::Stream<Item = AgentStreamEvent> + Unpin,
{
    let mut task_events = Vec::new();
    let mut events_complete = true;
    let mut partial_content = String::new();
    let mut partial_content_seen = false;
    let mut runtime_error = None;
    let mut final_response = None;
    loop {
        let event = if let Some(deadline) = deadline {
            timeout_at(deadline, stream.next()).await
        } else {
            Ok(stream.next().await)
        };
        match event {
            Ok(Some(AgentStreamEvent::Task(event))) => {
                if task_events.len() < 4096 {
                    task_events.push(event);
                } else {
                    events_complete = false;
                }
            }
            Ok(Some(AgentStreamEvent::Chunk(StreamChunk::Content { text }))) => {
                partial_content_seen = true;
                partial_content.push_str(&text);
            }
            Ok(Some(AgentStreamEvent::Chunk(StreamChunk::Error { message }))) => {
                return TurnOperation {
                    task_events,
                    events_complete,
                    response_content: partial_content,
                    response_metadata: None,
                    response_present: partial_content_seen,
                    runtime_error: Some(message),
                };
            }
            Ok(Some(AgentStreamEvent::Chunk(_))) => {}
            Ok(Some(AgentStreamEvent::Final(response))) => {
                final_response = Some(response);
                break;
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

    if let Some(response) = final_response {
        return TurnOperation {
            task_events,
            events_complete,
            response_content: response.content,
            response_metadata: response.metadata,
            response_present: true,
            runtime_error,
        };
    }
    if runtime_error.is_none() {
        runtime_error = Some("stream ended before Final".to_string());
    }
    TurnOperation {
        task_events,
        events_complete,
        response_content: partial_content,
        response_metadata: None,
        response_present: partial_content_seen,
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
        let actual = std::mem::take(&mut detail.actual);
        let expected = std::mem::take(&mut detail.expected);
        // Task paths may contain domain IDs as map keys; recursive value-only redaction cannot safely export those subtrees.
        if detail.assertion.starts_with("task.")
            && (actual.is_object() || actual.is_array() || expected.is_object())
        {
            detail.actual = json!({"redacted":true});
            detail.expected = json!({"redacted":true});
        } else {
            detail.actual = redact_value(actual, true, 0);
            detail.expected = redact_value(expected, true, 0);
        }
    }
}

fn error_result(
    scenario: &Scenario,
    error: EvalError,
    category: FailureCategory,
    redact: bool,
) -> ScenarioResult {
    ScenarioResult {
        id: scenario.id.clone(),
        name: scenario.name.clone(),
        tags: scenario.tags.clone(),
        language: scenario.language.clone(),
        status: ScenarioStatus::Error {
            message: redact_text(&error.to_string(), redact, 0).value,
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
    let observations: Vec<_> = results
        .iter()
        .flat_map(|scenario| &scenario.attempts)
        .flat_map(|attempt| &attempt.turns)
        .flat_map(|turn| {
            turn.evidence.observability.iter().chain(
                turn.interactions
                    .iter()
                    .filter_map(|interaction| interaction.evidence.observability.as_ref()),
            )
        })
        .collect();
    if observations.is_empty() {
        return None;
    }
    let mut events = std::collections::BTreeMap::new();
    let mut dropped = 0;
    for observation in observations {
        dropped += observation
            .report
            .as_ref()
            .map_or(0, |report| report.dropped_events);
        for event in &observation.events {
            if let Some(old) = events.get(&event.span_id) {
                if serde_json::to_value(old).ok() != serde_json::to_value(event).ok() {
                    dropped += 1;
                }
            } else {
                events.insert(event.span_id.clone(), event.clone());
            }
        }
    }
    Some(ai_agents_observability::report::generate_report(
        &events.into_values().collect::<Vec<_>>(),
        Vec::new(),
        dropped,
    ))
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
mod routing_tests {
    use super::*;

    #[tokio::test]
    async fn hierarchy_web_extraction_uses_fixture_alias_and_cache_through_runner() {
        let directory = std::env::temp_dir().join(format!(
            "hierarchical-web-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("agent.yaml"),
            r#"
name: routed-web
system_prompt: Fetch information.
llms:
  main: {provider: openai, model: mock-main}
  extraction: {provider: openai, model: mock-extraction}
llm:
  default: main
  router:
    web:
      extract: extraction
tools: [web_fetch]
"#,
        )
        .unwrap();
        std::fs::write(directory.join("suite.yaml"),r#"
name: Hierarchical Web Extraction
agent: agent.yaml
settings:
  retries: 0
fixtures:
  llm:
    mode: mock
    responses_by_alias:
      main:
        - '{"tool":"web_fetch","arguments":{"url":"https://fixture.example/data","prompt":"Extract."}}'
        - '{"tool":"web_fetch","arguments":{"url":"https://fixture.example/data","prompt":"Extract."}}'
        - DONE
      extraction:
        - EXTRACTED
        - UNEXPECTED_EXTRA_CALL
  web_fetch_transport:
    routes:
      - url: https://fixture.example/data
        status: 200
        body: RAW
scenarios:
  - id: extraction-and-cache
    turns:
      - input: Fetch twice.
        assert:
          all:
            - response_contains: DONE
            - tool_called:
                id: web_fetch
                count_gte: 2
                executed: true
                success: true
                result_path: {path: content, eq: EXTRACTED}
            - tool_called:
                id: web_fetch
                executed: true
                success: true
                result_path: {path: from_cache, eq: true}
"#).unwrap();
        let result =
            EvalRunner::from_file(directory.join("suite.yaml"), EvalRunnerOptions::default())
                .unwrap()
                .run()
                .await
                .unwrap();
        assert_eq!(result.passed, 1, "{result:?}");
        std::fs::remove_dir_all(directory).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    tokio::task_local! {
        pub(super) static CONSTRUCTION_GATE: Arc<tokio::sync::Semaphore>;
    }

    fn attempt_workspace(id: &str) -> PathBuf {
        std::env::temp_dir().join(format!("ai-agents-eval-{id}"))
    }

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
        let workspace = attempt_workspace("attempt-a");
        let context = AttemptFixtureContext {
            isolation_id: "attempt-a".to_string(),
            workspace: workspace.clone(),
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
            file_spec.storage.get_path().map(PathBuf::from),
            Some(workspace.join("parent-storage"))
        );
        let shared = file_spec
            .spawner
            .as_ref()
            .unwrap()
            .shared_storage
            .as_ref()
            .unwrap();
        assert_eq!(
            shared.get_path().map(PathBuf::from),
            Some(workspace.join("spawner-shared-storage.db"))
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
            workspace: attempt_workspace("attempt-b"),
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
        let first_workspace = attempt_workspace("attempt-a");
        let second_workspace = attempt_workspace("attempt-b");
        let first_context = AttemptFixtureContext {
            isolation_id: "attempt-a".to_string(),
            workspace: first_workspace.clone(),
            mock_server_base_url: None,
        };
        let second_context = AttemptFixtureContext {
            isolation_id: "attempt-b".to_string(),
            workspace: second_workspace.clone(),
            mock_server_base_url: None,
        };

        let mut first = source.clone();
        apply_workspace_policy(&mut first, Some(&config), &first_context).unwrap();
        let mut second = source.clone();
        apply_workspace_policy(&mut second, Some(&config), &second_context).unwrap();

        assert!(first.tool_security.fail_closed);
        assert_eq!(
            first.tool_security.tools["file_read"].read_paths,
            vec![
                "./source".to_string(),
                first_workspace.display().to_string()
            ]
        );
        assert_eq!(
            first.tool_security.tools["file_write"].write_paths,
            vec![
                "./output".to_string(),
                first_workspace.display().to_string()
            ]
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
            vec![
                "./source".to_string(),
                second_workspace.display().to_string()
            ]
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
            workspace: attempt_workspace("attempt-a"),
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
            .build_agent(BuildAgentParams {
                agent_path: &dir.join("agent.yaml"),
                base_dir: &dir,
                attempt_context: &attempt_context,
                tool_log: tool_log.clone(),
                approval_log: approval_log.clone(),
                llm_log: llm_log.clone(),
                approval_handler: None,
                budget: None,
            })
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
            .build_agent(BuildAgentParams {
                agent_path: &dir.join("agent.yaml"),
                base_dir: &dir,
                attempt_context: &attempt_context,
                tool_log,
                approval_log,
                llm_log,
                approval_handler: None,
                budget: None,
            })
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
    async fn streaming_final_response_is_authoritative() {
        let response = ai_agents_runtime::AgentResponse::new("authoritative")
            .with_metadata("source", json!("final"));
        let events = stream::iter(vec![
            AgentStreamEvent::Chunk(StreamChunk::Content {
                text: "partial content".to_string(),
            }),
            AgentStreamEvent::Chunk(StreamChunk::Done {}),
            AgentStreamEvent::Final(response),
        ]);
        let operation =
            consume_stream_response(events, TokioInstant::now() + Duration::from_secs(1), 1_000)
                .await;

        assert_eq!(operation.response_content, "authoritative");
        assert_eq!(
            operation
                .response_metadata
                .as_ref()
                .and_then(|metadata| metadata.get("source")),
            Some(&json!("final"))
        );
        assert!(operation.response_present);
        assert_eq!(operation.runtime_error, None);
    }

    #[tokio::test]
    async fn streaming_final_response_is_present_without_content_chunks() {
        let events = stream::iter(vec![AgentStreamEvent::Final(
            ai_agents_runtime::AgentResponse::new(""),
        )]);
        let operation =
            consume_stream_response(events, TokioInstant::now() + Duration::from_secs(1), 1_000)
                .await;

        assert_eq!(operation.response_content, "");
        assert!(operation.response_present);
        assert_eq!(operation.runtime_error, None);
    }

    #[tokio::test]
    async fn streaming_error_is_terminal_and_retains_prior_partial_content() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let polled = Arc::new(AtomicUsize::new(0));
        let polled_events = Arc::clone(&polled);
        let events = stream::iter(vec![
            AgentStreamEvent::Chunk(StreamChunk::Content {
                text: "before ".to_string(),
            }),
            AgentStreamEvent::Chunk(StreamChunk::Error {
                message: "stream failed".to_string(),
            }),
            AgentStreamEvent::Chunk(StreamChunk::Content {
                text: "after".to_string(),
            }),
            AgentStreamEvent::Chunk(StreamChunk::Done {}),
        ])
        .inspect(move |_| {
            polled_events.fetch_add(1, Ordering::Relaxed);
        });
        let operation =
            consume_stream_response(events, TokioInstant::now() + Duration::from_secs(1), 1_000)
                .await;

        assert_eq!(operation.response_content, "before ");
        assert!(operation.response_present);
        assert_eq!(operation.response_metadata, None);
        assert_eq!(operation.runtime_error.as_deref(), Some("stream failed"));
        assert_eq!(polled.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn streaming_eof_without_final_is_incomplete() {
        let events = stream::iter(vec![AgentStreamEvent::Chunk(StreamChunk::Content {
            text: "partial".to_string(),
        })]);
        let operation =
            consume_stream_response(events, TokioInstant::now() + Duration::from_secs(1), 1_000)
                .await;

        assert_eq!(operation.response_content, "partial");
        assert!(operation.response_present);
        assert_eq!(operation.response_metadata, None);
        assert_eq!(
            operation.runtime_error.as_deref(),
            Some("stream ended before Final")
        );
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

    // Creating the heap-owned construction future must defer parsing and retain fail-closed preflight.
    #[tokio::test]
    async fn construction_is_lazy_and_rejects_invalid_hierarchy_before_calls() {
        let workspace = AttemptWorkspace::create(None).unwrap();
        let path = workspace.workspace.join("agent.yaml");
        let runner = EvalRunner {
            suite_path: workspace.workspace.join("suite.yaml"),
            host: None,
                        suite: parse_eval_suite_yaml("name: construction\nagent: agent.yaml\nfixtures:\n  llm:\n    mode: mock\nscenarios:\n  - id: check\n    turns:\n      - input: hello\n").unwrap(),
            options: EvalRunnerOptions::default(),
        };
        let llm_log = RecordingLlmLog::default();
        let create_build = || {
            runner.build_agent(BuildAgentParams {
                agent_path: &path,
                base_dir: &workspace.workspace,
                attempt_context: &workspace,
                tool_log: RecordingToolLog::new(),
                approval_log: RecordingApprovalLog::default(),
                llm_log: llm_log.clone(),
                approval_handler: None,
                budget: None,
            })
        };
        let unpolled = create_build();
        assert_eq!(Arc::strong_count(&llm_log.records), 2);
        drop(unpolled);
        assert_eq!(Arc::strong_count(&llm_log.records), 1);
        let build = create_build();
        assert!(!path.exists());
        std::fs::write(&path, "name: invalid\nsystem_prompt: Test\nllms:\n  main: {provider: openai, model: mock}\nllm:\n  default: main\n  router:\n    state:\n      transition: missing\n").unwrap();
        let error = match build.await {
            Ok(_) => panic!("Missing hierarchy alias must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("missing"), "{error}");
        assert_eq!(llm_log.len(), 0);
    }

    // A polled construction must release its partially built hooks without taking ownership of the caller's workspace.
    #[tokio::test]
    async fn dropping_pending_construction_releases_builder_hooks() {
        let workspace = AttemptWorkspace::create(None).unwrap();
        write_test_agent(&workspace.workspace);
        let path = workspace.workspace.join("agent.yaml");
        let workspace_path = workspace.workspace.clone();
        let runner = EvalRunner {
            suite_path: workspace.workspace.join("suite.yaml"),
            host: None,
                        suite: parse_eval_suite_yaml("name: pending-build\nagent: agent.yaml\nfixtures:\n  llm:\n    mode: mock\n    responses: [ok]\nscenarios:\n  - id: check\n    turns:\n      - input: hello\n").unwrap(),
            options: EvalRunnerOptions::default(),
        };
        let llm_log = RecordingLlmLog::default();
        let approval_log = RecordingApprovalLog::default();
        let build = runner.build_agent(BuildAgentParams {
            agent_path: &path,
            base_dir: &workspace.workspace,
            attempt_context: &workspace,
            tool_log: RecordingToolLog::new(),
            approval_log: approval_log.clone(),
            llm_log: llm_log.clone(),
            approval_handler: None,
            budget: None,
        });
        let gate = Arc::new(tokio::sync::Semaphore::new(0));
        let mut pending = Box::pin(CONSTRUCTION_GATE.scope(gate.clone(), build));
        assert!(futures::poll!(pending.as_mut()).is_pending());
        // The pending builder and actual provider recorder both retain the log until construction is dropped.
        assert_eq!(Arc::strong_count(&llm_log.records), 3);
        assert_eq!(Arc::strong_count(&approval_log.records), 2);
        assert_eq!(
            Arc::strong_count(&gate),
            3,
            "The polled construction must hold its gate"
        );
        assert_eq!(llm_log.len(), 0);
        drop(pending);
        assert_eq!(Arc::strong_count(&llm_log.records), 1);
        assert_eq!(Arc::strong_count(&approval_log.records), 1);
        assert_eq!(Arc::strong_count(&gate), 1);
        assert!(workspace_path.exists());
        drop(workspace);
        assert!(!workspace_path.exists());
    }

    // Cancellation after boxed construction must release the caller-owned workspace and pending model future.
    #[tokio::test]
    async fn dropping_constructed_turn_releases_attempt_workspace() {
        let source = AttemptWorkspace::create(None).unwrap();
        write_test_agent(&source.workspace);
        let agent_path = source.workspace.join("agent.yaml");
        let runner = EvalRunner {
            suite_path: source.workspace.join("suite.yaml"),
            host: None,
                        suite: parse_eval_suite_yaml("name: drop\nagent: agent.yaml\nfixtures:\n  llm:\n    mode: mock\n    responses: [ok]\n    delays_by_alias: {default: 60000}\nscenarios:\n  - id: pending\n    turns:\n      - input: hello\n").unwrap(),
            options: EvalRunnerOptions::default(),
        };
        let workspace = AttemptWorkspace::create(None).unwrap();
        let workspace_path = workspace.workspace.clone();
        let base_dir = source.workspace.clone();
        let llm_log = RecordingLlmLog::default();
        let future_log = llm_log.clone();
        let mut operation = Box::pin(async move {
            let agent = runner
                .build_agent(BuildAgentParams {
                    agent_path: &agent_path,
                    base_dir: &base_dir,
                    attempt_context: &workspace,
                    tool_log: RecordingToolLog::new(),
                    approval_log: RecordingApprovalLog::default(),
                    llm_log: future_log,
                    approval_handler: None,
                    budget: None,
                })
                .await
                .unwrap();
            agent.chat("hello").await.unwrap();
            drop(agent);
            drop(workspace);
        });
        assert!(futures::poll!(operation.as_mut()).is_pending());
        assert_eq!(
            llm_log.len(),
            1,
            "The turn must reach the pending fixture provider"
        );
        assert!(workspace_path.exists());
        drop(operation);
        assert!(!workspace_path.exists());
        assert_eq!(llm_log.len(), 1);
    }

    // Parallel scenario futures must preserve declaration order, full rebuilds, and stream finalization.
    #[tokio::test]
    async fn parallel_scenarios_preserve_order_and_full_reset_fixtures() {
        let source = AttemptWorkspace::create(None).unwrap();
        write_test_agent(&source.workspace);
        let runner = EvalRunner {
            suite_path: source.workspace.join("suite.yaml"),
            host: None,
            suite: parse_eval_suite_yaml(
                r#"
name: parallel-reset
agent: agent.yaml
settings:
  parallel: true
  max_concurrent: 2
  retries: 0
fixtures:
  llm:
    mode: mock
    responses: [ok]
scenarios:
  - id: rebuild
    steps:
      - !run
        turns:
          - input: before
            assert: {response_contains: ok}
      - !reset_agent
        profile: full_runtime
      - !run
        turns:
          - input: after
            assert: {response_contains: ok}
  - id: stream
    turns:
      - input: streaming
        stream: true
        assert: {response_contains: ok}
"#,
            )
            .unwrap(),
            options: EvalRunnerOptions::default(),
        };
        let result = runner.run().await.unwrap();
        assert_eq!(result.passed, 2, "{result:?}");
        assert_eq!(result.scenarios[0].id, "rebuild");
        assert_eq!(result.scenarios[1].id, "stream");
        let turns = &result.scenarios[0].attempts[0].turns;
        assert_eq!(turns.len(), 2);
        assert!(
            turns
                .iter()
                .all(|turn| turn.response_present && turn.evidence.llm_requests.len() == 1)
        );
        let messages = &turns[1].evidence.llm_requests[0].messages;
        assert!(!messages.iter().any(|message| message.content == "before"));
        assert!(messages.iter().any(|message| message.content == "after"));
        assert!(result.scenarios[1].attempts[0].turns[0].response_present);
    }

    #[test]
    fn runtime_error_expectations_retain_turns_and_control_retries() {
        std::thread::Builder::new()
            .name("eval-runtime-error-test".to_string())
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
    fn observability_evidence_is_scoped_to_each_turn() {
        std::thread::Builder::new()
            .name("eval-turn-observability-test".to_string())
            .spawn(|| {
                let runtime = tokio::runtime::Runtime::new().unwrap();
                runtime.block_on(async {
                    let dir = std::env::temp_dir().join(format!(
                        "ai_agents_eval_turn_observability_test_{}",
                        uuid::Uuid::new_v4()
                    ));
                    std::fs::create_dir_all(&dir).unwrap();
                    write_test_agent(&dir);
                    let result = run_test_suite(
                        &dir,
                        "turn-observability.yaml",
                        r#"
name: Turn Observability
agent: agent.yaml
settings:
  redact_outputs: false
observability:
  enabled: true
  aggregation:
    dimensions: [purpose]
  export:
    write_report: false
fixtures:
  llm:
    mode: mock
    responses: [first answer, second answer]
scenarios:
  - id: separate-turns
    turns:
      - input: First
      - input: Second
"#,
                    )
                    .await;

                    assert_eq!(result.passed, 1);
                    let turns = &result.scenarios[0].attempts[0].turns;
                    assert_eq!(turns.len(), 2);
                    let first = turns[0].evidence.observability.as_ref().unwrap();
                    let second = turns[1].evidence.observability.as_ref().unwrap();
                    let first_report = first.report.as_ref().unwrap();
                    let second_report = second.report.as_ref().unwrap();
                    assert!(first_report.summary.total_events > 0);
                    assert_eq!(
                        second_report.summary.total_events,
                        first_report.summary.total_events
                    );
                    assert_eq!(
                        first_report.summary.total_events,
                        first.span_ids.len() as u64
                    );
                    assert_eq!(
                        second_report.summary.total_events,
                        second.span_ids.len() as u64
                    );
                    assert_ne!(first.trace_id, second.trace_id);
                    let first_spans: HashSet<_> = first.span_ids.iter().collect();
                    assert!(
                        second
                            .span_ids
                            .iter()
                            .all(|span_id| !first_spans.contains(span_id))
                    );
                    let _ = std::fs::remove_dir_all(dir);
                });
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn runner_default_judge_is_strict_and_redacts_failures() {
        std::thread::Builder::new()
            .name("eval-judge-contract-test".to_string())
            .spawn(|| {
                let runtime = tokio::runtime::Runtime::new().unwrap();
                runtime.block_on(async {
                    let dir = std::env::temp_dir().join(format!(
                        "ai_agents_eval_judge_contract_test_{}",
                        uuid::Uuid::new_v4()
                    ));
                    std::fs::create_dir_all(&dir).unwrap();
                    write_test_agent(&dir);
                    let strict_json = r#"{"criteria_scores":[{"name":"relevance","score":0.9,"explanation":"ok"}],"overall_score":0.9,"overall_feedback":"ok","passed":false}"#;
                    let success_yaml = format!(
                        r#"
name: Strict Judge Success
agent: agent.yaml
fixtures:
  llm:
    mode: mock
    responses:
      - "Agent answer"
      - '{strict_json}'
scenarios:
  - id: strict-success
    turns:
      - input: Hello
        assert:
          judge:
            criteria: ["Relevant"]
"#
                    );
                    let success =
                        run_test_suite(&dir, "strict-success.yaml", &success_yaml).await;
                    assert_eq!(success.passed, 1);

                    let fenced_yaml = format!(
                        r#"
name: Strict Judge Failure
agent: agent.yaml
settings:
  redact_outputs: true
fixtures:
  llm:
    mode: mock
    responses:
      - "Agent answer"
      - |
        ```json
        {strict_json}
        ```
scenarios:
  - id: strict-failure
    turns:
      - input: Hello
        assert:
          judge:
            criteria: ["Relevant"]
"#
                    );
                    let failure =
                        run_test_suite(&dir, "strict-failure.yaml", &fenced_yaml).await;
                    assert_eq!(failure.failed, 1);
                    assert_eq!(
                        failure.scenarios[0].failure_category,
                        Some(FailureCategory::JudgeError)
                    );
                    let turn = &failure.scenarios[0].attempts[0].turns[0];
                    assert_eq!(turn.input.value, "[redacted]");
                    assert_eq!(turn.response.value, "[redacted]");
                    let serialized = serde_json::to_string(&failure).unwrap();
                    assert!(!serialized.contains("Agent answer"));
                    assert!(!serialized.contains("criteria_scores"));
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

use std::collections::HashMap;
use std::path::PathBuf;

use ai_agents_observability::ObservabilityConfig;
use ai_agents_observability::ObservabilityReport;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::assertion::{Assertion, AssertionResultDetail};
use crate::evidence::TurnEvidence;
use crate::fixtures::FixturesConfig;
use crate::redaction::RedactedString;
use crate::reset::ResetOptions;
use crate::{EvalError, Result};

/// Top-level evaluation suite loaded from YAML or JSONL.
#[derive(Debug, Clone, Deserialize)]
pub struct EvalSuite {
    /// Human-readable name or criterion name.
    pub name: String,
    /// Agent YAML path used for this run.
    #[serde(default)]
    pub agent: Option<PathBuf>,
    /// Execution settings for the suite.
    #[serde(default)]
    pub settings: EvalSettings,
    /// Observability assertion, setting, or report value.
    #[serde(default)]
    pub observability: Option<ObservabilityConfig>,
    /// Fixtures applied while building and running agents.
    #[serde(default)]
    pub fixtures: FixturesConfig,
    /// Scenario test cases in this suite.
    #[serde(default)]
    pub scenarios: Vec<Scenario>,
}

impl EvalSuite {
    pub fn validate(&self, cli_agent: Option<&PathBuf>) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(EvalError::Config(
                "eval suite name must not be empty".into(),
            ));
        }
        if cli_agent.is_none() && self.agent.is_none() {
            return Err(EvalError::Config(
                "agent path is required in suite or CLI".into(),
            ));
        }
        if self.settings.max_concurrent == 0 {
            return Err(EvalError::Config(
                "settings.max_concurrent must be greater than zero".into(),
            ));
        }
        if self.settings.timeout_per_turn_ms == 0 {
            return Err(EvalError::Config(
                "settings.timeout_per_turn_ms must be greater than zero".into(),
            ));
        }
        if matches!(
            self.settings.isolation,
            IsolationMode::Suite | IsolationMode::None
        ) {
            return Err(EvalError::Config(
                "settings.isolation currently supports scenario or turn".into(),
            ));
        }
        if self.settings.parallel && self.settings.isolation != IsolationMode::Scenario {
            return Err(EvalError::Config(
                "settings.parallel currently requires isolation: scenario".into(),
            ));
        }
        if self.settings.parallel
            && self
                .scenarios
                .iter()
                .any(|scenario| !scenario.env.is_empty())
        {
            return Err(EvalError::Config(
                "scenario.env cannot be used with parallel execution".into(),
            ));
        }
        if self
            .scenarios
            .iter()
            .any(|scenario| scenario.budget.max_cost_usd.is_some())
        {
            let cost = self
                .observability
                .as_ref()
                .map(|observability| &observability.cost)
                .ok_or_else(|| {
                    EvalError::Config(
                        "budget.max_cost_usd requires suite observability.cost pricing".into(),
                    )
                })?;
            if !cost.enabled {
                return Err(EvalError::Config(
                    "budget.max_cost_usd requires observability.cost.enabled: true".into(),
                ));
            }
            if cost.pricing.is_empty() && cost.pricing_file.is_none() {
                return Err(EvalError::Config(
                    "budget.max_cost_usd requires observability.cost.pricing or pricing_file"
                        .into(),
                ));
            }
        }
        let mut ids = std::collections::HashSet::new();
        for scenario in &self.scenarios {
            if scenario.id.trim().is_empty() {
                return Err(EvalError::Config("scenario id must not be empty".into()));
            }
            if !ids.insert(scenario.id.clone()) {
                return Err(EvalError::Config(format!(
                    "duplicate scenario id: {}",
                    scenario.id
                )));
            }
            if !scenario.skip.is_skipped() && scenario.turns.is_empty() && scenario.steps.is_empty()
            {
                return Err(EvalError::Config(format!(
                    "scenario '{}' must define turns or steps",
                    scenario.id
                )));
            }
            scenario.budget.validate(&scenario.id)?;
        }
        Ok(())
    }
}

/// Execution policy for an evaluation suite.
#[derive(Debug, Clone, Deserialize)]
pub struct EvalSettings {
    /// Optional temperature override for eval LLMs.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Optional provider seed stored in LLM extra config.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Default timeout for one turn in milliseconds.
    #[serde(default = "default_turn_timeout")]
    pub timeout_per_turn_ms: u64,
    /// Timeout for one scenario attempt in milliseconds.
    #[serde(default)]
    pub timeout_per_scenario_ms: Option<u64>,
    /// Optional retry count or suite retry count.
    #[serde(default)]
    pub retries: u32,
    /// Delay between retry attempts in milliseconds.
    #[serde(default = "default_retry_delay")]
    pub retry_delay_ms: u64,
    /// Runtime isolation mode for scenarios or turns.
    #[serde(default)]
    pub isolation: IsolationMode,
    /// Optional scenario concurrency override.
    #[serde(default)]
    pub parallel: bool,
    /// Maximum concurrently running scenarios.
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,
    /// Stop after the first failed or errored scenario.
    #[serde(default)]
    pub fail_fast: bool,
    /// Whether output artifacts should redact sensitive strings.
    #[serde(default = "default_true")]
    pub redact_outputs: bool,
}

impl Default for EvalSettings {
    fn default() -> Self {
        Self {
            temperature: None,
            seed: None,
            timeout_per_turn_ms: default_turn_timeout(),
            timeout_per_scenario_ms: None,
            retries: 0,
            retry_delay_ms: default_retry_delay(),
            isolation: IsolationMode::Scenario,
            parallel: false,
            max_concurrent: default_max_concurrent(),
            fail_fast: false,
            redact_outputs: true,
        }
    }
}

/// Runtime isolation mode requested by a suite.
#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IsolationMode {
    Turn,
    #[default]
    Scenario,
    Suite,
    None,
}

/// One test case inside an evaluation suite.
#[derive(Debug, Clone, Deserialize)]
pub struct Scenario {
    /// Stable identifier for this item.
    pub id: String,
    /// Human-readable name or criterion name.
    #[serde(default)]
    pub name: Option<String>,
    /// Tags used by filters and grouped metrics.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Optional language label for filtering, metrics, and judge context.
    #[serde(default)]
    pub language: Option<String>,
    /// Actor ID used for this scenario, turn, or assertion.
    #[serde(default)]
    pub actor: Option<String>,
    /// Runtime or fixture context value.
    #[serde(default)]
    pub context: Value,
    /// env value for Scenario.
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// skip value for Scenario.
    #[serde(default)]
    pub skip: SkipConfig,
    /// Protective limits shared by all attempts, turns, resets, and LLM aliases.
    #[serde(default)]
    pub budget: ScenarioBudget,
    /// Turns executed by this scenario or step.
    #[serde(default)]
    pub turns: Vec<Turn>,
    /// Advanced steps executed after direct turns.
    #[serde(default)]
    pub steps: Vec<ScenarioStep>,
}

/// Hard provider-usage limits for one scenario, shared across retries and resets.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ScenarioBudget {
    /// Maximum provider calls that may start for this scenario.
    #[serde(default)]
    pub max_llm_calls: Option<u64>,
    /// Maximum provider-reported or conservatively estimated tokens.
    #[serde(default)]
    pub max_total_tokens: Option<u64>,
    /// Maximum estimated provider cost in US dollars.
    #[serde(default)]
    pub max_cost_usd: Option<f64>,
}

impl ScenarioBudget {
    pub(crate) fn is_configured(&self) -> bool {
        self.max_llm_calls.is_some()
            || self.max_total_tokens.is_some()
            || self.max_cost_usd.is_some()
    }

    fn validate(&self, scenario_id: &str) -> Result<()> {
        if self.max_llm_calls == Some(0) {
            return Err(EvalError::Config(format!(
                "scenario '{scenario_id}' budget.max_llm_calls must be greater than zero"
            )));
        }
        if self.max_total_tokens == Some(0) {
            return Err(EvalError::Config(format!(
                "scenario '{scenario_id}' budget.max_total_tokens must be greater than zero"
            )));
        }
        if let Some(max_cost_usd) = self.max_cost_usd
            && (!max_cost_usd.is_finite() || max_cost_usd <= 0.0)
        {
            return Err(EvalError::Config(format!(
                "scenario '{scenario_id}' budget.max_cost_usd must be finite and greater than zero"
            )));
        }
        Ok(())
    }
}

/// One user input and assertion block inside a scenario.
#[derive(Debug, Clone)]
pub struct Turn {
    /// User input sent to the runtime.
    pub input: String,
    /// Actor ID used for this scenario, turn, or assertion.
    pub actor: Option<String>,
    /// Runtime or fixture context value.
    pub context: Value,
    /// Whether to use streaming chat for this turn.
    pub stream: Option<bool>,
    /// Optional timeout override for this turn.
    pub timeout_ms: Option<u64>,
    /// Assertions evaluated after this turn.
    pub assertions: Option<Assertion>,
}

const EXPECT_ERROR_CONTEXT_KEY: &str = "__ai_agents_eval_expect_error";

#[derive(Deserialize)]
struct TurnDefinition {
    input: String,
    #[serde(default)]
    actor: Option<String>,
    #[serde(default)]
    context: Value,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    timeout_ms: Option<u64>,
    #[serde(default)]
    expect_error: Option<ExpectedError>,
    #[serde(default, rename = "assert")]
    assertions: Option<Assertion>,
}

impl<'de> Deserialize<'de> for Turn {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let definition = TurnDefinition::deserialize(deserializer)?;
        let mut context = definition.context;
        if let Some(expect_error) = definition.expect_error {
            let object = match &mut context {
                Value::Object(object) => object,
                _ => {
                    context = Value::Object(serde_json::Map::new());
                    context
                        .as_object_mut()
                        .expect("context was replaced with an object")
                }
            };
            object.insert(
                EXPECT_ERROR_CONTEXT_KEY.to_string(),
                serde_json::to_value(expect_error).map_err(serde::de::Error::custom)?,
            );
        }
        Ok(Self {
            input: definition.input,
            actor: definition.actor,
            context,
            stream: definition.stream,
            timeout_ms: definition.timeout_ms,
            assertions: definition.assertions,
        })
    }
}

pub(crate) fn turn_expected_error(turn: &Turn) -> Option<ExpectedError> {
    turn.context
        .get(EXPECT_ERROR_CONTEXT_KEY)
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
}

pub(crate) fn turn_runtime_context(turn: &Turn) -> Value {
    let mut context = turn.context.clone();
    if let Value::Object(object) = &mut context {
        object.remove(EXPECT_ERROR_CONTEXT_KEY);
    }
    context
}

/// Runtime error expectation accepting one substring or a list of alternatives.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ExpectedError {
    One(String),
    Any(Vec<String>),
}

impl ExpectedError {
    pub fn matches(&self, error: &str) -> bool {
        self.items().iter().any(|expected| error.contains(expected))
    }

    pub fn items(&self) -> Vec<&str> {
        match self {
            Self::One(value) => vec![value.as_str()],
            Self::Any(values) => values.iter().map(String::as_str).collect(),
        }
    }
}

/// Boolean or reason-string skip configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SkipConfig {
    Bool(bool),
    Reason(String),
}

impl Default for SkipConfig {
    fn default() -> Self {
        Self::Bool(false)
    }
}

impl SkipConfig {
    pub fn is_skipped(&self) -> bool {
        match self {
            Self::Bool(value) => *value,
            Self::Reason(_) => true,
        }
    }

    pub fn reason(&self) -> Option<String> {
        match self {
            Self::Bool(_) => None,
            Self::Reason(reason) => Some(reason.clone()),
        }
    }
}

/// Advanced scenario action used outside direct turn lists.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScenarioStep {
    Run(RunStep),
    ResetAgent(ResetStepConfig),
    SaveSession(String),
    LoadSession(String),
    SetContext { values: Value },
    SetActor { actor: String },
    CleanupExpired,
}

/// Advanced step that runs turns and can save a session.
#[derive(Debug, Clone, Deserialize)]
pub struct RunStep {
    /// Turns executed by this scenario or step.
    #[serde(default)]
    pub turns: Vec<Turn>,
    /// Optional session name saved after a run step.
    #[serde(default)]
    pub save_session: Option<String>,
}

/// Boolean or object form for reset-agent steps.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ResetStepConfig {
    Bool(bool),
    Options(ResetOptions),
}

/// Top-level result returned by an eval suite run.
#[derive(Debug, Clone, Serialize)]
pub struct EvalResult {
    /// Machine-readable output schema version.
    pub schema_version: u32,
    /// Parsed and validated suite.
    pub suite: String,
    /// Agent YAML path used for this run.
    pub agent: String,
    /// Total count for this result or group.
    pub total: usize,
    /// Passed count or boolean result.
    pub passed: usize,
    /// Failed or errored count for this result or group.
    pub failed: usize,
    /// Skipped count for this result or group.
    pub skipped: usize,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Scenario test cases in this suite.
    pub scenarios: Vec<ScenarioResult>,
    /// metrics value for EvalResult.
    pub metrics: crate::metrics::EvalMetrics,
    /// Observability assertion, setting, or report value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observability: Option<ObservabilityReport>,
}

/// Result for one evaluated scenario.
#[derive(Debug, Clone, Serialize)]
pub struct ScenarioResult {
    /// Stable identifier for this item.
    pub id: String,
    /// Human-readable name or criterion name.
    pub name: Option<String>,
    /// Tags used by filters and grouped metrics.
    pub tags: Vec<String>,
    /// Optional language label for filtering, metrics, and judge context.
    pub language: Option<String>,
    /// Final or normalized status value.
    pub status: ScenarioStatus,
    /// High-level failure category for metrics.
    pub failure_category: Option<FailureCategory>,
    /// Number of scenarios that passed after retry.
    pub flaky: bool,
    /// Attempt results in execution order.
    pub attempts: Vec<AttemptResult>,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Number of retries consumed by this scenario.
    pub retries_used: u32,
}

/// Result for one scenario attempt.
#[derive(Debug, Clone, Serialize)]
pub struct AttemptResult {
    /// Zero-based attempt index.
    pub attempt: u32,
    /// Turns executed by this scenario or step.
    pub turns: Vec<TurnResult>,
    /// Final or normalized status value.
    pub status: ScenarioStatus,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

/// Final status for a scenario result.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScenarioStatus {
    Passed,
    Failed { reason: String },
    Skipped { reason: Option<String> },
    Error { message: String },
}

impl ScenarioStatus {
    pub fn is_passed(&self) -> bool {
        matches!(self, Self::Passed)
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, Self::Failed { .. })
    }

    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }
}

/// High-level failure category used by metrics and reports.
#[derive(Debug, Clone, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FailureCategory {
    ConfigError,
    RuntimeError,
    AssertionFailed,
    JudgeError,
    FlakyPass,
}

/// Result for one evaluated turn.
#[derive(Debug, Clone, Serialize)]
pub struct TurnResult {
    /// Zero-based turn index within the scenario.
    pub index: usize,
    /// User input sent to the runtime.
    pub input: RedactedString,
    /// Assistant response text or redacted output value.
    pub response: RedactedString,
    /// Whether the runtime produced an assistant response.
    pub response_present: bool,
    /// Runtime error emitted while executing this turn.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_error: Option<RedactedString>,
    /// Current or expected state name.
    pub state: Option<String>,
    /// Optional response or tool metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    /// Full assertion-time evidence for this turn.
    #[serde(skip_serializing)]
    pub evidence: TurnEvidence,
    /// Assertion details produced for this turn.
    pub assertion_results: Vec<AssertionResultDetail>,
    /// latency_ms value for TurnResult.
    pub latency_ms: u64,
    /// Optional observability span ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observability_span_id: Option<String>,
}

fn default_turn_timeout() -> u64 {
    30_000
}

fn default_retry_delay() -> u64 {
    1_000
}

fn default_max_concurrent() -> usize {
    4
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn suite() -> EvalSuite {
        EvalSuite {
            name: "suite".to_string(),
            agent: Some(PathBuf::from("agent.yaml")),
            settings: EvalSettings::default(),
            observability: None,
            fixtures: FixturesConfig::default(),
            scenarios: vec![Scenario {
                id: "scenario-1".to_string(),
                name: None,
                tags: vec!["smoke".to_string()],
                language: Some("en".to_string()),
                actor: None,
                context: Value::Null,
                env: HashMap::new(),
                skip: SkipConfig::default(),
                budget: ScenarioBudget::default(),
                turns: vec![Turn {
                    input: "hello".to_string(),
                    actor: None,
                    context: Value::Null,
                    stream: None,
                    timeout_ms: None,
                    assertions: None,
                }],
                steps: Vec::new(),
            }],
        }
    }

    #[test]
    fn validation_accepts_minimal_suite() {
        assert!(suite().validate(None).is_ok());
    }

    #[test]
    fn validation_rejects_duplicate_ids() {
        let mut suite = suite();
        suite.scenarios.push(suite.scenarios[0].clone());
        let error = suite.validate(None).unwrap_err().to_string();
        assert!(error.contains("duplicate scenario id"));
    }

    #[test]
    fn expected_error_accepts_string_and_list() {
        let one: ExpectedError = serde_yaml::from_str("timeout").unwrap();
        let many: ExpectedError = serde_yaml::from_str("[timeout, unavailable]").unwrap();
        assert!(one.matches("turn timeout"));
        assert!(many.matches("service unavailable"));
        assert!(!many.matches("permission denied"));

        let turn: Turn =
            serde_yaml::from_str("input: hello\nexpect_error: [timeout, unavailable]").unwrap();
        assert!(turn_expected_error(&turn).unwrap().matches("turn timeout"));
        assert_eq!(
            turn_runtime_context(&turn),
            Value::Object(serde_json::Map::new())
        );
    }

    #[test]
    fn validation_rejects_parallel_env() {
        let mut suite = suite();
        suite.settings.parallel = true;
        suite.scenarios[0]
            .env
            .insert("TOKEN".to_string(), "secret".to_string());
        let error = suite.validate(None).unwrap_err().to_string();
        assert!(error.contains("scenario.env"));
    }

    #[test]
    fn validation_rejects_invalid_scenario_budgets() {
        let mut suite = suite();
        suite.scenarios[0].budget.max_llm_calls = Some(0);
        let error = suite.validate(None).unwrap_err().to_string();
        assert!(error.contains("budget.max_llm_calls"));

        suite.scenarios[0].budget.max_llm_calls = Some(1);
        suite.scenarios[0].budget.max_total_tokens = Some(0);
        let error = suite.validate(None).unwrap_err().to_string();
        assert!(error.contains("budget.max_total_tokens"));

        suite.scenarios[0].budget.max_total_tokens = Some(1);
        suite.scenarios[0].budget.max_cost_usd = Some(f64::NAN);
        let error = suite.validate(None).unwrap_err().to_string();
        assert!(error.contains("budget.max_cost_usd"));
    }

    #[test]
    fn validation_requires_pricing_for_cost_budgets() {
        let mut suite = suite();
        suite.scenarios[0].budget.max_cost_usd = Some(0.01);
        let error = suite.validate(None).unwrap_err().to_string();
        assert!(error.contains("observability.cost pricing"));

        suite.observability = Some(ObservabilityConfig::default());
        let error = suite.validate(None).unwrap_err().to_string();
        assert!(error.contains("pricing or pricing_file"));

        suite.observability.as_mut().unwrap().cost.pricing_file = Some("pricing.yaml".to_string());
        assert!(suite.validate(None).is_ok());
    }
}

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

#[derive(Debug, Clone, Deserialize)]
pub struct EvalSuite {
    pub name: String,
    #[serde(default)]
    pub agent: Option<PathBuf>,
    #[serde(default)]
    pub settings: EvalSettings,
    #[serde(default)]
    pub observability: Option<ObservabilityConfig>,
    #[serde(default)]
    pub fixtures: FixturesConfig,
    #[serde(default)]
    pub scenarios: Vec<Scenario>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EvalSettings {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default = "default_turn_timeout")]
    pub timeout_per_turn_ms: u64,
    #[serde(default)]
    pub timeout_per_scenario_ms: Option<u64>,
    #[serde(default)]
    pub retries: u32,
    #[serde(default = "default_retry_delay")]
    pub retry_delay_ms: u64,
    #[serde(default)]
    pub isolation: IsolationMode,
    #[serde(default)]
    pub parallel: bool,
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,
    #[serde(default)]
    pub fail_fast: bool,
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

#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IsolationMode {
    Turn,
    #[default]
    Scenario,
    Suite,
    None,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Scenario {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub context: Value,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub skip: SkipConfig,
    #[serde(default)]
    pub turns: Vec<Turn>,
    #[serde(default)]
    pub steps: Vec<ScenarioStep>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Turn {
    pub input: String,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub context: Value,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    #[serde(default, rename = "assert")]
    pub assertions: Option<Assertion>,
}

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

#[derive(Debug, Clone, Deserialize)]
pub struct RunStep {
    #[serde(default)]
    pub turns: Vec<Turn>,
    #[serde(default)]
    pub save_session: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ResetStepConfig {
    Bool(bool),
    Options(ResetOptions),
}

#[derive(Debug, Clone, Serialize)]
pub struct EvalResult {
    pub suite: String,
    pub agent: String,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub duration_ms: u64,
    pub scenarios: Vec<ScenarioResult>,
    pub metrics: crate::metrics::EvalMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observability: Option<ObservabilityReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScenarioResult {
    pub id: String,
    pub name: Option<String>,
    pub tags: Vec<String>,
    pub language: Option<String>,
    pub status: ScenarioStatus,
    pub failure_category: Option<FailureCategory>,
    pub flaky: bool,
    pub attempts: Vec<AttemptResult>,
    pub duration_ms: u64,
    pub retries_used: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct AttemptResult {
    pub attempt: u32,
    pub turns: Vec<TurnResult>,
    pub status: ScenarioStatus,
    pub duration_ms: u64,
}

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

#[derive(Debug, Clone, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FailureCategory {
    ConfigError,
    RuntimeError,
    AssertionFailed,
    JudgeError,
    FlakyPass,
}

#[derive(Debug, Clone, Serialize)]
pub struct TurnResult {
    pub index: usize,
    pub input: RedactedString,
    pub response: RedactedString,
    pub state: Option<String>,
    pub metadata: Option<Value>,
    pub evidence: TurnEvidence,
    pub assertion_results: Vec<AssertionResultDetail>,
    pub latency_ms: u64,
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

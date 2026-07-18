//! Declarative evaluation runner for YAML-defined agents.

pub mod assertion;
mod budget;
pub mod compatibility;
pub mod evidence;
pub mod fixtures;
pub mod judge;
pub mod metrics;
pub mod output;
pub mod redaction;
pub mod reset;
pub mod runner;
pub mod suite;

pub use assertion::{AssertionOutcome, AssertionResultDetail, evaluate_assertion};
pub use evidence::{DisambiguationStatus, ToolExecutionRecord, ToolExecutionSource, TurnEvidence};
pub use fixtures::{FixturesConfig, LlmFixtureMode, RecordingToolLog};
pub use judge::{JudgeConfig, JudgeInput, JudgeResolver, JudgeResult, LLMJudge};
pub use output::write_outputs;
pub use redaction::RedactedString;
pub use reset::{ResetOptions, ResetProfile};
pub use runner::{EvalRunner, EvalRunnerOptions};
pub use suite::{
    EvalResult, EvalSettings, EvalSuite, ScenarioBudget, ScenarioResult, ScenarioStatus,
};

pub type Result<T> = std::result::Result<T, EvalError>;

/// Error type used by the evaluation crate.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("runtime error: {0}")]
    Runtime(String),
    #[error("assertion failed: {0}")]
    Assertion(String),
    #[error("judge error: {0}")]
    Judge(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("yaml error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

impl From<ai_agents_core::AgentError> for EvalError {
    fn from(error: ai_agents_core::AgentError) -> Self {
        Self::Runtime(error.to_string())
    }
}

impl From<ai_agents_core::LLMError> for EvalError {
    fn from(error: ai_agents_core::LLMError) -> Self {
        Self::Runtime(error.to_string())
    }
}

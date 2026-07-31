use std::sync::Arc;

use ai_agents_core::{ChatMessage, LLMProvider};
use ai_agents_llm::LLMRegistry;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{EvalError, Result};

/// Semantic judge assertion declared in an eval suite.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JudgeAssertion {
    /// Optional LLM alias or provider used for judge calls.
    #[serde(default)]
    pub llm: Option<String>,
    /// Minimum overall score required to pass.
    #[serde(default = "default_threshold")]
    pub pass_threshold: f32,
    /// Criteria used by the judge prompt.
    #[serde(default)]
    pub criteria: Vec<JudgeCriterion>,
}

/// Text or weighted object form for judge criteria.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum JudgeCriterion {
    Text(String),
    Object {
        name: String,
        description: String,
        #[serde(default = "default_weight")]
        weight: f32,
    },
}

/// Default behavior for LLM judge evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgeConfig {
    /// Whether this feature is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Optional LLM alias or provider used for judge calls.
    #[serde(default)]
    pub llm: Option<String>,
    /// Criteria used when assertions omit criteria.
    #[serde(default)]
    pub default_criteria: Vec<JudgeCriterion>,
    /// Minimum overall score required to pass.
    #[serde(default = "default_threshold")]
    pub pass_threshold: f32,
    /// Whether judge responses must be strict JSON.
    #[serde(default = "default_true")]
    pub require_json: bool,
}

impl Default for JudgeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            llm: None,
            default_criteria: Vec::new(),
            pass_threshold: default_threshold(),
            require_json: true,
        }
    }
}

/// Parsed JSON result returned by an LLM judge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeResult {
    /// Scores for individual criteria.
    pub criteria_scores: Vec<CriterionScore>,
    /// Aggregated score used for pass or fail.
    pub overall_score: f32,
    /// Brief feedback returned by the judge.
    pub overall_feedback: String,
    /// Passed count or boolean result.
    pub passed: bool,
    /// Optional raw judge response for debugging.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<String>,
}

/// Score for one criterion inside a judge result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriterionScore {
    /// Human-readable name or criterion name.
    pub name: String,
    /// Numeric score assigned by the judge.
    pub score: f32,
    /// Brief explanation for the score.
    #[serde(default)]
    pub explanation: String,
}

/// Context passed to the judge prompt for one evaluation.
pub struct JudgeInput<'a> {
    /// Assistant response text or redacted output value.
    pub response: &'a str,
    /// Optional user input for judge prompt context.
    pub user_input: Option<&'a str>,
    /// Optional scenario ID for judge prompt context.
    pub scenario_id: Option<&'a str>,
    /// Optional language label for filtering, metrics, and judge context.
    pub language: Option<&'a str>,
}

/// Resolves judge LLM aliases from the runtime registry.
pub struct JudgeResolver {
    /// Runtime LLM registry used to resolve aliases.
    registry: Arc<LLMRegistry>,
    /// Configuration used by this component.
    config: JudgeConfig,
}

impl JudgeResolver {
    pub fn new(registry: Arc<LLMRegistry>, config: JudgeConfig) -> Self {
        Self { registry, config }
    }

    pub fn resolve(&self, alias: Option<&str>) -> Result<LLMJudge> {
        let llm = if let Some(alias) = alias {
            self.registry
                .get(alias)
                .map_err(|error| EvalError::Judge(error.to_string()))?
        } else {
            self.registry
                .router()
                .or_else(|_| self.registry.default())
                .map_err(|error| EvalError::Judge(error.to_string()))?
        };
        Ok(LLMJudge::new(llm, self.config.clone()))
    }
}

/// Wrapper that asks an LLM to score semantic response quality.
pub struct LLMJudge {
    /// Optional LLM alias or provider used for judge calls.
    llm: Arc<dyn LLMProvider>,
    /// Configuration used by this component.
    config: JudgeConfig,
}

impl LLMJudge {
    pub fn new(llm: Arc<dyn LLMProvider>, config: JudgeConfig) -> Self {
        Self { llm, config }
    }

    pub async fn evaluate(
        &self,
        response: &str,
        assertion: &JudgeAssertion,
    ) -> Result<JudgeResult> {
        self.evaluate_input(
            JudgeInput {
                response,
                user_input: None,
                scenario_id: None,
                language: None,
            },
            assertion,
        )
        .await
    }

    pub async fn evaluate_input(
        &self,
        input: JudgeInput<'_>,
        assertion: &JudgeAssertion,
    ) -> Result<JudgeResult> {
        let criteria = if assertion.criteria.is_empty() {
            self.config.default_criteria.clone()
        } else {
            assertion.criteria.clone()
        };
        if criteria.is_empty() {
            return Err(EvalError::Judge("judge assertion has no criteria".into()));
        }
        let threshold = assertion.pass_threshold;
        let prompt = build_prompt(input, &criteria, threshold);
        let llm_response = self
            .llm
            .complete(&[ChatMessage::user(&prompt)], None)
            .await
            .map_err(|error| EvalError::Judge(error.to_string()))?;
        let value = extract_json(&llm_response.content)
            .ok_or_else(|| EvalError::Judge("judge did not return JSON".into()))?;
        let mut result: JudgeResult = serde_json::from_value(value)
            .map_err(|error| EvalError::Judge(format!("invalid judge JSON: {}", error)))?;
        result.passed = result.overall_score >= threshold;
        if !self.config.require_json {
            result.raw_response = Some(llm_response.content);
        }
        Ok(result)
    }
}

fn build_prompt(input: JudgeInput<'_>, criteria: &[JudgeCriterion], threshold: f32) -> String {
    let criteria_text = criteria
        .iter()
        .enumerate()
        .map(|(idx, criterion)| match criterion {
            JudgeCriterion::Text(text) => format!("{}. {} (weight 1.0)", idx + 1, text),
            JudgeCriterion::Object {
                name,
                description,
                weight,
            } => {
                format!("{}. {}: {} (weight {})", idx + 1, name, description, weight)
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let user_input = input.user_input.unwrap_or("");
    let scenario_id = input.scenario_id.unwrap_or("");
    let language = input.language.unwrap_or("");
    let response = input.response;
    format!(
        r#"Evaluate the assistant response against the criteria.
Evaluate semantic meaning across languages. Do not require exact wording unless a criterion says so.
Return strict JSON only with this shape:
{{"criteria_scores":[{{"name":"criterion","score":0.0,"explanation":"brief"}}],"overall_score":0.0,"overall_feedback":"brief","passed":false}}
Pass threshold: {threshold}

Scenario ID: {scenario_id}
Language: {language}
User input: {user_input}

Criteria:
{criteria_text}

Assistant response:
{response}"#
    )
}

fn extract_json(text: &str) -> Option<Value> {
    if let Ok(value) = serde_json::from_str(text.trim()) {
        return Some(value);
    }
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    serde_json::from_str(&text[start..=end]).ok()
}

fn default_threshold() -> f32 {
    0.75
}

fn default_weight() -> f32 {
    1.0
}

fn default_true() -> bool {
    true
}

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
        self.resolve_role(ai_agents_llm::LLMRole::EvaluationResponse, alias)
    }

    /// Selects a judge role without activating the unused suite-wide alias field.
    pub fn resolve_role(
        &self,
        role: ai_agents_llm::LLMRole,
        alias: Option<&str>,
    ) -> Result<LLMJudge> {
        let llm = if let Some(resolved) = self
            .registry
            .resolve_role_override(role, alias)
            .map_err(|error| EvalError::Judge(error.to_string()))?
        {
            resolved.provider
        } else if let Some(alias) = alias {
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
        let value = if self.config.require_json {
            serde_json::from_str(llm_response.content.trim())
                .map_err(|_| EvalError::Judge("judge did not return strict JSON".into()))?
        } else {
            extract_json(&llm_response.content)
                .ok_or_else(|| EvalError::Judge("judge did not return JSON".into()))?
        };
        let mut result: JudgeResult = serde_json::from_value(value)
            .map_err(|_| EvalError::Judge("judge JSON has an invalid result structure".into()))?;
        result.passed = result.overall_score >= threshold;
        result.raw_response = if self.config.require_json {
            None
        } else {
            Some(llm_response.content)
        };
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

#[cfg(test)]
mod tests {
    use super::*;
    use ai_agents_core::{FinishReason, LLMResponse};
    use ai_agents_llm::mock::MockLLMProvider;

    fn assertion() -> JudgeAssertion {
        JudgeAssertion {
            llm: None,
            pass_threshold: 0.75,
            criteria: vec![JudgeCriterion::Text("Relevant".into())],
        }
    }

    fn result_json(extra: &str) -> String {
        format!(
            r#"{{"criteria_scores":[{{"name":"Relevant","score":0.9,"explanation":"ok"}}],"overall_score":0.9,"overall_feedback":"ok","passed":false{extra}}}"#
        )
    }

    fn judge(response: String, require_json: bool) -> LLMJudge {
        let mut provider = MockLLMProvider::new("judge");
        provider.add_response(LLMResponse::new(response, FinishReason::Stop));
        LLMJudge::new(
            Arc::new(provider),
            JudgeConfig {
                require_json,
                ..JudgeConfig::default()
            },
        )
    }

    #[tokio::test]
    async fn strict_judge_requires_the_entire_response_to_be_json() {
        let strict = judge(format!("result: {}", result_json("")), true);
        let error = strict.evaluate("answer", &assertion()).await.unwrap_err();
        assert!(error.to_string().contains("strict JSON"));

        let strict = judge(format!("\n {} \n", result_json("")), true);
        let result = strict.evaluate("answer", &assertion()).await.unwrap();
        assert!(result.passed);
        assert!(result.raw_response.is_none());
    }

    #[tokio::test]
    async fn lenient_judge_extracts_json_and_keeps_the_actual_response() {
        let response = format!("result: {}", result_json(""));
        let result = judge(response.clone(), false)
            .evaluate("answer", &assertion())
            .await
            .unwrap();

        assert!(result.passed);
        assert_eq!(result.raw_response.as_deref(), Some(response.as_str()));
    }

    #[tokio::test]
    async fn framework_owns_the_raw_response_field() {
        let injected = ",\"raw_response\":\"MODEL_VALUE\"";
        let strict = judge(result_json(injected), true)
            .evaluate("answer", &assertion())
            .await
            .unwrap();
        assert!(strict.raw_response.is_none());

        let response = result_json(injected);
        let lenient = judge(response.clone(), false)
            .evaluate("answer", &assertion())
            .await
            .unwrap();
        assert_eq!(lenient.raw_response.as_deref(), Some(response.as_str()));
    }

    #[tokio::test]
    async fn invalid_result_errors_do_not_include_model_values() {
        let response = r#"{"criteria_scores":[],"overall_score":"PRIVATE_SENTINEL","overall_feedback":"ok","passed":false}"#;
        for require_json in [true, false] {
            let error = judge(response.to_string(), require_json)
                .evaluate("answer", &assertion())
                .await
                .unwrap_err()
                .to_string();
            assert!(error.contains("invalid result structure"));
            assert!(!error.contains("PRIVATE_SENTINEL"));
        }
    }
}

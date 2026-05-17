use std::sync::Arc;

use ai_agents_core::{ChatMessage, LLMProvider};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{EvalError, Result};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JudgeAssertion {
    #[serde(default)]
    pub llm: Option<String>,
    #[serde(default = "default_threshold")]
    pub pass_threshold: f32,
    #[serde(default)]
    pub criteria: Vec<JudgeCriterion>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum JudgeCriterion {
    Text(String),
    Object {
        name: String,
        description: String,
        #[serde(default = "default_weight")]
        weight: f32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub llm: Option<String>,
    #[serde(default)]
    pub default_criteria: Vec<JudgeCriterion>,
    #[serde(default = "default_threshold")]
    pub pass_threshold: f32,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeResult {
    pub criteria_scores: Vec<CriterionScore>,
    pub overall_score: f32,
    pub overall_feedback: String,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriterionScore {
    pub name: String,
    pub score: f32,
    #[serde(default)]
    pub explanation: String,
}

pub struct LLMJudge {
    llm: Arc<dyn LLMProvider>,
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
        let criteria = if assertion.criteria.is_empty() {
            self.config.default_criteria.clone()
        } else {
            assertion.criteria.clone()
        };
        if criteria.is_empty() {
            return Err(EvalError::Judge("judge assertion has no criteria".into()));
        }
        let threshold = assertion.pass_threshold;
        let prompt = build_prompt(response, &criteria, threshold);
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

fn build_prompt(response: &str, criteria: &[JudgeCriterion], threshold: f32) -> String {
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
    format!(
        r#"Evaluate the assistant response against the criteria.
Evaluate semantic meaning across languages. Do not require exact wording unless a criterion says so.
Return strict JSON only with this shape:
{{"criteria_scores":[{{"name":"criterion","score":0.0,"explanation":"brief"}}],"overall_score":0.0,"overall_feedback":"brief","passed":false}}
Pass threshold: {threshold}

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

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub passed: bool,
    pub confidence: f32,
    pub criteria_results: Vec<CriterionResult>,
}

impl EvaluationResult {
    pub fn new(passed: bool, confidence: f32) -> Self {
        Self {
            passed,
            confidence,
            criteria_results: Vec::new(),
        }
    }

    pub fn with_criteria(mut self, criteria: Vec<CriterionResult>) -> Self {
        self.criteria_results = criteria;
        self
    }

    pub fn passed_all(&self) -> bool {
        self.criteria_results.iter().all(|c| c.passed)
    }

    pub fn failed_criteria(&self) -> impl Iterator<Item = &CriterionResult> {
        self.criteria_results.iter().filter(|c| !c.passed)
    }

    pub fn passing_criteria(&self) -> impl Iterator<Item = &CriterionResult> {
        self.criteria_results.iter().filter(|c| c.passed)
    }

    pub fn pass_rate(&self) -> f32 {
        if self.criteria_results.is_empty() {
            return if self.passed { 1.0 } else { 0.0 };
        }
        let passed = self.criteria_results.iter().filter(|c| c.passed).count();
        passed as f32 / self.criteria_results.len() as f32
    }
}

impl Default for EvaluationResult {
    fn default() -> Self {
        Self {
            passed: false,
            confidence: 0.0,
            criteria_results: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriterionResult {
    pub criterion: String,
    pub passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl CriterionResult {
    pub fn pass(criterion: impl Into<String>) -> Self {
        Self {
            criterion: criterion.into(),
            passed: true,
            reason: None,
        }
    }

    pub fn fail(criterion: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            criterion: criterion.into(),
            passed: false,
            reason: Some(reason.into()),
        }
    }

    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionAttempt {
    pub response: String,
    pub evaluation: EvaluationResult,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feedback: Option<String>,
}

impl ReflectionAttempt {
    pub fn new(response: impl Into<String>, evaluation: EvaluationResult) -> Self {
        Self {
            response: response.into(),
            evaluation,
            feedback: None,
        }
    }

    pub fn with_feedback(mut self, feedback: impl Into<String>) -> Self {
        self.feedback = Some(feedback.into());
        self
    }

    pub fn passed(&self) -> bool {
        self.evaluation.passed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluation_result_creation() {
        let result = EvaluationResult::new(true, 0.95);
        assert!(result.passed);
        assert_eq!(result.confidence, 0.95);
        assert!(result.criteria_results.is_empty());
    }

    #[test]
    fn test_evaluation_result_with_criteria() {
        let criteria = vec![
            CriterionResult::pass("Addresses question"),
            CriterionResult::fail("Complete response", "Response is truncated"),
            CriterionResult::pass("Helpful"),
        ];

        let result = EvaluationResult::new(false, 0.6).with_criteria(criteria);

        assert!(!result.passed);
        assert!(!result.passed_all());
        assert_eq!(result.failed_criteria().count(), 1);
        assert_eq!(result.passing_criteria().count(), 2);
        assert!((result.pass_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_evaluation_result_pass_rate() {
        let empty = EvaluationResult::new(true, 1.0);
        assert_eq!(empty.pass_rate(), 1.0);

        let empty_failed = EvaluationResult::new(false, 0.0);
        assert_eq!(empty_failed.pass_rate(), 0.0);
    }

    #[test]
    fn test_criterion_result_pass() {
        let criterion = CriterionResult::pass("Response is clear");
        assert!(criterion.passed);
        assert_eq!(criterion.criterion, "Response is clear");
        assert!(criterion.reason.is_none());
    }

    #[test]
    fn test_criterion_result_fail() {
        let criterion = CriterionResult::fail("Accurate information", "Contains factual errors");
        assert!(!criterion.passed);
        assert_eq!(criterion.criterion, "Accurate information");
        assert_eq!(
            criterion.reason,
            Some("Contains factual errors".to_string())
        );
    }

    #[test]
    fn test_criterion_result_with_reason() {
        let criterion = CriterionResult::pass("Good response").with_reason("Excellent formatting");
        assert!(criterion.passed);
        assert_eq!(criterion.reason, Some("Excellent formatting".to_string()));
    }

    #[test]
    fn test_reflection_attempt_creation() {
        let evaluation = EvaluationResult::new(true, 0.9);
        let attempt = ReflectionAttempt::new("This is my response", evaluation);

        assert_eq!(attempt.response, "This is my response");
        assert!(attempt.passed());
        assert!(attempt.feedback.is_none());
    }

    #[test]
    fn test_reflection_attempt_with_feedback() {
        let evaluation = EvaluationResult::new(false, 0.4);
        let attempt = ReflectionAttempt::new("Initial response", evaluation)
            .with_feedback("Be more specific");

        assert!(!attempt.passed());
        assert_eq!(attempt.feedback, Some("Be more specific".to_string()));
    }

    #[test]
    fn test_evaluation_result_serde() {
        let criteria = vec![
            CriterionResult::pass("Clear"),
            CriterionResult::fail("Complete", "Missing details"),
        ];
        let result = EvaluationResult::new(false, 0.7).with_criteria(criteria);

        let json = serde_json::to_string(&result).unwrap();
        let parsed: EvaluationResult = serde_json::from_str(&json).unwrap();

        assert!(!parsed.passed);
        assert_eq!(parsed.confidence, 0.7);
        assert_eq!(parsed.criteria_results.len(), 2);
    }

    #[test]
    fn test_reflection_attempt_serde() {
        let evaluation = EvaluationResult::new(true, 0.95);
        let attempt = ReflectionAttempt::new("Response text", evaluation);

        let json = serde_json::to_string(&attempt).unwrap();
        let parsed: ReflectionAttempt = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.response, "Response text");
        assert!(parsed.passed());
    }
}

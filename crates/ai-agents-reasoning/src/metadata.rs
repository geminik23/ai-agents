use serde::{Deserialize, Serialize};

use crate::evaluation::{EvaluationResult, ReflectionAttempt};
use crate::mode::ReasoningMode;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningMetadata {
    pub mode_used: ReasoningMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    pub iterations: u32,
    pub auto_detected: bool,
}

impl ReasoningMetadata {
    pub fn new(mode_used: ReasoningMode) -> Self {
        Self {
            mode_used,
            thinking: None,
            iterations: 1,
            auto_detected: false,
        }
    }

    pub fn with_thinking(mut self, thinking: impl Into<String>) -> Self {
        self.thinking = Some(thinking.into());
        self
    }

    pub fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn with_auto_detected(mut self, auto_detected: bool) -> Self {
        self.auto_detected = auto_detected;
        self
    }

    pub fn has_thinking(&self) -> bool {
        self.thinking.is_some()
    }
}

impl Default for ReasoningMetadata {
    fn default() -> Self {
        Self::new(ReasoningMode::None)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionMetadata {
    pub attempts: u32,
    pub final_evaluation: EvaluationResult,
    pub history: Vec<ReflectionAttempt>,
}

impl ReflectionMetadata {
    pub fn new(final_evaluation: EvaluationResult) -> Self {
        Self {
            attempts: 1,
            final_evaluation,
            history: Vec::new(),
        }
    }

    pub fn with_attempts(mut self, attempts: u32) -> Self {
        self.attempts = attempts;
        self
    }

    pub fn with_history(mut self, history: Vec<ReflectionAttempt>) -> Self {
        self.history = history;
        self
    }

    pub fn add_attempt(&mut self, attempt: ReflectionAttempt) {
        self.history.push(attempt);
        self.attempts = self.history.len() as u32;
    }

    pub fn passed(&self) -> bool {
        self.final_evaluation.passed
    }

    pub fn required_retries(&self) -> bool {
        self.attempts > 1
    }
}

impl Default for ReflectionMetadata {
    fn default() -> Self {
        Self {
            attempts: 0,
            final_evaluation: EvaluationResult::default(),
            history: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::CriterionResult;

    #[test]
    fn test_reasoning_metadata_creation() {
        let metadata = ReasoningMetadata::new(ReasoningMode::CoT);
        assert_eq!(metadata.mode_used, ReasoningMode::CoT);
        assert!(metadata.thinking.is_none());
        assert_eq!(metadata.iterations, 1);
        assert!(!metadata.auto_detected);
    }

    #[test]
    fn test_reasoning_metadata_with_thinking() {
        let metadata = ReasoningMetadata::new(ReasoningMode::CoT)
            .with_thinking("Step 1: Analyze the problem\nStep 2: Find solution");

        assert!(metadata.has_thinking());
        assert!(metadata.thinking.unwrap().contains("Step 1"));
    }

    #[test]
    fn test_reasoning_metadata_auto_detected() {
        let metadata = ReasoningMetadata::new(ReasoningMode::Auto)
            .with_auto_detected(true)
            .with_iterations(3);

        assert!(metadata.auto_detected);
        assert_eq!(metadata.iterations, 3);
    }

    #[test]
    fn test_reasoning_metadata_default() {
        let metadata = ReasoningMetadata::default();
        assert_eq!(metadata.mode_used, ReasoningMode::None);
        assert!(!metadata.has_thinking());
        assert_eq!(metadata.iterations, 1);
    }

    #[test]
    fn test_reflection_metadata_creation() {
        let evaluation = EvaluationResult::new(true, 0.95);
        let metadata = ReflectionMetadata::new(evaluation);

        assert!(metadata.passed());
        assert_eq!(metadata.attempts, 1);
        assert!(metadata.history.is_empty());
    }

    #[test]
    fn test_reflection_metadata_with_history() {
        let eval1 = EvaluationResult::new(false, 0.5);
        let attempt1 = ReflectionAttempt::new("First response", eval1);

        let eval2 = EvaluationResult::new(true, 0.9);
        let attempt2 = ReflectionAttempt::new("Second response", eval2.clone());

        let metadata = ReflectionMetadata::new(eval2)
            .with_attempts(2)
            .with_history(vec![attempt1, attempt2]);

        assert!(metadata.passed());
        assert!(metadata.required_retries());
        assert_eq!(metadata.history.len(), 2);
    }

    #[test]
    fn test_reflection_metadata_add_attempt() {
        let eval = EvaluationResult::new(false, 0.4);
        let mut metadata = ReflectionMetadata::default();

        metadata.add_attempt(ReflectionAttempt::new("Response 1", eval.clone()));
        assert_eq!(metadata.attempts, 1);

        metadata.add_attempt(ReflectionAttempt::new("Response 2", eval));
        assert_eq!(metadata.attempts, 2);
    }

    #[test]
    fn test_reflection_metadata_default() {
        let metadata = ReflectionMetadata::default();
        assert_eq!(metadata.attempts, 0);
        assert!(!metadata.passed());
        assert!(metadata.history.is_empty());
    }

    #[test]
    fn test_reasoning_metadata_serde() {
        let metadata = ReasoningMetadata::new(ReasoningMode::React)
            .with_thinking("Thought process")
            .with_iterations(5)
            .with_auto_detected(true);

        let json = serde_json::to_string(&metadata).unwrap();
        let parsed: ReasoningMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.mode_used, ReasoningMode::React);
        assert_eq!(parsed.thinking, Some("Thought process".to_string()));
        assert_eq!(parsed.iterations, 5);
        assert!(parsed.auto_detected);
    }

    #[test]
    fn test_reflection_metadata_serde() {
        let criteria = vec![CriterionResult::pass("Clear response")];
        let evaluation = EvaluationResult::new(true, 0.85).with_criteria(criteria);
        let metadata = ReflectionMetadata::new(evaluation).with_attempts(2);

        let json = serde_json::to_string(&metadata).unwrap();
        let parsed: ReflectionMetadata = serde_json::from_str(&json).unwrap();

        assert!(parsed.passed());
        assert_eq!(parsed.attempts, 2);
        assert_eq!(parsed.final_evaluation.criteria_results.len(), 1);
    }
}

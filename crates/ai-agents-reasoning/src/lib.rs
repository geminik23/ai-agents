//! Reasoning and reflection capabilities for AI Agents framework

mod config;
mod evaluation;
mod metadata;
mod mode;
mod plan;
mod planning;

pub use config::{ReasoningConfig, ReflectionConfig};
pub use evaluation::{CriterionResult, EvaluationResult, ReflectionAttempt};
pub use metadata::{ReasoningMetadata, ReflectionMetadata};
pub use mode::{ReasoningMode, ReasoningOutput, ReflectionMode};
pub use plan::{Plan, PlanAction, PlanStatus, PlanStep, StepStatus};
pub use planning::{
    PlanAvailableActions, PlanReflectionConfig, PlanningConfig, StepFailureAction, StringOrList,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_reasoning_config() {
        let yaml = r#"
mode: plan_and_execute
judge_llm: router
output: tagged
max_iterations: 10
planning:
  planner_llm: router
  max_steps: 15
  available:
    tools: all
    skills:
      - analyze
      - summarize
  reflection:
    enabled: true
    on_step_failure: replan
    max_replans: 3
"#;
        let config: ReasoningConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.mode, ReasoningMode::PlanAndExecute);
        assert!(config.is_enabled());
        assert!(config.needs_planning());

        let planning = config.get_planning().unwrap();
        assert_eq!(planning.max_steps, 15);
        assert!(planning.available.tools.is_all());
        assert!(!planning.available.skills.is_all());
        assert!(planning.available.skills.allows("analyze"));
        assert!(planning.reflection.enabled);
    }

    #[test]
    fn test_full_reflection_config() {
        let yaml = r#"
enabled: auto
evaluator_llm: router
max_retries: 3
criteria:
  - "Response addresses the question"
  - "Response is helpful"
  - "Response is accurate"
pass_threshold: 0.75
"#;
        let config: ReflectionConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.is_auto());
        assert!(config.requires_evaluation());
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.criteria.len(), 3);
        assert_eq!(config.pass_threshold, 0.75);
    }

    #[test]
    fn test_plan_workflow() {
        let mut plan = Plan::new("Analyze weather and make recommendation");

        let step1 = PlanStep::new(
            "Get weather for Seoul",
            PlanAction::tool("weather", serde_json::json!({"city": "Seoul"})),
        )
        .with_id("get_seoul");

        let step2 = PlanStep::new(
            "Get weather for Tokyo",
            PlanAction::tool("weather", serde_json::json!({"city": "Tokyo"})),
        )
        .with_id("get_tokyo");

        let step3 = PlanStep::new(
            "Compare weather data",
            PlanAction::think("Compare the weather"),
        )
        .with_id("compare")
        .with_dependencies(vec!["get_seoul".to_string(), "get_tokyo".to_string()]);

        let step4 = PlanStep::new(
            "Make recommendation",
            PlanAction::respond("Based on {{ compare }}..."),
        )
        .with_id("respond")
        .with_dependencies(vec!["compare".to_string()]);

        plan.add_step(step1);
        plan.add_step(step2);
        plan.add_step(step3);
        plan.add_step(step4);

        assert_eq!(plan.steps.len(), 4);

        // Both get_seoul and get_tokyo can run in parallel (no dependencies)
        let executable: Vec<_> = plan
            .steps
            .iter()
            .filter(|s| {
                s.status.is_pending()
                    && s.dependencies
                        .iter()
                        .all(|dep| plan.is_step_completed_pub(dep))
            })
            .collect();
        assert_eq!(executable.len(), 2);
    }

    #[test]
    fn test_evaluation_workflow() {
        let criteria = vec![
            CriterionResult::pass("Addresses question"),
            CriterionResult::fail("Complete response", "Missing conclusion"),
        ];

        let evaluation = EvaluationResult::new(false, 0.6).with_criteria(criteria);

        assert!(!evaluation.passed);
        assert!(!evaluation.passed_all());
        assert_eq!(evaluation.failed_criteria().count(), 1);
        assert!((evaluation.pass_rate() - 0.5).abs() < 0.01);

        let attempt = ReflectionAttempt::new("Initial response", evaluation.clone())
            .with_feedback("Add a conclusion to complete the response");

        assert!(!attempt.passed());
        assert!(attempt.feedback.is_some());

        let mut metadata = ReflectionMetadata::new(evaluation);
        metadata.add_attempt(attempt);
        assert_eq!(metadata.attempts, 1);
    }

    #[test]
    fn test_reasoning_metadata() {
        let metadata = ReasoningMetadata::new(ReasoningMode::CoT)
            .with_thinking("Step 1: Understand\nStep 2: Analyze\nStep 3: Conclude")
            .with_iterations(1)
            .with_auto_detected(false);

        assert_eq!(metadata.mode_used, ReasoningMode::CoT);
        assert!(metadata.has_thinking());
        assert!(!metadata.auto_detected);
    }

    impl Plan {
        fn is_step_completed_pub(&self, step_id: &str) -> bool {
            self.steps
                .iter()
                .find(|s| s.id == step_id)
                .map(|s| s.status.is_completed())
                .unwrap_or(false)
        }
    }
}

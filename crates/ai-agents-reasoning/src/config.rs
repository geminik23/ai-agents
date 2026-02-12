use serde::{Deserialize, Serialize};

use crate::mode::{ReasoningMode, ReasoningOutput, ReflectionMode};
use crate::planning::PlanningConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    #[serde(default)]
    pub mode: ReasoningMode,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_llm: Option<String>,

    #[serde(default)]
    pub output: ReasoningOutput,

    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planning: Option<PlanningConfig>,
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            mode: ReasoningMode::None,
            judge_llm: None,
            output: ReasoningOutput::Hidden,
            max_iterations: default_max_iterations(),
            planning: None,
        }
    }
}

fn default_max_iterations() -> u32 {
    5
}

impl ReasoningConfig {
    pub fn new(mode: ReasoningMode) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }

    pub fn with_judge_llm(mut self, llm: impl Into<String>) -> Self {
        self.judge_llm = Some(llm.into());
        self
    }

    pub fn with_output(mut self, output: ReasoningOutput) -> Self {
        self.output = output;
        self
    }

    pub fn with_max_iterations(mut self, max: u32) -> Self {
        self.max_iterations = max;
        self
    }

    pub fn with_planning(mut self, planning: PlanningConfig) -> Self {
        self.planning = Some(planning);
        self
    }

    pub fn is_enabled(&self) -> bool {
        !matches!(self.mode, ReasoningMode::None)
    }

    pub fn needs_planning(&self) -> bool {
        self.mode.uses_planning()
    }

    pub fn get_planning(&self) -> Option<&PlanningConfig> {
        self.planning.as_ref()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionConfig {
    #[serde(default)]
    pub enabled: ReflectionMode,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluator_llm: Option<String>,

    #[serde(default = "default_max_retries")]
    pub max_retries: u32,

    #[serde(default = "default_criteria")]
    pub criteria: Vec<String>,

    #[serde(default = "default_pass_threshold")]
    pub pass_threshold: f32,
}

impl Default for ReflectionConfig {
    fn default() -> Self {
        Self {
            enabled: ReflectionMode::Disabled,
            evaluator_llm: None,
            max_retries: default_max_retries(),
            criteria: default_criteria(),
            pass_threshold: default_pass_threshold(),
        }
    }
}

fn default_max_retries() -> u32 {
    2
}

fn default_pass_threshold() -> f32 {
    0.7
}

fn default_criteria() -> Vec<String> {
    vec![
        "Response directly addresses the user's question".to_string(),
        "Response is complete and not cut off".to_string(),
        "Response is accurate and helpful".to_string(),
    ]
}

impl ReflectionConfig {
    pub fn new(enabled: ReflectionMode) -> Self {
        Self {
            enabled,
            ..Default::default()
        }
    }

    pub fn enabled() -> Self {
        Self::new(ReflectionMode::Enabled)
    }

    pub fn disabled() -> Self {
        Self::new(ReflectionMode::Disabled)
    }

    pub fn auto() -> Self {
        Self::new(ReflectionMode::Auto)
    }

    pub fn with_evaluator_llm(mut self, llm: impl Into<String>) -> Self {
        self.evaluator_llm = Some(llm.into());
        self
    }

    pub fn with_max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    pub fn with_criteria(mut self, criteria: Vec<String>) -> Self {
        self.criteria = criteria;
        self
    }

    pub fn add_criterion(mut self, criterion: impl Into<String>) -> Self {
        self.criteria.push(criterion.into());
        self
    }

    pub fn with_pass_threshold(mut self, threshold: f32) -> Self {
        self.pass_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.is_enabled()
    }

    pub fn is_auto(&self) -> bool {
        self.enabled.is_auto()
    }

    pub fn requires_evaluation(&self) -> bool {
        self.enabled.requires_evaluation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoning_config_default() {
        let config = ReasoningConfig::default();
        assert_eq!(config.mode, ReasoningMode::None);
        assert!(config.judge_llm.is_none());
        assert_eq!(config.output, ReasoningOutput::Hidden);
        assert_eq!(config.max_iterations, 5);
        assert!(config.planning.is_none());
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_reasoning_config_new() {
        let config = ReasoningConfig::new(ReasoningMode::CoT);
        assert_eq!(config.mode, ReasoningMode::CoT);
        assert!(config.is_enabled());
    }

    #[test]
    fn test_reasoning_config_builder() {
        let config = ReasoningConfig::new(ReasoningMode::PlanAndExecute)
            .with_judge_llm("router")
            .with_output(ReasoningOutput::Tagged)
            .with_max_iterations(10)
            .with_planning(PlanningConfig::default());

        assert_eq!(config.judge_llm, Some("router".to_string()));
        assert_eq!(config.output, ReasoningOutput::Tagged);
        assert_eq!(config.max_iterations, 10);
        assert!(config.planning.is_some());
        assert!(config.needs_planning());
    }

    #[test]
    fn test_reasoning_config_serde() {
        let yaml = r#"
mode: cot
judge_llm: router
output: tagged
max_iterations: 8
"#;
        let config: ReasoningConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.mode, ReasoningMode::CoT);
        assert_eq!(config.judge_llm, Some("router".to_string()));
        assert_eq!(config.output, ReasoningOutput::Tagged);
        assert_eq!(config.max_iterations, 8);
    }

    #[test]
    fn test_reasoning_config_serde_with_planning() {
        let yaml = r#"
mode: plan_and_execute
planning:
  planner_llm: router
  max_steps: 15
  available:
    tools: all
    skills:
      - skill1
      - skill2
"#;
        let config: ReasoningConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.mode, ReasoningMode::PlanAndExecute);
        assert!(config.needs_planning());
        let planning = config.get_planning().unwrap();
        assert_eq!(planning.planner_llm, Some("router".to_string()));
        assert_eq!(planning.max_steps, 15);
    }

    #[test]
    fn test_reflection_config_default() {
        let config = ReflectionConfig::default();
        assert_eq!(config.enabled, ReflectionMode::Disabled);
        assert!(config.evaluator_llm.is_none());
        assert_eq!(config.max_retries, 2);
        assert_eq!(config.criteria.len(), 3);
        assert_eq!(config.pass_threshold, 0.7);
        assert!(!config.is_enabled());
        assert!(!config.requires_evaluation());
    }

    #[test]
    fn test_reflection_config_constructors() {
        let enabled = ReflectionConfig::enabled();
        assert!(enabled.is_enabled());

        let disabled = ReflectionConfig::disabled();
        assert!(!disabled.is_enabled());

        let auto = ReflectionConfig::auto();
        assert!(auto.is_auto());
        assert!(auto.requires_evaluation());
    }

    #[test]
    fn test_reflection_config_builder() {
        let config = ReflectionConfig::enabled()
            .with_evaluator_llm("evaluator")
            .with_max_retries(5)
            .with_criteria(vec!["Criterion 1".to_string()])
            .add_criterion("Criterion 2")
            .with_pass_threshold(0.85);

        assert_eq!(config.evaluator_llm, Some("evaluator".to_string()));
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.criteria.len(), 2);
        assert_eq!(config.pass_threshold, 0.85);
    }

    #[test]
    fn test_reflection_config_threshold_clamping() {
        let config = ReflectionConfig::default().with_pass_threshold(1.5);
        assert_eq!(config.pass_threshold, 1.0);

        let config = ReflectionConfig::default().with_pass_threshold(-0.5);
        assert_eq!(config.pass_threshold, 0.0);
    }

    #[test]
    fn test_reflection_config_serde() {
        let yaml = r#"
enabled: auto
evaluator_llm: router
max_retries: 3
criteria:
  - "Response is clear"
  - "Response is accurate"
pass_threshold: 0.8
"#;
        let config: ReflectionConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.is_auto());
        assert_eq!(config.evaluator_llm, Some("router".to_string()));
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.criteria.len(), 2);
        assert_eq!(config.pass_threshold, 0.8);
    }

    #[test]
    fn test_reflection_config_serde_enabled_alias() {
        let yaml = r#"enabled: true"#;
        let config: ReflectionConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.is_enabled());

        let yaml = r#"enabled: false"#;
        let config: ReflectionConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(!config.is_enabled());
    }
}

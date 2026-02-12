use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningMode {
    #[default]
    None,
    #[serde(alias = "cot")]
    CoT,
    React,
    PlanAndExecute,
    Auto,
}

impl ReasoningMode {
    pub fn requires_llm_judge(&self) -> bool {
        matches!(self, ReasoningMode::Auto)
    }

    pub fn uses_planning(&self) -> bool {
        matches!(self, ReasoningMode::PlanAndExecute)
    }

    pub fn uses_iteration(&self) -> bool {
        matches!(self, ReasoningMode::React | ReasoningMode::PlanAndExecute)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ReflectionMode {
    #[default]
    #[serde(alias = "false")]
    Disabled,
    #[serde(alias = "true")]
    Enabled,
    Auto,
}

impl ReflectionMode {
    pub fn is_enabled(&self) -> bool {
        matches!(self, ReflectionMode::Enabled)
    }

    pub fn is_auto(&self) -> bool {
        matches!(self, ReflectionMode::Auto)
    }

    pub fn requires_evaluation(&self) -> bool {
        !matches!(self, ReflectionMode::Disabled)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningOutput {
    #[default]
    Hidden,
    Visible,
    Tagged,
}

impl ReasoningOutput {
    pub fn should_include_thinking(&self) -> bool {
        !matches!(self, ReasoningOutput::Hidden)
    }

    pub fn uses_tags(&self) -> bool {
        matches!(self, ReasoningOutput::Tagged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoning_mode_default() {
        let mode: ReasoningMode = Default::default();
        assert_eq!(mode, ReasoningMode::None);
    }

    #[test]
    fn test_reasoning_mode_serde() {
        let mode: ReasoningMode = serde_json::from_str(r#""cot""#).unwrap();
        assert_eq!(mode, ReasoningMode::CoT);

        let mode: ReasoningMode = serde_json::from_str(r#""plan_and_execute""#).unwrap();
        assert_eq!(mode, ReasoningMode::PlanAndExecute);

        let mode: ReasoningMode = serde_json::from_str(r#""auto""#).unwrap();
        assert_eq!(mode, ReasoningMode::Auto);
    }

    #[test]
    fn test_reasoning_mode_requires_llm_judge() {
        assert!(!ReasoningMode::None.requires_llm_judge());
        assert!(!ReasoningMode::CoT.requires_llm_judge());
        assert!(ReasoningMode::Auto.requires_llm_judge());
    }

    #[test]
    fn test_reasoning_mode_uses_planning() {
        assert!(!ReasoningMode::None.uses_planning());
        assert!(!ReasoningMode::CoT.uses_planning());
        assert!(ReasoningMode::PlanAndExecute.uses_planning());
    }

    #[test]
    fn test_reflection_mode_default() {
        let mode: ReflectionMode = Default::default();
        assert_eq!(mode, ReflectionMode::Disabled);
    }

    #[test]
    fn test_reflection_mode_serde_aliases() {
        let mode: ReflectionMode = serde_json::from_str(r#""true""#).unwrap();
        assert_eq!(mode, ReflectionMode::Enabled);

        let mode: ReflectionMode = serde_json::from_str(r#""false""#).unwrap();
        assert_eq!(mode, ReflectionMode::Disabled);

        let mode: ReflectionMode = serde_json::from_str(r#""enabled""#).unwrap();
        assert_eq!(mode, ReflectionMode::Enabled);
    }

    #[test]
    fn test_reflection_mode_methods() {
        assert!(!ReflectionMode::Disabled.is_enabled());
        assert!(ReflectionMode::Enabled.is_enabled());
        assert!(!ReflectionMode::Auto.is_enabled());

        assert!(ReflectionMode::Auto.is_auto());
        assert!(!ReflectionMode::Enabled.is_auto());

        assert!(!ReflectionMode::Disabled.requires_evaluation());
        assert!(ReflectionMode::Enabled.requires_evaluation());
        assert!(ReflectionMode::Auto.requires_evaluation());
    }

    #[test]
    fn test_reasoning_output_default() {
        let output: ReasoningOutput = Default::default();
        assert_eq!(output, ReasoningOutput::Hidden);
    }

    #[test]
    fn test_reasoning_output_methods() {
        assert!(!ReasoningOutput::Hidden.should_include_thinking());
        assert!(ReasoningOutput::Visible.should_include_thinking());
        assert!(ReasoningOutput::Tagged.should_include_thinking());

        assert!(!ReasoningOutput::Hidden.uses_tags());
        assert!(!ReasoningOutput::Visible.uses_tags());
        assert!(ReasoningOutput::Tagged.uses_tags());
    }
}

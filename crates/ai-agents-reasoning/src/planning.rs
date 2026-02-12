use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanningConfig {
    #[serde(default)]
    pub planner_llm: Option<String>,

    #[serde(default = "default_max_steps")]
    pub max_steps: u32,

    #[serde(default)]
    pub available: PlanAvailableActions,

    #[serde(default)]
    pub reflection: PlanReflectionConfig,
}

impl Default for PlanningConfig {
    fn default() -> Self {
        Self {
            planner_llm: None,
            max_steps: default_max_steps(),
            available: PlanAvailableActions::default(),
            reflection: PlanReflectionConfig::default(),
        }
    }
}

fn default_max_steps() -> u32 {
    10
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlanAvailableActions {
    #[serde(default = "default_all_string")]
    pub tools: StringOrList,

    #[serde(default = "default_all_string")]
    pub skills: StringOrList,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StringOrList {
    All(String),
    List(Vec<String>),
}

impl Default for StringOrList {
    fn default() -> Self {
        StringOrList::All("all".to_string())
    }
}

fn default_all_string() -> StringOrList {
    StringOrList::All("all".to_string())
}

impl StringOrList {
    pub fn is_all(&self) -> bool {
        matches!(self, StringOrList::All(s) if s == "all")
    }

    pub fn allows(&self, id: &str) -> bool {
        match self {
            StringOrList::All(s) if s == "all" => true,
            StringOrList::All(_) => false,
            StringOrList::List(list) => list.iter().any(|s| s == id),
        }
    }

    pub fn as_list(&self) -> Option<&[String]> {
        match self {
            StringOrList::List(list) => Some(list),
            StringOrList::All(_) => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanReflectionConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default)]
    pub on_step_failure: StepFailureAction,

    #[serde(default = "default_max_replans")]
    pub max_replans: u32,
}

impl Default for PlanReflectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            on_step_failure: StepFailureAction::default(),
            max_replans: default_max_replans(),
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_max_replans() -> u32 {
    2
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StepFailureAction {
    #[default]
    Replan,
    Skip,
    Abort,
    Continue,
}

impl StepFailureAction {
    pub fn should_stop(&self) -> bool {
        matches!(self, StepFailureAction::Abort)
    }

    pub fn should_replan(&self) -> bool {
        matches!(self, StepFailureAction::Replan)
    }

    pub fn should_skip(&self) -> bool {
        matches!(self, StepFailureAction::Skip)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planning_config_default() {
        let config = PlanningConfig::default();
        assert!(config.planner_llm.is_none());
        assert_eq!(config.max_steps, 10);
        assert!(config.available.tools.is_all());
        assert!(config.available.skills.is_all());
        assert!(config.reflection.enabled);
    }

    #[test]
    fn test_planning_config_serde() {
        let yaml = r#"
planner_llm: router
max_steps: 15
available:
  tools: all
  skills:
    - skill1
    - skill2
reflection:
  enabled: true
  on_step_failure: skip
  max_replans: 3
"#;
        let config: PlanningConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.planner_llm, Some("router".to_string()));
        assert_eq!(config.max_steps, 15);
        assert!(config.available.tools.is_all());
        assert!(!config.available.skills.is_all());
        assert!(config.available.skills.allows("skill1"));
        assert!(!config.available.skills.allows("skill3"));
        assert!(config.reflection.enabled);
        assert_eq!(config.reflection.on_step_failure, StepFailureAction::Skip);
        assert_eq!(config.reflection.max_replans, 3);
    }

    #[test]
    fn test_string_or_list() {
        let all = StringOrList::All("all".to_string());
        assert!(all.is_all());
        assert!(all.allows("anything"));
        assert!(all.as_list().is_none());

        let list = StringOrList::List(vec!["a".to_string(), "b".to_string()]);
        assert!(!list.is_all());
        assert!(list.allows("a"));
        assert!(list.allows("b"));
        assert!(!list.allows("c"));
        assert_eq!(
            list.as_list(),
            Some(["a".to_string(), "b".to_string()].as_slice())
        );
    }

    #[test]
    fn test_step_failure_action() {
        assert!(StepFailureAction::Abort.should_stop());
        assert!(!StepFailureAction::Replan.should_stop());

        assert!(StepFailureAction::Replan.should_replan());
        assert!(!StepFailureAction::Skip.should_replan());

        assert!(StepFailureAction::Skip.should_skip());
        assert!(!StepFailureAction::Continue.should_skip());
    }

    #[test]
    fn test_plan_reflection_config_default() {
        let config = PlanReflectionConfig::default();
        assert!(config.enabled);
        assert_eq!(config.on_step_failure, StepFailureAction::Replan);
        assert_eq!(config.max_replans, 2);
    }
}

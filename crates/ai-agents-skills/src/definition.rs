use ai_agents_reasoning::{ReasoningConfig, ReflectionConfig};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDefinition {
    #[serde(alias = "skill")]
    pub id: String,
    pub description: String,
    pub trigger: String,
    pub steps: Vec<SkillStep>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reflection: Option<ReflectionConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SkillStep {
    Tool {
        tool: String,
        #[serde(default)]
        args: Option<Value>,
        #[serde(default)]
        output_as: Option<String>,
    },
    Prompt {
        prompt: String,
        #[serde(default)]
        llm: Option<String>,
    },
}

impl SkillStep {
    pub fn is_tool(&self) -> bool {
        matches!(self, SkillStep::Tool { .. })
    }

    pub fn is_prompt(&self) -> bool {
        matches!(self, SkillStep::Prompt { .. })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SkillRef {
    Name(String),
    File { file: PathBuf },
    Inline(SkillDefinition),
}

impl SkillRef {
    pub fn name(id: impl Into<String>) -> Self {
        SkillRef::Name(id.into())
    }

    pub fn file(path: impl Into<PathBuf>) -> Self {
        SkillRef::File { file: path.into() }
    }

    pub fn inline(def: SkillDefinition) -> Self {
        SkillRef::Inline(def)
    }
}

#[derive(Debug, Clone, Default)]
pub struct SkillContext {
    pub user_input: String,
    pub step_results: Vec<StepResult>,
    pub extra: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub step_index: usize,
    pub args: Option<Value>,
    pub result: Value,
}

impl SkillContext {
    pub fn new(user_input: impl Into<String>) -> Self {
        Self {
            user_input: user_input.into(),
            step_results: Vec::new(),
            extra: Value::Null,
        }
    }

    pub fn with_extra(mut self, extra: Value) -> Self {
        self.extra = extra;
        self
    }

    pub fn add_result(&mut self, step_index: usize, args: Option<Value>, result: Value) {
        self.step_results.push(StepResult {
            step_index,
            args,
            result,
        });
    }

    pub fn get_result(&self, index: usize) -> Option<&StepResult> {
        self.step_results.iter().find(|r| r.step_index == index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_definition_parse() {
        let yaml = r#"
id: weather_clothes
description: "Weather-based clothing recommendation"
trigger: "When user asks about clothing"
steps:
  - tool: get_temperature
    args:
      location: "Seoul"
  - prompt: |
      Temperature: {{ steps[0].result.temperature }}
      Please recommend appropriate clothing.
"#;
        let def: SkillDefinition = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(def.id, "weather_clothes");
        assert_eq!(def.steps.len(), 2);
        assert!(def.steps[0].is_tool());
        assert!(def.steps[1].is_prompt());
    }

    #[test]
    fn test_skill_ref_name() {
        let yaml = r#""weather_clothes""#;
        let skill_ref: SkillRef = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(skill_ref, SkillRef::Name(name) if name == "weather_clothes"));
    }

    #[test]
    fn test_skill_ref_file() {
        let yaml = r#"file: ./skills/my_skill.yaml"#;
        let skill_ref: SkillRef = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(skill_ref, SkillRef::File { .. }));
    }

    #[test]
    fn test_skill_ref_inline() {
        let yaml = r#"
id: inline_skill
description: "Inline skill"
trigger: "When user asks"
steps:
  - prompt: "Hello"
"#;
        let skill_ref: SkillRef = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(skill_ref, SkillRef::Inline(_)));
    }

    #[test]
    fn test_skill_definition_with_reasoning() {
        let yaml = r#"
id: analysis_skill
description: "Analyze data"
trigger: "When user asks for analysis"
reasoning:
  mode: cot
reflection:
  enabled: true
  criteria:
    - "Analysis is thorough"
steps:
  - prompt: "Analyze the input"
"#;
        let def: SkillDefinition = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(def.id, "analysis_skill");
        assert!(def.reasoning.is_some());
        assert!(def.reflection.is_some());

        let reasoning = def.reasoning.unwrap();
        assert_eq!(reasoning.mode, ai_agents_reasoning::ReasoningMode::CoT);

        let reflection = def.reflection.unwrap();
        assert!(reflection.is_enabled());
    }

    #[test]
    fn test_skill_ref_inline_with_reasoning() {
        let yaml = r#"
id: inline_with_reasoning
description: "Inline skill with reasoning"
trigger: "When user asks"
reasoning:
  mode: none
steps:
  - prompt: "Hello"
"#;
        let skill_ref: SkillRef = serde_yaml::from_str(yaml).unwrap();
        if let SkillRef::Inline(def) = skill_ref {
            assert!(def.reasoning.is_some());
            let reasoning = def.reasoning.unwrap();
            assert_eq!(reasoning.mode, ai_agents_reasoning::ReasoningMode::None);
        } else {
            panic!("Expected inline skill");
        }
    }

    #[test]
    fn test_skill_context() {
        let mut ctx = SkillContext::new("what to wear?");
        ctx.add_result(0, None, serde_json::json!({"temperature": 15}));

        assert_eq!(ctx.user_input, "what to wear?");
        assert_eq!(ctx.step_results.len(), 1);
        assert!(ctx.get_result(0).is_some());
        assert!(ctx.get_result(1).is_none());
    }
}

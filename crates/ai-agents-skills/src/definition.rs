use ai_agents_core::autonomy::AutonomyOverride;
use ai_agents_disambiguation::SkillDisambiguationOverride;
use ai_agents_reasoning::{ReasoningConfig, ReflectionConfig};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SkillDefinition {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub autonomy: Option<AutonomyOverride>,
    #[serde(alias = "skill")]
    pub id: String,
    pub description: String,
    pub trigger: String,
    pub steps: Vec<SkillStep>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reflection: Option<ReflectionConfig>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disambiguation: Option<SkillDisambiguationOverride>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
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

/// A named, file-backed, or inline skill reference.
///
/// The inline variant remains unboxed to preserve the frozen public construction and serde shape.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SkillContext {
    pub user_input: String,
    pub step_results: Vec<StepResult>,
    pub extra: Value,
}

/// Exact script progress retains rendered pending arguments and completed template inputs without rerouting.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SkillExecutionCursor {
    pub execution_id: String,
    pub definition: SkillDefinition,
    pub context: SkillContext,
    pub next_step: usize,
    pub pending: Option<ai_agents_core::ToolExecutionRequest>,
    pub response: Option<String>,
}

impl SkillExecutionCursor {
    /// A cursor starts before any prompt or tool effect; only its bound definition may advance it.
    pub fn new(skill: &SkillDefinition, input: &str, extra: Value) -> Self {
        Self {
            execution_id: uuid::Uuid::new_v4().to_string(),
            definition: skill.clone(),
            context: SkillContext::new(input).with_extra(extra),
            next_step: 0,
            pending: None,
            response: None,
        }
    }

    /// Completed results form an exact prefix and a pending request belongs to the next declared tool step.
    pub fn validate(&self, skill: &SkillDefinition) -> ai_agents_core::Result<()> {
        if self.execution_id.is_empty()
            || self.execution_id.len() > 128
            || serde_json::to_value(&self.definition)? != serde_json::to_value(skill)?
            || self.next_step > skill.steps.len()
            || self.context.step_results.len() != self.next_step
            || self
                .context
                .step_results
                .iter()
                .enumerate()
                .any(|(index, result)| result.step_index != index)
        {
            return Err(ai_agents_core::AgentError::Skill(
                "invalid or changed skill continuation".into(),
            ));
        }
        if let Some(request) = &self.pending
            && (request.call_id.is_empty()
                || !matches!(&request.source, ai_agents_core::ToolCallSource::Skill {skill_id,step_index} if skill_id == &skill.id && *step_index == self.next_step)
                || !matches!(skill.steps.get(self.next_step), Some(SkillStep::Tool {tool,..}) if tool == &request.requested_name))
        {
            return Err(ai_agents_core::AgentError::Skill(
                "pending skill operation is not bound to its step".into(),
            ));
        }
        Ok(())
    }

    /// A resumed result advances once and preserves original rendered arguments separately from approved execution arguments.
    pub fn accept_tool_result(
        &mut self,
        record: ai_agents_core::ToolExecutionRecord,
    ) -> ai_agents_core::Result<()> {
        let request = self
            .pending
            .as_ref()
            .ok_or_else(|| ai_agents_core::AgentError::Skill("skill has no pending tool".into()))?;
        if request.call_id != record.call_id
            || (!matches!(
                record.source,
                ai_agents_core::ToolCallSource::Fallback { .. }
            ) && (request.requested_name != record.requested_name
                || request.arguments != record.arguments))
        {
            return Err(ai_agents_core::AgentError::Skill(
                "skill result binding changed".into(),
            ));
        }
        if !record.success {
            return Err(ai_agents_core::AgentError::Skill(format!(
                "Tool '{}' failed: {}",
                request.requested_name,
                record.model_output_string()
            )));
        }
        self.context.add_result_with_metadata(
            self.next_step,
            Some(request.arguments.clone()),
            record.model_output_value(),
            Some(serde_json::to_value(&record)?),
        );
        self.next_step += 1;
        self.pending = None;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepResult {
    pub step_index: usize,
    pub args: Option<Value>,
    pub result: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
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
        self.add_result_with_metadata(step_index, args, result, None);
    }

    pub fn add_result_with_metadata(
        &mut self,
        step_index: usize,
        args: Option<Value>,
        result: Value,
        metadata: Option<Value>,
    ) {
        self.step_results.push(StepResult {
            step_index,
            args,
            result,
            metadata,
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
    fn test_skill_ref_rejects_null_non_string_and_field_typos() {
        for yaml in ["null", "42", "file: ./skills/my_skill.yaml\nflie: true\n"] {
            assert!(serde_yaml::from_str::<SkillRef>(yaml).is_err(), "{yaml}");
        }
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
    fn test_skill_definition_with_disambiguation() {
        let yaml = r#"
id: transfer_money
description: "Transfer money between accounts"
trigger: "When user wants to transfer money"
disambiguation:
  enabled: true
  threshold: 0.9
  required_clarity:
    - recipient
    - amount
  clarification_templates:
    missing_recipient: "누구에게 보내시겠습니까?"
    missing_amount: "얼마를 보내시겠습니까?"
steps:
  - prompt: "Processing transfer"
"#;
        let def: SkillDefinition = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(def.id, "transfer_money");
        let disambig = def.disambiguation.as_ref().unwrap();
        assert_eq!(disambig.enabled, Some(true));
        assert_eq!(disambig.threshold, Some(0.9));
        assert_eq!(disambig.required_clarity.len(), 2);
        assert_eq!(disambig.clarification_templates.len(), 2);
        assert_eq!(
            disambig
                .clarification_templates
                .get("missing_recipient")
                .unwrap(),
            "누구에게 보내시겠습니까?"
        );
    }

    #[test]
    fn test_skill_definition_accepts_alias_and_empty_disambiguation_fields() {
        let yaml = r#"
skill: transfer_money
description: "Transfer money"
trigger: "When user wants to transfer money"
disambiguation:
  required_clarity: []
  clarification_templates: {}
steps: []
"#;
        let def: SkillDefinition = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(def.id, "transfer_money");
        let disambiguation = def.disambiguation.unwrap();
        assert!(disambiguation.required_clarity.is_empty());
        assert!(disambiguation.clarification_templates.is_empty());
        assert!(def.steps.is_empty());
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

use std::sync::Arc;

use ai_agents_core::{AgentError, ChatMessage, LLMProvider, Result};

use crate::definition::SkillDefinition;

pub struct SkillRouter {
    llm: Arc<dyn LLMProvider>,
    skills: Vec<SkillDefinition>,
}

impl SkillRouter {
    pub fn new(llm: Arc<dyn LLMProvider>, skills: Vec<SkillDefinition>) -> Self {
        Self { llm, skills }
    }

    pub fn with_skills(mut self, skills: Vec<SkillDefinition>) -> Self {
        self.skills = skills;
        self
    }

    pub fn add_skill(&mut self, skill: SkillDefinition) {
        self.skills.push(skill);
    }

    pub fn skills(&self) -> &[SkillDefinition] {
        &self.skills
    }

    pub async fn select_skill(&self, user_input: &str) -> Result<Option<String>> {
        if self.skills.is_empty() {
            return Ok(None);
        }

        let skills_desc = self
            .skills
            .iter()
            .map(|s| format!("- {}: {} (trigger: {})", s.id, s.description, s.trigger))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            r#"Analyze the user input and select an appropriate skill.

Available skills:
{}

User input: "{}"

Return ONLY the skill id if one matches. Return "none" if no skill matches.
Do not include any explanation."#,
            skills_desc, user_input
        );

        let response = self
            .llm
            .complete(&[ChatMessage::user(&prompt)], None)
            .await
            .map_err(|e| AgentError::LLM(e.to_string()))?;

        let selected = response.content.trim().to_lowercase();

        if selected == "none" {
            return Ok(None);
        }

        if self.skills.iter().any(|s| s.id.to_lowercase() == selected) {
            let original_id = self
                .skills
                .iter()
                .find(|s| s.id.to_lowercase() == selected)
                .map(|s| s.id.clone())
                .unwrap();
            Ok(Some(original_id))
        } else {
            Ok(None)
        }
    }

    pub fn get_skill(&self, id: &str) -> Option<&SkillDefinition> {
        self.skills.iter().find(|s| s.id == id)
    }

    /// Select a skill from a filtered subset of available skills
    pub async fn select_skill_filtered(
        &self,
        user_input: &str,
        allowed_skill_ids: &[&str],
    ) -> Result<Option<String>> {
        if allowed_skill_ids.is_empty() {
            return Ok(None);
        }

        let filtered_skills: Vec<&SkillDefinition> = self
            .skills
            .iter()
            .filter(|s| allowed_skill_ids.contains(&s.id.as_str()))
            .collect();

        if filtered_skills.is_empty() {
            return Ok(None);
        }

        let skills_desc = filtered_skills
            .iter()
            .map(|s| format!("- {}: {} (trigger: {})", s.id, s.description, s.trigger))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            r#"Analyze the user input and select an appropriate skill.

Available skills:
{}

User input: "{}"

Return ONLY the skill id if one matches. Return "none" if no skill matches.
Do not include any explanation."#,
            skills_desc, user_input
        );

        let response = self
            .llm
            .complete(&[ChatMessage::user(&prompt)], None)
            .await
            .map_err(|e| AgentError::LLM(e.to_string()))?;

        let selected = response.content.trim().to_lowercase();

        if selected == "none" {
            return Ok(None);
        }

        if filtered_skills
            .iter()
            .any(|s| s.id.to_lowercase() == selected)
        {
            let original_id = filtered_skills
                .iter()
                .find(|s| s.id.to_lowercase() == selected)
                .map(|s| s.id.clone())
                .unwrap();
            Ok(Some(original_id))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definition::SkillStep;
    use ai_agents_core::{FinishReason, LLMResponse};
    use ai_agents_llm::mock::MockLLMProvider;

    fn create_test_skills() -> Vec<SkillDefinition> {
        vec![
            SkillDefinition {
                id: "weather_clothes".to_string(),
                description: "Recommend clothes based on weather".to_string(),
                trigger: "When user asks about what to wear".to_string(),
                steps: vec![SkillStep::Prompt {
                    prompt: "Recommend clothes".to_string(),
                    llm: None,
                }],
                reasoning: None,
                reflection: None,
            },
            SkillDefinition {
                id: "calculator".to_string(),
                description: "Perform calculations".to_string(),
                trigger: "When user needs math calculations".to_string(),
                steps: vec![SkillStep::Tool {
                    tool: "calculator".to_string(),
                    args: None,
                    output_as: None,
                }],
                reasoning: None,
                reflection: None,
            },
        ]
    }

    fn create_mock_with_response(response: &str) -> MockLLMProvider {
        let mut mock = MockLLMProvider::new("router_mock");
        mock.add_response(LLMResponse::new(response, FinishReason::Stop));
        mock
    }

    #[tokio::test]
    async fn test_router_empty_skills() {
        let mock = create_mock_with_response("none");
        let router = SkillRouter::new(Arc::new(mock), vec![]);

        let result = router.select_skill("hello").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_router_selects_skill() {
        let mock = create_mock_with_response("weather_clothes");
        let router = SkillRouter::new(Arc::new(mock), create_test_skills());

        let result = router
            .select_skill("What should I wear today?")
            .await
            .unwrap();
        assert_eq!(result, Some("weather_clothes".to_string()));
    }

    #[tokio::test]
    async fn test_router_no_match() {
        let mock = create_mock_with_response("none");
        let router = SkillRouter::new(Arc::new(mock), create_test_skills());

        let result = router.select_skill("Tell me a joke").await.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_skill() {
        let mock = MockLLMProvider::new("router_mock");
        let router = SkillRouter::new(Arc::new(mock), create_test_skills());

        assert!(router.get_skill("weather_clothes").is_some());
        assert!(router.get_skill("calculator").is_some());
        assert!(router.get_skill("unknown").is_none());
    }
}

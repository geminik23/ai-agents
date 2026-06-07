use ai_agents_skills::SkillDefinition;

/// Skill selected by a speculative routing branch before execution commits.
#[derive(Debug, Clone)]
pub struct SkillCandidate {
    pub skill_id: String,
    pub skill: SkillDefinition,
}

impl SkillCandidate {
    pub fn new(skill_id: impl Into<String>, skill: SkillDefinition) -> Self {
        Self {
            skill_id: skill_id.into(),
            skill,
        }
    }
}

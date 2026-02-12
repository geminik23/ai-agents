use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ai_agents_core::{AgentError, Result};

use crate::definition::{SkillDefinition, SkillRef};

pub struct SkillLoader {
    search_paths: Vec<PathBuf>,
    cache: HashMap<String, SkillDefinition>,
}

impl SkillLoader {
    pub fn new() -> Self {
        Self {
            search_paths: vec![PathBuf::from("templates/skills")],
            cache: HashMap::new(),
        }
    }

    pub fn add_search_path(&mut self, path: impl Into<PathBuf>) {
        self.search_paths.push(path.into());
    }

    pub fn load_refs(&mut self, refs: &[SkillRef]) -> Result<Vec<SkillDefinition>> {
        let mut skills = Vec::new();
        for skill_ref in refs {
            let skill = self.load_ref(skill_ref)?;
            skills.push(skill);
        }
        Ok(skills)
    }

    pub fn load_ref(&mut self, skill_ref: &SkillRef) -> Result<SkillDefinition> {
        match skill_ref {
            SkillRef::Name(name) => self.load_by_name(name),
            SkillRef::File { file } => self.load_from_path(file),
            SkillRef::Inline(def) => Ok(def.clone()),
        }
    }

    pub fn load_by_name(&mut self, name: &str) -> Result<SkillDefinition> {
        if let Some(cached) = self.cache.get(name) {
            return Ok(cached.clone());
        }

        let file_name = format!("{}.skill.yaml", name);

        for search_path in &self.search_paths {
            let path = search_path.join(&file_name);
            if path.exists() {
                let skill = self.load_from_path(&path)?;
                self.cache.insert(name.to_string(), skill.clone());
                return Ok(skill);
            }
        }

        Err(AgentError::Skill(format!(
            "Skill '{}' not found in search paths: {:?}",
            name, self.search_paths
        )))
    }

    pub fn load_from_path(&mut self, path: &Path) -> Result<SkillDefinition> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            AgentError::Skill(format!("Failed to read skill file {:?}: {}", path, e))
        })?;

        let skill: SkillDefinition = serde_yaml::from_str(&content).map_err(|e| {
            AgentError::Skill(format!("Failed to parse skill file {:?}: {}", path, e))
        })?;

        self.cache.insert(skill.id.clone(), skill.clone());
        Ok(skill)
    }

    pub fn get_cached(&self, id: &str) -> Option<&SkillDefinition> {
        self.cache.get(id)
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for SkillLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definition::SkillStep;

    #[test]
    fn test_loader_inline() {
        let mut loader = SkillLoader::new();

        let inline_skill = SkillDefinition {
            id: "test_skill".to_string(),
            description: "Test".to_string(),
            trigger: "When testing".to_string(),
            steps: vec![SkillStep::Prompt {
                prompt: "Hello".to_string(),
                llm: None,
            }],
            reasoning: None,
            reflection: None,
            disambiguation: None,
        };

        let skill_ref = SkillRef::Inline(inline_skill.clone());
        let loaded = loader.load_ref(&skill_ref).unwrap();

        assert_eq!(loaded.id, "test_skill");
        assert_eq!(loaded.steps.len(), 1);
    }

    #[test]
    fn test_loader_missing_skill() {
        let mut loader = SkillLoader::new();
        let result = loader.load_by_name("nonexistent_skill");
        assert!(result.is_err());
    }

    #[test]
    fn test_loader_cache() {
        let mut loader = SkillLoader::new();

        let inline_skill = SkillDefinition {
            id: "cached_skill".to_string(),
            description: "Cached".to_string(),
            trigger: "When cached".to_string(),
            steps: vec![],
            reasoning: None,
            reflection: None,
            disambiguation: None,
        };

        loader
            .cache
            .insert("cached_skill".to_string(), inline_skill);
        assert!(loader.get_cached("cached_skill").is_some());
        assert!(loader.get_cached("unknown").is_none());

        loader.clear_cache();
        assert!(loader.get_cached("cached_skill").is_none());
    }
}

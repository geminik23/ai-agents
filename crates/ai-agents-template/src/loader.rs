//! Template file loading

use ai_agents_core::{AgentError, Result};
use std::collections::HashMap;
use std::path::PathBuf;

/// Template loader for loading YAML files from filesystem
pub struct TemplateLoader {
    search_paths: Vec<PathBuf>,
    variables: HashMap<String, String>,
}

impl TemplateLoader {
    /// Create a new template loader
    pub fn new() -> Self {
        Self {
            search_paths: vec![PathBuf::from("templates")],
            variables: HashMap::new(),
        }
    }

    pub fn add_search_path(&mut self, path: impl Into<PathBuf>) -> &mut Self {
        self.search_paths.push(path.into());
        self
    }

    pub fn set_variable(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.variables.insert(key.into(), value.into());
        self
    }

    /// Set multiple variables at once
    pub fn set_variables(&mut self, vars: HashMap<String, String>) -> &mut Self {
        self.variables.extend(vars);
        self
    }

    pub fn get_variable(&self, key: &str) -> Option<&str> {
        self.variables.get(key).map(|s| s.as_str())
    }

    pub fn load_template(&self, name: &str) -> Result<String> {
        // Try to load from search paths
        for search_path in &self.search_paths {
            let mut path = search_path.clone();

            // Handle both "template.yaml" and "template" (add .yaml if missing)
            if !name.ends_with(".yaml") && !name.ends_with(".yml") {
                path.push(format!("{}.yaml", name));
            } else {
                path.push(name);
            }

            if path.exists() {
                let content = std::fs::read_to_string(&path).map_err(|e| {
                    AgentError::TemplateError(format!("Failed to read {}: {}", path.display(), e))
                })?;
                return Ok(content);
            }
        }

        Err(AgentError::TemplateError(format!(
            "Template '{}' not found in search paths: {:?}",
            name, self.search_paths
        )))
    }

    pub fn template_exists(&self, name: &str) -> bool {
        for search_path in &self.search_paths {
            let mut path = search_path.clone();
            if !name.ends_with(".yaml") && !name.ends_with(".yml") {
                path.push(format!("{}.yaml", name));
            } else {
                path.push(name);
            }
            if path.exists() {
                return true;
            }
        }
        false
    }

    pub fn search_paths(&self) -> &[PathBuf] {
        &self.search_paths
    }

    pub fn variables(&self) -> &HashMap<String, String> {
        &self.variables
    }
}

impl Default for TemplateLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    use tempfile::TempDir;

    fn create_test_template(dir: &Path, name: &str, content: &str) {
        let path = dir.join(name);
        fs::write(path, content).unwrap();
    }

    #[test]
    fn test_loader_new() {
        let loader = TemplateLoader::new();
        assert_eq!(loader.search_paths.len(), 1);
        assert!(loader.variables.is_empty());
    }

    #[test]
    fn test_set_variable() {
        let mut loader = TemplateLoader::new();
        loader.set_variable("key", "value");
        assert_eq!(loader.get_variable("key"), Some("value"));
    }

    #[test]
    fn test_set_variables() {
        let mut loader = TemplateLoader::new();
        let mut vars = HashMap::new();
        vars.insert("key1".to_string(), "value1".to_string());
        vars.insert("key2".to_string(), "value2".to_string());
        loader.set_variables(vars);
        assert_eq!(loader.get_variable("key1"), Some("value1"));
        assert_eq!(loader.get_variable("key2"), Some("value2"));
    }

    #[test]
    fn test_add_search_path() {
        let mut loader = TemplateLoader::new();
        loader.add_search_path("/custom/path");
        assert_eq!(loader.search_paths.len(), 2);
    }

    #[test]
    fn test_load_template() {
        let temp_dir = TempDir::new().unwrap();
        create_test_template(temp_dir.path(), "test.yaml", "name: TestAgent");

        let mut loader = TemplateLoader::new();
        loader.add_search_path(temp_dir.path());

        let content = loader.load_template("test.yaml").unwrap();
        assert_eq!(content, "name: TestAgent");
    }

    #[test]
    fn test_load_template_auto_extension() {
        let temp_dir = TempDir::new().unwrap();
        create_test_template(temp_dir.path(), "test.yaml", "name: TestAgent");

        let mut loader = TemplateLoader::new();
        loader.add_search_path(temp_dir.path());

        // Should work without .yaml extension
        let content = loader.load_template("test").unwrap();
        assert_eq!(content, "name: TestAgent");
    }

    #[test]
    fn test_load_template_not_found() {
        let loader = TemplateLoader::new();
        let result = loader.load_template("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_template_exists() {
        let temp_dir = TempDir::new().unwrap();
        create_test_template(temp_dir.path(), "exists.yaml", "test");

        let mut loader = TemplateLoader::new();
        loader.add_search_path(temp_dir.path());

        assert!(loader.template_exists("exists"));
        assert!(!loader.template_exists("notexists"));
    }

    #[test]
    fn test_search_paths_priority() {
        let temp_dir1 = TempDir::new().unwrap();
        let temp_dir2 = TempDir::new().unwrap();

        create_test_template(temp_dir1.path(), "test.yaml", "first");
        create_test_template(temp_dir2.path(), "test.yaml", "second");

        let mut loader = TemplateLoader::new();
        loader.add_search_path(temp_dir1.path());
        loader.add_search_path(temp_dir2.path());

        // Should load from first path
        let content = loader.load_template("test").unwrap();
        assert_eq!(content, "first");
    }
}

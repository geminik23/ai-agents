//! Build an AgentSpawner from a parsed SpawnerConfig and wire spawner tools.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ai_agents_core::{AgentError, AgentStorage, Result, Tool};
use ai_agents_llm::LLMRegistry;

use crate::AgentBuilder;
use crate::spec::{AgentSpec, SpawnerConfig, TemplateSource};

use super::registry::AgentRegistry;
use super::spawner::{AgentSpawner, ResolvedTemplate};
use super::tools::{GenerateAgentTool, ListAgentsTool, RemoveAgentTool, SendMessageTool};

/// Extract description and variable declarations from a raw template YAML string.
fn extract_template_metadata(raw: &str) -> (Option<String>, Option<HashMap<String, String>>) {
    // Parses the template as `serde_yaml::Value` (Jinja2 expressions are valid YAML strings)
    let val: serde_yaml::Value = match serde_yaml::from_str(raw) {
        Ok(v) => v,
        Err(_) => return (None, None),
    };

    // reads `description` and `metadata.template.variables`.
    // description: "..."
    let description = val
        .get("description")
        .and_then(|v| v.as_str())
        .map(String::from);

    // metadata.template.variables: { role: "...", personality: "..." }
    let variables = val
        .get("metadata")
        .and_then(|m| m.get("template"))
        .and_then(|t| t.get("variables"))
        .and_then(|v| v.as_mapping())
        .map(|mapping| {
            mapping
                .iter()
                .filter_map(|(k, v)| {
                    let key = k.as_str()?.to_string();
                    let val = v.as_str()?.to_string();
                    Some((key, val))
                })
                .collect()
        });

    (description, variables)
}

/// Resolve a `TemplateSource` map into `ResolvedTemplate` values by reading files and extracting metadata.
pub fn resolve_templates(
    templates: &HashMap<String, TemplateSource>,
    base_dir: Option<&Path>,
) -> Result<HashMap<String, ResolvedTemplate>> {
    let mut resolved = HashMap::with_capacity(templates.len());
    for (name, source) in templates {
        let content = match source {
            TemplateSource::Inline(s) => s.clone(),
            TemplateSource::File { path } => {
                let full_path = resolve_path(path, base_dir);
                std::fs::read_to_string(&full_path).map_err(|e| {
                    AgentError::Config(format!(
                        "Failed to read template '{}' from '{}': {}",
                        name,
                        full_path.display(),
                        e
                    ))
                })?
            }
        };

        let (description, variables) = extract_template_metadata(&content);

        resolved.insert(
            name.clone(),
            ResolvedTemplate {
                content,
                description,
                variables,
            },
        );
    }
    Ok(resolved)
}

/// Resolve a path string against a base directory. Absolute paths are used as-is.
fn resolve_path(path: &str, base_dir: Option<&Path>) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        // File paths are resolved against `base_dir` (typically the parent YAML's directory).
        // When `base_dir` is `None`, relative paths resolve against the current working directory.
        match base_dir {
            Some(dir) => dir.join(p),
            None => p.to_path_buf(),
        }
    }
}

/// Construct an `AgentSpawner` from the `spawner:` section of an AgentSpec.
pub fn spawner_from_config(
    config: &SpawnerConfig,
    llm_registry: Option<LLMRegistry>,
    storage: Option<Arc<dyn AgentStorage>>,
    base_dir: Option<&Path>,
) -> Result<AgentSpawner> {
    let mut spawner = AgentSpawner::new();

    // Shared LLMs: the YAML flag says "reuse parent's LLMs".
    if config.shared_llms {
        if let Some(reg) = llm_registry {
            spawner = spawner.with_shared_llms(reg);
        }
    }

    // Shared storage: caller resolves StorageConfig into Arc<dyn AgentStorage>.
    if let Some(st) = storage {
        spawner = spawner.with_shared_storage(st);
    }

    if !config.shared_context.is_empty() {
        spawner = spawner.with_shared_context_map(config.shared_context.clone());
    }

    if let Some(max) = config.max_agents {
        spawner = spawner.with_max_agents(max);
    }

    if let Some(ref prefix) = config.name_prefix {
        spawner = spawner.with_name_prefix(prefix.clone());
    }

    // Resolve file-path templates and extract metadata before storing.
    if !config.templates.is_empty() {
        let resolved = resolve_templates(&config.templates, base_dir)?;
        spawner = spawner.with_templates(resolved);
    }

    if let Some(ref allowed) = config.allowed_tools {
        spawner = spawner.with_allowed_tools(allowed.clone());
    }

    Ok(spawner)
}

/// Create the four spawner tools wired to the given spawner and registry.
pub fn configure_spawner_tools(
    spawner: Arc<AgentSpawner>,
    registry: Arc<AgentRegistry>,
    llm: Arc<LLMRegistry>,
    sender_id: impl Into<String>,
) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(GenerateAgentTool::new(
            Arc::clone(&spawner),
            Arc::clone(&registry),
            llm,
        )),
        Arc::new(SendMessageTool::new(Arc::clone(&registry), sender_id)),
        Arc::new(ListAgentsTool::new(Arc::clone(&registry))),
        Arc::new(RemoveAgentTool::new(Arc::clone(&registry)).with_spawner(spawner)),
    ]
}

/// Wire spawner tools into an AgentBuilder when the spec has a `spawner:` section.
/// Call after `auto_configure_llms()` and `auto_configure_features()`.
pub async fn auto_configure_spawner(
    mut builder: AgentBuilder,
    spec: &AgentSpec,
    llm_registry: Option<&LLMRegistry>,
    base_dir: Option<&Path>,
) -> Result<(
    AgentBuilder,
    Option<(Arc<AgentSpawner>, Arc<AgentRegistry>)>,
)> {
    let spawner_config = match spec.spawner {
        Some(ref c) => c,
        None => return Ok((builder, None)),
    };

    let mut spawner = AgentSpawner::new();

    if spawner_config.shared_llms {
        if let Some(reg) = llm_registry {
            spawner = spawner.with_shared_llms(reg.clone());
        }
    }

    if !spawner_config.shared_context.is_empty() {
        spawner = spawner.with_shared_context_map(spawner_config.shared_context.clone());
    }

    if let Some(max) = spawner_config.max_agents {
        spawner = spawner.with_max_agents(max);
    }

    if let Some(ref prefix) = spawner_config.name_prefix {
        spawner = spawner.with_name_prefix(prefix.clone());
    }

    // Resolve file-path templates and extract metadata before storing.
    if !spawner_config.templates.is_empty() {
        let resolved = resolve_templates(&spawner_config.templates, base_dir)?;
        spawner = spawner.with_templates(resolved);
    }

    if let Some(ref allowed) = spawner_config.allowed_tools {
        spawner = spawner.with_allowed_tools(allowed.clone());
    }

    // Resolve shared storage from YAML config into a live backend.
    if let Some(ref sc) = spawner_config.shared_storage {
        let converted = crate::spec::storage::to_storage_config(sc);
        if let Some(st) = ai_agents_storage::create_storage(&converted).await? {
            spawner = spawner.with_shared_storage(st);
        }
    }

    let spawner = Arc::new(spawner);
    let registry = Arc::new(AgentRegistry::new());

    let llm_for_tools = Arc::new(llm_registry.cloned().unwrap_or_default());

    let tools = configure_spawner_tools(
        Arc::clone(&spawner),
        Arc::clone(&registry),
        llm_for_tools,
        &spec.name,
    );

    for tool in tools {
        builder = builder.tool(tool);
    }

    Ok((builder, Some((spawner, registry))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::OrchestrationToolsConfig;

    // -- extract_template_metadata tests --

    #[test]
    fn test_extract_metadata_full() {
        let raw = r#"
name: "{{ name }}"
description: "A test NPC template"
metadata:
  template:
    variables:
      role: "NPC occupation"
      personality: "Personality description"
system_prompt: "You are {{ name }}."
"#;
        let (desc, vars) = extract_template_metadata(raw);
        assert_eq!(desc.as_deref(), Some("A test NPC template"));
        let vars = vars.unwrap();
        assert_eq!(vars.get("role").unwrap(), "NPC occupation");
        assert_eq!(vars.get("personality").unwrap(), "Personality description");
    }

    #[test]
    fn test_extract_metadata_no_metadata() {
        let raw = "name: \"{{ name }}\"\nsystem_prompt: \"hello\"";
        let (desc, vars) = extract_template_metadata(raw);
        assert!(desc.is_none());
        assert!(vars.is_none());
    }

    #[test]
    fn test_extract_metadata_description_only() {
        let raw = "name: test\ndescription: \"Just a description\"\nsystem_prompt: hi";
        let (desc, vars) = extract_template_metadata(raw);
        assert_eq!(desc.as_deref(), Some("Just a description"));
        assert!(vars.is_none());
    }

    #[test]
    fn test_extract_metadata_variables_only() {
        let raw = r#"
name: "{{ name }}"
metadata:
  template:
    variables:
      department: "Support department"
system_prompt: "You work in {{ department }}."
"#;
        let (desc, vars) = extract_template_metadata(raw);
        assert!(desc.is_none());
        let vars = vars.unwrap();
        assert_eq!(vars.len(), 1);
        assert_eq!(vars.get("department").unwrap(), "Support department");
    }

    #[test]
    fn test_extract_metadata_invalid_yaml() {
        let raw = "{{{{ totally broken yaml";
        let (desc, vars) = extract_template_metadata(raw);
        assert!(desc.is_none());
        assert!(vars.is_none());
    }

    #[test]
    fn test_extract_metadata_with_jinja2_expressions() {
        let raw = r#"
name: "{{ name }}"
description: "NPC template"
metadata:
  template:
    variables:
      role: "occupation"
system_prompt: "You are {{ name }}, a {{ role }}."
"#;
        let (desc, vars) = extract_template_metadata(raw);
        assert_eq!(desc.as_deref(), Some("NPC template"));
        assert!(vars.is_some());
        assert_eq!(vars.unwrap().get("role").unwrap(), "occupation");
    }

    #[test]
    fn test_resolve_templates_inline_extracts_metadata() {
        let mut templates = HashMap::new();
        templates.insert(
            "agent".to_string(),
            TemplateSource::Inline(
                "name: test\ndescription: \"Inline agent\"\nsystem_prompt: hi".to_string(),
            ),
        );
        let resolved = resolve_templates(&templates, None).unwrap();
        let tpl = resolved.get("agent").unwrap();
        assert_eq!(tpl.description.as_deref(), Some("Inline agent"));
        assert!(tpl.content.contains("name: test"));
    }

    #[test]
    fn test_resolve_templates_no_metadata_backward_compat() {
        let mut templates = HashMap::new();
        templates.insert(
            "bare".to_string(),
            TemplateSource::Inline("name: test\nsystem_prompt: hi".to_string()),
        );
        let resolved = resolve_templates(&templates, None).unwrap();
        let tpl = resolved.get("bare").unwrap();
        assert!(tpl.description.is_none());
        assert!(tpl.variables.is_none());
        assert!(tpl.content.contains("name: test"));
    }

    #[test]
    fn test_resolve_templates_file_extracts_metadata() {
        let dir = std::env::temp_dir().join("ai_agents_test_tpl_meta");
        let _ = std::fs::create_dir_all(&dir);
        let tpl = dir.join("npc.yaml");
        std::fs::write(
            &tpl,
            r#"
name: "{{ name }}"
description: "Test NPC"
metadata:
  template:
    variables:
      role: "occupation"
system_prompt: "You are {{ name }}."
"#,
        )
        .unwrap();

        let mut templates = HashMap::new();
        templates.insert(
            "npc".to_string(),
            TemplateSource::File {
                path: "npc.yaml".to_string(),
            },
        );
        let resolved = resolve_templates(&templates, Some(&dir)).unwrap();
        let rt = resolved.get("npc").unwrap();
        assert_eq!(rt.description.as_deref(), Some("Test NPC"));
        assert_eq!(
            rt.variables.as_ref().unwrap().get("role").unwrap(),
            "occupation"
        );
        assert!(rt.content.contains("{{ name }}"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_resolve_templates_file_path() {
        let dir = std::env::temp_dir().join("ai_agents_test_resolve_tpl_24");
        let _ = std::fs::create_dir_all(&dir);
        let tpl_path = dir.join("npc.yaml");
        std::fs::write(&tpl_path, "name: {{ name }}\nsystem_prompt: hi").unwrap();

        let mut templates = HashMap::new();
        templates.insert(
            "npc".to_string(),
            TemplateSource::File {
                path: "npc.yaml".to_string(),
            },
        );
        let resolved = resolve_templates(&templates, Some(&dir)).unwrap();
        assert!(resolved.get("npc").unwrap().content.contains("{{ name }}"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_resolve_templates_absolute_path() {
        let dir = std::env::temp_dir().join("ai_agents_test_resolve_abs_24");
        let _ = std::fs::create_dir_all(&dir);
        let tpl_path = dir.join("guard.yaml");
        std::fs::write(&tpl_path, "name: Guard\nsystem_prompt: hi").unwrap();

        let mut templates = HashMap::new();
        templates.insert(
            "guard".to_string(),
            TemplateSource::File {
                path: tpl_path.to_str().unwrap().to_string(),
            },
        );
        let resolved = resolve_templates(&templates, None).unwrap();
        assert!(
            resolved
                .get("guard")
                .unwrap()
                .content
                .contains("name: Guard")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_resolve_templates_missing_file_errors() {
        let mut templates = HashMap::new();
        templates.insert(
            "missing".to_string(),
            TemplateSource::File {
                path: "./nonexistent_template_abc123.yaml".to_string(),
            },
        );
        let result = resolve_templates(&templates, None);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("missing"));
        assert!(err.contains("nonexistent_template_abc123.yaml"));
    }

    #[test]
    fn test_resolve_templates_mixed() {
        let dir = std::env::temp_dir().join("ai_agents_test_resolve_mixed_24");
        let _ = std::fs::create_dir_all(&dir);
        let tpl_path = dir.join("from_file.yaml");
        std::fs::write(&tpl_path, "name: FromFile\nsystem_prompt: hi").unwrap();

        let mut templates = HashMap::new();
        templates.insert(
            "inline".to_string(),
            TemplateSource::Inline("name: Inline\nsystem_prompt: hi".to_string()),
        );
        templates.insert(
            "file".to_string(),
            TemplateSource::File {
                path: "from_file.yaml".to_string(),
            },
        );

        let resolved = resolve_templates(&templates, Some(&dir)).unwrap();
        assert!(resolved.get("inline").unwrap().content.contains("Inline"));
        assert!(resolved.get("file").unwrap().content.contains("FromFile"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_resolve_path_relative_with_base() {
        let base = Path::new("/home/user/agents");
        let result = resolve_path("templates/npc.yaml", Some(base));
        assert_eq!(
            result,
            PathBuf::from("/home/user/agents/templates/npc.yaml")
        );
    }

    #[test]
    fn test_resolve_path_relative_dot_slash() {
        let base = Path::new("/home/user/agents");
        let result = resolve_path("./templates/npc.yaml", Some(base));
        assert_eq!(
            result,
            PathBuf::from("/home/user/agents/./templates/npc.yaml")
        );
    }

    #[test]
    fn test_resolve_path_absolute_ignores_base() {
        let base = Path::new("/home/user/agents");
        let result = resolve_path("/opt/templates/npc.yaml", Some(base));
        assert_eq!(result, PathBuf::from("/opt/templates/npc.yaml"));
    }

    #[test]
    fn test_resolve_path_no_base_returns_raw() {
        let result = resolve_path("templates/npc.yaml", None);
        assert_eq!(result, PathBuf::from("templates/npc.yaml"));
    }

    // -- spawner_from_config tests --

    #[test]
    fn test_spawner_from_empty_config() {
        let config = SpawnerConfig::default();
        let spawner = spawner_from_config(&config, None, None, None).unwrap();
        assert_eq!(spawner.spawned_count(), 0);
    }

    #[test]
    fn test_spawner_from_populated_config() {
        let mut templates = HashMap::new();
        templates.insert(
            "base".to_string(),
            TemplateSource::Inline(
                "name: {{ name }}\ndescription: \"Base\"\nsystem_prompt: hello".to_string(),
            ),
        );

        let mut shared_ctx = HashMap::new();
        shared_ctx.insert("world".to_string(), serde_json::json!("Fantasy"));

        let config = SpawnerConfig {
            shared_llms: false,
            shared_storage: None,
            shared_context: shared_ctx,
            max_agents: Some(10),
            name_prefix: Some("npc_".to_string()),
            templates,
            allowed_tools: Some(vec!["echo".to_string()]),
            auto_spawn: Vec::new(),
            orchestration_tools: OrchestrationToolsConfig::default(),
        };

        let spawner = spawner_from_config(&config, None, None, None).unwrap();
        let tpl = spawner.templates().get("base").unwrap();
        assert!(tpl.content.contains("{{ name }}"));
        assert_eq!(tpl.description.as_deref(), Some("Base"));
    }

    #[test]
    fn test_spawner_from_config_with_storage() {
        let storage: Arc<dyn AgentStorage> =
            Arc::new(ai_agents_storage::FileStorage::new("/tmp/test_spawner_cfg"));
        let config = SpawnerConfig {
            shared_storage: Some(crate::spec::StorageConfig::sqlite("./test.db")),
            ..Default::default()
        };
        let spawner = spawner_from_config(&config, None, Some(storage), None).unwrap();
        assert!(spawner.shared_storage().is_some());
    }

    #[test]
    fn test_spawner_from_config_without_storage() {
        let config = SpawnerConfig::default();
        let spawner = spawner_from_config(&config, None, None, None).unwrap();
        assert!(spawner.shared_storage().is_none());
    }
}

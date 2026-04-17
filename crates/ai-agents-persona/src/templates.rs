//! Persona template registry for reusable persona components.

use std::collections::HashMap;

use serde_json::Value;

use ai_agents_core::{AgentError, Result};

use ai_agents_core::set_dot_path;

use crate::config::PersonaConfig;

/// Registry for reusable persona-only templates. Single-level inheritance only.
pub struct PersonaTemplateRegistry {
    templates: HashMap<String, PersonaConfig>,
}

impl PersonaTemplateRegistry {
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    /// Register a named persona template.
    pub fn register(&mut self, name: &str, template: PersonaConfig) {
        self.templates.insert(name.to_string(), template);
    }

    /// Resolve a template by name, applying optional dot-notation overrides.
    pub fn resolve(
        &self,
        name: &str,
        overrides: Option<&HashMap<String, Value>>,
    ) -> Result<PersonaConfig> {
        let base = self
            .templates
            .get(name)
            .ok_or_else(|| AgentError::Config(format!("Persona template '{}' not found", name)))?;

        let mut merged = base.clone();
        // Strip any templates ref from the resolved config to prevent recursive resolution.
        merged.templates = None;

        if let Some(overrides) = overrides {
            let mut serialized = serde_json::to_value(&merged).map_err(|e| {
                AgentError::Config(format!("Failed to serialize persona template: {}", e))
            })?;

            for (path, value) in overrides {
                serialized = set_dot_path(serialized, path, value.clone())?;
            }

            merged = serde_json::from_value(serialized).map_err(|e| {
                AgentError::Config(format!("Failed to apply template overrides: {}", e))
            })?;
        }

        Ok(merged)
    }

    /// Deep-merge an instance config on top of a resolved template. Instance fields take priority.
    pub fn merge_with_instance(
        base: &PersonaConfig,
        instance: &PersonaConfig,
    ) -> Result<PersonaConfig> {
        let base_value = serde_json::to_value(base)
            .map_err(|e| AgentError::Config(format!("Failed to serialize base config: {}", e)))?;
        let instance_value = serde_json::to_value(instance).map_err(|e| {
            AgentError::Config(format!("Failed to serialize instance config: {}", e))
        })?;

        let merged_value = deep_merge_values(base_value, instance_value);

        let mut merged: PersonaConfig = serde_json::from_value(merged_value).map_err(|e| {
            AgentError::Config(format!("Failed to deserialize merged config: {}", e))
        })?;
        // Remove template ref from the final merged config.
        merged.templates = None;

        Ok(merged)
    }

    /// Check if a template name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.templates.contains_key(name)
    }

    /// List all registered template names.
    pub fn list(&self) -> Vec<&str> {
        self.templates.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for PersonaTemplateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Deep-merge two JSON values. Override wins for leaves; null in override does not overwrite base.
fn deep_merge_values(base: Value, override_val: Value) -> Value {
    match (base, override_val) {
        (Value::Object(mut base_map), Value::Object(override_map)) => {
            for (key, override_v) in override_map {
                if override_v.is_null() {
                    continue;
                }
                let merged = if let Some(base_v) = base_map.remove(&key) {
                    deep_merge_values(base_v, override_v)
                } else {
                    override_v
                };
                base_map.insert(key, merged);
            }
            Value::Object(base_map)
        }
        // For non-object types, override wins (unless null).
        (_base, override_val) if !override_val.is_null() => override_val,
        (base, _) => base,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use serde_json::json;

    fn make_guard_template() -> PersonaConfig {
        PersonaConfig {
            identity: Some(PersonaIdentity {
                name: "Guard".into(),
                role: "Guard".into(),
                description: Some("A generic guard".into()),
                backstory: None,
                affiliation: Some("City Watch".into()),
            }),
            traits: Some(PersonaTraits {
                personality: vec!["disciplined".into(), "observant".into()],
                values: None,
                fears: None,
                speaking_style: Some("formal, clipped sentences".into()),
            }),
            goals: Some(PersonaGoals {
                primary: vec!["protect_area".into(), "maintain_order".into()],
                hidden: None,
            }),
            secrets: None,
            evolution: None,
            templates: None,
            max_prompt_tokens: None,
        }
    }

    fn make_merchant_template() -> PersonaConfig {
        PersonaConfig {
            identity: Some(PersonaIdentity {
                name: "Merchant".into(),
                role: "Merchant".into(),
                description: None,
                backstory: None,
                affiliation: None,
            }),
            traits: Some(PersonaTraits {
                personality: vec!["friendly".into(), "profit-motivated".into()],
                values: Some(vec!["commerce".into(), "fairness".into()]),
                fears: None,
                speaking_style: Some("warm and persuasive".into()),
            }),
            goals: None,
            secrets: None,
            evolution: None,
            templates: None,
            max_prompt_tokens: None,
        }
    }

    #[test]
    fn test_register_and_contains() {
        let mut registry = PersonaTemplateRegistry::new();
        assert!(!registry.contains("guard_base"));

        registry.register("guard_base", make_guard_template());
        assert!(registry.contains("guard_base"));
        assert!(!registry.contains("merchant_base"));
    }

    #[test]
    fn test_list_templates() {
        let mut registry = PersonaTemplateRegistry::new();
        registry.register("guard", make_guard_template());
        registry.register("merchant", make_merchant_template());

        let mut names = registry.list();
        names.sort();
        assert_eq!(names, vec!["guard", "merchant"]);
    }

    #[test]
    fn test_resolve_no_overrides() {
        let mut registry = PersonaTemplateRegistry::new();
        registry.register("guard", make_guard_template());

        let resolved = registry.resolve("guard", None).unwrap();
        assert_eq!(resolved.identity.as_ref().unwrap().name, "Guard");
        assert_eq!(resolved.identity.as_ref().unwrap().role, "Guard");
        assert_eq!(
            resolved.identity.as_ref().unwrap().affiliation.as_deref(),
            Some("City Watch")
        );
        // templates ref should be stripped.
        assert!(resolved.templates.is_none());
    }

    #[test]
    fn test_resolve_with_overrides() {
        let mut registry = PersonaTemplateRegistry::new();
        registry.register("guard", make_guard_template());

        let mut overrides = HashMap::new();
        overrides.insert("identity.name".to_string(), json!("Captain Tam"));
        overrides.insert(
            "identity.backstory".to_string(),
            json!("A former sailor turned guard."),
        );

        let resolved = registry.resolve("guard", Some(&overrides)).unwrap();
        assert_eq!(resolved.identity.as_ref().unwrap().name, "Captain Tam");
        assert_eq!(
            resolved.identity.as_ref().unwrap().backstory.as_deref(),
            Some("A former sailor turned guard.")
        );
        // Non-overridden fields preserved.
        assert_eq!(resolved.identity.as_ref().unwrap().role, "Guard");
        assert_eq!(
            resolved.identity.as_ref().unwrap().affiliation.as_deref(),
            Some("City Watch")
        );
    }

    #[test]
    fn test_resolve_missing_template() {
        let registry = PersonaTemplateRegistry::new();
        let result = registry.resolve("nonexistent", None);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Persona template 'nonexistent' not found")
        );
    }

    #[test]
    fn test_resolve_strips_nested_templates() {
        let mut template = make_guard_template();
        template.templates = Some(PersonaTemplateRef {
            base: "should_be_ignored".into(),
            overrides: None,
        });

        let mut registry = PersonaTemplateRegistry::new();
        registry.register("nested", template);

        let resolved = registry.resolve("nested", None).unwrap();
        assert!(resolved.templates.is_none());
    }

    #[test]
    fn test_merge_with_instance_identity_override() {
        let base = make_guard_template();
        let instance = PersonaConfig {
            identity: Some(PersonaIdentity {
                name: "Captain Elira".into(),
                role: "Harbor Guard Captain".into(),
                description: None,
                backstory: Some("A veteran soldier.".into()),
                affiliation: None,
            }),
            traits: None,
            goals: None,
            secrets: None,
            evolution: None,
            templates: None,
            max_prompt_tokens: None,
        };

        let merged = PersonaTemplateRegistry::merge_with_instance(&base, &instance).unwrap();

        // Instance values take priority.
        assert_eq!(merged.identity.as_ref().unwrap().name, "Captain Elira");
        assert_eq!(
            merged.identity.as_ref().unwrap().role,
            "Harbor Guard Captain"
        );
        assert_eq!(
            merged.identity.as_ref().unwrap().backstory.as_deref(),
            Some("A veteran soldier.")
        );
        // Base values fill in what instance doesn't provide.
        // Note: affiliation was Some("City Watch") in base but None in instance.
        // Since the instance has an identity object, its fields take priority.
        // But because the merge is object-level deep, and affiliation is null in instance,
        // the base value is preserved.
        assert_eq!(
            merged.identity.as_ref().unwrap().affiliation.as_deref(),
            Some("City Watch")
        );
        // Traits come entirely from base since instance has none.
        assert!(merged.traits.is_some());
        assert_eq!(
            merged.traits.as_ref().unwrap().personality,
            vec!["disciplined".to_string(), "observant".to_string()]
        );
        // Goals from base.
        assert!(merged.goals.is_some());
    }

    #[test]
    fn test_merge_with_instance_traits_override() {
        let base = make_merchant_template();
        let instance = PersonaConfig {
            identity: None,
            traits: Some(PersonaTraits {
                personality: vec!["shrewd".into(), "cunning".into()],
                values: None,
                fears: None,
                speaking_style: None,
            }),
            goals: Some(PersonaGoals {
                primary: vec!["maximize_profit".into()],
                hidden: None,
            }),
            secrets: None,
            evolution: None,
            templates: None,
            max_prompt_tokens: None,
        };

        let merged = PersonaTemplateRegistry::merge_with_instance(&base, &instance).unwrap();

        // Instance personality overrides base.
        assert_eq!(
            merged.traits.as_ref().unwrap().personality,
            vec!["shrewd".to_string(), "cunning".to_string()]
        );
        // Base speaking_style preserved (instance was None/null).
        assert_eq!(
            merged.traits.as_ref().unwrap().speaking_style.as_deref(),
            Some("warm and persuasive")
        );
        // Instance goals take priority.
        assert_eq!(
            merged.goals.as_ref().unwrap().primary,
            vec!["maximize_profit".to_string()]
        );
        // Identity from base.
        assert_eq!(merged.identity.as_ref().unwrap().name, "Merchant");
    }

    #[test]
    fn test_merge_strips_templates_ref() {
        let base = make_guard_template();
        let mut instance = PersonaConfig::default();
        instance.templates = Some(PersonaTemplateRef {
            base: "something".into(),
            overrides: None,
        });

        let merged = PersonaTemplateRegistry::merge_with_instance(&base, &instance).unwrap();
        assert!(merged.templates.is_none());
    }

    #[test]
    fn test_deep_merge_values_basic() {
        let base = json!({"a": 1, "b": 2});
        let over = json!({"b": 3, "c": 4});
        let merged = deep_merge_values(base, over);
        assert_eq!(merged, json!({"a": 1, "b": 3, "c": 4}));
    }

    #[test]
    fn test_deep_merge_values_nested() {
        let base = json!({"x": {"a": 1, "b": 2}, "y": 10});
        let over = json!({"x": {"b": 99, "c": 3}});
        let merged = deep_merge_values(base, over);
        assert_eq!(merged, json!({"x": {"a": 1, "b": 99, "c": 3}, "y": 10}));
    }

    #[test]
    fn test_deep_merge_values_null_does_not_overwrite() {
        let base = json!({"a": 1, "b": 2});
        let over = json!({"a": null, "c": 3});
        let merged = deep_merge_values(base, over);
        assert_eq!(merged, json!({"a": 1, "b": 2, "c": 3}));
    }

    #[test]
    fn test_deep_merge_values_array_replaces() {
        let base = json!({"items": [1, 2]});
        let over = json!({"items": [3, 4, 5]});
        let merged = deep_merge_values(base, over);
        // Arrays are replaced, not merged element-wise.
        assert_eq!(merged, json!({"items": [3, 4, 5]}));
    }

    #[test]
    fn test_default_registry() {
        let registry = PersonaTemplateRegistry::default();
        assert!(registry.list().is_empty());
    }

    #[test]
    fn test_resolve_override_traits() {
        let mut registry = PersonaTemplateRegistry::new();
        registry.register("guard", make_guard_template());

        let mut overrides = HashMap::new();
        overrides.insert(
            "traits.personality".to_string(),
            json!(["brave", "reckless"]),
        );

        let resolved = registry.resolve("guard", Some(&overrides)).unwrap();
        assert_eq!(
            resolved.traits.as_ref().unwrap().personality,
            vec!["brave".to_string(), "reckless".to_string()]
        );
        // Other trait fields preserved.
        assert_eq!(
            resolved.traits.as_ref().unwrap().speaking_style.as_deref(),
            Some("formal, clipped sentences")
        );
    }

    #[test]
    fn test_register_overwrites_existing() {
        let mut registry = PersonaTemplateRegistry::new();
        registry.register("guard", make_guard_template());
        registry.register("guard", make_merchant_template());

        let resolved = registry.resolve("guard", None).unwrap();
        assert_eq!(resolved.identity.as_ref().unwrap().name, "Merchant");
    }
}

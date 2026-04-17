//! Runtime persona manager with RwLock interior mutability.
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use serde_json::Value;

use ai_agents_context::TemplateRenderer;
use ai_agents_core::{AgentError, Result};

use crate::conditions::evaluate_secrets;
use crate::config::{EvolutionConfig, PersonaConfig, PersonaIdentity, PersonaTraits};
use crate::evolution::{PersonaChange, evolve_field, validate_mutable_fields};
use crate::prompt;
use crate::snapshot::PersonaSnapshot;
use crate::templates::PersonaTemplateRegistry;

/// Return type for render_prompt(). Bundles prompt text with newly revealed secrets for hook firing.
#[derive(Debug, Clone)]
pub struct PersonaRenderResult {
    /// Rendered persona prompt text to prepend to the system prompt.
    pub prompt: String,

    /// Secrets revealed for the first time in this render call.
    pub newly_revealed: Vec<String>,
}

/// Runtime persona manager held by RuntimeAgent as Arc<PersonaManager>.
pub struct PersonaManager {
    /// Current persona state.
    config: RwLock<PersonaConfig>,

    /// Change history - populated when evolution.track_changes is enabled.
    history: RwLock<Vec<PersonaChange>>,

    /// Set of secret indices revealed on previous checks.
    revealed_indices: RwLock<HashSet<usize>>,

    /// Template renderer for Jinja2 in persona fields.
    renderer: TemplateRenderer,
}

impl PersonaManager {
    /// Create from config, resolving templates and validating identity and mutable_fields.
    pub fn from_config(
        config: PersonaConfig,
        registry: Option<Arc<PersonaTemplateRegistry>>,
        renderer: TemplateRenderer,
    ) -> Result<Self> {
        let resolved = if let Some(ref tmpl_ref) = config.templates {
            let reg = registry.as_ref().ok_or_else(|| {
                AgentError::Config(format!(
                    "Persona references template '{}' but no PersonaTemplateRegistry was provided",
                    tmpl_ref.base
                ))
            })?;

            let base_resolved = reg.resolve(&tmpl_ref.base, tmpl_ref.overrides.as_ref())?;

            // Deep-merge instance config on top of resolved template.
            PersonaTemplateRegistry::merge_with_instance(&base_resolved, &config)?
        } else {
            config
        };

        // Validate identity is present after resolution.
        if resolved.identity.is_none() {
            return Err(AgentError::Config(
                "Persona config must have identity after template resolution. \
                 Provide identity directly or via a template."
                    .into(),
            ));
        }

        // Validate mutable_fields if evolution is configured.
        if let Some(ref evo) = resolved.evolution {
            validate_mutable_fields(&evo.mutable_fields)?;
        }

        Ok(Self {
            config: RwLock::new(resolved),
            history: RwLock::new(Vec::new()),
            revealed_indices: RwLock::new(HashSet::new()),
            renderer,
        })
    }

    /// Render the persona prompt section and evaluate secret conditions.
    /// Returns prompt text and any secrets revealed for the first time.
    pub fn render_prompt(&self, context: &HashMap<String, Value>) -> Result<PersonaRenderResult> {
        let config = self
            .config
            .read()
            .map_err(|e| AgentError::Config(format!("Persona config lock poisoned: {}", e)))?;

        // Render the base prompt (identity + traits + goals).
        let base_prompt = prompt::render_full_prompt(&config, context, &self.renderer)?;

        // Evaluate secrets and find newly revealed ones.
        let (revealed_contents, newly_revealed) = self.evaluate_and_track_secrets(&config, context);

        // Append secrets section if any are revealed.
        let secrets_section = prompt::render_secrets_section(&revealed_contents);
        let full_prompt = if secrets_section.is_empty() {
            base_prompt.clone()
        } else {
            format!("{}\n\n{}", base_prompt, secrets_section)
        };

        // Check token budget and use condensed format if needed.
        let final_prompt = if let Some(max_tokens) = config.max_prompt_tokens {
            let estimated = prompt::estimate_tokens(&full_prompt);
            if estimated > max_tokens {
                let condensed = prompt::render_condensed_prompt(&config, context, &self.renderer)?;
                condensed
            } else {
                full_prompt
            }
        } else {
            full_prompt
        };

        Ok(PersonaRenderResult {
            prompt: final_prompt,
            newly_revealed,
        })
    }

    /// Get structured identity for programmatic access.
    pub fn identity(&self) -> PersonaIdentity {
        let config = self.config.read().expect("Persona config lock poisoned");
        config
            .identity
            .clone()
            .expect("PersonaManager always has identity after construction")
    }

    /// Get current traits (if defined).
    pub fn traits(&self) -> Option<PersonaTraits> {
        let config = self.config.read().expect("Persona config lock poisoned");
        config.traits.clone()
    }

    /// Get the current persona config (cloned).
    pub fn config(&self) -> PersonaConfig {
        let config = self.config.read().expect("Persona config lock poisoned");
        config.clone()
    }

    /// Mutate a persona field. Returns error if evolution is disabled or field is not mutable.
    pub fn evolve(
        &self,
        field: &str,
        new_value: Value,
        reason: Option<&str>,
    ) -> Result<PersonaChange> {
        let mut config = self
            .config
            .write()
            .map_err(|e| AgentError::Config(format!("Persona config lock poisoned: {}", e)))?;

        let evolution = config.evolution.clone().unwrap_or_default();

        let (new_config, change) = evolve_field(&config, &evolution, field, new_value, reason)?;

        // Record change in history if tracking is enabled.
        if evolution.track_changes {
            let mut history = self
                .history
                .write()
                .map_err(|e| AgentError::Config(format!("Persona history lock poisoned: {}", e)))?;
            history.push(change.clone());
        }

        *config = new_config;

        Ok(change)
    }

    /// Get evolution history.
    pub fn history(&self) -> Vec<PersonaChange> {
        let history = self.history.read().expect("Persona history lock poisoned");
        history.clone()
    }

    /// Return content strings of all secrets whose conditions are currently satisfied.
    pub fn revealed_secrets(&self, context: &HashMap<String, Value>) -> Vec<String> {
        let config = self.config.read().expect("Persona config lock poisoned");
        let secrets = match config.secrets.as_ref() {
            Some(s) if !s.is_empty() => s,
            _ => return vec![],
        };

        let revealed_indices = evaluate_secrets(secrets, context);
        revealed_indices
            .into_iter()
            .map(|i| secrets[i].content.clone())
            .collect()
    }

    /// Returns true when PersonaEvolveTool should be auto-registered.
    pub fn should_register_evolve_tool(&self) -> bool {
        let config = self.config.read().expect("Persona config lock poisoned");
        config
            .evolution
            .as_ref()
            .map(|e| e.enabled && e.allow_llm_evolve)
            .unwrap_or(false)
    }

    /// Get the evolution config (if any).
    pub fn evolution_config(&self) -> Option<EvolutionConfig> {
        let config = self.config.read().expect("Persona config lock poisoned");
        config.evolution.clone()
    }

    /// Create a serializable snapshot for persistence.
    pub fn snapshot(&self) -> PersonaSnapshot {
        let config = self.config.read().expect("Persona config lock poisoned");
        let history = self.history.read().expect("Persona history lock poisoned");
        let revealed = self
            .revealed_indices
            .read()
            .expect("Persona revealed lock poisoned");

        PersonaSnapshot::new(config.clone(), history.clone(), revealed.clone())
    }

    /// Restore state from a snapshot, replacing internal state via RwLock.
    pub fn restore(&self, snapshot: PersonaSnapshot) {
        let mut config = self.config.write().expect("Persona config lock poisoned");
        let mut history = self.history.write().expect("Persona history lock poisoned");
        let mut revealed = self
            .revealed_indices
            .write()
            .expect("Persona revealed lock poisoned");

        *config = snapshot.config;
        *history = snapshot.history;
        *revealed = snapshot.revealed_indices;
    }

    /// Restore from a serde_json::Value stored in AgentSnapshot.
    pub fn restore_from_value(&self, value: Value) -> Result<()> {
        let snapshot = PersonaSnapshot::from_value(value)?;
        self.restore(snapshot);
        Ok(())
    }

    /// Serialize snapshot to serde_json::Value for AgentSnapshot storage.
    pub fn snapshot_as_value(&self) -> Result<Value> {
        self.snapshot().to_value()
    }

    /// Evaluate secrets against context and return (all revealed, newly revealed) content strings.
    fn evaluate_and_track_secrets(
        &self,
        config: &PersonaConfig,
        context: &HashMap<String, Value>,
    ) -> (Vec<String>, Vec<String>) {
        let secrets = match config.secrets.as_ref() {
            Some(s) if !s.is_empty() => s,
            _ => return (vec![], vec![]),
        };

        let current_revealed: HashSet<usize> =
            evaluate_secrets(secrets, context).into_iter().collect();

        let mut prev_revealed = self
            .revealed_indices
            .write()
            .expect("Persona revealed lock poisoned");

        let newly_revealed: Vec<usize> = current_revealed
            .iter()
            .filter(|i| !prev_revealed.contains(i))
            .copied()
            .collect();

        // Update the tracked set with all currently revealed indices.
        for &idx in &current_revealed {
            prev_revealed.insert(idx);
        }

        let all_contents: Vec<String> = current_revealed
            .iter()
            .map(|&i| secrets[i].content.clone())
            .collect();

        let new_contents: Vec<String> = newly_revealed
            .iter()
            .map(|&i| secrets[i].content.clone())
            .collect();

        (all_contents, new_contents)
    }
}

impl std::fmt::Debug for PersonaManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let config = self.config.read().ok();
        let identity_name = config
            .as_ref()
            .and_then(|c| c.identity.as_ref())
            .map(|id| id.name.as_str())
            .unwrap_or("(locked)");

        f.debug_struct("PersonaManager")
            .field("identity_name", &identity_name)
            .field(
                "has_evolution",
                &config
                    .as_ref()
                    .and_then(|c| c.evolution.as_ref())
                    .map(|e| e.enabled)
                    .unwrap_or(false),
            )
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use serde_json::json;

    fn test_renderer() -> TemplateRenderer {
        TemplateRenderer::new()
    }

    fn make_basic_config() -> PersonaConfig {
        PersonaConfig {
            identity: Some(PersonaIdentity {
                name: "Captain Elira".into(),
                role: "Harbor Guard Captain".into(),
                description: Some("A disciplined former soldier".into()),
                backstory: Some("Former soldier who served in the Eastern Campaign.".into()),
                affiliation: Some("Harbor Watch".into()),
            }),
            traits: Some(PersonaTraits {
                personality: vec!["disciplined".into(), "suspicious".into(), "loyal".into()],
                values: Some(vec!["duty".into(), "order".into()]),
                fears: Some(vec!["betrayal".into()]),
                speaking_style: Some("formal military cadence".into()),
            }),
            goals: Some(PersonaGoals {
                primary: vec!["protect_harbor".into(), "investigate_smuggling".into()],
                hidden: Some(vec!["Find the spy".into()]),
            }),
            secrets: None,
            evolution: None,
            templates: None,
            max_prompt_tokens: None,
        }
    }

    fn make_evolvable_config() -> PersonaConfig {
        let mut config = make_basic_config();
        config.evolution = Some(EvolutionConfig {
            enabled: true,
            mutable_fields: vec![
                "traits.personality".into(),
                "traits.speaking_style".into(),
                "goals.primary".into(),
            ],
            track_changes: true,
            allow_llm_evolve: false,
        });
        config
    }

    fn make_secret_config() -> PersonaConfig {
        let mut config = make_basic_config();
        config.secrets = Some(vec![
            PersonaSecret {
                content: "Investigating a smuggling ring".into(),
                reveal_conditions: Some(SecretRevealCondition::Context({
                    let mut m = HashMap::new();
                    m.insert(
                        "trust".to_string(),
                        ai_agents_state::ContextMatcher::Compare(ai_agents_state::CompareOp::Gte(
                            0.8,
                        )),
                    );
                    m
                })),
            },
            PersonaSecret {
                content: "Knows the spy's identity".into(),
                reveal_conditions: Some(SecretRevealCondition::Context({
                    let mut m = HashMap::new();
                    m.insert(
                        "trust".to_string(),
                        ai_agents_state::ContextMatcher::Compare(ai_agents_state::CompareOp::Gte(
                            0.95,
                        )),
                    );
                    m
                })),
            },
            PersonaSecret {
                content: "Manual secret".into(),
                reveal_conditions: None,
            },
        ]);
        config
    }

    #[test]
    fn test_from_config_basic() {
        let config = make_basic_config();
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();

        let identity = manager.identity();
        assert_eq!(identity.name, "Captain Elira");
        assert_eq!(identity.role, "Harbor Guard Captain");
    }

    #[test]
    fn test_from_config_no_identity_fails() {
        let config = PersonaConfig::default();
        let result = PersonaManager::from_config(config, None, test_renderer());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("identity"));
    }

    #[test]
    fn test_from_config_validates_mutable_fields() {
        let mut config = make_basic_config();
        config.evolution = Some(EvolutionConfig {
            enabled: true,
            mutable_fields: vec!["secrets.0.content".into()],
            track_changes: false,
            allow_llm_evolve: false,
        });

        let result = PersonaManager::from_config(config, None, test_renderer());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("secrets"));
    }

    #[test]
    fn test_from_config_validates_unknown_paths() {
        let mut config = make_basic_config();
        config.evolution = Some(EvolutionConfig {
            enabled: true,
            mutable_fields: vec!["traits.personailty".into()],
            track_changes: false,
            allow_llm_evolve: false,
        });

        let result = PersonaManager::from_config(config, None, test_renderer());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown"));
    }

    #[test]
    fn test_render_prompt_basic() {
        let config = make_basic_config();
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();

        let ctx = HashMap::new();
        let result = manager.render_prompt(&ctx).unwrap();

        assert!(result.prompt.contains("Captain Elira"));
        assert!(result.prompt.contains("Harbor Guard Captain"));
        assert!(result.prompt.contains("disciplined"));
        assert!(result.prompt.contains("protect_harbor"));
        assert!(result.newly_revealed.is_empty());
    }

    #[test]
    fn test_render_prompt_condensed_when_over_budget() {
        let mut config = make_basic_config();
        config.max_prompt_tokens = Some(10); // Very low budget to force condensed format.

        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();
        let ctx = HashMap::new();
        let result = manager.render_prompt(&ctx).unwrap();

        // Condensed format should NOT contain backstory or goals.
        assert!(!result.prompt.contains("Eastern Campaign"));
        assert!(!result.prompt.contains("## Goals"));
        // But should contain name and role.
        assert!(result.prompt.contains("Captain Elira"));
    }

    #[test]
    fn test_render_prompt_no_condensed_when_under_budget() {
        let mut config = make_basic_config();
        config.max_prompt_tokens = Some(10000); // Very high budget.

        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();
        let ctx = HashMap::new();
        let result = manager.render_prompt(&ctx).unwrap();

        // Should use full format.
        assert!(result.prompt.contains("## Identity"));
        assert!(result.prompt.contains("Eastern Campaign"));
    }

    #[test]
    fn test_render_prompt_with_secrets_revealed() {
        let config = make_secret_config();
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();

        let mut ctx = HashMap::new();
        ctx.insert("trust".to_string(), json!(0.9));

        let result = manager.render_prompt(&ctx).unwrap();

        // First secret should be revealed (trust 0.9 >= 0.8).
        assert!(result.prompt.contains("Investigating a smuggling ring"));
        // Second secret not revealed (trust 0.9 < 0.95).
        assert!(!result.prompt.contains("Knows the spy's identity"));
        // Manual secret never auto-reveals.
        assert!(!result.prompt.contains("Manual secret"));

        // Should report newly revealed.
        assert_eq!(result.newly_revealed.len(), 1);
        assert_eq!(result.newly_revealed[0], "Investigating a smuggling ring");
    }

    #[test]
    fn test_render_prompt_secrets_not_newly_revealed_second_time() {
        let config = make_secret_config();
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();

        let mut ctx = HashMap::new();
        ctx.insert("trust".to_string(), json!(0.9));

        // First call - secret is newly revealed.
        let result1 = manager.render_prompt(&ctx).unwrap();
        assert_eq!(result1.newly_revealed.len(), 1);

        // Second call with same context - secret is still in prompt but NOT newly revealed.
        let result2 = manager.render_prompt(&ctx).unwrap();
        assert!(result2.prompt.contains("Investigating a smuggling ring"));
        assert!(result2.newly_revealed.is_empty());
    }

    #[test]
    fn test_render_prompt_more_secrets_revealed_over_time() {
        let config = make_secret_config();
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();

        // First: trust 0.9 - only first secret revealed.
        let mut ctx = HashMap::new();
        ctx.insert("trust".to_string(), json!(0.9));
        let r1 = manager.render_prompt(&ctx).unwrap();
        assert_eq!(r1.newly_revealed.len(), 1);
        assert_eq!(r1.newly_revealed[0], "Investigating a smuggling ring");

        // Second: trust 0.96 - second secret now also revealed.
        ctx.insert("trust".to_string(), json!(0.96));
        let r2 = manager.render_prompt(&ctx).unwrap();
        assert_eq!(r2.newly_revealed.len(), 1);
        assert_eq!(r2.newly_revealed[0], "Knows the spy's identity");

        // Both secrets in prompt.
        assert!(r2.prompt.contains("Investigating a smuggling ring"));
        assert!(r2.prompt.contains("Knows the spy's identity"));
    }

    #[test]
    fn test_identity_accessor() {
        let config = make_basic_config();
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();

        let identity = manager.identity();
        assert_eq!(identity.name, "Captain Elira");
        assert_eq!(identity.role, "Harbor Guard Captain");
        assert_eq!(identity.affiliation.as_deref(), Some("Harbor Watch"));
    }

    #[test]
    fn test_traits_accessor() {
        let config = make_basic_config();
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();

        let traits = manager.traits().unwrap();
        assert_eq!(
            traits.personality,
            vec!["disciplined", "suspicious", "loyal"]
        );
        assert_eq!(
            traits.speaking_style.as_deref(),
            Some("formal military cadence")
        );
    }

    #[test]
    fn test_traits_accessor_none() {
        let config = PersonaConfig {
            identity: Some(PersonaIdentity {
                name: "Simple".into(),
                role: "Bot".into(),
                description: None,
                backstory: None,
                affiliation: None,
            }),
            traits: None,
            ..Default::default()
        };
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();
        assert!(manager.traits().is_none());
    }

    #[test]
    fn test_evolve_personality() {
        let config = make_evolvable_config();
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();

        let change = manager
            .evolve(
                "traits.personality",
                json!(["confident", "assertive"]),
                Some("Character growth"),
            )
            .unwrap();

        assert_eq!(change.field, "traits.personality");
        assert_eq!(
            change.old_value,
            json!(["disciplined", "suspicious", "loyal"])
        );
        assert_eq!(change.new_value, json!(["confident", "assertive"]));
        assert_eq!(change.reason.as_deref(), Some("Character growth"));

        // Verify the change is reflected in the manager.
        let traits = manager.traits().unwrap();
        assert_eq!(traits.personality, vec!["confident", "assertive"]);
    }

    #[test]
    fn test_evolve_speaking_style() {
        let config = make_evolvable_config();
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();

        let change = manager
            .evolve("traits.speaking_style", json!("casual and friendly"), None)
            .unwrap();

        assert_eq!(change.old_value, json!("formal military cadence"));
        assert_eq!(change.new_value, json!("casual and friendly"));

        let traits = manager.traits().unwrap();
        assert_eq!(
            traits.speaking_style.as_deref(),
            Some("casual and friendly")
        );
    }

    #[test]
    fn test_evolve_disabled() {
        let config = make_basic_config(); // No evolution config.
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();

        let result = manager.evolve("traits.personality", json!(["new"]), None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not enabled"));
    }

    #[test]
    fn test_evolve_field_not_mutable() {
        let config = make_evolvable_config();
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();

        let result = manager.evolve("identity.name", json!("New Name"), None);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not in persona mutable_fields")
        );
    }

    #[test]
    fn test_evolve_rejects_secrets() {
        let mut config = make_evolvable_config();
        // Manually add secrets to mutable_fields (bypassing validation for test).
        config
            .evolution
            .as_mut()
            .unwrap()
            .mutable_fields
            .push("secrets".into());

        // from_config would reject this, so create directly.
        // Instead, test the evolve method's own rejection.
        let manager = PersonaManager {
            config: RwLock::new(config),
            history: RwLock::new(Vec::new()),
            revealed_indices: RwLock::new(HashSet::new()),
            renderer: test_renderer(),
        };

        let result = manager.evolve("secrets", json!([]), None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("immutable"));
    }

    #[test]
    fn test_evolve_tracks_history() {
        let config = make_evolvable_config();
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();

        manager
            .evolve("traits.personality", json!(["bold"]), Some("First change"))
            .unwrap();
        manager
            .evolve(
                "traits.personality",
                json!(["bold", "wise"]),
                Some("Second change"),
            )
            .unwrap();

        let history = manager.history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].reason.as_deref(), Some("First change"));
        assert_eq!(history[1].reason.as_deref(), Some("Second change"));
    }

    #[test]
    fn test_evolve_no_history_when_not_tracking() {
        let mut config = make_evolvable_config();
        config.evolution.as_mut().unwrap().track_changes = false;

        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();
        manager
            .evolve("traits.personality", json!(["new"]), None)
            .unwrap();

        let history = manager.history();
        assert!(history.is_empty());
    }

    #[test]
    fn test_should_register_evolve_tool() {
        let mut config = make_evolvable_config();
        config.evolution.as_mut().unwrap().allow_llm_evolve = true;

        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();
        assert!(manager.should_register_evolve_tool());
    }

    #[test]
    fn test_should_not_register_evolve_tool_when_disabled() {
        let config = make_evolvable_config(); // allow_llm_evolve = false
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();
        assert!(!manager.should_register_evolve_tool());
    }

    #[test]
    fn test_should_not_register_evolve_tool_no_evolution() {
        let config = make_basic_config(); // No evolution config.
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();
        assert!(!manager.should_register_evolve_tool());
    }

    #[test]
    fn test_snapshot_and_restore() {
        let config = make_evolvable_config();
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();

        // Evolve a field.
        manager
            .evolve("traits.personality", json!(["bold"]), Some("Test change"))
            .unwrap();

        // Take a snapshot.
        let snapshot = manager.snapshot();
        assert_eq!(snapshot.history.len(), 1);

        // Now evolve again.
        manager
            .evolve(
                "traits.personality",
                json!(["brave"]),
                Some("Another change"),
            )
            .unwrap();
        assert_eq!(manager.traits().unwrap().personality, vec!["brave"]);
        assert_eq!(manager.history().len(), 2);

        // Restore from snapshot.
        manager.restore(snapshot);
        assert_eq!(manager.traits().unwrap().personality, vec!["bold"]);
        assert_eq!(manager.history().len(), 1);
    }

    #[test]
    fn test_snapshot_as_value_and_restore_from_value() {
        let config = make_evolvable_config();
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();

        manager
            .evolve(
                "traits.personality",
                json!(["evolved"]),
                Some("Snapshot test"),
            )
            .unwrap();

        let value = manager.snapshot_as_value().unwrap();
        assert!(value.is_object());

        // Create a new manager to restore into.
        let config2 = make_evolvable_config();
        let manager2 = PersonaManager::from_config(config2, None, test_renderer()).unwrap();

        manager2.restore_from_value(value).unwrap();
        assert_eq!(manager2.traits().unwrap().personality, vec!["evolved"]);
        assert_eq!(manager2.history().len(), 1);
    }

    #[test]
    fn test_snapshot_preserves_revealed_indices() {
        let config = make_secret_config();
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();

        // Trigger secret reveal.
        let mut ctx = HashMap::new();
        ctx.insert("trust".to_string(), json!(0.9));
        let r1 = manager.render_prompt(&ctx).unwrap();
        assert_eq!(r1.newly_revealed.len(), 1);

        // Snapshot.
        let snapshot = manager.snapshot();
        assert!(snapshot.revealed_indices.contains(&0));

        // Restore.
        let config2 = make_secret_config();
        let manager2 = PersonaManager::from_config(config2, None, test_renderer()).unwrap();
        manager2.restore(snapshot);

        // After restore, the same secret should NOT be newly revealed.
        let r2 = manager2.render_prompt(&ctx).unwrap();
        assert!(r2.newly_revealed.is_empty());
        // But it should still be in the prompt.
        assert!(r2.prompt.contains("Investigating a smuggling ring"));
    }

    #[test]
    fn test_revealed_secrets_accessor() {
        let config = make_secret_config();
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();

        let mut ctx = HashMap::new();
        ctx.insert("trust".to_string(), json!(0.9));

        let revealed = manager.revealed_secrets(&ctx);
        assert_eq!(revealed.len(), 1);
        assert_eq!(revealed[0], "Investigating a smuggling ring");
    }

    #[test]
    fn test_revealed_secrets_no_secrets() {
        let config = make_basic_config(); // No secrets.
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();

        let ctx = HashMap::new();
        let revealed = manager.revealed_secrets(&ctx);
        assert!(revealed.is_empty());
    }

    #[test]
    fn test_template_resolution() {
        let mut registry = PersonaTemplateRegistry::new();
        registry.register(
            "guard_base",
            PersonaConfig {
                identity: Some(PersonaIdentity {
                    name: "Guard".into(),
                    role: "Guard".into(),
                    description: Some("A generic guard".into()),
                    backstory: None,
                    affiliation: Some("City Watch".into()),
                }),
                traits: Some(PersonaTraits {
                    personality: vec!["disciplined".into()],
                    values: None,
                    fears: None,
                    speaking_style: Some("formal".into()),
                }),
                goals: None,
                secrets: None,
                evolution: None,
                templates: None,
                max_prompt_tokens: None,
            },
        );

        let config = PersonaConfig {
            identity: None,
            traits: None,
            goals: Some(PersonaGoals {
                primary: vec!["protect_harbor".into()],
                hidden: None,
            }),
            secrets: None,
            evolution: None,
            templates: Some(PersonaTemplateRef {
                base: "guard_base".into(),
                overrides: Some({
                    let mut m = HashMap::new();
                    m.insert("identity.name".to_string(), json!("Captain Tam"));
                    m
                }),
            }),
            max_prompt_tokens: None,
        };

        let registry = Arc::new(registry);
        let manager = PersonaManager::from_config(config, Some(registry), test_renderer()).unwrap();

        let identity = manager.identity();
        assert_eq!(identity.name, "Captain Tam");
        assert_eq!(identity.role, "Guard");
        assert_eq!(identity.affiliation.as_deref(), Some("City Watch"));

        // Goals from instance config.
        let config = manager.config();
        assert_eq!(
            config.goals.as_ref().unwrap().primary,
            vec!["protect_harbor".to_string()]
        );
    }

    #[test]
    fn test_template_resolution_missing_registry() {
        let config = PersonaConfig {
            identity: None,
            templates: Some(PersonaTemplateRef {
                base: "guard_base".into(),
                overrides: None,
            }),
            ..Default::default()
        };

        let result = PersonaManager::from_config(config, None, test_renderer());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("no PersonaTemplateRegistry")
        );
    }

    #[test]
    fn test_template_resolution_missing_template() {
        let registry = Arc::new(PersonaTemplateRegistry::new());

        let config = PersonaConfig {
            identity: None,
            templates: Some(PersonaTemplateRef {
                base: "nonexistent".into(),
                overrides: None,
            }),
            ..Default::default()
        };

        let result = PersonaManager::from_config(config, Some(registry), test_renderer());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_debug_format() {
        let config = make_basic_config();
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();
        let debug = format!("{:?}", manager);
        assert!(debug.contains("Captain Elira"));
        assert!(debug.contains("PersonaManager"));
    }

    #[test]
    fn test_multiple_evolves_accumulate() {
        let config = make_evolvable_config();
        let manager = PersonaManager::from_config(config, None, test_renderer()).unwrap();

        manager
            .evolve("traits.personality", json!(["step1"]), Some("Step 1"))
            .unwrap();
        manager
            .evolve("traits.speaking_style", json!("casual"), Some("Step 2"))
            .unwrap();
        manager
            .evolve("goals.primary", json!(["new_goal"]), Some("Step 3"))
            .unwrap();

        let traits = manager.traits().unwrap();
        assert_eq!(traits.personality, vec!["step1"]);
        assert_eq!(traits.speaking_style.as_deref(), Some("casual"));

        let config = manager.config();
        assert_eq!(
            config.goals.as_ref().unwrap().primary,
            vec!["new_goal".to_string()]
        );

        assert_eq!(manager.history().len(), 3);
    }
}

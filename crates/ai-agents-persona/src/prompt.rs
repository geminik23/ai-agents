//! Persona prompt rendering - converts structured persona config into system prompt text.

use std::collections::HashMap;

use serde_json::Value;

use ai_agents_context::TemplateRenderer;
use ai_agents_core::Result;

use crate::config::{PersonaConfig, PersonaGoals, PersonaIdentity, PersonaTraits};

/// Render the full persona prompt from a PersonaConfig.
/// Renders Jinja2 templates in backstory and text fields using context.
/// Returns the prompt string (without secrets - those are appended separately by the manager).
pub fn render_full_prompt(
    config: &PersonaConfig,
    context: &HashMap<String, Value>,
    renderer: &TemplateRenderer,
) -> Result<String> {
    let mut sections = Vec::new();

    if let Some(ref identity) = config.identity {
        sections.push(render_identity(identity, context, renderer)?);
    }

    if let Some(ref traits) = config.traits {
        let traits_text = render_traits(traits);
        if !traits_text.is_empty() {
            sections.push(traits_text);
        }
    }

    if let Some(ref goals) = config.goals {
        let goals_text = render_goals(goals);
        if !goals_text.is_empty() {
            sections.push(goals_text);
        }
    }

    Ok(sections.join("\n\n"))
}

/// Render a condensed persona prompt that fits within a token budget.
/// Includes only: name, role, affiliation, top personality traits, and speaking style.
pub fn render_condensed_prompt(
    config: &PersonaConfig,
    context: &HashMap<String, Value>,
    renderer: &TemplateRenderer,
) -> Result<String> {
    let mut parts = Vec::new();

    if let Some(ref identity) = config.identity {
        let mut line = format!("You are {}, {}", identity.name, identity.role);
        if let Some(ref aff) = identity.affiliation {
            let rendered_aff = render_text_field(aff, context, renderer)?;
            line.push_str(&format!(" ({})", rendered_aff));
        }
        line.push('.');
        parts.push(line);
    }

    if let Some(ref traits) = config.traits {
        let mut trait_parts = Vec::new();
        if !traits.personality.is_empty() {
            trait_parts.push(format!("Personality: {}", traits.personality.join(", ")));
        }
        if let Some(ref style) = traits.speaking_style {
            let rendered = render_text_field(style, context, renderer)?;
            trait_parts.push(format!("Style: {}", rendered));
        }
        if !trait_parts.is_empty() {
            parts.push(trait_parts.join(". ") + ".");
        }
    }

    Ok(parts.join("\n"))
}

/// Append revealed secrets as a separate section.
pub fn render_secrets_section(revealed_contents: &[String]) -> String {
    if revealed_contents.is_empty() {
        return String::new();
    }
    let mut lines = vec!["## Confidential (share only if appropriate)".to_string()];
    for content in revealed_contents {
        lines.push(format!("- {}", content));
    }
    lines.join("\n")
}

// TODO: fix this later
/// Estimate token count using the char/4 heuristic.
/// Same heuristic as RuntimeAgent::estimate_tokens().
pub fn estimate_tokens(text: &str) -> u32 {
    (text.len() / 4) as u32
}

fn render_identity(
    identity: &PersonaIdentity,
    context: &HashMap<String, Value>,
    renderer: &TemplateRenderer,
) -> Result<String> {
    let mut lines = Vec::new();

    lines.push("## Identity".to_string());

    let mut intro = format!("You are {}, {}", identity.name, identity.role);
    intro.push('.');
    lines.push(intro);

    if let Some(ref aff) = identity.affiliation {
        let rendered = render_text_field(aff, context, renderer)?;
        lines.push(format!("Affiliation: {}.", rendered));
    }

    if let Some(ref backstory) = identity.backstory {
        let rendered = render_text_field(backstory, context, renderer)?;
        let trimmed = rendered.trim();
        if !trimmed.is_empty() {
            lines.push(String::new());
            lines.push(trimmed.to_string());
        }
    }

    Ok(lines.join("\n"))
}

fn render_traits(traits: &PersonaTraits) -> String {
    let mut lines = Vec::new();

    lines.push("## Personality & Style".to_string());

    let mut has_content = false;

    if !traits.personality.is_empty() {
        lines.push(format!("Personality: {}.", traits.personality.join(", ")));
        has_content = true;
    }

    if let Some(ref values) = traits.values {
        if !values.is_empty() {
            lines.push(format!("Values: {}.", values.join(", ")));
            has_content = true;
        }
    }

    if let Some(ref fears) = traits.fears {
        if !fears.is_empty() {
            lines.push(format!("Fears: {}.", fears.join(", ")));
            has_content = true;
        }
    }

    if let Some(ref style) = traits.speaking_style {
        lines.push(format!("Speaking style: {}.", style));
        has_content = true;
    }

    if has_content {
        lines.join("\n")
    } else {
        String::new()
    }
}

fn render_goals(goals: &PersonaGoals) -> String {
    if goals.primary.is_empty() {
        return String::new();
    }

    let mut lines = Vec::new();
    lines.push("## Goals".to_string());
    for goal in &goals.primary {
        lines.push(format!("- {}", goal));
    }
    lines.join("\n")
}

/// Render a text field through Jinja2 if it contains template syntax.
fn render_text_field(
    text: &str,
    context: &HashMap<String, Value>,
    renderer: &TemplateRenderer,
) -> Result<String> {
    if text.contains("{{") || text.contains("{%") {
        renderer.render(text, context)
    } else {
        Ok(text.to_string())
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

    fn empty_context() -> HashMap<String, Value> {
        HashMap::new()
    }

    fn make_full_config() -> PersonaConfig {
        PersonaConfig {
            identity: Some(PersonaIdentity {
                name: "Captain Elira".into(),
                role: "Harbor Guard Captain".into(),
                description: Some("A disciplined former soldier".into()),
                backstory: Some(
                    "Former soldier who served in the Eastern Campaign.\nNow guards the harbor."
                        .into(),
                ),
                affiliation: Some("Harbor Watch".into()),
            }),
            traits: Some(PersonaTraits {
                personality: vec!["disciplined".into(), "suspicious".into(), "loyal".into()],
                values: Some(vec!["duty".into(), "order".into(), "justice".into()]),
                fears: Some(vec!["civil_unrest".into(), "betrayal".into()]),
                speaking_style: Some("formal military cadence, short clipped sentences".into()),
            }),
            goals: Some(PersonaGoals {
                primary: vec!["protect_harbor".into(), "investigate_smuggling".into()],
                hidden: Some(vec!["Find the spy within the Watch".into()]),
            }),
            secrets: None,
            evolution: None,
            templates: None,
            max_prompt_tokens: None,
        }
    }

    #[test]
    fn test_render_full_prompt() {
        let config = make_full_config();
        let ctx = empty_context();
        let renderer = test_renderer();

        let result = render_full_prompt(&config, &ctx, &renderer).unwrap();

        assert!(result.contains("## Identity"));
        assert!(result.contains("You are Captain Elira, Harbor Guard Captain."));
        assert!(result.contains("Affiliation: Harbor Watch."));
        assert!(result.contains("Former soldier who served in the Eastern Campaign."));

        assert!(result.contains("## Personality & Style"));
        assert!(result.contains("Personality: disciplined, suspicious, loyal."));
        assert!(result.contains("Values: duty, order, justice."));
        assert!(result.contains("Fears: civil_unrest, betrayal."));
        assert!(
            result.contains("Speaking style: formal military cadence, short clipped sentences.")
        );

        assert!(result.contains("## Goals"));
        assert!(result.contains("- protect_harbor"));
        assert!(result.contains("- investigate_smuggling"));

        // Hidden goals should NOT appear in the prompt.
        assert!(!result.contains("Find the spy"));
    }

    #[test]
    fn test_render_minimal_prompt() {
        let config = PersonaConfig {
            identity: Some(PersonaIdentity {
                name: "Support Agent".into(),
                role: "Customer Support".into(),
                description: None,
                backstory: None,
                affiliation: None,
            }),
            traits: None,
            goals: None,
            secrets: None,
            evolution: None,
            templates: None,
            max_prompt_tokens: None,
        };
        let ctx = empty_context();
        let renderer = test_renderer();

        let result = render_full_prompt(&config, &ctx, &renderer).unwrap();
        assert!(result.contains("You are Support Agent, Customer Support."));
        assert!(!result.contains("## Personality"));
        assert!(!result.contains("## Goals"));
    }

    #[test]
    fn test_render_condensed_prompt() {
        let config = make_full_config();
        let ctx = empty_context();
        let renderer = test_renderer();

        let result = render_condensed_prompt(&config, &ctx, &renderer).unwrap();

        assert!(result.contains("You are Captain Elira, Harbor Guard Captain (Harbor Watch)."));
        assert!(result.contains("Personality: disciplined, suspicious, loyal"));
        assert!(result.contains("Style: formal military cadence"));

        // Condensed should NOT contain backstory, values, fears, or goals.
        assert!(!result.contains("Former soldier"));
        assert!(!result.contains("Values:"));
        assert!(!result.contains("Fears:"));
        assert!(!result.contains("## Goals"));
    }

    #[test]
    fn test_render_condensed_no_traits() {
        let config = PersonaConfig {
            identity: Some(PersonaIdentity {
                name: "Bot".into(),
                role: "Assistant".into(),
                description: None,
                backstory: None,
                affiliation: None,
            }),
            traits: None,
            goals: None,
            secrets: None,
            evolution: None,
            templates: None,
            max_prompt_tokens: None,
        };
        let ctx = empty_context();
        let renderer = test_renderer();

        let result = render_condensed_prompt(&config, &ctx, &renderer).unwrap();
        assert_eq!(result, "You are Bot, Assistant.");
    }

    #[test]
    fn test_render_secrets_section() {
        let secrets = vec![
            "Investigating a smuggling ring".to_string(),
            "The captain knows the spy's identity".to_string(),
        ];
        let result = render_secrets_section(&secrets);
        assert!(result.contains("## Confidential (share only if appropriate)"));
        assert!(result.contains("- Investigating a smuggling ring"));
        assert!(result.contains("- The captain knows the spy's identity"));
    }

    #[test]
    fn test_render_secrets_section_empty() {
        let result = render_secrets_section(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_estimate_tokens() {
        let text = "This is a test string with some content.";
        let tokens = estimate_tokens(text);
        assert_eq!(tokens, (text.len() / 4) as u32);
    }

    #[test]
    fn test_render_backstory_with_jinja2() {
        let config = PersonaConfig {
            identity: Some(PersonaIdentity {
                name: "Agent".into(),
                role: "Support".into(),
                description: None,
                backstory: Some(
                    "You work for {{ context.company.name }} in the {{ context.company.dept }} department."
                        .into(),
                ),
                affiliation: None,
            }),
            traits: None,
            goals: None,
            secrets: None,
            evolution: None,
            templates: None,
            max_prompt_tokens: None,
        };

        let mut ctx = HashMap::new();
        ctx.insert(
            "company".to_string(),
            json!({"name": "Acme Corp", "dept": "Engineering"}),
        );
        let renderer = test_renderer();

        let result = render_full_prompt(&config, &ctx, &renderer).unwrap();
        assert!(result.contains("You work for Acme Corp in the Engineering department."));
    }

    #[test]
    fn test_render_traits_empty_personality() {
        let traits = PersonaTraits {
            personality: vec![],
            values: None,
            fears: None,
            speaking_style: Some("casual".into()),
        };
        let text = render_traits(&traits);
        assert!(!text.contains("Personality:"));
        assert!(text.contains("Speaking style: casual."));
    }

    #[test]
    fn test_render_goals_empty() {
        let goals = PersonaGoals {
            primary: vec![],
            hidden: None,
        };
        let text = render_goals(&goals);
        assert!(text.is_empty());
    }

    #[test]
    fn test_render_identity_no_affiliation_no_backstory() {
        let identity = PersonaIdentity {
            name: "Simple".into(),
            role: "Bot".into(),
            description: None,
            backstory: None,
            affiliation: None,
        };
        let ctx = empty_context();
        let renderer = test_renderer();

        let result = render_identity(&identity, &ctx, &renderer).unwrap();
        assert!(result.contains("## Identity"));
        assert!(result.contains("You are Simple, Bot."));
        assert!(!result.contains("Affiliation:"));
    }

    #[test]
    fn test_render_traits_only_values() {
        let traits = PersonaTraits {
            personality: vec![],
            values: Some(vec!["honesty".into()]),
            fears: None,
            speaking_style: None,
        };
        let text = render_traits(&traits);
        assert!(text.contains("Values: honesty."));
        assert!(!text.contains("Personality:"));
        assert!(!text.contains("Fears:"));
        assert!(!text.contains("Speaking style:"));
    }
}

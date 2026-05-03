use serde_json::{Value, json};

use crate::config::InjectionFormat;
use crate::types::{Relationship, RelationshipModel};

/// Convert a relationship into the structured context object injected at `relationships.current_actor`.
pub fn relationship_to_context_value(relationship: &Relationship) -> Value {
    let mut root = serde_json::Map::new();
    root.insert("actor_id".to_string(), json!(relationship.actor_id));
    if let Some(ref name) = relationship.actor_name {
        root.insert("actor_name".to_string(), json!(name));
    } else {
        root.insert("actor_name".to_string(), Value::Null);
    }
    root.insert(
        "interaction_count".to_string(),
        json!(relationship.interaction_count),
    );
    root.insert(
        "first_interaction".to_string(),
        json!(relationship.first_interaction.to_rfc3339()),
    );
    root.insert(
        "last_interaction".to_string(),
        json!(relationship.last_interaction.to_rfc3339()),
    );
    root.insert("model".to_string(), json!(relationship.model));
    root.insert("dimensions".to_string(), json!(relationship.dimensions));
    root.insert("agent_to_actor".to_string(), json!(relationship.dimensions));

    if matches!(relationship.model, RelationshipModel::TwoSided) {
        root.insert(
            "perceived_actor_to_agent".to_string(),
            json!(relationship.perceived_actor_to_agent),
        );
        root.insert(
            "mutual".to_string(),
            json!(relationship.mutual_dimensions()),
        );
    }

    for (name, value) in &relationship.dimensions {
        root.insert(name.clone(), json!(value));
    }

    Value::Object(root)
}

/// Format relationship memory for prompt injection using the requested format and token budget.
pub fn format_relationship(
    relationship: &Relationship,
    format: &InjectionFormat,
    max_tokens: usize,
) -> String {
    let text = match format {
        InjectionFormat::Summary => format_summary(relationship),
        InjectionFormat::ScoresOnly => format_scores_only(relationship),
        InjectionFormat::Full => format_full(relationship),
    };
    truncate_to_tokens(&text, max_tokens)
}

fn format_summary(relationship: &Relationship) -> String {
    let actor = relationship
        .actor_name
        .as_deref()
        .unwrap_or(&relationship.actor_id);
    let score_text = format_scores_for_relationship(relationship);

    let event_text = relationship
        .notable_events
        .last()
        .map(|event| format!(" Most recent notable event: {}.", event.description))
        .unwrap_or_default();

    format!(
        "Relationship with {}: {}. Interactions: {}.{}",
        actor, score_text, relationship.interaction_count, event_text
    )
}

fn format_scores_only(relationship: &Relationship) -> String {
    format_scores_for_relationship(relationship)
}

fn format_scores_for_relationship(relationship: &Relationship) -> String {
    if matches!(relationship.model, RelationshipModel::TwoSided) {
        format!(
            "agent_to_actor [{}]; perceived_actor_to_agent [{}]; mutual [{}]",
            format_scores(&relationship.dimensions),
            format_scores(&relationship.perceived_actor_to_agent),
            format_scores(&relationship.mutual_dimensions())
        )
    } else {
        format_scores(&relationship.dimensions)
    }
}

fn format_scores(scores: &std::collections::HashMap<String, f64>) -> String {
    let mut scores: Vec<_> = scores.iter().collect();
    scores.sort_by(|a, b| a.0.cmp(b.0));
    scores
        .into_iter()
        .map(|(name, value)| format!("{}={:.2}", name, value))
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_full(relationship: &Relationship) -> String {
    let actor = relationship
        .actor_name
        .as_deref()
        .unwrap_or(&relationship.actor_id);
    let mut lines = vec![format!("Relationship with {}:", actor)];

    lines.push(format!(
        "- Agent to actor: {}",
        format_scores(&relationship.dimensions)
    ));
    if matches!(relationship.model, RelationshipModel::TwoSided) {
        lines.push(format!(
            "- Perceived actor to agent: {}",
            format_scores(&relationship.perceived_actor_to_agent)
        ));
        lines.push(format!(
            "- Mutual: {}",
            format_scores(&relationship.mutual_dimensions())
        ));
    }
    lines.push(format!(
        "- Interaction count: {}",
        relationship.interaction_count
    ));
    lines.push(format!(
        "- First interaction: {}",
        relationship.first_interaction.to_rfc3339()
    ));
    lines.push(format!(
        "- Last interaction: {}",
        relationship.last_interaction.to_rfc3339()
    ));

    if !relationship.notable_events.is_empty() {
        lines.push("- Notable events:".to_string());
        for event in relationship.notable_events.iter().rev().take(5).rev() {
            lines.push(format!(
                "  - [{}] {}",
                event.timestamp.date_naive(),
                event.description
            ));
        }
    }

    lines.join("\n")
}

fn truncate_to_tokens(text: &str, max_tokens: usize) -> String {
    if max_tokens == 0 {
        return String::new();
    }
    let max_chars = max_tokens.saturating_mul(4);
    if text.len() <= max_chars {
        return text.to_string();
    }
    let mut truncated = text.chars().take(max_chars).collect::<String>();
    truncated.push_str("...");
    truncated
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::defaults::builtin_dimensions;
    use crate::types::{Relationship, RelationshipModel};

    use super::*;

    #[test]
    fn test_context_value_contains_shortcut_dimensions() {
        let rel = Relationship::new(
            "actor_1",
            None,
            &builtin_dimensions(),
            RelationshipModel::OneSided,
        );
        let value = relationship_to_context_value(&rel);
        assert!(value.get("trust").is_some());
        assert!(value.get("dimensions").is_some());
    }

    #[test]
    fn test_scores_only_format() {
        let mut rel = Relationship::new(
            "actor_1",
            None,
            &builtin_dimensions(),
            RelationshipModel::OneSided,
        );
        rel.dimensions = HashMap::from([("trust".to_string(), 0.75)]);
        let text = format_relationship(&rel, &InjectionFormat::ScoresOnly, 100);
        assert!(text.contains("trust=0.75"));
    }
}

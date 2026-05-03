use std::collections::HashMap;

use crate::types::RelationshipDimensionDefinition;

/// Return the built-in relationship dimensions used by shorthand configs.
pub fn builtin_dimensions() -> HashMap<String, RelationshipDimensionDefinition> {
    let mut map = HashMap::new();
    map.insert(
        "trust".to_string(),
        RelationshipDimensionDefinition::new("How much the agent trusts the actor", -1.0, 1.0, 0.0),
    );
    map.insert(
        "sentiment".to_string(),
        RelationshipDimensionDefinition::new(
            "Overall positive or negative emotional stance toward the actor",
            -1.0,
            1.0,
            0.0,
        ),
    );
    map.insert(
        "familiarity".to_string(),
        RelationshipDimensionDefinition::new("How well the agent knows the actor", 0.0, 1.0, 0.0),
    );
    map.insert(
        "rapport".to_string(),
        RelationshipDimensionDefinition::new(
            "Strength of social connection and conversational ease",
            0.0,
            1.0,
            0.0,
        ),
    );
    map
}

/// Return the default built-in dimension names used when none are specified.
pub fn default_dimension_names() -> Vec<String> {
    vec![
        "trust".to_string(),
        "sentiment".to_string(),
        "familiarity".to_string(),
        "rapport".to_string(),
    ]
}

/// Build a fallback definition for a custom dimension that is referenced in shorthand form but has no explicit metadata.
pub fn fallback_dimension(name: &str) -> RelationshipDimensionDefinition {
    RelationshipDimensionDefinition::new(
        format!("Custom relationship dimension named {}", name),
        -1.0,
        1.0,
        0.0,
    )
}

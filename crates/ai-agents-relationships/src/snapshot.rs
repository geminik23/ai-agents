use serde::{Deserialize, Serialize};
use serde_json::Value;

use ai_agents_core::{AgentError, Result};

use crate::types::Relationship;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RelationshipSnapshot {
    #[serde(default)]
    pub relationships: Vec<Relationship>,
}

impl RelationshipSnapshot {
    pub fn new(relationships: Vec<Relationship>) -> Self {
        Self { relationships }
    }

    pub fn to_value(&self) -> Result<Value> {
        serde_json::to_value(self).map_err(|e| AgentError::Config(e.to_string()))
    }

    pub fn from_value(value: Value) -> Result<Self> {
        serde_json::from_value(value).map_err(|e| AgentError::Config(e.to_string()))
    }
}

pub fn relationship_to_value(relationship: &Relationship) -> Result<Value> {
    serde_json::to_value(relationship).map_err(|e| AgentError::Config(e.to_string()))
}

pub fn relationship_from_value(value: Value) -> Result<Relationship> {
    serde_json::from_value(value).map_err(|e| AgentError::Config(e.to_string()))
}

#[cfg(test)]
mod tests {
    use crate::defaults::builtin_dimensions;
    use crate::types::Relationship;

    use super::*;

    #[test]
    fn test_snapshot_roundtrip() {
        let rel = Relationship::new("actor_1", None, &builtin_dimensions());
        let snapshot = RelationshipSnapshot::new(vec![rel]);
        let value = snapshot.to_value().unwrap();
        let restored = RelationshipSnapshot::from_value(value).unwrap();
        assert_eq!(restored.relationships.len(), 1);
    }
}

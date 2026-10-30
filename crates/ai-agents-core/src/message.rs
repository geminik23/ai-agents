//! Core message types for AI Agents framework

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
    Function,
}

/// Controller-derived history remains available for provider replay, but is not actor-memory evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MessageProvenance {
    pub run_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<MessageProvenance>,
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<DateTime<Utc>>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            provenance: None,
            role: Role::System,
            content: content.into(),
            name: None,
            timestamp: Some(Utc::now()),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            provenance: None,
            role: Role::User,
            content: content.into(),
            name: None,
            timestamp: Some(Utc::now()),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            provenance: None,
            role: Role::Assistant,
            content: content.into(),
            name: None,
            timestamp: Some(Utc::now()),
        }
    }

    pub fn tool(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            provenance: None,
            role: Role::Tool,
            content: content.into(),
            name: Some(name.into()),
            timestamp: Some(Utc::now()),
        }
    }

    pub fn function(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            provenance: None,
            role: Role::Function,
            content: content.into(),
            name: Some(name.into()),
            timestamp: Some(Utc::now()),
        }
    }
}

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use uuid::Uuid;

use crate::event::ObservationPurpose;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanContext {
    pub trace_id: String,
    pub span_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,
    pub turn_id: String,
    pub agent_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default)]
    pub purpose: ObservationPurpose,
    #[serde(default)]
    pub tags: HashMap<String, String>,
}

impl SpanContext {
    pub fn new_root(agent_id: impl Into<String>) -> Self {
        Self {
            trace_id: Uuid::new_v4().to_string(),
            span_id: Uuid::new_v4().to_string(),
            parent_span_id: None,
            turn_id: Uuid::new_v4().to_string(),
            agent_id: agent_id.into(),
            actor_id: None,
            session_id: None,
            state: None,
            language: None,
            purpose: ObservationPurpose::default(),
            tags: HashMap::new(),
        }
    }

    pub fn child(&self) -> Self {
        let mut child = self.clone();
        child.parent_span_id = Some(self.span_id.clone());
        child.span_id = Uuid::new_v4().to_string();
        child
    }

    pub fn with_actor(mut self, actor_id: Option<String>) -> Self {
        self.actor_id = actor_id;
        self
    }

    pub fn with_session(mut self, session_id: Option<String>) -> Self {
        self.session_id = session_id;
        self
    }

    pub fn with_state(mut self, state: Option<String>) -> Self {
        self.state = state;
        self
    }

    pub fn with_language(mut self, language: Option<String>) -> Self {
        self.language = language;
        self
    }

    pub fn with_purpose(mut self, purpose: ObservationPurpose) -> Self {
        self.purpose = purpose;
        self
    }
}

tokio::task_local! {
    static OBSERVATION_CONTEXT: SpanContext;
}

pub fn current_observation_context() -> Option<SpanContext> {
    OBSERVATION_CONTEXT.try_with(Clone::clone).ok()
}

pub async fn with_observation_context<F, T>(context: SpanContext, future: F) -> T
where
    F: Future<Output = T>,
{
    OBSERVATION_CONTEXT.scope(context, future).await
}

pub async fn with_observation_purpose<F, T>(purpose: ObservationPurpose, future: F) -> T
where
    F: Future<Output = T>,
{
    let mut context =
        current_observation_context().unwrap_or_else(|| SpanContext::new_root("unknown"));
    context.purpose = purpose;
    OBSERVATION_CONTEXT.scope(context, future).await
}

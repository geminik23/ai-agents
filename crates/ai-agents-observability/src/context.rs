use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use uuid::Uuid;

use crate::event::ObservationPurpose;

/// Task-local trace labels carried through async agent work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanContext {
    /// Connects all spans that belong to the same user turn or multi-agent flow.
    pub trace_id: String,
    /// Identifies the current span context.
    pub span_id: String,
    /// Links this span to the parent span when work is nested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,
    /// Identifies the current chat turn for this agent call.
    pub turn_id: String,
    /// Identifies the agent currently producing events.
    pub agent_id: String,
    /// Identifies the user, player, customer, or agent actor when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor_id: Option<String>,
    /// Identifies the runtime or persisted session when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Records the current state-machine state when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    /// Records the language label used for language aggregation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Explains why the observed operation is running.
    #[serde(default)]
    pub purpose: ObservationPurpose,
    /// Extra safe labels copied into event dimensions.
    #[serde(default)]
    pub tags: HashMap<String, String>,
}

impl SpanContext {
    /// Creates a new root context for an external agent turn.
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

    /// Creates a child context that keeps the trace ID and links to this span.
    pub fn child(&self) -> Self {
        let mut child = self.clone();
        child.parent_span_id = Some(self.span_id.clone());
        child.span_id = Uuid::new_v4().to_string();
        child
    }

    /// Creates a child context and switches attribution to another agent.
    pub fn child_for_agent(&self, agent_id: impl Into<String>) -> Self {
        let mut child = self.child();
        child.agent_id = agent_id.into();
        child
    }

    /// Assigns a new turn ID while preserving the existing trace.
    pub fn with_new_turn(mut self) -> Self {
        self.turn_id = Uuid::new_v4().to_string();
        self
    }

    /// Adds a safe dimension tag to the context.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
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

/// Returns a clone of the current task-local observation context.
pub fn current_observation_context() -> Option<SpanContext> {
    OBSERVATION_CONTEXT.try_with(Clone::clone).ok()
}

/// Runs a future with the supplied task-local observation context.
pub async fn with_observation_context<F, T>(context: SpanContext, future: F) -> T
where
    F: Future<Output = T>,
{
    OBSERVATION_CONTEXT.scope(context, future).await
}

/// Runs a future with the current context but a different purpose label.
pub async fn with_observation_purpose<F, T>(purpose: ObservationPurpose, future: F) -> T
where
    F: Future<Output = T>,
{
    let mut context =
        current_observation_context().unwrap_or_else(|| SpanContext::new_root("unknown"));
    context.purpose = purpose;
    OBSERVATION_CONTEXT.scope(context, future).await
}

/// Runs a future after applying a transformation to the current context.
pub async fn with_updated_observation_context<F, T, U>(update: U, future: F) -> T
where
    F: Future<Output = T>,
    U: FnOnce(SpanContext) -> SpanContext,
{
    let context = current_observation_context().unwrap_or_else(|| SpanContext::new_root("unknown"));
    OBSERVATION_CONTEXT.scope(update(context), future).await
}

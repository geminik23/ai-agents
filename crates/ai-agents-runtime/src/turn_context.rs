use serde::{Deserialize, Serialize};

use tokio::task_local;

task_local! {
    static TURN_ACTOR_CONTEXT: TurnActorContext;
}

/// Turn-scoped actor identity forwarded through direct chats, registry sends, and orchestration calls.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnActorContext {
    /// Original user, customer, player, or other top-level actor for the turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin_actor_id: Option<String>,
    /// Immediate agent sender for inter-agent hops within the same turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sender_agent_id: Option<String>,
}

impl TurnActorContext {
    /// Create an empty turn actor context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the original actor ID for the turn.
    pub fn with_origin_actor(mut self, actor_id: impl Into<String>) -> Self {
        self.origin_actor_id = Some(actor_id.into());
        self
    }

    /// Set the immediate sender agent ID for the turn.
    pub fn with_sender_agent(mut self, agent_id: impl Into<String>) -> Self {
        self.sender_agent_id = Some(agent_id.into());
        self
    }

    /// Clone this context while overriding the immediate sender agent ID.
    pub fn for_sender(&self, agent_id: impl Into<String>) -> Self {
        let mut next = self.clone();
        next.sender_agent_id = Some(agent_id.into());
        next
    }

    /// Return the effective actor ID for memory features, preferring the original
    /// actor and falling back to the immediate sender agent.
    pub fn effective_actor_id(&self) -> Option<&str> {
        self.origin_actor_id
            .as_deref()
            .or(self.sender_agent_id.as_deref())
    }

    /// Returns `true` when no actor identity is attached to the turn.
    pub fn is_empty(&self) -> bool {
        self.origin_actor_id.is_none() && self.sender_agent_id.is_none()
    }
}

pub(crate) async fn scope_actor_context<F, T>(context: TurnActorContext, future: F) -> T
where
    F: std::future::Future<Output = T> + Send,
    T: Send,
{
    TURN_ACTOR_CONTEXT.scope(context, future).await
}

pub(crate) fn current_turn_actor_context() -> Option<TurnActorContext> {
    TURN_ACTOR_CONTEXT.try_with(Clone::clone).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effective_actor_prefers_origin() {
        let ctx = TurnActorContext::new()
            .with_origin_actor("user_1")
            .with_sender_agent("agent_a");
        assert_eq!(ctx.effective_actor_id(), Some("user_1"));
    }

    #[test]
    fn test_effective_actor_falls_back_to_sender() {
        let ctx = TurnActorContext::new().with_sender_agent("agent_a");
        assert_eq!(ctx.effective_actor_id(), Some("agent_a"));
    }

    #[test]
    fn test_for_sender_preserves_origin() {
        let ctx = TurnActorContext::new().with_origin_actor("user_1");
        let next = ctx.for_sender("agent_b");
        assert_eq!(next.origin_actor_id.as_deref(), Some("user_1"));
        assert_eq!(next.sender_agent_id.as_deref(), Some("agent_b"));
    }
}

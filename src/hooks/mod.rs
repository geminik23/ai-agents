use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, error, info, warn};

use crate::agent::AgentResponse;
use crate::error::AgentError;
use crate::hitl::{ApprovalRequest, ApprovalResult};
use crate::llm::{ChatMessage, LLMResponse};
use crate::tools::ToolResult;

#[async_trait]
pub trait AgentHooks: Send + Sync {
    async fn on_message_received(&self, _message: &str) {}

    async fn on_llm_start(&self, _messages: &[ChatMessage]) {}

    async fn on_llm_complete(&self, _response: &LLMResponse, _duration_ms: u64) {}

    async fn on_tool_start(&self, _tool: &str, _args: &Value) {}

    async fn on_tool_complete(&self, _tool: &str, _result: &ToolResult, _duration_ms: u64) {}

    async fn on_state_transition(&self, _from: Option<&str>, _to: &str, _reason: &str) {}

    async fn on_error(&self, _error: &AgentError) {}

    async fn on_response(&self, _response: &AgentResponse) {}

    async fn on_approval_requested(&self, _request: &ApprovalRequest) {}

    async fn on_approval_result(&self, _request_id: &str, _result: &ApprovalResult) {}
}

pub struct NoopHooks;

#[async_trait]
impl AgentHooks for NoopHooks {}

pub struct LoggingHooks {
    prefix: String,
}

impl LoggingHooks {
    pub fn new() -> Self {
        Self {
            prefix: "[Agent]".to_string(),
        }
    }

    pub fn with_prefix(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }
}

impl Default for LoggingHooks {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AgentHooks for LoggingHooks {
    async fn on_message_received(&self, message: &str) {
        let preview = if message.len() > 100 {
            format!("{}...", &message[..100])
        } else {
            message.to_string()
        };
        info!("{} Message received: {}", self.prefix, preview);
    }

    async fn on_llm_start(&self, messages: &[ChatMessage]) {
        debug!(
            "{} LLM starting with {} messages",
            self.prefix,
            messages.len()
        );
    }

    async fn on_llm_complete(&self, response: &LLMResponse, duration_ms: u64) {
        info!(
            "{} LLM complete in {}ms, tokens: {:?}",
            self.prefix, duration_ms, response.usage
        );
    }

    async fn on_tool_start(&self, tool: &str, args: &Value) {
        debug!("{} Tool {} starting with args: {}", self.prefix, tool, args);
    }

    async fn on_tool_complete(&self, tool: &str, result: &ToolResult, duration_ms: u64) {
        if result.success {
            info!(
                "{} Tool {} completed in {}ms",
                self.prefix, tool, duration_ms
            );
        } else {
            warn!(
                "{} Tool {} failed in {}ms: {}",
                self.prefix, tool, duration_ms, result.output
            );
        }
    }

    async fn on_state_transition(&self, from: Option<&str>, to: &str, reason: &str) {
        info!(
            "{} State transition: {:?} -> {} ({})",
            self.prefix, from, to, reason
        );
    }

    async fn on_error(&self, err: &AgentError) {
        error!("{} Error: {}", self.prefix, err);
    }

    async fn on_response(&self, response: &AgentResponse) {
        let preview = if response.content.len() > 100 {
            format!("{}...", &response.content[..100])
        } else {
            response.content.clone()
        };
        debug!("{} Response: {}", self.prefix, preview);
    }

    async fn on_approval_requested(&self, request: &ApprovalRequest) {
        info!(
            "{} Approval requested [{}]: {}",
            self.prefix, request.id, request.message
        );
    }

    async fn on_approval_result(&self, request_id: &str, result: &ApprovalResult) {
        match result {
            ApprovalResult::Approved => {
                info!("{} Approval [{}]: approved", self.prefix, request_id);
            }
            ApprovalResult::Rejected { reason } => {
                warn!(
                    "{} Approval [{}]: rejected ({:?})",
                    self.prefix, request_id, reason
                );
            }
            ApprovalResult::Modified { .. } => {
                info!(
                    "{} Approval [{}]: approved with modifications",
                    self.prefix, request_id
                );
            }
            ApprovalResult::Timeout => {
                warn!("{} Approval [{}]: timeout", self.prefix, request_id);
            }
        }
    }
}

pub struct CompositeHooks {
    hooks: Vec<Arc<dyn AgentHooks>>,
}

impl CompositeHooks {
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    pub fn add(mut self, hooks: Arc<dyn AgentHooks>) -> Self {
        self.hooks.push(hooks);
        self
    }

    pub fn with_hooks(hooks: Vec<Arc<dyn AgentHooks>>) -> Self {
        Self { hooks }
    }
}

impl Default for CompositeHooks {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AgentHooks for CompositeHooks {
    async fn on_message_received(&self, message: &str) {
        for hook in &self.hooks {
            hook.on_message_received(message).await;
        }
    }

    async fn on_llm_start(&self, messages: &[ChatMessage]) {
        for hook in &self.hooks {
            hook.on_llm_start(messages).await;
        }
    }

    async fn on_llm_complete(&self, response: &LLMResponse, duration_ms: u64) {
        for hook in &self.hooks {
            hook.on_llm_complete(response, duration_ms).await;
        }
    }

    async fn on_tool_start(&self, tool: &str, args: &Value) {
        for hook in &self.hooks {
            hook.on_tool_start(tool, args).await;
        }
    }

    async fn on_tool_complete(&self, tool: &str, result: &ToolResult, duration_ms: u64) {
        for hook in &self.hooks {
            hook.on_tool_complete(tool, result, duration_ms).await;
        }
    }

    async fn on_state_transition(&self, from: Option<&str>, to: &str, reason: &str) {
        for hook in &self.hooks {
            hook.on_state_transition(from, to, reason).await;
        }
    }

    async fn on_error(&self, error: &AgentError) {
        for hook in &self.hooks {
            hook.on_error(error).await;
        }
    }

    async fn on_response(&self, response: &AgentResponse) {
        for hook in &self.hooks {
            hook.on_response(response).await;
        }
    }

    async fn on_approval_requested(&self, request: &ApprovalRequest) {
        for hook in &self.hooks {
            hook.on_approval_requested(request).await;
        }
    }

    async fn on_approval_result(&self, request_id: &str, result: &ApprovalResult) {
        for hook in &self.hooks {
            hook.on_approval_result(request_id, result).await;
        }
    }
}

pub struct HookTimer {
    start: Instant,
}

impl HookTimer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    struct RecordingHooks {
        events: Arc<Mutex<Vec<String>>>,
    }

    impl RecordingHooks {
        fn new() -> Self {
            Self {
                events: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn events(&self) -> Vec<String> {
            self.events.lock().clone()
        }
    }

    #[async_trait]
    impl AgentHooks for RecordingHooks {
        async fn on_message_received(&self, message: &str) {
            self.events
                .lock()
                .push(format!("message_received:{}", message));
        }

        async fn on_llm_start(&self, messages: &[ChatMessage]) {
            self.events
                .lock()
                .push(format!("llm_start:{}", messages.len()));
        }

        async fn on_llm_complete(&self, _response: &LLMResponse, duration_ms: u64) {
            self.events
                .lock()
                .push(format!("llm_complete:{}", duration_ms));
        }

        async fn on_tool_start(&self, tool: &str, _args: &Value) {
            self.events.lock().push(format!("tool_start:{}", tool));
        }

        async fn on_tool_complete(&self, tool: &str, result: &ToolResult, duration_ms: u64) {
            self.events.lock().push(format!(
                "tool_complete:{}:{}:{}",
                tool, result.success, duration_ms
            ));
        }

        async fn on_state_transition(&self, from: Option<&str>, to: &str, reason: &str) {
            self.events
                .lock()
                .push(format!("state_transition:{:?}:{}:{}", from, to, reason));
        }

        async fn on_error(&self, error: &AgentError) {
            self.events.lock().push(format!("error:{}", error));
        }

        async fn on_response(&self, response: &AgentResponse) {
            self.events
                .lock()
                .push(format!("response:{}", response.content.len()));
        }

        async fn on_approval_requested(&self, request: &ApprovalRequest) {
            self.events
                .lock()
                .push(format!("approval_requested:{}", request.id));
        }

        async fn on_approval_result(&self, request_id: &str, result: &ApprovalResult) {
            let status = match result {
                ApprovalResult::Approved => "approved",
                ApprovalResult::Rejected { .. } => "rejected",
                ApprovalResult::Modified { .. } => "modified",
                ApprovalResult::Timeout => "timeout",
            };
            self.events
                .lock()
                .push(format!("approval_result:{}:{}", request_id, status));
        }
    }

    #[tokio::test]
    async fn test_noop_hooks() {
        let hooks = NoopHooks;
        hooks.on_message_received("test").await;
        hooks.on_llm_start(&[]).await;
    }

    #[tokio::test]
    async fn test_logging_hooks() {
        let hooks = LoggingHooks::new();
        hooks.on_message_received("test message").await;
        hooks.on_llm_start(&[ChatMessage::user("hello")]).await;
    }

    #[tokio::test]
    async fn test_recording_hooks() {
        let hooks = RecordingHooks::new();

        hooks.on_message_received("hello").await;
        hooks.on_llm_start(&[ChatMessage::user("test")]).await;

        let events = hooks.events();
        assert_eq!(events.len(), 2);
        assert!(events[0].contains("message_received"));
        assert!(events[1].contains("llm_start"));
    }

    #[tokio::test]
    async fn test_composite_hooks_with_vec() {
        let hooks1 = Arc::new(RecordingHooks::new());
        let hooks2 = Arc::new(RecordingHooks::new());

        let composite = CompositeHooks::with_hooks(vec![
            hooks1.clone() as Arc<dyn AgentHooks>,
            hooks2.clone() as Arc<dyn AgentHooks>,
        ]);

        composite
            .on_tool_start("calculator", &serde_json::json!({}))
            .await;

        assert_eq!(hooks1.events().len(), 1);
        assert_eq!(hooks2.events().len(), 1);
    }

    #[tokio::test]
    async fn test_hook_timer() {
        let timer = HookTimer::start();
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        let elapsed = timer.elapsed_ms();
        assert!(elapsed >= 10);
    }

    #[test]
    fn test_composite_hooks_default() {
        let hooks = CompositeHooks::default();
        assert!(hooks.hooks.is_empty());
    }
}

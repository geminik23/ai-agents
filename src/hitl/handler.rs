use async_trait::async_trait;
use std::sync::Arc;

use super::types::{ApprovalRequest, ApprovalResult};

#[async_trait]
pub trait ApprovalHandler: Send + Sync {
    async fn request_approval(&self, request: ApprovalRequest) -> ApprovalResult;

    fn preferred_language(&self) -> Option<String> {
        None
    }

    fn supported_languages(&self) -> Option<Vec<String>> {
        None
    }
}

pub struct RejectAllHandler;

impl RejectAllHandler {
    pub fn new() -> Self {
        Self
    }
}

impl Default for RejectAllHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ApprovalHandler for RejectAllHandler {
    async fn request_approval(&self, request: ApprovalRequest) -> ApprovalResult {
        tracing::warn!(
            "[HITL] Auto-rejecting: {} (no handler configured)",
            request.message
        );
        ApprovalResult::rejected_with_reason("No approval handler configured")
    }
}

// This is for testing purposes
pub struct AutoApproveHandler;

impl AutoApproveHandler {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AutoApproveHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ApprovalHandler for AutoApproveHandler {
    async fn request_approval(&self, request: ApprovalRequest) -> ApprovalResult {
        tracing::info!("[HITL] Auto-approving: {}", request.message);
        ApprovalResult::approved()
    }
}

pub struct CallbackHandler<F>
where
    F: Fn(ApprovalRequest) -> ApprovalResult + Send + Sync,
{
    callback: F,
}

impl<F> CallbackHandler<F>
where
    F: Fn(ApprovalRequest) -> ApprovalResult + Send + Sync,
{
    pub fn new(callback: F) -> Self {
        Self { callback }
    }
}

#[async_trait]
impl<F> ApprovalHandler for CallbackHandler<F>
where
    F: Fn(ApprovalRequest) -> ApprovalResult + Send + Sync,
{
    async fn request_approval(&self, request: ApprovalRequest) -> ApprovalResult {
        (self.callback)(request)
    }
}

pub struct LocalizedHandler {
    inner: Arc<dyn ApprovalHandler>,
    language: String,
    supported: Option<Vec<String>>,
}

impl LocalizedHandler {
    pub fn new(inner: Arc<dyn ApprovalHandler>, language: impl Into<String>) -> Self {
        Self {
            inner,
            language: language.into(),
            supported: None,
        }
    }

    pub fn with_supported(mut self, languages: Vec<String>) -> Self {
        self.supported = Some(languages);
        self
    }
}

#[async_trait]
impl ApprovalHandler for LocalizedHandler {
    async fn request_approval(&self, request: ApprovalRequest) -> ApprovalResult {
        self.inner.request_approval(request).await
    }

    fn preferred_language(&self) -> Option<String> {
        Some(self.language.clone())
    }

    fn supported_languages(&self) -> Option<Vec<String>> {
        self.supported.clone()
    }
}

pub fn create_handler<F>(callback: F) -> Arc<dyn ApprovalHandler>
where
    F: Fn(ApprovalRequest) -> ApprovalResult + Send + Sync + 'static,
{
    Arc::new(CallbackHandler::new(callback))
}

pub fn create_localized_handler<F>(
    callback: F,
    language: impl Into<String>,
) -> Arc<dyn ApprovalHandler>
where
    F: Fn(ApprovalRequest) -> ApprovalResult + Send + Sync + 'static,
{
    Arc::new(LocalizedHandler::new(
        Arc::new(CallbackHandler::new(callback)),
        language,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hitl::types::ApprovalTrigger;

    fn create_test_request() -> ApprovalRequest {
        ApprovalRequest::new(
            ApprovalTrigger::tool("test_tool", serde_json::json!({})),
            "Test approval",
        )
    }

    #[tokio::test]
    async fn test_reject_all_handler() {
        let handler = RejectAllHandler::new();
        let request = create_test_request();
        let result = handler.request_approval(request).await;
        assert!(result.is_rejected());
        assert!(handler.preferred_language().is_none());
        assert!(handler.supported_languages().is_none());
    }

    #[tokio::test]
    async fn test_auto_approve_handler() {
        let handler = AutoApproveHandler::new();
        let request = create_test_request();
        let result = handler.request_approval(request).await;
        assert!(result.is_approved());
    }

    #[tokio::test]
    async fn test_callback_handler() {
        let handler = CallbackHandler::new(|_| ApprovalResult::approved());
        let request = create_test_request();
        let result = handler.request_approval(request).await;
        assert!(result.is_approved());
    }

    #[tokio::test]
    async fn test_callback_handler_with_rejection() {
        let handler = CallbackHandler::new(|req| {
            if req.message.contains("dangerous") {
                ApprovalResult::rejected_with_reason("Dangerous operation")
            } else {
                ApprovalResult::approved()
            }
        });

        let safe_request = ApprovalRequest::new(
            ApprovalTrigger::tool("safe", serde_json::json!({})),
            "Safe operation",
        );
        let result = handler.request_approval(safe_request).await;
        assert!(result.is_approved());

        let dangerous_request = ApprovalRequest::new(
            ApprovalTrigger::tool("danger", serde_json::json!({})),
            "dangerous operation",
        );
        let result = handler.request_approval(dangerous_request).await;
        assert!(result.is_rejected());
    }

    #[tokio::test]
    async fn test_create_handler_helper() {
        let handler = create_handler(|_| ApprovalResult::approved());
        let request = create_test_request();
        let result = handler.request_approval(request).await;
        assert!(result.is_approved());
    }

    #[tokio::test]
    async fn test_localized_handler() {
        let inner = Arc::new(AutoApproveHandler::new());
        let handler = LocalizedHandler::new(inner, "ko")
            .with_supported(vec!["ko".to_string(), "en".to_string()]);

        assert_eq!(handler.preferred_language(), Some("ko".to_string()));
        assert_eq!(
            handler.supported_languages(),
            Some(vec!["ko".to_string(), "en".to_string()])
        );

        let request = create_test_request();
        let result = handler.request_approval(request).await;
        assert!(result.is_approved());
    }

    #[tokio::test]
    async fn test_create_localized_handler_helper() {
        let handler = create_localized_handler(|_| ApprovalResult::approved(), "ja");

        assert_eq!(handler.preferred_language(), Some("ja".to_string()));

        let request = create_test_request();
        let result = handler.request_approval(request).await;
        assert!(result.is_approved());
    }

    #[test]
    fn test_reject_all_default() {
        let handler = RejectAllHandler::default();
        assert!(std::mem::size_of_val(&handler) == 0);
    }

    #[test]
    fn test_auto_approve_default() {
        let handler = AutoApproveHandler::default();
        assert!(std::mem::size_of_val(&handler) == 0);
    }
}

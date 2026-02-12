use async_trait::async_trait;
use futures::stream;
use parking_lot::RwLock;
use std::sync::Arc;

use ai_agents_core::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
    TokenUsage,
};

/// Mock LLM provider for testing
#[derive(Clone)]
pub struct MockLLMProvider {
    inner: Arc<RwLock<MockLLMProviderInner>>,
}

struct MockLLMProviderInner {
    #[allow(dead_code)]
    name: String,
    responses: Vec<LLMResponse>,
    legacy_responses: Vec<String>,
    response_index: usize,
    cycle_responses: bool,
    call_history: Vec<MockCall>,
    should_error: bool,
    error_message: String,
    latency_ms: u64,
    features: Vec<LLMFeature>,
}

#[derive(Debug, Clone)]
pub struct MockCall {
    pub messages: Vec<ChatMessage>,
    pub config: Option<LLMConfig>,
    pub timestamp: std::time::Instant,
}

impl MockLLMProvider {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(MockLLMProviderInner {
                name: name.into(),
                responses: Vec::new(),
                legacy_responses: Vec::new(),
                response_index: 0,
                cycle_responses: false,
                call_history: Vec::new(),
                should_error: false,
                error_message: "Mock error".to_string(),
                latency_ms: 0,
                features: vec![LLMFeature::Streaming, LLMFeature::SystemMessages],
            })),
        }
    }

    pub fn add_response(&mut self, response: LLMResponse) {
        let mut inner = self.inner.write();
        inner.responses.push(response);
    }

    pub fn set_feature_support(&mut self, feature: LLMFeature, supported: bool) {
        let mut inner = self.inner.write();
        if supported {
            if !inner.features.contains(&feature) {
                inner.features.push(feature);
            }
        } else {
            inner.features.retain(|f| f != &feature);
        }
    }

    pub fn set_response(&mut self, response: impl Into<String>) {
        let mut inner = self.inner.write();
        inner.legacy_responses = vec![response.into()];
        inner.responses.clear();
        inner.response_index = 0;
    }

    pub fn set_responses(&mut self, responses: Vec<String>, cycle: bool) {
        let mut inner = self.inner.write();
        inner.legacy_responses = responses;
        inner.responses.clear();
        inner.response_index = 0;
        inner.cycle_responses = cycle;
    }

    pub fn set_error(&mut self, error_message: impl Into<String>) {
        let mut inner = self.inner.write();
        inner.should_error = true;
        inner.error_message = error_message.into();
    }

    pub fn clear_error(&mut self) {
        let mut inner = self.inner.write();
        inner.should_error = false;
    }

    pub fn set_latency(&mut self, latency_ms: u64) {
        let mut inner = self.inner.write();
        inner.latency_ms = latency_ms;
    }

    pub fn call_count(&self) -> usize {
        self.inner.read().call_history.len()
    }

    pub fn call_history(&self) -> Vec<MockCall> {
        self.inner.read().call_history.clone()
    }

    pub fn last_call(&self) -> Option<MockCall> {
        self.inner.read().call_history.last().cloned()
    }

    pub fn clear_history(&mut self) {
        let mut inner = self.inner.write();
        inner.call_history.clear();
    }

    pub fn reset(&mut self) {
        let mut inner = self.inner.write();
        inner.responses.clear();
        inner.legacy_responses.clear();
        inner.response_index = 0;
        inner.cycle_responses = false;
        inner.call_history.clear();
        inner.should_error = false;
        inner.error_message = "Mock error".to_string();
        inner.latency_ms = 0;
    }

    fn get_next_response(&self) -> LLMResponse {
        let mut inner = self.inner.write();

        if !inner.responses.is_empty() {
            let response = inner.responses[inner.response_index].clone();

            if inner.cycle_responses {
                inner.response_index = (inner.response_index + 1) % inner.responses.len();
            } else if inner.response_index < inner.responses.len() - 1 {
                inner.response_index += 1;
            }

            return response;
        }

        if !inner.legacy_responses.is_empty() {
            let content = inner.legacy_responses[inner.response_index].clone();

            if inner.cycle_responses {
                inner.response_index = (inner.response_index + 1) % inner.legacy_responses.len();
            } else if inner.response_index < inner.legacy_responses.len() - 1 {
                inner.response_index += 1;
            }

            return LLMResponse {
                content,
                finish_reason: FinishReason::Stop,
                usage: None,
                model: Some("mock-model".to_string()),
                metadata: std::collections::HashMap::new(),
            };
        }

        LLMResponse {
            content: "Mock response".to_string(),
            finish_reason: FinishReason::Stop,
            usage: None,
            model: Some("mock-model".to_string()),
            metadata: std::collections::HashMap::new(),
        }
    }

    fn record_call(&self, messages: &[ChatMessage], config: Option<&LLMConfig>) {
        let mut inner = self.inner.write();
        inner.call_history.push(MockCall {
            messages: messages.to_vec(),
            config: config.cloned(),
            timestamp: std::time::Instant::now(),
        });
    }

    /// Simulate latency if configured
    async fn simulate_latency(&self) {
        let latency_ms = self.inner.read().latency_ms;
        if latency_ms > 0 {
            tokio::time::sleep(tokio::time::Duration::from_millis(latency_ms)).await;
        }
    }

    fn estimate_tokens(messages: &[ChatMessage]) -> u32 {
        let total_chars: usize = messages.iter().map(|m| m.content.len()).sum();
        (total_chars / 4) as u32
    }
}

impl Default for MockLLMProvider {
    fn default() -> Self {
        Self::new("default")
    }
}

#[async_trait]
impl LLMProvider for MockLLMProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<LLMResponse, LLMError> {
        self.record_call(messages, config);
        self.simulate_latency().await;

        let should_error = self.inner.read().should_error;
        if should_error {
            let error_message = self.inner.read().error_message.clone();
            return Err(LLMError::Other(error_message));
        }

        let mut response = self.get_next_response();

        if response.usage.is_none() {
            let prompt_tokens = Self::estimate_tokens(messages);
            let completion_tokens = (response.content.len() / 4) as u32;
            response.usage = Some(TokenUsage::new(prompt_tokens, completion_tokens));
        }

        Ok(response)
    }

    async fn complete_stream(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError>
    {
        self.record_call(messages, config);
        self.simulate_latency().await;

        let should_error = self.inner.read().should_error;
        if should_error {
            let error_message = self.inner.read().error_message.clone();
            return Err(LLMError::Other(error_message));
        }

        let response = self.get_next_response();
        let content = response.content;

        let words: Vec<&str> = content.split_whitespace().collect();
        let mut chunks = Vec::new();

        for (i, word) in words.iter().enumerate() {
            let delta = if i == 0 {
                word.to_string()
            } else {
                format!(" {}", word)
            };

            let is_final = i == words.len() - 1;
            let chunk = if is_final {
                let usage = if let Some(u) = response.usage {
                    Some(u)
                } else {
                    let prompt_tokens = Self::estimate_tokens(messages);
                    let completion_tokens = (content.len() / 4) as u32;
                    Some(TokenUsage::new(prompt_tokens, completion_tokens))
                };
                LLMChunk::final_chunk(delta, response.finish_reason.clone(), usage)
            } else {
                LLMChunk::new(delta, false)
            };

            chunks.push(Ok(chunk));
        }

        Ok(Box::new(stream::iter(chunks)))
    }

    fn provider_name(&self) -> &str {
        "mock"
    }

    fn supports(&self, feature: LLMFeature) -> bool {
        self.inner.read().features.contains(&feature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_provider_basic() {
        let mut mock = MockLLMProvider::new("test");

        mock.set_response("Mock response");
        let messages = vec![ChatMessage::user("Hello")];
        let response = mock.complete(&messages, None).await.unwrap();
        assert_eq!(response.content, "Mock response");
        assert_eq!(response.finish_reason, FinishReason::Stop);

        mock.add_response(LLMResponse::new("Added response", FinishReason::Stop));
        let response2 = mock.complete(&messages, None).await.unwrap();
        assert_eq!(response2.content, "Added response");
        assert_eq!(response2.finish_reason, FinishReason::Stop);
    }

    #[tokio::test]
    async fn test_multiple_responses() {
        let mut mock = MockLLMProvider::new("test");
        mock.set_responses(
            vec![
                "First".to_string(),
                "Second".to_string(),
                "Third".to_string(),
            ],
            false,
        );

        let messages = vec![ChatMessage::user("Hello")];

        let r1 = mock.complete(&messages, None).await.unwrap();
        assert_eq!(r1.content, "First");

        let r2 = mock.complete(&messages, None).await.unwrap();
        assert_eq!(r2.content, "Second");

        let r3 = mock.complete(&messages, None).await.unwrap();
        assert_eq!(r3.content, "Third");

        let r4 = mock.complete(&messages, None).await.unwrap();
        assert_eq!(r4.content, "Third");
    }

    #[tokio::test]
    async fn test_cycle_responses() {
        let mut mock = MockLLMProvider::new("test");
        mock.set_responses(vec!["A".to_string(), "B".to_string()], true);

        let messages = vec![ChatMessage::user("Hello")];

        assert_eq!(mock.complete(&messages, None).await.unwrap().content, "A");
        assert_eq!(mock.complete(&messages, None).await.unwrap().content, "B");
        assert_eq!(mock.complete(&messages, None).await.unwrap().content, "A");
        assert_eq!(mock.complete(&messages, None).await.unwrap().content, "B");
    }

    #[tokio::test]
    async fn test_error_handling() {
        let mut mock = MockLLMProvider::new("test");
        mock.set_error("Test error");

        let messages = vec![ChatMessage::user("Hello")];
        let result = mock.complete(&messages, None).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Test error"));
    }

    #[tokio::test]
    async fn test_clear_error() {
        let mut mock = MockLLMProvider::new("test");
        mock.set_error("Test error");
        mock.clear_error();

        let messages = vec![ChatMessage::user("Hello")];
        let result = mock.complete(&messages, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_call_history() {
        let mut mock = MockLLMProvider::new("test");
        mock.set_response("test");
        assert_eq!(mock.call_count(), 0);

        let messages1 = vec![ChatMessage::user("First")];
        mock.complete(&messages1, None).await.unwrap();
        assert_eq!(mock.call_count(), 1);

        let messages2 = vec![ChatMessage::user("Second")];
        mock.complete(&messages2, None).await.unwrap();
        assert_eq!(mock.call_count(), 2);

        let history = mock.call_history();
        assert_eq!(history.len(), 2);

        let last = mock.last_call().unwrap();
        assert_eq!(last.messages[0].content, "Second");
    }

    #[tokio::test]
    async fn test_clear_history() {
        let mut mock = MockLLMProvider::new("test");
        mock.set_response("test");

        let messages = vec![ChatMessage::user("Hello")];
        mock.complete(&messages, None).await.unwrap();
        assert_eq!(mock.call_count(), 1);

        mock.clear_history();
        assert_eq!(mock.call_count(), 0);
    }

    #[tokio::test]
    async fn test_reset() {
        let mut mock = MockLLMProvider::new("test");
        mock.set_response("Custom");
        mock.set_error("Error");
        mock.set_latency(100);

        let messages = vec![ChatMessage::user("Hello")];
        let _ = mock.complete(&messages, None).await;

        mock.reset();
        mock.set_response("Mock response");

        assert_eq!(mock.call_count(), 0);
        let result = mock.complete(&messages, None).await.unwrap();
        assert_eq!(result.content, "Mock response");
    }

    #[tokio::test]
    async fn test_token_estimation() {
        let mut mock = MockLLMProvider::new("test");
        mock.set_response("test");
        let messages = vec![
            ChatMessage::user("Hello world"),
            ChatMessage::assistant("Hi there"),
        ];

        let response = mock.complete(&messages, None).await.unwrap();
        assert!(response.usage.is_some());

        let usage = response.usage.unwrap();
        assert!(usage.prompt_tokens > 0);
        assert!(usage.completion_tokens > 0);
        assert_eq!(
            usage.total_tokens,
            usage.prompt_tokens + usage.completion_tokens
        );
    }

    #[tokio::test]
    async fn test_streaming() {
        use futures::StreamExt;

        let mut mock = MockLLMProvider::new("test");
        mock.set_response("Hello world test");

        let messages = vec![ChatMessage::user("Hi")];
        let stream = mock.complete_stream(&messages, None).await.unwrap();

        let chunks: Vec<_> = stream.collect().await;
        assert!(!chunks.is_empty());

        for chunk in &chunks {
            assert!(chunk.is_ok());
        }

        let last = chunks.last().unwrap().as_ref().unwrap();
        assert!(last.is_final);
    }
}

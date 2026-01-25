//! Summarizer trait and implementations for memory compression

use std::sync::Arc;

use async_trait::async_trait;

use ai_agents_core::{ChatMessage, LLMProvider, Result, Role};

#[async_trait]
pub trait Summarizer: Send + Sync {
    async fn summarize(&self, messages: &[ChatMessage]) -> Result<String>;

    fn max_batch_size(&self) -> usize {
        20
    }

    async fn merge_summaries(&self, summaries: &[String]) -> Result<String> {
        Ok(summaries.join("\n\n"))
    }
}

pub struct LLMSummarizer {
    llm: Arc<dyn LLMProvider>,
    prompt_template: String,
    merge_prompt_template: String,
    max_batch_size: usize,
}

impl LLMSummarizer {
    pub fn new(llm: Arc<dyn LLMProvider>) -> Self {
        Self {
            llm,
            prompt_template: DEFAULT_SUMMARY_PROMPT.to_string(),
            merge_prompt_template: DEFAULT_MERGE_PROMPT.to_string(),
            max_batch_size: 20,
        }
    }

    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt_template = prompt.into();
        self
    }

    pub fn with_merge_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.merge_prompt_template = prompt.into();
        self
    }

    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size.max(1);
        self
    }

    fn format_messages(&self, messages: &[ChatMessage]) -> String {
        messages
            .iter()
            .map(|m| format!("{}: {}", format_role(&m.role), m.content))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn format_role(role: &Role) -> &'static str {
    match role {
        Role::System => "System",
        Role::User => "User",
        Role::Assistant => "Assistant",
        Role::Tool => "Tool",
        Role::Function => "Function",
    }
}

#[async_trait]
impl Summarizer for LLMSummarizer {
    async fn summarize(&self, messages: &[ChatMessage]) -> Result<String> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        let conversation = self.format_messages(messages);
        let prompt = self
            .prompt_template
            .replace("{conversation}", &conversation);

        let llm_messages = vec![ChatMessage::user(&prompt)];

        let response = self.llm.complete(&llm_messages, None).await?;
        Ok(response.content.trim().to_string())
    }

    fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    async fn merge_summaries(&self, summaries: &[String]) -> Result<String> {
        if summaries.is_empty() {
            return Ok(String::new());
        }

        if summaries.len() == 1 {
            return Ok(summaries[0].clone());
        }

        let combined = summaries.join("\n---\n");
        let prompt = self.merge_prompt_template.replace("{summaries}", &combined);

        let llm_messages = vec![ChatMessage::user(&prompt)];

        let response = self.llm.complete(&llm_messages, None).await?;
        Ok(response.content.trim().to_string())
    }
}

pub const DEFAULT_SUMMARY_PROMPT: &str = r#"Summarize the following conversation concisely, preserving key information, decisions, and context that would be important for continuing the conversation:

{conversation}

Summary:"#;

pub const DEFAULT_MERGE_PROMPT: &str = r#"Merge the following conversation summaries into a single coherent summary, preserving all important information:

{summaries}

Merged Summary:"#;

pub struct NoopSummarizer;

#[async_trait]
impl Summarizer for NoopSummarizer {
    async fn summarize(&self, messages: &[ChatMessage]) -> Result<String> {
        Ok(messages
            .iter()
            .map(|m| m.content.clone())
            .collect::<Vec<_>>()
            .join(" | "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_agents_core::{FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMResponse};
    use parking_lot::Mutex;

    struct MockLLMProvider {
        responses: Mutex<Vec<String>>,
    }

    impl MockLLMProvider {
        fn new(responses: Vec<String>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for MockLLMProvider {
        async fn complete(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> std::result::Result<LLMResponse, LLMError> {
            let response = self
                .responses
                .lock()
                .pop()
                .unwrap_or_else(|| "Summary of conversation".to_string());
            Ok(LLMResponse::new(response, FinishReason::Stop))
        }

        async fn complete_stream(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> std::result::Result<
            Box<dyn futures::Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
            LLMError,
        > {
            Err(LLMError::Other(
                "Streaming not supported in mock".to_string(),
            ))
        }

        fn provider_name(&self) -> &str {
            "mock"
        }

        fn supports(&self, _feature: LLMFeature) -> bool {
            true
        }
    }

    fn make_message(role: Role, content: &str) -> ChatMessage {
        ChatMessage {
            role,
            content: content.to_string(),
            name: None,
            timestamp: None,
        }
    }

    #[tokio::test]
    async fn test_llm_summarizer_basic() {
        let provider = Arc::new(MockLLMProvider::new(vec!["Test summary".to_string()]));
        let summarizer = LLMSummarizer::new(provider);

        let messages = vec![
            make_message(Role::User, "Hello"),
            make_message(Role::Assistant, "Hi there!"),
        ];

        let summary = summarizer.summarize(&messages).await.unwrap();
        assert_eq!(summary, "Test summary");
    }

    #[tokio::test]
    async fn test_llm_summarizer_empty_messages() {
        let provider = Arc::new(MockLLMProvider::new(vec![]));
        let summarizer = LLMSummarizer::new(provider);

        let summary = summarizer.summarize(&[]).await.unwrap();
        assert!(summary.is_empty());
    }

    #[tokio::test]
    async fn test_llm_summarizer_custom_prompt() {
        let provider = Arc::new(MockLLMProvider::new(vec!["Custom summary".to_string()]));
        let summarizer = LLMSummarizer::new(provider).with_prompt("Custom prompt: {conversation}");

        let messages = vec![make_message(Role::User, "Test")];
        let summary = summarizer.summarize(&messages).await.unwrap();
        assert_eq!(summary, "Custom summary");
    }

    #[tokio::test]
    async fn test_merge_summaries() {
        let provider = Arc::new(MockLLMProvider::new(vec!["Merged summary".to_string()]));
        let summarizer = LLMSummarizer::new(provider);

        let summaries = vec!["Summary 1".to_string(), "Summary 2".to_string()];
        let merged = summarizer.merge_summaries(&summaries).await.unwrap();
        assert_eq!(merged, "Merged summary");
    }

    #[tokio::test]
    async fn test_merge_single_summary() {
        let provider = Arc::new(MockLLMProvider::new(vec![]));
        let summarizer = LLMSummarizer::new(provider);

        let summaries = vec!["Only summary".to_string()];
        let merged = summarizer.merge_summaries(&summaries).await.unwrap();
        assert_eq!(merged, "Only summary");
    }

    #[tokio::test]
    async fn test_noop_summarizer() {
        let summarizer = NoopSummarizer;

        let messages = vec![
            make_message(Role::User, "Hello"),
            make_message(Role::Assistant, "Hi"),
        ];

        let summary = summarizer.summarize(&messages).await.unwrap();
        assert!(summary.contains("Hello"));
        assert!(summary.contains("Hi"));
    }

    #[test]
    fn test_max_batch_size() {
        let provider = Arc::new(MockLLMProvider::new(vec![]));
        let summarizer = LLMSummarizer::new(provider).with_batch_size(10);
        assert_eq!(summarizer.max_batch_size(), 10);
    }

    #[test]
    fn test_format_role() {
        assert_eq!(format_role(&Role::User), "User");
        assert_eq!(format_role(&Role::Assistant), "Assistant");
        assert_eq!(format_role(&Role::System), "System");
        assert_eq!(format_role(&Role::Tool), "Tool");
        assert_eq!(format_role(&Role::Function), "Function");
    }
}

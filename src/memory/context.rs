//! Conversation context for memory management

use serde::{Deserialize, Serialize};

use crate::llm::{ChatMessage, Role};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConversationContext {
    pub summary: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub total_messages: usize,
    pub summarized_count: usize,
}

impl ConversationContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_messages(messages: Vec<ChatMessage>) -> Self {
        let total = messages.len();
        Self {
            summary: None,
            messages,
            total_messages: total,
            summarized_count: 0,
        }
    }

    pub fn with_summary(mut self, summary: String, summarized_count: usize) -> Self {
        self.summary = Some(summary);
        self.summarized_count = summarized_count;
        self
    }

    pub fn to_llm_messages(&self) -> Vec<ChatMessage> {
        let mut result = Vec::new();

        if let Some(ref summary) = self.summary {
            result.push(ChatMessage {
                role: Role::System,
                content: format!("[Previous conversation summary]\n{}", summary),
                name: None,
                timestamp: None,
            });
        }

        result.extend(self.messages.clone());
        result
    }

    pub fn to_llm_messages_with_budget(&self, max_tokens: u32) -> Vec<ChatMessage> {
        let mut result = Vec::new();
        let mut used_tokens = 0u32;

        if let Some(ref summary) = self.summary {
            let summary_msg = ChatMessage {
                role: Role::System,
                content: format!("[Previous conversation summary]\n{}", summary),
                name: None,
                timestamp: None,
            };
            let tokens = estimate_message_tokens(&summary_msg);
            if tokens <= max_tokens {
                used_tokens = tokens;
                result.push(summary_msg);
            }
        }

        let mut messages_to_add: Vec<&ChatMessage> = Vec::new();
        for msg in self.messages.iter().rev() {
            let tokens = estimate_message_tokens(msg);
            if used_tokens + tokens <= max_tokens {
                used_tokens += tokens;
                messages_to_add.push(msg);
            } else {
                break;
            }
        }

        messages_to_add.reverse();
        for msg in messages_to_add {
            result.push(msg.clone());
        }

        result
    }

    pub fn estimated_tokens(&self) -> u32 {
        let summary_tokens = self
            .summary
            .as_ref()
            .map(|s| estimate_tokens(s))
            .unwrap_or(0);

        let message_tokens: u32 = self.messages.iter().map(estimate_message_tokens).sum();

        summary_tokens + message_tokens
    }

    pub fn is_empty(&self) -> bool {
        self.summary.is_none() && self.messages.is_empty()
    }

    pub fn message_count(&self) -> usize {
        self.messages.len()
    }
}

/// Language-aware token estimation for multi-language support
pub fn estimate_tokens(text: &str) -> u32 {
    if text.is_empty() {
        return 0;
    }

    let ascii_chars = text.chars().filter(|c| c.is_ascii()).count();
    let cjk_chars = text.chars().filter(|c| is_cjk(*c)).count();
    let other_chars = text.chars().count() - ascii_chars - cjk_chars;

    let estimated =
        (ascii_chars as f64 / 4.0) + (cjk_chars as f64 * 1.5) + (other_chars as f64 * 1.0);

    estimated.ceil().max(1.0) as u32
}

fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}' |   // CJK Unified Ideographs
        '\u{3400}'..='\u{4DBF}' |   // CJK Extension A
        '\u{AC00}'..='\u{D7AF}' |   // Korean Hangul
        '\u{3040}'..='\u{30FF}' |   // Japanese Hiragana/Katakana
        '\u{31F0}'..='\u{31FF}'     // Katakana Extensions
    )
}

pub fn estimate_message_tokens(message: &ChatMessage) -> u32 {
    let role_tokens = 4u32;
    let content_tokens = estimate_tokens(&message.content);
    let name_tokens = message
        .name
        .as_ref()
        .map(|n| estimate_tokens(n))
        .unwrap_or(0);
    role_tokens + content_tokens + name_tokens
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressResult {
    NotNeeded,
    Compressed {
        messages_summarized: usize,
        new_summary_length: usize,
        tokens_saved: u32,
    },
    AlreadyCompressed,
    Failed {
        error: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_message(role: Role, content: &str) -> ChatMessage {
        ChatMessage {
            role,
            content: content.to_string(),
            name: None,
            timestamp: None,
        }
    }

    #[test]
    fn test_conversation_context_new() {
        let ctx = ConversationContext::new();
        assert!(ctx.is_empty());
        assert_eq!(ctx.message_count(), 0);
        assert!(ctx.summary.is_none());
    }

    #[test]
    fn test_conversation_context_with_messages() {
        let messages = vec![
            make_message(Role::User, "Hello"),
            make_message(Role::Assistant, "Hi there!"),
        ];
        let ctx = ConversationContext::with_messages(messages);
        assert_eq!(ctx.message_count(), 2);
        assert_eq!(ctx.total_messages, 2);
        assert!(!ctx.is_empty());
    }

    #[test]
    fn test_conversation_context_with_summary() {
        let messages = vec![make_message(Role::User, "Current message")];
        let ctx = ConversationContext::with_messages(messages)
            .with_summary("Previous discussion about weather".to_string(), 5);

        assert!(ctx.summary.is_some());
        assert_eq!(ctx.summarized_count, 5);

        let llm_messages = ctx.to_llm_messages();
        assert_eq!(llm_messages.len(), 2);
        assert!(
            llm_messages[0]
                .content
                .contains("Previous conversation summary")
        );
    }

    #[test]
    fn test_to_llm_messages_without_summary() {
        let messages = vec![
            make_message(Role::User, "Hello"),
            make_message(Role::Assistant, "Hi!"),
        ];
        let ctx = ConversationContext::with_messages(messages);

        let llm_messages = ctx.to_llm_messages();
        assert_eq!(llm_messages.len(), 2);
        assert_eq!(llm_messages[0].role, Role::User);
    }

    #[test]
    fn test_estimated_tokens() {
        let ctx = ConversationContext::with_messages(vec![
            make_message(Role::User, "Hello world"),
            make_message(Role::Assistant, "Hi there"),
        ]);

        let tokens = ctx.estimated_tokens();
        assert!(tokens > 0);
    }

    #[test]
    fn test_to_llm_messages_with_budget() {
        let messages: Vec<ChatMessage> = (0..10)
            .map(|i| make_message(Role::User, &format!("Message number {}", i)))
            .collect();
        let ctx = ConversationContext::with_messages(messages);

        let limited = ctx.to_llm_messages_with_budget(50);
        assert!(limited.len() < 10);
    }

    #[test]
    fn test_estimate_tokens_english() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("test"), 1);
        assert_eq!(estimate_tokens("hello world"), 3);
    }

    #[test]
    fn test_estimate_tokens_korean() {
        let tokens = estimate_tokens("안녕하세요");
        assert!(
            tokens >= 5,
            "Korean text should have more tokens: {}",
            tokens
        );
    }

    #[test]
    fn test_estimate_tokens_japanese() {
        let tokens = estimate_tokens("こんにちは");
        assert!(
            tokens >= 5,
            "Japanese text should have more tokens: {}",
            tokens
        );
    }

    #[test]
    fn test_estimate_tokens_chinese() {
        let tokens = estimate_tokens("你好世界");
        assert!(
            tokens >= 4,
            "Chinese text should have more tokens: {}",
            tokens
        );
    }

    #[test]
    fn test_estimate_tokens_mixed() {
        let tokens = estimate_tokens("Hello 안녕 World 世界");
        assert!(tokens >= 6, "Mixed text: {}", tokens);
    }

    #[test]
    fn test_compress_result_variants() {
        let not_needed = CompressResult::NotNeeded;
        assert!(matches!(not_needed, CompressResult::NotNeeded));

        let compressed = CompressResult::Compressed {
            messages_summarized: 5,
            new_summary_length: 100,
            tokens_saved: 500,
        };
        if let CompressResult::Compressed {
            messages_summarized,
            ..
        } = compressed
        {
            assert_eq!(messages_summarized, 5);
        }

        let failed = CompressResult::Failed {
            error: "test error".to_string(),
        };
        if let CompressResult::Failed { error } = failed {
            assert_eq!(error, "test error");
        }
    }
}

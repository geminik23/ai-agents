//! Conversation context for memory management

use serde::{Deserialize, Serialize};

use super::token_budget::TokenAllocation;
use ai_agents_core::{ChatMessage, Role};

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

    /// Build LLM messages with per-component token budgets.
    pub fn to_llm_messages_with_allocation(
        &self,
        allocation: &TokenAllocation,
    ) -> Vec<ChatMessage> {
        let mut result = Vec::new();

        // Summary - capped to allocation.summary tokens
        if let Some(ref summary) = self.summary {
            let summary_content = format!("[Previous conversation summary]\n{}", summary);
            let summary_tokens = estimate_tokens(&summary_content);

            let final_content = if summary_tokens > allocation.summary {
                let ratio = summary_content.len() as f64 / summary_tokens as f64;
                let target_chars = (allocation.summary as f64 * ratio) as usize;
                let truncated = &summary_content[..target_chars.min(summary_content.len())];
                format!("{}...", truncated)
            } else {
                summary_content
            };

            result.push(ChatMessage {
                role: Role::System,
                content: final_content,
                name: None,
                timestamp: None,
            });
        }

        // Recent messages - capped to allocation.recent_messages tokens
        let mut used_message_tokens = 0u32;
        let mut messages_to_add: Vec<&ChatMessage> = Vec::new();

        for msg in self.messages.iter().rev() {
            let tokens = estimate_message_tokens(msg);
            if used_message_tokens + tokens <= allocation.recent_messages {
                used_message_tokens += tokens;
                messages_to_add.push(msg);
            } else {
                break;
            }
        }

        messages_to_add.reverse();
        for msg in messages_to_add {
            result.push(msg.clone());
        }

        // TODO:
        // Facts - reserved for 'Session Management' feature, not injected yet.

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
    fn test_to_llm_messages_with_allocation_caps_summary() {
        let long_summary = "x".repeat(10000); // ~2500 tokens
        let messages = vec![make_message(Role::User, "Hello")];
        let ctx = ConversationContext::with_messages(messages).with_summary(long_summary, 50);

        let allocation = TokenAllocation {
            summary: 100,
            recent_messages: 2048,
            facts: 512,
        };

        let result = ctx.to_llm_messages_with_allocation(&allocation);
        // Summary should be truncated
        let summary_msg = &result[0];
        let summary_tokens = estimate_tokens(&summary_msg.content);
        assert!(
            summary_tokens <= 120,
            "Summary should be roughly capped: got {}",
            summary_tokens
        );
        // Recent message should still be present
        assert!(result.len() >= 2);
    }

    #[test]
    fn test_to_llm_messages_with_allocation_caps_recent() {
        let messages: Vec<ChatMessage> = (0..50)
            .map(|i| {
                make_message(
                    Role::User,
                    &format!(
                        "Message number {} with some extra text to increase tokens",
                        i
                    ),
                )
            })
            .collect();
        let ctx = ConversationContext::with_messages(messages);

        let allocation = TokenAllocation {
            summary: 1024,
            recent_messages: 200,
            facts: 512,
        };

        let result = ctx.to_llm_messages_with_allocation(&allocation);
        assert!(
            result.len() < 50,
            "Should have fewer messages due to cap: got {}",
            result.len()
        );
        // Messages should be the most recent
        let last = &result[result.len() - 1];
        assert!(
            last.content.contains("49"),
            "Last message should be the most recent"
        );
    }

    #[test]
    fn test_to_llm_messages_with_allocation_no_summary() {
        let messages = vec![
            make_message(Role::User, "Hello"),
            make_message(Role::Assistant, "Hi!"),
        ];
        let ctx = ConversationContext::with_messages(messages);

        let allocation = TokenAllocation {
            summary: 1024,
            recent_messages: 2048,
            facts: 512,
        };

        let result = ctx.to_llm_messages_with_allocation(&allocation);
        assert_eq!(result.len(), 2);
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

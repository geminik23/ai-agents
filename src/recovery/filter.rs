use crate::llm::ChatMessage;

/// Trait for custom message filtering logic during context overflow summarization
pub trait MessageFilter: Send + Sync {
    /// Filter messages into two groups: to_summarize and to_keep
    fn filter(
        &self,
        messages: &[ChatMessage],
        keep_recent: usize,
    ) -> (Vec<ChatMessage>, Vec<ChatMessage>);

    fn name(&self) -> &str {
        "custom"
    }
}

/// Default filter: Keep N most recent messages, summarize older ones
#[derive(Debug, Default)]
pub struct KeepRecentFilter;

impl MessageFilter for KeepRecentFilter {
    fn filter(
        &self,
        messages: &[ChatMessage],
        keep_recent: usize,
    ) -> (Vec<ChatMessage>, Vec<ChatMessage>) {
        if messages.len() <= keep_recent {
            return (vec![], messages.to_vec());
        }
        let split = messages.len() - keep_recent;
        (messages[..split].to_vec(), messages[split..].to_vec())
    }

    fn name(&self) -> &str {
        "keep_recent"
    }
}

/// Filter by message role - keep important roles, summarize others
#[derive(Debug, Clone)]
pub struct ByRoleFilter {
    pub keep_roles: Vec<String>,
}

impl MessageFilter for ByRoleFilter {
    fn filter(
        &self,
        messages: &[ChatMessage],
        keep_recent: usize,
    ) -> (Vec<ChatMessage>, Vec<ChatMessage>) {
        let mut to_summarize = Vec::new();
        let mut to_keep = Vec::new();
        let recent_start = messages.len().saturating_sub(keep_recent);

        for (i, msg) in messages.iter().enumerate() {
            if i >= recent_start {
                to_keep.push(msg.clone());
            } else if self
                .keep_roles
                .iter()
                .any(|r| r == &format!("{:?}", msg.role).to_lowercase())
            {
                to_keep.push(msg.clone());
            } else {
                to_summarize.push(msg.clone());
            }
        }
        (to_summarize, to_keep)
    }

    fn name(&self) -> &str {
        "by_role"
    }
}

/// Skip messages containing certain patterns (exclude from summary entirely)
#[derive(Debug, Clone)]
pub struct SkipPatternFilter {
    pub skip_if_contains: Vec<String>,
}

impl MessageFilter for SkipPatternFilter {
    fn filter(
        &self,
        messages: &[ChatMessage],
        keep_recent: usize,
    ) -> (Vec<ChatMessage>, Vec<ChatMessage>) {
        let mut to_summarize = Vec::new();
        let mut to_keep = Vec::new();
        let recent_start = messages.len().saturating_sub(keep_recent);

        for (i, msg) in messages.iter().enumerate() {
            let should_skip = self
                .skip_if_contains
                .iter()
                .any(|pattern| msg.content.contains(pattern));

            if i >= recent_start {
                to_keep.push(msg.clone());
            } else if should_skip {
                continue;
            } else {
                to_summarize.push(msg.clone());
            }
        }
        (to_summarize, to_keep)
    }

    fn name(&self) -> &str {
        "skip_pattern"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Role;

    fn msg(role: Role, content: &str) -> ChatMessage {
        ChatMessage {
            role,
            content: content.to_string(),
            name: None,
            timestamp: None,
        }
    }

    #[test]
    fn test_keep_recent() {
        let messages = vec![
            msg(Role::User, "1"),
            msg(Role::Assistant, "2"),
            msg(Role::User, "3"),
            msg(Role::Assistant, "4"),
            msg(Role::User, "5"),
        ];
        let filter = KeepRecentFilter;
        let (summarize, keep) = filter.filter(&messages, 2);
        assert_eq!(summarize.len(), 3);
        assert_eq!(keep.len(), 2);
        assert_eq!(keep[0].content, "4");
        assert_eq!(keep[1].content, "5");
    }

    #[test]
    fn test_by_role() {
        let messages = vec![
            msg(Role::User, "u1"),
            msg(Role::Function, "f1"),
            msg(Role::Assistant, "a1"),
            msg(Role::Function, "f2"),
            msg(Role::User, "u2"),
        ];
        let filter = ByRoleFilter {
            keep_roles: vec!["user".to_string(), "assistant".to_string()],
        };
        let (summarize, keep) = filter.filter(&messages, 1);
        assert_eq!(summarize.len(), 2);
        assert_eq!(keep.len(), 3);
    }

    #[test]
    fn test_skip_pattern() {
        let messages = vec![
            msg(Role::User, "normal"),
            msg(Role::Assistant, "[DEBUG] verbose"),
            msg(Role::User, "another"),
            msg(Role::Function, "[TOOL] output"),
            msg(Role::User, "recent"),
        ];
        let filter = SkipPatternFilter {
            skip_if_contains: vec!["[DEBUG]".to_string(), "[TOOL]".to_string()],
        };
        let (summarize, keep) = filter.filter(&messages, 1);
        assert_eq!(summarize.len(), 2);
        assert_eq!(keep.len(), 1);
    }
}

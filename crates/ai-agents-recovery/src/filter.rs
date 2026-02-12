use ai_agents_core::ChatMessage;

pub trait MessageFilter: Send + Sync {
    fn filter(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage>;
    fn name(&self) -> &str {
        "custom"
    }
}

#[derive(Debug)]
pub struct KeepRecentFilter {
    pub keep_count: usize,
}

impl KeepRecentFilter {
    pub fn new(keep_count: usize) -> Self {
        Self { keep_count }
    }
}

impl Default for KeepRecentFilter {
    fn default() -> Self {
        Self { keep_count: 10 }
    }
}

impl MessageFilter for KeepRecentFilter {
    fn filter(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        if messages.len() <= self.keep_count {
            return messages;
        }
        messages
            .into_iter()
            .rev()
            .take(self.keep_count)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    fn name(&self) -> &str {
        "keep_recent"
    }
}

#[derive(Debug, Clone)]
pub struct ByRoleFilter {
    pub keep_roles: Vec<String>,
}

impl ByRoleFilter {
    pub fn new(keep_roles: Vec<String>) -> Self {
        Self { keep_roles }
    }
}

impl MessageFilter for ByRoleFilter {
    fn filter(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        messages
            .into_iter()
            .filter(|msg| {
                let role_str = format!("{:?}", msg.role).to_lowercase();
                self.keep_roles.iter().any(|r| r.to_lowercase() == role_str)
            })
            .collect()
    }

    fn name(&self) -> &str {
        "by_role"
    }
}

#[derive(Debug, Clone)]
pub struct SkipPatternFilter {
    pub skip_if_contains: Vec<String>,
}

impl SkipPatternFilter {
    pub fn new(skip_if_contains: Vec<String>) -> Self {
        Self { skip_if_contains }
    }
}

impl MessageFilter for SkipPatternFilter {
    fn filter(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        messages
            .into_iter()
            .filter(|msg| {
                !self
                    .skip_if_contains
                    .iter()
                    .any(|pattern| msg.content.contains(pattern))
            })
            .collect()
    }

    fn name(&self) -> &str {
        "skip_pattern"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_agents_core::Role;

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
        let filter = KeepRecentFilter::new(2);
        let result = filter.filter(messages);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].content, "4");
        assert_eq!(result[1].content, "5");
    }

    #[test]
    fn test_keep_recent_fewer_than_count() {
        let messages = vec![msg(Role::User, "1"), msg(Role::Assistant, "2")];
        let filter = KeepRecentFilter::new(5);
        let result = filter.filter(messages);
        assert_eq!(result.len(), 2);
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
        let filter = ByRoleFilter::new(vec!["user".to_string(), "assistant".to_string()]);
        let result = filter.filter(messages);
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|m| m.role != Role::Function));
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
        let filter = SkipPatternFilter::new(vec!["[DEBUG]".to_string(), "[TOOL]".to_string()]);
        let result = filter.filter(messages);
        assert_eq!(result.len(), 3);
        assert!(
            result
                .iter()
                .all(|m| !m.content.contains("[DEBUG]") && !m.content.contains("[TOOL]"))
        );
    }

    #[test]
    fn test_default_keep_recent() {
        let filter = KeepRecentFilter::default();
        assert_eq!(filter.keep_count, 10);
    }
}

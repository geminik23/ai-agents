use ai_agents_core::ToolCall;

/// Side-effect-free model output produced before a branch commits.
#[derive(Debug, Clone)]
pub enum MainResponseDraft {
    Text {
        raw_content: String,
        thinking: Option<String>,
    },
    ToolCalls {
        raw_content: String,
        calls: Vec<ToolCall>,
        thinking: Option<String>,
    },
}

impl MainResponseDraft {
    pub fn raw_content(&self) -> &str {
        match self {
            Self::Text { raw_content, .. } => raw_content,
            Self::ToolCalls { raw_content, .. } => raw_content,
        }
    }

    pub fn thinking(&self) -> Option<&str> {
        match self {
            Self::Text { thinking, .. } => thinking.as_deref(),
            Self::ToolCalls { thinking, .. } => thinking.as_deref(),
        }
    }
}

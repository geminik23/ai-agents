use ai_agents_reasoning::ReasoningMode;

/// Branch-safe auto reasoning decision.
#[derive(Debug, Clone)]
pub struct ReasoningBranchResult {
    pub mode: ReasoningMode,
}

impl ReasoningBranchResult {
    pub fn new(mode: ReasoningMode) -> Self {
        Self { mode }
    }
}

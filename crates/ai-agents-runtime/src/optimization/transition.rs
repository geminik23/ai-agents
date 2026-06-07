use crate::optimization::TransitionCandidate;

/// Result of a response-independent transition branch.
#[derive(Debug, Clone)]
pub struct TransitionBranchResult {
    pub candidate: Option<TransitionCandidate>,
}

impl TransitionBranchResult {
    pub fn miss() -> Self {
        Self { candidate: None }
    }

    pub fn selected(candidate: TransitionCandidate) -> Self {
        Self {
            candidate: Some(candidate),
        }
    }
}

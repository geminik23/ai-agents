use std::collections::HashMap;
use std::future::Future;

use ai_agents_observability::{
    EventStatus, EventType, ObservabilityManager, ObservationPurpose,
    with_updated_observation_context,
};
use std::sync::Arc;

use super::branch::{RuntimeCommitBehavior, RuntimeOptimizationKind};

/// Runs a branch future with task-local labels that defer observed events until finalization.
pub async fn with_branch_observation<F, T>(
    branch_id: &str,
    optimization: RuntimeOptimizationKind,
    commit_behavior: RuntimeCommitBehavior,
    future: F,
) -> T
where
    F: Future<Output = T>,
{
    let branch = branch_id.to_string();
    with_updated_observation_context(
        move |context| {
            context
                .with_tag("runtime.defer_observation", "true")
                .with_tag("runtime.branch_id", branch)
                .with_tag("runtime.optimization", optimization.as_label())
                .with_tag("runtime.commit_behavior", commit_behavior.as_label())
                .with_tag("runtime.speculative", "true")
        },
        future,
    )
    .await
}

/// Builds finalization tags shared by all branch outcomes.
pub fn branch_finalization_tags(
    optimization: RuntimeOptimizationKind,
    commit_behavior: RuntimeCommitBehavior,
) -> HashMap<String, String> {
    let mut tags = HashMap::new();
    tags.insert(
        "runtime.optimization".to_string(),
        optimization.as_label().to_string(),
    );
    tags.insert(
        "optimization".to_string(),
        optimization.as_label().to_string(),
    );
    tags.insert(
        "runtime.commit_behavior".to_string(),
        commit_behavior.as_label().to_string(),
    );
    tags.insert(
        "commit_behavior".to_string(),
        commit_behavior.as_label().to_string(),
    );
    tags.insert("runtime.speculative".to_string(), "true".to_string());
    tags.insert("speculative".to_string(), "true".to_string());
    tags
}

/// Finalizes a branch if an observability manager is configured.
pub fn finalize_branch(
    manager: Option<&Arc<ObservabilityManager>>,
    branch_id: &str,
    status: &str,
    winner: bool,
    optimization: RuntimeOptimizationKind,
    commit_behavior: RuntimeCommitBehavior,
) {
    if let Some(manager) = manager {
        let tags = branch_finalization_tags(optimization, commit_behavior);
        let finalized =
            manager.finalize_pending_branch(branch_id, status.to_string(), winner, tags.clone());
        if finalized == 0 {
            let mut lifecycle_tags = tags;
            lifecycle_tags.insert("runtime.branch_status".to_string(), status.to_string());
            lifecycle_tags.insert("branch_status".to_string(), status.to_string());
            lifecycle_tags.insert("runtime.winner".to_string(), winner.to_string());
            lifecycle_tags.insert("winner".to_string(), winner.to_string());
            manager.record_lifecycle_event(
                EventType::MemoryOperation {
                    operation: format!("runtime_branch_{}", status),
                },
                ObservationPurpose::Other("runtime_branch".to_string()),
                event_status_for_branch(status),
                0,
                lifecycle_tags,
                None,
            );
        }
    }
}

fn event_status_for_branch(status: &str) -> EventStatus {
    match status {
        "failed" => EventStatus::Error,
        "cancelled" => EventStatus::Cancelled,
        "discarded" => EventStatus::Skipped,
        _ => EventStatus::Success,
    }
}

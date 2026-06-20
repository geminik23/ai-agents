use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use ai_agents_core::{AgentError, Result};
use futures::stream::{FuturesUnordered, StreamExt};

use super::branch::{
    RuntimeBranch, RuntimeBranchOutcome, RuntimeBranchResult, RuntimeBranchStatus,
};
use super::turn::TurnOptimizationContext;

/// Small per-turn scheduler guard for branch limits and reservations.
#[derive(Debug)]
pub struct TurnBranchScheduler {
    max_parallel_tasks: usize,
    active_tasks: usize,
}

pub type BranchFuture<'a> = Pin<Box<dyn Future<Output = RuntimeBranchResult> + Send + 'a>>;

pub struct ScheduledBranchSet<'a> {
    scheduler: TurnBranchScheduler,
    pending: HashMap<String, RuntimeBranch>,
    futures: FuturesUnordered<Pin<Box<dyn Future<Output = RuntimeBranchOutcome> + Send + 'a>>>,
}

impl TurnBranchScheduler {
    pub fn new(max_parallel_tasks: usize) -> Result<Self> {
        if max_parallel_tasks == 0 {
            return Err(AgentError::InvalidSpec(
                "runtime.optimization.max_parallel_runtime_tasks must be greater than 0".into(),
            ));
        }
        Ok(Self {
            max_parallel_tasks,
            active_tasks: 0,
        })
    }

    pub fn can_schedule_branch(&self) -> bool {
        self.active_tasks < self.max_parallel_tasks
    }

    pub fn reserve_task(&mut self) -> bool {
        if !self.can_schedule_branch() {
            return false;
        }
        self.active_tasks += 1;
        true
    }

    pub fn reserve_llm_branch(&mut self, turn: &mut TurnOptimizationContext) -> bool {
        if !self.can_schedule_branch() || !turn.reserve_speculative_llm_call() {
            return false;
        }
        self.active_tasks += 1;
        true
    }

    pub fn reserve_branch(&mut self, turn: &mut TurnOptimizationContext) -> bool {
        self.reserve_llm_branch(turn)
    }

    pub fn release_task(&mut self) {
        self.active_tasks = self.active_tasks.saturating_sub(1);
    }

    pub fn complete_branch(&mut self, branch: &mut RuntimeBranch) -> Result<()> {
        self.release_task();
        branch.transition_to(RuntimeBranchStatus::Completed)
    }
}

impl<'a> ScheduledBranchSet<'a> {
    pub fn new(max_parallel_tasks: usize) -> Result<Self> {
        Ok(Self {
            scheduler: TurnBranchScheduler::new(max_parallel_tasks)?,
            pending: HashMap::new(),
            futures: FuturesUnordered::new(),
        })
    }

    pub fn reserve_task(&mut self) -> bool {
        self.scheduler.reserve_task()
    }

    pub fn release_task(&mut self) {
        self.scheduler.release_task();
    }

    pub fn schedule(&mut self, branch: RuntimeBranch, future: BranchFuture<'a>) -> bool {
        if !self.scheduler.can_schedule_branch() {
            return false;
        }
        self.scheduler.active_tasks += 1;
        let id = branch.branch_id();
        self.pending.insert(id.clone(), branch.clone());
        self.futures.push(Box::pin(async move {
            let mut branch = branch;
            let result = future.await;
            let _ = branch.complete();
            RuntimeBranchOutcome { branch, result }
        }));
        true
    }

    pub async fn next_completed(&mut self) -> Option<RuntimeBranchOutcome> {
        let outcome = self.futures.next().await?;
        self.pending.remove(&outcome.branch.branch_id());
        self.scheduler.release_task();
        Some(outcome)
    }

    pub fn cancel_pending(&mut self) -> Vec<RuntimeBranch> {
        self.futures = FuturesUnordered::new();
        let pending = std::mem::take(&mut self.pending)
            .into_values()
            .collect::<Vec<_>>();
        for _ in &pending {
            self.scheduler.release_task();
        }
        pending
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimization::{
        RuntimeCommitBehavior, RuntimeOptimizationKind, RuntimeTaskPriority, RuntimeTaskPurpose,
    };
    use std::collections::HashMap;

    #[test]
    fn scheduler_enforces_parallel_limit() {
        let mut scheduler = TurnBranchScheduler::new(1).unwrap();
        let mut turn = TurnOptimizationContext::new("input", HashMap::new(), 2);
        assert!(scheduler.reserve_branch(&mut turn));
        assert!(!scheduler.reserve_branch(&mut turn));
    }

    #[test]
    fn scheduler_enforces_speculative_call_limit() {
        let mut scheduler = TurnBranchScheduler::new(2).unwrap();
        let mut turn = TurnOptimizationContext::new("input", HashMap::new(), 1);
        assert!(scheduler.reserve_branch(&mut turn));
        assert!(!scheduler.reserve_branch(&mut turn));
    }

    #[test]
    fn scheduled_branch_set_cancels_pending_branches() {
        let mut branches = ScheduledBranchSet::new(1).unwrap();
        let branch = RuntimeBranch::new(
            RuntimeTaskPurpose::MainResponse,
            RuntimeOptimizationKind::SpeculativeSkillRouting,
            RuntimeTaskPriority::Normal,
            RuntimeCommitBehavior::FinalResponse,
        );
        let branch_id = branch.branch_id();

        assert!(branches.schedule(
            branch,
            Box::pin(async { std::future::pending::<RuntimeBranchResult>().await }),
        ));
        let pending = branches.cancel_pending();

        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].branch_id(), branch_id);
        assert!(branches.is_empty());
        assert!(branches.reserve_task());
    }

    #[tokio::test]
    async fn scheduled_branch_set_keeps_completed_outcomes_out_of_cancelled_set() {
        let mut branches = ScheduledBranchSet::new(1).unwrap();
        let branch = RuntimeBranch::new(
            RuntimeTaskPurpose::MainResponse,
            RuntimeOptimizationKind::SpeculativeSkillRouting,
            RuntimeTaskPriority::Normal,
            RuntimeCommitBehavior::FinalResponse,
        );
        let branch_id = branch.branch_id();

        assert!(branches.schedule(branch, Box::pin(async { RuntimeBranchResult::Cancelled }),));
        let outcome = branches.next_completed().await.unwrap();
        let pending = branches.cancel_pending();

        assert_eq!(outcome.branch.branch_id(), branch_id);
        assert!(matches!(outcome.result, RuntimeBranchResult::Cancelled));
        assert!(pending.is_empty());
        assert!(branches.reserve_task());
    }
}

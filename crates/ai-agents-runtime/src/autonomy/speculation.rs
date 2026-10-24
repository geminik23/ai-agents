//! Branch-local witnesses forward the existing run ledger; they never mint an execution budget or authorize a saved identifier.

use super::current_execution;
use ai_agents_core::autonomy::{
    InvocationAdmission, ProviderCostBound, ProviderCostSettlement, TaskFinalAdmission,
};
use ai_agents_core::{AgentError, LLMError, Result};
use async_trait::async_trait;
use serde_json::Value;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

/// Records acknowledged admissions and proven non-invocation refunds, separately from a scheduler's selected/no-match labels.
#[derive(Debug, Default)]
pub(crate) struct BranchAdmission {
    admissions: AtomicU64,
    closed_denial: AtomicBool,
}

impl BranchAdmission {
    /// Only the live closed-guard witness plus zero retained admissions proves that unfinished selection was not invoked.
    pub(crate) fn unfinished(&self) -> bool {
        self.closed_denial.load(Ordering::Acquire) && !self.admitted()
    }

    /// A returned error after any consumed or uncertain attempt is not permission to replay that branch.
    pub(crate) fn admitted(&self) -> bool {
        self.admissions.load(Ordering::Acquire) != 0
    }

    // The underlying ledger validates identity and single-use settlement before its witness can release a proven unused admission.
    fn settled(&self, value: &Value) {
        if value.get("not_invoked") == Some(&Value::Bool(true)) {
            let _ = self
                .admissions
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                    value.checked_sub(1)
                });
        }
    }
}

#[derive(Debug)]
struct BranchRestriction {
    closed: Arc<AtomicBool>,
    witness: Arc<BranchAdmission>,
}

impl TaskFinalAdmission for BranchRestriction {
    // Closing loses only new invocations; already dispatched futures remain polled until they settle or the original run stops.
    fn validate(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            self.witness.closed_denial.store(true, Ordering::Release);
            Err(AgentError::Other(
                "speculative branch admission is closed".into(),
            ))
        } else {
            Ok(())
        }
    }

    // Local closure prohibits new requests and retries, not settlement of an already dispatched losing call.
    fn validate_dispatched(&self) -> Result<()> {
        Ok(())
    }
}

struct BranchLedger {
    parent: Arc<dyn InvocationAdmission>,
    witness: Arc<BranchAdmission>,
}

#[async_trait]
impl InvocationAdmission for BranchLedger {
    // Admission and all resource enforcement stay in the exact original run ledger.
    async fn admit_llm(&self) -> std::result::Result<String, LLMError> {
        let id = self.parent.admit_llm().await?;
        self.witness.admissions.fetch_add(1, Ordering::AcqRel);
        Ok(id)
    }
    // Priced bounds are forwarded unchanged; branch-local accounting creates no allowance.
    async fn admit_priced_llm(
        &self,
        bound: Option<ProviderCostBound>,
    ) -> std::result::Result<String, LLMError> {
        let id = self.parent.admit_priced_llm(bound).await?;
        self.witness.admissions.fetch_add(1, Ordering::AcqRel);
        Ok(id)
    }
    // Unacknowledged settlement cannot turn an invoked branch into unfinished work.
    async fn settle_llm(&self, id: &str, value: Value) -> std::result::Result<(), LLMError> {
        self.parent.settle_llm(id, value.clone()).await?;
        self.witness.settled(&value);
        Ok(())
    }
    // Matching cost settlement and proven unused refunds retain their original request identity and backend acknowledgement.
    async fn settle_priced_llm(
        &self,
        id: &str,
        value: Value,
        settlement: Option<ProviderCostSettlement>,
    ) -> std::result::Result<(), LLMError> {
        self.parent
            .settle_priced_llm(id, value.clone(), settlement)
            .await?;
        self.witness.settled(&value);
        Ok(())
    }
    // Missing priced capability remains a parent-wide denial, not an optional-branch error to ignore.
    fn reject_priced_request(&self) {
        self.parent.reject_priced_request();
    }
    // The parent selects whether this run requires request-matched price proof.
    fn requires_priced_llm(&self) -> bool {
        self.parent.requires_priced_llm()
    }
    // Abandoned calls retain the original dispatched marker and uncertainty policy.
    fn abandon_llm(&self, id: &str) {
        self.parent.abandon_llm(id);
    }
    // Closing local admission does not cancel already dispatched provider settlement or stop the eventual winning branch.
    fn execution_stopped(&self) -> bool {
        self.parent.execution_stopped()
    }
    // A branch can never widen the run's remaining allowance.
    fn remaining_duration(&self) -> Option<std::time::Duration> {
        self.parent.remaining_duration()
    }
}

/// Installs a restriction-only branch view over the same parent ledger before first polling an existing scheduler future.
pub(crate) async fn scope_branch<F: std::future::Future>(
    closed: Arc<AtomicBool>,
    witness: Arc<BranchAdmission>,
    future: F,
) -> F::Output {
    if current_execution().is_some()
        && let Some(parent) = ai_agents_llm::current_invocation_admission()
    {
        let guard = Arc::new(BranchRestriction {
            closed,
            witness: witness.clone(),
        });
        ai_agents_llm::scope_invocation_admission(
            Arc::new(BranchLedger { parent, witness }),
            ai_agents_llm::scope_invocation_evidence(vec![guard], future),
        )
        .await
    } else {
        future.await
    }
}

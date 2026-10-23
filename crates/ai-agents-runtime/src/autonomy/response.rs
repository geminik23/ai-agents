//! Private retained response bindings restrict only the exact pending tool's pre-dispatch admission.

use super::*;
use ai_agents_core::{AgentError, Result, ToolExecutionRequest};
use std::{
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

/// Live review and original clocks survive pauses through the owner, never through serialized identifiers alone.
#[derive(Debug)]
pub(crate) struct RetainedResponse {
    pub(crate) request: ToolExecutionRequest,
    pub(crate) expires_at: Option<chrono::DateTime<chrono::Utc>>,
    pub(crate) deadline: Option<Instant>,
    pub(crate) review: Option<Arc<dyn TaskFinalAdmission>>,
    pub(crate) timeout_disposition: AtomicBool,
    pub(crate) applied: AtomicBool,
}

#[derive(Debug, Default)]
struct ResponseAdmission {
    bindings: parking_lot::Mutex<Vec<Arc<RetainedResponse>>>,
    missing: AtomicBool,
    applied: AtomicBool,
}
impl TaskFinalAdmission for ResponseAdmission {
    /// Only pre-body admission is restricted; already invoked effects retain their ordinary settlement and cancellation rules.
    fn validate(&self) -> Result<()> {
        if self.applied.load(Ordering::Acquire) {
            return Ok(());
        }
        if self.missing.load(Ordering::Acquire) {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        for binding in self.bindings.lock().iter() {
            if let Some(review) = &binding.review {
                review.validate()?;
            }
            if !binding.timeout_disposition.load(Ordering::Acquire)
                && !binding.applied.load(Ordering::Acquire)
                && (binding
                    .expires_at
                    .is_some_and(|at| chrono::Utc::now() >= at)
                    || binding.deadline.is_some_and(|at| Instant::now() >= at))
            {
                return Err(AgentError::HITLTimeout);
            }
        }
        Ok(())
    }
}

tokio::task_local! { static RESPONSE_ADMISSION: Arc<ResponseAdmission>; }

/// Eagerly pins only receipts for this exact executor request, before localization or any other response-dependent preparation.
/// Fallbacks share call IDs but cannot inherit another hop's original arguments or timeout disposition.
pub(crate) async fn scope_response_admission<F: Future>(
    request: ToolExecutionRequest,
    ids: Vec<String>,
    future: F,
) -> F::Output {
    let admission = Arc::new(ResponseAdmission::default());
    if let Some(execution) = current_execution() {
        let bindings = execution.owner.responses.lock();
        for id in ids {
            if let Some(binding) = bindings.get(&id) {
                if !serde_json::to_value(&binding.request)
                    .ok()
                    .zip(serde_json::to_value(&request).ok())
                    .is_some_and(|(reviewed, current)| reviewed == current)
                {
                    continue;
                }
                if super::suspension::timeout_approved(&id) {
                    binding.timeout_disposition.store(true, Ordering::Release);
                }
                admission.bindings.lock().push(binding.clone());
            }
        }
    }
    RESPONSE_ADMISSION
        .scope(
            admission.clone(),
            ai_agents_llm::scope_invocation_evidence(vec![admission], future),
        )
        .await
}

/// Receipt consumption must find the same eagerly installed live binding, even if current approval gates changed.
pub(crate) fn bind_response(request_id: &str, call_id: &str) -> Result<()> {
    let admission = RESPONSE_ADMISSION
        .try_with(Clone::clone)
        .map_err(|_| TaskRunStorageError::InvalidCheckpoint)?;
    let binding = current_execution()
        .and_then(|execution| execution.owner.responses.lock().get(request_id).cloned());
    if binding.as_ref().is_none_or(|binding| {
        binding.request.call_id != call_id
            || !admission
                .bindings
                .lock()
                .iter()
                .any(|old| Arc::ptr_eq(old, binding))
    }) {
        admission.missing.store(true, Ordering::Release);
    }
    admission.validate()
}

/// A stale response stops further model work as well as its pending body; otherwise a denial could be followed by a fabricated completion.
pub(crate) fn check_response_admission() -> Result<()> {
    let result = RESPONSE_ADMISSION
        .try_with(|admission| admission.validate())
        .unwrap_or(Ok(()));
    if result.is_err()
        && let Some(execution) = current_execution()
    {
        execution.stop("retained_response_invalid");
    }
    result
}

/// The final synchronous check precedes this mark; expiry afterward cannot retroactively certify non-invocation.
pub(crate) fn response_applied() {
    let _ = RESPONSE_ADMISSION.try_with(|admission| {
        admission.applied.store(true, Ordering::Release);
        for binding in admission.bindings.lock().iter() {
            binding.applied.store(true, Ordering::Release);
        }
    });
}

/// Only acknowledged executable paused state determines which receipts a future continuation may still need.
/// This bounded traversal only retains existing live bindings; matching identifiers never create execution authority.
pub(crate) fn retain_response_bindings(owner: &RunOwner, payload: &serde_json::Value) {
    let mut ids = std::collections::BTreeSet::new();
    let mut values = vec![payload];
    while let Some(value) = values.pop() {
        match value {
            serde_json::Value::Array(items) => values.extend(items),
            serde_json::Value::Object(items) => {
                if let Some(id) = items.get("request_id").and_then(serde_json::Value::as_str) {
                    ids.insert(id);
                }
                values.extend(
                    items
                        .iter()
                        .filter(|(key, _)| {
                            !matches!(
                                key.as_str(),
                                "evidence"
                                    | "completed"
                                    | "result"
                                    | "results"
                                    | "runtime"
                                    | "response"
                            )
                        })
                        .map(|(_, value)| value),
                );
            }
            _ => {}
        }
    }
    owner
        .responses
        .lock()
        .retain(|id, _| ids.contains(id.as_str()));
}

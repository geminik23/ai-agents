use crate::*;

#[path = "managed_stream_tests.rs"]
mod stream_tests;
use ai_agents_core::autonomy::InvocationAdmission;
use async_trait::async_trait;
use futures::StreamExt;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant, SystemTime};

#[derive(Default)]
struct Admission {
    delay: bool,
    delay_remaining: bool,
    admissions: AtomicUsize,
    settlements: AtomicUsize,
    refunds: AtomicUsize,
    abandonments: AtomicUsize,
    stop: AtomicBool,
}
#[async_trait]
impl InvocationAdmission for Admission {
    async fn admit_llm(&self) -> Result<String, LLMError> {
        self.admissions.fetch_add(1, Ordering::SeqCst);
        if self.delay {
            let remaining = current_invocation_bounds().unwrap().remaining();
            std::thread::sleep(remaining + Duration::from_millis(2));
        }
        Ok("attempt".into())
    }
    async fn settle_llm(&self, _: &str, result: serde_json::Value) -> Result<(), LLMError> {
        self.settlements.fetch_add(1, Ordering::SeqCst);
        if result["not_invoked"] == serde_json::json!(true) {
            self.refunds.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    }
    fn abandon_llm(&self, _: &str) {
        self.abandonments.fetch_add(1, Ordering::SeqCst);
    }
    fn execution_stopped(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }
    fn remaining_duration(&self) -> Option<Duration> {
        if self.delay_remaining {
            std::thread::sleep(
                current_invocation_bounds().unwrap().remaining() + Duration::from_millis(2),
            );
        }
        Some(Duration::from_secs(60))
    }
}

#[derive(Default)]
struct Provider(AtomicUsize);
#[async_trait]
impl LLMProvider for Provider {
    async fn complete(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> Result<LLMResponse, LLMError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(LLMResponse::new("done", FinishReason::Stop))
    }
    async fn complete_stream(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError>
    {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(futures::stream::empty()))
    }
    fn provider_name(&self) -> &str {
        "bounded mock"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
}

// The original check bound is deliberately smaller than the run allowance and survives awaited admission.
#[tokio::test]
async fn invocation_bounds_late_admission_never_polls_provider() {
    for mode in 0..3 {
        let delayed = mode != 0;
        let admission = Arc::new(Admission {
            delay: mode == 1,
            delay_remaining: mode == 2,
            ..Default::default()
        });
        let provider = Arc::new(Provider::default());
        let managed = managed_provider(provider.clone());
        let bounds = InvocationBounds::new(
            SystemTime::now() + Duration::from_secs(10),
            Instant::now() + Duration::from_millis(30),
        );
        let result = scope_invocation_bounds(
            Some(bounds),
            scope_invocation_admission(
                admission.clone(),
                managed.complete(&[ChatMessage::user("test")], None),
            ),
        )
        .await;
        assert_eq!(result.is_ok(), !delayed);
        assert_eq!(provider.0.load(Ordering::SeqCst), usize::from(!delayed));
        assert_eq!(admission.admissions.load(Ordering::SeqCst), 1);
        assert_eq!(
            admission.refunds.load(Ordering::SeqCst),
            usize::from(delayed)
        );
        assert_eq!(admission.abandonments.load(Ordering::SeqCst), 0);
    }
}

// A returned stream retains its opening restriction outside the scope; post-dispatch expiry never refunds an invoked attempt.
#[tokio::test]
async fn invocation_bounds_stream_lifetime_retains_accounting_and_independent_limits() {
    for ledger in [false, true] {
        let provider = Arc::new(Provider::default());
        let managed = managed_provider(provider.clone());
        let admission = Arc::new(Admission::default());
        let bounds = InvocationBounds::new(
            SystemTime::now() + Duration::from_secs(10),
            Instant::now() + Duration::from_millis(30),
        );
        let work = async {
            if ledger {
                scope_invocation_admission(
                    admission.clone(),
                    managed.complete_stream(&[ChatMessage::user("test")], None),
                )
                .await
            } else {
                managed
                    .complete_stream(&[ChatMessage::user("test")], None)
                    .await
            }
        };
        let mut stream = scope_invocation_bounds(Some(bounds), work).await.unwrap();
        assert_eq!(provider.0.load(Ordering::SeqCst), 1);
        std::thread::sleep(bounds.remaining() + Duration::from_millis(2));
        assert!(stream.next().await.unwrap().is_err());
        drop(stream);
        assert_eq!(admission.refunds.load(Ordering::SeqCst), 0);
        assert_eq!(admission.settlements.load(Ordering::SeqCst), 0);
        assert_eq!(
            admission.abandonments.load(Ordering::SeqCst),
            usize::from(ledger)
        );
    }
}

struct OverrunProvider;
#[async_trait]
impl LLMProvider for OverrunProvider {
    async fn complete(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> Result<LLMResponse, LLMError> {
        std::thread::sleep(
            current_invocation_bounds().unwrap().remaining() + Duration::from_millis(2),
        );
        Ok(LLMResponse::new("late success", FinishReason::Stop))
    }
    async fn complete_stream(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError>
    {
        unreachable!()
    }
    fn provider_name(&self) -> &str {
        "overrun"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
}

// Non-preemptible provider work can overrun, but its late return cannot become an eligible result or an unused-attempt refund.
#[tokio::test]
async fn invocation_bounds_synchronous_overrun_rejects_late_success() {
    let admission = Arc::new(Admission::default());
    let provider = managed_provider(Arc::new(OverrunProvider));
    let bounds = InvocationBounds::new(
        SystemTime::now() + Duration::from_secs(10),
        Instant::now() + Duration::from_millis(30),
    );
    let result = scope_invocation_bounds(
        Some(bounds),
        scope_invocation_admission(
            admission.clone(),
            provider.complete(&[ChatMessage::user("test")], None),
        ),
    )
    .await;
    assert!(
        result.is_err(),
        "late synchronous success escaped its original bound"
    );
    assert_eq!(admission.admissions.load(Ordering::SeqCst), 1);
    assert_eq!(admission.refunds.load(Ordering::SeqCst), 0);
}

#[derive(Debug)]
struct LastPollProof {
    checks: AtomicUsize,
    slow: bool,
}
impl ai_agents_core::autonomy::TaskFinalAdmission for LastPollProof {
    fn validate(&self) -> ai_agents_core::Result<()> {
        if self.checks.fetch_add(1, Ordering::SeqCst) == 1 {
            if self.slow {
                std::thread::sleep(
                    current_invocation_bounds().unwrap().remaining() + Duration::from_millis(2),
                );
            } else {
                return Err(ai_agents_core::AgentError::Other(
                    "last pre-poll rejection".into(),
                ));
            }
        }
        Ok(())
    }
}

// A rejected or slow last proof sample cannot mark an unpolled provider dispatched or lose its proven non-invocation refund.
#[tokio::test]
async fn invocation_bounds_last_proof_precedes_actual_first_poll() {
    for slow in [false, true] {
        let admission = Arc::new(Admission::default());
        let raw = Arc::new(Provider::default());
        let provider = managed_provider(raw.clone());
        let bounds = InvocationBounds::new(
            SystemTime::now() + Duration::from_secs(10),
            Instant::now() + Duration::from_millis(30),
        );
        let proof = Arc::new(LastPollProof {
            checks: AtomicUsize::new(0),
            slow,
        });
        let result = scope_invocation_evidence(
            vec![proof],
            scope_invocation_bounds(
                Some(bounds),
                scope_invocation_admission(
                    admission.clone(),
                    provider.complete(&[ChatMessage::user("test")], None),
                ),
            ),
        )
        .await;
        assert!(result.is_err());
        assert_eq!(raw.0.load(Ordering::SeqCst), 0);
        assert_eq!(admission.refunds.load(Ordering::SeqCst), 1);
        assert_eq!(admission.abandonments.load(Ordering::SeqCst), 0);
    }
}

// Neither a longer child allowance nor wall-clock reprojection can widen the original monotonic restriction.
#[tokio::test]
async fn invocation_bounds_nested_scopes_only_narrow() {
    let parent = InvocationBounds::new(
        SystemTime::now() + Duration::from_secs(10),
        Instant::now() + Duration::from_millis(30),
    );
    let child = InvocationBounds::new(
        SystemTime::now() + Duration::from_secs(20),
        Instant::now() + Duration::from_secs(20),
    );
    scope_invocation_bounds(
        Some(parent),
        scope_invocation_bounds(Some(child), async {
            assert!(current_invocation_bounds().unwrap().remaining() <= Duration::from_millis(30));
        }),
    )
    .await;
    let expired = InvocationBounds::new(
        SystemTime::now() + Duration::from_secs(60),
        Instant::now() - Duration::from_millis(1),
    );
    let provider = Arc::new(Provider::default());
    assert!(
        scope_invocation_bounds(
            Some(expired),
            managed_provider(provider.clone()).complete(&[ChatMessage::user("test")], None)
        )
        .await
        .is_err()
    );
    assert_eq!(provider.0.load(Ordering::SeqCst), 0);
}

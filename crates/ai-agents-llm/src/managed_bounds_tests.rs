use crate::*;

#[path = "managed_stream_tests.rs"]
mod stream_tests;
use ai_agents_core::autonomy::InvocationAdmission;
use async_trait::async_trait;
use futures::StreamExt;
use std::cell::Cell;
use std::ops::AsyncFnOnce;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant, SystemTime};

const TEST_INVOCATION_WINDOW: Duration = Duration::from_millis(30);

tokio::task_local! {
    static INVOCATION_NOW: Cell<Instant>;
}

// Keep the clock active so returned streams can sample their captured bound outside the opening scope.
async fn with_test_deadline<T>(work: impl AsyncFnOnce(InvocationBounds) -> T) -> T {
    let now = Instant::now();
    let bounds = InvocationBounds::for_test_clock(now + TEST_INVOCATION_WINDOW);
    INVOCATION_NOW
        .scope(
            Cell::new(now),
            InvocationBounds::scope_test_clock(|| INVOCATION_NOW.with(Cell::get), work(bounds)),
        )
        .await
}

fn expire_invocation() {
    INVOCATION_NOW.with(|now| now.set(now.get() + TEST_INVOCATION_WINDOW));
}

#[derive(Default)]
struct Admission {
    expire_on_admission: bool,
    expire_on_remaining: bool,
    priced: bool,
    admissions: AtomicUsize,
    settlements: AtomicUsize,
    refunds: AtomicUsize,
    abandonments: AtomicUsize,
    stop: AtomicBool,
}
#[async_trait]
impl InvocationAdmission for Admission {
    fn requires_priced_llm(&self) -> bool {
        self.priced
    }
    fn reject_priced_request(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }
    async fn admit_llm(&self) -> Result<String, LLMError> {
        self.admissions.fetch_add(1, Ordering::SeqCst);
        if self.expire_on_admission {
            expire_invocation();
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
        if self.expire_on_remaining {
            expire_invocation();
        }
        Some(Duration::from_secs(60))
    }
}

#[derive(Default)]
struct Provider(AtomicUsize);
#[async_trait]
impl LLMProvider for Provider {
    fn uses_choice_for_routing(&self) -> bool {
        true
    }
    async fn choose(
        &self,
        _: &ai_agents_core::ChoiceRequest<'_>,
    ) -> Result<ai_agents_core::ChoiceDecision, LLMError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(ai_agents_core::ChoiceDecision {
            selected: None,
            usage: Some(TokenUsage::new(4, 0)),
        })
    }
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

#[tokio::test]
async fn structured_choice_manages_refusal_once_and_rejects_a_hard_price_limit() {
    let provider = Arc::new(Provider::default());
    let wrapped = managed_provider(provider.clone());
    let candidates = [
        ai_agents_core::ChoiceCandidate {
            id: "one",
            description: None,
        },
        ai_agents_core::ChoiceCandidate {
            id: "two",
            description: None,
        },
    ];
    let request = ai_agents_core::ChoiceRequest {
        input: "input",
        question: "pick one",
        candidates: &candidates,
    };
    for priced in [false, true] {
        let admission = Arc::new(Admission {
            priced,
            ..Default::default()
        });
        let result = scope_invocation_admission(
            admission.clone(),
            managed_choice(wrapped.as_ref(), &request),
        )
        .await;
        if priced {
            assert!(matches!(result, Err(LLMError::Config(_))));
            assert!(admission.stop.load(Ordering::SeqCst));
            assert_eq!(admission.admissions.load(Ordering::SeqCst), 0);
            assert_eq!(admission.settlements.load(Ordering::SeqCst), 0);
        } else {
            let refusal = result.unwrap();
            assert_eq!(refusal.selected, None);
            assert_eq!(refusal.usage.unwrap().total_tokens, 4);
            assert_eq!(admission.admissions.load(Ordering::SeqCst), 1);
            assert_eq!(admission.settlements.load(Ordering::SeqCst), 1);
        }
        assert_eq!(provider.0.load(Ordering::SeqCst), 1);
    }
}

// The original check bound is deliberately smaller than the run allowance and survives awaited admission.
#[tokio::test]
async fn invocation_bounds_late_admission_never_polls_provider() {
    for mode in 0..3 {
        let delayed = mode != 0;
        let admission = Arc::new(Admission {
            expire_on_admission: mode == 1,
            expire_on_remaining: mode == 2,
            ..Default::default()
        });
        let provider = Arc::new(Provider::default());
        let managed = managed_provider(provider.clone());
        let result = with_test_deadline(async |bounds| {
            scope_invocation_bounds(
                Some(bounds),
                scope_invocation_admission(
                    admission.clone(),
                    managed.complete(&[ChatMessage::user("test")], None),
                ),
            )
            .await
        })
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
        with_test_deadline(async |bounds| {
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
            expire_invocation();
            assert!(stream.next().await.unwrap().is_err());
            drop(stream);
        })
        .await;
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
    expires: bool,
}
impl ai_agents_core::autonomy::TaskFinalAdmission for LastPollProof {
    fn validate(&self) -> ai_agents_core::Result<()> {
        if self.checks.fetch_add(1, Ordering::SeqCst) == 1 {
            if self.expires {
                expire_invocation();
            } else {
                return Err(ai_agents_core::AgentError::Other(
                    "last pre-poll rejection".into(),
                ));
            }
        }
        Ok(())
    }
}

// A rejected or expired last proof sample cannot mark an unpolled provider dispatched or lose its proven non-invocation refund.
#[tokio::test]
async fn invocation_bounds_last_proof_precedes_actual_first_poll() {
    for expires in [false, true] {
        let admission = Arc::new(Admission::default());
        let raw = Arc::new(Provider::default());
        let provider = managed_provider(raw.clone());
        let proof = Arc::new(LastPollProof {
            checks: AtomicUsize::new(0),
            expires,
        });
        let result = with_test_deadline(async |bounds| {
            scope_invocation_evidence(
                vec![proof],
                scope_invocation_bounds(
                    Some(bounds),
                    scope_invocation_admission(
                        admission.clone(),
                        provider.complete(&[ChatMessage::user("test")], None),
                    ),
                ),
            )
            .await
        })
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

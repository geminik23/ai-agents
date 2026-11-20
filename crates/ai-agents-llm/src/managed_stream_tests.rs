use super::*;

#[derive(Debug)]
struct Proof(Arc<AtomicBool>);
impl ai_agents_core::autonomy::TaskFinalAdmission for Proof {
    fn validate(&self) -> ai_agents_core::Result<()> {
        if self.0.load(Ordering::SeqCst) {
            Err(ai_agents_core::AgentError::Other("proof changed".into()))
        } else {
            Ok(())
        }
    }
}

struct StreamProvider {
    mode: &'static str,
    entered: Arc<tokio::sync::Semaphore>,
    release: Arc<tokio::sync::Semaphore>,
}
#[async_trait]
impl LLMProvider for StreamProvider {
    async fn complete(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> Result<LLMResponse, LLMError> {
        unreachable!()
    }
    async fn complete_stream(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError>
    {
        let mode = self.mode;
        let entered = self.entered.clone();
        let release = self.release.clone();
        let stream = futures::stream::unfold(0, move |index| {
            let entered = entered.clone();
            let release = release.clone();
            async move {
                if index == 0 {
                    return Some((Ok(LLMChunk::new("first", false)), 1));
                }
                if index > 1 {
                    return None;
                }
                if matches!(mode, "pending" | "proof") {
                    entered.add_permits(1);
                    release.acquire().await.unwrap().forget();
                }
                if mode == "overrun" {
                    std::thread::sleep(
                        current_invocation_bounds().unwrap().remaining() + Duration::from_millis(2),
                    );
                }
                if mode == "error" {
                    Some((Err(LLMError::Other("stream error".into())), 2))
                } else if mode == "overrun" {
                    Some((Ok(LLMChunk::new("late", true)), 2))
                } else {
                    None
                }
            }
        });
        Ok(Box::new(Box::pin(stream)))
    }
    fn provider_name(&self) -> &str {
        "stream witness"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
}

#[derive(Default)]
struct AckAdmission {
    inner: Admission,
    reached: tokio::sync::Notify,
    release: tokio::sync::Notify,
}
#[async_trait]
impl InvocationAdmission for AckAdmission {
    async fn admit_llm(&self) -> Result<String, LLMError> {
        self.inner.admit_llm().await
    }
    async fn settle_llm(&self, id: &str, result: serde_json::Value) -> Result<(), LLMError> {
        assert!(
            current_invocation_bounds().is_some(),
            "settlement lost the opening restriction scope"
        );
        self.reached.notify_one();
        self.release.notified().await;
        self.inner.settle_llm(id, result).await
    }
    fn abandon_llm(&self, id: &str) {
        self.inner.abandon_llm(id);
    }
    fn execution_stopped(&self) -> bool {
        self.inner.execution_stopped()
    }
    fn remaining_duration(&self) -> Option<Duration> {
        Some(Duration::from_secs(60))
    }
}

// A late acknowledged settlement stays charged, but cannot expose a successful completion or EOF after authority changed during the wait.
#[tokio::test]
async fn managed_post_ack_result_eligibility_completion_and_stream() {
    for streaming in [false, true] {
        for mode in ["proof", "cancel", "expiry"] {
            let admission = Arc::new(AckAdmission::default());
            let changed = Arc::new(AtomicBool::new(false));
            let provider = if streaming {
                managed_provider(Arc::new(StreamProvider {
                    mode: "eof",
                    entered: Arc::new(tokio::sync::Semaphore::new(0)),
                    release: Arc::new(tokio::sync::Semaphore::new(0)),
                }))
            } else {
                managed_provider(Arc::new(Provider::default()))
            };
            let bounds = InvocationBounds::new(
                SystemTime::now() + Duration::from_secs(30),
                Instant::now()
                    + if mode == "expiry" {
                        Duration::from_millis(50)
                    } else {
                        Duration::from_secs(20)
                    },
            );
            let body = async {
                let messages = [ChatMessage::user("test")];
                if streaming {
                    let mut stream = provider.complete_stream(&messages, None).await?;
                    assert_eq!(stream.next().await.unwrap()?.delta, "first");
                    match stream.next().await {
                        Some(Err(error)) => Err(error),
                        None => Ok(()),
                        other => panic!("unexpected EOF: {other:?}"),
                    }
                } else {
                    provider.complete(&messages, None).await.map(|_| ())
                }
            };
            let mut work = Box::pin(scope_invocation_evidence(
                vec![Arc::new(Proof(changed.clone()))],
                scope_invocation_bounds(
                    Some(bounds),
                    scope_invocation_admission(admission.clone(), body),
                ),
            ));
            tokio::select! { result = &mut work => panic!("settlement did not wait: {result:?}"), _ = admission.reached.notified() => {} }
            match mode {
                "proof" => changed.store(true, Ordering::SeqCst),
                "cancel" => admission.inner.stop.store(true, Ordering::SeqCst),
                _ => std::thread::sleep(bounds.remaining() + Duration::from_millis(2)),
            }
            admission.release.notify_one();
            assert!(work.await.is_err(), "{streaming}/{mode}");
            assert_eq!(admission.inner.settlements.load(Ordering::SeqCst), 1);
            assert_eq!(admission.inner.refunds.load(Ordering::SeqCst), 0);
            assert_eq!(admission.inner.abandonments.load(Ordering::SeqCst), 0);
        }
    }
}

// Returned nonempty streams retain opening restrictions through EOF/error settlement and never refund dispatched work on drop or stop.
#[tokio::test]
async fn managed_stream_terminal_and_interruption_matrix() {
    for mode in ["eof", "error", "drop", "pending", "proof", "overrun"] {
        let admission = Arc::new(Admission::default());
        let changed = Arc::new(AtomicBool::new(false));
        let entered = Arc::new(tokio::sync::Semaphore::new(0));
        let release = Arc::new(tokio::sync::Semaphore::new(0));
        let provider = managed_provider(Arc::new(StreamProvider {
            mode,
            entered: entered.clone(),
            release: release.clone(),
        }));
        let bounds = InvocationBounds::new(
            SystemTime::now() + Duration::from_secs(30),
            Instant::now() + Duration::from_secs(20),
        );
        let mut stream = scope_invocation_evidence(
            vec![Arc::new(Proof(changed.clone()))],
            scope_invocation_bounds(
                Some(bounds),
                scope_invocation_admission(
                    admission.clone(),
                    provider.complete_stream(&[ChatMessage::user("stream")], None),
                ),
            ),
        )
        .await
        .unwrap();
        assert_eq!(stream.next().await.unwrap().unwrap().delta, "first");
        if mode == "drop" {
            drop(stream);
        } else if mode == "overrun" {
            // The shorter consumption bound intersects the captured opening bound without changing the run allowance.
            let cut = InvocationBounds::new(
                SystemTime::now() + Duration::from_secs(10),
                Instant::now() + Duration::from_millis(30),
            );
            assert!(
                scope_invocation_bounds(Some(cut), stream.next())
                    .await
                    .unwrap()
                    .is_err()
            );
            drop(stream);
        } else if matches!(mode, "pending" | "proof") {
            let mut next = Box::pin(stream.next());
            tokio::select! { result = &mut next => panic!("stream was not pending: {result:?}"), permit = entered.acquire() => permit.unwrap().forget() }
            if mode == "pending" {
                admission.stop.store(true, Ordering::SeqCst);
            } else {
                changed.store(true, Ordering::SeqCst);
            }
            release.add_permits(1);
            assert!(next.await.unwrap().is_err());
            drop(stream);
        } else if mode == "error" {
            assert!(stream.next().await.unwrap().is_err());
            assert!(stream.next().await.is_none());
            drop(stream);
        } else {
            assert!(stream.next().await.is_none());
            drop(stream);
        }
        assert_eq!(admission.admissions.load(Ordering::SeqCst), 1, "{mode}");
        assert_eq!(admission.refunds.load(Ordering::SeqCst), 0, "{mode}");
        assert_eq!(
            admission.settlements.load(Ordering::SeqCst),
            usize::from(matches!(mode, "eof" | "error")),
            "{mode}"
        );
        assert_eq!(
            admission.abandonments.load(Ordering::SeqCst),
            usize::from(!matches!(mode, "eof" | "error")),
            "{mode}"
        );
    }
}

//! Real shared-controller acceptance through the bounded TUI transport, without terminal, credentials or external I/O.
use super::*;
use crate::session::{Input, OperationResult, SessionResult};
use ai_agents::AgentResponse;
use ai_agents::llm::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
};
use ai_agents::{AgentBuilder, ApprovalHandler, ApprovalRequest, TaskRunStatus};
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::task::JoinHandle;

const TASK: &str = "name: TuiFixture\nsystem_prompt: Produce a verified report.\ntools: []\nautonomy:\n  enabled: true\n  max_turns: 4\n  max_llm_calls: 4\n  completion: {response_contains: done}\n  premature_finish: {action: ask_user, on_unavailable: stop}\n";
struct Provider {
    calls: AtomicUsize,
}
#[async_trait::async_trait]
impl LLMProvider for Provider {
    async fn complete(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(LLMResponse::new(
            if n == 0 { "draft" } else { "done" },
            FinishReason::Stop,
        ))
    }
    async fn complete_stream(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<
        Box<dyn futures::Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
        LLMError,
    > {
        Ok(Box::new(futures::stream::empty()))
    }
    fn provider_name(&self) -> &str {
        "tui-offline"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
}
struct Harness {
    agent: Arc<RuntimeAgent>,
    provider: Arc<Provider>,
    input: mpsc::Sender<Input>,
    output: mpsc::Receiver<Output>,
    driver: JoinHandle<Result<()>>,
}
impl Harness {
    async fn new(yaml: &str, streaming: bool) -> Self {
        let (io, input, output) = FrontendSession::embedded();
        let provider = Arc::new(Provider {
            calls: AtomicUsize::new(0),
        });
        let agent = Arc::new(
            AgentBuilder::from_yaml(yaml)
                .unwrap()
                .llm(provider.clone())
                .approval_handler(io.bridge())
                .build_task_host()
                .await
                .unwrap(),
        );
        agent.set_question_handler(Some(io.bridge()));
        let mut config = CliReplConfig::default();
        if streaming {
            config.mode = crate::repl::ReplMode::Streaming;
        }
        let driver = tokio::spawn(run(io, agent.clone(), config));
        Self {
            agent,
            provider,
            input,
            output,
            driver,
        }
    }
    async fn next(&mut self) -> Output {
        tokio::time::timeout(Duration::from_secs(10), self.output.recv())
            .await
            .expect("foreground stalled")
            .expect("output closed")
    }
    async fn prompt(&mut self) -> u64 {
        loop {
            if let Output::Prompt { id, .. } = self.next().await {
                return id;
            }
        }
    }
    async fn result(&mut self) -> Box<SessionResult> {
        loop {
            if let Output::Result { result, .. } = self.next().await {
                return result;
            }
        }
    }
    async fn finish(mut self) {
        self.input.send(Input::Line("/quit".into())).await.unwrap();
        tokio::time::timeout(Duration::from_secs(10),async {loop {tokio::select! {result=&mut self.driver=>{result.unwrap().unwrap();break;},_=self.output.recv()=>{}}}}).await.expect("shutdown stalled");
    }
}
use std::time::Duration;

#[tokio::test]
async fn tui_endpoint_watch_loss_rejects_an_outstanding_ordinary_callback() {
    let (mut io, _input, mut output) = FrontendSession::embedded();
    let (endpoint, rx) = tokio::sync::watch::channel(crate::session::Endpoint::Open);
    io.set_endpoint(rx);
    let bridge = io.bridge();
    let (_, events) = mpsc::channel(1);
    let driver = tokio::spawn(async move {
        io.drive(
            async move {
                let response = ai_agents::tools::QuestionHandler::ask_question(
                    bridge.as_ref(),
                    ai_agents::tools::QuestionRequest {
                        question: "Decision".into(),
                        allow_other: true,
                        default: Some(serde_json::json!("must not submit")),
                        ..Default::default()
                    },
                )
                .await;
                assert!(!response.answered);
                assert!(response.unavailable);
                Ok(OperationResult::Chat(AgentResponse::new("unavailable")))
            },
            events,
        )
        .await
    });
    loop {
        if matches!(output.recv().await, Some(Output::Prompt { .. })) {
            break;
        }
    }
    endpoint.send(crate::session::Endpoint::Closed).unwrap();
    tokio::time::timeout(Duration::from_secs(2), driver)
        .await
        .expect("ordinary callback survived endpoint loss")
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn tui_priority_shutdown_survives_full_transport_queues() {
    let (mut io, input, _output) = FrontendSession::embedded();
    let (endpoint, rx) = tokio::sync::watch::channel(crate::session::Endpoint::Open);
    io.set_endpoint(rx);
    for _ in 0..32 {
        io.send_ui(Output::Text("full".into())).await.unwrap();
    }
    for _ in 0..16 {
        input
            .try_send(Input::UiLine {
                generation: 0,
                line: "/task list".into(),
            })
            .unwrap();
    }
    let provider = Arc::new(Provider {
        calls: AtomicUsize::new(1),
    });
    let agent = Arc::new(
        AgentBuilder::from_yaml(TASK)
            .unwrap()
            .llm(provider.clone())
            .build_task_host()
            .await
            .unwrap(),
    );
    let (tx, events) = mpsc::channel(32);
    let driver = tokio::spawn(async move {
        io.drive(
            crate::session::task_work(
                agent,
                TaskStreamOperation::Run {
                    objective: "Research".into(),
                    profile: None,
                },
                tx,
                false,
            ),
            events,
        )
        .await
    });
    tokio::time::timeout(Duration::from_secs(10), async {
        while provider.calls.load(Ordering::SeqCst) == 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    endpoint.send(crate::session::Endpoint::Shutdown).unwrap();
    let result = tokio::time::timeout(Duration::from_secs(10), driver)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(result.quit);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn tui_pre_admission_rejection_and_stale_generation_do_not_answer_new_work() {
    let mut h = Harness::new(TASK, false).await;
    h.input
        .send(Input::UiLine {
            generation: 1,
            line: "/task run --profile missing-objective".into(),
        })
        .await
        .unwrap();
    assert!(matches!(
        h.next().await,
        Output::Rejected { generation: 1, .. }
    ));
    assert_eq!(h.provider.calls.load(Ordering::SeqCst), 0);
    h.input
        .send(Input::UiLine {
            generation: 2,
            line: "/task run Research".into(),
        })
        .await
        .unwrap();
    let id = h.prompt().await;
    h.input
        .send(Input::UiLine {
            generation: 1,
            line: "continue".into(),
        })
        .await
        .unwrap();
    loop {
        if let Output::Diagnostic(text) = h.next().await
            && text.contains("stale operation")
        {
            break;
        }
    }
    assert_eq!(h.provider.calls.load(Ordering::SeqCst), 1);
    h.input
        .send(Input::Reply {
            id,
            line: "continue".into(),
        })
        .await
        .unwrap();
    let result = h.result().await;
    assert_eq!(
        result.results.values().next().unwrap().run.status,
        TaskRunStatus::Completed
    );
    h.finish().await;
}

#[tokio::test]
async fn tui_transport_exact_answer_resumes_same_run_without_tools_or_token_streaming() {
    let mut h = Harness::new(TASK, false).await;
    h.input
        .send(Input::Line("/task run 보고서 작성".into()))
        .await
        .unwrap();
    let id = h.prompt().await;
    assert_eq!(h.provider.calls.load(Ordering::SeqCst), 1);
    h.input
        .send(Input::Reply {
            id: id + 1,
            line: "continue".into(),
        })
        .await
        .unwrap();
    loop {
        if let Output::Diagnostic(text) = h.next().await
            && text.contains("stale")
        {
            break;
        }
    }
    assert_eq!(h.provider.calls.load(Ordering::SeqCst), 1);
    h.input
        .send(Input::Reply {
            id,
            line: serde_json::json!({"answered":true,"other_text":"cancel"}).to_string(),
        })
        .await
        .unwrap();
    let result = h.result().await;
    let OperationResult::Task(initial) = &result.primary else {
        panic!("not a task")
    };
    let done = result.results.get(&initial.run.key.run_id).unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(done.run.counters.llm_attempts, 2);
    assert_eq!(h.provider.calls.load(Ordering::SeqCst), 2);
    assert_eq!(done.run.objective, "보고서 작성");
    h.finish().await;
}

#[tokio::test]
async fn tui_transport_automatic_state_blocking_and_streamed_keep_task_lifecycle() {
    for streamed in [false, true] {
        let yaml = TASK
            .replace(
                "autonomy:\n",
                "states:\n  initial: report\n  states:\n    report:\n      autonomy:\n",
            )
            .lines()
            .map(|line| {
                if line.starts_with("  ")
                    && !line.starts_with("  initial:")
                    && !line.starts_with("  states:")
                    && !line.starts_with("    report:")
                    && !line.starts_with("      autonomy:")
                {
                    format!("      {line}")
                } else {
                    line.to_owned()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        let mut h = Harness::new(&yaml, streamed).await;
        h.input.send(Input::Line("Research".into())).await.unwrap();
        let id = h.prompt().await;
        h.input
            .send(Input::Reply {
                id,
                line: "continue".into(),
            })
            .await
            .unwrap();
        let result = h.result().await;
        assert_eq!(result.results.len(), 1);
        assert!(
            result
                .results
                .values()
                .all(|r| r.run.status == TaskRunStatus::Completed
                    && r.run.counters.llm_attempts == 2)
        );
        assert_eq!(h.provider.calls.load(Ordering::SeqCst), 2);
        h.finish().await;
    }
}

#[tokio::test]
async fn tui_transport_shutdown_of_volatile_pause_acknowledges_without_more_provider_work() {
    let mut h = Harness::new(TASK, false).await;
    h.input
        .send(Input::Line("/task run Research".into()))
        .await
        .unwrap();
    h.prompt().await;
    let agent = h.agent.clone();
    let provider = h.provider.clone();
    h.finish().await;
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    // Actual controller custody, not a UI marker, must permit another admitted task after cleanup.
    let next = agent.run_task("next", None).await.unwrap();
    assert_eq!(next.run.status, TaskRunStatus::Completed);
}

#[tokio::test]
async fn tui_transport_stale_ordinary_callback_cannot_approve_the_next_request() {
    let (mut io, input, mut output) = FrontendSession::embedded();
    let bridge = io.bridge();
    let (_, events) = mpsc::channel(1);
    let driver = tokio::spawn(async move {
        io.drive(
            async move {
                for _ in 0..2 {
                    let request = ApprovalRequest::new(
                        ai_agents::hitl::ApprovalTrigger::tool("echo", serde_json::json!({})),
                        "Approve?",
                    );
                    assert!(bridge.request_approval(request).await.is_approved());
                }
                Ok(OperationResult::Chat(AgentResponse::new("done")))
            },
            events,
        )
        .await
    });
    let mut ids = vec![];
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            match output.recv().await.unwrap() {
                Output::Prompt { id, .. } => {
                    ids.push(id);
                    if ids.len() == 2 {
                        input
                            .send(Input::Reply {
                                id: ids[0],
                                line: "yes".into(),
                            })
                            .await
                            .unwrap();
                    } else {
                        input
                            .send(Input::Reply {
                                id,
                                line: "yes".into(),
                            })
                            .await
                            .unwrap();
                    }
                }
                Output::Diagnostic(text) if text.contains("stale") => {
                    input
                        .send(Input::Reply {
                            id: ids[1],
                            line: "yes".into(),
                        })
                        .await
                        .unwrap();
                    break;
                }
                _ => {}
            }
        }
        driver.await.unwrap().unwrap();
    })
    .await
    .unwrap();
    assert_ne!(ids[0], ids[1]);
}

#[tokio::test]
async fn tui_transport_endpoint_loss_never_submits_a_default_or_continuation() {
    let mut h = Harness::new(TASK, false).await;
    h.input
        .send(Input::Line("/task run Research".into()))
        .await
        .unwrap();
    h.prompt().await;
    h.input.send(Input::Eof).await.unwrap();
    let result = h.result().await;
    assert!(
        result
            .results
            .values()
            .all(|r| r.run.status == TaskRunStatus::Incomplete)
    );
    assert_eq!(h.provider.calls.load(Ordering::SeqCst), 1);
    // EOF already withdraws the only endpoint; the original host loop then returns.
    h.driver.await.unwrap().unwrap();
}

#[tokio::test]
async fn tui_transport_full_display_does_not_hold_shutdown_settlement() {
    let (mut io, input, _output) = FrontendSession::embedded();
    for _ in 0..32 {
        io.send_ui(Output::Text("occupied".into())).await.unwrap();
    }
    let provider = Arc::new(Provider {
        calls: AtomicUsize::new(1),
    });
    let agent = Arc::new(
        AgentBuilder::from_yaml(TASK)
            .unwrap()
            .llm(provider.clone())
            .build_task_host()
            .await
            .unwrap(),
    );
    let (events, rx) = mpsc::channel(32);
    let driver = tokio::spawn(async move {
        io.drive(
            crate::session::task_work(
                agent,
                TaskStreamOperation::Run {
                    objective: "Research".into(),
                    profile: None,
                },
                events,
                false,
            ),
            rx,
        )
        .await
    });
    tokio::time::timeout(Duration::from_secs(10), async {
        while provider.calls.load(Ordering::SeqCst) == 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    input.send(Input::Line("/quit".into())).await.unwrap();
    let result = tokio::time::timeout(Duration::from_secs(10), driver)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(result.quit);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
}

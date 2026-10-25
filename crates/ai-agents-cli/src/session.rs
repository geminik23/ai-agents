//! Terminal-independent foreground I/O adapter; execution authority remains in the runtime controller.

use ai_agents::autonomy::{ExternalSignal, TaskResumeInput, TaskRunLimits};
use ai_agents::tools::{QuestionHandler, QuestionRequest, QuestionResponse};
use ai_agents::{
    Agent, AgentResponse, AgentStreamEvent, ApprovalHandler, ApprovalRequest, ApprovalResult,
    RuntimeAgent, TaskHostInteraction, TaskHostPrompt, TaskHostRun, TaskHostSession,
    TaskRunDisplay, TaskRunResult, TaskRunStatus, TaskStreamOperation,
};
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use futures::{StreamExt, future::BoxFuture, stream::FuturesUnordered};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    future::Future,
    io::{self, BufRead, Read, Write},
    sync::Arc,
};
use tokio::sync::{mpsc, oneshot};

const INPUT_BYTES: usize = 65_536;

enum Input {
    Line(String),
    Eof,
    Error,
}
enum Output {
    Text(String),
    Diagnostic(String),
    Flush(oneshot::Sender<()>),
}
enum Callback {
    Approval(ApprovalRequest, oneshot::Sender<ApprovalResult>),
    Question(QuestionRequest, oneshot::Sender<QuestionResponse>),
}

/// One input bridge is shared by ordinary callback UI; task responses never enter this callback protocol.
#[derive(Clone)]
pub(crate) struct InputBridge {
    tx: mpsc::Sender<Callback>,
    show_context: Arc<std::sync::atomic::AtomicBool>,
}
impl InputBridge {
    pub(crate) fn set_show_context(&self, available: bool) {
        self.show_context
            .store(available, std::sync::atomic::Ordering::Release);
    }
}
#[async_trait]
impl ApprovalHandler for InputBridge {
    async fn request_approval(&self, request: ApprovalRequest) -> ApprovalResult {
        let (tx, rx) = oneshot::channel();
        if self.tx.send(Callback::Approval(request, tx)).await.is_err() {
            return ApprovalResult::rejected(None);
        }
        rx.await.unwrap_or_else(|_| ApprovalResult::rejected(None))
    }
}
#[async_trait]
impl QuestionHandler for InputBridge {
    async fn ask_question(&self, request: QuestionRequest) -> QuestionResponse {
        let (tx, rx) = oneshot::channel();
        if self.tx.send(Callback::Question(request, tx)).await.is_err() {
            return unavailable_question();
        }
        rx.await.unwrap_or_else(|_| unavailable_question())
    }
}
fn unavailable_question() -> QuestionResponse {
    QuestionResponse {
        answered: false,
        selected: vec![],
        other_text: None,
        timed_out: false,
        unavailable: true,
    }
}

/// Bounded display forwarding is independent of exact task results and backpressures the original foreground future.
pub(crate) type DisplaySender = mpsc::Sender<String>;
#[derive(Debug)]
pub(crate) struct ProtectedSession;
impl std::fmt::Display for ProtectedSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("foreground task ownership remains protected; inspect before recovery or further execution")
    }
}
impl std::error::Error for ProtectedSession {}
#[derive(Debug)]
struct InputRejection(String);
impl std::fmt::Display for InputRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for InputRejection {}
pub(crate) enum OperationResult {
    Chat(AgentResponse),
    Task(Box<TaskRunResult>),
}
pub(crate) struct SessionResult {
    pub primary: OperationResult,
    pub results: BTreeMap<String, TaskRunResult>,
    pub quit: bool,
}
struct TaskPrompt {
    run: TaskHostRun,
    interaction: TaskHostInteraction,
}
enum Prompt {
    Task(Box<TaskPrompt>),
    Callback(Callback),
}
struct View {
    run: TaskHostRun,
    display: TaskRunDisplay,
    interaction: Option<TaskHostInteraction>,
}
type Control = BoxFuture<'static, (String, Result<Option<TaskRunResult>>)>;
type InputResponse = std::result::Result<Option<(String, Control)>, (Box<Prompt>, anyhow::Error)>;

/// Owns only terminal input/output and polling, not task state, execution ordering, admission or cleanup semantics.
pub(crate) struct FrontendSession {
    input: mpsc::Receiver<Input>,
    queued: VecDeque<String>,
    callbacks: mpsc::Receiver<Callback>,
    bridge: Arc<InputBridge>,
    eof: bool,
    json: bool,
    output: mpsc::Sender<Output>,
    output_failed: mpsc::Receiver<()>,
    suppressed_output: bool,
    interrupt: std::pin::Pin<Box<dyn Future<Output = io::Result<()>> + Send>>,
}

impl FrontendSession {
    pub(crate) fn terminal(json: bool, input_available: bool) -> Self {
        let (tx, input) = mpsc::channel(16);
        if input_available {
            // The single read-only OS input thread cannot keep Tokio shutdown waiting for an uninterruptible stdin read.
            // It owns no runtime or work capability and stops when its bounded receiver disappears or EOF is observed.
            std::thread::spawn(move || {
                let stdin = io::stdin();
                let mut reader = stdin.lock();
                loop {
                    let mut line = String::new();
                    let read = reader
                        .by_ref()
                        .take((INPUT_BYTES + 1) as u64)
                        .read_line(&mut line);
                    let event = match read {
                        Ok(0) => Input::Eof,
                        Ok(_) if line.len() <= INPUT_BYTES => {
                            Input::Line(line.trim_end_matches(['\r', '\n']).to_string())
                        }
                        _ => Input::Error,
                    };
                    let terminal = !matches!(event, Input::Line(_));
                    if tx.blocking_send(event).is_err() || terminal {
                        break;
                    }
                }
            });
        }
        let (output, mut writes) = mpsc::channel::<Output>(32);
        let (failed_tx, output_failed) = mpsc::channel(1);
        // A slow or closed output pipe never owns task polling or runtime locks.
        std::thread::spawn(move || {
            let stdout = io::stdout();
            while let Some(message) = writes.blocking_recv() {
                let text = match message {
                    Output::Text(text) => text,
                    Output::Diagnostic(text) => {
                        if io::stderr()
                            .write_all(text.as_bytes())
                            .and_then(|_| io::stderr().flush())
                            .is_err()
                        {
                            let _ = failed_tx.blocking_send(());
                            break;
                        }
                        continue;
                    }
                    Output::Flush(ack) => {
                        let _ = ack.send(());
                        continue;
                    }
                };
                let mut writer = stdout.lock();
                if writer
                    .write_all(text.as_bytes())
                    .and_then(|_| writer.flush())
                    .is_err()
                {
                    let _ = failed_tx.blocking_send(());
                    break;
                }
            }
        });
        let (callback_tx, callbacks) = mpsc::channel(32);
        Self {
            input,
            queued: VecDeque::new(),
            callbacks,
            bridge: Arc::new(InputBridge {
                tx: callback_tx,
                show_context: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            }),
            eof: !input_available,
            json,
            output,
            output_failed,
            suppressed_output: false,
            interrupt: Box::pin(tokio::signal::ctrl_c()),
        }
    }

    pub(crate) fn output_suppressed(&self) -> bool {
        self.suppressed_output
    }

    pub(crate) fn output_closed(&self) -> bool {
        self.output.is_closed()
    }

    pub(crate) fn bridge(&self) -> Arc<InputBridge> {
        self.bridge.clone()
    }

    pub(crate) async fn write(&self, text: impl Into<String>) -> Result<()> {
        self.output
            .send(Output::Text(text.into()))
            .await
            .map_err(|_| anyhow::anyhow!("output unavailable; task cleanup is not implied"))
    }

    pub(crate) async fn finish(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.output
            .send(Output::Flush(tx))
            .await
            .map_err(|_| anyhow::anyhow!("output unavailable"))?;
        rx.await.context("output flush was not acknowledged")
    }

    pub(crate) async fn next_line(&mut self) -> Result<Option<String>> {
        if let Some(line) = self.queued.pop_front() {
            return Ok(Some(line));
        }
        if self.eof {
            return Ok(None);
        }
        let input = tokio::select! {
            input=self.input.recv()=>input,
            _=&mut self.interrupt=>{self.eof=true;None},
        };
        match input {
            Some(Input::Line(line)) => Ok(Some(line)),
            Some(Input::Error) => {
                self.eof = true;
                bail!("stdin failed or input exceeds 64 KiB");
            }
            _ => {
                self.eof = true;
                Ok(None)
            }
        }
    }

    /// Polls the original operation and exact controls together, including ordinary parents waiting on independently hosted children.
    pub(crate) async fn drive<F>(
        &mut self,
        work: F,
        mut events: mpsc::Receiver<String>,
    ) -> Result<SessionResult>
    where
        F: Future<Output = Result<OperationResult>> + Send,
    {
        // Register and retain the interrupt listener before admitting work; recreating it on every select can lose a signal between receivers.
        let early_signal = futures::future::poll_fn(|cx| {
            std::task::Poll::Ready(match self.interrupt.as_mut().poll(cx) {
                std::task::Poll::Ready(result) => Some(result),
                std::task::Poll::Pending => None,
            })
        })
        .await;
        if let Some(result) = early_signal {
            self.eof = true;
            self.interrupt = Box::pin(tokio::signal::ctrl_c());
            result?;
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "foreground interrupted before task admission",
            )
            .into());
        }
        let host = TaskHostSession::new(!self.eof || !self.queued.is_empty());
        let scoped = host.clone();
        let mut work = Box::pin(async move { scoped.scope(work).await });
        let mut primary = None;
        let mut controls = FuturesUnordered::<Control>::new();
        let (updates_tx, mut updates_rx) = mpsc::channel::<String>(32);
        let mut work_finished = false;
        let mut failed_runs = HashSet::<String>::new();
        let mut observation_tick = tokio::time::interval(std::time::Duration::from_millis(100));
        let mut busy = HashSet::<String>::new();
        let mut cancelled = HashSet::<String>::new();
        let mut results = BTreeMap::new();
        let mut views: Vec<View> = vec![];
        let mut refresh: Option<BoxFuture<'static, Result<Vec<View>>>> = None;
        let mut dirty = true;
        let mut presented: Option<Prompt> = None;
        let mut pending_output = None::<Output>;
        let mut stopping = false;
        let mut quit = false;
        let mut output_lost = false;
        let mut events_open = true;
        let mut operation_error = None;
        let mut input_closed = false;
        loop {
            // Once shutdown begins, display backpressure must not hold execution or control futures hostage.
            // The read-only output worker may remain blocked, but it owns no runtime work and final delivery is explicitly suppressed.
            if stopping {
                pending_output = None;
                if let Some(Prompt::Callback(callback)) = presented.take() {
                    reject_callback(callback);
                }
            }
            let callback_closed = match &presented {
                Some(Prompt::Callback(Callback::Approval(_, tx))) => tx.is_closed(),
                Some(Prompt::Callback(Callback::Question(_, tx))) => tx.is_closed(),
                _ => false,
            };
            if callback_closed {
                presented = None;
            }
            if self.eof && self.queued.is_empty() && !input_closed {
                host.close_input();
                input_closed = true;
                dirty = true;
            }
            if dirty && refresh.is_none() {
                let session = host.clone();
                refresh = Some(Box::pin(async move { inspect(&session).await }));
                dirty = false;
            }
            if refresh.is_none() && !dirty {
                for view in &mut views {
                    let id = view.run.key.run_id.clone();
                    if busy.contains(&id) || failed_runs.contains(&id) {
                        continue;
                    }
                    if view.display.summary.status == TaskRunStatus::Paused
                        && view.run.published_revision().is_some()
                        && (stopping
                            || !host.interaction_available()
                            || view.display.summary.cancel_requested)
                    {
                        let run = view.run.clone();
                        let revision = view.display.summary.revision;
                        let scoped = host.clone();
                        busy.insert(id.clone());
                        let cancel = stopping || view.display.summary.cancel_requested;
                        controls.push(Box::pin(async move {
                            let result = scoped
                                .scope(async {
                                    if cancel {
                                        run.runner.cancel_paused(&run.key.run_id, revision).await
                                    } else {
                                        run.runner
                                            .input_unavailable_paused(&run.key.run_id, revision)
                                            .await
                                    }
                                })
                                .await
                                .map(Some)
                                .map_err(Into::into);
                            (id, result)
                        }));
                        presented = None;
                    } else if stopping
                        && view.run.execution_admitted()
                        && view.display.summary.status == TaskRunStatus::Running
                        && cancelled.insert(id.clone())
                    {
                        let run = view.run.clone();
                        let scoped = host.clone();
                        busy.insert(id.clone());
                        controls.push(Box::pin(async move {
                            let result = scoped
                                .scope(run.runner.request_running_cancel(&run.key.run_id))
                                .await
                                .map(|_| None)
                                .map_err(Into::into);
                            (id, result)
                        }));
                    }
                }
                if presented.is_none()
                    && pending_output.is_none()
                    && host.interaction_available()
                    && !stopping
                    && let Some(view) = views
                        .iter_mut()
                        .find(|v| v.interaction.is_some() && !busy.contains(&v.run.key.run_id))
                {
                    let prompt = TaskPrompt {
                        run: view.run.clone(),
                        interaction: view.interaction.take().unwrap(),
                    };
                    pending_output = Some(diagnostic(render_task_prompt(&prompt), self.json));
                    presented = Some(Prompt::Task(Box::new(prompt)));
                }
                // RecoveryRequired ends this foreground operation but deliberately retains protected runtime custody.
                let all_settled = views.iter().all(|view| {
                    view.display.summary.status.is_terminal()
                        || (view.display.summary.status == TaskRunStatus::RecoveryRequired
                            && view.run.published_revision() == Some(view.display.summary.revision))
                });
                if primary.is_some()
                    && controls.is_empty()
                    && all_settled
                    && presented.is_none()
                    && !events_open
                    && updates_rx.is_empty()
                    && pending_output.is_none()
                {
                    break;
                }
                if operation_error.is_some()
                    && work_finished
                    && controls.is_empty()
                    && !dirty
                    && pending_output.is_none()
                {
                    break;
                }
            }
            // Buffered plain replies wait for a real prompt; input read-ahead is bounded and never creates speculative chat turns.
            if presented.is_some() && !self.queued.is_empty() {
                let line = self.queued.pop_front().unwrap();
                match respond(
                    presented.take().unwrap(),
                    &line,
                    host.clone(),
                    updates_tx.clone(),
                    self.json,
                ) {
                    Ok(Some((id, control))) => {
                        busy.insert(id);
                        controls.push(control);
                    }
                    Ok(None) => {
                        dirty = true;
                    }
                    Err((prompt, error)) => {
                        presented = Some(*prompt);
                        pending_output =
                            Some(diagnostic(format!("[input error] {error}\n"), self.json));
                    }
                }
                continue;
            }
            tokio::select! {
                result=&mut work, if !work_finished => {
                    work_finished=true;
                    match result {
                        Ok(value) => { if let OperationResult::Task(task)=&value { record_result(&mut results,(**task).clone()); } primary=Some(value); }
                        Err(error) => { operation_error=Some(error); stopping=true; host.close_input(); }
                    }
                    dirty=true;
                }
                Some((id,result))=controls.next(), if !controls.is_empty() => {
                    busy.remove(&id);
                    match result {
                        Ok(Some(result)) => {record_result(&mut results,result);}
                        Ok(None) => {},
                        Err(error) => {
                            // Never retry an ambiguous write or rebind a stale response to a newer revision.
                            if stopping || !host.interaction_available() || !response_rejection(&error) {
                                failed_runs.insert(id); operation_error=Some(error); stopping=true;host.close_input();
                            }
                            else { pending_output=Some(diagnostic(format!("[control rejected] {error}\n"),self.json)); }
                        }
                    }
                    dirty=true;
                }
                value=async { refresh.as_mut().unwrap().await }, if refresh.is_some() => {
                    refresh=None;
                    match value { Ok(new_views)=>views=new_views, Err(error)=>{operation_error=Some(error);stopping=true;host.close_input();} }
                }
                _=host.changed() => {dirty=true;}
                _=observation_tick.tick(), if !host.runs().is_empty() => {dirty=true;}
                update=updates_rx.recv(), if pending_output.is_none() => {if let Some(text)=update && !stopping {pending_output=Some(Output::Text(text));}}
                callback=self.callbacks.recv(), if presented.is_none() && pending_output.is_none() => {
                    if let Some(callback)=callback {
                        if stopping || !host.interaction_available() { reject_callback(callback); }
                        else { pending_output=Some(diagnostic(render_callback(&callback,self.bridge.show_context.load(std::sync::atomic::Ordering::Acquire)),self.json)); presented=Some(Prompt::Callback(callback)); }
                    }
                }
                event=events.recv(), if events_open && pending_output.is_none() => {
                    match event { Some(text)=>{if !stopping {pending_output=Some(Output::Text(text));}}, None=>events_open=false }
                }
                permit=self.output.reserve(), if pending_output.is_some() && !output_lost && !stopping => {
                    match permit {
                        Ok(permit)=>{ permit.send(pending_output.take().unwrap()); }
                        Err(_)=>{output_lost=true;self.suppressed_output=true;pending_output=None;stopping=true;host.close_input();}
                    }
                }
                failed=self.output_failed.recv(), if !output_lost => {
                    if failed.is_some() {output_lost=true;self.suppressed_output=true;pending_output=None;stopping=true;host.close_input();}
                }
                input=self.input.recv(), if !self.eof && self.queued.len()<16 => {
                    match input {
                        Some(Input::Line(line)) if (line=="/quit" || line=="/exit") && !host.runs().is_empty() => {stopping=true;quit=true;self.suppressed_output=true;host.close_input(); if let Some(Prompt::Callback(callback))=presented.take(){reject_callback(callback);} dirty=true;}
                        Some(Input::Line(line)) if line.starts_with("/task ") && presented.is_none() => {
                            match command(&line, &host,updates_tx.clone(),self.json) {
                                Ok((id,control))=>{busy.insert(id);controls.push(control);},
                                Err(error)=>pending_output=Some(diagnostic(format!("[control rejected] {error}\n"),self.json)),
                            }
                        }
                        Some(Input::Line(line))=>self.queued.push_back(line),
                        _=>{ self.eof=true; if self.queued.is_empty(){host.close_input(); if let Some(Prompt::Callback(callback))=presented.take(){reject_callback(callback);} dirty=true;} }
                    }
                }
                signal=&mut self.interrupt => {
                    self.interrupt=Box::pin(tokio::signal::ctrl_c());
                    if signal.is_ok() {
                        if stopping {eprintln!("Forced exit: cleanup is unconfirmed; inspect task ownership before recovery."); std::process::exit(1);}
                        stopping=true;quit=true;self.suppressed_output=true;host.close_input(); if let Some(Prompt::Callback(callback))=presented.take(){reject_callback(callback);} dirty=true;
                    } else {
                        self.interrupt=Box::pin(std::future::pending());
                        operation_error=Some(anyhow::anyhow!("interrupt delivery failed"));stopping=true;host.close_input();dirty=true;
                    }
                }
            }
            if output_lost {
                pending_output = None;
            }
        }
        if output_lost {
            bail!("output unavailable; foreground settlement was attempted, not rollback");
        }
        if let Some(error) = operation_error {
            return Err(if host.has_protected_work() {
                error.context(ProtectedSession)
            } else {
                error
            });
        }
        Ok(SessionResult {
            primary: primary.context("foreground operation returned no result")?,
            results,
            quit,
        })
    }
}

async fn inspect(session: &TaskHostSession) -> Result<Vec<View>> {
    let mut views = vec![];
    for run in session.runs() {
        // Admission can publish its handle before initial storage setup; no missing row is treated as execution success.
        if run
            .runner
            .task_store()
            .load(&run.key.run_id)
            .await?
            .is_none()
        {
            continue;
        }
        let display = run.runner.task_display(&run.key.run_id).await?;
        let interaction = if display.summary.status == TaskRunStatus::Paused
            && run.published_revision() == Some(display.summary.revision)
        {
            match run
                .runner
                .task_interaction(&run.key.run_id, display.summary.revision)
                .await
            {
                Ok(interaction) => interaction,
                Err(ai_agents::error::AgentError::TaskRunStorage(
                    ai_agents::TaskRunStorageError::Conflict,
                )) => None,
                Err(error) => return Err(error.into()),
            }
        } else {
            None
        };
        views.push(View {
            run,
            display,
            interaction,
        });
    }
    Ok(views)
}

fn diagnostic(text: String, json_mode: bool) -> Output {
    if json_mode {
        Output::Diagnostic(text)
    } else {
        Output::Text(text)
    }
}

fn render_task_prompt(prompt: &TaskPrompt) -> String {
    let i = &prompt.interaction;
    let description = match &i.prompt {
        TaskHostPrompt::Approval { message, context } => {
            format!(
                "Approval: {message}\n{}\nEnter yes / no, or JSON {{\"changes\":{{...}}}}",
                approval_context(context)
            )
        }
        TaskHostPrompt::Question(q) => format!(
            "Question: {}\nOptions: {:?}; multi-select: {}; free text: {}\nEnter option numbers, free text, or an exact JSON answer",
            q.question, q.options, q.multi_select, q.allow_other
        ),
        TaskHostPrompt::Continue => "Host continuation: enter continue (or stop/cancel)".into(),
        TaskHostPrompt::Capacity {
            reason,
            original_limits,
        } => format!(
            "Capacity: {reason}\nCurrent limits: {original_limits:?}\nUse /task limits RUN REV JSON, or stop/cancel"
        ),
        TaskHostPrompt::StageReview {
            next_stage,
            finishes_lifecycle,
            instruction,
        } => format!(
            "Stage review: next={next_stage:?}, finishes_lifecycle={finishes_lifecycle}\n{}\nEnter yes / no",
            instruction.as_deref().unwrap_or("")
        ),
        TaskHostPrompt::PlanReview { instruction } => {
            format!("Plan review: {instruction}\nEnter yes / no")
        }
        TaskHostPrompt::ExternalCondition { binding } => format!(
            "External condition: {binding}\nEnter explicit signal JSON with event_id, target_revision and payload, or stop/cancel"
        ),
        TaskHostPrompt::ValidationObservation => {
            "Validation observation: enter yes / no, or an exact JSON answer".into()
        }
    };
    // Exact prompts are deliberately a local interaction surface, never copied into JSON event/export output.
    format!(
        "\n[task {} revision {} request {}]\n{description}\nReply > ",
        i.key.run_id, i.revision, i.request_id
    )
}
// Only operation-review fields are shown locally; diagnostics and JSON task exports never receive this exact prompt context.
fn approval_context(context: &Value) -> String {
    let selected = context
        .as_object()
        .map(|map| {
            map.iter()
                .filter(|(key, _)| {
                    matches!(
                        key.as_str(),
                        "tool"
                            | "tool_name"
                            | "arguments"
                            | "args"
                            | "path"
                            | "command"
                            | "operation"
                            | "summary"
                            | "description"
                            | "diff"
                            | "from"
                            | "to"
                            | "plan"
                    )
                })
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect::<serde_json::Map<_, _>>()
        })
        .unwrap_or_default();
    let text = serde_json::to_string_pretty(&selected).unwrap_or_default();
    if text.chars().count() > 16_384 {
        format!(
            "{}\n[review display truncated; stop if the operation cannot be verified]",
            text.chars().take(16_384).collect::<String>()
        )
    } else {
        text
    }
}

fn render_callback(callback: &Callback, show_context: bool) -> String {
    match callback {
        Callback::Approval(request, _) => {
            format!(
                "\n[ordinary approval] {}\n{}\nApprove? [y/N] ",
                request.message,
                if show_context {
                    approval_context(&json!(request.context))
                } else {
                    String::new()
                }
            )
        }
        Callback::Question(request, _) => format!(
            "\n[ordinary question] {}\nOptions: {:?}\nReply > ",
            request.question, request.options
        ),
    }
}
fn reject_callback(callback: Callback) {
    match callback {
        Callback::Approval(_, tx) => {
            let _ = tx.send(ApprovalResult::rejected(None));
        }
        Callback::Question(_, tx) => {
            let _ = tx.send(unavailable_question());
        }
    }
}
fn approval(line: &str) -> Result<ApprovalResult> {
    match line.trim().to_ascii_lowercase().as_str() {
        "yes" | "y" | "approve" => Ok(ApprovalResult::approved()),
        "" | "no" | "n" | "reject" => Ok(ApprovalResult::rejected(None)),
        _ => {
            let value: Value =
                serde_json::from_str(line).context("expected explicit yes/no or changes JSON")?;
            Ok(ApprovalResult::modified(serde_json::from_value(
                value.get("changes").cloned().context("missing changes")?,
            )?))
        }
    }
}
fn record_result(results: &mut BTreeMap<String, TaskRunResult>, result: TaskRunResult) {
    let id = result.run.key.run_id.clone();
    if results
        .get(&id)
        .is_none_or(|old| !old.run.status.is_terminal() && old.run.revision < result.run.revision)
    {
        results.insert(id, result);
    }
}

fn response_rejection(error: &anyhow::Error) -> bool {
    if error.is::<InputRejection>() {
        return true;
    }
    use ai_agents::TaskRunStorageError;
    use ai_agents::error::AgentError;
    matches!(
        error.downcast_ref::<AgentError>(),
        Some(
            AgentError::Config(_)
                | AgentError::InvalidSpec(_)
                | AgentError::HITLTimeout
                | AgentError::TaskRunStorage(
                    TaskRunStorageError::Conflict
                        | TaskRunStorageError::NotResumable
                        | TaskRunStorageError::AdmissionExpired
                )
        )
    )
}

fn event_text(event: &ai_agents::TaskRunEvent, json_mode: bool) -> String {
    if json_mode {
        format!(
            "{}\n",
            serde_json::to_string(event).expect("bounded task event serialization")
        )
    } else {
        format!(
            "[task {} revision {:?}: {:?}] {}\n",
            event.run_id.as_deref().unwrap_or("pending"),
            event.revision,
            event.kind,
            event.data
        )
    }
}

fn question(request: &QuestionRequest, line: &str) -> Result<Value> {
    if line.trim().is_empty() {
        let default = request
            .default
            .clone()
            .context("empty input is not an answer; no default is configured")?;
        return serde_json::to_value(crate::question::response_from_default(Some(default), false))
            .map_err(Into::into);
    }
    if let Ok(value) = serde_json::from_str::<Value>(line)
        && (value.is_object() || value.is_string())
    {
        return Ok(value);
    }
    let numbers = line
        .split(',')
        .map(|part| part.trim().parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>();
    if let Ok(numbers) = numbers
        && !request.options.is_empty()
    {
        let selected = numbers
            .into_iter()
            .map(|n| {
                n.checked_sub(1)
                    .and_then(|i| request.options.get(i))
                    .cloned()
                    .context("option out of range")
            })
            .collect::<Result<Vec<_>>>()?;
        if !request.multi_select && selected.len() != 1 {
            bail!("select exactly one option");
        }
        return Ok(json!({"answered":true,"selected":selected}));
    }
    if request.options.iter().any(|option| option == line) {
        return Ok(json!({"answered":true,"selected":[line]}));
    }
    if request.allow_other {
        Ok(json!({"answered":true,"other_text":line}))
    } else {
        bail!("select a declared option or supply a valid JSON answer")
    }
}
fn respond(
    prompt: Prompt,
    line: &str,
    session: TaskHostSession,
    updates: DisplaySender,
    json_mode: bool,
) -> InputResponse {
    match prompt {
        Prompt::Callback(callback) => match callback {
            Callback::Approval(request, tx) => match approval(line) {
                Ok(answer) => {
                    let _ = tx.send(answer);
                    Ok(None)
                }
                Err(error) => Err((
                    Box::new(Prompt::Callback(Callback::Approval(request, tx))),
                    error,
                )),
            },
            Callback::Question(request, tx) => match question(&request, line).and_then(|v| {
                if v.is_string() {
                    Ok(QuestionResponse {
                        answered: true,
                        selected: vec![v.as_str().unwrap().into()],
                        other_text: None,
                        timed_out: false,
                        unavailable: false,
                    })
                } else {
                    serde_json::from_value(v).map_err(Into::into)
                }
            }) {
                Ok(answer) => {
                    let _ = tx.send(answer);
                    Ok(None)
                }
                Err(error) => Err((
                    Box::new(Prompt::Callback(Callback::Question(request, tx))),
                    error,
                )),
            },
        },
        Prompt::Task(prompt) => {
            let operation = if line.starts_with("/task ") {
                return match command(line, &session, updates.clone(), json_mode) {
                    Ok(control) => Ok(Some(control)),
                    Err(error) => Err((Box::new(Prompt::Task(prompt)), error)),
                };
            } else {
                match response_operation(&prompt, line) {
                    Ok(value) => value,
                    Err(error) => return Err((Box::new(Prompt::Task(prompt)), error)),
                }
            };
            let id = prompt.run.key.run_id.clone();
            let run = prompt.run;
            Ok(Some((
                id.clone(),
                Box::pin(async move {
                    let result = session
                        .scope(async {
                            let mut stream =
                                run.runner.stream_operation(operation, Default::default())?;
                            while let Some(event) = stream.next_event().await {
                                updates.send(event_text(&event, json_mode)).await.ok();
                            }
                            stream
                                .take_result()
                                .context("control returned no exact result")?
                                .map_err(Into::into)
                        })
                        .await
                        .map(Some);
                    (id, result)
                }),
            )))
        }
    }
}
fn response_operation(prompt: &TaskPrompt, line: &str) -> Result<TaskStreamOperation> {
    let i = &prompt.interaction;
    let base = |input| TaskStreamOperation::Resume {
        run_id: i.key.run_id.clone(),
        revision: i.revision,
        input,
    };
    if line == "cancel" {
        return Ok(TaskStreamOperation::Cancel {
            run_id: i.key.run_id.clone(),
            revision: i.revision,
        });
    }
    if line == "stop" {
        return Ok(TaskStreamOperation::Stop {
            run_id: i.key.run_id.clone(),
            revision: i.revision,
            reason: "host requested stop".into(),
        });
    }
    if let Some(review) = &i.restoration_review {
        // A fresh restoration review is a distinct decision and cannot also consume the original pending answer.
        if !review.approved {
            bail!(
                "explicit restoration review is required through its bound host API before answering"
            );
        }
    }
    Ok(match &i.prompt {
        TaskHostPrompt::Question(request) => base(TaskResumeInput::UserAnswer {
            request_id: i.request_id.clone(),
            answer: question(request, line)?,
        }),
        TaskHostPrompt::Continue => {
            if line != "continue" {
                bail!("enter continue, stop or cancel");
            }
            base(TaskResumeInput::Continue)
        }
        TaskHostPrompt::Capacity { .. } => {
            if line != "continue" {
                bail!(
                    "revise limits with /task limits RUN REV JSON, then explicitly continue, or stop/cancel"
                );
            }
            base(TaskResumeInput::Continue)
        }
        TaskHostPrompt::ExternalCondition { .. } => {
            let value: Value = serde_json::from_str(line)?;
            base(TaskResumeInput::ExternalSignal(ExternalSignal {
                request_id: i.request_id.clone(),
                event_id: value["event_id"]
                    .as_str()
                    .context("missing event_id")?
                    .into(),
                target_revision: value["target_revision"].as_str().map(str::to_owned),
                payload: value.get("payload").cloned().unwrap_or(Value::Null),
            }))
        }
        TaskHostPrompt::ValidationObservation => {
            if let Ok(result) = approval(line) {
                base(TaskResumeInput::Approval {
                    request_id: i.request_id.clone(),
                    result,
                })
            } else {
                base(TaskResumeInput::UserAnswer {
                    request_id: i.request_id.clone(),
                    answer: serde_json::from_str(line)?,
                })
            }
        }
        _ => base(TaskResumeInput::Approval {
            request_id: i.request_id.clone(),
            result: approval(line)?,
        }),
    })
}

fn command(
    line: &str,
    session: &TaskHostSession,
    updates: DisplaySender,
    json_mode: bool,
) -> Result<(String, Control)> {
    let mut words = line.splitn(5, ' ');
    let _ = words.next();
    let action = words.next().context("missing task action")?.to_owned();
    if !matches!(
        action.as_str(),
        "list" | "show" | "resume" | "cancel" | "stop" | "objective" | "limits"
    ) {
        bail!("supported controls: list/show/resume/cancel/stop/objective/limits");
    }
    if action == "list" {
        let session = session.clone();
        return Ok((
            String::new(),
            Box::pin(async move {
                let result = async {
                    for run in session.runs() {
                        let view = run.runner.task_display(&run.key.run_id).await?;
                        updates
                            .send(format!("{}\n", serde_json::to_string(&view.summary)?))
                            .await
                            .ok();
                    }
                    Ok(None)
                }
                .await;
                (String::new(), result)
            }),
        ));
    }
    let id = words
        .next()
        .context("expected /task ACTION RUN REV [PAYLOAD]")?
        .to_owned();
    let run = session
        .runs()
        .into_iter()
        .find(|run| run.key.run_id == id)
        .context("run is not a participant in this foreground session")?;
    if action == "show" {
        return Ok((
            id.clone(),
            Box::pin(async move {
                let result = async {
                    let view = run.runner.task_display(&id).await?;
                    updates
                        .send(format!(
                            "{}{}\n",
                            if json_mode { "" } else { "\n" },
                            serde_json::to_string(&view)?
                        ))
                        .await
                        .ok();
                    Ok(None)
                }
                .await;
                (id, result)
            }),
        ));
    }
    let revision = words
        .next()
        .context("missing exact revision")?
        .parse::<u64>()?;
    let payload = words.next().unwrap_or("").to_owned();
    if action == "resume" {
        payload.split_once(' ').context("expected REQUEST ANSWER")?;
    }
    if action == "objective" && payload.trim().is_empty() {
        bail!("objective must not be empty");
    }
    if action == "limits" {
        serde_json::from_str::<TaskRunLimits>(&payload)?;
    }
    let run = session
        .runs()
        .into_iter()
        .find(|run| run.key.run_id == id)
        .context("run is not a participant in this foreground session")?;
    let session = session.clone();
    let result_id = id.clone();
    Ok((
        id.clone(),
        Box::pin(async move {
            let result = session
                .scope(async {
                    if action == "cancel" {
                        let view = run.runner.task_display(&id).await?;
                        if view.summary.status == TaskRunStatus::Paused {
                            return run
                                .runner
                                .cancel_paused(&id, revision)
                                .await
                                .map(Some)
                                .map_err(Into::into);
                        }
                        let receipt = run.runner.cancel_task(&id, revision).await?;
                        updates
                            .send(format!("{}\n", serde_json::to_string(&receipt)?))
                            .await
                            .ok();
                        return Ok(None);
                    }
                    let operation = match action.as_str() {
                        "stop" => TaskStreamOperation::Stop {
                            run_id: id,
                            revision,
                            reason: if payload.is_empty() {
                                "host requested stop".into()
                            } else {
                                payload
                            },
                        },
                        "objective" => TaskStreamOperation::ModifyObjective {
                            run_id: id,
                            revision,
                            objective: payload,
                            reason: "foreground objective revision".into(),
                        },
                        "limits" => TaskStreamOperation::ReviseLimits {
                            run_id: id,
                            revision,
                            limits: serde_json::from_str::<TaskRunLimits>(&payload)?,
                            reason: "foreground limit revision".into(),
                        },
                        "resume" => {
                            let (request, answer) =
                                payload.split_once(' ').context("expected REQUEST ANSWER")?;
                            let interaction = run
                                .runner
                                .task_interaction(&id, revision)
                                .await?
                                .ok_or(ai_agents::TaskRunStorageError::NotResumable)?;
                            if interaction.request_id != request {
                                return Err(ai_agents::error::AgentError::Config(
                                    "request identity mismatch".into(),
                                )
                                .into());
                            }
                            response_operation(
                                &TaskPrompt {
                                    run: run.clone(),
                                    interaction,
                                },
                                answer,
                            )
                            .map_err(|error| InputRejection(error.to_string()))?
                        }
                        _ => bail!("supported controls: resume/cancel/stop/objective/limits"),
                    };
                    let mut stream = run.runner.stream_operation(operation, Default::default())?;
                    while let Some(event) = stream.next_event().await {
                        updates.send(event_text(&event, json_mode)).await.ok();
                    }
                    Ok(Some(
                        stream
                            .take_result()
                            .context("control returned no result")??,
                    ))
                })
                .await;
            (result_id, result)
        }),
    ))
}

pub(crate) async fn task_work(
    agent: Arc<RuntimeAgent>,
    operation: TaskStreamOperation,
    events: DisplaySender,
    json_mode: bool,
) -> Result<OperationResult> {
    let runner = agent.task_runner()?;
    let mut stream = agent.task_operation_stream(&runner, operation, Default::default())?;
    while let Some(event) = stream.next_event().await {
        let text = event_text(&event, json_mode);
        if events.send(text).await.is_err() {
            return stream
                .close()
                .await
                .context("interrupted before execution acknowledgement")?
                .map(|r| OperationResult::Task(Box::new(r)))
                .map_err(Into::into);
        }
    }
    Ok(OperationResult::Task(Box::new(
        stream
            .take_result()
            .context("task stream ended without an exact result")??,
    )))
}

pub(crate) async fn chat_work(
    agent: Arc<RuntimeAgent>,
    input: String,
    config: crate::repl::CliReplConfig,
    events: DisplaySender,
) -> Result<OperationResult> {
    if config.mode != crate::repl::ReplMode::Streaming {
        return agent
            .chat(&input)
            .await
            .map(OperationResult::Chat)
            .map_err(Into::into);
    }
    let mut stream = agent.chat_stream_events(&input).await?;
    let mut content = String::new();
    let mut observed_tools = Vec::new();
    while let Some(event) = stream.next().await {
        match event {
            AgentStreamEvent::Task(event) => {
                events.send(format!("[task {:?}]\n", event.kind)).await.ok();
            }
            AgentStreamEvent::Chunk(ai_agents::StreamChunk::Content { text }) => {
                content.push_str(&text);
                events.send(text).await.ok();
            }
            AgentStreamEvent::Chunk(ai_agents::StreamChunk::ToolCallStart { name, .. }) => {
                if !observed_tools.contains(&name) {
                    observed_tools.push(name.clone());
                }
                if config.show_tool_calls {
                    events.send(format!("\n[Tool: {name}... ")).await.ok();
                }
            }
            AgentStreamEvent::Chunk(ai_agents::StreamChunk::ToolResult {
                output, success, ..
            }) => {
                if config.show_tool_calls {
                    events
                        .send(if success {
                            "ok]\n".into()
                        } else {
                            format!("failed: {output}]\n")
                        })
                        .await
                        .ok();
                }
            }
            AgentStreamEvent::Chunk(ai_agents::StreamChunk::StateTransition { from, to }) => {
                if config.show_state_transitions {
                    events
                        .send(format!(
                            "\n[State: {} -> {to}]\n",
                            from.as_deref().unwrap_or("none")
                        ))
                        .await
                        .ok();
                }
            }
            AgentStreamEvent::Chunk(ai_agents::StreamChunk::Error { message }) => {
                bail!("{message}")
            }
            AgentStreamEvent::Final(response) => {
                match crate::stream_reconcile::reconcile_final_content(&content, &response.content)
                {
                    crate::stream_reconcile::FinalContent::Unchanged => {}
                    crate::stream_reconcile::FinalContent::Suffix(suffix) => {
                        events.send(suffix.to_string()).await.ok();
                    }
                    crate::stream_reconcile::FinalContent::Authoritative(content) => {
                        events
                            .send(format!("\nAgent (authoritative final): {content}\n"))
                            .await
                            .ok();
                    }
                }
                if config.show_tool_calls {
                    let names = crate::stream_reconcile::unseen_unique_tool_names(
                        response
                            .tool_calls
                            .as_ref()
                            .into_iter()
                            .flatten()
                            .map(|call| call.name.clone()),
                        &observed_tools,
                    );
                    if !names.is_empty() {
                        events
                            .send(format!("\nTools used: {}\n", names.join(", ")))
                            .await
                            .ok();
                    }
                }
                return Ok(OperationResult::Chat(response));
            }
            _ => {}
        }
    }
    bail!("stream ended before Final")
}

#[cfg(test)]
mod tests {
    use super::*;
    struct DoneProvider;
    #[async_trait]
    impl ai_agents::llm::LLMProvider for DoneProvider {
        async fn complete(
            &self,
            _: &[ai_agents::llm::ChatMessage],
            _: Option<&ai_agents::llm::LLMConfig>,
        ) -> std::result::Result<ai_agents::llm::LLMResponse, ai_agents::llm::LLMError> {
            Ok(ai_agents::llm::LLMResponse::new(
                "done",
                ai_agents::llm::FinishReason::Stop,
            ))
        }
        async fn complete_stream(
            &self,
            _: &[ai_agents::llm::ChatMessage],
            _: Option<&ai_agents::llm::LLMConfig>,
        ) -> std::result::Result<
            Box<
                dyn futures::Stream<
                        Item = std::result::Result<
                            ai_agents::llm::LLMChunk,
                            ai_agents::llm::LLMError,
                        >,
                    > + Unpin
                    + Send,
            >,
            ai_agents::llm::LLMError,
        > {
            Ok(Box::new(futures::stream::empty()))
        }
        fn provider_name(&self) -> &str {
            "done-fixture"
        }
        fn supports(&self, _: ai_agents::llm::LLMFeature) -> bool {
            false
        }
    }

    #[tokio::test]
    async fn shutdown_drains_execution_when_the_display_sink_is_full() {
        let provider = DoneProvider;
        let agent=Arc::new(ai_agents::AgentBuilder::from_yaml("name: Backpressure\nsystem_prompt: Helpful\nautonomy:\n  enabled: true\n  completion: {response_contains: done}\n").unwrap().llm(Arc::new(provider)).build_task_host().await.unwrap());
        let runner = agent.task_runner().unwrap();
        let (input_tx, input) = mpsc::channel(1);
        let (callback_tx, callbacks) = mpsc::channel(1);
        let (output, _blocked_receiver) = mpsc::channel(1);
        output
            .send(Output::Text("already full".into()))
            .await
            .unwrap();
        let (_failure_guard, output_failed) = mpsc::channel(1);
        let mut io = FrontendSession {
            input,
            queued: VecDeque::new(),
            callbacks,
            bridge: Arc::new(InputBridge {
                tx: callback_tx,
                show_context: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            }),
            eof: false,
            json: true,
            output,
            output_failed,
            suppressed_output: false,
            interrupt: Box::pin(std::future::pending()),
        };
        let (tx, rx) = mpsc::channel(1);
        let done = {
            let work = io.drive(
                task_work(
                    agent.clone(),
                    TaskStreamOperation::Run {
                        objective: "analysis".into(),
                        profile: None,
                    },
                    tx,
                    true,
                ),
                rx,
            );
            tokio::pin!(work);
            tokio::time::timeout(std::time::Duration::from_secs(10),async {
                loop {
                    tokio::select! {
                        result=&mut work=>panic!("output unexpectedly drained: {}",result.is_ok()),
                        _=tokio::task::yield_now()=>{
                            if runner.task_store().list().await.unwrap().iter().any(|row|row.status==TaskRunStatus::Completed){break;}
                        }
                    }
                }
            }).await.unwrap();
            input_tx.send(Input::Line("/quit".into())).await.unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(10), &mut work)
                .await
                .expect("display pressure must not stop foreground settlement")
                .unwrap()
        };
        assert!(done.quit);
        let OperationResult::Task(result) = done.primary else {
            panic!("expected task result")
        };
        assert_eq!(result.run.status, TaskRunStatus::Completed);
        assert_eq!(result.run.counters.llm_attempts, 1);
        let mut recorded = BTreeMap::new();
        record_result(&mut recorded, (*result).clone());
        let mut late = (*result).clone();
        late.run.status = TaskRunStatus::Paused;
        late.run.revision -= 1;
        record_result(&mut recorded, late);
        assert_eq!(
            recorded[&result.run.key.run_id].run.status,
            TaskRunStatus::Completed
        );
    }
}

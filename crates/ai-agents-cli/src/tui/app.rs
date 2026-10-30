//
// TUI application state, event handling, and rendering.
//

use std::sync::Arc;
use std::time::Instant;

use tracing::Level;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders};
use tokio::sync::mpsc::{Receiver, Sender, channel};
use tui_textarea::TextArea;

use ai_agents::memory::estimate_tokens;
use ai_agents::tools::{QuestionRequest, QuestionResponse};
use ai_agents::{Agent, AgentResponse, AgentStreamEvent, RuntimeAgent, StreamChunk};
use tokio::sync::oneshot;

use super::commands::CommandContext;
use crate::question::response_from_default;
use crate::repl::{CliReplConfig, ReplMode};
use crate::stream_reconcile::unique_tool_names;
use crate::stream_terminal::{
    INCOMPLETE_STREAM_ERROR, StreamDriveControl, StreamDriveOutcome,
    drive_agent_stream_until_closed,
};
use crate::tui::event::{AppMessage, PendingQuestion};
use crate::tui::palette::{THEME_NAMES, resolve_theme, theme_bg_color};
use crate::tui::theme::Theme;
use crate::tui::widgets::{
    agents_panel::{AgentEntry, AgentsPanelState, render_agents_panel},
    chat::{ChatState, DisplayMessage, Role, render_chat},
    completion::{CompletionState, SLASH_COMMANDS, render_completions},
    context_panel::{ContextPanelState, render_context_panel},
    facts_panel::render_facts_panel,
    help_panel::render_help_panel,
    hint_bar::{HintBarState, render_hint_bar},
    memory_panel::{MemoryPanelState, render_memory_panel},
    modal::{ModalState, render_modal},
    persona_panel::{PersonaPanelState, render_persona_panel},
    relationship_panel::{
        RelationshipDimensionEntry, RelationshipEventEntry, RelationshipPanelState,
        render_relationship_panel,
    },
    state_panel::{StatePanelState, render_state_panel},
    status_bar::{StatusBarState, render_status_bar},
    toast::{Toast, render_toast},
    tools_panel::{LastToolCall, ToolsPanelState, render_tools_panel},
};

fn preview_text(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }

    let mut chars = text.chars();
    let preview: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{}...", preview)
    } else {
        text.to_string()
    }
}

#[derive(Debug)]
enum TurnMessage {
    StreamEvent {
        turn_id: u64,
        event: Box<AgentStreamEvent>,
    },
    ChatResponse {
        turn_id: u64,
        response: Box<AgentResponse>,
    },
    ChatError {
        turn_id: u64,
        message: String,
    },
}

fn is_current_turn(active_turn_id: &Option<u64>, turn_id: u64) -> bool {
    *active_turn_id == Some(turn_id)
}

// Clears provisional UI state without committing it as an agent response.
fn record_stream_error(
    chat: &mut ChatState,
    is_thinking: &mut bool,
    current_tools: &mut Vec<String>,
    message: &str,
) {
    *is_thinking = false;
    chat.streaming_content = None;
    current_tools.clear();
    chat.messages.push(DisplayMessage {
        role: Role::System,
        content: format!("[Error] {}", message),
        tools: Vec::new(),
        state_transition: None,
        timing_ms: None,
    });
    chat.auto_scroll = true;
}

// Commits only the authoritative final response and final tool records to chat history.
fn record_final_response(
    chat: &mut ChatState,
    is_thinking: &mut bool,
    current_tools: &mut Vec<String>,
    observed_tool_names: &mut Vec<String>,
    response: AgentResponse,
    elapsed: Option<u64>,
) {
    *is_thinking = false;
    chat.streaming_content = None;
    current_tools.clear();

    let tool_names = unique_tool_names(
        response
            .tool_calls
            .as_ref()
            .into_iter()
            .flatten()
            .map(|call| call.name.clone()),
    );
    for name in &tool_names {
        if !observed_tool_names.contains(name) {
            observed_tool_names.push(name.clone());
        }
    }

    chat.messages.push(DisplayMessage {
        role: Role::Agent,
        content: response.content,
        tools: tool_names,
        state_transition: None,
        timing_ms: elapsed,
    });
    chat.auto_scroll = true;
}

// Forwards one turn and stops polling as soon as the TUI receiver is gone.
fn send_turn_message(
    turn_tx: &Sender<TurnMessage>,
    wake_tx: &Sender<AppMessage>,
    message: TurnMessage,
) -> bool {
    turn_tx.try_send(message).is_ok() && wake_tx.try_send(AppMessage::Tick).is_ok()
}

async fn forward_stream_to_tui<S>(
    stream: S,
    turn_tx: &Sender<TurnMessage>,
    wake_tx: &Sender<AppMessage>,
    turn_id: u64,
) -> StreamDriveOutcome
where
    S: futures::Stream<Item = AgentStreamEvent>,
{
    let outcome = drive_agent_stream_until_closed(
        stream,
        |event| {
            if send_turn_message(
                turn_tx,
                wake_tx,
                TurnMessage::StreamEvent {
                    turn_id,
                    event: Box::new(event),
                },
            ) {
                StreamDriveControl::Continue
            } else {
                StreamDriveControl::ConsumerClosed
            }
        },
        wake_tx.closed(),
    )
    .await;

    if outcome == StreamDriveOutcome::IncompleteEof
        && !send_turn_message(
            turn_tx,
            wake_tx,
            TurnMessage::ChatError {
                turn_id,
                message: INCOMPLETE_STREAM_ERROR.to_string(),
            },
        )
    {
        return StreamDriveOutcome::ConsumerClosed;
    }
    outcome
}

/// Result from update() indicating whether the app should quit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateResult {
    Continue,
    Quit,
}

/// Identifies which panel occupies a side slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelSlot {
    Help,
    States,
    Memory,
    Context,
    Tools,
    Persona,
    Facts,
    Agents,
    Relationship,
}

/// Main TUI application state, owning all widgets, agent handle, and input.
pub struct App {
    agent: Arc<RuntimeAgent>,
    config: CliReplConfig,
    theme: Theme,
    theme_name: String,
    bg_fill: Option<Color>,
    wake_tx: Sender<AppMessage>,
    turn_tx: Sender<TurnMessage>,
    turn_rx: Receiver<TurnMessage>,

    // Input
    input: TextArea<'static>,
    is_command_mode: bool,

    // Slash command completion popup
    completions: CompletionState,

    // Chat
    chat: ChatState,

    // Agent activity
    is_thinking: bool,
    spinner_frame: usize,
    chat_start: Option<Instant>,
    next_turn_id: u64,
    active_turn_id: Option<u64>,

    // Tool tracking from stream events
    current_tools: Vec<String>,
    observed_tool_names: Vec<String>,
    last_tool_call: Option<LastToolCall>,

    // Side panels
    left_panel: Option<PanelSlot>,
    right_panel: Option<PanelSlot>,

    // Modal overlay
    modal: Option<ModalState>,

    // Pending ask_user response
    pending_question: Option<PendingQuestionState>,

    // Toast notifications
    toasts: Vec<Toast>,
    host_input: Option<Sender<crate::session::Input>>,
    host_busy: bool,
    host_generation: u64,
    hidden_modal: Option<ModalState>,
    host_prompt: Option<(u64, String, Option<QuestionRequest>)>,
    host_answering: bool,
    task_views: std::collections::BTreeMap<
        String,
        (
            ai_agents::TaskRunDisplay,
            Option<ai_agents::TaskHostDetails>,
        ),
    >,
    task_panel: bool,
    task_selected: usize,
    task_scroll: u16,
    timeline: std::collections::VecDeque<String>,
    last_result: Option<Box<crate::session::SessionResult>>,
    pub(crate) commands:
        futures::stream::FuturesUnordered<futures::future::BoxFuture<'static, Vec<String>>>,
}

struct PendingQuestionState {
    request: QuestionRequest,
    respond_to: Option<oneshot::Sender<QuestionResponse>>,
}

impl App {
    pub(crate) fn attach_host(&mut self, input: Sender<crate::session::Input>) {
        self.host_input = Some(input);
    }

    fn bound_display(&mut self) {
        while self.chat.messages.len() > 256 {
            self.chat.messages.remove(0);
        }
        for message in &mut self.chat.messages {
            if message.content.len() > 16_384 {
                message.content = message.content.chars().take(4096).collect::<String>()
                    + "\n[display truncated]";
            }
        }
        if let Some(text) = &mut self.chat.streaming_content
            && text.len() > 16_384
        {
            *text = text.chars().take(4096).collect::<String>() + "\n[preview truncated]";
        }
        let mut bytes = self
            .chat
            .messages
            .iter()
            .map(|m| m.content.len())
            .sum::<usize>();
        while bytes > 32_768 && self.chat.messages.len() > 1 {
            bytes -= self.chat.messages.remove(0).content.len();
        }
        self.toasts.truncate(8);
    }

    fn reopen_prompt(&mut self) {
        if self.host_answering {
            self.add_toast("Decision submitted; waiting for its acknowledged outcome.");
            return;
        }
        if let Some(modal) = self.hidden_modal.take() {
            self.modal = Some(modal);
            return;
        }
        if let Some((id, text, question)) = &self.host_prompt {
            let request = question.clone().unwrap_or_else(|| QuestionRequest {
                question: text.clone(),
                allow_other: true,
                ..Default::default()
            });
            let mut modal = ModalState::question(
                &request.question,
                request.options.clone(),
                request.multi_select,
                request.allow_other,
                request.default.as_ref().map(ToString::to_string),
            );
            modal.title = format!("DECISION {id} - Esc hides, Ctrl+P reopens");
            modal.message = text.clone();
            self.modal = Some(modal);
        }
    }

    pub(crate) fn host_output(&mut self, output: crate::session::Output) {
        use crate::session::{OperationResult, Output};
        match output {
            Output::Prompt { id, text, question } => {
                self.hidden_modal = None;
                self.host_prompt = Some((id, text, question));
                self.host_answering = false;
                self.reopen_prompt();
            }
            Output::PromptClosed(id) => {
                if self.host_prompt.as_ref().is_some_and(|p| p.0 == id) {
                    self.host_prompt = None;
                    self.hidden_modal = None;
                    self.host_answering = false;
                    self.modal = None;
                }
            }
            Output::View { display, details } => {
                let first_view = self.task_views.is_empty();
                let id = display.summary.key.run_id.clone();
                let newly_observed = !self.task_views.contains_key(&id);
                if self
                    .task_views
                    .get(&id)
                    .is_none_or(|(old, _)| old.summary.revision <= display.summary.revision)
                {
                    if self.task_views.len() >= 1024
                        && !self.task_views.contains_key(&id)
                        && let Some(old) = self
                            .task_views
                            .iter()
                            .find(|(_, (v, _))| v.summary.status.is_terminal())
                            .map(|(k, _)| k.clone())
                    {
                        self.task_views.remove(&old);
                    }
                    self.task_views.insert(id.clone(), (*display, details));
                    if newly_observed {
                        self.task_selected = self
                            .task_views
                            .keys()
                            .position(|key| key == &id)
                            .unwrap_or(0);
                        self.task_scroll = 0;
                    }
                }
                if first_view {
                    self.task_panel = true;
                }
            }
            Output::Text(text) => {
                self.timeline.push_back(text.chars().take(2048).collect());
                while self.timeline.len() > 128 {
                    self.timeline.pop_front();
                }
                self.chat
                    .streaming_content
                    .get_or_insert_with(String::new)
                    .push_str(&text);
            }
            Output::Diagnostic(text) => {
                self.host_answering = false;
                self.add_system_message(&text);
            }
            Output::Flush(ack) => {
                let _ = ack.send(());
            }
            Output::Rejected {
                generation,
                message,
            } => {
                if generation == self.host_generation {
                    self.host_busy = false;
                    self.is_thinking = false;
                }
                self.add_system_message(&message);
            }
            Output::Result { generation, result } => {
                if generation != self.host_generation {
                    return;
                }
                self.host_busy = false;
                self.hidden_modal = None;
                self.host_prompt = None;
                self.host_answering = false;
                self.modal = None;
                let response = match &result.primary {
                    OperationResult::Chat(response) => response
                        .metadata
                        .as_ref()
                        .and_then(|m| m.get("task_run"))
                        .and_then(|m| m.get("run_id"))
                        .and_then(|id| id.as_str())
                        .and_then(|id| result.results.get(id))
                        .and_then(|r| r.final_response.clone())
                        .unwrap_or_else(|| response.clone()),
                    OperationResult::Task(initial) => result
                        .results
                        .get(&initial.run.key.run_id)
                        .unwrap_or(initial)
                        .final_response
                        .clone()
                        .unwrap_or_else(|| {
                            AgentResponse::new(format!(
                                "Task status: {:?}",
                                result
                                    .results
                                    .get(&initial.run.key.run_id)
                                    .unwrap_or(initial)
                                    .run
                                    .status
                            ))
                        }),
                };
                for task in result.results.values() {
                    self.add_system_message(&format!(
                        "Task {} revision {}: {:?}",
                        task.run.key.run_id, task.run.revision, task.run.status
                    ));
                }
                self.store_final_response(response);
                self.task_panel = false;
                self.last_result = Some(result);
            }
        }
        self.bound_display();
    }

    fn render_tasks(&self, area: Rect, frame: &mut Frame) {
        use ratatui::widgets::{Clear, Paragraph, Wrap};
        Clear.render(area, frame.buffer_mut());
        let mut lines = vec![format!(
            "Task views: {} | Ctrl+Up/Down selects | Ctrl+PgUp/PgDn scrolls",
            self.task_views.len()
        )];
        if let Some((id, (display, details))) = self.task_views.iter().nth(self.task_selected) {
            lines.push(format!(
                "Run {id} revision {}: {:?} | cancel requested: {}",
                display.summary.revision, display.summary.status, display.summary.cancel_requested
            ));
            lines.push(format!(
                "Retained locally: {} | restoration review: {:?}",
                display.locally_retained, display.restoration_review_required
            ));
            lines.push(format!(
                "Active observed: {} ms / {} s | original expiry: {:?}",
                display.clocks.active_millis,
                display.limits.max_active_time_seconds,
                display.clocks.expires_at
            ));
            lines.push(format!(
                "Cumulative: {} turns; {} LLM attempts; {} tool attempts",
                display.counters.turns,
                display.counters.llm_attempts,
                display.counters.tool_attempts
            ));
            lines.push(format!(
                "Last gate: {:?} (historical cycle {:?}, objective revision {:?})",
                display.last_gate, display.last_gate_cycle, display.last_gate_objective_revision
            ));
            if let Some(details) = details
                && details.revision == display.summary.revision
            {
                lines.extend(details.lines.clone());
                if details.truncated {
                    lines.push("[local details truncated; exact result unchanged]".into());
                }
            }
        } else {
            lines.push("No task admitted. Ordinary chat stays ordinary; /task run [--profile ID] OBJECTIVE starts a task.".into());
        }
        lines.push("Timeline (bounded observations; not completion authority):".into());
        lines.extend(self.timeline.iter().cloned());
        Paragraph::new(lines.join("\n"))
            .wrap(Wrap { trim: false })
            .scroll((self.task_scroll, 0))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("TASK - F10 hides; Ctrl+P pending decision"),
            )
            .render(area, frame.buffer_mut());
    }

    fn schedule_command(&mut self, input: &str) {
        if !self.commands.is_empty() {
            self.add_toast("A session command is still pending.");
            return;
        }
        if self.host_busy && !input.starts_with("/sessions") {
            self.add_toast(
                "Task owns this runtime; use its exact task controls before changing the session.",
            );
            return;
        }
        let agent = self.agent.clone();
        let input = input.to_owned();
        self.commands.push(Box::pin(async move {
            let mut context = CommandContext {
                agent,
                messages: vec![],
            };
            context.run(&input).await;
            context.messages
        }));
    }

    pub(crate) fn command_output(&mut self, messages: Vec<String>) {
        let actor = self.agent.actor_id();
        self.task_views
            .retain(|_, (view, _)| view.summary.actor_id == actor);
        self.task_selected = self
            .task_selected
            .min(self.task_views.len().saturating_sub(1));
        for message in messages {
            self.add_system_message(&message);
        }
        self.bound_display();
    }

    /// Build initial application state from an agent, config, and message sender.
    pub fn new(
        agent: Arc<RuntimeAgent>,
        config: CliReplConfig,
        tx: Sender<AppMessage>,
        theme: Theme,
        theme_name: String,
    ) -> Self {
        let mut input = TextArea::default();
        input.set_cursor_line_style(Style::default());
        input.set_placeholder_text("Type a message...");

        let mut chat = ChatState::new();
        chat.show_tool_calls = config.show_tool_calls;
        chat.show_timing = config.show_timing;

        if let Some(ref welcome) = config.welcome {
            chat.messages.push(DisplayMessage {
                role: Role::System,
                content: welcome.clone(),
                tools: Vec::new(),
                state_transition: None,
                timing_ms: None,
            });
        }

        if !config.hints.is_empty() {
            let grouped = config.hints.join("\n");
            chat.messages.push(DisplayMessage {
                role: Role::Hint,
                content: grouped,
                tools: Vec::new(),
                state_transition: None,
                timing_ms: None,
            });
        }

        let bg_fill = theme_bg_color(&theme_name);
        let (turn_tx, turn_rx) = channel(32);
        Self {
            agent,
            config,
            theme,
            theme_name,
            bg_fill,
            wake_tx: tx,
            turn_tx,
            turn_rx,
            input,
            is_command_mode: false,
            completions: CompletionState::new(),
            chat,
            is_thinking: false,
            spinner_frame: 0,
            chat_start: None,
            next_turn_id: 1,
            active_turn_id: None,
            current_tools: Vec::new(),
            observed_tool_names: Vec::new(),
            last_tool_call: None,
            left_panel: None,
            right_panel: None,
            modal: None,
            pending_question: None,
            toasts: Vec::new(),
            host_input: None,
            host_busy: false,
            host_generation: 0,
            hidden_modal: None,
            host_prompt: None,
            host_answering: false,
            task_views: Default::default(),
            task_panel: false,
            task_selected: 0,
            task_scroll: 0,
            timeline: Default::default(),
            last_result: None,
            commands: Default::default(),
        }
    }

    /// Process one incoming event and return whether the app should keep running.
    pub async fn update(&mut self, msg: AppMessage) -> UpdateResult {
        self.drain_turn_messages();
        self.bound_display();
        if let AppMessage::Key(key) = &msg {
            if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                return UpdateResult::Quit;
            }
            if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('p') {
                self.reopen_prompt();
                return UpdateResult::Continue;
            }
            if self.host_prompt.is_some() && key.code == KeyCode::Esc {
                self.hidden_modal = self.modal.take();
                return UpdateResult::Continue;
            }
        }
        if self.modal.is_some()
            && let AppMessage::Key(key) = msg
        {
            return self.handle_modal_key(key);
        }

        match msg {
            AppMessage::Key(key) => self.handle_key(key).await,
            AppMessage::Resize(_, _) => UpdateResult::Continue,
            AppMessage::StreamEvent(event) => {
                let terminal = matches!(
                    &*event,
                    AgentStreamEvent::Final(_) | AgentStreamEvent::Chunk(StreamChunk::Error { .. })
                );
                self.handle_stream_event(*event);
                if terminal {
                    self.active_turn_id = None;
                }
                UpdateResult::Continue
            }
            AppMessage::ChatResponse(response) => {
                self.store_final_response(*response);
                self.active_turn_id = None;
                UpdateResult::Continue
            }
            AppMessage::ChatError(message) => {
                record_stream_error(
                    &mut self.chat,
                    &mut self.is_thinking,
                    &mut self.current_tools,
                    &message,
                );
                self.active_turn_id = None;
                UpdateResult::Continue
            }
            AppMessage::Question(question) => {
                self.show_question_modal(question);
                UpdateResult::Continue
            }
            AppMessage::Tick => {
                self.spinner_frame = self.spinner_frame.wrapping_add(1);
                for toast in &mut self.toasts {
                    toast.tick();
                }
                self.toasts.retain(|t| !t.is_expired());
                UpdateResult::Continue
            }
            AppMessage::Log(entry) => {
                let level_tag = match entry.level {
                    Level::ERROR => "[ERROR",
                    Level::WARN => "[WARN ",
                    Level::INFO => "[INFO ",
                    Level::DEBUG => "[DEBUG",
                    Level::TRACE => "[TRACE",
                };
                let short_target = entry.target.rsplit("::").next().unwrap_or(&entry.target);
                let content = format!("{} {}] {}", level_tag, short_target, entry.message);

                self.chat.messages.push(DisplayMessage {
                    role: Role::Log,
                    content,
                    tools: Vec::new(),
                    state_transition: None,
                    timing_ms: None,
                });
                self.chat.auto_scroll = true;
                UpdateResult::Continue
            }
        }
    }

    // Applies queued messages only when their internally spawned turn still owns the UI.
    fn drain_turn_messages(&mut self) {
        while let Ok(message) = self.turn_rx.try_recv() {
            match message {
                TurnMessage::StreamEvent { turn_id, event } => {
                    if !is_current_turn(&self.active_turn_id, turn_id) {
                        continue;
                    }
                    let terminal = matches!(
                        &*event,
                        AgentStreamEvent::Final(_)
                            | AgentStreamEvent::Chunk(StreamChunk::Error { .. })
                    );
                    self.handle_stream_event(*event);
                    if terminal {
                        self.active_turn_id = None;
                    }
                }
                TurnMessage::ChatResponse { turn_id, response } => {
                    if is_current_turn(&self.active_turn_id, turn_id) {
                        self.store_final_response(*response);
                        self.active_turn_id = None;
                    }
                }
                TurnMessage::ChatError { turn_id, message } => {
                    if is_current_turn(&self.active_turn_id, turn_id) {
                        record_stream_error(
                            &mut self.chat,
                            &mut self.is_thinking,
                            &mut self.current_tools,
                            &message,
                        );
                        self.active_turn_id = None;
                    }
                }
            }
        }
    }

    async fn handle_key(&mut self, key: KeyEvent) -> UpdateResult {
        // Ctrl+C always quits.
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return UpdateResult::Quit;
        }

        // Ctrl+L clears the chat display.
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('l') {
            self.chat.messages.clear();
            self.chat.scroll_offset = 0;
            return UpdateResult::Continue;
        }

        // Ctrl+S quick-saves the session.
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('s') {
            self.schedule_command("/save default");
            return UpdateResult::Continue;
        }

        // Ctrl+T cycles the color theme.
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('t') {
            self.cycle_theme();
            return UpdateResult::Continue;
        }

        if key.code == KeyCode::F(10) {
            self.task_panel = !self.task_panel;
            return UpdateResult::Continue;
        }
        if self.task_panel && key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Up => self.task_selected = self.task_selected.saturating_sub(1),
                KeyCode::Down => {
                    self.task_selected =
                        (self.task_selected + 1).min(self.task_views.len().saturating_sub(1))
                }
                KeyCode::PageUp => self.task_scroll = self.task_scroll.saturating_sub(10),
                KeyCode::PageDown => self.task_scroll = self.task_scroll.saturating_add(10),
                _ => {}
            }
        }
        // Scroll keys
        match key.code {
            KeyCode::PageUp => {
                self.chat.scroll_offset = self.chat.scroll_offset.saturating_sub(10);
                self.chat.auto_scroll = false;
                return UpdateResult::Continue;
            }
            KeyCode::PageDown => {
                self.chat.scroll_offset = self.chat.scroll_offset.saturating_add(10);
                self.chat.auto_scroll = true;
                return UpdateResult::Continue;
            }
            _ => {}
        }

        // F-key panel toggles
        match key.code {
            KeyCode::F(1) => {
                self.toggle_panel(PanelSlot::Help);
                return UpdateResult::Continue;
            }
            KeyCode::F(2) => {
                self.toggle_panel(PanelSlot::States);
                return UpdateResult::Continue;
            }
            KeyCode::F(3) => {
                self.toggle_panel(PanelSlot::Memory);
                return UpdateResult::Continue;
            }
            KeyCode::F(4) => {
                self.toggle_panel(PanelSlot::Context);
                return UpdateResult::Continue;
            }
            KeyCode::F(5) => {
                self.toggle_panel(PanelSlot::Tools);
                return UpdateResult::Continue;
            }
            KeyCode::F(6) => {
                self.toggle_panel(PanelSlot::Persona);
                return UpdateResult::Continue;
            }
            KeyCode::F(7) => {
                self.toggle_panel(PanelSlot::Facts);
                return UpdateResult::Continue;
            }
            KeyCode::F(8) => {
                self.toggle_panel(PanelSlot::Agents);
                return UpdateResult::Continue;
            }
            KeyCode::F(9) => {
                self.toggle_panel(PanelSlot::Relationship);
                return UpdateResult::Continue;
            }
            _ => {}
        }

        // Completion popup intercept (before textarea and before existing Enter/Tab).
        if self.completions.visible {
            match key.code {
                KeyCode::Up => {
                    self.completions.move_up();
                    return UpdateResult::Continue;
                }
                KeyCode::Down => {
                    self.completions.move_down();
                    return UpdateResult::Continue;
                }
                KeyCode::Tab => {
                    if let Some(cmd) = self.completions.selected_command() {
                        let cmd = cmd.to_string();
                        self.input.select_all();
                        self.input.cut();
                        self.input.insert_str(&cmd);
                    }
                    self.completions.close();
                    return UpdateResult::Continue;
                }
                KeyCode::Enter => {
                    if let Some(cmd) = self.completions.selected_command() {
                        let cmd = cmd.to_string();
                        self.input.select_all();
                        self.input.cut();
                        self.input.insert_str(&cmd);
                    }
                    self.completions.close();
                    // Fall through to the existing Enter handler.
                }
                KeyCode::Esc => {
                    self.completions.close();
                    return UpdateResult::Continue;
                }
                _ => {
                    // Let the key go through to the textarea below.
                    // update_completions() will run after input.input(key).
                }
            }
        }

        // Esc abandons only this TUI's active display, closes panels, or closes completion; the runtime/provider request is not cancelled here.
        if key.code == KeyCode::Esc {
            if self.completions.visible {
                self.completions.close();
                return UpdateResult::Continue;
            }
            if self.is_thinking {
                self.is_thinking = false;
                self.chat.streaming_content = None;
                self.current_tools.clear();
                self.active_turn_id = None;
                return UpdateResult::Continue;
            }
            if self.left_panel.is_some() || self.right_panel.is_some() {
                self.left_panel = None;
                self.right_panel = None;
                return UpdateResult::Continue;
            }
            return UpdateResult::Continue;
        }

        // Tab accepts the selected completion (popup already handled above when visible).
        if key.code == KeyCode::Tab {
            return UpdateResult::Continue;
        }

        // Enter sends the current input.  Intercept before textarea so
        // Enter does not insert a newline into the buffer.
        if key.code == KeyCode::Enter && (!self.is_thinking || self.host_input.is_some()) {
            self.completions.close();
            let lines: Vec<String> = self.input.lines().iter().map(|s| s.to_string()).collect();
            let text = lines.join("\n").trim().to_string();
            if text.is_empty() {
                return UpdateResult::Continue;
            }
            if text.len() > 65_536 {
                self.add_toast("Input exceeds the 64 KiB transport limit; nothing was submitted.");
                return UpdateResult::Continue;
            }
            self.input.select_all();
            self.input.cut();

            if let Some(tx) = self.host_input.clone()
                && (text.starts_with("/task ") || !text.starts_with('/'))
            {
                if !self.commands.is_empty() && !self.host_busy {
                    self.add_toast("Wait for the pending session command before starting work.");
                    return UpdateResult::Continue;
                }
                if !self.host_busy && text == "/task list" {
                    let rows = self
                        .task_views
                        .values()
                        .map(|(v, _)| {
                            format!(
                                "{} revision {}: {:?}",
                                v.summary.key.run_id, v.summary.revision, v.summary.status
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    self.add_system_message(&rows);
                    return UpdateResult::Continue;
                }
                if !self.host_busy
                    && let Some(id) = text.strip_prefix("/task show ")
                {
                    let value = self
                        .task_views
                        .get(id)
                        .map(|(v, _)| serde_json::to_string(v).unwrap())
                        .unwrap_or_else(|| "No observed scoped run".into());
                    self.add_system_message(&value);
                    return UpdateResult::Continue;
                }
                if self.host_busy && (!text.starts_with("/task ") || text.starts_with("/task run"))
                {
                    self.add_toast("Work is retained; answer its prompt or use task controls.");
                    return UpdateResult::Continue;
                }
                let generation = if self.host_busy {
                    self.host_generation
                } else {
                    self.host_generation.wrapping_add(1)
                };
                if let Err(e) = super::host::send(
                    &tx,
                    crate::session::Input::UiLine {
                        generation,
                        line: text.clone(),
                    },
                ) {
                    self.add_toast(&e.to_string());
                    return UpdateResult::Continue;
                }
                if !self.host_busy {
                    self.host_generation = generation;
                    self.host_busy = true;
                    self.chat_start = Some(Instant::now());
                    self.is_thinking = true;
                    self.observed_tool_names.clear();
                    self.add_system_message(&format!("You: {text}"));
                }
                return UpdateResult::Continue;
            }
            if text.starts_with('/') {
                return self.handle_slash_command(&text).await;
            }

            self.chat.messages.push(DisplayMessage {
                role: Role::User,
                content: text.clone(),
                tools: Vec::new(),
                state_transition: None,
                timing_ms: None,
            });
            self.chat.auto_scroll = true;
            self.is_thinking = true;
            self.chat_start = Some(Instant::now());
            self.current_tools.clear();
            let turn_id = self.next_turn_id;
            self.next_turn_id = self.next_turn_id.wrapping_add(1);
            self.active_turn_id = Some(turn_id);

            let agent = Arc::clone(&self.agent);
            let turn_tx = self.turn_tx.clone();
            let wake_tx = self.wake_tx.clone();
            let streaming = self.config.mode == ReplMode::Streaming;

            tokio::spawn(async move {
                if streaming {
                    match agent.chat_stream_events(&text).await {
                        Ok(stream) => {
                            let _ =
                                forward_stream_to_tui(stream, &turn_tx, &wake_tx, turn_id).await;
                        }
                        Err(e) => {
                            let _ = send_turn_message(
                                &turn_tx,
                                &wake_tx,
                                TurnMessage::ChatError {
                                    turn_id,
                                    message: e.to_string(),
                                },
                            );
                        }
                    }
                } else {
                    match agent.chat(&text).await {
                        Ok(response) => {
                            let _ = send_turn_message(
                                &turn_tx,
                                &wake_tx,
                                TurnMessage::ChatResponse {
                                    turn_id,
                                    response: Box::new(response),
                                },
                            );
                        }
                        Err(e) => {
                            let _ = send_turn_message(
                                &turn_tx,
                                &wake_tx,
                                TurnMessage::ChatError {
                                    turn_id,
                                    message: e.to_string(),
                                },
                            );
                        }
                    }
                }
            });

            return UpdateResult::Continue;
        }

        // Input growth is bounded independently of the exact task result and recovery checkpoint.
        if matches!(key.code, KeyCode::Char(_))
            && self.input.lines().iter().map(String::len).sum::<usize>() >= 65_536
        {
            return UpdateResult::Continue;
        }
        // Forward everything else to the text area, then refresh completions.
        self.input.input(key);
        self.update_completions();
        UpdateResult::Continue
    }

    /// Update the completion popup based on current input text.
    fn update_completions(&mut self) {
        let text = self
            .input
            .lines()
            .first()
            .map(|s| s.to_string())
            .unwrap_or_default();

        // Only show popup when "/" is first char and no space yet (not typing args).
        if !text.starts_with('/') || text.contains(' ') {
            self.completions.close();
            self.is_command_mode = text.starts_with('/');
            return;
        }

        self.is_command_mode = true;
        let prefix = text.to_lowercase();

        self.completions.items = SLASH_COMMANDS
            .iter()
            .filter(|cmd| cmd.name.starts_with(&prefix))
            .collect();

        if self.completions.items.is_empty() {
            self.completions.visible = false;
        } else {
            self.completions.visible = true;
            self.completions.selected = self
                .completions
                .selected
                .min(self.completions.items.len().saturating_sub(1));
        }
    }

    /// Cycle to the next color theme.
    fn cycle_theme(&mut self) {
        let current_idx = THEME_NAMES
            .iter()
            .position(|n| *n == self.theme_name)
            .unwrap_or(0);
        let next_idx = (current_idx + 1) % THEME_NAMES.len();
        self.theme_name = THEME_NAMES[next_idx].to_string();
        if let Some(theme) = resolve_theme(&self.theme_name) {
            self.theme = theme;
        }
        self.bg_fill = theme_bg_color(&self.theme_name);
        self.add_toast(&format!("Theme: {}", self.theme_name));
    }

    fn handle_stream_event(&mut self, event: AgentStreamEvent) {
        match event {
            AgentStreamEvent::Chunk(chunk) => self.handle_stream_chunk(chunk),
            AgentStreamEvent::Final(response) => self.store_final_response(response),
            _ => {}
        }
    }

    fn store_final_response(&mut self, response: AgentResponse) {
        let elapsed = self.chat_start.map(|s| s.elapsed().as_millis() as u64);
        record_final_response(
            &mut self.chat,
            &mut self.is_thinking,
            &mut self.current_tools,
            &mut self.observed_tool_names,
            response,
            elapsed,
        );
    }

    fn handle_stream_chunk(&mut self, chunk: StreamChunk) {
        match chunk {
            StreamChunk::Content { text } => {
                let current = self.chat.streaming_content.get_or_insert_with(String::new);
                current.push_str(&text);
                self.chat.auto_scroll = true;
            }
            StreamChunk::ToolCallStart { name, .. } => {
                if !self.current_tools.contains(&name) {
                    self.current_tools.push(name.clone());
                }
                if !self.observed_tool_names.contains(&name) {
                    self.observed_tool_names.push(name);
                }
            }
            StreamChunk::ToolCallDelta { .. } => {}
            StreamChunk::ToolCallEnd { .. } => {}
            StreamChunk::ToolResult {
                name,
                output,
                success,
                ..
            } => {
                let elapsed = self
                    .chat_start
                    .map(|s| s.elapsed().as_millis() as u64)
                    .unwrap_or(0);
                let preview = preview_text(&output, 30);
                self.last_tool_call = Some(LastToolCall {
                    name: name.clone(),
                    input_preview: String::new(),
                    output_preview: preview,
                    duration_ms: elapsed,
                });
                let _ = success;
            }
            StreamChunk::StateTransition { from, to } => {
                let from_str = from.unwrap_or_else(|| "-".to_string());
                self.chat.messages.push(DisplayMessage {
                    role: Role::System,
                    content: format!("State: {} -> {}", from_str, to),
                    tools: Vec::new(),
                    state_transition: Some((from_str, to)),
                    timing_ms: None,
                });
            }
            StreamChunk::Done {} => {
                // AgentStreamEvent::Final owns committed message and tool state.
            }
            StreamChunk::Error { message } => {
                record_stream_error(
                    &mut self.chat,
                    &mut self.is_thinking,
                    &mut self.current_tools,
                    &message,
                );
            }
        }
    }

    async fn handle_slash_command(&mut self, input: &str) -> UpdateResult {
        let lower = input.to_lowercase();
        let trimmed = lower.as_str();
        if self.host_busy
            && (trimmed.starts_with("/context set ") || trimmed.starts_with("/context unset "))
        {
            self.add_toast(
                "Work is retained; context changes must not replace its pending decision.",
            );
            return UpdateResult::Continue;
        }
        if [
            "/reset",
            "/save",
            "/load",
            "/delete",
            "/sessions",
            "/cleanup",
            "/actor",
            "/facts",
            "/relationship",
            "/rel",
        ]
        .iter()
        .any(|cmd| trimmed == *cmd || trimmed.starts_with(&format!("{cmd} ")))
        {
            if matches!(trimmed, "/relationship" | "/rel") {
                self.toggle_panel(PanelSlot::Relationship);
            }
            self.schedule_command(input);
            return UpdateResult::Continue;
        }

        match trimmed {
            "/quit" | "/exit" => return UpdateResult::Quit,
            "/help" | "/?" => {
                self.toggle_panel(PanelSlot::Help);
            }
            "/state" => match self.agent.current_state() {
                Some(state) => self.add_system_message(&format!("Current state: {}", state)),
                None => self.add_system_message("No state machine active."),
            },
            "/history" => {
                let history = self.agent.state_history();
                if history.is_empty() {
                    self.add_system_message("No state transitions yet.");
                } else {
                    let mut msg = "State transitions:".to_string();
                    for event in &history {
                        msg.push_str(&format!(
                            "\n  {} -> {} ({})",
                            event.from, event.to, event.reason
                        ));
                    }
                    self.add_system_message(&msg);
                }
            }
            "/info" => {
                let info = self.agent.info();
                let mut msg = format!("Agent: {} v{}", info.name, info.version);
                if let Some(ref desc) = info.description {
                    msg.push_str(&format!("\nDescription: {}", desc));
                }
                msg.push_str(&format!("\nSkills: {}", self.agent.skills().len()));
                if let Some(state) = self.agent.current_state() {
                    msg.push_str(&format!("\nState: {}", state));
                }
                self.add_system_message(&msg);
            }
            "/memory" | "/mem" => {
                self.toggle_panel(PanelSlot::Memory);
            }
            _ if trimmed.starts_with("/context") => {
                self.handle_context_command(input);
            }
            _ => {
                self.add_system_message(&format!("Unknown command: {}", input));
            }
        }
        UpdateResult::Continue
    }

    fn handle_context_command(&mut self, input: &str) {
        let parts: Vec<&str> = input.split_whitespace().collect();
        match parts.get(1).map(|s| s.to_lowercase()).as_deref() {
            None => {
                self.toggle_panel(PanelSlot::Context);
            }
            Some("set") => {
                let key = match parts.get(2) {
                    Some(k) => *k,
                    None => {
                        self.add_system_message("Usage: /context set <key> <value>");
                        return;
                    }
                };
                if parts.len() < 4 {
                    self.add_system_message("Usage: /context set <key> <value>");
                    return;
                }
                let raw_value = parts[3..].join(" ");
                let value: serde_json::Value = serde_json::from_str(&raw_value)
                    .unwrap_or(serde_json::Value::String(raw_value));
                match self.agent.set_context(key, value) {
                    Ok(()) => self.add_toast(&format!("Set: {}", key)),
                    Err(e) => self.add_system_message(&format!("[Error] {}", e)),
                }
            }
            Some("unset") => {
                let key = match parts.get(2) {
                    Some(k) => *k,
                    None => {
                        self.add_system_message("Usage: /context unset <key>");
                        return;
                    }
                };
                match self.agent.remove_context(key) {
                    Ok(Some(_)) => self.add_toast(&format!("Removed: {}", key)),
                    Ok(None) => self.add_system_message(&format!("Key not found: {}", key)),
                    Err(error) => {
                        self.add_system_message(&format!("Context removal failed: {}", error))
                    }
                }
            }
            Some(other) => {
                self.add_system_message(&format!("Unknown: /context {}. Use: set, unset", other));
            }
        }
    }

    fn handle_modal_key(&mut self, key: KeyEvent) -> UpdateResult {
        if let Some(modal) = self.modal.as_mut() {
            if key.code == KeyCode::PageUp {
                modal.review_scroll = modal.review_scroll.saturating_sub(5);
                return UpdateResult::Continue;
            }
            if key.code == KeyCode::PageDown {
                modal.review_scroll = modal.review_scroll.saturating_add(5);
                return UpdateResult::Continue;
            }
        }
        // Check if this is a question modal and extract the state.
        let is_question = self.modal.as_ref().is_some_and(|m| m.question.is_some());

        if is_question {
            // Move the modal out, process the question key, then restore.
            let mut modal = self.modal.take().unwrap();
            if let Some(ref mut q) = modal.question {
                self.handle_question_modal_key(key, q);
            }
            // Completion consumes its request; navigation must restore the temporarily moved modal.
            if (self.pending_question.is_none() && self.host_prompt.is_none())
                || self.host_answering
                || (key.code == KeyCode::Enter
                    && modal.question.as_ref().is_some_and(|q| {
                        q.actions
                            .get(q.selected_action)
                            .is_some_and(|a| a == "Cancel")
                            && q.focus == crate::tui::widgets::modal::QuestionFocus::Actions
                    }))
            {
                return UpdateResult::Continue;
            }
            self.modal = Some(modal);
            return UpdateResult::Continue;
        }

        if let Some(ref mut modal) = self.modal {
            // Standard approval/confirm modals use left/right navigation.
            match key.code {
                KeyCode::Left | KeyCode::Tab => {
                    if modal.selected_button > 0 {
                        modal.selected_button -= 1;
                    }
                }
                KeyCode::Right | KeyCode::BackTab => {
                    if modal.selected_button + 1 < modal.buttons.len() {
                        modal.selected_button += 1;
                    }
                }
                KeyCode::Enter => {
                    let selected = modal.buttons.get(modal.selected_button).cloned();
                    self.complete_pending_question(selected, false);
                    self.modal = None;
                }
                KeyCode::Esc => {
                    self.complete_pending_question(None, false);
                    self.modal = None;
                }
                _ => {}
            }
        }
        UpdateResult::Continue
    }

    fn handle_question_modal_key(
        &mut self,
        key: KeyEvent,
        q: &mut crate::tui::widgets::modal::QuestionModalState,
    ) -> UpdateResult {
        use crate::tui::widgets::modal::QuestionFocus;

        match q.focus {
            QuestionFocus::Options => match key.code {
                KeyCode::Up => {
                    if q.focused_option > 0 {
                        q.focused_option -= 1;
                    }
                }
                KeyCode::Down => {
                    if q.focused_option + 1 < q.options.len() {
                        q.focused_option += 1;
                    } else if !q.actions.is_empty() {
                        q.focus = QuestionFocus::Actions;
                        q.selected_action = 0;
                    } else if q.allow_other {
                        q.focus = QuestionFocus::TextInput;
                    }
                }
                KeyCode::Tab => {
                    if q.allow_other {
                        q.focus = QuestionFocus::TextInput;
                    } else if !q.actions.is_empty() {
                        q.focus = QuestionFocus::Actions;
                        q.selected_action = 0;
                    }
                }
                KeyCode::Char(' ') => {
                    if q.multi_select {
                        q.toggle_current();
                    } else {
                        q.select_current_single();
                    }
                }
                KeyCode::Enter => {
                    if q.multi_select {
                        q.toggle_current();
                    } else {
                        q.select_current_single();
                        let selected = q.selected_labels();
                        self.complete_question_response(selected, None, false);
                    }
                }
                KeyCode::Esc => {
                    self.complete_question_response(Vec::new(), None, false);
                }
                _ => {}
            },
            QuestionFocus::TextInput => match key.code {
                KeyCode::Up => {
                    if !q.options.is_empty() {
                        q.focus = QuestionFocus::Options;
                    }
                }
                KeyCode::Down | KeyCode::Tab => {
                    if !q.actions.is_empty() {
                        q.focus = QuestionFocus::Actions;
                        q.selected_action = 0;
                    }
                }
                KeyCode::Enter => {
                    if !q.text_input.trim().is_empty() {
                        self.complete_question_response(
                            Vec::new(),
                            Some(q.text_input.trim().to_string()),
                            false,
                        );
                    }
                }
                KeyCode::Esc => {
                    self.complete_question_response(Vec::new(), None, false);
                }
                KeyCode::Backspace => {
                    q.text_input.pop();
                }
                KeyCode::Char(ch) if q.text_input.len() + ch.len_utf8() <= 65_536 => {
                    q.text_input.push(ch);
                }
                _ => {}
            },
            QuestionFocus::Actions => match key.code {
                KeyCode::Left => {
                    if q.selected_action > 0 {
                        q.selected_action -= 1;
                    }
                }
                KeyCode::Right | KeyCode::Tab => {
                    if q.selected_action + 1 < q.actions.len() {
                        q.selected_action += 1;
                    }
                }
                KeyCode::Up => {
                    if q.allow_other {
                        q.focus = QuestionFocus::TextInput;
                    } else if !q.options.is_empty() {
                        q.focus = QuestionFocus::Options;
                    }
                }
                KeyCode::Enter => {
                    let action = q.actions.get(q.selected_action).cloned();
                    match action.as_deref() {
                        Some("Submit") => {
                            let selected = q.selected_labels();
                            if self.host_prompt.is_some() && selected.is_empty() {
                                self.add_toast("Select an answer explicitly.");
                                return UpdateResult::Continue;
                            }
                            self.complete_question_response(selected, None, false);
                        }
                        Some("Submit text") => {
                            if self.host_prompt.is_some() && q.text_input.trim().is_empty() {
                                self.add_toast("Enter an answer or explicitly select Use default.");
                                return UpdateResult::Continue;
                            }
                            let text = if q.text_input.trim().is_empty() {
                                None
                            } else {
                                Some(q.text_input.trim().to_string())
                            };
                            self.complete_question_response(Vec::new(), text, false);
                        }
                        Some("Cancel") if self.host_prompt.is_some() => {
                            self.modal = None;
                            return UpdateResult::Continue;
                        }
                        Some("Use default") => {
                            self.complete_question_response(Vec::new(), None, false);
                        }
                        _ => {
                            self.complete_question_response(Vec::new(), None, false);
                        }
                    }
                }
                KeyCode::Esc => {
                    self.complete_question_response(Vec::new(), None, false);
                }
                _ => {}
            },
        }
        UpdateResult::Continue
    }

    fn complete_question_response(
        &mut self,
        selected: Vec<String>,
        other_text: Option<String>,
        timed_out: bool,
    ) {
        if let Some((id, _, request)) = &self.host_prompt {
            if self.host_answering {
                return;
            }
            let line = if let Some(request) = request {
                let response = if selected.is_empty() && other_text.is_none() {
                    response_from_default(request.default.clone(), false)
                } else {
                    QuestionResponse {
                        answered: true,
                        selected,
                        other_text,
                        timed_out: false,
                        unavailable: false,
                    }
                };
                serde_json::to_string(&response).expect("question response")
            } else {
                other_text.unwrap_or_else(|| selected.join(","))
            };
            if let Some(tx) = &self.host_input {
                match super::host::send(tx, crate::session::Input::Reply { id: *id, line }) {
                    Ok(()) => {
                        self.host_answering = true;
                        self.modal = None;
                    }
                    Err(e) => self.add_toast(&e.to_string()),
                }
            }
            return;
        }
        let Some(mut pending) = self.pending_question.take() else {
            self.modal = None;
            return;
        };
        let has_selection = !selected.is_empty();
        let has_text = other_text.is_some();
        let response = if !has_selection && !has_text {
            response_from_default(pending.request.default, timed_out)
        } else {
            QuestionResponse {
                answered: true,
                selected,
                other_text,
                timed_out,
                unavailable: false,
            }
        };
        if let Some(sender) = pending.respond_to.take() {
            let _ = sender.send(response);
        }
        self.modal = None;
    }

    fn toggle_panel(&mut self, panel: PanelSlot) {
        match panel {
            PanelSlot::Help | PanelSlot::States => {
                if self.left_panel == Some(panel) {
                    self.left_panel = None;
                } else {
                    self.left_panel = Some(panel);
                }
            }
            _ => {
                if self.right_panel == Some(panel) {
                    self.right_panel = None;
                } else {
                    self.right_panel = Some(panel);
                }
            }
        }
    }

    fn add_system_message(&mut self, content: &str) {
        self.chat.messages.push(DisplayMessage {
            role: Role::System,
            content: content.to_string(),
            tools: Vec::new(),
            state_transition: None,
            timing_ms: None,
        });
        self.chat.auto_scroll = true;
    }

    fn add_toast(&mut self, message: &str) {
        self.toasts.push(Toast::new(message, 30));
    }

    /// Show a modal dialog for confirmations or approvals.
    pub fn show_modal(&mut self, modal: ModalState) {
        self.modal = Some(modal);
    }

    fn show_question_modal(&mut self, question: PendingQuestion) {
        let default_label = question.request.default.as_ref().map(|v| {
            if let Some(s) = v.as_str() {
                s.to_string()
            } else {
                v.to_string()
            }
        });
        let modal = ModalState::question(
            &question.request.question,
            question.request.options.clone(),
            question.request.multi_select,
            question.request.allow_other,
            default_label,
        );
        self.pending_question = Some(PendingQuestionState {
            request: question.request,
            respond_to: Some(question.respond_to),
        });
        self.modal = Some(modal);
    }

    fn complete_pending_question(&mut self, selected: Option<String>, timed_out: bool) {
        let Some(mut pending) = self.pending_question.take() else {
            return;
        };
        let response = match selected {
            Some(label) if label == "Use default" || label == "Cancel" => {
                response_from_default(pending.request.default, timed_out)
            }
            Some(label) => QuestionResponse {
                answered: true,
                selected: vec![label],
                other_text: None,
                timed_out,
                unavailable: false,
            },
            None => response_from_default(pending.request.default, timed_out),
        };
        if let Some(sender) = pending.respond_to.take() {
            let _ = sender.send(response);
        }
    }

    /// Compose the full TUI layout into the terminal frame.
    pub fn render(&mut self, frame: &mut Frame) {
        let size = frame.area();

        // For RGB themes, fill the entire alternate screen with the theme background
        // before any widgets are drawn. ANSI themes (dark, light) leave bg_fill as
        // None and defer to the terminal's native background color.
        if let Some(bg) = self.bg_fill {
            frame.buffer_mut().set_style(size, Style::default().bg(bg));
        }

        // Vertical: title bar | content | input | hint bar
        let input_height = 3u16;
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(5),
                Constraint::Length(input_height),
                Constraint::Length(1),
            ])
            .split(size);

        // Title bar
        let status = self.build_status_state();
        render_status_bar(main_chunks[0], frame.buffer_mut(), &status, &self.theme);

        // Content area: optional left panel, chat, optional right panel.
        let content_area = main_chunks[1];
        let (left_area, chat_area, right_area) = self.split_content(content_area);

        if let (Some(area), Some(panel)) = (left_area, self.left_panel) {
            self.render_panel(area, frame, panel);
        }

        let chat_height = chat_area.height;
        let total_lines = self.chat.total_lines(chat_area.width);
        self.chat.scroll_to_bottom(chat_height, total_lines);
        render_chat(chat_area, frame.buffer_mut(), &self.chat, &self.theme);

        if let (Some(area), Some(panel)) = (right_area, self.right_panel) {
            self.render_panel(area, frame, panel);
        }

        // Input area with a top border.
        let input_block = Block::default()
            .borders(Borders::TOP)
            .border_style(self.theme.border_style);
        let input_inner = input_block.inner(main_chunks[2]);
        input_block.render(main_chunks[2], frame.buffer_mut());
        frame.render_widget(&self.input, input_inner);

        // Completion popup (overlay above input).
        if self.completions.visible {
            render_completions(
                main_chunks[2],
                size,
                frame.buffer_mut(),
                &self.completions,
                &self.theme,
            );
        }

        // Hint bar
        let hint_state = HintBarState {
            is_command_mode: self.is_command_mode,
            panels_enabled: true,
        };
        render_hint_bar(main_chunks[3], frame.buffer_mut(), &hint_state, &self.theme);

        if self.task_panel {
            self.render_tasks(content_area, frame);
        }
        // Modal overlay
        if let Some(ref modal) = self.modal {
            render_modal(size, frame.buffer_mut(), modal, &self.theme);
        }

        // Toast overlay (show the first active toast)
        if let Some(toast) = self.toasts.first() {
            render_toast(size, frame.buffer_mut(), toast, &self.theme);
        }
    }

    fn split_content(&self, area: Rect) -> (Option<Rect>, Rect, Option<Rect>) {
        let panel_width = 22u16;

        match (self.left_panel, self.right_panel) {
            (Some(_), Some(_)) => {
                let chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Length(panel_width),
                        Constraint::Min(20),
                        Constraint::Length(panel_width),
                    ])
                    .split(area);
                (Some(chunks[0]), chunks[1], Some(chunks[2]))
            }
            (Some(_), None) => {
                let chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Length(panel_width), Constraint::Min(20)])
                    .split(area);
                (Some(chunks[0]), chunks[1], None)
            }
            (None, Some(_)) => {
                let chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Min(20), Constraint::Length(panel_width)])
                    .split(area);
                (None, chunks[0], Some(chunks[1]))
            }
            (None, None) => (None, area, None),
        }
    }

    fn render_panel(&self, area: Rect, frame: &mut Frame, panel: PanelSlot) {
        match panel {
            PanelSlot::Help => render_help_panel(area, frame.buffer_mut(), &self.theme),
            PanelSlot::States => {
                let state = self.build_state_panel();
                render_state_panel(area, frame.buffer_mut(), &state, &self.theme);
            }
            PanelSlot::Memory => {
                let state = self.build_memory_panel();
                render_memory_panel(area, frame.buffer_mut(), &state, &self.theme);
            }
            PanelSlot::Context => {
                let state = ContextPanelState {
                    values: self.agent.get_context(),
                };
                render_context_panel(area, frame.buffer_mut(), &state, &self.theme);
            }
            PanelSlot::Tools => {
                let state = self.build_tools_panel();
                render_tools_panel(area, frame.buffer_mut(), &state, &self.theme);
            }
            PanelSlot::Persona => {
                let state = self.build_persona_panel();
                render_persona_panel(area, frame.buffer_mut(), &state, &self.theme);
            }
            PanelSlot::Facts => {
                let state = self.build_facts_panel();
                render_facts_panel(area, frame.buffer_mut(), &state, &self.theme);
            }
            PanelSlot::Agents => {
                let state = self.build_agents_panel();
                render_agents_panel(area, frame.buffer_mut(), &state, &self.theme);
            }
            PanelSlot::Relationship => {
                let state = self.build_relationship_panel();
                render_relationship_panel(area, frame.buffer_mut(), &state, &self.theme);
            }
        }
    }

    fn build_status_state(&self) -> StatusBarState {
        let info = self.agent.info();
        StatusBarState {
            agent_name: info.name.clone(),
            agent_version: info.version.clone(),
            current_state: self.agent.current_state(),
            budget_percent: self.agent.memory_token_budget().map(|_| 0.0),
            is_thinking: self.is_thinking,
            spinner_frame: self.spinner_frame,
            actor_id: self.agent.actor_id(),
        }
    }

    fn build_facts_panel(&self) -> crate::tui::widgets::facts_panel::FactsPanelState {
        use crate::tui::widgets::facts_panel::{FactEntry, FactsPanelState};
        let facts = self
            .agent
            .actor_facts()
            .into_iter()
            .map(|f| FactEntry {
                category: f.category.to_string(),
                content: f.content,
                confidence: f.confidence,
            })
            .collect();
        FactsPanelState {
            actor_id: self.agent.actor_id(),
            facts,
        }
    }

    fn build_relationship_panel(&self) -> RelationshipPanelState {
        let actor_id = self.agent.actor_id();
        let Some(manager) = self.agent.relationship_manager() else {
            return RelationshipPanelState {
                actor_id,
                configured: false,
                dimensions: Vec::new(),
                perceived_dimensions: Vec::new(),
                mutual_dimensions: Vec::new(),
                interaction_count: 0,
                events: Vec::new(),
            };
        };
        let relationship = actor_id.as_ref().and_then(|id| manager.get(id));
        let mut dimensions = relationship
            .as_ref()
            .map(|r| {
                r.dimensions
                    .iter()
                    .map(|(name, value)| RelationshipDimensionEntry {
                        name: name.clone(),
                        value: *value,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        dimensions.sort_by(|a, b| a.name.cmp(&b.name));
        let mut perceived_dimensions = relationship
            .as_ref()
            .map(|r| {
                r.perceived_actor_to_agent
                    .iter()
                    .map(|(name, value)| RelationshipDimensionEntry {
                        name: name.clone(),
                        value: *value,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        perceived_dimensions.sort_by(|a, b| a.name.cmp(&b.name));
        let mut mutual_dimensions = relationship
            .as_ref()
            .map(|r| {
                r.mutual_dimensions()
                    .iter()
                    .map(|(name, value)| RelationshipDimensionEntry {
                        name: name.clone(),
                        value: *value,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if perceived_dimensions.is_empty() {
            mutual_dimensions.clear();
        } else {
            mutual_dimensions.sort_by(|a, b| a.name.cmp(&b.name));
        }
        let events = relationship
            .as_ref()
            .map(|r| {
                r.notable_events
                    .iter()
                    .rev()
                    .take(5)
                    .map(|event| RelationshipEventEntry {
                        description: event.description.clone(),
                        significance: event.significance,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        RelationshipPanelState {
            actor_id,
            configured: true,
            dimensions,
            perceived_dimensions,
            mutual_dimensions,
            interaction_count: relationship.map(|r| r.interaction_count).unwrap_or(0),
            events,
        }
    }

    fn build_state_panel(&self) -> StatePanelState {
        let history = self.agent.state_history();
        let mut states: Vec<String> = Vec::new();
        for event in &history {
            if !states.contains(&event.from) {
                states.push(event.from.clone());
            }
            if !states.contains(&event.to) {
                states.push(event.to.clone());
            }
        }
        if let Some(current) = self.agent.current_state()
            && !states.contains(&current)
        {
            states.push(current);
        }

        StatePanelState {
            current_state: self.agent.current_state(),
            states,
            turn_count: history.len(),
            fallback: None,
            global_transitions: Vec::new(),
        }
    }

    fn build_memory_panel(&self) -> MemoryPanelState {
        let budget = self.agent.memory_token_budget();
        let facts_tokens: u32 = self
            .agent
            .actor_facts()
            .iter()
            .map(|f| (f.content.len() as u32 / 4) + 10)
            .sum();
        let relationship_tokens = self
            .agent
            .relationship_memory_text()
            .map(|text| estimate_tokens(&text))
            .unwrap_or(0);
        MemoryPanelState {
            message_count: 0,
            has_summary: false,
            summary_tokens: 0,
            recent_tokens: 0,
            facts_tokens,
            relationship_tokens,
            budget_total: budget.map(|b| b.total),
            budget_summary: budget.map(|b| b.allocation.summary),
            budget_recent: budget.map(|b| b.allocation.recent_messages),
            budget_facts: budget.map(|b| b.allocation.facts),
            budget_relationships: budget.map(|b| b.allocation.relationships),
            overflow_strategy: budget.map(|b| format!("{:?}", b.overflow_strategy)),
            warn_at: budget.map(|b| b.warn_at_percent as u32),
        }
    }

    fn build_tools_panel(&self) -> ToolsPanelState {
        let tool_names = unique_tool_names(
            self.observed_tool_names
                .iter()
                .chain(&self.current_tools)
                .cloned(),
        );
        ToolsPanelState {
            tool_names,
            last_call: self.last_tool_call.clone(),
        }
    }

    fn build_persona_panel(&self) -> PersonaPanelState {
        match self.agent.persona_manager() {
            Some(pm) => {
                let config = pm.config();
                PersonaPanelState {
                    name: config.identity.as_ref().map(|i| i.name.clone()),
                    role: config.identity.as_ref().map(|i| i.role.clone()),
                    traits: config
                        .traits
                        .as_ref()
                        .map(|t| t.personality.clone())
                        .unwrap_or_default(),
                    goals: config
                        .goals
                        .as_ref()
                        .map(|g| g.primary.clone())
                        .unwrap_or_default(),
                    hidden_secrets: config.secrets.as_ref().map(|s| s.len()).unwrap_or(0),
                }
            }
            None => PersonaPanelState {
                name: None,
                role: None,
                traits: Vec::new(),
                goals: Vec::new(),
                hidden_secrets: 0,
            },
        }
    }

    fn build_agents_panel(&self) -> AgentsPanelState {
        let mut agents = Vec::new();
        if let Some(registry) = self.agent.spawner_registry() {
            for info in registry.list() {
                let state = registry.get(&info.id).and_then(|a| a.current_state());
                agents.push(AgentEntry {
                    id: info.id.clone(),
                    name: info.name.clone(),
                    state,
                });
            }
        }
        AgentsPanelState {
            agents,
            orchestration_pattern: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_agents::llm::{
        ChatMessage, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
    };
    use ai_agents::{AgentBuilder, AgentResponse};
    use futures::stream;
    use tokio::sync::mpsc::{channel, error::TryRecvError};

    #[tokio::test]
    async fn question_navigation_preserves_modal_and_global_quit_works_inside_it() {
        let mut app = test_app();
        let (tx, _rx) = oneshot::channel();
        app.show_question_modal(PendingQuestion {
            request: QuestionRequest {
                question: "선택".into(),
                options: vec!["첫째".into(), "둘째".into()],
                multi_select: true,
                allow_other: true,
                ..Default::default()
            },
            respond_to: tx,
        });
        app.update(AppMessage::Key(KeyEvent::new(
            KeyCode::Down,
            KeyModifiers::NONE,
        )))
        .await;
        assert_eq!(
            app.modal
                .as_ref()
                .unwrap()
                .question
                .as_ref()
                .unwrap()
                .focused_option,
            1
        );
        assert_eq!(
            app.update(AppMessage::Key(KeyEvent::new(
                KeyCode::Char('c'),
                KeyModifiers::CONTROL
            )))
            .await,
            UpdateResult::Quit
        );
    }

    #[tokio::test]
    async fn task_question_escape_hides_without_default_and_exact_korean_answer_is_not_a_command() {
        let mut app = test_app();
        let (tx, mut rx) = channel(16);
        app.attach_host(tx);
        app.host_output(crate::session::Output::Prompt {
            id: 7,
            text: "local secret".into(),
            question: Some(QuestionRequest {
                question: "답변".into(),
                allow_other: true,
                default: Some(serde_json::json!("default")),
                ..Default::default()
            }),
        });
        app.update(AppMessage::Key(KeyEvent::new(
            KeyCode::Esc,
            KeyModifiers::NONE,
        )))
        .await;
        assert!(app.modal.is_none());
        assert!(rx.try_recv().is_err());
        app.update(AppMessage::Key(KeyEvent::new(
            KeyCode::Char('p'),
            KeyModifiers::CONTROL,
        )))
        .await;
        for ch in "중단하지 마세요".chars() {
            app.update(AppMessage::Key(KeyEvent::new(
                KeyCode::Char(ch),
                KeyModifiers::NONE,
            )))
            .await;
        }
        app.update(AppMessage::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )))
        .await;
        let crate::session::Input::Reply { id, line } = rx.try_recv().unwrap() else {
            panic!("not a bound answer")
        };
        assert_eq!(id, 7);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&line).unwrap()["other_text"],
            "중단하지 마세요"
        );
    }

    #[test]
    fn late_previous_operation_results_do_not_unlock_current_input() {
        let mut app = test_app();
        app.host_generation = 2;
        app.host_busy = true;
        app.host_output(crate::session::Output::Result {
            generation: 1,
            result: Box::new(crate::session::SessionResult {
                primary: crate::session::OperationResult::Chat(AgentResponse::new("old")),
                results: Default::default(),
                quit: false,
            }),
        });
        assert!(app.host_busy);
        assert!(app.last_result.is_none());
        app.host_output(crate::session::Output::Rejected {
            generation: 1,
            message: "old rejection".into(),
        });
        assert!(app.host_busy);
        app.host_output(crate::session::Output::Rejected {
            generation: 2,
            message: "current rejection".into(),
        });
        assert!(!app.host_busy);
    }

    #[tokio::test]
    async fn hidden_question_draft_and_scroll_survive_reopening() {
        let mut app = test_app();
        let (tx, _rx) = channel(16);
        app.attach_host(tx);
        app.host_output(crate::session::Output::Prompt {
            id: 1,
            text: (0..50).map(|n| format!("review line {n}\n")).collect(),
            question: None,
        });
        app.modal
            .as_mut()
            .unwrap()
            .question
            .as_mut()
            .unwrap()
            .text_input = "내 초안".into();
        app.update(AppMessage::Key(KeyEvent::new(
            KeyCode::PageDown,
            KeyModifiers::NONE,
        )))
        .await;
        app.update(AppMessage::Key(KeyEvent::new(
            KeyCode::Esc,
            KeyModifiers::NONE,
        )))
        .await;
        app.update(AppMessage::Key(KeyEvent::new(
            KeyCode::Char('p'),
            KeyModifiers::CONTROL,
        )))
        .await;
        let modal = app.modal.as_ref().unwrap();
        assert_eq!(modal.review_scroll, 5);
        assert_eq!(modal.question.as_ref().unwrap().text_input, "내 초안");
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(rendered.contains("review line"));
    }

    #[test]
    fn stale_prompt_close_cannot_remove_newer_prompt() {
        let mut app = test_app();
        for id in [1, 2] {
            app.host_output(crate::session::Output::Prompt {
                id,
                text: "Decision".into(),
                question: None,
            });
        }
        app.host_output(crate::session::Output::PromptClosed(1));
        assert_eq!(app.host_prompt.as_ref().unwrap().0, 2);
        assert!(app.modal.is_some());
    }

    #[test]
    fn tiny_terminals_and_long_korean_questions_render_without_panicking() {
        let mut app = test_app();
        app.host_output(crate::session::Output::Prompt {
            id: 1,
            text: "긴 질문".repeat(1024),
            question: None,
        });
        for (w, h) in [(0, 0), (2, 2), (8, 4), (20, 8), (80, 24)] {
            let mut terminal =
                ratatui::Terminal::new(ratatui::backend::TestBackend::new(w, h)).unwrap();
            terminal.draw(|frame| app.render(frame)).unwrap();
        }
    }

    #[test]
    fn bounded_history_does_not_truncate_the_authoritative_result() {
        let mut app = test_app();
        let exact = "한글".repeat(20_000);
        app.host_output(crate::session::Output::Result {
            generation: 0,
            result: Box::new(crate::session::SessionResult {
                primary: crate::session::OperationResult::Chat(AgentResponse::new(&exact)),
                results: Default::default(),
                quit: false,
            }),
        });
        let crate::session::OperationResult::Chat(response) =
            &app.last_result.as_ref().unwrap().primary
        else {
            panic!()
        };
        assert_eq!(response.content, exact);
        assert!(app.chat.messages.last().unwrap().content.len() < 16_384);
        for _ in 0..1000 {
            app.add_system_message("display");
        }
        app.bound_display();
        assert!(app.chat.messages.len() <= 256);
    }

    struct UnusedLlm;

    #[async_trait::async_trait]
    impl LLMProvider for UnusedLlm {
        async fn complete(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> Result<LLMResponse, LLMError> {
            Err(LLMError::Other("unused test provider".to_string()))
        }

        async fn complete_stream(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> Result<
            Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>,
            LLMError,
        > {
            Err(LLMError::Other("unused test provider".to_string()))
        }

        fn provider_name(&self) -> &str {
            "unused-test-provider"
        }

        fn supports(&self, _feature: LLMFeature) -> bool {
            false
        }
    }

    fn test_app() -> App {
        let agent = AgentBuilder::new()
            .system_prompt("Test TUI ownership.")
            .llm(Arc::new(UnusedLlm))
            .build()
            .expect("test agent should build");
        let (wake_tx, _wake_rx) = channel(32);
        App::new(
            Arc::new(agent),
            CliReplConfig::default(),
            wake_tx,
            Theme::default(),
            "dark".to_string(),
        )
    }

    fn content(text: &str) -> AgentStreamEvent {
        AgentStreamEvent::Chunk(StreamChunk::Content {
            text: text.to_string(),
        })
    }

    fn agent_messages(app: &App) -> Vec<(String, Vec<String>)> {
        app.chat
            .messages
            .iter()
            .filter(|message| message.role == Role::Agent)
            .map(|message| (message.content.clone(), message.tools.clone()))
            .collect()
    }

    #[tokio::test]
    async fn incomplete_eof_forwards_content_then_chat_error() {
        let (turn_tx, mut turn_rx) = channel(32);
        let (wake_tx, mut wake_rx) = channel(32);

        let outcome = forward_stream_to_tui(
            stream::iter(vec![content("partial")]),
            &turn_tx,
            &wake_tx,
            7,
        )
        .await;

        assert_eq!(outcome, StreamDriveOutcome::IncompleteEof);
        assert!(matches!(
            turn_rx.recv().await,
            Some(TurnMessage::StreamEvent { turn_id: 7, .. })
        ));
        assert!(matches!(
            turn_rx.recv().await,
            Some(TurnMessage::ChatError { turn_id: 7, message })
                if message == INCOMPLETE_STREAM_ERROR
        ));
        assert!(matches!(wake_rx.recv().await, Some(AppMessage::Tick)));
        assert!(matches!(wake_rx.recv().await, Some(AppMessage::Tick)));
        assert!(matches!(turn_rx.try_recv(), Err(TryRecvError::Empty)));
    }

    #[tokio::test]
    async fn final_is_forwarded_without_a_second_terminal_message() {
        let (turn_tx, mut turn_rx) = channel(32);
        let (wake_tx, mut wake_rx) = channel(32);
        let events = stream::iter(vec![AgentStreamEvent::Final(AgentResponse::new("answer"))]);

        let outcome = forward_stream_to_tui(events, &turn_tx, &wake_tx, 7).await;

        assert_eq!(outcome, StreamDriveOutcome::Final);
        assert!(matches!(
            turn_rx.recv().await,
            Some(TurnMessage::StreamEvent { turn_id: 7, event })
                if matches!(*event, AgentStreamEvent::Final(_))
        ));
        assert!(matches!(wake_rx.recv().await, Some(AppMessage::Tick)));
        assert!(matches!(turn_rx.try_recv(), Err(TryRecvError::Empty)));
    }

    #[tokio::test]
    async fn terminal_error_is_forwarded_without_incomplete_eof() {
        let (turn_tx, mut turn_rx) = channel(32);
        let (wake_tx, mut wake_rx) = channel(32);
        let events = stream::iter(vec![
            AgentStreamEvent::Chunk(StreamChunk::Error {
                message: "failed".to_string(),
            }),
            content("after"),
        ]);

        let outcome = forward_stream_to_tui(events, &turn_tx, &wake_tx, 7).await;

        assert_eq!(outcome, StreamDriveOutcome::TerminalError);
        assert!(matches!(
            turn_rx.recv().await,
            Some(TurnMessage::StreamEvent { turn_id: 7, event })
                if matches!(*event, AgentStreamEvent::Chunk(StreamChunk::Error { .. }))
        ));
        assert!(matches!(wake_rx.recv().await, Some(AppMessage::Tick)));
        assert!(matches!(turn_rx.try_recv(), Err(TryRecvError::Empty)));
    }

    #[tokio::test]
    async fn closed_receiver_interrupts_a_pending_source() {
        let (turn_tx, _turn_rx) = channel(32);
        let (wake_tx, wake_rx) = channel(32);
        drop(wake_rx);

        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            forward_stream_to_tui(stream::pending(), &turn_tx, &wake_tx, 7),
        )
        .await
        .expect("closed receiver should stop a pending stream");

        assert_eq!(outcome, StreamDriveOutcome::ConsumerClosed);
    }

    #[test]
    fn abandoned_turn_messages_cannot_mutate_a_newer_active_turn() {
        let mut app = test_app();
        app.active_turn_id = Some(2);
        app.is_thinking = true;
        app.spinner_frame = 17;
        app.chat.streaming_content = Some("newer preview".to_string());
        app.current_tools = vec!["newer-tool".to_string()];
        app.chat.messages.push(DisplayMessage {
            role: Role::Agent,
            content: "newer agent message".to_string(),
            tools: vec!["committed-tool".to_string()],
            state_transition: None,
            timing_ms: None,
        });

        let spinner_before = (app.is_thinking, app.spinner_frame);
        let preview_before = app.chat.streaming_content.clone();
        let tools_before = app.current_tools.clone();
        let agent_messages_before = agent_messages(&app);

        for message in [
            TurnMessage::StreamEvent {
                turn_id: 1,
                event: Box::new(content("stale content")),
            },
            TurnMessage::StreamEvent {
                turn_id: 1,
                event: Box::new(AgentStreamEvent::Final(AgentResponse::new(
                    "stale stream final",
                ))),
            },
            TurnMessage::StreamEvent {
                turn_id: 1,
                event: Box::new(AgentStreamEvent::Chunk(StreamChunk::Error {
                    message: "stale stream error".to_string(),
                })),
            },
            TurnMessage::ChatError {
                turn_id: 1,
                message: "stale blocking error".to_string(),
            },
            TurnMessage::ChatResponse {
                turn_id: 1,
                response: Box::new(AgentResponse::new("stale blocking response")),
            },
        ] {
            app.turn_tx
                .try_send(message)
                .expect("turn queue should remain open");
        }

        app.drain_turn_messages();

        assert_eq!(app.active_turn_id, Some(2));
        assert_eq!((app.is_thinking, app.spinner_frame), spinner_before);
        assert_eq!(app.chat.streaming_content, preview_before);
        assert_eq!(app.current_tools, tools_before);
        assert_eq!(agent_messages(&app), agent_messages_before);
    }

    #[tokio::test]
    async fn escape_locally_abandons_ui_state_and_the_next_turn_can_commit() {
        let mut app = test_app();
        app.next_turn_id = 2;
        app.active_turn_id = Some(1);
        app.is_thinking = true;
        app.spinner_frame = 23;
        app.chat.streaming_content = Some("abandoned preview".to_string());
        app.current_tools = vec!["abandoned-tool".to_string()];

        let result = app
            .handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
            .await;

        assert_eq!(result, UpdateResult::Continue);
        assert_eq!(app.active_turn_id, None);
        assert!(!app.is_thinking);
        assert_eq!(app.spinner_frame, 23);
        assert!(app.chat.streaming_content.is_none());
        assert!(app.current_tools.is_empty());
        assert!(agent_messages(&app).is_empty());

        app.turn_tx
            .try_send(TurnMessage::StreamEvent {
                turn_id: 1,
                event: Box::new(content("late abandoned content")),
            })
            .expect("local abandon should not close the producer queue");
        app.drain_turn_messages();
        assert!(app.chat.streaming_content.is_none());
        assert!(agent_messages(&app).is_empty());

        let new_turn_id = app.next_turn_id;
        app.next_turn_id = app.next_turn_id.wrapping_add(1);
        app.active_turn_id = Some(new_turn_id);
        app.is_thinking = true;
        app.current_tools = vec!["new-tool".to_string()];

        app.turn_tx
            .try_send(TurnMessage::StreamEvent {
                turn_id: new_turn_id,
                event: Box::new(content("new preview")),
            })
            .expect("new turn content should enqueue");
        app.drain_turn_messages();

        assert_eq!(app.active_turn_id, Some(new_turn_id));
        assert!(app.is_thinking);
        assert_eq!(app.chat.streaming_content.as_deref(), Some("new preview"));
        assert_eq!(app.current_tools, vec!["new-tool"]);

        app.turn_tx
            .try_send(TurnMessage::StreamEvent {
                turn_id: new_turn_id,
                event: Box::new(AgentStreamEvent::Final(AgentResponse::new(
                    "new authoritative response",
                ))),
            })
            .expect("new turn final response should enqueue");
        app.drain_turn_messages();

        assert_eq!(app.active_turn_id, None);
        assert!(!app.is_thinking);
        assert!(app.chat.streaming_content.is_none());
        assert!(app.current_tools.is_empty());
        assert_eq!(
            agent_messages(&app),
            vec![("new authoritative response".to_string(), Vec::new())]
        );
    }

    #[test]
    fn final_response_commits_authoritative_content_and_unique_tools_once() {
        let mut chat = ChatState::new();
        chat.streaming_content = Some("partial".to_string());
        let mut is_thinking = true;
        let mut current_tools = vec!["search".to_string()];
        let mut observed_tool_names = vec!["search".to_string()];
        let response = AgentResponse {
            content: "authoritative".to_string(),
            metadata: None,
            tool_calls: Some(vec![
                ai_agents::agent::ToolCall {
                    id: "call-1".to_string(),
                    name: "search".to_string(),
                    arguments: serde_json::json!({}),
                },
                ai_agents::agent::ToolCall {
                    id: "call-2".to_string(),
                    name: "search".to_string(),
                    arguments: serde_json::json!({}),
                },
                ai_agents::agent::ToolCall {
                    id: "call-3".to_string(),
                    name: "read".to_string(),
                    arguments: serde_json::json!({}),
                },
            ]),
        };

        record_final_response(
            &mut chat,
            &mut is_thinking,
            &mut current_tools,
            &mut observed_tool_names,
            response,
            Some(42),
        );

        assert!(!is_thinking);
        assert!(chat.streaming_content.is_none());
        assert!(current_tools.is_empty());
        assert_eq!(observed_tool_names, vec!["search", "read"]);
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, Role::Agent);
        assert_eq!(chat.messages[0].content, "authoritative");
        assert_eq!(chat.messages[0].tools, vec!["search", "read"]);
        assert_eq!(chat.messages[0].timing_ms, Some(42));
    }

    #[test]
    fn stream_error_clears_provisional_state_without_committing_agent_content() {
        let mut chat = ChatState::new();
        chat.streaming_content = Some("partial".to_string());
        let mut is_thinking = true;
        let mut current_tools = vec!["search".to_string()];

        record_stream_error(
            &mut chat,
            &mut is_thinking,
            &mut current_tools,
            INCOMPLETE_STREAM_ERROR,
        );

        assert!(!is_thinking);
        assert!(chat.streaming_content.is_none());
        assert!(current_tools.is_empty());
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, Role::System);
        assert_eq!(
            chat.messages[0].content,
            format!("[Error] {}", INCOMPLETE_STREAM_ERROR)
        );
        assert!(
            !chat
                .messages
                .iter()
                .any(|message| message.role == Role::Agent)
        );
    }
}

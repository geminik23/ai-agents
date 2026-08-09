//
// TUI application state, event handling, and rendering.
//

use std::sync::Arc;
use std::time::Instant;

use tracing::Level;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tui_textarea::TextArea;

use ai_agents::memory::estimate_tokens;
use ai_agents::tools::{QuestionRequest, QuestionResponse};
use ai_agents::{Agent, AgentResponse, AgentStreamEvent, RuntimeAgent, StreamChunk};
use tokio::sync::oneshot;

use crate::question::response_from_default;
use crate::repl::{CliReplConfig, ReplMode, parse_relationship_perspective};
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
    turn_tx: &UnboundedSender<TurnMessage>,
    wake_tx: &UnboundedSender<AppMessage>,
    message: TurnMessage,
) -> bool {
    turn_tx.send(message).is_ok() && wake_tx.send(AppMessage::Tick).is_ok()
}

async fn forward_stream_to_tui<S>(
    stream: S,
    turn_tx: &UnboundedSender<TurnMessage>,
    wake_tx: &UnboundedSender<AppMessage>,
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
    wake_tx: UnboundedSender<AppMessage>,
    turn_tx: UnboundedSender<TurnMessage>,
    turn_rx: UnboundedReceiver<TurnMessage>,

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
}

struct PendingQuestionState {
    request: QuestionRequest,
    respond_to: Option<oneshot::Sender<QuestionResponse>>,
}

impl App {
    /// Build initial application state from an agent, config, and message sender.
    pub fn new(
        agent: Arc<RuntimeAgent>,
        config: CliReplConfig,
        tx: UnboundedSender<AppMessage>,
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
        let (turn_tx, turn_rx) = unbounded_channel();
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
        }
    }

    /// Process one incoming event and return whether the app should keep running.
    pub async fn update(&mut self, msg: AppMessage) -> UpdateResult {
        self.drain_turn_messages();
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
            match self.agent.save_session("default").await {
                Ok(()) => self.add_toast("Session saved"),
                Err(e) => self.add_toast(&format!("Save failed: {}", e)),
            }
            return UpdateResult::Continue;
        }

        // Ctrl+T cycles the color theme.
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('t') {
            self.cycle_theme();
            return UpdateResult::Continue;
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

        // Esc cancels streaming, closes panels, or closes completion.
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
        if key.code == KeyCode::Enter && !self.is_thinking {
            self.completions.close();
            let lines: Vec<String> = self.input.lines().iter().map(|s| s.to_string()).collect();
            let text = lines.join("\n").trim().to_string();
            if text.is_empty() {
                return UpdateResult::Continue;
            }
            self.input.select_all();
            self.input.cut();

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

        match trimmed {
            "/quit" | "/exit" => return UpdateResult::Quit,
            "/help" | "/?" => {
                self.toggle_panel(PanelSlot::Help);
            }
            "/reset" => match self.agent.reset().await {
                Ok(()) => self.add_system_message("Agent reset."),
                Err(e) => self.add_system_message(&format!("[Error] Reset failed: {}", e)),
            },
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
            _ if trimmed.starts_with("/save") => {
                let name = input.split_whitespace().nth(1).unwrap_or("default");
                match self.agent.save_session(name).await {
                    Ok(()) => self.add_toast(&format!("Saved '{}'", name)),
                    Err(e) => self.add_system_message(&format!("[Error] Save failed: {}", e)),
                }
            }
            _ if trimmed.starts_with("/load") => {
                let name = input.split_whitespace().nth(1).unwrap_or("default");
                match self.agent.load_session(name).await {
                    Ok(true) => self.add_toast(&format!("Loaded '{}'", name)),
                    Ok(false) => self.add_system_message(&format!("Session '{}' not found.", name)),
                    Err(e) => self.add_system_message(&format!("[Error] Load failed: {}", e)),
                }
            }
            _ if trimmed.starts_with("/delete") => {
                let name = match input.split_whitespace().nth(1) {
                    Some(n) => n.to_string(),
                    None => {
                        self.add_system_message("Usage: /delete <session_name>");
                        return UpdateResult::Continue;
                    }
                };
                match self.agent.delete_session(&name).await {
                    Ok(()) => self.add_toast(&format!("Deleted '{}'", name)),
                    Err(e) => self.add_system_message(&format!("[Error] Delete failed: {}", e)),
                }
            }
            _ if trimmed.starts_with("/sessions") => {
                self.handle_tui_sessions_command(input).await;
            }
            "/cleanup" => match self.agent.cleanup_expired_sessions().await {
                Ok(0) => self.add_system_message("No expired sessions to clean up."),
                Ok(n) => self.add_toast(&format!("Cleaned up {} expired session(s).", n)),
                Err(e) => self.add_system_message(&format!("[Error] Cleanup failed: {}", e)),
            },
            _ if trimmed.starts_with("/actor") => {
                self.handle_tui_actor_command(input).await;
            }
            _ if trimmed.starts_with("/facts") => {
                self.handle_tui_facts_command(input).await;
            }
            _ if trimmed.starts_with("/relationship") || trimmed.starts_with("/rel") => {
                self.handle_tui_relationship_command(input).await;
            }
            _ => {
                self.add_system_message(&format!("Unknown command: {}", input));
            }
        }
        UpdateResult::Continue
    }

    async fn handle_tui_actor_command(&mut self, input: &str) {
        let parts: Vec<&str> = input.split_whitespace().collect();
        match parts.get(1).map(|s| s.to_lowercase()).as_deref() {
            None | Some("") => match self.agent.actor_id() {
                Some(id) => self.add_system_message(&format!("Actor: {}", id)),
                None => self.add_system_message("No actor ID set. Use /actor set <id>"),
            },
            Some("set") => {
                if let Some(id) = parts.get(2) {
                    match self.agent.set_actor_id(id) {
                        Ok(()) => {
                            if let Err(e) = self.agent.load_actor_memory().await {
                                self.add_system_message(&format!(
                                    "[Warning] Actor set but failed to load memory: {}",
                                    e
                                ));
                            } else {
                                let count = self.agent.actor_facts().len();
                                self.add_toast(&format!(
                                    "Actor set to: {}. {} fact(s) loaded.",
                                    id, count
                                ));
                            }
                        }
                        Err(e) => self.add_system_message(&format!("[Error] {}", e)),
                    }
                } else {
                    self.add_system_message("Usage: /actor set <id>");
                }
            }
            Some("facts") => {
                let actor_id = match self.agent.actor_id() {
                    Some(id) => id,
                    None => {
                        self.add_system_message("No actor ID set. Use /actor set <id>");
                        return;
                    }
                };
                let facts = self.agent.actor_facts();
                let cat_filter = parts.get(2).map(|s| s.to_lowercase());
                let filtered: Vec<_> = if let Some(ref cat) = cat_filter {
                    facts
                        .iter()
                        .filter(|f| f.category.to_string().to_lowercase() == *cat)
                        .collect()
                } else {
                    facts.iter().collect()
                };
                if filtered.is_empty() {
                    self.add_system_message(&format!("No facts for actor: {}", actor_id));
                } else {
                    let mut msg =
                        format!("Facts for actor {} ({} shown):", actor_id, filtered.len());
                    for fact in &filtered {
                        msg.push_str(&format!(
                            "\n  [{}] {} ({:.2})",
                            fact.category, fact.content, fact.confidence
                        ));
                    }
                    self.add_system_message(&msg);
                }
            }
            Some("delete") => {
                let actor_id = match self.agent.actor_id() {
                    Some(id) => id,
                    None => {
                        self.add_system_message("No actor ID set. Use /actor set <id>");
                        return;
                    }
                };
                match self.agent.delete_actor_data(&actor_id).await {
                    Ok(()) => self.add_toast(&format!("All data deleted for actor: {}", actor_id)),
                    Err(e) => self.add_system_message(&format!("[Error] {}", e)),
                }
            }
            _ => {
                self.add_system_message("Usage: /actor [set <id> | facts [category] | delete]");
            }
        }
    }

    async fn handle_tui_facts_command(&mut self, input: &str) {
        let parts: Vec<&str> = input.split_whitespace().collect();
        match parts.get(1).map(|s| s.to_lowercase()).as_deref() {
            Some("extract") => {
                let n: usize = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
                match self.agent.extract_facts(n).await {
                    Ok(facts) if facts.is_empty() => {
                        self.add_system_message("No new facts extracted.");
                    }
                    Ok(facts) => {
                        let mut msg = format!("Extracted {} fact(s):", facts.len());
                        for fact in &facts {
                            msg.push_str(&format!(
                                "\n  [{}] {} ({:.2})",
                                fact.category, fact.content, fact.confidence
                            ));
                        }
                        self.add_system_message(&msg);
                    }
                    Err(e) => {
                        self.add_system_message(&format!("[Error] Extraction failed: {}", e));
                    }
                }
            }
            _ => {
                let facts = self.agent.actor_facts();
                if facts.is_empty() {
                    self.add_system_message("No facts for current actor.");
                } else {
                    let mut msg = format!("Facts ({}):", facts.len());
                    for fact in &facts {
                        msg.push_str(&format!(
                            "\n  [{}] {} ({:.2})",
                            fact.category, fact.content, fact.confidence
                        ));
                    }
                    self.add_system_message(&msg);
                }
            }
        }
    }

    async fn handle_tui_relationship_command(&mut self, input: &str) {
        let parts: Vec<&str> = input.split_whitespace().collect();
        let Some(manager) = self.agent.relationship_manager() else {
            self.add_system_message("Relationship memory is not configured.");
            return;
        };
        let actor_id = match self.agent.actor_id() {
            Some(id) => id,
            None => {
                self.add_system_message("No actor ID set. Use /actor set <id>");
                return;
            }
        };
        let _ = self.agent.load_actor_relationship().await;

        match parts.get(1).map(|s| s.to_lowercase()).as_deref() {
            None | Some("") => self.toggle_panel(PanelSlot::Relationship),
            Some("events") => {
                let relationship = manager.get_or_create(&actor_id, None);
                if relationship.notable_events.is_empty() {
                    self.add_system_message(&format!(
                        "No notable relationship events for actor: {}",
                        actor_id
                    ));
                    return;
                }
                let mut msg = format!("Relationship events for actor {}:", actor_id);
                for event in &relationship.notable_events {
                    msg.push_str(&format!(
                        "\n  [{}] ({:.2}) {}",
                        event.timestamp.to_rfc3339(),
                        event.significance,
                        event.description
                    ));
                }
                self.add_system_message(&msg);
            }
            Some("set") => {
                let dimension = match parts.get(2) {
                    Some(d) => *d,
                    None => {
                        self.add_system_message(
                            "Usage: /relationship set <dimension> <delta> [reason]",
                        );
                        return;
                    }
                };
                let delta = match parts.get(3).and_then(|s| s.parse::<f64>().ok()) {
                    Some(v) => v,
                    None => {
                        self.add_system_message(
                            "Usage: /relationship set <dimension> <delta> [reason]",
                        );
                        return;
                    }
                };
                let reason = if parts.len() > 4 {
                    Some(parts[4..].join(" "))
                } else {
                    None
                };
                match self
                    .agent
                    .update_relationship_dimension(dimension, delta, reason.as_deref())
                    .await
                {
                    Ok(change) => self.add_toast(&format!(
                        "{} {} {:+.2} -> {:.2}",
                        change.perspective, change.dimension, change.delta, change.current
                    )),
                    Err(e) => self.add_system_message(&format!("[Error] {}", e)),
                }
            }
            Some("setp") => {
                let perspective = match parts.get(2) {
                    Some(value) => match parse_relationship_perspective(value) {
                        Ok(p) => p,
                        Err(e) => {
                            self.add_system_message(&format!(
                                "{}. Usage: /relationship setp <perspective> <dimension> <delta> [reason]",
                                e
                            ));
                            return;
                        }
                    },
                    None => {
                        self.add_system_message(
                            "Usage: /relationship setp <perspective> <dimension> <delta> [reason]",
                        );
                        return;
                    }
                };
                let dimension = match parts.get(3) {
                    Some(d) => *d,
                    None => {
                        self.add_system_message(
                            "Usage: /relationship setp <perspective> <dimension> <delta> [reason]",
                        );
                        return;
                    }
                };
                let delta = match parts.get(4).and_then(|s| s.parse::<f64>().ok()) {
                    Some(v) => v,
                    None => {
                        self.add_system_message(
                            "Usage: /relationship setp <perspective> <dimension> <delta> [reason]",
                        );
                        return;
                    }
                };
                let reason = if parts.len() > 5 {
                    Some(parts[5..].join(" "))
                } else {
                    None
                };
                match self
                    .agent
                    .update_relationship_dimension_for_perspective(
                        perspective,
                        dimension,
                        delta,
                        reason.as_deref(),
                    )
                    .await
                {
                    Ok(change) => self.add_toast(&format!(
                        "{} {} {:+.2} -> {:.2}",
                        change.perspective, change.dimension, change.delta, change.current
                    )),
                    Err(e) => self.add_system_message(&format!("[Error] {}", e)),
                }
            }
            _ => self.add_system_message(
                "Usage: /relationship, /relationship events, /relationship set <dimension> <delta> [reason], /relationship setp <perspective> <dimension> <delta> [reason]",
            ),
        }
    }

    async fn handle_tui_sessions_command(&mut self, input: &str) {
        let parts: Vec<&str> = input.split_whitespace().collect();
        let mut actor_filter: Option<String> = None;
        let mut tag_filter: Option<String> = None;
        let mut i = 1;
        while i < parts.len() {
            match parts[i] {
                "--actor" => {
                    if let Some(v) = parts.get(i + 1) {
                        actor_filter = Some(v.to_string());
                        i += 2;
                        continue;
                    }
                }
                "--tag" => {
                    if let Some(v) = parts.get(i + 1) {
                        tag_filter = Some(v.to_string());
                        i += 2;
                        continue;
                    }
                }
                _ => {}
            }
            i += 1;
        }

        if actor_filter.is_some() || tag_filter.is_some() {
            let filter = ai_agents::facts::SessionFilter {
                actor_id: actor_filter,
                tags: tag_filter.map(|t| vec![t]),
                agent_id: None,
                created_after: None,
                created_before: None,
                limit: None,
            };
            match self.agent.list_sessions_filtered(&filter).await {
                Ok(summaries) if summaries.is_empty() => {
                    self.add_system_message("No sessions match the filter.");
                }
                Ok(summaries) => {
                    let mut msg = format!("Sessions ({} matched):", summaries.len());
                    for s in &summaries {
                        let actor = s.actor_id.as_deref().unwrap_or("-");
                        let tags = if s.tags.is_empty() {
                            String::new()
                        } else {
                            format!(" [{}]", s.tags.join(","))
                        };
                        msg.push_str(&format!(
                            "\n  {}  actor={}{}  msgs={}",
                            s.session_id, actor, tags, s.message_count
                        ));
                    }
                    self.add_system_message(&msg);
                }
                Err(e) => self.add_system_message(&format!("[Error] {}", e)),
            }
            return;
        }

        match self.agent.list_sessions().await {
            Ok(sessions) if sessions.is_empty() => {
                self.add_system_message("No saved sessions.");
            }
            Ok(sessions) => {
                let mut msg = format!("Sessions ({}):", sessions.len());
                for s in &sessions {
                    msg.push_str(&format!("\n  {}", s));
                }
                self.add_system_message(&msg);
            }
            Err(e) => self.add_system_message(&format!("[Error] {}", e)),
        }
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
                    Some(_) => self.add_toast(&format!("Removed: {}", key)),
                    None => self.add_system_message(&format!("Key not found: {}", key)),
                }
            }
            Some(other) => {
                self.add_system_message(&format!("Unknown: /context {}. Use: set, unset", other));
            }
        }
    }

    fn handle_modal_key(&mut self, key: KeyEvent) -> UpdateResult {
        // Check if this is a question modal and extract the state.
        let is_question = self.modal.as_ref().is_some_and(|m| m.question.is_some());

        if is_question {
            // Move the modal out, process the question key, then restore.
            let mut modal = self.modal.take().unwrap();
            if let Some(ref mut q) = modal.question {
                self.handle_question_modal_key(key, q);
            }
            // If modal was cleared during handling, keep it cleared.
            if self.modal.is_none() {
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
                KeyCode::Char(ch) => {
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
                            self.complete_question_response(selected, None, false);
                        }
                        Some("Submit text") => {
                            let text = if q.text_input.trim().is_empty() {
                                None
                            } else {
                                Some(q.text_input.trim().to_string())
                            };
                            self.complete_question_response(Vec::new(), text, false);
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
    use tokio::sync::mpsc::{error::TryRecvError, unbounded_channel};

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
        let (wake_tx, _wake_rx) = unbounded_channel();
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
        let (turn_tx, mut turn_rx) = unbounded_channel();
        let (wake_tx, mut wake_rx) = unbounded_channel();

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
        let (turn_tx, mut turn_rx) = unbounded_channel();
        let (wake_tx, mut wake_rx) = unbounded_channel();
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
        let (turn_tx, mut turn_rx) = unbounded_channel();
        let (wake_tx, mut wake_rx) = unbounded_channel();
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
        let (turn_tx, _turn_rx) = unbounded_channel();
        let (wake_tx, wake_rx) = unbounded_channel();
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
                .send(message)
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
            .send(TurnMessage::StreamEvent {
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
            .send(TurnMessage::StreamEvent {
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
            .send(TurnMessage::StreamEvent {
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

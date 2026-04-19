//
// TUI application state, event handling, and rendering.
//

use std::sync::Arc;
use std::time::Instant;

use tracing::Level;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders};
use tokio::sync::mpsc::UnboundedSender;
use tui_textarea::TextArea;

use ai_agents::{Agent, RuntimeAgent, StreamChunk};

use crate::repl::{CliReplConfig, ReplMode};
use crate::tui::event::AppMessage;
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
    state_panel::{StatePanelState, render_state_panel},
    status_bar::{StatusBarState, render_status_bar},
    toast::{Toast, render_toast},
    tools_panel::{LastToolCall, ToolsPanelState, render_tools_panel},
};

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
}

/// Main TUI application state, owning all widgets, agent handle, and input.
pub struct App {
    agent: Arc<RuntimeAgent>,
    config: CliReplConfig,
    theme: Theme,
    theme_name: String,
    bg_fill: Option<Color>,
    tx: UnboundedSender<AppMessage>,

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

    // Tool tracking from stream events
    current_tools: Vec<String>,
    observed_tool_names: Vec<String>,
    last_tool_call: Option<LastToolCall>,

    // Side panels
    left_panel: Option<PanelSlot>,
    right_panel: Option<PanelSlot>,

    // Modal overlay
    modal: Option<ModalState>,

    // Toast notifications
    toasts: Vec<Toast>,
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
        Self {
            agent,
            config,
            theme,
            theme_name,
            bg_fill,
            tx,
            input,
            is_command_mode: false,
            completions: CompletionState::new(),
            chat,
            is_thinking: false,
            spinner_frame: 0,
            chat_start: None,
            current_tools: Vec::new(),
            observed_tool_names: Vec::new(),
            last_tool_call: None,
            left_panel: None,
            right_panel: None,
            modal: None,
            toasts: Vec::new(),
        }
    }

    /// Process one incoming event and return whether the app should keep running.
    pub async fn update(&mut self, msg: AppMessage) -> UpdateResult {
        if self.modal.is_some() {
            if let AppMessage::Key(key) = msg {
                return self.handle_modal_key(key);
            }
        }

        match msg {
            AppMessage::Key(key) => self.handle_key(key).await,
            AppMessage::Resize(_, _) => UpdateResult::Continue,
            AppMessage::StreamChunk(chunk) => {
                self.handle_stream_chunk(chunk);
                UpdateResult::Continue
            }
            AppMessage::ChatResponse(response) => {
                let elapsed = self.chat_start.map(|s| s.elapsed().as_millis() as u64);
                self.is_thinking = false;
                self.chat.streaming_content = None;

                let tool_names = response
                    .tool_calls
                    .as_ref()
                    .map(|tc| tc.iter().map(|t| t.name.clone()).collect())
                    .unwrap_or_default();

                self.chat.messages.push(DisplayMessage {
                    role: Role::Agent,
                    content: response.content.clone(),
                    tools: tool_names,
                    state_transition: None,
                    timing_ms: elapsed,
                });
                self.chat.auto_scroll = true;
                UpdateResult::Continue
            }
            AppMessage::ChatError(err) => {
                self.is_thinking = false;
                self.chat.streaming_content = None;
                self.add_system_message(&format!("[Error] {}", err));
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

            let agent = Arc::clone(&self.agent);
            let tx = self.tx.clone();
            let streaming = self.config.mode == ReplMode::Streaming;

            tokio::spawn(async move {
                if streaming {
                    match agent.chat_stream(&text).await {
                        Ok(mut stream) => {
                            use futures::StreamExt;
                            while let Some(chunk) = stream.next().await {
                                if tx.send(AppMessage::StreamChunk(chunk)).is_err() {
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(AppMessage::ChatError(e.to_string()));
                        }
                    }
                } else {
                    match agent.chat(&text).await {
                        Ok(response) => {
                            let _ = tx.send(AppMessage::ChatResponse(Box::new(response)));
                        }
                        Err(e) => {
                            let _ = tx.send(AppMessage::ChatError(e.to_string()));
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

    fn handle_stream_chunk(&mut self, chunk: StreamChunk) {
        match chunk {
            StreamChunk::Content { text } => {
                let current = self.chat.streaming_content.get_or_insert_with(String::new);
                current.push_str(&text);
                self.chat.auto_scroll = true;
            }
            StreamChunk::ToolCallStart { name, .. } => {
                self.current_tools.push(name.clone());
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
                let preview = if output.len() > 30 {
                    format!("{}...", &output[..30])
                } else {
                    output
                };
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
                let elapsed = self.chat_start.map(|s| s.elapsed().as_millis() as u64);
                let content = self.chat.streaming_content.take().unwrap_or_default();
                self.is_thinking = false;
                self.chat.messages.push(DisplayMessage {
                    role: Role::Agent,
                    content,
                    tools: self.current_tools.drain(..).collect(),
                    state_transition: None,
                    timing_ms: elapsed,
                });
                self.chat.auto_scroll = true;
            }
            StreamChunk::Error { message } => {
                self.is_thinking = false;
                self.chat.streaming_content = None;
                self.add_system_message(&format!("[Error] {}", message));
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
                    .unwrap_or_else(|_| serde_json::Value::String(raw_value));
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
        if let Some(ref mut modal) = self.modal {
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
                KeyCode::Enter | KeyCode::Esc => {
                    self.modal = None;
                }
                _ => {}
            }
        }
        UpdateResult::Continue
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
                render_facts_panel(area, frame.buffer_mut(), &self.theme);
            }
            PanelSlot::Agents => {
                let state = self.build_agents_panel();
                render_agents_panel(area, frame.buffer_mut(), &state, &self.theme);
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
        if let Some(current) = self.agent.current_state() {
            if !states.contains(&current) {
                states.push(current);
            }
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
        MemoryPanelState {
            message_count: 0,
            has_summary: false,
            summary_tokens: 0,
            recent_tokens: 0,
            budget_total: budget.map(|b| b.total),
            budget_summary: budget.map(|b| b.allocation.summary),
            budget_recent: budget.map(|b| b.allocation.recent_messages),
            budget_facts: budget.map(|b| b.allocation.facts),
            overflow_strategy: budget.map(|b| format!("{:?}", b.overflow_strategy)),
            warn_at: budget.map(|b| b.warn_at_percent as u32),
        }
    }

    fn build_tools_panel(&self) -> ToolsPanelState {
        ToolsPanelState {
            tool_names: self.observed_tool_names.clone(),
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

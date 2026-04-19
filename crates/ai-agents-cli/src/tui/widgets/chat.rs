//
// Chat area widget: scrollable message history with role colors.
//

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::tui::theme::Theme;

/// Display role for messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Agent,
    System,
}

/// A single message for display.
#[derive(Debug, Clone)]
pub struct DisplayMessage {
    pub role: Role,
    pub content: String,
    pub tools: Vec<String>,
    pub state_transition: Option<(String, String)>,
    pub timing_ms: Option<u64>,
}

/// Tracks chat messages, scroll position, and display options.
pub struct ChatState {
    pub messages: Vec<DisplayMessage>,
    pub scroll_offset: u16,
    pub auto_scroll: bool,
    pub streaming_content: Option<String>,
    pub show_tool_calls: bool,
    pub show_timing: bool,
}

impl ChatState {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            scroll_offset: 0,
            auto_scroll: true,
            streaming_content: None,
            show_tool_calls: false,
            show_timing: false,
        }
    }

    /// Calculate the total number of lines needed for all messages.
    pub fn total_lines(&self, width: u16) -> u16 {
        let w = width.saturating_sub(2) as usize;
        if w == 0 {
            return 0;
        }
        let mut total: u16 = 0;
        for msg in &self.messages {
            total += message_lines(msg, w, self.show_tool_calls, self.show_timing);
            total += 1;
        }
        if let Some(ref sc) = self.streaming_content {
            let line = format!("Agent: {}", sc);
            total += (line.len() as u16 / w as u16) + 1;
            total += 1;
        }
        total
    }

    /// Scroll to the bottom if auto-scroll is enabled.
    pub fn scroll_to_bottom(&mut self, visible_height: u16, content_height: u16) {
        if self.auto_scroll && content_height > visible_height {
            self.scroll_offset = content_height.saturating_sub(visible_height);
        }
    }
}

fn message_lines(msg: &DisplayMessage, width: usize, show_tools: bool, show_timing: bool) -> u16 {
    let prefix_len = match msg.role {
        Role::User => 5,
        Role::Agent => 7,
        Role::System => 0,
    };
    // Count actual content lines including newlines in the message.
    let content_lines = msg.content.split('\n').count() as u16;
    let first_line_len = prefix_len + msg.content.split('\n').next().map(|s| s.len()).unwrap_or(0);
    // Approximate wrap lines for the first line.
    let wrap_extra = if width > 0 {
        (first_line_len / width) as u16
    } else {
        0
    };
    let mut lines = content_lines + wrap_extra;
    if show_tools && !msg.tools.is_empty() {
        lines += 1;
    }
    if show_timing && msg.timing_ms.is_some() {
        lines += 1;
    }
    if msg.state_transition.is_some() {
        lines += 1;
    }
    lines
}

/// Strip surrounding quotes from a string if present.
fn strip_quotes(s: &str) -> &str {
    let trimmed = s.trim();
    if trimmed.len() >= 2
        && ((trimmed.starts_with('"') && trimmed.ends_with('"'))
            || (trimmed.starts_with('\'') && trimmed.ends_with('\'')))
    {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    }
}

/// Render the chat message area into the given rect.
pub fn render_chat(area: Rect, buf: &mut Buffer, state: &ChatState, theme: &Theme) {
    let block = Block::default().borders(Borders::NONE);
    let inner = block.inner(area);
    block.render(area, buf);

    let mut lines: Vec<Line> = Vec::new();

    for msg in &state.messages {
        let (prefix, style) = match msg.role {
            Role::User => ("You: ", theme.user_style),
            Role::Agent => ("Agent: ", theme.agent_style),
            Role::System => ("", theme.system_style),
        };

        // Split content on newlines so multi-line messages render correctly.
        let clean_content = strip_quotes(&msg.content);
        let content_lines: Vec<&str> = clean_content.split('\n').collect();

        for (i, line_text) in content_lines.iter().enumerate() {
            if i == 0 && !prefix.is_empty() {
                // First line gets the role prefix.
                lines.push(Line::from(vec![
                    Span::styled(prefix, style.add_modifier(Modifier::BOLD)),
                    Span::styled(*line_text, style),
                ]));
            } else if i == 0 {
                // System messages have no prefix.
                lines.push(Line::from(Span::styled(*line_text, style)));
            } else {
                // Continuation lines indented to align with content.
                let indent = " ".repeat(prefix.len());
                lines.push(Line::from(vec![
                    Span::raw(indent),
                    Span::styled(*line_text, style),
                ]));
            }
        }

        if state.show_tool_calls && !msg.tools.is_empty() {
            let tool_text = format!("  [Tools: {}]", msg.tools.join(", "));
            lines.push(Line::from(Span::styled(tool_text, theme.tool_style)));
        }

        if let Some((ref from, ref to)) = msg.state_transition {
            let trans = format!("  [State: {} -> {}]", from, to);
            lines.push(Line::from(Span::styled(trans, theme.system_style)));
        }

        if state.show_timing {
            if let Some(ms) = msg.timing_ms {
                let timing = format!("  ({:.1}s)", ms as f64 / 1000.0);
                lines.push(Line::from(Span::styled(timing, theme.hint_style)));
            }
        }

        lines.push(Line::from(""));
    }

    if let Some(ref content) = state.streaming_content {
        // Streaming content can also contain newlines.
        let stream_lines: Vec<&str> = content.split('\n').collect();
        for (i, line_text) in stream_lines.iter().enumerate() {
            if i == 0 {
                lines.push(Line::from(vec![
                    Span::styled("Agent: ", theme.agent_style.add_modifier(Modifier::BOLD)),
                    Span::styled(*line_text, theme.agent_style),
                    Span::styled("|", theme.spinner_style),
                ]));
            } else {
                lines.push(Line::from(vec![
                    Span::raw("       "),
                    Span::styled(*line_text, theme.agent_style),
                ]));
            }
        }
        lines.push(Line::from(""));
    }

    let text = Text::from(lines);
    let paragraph = Paragraph::new(text)
        .wrap(Wrap { trim: false })
        .scroll((state.scroll_offset, 0));

    paragraph.render(inner, buf);
}

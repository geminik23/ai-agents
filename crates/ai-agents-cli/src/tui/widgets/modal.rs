//
// Modal dialog for HITL approval and confirmations.
//

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};

use crate::tui::theme::Theme;

pub struct ModalState {
    pub title: String,
    pub message: String,
    pub context_lines: Vec<(String, String)>,
    pub selected_button: usize,
    pub buttons: Vec<String>,
}

impl ModalState {
    /// Create an approval modal.
    pub fn approval(message: &str, context: Vec<(String, String)>) -> Self {
        Self {
            title: "APPROVAL REQUIRED".to_string(),
            message: message.to_string(),
            context_lines: context,
            selected_button: 0,
            buttons: vec!["Approve".to_string(), "Reject".to_string()],
        }
    }

    /// Create a confirmation modal.
    pub fn confirm(title: &str, message: &str) -> Self {
        Self {
            title: title.to_string(),
            message: message.to_string(),
            context_lines: Vec::new(),
            selected_button: 0,
            buttons: vec!["Yes".to_string(), "No".to_string()],
        }
    }
}

pub fn render_modal(area: Rect, buf: &mut Buffer, state: &ModalState, theme: &Theme) {
    // Center the modal
    let width = (area.width * 60 / 100).min(50).max(30);
    let content_lines = 6 + state.context_lines.len() as u16;
    let height = content_lines.min(area.height - 4).max(8);
    let x = area.x + (area.width.saturating_sub(width)) / 2;
    let y = area.y + (area.height.saturating_sub(height)) / 2;
    let modal_area = Rect::new(x, y, width, height);

    // Clear the area behind the modal
    Clear.render(modal_area, buf);

    let block = Block::default()
        .title(format!(" {} ", state.title))
        .borders(Borders::ALL)
        .border_style(theme.highlight_style)
        .title_style(theme.highlight_style);
    let inner = block.inner(modal_area);
    block.render(modal_area, buf);

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!("  {}", state.message),
        theme.value_style,
    )));

    if !state.context_lines.is_empty() {
        lines.push(Line::from(""));
        for (k, v) in &state.context_lines {
            lines.push(Line::from(vec![
                Span::styled(format!("  {}: ", k), theme.label_style),
                Span::styled(v.as_str(), theme.value_style),
            ]));
        }
    }

    lines.push(Line::from(""));

    // Buttons
    let mut button_spans: Vec<Span> = vec![Span::raw("        ")];
    for (i, btn) in state.buttons.iter().enumerate() {
        let style = if i == state.selected_button {
            theme.highlight_style
        } else {
            theme.hint_style
        };
        let label = if i == state.selected_button {
            format!("[ {} ]", btn)
        } else {
            format!("  {}  ", btn)
        };
        button_spans.push(Span::styled(label, style));
        button_spans.push(Span::raw("  "));
    }
    lines.push(Line::from(button_spans));

    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    paragraph.render(inner, buf);
}


//
// Modal dialog for HITL approval, confirmations, and structured questions.
//

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};

use crate::tui::theme::Theme;

/// Focus target within a question modal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionFocus {
    Options,
    TextInput,
    Actions,
}

/// State for a structured question modal with multi-select and free-text support.
#[derive(Debug, Clone)]
pub struct QuestionModalState {
    pub question: String,
    pub options: Vec<String>,
    pub multi_select: bool,
    pub allow_other: bool,
    pub default_label: Option<String>,
    pub selected_options: Vec<bool>,
    pub focused_option: usize,
    pub focus: QuestionFocus,
    pub text_input: String,
    pub selected_action: usize,
    pub actions: Vec<String>,
}

impl QuestionModalState {
    /// Build question modal state from a request and default label.
    pub fn new(
        question: &str,
        options: Vec<String>,
        multi_select: bool,
        allow_other: bool,
        default_label: Option<String>,
    ) -> Self {
        let selected_options = vec![false; options.len()];
        let mut actions = Vec::new();
        if !options.is_empty() {
            actions.push("Submit".to_string());
        }
        if allow_other {
            actions.push("Submit text".to_string());
        }
        if default_label.is_some() {
            actions.push("Use default".to_string());
        }
        actions.push("Cancel".to_string());
        let focus = if options.is_empty() && allow_other {
            QuestionFocus::TextInput
        } else {
            QuestionFocus::Options
        };
        Self {
            question: question.to_string(),
            options,
            multi_select,
            allow_other,
            default_label,
            selected_options,
            focused_option: 0,
            focus,
            text_input: String::new(),
            selected_action: 0,
            actions,
        }
    }

    /// Toggle the currently focused option.
    pub fn toggle_current(&mut self) {
        if let Some(slot) = self.selected_options.get_mut(self.focused_option) {
            *slot = !*slot;
        }
    }

    /// Select only the currently focused option for single-select mode.
    pub fn select_current_single(&mut self) {
        for slot in &mut self.selected_options {
            *slot = false;
        }
        if let Some(slot) = self.selected_options.get_mut(self.focused_option) {
            *slot = true;
        }
    }

    /// Return the selected option labels.
    pub fn selected_labels(&self) -> Vec<String> {
        self.options
            .iter()
            .zip(self.selected_options.iter())
            .filter(|(_, selected)| **selected)
            .map(|(label, _)| label.clone())
            .collect()
    }
}

pub struct ModalState {
    pub title: String,
    pub message: String,
    pub context_lines: Vec<(String, String)>,
    pub selected_button: usize,
    pub buttons: Vec<String>,
    pub question: Option<QuestionModalState>,
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
            question: None,
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
            question: None,
        }
    }

    /// Create a structured question modal.
    pub fn question(
        message: &str,
        options: Vec<String>,
        multi_select: bool,
        allow_other: bool,
        default_label: Option<String>,
    ) -> Self {
        let question_state =
            QuestionModalState::new(message, options, multi_select, allow_other, default_label);
        let buttons = question_state.actions.clone();
        Self {
            title: "QUESTION".to_string(),
            message: message.to_string(),
            context_lines: Vec::new(),
            selected_button: 0,
            buttons,
            question: Some(question_state),
        }
    }
}

pub fn render_modal(area: Rect, buf: &mut Buffer, state: &ModalState, theme: &Theme) {
    // Center the modal
    let width = (area.width * 60 / 100).clamp(30, 60);
    let content_lines = if state.question.is_some() {
        4 + state
            .question
            .as_ref()
            .map(|q| q.options.len() as u16 + 4)
            .unwrap_or(6)
    } else {
        6 + state.context_lines.len() as u16
    };
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

    if let Some(q) = &state.question {
        if !q.options.is_empty() {
            lines.push(Line::from(""));
            for (i, option) in q.options.iter().enumerate() {
                let marker = if q.multi_select {
                    if q.selected_options[i] { "[x]" } else { "[ ]" }
                } else if q.selected_options[i] {
                    "(>)"
                } else {
                    "( )"
                };
                let style = if i == q.focused_option && q.focus == QuestionFocus::Options {
                    theme.highlight_style
                } else {
                    theme.hint_style
                };
                lines.push(Line::from(Span::styled(
                    format!("  {} {}", marker, option),
                    style,
                )));
            }
        }

        if q.allow_other {
            lines.push(Line::from(""));
            let style = if q.focus == QuestionFocus::TextInput {
                theme.highlight_style
            } else {
                theme.hint_style
            };
            lines.push(Line::from(Span::styled(
                format!("  Text: {}_", q.text_input),
                style,
            )));
        }

        if let Some(ref default) = q.default_label {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("  Default: {}", default),
                theme.label_style,
            )));
        }

        lines.push(Line::from(""));
        let mut button_spans: Vec<Span> = vec![Span::raw("        ")];
        for (i, btn) in q.actions.iter().enumerate() {
            let style = if i == q.selected_action && q.focus == QuestionFocus::Actions {
                theme.highlight_style
            } else {
                theme.hint_style
            };
            let label = if i == q.selected_action && q.focus == QuestionFocus::Actions {
                format!("[ {} ]", btn)
            } else {
                format!("  {}  ", btn)
            };
            button_spans.push(Span::styled(label, style));
            button_spans.push(Span::raw("  "));
        }
        lines.push(Line::from(button_spans));
    } else {
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
    }

    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    paragraph.render(inner, buf);
}

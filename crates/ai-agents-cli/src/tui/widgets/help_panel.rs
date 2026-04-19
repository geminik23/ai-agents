//
// Help panel: key bindings reference.
//

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::tui::theme::Theme;

pub fn render_help_panel(area: Rect, buf: &mut Buffer, theme: &Theme) {
    let block = Block::default()
        .title(" Help ")
        .borders(Borders::ALL)
        .border_style(theme.panel_border)
        .title_style(theme.panel_title);

    let help_text = vec![
        Line::from(Span::styled("Key Bindings:", theme.highlight_style)),
        Line::from(""),
        Line::from(vec![
            Span::styled(" Enter     ", theme.label_style),
            Span::styled("Send message", theme.value_style),
        ]),
        Line::from(vec![
            Span::styled(" Ctrl+C    ", theme.label_style),
            Span::styled("Quit", theme.value_style),
        ]),
        Line::from(vec![
            Span::styled(" Ctrl+L    ", theme.label_style),
            Span::styled("Clear chat", theme.value_style),
        ]),
        Line::from(vec![
            Span::styled(" PgUp/Down ", theme.label_style),
            Span::styled("Scroll", theme.value_style),
        ]),
        Line::from(vec![
            Span::styled(" Esc       ", theme.label_style),
            Span::styled("Cancel/close", theme.value_style),
        ]),
        Line::from(vec![
            Span::styled(" Ctrl+S    ", theme.label_style),
            Span::styled("Quick save", theme.value_style),
        ]),
        Line::from(""),
        Line::from(Span::styled("Panels:", theme.highlight_style)),
        Line::from(""),
        Line::from(vec![
            Span::styled(" F1 ", theme.label_style),
            Span::styled("Help", theme.value_style),
        ]),
        Line::from(vec![
            Span::styled(" F2 ", theme.label_style),
            Span::styled("States", theme.value_style),
        ]),
        Line::from(vec![
            Span::styled(" F3 ", theme.label_style),
            Span::styled("Memory", theme.value_style),
        ]),
        Line::from(vec![
            Span::styled(" F4 ", theme.label_style),
            Span::styled("Context", theme.value_style),
        ]),
        Line::from(vec![
            Span::styled(" F5 ", theme.label_style),
            Span::styled("Tools", theme.value_style),
        ]),
        Line::from(vec![
            Span::styled(" F6 ", theme.label_style),
            Span::styled("Persona", theme.value_style),
        ]),
        Line::from(vec![
            Span::styled(" F7 ", theme.label_style),
            Span::styled("Facts", theme.value_style),
        ]),
        Line::from(vec![
            Span::styled(" F8 ", theme.label_style),
            Span::styled("Agents", theme.value_style),
        ]),
        Line::from(""),
        Line::from(Span::styled("Commands:", theme.highlight_style)),
        Line::from(""),
        Line::from(vec![
            Span::styled(" /help     ", theme.label_style),
            Span::styled("This help", theme.value_style),
        ]),
        Line::from(vec![
            Span::styled(" /quit     ", theme.label_style),
            Span::styled("Exit", theme.value_style),
        ]),
        Line::from(vec![
            Span::styled(" /reset    ", theme.label_style),
            Span::styled("Clear memory", theme.value_style),
        ]),
        Line::from(vec![
            Span::styled(" /state    ", theme.label_style),
            Span::styled("Current state", theme.value_style),
        ]),
        Line::from(vec![
            Span::styled(" /context  ", theme.label_style),
            Span::styled("Show context", theme.value_style),
        ]),
        Line::from(vec![
            Span::styled(" /memory   ", theme.label_style),
            Span::styled("Memory status", theme.value_style),
        ]),
        Line::from(vec![
            Span::styled(" /save     ", theme.label_style),
            Span::styled("Save session", theme.value_style),
        ]),
        Line::from(vec![
            Span::styled(" /load     ", theme.label_style),
            Span::styled("Load session", theme.value_style),
        ]),
    ];

    let paragraph = Paragraph::new(help_text)
        .block(block)
        .wrap(Wrap { trim: false });
    paragraph.render(area, buf);
}

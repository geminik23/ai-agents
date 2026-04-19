//
// Persona panel: agent identity, traits, goals.
//

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::tui::theme::Theme;

pub struct PersonaPanelState {
    pub name: Option<String>,
    pub role: Option<String>,
    pub traits: Vec<String>,
    pub goals: Vec<String>,
    pub hidden_secrets: usize,
}

pub fn render_persona_panel(
    area: Rect,
    buf: &mut Buffer,
    state: &PersonaPanelState,
    theme: &Theme,
) {
    let block = Block::default()
        .title(" Persona ")
        .borders(Borders::ALL)
        .border_style(theme.panel_border)
        .title_style(theme.panel_title);
    let inner = block.inner(area);
    block.render(area, buf);

    if state.name.is_none() {
        let text = Paragraph::new("  No persona\n  configured.").style(theme.hint_style);
        text.render(inner, buf);
        return;
    }

    let mut lines: Vec<Line> = Vec::new();

    if let Some(ref name) = state.name {
        lines.push(Line::from(Span::styled(
            name.as_str(),
            theme.highlight_style,
        )));
    }
    if let Some(ref role) = state.role {
        lines.push(Line::from(Span::styled(role.as_str(), theme.value_style)));
    }

    if !state.traits.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Traits:", theme.label_style)));
        for t in &state.traits {
            lines.push(Line::from(Span::styled(
                format!(" {}", t),
                theme.value_style,
            )));
        }
    }

    if !state.goals.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Goals:", theme.label_style)));
        for g in &state.goals {
            lines.push(Line::from(Span::styled(
                format!(" {}", g),
                theme.value_style,
            )));
        }
    }

    if state.hidden_secrets > 0 {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("Secrets: {} hidden", state.hidden_secrets),
            theme.hint_style,
        )));
    }

    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    paragraph.render(inner, buf);
}

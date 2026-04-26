//
// Facts panel widget: shows live actor facts when available.
//

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::tui::theme::Theme;

pub struct FactEntry {
    pub category: String,
    pub content: String,
    pub confidence: f32,
}

pub struct FactsPanelState {
    pub actor_id: Option<String>,
    pub facts: Vec<FactEntry>,
}

pub fn render_facts_panel(area: Rect, buf: &mut Buffer, state: &FactsPanelState, theme: &Theme) {
    let title = match state.actor_id {
        Some(ref id) => format!(" Facts [{}] ", id),
        None => " Facts ".to_string(),
    };

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(theme.panel_border)
        .title_style(theme.panel_title);

    if state.facts.is_empty() {
        let text = Paragraph::new(vec![
            Line::from(""),
            Line::from(Span::styled("  No facts yet.", theme.hint_style)),
            Line::from(Span::styled("  Enable via", theme.hint_style)),
            Line::from(Span::styled("  memory.facts in", theme.hint_style)),
            Line::from(Span::styled("  YAML, set an actor,", theme.hint_style)),
            Line::from(Span::styled("  and chat to extract.", theme.hint_style)),
        ])
        .block(block)
        .wrap(Wrap { trim: false });
        text.render(area, buf);
        return;
    }

    let mut lines: Vec<Line> = Vec::with_capacity(state.facts.len() * 2 + 1);
    lines.push(Line::from(""));
    for fact in &state.facts {
        let header = format!("  [{}] ({:.2})", fact.category, fact.confidence);
        lines.push(Line::from(Span::styled(header, theme.panel_title)));
        lines.push(Line::from(Span::styled(
            format!("    {}", fact.content),
            theme.hint_style,
        )));
    }

    let text = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    text.render(area, buf);
}

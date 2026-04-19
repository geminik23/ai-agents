//
// State panel: state machine visualization.
//

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::tui::theme::Theme;

pub struct StatePanelState {
    pub current_state: Option<String>,
    pub states: Vec<String>,
    pub turn_count: usize,
    pub fallback: Option<String>,
    pub global_transitions: Vec<String>,
}

pub fn render_state_panel(area: Rect, buf: &mut Buffer, state: &StatePanelState, theme: &Theme) {
    let block = Block::default()
        .title(" States ")
        .borders(Borders::ALL)
        .border_style(theme.panel_border)
        .title_style(theme.panel_title);

    if state.states.is_empty() && state.current_state.is_none() {
        let text = Paragraph::new("  No state machine\n  configured.")
            .block(block)
            .wrap(Wrap { trim: false });
        text.render(area, buf);
        return;
    }

    let mut lines: Vec<Line> = Vec::new();
    for s in &state.states {
        let is_current = state.current_state.as_deref() == Some(s.as_str());
        if is_current {
            lines.push(Line::from(vec![
                Span::styled("> ", theme.state_current),
                Span::styled(s.as_str(), theme.state_current),
                Span::styled(format!("  [{}]", state.turn_count), theme.hint_style),
            ]));
        } else {
            lines.push(Line::from(Span::styled(
                format!("  {}", s),
                theme.state_normal,
            )));
        }
        lines.push(Line::from(Span::styled("    |", theme.hint_style)));
    }
    if !lines.is_empty() {
        lines.pop();
    }

    if let Some(ref fb) = state.fallback {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  {} (fb)", fb),
            theme.hint_style,
        )));
    }

    if !state.global_transitions.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Global:", theme.label_style)));
        for t in &state.global_transitions {
            lines.push(Line::from(Span::styled(
                format!("  -> {}", t),
                theme.hint_style,
            )));
        }
    }

    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    paragraph.render(area, buf);
}

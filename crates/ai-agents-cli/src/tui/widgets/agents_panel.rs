//
// Agents panel: spawned agents and orchestration status.
//

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::tui::theme::Theme;

pub struct AgentEntry {
    pub id: String,
    pub name: String,
    pub state: Option<String>,
}

pub struct AgentsPanelState {
    pub agents: Vec<AgentEntry>,
    pub orchestration_pattern: Option<String>,
}

pub fn render_agents_panel(area: Rect, buf: &mut Buffer, state: &AgentsPanelState, theme: &Theme) {
    let title = format!(" Agents ({}) ", state.agents.len());
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(theme.panel_border)
        .title_style(theme.panel_title);
    let inner = block.inner(area);
    block.render(area, buf);

    if state.agents.is_empty() {
        let text = Paragraph::new("  No spawner\n  configured.").style(theme.hint_style);
        text.render(inner, buf);
        return;
    }

    let mut lines: Vec<Line> = Vec::new();
    for agent in &state.agents {
        lines.push(Line::from(Span::styled(
            format!(" {}", agent.id),
            theme.highlight_style,
        )));
        lines.push(Line::from(Span::styled(
            format!("  \"{}\"", agent.name),
            theme.value_style,
        )));
        if let Some(ref s) = agent.state {
            lines.push(Line::from(Span::styled(
                format!("  state: {}", s),
                theme.hint_style,
            )));
        }
        lines.push(Line::from(""));
    }

    if let Some(ref pattern) = state.orchestration_pattern {
        lines.push(Line::from(Span::styled(
            format!("Pattern: {}", pattern),
            theme.label_style,
        )));
    }

    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    paragraph.render(inner, buf);
}

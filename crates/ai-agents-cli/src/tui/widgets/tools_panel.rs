//
// Tools panel: available tools and last call result.
//

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::tui::theme::Theme;

pub struct ToolsPanelState {
    pub tool_names: Vec<String>,
    pub last_call: Option<LastToolCall>,
}

#[derive(Clone)]
pub struct LastToolCall {
    pub name: String,
    pub input_preview: String,
    pub output_preview: String,
    pub duration_ms: u64,
}

pub fn render_tools_panel(area: Rect, buf: &mut Buffer, state: &ToolsPanelState, theme: &Theme) {
    let block = Block::default()
        .title(format!(" Tools ({}) ", state.tool_names.len()))
        .borders(Borders::ALL)
        .border_style(theme.panel_border)
        .title_style(theme.panel_title);
    let inner = block.inner(area);
    block.render(area, buf);

    let mut lines: Vec<Line> = Vec::new();

    for name in &state.tool_names {
        lines.push(Line::from(Span::styled(
            format!(" {}", name),
            theme.value_style,
        )));
    }

    if let Some(ref call) = state.last_call {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Last call:", theme.label_style)));
        lines.push(Line::from(Span::styled(
            format!("  {}", call.name),
            theme.tool_style,
        )));
        if !call.input_preview.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("  > {}", call.input_preview),
                theme.hint_style,
            )));
        }
        if !call.output_preview.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("  < {}", call.output_preview),
                theme.value_style,
            )));
        }
        lines.push(Line::from(Span::styled(
            format!("  ({:.1}s)", call.duration_ms as f64 / 1000.0),
            theme.hint_style,
        )));
    }

    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    paragraph.render(inner, buf);
}

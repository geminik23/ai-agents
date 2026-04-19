//
// Title bar widget: agent name, state, budget, spinner.
//

use ratatui::prelude::*;
use ratatui::widgets::Paragraph;

use crate::tui::theme::{SPINNER_FRAMES, Theme};

pub struct StatusBarState {
    pub agent_name: String,
    pub agent_version: String,
    pub current_state: Option<String>,
    pub budget_percent: Option<f64>,
    pub is_thinking: bool,
    pub spinner_frame: usize,
}

pub fn render_status_bar(area: Rect, buf: &mut Buffer, state: &StatusBarState, theme: &Theme) {
    // Fill the entire row with the status background color.
    for x in area.x..area.x + area.width {
        if let Some(cell) = buf.cell_mut(Position::new(x, area.y)) {
            cell.set_style(theme.status_bg);
            cell.set_char(' ');
        }
    }

    let spinner = if state.is_thinking {
        let frame = state.spinner_frame % SPINNER_FRAMES.len();
        format!(" {} thinking", SPINNER_FRAMES[frame])
    } else {
        String::new()
    };

    let state_str = state
        .current_state
        .as_deref()
        .map(|s| format!("  [{}]", s))
        .unwrap_or_default();

    let budget_str = state
        .budget_percent
        .map(|p| format!("  {:.0}% tok", p))
        .unwrap_or_default();

    // Left side: agent identity and state.
    let left_spans = vec![
        Span::styled(" ", theme.status_fg),
        Span::styled(
            &state.agent_name,
            theme.status_fg.add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!(" v{}", state.agent_version), theme.status_fg),
        Span::styled(&state_str, theme.status_fg),
    ];

    // Right side: budget and spinner.
    let right_text = format!("{}{} ", budget_str, spinner);
    let right_width = right_text.len() as u16;

    // Render left-aligned portion.
    let left_line = Line::from(left_spans);
    let left_para = Paragraph::new(left_line).style(theme.status_bg);
    let left_area = Rect::new(area.x, area.y, area.width.saturating_sub(right_width), 1);
    left_para.render(left_area, buf);

    // Render right-aligned portion.
    if right_width > 0 && area.width > right_width {
        let right_x = area.x + area.width - right_width;
        let right_area = Rect::new(right_x, area.y, right_width, 1);
        let right_para = Paragraph::new(right_text)
            .style(theme.status_fg)
            .alignment(Alignment::Right);
        right_para.render(right_area, buf);
    }
}

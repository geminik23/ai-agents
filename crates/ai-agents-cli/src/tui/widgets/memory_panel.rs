//
// Memory panel: token budget bars and compression stats.
//

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::tui::theme::Theme;

pub struct MemoryPanelState {
    pub message_count: usize,
    pub has_summary: bool,
    pub summary_tokens: u32,
    pub recent_tokens: u32,
    pub budget_total: Option<u32>,
    pub budget_summary: Option<u32>,
    pub budget_recent: Option<u32>,
    pub budget_facts: Option<u32>,
    pub overflow_strategy: Option<String>,
    pub warn_at: Option<u32>,
}

impl MemoryPanelState {
    pub fn budget_percent(&self) -> Option<f64> {
        self.budget_total.map(|total| {
            if total == 0 {
                return 0.0;
            }
            let used = self.summary_tokens + self.recent_tokens;
            used as f64 / total as f64 * 100.0
        })
    }
}

pub fn render_memory_panel(area: Rect, buf: &mut Buffer, state: &MemoryPanelState, theme: &Theme) {
    let block = Block::default()
        .title(" Memory ")
        .borders(Borders::ALL)
        .border_style(theme.panel_border)
        .title_style(theme.panel_title);
    let inner = block.inner(area);
    block.render(area, buf);

    let mut lines: Vec<Line> = Vec::new();

    // Message count
    if state.has_summary {
        lines.push(Line::from(vec![
            Span::styled("Msgs: ", theme.label_style),
            Span::styled(format!("{} + S", state.message_count), theme.value_style),
        ]));
    } else {
        lines.push(Line::from(vec![
            Span::styled("Msgs: ", theme.label_style),
            Span::styled(format!("{}", state.message_count), theme.value_style),
        ]));
    }

    // Summary tokens
    lines.push(Line::from(vec![
        Span::styled("Sum:  ", theme.label_style),
        Span::styled(format!("{} tok", state.summary_tokens), theme.value_style),
    ]));

    lines.push(Line::from(""));

    // Budget section
    if let Some(total) = state.budget_total {
        let used = state.summary_tokens + state.recent_tokens;
        let pct = if total > 0 {
            used as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        let budget_style = theme.budget_style(pct);

        lines.push(Line::from(Span::styled("Budget:", theme.label_style)));

        // Budget bar as text
        let bar_width = 10usize;
        let filled = ((pct / 100.0) * bar_width as f64) as usize;
        let empty = bar_width.saturating_sub(filled);
        let bar = format!("[{}{}] {:.0}%", "=".repeat(filled), " ".repeat(empty), pct);
        lines.push(Line::from(Span::styled(bar, budget_style)));

        lines.push(Line::from(""));

        // Per-component breakdown
        if let Some(bs) = state.budget_summary {
            lines.push(Line::from(vec![
                Span::styled(" sum  ", theme.label_style),
                Span::styled(
                    format!("{}/{}", state.summary_tokens, bs),
                    theme.value_style,
                ),
            ]));
        }
        if let Some(br) = state.budget_recent {
            lines.push(Line::from(vec![
                Span::styled(" msgs ", theme.label_style),
                Span::styled(format!("{}/{}", state.recent_tokens, br), theme.value_style),
            ]));
        }
        if let Some(bf) = state.budget_facts {
            lines.push(Line::from(vec![
                Span::styled(" fact ", theme.label_style),
                Span::styled(format!("0/{}", bf), theme.value_style),
            ]));
        }

        lines.push(Line::from(""));

        if let Some(ref strategy) = state.overflow_strategy {
            lines.push(Line::from(vec![
                Span::styled("Over: ", theme.label_style),
                Span::styled(strategy.as_str(), theme.value_style),
            ]));
        }
        if let Some(warn) = state.warn_at {
            lines.push(Line::from(vec![
                Span::styled("Warn: ", theme.label_style),
                Span::styled(format!("{}%", warn), theme.value_style),
            ]));
        }
    } else {
        lines.push(Line::from(Span::styled("No budget", theme.hint_style)));
    }

    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    paragraph.render(inner, buf);
}

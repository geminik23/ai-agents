//
// Facts panel: placeholder until F08 is implemented.
//

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::tui::theme::Theme;

pub fn render_facts_panel(area: Rect, buf: &mut Buffer, theme: &Theme) {
    let block = Block::default()
        .title(" Facts ")
        .borders(Borders::ALL)
        .border_style(theme.panel_border)
        .title_style(theme.panel_title);

    let text = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled("  No facts configured.", theme.hint_style)),
        Line::from(Span::styled("  Enable via", theme.hint_style)),
        Line::from(Span::styled("  memory.facts in", theme.hint_style)),
        Line::from(Span::styled("  YAML.", theme.hint_style)),
    ])
    .block(block)
    .wrap(Wrap { trim: false });
    text.render(area, buf);
}

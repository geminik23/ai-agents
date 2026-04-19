//
// Hint bar widget: key bindings and contextual hints.
//

use ratatui::prelude::*;
use ratatui::widgets::Paragraph;

use crate::tui::theme::Theme;

pub struct HintBarState {
    pub is_command_mode: bool,
    pub panels_enabled: bool,
}

pub fn render_hint_bar(area: Rect, buf: &mut Buffer, state: &HintBarState, theme: &Theme) {
    let hints = if state.is_command_mode {
        " /help  /quit  /state  /memory  /context  Esc cancel"
    } else if state.panels_enabled {
        " Enter send  Ctrl+C quit  Ctrl+L clear  Ctrl+T theme  F1-F8 panels  / commands"
    } else {
        " Enter send  Ctrl+C quit  Ctrl+L clear  / commands"
    };

    let paragraph = Paragraph::new(hints).style(theme.hint_style);
    paragraph.render(area, buf);
}

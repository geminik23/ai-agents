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
        " /task run OBJECTIVE | F10 tasks | Ctrl+P pending | Esc hides"
    } else if state.panels_enabled {
        " Enter send | Ctrl+C shutdown | F10 tasks | Ctrl+P pending | / commands"
    } else {
        " Enter send  Ctrl+C quit  Ctrl+L clear  / commands"
    };

    let paragraph = Paragraph::new(hints).style(theme.hint_style);
    paragraph.render(area, buf);
}

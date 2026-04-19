//
// Toast notification widget.
//

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, Paragraph};

use crate::tui::theme::Theme;

pub struct Toast {
    pub message: String,
    pub remaining_ticks: u32,
}

impl Toast {
    pub fn new(message: impl Into<String>, duration_ticks: u32) -> Self {
        Self {
            message: message.into(),
            remaining_ticks: duration_ticks,
        }
    }

    pub fn is_expired(&self) -> bool {
        self.remaining_ticks == 0
    }

    pub fn tick(&mut self) {
        self.remaining_ticks = self.remaining_ticks.saturating_sub(1);
    }
}

pub fn render_toast(area: Rect, buf: &mut Buffer, toast: &Toast, theme: &Theme) {
    let width = (toast.message.len() as u16 + 4).min(area.width - 4);
    let x = area.x + area.width.saturating_sub(width) - 2;
    let y = area.y + 2;
    let toast_area = Rect::new(x, y, width, 3);

    Clear.render(toast_area, buf);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.toast_style);

    let paragraph = Paragraph::new(format!(" {} ", toast.message))
        .style(theme.toast_style)
        .block(block);
    paragraph.render(toast_area, buf);
}

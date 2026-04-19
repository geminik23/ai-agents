//
// Completion popup widget for slash command auto-complete.
//

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear};

use crate::tui::theme::Theme;

/// A slash command with name and description for the popup.
pub struct SlashCommand {
    pub name: &'static str,
    pub description: &'static str,
}

/// All available slash commands shown in the completion popup.
pub const SLASH_COMMANDS: &[SlashCommand] = &[
    SlashCommand {
        name: "/help",
        description: "Commands and key bindings",
    },
    SlashCommand {
        name: "/quit",
        description: "Exit the application",
    },
    SlashCommand {
        name: "/exit",
        description: "Exit the application",
    },
    SlashCommand {
        name: "/reset",
        description: "Clear memory and reset state",
    },
    SlashCommand {
        name: "/state",
        description: "Current state",
    },
    SlashCommand {
        name: "/history",
        description: "State transition log",
    },
    SlashCommand {
        name: "/info",
        description: "Agent name, version, skills",
    },
    SlashCommand {
        name: "/memory",
        description: "Memory and token budget",
    },
    SlashCommand {
        name: "/context",
        description: "Show or set context values",
    },
    SlashCommand {
        name: "/save",
        description: "Save session",
    },
    SlashCommand {
        name: "/load",
        description: "Load saved session",
    },
    SlashCommand {
        name: "/sessions",
        description: "List saved sessions",
    },
    SlashCommand {
        name: "/delete",
        description: "Delete saved session",
    },
];

/// Tracks the current completion popup state.
pub struct CompletionState {
    pub items: Vec<&'static SlashCommand>,
    pub selected: usize,
    pub visible: bool,
}

impl CompletionState {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            selected: 0,
            visible: false,
        }
    }

    pub fn selected_command(&self) -> Option<&str> {
        self.items.get(self.selected).map(|c| c.name)
    }

    pub fn move_up(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub fn move_down(&mut self) {
        if self.selected + 1 < self.items.len() {
            self.selected += 1;
        }
    }

    pub fn close(&mut self) {
        self.visible = false;
        self.items.clear();
        self.selected = 0;
    }
}

/// Render the completion popup as an overlay above the input area.
pub fn render_completions(
    input_area: Rect,
    _terminal_size: Rect,
    buf: &mut Buffer,
    state: &CompletionState,
    theme: &Theme,
) {
    if !state.visible || state.items.is_empty() {
        return;
    }

    let item_count = state.items.len() as u16;
    // No cap - show all items. The command list is small enough to fit.
    let popup_height = item_count + 2; // +2 for top and bottom borders

    // Position above the input area, clamped so it never overlaps the status bar (row 1).
    let popup_y = input_area.y.saturating_sub(popup_height).max(1);
    let actual_height = input_area.y.saturating_sub(popup_y);

    if actual_height < 3 {
        // Not enough room for even one item plus borders.
        return;
    }

    // Width: 1 marker + 1 space + 13-char padded name + 1 space + longest description + 1 padding.
    let content_width = state
        .items
        .iter()
        .map(|c| 16 + c.description.len()) // "> " + 13-char name + " " + description
        .max()
        .unwrap_or(30) as u16;
    let popup_width = (content_width + 2) // +2 for borders
        .min(input_area.width.saturating_sub(2))
        .max(30);
    let popup_x = input_area.x + 1;

    let popup_area = Rect::new(popup_x, popup_y, popup_width, actual_height);

    // Clear the area so chat text does not bleed through.
    Clear.render(popup_area, buf);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.panel_border);
    let inner = block.inner(popup_area);
    block.render(popup_area, buf);

    // Scroll window: keep selected item visible.
    let visible_rows = inner.height as usize;
    let scroll_offset = if state.selected >= visible_rows {
        state.selected - visible_rows + 1
    } else {
        0
    };

    for i in 0..visible_rows {
        let item_idx = scroll_offset + i;
        if item_idx >= state.items.len() {
            break;
        }

        let cmd = &state.items[item_idx];
        let y = inner.y + i as u16;
        let is_selected = item_idx == state.selected;

        let marker = if is_selected { ">" } else { " " };

        let row_style = if is_selected {
            theme.highlight_style
        } else {
            Style::default().fg(theme.value_style.fg.unwrap_or(Color::White))
        };

        let desc_style = if is_selected {
            theme.highlight_style
        } else {
            theme.hint_style
        };

        let name_span = Span::styled(format!("{} {:<13} ", marker, cmd.name), row_style);
        let desc_span = Span::styled(cmd.description, desc_style);
        let line = Line::from(vec![name_span, desc_span]);
        buf.set_line(inner.x, y, &line, inner.width);
    }
}

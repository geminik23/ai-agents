//
// TUI color definitions.
//

use ratatui::style::{Color, Modifier, Style};

use super::palette::ThemeColors;

/// Color and style theme for the TUI.
pub struct Theme {
    pub user_style: Style,
    pub agent_style: Style,
    pub system_style: Style,
    pub error_style: Style,
    pub status_bg: Style,
    pub status_fg: Style,
    pub hint_style: Style,
    pub border_style: Style,
    pub highlight_style: Style,
    pub input_style: Style,
    pub tool_style: Style,
    pub state_current: Style,
    pub state_normal: Style,
    pub panel_title: Style,
    pub panel_border: Style,
    pub label_style: Style,
    pub value_style: Style,
    pub budget_low: Style,
    pub budget_mid: Style,
    pub budget_high: Style,
    pub spinner_style: Style,
    pub toast_style: Style,
    pub log_style: Style,
    pub hint_text_style: Style,
}

impl Default for Theme {
    fn default() -> Self {
        Self::dark()
    }
}

impl Theme {
    /// Dark theme matching typical terminal backgrounds.
    pub fn dark() -> Self {
        Self {
            user_style: Style::default().fg(Color::Cyan),
            agent_style: Style::default().fg(Color::White),
            system_style: Style::default().fg(Color::Yellow),
            error_style: Style::default().fg(Color::Red),
            status_bg: Style::default().bg(Color::DarkGray),
            status_fg: Style::default().fg(Color::White).bg(Color::DarkGray),
            hint_style: Style::default().fg(Color::DarkGray),
            border_style: Style::default().fg(Color::DarkGray),
            highlight_style: Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
            input_style: Style::default().fg(Color::White),
            tool_style: Style::default().fg(Color::Magenta),
            state_current: Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
            state_normal: Style::default().fg(Color::Gray),
            panel_title: Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
            panel_border: Style::default().fg(Color::DarkGray),
            label_style: Style::default().fg(Color::DarkGray),
            value_style: Style::default().fg(Color::White),
            budget_low: Style::default().fg(Color::Green),
            budget_mid: Style::default().fg(Color::Yellow),
            budget_high: Style::default().fg(Color::Red),
            spinner_style: Style::default().fg(Color::Cyan),
            toast_style: Style::default().fg(Color::Black).bg(Color::Yellow),
            log_style: Style::default().fg(Color::DarkGray),
            hint_text_style: Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::ITALIC),
        }
    }

    /// Light theme for light terminal backgrounds.
    pub fn light() -> Self {
        Self {
            user_style: Style::default().fg(Color::Blue),
            agent_style: Style::default().fg(Color::Black),
            system_style: Style::default().fg(Color::DarkGray),
            error_style: Style::default().fg(Color::Red),
            status_bg: Style::default().bg(Color::LightBlue),
            status_fg: Style::default().fg(Color::Black).bg(Color::LightBlue),
            hint_style: Style::default().fg(Color::Gray),
            border_style: Style::default().fg(Color::Gray),
            highlight_style: Style::default()
                .fg(Color::Blue)
                .add_modifier(Modifier::BOLD),
            input_style: Style::default().fg(Color::Black),
            tool_style: Style::default().fg(Color::Magenta),
            state_current: Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
            state_normal: Style::default().fg(Color::DarkGray),
            panel_title: Style::default()
                .fg(Color::Blue)
                .add_modifier(Modifier::BOLD),
            panel_border: Style::default().fg(Color::Gray),
            label_style: Style::default().fg(Color::Gray),
            value_style: Style::default().fg(Color::Black),
            budget_low: Style::default().fg(Color::Green),
            budget_mid: Style::default().fg(Color::Yellow),
            budget_high: Style::default().fg(Color::Red),
            spinner_style: Style::default().fg(Color::Blue),
            toast_style: Style::default().fg(Color::White).bg(Color::Blue),
            log_style: Style::default().fg(Color::Gray),
            hint_text_style: Style::default()
                .fg(Color::Blue)
                .add_modifier(Modifier::ITALIC),
        }
    }

    /// Build a theme from an RGB color palette.
    pub fn from_palette(p: &ThemeColors) -> Self {
        Self {
            user_style: Style::default().fg(p.accent),
            agent_style: Style::default().fg(p.fg),
            system_style: Style::default().fg(p.warning),
            error_style: Style::default().fg(p.error),
            status_bg: Style::default().bg(p.surface),
            status_fg: Style::default().fg(p.fg).bg(p.surface),
            hint_style: Style::default().fg(p.dim),
            border_style: Style::default().fg(p.muted),
            highlight_style: Style::default().fg(p.accent).add_modifier(Modifier::BOLD),
            input_style: Style::default().fg(p.fg),
            tool_style: Style::default().fg(p.secondary),
            state_current: Style::default().fg(p.success).add_modifier(Modifier::BOLD),
            state_normal: Style::default().fg(p.dim),
            panel_title: Style::default().fg(p.accent).add_modifier(Modifier::BOLD),
            panel_border: Style::default().fg(p.muted),
            label_style: Style::default().fg(p.dim),
            value_style: Style::default().fg(p.fg),
            budget_low: Style::default().fg(p.success),
            budget_mid: Style::default().fg(p.warning),
            budget_high: Style::default().fg(p.error),
            spinner_style: Style::default().fg(p.info),
            toast_style: Style::default().fg(p.bg).bg(p.warning),
            log_style: Style::default().fg(p.dim),
            hint_text_style: Style::default().fg(p.info).add_modifier(Modifier::ITALIC),
        }
    }

    /// Return budget-level style based on usage percentage.
    pub fn budget_style(&self, percent: f64) -> Style {
        if percent >= 85.0 {
            self.budget_high
        } else if percent >= 60.0 {
            self.budget_mid
        } else {
            self.budget_low
        }
    }
}

/// Spinner frames for the thinking indicator.
pub const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

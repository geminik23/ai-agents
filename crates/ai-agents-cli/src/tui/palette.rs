//
// Color palettes for TUI themes.
// Each palette defines semantic color slots using exact RGB values.
//

use ratatui::style::Color;

use super::theme::Theme;

/// Semantic color slots that define a theme's visual identity.
/// Role names (not color names) so any palette produces a coherent theme.
pub struct ThemeColors {
    pub bg: Color,
    pub fg: Color,
    pub dim: Color,
    pub muted: Color,
    pub accent: Color,
    pub secondary: Color,
    pub success: Color,
    pub warning: Color,
    pub error: Color,
    pub info: Color,
    pub surface: Color,
}

impl ThemeColors {
    pub fn one_dark() -> Self {
        Self {
            bg: Color::Rgb(40, 44, 52),
            fg: Color::Rgb(171, 178, 191),
            dim: Color::Rgb(92, 99, 112),
            muted: Color::Rgb(62, 68, 81),
            accent: Color::Rgb(97, 175, 239),
            secondary: Color::Rgb(198, 120, 221),
            success: Color::Rgb(152, 195, 121),
            warning: Color::Rgb(229, 192, 123),
            error: Color::Rgb(224, 108, 117),
            info: Color::Rgb(86, 182, 194),
            surface: Color::Rgb(33, 37, 43),
        }
    }

    pub fn catppuccin_mocha() -> Self {
        Self {
            bg: Color::Rgb(30, 30, 46),
            fg: Color::Rgb(205, 214, 244),
            dim: Color::Rgb(108, 112, 134),
            muted: Color::Rgb(69, 71, 90),
            accent: Color::Rgb(137, 180, 250),
            secondary: Color::Rgb(203, 166, 247),
            success: Color::Rgb(166, 227, 161),
            warning: Color::Rgb(249, 226, 175),
            error: Color::Rgb(243, 139, 168),
            info: Color::Rgb(116, 199, 236),
            surface: Color::Rgb(49, 50, 68),
        }
    }

    pub fn dracula() -> Self {
        Self {
            bg: Color::Rgb(40, 42, 54),
            fg: Color::Rgb(248, 248, 242),
            dim: Color::Rgb(98, 114, 164),
            muted: Color::Rgb(68, 71, 90),
            accent: Color::Rgb(139, 233, 253),
            secondary: Color::Rgb(189, 147, 249),
            success: Color::Rgb(80, 250, 123),
            warning: Color::Rgb(241, 250, 140),
            error: Color::Rgb(255, 85, 85),
            info: Color::Rgb(255, 121, 198),
            surface: Color::Rgb(55, 57, 73),
        }
    }

    pub fn tokyo_night() -> Self {
        Self {
            bg: Color::Rgb(26, 27, 38),
            fg: Color::Rgb(192, 202, 245),
            dim: Color::Rgb(86, 95, 137),
            muted: Color::Rgb(54, 58, 79),
            accent: Color::Rgb(122, 162, 247),
            secondary: Color::Rgb(187, 154, 247),
            success: Color::Rgb(158, 206, 106),
            warning: Color::Rgb(224, 175, 104),
            error: Color::Rgb(247, 118, 142),
            info: Color::Rgb(125, 207, 255),
            surface: Color::Rgb(36, 40, 59),
        }
    }

    pub fn vscode_dark() -> Self {
        Self {
            bg: Color::Rgb(30, 30, 30),
            fg: Color::Rgb(212, 212, 212),
            dim: Color::Rgb(128, 128, 128),
            muted: Color::Rgb(59, 59, 59),
            accent: Color::Rgb(86, 156, 214),
            secondary: Color::Rgb(197, 134, 192),
            success: Color::Rgb(106, 153, 85),
            warning: Color::Rgb(220, 220, 170),
            error: Color::Rgb(244, 71, 71),
            info: Color::Rgb(78, 201, 176),
            surface: Color::Rgb(37, 37, 38),
        }
    }

    pub fn nord() -> Self {
        Self {
            bg: Color::Rgb(46, 52, 64),
            fg: Color::Rgb(216, 222, 233),
            dim: Color::Rgb(76, 86, 106),
            muted: Color::Rgb(59, 66, 82),
            accent: Color::Rgb(136, 192, 208),
            secondary: Color::Rgb(180, 142, 173),
            success: Color::Rgb(163, 190, 140),
            warning: Color::Rgb(235, 203, 139),
            error: Color::Rgb(191, 97, 106),
            info: Color::Rgb(129, 161, 193),
            surface: Color::Rgb(59, 66, 82),
        }
    }

    pub fn gruvbox_dark() -> Self {
        Self {
            bg: Color::Rgb(40, 40, 40),
            fg: Color::Rgb(235, 219, 178),
            dim: Color::Rgb(146, 131, 116),
            muted: Color::Rgb(80, 73, 69),
            accent: Color::Rgb(131, 165, 152),
            secondary: Color::Rgb(211, 134, 155),
            success: Color::Rgb(184, 187, 38),
            warning: Color::Rgb(250, 189, 47),
            error: Color::Rgb(251, 73, 52),
            info: Color::Rgb(142, 192, 124),
            surface: Color::Rgb(50, 48, 47),
        }
    }

    pub fn one_half_light() -> Self {
        Self {
            bg: Color::Rgb(250, 250, 250),
            fg: Color::Rgb(56, 58, 66),
            dim: Color::Rgb(160, 161, 167),
            muted: Color::Rgb(210, 210, 210),
            accent: Color::Rgb(1, 132, 188),
            secondary: Color::Rgb(166, 38, 164),
            success: Color::Rgb(80, 161, 79),
            warning: Color::Rgb(193, 132, 1),
            error: Color::Rgb(228, 86, 73),
            info: Color::Rgb(1, 132, 188),
            surface: Color::Rgb(237, 237, 237),
        }
    }

    pub fn github_light() -> Self {
        Self {
            bg: Color::Rgb(255, 255, 255),
            fg: Color::Rgb(36, 41, 47),
            dim: Color::Rgb(101, 109, 118),
            muted: Color::Rgb(216, 222, 228),
            accent: Color::Rgb(9, 105, 218),
            secondary: Color::Rgb(130, 80, 223),
            success: Color::Rgb(26, 127, 55),
            warning: Color::Rgb(154, 103, 0),
            error: Color::Rgb(207, 34, 46),
            info: Color::Rgb(9, 105, 218),
            surface: Color::Rgb(246, 248, 250),
        }
    }
}

/// All available theme names, dark themes first then light themes.
pub const THEME_NAMES: &[&str] = &[
    "dark",
    "one-dark",
    "catppuccin-mocha",
    "dracula",
    "tokyo-night",
    "vscode-dark",
    "nord",
    "gruvbox-dark",
    "light",
    "one-half-light",
    "github-light",
];

/// Resolve a theme name to a Theme instance.
pub fn resolve_theme(name: &str) -> Option<Theme> {
    match name {
        "dark" => Some(Theme::dark()),
        "light" => Some(Theme::light()),
        "one-dark" => Some(Theme::from_palette(&ThemeColors::one_dark())),
        "catppuccin-mocha" => Some(Theme::from_palette(&ThemeColors::catppuccin_mocha())),
        "dracula" => Some(Theme::from_palette(&ThemeColors::dracula())),
        "tokyo-night" => Some(Theme::from_palette(&ThemeColors::tokyo_night())),
        "vscode-dark" => Some(Theme::from_palette(&ThemeColors::vscode_dark())),
        "nord" => Some(Theme::from_palette(&ThemeColors::nord())),
        "gruvbox-dark" => Some(Theme::from_palette(&ThemeColors::gruvbox_dark())),
        "one-half-light" => Some(Theme::from_palette(&ThemeColors::one_half_light())),
        "github-light" => Some(Theme::from_palette(&ThemeColors::github_light())),
        _ => None,
    }
}

/// Return the background fill color for RGB themes, or None for ANSI fallback themes.
/// ANSI themes (dark, light) defer to the terminal's native background color.
/// RGB themes own their background and must paint it explicitly.
pub fn theme_bg_color(name: &str) -> Option<Color> {
    match name {
        "dark" | "light" => None,
        "one-dark" => Some(ThemeColors::one_dark().bg),
        "catppuccin-mocha" => Some(ThemeColors::catppuccin_mocha().bg),
        "dracula" => Some(ThemeColors::dracula().bg),
        "tokyo-night" => Some(ThemeColors::tokyo_night().bg),
        "vscode-dark" => Some(ThemeColors::vscode_dark().bg),
        "nord" => Some(ThemeColors::nord().bg),
        "gruvbox-dark" => Some(ThemeColors::gruvbox_dark().bg),
        "one-half-light" => Some(ThemeColors::one_half_light().bg),
        "github-light" => Some(ThemeColors::github_light().bg),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_named_themes_resolve() {
        for name in THEME_NAMES {
            assert!(
                resolve_theme(name).is_some(),
                "theme '{}' failed to resolve",
                name
            );
        }
    }

    #[test]
    fn unknown_theme_returns_none() {
        assert!(resolve_theme("nonexistent").is_none());
    }

    #[test]
    fn theme_names_not_empty() {
        assert!(!THEME_NAMES.is_empty());
    }

    #[test]
    fn ansi_themes_have_no_bg_fill() {
        assert!(theme_bg_color("dark").is_none());
        assert!(theme_bg_color("light").is_none());
    }

    #[test]
    fn rgb_themes_have_bg_fill() {
        for name in THEME_NAMES {
            if *name == "dark" || *name == "light" {
                continue;
            }
            assert!(
                theme_bg_color(name).is_some(),
                "RGB theme '{}' missing bg fill",
                name
            );
        }
    }

    #[test]
    fn unknown_theme_has_no_bg_fill() {
        assert!(theme_bg_color("nonexistent").is_none());
    }
}

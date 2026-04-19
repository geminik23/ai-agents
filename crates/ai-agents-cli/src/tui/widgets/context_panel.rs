//
// Context panel: current context key-value tree.
//

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use std::collections::HashMap;

use crate::tui::theme::Theme;

pub struct ContextPanelState {
    pub values: HashMap<String, serde_json::Value>,
}

pub fn render_context_panel(
    area: Rect,
    buf: &mut Buffer,
    state: &ContextPanelState,
    theme: &Theme,
) {
    let block = Block::default()
        .title(" Context ")
        .borders(Borders::ALL)
        .border_style(theme.panel_border)
        .title_style(theme.panel_title);
    let inner = block.inner(area);
    block.render(area, buf);

    if state.values.is_empty() {
        let text = Paragraph::new("  (no context)").style(theme.hint_style);
        text.render(inner, buf);
        return;
    }

    let mut lines: Vec<Line> = Vec::new();
    let mut keys: Vec<&String> = state.values.keys().collect();
    keys.sort();

    let mut runtime_count = 0u32;
    let mut builtin_count = 0u32;

    for key in &keys {
        let value = &state.values[*key];
        match value {
            serde_json::Value::Object(map) => {
                lines.push(Line::from(Span::styled(
                    format!("{}:", key),
                    theme.label_style,
                )));
                let mut subkeys: Vec<&String> = map.keys().collect();
                subkeys.sort();
                for sk in subkeys {
                    let sv = &map[sk];
                    let display = match sv {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    lines.push(Line::from(vec![
                        Span::styled(format!("  {}: ", sk), theme.label_style),
                        Span::styled(display, theme.value_style),
                    ]));
                }
                runtime_count += 1;
            }
            serde_json::Value::String(s) => {
                lines.push(Line::from(vec![
                    Span::styled(format!("{}: ", key), theme.label_style),
                    Span::styled(s.as_str(), theme.value_style),
                ]));
                builtin_count += 1;
            }
            other => {
                lines.push(Line::from(vec![
                    Span::styled(format!("{}: ", key), theme.label_style),
                    Span::styled(other.to_string(), theme.value_style),
                ]));
                builtin_count += 1;
            }
        }
        lines.push(Line::from(""));
    }

    lines.push(Line::from(Span::styled(
        format!("(rt: {} bi: {})", runtime_count, builtin_count),
        theme.hint_style,
    )));

    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    paragraph.render(inner, buf);
}

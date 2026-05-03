//
// Relationship panel: current actor relationship dimensions and events.
//

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::tui::theme::Theme;

pub struct RelationshipDimensionEntry {
    pub name: String,
    pub value: f64,
}

pub struct RelationshipEventEntry {
    pub description: String,
    pub significance: f64,
}

pub struct RelationshipPanelState {
    pub actor_id: Option<String>,
    pub configured: bool,
    pub dimensions: Vec<RelationshipDimensionEntry>,
    pub perceived_dimensions: Vec<RelationshipDimensionEntry>,
    pub mutual_dimensions: Vec<RelationshipDimensionEntry>,
    pub interaction_count: u32,
    pub events: Vec<RelationshipEventEntry>,
}

pub fn render_relationship_panel(
    area: Rect,
    buf: &mut Buffer,
    state: &RelationshipPanelState,
    theme: &Theme,
) {
    let title = match state.actor_id {
        Some(ref id) => format!(" Relationship [{}] ", id),
        None => " Relationship ".to_string(),
    };

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(theme.panel_border)
        .title_style(theme.panel_title);
    let inner = block.inner(area);
    block.render(area, buf);

    if !state.configured {
        let text = Paragraph::new("  Not configured.\n  Enable via\n  memory.relationships.")
            .style(theme.hint_style)
            .wrap(Wrap { trim: false });
        text.render(inner, buf);
        return;
    }

    if state.actor_id.is_none() {
        let text = Paragraph::new("  No actor set.\n  Use /actor set <id>.")
            .style(theme.hint_style)
            .wrap(Wrap { trim: false });
        text.render(inner, buf);
        return;
    }

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(
        "Agent to actor",
        theme.highlight_style,
    )));
    if state.dimensions.is_empty() {
        lines.push(Line::from(Span::styled("  none", theme.hint_style)));
    } else {
        for dim in &state.dimensions {
            lines.push(Line::from(vec![
                Span::styled(format!(" {}", dim.name), theme.label_style),
                Span::styled(format!(": {:.2}", dim.value), theme.value_style),
            ]));
        }
    }

    if !state.perceived_dimensions.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Perceived actor to agent",
            theme.highlight_style,
        )));
        for dim in &state.perceived_dimensions {
            lines.push(Line::from(vec![
                Span::styled(format!(" {}", dim.name), theme.label_style),
                Span::styled(format!(": {:.2}", dim.value), theme.value_style),
            ]));
        }
    }

    if !state.mutual_dimensions.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Mutual", theme.highlight_style)));
        for dim in &state.mutual_dimensions {
            lines.push(Line::from(vec![
                Span::styled(format!(" {}", dim.name), theme.label_style),
                Span::styled(format!(": {:.2}", dim.value), theme.value_style),
            ]));
        }
    }

    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("Interactions: ", theme.label_style),
        Span::styled(state.interaction_count.to_string(), theme.value_style),
    ]));

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled("Events", theme.highlight_style)));
    if state.events.is_empty() {
        lines.push(Line::from(Span::styled("  none", theme.hint_style)));
    } else {
        for event in &state.events {
            lines.push(Line::from(Span::styled(
                format!(" {:.2} {}", event.significance, event.description),
                theme.value_style,
            )));
        }
    }

    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    paragraph.render(inner, buf);
}

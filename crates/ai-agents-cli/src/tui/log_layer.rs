//
// Tracing layer that captures log events into the TUI message channel.
//

use chrono::Utc;
use tokio::sync::mpsc::UnboundedSender;
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::layer::{Context, Layer};

use super::event::AppMessage;

/// A log entry captured from tracing.
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level: Level,
    pub target: String,
    pub message: String,
    pub timestamp: chrono::DateTime<Utc>,
}

/// Tracing layer that sends log events through the AppMessage channel.
pub struct TuiLogLayer {
    tx: UnboundedSender<AppMessage>,
    min_level: Level,
}

impl TuiLogLayer {
    pub fn new(tx: UnboundedSender<AppMessage>, min_level: Level) -> Self {
        Self { tx, min_level }
    }
}

impl<S: Subscriber> Layer<S> for TuiLogLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let meta = event.metadata();
        if meta.level() > &self.min_level {
            return;
        }

        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);

        let entry = LogEntry {
            level: *meta.level(),
            target: meta.target().to_string(),
            message: visitor.message,
            timestamp: Utc::now(),
        };

        let _ = self.tx.send(AppMessage::Log(entry));
    }
}

/// Visitor that extracts the message field from a tracing event.
#[derive(Default)]
struct MessageVisitor {
    message: String,
}

impl Visit for MessageVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{:?}", value);
        }
    }
}

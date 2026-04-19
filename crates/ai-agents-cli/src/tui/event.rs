//
// TUI event types and async event loop.
//

use crossterm::event::KeyEvent;

use ai_agents::{AgentResponse, StreamChunk};

use super::log_layer::LogEntry;

/// Messages from all async sources into the main event loop.
#[derive(Debug)]
pub enum AppMessage {
    // Terminal events
    Key(KeyEvent),
    Resize(u16, u16),

    // Agent responses
    StreamChunk(StreamChunk),
    ChatResponse(Box<AgentResponse>),
    ChatError(String),

    // Background
    Tick,

    // Captured tracing output
    Log(LogEntry),
}

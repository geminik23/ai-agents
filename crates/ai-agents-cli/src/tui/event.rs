//
// TUI event types and async event loop.
//

use crossterm::event::KeyEvent;
use tokio::sync::oneshot;

use ai_agents::tools::{QuestionRequest, QuestionResponse};
use ai_agents::{AgentResponse, StreamChunk};

use super::log_layer::LogEntry;

/// Pending question request sent from `ask_user` to the TUI event loop.
pub struct PendingQuestion {
    /// Question payload shown in a modal.
    pub request: QuestionRequest,
    /// Response channel completed by modal selection.
    pub respond_to: oneshot::Sender<QuestionResponse>,
}

impl std::fmt::Debug for PendingQuestion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingQuestion")
            .field("request", &self.request)
            .finish_non_exhaustive()
    }
}

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

    // Host questions
    Question(PendingQuestion),

    // Background
    Tick,

    // Captured tracing output
    Log(LogEntry),
}

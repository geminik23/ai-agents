use ai_agents_core::{AgentError, Result};

use super::response::MainResponseDraft;
use crate::StreamChunk;

/// Fully buffered stream branch output before the branch commits.
#[derive(Debug, Clone)]
pub struct StreamingDraftResult {
    pub draft: MainResponseDraft,
    pub chunks: Vec<StreamChunk>,
}

impl StreamingDraftResult {
    pub fn new(draft: MainResponseDraft, chunks: Vec<StreamChunk>) -> Self {
        Self { draft, chunks }
    }
}

/// Bounded buffer used while streaming output waits for routing to resolve.
#[derive(Debug, Clone)]
pub struct StreamBranchBuffer {
    max_chunks: usize,
    chunks: Vec<StreamChunk>,
}

impl StreamBranchBuffer {
    pub fn new(max_chunks: usize) -> Result<Self> {
        if max_chunks == 0 {
            return Err(AgentError::InvalidSpec(
                "streaming.buffer_size must be greater than 0 for buffered routing".into(),
            ));
        }
        Ok(Self {
            max_chunks,
            chunks: Vec::new(),
        })
    }

    pub fn push(&mut self, chunk: StreamChunk) -> Result<()> {
        if self.chunks.len() >= self.max_chunks {
            return Err(AgentError::Other(format!(
                "stream buffer filled before routing resolved (limit {})",
                self.max_chunks
            )));
        }
        self.chunks.push(chunk);
        Ok(())
    }

    pub fn drain(self) -> Vec<StreamChunk> {
        self.chunks
    }

    pub fn is_full(&self) -> bool {
        self.chunks.len() >= self.max_chunks
    }

    pub fn discard(self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_rejects_zero_capacity() {
        assert!(StreamBranchBuffer::new(0).is_err());
    }

    #[test]
    fn buffer_reports_full_capacity() {
        let mut buffer = StreamBranchBuffer::new(1).unwrap();
        buffer.push(StreamChunk::content("a")).unwrap();
        assert!(buffer.push(StreamChunk::content("b")).is_err());
    }
}

use serde::{Deserialize, Serialize};

/// Represents a chunk of streamed response from the agent
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamChunk {
    /// Text content from the LLM
    Content { text: String },
    /// A tool call is starting
    ToolCallStart { id: String, name: String },
    /// Incremental arguments for a tool call
    ToolCallDelta { id: String, arguments: String },
    /// A tool call has completed
    ToolCallEnd { id: String },
    /// Tool execution result
    ToolResult {
        id: String,
        name: String,
        output: String,
        success: bool,
    },
    /// State transition occurred
    StateTransition { from: Option<String>, to: String },
    /// Stream has completed
    Done {},
    /// An error occurred
    Error { message: String },
}

impl StreamChunk {
    pub fn content(text: impl Into<String>) -> Self {
        StreamChunk::Content { text: text.into() }
    }

    pub fn tool_start(id: impl Into<String>, name: impl Into<String>) -> Self {
        StreamChunk::ToolCallStart {
            id: id.into(),
            name: name.into(),
        }
    }

    pub fn tool_delta(id: impl Into<String>, arguments: impl Into<String>) -> Self {
        StreamChunk::ToolCallDelta {
            id: id.into(),
            arguments: arguments.into(),
        }
    }

    pub fn tool_end(id: impl Into<String>) -> Self {
        StreamChunk::ToolCallEnd { id: id.into() }
    }

    pub fn tool_result(
        id: impl Into<String>,
        name: impl Into<String>,
        output: impl Into<String>,
        success: bool,
    ) -> Self {
        StreamChunk::ToolResult {
            id: id.into(),
            name: name.into(),
            output: output.into(),
            success,
        }
    }

    pub fn state_transition(from: Option<String>, to: impl Into<String>) -> Self {
        StreamChunk::StateTransition {
            from,
            to: to.into(),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        StreamChunk::Error {
            message: message.into(),
        }
    }

    pub fn is_done(&self) -> bool {
        matches!(self, StreamChunk::Done {})
    }

    pub fn is_error(&self) -> bool {
        matches!(self, StreamChunk::Error { .. })
    }

    pub fn is_content(&self) -> bool {
        matches!(self, StreamChunk::Content { .. })
    }
}

/// Configuration for streaming behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Whether streaming is enabled
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Buffer size for streaming chunks
    #[serde(default = "default_buffer_size")]
    pub buffer_size: usize,
    /// Include tool call events in the stream
    #[serde(default = "default_true")]
    pub include_tool_events: bool,
    /// Include state transition events in the stream
    #[serde(default = "default_true")]
    pub include_state_events: bool,
}

fn default_true() -> bool {
    true
}

fn default_buffer_size() -> usize {
    32
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            buffer_size: default_buffer_size(),
            include_tool_events: true,
            include_state_events: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_chunk_constructors() {
        let content = StreamChunk::content("Hello");
        assert!(content.is_content());

        let tool_start = StreamChunk::tool_start("id1", "calculator");
        assert!(matches!(tool_start, StreamChunk::ToolCallStart { .. }));

        let tool_delta = StreamChunk::tool_delta("id1", r#"{"expr":"1+1"}"#);
        assert!(matches!(tool_delta, StreamChunk::ToolCallDelta { .. }));

        let tool_end = StreamChunk::tool_end("id1");
        assert!(matches!(tool_end, StreamChunk::ToolCallEnd { .. }));

        let done = StreamChunk::Done {};
        assert!(done.is_done());

        let error = StreamChunk::error("Something went wrong");
        assert!(error.is_error());
    }

    #[test]
    fn test_stream_chunk_serialization() {
        let content = StreamChunk::content("Hello");
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("content"));
        assert!(json.contains("Hello"));

        let tool_start = StreamChunk::tool_start("id1", "calculator");
        let json = serde_json::to_string(&tool_start).unwrap();
        assert!(json.contains("tool_call_start"));
        assert!(json.contains("calculator"));
    }

    #[test]
    fn test_streaming_config_defaults() {
        let config = StreamingConfig::default();
        assert!(config.enabled);
        assert_eq!(config.buffer_size, 32);
        assert!(config.include_tool_events);
        assert!(config.include_state_events);
    }

    #[test]
    fn test_streaming_config_deserialization() {
        let yaml = r#"
enabled: true
buffer_size: 64
include_tool_events: false
"#;
        let config: StreamingConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.enabled);
        assert_eq!(config.buffer_size, 64);
        assert!(!config.include_tool_events);
        assert!(config.include_state_events);
    }

    #[test]
    fn test_tool_result_chunk() {
        let result = StreamChunk::tool_result("id1", "calculator", "42", true);
        match result {
            StreamChunk::ToolResult {
                id,
                name,
                output,
                success,
            } => {
                assert_eq!(id, "id1");
                assert_eq!(name, "calculator");
                assert_eq!(output, "42");
                assert!(success);
            }
            _ => panic!("Expected ToolResult"),
        }
    }

    #[test]
    fn test_state_transition_chunk() {
        let transition = StreamChunk::state_transition(Some("greeting".to_string()), "support");
        match transition {
            StreamChunk::StateTransition { from, to } => {
                assert_eq!(from, Some("greeting".to_string()));
                assert_eq!(to, "support");
            }
            _ => panic!("Expected StateTransition"),
        }
    }

    #[test]
    fn test_stream_chunk_done_serialization() {
        let done = StreamChunk::Done {};
        let json = serde_json::to_string(&done).unwrap();
        assert!(json.contains("done"));
    }

    #[test]
    fn test_stream_chunk_error_serialization() {
        let error = StreamChunk::error("Test error");
        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains("error"));
        assert!(json.contains("Test error"));
    }

    #[test]
    fn test_stream_chunk_tool_result_serialization() {
        let result = StreamChunk::tool_result("id1", "calculator", "42", true);
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("tool_result"));
        assert!(json.contains("calculator"));
        assert!(json.contains("42"));
        assert!(json.contains("true"));
    }

    #[test]
    fn test_streaming_config_full_yaml() {
        let yaml = r#"
enabled: false
buffer_size: 128
include_tool_events: false
include_state_events: false
"#;
        let config: StreamingConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(!config.enabled);
        assert_eq!(config.buffer_size, 128);
        assert!(!config.include_tool_events);
        assert!(!config.include_state_events);
    }

    #[test]
    fn test_stream_chunk_deserialization() {
        let json = r#"{"type":"content","text":"Hello"}"#;
        let chunk: StreamChunk = serde_json::from_str(json).unwrap();
        assert!(chunk.is_content());

        let json = r#"{"type":"done"}"#;
        let chunk: StreamChunk = serde_json::from_str(json).unwrap();
        assert!(chunk.is_done());

        let json = r#"{"type":"error","message":"fail"}"#;
        let chunk: StreamChunk = serde_json::from_str(json).unwrap();
        assert!(chunk.is_error());
    }

    #[test]
    fn test_stream_chunk_tool_events() {
        let start = StreamChunk::tool_start("tool-1", "http");
        let delta = StreamChunk::tool_delta("tool-1", r#"{"url":"test"}"#);
        let end = StreamChunk::tool_end("tool-1");

        match start {
            StreamChunk::ToolCallStart { id, name } => {
                assert_eq!(id, "tool-1");
                assert_eq!(name, "http");
            }
            _ => panic!("Expected ToolCallStart"),
        }

        match delta {
            StreamChunk::ToolCallDelta { id, arguments } => {
                assert_eq!(id, "tool-1");
                assert!(arguments.contains("url"));
            }
            _ => panic!("Expected ToolCallDelta"),
        }

        match end {
            StreamChunk::ToolCallEnd { id } => {
                assert_eq!(id, "tool-1");
            }
            _ => panic!("Expected ToolCallEnd"),
        }
    }
}

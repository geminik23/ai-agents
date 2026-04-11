//! Multi-agent orchestration functions.
//! Coordination patterns built on top of AgentRegistry primitives.

pub mod aggregation;
pub mod concurrent;
pub mod context;
pub mod group_chat;
pub mod handoff;
pub mod pipeline;
pub mod route;
pub mod tools;
pub mod types;

pub use concurrent::concurrent;
pub use group_chat::group_chat;
pub use handoff::handoff;
pub use pipeline::pipeline;
pub use route::route;
pub use types::*;

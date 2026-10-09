//! Shared autonomy schema; this module contains no executable task controller.

mod config;
mod gate;
mod storage;

pub use config::*;
pub use gate::*;
pub use storage::*;

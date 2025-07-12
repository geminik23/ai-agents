//! Error recovery system with retry logic, backoff strategies, and graceful degradation

mod config;
mod error;
mod manager;

pub use config::*;
pub use error::*;
pub use manager::*;

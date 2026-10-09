//! Shared autonomy schema; this module contains no executable task controller.

mod config;
mod execution;
pub use execution::{
    InvocationAdmission, ProviderCostBound, ProviderCostSettlement, ProviderRequest,
    ToolWriteFootprint, validate_cost_settlement,
};
mod gate;
mod storage;

pub use config::*;
pub use gate::*;
pub use storage::*;

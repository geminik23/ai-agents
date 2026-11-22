//! Clock samples for retained human approvals, including executable plan adoption.
//! A scoped test clock leaves parent invocation budgets and journal timestamps on their own clocks.

use chrono::{DateTime, Utc};
use std::time::Instant;

#[cfg(test)]
tokio::task_local! {
    pub(super) static MANUAL: std::sync::Arc<super::invocation_clock::ManualClock>;
}

// The live receipt carries the original monotonic bound across suspension.
pub(crate) fn now() -> Instant {
    #[cfg(test)]
    if let Ok(now) = MANUAL.try_with(|clock| clock.now()) {
        return now;
    }
    Instant::now()
}

// The persisted UTC expiry is checked alongside the live bound.
pub(crate) fn utc_now() -> DateTime<Utc> {
    #[cfg(test)]
    if let Ok(now) = MANUAL.try_with(|clock| clock.utc_now()) {
        return now;
    }
    Utc::now()
}

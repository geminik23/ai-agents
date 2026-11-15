//! Read-only OS synchronization evidence for conservative cross-process UTC bounds.

use ai_agents_core::{AgentError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant, SystemTime};

/// An uncertainty interval in the same UTC domain used by task storage.
/// Its lower end is used when capturing a future cut and its upper end when admitting resumed work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskUtcEvidence {
    pub earliest: DateTime<Utc>,
    pub latest: DateTime<Utc>,
}

impl TaskUtcEvidence {
    /// Validates the measured error bound without silently capping or replacing missing evidence.
    fn measured(now: DateTime<Utc>, error_micros: i64, sampling: Duration) -> Result<Self> {
        if error_micros < 0 {
            return Err(unavailable());
        }
        let error = chrono::Duration::microseconds(error_micros)
            .checked_add(&chrono::Duration::from_std(sampling).map_err(|_| unavailable())?)
            .ok_or_else(unavailable)?;
        Ok(Self {
            earliest: now.checked_sub_signed(error).ok_or_else(unavailable)?,
            latest: now.checked_add_signed(error).ok_or_else(unavailable)?,
        })
    }

    /// Rejects an invalid interval or demonstrable rollback against a previously certified observation.
    /// Overlapping intervals do not prove elapsed time and must not replenish an original allowance.
    pub fn validate_after(&self, previous: &Self) -> Result<()> {
        if self.earliest > self.latest || previous.earliest > previous.latest {
            return Err(unavailable());
        }
        if self.latest < previous.earliest {
            return Err(super::TaskRunStorageError::ClockDiscontinuity.into());
        }
        Ok(())
    }

    // A previously certified larger upper bound is still conservative when fresh uncertainty shrinks.
    // Fresh evidence remains mandatory and a definitely backward interval is never clamped into acceptance.
    pub(crate) fn retain_upper_after(mut self, previous: &Self) -> Result<Self> {
        self.validate_after(previous)?;
        self.latest = self.latest.max(previous.latest);
        Ok(self)
    }
}

/// Reads synchronization status and error bounds without changing OS time or starting a time service.
/// The OS is the selected trust authority; this does not authenticate a hypervisor snapshot or a remote machine.
pub struct SystemTaskClock;

impl SystemTaskClock {
    /// Brackets UTC sampling with synchronized kernel observations and includes sampling time in the error interval.
    pub fn sample() -> Result<TaskUtcEvidence> {
        let start = Instant::now();
        let wall_start = SystemTime::now();
        let before = synchronized_error()?;
        let now = Utc::now();
        let after = synchronized_error()?;
        // Instant excludes system suspension on some platforms; a forward wall span also bounds that sampling gap.
        let wall_span = SystemTime::now()
            .duration_since(wall_start)
            .map_err(|_| super::TaskRunStorageError::ClockDiscontinuity)?;
        TaskUtcEvidence::measured(now, before.max(after), start.elapsed().max(wall_span))
    }
}

// Errors describe the missing authority without presenting an unchecked OS timestamp as a certified fallback.
fn unavailable() -> AgentError {
    AgentError::Config(
        "task restoration requires synchronized OS time with a valid error bound".into(),
    )
}

// A zero modes mask only reads the kernel's clock state; it never requests an adjustment or needs elevated privileges.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn synchronized_error() -> Result<i64> {
    // SAFETY: timex is a C data structure whose zero modes mask requests a read-only query.
    let mut reading: libc::timex = unsafe { std::mem::zeroed() };
    #[cfg(target_os = "linux")]
    // SAFETY: reading is initialized, correctly aligned and lives for the complete system call.
    let state = unsafe { libc::adjtimex(&mut reading) };
    #[cfg(target_os = "macos")]
    // SAFETY: reading is initialized, correctly aligned and lives for the complete system call.
    let state = unsafe { libc::ntp_adjtime(&mut reading) };
    checked_error(state, &reading)
}

// Missing synchronization or kernel clock errors cannot authorize a raw UTC fallback.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn checked_error(state: libc::c_int, reading: &libc::timex) -> Result<i64> {
    // Leap adjustments are not treated as ordinary continuity until the kernel reports synchronized steady time again.
    if state != libc::TIME_OK || reading.status & (libc::STA_UNSYNC | libc::STA_CLOCKERR) != 0 {
        return Err(unavailable());
    }
    // c_long is 32 bits on some Linux targets and 64 bits on the current native targets.
    #[allow(clippy::unnecessary_cast)]
    let error = reading.maxerror as i64;
    if error < 0 {
        return Err(unavailable());
    }
    Ok(error)
}

// Platforms without an implemented evidence reader retain explicit rejection rather than a weaker clock contract.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn synchronized_error() -> Result<i64> {
    Err(unavailable())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Controlled kernel observations test rejection without changing the host's actual synchronization state.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn missing_kernel_evidence_never_falls_back_to_wall_time() {
        // SAFETY: this initialized C value is test data and is never passed to a mutating system call.
        let mut reading: libc::timex = unsafe { std::mem::zeroed() };
        reading.maxerror = 200;
        assert_eq!(checked_error(libc::TIME_OK, &reading).unwrap(), 200);
        for status in [libc::STA_UNSYNC, libc::STA_CLOCKERR] {
            reading.status = status;
            assert!(checked_error(libc::TIME_OK, &reading).is_err());
        }
        reading.status = 0;
        for state in [-1, libc::TIME_ERROR, libc::TIME_INS, libc::TIME_DEL] {
            assert!(checked_error(state, &reading).is_err());
        }
        reading.maxerror = -1;
        assert!(checked_error(libc::TIME_OK, &reading).is_err());
    }

    // Better synchronization may lower the reported upper error bound without moving real time backward.
    #[test]
    fn shrinking_uncertainty_keeps_conservative_admission_without_hiding_rollback() {
        let now = DateTime::from_timestamp(1_800_000_000, 0).unwrap();
        let previous = TaskUtcEvidence::measured(now, 1000, Duration::ZERO).unwrap();
        let fresh = TaskUtcEvidence::measured(
            now + chrono::Duration::microseconds(100),
            100,
            Duration::ZERO,
        )
        .unwrap();
        let retained = fresh.clone().retain_upper_after(&previous).unwrap();
        assert_eq!(retained.earliest, fresh.earliest);
        assert_eq!(retained.latest, previous.latest);
        let old =
            TaskUtcEvidence::measured(now - chrono::Duration::seconds(1), 100, Duration::ZERO)
                .unwrap();
        assert!(old.retain_upper_after(&retained).is_err());
    }

    // Both ends matter: capture uses the earlier cut while resume spends the larger possible elapsed interval.
    #[test]
    fn uncertainty_cannot_extend_a_captured_deadline() {
        let now = DateTime::from_timestamp(1_800_000_000, 0).unwrap();
        let source = TaskUtcEvidence::measured(now, 200, Duration::from_micros(20)).unwrap();
        let resumed = TaskUtcEvidence::measured(
            now + chrono::Duration::seconds(1),
            300,
            Duration::from_micros(10),
        )
        .unwrap();
        resumed.validate_after(&source).unwrap();
        let remaining = source.earliest + chrono::Duration::seconds(2) - resumed.latest;
        assert_eq!(remaining, chrono::Duration::microseconds(999_470));
        assert!(remaining < chrono::Duration::seconds(1));
    }

    // Invalid or reversed evidence is rejected; there is no fallback to a clamped stored timestamp.
    #[test]
    fn invalid_error_and_rollback_remain_errors() {
        let now = DateTime::from_timestamp(1_800_000_000, 0).unwrap();
        assert!(TaskUtcEvidence::measured(now, -1, Duration::ZERO).is_err());
        let source = TaskUtcEvidence::measured(now, 100, Duration::ZERO).unwrap();
        let old =
            TaskUtcEvidence::measured(now - chrono::Duration::seconds(1), 100, Duration::ZERO)
                .unwrap();
        assert!(matches!(
            old.validate_after(&source),
            Err(AgentError::TaskRunStorage(
                super::super::TaskRunStorageError::ClockDiscontinuity
            ))
        ));
        let invalid = TaskUtcEvidence {
            earliest: source.latest,
            latest: source.earliest,
        };
        assert!(invalid.validate_after(&source).is_err());
    }
}

//! Samples and waits on local tool/message deadlines through the same clock.
//! Production uses native wall/monotonic time and Tokio timers; only unit tests can install a task-local manual clock.
//! Journal timestamps, aggregate run accounting, provider I/O and reconstruction clocks retain their own contracts.

use chrono::{DateTime, Utc};
use std::{
    future::Future,
    time::{Duration, Instant},
};

/// Samples the monotonic clock used to create and recheck an invocation's original bound.
pub(crate) fn now() -> Instant {
    #[cfg(test)]
    if let Ok(clock) = MANUAL.try_with(Clone::clone) {
        return clock.now();
    }
    Instant::now()
}

/// Samples the wall clock paired with invocation deadlines, without changing backend metadata time.
pub(crate) fn utc_now() -> DateTime<Utc> {
    #[cfg(test)]
    if let Ok(clock) = MANUAL.try_with(Clone::clone) {
        return clock.utc_now();
    }
    Utc::now()
}

/// Waits on the same clock as the retained deadline; storage readiness never advances manual test time.
pub(crate) async fn sleep_until(deadline: Instant) {
    #[cfg(test)]
    if let Ok(clock) = MANUAL.try_with(Clone::clone) {
        loop {
            let changed = clock.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if clock.now() >= deadline {
                return;
            }
            changed.await;
        }
    }
    tokio::time::sleep_until(deadline.into()).await;
}

/// Preserves Tokio's native timeout behavior unless a unit test explicitly controls this invocation clock.
pub(crate) async fn timeout<F: Future>(duration: Duration, work: F) -> Option<F::Output> {
    #[cfg(test)]
    if MANUAL.try_with(|_| ()).is_ok() {
        let deadline = now().checked_add(duration).expect("test deadline overflow");
        tokio::select! {
            biased;
            result = work => return Some(result),
            _ = sleep_until(deadline) => return None,
        }
    }
    tokio::time::timeout(duration, work).await.ok()
}

#[cfg(test)]
tokio::task_local! { pub(super) static MANUAL: std::sync::Arc<ManualClock>; }

#[cfg(test)]
pub(crate) struct ManualClock {
    monotonic: Instant,
    utc: DateTime<Utc>,
    elapsed: parking_lot::Mutex<Duration>,
    changed: tokio::sync::Notify,
}

#[cfg(test)]
impl ManualClock {
    /// A distinct epoch exposes accidental native-clock reads; unsupported past instants fall back to the native epoch.
    pub(crate) fn new(epoch_offset: Duration) -> std::sync::Arc<Self> {
        let native = Instant::now();
        let monotonic = native.checked_sub(epoch_offset).unwrap_or(native);
        let utc =
            Utc::now() - chrono::Duration::from_std(native.duration_since(monotonic)).unwrap();
        std::sync::Arc::new(Self {
            monotonic,
            utc,
            elapsed: Default::default(),
            changed: Default::default(),
        })
    }

    /// Each scoped future owns its clock selection; concurrent tests and ordinary consumers remain independent.
    pub(crate) async fn scope<F: Future>(self: &std::sync::Arc<Self>, work: F) -> F::Output {
        MANUAL
            .scope(
                self.clone(),
                ai_agents_llm::InvocationBounds::scope_test_clock(now, work),
            )
            .await
    }

    /// Returns the current manual instant without consulting elapsed host time.
    pub(crate) fn now(&self) -> Instant {
        self.monotonic + *self.elapsed.lock()
    }

    /// Projects the same controlled elapsed time onto UTC for wall-clock deadline checks.
    pub(crate) fn utc_now(&self) -> DateTime<Utc> {
        self.utc + chrono::Duration::from_std(*self.elapsed.lock()).unwrap()
    }

    /// Advancing is monotonic and wakes existing waits without changing their original deadlines.
    pub(crate) fn advance_to(&self, instant: Instant) {
        let mut elapsed = self.elapsed.lock();
        let next = instant
            .checked_duration_since(self.monotonic)
            .expect("clock before epoch");
        assert!(next >= *elapsed, "manual clock moved backwards");
        *elapsed = next;
        drop(elapsed);
        self.changed.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent scoped clocks must neither wake a sibling nor leak a frozen epoch into ordinary callers.
    #[tokio::test]
    async fn manual_deadline_waits_are_scoped_and_keep_their_original_cut() {
        let real_before = Instant::now();
        let first = ManualClock::new(Duration::from_secs(4));
        let second = ManualClock::new(Duration::from_secs(8));
        let first_cut = first.now() + Duration::from_secs(2);
        let second_cut = second.now() + Duration::from_secs(2);
        let mut a = std::pin::pin!(first.scope(sleep_until(first_cut)));
        let mut b = std::pin::pin!(second.scope(sleep_until(second_cut)));
        assert!(futures::poll!(a.as_mut()).is_pending());
        assert!(futures::poll!(b.as_mut()).is_pending());
        first.advance_to(first_cut - Duration::from_millis(1));
        assert!(futures::poll!(a.as_mut()).is_pending());
        first.advance_to(first_cut);
        assert!(futures::poll!(a.as_mut()).is_ready());
        assert!(futures::poll!(b.as_mut()).is_pending());
        second.advance_to(second_cut);
        assert!(futures::poll!(b.as_mut()).is_ready());
        assert!(now() >= real_before);
    }

    // Manual wall and monotonic observations advance together; native timeout behavior remains the fallback.
    #[tokio::test]
    async fn manual_timeout_and_wall_clock_share_one_advance() {
        let clock = ManualClock::new(Duration::from_secs(4));
        clock
            .scope(async {
                let initial = utc_now();
                let mut waiting = std::pin::pin!(timeout(
                    Duration::from_secs(2),
                    std::future::pending::<()>()
                ));
                assert!(futures::poll!(waiting.as_mut()).is_pending());
                clock.advance_to(clock.now() + Duration::from_secs(2));
                assert_eq!(utc_now() - initial, chrono::Duration::seconds(2));
                assert_eq!(waiting.await, None);
            })
            .await;
        assert_eq!(timeout(Duration::ZERO, async { 7 }).await, Some(7));
        assert_eq!(
            timeout(Duration::ZERO, std::future::pending::<()>()).await,
            None
        );
    }
}

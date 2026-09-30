//! Hybrid Logical Clock (HLC) for causally consistent timestamps.
//!
//! An HLC combines wall-clock time with a logical counter to guarantee
//! monotonically increasing timestamps even under clock skew. This is
//! critical for Last-Write-Wins (LWW) conflict resolution in replication.
//!
//! # Encoding
//!
//! Packed into a single `u64` for zero-cost serialization:
//! - Upper 48 bits: physical time in milliseconds since Unix epoch
//! - Lower 16 bits: logical counter (0..65535)
//!
//! This encoding means plain `u64` comparison produces the correct total
//! order (physical time first, then logical counter), and existing
//! serialized wall-clock timestamps (with counter = 0) remain valid.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(target_arch = "wasm32")]
use web_time::{SystemTime, UNIX_EPOCH};

const LOGICAL_BITS: u32 = 16;
const LOGICAL_MASK: u64 = (1 << LOGICAL_BITS) - 1; // 0xFFFF
const MAX_PHYSICAL_MS: u64 = u64::MAX >> LOGICAL_BITS;
fn checked_strictly_after(timestamp: HlcTimestamp) -> Option<HlcTimestamp> {
    timestamp
        .as_u64()
        .checked_add(1)
        .map(HlcTimestamp::from_u64)
}

fn checked_next_for_now(last: HlcTimestamp, physical_ms: u64) -> Option<HlcTimestamp> {
    if physical_ms > MAX_PHYSICAL_MS {
        return None;
    }
    if physical_ms > last.physical_ms() {
        Some(HlcTimestamp::new(physical_ms, 0))
    } else {
        // Incrementing the complete packed representation carries a full
        // logical counter into the next physical millisecond.
        checked_strictly_after(last)
    }
}

fn checked_next_for_update(
    last: HlcTimestamp,
    received: HlcTimestamp,
    physical_ms: u64,
) -> Option<HlcTimestamp> {
    if physical_ms > MAX_PHYSICAL_MS {
        return None;
    }
    let last_physical = last.physical_ms();
    let received_physical = received.physical_ms();
    let max_physical = physical_ms.max(last_physical).max(received_physical);

    if max_physical == physical_ms && physical_ms > last_physical && physical_ms > received_physical
    {
        Some(HlcTimestamp::new(physical_ms, 0))
    } else if max_physical == last_physical && last_physical == received_physical {
        checked_strictly_after(HlcTimestamp::new(
            max_physical,
            last.logical().max(received.logical()),
        ))
    } else if max_physical == last_physical {
        checked_strictly_after(last)
    } else {
        checked_strictly_after(received)
    }
}

/// There is no recoverable timestamp after the greatest packed `u64` value.
/// Continuing would let a caller catch an ordinary panic after mutating state
/// but before staging its ordered event, so mathematical exhaustion is a
/// deliberate fail-stop boundary.
#[cold]
#[inline(never)]
fn abort_terminal_exhaustion() -> ! {
    std::process::abort()
}

/// A Hybrid Logical Clock timestamp.
///
/// Encodes physical milliseconds (upper 48 bits) and a logical counter
/// (lower 16 bits) into a single `u64`. Implements `Ord` via the raw
/// value, giving a total order consistent with causality.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[repr(transparent)]
pub struct HlcTimestamp(u64);

impl HlcTimestamp {
    /// Creates an HLC timestamp from physical milliseconds and logical counter.
    #[inline]
    #[must_use]
    pub const fn new(physical_ms: u64, logical: u16) -> Self {
        Self((physical_ms << LOGICAL_BITS) | (logical as u64))
    }

    /// Creates an HLC timestamp from the current wall clock (logical = 0).
    ///
    /// # Process termination
    ///
    /// Aborts if wall-clock milliseconds no longer fit the timestamp's 48-bit
    /// physical component. [`HlcTimestamp::new`] remains a compatibility
    /// constructor and deliberately retains its historical truncating shape.
    #[must_use]
    pub fn now() -> Self {
        let physical_ms = wall_clock_ms();
        if physical_ms > MAX_PHYSICAL_MS {
            abort_terminal_exhaustion();
        }
        Self::new(physical_ms, 0)
    }

    /// Returns the physical time component in milliseconds since Unix epoch.
    #[inline]
    #[must_use]
    pub const fn physical_ms(&self) -> u64 {
        self.0 >> LOGICAL_BITS
    }

    /// Returns the logical counter component.
    #[inline]
    #[must_use]
    pub const fn logical(&self) -> u16 {
        (self.0 & LOGICAL_MASK) as u16
    }

    /// Returns the raw `u64` encoding.
    #[inline]
    #[must_use]
    pub const fn as_u64(&self) -> u64 {
        self.0
    }

    /// Creates an HLC timestamp from a raw `u64`.
    ///
    /// This is backward-compatible: plain wall-clock millisecond values
    /// (with logical counter = 0) decode correctly.
    #[inline]
    #[must_use]
    pub const fn from_u64(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns zero (the smallest possible timestamp).
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self(0)
    }
}

impl fmt::Debug for HlcTimestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HlcTimestamp({}.{})", self.physical_ms(), self.logical())
    }
}

impl fmt::Display for HlcTimestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.physical_ms(), self.logical())
    }
}

impl Default for HlcTimestamp {
    fn default() -> Self {
        Self::zero()
    }
}

impl From<u64> for HlcTimestamp {
    fn from(raw: u64) -> Self {
        Self::from_u64(raw)
    }
}

impl From<HlcTimestamp> for u64 {
    fn from(ts: HlcTimestamp) -> Self {
        ts.as_u64()
    }
}

/// A thread-safe Hybrid Logical Clock.
///
/// Each call to [`now()`](HlcClock::now) returns a timestamp strictly
/// greater than the previous one, even if the wall clock hasn't advanced
/// or has gone backward.
///
/// For multi-node replication, call [`update()`](HlcClock::update) when
/// receiving a remote timestamp to advance the local clock past the
/// remote's, preserving causal ordering across nodes.
pub struct HlcClock {
    last: AtomicU64,
}

impl HlcClock {
    /// Creates a new HLC clock initialized to the current wall-clock time.
    ///
    /// # Process termination
    ///
    /// Aborts if wall-clock milliseconds no longer fit the clock's 48-bit
    /// physical component.
    #[must_use]
    pub fn new() -> Self {
        let physical_ms = wall_clock_ms();
        if physical_ms > MAX_PHYSICAL_MS {
            abort_terminal_exhaustion();
        }
        Self {
            last: AtomicU64::new(HlcTimestamp::new(physical_ms, 0).as_u64()),
        }
    }

    /// Returns a monotonically increasing HLC timestamp.
    ///
    /// - If the wall clock has advanced past the last timestamp, resets
    ///   the logical counter to 0.
    /// - If the wall clock equals the last timestamp's physical time,
    ///   increments the logical counter.
    /// - If the wall clock went backward, keeps the last physical time
    ///   and increments the logical counter.
    ///
    /// Uses a CAS loop for lock-free thread safety.
    ///
    /// # Process termination
    ///
    /// Aborts the process if the complete packed timestamp space is exhausted
    /// at `u64::MAX`, where no strictly greater timestamp exists. This
    /// mathematical terminal state is not a recoverable clock outcome.
    pub fn now(&self) -> HlcTimestamp {
        self.now_at(wall_clock_ms())
    }

    fn now_at(&self, pt: u64) -> HlcTimestamp {
        loop {
            let last_raw = self.last.load(Ordering::Acquire);
            let last = HlcTimestamp::from_u64(last_raw);
            let Some(next) = checked_next_for_now(last, pt) else {
                abort_terminal_exhaustion()
            };

            if self
                .last
                .compare_exchange_weak(last_raw, next.as_u64(), Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return next;
            }
            // CAS failed: retry with updated `last`
        }
    }

    /// Merges a remote timestamp into the local clock.
    ///
    /// Advances the local clock to be strictly greater than both the
    /// current local time and the received remote timestamp. This
    /// preserves the causality guarantee: if event A happened-before
    /// event B, then `ts(A) < ts(B)`.
    ///
    /// # Process termination
    ///
    /// Aborts the process if advancing beyond the local or received timestamp
    /// would exceed the complete packed timestamp space at `u64::MAX`. This
    /// mathematical terminal state is not a recoverable clock outcome.
    pub fn update(&self, received: HlcTimestamp) -> HlcTimestamp {
        self.update_at(received, wall_clock_ms())
    }

    fn update_at(&self, received: HlcTimestamp, pt: u64) -> HlcTimestamp {
        loop {
            let last_raw = self.last.load(Ordering::Acquire);
            let last = HlcTimestamp::from_u64(last_raw);
            let Some(next) = checked_next_for_update(last, received, pt) else {
                abort_terminal_exhaustion()
            };

            if self
                .last
                .compare_exchange_weak(last_raw, next.as_u64(), Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return next;
            }
        }
    }

    /// Returns the last assigned timestamp without advancing the clock.
    #[must_use]
    pub fn peek(&self) -> HlcTimestamp {
        HlcTimestamp::from_u64(self.last.load(Ordering::Acquire))
    }
}

impl Default for HlcClock {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for HlcClock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HlcClock(last={})", self.peek())
    }
}

/// Reads the current wall-clock time in milliseconds since Unix epoch.
fn wall_clock_ms() -> u64 {
    let duration = SystemTime::now().duration_since(UNIX_EPOCH);
    // reason: wall-clock millis since epoch fits u64 for thousands of years
    #[allow(clippy::cast_possible_truncation)]
    duration.map_or(0, |d| d.as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::{Arc, Barrier};
    use std::thread;

    #[test]
    fn now_carries_logical_overflow_into_the_physical_component() {
        let clock = HlcClock {
            last: AtomicU64::new(HlcTimestamp::new(41, u16::MAX).as_u64()),
        };

        let next = clock.now_at(0);

        assert_eq!(next, HlcTimestamp::new(42, 0));
    }

    #[test]
    fn update_carries_logical_overflow_into_the_physical_component() {
        let clock = HlcClock {
            last: AtomicU64::new(HlcTimestamp::new(41, u16::MAX).as_u64()),
        };

        let next = clock.update_at(HlcTimestamp::new(41, u16::MAX), 0);

        assert_eq!(next, HlcTimestamp::new(42, 0));
    }

    #[test]
    fn now_checked_recurrence_reports_terminal_exhaustion() {
        assert_eq!(
            checked_next_for_now(HlcTimestamp::from_u64(u64::MAX), 0),
            None
        );
    }

    #[test]
    fn now_checked_recurrence_rejects_unencodable_physical_time() {
        assert_eq!(
            checked_next_for_now(HlcTimestamp::new(1, 0), MAX_PHYSICAL_MS + 1),
            None
        );
    }

    #[test]
    fn update_checked_recurrence_reports_terminal_exhaustion() {
        assert_eq!(
            checked_next_for_update(HlcTimestamp::new(1, 0), HlcTimestamp::from_u64(u64::MAX), 0,),
            None
        );
    }

    #[test]
    fn update_checked_recurrence_rejects_unencodable_physical_time() {
        assert_eq!(
            checked_next_for_update(
                HlcTimestamp::new(1, 0),
                HlcTimestamp::new(2, 0),
                MAX_PHYSICAL_MS + 1,
            ),
            None
        );
    }

    #[test]
    fn encoding_roundtrip() {
        let ts = HlcTimestamp::new(1_700_000_000_000, 42);
        assert_eq!(ts.physical_ms(), 1_700_000_000_000);
        assert_eq!(ts.logical(), 42);
        assert_eq!(HlcTimestamp::from_u64(ts.as_u64()), ts);
    }

    #[test]
    fn backward_compatible_with_plain_u64() {
        // A plain wall-clock ms value (logical = 0) roundtrips correctly
        let plain_ms: u64 = 1_700_000_000_000;
        let ts = HlcTimestamp::from_u64(plain_ms << LOGICAL_BITS);
        assert_eq!(ts.physical_ms(), plain_ms);
        assert_eq!(ts.logical(), 0);
    }

    #[test]
    fn ordering_physical_major() {
        let a = HlcTimestamp::new(100, 5);
        let b = HlcTimestamp::new(101, 0);
        assert!(a < b, "later physical time should be greater");
    }

    #[test]
    fn ordering_logical_minor() {
        let a = HlcTimestamp::new(100, 3);
        let b = HlcTimestamp::new(100, 4);
        assert!(
            a < b,
            "higher logical counter at same physical should be greater"
        );
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "requires wall clock (SystemTime::now) unavailable under Miri isolation"
    )]
    fn sequential_now_strictly_increasing() {
        let clock = HlcClock::new();
        let mut prev = clock.now();
        for _ in 0..1000 {
            let next = clock.now();
            assert!(
                next > prev,
                "HLC must be strictly increasing: {prev} then {next}"
            );
            prev = next;
        }
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "requires wall clock (SystemTime::now) unavailable under Miri isolation"
    )]
    fn concurrent_now_all_unique() {
        let clock = Arc::new(HlcClock::new());
        let num_threads = 8;
        let per_thread = 100;
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let clock = Arc::clone(&clock);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    let mut results = Vec::with_capacity(per_thread);
                    for _ in 0..per_thread {
                        results.push(clock.now().as_u64());
                    }
                    results
                })
            })
            .collect();

        let all: Vec<u64> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();

        let unique: HashSet<u64> = all.iter().copied().collect();
        assert_eq!(
            unique.len(),
            all.len(),
            "All concurrent HLC timestamps must be unique"
        );
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "requires wall clock (SystemTime::now) unavailable under Miri isolation"
    )]
    fn update_advances_past_remote() {
        let clock = HlcClock::new();

        // Simulate a remote timestamp far in the future
        let remote = HlcTimestamp::new(wall_clock_ms() + 10_000, 5);
        let merged = clock.update(remote);

        assert!(
            merged > remote,
            "Merged timestamp must be > remote: {merged} vs {remote}"
        );

        // Subsequent local now() must also be > remote
        let next = clock.now();
        assert!(
            next > remote,
            "Post-update now() must be > remote: {next} vs {remote}"
        );
    }

    #[test]
    fn zero_and_default() {
        assert_eq!(HlcTimestamp::zero().as_u64(), 0);
        assert_eq!(HlcTimestamp::default(), HlcTimestamp::zero());
    }

    #[test]
    fn display_format() {
        let ts = HlcTimestamp::new(1000, 42);
        assert_eq!(format!("{ts}"), "1000.42");
    }

    #[test]
    fn debug_format() {
        let ts = HlcTimestamp::new(500, 7);
        let dbg = format!("{ts:?}");
        assert!(dbg.contains("500"), "Debug should contain physical_ms");
        assert!(dbg.contains('7'), "Debug should contain logical");
    }

    #[test]
    fn from_u64_roundtrip() {
        let raw: u64 = 0xDEAD_BEEF_0042;
        let ts: HlcTimestamp = raw.into();
        let back: u64 = ts.into();
        assert_eq!(raw, back);
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "requires wall clock (SystemTime::now) unavailable under Miri isolation"
    )]
    fn update_local_wall_clock_is_newest() {
        // Scenario: both last and received are far in the past, so wall clock wins
        let clock = HlcClock {
            last: AtomicU64::new(HlcTimestamp::new(1, 0).as_u64()),
        };
        let remote = HlcTimestamp::new(2, 3);
        let merged = clock.update(remote);
        // Wall clock (current time) is much larger than 2ms, so it should dominate
        let now_ms = wall_clock_ms();
        assert!(
            merged.physical_ms() >= now_ms - 1,
            "When wall clock is newest, physical should match current time"
        );
        assert_eq!(
            merged.logical(),
            0,
            "When wall clock advances past both, logical resets to 0"
        );
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "requires wall clock (SystemTime::now) unavailable under Miri isolation"
    )]
    fn update_all_three_equal() {
        // Use a future physical time so the wall clock cannot overtake it
        // between setup and the update() call (avoids CI timing flakes).
        let pt = wall_clock_ms() + 60_000;
        let clock = HlcClock {
            last: AtomicU64::new(HlcTimestamp::new(pt, 5).as_u64()),
        };
        let remote = HlcTimestamp::new(pt, 10);
        let merged = clock.update(remote);
        // All three physical times equal (wall < pt, last = pt, remote = pt).
        // HLC picks max physical = pt, then max(last.logical=5, remote.logical=10) + 1 = 11
        assert_eq!(merged.physical_ms(), pt);
        assert_eq!(merged.logical(), 11);
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "requires wall clock (SystemTime::now) unavailable under Miri isolation"
    )]
    fn update_local_hlc_ahead() {
        // last is far in the future compared to both wall clock and remote
        let future_pt = wall_clock_ms() + 100_000;
        let clock = HlcClock {
            last: AtomicU64::new(HlcTimestamp::new(future_pt, 3).as_u64()),
        };
        let remote = HlcTimestamp::new(1, 0); // far in the past
        let merged = clock.update(remote);
        assert_eq!(
            merged.physical_ms(),
            future_pt,
            "Local HLC physical should dominate"
        );
        assert_eq!(merged.logical(), 4, "Should increment local logical by 1");
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "requires wall clock (SystemTime::now) unavailable under Miri isolation"
    )]
    fn update_remote_ahead() {
        // remote is far in the future, local is current
        let clock = HlcClock::new();
        let future_pt = wall_clock_ms() + 200_000;
        let remote = HlcTimestamp::new(future_pt, 7);
        let merged = clock.update(remote);
        assert_eq!(
            merged.physical_ms(),
            future_pt,
            "Remote physical should dominate"
        );
        assert_eq!(merged.logical(), 8, "Should increment remote logical by 1");
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "requires wall clock (SystemTime::now) unavailable under Miri isolation"
    )]
    fn clock_debug_format() {
        let clock = HlcClock::new();
        let dbg = format!("{clock:?}");
        assert!(dbg.starts_with("HlcClock(last="));
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "requires wall clock (SystemTime::now) unavailable under Miri isolation"
    )]
    fn peek_does_not_advance() {
        let clock = HlcClock::new();
        let a = clock.peek();
        let b = clock.peek();
        assert_eq!(a, b, "peek() should not advance the clock");
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "requires wall clock (SystemTime::now) unavailable under Miri isolation"
    )]
    fn default_clock() {
        // HlcClock::default() should work (it calls new())
        let _clock = HlcClock::default();
    }
}

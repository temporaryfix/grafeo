//! Test-only allocator traffic and one-shot failure witness shared by test binaries.
#![allow(
    unsafe_code,
    reason = "test-only transparent System allocation instrumentation"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub alloc: usize,
    pub zeroed: usize,
    pub realloc: usize,
    pub dealloc: usize,
}

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static COUNTS: Cell<Counts> = const { Cell::new(Counts { alloc: 0, zeroed: 0, realloc: 0, dealloc: 0 }) };
    static FAILURE: Cell<Failure> = const { Cell::new(Failure { layout: None, skip: 0, fired: false }) };
}

#[derive(Clone, Copy)]
struct Failure {
    layout: Option<(usize, usize)>,
    skip: usize,
    fired: bool,
}

fn refuse(size: usize, align: usize) -> bool {
    matches!(
        FAILURE.try_with(|state| {
            let mut failure = state.get();
            if failure.layout == Some((size, align)) {
                if failure.skip > 0 {
                    failure.skip -= 1;
                    state.set(failure);
                    return false;
                }
                state.set(Failure {
                    layout: None,
                    skip: 0,
                    fired: true,
                });
                true
            } else {
                false
            }
        }),
        Ok(true)
    )
}

/// Skips `skip` matching requests, then refuses one allocation on this thread.
/// The refused request must be made
/// through a fallible allocator API; an infallible allocation can abort.
/// Restores any enclosing witness even if the callback unwinds.
pub fn with_failure<T>(
    size: usize,
    align: usize,
    skip: usize,
    action: impl FnOnce() -> T,
) -> (T, bool) {
    struct Restore(Failure);
    impl Drop for Restore {
        fn drop(&mut self) {
            FAILURE.with(|state| state.set(self.0));
        }
    }
    let _restore = Restore(FAILURE.with(|state| {
        state.replace(Failure {
            layout: Some((size, align)),
            skip,
            fired: false,
        })
    }));
    let result = action();
    (result, FAILURE.with(|state| state.get().fired))
}

fn count(update: impl FnOnce(&mut Counts)) {
    // TLS destruction may make these unavailable. The allocator path must
    // neither allocate nor panic; no counting occurs after thread teardown.
    let _ = ENABLED.try_with(|enabled| {
        if enabled.get() {
            let _ = COUNTS.try_with(|counts| {
                let mut value = counts.get();
                update(&mut value);
                counts.set(value);
            });
        }
    });
}

pub struct CountingSystem;

// SAFETY: every pointer/layout is passed through to System unless a one-shot
// fault returns null, as permitted by GlobalAlloc. Failed realloc leaves the
// original allocation untouched. The TLS instrumentation cannot allocate or panic.
unsafe impl GlobalAlloc for CountingSystem {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(|n| n.alloc = n.alloc.wrapping_add(1));
        if refuse(layout.size(), layout.align()) {
            return std::ptr::null_mut();
        }
        // SAFETY: inherited GlobalAlloc caller preconditions.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(|n| n.zeroed = n.zeroed.wrapping_add(1));
        if refuse(layout.size(), layout.align()) {
            return std::ptr::null_mut();
        }
        // SAFETY: inherited GlobalAlloc caller preconditions.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count(|n| n.realloc = n.realloc.wrapping_add(1));
        if refuse(size, layout.align()) {
            return std::ptr::null_mut();
        }
        // SAFETY: inherited GlobalAlloc caller preconditions.
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        count(|n| n.dealloc = n.dealloc.wrapping_add(1));
        // SAFETY: inherited GlobalAlloc caller preconditions.
        unsafe { System.dealloc(ptr, layout) }
    }
}

pub fn start() {
    COUNTS.with(|count| count.set(Counts::default()));
    ENABLED.with(|enabled| enabled.set(true));
}

pub fn stop() -> Counts {
    ENABLED.with(|enabled| enabled.set(false));
    COUNTS.with(Cell::get)
}

#[global_allocator]
static ALLOCATOR: CountingSystem = CountingSystem;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_failure_is_one_shot_and_thread_local() {
        let ((other, skipped, refused, retry), fired) = with_failure(4099, 1, 1, || {
            let other = std::thread::spawn(|| Vec::<u8>::new().try_reserve_exact(4099)).join();
            let skipped = Vec::<u8>::new().try_reserve_exact(4099);
            let refused = Vec::<u8>::new().try_reserve_exact(4099);
            let retry = Vec::<u8>::new().try_reserve_exact(4099);
            (other, skipped, refused, retry)
        });
        assert!(matches!(other, Ok(Ok(()))));
        assert!(skipped.is_ok());
        assert!(refused.is_err());
        assert!(retry.is_ok());
        assert!(fired);
    }

    #[test]
    fn allocation_failure_keeps_realloc_source_and_restores_nested_scope() {
        let mut values = Vec::<u8>::with_capacity(13);
        values.push(42);
        let ((failed, nested), fired) = with_failure(27, 1, 0, || {
            let nested = with_failure(37, 1, 0, || Vec::<u8>::new().try_reserve_exact(37));
            (values.try_reserve_exact(26), nested)
        });
        assert!(failed.is_err() && fired);
        assert!(nested.0.is_err() && nested.1);
        assert_eq!(values, [42]);
        assert_eq!(values.capacity(), 13);
        assert!(values.try_reserve_exact(26).is_ok());
    }
}

//! Execution scheduler: one kernel, three runtimes.
//!
//! Parallelism is not a second engine. Native, `wasm32-unknown-unknown`, and
//! WASI 0.3 all call the same fill / leapfrog loops. This type selects how
//! those loops yield — it does **not** fork a WASI-only engine:
//!
//! - [`Scheduler::Immediate`]: run to completion. **Native default** (also
//!   used for tiny graphs on every target).
//! - [`Scheduler::Cooperative`]: yield every `yield_every` sources so a host
//!   can poll. Native uses [`std::thread::yield_now`]; a browser or WASI 0.3
//!   host supplies `on_yield` (`requestAnimationFrame` / `task.yield`). No
//!   WIT in this crate.
//!
//! Threads arrive when WASI 0.3 ships them; do not add rayon here. Native
//! already has [`crate::execution::parallel`] (morsel + rayon) behind the
//! `parallel` feature — that is the same kernel, a different scheduler.

/// How a long expand/fill loop should run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheduler {
    /// No yield. Native default and the fast path on every target.
    Immediate,
    /// Cooperative yield every `yield_every` sources.
    ///
    /// Same loops as [`Self::Immediate`]. Use on a shared native thread, in
    /// the browser, or under WASI 0.3 — pass the host yield in `on_yield`.
    Cooperative {
        /// Sources between yields. `0` is treated as `1`.
        yield_every: usize,
    },
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::Immediate
    }
}

impl Scheduler {
    /// Platform default yield body for [`Self::Cooperative`].
    ///
    /// Native: `std::thread::yield_now` so another thread (UI, other query)
    /// can run. `wasm32` / WASI: no-op — the host must pass `task.yield` or
    /// an event-loop yield as `on_yield`.
    #[inline]
    pub fn default_on_yield() {
        #[cfg(not(target_family = "wasm"))]
        std::thread::yield_now();
    }

    /// [`Self::for_each`] with [`Self::default_on_yield`].
    pub fn for_each_default<F>(&self, n: usize, step: F)
    where
        F: FnMut(usize),
    {
        self.for_each(n, step, Self::default_on_yield);
    }

    /// Runs `step` for `n` items, yielding when this scheduler says so.
    ///
    /// `on_yield` is caller-supplied. Native Cooperative typically uses
    /// [`Self::default_on_yield`]. WASI 0.3 maps the hook to `task.yield`.
    pub fn for_each<F, Y>(&self, n: usize, mut step: F, mut on_yield: Y)
    where
        F: FnMut(usize),
        Y: FnMut(),
    {
        match *self {
            Self::Immediate => {
                for i in 0..n {
                    step(i);
                }
            }
            Self::Cooperative { yield_every } => {
                let every = yield_every.max(1);
                for i in 0..n {
                    step(i);
                    if (i + 1) % every == 0 && i + 1 < n {
                        on_yield();
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immediate_never_yields() {
        let mut yields = 0u32;
        let mut seen = 0u32;
        Scheduler::Immediate.for_each(10, |_| seen += 1, || yields += 1);
        assert_eq!(seen, 10);
        assert_eq!(yields, 0);
    }

    #[test]
    fn cooperative_yields_between_batches() {
        let mut yields = 0u32;
        Scheduler::Cooperative { yield_every: 3 }.for_each(10, |_| {}, || yields += 1);
        assert_eq!(yields, 3);
    }

    #[test]
    fn default_is_immediate() {
        assert_eq!(Scheduler::default(), Scheduler::Immediate);
    }

    #[test]
    fn cooperative_default_yield_runs_on_native() {
        let mut seen = 0u32;
        Scheduler::Cooperative { yield_every: 2 }.for_each_default(5, |_| seen += 1);
        assert_eq!(seen, 5);
    }
}

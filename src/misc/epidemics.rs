//! SIR epidemics (`igraph_epidemics.h`).
//!
//! `igraph_sir_init` / `igraph_sir_destroy` are not wrapped: they are memory
//! plumbing. [`SirRun`] owns the four vectors of a run and frees them on drop.

use crate::{error::Result, ffi::*, igraph_call};
use std::mem::MaybeUninit;

/// The outcome of one run of the SIR epidemic model (the owned counterpart of
/// `igraph_sir_t`), as returned by [`Graph::sir`](crate::Graph::sir).
///
/// The four vectors have the same length: entry `k` describes the state of
/// the population right after the `k`-th event (an infection or a recovery).
/// Entry `0` is the initial state, at time `0`, with a single infected
/// individual; the last entry is the moment the last infected individual
/// recovers. At every step `susceptible + infected + recovered` equals the
/// number of vertices of the graph.
#[derive(Debug, Clone, PartialEq)]
pub struct SirRun {
    /// The (continuous) times of the events, starting at `0.0`, non-decreasing.
    pub times: Vec<f64>,
    /// Number of susceptible individuals after each event.
    pub susceptible: Vec<i64>,
    /// Number of infected individuals after each event.
    pub infected: Vec<i64>,
    /// Number of recovered individuals after each event.
    pub recovered: Vec<i64>,
}

impl SirRun {
    /// Number of recorded states (events plus the initial state).
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Whether no state was recorded (never the case for runs produced by igraph).
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// The time at which the epidemic died out (the time of the last event).
    pub fn duration(&self) -> f64 {
        self.times.last().copied().unwrap_or(0.0)
    }

    /// The *final size* of the epidemic: how many individuals were ever
    /// infected (they have all recovered at the end).
    pub fn final_size(&self) -> i64 {
        self.recovered.last().copied().unwrap_or(0)
    }

    /// The largest number of simultaneously infected individuals.
    pub fn peak_infected(&self) -> i64 {
        self.infected.iter().copied().max().unwrap_or(0)
    }

    /// Iterates over `(time, susceptible, infected, recovered)` tuples.
    pub fn states(&self) -> impl Iterator<Item = (f64, i64, i64, i64)> + '_ {
        (0..self.len()).map(move |k| {
            (
                self.times[k],
                self.susceptible[k],
                self.infected[k],
                self.recovered[k],
            )
        })
    }
}

impl From<igraph_sir_t> for SirRun {
    fn from(sir: igraph_sir_t) -> Self {
        // The fields are owned igraph vectors: they are freed when `sir` drops.
        Self {
            times: sir.times.to_vec(),
            susceptible: sir.no_s.to_vec(),
            infected: sir.no_i.to_vec(),
            recovered: sir.no_r.to_vec(),
        }
    }
}

/// Owns an `igraph_vector_ptr_t` filled by `igraph_sir` with heap allocated
/// `igraph_sir_t` items, and frees everything (items included) on drop.
struct SirList(igraph_vector_ptr_t);

impl SirList {
    fn new() -> Self {
        crate::error::ensure_init();
        let mut raw = MaybeUninit::<igraph_vector_ptr_t>::uninit();
        crate::error::check(unsafe { igraph_vector_ptr_init(raw.as_mut_ptr(), 0) })
            .expect("igraph failed to allocate a pointer vector");
        Self(unsafe { raw.assume_init() })
    }

    /// Moves the runs out of the list, leaving null pointers behind.
    fn take_runs(&mut self) -> Vec<SirRun> {
        let n = unsafe { igraph_vector_ptr_size(&self.0) } as usize;
        let mut runs = Vec::with_capacity(n);
        for k in 0..n {
            let slot = unsafe { self.0.stor_begin.add(k) };
            let item = unsafe { *slot } as *mut igraph_sir_t;
            if item.is_null() {
                continue;
            }
            // Take ownership of the four vectors, then free the C allocation
            // holding the struct itself (allocated with IGRAPH_CALLOC).
            let sir = unsafe { std::ptr::read(item) };
            unsafe {
                igraph_free(item.cast());
                *slot = std::ptr::null_mut();
            }
            runs.push(SirRun::from(sir));
        }
        runs
    }
}

impl Drop for SirList {
    fn drop(&mut self) {
        // Frees any run not moved out (e.g. never reached on error paths,
        // where igraph already nulled them), then the pointer vector itself.
        drop(self.take_runs());
        unsafe { igraph_vector_ptr_destroy(&mut self.0) };
    }
}

impl igraph_t {
    /// Runs `num_simulations` stochastic SIR (susceptible–infected–recovered)
    /// epidemics on the graph.
    ///
    /// Each individual (vertex) is susceptible, infected or recovered;
    /// recovered individuals are immune. A susceptible vertex with `n`
    /// infected neighbors becomes infected at rate `n * beta`, an infected one
    /// recovers at rate `gamma` (both are rates of exponential
    /// distributions, so the model runs in continuous time, as a Gillespie
    /// simulation). Every simulation starts with a single, uniformly chosen,
    /// infected vertex and stops when no infected vertex is left. It uses the
    /// thread's default random number generator: seed it with
    /// [`rng::seed`](crate::rng::seed) for reproducible runs.
    ///
    /// Edge directions are ignored (with a warning) for directed graphs, so a
    /// directed graph with a pair of opposite edges `u → v`, `v → u` counts
    /// as a multigraph and is rejected.
    /// Seeded runs are reproducible, also when several threads simulate at
    /// the same time: each thread has its own default generator.
    ///
    /// See also [`Graph::famous`](crate::Graph::famous) and the random graph models of
    /// [`crate::games`] for contact networks, and
    /// [`set_interruption_handler`](crate::misc::set_interruption_handler) to
    /// cancel long simulations.
    ///
    /// Binds [`igraph_sir`](https://igraph.org/c/html/latest/igraph-Processes.html#igraph_sir).
    /// Time complexity: `O(num_simulations * (|V| + |E| log |V|))`.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) when the
    /// graph has no vertices or is not simple, when `beta < 0`, `gamma <= 0`
    /// or `num_simulations == 0`. The computation can be stopped by an
    /// [interruption handler](crate::misc::set_interruption_handler)
    /// ([`ErrorKind::Interrupted`](crate::ErrorKind::Interrupted)).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(42)?;
    /// let ring = Graph::ring(10, false, false, true)?;
    /// let runs = ring.sir(2.0, 1.0, 5)?;
    /// assert_eq!(runs.len(), 5);
    /// for run in &runs {
    ///     for (_, s, i, r) in run.states() {
    ///         assert_eq!(s + i + r, 10); // the population is conserved
    ///     }
    ///     assert_eq!(*run.infected.last().unwrap(), 0); // the epidemic dies out
    /// }
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// Epidemics spread further on the karate club network when the
    /// infection rate grows:
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(1)?;
    /// let club = Graph::famous("Zachary")?;
    /// let mean_final_size = |beta: f64| -> igraph::Result<f64> {
    ///     let runs = club.sir(beta, 1.0, 300)?;
    ///     Ok(runs.iter().map(|r| r.final_size() as f64).sum::<f64>() / 300.0)
    /// };
    /// assert!(mean_final_size(0.05)? < mean_final_size(1.0)?);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn sir(&self, beta: f64, gamma: f64, num_simulations: usize) -> Result<Vec<SirRun>> {
        let mut list = SirList::new();
        igraph_call!(igraph_sir(
            self,
            beta,
            gamma,
            num_simulations as igraph_int_t,
            &mut list.0
        ))?;
        Ok(list.take_runs())
    }
}

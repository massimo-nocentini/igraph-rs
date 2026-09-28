//! Cliques, independent vertex sets and vertex/edge colorings
//! (`igraph_cliques.h`, `igraph_coloring.h`).
//!
//! A *clique* is a set of vertices that are pairwise adjacent; an
//! *independent vertex set* is a set of vertices no two of which are adjacent
//! (i.e. a clique of the complementer graph). A *proper vertex coloring*
//! assigns colors to vertices so that adjacent vertices always differ.
//! The clique and independent-set functions ignore edge directions (igraph
//! emits a warning for directed graphs), self-loops and multi-edges. The
//! coloring checks ignore self-loops too, but
//! [`is_bipartite_coloring`](Graph::is_bipartite_coloring) reports edge
//! directions and [`is_edge_coloring`](Graph::is_edge_coloring) treats
//! parallel edges as adjacent.
//!
//! Three families of sets are distinguished, both for cliques and for
//! independent sets:
//!
//! - **all** sets (every clique, including every subset of a clique);
//! - **maximal** sets, which cannot be extended by adding one more vertex;
//! - **largest** (maximum) sets, those of the largest possible size. The size
//!   of the largest clique is the *clique number* ω(G), the size of the
//!   largest independent set is the *independence number* α(G).
//!
//! # Size ranges
//!
//! The enumeration functions take the admissible sizes as a Rust range of
//! [`usize`] ([`RangeBounds`]): `..` means "any size", `3..` "at least 3",
//! `2..=4` "between 2 and 4", `..5` "fewer than 5". An empty range (e.g.
//! `4..=2` or `..1`) is rejected with an [`ErrorKind::InvalidValue`] error,
//! since no set could ever satisfy it. A lower bound larger than the number
//! of vertices is not an error: the result is simply empty. An optional
//! `max_results` stops the search after that many sets were found (`None`
//! means no limit, `Some(0)` returns nothing).
//!
//! # Nesting searches in callbacks
//!
//! The functions based on the Cliquer library
//! ([`cliques`](Graph::cliques), [`cliques_callback`](Graph::cliques_callback),
//! [`clique_size_hist`](Graph::clique_size_hist) and the weighted clique
//! functions) share per-thread state inside igraph: in igraph 1.0.0 and 1.0.1
//! the Cliquer wrapper keeps the options of the running search in a single
//! thread-local global that every search overwrites. Calling one of them from
//! inside a [`cliques_callback`](Graph::cliques_callback) closure is therefore
//! refused with an [`ErrorKind::Failure`] error instead of corrupting the
//! running search. All other functions (for instance
//! [`maximal_cliques`](Graph::maximal_cliques) or
//! [`clique_number`](Graph::clique_number)) may be used freely there, and
//! [`maximal_cliques_callback`](Graph::maximal_cliques_callback) closures may
//! call anything. The restriction is per thread: searches running on other
//! threads are independent. A nested call that fails only returns its `Err`
//! to the closure: the running search is not affected and can go on.
//!
//! # Example
//!
//! ```
//! use igraph::{cliques::ColoringGreedy, prelude::*};
//!
//! // Two triangles {0, 1, 2} and {2, 3, 4} sharing vertex 2, plus the edge 4-5.
//! let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4), (4, 5)], 6, false)
//!     .unwrap();
//!
//! assert_eq!(g.clique_number().unwrap(), 3);
//! let mut largest = g.largest_cliques().unwrap();
//! largest.iter_mut().for_each(|c| c.sort());
//! largest.sort();
//! assert_eq!(largest, vec![vec![0, 1, 2], vec![2, 3, 4]]);
//!
//! // Maximal cliques by size: none of size 1, one of size 2 (4-5), two triangles.
//! assert_eq!(g.maximal_cliques_hist(..).unwrap(), vec![0, 1, 2]);
//!
//! // Vertices 0, 3 and 5 are pairwise non-adjacent.
//! assert_eq!(g.independence_number().unwrap(), 3);
//!
//! // A proper coloring needs at least ω(G) = 3 colors.
//! let colors = g.vertex_coloring_greedy(ColoringGreedy::DSatur).unwrap();
//! assert!(g.is_vertex_coloring(&colors).unwrap());
//! assert_eq!(colors.iter().max(), Some(&2));
//!
//! // Zachary's karate club: its 45 triangles are its cliques of size 3, and
//! // its two largest cliques have 5 members.
//! let karate = Graph::famous("Zachary").unwrap();
//! assert_eq!(karate.clique_size_hist(3..=3).unwrap(), vec![0, 0, 45]);
//! assert_eq!(karate.count_triangles().unwrap(), 45.0);
//! assert_eq!(karate.clique_number().unwrap(), 5);
//! ```
//!
//! # Provided functionality
//!
//! | Rust method | C function | Computes |
//! |---|---|---|
//! | [`cliques`](Graph::cliques) | `igraph_cliques` | all cliques in a size range |
//! | [`cliques_callback`](Graph::cliques_callback) | `igraph_cliques_callback` | streams all cliques to a closure |
//! | [`clique_size_hist`](Graph::clique_size_hist) | `igraph_clique_size_hist` | number of cliques of each size |
//! | [`largest_cliques`](Graph::largest_cliques) | `igraph_largest_cliques` | the maximum cliques |
//! | [`clique_number`](Graph::clique_number) | `igraph_clique_number` | ω(G) |
//! | [`maximal_cliques`](Graph::maximal_cliques) | `igraph_maximal_cliques` | maximal cliques (Bron–Kerbosch) |
//! | [`maximal_cliques_callback`](Graph::maximal_cliques_callback) | `igraph_maximal_cliques_callback` | streams maximal cliques to a closure |
//! | [`maximal_cliques_count`](Graph::maximal_cliques_count) | `igraph_maximal_cliques_count` | number of maximal cliques |
//! | [`maximal_cliques_hist`](Graph::maximal_cliques_hist) | `igraph_maximal_cliques_hist` | maximal cliques by size |
//! | [`maximal_cliques_subset`](Graph::maximal_cliques_subset) | `igraph_maximal_cliques_subset` | maximal cliques started from some vertices |
//! | [`write_maximal_cliques`](Graph::write_maximal_cliques) | `igraph_maximal_cliques_file` | writes maximal cliques to an [`io::Write`] |
//! | [`weighted_cliques`](Graph::weighted_cliques) | `igraph_weighted_cliques` | (maximal) cliques in a weight range |
//! | [`largest_weighted_cliques`](Graph::largest_weighted_cliques) | `igraph_largest_weighted_cliques` | heaviest cliques |
//! | [`weighted_clique_number`](Graph::weighted_clique_number) | `igraph_weighted_clique_number` | weight of the heaviest clique |
//! | [`independent_vertex_sets`](Graph::independent_vertex_sets) | `igraph_independent_vertex_sets` | all independent sets in a size range |
//! | [`maximal_independent_vertex_sets`](Graph::maximal_independent_vertex_sets) | `igraph_maximal_independent_vertex_sets` | maximal independent sets |
//! | [`largest_independent_vertex_sets`](Graph::largest_independent_vertex_sets) | `igraph_largest_independent_vertex_sets` | maximum independent sets |
//! | [`independence_number`](Graph::independence_number) | `igraph_independence_number` | α(G) |
//! | [`vertex_coloring_greedy`](Graph::vertex_coloring_greedy) | `igraph_vertex_coloring_greedy` | a proper vertex coloring |
//! | [`is_vertex_coloring`](Graph::is_vertex_coloring) | `igraph_is_vertex_coloring` | checks a vertex coloring |
//! | [`is_bipartite_coloring`](Graph::is_bipartite_coloring) | `igraph_is_bipartite_coloring` | checks a 2-coloring, and edge orientation |
//! | [`is_edge_coloring`](Graph::is_edge_coloring) | `igraph_is_edge_coloring` | checks an edge coloring |
//!
//! Cliques and independent sets are returned as `Vec<Vec<VertexId>>`; the
//! order of the sets, and of the vertices inside each set, is unspecified
//! (sort them if you need a canonical form).
//!
//! # See also
//!
//! | Task | Elsewhere in the crate |
//! |---|---|
//! | test whether a *given* vertex set is a clique / independent set | [`Graph::is_clique`], [`Graph::is_independent_vertex_set`] (structural) |
//! | triangles (the 3-cliques) only | [`Graph::count_triangles`], [`Graph::list_triangles`] (isomorphism), [`Graph::transitivity_undirected`] (mixing) |
//! | switch between cliques and independent sets | [`Graph::complementer`] (operators): α(G) = ω(complement of G) |
//! | degeneracy, which bounds the cost of [`maximal_cliques`](Graph::maximal_cliques) | [`Graph::coreness`]: the degeneracy is the largest coreness |
//! | 2-colorings | [`Graph::is_bipartite`], [`Graph::bipartite_types`] (bipartite) |
//! | graph classes where χ(G) = ω(G) | [`Graph::is_perfect`], [`Graph::is_chordal`] (structural) |
//! | graphs with known ω, α and χ | [`Graph::full`], [`Graph::turan`], [`Graph::wheel`], [`Graph::famous`] (constructors), [`Graph::mycielskian`] (operators: triangle-free graphs of growing chromatic number) |

use crate::{
    constants::NeighborMode,
    error::{Error, ErrorKind, Result, catch_panic},
    ffi::*,
    graph::{Graph, VertexId},
    igraph_call,
    list::VectorIntList,
    vector::{Vector, VectorBool, VectorInt},
};
use std::{
    cell::Cell,
    ffi::{c_char, c_void},
    io,
    ops::{Bound, ControlFlow, RangeBounds},
};

crate::ffi_enum! {
    /// Vertex ordering heuristic of [`Graph::vertex_coloring_greedy`]
    /// (`igraph_coloring_greedy_t`).
    pub enum ColoringGreedy: igraph_coloring_greedy_t {
        /// Start from a vertex of maximum degree, then color next the vertex
        /// with the largest number of already colored neighbors
        /// (`IGRAPH_COLORING_GREEDY_COLORED_NEIGHBORS`). Both counts include
        /// multi-edges with their multiplicity.
        ColoredNeighbors = igraph_coloring_greedy_t_IGRAPH_COLORING_GREEDY_COLORED_NEIGHBORS,
        /// Color next the vertex with the largest number of distinct colors
        /// in its neighborhood (its *saturation degree*), breaking ties by the
        /// number of uncolored neighbors (`IGRAPH_COLORING_GREEDY_DSATUR`).
        /// This is Brélaz's DSatur heuristic (Commun. ACM 22(4), 1979,
        /// <https://doi.org/10.1145/359094.359101>); it is exact on bipartite
        /// graphs, cycles and wheels.
        DSatur = igraph_coloring_greedy_t_IGRAPH_COLORING_GREEDY_DSATUR,
    }
}

impl Default for ColoringGreedy {
    /// [`ColoringGreedy::ColoredNeighbors`], igraph's historical default.
    fn default() -> Self {
        Self::ColoredNeighbors
    }
}

/// Options of [`Graph::weighted_cliques`].
///
/// The default finds *all* cliques of any weight, without limits.
///
/// Cliquer only supports integer weights, so both bounds are truncated to
/// their integer part. Unlike in C, where a non-positive maximum means "no
/// bound", here `None` means "no bound" and a maximum below 1 (which no
/// clique can meet, as vertex weights are at least 1) is rejected as an empty
/// range.
///
/// ```
/// use igraph::cliques::WeightedCliqueOptions;
/// let opts = WeightedCliqueOptions::default().with_maximal(true).with_min_weight(7.0);
/// assert!(opts.maximal && opts.max_weight.is_none());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WeightedCliqueOptions {
    /// Only return *maximal* cliques (default `false`).
    pub maximal: bool,
    /// Minimum total weight of a returned clique (inclusive), `None` for no
    /// lower bound. Truncated to its integer part; values of at most 1 are
    /// the same as no bound.
    pub min_weight: Option<f64>,
    /// Maximum total weight of a returned clique (inclusive), `None` for no
    /// upper bound. Truncated to its integer part.
    pub max_weight: Option<f64>,
    /// Stop after this many cliques were found, `None` for no limit
    /// (`Some(0)` returns nothing).
    pub max_results: Option<usize>,
}

impl WeightedCliqueOptions {
    /// Sets the [`maximal`](#structfield.maximal) field.
    pub fn with_maximal(mut self, maximal: bool) -> Self {
        self.maximal = maximal;
        self
    }

    /// Sets the [`min_weight`](#structfield.min_weight) field.
    pub fn with_min_weight(mut self, weight: f64) -> Self {
        self.min_weight = Some(weight);
        self
    }

    /// Sets the [`max_weight`](#structfield.max_weight) field.
    pub fn with_max_weight(mut self, weight: f64) -> Self {
        self.max_weight = Some(weight);
        self
    }

    /// Sets the [`max_results`](#structfield.max_results) field.
    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.max_results = Some(max_results);
        self
    }

    /// Sets the [`maximal`](#structfield.maximal) field; use
    /// [`with_maximal`](Self::with_maximal) instead.
    #[deprecated(note = "use with_maximal")]
    pub fn maximal(self, maximal: bool) -> Self {
        self.with_maximal(maximal)
    }

    /// Sets the [`min_weight`](#structfield.min_weight) field; use
    /// [`with_min_weight`](Self::with_min_weight) instead.
    #[deprecated(note = "use with_min_weight")]
    pub fn min_weight(self, weight: f64) -> Self {
        self.with_min_weight(weight)
    }

    /// Sets the [`max_weight`](#structfield.max_weight) field; use
    /// [`with_max_weight`](Self::with_max_weight) instead.
    #[deprecated(note = "use with_max_weight")]
    pub fn max_weight(self, weight: f64) -> Self {
        self.with_max_weight(weight)
    }

    /// Sets the [`max_results`](#structfield.max_results) field; use
    /// [`with_max_results`](Self::with_max_results) instead.
    #[deprecated(note = "use with_max_results")]
    pub fn max_results(self, max_results: usize) -> Self {
        self.with_max_results(max_results)
    }
}

/// Converts a size range into igraph's `(min_size, max_size)` convention,
/// where a non-positive value means "no bound".
///
/// Returns `Ok(None)` when no vertex set of `graph` can satisfy the range
/// (its lower bound exceeds the number of vertices): callers then return an
/// empty result without calling igraph. This also keeps igraph away from
/// absurd bounds: Cliquer allocates `min_size` integers up front and
/// `igraph_clique_size_hist` allocates `max_size` doubles, and both cast the
/// bounds to a C `int`.
fn size_bounds(
    graph: &Graph,
    sizes: &impl RangeBounds<usize>,
) -> Result<Option<(igraph_int_t, igraph_int_t)>> {
    let min = match sizes.start_bound() {
        Bound::Included(&n) => n,
        Bound::Excluded(&n) => n.saturating_add(1),
        Bound::Unbounded => 0,
    };
    let max = match sizes.end_bound() {
        Bound::Included(&n) => Some(n),
        Bound::Excluded(&n) => Some(n.checked_sub(1).ok_or_else(empty_range)?),
        Bound::Unbounded => None,
    };
    if let Some(max) = max {
        // igraph interprets 0 as "unbounded": reject it explicitly.
        if max == 0 || max < min {
            return Err(empty_range());
        }
    }
    let n = graph.vcount();
    if min > n {
        return Ok(None);
    }
    // No set is larger than the vertex set: a larger bound is no bound.
    let max = max.filter(|&m| m < n).unwrap_or(0);
    Ok(Some((min as igraph_int_t, max as igraph_int_t)))
}

fn empty_range() -> Error {
    Error::invalid("the range of admissible sizes is empty")
}

fn limit(max_results: Option<usize>) -> igraph_int_t {
    max_results.map_or(IGRAPH_UNLIMITED as igraph_int_t, |n| {
        igraph_int_t::try_from(n).unwrap_or(igraph_int_t::MAX)
    })
}

/// Largest value Cliquer can represent (it stores weights as C `int`s).
const CLIQUER_MAX: f64 = i32::MAX as f64;

/// Validates vertex weights for Cliquer: finite, positive after truncation,
/// and small enough that no clique weight overflows Cliquer's `int` sums.
/// (igraph 1.0.0 and 1.0.1 convert them to `int` without checks, which is
/// undefined behavior in C for NaN or out-of-range values.)
fn check_weights(graph: &Graph, weights: Option<&[f64]>) -> Result<()> {
    let Some(w) = weights else { return Ok(()) };
    if w.len() != graph.vcount() {
        return Err(Error::invalid(format!(
            "the vertex weight vector has length {}, but the graph has {} vertices",
            w.len(),
            graph.vcount()
        )));
    }
    if let Some((v, x)) = w
        .iter()
        .enumerate()
        .find(|&(_, &x)| !(x.is_finite() && x.trunc() >= 1.0 && x <= CLIQUER_MAX))
    {
        return Err(Error::invalid(format!(
            "vertex weights must be positive integers, but vertex {v} has weight {x}"
        )));
    }
    let total: f64 = w.iter().map(|x| x.trunc()).sum();
    if total > CLIQUER_MAX {
        return Err(Error::invalid(format!(
            "the total vertex weight {total} exceeds the limit {CLIQUER_MAX} of Cliquer"
        )));
    }
    Ok(())
}

/// Converts the weight range of [`WeightedCliqueOptions`] into igraph's
/// `(min_weight, max_weight)` convention (0 meaning "no bound").
///
/// `heaviest` is an upper bound on the weight of any clique (the total
/// vertex weight, or the vertex count without weights). Returns `Ok(None)`
/// when no clique can reach `min_weight`, so that callers return an empty
/// result without calling igraph; a maximum of at least `heaviest` is no
/// bound. This also keeps both bounds within Cliquer's C `int` range, and
/// keeps absurd sizes away from the unweighted fallbacks.
fn weight_bounds(options: &WeightedCliqueOptions, heaviest: f64) -> Result<Option<(f64, f64)>> {
    let finite = |x: f64, what: &str| {
        if x.is_finite() {
            Ok(x.trunc())
        } else {
            Err(Error::invalid(format!(
                "the {what} weight must be finite, got {x}"
            )))
        }
    };
    let min = match options.min_weight {
        Some(m) => finite(m, "minimum")?.max(0.0),
        None => 0.0,
    };
    let max = match options.max_weight {
        Some(m) => {
            let m = finite(m, "maximum")?;
            // igraph reads a non-positive maximum as "no bound": since
            // weights are at least 1, such a range is really empty.
            if m < 1.0 || m < min {
                return Err(Error::invalid(
                    "the range of admissible clique weights is empty",
                ));
            }
            m
        }
        None => 0.0,
    };
    if min > heaviest {
        return Ok(None);
    }
    Ok(Some((min, if max >= heaviest { 0.0 } else { max })))
}

thread_local! {
    /// Whether a Cliquer-based search is running on this thread.
    static CLIQUER_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

/// Marks a Cliquer-based search as running on this thread.
///
/// igraph's Cliquer wrapper (`src/cliques/cliquer_wrapper.c`, unchanged in
/// igraph 1.0.0 and 1.0.1) keeps its search options (the result sink and
/// the per-clique callback) in a *thread-local global* that every search
/// overwrites and never restores. Starting a second Cliquer search from
/// inside the closure of [`Graph::cliques_callback`] would therefore leave
/// the outer search writing through dangling pointers. This guard turns
/// such a nested call into an error instead.
struct CliquerGuard;

impl CliquerGuard {
    fn enter() -> Result<Self> {
        if CLIQUER_ACTIVE.with(|a| a.replace(true)) {
            return Err(Error::new(
                ErrorKind::Failure,
                "Cliquer-based clique searches (cliques, clique_size_hist, cliques_callback \
                 and the weighted clique functions) cannot be nested inside a \
                 cliques_callback closure",
            ));
        }
        Ok(CliquerGuard)
    }
}

impl Drop for CliquerGuard {
    fn drop(&mut self) {
        CLIQUER_ACTIVE.with(|a| a.set(false));
    }
}

/// Runs `f`, an igraph call on `graph`, with the graph's property cache
/// dropped before and after it when `graph` is directed.
///
/// igraph bug (1.0.0 and 1.0.1): many functions that ignore edge directions
/// (triangles, cliques, independent sets, greedy coloring, ...) build an
/// adjacency list with `igraph_adjlist_init(.., IGRAPH_ALL, ..,
/// IGRAPH_NO_MULTIPLE)` (or the lazy variant), which trusts and updates the
/// cached "has multi-edges" flag. In mode ALL, a mutual pair `u -> v`,
/// `v -> u` of a *directed* graph looks like a multi-edge, so:
/// - if "no multi-edges" was already cached (a correct value, e.g. written by
///   [`Graph::has_multiple`]), the deduplication is skipped and mutual
///   neighbors are counted twice (wrong triangle counts, duplicated cliques);
/// - otherwise "has multi-edges" is cached, which is wrong for the directed
///   graph (igraph only calls same-direction edges multi-edges), and it
///   survives edge additions and clones.
///
/// Dropping the cache on both sides avoids both; it only costs the
/// recomputation of cached properties. `Graph` is `Send` but not `Sync`, so
/// no other thread can observe the cache through `&Graph` meanwhile. The
/// cache is also dropped when `f` panics (e.g. resuming a callback panic).
pub(crate) fn directed_cache_guard<T>(graph: &Graph, f: impl FnOnce() -> T) -> T {
    graph.with_fresh_multi_cache(f)
}

fn counts(hist: Vector) -> Vec<usize> {
    hist.iter().map(|&c| c as usize).collect()
}

/// C trampoline running a Rust closure for each clique.
///
/// The closure runs in a fresh level of igraph's "finally" stack
/// (`IGRAPH_FINALLY_ENTER` / `IGRAPH_FINALLY_EXIT`). The running clique
/// search keeps its temporary objects on that stack, and igraph's error
/// handler frees the current level when a call fails: without the new level,
/// an igraph call made by the closure that fails (and returns an `Err` to the
/// closure, which may well ignore it) would free the objects of the search
/// that is still running, a use-after-free in C.
///
/// [`catch_panic`] does not open a level itself (as of this writing), so this
/// is the protection for the clique callbacks. Should it start doing so, this
/// level becomes redundant but stays harmless: finally levels nest, and each
/// one is closed by its own opener.
unsafe extern "C" fn clique_trampoline<F>(
    clique: *const igraph_vector_int_t,
    arg: *mut c_void,
) -> igraph_error_t
where
    F: FnMut(&[VertexId]) -> ControlFlow<()>,
{
    // SAFETY: plain bookkeeping on igraph's thread-local finally stack; the
    // matching EXIT runs below, as `catch_panic` never unwinds.
    unsafe { IGRAPH_FINALLY_ENTER() };
    let code = catch_panic(|| {
        // SAFETY: `arg` is the `&mut F` handed to igraph by the wrapper that
        // started the search, and `clique` is a valid vector owned by igraph
        // for the duration of this call.
        let f = unsafe { &mut *(arg as *mut F) };
        let clique = unsafe { &*clique };
        match f(clique.as_slice()) {
            ControlFlow::Continue(()) => igraph_error_type_t_IGRAPH_SUCCESS,
            ControlFlow::Break(()) => igraph_error_type_t_IGRAPH_STOP,
        }
    });
    // SAFETY: closes the level opened above. Every igraph call made by the
    // closure has returned, and a failed one has already freed its own
    // objects, so the level is empty again.
    unsafe { IGRAPH_FINALLY_EXIT() };
    code
}

/// Opens an in-memory C stream, runs `f` on it and returns what was written.
fn with_memstream(f: impl FnOnce(*mut FILE) -> Result<()>) -> Result<Vec<u8>> {
    let mut buf: *mut c_char = std::ptr::null_mut();
    let mut size: usize = 0;
    // SAFETY: `buf` and `size` outlive the stream, which is closed below.
    let file = unsafe { open_memstream(&mut buf, &mut size) };
    if file.is_null() {
        return Err(Error::new(
            ErrorKind::File,
            "cannot open an in-memory stream",
        ));
    }
    let outcome = f(file);
    // SAFETY: closing the stream finalizes `buf`/`size`; `buf` is then ours to free.
    let closed = unsafe { fclose(file) } == 0;
    let bytes = if buf.is_null() {
        Vec::new()
    } else {
        let bytes = unsafe { std::slice::from_raw_parts(buf as *const u8, size) }.to_vec();
        unsafe { free(buf as *mut c_void) };
        bytes
    };
    outcome?;
    if !closed {
        return Err(Error::new(
            ErrorKind::File,
            "cannot flush an in-memory stream",
        ));
    }
    Ok(bytes)
}

impl igraph_t {
    // ------------------------------------------------------------------
    // All cliques (Cliquer)
    // ------------------------------------------------------------------

    /// Finds all cliques whose size lies in `sizes`.
    ///
    /// Every clique is reported, not only the maximal ones: a triangle
    /// contributes three cliques of size 1, three of size 2 and one of size 3.
    /// The search stops after `max_results` cliques when that is `Some`.
    /// If you only need the size of the largest clique, use
    /// [`clique_number`](Self::clique_number) instead; to only list the
    /// maximal ones, use [`maximal_cliques`](Self::maximal_cliques). Edge
    /// directions, self-loops and multi-edges are ignored.
    ///
    /// See also [`is_clique`](Self::is_clique) to test a given vertex set,
    /// and [`list_triangles`](Self::list_triangles) for cliques of size 3
    /// only.
    ///
    /// The implementation uses the Cliquer library (version 1.21) by Sampo
    /// Niskanen and Patric R. J. Östergård. Time complexity: exponential.
    ///
    /// Binds [`igraph_cliques`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_cliques).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `sizes` is empty.
    /// [`ErrorKind::Failure`] if called from inside a
    /// [`cliques_callback`](Self::cliques_callback) closure (see the
    /// [module docs](self#nesting-searches-in-callbacks)).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // K4 has C(4, 3) = 4 triangles.
    /// let k4 = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], 4, false)
    ///     .unwrap();
    /// assert_eq!(k4.cliques(3..=3, None).unwrap().len(), 4);
    /// // 15 = 2^4 - 1 non-empty cliques in total, but we only want two of them.
    /// assert_eq!(k4.cliques(.., None).unwrap().len(), 15);
    /// assert_eq!(k4.cliques(.., Some(2)).unwrap().len(), 2);
    /// ```
    pub fn cliques(
        &self,
        sizes: impl RangeBounds<usize>,
        max_results: Option<usize>,
    ) -> Result<Vec<Vec<VertexId>>> {
        let Some((min, max)) = size_bounds(self, &sizes)? else {
            return Ok(Vec::new());
        };
        let _guard = CliquerGuard::enter()?;
        let mut res = VectorIntList::new();
        igraph_call!(igraph_cliques(self, &mut res, min, max, limit(max_results)))?;
        Ok(res.to_vecs())
    }

    /// Calls `f` for each clique whose size lies in `sizes`, without storing
    /// them.
    ///
    /// The closure receives the vertex ids of the clique (a borrowed slice,
    /// copy it if you want to keep it) and returns
    /// [`ControlFlow::Continue`] to go on or [`ControlFlow::Break`] to stop the
    /// search early; stopping is not an error. Cliques are produced in the
    /// same order as by [`cliques`](Self::cliques). A panic inside `f` stops
    /// the search and is propagated to the caller.
    ///
    /// The closure must not start another Cliquer-based search
    /// ([`cliques`](Self::cliques), [`clique_size_hist`](Self::clique_size_hist),
    /// `cliques_callback` itself or the weighted clique functions): such
    /// nested calls return an [`ErrorKind::Failure`] error, because igraph
    /// keeps the state of the running search in per-thread globals. Other
    /// functions, e.g. [`maximal_cliques`](Self::maximal_cliques), are fine.
    ///
    /// Binds [`igraph_cliques_callback`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_cliques_callback).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `sizes` is empty.
    /// [`ErrorKind::Failure`] if called from inside another
    /// `cliques_callback` closure.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use std::ops::ControlFlow;
    /// let k5 = Graph::from_edges(
    ///     &[(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
    ///     5,
    ///     false,
    /// )
    /// .unwrap();
    /// // Find the first triangle containing vertex 4, then stop.
    /// let mut found = None;
    /// k5.cliques_callback(3..=3, |c| {
    ///     if c.contains(&4) {
    ///         found = Some(c.to_vec());
    ///         ControlFlow::Break(())
    ///     } else {
    ///         ControlFlow::Continue(())
    ///     }
    /// })
    /// .unwrap();
    /// assert!(found.unwrap().contains(&4));
    /// ```
    pub fn cliques_callback<F>(&self, sizes: impl RangeBounds<usize>, mut f: F) -> Result<()>
    where
        F: FnMut(&[VertexId]) -> ControlFlow<()>,
    {
        let Some((min, max)) = size_bounds(self, &sizes)? else {
            return Ok(());
        };
        let _guard = CliquerGuard::enter()?;
        igraph_call!(igraph_cliques_callback(
            self,
            min,
            max,
            Some(clique_trampoline::<F>),
            &mut f as *mut F as *mut c_void
        ))
    }

    /// Counts the cliques of each size.
    ///
    /// Element `i` of the result is the number of cliques of size `i + 1`;
    /// sizes below the lower bound of `sizes` get a zero count, and the vector
    /// ends at the largest size found (so it is empty for the null graph).
    /// The counts are those of [`cliques`](Self::cliques), without storing the
    /// cliques: `hist[0]` is the number of vertices, `hist[1]` the number of
    /// adjacent vertex pairs and `hist[2]` the number of triangles (see
    /// [`count_triangles`](Self::count_triangles)). Uses Cliquer; time
    /// complexity: exponential.
    ///
    /// Binds [`igraph_clique_size_hist`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_clique_size_hist).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `sizes` is empty.
    /// [`ErrorKind::Failure`] if called from inside a
    /// [`cliques_callback`](Self::cliques_callback) closure.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // A triangle: 3 vertices, 3 edges, 1 triangle.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, false).unwrap();
    /// assert_eq!(g.clique_size_hist(..).unwrap(), vec![3, 3, 1]);
    /// assert_eq!(g.clique_size_hist(2..).unwrap(), vec![0, 3, 1]);
    /// ```
    pub fn clique_size_hist(&self, sizes: impl RangeBounds<usize>) -> Result<Vec<usize>> {
        let Some((min, max)) = size_bounds(self, &sizes)? else {
            return Ok(Vec::new());
        };
        let _guard = CliquerGuard::enter()?;
        let mut hist = Vector::new();
        igraph_call!(igraph_clique_size_hist(self, &mut hist, min, max))?;
        Ok(counts(hist))
    }

    /// Finds all the largest (maximum) cliques.
    ///
    /// A clique is *largest* if no other clique has more vertices. Largest
    /// cliques are always maximal, but a maximal clique need not be largest.
    /// The null graph has no cliques at all. The maximal cliques are
    /// enumerated with the same algorithm as
    /// [`maximal_cliques`](Self::maximal_cliques), keeping only the largest.
    /// All returned cliques have [`clique_number`](Self::clique_number)
    /// vertices.
    ///
    /// Time complexity: O(3^(|V|/3)) in the worst case.
    ///
    /// Binds [`igraph_largest_cliques`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_largest_cliques).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // The two 5-cliques of Zachary's karate club share the four leaders
    /// // 0, 1, 2, 3.
    /// let karate = Graph::famous("Zachary").unwrap();
    /// let mut largest = karate.largest_cliques().unwrap();
    /// largest.iter_mut().for_each(|c| c.sort());
    /// largest.sort();
    /// assert_eq!(largest, vec![vec![0, 1, 2, 3, 7], vec![0, 1, 2, 3, 13]]);
    /// ```
    pub fn largest_cliques(&self) -> Result<Vec<Vec<VertexId>>> {
        let mut res = VectorIntList::new();
        directed_cache_guard(self, || {
            igraph_call!(igraph_largest_cliques(self, &mut res))
        })?;
        Ok(res.to_vecs())
    }

    /// The clique number ω(G): the size of the largest clique.
    ///
    /// It is 0 for the null graph and 1 for a graph without edges. It is a
    /// lower bound for the chromatic number, i.e. the number of colors of
    /// any proper coloring (see
    /// [`vertex_coloring_greedy`](Self::vertex_coloring_greedy)); the two
    /// are equal for [perfect](Self::is_perfect) graphs.
    /// Time complexity: O(3^(|V|/3)) in the worst case.
    ///
    /// Binds [`igraph_clique_number`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_clique_number).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let path = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// assert_eq!(path.clique_number().unwrap(), 2);
    /// assert_eq!(Graph::new(0, false).clique_number().unwrap(), 0);
    /// ```
    pub fn clique_number(&self) -> Result<usize> {
        let mut no: igraph_int_t = 0;
        directed_cache_guard(self, || igraph_call!(igraph_clique_number(self, &mut no)))?;
        Ok(no as usize)
    }

    // ------------------------------------------------------------------
    // Maximal cliques (Eppstein–Löffler–Strash)
    // ------------------------------------------------------------------

    /// Finds the maximal cliques whose size lies in `sizes`.
    ///
    /// A *maximal* clique is not a proper subset of any other clique.
    /// Isolated vertices are maximal cliques of size 1. No guarantees are given
    /// about the order of the cliques or of the vertices inside them.
    ///
    /// The implementation is the Bron–Kerbosch variant with degeneracy
    /// ordering by Eppstein, Löffler and Strash (2010,
    /// <https://arxiv.org/abs/1006.5440>). Time complexity: O(d (n − d)
    /// 3^(d/3)) in the worst case, where d is the degeneracy of the graph
    /// (typically small for sparse graphs; it is the largest value of
    /// [`coreness`](Self::coreness)).
    ///
    /// See also [`maximal_cliques_count`](Self::maximal_cliques_count),
    /// [`maximal_cliques_hist`](Self::maximal_cliques_hist) and
    /// [`maximal_cliques_callback`](Self::maximal_cliques_callback) to avoid
    /// storing the cliques, and
    /// [`maximal_independent_vertex_sets`](Self::maximal_independent_vertex_sets)
    /// for the maximal cliques of the complementer graph.
    ///
    /// Binds [`igraph_maximal_cliques`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_maximal_cliques).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `sizes` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // A "bowtie": triangles {0,1,2} and {2,3,4}.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)], 5, false)
    ///     .unwrap();
    /// let mut cliques = g.maximal_cliques(.., None).unwrap();
    /// cliques.iter_mut().for_each(|c| c.sort());
    /// cliques.sort();
    /// assert_eq!(cliques, vec![vec![0, 1, 2], vec![2, 3, 4]]);
    /// ```
    pub fn maximal_cliques(
        &self,
        sizes: impl RangeBounds<usize>,
        max_results: Option<usize>,
    ) -> Result<Vec<Vec<VertexId>>> {
        let Some((min, max)) = size_bounds(self, &sizes)? else {
            return Ok(Vec::new());
        };
        let mut res = VectorIntList::new();
        directed_cache_guard(self, || {
            igraph_call!(igraph_maximal_cliques(
                self,
                &mut res,
                min,
                max,
                limit(max_results)
            ))
        })?;
        Ok(res.to_vecs())
    }

    /// Calls `f` for each maximal clique whose size lies in `sizes`.
    ///
    /// This is the streaming version of [`maximal_cliques`](Self::maximal_cliques):
    /// the cliques are not stored, which is useful for graphs with a huge
    /// number of them. The closure receives a borrowed slice (copy it to keep
    /// it) and returns [`ControlFlow::Break`] to stop the search early, which
    /// is not an error. A panic inside `f` is propagated to the caller.
    ///
    /// Binds [`igraph_maximal_cliques_callback`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_maximal_cliques_callback).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `sizes` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use std::ops::ControlFlow;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false).unwrap();
    /// let mut sizes = vec![];
    /// g.maximal_cliques_callback(.., |c| {
    ///     sizes.push(c.len());
    ///     ControlFlow::Continue(())
    /// })
    /// .unwrap();
    /// sizes.sort();
    /// assert_eq!(sizes, vec![2, 3]);
    /// ```
    pub fn maximal_cliques_callback<F>(
        &self,
        sizes: impl RangeBounds<usize>,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(&[VertexId]) -> ControlFlow<()>,
    {
        let Some((min, max)) = size_bounds(self, &sizes)? else {
            return Ok(());
        };
        directed_cache_guard(self, || {
            igraph_call!(igraph_maximal_cliques_callback(
                self,
                min,
                max,
                Some(clique_trampoline::<F>),
                &mut f as *mut F as *mut c_void
            ))
        })
    }

    /// Counts the maximal cliques whose size lies in `sizes`, without storing
    /// them.
    ///
    /// Same algorithm and complexity as [`maximal_cliques`](Self::maximal_cliques).
    ///
    /// Binds [`igraph_maximal_cliques_count`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_maximal_cliques_count).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `sizes` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // Each edge of a 5-cycle is a maximal clique.
    /// let c5 = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 5, false).unwrap();
    /// assert_eq!(c5.maximal_cliques_count(..).unwrap(), 5);
    /// assert_eq!(c5.maximal_cliques_count(3..).unwrap(), 0);
    /// ```
    pub fn maximal_cliques_count(&self, sizes: impl RangeBounds<usize>) -> Result<usize> {
        let Some((min, max)) = size_bounds(self, &sizes)? else {
            return Ok(0);
        };
        let mut no: igraph_int_t = 0;
        directed_cache_guard(self, || {
            igraph_call!(igraph_maximal_cliques_count(self, &mut no, min, max))
        })?;
        Ok(no as usize)
    }

    /// Counts the maximal cliques of each size.
    ///
    /// Element `i` of the result is the number of maximal cliques of size
    /// `i + 1` (size-1 maximal cliques are the isolated vertices); sizes below
    /// the lower bound of `sizes` get a zero count, and the vector ends at the
    /// largest size found. The counts sum to
    /// [`maximal_cliques_count`](Self::maximal_cliques_count), and the length
    /// of the unrestricted histogram is the
    /// [`clique_number`](Self::clique_number).
    ///
    /// Binds [`igraph_maximal_cliques_hist`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_maximal_cliques_hist).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `sizes` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // A triangle with a pendant edge, plus an isolated vertex.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 5, false).unwrap();
    /// assert_eq!(g.maximal_cliques_hist(..).unwrap(), vec![1, 1, 1]);
    /// assert_eq!(g.maximal_cliques_hist(2..).unwrap(), vec![0, 1, 1]);
    /// ```
    pub fn maximal_cliques_hist(&self, sizes: impl RangeBounds<usize>) -> Result<Vec<usize>> {
        let Some((min, max)) = size_bounds(self, &sizes)? else {
            return Ok(Vec::new());
        };
        let mut hist = Vector::new();
        directed_cache_guard(self, || {
            igraph_call!(igraph_maximal_cliques_hist(self, &mut hist, min, max))
        })?;
        Ok(counts(hist))
    }

    /// Finds the maximal cliques reached from a subset of initial vertices.
    ///
    /// The Eppstein–Löffler–Strash algorithm processes the vertices one by one
    /// in degeneracy order, each time listing the maximal cliques that contain
    /// that vertex and none of the vertices processed before it. This function
    /// only runs the outer loop over the vertices in `subset`: running it on
    /// the parts of a partition of the vertex set and concatenating the results
    /// yields every maximal clique exactly once, which makes it a building
    /// block for parallel enumeration.
    ///
    /// Binds [`igraph_maximal_cliques_subset`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_maximal_cliques_subset)
    /// (without its optional file output, see
    /// [`write_maximal_cliques`](Self::write_maximal_cliques)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `sizes` is empty,
    /// [`ErrorKind::InvalidVertexId`] if `subset` contains an invalid id.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)], 5, false)
    ///     .unwrap();
    /// let all = g.maximal_cliques(.., None).unwrap().len();
    /// let evens = g.maximal_cliques_subset(&[0, 2, 4], .., None).unwrap();
    /// let odds = g.maximal_cliques_subset(&[1, 3], .., None).unwrap();
    /// assert_eq!(evens.len() + odds.len(), all);
    /// ```
    pub fn maximal_cliques_subset(
        &self,
        subset: &[VertexId],
        sizes: impl RangeBounds<usize>,
        max_results: Option<usize>,
    ) -> Result<Vec<Vec<VertexId>>> {
        let bounds = size_bounds(self, &sizes)?;
        let n = self.vcount() as VertexId;
        if let Some(&v) = subset.iter().find(|&&v| !(0..n).contains(&v)) {
            return Err(Error::new(
                ErrorKind::InvalidVertexId,
                format!("vertex id {v} is out of range in the subset of initial vertices"),
            ));
        }
        let Some((min, max)) = bounds else {
            return Ok(Vec::new());
        };
        let subset = VectorInt::view(subset);
        let mut res = VectorIntList::new();
        let mut no: igraph_int_t = 0;
        directed_cache_guard(self, || {
            igraph_call!(igraph_maximal_cliques_subset(
                self,
                subset.as_ptr(),
                &mut res,
                &mut no,
                std::ptr::null_mut(),
                min,
                max,
                limit(max_results)
            ))
        })?;
        Ok(res.to_vecs())
    }

    /// Writes the maximal cliques whose size lies in `sizes` to `out`, one
    /// clique per line as space-separated vertex ids.
    ///
    /// This mirrors the C function, which streams the cliques to a `FILE *`
    /// instead of storing them: igraph writes into an in-memory stream that is
    /// then copied into `out`.
    ///
    /// Binds [`igraph_maximal_cliques_file`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_maximal_cliques_file).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `sizes` is empty, [`ErrorKind::File`]
    /// if writing to `out` (or to the intermediate stream) fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 4, false).unwrap();
    /// let mut out = Vec::new();
    /// g.write_maximal_cliques(&mut out, 2.., None).unwrap();
    /// let text = String::from_utf8(out).unwrap();
    /// assert_eq!(text.lines().count(), 1); // the triangle; vertex 3 is too small
    /// ```
    pub fn write_maximal_cliques<W: io::Write + ?Sized>(
        &self,
        out: &mut W,
        sizes: impl RangeBounds<usize>,
        max_results: Option<usize>,
    ) -> Result<()> {
        let Some((min, max)) = size_bounds(self, &sizes)? else {
            return Ok(());
        };
        let bytes = with_memstream(|file| {
            directed_cache_guard(self, || {
                igraph_call!(igraph_maximal_cliques_file(
                    self,
                    file,
                    min,
                    max,
                    limit(max_results)
                ))
            })
        })?;
        out.write_all(&bytes)
            .map_err(|e| Error::new(ErrorKind::File, e.to_string()))
    }

    // ------------------------------------------------------------------
    // Weighted cliques (Cliquer)
    // ------------------------------------------------------------------

    /// Finds the cliques whose total vertex weight lies in a range.
    ///
    /// The weight of a clique is the sum of the weights of its vertices.
    /// Only positive integer weights are supported: fractional weights (and
    /// weight bounds) are truncated to their integer part, with a warning.
    /// Weights must be finite, at least 1, and sum to at most `i32::MAX`
    /// (Cliquer computes with C `int`s).
    /// With `weights = None` every vertex weighs 1, so weights are sizes and
    /// this is [`cliques`](Self::cliques) or [`maximal_cliques`](Self::maximal_cliques).
    /// See [`WeightedCliqueOptions`] for the range, maximality and limit.
    ///
    /// Uses the Cliquer library. Time complexity: exponential.
    ///
    /// Binds [`igraph_weighted_cliques`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_weighted_cliques).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `weights` has the wrong length or an
    /// invalid weight (see above), if a bound is not finite, or if the weight
    /// range is empty (`max_weight < 1` or `max_weight < min_weight`).
    /// [`ErrorKind::Failure`] if called from inside a
    /// [`cliques_callback`](Self::cliques_callback) closure (see the
    /// [module docs](self#nesting-searches-in-callbacks)).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::{cliques::WeightedCliqueOptions, prelude::*};
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false).unwrap();
    /// let w = [1.0, 1.0, 1.0, 10.0];
    /// // Maximal cliques weighing at least 5: only the edge {2, 3} (weight 11).
    /// let opts = WeightedCliqueOptions::default().with_maximal(true).with_min_weight(5.0);
    /// let mut heavy = g.weighted_cliques(Some(&w), &opts).unwrap();
    /// heavy[0].sort();
    /// assert_eq!(heavy, vec![vec![2, 3]]);
    /// ```
    pub fn weighted_cliques(
        &self,
        weights: Option<&[f64]>,
        options: &WeightedCliqueOptions,
    ) -> Result<Vec<Vec<VertexId>>> {
        check_weights(self, weights)?;
        // Validated above: the sum is exact and at most `i32::MAX`.
        let heaviest = weights.map_or(self.vcount() as f64, |w| w.iter().map(|x| x.trunc()).sum());
        let Some((min, max)) = weight_bounds(options, heaviest)? else {
            return Ok(Vec::new());
        };
        // Without weights, maximal cliques use the Eppstein–Löffler–Strash
        // algorithm; every other case goes through Cliquer.
        let _guard = if weights.is_some() || !options.maximal {
            Some(CliquerGuard::enter()?)
        } else {
            None
        };
        let w = weights.map(Vector::view);
        let wp = w.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let mut res = VectorIntList::new();
        igraph_call!(igraph_weighted_cliques(
            self,
            wp,
            &mut res,
            options.maximal,
            min,
            max,
            limit(options.max_results)
        ))?;
        Ok(res.to_vecs())
    }

    /// Finds the cliques of largest total vertex weight.
    ///
    /// Only positive integer weights are supported (fractional weights are
    /// truncated). With `weights = None` this is
    /// [`largest_cliques`](Self::largest_cliques). The total weight of every
    /// returned clique is the
    /// [`weighted_clique_number`](Self::weighted_clique_number). A heaviest
    /// clique is always maximal, but it need not be a largest one.
    ///
    /// Uses the Cliquer library when `weights` is given. Time complexity:
    /// exponential.
    ///
    /// Binds [`igraph_largest_weighted_cliques`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_largest_weighted_cliques).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `weights` has the wrong length or a
    /// weight that is not finite, below 1, or makes the total exceed
    /// `i32::MAX`. [`ErrorKind::Failure`] if `weights` is given and this is
    /// called from inside a [`cliques_callback`](Self::cliques_callback)
    /// closure.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // A triangle {0, 1, 2} of light vertices and a heavy edge {2, 3}.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false).unwrap();
    /// let mut heaviest = g.largest_weighted_cliques(Some(&[1.0, 1.0, 1.0, 10.0])).unwrap();
    /// heaviest[0].sort();
    /// assert_eq!(heaviest, vec![vec![2, 3]]);
    /// // Unweighted, the triangle wins.
    /// assert_eq!(g.largest_weighted_cliques(None).unwrap().len(), 1);
    /// assert_eq!(g.largest_weighted_cliques(None).unwrap()[0].len(), 3);
    /// ```
    pub fn largest_weighted_cliques(&self, weights: Option<&[f64]>) -> Result<Vec<Vec<VertexId>>> {
        check_weights(self, weights)?;
        let _guard = weights.map(|_| CliquerGuard::enter()).transpose()?;
        let w = weights.map(Vector::view);
        let wp = w.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let mut res = VectorIntList::new();
        igraph_call!(igraph_largest_weighted_cliques(self, wp, &mut res))?;
        Ok(res.to_vecs())
    }

    /// The weighted clique number: the largest total vertex weight of a
    /// clique.
    ///
    /// Only positive integer weights are supported (fractional weights are
    /// truncated). With `weights = None` this is the
    /// [`clique_number`](Self::clique_number). It is 0 for the null graph.
    /// Uses the Cliquer library when `weights` is given. Time complexity:
    /// exponential.
    ///
    /// Binds [`igraph_weighted_clique_number`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_weighted_clique_number).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `weights` has the wrong length or a
    /// weight that is not finite, below 1, or makes the total exceed
    /// `i32::MAX`. [`ErrorKind::Failure`] if `weights` is given and this is
    /// called from inside a [`cliques_callback`](Self::cliques_callback)
    /// closure.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false).unwrap();
    /// assert_eq!(g.weighted_clique_number(Some(&[1.0, 1.0, 1.0, 10.0])).unwrap(), 11.0);
    /// assert_eq!(g.weighted_clique_number(None).unwrap(), 3.0);
    /// ```
    pub fn weighted_clique_number(&self, weights: Option<&[f64]>) -> Result<f64> {
        check_weights(self, weights)?;
        let _guard = weights.map(|_| CliquerGuard::enter()).transpose()?;
        let w = weights.map(Vector::view);
        let wp = w.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let mut res: igraph_real_t = 0.0;
        igraph_call!(igraph_weighted_clique_number(self, wp, &mut res))?;
        Ok(res)
    }

    // ------------------------------------------------------------------
    // Independent vertex sets
    // ------------------------------------------------------------------

    /// Finds all independent vertex sets whose size lies in `sizes`.
    ///
    /// A vertex set is *independent* if no two of its vertices are adjacent.
    /// As with [`cliques`](Self::cliques), every such set is reported, not only
    /// the maximal ones. If you only need the size of the largest one, use
    /// [`independence_number`](Self::independence_number). The independent
    /// sets of G are exactly the [`cliques`](Self::cliques) of its
    /// [complementer](Self::complementer). Edge directions are ignored.
    ///
    /// See also [`is_independent_vertex_set`](Self::is_independent_vertex_set)
    /// to test a given vertex set.
    ///
    /// The implementation was ported from the Very Nauty Graph Library by
    /// Keith Briggs and uses the algorithm of Tsukiyama, Ide, Ariyoshi and
    /// Shirakawa (SIAM J. Computing 6:505–517, 1977).
    ///
    /// Binds [`igraph_independent_vertex_sets`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_independent_vertex_sets).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `sizes` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // In the path 0-1-2-3 the independent pairs are {0,2}, {0,3} and {1,3}.
    /// let path = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let mut pairs = path.independent_vertex_sets(2..=2, None).unwrap();
    /// pairs.iter_mut().for_each(|s| s.sort());
    /// pairs.sort();
    /// assert_eq!(pairs, vec![vec![0, 2], vec![0, 3], vec![1, 3]]);
    /// ```
    pub fn independent_vertex_sets(
        &self,
        sizes: impl RangeBounds<usize>,
        max_results: Option<usize>,
    ) -> Result<Vec<Vec<VertexId>>> {
        let Some((min, max)) = size_bounds(self, &sizes)? else {
            return Ok(Vec::new());
        };
        let mut res = VectorIntList::new();
        directed_cache_guard(self, || {
            igraph_call!(igraph_independent_vertex_sets(
                self,
                &mut res,
                min,
                max,
                limit(max_results)
            ))
        })?;
        Ok(res.to_vecs())
    }

    /// Finds all the largest (maximum) independent vertex sets.
    ///
    /// An independent set is *largest* if no other independent set has more
    /// vertices; its size is the [independence number](Self::independence_number).
    /// Largest independent sets are always maximal (see
    /// [`maximal_independent_vertex_sets`](Self::maximal_independent_vertex_sets)),
    /// but not conversely. Edge directions are ignored (with a warning).
    ///
    /// Uses the algorithm of Tsukiyama et al. (1977), ported from the Very
    /// Nauty Graph Library.
    ///
    /// Binds [`igraph_largest_independent_vertex_sets`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_largest_independent_vertex_sets).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // The Petersen graph has exactly five independent sets of size 4.
    /// let petersen = Graph::famous("Petersen").unwrap();
    /// let largest = petersen.largest_independent_vertex_sets().unwrap();
    /// assert_eq!(largest.len(), 5);
    /// assert!(largest.iter().all(|s| s.len() == 4));
    /// assert!(petersen.is_independent_vertex_set(&largest[0]).unwrap());
    /// ```
    pub fn largest_independent_vertex_sets(&self) -> Result<Vec<Vec<VertexId>>> {
        let mut res = VectorIntList::new();
        directed_cache_guard(self, || {
            igraph_call!(igraph_largest_independent_vertex_sets(self, &mut res))
        })?;
        Ok(res.to_vecs())
    }

    /// Finds the maximal independent vertex sets whose size lies in `sizes`.
    ///
    /// A *maximal* independent set cannot be extended by adding any other
    /// vertex; equivalently it is an independent *dominating* set. Maximal
    /// independent sets of G are the maximal cliques of its complementer.
    ///
    /// Uses the algorithm of Tsukiyama et al. (1977), as implemented by Kevin
    /// O'Neill and K. M. Briggs in the Very Nauty Graph Library. Edge
    /// directions are ignored (with a warning).
    ///
    /// Binds [`igraph_maximal_independent_vertex_sets`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_maximal_independent_vertex_sets).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `sizes` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // In the star with center 0, the maximal independent sets are {0}
    /// // and the set of all leaves.
    /// let star = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], 4, false).unwrap();
    /// let mut sets = star.maximal_independent_vertex_sets(.., None).unwrap();
    /// sets.iter_mut().for_each(|s| s.sort());
    /// sets.sort();
    /// assert_eq!(sets, vec![vec![0], vec![1, 2, 3]]);
    /// // They are the maximal cliques of the complementer graph.
    /// let comp = star.complementer(false).unwrap();
    /// assert_eq!(comp.maximal_cliques_count(..).unwrap(), 2);
    /// ```
    pub fn maximal_independent_vertex_sets(
        &self,
        sizes: impl RangeBounds<usize>,
        max_results: Option<usize>,
    ) -> Result<Vec<Vec<VertexId>>> {
        let Some((min, max)) = size_bounds(self, &sizes)? else {
            return Ok(Vec::new());
        };
        let mut res = VectorIntList::new();
        directed_cache_guard(self, || {
            igraph_call!(igraph_maximal_independent_vertex_sets(
                self,
                &mut res,
                min,
                max,
                limit(max_results)
            ))
        })?;
        Ok(res.to_vecs())
    }

    /// The independence number α(G): the size of the largest independent
    /// vertex set.
    ///
    /// By definition α(G) = ω(complement of G) (see
    /// [`complementer`](Self::complementer)). It is 0 for the null graph.
    /// Every color class of a proper vertex coloring is an independent set,
    /// so α(G) · χ(G) ≥ |V|. Edge directions are ignored (with a warning).
    ///
    /// Binds [`igraph_independence_number`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_independence_number).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // α(C5) = 2: any three vertices of a pentagon contain an edge.
    /// let c5 = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 5, false).unwrap();
    /// assert_eq!(c5.independence_number().unwrap(), 2);
    /// ```
    pub fn independence_number(&self) -> Result<usize> {
        let mut no: igraph_int_t = 0;
        directed_cache_guard(self, || {
            igraph_call!(igraph_independence_number(self, &mut no))
        })?;
        Ok(no as usize)
    }

    // ------------------------------------------------------------------
    // Coloring (igraph_coloring.h)
    // ------------------------------------------------------------------

    /// Computes a proper vertex coloring greedily.
    ///
    /// Colors are the integers `0, 1, 2, ...`; element `v` of the result is
    /// the color of vertex `v`, and adjacent vertices always get different
    /// colors. Vertices are colored one at a time, each receiving the smallest
    /// color not used by its already colored neighbors, in an order chosen by
    /// `heuristic` (see [`ColoringGreedy`]). The number of colors used is an
    /// upper bound on the chromatic number, not necessarily optimal; the
    /// [`clique_number`](Self::clique_number) is a lower bound. Edge
    /// directions and self-loops are ignored. Multi-edges never affect the
    /// validity of the coloring, but [`ColoringGreedy::ColoredNeighbors`]
    /// counts them with their multiplicity when choosing the next vertex
    /// (so the colors can differ from those of the simplified graph), while
    /// [`ColoringGreedy::DSatur`] ignores them. The algorithm is
    /// deterministic (it does not use the random number generator).
    ///
    /// See also [`bipartite_types`](Self::bipartite_types), which finds a
    /// 2-coloring whenever one exists.
    ///
    /// Binds [`igraph_vertex_coloring_greedy`](https://igraph.org/c/html/latest/igraph-Coloring.html#igraph_vertex_coloring_greedy).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::{cliques::ColoringGreedy, prelude::*};
    /// // An even cycle is bipartite: DSatur finds a 2-coloring.
    /// let c6 = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)], 6, false)
    ///     .unwrap();
    /// let colors = c6.vertex_coloring_greedy(ColoringGreedy::DSatur).unwrap();
    /// assert!(c6.is_vertex_coloring(&colors).unwrap());
    /// assert_eq!(*colors.iter().max().unwrap(), 1);
    /// ```
    pub fn vertex_coloring_greedy(&self, heuristic: ColoringGreedy) -> Result<Vec<i64>> {
        let mut colors = VectorInt::new();
        directed_cache_guard(self, || {
            igraph_call!(igraph_vertex_coloring_greedy(
                self,
                &mut colors,
                heuristic.into()
            ))
        })?;
        Ok(colors.into())
    }

    /// Checks whether `colors` (one integer per vertex) is a proper vertex
    /// coloring, i.e. no edge joins two vertices of the same color.
    ///
    /// Colors may be any integers (they need not be consecutive). Edge
    /// directions are ignored, and so are self-loops. Time complexity: O(|E|).
    ///
    /// Binds [`igraph_is_vertex_coloring`](https://igraph.org/c/html/latest/igraph-Coloring.html#igraph_is_vertex_coloring).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `colors.len()` differs from the number
    /// of vertices.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let triangle = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, false).unwrap();
    /// assert!(triangle.is_vertex_coloring(&[7, -1, 3]).unwrap());
    /// assert!(!triangle.is_vertex_coloring(&[0, 1, 0]).unwrap());
    /// ```
    pub fn is_vertex_coloring(&self, colors: &[i64]) -> Result<bool> {
        let types = VectorInt::view(colors);
        let mut res = false;
        igraph_call!(igraph_is_vertex_coloring(self, types.as_ptr(), &mut res))?;
        Ok(res)
    }

    /// Checks whether `types` (one boolean per vertex) is a valid bipartite
    /// coloring, and if so reports the orientation of the edges.
    ///
    /// Returns `None` if some edge joins two vertices of the same type
    /// (self-loops are ignored), otherwise `Some(mode)` where, for a directed
    /// graph, `mode` is [`NeighborMode::Out`] when all edges go from `false`
    /// to `true` vertices, [`NeighborMode::In`] when they all go from `true`
    /// to `false`, and [`NeighborMode::All`] when both directions occur. It is
    /// always [`NeighborMode::All`] for undirected graphs, and for directed
    /// graphs without non-loop edges. Time complexity:
    /// O(|E|).
    ///
    /// See also [`bipartite_types`](Self::bipartite_types) to find such a
    /// coloring, and [`is_bipartite`](Self::is_bipartite).
    ///
    /// Binds [`igraph_is_bipartite_coloring`](https://igraph.org/c/html/latest/igraph-Coloring.html#igraph_is_bipartite_coloring).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `types.len()` differs from the number of
    /// vertices.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // Directed edges from "users" (false) to "items" (true).
    /// let g = Graph::from_edges(&[(0, 2), (1, 2), (1, 3)], 4, true).unwrap();
    /// let types = [false, false, true, true];
    /// assert_eq!(g.is_bipartite_coloring(&types).unwrap(), Some(NeighborMode::Out));
    /// assert_eq!(g.is_bipartite_coloring(&[false, true, true, false]).unwrap(), None);
    /// ```
    pub fn is_bipartite_coloring(&self, types: &[bool]) -> Result<Option<NeighborMode>> {
        let types = VectorBool::view(types);
        let mut res = false;
        let mut mode: igraph_neimode_t = igraph_neimode_t_IGRAPH_ALL;
        igraph_call!(igraph_is_bipartite_coloring(
            self,
            types.as_ptr(),
            &mut res,
            &mut mode
        ))?;
        if res {
            Ok(Some(NeighborMode::try_from(mode)?))
        } else {
            Ok(None)
        }
    }

    /// Checks whether `colors` (one integer per edge) is a proper edge
    /// coloring, i.e. no two edges sharing an endpoint have the same color.
    ///
    /// A self-loop is not considered adjacent to itself, so graphs with
    /// self-loops can still be properly edge-colored. Time complexity:
    /// O(|V| d log d), where d is the maximum degree.
    ///
    /// Binds [`igraph_is_edge_coloring`](https://igraph.org/c/html/latest/igraph-Coloring.html#igraph_is_edge_coloring).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `colors.len()` differs from the number
    /// of edges.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // The perfect matchings {01, 23}, {02, 13}, {03, 12} 3-edge-color K4.
    /// let k4 = Graph::from_edges(&[(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)], 4, false)
    ///     .unwrap();
    /// assert!(k4.is_edge_coloring(&[0, 0, 1, 1, 2, 2]).unwrap());
    /// assert!(!k4.is_edge_coloring(&[0, 1, 0, 1, 2, 2]).unwrap());
    /// ```
    pub fn is_edge_coloring(&self, colors: &[i64]) -> Result<bool> {
        let types = VectorInt::view(colors);
        let mut res = false;
        igraph_call!(igraph_is_edge_coloring(self, types.as_ptr(), &mut res))?;
        Ok(res)
    }
}

//! Breadth-first and depth-first traversals, with Rust closures as visitors
//! (`igraph_visitor.h`).
//!
//! This module binds the three graph traversal functions of igraph's
//! [Visitors](https://igraph.org/c/html/latest/igraph-Visitors.html) chapter:
//!
//! | Rust method | C function | What you get |
//! |-------------|------------|--------------|
//! | [`Graph::bfs`] | [`igraph_bfs`](https://igraph.org/c/html/latest/igraph-Visitors.html#igraph_bfs) | [`BfsResult`]: order, rank, parents, pred, succ, dist |
//! | [`Graph::bfs_with`] | `igraph_bfs` + `igraph_bfshandler_t` | same, calling a closure on every visited vertex ([`BfsVisit`]) |
//! | [`Graph::bfs_simple`] | [`igraph_bfs_simple`](https://igraph.org/c/html/latest/igraph-Visitors.html#igraph_bfs_simple) | [`BfsSimpleResult`]: order, distance layers, parents |
//! | [`Graph::dfs`] | [`igraph_dfs`](https://igraph.org/c/html/latest/igraph-Visitors.html#igraph_dfs) | [`DfsResult`]: discovery and finishing orders, parents, dist |
//! | [`Graph::dfs_with`] | `igraph_dfs` + `igraph_dfshandler_t` (in and out) | same, calling a closure on every [`DfsEvent`] |
//!
//! Traversal parameters are grouped in [`BfsOptions`] and [`DfsOptions`]
//! (edge direction to follow, whether to restart from unreachable vertices,
//! and, for BFS, an optional restricted vertex set).
//!
//! # Visitors are closures
//!
//! The callback variants accept any `FnMut` closure returning
//! [`ControlFlow<()>`](std::ops::ControlFlow): return
//! [`ControlFlow::Continue(())`](std::ops::ControlFlow::Continue) to go on,
//! or [`ControlFlow::Break(())`](std::ops::ControlFlow::Break) to stop the
//! traversal early. Stopping is *not* an error (it maps to igraph's
//! `IGRAPH_STOP`): the wrapper returns `Ok` with the partial results computed
//! so far and sets the `stopped` flag of the result. If the closure panics,
//! the traversal is aborted inside igraph (no unwinding crosses the FFI
//! boundary) and the panic is then resumed in the calling Rust code.
//!
//! # Rusty results
//!
//! Where igraph stores negative sentinels (`-1` for "root", `-2` for "not
//! visited"), the result structs use [`Option`]s instead, and the visiting
//! orders only contain the vertices that were actually reached (igraph pads
//! them with `-1`).
//!
//! # Differences from the raw C functions
//!
//! The bindings smooth over a few rough edges of `src/graph/visitors.c`,
//! which are present in igraph 1.0.0 and 1.0.1 (the file is unchanged
//! between the two releases):
//!
//! - `igraph_dfs` reports a wrong depth to its out-callback (one less than
//!   the depth given at discovery) and, after a restart with `unreachable`,
//!   its depth counter is not reset: the roots of later trees still get
//!   depth 0, but the other vertices of the second tree get one less than
//!   their true depth, those of the third tree two less, and so on (the raw
//!   `dist` output can even become `-1`, the "not visited" sentinel).
//!   [`DfsEvent`] and [`DfsResult::dist`] carry the true depths, tracked on
//!   the Rust side.
//! - `igraph_bfs` with `unreachable = true` reads out of bounds on the null
//!   graph: the bindings answer that case without calling igraph.
//! - `igraph_bfs_simple` does not validate its root, and `igraph_dfs`
//!   reports an invalid root as `IGRAPH_EINVAL`: both methods check the
//!   root first and report [`ErrorKind::InvalidVertexId`], like
//!   [`Graph::bfs`].
//! - After an early stop, `igraph_bfs` has already assigned a parent to the
//!   vertices waiting in its queue; [`BfsResult::parents`] only describes
//!   the vertices that were actually visited.
//!
//! # See also
//!
//! Many questions that can be answered with a hand-written traversal have a
//! dedicated (and usually faster) function elsewhere in the crate:
//!
//! - distances and shortest paths: [`Graph::distances`],
//!   [`Graph::get_shortest_path`], [`Graph::get_shortest_paths`],
//!   [`Graph::eccentricity`] (all in [`crate::paths`]);
//! - reachability and components: [`Graph::subcomponent`] (the vertices a
//!   BFS from one root reaches), [`Graph::connected_components`],
//!   [`Graph::neighborhood`] (vertices within a given number of hops);
//! - DAGs and cycles: [`Graph::topological_sorting`], [`Graph::is_dag`],
//!   [`Graph::find_cycle`] (in [`crate::cycles`]);
//! - trees: [`Graph::unfold_tree`] (unrolls a graph into a BFS tree),
//!   [`Graph::kary_tree`] and [`Graph::famous`] to build test inputs;
//! - low-level neighbor access for your own traversals: [`crate::adjlist`].
//!
//! # Example
//!
//! ```
//! use igraph::prelude::*;
//! use igraph::visitor::{BfsOptions, DfsOptions};
//! use std::ops::ControlFlow;
//!
//! // A small binary tree:      0
//! //                         /   \
//! //                        1     2
//! //                       / \   /
//! //                      3   4 5
//! let tree = Graph::kary_tree(6, 2, TreeMode::Undirected)?;
//!
//! let bfs = tree.bfs(&[0], &BfsOptions::default())?;
//! assert_eq!(bfs.order, [0, 1, 2, 3, 4, 5]);
//! assert_eq!(bfs.dist, [Some(0), Some(1), Some(1), Some(2), Some(2), Some(2)]);
//! assert_eq!(bfs.path_to(5), Some(vec![0, 2, 5]));
//!
//! let dfs = tree.dfs(0, &DfsOptions::default())?;
//! assert_eq!(dfs.order, [0, 1, 3, 4, 2, 5]);     // pre-order
//! assert_eq!(dfs.order_out, [3, 4, 1, 5, 2, 0]); // post-order
//!
//! // Stop as soon as a vertex at distance 2 is found.
//! let mut first_deep = None;
//! let partial = tree.bfs_with(&[0], &BfsOptions::default(), |visit| {
//!     if visit.dist == 2 {
//!         first_deep = Some(visit.vid);
//!         return ControlFlow::Break(());
//!     }
//!     ControlFlow::Continue(())
//! })?;
//! assert_eq!(first_deep, Some(3));
//! assert!(partial.stopped);
//!
//! // In an unweighted graph, BFS distances are shortest-path lengths.
//! let d = tree.distances(0, .., None, NeighborMode::All)?;
//! assert_eq!(d.to_rows()[0], [0.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
//! # Ok::<(), igraph::Error>(())
//! ```

use crate::{
    constants::NeighborMode,
    error::{Error, ErrorKind, Result, catch_panic},
    ffi::*,
    graph::{Graph, VertexId},
    igraph_call,
    vector::VectorInt,
};
use std::{ffi::c_void, ops::ControlFlow, ptr};

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Parameters of a breadth-first search ([`Graph::bfs`], [`Graph::bfs_with`]).
///
/// The defaults are: follow out-edges ([`NeighborMode::Out`]), do **not**
/// visit vertices unreachable from the roots, no restriction.
///
/// ```
/// use igraph::prelude::*;
/// use igraph::visitor::BfsOptions;
///
/// let restricted = [0, 1, 2];
/// let opts = BfsOptions::default()
///     .with_mode(NeighborMode::All)
///     .with_unreachable(true)
///     .with_restricted(&restricted);
/// assert_eq!(opts.mode, NeighborMode::All);
/// assert!(opts.unreachable);
/// assert_eq!(opts.restricted, Some(&restricted[..]));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BfsOptions<'a> {
    /// Which edges to follow in directed graphs: [`NeighborMode::Out`] follows
    /// the edge directions, [`NeighborMode::In`] goes against them and
    /// [`NeighborMode::All`] ignores them. Ignored for undirected graphs.
    pub mode: NeighborMode,
    /// If `true`, once the roots are exhausted, further searches are started
    /// from the not yet visited vertices, in increasing id order, until every
    /// (allowed) vertex has been visited.
    pub unreachable: bool,
    /// If set, the search only walks on these vertices: every other vertex
    /// is treated as already visited (even when it is given as a root, in
    /// which case it is silently skipped).
    pub restricted: Option<&'a [VertexId]>,
}

impl Default for BfsOptions<'_> {
    fn default() -> Self {
        Self {
            mode: NeighborMode::Out,
            unreachable: false,
            restricted: None,
        }
    }
}

impl<'a> BfsOptions<'a> {
    /// Sets [`mode`](Self::mode).
    pub fn with_mode(mut self, mode: NeighborMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets [`unreachable`](Self::unreachable).
    pub fn with_unreachable(mut self, unreachable: bool) -> Self {
        self.unreachable = unreachable;
        self
    }

    /// Sets [`restricted`](Self::restricted) to `Some(vertices)`.
    pub fn with_restricted(mut self, vertices: &'a [VertexId]) -> Self {
        self.restricted = Some(vertices);
        self
    }
}

/// Parameters of a depth-first search ([`Graph::dfs`], [`Graph::dfs_with`]).
///
/// The defaults are: follow out-edges ([`NeighborMode::Out`]) and do **not**
/// visit vertices unreachable from the root.
///
/// ```
/// use igraph::prelude::*;
/// use igraph::visitor::DfsOptions;
///
/// // 0 -> 1, 2 -> 1: walking against the edges from 1 finds 0 and 2.
/// let g = Graph::from_edges(&[(0, 1), (2, 1)], 3, true)?;
/// let opts = DfsOptions::default().with_mode(NeighborMode::In);
/// assert_eq!(g.dfs(1, &opts)?.order, [1, 0, 2]);
/// assert_eq!(g.dfs(1, &DfsOptions::default())?.order, [1]);
/// # Ok::<(), igraph::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DfsOptions {
    /// Which edges to follow in directed graphs (see [`BfsOptions::mode`]).
    /// Ignored for undirected graphs.
    pub mode: NeighborMode,
    /// If `true`, once the tree of the root is complete, further searches are
    /// started from the not yet visited vertices, in increasing id order.
    pub unreachable: bool,
}

impl Default for DfsOptions {
    fn default() -> Self {
        Self {
            mode: NeighborMode::Out,
            unreachable: false,
        }
    }
}

impl DfsOptions {
    /// Sets [`mode`](Self::mode).
    pub fn with_mode(mut self, mode: NeighborMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets [`unreachable`](Self::unreachable).
    pub fn with_unreachable(mut self, unreachable: bool) -> Self {
        self.unreachable = unreachable;
        self
    }
}

// ---------------------------------------------------------------------------
// Results and events
// ---------------------------------------------------------------------------

/// What a breadth-first search visitor sees each time a vertex is visited
/// (the arguments of igraph's `igraph_bfshandler_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BfsVisit {
    /// The vertex being visited.
    pub vid: VertexId,
    /// The vertex visited just before, or `None` if `vid` is the root of a
    /// search tree.
    pub pred: Option<VertexId>,
    /// The vertex that will be visited next, or `None` if `vid` is the last
    /// vertex of its search tree.
    pub succ: Option<VertexId>,
    /// The rank of `vid`, i.e. its position in the visiting order (from 0).
    pub rank: usize,
    /// The distance (number of hops) of `vid` from the root of its search
    /// tree.
    pub dist: usize,
}

/// Results of a breadth-first search ([`Graph::bfs`], [`Graph::bfs_with`]).
///
/// The per-vertex vectors (`rank`, `parents`, `pred`, `succ`, `dist`) have
/// one entry per vertex of the graph, `None` for vertices that were not
/// visited (e.g. unreachable, outside the restricted set, or not reached
/// because the visitor stopped the search).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BfsResult {
    /// The visited vertices, in visiting order.
    pub order: Vec<VertexId>,
    /// The rank (position in [`order`](Self::order)) of each vertex.
    pub rank: Vec<Option<usize>>,
    /// The parent of each vertex in the BFS forest: `None` for the roots of
    /// the search trees and for unvisited vertices (igraph itself already
    /// assigns a parent to vertices that were queued but not yet visited when
    /// the visitor stopped the search; the bindings report them as `None`
    /// too, so that `parents` always describes the visited forest).
    pub parents: Vec<Option<VertexId>>,
    /// The vertex visited just before each vertex: `None` for the roots of
    /// the search trees and for unvisited vertices.
    pub pred: Vec<Option<VertexId>>,
    /// The vertex visited just after each vertex: `None` for the last vertex
    /// of each search tree and for unvisited vertices. When the visitor stops
    /// the search, the successor of the vertex it stopped at is `None` too.
    pub succ: Vec<Option<VertexId>>,
    /// The distance of each vertex from the root of its search tree.
    pub dist: Vec<Option<usize>>,
    /// `true` if the visitor closure returned
    /// [`ControlFlow::Break`], so that the search ended early.
    pub stopped: bool,
}

impl BfsResult {
    /// Whether vertex `v` was visited by the search.
    pub fn is_visited(&self, v: VertexId) -> bool {
        usize::try_from(v)
            .ok()
            .and_then(|i| self.rank.get(i))
            .is_some_and(Option::is_some)
    }

    /// The roots of the search trees, in the order they were used.
    ///
    /// In an undirected graph searched with [`BfsOptions::unreachable`] set
    /// (and no restriction), there is exactly one root per connected
    /// component, see [`Graph::connected_components`].
    pub fn roots(&self) -> Vec<VertexId> {
        self.order
            .iter()
            .copied()
            .filter(|&v| self.parents[v as usize].is_none())
            .collect()
    }

    /// The path of the BFS forest from the root of `v`'s search tree down to
    /// `v` (a shortest path in unweighted graphs), or `None` if `v` was not
    /// visited. [`Graph::get_shortest_path`] computes a single such path
    /// directly (also with weights).
    pub fn path_to(&self, v: VertexId) -> Option<Vec<VertexId>> {
        tree_path(&self.parents, &self.rank, v)
    }
}

/// Results of [`Graph::bfs_simple`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BfsSimpleResult {
    /// The visited vertices, in visiting order (only the ones reachable from
    /// the root).
    pub order: Vec<VertexId>,
    /// Layer boundaries: the vertices at distance `i` from the root are
    /// `order[layers[i]..layers[i + 1]]`. It has one more element than the
    /// number of layers; the last one is `order.len()`.
    pub layers: Vec<usize>,
    /// The parent of each vertex in the BFS tree: `None` for the root and for
    /// vertices that were not reached.
    pub parents: Vec<Option<VertexId>>,
}

impl BfsSimpleResult {
    /// Number of distance layers (the eccentricity of the root plus one).
    pub fn num_layers(&self) -> usize {
        self.layers.len().saturating_sub(1)
    }

    /// The vertices at distance `i` from the root (empty if `i` is too large).
    pub fn layer(&self, i: usize) -> &[VertexId] {
        match (self.layers.get(i), self.layers.get(i + 1)) {
            (Some(&a), Some(&b)) => &self.order[a..b],
            _ => &[],
        }
    }

    /// Iterates over the distance layers, from the root outwards.
    pub fn iter_layers(&self) -> impl Iterator<Item = &[VertexId]> + '_ {
        self.layers.windows(2).map(|w| &self.order[w[0]..w[1]])
    }
}

/// An event reported to a depth-first search visitor ([`Graph::dfs_with`]).
///
/// igraph has two DFS callbacks: one when a vertex is *discovered*
/// (`in_callback`) and one when its subtree is *finished* (`out_callback`).
/// A single Rust closure receives both, as the two variants of this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DfsEvent {
    /// A vertex has just been discovered (pre-order).
    Discover {
        /// The discovered vertex.
        vid: VertexId,
        /// Its distance (depth) from the root of its search tree.
        dist: usize,
    },
    /// The whole subtree of a vertex has been explored (post-order).
    Finish {
        /// The finished vertex.
        vid: VertexId,
        /// Its distance (depth) from the root of its search tree: the same
        /// value as in the matching [`Discover`](DfsEvent::Discover) event
        /// (the raw C callback receives one less).
        dist: usize,
    },
}

impl DfsEvent {
    /// The vertex the event is about.
    pub fn vid(&self) -> VertexId {
        match *self {
            Self::Discover { vid, .. } | Self::Finish { vid, .. } => vid,
        }
    }

    /// The depth of the vertex in its search tree.
    pub fn dist(&self) -> usize {
        match *self {
            Self::Discover { dist, .. } | Self::Finish { dist, .. } => dist,
        }
    }
}

/// Results of a depth-first search ([`Graph::dfs`], [`Graph::dfs_with`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DfsResult {
    /// The vertices in the order they were discovered (pre-order).
    pub order: Vec<VertexId>,
    /// The vertices in the order their subtrees were completed (post-order).
    /// If the search was stopped early, it may be shorter than
    /// [`order`](Self::order).
    pub order_out: Vec<VertexId>,
    /// The parent of each vertex in the DFS forest: `None` for the roots of
    /// the search trees and for unvisited vertices.
    pub parents: Vec<Option<VertexId>>,
    /// The depth of each vertex in its DFS tree (the number of tree edges
    /// between it and the root of its tree), `None` if not visited. Unlike
    /// the raw output of `igraph_dfs`, it stays exact after a restart with
    /// [`DfsOptions::unreachable`] (see the [module docs](self)).
    pub dist: Vec<Option<usize>>,
    /// `true` if the visitor closure returned [`ControlFlow::Break`].
    pub stopped: bool,
}

impl DfsResult {
    /// Whether vertex `v` was discovered by the search.
    pub fn is_visited(&self, v: VertexId) -> bool {
        usize::try_from(v)
            .ok()
            .and_then(|i| self.dist.get(i))
            .is_some_and(Option::is_some)
    }

    /// The path of the DFS forest from the root of `v`'s tree down to `v`,
    /// or `None` if `v` was not visited.
    pub fn path_to(&self, v: VertexId) -> Option<Vec<VertexId>> {
        tree_path(&self.parents, &self.dist, v)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Follows `parents` from `v` up to a root; `visited[v]` tells whether `v`
/// belongs to the forest at all.
fn tree_path<T>(
    parents: &[Option<VertexId>],
    visited: &[Option<T>],
    v: VertexId,
) -> Option<Vec<VertexId>> {
    let i = usize::try_from(v).ok()?;
    visited.get(i)?.as_ref()?;
    let mut path = vec![v];
    let mut cur = v;
    while let Some(p) = parents[cur as usize] {
        path.push(p);
        cur = p;
        if path.len() > parents.len() {
            return None; // defensive: never loop on a malformed forest
        }
    }
    path.reverse();
    Some(path)
}

/// Converts an igraph id vector with negative sentinels into `Option`s.
fn opt_ids(v: VectorInt) -> Vec<Option<VertexId>> {
    v.iter().map(|&x| (x >= 0).then_some(x)).collect()
}

/// Converts an igraph count vector with negative sentinels into `Option`s.
fn opt_counts(v: VectorInt) -> Vec<Option<usize>> {
    v.iter().map(|&x| usize::try_from(x).ok()).collect()
}

/// Keeps the visited prefix of an order vector padded with `-1`.
fn visited_prefix(v: VectorInt) -> Vec<VertexId> {
    v.iter().copied().take_while(|&x| x >= 0).collect()
}

fn check_vertex(graph: &Graph, v: VertexId, what: &str) -> Result<()> {
    if v < 0 || v as usize >= graph.vcount() {
        return Err(Error::new(
            ErrorKind::InvalidVertexId,
            format!(
                "invalid {what} vertex id {v} (the graph has {} vertices)",
                graph.vcount()
            ),
        ));
    }
    Ok(())
}

fn flow_to_code(flow: ControlFlow<()>, stopped: &mut bool) -> igraph_error_t {
    match flow {
        ControlFlow::Continue(()) => igraph_error_type_t_IGRAPH_SUCCESS,
        ControlFlow::Break(()) => {
            *stopped = true;
            igraph_error_type_t_IGRAPH_STOP
        }
    }
}

/// Runs the Rust side of a BFS/DFS callback in a new level of igraph's
/// "finally" stack (`IGRAPH_FINALLY_ENTER` / `IGRAPH_FINALLY_EXIT`), catching
/// panics.
///
/// The running search keeps its temporaries (queue, stack, bitsets) on that
/// stack, and igraph's error handler frees the objects of the current level
/// when a call fails. Without a new level, an igraph call made by the user
/// closure that fails (returning an `Err` the closure may well ignore) would
/// free the temporaries of the search that is still running, a
/// use-after-free in C.
fn in_finally_level(f: impl FnOnce() -> igraph_error_t) -> igraph_error_t {
    // SAFETY: plain bookkeeping on igraph's thread-local finally stack; the
    // matching EXIT runs below, as `catch_panic` never unwinds.
    unsafe { IGRAPH_FINALLY_ENTER() };
    let code = catch_panic(f);
    // SAFETY: closes the level opened above. Every igraph call made by the
    // closure has returned, and a failed one has already freed its objects.
    unsafe { IGRAPH_FINALLY_EXIT() };
    code
}

struct BfsState<F> {
    f: F,
    stopped: bool,
}

unsafe extern "C" fn bfs_trampoline<F>(
    _graph: *const igraph_t,
    vid: igraph_int_t,
    pred: igraph_int_t,
    succ: igraph_int_t,
    rank: igraph_int_t,
    dist: igraph_int_t,
    extra: *mut c_void,
) -> igraph_error_t
where
    F: FnMut(BfsVisit) -> ControlFlow<()>,
{
    in_finally_level(|| {
        // SAFETY: `extra` is the `&mut BfsState<F>` passed by `bfs_impl`,
        // alive and exclusively borrowed for the whole C call.
        let state = unsafe { &mut *(extra as *mut BfsState<F>) };
        let visit = BfsVisit {
            vid,
            pred: (pred >= 0).then_some(pred),
            succ: (succ >= 0).then_some(succ),
            rank: rank as usize,
            dist: dist as usize,
        };
        let flow = (state.f)(visit);
        flow_to_code(flow, &mut state.stopped)
    })
}

/// State shared by the two DFS trampolines. igraph's own `dist` argument to
/// the out-callback is off by one, and both callbacks' `dist` drift after a
/// restart with
/// `unreachable` (igraph 1.0.0 and 1.0.1: `act_dist` is never reset for a new
/// root), so the depth is tracked here: `depth` is the size of the DFS
/// stack, incremented on discovery and decremented on completion.
struct DfsState<F> {
    f: F,
    depth: usize,
    stopped: bool,
}

unsafe extern "C" fn dfs_in_trampoline<F>(
    _graph: *const igraph_t,
    vid: igraph_int_t,
    _dist: igraph_int_t,
    extra: *mut c_void,
) -> igraph_error_t
where
    F: FnMut(DfsEvent) -> ControlFlow<()>,
{
    in_finally_level(|| {
        // SAFETY: see `bfs_trampoline`.
        let state = unsafe { &mut *(extra as *mut DfsState<F>) };
        let dist = state.depth;
        state.depth += 1;
        let flow = (state.f)(DfsEvent::Discover { vid, dist });
        flow_to_code(flow, &mut state.stopped)
    })
}

unsafe extern "C" fn dfs_out_trampoline<F>(
    _graph: *const igraph_t,
    vid: igraph_int_t,
    _dist: igraph_int_t,
    extra: *mut c_void,
) -> igraph_error_t
where
    F: FnMut(DfsEvent) -> ControlFlow<()>,
{
    in_finally_level(|| {
        // SAFETY: see `bfs_trampoline`.
        let state = unsafe { &mut *(extra as *mut DfsState<F>) };
        state.depth = state.depth.saturating_sub(1);
        let dist = state.depth;
        let flow = (state.f)(DfsEvent::Finish { vid, dist });
        flow_to_code(flow, &mut state.stopped)
    })
}

// ---------------------------------------------------------------------------
// Graph methods
// ---------------------------------------------------------------------------

impl igraph_t {
    fn bfs_impl(
        &self,
        roots: &[VertexId],
        options: &BfsOptions<'_>,
        callback: igraph_bfshandler_t,
        extra: *mut c_void,
    ) -> Result<BfsResult> {
        if self.vcount() == 0 {
            // igraph 1.0.0 and 1.0.1 read out of bounds when `unreachable` is
            // set on the null graph (`IGRAPH_BIT_TEST(added, 0)` on an empty
            // bitset): handle it here (any given id is invalid).
            if let Some(&v) = roots.iter().chain(options.restricted.unwrap_or(&[])).next() {
                check_vertex(self, v, "root or restricted")?;
            }
            return Ok(BfsResult {
                order: vec![],
                rank: vec![],
                parents: vec![],
                pred: vec![],
                succ: vec![],
                dist: vec![],
                stopped: false,
            });
        }
        let roots_v = VectorInt::view(roots);
        let restricted_v = options.restricted.map(VectorInt::view);
        let restricted_p = restricted_v.as_ref().map_or(ptr::null(), |v| v.as_ptr());
        let mut order = VectorInt::new();
        let mut rank = VectorInt::new();
        let mut parents = VectorInt::new();
        let mut pred = VectorInt::new();
        let mut succ = VectorInt::new();
        let mut dist = VectorInt::new();
        igraph_call!(igraph_bfs(
            self,
            0,
            roots_v.as_ptr(),
            options.mode.into(),
            options.unreachable,
            restricted_p,
            &mut order,
            &mut rank,
            &mut parents,
            &mut pred,
            &mut succ,
            &mut dist,
            callback,
            extra,
        ))?;
        let rank = opt_counts(rank);
        // igraph assigns `parents` when a vertex is *enqueued*, not when it is
        // visited: after an early stop, vertices still waiting in the queue
        // would have a parent but no rank/dist. Keep the documented invariant
        // "parent is `Some` only for visited, non-root vertices".
        let parents = opt_ids(parents)
            .into_iter()
            .zip(&rank)
            .map(|(p, r)| r.and(p))
            .collect();
        Ok(BfsResult {
            order: visited_prefix(order),
            rank,
            parents,
            pred: opt_ids(pred),
            succ: opt_ids(succ),
            dist: opt_counts(dist),
            stopped: false,
        })
    }

    /// Breadth-first search from one or more root vertices.
    ///
    /// The search starts from `roots[0]`; when its tree is exhausted, it
    /// continues from the next root in `roots` that was not visited yet, and
    /// so on (roots already reached from a previous one are skipped). If
    /// [`options.unreachable`](BfsOptions::unreachable) is `true`, the
    /// remaining vertices are then used as roots too, in increasing id order,
    /// so that the whole graph is traversed. Neighbors are enqueued in the
    /// order of the graph's adjacency lists (increasing neighbor id). An
    /// empty `roots` slice is allowed: nothing is visited unless
    /// `unreachable` is set.
    ///
    /// The result gathers every output of the C function: the visiting
    /// order, and, for each vertex, its rank, BFS-tree parent, predecessor
    /// and successor in the visiting order, and distance from its root. See
    /// [`BfsResult`]. Use [`Graph::bfs_with`] to run a visitor closure during
    /// the search, and [`Graph::bfs_simple`] for a lighter single-root
    /// variant that also reports distance layers.
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// Binds [`igraph_bfs`](https://igraph.org/c/html/latest/igraph-Visitors.html#igraph_bfs).
    ///
    /// See also [`Graph::subcomponent`] (just the set of reachable
    /// vertices), [`Graph::distances`] and [`Graph::get_shortest_paths`]
    /// (weighted distances and paths), [`Graph::connected_components`], and
    /// [`Graph::unfold_tree`] (turns the graph into its BFS tree).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] if a root or a restricted vertex does
    /// not exist.
    ///
    /// # Examples
    ///
    /// Two disjoint 4-cycles `0-1-2-3` and `4-5-6-7`:
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::visitor::BfsOptions;
    ///
    /// let square = Graph::ring(4, false, false, true)?;
    /// let g = square.disjoint_union(&square)?;
    /// let r = g.bfs(&[0], &BfsOptions::default())?;
    /// assert_eq!(r.order, [0, 1, 3, 2]);
    /// assert_eq!(r.dist[2], Some(2));
    /// assert!(!r.is_visited(5));
    ///
    /// let all = g.bfs(&[0], &BfsOptions::default().with_unreachable(true))?;
    /// assert_eq!(all.order, [0, 1, 3, 2, 4, 5, 7, 6]);
    /// // One search tree per connected component.
    /// assert_eq!(all.roots(), [0, 4]);
    /// assert_eq!(g.connected_components(Connectedness::Weak)?.count, 2);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn bfs(&self, roots: &[VertexId], options: &BfsOptions<'_>) -> Result<BfsResult> {
        self.bfs_impl(roots, options, None, ptr::null_mut())
    }

    /// Breadth-first search calling a visitor closure on every visited vertex.
    ///
    /// Same traversal as [`Graph::bfs`]; in addition, `visitor` is called
    /// each time a vertex is *visited* (dequeued), after all its unvisited
    /// neighbors have been enqueued, with a [`BfsVisit`] describing the
    /// vertex, its predecessor and successor in the visiting order, its rank
    /// and its distance from the root.
    ///
    /// Return [`ControlFlow::Continue`] to go on or [`ControlFlow::Break`] to
    /// stop: the search then ends normally, the returned [`BfsResult`] has
    /// [`stopped`](BfsResult::stopped) set, and contains the values computed
    /// so far (the vertices not yet visited are `None`). Note that, when
    /// stopping, the vertex the visitor stopped at has already been visited
    /// (it is in `order`), but its `succ` entry is `None`. If `visitor`
    /// panics, the search is aborted and the panic resumes in the caller.
    ///
    /// Binds [`igraph_bfs`](https://igraph.org/c/html/latest/igraph-Visitors.html#igraph_bfs)
    /// with an [`igraph_bfshandler_t`](https://igraph.org/c/html/latest/igraph-Visitors.html#igraph_bfshandler_t)
    /// callback.
    ///
    /// See also [`Graph::dfs_with`] for the depth-first counterpart.
    ///
    /// # Errors
    ///
    /// Same as [`Graph::bfs`].
    ///
    /// # Examples
    ///
    /// Find the first vertex of a path that is at least three hops away,
    /// without visiting the rest of the graph:
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::visitor::BfsOptions;
    /// use std::ops::ControlFlow;
    ///
    /// let path = Graph::ring(6, false, false, false)?; // 0 - 1 - 2 - 3 - 4 - 5
    /// let mut seen = vec![];
    /// let r = path.bfs_with(&[0], &BfsOptions::default(), |v| {
    ///     seen.push(v.vid);
    ///     if v.dist >= 3 { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
    /// })?;
    /// assert_eq!(seen, [0, 1, 2, 3]);
    /// assert!(r.stopped);
    /// assert_eq!(r.order, [0, 1, 2, 3]);
    /// assert_eq!(r.rank[5], None);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn bfs_with<F>(
        &self,
        roots: &[VertexId],
        options: &BfsOptions<'_>,
        visitor: F,
    ) -> Result<BfsResult>
    where
        F: FnMut(BfsVisit) -> ControlFlow<()>,
    {
        let mut state = BfsState {
            f: visitor,
            stopped: false,
        };
        let extra = &mut state as *mut BfsState<F> as *mut c_void;
        let mut result = self.bfs_impl(roots, options, Some(bfs_trampoline::<F>), extra)?;
        result.stopped = state.stopped;
        Ok(result)
    }

    /// Simple single-source breadth-first search, reporting distance layers.
    ///
    /// Visits the vertices reachable from `root` (following edges according
    /// to `mode` in directed graphs; `mode` is ignored for undirected graphs)
    /// and returns the visiting order, the *layers* (vertices grouped by
    /// their distance from `root`) and the BFS-tree parents. This is the
    /// lighter alternative to [`Graph::bfs`] when only these outputs matter.
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// Binds [`igraph_bfs_simple`](https://igraph.org/c/html/latest/igraph-Visitors.html#igraph_bfs_simple).
    ///
    /// See also [`Graph::neighborhood`] (the vertices within a number of
    /// hops, possibly from several roots) and [`Graph::eccentricity`] (which,
    /// computed with the same `mode`, is `num_layers() - 1`: igraph ignores
    /// unreachable vertices there too).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] if `root` does not exist (checked on
    /// the Rust side: igraph 1.0.0 and 1.0.1 do not validate it).
    ///
    /// # Examples
    ///
    /// The layers of a complete binary tree with 7 vertices:
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// let t = Graph::kary_tree(7, 2, TreeMode::Undirected)?;
    /// let r = t.bfs_simple(0, NeighborMode::All)?;
    /// assert_eq!(r.num_layers(), 3);
    /// assert_eq!(r.layer(0), [0]);
    /// assert_eq!(r.layer(1), [1, 2]);
    /// assert_eq!(r.layer(2), [3, 4, 5, 6]);
    /// assert_eq!(r.parents[4], Some(1));
    /// // The eccentricity of the root is the index of the last layer.
    /// assert_eq!(t.eccentricity(0, None, NeighborMode::All)?, [2.0]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn bfs_simple(&self, root: VertexId, mode: NeighborMode) -> Result<BfsSimpleResult> {
        check_vertex(self, root, "root")?;
        let mut order = VectorInt::new();
        let mut layers = VectorInt::new();
        let mut parents = VectorInt::new();
        igraph_call!(igraph_bfs_simple(
            self,
            root,
            mode.into(),
            &mut order,
            &mut layers,
            &mut parents,
        ))?;
        Ok(BfsSimpleResult {
            order: order.into(),
            layers: layers.iter().map(|&x| x as usize).collect(),
            parents: opt_ids(parents),
        })
    }

    fn dfs_impl(
        &self,
        root: VertexId,
        options: &DfsOptions,
        callbacks: (igraph_dfshandler_t, igraph_dfshandler_t),
        extra: *mut c_void,
    ) -> Result<DfsResult> {
        check_vertex(self, root, "root")?;
        let mut order = VectorInt::new();
        let mut order_out = VectorInt::new();
        let mut parents = VectorInt::new();
        igraph_call!(igraph_dfs(
            self,
            root,
            options.mode.into(),
            options.unreachable,
            &mut order,
            &mut order_out,
            &mut parents,
            ptr::null_mut(),
            callbacks.0,
            callbacks.1,
            extra,
        ))?;
        let order = visited_prefix(order);
        let parents = opt_ids(parents);
        // igraph's `dist` output drifts after a restart with `unreachable`
        // (igraph 1.0.0 and 1.0.1); recompute it from the parents, which are
        // always exact. Parents are
        // discovered before their children, so one pass in `order` suffices.
        let mut dist: Vec<Option<usize>> = vec![None; self.vcount()];
        for &v in &order {
            dist[v as usize] = Some(match parents[v as usize] {
                Some(p) => dist[p as usize].map_or(0, |d| d + 1),
                None => 0,
            });
        }
        Ok(DfsResult {
            order,
            order_out: visited_prefix(order_out),
            parents,
            dist,
            stopped: false,
        })
    }

    /// Depth-first search from a root vertex.
    ///
    /// Explores the graph depth first from `root`, following edges according
    /// to [`options.mode`](DfsOptions::mode) in directed graphs; neighbors
    /// are tried in adjacency list order (increasing neighbor id). If
    /// [`options.unreachable`](DfsOptions::unreachable) is set, further
    /// searches are started from the unvisited vertices, in
    /// increasing id order, until all vertices are visited.
    ///
    /// Returns the discovery order (pre-order), the completion order
    /// (post-order), the DFS-forest parents and the depth of each vertex,
    /// see [`DfsResult`]. Use [`Graph::dfs_with`] to react to discovery and
    /// completion events while the search runs.
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// Binds [`igraph_dfs`](https://igraph.org/c/html/latest/igraph-Visitors.html#igraph_dfs).
    ///
    /// See also [`Graph::topological_sorting`], [`Graph::is_dag`] and
    /// [`Graph::find_cycle`] (in [`crate::cycles`]), which answer the most
    /// common DFS questions directly, and [`Graph::biconnected_components`].
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] if `root` does not exist (checked on
    /// the Rust side; igraph 1.0.0 and 1.0.1 would report
    /// [`ErrorKind::InvalidValue`]). In particular, the graph must have at
    /// least one vertex.
    ///
    /// # Examples
    ///
    /// A reverse post-order of a DAG is a topological order:
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::visitor::DfsOptions;
    ///
    /// // 0 → 1 → 3, 0 → 2 → 3
    /// let dag = Graph::from_edges(&[(0, 1), (0, 2), (1, 3), (2, 3)], 4, true)?;
    /// let r = dag.dfs(0, &DfsOptions::default())?;
    /// assert_eq!(r.order, [0, 1, 3, 2]);
    /// assert_eq!(r.order_out, [3, 1, 2, 0]);
    /// let topo: Vec<_> = r.order_out.iter().rev().copied().collect();
    /// assert_eq!(topo, [0, 2, 1, 3]);
    /// assert_eq!(r.dist[3], Some(2));
    /// // `topological_sorting` finds another valid order (Kahn's algorithm).
    /// assert_eq!(dag.topological_sorting(NeighborMode::Out)?, [0, 1, 2, 3]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn dfs(&self, root: VertexId, options: &DfsOptions) -> Result<DfsResult> {
        self.dfs_impl(root, options, (None, None), ptr::null_mut())
    }

    /// Depth-first search calling a visitor closure on discovery and
    /// completion of every vertex.
    ///
    /// Same traversal as [`Graph::dfs`]; in addition `visitor` receives a
    /// [`DfsEvent::Discover`] when a vertex is first reached and a
    /// [`DfsEvent::Finish`] when its whole subtree has been explored (the
    /// two C callbacks `in_callback` and `out_callback`, merged in a single
    /// closure so that they can share mutable state). Both events carry the
    /// depth of the vertex in its DFS tree.
    ///
    /// Return [`ControlFlow::Break`] to stop the search: the call still
    /// returns `Ok` with the partial results and
    /// [`stopped`](DfsResult::stopped) set. If `visitor` panics, the search
    /// is aborted and the panic resumes in the caller.
    ///
    /// Binds [`igraph_dfs`](https://igraph.org/c/html/latest/igraph-Visitors.html#igraph_dfs)
    /// with two [`igraph_dfshandler_t`](https://igraph.org/c/html/latest/igraph-Visitors.html#igraph_dfshandler_t)
    /// callbacks.
    ///
    /// See also [`Graph::bfs_with`] for the breadth-first counterpart.
    ///
    /// # Errors
    ///
    /// Same as [`Graph::dfs`].
    ///
    /// # Examples
    ///
    /// Print a tree as an indented outline, using the discovery events:
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::visitor::{DfsEvent, DfsOptions};
    /// use std::ops::ControlFlow;
    ///
    /// let t = Graph::from_edges(&[(0, 1), (0, 2), (1, 3)], 4, false)?;
    /// let mut outline = String::new();
    /// t.dfs_with(0, &DfsOptions::default(), |e| {
    ///     if let DfsEvent::Discover { vid, dist } = e {
    ///         outline += &format!("{}{vid}\n", "  ".repeat(dist));
    ///     }
    ///     ControlFlow::Continue(())
    /// })?;
    /// assert_eq!(outline, "0\n  1\n    3\n  2\n");
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn dfs_with<F>(&self, root: VertexId, options: &DfsOptions, visitor: F) -> Result<DfsResult>
    where
        F: FnMut(DfsEvent) -> ControlFlow<()>,
    {
        let mut state = DfsState {
            f: visitor,
            depth: 0,
            stopped: false,
        };
        let extra = &mut state as *mut DfsState<F> as *mut c_void;
        let callbacks = (
            Some(dfs_in_trampoline::<F> as _),
            Some(dfs_out_trampoline::<F> as _),
        );
        let mut result = self.dfs_impl(root, options, callbacks, extra)?;
        result.stopped = state.stopped;
        Ok(result)
    }
}

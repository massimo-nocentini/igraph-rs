//! Graph cycles: acyclicity, topological orders, feedback sets, cycle
//! enumeration, cycle bases and Eulerian paths.
//!
//! This module binds the C headers `igraph_cycles.h` and `igraph_eulerian.h`,
//! documented in the [Graph cycles](https://igraph.org/c/html/latest/igraph-Cycles.html)
//! chapter of the igraph C manual. All functions are methods of [`Graph`].
//!
//! | Task | Method | Result |
//! |------|--------|--------|
//! | Is the graph a directed acyclic graph? | [`Graph::is_dag`] | `bool` |
//! | Order the vertices of a DAG | [`Graph::topological_sorting`] | `Vec<VertexId>` |
//! | Find *one* cycle, if any | [`Graph::find_cycle`] | `Option<`[`Cycle`]`>` |
//! | List all simple cycles | [`Graph::simple_cycles`] | `Vec<`[`Cycle`]`>` |
//! | Visit simple cycles lazily, with early stop | [`Graph::simple_cycles_callback`] | `()` |
//! | Fundamental cycle basis (from a BFS tree) | [`Graph::fundamental_cycles`] | `Vec<Vec<EdgeId>>` |
//! | Minimum weight cycle basis (Horton), see [`MinimumCycleBasisOptions`] | [`Graph::minimum_cycle_basis`] | `Vec<Vec<EdgeId>>` |
//! | Edges whose removal breaks all cycles | [`Graph::feedback_arc_set`] | `Vec<EdgeId>` |
//! | Vertices whose removal breaks all cycles | [`Graph::feedback_vertex_set`] | `Vec<VertexId>` |
//! | Does an Eulerian path / cycle exist? | [`Graph::is_eulerian`] | [`EulerianStatus`] |
//! | Find an Eulerian path | [`Graph::eulerian_path`] | [`EulerianWalk`] |
//! | Find an Eulerian cycle | [`Graph::eulerian_cycle`] | [`EulerianWalk`] |
//!
//! Cycles are described both by their vertices and by their edges (see
//! [`Cycle`]): with multi-edges, the vertex sequence alone would be ambiguous.
//! Cycle bases are returned as edge-id lists only, as igraph does.
//!
//! Some functions ([`Graph::simple_cycles`], [`Graph::simple_cycles_callback`],
//! [`Graph::fundamental_cycles`] and [`Graph::minimum_cycle_basis`]) are marked
//! *experimental* in igraph 1.0 (both 1.0.0 and 1.0.1): their behavior may
//! change in future releases of the C library. None of the C sources bound here
//! changed between igraph 1.0.0 and 1.0.1.
//!
//! # See also
//!
//! Related functionality in other modules:
//!
//! - [`Graph::is_acyclic`], [`Graph::is_forest`] and [`Graph::is_tree`]
//!   (structural properties): acyclicity tests that, unlike [`Graph::is_dag`],
//!   also apply to undirected graphs.
//! - [`Graph::girth`] and [`Graph::girth_with_cycle`]: the length of (and a
//!   vertex list for) a *shortest* cycle, where [`Graph::find_cycle`] returns an
//!   arbitrary one.
//! - [`Graph::list_triangles`] and [`Graph::count_triangles`]: the cycles of
//!   length 3, faster than [`Graph::simple_cycles`] with a length bound.
//! - [`Graph::get_all_simple_paths`]: simple *paths* instead of cycles.
//! - [`Graph::minimum_spanning_tree`] and [`Graph::connected_components`]: the
//!   complement of a spanning forest (of a *maximum* weight one, if weighted)
//!   is a minimum feedback arc set of an undirected graph, and the cycle space
//!   has dimension |E| - |V| + #components.
//! - [`Graph::transitive_closure`] and [`Graph::dfs`]: reachability and the
//!   depth-first search underlying topological orders.
//! - [`Graph::de_bruijn`] and [`Graph::kautz`]: balanced directed graphs, all
//!   of which have Eulerian cycles.
//!
//! # Example
//!
//! Scheduling tasks with dependencies: a topological order exists exactly
//! when the dependency graph has no cycle.
//!
//! ```
//! use igraph::prelude::*;
//!
//! // 0: fetch, 1: configure, 2: build, 3: test, 4: package
//! let mut deps = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (2, 4), (3, 4)], 5, true)?;
//! assert!(deps.is_dag()?);
//! assert_eq!(deps.topological_sorting(NeighborMode::Out)?, vec![0, 1, 2, 3, 4]);
//!
//! // Someone adds a circular dependency "package -> configure" ...
//! deps.add_edge(4, 1)?;
//! assert!(!deps.is_dag()?);
//! let cycle = deps.find_cycle(NeighborMode::Out)?.expect("there is a cycle now");
//! assert!(cycle.vertices.contains(&4) && cycle.vertices.contains(&1));
//! // ... and removing the edges of a feedback arc set fixes it again.
//! let fas = deps.feedback_arc_set(None, FasAlgorithm::ExactIp)?;
//! assert_eq!(fas.len(), 1);
//! deps.delete_edges(&fas)?;
//! assert!(deps.is_dag()?);
//! # Ok::<(), igraph::Error>(())
//! ```

use crate::{
    constants::{FasAlgorithm, FvsAlgorithm, NeighborMode},
    error::{Error, Result, catch_panic},
    ffi::*,
    graph::{EdgeId, VertexId},
    igraph_call,
    list::VectorIntList,
    vector::{Vector, VectorInt},
};
use std::{ffi::c_void, ops::ControlFlow};

#[cfg(doc)]
use crate::graph::Graph;

/// A cycle (closed walk) of a graph, given both as vertices and edges.
///
/// `edges[i]` connects `vertices[i]` and `vertices[(i + 1) % len]`: the first
/// vertex is *not* repeated at the end, so `vertices.len() == edges.len()`,
/// which is the length of the cycle. A self-loop is a cycle of length 1, and
/// two parallel edges form a cycle of length 2 (when they can be traversed in
/// opposite directions: undirected edges, mutual directed edges, or any
/// directed edges with `NeighborMode::All`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Cycle {
    /// The vertices of the cycle, in traversal order.
    pub vertices: Vec<VertexId>,
    /// The edges of the cycle, in traversal order.
    pub edges: Vec<EdgeId>,
}

impl Cycle {
    /// The length of the cycle, i.e. its number of edges.
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Whether the cycle has no edges (never the case for cycles returned by
    /// this module).
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

/// Options for [`Graph::simple_cycles`] and [`Graph::simple_cycles_callback`].
///
/// The default searches for all simple cycles of any length, following edge
/// directions (`NeighborMode::Out`) in directed graphs. Lengths count edges,
/// which for simple cycles equals the number of vertices.
///
/// ```
/// use igraph::{cycles::SimpleCyclesOptions, prelude::*};
/// let opts = SimpleCyclesOptions::default().with_max_length(4).with_max_results(10);
/// assert_eq!(opts.max_cycle_length, Some(4));
/// assert_eq!(opts.mode, NeighborMode::Out);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimpleCyclesOptions {
    /// How edge directions are considered in directed graphs: `Out` follows
    /// them, `In` follows them backwards, `All` ignores them. Ignored for
    /// undirected graphs. Default: `Out`.
    pub mode: NeighborMode,
    /// Only report cycles with at least this many edges (`None`: no limit).
    pub min_cycle_length: Option<usize>,
    /// Only report cycles with at most this many edges (`None`: no limit).
    /// Bounding the length also bounds the search, making it much faster.
    pub max_cycle_length: Option<usize>,
    /// Stop after this many cycles have been reported (`None`: no limit).
    pub max_results: Option<usize>,
}

impl Default for SimpleCyclesOptions {
    fn default() -> Self {
        Self {
            mode: NeighborMode::Out,
            min_cycle_length: None,
            max_cycle_length: None,
            max_results: None,
        }
    }
}

impl SimpleCyclesOptions {
    /// Sets [`mode`](Self::mode).
    pub fn with_mode(mut self, mode: NeighborMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets [`min_cycle_length`](Self::min_cycle_length).
    pub fn with_min_length(mut self, len: usize) -> Self {
        self.min_cycle_length = Some(len);
        self
    }

    /// Sets [`max_cycle_length`](Self::max_cycle_length).
    pub fn with_max_length(mut self, len: usize) -> Self {
        self.max_cycle_length = Some(len);
        self
    }

    /// Sets [`max_results`](Self::max_results).
    pub fn with_max_results(mut self, n: usize) -> Self {
        self.max_results = Some(n);
        self
    }
}

/// Options for [`Graph::minimum_cycle_basis`].
///
/// The default computes an exact minimum cycle basis with each cycle listed
/// in cycle order (`bfs_cutoff: None, complete: true, use_cycle_order: true`),
/// the same defaults as the igraph R and Python interfaces.
///
/// ```
/// use igraph::cycles::MinimumCycleBasisOptions;
/// let fast = MinimumCycleBasisOptions::default().with_bfs_cutoff(3).with_complete(false);
/// assert_eq!(fast.bfs_cutoff, Some(3));
/// assert!(fast.use_cycle_order);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MinimumCycleBasisOptions {
    /// `None` computes an exact minimum basis. `Some(k)` limits the depth of
    /// the BFS trees used to generate candidate cycles, which can speed up
    /// the computation substantially: then only the returned cycles of length
    /// at most `2k + 1` are guaranteed to belong to some minimum basis.
    /// Default: `None`.
    pub bfs_cutoff: Option<usize>,
    /// Only used with a [`bfs_cutoff`](Self::bfs_cutoff). If `true`, a
    /// complete basis is still returned (its cycles longer than `2k + 1` may
    /// not be minimal); if `false`, only the cycles of length at most `2k + 1`
    /// are returned, which is faster but does not span the whole cycle space.
    /// Default: `true`.
    pub complete: bool,
    /// If `true`, the edge ids of each cycle are listed in the order they
    /// appear along the cycle (at a small cost); if `false`, their order is
    /// unspecified. Default: `true`.
    pub use_cycle_order: bool,
}

impl Default for MinimumCycleBasisOptions {
    fn default() -> Self {
        Self {
            bfs_cutoff: None,
            complete: true,
            use_cycle_order: true,
        }
    }
}

impl MinimumCycleBasisOptions {
    /// Sets [`bfs_cutoff`](Self::bfs_cutoff) to `Some(k)`.
    pub fn with_bfs_cutoff(mut self, k: usize) -> Self {
        self.bfs_cutoff = Some(k);
        self
    }

    /// Sets [`complete`](Self::complete).
    pub fn with_complete(mut self, complete: bool) -> Self {
        self.complete = complete;
        self
    }

    /// Sets [`use_cycle_order`](Self::use_cycle_order).
    pub fn with_cycle_order(mut self, use_cycle_order: bool) -> Self {
        self.use_cycle_order = use_cycle_order;
        self
    }
}

/// Whether a graph has an Eulerian path and/or cycle, see [`Graph::is_eulerian`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EulerianStatus {
    /// `true` if some walk traverses every edge exactly once.
    pub has_path: bool,
    /// `true` if some *closed* walk traverses every edge exactly once
    /// (this implies `has_path`).
    pub has_cycle: bool,
}

/// An Eulerian path or cycle, see [`Graph::eulerian_path`] and
/// [`Graph::eulerian_cycle`].
///
/// `edges[i]` connects `vertices[i]` and `vertices[i + 1]`, so for a
/// non-empty walk `vertices.len() == edges.len() + 1`; for a cycle the first
/// and last vertices coincide. Every edge of the graph appears in `edges`
/// exactly once.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct EulerianWalk {
    /// The visited vertices, in order.
    pub vertices: Vec<VertexId>,
    /// The traversed edges, in order.
    pub edges: Vec<EdgeId>,
}

/// `Option<usize>` limit → igraph's "negative means unlimited" convention.
/// Values above `igraph_int_t::MAX` saturate (they are unreachable anyway),
/// instead of wrapping to a negative "unlimited" value.
fn limit(value: Option<usize>) -> igraph_int_t {
    value.map_or(-1, |v| {
        igraph_int_t::try_from(v).unwrap_or(igraph_int_t::MAX)
    })
}

/// `Option<usize>` BFS cutoff → igraph's "negative means no cutoff" real.
fn cutoff(value: Option<usize>) -> igraph_real_t {
    value.map_or(-1.0, |c| c as igraph_real_t)
}

/// Pointer to an optional weight view, or NULL.
fn opt_ptr(v: &Option<crate::vector::View<'_, Vector>>) -> *const igraph_vector_t {
    v.as_ref().map_or(std::ptr::null(), |v| v.as_ptr())
}

struct CycleVisitor<'f, F> {
    f: &'f mut F,
    remaining: Option<usize>,
}

unsafe extern "C" fn cycle_trampoline<F>(
    vertices: *const igraph_vector_int_t,
    edges: *const igraph_vector_int_t,
    arg: *mut c_void,
) -> igraph_error_t
where
    F: FnMut(&[VertexId], &[EdgeId]) -> ControlFlow<()>,
{
    // The closure runs in a fresh level of igraph's "finally" stack:
    // otherwise a failing igraph call made by the closure would free the
    // temporaries of the running cycle search (a use-after-free in C).
    // SAFETY: bookkeeping on igraph's thread-local finally stack; the
    // matching EXIT runs below, as `catch_panic` never unwinds.
    unsafe { IGRAPH_FINALLY_ENTER() };
    let code = catch_panic(|| {
        // SAFETY: `arg` is the `CycleVisitor<F>` passed by `simple_cycles_callback`,
        // alive and exclusively borrowed for the whole C call; the vectors are
        // valid for the duration of this callback.
        let visitor = unsafe { &mut *(arg as *mut CycleVisitor<'_, F>) };
        let (vertices, edges) = unsafe { ((*vertices).as_slice(), (*edges).as_slice()) };
        // `remaining` is never `Some(0)` here: the wrapper returns early for a
        // zero cap, and we answer IGRAPH_STOP as soon as it reaches zero.
        let flow = (visitor.f)(vertices, edges);
        if let Some(r) = visitor.remaining.as_mut() {
            *r -= 1;
        }
        match flow {
            ControlFlow::Break(()) => igraph_error_type_t_IGRAPH_STOP,
            ControlFlow::Continue(()) if visitor.remaining == Some(0) => {
                igraph_error_type_t_IGRAPH_STOP
            }
            ControlFlow::Continue(()) => igraph_error_type_t_IGRAPH_SUCCESS,
        }
    });
    // SAFETY: closes the level opened above; any failed nested call has
    // already freed its own objects of that level.
    unsafe { IGRAPH_FINALLY_EXIT() };
    code
}

impl igraph_t {
    /// Checks whether the graph is a directed acyclic graph (DAG).
    ///
    /// A DAG is a directed graph without directed cycles (self-loops count as
    /// cycles here). Undirected graphs are never DAGs: this returns `false` for
    /// them. The result is cached inside the graph, so repeated calls without
    /// modifications in between take O(1) time.
    ///
    /// Time complexity: O(|V|+|E|).
    ///
    /// See also [`is_acyclic`](Self::is_acyclic), which agrees with this
    /// function on directed graphs but also tests undirected graphs for
    /// cycles, and [`find_cycle`](Self::find_cycle), which returns a cycle
    /// proving that a graph is not a DAG.
    ///
    /// Binds [`igraph_is_dag`](https://igraph.org/c/html/latest/igraph-Cycles.html#igraph_is_dag).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let chain = Graph::from_edges(&[(0, 1), (1, 2)], 3, true)?;
    /// assert!(chain.is_dag()?);
    /// let triangle = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, true)?;
    /// assert!(!triangle.is_dag()?);
    /// // Undirected graphs are not DAGs, even when they are trees.
    /// let undirected = Graph::from_edges(&[(0, 1)], 2, false)?;
    /// assert!(!undirected.is_dag()?);
    /// assert!(undirected.is_acyclic()?); // ... but they can be acyclic
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn is_dag(&self) -> Result<bool> {
        let mut res = false;
        igraph_call!(igraph_is_dag(self, &mut res))?;
        Ok(res)
    }

    /// Computes a topological sorting of a directed acyclic graph.
    ///
    /// A topological sorting is a linear ordering of the vertices in which
    /// every vertex comes before all the vertices it has edges to. Every DAG
    /// has at least one, and possibly many; one of them is returned.
    ///
    /// With `mode = NeighborMode::Out` each vertex precedes its successors, so
    /// the sources (no incoming edges) come first; with `NeighborMode::In` each
    /// vertex precedes its predecessors, so the sinks come first.
    /// Self-loops are ignored.
    ///
    /// Time complexity: O(|V|+|E|).
    ///
    /// See also [`is_dag`](Self::is_dag) to test beforehand whether an order
    /// exists, [`feedback_arc_set`](Self::feedback_arc_set) to make a cyclic
    /// graph sortable, and [`transitive_closure`](Self::transitive_closure)
    /// for the full "must come before" relation.
    ///
    /// Binds [`igraph_topological_sorting`](https://igraph.org/c/html/latest/igraph-Cycles.html#igraph_topological_sorting).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the graph
    /// contains a cycle (other than self-loops), if the graph is undirected,
    /// or if `mode` is `NeighborMode::All`.
    ///
    /// # Examples
    ///
    /// The example graph from the igraph documentation (and Wikipedia):
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(
    ///     &[(0, 3), (0, 4), (1, 3), (2, 4), (2, 7), (3, 5), (3, 6), (3, 7), (4, 6)],
    ///     8,
    ///     true,
    /// )?;
    /// assert_eq!(g.topological_sorting(NeighborMode::Out)?, vec![0, 1, 2, 3, 4, 5, 7, 6]);
    /// assert_eq!(g.topological_sorting(NeighborMode::In)?, vec![5, 6, 7, 4, 3, 2, 0, 1]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn topological_sorting(&self, mode: NeighborMode) -> Result<Vec<VertexId>> {
        let mut res = VectorInt::new();
        igraph_call!(igraph_topological_sorting(self, &mut res, mode.into()))?;
        Ok(res.into())
    }

    /// Finds a single cycle of the graph, or `None` if the graph is acyclic.
    ///
    /// `mode` selects how edge directions are considered in directed graphs:
    /// `Out` follows them, `In` follows them backwards (the returned cycle is
    /// then listed against the edge directions) and `All` ignores them. It is
    /// ignored for undirected graphs. Self-loops and multi-edges count as
    /// cycles of length 1 and 2.
    ///
    /// igraph lists the vertices so that each edge *enters* the vertex at the
    /// same position; this wrapper rotates them by one so that the result
    /// follows the [`Cycle`] layout (`edges[i]` goes from `vertices[i]` to the next vertex), like
    /// [`simple_cycles`](Self::simple_cycles) does.
    ///
    /// The cycle is not necessarily a shortest one: use
    /// [`girth_with_cycle`](Self::girth_with_cycle) for that (undirected,
    /// cycles of length at least 3), or [`simple_cycles`](Self::simple_cycles)
    /// to list all cycles. [`is_acyclic`](Self::is_acyclic) only answers
    /// whether a cycle exists.
    ///
    /// Time complexity: O(|V|+|E|).
    ///
    /// Binds [`igraph_find_cycle`](https://igraph.org/c/html/latest/igraph-Cycles.html#igraph_find_cycle).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4), (2, 0)], 5, true)?;
    /// let c = g.find_cycle(NeighborMode::Out)?.unwrap();
    /// assert_eq!(c.vertices, vec![0, 1, 2]);
    /// assert_eq!(c.edges, vec![0, 1, 4]); // 0 -> 1 -> 2 -> 0
    ///
    /// let tree = Graph::from_edges(&[(0, 1), (0, 2)], 3, false)?;
    /// assert_eq!(tree.find_cycle(NeighborMode::All)?, None);
    ///
    /// // Any cycle is at least as long as the girth.
    /// let petersen = Graph::famous("Petersen")?;
    /// let c = petersen.find_cycle(NeighborMode::All)?.unwrap();
    /// assert!(c.len() >= petersen.girth()?.unwrap());
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn find_cycle(&self, mode: NeighborMode) -> Result<Option<Cycle>> {
        let mut vertices = VectorInt::new();
        let mut edges = VectorInt::new();
        igraph_call!(igraph_find_cycle(
            self,
            &mut vertices,
            &mut edges,
            mode.into()
        ))?;
        if edges.is_empty() {
            Ok(None)
        } else {
            let mut vertices: Vec<VertexId> = vertices.into();
            vertices.rotate_right(1);
            Ok(Some(Cycle {
                vertices,
                edges: edges.into(),
            }))
        }
    }

    /// Lists the simple cycles of the graph (Johnson's algorithm).
    ///
    /// A simple cycle is a closed walk without repeated vertices. Each cycle
    /// is reported once (in undirected graphs, the two traversal directions
    /// of a cycle count as one). Self-loops are cycles of length 1; two
    /// parallel edges give a cycle of length 2 when they can be traversed in
    /// opposite directions (so not two parallel arcs `u -> v` with
    /// `NeighborMode::Out`). The search can be restricted with
    /// [`SimpleCyclesOptions`]: direction handling, minimum and maximum cycle
    /// length and a cap on the number of results. The number of simple cycles
    /// can grow exponentially with the size of the graph: bound the lengths or
    /// the result count on large graphs, or use
    /// [`simple_cycles_callback`](Self::simple_cycles_callback) to avoid
    /// storing them.
    ///
    /// See also [`list_triangles`](Self::list_triangles) for the cycles of
    /// length 3 only, [`find_cycle`](Self::find_cycle) for a single cycle, and
    /// [`get_all_simple_paths`](Self::get_all_simple_paths) for simple paths.
    ///
    /// This function is *experimental* in igraph 1.0.
    ///
    /// Reference: Johnson DB, *Finding all the elementary circuits of a
    /// directed graph*, SIAM J. Comput. 4(1):77-84 (1975).
    ///
    /// Binds [`igraph_simple_cycles`](https://igraph.org/c/html/latest/igraph-Cycles.html#igraph_simple_cycles).
    ///
    /// # Examples
    ///
    /// A square with one diagonal has three cycles: two triangles and the
    /// outer square.
    ///
    /// ```
    /// use igraph::{cycles::SimpleCyclesOptions, prelude::*};
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], 4, false)?;
    /// let cycles = g.simple_cycles(&SimpleCyclesOptions::default())?;
    /// let mut lengths: Vec<usize> = cycles.iter().map(|c| c.len()).collect();
    /// lengths.sort();
    /// assert_eq!(lengths, vec![3, 3, 4]);
    /// // Only the triangles:
    /// let triangles = g.simple_cycles(&SimpleCyclesOptions::default().with_max_length(3))?;
    /// assert_eq!(triangles.len(), 2);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn simple_cycles(&self, options: &SimpleCyclesOptions) -> Result<Vec<Cycle>> {
        if options.max_results == Some(0) {
            return Ok(Vec::new());
        }
        let mut vertices = VectorIntList::new();
        let mut edges = VectorIntList::new();
        igraph_call!(igraph_simple_cycles(
            self,
            &mut vertices,
            &mut edges,
            options.mode.into(),
            limit(options.min_cycle_length),
            limit(options.max_cycle_length),
            limit(options.max_results)
        ))?;
        Ok(vertices
            .iter()
            .zip(edges.iter())
            .map(|(v, e)| Cycle {
                vertices: v.to_vec(),
                edges: e.to_vec(),
            })
            .collect())
    }

    /// Visits the simple cycles of the graph with a closure (Johnson's
    /// algorithm), without storing them.
    ///
    /// This is the streaming variant of [`simple_cycles`](Self::simple_cycles),
    /// with the same semantics and options. For each cycle, `f` is called with
    /// its vertices and edges (see [`Cycle`] for their layout); it returns
    /// [`ControlFlow::Continue`] to keep searching or [`ControlFlow::Break`]
    /// to stop the search early (this is not an error). The search also stops
    /// after [`max_results`](SimpleCyclesOptions::max_results) cycles.
    ///
    /// If `f` panics, the search is aborted and the panic is resumed in the
    /// caller once igraph has cleaned up.
    ///
    /// This function is *experimental* in igraph 1.0.
    ///
    /// Binds [`igraph_simple_cycles_callback`](https://igraph.org/c/html/latest/igraph-Cycles.html#igraph_simple_cycles_callback).
    ///
    /// # Examples
    ///
    /// Is there a cycle through vertex 3 in the complete graph `K5`? Stop at
    /// the first one.
    ///
    /// ```
    /// use igraph::{cycles::SimpleCyclesOptions, prelude::*};
    /// use std::ops::ControlFlow;
    /// let k5 = Graph::full(5, false, false)?;
    /// let mut total = 0;
    /// k5.simple_cycles_callback(&SimpleCyclesOptions::default(), |_, _| {
    ///     total += 1;
    ///     ControlFlow::Continue(())
    /// })?;
    /// assert_eq!(total, 37); // 10 triangles + 15 squares + 12 pentagons
    ///
    /// let mut found = None;
    /// k5.simple_cycles_callback(&SimpleCyclesOptions::default(), |vs, _| {
    ///     if vs.contains(&3) {
    ///         found = Some(vs.to_vec());
    ///         return ControlFlow::Break(());
    ///     }
    ///     ControlFlow::Continue(())
    /// })?;
    /// assert!(found.unwrap().contains(&3));
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn simple_cycles_callback<F>(&self, options: &SimpleCyclesOptions, mut f: F) -> Result<()>
    where
        F: FnMut(&[VertexId], &[EdgeId]) -> ControlFlow<()>,
    {
        if options.max_results == Some(0) {
            return Ok(());
        }
        let mut visitor = CycleVisitor {
            f: &mut f,
            remaining: options.max_results,
        };
        igraph_call!(igraph_simple_cycles_callback(
            self,
            options.mode.into(),
            limit(options.min_cycle_length),
            limit(options.max_cycle_length),
            Some(cycle_trampoline::<F>),
            &mut visitor as *mut CycleVisitor<'_, F> as *mut c_void
        ))
    }

    /// Computes a fundamental cycle basis, from breadth-first search trees.
    ///
    /// Every edge not in a BFS spanning forest closes exactly one cycle with
    /// the tree edges: these *fundamental cycles* form a basis of the cycle
    /// space, whose dimension is the cyclomatic number |E| - |V| + c (c being
    /// the number of connected components). Each cycle is returned as a list
    /// of edge ids, in cycle order. Edge directions are ignored; multi-edges
    /// and self-loops are supported.
    ///
    /// - `start`: `None` returns a complete basis; `Some(v)` returns only the
    ///   fundamental cycles of the BFS tree rooted at `v`, i.e. of the
    ///   (weakly) connected component of `v`.
    /// - `bfs_cutoff`: `None` returns a complete basis; `Some(k)` limits the
    ///   BFS depth, so that only cycles of length at most `2k + 1` are found.
    ///
    /// Time complexity: O(|V|+|E|). This function is *experimental* in igraph
    /// 1.0. (The C function also takes a `weights` argument, currently unused
    /// by igraph, so it is not exposed.)
    ///
    /// See also [`minimum_cycle_basis`](Self::minimum_cycle_basis) for a basis
    /// of shortest total length, and
    /// [`connected_components`](Self::connected_components) for `c`.
    ///
    /// Binds [`igraph_fundamental_cycles`](https://igraph.org/c/html/latest/igraph-Cycles.html#igraph_fundamental_cycles).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) if
    /// `start` is not a vertex of the graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // Two triangles sharing the edge 0-2: the cycle space has dimension 5 - 4 + 1 = 2.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3), (3, 0)], 4, false)?;
    /// let basis = g.fundamental_cycles(None, None)?;
    /// assert_eq!(basis.len(), 2);
    ///
    /// // In general: one basis cycle per edge outside a spanning forest.
    /// let karate = Graph::famous("Zachary")?;
    /// let c = karate.connected_components(Connectedness::Weak)?.count;
    /// let dim = karate.ecount() - karate.vcount() + c;
    /// assert_eq!(karate.fundamental_cycles(None, None)?.len(), dim); // 78 - 34 + 1
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn fundamental_cycles(
        &self,
        start: Option<VertexId>,
        bfs_cutoff: Option<usize>,
    ) -> Result<Vec<Vec<EdgeId>>> {
        if let Some(v) = start
            && (v < 0 || v as usize >= self.vcount())
        {
            return Err(Error::new(
                crate::ErrorKind::InvalidVertexId,
                format!("invalid start vertex {v} for fundamental cycles"),
            ));
        }
        let mut res = VectorIntList::new();
        igraph_call!(igraph_fundamental_cycles(
            self,
            std::ptr::null(),
            &mut res,
            start.unwrap_or(-1),
            cutoff(bfs_cutoff)
        ))?;
        Ok(res.to_vecs())
    }

    /// Computes a minimum weight cycle basis (modified Horton algorithm).
    ///
    /// A minimum cycle basis is a basis of the cycle space whose total length
    /// is as small as possible; its cycles are returned as edge-id lists,
    /// sorted by increasing length. Edge directions are ignored; multi-edges
    /// and self-loops are supported.
    ///
    /// The search is tuned by [`MinimumCycleBasisOptions`]: an optional BFS
    /// cutoff trading exactness for speed, whether a cut-off basis should
    /// still be completed, and whether cycles are listed in cycle order.
    ///
    /// This function is *experimental* in igraph 1.0. (The C function also
    /// takes a `weights` argument, currently unused by igraph, so it is not
    /// exposed.)
    ///
    /// For a simple graph with at least one cycle, the first (shortest) basis
    /// cycle has the length of the [`girth`](Self::girth). See also
    /// [`fundamental_cycles`](Self::fundamental_cycles), a faster but usually
    /// longer basis.
    ///
    /// Reference: Horton JD, *A polynomial-time algorithm to find the shortest
    /// cycle basis of a graph*, SIAM J. Comput. 16(2):358-366 (1987).
    ///
    /// Binds [`igraph_minimum_cycle_basis`](https://igraph.org/c/html/latest/igraph-Cycles.html#igraph_minimum_cycle_basis).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::{cycles::MinimumCycleBasisOptions, prelude::*};
    /// // A square with a diagonal: the minimum basis is made of the two triangles.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], 4, false)?;
    /// let basis = g.minimum_cycle_basis(&MinimumCycleBasisOptions::default())?;
    /// assert_eq!(basis.iter().map(Vec::len).collect::<Vec<_>>(), vec![3, 3]);
    /// // The outer square is the sum (symmetric difference) of the two
    /// // triangles: the shared diagonal, edge 4, cancels out.
    /// let mut parity = [false; 5];
    /// for &e in basis.concat().iter() {
    ///     parity[e as usize] ^= true;
    /// }
    /// assert_eq!(parity, [true, true, true, true, false]);
    ///
    /// // The Petersen graph has girth 5: its 15 - 10 + 1 = 6 basis cycles are pentagons.
    /// let petersen = Graph::famous("Petersen")?;
    /// let basis = petersen.minimum_cycle_basis(&MinimumCycleBasisOptions::default())?;
    /// assert_eq!(basis.iter().map(Vec::len).collect::<Vec<_>>(), vec![5; 6]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn minimum_cycle_basis(
        &self,
        options: &MinimumCycleBasisOptions,
    ) -> Result<Vec<Vec<EdgeId>>> {
        let mut res = VectorIntList::new();
        igraph_call!(igraph_minimum_cycle_basis(
            self,
            std::ptr::null(),
            &mut res,
            cutoff(options.bfs_cutoff),
            options.complete,
            options.use_cycle_order
        ))?;
        Ok(res.to_vecs())
    }

    /// Finds a feedback arc set: edges whose removal makes the graph acyclic.
    ///
    /// One is usually interested in a *minimum* feedback arc set, i.e. one of
    /// smallest total weight (`weights`, one per edge, or `None` for unit
    /// weights; weights are meant to be non-negative). For undirected graphs
    /// this is easy: the complement of a maximum weight spanning forest (and
    /// `algo` is ignored). For directed
    /// graphs the problem is NP-complete and `algo` selects the method:
    ///
    /// - [`FasAlgorithm::ExactIp`]: exact minimum via integer programming,
    ///   picking the best available formulation (currently `ExactIpCg`).
    ///   Exponential in the worst case.
    /// - [`FasAlgorithm::ExactIpCg`]: exact, set-cover formulation with
    ///   incremental cycle (constraint) generation.
    /// - [`FasAlgorithm::ExactIpTi`]: exact, topological-order formulation
    ///   with triangle inequalities; usually much slower.
    /// - [`FasAlgorithm::ApproxEades`]: the linear time O(|E|) heuristic of
    ///   Eades, Lin and Smyth (1993), returning fewer than |E|/2 - |V|/6 edges.
    ///
    /// References: Eades P, Lin X, Smyth WF, Inf. Proc. Letters 47(6):319-323
    /// (1993); Baharev A et al., ACM J. Exp. Algorithmics 26:1-28 (2021).
    ///
    /// Time complexity: depends on `algo` (see above).
    ///
    /// See also [`feedback_vertex_set`](Self::feedback_vertex_set) for the
    /// vertex version, and, for undirected graphs,
    /// [`minimum_spanning_tree`](Self::minimum_spanning_tree): the edges not
    /// in a *maximum* weight spanning forest form a minimum feedback arc set.
    /// Delete the returned edges at once with
    /// [`delete_edges`](Self::delete_edges).
    ///
    /// Binds [`igraph_feedback_arc_set`](https://igraph.org/c/html/latest/igraph-Cycles.html#igraph_feedback_arc_set).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the
    /// weight vector has the wrong length or invalid values;
    /// [`ErrorKind::Unimplemented`](crate::ErrorKind::Unimplemented) for the
    /// exact methods when igraph was built without GLPK.
    ///
    /// # Examples
    ///
    /// The graph of igraph's own example program:
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let edges = [(0, 1), (1, 2), (2, 0), (2, 3), (2, 4), (0, 4), (4, 3), (5, 0), (6, 5)];
    /// let g = Graph::from_edges(&edges, 7, true)?;
    /// assert_eq!(g.feedback_arc_set(None, FasAlgorithm::ExactIp)?, vec![0]);
    /// // Make edge 0 expensive to cut: another edge of the triangle goes.
    /// let w = [3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    /// let fas = g.feedback_arc_set(Some(&w), FasAlgorithm::ExactIp)?;
    /// assert!(fas == [1] || fas == [2]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn feedback_arc_set(
        &self,
        weights: Option<&[f64]>,
        algo: FasAlgorithm,
    ) -> Result<Vec<EdgeId>> {
        if let Some(w) = weights
            && w.len() != self.ecount()
        {
            return Err(Error::invalid(format!(
                "weight vector length ({}) must match the number of edges ({})",
                w.len(),
                self.ecount()
            )));
        }
        let w = weights.map(Vector::view);
        let mut res = VectorInt::new();
        igraph_call!(igraph_feedback_arc_set(
            self,
            &mut res,
            opt_ptr(&w),
            algo.into()
        ))?;
        Ok(res.into())
    }

    /// Finds a minimum feedback vertex set: vertices whose removal makes the
    /// graph acyclic.
    ///
    /// Minimizes the total vertex weight (`vertex_weights`, one per vertex, or
    /// `None` for unit weights). The problem is NP-complete both for directed
    /// and undirected graphs; the only method, [`FvsAlgorithm::ExactIp`], uses
    /// integer programming with incremental cycle generation (like
    /// [`FasAlgorithm::ExactIpCg`]) and is exponential in the worst case.
    /// Self-loops count as cycles, so looped vertices are always included.
    ///
    /// Time complexity: exponential in the worst case. See also
    /// [`feedback_arc_set`](Self::feedback_arc_set), and
    /// [`delete_vertices`](Self::delete_vertices) or
    /// [`induced_subgraph`](Self::induced_subgraph) to remove the vertices.
    ///
    /// Binds [`igraph_feedback_vertex_set`](https://igraph.org/c/html/latest/igraph-Cycles.html#igraph_feedback_vertex_set).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the
    /// weight vector has the wrong length or invalid values;
    /// [`ErrorKind::Unimplemented`](crate::ErrorKind::Unimplemented) when
    /// igraph was built without GLPK.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // Two triangles sharing vertex 0 (a "bowtie"): removing 0 breaks both.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)], 5, false)?;
    /// assert_eq!(g.feedback_vertex_set(None, FvsAlgorithm::ExactIp)?, vec![0]);
    ///
    /// // The Petersen graph needs three vertices removed to become a forest.
    /// let mut petersen = Graph::famous("Petersen")?;
    /// let fvs = petersen.feedback_vertex_set(None, FvsAlgorithm::ExactIp)?;
    /// assert_eq!(fvs.len(), 3);
    /// petersen.delete_vertices(&fvs)?;
    /// assert!(petersen.is_forest(NeighborMode::All)?);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn feedback_vertex_set(
        &self,
        vertex_weights: Option<&[f64]>,
        algo: FvsAlgorithm,
    ) -> Result<Vec<VertexId>> {
        if let Some(w) = vertex_weights
            && w.len() != self.vcount()
        {
            return Err(Error::invalid(format!(
                "vertex weight vector length ({}) must match the number of vertices ({})",
                w.len(),
                self.vcount()
            )));
        }
        let w = vertex_weights.map(Vector::view);
        let mut res = VectorInt::new();
        igraph_call!(igraph_feedback_vertex_set(
            self,
            &mut res,
            opt_ptr(&w),
            algo.into()
        ))?;
        Ok(res.into())
    }

    /// Checks whether the graph has an Eulerian path and/or an Eulerian cycle.
    ///
    /// An Eulerian path traverses every edge exactly once; an Eulerian cycle
    /// is a closed one. By Euler's theorem, a connected (ignoring isolated
    /// vertices) undirected graph has an Eulerian cycle iff all degrees are
    /// even, and a path iff at most two degrees are odd. A directed graph
    /// whose edges lie in a single weakly connected component has an Eulerian
    /// cycle iff every in-degree equals the out-degree, and a path iff this
    /// holds except for at most one start vertex (out-degree one larger) and
    /// one end vertex (in-degree one larger). A graph without edges trivially
    /// has both.
    ///
    /// Time complexity: O(|V|+|E|).
    ///
    /// See also [`degree`](Self::degree) and
    /// [`is_connected`](Self::is_connected), the ingredients of Euler's
    /// theorem, and [`eulerian_path`](Self::eulerian_path) /
    /// [`eulerian_cycle`](Self::eulerian_cycle) to construct the walks.
    ///
    /// Binds [`igraph_is_eulerian`](https://igraph.org/c/html/latest/igraph-Cycles.html#igraph_is_eulerian).
    ///
    /// # Examples
    ///
    /// The seven bridges of Königsberg have no Eulerian path:
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // 0: island Kneiphof, 1: north bank, 2: south bank, 3: east island Lomse
    /// let bridges = [(0, 1), (0, 1), (0, 2), (0, 2), (0, 3), (1, 3), (2, 3)];
    /// let konigsberg = Graph::from_edges(&bridges, 4, false)?;
    /// let status = konigsberg.is_eulerian()?;
    /// assert!(!status.has_path && !status.has_cycle);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn is_eulerian(&self) -> Result<EulerianStatus> {
        let mut has_path = false;
        let mut has_cycle = false;
        igraph_call!(igraph_is_eulerian(self, &mut has_path, &mut has_cycle))?;
        Ok(EulerianStatus {
            has_path,
            has_cycle,
        })
    }

    /// Finds an Eulerian path, traversing every edge exactly once
    /// (Hierholzer's algorithm).
    ///
    /// If the graph has no edges, an empty walk is returned. When the graph
    /// also has an Eulerian cycle, the returned path is necessarily closed
    /// (an Eulerian path with distinct ends exists only when exactly two
    /// vertices have odd degree, or unbalanced in/out-degrees if directed).
    ///
    /// Time complexity: O(|V|+|E|).
    ///
    /// Binds [`igraph_eulerian_path`](https://igraph.org/c/html/latest/igraph-Cycles.html#igraph_eulerian_path).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::NoSolution`](crate::ErrorKind::NoSolution) if the graph has
    /// no Eulerian path (check first with [`is_eulerian`](Self::is_eulerian)).
    ///
    /// # Examples
    ///
    /// Drawing the "house of Santa Claus" without lifting the pen:
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // square 0-1-2-3, both diagonals, and the roof 2-4-3
    /// let house = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3), (2, 4), (4, 3)];
    /// let g = Graph::from_edges(&house, 5, false)?;
    /// let walk = g.eulerian_path()?;
    /// assert_eq!(walk.edges.len(), 8);
    /// assert_eq!(walk.vertices.len(), 9);
    /// // It must start and end at the two odd-degree corners 0 and 1.
    /// let ends = [walk.vertices[0], walk.vertices[8]];
    /// assert!(ends == [0, 1] || ends == [1, 0]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn eulerian_path(&self) -> Result<EulerianWalk> {
        let mut edges = VectorInt::new();
        let mut vertices = VectorInt::new();
        igraph_call!(igraph_eulerian_path(self, &mut edges, &mut vertices))?;
        Ok(EulerianWalk {
            vertices: vertices.into(),
            edges: edges.into(),
        })
    }

    /// Finds an Eulerian cycle, a closed walk traversing every edge exactly
    /// once (Hierholzer's algorithm).
    ///
    /// The first and last returned vertices coincide. If the graph has no
    /// edges, an empty walk is returned.
    ///
    /// Time complexity: O(|V|+|E|).
    ///
    /// Binds [`igraph_eulerian_cycle`](https://igraph.org/c/html/latest/igraph-Cycles.html#igraph_eulerian_cycle).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::NoSolution`](crate::ErrorKind::NoSolution) if the graph has
    /// no Eulerian cycle.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let triangle = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, false)?;
    /// let c = triangle.eulerian_cycle()?;
    /// assert_eq!(c.edges, vec![0, 1, 2]);
    /// assert_eq!(c.vertices, vec![0, 1, 2, 0]);
    ///
    /// let path = Graph::from_edges(&[(0, 1), (1, 2)], 3, false)?;
    /// assert_eq!(path.eulerian_cycle().unwrap_err().kind(), ErrorKind::NoSolution);
    ///
    /// // An Eulerian cycle of the binary de Bruijn graph B(2, 2) traverses all
    /// // 8 three-bit words once: reading one bit per step gives a de Bruijn
    /// // sequence, which contains every 3-bit word (cyclically) exactly once.
    /// let b = Graph::de_bruijn(2, 2)?;
    /// let tour = b.eulerian_cycle()?;
    /// let bits: Vec<i64> = tour.vertices[1..].iter().map(|v| v & 1).collect();
    /// let mut words: Vec<i64> =
    ///     (0..8).map(|i| 4 * bits[i] + 2 * bits[(i + 1) % 8] + bits[(i + 2) % 8]).collect();
    /// words.sort();
    /// assert_eq!(words, (0..8).collect::<Vec<_>>());
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn eulerian_cycle(&self) -> Result<EulerianWalk> {
        let mut edges = VectorInt::new();
        let mut vertices = VectorInt::new();
        igraph_call!(igraph_eulerian_cycle(self, &mut edges, &mut vertices))?;
        Ok(EulerianWalk {
            vertices: vertices.into(),
            edges: edges.into(),
        })
    }
}

//! Maximum flows, minimum cuts and graph connectivity (`igraph_flow.h`).
//!
//! A *flow network* is a graph whose edges carry a non-negative *capacity*
//! (a `&[f64]` indexed by edge id; when omitted every edge has capacity 1).
//! A *flow* from a `source` to a `target` assigns to every edge an amount not
//! exceeding its capacity such that, at every other vertex, what comes in
//! goes out. The celebrated *max-flow min-cut theorem* (Ford and Fulkerson,
//! 1956) states that the largest possible flow value equals the smallest
//! total capacity of a set of edges whose removal disconnects the target from
//! the source. Almost everything in this module is built on that identity:
//! edge and vertex connectivities, disjoint paths, cohesion measures and the
//! Gomory–Hu tree are all max-flow computations in disguise.
//!
//! All the functions are methods of [`Graph`], with named result structs
//! whenever the C function has several outputs.
//!
//! # Example: max-flow = min-cut
//!
//! ```
//! use igraph::prelude::*;
//!
//! // A small directed pipeline network: 0 is the source, 3 the sink.
//! let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)], 4, true)?;
//! let capacity = [3.0, 2.0, 1.0, 2.0, 3.0];
//!
//! let mf = g.maxflow(0, 3, Some(&capacity))?;
//! assert_eq!(mf.value, 5.0);
//!
//! // The edges of the minimum cut have a total capacity equal to the flow.
//! let cut_capacity: f64 = mf.cut.iter().map(|&e| capacity[e as usize]).sum();
//! assert_eq!(cut_capacity, mf.value);
//! assert_eq!(g.st_mincut_value(0, 3, Some(&capacity))?, 5.0);
//!
//! // The flow never exceeds the capacities.
//! assert!(mf.flow.iter().zip(&capacity).all(|(f, c)| *f <= *c));
//! # Ok::<(), igraph::Error>(())
//! ```
//!
//! # Provided functionality
//!
//! | Method | C function | Computes |
//! |---|---|---|
//! | [`Graph::maxflow`] | `igraph_maxflow` | value, per-edge flow, minimum cut and both sides ([`MaxFlow`]) |
//! | [`Graph::maxflow_value`] | `igraph_maxflow_value` | only the value of the maximum flow |
//! | [`Graph::maxflow_value_with_stats`] | `igraph_maxflow_value` | value plus push-relabel statistics ([`MaxflowStats`]) |
//! | [`Graph::st_mincut`] | `igraph_st_mincut` | minimum s-t cut ([`Cut`]) |
//! | [`Graph::st_mincut_value`] | `igraph_st_mincut_value` | value of the minimum s-t cut |
//! | [`Graph::mincut`] | `igraph_mincut` | minimum cut of the whole graph ([`Cut`]) |
//! | [`Graph::mincut_value`] | `igraph_mincut_value` | value of the minimum cut of the whole graph |
//! | [`Graph::st_vertex_connectivity`] | `igraph_st_vertex_connectivity` | vertex connectivity of a pair |
//! | [`Graph::vertex_connectivity`] | `igraph_vertex_connectivity` | vertex connectivity of the graph |
//! | [`Graph::st_edge_connectivity`] | `igraph_st_edge_connectivity` | edge connectivity of a pair |
//! | [`Graph::edge_connectivity`] | `igraph_edge_connectivity` | edge connectivity of the graph |
//! | [`Graph::edge_disjoint_paths`] | `igraph_edge_disjoint_paths` | number of edge-disjoint paths |
//! | [`Graph::vertex_disjoint_paths`] | `igraph_vertex_disjoint_paths` | number of vertex-disjoint paths |
//! | [`Graph::adhesion`] | `igraph_adhesion` | White–Harary adhesion (edge connectivity) |
//! | [`Graph::cohesion`] | `igraph_cohesion` | White–Harary cohesion (vertex connectivity) |
//! | [`Graph::even_tarjan_reduction`] | `igraph_even_tarjan_reduction` | vertex-splitting reduction ([`EvenTarjanReduction`]) |
//! | [`Graph::residual_graph`] | `igraph_residual_graph` | residual network of a flow ([`ResidualGraph`]) |
//! | [`Graph::reverse_residual_graph`] | `igraph_reverse_residual_graph` | reverse residual network of a flow |
//! | [`Graph::dominator_tree`] | `igraph_dominator_tree` | Lengauer–Tarjan dominator tree ([`DominatorTree`]) |
//! | [`Graph::all_st_cuts`] | `igraph_all_st_cuts` | every s-t edge cut ([`StCuts`]) |
//! | [`Graph::all_st_mincuts`] | `igraph_all_st_mincuts` | every minimum s-t edge cut ([`StMinCuts`]) |
//! | [`Graph::gomory_hu_tree`] | `igraph_gomory_hu_tree` | Gomory–Hu tree of all pairwise flows ([`GomoryHuTree`]) |
//!
//! Capacities must be finite and non-negative and, when given, have exactly
//! one entry per edge. igraph itself does not validate the sign of the
//! capacities (negative or NaN entries silently produce meaningless flows, and
//! infinite ones produce NaN flows), so a wrong length or an invalid entry is
//! reported as [`ErrorKind::InvalidValue`] before calling into C.
//!
//! The C reference for everything here is the
//! [Flows chapter](https://igraph.org/c/html/latest/igraph-Flows.html) of the
//! igraph manual.
//!
//! # Example: how robust is the karate club?
//!
//! ```
//! use igraph::prelude::*;
//!
//! let karate = Graph::famous("Zachary")?;
//! // Vertex 11 has a single friend, so one edge (or one vertex) isolates it.
//! assert_eq!(karate.edge_connectivity(true)?, 1);
//! assert_eq!(karate.vertex_connectivity(true)?, 1);
//! // The two leaders, 0 and 33, are far better connected: by Menger's
//! // theorem their local edge connectivity is the number of edge-disjoint
//! // paths, which is also the unit-capacity maximum flow.
//! let k = karate.st_edge_connectivity(0, 33)?;
//! assert_eq!(k, 10);
//! assert_eq!(karate.edge_disjoint_paths(0, 33)?, k);
//! assert_eq!(karate.maxflow_value(0, 33, None)?, k as f64);
//! // A Gomory–Hu tree stores all 561 pairwise flows in 33 numbers.
//! let gh = karate.gomory_hu_tree(None)?;
//! assert_eq!(gh.flow_between(0, 33), Some(k as f64));
//! # Ok::<(), igraph::Error>(())
//! ```
//!
//! # See also
//!
//! - [`Graph::is_connected`], [`Graph::articulation_points`] and
//!   [`Graph::bridges`] answer the "connectivity at least 1 / 2" questions in
//!   linear time, without any flow computation.
//! - [`Graph::minimum_size_separators`], [`Graph::all_minimal_st_separators`],
//!   [`Graph::is_separator`] and [`Graph::cohesive_blocks`] list the vertex
//!   sets behind [`Graph::vertex_connectivity`].
//! - [`Graph::maximum_bipartite_matching`] solves the classic flow
//!   application of matching the two sides of a bipartite graph.
//! - [`Graph::read_graph_dimacs_flow`] and [`Graph::write_graph_dimacs_flow`]
//!   read and write maximum flow instances in the DIMACS format.

use crate::{
    constants::{NeighborMode, VconnNei},
    error::{Error, ErrorKind, Result},
    ffi::*,
    graph::{EdgeId, Graph, VertexId},
    igraph_call,
    list::VectorIntList,
    vector::{Vector, VectorInt},
};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Statistics collected by igraph's push-relabel maximum flow solver
/// (`igraph_maxflow_stats_t`).
///
/// They are mostly interesting for benchmarking and for understanding how
/// much work the Goldberg–Tarjan algorithm had to do on a given network.
/// For undirected graphs igraph solves the flow problem on a directed graph
/// with every edge doubled, and the statistics refer to that graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct MaxflowStats {
    /// Number of push operations performed (`nopush`).
    pub pushes: i64,
    /// Number of relabel operations performed (`norelabel`).
    pub relabels: i64,
    /// Number of times the gap heuristic was applied (`nogap`).
    pub gaps: i64,
    /// Total number of vertices removed from further consideration by the
    /// gap heuristic (`nogapnodes`).
    pub gap_nodes: i64,
    /// Number of reverse breadth-first searches used to (re)compute the
    /// height function (`nobfs`); always at least one, as one runs before
    /// the algorithm starts.
    pub bfs_runs: i64,
}

impl From<igraph_maxflow_stats_t> for MaxflowStats {
    fn from(s: igraph_maxflow_stats_t) -> Self {
        Self {
            pushes: s.nopush,
            relabels: s.norelabel,
            gaps: s.nogap,
            gap_nodes: s.nogapnodes,
            bfs_runs: s.nobfs,
        }
    }
}

fn empty_stats() -> igraph_maxflow_stats_t {
    igraph_maxflow_stats_t {
        nopush: 0,
        norelabel: 0,
        nogap: 0,
        nogapnodes: 0,
        nobfs: 0,
    }
}

/// A maximum flow between two vertices, as computed by [`Graph::maxflow`].
#[derive(Debug, Clone, PartialEq)]
pub struct MaxFlow {
    /// The value of the maximum flow, i.e. the net amount entering the target.
    pub value: f64,
    /// The flow on each edge, indexed by edge id.
    ///
    /// In **undirected** graphs the sign encodes the direction: a positive
    /// value means the flow goes from the smaller vertex id to the larger
    /// one, a negative value the other way round.
    pub flow: Vec<f64>,
    /// Ids of the edges of the minimum cut corresponding to this flow; their
    /// capacities sum up to [`value`](Self::value).
    pub cut: Vec<EdgeId>,
    /// The side of the minimum cut containing the source.
    pub partition: Vec<VertexId>,
    /// The side of the minimum cut containing the target.
    pub partition2: Vec<VertexId>,
    /// Operation counts of the push-relabel solver.
    pub stats: MaxflowStats,
}

/// An edge cut splitting the vertices into two sides, as returned by
/// [`Graph::st_mincut`] and [`Graph::mincut`].
#[derive(Debug, Clone, PartialEq)]
pub struct Cut {
    /// Total capacity of the edges in the cut.
    pub value: f64,
    /// Ids of the edges in the cut.
    pub cut: Vec<EdgeId>,
    /// Vertices of the first side (for s-t cuts, the side of the source).
    pub partition: Vec<VertexId>,
    /// Vertices of the second side (for s-t cuts, the side of the target).
    pub partition2: Vec<VertexId>,
}

/// The Even–Tarjan reduction of a graph, see [`Graph::even_tarjan_reduction`].
#[derive(Debug, Clone, PartialEq)]
pub struct EvenTarjanReduction {
    /// The reduced directed graph, with `2 n` vertices and `n + 2 m` edges.
    pub graph: Graph,
    /// Capacities of the reduced graph's edges: 1 for the first `n` edges
    /// (the `i' → i''` "vertex" edges) and `n` (standing for infinity) for the
    /// remaining `2 m` edges.
    pub capacity: Vec<f64>,
}

/// The residual network of a flow, see [`Graph::residual_graph`].
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualGraph {
    /// The directed residual graph, on the same vertex set as the original.
    pub graph: Graph,
    /// Residual capacity (capacity minus flow) of each edge of
    /// [`graph`](Self::graph), in edge-id order.
    pub capacity: Vec<f64>,
}

/// A dominator tree of a flowgraph, see [`Graph::dominator_tree`].
#[derive(Debug, Clone, PartialEq)]
pub struct DominatorTree {
    /// Immediate dominator of each vertex, indexed by vertex id. It is
    /// `None` for the root itself and for the vertices unreachable from the
    /// root (the latter are also listed in [`leftout`](Self::leftout)).
    pub dom: Vec<Option<VertexId>>,
    /// The dominator tree as a directed graph on the same vertex set, with an
    /// edge from `idom(w)` to `w` (reversed when `mode` is
    /// [`NeighborMode::In`]); unreachable vertices are isolated.
    pub tree: Graph,
    /// Ids of the vertices that are not reachable from the root.
    pub leftout: Vec<VertexId>,
}

impl DominatorTree {
    /// Returns the chain of dominators of `v`, from its immediate dominator up
    /// to the root (empty for the root and for unreachable vertices).
    ///
    /// Panics if `v` is not a valid vertex id of the analysed graph.
    pub fn dominators(&self, v: VertexId) -> Vec<VertexId> {
        let mut chain = vec![];
        let mut cur = self.dom[v as usize];
        while let Some(d) = cur {
            chain.push(d);
            cur = self.dom[d as usize];
        }
        chain
    }
}

/// All minimal s-t edge cuts of a directed graph, see [`Graph::all_st_cuts`].
#[derive(Debug, Clone, PartialEq)]
pub struct StCuts {
    /// Every minimal cut, as a list of edge ids.
    pub cuts: Vec<Vec<EdgeId>>,
    /// For each cut, the vertex set `X` generating it: the cut consists of
    /// all edges from `X` to its complement. `X` contains the source.
    pub partition1s: Vec<Vec<VertexId>>,
}

/// All minimum s-t edge cuts of a directed graph, see
/// [`Graph::all_st_mincuts`].
#[derive(Debug, Clone, PartialEq)]
pub struct StMinCuts {
    /// The (common) total capacity of the minimum cuts.
    pub value: f64,
    /// Every minimum cut, as a list of edge ids.
    pub cuts: Vec<Vec<EdgeId>>,
    /// For each cut, the vertex set `X` generating it: the cut consists of
    /// all edges from `X` to its complement. `X` contains the source.
    pub partition1s: Vec<Vec<VertexId>>,
}

/// A Gomory–Hu tree, see [`Graph::gomory_hu_tree`].
#[derive(Debug, Clone, PartialEq)]
pub struct GomoryHuTree {
    /// The tree: an undirected graph on the same vertices as the input graph,
    /// with `n - 1` edges (edge `i - 1` joins vertex `i` to its tree parent).
    pub tree: Graph,
    /// The flow value annotating each tree edge, indexed by tree edge id.
    pub flows: Vec<f64>,
}

impl GomoryHuTree {
    /// The maximum flow (equivalently, minimum cut) value between `u` and `v`
    /// in the *original* graph, read off the tree as the minimum edge
    /// annotation along the tree path from `u` to `v`.
    ///
    /// Returns `None` when `u == v` or when either id is out of range.
    pub fn flow_between(&self, u: VertexId, v: VertexId) -> Option<f64> {
        let n = self.tree.vcount();
        if u == v || u < 0 || v < 0 || u as usize >= n || v as usize >= n {
            return None;
        }
        let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
        for (e, (a, b)) in self.tree.edge_list().into_iter().enumerate() {
            adj[a as usize].push((b as usize, self.flows[e]));
            adj[b as usize].push((a as usize, self.flows[e]));
        }
        // Breadth-first search carrying the bottleneck along the path.
        let mut best = vec![None; n];
        best[u as usize] = Some(f64::INFINITY);
        let mut queue = VecDeque::from([u as usize]);
        while let Some(x) = queue.pop_front() {
            let bx = best[x]?;
            for &(y, f) in &adj[x] {
                if best[y].is_none() {
                    best[y] = Some(bx.min(f));
                    queue.push_back(y);
                }
            }
        }
        best[v as usize]
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn check_len(what: &str, data: &[f64], expected: usize) -> Result<()> {
    if data.len() != expected {
        return Err(Error::invalid(format!(
            "the {what} vector has length {}, but the graph has {expected} edges",
            data.len()
        )));
    }
    Ok(())
}

/// Validates an optional capacity vector: one finite, non-negative entry per
/// edge.
///
/// The length check is required for soundness, not just for nicer errors:
/// for undirected graphs `igraph_maxflow` (and therefore every function built
/// on it, including `igraph_gomory_hu_tree`) reads `capacity[i]` for every
/// edge *before* validating the length (`igraph_i_maxflow_undirected` in
/// `flow.c`, igraph 1.0.0 and 1.0.1), so a short vector would be read out of
/// bounds.
fn check_capacity(graph: &Graph, capacity: Option<&[f64]>) -> Result<()> {
    let Some(c) = capacity else {
        return Ok(());
    };
    check_len("capacity", c, graph.ecount())?;
    match c.iter().position(|x| !(x.is_finite() && *x >= 0.0)) {
        Some(e) => Err(Error::invalid(format!(
            "capacities must be finite and non-negative, but edge {e} has capacity {}",
            c[e]
        ))),
        None => Ok(()),
    }
}

fn check_vertex(graph: &Graph, v: VertexId, what: &str) -> Result<()> {
    if v < 0 || v as usize >= graph.vcount() {
        return Err(Error::new(
            ErrorKind::InvalidVertexId,
            format!("invalid {what} vertex id {v}"),
        ));
    }
    Ok(())
}

fn check_pair(graph: &Graph, source: VertexId, target: VertexId) -> Result<()> {
    check_vertex(graph, source, "source")?;
    check_vertex(graph, target, "target")
}

fn count(value: igraph_int_t) -> usize {
    usize::try_from(value).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Methods
// ---------------------------------------------------------------------------

impl igraph_t {
    /// Maximum flow between `source` and `target`, together with a minimum
    /// cut certifying its optimality.
    ///
    /// Uses the push-relabel algorithm of Goldberg and Tarjan (J. ACM 35(4),
    /// 1988). A flow assigns to each edge a non-negative amount not larger
    /// than its capacity, and conserves the flow at every vertex other than
    /// the source and the target; its value is the net flow entering the
    /// target. The result contains the value, the flow on every edge, the
    /// edges of the corresponding minimum cut and the two sides of that cut
    /// (the first containing the source, the second the target).
    ///
    /// Works on directed and undirected graphs; in undirected graphs the sign
    /// of [`MaxFlow::flow`] tells the direction (see its docs).
    /// `capacity` gives one non-negative capacity per edge; `None` means that
    /// every edge has capacity 1.
    ///
    /// Time complexity: O(|V|³), usually much faster in practice.
    ///
    /// See also [`Graph::residual_graph`] to inspect the leftover capacities,
    /// and [`Graph::maximum_bipartite_matching`] for the matching problem that
    /// is usually solved as a unit-capacity flow.
    ///
    /// Binds [`igraph_maxflow`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_maxflow).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] for invalid `source`/`target`,
    /// [`ErrorKind::InvalidValue`] if they coincide, or if the capacity vector
    /// has the wrong length or a negative, NaN or infinite entry.
    ///
    /// # Examples
    ///
    /// The example of igraph's `flow2.c`:
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// let g = Graph::from_edges(
    ///     &[(0, 1), (1, 2), (2, 3), (0, 5), (5, 4), (4, 3), (3, 0)], 6, true)?;
    /// let capacity = [3.0, 1.0, 2.0, 10.0, 1.0, 3.0, 2.0];
    /// let mf = g.maxflow(0, 2, Some(&capacity))?;
    /// assert_eq!(mf.value, 1.0);
    /// assert_eq!(mf.flow, vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    /// assert_eq!(mf.partition, vec![0, 1, 3, 4, 5]);
    /// assert_eq!(mf.partition2, vec![2]);
    /// assert_eq!(mf.cut, vec![1]); // the edge 1 -> 2
    /// assert!(mf.stats.bfs_runs >= 1);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn maxflow(
        &self,
        source: VertexId,
        target: VertexId,
        capacity: Option<&[f64]>,
    ) -> Result<MaxFlow> {
        check_capacity(self, capacity)?;
        check_pair(self, source, target)?;
        let cap = capacity.map(Vector::view);
        let cap_ptr = cap.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let mut value = 0.0;
        let mut flow = Vector::new();
        let mut cut = VectorInt::new();
        let mut partition = VectorInt::new();
        let mut partition2 = VectorInt::new();
        let mut stats = empty_stats();
        igraph_call!(igraph_maxflow(
            self,
            &mut value,
            &mut flow,
            &mut cut,
            &mut partition,
            &mut partition2,
            source,
            target,
            cap_ptr,
            &mut stats,
        ))?;
        Ok(MaxFlow {
            value,
            flow: flow.into(),
            cut: cut.into(),
            partition: partition.into(),
            partition2: partition2.into(),
            stats: stats.into(),
        })
    }

    /// Value of the maximum flow between `source` and `target`.
    ///
    /// Same algorithm as [`Graph::maxflow`], but only the value is computed.
    /// By the max-flow min-cut theorem this equals
    /// [`Graph::st_mincut_value`]. `capacity: None` means unit capacities.
    ///
    /// Time complexity: O(|V|³).
    ///
    /// Binds [`igraph_maxflow_value`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_maxflow_value).
    ///
    /// # Errors
    ///
    /// As for [`Graph::maxflow`].
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Two parallel routes from 0 to 3, each able to carry one unit.
    /// let g = Graph::from_edges(&[(0, 1), (1, 3), (0, 2), (2, 3)], 4, true)?;
    /// assert_eq!(g.maxflow_value(0, 3, None)?, 2.0);
    /// assert_eq!(g.maxflow_value(0, 3, Some(&[5.0, 1.0, 2.0, 7.0]))?, 3.0);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn maxflow_value(
        &self,
        source: VertexId,
        target: VertexId,
        capacity: Option<&[f64]>,
    ) -> Result<f64> {
        self.maxflow_value_with_stats(source, target, capacity)
            .map(|(value, _)| value)
    }

    /// Value of the maximum flow between `source` and `target`, together with
    /// the operation counts of the push-relabel solver.
    ///
    /// See [`Graph::maxflow_value`] and [`MaxflowStats`].
    ///
    /// Binds [`igraph_maxflow_value`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_maxflow_value).
    ///
    /// # Errors
    ///
    /// As for [`Graph::maxflow`].
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// let g = Graph::famous("Zachary")?;
    /// let (value, stats) = g.maxflow_value_with_stats(0, 33, None)?;
    /// assert_eq!(value, 10.0);
    /// assert!(stats.pushes > 0);
    /// assert!(stats.bfs_runs >= 1); // the initial global relabelling
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn maxflow_value_with_stats(
        &self,
        source: VertexId,
        target: VertexId,
        capacity: Option<&[f64]>,
    ) -> Result<(f64, MaxflowStats)> {
        check_capacity(self, capacity)?;
        check_pair(self, source, target)?;
        let cap = capacity.map(Vector::view);
        let cap_ptr = cap.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let mut value = 0.0;
        let mut stats = empty_stats();
        igraph_call!(igraph_maxflow_value(
            self, &mut value, source, target, cap_ptr, &mut stats
        ))?;
        Ok((value, stats.into()))
    }

    /// Minimum cut between a source and a target vertex.
    ///
    /// Finds the edge set of smallest total capacity whose removal eliminates
    /// all (directed, in directed graphs) paths from `source` to `target`,
    /// together with the two sides of the cut (the first contains the source,
    /// the second the target). Computed with [`Graph::maxflow`].
    /// `capacity: None` means unit capacities.
    ///
    /// Binds [`igraph_st_mincut`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_st_mincut).
    ///
    /// # Errors
    ///
    /// As for [`Graph::maxflow`].
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // A "barbell": two triangles joined by the bridge 2 - 3.
    /// let g = Graph::from_edges(
    ///     &[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)], 6, false)?;
    /// let cut = g.st_mincut(0, 5, None)?;
    /// assert_eq!(cut.value, 1.0);
    /// assert_eq!(cut.cut, vec![3]);
    /// assert_eq!(cut.partition, vec![0, 1, 2]);
    /// assert_eq!(cut.partition2, vec![3, 4, 5]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn st_mincut(
        &self,
        source: VertexId,
        target: VertexId,
        capacity: Option<&[f64]>,
    ) -> Result<Cut> {
        check_capacity(self, capacity)?;
        check_pair(self, source, target)?;
        let cap = capacity.map(Vector::view);
        let cap_ptr = cap.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let mut value = 0.0;
        let mut cut = VectorInt::new();
        let mut partition = VectorInt::new();
        let mut partition2 = VectorInt::new();
        igraph_call!(igraph_st_mincut(
            self,
            &mut value,
            &mut cut,
            &mut partition,
            &mut partition2,
            source,
            target,
            cap_ptr,
        ))?;
        Ok(Cut {
            value,
            cut: cut.into(),
            partition: partition.into(),
            partition2: partition2.into(),
        })
    }

    /// Value of the minimum cut between a source and a target vertex.
    ///
    /// The minimum total capacity of edges to remove in order to eliminate
    /// all paths from `source` to `target` (directed paths in directed
    /// graphs). Equal to [`Graph::maxflow_value`] by the max-flow min-cut
    /// theorem. `capacity: None` means unit capacities.
    ///
    /// Time complexity: O(|V|³).
    ///
    /// Binds [`igraph_st_mincut_value`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_st_mincut_value).
    ///
    /// # Errors
    ///
    /// As for [`Graph::maxflow`].
    ///
    /// # Examples
    ///
    /// The network of igraph's `igraph_st_mincut_value.c` unit test:
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// let g = Graph::from_edges(
    ///     &[(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)], 6, true)?;
    /// let capacity = [5.0, 2.0, 2.0, 3.0, 4.0, 1.0, 2.0, 5.0];
    /// assert_eq!(g.st_mincut_value(0, 5, Some(&capacity))?, 7.0);
    /// assert_eq!(g.maxflow_value(0, 5, Some(&capacity))?, 7.0);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn st_mincut_value(
        &self,
        source: VertexId,
        target: VertexId,
        capacity: Option<&[f64]>,
    ) -> Result<f64> {
        check_capacity(self, capacity)?;
        check_pair(self, source, target)?;
        let cap = capacity.map(Vector::view);
        let cap_ptr = cap.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let mut value = 0.0;
        igraph_call!(igraph_st_mincut_value(
            self, &mut value, source, target, cap_ptr
        ))?;
        Ok(value)
    }

    /// Minimum cut of the whole graph.
    ///
    /// The set of edges of minimum total capacity whose removal disconnects
    /// the graph (makes it not strongly connected, for directed graphs),
    /// with the two resulting vertex sides. Undirected graphs use the
    /// Stoer–Wagner algorithm (J. ACM 44, 1997), in
    /// O(|V||E| + |V|² log |V|); directed graphs compute 2|V| − 2 maximum
    /// flows, O(|V|⁴). `capacity: None` means unit capacities.
    ///
    /// If the graph is already disconnected the value is 0 and the cut empty.
    /// Degenerate graphs follow igraph: with a single vertex (or a directed
    /// graph with no vertices) the value is `f64::INFINITY` and the cut empty,
    /// while an undirected graph with no vertices counts as disconnected
    /// (value 0, everything empty).
    ///
    /// Note: for directed graphs `igraph_mincut` discards the error code of
    /// its internal max-flow computations (still the case in igraph 1.0.0 and
    /// 1.0.1). The only such errors not already excluded by the argument
    /// checks of this wrapper are out-of-memory conditions.
    ///
    /// See also [`Graph::edge_connectivity`] (the unit-capacity value) and
    /// [`Graph::bridges`] (all the edges forming a cut of size one).
    ///
    /// Binds [`igraph_mincut`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_mincut).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if the capacity vector has the wrong length
    /// or an entry that is negative, NaN or infinite.
    ///
    /// # Examples
    ///
    /// The weighted example of igraph's `igraph_mincut.c`:
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// let g = Graph::from_edges(&[
    ///     (0, 1), (0, 4), (1, 2), (1, 4), (1, 5), (2, 3),
    ///     (2, 6), (3, 6), (3, 7), (4, 5), (5, 6), (6, 7),
    /// ], 8, false)?;
    /// let w = [2.0, 3.0, 3.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 3.0, 1.0, 3.0];
    /// let cut = g.mincut(Some(&w))?;
    /// assert_eq!(cut.value, 4.0);
    /// assert_eq!(cut.partition, vec![2, 3, 6, 7]);
    /// assert_eq!(cut.partition2, vec![0, 1, 4, 5]);
    /// assert_eq!(cut.cut, vec![2, 10]); // 1-2 and 5-6
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn mincut(&self, capacity: Option<&[f64]>) -> Result<Cut> {
        check_capacity(self, capacity)?;
        if self.vcount() == 0 && !self.is_directed() {
            // igraph (1.0.0 and 1.0.1) handles this case as "disconnected"
            // and sizes the first side from `csize[0]` of an *empty*
            // component-size vector, i.e. past its logical end. Answer
            // directly with the same result instead of relying on that.
            return Ok(Cut {
                value: 0.0,
                cut: vec![],
                partition: vec![],
                partition2: vec![],
            });
        }
        let cap = capacity.map(Vector::view);
        let cap_ptr = cap.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let mut value = 0.0;
        let mut cut = VectorInt::new();
        let mut partition = VectorInt::new();
        let mut partition2 = VectorInt::new();
        igraph_call!(igraph_mincut(
            self,
            &mut value,
            &mut partition,
            &mut partition2,
            &mut cut,
            cap_ptr,
        ))?;
        Ok(Cut {
            value,
            cut: cut.into(),
            partition: partition.into(),
            partition2: partition2.into(),
        })
    }

    /// Value of the minimum cut of the whole graph.
    ///
    /// The minimum total capacity of edges whose removal makes the graph not
    /// strongly connected (0 if it is already disconnected). Uses
    /// Stoer–Wagner for undirected graphs, O(log|V| · |V|²), and maximum flows
    /// from a fixed vertex in both directions for directed graphs, O(|V|⁴).
    /// For a single vertex, or a directed graph without vertices, the result
    /// is `f64::INFINITY`; an undirected graph without vertices counts as
    /// disconnected and gives 0. `capacity: None` means unit capacities.
    ///
    /// Binds [`igraph_mincut_value`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_mincut_value).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if the capacity vector has the wrong length
    /// or an entry that is negative, NaN or infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // A directed cycle is strongly connected, but a single edge breaks it.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, true)?;
    /// assert_eq!(g.mincut_value(None)?, 1.0);
    /// assert_eq!(g.mincut_value(Some(&[4.0, 2.5, 3.0]))?, 2.5);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn mincut_value(&self, capacity: Option<&[f64]>) -> Result<f64> {
        check_capacity(self, capacity)?;
        let cap = capacity.map(Vector::view);
        let cap_ptr = cap.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let mut value = 0.0;
        igraph_call!(igraph_mincut_value(self, &mut value, cap_ptr))?;
        Ok(value)
    }

    /// Vertex connectivity of a pair of vertices.
    ///
    /// The minimum number of vertices whose deletion eliminates all paths
    /// from `source` to `target` (directed paths in directed graphs). When
    /// the two vertices are not adjacent this equals the number of internally
    /// vertex-disjoint paths between them (Menger's theorem).
    ///
    /// Adjacent vertices cannot be separated by removing vertices; `neighbors`
    /// decides what happens then:
    /// [`VconnNei::Error`] fails with an error, [`VconnNei::Negative`] returns
    /// −1, [`VconnNei::NumberOfNodes`] returns the number of vertices, and
    /// [`VconnNei::Ignore`] ignores the direct edges and counts the vertices
    /// needed to break every other path. This is why the result is signed.
    ///
    /// Time complexity: O(|V|³).
    ///
    /// See also [`Graph::vertex_disjoint_paths`] (which counts the direct
    /// edges too) and [`Graph::is_separator`] to check a candidate vertex set.
    ///
    /// Binds [`igraph_st_vertex_connectivity`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_st_vertex_connectivity).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] for invalid ids, [`ErrorKind::InvalidValue`]
    /// if `source == target` or the vertices are adjacent and `neighbors` is
    /// [`VconnNei::Error`].
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // In the complete graph K6, two adjacent vertices are joined by 4 other
    /// // paths of length two.
    /// let k6 = Graph::full(6, false, false)?;
    /// assert_eq!(k6.st_vertex_connectivity(0, 1, VconnNei::Ignore)?, 4);
    /// assert_eq!(k6.st_vertex_connectivity(0, 1, VconnNei::Negative)?, -1);
    /// assert_eq!(k6.st_vertex_connectivity(0, 1, VconnNei::NumberOfNodes)?, 6);
    /// assert!(k6.st_vertex_connectivity(0, 1, VconnNei::Error).is_err());
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn st_vertex_connectivity(
        &self,
        source: VertexId,
        target: VertexId,
        neighbors: VconnNei,
    ) -> Result<i64> {
        check_pair(self, source, target)?;
        let mut res = 0;
        igraph_call!(igraph_st_vertex_connectivity(
            self,
            &mut res,
            source,
            target,
            neighbors.into()
        ))?;
        Ok(res)
    }

    /// Vertex connectivity of the graph.
    ///
    /// The minimum of the vertex connectivity over all pairs of vertices,
    /// i.e. the minimum number of vertices whose removal disconnects the
    /// graph (the vertex count minus one for complete graphs). It coincides
    /// with the *group cohesion* of White and Harary, see [`Graph::cohesion`].
    ///
    /// With `checks = true` igraph first performs cheap tests: a graph that is
    /// not (strongly) connected has connectivity 0, a graph with a vertex of
    /// degree 1 has connectivity 1, and a complete graph has `n - 1`. They
    /// are recommended as the general computation is expensive, O(|V|⁵).
    ///
    /// See also [`Graph::articulation_points`] (a connected graph on at least
    /// three vertices has vertex connectivity 1 exactly when it has one) and
    /// [`Graph::minimum_size_separators`], which lists every vertex set of
    /// this size whose removal disconnects the graph.
    ///
    /// Binds [`igraph_vertex_connectivity`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_vertex_connectivity).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // A cycle survives the removal of any single vertex, but not of two.
    /// let c = Graph::ring(5, false, false, true)?;
    /// assert_eq!(c.vertex_connectivity(true)?, 2);
    /// assert_eq!(c.vertex_connectivity(false)?, 2);
    /// // The five minimum separators are the pairs of non-adjacent vertices.
    /// assert_eq!(c.minimum_size_separators()?.len(), 5);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn vertex_connectivity(&self, checks: bool) -> Result<usize> {
        let mut res = 0;
        igraph_call!(igraph_vertex_connectivity(self, &mut res, checks))?;
        Ok(count(res))
    }

    /// Edge connectivity of a pair of vertices.
    ///
    /// The minimum number of edges to delete in order to eliminate all paths
    /// from `source` to `target` (directed paths in directed graphs); it is
    /// the unit-capacity maximum flow between them, and equals
    /// [`Graph::edge_disjoint_paths`].
    ///
    /// Time complexity: O(|V|³).
    ///
    /// Binds [`igraph_st_edge_connectivity`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_st_edge_connectivity).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] for invalid ids, [`ErrorKind::InvalidValue`]
    /// if `source == target`.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Two vertices of a 4-cycle are joined by two edge-disjoint routes.
    /// let c = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], 4, false)?;
    /// assert_eq!(c.st_edge_connectivity(0, 2)?, 2);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn st_edge_connectivity(&self, source: VertexId, target: VertexId) -> Result<usize> {
        check_pair(self, source, target)?;
        let mut res = 0;
        igraph_call!(igraph_st_edge_connectivity(self, &mut res, source, target))?;
        Ok(count(res))
    }

    /// Edge connectivity of the graph.
    ///
    /// The minimum of the edge connectivity over all pairs of vertices, i.e.
    /// the minimum number of edges whose removal disconnects the graph. It is
    /// the *group adhesion* of White and Harary, see [`Graph::adhesion`].
    /// Graphs with at most one vertex have edge connectivity 0.
    ///
    /// With `checks = true` igraph first checks connectivity (0 if the graph
    /// is not (strongly) connected) and minimum degree (1 if some vertex has
    /// degree 1), which is much cheaper than the full computation.
    ///
    /// Time complexity: O(log|V| · |V|²) for undirected graphs, O(|V|⁴) for
    /// directed graphs.
    ///
    /// See also [`Graph::bridges`] (a connected undirected graph has edge
    /// connectivity 1 exactly when it has a bridge) and [`Graph::mincut`],
    /// which also returns a cut realising the minimum.
    ///
    /// Binds [`igraph_edge_connectivity`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_edge_connectivity).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Two triangles sharing only vertex 2 ("bowtie"): no bridge, so edge
    /// // connectivity 2, but vertex 2 is a cut vertex.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)], 5, false)?;
    /// assert_eq!(g.edge_connectivity(true)?, 2);
    /// assert_eq!(g.vertex_connectivity(true)?, 1);
    /// assert!(g.bridges()?.is_empty());
    /// assert_eq!(g.articulation_points()?, vec![2]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn edge_connectivity(&self, checks: bool) -> Result<usize> {
        let mut res = 0;
        igraph_call!(igraph_edge_connectivity(self, &mut res, checks))?;
        Ok(count(res))
    }

    /// Maximum number of edge-disjoint paths between two vertices.
    ///
    /// Paths are edge-disjoint when they share no edge; directed paths are
    /// considered in directed graphs. The number equals the edge connectivity
    /// of the pair (see [`Graph::st_edge_connectivity`]) and is computed with
    /// maximum flows, in O(|V|³).
    ///
    /// Binds [`igraph_edge_disjoint_paths`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_edge_disjoint_paths).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] for invalid ids,
    /// [`ErrorKind::Unimplemented`] if `source == target`.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Bowtie: both triangles pass through vertex 2, yet two edge-disjoint
    /// // routes exist from 0 to 3.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)], 5, false)?;
    /// assert_eq!(g.edge_disjoint_paths(0, 3)?, 2);
    /// assert_eq!(g.vertex_disjoint_paths(0, 3)?, 1);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn edge_disjoint_paths(&self, source: VertexId, target: VertexId) -> Result<usize> {
        check_pair(self, source, target)?;
        let mut res = 0;
        igraph_call!(igraph_edge_disjoint_paths(self, &mut res, source, target))?;
        Ok(count(res))
    }

    /// Maximum number of vertex-disjoint paths between two vertices.
    ///
    /// Paths are vertex-disjoint when they share no vertex other than their
    /// endpoints; directed paths are considered in directed graphs. When
    /// `source` and `target` are not adjacent this is their vertex
    /// connectivity; every direct edge from `source` to `target` (either
    /// orientation in undirected graphs, parallel edges counted separately)
    /// contributes one extra path. Computed with maximum flows, O(|V|³).
    ///
    /// Binds [`igraph_vertex_disjoint_paths`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_vertex_disjoint_paths).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] for invalid ids,
    /// [`ErrorKind::Unimplemented`] if `source == target`.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Triangle: the direct edge plus the path through the third vertex.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, false)?;
    /// assert_eq!(g.vertex_disjoint_paths(0, 1)?, 2);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn vertex_disjoint_paths(&self, source: VertexId, target: VertexId) -> Result<usize> {
        check_pair(self, source, target)?;
        let mut res = 0;
        igraph_call!(igraph_vertex_disjoint_paths(self, &mut res, source, target))?;
        Ok(count(res))
    }

    /// Graph adhesion: the edge connectivity with uniform edge weights.
    ///
    /// Defined by White and Harary (Sociological Methodology 31, 2001); it is
    /// the same as [`Graph::edge_connectivity`]. `checks` enables the same
    /// cheap connectivity/degree shortcuts.
    ///
    /// Time complexity: O(log|V| · |V|²) for undirected graphs, O(|V|⁴) for
    /// directed ones.
    ///
    /// Binds [`igraph_adhesion`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_adhesion).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Every vertex of the Petersen graph has three neighbours, and no
    /// // smaller edge set disconnects it.
    /// let petersen = Graph::famous("Petersen")?;
    /// assert_eq!(petersen.adhesion(true)?, 3);
    /// assert_eq!(petersen.adhesion(true)?, petersen.edge_connectivity(false)?);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn adhesion(&self, checks: bool) -> Result<usize> {
        let mut res = 0;
        igraph_call!(igraph_adhesion(self, &mut res, checks))?;
        Ok(count(res))
    }

    /// Graph cohesion: the vertex connectivity of the graph.
    ///
    /// Defined by White and Harary (Sociological Methodology 31, 2001); it is
    /// the same as [`Graph::vertex_connectivity`]. `checks` enables the same
    /// cheap connectivity/degree/completeness shortcuts.
    ///
    /// Time complexity: O(|V|⁴), more like O(|V|²) in practice.
    ///
    /// See also [`Graph::cohesive_blocks`], the hierarchy of maximally
    /// cohesive subgraphs built on this measure.
    ///
    /// Binds [`igraph_cohesion`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_cohesion).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // The 3-dimensional cube is 3-connected in both senses.
    /// let cube = Graph::hypercube(3, false)?;
    /// assert_eq!(cube.cohesion(true)?, 3);
    /// assert_eq!(cube.adhesion(true)?, 3);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn cohesion(&self, checks: bool) -> Result<usize> {
        let mut res = 0;
        igraph_call!(igraph_cohesion(self, &mut res, checks))?;
        Ok(count(res))
    }

    /// Even–Tarjan reduction: turns vertex cuts into edge cuts.
    ///
    /// Builds a directed graph with `2 n` vertices: each vertex `i` becomes
    /// `i' = i` and `i'' = i + n`, joined by an edge `i' → i''` (these come
    /// first, capacity 1). Each original edge `(i, j)` becomes the two edges
    /// `i'' → j'` and `j'' → i'` (capacity `n`, standing for infinity). A
    /// minimum `i'' → j'` cut in the reduced graph then corresponds to a
    /// minimum vertex separator of `i` and `j` in the original graph (Even and
    /// Tarjan, SIAM J. Comput. 4(4), 1975; Kanevsky, Networks 23, 1993).
    ///
    /// Directedness of the input is not checked; the reduction is normally
    /// applied to directed graphs.
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// Binds [`igraph_even_tarjan_reduction`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_even_tarjan_reduction).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Path 0 - 1 - 2: vertex 1 separates 0 from 2.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, false)?;
    /// let et = g.even_tarjan_reduction()?;
    /// assert_eq!((et.graph.vcount(), et.graph.ecount()), (6, 7));
    /// assert_eq!(et.capacity, vec![1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0]);
    /// // Flow from 0'' (= 3) to 2' (= 2): one vertex must be removed.
    /// assert_eq!(et.graph.maxflow_value(3, 2, Some(&et.capacity))?, 1.0);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn even_tarjan_reduction(&self) -> Result<EvenTarjanReduction> {
        let mut capacity = Vector::new();
        let graph =
            Graph::init_with(|g| unsafe { igraph_even_tarjan_reduction(self, g, &mut capacity) })?;
        Ok(EvenTarjanReduction {
            graph,
            capacity: capacity.into(),
        })
    }

    /// Residual graph of a flow.
    ///
    /// Given the edge `capacity` and a `flow` on each edge (e.g.
    /// [`MaxFlow::flow`] of a directed graph), returns the directed graph
    /// containing, in edge-id order, every edge whose flow is strictly below
    /// its capacity, together with its residual capacity
    /// `capacity - flow`. The vertex set is unchanged.
    ///
    /// Binds `igraph_residual_graph` (undocumented in the C reference manual,
    /// see [the Flows chapter](https://igraph.org/c/html/latest/igraph-Flows.html)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if `capacity` or `flow` do not have one
    /// entry per edge, or if a capacity is negative, NaN or infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 2)], 3, true)?;
    /// let capacity = [2.0, 1.0, 1.0];
    /// let mf = g.maxflow(0, 2, Some(&capacity))?;
    /// assert_eq!(mf.flow, vec![1.0, 1.0, 1.0]);
    /// let res = g.residual_graph(&capacity, &mf.flow)?;
    /// // Only 0 -> 1 is not saturated.
    /// assert_eq!(res.graph.edge_list(), vec![(0, 1)]);
    /// assert_eq!(res.capacity, vec![1.0]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn residual_graph(&self, capacity: &[f64], flow: &[f64]) -> Result<ResidualGraph> {
        check_capacity(self, Some(capacity))?;
        check_len("flow", flow, self.ecount())?;
        let cap = Vector::view(capacity);
        let fl = Vector::view(flow);
        let mut residual_capacity = Vector::new();
        let graph = Graph::init_with(|g| unsafe {
            igraph_residual_graph(self, cap.as_ptr(), g, &mut residual_capacity, fl.as_ptr())
        })?;
        Ok(ResidualGraph {
            graph,
            capacity: residual_capacity.into(),
        })
    }

    /// Reverse residual graph of a flow.
    ///
    /// For every edge `u → v` of the input, in edge-id order, the result
    /// contains `u → v` if the edge carries a positive flow, and `v → u` if
    /// its flow is below its capacity. This is the graph used by
    /// [`Graph::all_st_mincuts`] to enumerate the minimum cuts.
    /// `capacity: None` means unit capacities.
    ///
    /// Binds `igraph_reverse_residual_graph` (undocumented in the C reference
    /// manual, see [the Flows chapter](https://igraph.org/c/html/latest/igraph-Flows.html)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if `capacity` or `flow` do not have one
    /// entry per edge, or if a capacity is negative, NaN or infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 2)], 3, true)?;
    /// let rr = g.reverse_residual_graph(Some(&[2.0, 1.0, 1.0]), &[1.0, 1.0, 1.0])?;
    /// assert_eq!(rr.edge_list(), vec![(0, 1), (1, 0), (1, 2), (0, 2)]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn reverse_residual_graph(&self, capacity: Option<&[f64]>, flow: &[f64]) -> Result<Graph> {
        check_capacity(self, capacity)?;
        check_len("flow", flow, self.ecount())?;
        let cap = capacity.map(Vector::view);
        let cap_ptr = cap.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let fl = Vector::view(flow);
        Graph::init_with(|g| unsafe {
            igraph_reverse_residual_graph(self, cap_ptr, g, fl.as_ptr())
        })
    }

    /// Dominator tree of a flowgraph rooted at `root`.
    ///
    /// In a directed graph where every vertex is reachable from `root`, a
    /// vertex `v` *dominates* `w ≠ v` if every path from the root to `w`
    /// passes through `v`. The *immediate dominator* `idom(w)` is the
    /// dominator of `w` dominated by all its other dominators; the edges
    /// `idom(w) → w` form a tree rooted at `root`, and `v` dominates `w` iff
    /// `v` is an ancestor of `w` in it. Implemented with the Lengauer–Tarjan
    /// algorithm (ACM TOPLAS 1, 1979), in O(|V| + |E| α(|E|, |V|)).
    ///
    /// `mode` must be [`NeighborMode::Out`] or [`NeighborMode::In`]; with
    /// `In` all edges are followed backwards (post-dominators). Vertices not
    /// reachable from the root are reported in
    /// [`DominatorTree::leftout`] and are isolated in the tree.
    ///
    /// See also [`Graph::subcomponent`] for plain reachability from the root.
    ///
    /// Binds [`igraph_dominator_tree`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_dominator_tree).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] for an invalid root,
    /// [`ErrorKind::InvalidValue`] for undirected graphs or
    /// `mode == NeighborMode::All`.
    ///
    /// # Examples
    ///
    /// The example of igraph's `dominator_tree.c`:
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// let g = Graph::from_edges(&[
    ///     (0, 9), (1, 0), (1, 2), (2, 3), (2, 7), (3, 1), (4, 1), (4, 3),
    ///     (5, 2), (5, 3), (5, 4), (5, 8), (6, 5), (6, 9), (8, 7),
    /// ], 10, true)?;
    /// let dt = g.dominator_tree(9, NeighborMode::In)?;
    /// assert_eq!(dt.dom[..3], [Some(9), Some(0), Some(3)]);
    /// assert_eq!(dt.dom[9], None); // the root
    /// assert_eq!(dt.leftout, vec![7, 8]);
    /// assert_eq!(dt.dominators(2), vec![3, 1, 0, 9]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn dominator_tree(&self, root: VertexId, mode: NeighborMode) -> Result<DominatorTree> {
        check_vertex(self, root, "root")?;
        let mut dom = VectorInt::new();
        let mut leftout = VectorInt::new();
        let tree = Graph::init_with(|g| unsafe {
            igraph_dominator_tree(self, root, &mut dom, g, &mut leftout, mode.into())
        })?;
        Ok(DominatorTree {
            dom: dom.iter().map(|&d| (d >= 0).then_some(d)).collect(),
            tree,
            leftout: leftout.into(),
        })
    }

    /// Lists all minimal edge cuts between `source` and `target` in a
    /// directed graph.
    ///
    /// Following Provan and Shier (Algorithmica 15, 1996), an s-t cut here is
    /// a *minimal* set of edges whose removal leaves no directed path from
    /// `source` to `target`: supersets of a cut are not listed (on the path
    /// `0 → 1 → 2` the cuts are `{0 → 1}` and `{1 → 2}`, not both edges
    /// together). Every such cut is listed exactly once, both as its set of
    /// edges and as the vertex set `X ∋ source` generating it (the cut is the
    /// set of edges leaving `X`). Runs in O(n (|V| + |E|)) where n is the
    /// number of cuts — which can be exponential in the size of the graph.
    ///
    /// When `target` is not reachable from `source`, or `source == target`,
    /// igraph lists no cuts at all (the result is empty, not an error).
    ///
    /// Binds [`igraph_all_st_cuts`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_all_st_cuts).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Unimplemented`] for undirected graphs,
    /// [`ErrorKind::InvalidVertexId`] for invalid ids.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // On a directed path, each edge alone is a cut.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, true)?;
    /// let all = g.all_st_cuts(0, 3)?;
    /// assert_eq!(all.cuts, vec![vec![0], vec![1], vec![2]]);
    /// assert_eq!(all.partition1s, vec![vec![0], vec![0, 1], vec![0, 1, 2]]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn all_st_cuts(&self, source: VertexId, target: VertexId) -> Result<StCuts> {
        check_pair(self, source, target)?;
        let mut cuts = VectorIntList::new();
        let mut partition1s = VectorIntList::new();
        igraph_call!(igraph_all_st_cuts(
            self,
            &mut cuts,
            &mut partition1s,
            source,
            target
        ))?;
        Ok(StCuts {
            cuts: cuts.into(),
            partition1s: partition1s.into(),
        })
    }

    /// Lists all minimum-capacity edge cuts between `source` and `target` in
    /// a directed graph.
    ///
    /// Several cuts may share the minimum total capacity. Each is returned as
    /// its edge set and as the generating vertex set `X ∋ source`. Capacities
    /// must be strictly positive (`None` means unit capacities); integer
    /// capacities are recommended, as round-off may hide some cuts otherwise.
    /// Uses a maximum flow followed by the Provan–Shier enumeration on the
    /// [reverse residual graph](Graph::reverse_residual_graph). When `target`
    /// is not reachable from `source` the value is 0 and no cut is listed.
    ///
    /// Binds [`igraph_all_st_mincuts`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_all_st_mincuts).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Unimplemented`] for undirected graphs,
    /// [`ErrorKind::InvalidVertexId`] for invalid ids,
    /// [`ErrorKind::InvalidValue`] if `source == target`, or for a capacity
    /// vector of the wrong length or with non-positive or non-finite entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Two parallel branches 1->2->4 and 1->3->4 between a single entry
    /// // edge 0->1 and a single exit edge 4->5.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)], 6, true)?;
    /// let m = g.all_st_mincuts(0, 5, None)?;
    /// assert_eq!(m.value, 1.0);
    /// assert_eq!(m.cuts, vec![vec![0], vec![5]]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn all_st_mincuts(
        &self,
        source: VertexId,
        target: VertexId,
        capacity: Option<&[f64]>,
    ) -> Result<StMinCuts> {
        check_capacity(self, capacity)?;
        check_pair(self, source, target)?;
        let cap = capacity.map(Vector::view);
        let cap_ptr = cap.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let mut value = 0.0;
        let mut cuts = VectorIntList::new();
        let mut partition1s = VectorIntList::new();
        igraph_call!(igraph_all_st_mincuts(
            self,
            &mut value,
            &mut cuts,
            &mut partition1s,
            source,
            target,
            cap_ptr,
        ))?;
        Ok(StMinCuts {
            value,
            cuts: cuts.into(),
            partition1s: partition1s.into(),
        })
    }

    /// Gomory–Hu tree of an undirected graph.
    ///
    /// A tree on the same vertices whose edges are annotated with flow values
    /// such that, for every pair `(u, v)`, the maximum flow (minimum cut)
    /// between `u` and `v` in the original graph is the minimum annotation
    /// along the tree path from `u` to `v` (see
    /// [`GomoryHuTree::flow_between`]). All `n (n - 1) / 2` pairwise flows are
    /// thus encoded by `n - 1` numbers. Built with Gusfield's algorithm (SIAM
    /// J. Comput. 19(1), 1990) using `n - 1` max-flow computations, O(|V|⁴).
    /// `capacity: None` means unit capacities. The smallest annotation is the
    /// global minimum cut value ([`Graph::mincut_value`]).
    ///
    /// Binds [`igraph_gomory_hu_tree`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_gomory_hu_tree).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] for directed graphs or a capacity vector
    /// of the wrong length or with a negative, NaN or infinite entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2), (2, 3)], 4, false)?;
    /// let gh = g.gomory_hu_tree(None)?;
    /// assert_eq!(gh.tree.ecount(), 3);
    /// assert_eq!(gh.flow_between(0, 1), Some(2.0)); // inside the triangle
    /// assert_eq!(gh.flow_between(0, 3), Some(1.0)); // through the pendant edge
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn gomory_hu_tree(&self, capacity: Option<&[f64]>) -> Result<GomoryHuTree> {
        check_capacity(self, capacity)?;
        let cap = capacity.map(Vector::view);
        let cap_ptr = cap.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let mut flows = Vector::new();
        let tree =
            Graph::init_with(|g| unsafe { igraph_gomory_hu_tree(self, g, &mut flows, cap_ptr) })?;
        Ok(GomoryHuTree {
            tree,
            flows: flows.into(),
        })
    }
}

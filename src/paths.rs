//! Shortest paths, distances, eccentricity, efficiency, random walks and
//! related path algorithms (`igraph_paths.h`).
//!
//! Everything here is a method of [`Graph`] (or a free function when no graph
//! is involved). Distances are returned as a [`Matrix`] (rows are sources,
//! columns are targets, unreachable pairs hold `f64::INFINITY`), paths as
//! `Vec<VertexId>` / `Vec<EdgeId>`, and multi-output functions return small
//! named structs such as [`ShortestPaths`] or [`Diameter`].
//!
//! Edge weights are always passed as `Option<&[f64]>` indexed by edge id:
//! `None` means an unweighted graph (every edge has length one). Most
//! functions ignore edges with a positive infinite weight.
//!
//! # Example
//!
//! ```
//! use igraph::prelude::*;
//!
//! // A weighted directed "diamond" with a shortcut: 0 → 1 → 3 and 0 → 2 → 3.
//! let g = Graph::from_edges(&[(0, 1), (1, 3), (0, 2), (2, 3), (0, 3)], 4, true).unwrap();
//! let w = [1.0, 1.0, 2.0, 2.0, 5.0];
//!
//! // Unweighted: the direct edge wins.
//! assert_eq!(g.get_shortest_path(0, 3, None, NeighborMode::Out).unwrap().vertices, vec![0, 3]);
//! // Weighted: go through vertex 1 (length 2 instead of 5).
//! let p = g.get_shortest_path(0, 3, Some(&w), NeighborMode::Out).unwrap();
//! assert_eq!(p.vertices, vec![0, 1, 3]);
//! assert_eq!(p.edges, vec![0, 1]);
//!
//! let d = g.distances(0, .., Some(&w), NeighborMode::Out).unwrap();
//! assert_eq!(d.row(0), vec![0.0, 1.0, 2.0, 2.0]);
//!
//! // The three shortest routes from 0 to 3, by increasing length.
//! let k = g.get_k_shortest_paths(0, 3, 3, Some(&w), NeighborMode::Out).unwrap();
//! let routes: Vec<_> = k.into_iter().map(|p| p.vertices).collect();
//! assert_eq!(routes, vec![vec![0, 1, 3], vec![0, 2, 3], vec![0, 3]]);
//!
//! // Directed diameter (the longest shortest path) of the unweighted graph.
//! assert_eq!(g.diameter().unwrap(), 1.0);
//! ```
//!
//! # Provided functionality
//!
//! | Topic | Methods |
//! |-------|---------|
//! | Distance matrices | [`distances`](Graph::distances), [`distances_cutoff`](Graph::distances_cutoff), [`distances_dijkstra`](Graph::distances_dijkstra), [`distances_dijkstra_cutoff`](Graph::distances_dijkstra_cutoff), [`distances_bellman_ford`](Graph::distances_bellman_ford), [`distances_johnson`](Graph::distances_johnson), [`distances_floyd_warshall`](Graph::distances_floyd_warshall) |
//! | One shortest path per target | [`get_shortest_paths`](Graph::get_shortest_paths), [`get_shortest_paths_dijkstra`](Graph::get_shortest_paths_dijkstra), [`get_shortest_paths_bellman_ford`](Graph::get_shortest_paths_bellman_ford) |
//! | A single shortest path | [`get_shortest_path`](Graph::get_shortest_path), [`get_shortest_path_dijkstra`](Graph::get_shortest_path_dijkstra), [`get_shortest_path_bellman_ford`](Graph::get_shortest_path_bellman_ford), [`get_shortest_path_astar`](Graph::get_shortest_path_astar) |
//! | All shortest paths | [`get_all_shortest_paths`](Graph::get_all_shortest_paths), [`get_all_shortest_paths_dijkstra`](Graph::get_all_shortest_paths_dijkstra) |
//! | Other path enumerations | [`get_k_shortest_paths`](Graph::get_k_shortest_paths), [`get_all_simple_paths`](Graph::get_all_simple_paths) |
//! | Widest (bottleneck) paths | [`get_widest_paths`](Graph::get_widest_paths), [`get_widest_path`](Graph::get_widest_path), [`widest_path_widths_dijkstra`](Graph::widest_path_widths_dijkstra), [`widest_path_widths_floyd_warshall`](Graph::widest_path_widths_floyd_warshall) |
//! | Global path statistics | [`diameter`](Graph::diameter), [`diameter_with_path`](Graph::diameter_with_path), [`pseudo_diameter`](Graph::pseudo_diameter), [`radius`](Graph::radius), [`average_path_length`](Graph::average_path_length), [`average_path_length_details`](Graph::average_path_length_details), [`path_length_hist`](Graph::path_length_hist) |
//! | Vertex path statistics | [`eccentricity`](Graph::eccentricity), [`graph_center`](Graph::graph_center) |
//! | Efficiency | [`global_efficiency`](Graph::global_efficiency), [`local_efficiency`](Graph::local_efficiency), [`average_local_efficiency`](Graph::average_local_efficiency) |
//! | Miscellaneous | [`random_walk`](Graph::random_walk), [`spanner`](Graph::spanner), [`voronoi`](Graph::voronoi), [`vertex_path_from_edge_path`](Graph::vertex_path_from_edge_path), [`expand_path_to_pairs`] |
//!
//! Global statistics on a classic network, built with
//! [`Graph::famous`] from the [`constructors`](crate::constructors) module:
//!
//! ```
//! use igraph::prelude::*;
//!
//! let karate = Graph::famous("Zachary").unwrap();
//! assert_eq!(karate.diameter().unwrap(), 5.0);
//! assert_eq!(karate.radius(None, NeighborMode::All).unwrap(), 3.0);
//! let apl = karate.average_path_length(None, false, true).unwrap();
//! assert!((apl - 2.408199643).abs() < 1e-9);
//! // Histogram of the 561 vertex pairs by distance (the 78 edges first).
//! let h = karate.path_length_hist(false).unwrap();
//! assert_eq!(h.counts[0], 78.0);
//! assert_eq!(h.counts.iter().sum::<f64>(), 561.0);
//! ```
//!
//! # See also
//!
//! - [`centrality`](crate::centrality): distance based centralities such as
//!   [`closeness`](Graph::closeness),
//!   [`harmonic_centrality`](Graph::harmonic_centrality) and
//!   [`betweenness`](Graph::betweenness) (which counts shortest paths).
//! - [`visitor`](crate::visitor): breadth-first and depth-first traversals
//!   ([`bfs`](Graph::bfs), [`dfs`](Graph::dfs)), the building blocks of
//!   unweighted shortest paths.
//! - [`components`](crate::components): reachability questions
//!   ([`is_connected`](Graph::is_connected),
//!   [`reachability`](Graph::reachability)) and bounded-distance
//!   neighborhoods ([`neighborhood`](Graph::neighborhood)).
//! - [`structural`](crate::structural): [`girth`](Graph::girth) (the
//!   shortest cycle) and [`minimum_spanning_tree`](Graph::minimum_spanning_tree).
//! - [`cycles`](crate::cycles): closed walks and special walks such as
//!   [`find_cycle`](Graph::find_cycle) and
//!   [`eulerian_path`](Graph::eulerian_path) (whose edge sequences
//!   [`vertex_path_from_edge_path`](Graph::vertex_path_from_edge_path)
//!   converts to vertices).
//! - [`flow`](crate::flow): [`maxflow`](Graph::maxflow), the "capacity"
//!   counterpart of the widest paths offered here.
//! - [`community`](crate::community):
//!   [`community_voronoi`](Graph::community_voronoi), a community detection
//!   method built on [`voronoi`](Graph::voronoi).

use crate::{
    constants::{NeighborMode, RandomWalkStuck, VoronoiTiebreaker},
    error::{Error, ErrorKind, Result, catch_panic},
    ffi::*,
    ffi_enum,
    graph::{EdgeId, VertexId},
    igraph_call,
    list::VectorIntList,
    matrix::Matrix,
    selector::{EdgeSelector, VertexSelector},
    vector::{Vector, VectorInt, View},
};
use std::{ffi::c_void, ptr};

#[cfg(doc)]
use crate::graph::Graph;

ffi_enum! {
    /// Variant of the Floyd–Warshall algorithm used by
    /// [`Graph::distances_floyd_warshall`] (`igraph_floyd_warshall_algorithm_t`).
    ///
    /// The default is [`Automatic`](Self::Automatic).
    #[derive(Default)]
    pub enum FloydWarshallAlgorithm: igraph_floyd_warshall_algorithm_t {
        /// Let igraph choose the best performing variant (currently always `Tree`).
        #[default]
        Automatic = igraph_floyd_warshall_algorithm_t_IGRAPH_FLOYD_WARSHALL_AUTOMATIC,
        /// The textbook O(|V|³) Floyd–Warshall algorithm.
        Original = igraph_floyd_warshall_algorithm_t_IGRAPH_FLOYD_WARSHALL_ORIGINAL,
        /// The "Tree" speed-up of Brodnik, Grgurovič and Požar, faster in most cases.
        Tree = igraph_floyd_warshall_algorithm_t_IGRAPH_FLOYD_WARSHALL_TREE,
    }
}

/// A single path, described both by the vertices it visits and by the edges it
/// traverses.
///
/// For a path of `k` edges, `vertices` has `k + 1` entries (source and target
/// included) and `edges` has `k` entries. An empty `vertices` list means that
/// there is no path.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GraphPath {
    /// The vertex ids along the path, including both endpoints.
    pub vertices: Vec<VertexId>,
    /// The edge ids along the path.
    pub edges: Vec<EdgeId>,
}

impl GraphPath {
    /// Number of edges of the path (its unweighted length).
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Whether the path has no edges (no path at all, or a path from a vertex
    /// to itself).
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Whether a path was found, i.e. `vertices` is non-empty.
    pub fn exists(&self) -> bool {
        !self.vertices.is_empty()
    }

    /// Sum of the weights of the traversed edges (`weights` is indexed by edge id).
    ///
    /// # Panics
    /// If some edge id of the path is out of the bounds of `weights`.
    pub fn weight(&self, weights: &[f64]) -> f64 {
        self.edges.iter().map(|&e| weights[e as usize]).sum()
    }
}

/// Single-source shortest (or widest) paths towards a set of targets, as
/// returned by [`Graph::get_shortest_paths`] and friends.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ShortestPaths {
    /// `vertices[i]` lists the vertices on the path to the `i`-th target
    /// (empty if the target is unreachable).
    pub vertices: Vec<Vec<VertexId>>,
    /// `edges[i]` lists the edges on the path to the `i`-th target.
    pub edges: Vec<Vec<EdgeId>>,
    /// The shortest path tree, indexed by vertex id: the vertex from which each
    /// vertex was reached. The source has `-1`, and vertices not reached
    /// during the search have `-2` (the search stops as soon as all the
    /// targets are reached).
    pub parents: Vec<VertexId>,
    /// The shortest path tree, indexed by vertex id: the edge through which
    /// each vertex was reached; `-1` for the source and unreached vertices.
    pub inbound_edges: Vec<EdgeId>,
}

impl ShortestPaths {
    /// The `i`-th path (towards the `i`-th target) as a [`GraphPath`].
    pub fn path(&self, i: usize) -> Option<GraphPath> {
        Some(GraphPath {
            vertices: self.vertices.get(i)?.clone(),
            edges: self.edges.get(i)?.clone(),
        })
    }
}

/// All the shortest paths from one source, as returned by
/// [`Graph::get_all_shortest_paths`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AllShortestPaths {
    /// Vertex lists of all the shortest paths, grouped by target in increasing
    /// target id order. Unreachable targets contribute no path.
    pub vertices: Vec<Vec<VertexId>>,
    /// Edge lists of the same paths, in the same order.
    pub edges: Vec<Vec<EdgeId>>,
    /// Number of shortest paths from the source to each vertex (indexed by
    /// vertex id). Only accurate for the requested targets.
    pub nrgeo: Vec<i64>,
}

/// The diameter of a graph together with one longest geodesic, see
/// [`Graph::diameter_with_path`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Diameter {
    /// The diameter: the (weighted) length of the longest shortest path.
    /// `NaN` for the null graph; `INFINITY` for disconnected graphs when
    /// `unconn` is `false`.
    pub length: f64,
    /// Source of a longest geodesic (`None` if there is no such path).
    pub from: Option<VertexId>,
    /// Target of a longest geodesic (`None` if there is no such path).
    pub to: Option<VertexId>,
    /// The longest geodesic, by vertices and edges.
    pub path: GraphPath,
}

/// A pseudo-diameter and its endpoints, see [`Graph::pseudo_diameter`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PseudoDiameter {
    /// The eccentricity of the pseudo-peripheral vertex found: a lower bound
    /// of the diameter.
    pub length: f64,
    /// Source of the corresponding path (`None` if there is no such path).
    pub from: Option<VertexId>,
    /// Target of the corresponding path (`None` if there is no such path).
    pub to: Option<VertexId>,
}

/// Average shortest path length and number of disconnected pairs, see
/// [`Graph::average_path_length_details`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AveragePathLength {
    /// The average length of the shortest paths.
    pub average: f64,
    /// The number of ordered vertex pairs `(u, v)` such that `v` is not
    /// reachable from `u`.
    pub unconnected_pairs: f64,
}

/// Histogram of shortest path lengths, see [`Graph::path_length_hist`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PathLengthHistogram {
    /// `counts[i]` is the number of vertex pairs at distance `i + 1`.
    pub counts: Vec<f64>,
    /// The number of vertex pairs whose second vertex is unreachable from the first.
    pub unconnected: f64,
}

/// A Voronoi partitioning, see [`Graph::voronoi`].
///
/// Not to be confused with [`community::Voronoi`](crate::community::Voronoi),
/// the result of the Voronoi *community detection*
/// [`Graph::community_voronoi`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct VoronoiPartition {
    /// For each vertex, the *index* (into the `generators` slice) of the
    /// generator it belongs to, or `-1` when no generator reaches it.
    pub membership: Vec<i64>,
    /// For each vertex, the distance to/from its generator (`INFINITY` when unreachable).
    pub distances: Vec<f64>,
}

/// Length limits for [`Graph::get_all_simple_paths`].
///
/// `None` means "no limit"; the default has no limits at all.
///
/// ```
/// use igraph::paths::SimplePathsOptions;
/// let opts = SimplePathsOptions::default().with_max_len(3).with_max_results(10);
/// assert_eq!(opts.min_len, None);
/// assert_eq!(opts.max_len, Some(3));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SimplePathsOptions {
    /// Minimum length (number of edges) of the returned paths.
    pub min_len: Option<usize>,
    /// Maximum length (number of edges) of the returned paths.
    pub max_len: Option<usize>,
    /// Stop after this many paths have been found.
    pub max_results: Option<usize>,
}

impl SimplePathsOptions {
    /// Sets the minimum path length.
    pub fn with_min_len(mut self, min_len: usize) -> Self {
        self.min_len = Some(min_len);
        self
    }

    /// Sets the maximum path length.
    pub fn with_max_len(mut self, max_len: usize) -> Self {
        self.max_len = Some(max_len);
        self
    }

    /// Sets the maximum number of returned paths.
    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.max_results = Some(max_results);
        self
    }
}

// ---------------------------------------------------------------------------
// Private helpers.

fn weights_view(weights: Option<&[f64]>) -> Option<View<'_, Vector>> {
    weights.map(Vector::view)
}

fn weights_ptr(view: &Option<View<'_, Vector>>) -> *const igraph_vector_t {
    view.as_ref().map_or(ptr::null(), |v| v.as_ptr())
}

fn opt_id(id: igraph_int_t) -> Option<VertexId> {
    (id >= 0).then_some(id)
}

/// Converts a count to `igraph_int_t`, saturating at `igraph_int_t::MAX`
/// (a plain `as` cast would turn huge counts into negative values, which
/// igraph interprets as "unlimited" or rejects).
fn saturating_int(value: usize) -> igraph_int_t {
    igraph_int_t::try_from(value).unwrap_or(igraph_int_t::MAX)
}

fn limit(value: Option<usize>) -> igraph_int_t {
    value.map_or(IGRAPH_UNLIMITED as igraph_int_t, saturating_int)
}

/// Maps an optional vertex to igraph's "negative means automatic"
/// convention, rejecting explicit ids outside `0..vcount` (igraph would
/// silently treat negative ones as `None`, and some functions report
/// `IGRAPH_EINVAL` rather than `IGRAPH_EINVVID` for too large ones).
fn opt_vertex_arg(vertex: Option<VertexId>, vcount: usize) -> Result<igraph_int_t> {
    match vertex {
        Some(v) if !(0..vcount as VertexId).contains(&v) => Err(Error::new(
            ErrorKind::InvalidVertexId,
            format!("invalid vertex id {v}"),
        )),
        Some(v) => Ok(v),
        None => Ok(-1),
    }
}

/// igraph treats negative cutoffs as "no cutoff"; a NaN cutoff has no
/// meaning and is rejected.
fn cutoff_value(cutoff: Option<f64>) -> Result<igraph_real_t> {
    match cutoff {
        Some(c) if c.is_nan() => Err(Error::invalid("the cutoff must not be NaN")),
        Some(c) => Ok(c),
        None => Ok(-1.0),
    }
}

impl igraph_t {
    /// Checks that the weight vector, if given, has one entry per edge.
    fn paths_check_weights(&self, weights: Option<&[f64]>) -> Result<()> {
        match weights {
            Some(w) if w.len() != self.ecount() => Err(Error::invalid(format!(
                "the weight vector has length {} but the graph has {} edges",
                w.len(),
                self.ecount()
            ))),
            _ => Ok(()),
        }
    }

    /// Shared plumbing of the `igraph_distances*`-like functions.
    fn distance_matrix<'a>(
        &self,
        from: impl Into<VertexSelector<'a>>,
        to: impl Into<VertexSelector<'a>>,
        weights: Option<&[f64]>,
        call: impl FnOnce(
            *mut igraph_matrix_t,
            igraph_vs_t,
            igraph_vs_t,
            *const igraph_vector_t,
        ) -> igraph_error_t,
    ) -> Result<Matrix> {
        self.paths_check_weights(weights)?;
        let from = from.into().to_raw()?;
        let to = to.into().to_raw()?;
        let w = weights_view(weights);
        let mut res = Matrix::new();
        igraph_call!(call(&mut res, from.get(), to.get(), weights_ptr(&w)))?;
        Ok(res)
    }

    /// Shared plumbing of the `igraph_get_shortest_paths*`-like functions.
    fn shortest_paths_with<'a>(
        &self,
        to: impl Into<VertexSelector<'a>>,
        weights: Option<&[f64]>,
        call: impl FnOnce(
            *mut igraph_vector_int_list_t,
            *mut igraph_vector_int_list_t,
            igraph_vs_t,
            *const igraph_vector_t,
            *mut igraph_vector_int_t,
            *mut igraph_vector_int_t,
        ) -> igraph_error_t,
    ) -> Result<ShortestPaths> {
        self.paths_check_weights(weights)?;
        let to = to.into().to_raw()?;
        let w = weights_view(weights);
        let mut vertices = VectorIntList::new();
        let mut edges = VectorIntList::new();
        let mut parents = VectorInt::new();
        let mut inbound = VectorInt::new();
        igraph_call!(call(
            &mut vertices,
            &mut edges,
            to.get(),
            weights_ptr(&w),
            &mut parents,
            &mut inbound
        ))?;
        Ok(ShortestPaths {
            vertices: vertices.to_vecs(),
            edges: edges.to_vecs(),
            parents: parents.into(),
            inbound_edges: inbound.into(),
        })
    }

    /// Shared plumbing of the `igraph_get_shortest_path*`-like functions.
    fn single_path_with(
        &self,
        weights: Option<&[f64]>,
        call: impl FnOnce(
            *mut igraph_vector_int_t,
            *mut igraph_vector_int_t,
            *const igraph_vector_t,
        ) -> igraph_error_t,
    ) -> Result<GraphPath> {
        self.paths_check_weights(weights)?;
        let w = weights_view(weights);
        let mut vertices = VectorInt::new();
        let mut edges = VectorInt::new();
        igraph_call!(call(&mut vertices, &mut edges, weights_ptr(&w)))?;
        Ok(GraphPath {
            vertices: vertices.into(),
            edges: edges.into(),
        })
    }

    /// Shared plumbing of the `igraph_get_all_shortest_paths*` functions.
    fn all_shortest_paths_with<'a>(
        &self,
        to: impl Into<VertexSelector<'a>>,
        weights: Option<&[f64]>,
        call: impl FnOnce(
            *mut igraph_vector_int_list_t,
            *mut igraph_vector_int_list_t,
            *mut igraph_vector_int_t,
            igraph_vs_t,
            *const igraph_vector_t,
        ) -> igraph_error_t,
    ) -> Result<AllShortestPaths> {
        self.paths_check_weights(weights)?;
        let to = to.into().to_raw()?;
        let w = weights_view(weights);
        let mut vertices = VectorIntList::new();
        let mut edges = VectorIntList::new();
        let mut nrgeo = VectorInt::new();
        igraph_call!(call(
            &mut vertices,
            &mut edges,
            &mut nrgeo,
            to.get(),
            weights_ptr(&w)
        ))?;
        Ok(AllShortestPaths {
            vertices: vertices.to_vecs(),
            edges: edges.to_vecs(),
            nrgeo: nrgeo.into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Diameter, radius, eccentricity and averages.

impl igraph_t {
    /// The diameter of the graph: the length of its longest shortest path.
    ///
    /// This is the backwards-compatible shorthand of
    /// [`diameter_with_path`](Self::diameter_with_path) for the most common
    /// case: unweighted, following edge directions in directed graphs
    /// (`directed = self.is_directed()`), and, for disconnected graphs,
    /// returning the longest geodesic *within* a component (`unconn = true`).
    /// The diameter of the null graph is `NaN`.
    ///
    /// See also [`girth`](Self::girth), the length of the *shortest* cycle,
    /// and [`radius`](Self::radius), the smallest eccentricity.
    ///
    /// Binds [`igraph_diameter`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_diameter).
    /// Time complexity: O(|V| |E|).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // A path on 5 vertices has diameter 4 ...
    /// let path = Graph::path_graph(5, false, false).unwrap();
    /// assert_eq!(path.diameter().unwrap(), 4.0);
    /// // ... and closing it into a cycle halves it.
    /// let mut cycle = path.clone();
    /// cycle.add_edge(4, 0).unwrap();
    /// assert_eq!(cycle.diameter().unwrap(), 2.0);
    /// assert!(Graph::new(0, false).diameter().unwrap().is_nan());
    /// ```
    pub fn diameter(&self) -> Result<f64> {
        let mut res = 0.0;
        igraph_call!(igraph_diameter(
            self,
            ptr::null(),
            &mut res,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            self.is_directed(),
            true
        ))?;
        Ok(res)
    }

    /// The (weighted) diameter of the graph together with its endpoints and
    /// one longest geodesic.
    ///
    /// The diameter is the maximum eccentricity of the vertices, i.e. the
    /// length of the longest shortest path.
    ///
    /// - `weights`: optional edge lengths (Dijkstra's algorithm is used when
    ///   given); edges with positive infinite weight are ignored.
    /// - `directed`: whether to follow edge directions (ignored for
    ///   undirected graphs).
    /// - `unconn`: for disconnected graphs, `true` returns the longest
    ///   geodesic within a component, `false` returns `INFINITY`.
    ///
    /// The null graph has diameter `NaN`; `from`/`to` are `None` and the path
    /// is empty when there is no diameter path.
    ///
    /// Binds [`igraph_diameter`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_diameter).
    /// Time complexity: O(|V| |E|) unweighted, O(|V| |E| log |E|) weighted.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // The directed path 0 → 1 → ... → 9 (a non-circular directed ring, as
    /// // in igraph's own example).
    /// let ring = Graph::ring(10, true, false, false).unwrap();
    /// let d = ring.diameter_with_path(None, true, true).unwrap();
    /// assert_eq!(d.length, 9.0);
    /// assert_eq!((d.from, d.to), (Some(0), Some(9)));
    /// assert_eq!(d.path.vertices, (0..10).collect::<Vec<_>>());
    /// assert_eq!(d.path.edges, (0..9).collect::<Vec<_>>());
    /// ```
    pub fn diameter_with_path(
        &self,
        weights: Option<&[f64]>,
        directed: bool,
        unconn: bool,
    ) -> Result<Diameter> {
        self.paths_check_weights(weights)?;
        let w = weights_view(weights);
        let mut length = 0.0;
        let (mut from, mut to) = (-1, -1);
        let mut vpath = VectorInt::new();
        let mut epath = VectorInt::new();
        igraph_call!(igraph_diameter(
            self,
            weights_ptr(&w),
            &mut length,
            &mut from,
            &mut to,
            &mut vpath,
            &mut epath,
            directed,
            unconn
        ))?;
        Ok(Diameter {
            length,
            from: opt_id(from),
            to: opt_id(to),
            path: GraphPath {
                vertices: vpath.into(),
                edges: epath.into(),
            },
        })
    }

    /// An approximation (and lower bound) of the diameter, computed from a
    /// pseudo-peripheral vertex.
    ///
    /// A pseudo-peripheral vertex `v` is such that for every vertex `u` as far
    /// away from `v` as possible, `v` is also as far away from `u` as
    /// possible. The search starts at `start` (a random vertex when `None`);
    /// in disconnected graphs the result refers to the component of the start
    /// vertex. `directed` and `unconn` have the same meaning as in
    /// [`diameter_with_path`](Self::diameter_with_path): with `unconn =
    /// false` a disconnected graph gives `INFINITY` and no endpoints. Returns
    /// `NaN` for the null graph. With `start = None` the start vertex is drawn
    /// from the calling thread's default random number generator (seed it
    /// with [`rng::seed`](crate::rng::seed) for reproducible results).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`] if
    /// `start` is not a vertex of the graph (negative ids included).
    ///
    /// Binds [`igraph_pseudo_diameter`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_pseudo_diameter).
    /// Time complexity: O(|V| |E| log |E|).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let path = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let pd = path.pseudo_diameter(None, Some(1), false, true).unwrap();
    /// // On trees the pseudo-diameter is exact.
    /// assert_eq!(pd.length, 3.0);
    /// ```
    pub fn pseudo_diameter(
        &self,
        weights: Option<&[f64]>,
        start: Option<VertexId>,
        directed: bool,
        unconn: bool,
    ) -> Result<PseudoDiameter> {
        self.paths_check_weights(weights)?;
        let start = opt_vertex_arg(start, self.vcount())?;
        let w = weights_view(weights);
        let mut length = 0.0;
        let (mut from, mut to) = (-1, -1);
        self.with_fresh_multi_cache(|| {
            igraph_call!(igraph_pseudo_diameter(
                self,
                weights_ptr(&w),
                &mut length,
                start,
                &mut from,
                &mut to,
                directed,
                unconn
            ))
        })?;
        Ok(PseudoDiameter {
            length,
            from: opt_id(from),
            to: opt_id(to),
        })
    }

    /// Eccentricity of the selected vertices: the largest distance from (or
    /// to, depending on `mode`) each vertex to any vertex reachable from it.
    ///
    /// Vertex pairs in different components are ignored, so isolated vertices
    /// have eccentricity zero. `weights` must be non-negative and not NaN;
    /// edges with infinite weight are ignored. The maximum eccentricity is
    /// the [diameter](Self::diameter_with_path), the minimum the
    /// [radius](Self::radius). See also [`closeness`](Self::closeness), which
    /// averages the distances instead of taking their maximum.
    ///
    /// Binds [`igraph_eccentricity`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_eccentricity).
    /// Time complexity: O(|V| |E| log |V| + |V|).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // A star: the center has eccentricity 1, the leaves 2.
    /// let star = Graph::star(4, StarMode::Undirected, 0).unwrap();
    /// assert_eq!(star.eccentricity(.., None, NeighborMode::All).unwrap(), vec![1.0, 2.0, 2.0, 2.0]);
    /// // Only some vertices, in the requested order.
    /// assert_eq!(star.eccentricity(vec![3, 0], None, NeighborMode::All).unwrap(), vec![2.0, 1.0]);
    /// ```
    pub fn eccentricity<'a>(
        &self,
        vids: impl Into<VertexSelector<'a>>,
        weights: Option<&[f64]>,
        mode: NeighborMode,
    ) -> Result<Vec<f64>> {
        self.paths_check_weights(weights)?;
        let vs = vids.into().to_raw()?;
        let w = weights_view(weights);
        let mut res = Vector::new();
        self.with_fresh_multi_cache(|| {
            igraph_call!(igraph_eccentricity(
                self,
                weights_ptr(&w),
                &mut res,
                vs.get(),
                mode.into()
            ))
        })?;
        Ok(res.into())
    }

    /// The radius of the graph: the smallest eccentricity of its vertices
    /// (`NaN` for the null graph).
    ///
    /// Binds [`igraph_radius`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_radius).
    /// Time complexity: O(|V| |E| log |V| + |V|).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let path = Graph::path_graph(5, false, false).unwrap();
    /// assert_eq!(path.radius(None, NeighborMode::All).unwrap(), 2.0);
    /// // Weighted: stretching the first edge moves the center towards it.
    /// let w = [10.0, 1.0, 1.0, 1.0];
    /// assert_eq!(path.radius(Some(&w), NeighborMode::All).unwrap(), 10.0);
    /// ```
    pub fn radius(&self, weights: Option<&[f64]>, mode: NeighborMode) -> Result<f64> {
        self.paths_check_weights(weights)?;
        let w = weights_view(weights);
        let mut res = 0.0;
        self.with_fresh_multi_cache(|| {
            igraph_call!(igraph_radius(self, weights_ptr(&w), &mut res, mode.into()))
        })?;
        Ok(res)
    }

    /// The center of the graph: the vertices of minimum eccentricity.
    ///
    /// In disconnected graphs the minimum is taken across all components.
    /// This function is marked *experimental* in igraph.
    ///
    /// Binds [`igraph_graph_center`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_graph_center).
    /// Time complexity: O(|V| |E| log |V| + |V|).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // The center of a path with an even number of vertices is its middle edge.
    /// let path = Graph::path_graph(4, false, false).unwrap();
    /// assert_eq!(path.graph_center(None, NeighborMode::All).unwrap(), vec![1, 2]);
    /// ```
    pub fn graph_center(
        &self,
        weights: Option<&[f64]>,
        mode: NeighborMode,
    ) -> Result<Vec<VertexId>> {
        self.paths_check_weights(weights)?;
        let w = weights_view(weights);
        let mut res = VectorInt::new();
        self.with_fresh_multi_cache(|| {
            igraph_call!(igraph_graph_center(
                self,
                weights_ptr(&w),
                &mut res,
                mode.into()
            ))
        })?;
        Ok(res.into())
    }

    /// The average shortest path length over all ordered pairs of distinct
    /// vertices.
    ///
    /// - `weights`: optional non-negative edge lengths.
    /// - `directed`: whether to follow edge directions (ignored for
    ///   undirected graphs).
    /// - `unconn`: if `true`, only pairs connected by a path are averaged;
    ///   if `false`, disconnected graphs give `INFINITY`.
    ///
    /// Returns `NaN` when no pair can be included (e.g. fewer than two
    /// vertices). See [`average_path_length_details`](Self::average_path_length_details)
    /// to also obtain the number of disconnected pairs, and
    /// [`closeness`](Self::closeness) for the per-vertex (inverse) averages.
    ///
    /// Binds [`igraph_average_path_length`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_average_path_length).
    /// Time complexity: O(|V| |E| log |E| + |V|).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // In a triangle every pair is adjacent.
    /// let k3 = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, false).unwrap();
    /// assert_eq!(k3.average_path_length(None, false, true).unwrap(), 1.0);
    /// // Path 0-1-2: distances 1, 1, 2 → average 4/3.
    /// let p3 = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// assert!((p3.average_path_length(None, false, true).unwrap() - 4.0 / 3.0).abs() < 1e-12);
    /// ```
    pub fn average_path_length(
        &self,
        weights: Option<&[f64]>,
        directed: bool,
        unconn: bool,
    ) -> Result<f64> {
        Ok(self
            .average_path_length_details(weights, directed, unconn)?
            .average)
    }

    /// Like [`average_path_length`](Self::average_path_length), also
    /// returning the number of ordered vertex pairs `(u, v)` with `v`
    /// unreachable from `u`.
    ///
    /// Binds [`igraph_average_path_length`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_average_path_length).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // Two disjoint edges: 4 ordered connected pairs, 8 unconnected ones.
    /// let g = Graph::from_edges(&[(0, 1), (2, 3)], 4, false).unwrap();
    /// let apl = g.average_path_length_details(None, false, true).unwrap();
    /// assert_eq!(apl.average, 1.0);
    /// assert_eq!(apl.unconnected_pairs, 8.0);
    /// ```
    pub fn average_path_length_details(
        &self,
        weights: Option<&[f64]>,
        directed: bool,
        unconn: bool,
    ) -> Result<AveragePathLength> {
        self.paths_check_weights(weights)?;
        let w = weights_view(weights);
        let (mut average, mut unconnected_pairs) = (0.0, 0.0);
        igraph_call!(igraph_average_path_length(
            self,
            weights_ptr(&w),
            &mut average,
            &mut unconnected_pairs,
            directed,
            unconn
        ))?;
        Ok(AveragePathLength {
            average,
            unconnected_pairs,
        })
    }

    /// Histogram of the (unweighted) shortest path lengths between all vertex
    /// pairs.
    ///
    /// `counts[0]` is the number of pairs at distance 1, `counts[1]` at
    /// distance 2, and so on. In undirected graphs (or with `directed =
    /// false`) each unordered pair is counted once; in directed graphs with
    /// `directed = true` both directions are counted.
    ///
    /// Binds [`igraph_path_length_hist`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_path_length_hist).
    /// Time complexity: O(|V| |E|).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // A path on 4 vertices: 3 pairs at distance 1, 2 at distance 2, 1 at distance 3.
    /// let p4 = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let h = p4.path_length_hist(false).unwrap();
    /// assert_eq!(h.counts, vec![3.0, 2.0, 1.0]);
    /// assert_eq!(h.unconnected, 0.0);
    /// ```
    pub fn path_length_hist(&self, directed: bool) -> Result<PathLengthHistogram> {
        let mut counts = Vector::new();
        let mut unconnected = 0.0;
        igraph_call!(igraph_path_length_hist(
            self,
            &mut counts,
            &mut unconnected,
            directed
        ))?;
        Ok(PathLengthHistogram {
            counts: counts.into(),
            unconnected,
        })
    }

    /// The global efficiency of the network: the average of the inverse
    /// distances between all ordered pairs of distinct vertices,
    /// `E = 1/(N(N-1)) Σ_{i≠j} 1/d_ij` (Latora & Marchiori, 2001).
    ///
    /// Unreachable pairs contribute zero, so, unlike the
    /// [average path length](Self::average_path_length), it is well defined
    /// for disconnected graphs; graphs with fewer than two vertices give
    /// `NaN`. It equals the mean of the normalized
    /// [`harmonic_centrality`](Self::harmonic_centrality) of the vertices.
    ///
    /// Binds [`igraph_global_efficiency`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_global_efficiency).
    /// Time complexity: O(|V| |E|) unweighted, O(|V| |E| log |E| + |V|) weighted.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let k4 = Graph::full(4, false, false).unwrap();
    /// assert_eq!(k4.global_efficiency(None, false).unwrap(), 1.0);
    /// // Path 0-1-2: inverse distances 1, 1, 1/2 (each counted twice) → 5/6.
    /// let p3 = Graph::path_graph(3, false, false).unwrap();
    /// assert!((p3.global_efficiency(None, false).unwrap() - 5.0 / 6.0).abs() < 1e-12);
    /// ```
    pub fn global_efficiency(&self, weights: Option<&[f64]>, directed: bool) -> Result<f64> {
        self.paths_check_weights(weights)?;
        let w = weights_view(weights);
        let mut res = 0.0;
        igraph_call!(igraph_global_efficiency(
            self,
            weights_ptr(&w),
            &mut res,
            directed
        ))?;
        Ok(res)
    }

    /// The local efficiency around each selected vertex.
    ///
    /// The vertex is removed and the average inverse distance between its
    /// neighbors (through the rest of the network) is computed. Unreachable
    /// pairs contribute zero; vertices with fewer than two neighbors have
    /// local efficiency zero. `mode` selects which neighbors form the local
    /// neighborhood in directed graphs (`NeighborMode::All` is a sensible
    /// default), `directed` whether distances follow edge directions.
    /// It is a distance based analogue of the local clustering coefficient
    /// ([`transitivity_local_undirected`](Self::transitivity_local_undirected)).
    ///
    /// Binds [`igraph_local_efficiency`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_local_efficiency).
    /// Time complexity: O(|E|² log |E|) weighted, O(|E|²) unweighted.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // In a 4-cycle the two neighbors of a vertex are at distance 2 once it is removed.
    /// let c4 = Graph::cycle_graph(4, false, false).unwrap();
    /// assert_eq!(c4.local_efficiency(.., None, false, NeighborMode::All).unwrap(), vec![0.5; 4]);
    /// ```
    pub fn local_efficiency<'a>(
        &self,
        vids: impl Into<VertexSelector<'a>>,
        weights: Option<&[f64]>,
        directed: bool,
        mode: NeighborMode,
    ) -> Result<Vec<f64>> {
        self.paths_check_weights(weights)?;
        let vs = vids.into().to_raw()?;
        let w = weights_view(weights);
        let mut res = Vector::new();
        igraph_call!(igraph_local_efficiency(
            self,
            weights_ptr(&w),
            &mut res,
            vs.get(),
            directed,
            mode.into()
        ))?;
        Ok(res.into())
    }

    /// The average of the [local efficiencies](Self::local_efficiency) of all
    /// vertices (zero for the null graph).
    ///
    /// Binds [`igraph_average_local_efficiency`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_average_local_efficiency).
    /// Time complexity: O(|E|² log |E|) weighted, O(|E|²) unweighted.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let c4 = Graph::cycle_graph(4, false, false).unwrap();
    /// assert_eq!(c4.average_local_efficiency(None, false, NeighborMode::All).unwrap(), 0.5);
    /// ```
    pub fn average_local_efficiency(
        &self,
        weights: Option<&[f64]>,
        directed: bool,
        mode: NeighborMode,
    ) -> Result<f64> {
        self.paths_check_weights(weights)?;
        let w = weights_view(weights);
        let mut res = 0.0;
        igraph_call!(igraph_average_local_efficiency(
            self,
            weights_ptr(&w),
            &mut res,
            directed,
            mode.into()
        ))?;
        Ok(res)
    }
}

// ---------------------------------------------------------------------------
// Distance matrices.

impl igraph_t {
    /// Shortest path lengths between the `from` and `to` vertices.
    ///
    /// Row `i` of the result holds the distances from the `i`-th source to
    /// every target (`INFINITY` if unreachable). `to` must not contain
    /// duplicates. `mode` chooses outgoing (`Out`), incoming (`In`) or
    /// undirected (`All`) paths in directed graphs.
    ///
    /// With `weights = None` a BFS is used; with weights, igraph picks the
    /// most suitable algorithm: Floyd–Warshall for dense all-pairs problems,
    /// Dijkstra for non-negative weights, and Bellman–Ford or Johnson when
    /// negative weights are present.
    ///
    /// See also [`bfs`](Self::bfs) (which also reports BFS distances),
    /// [`neighborhood`](Self::neighborhood) (the vertices within a given
    /// distance) and [`closeness`](Self::closeness) /
    /// [`harmonic_centrality`](Self::harmonic_centrality), which summarize
    /// the rows of this matrix.
    ///
    /// Binds [`igraph_distances`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_distances).
    /// Time complexity (unweighted): O(n(|V| + |E|)) for n sources.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`] for
    /// invalid vertices, [`ErrorKind::NegativeCycle`]
    /// if negative weights form a negative cycle.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let p = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// let d = p.distances(.., .., None, NeighborMode::All).unwrap();
    /// assert_eq!(d.to_rows(), vec![vec![0.0, 1.0, 2.0], vec![1.0, 0.0, 1.0], vec![2.0, 1.0, 0.0]]);
    /// // Only from vertex 0 to vertices 1 and 2:
    /// let d = p.distances(0, vec![1, 2], None, NeighborMode::All).unwrap();
    /// assert_eq!(d.to_rows(), vec![vec![1.0, 2.0]]);
    /// ```
    pub fn distances<'a>(
        &self,
        from: impl Into<VertexSelector<'a>>,
        to: impl Into<VertexSelector<'a>>,
        weights: Option<&[f64]>,
        mode: NeighborMode,
    ) -> Result<Matrix> {
        self.distance_matrix(from, to, weights, |res, from, to, w| unsafe {
            igraph_distances(self, w, res, from, to, mode.into())
        })
    }

    /// Like [`distances`](Self::distances), but paths longer than `cutoff`
    /// are ignored (their length is reported as `INFINITY`).
    ///
    /// `cutoff = None` (or a negative value) means no cutoff. The search from
    /// each source stops at the cutoff, which may save much time. With
    /// `weights` this is the same as
    /// [`distances_dijkstra_cutoff`](Self::distances_dijkstra_cutoff).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] for a NaN
    /// cutoff or invalid weights,
    /// [`ErrorKind::InvalidVertexId`] for
    /// invalid vertices.
    ///
    /// Binds [`igraph_distances_cutoff`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_distances_cutoff).
    /// Time complexity: O(s |E| + |V|) for s sources (unweighted).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let p = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let d = p.distances_cutoff(0, .., None, NeighborMode::All, Some(2.0)).unwrap();
    /// assert_eq!(d.row(0), vec![0.0, 1.0, 2.0, f64::INFINITY]);
    /// ```
    pub fn distances_cutoff<'a>(
        &self,
        from: impl Into<VertexSelector<'a>>,
        to: impl Into<VertexSelector<'a>>,
        weights: Option<&[f64]>,
        mode: NeighborMode,
        cutoff: Option<f64>,
    ) -> Result<Matrix> {
        let cutoff = cutoff_value(cutoff)?;
        self.distance_matrix(from, to, weights, |res, from, to, w| unsafe {
            igraph_distances_cutoff(self, w, res, from, to, mode.into(), cutoff)
        })
    }

    /// Weighted shortest path lengths with Dijkstra's algorithm (binary heap),
    /// run independently from each source.
    ///
    /// Weights must be non-negative and not NaN; `None` falls back to the
    /// unweighted [`distances`](Self::distances).
    ///
    /// Binds [`igraph_distances_dijkstra`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_distances_dijkstra).
    /// Time complexity: O(s |E| log |V| + |V|) for s sources.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] for
    /// negative or NaN weights, or a weight vector of the wrong length.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let tri = Graph::from_edges(&[(0, 1), (1, 2), (0, 2)], 3, false).unwrap();
    /// let d = tri.distances_dijkstra(0, 2, Some(&[1.0, 1.0, 5.0]), NeighborMode::All).unwrap();
    /// assert_eq!(d[(0, 0)], 2.0);
    /// ```
    pub fn distances_dijkstra<'a>(
        &self,
        from: impl Into<VertexSelector<'a>>,
        to: impl Into<VertexSelector<'a>>,
        weights: Option<&[f64]>,
        mode: NeighborMode,
    ) -> Result<Matrix> {
        self.distance_matrix(from, to, weights, |res, from, to, w| unsafe {
            igraph_distances_dijkstra(self, res, from, to, w, mode.into())
        })
    }

    /// Like [`distances_dijkstra`](Self::distances_dijkstra), ignoring paths
    /// longer than `cutoff` (`None` or negative: no cutoff).
    ///
    /// Binds [`igraph_distances_dijkstra_cutoff`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_distances_dijkstra_cutoff).
    /// Time complexity: at most O(s |E| log |V| + |V|); the cutoff limits the
    /// explored region.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // Path 0 -2- 1 -2- 2 -2- 3: with cutoff 4 vertex 3 (at distance 6) is out of reach.
    /// let p = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let d = p
    ///     .distances_dijkstra_cutoff(0, .., Some(&[2.0; 3]), NeighborMode::All, Some(4.0))
    ///     .unwrap();
    /// assert_eq!(d.row(0), vec![0.0, 2.0, 4.0, f64::INFINITY]);
    /// ```
    pub fn distances_dijkstra_cutoff<'a>(
        &self,
        from: impl Into<VertexSelector<'a>>,
        to: impl Into<VertexSelector<'a>>,
        weights: Option<&[f64]>,
        mode: NeighborMode,
        cutoff: Option<f64>,
    ) -> Result<Matrix> {
        let cutoff = cutoff_value(cutoff)?;
        self.distance_matrix(from, to, weights, |res, from, to, w| unsafe {
            igraph_distances_dijkstra_cutoff(self, res, from, to, w, mode.into(), cutoff)
        })
    }

    /// Weighted shortest path lengths with the Bellman–Ford algorithm, which
    /// allows negative weights (but no negative cycles).
    ///
    /// If there are no negative weights,
    /// [`distances_dijkstra`](Self::distances_dijkstra) is faster.
    ///
    /// Binds [`igraph_distances_bellman_ford`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_distances_bellman_ford).
    /// Time complexity: O(s |E| |V|) for s sources.
    ///
    /// # Errors
    /// [`ErrorKind::NegativeCycle`] when a
    /// negative cycle is reachable.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 2)], 3, true).unwrap();
    /// let d = g.distances_bellman_ford(0, .., Some(&[2.0, -3.0, 1.0]), NeighborMode::Out).unwrap();
    /// assert_eq!(d.row(0), vec![0.0, 2.0, -1.0]);
    /// ```
    pub fn distances_bellman_ford<'a>(
        &self,
        from: impl Into<VertexSelector<'a>>,
        to: impl Into<VertexSelector<'a>>,
        weights: Option<&[f64]>,
        mode: NeighborMode,
    ) -> Result<Matrix> {
        self.distance_matrix(from, to, weights, |res, from, to, w| unsafe {
            igraph_distances_bellman_ford(self, res, from, to, w, mode.into())
        })
    }

    /// Weighted shortest path lengths with Johnson's algorithm: negative
    /// weights are allowed (directed graphs only, no negative cycles).
    ///
    /// A single Bellman–Ford run reweights the edges to non-negative values,
    /// then Dijkstra is run from each source; this beats Bellman–Ford when
    /// there are many sources. Without negative weights Dijkstra is used
    /// directly; with `None` the unweighted algorithm is used. Undirected
    /// graphs with any negative weight are rejected, even when no negative
    /// edge is reachable from the sources, and so is `NeighborMode::All`
    /// combined with negative weights: igraph treats an undirected negative
    /// edge as a negative cycle.
    ///
    /// Binds [`igraph_distances_johnson`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_distances_johnson).
    /// Time complexity: O(s |V| log |V| + |V| |E|).
    ///
    /// # Errors
    /// [`ErrorKind::NegativeCycle`] when a negative cycle is reachable, and
    /// also for negative weights in an undirected graph or together with
    /// `NeighborMode::All`; [`ErrorKind::InvalidValue`] for NaN weights or a
    /// weight vector of the wrong length.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // 0 → 1 → 2 with a negative "discount" edge 1 → 2.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 2)], 3, true).unwrap();
    /// let w = [2.0, -3.0, 1.0];
    /// let d = g.distances_johnson(.., .., Some(&w), NeighborMode::Out).unwrap();
    /// assert_eq!(d.row(0), vec![0.0, 2.0, -1.0]);
    /// // Following the edges backwards gives the transposed matrix.
    /// let d_in = g.distances_johnson(.., .., Some(&w), NeighborMode::In).unwrap();
    /// assert_eq!(d_in, d.transposed());
    /// ```
    pub fn distances_johnson<'a>(
        &self,
        from: impl Into<VertexSelector<'a>>,
        to: impl Into<VertexSelector<'a>>,
        weights: Option<&[f64]>,
        mode: NeighborMode,
    ) -> Result<Matrix> {
        self.distance_matrix(from, to, weights, |res, from, to, w| unsafe {
            igraph_distances_johnson(self, res, from, to, w, mode.into())
        })
    }

    /// All-pairs weighted shortest path lengths with the Floyd–Warshall
    /// algorithm (or one of its faster variants, see
    /// [`FloydWarshallAlgorithm`]).
    ///
    /// Negative weights are allowed but negative cycles are not. The full
    /// all-pairs matrix is always computed internally, `from` and `to` only
    /// subset it. Useful for very dense graphs.
    ///
    /// Binds [`igraph_distances_floyd_warshall`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_distances_floyd_warshall).
    /// Time complexity: O(|V|³ + |E|) for the original variant, expected
    /// O(|V|² log² |V|) for the tree variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::{paths::FloydWarshallAlgorithm, prelude::*};
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, true).unwrap();
    /// let d = g
    ///     .distances_floyd_warshall(.., .., None, NeighborMode::Out, FloydWarshallAlgorithm::Original)
    ///     .unwrap();
    /// assert_eq!(d.row(0), vec![0.0, 1.0, 2.0]);
    /// ```
    pub fn distances_floyd_warshall<'a>(
        &self,
        from: impl Into<VertexSelector<'a>>,
        to: impl Into<VertexSelector<'a>>,
        weights: Option<&[f64]>,
        mode: NeighborMode,
        method: FloydWarshallAlgorithm,
    ) -> Result<Matrix> {
        self.distance_matrix(from, to, weights, |res, from, to, w| unsafe {
            igraph_distances_floyd_warshall(self, res, from, to, w, mode.into(), method.into())
        })
    }
}

// ---------------------------------------------------------------------------
// Shortest paths themselves.

impl igraph_t {
    /// One shortest path from `from` to each of the `to` vertices.
    ///
    /// When several geodesics exist only one is returned (see
    /// [`get_all_shortest_paths`](Self::get_all_shortest_paths)). `to` may
    /// contain duplicates. With `weights` the weighted algorithms are used
    /// (Dijkstra, or Bellman–Ford for negative weights). The result also
    /// contains the shortest path tree (`parents` / `inbound_edges`).
    ///
    /// Binds [`igraph_get_shortest_paths`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_shortest_paths).
    /// Time complexity: O(|V| + |E|) unweighted.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let p = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let sp = p.get_shortest_paths(0, vec![3, 1], None, NeighborMode::All).unwrap();
    /// assert_eq!(sp.vertices, vec![vec![0, 1, 2, 3], vec![0, 1]]);
    /// assert_eq!(sp.edges, vec![vec![0, 1, 2], vec![0]]);
    /// assert_eq!(sp.parents, vec![-1, 0, 1, 2]);
    /// ```
    pub fn get_shortest_paths<'a>(
        &self,
        from: VertexId,
        to: impl Into<VertexSelector<'a>>,
        weights: Option<&[f64]>,
        mode: NeighborMode,
    ) -> Result<ShortestPaths> {
        self.shortest_paths_with(to, weights, |v, e, to, w, p, i| unsafe {
            igraph_get_shortest_paths(self, w, v, e, from, to, mode.into(), p, i)
        })
    }

    /// Weighted shortest paths from one vertex with Dijkstra's algorithm
    /// (non-negative weights); `None` falls back to BFS.
    ///
    /// The result has the same shape as for
    /// [`get_shortest_paths`](Self::get_shortest_paths); the search stops as
    /// soon as all the targets are reached, so `parents` may contain `-2`
    /// for vertices that were never reached.
    ///
    /// Binds [`igraph_get_shortest_paths_dijkstra`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_shortest_paths_dijkstra).
    /// Time complexity: O(|E| log |V| + |V|).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] for negative or NaN weights,
    /// [`ErrorKind::InvalidVertexId`] for invalid vertices.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // A square 0-1-2-3-0 whose edge (3, 0) is slow.
    /// let c4 = Graph::cycle_graph(4, false, false).unwrap();
    /// let w = [1.0, 1.0, 1.0, 5.0];
    /// let sp = c4.get_shortest_paths_dijkstra(0, .., Some(&w), NeighborMode::All).unwrap();
    /// assert_eq!(sp.vertices[3], vec![0, 1, 2, 3]);
    /// assert_eq!(sp.parents, vec![-1, 0, 1, 2]);
    /// assert_eq!(sp.inbound_edges, vec![-1, 0, 1, 2]);
    /// ```
    pub fn get_shortest_paths_dijkstra<'a>(
        &self,
        from: VertexId,
        to: impl Into<VertexSelector<'a>>,
        weights: Option<&[f64]>,
        mode: NeighborMode,
    ) -> Result<ShortestPaths> {
        self.shortest_paths_with(to, weights, |v, e, to, w, p, i| unsafe {
            igraph_get_shortest_paths_dijkstra(self, v, e, from, to, w, mode.into(), p, i)
        })
    }

    /// Weighted shortest paths from one vertex with the Bellman–Ford
    /// algorithm, allowing negative weights (but no negative cycles).
    ///
    /// Binds [`igraph_get_shortest_paths_bellman_ford`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_shortest_paths_bellman_ford).
    /// Time complexity: O(|E| |V|).
    ///
    /// # Errors
    /// [`ErrorKind::NegativeCycle`] when a
    /// negative cycle is found.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // The negative edge 1 → 2 makes the detour through 1 the shortest route to 2.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 2)], 3, true).unwrap();
    /// let sp = g
    ///     .get_shortest_paths_bellman_ford(0, .., Some(&[2.0, -3.0, 1.0]), NeighborMode::Out)
    ///     .unwrap();
    /// assert_eq!(sp.vertices, vec![vec![0], vec![0, 1], vec![0, 1, 2]]);
    /// ```
    pub fn get_shortest_paths_bellman_ford<'a>(
        &self,
        from: VertexId,
        to: impl Into<VertexSelector<'a>>,
        weights: Option<&[f64]>,
        mode: NeighborMode,
    ) -> Result<ShortestPaths> {
        self.shortest_paths_with(to, weights, |v, e, to, w, p, i| unsafe {
            igraph_get_shortest_paths_bellman_ford(self, v, e, from, to, w, mode.into(), p, i)
        })
    }

    /// A single shortest path between two vertices (an arbitrary one if there
    /// are several). An empty [`GraphPath`] means `to` is unreachable (the
    /// BFS and Dijkstra searches then also emit an igraph warning).
    ///
    /// Binds [`igraph_get_shortest_path`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_shortest_path).
    /// Time complexity: O(|V| + |E|) unweighted.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let c = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 5, false).unwrap();
    /// let p = c.get_shortest_path(0, 3, None, NeighborMode::All).unwrap();
    /// assert_eq!(p.vertices, vec![0, 4, 3]);
    /// assert_eq!(p.len(), 2);
    /// ```
    pub fn get_shortest_path(
        &self,
        from: VertexId,
        to: VertexId,
        weights: Option<&[f64]>,
        mode: NeighborMode,
    ) -> Result<GraphPath> {
        self.single_path_with(weights, |v, e, w| unsafe {
            igraph_get_shortest_path(self, w, v, e, from, to, mode.into())
        })
    }

    /// A single weighted shortest path with Dijkstra's algorithm
    /// (non-negative weights; `None` falls back to BFS). An empty
    /// [`GraphPath`] means that `to` is unreachable.
    ///
    /// Binds [`igraph_get_shortest_path_dijkstra`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_shortest_path_dijkstra).
    /// Time complexity: O(|E| log |V| + |V|).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let c4 = Graph::cycle_graph(4, false, false).unwrap();
    /// let w = [1.0, 1.0, 1.0, 5.0];
    /// let p = c4.get_shortest_path_dijkstra(0, 3, Some(&w), NeighborMode::All).unwrap();
    /// assert_eq!(p.vertices, vec![0, 1, 2, 3]);
    /// assert_eq!(p.weight(&w), 3.0);
    /// ```
    pub fn get_shortest_path_dijkstra(
        &self,
        from: VertexId,
        to: VertexId,
        weights: Option<&[f64]>,
        mode: NeighborMode,
    ) -> Result<GraphPath> {
        self.single_path_with(weights, |v, e, w| unsafe {
            igraph_get_shortest_path_dijkstra(self, v, e, from, to, w, mode.into())
        })
    }

    /// A single weighted shortest path with the Bellman–Ford algorithm
    /// (negative weights allowed, negative cycles are not).
    ///
    /// Binds [`igraph_get_shortest_path_bellman_ford`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_shortest_path_bellman_ford).
    /// Time complexity: O(|E| |V|).
    ///
    /// # Errors
    /// [`ErrorKind::NegativeCycle`] when a negative cycle is found.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 2)], 3, true).unwrap();
    /// let w = [2.0, -3.0, 1.0];
    /// let p = g.get_shortest_path_bellman_ford(0, 2, Some(&w), NeighborMode::Out).unwrap();
    /// assert_eq!(p.edges, vec![0, 1]);
    /// assert_eq!(p.weight(&w), -1.0);
    /// // Closing a negative cycle makes shortest paths meaningless.
    /// let mut cyclic = g.clone();
    /// cyclic.add_edge(2, 0).unwrap();
    /// let err = cyclic
    ///     .get_shortest_path_bellman_ford(0, 2, Some(&[2.0, -3.0, 1.0, 0.5]), NeighborMode::Out)
    ///     .unwrap_err();
    /// assert_eq!(err.kind(), ErrorKind::NegativeCycle);
    /// ```
    pub fn get_shortest_path_bellman_ford(
        &self,
        from: VertexId,
        to: VertexId,
        weights: Option<&[f64]>,
        mode: NeighborMode,
    ) -> Result<GraphPath> {
        self.single_path_with(weights, |v, e, w| unsafe {
            igraph_get_shortest_path_bellman_ford(self, v, e, from, to, w, mode.into())
        })
    }

    /// A single shortest path with the A* algorithm, guided by a heuristic.
    ///
    /// `heuristic(v, to)` must estimate the distance from the candidate
    /// vertex `v` to the target `to`; smaller values make `v` a better
    /// candidate. The result is a true shortest path if the heuristic is
    /// *admissible*, i.e. never overestimates the distance. A heuristic
    /// returning always `0.0` turns A* into Dijkstra's algorithm. Weights must
    /// be non-negative. The heuristic should return finite, non-negative
    /// values (NaN estimates break the priority queue ordering and give
    /// meaningless paths). A panic in the heuristic aborts the search and is
    /// propagated to the caller. An empty [`GraphPath`] means that `to` is
    /// unreachable.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`] for
    /// invalid `from`/`to`,
    /// [`ErrorKind::InvalidValue`] for
    /// negative or NaN weights.
    ///
    /// Binds [`igraph_get_shortest_path_astar`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_shortest_path_astar).
    /// Time complexity: worst case O(|E| log |V| + |V|); better heuristics
    /// mean faster searches.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // A 10 x 10 grid; vertex id = x + 10 y. Manhattan distance is admissible.
    /// let n = 10;
    /// let grid = Graph::square_lattice(&[10, 10], 1, false, false, None).unwrap();
    /// let manhattan = |a: i64, b: i64| ((a % n - b % n).abs() + (a / n - b / n).abs()) as f64;
    /// let p = grid.get_shortest_path_astar(0, 99, None, NeighborMode::All, manhattan).unwrap();
    /// assert_eq!(p.len(), 18);
    /// ```
    pub fn get_shortest_path_astar<F>(
        &self,
        from: VertexId,
        to: VertexId,
        weights: Option<&[f64]>,
        mode: NeighborMode,
        mut heuristic: F,
    ) -> Result<GraphPath>
    where
        F: FnMut(VertexId, VertexId) -> f64,
    {
        unsafe extern "C" fn trampoline<F: FnMut(VertexId, VertexId) -> f64>(
            result: *mut igraph_real_t,
            from: igraph_int_t,
            to: igraph_int_t,
            extra: *mut c_void,
        ) -> igraph_error_t {
            // The heuristic runs in a fresh level of igraph's "finally"
            // stack: otherwise a failing igraph call made by the closure
            // would free the temporaries of the running A* search (a
            // use-after-free in C).
            // SAFETY: bookkeeping on igraph's thread-local finally stack; the
            // matching EXIT runs below, as `catch_panic` never unwinds.
            unsafe { IGRAPH_FINALLY_ENTER() };
            let code = catch_panic(|| {
                let f = unsafe { &mut *(extra as *mut F) };
                let value = f(from, to);
                unsafe { *result = value };
                igraph_error_type_t_IGRAPH_SUCCESS
            });
            // SAFETY: closes the level opened above; any failed nested call
            // has already freed its own objects of that level.
            unsafe { IGRAPH_FINALLY_EXIT() };
            code
        }
        let extra = &mut heuristic as *mut F as *mut c_void;
        self.single_path_with(weights, |v, e, w| unsafe {
            igraph_get_shortest_path_astar(
                self,
                v,
                e,
                from,
                to,
                w,
                mode.into(),
                Some(trampoline::<F>),
                extra,
            )
        })
    }

    /// *All* the shortest paths (geodesics) from `from` to the `to` vertices.
    ///
    /// Paths are grouped by target in increasing vertex id order; unreachable
    /// targets contribute nothing. Multi-edges are considered separately, so
    /// multigraphs may yield very many paths. `nrgeo[v]` counts the
    /// geodesics from `from` to `v`. With `weights`, Dijkstra's algorithm is
    /// used (see [`get_all_shortest_paths_dijkstra`](Self::get_all_shortest_paths_dijkstra)).
    /// Counting geodesics through each vertex is what
    /// [`betweenness`](Self::betweenness) does, for all sources at once.
    ///
    /// Binds [`igraph_get_all_shortest_paths`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_all_shortest_paths).
    /// Time complexity: O(|V| + |E|) for most graphs, O(|V|²) worst case.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // A 4-cycle has two geodesics between opposite corners.
    /// let c4 = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], 4, false).unwrap();
    /// let all = c4.get_all_shortest_paths(0, 2, None, NeighborMode::All).unwrap();
    /// let mut paths = all.vertices.clone();
    /// paths.sort();
    /// assert_eq!(paths, vec![vec![0, 1, 2], vec![0, 3, 2]]);
    /// assert_eq!(all.nrgeo[2], 2);
    /// ```
    pub fn get_all_shortest_paths<'a>(
        &self,
        from: VertexId,
        to: impl Into<VertexSelector<'a>>,
        weights: Option<&[f64]>,
        mode: NeighborMode,
    ) -> Result<AllShortestPaths> {
        self.all_shortest_paths_with(to, weights, |v, e, n, to, w| unsafe {
            igraph_get_all_shortest_paths(self, w, v, e, n, from, to, mode.into())
        })
    }

    /// All the weighted shortest paths from one vertex, with Dijkstra's
    /// algorithm (non-negative weights; `None` falls back to the unweighted
    /// [`get_all_shortest_paths`](Self::get_all_shortest_paths)).
    ///
    /// Binds [`igraph_get_all_shortest_paths_dijkstra`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_all_shortest_paths_dijkstra).
    /// Time complexity: O(|E| log |V| + |V|).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let c4 = Graph::cycle_graph(4, false, false).unwrap();
    /// // Equal weights: two geodesics between opposite corners ...
    /// let all = c4.get_all_shortest_paths_dijkstra(0, 2, Some(&[1.0; 4]), NeighborMode::All).unwrap();
    /// assert_eq!(all.nrgeo[2], 2);
    /// // ... a slow edge (3, 0) leaves only one.
    /// let w = [1.0, 1.0, 1.0, 5.0];
    /// let all = c4.get_all_shortest_paths_dijkstra(0, 2, Some(&w), NeighborMode::All).unwrap();
    /// assert_eq!(all.vertices, vec![vec![0, 1, 2]]);
    /// assert_eq!(all.edges, vec![vec![0, 1]]);
    /// ```
    pub fn get_all_shortest_paths_dijkstra<'a>(
        &self,
        from: VertexId,
        to: impl Into<VertexSelector<'a>>,
        weights: Option<&[f64]>,
        mode: NeighborMode,
    ) -> Result<AllShortestPaths> {
        self.all_shortest_paths_with(to, weights, |v, e, n, to, w| unsafe {
            igraph_get_all_shortest_paths_dijkstra(self, v, e, n, from, to, w, mode.into())
        })
    }

    /// The `k` shortest paths between two vertices, in order of increasing
    /// length (Yen's algorithm). Fewer than `k` paths are returned when fewer
    /// exist. Infinite weights are treated as missing edges.
    ///
    /// Binds [`igraph_get_k_shortest_paths`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_k_shortest_paths).
    /// Time complexity: O(k |V| (|V| log |V| + |E|)).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // In a 5-cycle there are exactly two paths between any two vertices.
    /// let c5 = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 5, false).unwrap();
    /// let paths = c5.get_k_shortest_paths(0, 2, 10, None, NeighborMode::All).unwrap();
    /// assert_eq!(paths.len(), 2);
    /// assert_eq!(paths[0].vertices, vec![0, 1, 2]);
    /// assert_eq!(paths[1].vertices, vec![0, 4, 3, 2]);
    /// ```
    pub fn get_k_shortest_paths(
        &self,
        from: VertexId,
        to: VertexId,
        k: usize,
        weights: Option<&[f64]>,
        mode: NeighborMode,
    ) -> Result<Vec<GraphPath>> {
        self.paths_check_weights(weights)?;
        let w = weights_view(weights);
        let mut vpaths = VectorIntList::new();
        let mut epaths = VectorIntList::new();
        igraph_call!(igraph_get_k_shortest_paths(
            self,
            weights_ptr(&w),
            &mut vpaths,
            &mut epaths,
            saturating_int(k),
            from,
            to,
            mode.into()
        ))?;
        Ok(vpaths
            .to_vecs()
            .into_iter()
            .zip(epaths.to_vecs())
            .map(|(vertices, edges)| GraphPath { vertices, edges })
            .collect())
    }

    /// All the *simple* paths (no repeated vertex) starting at `from` and
    /// ending at one of the `to` vertices, as vertex lists.
    ///
    /// Multi-edges are ignored. There may be exponentially many simple paths:
    /// use `options` to bound their length or number. Paths are listed in
    /// the order they are found.
    ///
    /// Binds [`igraph_get_all_simple_paths`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_all_simple_paths).
    /// Time complexity: O(n!) in the worst case.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::{paths::SimplePathsOptions, prelude::*};
    /// let k4 = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], 4, false).unwrap();
    /// // Paths from 0 to 3: the edge, two of length 2, two of length 3.
    /// let all = k4.get_all_simple_paths(0, 3, NeighborMode::All, SimplePathsOptions::default()).unwrap();
    /// assert_eq!(all.len(), 5);
    /// let short = SimplePathsOptions::default().with_max_len(2);
    /// assert_eq!(k4.get_all_simple_paths(0, 3, NeighborMode::All, short).unwrap().len(), 3);
    /// ```
    pub fn get_all_simple_paths<'a>(
        &self,
        from: VertexId,
        to: impl Into<VertexSelector<'a>>,
        mode: NeighborMode,
        options: SimplePathsOptions,
    ) -> Result<Vec<Vec<VertexId>>> {
        let to = to.into().to_raw()?;
        let mut res = VectorIntList::new();
        self.with_fresh_multi_cache(|| {
            igraph_call!(igraph_get_all_simple_paths(
                self,
                &mut res,
                from,
                to.get(),
                mode.into(),
                limit(options.min_len),
                limit(options.max_len),
                limit(options.max_results)
            ))
        })?;
        Ok(res.to_vecs())
    }
}

// ---------------------------------------------------------------------------
// Widest paths.

impl igraph_t {
    /// Widest (maximum bottleneck) paths from one vertex to the `to` vertices.
    ///
    /// The width of a path is the smallest weight among its edges, and a
    /// widest path maximizes it. `weights` are *widths* and are mandatory;
    /// they may be negative but not NaN, edges of width `-INFINITY` are
    /// ignored. The result has the same shape as for
    /// [`get_shortest_paths`](Self::get_shortest_paths). A widest path
    /// tells how much can be pushed along a *single* route; see
    /// [`maxflow`](Self::maxflow) for the total capacity over all routes.
    ///
    /// Binds [`igraph_get_widest_paths`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_widest_paths).
    /// Time complexity: O(|E| log |E| + |V|).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // Two routes from 0 to 3: a thin direct pipe and a wide detour.
    /// let g = Graph::from_edges(&[(0, 3), (0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let widths = [1.0, 10.0, 8.0, 9.0];
    /// let wp = g.get_widest_paths(0, .., &widths, NeighborMode::All).unwrap();
    /// assert_eq!(wp.vertices, vec![vec![0], vec![0, 1], vec![0, 1, 2], vec![0, 1, 2, 3]]);
    /// assert_eq!(wp.parents, vec![-1, 0, 1, 2]);
    /// ```
    pub fn get_widest_paths<'a>(
        &self,
        from: VertexId,
        to: impl Into<VertexSelector<'a>>,
        weights: &[f64],
        mode: NeighborMode,
    ) -> Result<ShortestPaths> {
        self.shortest_paths_with(to, Some(weights), |v, e, to, w, p, i| unsafe {
            igraph_get_widest_paths(self, v, e, from, to, w, mode.into(), p, i)
        })
    }

    /// A single widest (maximum bottleneck) path between two vertices, see
    /// [`get_widest_paths`](Self::get_widest_paths).
    ///
    /// Binds [`igraph_get_widest_path`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_widest_path).
    /// Time complexity: O(|E| log |E| + |V|).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // Two routes from 0 to 3: a thin direct pipe and a wide detour.
    /// let g = Graph::from_edges(&[(0, 3), (0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let widths = [1.0, 10.0, 8.0, 9.0];
    /// let p = g.get_widest_path(0, 3, &widths, NeighborMode::All).unwrap();
    /// assert_eq!(p.vertices, vec![0, 1, 2, 3]);
    /// ```
    pub fn get_widest_path(
        &self,
        from: VertexId,
        to: VertexId,
        weights: &[f64],
        mode: NeighborMode,
    ) -> Result<GraphPath> {
        self.single_path_with(Some(weights), |v, e, w| unsafe {
            igraph_get_widest_path(self, v, e, from, to, w, mode.into())
        })
    }

    /// Widths of the widest paths between vertices, with a modified
    /// Floyd–Warshall algorithm (good for dense graphs).
    ///
    /// Unreachable pairs have width `-INFINITY`, a vertex has width
    /// `INFINITY` to itself. The full matrix is always computed internally.
    /// The result equals that of
    /// [`widest_path_widths_dijkstra`](Self::widest_path_widths_dijkstra).
    ///
    /// Binds [`igraph_widest_path_widths_floyd_warshall`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_widest_path_widths_floyd_warshall).
    /// Time complexity: O(|V|³).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // Directed: 0 → 1 → 2 is wide, the direct edge 0 → 2 is thin, 2 reaches nothing.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 2)], 3, true).unwrap();
    /// let w = g
    ///     .widest_path_widths_floyd_warshall(.., .., &[5.0, 4.0, 1.0], NeighborMode::Out)
    ///     .unwrap();
    /// let inf = f64::INFINITY;
    /// assert_eq!(w.to_rows(), vec![vec![inf, 5.0, 4.0], vec![-inf, inf, 4.0], vec![-inf, -inf, inf]]);
    /// ```
    pub fn widest_path_widths_floyd_warshall<'a>(
        &self,
        from: impl Into<VertexSelector<'a>>,
        to: impl Into<VertexSelector<'a>>,
        weights: &[f64],
        mode: NeighborMode,
    ) -> Result<Matrix> {
        self.distance_matrix(from, to, Some(weights), |res, from, to, w| unsafe {
            igraph_widest_path_widths_floyd_warshall(self, res, from, to, w, mode.into())
        })
    }

    /// Widths of the widest paths between vertices, with a modified Dijkstra
    /// algorithm run from each source (good for sparse graphs). `to` must not
    /// contain duplicates.
    ///
    /// Unreachable pairs have width `-INFINITY`, a vertex has width
    /// `INFINITY` to itself.
    ///
    /// Binds [`igraph_widest_path_widths_dijkstra`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_widest_path_widths_dijkstra).
    /// Time complexity: O(s (|E| log |E| + |V|)) for s sources.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 3), (0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let w = g.widest_path_widths_dijkstra(0, .., &[1.0, 10.0, 8.0, 9.0], NeighborMode::All).unwrap();
    /// assert_eq!(w.row(0), vec![f64::INFINITY, 10.0, 8.0, 8.0]);
    /// ```
    pub fn widest_path_widths_dijkstra<'a>(
        &self,
        from: impl Into<VertexSelector<'a>>,
        to: impl Into<VertexSelector<'a>>,
        weights: &[f64],
        mode: NeighborMode,
    ) -> Result<Matrix> {
        self.distance_matrix(from, to, Some(weights), |res, from, to, w| unsafe {
            igraph_widest_path_widths_dijkstra(self, res, from, to, w, mode.into())
        })
    }
}

// ---------------------------------------------------------------------------
// Random walks, spanners, Voronoi partitions and path conversions.

impl igraph_t {
    /// A random walk of (at most) `steps` steps starting at `start`.
    ///
    /// The walk follows edges according to `mode` (in directed graphs);
    /// multi-edges and loops are respected. With `weights` the next edge is
    /// chosen with probability proportional to its weight (non-negative, at
    /// least one positive weight among the out-edges of each vertex). When
    /// the walk reaches a vertex without outgoing edges, `stuck` decides
    /// whether to return the shorter walk ([`RandomWalkStuck::Return`]) or to
    /// fail ([`RandomWalkStuck::Error`]). The returned path has `steps + 1`
    /// vertices and `steps` edges unless it got stuck. Draws from the calling
    /// thread's default random number generator: seed it with
    /// [`rng::seed`](crate::rng::seed) for reproducible walks, or run the
    /// walk with a private generator through
    /// [`Rng::scoped`](crate::rng::Rng::scoped).
    ///
    /// Binds [`igraph_random_walk`](https://igraph.org/c/html/latest/igraph-Visitors.html#igraph_random_walk).
    /// Time complexity: O(l + d) unweighted, O(l log k + d) weighted, where l
    /// is the walk length, d the total degree of the visited vertices and k
    /// the average degree.
    ///
    /// # Errors
    /// - [`ErrorKind::RandomWalkStuck`] if
    ///   the walk gets stuck and `stuck` is [`RandomWalkStuck::Error`] (the
    ///   partial walk is discarded; use [`RandomWalkStuck::Return`] to keep it).
    /// - [`ErrorKind::InvalidValue`] for an
    ///   invalid `start` vertex (igraph 1.0.0 and 1.0.1 report
    ///   `IGRAPH_EINVAL` here, not `IGRAPH_EINVVID`), invalid weights
    ///   (negative, NaN or wrong length), or `steps >= i64::MAX`. The Rust side
    ///   also rejects infinite weights, and weights whose total over the edges
    ///   a walk can leave some vertex through exceeds `f64::MAX / 2` (loops
    ///   count twice in `All` mode): with such weights igraph 1.0.1 never
    ///   terminates. The check covers every vertex, not only the visited ones.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let c = Graph::cycle_graph(4, false, false).unwrap();
    /// // Seeding the (per-thread) default generator makes the walk reproducible.
    /// rng::seed(42).unwrap();
    /// let walk = c.random_walk(0, 20, None, NeighborMode::All, RandomWalkStuck::Error).unwrap();
    /// rng::seed(42).unwrap();
    /// let again = c.random_walk(0, 20, None, NeighborMode::All, RandomWalkStuck::Error).unwrap();
    /// assert_eq!(walk, again);
    /// assert_eq!(walk.vertices.len(), 21);
    /// assert_eq!(walk.edges.len(), 20);
    /// // Consecutive vertices are adjacent.
    /// for pair in walk.vertices.windows(2) {
    ///     assert!(c.get_eid(pair[0], pair[1], false).unwrap().is_some());
    /// }
    /// // A private generator leaves the default one untouched.
    /// let mut private = Rng::new(RngType::Pcg64, 7).unwrap();
    /// let w = private
    ///     .scoped(|| c.random_walk(0, 5, None, NeighborMode::All, RandomWalkStuck::Error))
    ///     .unwrap();
    /// assert_eq!(w.len(), 5);
    /// ```
    pub fn random_walk(
        &self,
        start: VertexId,
        steps: usize,
        weights: Option<&[f64]>,
        mode: NeighborMode,
        stuck: RandomWalkStuck,
    ) -> Result<GraphPath> {
        // igraph allocates `steps + 1` vertices: `steps` must stay below
        // `igraph_int_t::MAX` to avoid a signed overflow in C.
        let steps = match igraph_int_t::try_from(steps) {
            Ok(s) if s < igraph_int_t::MAX => s,
            _ => {
                return Err(Error::invalid(format!(
                    "too many random walk steps: {steps}"
                )));
            }
        };
        // igraph 1.0.1 rejects only negative and NaN weights. An infinite
        // weight, or finite weights whose sum at a vertex overflows, make the
        // cumulative distribution of that vertex infinite, and sampling from
        // [0, inf) never terminates in C.
        if let Some(ws) = weights {
            self.random_walk_check_weights(ws, mode)?;
        }
        self.single_path_with(weights, |v, e, w| unsafe {
            igraph_random_walk(self, w, v, e, start, mode.into(), steps, stuck.into())
        })
    }

    /// Rust-side weight check of [`random_walk`](Self::random_walk): every
    /// weight must be finite, and the total weight of the edges a walk can
    /// leave each vertex through (loops count twice in `All` mode, as in
    /// igraph's incidence lists) must be at most `f64::MAX / 2`. The factor
    /// 2 leaves room for the rounding of igraph's own summation order.
    /// Weights of the wrong length are left to igraph, which reports them.
    fn random_walk_check_weights(&self, ws: &[f64], mode: NeighborMode) -> Result<()> {
        if ws.iter().any(|x| !x.is_finite()) {
            return Err(Error::invalid("random walk weights must be finite"));
        }
        if ws.len() != self.ecount() {
            return Ok(());
        }
        let (out, inn) = match mode {
            _ if !self.is_directed() => (true, true),
            NeighborMode::Out => (true, false),
            NeighborMode::In => (false, true),
            _ => (true, true),
        };
        let mut total = vec![0.0_f64; self.vcount()];
        for (&(from, to), &w) in self.edges(EdgeSelector::All)?.iter().zip(ws) {
            if out {
                total[from as usize] += w;
            }
            if inn {
                total[to as usize] += w;
            }
        }
        if total.iter().any(|&t| t > f64::MAX / 2.0) {
            return Err(Error::invalid(
                "the total random walk weight at some vertex overflows",
            ));
        }
        Ok(())
    }

    /// The edge ids of a *spanner* of the graph with stretch factor `stretch`.
    ///
    /// A t-spanner is a subgraph `H` with the same vertices, in which the
    /// distance between any two vertices is at most `t` times their distance
    /// in the original graph. The randomized Baswana–Sen algorithm is used
    /// (drawing from the thread's default random number generator); edge
    /// directions are ignored. Use
    /// [`subgraph_from_edges`](Self::subgraph_from_edges) (with
    /// `delete_vertices = false`) to extract the spanner as a graph. See also
    /// [`minimum_spanning_tree`](Self::minimum_spanning_tree), the sparsest
    /// connected subgraph, which offers no stretch guarantee.
    ///
    /// Binds [`igraph_spanner`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_spanner).
    /// Expected time complexity: O(k |E|), with k = (t + 1) / 2.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `stretch` is below one or not finite
    /// (checked on the Rust side for NaN and infinity: igraph 1.0.1 accepts a
    /// NaN stretch and never terminates for an infinite one), or if the
    /// weights are negative, NaN or of the wrong length.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let k8 = Graph::full(8, false, false).unwrap();
    /// rng::seed(1).unwrap();
    /// let kept = k8.spanner(3.0, None).unwrap();
    /// assert!(kept.len() <= k8.ecount());
    /// // Distances in the spanner are at most 3 times the original ones (all 1 here).
    /// let h = k8.subgraph_from_edges(kept, false).unwrap();
    /// assert_eq!(h.vcount(), 8);
    /// let d = h.distances(.., .., None, NeighborMode::All).unwrap();
    /// assert!(d.as_slice().iter().all(|&x| x <= 3.0));
    /// ```
    pub fn spanner(&self, stretch: f64, weights: Option<&[f64]>) -> Result<Vec<EdgeId>> {
        // igraph runs `(stretch + 1) / 2 - 1` clustering rounds: an infinite
        // stretch never terminates and a NaN one is silently accepted.
        if !stretch.is_finite() {
            return Err(Error::invalid(format!(
                "the stretch factor must be finite, got {stretch}"
            )));
        }
        self.paths_check_weights(weights)?;
        let w = weights_view(weights);
        let mut res = VectorInt::new();
        igraph_call!(igraph_spanner(self, &mut res, stretch, weights_ptr(&w)))?;
        Ok(res.into())
    }

    /// Voronoi partitioning: assigns each vertex to its closest generator.
    ///
    /// Distances are computed from the generators (`mode = Out`), towards
    /// them (`In`) or ignoring directions (`All`); BFS is used for unweighted
    /// graphs, Dijkstra for (non-negative) weights. `tiebreaker` decides
    /// which generator wins at equal distance; note that
    /// [`VoronoiTiebreaker::Random`] may produce non-contiguous cells
    /// (and draws from the thread's default random number generator). See
    /// also [`community_voronoi`](Self::community_voronoi), which picks the
    /// generators automatically to detect communities.
    ///
    /// Binds [`igraph_voronoi`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_voronoi).
    /// Time complexity: O(log |S| |E| log |V| + |V|) weighted, O(log |S| |E| +
    /// |V|) unweighted, for |S| generators.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // Path 0-1-2-3-4-5 with generators at both ends.
    /// let p = Graph::path_graph(6, false, false).unwrap();
    /// let v = p.voronoi(&[0, 5], None, NeighborMode::All, VoronoiTiebreaker::First).unwrap();
    /// assert_eq!(v.membership, vec![0, 0, 0, 1, 1, 1]);
    /// assert_eq!(v.distances, vec![0.0, 1.0, 2.0, 2.0, 1.0, 0.0]);
    /// ```
    pub fn voronoi(
        &self,
        generators: &[VertexId],
        weights: Option<&[f64]>,
        mode: NeighborMode,
        tiebreaker: VoronoiTiebreaker,
    ) -> Result<VoronoiPartition> {
        self.paths_check_weights(weights)?;
        let gens = VectorInt::view(generators);
        let w = weights_view(weights);
        let mut membership = VectorInt::new();
        let mut distances = Vector::new();
        igraph_call!(igraph_voronoi(
            self,
            &mut membership,
            &mut distances,
            gens.as_ptr(),
            weights_ptr(&w),
            mode.into(),
            tiebreaker.into()
        ))?;
        Ok(VoronoiPartition {
            membership: membership.into(),
            distances: distances.into(),
        })
    }

    /// Converts a walk given by its edge ids into the sequence of vertices it
    /// visits (one more than the number of edges).
    ///
    /// `start` is the first vertex; with `None` it is inferred from the walk
    /// (which then must have at least one edge). Vertices may repeat, so any
    /// walk, cycle or path is accepted, e.g. the edge sequences returned by
    /// [`find_cycle`](Self::find_cycle) or [`eulerian_path`](Self::eulerian_path).
    /// Edge ids are validated on the Rust side (igraph 1.0.0 and 1.0.1 do not
    /// check them).
    ///
    /// Binds [`igraph_vertex_path_from_edge_path`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_vertex_path_from_edge_path).
    /// Time complexity: O(n) for a walk of length n.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if the
    /// edges do not form a continuous walk from `start`, or if `start` is
    /// `None` and the walk is empty;
    /// [`ErrorKind::InvalidEdgeId`] for an
    /// edge id out of range;
    /// [`ErrorKind::InvalidVertexId`] for
    /// an invalid `start` vertex.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, true).unwrap();
    /// let v = g.vertex_path_from_edge_path(Some(0), &[0, 1, 2], NeighborMode::Out).unwrap();
    /// assert_eq!(v, vec![0, 1, 2, 3]);
    /// // Walking backwards along the directed edges.
    /// let v = g.vertex_path_from_edge_path(Some(3), &[2, 1], NeighborMode::In).unwrap();
    /// assert_eq!(v, vec![3, 2, 1]);
    /// ```
    pub fn vertex_path_from_edge_path(
        &self,
        start: Option<VertexId>,
        edge_path: &[EdgeId],
        mode: NeighborMode,
    ) -> Result<Vec<VertexId>> {
        // igraph 1.0.0 and 1.0.1 do not validate the edge ids here and would
        // read out of bounds: check them on the Rust side.
        let ecount = self.ecount() as EdgeId;
        if let Some(&e) = edge_path.iter().find(|&&e| !(0..ecount).contains(&e)) {
            return Err(Error::new(
                ErrorKind::InvalidEdgeId,
                format!("invalid edge id {e} in edge path"),
            ));
        }
        let start = opt_vertex_arg(start, self.vcount())?;
        let edges = VectorInt::view(edge_path);
        let mut res = VectorInt::new();
        igraph_call!(igraph_vertex_path_from_edge_path(
            self,
            start,
            edges.as_ptr(),
            &mut res,
            mode.into()
        ))?;
        Ok(res.into())
    }
}

/// Turns a path given as a vertex sequence into the list of its consecutive
/// vertex pairs, e.g. `[a, b, c]` into `[(a, b), (b, c)]`.
///
/// The result can be passed to [`Graph::get_eids`] to find the edges along
/// the path. Paths with fewer than two vertices give an empty list.
///
/// Binds [`igraph_expand_path_to_pairs`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_expand_path_to_pairs).
///
/// # Examples
///
/// ```
/// use igraph::{paths::expand_path_to_pairs, prelude::*};
/// assert_eq!(expand_path_to_pairs(&[0, 4, 2]).unwrap(), vec![(0, 4), (4, 2)]);
/// assert!(expand_path_to_pairs(&[7]).unwrap().is_empty());
///
/// let g = Graph::from_edges(&[(0, 4), (4, 2)], 5, false).unwrap();
/// let pairs = expand_path_to_pairs(&[0, 4, 2]).unwrap();
/// // Graph::get_eids (core) turns the pairs into edge ids.
/// assert_eq!(g.get_eids(&pairs, false).unwrap(), vec![0, 1]);
/// ```
pub fn expand_path_to_pairs(path: &[VertexId]) -> Result<Vec<(VertexId, VertexId)>> {
    let mut v = VectorInt::from_slice(path);
    igraph_call!(igraph_expand_path_to_pairs(&mut v))?;
    Ok(v.as_chunks::<2>().0.iter().map(|&[a, b]| (a, b)).collect())
}

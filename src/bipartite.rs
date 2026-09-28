//! Bipartite (two-mode) graphs and matchings (`igraph_bipartite.h`, `igraph_matching.h`).
//!
//! A graph is *bipartite* if its vertices can be split into two classes so
//! that every edge connects vertices of different classes: actors and
//! movies, authors and papers, workers and jobs. igraph has no dedicated
//! bipartite graph type: an ordinary [`Graph`] is paired with a *types*
//! vector, one `bool` per vertex, telling which class each vertex belongs
//! to. In this crate the types are plain `&[bool]` inputs and `Vec<bool>`
//! outputs; by igraph's convention vertices of type `false` form the *first*
//! (or "bottom") class and vertices of type `true` the *second* ("top") one.
//!
//! Functions that *create* bipartite graphs return a [`BipartiteGraph`],
//! which bundles the graph with its types and offers shortcuts to the
//! analysis methods below.
//!
//! # Example
//!
//! ```
//! use igraph::{bipartite::BipartiteGraph, prelude::*};
//!
//! // Three workers (vertices 0..3, type false) and three jobs (3..6, type true).
//! let types = vec![false, false, false, true, true, true];
//! let edges = [(0, 3), (0, 4), (1, 3), (2, 4), (2, 5)];
//! let staff = BipartiteGraph::new(types, &edges, false).unwrap();
//!
//! // Everyone can get a job: the maximum matching is perfect.
//! let m = staff.maximum_matching(None).unwrap();
//! assert_eq!(m.size, 3);
//! assert_eq!(m.mate(1), Some(3)); // worker 1 can only take job 3
//! assert!(staff.graph.is_matching(Some(&staff.types), &m.matching).unwrap());
//!
//! // Workers sharing a job skill are connected in the projection.
//! let p = staff.projection().unwrap();
//! assert_eq!(p.proj1.edge_list(), vec![(0, 1), (0, 2)]);
//! ```
//!
//! # Provided functionality
//!
//! | Rust | C function | What it does |
//! |------|------------|--------------|
//! | [`Graph::is_bipartite`], [`Graph::bipartite_types`], [`BipartiteGraph::from_graph`] | `igraph_is_bipartite` | test bipartiteness, find a 2-coloring |
//! | [`Graph::create_bipartite`], [`BipartiteGraph::new`] | `igraph_create_bipartite` | build from types + edges, checking bipartiteness |
//! | [`Graph::full_bipartite`] | `igraph_full_bipartite` | complete bipartite graph *K(n1, n2)* |
//! | [`Graph::biadjacency`] | `igraph_biadjacency` | graph from a bipartite adjacency matrix |
//! | [`Graph::weighted_biadjacency`] | `igraph_weighted_biadjacency` | weighted graph from a bipartite adjacency matrix |
//! | [`Graph::get_biadjacency`], [`BipartiteGraph::biadjacency`] | `igraph_get_biadjacency` | bipartite adjacency matrix of a graph |
//! | [`Graph::bipartite_projection_size`] | `igraph_bipartite_projection_size` | sizes of the two projections |
//! | [`Graph::bipartite_projection`], [`Graph::bipartite_projection_of`], [`BipartiteGraph::projection`] | `igraph_bipartite_projection` | the one-mode projections |
//! | [`Graph::bipartite_game_gnp`] | `igraph_bipartite_game_gnp` | random *G(n1, n2, p)* graph |
//! | [`Graph::bipartite_game_gnm`] | `igraph_bipartite_game_gnm` | random *G(n1, n2, m)* graph |
//! | [`Graph::bipartite_iea_game`] | `igraph_bipartite_iea_game` (implemented with `igraph_bipartite_game_gnm`, working around an igraph 1.0.0 and 1.0.1 bug) | random multigraph by independent edge assignment |
//! | [`Graph::is_matching`] | `igraph_is_matching` | validity of a matching |
//! | [`Graph::is_maximal_matching`] | `igraph_is_maximal_matching` | maximality of a matching |
//! | [`Graph::maximum_bipartite_matching`], [`Graph::maximum_bipartite_matching_eps`], [`BipartiteGraph::maximum_matching`] | `igraph_maximum_bipartite_matching` | maximum (weighted) bipartite matching |
//!
//! # Related functionality in other modules
//!
//! | Rust | What it does |
//! |------|--------------|
//! | [`Graph::layout_bipartite`] ([`layout`](crate::layout)) | two-row drawing of a bipartite graph, from its types |
//! | [`Graph::realize_bipartite_degree_sequence`] ([`constructors`](crate::constructors)) | deterministic bipartite graph with given degrees |
//! | [`is_bigraphical`](crate::mixing::is_bigraphical) ([`mixing`](crate::mixing)) | whether two degree sequences can be realized by a bipartite graph |
//! | [`Graph::full_multipartite`], [`Graph::turan`] ([`constructors`](crate::constructors)) | complete *k*-partite graphs, generalizing [`Graph::full_bipartite`] |
//! | [`Graph::erdos_renyi_game_gnp`], [`Graph::erdos_renyi_game_gnm`] ([`games`](crate::games)) | one-mode versions of the random bipartite games |
//! | [`Graph::maxflow_value`] ([`flow`](crate::flow)) | maximum flow: a bipartite matching is a unit-capacity flow from one class to the other |
//! | [`Graph::girth`] ([`structural`](crate::structural)) | shortest cycle; a graph is bipartite iff it has no odd cycle |
//! | [`Graph::is_bipartite_coloring`] ([`cliques`](crate::cliques)) | whether a *given* types vector is a proper 2-coloring, and how the edges are oriented |
//! | [`Graph::read_graph_pajek`] ([`foreign`](crate::foreign)) | reads two-mode Pajek networks, storing the types in the `type` vertex attribute when [`attributes`](crate::attributes) are enabled |
//!
//! # Matchings
//!
//! A matching is represented, as in igraph, by a vector with one entry per
//! vertex: entry `i` is the vertex matched to `i`, or [`UNMATCHED`] (`-1`)
//! if `i` is unmatched. [`BipartiteMatching`] offers [`mate`](BipartiteMatching::mate)
//! and [`pairs`](BipartiteMatching::pairs) to read it in a friendlier way.
//!
//! See the igraph C documentation chapter on
//! [bipartite graphs](https://igraph.org/c/html/latest/igraph-Bipartite.html)
//! and the section on
//! [matchings](https://igraph.org/c/html/latest/igraph-Structural.html#matchings).

use crate::{
    constants::{EdgeTypeSw, NeighborMode},
    error::{Error, Result},
    ffi::*,
    graph::{Graph, VertexId},
    igraph_call,
    matrix::Matrix,
    vector::{Vector, VectorBool, VectorInt},
};
use std::ptr;

/// Marker used in matching vectors for unmatched vertices (`-1`).
pub const UNMATCHED: VertexId = -1;

/// Largest matrix entry accepted as an edge multiplicity by
/// [`Graph::biadjacency`] (2^53, beyond which not every integer is an `f64`).
const MAX_MULTIPLICITY: f64 = 9_007_199_254_740_992.0;

/// Converts a count into an `igraph_int_t`, failing on overflow.
fn to_int(n: usize, what: &str) -> Result<igraph_int_t> {
    igraph_int_t::try_from(n).map_err(|_| Error::invalid(format!("{what} is too large: {n}")))
}

/// Checks that `types` has one entry per vertex of `graph`.
fn check_types(graph: &Graph, types: &[bool]) -> Result<()> {
    if types.len() != graph.vcount() {
        return Err(Error::invalid(format!(
            "the types vector has length {}, but the graph has {} vertices",
            types.len(),
            graph.vcount()
        )));
    }
    Ok(())
}

/// Checks that `weights`, if given, has one entry per edge of `graph`.
fn check_weights(graph: &Graph, weights: Option<&[f64]>) -> Result<()> {
    match weights {
        Some(w) if w.len() != graph.ecount() => Err(Error::invalid(format!(
            "the weights vector has length {}, but the graph has {} edges",
            w.len(),
            graph.ecount()
        ))),
        _ => Ok(()),
    }
}

/// Checks that every edge of `graph` joins vertices of different types
/// (`types` must already have the right length).
///
/// igraph's matching code (`misc/matching.c`, unchanged in igraph 1.0.0 and
/// 1.0.1) only partially performs this check: with an undetected same-type
/// edge it silently returns wrong results.
fn check_bipartite_edges(graph: &Graph, types: &[bool]) -> Result<()> {
    for (e, (u, v)) in graph.edge_list().into_iter().enumerate() {
        if types[u as usize] == types[v as usize] {
            return Err(Error::invalid(format!(
                "edge {e} ({u}, {v}) joins two vertices of the same type"
            )));
        }
    }
    Ok(())
}

/// Checks that all `weights`, if given, are finite.
fn check_finite_weights(weights: Option<&[f64]>) -> Result<()> {
    match weights
        .into_iter()
        .flatten()
        .enumerate()
        .find(|(_, x)| !x.is_finite())
    {
        Some((i, x)) => Err(Error::invalid(format!(
            "the weight of edge {i} is not finite: {x}"
        ))),
        None => Ok(()),
    }
}

/// A graph together with its vertex types: the output of the bipartite
/// constructors and generators of this module.
///
/// Vertices with type `false` form the first class, vertices with type
/// `true` the second one. The fields are public: take them apart freely
/// (or use [`into_parts`](Self::into_parts)).
#[derive(Debug, Clone, PartialEq)]
pub struct BipartiteGraph {
    /// The graph.
    pub graph: Graph,
    /// The type of each vertex, indexed by vertex id.
    pub types: Vec<bool>,
}

impl BipartiteGraph {
    /// Creates a bipartite graph from vertex types and a list of edges,
    /// checking that every edge connects vertices of different types
    /// (`igraph_create_bipartite`).
    ///
    /// The number of vertices is `types.len()`. See [`Graph::create_bipartite`].
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if an edge
    /// joins two vertices of the same type, or
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) if an
    /// endpoint is out of range.
    ///
    /// # Examples
    /// ```
    /// use igraph::bipartite::BipartiteGraph;
    /// let b = BipartiteGraph::new(vec![false, true, false], &[(0, 1), (2, 1)], false).unwrap();
    /// assert_eq!(b.part(true), vec![1]);
    /// assert!(BipartiteGraph::new(vec![false, false], &[(0, 1)], false).is_err());
    /// ```
    pub fn new(types: Vec<bool>, edges: &[(VertexId, VertexId)], directed: bool) -> Result<Self> {
        let graph = Graph::create_bipartite(&types, edges, directed)?;
        Ok(Self { graph, types })
    }

    /// Wraps `graph` with a 2-coloring found by [`Graph::bipartite_types`],
    /// or returns `Ok(None)` if the graph is not bipartite.
    ///
    /// # Examples
    /// ```
    /// use igraph::{bipartite::BipartiteGraph, prelude::*};
    /// let square = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], 4, false).unwrap();
    /// let b = BipartiteGraph::from_graph(square).unwrap().unwrap();
    /// assert_eq!(b.types, vec![false, true, false, true]);
    /// let triangle = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, false).unwrap();
    /// assert!(BipartiteGraph::from_graph(triangle).unwrap().is_none());
    /// ```
    pub fn from_graph(graph: Graph) -> Result<Option<Self>> {
        Ok(graph.bipartite_types()?.map(|types| Self { graph, types }))
    }

    /// Splits `self` into the graph and its types.
    pub fn into_parts(self) -> (Graph, Vec<bool>) {
        (self.graph, self.types)
    }

    /// The ids of the vertices having type `kind`, in increasing order.
    pub fn part(&self, kind: bool) -> Vec<VertexId> {
        (0..)
            .zip(&self.types)
            .filter(|&(_, &t)| t == kind)
            .map(|(v, _)| v)
            .collect()
    }

    /// The number of vertices of type `false` and of type `true`.
    pub fn part_sizes(&self) -> (usize, usize) {
        let n2 = self.types.iter().filter(|&&t| t).count();
        (self.types.len() - n2, n2)
    }

    /// Both one-mode projections, see [`Graph::bipartite_projection`].
    pub fn projection(&self) -> Result<BipartiteProjection> {
        self.graph.bipartite_projection(&self.types, None)
    }

    /// The bipartite adjacency matrix, see [`Graph::get_biadjacency`].
    pub fn biadjacency(&self, weights: Option<&[f64]>) -> Result<Biadjacency> {
        self.graph.get_biadjacency(&self.types, weights)
    }

    /// A maximum (weighted) matching, see [`Graph::maximum_bipartite_matching`].
    pub fn maximum_matching(&self, weights: Option<&[f64]>) -> Result<BipartiteMatching> {
        self.graph.maximum_bipartite_matching(&self.types, weights)
    }
}

/// A bipartite graph with edge weights, the output of [`Graph::weighted_biadjacency`].
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedBipartiteGraph {
    /// The graph.
    pub graph: Graph,
    /// The type of each vertex, indexed by vertex id.
    pub types: Vec<bool>,
    /// The weight of each edge, indexed by edge id.
    pub weights: Vec<f64>,
}

/// A bipartite adjacency matrix, the output of [`Graph::get_biadjacency`].
#[derive(Debug, Clone, PartialEq)]
pub struct Biadjacency {
    /// The matrix: rows correspond to vertices of type `false`, columns to
    /// vertices of type `true`; an element is the number of edges (or the
    /// total weight of the edges) between the two vertices.
    pub matrix: Matrix,
    /// The vertex id of each row.
    pub row_ids: Vec<VertexId>,
    /// The vertex id of each column.
    pub col_ids: Vec<VertexId>,
}

/// Vertex and edge counts of the two projections of a bipartite graph,
/// the output of [`Graph::bipartite_projection_size`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionSize {
    /// Number of vertices of the first projection (type `false`).
    pub vcount1: usize,
    /// Number of edges of the first projection.
    pub ecount1: usize,
    /// Number of vertices of the second projection (type `true`).
    pub vcount2: usize,
    /// Number of edges of the second projection.
    pub ecount2: usize,
}

/// The two one-mode projections of a bipartite graph, the output of
/// [`Graph::bipartite_projection`].
///
/// Each projection has one vertex per vertex of the corresponding type,
/// numbered in increasing order of the original ids; two vertices are
/// connected if they share at least one neighbor in the other class.
#[derive(Debug, Clone, PartialEq)]
pub struct BipartiteProjection {
    /// The first projection.
    pub proj1: Graph,
    /// The second projection.
    pub proj2: Graph,
    /// For each edge of `proj1`, the number of common neighbors of its
    /// endpoints in the original graph (in a multigraph: the number of
    /// paths of length two between them).
    pub multiplicity1: Vec<i64>,
    /// For each edge of `proj2`, the number of common neighbors of its
    /// endpoints in the original graph (in a multigraph: the number of
    /// paths of length two between them).
    pub multiplicity2: Vec<i64>,
}

/// A maximum matching of a bipartite graph, the output of
/// [`Graph::maximum_bipartite_matching`].
#[derive(Debug, Clone, PartialEq)]
pub struct BipartiteMatching {
    /// The number of matched pairs (the cardinality of the matching).
    pub size: usize,
    /// The total weight of the matched edges; equal to `size` for
    /// unweighted graphs.
    pub weight: f64,
    /// For each vertex, the vertex it is matched to, or [`UNMATCHED`].
    pub matching: Vec<VertexId>,
}

impl BipartiteMatching {
    /// The vertex matched to `v`, or `None` if `v` is unmatched or out of range.
    pub fn mate(&self, v: VertexId) -> Option<VertexId> {
        usize::try_from(v)
            .ok()
            .and_then(|i| self.matching.get(i))
            .copied()
            .filter(|&m| m >= 0)
    }

    /// The matched pairs `(u, v)` with `u < v`, sorted by `u`.
    pub fn pairs(&self) -> Vec<(VertexId, VertexId)> {
        (0..)
            .zip(&self.matching)
            .filter(|&(u, &v)| v > u)
            .map(|(u, &v)| (u, v))
            .collect()
    }

    /// Whether vertex `v` is matched.
    pub fn is_matched(&self, v: VertexId) -> bool {
        self.mate(v).is_some()
    }
}

/// Parameters shared by the random bipartite generators
/// [`Graph::bipartite_game_gnp`] and [`Graph::bipartite_game_gnm`].
///
/// The default is an undirected simple graph from the classic
/// (edge-unlabeled) Erdős–Rényi model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BipartiteGameOptions {
    /// Whether to generate a directed graph (default `false`).
    pub directed: bool,
    /// Direction of the edges in directed graphs (default [`NeighborMode::Out`]):
    /// `Out` points from bottom (type `false`) to top (type `true`) vertices,
    /// `In` the other way around; with `All` each edge direction is sampled
    /// independently, so mutual edges may appear. Ignored if undirected.
    pub mode: NeighborMode,
    /// Allowed edge types (default [`EdgeTypeSw::Simple`]): use
    /// [`EdgeTypeSw::Multi`] to allow multi-edges. Self-loops are never
    /// possible in a bipartite graph, so [`EdgeTypeSw::Loops`] is the same
    /// as [`EdgeTypeSw::Simple`].
    pub allowed_edge_types: EdgeTypeSw,
    /// If `true`, sample uniformly from ordered edge lists (edge-labeled
    /// graphs) instead of the classic model (default `false`). Only
    /// relevant when multi-edges are allowed: with simple graphs,
    /// [`Graph::bipartite_game_gnm`] only shuffles the edge order and
    /// [`Graph::bipartite_game_gnp`] fails with
    /// [`ErrorKind::Unimplemented`](crate::ErrorKind::Unimplemented).
    pub edge_labeled: bool,
}

impl Default for BipartiteGameOptions {
    fn default() -> Self {
        Self {
            directed: false,
            mode: NeighborMode::Out,
            allowed_edge_types: EdgeTypeSw::Simple,
            edge_labeled: false,
        }
    }
}

impl BipartiteGameOptions {
    /// Sets [`directed`](Self::directed) and [`mode`](Self::mode).
    pub fn with_directed(mut self, directed: bool, mode: NeighborMode) -> Self {
        self.directed = directed;
        self.mode = mode;
        self
    }

    /// Sets [`allowed_edge_types`](Self::allowed_edge_types).
    pub fn with_allowed_edge_types(mut self, allowed: EdgeTypeSw) -> Self {
        self.allowed_edge_types = allowed;
        self
    }

    /// Sets [`edge_labeled`](Self::edge_labeled).
    pub fn with_edge_labeled(mut self, edge_labeled: bool) -> Self {
        self.edge_labeled = edge_labeled;
        self
    }
}

impl igraph_t {
    /// Whether the graph is bipartite, i.e. whether its vertices can be
    /// 2-colored so that no edge joins vertices of the same color.
    ///
    /// Equivalently, the graph has no cycle of odd length; a graph with a
    /// self-loop is never bipartite. Edge directions are ignored. Use
    /// [`bipartite_types`](Self::bipartite_types) to also get a coloring.
    ///
    /// Binds [`igraph_is_bipartite`](https://igraph.org/c/html/latest/igraph-Bipartite.html#igraph_is_bipartite).
    /// Time complexity: O(|V|+|E|).
    ///
    /// See also [`Graph::girth`] (a bipartite graph has even girth, or no
    /// cycle at all), [`Graph::is_bipartite_coloring`] (checks a given
    /// types vector instead of finding one) and
    /// [`Graph::vertex_coloring_greedy`] (colorings with more than two
    /// colors).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let c6 = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)], 6, false).unwrap();
    /// assert!(c6.is_bipartite().unwrap());
    /// let c5 = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 5, false).unwrap();
    /// assert!(!c5.is_bipartite().unwrap());
    /// // The Heawood graph (incidence graph of the Fano plane) is bipartite,
    /// // the Petersen graph is not (it has 5-cycles).
    /// assert!(Graph::famous("Heawood").unwrap().is_bipartite().unwrap());
    /// assert!(!Graph::famous("Petersen").unwrap().is_bipartite().unwrap());
    /// ```
    pub fn is_bipartite(&self) -> Result<bool> {
        let mut res = false;
        igraph_call!(igraph_is_bipartite(self, &mut res, ptr::null_mut()))?;
        Ok(res)
    }

    /// A 2-coloring (vertex types) witnessing that the graph is bipartite, or
    /// `None` if it is not.
    ///
    /// The coloring is not unique in general: e.g. each connected component
    /// can be flipped independently. Binds
    /// [`igraph_is_bipartite`](https://igraph.org/c/html/latest/igraph-Bipartite.html#igraph_is_bipartite).
    /// Time complexity: O(|V|+|E|).
    ///
    /// The types can be fed directly to the other functions of this module,
    /// or to [`Graph::layout_bipartite`] to draw the graph in two rows.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let path = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// let types = path.bipartite_types().unwrap().unwrap();
    /// assert_ne!(types[0], types[1]);
    /// assert_ne!(types[1], types[2]);
    /// // Draw it in two rows: `true` vertices at y = 0, `false` ones at y = 1.
    /// let layout = path.layout_bipartite(&types, 1.0, 1.0, 100).unwrap();
    /// for (v, &t) in types.iter().enumerate() {
    ///     assert_eq!(layout[(v, 1)], if t { 0.0 } else { 1.0 });
    /// }
    /// ```
    pub fn bipartite_types(&self) -> Result<Option<Vec<bool>>> {
        let mut res = false;
        let mut types = VectorBool::new();
        igraph_call!(igraph_is_bipartite(self, &mut res, &mut types))?;
        Ok(res.then(|| types.into()))
    }

    /// Creates a bipartite graph from vertex types and edges, checking that
    /// every edge connects vertices of different types.
    ///
    /// The graph has `types.len()` vertices; every endpoint in `edges` must
    /// be smaller than that. It is [`Graph::from_edges`] plus a
    /// bipartiteness check. [`BipartiteGraph::new`] does the same and keeps
    /// the types together with the graph.
    ///
    /// Binds [`igraph_create_bipartite`](https://igraph.org/c/html/latest/igraph-Bipartite.html#igraph_create_bipartite).
    /// Time complexity: O(|V|+|E|).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if an edge
    /// joins two vertices of the same type,
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) if an
    /// endpoint is out of range.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let types = [false, true, false, true];
    /// let g = Graph::create_bipartite(&types, &[(0, 1), (1, 2), (2, 3)], true).unwrap();
    /// assert_eq!((g.vcount(), g.ecount(), g.is_directed()), (4, 3, true));
    /// let err = Graph::create_bipartite(&types, &[(0, 2)], false).unwrap_err();
    /// assert_eq!(err.kind(), ErrorKind::InvalidValue);
    /// ```
    pub fn create_bipartite(
        types: &[bool],
        edges: &[(VertexId, VertexId)],
        directed: bool,
    ) -> Result<Graph> {
        let n = types.len() as igraph_int_t;
        if let Some(&(u, v)) = edges
            .iter()
            .find(|&&(u, v)| u < 0 || v < 0 || u >= n || v >= n)
        {
            return Err(Error::new(
                crate::ErrorKind::InvalidVertexId,
                format!("edge ({u}, {v}) has an endpoint outside 0..{n}"),
            ));
        }
        let flat: Vec<VertexId> = edges.iter().flat_map(|&(u, v)| [u, v]).collect();
        let t = VectorBool::view(types);
        let e = VectorInt::view(&flat);
        Graph::init_with(|g| unsafe {
            igraph_create_bipartite(g, t.as_ptr(), e.as_ptr(), directed)
        })
    }

    /// Creates the complete bipartite graph *K(n1, n2)*.
    ///
    /// The first `n1` vertices have type `false`, the following `n2` type
    /// `true`, and every vertex of the first kind is connected to every
    /// vertex of the second kind. In directed graphs, `mode` gives the edge
    /// directions: [`NeighborMode::Out`] from the first kind to the second,
    /// [`NeighborMode::In`] the opposite, [`NeighborMode::All`] mutual edges
    /// (so `2·n1·n2` edges). `mode` is ignored for undirected graphs.
    ///
    /// Binds [`igraph_full_bipartite`](https://igraph.org/c/html/latest/igraph-Bipartite.html#igraph_full_bipartite).
    /// Time complexity: O(|V|+|E|).
    ///
    /// See also [`Graph::full_multipartite`] for complete *k*-partite graphs
    /// and [`Graph::realize_bipartite_degree_sequence`] for bipartite graphs
    /// with prescribed degrees.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let k33 = Graph::full_bipartite(3, 3, false, NeighborMode::All).unwrap();
    /// assert_eq!(k33.graph.ecount(), 9);
    /// assert_eq!(k33.types, vec![false, false, false, true, true, true]);
    /// let d = Graph::full_bipartite(2, 3, true, NeighborMode::All).unwrap();
    /// assert_eq!(d.graph.ecount(), 12);
    /// ```
    pub fn full_bipartite(
        n1: usize,
        n2: usize,
        directed: bool,
        mode: NeighborMode,
    ) -> Result<BipartiteGraph> {
        let (n1, n2) = (to_int(n1, "n1")?, to_int(n2, "n2")?);
        let mut types = VectorBool::new();
        let graph = Graph::init_with(|g| unsafe {
            igraph_full_bipartite(g, &mut types, n1, n2, directed, mode.into())
        })?;
        Ok(BipartiteGraph {
            graph,
            types: types.into(),
        })
    }

    /// Creates a bipartite graph from a bipartite adjacency matrix.
    ///
    /// For an `n × m` matrix, the graph has `n` vertices of type `false`
    /// (the rows, ids `0..n`) followed by `m` vertices of type `true` (the
    /// columns, ids `n..n+m`). If `multiple` is `false`, one edge is created
    /// for every non-zero element; if it is `true`, element `(i, j)` gives
    /// the number of edges between row `i` and column `j` (fractional parts
    /// are discarded; negative, infinite and NaN entries are an error). In directed graphs
    /// `mode` gives the directions: [`NeighborMode::Out`] from rows to
    /// columns, [`NeighborMode::In`] from columns to rows, [`NeighborMode::All`]
    /// mutual edges.
    ///
    /// Binds [`igraph_biadjacency`](https://igraph.org/c/html/latest/igraph-Bipartite.html#igraph_biadjacency).
    /// Time complexity: O(n·m) plus the number of created edges.
    ///
    /// See also [`Graph::adjacency`] for ordinary (square) adjacency
    /// matrices, and [`get_biadjacency`](Self::get_biadjacency) for the
    /// inverse conversion.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `multiple` is `true` and an entry is negative, not finite, or too
    /// large to be an edge count.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let m = Matrix::from_rows(&[[0.0, 1.0, 2.0], [1.0, 0.0, 0.0]]).unwrap();
    /// let b = Graph::biadjacency(&m, false, NeighborMode::All, true).unwrap();
    /// assert_eq!(b.types, vec![false, false, true, true, true]);
    /// let mut edges = b.graph.edge_list();
    /// edges.sort();
    /// assert_eq!(edges, vec![(0, 3), (0, 4), (0, 4), (1, 2)]);
    /// ```
    pub fn biadjacency(
        biadjmatrix: &Matrix,
        directed: bool,
        mode: NeighborMode,
        multiple: bool,
    ) -> Result<BipartiteGraph> {
        if multiple {
            // igraph 1.0.0 and 1.0.1 only reject negative entries and then
            // convert them to integers with a plain C cast, which is undefined
            // behavior for NaN, infinite or out-of-range values.
            if let Some(&x) = biadjmatrix
                .as_slice()
                .iter()
                .find(|x| !(x.is_finite() && (0.0..=MAX_MULTIPLICITY).contains(*x)))
            {
                return Err(Error::invalid(format!(
                    "invalid edge multiplicity in the bipartite adjacency matrix: {x}"
                )));
            }
        }
        let mut types = VectorBool::new();
        let graph = Graph::init_with(|g| unsafe {
            igraph_biadjacency(g, &mut types, biadjmatrix, directed, mode.into(), multiple)
        })?;
        Ok(BipartiteGraph {
            graph,
            types: types.into(),
        })
    }

    /// Creates a weighted bipartite graph from a bipartite adjacency matrix.
    ///
    /// Like [`biadjacency`](Self::biadjacency), but a single edge is created
    /// for every non-zero element, and the element becomes its weight (any
    /// real value, including negative, infinite and NaN ones). With
    /// [`NeighborMode::All`] in directed graphs, both edges of a mutual pair
    /// get the same weight.
    ///
    /// Binds [`igraph_weighted_biadjacency`](https://igraph.org/c/html/latest/igraph-Bipartite.html#igraph_weighted_biadjacency).
    /// Time complexity: O(n·m).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let m = Matrix::from_rows(&[[0.0, -4.5, 2.3], [-0.1, 0.0, 0.0]]).unwrap();
    /// let w = Graph::weighted_biadjacency(&m, false, NeighborMode::All).unwrap();
    /// assert_eq!(w.graph.ecount(), 3);
    /// let mut weights = w.weights.clone();
    /// weights.sort_by(f64::total_cmp);
    /// assert_eq!(weights, vec![-4.5, -0.1, 2.3]);
    /// ```
    pub fn weighted_biadjacency(
        biadjmatrix: &Matrix,
        directed: bool,
        mode: NeighborMode,
    ) -> Result<WeightedBipartiteGraph> {
        let mut types = VectorBool::new();
        let mut weights = Vector::new();
        let graph = Graph::init_with(|g| unsafe {
            igraph_weighted_biadjacency(
                g,
                &mut types,
                &mut weights,
                biadjmatrix,
                directed,
                mode.into(),
            )
        })?;
        Ok(WeightedBipartiteGraph {
            graph,
            types: types.into(),
            weights: weights.into(),
        })
    }

    /// The bipartite adjacency matrix of the graph (the inverse of
    /// [`biadjacency`](Self::biadjacency)).
    ///
    /// Rows correspond to vertices of type `false`, columns to vertices of
    /// type `true`, both in increasing id order (their ids are returned in
    /// [`Biadjacency::row_ids`] and [`Biadjacency::col_ids`]). Element
    /// `(i, j)` is the number of edges between the two vertices, regardless
    /// of their direction, or the sum of their weights if `weights` is
    /// given. Edges within a class are ignored, with a warning.
    ///
    /// Binds [`igraph_get_biadjacency`](https://igraph.org/c/html/latest/igraph-Bipartite.html#igraph_get_biadjacency).
    /// Time complexity: O(|E|).
    ///
    /// See also [`Graph::get_adjacency`] for the full `|V| × |V|` adjacency
    /// matrix: the bipartite one is its off-diagonal block.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `types`
    /// or `weights` have the wrong length.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (2, 1), (2, 3)], 4, false).unwrap();
    /// let b = g.get_biadjacency(&[false, true, false, true], Some(&[1.0, 2.0, 5.0])).unwrap();
    /// assert_eq!(b.matrix.to_rows(), vec![vec![1.0, 0.0], vec![2.0, 5.0]]);
    /// assert_eq!((b.row_ids, b.col_ids), (vec![0, 2], vec![1, 3]));
    /// ```
    pub fn get_biadjacency(&self, types: &[bool], weights: Option<&[f64]>) -> Result<Biadjacency> {
        check_types(self, types)?;
        check_weights(self, weights)?;
        let t = VectorBool::view(types);
        let w = weights.map(Vector::view);
        let wp = w.as_ref().map_or(ptr::null(), |v| v.as_ptr());
        let mut matrix = Matrix::new();
        let mut row_ids = VectorInt::new();
        let mut col_ids = VectorInt::new();
        igraph_call!(igraph_get_biadjacency(
            self,
            t.as_ptr(),
            wp,
            &mut matrix,
            &mut row_ids,
            &mut col_ids
        ))?;
        Ok(Biadjacency {
            matrix,
            row_ids: row_ids.into(),
            col_ids: col_ids.into(),
        })
    }

    /// The number of vertices and edges of the two projections, without
    /// computing them.
    ///
    /// Useful to estimate the memory needed by
    /// [`bipartite_projection`](Self::bipartite_projection) on large graphs.
    /// The first projection is the one of the vertices of type `false`.
    ///
    /// Binds [`igraph_bipartite_projection_size`](https://igraph.org/c/html/latest/igraph-Bipartite.html#igraph_bipartite_projection_size).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `types`
    /// has the wrong length or an edge joins vertices of the same type.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let k23 = Graph::full_bipartite(2, 3, false, NeighborMode::All).unwrap();
    /// let s = k23.graph.bipartite_projection_size(&k23.types).unwrap();
    /// assert_eq!((s.vcount1, s.ecount1, s.vcount2, s.ecount2), (2, 1, 3, 3));
    /// ```
    pub fn bipartite_projection_size(&self, types: &[bool]) -> Result<ProjectionSize> {
        check_types(self, types)?;
        let t = VectorBool::view(types);
        let (mut v1, mut e1, mut v2, mut e2) = (0, 0, 0, 0);
        igraph_call!(igraph_bipartite_projection_size(
            self,
            t.as_ptr(),
            &mut v1,
            &mut e1,
            &mut v2,
            &mut e2
        ))?;
        Ok(ProjectionSize {
            vcount1: v1 as usize,
            ecount1: e1 as usize,
            vcount2: v2 as usize,
            ecount2: e2 as usize,
        })
    }

    /// Both one-mode projections of a bipartite graph.
    ///
    /// The projection onto a class has one vertex per vertex of that class
    /// (in increasing order of the original ids), and two vertices are
    /// connected if they have at least one common neighbor in the other
    /// class; the number of common neighbors is returned as the edge
    /// *multiplicity*. Edge directions are ignored and the projections are
    /// undirected simple graphs.
    ///
    /// By default (`probe1 = None`), [`proj1`](BipartiteProjection::proj1)
    /// is the projection of the vertices of type `false`. If `probe1` is
    /// `Some(v)`, `proj1` is the projection containing vertex `v` instead.
    ///
    /// Binds [`igraph_bipartite_projection`](https://igraph.org/c/html/latest/igraph-Bipartite.html#igraph_bipartite_projection).
    /// Time complexity: O(|V|·d²+|E|), d being the average degree.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `types`
    /// has the wrong length or an edge joins vertices of the same type;
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) if
    /// `probe1` is not a vertex.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // Two papers (0, 1) and three authors (2, 3, 4).
    /// let types = [false, false, true, true, true];
    /// let g = Graph::create_bipartite(&types, &[(0, 2), (0, 3), (1, 3), (1, 4)], false).unwrap();
    /// let p = g.bipartite_projection(&types, None).unwrap();
    /// // The papers share author 3.
    /// assert_eq!(p.proj1.edge_list(), vec![(0, 1)]);
    /// // Co-authorship: 2-3 and 3-4 (projected ids 0, 1, 2).
    /// assert_eq!(p.proj2.edge_list(), vec![(0, 1), (1, 2)]);
    /// assert_eq!(p.multiplicity2, vec![1, 1]);
    /// ```
    pub fn bipartite_projection(
        &self,
        types: &[bool],
        probe1: Option<VertexId>,
    ) -> Result<BipartiteProjection> {
        check_types(self, types)?;
        let probe1 = match probe1 {
            Some(v) if v < 0 || v as usize >= self.vcount() => {
                return Err(Error::new(
                    crate::ErrorKind::InvalidVertexId,
                    format!("invalid probe vertex {v}"),
                ));
            }
            Some(v) => v,
            None => -1,
        };
        let t = VectorBool::view(types);
        let mut m1 = VectorInt::new();
        let mut m2 = VectorInt::new();
        let mut proj2 = None;
        let mut inner_error = None;
        let proj1 = Graph::init_with(|p1| {
            match Graph::init_with(|p2| unsafe {
                igraph_bipartite_projection(self, t.as_ptr(), p1, p2, &mut m1, &mut m2, probe1)
            }) {
                Ok(g) => {
                    proj2 = Some(g);
                    igraph_error_type_t_IGRAPH_SUCCESS
                }
                // On failure igraph has already destroyed `proj1`: nothing leaks.
                Err(e) => {
                    let code = e.code();
                    inner_error = Some(e);
                    code
                }
            }
        });
        match (proj1, proj2) {
            (Ok(proj1), Some(proj2)) => Ok(BipartiteProjection {
                proj1,
                proj2,
                multiplicity1: m1.into(),
                multiplicity2: m2.into(),
            }),
            // The inner call consumed igraph's error record: report its error.
            (Err(e), _) => Err(inner_error.unwrap_or(e)),
            (Ok(_), None) => Err(Error::new(crate::ErrorKind::Internal, "missing projection")),
        }
    }

    /// The one-mode projection onto the vertices of type `kind`, with its
    /// edge multiplicities.
    ///
    /// It computes only one of the two projections of
    /// [`bipartite_projection`](Self::bipartite_projection), saving time and
    /// memory when the other one is not needed.
    ///
    /// Binds [`igraph_bipartite_projection`](https://igraph.org/c/html/latest/igraph-Bipartite.html#igraph_bipartite_projection).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `types`
    /// has the wrong length or an edge joins vertices of the same type.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let star = Graph::full_bipartite(1, 4, false, NeighborMode::All).unwrap();
    /// // The four leaves all share the center: they form a K4.
    /// let (leaves, mult) = star.graph.bipartite_projection_of(&star.types, true).unwrap();
    /// assert_eq!((leaves.vcount(), leaves.ecount()), (4, 6));
    /// assert!(mult.iter().all(|&m| m == 1));
    /// ```
    pub fn bipartite_projection_of(&self, types: &[bool], kind: bool) -> Result<(Graph, Vec<i64>)> {
        check_types(self, types)?;
        let t = VectorBool::view(types);
        let mut mult = VectorInt::new();
        let proj = if kind {
            Graph::init_with(|p| unsafe {
                igraph_bipartite_projection(
                    self,
                    t.as_ptr(),
                    ptr::null_mut(),
                    p,
                    ptr::null_mut(),
                    &mut mult,
                    -1,
                )
            })?
        } else {
            Graph::init_with(|p| unsafe {
                igraph_bipartite_projection(
                    self,
                    t.as_ptr(),
                    p,
                    ptr::null_mut(),
                    &mut mult,
                    ptr::null_mut(),
                    -1,
                )
            })?
        };
        Ok((proj, mult.into()))
    }

    /// A random bipartite graph from the *G(n1, n2, p)* model.
    ///
    /// Every possible edge between the `n1` bottom vertices (type `false`,
    /// ids `0..n1`) and the `n2` top vertices (type `true`) is realized
    /// independently with probability `p`. When multi-edges are allowed
    /// (see [`BipartiteGameOptions`]), `p` is the *expected number* of edges
    /// between each pair and may exceed 1.
    ///
    /// Binds [`igraph_bipartite_game_gnp`](https://igraph.org/c/html/latest/igraph-Bipartite.html#igraph_bipartite_game_gnp).
    /// Uses the thread's default random number generator (see [`crate::rng`]).
    /// Time complexity: O(|V|+|E|).
    ///
    /// See also [`Graph::erdos_renyi_game_gnp`], the one-mode version, and
    /// [`bipartite_game_gnm`](Self::bipartite_game_gnm) for a fixed number
    /// of edges.
    ///
    /// Self-loops are impossible in a bipartite graph, so
    /// [`EdgeTypeSw::Loops`] behaves like [`EdgeTypeSw::Simple`].
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `p` is
    /// NaN, infinite, negative, or larger than 1 without multi-edges;
    /// [`ErrorKind::Unimplemented`](crate::ErrorKind::Unimplemented) if
    /// [`BipartiteGameOptions::edge_labeled`] is set without multi-edges
    /// (igraph 1.0.0 and 1.0.1 implement the edge-labeled G(n1, n2, p) model
    /// only for multigraphs).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// rng::seed(7).unwrap();
    /// let b = Graph::bipartite_game_gnp(10, 20, 0.3, &Default::default()).unwrap();
    /// assert_eq!(b.graph.vcount(), 30);
    /// assert!(b.graph.is_bipartite().unwrap());
    /// let full = Graph::bipartite_game_gnp(3, 4, 1.0, &Default::default()).unwrap();
    /// assert_eq!(full.graph.ecount(), 12);
    /// ```
    pub fn bipartite_game_gnp(
        n1: usize,
        n2: usize,
        p: f64,
        options: &BipartiteGameOptions,
    ) -> Result<BipartiteGraph> {
        // igraph 1.0.0 and 1.0.1 let NaN through (`p < 0.0 || p > 1.0` is
        // false for NaN) and misbehave on infinite values.
        if !p.is_finite() {
            return Err(Error::invalid(format!(
                "the edge probability (or multiplicity) must be finite, got {p}"
            )));
        }
        let (n1, n2) = (to_int(n1, "n1")?, to_int(n2, "n2")?);
        let mut types = VectorBool::new();
        let graph = Graph::init_with(|g| unsafe {
            igraph_bipartite_game_gnp(
                g,
                &mut types,
                n1,
                n2,
                p,
                options.directed,
                options.mode.into(),
                options.allowed_edge_types.into(),
                options.edge_labeled,
            )
        })?;
        Ok(BipartiteGraph {
            graph,
            types: types.into(),
        })
    }

    /// A uniformly random bipartite graph from the *G(n1, n2, m)* model:
    /// `n1` bottom vertices (type `false`), `n2` top vertices (type `true`)
    /// and exactly `m` edges.
    ///
    /// With [`BipartiteGameOptions::edge_labeled`], sampling is uniform over
    /// ordered edge lists rather than over graphs (this matters only when
    /// multi-edges are allowed).
    ///
    /// Binds [`igraph_bipartite_game_gnm`](https://igraph.org/c/html/latest/igraph-Bipartite.html#igraph_bipartite_game_gnm).
    /// Uses the thread's default random number generator (see [`crate::rng`]).
    /// Time complexity: O(|V|+|E|).
    ///
    /// See also [`Graph::erdos_renyi_game_gnm`], the one-mode version, and
    /// [`bipartite_iea_game`](Self::bipartite_iea_game) for a faster,
    /// non-uniform multigraph model.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `m` is
    /// positive while `n1` or `n2` is zero, or, without multi-edges, if `m`
    /// is larger than the number of possible edges (`n1·n2`, or `2·n1·n2`
    /// for directed graphs with [`NeighborMode::All`]).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// rng::seed(1).unwrap();
    /// let b = Graph::bipartite_game_gnm(5, 5, 12, &Default::default()).unwrap();
    /// assert_eq!(b.graph.ecount(), 12);
    /// assert!(Graph::bipartite_game_gnm(2, 2, 5, &Default::default()).is_err());
    /// ```
    pub fn bipartite_game_gnm(
        n1: usize,
        n2: usize,
        m: usize,
        options: &BipartiteGameOptions,
    ) -> Result<BipartiteGraph> {
        let (n1, n2, m) = (to_int(n1, "n1")?, to_int(n2, "n2")?, to_int(m, "m")?);
        let mut types = VectorBool::new();
        let graph = Graph::init_with(|g| unsafe {
            igraph_bipartite_game_gnm(
                g,
                &mut types,
                n1,
                n2,
                m,
                options.directed,
                options.mode.into(),
                options.allowed_edge_types.into(),
                options.edge_labeled,
            )
        })?;
        Ok(BipartiteGraph {
            graph,
            types: types.into(),
        })
    }

    /// A random bipartite multigraph by *independent edge assignment* (IEA):
    /// each of the `m` edges joins a uniformly random bottom–top pair,
    /// independently of the others.
    ///
    /// There are `n1` bottom vertices (type `false`) and `n2` top vertices
    /// (type `true`). The resulting multigraphs are not uniformly sampled:
    /// a graph has probability proportional to `1 / ∏ A_ij!`, so all simple
    /// graphs are equally likely. `mode` directs the edges of directed
    /// graphs as in [`BipartiteGameOptions::mode`].
    ///
    /// This model is *experimental* in igraph (1.0.0 and 1.0.1). It is the
    /// same as [`bipartite_game_gnm`](Self::bipartite_game_gnm) with
    /// multi-edges allowed and [`BipartiteGameOptions::edge_labeled`] set.
    /// Uses the thread's default random number generator (see [`crate::rng`]).
    /// See also [`Graph::iea_game`], the one-mode version.
    ///
    /// Implements [`igraph_bipartite_iea_game`](https://igraph.org/c/html/latest/igraph-Bipartite.html#igraph_bipartite_iea_game)
    /// through `igraph_bipartite_game_gnm`: in igraph 1.0.0 and 1.0.1 the C
    /// function forwards to the *edge-unlabeled* multigraph G(n1, n2, m)
    /// model by mistake, and so samples multigraphs uniformly instead of by
    /// independent edge assignment (unlike its one-mode counterpart
    /// `igraph_iea_game`). This wrapper calls the edge-labeled model
    /// directly, which is the documented IEA process.
    /// Time complexity: O(|V|+|E|).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// rng::seed(3).unwrap();
    /// // 100 edges between 2 x 2 vertices: plenty of multi-edges.
    /// let b = Graph::bipartite_iea_game(2, 2, 100, false, NeighborMode::Out).unwrap();
    /// assert_eq!(b.graph.ecount(), 100);
    /// assert!(b.graph.edge_list().iter().all(|&(u, v)| u < 2 && v >= 2));
    /// ```
    pub fn bipartite_iea_game(
        n1: usize,
        n2: usize,
        m: usize,
        directed: bool,
        mode: NeighborMode,
    ) -> Result<BipartiteGraph> {
        let (n1, n2, m) = (to_int(n1, "n1")?, to_int(n2, "n2")?, to_int(m, "m")?);
        let mut types = VectorBool::new();
        // Not `igraph_bipartite_iea_game`, which is buggy: see the docs.
        let graph = Graph::init_with(|g| unsafe {
            igraph_bipartite_game_gnm(
                g,
                &mut types,
                n1,
                n2,
                m,
                directed,
                mode.into(),
                EdgeTypeSw::Multi.into(),
                true,
            )
        })?;
        Ok(BipartiteGraph {
            graph,
            types: types.into(),
        })
    }

    /// Whether `matching` is a valid matching of the graph.
    ///
    /// `matching` has one entry per vertex: the vertex it is matched to, or
    /// [`UNMATCHED`] (`-1`). It is valid if its length is the number of
    /// vertices, it is symmetric (`matching[matching[i]] == i`), and every
    /// matched pair is joined by an edge (directions are ignored). If
    /// `types` is given, matched vertices must also have different types.
    /// An invalid vector yields `Ok(false)`, not an error.
    ///
    /// Binds [`igraph_is_matching`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_is_matching).
    /// Time complexity: O(|V|+|E|).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let path = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// assert!(path.is_matching(None, &[1, 0, 3, 2]).unwrap());
    /// assert!(path.is_matching(None, &[-1, 2, 1, -1]).unwrap());
    /// assert!(!path.is_matching(None, &[3, -1, -1, 0]).unwrap()); // no edge 0-3
    /// ```
    pub fn is_matching(&self, types: Option<&[bool]>, matching: &[VertexId]) -> Result<bool> {
        self.check_matching(types, matching, igraph_is_matching)
    }

    /// Whether `matching` is a *maximal* matching of the graph: a valid
    /// matching (see [`is_matching`](Self::is_matching)) that cannot be
    /// extended, i.e. no two unmatched vertices are adjacent (if `types` is
    /// given, only edges joining vertices of different types count).
    ///
    /// A maximal matching is not necessarily *maximum* (largest): on the
    /// path `0-1-2-3`, matching only `1-2` is maximal but not maximum.
    ///
    /// Binds [`igraph_is_maximal_matching`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_is_maximal_matching).
    /// Time complexity: O(|V|+|E|).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let path = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// assert!(path.is_maximal_matching(None, &[-1, 2, 1, -1]).unwrap());
    /// assert!(!path.is_maximal_matching(None, &[1, 0, -1, -1]).unwrap()); // 2-3 can be added
    /// ```
    pub fn is_maximal_matching(
        &self,
        types: Option<&[bool]>,
        matching: &[VertexId],
    ) -> Result<bool> {
        self.check_matching(types, matching, igraph_is_maximal_matching)
    }

    fn check_matching(
        &self,
        types: Option<&[bool]>,
        matching: &[VertexId],
        f: unsafe extern "C" fn(
            *const igraph_t,
            *const igraph_vector_bool_t,
            *const igraph_vector_int_t,
            *mut igraph_bool_t,
        ) -> igraph_error_t,
    ) -> Result<bool> {
        if let Some(t) = types {
            check_types(self, t)?;
        }
        let t = types.map(VectorBool::view);
        let tp = t.as_ref().map_or(ptr::null(), |v| v.as_ptr());
        let m = VectorInt::view(matching);
        let mut res = false;
        igraph_call!(f(self, tp, m.as_ptr(), &mut res))?;
        Ok(res)
    }

    /// A maximum matching of a bipartite graph: the largest set of edges no
    /// two of which share a vertex or, with `weights`, the matching of
    /// largest total weight.
    ///
    /// Unweighted matchings are found with a push-relabel algorithm, in
    /// O(√|V|·|E|) time; weighted ones with the Hungarian algorithm, in
    /// O(|V|·|E|) time. The weighted algorithm is reliable only for integer
    /// weights; slacks of at most `f64::EPSILON` are considered zero (use
    /// [`maximum_bipartite_matching_eps`](Self::maximum_bipartite_matching_eps)
    /// to tune this tolerance). Edge directions are ignored.
    ///
    /// A weighted maximum matching maximizes the total weight, not the
    /// number of pairs: edges of negative weight are never chosen, so
    /// [`BipartiteMatching::size`] may be smaller than in the unweighted
    /// case.
    ///
    /// The size of an unweighted maximum matching equals the maximum flow
    /// from one class to the other with unit capacities (see
    /// [`Graph::maxflow_value`]) and, by König's theorem, the size of a
    /// minimum vertex cover. Check a result with
    /// [`is_matching`](Self::is_matching) and
    /// [`is_maximal_matching`](Self::is_maximal_matching).
    ///
    /// Binds [`igraph_maximum_bipartite_matching`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_maximum_bipartite_matching).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `types`
    /// or `weights` have the wrong length, an edge joins two vertices of
    /// the same type (self-loops included), or a weight is not finite.
    /// These are checked on the Rust side, because igraph's own checks are
    /// incomplete (in igraph 1.0.0 and 1.0.1): it can silently return a wrong
    /// matching, or loop forever on infinite weights.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // Weighted example from igraph's unit tests.
    /// let types: Vec<bool> = (0..10).map(|i| i >= 5).collect();
    /// let g = Graph::from_edges(&[(0, 8), (2, 7), (3, 7), (3, 8), (4, 5), (4, 9)], 10, false).unwrap();
    /// let w = [8.0, 5.0, 9.0, 18.0, 20.0, 13.0];
    /// let m = g.maximum_bipartite_matching(&types, Some(&w)).unwrap();
    /// assert_eq!(m.weight, 43.0); // 2-7, 3-8 and 4-5: 5 + 18 + 20
    /// assert_eq!(m.pairs(), vec![(2, 7), (3, 8), (4, 5)]);
    /// // Vertices 0 and 3 compete for 8, 2 and 3 for 7: at most 3 pairs.
    /// let m = g.maximum_bipartite_matching(&types, None).unwrap();
    /// assert_eq!((m.size, m.weight), (3, 3.0));
    /// ```
    pub fn maximum_bipartite_matching(
        &self,
        types: &[bool],
        weights: Option<&[f64]>,
    ) -> Result<BipartiteMatching> {
        self.maximum_bipartite_matching_eps(types, weights, f64::EPSILON)
    }

    /// Like [`maximum_bipartite_matching`](Self::maximum_bipartite_matching),
    /// with an explicit tolerance `eps` for the equality tests of the
    /// weighted algorithm (ignored if `weights` is `None`).
    ///
    /// An edge is considered *tight* when its slack (the difference between
    /// the sum of the dual labels of its endpoints and its weight) is at most
    /// `eps`; a small positive value avoids the accumulation of rounding
    /// errors with fractional weights, while with integer weights
    /// `eps = 0.0` is safe. A negative `eps` is clamped to zero by igraph
    /// (with a warning).
    ///
    /// Binds [`igraph_maximum_bipartite_matching`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_maximum_bipartite_matching).
    ///
    /// # Errors
    /// As [`maximum_bipartite_matching`](Self::maximum_bipartite_matching);
    /// also [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `eps` is NaN (igraph would never terminate).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let k22 = Graph::full_bipartite(2, 2, false, NeighborMode::All).unwrap();
    /// // Edges 0-2, 0-3, 1-2, 1-3: the diagonal 0-2, 1-3 is the best.
    /// let w = [0.5, 0.25, 0.25, 0.5];
    /// let m = k22.graph.maximum_bipartite_matching_eps(&k22.types, Some(&w), 1e-9).unwrap();
    /// assert_eq!(m.pairs(), vec![(0, 2), (1, 3)]);
    /// assert_eq!(m.weight, 1.0);
    /// ```
    pub fn maximum_bipartite_matching_eps(
        &self,
        types: &[bool],
        weights: Option<&[f64]>,
        eps: f64,
    ) -> Result<BipartiteMatching> {
        check_types(self, types)?;
        check_weights(self, weights)?;
        check_finite_weights(weights)?;
        check_bipartite_edges(self, types)?;
        if eps.is_nan() {
            return Err(Error::invalid("eps must not be NaN"));
        }
        let t = VectorBool::view(types);
        let w = weights.map(Vector::view);
        let wp = w.as_ref().map_or(ptr::null(), |v| v.as_ptr());
        let mut size: igraph_int_t = 0;
        let mut weight: igraph_real_t = 0.0;
        let mut matching = VectorInt::new();
        igraph_call!(igraph_maximum_bipartite_matching(
            self,
            t.as_ptr(),
            &mut size,
            &mut weight,
            &mut matching,
            wp,
            eps
        ))?;
        Ok(BipartiteMatching {
            size: size as usize,
            weight,
            matching: matching.into(),
        })
    }
}

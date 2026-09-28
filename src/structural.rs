//! Basic structural properties of graphs (`igraph_structural.h`).
//!
//! This module binds the functions declared in igraph's
//! [`igraph_structural.h`](https://igraph.org/c/html/latest/igraph-Structural.html)
//! header, as further methods of [`Graph`]. They answer the everyday
//! questions one asks about a network:
//!
//! - *how big and how dense is it?* ([`density`](Graph::density),
//!   [`mean_degree`](Graph::mean_degree), [`maxdegree`](Graph::maxdegree),
//!   [`strength`](Graph::strength));
//! - *is it a simple graph?* (loops, multi-edges and mutual edges);
//! - *what kind of graph is it?* (tree, forest, acyclic, complete, chordal,
//!   perfect; cliques and independent sets);
//! - *how are degrees correlated?* (average nearest neighbor degree,
//!   `k_nn(k)`, rich-club density sequence);
//! - *which spanning trees does it have?* (minimum and uniformly random);
//! - *what does its Laplacian look like?* (dense and sparse).
//!
//! # Example
//!
//! ```
//! use igraph::prelude::*;
//!
//! // A 5-cycle with a chord, plus a pendant vertex.
//! let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (4, 5)], 6, false)?;
//! assert_eq!(g.mean_degree(true)?, 14.0 / 6.0);
//! assert_eq!(g.maxdegree(.., NeighborMode::All, Loops::Twice)?, 3);
//! assert!((g.density(None, false)? - 7.0 / 15.0).abs() < 1e-12);
//! assert!(g.is_simple(true)?);
//! assert_eq!(g.girth()?, Some(3));
//! assert!(!g.is_tree(NeighborMode::All)?);
//! assert!(g.is_clique(&[0, 1, 2], false)?);
//! assert!(g.is_independent_vertex_set(&[1, 3, 5])?);
//!
//! // A spanning tree of a connected graph has n - 1 edges.
//! let mst = g.minimum_spanning_tree(None, MstAlgorithm::Automatic)?;
//! assert_eq!(mst.len(), 5);
//!
//! // Zachary's karate club: 34 members, 78 friendships, and the two leaders
//! // (vertices 0 and 33) are the best connected members.
//! let karate = Graph::famous("Zachary")?;
//! assert_eq!(karate.maxdegree(.., NeighborMode::All, Loops::Twice)?, 17);
//! let hubs = karate.sort_vertex_ids_by_degree(.., NeighborMode::All, Loops::Twice, Order::Descending, false)?;
//! assert_eq!(&hubs[..2], &[33, 0]);
//! # Ok::<(), igraph::Error>(())
//! ```
//!
//! # Provided functionality
//!
//! | Topic | Methods of [`Graph`] |
//! |-------|----------------------|
//! | Adjacency | [`are_adjacent`](Graph::are_adjacent), [`subcomponent`](Graph::subcomponent) |
//! | Density and degrees | [`density`](Graph::density), [`mean_degree`](Graph::mean_degree), [`maxdegree`](Graph::maxdegree), [`strength`](Graph::strength), [`sort_vertex_ids_by_degree`](Graph::sort_vertex_ids_by_degree), [`diversity`](Graph::diversity) |
//! | Loops and multi-edges | [`has_loop`](Graph::has_loop), [`count_loops`](Graph::count_loops), [`is_loop`](Graph::is_loop), [`has_multiple`](Graph::has_multiple), [`is_multiple`](Graph::is_multiple), [`count_multiple`](Graph::count_multiple), [`count_multiple_1`](Graph::count_multiple_1), [`is_simple`](Graph::is_simple) |
//! | Reciprocity | [`reciprocity`](Graph::reciprocity), [`is_mutual`](Graph::is_mutual), [`has_mutual`](Graph::has_mutual) |
//! | Graph classes | [`is_tree`](Graph::is_tree), [`tree_root`](Graph::tree_root), [`is_forest`](Graph::is_forest), [`forest_roots`](Graph::forest_roots), [`is_acyclic`](Graph::is_acyclic), [`is_complete`](Graph::is_complete), [`is_perfect`](Graph::is_perfect), [`is_chordal`](Graph::is_chordal), [`is_chordal_with`](Graph::is_chordal_with), [`maximum_cardinality_search`](Graph::maximum_cardinality_search) |
//! | Vertex sets | [`is_clique`](Graph::is_clique), [`is_independent_vertex_set`](Graph::is_independent_vertex_set) |
//! | Cycles | [`girth`](Graph::girth), [`girth_with_cycle`](Graph::girth_with_cycle) |
//! | Spanning trees | [`minimum_spanning_tree`](Graph::minimum_spanning_tree), [`random_spanning_tree`](Graph::random_spanning_tree), [`unfold_tree`](Graph::unfold_tree) |
//! | Degree correlations | [`avg_nearest_neighbor_degree`](Graph::avg_nearest_neighbor_degree), [`degree_correlation_vector`](Graph::degree_correlation_vector), [`rich_club_sequence`](Graph::rich_club_sequence) |
//! | Spectral | [`get_laplacian`](Graph::get_laplacian), [`get_laplacian_sparse`](Graph::get_laplacian_sparse), [`get_laplacian_sparsemat`](Graph::get_laplacian_sparsemat), [`LaplacianNormalization`] |
//!
//! All the functions of the header are covered.
//!
//! # See also
//!
//! - Degrees and neighbors: [`Graph::degree`], [`Graph::neighbors`] (core);
//!   removing loops and multi-edges: [`Graph::simplify`] ([`operators`](crate::operators)).
//! - Connectivity: [`Graph::connected_components`], [`Graph::is_connected`],
//!   [`Graph::count_reachable`] ([`components`](crate::components)).
//! - Cycles and DAGs: [`Graph::find_cycle`], [`Graph::minimum_cycle_basis`],
//!   [`Graph::is_dag`], [`Graph::topological_sorting`] ([`cycles`](crate::cycles)).
//! - Cliques, independent sets and colorings: [`Graph::largest_cliques`],
//!   [`Graph::clique_number`], [`Graph::independence_number`],
//!   [`Graph::vertex_coloring_greedy`] ([`cliques`](crate::cliques)).
//! - Degree correlations as a single number: [`Graph::assortativity_degree`],
//!   [`Graph::joint_degree_matrix`] ([`mixing`](crate::mixing)).
//! - Matrices of a graph: [`Graph::get_adjacency`], [`Graph::get_adjacency_sparse`]
//!   ([`conversion`](crate::conversion)); spectra of the Laplacian with
//!   [`lapack_dsyevr`](crate::linalg::lapack_dsyevr) and
//!   [`Graph::laplacian_spectral_embedding`] ([`linalg`](crate::linalg)).
//! - Trees: [`Graph::kary_tree`] ([`constructors`](crate::constructors)),
//!   [`Graph::tree_game`] ([`games`](crate::games)), [`Graph::to_prufer`]
//!   ([`conversion`](crate::conversion)), [`Graph::bfs`] ([`visitor`](crate::visitor)).
//!
//! # Notes on igraph 1.0.0 and 1.0.1
//!
//! A few wrappers work around behaviours of the C library that are present
//! in both igraph 1.0.0 and 1.0.1 (the source files involved are unchanged
//! in 1.0.1), so that the Rust API is consistent and memory safe:
//!
//! - [`count_multiple_1`](Graph::count_multiple_1) validates the edge id and
//!   reports an undirected self-loop once, like [`count_multiple`](Graph::count_multiple);
//! - [`get_laplacian_sparse`](Graph::get_laplacian_sparse) (and
//!   [`get_laplacian_sparsemat`](Graph::get_laplacian_sparsemat)) ignore edge
//!   directions with [`NeighborMode::All`], like the dense
//!   [`get_laplacian`](Graph::get_laplacian);
//! - [`is_chordal_with`](Graph::is_chordal_with) checks that the given
//!   vertex orders are permutations (igraph only checks their lengths);
//! - [`diversity`](Graph::diversity) recomputes the value of degree-one
//!   vertices, which igraph derives from the weight of edge 0 instead of
//!   the vertex's own edge.
//!
//! One wrapper differs from the C calling convention on purpose:
//! [`random_spanning_tree`](Graph::random_spanning_tree) spells igraph's
//! documented "negative vertex id means all components" as `None`, and
//! rejects negative ids instead of silently spanning all components.

use crate::{
    constants::*,
    error::{Error, Result},
    ffi::*,
    graph::{EdgeId, Graph, VertexId},
    igraph_call,
    linalg::SparseMat,
    matrix::Matrix,
    selector::{EdgeSelector, VertexSelector},
    vector::{Vector, VectorBool, VectorInt},
};
use std::{collections::BTreeMap, ptr};

crate::ffi_enum! {
    /// Normalization of the Laplacian matrix (`igraph_laplacian_normalization_t`),
    /// used by [`Graph::get_laplacian`], [`Graph::get_laplacian_sparse`] and
    /// [`Graph::get_laplacian_sparsemat`].
    ///
    /// `A` is the (possibly weighted) adjacency matrix and `D` the diagonal
    /// matrix of degrees (or strengths, if weighted); out-, in- or total
    /// degrees are used according to the `mode` argument.
    pub enum LaplacianNormalization: igraph_laplacian_normalization_t {
        /// Unnormalized Laplacian, `L = D - A`.
        Unnormalized = igraph_laplacian_normalization_t_IGRAPH_LAPLACIAN_UNNORMALIZED,
        /// Symmetrically normalized Laplacian, `L = I - D^(-1/2) A D^(-1/2)`.
        Symmetric = igraph_laplacian_normalization_t_IGRAPH_LAPLACIAN_SYMMETRIC,
        /// Left-stochastic normalized Laplacian, `L = I - D^-1 A` (rows sum to
        /// zero when `D` holds the row sums of `A`, e.g. out-degrees).
        Left = igraph_laplacian_normalization_t_IGRAPH_LAPLACIAN_LEFT,
        /// Right-stochastic normalized Laplacian, `L = I - A D^-1` (columns sum
        /// to zero when `D` holds the column sums of `A`, e.g. in-degrees).
        Right = igraph_laplacian_normalization_t_IGRAPH_LAPLACIAN_RIGHT,
    }
}

impl Default for LaplacianNormalization {
    /// [`LaplacianNormalization::Unnormalized`].
    fn default() -> Self {
        Self::Unnormalized
    }
}

/// Result of [`Graph::avg_nearest_neighbor_degree`].
#[derive(Debug, Clone, PartialEq)]
pub struct NeighborDegree {
    /// `knn[i]` is the (possibly weighted) average degree of the neighbors of
    /// the `i`-th selected vertex; NaN for isolated vertices (with weights:
    /// for vertices of zero strength).
    pub knn: Vec<f64>,
    /// The `k_nn(k)` degree correlation function: `knnk[k - 1]` is the mean
    /// of `knn` over the selected vertices of degree `k` (with weights, the
    /// mean weighted by their strengths), NaN for degrees that do not occur.
    /// Note the shift: the first element is for degree one. Its length is the
    /// maximum degree (according to `mode`) of the whole graph.
    pub knnk: Vec<f64>,
}

/// Result of [`Graph::maximum_cardinality_search`].
#[derive(Debug, Clone, PartialEq)]
pub struct CardinalitySearch {
    /// `alpha[v]` is the rank of vertex `v`, in `0..n`; visiting vertices by
    /// *decreasing* rank always picks the vertex with most visited neighbors.
    pub alpha: Vec<i64>,
    /// The inverse permutation of `alpha`: `alpham1[r]` is the vertex of rank
    /// `r`, i.e. the vertices in *reverse* maximum cardinality search order.
    pub alpham1: Vec<VertexId>,
}

/// Result of [`Graph::is_chordal_with`].
#[derive(Debug, Clone, PartialEq)]
pub struct Chordality {
    /// Whether the graph is chordal.
    pub is_chordal: bool,
    /// The fill-in (chordal completion): edges whose addition makes the
    /// graph chordal. Empty for chordal graphs; not necessarily minimal.
    pub fill_in: Vec<(VertexId, VertexId)>,
    /// The triangulated graph: a copy of the original graph (same
    /// directedness) with the fill-in edges appended after the original ones.
    pub triangulated: Graph,
}

/// Result of [`Graph::unfold_tree`].
#[derive(Debug, Clone, PartialEq)]
pub struct UnfoldedTree {
    /// The unfolded tree (or forest); it has the directedness of the input.
    pub tree: Graph,
    /// `vertex_index[v]` is the vertex of the original graph that vertex `v`
    /// of `tree` is a copy of. The first `vcount` entries are the identity.
    pub vertex_index: Vec<VertexId>,
}

/// Returns a pointer to the optional weight view, or null.
fn weights_ptr(w: &Option<crate::vector::View<'_, Vector>>) -> *const igraph_vector_t {
    w.as_ref().map_or(ptr::null(), |v| v.as_ptr())
}

/// Checks that an optional weight vector has one entry per edge.
fn check_weights(graph: &Graph, weights: Option<&[f64]>) -> Result<()> {
    match weights {
        Some(w) if w.len() != graph.ecount() => Err(Error::invalid(format!(
            "weight vector length ({}) must match the number of edges ({})",
            w.len(),
            graph.ecount()
        ))),
        _ => Ok(()),
    }
}

/// Whether `p` is a permutation of `0..n`.
fn is_permutation(p: &[i64], n: usize) -> bool {
    if p.len() != n {
        return false;
    }
    let mut seen = vec![false; n];
    p.iter()
        .all(|&x| (0..n as i64).contains(&x) && !std::mem::replace(&mut seen[x as usize], true))
}

/// Runs `f` (a C call building an `IGRAPH_ALL`, `IGRAPH_NO_MULTIPLE`
/// adjacency list) so that igraph's property cache cannot switch off the
/// removal of duplicate neighbors.
///
/// igraph 1.0.1 `igraph_adjlist_init` / `igraph_lazy_adjlist_init` skip the
/// deduplication when the cache says the graph has no multi-edges. On a
/// directed graph with a mutual pair `u -> v`, `v -> u` that is wrong in
/// `IGRAPH_ALL` mode (each of `u`, `v` is listed twice as a neighbor of the
/// other), and `adjlist_init` then even caches `HAS_MULTI = true`. For
/// directed graphs the cache is therefore cleared before and after `f`.
/// Undirected graphs are not affected.
/// See [`Graph::with_fresh_multi_cache`](crate::graph); this delegates to it
/// (which also drops the cache when `f` panics).
pub(crate) fn with_multi_cache_guard<T>(graph: &Graph, f: impl FnOnce() -> T) -> T {
    graph.with_fresh_multi_cache(f)
}

impl igraph_t {
    // ------------------------------------------------------------------
    // Basic queries
    // ------------------------------------------------------------------

    /// Decides whether there is an edge from `v1` to `v2`.
    ///
    /// In directed graphs the direction matters (`v1 → v2`); in undirected
    /// graphs the relation is symmetric.
    ///
    /// Binds [`igraph_are_adjacent`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_are_adjacent).
    /// Time complexity: `O(min(log d1, log d2))` where `d1` is the
    /// (out-)degree of `v1` and `d2` the (in-)degree of `v2`.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) for
    /// invalid vertex ids.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1)], 3, true)?;
    /// assert!(g.are_adjacent(0, 1)?);
    /// assert!(!g.are_adjacent(1, 0)?);
    /// assert_eq!(g.are_adjacent(0, 9).unwrap_err().kind(), ErrorKind::InvalidVertexId);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// See also [`Graph::neighbors`] and [`Graph::get_adjacency`] for the
    /// whole neighborhood or adjacency matrix.
    pub fn are_adjacent(&self, v1: VertexId, v2: VertexId) -> Result<bool> {
        let mut res = false;
        igraph_call!(igraph_are_adjacent(self, v1, v2, &mut res))?;
        Ok(res)
    }

    /// The multiplicity of the selected edges: for each edge, the number of
    /// edges between its two endpoints (itself included).
    ///
    /// A simple graph gives all ones. In directed graphs, `(a, b)` and
    /// `(b, a)` are different edges and don't count towards each other.
    ///
    /// Binds [`igraph_count_multiple`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_count_multiple).
    /// Time complexity: `O(E d)`, `E` the number of edges to check and `d`
    /// the average degree of their tails.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (0, 1), (1, 2), (0, 1)], 3, false)?;
    /// assert_eq!(g.count_multiple(..)?, vec![3, 3, 1, 3]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// See also [`is_multiple`](Self::is_multiple), and [`Graph::simplify`]
    /// to merge parallel edges.
    pub fn count_multiple<'a>(&self, edges: impl Into<EdgeSelector<'a>>) -> Result<Vec<i64>> {
        let es = edges.into().to_raw()?;
        let mut res = VectorInt::new();
        igraph_call!(igraph_count_multiple(self, &mut res, es.get()))?;
        Ok(res.into())
    }

    /// The multiplicity of a single edge, see [`count_multiple`](Self::count_multiple).
    ///
    /// The result always agrees with the corresponding entry of
    /// [`count_multiple`](Self::count_multiple), self-loops of undirected
    /// graphs included: the C function of igraph 1.0.0 and 1.0.1 counts each
    /// undirected self-loop twice (it scans the neighbor list of the tail,
    /// where a loop appears twice), and this wrapper halves that count.
    ///
    /// Binds [`igraph_count_multiple_1`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_count_multiple_1).
    /// Time complexity: `O(d)`, the out-degree of the tail of the edge.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidEdgeId`](crate::ErrorKind::InvalidEdgeId) for an invalid edge id.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 0), (2, 2), (2, 2)], 3, false)?;
    /// assert_eq!(g.count_multiple_1(0)?, 2);
    /// assert_eq!(g.count_multiple_1(2)?, 2); // two self-loops on vertex 2
    /// assert_eq!(g.count_multiple_1(4).unwrap_err().kind(), ErrorKind::InvalidEdgeId);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn count_multiple_1(&self, edge: EdgeId) -> Result<i64> {
        // igraph 1.0.0 and 1.0.1 do not validate the id (they would read
        // out of bounds).
        if !(0..self.ecount() as EdgeId).contains(&edge) {
            return Err(Error::new(
                crate::ErrorKind::InvalidEdgeId,
                format!("invalid edge id {edge}"),
            ));
        }
        let mut res = 0;
        igraph_call!(igraph_count_multiple_1(self, &mut res, edge))?;
        // igraph 1.0.0 and 1.0.1 count the neighbors of the tail, where an
        // undirected self-loop appears twice: halve the count so that it
        // agrees with `igraph_count_multiple` (which reports the number of
        // loops).
        let (from, to) = self.edge(edge)?;
        if from == to && !self.is_directed() {
            res /= 2;
        }
        Ok(res)
    }

    /// The density of the graph: the ratio of the number of edges to the
    /// largest possible number of edges.
    ///
    /// The maximum number of edges is `n(n-1)/2` for undirected and `n(n-1)`
    /// for directed graphs; when `loops` is `true`, self-loops are considered
    /// possible and these become `n(n+1)/2` and `n²`. With `loops = false`
    /// the result is only correct if the graph has no loops (this is not
    /// checked).
    ///
    /// With `weights`, the total edge weight is used instead of the edge
    /// count, and the result may exceed 1 (the same happens with
    /// multigraphs). For the null graph the result is NaN.
    ///
    /// Binds [`igraph_density`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_density).
    /// Time complexity: `O(1)` (`O(|E|)` when weighted).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let triangle = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, false)?;
    /// assert_eq!(triangle.density(None, false)?, 1.0);
    /// assert_eq!(triangle.density(None, true)?, 0.5); // 3 of 6 possible edges
    /// assert_eq!(triangle.density(Some(&[0.5, 0.5, 0.5]), false)?, 0.5);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// See also [`mean_degree`](Self::mean_degree) and
    /// [`rich_club_sequence`](Self::rich_club_sequence), which computes the
    /// density of a sequence of shrinking subgraphs.
    pub fn density(&self, weights: Option<&[f64]>, loops: bool) -> Result<f64> {
        check_weights(self, weights)?;
        let w = weights.map(Vector::view);
        let mut res = 0.0;
        igraph_call!(igraph_density(self, weights_ptr(&w), &mut res, loops))?;
        Ok(res)
    }

    /// The structural diversity index of the selected vertices.
    ///
    /// It is the Shannon entropy of the weights of the incident edges,
    /// normalized by the logarithm of the degree:
    /// `D(i) = H(i) / log k(i)` with `H(i) = -Σ_j p_ij log p_ij` and
    /// `p_ij = w_ij / Σ_l w_il`, `k(i)` being the degree (zero-weight edges
    /// included). Isolated vertices get NaN, vertices of degree one get 0
    /// (NaN if the weight of their only edge is zero, as for any vertex whose
    /// incident weights are all zero). Defined by Eagle, Macy and Claxton
    /// (Science 328, 2010).
    ///
    /// igraph 1.0.0 and 1.0.1 compute the value of degree-one vertices from
    /// the weight of edge 0 rather than of the vertex's own edge; this
    /// wrapper recomputes those entries.
    ///
    /// The graph must be undirected and without multi-edges; `weights`
    /// (required, one non-negative value per edge) are mandatory.
    ///
    /// Binds [`igraph_diversity`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_diversity).
    /// Time complexity: `O(|V| + |E|)`.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for directed
    /// graphs, multigraphs, negative weights or a wrong number of weights.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A star with equal weights has maximal diversity at the center.
    /// let star = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], 4, false)?;
    /// let d = star.diversity(&[1.0, 1.0, 1.0], ..)?;
    /// assert!((d[0] - 1.0).abs() < 1e-12);
    /// assert_eq!(&d[1..], &[0.0, 0.0, 0.0]);
    /// // A zero weight: the center's entropy is log 2 over log 3 (the edge
    /// // still counts in its degree), and leaf 1 has no weight at all.
    /// let d = star.diversity(&[0.0, 1.0, 1.0], ..)?;
    /// assert!((d[0] - 2f64.ln() / 3f64.ln()).abs() < 1e-12);
    /// assert!(d[1].is_nan());
    /// assert_eq!(&d[2..], &[0.0, 0.0]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn diversity<'a>(
        &self,
        weights: &[f64],
        vertices: impl Into<VertexSelector<'a>>,
    ) -> Result<Vec<f64>> {
        check_weights(self, Some(weights))?;
        let vs = vertices.into().to_raw()?;
        let w = Vector::view(weights);
        let mut res = Vector::new();
        igraph_call!(igraph_diversity(self, w.as_ptr(), &mut res, vs.get()))?;
        // igraph 1.0.0 and 1.0.1 decide the value of a degree-one vertex by
        // looking at the weight of edge 0 instead of the vertex's own edge:
        // recompute these entries (0 if the edge weight is positive, NaN if
        // it is zero, as documented).
        let mut ids = VectorInt::new();
        igraph_call!(igraph_vs_as_vector(self, vs.get(), &mut ids))?;
        let degrees = self.degree(&ids, NeighborMode::All, Loops::Twice)?;
        let mut res: Vec<f64> = res.into();
        for ((d, &v), _) in res
            .iter_mut()
            .zip(ids.iter())
            .zip(&degrees)
            .filter(|(_, k)| **k == 1)
        {
            if let [e] = self.incident(v, NeighborMode::All, Loops::Twice)?[..] {
                *d = if weights[e as usize] > 0.0 {
                    0.0
                } else {
                    f64::NAN
                };
            }
        }
        Ok(res)
    }

    /// The girth of the graph: the length of its shortest cycle, or `None`
    /// if the graph has no cycles.
    ///
    /// Edge directions are ignored; self-loops and multi-edges are ignored
    /// as well, i.e. cycles of length 1 or 2 are not considered, so the
    /// girth is always at least 3. `None` is returned exactly for graphs
    /// without such cycles (forests, possibly with loops and multi-edges).
    /// Algorithm by Itai and Rodeh (1977).
    ///
    /// Mutual edges `u -> v`, `v -> u` of a directed graph count as a single
    /// undirected edge. igraph 1.0.1 can get this wrong when its property
    /// cache already records "no multi-edges" (a girth of 2, or heap
    /// corruption in the chordality code), so for directed graphs this
    /// wrapper clears the cache around the C call.
    ///
    /// Binds [`igraph_girth`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_girth).
    /// Time complexity: `O((|V| + |E|)²)` in general, `O(|V| + |E|)` for
    /// acyclic graphs.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // The Petersen graph has girth 5.
    /// assert_eq!(Graph::famous("Petersen")?.girth()?, Some(5));
    ///
    /// let path = Graph::from_edges(&[(0, 1), (1, 2)], 3, false)?;
    /// assert_eq!(path.girth()?, None);
    /// // Loops and multi-edges do not form cycles here.
    /// let multi = Graph::from_edges(&[(0, 0), (0, 1), (0, 1)], 2, false)?;
    /// assert_eq!(multi.girth()?, None);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// See also [`Graph::find_cycle`] (any cycle, respecting directions) and
    /// [`Graph::minimum_cycle_basis`] (a basis of short cycles).
    pub fn girth(&self) -> Result<Option<usize>> {
        let mut girth = 0.0;
        with_multi_cache_guard(self, || {
            igraph_call!(igraph_girth(self, &mut girth, ptr::null_mut()))
        })?;
        Ok(girth.is_finite().then_some(girth as usize))
    }

    /// Like [`girth`](Self::girth), but also returns the vertex ids of one
    /// shortest cycle of length at least 3, in cycle order (consecutive
    /// vertices, and the last and the first, are adjacent); `None` if there
    /// is no such cycle.
    ///
    /// Binds [`igraph_girth`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_girth).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)], 4, false)?;
    /// let (girth, cycle) = g.girth_with_cycle()?.unwrap();
    /// assert_eq!(girth, 3);
    /// assert_eq!(cycle.len(), 3);
    /// assert!(g.is_clique(&cycle, false)?); // a triangle
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn girth_with_cycle(&self) -> Result<Option<(usize, Vec<VertexId>)>> {
        let mut girth = 0.0;
        let mut circle = VectorInt::new();
        with_multi_cache_guard(self, || {
            igraph_call!(igraph_girth(self, &mut girth, &mut circle))
        })?;
        Ok(girth.is_finite().then(|| (girth as usize, circle.into())))
    }

    /// Whether the graph has at least one self-loop.
    ///
    /// The result is cached in the graph, so repeated calls are `O(1)`.
    ///
    /// Binds [`igraph_has_loop`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_has_loop).
    /// Time complexity: `O(|E|)`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let mut g = Graph::from_edges(&[(0, 1), (1, 1)], 2, false)?;
    /// assert!(g.has_loop()?);
    /// g.simplify(true, true)?;
    /// assert!(!g.has_loop()?);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// See also [`count_loops`](Self::count_loops), [`is_loop`](Self::is_loop)
    /// and [`Graph::simplify`].
    pub fn has_loop(&self) -> Result<bool> {
        let mut res = false;
        igraph_call!(igraph_has_loop(self, &mut res))?;
        Ok(res)
    }

    /// Whether the graph has at least one multi-edge (two or more edges
    /// with the same endpoints, and the same direction in directed graphs).
    ///
    /// The result is cached in the graph, so repeated calls are `O(1)`.
    ///
    /// Binds [`igraph_has_multiple`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_has_multiple).
    /// Time complexity: `O(|E| d)`, `d` the average degree.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let mut g = Graph::from_edges(&[(0, 1), (1, 0)], 2, true)?;
    /// assert!(!g.has_multiple()?); // opposite directions
    /// g.add_edge(0, 1)?;
    /// assert!(g.has_multiple()?);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn has_multiple(&self) -> Result<bool> {
        let mut res = false;
        igraph_call!(igraph_has_multiple(self, &mut res))?;
        Ok(res)
    }

    /// The number of self-loops in the graph.
    ///
    /// Binds [`igraph_count_loops`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_count_loops).
    /// Time complexity: `O(|E|)`.
    pub fn count_loops(&self) -> Result<usize> {
        let mut res = 0;
        igraph_call!(igraph_count_loops(self, &mut res))?;
        Ok(res as usize)
    }

    /// For each selected edge, whether it is a self-loop.
    ///
    /// Binds [`igraph_is_loop`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_is_loop).
    /// Time complexity: `O(e)`, the number of edges to check.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 0), (0, 1), (1, 1)], 2, false)?;
    /// assert_eq!(g.is_loop(..)?, vec![true, false, true]);
    /// assert_eq!(g.count_loops()?, 2);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn is_loop<'a>(&self, edges: impl Into<EdgeSelector<'a>>) -> Result<Vec<bool>> {
        let es = edges.into().to_raw()?;
        let mut res = VectorBool::new();
        igraph_call!(igraph_is_loop(self, &mut res, es.get()))?;
        Ok(res.into())
    }

    /// For each selected edge, whether it is a multi-edge.
    ///
    /// Only the *second and further* occurrences of parallel edges are
    /// flagged, so that deleting all the flagged edges yields a graph
    /// without multi-edges (as [`Graph::simplify`] does). In undirected
    /// graphs `(a, b)` and `(b, a)` are parallel; in directed ones they are not.
    ///
    /// Binds [`igraph_is_multiple`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_is_multiple).
    /// Time complexity: `O(e d)`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 1), (0, 1), (1, 0)], 3, true)?;
    /// assert_eq!(g.is_multiple(..)?, vec![false, false, false, true, false]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn is_multiple<'a>(&self, edges: impl Into<EdgeSelector<'a>>) -> Result<Vec<bool>> {
        let es = edges.into().to_raw()?;
        let mut res = VectorBool::new();
        igraph_call!(igraph_is_multiple(self, &mut res, es.get()))?;
        Ok(res.into())
    }

    /// For each selected edge, whether it is *mutual*, i.e. whether the
    /// reversed edge exists as well.
    ///
    /// `loops` decides whether directed self-loops count as mutual. In
    /// undirected graphs every edge is mutual. Multiplicities are not
    /// considered: two `(a, b)` edges and one `(b, a)` edge are all mutual.
    ///
    /// Binds [`igraph_is_mutual`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_is_mutual).
    /// Time complexity: `O(n log d)`, `n` the number of edges to check.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 0), (1, 2), (2, 2)], 3, true)?;
    /// assert_eq!(g.is_mutual(.., true)?, vec![true, true, false, true]);
    /// assert_eq!(g.is_mutual(.., false)?, vec![true, true, false, false]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn is_mutual<'a>(
        &self,
        edges: impl Into<EdgeSelector<'a>>,
        loops: bool,
    ) -> Result<Vec<bool>> {
        let es = edges.into().to_raw()?;
        let mut res = VectorBool::new();
        igraph_call!(igraph_is_mutual(self, &mut res, es.get(), loops))?;
        Ok(res.into())
    }

    /// Whether the graph has at least one mutual edge pair (see
    /// [`is_mutual`](Self::is_mutual)).
    ///
    /// Undirected graphs have mutual edges exactly when they have edges. A
    /// directed graph without mutual edges (and loops) is an *oriented graph*.
    ///
    /// Binds [`igraph_has_mutual`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_has_mutual).
    /// Time complexity: `O(|E| log d)`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let oriented = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, true)?;
    /// assert!(!oriented.has_mutual(true)?);
    /// let with_loop = Graph::from_edges(&[(0, 1), (1, 1)], 2, true)?;
    /// assert!(with_loop.has_mutual(true)?);   // the loop counts as mutual
    /// assert!(!with_loop.has_mutual(false)?);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn has_mutual(&self, loops: bool) -> Result<bool> {
        let mut res = false;
        igraph_call!(igraph_has_mutual(self, &mut res, loops))?;
        Ok(res)
    }

    /// Whether the graph is simple, i.e. it has no self-loops and no
    /// multi-edges.
    ///
    /// With `directed = false`, edge directions are ignored, so a directed
    /// graph with a mutual edge pair is considered non-simple. Ignored for
    /// undirected graphs.
    ///
    /// Binds [`igraph_is_simple`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_is_simple).
    /// Time complexity: `O(|V| + |E|)`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 0)], 2, true)?;
    /// assert!(g.is_simple(true)?);
    /// assert!(!g.is_simple(false)?);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn is_simple(&self, directed: bool) -> Result<bool> {
        let mut res = false;
        igraph_call!(igraph_is_simple(self, &mut res, directed))?;
        Ok(res)
    }

    /// Whether the graph is a tree.
    ///
    /// An undirected graph is a tree if it is connected and has no cycles.
    /// For directed graphs `mode` selects the test: [`NeighborMode::Out`]
    /// for out-trees (arborescences, edges pointing away from the root),
    /// [`NeighborMode::In`] for in-trees, [`NeighborMode::All`] to ignore
    /// directions. The null graph is *not* a tree.
    ///
    /// Binds [`igraph_is_tree`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_is_tree).
    /// Time complexity: `O(|V| + |E|)`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (0, 2), (2, 3)], 4, true)?;
    /// assert!(g.is_tree(NeighborMode::Out)?);
    /// assert!(!g.is_tree(NeighborMode::In)?);
    /// assert_eq!(g.tree_root(NeighborMode::Out)?, Some(0));
    /// // A complete binary tree on 15 vertices.
    /// assert!(Graph::kary_tree(15, 2, TreeMode::Out)?.is_tree(NeighborMode::Out)?);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// See also [`Graph::kary_tree`] and [`Graph::tree_game`] to build trees,
    /// and [`Graph::to_prufer`] to encode them.
    pub fn is_tree(&self, mode: NeighborMode) -> Result<bool> {
        let mut res = false;
        igraph_call!(igraph_is_tree(self, &mut res, ptr::null_mut(), mode.into()))?;
        Ok(res)
    }

    /// The root of the graph if it is a tree (see [`is_tree`](Self::is_tree)),
    /// `None` otherwise.
    ///
    /// For out-trees (in-trees) the root is the unique vertex of zero in-degree
    /// (out-degree); with [`NeighborMode::All`] or in undirected graphs any
    /// vertex can be the root, and vertex 0 is returned.
    ///
    /// Binds [`igraph_is_tree`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_is_tree).
    pub fn tree_root(&self, mode: NeighborMode) -> Result<Option<VertexId>> {
        let mut res = false;
        let mut root = -1;
        igraph_call!(igraph_is_tree(self, &mut res, &mut root, mode.into()))?;
        Ok(res.then_some(root))
    }

    /// Whether the graph is a forest, i.e. every connected component is a
    /// tree (equivalently, there are no undirected cycles).
    ///
    /// `mode` has the same meaning as in [`is_tree`](Self::is_tree). The
    /// null graph *is* a forest. The result is cached for undirected tests.
    ///
    /// Binds [`igraph_is_forest`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_is_forest).
    /// Time complexity: `O(|V| + |E|)`.
    pub fn is_forest(&self, mode: NeighborMode) -> Result<bool> {
        let mut res = false;
        igraph_call!(igraph_is_forest(
            self,
            &mut res,
            ptr::null_mut(),
            mode.into()
        ))?;
        Ok(res)
    }

    /// The roots of the trees if the graph is a forest (see
    /// [`is_forest`](Self::is_forest)), `None` otherwise.
    ///
    /// With [`NeighborMode::All`] or in undirected graphs, one vertex per
    /// component is returned (the smallest id); for out- (in-) forests the
    /// roots are the vertices of zero in- (out-) degree.
    ///
    /// Binds [`igraph_is_forest`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_is_forest).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (2, 3), (2, 4)], 6, false)?;
    /// assert_eq!(g.forest_roots(NeighborMode::All)?, Some(vec![0, 2, 5]));
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn forest_roots(&self, mode: NeighborMode) -> Result<Option<Vec<VertexId>>> {
        let mut res = false;
        let mut roots = VectorInt::new();
        igraph_call!(igraph_is_forest(self, &mut res, &mut roots, mode.into()))?;
        Ok(res.then(|| roots.into()))
    }

    /// Whether the graph has no cycles, taking edge directions into account:
    /// for directed graphs this is the same as being a DAG.
    ///
    /// Self-loops are cycles. The result is cached in the graph.
    ///
    /// Binds [`igraph_is_acyclic`](https://igraph.org/c/html/latest/igraph-Cycles.html#igraph_is_acyclic).
    /// Time complexity: `O(|V| + |E|)`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let dag = Graph::from_edges(&[(0, 1), (1, 2), (0, 2)], 3, true)?;
    /// assert!(dag.is_acyclic()?);
    /// assert!(!dag.is_forest(NeighborMode::All)?); // but it has an undirected cycle
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// See also [`Graph::is_dag`] (false for undirected graphs),
    /// [`Graph::topological_sorting`] and [`Graph::find_cycle`].
    pub fn is_acyclic(&self) -> Result<bool> {
        let mut res = false;
        igraph_call!(igraph_is_acyclic(self, &mut res))?;
        Ok(res)
    }

    /// The maximum degree among the selected vertices (0 if the selection
    /// is empty).
    ///
    /// `mode` selects out-, in- or total degree (ignored for undirected
    /// graphs); `loops` how self-loops are counted.
    ///
    /// Binds [`igraph_maxdegree`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_maxdegree).
    /// Time complexity: `O(v)` when loops are counted, `O(v d)` otherwise.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) for invalid vertices.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (0, 2), (3, 0), (1, 1)], 4, true)?;
    /// assert_eq!(g.maxdegree(.., NeighborMode::Out, Loops::Twice)?, 2);
    /// assert_eq!(g.maxdegree(.., NeighborMode::All, Loops::Twice)?, 3);
    /// assert_eq!(g.maxdegree(&[1], NeighborMode::All, Loops::Twice)?, 3); // loop counted twice
    /// assert_eq!(g.maxdegree(&[1], NeighborMode::All, Loops::None)?, 1);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// See also [`Graph::degree`] for the individual degrees.
    pub fn maxdegree<'a>(
        &self,
        vertices: impl Into<VertexSelector<'a>>,
        mode: NeighborMode,
        loops: Loops,
    ) -> Result<i64> {
        let vs = vertices.into().to_raw()?;
        let mut res = 0;
        igraph_call!(igraph_maxdegree(
            self,
            &mut res,
            vs.get(),
            mode.into(),
            loops.into()
        ))?;
        Ok(res)
    }

    /// The mean degree of the graph.
    ///
    /// In directed graphs the mean out-degree equals the mean in-degree, and
    /// that is what is returned (i.e. `|E| / |V|`); in undirected graphs it is
    /// `2|E| / |V|`. With `loops = false`, self-loops are ignored. For the
    /// null graph the result is NaN.
    ///
    /// Binds [`igraph_mean_degree`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_mean_degree).
    /// Time complexity: `O(1)` with loops, `O(|E|)` without.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 2)], 3, false)?;
    /// assert_eq!(g.mean_degree(true)?, 2.0);
    /// assert!((g.mean_degree(false)? - 4.0 / 3.0).abs() < 1e-12);
    /// assert!(Graph::new(0, false).mean_degree(true)?.is_nan());
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn mean_degree(&self, loops: bool) -> Result<f64> {
        let mut res = 0.0;
        igraph_call!(igraph_mean_degree(self, &mut res, loops))?;
        Ok(res)
    }

    /// The reciprocity of a directed graph.
    ///
    /// With [`Reciprocity::Default`] it is the probability that the reverse
    /// of a randomly chosen edge is also in the graph,
    /// `1 - Σ_ij |A_ij - A_ji| / (2 Σ_ij A_ij)` (in multigraphs each parallel
    /// edge needs its own reverse). With [`Reciprocity::Ratio`] it is the
    /// number of mutually connected (unordered) vertex pairs divided by the
    /// number of connected pairs.
    ///
    /// `ignore_loops` excludes self-loops from the count; otherwise they
    /// count as mutual. Undirected graphs always give 1; directed graphs
    /// without edges give NaN.
    ///
    /// Binds [`igraph_reciprocity`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_reciprocity).
    /// Time complexity: `O(|V| + |E|)`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 1)], 3, true)?;
    /// assert!((g.reciprocity(false, Reciprocity::Default)? - 2.0 / 3.0).abs() < 1e-15);
    /// assert_eq!(g.reciprocity(false, Reciprocity::Ratio)?, 0.5);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// See also [`is_mutual`](Self::is_mutual) for the individual edges.
    pub fn reciprocity(&self, ignore_loops: bool, mode: Reciprocity) -> Result<f64> {
        let mut res = 0.0;
        igraph_call!(igraph_reciprocity(
            self,
            &mut res,
            ignore_loops,
            mode.into()
        ))?;
        Ok(res)
    }

    /// The strength (weighted degree) of the selected vertices: the sum of
    /// the weights of the incident edges.
    ///
    /// Without weights this is the ordinary degree (as `f64`). `mode` and
    /// `loops` are as in [`degree`](Graph::degree).
    ///
    /// Binds [`igraph_strength`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_strength).
    /// Time complexity: `O(|V| + |E|)`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2)], 3, true)?;
    /// let w = [1.5, 2.0, 4.0];
    /// assert_eq!(g.strength(.., NeighborMode::Out, Loops::Twice, Some(&w))?, vec![3.5, 4.0, 0.0]);
    /// assert_eq!(g.strength(.., NeighborMode::In, Loops::Twice, Some(&w))?, vec![0.0, 1.5, 6.0]);
    /// assert_eq!(g.strength(.., NeighborMode::All, Loops::Twice, None)?, vec![2.0, 2.0, 2.0]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn strength<'a>(
        &self,
        vertices: impl Into<VertexSelector<'a>>,
        mode: NeighborMode,
        loops: Loops,
        weights: Option<&[f64]>,
    ) -> Result<Vec<f64>> {
        check_weights(self, weights)?;
        let vs = vertices.into().to_raw()?;
        let w = weights.map(Vector::view);
        let mut res = Vector::new();
        igraph_call!(igraph_strength(
            self,
            &mut res,
            vs.get(),
            mode.into(),
            loops.into(),
            weights_ptr(&w)
        ))?;
        Ok(res.into())
    }

    /// The selected vertices sorted by degree.
    ///
    /// `mode` and `loops` define the degree, `order` the sort direction.
    /// With `only_indices = true` the result contains positions within the
    /// selection instead of vertex ids (this makes no difference when all
    /// vertices are selected).
    ///
    /// Binds [`igraph_sort_vertex_ids_by_degree`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_sort_vertex_ids_by_degree).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A star with center 2.
    /// let g = Graph::from_edges(&[(2, 0), (2, 1), (2, 3), (3, 4)], 5, false)?;
    /// let hubs = g.sort_vertex_ids_by_degree(.., NeighborMode::All, Loops::Twice, Order::Descending, false)?;
    /// assert_eq!(hubs[0], 2);
    /// assert_eq!(hubs[1], 3);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// Sorting by increasing degree gives a natural vertex order for
    /// [`rich_club_sequence`](Self::rich_club_sequence).
    pub fn sort_vertex_ids_by_degree<'a>(
        &self,
        vertices: impl Into<VertexSelector<'a>>,
        mode: NeighborMode,
        loops: Loops,
        order: Order,
        only_indices: bool,
    ) -> Result<Vec<i64>> {
        let vs = vertices.into().to_raw()?;
        let mut res = VectorInt::new();
        igraph_call!(igraph_sort_vertex_ids_by_degree(
            self,
            &mut res,
            vs.get(),
            mode.into(),
            loops.into(),
            order.into(),
            only_indices
        ))?;
        Ok(res.into())
    }

    /// Whether the graph is *perfect*: the chromatic number of every induced
    /// subgraph equals the size of its largest clique.
    ///
    /// The check is based on the strong perfect graph theorem (Chudnovsky,
    /// Robertson, Seymour and Thomas). It may build the complement graph,
    /// consuming a lot of memory on large graphs. Worst-case exponential.
    ///
    /// Binds [`igraph_is_perfect`](https://igraph.org/c/html/latest/igraph-Coloring.html#igraph_is_perfect).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the graph
    /// is directed or not simple.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let c5 = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 5, false)?;
    /// assert!(!c5.is_perfect()?); // an odd hole
    /// let c6 = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)], 6, false)?;
    /// assert!(c6.is_perfect()?); // bipartite
    /// assert!(!Graph::famous("Chvatal")?.is_perfect()?);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// See also [`is_chordal`](Self::is_chordal) (chordal graphs are perfect)
    /// and [`Graph::is_bipartite`] (so are bipartite graphs); for perfect
    /// graphs [`Graph::clique_number`] equals the chromatic number.
    pub fn is_perfect(&self) -> Result<bool> {
        let mut res = false;
        igraph_call!(igraph_is_perfect(self, &mut res))?;
        Ok(res)
    }

    // ------------------------------------------------------------------
    // Structural properties
    // ------------------------------------------------------------------

    /// Whether the graph is complete: all pairs of distinct vertices are
    /// adjacent (in both directions, for directed graphs).
    ///
    /// Self-loops and multi-edges are ignored. The null graph and the
    /// singleton graph are complete.
    ///
    /// Binds [`igraph_is_complete`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_is_complete).
    /// Time complexity: `O(|V| + |E|)`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// assert!(Graph::full(5, true, false)?.is_complete()?);
    /// let one_way = Graph::from_edges(&[(0, 1)], 2, true)?;
    /// assert!(!one_way.is_complete()?); // the edge 1 -> 0 is missing
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn is_complete(&self) -> Result<bool> {
        let mut res = false;
        igraph_call!(igraph_is_complete(self, &mut res))?;
        Ok(res)
    }

    /// Whether the candidate vertices form a clique (all pairs adjacent).
    ///
    /// With `directed = true` in a directed graph, both directions are
    /// required between every pair. Empty and singleton sets are cliques.
    ///
    /// Binds [`igraph_is_clique`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_is_clique).
    /// Time complexity: `O(n² log d)`, `n` the size of the candidate set.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let karate = Graph::famous("Zachary")?;
    /// assert!(karate.is_clique(&[0, 1, 2, 3, 7], false)?);
    /// // A largest clique found by the cliques module is, of course, a clique.
    /// let largest = &karate.largest_cliques()?[0];
    /// assert!(karate.is_clique(largest, false)?);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// See also [`Graph::maximal_cliques`] and [`Graph::largest_cliques`] to
    /// find cliques.
    pub fn is_clique<'a>(
        &self,
        candidate: impl Into<VertexSelector<'a>>,
        directed: bool,
    ) -> Result<bool> {
        let vs = candidate.into().to_raw()?;
        let mut res = false;
        igraph_call!(igraph_is_clique(self, vs.get(), directed, &mut res))?;
        Ok(res)
    }

    /// Whether the candidate vertices form an independent set (no pair is
    /// adjacent). Empty and singleton sets are independent.
    ///
    /// Self-loops are ignored: a vertex with a loop can still be part of an
    /// independent set.
    ///
    /// Binds [`igraph_is_independent_vertex_set`](https://igraph.org/c/html/latest/igraph-Cliques.html#igraph_is_independent_vertex_set).
    /// Time complexity: `O(n² log d)`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let c6 = Graph::ring(6, false, false, true)?;
    /// assert!(c6.is_independent_vertex_set(&[0, 2, 4])?);
    /// assert!(!c6.is_independent_vertex_set(&[0, 1])?);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// See also [`Graph::independence_number`] and
    /// [`Graph::largest_independent_vertex_sets`].
    pub fn is_independent_vertex_set<'a>(
        &self,
        candidate: impl Into<VertexSelector<'a>>,
    ) -> Result<bool> {
        let vs = candidate.into().to_raw()?;
        let mut res = false;
        igraph_call!(igraph_is_independent_vertex_set(self, vs.get(), &mut res))?;
        Ok(res)
    }

    /// The edge ids of a minimum weight spanning tree (a spanning forest if
    /// the graph is disconnected).
    ///
    /// Edge directions are ignored. Without `weights` (or with
    /// [`MstAlgorithm::Unweighted`]) an arbitrary spanning tree is returned.
    /// The result is deterministic. To get a *maximum* spanning tree, negate
    /// the weights. Weights must not be NaN.
    ///
    /// Binds [`igraph_minimum_spanning_tree`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_minimum_spanning_tree).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A square with a heavy edge 3-0 and a light diagonal.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], 4, false)?;
    /// let w = [1.0, 2.0, 1.0, 9.0, 0.5];
    /// let mut tree = g.minimum_spanning_tree(Some(&w), MstAlgorithm::Kruskal)?;
    /// tree.sort();
    /// assert_eq!(tree, vec![0, 2, 4]);
    /// // Keep only the tree edges (and all the vertices).
    /// let t = g.subgraph_from_edges(&tree, false)?;
    /// assert!(t.is_tree(NeighborMode::All)?);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// See also [`random_spanning_tree`](Self::random_spanning_tree) and
    /// [`Graph::subgraph_from_edges`] to turn the edge ids into a graph.
    pub fn minimum_spanning_tree(
        &self,
        weights: Option<&[f64]>,
        method: MstAlgorithm,
    ) -> Result<Vec<EdgeId>> {
        check_weights(self, weights)?;
        let w = weights.map(Vector::view);
        let mut res = VectorInt::new();
        igraph_call!(igraph_minimum_spanning_tree(
            self,
            &mut res,
            weights_ptr(&w),
            method.into()
        ))?;
        Ok(res.into())
    }

    /// The edge ids of a spanning tree sampled uniformly at random, via a
    /// loop-erased random walk (Wilson's algorithm).
    ///
    /// Edge directions are ignored; edge multiplicities affect the sampling
    /// frequency. With `vertex = Some(v)` only the component of `v` is
    /// spanned; with `None`, a random spanning forest of all components is
    /// generated. Draws from the calling thread's default random number
    /// generator: seed it with [`rng::seed`](crate::rng::seed) for
    /// reproducible results (each thread has its own generator).
    ///
    /// The number of distinct spanning trees of a connected graph is given by
    /// Kirchhoff's matrix-tree theorem from the eigenvalues of
    /// [`get_laplacian`](Self::get_laplacian): it is the product of the
    /// non-zero ones divided by the number of vertices.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) if
    /// `vertex` is not a valid vertex id.
    ///
    /// Binds [`igraph_random_spanning_tree`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_random_spanning_tree).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// rng::seed(7)?;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4)], 5, false)?;
    /// assert_eq!(g.random_spanning_tree(None)?.len(), 3);    // forest: 2 + 1 edges
    /// assert_eq!(g.random_spanning_tree(Some(3))?.len(), 1); // only 3-4
    ///
    /// // Same seed, same tree.
    /// rng::seed(7)?;
    /// let a = g.random_spanning_tree(None)?;
    /// rng::seed(7)?;
    /// assert_eq!(g.random_spanning_tree(None)?, a);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn random_spanning_tree(&self, vertex: Option<VertexId>) -> Result<Vec<EdgeId>> {
        if let Some(v) = vertex.filter(|&v| v < 0) {
            return Err(Error::new(
                crate::ErrorKind::InvalidVertexId,
                format!("invalid vertex id {v}"),
            ));
        }
        let mut res = VectorInt::new();
        igraph_call!(igraph_random_spanning_tree(
            self,
            &mut res,
            vertex.unwrap_or(-1)
        ))?;
        Ok(res.into())
    }

    /// The vertices reachable from `vertex` (itself included).
    ///
    /// [`NeighborMode::Out`] follows edges, [`NeighborMode::In`] follows them
    /// backwards, [`NeighborMode::All`] ignores directions (which is *not*
    /// the union of the other two). In undirected graphs this is the
    /// connected component of `vertex`. The vertices are listed in BFS order,
    /// starting with `vertex`.
    ///
    /// Binds [`igraph_subcomponent`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_subcomponent).
    /// Time complexity: `O(|V| + |E|)`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (3, 1)], 4, true)?;
    /// assert_eq!(g.subcomponent(1, NeighborMode::Out)?, vec![1, 2]);
    /// let mut up = g.subcomponent(1, NeighborMode::In)?;
    /// up.sort();
    /// assert_eq!(up, vec![0, 1, 3]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) for
    /// an invalid `vertex`.
    ///
    /// See also [`Graph::connected_components`] (all components at once),
    /// [`Graph::count_reachable`] and [`Graph::bfs`].
    pub fn subcomponent(&self, vertex: VertexId, mode: NeighborMode) -> Result<Vec<VertexId>> {
        let mut res = VectorInt::new();
        igraph_call!(igraph_subcomponent(self, &mut res, vertex, mode.into()))?;
        Ok(res.into())
    }

    /// Unfolds the graph into a tree (or forest) by a breadth-first search
    /// from `roots`, replicating vertices each time they are reached again.
    ///
    /// Every edge of the graph appears exactly once in the result: non-tree
    /// edges lead to fresh copies of their endpoints. `mode` controls which
    /// edges the search follows in directed graphs. Give one root per
    /// component to unfold every component.
    ///
    /// Binds [`igraph_unfold_tree`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_unfold_tree).
    /// Time complexity: `O(|V| + |E|)`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // Unfolding a triangle from vertex 0: the closing edge copies a vertex.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, false)?;
    /// let unfolded = g.unfold_tree(NeighborMode::All, &[0])?;
    /// assert_eq!((unfolded.tree.vcount(), unfolded.tree.ecount()), (4, 3));
    /// assert!(unfolded.tree.is_tree(NeighborMode::All)?);
    /// assert_eq!(&unfolded.vertex_index[..3], &[0, 1, 2]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) for
    /// an invalid root.
    pub fn unfold_tree(&self, mode: NeighborMode, roots: &[VertexId]) -> Result<UnfoldedTree> {
        let r = VectorInt::view(roots);
        let mut index = VectorInt::new();
        let tree = Graph::init_with(|t| unsafe {
            igraph_unfold_tree(self, t, mode.into(), r.as_ptr(), &mut index)
        })?;
        Ok(UnfoldedTree {
            tree,
            vertex_index: index.into(),
        })
    }

    /// Maximum cardinality search (Tarjan and Yannakakis, 1984).
    ///
    /// Assigns a rank to every vertex so that visiting vertices by
    /// decreasing rank always picks the one with the most already visited
    /// neighbors. A graph is chordal iff any two higher-ranked neighbors of
    /// every vertex are adjacent. Edge directions are ignored.
    ///
    /// Mutual edges `u -> v`, `v -> u` of a directed graph count as a single
    /// undirected edge. igraph 1.0.1 can get this wrong when its property
    /// cache already records "no multi-edges" (a girth of 2, or heap
    /// corruption in the chordality code), so for directed graphs this
    /// wrapper clears the cache around the C call.
    ///
    /// Binds [`igraph_maximum_cardinality_search`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_maximum_cardinality_search).
    /// Time complexity: `O(|V| + |E|)`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false)?;
    /// let mcs = g.maximum_cardinality_search()?;
    /// for (v, &rank) in mcs.alpha.iter().enumerate() {
    ///     assert_eq!(mcs.alpham1[rank as usize], v as i64);
    /// }
    /// // The ranks can be reused by the chordality test.
    /// let c = g.is_chordal_with(Some(&mcs.alpha), Some(&mcs.alpham1))?;
    /// assert!(c.is_chordal);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn maximum_cardinality_search(&self) -> Result<CardinalitySearch> {
        let mut alpha = VectorInt::new();
        let mut alpham1 = VectorInt::new();
        with_multi_cache_guard(self, || {
            igraph_call!(igraph_maximum_cardinality_search(
                self,
                &mut alpha,
                &mut alpham1
            ))
        })?;
        Ok(CardinalitySearch {
            alpha: alpha.into(),
            alpham1: alpham1.into(),
        })
    }

    /// Whether the graph is chordal: every cycle of four or more vertices
    /// has a chord (equivalently, every induced cycle is a triangle).
    ///
    /// Edge directions are ignored. See [`is_chordal_with`](Self::is_chordal_with)
    /// to also get the fill-in.
    ///
    /// Mutual edges `u -> v`, `v -> u` of a directed graph count as a single
    /// undirected edge. igraph 1.0.1 can get this wrong when its property
    /// cache already records "no multi-edges" (a girth of 2, or heap
    /// corruption in the chordality code), so for directed graphs this
    /// wrapper clears the cache around the C call.
    ///
    /// Binds [`igraph_is_chordal`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_is_chordal).
    /// Time complexity: linear.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let square = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], 4, false)?;
    /// assert!(!square.is_chordal()?);
    /// let c = square.is_chordal_with(None, None)?;
    /// assert_eq!(c.fill_in.len(), 1); // one diagonal suffices
    /// assert!(c.triangulated.is_chordal()?);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn is_chordal(&self) -> Result<bool> {
        let mut res = false;
        with_multi_cache_guard(self, || {
            igraph_call!(igraph_is_chordal(
                self,
                ptr::null(),
                ptr::null(),
                &mut res,
                ptr::null_mut(),
                ptr::null_mut()
            ))
        })?;
        Ok(res)
    }

    /// Chordality test that also returns the fill-in (chordal completion)
    /// and the triangulated graph.
    ///
    /// `alpha` and/or `alpham1` may be supplied from a previous
    /// [`maximum_cardinality_search`](Self::maximum_cardinality_search) on
    /// the same graph (if only one is given, the other is its inverse);
    /// with `None`, the search is performed internally. The fill-in is not
    /// necessarily minimal.
    ///
    /// Binds [`igraph_is_chordal`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_is_chordal).
    ///
    /// igraph 1.0.0 and 1.0.1 only check the lengths of `alpha` and `alpham1`
    /// and then index with their values: this wrapper also checks that they
    /// are permutations, so that bad input can never cause out-of-bounds reads.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `alpha`
    /// or `alpham1` is not a permutation of `0..vcount`, or if both are given
    /// and they are not inverse of each other.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A 5-cycle needs two chords to become chordal.
    /// let c5 = Graph::ring(5, false, false, true)?;
    /// let c = c5.is_chordal_with(None, None)?;
    /// assert!(!c.is_chordal);
    /// assert_eq!(c.fill_in.len(), 2);
    /// assert_eq!(c.triangulated.ecount(), 7);
    /// assert!(c.triangulated.is_chordal()?);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn is_chordal_with(
        &self,
        alpha: Option<&[i64]>,
        alpham1: Option<&[VertexId]>,
    ) -> Result<Chordality> {
        // igraph 1.0.0 and 1.0.1 only check the lengths and then index with
        // the values: validate them here so that bad input can never read
        // out of bounds.
        let n = self.vcount();
        for (name, perm) in [("alpha", alpha), ("alpham1", alpham1)] {
            if let Some(p) = perm
                && !is_permutation(p, n)
            {
                return Err(Error::invalid(format!(
                    "{name} must be a permutation of 0..{n}"
                )));
            }
        }
        if let (Some(a), Some(am1)) = (alpha, alpham1)
            && a.iter()
                .enumerate()
                .any(|(v, &r)| am1[r as usize] != v as i64)
        {
            return Err(Error::invalid("alpham1 must be the inverse of alpha"));
        }
        let a = alpha.map(VectorInt::view);
        let am1 = alpham1.map(VectorInt::view);
        let ap = a.as_ref().map_or(ptr::null(), |v| v.as_ptr());
        let am1p = am1.as_ref().map_or(ptr::null(), |v| v.as_ptr());
        let mut chordal = false;
        let mut fill_in = VectorInt::new();
        let triangulated = with_multi_cache_guard(self, || {
            Graph::init_with(|t| unsafe {
                igraph_is_chordal(self, ap, am1p, &mut chordal, &mut fill_in, t)
            })
        })?;
        Ok(Chordality {
            is_chordal: chordal,
            fill_in: fill_in
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&[a, b]| (a, b))
                .collect(),
            triangulated,
        })
    }

    /// Average degree of the neighbors of each selected vertex (`knn`), and
    /// its average as a function of the vertex degree (`knnk`).
    ///
    /// `mode` chooses which neighbors are considered and
    /// `neighbor_degree_mode` which of their degrees is averaged (both
    /// ignored for undirected graphs). With `weights`, the weighted average
    /// `k_nn,u = 1/s_u Σ_v w_uv k_v` of Barrat et al. (PNAS 2004) is computed,
    /// `s_u` being the strength of `u`, and `knnk` averages `knn` weighted by
    /// the strengths. Isolated vertices (with weights: vertices of zero
    /// strength) get NaN, as do degrees that don't occur in `knnk`.
    ///
    /// Binds [`igraph_avg_nearest_neighbor_degree`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_avg_nearest_neighbor_degree).
    /// Time complexity: `O(|V| + |E|)`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // In a star, the center's neighbors have degree 1, the leaves' neighbor degree 3.
    /// let star = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], 4, false)?;
    /// let nd = star.avg_nearest_neighbor_degree(.., NeighborMode::All, NeighborMode::All, None)?;
    /// assert_eq!(nd.knn, vec![1.0, 3.0, 3.0, 3.0]);
    /// assert_eq!(nd.knnk[0], 3.0); // degree-1 vertices
    /// assert_eq!(nd.knnk[2], 1.0); // degree-3 vertices
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// See also [`degree_correlation_vector`](Self::degree_correlation_vector)
    /// (the same `k_nn(k)`, averaged over edges, with more mode choices) and
    /// [`Graph::assortativity_degree`], which summarizes degree correlations
    /// in a single number: a decreasing `k_nn(k)` goes with a negative
    /// assortativity.
    pub fn avg_nearest_neighbor_degree<'a>(
        &self,
        vertices: impl Into<VertexSelector<'a>>,
        mode: NeighborMode,
        neighbor_degree_mode: NeighborMode,
        weights: Option<&[f64]>,
    ) -> Result<NeighborDegree> {
        check_weights(self, weights)?;
        let vs = vertices.into().to_raw()?;
        let w = weights.map(Vector::view);
        let mut knn = Vector::new();
        let mut knnk = Vector::new();
        igraph_call!(igraph_avg_nearest_neighbor_degree(
            self,
            vs.get(),
            mode.into(),
            neighbor_degree_mode.into(),
            &mut knn,
            &mut knnk,
            weights_ptr(&w)
        ))?;
        Ok(NeighborDegree {
            knn: knn.into(),
            knnk: knnk.into(),
        })
    }

    /// The degree correlation function `k_nn(k)`: `result[k]` is the mean
    /// degree of the targets of edges whose source has degree `k` (NaN for
    /// degrees that don't occur). Unlike
    /// [`avg_nearest_neighbor_degree`](Self::avg_nearest_neighbor_degree),
    /// index 0 is for degree 0.
    ///
    /// The average is over all directed edges; undirected edges count as two
    /// reciprocal directed ones. `from_mode` and `to_mode` define the degree
    /// of sources and targets (out-in, out-out, in-in, in-out correlations).
    /// With `directed_neighbors = false`, directed edges are also treated as
    /// reciprocal pairs. With `weights`, weighted averages are computed.
    ///
    /// Binds [`igraph_degree_correlation_vector`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_degree_correlation_vector).
    /// Time complexity: `O(|V| + |E|)`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let star = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], 4, false)?;
    /// let knnk = star.degree_correlation_vector(None, NeighborMode::All, NeighborMode::All, true)?;
    /// assert_eq!(knnk[1], 3.0);
    /// assert_eq!(knnk[3], 1.0);
    /// assert!(knnk[0].is_nan() && knnk[2].is_nan());
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// See also [`Graph::joint_degree_matrix`], from which `k_nn(k)` can be
    /// derived, and [`Graph::assortativity_degree`].
    pub fn degree_correlation_vector(
        &self,
        weights: Option<&[f64]>,
        from_mode: NeighborMode,
        to_mode: NeighborMode,
        directed_neighbors: bool,
    ) -> Result<Vec<f64>> {
        check_weights(self, weights)?;
        let w = weights.map(Vector::view);
        let mut res = Vector::new();
        igraph_call!(igraph_degree_correlation_vector(
            self,
            weights_ptr(&w),
            &mut res,
            from_mode.into(),
            to_mode.into(),
            directed_neighbors
        ))?;
        Ok(res.into())
    }

    /// Density of the subgraphs left after removing vertices one by one in
    /// the given order (the *rich-club* sequence).
    ///
    /// `result[i]` is the density of the graph remaining after the first
    /// `i` vertices of `vertex_order` have been removed (so `result[0]` is the
    /// density of the whole graph). With `normalized = false` the remaining
    /// edge count (or total weight) is returned instead. `loops` decides
    /// whether self-loops are possible when computing the maximum number of
    /// edges (see [`density`](Self::density)); `directed = false` treats a
    /// directed graph as undirected. Removing vertices by increasing degree
    /// reveals whether high-degree vertices are densely interconnected.
    ///
    /// `loops` is ignored when `normalized` is `false`. If `loops` is `false`
    /// but the graph has self-loops, igraph emits a warning and still divides
    /// by the loop-free maximum, so densities may exceed 1.
    ///
    /// This function is marked *experimental* in igraph 1.0.0 and 1.0.1.
    ///
    /// Binds [`igraph_rich_club_sequence`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_rich_club_sequence).
    /// Time complexity: `O(|V| + |E|)`.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `vertex_order` is not a permutation of the vertex ids (wrong length,
    /// out-of-range or repeated entries) or the weights have the wrong length.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A triangle with a pendant vertex 3 attached to 0; peel off the pendant first.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (0, 3)], 4, false)?;
    /// let seq = g.rich_club_sequence(None, &[3, 0, 1, 2], true, false, false)?;
    /// assert!((seq[0] - 4.0 / 6.0).abs() < 1e-12);
    /// assert_eq!(seq[1], 1.0); // the triangle is a clique
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// A natural order removes vertices by increasing degree, see
    /// [`sort_vertex_ids_by_degree`](Self::sort_vertex_ids_by_degree).
    pub fn rich_club_sequence(
        &self,
        weights: Option<&[f64]>,
        vertex_order: &[VertexId],
        normalized: bool,
        loops: bool,
        directed: bool,
    ) -> Result<Vec<f64>> {
        check_weights(self, weights)?;
        let w = weights.map(Vector::view);
        let order = VectorInt::view(vertex_order);
        let mut res = Vector::new();
        igraph_call!(igraph_rich_club_sequence(
            self,
            weights_ptr(&w),
            &mut res,
            order.as_ptr(),
            normalized,
            loops,
            directed
        ))?;
        Ok(res.into())
    }

    // ------------------------------------------------------------------
    // Spectral properties
    // ------------------------------------------------------------------

    /// The (dense) Laplacian matrix of the graph.
    ///
    /// `L_ij = -A_ij` for `i ≠ j` and `L_ii = d_i - A_ii`, where `A` is the
    /// (possibly weighted) adjacency matrix and `d_i` the degree (strength).
    /// In directed graphs `mode` selects out-degrees (rows sum to zero),
    /// in-degrees (columns sum to zero), or ignores directions
    /// ([`NeighborMode::All`]). In undirected graphs `A_ii` is twice the
    /// number (weight) of self-loops. See [`LaplacianNormalization`] for the
    /// normalized variants. Weights must be non-negative and not NaN.
    ///
    /// For undirected graphs the unnormalized Laplacian is symmetric and
    /// positive semi-definite. Without weights (or with positive weights) the
    /// multiplicity of its zero eigenvalue is the number of connected
    /// components, and, without weights, (matrix-tree theorem) the number of
    /// spanning trees of a connected graph is the product of the non-zero
    /// eigenvalues divided by `|V|`.
    ///
    /// Binds [`igraph_get_laplacian`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_laplacian).
    /// Time complexity: `O(|V|²)`.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for
    /// negative or NaN weights, a wrong number of weights, or a normalization
    /// that would divide by a zero degree of a non-isolated vertex (e.g.
    /// [`LaplacianNormalization::Symmetric`] with [`NeighborMode::Out`] and a
    /// vertex with in-edges only).
    ///
    /// # Examples
    /// ```
    /// use igraph::{prelude::*, structural::LaplacianNormalization};
    /// let path = Graph::from_edges(&[(0, 1), (1, 2)], 3, false)?;
    /// let l = path.get_laplacian(NeighborMode::All, LaplacianNormalization::Unnormalized, None)?;
    /// assert_eq!(l.to_rows(), vec![
    ///     vec![1.0, -1.0, 0.0],
    ///     vec![-1.0, 2.0, -1.0],
    ///     vec![0.0, -1.0, 1.0],
    /// ]);
    ///
    /// // Matrix-tree theorem: K4 has 4^(4-2) = 16 spanning trees.
    /// use igraph::linalg::{lapack_dsyevr, SymmetricRange};
    /// let k4 = Graph::full(4, false, false)?;
    /// let l = k4.get_laplacian(NeighborMode::All, LaplacianNormalization::Unnormalized, None)?;
    /// let eigen = lapack_dsyevr(&l, &SymmetricRange::All, 1e-12)?;
    /// let trees: f64 = eigen.values[1..].iter().product::<f64>() / 4.0;
    /// assert!((trees - 16.0).abs() < 1e-9);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    ///
    /// See also [`get_laplacian_sparse`](Self::get_laplacian_sparse),
    /// [`Graph::get_adjacency`], and [`Graph::laplacian_spectral_embedding`].
    pub fn get_laplacian(
        &self,
        mode: NeighborMode,
        normalization: LaplacianNormalization,
        weights: Option<&[f64]>,
    ) -> Result<Matrix> {
        check_weights(self, weights)?;
        let w = weights.map(Vector::view);
        let mut res = Matrix::new();
        igraph_call!(igraph_get_laplacian(
            self,
            &mut res,
            mode.into(),
            normalization.into(),
            weights_ptr(&w)
        ))?;
        Ok(res)
    }

    /// The Laplacian matrix in sparse form, as sorted `(row, column, value)`
    /// triplets of its non-zero entries.
    ///
    /// Same definition as [`get_laplacian`](Self::get_laplacian); it takes
    /// `O(|V| + |E|)` time and memory, which makes it suitable for large
    /// sparse graphs. Duplicate entries produced by igraph (e.g. for
    /// multi-edges) are summed, and entries that sum to exactly zero are
    /// dropped. Use [`get_laplacian_sparsemat`](Self::get_laplacian_sparsemat)
    /// to get a [`SparseMat`] for the sparse linear algebra of
    /// [`linalg`](crate::linalg) instead.
    ///
    /// For directed graphs with [`NeighborMode::All`], the graph is treated
    /// as undirected, exactly like the dense variant does (the sparse C
    /// implementation of igraph 1.0.0 and 1.0.1 alone would not symmetrize
    /// the matrix).
    ///
    /// Binds [`igraph_get_laplacian_sparse`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_laplacian_sparse).
    ///
    /// # Errors
    /// As for [`get_laplacian`](Self::get_laplacian).
    ///
    /// # Examples
    /// ```
    /// use igraph::{prelude::*, structural::LaplacianNormalization};
    /// let g = Graph::from_edges(&[(0, 1)], 3, false)?;
    /// let l = g.get_laplacian_sparse(NeighborMode::All, LaplacianNormalization::Unnormalized, None)?;
    /// assert_eq!(l, vec![(0, 0, 1.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 1.0)]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn get_laplacian_sparse(
        &self,
        mode: NeighborMode,
        normalization: LaplacianNormalization,
        weights: Option<&[f64]>,
    ) -> Result<Vec<(VertexId, VertexId, f64)>> {
        let sparse = self.get_laplacian_sparsemat(mode, normalization, weights)?;
        if !sparse.is_triplet() {
            // igraph builds the matrix in triplet form; be defensive anyway.
            return Err(Error::new(
                crate::ErrorKind::Internal,
                "expected a sparse Laplacian in triplet form",
            ));
        }
        let el = sparse.getelements()?;
        let mut entries: BTreeMap<(VertexId, VertexId), f64> = BTreeMap::new();
        for ((&i, &j), &x) in el.i.iter().zip(&el.j).zip(&el.x) {
            *entries.entry((i, j)).or_insert(0.0) += x;
        }
        Ok(entries
            .into_iter()
            .filter(|&(_, x)| x != 0.0)
            .map(|((i, j), x)| (i, j, x))
            .collect())
    }

    /// The Laplacian matrix as a [`SparseMat`] (in triplet form, possibly
    /// with duplicate entries, which sparse operations sum up).
    ///
    /// Same definition, and same handling of directed graphs with
    /// [`NeighborMode::All`], as [`get_laplacian_sparse`](Self::get_laplacian_sparse).
    /// The result can be fed to the sparse linear algebra of
    /// [`linalg`](crate::linalg), e.g. [`SparseMat::mul_vec`] or
    /// [`SparseMat::to_dense`].
    ///
    /// Binds [`igraph_get_laplacian_sparse`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_laplacian_sparse).
    ///
    /// # Errors
    /// As for [`get_laplacian`](Self::get_laplacian).
    ///
    /// # Examples
    /// ```
    /// use igraph::{prelude::*, structural::LaplacianNormalization};
    /// let path = Graph::ring(4, false, false, false)?;
    /// let l = path.get_laplacian_sparsemat(NeighborMode::All, LaplacianNormalization::Unnormalized, None)?;
    /// // The Laplacian annihilates constant vectors...
    /// assert_eq!(l.mul_vec(&[1.0; 4])?, vec![0.0; 4]);
    /// // ...and x^T L x is the sum of (x_i - x_j)^2 over the edges.
    /// let x = [0.0, 1.0, 3.0, 6.0];
    /// let lx = l.mul_vec(&x)?;
    /// let quad: f64 = x.iter().zip(&lx).map(|(a, b)| a * b).sum();
    /// assert_eq!(quad, 1.0 + 4.0 + 9.0);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn get_laplacian_sparsemat(
        &self,
        mode: NeighborMode,
        normalization: LaplacianNormalization,
        weights: Option<&[f64]>,
    ) -> Result<SparseMat> {
        check_weights(self, weights)?;
        // igraph 1.0.0 and 1.0.1's sparse variant does not symmetrize the
        // off-diagonal entries of directed graphs with `IGRAPH_ALL` (unlike
        // the dense one): ignore directions explicitly, keeping edge ids
        // (and weights).
        let undirected;
        let graph = if self.is_directed() && mode == NeighborMode::All {
            undirected = Graph::from_edges(&self.edge_list(), self.vcount(), false)?;
            &undirected
        } else {
            self
        };
        let w = weights.map(Vector::view);
        let n = self.vcount();
        let mut sparse = SparseMat::new(n, n)?;
        igraph_call!(igraph_get_laplacian_sparse(
            graph,
            &mut sparse,
            mode.into(),
            normalization.into(),
            weights_ptr(&w)
        ))?;
        Ok(sparse)
    }
}

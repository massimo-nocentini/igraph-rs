//! Clustering, degree correlations and graphicality: transitivity,
//! assortativity, mixing matrices and degree-sequence realizability.
//!
//! This module binds three igraph headers:
//!
//! * `igraph_transitivity.h` — *how clustered is a graph?* The global
//!   transitivity (fraction of closed connected triples), the local clustering
//!   coefficient of Watts and Strogatz, its average, Barrat's weighted variant
//!   and the edge clustering coefficient of Radicchi *et al.*;
//! * `igraph_mixing.h` — *who connects to whom?* Newman's assortativity
//!   coefficients (for numeric values, for categories and for degrees), the
//!   joint degree matrix, the joint degree distribution and the mixing matrix
//!   of vertex categories;
//! * `igraph_graphicality.h` — *can a degree sequence be realized?* The
//!   Erdős–Gallai / Fulkerson–Chen–Anstee / Gale–Ryser tests, extended to
//!   graphs with self-loops and/or multi-edges.
//!
//! Functions that take a graph are methods of [`Graph`]; the graphicality tests
//! work on plain degree slices and are free functions.
//!
//! | Rust | C function | Computes |
//! |------|------------|----------|
//! | [`Graph::transitivity_undirected`] | `igraph_transitivity_undirected` | global clustering coefficient |
//! | [`Graph::transitivity_local_undirected`] | `igraph_transitivity_local_undirected` | local clustering coefficient of vertices |
//! | [`Graph::transitivity_avglocal_undirected`] | `igraph_transitivity_avglocal_undirected` | average local clustering coefficient |
//! | [`Graph::transitivity_barrat`] | `igraph_transitivity_barrat` | Barrat's weighted local clustering |
//! | [`Graph::ecc`] | `igraph_ecc` | edge clustering coefficient (3- and 4-cycles) |
//! | [`Graph::assortativity`] | `igraph_assortativity` | assortativity by numeric vertex values |
//! | [`Graph::assortativity_nominal`] | `igraph_assortativity_nominal` | assortativity by vertex categories |
//! | [`Graph::assortativity_degree`] | `igraph_assortativity_degree` | degree assortativity |
//! | [`Graph::joint_degree_matrix`] | `igraph_joint_degree_matrix` | edge counts between degree classes |
//! | [`Graph::joint_degree_distribution`] | `igraph_joint_degree_distribution` | joint degree distribution `P_ij` |
//! | [`Graph::joint_type_distribution`] | `igraph_joint_type_distribution` | mixing matrix of vertex categories |
//! | [`is_graphical`] | `igraph_is_graphical` | is a (bi-)degree sequence realizable? |
//! | [`is_bigraphical`] | `igraph_is_bigraphical` | is a pair of sequences realizable as a bipartite graph? |
//!
//! Which kinds of edges the graphicality tests may use is described by
//! [`AllowedEdgeTypes`] (defined in [`constants`](crate::constants) and shared
//! with the [`games`](crate::games) and [`constructors`](crate::constructors)
//! modules), which also converts from [`EdgeTypeSw`].
//!
//! All functions of this module are deterministic.
//!
//! # See also
//!
//! * Triangles themselves: [`Graph::count_triangles`],
//!   [`Graph::count_adjacent_triangles`] and [`Graph::list_triangles`]
//!   (the global transitivity is `3 × triangles / connected triples`), the
//!   [triad census](Graph::triad_census) and [motifs](Graph::motifs_randesu).
//! * Degree correlations as functions of the degree: the average nearest
//!   neighbor degree [`Graph::avg_nearest_neighbor_degree`] and
//!   [`Graph::degree_correlation_vector`] (`k_nn(k)`, derivable from
//!   [`Graph::joint_degree_distribution`]); vertex [strengths](Graph::strength)
//!   for weighted degrees; the [rich-club coefficient](Graph::rich_club_sequence).
//! * Categories: the unnormalized nominal assortativity is the
//!   [modularity](Graph::modularity) of the partition, and community detection
//!   (e.g. [`Graph::community_multilevel`]) finds partitions with high values.
//! * Degree sequences: build a graph from a graphical sequence with
//!   [`Graph::realize_degree_sequence`] /
//!   [`Graph::realize_bipartite_degree_sequence`] (deterministic) or
//!   [`Graph::degree_sequence_game`] / [`Graph::k_regular_game`] (random);
//!   randomize a graph while keeping its degrees with [`Graph::rewire`], the
//!   usual null model against which clustering and assortativity are compared.
//!
//! # Example
//!
//! ```
//! use igraph::mixing::{is_graphical, AllowedEdgeTypes};
//! use igraph::prelude::*;
//!
//! // A "bow tie": two triangles sharing vertex 2.
//! let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)], 5, false)?;
//!
//! // 2 triangles, 3 * 2 = 6 closed triples out of 10 connected triples.
//! let t = g.transitivity_undirected(TransitivityMode::Nan)?;
//! assert!((t - 0.6).abs() < 1e-12);
//! assert_eq!(g.count_triangles()?, 2.0);
//!
//! // The hub closes 2 out of the 6 pairs of its neighbors, the others all theirs.
//! let local = g.transitivity_local_undirected(.., TransitivityMode::Zero)?;
//! assert!((local[2] - 1.0 / 3.0).abs() < 1e-12);
//! assert_eq!(local[0], 1.0);
//!
//! // High-degree hub attached to low-degree vertices: disassortative.
//! assert!(g.assortativity_degree(false)? < 0.0);
//!
//! // Its degree sequence is, of course, graphical.
//! let degrees = g.degree(.., NeighborMode::All, Loops::Twice)?;
//! assert!(is_graphical(&degrees, None, AllowedEdgeTypes::SIMPLE)?);
//! // ...while an odd degree sum never is.
//! assert!(!is_graphical(&[3, 3, 3], None, AllowedEdgeTypes::ALL)?);
//!
//! // Zachary's karate club: clustered, and its hubs avoid each other.
//! let karate = Graph::famous("Zachary")?;
//! let c = karate.transitivity_undirected(TransitivityMode::Nan)?;
//! assert!((c - 0.2556818).abs() < 1e-6);
//! assert!((karate.assortativity_degree(false)? + 0.475613).abs() < 1e-6);
//! # Ok::<(), igraph::Error>(())
//! ```

use crate::{
    constants::*,
    error::{Error, Result},
    ffi::*,
    graph::Graph,
    igraph_call,
    matrix::Matrix,
    selector::{EdgeSelector, VertexSelector},
    vector::{Vector, VectorInt},
};

/// Which kinds of edges a realization of a degree sequence may contain
/// (`igraph_edge_type_sw_t`), used by [`is_graphical`] and [`is_bigraphical`].
///
/// Re-exported from [`constants`](crate::constants::AllowedEdgeTypes): the
/// same type is also available as `games::AllowedEdgeTypes` and
/// `constructors::AllowedEdgeTypes`, so a graphicality test and the generator
/// realizing the sequence (e.g. [`Graph::realize_degree_sequence`]) can share
/// the same value.
pub use crate::constants::AllowedEdgeTypes;

/// Options of [`Graph::joint_degree_distribution`].
///
/// The defaults describe the classical joint degree distribution: for each
/// directed connection `u -> v`, the out-degree of `u` and the in-degree of
/// `v`, normalized so that the entries sum to one, with automatically sized
/// rows and columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JointDegreeDistributionOptions {
    /// How to compute the degree of source vertices ([`NeighborMode::Out`],
    /// [`NeighborMode::In`] or [`NeighborMode::All`]). Ignored for undirected
    /// graphs. Default: [`NeighborMode::Out`].
    pub from_mode: NeighborMode,
    /// How to compute the degree of target vertices. Ignored for undirected
    /// graphs. Default: [`NeighborMode::In`].
    pub to_mode: NeighborMode,
    /// Whether to consider `u -> v` connections as directed. When `false`,
    /// every edge is counted in both directions. Ignored for undirected
    /// graphs. Default: `true`.
    pub directed_neighbors: bool,
    /// Whether to normalize the matrix so that its entries sum to 1. If
    /// `false`, entries are connection counts (or total weights).
    /// Default: `true`.
    pub normalized: bool,
    /// Largest source degree to consider (the result has one more row);
    /// `None` uses the largest source degree in the graph. Default: `None`.
    pub max_from_degree: Option<usize>,
    /// Largest target degree to consider (the result has one more column);
    /// `None` uses the largest target degree in the graph. Default: `None`.
    pub max_to_degree: Option<usize>,
}

impl Default for JointDegreeDistributionOptions {
    fn default() -> Self {
        Self {
            from_mode: NeighborMode::Out,
            to_mode: NeighborMode::In,
            directed_neighbors: true,
            normalized: true,
            max_from_degree: None,
            max_to_degree: None,
        }
    }
}

impl JointDegreeDistributionOptions {
    /// Sets how source and target degrees are computed.
    pub fn with_modes(mut self, from_mode: NeighborMode, to_mode: NeighborMode) -> Self {
        self.from_mode = from_mode;
        self.to_mode = to_mode;
        self
    }

    /// Sets [`directed_neighbors`](Self::directed_neighbors).
    pub fn with_directed_neighbors(mut self, directed_neighbors: bool) -> Self {
        self.directed_neighbors = directed_neighbors;
        self
    }

    /// Sets [`normalized`](Self::normalized).
    pub fn with_normalized(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }

    /// Sets the largest source and target degrees to consider.
    pub fn with_max_degrees(mut self, max_from: Option<usize>, max_to: Option<usize>) -> Self {
        self.max_from_degree = max_from;
        self.max_to_degree = max_to;
        self
    }
}

/// Converts an optional limit to igraph's convention (negative = unlimited).
///
/// Limits that do not fit an `igraph_int_t` (or whose successor, the matrix
/// dimension, would overflow) are rejected: a plain cast would turn them into
/// negative numbers, silently meaning "unlimited", and `i64::MAX + 1` is
/// signed overflow (undefined behavior) in the C code.
fn limit(max: Option<usize>) -> Result<igraph_int_t> {
    match max {
        None => Ok(IGRAPH_UNLIMITED as igraph_int_t),
        Some(m) if m < igraph_int_t::MAX as usize => Ok(m as igraph_int_t),
        Some(m) => Err(Error::invalid(format!("degree limit {m} is too large"))),
    }
}

/// Checks that vertex types are valid category indices: non-negative, and
/// small enough for `type + 1` (a matrix dimension) not to overflow.
///
/// igraph 1.0.0 and 1.0.1 do not check the *target* types of
/// `igraph_joint_type_distribution` (`mixing_matrix()` in `misc/mixing.c`
/// validates `from_types` twice), so a negative target type would make it
/// write before the start of the matrix: this check must run before every
/// such call.
fn check_types(what: &str, types: &[igraph_int_t]) -> Result<()> {
    match types
        .iter()
        .find(|&&t| !(0..igraph_int_t::MAX).contains(&t))
    {
        Some(t) => Err(Error::invalid(format!(
            "invalid {what} value {t}: vertex types must be non-negative (and less than i64::MAX)"
        ))),
        None => Ok(()),
    }
}

/// Pointer to an optional real vector view, or null.
fn opt_ptr<V>(v: &Option<crate::vector::View<'_, V>>) -> *const V {
    v.as_ref().map_or(std::ptr::null(), |v| v.as_ptr())
}

/// Number of rows/columns of a mixing matrix: `max + 1` if given, otherwise
/// one more than the largest type (0 when there are no vertices).
fn dimension(max: Option<usize>, types: &[igraph_int_t]) -> usize {
    match max {
        Some(m) => m + 1,
        None => types.iter().max().map_or(0, |&t| t.max(-1) as usize + 1),
    }
}

impl Graph {
    /// Rust implementation of igraph's (static) `mixing_matrix`, used for the
    /// cases in which the C code of igraph 1.0.0 and 1.0.1 would write out of
    /// bounds.
    ///
    /// Every edge `u -> v` adds its weight to entry
    /// `(from_types[u], to_types[v])` and, unless `directed_neighbors`, the
    /// reverse connection `v -> u` adds it to `(from_types[v], to_types[u])`;
    /// connections falling outside the `dims` matrix are ignored. Types must be
    /// non-negative and of the right lengths (checked by the callers).
    fn mixing_matrix(
        &self,
        weights: Option<&[f64]>,
        from_types: &[igraph_int_t],
        to_types: &[igraph_int_t],
        directed_neighbors: bool,
        normalized: bool,
        (nrow, ncol): (usize, usize),
    ) -> Result<Matrix> {
        // Allocate through igraph so that huge (user-given) dimensions give an
        // error instead of aborting. Dimensions fit `igraph_int_t`: they are
        // one more than a validated limit or a non-negative type/degree.
        let mut p = Matrix::new();
        igraph_call!(igraph_matrix_resize(
            &mut p,
            nrow as igraph_int_t,
            ncol as igraph_int_t
        ))?;
        p.as_mut_slice().fill(0.0);
        let mut sum = 0.0;
        let mut add = |a: igraph_int_t, b: igraph_int_t, w: f64| {
            let (a, b) = (a as usize, b as usize);
            if a < nrow && b < ncol {
                p[(a, b)] += w;
                sum += w;
            }
        };
        for (eid, (u, v)) in self.edge_list().into_iter().enumerate() {
            let w = weights.map_or(1.0, |w| w[eid]);
            add(from_types[u as usize], to_types[v as usize], w);
            if !directed_neighbors {
                add(from_types[v as usize], to_types[u as usize], w);
            }
        }
        if normalized && self.ecount() > 0 {
            p.as_mut_slice().iter_mut().for_each(|x| *x /= sum);
        }
        Ok(p)
    }

    fn mixing_check_weights(&self, weights: Option<&[f64]>) -> Result<()> {
        match weights {
            Some(w) if w.len() != self.ecount() => Err(Error::invalid(format!(
                "weight vector length ({}) does not match the number of edges ({})",
                w.len(),
                self.ecount()
            ))),
            _ => Ok(()),
        }
    }

    fn check_vertex_values<T>(&self, what: &str, values: &[T]) -> Result<()> {
        if values.len() != self.vcount() {
            return Err(Error::invalid(format!(
                "{what} vector length ({}) does not match the number of vertices ({})",
                values.len(),
                self.vcount()
            )));
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // igraph_transitivity.h
    // ------------------------------------------------------------------

    /// Global transitivity (clustering coefficient) of the graph.
    ///
    /// The transitivity is the probability that two neighbors of a vertex are
    /// connected; more precisely, it is the ratio between the number of
    /// *closed* connected triples (three times the number of triangles) and
    /// the number of connected triples. Edge directions and multiplicities are
    /// ignored. This single number differs from the
    /// [average local transitivity](Self::transitivity_avglocal_undirected),
    /// which weights all vertices equally.
    ///
    /// `mode` says what to return for graphs without connected triples:
    /// [`TransitivityMode::Nan`] gives `NaN`, [`TransitivityMode::Zero`] gives 0.
    ///
    /// Binds [`igraph_transitivity_undirected`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_transitivity_undirected).
    /// Reference: S. Wasserman and K. Faust, *Social Network Analysis: Methods
    /// and Applications*, Cambridge University Press (1994).
    ///
    /// See also [`Graph::count_triangles`]: for a simple graph with degrees
    /// `d_v`, the transitivity is `3 T / Σ_v d_v (d_v - 1) / 2`.
    ///
    /// Time complexity: O(|V| d²), d being the average degree.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // A triangle with a pendant edge: 3 closed triples out of 5.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false)?;
    /// assert!((g.transitivity_undirected(TransitivityMode::Nan)? - 0.6).abs() < 1e-12);
    ///
    /// // A path has connected triples but no triangle; a single edge has neither.
    /// let path = Graph::from_edges(&[(0, 1), (1, 2)], 3, false)?;
    /// assert_eq!(path.transitivity_undirected(TransitivityMode::Nan)?, 0.0);
    /// let edge = Graph::from_edges(&[(0, 1)], 2, false)?;
    /// assert!(edge.transitivity_undirected(TransitivityMode::Nan)?.is_nan());
    /// assert_eq!(edge.transitivity_undirected(TransitivityMode::Zero)?, 0.0);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn transitivity_undirected(&self, mode: TransitivityMode) -> Result<f64> {
        let mut res = 0.0;
        self.with_fresh_multi_cache(|| {
            igraph_call!(igraph_transitivity_undirected(self, &mut res, mode.into()))
        })?;
        Ok(res)
    }

    /// Local transitivity (clustering coefficient) of the selected vertices.
    ///
    /// For each vertex, the fraction of pairs of its neighbors that are
    /// themselves connected (Watts–Strogatz clustering coefficient). Edge
    /// directions and multiplicities are ignored. Vertices with fewer than two
    /// neighbors get `NaN` with [`TransitivityMode::Nan`] and 0 with
    /// [`TransitivityMode::Zero`]. The result follows the order of `vids`.
    ///
    /// Binds [`igraph_transitivity_local_undirected`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_transitivity_local_undirected).
    /// Reference: D. J. Watts and S. Strogatz, *Collective dynamics of
    /// small-world networks*, Nature 393, 440–442 (1998).
    ///
    /// See also [`Graph::count_adjacent_triangles`]: in a simple graph the
    /// local transitivity of `v` is `t_v / (d_v (d_v - 1) / 2)`, `t_v` being
    /// the number of triangles through `v`.
    ///
    /// Time complexity: O(n d²), n being the number of selected vertices and d
    /// the average degree.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) if the
    /// selector contains a non-existent vertex.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false)?;
    /// let c = g.transitivity_local_undirected(.., TransitivityMode::Zero)?;
    /// assert_eq!(c[..2], [1.0, 1.0]);
    /// assert!((c[2] - 1.0 / 3.0).abs() < 1e-12);
    /// assert_eq!(c[3], 0.0); // a leaf
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn transitivity_local_undirected<'a>(
        &self,
        vids: impl Into<VertexSelector<'a>>,
        mode: TransitivityMode,
    ) -> Result<Vec<f64>> {
        let vs = vids.into().to_raw()?;
        let mut res = Vector::new();
        self.with_fresh_multi_cache(|| {
            igraph_call!(igraph_transitivity_local_undirected(
                self,
                &mut res,
                vs.get(),
                mode.into()
            ))
        })?;
        Ok(res.into())
    }

    /// Average local transitivity (average clustering coefficient).
    ///
    /// The mean of the [local transitivities](Self::transitivity_local_undirected)
    /// of all vertices. Vertices with fewer than two neighbors are left out of
    /// the average with [`TransitivityMode::Nan`] (the result is `NaN` if no
    /// vertex has two neighbors), and counted as zero with
    /// [`TransitivityMode::Zero`]. Edge directions and multiplicities are
    /// ignored.
    ///
    /// Binds [`igraph_transitivity_avglocal_undirected`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_transitivity_avglocal_undirected).
    /// Reference: D. J. Watts and S. Strogatz, *Collective dynamics of
    /// small-world networks*, Nature 393, 440–442 (1998). A small-world graph
    /// (e.g. [`Graph::watts_strogatz_game`] with a small rewiring probability)
    /// has a much higher average clustering than an
    /// [Erdős–Rényi graph](Graph::erdos_renyi_game_gnm) of the same density.
    ///
    /// Time complexity: O(|V| d²).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false)?;
    /// // Local values: 1, 1, 1/3 and (leaf) NaN or 0.
    /// let skip = g.transitivity_avglocal_undirected(TransitivityMode::Nan)?;
    /// let zero = g.transitivity_avglocal_undirected(TransitivityMode::Zero)?;
    /// assert!((skip - 7.0 / 9.0).abs() < 1e-12);
    /// assert!((zero - 7.0 / 12.0).abs() < 1e-12);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn transitivity_avglocal_undirected(&self, mode: TransitivityMode) -> Result<f64> {
        let mut res = 0.0;
        self.with_fresh_multi_cache(|| {
            igraph_call!(igraph_transitivity_avglocal_undirected(
                self,
                &mut res,
                mode.into()
            ))
        })?;
        Ok(res)
    }

    /// Barrat's weighted local transitivity of the selected vertices.
    ///
    /// For a vertex `i`, every triangle `i`, `j`, `h` contributes the total
    /// weight `w_ij + w_ih` of the two triangle edges incident on `i` (in
    /// equation (5) each triangle appears twice, as the ordered pairs `(j, h)`
    /// and `(h, j)`, with the mean weight `(w_ij + w_ih) / 2`); the sum is
    /// divided by `s_i (k_i - 1)`, where `s_i` is the strength and `k_i` the
    /// degree of `i` (equation (5) of A. Barrat, M. Barthélemy,
    /// R. Pastor-Satorras and A. Vespignani, *The architecture of complex
    /// weighted networks*, PNAS 101, 3747 (2004)). With equal weights it
    /// coincides with the [unweighted local transitivity](Self::transitivity_local_undirected).
    ///
    /// Edge directions are ignored; the graph must not have multi-edges (for
    /// directed graphs, not even mutual pairs `u -> v`, `v -> u`, which become
    /// multi-edges once directions are ignored). If `weights`
    /// is `None`, igraph emits a warning and falls back to the unweighted
    /// local transitivity. When the denominator `s_i (k_i - 1)` is zero
    /// (fewer than two incident edges, or zero strength),
    /// [`TransitivityMode::Zero`] gives 0, while [`TransitivityMode::Nan`]
    /// performs the division: `NaN` (`0 / 0`), or `±∞` if a zero strength
    /// comes from weights of mixed signs around closed triangles.
    ///
    /// Binds [`igraph_transitivity_barrat`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_transitivity_barrat).
    /// See also [`Graph::strength`] for the `s_i`.
    ///
    /// Time complexity: O(|V| d²).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the
    /// weight vector has the wrong length or the graph has multi-edges (or
    /// mutual directed edges);
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) for
    /// invalid vertices.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // igraph's unit test graph: two triangles 0-1-2 and 1-2-3, a tail 3-4, isolated 5.
    /// let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)], 6, false)?;
    /// let w = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    /// let t = g.transitivity_barrat(.., Some(&w), TransitivityMode::Zero)?;
    /// let expected = [1.0, 0.75, 0.625, 0.277778, 0.0, 0.0];
    /// for (a, b) in t.iter().zip(expected) {
    ///     assert!((a - b).abs() < 1e-6);
    /// }
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn transitivity_barrat<'a>(
        &self,
        vids: impl Into<VertexSelector<'a>>,
        weights: Option<&[f64]>,
        mode: TransitivityMode,
    ) -> Result<Vec<f64>> {
        self.mixing_check_weights(weights)?;
        let vs = vids.into().to_raw()?;
        let w = weights.map(Vector::view);
        let mut res = Vector::new();
        self.with_fresh_multi_cache(|| {
            igraph_call!(igraph_transitivity_barrat(
                self,
                &mut res,
                vs.get(),
                opt_ptr(&w),
                mode.into()
            ))
        })?;
        Ok(res.into())
    }

    /// Edge clustering coefficient of the selected edges.
    ///
    /// For an edge `(i, j)`, let `z` be the number of `k`-cycles it belongs
    /// to and `s` the largest such number compatible with the degrees of its
    /// endpoints: `s = min(d_i - 1, d_j - 1)` for `k = 3` and
    /// `s = (d_i - 1)(d_j - 1)` for `k = 4`. The coefficient is
    ///
    /// ```text
    /// C = (z + offset) / s      (normalize = true)
    /// C =  z + offset           (normalize = false)
    /// ```
    ///
    /// where `offset` is 1 if `offset` is `true` and 0 otherwise. The original
    /// definition of Radicchi *et al.* (PNAS 101, 2658 (2004)) uses
    /// `offset = true, normalize = true`; with `offset = false` the normalized
    /// value is at most 1, which for `k = 3` is achieved by every edge of a
    /// complete graph. When normalizing, edges with `s = 0` (an endpoint of
    /// degree 1, or a self-loop, which igraph assigns `z = s = 0`) get `NaN`
    /// without offset (`0 / 0`) and `+∞` with it (`1 / 0`). Multiplicities
    /// are ignored when listing cycles but not in the degrees. The result
    /// follows the order of `eids`.
    ///
    /// Only `k = 3` and `k = 4` are currently supported.
    ///
    /// Binds [`igraph_ecc`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_ecc).
    /// See also [`Graph::list_triangles`] (for `k = 3`, the unnormalized,
    /// offset-free coefficient of an edge is the number of listed triangles
    /// containing it) and [`Graph::community_edge_betweenness`], the other
    /// classic edge-removal criterion for divisive community detection.
    ///
    /// Time complexity: O(|V| d log d + |E| d) for `k = 3`,
    /// O(|V| d log d + |E| d²) for `k = 4`.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `k < 3`;
    /// [`ErrorKind::Unimplemented`](crate::ErrorKind::Unimplemented) if `k > 4`;
    /// [`ErrorKind::InvalidEdgeId`](crate::ErrorKind::InvalidEdgeId) for invalid edges.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // In K4 each edge is in 2 triangles, the most its degree-3 endpoints allow.
    /// let k4 = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], 4, false)?;
    /// assert!(k4.ecc(.., 3, false, true)?.iter().all(|&c| c == 1.0));
    /// assert_eq!(k4.ecc(0, 3, true, false)?, vec![3.0]); // 2 triangles, plus one
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn ecc<'a>(
        &self,
        eids: impl Into<EdgeSelector<'a>>,
        k: usize,
        offset: bool,
        normalize: bool,
    ) -> Result<Vec<f64>> {
        let es = eids.into().to_raw()?;
        let mut res = Vector::new();
        self.with_fresh_multi_cache(|| {
            igraph_call!(igraph_ecc(
                self,
                &mut res,
                es.get(),
                k as igraph_int_t,
                offset,
                normalize
            ))
        })?;
        Ok(res.into())
    }

    // ------------------------------------------------------------------
    // igraph_mixing.h
    // ------------------------------------------------------------------

    /// Assortativity coefficient based on numeric vertex values.
    ///
    /// With `normalized = true` this is the Pearson correlation of the values
    /// `x` found at the two ends of the edges (Newman's assortativity
    /// coefficient, in `[-1, 1]`); with `normalized = false` it is the
    /// covariance
    ///
    /// ```text
    /// cov(x_out, x_in) = 1/m Σ_ij (A_ij - k_i^out k_j^in / m) x_i x_j
    /// ```
    ///
    /// For directed graphs (with `directed = true`) the value of the edge
    /// source is taken from `values` and the one of the target from
    /// `values_in`, if given (otherwise from `values` as well). Undirected
    /// graphs (and directed ones with `directed = false`) are treated as
    /// directed graphs with every edge reciprocated, so self-loops count
    /// twice; in that case `values_in` is ignored, with a warning if given.
    /// `directed` is ignored for undirected graphs.
    ///
    /// When `weights` are given they act as edge multiplicities: `m` becomes
    /// the total weight and degrees become strengths.
    ///
    /// Binds [`igraph_assortativity`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_assortativity).
    /// See also [`Graph::assortativity_degree`] (values = degrees) and
    /// [`Graph::strength`] (weighted degrees as values).
    /// References: M. E. J. Newman, *Mixing patterns in networks*, Phys. Rev.
    /// E 67, 026126 (2003); *Assortative mixing in networks*, Phys. Rev. Lett.
    /// 89, 208701 (2002).
    ///
    /// Time complexity: O(|E|).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if a value
    /// or weight vector has the wrong length.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // A path 0-1-2-3 with values increasing along it: neighbors are alike.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false)?;
    /// let r = g.assortativity(None, &[1.0, 2.0, 3.0, 4.0], None, false, true)?;
    /// assert!(r > 0.0);
    /// // Alternating values: every edge joins a "low" and a "high" vertex.
    /// let r = g.assortativity(None, &[0.0, 1.0, 0.0, 1.0], None, false, true)?;
    /// assert!((r + 1.0).abs() < 1e-12);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn assortativity(
        &self,
        weights: Option<&[f64]>,
        values: &[f64],
        values_in: Option<&[f64]>,
        directed: bool,
        normalized: bool,
    ) -> Result<f64> {
        self.mixing_check_weights(weights)?;
        self.check_vertex_values("values", values)?;
        if let Some(vi) = values_in {
            self.check_vertex_values("values_in", vi)?;
        }
        let w = weights.map(Vector::view);
        let v = Vector::view(values);
        let vi = values_in.map(Vector::view);
        let mut res = 0.0;
        igraph_call!(igraph_assortativity(
            self,
            opt_ptr(&w),
            v.as_ptr(),
            opt_ptr(&vi),
            &mut res,
            directed,
            normalized
        ))?;
        Ok(res)
    }

    /// Assortativity coefficient based on vertex categories.
    ///
    /// `types[v]` is the (non-negative integer) category of vertex `v`. The
    /// normalized coefficient (`normalized = true`, the usual choice) is 1 when
    /// all edges stay within categories, -1 for a perfectly disassortative
    /// network, and asymptotically 0 for random connections. The unnormalized
    /// version equals the [modularity](Graph::modularity) of the partition
    /// into categories (with resolution 1):
    ///
    /// ```text
    /// Q = 1/m Σ_ij (A_ij - k_i^out k_j^in / m) δ(t_i, t_j)
    /// ```
    ///
    /// and the normalized one is `Q` divided by its largest possible value
    /// `1 - 1/m² Σ_ij k_i^out k_j^in δ(t_i, t_j)`, i.e. `Q / (1 - Σ_t a_t b_t)`
    /// with `a_t` (`b_t`) the fraction of edges starting (ending) in category
    /// `t`. (The C documentation of 1.0.1 writes this denominator as
    /// `1/m Σ_ij (m - k_i^out k_j^in δ(t_i, t_j) / m)`, which is not what the
    /// code computes.)
    ///
    /// `directed` says whether to consider edge directions (ignored for
    /// undirected graphs, which are treated as directed graphs with reciprocal
    /// edges, so self-loops count twice). The null graph gives `NaN`.
    ///
    /// Weighted nominal assortativity is not implemented by igraph (1.0.0 and
    /// 1.0.1 fail with `IGRAPH_UNIMPLEMENTED` when weights are given), so this
    /// wrapper takes no weights; use [`Graph::modularity`] for the weighted
    /// unnormalized value.
    ///
    /// Binds [`igraph_assortativity_nominal`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_assortativity_nominal).
    /// Reference: M. E. J. Newman, *Mixing patterns in networks*, Phys. Rev. E
    /// 67, 026126 (2003).
    ///
    /// Time complexity: O(|E| + t), t being the number of categories.
    ///
    /// See also [`Graph::joint_type_distribution`], the full mixing matrix of
    /// the categories.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `types`
    /// does not have one entry per vertex or contains negative values.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Two triangles joined by one bridge; categories = triangles.
    /// let g = Graph::from_edges(
    ///     &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)], 6, false)?;
    /// let r = g.assortativity_nominal(&[0, 0, 0, 1, 1, 1], false, true)?;
    /// assert!(r > 0.7);
    /// // A bipartite labeling of a bipartite graph is perfectly disassortative.
    /// let square = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], 4, false)?;
    /// let r = square.assortativity_nominal(&[0, 1, 0, 1], false, true)?;
    /// assert!((r + 1.0).abs() < 1e-12);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn assortativity_nominal(
        &self,
        types: &[igraph_int_t],
        directed: bool,
        normalized: bool,
    ) -> Result<f64> {
        self.check_vertex_values("types", types)?;
        check_types("types", types)?;
        let t = VectorInt::view(types);
        let mut res = 0.0;
        igraph_call!(igraph_assortativity_nominal(
            self,
            std::ptr::null(),
            t.as_ptr(),
            &mut res,
            directed,
            normalized
        ))?;
        Ok(res)
    }

    /// Degree assortativity: do high-degree vertices link to each other?
    ///
    /// The [assortativity](Self::assortativity) coefficient with the vertex
    /// degrees as values, normalized (Pearson correlation of the degrees at
    /// the two ends of the edges). With `directed = true` on a directed graph,
    /// out-degrees are used for edge sources and in-degrees for edge targets;
    /// otherwise total degrees are used. Social networks tend to be
    /// assortative (> 0), technological and biological ones disassortative
    /// (< 0). For regular graphs the correlation is undefined and the result
    /// is `NaN`.
    ///
    /// Binds [`igraph_assortativity_degree`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_assortativity_degree).
    /// Loops count twice in the degrees and multi-edges are counted with
    /// their multiplicity; the unnormalized covariance can be obtained with
    /// [`Graph::assortativity`] and [`Graph::strength`] values.
    ///
    /// See also [`Graph::avg_nearest_neighbor_degree`] and
    /// [`Graph::degree_correlation_vector`], which show the degree correlation
    /// as a function of the degree instead of summarizing it in one number.
    ///
    /// Time complexity: O(|E| + |V|).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // In a star, the hub is only linked to leaves: perfectly disassortative.
    /// let star = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (0, 4)], 5, false)?;
    /// assert!((star.assortativity_degree(false)? + 1.0).abs() < 1e-12);
    /// // A cycle is regular: the coefficient is undefined.
    /// let c = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, false)?;
    /// assert!(c.assortativity_degree(false)?.is_nan());
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn assortativity_degree(&self, directed: bool) -> Result<f64> {
        let mut res = 0.0;
        igraph_call!(igraph_assortativity_degree(self, &mut res, directed))?;
        Ok(res)
    }

    /// Joint degree matrix: number (or total weight) of edges between degree classes.
    ///
    /// Entry `(i - 1, j - 1)` of the result holds `J_ij`, the number of edges
    /// (or the total weight, if `weights` are given) between vertices of
    /// (out-)degree `i` and vertices of (in-)degree `j`. Each edge, self-loops
    /// included, is counted exactly once: for ordered degree pairs `(i, j)` in
    /// directed graphs, whose entries then sum to the number of edges `m` (or
    /// total weight), and for unordered pairs in undirected graphs, whose
    /// matrix is symmetric and whose upper triangle (diagonal included) sums
    /// to `m` (without limits; with limits, only the part that fits).
    /// `J_ij / m` is the probability that a random edge joins degrees `i` and
    /// `j`.
    ///
    /// `max_out_degree` / `max_in_degree` set the number of rows / columns;
    /// `None` uses the largest (out-/in-)degree of the graph. Edges whose
    /// degree pair falls outside the matrix are not counted. Unlike
    /// [`joint_degree_distribution`](Self::joint_degree_distribution), there
    /// is no row or column for degree zero, and undirected same-degree
    /// connections are counted once instead of twice.
    ///
    /// Binds [`igraph_joint_degree_matrix`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_joint_degree_matrix).
    /// It is a finer description of a network than its degree sequence:
    /// degree-preserving [rewiring](Graph::rewire) keeps the degrees but in
    /// general changes this matrix. See also
    /// [`joint_degree_distribution`](Self::joint_degree_distribution).
    /// Reference: I. Stanton and A. Pinar, *Constructing and sampling graphs
    /// with a prescribed joint degree distribution*, ACM J. Exp. Algorithmics
    /// 17, 3.5 (2012).
    ///
    /// Time complexity: O(|E|).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the
    /// weight vector has the wrong length or a limit does not fit an `i64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // A star with 3 leaves: 3 edges between degree 3 and degree 1.
    /// let star = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], 4, false)?;
    /// let j = star.joint_degree_matrix(None, None, None)?;
    /// assert_eq!(j.to_rows(), vec![
    ///     vec![0.0, 0.0, 3.0],
    ///     vec![0.0, 0.0, 0.0],
    ///     vec![3.0, 0.0, 0.0],
    /// ]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn joint_degree_matrix(
        &self,
        weights: Option<&[f64]>,
        max_out_degree: Option<usize>,
        max_in_degree: Option<usize>,
    ) -> Result<Matrix> {
        self.mixing_check_weights(weights)?;
        let (max_out, max_in) = (limit(max_out_degree)?, limit(max_in_degree)?);
        let w = weights.map(Vector::view);
        let mut jdm = Matrix::new();
        igraph_call!(igraph_joint_degree_matrix(
            self,
            opt_ptr(&w),
            &mut jdm,
            max_out,
            max_in
        ))?;
        Ok(jdm)
    }

    /// Joint degree distribution `P_ij` of connected vertex pairs.
    ///
    /// Entry `(i, j)` is the probability that a randomly chosen *ordered* pair
    /// of connected vertices `u -> v` has degrees `i` (for `u`, computed with
    /// [`from_mode`](JointDegreeDistributionOptions::from_mode)) and `j` (for
    /// `v`, computed with [`to_mode`](JointDegreeDistributionOptions::to_mode)).
    /// An undirected graph behaves like the directed graph with all edges
    /// reciprocated. Without normalization the entries are connection counts
    /// (or total weights): without degree limits they sum to the number of
    /// edges of a directed graph (twice that with `directed_neighbors =
    /// false`) and to twice that of an undirected one. Rows and columns for
    /// degree 0 are included.
    ///
    /// Related quantities: the degree correlation function is
    /// `k_nn(k) = Σ_j j P_kj / Σ_j P_kj` and the unnormalized degree
    /// assortativity is `Σ_ij i j (P_ij - q_i r_j)` with `q` and `r` the row
    /// and column sums. Compare with [`joint_degree_matrix`](Self::joint_degree_matrix),
    /// whose undirected diagonal is half of the unnormalized `P_ii`.
    ///
    /// When connections are counted in both directions (undirected graphs,
    /// or `directed_neighbors = false`), each reverse connection `v -> u`
    /// contributes to entry `(deg_from(v), deg_to(u))` if it falls within
    /// the matrix, and normalization divides by the total weight of the
    /// connections that fall within it. In the cases where igraph 1.0.0 and
    /// 1.0.1 would index out of bounds (non-square limits, or
    /// `from_mode != to_mode` without `directed_neighbors`), this wrapper
    /// computes the matrix in Rust with exactly these semantics.
    ///
    /// Binds [`igraph_joint_degree_distribution`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_joint_degree_distribution).
    /// See also [`Graph::degree_correlation_vector`], which computes `k_nn(k)`
    /// directly, and [`Graph::assortativity`] with degree values.
    ///
    /// Time complexity: O(|E|).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the
    /// weight vector has the wrong length or a degree limit is `i64::MAX` or
    /// more.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::mixing::JointDegreeDistributionOptions;
    /// use igraph::prelude::*;
    ///
    /// let star = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], 4, false)?;
    /// let p = star.joint_degree_distribution(None, &JointDegreeDistributionOptions::default())?;
    /// assert_eq!(p.shape(), (4, 4));
    /// // Half of the ordered pairs go hub -> leaf, half leaf -> hub.
    /// assert_eq!(p[(3, 1)], 0.5);
    /// assert_eq!(p[(1, 3)], 0.5);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn joint_degree_distribution(
        &self,
        weights: Option<&[f64]>,
        options: &JointDegreeDistributionOptions,
    ) -> Result<Matrix> {
        self.mixing_check_weights(weights)?;
        let max_from = limit(options.max_from_degree)?;
        let max_to = limit(options.max_to_degree)?;
        let directed = self.is_directed();
        if !(directed && options.directed_neighbors) {
            // igraph 1.0.0 and 1.0.1 write the reverse entry of each connection without
            // bounds checking: unless the matrix is square and the source and
            // target degrees coincide, it would write out of bounds (or into
            // the wrong cell). Compute such cases on the Rust side.
            let (from_mode, to_mode) = if directed {
                (options.from_mode, options.to_mode)
            } else {
                (NeighborMode::All, NeighborMode::All)
            };
            let deg_from = self.degree(.., from_mode, Loops::Twice)?;
            let deg_to = if to_mode == from_mode {
                deg_from.clone()
            } else {
                self.degree(.., to_mode, Loops::Twice)?
            };
            let nrow = dimension(options.max_from_degree, &deg_from);
            let ncol = dimension(options.max_to_degree, &deg_to);
            if from_mode != to_mode || nrow != ncol {
                return self.mixing_matrix(
                    weights,
                    &deg_from,
                    &deg_to,
                    false,
                    options.normalized,
                    (nrow, ncol),
                );
            }
        }
        let w = weights.map(Vector::view);
        let mut p = Matrix::new();
        igraph_call!(igraph_joint_degree_distribution(
            self,
            opt_ptr(&w),
            &mut p,
            options.from_mode.into(),
            options.to_mode.into(),
            options.directed_neighbors,
            options.normalized,
            max_from,
            max_to
        ))?;
        Ok(p)
    }

    /// Mixing matrix of vertex categories.
    ///
    /// Entry `(i, j)` is proportional to the probability that a randomly
    /// chosen ordered pair of connected vertices `u -> v` has `from_types[u] = i`
    /// and `to_types[v] = j` (`to_types = None` reuses `from_types`). Types
    /// must be non-negative integers; the matrix has one more row/column than
    /// the largest source/target type, so re-index sparse labels first.
    /// Undirected graphs (or `directed = false`) count each edge in both
    /// directions. With `normalized = true` the entries sum to 1; otherwise
    /// they are connection counts (or total weights). When connections are
    /// counted in both directions, the reverse connection `v -> u` of an edge
    /// contributes to `(from_types[v], to_types[u])`; with distinct
    /// `to_types` this case is computed in Rust, because igraph 1.0.0 and
    /// 1.0.1 would index out of bounds.
    ///
    /// With a single normalized categorization `M`, row sums `a` and column
    /// sums `b`, the [modularity](Graph::modularity) of the partition is
    /// `Q = Σ_i M_ii - Σ_i a_i b_i` and the
    /// [nominal assortativity](Self::assortativity_nominal) is
    /// `Q / (1 - Σ_i a_i b_i)`.
    ///
    /// Binds [`igraph_joint_type_distribution`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_joint_type_distribution).
    ///
    /// Time complexity: O(|E|).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the
    /// weight vector or a type vector has the wrong length, or a type vector
    /// contains negative values (checked on the Rust side for `to_types` too,
    /// which igraph 1.0.0 and 1.0.1 themselves forget to validate).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // igraph's unit test: a small undirected multigraph with loops and 3 types.
    /// let g = Graph::from_flat_edges(
    ///     &[3, 0, 0, 3, 0, 2, 3, 1, 5, 5, 4, 2, 1, 1, 1, 1, 0, 1, 5, 1], 6, false)?;
    /// let m = g.joint_type_distribution(None, &[0, 0, 1, 1, 2, 2], None, false, false)?;
    /// assert_eq!(m.to_rows(), vec![
    ///     vec![6.0, 4.0, 1.0],
    ///     vec![4.0, 0.0, 1.0],
    ///     vec![1.0, 1.0, 2.0],
    /// ]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn joint_type_distribution(
        &self,
        weights: Option<&[f64]>,
        from_types: &[igraph_int_t],
        to_types: Option<&[igraph_int_t]>,
        directed: bool,
        normalized: bool,
    ) -> Result<Matrix> {
        self.mixing_check_weights(weights)?;
        self.check_vertex_values("from_types", from_types)?;
        check_types("from_types", from_types)?;
        if let Some(tt) = to_types {
            self.check_vertex_values("to_types", tt)?;
            // Mandatory for soundness: igraph 1.0.0 and 1.0.1 never check target types.
            check_types("to_types", tt)?;
        }
        if let Some(tt) = to_types
            && tt != from_types
            && !(directed && self.is_directed())
        {
            // Distinct source and target types with reciprocal counting: igraph
            // 1.0.0 and 1.0.1 would write the reverse entries out of bounds, see
            // `joint_degree_distribution`. Compute it on the Rust side.
            let dims = (dimension(None, from_types), dimension(None, tt));
            return self.mixing_matrix(weights, from_types, tt, false, normalized, dims);
        }
        let w = weights.map(Vector::view);
        let ft = VectorInt::view(from_types);
        let tt = to_types.map(VectorInt::view);
        let mut p = Matrix::new();
        igraph_call!(igraph_joint_type_distribution(
            self,
            opt_ptr(&w),
            &mut p,
            ft.as_ptr(),
            opt_ptr(&tt),
            directed,
            normalized
        ))?;
        Ok(p)
    }
}

// ----------------------------------------------------------------------
// igraph_graphicality.h
// ----------------------------------------------------------------------

/// Pre-screens degree sequences before handing them to the graphicality
/// tests of igraph 1.0.0 and 1.0.1 (`src/misc/graphicality.c`), which add up
/// degrees in `igraph_int_t` without overflow checks (`dsum += d`,
/// `2*dmax`, `sumdiff += din - dout`, `sum1 += d`, even the parity update
/// `sum_parity + d`): signed overflow is undefined behavior in C, so huge
/// degrees must never reach them.
///
/// Returns `Some(false)` if an entry is negative (every igraph test answers
/// "not graphical" for those, but may overflow on earlier entries before
/// seeing the negative one), an error if the total exceeds `i64::MAX / 2`
/// (which keeps every intermediate value of the C code in range), and `None`
/// if igraph can be called.
fn screen_degrees(seqs: &[&[igraph_int_t]]) -> Result<Option<bool>> {
    let mut total: i128 = 0;
    for &d in seqs.iter().flat_map(|s| s.iter()) {
        if d < 0 {
            return Ok(Some(false));
        }
        total += i128::from(d);
    }
    if total > i128::from(igraph_int_t::MAX / 2) {
        return Err(Error::invalid(format!(
            "degree sum {total} is too large (more than i64::MAX / 2)"
        )));
    }
    Ok(None)
}

/// Is there a graph with the given degree sequence?
///
/// For an undirected graph pass the degrees as `out_degrees` and
/// `in_degrees = None`; for a directed graph pass both the out- and the
/// in-degree sequences (of equal length). `allowed` says which edges the
/// realization may use (anything convertible into [`AllowedEdgeTypes`], e.g.
/// [`EdgeTypeSw::Simple`] or `AllowedEdgeTypes::ALL`). Sequences with negative
/// entries are simply not graphical.
///
/// The tests used are, for undirected graphs: the Erdős–Gallai conditions
/// (Cloteaux's linear-time algorithm) for simple graphs; an even degree sum
/// when loops and multi-edges are allowed; additionally a degree sum at least
/// twice the maximum degree for loopless multigraphs; Cairns–Mendan's modified
/// Erdős–Gallai conditions for at most one self-loop per vertex. For directed
/// graphs: the Fulkerson–Chen–Anstee theorem with Berger's relaxation for
/// simple digraphs; equal in- and out-degree sums with loops and multi-edges;
/// additionally an out-degree sum at least the maximum total degree for
/// loopless multigraphs; the Gale–Ryser theorem when single self-loops are
/// allowed.
///
/// Binds [`igraph_is_graphical`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_is_graphical).
/// See also [`Graph::realize_degree_sequence`], which builds a realization
/// of a graphical sequence (for the edge types it implements: simple, loopless
/// multi- and loopy multigraphs when undirected, simple digraphs when
/// directed, it succeeds exactly when this test says yes; single self-loops
/// without multi-edges, and non-simple digraphs, fail with
/// [`ErrorKind::Unimplemented`](crate::ErrorKind::Unimplemented)), and
/// [`Graph::degree_sequence_game`], which samples one at random.
///
/// Time complexity: O(n), n being the length of the sequence(s).
///
/// # Errors
///
/// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the out- and
/// in-degree sequences have different lengths, or if the sum of the
/// (non-negative) degrees exceeds `i64::MAX / 2`: igraph would add them up
/// with signed overflow, which is undefined behavior in C, so such
/// sequences are rejected on the Rust side.
///
/// # Examples
///
/// ```
/// use igraph::mixing::{is_graphical, AllowedEdgeTypes};
/// use igraph::prelude::*;
///
/// // (3, 3): two vertices of degree 3 need a triple edge or loops.
/// assert!(!is_graphical(&[3, 3], None, EdgeTypeSw::Simple)?);
/// assert!(is_graphical(&[3, 3], None, EdgeTypeSw::Multi)?);
/// assert!(is_graphical(&[3, 3], None, EdgeTypeSw::Loops)?);
/// // (1, 2, 5) needs multi-edges *and* loops.
/// assert!(!is_graphical(&[1, 2, 5], None, EdgeTypeSw::Multi)?);
/// assert!(is_graphical(&[1, 2, 5], None, AllowedEdgeTypes::ALL)?);
/// // Directed: a 3-cycle has out- and in-degrees all equal to 1.
/// assert!(is_graphical(&[1, 1, 1], Some(&[1, 1, 1]), EdgeTypeSw::Simple)?);
/// assert!(!is_graphical(&[2, 0], Some(&[0, 2]), EdgeTypeSw::Simple)?);
///
/// // A graphical sequence can be realized with the same edge types.
/// let seq = [3, 3, 2, 2, 2, 1, 1];
/// assert!(is_graphical(&seq, None, AllowedEdgeTypes::SIMPLE)?);
/// let g = Graph::realize_degree_sequence(
///     &seq, None, AllowedEdgeTypes::SIMPLE, RealizeDegseq::Smallest)?;
/// assert_eq!(g.degree(.., NeighborMode::All, Loops::Twice)?, seq);
/// # Ok::<(), igraph::Error>(())
/// ```
pub fn is_graphical(
    out_degrees: &[igraph_int_t],
    in_degrees: Option<&[igraph_int_t]>,
    allowed: impl Into<AllowedEdgeTypes>,
) -> Result<bool> {
    // With sequences of different lengths igraph errors out before reading
    // any degree; otherwise pre-screen the values (see `screen_degrees`).
    if in_degrees.is_none_or(|inn| inn.len() == out_degrees.len())
        && let Some(answer) = screen_degrees(&[out_degrees, in_degrees.unwrap_or(&[])])?
    {
        return Ok(answer);
    }
    let out = VectorInt::view(out_degrees);
    let inn = in_degrees.map(VectorInt::view);
    let mut res = false;
    igraph_call!(igraph_is_graphical(
        out.as_ptr(),
        opt_ptr(&inn),
        allowed.into().to_raw(),
        &mut res
    ))?;
    Ok(res)
}

/// Is there a bipartite graph with the given pair of degree sequences?
///
/// `degrees1` and `degrees2` are the degrees of the vertices in the two
/// partitions. When multi-edges are allowed it suffices that both sequences
/// have the same sum (and no negative entry); for simple graphs the Gale–Ryser
/// theorem is used with Berger's relaxation. Self-loops are meaningless in
/// bipartite graphs, so only [`AllowedEdgeTypes::SIMPLE`] and
/// [`AllowedEdgeTypes::MULTI`] matter (the `loops` flag is ignored).
///
/// Binds [`igraph_is_bigraphical`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_is_bigraphical).
/// See also [`Graph::realize_bipartite_degree_sequence`], which builds a
/// realization, and [`Graph::is_bipartite`].
///
/// Time complexity: O(n), n being the length of the longer sequence.
///
/// # Errors
///
/// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the sum of
/// the (non-negative) degrees exceeds `i64::MAX / 2` (see [`is_graphical`]).
///
/// # Examples
///
/// ```
/// use igraph::mixing::is_bigraphical;
/// use igraph::prelude::*;
///
/// // K_{2,3}: two vertices of degree 3 and three of degree 2.
/// assert!(is_bigraphical(&[3, 3], &[2, 2, 2], EdgeTypeSw::Simple)?);
/// // One vertex cannot have 4 distinct neighbors among 3.
/// assert!(!is_bigraphical(&[4], &[2, 1, 1], EdgeTypeSw::Simple)?);
/// assert!(is_bigraphical(&[4], &[2, 1, 1], EdgeTypeSw::Multi)?);
///
/// // Realize K_{2,3}'s sequences: the result is the complete bipartite graph.
/// let g = Graph::realize_bipartite_degree_sequence(
///     &[3, 3], &[2, 2, 2], EdgeTypeSw::Simple, RealizeDegseq::Smallest)?;
/// assert_eq!((g.vcount(), g.ecount()), (5, 6));
/// assert!(g.is_bipartite()?);
/// # Ok::<(), igraph::Error>(())
/// ```
pub fn is_bigraphical(
    degrees1: &[igraph_int_t],
    degrees2: &[igraph_int_t],
    allowed: impl Into<AllowedEdgeTypes>,
) -> Result<bool> {
    if let Some(answer) = screen_degrees(&[degrees1, degrees2])? {
        return Ok(answer);
    }
    let d1 = VectorInt::view(degrees1);
    let d2 = VectorInt::view(degrees2);
    let mut res = false;
    igraph_call!(igraph_is_bigraphical(
        d1.as_ptr(),
        d2.as_ptr(),
        allowed.into().to_raw(),
        &mut res
    ))?;
    Ok(res)
}

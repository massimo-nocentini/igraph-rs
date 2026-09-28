//! Centrality measures, graph centralization and local scan statistics
//! (`igraph_centrality.h`, `igraph_scan.h`).
//!
//! A *centrality* assigns every vertex (or edge) a score that tells how
//! "important" it is in the network. Different notions of importance lead to
//! different measures: being close to everybody else ([closeness](igraph_t::closeness),
//! [harmonic centrality](igraph_t::harmonic_centrality)), lying on many shortest
//! paths ([betweenness](igraph_t::betweenness)), being connected to other
//! important vertices ([eigenvector centrality](igraph_t::eigenvector_centrality),
//! [PageRank](igraph_t::pagerank), [hub and authority scores](igraph_t::hub_and_authority_scores)),
//! or bridging structural holes ([Burt's constraint](igraph_t::constraint)).
//!
//! A *centralization* index condenses vertex-level scores into a single number
//! describing how much the whole graph is dominated by a single vertex; it is
//! usually normalized by its value on the most centralized graph of the same
//! size (typically a star), the *theoretical maximum*.
//!
//! *Local scan statistics* count edges (or sum edge weights) within the
//! neighborhoods of vertices, and are used for anomaly detection in (time
//! series of) graphs.
//!
//! All the functions are methods of [`Graph`], except for the graph-free
//! [`centralization`] and the `*_tmax` free functions that compute theoretical
//! maxima from a number of vertices only.
//!
//! # Example
//!
//! ```
//! use igraph::prelude::*;
//! use igraph::centrality::PageRankOptions;
//!
//! // A star with 5 leaves: the center is the most central in every sense.
//! let star = Graph::star(6, StarMode::Undirected, 0).unwrap();
//!
//! let btw = star.betweenness(None, .., true, false).unwrap();
//! assert_eq!(btw, vec![10.0, 0.0, 0.0, 0.0, 0.0, 0.0]); // C(5, 2) = 10 pairs of leaves
//!
//! let clo = star.closeness(.., NeighborMode::All, None, true).unwrap();
//! assert_eq!(clo[0], 1.0);
//!
//! let pr = star.pagerank(None, .., &PageRankOptions::default()).unwrap();
//! assert!((pr.scores.iter().sum::<f64>() - 1.0).abs() < 1e-12);
//! assert!(pr.scores[0] > pr.scores[1]);
//!
//! // The star is the most centralized graph: normalized centralization is 1.
//! let c = star.centralization_degree(NeighborMode::All, Loops::None, true).unwrap();
//! assert!((c.centralization - 1.0).abs() < 1e-12);
//!
//! // In Zachary's karate club the instructor (0) and the administrator (33)
//! // are the two most "between" members.
//! let karate = Graph::famous("Zachary").unwrap();
//! let btw = karate.betweenness(None, .., false, false).unwrap();
//! let mut order: Vec<usize> = (0..34).collect();
//! order.sort_by(|&a, &b| btw[b].total_cmp(&btw[a]));
//! assert_eq!(&order[..2], &[0, 33]);
//! ```
//!
//! # Provided functionality
//!
//! | Measure | Methods | C functions |
//! |---|---|---|
//! | Closeness | [`closeness`](igraph_t::closeness), [`closeness_cutoff`](igraph_t::closeness_cutoff), [`closeness_reachability`](igraph_t::closeness_reachability) | `igraph_closeness`, `igraph_closeness_cutoff` |
//! | Harmonic centrality | [`harmonic_centrality`](igraph_t::harmonic_centrality), [`harmonic_centrality_cutoff`](igraph_t::harmonic_centrality_cutoff) | `igraph_harmonic_centrality[_cutoff]` |
//! | Vertex betweenness | [`betweenness`](igraph_t::betweenness), [`betweenness_cutoff`](igraph_t::betweenness_cutoff), [`betweenness_subset`](igraph_t::betweenness_subset) | `igraph_betweenness[_cutoff/_subset]` |
//! | Edge betweenness | [`edge_betweenness`](igraph_t::edge_betweenness), [`edge_betweenness_cutoff`](igraph_t::edge_betweenness_cutoff), [`edge_betweenness_subset`](igraph_t::edge_betweenness_subset) | `igraph_edge_betweenness[_cutoff/_subset]` |
//! | PageRank | [`pagerank`](igraph_t::pagerank), [`personalized_pagerank`](igraph_t::personalized_pagerank), [`personalized_pagerank_vs`](igraph_t::personalized_pagerank_vs) | `igraph_pagerank`, `igraph_personalized_pagerank[_vs]` |
//! | Spectral | [`eigenvector_centrality`](igraph_t::eigenvector_centrality), [`hub_and_authority_scores`](igraph_t::hub_and_authority_scores) | `igraph_eigenvector_centrality`, `igraph_hub_and_authority_scores` |
//! | Structural holes | [`constraint`](igraph_t::constraint) | `igraph_constraint` |
//! | Edge convergence | [`convergence_degree`](igraph_t::convergence_degree) | `igraph_convergence_degree` |
//! | Centralization | [`centralization`], [`centralization_degree`](igraph_t::centralization_degree), [`centralization_betweenness`](igraph_t::centralization_betweenness), [`centralization_closeness`](igraph_t::centralization_closeness), [`centralization_eigenvector_centrality`](igraph_t::centralization_eigenvector_centrality) | `igraph_centralization*` |
//! | Theoretical maxima | [`centralization_degree_tmax`], [`centralization_betweenness_tmax`], [`centralization_closeness_tmax`], [`centralization_eigenvector_centrality_tmax`] and the homonymous methods of [`Graph`] | `igraph_centralization_*_tmax` |
//! | Local scan statistics | [`local_scan_0`](igraph_t::local_scan_0), [`local_scan_1_ecount`](igraph_t::local_scan_1_ecount), [`local_scan_k_ecount`](igraph_t::local_scan_k_ecount), their `_them` variants, [`local_scan_subset_ecount`](igraph_t::local_scan_subset_ecount), [`local_scan_neighborhood_ecount`](igraph_t::local_scan_neighborhood_ecount) | `igraph_local_scan_*` |
//!
//! # Conventions
//!
//! - Edge weights are passed as `Option<&[f64]>`, one weight per edge, `None`
//!   meaning an unweighted computation. The shortest-path based measures
//!   (closeness, harmonic centrality, betweenness) and PageRank reject NaN
//!   weights; betweenness also requires them to be strictly positive.
//! - Cutoffs are `Option<f64>`: `None` means no limit on the path lengths
//!   (the exact measure is computed); `Some(c)` only considers paths of length
//!   at most `c` (a negative `c` also means "no limit", as in C).
//! - Vertex and edge sets are anything convertible into a
//!   [`VertexSelector`]/[`EdgeSelector`], e.g. `..` (all), a single id, a
//!   slice or a range of ids. Results follow the order of the selector.
//! - ARPACK-based computations (eigenvector centrality, hub and authority
//!   scores, [`PageRankAlgo::Arpack`]) always run with igraph's default ARPACK
//!   options, which are adequate for virtually every graph.
//! - igraph reports non-fatal problems (e.g. eigenvector centrality of a
//!   disconnected graph) as *warnings*, not errors: collect them with
//!   [`take_warnings`](crate::error::take_warnings).
//!
//! # See also
//!
//! - Shortest-path quantities that closeness and betweenness are built on:
//!   [`distances`](igraph_t::distances),
//!   [`eccentricity`](igraph_t::eccentricity), [`radius`](igraph_t::radius)
//!   and [`average_path_length`](igraph_t::average_path_length) in
//!   [`paths`](crate::paths).
//! - Degree-like measures: [`degree`](igraph_t::degree),
//!   [`strength`](igraph_t::strength) and [`maxdegree`](igraph_t::maxdegree);
//!   k-core decomposition with [`coreness`](igraph_t::coreness).
//! - Community detection by removing high-betweenness edges:
//!   [`community_edge_betweenness`](igraph_t::community_edge_betweenness).
//! - The whole spectrum of the adjacency matrix, of which eigenvector
//!   centrality is the leading eigenvector:
//!   [`eigen_adjacency`](igraph_t::eigen_adjacency).
//! - Local clustering, the other classic ego-network measure next to Burt's
//!   constraint: [`transitivity_local_undirected`](igraph_t::transitivity_local_undirected)
//!   and [`count_adjacent_triangles`](igraph_t::count_adjacent_triangles).

use crate::{
    constants::{Loops, NeighborMode},
    error::{Error, Result},
    ffi::*,
    graph::{Graph, VertexId},
    igraph_call,
    list::VectorIntList,
    selector::{EdgeSelector, VertexSelector},
    vector::{Vector, VectorInt},
};

crate::ffi_enum! {
    /// The algorithm used to compute PageRank (`igraph_pagerank_algo_t`).
    pub enum PageRankAlgo: igraph_pagerank_algo_t {
        /// Phrase PageRank as an eigenvalue problem and solve it with ARPACK
        /// (the default before igraph 0.7). The returned
        /// [`value`](EigenScores::value) is the eigenvalue, which should be 1:
        /// checking it detects convergence failures.
        Arpack = igraph_pagerank_algo_t_IGRAPH_PAGERANK_ALGO_ARPACK,
        /// Solve a linear system with the PRPACK library
        /// (<https://github.com/dgleich/prpack>). Recommended, and the default.
        Prpack = igraph_pagerank_algo_t_IGRAPH_PAGERANK_ALGO_PRPACK,
    }
}

impl Default for PageRankAlgo {
    /// [`PageRankAlgo::Prpack`], igraph's recommended implementation.
    fn default() -> Self {
        Self::Prpack
    }
}

/// Tuning parameters of the PageRank family of functions
/// ([`pagerank`](igraph_t::pagerank), [`personalized_pagerank`](igraph_t::personalized_pagerank),
/// [`personalized_pagerank_vs`](igraph_t::personalized_pagerank_vs)).
///
/// The [`Default`] is the classic setting: damping `0.85`, directed paths,
/// [`PageRankAlgo::Prpack`].
///
/// ```
/// use igraph::centrality::{PageRankAlgo, PageRankOptions};
///
/// let opts = PageRankOptions::default().with_damping(0.5).with_algo(PageRankAlgo::Arpack);
/// assert_eq!((opts.damping, opts.directed, opts.algo), (0.5, true, PageRankAlgo::Arpack));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageRankOptions {
    /// The damping factor (`d` in the original paper): the probability that the
    /// random walker follows an edge instead of restarting. Must be in `[0, 1]`.
    pub damping: f64,
    /// Whether to follow edge directions in directed graphs (ignored for
    /// undirected graphs).
    pub directed: bool,
    /// The implementation to use.
    pub algo: PageRankAlgo,
}

impl Default for PageRankOptions {
    fn default() -> Self {
        Self {
            damping: 0.85,
            directed: true,
            algo: PageRankAlgo::Prpack,
        }
    }
}

impl PageRankOptions {
    /// Sets the [`damping`](Self::damping) factor.
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping;
        self
    }

    /// Sets whether edge directions are followed ([`directed`](Self::directed)).
    pub fn with_directed(mut self, directed: bool) -> Self {
        self.directed = directed;
        self
    }

    /// Sets the implementation ([`algo`](Self::algo)).
    pub fn with_algo(mut self, algo: PageRankAlgo) -> Self {
        self.algo = algo;
        self
    }
}

/// Closeness scores together with reachability information, returned by
/// [`closeness_reachability`](igraph_t::closeness_reachability).
#[derive(Debug, Clone, PartialEq)]
pub struct Closeness {
    /// The closeness centrality of each requested vertex (NaN for vertices
    /// that reach no other vertex).
    pub scores: Vec<f64>,
    /// For each requested vertex, the number of vertices reachable from it
    /// (within the cutoff, if any), not counting the vertex itself.
    pub reachable_count: Vec<i64>,
    /// Whether every vertex of the graph was reachable from each requested
    /// vertex. `false` proves that the graph is disconnected; `true` proves it
    /// is connected if the graph is undirected, or if it is directed and all
    /// vertices were requested.
    pub all_reachable: bool,
}

/// Vertex scores that are an eigenvector, with the corresponding eigenvalue.
///
/// Returned by [`pagerank`](igraph_t::pagerank) and its personalized
/// variants, and by [`eigenvector_centrality`](igraph_t::eigenvector_centrality).
#[derive(Debug, Clone, PartialEq)]
pub struct EigenScores {
    /// The score of each vertex (for PageRank: of each requested vertex).
    pub scores: Vec<f64>,
    /// The eigenvalue. For PageRank it is always `1.0` with PRPACK and should
    /// be very close to one with ARPACK. For eigenvector centrality it is the
    /// leading eigenvalue of the adjacency matrix (zero for acyclic graphs,
    /// where the measure is not meaningful).
    pub value: f64,
}

/// Kleinberg's hub and authority scores, see
/// [`hub_and_authority_scores`](igraph_t::hub_and_authority_scores).
#[derive(Debug, Clone, PartialEq)]
pub struct HubAuthority {
    /// Hub score of each vertex, scaled so that the maximum is 1.
    pub hubs: Vec<f64>,
    /// Authority score of each vertex, scaled so that the maximum is 1.
    pub authorities: Vec<f64>,
    /// The leading eigenvalue of `A Aᵀ` (equivalently, of `Aᵀ A`).
    pub value: f64,
}

/// Convergence degrees of the edges, see
/// [`convergence_degree`](igraph_t::convergence_degree).
#[derive(Debug, Clone, PartialEq)]
pub struct ConvergenceDegree {
    /// The convergence degree of each edge, in `(-1, 1)`.
    pub result: Vec<f64>,
    /// The size of the input set of each edge.
    pub ins: Vec<f64>,
    /// The size of the output set of each edge.
    pub outs: Vec<f64>,
}

/// Vertex-level scores together with the graph-level centralization index.
///
/// Returned by [`centralization_degree`](igraph_t::centralization_degree),
/// [`centralization_betweenness`](igraph_t::centralization_betweenness) and
/// [`centralization_closeness`](igraph_t::centralization_closeness).
#[derive(Debug, Clone, PartialEq)]
pub struct Centralization {
    /// The vertex-level centrality scores of all vertices.
    pub scores: Vec<f64>,
    /// The graph-level centralization index (normalized if requested).
    pub centralization: f64,
    /// The centralization of the most centralized graph with the same number
    /// of vertices (and directedness).
    pub theoretical_max: f64,
}

/// Eigenvector centralities together with the graph-level centralization
/// index, see [`centralization_eigenvector_centrality`](igraph_t::centralization_eigenvector_centrality).
#[derive(Debug, Clone, PartialEq)]
pub struct EigenvectorCentralization {
    /// The eigenvector centrality of all vertices, scaled so that the maximum is 1.
    pub scores: Vec<f64>,
    /// The leading eigenvalue.
    pub value: f64,
    /// The graph-level centralization index (normalized if requested).
    pub centralization: f64,
    /// The centralization of the most centralized graph with the same number
    /// of vertices (and directedness).
    pub theoretical_max: f64,
}

/// Validates an optional per-edge weight vector and builds a view over it.
fn weights_view<'w>(
    graph: &Graph,
    weights: Option<&'w [f64]>,
) -> Result<Option<crate::vector::View<'w, Vector>>> {
    if let Some(w) = weights
        && w.len() != graph.ecount()
    {
        return Err(Error::invalid(format!(
            "the weight vector has length {}, but the graph has {} edges",
            w.len(),
            graph.ecount()
        )));
    }
    Ok(weights.map(Vector::view))
}

/// Largest sum of absolute weights passed unchanged to the eigenvector-based
/// (ARPACK / PRPACK) routines: squares of such sums stay finite.
const MAX_ABS_WEIGHT_SUM: f64 = 1e150;

/// Validates weights for the eigenvector-based centralities and, if they are
/// so large that the iterations could overflow, rescales them.
///
/// igraph (1.0.0 and 1.0.1) does not reject infinite weights (nor NaN ones,
/// except for PageRank with ARPACK), and huge finite weights overflow the
/// matrix-vector products: ARPACK then aborts the whole process (an f2c
/// `STOP`), PRPACK silently returns NaN scores. All these measures are
/// invariant under scaling the weights (only the eigenvalue scales, by the
/// returned factor for the adjacency matrix), so weights whose absolute sum
/// exceeds [`MAX_ABS_WEIGHT_SUM`] are divided by the smallest power of two
/// that brings the sum below it. A power of two keeps the division exact
/// and, unlike normalizing the largest weight to `1`, does not flush small
/// weights to (sub)normal zero: e.g. PageRank divides by the out-strength,
/// and a vertex whose only out-weight became `1e-320` would get infinite,
/// then `NaN`, transition probabilities.
/// Returns the rescaled weights (if any) and the scale factor.
fn spectral_weights(weights: Option<&[f64]>) -> Result<(Option<Vec<f64>>, f64)> {
    let Some(w) = weights else {
        return Ok((None, 1.0));
    };
    if let Some(&x) = w.iter().find(|x| !x.is_finite()) {
        return Err(Error::invalid(format!(
            "the weights must be finite, found {x}"
        )));
    }
    let sum: f64 = w.iter().map(|x| x.abs()).sum();
    if sum <= MAX_ABS_WEIGHT_SUM {
        return Ok((None, 1.0));
    }
    // `sum` may have overflowed: compute log2(sum) as log2(max) + log2(sum / max).
    let max = w.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let rel: f64 = w.iter().map(|x| x.abs() / max).sum();
    let excess = max.log2() + rel.log2() - MAX_ABS_WEIGHT_SUM.log2();
    // One extra halving absorbs the rounding of the logarithms. The exponent
    // is at most about 1024 + 64 - 498, so the factor is finite.
    let scale = 2f64.powi(excess.ceil() as i32 + 1);
    Ok((Some(w.iter().map(|x| x / scale).collect()), scale))
}

/// Validates the PageRank damping factor: igraph's own range check lets NaN
/// through (and ARPACK then aborts the process).
fn check_damping(damping: f64) -> Result<()> {
    if !(0.0..=1.0).contains(&damping) {
        return Err(Error::invalid(format!(
            "the PageRank damping factor must be in [0, 1], got {damping}"
        )));
    }
    Ok(())
}

fn ptr_of(view: &Option<crate::vector::View<'_, Vector>>) -> *const igraph_vector_t {
    view.as_ref().map_or(std::ptr::null(), |v| v.as_ptr())
}

fn raw_cutoff(cutoff: Option<f64>) -> f64 {
    cutoff.unwrap_or(-1.0)
}

fn raw_nodes(nodes: usize) -> Result<igraph_int_t> {
    igraph_int_t::try_from(nodes).map_err(|_| Error::invalid("too many vertices"))
}

/// Checks that `them` is compatible with `us` for the `_them` scan statistics.
fn check_them(us: &Graph, them: &Graph) -> Result<()> {
    if us.vcount() != them.vcount() {
        return Err(Error::invalid(
            "the two graphs must have the same number of vertices",
        ));
    }
    if us.is_directed() != them.is_directed() {
        return Err(Error::invalid(
            "the two graphs must have the same directedness",
        ));
    }
    Ok(())
}

/// Computes the graph-level centralization index from vertex-level scores
/// (`igraph_centralization`).
///
/// The (unnormalized) centralization is `C = Σ_v (max_u c_u − c_v)`, the sum of
/// the deviations from the largest score. If `normalized` is true,
/// `C / theoretical_max` is returned instead, where `theoretical_max` is the
/// centralization of the most centralized structure with the same number of
/// vertices (usually a star, see e.g. [`centralization_degree_tmax`]); it is
/// ignored otherwise. An empty `scores` slice gives NaN. Time complexity:
/// O(n), the number of scores.
///
/// Binds [`igraph_centralization`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_centralization).
///
/// # Examples
///
/// ```
/// use igraph::centrality::centralization;
///
/// assert_eq!(centralization(&[3.0, 1.0, 1.0, 1.0], 0.0, false), 6.0);
/// assert_eq!(centralization(&[3.0, 1.0, 1.0, 1.0], 6.0, true), 1.0);
/// ```
pub fn centralization(scores: &[f64], theoretical_max: f64, normalized: bool) -> f64 {
    let view = Vector::view(scores);
    unsafe { igraph_centralization(view.as_ptr(), theoretical_max, normalized) }
}

/// Theoretical maximum of degree centralization for a graph with `nodes`
/// vertices (`igraph_centralization_degree_tmax` with a null graph).
///
/// The graph is considered directed unless `mode` is [`NeighborMode::All`].
/// The most centralized structure is the star (the in- or out-star for
/// directed graphs). `loops` tells whether self-loops count (and how) in the
/// degree, since they change the maximum. For `nodes == 0` the result is NaN.
/// See [`igraph_t::centralization_degree_tmax`] to read size and directedness
/// from a graph. Time complexity: O(1).
///
/// Binds [`igraph_centralization_degree_tmax`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_centralization_degree_tmax).
///
/// # Examples
///
/// ```
/// use igraph::{centrality::centralization_degree_tmax, prelude::*};
///
/// // Undirected star on 5 vertices: (n - 1)(n - 2) = 12.
/// assert_eq!(centralization_degree_tmax(5, NeighborMode::All, Loops::None).unwrap(), 12.0);
/// ```
pub fn centralization_degree_tmax(nodes: usize, mode: NeighborMode, loops: Loops) -> Result<f64> {
    let mut res = 0.0;
    igraph_call!(igraph_centralization_degree_tmax(
        std::ptr::null(),
        raw_nodes(nodes)?,
        mode.into(),
        loops.into(),
        &mut res
    ))?;
    Ok(res)
}

/// Theoretical maximum of betweenness centralization for a graph with `nodes`
/// vertices (`igraph_centralization_betweenness_tmax` with a null graph).
///
/// `directed` tells whether directed paths are used. The most centralized
/// structure is the star. See [`igraph_t::centralization_betweenness_tmax`]
/// to read size and directedness from a graph. Time complexity: O(1).
///
/// Binds [`igraph_centralization_betweenness_tmax`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_centralization_betweenness_tmax).
///
/// # Examples
///
/// ```
/// use igraph::centrality::centralization_betweenness_tmax;
///
/// // Undirected: (n - 1)^2 (n - 2) / 2.
/// assert_eq!(centralization_betweenness_tmax(5, false).unwrap(), 24.0);
/// ```
pub fn centralization_betweenness_tmax(nodes: usize, directed: bool) -> Result<f64> {
    let mut res = 0.0;
    igraph_call!(igraph_centralization_betweenness_tmax(
        std::ptr::null(),
        raw_nodes(nodes)?,
        directed,
        &mut res
    ))?;
    Ok(res)
}

/// Theoretical maximum of closeness centralization for a graph with `nodes`
/// vertices (`igraph_centralization_closeness_tmax` with a null graph).
///
/// The graph is considered directed unless `mode` is [`NeighborMode::All`].
/// The most centralized structure is the star. The maximum refers to
/// *normalized* closeness scores, as used by
/// [`centralization_closeness`](igraph_t::centralization_closeness). Time
/// complexity: O(1).
///
/// Binds [`igraph_centralization_closeness_tmax`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_centralization_closeness_tmax).
///
/// # Examples
///
/// ```
/// use igraph::{centrality::centralization_closeness_tmax, prelude::*};
///
/// // Undirected: (n - 1)(n - 2) / (2n - 3).
/// let t = centralization_closeness_tmax(5, NeighborMode::All).unwrap();
/// assert!((t - 12.0 / 7.0).abs() < 1e-12);
/// ```
pub fn centralization_closeness_tmax(nodes: usize, mode: NeighborMode) -> Result<f64> {
    let mut res = 0.0;
    igraph_call!(igraph_centralization_closeness_tmax(
        std::ptr::null(),
        raw_nodes(nodes)?,
        mode.into(),
        &mut res
    ))?;
    Ok(res)
}

/// Theoretical maximum of eigenvector centralization for a graph with
/// `nodes` vertices (`igraph_centralization_eigenvector_centrality_tmax`
/// with a null graph).
///
/// The graph is considered directed unless `mode` is [`NeighborMode::All`].
/// The most centralized undirected structure is a graph with a single edge;
/// the directed one is the in-star (for [`NeighborMode::Out`]) or the out-star
/// (for [`NeighborMode::In`]). Scores are assumed to be scaled so that the
/// maximum is 1. Time complexity: O(1).
///
/// Binds [`igraph_centralization_eigenvector_centrality_tmax`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_centralization_eigenvector_centrality_tmax).
///
/// # Examples
///
/// ```
/// use igraph::{centrality::centralization_eigenvector_centrality_tmax, prelude::*};
///
/// // Undirected: n - 2 (one edge, all other scores zero).
/// assert_eq!(centralization_eigenvector_centrality_tmax(10, NeighborMode::All).unwrap(), 8.0);
/// ```
pub fn centralization_eigenvector_centrality_tmax(nodes: usize, mode: NeighborMode) -> Result<f64> {
    let mut res = 0.0;
    igraph_call!(igraph_centralization_eigenvector_centrality_tmax(
        std::ptr::null(),
        raw_nodes(nodes)?,
        mode.into(),
        &mut res
    ))?;
    Ok(res)
}

impl igraph_t {
    // ------------------------------------------------------------------
    // Closeness and harmonic centrality
    // ------------------------------------------------------------------

    /// Closeness centrality of the selected vertices (`igraph_closeness`).
    ///
    /// The closeness of a vertex is the inverse of the mean distance to (or
    /// from) all other vertices; it measures how easily other vertices are
    /// reached from it. `mode` selects the paths in directed graphs:
    /// [`NeighborMode::Out`] uses distances *from* the vertex,
    /// [`NeighborMode::In`] distances *to* it and [`NeighborMode::All`] ignores
    /// directions. With `weights`, path lengths are the sums of edge weights.
    ///
    /// If `normalized` is true the result is the inverse of the *mean* distance,
    /// otherwise the inverse of the *sum* of distances.
    ///
    /// Closeness is meaningful for connected graphs only: in disconnected
    /// graphs igraph only considers *reachable* vertices (in undirected graphs
    /// this is closeness computed per component). Isolated vertices get NaN.
    /// Use [`closeness_reachability`](Self::closeness_reachability) to detect
    /// disconnectedness, or consider [`harmonic_centrality`](Self::harmonic_centrality).
    ///
    /// Time complexity: O(n|E|) unweighted, O(n|E|log|V| + |V|) weighted, for
    /// `n` requested vertices.
    ///
    /// See also [`distances`](Self::distances) for the underlying distance
    /// matrix and [`eccentricity`](Self::eccentricity) for the *largest*
    /// (instead of the mean) distance from a vertex.
    ///
    /// Binds [`igraph_closeness`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_closeness).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) for an
    /// invalid vertex, [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue)
    /// for weights of the wrong length or containing NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Path 0 - 1 - 2: the middle vertex is at distance 1 from both ends.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// assert_eq!(g.closeness(.., NeighborMode::All, None, false).unwrap(), vec![1.0 / 3.0, 0.5, 1.0 / 3.0]);
    /// assert_eq!(g.closeness(1, NeighborMode::All, None, true).unwrap(), vec![1.0]);
    /// ```
    pub fn closeness<'a>(
        &self,
        vids: impl Into<VertexSelector<'a>>,
        mode: NeighborMode,
        weights: Option<&[f64]>,
        normalized: bool,
    ) -> Result<Vec<f64>> {
        let vs = vids.into().to_raw()?;
        let w = weights_view(self, weights)?;
        let mut res = Vector::new();
        igraph_call!(igraph_closeness(
            self,
            &mut res,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            vs.get(),
            mode.into(),
            ptr_of(&w),
            normalized
        ))?;
        Ok(res.into())
    }

    /// Range-limited closeness centrality (`igraph_closeness_cutoff`).
    ///
    /// Like [`closeness`](Self::closeness), but only shortest paths of length
    /// at most `cutoff` are considered (vertices farther away count as
    /// unreachable). `None` (or a negative cutoff) computes the exact closeness.
    /// Smaller cutoffs make the computation faster.
    ///
    /// Binds [`igraph_closeness_cutoff`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_closeness_cutoff).
    ///
    /// # Errors
    ///
    /// As for [`closeness`](Self::closeness).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Path 0 - 1 - 2 - 3: within distance 1, vertex 0 only reaches vertex 1.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let c = g.closeness_cutoff(0, NeighborMode::All, None, true, Some(1.0)).unwrap();
    /// assert_eq!(c, vec![1.0]);
    /// ```
    pub fn closeness_cutoff<'a>(
        &self,
        vids: impl Into<VertexSelector<'a>>,
        mode: NeighborMode,
        weights: Option<&[f64]>,
        normalized: bool,
        cutoff: Option<f64>,
    ) -> Result<Vec<f64>> {
        Ok(self
            .closeness_reachability(vids, mode, weights, normalized, cutoff)?
            .scores)
    }

    /// Closeness centrality with reachability information
    /// (`igraph_closeness_cutoff` with all outputs).
    ///
    /// Besides the (possibly range-limited, see
    /// [`closeness_cutoff`](Self::closeness_cutoff)) closeness scores, it
    /// returns the number of vertices reachable from each requested vertex and
    /// whether all vertices were reachable, see [`Closeness`]. These make it
    /// possible to compute the generalizations of closeness to disconnected
    /// graphs that rescale scores by the size of the reachable set.
    ///
    /// Binds [`igraph_closeness_cutoff`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_closeness_cutoff).
    ///
    /// # Errors
    ///
    /// As for [`closeness`](Self::closeness).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Two disjoint edges: closeness is computed within each component.
    /// let g = Graph::from_edges(&[(0, 1), (2, 3)], 4, false).unwrap();
    /// let c = g.closeness_reachability(.., NeighborMode::All, None, true, None).unwrap();
    /// assert_eq!(c.scores, vec![1.0; 4]);
    /// assert_eq!(c.reachable_count, vec![1; 4]);
    /// assert!(!c.all_reachable);
    /// ```
    pub fn closeness_reachability<'a>(
        &self,
        vids: impl Into<VertexSelector<'a>>,
        mode: NeighborMode,
        weights: Option<&[f64]>,
        normalized: bool,
        cutoff: Option<f64>,
    ) -> Result<Closeness> {
        let vs = vids.into().to_raw()?;
        let w = weights_view(self, weights)?;
        let mut res = Vector::new();
        let mut reach = VectorInt::new();
        let mut all = false;
        igraph_call!(igraph_closeness_cutoff(
            self,
            &mut res,
            &mut reach,
            &mut all,
            vs.get(),
            mode.into(),
            ptr_of(&w),
            normalized,
            raw_cutoff(cutoff)
        ))?;
        Ok(Closeness {
            scores: res.into(),
            reachable_count: reach.into(),
            all_reachable: all,
        })
    }

    /// Harmonic centrality of the selected vertices (`igraph_harmonic_centrality`).
    ///
    /// The harmonic centrality of a vertex is the sum (or, if `normalized`,
    /// the mean over the other `|V| - 1` vertices) of the inverse distances to
    /// all other vertices; unreachable vertices contribute zero, which makes
    /// this measure well-behaved on disconnected graphs, unlike closeness.
    /// `mode` and `weights` are as in [`closeness`](Self::closeness).
    ///
    /// References: M. Marchiori and V. Latora, *Harmony in the small-world*,
    /// Physica A 285 (2000); S. Vigna and P. Boldi, *Axioms for Centrality*,
    /// Internet Mathematics 10 (2014).
    ///
    /// Time complexity: O(n|E|) unweighted, O(n|E|log|V| + |V|) weighted.
    ///
    /// See also [`global_efficiency`](Self::global_efficiency): the mean of
    /// the normalized harmonic centralities of all vertices is the global
    /// efficiency of the graph.
    ///
    /// Binds [`igraph_harmonic_centrality`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_harmonic_centrality).
    ///
    /// # Errors
    ///
    /// As for [`closeness`](Self::closeness).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Path 0 - 1 - 2: vertex 0 has 1/1 + 1/2 = 1.5.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// assert_eq!(g.harmonic_centrality(.., NeighborMode::All, None, false).unwrap(), vec![1.5, 2.0, 1.5]);
    /// ```
    pub fn harmonic_centrality<'a>(
        &self,
        vids: impl Into<VertexSelector<'a>>,
        mode: NeighborMode,
        weights: Option<&[f64]>,
        normalized: bool,
    ) -> Result<Vec<f64>> {
        let vs = vids.into().to_raw()?;
        let w = weights_view(self, weights)?;
        let mut res = Vector::new();
        igraph_call!(igraph_harmonic_centrality(
            self,
            &mut res,
            vs.get(),
            mode.into(),
            ptr_of(&w),
            normalized
        ))?;
        Ok(res.into())
    }

    /// Range-limited harmonic centrality (`igraph_harmonic_centrality_cutoff`).
    ///
    /// Like [`harmonic_centrality`](Self::harmonic_centrality), but vertices
    /// farther than `cutoff` contribute zero. `None` (or a negative value)
    /// computes the exact harmonic centrality. Note that normalization still
    /// divides by `|V| - 1`.
    ///
    /// Binds [`igraph_harmonic_centrality_cutoff`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_harmonic_centrality_cutoff).
    ///
    /// # Errors
    ///
    /// As for [`closeness`](Self::closeness).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // With cutoff 1 the harmonic centrality is just degree / (n - 1).
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// let h = g.harmonic_centrality_cutoff(.., NeighborMode::All, None, true, Some(1.0)).unwrap();
    /// assert_eq!(h, vec![0.5, 1.0, 0.5]);
    /// ```
    pub fn harmonic_centrality_cutoff<'a>(
        &self,
        vids: impl Into<VertexSelector<'a>>,
        mode: NeighborMode,
        weights: Option<&[f64]>,
        normalized: bool,
        cutoff: Option<f64>,
    ) -> Result<Vec<f64>> {
        let vs = vids.into().to_raw()?;
        let w = weights_view(self, weights)?;
        let mut res = Vector::new();
        igraph_call!(igraph_harmonic_centrality_cutoff(
            self,
            &mut res,
            vs.get(),
            mode.into(),
            ptr_of(&w),
            normalized,
            raw_cutoff(cutoff)
        ))?;
        Ok(res.into())
    }

    // ------------------------------------------------------------------
    // Betweenness
    // ------------------------------------------------------------------

    /// Betweenness centrality of the selected vertices (`igraph_betweenness`).
    ///
    /// The betweenness of a vertex `v` is the number of shortest paths passing
    /// through it; when two vertices are joined by several shortest paths,
    /// only the fraction of them passing through `v` is counted (Brandes'
    /// algorithm). With `weights`, weighted shortest paths are used.
    /// `directed` tells whether to follow edge directions (ignored for
    /// undirected graphs). If `normalized`, scores are divided by the number of
    /// vertex pairs: `n(n-1)` ordered pairs when directed paths are used,
    /// `n(n-1)/2` unordered pairs otherwise (note: *not* the `(n-1)(n-2)`
    /// convention of some textbooks).
    ///
    /// `vids` only selects which scores are returned: internally the
    /// betweenness of all vertices is computed. Time complexity: O(|V||E|).
    ///
    /// Reference: U. Brandes, *A faster algorithm for betweenness centrality*,
    /// J. Math. Sociol. 25(2), 163–177 (2001).
    ///
    /// See also [`get_all_shortest_paths`](Self::get_all_shortest_paths) to
    /// list the shortest paths that are being counted.
    ///
    /// Binds [`igraph_betweenness`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_betweenness).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) for an
    /// invalid vertex, [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue)
    /// for invalid weights.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Path 0 - 1 - 2 - 3: the inner vertices each lie on 2 shortest paths.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// assert_eq!(g.betweenness(None, .., false, false).unwrap(), vec![0.0, 2.0, 2.0, 0.0]);
    /// ```
    pub fn betweenness<'a>(
        &self,
        weights: Option<&[f64]>,
        vids: impl Into<VertexSelector<'a>>,
        directed: bool,
        normalized: bool,
    ) -> Result<Vec<f64>> {
        let vs = vids.into().to_raw()?;
        let w = weights_view(self, weights)?;
        let mut res = Vector::new();
        igraph_call!(igraph_betweenness(
            self,
            ptr_of(&w),
            &mut res,
            vs.get(),
            directed,
            normalized
        ))?;
        Ok(res.into())
    }

    /// Range-limited betweenness centrality (`igraph_betweenness_cutoff`).
    ///
    /// Like [`betweenness`](Self::betweenness), but only shortest paths of
    /// length at most `cutoff` are counted. `None` (or a negative value)
    /// computes the exact betweenness.
    ///
    /// Binds [`igraph_betweenness_cutoff`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_betweenness_cutoff).
    ///
    /// # Errors
    ///
    /// As for [`betweenness`](Self::betweenness).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // With cutoff 2 only the paths 0-1-2 and 1-2-3 pass through inner vertices.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// assert_eq!(g.betweenness_cutoff(None, .., false, false, Some(2.0)).unwrap(), vec![0.0, 1.0, 1.0, 0.0]);
    /// ```
    pub fn betweenness_cutoff<'a>(
        &self,
        weights: Option<&[f64]>,
        vids: impl Into<VertexSelector<'a>>,
        directed: bool,
        normalized: bool,
        cutoff: Option<f64>,
    ) -> Result<Vec<f64>> {
        let vs = vids.into().to_raw()?;
        let w = weights_view(self, weights)?;
        let mut res = Vector::new();
        igraph_call!(igraph_betweenness_cutoff(
            self,
            ptr_of(&w),
            &mut res,
            vs.get(),
            directed,
            normalized,
            raw_cutoff(cutoff)
        ))?;
        Ok(res.into())
    }

    /// Betweenness restricted to paths between a set of sources and a set of
    /// targets (`igraph_betweenness_subset`).
    ///
    /// Only the shortest paths starting in `sources` and ending in `targets`
    /// are counted. Scores are returned for `vids`. In undirected graphs each
    /// source-target pair contributes with weight 1/2, so that with
    /// `sources == targets` (where every pair is met from both ends) the
    /// result agrees with [`betweenness`](Self::betweenness); in particular
    /// selecting all vertices as sources and targets gives the ordinary
    /// betweenness. Normalization is not
    /// implemented by igraph for this variant, so the scores are always raw
    /// path counts. Time complexity: O(|S||E|), `S` being the source set.
    ///
    /// Binds [`igraph_betweenness_subset`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_betweenness_subset).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) for an
    /// invalid vertex in any of the selectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Directed path 0 -> 1 -> 2 -> 3, only paths from 0 to 3 count.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, true).unwrap();
    /// assert_eq!(g.betweenness_subset(None, 0, 3, .., true).unwrap(), vec![0.0, 1.0, 1.0, 0.0]);
    /// // Undirected, the single pair {0, 3} counts 1/2.
    /// let u = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// assert_eq!(u.betweenness_subset(None, 0, 3, .., false).unwrap(), vec![0.0, 0.5, 0.5, 0.0]);
    /// // With all vertices as sources and targets it is the usual betweenness.
    /// assert_eq!(u.betweenness_subset(None, .., .., .., false).unwrap(), u.betweenness(None, .., false, false).unwrap());
    /// ```
    pub fn betweenness_subset<'a>(
        &self,
        weights: Option<&[f64]>,
        sources: impl Into<VertexSelector<'a>>,
        targets: impl Into<VertexSelector<'a>>,
        vids: impl Into<VertexSelector<'a>>,
        directed: bool,
    ) -> Result<Vec<f64>> {
        let src = sources.into().to_raw()?;
        let tgt = targets.into().to_raw()?;
        let vs = vids.into().to_raw()?;
        let w = weights_view(self, weights)?;
        let mut res = Vector::new();
        igraph_call!(igraph_betweenness_subset(
            self,
            ptr_of(&w),
            &mut res,
            src.get(),
            tgt.get(),
            vs.get(),
            directed,
            false
        ))?;
        Ok(res.into())
    }

    /// Betweenness centrality of the selected edges (`igraph_edge_betweenness`).
    ///
    /// The betweenness of an edge is the number of shortest paths passing
    /// through it (fractionally, when there are several shortest paths).
    /// Parameters are as in [`betweenness`](Self::betweenness); `eids` only
    /// selects the returned scores. Removing the edge with the highest
    /// betweenness is the basic step of the Girvan–Newman community detection
    /// method, available as
    /// [`community_edge_betweenness`](Self::community_edge_betweenness).
    /// Time complexity: O(|V||E|).
    ///
    /// Binds [`igraph_edge_betweenness`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_edge_betweenness).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidEdgeId`](crate::ErrorKind::InvalidEdgeId) for an
    /// invalid edge, [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue)
    /// for invalid weights.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Two triangles joined by the bridge 2 - 3: all 9 cross pairs use it.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (3, 5)], 6, false).unwrap();
    /// let eb = g.edge_betweenness(None, .., false, false).unwrap();
    /// assert_eq!(eb[3], 9.0);
    /// ```
    pub fn edge_betweenness<'a>(
        &self,
        weights: Option<&[f64]>,
        eids: impl Into<EdgeSelector<'a>>,
        directed: bool,
        normalized: bool,
    ) -> Result<Vec<f64>> {
        let es = eids.into().to_raw()?;
        let w = weights_view(self, weights)?;
        let mut res = Vector::new();
        igraph_call!(igraph_edge_betweenness(
            self,
            ptr_of(&w),
            &mut res,
            es.get(),
            directed,
            normalized
        ))?;
        Ok(res.into())
    }

    /// Range-limited edge betweenness (`igraph_edge_betweenness_cutoff`).
    ///
    /// Like [`edge_betweenness`](Self::edge_betweenness), counting only
    /// shortest paths of length at most `cutoff` (`None` = no limit).
    ///
    /// Binds [`igraph_edge_betweenness_cutoff`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_edge_betweenness_cutoff).
    ///
    /// # Errors
    ///
    /// As for [`edge_betweenness`](Self::edge_betweenness).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // With cutoff 1, every edge only carries the path between its endpoints.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// assert_eq!(g.edge_betweenness_cutoff(None, .., false, false, Some(1.0)).unwrap(), vec![1.0; 3]);
    /// ```
    pub fn edge_betweenness_cutoff<'a>(
        &self,
        weights: Option<&[f64]>,
        eids: impl Into<EdgeSelector<'a>>,
        directed: bool,
        normalized: bool,
        cutoff: Option<f64>,
    ) -> Result<Vec<f64>> {
        let es = eids.into().to_raw()?;
        let w = weights_view(self, weights)?;
        let mut res = Vector::new();
        igraph_call!(igraph_edge_betweenness_cutoff(
            self,
            ptr_of(&w),
            &mut res,
            es.get(),
            directed,
            normalized,
            raw_cutoff(cutoff)
        ))?;
        Ok(res.into())
    }

    /// Edge betweenness restricted to paths between a set of sources and a
    /// set of targets (`igraph_edge_betweenness_subset`).
    ///
    /// Only shortest paths from `sources` to `targets` are counted; scores are
    /// returned for `eids`. As in [`betweenness_subset`](Self::betweenness_subset),
    /// in undirected graphs each source-target pair has weight 1/2.
    /// Normalization is not implemented by igraph for this variant. Time complexity: O(|S||E|).
    ///
    /// Binds [`igraph_edge_betweenness_subset`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_edge_betweenness_subset).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) for an
    /// invalid source or target vertex.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, true).unwrap();
    /// assert_eq!(g.edge_betweenness_subset(None, 0, 2, .., true).unwrap(), vec![1.0, 1.0, 0.0]);
    /// ```
    pub fn edge_betweenness_subset<'a>(
        &self,
        weights: Option<&[f64]>,
        sources: impl Into<VertexSelector<'a>>,
        targets: impl Into<VertexSelector<'a>>,
        eids: impl Into<EdgeSelector<'a>>,
        directed: bool,
    ) -> Result<Vec<f64>> {
        let src = sources.into().to_raw()?;
        let tgt = targets.into().to_raw()?;
        let es = eids.into().to_raw()?;
        let w = weights_view(self, weights)?;
        let mut res = Vector::new();
        igraph_call!(igraph_edge_betweenness_subset(
            self,
            ptr_of(&w),
            &mut res,
            src.get(),
            tgt.get(),
            es.get(),
            directed,
            false
        ))?;
        Ok(res.into())
    }

    // ------------------------------------------------------------------
    // PageRank
    // ------------------------------------------------------------------

    /// Google PageRank of the selected vertices (`igraph_pagerank`).
    ///
    /// The PageRank of a vertex is the fraction of time a random walker
    /// spends on it. The walker follows out-edges with probabilities
    /// proportional to their `weights` (which must be non-negative), and at
    /// each step restarts from a uniformly random vertex with probability
    /// `1 - damping`; it also restarts when stuck in a sink vertex. Scores of
    /// all vertices sum to one. In undirected graphs PageRank tends to be
    /// proportional to degree as the damping approaches 1, so it is mostly
    /// useful for directed graphs. See [`PageRankOptions`] for the damping,
    /// directedness and algorithm.
    ///
    /// `vids` only selects the returned scores: all of them are computed
    /// anyway. Time complexity: usually O(|E|).
    ///
    /// Reference: S. Brin and L. Page, *The Anatomy of a Large-Scale
    /// Hypertextual Web Search Engine*, WWW7 (1998).
    ///
    /// Binds [`igraph_pagerank`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_pagerank).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) for an
    /// invalid vertex, [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue)
    /// for a damping outside `[0, 1]` (or `NaN`) or invalid (negative, `NaN`
    /// or infinite) weights. These are checked on the Rust side: igraph lets
    /// a `NaN` damping and infinite weights through, and then either aborts
    /// the process (ARPACK) or returns `NaN` scores (PRPACK). Huge finite
    /// weights are rescaled (PageRank does not depend on the scale of the
    /// weights).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::centrality::PageRankOptions;
    ///
    /// // A directed cycle: by symmetry every vertex has PageRank 1/4.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], 4, true).unwrap();
    /// let pr = g.pagerank(None, .., &PageRankOptions::default()).unwrap();
    /// assert_eq!(pr.value, 1.0);
    /// assert!(pr.scores.iter().all(|&p| (p - 0.25).abs() < 1e-12));
    /// ```
    pub fn pagerank<'a>(
        &self,
        weights: Option<&[f64]>,
        vids: impl Into<VertexSelector<'a>>,
        options: &PageRankOptions,
    ) -> Result<EigenScores> {
        let vs = vids.into().to_raw()?;
        check_damping(options.damping)?;
        let (scaled, _) = spectral_weights(weights)?;
        let w = weights_view(self, scaled.as_deref().or(weights))?;
        let mut res = Vector::new();
        let mut value = 0.0;
        // ARPACK keeps thread-local state: refuse to nest it (e.g. from an
        // interruption handler running inside another ARPACK computation).
        let _arpack = if options.algo == PageRankAlgo::Arpack {
            Some(crate::linalg::ArpackGuard::enter()?)
        } else {
            None
        };
        igraph_call!(igraph_pagerank(
            self,
            ptr_of(&w),
            &mut res,
            &mut value,
            options.damping,
            options.directed,
            vs.get(),
            options.algo.into(),
            std::ptr::null_mut()
        ))?;
        Ok(EigenScores {
            scores: res.into(),
            value,
        })
    }

    /// Personalized PageRank with an arbitrary restart distribution
    /// (`igraph_personalized_pagerank`).
    ///
    /// Like [`pagerank`](Self::pagerank), but when the random walker restarts
    /// (with probability `1 - damping`, or when stuck in a sink), the new
    /// starting vertex is drawn from the distribution `reset` (one
    /// non-negative entry per vertex, not necessarily normalized) instead of
    /// uniformly. `reset = None` gives the ordinary PageRank.
    ///
    /// Binds [`igraph_personalized_pagerank`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_personalized_pagerank).
    ///
    /// # Errors
    ///
    /// As for [`pagerank`](Self::pagerank); also
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `reset`
    /// has the wrong length, negative entries or sums to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::centrality::PageRankOptions;
    ///
    /// // Always restarting from the leaf 0 of a path breaks its symmetry.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let reset = [1.0, 0.0, 0.0, 0.0];
    /// let pr = g.personalized_pagerank(None, Some(&reset), .., &PageRankOptions::default()).unwrap();
    /// assert!(pr.scores[0] > pr.scores[3] && pr.scores[1] > pr.scores[2]);
    /// ```
    pub fn personalized_pagerank<'a>(
        &self,
        weights: Option<&[f64]>,
        reset: Option<&[f64]>,
        vids: impl Into<VertexSelector<'a>>,
        options: &PageRankOptions,
    ) -> Result<EigenScores> {
        if let Some(r) = reset
            && r.len() != self.vcount()
        {
            return Err(Error::invalid(format!(
                "the reset vector has length {}, but the graph has {} vertices",
                r.len(),
                self.vcount()
            )));
        }
        let vs = vids.into().to_raw()?;
        check_damping(options.damping)?;
        let (scaled, _) = spectral_weights(weights)?;
        let w = weights_view(self, scaled.as_deref().or(weights))?;
        let r = reset.map(Vector::view);
        let mut res = Vector::new();
        let mut value = 0.0;
        // ARPACK keeps thread-local state: refuse to nest it (e.g. from an
        // interruption handler running inside another ARPACK computation).
        let _arpack = if options.algo == PageRankAlgo::Arpack {
            Some(crate::linalg::ArpackGuard::enter()?)
        } else {
            None
        };
        igraph_call!(igraph_personalized_pagerank(
            self,
            ptr_of(&w),
            &mut res,
            &mut value,
            ptr_of(&r),
            options.damping,
            options.directed,
            vs.get(),
            options.algo.into(),
            std::ptr::null_mut()
        ))?;
        Ok(EigenScores {
            scores: res.into(),
            value,
        })
    }

    /// Personalized PageRank restarting from a set of vertices
    /// (`igraph_personalized_pagerank_vs`).
    ///
    /// Like [`personalized_pagerank`](Self::personalized_pagerank), with the
    /// restart vertex chosen uniformly among `reset_vids` (duplicates count
    /// multiple times). Restarting always from a single vertex gives a
    /// "proximity to this vertex" measure, widely used for recommendations.
    ///
    /// Binds [`igraph_personalized_pagerank_vs`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_personalized_pagerank_vs).
    ///
    /// # Errors
    ///
    /// As for [`pagerank`](Self::pagerank); an empty or invalid `reset_vids`
    /// is an error too.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::centrality::PageRankOptions;
    ///
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let pr = g.personalized_pagerank_vs(None, 3, .., &PageRankOptions::default()).unwrap();
    /// assert!(pr.scores[3] > pr.scores[0]);
    /// ```
    pub fn personalized_pagerank_vs<'a>(
        &self,
        weights: Option<&[f64]>,
        reset_vids: impl Into<VertexSelector<'a>>,
        vids: impl Into<VertexSelector<'a>>,
        options: &PageRankOptions,
    ) -> Result<EigenScores> {
        let reset = reset_vids.into().to_raw()?;
        let vs = vids.into().to_raw()?;
        check_damping(options.damping)?;
        let (scaled, _) = spectral_weights(weights)?;
        let w = weights_view(self, scaled.as_deref().or(weights))?;
        let mut res = Vector::new();
        let mut value = 0.0;
        // ARPACK keeps thread-local state: refuse to nest it (e.g. from an
        // interruption handler running inside another ARPACK computation).
        let _arpack = if options.algo == PageRankAlgo::Arpack {
            Some(crate::linalg::ArpackGuard::enter()?)
        } else {
            None
        };
        igraph_call!(igraph_personalized_pagerank_vs(
            self,
            ptr_of(&w),
            &mut res,
            &mut value,
            reset.get(),
            options.damping,
            options.directed,
            vs.get(),
            options.algo.into(),
            std::ptr::null_mut()
        ))?;
        Ok(EigenScores {
            scores: res.into(),
            value,
        })
    }

    // ------------------------------------------------------------------
    // Spectral centralities
    // ------------------------------------------------------------------

    /// Eigenvector centrality of all vertices (`igraph_eigenvector_centrality`).
    ///
    /// The eigenvector centrality of a vertex is proportional to the sum of
    /// the centralities of its neighbors: it is the eigenvector of the
    /// adjacency matrix belonging to the largest positive eigenvalue, which is
    /// non-negative when weights are non-negative. Scores are scaled so that
    /// the maximum is 1 (unless all are zero). In undirected graphs a self-loop
    /// counts twice on the diagonal; weights of parallel edges add up.
    ///
    /// `mode` matters for directed graphs only: [`NeighborMode::Out`] (the
    /// standard choice) gives each vertex the sum of the centralities of the
    /// vertices *pointing to it* (left eigenvector); [`NeighborMode::In`] the
    /// sum over the vertices it points to; [`NeighborMode::All`] ignores
    /// directions.
    ///
    /// The measure is meaningful only for (strongly) connected graphs: in a
    /// disconnected undirected graph all but one component typically get
    /// zeros (igraph emits a warning). Directed acyclic graphs have no positive
    /// eigenvalue: the returned [`value`](EigenScores::value) is then zero.
    /// For directed graphs, consider
    /// [`hub_and_authority_scores`](Self::hub_and_authority_scores). Time
    /// complexity: usually O(|V| + |E|).
    ///
    /// See also [`connected_components`](Self::connected_components) to split
    /// a disconnected graph first (and [`is_dag`](Self::is_dag) to detect the
    /// acyclic case), and [`eigen_adjacency`](Self::eigen_adjacency) for other
    /// eigenpairs of the adjacency matrix.
    ///
    /// Binds [`igraph_eigenvector_centrality`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_eigenvector_centrality).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for invalid
    /// weights (wrong length, `NaN` or infinite: igraph does not check the
    /// latter and ARPACK would abort the process);
    /// [`ErrorKind::Arpack`](crate::ErrorKind::Arpack) if the eigensolver
    /// fails. Weights so large that the iterations could overflow (absolute
    /// sum above `1e150`) are divided by a power of two before the call, and
    /// the eigenvalue is scaled back (it may then be infinite, when it is not
    /// representable): the scores do not depend on the scale of the weights.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Weighted star with center 0 and weights 1..9 (igraph's example).
    /// let edges: Vec<(i64, i64)> = (1..10).map(|i| (0, i)).collect();
    /// let g = Graph::from_edges(&edges, 10, false).unwrap();
    /// let w: Vec<f64> = (1..10).map(f64::from).collect();
    /// let ec = g.eigenvector_centrality(NeighborMode::Out, Some(&w)).unwrap();
    /// assert!((ec.value - 16.8819).abs() < 1e-4);
    /// assert_eq!(ec.scores[0], 1.0);
    /// assert!((ec.scores[1] - 0.0592349).abs() < 1e-6);
    /// ```
    pub fn eigenvector_centrality(
        &self,
        mode: NeighborMode,
        weights: Option<&[f64]>,
    ) -> Result<EigenScores> {
        let (scaled, scale) = spectral_weights(weights)?;
        let w = weights_view(self, scaled.as_deref().or(weights))?;
        let mut res = Vector::new();
        let mut value = 0.0;
        // ARPACK keeps thread-local state: refuse to nest it.
        let _arpack = crate::linalg::ArpackGuard::enter()?;
        igraph_call!(igraph_eigenvector_centrality(
            self,
            &mut res,
            &mut value,
            mode.into(),
            ptr_of(&w),
            std::ptr::null_mut()
        ))?;
        Ok(EigenScores {
            scores: res.into(),
            value: value * scale,
        })
    }

    /// Kleinberg's hub and authority scores (HITS)
    /// (`igraph_hub_and_authority_scores`).
    ///
    /// The authority score of a vertex is proportional to the sum of the hub
    /// scores of the vertices pointing to it, and its hub score to the sum of
    /// the authority scores of the vertices it points to. Hubs and authorities
    /// are the principal eigenvectors of `A Aᵀ` and `Aᵀ A`; igraph guarantees
    /// that the two returned vectors match (`h = A a`, `a = Aᵀ h`, up to
    /// scaling) even when the eigenvalue is degenerate. Both are scaled to
    /// have maximum 1. Edge weights should be non-negative (igraph warns
    /// otherwise).
    ///
    /// In undirected graphs both vectors coincide with the [eigenvector
    /// centrality](Self::eigenvector_centrality) (computed by it directly,
    /// with a warning) and [`value`](HubAuthority::value) is the square of its
    /// eigenvalue. A graph without edges gives all-ones scores and value 0.
    ///
    /// In extremely sparse graphs, where no single connected component
    /// dominates the graphs of `A Aᵀ` and `Aᵀ A`, the solution is not unique
    /// and many scores are zero: igraph then emits a warning (retrieve it with
    /// [`take_warnings`](crate::error::take_warnings)) when *more than 30%* of
    /// the hub scores are zero (below `10 ε` in absolute value), on directed
    /// graphs with at least 10 vertices. (In igraph 1.0.0 a rounding bug made
    /// it warn for any zero score; this was fixed in 1.0.1.)
    ///
    /// Time complexity: usually O(|V|).
    ///
    /// Reference: J. Kleinberg, *Authoritative sources in a hyperlinked
    /// environment*, J. ACM 46 (1999).
    ///
    /// See also [`pagerank`](Self::pagerank), another random-walk based
    /// ranking for directed graphs.
    ///
    /// Binds [`igraph_hub_and_authority_scores`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_hub_and_authority_scores).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for invalid
    /// weights (wrong length, `NaN` or infinite: igraph does not check the
    /// latter and ARPACK would abort the process);
    /// [`ErrorKind::Arpack`](crate::ErrorKind::Arpack) if the eigensolver
    /// fails. Weights so large that the iterations could overflow (absolute
    /// sum above `1e150`) are divided by a power of two before the call, and
    /// the eigenvalue is scaled back (it may then be infinite, when it is not
    /// representable): the scores do not depend on the scale of the weights.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Two pages linking to a third one: two perfect hubs, one authority.
    /// let g = Graph::from_edges(&[(0, 2), (1, 2)], 3, true).unwrap();
    /// let hits = g.hub_and_authority_scores(None).unwrap();
    /// assert_eq!(hits.hubs, vec![1.0, 1.0, 0.0]);
    /// assert_eq!(hits.authorities, vec![0.0, 0.0, 1.0]);
    /// assert!((hits.value - 2.0).abs() < 1e-9);
    /// ```
    pub fn hub_and_authority_scores(&self, weights: Option<&[f64]>) -> Result<HubAuthority> {
        let (scaled, scale) = spectral_weights(weights)?;
        let w = weights_view(self, scaled.as_deref().or(weights))?;
        let mut hubs = Vector::new();
        let mut auth = Vector::new();
        // Always request `value`: for undirected graphs igraph (1.0.0 and
        // 1.0.1) squares `*value` without checking it for NULL.
        let mut value = 0.0;
        // ARPACK keeps thread-local state: refuse to nest it.
        let _arpack = crate::linalg::ArpackGuard::enter()?;
        igraph_call!(igraph_hub_and_authority_scores(
            self,
            &mut hubs,
            &mut auth,
            &mut value,
            ptr_of(&w),
            std::ptr::null_mut()
        ))?;
        Ok(HubAuthority {
            hubs: hubs.into(),
            authorities: auth.into(),
            // The eigenvalue of `A Aᵀ` scales quadratically.
            value: value * scale * scale,
        })
    }

    // ------------------------------------------------------------------
    // Structural holes and convergence
    // ------------------------------------------------------------------

    /// Burt's constraint scores of the selected vertices (`igraph_constraint`).
    ///
    /// Constraint measures how much a vertex's ego network is closed: it is
    /// high when ego has few, or mutually strongly related (redundant),
    /// contacts, and low for vertices bridging *structural holes*. Formally
    /// `C[i] = Σ_{j ≠ i} (p[i,j] + Σ_{q ≠ i,j} p[i,q] p[q,j])²` over the
    /// neighbors `j`, with proportional tie strengths
    /// `p[i,j] = (a[i,j] + a[j,i]) / Σ_k (a[i,k] + a[k,i])`, `a` being the
    /// (weighted) adjacency matrix. It is undefined (NaN) for isolated
    /// vertices. Time complexity: O(|V| + |E| + n d²), `d` the average degree.
    ///
    /// Reference: R. S. Burt, *Structural holes and good ideas*, American
    /// Journal of Sociology 110, 349–399 (2004).
    ///
    /// See also [`transitivity_local_undirected`](Self::transitivity_local_undirected),
    /// the local clustering coefficient, a related measure of ego-network
    /// closure.
    ///
    /// Binds [`igraph_constraint`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_constraint).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) for an
    /// invalid vertex, [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue)
    /// for weights of the wrong length.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // In a path, the end vertices are fully constrained by their only contact.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// assert_eq!(g.constraint(.., None).unwrap(), vec![1.0, 0.5, 0.5, 1.0]);
    /// ```
    pub fn constraint<'a>(
        &self,
        vids: impl Into<VertexSelector<'a>>,
        weights: Option<&[f64]>,
    ) -> Result<Vec<f64>> {
        let vs = vids.into().to_raw()?;
        let w = weights_view(self, weights)?;
        let mut res = Vector::new();
        igraph_call!(igraph_constraint(self, &mut res, vs.get(), ptr_of(&w)))?;
        Ok(res.into())
    }

    /// Convergence degree of every edge (`igraph_convergence_degree`).
    ///
    /// The *input set* of an edge is the set of vertices where the shortest
    /// paths passing through it originate, the *output set* where they
    /// terminate. The convergence degree is `(|in| − |out|) / (|in| + |out|)`,
    /// in `(-1, 1)`: positive values mark *convergent* edges (paths coming
    /// from many vertices and going to few), negative ones *divergent* edges.
    /// In undirected graphs the edge is oriented arbitrarily and the absolute
    /// value is reported. Time complexity: O(|V||E|).
    ///
    /// Binds [`igraph_convergence_degree`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_convergence_degree).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // An in-star 1,2,3,4 -> 0 followed by 0 -> 5 (igraph's unit test).
    /// let g = Graph::from_edges(&[(1, 0), (2, 0), (3, 0), (4, 0), (0, 5)], 6, true).unwrap();
    /// let cd = g.convergence_degree().unwrap();
    /// assert!((cd.result[4] - 2.0 / 3.0).abs() < 1e-9);
    /// ```
    pub fn convergence_degree(&self) -> Result<ConvergenceDegree> {
        let mut result = Vector::new();
        let mut ins = Vector::new();
        let mut outs = Vector::new();
        igraph_call!(igraph_convergence_degree(
            self,
            &mut result,
            &mut ins,
            &mut outs
        ))?;
        Ok(ConvergenceDegree {
            result: result.into(),
            ins: ins.into(),
            outs: outs.into(),
        })
    }

    // ------------------------------------------------------------------
    // Centralization
    // ------------------------------------------------------------------

    /// Degree centralization of the graph (`igraph_centralization_degree`).
    ///
    /// Computes the degrees of all vertices (with `mode` for directed graphs
    /// and the `loops` counting convention, see [`Loops`]) and their
    /// [`centralization`] index, normalized by the theoretical maximum (the
    /// star) if `normalized` is true. Time complexity: O(|V| + |E|).
    ///
    /// See also [`degree`](Self::degree) and [`maxdegree`](Self::maxdegree).
    ///
    /// Binds [`igraph_centralization_degree`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_centralization_degree).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // A cycle is perfectly decentralized.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], 4, false).unwrap();
    /// let c = g.centralization_degree(NeighborMode::All, Loops::None, true).unwrap();
    /// assert_eq!(c.scores, vec![2.0; 4]);
    /// assert_eq!(c.centralization, 0.0);
    /// assert_eq!(c.theoretical_max, 6.0);
    /// ```
    pub fn centralization_degree(
        &self,
        mode: NeighborMode,
        loops: Loops,
        normalized: bool,
    ) -> Result<Centralization> {
        let mut res = Vector::new();
        let (mut cent, mut tmax) = (0.0, 0.0);
        igraph_call!(igraph_centralization_degree(
            self,
            &mut res,
            mode.into(),
            loops.into(),
            &mut cent,
            &mut tmax,
            normalized
        ))?;
        Ok(Centralization {
            scores: res.into(),
            centralization: cent,
            theoretical_max: tmax,
        })
    }

    /// Theoretical maximum of degree centralization for graphs with the
    /// size and directedness of `self` (`igraph_centralization_degree_tmax`).
    ///
    /// `mode` is ignored for undirected graphs. See the free function
    /// [`centralization_degree_tmax`] to specify the number of vertices
    /// directly.
    ///
    /// Binds [`igraph_centralization_degree_tmax`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_centralization_degree_tmax).
    pub fn centralization_degree_tmax(&self, mode: NeighborMode, loops: Loops) -> Result<f64> {
        let mut res = 0.0;
        igraph_call!(igraph_centralization_degree_tmax(
            self,
            0,
            mode.into(),
            loops.into(),
            &mut res
        ))?;
        Ok(res)
    }

    /// Betweenness centralization of the graph
    /// (`igraph_centralization_betweenness`).
    ///
    /// Computes the (unweighted) [`betweenness`](Self::betweenness) of all
    /// vertices, following directions if `directed`, and its
    /// [`centralization`] index, normalized by the theoretical maximum (the
    /// star) if `normalized`. Time complexity: O(|V||E|).
    ///
    /// Binds [`igraph_centralization_betweenness`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_centralization_betweenness).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// let star = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (0, 4)], 5, false).unwrap();
    /// let c = star.centralization_betweenness(false, true).unwrap();
    /// assert_eq!(c.centralization, 1.0);
    /// ```
    pub fn centralization_betweenness(
        &self,
        directed: bool,
        normalized: bool,
    ) -> Result<Centralization> {
        let mut res = Vector::new();
        let (mut cent, mut tmax) = (0.0, 0.0);
        igraph_call!(igraph_centralization_betweenness(
            self, &mut res, directed, &mut cent, &mut tmax, normalized
        ))?;
        Ok(Centralization {
            scores: res.into(),
            centralization: cent,
            theoretical_max: tmax,
        })
    }

    /// Theoretical maximum of betweenness centralization for graphs with the
    /// size and directedness of `self`
    /// (`igraph_centralization_betweenness_tmax`).
    ///
    /// `directed` is ignored for undirected graphs. See the free function
    /// [`centralization_betweenness_tmax`] to give the number of vertices.
    ///
    /// Binds [`igraph_centralization_betweenness_tmax`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_centralization_betweenness_tmax).
    pub fn centralization_betweenness_tmax(&self, directed: bool) -> Result<f64> {
        let mut res = 0.0;
        igraph_call!(igraph_centralization_betweenness_tmax(
            self, 0, directed, &mut res
        ))?;
        Ok(res)
    }

    /// Closeness centralization of the graph
    /// (`igraph_centralization_closeness`).
    ///
    /// Computes the (unweighted, normalized) [`closeness`](Self::closeness)
    /// of all vertices, using `mode` for directed graphs, and its
    /// [`centralization`] index, normalized by the theoretical maximum (the
    /// star) if `normalized`. Time complexity: O(|V||E|).
    ///
    /// Binds [`igraph_centralization_closeness`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_centralization_closeness).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// let star = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (0, 4)], 5, false).unwrap();
    /// let c = star.centralization_closeness(NeighborMode::All, true).unwrap();
    /// assert!((c.centralization - 1.0).abs() < 1e-12);
    /// ```
    pub fn centralization_closeness(
        &self,
        mode: NeighborMode,
        normalized: bool,
    ) -> Result<Centralization> {
        let mut res = Vector::new();
        let (mut cent, mut tmax) = (0.0, 0.0);
        igraph_call!(igraph_centralization_closeness(
            self,
            &mut res,
            mode.into(),
            &mut cent,
            &mut tmax,
            normalized
        ))?;
        Ok(Centralization {
            scores: res.into(),
            centralization: cent,
            theoretical_max: tmax,
        })
    }

    /// Theoretical maximum of closeness centralization for graphs with the
    /// size and directedness of `self` (`igraph_centralization_closeness_tmax`).
    ///
    /// `mode` is ignored for undirected graphs. See the free function
    /// [`centralization_closeness_tmax`] to give the number of vertices.
    ///
    /// Binds [`igraph_centralization_closeness_tmax`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_centralization_closeness_tmax).
    pub fn centralization_closeness_tmax(&self, mode: NeighborMode) -> Result<f64> {
        let mut res = 0.0;
        igraph_call!(igraph_centralization_closeness_tmax(
            self,
            0,
            mode.into(),
            &mut res
        ))?;
        Ok(res)
    }

    /// Eigenvector centralization of the graph
    /// (`igraph_centralization_eigenvector_centrality`).
    ///
    /// Computes the (unweighted) [eigenvector
    /// centrality](Self::eigenvector_centrality) of all vertices, scaled so
    /// that the maximum is 1, and its [`centralization`] index, normalized by
    /// the theoretical maximum if `normalized`. Note that eigenvector scores
    /// have no natural scale, so the centralization depends on the choice of
    /// scaling by the maximum (∞-norm). The most centralized undirected graph
    /// is a single edge (plus isolated vertices). `mode` is as in
    /// [`eigenvector_centrality`](Self::eigenvector_centrality).
    ///
    /// Binds [`igraph_centralization_eigenvector_centrality`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_centralization_eigenvector_centrality).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Arpack`](crate::ErrorKind::Arpack) if the eigensolver fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// let g = Graph::from_edges(&[(0, 1)], 10, false).unwrap();
    /// let c = g.centralization_eigenvector_centrality(NeighborMode::All, true).unwrap();
    /// assert!((c.centralization - 1.0).abs() < 1e-9);
    /// ```
    pub fn centralization_eigenvector_centrality(
        &self,
        mode: NeighborMode,
        normalized: bool,
    ) -> Result<EigenvectorCentralization> {
        let mut res = Vector::new();
        let (mut value, mut cent, mut tmax) = (0.0, 0.0, 0.0);
        // ARPACK keeps thread-local state: refuse to nest it.
        let _arpack = crate::linalg::ArpackGuard::enter()?;
        igraph_call!(igraph_centralization_eigenvector_centrality(
            self,
            &mut res,
            &mut value,
            mode.into(),
            std::ptr::null_mut(),
            &mut cent,
            &mut tmax,
            normalized
        ))?;
        Ok(EigenvectorCentralization {
            scores: res.into(),
            value,
            centralization: cent,
            theoretical_max: tmax,
        })
    }

    /// Theoretical maximum of eigenvector centralization for graphs with the
    /// size and directedness of `self`
    /// (`igraph_centralization_eigenvector_centrality_tmax`).
    ///
    /// `mode` is ignored for undirected graphs. See the free function
    /// [`centralization_eigenvector_centrality_tmax`] to give the number of
    /// vertices.
    ///
    /// Binds [`igraph_centralization_eigenvector_centrality_tmax`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_centralization_eigenvector_centrality_tmax).
    pub fn centralization_eigenvector_centrality_tmax(&self, mode: NeighborMode) -> Result<f64> {
        let mut res = 0.0;
        igraph_call!(igraph_centralization_eigenvector_centrality_tmax(
            self,
            0,
            mode.into(),
            &mut res
        ))?;
        Ok(res)
    }

    // ------------------------------------------------------------------
    // Local scan statistics
    // ------------------------------------------------------------------

    /// Local scan statistic with `k = 0` (`igraph_local_scan_0`).
    ///
    /// By convention the 0-scan of a vertex is its degree (`weights = None`)
    /// or its strength (the sum of incident edge weights, as computed by
    /// [`strength`](Self::strength)). `mode` selects out-, in- or all edges in
    /// directed graphs.
    ///
    /// Reference: C. E. Priebe et al., *Scan Statistics on Enron Graphs*,
    /// Comput. Math. Organ. Theory (2005).
    ///
    /// Binds [`igraph_local_scan_0`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_local_scan_0).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// assert_eq!(g.local_scan_0(None, NeighborMode::All).unwrap(), vec![1.0, 2.0, 1.0]);
    /// assert_eq!(g.local_scan_0(Some(&[0.5, 2.0]), NeighborMode::All).unwrap(), vec![0.5, 2.5, 2.0]);
    /// ```
    pub fn local_scan_0(&self, weights: Option<&[f64]>, mode: NeighborMode) -> Result<Vec<f64>> {
        let w = weights_view(self, weights)?;
        let mut res = Vector::new();
        igraph_call!(igraph_local_scan_0(self, &mut res, ptr_of(&w), mode.into()))?;
        Ok(res.into())
    }

    /// "Them" local scan statistic with `k = 0` (`igraph_local_scan_0_them`).
    ///
    /// Neighborhoods are taken from `self` ("us"), but edges are counted (or
    /// their `weights_them` summed) in the graph `them`, which must have the
    /// same vertices and directedness: the result is, for every vertex, the
    /// number of `them` edges incident to it that also connect it in `us`.
    /// Useful to compare two snapshots of a network.
    ///
    /// Binds [`igraph_local_scan_0_them`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_local_scan_0_them).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the two
    /// graphs differ in size or directedness, or `weights_them` does not have
    /// one entry per edge of `them`.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// let us = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// let them = Graph::from_edges(&[(0, 1), (0, 2)], 3, false).unwrap();
    /// assert_eq!(us.local_scan_0_them(&them, None, NeighborMode::All).unwrap(), vec![1.0, 1.0, 0.0]);
    /// ```
    pub fn local_scan_0_them(
        &self,
        them: &Graph,
        weights_them: Option<&[f64]>,
        mode: NeighborMode,
    ) -> Result<Vec<f64>> {
        check_them(self, them)?;
        let w = weights_view(them, weights_them)?;
        let mut res = Vector::new();
        igraph_call!(igraph_local_scan_0_them(
            self,
            them,
            &mut res,
            ptr_of(&w),
            mode.into()
        ))?;
        Ok(res.into())
    }

    /// Local scan statistic with `k = 1` (`igraph_local_scan_1_ecount`).
    ///
    /// For every vertex, the number of edges (or the sum of their weights) in
    /// the subgraph induced by its closed 1-neighborhood (the vertex and its
    /// neighbors along `mode`). For undirected simple graphs this is
    /// `degree + number of triangles` through the vertex (see
    /// [`count_adjacent_triangles`](Self::count_adjacent_triangles)).
    ///
    /// Binds [`igraph_local_scan_1_ecount`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_local_scan_1_ecount).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // A triangle with a pendant vertex 3 attached to 2.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 2), (2, 3)], 4, false).unwrap();
    /// assert_eq!(g.local_scan_1_ecount(None, NeighborMode::All).unwrap(), vec![3.0, 3.0, 4.0, 1.0]);
    /// ```
    pub fn local_scan_1_ecount(
        &self,
        weights: Option<&[f64]>,
        mode: NeighborMode,
    ) -> Result<Vec<f64>> {
        let w = weights_view(self, weights)?;
        let mut res = Vector::new();
        igraph_call!(igraph_local_scan_1_ecount(
            self,
            &mut res,
            ptr_of(&w),
            mode.into()
        ))?;
        Ok(res.into())
    }

    /// "Them" local scan statistic with `k = 1`
    /// (`igraph_local_scan_1_ecount_them`).
    ///
    /// For every vertex, the number of edges of `them` (or the sum of their
    /// `weights_them`) inside the closed 1-neighborhood of the vertex in
    /// `self`. The graphs must have the same vertices and directedness.
    ///
    /// Binds [`igraph_local_scan_1_ecount_them`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_local_scan_1_ecount_them).
    ///
    /// # Errors
    ///
    /// As for [`local_scan_0_them`](Self::local_scan_0_them).
    pub fn local_scan_1_ecount_them(
        &self,
        them: &Graph,
        weights_them: Option<&[f64]>,
        mode: NeighborMode,
    ) -> Result<Vec<f64>> {
        check_them(self, them)?;
        let w = weights_view(them, weights_them)?;
        let mut res = Vector::new();
        self.with_fresh_multi_cache(|| {
            igraph_call!(igraph_local_scan_1_ecount_them(
                self,
                them,
                &mut res,
                ptr_of(&w),
                mode.into()
            ))
        })?;
        Ok(res.into())
    }

    /// Local scan statistic for `k`-neighborhoods (`igraph_local_scan_k_ecount`).
    ///
    /// For every vertex, the number of edges (or the sum of their weights) in
    /// the subgraph induced by the vertices within distance `k` along `mode`.
    /// `k = 0` is special-cased to [`local_scan_0`](Self::local_scan_0) (degree
    /// or strength), `k = 1` to [`local_scan_1_ecount`](Self::local_scan_1_ecount).
    /// The neighborhoods themselves are given by
    /// [`neighborhood`](Self::neighborhood) with `order = k`.
    ///
    /// Binds [`igraph_local_scan_k_ecount`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_local_scan_k_ecount).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for weights
    /// of the wrong length.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // In a path of 5 vertices, the 2-neighborhood of the center is everything.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4)], 5, false).unwrap();
    /// assert_eq!(g.local_scan_k_ecount(2, None, NeighborMode::All).unwrap(), vec![2.0, 3.0, 4.0, 3.0, 2.0]);
    /// ```
    pub fn local_scan_k_ecount(
        &self,
        k: usize,
        weights: Option<&[f64]>,
        mode: NeighborMode,
    ) -> Result<Vec<f64>> {
        let k = igraph_int_t::try_from(k).map_err(|_| Error::invalid("k is too large"))?;
        let w = weights_view(self, weights)?;
        let mut res = Vector::new();
        igraph_call!(igraph_local_scan_k_ecount(
            self,
            k,
            &mut res,
            ptr_of(&w),
            mode.into()
        ))?;
        Ok(res.into())
    }

    /// "Them" local scan statistic for `k`-neighborhoods
    /// (`igraph_local_scan_k_ecount_them`).
    ///
    /// For every vertex, the number of edges of `them` (or the sum of their
    /// `weights_them`) inside the `k`-neighborhood of the vertex computed in
    /// `self`. The graphs must have the same vertices and directedness.
    ///
    /// Binds [`igraph_local_scan_k_ecount_them`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_local_scan_k_ecount_them).
    ///
    /// # Errors
    ///
    /// As for [`local_scan_0_them`](Self::local_scan_0_them).
    pub fn local_scan_k_ecount_them(
        &self,
        them: &Graph,
        k: usize,
        weights_them: Option<&[f64]>,
        mode: NeighborMode,
    ) -> Result<Vec<f64>> {
        check_them(self, them)?;
        let k = igraph_int_t::try_from(k).map_err(|_| Error::invalid("k is too large"))?;
        let w = weights_view(them, weights_them)?;
        let mut res = Vector::new();
        self.with_fresh_multi_cache(|| {
            igraph_call!(igraph_local_scan_k_ecount_them(
                self,
                them,
                k,
                &mut res,
                ptr_of(&w),
                mode.into()
            ))
        })?;
        Ok(res.into())
    }

    /// Edge counts in the subgraphs induced by arbitrary vertex subsets
    /// (`igraph_local_scan_subset_ecount`).
    ///
    /// Returns, for each subset, the number of edges (or the sum of their
    /// weights) of the subgraph it induces. Multi-edges and self-loops count
    /// (a loop counts once). Each subset should be a *set*: igraph does not
    /// deduplicate, so a repeated vertex makes its incident edges count more
    /// than once. Without weights, the count for a duplicate-free subset
    /// equals the edge count of the corresponding
    /// [`induced_subgraph`](Self::induced_subgraph). Time complexity:
    /// O(Σ_S Σ_{v ∈ S} deg(v)).
    ///
    /// Binds [`igraph_local_scan_subset_ecount`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_local_scan_subset_ecount).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for an
    /// invalid vertex in a subset (igraph reports it as `IGRAPH_EINVAL`, not as
    /// an invalid vertex id) or for weights of the wrong length.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    ///
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 2), (2, 3)], 4, false).unwrap();
    /// let counts = g.local_scan_subset_ecount(None, &[vec![0, 1, 2], vec![2, 3], vec![]]).unwrap();
    /// assert_eq!(counts, vec![3.0, 1.0, 0.0]);
    /// ```
    pub fn local_scan_subset_ecount<S: AsRef<[VertexId]>>(
        &self,
        weights: Option<&[f64]>,
        subsets: &[S],
    ) -> Result<Vec<f64>> {
        let w = weights_view(self, weights)?;
        let list: VectorIntList = subsets.iter().map(|s| s.as_ref()).collect();
        let mut res = Vector::new();
        igraph_call!(igraph_local_scan_subset_ecount(
            self,
            &mut res,
            ptr_of(&w),
            &list
        ))?;
        Ok(res.into())
    }

    /// Edge counts in precomputed neighborhoods, one per vertex
    /// (`igraph_local_scan_neighborhood_ecount`).
    ///
    /// Like [`local_scan_subset_ecount`](Self::local_scan_subset_ecount), but
    /// `neighborhoods` must contain exactly one vertex set per vertex of the
    /// graph. The C documentation marks this function as deprecated in favor
    /// of `igraph_local_scan_subset_ecount` since igraph 0.10, hence the
    /// Rust `#[deprecated]` attribute.
    ///
    /// Binds [`igraph_local_scan_neighborhood_ecount`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_local_scan_neighborhood_ecount).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the
    /// number of neighborhoods differs from the number of vertices, or as for
    /// [`local_scan_subset_ecount`](Self::local_scan_subset_ecount).
    #[deprecated(
        since = "1.0.1",
        note = "deprecated in igraph 0.10: use `local_scan_subset_ecount`"
    )]
    pub fn local_scan_neighborhood_ecount<S: AsRef<[VertexId]>>(
        &self,
        weights: Option<&[f64]>,
        neighborhoods: &[S],
    ) -> Result<Vec<f64>> {
        let w = weights_view(self, weights)?;
        let list: VectorIntList = neighborhoods.iter().map(|s| s.as_ref()).collect();
        let mut res = Vector::new();
        igraph_call!(igraph_local_scan_neighborhood_ecount(
            self,
            &mut res,
            ptr_of(&w),
            &list
        ))?;
        Ok(res.into())
    }
}

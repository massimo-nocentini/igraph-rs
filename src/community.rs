//! Community detection, modularity, partition comparison and hierarchical
//! random graphs (`igraph_community.h`, `igraph_hrg.h`).
//!
//! *Community detection* clusters the vertices of a network into groups that
//! are densely connected internally and sparsely connected with each other.
//! A clustering is represented, as everywhere in igraph, by a **membership
//! vector**: `membership[v]` is the community id of vertex `v`, ids being
//! numbered from zero. Hierarchical methods also return a **dendrogram** as a
//! list of *merges* `(a, b)`: the `i`-th merge joins the clusters `a` and `b`
//! into a new cluster with id `n + i` (`n` being the number of vertices).
//!
//! Good introductions to the topic are S. Fortunato, *Community Detection in
//! Graphs*, Physics Reports 486 (2010) and S. Fortunato and D. Hric,
//! *Community Detection in Networks: A User Guide*, Physics Reports 659 (2016).
//!
//! # Examples
//!
//! Two 5-cliques joined by a single edge are split into their cliques by
//! every reasonable method:
//!
//! ```
//! use igraph::prelude::*;
//! use igraph::community::compare_communities;
//!
//! let mut edges = vec![];
//! for base in [0, 5] {
//!     for i in 0..5 {
//!         for j in i + 1..5 {
//!             edges.push((base + i, base + j));
//!         }
//!     }
//! }
//! edges.push((0, 5));
//! let g = Graph::from_edges(&edges, 10, false).unwrap();
//!
//! let louvain = g.community_multilevel(None, 1.0).unwrap();
//! assert_eq!(louvain.membership, vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
//! assert!((louvain.modularity() - 0.4524).abs() < 1e-4);
//!
//! let greedy = g.community_fastgreedy(None).unwrap();
//! let nmi = compare_communities(&louvain.membership, &greedy.membership,
//!                               CommunityComparison::Nmi).unwrap();
//! assert!((nmi - 1.0).abs() < 1e-12);
//! ```
//!
//! A typical workflow on a real network: Zachary's karate club (from
//! [`Graph::famous`]) is clustered with a seeded Leiden run, and the
//! communities are then collapsed into a weighted "community graph" with
//! [`Graph::contract_vertices`], whose modularity for the singleton partition
//! is, by definition, the modularity of the clustering:
//!
//! ```
//! use igraph::prelude::*;
//! use igraph::community::{LeidenObjective, LeidenOptions};
//!
//! let karate = Graph::famous("Zachary").unwrap();
//! rng::seed(42).unwrap();
//! let leiden = karate
//!     .community_leiden_simple(None, LeidenObjective::Modularity,
//!                              &LeidenOptions::default().with_iterations(None))
//!     .unwrap();
//! assert_eq!(leiden.nb_clusters, 4);
//! assert!((leiden.quality - 0.4198).abs() < 1e-4); // the known optimum
//!
//! let mut quotient = karate.clone();
//! quotient.contract_vertices(&leiden.membership).unwrap();
//! let singletons: Vec<i64> = (0..leiden.nb_clusters as i64).collect();
//! let q = quotient.modularity(&singletons, None, 1.0, false).unwrap();
//! assert!((q - leiden.quality).abs() < 1e-12);
//! ```
//!
//! # Provided functionality
//!
//! | Rust API | C function | What |
//! |---|---|---|
//! | [`Graph::community_multilevel`] | `igraph_community_multilevel` | Louvain modularity optimization |
//! | [`Graph::community_leiden`] | `igraph_community_leiden` | Leiden, raw vertex weights |
//! | [`Graph::community_leiden_simple`] | `igraph_community_leiden_simple` | Leiden with modularity / CPM / ER objective |
//! | [`Graph::community_fastgreedy`] | `igraph_community_fastgreedy` | Clauset–Newman–Moore greedy agglomeration |
//! | [`Graph::community_walktrap`] | `igraph_community_walktrap` | random-walk distances (Pons–Latapy) |
//! | [`Graph::community_edge_betweenness`] | `igraph_community_edge_betweenness` | Girvan–Newman divisive method |
//! | [`Graph::community_eb_get_merges`] | `igraph_community_eb_get_merges` | dendrogram from an edge removal order |
//! | [`Graph::community_leading_eigenvector`], [`Graph::community_leading_eigenvector_with`] | `igraph_community_leading_eigenvector` | Newman's spectral method |
//! | [`Graph::community_spinglass`] | `igraph_community_spinglass` | Reichardt–Bornholdt Potts model |
//! | [`Graph::community_spinglass_single`] | `igraph_community_spinglass_single` | community of a single vertex |
//! | [`Graph::community_label_propagation`] | `igraph_community_label_propagation` | label propagation |
//! | [`Graph::community_infomap`] | `igraph_community_infomap` | map equation (Infomap) |
//! | [`Graph::community_fluid_communities`] | `igraph_community_fluid_communities` | fluid communities |
//! | [`Graph::community_voronoi`] | `igraph_community_voronoi` | Voronoi partitioning |
//! | [`Graph::community_optimal_modularity`] | `igraph_community_optimal_modularity` | exact maximum modularity (GLPK) |
//! | [`Graph::modularity`] | `igraph_modularity` | modularity of a partition |
//! | [`Graph::modularity_matrix`] | `igraph_modularity_matrix` | the modularity matrix `B` |
//! | [`Graph::coreness`] | `igraph_coreness` | k-core decomposition |
//! | [`Graph::trussness`] | `igraph_trussness` | k-truss decomposition |
//! | [`community_to_membership`], [`Dendrogram::cut`] | `igraph_community_to_membership` | cut a dendrogram |
//! | [`le_community_to_membership`] | `igraph_le_community_to_membership` | cut a leading eigenvector dendrogram |
//! | [`reindex_membership`] | `igraph_reindex_membership` | make community ids contiguous |
//! | [`compare_communities`] | `igraph_compare_communities` | VI, NMI, split-join, (adjusted) Rand |
//! | [`split_join_distance`] | `igraph_split_join_distance` | both projection distances |
//! | [`Graph::hrg_fit`], [`Graph::hrg_refit`] | `igraph_hrg_fit` | fit a hierarchical random graph by MCMC |
//! | [`Graph::hrg_consensus`], [`Graph::hrg_predict`] | `igraph_hrg_consensus`, `igraph_hrg_predict` | consensus dendrogram, missing link prediction |
//! | [`Hrg::create`], [`Hrg::size`] | `igraph_hrg_create`, `igraph_hrg_size` | build / inspect an [`Hrg`] |
//! | [`Hrg::sample`], [`Hrg::sample_many`], [`Graph::hrg_game`] | `igraph_hrg_sample`, `igraph_hrg_sample_many`, `igraph_hrg_game` | sample graphs from an HRG |
//! | [`Graph::from_hrg_dendrogram`], [`Hrg::dendrogram`] | `igraph_from_hrg_dendrogram` | an HRG dendrogram as a tree |
//!
//! Not bound: `igraph_hrg_resize` (plain storage sizing: [`Hrg::create`]
//! and [`Graph::hrg_fit`] size the model themselves, and resizing leaves the
//! new tree entries uninitialized, which [`Hrg::sample`] would then read as
//! vertex ids), and `igraph_hrg_init` / `igraph_hrg_destroy`, which the
//! constructors and `Drop` of [`Hrg`] call.
//!
//! Randomized methods (Louvain, Leiden, label propagation, spinglass,
//! Infomap, fluid communities, Voronoi generator ties, HRG, ...) draw from
//! the random number generator of the calling thread (each thread has its
//! own): call [`rng::seed`](crate::rng::seed) first for reproducible
//! results, or run them with a dedicated generator through
//! [`Rng::scoped`](crate::rng::Rng::scoped).
//!
//! # See also
//!
//! - [`components`](crate::components): [`Graph::connected_components`]
//!   (spinglass and fluid communities need connected graphs, the leading
//!   eigenvector method starts from the components).
//! - [`centrality`](crate::centrality): [`Graph::edge_betweenness`], the
//!   quantity driving [`Graph::community_edge_betweenness`].
//! - [`mixing`](crate::mixing): [`Graph::assortativity_nominal`], whose
//!   unnormalized value is the modularity of a partition, and
//!   [`Graph::ecc`], the edge clustering coefficient used by
//!   [`Graph::community_voronoi`].
//! - [`paths`](crate::paths): [`Graph::voronoi`], the plain Voronoi
//!   partition around given generators.
//! - [`operators`](crate::operators): [`Graph::contract_vertices`] and
//!   [`Graph::induced_subgraph`] to build the community graph or extract a
//!   community.
//! - [`games`](crate::games): [`Graph::sbm_game`] generates graphs with a
//!   planted community structure, to benchmark the methods.
//! - [`cliques`](crate::cliques) and [`Graph::list_triangles`] for the dense
//!   substructures behind [`Graph::coreness`] and [`Graph::trussness`].

use crate::{
    constants::{
        CommunityComparison, LpaVariant, NeighborMode, SpincommUpdate, SpinglassImplementation,
    },
    error::{Error, ErrorKind, Result, catch_panic},
    ffi::*,
    graph::{EdgeId, Graph, VertexId},
    igraph_call,
    list::{GraphList, VectorList},
    matrix::{Matrix, MatrixInt},
    vector::{Vector, VectorBool, VectorInt},
};
use std::{
    ffi::{c_int, c_void},
    fmt,
    mem::MaybeUninit,
    ops::ControlFlow,
    ptr,
};

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Builds a two-column merges matrix from `(a, b)` pairs.
fn merges_to_matrix(merges: &[(i64, i64)]) -> MatrixInt {
    let mut m = MatrixInt::zeros(merges.len(), 2);
    for (i, &(a, b)) in merges.iter().enumerate() {
        m[(i, 0)] = a;
        m[(i, 1)] = b;
    }
    m
}

/// Reads a two-column merges matrix into `(a, b)` pairs.
fn matrix_to_merges(m: &MatrixInt) -> Vec<(i64, i64)> {
    if m.ncol() < 2 {
        return Vec::new();
    }
    (0..m.nrow()).map(|i| (m[(i, 0)], m[(i, 1)])).collect()
}

/// Null or a pointer to the viewed weights.
macro_rules! opt_view {
    ($name:ident, $slice:expr) => {
        let $name = $slice.map(Vector::view);
        let $name = $name.as_ref().map_or(ptr::null(), |v| v.as_ptr());
    };
}

/// Validates the first `steps` rows of a merges matrix over `nodes` leaves.
///
/// igraph's `igraph_community_to_membership` (igraph 1.0.0 and 1.0.1) indexes
/// its work arrays with the cluster ids found in `merges` without any bounds
/// check, so malformed input would make it read and write out of bounds. Here we require that the `i`-th
/// merge only refers to leaves (`0..nodes`) or to clusters created by earlier
/// merges (`nodes..nodes + i`), and that no cluster is merged twice.
///
/// The bookkeeping is proportional to the number of merges, not to `nodes`:
/// a huge `nodes` is left to igraph, which reports a clean out-of-memory
/// error when it cannot allocate its `nodes`-long work vectors.
fn check_merges(merges: &[(i64, i64)], nodes: usize, steps: usize) -> Result<()> {
    if nodes > igraph_int_t::MAX as usize {
        return Err(Error::invalid(format!(
            "the number of leaves ({nodes}) exceeds the largest igraph integer"
        )));
    }
    let steps = steps.min(merges.len());
    let mut used = std::collections::HashSet::with_capacity(2 * steps);
    for (i, &(a, b)) in merges[..steps].iter().enumerate() {
        // `nodes <= i64::MAX` and `i < merges.len() <= isize::MAX`: no overflow.
        let limit = nodes + i;
        for c in [a, b] {
            if !usize::try_from(c).is_ok_and(|c| c < limit) {
                return Err(Error::invalid(format!(
                    "merge {i} refers to cluster {c}, but only clusters 0..{limit} exist at that point"
                )));
            }
            if !used.insert(c) {
                return Err(Error::invalid(format!(
                    "the merges contain multiple merges of cluster {c}"
                )));
            }
        }
    }
    Ok(())
}

/// Rejects `NaN` and infinite entries (which several igraph routines do not
/// check, see the callers).
fn check_finite(what: &str, values: Option<&[f64]>) -> Result<()> {
    if let Some(&x) = values.and_then(|v| v.iter().find(|x| !x.is_finite())) {
        return Err(Error::invalid(format!(
            "{what} must only contain finite values, found {x}"
        )));
    }
    Ok(())
}

/// Sum of the absolute values (`inf` on overflow).
fn abs_sum(values: &[f64]) -> f64 {
    values.iter().map(|x| x.abs()).sum()
}

/// Largest sum of absolute weights accepted by the ARPACK-based leading
/// eigenvector method and by spinglass: squares of such sums stay finite.
const MAX_ABS_WEIGHT_SUM: f64 = 1e150;

/// Largest spinglass starting temperature accepted: igraph multiplies it by
/// 1.1 twice while estimating the actual start temperature.
const MAX_SPINGLASS_TEMPERATURE: f64 = 1e300;

/// Largest magnitude of the spinglass resolution parameters accepted: they
/// multiply (sums of) weights, bounded by [`MAX_ABS_WEIGHT_SUM`], so the
/// energies stay finite. With `gamma = gamma_minus = f64::MAX` the `Neg`
/// implementation never terminates.
const MAX_SPINGLASS_GAMMA: f64 = 1e150;

/// Validates what igraph's spinglass code does not (igraph 1.0.0 and 1.0.1):
/// non-finite values make its annealing loops run forever, and a huge number
/// of spins overflows its `spins + 1` sized allocations.
fn check_spinglass(weights: Option<&[f64]>, o: &SpinglassOptions, annealing: bool) -> Result<()> {
    check_finite("the weight vector", weights)?;
    // Overflowing weight sums turn the energies into NaN: same endless loop.
    if let Some(w) = weights
        && abs_sum(w) > MAX_ABS_WEIGHT_SUM
    {
        return Err(Error::invalid(format!(
            "the sum of the absolute weights must be at most {MAX_ABS_WEIGHT_SUM:e}, got {:e}",
            abs_sum(w)
        )));
    }
    if !(2..=i32::MAX as usize).contains(&o.spins) {
        return Err(Error::invalid(format!(
            "the number of spins must be in 2..={}, got {}",
            i32::MAX,
            o.spins
        )));
    }
    if !(0.0..=MAX_SPINGLASS_GAMMA).contains(&o.gamma) {
        return Err(Error::invalid(format!(
            "gamma must be in [0, {MAX_SPINGLASS_GAMMA:e}], got {}",
            o.gamma
        )));
    }
    if !annealing {
        return Ok(());
    }
    // `gamma_minus` is only used (and only checked) by the `Neg` implementation.
    if o.implementation == SpinglassImplementation::Neg
        && !(-MAX_SPINGLASS_GAMMA..=MAX_SPINGLASS_GAMMA).contains(&o.gamma_minus)
    {
        return Err(Error::invalid(format!(
            "gamma_minus must be in [-{MAX_SPINGLASS_GAMMA:e}, {MAX_SPINGLASS_GAMMA:e}], got {}",
            o.gamma_minus
        )));
    }
    if !(0.0..1.0).contains(&o.cooling_factor) {
        return Err(Error::invalid(format!(
            "the cooling factor must be in [0, 1), got {}",
            o.cooling_factor
        )));
    }
    for (what, t) in [
        ("starting", o.start_temperature),
        ("stopping", o.stop_temperature),
    ] {
        if !(0.0..=MAX_SPINGLASS_TEMPERATURE).contains(&t) {
            return Err(Error::invalid(format!(
                "the {what} temperature must be in [0, {MAX_SPINGLASS_TEMPERATURE:e}], got {t}"
            )));
        }
    }
    Ok(())
}

fn check_len(what: &str, len: usize, expected: usize) -> Result<()> {
    if len != expected {
        return Err(Error::invalid(format!(
            "{what} has length {len}, but {expected} was expected"
        )));
    }
    Ok(())
}

/// Number of distinct values in a membership vector (ignoring negative ids).
fn count_distinct(membership: &[i64]) -> usize {
    let mut ids: Vec<i64> = membership.iter().copied().filter(|&c| c >= 0).collect();
    ids.sort_unstable();
    ids.dedup();
    ids.len()
}

/// Groups vertex ids by community id (`membership` must be non-negative).
fn groups_of(membership: &[i64]) -> Vec<Vec<VertexId>> {
    let k = membership
        .iter()
        .copied()
        .max()
        .map_or(0, |m| (m + 1).max(0) as usize);
    let mut groups = vec![Vec::new(); k];
    for (v, &c) in membership.iter().enumerate() {
        if c >= 0 {
            groups[c as usize].push(v as VertexId);
        }
    }
    groups
}

/// Community sizes, indexed by community id.
fn sizes_of(membership: &[i64]) -> Vec<usize> {
    groups_of(membership).iter().map(Vec::len).collect()
}

macro_rules! membership_helpers {
    ($ty:ident) => {
        impl $ty {
            /// Number of communities in [`membership`](Self::membership).
            pub fn num_communities(&self) -> usize {
                count_distinct(&self.membership)
            }

            /// Size of each community, indexed by community id.
            pub fn sizes(&self) -> Vec<usize> {
                sizes_of(&self.membership)
            }

            /// The vertices of each community, indexed by community id.
            pub fn communities(&self) -> Vec<Vec<VertexId>> {
                groups_of(&self.membership)
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

crate::ffi_enum! {
    /// Objective function optimized by [`Graph::community_leiden_simple`]
    /// (`igraph_leiden_objective_t`).
    ///
    /// With `A` the adjacency matrix, `m` the total edge weight, `k` the
    /// degrees, `γ` the resolution and `δ(c_i, c_j)` the co-membership
    /// indicator:
    pub enum LeidenObjective: igraph_leiden_objective_t {
        /// Generalized modularity `Q = 1/(2m) Σ_ij (A_ij − γ k_i k_j / (2m)) δ(c_i, c_j)`
        /// (directed: `1/m Σ_ij (A_ij − γ k^out_i k^in_j / m) δ(c_i, c_j)`),
        /// i.e. a configuration model null model. Weights must be non-negative.
        Modularity = igraph_leiden_objective_t_IGRAPH_LEIDEN_OBJECTIVE_MODULARITY,
        /// Constant Potts model `Q = 1/(2m) Σ_ij (A_ij − γ) δ(c_i, c_j)`, free of
        /// the resolution limit. Negative weights are allowed.
        Cpm = igraph_leiden_objective_t_IGRAPH_LEIDEN_OBJECTIVE_CPM,
        /// Erdős–Rényi null model `Q = 1/(2m) Σ_ij (A_ij − γ p) δ(c_i, c_j)`,
        /// `p` being the weighted density. Weights must be non-negative.
        ErdosRenyi = igraph_leiden_objective_t_IGRAPH_LEIDEN_OBJECTIVE_ER,
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// A flat partition of the vertices together with its modularity.
///
/// Returned by [`Graph::community_optimal_modularity`].
#[derive(Debug, Clone, PartialEq)]
pub struct Clustering {
    /// Community id of each vertex, numbered from zero.
    pub membership: Vec<i64>,
    /// Modularity of the partition (`NaN` for graphs without edges).
    pub modularity: f64,
}
membership_helpers!(Clustering);

/// Result of the multi-level (Louvain) algorithm,
/// see [`Graph::community_multilevel`].
#[derive(Debug, Clone, PartialEq)]
pub struct Multilevel {
    /// Community id of each vertex in the final (best) level.
    pub membership: Vec<i64>,
    /// Membership vector after each aggregation level, coarsest last (the
    /// last one equals [`membership`](Self::membership)). Empty if no merge
    /// improved modularity, in which case every vertex is its own community.
    pub levels: Vec<Vec<i64>>,
    /// Modularity (at the requested resolution) after each level, in the
    /// order of [`levels`](Self::levels); if `levels` is empty, the single
    /// value is the modularity of the singleton partition.
    pub modularities: Vec<f64>,
}
membership_helpers!(Multilevel);

impl Multilevel {
    /// Modularity of the final partition (the last entry of
    /// [`modularities`](Self::modularities)), `NaN` if it is empty.
    pub fn modularity(&self) -> f64 {
        self.modularities.last().copied().unwrap_or(f64::NAN)
    }
}

/// Result of the Leiden algorithm, see [`Graph::community_leiden`].
#[derive(Debug, Clone, PartialEq)]
pub struct Leiden {
    /// Community id of each vertex.
    pub membership: Vec<i64>,
    /// Number of clusters in [`membership`](Self::membership).
    pub nb_clusters: usize,
    /// Value of the optimized objective function (quality) of the partition.
    pub quality: f64,
}
membership_helpers!(Leiden);

/// A hierarchical clustering (dendrogram) together with the modularity of
/// each of its levels and the best cut.
///
/// Returned by [`Graph::community_fastgreedy`], [`Graph::community_walktrap`]
/// and [`Graph::community_eb_get_merges`].
#[derive(Debug, Clone, PartialEq)]
pub struct Dendrogram {
    /// Number of leaves, i.e. vertices of the clustered graph.
    pub num_vertices: usize,
    /// The merges: the `i`-th pair `(a, b)` joins clusters `a` and `b` into
    /// cluster `num_vertices + i`; ids below `num_vertices` are single vertices.
    pub merges: Vec<(i64, i64)>,
    /// Modularity before the first merge and after each merge
    /// (`merges.len() + 1` values), empty if not computed.
    pub modularity: Vec<f64>,
    /// Membership vector of the cut with the highest modularity.
    pub membership: Vec<i64>,
}
membership_helpers!(Dendrogram);

impl Dendrogram {
    /// The highest modularity along the dendrogram (`NaN` if unknown).
    pub fn max_modularity(&self) -> f64 {
        self.modularity.iter().copied().fold(f64::NAN, f64::max)
    }

    /// Cuts the dendrogram into `num_communities` clusters, returning the
    /// membership vector (see [`community_to_membership`]).
    ///
    /// This is how to get a clustering with a prescribed number of
    /// communities instead of the modularity-maximizing
    /// [`membership`](Self::membership).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `num_communities` is zero (for a
    /// non-empty graph), exceeds the number of vertices, or the dendrogram
    /// has not enough merges to reach it.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // Three triangles in a row, joined by single edges.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3),
    ///                             (6, 7), (7, 8), (8, 6), (2, 3), (5, 6)], 9, false).unwrap();
    /// let d = g.community_walktrap(None, 4).unwrap();
    /// assert_eq!(d.cut(3).unwrap(), d.membership); // the best cut has 3 clusters
    /// let two = d.cut(2).unwrap();
    /// assert_eq!(two.iter().filter(|&&c| c == two[0]).count() % 3, 0);
    /// assert_eq!(d.cut(1).unwrap(), vec![0; 9]);
    /// assert!(d.cut(10).is_err());
    /// ```
    pub fn cut(&self, num_communities: usize) -> Result<Vec<i64>> {
        if num_communities > self.num_vertices || num_communities == 0 && self.num_vertices > 0 {
            return Err(Error::invalid(format!(
                "cannot cut a dendrogram over {} vertices into {num_communities} clusters",
                self.num_vertices
            )));
        }
        let steps = self.num_vertices - num_communities;
        community_to_membership(&self.merges, self.num_vertices, steps).map(|(m, _)| m)
    }
}

/// Result of the Girvan–Newman algorithm, see [`Graph::community_edge_betweenness`].
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeBetweennessCommunities {
    /// The ids of the removed edges, in order of removal.
    pub removed_edges: Vec<EdgeId>,
    /// Betweenness of each removed edge at the moment of its removal
    /// (not divided by the weights).
    pub edge_betweenness: Vec<f64>,
    /// The dendrogram, obtained by replaying the removals backwards.
    pub merges: Vec<(i64, i64)>,
    /// Indices into [`removed_edges`](Self::removed_edges) of the edges whose
    /// removal split a component, in reverse order.
    pub bridges: Vec<i64>,
    /// Modularity of each division: before the first merge (all components
    /// split into single vertices) and after each merge, i.e.
    /// `merges.len() + 1` values in the order of [`merges`](Self::merges).
    pub modularity: Vec<f64>,
    /// Membership vector of the division with the highest modularity.
    pub membership: Vec<i64>,
}
membership_helpers!(EdgeBetweennessCommunities);

/// Result of [`Graph::community_eb_get_merges`].
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeRemovalMerges {
    /// The dendrogram with its modularity values and best membership.
    pub dendrogram: Dendrogram,
    /// Indices into the given edge sequence of the edges whose removal split a
    /// component, in reverse order.
    pub bridges: Vec<i64>,
}

/// One step of the history of [`Graph::community_leading_eigenvector`]
/// (`igraph_leading_eigenvector_community_history_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LeadingEigenvectorEvent {
    /// The algorithm started from the connected components of the graph.
    StartFull,
    /// The algorithm started from a given partition with this many communities.
    StartGiven {
        /// Initial number of communities.
        communities: i64,
    },
    /// A community was split in two: the first part keeps the id, the second
    /// one gets the number of communities before the split as id.
    Split {
        /// The id of the split community.
        community: i64,
    },
    /// Splitting the community would not increase modularity.
    Failed {
        /// The id of the community that was not split.
        community: i64,
    },
}

fn decode_history(raw: &[i64]) -> Vec<LeadingEigenvectorEvent> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        let code = raw[i] as igraph_leading_eigenvector_community_history_t;
        let next = raw.get(i + 1).copied().unwrap_or(-1);
        match code {
            igraph_leading_eigenvector_community_history_t_IGRAPH_LEVC_HIST_START_FULL => {
                out.push(LeadingEigenvectorEvent::StartFull);
                i += 1;
            }
            igraph_leading_eigenvector_community_history_t_IGRAPH_LEVC_HIST_START_GIVEN => {
                out.push(LeadingEigenvectorEvent::StartGiven { communities: next });
                i += 2;
            }
            igraph_leading_eigenvector_community_history_t_IGRAPH_LEVC_HIST_SPLIT => {
                out.push(LeadingEigenvectorEvent::Split { community: next });
                i += 2;
            }
            igraph_leading_eigenvector_community_history_t_IGRAPH_LEVC_HIST_FAILED => {
                out.push(LeadingEigenvectorEvent::Failed { community: next });
                i += 2;
            }
            _ => i += 1,
        }
    }
    out
}

/// Result of Newman's leading eigenvector method,
/// see [`Graph::community_leading_eigenvector`].
#[derive(Debug, Clone, PartialEq)]
pub struct LeadingEigenvector {
    /// Community id of each vertex after all the splits.
    pub membership: Vec<i64>,
    /// The splits, replayed backwards as merges of *community* ids (not
    /// vertex ids): with `p` final communities, the first pair forms
    /// community `p`, the second `p + 1`, ... Use
    /// [`le_community_to_membership`] to undo splits.
    pub merges: Vec<(i64, i64)>,
    /// Modularity of the final division.
    pub modularity: f64,
    /// Eigenvalue computed at each step (`NaN` for steps given by the
    /// initial partition); non-positive values did not result in a split.
    pub eigenvalues: Vec<f64>,
    /// Eigenvector computed at each step, restricted to the vertices of the
    /// community being split (empty for steps given by the initial partition).
    pub eigenvectors: Vec<Vec<f64>>,
    /// A trace of the algorithm.
    pub history: Vec<LeadingEigenvectorEvent>,
}
membership_helpers!(LeadingEigenvector);

/// The state passed to the callback of
/// [`Graph::community_leading_eigenvector_with`] after each eigenvector
/// computation.
pub struct LeadingEigenvectorStep<'a> {
    membership: &'a [i64],
    community: i64,
    eigenvalue: f64,
    eigenvector: &'a [f64],
    multiplier: igraph_arpack_function_t,
    arpack_extra: *mut c_void,
}

impl fmt::Debug for LeadingEigenvectorStep<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LeadingEigenvectorStep")
            .field("membership", &self.membership)
            .field("community", &self.community)
            .field("eigenvalue", &self.eigenvalue)
            .field("eigenvector", &self.eigenvector)
            .finish()
    }
}

impl LeadingEigenvectorStep<'_> {
    /// The current membership vector, before applying the split implied by
    /// this eigenvector.
    pub fn membership(&self) -> &[i64] {
        self.membership
    }

    /// The id of the community the algorithm is trying to split.
    pub fn community(&self) -> i64 {
        self.community
    }

    /// The leading eigenvalue just found; the community is split only if it
    /// is positive.
    pub fn eigenvalue(&self) -> f64 {
        self.eigenvalue
    }

    /// The corresponding eigenvector, with one element per vertex of the
    /// community (in increasing vertex id order).
    pub fn eigenvector(&self) -> &[f64] {
        self.eigenvector
    }

    /// Multiplies `x` by the (generalized) modularity matrix of the community
    /// being split, i.e. runs the ARPACK matrix-vector product igraph used to
    /// find this eigenvector. `x` must have the length of
    /// [`eigenvector`](Self::eigenvector).
    ///
    /// For instance `step.multiply(step.eigenvector())` is
    /// `eigenvalue * eigenvector` up to numerical accuracy.
    ///
    /// # Errors
    /// If `x` has a wrong length.
    pub fn multiply(&self, x: &[f64]) -> Result<Vec<f64>> {
        check_len("the vector to multiply", x.len(), self.eigenvector.len())?;
        let mut to = vec![0.0; x.len()];
        let Some(f) = self.multiplier else {
            return Err(Error::invalid("no ARPACK multiplier available"));
        };
        // Called from inside the leading eigenvector callback: no
        // `igraph_call!` here, so forget any stale error record ourselves.
        crate::error::reset_last_error();
        // igraph's multiplier writes `n` values into `to` and reads `n` from `from`.
        let code = unsafe {
            f(
                to.as_mut_ptr(),
                x.as_ptr(),
                x.len() as c_int,
                self.arpack_extra,
            )
        };
        crate::error::check(code)?;
        Ok(to)
    }
}

/// Result of the spinglass method, see [`Graph::community_spinglass`].
#[derive(Debug, Clone, PartialEq)]
pub struct Spinglass {
    /// Community id of each vertex.
    pub membership: Vec<i64>,
    /// Size of each community, indexed by community id.
    pub csize: Vec<i64>,
    /// Generalized modularity (with resolution `gamma`) of the result.
    pub modularity: f64,
    /// Temperature at the end of the simulated annealing.
    pub temperature: f64,
}
membership_helpers!(Spinglass);

/// Result of [`Graph::community_spinglass_single`].
#[derive(Debug, Clone, PartialEq)]
pub struct SpinglassSingle {
    /// The vertices in the community of the given vertex.
    pub community: Vec<VertexId>,
    /// Cohesion index of the community.
    pub cohesion: f64,
    /// Adhesion index of the community.
    pub adhesion: f64,
    /// Number (or total weight) of the edges inside the community.
    pub inner_links: f64,
    /// Number (or total weight) of the edges leaving the community.
    pub outer_links: f64,
}

/// Result of Infomap, see [`Graph::community_infomap`].
#[derive(Debug, Clone, PartialEq)]
pub struct Infomap {
    /// Community id of each vertex.
    pub membership: Vec<i64>,
    /// Code length (in bits) of the partition: the expected description
    /// length of a random walk step under the map equation.
    pub codelength: f64,
}
membership_helpers!(Infomap);

/// Result of Voronoi partitioning, see [`Graph::community_voronoi`].
#[derive(Debug, Clone, PartialEq)]
pub struct Voronoi {
    /// Community id of each vertex.
    pub membership: Vec<i64>,
    /// The generator vertex of each community.
    pub generators: Vec<VertexId>,
    /// Modularity of the partition.
    pub modularity: f64,
}
membership_helpers!(Voronoi);

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Options of [`Graph::community_leiden`] and [`Graph::community_leiden_simple`].
#[derive(Debug, Clone, PartialEq)]
pub struct LeidenOptions<'a> {
    /// Resolution parameter `γ` (default `1.0`). Note that for the raw
    /// [`Graph::community_leiden`] interface, modularity is obtained with
    /// degrees as vertex weights and `γ = 1 / (2m)`.
    pub resolution: f64,
    /// Randomness in the refinement step (default `0.01`).
    pub beta: f64,
    /// Number of iterations of the core algorithm (default `Some(2)`);
    /// `None` iterates until an iteration does not change the partition.
    pub iterations: Option<u32>,
    /// Starting partition (default `None`: singletons).
    pub initial: Option<&'a [i64]>,
}

impl Default for LeidenOptions<'_> {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            beta: 0.01,
            iterations: Some(2),
            initial: None,
        }
    }
}

impl<'a> LeidenOptions<'a> {
    /// Sets the resolution parameter.
    pub fn with_resolution(mut self, resolution: f64) -> Self {
        self.resolution = resolution;
        self
    }
    /// Sets the refinement randomness `beta`.
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }
    /// Sets the number of iterations (`None`: until convergence).
    pub fn with_iterations(mut self, iterations: Option<u32>) -> Self {
        self.iterations = iterations;
        self
    }
    /// Sets the starting partition.
    pub fn with_initial(mut self, initial: &'a [i64]) -> Self {
        self.initial = Some(initial);
        self
    }
}

/// Options of [`Graph::community_spinglass`] and
/// [`Graph::community_spinglass_single`], with the defaults suggested by igraph.
#[derive(Debug, Clone, PartialEq)]
pub struct SpinglassOptions {
    /// Number of spins, i.e. the maximum number of communities (default `25`).
    pub spins: usize,
    /// Update all spins in parallel (default `false`); not supported by
    /// [`SpinglassImplementation::Neg`].
    pub parallel_update: bool,
    /// Starting temperature (default `1.0`).
    pub start_temperature: f64,
    /// Stopping temperature (default `0.01`).
    pub stop_temperature: f64,
    /// Cooling factor of the simulated annealing (default `0.99`).
    pub cooling_factor: f64,
    /// Null model: [`SpincommUpdate::Config`] (configuration model, default)
    /// or [`SpincommUpdate::Simple`] (Erdős–Rényi).
    pub update_rule: SpincommUpdate,
    /// Resolution parameter `γ` (default `1.0`, must be in `[0, 1e150]`).
    pub gamma: f64,
    /// Implementation: [`SpinglassImplementation::Orig`] (default, faster) or
    /// [`SpinglassImplementation::Neg`] (allows negative weights).
    pub implementation: SpinglassImplementation,
    /// Resolution for the negative part of the network, `Neg` only (default
    /// `1.0`, magnitude at most `1e150`).
    pub gamma_minus: f64,
}

impl Default for SpinglassOptions {
    fn default() -> Self {
        Self {
            spins: 25,
            parallel_update: false,
            start_temperature: 1.0,
            stop_temperature: 0.01,
            cooling_factor: 0.99,
            update_rule: SpincommUpdate::Config,
            gamma: 1.0,
            implementation: SpinglassImplementation::Orig,
            gamma_minus: 1.0,
        }
    }
}

/// Options of [`Graph::community_label_propagation`].
#[derive(Debug, Clone, PartialEq)]
pub struct LabelPropagationOptions<'a> {
    /// Direction of label propagation in directed graphs (default
    /// [`NeighborMode::All`]: ignore directions). `Out` propagates labels
    /// along the edges, `In` backwards.
    pub mode: NeighborMode,
    /// Initial labels, one per vertex: a label is an id in `0..vcount`, and
    /// negative values mean "unlabeled" (default `None`: every vertex has its
    /// own label). Label values carry no meaning and may be renumbered; only
    /// co-membership matters.
    pub initial: Option<&'a [i64]>,
    /// Which initial labels are fixed (only meaningful with
    /// [`initial`](Self::initial); unlabeled vertices cannot be fixed, and
    /// igraph ignores their flag with a warning). Fixed vertices keep their
    /// co-membership: two fixed vertices end up in the same community iff
    /// they had the same initial label.
    pub fixed: Option<&'a [bool]>,
    /// Algorithm variant (default [`LpaVariant::Dominance`]).
    pub variant: LpaVariant,
}

impl Default for LabelPropagationOptions<'_> {
    fn default() -> Self {
        Self {
            mode: NeighborMode::All,
            initial: None,
            fixed: None,
            variant: LpaVariant::Dominance,
        }
    }
}

/// Options of [`Graph::community_infomap`].
#[derive(Debug, Clone, PartialEq)]
pub struct InfomapOptions {
    /// Number of attempts to partition the network, the best one is kept
    /// (default `10`, at least `1` and at most `u32::MAX`).
    pub trials: usize,
    /// Add a Bayesian prior network to avoid overfitting missing links
    /// (default `false`).
    pub regularized: bool,
    /// Multiplier of the default regularization strength (default `1.0`).
    pub regularization_strength: f64,
}

impl Default for InfomapOptions {
    fn default() -> Self {
        Self {
            trials: 10,
            regularized: false,
            regularization_strength: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Non-graph functions
// ---------------------------------------------------------------------------

/// Cuts a dendrogram after `steps` merges, returning the membership vector
/// and the size of each community.
///
/// The dendrogram has `nodes` leaves (the vertices) and is given by its
/// `merges`, in the format of [`Graph::community_fastgreedy`],
/// [`Graph::community_walktrap`] or [`Graph::community_edge_betweenness`]:
/// the `i`-th pair joins two dendrogram nodes into node `nodes + i`. After
/// `steps` merges, `nodes - steps` communities remain, numbered from zero.
/// `steps` may not exceed the number of merges. Time complexity: O(|V|).
/// [`Dendrogram::cut`] does the same given a number of communities.
///
/// Binds [`igraph_community_to_membership`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_community_to_membership).
///
/// # Errors
/// [`ErrorKind::InvalidValue`] if `steps`
/// is too large or the merges are malformed.
///
/// # Examples
/// ```
/// use igraph::community::community_to_membership;
/// // 4 leaves: merge 0+1 into 4, 2+3 into 5, then 4+5 into 6.
/// let merges = [(0, 1), (2, 3), (4, 5)];
/// let (membership, sizes) = community_to_membership(&merges, 4, 2).unwrap();
/// assert_eq!(membership, vec![1, 1, 0, 0]);
/// assert_eq!(sizes, vec![2, 2]);
/// ```
pub fn community_to_membership(
    merges: &[(i64, i64)],
    nodes: usize,
    steps: usize,
) -> Result<(Vec<i64>, Vec<i64>)> {
    check_merges(merges, nodes, steps)?;
    let m = merges_to_matrix(merges);
    let mut membership = VectorInt::new();
    let mut csize = VectorInt::new();
    igraph_call!(igraph_community_to_membership(
        &m,
        nodes as igraph_int_t,
        steps as igraph_int_t,
        &mut membership,
        &mut csize
    ))?;
    Ok((membership.into(), csize.into()))
}

/// Applies `steps` merges of a leading eigenvector dendrogram to an initial
/// partition, returning the new membership vector and community sizes.
///
/// Unlike [`community_to_membership`], the dendrogram leaves are the `m`
/// communities of `membership` (ids `0..m`, contiguous), and the `i`-th merge
/// forms community `m + i`, as produced by
/// [`Graph::community_leading_eigenvector`]. The result has `m - steps`
/// communities. Time complexity: O(|V|).
///
/// Binds [`igraph_le_community_to_membership`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_le_community_to_membership).
///
/// # Errors
/// [`ErrorKind::InvalidValue`] for
/// non-contiguous or negative ids, too many steps, or merges referring to
/// already merged clusters.
///
/// # Examples
/// ```
/// use igraph::community::le_community_to_membership;
/// let (membership, sizes) = le_community_to_membership(&[(1, 3)], 1, &[0, 1, 2, 3, 4]).unwrap();
/// assert_eq!(membership, vec![1, 0, 2, 0, 3]);
/// assert_eq!(sizes, vec![2, 1, 1, 1]);
/// ```
pub fn le_community_to_membership(
    merges: &[(i64, i64)],
    steps: usize,
    membership: &[i64],
) -> Result<(Vec<i64>, Vec<i64>)> {
    // The leaves of the dendrogram are the initial communities.
    let components = membership.iter().max().map_or(0, |&m| m.saturating_add(1));
    if components > membership.len() as i64 {
        return Err(Error::invalid(format!(
            "invalid membership vector: {components} communities for {} elements",
            membership.len()
        )));
    }
    check_merges(merges, components.max(0) as usize, steps)?;
    let m = merges_to_matrix(merges);
    let mut memb = VectorInt::from_slice(membership);
    let mut csize = VectorInt::new();
    igraph_call!(igraph_le_community_to_membership(
        &m,
        steps as igraph_int_t,
        &mut memb,
        &mut csize
    ))?;
    Ok((memb.into(), csize.into()))
}

/// Relabels a membership vector in place so that community ids are
/// `0..k`, and returns the mapping from new to old ids (its length `k` is the
/// number of communities).
///
/// When all ids lie in `0..n` (`n` being the length of `membership`), new
/// ids are assigned in order of first appearance; otherwise (negative or
/// large ids) they follow the increasing order of the old ids.
/// Time complexity: O(n) in the first case, O(n log n) otherwise.
///
/// Binds [`igraph_reindex_membership`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_reindex_membership).
///
/// # Examples
/// ```
/// use igraph::community::reindex_membership;
/// // Ids in 0..4: numbered by first appearance.
/// let mut membership = vec![3, 3, 1, 1];
/// assert_eq!(reindex_membership(&mut membership).unwrap(), vec![3, 1]);
/// assert_eq!(membership, vec![0, 0, 1, 1]);
/// // An id outside 0..4: numbered in increasing order of the old ids.
/// let mut membership = vec![7, 3, 7, 10];
/// let new_to_old = reindex_membership(&mut membership).unwrap();
/// assert_eq!(membership, vec![1, 0, 1, 2]);
/// assert_eq!(new_to_old, vec![3, 7, 10]);
/// ```
pub fn reindex_membership(membership: &mut [i64]) -> Result<Vec<i64>> {
    let mut memb = VectorInt::from_slice(membership);
    let mut new_to_old = VectorInt::new();
    let mut nb: igraph_int_t = 0;
    igraph_call!(igraph_reindex_membership(
        &mut memb,
        &mut new_to_old,
        &mut nb
    ))?;
    membership.copy_from_slice(&memb);
    Ok(new_to_old.into())
}

/// Compares two partitions of the same set with the given measure.
///
/// - [`CommunityComparison::Vi`]: variation of information (Meilă 2003),
///   `VI = H(C1) + H(C2) − 2 MI(C1, C2)` in natural units; 0 iff equal.
/// - [`CommunityComparison::Nmi`]: normalized mutual information (Danon et
///   al. 2005), `2 MI / (H(C1) + H(C2))` in `(0, 1]`; 1 iff equal.
/// - [`CommunityComparison::SplitJoin`]: split-join distance (van Dongen 2000),
///   the sum of both [`split_join_distance`]s.
/// - [`CommunityComparison::Rand`]: Rand index (1971), fraction of vertex
///   pairs on which the two partitions agree.
/// - [`CommunityComparison::AdjustedRand`]: Hubert–Arabie adjusted Rand
///   index, corrected for chance (may be negative; `NaN` when undefined).
///
/// Community ids need not be contiguous. Time complexity: O(n log n).
///
/// Binds [`igraph_compare_communities`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_compare_communities).
///
/// # Errors
/// [`ErrorKind::InvalidValue`] if the lengths
/// differ, or for the Rand indices with fewer than two elements.
///
/// # Examples
/// ```
/// use igraph::prelude::*;
/// use igraph::community::compare_communities;
/// let a = [2, 0, 2, 1, 1, 0, 2, 2, 1, 2];
/// let b = [1, 1, 2, 1, 1, 0, 2, 2, 0, 2];
/// let rand = compare_communities(&a, &b, CommunityComparison::Rand).unwrap();
/// assert!((rand - 0.711111).abs() < 1e-6);
/// // Relabeling does not matter:
/// let vi = compare_communities(&[0, 1], &[1, 0], CommunityComparison::Vi).unwrap();
/// assert_eq!(vi, 0.0);
/// ```
pub fn compare_communities(
    comm1: &[i64],
    comm2: &[i64],
    method: CommunityComparison,
) -> Result<f64> {
    let c1 = VectorInt::view(comm1);
    let c2 = VectorInt::view(comm2);
    let mut res = 0.0;
    igraph_call!(igraph_compare_communities(
        c1.as_ptr(),
        c2.as_ptr(),
        &mut res,
        method.into()
    ))?;
    Ok(res)
}

/// The two projection distances between two partitions, whose sum is the
/// split-join distance of van Dongen.
///
/// For each set of the first partition the best matching (maximum overlap)
/// set of the second one is found; the first distance is the number of
/// elements minus the sum of these overlaps. The second distance is the same
/// with the roles swapped. A distance is zero iff the corresponding partition
/// is a refinement of the other one. Time complexity: O(n log n).
///
/// Binds [`igraph_split_join_distance`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_split_join_distance).
///
/// # Errors
/// [`ErrorKind::InvalidValue`] if the lengths differ.
///
/// # Examples
/// ```
/// use igraph::community::split_join_distance;
/// // Singletons refine the one-block partition:
/// assert_eq!(split_join_distance(&[0, 1, 2, 3, 4], &[0; 5]).unwrap(), (0, 4));
/// ```
pub fn split_join_distance(comm1: &[i64], comm2: &[i64]) -> Result<(i64, i64)> {
    let c1 = VectorInt::view(comm1);
    let c2 = VectorInt::view(comm2);
    let (mut d12, mut d21) = (0, 0);
    igraph_call!(igraph_split_join_distance(
        c1.as_ptr(),
        c2.as_ptr(),
        &mut d12,
        &mut d21
    ))?;
    Ok((d12, d21))
}

// ---------------------------------------------------------------------------
// Leading eigenvector callback
// ---------------------------------------------------------------------------

unsafe extern "C" fn levc_trampoline<F>(
    membership: *const igraph_vector_int_t,
    comm: igraph_int_t,
    eigenvalue: igraph_real_t,
    eigenvector: *const igraph_vector_t,
    multiplier: igraph_arpack_function_t,
    arpack_extra: *mut c_void,
    extra: *mut c_void,
) -> igraph_error_t
where
    F: FnMut(&LeadingEigenvectorStep<'_>) -> ControlFlow<()>,
{
    // The closure runs in a fresh level of igraph's "finally" stack. The
    // running `igraph_community_leading_eigenvector` keeps its temporaries
    // (adjacency lists, ARPACK storage, ...) on that stack, and igraph's
    // error handler frees the current level when a call fails: without a new
    // level, a failing igraph call made by the closure (or by
    // `LeadingEigenvectorStep::multiply`) would free the objects of the
    // computation that is still running, a use-after-free in C.
    // SAFETY: plain bookkeeping on igraph's thread-local finally stack; the
    // matching EXIT runs below, as `catch_panic` never unwinds.
    unsafe { IGRAPH_FINALLY_ENTER() };
    let code = catch_panic(|| {
        // SAFETY: `extra` is the `&mut F` passed by `community_leading_eigenvector_with`,
        // and the vectors are valid for the duration of the callback.
        let f = unsafe { &mut *(extra as *mut F) };
        let step = LeadingEigenvectorStep {
            membership: unsafe { (*membership).as_slice() },
            community: comm,
            eigenvalue,
            eigenvector: unsafe { (*eigenvector).as_slice() },
            multiplier,
            arpack_extra,
        };
        match f(&step) {
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

// ---------------------------------------------------------------------------
// Graph methods
// ---------------------------------------------------------------------------

impl igraph_t {
    fn community_check_weights(&self, what: &str, weights: Option<&[f64]>) -> Result<()> {
        match weights {
            Some(w) => check_len(what, w.len(), self.ecount()),
            None => Ok(()),
        }
    }

    fn check_vertex_vec<T>(&self, what: &str, v: Option<&[T]>) -> Result<()> {
        match v {
            Some(v) => check_len(what, v.len(), self.vcount()),
            None => Ok(()),
        }
    }

    /// The coreness (k-core index) of every vertex.
    ///
    /// The k-core of a graph is its maximal subgraph in which every vertex
    /// has degree at least `k`; the coreness of a vertex is the largest `k`
    /// such that it belongs to the k-core. For directed graphs `mode` selects
    /// in-cores ([`NeighborMode::In`]), out-cores ([`NeighborMode::Out`]) or
    /// the undirected version ([`NeighborMode::All`]); it is ignored for
    /// undirected graphs. Uses the O(|E|) algorithm of Batagelj and Zaversnik.
    ///
    /// The coreness of a vertex never exceeds its [degree](Graph::degree);
    /// the vertices of coreness `k` or more induce the k-core, which can be
    /// extracted with [`Graph::induced_subgraph`].
    ///
    /// Binds [`igraph_coreness`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_coreness).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A triangle with a pendant vertex.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false).unwrap();
    /// assert_eq!(g.coreness(NeighborMode::All).unwrap(), vec![2, 2, 2, 1]);
    /// ```
    pub fn coreness(&self, mode: NeighborMode) -> Result<Vec<i64>> {
        let mut res = VectorInt::new();
        igraph_call!(igraph_coreness(self, &mut res, mode.into()))?;
        Ok(res.into())
    }

    /// The trussness of every edge.
    ///
    /// A k-truss is a subgraph in which every edge lies in at least `k − 2`
    /// triangles of the subgraph; the trussness of an edge is the largest `k`
    /// such that it belongs to a k-truss. To get the k-truss, keep the edges
    /// with trussness `>= k`. Loops are allowed, multigraphs are not.
    /// Time complexity: O(|E|^1.5) (Wang and Cheng, 2012).
    ///
    /// Every edge has trussness at least 2, and more than 2 exactly when it
    /// lies in a triangle (see [`Graph::list_triangles`]); the edges of a
    /// `k`-clique (see [`Graph::maximal_cliques`]) have trussness at least `k`.
    ///
    /// Binds [`igraph_trussness`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_trussness).
    ///
    /// # Errors
    /// [`ErrorKind::Unimplemented`] for multigraphs.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A 4-clique (every edge in 2 triangles) plus a pendant edge.
    /// let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 4)], 5, false)
    ///     .unwrap();
    /// assert_eq!(g.trussness().unwrap(), vec![4, 4, 4, 4, 4, 4, 2]);
    /// ```
    pub fn trussness(&self) -> Result<Vec<i64>> {
        let mut res = VectorInt::new();
        igraph_call!(igraph_trussness(self, &mut res))?;
        Ok(res.into())
    }

    /// The modularity of a partition of the vertices.
    ///
    /// `Q = 1/(2m) Σ_ij (A_ij − γ k_i k_j / (2m)) δ(c_i, c_j)`, where `m` is
    /// the number of edges, `A` the adjacency matrix (with loops counted twice
    /// on the diagonal), `k` the degrees, `γ` the `resolution` (1 for the
    /// classical definition) and `c` the membership. With `directed = true`
    /// on a directed graph the Leicht–Newman version
    /// `Q = 1/m Σ_ij (A_ij − γ k^out_i k^in_j / m) δ(c_i, c_j)` is used. With
    /// weights, `A`, `k` and `m` are replaced by their weighted counterparts.
    /// For graphs without edges the modularity is `NaN`.
    ///
    /// Community ids need not be contiguous (empty communities are allowed).
    /// Time complexity: O(|V| + |E|).
    ///
    /// For non-negative ids and `resolution = 1`, this is the unnormalized
    /// nominal assortativity of the partition, see
    /// [`Graph::assortativity_nominal`].
    ///
    /// Binds [`igraph_modularity`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_modularity).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `membership` or `weights` have a wrong
    /// length, a weight is negative, or `resolution < 0`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // Two triangles joined by an edge.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)], 6, false)
    ///     .unwrap();
    /// let q = g.modularity(&[0, 0, 0, 1, 1, 1], None, 1.0, true).unwrap();
    /// assert!((q - 5.0 / 14.0).abs() < 1e-12);
    /// ```
    pub fn modularity(
        &self,
        membership: &[i64],
        weights: Option<&[f64]>,
        resolution: f64,
        directed: bool,
    ) -> Result<f64> {
        check_len("the membership vector", membership.len(), self.vcount())?;
        self.community_check_weights("the weight vector", weights)?;
        let memb = VectorInt::view(membership);
        opt_view!(w, weights);
        let mut q = 0.0;
        igraph_call!(igraph_modularity(
            self,
            memb.as_ptr(),
            w,
            resolution,
            directed,
            &mut q
        ))?;
        Ok(q)
    }

    /// The modularity matrix `B_ij = A_ij − γ k_i k_j / (2m)`.
    ///
    /// For directed graphs (and `directed = true`),
    /// `B_ij = A_ij − γ k^out_i k^in_j / m`. Loops of undirected graphs are
    /// counted twice in `A`; with weights, the weighted adjacency matrix and
    /// strengths are used. When there are no edges the result is undefined
    /// (`NaN`s). Then `Q = 1/(2m) Σ_ij B_ij δ(c_i, c_j)`, see [`Graph::modularity`].
    /// The adjacency part alone is [`Graph::get_adjacency`] (or this function
    /// with `resolution = 0`).
    ///
    /// Binds [`igraph_modularity_matrix`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_modularity_matrix).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let triangle = Graph::from_edges(&[(0, 1), (0, 2), (1, 2)], 3, false).unwrap();
    /// // With resolution 0 the modularity matrix is the adjacency matrix.
    /// let b = triangle.modularity_matrix(None, 0.0, false).unwrap();
    /// assert_eq!(b.to_rows(), vec![vec![0.0, 1.0, 1.0], vec![1.0, 0.0, 1.0], vec![1.0, 1.0, 0.0]]);
    /// ```
    pub fn modularity_matrix(
        &self,
        weights: Option<&[f64]>,
        resolution: f64,
        directed: bool,
    ) -> Result<Matrix> {
        self.community_check_weights("the weight vector", weights)?;
        opt_view!(w, weights);
        let mut res = Matrix::new();
        igraph_call!(igraph_modularity_matrix(
            self, w, resolution, &mut res, directed
        ))?;
        Ok(res)
    }

    /// Louvain community detection: multi-level greedy modularity optimization.
    ///
    /// Initially each vertex is a community; vertices are then moved, in
    /// random order, to the neighboring community that increases modularity
    /// the most, until no move helps. Communities are then contracted into
    /// single vertices and the process restarts, until there is a single
    /// vertex or modularity cannot increase. Higher `resolution` values give
    /// more, smaller communities (1 is the classical modularity). Weights
    /// must be non-negative. The graph must be undirected. Near linear time
    /// on sparse graphs (Blondel et al., 2008).
    ///
    /// The result contains the final membership and the membership and
    /// modularity after each level. For a directed graph, use
    /// [`Graph::community_leiden_simple`] (which supports directed modularity)
    /// or convert it first with [`Graph::to_undirected`].
    ///
    /// Binds [`igraph_community_multilevel`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_community_multilevel).
    ///
    /// # Errors
    /// For directed graphs, negative weights or `resolution < 0`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // Two triangles joined by an edge.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)], 6, false)
    ///     .unwrap();
    /// let res = g.community_multilevel(None, 1.0).unwrap();
    /// assert_eq!(res.membership, vec![0, 0, 0, 1, 1, 1]);
    /// assert_eq!(res.num_communities(), 2);
    /// ```
    pub fn community_multilevel(
        &self,
        weights: Option<&[f64]>,
        resolution: f64,
    ) -> Result<Multilevel> {
        self.community_check_weights("the weight vector", weights)?;
        opt_view!(w, weights);
        let mut membership = VectorInt::new();
        let mut memberships = MatrixInt::new();
        let mut modularity = Vector::new();
        igraph_call!(igraph_community_multilevel(
            self,
            w,
            resolution,
            &mut membership,
            &mut memberships,
            &mut modularity
        ))?;
        Ok(Multilevel {
            membership: membership.into(),
            levels: memberships.to_rows(),
            modularities: modularity.into(),
        })
    }

    fn leiden_start(&self, initial: Option<&[i64]>) -> Result<VectorInt> {
        match initial {
            Some(init) => {
                check_len("the initial membership", init.len(), self.vcount())?;
                Ok(VectorInt::from_slice(init))
            }
            None => Ok((0..self.vcount() as i64).collect()),
        }
    }

    /// Leiden community detection with explicit vertex weights.
    ///
    /// The Leiden algorithm (Traag, Waltman and van Eck, 2019) improves on
    /// Louvain by a refinement phase which guarantees well-connected
    /// communities. It maximizes
    /// `1/(2m) Σ_ij (A_ij − γ n_i n_j) δ(s_i, s_j)` (directed:
    /// `1/m Σ_ij (A_ij − γ n^out_i n^in_j) δ(s_i, s_j)`), where `n` are the
    /// vertex weights (`vertex_out_weights`, `vertex_in_weights`; `None` means
    /// all ones, and `vertex_in_weights` must be `None` for undirected graphs)
    /// and `γ` is [`LeidenOptions::resolution`]. With unit vertex weights this
    /// is the Constant Potts Model; with degrees as vertex weights and
    /// `γ = 1/(2m)` it is modularity (see [`Graph::community_leiden_simple`]
    /// for a more convenient interface). Edge weights may be negative.
    ///
    /// Binds [`igraph_community_leiden`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_community_leiden).
    ///
    /// # Errors
    /// If a vector has a wrong length.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::community::LeidenOptions;
    /// let mut edges = vec![(0, 5)];
    /// for base in [0, 5] {
    ///     for i in 0..5 {
    ///         for j in i + 1..5 { edges.push((base + i, base + j)); }
    ///     }
    /// }
    /// let g = Graph::from_edges(&edges, 10, false).unwrap();
    /// // Constant Potts Model with resolution 0.05, as in igraph's example.
    /// let opts = LeidenOptions::default().with_resolution(0.05).with_iterations(Some(1));
    /// let res = g.community_leiden(None, None, None, &opts).unwrap();
    /// assert_eq!(res.nb_clusters, 2);
    /// assert!((res.quality - 0.8929).abs() < 1e-4);
    /// ```
    pub fn community_leiden(
        &self,
        edge_weights: Option<&[f64]>,
        vertex_out_weights: Option<&[f64]>,
        vertex_in_weights: Option<&[f64]>,
        options: &LeidenOptions<'_>,
    ) -> Result<Leiden> {
        self.community_check_weights("the edge weight vector", edge_weights)?;
        self.check_vertex_vec("the vertex out-weight vector", vertex_out_weights)?;
        self.check_vertex_vec("the vertex in-weight vector", vertex_in_weights)?;
        opt_view!(ew, edge_weights);
        opt_view!(vo, vertex_out_weights);
        opt_view!(vi, vertex_in_weights);
        let mut membership = self.leiden_start(options.initial)?;
        let (mut nb, mut quality) = (0, 0.0);
        let iterations = options.iterations.map_or(-1, igraph_int_t::from);
        igraph_call!(igraph_community_leiden(
            self,
            ew,
            vo,
            vi,
            options.resolution,
            options.beta,
            options.initial.is_some(),
            iterations,
            &mut membership,
            &mut nb,
            &mut quality
        ))?;
        Ok(Leiden {
            membership: membership.into(),
            nb_clusters: nb as usize,
            quality,
        })
    }

    /// Leiden community detection optimizing a chosen objective function.
    ///
    /// A convenience interface to [`Graph::community_leiden`] which computes
    /// suitable vertex weights for [`LeidenObjective::Modularity`] (generalized
    /// modularity with resolution `γ`), [`LeidenObjective::Cpm`] (Constant
    /// Potts Model) or [`LeidenObjective::ErdosRenyi`]. Works on directed and
    /// undirected graphs. The reported quality is the value of the chosen
    /// objective. Near linear time on sparse graphs.
    ///
    /// Binds [`igraph_community_leiden_simple`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_community_leiden_simple).
    ///
    /// # Errors
    /// For negative weights with the modularity or ER objectives, or vectors of
    /// wrong length.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::community::{LeidenObjective, LeidenOptions};
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)], 6, false)
    ///     .unwrap();
    /// rng::seed(1).unwrap();
    /// let res = g
    ///     .community_leiden_simple(None, LeidenObjective::Modularity, &LeidenOptions::default())
    ///     .unwrap();
    /// assert_eq!(res.nb_clusters, 2);
    /// let q = g.modularity(&res.membership, None, 1.0, true).unwrap();
    /// assert!((res.quality - q).abs() < 1e-12);
    /// ```
    pub fn community_leiden_simple(
        &self,
        weights: Option<&[f64]>,
        objective: LeidenObjective,
        options: &LeidenOptions<'_>,
    ) -> Result<Leiden> {
        self.community_check_weights("the weight vector", weights)?;
        opt_view!(w, weights);
        let mut membership = self.leiden_start(options.initial)?;
        let (mut nb, mut quality) = (0, 0.0);
        let iterations = options.iterations.map_or(-1, igraph_int_t::from);
        igraph_call!(igraph_community_leiden_simple(
            self,
            w,
            objective.into(),
            options.resolution,
            options.beta,
            options.initial.is_some(),
            iterations,
            &mut membership,
            &mut nb,
            &mut quality
        ))?;
        Ok(Leiden {
            membership: membership.into(),
            nb_clusters: nb as usize,
            quality,
        })
    }

    /// Greedy agglomerative modularity optimization (Clauset, Newman and Moore).
    ///
    /// Starting from singletons, the pair of communities whose merge increases
    /// modularity the most is merged repeatedly, building a full dendrogram
    /// (with the improvements of Wakita and Tsurumi). The returned
    /// [`Dendrogram`] contains the merges, the modularity before and after
    /// each merge, and the membership with the highest modularity. The graph
    /// must not have multi-edges; weights must be non-negative.
    /// Time complexity: O(|E| + |V| log²|V|) typically.
    ///
    /// Binds [`igraph_community_fastgreedy`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_community_fastgreedy).
    ///
    /// # Errors
    /// For multigraphs (merge the parallel edges first with
    /// [`Graph::simplify`], see also [`Graph::has_multiple`]) or invalid
    /// weights.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // The example of igraph's documentation.
    /// let g = Graph::from_edges(
    ///     &[(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)], 6, false).unwrap();
    /// let weights = [10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    /// let d = g.community_fastgreedy(Some(&weights)).unwrap();
    /// assert_eq!(d.merges, vec![(1, 0), (2, 6), (3, 4), (8, 5), (9, 7)]);
    /// ```
    pub fn community_fastgreedy(&self, weights: Option<&[f64]>) -> Result<Dendrogram> {
        self.community_check_weights("the weight vector", weights)?;
        opt_view!(w, weights);
        let mut merges = MatrixInt::new();
        let mut modularity = Vector::new();
        let mut membership = VectorInt::new();
        igraph_call!(igraph_community_fastgreedy(
            self,
            w,
            &mut merges,
            &mut modularity,
            &mut membership
        ))?;
        Ok(Dendrogram {
            num_vertices: self.vcount(),
            merges: matrix_to_merges(&merges),
            modularity: modularity.into(),
            membership: membership.into(),
        })
    }

    /// Walktrap community detection based on short random walks (Pons and Latapy).
    ///
    /// Vertex similarity is measured by random walks of length `steps`
    /// (typically 3–8, 4 or 5 being a reasonable default); communities are
    /// merged agglomeratively (Ward's method). Edge directions are ignored;
    /// weights must be positive. Isolated vertices are allowed. Time
    /// complexity: O(|V|² log|V|) typically.
    ///
    /// Binds [`igraph_community_walktrap`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_community_walktrap).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let triangle = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, false).unwrap();
    /// let d = triangle.community_walktrap(None, 4).unwrap();
    /// assert_eq!(d.merges, vec![(1, 2), (0, 3)]);
    /// assert_eq!(d.membership, vec![0, 0, 0]);
    /// ```
    pub fn community_walktrap(&self, weights: Option<&[f64]>, steps: usize) -> Result<Dendrogram> {
        self.community_check_weights("the weight vector", weights)?;
        opt_view!(w, weights);
        let mut merges = MatrixInt::new();
        let mut modularity = Vector::new();
        let mut membership = VectorInt::new();
        igraph_call!(igraph_community_walktrap(
            self,
            w,
            steps as igraph_int_t,
            &mut merges,
            &mut modularity,
            &mut membership
        ))?;
        Ok(Dendrogram {
            num_vertices: self.vcount(),
            merges: matrix_to_merges(&merges),
            modularity: modularity.into(),
            membership: membership.into(),
        })
    }

    /// Girvan–Newman community detection by repeatedly removing the edge with
    /// the highest betweenness.
    ///
    /// Betweenness is recomputed after each removal, until no edges remain;
    /// the resulting divisive hierarchy is returned as a dendrogram together
    /// with the removal order, the betweenness of each removed edge, the
    /// "bridges", the modularity of each division and the best membership.
    /// With `weights`, the ratio betweenness / weight decides which edge to
    /// remove (strong edges are removed later), and weights are used for
    /// modularity. With `lengths`, shortest paths take edge lengths into
    /// account. For directed graphs `directed` selects directed betweenness and
    /// modularity (splits are into weakly connected components).
    /// The dendrogram is computed with [`Graph::community_eb_get_merges`], so
    /// the order of the two ids within a merge may differ from igraph's
    /// output when modularity and membership are not requested.
    /// Time complexity: O(|V| |E|²).
    ///
    /// The first removed edge is the one with the highest
    /// [`Graph::edge_betweenness`] (divided by its weight) in the original graph.
    ///
    /// Binds [`igraph_community_edge_betweenness`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_community_edge_betweenness).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `weights` or `lengths` have a wrong
    /// length or contain invalid values.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // The example of igraph's documentation.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (1, 4)], 5, false).unwrap();
    /// let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let res = g.community_edge_betweenness(Some(&weights), None, false).unwrap();
    /// assert_eq!(res.removed_edges, vec![0, 1, 3, 4, 2, 5]);
    /// assert_eq!(res.edge_betweenness, vec![2.0, 3.5, 6.0, 2.0, 1.0, 1.0]);
    /// assert_eq!(res.merges, vec![(4, 1), (2, 0), (3, 5), (7, 6)]);
    /// assert_eq!(res.bridges, vec![5, 4, 3, 2]);
    /// ```
    pub fn community_edge_betweenness(
        &self,
        weights: Option<&[f64]>,
        lengths: Option<&[f64]>,
        directed: bool,
    ) -> Result<EdgeBetweennessCommunities> {
        self.community_check_weights("the weight vector", weights)?;
        self.community_check_weights("the length vector", lengths)?;
        opt_view!(w, weights);
        opt_view!(l, lengths);
        let mut removed = VectorInt::new();
        let mut eb = Vector::new();
        let mut merges = MatrixInt::new();
        let mut bridges = VectorInt::new();
        let mut modularity = Vector::new();
        let mut membership = VectorInt::new();
        igraph_call!(igraph_community_edge_betweenness(
            self,
            &mut removed,
            &mut eb,
            &mut merges,
            &mut bridges,
            &mut modularity,
            &mut membership,
            directed,
            w,
            l
        ))?;
        Ok(EdgeBetweennessCommunities {
            removed_edges: removed.into(),
            edge_betweenness: eb.into(),
            merges: matrix_to_merges(&merges),
            bridges: bridges.into(),
            modularity: modularity.into(),
            membership: membership.into(),
        })
    }

    /// Builds the dendrogram of a sequence of edge removals.
    ///
    /// Given an order in which *all* the edges are removed (e.g.
    /// [`EdgeBetweennessCommunities::removed_edges`], but any order works),
    /// the removal process is replayed backwards and each time two components
    /// get connected a merge is recorded (component ids below `vcount` are
    /// vertices, merged components are numbered from `vcount`). Modularity
    /// (weighted if `weights` is given, directed if `directed`) is computed
    /// for each division, and the best membership is returned.
    /// Time complexity: O(|E| + |V| log|V|).
    ///
    /// Binds [`igraph_community_eb_get_merges`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_community_eb_get_merges).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidEdgeId`] for an invalid edge id in `edges`, and
    /// [`ErrorKind::InvalidValue`] if `edges` is otherwise not a permutation
    /// of the edge ids.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let path = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// // Remove the middle edge first, then the outer ones.
    /// let res = path.community_eb_get_merges(false, &[1, 0, 2], None).unwrap();
    /// assert_eq!(res.dendrogram.merges, vec![(3, 2), (1, 0), (4, 5)]);
    /// assert_eq!(res.dendrogram.membership, vec![0, 0, 1, 1]);
    /// ```
    pub fn community_eb_get_merges(
        &self,
        directed: bool,
        edges: &[EdgeId],
        weights: Option<&[f64]>,
    ) -> Result<EdgeRemovalMerges> {
        self.community_check_weights("the weight vector", weights)?;
        // igraph (1.0.0 and 1.0.1) only checks that the ids are valid and
        // that there are enough of them: with a repeated (hence a missing) edge fewer merges happen
        // than the rows it allocates, and the rest of its outputs would be
        // left uninitialized. Require a permutation of the edge ids.
        let m = self.ecount();
        check_len("the edge removal order", edges.len(), m)?;
        let mut seen = vec![false; m];
        for &e in edges {
            let Some(idx) = usize::try_from(e).ok().filter(|&e| e < m) else {
                return Err(Error::new(
                    ErrorKind::InvalidEdgeId,
                    format!("invalid edge id {e} in the edge removal order"),
                ));
            };
            if std::mem::replace(&mut seen[idx], true) {
                return Err(Error::invalid(format!(
                    "edge {e} appears more than once in the edge removal order"
                )));
            }
        }
        let e = VectorInt::view(edges);
        opt_view!(w, weights);
        let mut merges = MatrixInt::new();
        let mut bridges = VectorInt::new();
        let mut modularity = Vector::new();
        let mut membership = VectorInt::new();
        igraph_call!(igraph_community_eb_get_merges(
            self,
            directed,
            e.as_ptr(),
            w,
            &mut merges,
            &mut bridges,
            &mut modularity,
            &mut membership
        ))?;
        Ok(EdgeRemovalMerges {
            dendrogram: Dendrogram {
                num_vertices: self.vcount(),
                merges: matrix_to_merges(&merges),
                modularity: modularity.into(),
                membership: membership.into(),
            },
            bridges: bridges.into(),
        })
    }

    /// Newman's leading eigenvector method (recursive spectral bisection).
    ///
    /// Starting from the connected components (see
    /// [`Graph::connected_components`]) or from `start`, each
    /// community is split in two according to the signs of the leading
    /// eigenvector of its generalized modularity matrix, as long as this
    /// increases modularity, performing at most `steps` splits (`None`: as
    /// many as possible). The initial division into `c` components (or `c`
    /// start communities) counts as `c − 1` steps, so at most
    /// `max(steps, c − 1) + 1` communities are returned. Start community ids
    /// must lie in `0..vcount`; communities with at most two vertices are
    /// never split. Edge directions are ignored. ARPACK is used with
    /// igraph's default options. Time complexity: O(|E| + |V|² steps).
    ///
    /// Binds [`igraph_community_leading_eigenvector`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_community_leading_eigenvector).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `start` or `weights` have a wrong
    /// length, `start` contains ids outside `0..vcount`, or the weights
    /// contain `NaN` or infinite values or their absolute sum exceeds `1e150`
    /// (igraph does not check these, and ARPACK would abort the process);
    /// [`ErrorKind::Arpack`] if ARPACK fails.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)], 6, false)
    ///     .unwrap();
    /// let res = g.community_leading_eigenvector(None, None, None).unwrap();
    /// assert_eq!(res.num_communities(), 2);
    /// assert!((res.modularity - 5.0 / 14.0).abs() < 1e-12);
    /// ```
    pub fn community_leading_eigenvector(
        &self,
        weights: Option<&[f64]>,
        steps: Option<usize>,
        start: Option<&[i64]>,
    ) -> Result<LeadingEigenvector> {
        self.leading_eigenvector_impl(
            weights,
            steps,
            start,
            None::<fn(&LeadingEigenvectorStep<'_>) -> _>,
        )
    }

    /// Like [`Graph::community_leading_eigenvector`], calling `callback` after
    /// each eigenvector computation.
    ///
    /// The callback receives a [`LeadingEigenvectorStep`] describing the
    /// community being split, the eigenvalue and eigenvector, and can compute
    /// products with the community's modularity matrix. Returning
    /// [`ControlFlow::Break`] stops the algorithm (the partition found so far
    /// is returned); a panic in the callback is propagated.
    ///
    /// Binds [`igraph_community_leading_eigenvector`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_community_leading_eigenvector)
    /// with a callback.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use std::ops::ControlFlow;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)], 6, false)
    ///     .unwrap();
    /// let mut eigenvalues = vec![];
    /// let res = g
    ///     .community_leading_eigenvector_with(None, None, None, |step| {
    ///         eigenvalues.push(step.eigenvalue());
    ///         // B v = λ v
    ///         let bv = step.multiply(step.eigenvector()).unwrap();
    ///         for (x, v) in bv.iter().zip(step.eigenvector()) {
    ///             assert!((x - step.eigenvalue() * v).abs() < 1e-6);
    ///         }
    ///         ControlFlow::Continue(())
    ///     })
    ///     .unwrap();
    /// assert_eq!(eigenvalues, res.eigenvalues);
    /// ```
    pub fn community_leading_eigenvector_with<F>(
        &self,
        weights: Option<&[f64]>,
        steps: Option<usize>,
        start: Option<&[i64]>,
        callback: F,
    ) -> Result<LeadingEigenvector>
    where
        F: FnMut(&LeadingEigenvectorStep<'_>) -> ControlFlow<()>,
    {
        self.leading_eigenvector_impl(weights, steps, start, Some(callback))
    }

    fn leading_eigenvector_impl<F>(
        &self,
        weights: Option<&[f64]>,
        steps: Option<usize>,
        start: Option<&[i64]>,
        mut callback: Option<F>,
    ) -> Result<LeadingEigenvector>
    where
        F: FnMut(&LeadingEigenvectorStep<'_>) -> ControlFlow<()>,
    {
        self.community_check_weights("the weight vector", weights)?;
        // igraph (1.0.0 and 1.0.1) only checks the length of the weights:
        // NaN, infinite or so large weights that the modularity matrix-vector
        // products overflow make ARPACK abort the whole process.
        check_finite("the weight vector", weights)?;
        if let Some(w) = weights
            && abs_sum(w) > MAX_ABS_WEIGHT_SUM
        {
            return Err(Error::invalid(format!(
                "the sum of the absolute weights must be at most {MAX_ABS_WEIGHT_SUM:e}, got {:e}",
                abs_sum(w)
            )));
        }
        self.check_vertex_vec("the start membership", start)?;
        // igraph (1.0.0 and 1.0.1) only warns about community ids >= |V|, but
        // then indexes a |V|-long work vector with them when building the
        // merges: reject them.
        let n = self.vcount() as i64;
        if let Some(s) = start
            && let Some(&c) = s.iter().find(|&&c| !(0..n).contains(&c))
        {
            return Err(Error::invalid(format!(
                "the start membership contains the community id {c}, \
                 ids must be in 0..{n}"
            )));
        }
        // igraph takes the maximum of the start vector, which is undefined
        // for the null graph: there is nothing to start from anyway.
        let start = start.filter(|s| !s.is_empty());
        opt_view!(w, weights);
        let mut merges = MatrixInt::new();
        let mut membership = start.map_or_else(VectorInt::new, VectorInt::from_slice);
        let mut modularity = 0.0;
        let mut eigenvalues = Vector::new();
        let mut eigenvectors = VectorList::new();
        let mut history = VectorInt::new();
        let steps = steps.map_or(-1, |s| s.min(igraph_int_t::MAX as usize) as igraph_int_t);
        let (cb, extra): (igraph_community_leading_eigenvector_callback_t, *mut c_void) =
            match callback.as_mut() {
                Some(f) => (Some(levc_trampoline::<F>), f as *mut F as *mut c_void),
                None => (None, ptr::null_mut()),
            };
        // ARPACK keeps thread-local state: refuse to nest it.
        let _arpack = crate::linalg::ArpackGuard::enter()?;
        igraph_call!(igraph_community_leading_eigenvector(
            self,
            w,
            &mut merges,
            &mut membership,
            steps,
            ptr::null_mut(),
            &mut modularity,
            start.is_some(),
            &mut eigenvalues,
            &mut eigenvectors,
            &mut history,
            cb,
            extra
        ))?;
        Ok(LeadingEigenvector {
            membership: membership.into(),
            merges: matrix_to_merges(&merges),
            modularity,
            eigenvalues: eigenvalues.into(),
            eigenvectors: eigenvectors.to_vecs(),
            history: decode_history(&history),
        })
    }

    /// Spinglass community detection (Reichardt and Bornholdt).
    ///
    /// Finds communities as the ground state of a Potts spin glass by
    /// simulated annealing, with at most [`SpinglassOptions::spins`]
    /// communities. The `Neg` implementation (Traag and Bruggeman) supports
    /// negative weights. Edge directions are ignored. The graph must be
    /// connected (check with [`Graph::is_connected`]; cluster each
    /// component separately otherwise). The result is random: seed the
    /// thread's generator with [`rng::seed`](crate::rng::seed) for
    /// reproducibility.
    ///
    /// Binds [`igraph_community_spinglass`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_community_spinglass).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] for disconnected graphs, invalid
    /// parameters or invalid weights. Beyond igraph's own checks (e.g. the
    /// temperatures must be both zero, or both positive with the starting one
    /// larger), the following are rejected on the Rust side, as igraph would
    /// loop forever or overflow on them: `NaN` or infinite weights, or
    /// weights whose absolute sum exceeds `1e150`; a `gamma` outside
    /// `[0, 1e150]` (or `NaN`); with the `Neg` implementation, a
    /// `gamma_minus` of magnitude above `1e150` (or `NaN`); a cooling factor
    /// outside `[0, 1)` (or `NaN`); temperatures outside `[0, 1e300]`; and a
    /// number of spins outside `2..=i32::MAX`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::community::SpinglassOptions;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)], 6, false)
    ///     .unwrap();
    /// rng::seed(42).unwrap();
    /// let res = g.community_spinglass(None, &SpinglassOptions::default()).unwrap();
    /// assert_eq!(res.num_communities(), 2);
    /// assert!((res.modularity - 5.0 / 14.0).abs() < 1e-9);
    /// ```
    pub fn community_spinglass(
        &self,
        weights: Option<&[f64]>,
        options: &SpinglassOptions,
    ) -> Result<Spinglass> {
        self.community_check_weights("the weight vector", weights)?;
        check_spinglass(weights, options, true)?;
        opt_view!(w, weights);
        let (mut modularity, mut temperature) = (0.0, 0.0);
        let mut membership = VectorInt::new();
        let mut csize = VectorInt::new();
        igraph_call!(igraph_community_spinglass(
            self,
            w,
            &mut modularity,
            &mut temperature,
            &mut membership,
            &mut csize,
            options.spins as igraph_int_t,
            options.parallel_update,
            options.start_temperature,
            options.stop_temperature,
            options.cooling_factor,
            options.update_rule.into(),
            options.gamma,
            options.implementation.into(),
            options.gamma_minus
        ))?;
        Ok(Spinglass {
            membership: membership.into(),
            csize: csize.into(),
            modularity,
            temperature,
        })
    }

    /// The spinglass community of a single vertex, without computing the
    /// whole partition.
    ///
    /// Uses [`SpinglassOptions::spins`], [`update_rule`](SpinglassOptions::update_rule)
    /// and [`gamma`](SpinglassOptions::gamma) (the other options are ignored).
    /// Also returns the cohesion and adhesion indices of the community and the
    /// number (or weight) of its inner and outer edges. The graph must be
    /// connected.
    ///
    /// Binds [`igraph_community_spinglass_single`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_community_spinglass_single).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`] if `vertex` is not a vertex of the
    /// graph, and [`ErrorKind::InvalidValue`] for disconnected graphs or
    /// invalid parameters: spins outside `2..=i32::MAX`, a `gamma` outside
    /// `[0, 1e150]` (or `NaN`), `NaN` or infinite weights (or weights whose
    /// absolute sum exceeds `1e150`).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::community::SpinglassOptions;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)], 6, false)
    ///     .unwrap();
    /// rng::seed(42).unwrap();
    /// let res = g.community_spinglass_single(None, 0, &SpinglassOptions::default()).unwrap();
    /// let mut community = res.community.clone();
    /// community.sort();
    /// assert_eq!(community, vec![0, 1, 2]);
    /// // Three edges inside the triangle, one (the bridge) leaving it.
    /// assert_eq!((res.inner_links, res.outer_links), (3.0, 1.0));
    /// ```
    pub fn community_spinglass_single(
        &self,
        weights: Option<&[f64]>,
        vertex: VertexId,
        options: &SpinglassOptions,
    ) -> Result<SpinglassSingle> {
        self.community_check_weights("the weight vector", weights)?;
        check_spinglass(weights, options, false)?;
        // igraph (1.0.0 and 1.0.1) accepts `vertex == vcount` (an off-by-one
        // in its check) and then silently returns an empty community.
        if !(0..self.vcount() as VertexId).contains(&vertex) {
            return Err(Error::new(
                ErrorKind::InvalidVertexId,
                format!(
                    "invalid vertex id {vertex} for a graph with {} vertices",
                    self.vcount()
                ),
            ));
        }
        opt_view!(w, weights);
        let mut community = VectorInt::new();
        let (mut cohesion, mut adhesion, mut inner, mut outer) = (0.0, 0.0, 0.0, 0.0);
        igraph_call!(igraph_community_spinglass_single(
            self,
            w,
            vertex,
            &mut community,
            &mut cohesion,
            &mut adhesion,
            &mut inner,
            &mut outer,
            options.spins as igraph_int_t,
            options.update_rule.into(),
            options.gamma
        ))?;
        Ok(SpinglassSingle {
            community: community.into(),
            cohesion,
            adhesion,
            inner_links: inner,
            outer_links: outer,
        })
    }

    /// Label propagation community detection (Raghavan, Albert and Kumara;
    /// fast variant of Traag and Šubelj).
    ///
    /// Every vertex repeatedly adopts the label that is dominant (highest
    /// total edge weight) among its neighbors, until all labels are dominant.
    /// See [`LabelPropagationOptions`] for directed propagation, initial and
    /// fixed labels and the variants. Weights must be non-negative. Ties are
    /// broken at random, so seed the thread's generator for reproducible
    /// results. In directed graphs, labels circulate freely only within
    /// strongly connected components (see [`Graph::connected_components`]
    /// with [`Connectedness::Strong`](crate::constants::Connectedness::Strong)).
    /// Unlabeled vertices unreachable from labeled ones are labeled in an
    /// extra step (each such undirected component gets its own label).
    /// Time complexity: O(|V| + |E|) per iteration.
    ///
    /// Binds [`igraph_community_label_propagation`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_community_label_propagation).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] for vectors of wrong length, negative or
    /// `NaN` weights, or initial labels outside `0..vcount` (negative ones
    /// excepted).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::community::LabelPropagationOptions;
    /// let path = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4)], 5, false).unwrap();
    /// // Fix the labels of both ends; the others are unlabeled.
    /// let initial = [0, -1, -1, -1, 1];
    /// let fixed = [true, false, false, false, true];
    /// let opts = LabelPropagationOptions { initial: Some(&initial), fixed: Some(&fixed),
    ///                                      ..Default::default() };
    /// let m = path.community_label_propagation(None, &opts).unwrap();
    /// // The fixed vertices keep distinct labels, and nobody gets a third one
    /// // (ties are broken at random, so the boundary may vary).
    /// assert_ne!(m[0], m[4]);
    /// assert!(m.iter().all(|&l| l == m[0] || l == m[4]));
    /// ```
    pub fn community_label_propagation(
        &self,
        weights: Option<&[f64]>,
        options: &LabelPropagationOptions<'_>,
    ) -> Result<Vec<i64>> {
        self.community_check_weights("the weight vector", weights)?;
        self.check_vertex_vec("the initial labels", options.initial)?;
        self.check_vertex_vec("the fixed labels", options.fixed)?;
        // igraph (1.0.0 and 1.0.1) rejects labels above |V| but accepts
        // |V| itself (an off-by-one), then indexes |V|-long work vectors
        // with it: reject every label outside 0..|V| here.
        let n = self.vcount() as i64;
        if let Some(&l) = options
            .initial
            .and_then(|init| init.iter().find(|&&l| l >= n))
        {
            return Err(Error::invalid(format!(
                "the initial label {l} is out of range, labels must be in 0..{n} \
                 (or negative for unlabeled vertices)"
            )));
        }
        // igraph (1.0.0 and 1.0.1) takes the maximum of the initial labels,
        // which aborts the process for the null graph (an empty vector): the
        // labels of zero vertices carry no information, so drop them.
        let (initial, fixed) = if n == 0 {
            (None, None)
        } else {
            (options.initial, options.fixed)
        };
        opt_view!(w, weights);
        let init = initial.map(VectorInt::view);
        let init = init.as_ref().map_or(ptr::null(), |v| v.as_ptr());
        let fixed = fixed.map(VectorBool::view);
        let fixed = fixed.as_ref().map_or(ptr::null(), |v| v.as_ptr());
        let mut membership = VectorInt::new();
        igraph_call!(igraph_community_label_propagation(
            self,
            &mut membership,
            options.mode.into(),
            w,
            init,
            fixed,
            options.variant.into()
        ))?;
        Ok(membership.into())
    }

    /// Infomap community detection: minimizes the map equation, the expected
    /// description length of a random walk (Rosvall and Bergstrom).
    ///
    /// The random walker follows out-edges proportionally to `edge_weights`
    /// (non-negative) and teleports with probability 0.15 to a vertex chosen
    /// proportionally to `vertex_weights` (positive). Edge directions are
    /// taken into account. The best of [`InfomapOptions::trials`] attempts is
    /// returned, with its code length (in bits). The attempts are random.
    ///
    /// Binds [`igraph_community_infomap`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_community_infomap).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] for weight vectors of wrong length or
    /// invalid weights or trials, and [`ErrorKind::Unimplemented`] if igraph
    /// was built without Infomap support (as documented since igraph 1.0.1).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::community::InfomapOptions;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)], 6, false)
    ///     .unwrap();
    /// rng::seed(7).unwrap();
    /// let res = g.community_infomap(None, None, &InfomapOptions::default()).unwrap();
    /// assert_eq!(res.num_communities(), 2);
    /// assert!(res.codelength > 0.0);
    /// ```
    pub fn community_infomap(
        &self,
        edge_weights: Option<&[f64]>,
        vertex_weights: Option<&[f64]>,
        options: &InfomapOptions,
    ) -> Result<Infomap> {
        self.community_check_weights("the edge weight vector", edge_weights)?;
        self.check_vertex_vec("the vertex weight vector", vertex_weights)?;
        // igraph stores the number of trials in an `unsigned int`: larger
        // values would be silently truncated (2^32 trials becoming 0 trials
        // and a meaningless all-zero membership).
        let trials = u32::try_from(options.trials)
            .ok()
            .filter(|&t| t >= 1)
            .ok_or_else(|| {
                Error::invalid(format!(
                    "the number of Infomap trials must be in 1..={}, got {}",
                    u32::MAX,
                    options.trials
                ))
            })?;
        opt_view!(ew, edge_weights);
        opt_view!(vw, vertex_weights);
        let mut membership = VectorInt::new();
        let mut codelength = 0.0;
        igraph_call!(igraph_community_infomap(
            self,
            ew,
            vw,
            igraph_int_t::from(trials),
            options.regularized,
            options.regularization_strength,
            &mut membership,
            &mut codelength
        ))?;
        Ok(Infomap {
            membership: membership.into(),
            codelength,
        })
    }

    /// Fluid communities: `k` "fluids" expand and contract on the graph
    /// until they reach an equilibrium (Parés et al., 2017).
    ///
    /// The graph must be simple and connected; edge directions are ignored,
    /// weights are not supported. `k` must be positive and at most the number
    /// of vertices. The result is random (seed the thread's generator for
    /// reproducibility). Time complexity: O(|E|).
    ///
    /// Binds [`igraph_community_fluid_communities`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_community_fluid_communities).
    ///
    /// # Errors
    /// For non-simple graphs (see [`Graph::is_simple`]), disconnected
    /// graphs (see [`Graph::is_connected`]) or an invalid `k`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)], 6, false)
    ///     .unwrap();
    /// rng::seed(3).unwrap();
    /// let m = g.community_fluid_communities(2).unwrap();
    /// assert_eq!(m.iter().filter(|&&c| c == m[0]).count(), 3);
    /// ```
    pub fn community_fluid_communities(&self, k: usize) -> Result<Vec<i64>> {
        let mut membership = VectorInt::new();
        igraph_call!(igraph_community_fluid_communities(
            self,
            k as igraph_int_t,
            &mut membership
        ))?;
        Ok(membership.into())
    }

    /// Voronoi community detection (Deritei et al.; Molnár et al.). *Experimental in igraph.*
    ///
    /// Generator vertices are chosen as those with the largest local relative
    /// density `s m / (m + k)` within `radius` (`s` being the strength of the
    /// vertex, `m` the number of edges within its first-order neighborhood and
    /// `k` the number of edges with a single endpoint in it), and every vertex
    /// is assigned to its closest generator (ties broken at random) using the
    /// edge `lengths` divided by the edge clustering coefficient
    /// ([`Graph::ecc`]), as in [`Graph::voronoi`]. `weights` are used to
    /// select the generators and compute modularity. `mode` selects distances from ([`NeighborMode::Out`])
    /// or to ([`NeighborMode::In`]) the generators in directed graphs. With
    /// `radius = None` the radius maximizing modularity is chosen automatically;
    /// an explicit radius must be non-negative (larger radii give fewer
    /// communities). The graph must be simple.
    ///
    /// Binds [`igraph_community_voronoi`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_community_voronoi).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] for a negative or `NaN` radius, vectors of
    /// wrong length, `NaN` or infinite lengths or weights, or non-simple
    /// graphs. With `radius = None`, the sum of the lengths times the number
    /// of vertices must also be at most `1e150` (igraph 1.0.1 aborts the
    /// process when its radius search overflows; this is checked here).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)], 6, false)
    ///     .unwrap();
    /// rng::seed(42).unwrap();
    /// let res = g.community_voronoi(None, None, NeighborMode::All, None).unwrap();
    /// assert_eq!(res.num_communities(), 2);
    /// assert_eq!(res.generators.len(), 2);
    /// // Each generator lies in its own community, and the two triangles are found.
    /// for (c, &v) in res.generators.iter().enumerate() {
    ///     assert_eq!(res.membership[v as usize], c as i64);
    /// }
    /// assert!((res.modularity - 5.0 / 14.0).abs() < 1e-12);
    /// ```
    pub fn community_voronoi(
        &self,
        lengths: Option<&[f64]>,
        weights: Option<&[f64]>,
        mode: NeighborMode,
        radius: Option<f64>,
    ) -> Result<Voronoi> {
        self.community_check_weights("the length vector", lengths)?;
        self.community_check_weights("the weight vector", weights)?;
        // igraph (1.0.0 and 1.0.1) accepts infinite lengths and weights. With
        // an automatic radius, infinite (or so large that shortest path
        // lengths overflow) lengths make its radius optimizer abort the whole
        // process on an assertion. Lengths are scaled by `1 / ECC <= |V|`
        // internally, so shortest paths are at most `|V| * sum(lengths)`.
        check_finite("the length vector", lengths)?;
        check_finite("the weight vector", weights)?;
        if radius.is_none()
            && let Some(l) = lengths
            && abs_sum(l) * self.vcount() as f64 > MAX_ABS_WEIGHT_SUM
        {
            return Err(Error::invalid(format!(
                "with an automatic radius, the sum of the edge lengths times the number \
                 of vertices must be at most {MAX_ABS_WEIGHT_SUM:e}"
            )));
        }
        let radius = match radius {
            None => -1.0,
            Some(r) if r >= 0.0 => r,
            Some(r) => {
                return Err(Error::invalid(format!(
                    "the Voronoi radius must be non-negative, got {r}"
                )));
            }
        };
        opt_view!(l, lengths);
        opt_view!(w, weights);
        let mut membership = VectorInt::new();
        let mut generators = VectorInt::new();
        let mut modularity = 0.0;
        self.with_fresh_multi_cache(|| {
            igraph_call!(igraph_community_voronoi(
                self,
                &mut membership,
                &mut generators,
                &mut modularity,
                l,
                w,
                mode.into(),
                radius
            ))
        })?;
        Ok(Voronoi {
            membership: membership.into(),
            generators: generators.into(),
            modularity,
        })
    }

    /// The partition with the highest possible modularity, by integer
    /// programming (Brandes et al., 2008) with GLPK.
    ///
    /// Exact modularity maximization is NP-complete: graphs up to ~50 vertices
    /// are fine, a few hundred may be possible. Directed graphs are supported.
    /// `resolution` is the `γ` of [`Graph::modularity`].
    ///
    /// Binds [`igraph_community_optimal_modularity`](https://igraph.org/c/html/latest/igraph-Community.html#igraph_community_optimal_modularity).
    ///
    /// # Errors
    /// [`ErrorKind::Unimplemented`] if igraph
    /// was built without GLPK.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)], 6, false)
    ///     .unwrap();
    /// match g.community_optimal_modularity(None, 1.0) {
    ///     Ok(best) => assert!((best.modularity - 5.0 / 14.0).abs() < 1e-9),
    ///     Err(e) => assert_eq!(e.kind(), ErrorKind::Unimplemented), // no GLPK
    /// }
    /// ```
    pub fn community_optimal_modularity(
        &self,
        weights: Option<&[f64]>,
        resolution: f64,
    ) -> Result<Clustering> {
        self.community_check_weights("the weight vector", weights)?;
        opt_view!(w, weights);
        let mut modularity = 0.0;
        let mut membership = VectorInt::new();
        igraph_call!(igraph_community_optimal_modularity(
            self,
            w,
            resolution,
            &mut modularity,
            &mut membership
        ))?;
        Ok(Clustering {
            membership: membership.into(),
            modularity,
        })
    }
}

// ---------------------------------------------------------------------------
// Hierarchical random graphs
// ---------------------------------------------------------------------------

/// A hierarchical random graph (HRG) model (`igraph_hrg_t`), after Clauset,
/// Moore and Newman.
///
/// An HRG with `n` leaves (the vertices of the modeled graph) is a binary
/// dendrogram with `n − 1` internal nodes, each labeled with a probability
/// `p`: two vertices are connected with the probability of their lowest
/// common ancestor. Internal node `i` has a left and a right child: a
/// non-negative child id is a leaf (vertex), a negative one `-j - 1` is the
/// internal node `j`.
///
/// An `Hrg` is always a valid, complete dendrogram: it is obtained by fitting
/// ([`Graph::hrg_fit`]), from the MCMC of [`Graph::hrg_consensus`] and
/// [`Graph::hrg_predict`], or from an explicit tree with [`Hrg::create`]. Its
/// storage is freed on drop.
///
/// ```
/// use igraph::prelude::*;
/// use igraph::community::Hrg;
///
/// // A root (vertex 0) splitting into leaf 3 and an internal node (1) whose
/// // children are leaf 4 and internal node 2 with leaves 5 and 6.
/// let tree = Graph::from_edges(&[(0, 3), (0, 1), (1, 4), (1, 2), (2, 5), (2, 6)], 7, true).unwrap();
/// let hrg = Hrg::create(&tree, &[1.0, 0.0, 0.0]).unwrap();
/// assert_eq!(hrg.size(), 4);
/// // Leaf 3 (first leaf, id 0) connects to everyone, the others never connect.
/// let sample = hrg.sample().unwrap();
/// assert_eq!(sample.edge_list(), vec![(0, 1), (0, 2), (0, 3)]);
/// ```
pub struct Hrg {
    raw: igraph_hrg_t,
}

// The HRG only owns igraph vectors, which are `Send + Sync`.
unsafe impl Send for Hrg {}
unsafe impl Sync for Hrg {}

impl Hrg {
    /// An initialized, placeholder HRG of `n` leaves (only for internal use,
    /// as an output argument).
    fn placeholder(n: usize) -> Result<Self> {
        let mut raw = MaybeUninit::<igraph_hrg_t>::uninit();
        igraph_call!(igraph_hrg_init(raw.as_mut_ptr(), n as igraph_int_t))?;
        // Dropping `raw` drops its five vectors, like `igraph_hrg_destroy`.
        Ok(Self {
            raw: unsafe { raw.assume_init() },
        })
    }

    /// Whether the five vectors describe a complete binary dendrogram rooted
    /// at internal node 0, which is what igraph's HRG code assumes (in igraph
    /// 1.0.0 and 1.0.1 it follows the child indices without any check, and
    /// loops forever on cycles).
    fn is_valid_dendrogram(&self) -> bool {
        let (left, right) = (self.left(), self.right());
        let internal = left.len();
        let leaves = internal + 1;
        if internal == 0
            || right.len() != internal
            || self.prob().len() != internal
            || self.edges().len() != internal
            || self.vertices().len() != internal
        {
            return false;
        }
        // Each leaf and each non-root internal node is the child of exactly
        // one internal node.
        let mut leaf_seen = vec![false; leaves];
        let mut internal_seen = vec![false; internal];
        internal_seen[0] = true;
        for &c in left.iter().chain(right) {
            let seen = if c >= 0 {
                leaf_seen.get_mut(c as usize)
            } else {
                // `!c` is `-c - 1`, without overflow for `i64::MIN`.
                internal_seen.get_mut(!c as usize).filter(|_| !c != 0)
            };
            match seen {
                Some(seen) if !*seen => *seen = true,
                _ => return false,
            }
        }
        // ... and every internal node is reachable from the root (no cycles).
        let mut reached = 1;
        let mut stack = vec![0usize];
        while let Some(i) = stack.pop() {
            for c in [left[i], right[i]] {
                if c < 0 {
                    reached += 1;
                    if reached > internal {
                        return false;
                    }
                    stack.push(!c as usize);
                }
            }
        }
        reached == internal
    }

    /// Checks the dendrogram invariant on an HRG produced by igraph.
    fn validated(self) -> Result<Self> {
        if self.is_valid_dendrogram() {
            Ok(self)
        } else {
            Err(Error::new(
                ErrorKind::Internal,
                "igraph produced an invalid HRG dendrogram",
            ))
        }
    }

    /// Raw pointer to the underlying C struct, for FFI calls.
    pub fn as_ptr(&self) -> *const igraph_hrg_t {
        &self.raw
    }

    /// The number of leaves, i.e. of vertices of the modeled graph.
    ///
    /// Binds [`igraph_hrg_size`](https://igraph.org/c/html/latest/igraph-HRG.html#igraph_hrg_size).
    pub fn size(&self) -> usize {
        unsafe { igraph_hrg_size(&self.raw) as usize }
    }

    /// Left child of each internal node (non-negative: leaf, `-j - 1`: internal node `j`).
    pub fn left(&self) -> &[i64] {
        &self.raw.left
    }

    /// Right child of each internal node (non-negative: leaf, `-j - 1`: internal node `j`).
    pub fn right(&self) -> &[i64] {
        &self.raw.right
    }

    /// Connection probability of each internal node.
    pub fn prob(&self) -> &[f64] {
        &self.raw.prob
    }

    /// Edge count stored for each internal node. For models fitted by MCMC
    /// ([`Graph::hrg_fit`], ...) this is the number of graph edges whose
    /// endpoints have this node as lowest common ancestor (they sum to the
    /// number of edges of the graph); [`Hrg::create`] stores the number of
    /// dendrogram edges below the node instead.
    pub fn edges(&self) -> &[i64] {
        &self.raw.edges
    }

    /// Number of leaves in the subtree of each internal node.
    pub fn vertices(&self) -> &[i64] {
        &self.raw.vertices
    }

    /// Creates an HRG from its dendrogram given as a directed binary tree
    /// and the probabilities of its internal nodes.
    ///
    /// `tree` must be a simple directed tree with edges pointing away from
    /// the root, in which every internal node has exactly two children; it
    /// has `n` leaves and `n − 1` internal nodes (at least 3 vertices in total).
    /// `prob` has one entry per internal node (`vcount / 2` values), and
    /// `prob[v]` is the probability of the internal tree vertex `v`: igraph
    /// indexes it by tree vertex id, so the internal vertices must be
    /// numbered first (`0..n-1`), as in the example of [`Hrg`] (this is
    /// checked). The leaves of the tree become the vertices `0..n` of the
    /// model, in increasing id order.
    ///
    /// Binds [`igraph_hrg_create`](https://igraph.org/c/html/latest/igraph-HRG.html#igraph_hrg_create).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `tree`
    /// is not a valid dendrogram or `prob` has a wrong length.
    pub fn create(tree: &Graph, prob: &[f64]) -> Result<Self> {
        let n = tree.vcount();
        if n < 3 {
            return Err(Error::invalid("HRG tree must have at least three vertices"));
        }
        check_len("the HRG probability vector", prob.len(), n / 2)?;
        // igraph (1.0.0 and 1.0.1) indexes `prob` by the id of every tree
        // vertex with out-degree 2 without a bounds check (only after checking
        // that the tree is directed): make sure these ids are in range.
        let degrees = if tree.is_directed() {
            tree.degree(.., NeighborMode::Out, crate::constants::Loops::Twice)?
        } else {
            Vec::new()
        };
        if let Some(v) = degrees
            .iter()
            .enumerate()
            .find(|&(v, &d)| d == 2 && v >= prob.len())
        {
            return Err(Error::invalid(format!(
                "internal vertex {} of the HRG tree has no probability: internal \
                 vertices must have the smallest ids",
                v.0
            )));
        }
        let p = Vector::view(prob);
        let mut hrg = Self::placeholder(0)?;
        igraph_call!(igraph_hrg_create(&mut hrg.raw, tree, p.as_ptr()))?;
        hrg.validated()
    }

    /// Draws a random graph from the model: every pair of vertices is
    /// connected independently with the probability of its lowest common
    /// ancestor in the dendrogram. The result is undirected and simple.
    ///
    /// Binds [`igraph_hrg_sample`](https://igraph.org/c/html/latest/igraph-HRG.html#igraph_hrg_sample).
    pub fn sample(&self) -> Result<Graph> {
        Graph::init_with(|g| unsafe { igraph_hrg_sample(&self.raw, g) })
    }

    /// Draws `num_samples` independent random graphs from the model.
    ///
    /// Binds `igraph_hrg_sample_many` (see the
    /// [HRG chapter](https://igraph.org/c/html/latest/igraph-HRG.html) of the
    /// C documentation).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::community::Hrg;
    /// // Two leaves joined by a root with probability 1/2.
    /// let tree = Graph::from_edges(&[(0, 1), (0, 2)], 3, true).unwrap();
    /// let hrg = Hrg::create(&tree, &[0.5]).unwrap();
    /// rng::seed(1).unwrap();
    /// let samples = hrg.sample_many(1000).unwrap();
    /// let with_edge = samples.iter().filter(|s| s.ecount() == 1).count();
    /// assert!((400..600).contains(&with_edge));
    /// ```
    pub fn sample_many(&self, num_samples: usize) -> Result<Vec<Graph>> {
        let mut list = GraphList::new();
        igraph_call!(igraph_hrg_sample_many(
            &self.raw,
            &mut list,
            num_samples as igraph_int_t
        ))?;
        Ok(list.into_vec())
    }

    /// The dendrogram as a directed tree with the probability of each tree
    /// vertex, see [`Graph::from_hrg_dendrogram`].
    pub fn dendrogram(&self) -> Result<(Graph, Vec<f64>)> {
        Graph::from_hrg_dendrogram(self)
    }
}

impl Clone for Hrg {
    fn clone(&self) -> Self {
        Self {
            raw: igraph_hrg_t {
                left: self.raw.left.clone(),
                right: self.raw.right.clone(),
                prob: self.raw.prob.clone(),
                vertices: self.raw.vertices.clone(),
                edges: self.raw.edges.clone(),
            },
        }
    }
}

impl fmt::Debug for Hrg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Hrg")
            .field("size", &self.size())
            .field("left", &self.left())
            .field("right", &self.right())
            .field("prob", &self.prob())
            .field("edges", &self.edges())
            .field("vertices", &self.vertices())
            .finish()
    }
}

impl PartialEq for Hrg {
    fn eq(&self, other: &Self) -> bool {
        self.left() == other.left()
            && self.right() == other.right()
            && self.prob() == other.prob()
            && self.edges() == other.edges()
            && self.vertices() == other.vertices()
    }
}

/// Result of [`Graph::hrg_consensus`].
#[derive(Debug, Clone, PartialEq)]
pub struct HrgConsensus {
    /// Parent of each node of the consensus tree (`-1` for roots): ids
    /// `0..n` are the vertices of the graph, larger ids are vertex groups.
    pub parents: Vec<i64>,
    /// For each internal node of the consensus tree (ids `n..`), how many
    /// times its split occurred in the samples.
    pub weights: Vec<f64>,
    /// The model the sampling started from: a copy of the given start model,
    /// or the model fitted to equilibrium when none was given.
    pub hrg: Hrg,
}

/// Result of [`Graph::hrg_predict`].
#[derive(Debug, Clone, PartialEq)]
pub struct HrgPrediction {
    /// The candidate missing edges, most likely first.
    pub edges: Vec<(VertexId, VertexId)>,
    /// The estimated probability of each candidate edge.
    pub prob: Vec<f64>,
    /// The model the sampling started from: a copy of the given start model,
    /// or the model fitted to equilibrium when none was given.
    pub hrg: Hrg,
}

impl igraph_t {
    fn check_hrg_start(&self, hrg: &Hrg) -> Result<()> {
        if hrg.size() != self.vcount() {
            return Err(Error::invalid(format!(
                "the HRG has {} leaves but the graph has {} vertices",
                hrg.size(),
                self.vcount()
            )));
        }
        Ok(())
    }

    /// Fits a hierarchical random graph model to the graph by Markov chain
    /// Monte Carlo (Clauset, Moore and Newman, 2008).
    ///
    /// Runs `steps` MCMC steps, or, with `steps = 0`, until a convergence
    /// criterion is met (the average log-likelihood over 65536 steps
    /// stabilizes), starting from a random dendrogram. The returned model is
    /// the most likely dendrogram visited. Edge directions, multi-edges and
    /// loops are ignored; the graph needs at least 3 vertices. The chain is
    /// random: seed the thread's generator for reproducible models.
    ///
    /// With `steps > 0`, igraph (1.0.0 and 1.0.1) only records a dendrogram
    /// when a step improves on the likelihood of the random initial one; on
    /// tiny graphs this may never happen, and the binding then runs the chain
    /// to equilibrium instead, so that a valid model is always returned.
    ///
    /// Binds [`igraph_hrg_fit`](https://igraph.org/c/html/latest/igraph-HRG.html#igraph_hrg_fit).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)], 6, false)
    ///     .unwrap();
    /// rng::seed(42).unwrap();
    /// let hrg = g.hrg_fit(1000).unwrap();
    /// assert_eq!(hrg.size(), 6);
    /// assert!(hrg.prob().iter().all(|p| (0.0..=1.0).contains(p)));
    /// ```
    pub fn hrg_fit(&self, steps: usize) -> Result<Hrg> {
        let steps = igraph_int_t::try_from(steps)
            .map_err(|_| Error::invalid("the number of MCMC steps is too large"))?;
        let mut hrg = Hrg::placeholder(self.vcount())?;
        igraph_call!(igraph_hrg_fit(self, &mut hrg.raw, false, steps))?;
        if steps > 0 && !hrg.is_valid_dendrogram() {
            // With a fixed number of steps igraph records a dendrogram only
            // when a step improves on the likelihood of its random initial
            // dendrogram. If none did, `hrg` is still the zero-filled
            // placeholder, which is not a tree: run to equilibrium instead,
            // which always records its result.
            igraph_call!(igraph_hrg_fit(self, &mut hrg.raw, false, 0))?;
        }
        hrg.validated()
    }

    /// Continues fitting an HRG to the graph, starting the MCMC from `hrg`
    /// (updated in place). See [`Graph::hrg_fit`].
    ///
    /// `hrg` is replaced by the most likely dendrogram visited, and is left
    /// unchanged if no step improves on its likelihood (or on error). With
    /// `steps = 0` the chain runs until convergence.
    ///
    /// Binds [`igraph_hrg_fit`](https://igraph.org/c/html/latest/igraph-HRG.html#igraph_hrg_fit)
    /// with `start = true`.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if the size of `hrg` differs from the
    /// number of vertices.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)], 6, false)
    ///     .unwrap();
    /// rng::seed(42).unwrap();
    /// let mut hrg = g.hrg_fit(100).unwrap();
    /// g.hrg_refit(&mut hrg, 1000).unwrap();
    /// // Still a model of `g`: its internal nodes account for all the edges.
    /// assert_eq!(hrg.size(), 6);
    /// assert_eq!(hrg.edges().iter().sum::<i64>(), 7);
    /// ```
    pub fn hrg_refit(&self, hrg: &mut Hrg, steps: usize) -> Result<()> {
        self.check_hrg_start(hrg)?;
        let steps = igraph_int_t::try_from(steps)
            .map_err(|_| Error::invalid("the number of MCMC steps is too large"))?;
        // Work on a copy, so that `hrg` stays a valid model (and unchanged)
        // if anything goes wrong.
        let mut fitted = hrg.clone();
        igraph_call!(igraph_hrg_fit(self, &mut fitted.raw, true, steps))?;
        *hrg = fitted.validated()?;
        Ok(())
    }

    /// Consensus tree of the HRG models sampled for the graph.
    ///
    /// Starting from `start` (or from a freshly fitted model when `None`),
    /// `num_samples` HRGs are sampled by MCMC and the splits present in the
    /// majority of them form the consensus tree (a forest when some splits
    /// are not supported by a majority: `-1` marks its roots).
    ///
    /// Binds [`igraph_hrg_consensus`](https://igraph.org/c/html/latest/igraph-HRG.html#igraph_hrg_consensus).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `start` does not match the graph or the
    /// graph has fewer than 3 vertices.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)], 6, false)
    ///     .unwrap();
    /// rng::seed(3).unwrap();
    /// let cons = g.hrg_consensus(None, 10).unwrap();
    /// // Vertices 0..6 come first, then the internal nodes of the consensus tree.
    /// assert_eq!(cons.parents.len(), 6 + cons.weights.len());
    /// assert!(cons.parents[..6].iter().all(|&p| p == -1 || p >= 6));
    /// ```
    pub fn hrg_consensus(&self, start: Option<&Hrg>, num_samples: usize) -> Result<HrgConsensus> {
        // igraph stores the number of samples in a C `int` here.
        let num_samples = i32::try_from(num_samples)
            .map_err(|_| Error::invalid("the number of samples is too large"))?;
        let mut hrg = match start {
            Some(h) => {
                self.check_hrg_start(h)?;
                h.clone()
            }
            None => Hrg::placeholder(self.vcount())?,
        };
        let mut parents = VectorInt::new();
        let mut weights = Vector::new();
        igraph_call!(igraph_hrg_consensus(
            self,
            &mut parents,
            &mut weights,
            &mut hrg.raw,
            start.is_some(),
            igraph_int_t::from(num_samples)
        ))?;
        Ok(HrgConsensus {
            parents: parents.into(),
            weights: weights.into(),
            hrg: hrg.validated()?,
        })
    }

    /// Predicts missing edges with HRG models.
    ///
    /// Samples `num_samples` HRGs (starting from `start`, or from a freshly
    /// fitted model when `None`) and estimates, for every non-adjacent vertex
    /// pair, the probability that the edge exists but was not observed.
    /// `num_bins` controls the resolution of the probabilities (e.g. 25);
    /// note that igraph keeps a histogram of `num_bins + 1` values for every
    /// vertex pair, i.e. O(|V|² num_bins) memory. The candidates are the
    /// pairs of distinct, non-adjacent vertices, most likely first.
    ///
    /// Binds [`igraph_hrg_predict`](https://igraph.org/c/html/latest/igraph-HRG.html#igraph_hrg_predict).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `start` does not match the graph, the
    /// graph has fewer than 3 or more than 46341 vertices (igraph counts the
    /// candidate pairs in a C `int`), or `num_bins` is zero.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A 4-cycle: the two diagonals are the only missing links.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], 4, false).unwrap();
    /// rng::seed(1).unwrap();
    /// let pred = g.hrg_predict(None, 10, 10).unwrap();
    /// let mut edges = pred.edges.clone();
    /// edges.sort();
    /// assert_eq!(edges, vec![(0, 2), (1, 3)]);
    /// ```
    pub fn hrg_predict(
        &self,
        start: Option<&Hrg>,
        num_samples: usize,
        num_bins: usize,
    ) -> Result<HrgPrediction> {
        // igraph allocates `num_bins + 1` bins per vertex pair, counted in a
        // C `int`; zero bins would give NaN probabilities.
        let num_bins = i32::try_from(num_bins)
            .ok()
            .filter(|&b| (1..i32::MAX).contains(&b))
            .ok_or_else(|| Error::invalid("the number of bins must be in 1..2^31 - 1"))?;
        let num_samples = igraph_int_t::try_from(num_samples)
            .map_err(|_| Error::invalid("the number of samples is too large"))?;
        // igraph reports too small graphs with a generic failure here, and
        // computes the number of vertex pairs `n (n - 1) / 2` in a C `int`,
        // which overflows (undefined behavior) beyond 46341 vertices.
        let n = self.vcount();
        if !(3..=46341).contains(&n) {
            return Err(Error::invalid(format!(
                "HRG link prediction needs 3 to 46341 vertices, the graph has {n}"
            )));
        }
        let mut hrg = match start {
            Some(h) => {
                self.check_hrg_start(h)?;
                h.clone()
            }
            None => Hrg::placeholder(self.vcount())?,
        };
        let mut edges = VectorInt::new();
        let mut prob = Vector::new();
        igraph_call!(igraph_hrg_predict(
            self,
            &mut edges,
            &mut prob,
            &mut hrg.raw,
            start.is_some(),
            num_samples,
            igraph_int_t::from(num_bins)
        ))?;
        let edges = edges
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[a, b]| (a, b))
            .collect();
        Ok(HrgPrediction {
            edges,
            prob: prob.into(),
            hrg: hrg.validated()?,
        })
    }

    /// Samples a graph from an HRG model (same as [`Hrg::sample`]); listed
    /// with the other random graph generators of [`games`](crate::games) in
    /// the C API.
    ///
    /// Binds [`igraph_hrg_game`](https://igraph.org/c/html/latest/igraph-HRG.html#igraph_hrg_game).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)], 6, false)
    ///     .unwrap();
    /// rng::seed(42).unwrap();
    /// let hrg = g.hrg_fit(0).unwrap();
    /// let sample = Graph::hrg_game(&hrg).unwrap();
    /// assert_eq!(sample.vcount(), 6);
    /// assert!(!sample.is_directed());
    /// ```
    pub fn hrg_game(hrg: &Hrg) -> Result<Graph> {
        Graph::init_with(|g| unsafe { igraph_hrg_game(g, &hrg.raw) })
    }

    /// The dendrogram of an HRG as a directed tree, with the probability of
    /// each tree vertex.
    ///
    /// The tree has `2n − 1` vertices: `0..n` are the leaves (the modeled
    /// vertices, probability `NaN`) and `n + i` is internal node `i` (with
    /// probability `hrg.prob()[i]`); edges point from parents to children.
    /// Draw it with a tree layout such as
    /// [`Graph::layout_reingold_tilford`].
    ///
    /// Binds [`igraph_from_hrg_dendrogram`](https://igraph.org/c/html/latest/igraph-HRG.html#igraph_from_hrg_dendrogram).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::community::Hrg;
    /// let tree = Graph::from_edges(&[(0, 3), (0, 1), (1, 4), (1, 2), (2, 5), (2, 6)], 7, true).unwrap();
    /// let hrg = Hrg::create(&tree, &[1.0, 0.5, 0.0]).unwrap();
    /// let (dendrogram, prob) = Graph::from_hrg_dendrogram(&hrg).unwrap();
    /// assert_eq!((dendrogram.vcount(), dendrogram.ecount()), (7, 6));
    /// assert!(dendrogram.is_tree(NeighborMode::Out).unwrap());
    /// assert!(prob[..4].iter().all(|p| p.is_nan()));
    /// assert_eq!(&prob[4..], hrg.prob());
    /// ```
    pub fn from_hrg_dendrogram(hrg: &Hrg) -> Result<(Graph, Vec<f64>)> {
        let mut prob = Vector::new();
        let g =
            Graph::init_with(|g| unsafe { igraph_from_hrg_dendrogram(g, &hrg.raw, &mut prob) })?;
        Ok((g, prob.into()))
    }
}

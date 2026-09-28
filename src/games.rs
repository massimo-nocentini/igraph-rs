//! Random graph generators, a.k.a. *games* (`igraph_games.h`).
//!
//! igraph calls its random graph models *games*. Every generator of this
//! module is an associated function of [`Graph`] returning a brand new
//! `Result<Graph>` (or a small result struct when the model also produces
//! vertex types or coordinates), while the two rewiring functions are
//! methods mutating an existing graph in place.
//!
//! All the games draw their random numbers from the calling thread's default
//! random number generator: seed it with [`rng::seed`](crate::rng::seed) to
//! obtain reproducible graphs. Two runs with the same seed and the same
//! arguments produce *exactly* the same graph. Every thread has its own
//! default generator (see [`rng`](crate::rng)), so seeding is per-thread and
//! concurrent threads never disturb each other's streams; an independent
//! generator of a chosen [`RngType`](crate::rng::RngType) can be installed
//! for the duration of a closure with [`Rng::scoped`](crate::rng::Rng::scoped):
//!
//! ```
//! use igraph::prelude::*;
//!
//! let sample = || {
//!     rng::seed(42).unwrap();
//!     Graph::erdos_renyi_game_gnm(50, 100, false, EdgeTypeSw::Simple, false).unwrap()
//! };
//! // Same seed, same graph, even in another thread.
//! let other_thread = std::thread::spawn(sample).join().unwrap();
//! assert!(sample().is_same_graph(&other_thread).unwrap());
//!
//! // A scoped Mersenne Twister leaves the default generator untouched.
//! let mut mt = Rng::new(RngType::Mt19937, 42).unwrap();
//! let g = mt.scoped(|| Graph::tree_game(20, false, RandomTreeMethod::Prufer)).unwrap();
//! assert!(g.is_tree(NeighborMode::All).unwrap());
//! ```
//!
//! # Example
//!
//! ```
//! use igraph::{games::BarabasiOptions, prelude::*};
//!
//! rng::seed(42).unwrap();
//! // An Erdős–Rényi G(n, m) graph: 1000 vertices, 1000 edges, simple.
//! let g = Graph::erdos_renyi_game_gnm(1000, 1000, false, EdgeTypeSw::Simple, false).unwrap();
//! let degrees = g.degree(.., NeighborMode::All, Loops::Twice).unwrap();
//! let mean = degrees.iter().sum::<i64>() as f64 / g.vcount() as f64;
//! assert_eq!(mean, 2.0); // handshake lemma: 2m / n
//!
//! // A scale-free graph grown by preferential attachment.
//! let ba = Graph::barabasi_game(100, &BarabasiOptions::default().with_m(2)).unwrap();
//! assert_eq!(ba.vcount(), 100);
//! assert_eq!(ba.ecount(), 1 + 98 * 2); // vertex 1 has only one vertex to attach to
//! ```
//!
//! # Provided functionality
//!
//! | Model | Rust | C function |
//! |---|---|---|
//! | Erdős–Rényi `G(n, m)` | [`Graph::erdos_renyi_game_gnm`] | `igraph_erdos_renyi_game_gnm` |
//! | Erdős–Rényi `G(n, p)` | [`Graph::erdos_renyi_game_gnp`] | `igraph_erdos_renyi_game_gnp` |
//! | Independent edge assignment | [`Graph::iea_game`] | `igraph_iea_game` |
//! | Barabási–Albert / Price | [`Graph::barabasi_game`] | `igraph_barabasi_game` |
//! | Preferential attachment with aging | [`Graph::barabasi_aging_game`] | `igraph_barabasi_aging_game` |
//! | Recent degree | [`Graph::recent_degree_game`] | `igraph_recent_degree_game` |
//! | Recent degree with aging | [`Graph::recent_degree_aging_game`] | `igraph_recent_degree_aging_game` |
//! | Growing random graph | [`Graph::growing_random_game`] | `igraph_growing_random_game` |
//! | Prescribed degree sequence | [`Graph::degree_sequence_game`] | `igraph_degree_sequence_game` |
//! | Random regular graph | [`Graph::k_regular_game`] | `igraph_k_regular_game` |
//! | Static fitness | [`Graph::static_fitness_game`] | `igraph_static_fitness_game` |
//! | Static power law | [`Graph::static_power_law_game`] | `igraph_static_power_law_game` |
//! | Chung–Lu (expected degrees) | [`Graph::chung_lu_game`] | `igraph_chung_lu_game` |
//! | Watts–Strogatz small world | [`Graph::watts_strogatz_game`] | `igraph_watts_strogatz_game` |
//! | Rewire edges | [`Graph::rewire_edges`] | `igraph_rewire_edges` |
//! | Rewire directed edge endpoints | [`Graph::rewire_directed_edges`] | `igraph_rewire_directed_edges` |
//! | Forest fire | [`Graph::forest_fire_game`] | `igraph_forest_fire_game` |
//! | Stochastic block model | [`Graph::sbm_game`] | `igraph_sbm_game` |
//! | Hierarchical SBM | [`Graph::hsbm_game`], [`Graph::hsbm_list_game`] | `igraph_hsbm_game`, `igraph_hsbm_list_game` |
//! | Preference (block model with random types) | [`Graph::preference_game`] | `igraph_preference_game` |
//! | Asymmetric preference | [`Graph::asymmetric_preference_game`] | `igraph_asymmetric_preference_game` |
//! | Callaway traits | [`Graph::callaway_traits_game`] | `igraph_callaway_traits_game` |
//! | Establishment | [`Graph::establishment_game`] | `igraph_establishment_game` |
//! | Geometric random graph | [`Graph::grg_game`] | `igraph_grg_game` |
//! | Last citation | [`Graph::lastcit_game`] | `igraph_lastcit_game` |
//! | Cited type | [`Graph::cited_type_game`] | `igraph_cited_type_game` |
//! | Citing–cited type | [`Graph::citing_cited_type_game`] | `igraph_citing_cited_type_game` |
//! | Interconnected islands | [`Graph::simple_interconnected_islands_game`] | `igraph_simple_interconnected_islands_game` |
//! | Correlated graphs | [`Graph::correlated_game`], [`Graph::correlated_pair_game`] | `igraph_correlated_game`, `igraph_correlated_pair_game` |
//! | Uniform random tree | [`Graph::tree_game`] | `igraph_tree_game` |
//! | Random dot product graph | [`Graph::dot_product_game`] | `igraph_dot_product_game` |
//!
//! # See also
//!
//! - Deterministic generators (rings, lattices, trees, famous graphs, graphs
//!   realizing a degree sequence) live in [`constructors`](crate::constructors),
//!   e.g. [`Graph::famous`], [`Graph::square_lattice`],
//!   [`Graph::realize_degree_sequence`].
//! - Random *bipartite* graphs: [`Graph::bipartite_game_gnp`],
//!   [`Graph::bipartite_game_gnm`], [`Graph::bipartite_iea_game`]; graphs from
//!   a hierarchical random graph model: [`Graph::hrg_game`].
//! - Degree-preserving randomization of an existing graph: [`Graph::rewire`];
//!   testing whether a degree sequence is realizable at all:
//!   [`is_graphical`](crate::mixing::is_graphical).
//! - Random spatial graphs from given points (the deterministic counterpart
//!   of [`Graph::grg_game`]): [`Graph::nearest_neighbor_graph`].
//! - Measuring what the models produce: [`Graph::average_path_length`],
//!   [`Graph::transitivity_avglocal_undirected`], [`Graph::maxdegree`],
//!   [`power_law_fit`](crate::misc::power_law_fit),
//!   [`Graph::community_multilevel`] (e.g. to recover the blocks of an
//!   [`sbm_game`](Graph::sbm_game)).
//!
//! # Allowed edge types
//!
//! Several generators take an `allowed_edge_types` argument (the C type
//! `igraph_edge_type_sw_t`) controlling whether self-loops and multi-edges may
//! be created. It accepts both the shared [`EdgeTypeSw`] enum (simple graphs,
//! loops *or* multi-edges) and the [`AllowedEdgeTypes`] flag set (defined in
//! [`constants`](crate::constants) and re-exported here; the graphicality
//! tests of [`mixing`](crate::mixing) use the very same type), which can also
//! express *loops and multi-edges* together:
//!
//! ```
//! use igraph::{games::AllowedEdgeTypes, prelude::*};
//!
//! rng::seed(7).unwrap();
//! let any = AllowedEdgeTypes::LOOPS | AllowedEdgeTypes::MULTI;
//! assert_eq!(EdgeTypeSw::Loops | EdgeTypeSw::Multi, any);
//! let g = Graph::erdos_renyi_game_gnm(3, 50, false, any, false).unwrap();
//! assert_eq!(g.ecount(), 50); // impossible without multi-edges on 3 vertices
//! assert!(g.has_multiple().unwrap());
//! ```

use crate::{
    constants::*,
    error::{Error, Result},
    ffi::*,
    graph::Graph,
    igraph_call,
    list::{MatrixList, VectorList},
    matrix::Matrix,
    vector::{Vector, VectorInt},
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Converts a count to `igraph_int_t`, failing on overflow.
fn int(n: usize, what: &str) -> Result<igraph_int_t> {
    igraph_int_t::try_from(n).map_err(|_| Error::invalid(format!("{what} is too large: {n}")))
}

/// Converts a vertex count to `igraph_int_t`, requiring at most
/// `IGRAPH_VCOUNT_MAX` (`i64::MAX - 1`) vertices.
///
/// Several games compute `n + 1` (or `n / bins + 1`) without overflow checks,
/// which is undefined behaviour in C for `n == i64::MAX`.
fn vertex_count(n: usize) -> Result<igraph_int_t> {
    let n = int(n, "number of vertices")?;
    if n > IGRAPH_VCOUNT_MAX_I64 {
        return Err(Error::invalid(format!(
            "number of vertices is too large: {n}"
        )));
    }
    Ok(n)
}

/// `IGRAPH_VCOUNT_MAX` (a macro, not exported by bindgen).
const IGRAPH_VCOUNT_MAX_I64: igraph_int_t = igraph_int_t::MAX - 1;
/// `IGRAPH_ECOUNT_MAX` (a macro, not exported by bindgen).
const IGRAPH_ECOUNT_MAX_I64: igraph_int_t = igraph_int_t::MAX / 2;
/// `IGRAPH_MAX_EXACT_REAL`: integers up to 2^53 are exact as doubles.
const MAX_EXACT_REAL: igraph_int_t = 1 << 53;

/// Converts an edge count to `igraph_int_t`, requiring at most
/// `IGRAPH_ECOUNT_MAX` (`i64::MAX / 2`) edges: some games compute
/// `2 * num_edges` without overflow checks.
fn edge_count(m: usize) -> Result<igraph_int_t> {
    let m = int(m, "number of edges")?;
    if m > IGRAPH_ECOUNT_MAX_I64 {
        return Err(Error::new(
            crate::error::ErrorKind::Overflow,
            format!("number of edges is too large: {m}"),
        ));
    }
    Ok(m)
}

/// Requires the (floating point) sum of `values` to be finite: igraph samples
/// from cumulative sums, and an infinite total makes rejection samplers loop
/// forever or produce NaN probabilities.
fn finite_sum(values: &[f64], what: &str) -> Result<f64> {
    let sum: f64 = values.iter().sum();
    if sum.is_finite() {
        Ok(sum)
    } else {
        Err(Error::invalid(format!("the sum of {what} must be finite")))
    }
}

/// Error for integer arithmetic on sizes that igraph would overflow.
fn overflow(what: &str) -> Error {
    Error::new(
        crate::error::ErrorKind::Overflow,
        format!("{what} overflows a 64-bit integer"),
    )
}

/// Vertex count of the hierarchical SBM games: igraph converts the doubles
/// `round(rho[i] * m)` back to integers, which is exact (and defined) only
/// below 2^53.
fn hsbm_vertex_count(n: usize) -> Result<igraph_int_t> {
    let n = int(n, "number of vertices")?;
    if n >= MAX_EXACT_REAL {
        return Err(Error::invalid(format!(
            "number of vertices must be below 2^53 for the HSBM, got {n}"
        )));
    }
    Ok(n)
}

/// Pointer to an optional viewed input vector (null when absent).
fn opt_ptr<V>(v: &Option<crate::vector::View<'_, V>>) -> *const V {
    v.as_ref().map_or(std::ptr::null(), |v| v.as_ptr())
}

/// Rejects NaN parameters.
///
/// igraph checks its real parameters with comparisons such as
/// `p < 0 || p > 1`, which NaN passes. Several games then loop forever
/// (NaN rewiring probabilities) or abort the process (a NaN island
/// probability), so NaN is rejected on the Rust side.
fn not_nan(x: f64, what: &str) -> Result<()> {
    if x.is_nan() {
        Err(Error::invalid(format!("{what} must not be NaN")))
    } else {
        Ok(())
    }
}

/// Rejects NaN entries of a matrix (see [`not_nan`]).
fn no_nan_entries(m: &Matrix, what: &str) -> Result<()> {
    if m.as_slice().iter().any(|x| x.is_nan()) {
        Err(Error::invalid(format!("{what} must not contain NaN")))
    } else {
        Ok(())
    }
}

/// Requires every value to be finite and non-negative.
///
/// igraph samples from cumulative sums of these values: infinite entries
/// make several games loop forever, and some checks miss negative values.
fn finite_non_negative(values: &[f64], what: &str) -> Result<()> {
    match values.iter().find(|x| !(x.is_finite() && **x >= 0.0)) {
        Some(x) => Err(Error::invalid(format!(
            "{what} must be finite and non-negative, got {x}"
        ))),
        None => Ok(()),
    }
}

/// Requires a sampling distribution (finite, non-negative weights) with at
/// least one positive weight.
///
/// With only zero weights igraph computes the type index `-1` and indexes out
/// of bounds (an assertion failure that aborts the process).
fn distribution(values: &[f64], what: &str) -> Result<()> {
    finite_non_negative(values, what)?;
    if values.iter().all(|&x| x == 0.0) {
        return Err(Error::invalid(format!(
            "{what} must contain at least one positive value"
        )));
    }
    Ok(())
}

/// igraph divides by zero (`SIGFPE`) when asked to move an endpoint to a
/// vertex other than the fixed one in a single-vertex graph: reject it.
fn check_rewirable(graph: &Graph, prob: f64, loops: bool) -> Result<()> {
    not_nan(prob, "the rewiring probability")?;
    if !loops && prob > 0.0 && graph.vcount() == 1 && graph.ecount() > 0 {
        return Err(Error::invalid(
            "cannot rewire the edges of a single-vertex graph without allowing self-loops",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Allowed edge types
// ---------------------------------------------------------------------------

/// Which non-simple edges a random generator may create, as a set of flags
/// (the C type `igraph_edge_type_sw_t`).
///
/// This is [`constants::AllowedEdgeTypes`](crate::constants::AllowedEdgeTypes)
/// (also re-exported as `mixing::AllowedEdgeTypes` and
/// `constructors::AllowedEdgeTypes`), re-exported here because most
/// generators of this module take it: see there for its constants and
/// conversions. Every generator argument named
/// `allowed_edge_types` accepts anything convertible into it, i.e. an
/// [`EdgeTypeSw`] or a combination such as `EdgeTypeSw::Loops | EdgeTypeSw::Multi`.
pub use crate::constants::AllowedEdgeTypes;

fn sw(allowed: impl Into<AllowedEdgeTypes>) -> igraph_edge_type_sw_t {
    allowed.into().to_raw()
}

// ---------------------------------------------------------------------------
// Result structs and option structs
// ---------------------------------------------------------------------------

/// A random graph whose vertices carry a *type* (category), as produced by
/// the type-based games ([`Graph::preference_game`],
/// [`Graph::callaway_traits_game`], [`Graph::establishment_game`]).
#[derive(Debug, Clone, PartialEq)]
pub struct TypedGraph {
    /// The generated graph.
    pub graph: Graph,
    /// `types[v]` is the type of vertex `v`, numbered from zero.
    pub types: Vec<i64>,
}

/// Result of [`Graph::asymmetric_preference_game`]: every vertex has an
/// *out-type* and an *in-type*.
#[derive(Debug, Clone, PartialEq)]
pub struct AsymmetricTypedGraph {
    /// The generated (directed) graph.
    pub graph: Graph,
    /// `out_types[v]` is the outgoing type of vertex `v`.
    pub out_types: Vec<i64>,
    /// `in_types[v]` is the incoming type of vertex `v`.
    pub in_types: Vec<i64>,
}

/// Result of [`Graph::grg_game`]: a geometric random graph together with the
/// positions of its vertices in the unit square.
#[derive(Debug, Clone, PartialEq)]
pub struct GeometricGraph {
    /// The generated graph.
    pub graph: Graph,
    /// x coordinates of the vertices (sorted increasingly: vertex ids follow
    /// the x order).
    pub x: Vec<f64>,
    /// y coordinates of the vertices.
    pub y: Vec<f64>,
}

/// Options of [`Graph::barabasi_game`], with the defaults of the classic
/// undirected Barabási–Albert model (`m = 1`, `power = 1`, `A = 1`,
/// [`BarabasiAlgorithm::Psumtree`]).
///
/// ```
/// use igraph::prelude::*;
/// use igraph::games::BarabasiOptions;
///
/// let opts = BarabasiOptions::default().with_m(3).with_directed(true).with_power(0.5);
/// assert_eq!((opts.m, opts.directed, opts.power), (3, true, 0.5));
/// ```
#[derive(Debug, Clone)]
pub struct BarabasiOptions<'a> {
    /// Exponent of the preferential attachment: the probability of citing a
    /// vertex of degree `d` is proportional to `d^power + a` (default `1`).
    pub power: f64,
    /// Number of edges added with each new vertex, used when `outseq` is
    /// `None` (default `1`).
    pub m: usize,
    /// Explicit number of edges added with each vertex (the first entry is
    /// ignored, as the first vertex cannot cite anybody). With `start_from`,
    /// it refers only to the newly added vertices. Default `None`.
    pub outseq: Option<&'a [i64]>,
    /// Whether the out-degree also counts in the attractiveness (i.e. total
    /// degree instead of in-degree). Ignored (assumed `true`) for undirected
    /// graphs. Default `false`.
    pub outpref: bool,
    /// The constant attractiveness `A` of vertices (default `1`).
    pub a: f64,
    /// Whether to create a directed graph (default `false`).
    pub directed: bool,
    /// The sampling algorithm (default [`BarabasiAlgorithm::Psumtree`], that
    /// creates simple graphs).
    pub algorithm: BarabasiAlgorithm,
    /// An optional non-empty starting graph; by default the process starts
    /// from a single vertex (the C documentation mentions a clique of size
    /// `m`, but the implementation starts from one vertex in igraph 1.0.0 and
    /// 1.0.1). The generated graph contains it as its first vertices and
    /// edges.
    pub start_from: Option<&'a Graph>,
}

impl Default for BarabasiOptions<'_> {
    fn default() -> Self {
        Self {
            power: 1.0,
            m: 1,
            outseq: None,
            outpref: false,
            a: 1.0,
            directed: false,
            algorithm: BarabasiAlgorithm::Psumtree,
            start_from: None,
        }
    }
}

impl<'a> BarabasiOptions<'a> {
    /// Sets [`power`](Self::power).
    pub fn with_power(mut self, power: f64) -> Self {
        self.power = power;
        self
    }
    /// Sets [`m`](Self::m).
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }
    /// Sets [`outseq`](Self::outseq).
    pub fn with_outseq(mut self, outseq: &'a [i64]) -> Self {
        self.outseq = Some(outseq);
        self
    }
    /// Sets [`outpref`](Self::outpref).
    pub fn with_outpref(mut self, outpref: bool) -> Self {
        self.outpref = outpref;
        self
    }
    /// Sets the constant attractiveness [`a`](Self::a).
    pub fn with_a(mut self, a: f64) -> Self {
        self.a = a;
        self
    }
    /// Sets [`directed`](Self::directed).
    pub fn with_directed(mut self, directed: bool) -> Self {
        self.directed = directed;
        self
    }
    /// Sets [`algorithm`](Self::algorithm).
    pub fn with_algorithm(mut self, algorithm: BarabasiAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }
    /// Sets [`start_from`](Self::start_from).
    pub fn with_start_from(mut self, start_from: &'a Graph) -> Self {
        self.start_from = Some(start_from);
        self
    }
}

/// Options of [`Graph::barabasi_aging_game`].
///
/// The attractiveness of a vertex with (in-)degree `k` and age `l` is
/// `(deg_coef * k^pa_exp + zero_deg_appeal) * (age_coef * l^aging_exp + zero_age_appeal)`.
/// The defaults (those of R igraph's `sample_pa_age`) describe linear
/// preferential attachment without aging.
#[derive(Debug, Clone)]
pub struct BarabasiAgingOptions<'a> {
    /// Edges added per time step when `outseq` is `None` (default `1`).
    pub m: usize,
    /// Edges added in each time step (overrides `m`). Default `None`.
    pub outseq: Option<&'a [i64]>,
    /// Whether edges initiated by a vertex count in its attractiveness
    /// (default `false`).
    pub outpref: bool,
    /// Preferential attachment exponent (default `1`).
    pub pa_exp: f64,
    /// Aging exponent, usually negative (default `0`, i.e. no aging).
    pub aging_exp: f64,
    /// Number of age bins (default `300`).
    pub aging_bins: usize,
    /// Degree dependent attractiveness of zero-degree vertices (default `1`).
    pub zero_deg_appeal: f64,
    /// Age dependent attractiveness of age-zero vertices (default `0`).
    pub zero_age_appeal: f64,
    /// Coefficient of the degree term (default `1`).
    pub deg_coef: f64,
    /// Coefficient of the age term (default `1`).
    pub age_coef: f64,
    /// Whether to create a directed graph (default `true`).
    pub directed: bool,
}

impl Default for BarabasiAgingOptions<'_> {
    fn default() -> Self {
        Self {
            m: 1,
            outseq: None,
            outpref: false,
            pa_exp: 1.0,
            aging_exp: 0.0,
            aging_bins: 300,
            zero_deg_appeal: 1.0,
            zero_age_appeal: 0.0,
            deg_coef: 1.0,
            age_coef: 1.0,
            directed: true,
        }
    }
}

/// Options of [`Graph::recent_degree_game`].
///
/// The probability that a vertex is cited is proportional to
/// `k^power + zero_appeal`, where `k` is the number of edges it gained in the
/// last `window` time steps.
#[derive(Debug, Clone)]
pub struct RecentDegreeOptions<'a> {
    /// Exponent of the recent degree (default `1`).
    pub power: f64,
    /// Size of the time window (default `1`).
    pub window: usize,
    /// Edges added per time step when `outseq` is `None` (default `1`).
    pub m: usize,
    /// Edges added in each time step (overrides `m`). Default `None`.
    pub outseq: Option<&'a [i64]>,
    /// Whether edges originated by a vertex also count as recent edges
    /// (default `false`).
    pub outpref: bool,
    /// Attractiveness of the vertices without recent edges (default `1`).
    pub zero_appeal: f64,
    /// Whether to create a directed graph (default `false`).
    pub directed: bool,
}

impl Default for RecentDegreeOptions<'_> {
    fn default() -> Self {
        Self {
            power: 1.0,
            window: 1,
            m: 1,
            outseq: None,
            outpref: false,
            zero_appeal: 1.0,
            directed: false,
        }
    }
}

/// Options of [`Graph::recent_degree_aging_game`].
///
/// The attractiveness is `(k^pa_exp + zero_appeal) * l^aging_exp`, where `k`
/// is the number of edges gained in the last `window` steps and `l` the age.
#[derive(Debug, Clone)]
pub struct RecentDegreeAgingOptions<'a> {
    /// Edges added per time step when `outseq` is `None` (default `1`).
    pub m: usize,
    /// Edges added in each time step (overrides `m`). Default `None`.
    pub outseq: Option<&'a [i64]>,
    /// Whether edges initiated by a vertex are counted (default `false`).
    pub outpref: bool,
    /// Preferential attachment exponent (default `1`).
    pub pa_exp: f64,
    /// Aging exponent, usually negative (default `0`, i.e. no aging).
    pub aging_exp: f64,
    /// Number of age bins (default `300`).
    pub aging_bins: usize,
    /// Size of the time window (default `1`).
    pub window: usize,
    /// Degree dependent attractiveness of vertices without recent edges
    /// (default `1`).
    pub zero_appeal: f64,
    /// Whether to create a directed graph (default `false`).
    pub directed: bool,
}

impl Default for RecentDegreeAgingOptions<'_> {
    fn default() -> Self {
        Self {
            m: 1,
            outseq: None,
            outpref: false,
            pa_exp: 1.0,
            aging_exp: 0.0,
            aging_bins: 300,
            window: 1,
            zero_appeal: 1.0,
            directed: false,
        }
    }
}

// ---------------------------------------------------------------------------
// The games
// ---------------------------------------------------------------------------

impl igraph_t {
    /// Generates a uniformly random graph with a fixed number of vertices and
    /// edges: the Erdős–Rényi `G(n, m)` model.
    ///
    /// Among all graphs on `num_vertices` vertices with exactly `num_edges`
    /// edges (and respecting `allowed_edge_types`, which accepts an
    /// [`EdgeTypeSw`] or an [`AllowedEdgeTypes`]), one is drawn uniformly at
    /// random. With `edge_labeled = true` the sampling is uniform over
    /// *ordered edge lists* instead (see [`Graph::iea_game`]); pass `false` for
    /// the classic model.
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// See also [`Graph::erdos_renyi_game_gnp`] (independent edges) and
    /// [`Graph::bipartite_game_gnm`] (the bipartite analogue).
    ///
    /// Binds [`igraph_erdos_renyi_game_gnm`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_erdos_renyi_game_gnm).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `num_edges` is larger than the number of available vertex pairs (for
    /// simple graphs).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(42).unwrap();
    /// let g = Graph::erdos_renyi_game_gnm(10, 20, true, EdgeTypeSw::Simple, false).unwrap();
    /// assert_eq!((g.vcount(), g.ecount(), g.is_directed()), (10, 20, true));
    /// // Only 45 vertex pairs exist in a simple undirected graph on 10 vertices.
    /// let err = Graph::erdos_renyi_game_gnm(10, 46, false, EdgeTypeSw::Simple, false).unwrap_err();
    /// assert_eq!(err.kind(), ErrorKind::InvalidValue);
    /// ```
    pub fn erdos_renyi_game_gnm(
        num_vertices: usize,
        num_edges: usize,
        directed: bool,
        mode: impl Into<AllowedEdgeTypes>,
        edge_labeled: bool,
    ) -> Result<Graph> {
        let n = int(num_vertices, "number of vertices")?;
        let m = int(num_edges, "number of edges")?;
        let mode = sw(mode);
        Graph::init_with(|g| unsafe {
            igraph_erdos_renyi_game_gnm(g, n, m, directed, mode, edge_labeled)
        })
    }

    /// Generates a random graph where every vertex pair is connected
    /// independently with probability `p`: the Erdős–Rényi `G(n, p)` (or
    /// Gilbert) model.
    ///
    /// When multi-edges are allowed, `p` is the *expected number* of edges
    /// between any vertex pair (multiplicities are geometric:
    /// `P(m edges) = q (1 - q)^m` with `q = 1 / (1 + p)`). The expected mean
    /// degree is `p (n - 1)` without self-loops, `p (n + 1)` for undirected
    /// graphs with self-loops and `p n` for directed ones with self-loops; set
    /// `p = k / n` for a mean degree of about `k`. `edge_labeled = true`
    /// samples uniformly from ordered edge lists instead; igraph implements
    /// this variant only when multi-edges are allowed. Use `false` for the
    /// classic model.
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// See also [`Graph::erdos_renyi_game_gnm`] (fixed edge count),
    /// [`Graph::bipartite_game_gnp`] and [`Graph::mean_degree`].
    ///
    /// Binds [`igraph_erdos_renyi_game_gnp`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_erdos_renyi_game_gnp).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `p` is
    /// not in `[0, 1]` for graphs without multi-edges (or is negative for
    /// multigraphs), or is NaN or infinite, or if `num_vertices` exceeds
    /// igraph's maximum vertex count `i64::MAX - 1` (checked on the Rust
    /// side: igraph 1.0.1 overflows computing `n + 1` for loopy undirected
    /// graphs, and mishandles an infinite `p`);
    /// [`ErrorKind::Overflow`](crate::ErrorKind::Overflow) if, with
    /// `edge_labeled = true`, the expected number of edges exceeds 2^56
    /// (igraph 1.0.1 would overflow doubling the sampled edge count);
    /// [`ErrorKind::Unimplemented`](crate::ErrorKind::Unimplemented) for
    /// `edge_labeled = true` without multi-edges.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // p = 1 gives the complete graph, p = 0 the empty graph.
    /// let k5 = Graph::erdos_renyi_game_gnp(5, 1.0, false, EdgeTypeSw::Simple, false).unwrap();
    /// assert!(k5.is_same_graph(&Graph::full(5, false, false).unwrap()).unwrap());
    /// let e5 = Graph::erdos_renyi_game_gnp(5, 0.0, false, EdgeTypeSw::Simple, false).unwrap();
    /// assert_eq!(e5.ecount(), 0);
    ///
    /// // Mean degree p (n - 1) ≈ 4.
    /// rng::seed(42).unwrap();
    /// let g = Graph::erdos_renyi_game_gnp(2001, 0.002, false, EdgeTypeSw::Simple, false).unwrap();
    /// assert!((g.mean_degree(true).unwrap() - 4.0).abs() < 0.3);
    /// ```
    pub fn erdos_renyi_game_gnp(
        num_vertices: usize,
        p: f64,
        directed: bool,
        allowed_edge_types: impl Into<AllowedEdgeTypes>,
        edge_labeled: bool,
    ) -> Result<Graph> {
        let n = vertex_count(num_vertices)?;
        not_nan(p, "the edge probability")?;
        let allowed = allowed_edge_types.into();
        // With multi-edges igraph accepts any `p >= 0`; an infinite one gives
        // NaN sampling parameters (silently the empty graph, after converting
        // NaN to an integer in the edge-labeled variant).
        if p.is_infinite() {
            return Err(Error::invalid(
                "the expected edge multiplicity `p` must be finite",
            ));
        }
        if edge_labeled && allowed.multi {
            // igraph draws the edge count from a geometric distribution
            // with mean `max_edges * p` and computes `2 * count` unchecked:
            // keep the mean far below `IGRAPH_ECOUNT_MAX` (such a graph
            // could not be allocated anyway).
            let nf = n as f64;
            let max_edges = match (directed, allowed.loops) {
                (true, true) => nf * nf,
                (true, false) => nf * (nf - 1.0),
                (false, true) => nf * (nf + 1.0) / 2.0,
                (false, false) => nf * (nf - 1.0) / 2.0,
            };
            if max_edges * p > (IGRAPH_ECOUNT_MAX_I64 >> 6) as f64 {
                return Err(Error::new(
                    crate::error::ErrorKind::Overflow,
                    format!(
                        "the expected number of edges ({:e}) is too large",
                        max_edges * p
                    ),
                ));
            }
        }
        let mode = allowed.to_raw();
        Graph::init_with(|g| unsafe {
            igraph_erdos_renyi_game_gnp(g, n, p, directed, mode, edge_labeled)
        })
    }

    /// Generates a random multigraph by *independent edge assignment* (IEA):
    /// each of the `num_edges` edges is assigned to a uniformly random ordered
    /// vertex pair, independently of the others.
    ///
    /// This is uniform sampling of *edge-labeled* graphs: all simple graphs
    /// have the same probability, while multigraphs are down-weighted by the
    /// factorials of their edge multiplicities. `loops` controls whether
    /// self-loops can be produced. This function is marked *experimental* in
    /// igraph.
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// See also [`Graph::erdos_renyi_game_gnm`] with `edge_labeled = true`
    /// and [`Graph::bipartite_iea_game`].
    ///
    /// Binds [`igraph_iea_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_iea_game).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(1).unwrap();
    /// // Two vertices, 10 edges, no loops: all edges are parallel.
    /// let g = Graph::iea_game(2, 10, false, false).unwrap();
    /// assert!(!g.has_loop().unwrap());
    /// assert_eq!(g.count_multiple(..).unwrap(), vec![10; 10]);
    /// ```
    pub fn iea_game(
        num_vertices: usize,
        num_edges: usize,
        directed: bool,
        loops: bool,
    ) -> Result<Graph> {
        let n = int(num_vertices, "number of vertices")?;
        let m = int(num_edges, "number of edges")?;
        Graph::init_with(|g| unsafe { igraph_iea_game(g, n, m, directed, loops) })
    }

    /// Generates a graph by preferential attachment: the Barabási–Albert
    /// model and its variants (Price model, non-linear attachment).
    ///
    /// Starting from a single vertex (or from
    /// [`start_from`](BarabasiOptions::start_from)), vertices are added one at
    /// a time; each new vertex cites `m` (or `outseq[i]`) existing vertices,
    /// chosen with probability proportional to `d^power + A`, where `d` is the
    /// in-degree (or total degree with `outpref`, and always in undirected
    /// graphs). See [`BarabasiOptions`] for all the parameters and
    /// [`BarabasiAlgorithm`] for the sampling algorithms: `Bag` only supports
    /// `power = 1, A = 1` and may create multi-edges, `Psumtree` creates simple
    /// graphs, `PsumtreeMultiple` allows multi-edges.
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// See also [`Graph::barabasi_aging_game`] and
    /// [`Graph::recent_degree_game`] for variants of preferential attachment,
    /// and [`power_law_fit`](crate::misc::power_law_fit) to estimate the
    /// exponent of the resulting degree distribution.
    ///
    /// Binds [`igraph_barabasi_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_barabasi_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for a
    /// non-positive `A` (negative `A` when the total degree is used), negative
    /// `outseq` entries or an `outseq` whose length differs from the number
    /// of new vertices, an empty starting graph or one with more than
    /// `num_vertices` vertices, `power != 1` or `A != 1` with
    /// [`BarabasiAlgorithm::Bag`], or an undirected starting graph for a
    /// directed result without `outpref`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::games::BarabasiOptions;
    ///
    /// rng::seed(42).unwrap();
    /// let g = Graph::barabasi_game(1000, &BarabasiOptions::default().with_m(3)).unwrap();
    /// // Vertex 1 cites vertex 0 once, vertex 2 cites 0 and 1, then 3 edges each.
    /// assert_eq!(g.ecount(), 1 + 2 + 997 * 3);
    /// // Hubs emerge: the maximum degree is far above the mean (about 6).
    /// let max = g.maxdegree(.., NeighborMode::All, Loops::Twice).unwrap();
    /// assert!(max > 30);
    /// ```
    pub fn barabasi_game(num_vertices: usize, options: &BarabasiOptions<'_>) -> Result<Graph> {
        let n = int(num_vertices, "number of vertices")?;
        let m = int(options.m, "m")?;
        let outseq = options.outseq.map(VectorInt::view);
        let start = options
            .start_from
            .map_or(std::ptr::null(), |g| g as *const Graph);
        Graph::init_with(|g| unsafe {
            igraph_barabasi_game(
                g,
                n,
                options.power,
                m,
                opt_ptr(&outseq),
                options.outpref,
                options.a,
                options.directed,
                options.algorithm.into(),
                start,
            )
        })
    }

    /// Generates a graph by preferential attachment with *aging* of vertices.
    ///
    /// Starting from one vertex, a new vertex is added in each step and
    /// connected to `m` existing ones, chosen with probability proportional
    /// to `(deg_coef * k^pa_exp + zero_deg_appeal) * (age_coef * l^aging_exp + zero_age_appeal)`,
    /// where `k` is the (in-)degree and `l` the age of the vertex; the age is
    /// incremented every `floor(n / aging_bins) + 1` steps. See
    /// [`BarabasiAgingOptions`].
    ///
    /// Time complexity: O((|V| + |V|/aging_bins) log |V| + |E|).
    ///
    /// See also [`Graph::recent_degree_aging_game`], where only recently
    /// gained edges count.
    ///
    /// Binds [`igraph_barabasi_aging_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_barabasi_aging_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for zero
    /// `aging_bins`, negative appeal or coefficient terms, or an `outseq`
    /// whose length differs from the number of vertices.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::games::BarabasiAgingOptions;
    ///
    /// rng::seed(3).unwrap();
    /// let opts = BarabasiAgingOptions { m: 2, aging_exp: -1.0, aging_bins: 10, ..Default::default() };
    /// let g = Graph::barabasi_aging_game(50, &opts).unwrap();
    /// assert_eq!(g.vcount(), 50);
    /// assert_eq!(g.ecount(), 49 * 2);
    ///
    /// // From igraph's unit tests: a very steep aging preference for young
    /// // vertices makes every vertex cite its predecessor twice.
    /// let young = BarabasiAgingOptions {
    ///     m: 2, pa_exp: 0.0, aging_exp: -10.0, aging_bins: 6,
    ///     zero_deg_appeal: 0.1, deg_coef: 0.1, ..Default::default()
    /// };
    /// let line = Graph::barabasi_aging_game(5, &young).unwrap();
    /// let mut edges = line.edge_list();
    /// edges.sort();
    /// assert_eq!(edges, [(1, 0), (1, 0), (2, 1), (2, 1), (3, 2), (3, 2), (4, 3), (4, 3)]);
    /// ```
    pub fn barabasi_aging_game(
        num_vertices: usize,
        options: &BarabasiAgingOptions<'_>,
    ) -> Result<Graph> {
        let n = int(num_vertices, "number of vertices")?;
        let m = int(options.m, "m")?;
        let bins = int(options.aging_bins, "aging_bins")?;
        let outseq = options.outseq.map(VectorInt::view);
        let o = options;
        Graph::init_with(|g| unsafe {
            igraph_barabasi_aging_game(
                g,
                n,
                m,
                opt_ptr(&outseq),
                o.outpref,
                o.pa_exp,
                o.aging_exp,
                bins,
                o.zero_deg_appeal,
                o.zero_age_appeal,
                o.deg_coef,
                o.age_coef,
                o.directed,
            )
        })
    }

    /// Generates a growing graph where the attractiveness of a vertex depends
    /// on the number of edges it gained *recently*.
    ///
    /// In each of the `num_vertices` time steps a vertex is added, citing `m`
    /// (or `outseq[i]`) existing vertices chosen with probability proportional
    /// to `k^power + zero_appeal`, where `k` counts the edges gained in the
    /// last `window` steps. See [`RecentDegreeOptions`].
    ///
    /// Time complexity: O(|V| log |V| + |E|).
    ///
    /// See also [`Graph::barabasi_game`] (the whole degree counts) and
    /// [`Graph::recent_degree_aging_game`].
    ///
    /// Binds [`igraph_recent_degree_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_recent_degree_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for a
    /// negative `zero_appeal`, or an `outseq` whose length differs from the
    /// number of vertices. A `window` longer than `num_vertices` is
    /// equivalent to `num_vertices` (no edge ever leaves it) and is clamped
    /// on the Rust side, since igraph 1.0.1 sizes its history queue from it
    /// without overflow checks.
    ///
    /// # Examples
    /// ```
    /// use igraph::{games::RecentDegreeOptions, prelude::*};
    ///
    /// // From igraph's unit tests: a strong preference for recently cited
    /// // vertices makes a star of double edges around vertex 0.
    /// let opts = RecentDegreeOptions {
    ///     power: 30.0, window: 100, m: 2, zero_appeal: 0.001, directed: true,
    ///     ..Default::default()
    /// };
    /// let g = Graph::recent_degree_game(10, &opts).unwrap();
    /// assert_eq!(g.ecount(), 18);
    /// assert!(g.edge_list().iter().all(|&(_, to)| to == 0));
    /// ```
    pub fn recent_degree_game(
        num_vertices: usize,
        options: &RecentDegreeOptions<'_>,
    ) -> Result<Graph> {
        let n = vertex_count(num_vertices)?;
        let m = int(options.m, "m")?;
        // A window at least as long as the whole process is equivalent to
        // `window = n` (no edge ever leaves it); clamping keeps igraph's
        // history capacity `1.5 * window * |E| / n + 10` in range.
        let window = int(options.window.min(num_vertices), "window")?;
        let outseq = options.outseq.map(VectorInt::view);
        let o = options;
        Graph::init_with(|g| unsafe {
            igraph_recent_degree_game(
                g,
                n,
                o.power,
                window,
                m,
                opt_ptr(&outseq),
                o.outpref,
                o.zero_appeal,
                o.directed,
            )
        })
    }

    /// Preferential attachment based on the number of edges gained recently,
    /// with aging of vertices.
    ///
    /// Like [`Graph::barabasi_aging_game`], but the degree part counts only
    /// the edges gained in the last `window` steps: the attractiveness is
    /// `(k^pa_exp + zero_appeal) * l^aging_exp`. See
    /// [`RecentDegreeAgingOptions`].
    ///
    /// Time complexity: O((|V| + |V|/aging_bins) log |V| + |E|).
    ///
    /// Binds [`igraph_recent_degree_aging_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_recent_degree_aging_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for zero
    /// `aging_bins`, a negative `zero_appeal`, or an `outseq` whose length
    /// differs from the number of vertices. As in
    /// [`Graph::recent_degree_game`], a `window` longer than `num_vertices`
    /// is clamped to it (an equivalent model).
    ///
    /// # Examples
    /// ```
    /// use igraph::{games::RecentDegreeAgingOptions, prelude::*};
    ///
    /// // From igraph's unit tests: one citation per step yields a tree.
    /// let opts = RecentDegreeAgingOptions {
    ///     outpref: true, pa_exp: 2.0, aging_exp: 2.0, aging_bins: 5, window: 4,
    ///     ..Default::default()
    /// };
    /// let g = Graph::recent_degree_aging_game(10, &opts).unwrap();
    /// assert!(g.is_tree(NeighborMode::All).unwrap());
    /// ```
    pub fn recent_degree_aging_game(
        num_vertices: usize,
        options: &RecentDegreeAgingOptions<'_>,
    ) -> Result<Graph> {
        let n = vertex_count(num_vertices)?;
        let m = int(options.m, "m")?;
        let bins = int(options.aging_bins, "aging_bins")?;
        // See `recent_degree_game`: windows longer than the process are
        // equivalent to `window = n`.
        let window = int(options.window.min(num_vertices), "window")?;
        let outseq = options.outseq.map(VectorInt::view);
        let o = options;
        Graph::init_with(|g| unsafe {
            igraph_recent_degree_aging_game(
                g,
                n,
                m,
                opt_ptr(&outseq),
                o.outpref,
                o.pa_exp,
                o.aging_exp,
                bins,
                window,
                o.zero_appeal,
                o.directed,
            )
        })
    }

    /// Generates a growing random graph.
    ///
    /// Starting with one vertex, in each step a new vertex and `m` new edges
    /// are added. The endpoints of the edges are uniformly random vertices,
    /// unless `citation` is `true`, in which case every edge goes from the
    /// newest vertex to a uniformly chosen older one. Such graphs differ from
    /// non-growing random graphs (older vertices have higher degree). The
    /// result may contain multi-edges (and self-loops without `citation`).
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// See also [`Graph::barabasi_game`] (growth with preferential instead of
    /// uniform attachment).
    ///
    /// Binds [`igraph_growing_random_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_growing_random_game).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(5).unwrap();
    /// let g = Graph::growing_random_game(10, 2, true, true).unwrap();
    /// assert_eq!(g.ecount(), 9 * 2);
    /// // Citations always point back in time.
    /// assert!(g.edge_list().iter().all(|&(from, to)| from > to));
    /// ```
    pub fn growing_random_game(
        num_vertices: usize,
        m: usize,
        directed: bool,
        citation: bool,
    ) -> Result<Graph> {
        let n = int(num_vertices, "number of vertices")?;
        let m = int(m, "m")?;
        Graph::init_with(|g| unsafe { igraph_growing_random_game(g, n, m, directed, citation) })
    }

    /// Generates a random graph with a prescribed degree sequence.
    ///
    /// `out_degrees` is the degree sequence of an undirected graph (when
    /// `in_degrees` is `None`), or the out-degree sequence of a directed one.
    /// The [`DegreeSequenceMethod`] chooses the sampler:
    ///
    /// - `Configuration`: the configuration model; may create loops and
    ///   multi-edges.
    /// - `ConfigurationSimple`: configuration model with rejection of
    ///   non-simple results; uniform over simple graphs (can be slow).
    /// - `FastHeurSimple`: simple graphs, fast but not uniform.
    /// - `EdgeSwitchingSimple`: MCMC with degree-preserving edge switches;
    ///   simple graphs.
    /// - `VigerLatapy`: undirected *connected* simple graphs, approximately
    ///   uniform.
    ///
    /// Time complexity: O(|V| + |E|) for `Configuration` and
    /// `EdgeSwitchingSimple`, unknown for the others.
    ///
    /// See also [`Graph::realize_degree_sequence`] (a deterministic
    /// realization), [`is_graphical`](crate::mixing::is_graphical) (is the
    /// sequence realizable at all?) and [`Graph::rewire`] (degree-preserving
    /// randomization of an existing graph).
    ///
    /// Binds [`igraph_degree_sequence_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_degree_sequence_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the
    /// sequences are not realizable with the chosen method (e.g. odd degree
    /// sum, mismatched in/out sums or lengths, non-graphical sequence for the
    /// simple methods).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(42).unwrap();
    /// let degrees = [3, 3, 2, 2, 2, 1, 1];
    /// let g = Graph::degree_sequence_game(&degrees, None, DegreeSequenceMethod::VigerLatapy).unwrap();
    /// assert_eq!(g.degree(.., NeighborMode::All, Loops::Twice).unwrap(), degrees);
    /// // The Viger–Latapy sampler produces connected simple graphs.
    /// assert!(g.is_simple(true).unwrap());
    /// assert!(g.is_connected(Connectedness::Weak).unwrap());
    /// ```
    pub fn degree_sequence_game(
        out_degrees: &[i64],
        in_degrees: Option<&[i64]>,
        method: DegreeSequenceMethod,
    ) -> Result<Graph> {
        let out = VectorInt::view(out_degrees);
        let inn = in_degrees.map(VectorInt::view);
        Graph::init_with(|g| unsafe {
            igraph_degree_sequence_game(g, out.as_ptr(), opt_ptr(&inn), method.into())
        })
    }

    /// Generates a random `k`-regular graph: every vertex has degree `k` (or
    /// out- and in-degree `k` in the directed case).
    ///
    /// For undirected graphs, `num_vertices * k` must be even. With
    /// `multiple = false` the result is simple. The sampling is not uniform
    /// (it relies on [`Graph::degree_sequence_game`]).
    ///
    /// Time complexity: O(|V| + |E|) if `multiple` is `true`, unknown
    /// otherwise.
    ///
    /// See also [`Graph::ring`] and [`Graph::square_lattice`] for
    /// deterministic regular graphs.
    ///
    /// Binds [`igraph_k_regular_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_k_regular_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) e.g. for
    /// an odd number of vertices and odd `k` in an undirected graph.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(42).unwrap();
    /// let g = Graph::k_regular_game(10, 3, false, false).unwrap();
    /// assert_eq!(g.degree(.., NeighborMode::All, Loops::Twice).unwrap(), vec![3; 10]);
    /// assert!(Graph::k_regular_game(9, 3, false, false).is_err());
    /// ```
    pub fn k_regular_game(
        num_vertices: usize,
        k: usize,
        directed: bool,
        multiple: bool,
    ) -> Result<Graph> {
        let n = int(num_vertices, "number of vertices")?;
        let k = int(k, "k")?;
        Graph::init_with(|g| unsafe { igraph_k_regular_game(g, n, k, directed, multiple) })
    }

    /// Generates a non-growing random graph with edge probabilities
    /// proportional to vertex *fitness* scores.
    ///
    /// The graph has `fitness_out.len()` vertices and exactly `num_edges`
    /// edges. Pairs `(i, j)` are drawn with probabilities proportional to the
    /// fitnesses (out-fitness of `i` and in-fitness of `j` for directed
    /// graphs, which are created when `fitness_in` is given) and connected
    /// unless forbidden by `allowed_edge_types`. The expected degrees are
    /// proportional to the fitnesses (exactly so when loops and multi-edges
    /// are allowed). This is the model of Goh, Kahng and Kim (2001).
    ///
    /// Time complexity: O(|V| + |E| log |E|).
    ///
    /// See also [`Graph::static_power_law_game`] (power-law fitnesses) and
    /// [`Graph::chung_lu_game`] (independent edges with prescribed expected
    /// degrees).
    ///
    /// Binds [`igraph_static_fitness_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_static_fitness_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for
    /// negative or non-finite fitnesses, mismatched lengths, requesting edges
    /// when every fitness is zero, or more edges than the non-zero fitnesses
    /// allow in a graph without multi-edges. The fitnesses are validated on
    /// the Rust side: igraph 1.0.1 accepts a negative out- or in-fitness in
    /// the directed case (as long as the other vector is non-negative) and
    /// then creates edges to non-existent vertices, and loops forever on
    /// infinite or NaN fitnesses, or on finite ones whose sum is infinite
    /// (also rejected).
    /// [`ErrorKind::Overflow`](crate::ErrorKind::Overflow) for more than
    /// `i64::MAX / 2` edges (igraph 1.0.1 would overflow computing
    /// `2 * num_edges` and abort).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(42).unwrap();
    /// // Vertex 3 has zero fitness: it stays isolated.
    /// let g = Graph::static_fitness_game(3, &[1.0, 2.0, 3.0, 0.0], None, EdgeTypeSw::Simple).unwrap();
    /// assert_eq!(g.ecount(), 3);
    /// assert_eq!(g.degree_of(3, NeighborMode::All, Loops::Twice).unwrap(), 0);
    /// ```
    pub fn static_fitness_game(
        num_edges: usize,
        fitness_out: &[f64],
        fitness_in: Option<&[f64]>,
        allowed_edge_types: impl Into<AllowedEdgeTypes>,
    ) -> Result<Graph> {
        let m = edge_count(num_edges)?;
        finite_non_negative(fitness_out, "the out-fitness scores")?;
        finite_sum(fitness_out, "the out-fitness scores")?;
        if let Some(fi) = fitness_in {
            finite_non_negative(fi, "the in-fitness scores")?;
            finite_sum(fi, "the in-fitness scores")?;
        }
        let fo = Vector::view(fitness_out);
        let fi = fitness_in.map(Vector::view);
        let mode = sw(allowed_edge_types);
        Graph::init_with(|g| unsafe {
            igraph_static_fitness_game(g, m, fo.as_ptr(), opt_ptr(&fi), mode)
        })
    }

    /// Generates a non-growing random graph with expected power-law degree
    /// distributions.
    ///
    /// Uses [`Graph::static_fitness_game`] with fitnesses `i^(-alpha)` for
    /// `i = 1, ..., n`, where `alpha = 1 / (gamma - 1)` and `gamma` is the
    /// exponent: vertex `v` (counting from zero) gets `(n - v)^(-alpha)`, so
    /// the highest-numbered vertices are the hubs (the finite size correction
    /// adds a constant offset to `i`). Pass
    /// `exponent_in = Some(gamma_in)` for a directed graph (the in-fitnesses are
    /// shuffled to avoid in/out correlations), `None` for an undirected one.
    /// Exponents must be at least 2 (`f64::INFINITY` gives an Erdős–Rényi
    /// graph). `finite_size_correction` applies the correction of Cho et al.
    /// that reduces finite size effects for exponents below 3.
    ///
    /// Time complexity: O(|V| + |E| log |E|).
    ///
    /// See also [`power_law_fit`](crate::misc::power_law_fit) to estimate the
    /// exponent back from the degrees.
    ///
    /// Binds [`igraph_static_power_law_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_static_power_law_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for
    /// exponents below 2 (or NaN), or too many edges for a simple graph;
    /// [`ErrorKind::Overflow`](crate::ErrorKind::Overflow) for more than
    /// `i64::MAX / 2` edges.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(42).unwrap();
    /// let g = Graph::static_power_law_game(1000, 2000, 2.5, None, EdgeTypeSw::Simple, true).unwrap();
    /// assert_eq!((g.vcount(), g.ecount()), (1000, 2000));
    /// assert!(Graph::static_power_law_game(10, 10, 1.5, None, EdgeTypeSw::Simple, false).is_err());
    /// ```
    pub fn static_power_law_game(
        num_vertices: usize,
        num_edges: usize,
        exponent_out: f64,
        exponent_in: Option<f64>,
        allowed_edge_types: impl Into<AllowedEdgeTypes>,
        finite_size_correction: bool,
    ) -> Result<Graph> {
        let n = int(num_vertices, "number of vertices")?;
        let m = edge_count(num_edges)?;
        if exponent_out.is_nan() {
            return Err(Error::invalid("the out-degree exponent must not be NaN"));
        }
        // igraph treats any negative (or NaN) in-exponent as "undirected":
        // reject those explicitly so that `Some(_)` always means directed.
        let exponent_in = match exponent_in {
            Some(e) if e.is_nan() || e < 2.0 => {
                return Err(Error::invalid(format!(
                    "the in-degree exponent must be at least 2, got {e}"
                )));
            }
            Some(e) => e,
            None => -1.0,
        };
        let mode = sw(allowed_edge_types);
        Graph::init_with(|g| unsafe {
            igraph_static_power_law_game(
                g,
                n,
                m,
                exponent_out,
                exponent_in,
                mode,
                finite_size_correction,
            )
        })
    }

    /// Samples a graph from the Chung–Lu model, i.e. with prescribed
    /// *expected* degrees.
    ///
    /// Each pair `i, j` is connected independently with a probability
    /// depending on `q_ij = w_i w_j / S`, where `w` are the vertex weights
    /// (out-weights `out_weights` and in-weights `in_weights` in the directed
    /// case, created when `in_weights` is given) and `S` is their sum. The
    /// [`ChungLuVariant`] selects `p_ij = min(q_ij, 1)` (`Original`, where the
    /// expected degrees equal the weights when loops are allowed),
    /// `q_ij / (1 + q_ij)` (`MaxEnt`) or `1 - exp(-q_ij)` (`Nr`). `loops`
    /// controls whether self-loops may be created. Experimental in igraph.
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// See also [`Graph::static_fitness_game`] (fixed number of edges) and
    /// [`Graph::degree_sequence_game`] (exact degrees).
    ///
    /// Binds [`igraph_chung_lu_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_chung_lu_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for
    /// negative or non-finite weights, or in/out weights with different
    /// lengths or sums. Also, on the Rust side, when the sum of the weights
    /// is infinite or the product of the largest out- and in-weight
    /// overflows: igraph 1.0.1 would compute NaN probabilities and loop
    /// forever.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(42).unwrap();
    /// let weights = vec![2.0; 500];
    /// let g = Graph::chung_lu_game(&weights, None, true, ChungLuVariant::Original).unwrap();
    /// // With loops allowed, the expected degrees are exactly the weights.
    /// assert!((g.mean_degree(true).unwrap() - 2.0).abs() < 0.3);
    /// ```
    pub fn chung_lu_game(
        out_weights: &[f64],
        in_weights: Option<&[f64]>,
        loops: bool,
        variant: ChungLuVariant,
    ) -> Result<Graph> {
        // igraph checks each weight, but neither their sum nor the products
        // `w_i w_j`: an infinite one gives NaN connection probabilities, and
        // then a NaN geometric gap converted to an integer (UB, endless loop).
        finite_non_negative(out_weights, "the out-weights")?;
        finite_sum(out_weights, "the out-weights")?;
        let max = |w: &[f64]| w.iter().copied().fold(0.0, f64::max);
        let (max_out, max_in) = match in_weights {
            Some(w) => {
                finite_non_negative(w, "the in-weights")?;
                finite_sum(w, "the in-weights")?;
                (max(out_weights), max(w))
            }
            None => (max(out_weights), max(out_weights)),
        };
        if !(max_out * max_in).is_finite() {
            return Err(Error::invalid(
                "the weights are too large: the product of the largest out- and in-weight overflows",
            ));
        }
        let wo = Vector::view(out_weights);
        let wi = in_weights.map(Vector::view);
        Graph::init_with(|g| unsafe {
            igraph_chung_lu_game(g, wo.as_ptr(), opt_ptr(&wi), loops, variant.into())
        })
    }

    /// Generates a Watts–Strogatz small-world graph.
    ///
    /// A periodic `dim`-dimensional lattice with `size` vertices along each
    /// dimension is built, every vertex is connected to its neighbors within
    /// `nei` steps, and then *both* endpoints of each edge are rewired with
    /// probability `p` (as in [`Graph::rewire_edges`]). Note that this
    /// differs from the original model, which rewires one endpoint only: for
    /// `p = 1` the result is a `G(n, m)` graph.
    ///
    /// Time complexity: O(|V| d^o + |E|), `d` average degree, `o = nei`.
    ///
    /// See also [`Graph::square_lattice`] (the unrewired lattice),
    /// [`Graph::average_path_length`] and
    /// [`Graph::transitivity_avglocal_undirected`] (the two quantities of the
    /// small-world phenomenon).
    ///
    /// Binds [`igraph_watts_strogatz_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_watts_strogatz_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `dim`
    /// or `size` is zero, or `p` is not in `[0, 1]` (NaN included).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(42).unwrap();
    /// // A ring of 100 vertices, each linked to 2 neighbors on each side.
    /// let g = Graph::watts_strogatz_game(1, 100, 2, 0.05, EdgeTypeSw::Simple).unwrap();
    /// assert_eq!((g.vcount(), g.ecount()), (100, 200));
    /// // Without rewiring, the 1-dimensional lattice with nei = 1 is a cycle.
    /// let c = Graph::watts_strogatz_game(1, 10, 1, 0.0, EdgeTypeSw::Simple).unwrap();
    /// assert!(c.isomorphic(&Graph::ring(10, false, false, true).unwrap()).unwrap());
    /// ```
    pub fn watts_strogatz_game(
        dim: usize,
        size: usize,
        nei: usize,
        p: f64,
        allowed_edge_types: impl Into<AllowedEdgeTypes>,
    ) -> Result<Graph> {
        let dim = int(dim, "dim")?;
        let size = int(size, "size")?;
        let nei = int(nei, "nei")?;
        not_nan(p, "the rewiring probability")?;
        let mode = sw(allowed_edge_types);
        Graph::init_with(|g| unsafe { igraph_watts_strogatz_game(g, dim, size, nei, p, mode) })
    }

    /// Rewires the edges of the graph in place, with constant probability.
    ///
    /// Each endpoint of each edge is moved to a uniformly random vertex with
    /// probability `prob` (in `[0, 1]`), respecting `allowed_edge_types`. The
    /// number of vertices and edges is preserved (and the directedness).
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// Unlike [`Graph::rewire`], this does *not* preserve the degrees.
    ///
    /// Binds [`igraph_rewire_edges`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_rewire_edges).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `prob`
    /// is not in `[0, 1]` (NaN included), or if `prob > 0` and the graph has a
    /// single vertex and some edges but self-loops are not allowed (there is
    /// no other vertex to move an endpoint to; igraph 1.0.1 crashes with a
    /// division by zero there, so the Rust side rejects it).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(42).unwrap();
    /// let mut g = Graph::ring(4, false, false, true).unwrap();
    /// let before = g.edge_list();
    /// g.rewire_edges(0.0, EdgeTypeSw::Simple).unwrap(); // probability 0: no change
    /// assert_eq!(g.edge_list(), before);
    /// g.rewire_edges(1.0, EdgeTypeSw::Simple).unwrap();
    /// assert_eq!(g.ecount(), 4);
    /// assert!(g.is_simple(true).unwrap());
    /// ```
    pub fn rewire_edges(
        &mut self,
        prob: f64,
        allowed_edge_types: impl Into<AllowedEdgeTypes>,
    ) -> Result<()> {
        let allowed = allowed_edge_types.into();
        check_rewirable(self, prob, allowed.loops)?;
        igraph_call!(igraph_rewire_edges(self, prob, allowed.to_raw()))
    }

    /// Rewires one chosen endpoint of the directed edges in place, with
    /// constant probability.
    ///
    /// With `mode = Out` the *target* of each edge is rewired (the out-degree
    /// sequence is preserved), with `mode = In` the *source* (the in-degree
    /// sequence is preserved); `mode = All` and undirected graphs fall back to
    /// [`Graph::rewire_edges`]. `loops` allows self-loops. The result may
    /// contain multi-edges.
    ///
    /// Time complexity: O(|E|).
    ///
    /// Binds [`igraph_rewire_directed_edges`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_rewire_directed_edges).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `prob`
    /// is not in `[0, 1]` (NaN included), or if `prob > 0`, `loops` is
    /// `false` and the graph has a single vertex and some edges (igraph 1.0.1
    /// crashes with a division by zero there, so the Rust side rejects it).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(42).unwrap();
    /// let mut g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2), (3, 0)], 4, true).unwrap();
    /// let outdeg = g.degree(.., NeighborMode::Out, Loops::Twice).unwrap();
    /// g.rewire_directed_edges(1.0, false, NeighborMode::Out).unwrap();
    /// assert_eq!(g.degree(.., NeighborMode::Out, Loops::Twice).unwrap(), outdeg);
    /// ```
    pub fn rewire_directed_edges(
        &mut self,
        prob: f64,
        loops: bool,
        mode: NeighborMode,
    ) -> Result<()> {
        check_rewirable(self, prob, loops)?;
        igraph_call!(igraph_rewire_directed_edges(self, prob, loops, mode.into()))
    }

    /// Generates a growing network with the *forest fire* model of Leskovec,
    /// Kleinberg and Faloutsos.
    ///
    /// Each new vertex cites `ambs` uniformly chosen *ambassadors*; then, for
    /// each cited vertex `v`, it "burns" (cites) a geometrically distributed
    /// number of `v`'s not-yet-cited out-neighbors (mean `p / (1 - p)`,
    /// `p = fw_prob`) and in-neighbors (backward probability
    /// `bw_factor * fw_prob`), recursively. The model reproduces heavy tailed
    /// degrees, community structure, densification and shrinking diameters.
    ///
    /// Time complexity: TODO in igraph (roughly proportional to the number of
    /// burned vertices).
    ///
    /// Binds [`igraph_forest_fire_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_forest_fire_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) unless
    /// `0 <= fw_prob < 1` and `0 <= bw_factor * fw_prob < 1` (NaN values are
    /// rejected on the Rust side, igraph would silently accept them).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // With no burning and more ambassadors than vertices, every new vertex
    /// // cites all the previous ones: a transitive tournament.
    /// let g = Graph::forest_fire_game(5, 0.0, 0.0, 100, true).unwrap();
    /// assert_eq!(g.ecount(), 10);
    /// assert!(g.is_dag().unwrap());
    /// ```
    pub fn forest_fire_game(
        num_vertices: usize,
        fw_prob: f64,
        bw_factor: f64,
        ambs: usize,
        directed: bool,
    ) -> Result<Graph> {
        let n = int(num_vertices, "number of vertices")?;
        let ambs = int(ambs, "ambs")?;
        not_nan(fw_prob, "the forward burning probability")?;
        // Also catches `bw_factor = ±inf` with `fw_prob = 0`.
        not_nan(bw_factor * fw_prob, "the backward burning probability")?;
        Graph::init_with(|g| unsafe {
            igraph_forest_fire_game(g, n, fw_prob, bw_factor, ambs, directed)
        })
    }

    /// Samples a graph from a stochastic block model (SBM).
    ///
    /// Vertices are split into consecutive blocks of sizes `block_sizes`
    /// (vertex ids follow the block order); a vertex of block `i` and one of
    /// block `j` are connected with probability `pref_matrix[(i, j)]` (the
    /// expected edge multiplicity when multi-edges are allowed). The
    /// preference matrix must be square, and symmetric for undirected graphs.
    ///
    /// Time complexity: O(|V| + |E| + k²), `k` the number of blocks.
    ///
    /// See also [`Graph::hsbm_game`] (hierarchical version),
    /// [`Graph::preference_game`] (random block assignment) and community
    /// detection, e.g. [`Graph::community_multilevel`], to recover the blocks.
    ///
    /// Binds [`igraph_sbm_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_sbm_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for a
    /// non-square or (undirected) non-symmetric matrix, entries out of range
    /// (probabilities in `[0, 1]`, or non-negative expected multiplicities
    /// with multi-edges) or NaN, or negative block sizes or a number of
    /// blocks not matching the matrix. With multi-edges, expected
    /// multiplicities that are infinite or above about 9e15 are also
    /// rejected on the Rust side (igraph 1.0.1 silently returns no edges for
    /// the former and loops forever on the latter);
    /// [`ErrorKind::Overflow`](crate::ErrorKind::Overflow) if the block sizes
    /// overflow when summed (checked on the Rust side, before igraph sums
    /// them unchecked).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(42).unwrap();
    /// // Two cliques of 3 vertices, no edges in between.
    /// let pref = Matrix::from_rows(&[[1.0, 0.0], [0.0, 1.0]]).unwrap();
    /// let g = Graph::sbm_game(&pref, &[3, 3], false, EdgeTypeSw::Simple).unwrap();
    /// assert_eq!(g.ecount(), 6);
    /// assert!(g.edge_list().iter().all(|&(a, b)| (a < 3) == (b < 3)));
    /// ```
    pub fn sbm_game(
        pref_matrix: &Matrix,
        block_sizes: &[i64],
        directed: bool,
        allowed_edge_types: impl Into<AllowedEdgeTypes>,
    ) -> Result<Graph> {
        no_nan_entries(pref_matrix, "the preference matrix")?;
        let allowed = allowed_edge_types.into();
        // With multi-edges igraph samples gaps from a geometric distribution
        // with success probability `x / (1 + x)`: for infinite `x` that is
        // NaN (silently no edges), and when it rounds to 1 (`x` above about
        // 9e15) the gaps are all zero and igraph loops forever.
        if allowed.multi
            && let Some(x) = pref_matrix
                .as_slice()
                .iter()
                .find(|&&x| !x.is_finite() || x / (1.0 + x) >= 1.0)
        {
            return Err(Error::invalid(format!(
                "expected edge multiplicities must be finite and below 9e15, got {x}"
            )));
        }
        // igraph sums the block sizes before validating them.
        if let Some(b) = block_sizes.iter().find(|&&b| b < 0) {
            return Err(Error::invalid(format!(
                "block sizes must be non-negative, got {b}"
            )));
        }
        block_sizes
            .iter()
            .try_fold(0 as igraph_int_t, |acc, &b| acc.checked_add(b))
            .ok_or_else(|| overflow("the sum of the block sizes"))?;
        let sizes = VectorInt::view(block_sizes);
        let mode = allowed.to_raw();
        Graph::init_with(|g| unsafe {
            igraph_sbm_game(g, pref_matrix, sizes.as_ptr(), directed, mode)
        })
    }

    /// Samples an undirected graph from the *hierarchical* stochastic block
    /// model.
    ///
    /// The `num_vertices` vertices are split into blocks of `block_size`
    /// vertices (`num_vertices / block_size` must be an integer). Within each
    /// block, vertices form clusters with sizes given by the fractions `rho`
    /// (summing to 1, with `rho[i] * block_size` integral), and two vertices
    /// of clusters `i` and `j` of the same block are connected with
    /// probability `c[(i, j)]` (`c` square and symmetric). Vertices of
    /// different blocks are connected with probability `p`.
    ///
    /// Binds [`igraph_hsbm_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_hsbm_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `num_vertices` or `block_size` is zero, `block_size` does not divide
    /// `num_vertices`, `rho` does not sum to one or gives non-integral
    /// cluster sizes, `c` is not a symmetric `rho.len() × rho.len()` matrix
    /// of probabilities, or `p` is not in `[0, 1]` (NaN values are rejected
    /// on the Rust side, igraph would silently accept them), or
    /// `num_vertices` is 2^53 or more (cluster sizes are computed in
    /// floating point).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // One block, clusters of 6 and 4 vertices, only inter-cluster edges:
    /// // the complete bipartite graph K(6, 4).
    /// let c = Matrix::from_rows(&[[0.0, 1.0], [1.0, 0.0]]).unwrap();
    /// let g = Graph::hsbm_game(10, 10, &[0.6, 0.4], &c, 0.0).unwrap();
    /// assert_eq!(g.ecount(), 24);
    /// ```
    pub fn hsbm_game(
        num_vertices: usize,
        block_size: usize,
        rho: &[f64],
        c: &Matrix,
        p: f64,
    ) -> Result<Graph> {
        let n = hsbm_vertex_count(num_vertices)?;
        let m = int(block_size, "block size")?;
        not_nan(p, "the inter-block probability `p`")?;
        no_nan_entries(c, "the cluster connection matrix `c`")?;
        let rho = Vector::view(rho);
        Graph::init_with(|g| unsafe { igraph_hsbm_game(g, n, m, rho.as_ptr(), c, p) })
    }

    /// Hierarchical stochastic block model, general version with blocks of
    /// different shapes.
    ///
    /// Like [`Graph::hsbm_game`], but block `b` has `block_sizes[b]` vertices,
    /// cluster fractions `rhos[b]` and cluster connection matrix `cs[b]`.
    /// Vertices of different blocks are connected with probability `p`.
    ///
    /// Binds [`igraph_hsbm_list_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_hsbm_list_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the
    /// three lists are empty or have different lengths, a block size is not
    /// positive, the sizes do not sum to `num_vertices`, a `rho`/`c` pair is
    /// invalid (as in [`Graph::hsbm_game`]), `p` is not in `[0, 1]`, or
    /// `num_vertices` or a block size is 2^53 or more;
    /// [`ErrorKind::Overflow`](crate::ErrorKind::Overflow) if the block sizes
    /// overflow when summed.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // A block of 3 vertices forming a triangle, and a block of 4 whose two
    /// // clusters of 2 are fully connected to each other: K3 + C4.
    /// let one = Matrix::from_rows(&[[1.0]]).unwrap();
    /// let cross = Matrix::from_rows(&[[0.0, 1.0], [1.0, 0.0]]).unwrap();
    /// let g = Graph::hsbm_list_game(7, &[3, 4], &[vec![1.0], vec![0.5, 0.5]], &[one, cross], 0.0)
    ///     .unwrap();
    /// assert_eq!(g.ecount(), 3 + 4);
    /// assert_eq!(g.connected_components(Connectedness::Weak).unwrap().count, 2);
    /// ```
    pub fn hsbm_list_game(
        num_vertices: usize,
        block_sizes: &[i64],
        rhos: &[Vec<f64>],
        cs: &[Matrix],
        p: f64,
    ) -> Result<Graph> {
        let n = hsbm_vertex_count(num_vertices)?;
        not_nan(p, "the inter-block probability `p`")?;
        // igraph sums the block sizes before validating them, and a block
        // larger than 2^53 vertices makes `round(rho * m)` inexact or out of
        // range.
        if let Some(b) = block_sizes
            .iter()
            .find(|&&b| !(0..MAX_EXACT_REAL).contains(&b))
        {
            return Err(Error::invalid(format!(
                "block sizes must be non-negative and below 2^53, got {b}"
            )));
        }
        block_sizes
            .iter()
            .try_fold(0 as igraph_int_t, |acc, &b| acc.checked_add(b))
            .ok_or_else(|| overflow("the sum of the block sizes"))?;
        for c in cs {
            no_nan_entries(c, "the cluster connection matrices `cs`")?;
        }
        let mlist = VectorInt::view(block_sizes);
        let rholist = VectorList::from(rhos);
        let clist: MatrixList = cs.iter().cloned().collect();
        Graph::init_with(|g| unsafe {
            igraph_hsbm_list_game(g, n, mlist.as_ptr(), &rholist, &clist, p)
        })
    }

    /// Generates a graph with vertex types and type-dependent connection
    /// preferences (a block model with random block assignment).
    ///
    /// Each of the `num_vertices` vertices gets a type drawn from `type_dist`
    /// (uniform when `None`), then each vertex pair is connected with
    /// probability `pref_matrix[(type_u, type_v)]`. The number of types is the
    /// size of the square `pref_matrix`, whose entries must be probabilities
    /// in `[0, 1]`; it must be symmetric for undirected graphs. With
    /// `fixed_sizes = true`, `type_dist` gives the *exact number* of vertices
    /// of each type (whole numbers; equal groups when `None`, the first
    /// `num_vertices % types` groups getting one extra vertex), and the
    /// vertices are assigned to the types in order. `loops` allows self-loops.
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// See also [`Graph::sbm_game`] (fixed, consecutive blocks) and
    /// [`Graph::asymmetric_preference_game`].
    ///
    /// Binds [`igraph_preference_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_preference_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for an
    /// empty or non-square `pref_matrix`, entries outside `[0, 1]`, a
    /// non-symmetric matrix for undirected graphs, a `type_dist` of the wrong
    /// length or with negative or non-finite entries, a random `type_dist`
    /// (`fixed_sizes = false`) without any positive entry, or fixed sizes
    /// that are not whole numbers summing to `num_vertices`. The finiteness,
    /// positivity and integrality checks are done on the Rust side: igraph
    /// 1.0.1 loops forever on infinite weights, aborts on all-zero weights and
    /// leaves vertex types uninitialized for fractional sizes.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(42).unwrap();
    /// // Two fixed groups of 5, connected only across groups: bipartite.
    /// let pref = Matrix::from_rows(&[[0.0, 1.0], [1.0, 0.0]]).unwrap();
    /// let r = Graph::preference_game(10, Some(&[5.0, 5.0]), true, &pref, false, false).unwrap();
    /// assert_eq!(r.graph.ecount(), 25);
    /// assert_eq!(r.types.iter().filter(|&&t| t == 0).count(), 5);
    /// assert!(r.graph.is_bipartite().unwrap());
    /// ```
    pub fn preference_game(
        num_vertices: usize,
        type_dist: Option<&[f64]>,
        fixed_sizes: bool,
        pref_matrix: &Matrix,
        directed: bool,
        loops: bool,
    ) -> Result<TypedGraph> {
        let n = int(num_vertices, "number of vertices")?;
        let types = int(pref_matrix.nrow(), "number of types")?;
        if let Some(td) = type_dist {
            if fixed_sizes {
                finite_non_negative(td, "the group sizes")?;
                if let Some(x) = td.iter().find(|x| x.fract() != 0.0) {
                    return Err(Error::invalid(format!(
                        "the group sizes must be whole numbers, got {x}"
                    )));
                }
            } else {
                distribution(td, "the vertex type distribution")?;
            }
        }
        let td = type_dist.map(Vector::view);
        let mut node_types = VectorInt::new();
        let graph = Graph::init_with(|g| unsafe {
            igraph_preference_game(
                g,
                n,
                types,
                opt_ptr(&td),
                fixed_sizes,
                pref_matrix,
                &mut node_types,
                directed,
                loops,
            )
        })?;
        Ok(TypedGraph {
            graph,
            types: node_types.into(),
        })
    }

    /// Generates a directed graph with asymmetric vertex types and connection
    /// preferences.
    ///
    /// Every vertex gets an *out-type* and an *in-type*, drawn from the joint
    /// distribution `type_dist_matrix` (independent uniform types when
    /// `None`); then each ordered pair `(u, v)` is connected with probability
    /// `pref_matrix[(out_type_u, in_type_v)]`. The numbers of out- and
    /// in-types are the numbers of rows and columns of `pref_matrix`. `loops`
    /// allows self-loops.
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// Binds [`igraph_asymmetric_preference_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_asymmetric_preference_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for an
    /// empty `pref_matrix`, entries outside `[0, 1]`, or a `type_dist_matrix`
    /// of the wrong shape, with negative or non-finite entries or without a
    /// positive one (the last two are checked on the Rust side: igraph 1.0.1
    /// loops forever on infinite weights and aborts on all-zero ones).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(42).unwrap();
    /// // Out-type 0 vertices link to every in-type 1 vertex, nothing else.
    /// let pref = Matrix::from_rows(&[[0.0, 1.0], [0.0, 0.0]]).unwrap();
    /// let r = Graph::asymmetric_preference_game(20, None, &pref, false).unwrap();
    /// for (u, v) in r.graph.edge_list() {
    ///     assert_eq!((r.out_types[u as usize], r.in_types[v as usize]), (0, 1));
    /// }
    /// ```
    pub fn asymmetric_preference_game(
        num_vertices: usize,
        type_dist_matrix: Option<&Matrix>,
        pref_matrix: &Matrix,
        loops: bool,
    ) -> Result<AsymmetricTypedGraph> {
        let n = int(num_vertices, "number of vertices")?;
        let out_types = int(pref_matrix.nrow(), "number of out-types")?;
        let in_types = int(pref_matrix.ncol(), "number of in-types")?;
        if let Some(td) = type_dist_matrix {
            distribution(td.as_slice(), "the type distribution matrix")?;
        }
        let td = type_dist_matrix.map_or(std::ptr::null(), |m| m as *const Matrix);
        let mut outv = VectorInt::new();
        let mut inv = VectorInt::new();
        let graph = Graph::init_with(|g| unsafe {
            igraph_asymmetric_preference_game(
                g,
                n,
                out_types,
                in_types,
                td,
                pref_matrix,
                &mut outv,
                &mut inv,
                loops,
            )
        })?;
        Ok(AsymmetricTypedGraph {
            graph,
            out_types: outv.into(),
            in_types: inv.into(),
        })
    }

    /// Simulates a growing network with vertex types: the model of Callaway,
    /// Hopcroft, Kleinberg, Newman and Strogatz.
    ///
    /// In each time step a vertex is added, with a type drawn from
    /// `type_dist` (uniform when `None`); then `edges_per_step` times, two
    /// uniformly random vertices are picked and connected with probability
    /// `pref_matrix[(type_u, type_v)]`. The number of types is the size of
    /// the square `pref_matrix`. The two endpoints are drawn independently
    /// (with replacement) among all vertices added so far, so the result may
    /// contain self-loops and multi-edges; no edges are attempted in the step
    /// that adds the first vertex, hence at most `(n - 1) * edges_per_step`
    /// edges are created.
    ///
    /// Time complexity: O(|V| k log |V|), `k = edges_per_step`.
    ///
    /// Binds [`igraph_callaway_traits_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_callaway_traits_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for an
    /// empty or non-square `pref_matrix`, entries outside `[0, 1]`, a
    /// non-symmetric matrix for undirected graphs, or a `type_dist` of the
    /// wrong length, with negative or non-finite entries or without positive
    /// ones (infinite weights, on which igraph 1.0.1 loops forever, are
    /// rejected on the Rust side).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(42).unwrap();
    /// // Two types that only connect among themselves.
    /// let pref = Matrix::from_rows(&[[1.0, 0.0], [0.0, 1.0]]).unwrap();
    /// let r = Graph::callaway_traits_game(50, 2, None, &pref, false).unwrap();
    /// assert!(r.graph.ecount() <= 49 * 2);
    /// assert!(r.graph.edge_list().iter().all(|&(u, v)| r.types[u as usize] == r.types[v as usize]));
    /// ```
    pub fn callaway_traits_game(
        num_vertices: usize,
        edges_per_step: usize,
        type_dist: Option<&[f64]>,
        pref_matrix: &Matrix,
        directed: bool,
    ) -> Result<TypedGraph> {
        let n = int(num_vertices, "number of vertices")?;
        let types = int(pref_matrix.nrow(), "number of types")?;
        let k = int(edges_per_step, "edges_per_step")?;
        if let Some(td) = type_dist {
            distribution(td, "the vertex type distribution")?;
        }
        let td = type_dist.map(Vector::view);
        let mut node_types = VectorInt::new();
        let graph = Graph::init_with(|g| unsafe {
            igraph_callaway_traits_game(
                g,
                n,
                types,
                k,
                opt_ptr(&td),
                pref_matrix,
                directed,
                &mut node_types,
            )
        })?;
        Ok(TypedGraph {
            graph,
            types: node_types.into(),
        })
    }

    /// Generates a graph with a simple growing model with vertex types (the
    /// *establishment* game).
    ///
    /// In each time step a vertex with a random type (from `type_dist`,
    /// uniform when `None`) is added, and it tries to connect to `k`
    /// uniformly chosen existing vertices; each connection succeeds with
    /// probability `pref_matrix[(type_new, type_old)]`. The `k` candidates
    /// are distinct, so the result is simple; the first `k` vertices have too
    /// few predecessors and make no connection attempts. Edges point from the
    /// new vertex to the older one.
    ///
    /// Time complexity: O(|V| k log |V|).
    ///
    /// Binds [`igraph_establishment_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_establishment_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for an
    /// empty or non-square `pref_matrix`, entries outside `[0, 1]`, a
    /// non-symmetric matrix for undirected graphs, or a `type_dist` of the
    /// wrong length, with negative or non-finite entries or without positive
    /// ones (infinite weights, on which igraph 1.0.1 loops forever, are
    /// rejected on the Rust side).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(42).unwrap();
    /// // A single type whose connections always succeed: after the first
    /// // `k` vertices, every vertex cites exactly `k` older ones.
    /// let always = Matrix::from_rows(&[[1.0]]).unwrap();
    /// let r = Graph::establishment_game(20, 3, None, &always, true).unwrap();
    /// assert_eq!(r.graph.ecount(), (20 - 3) * 3);
    /// assert!(r.graph.is_simple(true).unwrap() && r.graph.is_dag().unwrap());
    /// ```
    pub fn establishment_game(
        num_vertices: usize,
        k: usize,
        type_dist: Option<&[f64]>,
        pref_matrix: &Matrix,
        directed: bool,
    ) -> Result<TypedGraph> {
        let n = int(num_vertices, "number of vertices")?;
        let types = int(pref_matrix.nrow(), "number of types")?;
        let k = int(k, "k")?;
        if let Some(td) = type_dist {
            distribution(td, "the vertex type distribution")?;
        }
        let td = type_dist.map(Vector::view);
        let mut node_types = VectorInt::new();
        let graph = Graph::init_with(|g| unsafe {
            igraph_establishment_game(
                g,
                n,
                types,
                k,
                opt_ptr(&td),
                pref_matrix,
                directed,
                &mut node_types,
            )
        })?;
        Ok(TypedGraph {
            graph,
            types: node_types.into(),
        })
    }

    /// Generates a geometric random graph.
    ///
    /// `num_vertices` points are dropped uniformly in the unit square (on a
    /// torus when `torus` is `true`), and pairs *strictly* closer than
    /// `radius` are connected (so a zero, negative or NaN radius gives no
    /// edges). Vertices are numbered by increasing x coordinate. The
    /// coordinates are returned in the [`GeometricGraph`].
    ///
    /// Time complexity: less than O(|V|² + |E|).
    ///
    /// See also [`Graph::nearest_neighbor_graph`], which builds the same kind
    /// of graph (with a `cutoff` distance) from given points, and
    /// [`Graph::spatial_edge_lengths`] to obtain Euclidean edge weights.
    ///
    /// Binds [`igraph_grg_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_grg_game).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(42).unwrap();
    /// let r = Graph::grg_game(100, 0.2, false).unwrap();
    /// for (u, v) in r.graph.edge_list() {
    ///     let (u, v) = (u as usize, v as usize);
    ///     assert!((r.x[u] - r.x[v]).hypot(r.y[u] - r.y[v]) < 0.2);
    /// }
    /// ```
    pub fn grg_game(num_vertices: usize, radius: f64, torus: bool) -> Result<GeometricGraph> {
        let n = int(num_vertices, "number of vertices")?;
        let mut x = Vector::new();
        let mut y = Vector::new();
        let graph =
            Graph::init_with(|g| unsafe { igraph_grg_game(g, n, radius, torus, &mut x, &mut y) })?;
        Ok(GeometricGraph {
            graph,
            x: x.into(),
            y: y.into(),
        })
    }

    /// Simulates a citation network where the attractiveness of a vertex
    /// depends on the time since it was *last cited*.
    ///
    /// In each step a vertex is added and cites `edges_per_node` vertices.
    /// Time is binned into `agebins` bins of width `n / agebins + 1`;
    /// `preference[b]` is the attractiveness of vertices last cited `b` bins
    /// ago, and the last element (`preference[agebins]`) is that of vertices
    /// never cited, which must be positive. So `preference` has length
    /// `agebins + 1`. Multi-edges may appear when `edges_per_node > 1`.
    ///
    /// Time complexity: O(|V| a + |E| log |V|), `a = agebins`.
    ///
    /// Binds [`igraph_lastcit_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_lastcit_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `agebins` is zero, `preference` does not have `agebins + 1` entries,
    /// has negative or non-finite entries, or its last entry is not positive,
    /// or `num_vertices` exceeds igraph's maximum `i64::MAX - 1`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Only never-cited vertices are attractive: a path.
    /// let g = Graph::lastcit_game(9, 1, 1, &[0.0, 1.0], false).unwrap();
    /// let path: Vec<_> = (0..8).map(|i| (i, i + 1)).collect();
    /// let mut edges: Vec<_> = g.edge_list().into_iter().map(|(a, b)| (a.min(b), a.max(b))).collect();
    /// edges.sort();
    /// assert_eq!(edges, path);
    /// ```
    pub fn lastcit_game(
        num_vertices: usize,
        edges_per_node: usize,
        agebins: usize,
        preference: &[f64],
        directed: bool,
    ) -> Result<Graph> {
        let n = vertex_count(num_vertices)?;
        let e = int(edges_per_node, "edges_per_node")?;
        let a = int(agebins, "agebins")?;
        let pref = Vector::view(preference);
        Graph::init_with(|g| unsafe { igraph_lastcit_game(g, n, e, a, pref.as_ptr(), directed) })
    }

    /// Simulates a citation network where the attractiveness of a vertex
    /// depends on its type (category).
    ///
    /// The graph has `types.len()` vertices; vertex `v` has type `types[v]`
    /// (numbered from zero). In each step one vertex is added and cites
    /// `edges_per_step` older vertices, chosen with probability proportional
    /// to `pref[type]` (`pref` must cover all types). Multi-edges may appear.
    /// As long as no earlier vertex has a positive attractiveness, igraph lets
    /// the new vertex cite *itself*, i.e. it creates self-loops (unlike the
    /// other citation games, which then pick a uniformly random older
    /// vertex).
    ///
    /// Time complexity: O((|V| + |E|) log |V|).
    ///
    /// Binds [`igraph_cited_type_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_cited_type_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for
    /// negative types, negative or non-finite preferences, or types not
    /// covered by `pref` (non-finite preferences, which igraph 1.0.1 accepts
    /// and then loops forever or creates only self-loops, and types
    /// `>= pref.len()` are rejected on the Rust side);
    /// [`ErrorKind::Overflow`](crate::ErrorKind::Overflow) if
    /// `2 × types.len() × edges_per_step` overflows an `i64`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Only type 0 (vertex 0) is attractive: a star of double edges.
    /// let g = Graph::cited_type_game(&[0, 1, 1, 1, 1], &[1.0, 0.0], 2, true).unwrap();
    /// assert_eq!(g.ecount(), 8);
    /// assert!(g.edge_list().iter().all(|&(_, to)| to == 0));
    ///
    /// // Nobody is attractive: every vertex but the first cites itself.
    /// let g = Graph::cited_type_game(&[0, 0, 0], &[0.0], 1, true).unwrap();
    /// assert_eq!(g.edge_list(), [(1, 1), (2, 2)]);
    /// ```
    pub fn cited_type_game(
        types: &[i64],
        pref: &[f64],
        edges_per_step: usize,
        directed: bool,
    ) -> Result<Graph> {
        let n = int(types.len(), "number of vertices")?;
        let k = int(edges_per_step, "edges_per_step")?;
        finite_non_negative(pref, "the preferences")?;
        // igraph reserves `n * k` edges unchecked (a wrapped reservation lets
        // it push ~2^62 edges), and its error message for an uncovered type
        // computes `max(types) + 1`, which overflows for `i64::MAX`.
        n.checked_mul(k)
            .and_then(|x| x.checked_mul(2))
            .ok_or_else(|| overflow("the number of edges (vertices × edges_per_step × 2)"))?;
        if let Some(t) = types
            .iter()
            .find(|&&t| t < 0 || usize::try_from(t).is_ok_and(|t| t >= pref.len()))
        {
            return Err(Error::invalid(format!(
                "vertex types must be in 0..{} (the length of `pref`), got {t}",
                pref.len()
            )));
        }
        let t = VectorInt::view(types);
        let p = Vector::view(pref);
        Graph::init_with(|g| unsafe {
            igraph_cited_type_game(g, n, t.as_ptr(), p.as_ptr(), k, directed)
        })
    }

    /// Simulates a citation network where the probability of a citation
    /// depends on the types of both the citing and the cited vertex.
    ///
    /// Like [`Graph::cited_type_game`], but `pref` is a square matrix:
    /// `pref[(i, j)]` is the attractiveness of a type `j` vertex for a
    /// citing vertex of type `i`.
    ///
    /// Time complexity: O((|V| + |E|) log |V|).
    ///
    /// Binds [`igraph_citing_cited_type_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_citing_cited_type_game).
    ///
    /// The matrix must be exactly `t × t`, where `t = max(types) + 1`. While
    /// no earlier vertex is attractive for the citing type, a uniformly
    /// random older vertex is cited.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for
    /// negative types, a `pref` that is not `t × t`, or negative or
    /// non-finite preferences. Negative types and types `>= pref.ncol()` are
    /// rejected on the Rust side: igraph 1.0.1 does not check the former
    /// and reads out of bounds, and computes `max(types) + 1` unchecked.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // From igraph's unit tests: type i only cites type i - 1 (type 0
    /// // cites type 2, but nobody of type 2 exists yet): a path.
    /// let line = Matrix::from_rows(&[
    ///     [0.0, 0.0, 1.0, 0.0, 0.0],
    ///     [1.0, 0.0, 0.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0, 0.0, 0.0],
    ///     [0.0, 0.0, 1.0, 0.0, 0.0],
    ///     [0.0, 0.0, 0.0, 1.0, 0.0],
    /// ])
    /// .unwrap();
    /// let g = Graph::citing_cited_type_game(&[0, 1, 2, 3, 4], &line, 1, true).unwrap();
    /// assert_eq!(g.edge_list(), [(1, 0), (2, 1), (3, 2), (4, 3)]);
    /// ```
    pub fn citing_cited_type_game(
        types: &[i64],
        pref: &Matrix,
        edges_per_step: usize,
        directed: bool,
    ) -> Result<Graph> {
        let n = int(types.len(), "number of vertices")?;
        let k = int(edges_per_step, "edges_per_step")?;
        // Soundness: igraph indexes its per-type arrays with these values
        // without checking the lower bound.
        if let Some(t) = types.iter().find(|&&t| t < 0) {
            return Err(Error::invalid(format!(
                "vertex types must be non-negative, got {t}"
            )));
        }
        // igraph computes `max(types) + 1` unchecked (overflow for
        // `i64::MAX`); a type outside the matrix is an error anyway.
        let ncol = pref.ncol();
        if let Some(t) = types
            .iter()
            .find(|&&t| usize::try_from(t).is_ok_and(|t| t >= ncol))
        {
            return Err(Error::invalid(format!(
                "vertex type {t} is not covered by the preference matrix with {ncol} columns"
            )));
        }
        let t = VectorInt::view(types);
        Graph::init_with(|g| unsafe {
            igraph_citing_cited_type_game(g, n, t.as_ptr(), pref, k, directed)
        })
    }

    /// Generates a simple graph made of interconnected *islands*, each a
    /// `G(n, p)` random graph.
    ///
    /// There are `islands_n` islands of `islands_size` vertices each (vertex
    /// ids are consecutive within an island); every possible edge inside an
    /// island is present with probability `islands_pin`, and exactly
    /// `n_inter` edges (at most `islands_size²`) join each pair of islands.
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// See also [`Graph::sbm_game`], a more general planted partition model.
    ///
    /// Binds [`igraph_simple_interconnected_islands_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_simple_interconnected_islands_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `islands_pin` is not in `[0, 1]` or `n_inter > islands_size²` (the C
    /// documentation says larger values are clamped, but igraph 1.0.1
    /// reports an error). A NaN `islands_pin` is rejected on the Rust side:
    /// igraph 1.0.1 accepts it and then aborts the process.
    /// [`ErrorKind::Overflow`](crate::ErrorKind::Overflow) if
    /// `islands_size²`, `islands_n × islands_size` or
    /// `n_inter × islands_n × (islands_n - 1)` overflows an `i64` (igraph
    /// computes them unchecked).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(42).unwrap();
    /// // 3 complete islands of 4 vertices, 2 bridges between each pair.
    /// let g = Graph::simple_interconnected_islands_game(3, 4, 1.0, 2).unwrap();
    /// assert_eq!(g.ecount(), 3 * 6 + 3 * 2);
    /// ```
    pub fn simple_interconnected_islands_game(
        islands_n: usize,
        islands_size: usize,
        islands_pin: f64,
        n_inter: usize,
    ) -> Result<Graph> {
        let a = int(islands_n, "islands_n")?;
        let b = int(islands_size, "islands_size")?;
        let c = int(n_inter, "n_inter")?;
        not_nan(islands_pin, "the edge probability within islands")?;
        // igraph computes these products in unchecked integer arithmetic.
        b.checked_mul(b).ok_or_else(|| overflow("islands_size²"))?;
        a.checked_mul(b)
            .ok_or_else(|| overflow("the number of vertices (islands_n × islands_size)"))?;
        (a.checked_sub(1))
            .and_then(|a1| a.checked_mul(a1))
            .and_then(|pairs| c.checked_mul(pairs))
            .ok_or_else(|| overflow("the number of inter-island edges"))?;
        Graph::init_with(|g| unsafe {
            igraph_simple_interconnected_islands_game(g, a, b, islands_pin, c)
        })
    }

    /// Generates a random graph correlated with this (simple) graph.
    ///
    /// The adjacency matrix of `self` is perturbed so that the Pearson
    /// correlation between the old and new adjacency matrices is `corr`
    /// (in `[0, 1]`), for a graph of density `p` (in the open interval
    /// `(0, 1)`, typically the density of `self`). The vertices of the result
    /// are then permuted by `permutation` (`permutation[i]` is the vertex of
    /// the original graph that becomes vertex `i`), if given. The result is
    /// directed when `self` is. For `corr = 0` the result is simply a fresh
    /// `G(n, p)` graph and the permutation is ignored (only its length is
    /// checked).
    ///
    /// See also [`Graph::permute_vertices`] (same permutation convention)
    /// and [`Graph::isomorphic`]; correlated pairs are the standard benchmark
    /// of graph matching.
    ///
    /// Binds [`igraph_correlated_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_correlated_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `corr`
    /// is not in `[0, 1]`, `p` is not in `(0, 1)` (NaN included), `self` is
    /// not simple, or `permutation` does not have one entry per vertex or
    /// (when `corr > 0`) is not a permutation of `0..n`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(42).unwrap();
    /// let g = Graph::erdos_renyi_game_gnp(30, 0.3, false, EdgeTypeSw::Simple, false).unwrap();
    /// // Perfect correlation reproduces the graph...
    /// let h = g.correlated_game(1.0, 0.3, None).unwrap();
    /// assert!(g.is_same_graph(&h).unwrap());
    /// // ... up to the requested relabelling.
    /// let perm: Vec<i64> = (0..30).map(|i| (i + 1) % 30).collect();
    /// let h = g.correlated_game(1.0, 0.3, Some(&perm)).unwrap();
    /// assert!(h.is_same_graph(&g.permute_vertices(&perm).unwrap()).unwrap());
    /// ```
    pub fn correlated_game(&self, corr: f64, p: f64, permutation: Option<&[i64]>) -> Result<Graph> {
        not_nan(corr, "the correlation")?;
        not_nan(p, "the edge probability")?;
        let perm = permutation.map(VectorInt::view);
        Graph::init_with(|g| unsafe { igraph_correlated_game(g, self, corr, p, opt_ptr(&perm)) })
    }

    /// Generates a pair of correlated random graphs.
    ///
    /// The first graph is a `G(n, p)` graph (simple), the second one is
    /// obtained from it with [`Graph::correlated_game`] with correlation
    /// `corr` and optional vertex `permutation`. This is exactly what
    /// `igraph_correlated_pair_game` does (drawing the same random numbers);
    /// it is implemented here by chaining the two steps because the C
    /// function leaks the first graph when the second step fails (still the
    /// case in igraph 1.0.0 and 1.0.1), which the Rust version avoids.
    ///
    /// Binds [`igraph_correlated_pair_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_correlated_pair_game).
    ///
    /// # Errors
    /// As for [`Graph::correlated_game`]; in particular `p` must lie in the
    /// open interval `(0, 1)` (the error surfaces after the first graph has
    /// been drawn, as in C).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(42).unwrap();
    /// let (a, b) = Graph::correlated_pair_game(20, 0.0, 0.5, false, None).unwrap();
    /// assert_eq!((a.vcount(), b.vcount()), (20, 20));
    /// // From igraph's unit tests: correlation 1 gives two identical graphs.
    /// let (a, b) = Graph::correlated_pair_game(10, 1.0, 0.5, true, None).unwrap();
    /// assert!(a.is_same_graph(&b).unwrap());
    /// ```
    pub fn correlated_pair_game(
        num_vertices: usize,
        corr: f64,
        p: f64,
        directed: bool,
        permutation: Option<&[i64]>,
    ) -> Result<(Graph, Graph)> {
        let first =
            Graph::erdos_renyi_game_gnp(num_vertices, p, directed, EdgeTypeSw::Simple, false)?;
        let second = first.correlated_game(corr, p, permutation)?;
        Ok((first, second))
    }

    /// Generates a uniformly random labelled tree on `num_vertices` vertices.
    ///
    /// [`RandomTreeMethod::Prufer`] samples uniform Prüfer sequences
    /// (undirected trees only); [`RandomTreeMethod::Lerw`] runs a
    /// loop-erased random walk on the complete graph (Wilson's algorithm).
    /// Directed trees are oriented away from the root. For `num_vertices = 0`
    /// the null graph is returned.
    ///
    /// See also [`Graph::is_tree`] and, for deterministic trees,
    /// [`Graph::kary_tree`].
    ///
    /// Binds [`igraph_tree_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_tree_game).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for
    /// directed Prüfer trees with at least two vertices (the Prüfer method
    /// only supports undirected trees; for 0 or 1 vertices no method is run
    /// and the empty or single-vertex graph is returned).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// rng::seed(42).unwrap();
    /// let t = Graph::tree_game(50, false, RandomTreeMethod::Lerw).unwrap();
    /// assert!(t.is_tree(NeighborMode::All).unwrap());
    /// let d = Graph::tree_game(50, true, RandomTreeMethod::Lerw).unwrap();
    /// assert!(d.is_tree(NeighborMode::Out).unwrap()); // an out-tree
    /// ```
    pub fn tree_game(
        num_vertices: usize,
        directed: bool,
        method: RandomTreeMethod,
    ) -> Result<Graph> {
        let n = int(num_vertices, "number of vertices")?;
        Graph::init_with(|g| unsafe { igraph_tree_game(g, n, directed, method.into()) })
    }

    /// Generates a random dot product graph.
    ///
    /// Each vertex has a latent position vector, a *column* of `vecs`; two
    /// vertices are connected with probability equal to the dot product of
    /// their vectors (negative products never produce an edge, products
    /// above one always do, with a warning).
    ///
    /// Time complexity: O(n² m), `n` vertices, `m` the vector length.
    ///
    /// See also [`sample_sphere_surface`](crate::misc::sample_sphere_surface)
    /// and [`sample_dirichlet`](crate::misc::sample_dirichlet), convenient
    /// ways of drawing latent positions, and the adjacency spectral embedding
    /// ([`Graph::adjacency_spectral_embedding`]), which estimates them back.
    ///
    /// Binds [`igraph_dot_product_game`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_dot_product_game).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // Orthonormal positions for vertices 0,1 vs 2: 0-1 always, 2 alone.
    /// let vecs = Matrix::from_rows(&[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).unwrap();
    /// let g = Graph::dot_product_game(&vecs, false).unwrap();
    /// assert_eq!(g.edge_list(), vec![(0, 1)]);
    /// ```
    pub fn dot_product_game(vecs: &Matrix, directed: bool) -> Result<Graph> {
        Graph::init_with(|g| unsafe { igraph_dot_product_game(g, vecs, directed) })
    }
}

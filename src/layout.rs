//! Graph layouts: placing vertices in the plane or in 3D space (`igraph_layout.h`).
//!
//! A *layout* assigns coordinates to every vertex of a graph, typically to
//! draw it. Every layout function of this module returns a [`Matrix`] with
//! **one row per vertex** (row `i` holds the coordinates of vertex `i`) and
//! two columns (`x`, `y`) for 2D layouts, or three columns (`x`, `y`, `z`) for
//! the `*_3d` variants. Use [`Matrix::row`], [`Matrix::column`] or
//! [`Matrix::to_rows`] to read them.
//!
//! This module binds the whole of the C header `igraph_layout.h`: simple
//! geometric layouts, force-directed layouts (with tunable *options structs*
//! whose [`Default`] follows the recommendations of the igraph documentation),
//! tree and layered layouts, dimensionality reduction layouts (MDS, UMAP) and
//! a few helpers.
//!
//! ```
//! use igraph::prelude::*;
//! use igraph::layout::FruchtermanReingoldOptions;
//!
//! // A 6-cycle.
//! let g = Graph::ring(6, false, false, true).unwrap();
//!
//! // Deterministic geometric layout: all vertices on the unit circle.
//! let circle = g.layout_circle(..).unwrap();
//! assert_eq!(circle.shape(), (6, 2));
//! for p in circle.rows() {
//!     assert!((p[0].hypot(p[1]) - 1.0).abs() < 1e-12);
//! }
//!
//! // Randomized force-directed layout, made reproducible by seeding the RNG.
//! let opts = FruchtermanReingoldOptions::default();
//! rng::seed(42).unwrap();
//! let fr = g.layout_fruchterman_reingold(&opts).unwrap();
//! rng::seed(42).unwrap();
//! assert_eq!(g.layout_fruchterman_reingold(&opts).unwrap(), fr);
//! assert_eq!(fr.shape(), (6, 2));
//! ```
//!
//! # Provided functionality
//!
//! | Family | 2D | 3D |
//! |--------|----|----|
//! | Random | [`layout_random`](Graph::layout_random) | [`layout_random_3d`](Graph::layout_random_3d) |
//! | Regular shapes | [`layout_circle`](Graph::layout_circle), [`layout_star`](Graph::layout_star), [`layout_grid`](Graph::layout_grid) | [`layout_sphere`](Graph::layout_sphere), [`layout_grid_3d`](Graph::layout_grid_3d) |
//! | Fruchterman–Reingold | [`layout_fruchterman_reingold`](Graph::layout_fruchterman_reingold) | [`layout_fruchterman_reingold_3d`](Graph::layout_fruchterman_reingold_3d) |
//! | Kamada–Kawai | [`layout_kamada_kawai`](Graph::layout_kamada_kawai) | [`layout_kamada_kawai_3d`](Graph::layout_kamada_kawai_3d) |
//! | DrL | [`layout_drl`](Graph::layout_drl) | [`layout_drl_3d`](Graph::layout_drl_3d) |
//! | UMAP | [`layout_umap`](Graph::layout_umap), [`layout_umap_compute_weights`](Graph::layout_umap_compute_weights) | [`layout_umap_3d`](Graph::layout_umap_3d) |
//! | Other force-directed | [`layout_lgl`](Graph::layout_lgl), [`layout_graphopt`](Graph::layout_graphopt), [`layout_gem`](Graph::layout_gem), [`layout_davidson_harel`](Graph::layout_davidson_harel) | |
//! | Trees | [`layout_reingold_tilford`](Graph::layout_reingold_tilford), [`layout_reingold_tilford_circular`](Graph::layout_reingold_tilford_circular), [`roots_for_tree_layout`](Graph::roots_for_tree_layout) | |
//! | Layered / bipartite | [`layout_sugiyama`](Graph::layout_sugiyama), [`layout_bipartite`](Graph::layout_bipartite) | |
//! | Distance based | [`layout_mds`](Graph::layout_mds) (any dimension) | |
//! | Helpers | [`layout_align`](Graph::layout_align) (any dimension), [`layout_merge_dla`] | |
//!
//! # Starting positions and reproducibility
//!
//! Iterative layouts (Fruchterman–Reingold, Kamada–Kawai, DrL, UMAP, graphopt,
//! GEM, Davidson–Harel) can start from a given layout: set the `initial` field
//! of their options (or pass it explicitly) to refine an existing layout,
//! otherwise they start from a random (or, for Kamada–Kawai, circular)
//! configuration. Random choices (random starts, LGL's random root, GEM's
//! vertex order, DLA random walks, ...) use the calling thread's default
//! random number generator. Every thread has its own, so
//! [`rng::seed`](crate::rng::seed) makes the layouts computed afterwards in
//! that thread reproducible, whatever other threads do. To leave the
//! thread's default stream untouched, run the layout inside
//! [`Rng::scoped`](crate::rng::Rng::scoped) with a generator of your own:
//!
//! ```
//! use igraph::prelude::*;
//! use igraph::layout::GemOptions;
//!
//! let g = Graph::ring(4, false, false, true).unwrap();
//! let run = || Rng::new(RngType::Pcg64, 42).unwrap().scoped(|| g.layout_gem(&GemOptions::default()));
//! assert_eq!(run().unwrap(), run().unwrap());
//! ```
//!
//! # See also
//!
//! - Graphs to lay out: [`Graph::famous`], [`Graph::ring`],
//!   [`Graph::kary_tree`], [`Graph::square_lattice`] (constructors module).
//! - Disconnected graphs: [`Graph::decompose`] splits a graph into its
//!   components, which can be laid out separately and merged with
//!   [`layout_merge_dla`].
//! - Inputs of some layouts: [`Graph::distances`] (the default distance
//!   matrix of [`layout_mds`](Graph::layout_mds)), [`Graph::bipartite_types`]
//!   (the `types` of [`layout_bipartite`](Graph::layout_bipartite)),
//!   [`Graph::feedback_arc_set`] (how [`layout_sugiyama`](Graph::layout_sugiyama)
//!   breaks cycles), [`Graph::nearest_neighbor_graph`] (a typical input of
//!   [`layout_umap`](Graph::layout_umap)).
//! - Using a layout: [`Graph::spatial_edge_lengths`] measures the edges of a
//!   drawing, [`convex_hull_2d`](crate::misc::convex_hull_2d) its outline, and
//!   community detection (e.g. [`Graph::community_multilevel`]) gives vertex
//!   colors.

use crate::{
    constants::{LayoutGrid, NeighborMode},
    error::{Error, Result},
    ffi::*,
    graph::{Graph, VertexId},
    igraph_call,
    list::MatrixList,
    matrix::Matrix,
    selector::VertexSelector,
    vector::{Vector, VectorBool, VectorInt, View},
};
use std::{ffi::c_void, mem::MaybeUninit, ptr};

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Zero-copy view of an optional real slice.
fn opt_view(v: Option<&[f64]>) -> Option<View<'_, Vector>> {
    v.map(Vector::view)
}

/// Raw pointer to an optional view, `NULL` when absent.
fn opt_ptr<V>(v: &Option<View<'_, V>>) -> *const V {
    v.as_ref().map_or(ptr::null(), |v| v.as_ptr())
}

/// Checks that an optional per-item vector has the expected length.
fn check_len<T>(name: &str, v: Option<&[T]>, expected: usize) -> Result<()> {
    match v {
        Some(v) if v.len() != expected => Err(Error::invalid(format!(
            "`{name}` has length {}, expected {expected}",
            v.len()
        ))),
        _ => Ok(()),
    }
}

/// Builds the result matrix, copying the starting positions if given.
///
/// Returns `(matrix, use_seed)`.
fn start_matrix(initial: Option<&Matrix>, n: usize, dim: usize) -> Result<(Matrix, bool)> {
    match initial {
        Some(m) if m.shape() != (n, dim) => Err(Error::invalid(format!(
            "the initial layout is {}x{}, expected {n}x{dim}",
            m.nrow(),
            m.ncol()
        ))),
        Some(m) if !m.as_slice().iter().all(|x| x.is_finite()) => Err(Error::invalid(
            "the initial layout contains NaN or infinite coordinates",
        )),
        Some(m) => Ok((m.clone(), true)),
        None => Ok((Matrix::new(), false)),
    }
}

/// Checks that an optional real parameter is finite and positive.
fn check_positive(name: &str, x: f64) -> Result<()> {
    if x.is_finite() && x > 0.0 {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "`{name}` must be finite and positive, got {x}"
        )))
    }
}

/// Largest number of grid cells [`Graph::layout_lgl`] accepts regardless of
/// the graph size (the limit is `max(LGL_MAX_CELLS, 4 |V|)`; the default
/// parameters give about `1.3 |V|` cells).
const LGL_MAX_CELLS: i64 = 1 << 28;

/// Largest vertex count (graph plus the extra `rootlevel` vertices) accepted
/// by the Reingold–Tilford layouts.
const RT_MAX_VERTICES: igraph_int_t = 1 << 40;

/// Invalidates the property cache of a graph when dropped.
struct CacheReset<'a>(Option<&'a Graph>);

impl Drop for CacheReset<'_> {
    fn drop(&mut self) {
        if let Some(g) = self.0 {
            g.invalidate_cache();
        }
    }
}

/// Largest edge weight accepted by [`Graph::layout_drl`]. DrL computes its
/// energies in single precision: in igraph 1.0.1 weights around `1e37`
/// already overflow them into NaN positions (converted to integers by its
/// density grid, which is undefined behavior); `1e20` leaves a wide margin.
const DRL_MAX_WEIGHT: f64 = 1e20;

/// Converts a count to `igraph_int_t`, saturating at `igraph_int_t::MAX`
/// (a plain `as` cast would turn huge counts into negative values, which
/// igraph interprets differently, e.g. as "automatic").
fn int(x: usize) -> igraph_int_t {
    igraph_int_t::try_from(x).unwrap_or(igraph_int_t::MAX)
}

/// Vertex count as an `f64`, at least one (for defaults that must be positive).
fn nf(g: &Graph) -> f64 {
    g.vcount().max(1) as f64
}

/// Checks that all the ids are valid vertex ids.
fn check_vertices(name: &str, ids: &[VertexId], n: usize) -> Result<()> {
    match ids.iter().find(|&&v| v < 0 || v as usize >= n) {
        Some(v) => Err(Error::new(
            crate::error::ErrorKind::InvalidVertexId,
            format!("`{name}` contains {v}, which is not a vertex id (the graph has {n} vertices)"),
        )),
        None => Ok(()),
    }
}

/// Coordinate bounds of 2D/3D force-directed layouts, as raw views.
struct Bounds<'a> {
    views: [Option<View<'a, Vector>>; 6],
}

impl<'a> Bounds<'a> {
    fn new(g: &Graph, bounds: [Option<&'a [f64]>; 6]) -> Result<Self> {
        const NAMES: [&str; 6] = ["minx", "maxx", "miny", "maxy", "minz", "maxz"];
        let n = g.vcount();
        for (i, (name, b)) in NAMES.iter().zip(bounds.iter()).enumerate() {
            check_len(name, *b, n)?;
            // A lower bound of `-inf` or an upper bound of `+inf` means "no
            // bound" and is fine. igraph requires the caller to rule out NaN,
            // and its random initial placement fails on a lower bound of
            // `+inf` or an upper bound of `-inf` with an error that the
            // force-directed layouts ignore, going on with a wrongly sized
            // matrix (heap corruption).
            let forbidden = if i % 2 == 0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
            if b.is_some_and(|b| b.iter().any(|&x| x.is_nan() || x == forbidden)) {
                return Err(Error::invalid(format!(
                    "`{name}` contains NaN or {forbidden}"
                )));
            }
        }
        Ok(Self {
            views: bounds.map(opt_view),
        })
    }

    fn ptr(&self, i: usize) -> *const Vector {
        opt_ptr(&self.views[i])
    }
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

crate::ffi_enum! {
    /// Heuristic used by [`Graph::roots_for_tree_layout`] to choose among
    /// several possible roots (`igraph_root_choice_t`).
    pub enum RootChoice: igraph_root_choice_t {
        /// Prefer the vertices with the highest degree (out- or in-degree in
        /// directed mode). Fast even on large graphs.
        Degree = igraph_root_choice_t_IGRAPH_ROOT_CHOICE_DEGREE,
        /// Prefer the vertices with the lowest eccentricity: usually gives
        /// "wide and shallow" trees, but takes quadratic time.
        Eccentricity = igraph_root_choice_t_IGRAPH_ROOT_CHOICE_ECCENTRICITY,
    }
}

crate::ffi_enum! {
    /// Predefined parameter templates for the DrL layout
    /// (`igraph_layout_drl_default_t`), see [`DrlOptions::from_template`].
    pub enum DrlTemplate: igraph_layout_drl_default_t {
        /// The default parameters.
        Default = igraph_layout_drl_default_t_IGRAPH_LAYOUT_DRL_DEFAULT,
        /// Slightly modified parameters giving a coarser layout.
        Coarsen = igraph_layout_drl_default_t_IGRAPH_LAYOUT_DRL_COARSEN,
        /// An even coarser layout.
        Coarsest = igraph_layout_drl_default_t_IGRAPH_LAYOUT_DRL_COARSEST,
        /// Refine an already computed layout.
        Refine = igraph_layout_drl_default_t_IGRAPH_LAYOUT_DRL_REFINE,
        /// Finalize an already refined layout.
        Final = igraph_layout_drl_default_t_IGRAPH_LAYOUT_DRL_FINAL,
    }
}

// ---------------------------------------------------------------------------
// Options structs
// ---------------------------------------------------------------------------

/// Parameters of the Fruchterman–Reingold layouts
/// ([`Graph::layout_fruchterman_reingold`] and
/// [`Graph::layout_fruchterman_reingold_3d`]).
///
/// The [`Default`] follows the igraph documentation: 500 iterations, a start
/// temperature of `sqrt(n) / 10`, automatic grid selection, unit weights, a
/// random start and no coordinate bounds.
///
/// ```
/// use igraph::layout::FruchtermanReingoldOptions;
/// let weights = [1.0, 2.0, 3.0];
/// let opts = FruchtermanReingoldOptions { niter: 1000, ..Default::default() }
///     .with_weights(&weights);
/// assert_eq!(opts.niter, 1000);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FruchtermanReingoldOptions<'a> {
    /// Number of iterations (default 500).
    pub niter: usize,
    /// Start temperature: the maximum movement of a vertex along one axis in
    /// one step; it decreases linearly to zero. `None` means `sqrt(n) / 10`.
    pub start_temp: Option<f64>,
    /// Whether to use the faster, less accurate grid-based variant (2D only;
    /// [`LayoutGrid::AutoGrid`], the default, uses it above 1000 vertices).
    pub grid: LayoutGrid,
    /// Positive edge weights multiplying the attraction along edges (higher
    /// weight: closer vertices; an isolated edge of weight `w` has length
    /// `w^(-1/3)`). `None` means unit weights.
    pub weights: Option<&'a [f64]>,
    /// Starting positions (`n` × 2, or `n` × 3 for the 3D version); `None`
    /// starts from a random layout.
    pub initial: Option<&'a Matrix>,
    /// Per-vertex minimum `x` coordinate (`-inf` means no bound).
    pub minx: Option<&'a [f64]>,
    /// Per-vertex maximum `x` coordinate (`+inf` means no bound).
    pub maxx: Option<&'a [f64]>,
    /// Per-vertex minimum `y` coordinate (`-inf` means no bound).
    pub miny: Option<&'a [f64]>,
    /// Per-vertex maximum `y` coordinate (`+inf` means no bound).
    pub maxy: Option<&'a [f64]>,
    /// Per-vertex minimum `z` coordinate (3D only).
    pub minz: Option<&'a [f64]>,
    /// Per-vertex maximum `z` coordinate (3D only).
    pub maxz: Option<&'a [f64]>,
}

impl Default for FruchtermanReingoldOptions<'_> {
    fn default() -> Self {
        Self {
            niter: 500,
            start_temp: None,
            grid: LayoutGrid::AutoGrid,
            weights: None,
            initial: None,
            minx: None,
            maxx: None,
            miny: None,
            maxy: None,
            minz: None,
            maxz: None,
        }
    }
}

impl<'a> FruchtermanReingoldOptions<'a> {
    /// Sets the edge weights.
    pub fn with_weights(mut self, weights: &'a [f64]) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Sets the starting positions.
    pub fn with_initial(mut self, initial: &'a Matrix) -> Self {
        self.initial = Some(initial);
        self
    }

    /// Sets the number of iterations.
    pub fn with_niter(mut self, niter: usize) -> Self {
        self.niter = niter;
        self
    }
}

/// Parameters of the Kamada–Kawai layouts ([`Graph::layout_kamada_kawai`] and
/// [`Graph::layout_kamada_kawai_3d`]).
///
/// The [`Default`] uses `50 n` iterations (the C documentation asks for at
/// least `10 n`; `50 n` is the default of the R and Python interfaces),
/// `epsilon = 0` (always run all the iterations), `kkconst = n` (as the C
/// documentation recommends), unit edge lengths, a circular (spherical in
/// 3D) start and no bounds.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct KamadaKawaiOptions<'a> {
    /// Maximum number of iterations; `None` means `50 n`.
    pub maxiter: Option<usize>,
    /// Stop when the maximum energy change falls below this value (`0.0`,
    /// the default, performs all `maxiter` iterations).
    pub epsilon: f64,
    /// The vertex attraction constant; `None` means the number of vertices.
    pub kkconst: Option<f64>,
    /// Positive edge *lengths* used in the shortest path computation
    /// (higher weight: farther vertices). `None` means unit lengths.
    pub weights: Option<&'a [f64]>,
    /// Starting positions (`n` × 2, or `n` × 3 for the 3D version). `None`
    /// starts from a circle (sphere) of radius `0.36 sqrt(n)`, or from a
    /// random layout when some bounds are given.
    pub initial: Option<&'a Matrix>,
    /// Per-vertex minimum `x` coordinate (`-inf` means no bound).
    pub minx: Option<&'a [f64]>,
    /// Per-vertex maximum `x` coordinate (`+inf` means no bound).
    pub maxx: Option<&'a [f64]>,
    /// Per-vertex minimum `y` coordinate (`-inf` means no bound).
    pub miny: Option<&'a [f64]>,
    /// Per-vertex maximum `y` coordinate (`+inf` means no bound).
    pub maxy: Option<&'a [f64]>,
    /// Per-vertex minimum `z` coordinate (3D only).
    pub minz: Option<&'a [f64]>,
    /// Per-vertex maximum `z` coordinate (3D only).
    pub maxz: Option<&'a [f64]>,
}

impl<'a> KamadaKawaiOptions<'a> {
    /// Sets the edge lengths.
    pub fn with_weights(mut self, weights: &'a [f64]) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Sets the starting positions.
    pub fn with_initial(mut self, initial: &'a Matrix) -> Self {
        self.initial = Some(initial);
        self
    }
}

/// Parameters of the Large Graph Layout ([`Graph::layout_lgl`]).
///
/// The [`Default`] follows the igraph documentation (`n` is the number of
/// vertices): 150 iterations, `maxdelta = n`, `area = n²`, `coolexp = 1.5`,
/// `repulserad = area · n`, `cellsize = area^(1/4)` and a random root.
#[derive(Debug, Clone, PartialEq)]
pub struct LglOptions {
    /// Maximum number of cooling iterations per layout step (default 150).
    pub maxiter: usize,
    /// Maximum movement of a vertex in one iteration; `None` means `n`.
    pub maxdelta: Option<f64>,
    /// Area of the square the vertices are placed on; `None` means `n²`.
    pub area: Option<f64>,
    /// The cooling exponent (default 1.5).
    pub coolexp: f64,
    /// Radius at which repulsion cancels attraction; `None` means `area · n`.
    pub repulserad: Option<f64>,
    /// Side of the grid cells; `None` means the fourth root of `area`.
    pub cellsize: Option<f64>,
    /// Root vertex, placed first; `None` picks a random vertex.
    pub root: Option<VertexId>,
}

impl Default for LglOptions {
    fn default() -> Self {
        Self {
            maxiter: 150,
            maxdelta: None,
            area: None,
            coolexp: 1.5,
            repulserad: None,
            cellsize: None,
            root: None,
        }
    }
}

/// Parameters of the Sugiyama layered layout ([`Graph::layout_sugiyama`]).
///
/// The [`Default`] uses automatic layering, unit gaps, 100 crossing
/// minimization iterations and unit weights.
#[derive(Debug, Clone, PartialEq)]
pub struct SugiyamaOptions<'a> {
    /// Non-negative layer index of every vertex. Empty layers are skipped
    /// when minimizing crossings, but keep their room on the `y` axis.
    /// `None` lets igraph compute a layering, after breaking cycles.
    pub layers: Option<&'a [i64]>,
    /// Preferred minimum horizontal gap between vertices of a layer (default 1).
    pub hgap: f64,
    /// Distance between consecutive layers (default 1).
    pub vgap: f64,
    /// Maximum number of iterations of the crossing minimization (default 100).
    pub maxiter: usize,
    /// Edge weights, used only to break cycles: igraph tends to reverse
    /// edges with smaller weights.
    pub weights: Option<&'a [f64]>,
}

impl Default for SugiyamaOptions<'_> {
    fn default() -> Self {
        Self {
            layers: None,
            hgap: 1.0,
            vgap: 1.0,
            maxiter: 100,
            weights: None,
        }
    }
}

impl<'a> SugiyamaOptions<'a> {
    /// Sets the layer of each vertex.
    pub fn with_layers(mut self, layers: &'a [i64]) -> Self {
        self.layers = Some(layers);
        self
    }
}

/// Result of [`Graph::layout_sugiyama`].
#[derive(Debug, Clone, PartialEq)]
pub struct SugiyamaLayout {
    /// The vertex coordinates, `n` × 2; the `y` coordinate of a vertex is
    /// its layer index times `vgap`.
    pub coords: Matrix,
    /// One matrix per edge with the extra *control points* (one per row)
    /// the edge must pass through, from its source to its target; edges
    /// spanning adjacent layers have a 0 × 2 matrix.
    pub routing: Vec<Matrix>,
}

/// Parameters of the graphopt layout ([`Graph::layout_graphopt`]).
///
/// The [`Default`] uses the original graphopt defaults: 500 iterations,
/// node charge 0.001, node mass 30, spring length 0, spring constant 1,
/// maximum movement 5, and a random start.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphoptOptions<'a> {
    /// Number of iterations (default 500).
    pub niter: usize,
    /// Charge of the vertices, for the electric repulsion (default 0.001).
    /// With zero charge each iteration is only `O(|E|)`.
    pub node_charge: f64,
    /// Mass of the vertices, for the spring forces (default 30).
    pub node_mass: f64,
    /// Rest length of the springs (default 0).
    pub spring_length: f64,
    /// Spring constant (default 1).
    pub spring_constant: f64,
    /// Maximum movement along an axis in a single step (default 5).
    pub max_sa_movement: f64,
    /// Starting positions (`n` × 2); `None` starts from a random layout.
    pub initial: Option<&'a Matrix>,
}

impl Default for GraphoptOptions<'_> {
    fn default() -> Self {
        Self {
            niter: 500,
            node_charge: 0.001,
            node_mass: 30.0,
            spring_length: 0.0,
            spring_constant: 1.0,
            max_sa_movement: 5.0,
            initial: None,
        }
    }
}

/// Parameters of the UMAP layouts ([`Graph::layout_umap`] and
/// [`Graph::layout_umap_3d`]).
///
/// The [`Default`] uses `min_dist = 0.01`, 500 epochs, unit distances and a
/// random start.
#[derive(Debug, Clone, PartialEq)]
pub struct UmapOptions<'a> {
    /// Distance (or, with `distances_are_weights`, weight) of every edge;
    /// `None` means all edges have the same distance.
    pub distances: Option<&'a [f64]>,
    /// How close two unconnected vertices may get before repelling each
    /// other; non-negative, typically in `[0, 1]` (default 0.01).
    pub min_dist: f64,
    /// Number of epochs of stochastic gradient descent, typically in
    /// `[30, 500]` (default 500).
    pub epochs: usize,
    /// Whether `distances` already holds UMAP weights (e.g. from
    /// [`Graph::layout_umap_compute_weights`]).
    pub distances_are_weights: bool,
    /// Starting positions (`n` × 2, or `n` × 3 for the 3D version); `None`
    /// starts from a random layout.
    pub initial: Option<&'a Matrix>,
}

impl Default for UmapOptions<'_> {
    fn default() -> Self {
        Self {
            distances: None,
            min_dist: 0.01,
            epochs: 500,
            distances_are_weights: false,
            initial: None,
        }
    }
}

/// Parameters of the GEM layout ([`Graph::layout_gem`]).
///
/// The [`Default`] follows the igraph documentation (`n` is the number of
/// vertices): `40 n²` iterations, `temp_max = n`, `temp_min = 0.1`,
/// `temp_init = sqrt(n)` and a random start.
#[derive(Debug, Clone, PartialEq)]
pub struct GemOptions<'a> {
    /// Maximum number of iterations (a single vertex update counts as one);
    /// `None` means `40 n²`.
    pub maxiter: Option<usize>,
    /// Maximum local temperature; `None` means `n`.
    pub temp_max: Option<f64>,
    /// Global temperature at which the algorithm stops (default 0.1).
    pub temp_min: f64,
    /// Initial local temperature; `None` means `sqrt(n)`.
    pub temp_init: Option<f64>,
    /// Starting positions (`n` × 2); `None` starts from a random layout.
    pub initial: Option<&'a Matrix>,
}

impl Default for GemOptions<'_> {
    fn default() -> Self {
        Self {
            maxiter: None,
            temp_max: None,
            temp_min: 0.1,
            temp_init: None,
            initial: None,
        }
    }
}

/// Parameters of the Davidson–Harel layout ([`Graph::layout_davidson_harel`]).
///
/// The [`Default`] follows the igraph documentation, where `d` is the density
/// of the graph (edge directions ignored): 10 annealing iterations,
/// `max(10, log2 n)` fine tuning iterations, cooling factor 0.75, node
/// distance weight 1, border weight 0, edge length weight `d / 10`, edge
/// crossing weight `1 - sqrt(d)`, node-edge distance weight `(1 - d) / 5`.
#[derive(Debug, Clone, PartialEq)]
pub struct DavidsonHarelOptions<'a> {
    /// Number of annealing iterations (default 10).
    pub maxiter: usize,
    /// Number of fine tuning iterations; `None` means `max(10, log2 n)`.
    pub fineiter: Option<usize>,
    /// Cooling factor, in `(0, 1)` (default 0.75).
    pub cool_fact: f64,
    /// Weight of the node-node distance energy term (default 1).
    pub weight_node_dist: f64,
    /// Weight of the distance-from-border term (default 0: vertices may sit
    /// on the border).
    pub weight_border: f64,
    /// Weight of the edge length term; `None` means density / 10.
    pub weight_edge_lengths: Option<f64>,
    /// Weight of the edge crossing term; `None` means `1 - sqrt(density)`.
    pub weight_edge_crossings: Option<f64>,
    /// Weight of the node-edge distance term; `None` means `(1 - density) / 5`.
    pub weight_node_edge_dist: Option<f64>,
    /// Starting positions (`n` × 2); `None` starts from a random layout.
    pub initial: Option<&'a Matrix>,
}

impl Default for DavidsonHarelOptions<'_> {
    fn default() -> Self {
        Self {
            maxiter: 10,
            fineiter: None,
            cool_fact: 0.75,
            weight_node_dist: 1.0,
            weight_border: 0.0,
            weight_edge_lengths: None,
            weight_edge_crossings: None,
            weight_node_edge_dist: None,
            initial: None,
        }
    }
}

/// Parameters of the DrL layout (`igraph_layout_drl_options_t`), used by
/// [`Graph::layout_drl`] and [`Graph::layout_drl_3d`].
///
/// This is the C struct itself: start from a [`DrlTemplate`] with
/// [`DrlOptions::from_template`] (or [`Default`], which is
/// [`DrlTemplate::Default`]) and adjust the public fields. The algorithm runs
/// through six phases (*init*, *liquid*, *expansion*, *cooldown*, *crunch*,
/// *simmer*), each with its number of iterations, start temperature,
/// attraction and (non-negative) damping multiplier; `edge_cut` in `[0, 1]`
/// controls how aggressively stressed edges are cut in the late phases
/// (default 32/40).
///
/// ```
/// use igraph::layout::{DrlOptions, DrlTemplate};
/// let mut opts = DrlOptions::from_template(DrlTemplate::Coarsest);
/// assert_eq!(opts.crunch_iterations, 200);
/// opts.edge_cut = 0.0; // no edge cutting
/// assert_eq!(DrlOptions::default().liquid_iterations, 200);
/// ```
pub type DrlOptions = igraph_layout_drl_options_t;

impl igraph_layout_drl_options_t {
    /// Parameters initialized from a predefined template
    /// ([`igraph_layout_drl_options_init`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_drl_options_init)).
    pub fn from_template(template: DrlTemplate) -> Self {
        let mut raw = MaybeUninit::<Self>::zeroed();
        // Cannot fail: every template is valid.
        igraph_call!(igraph_layout_drl_options_init(
            raw.as_mut_ptr(),
            template.into()
        ))
        .expect("igraph_layout_drl_options_init failed on a valid template");
        unsafe { raw.assume_init() }
    }
}

impl Default for igraph_layout_drl_options_t {
    fn default() -> Self {
        Self::from_template(DrlTemplate::Default)
    }
}

// A plain-old-data struct of numbers: copying it is trivially safe.
impl Copy for igraph_layout_drl_options_t {}

impl Clone for igraph_layout_drl_options_t {
    fn clone(&self) -> Self {
        *self
    }
}

// ---------------------------------------------------------------------------
// Layouts
// ---------------------------------------------------------------------------

impl igraph_t {
    /// Places the vertices uniformly at random in the square `[-1, 1]²`.
    ///
    /// Uses the calling thread's default random number generator (see
    /// [`rng::seed`](crate::rng::seed)). Time complexity: `O(|V|)`.
    /// Binds [`igraph_layout_random`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_random).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::new(10, false);
    /// rng::seed(1).unwrap();
    /// let l = g.layout_random().unwrap();
    /// assert_eq!(l.shape(), (10, 2));
    /// assert!(l.as_slice().iter().all(|c| (-1.0..=1.0).contains(c)));
    /// rng::seed(1).unwrap();
    /// assert_eq!(g.layout_random().unwrap(), l);
    /// ```
    pub fn layout_random(&self) -> Result<Matrix> {
        let mut res = Matrix::new();
        igraph_call!(igraph_layout_random(self, &mut res))?;
        Ok(res)
    }

    /// Places the vertices uniformly at random in the cube `[-1, 1]³`.
    ///
    /// The 3D version of [`layout_random`](Graph::layout_random). Time
    /// complexity: `O(|V|)`.
    /// Binds [`igraph_layout_random_3d`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_random_3d).
    pub fn layout_random_3d(&self) -> Result<Matrix> {
        let mut res = Matrix::new();
        igraph_call!(igraph_layout_random_3d(self, &mut res))?;
        Ok(res)
    }

    /// Places the vertices uniformly on the unit circle, in the given order.
    ///
    /// The `k`-th vertex of `order` (out of `m` selected vertices) is placed
    /// at angle `2πk/m`; vertices not in `order` are placed at the origin.
    /// Pass `..` to place all vertices in increasing id order. This is the
    /// natural drawing of [`Graph::ring`]. Time complexity: `O(|V|)`.
    /// Binds [`igraph_layout_circle`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_circle).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) if
    /// `order` contains invalid vertices.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::new(4, false);
    /// let l = g.layout_circle(..).unwrap();
    /// assert!((l[(1, 0)] - 0.0).abs() < 1e-12 && (l[(1, 1)] - 1.0).abs() < 1e-12);
    /// // Only two vertices on the circle, the others at the origin.
    /// let l = g.layout_circle(&[3, 1]).unwrap();
    /// assert_eq!(l.row(3), vec![1.0, 0.0]);
    /// assert_eq!(l.row(0), vec![0.0, 0.0]);
    /// ```
    pub fn layout_circle<'a>(&self, order: impl Into<VertexSelector<'a>>) -> Result<Matrix> {
        let vs = order.into().to_raw()?;
        let mut res = Matrix::new();
        igraph_call!(igraph_layout_circle(self, &mut res, vs.get()))?;
        Ok(res)
    }

    /// Star-like layout: `center` at the origin, the other vertices evenly
    /// spaced on the unit circle.
    ///
    /// The edges are ignored. The non-center vertices are placed in the
    /// order given by `order`, which must be a permutation of *all* the
    /// vertices (including the center), or in increasing id order if `None`;
    /// the first one is at angle zero. `center` is ignored for the null
    /// graph. Time complexity: `O(|V|)`. This is the natural drawing of
    /// [`Graph::star`].
    /// Binds [`igraph_layout_star`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_star).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `center` is not a vertex, or `order` is not a permutation of the
    /// vertices. (igraph itself only checks the length and the range of
    /// `order`: with a repeated vertex it would leave the rows of the missing
    /// vertices uninitialized, so this wrapper rejects such orders.)
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::star(5, StarMode::Undirected, 2).unwrap();
    /// let l = g.layout_star(2, None).unwrap();
    /// assert_eq!(l.row(2), vec![0.0, 0.0]);
    /// assert_eq!(l.row(0), vec![1.0, 0.0]);
    /// ```
    pub fn layout_star(&self, center: VertexId, order: Option<&[VertexId]>) -> Result<Matrix> {
        let n = self.vcount();
        check_len("order", order, n)?;
        if let Some(order) = order {
            // igraph does not initialize the rows of vertices missing from
            // `order`: insist on a permutation so no garbage is ever read.
            let mut seen = vec![false; n];
            for &v in order {
                match usize::try_from(v).ok().and_then(|i| seen.get_mut(i)) {
                    Some(s) if !*s => *s = true,
                    _ => {
                        return Err(Error::invalid(format!(
                            "`order` must be a permutation of the {n} vertices, \
                             but {v} is out of range or repeated"
                        )));
                    }
                }
            }
        }
        let order = order.map(VectorInt::view);
        let mut res = Matrix::new();
        igraph_call!(igraph_layout_star(self, &mut res, center, opt_ptr(&order)))?;
        Ok(res)
    }

    /// Places the vertices on a regular 2D grid, row by row.
    ///
    /// Vertex `i` gets coordinates `(i mod w, i div w)`, where the width `w`
    /// is `width`, or `ceil(sqrt(n))` if `None` (`Some(0)` is the same as
    /// `None`). With `width` equal to the first dimension, this draws the
    /// vertices of a [`Graph::square_lattice`] at their lattice points. Time
    /// complexity: `O(|V|)`.
    /// Binds [`igraph_layout_grid`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_grid).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::new(5, false);
    /// let l = g.layout_grid(Some(2)).unwrap();
    /// assert_eq!(l.to_rows(), vec![
    ///     vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![0.0, 2.0],
    /// ]);
    /// ```
    pub fn layout_grid(&self, width: Option<usize>) -> Result<Matrix> {
        let mut res = Matrix::new();
        let width = width.map_or(0, int);
        igraph_call!(igraph_layout_grid(self, &mut res, width))?;
        Ok(res)
    }

    /// Places the vertices on a regular 3D grid, filling rows (along `x`),
    /// then layers (along `y`), then stacking layers along `z`.
    ///
    /// `width` is the number of vertices in a row and `height` the number of
    /// rows in a layer; if both are `None` they are `ceil(cbrt(n))`, if one is
    /// `None` it is chosen so that layers are roughly square (`Some(0)` is
    /// the same as `None`). Time complexity: `O(|V|)`.
    /// Binds [`igraph_layout_grid_3d`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_grid_3d).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // 8 vertices: a 2 x 2 x 2 cube.
    /// let l = Graph::new(8, false).layout_grid_3d(None, None).unwrap();
    /// assert_eq!(l.row(0), vec![0.0, 0.0, 0.0]);
    /// assert_eq!(l.row(3), vec![1.0, 1.0, 0.0]);
    /// assert_eq!(l.row(7), vec![1.0, 1.0, 1.0]);
    /// ```
    pub fn layout_grid_3d(&self, width: Option<usize>, height: Option<usize>) -> Result<Matrix> {
        let mut res = Matrix::new();
        let width = width.map_or(0, int);
        let height = height.map_or(0, int);
        igraph_call!(igraph_layout_grid_3d(self, &mut res, width, height))?;
        Ok(res)
    }

    /// Places the vertices (more or less) uniformly on the unit sphere.
    ///
    /// Vertices are placed along a spiral wrapped around the sphere, in
    /// increasing id order, so consecutive ids end up close to each other
    /// (Saff & Kuijlaars, 1997). Time complexity: `O(|V|)`.
    /// Binds [`igraph_layout_sphere`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_sphere).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let l = Graph::new(20, false).layout_sphere().unwrap();
    /// for p in l.rows() {
    ///     let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    ///     assert!((r - 1.0).abs() < 1e-9);
    /// }
    /// ```
    pub fn layout_sphere(&self) -> Result<Matrix> {
        let mut res = Matrix::new();
        igraph_call!(igraph_layout_sphere(self, &mut res))?;
        Ok(res)
    }

    /// Force-directed layout in the plane with the Fruchterman–Reingold algorithm.
    ///
    /// It simulates an attractive force `f_a(d) = -w d²` between connected
    /// vertices (`w` is the edge weight) and a repulsive force `f_r(d) = 1/d`
    /// between all pairs, so the equilibrium length of an isolated edge is
    /// `w^(-1/3)` (the C documentation of igraph 1.0.0 and 1.0.1 says `1/w^3`,
    /// a typo). In
    /// disconnected graphs a weak attraction of weight `n^(-3/2)` between all
    /// pairs keeps the components close. The movement is limited by a
    /// temperature decreasing linearly to zero, and per-vertex coordinate
    /// bounds may be given. Uses the calling thread's default random number
    /// generator for the random start. See [`FruchtermanReingoldOptions`] for
    /// all the parameters. Time complexity: `O(|V|²)` per iteration (less with the
    /// grid variant).
    /// Binds [`igraph_layout_fruchterman_reingold`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_fruchterman_reingold).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for
    /// non-positive weights, vectors of the wrong length, inconsistent bounds,
    /// NaN bounds, a lower bound of `+inf` or an upper bound of `-inf` (an
    /// infinite bound in the other direction means "no bound"), or a start matrix of the wrong shape or with NaN or infinite
    /// coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::layout::FruchtermanReingoldOptions;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false).unwrap();
    /// rng::seed(7).unwrap();
    /// // Keep every vertex in the right half-plane.
    /// let minx = [0.0; 4];
    /// let opts = FruchtermanReingoldOptions { minx: Some(&minx), ..Default::default() };
    /// let l = g.layout_fruchterman_reingold(&opts).unwrap();
    /// assert!(l.column(0).iter().all(|&x| x >= 0.0));
    /// ```
    pub fn layout_fruchterman_reingold(
        &self,
        options: &FruchtermanReingoldOptions<'_>,
    ) -> Result<Matrix> {
        let n = self.vcount();
        check_len("weights", options.weights, self.ecount())?;
        let bounds = Bounds::new(
            self,
            [
                options.minx,
                options.maxx,
                options.miny,
                options.maxy,
                None,
                None,
            ],
        )?;
        let (mut res, use_seed) = start_matrix(options.initial, n, 2)?;
        let weights = opt_view(options.weights);
        let start_temp = options
            .start_temp
            .unwrap_or_else(|| (n as f64).sqrt() / 10.0);
        igraph_call!(igraph_layout_fruchterman_reingold(
            self,
            &mut res,
            use_seed,
            int(options.niter),
            start_temp,
            options.grid.into(),
            opt_ptr(&weights),
            bounds.ptr(0),
            bounds.ptr(1),
            bounds.ptr(2),
            bounds.ptr(3)
        ))?;
        Ok(res)
    }

    /// Force-directed layout in 3D space with the Fruchterman–Reingold algorithm.
    ///
    /// The 3D version of [`layout_fruchterman_reingold`](Graph::layout_fruchterman_reingold)
    /// (the `grid` option is ignored, the `minz`/`maxz` bounds are used).
    ///
    /// Caveat (igraph 1.0.0 and 1.0.1): on *disconnected* graphs the C code
    /// adds the `z` component of the repulsion acting on one vertex of each
    /// pair to its `y` displacement instead of its `z` displacement, so the
    /// forces are slightly unbalanced; connected graphs are not affected.
    /// Binds [`igraph_layout_fruchterman_reingold_3d`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_fruchterman_reingold_3d).
    ///
    /// # Errors
    /// As for the 2D version.
    pub fn layout_fruchterman_reingold_3d(
        &self,
        options: &FruchtermanReingoldOptions<'_>,
    ) -> Result<Matrix> {
        let n = self.vcount();
        check_len("weights", options.weights, self.ecount())?;
        let bounds = Bounds::new(
            self,
            [
                options.minx,
                options.maxx,
                options.miny,
                options.maxy,
                options.minz,
                options.maxz,
            ],
        )?;
        let (mut res, use_seed) = start_matrix(options.initial, n, 3)?;
        let weights = opt_view(options.weights);
        let start_temp = options
            .start_temp
            .unwrap_or_else(|| (n as f64).sqrt() / 10.0);
        igraph_call!(igraph_layout_fruchterman_reingold_3d(
            self,
            &mut res,
            use_seed,
            int(options.niter),
            start_temp,
            opt_ptr(&weights),
            bounds.ptr(0),
            bounds.ptr(1),
            bounds.ptr(2),
            bounds.ptr(3),
            bounds.ptr(4),
            bounds.ptr(5)
        ))?;
        Ok(res)
    }

    /// Force-directed layout in the plane with the Kamada–Kawai spring algorithm.
    ///
    /// A spring is placed between *every* pair of vertices `u`, `v`, whose
    /// rest length is proportional to their (undirected, possibly weighted)
    /// graph distance `d(u, v)`, namely `sqrt(n) · d(u, v) / D` where `D` is
    /// the largest finite distance (so the drawing has a size of about
    /// `sqrt(n)`), and whose stiffness `kkconst / d(u, v)²` decreases with
    /// that distance; the energy is then minimized one vertex at a time.
    /// Vertices in different components are treated as being at distance
    /// `D`. It works particularly well for lattice-like, locally connected
    /// graphs; memory is `O(|V|²)`, so it is not suitable for large graphs.
    /// Without a start layout (and with no bounds) it starts from a circle of
    /// radius `0.36 sqrt(n)`, so the result is deterministic.
    /// The target distances are the ones [`Graph::distances`] would compute
    /// with `NeighborMode::All`. See [`KamadaKawaiOptions`]. Time complexity:
    /// `O(|V|)` per iteration after an `O(|V|² log |V|)` initialization.
    /// Binds [`igraph_layout_kamada_kawai`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_kamada_kawai).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for
    /// non-positive weights or `kkconst`, wrong lengths or shapes, NaN bounds,
    /// a lower bound of `+inf` or an upper bound of `-inf`, or NaN or infinite
    /// start coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::layout::KamadaKawaiOptions;
    /// // A path 0 - 1 - 2: drawn straight, with edges of length sqrt(3) / 2.
    /// let g = Graph::ring(3, false, false, false).unwrap();
    /// let l = g.layout_kamada_kawai(&KamadaKawaiOptions::default()).unwrap();
    /// let d = |i: usize, j: usize| (l[(i, 0)] - l[(j, 0)]).hypot(l[(i, 1)] - l[(j, 1)]);
    /// assert!((d(0, 2) - d(0, 1) - d(1, 2)).abs() < 1e-3);
    /// assert!((d(0, 1) - 3f64.sqrt() / 2.0).abs() < 1e-3);
    /// ```
    pub fn layout_kamada_kawai(&self, options: &KamadaKawaiOptions<'_>) -> Result<Matrix> {
        self.kamada_kawai_impl(options, false)
    }

    /// Force-directed layout in 3D space with the Kamada–Kawai spring algorithm.
    ///
    /// The 3D version of [`layout_kamada_kawai`](Graph::layout_kamada_kawai);
    /// without a start layout (and with no bounds) it starts from a
    /// [`layout_sphere`](Graph::layout_sphere) of radius `0.36 sqrt(n)`.
    /// Binds [`igraph_layout_kamada_kawai_3d`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_kamada_kawai_3d).
    ///
    /// # Errors
    /// As for the 2D version.
    pub fn layout_kamada_kawai_3d(&self, options: &KamadaKawaiOptions<'_>) -> Result<Matrix> {
        self.kamada_kawai_impl(options, true)
    }

    fn kamada_kawai_impl(&self, options: &KamadaKawaiOptions<'_>, three_d: bool) -> Result<Matrix> {
        let n = self.vcount();
        check_len("weights", options.weights, self.ecount())?;
        let (minz, maxz) = if three_d {
            (options.minz, options.maxz)
        } else {
            (None, None)
        };
        let bounds = Bounds::new(
            self,
            [
                options.minx,
                options.maxx,
                options.miny,
                options.maxy,
                minz,
                maxz,
            ],
        )?;
        let (mut res, use_seed) = start_matrix(options.initial, n, if three_d { 3 } else { 2 })?;
        let weights = opt_view(options.weights);
        let maxiter = int(options.maxiter.unwrap_or(n.saturating_mul(50)));
        let kkconst = options.kkconst.unwrap_or_else(|| nf(self));
        if three_d {
            igraph_call!(igraph_layout_kamada_kawai_3d(
                self,
                &mut res,
                use_seed,
                maxiter,
                options.epsilon,
                kkconst,
                opt_ptr(&weights),
                bounds.ptr(0),
                bounds.ptr(1),
                bounds.ptr(2),
                bounds.ptr(3),
                bounds.ptr(4),
                bounds.ptr(5)
            ))?;
        } else {
            igraph_call!(igraph_layout_kamada_kawai(
                self,
                &mut res,
                use_seed,
                maxiter,
                options.epsilon,
                kkconst,
                opt_ptr(&weights),
                bounds.ptr(0),
                bounds.ptr(1),
                bounds.ptr(2),
                bounds.ptr(3)
            ))?;
        }
        Ok(res)
    }

    /// Force-directed layout for large (connected) graphs, inspired by the
    /// Large Graph Layout program.
    ///
    /// The root is placed first, then its neighbors, then the second
    /// neighbors and so on (following a BFS); after each layer a
    /// Fruchterman–Reingold-style simulated annealing runs, computing
    /// repulsion only between vertices in nearby grid cells. The graph must
    /// be connected (igraph warns otherwise; lay out the pieces from
    /// [`Graph::decompose`] separately and merge them with
    /// [`layout_merge_dla`] instead). See [`LglOptions`]. Time
    /// complexity: ideally `O(dia · maxiter · (|V| + |E|))`.
    /// Binds [`igraph_layout_lgl`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_lgl).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if a
    /// parameter is not finite and positive (`repulserad` may be `+inf`), or if `cellsize` is so small
    /// compared to `area` that the grid would have more than
    /// `max(2^28, 4 |V|)` cells;
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) for
    /// an invalid root.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::layout::LglOptions;
    /// let g = Graph::kary_tree(40, 3, TreeMode::Undirected).unwrap();
    /// rng::seed(5).unwrap();
    /// let l = g.layout_lgl(&LglOptions { root: Some(0), ..Default::default() }).unwrap();
    /// assert_eq!(l.shape(), (40, 2));
    /// assert!(l.as_slice().iter().all(|x| x.is_finite()));
    /// ```
    pub fn layout_lgl(&self, options: &LglOptions) -> Result<Matrix> {
        let n = self.vcount();
        let nn = nf(self);
        if let Some(root) = options.root {
            check_vertices("root", &[root], n)?;
        }
        let area = options.area.unwrap_or(nn * nn);
        let maxdelta = options.maxdelta.unwrap_or(nn);
        let repulserad = options.repulserad.unwrap_or(area * nn);
        let cellsize = options.cellsize.unwrap_or_else(|| area.sqrt().sqrt());
        if n > 0 {
            // igraph only rejects values `<= 0`: NaN and infinite values
            // reach its grid code, which aborts the process.
            check_positive("area", area)?;
            check_positive("maxdelta", maxdelta)?;
            check_positive("coolexp", options.coolexp)?;
            // An infinite cutoff radius only drops the `dist² / repulserad`
            // term from the repulsive force, which is harmless.
            if repulserad.is_nan() || repulserad <= 0.0 {
                return Err(Error::invalid(format!(
                    "`repulserad` must be positive, got {repulserad}"
                )));
            }
            check_positive("cellsize", cellsize)?;
            // igraph covers the disc of the given area with a square grid of
            // `steps × steps` cells (a C integer), which must not overflow
            // nor be absurdly large.
            let steps = (2.0 * (area / std::f64::consts::PI).sqrt() / cellsize).ceil();
            let cells = steps * steps;
            let limit = (LGL_MAX_CELLS as f64).max(4.0 * n as f64);
            if !cells.is_finite() || cells > limit {
                return Err(Error::invalid(format!(
                    "`cellsize` {cellsize} is too small for `area` {area}: \
                     the grid would have {cells:e} cells (at most {limit:e} allowed)"
                )));
            }
        }
        let mut res = Matrix::new();
        igraph_call!(igraph_layout_lgl(
            self,
            &mut res,
            int(options.maxiter),
            maxdelta,
            area,
            options.coolexp,
            repulserad,
            cellsize,
            options.root.unwrap_or(-1)
        ))?;
        Ok(res)
    }

    /// Reingold–Tilford tree layout: parents centered above their children.
    ///
    /// Vertices are placed in levels by distance from the root (`y` is the
    /// depth), and subtrees are packed as tightly as possible. If the graph
    /// is not a tree a BFS spanning tree is used. `mode` selects which edges
    /// are followed from a parent ([`NeighborMode::Out`], `In`, or `All`,
    /// which is forced for undirected graphs). `roots` should reach every
    /// vertex (e.g. one per component): igraph hangs any unreachable vertex
    /// directly below the (single) root, or places it next to the roots when
    /// there are several. With several roots, igraph joins them under a
    /// hidden common root, so the roots are drawn at depth `y = 1` and their
    /// children at `y = 2`. `None` (or empty) selects the roots
    /// automatically: in igraph 1.0.1 with [`RootChoice::Degree`] below 500
    /// vertices and [`RootChoice::Eccentricity`] from 500 on (the C
    /// documentation states the opposite, the code does this; see
    /// [`roots_for_tree_layout`](Graph::roots_for_tree_layout) for a
    /// controllable choice). `rootlevel`, only used with several given roots,
    /// gives the extra depth of each root, which is useful for forests.
    /// Trees are conveniently built with [`Graph::kary_tree`]; use
    /// [`Graph::is_tree`] to check whether the drawing shows every edge, and
    /// [`Graph::unfold_tree`] to get the tree itself when the graph has cycles.
    /// Binds [`igraph_layout_reingold_tilford`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_reingold_tilford).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) for
    /// invalid roots, [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue)
    /// if `rootlevel` has negative entries or a huge sum (the extra levels
    /// are added as vertices), or is non-empty with a length
    /// different from that of `roots` (when several roots are given).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // A binary tree of depth 2 rooted at 0.
    /// let g = Graph::kary_tree(7, 2, TreeMode::Undirected).unwrap();
    /// let l = g.layout_reingold_tilford(NeighborMode::All, Some(&[0]), None).unwrap();
    /// assert_eq!(l.column(1), &[0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]); // depths
    /// assert_eq!(l[(0, 0)], (l[(1, 0)] + l[(2, 0)]) / 2.0); // root centered
    /// ```
    pub fn layout_reingold_tilford(
        &self,
        mode: NeighborMode,
        roots: Option<&[VertexId]>,
        rootlevel: Option<&[i64]>,
    ) -> Result<Matrix> {
        self.reingold_tilford_impl(mode, roots, rootlevel, false)
    }

    /// Circular Reingold–Tilford tree layout: the root at the center and the
    /// levels on concentric circles.
    ///
    /// Same parameters as [`layout_reingold_tilford`](Graph::layout_reingold_tilford);
    /// the distance from the origin is the depth of a vertex, and the
    /// horizontal positions of the plain layout are mapped linearly to
    /// angles, the leftmost vertex at angle 0 and the rightmost one at
    /// `2π (n - 1) / n`.
    /// Binds [`igraph_layout_reingold_tilford_circular`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_reingold_tilford_circular).
    ///
    /// # Errors
    /// As for [`layout_reingold_tilford`](Graph::layout_reingold_tilford).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::kary_tree(13, 3, TreeMode::Undirected).unwrap();
    /// let l = g.layout_reingold_tilford_circular(NeighborMode::All, Some(&[0]), None).unwrap();
    /// let radius = |v: usize| l[(v, 0)].hypot(l[(v, 1)]);
    /// assert_eq!(radius(0), 0.0);
    /// assert!((radius(1) - 1.0).abs() < 1e-12 && (radius(12) - 2.0).abs() < 1e-12);
    /// ```
    pub fn layout_reingold_tilford_circular(
        &self,
        mode: NeighborMode,
        roots: Option<&[VertexId]>,
        rootlevel: Option<&[i64]>,
    ) -> Result<Matrix> {
        self.reingold_tilford_impl(mode, roots, rootlevel, true)
    }

    fn reingold_tilford_impl(
        &self,
        mode: NeighborMode,
        roots: Option<&[VertexId]>,
        rootlevel: Option<&[i64]>,
        circular: bool,
    ) -> Result<Matrix> {
        let n = self.vcount();
        if let Some(r) = roots {
            check_vertices("roots", r, n)?;
            // Like igraph, `rootlevel` is only used (and checked) when it is
            // non-empty and there are several roots.
            if let Some(levels) = rootlevel
                && r.len() > 1
                && !levels.is_empty()
                && levels.len() != r.len()
            {
                return Err(Error::invalid("`roots` and `rootlevel` lengths differ"));
            }
        }
        if let Some(levels) = rootlevel {
            if levels.iter().any(|&x| x < 0) {
                return Err(Error::invalid(
                    "`rootlevel` must not contain negative levels",
                ));
            }
            // igraph adds one vertex per level to a copy of the graph,
            // summing the levels unchecked.
            let total = levels
                .iter()
                .try_fold(int(n), |acc, &x| acc.checked_add(x))
                .filter(|&t| t <= RT_MAX_VERTICES);
            if total.is_none() {
                return Err(Error::invalid(format!(
                    "the sum of `rootlevel` is too large (the graph plus the extra \
                     levels must have at most {RT_MAX_VERTICES} vertices)"
                )));
            }
        }
        // igraph writes into the (nominally `const`) `roots` vector when
        // `rootlevel` is used: always hand it an owned copy, through a
        // pointer derived from a mutable borrow.
        let mut roots = roots.map(VectorInt::from_slice);
        let roots_ptr = roots
            .as_mut()
            .map_or(ptr::null(), |r| ptr::from_mut(r).cast_const());
        let rootlevel = rootlevel.map(VectorInt::view);
        let mut res = Matrix::new();
        // An earlier ALL-mode adjacency list of a directed graph with mutual
        // edges may have cached a wrong "has multi-edges" flag (igraph 1.0.1),
        // which makes igraph abort when its OUT/IN adjacency list finds none:
        // drop the cache before the call, and after it (ALL mode may write
        // the wrong value again).
        let guard = self.is_directed();
        if guard {
            self.invalidate_cache();
        }
        let _reset = CacheReset(guard.then_some(self));
        if circular {
            igraph_call!(igraph_layout_reingold_tilford_circular(
                self,
                &mut res,
                mode.into(),
                roots_ptr,
                opt_ptr(&rootlevel)
            ))?;
        } else {
            igraph_call!(igraph_layout_reingold_tilford(
                self,
                &mut res,
                mode.into(),
                roots_ptr,
                opt_ptr(&rootlevel)
            ))?;
        }
        Ok(res)
    }

    /// Chooses a minimal set of roots for a nice tree layout, such that all
    /// vertices are reachable from them.
    ///
    /// For undirected graphs (or `mode = All`) one root is chosen per
    /// connected component; in directed mode one per strongly connected
    /// component without incoming (for `Out`) or outgoing (for `In`) edges
    /// (the components are those of [`Graph::connected_components`]). Within
    /// a component the root is chosen with the given [`RootChoice`] heuristic
    /// ([`RootChoice::Eccentricity`] relates to [`Graph::eccentricity`]).
    /// Typically used with [`layout_reingold_tilford`](Graph::layout_reingold_tilford).
    /// Binds [`igraph_roots_for_tree_layout`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_roots_for_tree_layout).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::layout::RootChoice;
    /// // Two stars: centers 0 and 4.
    /// let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (4, 5), (4, 6)], 7, false).unwrap();
    /// let roots = g.roots_for_tree_layout(NeighborMode::All, RootChoice::Degree).unwrap();
    /// assert_eq!(roots, vec![0, 4]);
    /// ```
    pub fn roots_for_tree_layout(
        &self,
        mode: NeighborMode,
        heuristic: RootChoice,
    ) -> Result<Vec<VertexId>> {
        let mut roots = VectorInt::new();
        igraph_call!(igraph_roots_for_tree_layout(
            self,
            mode.into(),
            &mut roots,
            heuristic.into()
        ))?;
        Ok(roots.into())
    }

    /// Sugiyama layout for layered (directed acyclic) graphs, minimizing edge
    /// crossings.
    ///
    /// Vertices of the same layer are placed on the same horizontal line
    /// (`y = layer · vgap`; empty layers are skipped by the crossing
    /// minimization but still take room vertically); their `x`
    /// coordinates follow the heuristic of Sugiyama, Tagawa and Toda (1981).
    /// Without given layers, igraph breaks cycles (via a feedback arc set) and
    /// computes a layering itself. Edges spanning several layers are routed
    /// through dummy vertices, whose positions are returned in
    /// [`SugiyamaLayout::routing`]. Disconnected components are placed side
    /// by side. See [`SugiyamaOptions`]. Related: [`Graph::feedback_arc_set`]
    /// (the edges reversed to break cycles), [`Graph::is_dag`] and
    /// [`Graph::topological_sorting`].
    /// Binds [`igraph_layout_sugiyama`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_sugiyama).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `layers`
    /// or `weights` have the wrong length, or `layers` contains a negative
    /// index.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::layout::SugiyamaOptions;
    /// // 0 -> 1 -> 2 plus a shortcut 0 -> 2 spanning two layers.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 2)], 3, true).unwrap();
    /// let s = g.layout_sugiyama(&SugiyamaOptions::default()).unwrap();
    /// assert_eq!(s.coords.column(1), &[0.0, 1.0, 2.0]);
    /// assert_eq!(s.routing[2].nrow(), 1); // one bend for the shortcut
    /// ```
    pub fn layout_sugiyama(&self, options: &SugiyamaOptions<'_>) -> Result<SugiyamaLayout> {
        check_len("layers", options.layers, self.vcount())?;
        check_len("weights", options.weights, self.ecount())?;
        if options.layers.is_some_and(|l| l.iter().any(|&x| x < 0)) {
            return Err(Error::invalid("layer indices must not be negative"));
        }
        let layers = options.layers.map(VectorInt::view);
        let weights = opt_view(options.weights);
        let mut coords = Matrix::new();
        let mut routing = MatrixList::new();
        igraph_call!(igraph_layout_sugiyama(
            self,
            &mut coords,
            &mut routing,
            opt_ptr(&layers),
            options.hgap,
            options.vgap,
            int(options.maxiter),
            opt_ptr(&weights)
        ))?;
        Ok(SugiyamaLayout {
            coords,
            routing: routing.into_vec(),
        })
    }

    /// Places the vertices in a space of dimension `dim` with classical
    /// (Torgerson) multidimensional scaling.
    ///
    /// The Euclidean distances of the layout approximate the given symmetric
    /// `n` × `n` distance matrix (symmetry is not checked, and its diagonal
    /// is ignored), or, if `None`, the undirected shortest path lengths. `dim`
    /// must be at least 2 and at most the number of vertices. Disconnected
    /// graphs are laid out
    /// per component and merged with [`layout_merge_dla`], which only works
    /// for `dim = 2`. Vertices symmetric to each other (e.g. leaves of the
    /// same parent) may receive identical coordinates. Time complexity:
    /// usually around `O(|V|² dim)`. The default distance matrix is the one of
    /// [`Graph::distances`] with `NeighborMode::All`.
    /// Binds [`igraph_layout_mds`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_mds).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for a
    /// distance matrix of the wrong shape, `dim` out of range, or `dim > 2`
    /// on a disconnected graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // Three points on a line at 0, 1, 3: MDS recovers their distances.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// let d = Matrix::from_rows(&[[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]]).unwrap();
    /// let l = g.layout_mds(Some(&d), 2).unwrap();
    /// let dist = |i: usize, j: usize| (l[(i, 0)] - l[(j, 0)]).hypot(l[(i, 1)] - l[(j, 1)]);
    /// assert!((dist(0, 2) - 3.0).abs() < 1e-9 && (dist(0, 1) - 1.0).abs() < 1e-9);
    /// ```
    pub fn layout_mds(&self, dist: Option<&Matrix>, dim: usize) -> Result<Matrix> {
        let n = self.vcount();
        if let Some(d) = dist
            && d.shape() != (n, n)
        {
            return Err(Error::invalid(format!(
                "the distance matrix is {}x{}, expected {n}x{n}",
                d.nrow(),
                d.ncol()
            )));
        }
        if let Some(d) = dist {
            // NaN distances reach LAPACK, which converts them to integers.
            crate::linalg::check_finite(d.as_slice(), "the distance matrix")?;
        }
        let dist = dist.map_or(ptr::null(), |d| d as *const Matrix);
        let mut res = Matrix::new();
        igraph_call!(igraph_layout_mds(self, &mut res, dist, int(dim)))?;
        Ok(res)
    }

    /// Simple two-row layout for bipartite graphs.
    ///
    /// Vertices with type `true` are placed on the line `y = 0`, those with
    /// type `false` on `y = vgap`; positions within the rows are then
    /// optimized to reduce edge crossings with the Sugiyama heuristic, using
    /// `hgap` as the preferred minimum gap and at most `maxiter` iterations
    /// (100 is a reasonable default). Only the `types` matter, edges between
    /// vertices of the same type are allowed: a proper 2-coloring can be
    /// obtained with [`Graph::bipartite_types`], and the bipartite
    /// constructors (e.g. [`Graph::full_bipartite`]) return the types with
    /// the graph.
    /// Binds [`igraph_layout_bipartite`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_bipartite).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `types`
    /// has the wrong length or `hgap` is negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 2), (1, 2), (1, 3)], 4, false).unwrap();
    /// let l = g.layout_bipartite(&[false, false, true, true], 1.0, 1.0, 100).unwrap();
    /// assert_eq!(l.column(1), &[1.0, 1.0, 0.0, 0.0]);
    /// ```
    pub fn layout_bipartite(
        &self,
        types: &[bool],
        hgap: f64,
        vgap: f64,
        maxiter: usize,
    ) -> Result<Matrix> {
        check_len("types", Some(types), self.vcount())?;
        let types = VectorBool::view(types);
        let mut res = Matrix::new();
        igraph_call!(igraph_layout_bipartite(
            self,
            types.as_ptr(),
            &mut res,
            hgap,
            vgap,
            int(maxiter)
        ))?;
        Ok(res)
    }

    /// Layout with Uniform Manifold Approximation and Projection (UMAP), in
    /// the plane. **Experimental in igraph.**
    ///
    /// UMAP embeds a (typically sparse, e.g. k-nearest-neighbors) distance
    /// graph: distances are turned into exponentially decaying weights in
    /// `[0, 1]`, and a stochastic gradient descent on the cross-entropy then
    /// places strongly connected vertices close together, while repelling
    /// unconnected ones beyond `min_dist`. Without distances all edges have
    /// the same weight. If `distances_are_weights` is set, the `distances`
    /// are used directly as weights (see
    /// [`layout_umap_compute_weights`](Graph::layout_umap_compute_weights)).
    /// A typical input is a k-nearest-neighbor graph of data points built
    /// with [`Graph::nearest_neighbor_graph`], with the distances from
    /// [`Graph::spatial_edge_lengths`]. Uses the calling thread's default
    /// random number generator. See [`UmapOptions`].
    /// Binds [`igraph_layout_umap`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_umap).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for
    /// invalid distances, a negative `min_dist` or a wrongly shaped or non-finite start.
    pub fn layout_umap(&self, options: &UmapOptions<'_>) -> Result<Matrix> {
        self.umap_impl(options, false)
    }

    /// Layout with UMAP in 3D space. **Experimental in igraph.**
    ///
    /// The 3D version of [`layout_umap`](Graph::layout_umap).
    /// Binds [`igraph_layout_umap_3d`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_umap_3d).
    ///
    /// # Errors
    /// As for the 2D version.
    pub fn layout_umap_3d(&self, options: &UmapOptions<'_>) -> Result<Matrix> {
        self.umap_impl(options, true)
    }

    fn umap_impl(&self, options: &UmapOptions<'_>, three_d: bool) -> Result<Matrix> {
        check_len("distances", options.distances, self.ecount())?;
        let (mut res, use_seed) =
            start_matrix(options.initial, self.vcount(), if three_d { 3 } else { 2 })?;
        let distances = opt_view(options.distances);
        let f = if three_d {
            igraph_layout_umap_3d
        } else {
            igraph_layout_umap
        };
        igraph_call!(f(
            self,
            &mut res,
            use_seed,
            opt_ptr(&distances),
            options.min_dist,
            int(options.epochs),
            options.distances_are_weights
        ))?;
        Ok(res)
    }

    /// Computes the UMAP edge weights from the edge distances.
    /// **Experimental in igraph.**
    ///
    /// For each vertex a scale factor and a connectivity correction are
    /// computed, and distances become exponentially decaying weights in
    /// `[0, 1]`. The graph may be directed but must have no loops or multi-edges
    /// (pairs of opposite directed edges are allowed): such pairs are
    /// symmetrized with the *fuzzy union* `W = W1 + W2 - W1 W2`, stored on one
    /// of the two edges, while the other gets weight 0 (see
    /// [`Graph::is_simple`] and [`Graph::simplify`]). Loops are reported as
    /// errors, but multi-edges are not detected by igraph: all but one of
    /// the parallel edges silently get weight 0. Pass the result to
    /// [`layout_umap`](Graph::layout_umap) with `distances_are_weights`.
    /// `None` means that all edges have the same distance.
    /// Binds [`igraph_layout_umap_compute_weights`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_umap_compute_weights).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for
    /// negative or NaN distances, a wrong length, or loops.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let w = g.layout_umap_compute_weights(Some(&[1.0, 2.0, 3.0])).unwrap();
    /// assert_eq!(w.len(), 3);
    /// assert!(w.iter().all(|&x| (0.0..=1.0).contains(&x)));
    /// ```
    pub fn layout_umap_compute_weights(&self, distances: Option<&[f64]>) -> Result<Vec<f64>> {
        check_len("distances", distances, self.ecount())?;
        let distances = opt_view(distances);
        let mut weights = Vector::new();
        igraph_call!(igraph_layout_umap_compute_weights(
            self,
            opt_ptr(&distances),
            &mut weights
        ))?;
        Ok(weights.into())
    }

    /// The DrL (Distributed Recursive Layout) force-directed layout, in the plane.
    ///
    /// Designed for large graphs, it runs a sequence of simulated annealing
    /// phases driven by the given [`DrlOptions`] (start from a
    /// [`DrlTemplate`]), cutting highly stressed edges in the late phases to
    /// produce less dense, clustered layouts (Martin et al., 2008). `weights`
    /// must be positive (`None`: unit weights); `initial` gives optional
    /// starting positions (`n` × 2).
    /// Binds [`igraph_layout_drl`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_drl).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for
    /// negative damping multipliers, non-positive weights, weights above
    /// `1e20` (DrL computes in single precision and overflows; rescale such
    /// weights), NaN or infinite start coordinates, or wrong lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::layout::DrlOptions;
    /// let edges: Vec<(i64, i64)> = (0..10).map(|i| (i, (i + 1) % 10)).collect();
    /// let g = Graph::from_edges(&edges, 10, false).unwrap();
    /// rng::seed(3).unwrap();
    /// let l = g.layout_drl(&DrlOptions::default(), None, None).unwrap();
    /// assert_eq!(l.shape(), (10, 2));
    /// ```
    pub fn layout_drl(
        &self,
        options: &DrlOptions,
        weights: Option<&[f64]>,
        initial: Option<&Matrix>,
    ) -> Result<Matrix> {
        self.drl_impl(options, weights, initial, false)
    }

    /// The DrL force-directed layout in 3D space.
    ///
    /// The 3D version of [`layout_drl`](Graph::layout_drl) (`initial`, if
    /// given, must be `n` × 3).
    /// Binds [`igraph_layout_drl_3d`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_drl_3d).
    ///
    /// # Errors
    /// As for the 2D version.
    pub fn layout_drl_3d(
        &self,
        options: &DrlOptions,
        weights: Option<&[f64]>,
        initial: Option<&Matrix>,
    ) -> Result<Matrix> {
        self.drl_impl(options, weights, initial, true)
    }

    fn drl_impl(
        &self,
        options: &DrlOptions,
        weights: Option<&[f64]>,
        initial: Option<&Matrix>,
        three_d: bool,
    ) -> Result<Matrix> {
        check_len("weights", weights, self.ecount())?;
        if let Some(w) = weights
            && !w.iter().all(|&x| x > 0.0 && x <= DRL_MAX_WEIGHT)
        {
            // Huge weights overflow DrL's single-precision energies into NaN
            // positions, which its density grid converts to integers (UB).
            return Err(Error::invalid(format!(
                "DrL weights must be positive and at most {DRL_MAX_WEIGHT:e}"
            )));
        }
        let (mut res, use_seed) =
            start_matrix(initial, self.vcount(), if three_d { 3 } else { 2 })?;
        let weights = opt_view(weights);
        let f = if three_d {
            igraph_layout_drl_3d
        } else {
            igraph_layout_drl
        };
        igraph_call!(f(self, &mut res, use_seed, options, opt_ptr(&weights)))?;
        Ok(res)
    }

    /// The graphopt force-directed layout (a port of Michael Schmuhl's graphopt).
    ///
    /// Vertices are charged particles repelling each other (Coulomb's law,
    /// ignored beyond distance 500) and edges are springs (Hooke's law); the
    /// physical system is simulated for `niter` steps, without simulated
    /// annealing, so a stable fixed point is not guaranteed. A layout can be
    /// refined by passing it back as `initial`. See [`GraphoptOptions`].
    /// Time complexity: `O(niter (|V|² + |E|))`, or `O(niter |E|)` with zero
    /// charge.
    /// Binds [`igraph_layout_graphopt`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_graphopt).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for a
    /// wrongly shaped or non-finite start.
    ///
    /// # Examples
    ///
    /// Pure springs of rest length 1 (no charge) stretch a squeezed path
    /// along its diagonal (values from igraph's `igraph_layout_graphopt` unit
    /// test):
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::layout::GraphoptOptions;
    /// let g = Graph::ring(4, false, false, false).unwrap(); // the path 0 - 1 - 2 - 3
    /// let start = Matrix::from_rows(&[[0.15, -0.15], [0.05, -0.05], [-0.05, 0.05], [-0.15, 0.15]]).unwrap();
    /// let opts = GraphoptOptions {
    ///     node_charge: 0.0,
    ///     spring_length: 1.0,
    ///     spring_constant: 10.0,
    ///     initial: Some(&start),
    ///     ..Default::default()
    /// };
    /// let l = g.layout_graphopt(&opts).unwrap();
    /// assert!((l[(0, 0)] - 1.06066).abs() < 1e-5 && (l[(1, 1)] + 0.353553).abs() < 1e-5);
    /// ```
    pub fn layout_graphopt(&self, options: &GraphoptOptions<'_>) -> Result<Matrix> {
        let (mut res, use_seed) = start_matrix(options.initial, self.vcount(), 2)?;
        igraph_call!(igraph_layout_graphopt(
            self,
            &mut res,
            int(options.niter),
            options.node_charge,
            options.node_mass,
            options.spring_length,
            options.spring_constant,
            options.max_sa_movement,
            use_seed
        ))?;
        Ok(res)
    }

    /// The GEM force-directed layout (Frick, Ludwig and Mehldau, 1994).
    ///
    /// Vertices are updated one at a time in random order, each with its own
    /// local temperature adapted to detect oscillations and rotations; the
    /// algorithm stops when the global temperature drops below `temp_min`
    /// or after `maxiter` vertex updates. Edge directions are ignored. See
    /// [`GemOptions`]. Time complexity: `O(t · n · (n + e))` for `t` steps.
    /// Binds [`igraph_layout_gem`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_gem).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) unless
    /// `0 < temp_min <= temp_init <= temp_max`, or for a wrongly shaped or non-finite start.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::layout::GemOptions;
    /// let g = Graph::ring(8, false, false, true).unwrap();
    /// rng::seed(11).unwrap();
    /// let l = g.layout_gem(&GemOptions::default()).unwrap();
    /// assert_eq!(l.shape(), (8, 2));
    /// // Zero iterations keep the start.
    /// let opts = GemOptions { maxiter: Some(0), initial: Some(&l), ..Default::default() };
    /// assert_eq!(g.layout_gem(&opts).unwrap(), l);
    /// ```
    pub fn layout_gem(&self, options: &GemOptions<'_>) -> Result<Matrix> {
        let n = self.vcount();
        let nn = nf(self);
        let (mut res, use_seed) = start_matrix(options.initial, n, 2)?;
        igraph_call!(igraph_layout_gem(
            self,
            &mut res,
            use_seed,
            int(options
                .maxiter
                .unwrap_or_else(|| n.saturating_mul(n).saturating_mul(40))),
            options.temp_max.unwrap_or(nn),
            options.temp_min,
            options.temp_init.unwrap_or_else(|| nn.sqrt())
        ))?;
        if n == 0 {
            // igraph returns early on the null graph without resizing.
            res.resize(0, 2);
        }
        Ok(res)
    }

    /// The Davidson–Harel simulated annealing layout (1996).
    ///
    /// Minimizes an energy combining node-node distances, distances from the
    /// border, edge lengths, edge crossings and node-edge distances, first
    /// with simulated annealing then with a fine tuning phase; coordinates are
    /// kept within the bounds of the layout rectangle. Edge directions are
    /// ignored. The energy weights are hard to tune in general; see
    /// [`DavidsonHarelOptions`] for defaults determined by experimentation.
    /// Time complexity: `O(n² + m²)` per annealing iteration, `O(mn)` per fine
    /// tuning iteration.
    /// Binds [`igraph_layout_davidson_harel`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_davidson_harel).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `cool_fact` is not in `(0, 1)`, if `maxiter + fineiter` overflows a
    /// 64-bit integer, or for a wrongly shaped or non-finite start.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::layout::DavidsonHarelOptions;
    /// let g = Graph::full(10, false, false).unwrap();
    /// rng::seed(42).unwrap();
    /// let l = g.layout_davidson_harel(&DavidsonHarelOptions::default()).unwrap();
    /// assert_eq!(l.shape(), (10, 2));
    /// assert!(l.as_slice().iter().all(|x| x.abs() < 20.0));
    /// ```
    pub fn layout_davidson_harel(&self, options: &DavidsonHarelOptions<'_>) -> Result<Matrix> {
        let n = self.vcount();
        let density = if n < 2 {
            0.0
        } else {
            (self.ecount() as f64 / (n as f64 * (n as f64 - 1.0) / 2.0)).min(1.0)
        };
        let fineiter = options
            .fineiter
            .unwrap_or_else(|| ((n.max(1) as f64).log2() as usize).max(10));
        // igraph loops while `round < maxiter + fineiter`.
        let (maxiter, fineiter) = (int(options.maxiter), int(fineiter));
        if maxiter.checked_add(fineiter).is_none() {
            return Err(Error::invalid(
                "`maxiter + fineiter` does not fit in a 64-bit integer",
            ));
        }
        let (mut res, use_seed) = start_matrix(options.initial, n, 2)?;
        igraph_call!(igraph_layout_davidson_harel(
            self,
            &mut res,
            use_seed,
            maxiter,
            fineiter,
            options.cool_fact,
            options.weight_node_dist,
            options.weight_border,
            options.weight_edge_lengths.unwrap_or(density / 10.0),
            options
                .weight_edge_crossings
                .unwrap_or(1.0 - density.sqrt()),
            options
                .weight_node_edge_dist
                .unwrap_or((1.0 - density) / 5.0)
        ))?;
        Ok(res)
    }

    /// Centers a layout on the origin and rotates it to align it with the
    /// coordinate axes, in place.
    ///
    /// The principal axes are computed from the edge directions (weighted by
    /// squared edge lengths), or from the vertex positions if there are no
    /// edges of non-zero length. Useful after force-directed layouts; works
    /// in any dimension. Time complexity: `O(|V| + |E|)`.
    /// Binds [`igraph_layout_align`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_align).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the
    /// layout does not have one row per vertex, has zero columns (and the
    /// graph is not the null graph), or contains NaN or infinite coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // A diagonal segment becomes horizontal and centered.
    /// let g = Graph::from_edges(&[(0, 1)], 2, false).unwrap();
    /// let mut l = Matrix::from_rows(&[[1.0, 1.0], [3.0, 3.0]]).unwrap();
    /// g.layout_align(&mut l).unwrap();
    /// assert!(l[(0, 1)].abs() < 1e-9 && l[(1, 1)].abs() < 1e-9);
    /// assert!((l[(0, 0)] + l[(1, 0)]).abs() < 1e-9);
    /// ```
    pub fn layout_align(&self, layout: &mut Matrix) -> Result<()> {
        if layout.nrow() != self.vcount() {
            return Err(Error::invalid(format!(
                "the layout has {} rows, expected one per vertex ({})",
                layout.nrow(),
                self.vcount()
            )));
        }
        if !layout.as_slice().iter().all(|x| x.is_finite()) {
            return Err(Error::invalid(
                "the layout contains NaN or infinite coordinates",
            ));
        }
        igraph_call!(igraph_layout_align(self, layout))
    }
}

/// Merges the 2D layouts of several graphs (typically the components of a
/// graph) into one, using diffusion-limited aggregation (DLA).
///
/// Each layout is covered by a circle and rescaled so that the area of the
/// circle grows with the size of the graph; the largest layout is placed at
/// the origin and the others, from larger to smaller, perform random walks
/// until they stick next to the already placed ones. The result has the rows
/// of all the layouts, in the given order. The graphs are currently only
/// used for bookkeeping (igraph only looks at the coordinates). Uses the
/// calling thread's default random number generator. The typical input is
/// the list of components from [`Graph::decompose`], each laid out on its
/// own. `graphs` is any collection of graphs or references to graphs (like
/// the multi-graph functions of [`operators`](crate::operators)), e.g. the
/// `&Vec<Graph>` returned by `decompose` or a `&[&Graph]`.
/// Binds [`igraph_layout_merge_dla`](https://igraph.org/c/html/latest/igraph-Layout.html#igraph_layout_merge_dla).
///
/// # Errors
/// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the number
/// of graphs and layouts differ, no layout is given, a layout is empty or
/// is not 2D, or a layout does not have one row per vertex of its graph.
///
/// # Examples
///
/// ```
/// use igraph::prelude::*;
/// use igraph::layout::layout_merge_dla;
/// // A triangle and a separate edge, laid out component by component.
/// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4)], 5, false).unwrap();
/// let parts = g.decompose(Connectedness::Weak, None, 1).unwrap();
/// let layouts: Vec<Matrix> = parts.iter().map(|p| p.layout_circle(..).unwrap()).collect();
/// rng::seed(42).unwrap();
/// // Any collection of graphs works: `&Vec<Graph>`, `&[&Graph]`, `Vec<Graph>`...
/// let merged = layout_merge_dla(&parts, &layouts).unwrap();
/// assert_eq!(merged.shape(), (5, 2));
/// // Each component is only scaled and translated: the triangle stays equilateral.
/// let d = |i: usize, j: usize| (merged[(i, 0)] - merged[(j, 0)]).hypot(merged[(i, 1)] - merged[(j, 1)]);
/// assert!((d(0, 1) - d(1, 2)).abs() < 1e-9 && (d(1, 2) - d(2, 0)).abs() < 1e-9);
/// ```
pub fn layout_merge_dla<G: AsRef<Graph>>(
    graphs: impl IntoIterator<Item = G>,
    coords: &[Matrix],
) -> Result<Matrix> {
    let graphs: Vec<G> = graphs.into_iter().collect();
    let graphs: Vec<&Graph> = graphs.iter().map(AsRef::as_ref).collect();
    if graphs.len() != coords.len() {
        return Err(Error::invalid(format!(
            "{} graphs but {} layouts given",
            graphs.len(),
            coords.len()
        )));
    }
    if coords.is_empty() {
        return Err(Error::invalid("at least one layout is needed"));
    }
    for (i, (g, m)) in graphs.iter().zip(coords).enumerate() {
        if m.ncol() != 2 || m.nrow() == 0 {
            return Err(Error::invalid(format!(
                "layout {i} is {}x{}, but only non-empty 2D layouts can be merged",
                m.nrow(),
                m.ncol()
            )));
        }
        if m.nrow() != g.vcount() {
            return Err(Error::invalid(format!(
                "layout {i} has {} rows, but its graph has {} vertices",
                m.nrow(),
                g.vcount()
            )));
        }
    }
    let graph_ptrs = GraphPtrs::new(&graphs)?;
    let mut list = MatrixList::new();
    for m in coords {
        list.push(m.clone());
    }
    let mut res = Matrix::new();
    igraph_call!(igraph_layout_merge_dla(&graph_ptrs.raw, &list, &mut res))?;
    Ok(res)
}

/// A graph is trivially a reference to itself: this lets functions taking
/// several graphs, like [`layout_merge_dla`], accept collections of graphs
/// (`Vec<Graph>`, `&[Graph]`) as well as of references (`&[&Graph]`).
impl AsRef<Graph> for Graph {
    fn as_ref(&self) -> &Graph {
        self
    }
}

/// An `igraph_vector_ptr_t` of borrowed graph pointers (never owning them).
struct GraphPtrs<'a> {
    raw: igraph_vector_ptr_t,
    _borrow: std::marker::PhantomData<&'a Graph>,
}

impl<'a> GraphPtrs<'a> {
    fn new(graphs: &[&'a Graph]) -> Result<Self> {
        let mut raw = MaybeUninit::<igraph_vector_ptr_t>::uninit();
        igraph_call!(igraph_vector_ptr_init(raw.as_mut_ptr(), 0))?;
        // No item destructor is set: destroying the vector leaves the graphs alone.
        let mut ptrs = Self {
            raw: unsafe { raw.assume_init() },
            _borrow: std::marker::PhantomData,
        };
        for g in graphs {
            let p = *g as *const Graph as *mut c_void;
            igraph_call!(igraph_vector_ptr_push_back(&mut ptrs.raw, p))?;
        }
        Ok(ptrs)
    }
}

impl Drop for GraphPtrs<'_> {
    fn drop(&mut self) {
        unsafe { igraph_vector_ptr_destroy(&mut self.raw) };
    }
}

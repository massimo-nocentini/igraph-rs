//! Graph isomorphism, motifs and graphlets (`igraph_isomorphism.h`,
//! `igraph_motifs.h`, `igraph_graphlets.h`).
//!
//! Two graphs are *isomorphic* when they become indistinguishable once
//! their vertex labels are removed, i.e. when a bijection between their
//! vertex sets maps the edges of one onto the edges of the other. This
//! module wraps the four families of isomorphism tools of igraph:
//!
//! * the **generic** entry points [`Graph::isomorphic`] and
//!   [`Graph::subisomorphic`], which pick a suitable algorithm by
//!   themselves;
//! * **VF2** (Foggia, Sansone and Vento, 2001), which supports vertex and edge
//!   colors, arbitrary compatibility predicates written as Rust closures,
//!   counting and listing of all (sub)isomorphisms, and streaming them to a
//!   closure that can stop the search early; configure it with
//!   [`Vf2Options`];
//! * **Bliss** (Junttila and Kaski), a successor of NAUTY, which computes
//!   canonical labelings and automorphism groups, and reports the size of
//!   the automorphism group *exactly*, as a decimal string (see
//!   [`BlissInfo`] and the splitting heuristics [`BlissSh`]);
//! * **LAD** (Solnon, 2010) for (induced) subgraph isomorphism with
//!   per-vertex *domains*.
//!
//! In addition, all directed graphs on 3–4 vertices and all undirected
//! graphs on 3–6 vertices are numbered by *isomorphism classes*
//! ([`Graph::isoclass`], [`Graph::isoclass_create`], [`graph_count`]), which
//! are used by the **motif** finder ([`Graph::motifs_randesu`], FANMOD's
//! RAND-ESU algorithm), and the module also provides the dyad and triad
//! censuses, triangle listing and counting, and the **graphlet
//! decomposition** of weighted graphs (Azari Soufiani and Airoldi).
//!
//! VF2 and Bliss only support *simple* graphs (Bliss tolerates self-loops);
//! use [`Graph::simplify_and_colorize`] to encode multi-edges and self-loops
//! as edge and vertex colors first, as [`Graph::isomorphic`] does
//! automatically.
//!
//! # Example
//!
//! ```
//! use igraph::prelude::*;
//! use igraph::isomorphism::{BlissSh, Vf2Options};
//!
//! // A 4-cycle, and the same cycle with scrambled labels.
//! let c4 = Graph::ring(4, false, false, true).unwrap();
//! let scrambled = c4.permute_vertices(&[2, 0, 3, 1]).unwrap();
//! assert!(c4.isomorphic(&scrambled).unwrap());
//!
//! // VF2 also returns a witness mapping ...
//! let m = c4.isomorphic_vf2(&scrambled, &mut Vf2Options::new()).unwrap().unwrap();
//! for (u, v) in c4.edge_list() {
//!     let (a, b) = (m.map12[u as usize], m.map12[v as usize]);
//!     assert!(scrambled.get_eid(a, b, false).unwrap().is_some());
//! }
//! // ... and counts them: the dihedral group of the square has 8 elements.
//! assert_eq!(c4.count_isomorphisms_vf2(&scrambled, &mut Vf2Options::new()).unwrap(), 8);
//! assert_eq!(c4.count_automorphisms(None).unwrap(), 8.0);
//!
//! // A path on 3 vertices occurs 4 times in C4 as an induced subgraph (motif).
//! let hist = c4.motifs_randesu(3, None).unwrap();
//! assert!(hist[0].is_nan() && hist[1].is_nan()); // disconnected classes
//! assert_eq!(&hist[2..], &[4.0, 0.0]);
//!
//! // The Petersen graph has 120 symmetries, and Bliss reports them exactly.
//! let petersen = Graph::famous("Petersen").unwrap();
//! let info = petersen.count_automorphisms_bliss(None, BlissSh::Fl).unwrap();
//! assert_eq!(info.group_size, "120");
//! ```
//!
//! # Provided functionality
//!
//! | Rust | C function | What |
//! |------|------------|------|
//! | [`Graph::isomorphic`] | `igraph_isomorphic` | automatic isomorphism test |
//! | [`Graph::subisomorphic`] | `igraph_subisomorphic` | automatic subgraph isomorphism test |
//! | [`Graph::isomorphic_vf2`] | `igraph_isomorphic_vf2` | VF2 test with mapping |
//! | [`Graph::count_isomorphisms_vf2`] | `igraph_count_isomorphisms_vf2` | count isomorphisms |
//! | [`Graph::get_isomorphisms_vf2`] | `igraph_get_isomorphisms_vf2` | list isomorphisms |
//! | [`Graph::get_isomorphisms_vf2_callback`] | `igraph_get_isomorphisms_vf2_callback` | stream isomorphisms to a closure |
//! | [`Graph::subisomorphic_vf2`] | `igraph_subisomorphic_vf2` | VF2 subgraph test with mapping |
//! | [`Graph::count_subisomorphisms_vf2`] | `igraph_count_subisomorphisms_vf2` | count subgraph isomorphisms |
//! | [`Graph::get_subisomorphisms_vf2`] | `igraph_get_subisomorphisms_vf2` | list subgraph isomorphisms |
//! | [`Graph::get_subisomorphisms_vf2_callback`] | `igraph_get_subisomorphisms_vf2_callback` | stream subgraph isomorphisms |
//! | [`Graph::subisomorphic_lad`], [`Graph::get_subisomorphisms_lad`] | `igraph_subisomorphic_lad` | LAD, with domains and induced mode |
//! | [`Graph::isomorphic_bliss`] | `igraph_isomorphic_bliss` | Bliss test with mapping and statistics |
//! | [`Graph::canonical_permutation`], [`Graph::canonical_permutation_bliss`] | `igraph_canonical_permutation(_bliss)` | canonical labeling |
//! | [`Graph::canonical_form`] | (canonical labeling + `igraph_permute_vertices`) | canonical representative |
//! | [`Graph::count_automorphisms`], [`Graph::count_automorphisms_bliss`] | `igraph_count_automorphisms(_bliss)` | automorphism group size |
//! | [`Graph::automorphism_group`], [`Graph::automorphism_group_bliss`] | `igraph_automorphism_group(_bliss)` | automorphism group generators |
//! | [`Graph::simplify_and_colorize`] | `igraph_simplify_and_colorize` | multigraph → colored simple graph |
//! | [`invert_permutation`] | `igraph_invert_permutation` | inverse of a permutation |
//! | [`Graph::isoclass`], [`Graph::isoclass_subgraph`] | `igraph_isoclass(_subgraph)` | isomorphism class of small graphs |
//! | [`Graph::isoclass_create`] | `igraph_isoclass_create` | representative of an isomorphism class |
//! | [`graph_count`] | `igraph_graph_count` | number of unlabeled graphs |
//! | [`Graph::motifs_randesu`] | `igraph_motifs_randesu` | motif histogram |
//! | [`Graph::motifs_randesu_callback`] | `igraph_motifs_randesu_callback` | stream motifs to a closure |
//! | [`Graph::motifs_randesu_no`] | `igraph_motifs_randesu_no` | number of connected subgraphs |
//! | [`Graph::motifs_randesu_estimate`] | `igraph_motifs_randesu_estimate` | estimate of the above |
//! | [`Graph::dyad_census`] | `igraph_dyad_census` | mutual / asymmetric / null dyads |
//! | [`Graph::triad_census`] | `igraph_triad_census` | the 16 MAN triad types |
//! | [`Graph::count_triangles`], [`Graph::count_adjacent_triangles`], [`Graph::list_triangles`] | `igraph_count_triangles`, ... | triangles |
//! | [`Graph::graphlets`], [`Graph::graphlets_candidate_basis`], [`Graph::graphlets_project`] | `igraph_graphlets*` | graphlet decomposition |
//!
//! # See also
//!
//! * [`Graph::is_same_graph`] tests *labeled* equality (same vertex ids and
//!   edges), while this module tests equality up to relabeling;
//!   [`Graph::permute_vertices`] relabels a graph.
//! * [`Graph::famous`], [`Graph::full`],
//!   [`Graph::ring`] and [`Graph::lcf`] build the classic symmetric graphs
//!   used as test cases here; [`rng::seed`](crate::rng::seed) makes random
//!   relabelings and motif sampling reproducible (per thread).
//! * The [`cliques`](crate::cliques) module (complete subgraphs, used by the
//!   graphlet decomposition), and the transitivity measures of
//!   [`mixing`](crate::mixing), e.g. [`Graph::transitivity_undirected`],
//!   which are ratios of triangle counts.
//! * [`Graph::reciprocity`] summarizes the dyad census in one number.
//! * [`Graph::simplify`] and [`Graph::count_multiple`] for dealing with
//!   multigraphs without encoding them as colors.

use crate::{
    cliques::directed_cache_guard,
    error::{Error, ErrorKind, Result, catch_panic, catch_panic_or},
    ffi::*,
    graph::{Graph, VertexId},
    igraph_call,
    list::VectorIntList,
    selector::VertexSelector,
    vector::{Vector, VectorInt, View},
};
use std::{
    ffi::{CStr, c_void},
    fmt,
    mem::MaybeUninit,
    ptr,
};

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Zero-copy view of an optional integer slice.
fn opt_view(data: Option<&[i64]>) -> Option<View<'_, VectorInt>> {
    data.map(VectorInt::view)
}

/// Raw pointer of an optional view (null when absent).
fn opt_ptr<V>(view: &Option<View<'_, V>>) -> *const V {
    view.as_ref().map_or(ptr::null(), |v| v.as_ptr())
}

/// Runs `f` (the Rust side of a callback invoked by igraph) in a new level of
/// igraph's "finally" stack.
///
/// A user closure may itself call igraph functions. When one of those fails,
/// the error handler frees the objects of the current finally-stack level,
/// which without a new level would include the temporaries of the *outer*
/// igraph function that invoked the callback (a use after free; igraph 1.0.1
/// itself does not open a level around callbacks). The level is closed when
/// `f` returns.
///
/// [`catch_panic`] / [`catch_panic_or`] do not open a level themselves (as of
/// this writing), so this is the protection for the VF2 callbacks. Should
/// they start doing so, this level becomes redundant but stays harmless:
/// finally levels nest, and each one is closed by its own opener.
fn in_finally_level<T>(f: impl FnOnce() -> T) -> T {
    struct Exit;
    impl Drop for Exit {
        fn drop(&mut self) {
            unsafe { IGRAPH_FINALLY_EXIT() };
        }
    }
    unsafe { IGRAPH_FINALLY_ENTER() };
    let _exit = Exit;
    f()
}

fn check_len(what: &str, data: Option<&[i64]>, expected: usize) -> Result<()> {
    match data {
        Some(d) if d.len() != expected => Err(Error::invalid(format!(
            "{what} has length {}, expected {expected}",
            d.len()
        ))),
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// An isomorphism (or subgraph isomorphism) between two graphs, as a pair
/// of mutually inverse vertex maps.
///
/// For a full isomorphism both vectors are permutations. For a subgraph
/// isomorphism of `graph2` into `graph1`, `map21[v]` is the vertex of
/// `graph1` matched to vertex `v` of `graph2`, while `map12[u]` is the vertex
/// of `graph2` matched to `u`, or `-1` if `u` is not part of the match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsoMapping {
    /// Maps each vertex of the first graph to a vertex of the second one.
    pub map12: Vec<VertexId>,
    /// Maps each vertex of the second graph to a vertex of the first one.
    pub map21: Vec<VertexId>,
}

/// Statistics of a Bliss run (`igraph_bliss_info_t`).
///
/// Mostly useful to study the internal working of the algorithm, except for
/// [`group_size`](Self::group_size), the *exact* size of the automorphism
/// group, which may be astronomically large (e.g. `n!` for the complete
/// graph `K_n`) and is therefore given as a decimal string.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BlissInfo {
    /// Number of nodes in the search tree.
    pub nof_nodes: u64,
    /// Number of leaf nodes in the search tree.
    pub nof_leaf_nodes: u64,
    /// Number of bad nodes.
    pub nof_bad_nodes: u64,
    /// Number of canonical representative updates.
    pub nof_canupdates: u64,
    /// Number of generators of the automorphism group.
    pub nof_generators: u64,
    /// Maximum level of the search tree.
    pub max_level: u64,
    /// Size of the automorphism group, in base 10. It is empty when Bliss did
    /// not run to completion, e.g. in [`Graph::isomorphic_bliss`] when the
    /// two graphs have different vertex or edge counts.
    pub group_size: String,
}

impl BlissInfo {
    /// The group size parsed as a float (possibly rounded, or infinite for
    /// huge groups); `None` if [`group_size`](Self::group_size) is empty.
    pub fn group_size_f64(&self) -> Option<f64> {
        self.group_size.parse().ok()
    }

    /// The group size parsed as a `u128`; `None` if it is empty or too large.
    pub fn group_size_u128(&self) -> Option<u128> {
        self.group_size.parse().ok()
    }
}

/// Owner of a raw `igraph_bliss_info_t`, which frees the `group_size` string.
struct RawBlissInfo(igraph_bliss_info_t);

impl RawBlissInfo {
    fn new() -> Self {
        // All-zero is a valid "empty" info: counters at 0 and a null string.
        Self(unsafe { MaybeUninit::<igraph_bliss_info_t>::zeroed().assume_init() })
    }

    // `c_ulong` is `u64` on some platforms only: keep the casts portable.
    #[allow(clippy::unnecessary_cast)]
    fn to_info(&self) -> BlissInfo {
        let r = &self.0;
        let group_size = if r.group_size.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(r.group_size) }
                .to_string_lossy()
                .into_owned()
        };
        BlissInfo {
            nof_nodes: r.nof_nodes as u64,
            nof_leaf_nodes: r.nof_leaf_nodes as u64,
            nof_bad_nodes: r.nof_bad_nodes as u64,
            nof_canupdates: r.nof_canupdates as u64,
            nof_generators: r.nof_generators as u64,
            max_level: r.max_level as u64,
            group_size,
        }
    }
}

impl Drop for RawBlissInfo {
    fn drop(&mut self) {
        if !self.0.group_size.is_null() {
            unsafe { igraph_free(self.0.group_size.cast()) };
            self.0.group_size = ptr::null_mut();
        }
    }
}

crate::ffi_enum! {
    /// Splitting heuristics of Bliss (`igraph_bliss_sh_t`).
    ///
    /// They affect performance (and the particular generators returned by
    /// [`Graph::automorphism_group_bliss`]), never the correctness of the
    /// result. [`BlissSh::Fl`] is a good general-purpose default (the
    /// [`Default`] here), while [`BlissSh::Fsm`] is recommended for graphs with
    /// some combinatorial structure and is the default of the Bliss command
    /// line tool.
    #[derive(Default)]
    pub enum BlissSh: igraph_bliss_sh_t {
        /// First non-singleton cell.
        F = igraph_bliss_sh_t_IGRAPH_BLISS_F,
        /// First largest non-singleton cell (the default).
        #[default]
        Fl = igraph_bliss_sh_t_IGRAPH_BLISS_FL,
        /// First smallest non-singleton cell.
        Fs = igraph_bliss_sh_t_IGRAPH_BLISS_FS,
        /// First maximally non-trivially connected non-singleton cell.
        Fm = igraph_bliss_sh_t_IGRAPH_BLISS_FM,
        /// Largest maximally non-trivially connected non-singleton cell.
        Flm = igraph_bliss_sh_t_IGRAPH_BLISS_FLM,
        /// Smallest maximally non-trivially connected non-singleton cell.
        Fsm = igraph_bliss_sh_t_IGRAPH_BLISS_FSM,
    }
}

impl BlissSh {
    /// All the heuristics, in the order of the C enumeration.
    pub const ALL: [BlissSh; 6] = [Self::F, Self::Fl, Self::Fs, Self::Fm, Self::Flm, Self::Fsm];
}

/// Result of [`Graph::isomorphic_bliss`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlissIsomorphism {
    /// An isomorphism, or `None` if the graphs are not isomorphic.
    pub mapping: Option<IsoMapping>,
    /// Statistics of the canonization of the first graph.
    pub info1: BlissInfo,
    /// Statistics of the canonization of the second graph.
    pub info2: BlissInfo,
}

impl BlissIsomorphism {
    /// Whether the two graphs are isomorphic.
    pub fn is_isomorphic(&self) -> bool {
        self.mapping.is_some()
    }
}

/// Result of [`Graph::simplify_and_colorize`]: a colored simple graph that
/// encodes a multigraph.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorizedGraph {
    /// The simple graph (no multi-edges, no self-loops).
    pub graph: Graph,
    /// For each vertex, the number of self-loops it had in the input.
    pub vertex_color: Vec<i64>,
    /// For each edge of [`graph`](Self::graph), the number of parallel
    /// input edges it replaces.
    pub edge_color: Vec<i64>,
}

/// The dyad census of a graph, see [`Graph::dyad_census`].
///
/// `mutual + asymmetric + null == n (n - 1) / 2` for `n` vertices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DyadCensus {
    /// Pairs connected in both directions.
    pub mutual: f64,
    /// Pairs connected in exactly one direction.
    pub asymmetric: f64,
    /// Unconnected pairs.
    pub null: f64,
}

/// The triad census of a graph, see [`Graph::triad_census`].
///
/// Holds the number of vertex triples of each of the 16 types described by
/// Davis and Leinhardt's *MAN labels* ([`TriadCensus::NAMES`]): the digits
/// count **M**utual, **A**symmetric and **N**ull dyads of the triple, and the
/// letter distinguishes **D**own, **U**p, **C**yclic and **T**ransitive
/// variants. Index it by position (`census[3]`) or by label
/// (`census["021D"]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriadCensus {
    /// The 16 counts, in the order of [`TriadCensus::NAMES`].
    pub counts: [f64; 16],
}

impl TriadCensus {
    /// The MAN labels of the 16 triad types, in igraph's order:
    ///
    /// | idx | label | shape |
    /// |---|---|---|
    /// | 0 | `003` | empty |
    /// | 1 | `012` | A→B, C |
    /// | 2 | `102` | A↔B, C |
    /// | 3 | `021D` | A←B→C (out-star) |
    /// | 4 | `021U` | A→B←C (in-star) |
    /// | 5 | `021C` | A→B→C (directed line) |
    /// | 6 | `111D` | A↔B←C |
    /// | 7 | `111U` | A↔B→C |
    /// | 8 | `030T` | A→B←C, A→C |
    /// | 9 | `030C` | A←B←C, A→C (cycle) |
    /// | 10 | `201` | A↔B↔C |
    /// | 11 | `120D` | A←B→C, A↔C |
    /// | 12 | `120U` | A→B←C, A↔C |
    /// | 13 | `120C` | A→B→C, A↔C |
    /// | 14 | `210` | A→B↔C, A↔C |
    /// | 15 | `300` | complete |
    pub const NAMES: [&'static str; 16] = [
        "003", "012", "102", "021D", "021U", "021C", "111D", "111U", "030T", "030C", "201", "120D",
        "120U", "120C", "210", "300",
    ];

    /// The count of the triad type with the given MAN label, if valid.
    pub fn get(&self, label: &str) -> Option<f64> {
        Self::NAMES
            .iter()
            .position(|&n| n == label)
            .map(|i| self.counts[i])
    }

    /// Iterates over `(label, count)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, f64)> + '_ {
        Self::NAMES.iter().copied().zip(self.counts.iter().copied())
    }

    /// Total number of triples, `n (n-1) (n-2) / 6`.
    pub fn total(&self) -> f64 {
        self.counts.iter().sum()
    }
}

impl std::ops::Index<usize> for TriadCensus {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 {
        &self.counts[i]
    }
}

impl std::ops::Index<&str> for TriadCensus {
    type Output = f64;
    /// # Panics
    /// If `label` is not one of [`TriadCensus::NAMES`].
    fn index(&self, label: &str) -> &f64 {
        let i = Self::NAMES
            .iter()
            .position(|&n| n == label)
            .unwrap_or_else(|| panic!("unknown triad label {label:?}"));
        &self.counts[i]
    }
}

/// How [`Graph::motifs_randesu_estimate`] chooses the sample of vertices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MotifSample<'a> {
    /// Draw this many vertices uniformly at random (uses igraph's RNG).
    Random(usize),
    /// Use exactly these vertices.
    Vertices(&'a [VertexId]),
}

/// A graphlet basis with thresholds, see [`Graph::graphlets_candidate_basis`].
#[derive(Debug, Clone, PartialEq)]
pub struct GraphletBasis {
    /// The candidate cliques, as lists of vertex ids.
    pub cliques: Vec<Vec<VertexId>>,
    /// For each clique, the highest weight threshold at which it was found.
    pub thresholds: Vec<f64>,
}

/// A graphlet decomposition, see [`Graph::graphlets`].
#[derive(Debug, Clone, PartialEq)]
pub struct Graphlets {
    /// The graphlets (cliques), as lists of vertex ids, by decreasing weight.
    pub cliques: Vec<Vec<VertexId>>,
    /// The weight (`Mu`) of each graphlet.
    pub weights: Vec<f64>,
}

// ---------------------------------------------------------------------------
// VF2 configuration and callbacks
// ---------------------------------------------------------------------------

/// A vertex or edge compatibility predicate for VF2: it receives the id of a
/// vertex (edge) of the first graph and one of the second graph, and tells
/// whether they may be matched.
pub type CompatFn<'a> = Box<dyn FnMut(i64, i64) -> bool + 'a>;

/// Options of the VF2 functions: vertex and edge colors, and custom
/// compatibility predicates.
///
/// * With **vertex colors**, a vertex may only be matched to a vertex of the
///   same color; with **edge colors**, an edge only to an edge of the same
///   color. Colors are given for both graphs at once (igraph ignores the
///   colors, with a warning, when only one graph is colored).
/// * A **node (edge) compatibility closure** is called every time VF2 tries
///   to match vertex (edge) `i` of the first graph with vertex (edge) `j` of
///   the second, and can veto the match by returning `false`. If a closure
///   panics, the search is aborted and the panic is resumed when the
///   wrapper returns.
///
/// Build it with the builder methods and pass it as `&mut`, so that it
/// (and its closures) can be reused:
///
/// ```
/// use igraph::prelude::*;
/// use igraph::isomorphism::Vf2Options;
///
/// let path = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
/// // Colors pin the middle vertex: only the reflection 0 <-> 2 survives.
/// let colors = [0, 1, 0];
/// let mut opts = Vf2Options::new().with_vertex_colors(&colors, &colors);
/// assert_eq!(path.count_isomorphisms_vf2(&path, &mut opts).unwrap(), 2);
/// // A predicate forbidding fixed points leaves only the reflection.
/// let mut opts = Vf2Options::new().with_node_compat(|i, j| i != j || i == 1);
/// assert_eq!(path.get_isomorphisms_vf2(&path, &mut opts).unwrap(), vec![vec![2, 1, 0]]);
/// ```
#[derive(Default)]
pub struct Vf2Options<'a> {
    /// Vertex colors of the first graph.
    pub vertex_color1: Option<&'a [i64]>,
    /// Vertex colors of the second graph.
    pub vertex_color2: Option<&'a [i64]>,
    /// Edge colors of the first graph.
    pub edge_color1: Option<&'a [i64]>,
    /// Edge colors of the second graph.
    pub edge_color2: Option<&'a [i64]>,
    node_compat: Option<CompatFn<'a>>,
    edge_compat: Option<CompatFn<'a>>,
}

impl fmt::Debug for Vf2Options<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vf2Options")
            .field("vertex_color1", &self.vertex_color1)
            .field("vertex_color2", &self.vertex_color2)
            .field("edge_color1", &self.edge_color1)
            .field("edge_color2", &self.edge_color2)
            .field("node_compat", &self.node_compat.is_some())
            .field("edge_compat", &self.edge_compat.is_some())
            .finish()
    }
}

impl<'a> Vf2Options<'a> {
    /// No colors and no compatibility predicates: plain (sub)isomorphism.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the vertex colors of the first and second graph.
    pub fn with_vertex_colors(mut self, color1: &'a [i64], color2: &'a [i64]) -> Self {
        self.vertex_color1 = Some(color1);
        self.vertex_color2 = Some(color2);
        self
    }

    /// Sets the edge colors of the first and second graph.
    pub fn with_edge_colors(mut self, color1: &'a [i64], color2: &'a [i64]) -> Self {
        self.edge_color1 = Some(color1);
        self.edge_color2 = Some(color2);
        self
    }

    /// Sets the vertex compatibility predicate `f(v1, v2)`, where `v1` is a
    /// vertex of the first graph and `v2` one of the second graph.
    pub fn with_node_compat(mut self, f: impl FnMut(VertexId, VertexId) -> bool + 'a) -> Self {
        self.node_compat = Some(Box::new(f));
        self
    }

    /// Sets the edge compatibility predicate `f(e1, e2)`, where `e1` is an
    /// edge of the first graph and `e2` one of the second graph.
    pub fn with_edge_compat(mut self, f: impl FnMut(i64, i64) -> bool + 'a) -> Self {
        self.edge_compat = Some(Box::new(f));
        self
    }

    /// Sets the vertex colors; use
    /// [`with_vertex_colors`](Self::with_vertex_colors) instead.
    #[deprecated(note = "use with_vertex_colors")]
    pub fn vertex_colors(self, color1: &'a [i64], color2: &'a [i64]) -> Self {
        self.with_vertex_colors(color1, color2)
    }

    /// Sets the edge colors; use
    /// [`with_edge_colors`](Self::with_edge_colors) instead.
    #[deprecated(note = "use with_edge_colors")]
    pub fn edge_colors(self, color1: &'a [i64], color2: &'a [i64]) -> Self {
        self.with_edge_colors(color1, color2)
    }

    /// Sets the vertex compatibility predicate; use
    /// [`with_node_compat`](Self::with_node_compat) instead.
    #[deprecated(note = "use with_node_compat")]
    pub fn node_compat(self, f: impl FnMut(VertexId, VertexId) -> bool + 'a) -> Self {
        self.with_node_compat(f)
    }

    /// Sets the edge compatibility predicate; use
    /// [`with_edge_compat`](Self::with_edge_compat) instead.
    #[deprecated(note = "use with_edge_compat")]
    pub fn edge_compat(self, f: impl FnMut(i64, i64) -> bool + 'a) -> Self {
        self.with_edge_compat(f)
    }
}

/// A closure receiving `(map12, map21)` for each mapping found by VF2.
type IsoHandler<'h> = dyn FnMut(&[VertexId], &[VertexId]) -> bool + 'h;

/// The state shared by all the VF2 trampolines through the `arg` pointer.
struct Vf2State<'s, 'a, 'h> {
    node: Option<&'s mut CompatFn<'a>>,
    edge: Option<&'s mut CompatFn<'a>>,
    handler: Option<&'s mut IsoHandler<'h>>,
    panicked: bool,
}

unsafe extern "C" fn node_compat_trampoline(
    _graph1: *const igraph_t,
    _graph2: *const igraph_t,
    g1_num: igraph_int_t,
    g2_num: igraph_int_t,
    arg: *mut c_void,
) -> igraph_bool_t {
    let state = unsafe { &mut *(arg as *mut Vf2State<'_, '_, '_>) };
    if state.panicked {
        return false;
    }
    let Some(f) = state.node.as_mut() else {
        return true;
    };
    match in_finally_level(|| catch_panic_or(None, || Some(f(g1_num, g2_num)))) {
        Some(ok) => ok,
        None => {
            state.panicked = true;
            false
        }
    }
}

unsafe extern "C" fn edge_compat_trampoline(
    _graph1: *const igraph_t,
    _graph2: *const igraph_t,
    g1_num: igraph_int_t,
    g2_num: igraph_int_t,
    arg: *mut c_void,
) -> igraph_bool_t {
    let state = unsafe { &mut *(arg as *mut Vf2State<'_, '_, '_>) };
    if state.panicked {
        return false;
    }
    let Some(f) = state.edge.as_mut() else {
        return true;
    };
    match in_finally_level(|| catch_panic_or(None, || Some(f(g1_num, g2_num)))) {
        Some(ok) => ok,
        None => {
            state.panicked = true;
            false
        }
    }
}

unsafe extern "C" fn iso_handler_trampoline(
    map12: *const igraph_vector_int_t,
    map21: *const igraph_vector_int_t,
    arg: *mut c_void,
) -> igraph_error_t {
    let state = unsafe { &mut *(arg as *mut Vf2State<'_, '_, '_>) };
    if state.panicked {
        return igraph_error_type_t_IGRAPH_INTERRUPTED;
    }
    let Some(h) = state.handler.as_mut() else {
        return igraph_error_type_t_IGRAPH_SUCCESS;
    };
    let (m12, m21) = unsafe { ((*map12).as_slice(), (*map21).as_slice()) };
    in_finally_level(|| {
        catch_panic(|| {
            if h(m12, m21) {
                igraph_error_type_t_IGRAPH_SUCCESS
            } else {
                igraph_error_type_t_IGRAPH_STOP
            }
        })
    })
}

/// Raw arguments of a VF2 call, valid during [`with_vf2`]'s closure.
struct Vf2Raw {
    vc1: *const igraph_vector_int_t,
    vc2: *const igraph_vector_int_t,
    ec1: *const igraph_vector_int_t,
    ec2: *const igraph_vector_int_t,
    node: igraph_isocompat_t,
    edge: igraph_isocompat_t,
    handler: igraph_isohandler_t,
    arg: *mut c_void,
}

/// Prepares the raw VF2 arguments (color views, trampolines and state) and
/// runs `f` with them, converting the result code.
fn with_vf2(
    graph1: &Graph,
    graph2: &Graph,
    opts: &mut Vf2Options<'_>,
    handler: Option<&mut IsoHandler<'_>>,
    f: impl FnOnce(&Vf2Raw) -> igraph_error_t,
) -> Result<()> {
    check_len("vertex_color1", opts.vertex_color1, graph1.vcount())?;
    check_len("vertex_color2", opts.vertex_color2, graph2.vcount())?;
    check_len("edge_color1", opts.edge_color1, graph1.ecount())?;
    check_len("edge_color2", opts.edge_color2, graph2.ecount())?;
    let vc1 = opt_view(opts.vertex_color1);
    let vc2 = opt_view(opts.vertex_color2);
    let ec1 = opt_view(opts.edge_color1);
    let ec2 = opt_view(opts.edge_color2);
    let has_node = opts.node_compat.is_some();
    let has_edge = opts.edge_compat.is_some();
    let has_handler = handler.is_some();
    let mut state = Vf2State {
        node: opts.node_compat.as_mut(),
        edge: opts.edge_compat.as_mut(),
        handler,
        panicked: false,
    };
    let raw = Vf2Raw {
        vc1: opt_ptr(&vc1),
        vc2: opt_ptr(&vc2),
        ec1: opt_ptr(&ec1),
        ec2: opt_ptr(&ec2),
        node: if has_node {
            Some(node_compat_trampoline)
        } else {
            None
        },
        edge: if has_edge {
            Some(edge_compat_trampoline)
        } else {
            None
        },
        handler: if has_handler {
            Some(iso_handler_trampoline)
        } else {
            None
        },
        arg: (&mut state as *mut Vf2State<'_, '_, '_>).cast(),
    };
    igraph_call!(f(&raw))
}

/// First mapping found by LAD (if any), and all the mappings (if requested).
type LadResult = (Option<Vec<VertexId>>, Vec<Vec<VertexId>>);

// ---------------------------------------------------------------------------
// Motif callback
// ---------------------------------------------------------------------------

type MotifHandler<'h> = dyn FnMut(&[VertexId], usize) -> bool + 'h;

unsafe extern "C" fn motif_trampoline(
    _graph: *const igraph_t,
    vids: *const igraph_vector_int_t,
    isoclass: igraph_int_t,
    extra: *mut c_void,
) -> igraph_error_t {
    let f = unsafe { &mut *(extra as *mut &mut MotifHandler<'_>) };
    let vids = unsafe { (*vids).as_slice() };
    in_finally_level(|| {
        catch_panic(|| {
            if f(vids, isoclass as usize) {
                igraph_error_type_t_IGRAPH_SUCCESS
            } else {
                igraph_error_type_t_IGRAPH_STOP
            }
        })
    })
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Inverts a permutation of `0..n` (`igraph_invert_permutation`).
///
/// The result `inv` satisfies `inv[perm[i]] == i` for all `i`. Handy to turn
/// the labeling returned by [`Graph::canonical_permutation`] ("which vertex
/// goes to position `i`") into "which position does vertex `v` go to", or to
/// turn a `map21` of [`IsoMapping`] into a `map12`.
///
/// Binds [`igraph_invert_permutation`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_invert_permutation).
///
/// # Errors
/// [`ErrorKind::InvalidValue`] if `perm` is
/// not a permutation of `0..perm.len()`.
///
/// # Examples
/// ```
/// use igraph::isomorphism::invert_permutation;
/// assert_eq!(invert_permutation(&[2, 0, 1]).unwrap(), vec![1, 2, 0]);
/// assert!(invert_permutation(&[0, 0, 1]).is_err());
/// ```
pub fn invert_permutation(perm: &[i64]) -> Result<Vec<i64>> {
    let p = VectorInt::view(perm);
    let mut res = VectorInt::new();
    igraph_call!(igraph_invert_permutation(p.as_ptr(), &mut res))?;
    Ok(res.into())
}

/// The number of unlabeled simple graphs on `n` vertices (`igraph_graph_count`).
///
/// This is the number of isomorphism classes, i.e. the length of the
/// histogram returned by [`Graph::motifs_randesu`] and one more than the
/// largest valid [`Graph::isoclass`]. See OEIS [A000088] (undirected) and
/// [A000273] (directed). Time complexity: O(1).
///
/// Binds [`igraph_graph_count`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_graph_count).
///
/// [A000088]: https://oeis.org/A000088
/// [A000273]: https://oeis.org/A000273
///
/// # Errors
/// [`ErrorKind::Overflow`] when the count does not
/// fit in an `i64` (`n > 14` undirected, `n > 9` directed, on 64-bit
/// platforms).
///
/// # Examples
/// ```
/// use igraph::isomorphism::graph_count;
/// assert_eq!(graph_count(4, false).unwrap(), 11);
/// assert_eq!(graph_count(3, true).unwrap(), 16);
/// ```
pub fn graph_count(n: usize, directed: bool) -> Result<usize> {
    let n = igraph_int_t::try_from(n)
        .map_err(|_| Error::new(ErrorKind::Overflow, format!("graph size {n} is too large")))?;
    let mut count = 0;
    igraph_call!(igraph_graph_count(n, directed, &mut count))?;
    Ok(count as usize)
}

// ---------------------------------------------------------------------------
// Graph methods
// ---------------------------------------------------------------------------

impl igraph_t {
    // ----- generic --------------------------------------------------------

    /// Whether `self` and `other` are isomorphic (`igraph_isomorphic`).
    ///
    /// The algorithm is chosen automatically:
    /// 1. a directed and an undirected graph are an error;
    /// 2. if either graph has multi-edges, both are
    ///    [simplified and colorized](Self::simplify_and_colorize) and
    ///    compared with VF2;
    /// 3. graphs with different vertex or edge counts are not isomorphic;
    /// 4. small loop-free graphs supported by [`isoclass`](Self::isoclass)
    ///    (directed with 3–4 vertices, undirected with 3–6) use precomputed
    ///    O(1) data;
    /// 5. otherwise Bliss is used.
    ///
    /// Use [`isomorphic_vf2`](Self::isomorphic_vf2) or
    /// [`isomorphic_bliss`](Self::isomorphic_bliss) to obtain a mapping.
    /// Time complexity: exponential in the worst case.
    ///
    /// See also [`Graph::is_same_graph`], which compares *labeled* graphs
    /// (identical vertex ids and edges), and
    /// [`canonical_form`](Self::canonical_form) to test many graphs against
    /// each other.
    ///
    /// Binds [`igraph_isomorphic`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_isomorphic).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] when one
    /// graph is directed and the other is not.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let a = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// let b = Graph::from_edges(&[(0, 2), (2, 1)], 3, false).unwrap();
    /// let c = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, false).unwrap();
    /// assert!(a.isomorphic(&b).unwrap());
    /// assert!(!a.is_same_graph(&b).unwrap()); // different labels
    /// assert!(!a.isomorphic(&c).unwrap());
    /// // The Petersen graph is the generalized Petersen graph GP(5, 2).
    /// let p = Graph::famous("Petersen").unwrap();
    /// assert!(p.isomorphic(&Graph::generalized_petersen(5, 2).unwrap()).unwrap());
    /// ```
    pub fn isomorphic(&self, other: &Graph) -> Result<bool> {
        let mut iso = false;
        igraph_call!(igraph_isomorphic(self, other, &mut iso))?;
        Ok(iso)
    }

    /// Whether `pattern` is isomorphic to a (not necessarily induced)
    /// subgraph of `self` (`igraph_subisomorphic`).
    ///
    /// `self` is the larger graph. Currently this always uses VF2, and so
    /// does not support non-simple graphs (self-loops are an error,
    /// multi-edges give wrong results). Time complexity: exponential.
    ///
    /// See also [`subisomorphic_lad`](Self::subisomorphic_lad) (note its
    /// reversed argument order), which is often faster and can look for
    /// induced subgraphs, and [`Graph::clique_number`] for the special case
    /// of complete patterns.
    ///
    /// Binds [`igraph_subisomorphic`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_subisomorphic).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] when the graphs differ in directedness or
    /// have self-loops.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let square = Graph::ring(4, false, false, true).unwrap();
    /// let path3 = Graph::ring(3, false, false, false).unwrap();
    /// let triangle = Graph::full(3, false, false).unwrap();
    /// assert!(square.subisomorphic(&path3).unwrap());
    /// assert!(!square.subisomorphic(&triangle).unwrap());
    /// ```
    pub fn subisomorphic(&self, pattern: &Graph) -> Result<bool> {
        let mut iso = false;
        igraph_call!(igraph_subisomorphic(self, pattern, &mut iso))?;
        Ok(iso)
    }

    /// Turns a multigraph into a colored simple graph
    /// (`igraph_simplify_and_colorize`).
    ///
    /// Multi-edges are merged into single edges whose *color* is their
    /// multiplicity, and self-loops are removed, the *color* of each vertex
    /// being its number of self-loops. The result is suitable for the VF2
    /// functions (which only support simple graphs but honour colors): two
    /// multigraphs are isomorphic iff their colorized versions are
    /// isomorphic as colored graphs.
    ///
    /// See also [`Graph::simplify`], which removes multi-edges and self-loops
    /// in place without recording them, and [`Graph::count_multiple`].
    ///
    /// Binds [`igraph_simplify_and_colorize`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_simplify_and_colorize).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A double edge 0-1 plus a loop on 1.
    /// let g = Graph::from_edges(&[(0, 1), (0, 1), (1, 1)], 2, false).unwrap();
    /// let c = g.simplify_and_colorize().unwrap();
    /// assert_eq!(c.graph.ecount(), 1);
    /// assert_eq!(c.vertex_color, vec![0, 1]);
    /// assert_eq!(c.edge_color, vec![2]);
    /// ```
    pub fn simplify_and_colorize(&self) -> Result<ColorizedGraph> {
        let mut vertex_color = VectorInt::new();
        let mut edge_color = VectorInt::new();
        let graph = Graph::init_with(|res| unsafe {
            igraph_simplify_and_colorize(self, res, &mut vertex_color, &mut edge_color)
        })?;
        Ok(ColorizedGraph {
            graph,
            vertex_color: vertex_color.into(),
            edge_color: edge_color.into(),
        })
    }

    // ----- VF2 --------------------------------------------------------------

    /// Isomorphism test with VF2, returning a mapping
    /// (`igraph_isomorphic_vf2`).
    ///
    /// Returns `Some(mapping)` if `self` and `other` are isomorphic under the
    /// colors and compatibility predicates of `opts` (see [`Vf2Options`]),
    /// `None` otherwise. Both graphs must be simple: self-loops are
    /// rejected with an error, multi-edges are *not* detected and give
    /// wrong results (use [`simplify_and_colorize`](Self::simplify_and_colorize)
    /// first). Use [`subisomorphic_vf2`](Self::subisomorphic_vf2) for subgraphs.
    /// Time complexity: exponential.
    ///
    /// Binds [`igraph_isomorphic_vf2`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_isomorphic_vf2).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] on directedness mismatch, self-loops, or
    /// color vectors of the wrong length.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::isomorphism::Vf2Options;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    /// let h = Graph::from_edges(&[(2, 0), (1, 2)], 3, true).unwrap();
    /// let m = g.isomorphic_vf2(&h, &mut Vf2Options::new()).unwrap().unwrap();
    /// assert_eq!(m.map12, vec![1, 2, 0]); // 0->1->2 becomes 1->2->0
    /// assert_eq!(m.map21, vec![2, 0, 1]);
    /// ```
    pub fn isomorphic_vf2(
        &self,
        other: &Graph,
        opts: &mut Vf2Options<'_>,
    ) -> Result<Option<IsoMapping>> {
        let mut iso = false;
        let mut map12 = VectorInt::new();
        let mut map21 = VectorInt::new();
        with_vf2(self, other, opts, None, |r| unsafe {
            igraph_isomorphic_vf2(
                self, other, r.vc1, r.vc2, r.ec1, r.ec2, &mut iso, &mut map12, &mut map21, r.node,
                r.edge, r.arg,
            )
        })?;
        Ok(iso.then(|| IsoMapping {
            map12: map12.into(),
            map21: map21.into(),
        }))
    }

    /// Number of isomorphisms between `self` and `other` with VF2
    /// (`igraph_count_isomorphisms_vf2`).
    ///
    /// With `other == self` this is the number of automorphisms (but
    /// [`count_automorphisms`](Self::count_automorphisms) is much faster).
    /// Time complexity: exponential.
    ///
    /// Binds [`igraph_count_isomorphisms_vf2`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_count_isomorphisms_vf2).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::isomorphism::Vf2Options;
    /// let triangle = Graph::full(3, false, false).unwrap();
    /// assert_eq!(triangle.count_isomorphisms_vf2(&triangle, &mut Vf2Options::new()).unwrap(), 6);
    /// ```
    pub fn count_isomorphisms_vf2(
        &self,
        other: &Graph,
        opts: &mut Vf2Options<'_>,
    ) -> Result<usize> {
        let mut count = 0;
        with_vf2(self, other, opts, None, |r| unsafe {
            igraph_count_isomorphisms_vf2(
                self, other, r.vc1, r.vc2, r.ec1, r.ec2, &mut count, r.node, r.edge, r.arg,
            )
        })?;
        Ok(count as usize)
    }

    /// All the isomorphisms between `self` and `other` with VF2
    /// (`igraph_get_isomorphisms_vf2`).
    ///
    /// Each mapping is given as `map21`, i.e. `m[v]` is the vertex of `self`
    /// matched to vertex `v` of `other`. The result is empty if the graphs
    /// are not isomorphic. Time complexity: exponential.
    ///
    /// Binds [`igraph_get_isomorphisms_vf2`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_get_isomorphisms_vf2).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::isomorphism::Vf2Options;
    /// let p = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// let mut maps = p.get_isomorphisms_vf2(&p, &mut Vf2Options::new()).unwrap();
    /// maps.sort();
    /// assert_eq!(maps, vec![vec![0, 1, 2], vec![2, 1, 0]]);
    /// ```
    pub fn get_isomorphisms_vf2(
        &self,
        other: &Graph,
        opts: &mut Vf2Options<'_>,
    ) -> Result<Vec<Vec<VertexId>>> {
        let mut maps = VectorIntList::new();
        with_vf2(self, other, opts, None, |r| unsafe {
            igraph_get_isomorphisms_vf2(
                self, other, r.vc1, r.vc2, r.ec1, r.ec2, &mut maps, r.node, r.edge, r.arg,
            )
        })?;
        Ok(maps.to_vecs())
    }

    /// Streams the isomorphisms between `self` and `other` to a closure
    /// (`igraph_get_isomorphisms_vf2_callback`).
    ///
    /// `handler(map12, map21)` is called for each isomorphism found; return
    /// `true` to continue the search or `false` to stop it (which is not an
    /// error). A panic in the handler aborts the search and is resumed when
    /// this function returns. Time complexity: exponential.
    ///
    /// Binds [`igraph_get_isomorphisms_vf2_callback`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_get_isomorphisms_vf2_callback).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::isomorphism::Vf2Options;
    /// let k4 = Graph::full(4, false, false).unwrap();
    /// // Take the first three automorphisms of K4 only (out of 24).
    /// let mut found = vec![];
    /// k4.get_isomorphisms_vf2_callback(&k4, &mut Vf2Options::new(), |m12, _m21| {
    ///     found.push(m12.to_vec());
    ///     found.len() < 3
    /// })
    /// .unwrap();
    /// assert_eq!(found.len(), 3);
    /// ```
    pub fn get_isomorphisms_vf2_callback(
        &self,
        other: &Graph,
        opts: &mut Vf2Options<'_>,
        mut handler: impl FnMut(&[VertexId], &[VertexId]) -> bool,
    ) -> Result<()> {
        with_vf2(self, other, opts, Some(&mut handler), |r| unsafe {
            igraph_get_isomorphisms_vf2_callback(
                self,
                other,
                r.vc1,
                r.vc2,
                r.ec1,
                r.ec2,
                ptr::null_mut(),
                ptr::null_mut(),
                r.handler,
                r.node,
                r.edge,
                r.arg,
            )
        })
    }

    /// Subgraph isomorphism test with VF2, returning a mapping
    /// (`igraph_subisomorphic_vf2`).
    ///
    /// Decides whether `pattern` (the smaller graph) is isomorphic to a
    /// subgraph of `self` (the larger graph); the subgraph need not be
    /// induced. Returns `Some(mapping)` with `mapping.map21[v]` the vertex of
    /// `self` matched to vertex `v` of `pattern`, and `mapping.map12[u]` the
    /// vertex of `pattern` matched to `u` (or `-1`). Time complexity:
    /// exponential.
    ///
    /// Binds [`igraph_subisomorphic_vf2`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_subisomorphic_vf2).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::isomorphism::Vf2Options;
    /// let big = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false).unwrap();
    /// let triangle = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, false).unwrap();
    /// let m = big.subisomorphic_vf2(&triangle, &mut Vf2Options::new()).unwrap().unwrap();
    /// let mut hit = m.map21.clone();
    /// hit.sort();
    /// assert_eq!(hit, vec![0, 1, 2]);
    /// assert_eq!(m.map12[3], -1); // vertex 3 is not in the match
    /// ```
    pub fn subisomorphic_vf2(
        &self,
        pattern: &Graph,
        opts: &mut Vf2Options<'_>,
    ) -> Result<Option<IsoMapping>> {
        let mut iso = false;
        let mut map12 = VectorInt::new();
        let mut map21 = VectorInt::new();
        with_vf2(self, pattern, opts, None, |r| unsafe {
            igraph_subisomorphic_vf2(
                self, pattern, r.vc1, r.vc2, r.ec1, r.ec2, &mut iso, &mut map12, &mut map21,
                r.node, r.edge, r.arg,
            )
        })?;
        Ok(iso.then(|| IsoMapping {
            map12: map12.into(),
            map21: map21.into(),
        }))
    }

    /// Number of subgraph isomorphisms from `pattern` into `self` with VF2
    /// (`igraph_count_subisomorphisms_vf2`).
    ///
    /// Every embedding is counted, so each copy of `pattern` in `self` is
    /// counted as many times as `pattern` has automorphisms. Time
    /// complexity: exponential.
    ///
    /// Binds [`igraph_count_subisomorphisms_vf2`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_count_subisomorphisms_vf2).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::isomorphism::Vf2Options;
    /// let k4 = Graph::full(4, false, false).unwrap();
    /// let triangle = Graph::full(3, false, false).unwrap();
    /// // 4 triangles in K4, each with 6 automorphisms.
    /// assert_eq!(k4.count_subisomorphisms_vf2(&triangle, &mut Vf2Options::new()).unwrap(), 24);
    /// ```
    pub fn count_subisomorphisms_vf2(
        &self,
        pattern: &Graph,
        opts: &mut Vf2Options<'_>,
    ) -> Result<usize> {
        let mut count = 0;
        with_vf2(self, pattern, opts, None, |r| unsafe {
            igraph_count_subisomorphisms_vf2(
                self, pattern, r.vc1, r.vc2, r.ec1, r.ec2, &mut count, r.node, r.edge, r.arg,
            )
        })?;
        Ok(count as usize)
    }

    /// All the subgraph isomorphisms from `pattern` into `self` with VF2
    /// (`igraph_get_subisomorphisms_vf2`).
    ///
    /// Each mapping `m` is a `map21`: `m[v]` is the vertex of `self` matched
    /// to vertex `v` of `pattern`. Time complexity: exponential.
    ///
    /// Binds [`igraph_get_subisomorphisms_vf2`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_get_subisomorphisms_vf2).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::isomorphism::Vf2Options;
    /// let star = Graph::star(4, StarMode::Undirected, 0).unwrap();
    /// let edge = Graph::full(2, false, false).unwrap();
    /// let maps = star.get_subisomorphisms_vf2(&edge, &mut Vf2Options::new()).unwrap();
    /// assert_eq!(maps.len(), 6); // 3 edges x 2 orientations
    /// ```
    pub fn get_subisomorphisms_vf2(
        &self,
        pattern: &Graph,
        opts: &mut Vf2Options<'_>,
    ) -> Result<Vec<Vec<VertexId>>> {
        let mut maps = VectorIntList::new();
        with_vf2(self, pattern, opts, None, |r| unsafe {
            igraph_get_subisomorphisms_vf2(
                self, pattern, r.vc1, r.vc2, r.ec1, r.ec2, &mut maps, r.node, r.edge, r.arg,
            )
        })?;
        Ok(maps.to_vecs())
    }

    /// Streams the subgraph isomorphisms from `pattern` into `self` to a
    /// closure (`igraph_get_subisomorphisms_vf2_callback`).
    ///
    /// `handler(map12, map21)` is called for each embedding found (see
    /// [`IsoMapping`] for the meaning of the maps); return `true` to go on or
    /// `false` to stop the search early. A panic in the handler aborts the
    /// search and is resumed when this function returns. Time complexity:
    /// exponential.
    ///
    /// Binds [`igraph_get_subisomorphisms_vf2_callback`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_get_subisomorphisms_vf2_callback).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::isomorphism::Vf2Options;
    /// let c5 = Graph::ring(5, false, false, true).unwrap();
    /// let p3 = Graph::ring(3, false, false, false).unwrap();
    /// // Collect the middle vertices of the 2-paths of the pentagon.
    /// let mut centers = std::collections::BTreeSet::new();
    /// c5.get_subisomorphisms_vf2_callback(&p3, &mut Vf2Options::new(), |_, m21| {
    ///     centers.insert(m21[1]);
    ///     true
    /// })
    /// .unwrap();
    /// assert_eq!(centers.len(), 5);
    /// ```
    pub fn get_subisomorphisms_vf2_callback(
        &self,
        pattern: &Graph,
        opts: &mut Vf2Options<'_>,
        mut handler: impl FnMut(&[VertexId], &[VertexId]) -> bool,
    ) -> Result<()> {
        with_vf2(self, pattern, opts, Some(&mut handler), |r| unsafe {
            igraph_get_subisomorphisms_vf2_callback(
                self,
                pattern,
                r.vc1,
                r.vc2,
                r.ec1,
                r.ec2,
                ptr::null_mut(),
                ptr::null_mut(),
                r.handler,
                r.node,
                r.edge,
                r.arg,
            )
        })
    }

    // ----- LAD --------------------------------------------------------------

    fn lad_raw(
        &self,
        target: &Graph,
        domains: Option<&[Vec<VertexId>]>,
        induced: bool,
        all: bool,
    ) -> Result<LadResult> {
        if let Some(d) = domains
            && d.len() != self.vcount()
        {
            return Err(Error::invalid(format!(
                "there are {} domains but the pattern has {} vertices",
                d.len(),
                self.vcount()
            )));
        }
        // igraph 1.0.0 and 1.0.1 do not validate the domain entries: an
        // out-of-range id would be written out of bounds of an internal bitset.
        if let Some(d) = domains {
            let n = target.vcount() as i64;
            if let Some(&bad) = d.iter().flatten().find(|&&v| !(0..n).contains(&v)) {
                return Err(Error::new(
                    ErrorKind::InvalidVertexId,
                    format!("domain vertex {bad} is not a vertex of the target graph (0..{n})"),
                ));
            }
        }
        let domains: Option<VectorIntList> = domains.map(VectorIntList::from);
        let dp = domains
            .as_ref()
            .map_or(ptr::null(), |d| d as *const VectorIntList);
        let mut iso = false;
        let mut map = VectorInt::new();
        let mut maps = VectorIntList::new();
        let maps_ptr: *mut VectorIntList = if all { &mut maps } else { ptr::null_mut() };
        igraph_call!(igraph_subisomorphic_lad(
            self, target, dp, &mut iso, &mut map, maps_ptr, induced
        ))?;
        Ok((iso.then(|| map.into()), maps.to_vecs()))
    }

    /// Subgraph isomorphism with the LAD algorithm (`igraph_subisomorphic_lad`).
    ///
    /// **Note the argument order**, which follows the C library: `self` is
    /// the (smaller) *pattern* and `target` the (larger) graph searched.
    /// Returns `Some(map)` with `map[v]` the vertex of `target` matched to
    /// vertex `v` of the pattern, or `None` if the pattern does not occur.
    ///
    /// * `domains`: optionally, for each pattern vertex, the list of target
    ///   vertices it may be matched to (this is how LAD implements vertex
    ///   colors); it must have one entry per pattern vertex.
    /// * `induced`: whether to look for *induced* subgraphs only (then
    ///   non-edges of the pattern must map to non-edges).
    ///
    /// Works for directed and undirected graphs without multi-edges.
    /// Time complexity: exponential.
    ///
    /// Binds [`igraph_subisomorphic_lad`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_subisomorphic_lad).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `domains` does not have one entry per
    /// pattern vertex or the graphs differ in directedness or have
    /// multi-edges, and [`ErrorKind::InvalidVertexId`] if a domain contains
    /// an id that is not a vertex of `target`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let square = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], 4, false).unwrap();
    /// let path3 = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// let map = path3.subisomorphic_lad(&square, None, true).unwrap().unwrap();
    /// assert_eq!(map.len(), 3);
    /// // Force the pattern's middle vertex onto vertex 2 of the square.
    /// let domains = vec![vec![0, 1, 2, 3], vec![2], vec![0, 1, 2, 3]];
    /// let map = path3.subisomorphic_lad(&square, Some(&domains), true).unwrap().unwrap();
    /// assert_eq!(map[1], 2);
    /// ```
    pub fn subisomorphic_lad(
        &self,
        target: &Graph,
        domains: Option<&[Vec<VertexId>]>,
        induced: bool,
    ) -> Result<Option<Vec<VertexId>>> {
        Ok(self.lad_raw(target, domains, induced, false)?.0)
    }

    /// All the subgraph isomorphisms of the pattern `self` into `target`
    /// with the LAD algorithm (`igraph_subisomorphic_lad` with `maps`).
    ///
    /// Same arguments as [`subisomorphic_lad`](Self::subisomorphic_lad);
    /// returns every mapping (pattern vertex → target vertex).
    ///
    /// Binds [`igraph_subisomorphic_lad`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_subisomorphic_lad).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let square = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], 4, false).unwrap();
    /// let path3 = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// // 4 induced 2-paths, each traversed in 2 directions.
    /// assert_eq!(path3.get_subisomorphisms_lad(&square, None, true).unwrap().len(), 8);
    /// ```
    pub fn get_subisomorphisms_lad(
        &self,
        target: &Graph,
        domains: Option<&[Vec<VertexId>]>,
        induced: bool,
    ) -> Result<Vec<Vec<VertexId>>> {
        Ok(self.lad_raw(target, domains, induced, true)?.1)
    }

    // ----- Bliss -------------------------------------------------------------

    /// Isomorphism test with Bliss, with mapping and statistics
    /// (`igraph_isomorphic_bliss`).
    ///
    /// Both graphs are brought into canonical form with the splitting
    /// heuristic `sh` and compared. `colors1` and `colors2` are optional
    /// vertex colors (a colored vertex may only map to a vertex of the same
    /// color; if only one graph is colored, colors are ignored with a
    /// warning). Multi-edges are not supported (the result would be wrong),
    /// self-loops are. Time complexity: exponential, fast in practice.
    ///
    /// Binds [`igraph_isomorphic_bliss`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_isomorphic_bliss).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::isomorphism::BlissSh;
    /// let a = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let b = Graph::from_edges(&[(3, 1), (1, 0), (0, 2)], 4, false).unwrap();
    /// let res = a.isomorphic_bliss(&b, None, None, BlissSh::Fl).unwrap();
    /// assert!(res.is_isomorphic());
    /// assert_eq!(res.info1.group_size, "2");
    /// ```
    pub fn isomorphic_bliss(
        &self,
        other: &Graph,
        colors1: Option<&[i64]>,
        colors2: Option<&[i64]>,
        sh: BlissSh,
    ) -> Result<BlissIsomorphism> {
        check_len("colors1", colors1, self.vcount())?;
        check_len("colors2", colors2, other.vcount())?;
        let c1 = opt_view(colors1);
        let c2 = opt_view(colors2);
        let mut iso = false;
        let mut map12 = VectorInt::new();
        let mut map21 = VectorInt::new();
        let mut info1 = RawBlissInfo::new();
        let mut info2 = RawBlissInfo::new();
        igraph_call!(igraph_isomorphic_bliss(
            self,
            other,
            opt_ptr(&c1),
            opt_ptr(&c2),
            &mut iso,
            &mut map12,
            &mut map21,
            sh.into(),
            &mut info1.0,
            &mut info2.0
        ))?;
        Ok(BlissIsomorphism {
            mapping: iso.then(|| IsoMapping {
                map12: map12.into(),
                map21: map21.into(),
            }),
            info1: info1.to_info(),
            info2: info2.to_info(),
        })
    }

    /// The canonical labeling of the graph (`igraph_canonical_permutation`).
    ///
    /// Returns `labeling`, where `labeling[i]` is the vertex of `self` that
    /// becomes vertex `i` of the canonical form (since igraph 1.0; this is
    /// the convention of `igraph_permute_vertices`, and the inverse of the
    /// convention of igraph 0.10). Use [`invert_permutation`] to get, for
    /// each vertex, its position in the canonical form. Relabeling two graphs
    /// by their canonical labelings yields *identical* graphs if and only if
    /// the graphs are isomorphic (see [`canonical_form`](Self::canonical_form)).
    /// `colors` optionally assigns vertex colors, which must be preserved;
    /// colored graphs are isomorphic if and only if, in addition, their
    /// colors relabeled the same way (`colors[labeling[i]]`) are equal.
    /// Multi-edges are not
    /// supported. Uses Bliss with sensible defaults; see
    /// [`canonical_permutation_bliss`](Self::canonical_permutation_bliss) to
    /// tune it.
    ///
    /// Binds [`igraph_canonical_permutation`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_canonical_permutation).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::star(4, StarMode::Undirected, 0).unwrap();
    /// let mut sorted = g.canonical_permutation(None).unwrap();
    /// sorted.sort();
    /// assert_eq!(sorted, vec![0, 1, 2, 3]); // it is a permutation
    ///
    /// // The canonical form is the graph relabeled by the labeling...
    /// let labeling = g.canonical_permutation(None).unwrap();
    /// let relabeled = g.permute_vertices(&labeling).unwrap();
    /// assert!(relabeled.isomorphic(&g.canonical_form(None).unwrap()).unwrap());
    ///
    /// // ... built by hand: vertex v goes to position pos[v].
    /// let pos = igraph::isomorphism::invert_permutation(&labeling).unwrap();
    /// let mut edges: Vec<(i64, i64)> = g
    ///     .edge_list()
    ///     .into_iter()
    ///     .map(|(a, b)| {
    ///         let (a, b) = (pos[a as usize], pos[b as usize]);
    ///         (a.min(b), a.max(b))
    ///     })
    ///     .collect();
    /// edges.sort();
    /// let by_hand = Graph::from_edges(&edges, 4, false).unwrap();
    /// assert!(by_hand == g.canonical_form(None).unwrap());
    /// ```
    pub fn canonical_permutation(&self, colors: Option<&[i64]>) -> Result<Vec<VertexId>> {
        check_len("colors", colors, self.vcount())?;
        let c = opt_view(colors);
        let mut labeling = VectorInt::new();
        igraph_call!(igraph_canonical_permutation(
            self,
            opt_ptr(&c),
            &mut labeling
        ))?;
        Ok(labeling.into())
    }

    /// The canonical labeling of the graph computed by Bliss with the given
    /// splitting heuristic, together with the Bliss statistics
    /// (`igraph_canonical_permutation_bliss`).
    ///
    /// See [`canonical_permutation`](Self::canonical_permutation).
    ///
    /// Binds [`igraph_canonical_permutation_bliss`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_canonical_permutation_bliss).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::isomorphism::BlissSh;
    /// let k3 = Graph::full(3, false, false).unwrap();
    /// let (labeling, info) = k3.canonical_permutation_bliss(None, BlissSh::Fsm).unwrap();
    /// assert_eq!(labeling.len(), 3);
    /// assert_eq!(info.group_size, "6");
    /// ```
    pub fn canonical_permutation_bliss(
        &self,
        colors: Option<&[i64]>,
        sh: BlissSh,
    ) -> Result<(Vec<VertexId>, BlissInfo)> {
        check_len("colors", colors, self.vcount())?;
        let c = opt_view(colors);
        let mut labeling = VectorInt::new();
        let mut info = RawBlissInfo::new();
        igraph_call!(igraph_canonical_permutation_bliss(
            self,
            opt_ptr(&c),
            &mut labeling,
            sh.into(),
            &mut info.0
        ))?;
        Ok((labeling.into(), info.to_info()))
    }

    /// The canonical form of the graph: the graph with its vertices permuted
    /// by its [canonical labeling](Self::canonical_permutation).
    ///
    /// Two uncolored graphs are isomorphic if and only if their canonical
    /// forms are identical, i.e. compare equal with `==`
    /// ([`Graph::is_same_graph`]). This makes canonical forms perfect keys to
    /// deduplicate graphs up to isomorphism. With `colors`, the returned
    /// graph does not carry the colors: two colored graphs are isomorphic
    /// (as colored graphs) if and only if their canonical forms are identical
    /// *and* so are their colors relabeled by the canonical labelings, i.e.
    /// `colors[labeling[i]]` for each `i` (see
    /// [`canonical_permutation`](Self::canonical_permutation)).
    /// Multi-edges are not supported (encode them with
    /// [`simplify_and_colorize`](Self::simplify_and_colorize) first).
    ///
    /// Combines [`igraph_canonical_permutation`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_canonical_permutation)
    /// and [`igraph_permute_vertices`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_permute_vertices)
    /// (see [`Graph::permute_vertices`]), then sorts the edges (and, when
    /// undirected, their endpoints) so that the result does not depend on
    /// the input edge order.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let a = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let b = Graph::from_edges(&[(2, 0), (0, 3), (3, 1)], 4, false).unwrap();
    /// assert!(a.canonical_form(None).unwrap() == b.canonical_form(None).unwrap());
    /// ```
    pub fn canonical_form(&self, colors: Option<&[i64]>) -> Result<Graph> {
        let labeling = self.canonical_permutation(colors)?;
        let permuted = self.permute_vertices(&labeling)?;
        // igraph_permute_vertices keeps the original edge order; sort the
        // edges so that identical canonical forms compare equal.
        let mut edges: Vec<(VertexId, VertexId)> = permuted
            .edge_list()
            .into_iter()
            .map(|(a, b)| {
                if self.is_directed() || a <= b {
                    (a, b)
                } else {
                    (b, a)
                }
            })
            .collect();
        edges.sort_unstable();
        Graph::from_edges(&edges, self.vcount(), self.is_directed())
    }

    /// Number of automorphisms of the graph (`igraph_count_automorphisms`).
    ///
    /// An automorphism is an isomorphism of the graph onto itself, i.e. a
    /// symmetry. The count is returned as an `f64` because it can be huge;
    /// use [`count_automorphisms_bliss`](Self::count_automorphisms_bliss) to
    /// get the exact value as a string. `colors` optionally restricts to
    /// color-preserving automorphisms. Multi-edges are not supported.
    ///
    /// Binds [`igraph_count_automorphisms`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_count_automorphisms).
    ///
    /// # Errors
    /// [`ErrorKind::Overflow`] if the count does
    /// not fit into an `f64`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let star = Graph::star(5, StarMode::Undirected, 0).unwrap();
    /// assert_eq!(star.count_automorphisms(None).unwrap(), 24.0); // 4!
    /// // The Petersen graph's symmetry group is S5, of order 120.
    /// assert_eq!(Graph::famous("Petersen").unwrap().count_automorphisms(None).unwrap(), 120.0);
    /// // Coloring one leaf differently leaves 3! symmetries.
    /// assert_eq!(star.count_automorphisms(Some(&[0, 1, 1, 1, 2])).unwrap(), 6.0);
    /// ```
    pub fn count_automorphisms(&self, colors: Option<&[i64]>) -> Result<f64> {
        check_len("colors", colors, self.vcount())?;
        let c = opt_view(colors);
        let mut res = 0.0;
        igraph_call!(igraph_count_automorphisms(self, opt_ptr(&c), &mut res))?;
        Ok(res)
    }

    /// Number of automorphisms computed by Bliss, returned exactly in
    /// [`BlissInfo::group_size`] (`igraph_count_automorphisms_bliss`).
    ///
    /// Same semantics as [`count_automorphisms`](Self::count_automorphisms)
    /// (optional `colors`, no multi-edges), with the splitting heuristic `sh`
    /// chosen explicitly; the returned statistics also describe the search
    /// tree. Unlike the `f64` count, the decimal
    /// [`group_size`](BlissInfo::group_size) is exact for arbitrarily large
    /// groups.
    ///
    /// Binds [`igraph_count_automorphisms_bliss`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_count_automorphisms_bliss).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::isomorphism::BlissSh;
    /// let k23 = Graph::full(23, false, false).unwrap();
    /// let info = k23.count_automorphisms_bliss(None, BlissSh::F).unwrap();
    /// assert_eq!(info.group_size, "25852016738884976640000"); // 23!
    /// ```
    pub fn count_automorphisms_bliss(
        &self,
        colors: Option<&[i64]>,
        sh: BlissSh,
    ) -> Result<BlissInfo> {
        check_len("colors", colors, self.vcount())?;
        let c = opt_view(colors);
        let mut info = RawBlissInfo::new();
        igraph_call!(igraph_count_automorphisms_bliss(
            self,
            opt_ptr(&c),
            sh.into(),
            &mut info.0
        ))?;
        Ok(info.to_info())
    }

    /// Generators of the automorphism group of the graph
    /// (`igraph_automorphism_group`).
    ///
    /// Each generator is a permutation of the vertices (0-based). The set may
    /// not be minimal and depends on the algorithm's internals; every
    /// automorphism is a product of generators. `colors` optionally restricts
    /// to color-preserving automorphisms. Multi-edges are not supported.
    ///
    /// Binds [`igraph_automorphism_group`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_automorphism_group).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let path = Graph::ring(3, false, false, false).unwrap();
    /// assert_eq!(path.automorphism_group(None).unwrap(), vec![vec![2, 1, 0]]);
    /// ```
    pub fn automorphism_group(&self, colors: Option<&[i64]>) -> Result<Vec<Vec<VertexId>>> {
        check_len("colors", colors, self.vcount())?;
        let c = opt_view(colors);
        let mut gens = VectorIntList::new();
        igraph_call!(igraph_automorphism_group(self, opt_ptr(&c), &mut gens))?;
        Ok(gens.to_vecs())
    }

    /// Generators of the automorphism group computed by Bliss with the given
    /// splitting heuristic, with statistics (`igraph_automorphism_group_bliss`).
    ///
    /// See [`automorphism_group`](Self::automorphism_group).
    ///
    /// Binds [`igraph_automorphism_group_bliss`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_automorphism_group_bliss).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::isomorphism::BlissSh;
    /// let c4 = Graph::ring(4, false, false, true).unwrap();
    /// let (gens, info) = c4.automorphism_group_bliss(None, BlissSh::Fs).unwrap();
    /// assert_eq!(gens.len() as u64, info.nof_generators);
    /// assert_eq!(info.group_size, "8");
    /// ```
    pub fn automorphism_group_bliss(
        &self,
        colors: Option<&[i64]>,
        sh: BlissSh,
    ) -> Result<(Vec<Vec<VertexId>>, BlissInfo)> {
        check_len("colors", colors, self.vcount())?;
        let c = opt_view(colors);
        let mut gens = VectorIntList::new();
        let mut info = RawBlissInfo::new();
        igraph_call!(igraph_automorphism_group_bliss(
            self,
            opt_ptr(&c),
            &mut gens,
            sh.into(),
            &mut info.0
        ))?;
        Ok((gens.to_vecs(), info.to_info()))
    }

    // ----- isomorphism classes -------------------------------------------------

    /// The isomorphism class of a small graph (`igraph_isoclass`).
    ///
    /// Graphs with the same number of vertices and directedness are
    /// isomorphic iff they have the same class. Classes are numbered from 0
    /// (the empty graph) to [`graph_count`]`(n, directed) - 1` (the complete
    /// graph). Supported: directed graphs with 3–4 vertices, undirected
    /// graphs with 3–6 vertices. Multi-edges and self-loops are ignored.
    /// Time complexity: O(|E|).
    ///
    /// Binds [`igraph_isoclass`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_isoclass).
    ///
    /// # Errors
    /// [`ErrorKind::Unimplemented`] for unsupported sizes.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let empty = Graph::new(3, false);
    /// let triangle = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, false).unwrap();
    /// assert_eq!(empty.isoclass().unwrap(), 0);
    /// assert_eq!(triangle.isoclass().unwrap(), 3);
    /// ```
    pub fn isoclass(&self) -> Result<usize> {
        let mut class = 0;
        igraph_call!(igraph_isoclass(self, &mut class))?;
        Ok(class as usize)
    }

    /// The isomorphism class of the subgraph induced by `vids`
    /// (`igraph_isoclass_subgraph`).
    ///
    /// Same numbering and size limits as [`isoclass`](Self::isoclass); each
    /// vertex may appear at most once in `vids`. Time complexity:
    /// O((d+n)·n), d being the average degree and n the number of vertices.
    ///
    /// Binds [`igraph_isoclass_subgraph`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_isoclass_subgraph).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`] for invalid vertices,
    /// [`ErrorKind::InvalidValue`] if a vertex is selected twice, and
    /// [`ErrorKind::Unimplemented`] for unsupported subgraph sizes.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false).unwrap();
    /// assert_eq!(g.isoclass_subgraph(&[0, 1, 2]).unwrap(), 3); // triangle
    /// assert_eq!(g.isoclass_subgraph(&[1, 2, 3]).unwrap(), 2); // path
    /// ```
    pub fn isoclass_subgraph<'a>(&self, vids: impl Into<VertexSelector<'a>>) -> Result<usize> {
        let vs = vids.into().to_raw()?;
        // igraph 1.0.0 and 1.0.1 silently return a wrong class for repeated
        // vertices.
        let mut list = VectorInt::new();
        igraph_call!(igraph_vs_as_vector(self, vs.get(), &mut list))?;
        let mut sorted = list.to_vec();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(Error::invalid(
                "isoclass_subgraph: a vertex is selected twice",
            ));
        }
        let mut class = 0;
        igraph_call!(igraph_isoclass_subgraph(self, vs.get(), &mut class))?;
        Ok(class as usize)
    }

    /// Creates the canonical representative graph of an isomorphism class
    /// (`igraph_isoclass_create`).
    ///
    /// `size` is the number of vertices (3–4 directed, 3–6 undirected) and
    /// `number` the class, in `0..graph_count(size, directed)`. This is the
    /// inverse of [`isoclass`](Self::isoclass). Time complexity: O(|V|+|E|).
    /// Use it to draw or inspect the shapes counted by
    /// [`motifs_randesu`](Self::motifs_randesu).
    ///
    /// Binds [`igraph_isoclass_create`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_isoclass_create).
    ///
    /// # Errors
    /// For unsupported sizes or out-of-range class numbers.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // Class 15 of directed triads is the complete digraph.
    /// let g = Graph::isoclass_create(3, 15, true).unwrap();
    /// assert_eq!(g.ecount(), 6);
    /// assert_eq!(g.isoclass().unwrap(), 15);
    /// ```
    pub fn isoclass_create(size: usize, number: usize, directed: bool) -> Result<Graph> {
        Graph::init_with(|g| unsafe {
            igraph_isoclass_create(g, size as igraph_int_t, number as igraph_int_t, directed)
        })
    }

    // ----- motifs -------------------------------------------------------------

    /// The motif histogram of the graph with the RAND-ESU algorithm
    /// (`igraph_motifs_randesu`).
    ///
    /// Motifs are small weakly connected *induced* subgraphs. Entry `i` of
    /// the result counts the induced subgraphs on `size` vertices of
    /// [isomorphism class](Self::isoclass) `i`; classes that are not
    /// (weakly) connected are reported as `NaN`, since they are not counted.
    /// Supported sizes: 3–4 (directed), 3–6 (undirected); directed motifs are
    /// counted in directed graphs.
    ///
    /// `cut_prob` optionally gives, for each of the `size` levels of the
    /// search tree, the probability of cutting a branch (sampling à la
    /// FANMOD, Wernicke and Rasche 2006); `None` performs a complete search.
    /// Cutting uses igraph's RNG: seed it with [`rng::seed`](crate::rng::seed)
    /// for reproducible samples.
    ///
    /// To assess the significance of motif counts, compare them with those
    /// of degree-preserving randomizations, e.g. from [`Graph::rewire`].
    /// For size 3 in directed graphs, [`triad_census`](Self::triad_census)
    /// additionally counts the unconnected triads.
    ///
    /// Binds [`igraph_motifs_randesu`](https://igraph.org/c/html/latest/igraph-Motifs.html#igraph_motifs_randesu).
    ///
    /// # Errors
    /// For unsupported sizes, or if `cut_prob` does not have length `size`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A triangle with a pendant vertex: two 2-paths and one triangle.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false).unwrap();
    /// let hist = g.motifs_randesu(3, None).unwrap();
    /// assert_eq!(&hist[2..], &[2.0, 1.0]);
    /// ```
    pub fn motifs_randesu(&self, size: usize, cut_prob: Option<&[f64]>) -> Result<Vec<f64>> {
        let cp = cut_prob.map(Vector::view);
        let mut hist = Vector::new();
        igraph_call!(igraph_motifs_randesu(
            self,
            &mut hist,
            size as igraph_int_t,
            opt_ptr(&cp)
        ))?;
        Ok(hist.into())
    }

    /// Finds the motifs of the graph and streams them to a closure
    /// (`igraph_motifs_randesu_callback`).
    ///
    /// `f(vids, isoclass)` is called for every motif (weakly connected
    /// induced subgraph on `size` vertices) found, with its vertices and its
    /// [isomorphism class](Self::isoclass); return `true` to continue or
    /// `false` to stop the search. A panic in `f` aborts the search and is
    /// resumed when this function returns. See
    /// [`motifs_randesu`](Self::motifs_randesu) for `size` and `cut_prob`.
    ///
    /// Binds [`igraph_motifs_randesu_callback`](https://igraph.org/c/html/latest/igraph-Motifs.html#igraph_motifs_randesu_callback).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false).unwrap();
    /// let mut triangles = vec![];
    /// g.motifs_randesu_callback(3, None, |vids, class| {
    ///     if class == 3 {
    ///         let mut t = vids.to_vec();
    ///         t.sort();
    ///         triangles.push(t);
    ///     }
    ///     true
    /// })
    /// .unwrap();
    /// assert_eq!(triangles, vec![vec![0, 1, 2]]);
    /// ```
    pub fn motifs_randesu_callback(
        &self,
        size: usize,
        cut_prob: Option<&[f64]>,
        mut f: impl FnMut(&[VertexId], usize) -> bool,
    ) -> Result<()> {
        let cp = cut_prob.map(Vector::view);
        let mut dynf: &mut MotifHandler<'_> = &mut f;
        let extra = (&mut dynf as *mut &mut MotifHandler<'_>).cast::<c_void>();
        igraph_call!(igraph_motifs_randesu_callback(
            self,
            size as igraph_int_t,
            opt_ptr(&cp),
            Some(motif_trampoline),
            extra
        ))
    }

    /// Total number of motifs, i.e. of weakly connected induced subgraphs on
    /// `size` vertices (`igraph_motifs_randesu_no`).
    ///
    /// Unlike [`motifs_randesu`](Self::motifs_randesu), it does not classify
    /// the subgraphs, so arbitrarily large sizes are supported. The result
    /// is an integer returned as `f64` (to avoid overflow). See
    /// [`motifs_randesu`](Self::motifs_randesu) for `cut_prob`.
    ///
    /// Binds [`igraph_motifs_randesu_no`](https://igraph.org/c/html/latest/igraph-Motifs.html#igraph_motifs_randesu_no).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `size < 3` or `cut_prob` does not have
    /// length `size`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let k10 = Graph::full(10, false, false).unwrap();
    /// assert_eq!(k10.motifs_randesu_no(4, None).unwrap(), 210.0); // C(10, 4)
    /// ```
    pub fn motifs_randesu_no(&self, size: usize, cut_prob: Option<&[f64]>) -> Result<f64> {
        let cp = cut_prob.map(Vector::view);
        let mut no = 0.0;
        igraph_call!(igraph_motifs_randesu_no(
            self,
            &mut no,
            size as igraph_int_t,
            opt_ptr(&cp)
        ))?;
        Ok(no)
    }

    /// Estimates the total number of motifs of the given size
    /// (`igraph_motifs_randesu_estimate`).
    ///
    /// Takes a *sample* of vertices ([`MotifSample`]: either a number of
    /// distinct vertices drawn uniformly at random, or an explicit list),
    /// runs the RAND-ESU enumeration rooted at each sampled vertex `v`
    /// (which counts the connected induced subgraphs on `size` vertices
    /// whose smallest vertex id is `v`), and multiplies the total by
    /// `vcount / sample_size`. With every vertex in the sample the result
    /// equals [`motifs_randesu_no`](Self::motifs_randesu_no); otherwise it
    /// depends on the vertex numbering, even for very symmetric graphs.
    /// Useful for large graphs where counting all subgraphs is too slow. See
    /// [`motifs_randesu`](Self::motifs_randesu) for `cut_prob`. The random
    /// sample is drawn from the calling thread's RNG ([`rng::seed`](crate::rng::seed)).
    /// The null graph has an estimate of 0 whatever the sample.
    ///
    /// Binds [`igraph_motifs_randesu_estimate`](https://igraph.org/c/html/latest/igraph-Motifs.html#igraph_motifs_randesu_estimate).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `size < 3`, `cut_prob` does not have
    /// length `size`, or the sample is empty or larger than the graph
    /// ([`MotifSample::Random`]); [`ErrorKind::InvalidVertexId`] if
    /// [`MotifSample::Vertices`] has invalid vertices.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::isomorphism::MotifSample;
    /// let k10 = Graph::full(10, false, false).unwrap();
    /// // Sampling every vertex gives the exact count, C(10, 3).
    /// let all: Vec<i64> = (0..10).collect();
    /// let est = k10.motifs_randesu_estimate(3, None, MotifSample::Vertices(&all)).unwrap();
    /// assert_eq!(est, 120.0);
    /// // A random sample of 5 vertices gives an estimate, reproducible
    /// // once the (per-thread) RNG is seeded.
    /// igraph::rng::seed(42).unwrap();
    /// let est = k10.motifs_randesu_estimate(3, None, MotifSample::Random(5)).unwrap();
    /// assert!(est > 0.0);
    /// igraph::rng::seed(42).unwrap();
    /// assert_eq!(k10.motifs_randesu_estimate(3, None, MotifSample::Random(5)).unwrap(), est);
    /// ```
    pub fn motifs_randesu_estimate(
        &self,
        size: usize,
        cut_prob: Option<&[f64]>,
        sample: MotifSample<'_>,
    ) -> Result<f64> {
        let cp = cut_prob.map(Vector::view);
        let (sample_size, sample_view) = match sample {
            // A size above `igraph_int_t::MAX` would turn negative, which
            // igraph 1.0.1 does not reject (it aborts on an assertion).
            MotifSample::Random(n) => (
                igraph_int_t::try_from(n).map_err(|_| {
                    Error::invalid(format!("sample size {n} exceeds the number of vertices"))
                })?,
                None,
            ),
            MotifSample::Vertices(v) => (v.len() as igraph_int_t, Some(VectorInt::view(v))),
        };
        let mut est = 0.0;
        igraph_call!(igraph_motifs_randesu_estimate(
            self,
            &mut est,
            size as igraph_int_t,
            opt_ptr(&cp),
            sample_size,
            opt_ptr(&sample_view)
        ))?;
        Ok(est)
    }

    /// The dyad census of the graph (`igraph_dyad_census`), as defined by
    /// Holland and Leinhardt (1970).
    ///
    /// Classifies each unordered pair of vertices as *mutual* (edges in both
    /// directions), *asymmetric* (in one direction only) or *null* (not
    /// connected). In undirected graphs every connected pair is mutual.
    /// Multi-edges and self-loops do not matter. Time complexity: O(|V|+|E|).
    ///
    /// See also [`Graph::reciprocity`]: with the default mode it equals
    /// `2 mutual / (2 mutual + asymmetric)` on simple graphs.
    ///
    /// Binds [`igraph_dyad_census`](https://igraph.org/c/html/latest/igraph-Motifs.html#igraph_dyad_census).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 0), (1, 2)], 4, true).unwrap();
    /// let d = g.dyad_census().unwrap();
    /// assert_eq!((d.mutual, d.asymmetric, d.null), (1.0, 1.0, 4.0));
    /// ```
    pub fn dyad_census(&self) -> Result<DyadCensus> {
        let (mut mutual, mut asymmetric, mut null) = (0.0, 0.0, 0.0);
        igraph_call!(igraph_dyad_census(
            self,
            &mut mutual,
            &mut asymmetric,
            &mut null
        ))?;
        Ok(DyadCensus {
            mutual,
            asymmetric,
            null,
        })
    }

    /// The triad census of the graph (`igraph_triad_census`), as defined by
    /// Davis and Leinhardt (1972).
    ///
    /// Classifies every vertex triple into one of the 16 types of directed
    /// triads, see [`TriadCensus`]. Intended for directed graphs: undirected
    /// edges are treated as mutual (and igraph emits a warning). Note that
    /// the order differs from the isoclass order of
    /// [`motifs_randesu`](Self::motifs_randesu).
    ///
    /// Binds [`igraph_triad_census`](https://igraph.org/c/html/latest/igraph-Motifs.html#igraph_triad_census).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let cycle = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, true).unwrap();
    /// let t = cycle.triad_census().unwrap();
    /// assert_eq!(t["030C"], 1.0);
    /// assert_eq!(t.total(), 1.0);
    /// ```
    pub fn triad_census(&self) -> Result<TriadCensus> {
        let mut res = Vector::new();
        igraph_call!(igraph_triad_census(self, &mut res))?;
        let mut counts = [0.0; 16];
        if res.len() != 16 {
            return Err(Error::new(
                ErrorKind::Internal,
                "triad census of unexpected length",
            ));
        }
        counts.copy_from_slice(&res);
        Ok(TriadCensus { counts })
    }

    /// Number of triangles each selected vertex is part of
    /// (`igraph_count_adjacent_triangles`).
    ///
    /// Edge directions and multiplicities are ignored. Returned as `f64`
    /// (to avoid overflow). Time complexity: O(d²·n), d the average degree
    /// of the n queried vertices.
    ///
    /// See also [`Graph::transitivity_local_undirected`], the local
    /// clustering coefficient `t(v) / (d(v) (d(v) - 1) / 2)`.
    ///
    /// Binds [`igraph_count_adjacent_triangles`](https://igraph.org/c/html/latest/igraph-Motifs.html#igraph_count_adjacent_triangles).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false).unwrap();
    /// assert_eq!(g.count_adjacent_triangles(..).unwrap(), vec![1.0, 1.0, 1.0, 0.0]);
    /// ```
    pub fn count_adjacent_triangles<'a>(
        &self,
        vids: impl Into<VertexSelector<'a>>,
    ) -> Result<Vec<f64>> {
        let vs = vids.into().to_raw()?;
        let mut res = Vector::new();
        directed_cache_guard(self, || {
            igraph_call!(igraph_count_adjacent_triangles(self, &mut res, vs.get()))
        })?;
        Ok(res.into())
    }

    /// All the triangles of the graph, each listed once as a triple of
    /// vertex ids (`igraph_list_triangles`).
    ///
    /// Edge directions and multi-edges are ignored. Time complexity:
    /// O(d²·n). See also [`Graph::cliques`] for complete subgraphs of any
    /// size.
    ///
    /// Binds [`igraph_list_triangles`](https://igraph.org/c/html/latest/igraph-Motifs.html#igraph_list_triangles).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false).unwrap();
    /// let mut t = g.list_triangles().unwrap();
    /// t[0].sort();
    /// assert_eq!(t, vec![[0, 1, 2]]);
    /// ```
    pub fn list_triangles(&self) -> Result<Vec<[VertexId; 3]>> {
        let mut res = VectorInt::new();
        directed_cache_guard(self, || igraph_call!(igraph_list_triangles(self, &mut res)))?;
        Ok(res.as_chunks::<3>().0.to_vec())
    }

    /// Total number of triangles of the graph (`igraph_count_triangles`).
    ///
    /// Edge directions, multiplicities and self-loops are ignored. Returned
    /// as `f64` (to avoid overflow). Time complexity: O(|V|·d²).
    ///
    /// See also [`Graph::transitivity_undirected`], the global clustering
    /// coefficient `3 × triangles / connected triples`.
    ///
    /// Binds [`igraph_count_triangles`](https://igraph.org/c/html/latest/igraph-Motifs.html#igraph_count_triangles).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let k4 = Graph::full(4, false, false).unwrap();
    /// assert_eq!(k4.count_triangles().unwrap(), 4.0);
    /// // Zachary's karate club has 45 triangles.
    /// assert_eq!(Graph::famous("Zachary").unwrap().count_triangles().unwrap(), 45.0);
    /// ```
    pub fn count_triangles(&self) -> Result<f64> {
        let mut res = 0.0;
        directed_cache_guard(self, || {
            igraph_call!(igraph_count_triangles(self, &mut res))
        })?;
        Ok(res)
    }

    // ----- graphlets ---------------------------------------------------------

    fn isomorphism_check_weights(&self, weights: &[f64]) -> Result<()> {
        if weights.len() != self.ecount() {
            return Err(Error::invalid(format!(
                "weights has length {}, expected {} (one per edge)",
                weights.len(),
                self.ecount()
            )));
        }
        Ok(())
    }

    /// The candidate basis of the graphlet decomposition
    /// (`igraph_graphlets_candidate_basis`).
    ///
    /// First step of the graphlet decomposition (Azari Soufiani and Airoldi):
    /// the cliques of the graph thresholded at decreasing edge weights, with
    /// the highest threshold at which each was found. The graph must be
    /// simple; edge directions are ignored; `weights` (one per edge) are
    /// mandatory.
    ///
    /// An edgeless graph has an empty basis.
    ///
    /// Binds [`igraph_graphlets_candidate_basis`](https://igraph.org/c/html/latest/igraph-Graphlets.html#igraph_graphlets_candidate_basis).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `weights` does not have one entry per
    /// edge or the graph is not simple (ignoring directions).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A heavy triangle attached to a light edge.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false).unwrap();
    /// let basis = g.graphlets_candidate_basis(&[5.0, 5.0, 5.0, 1.0]).unwrap();
    /// assert_eq!(basis.cliques.len(), basis.thresholds.len());
    /// assert!(basis.cliques.iter().any(|c| c.len() == 3));
    /// ```
    pub fn graphlets_candidate_basis(&self, weights: &[f64]) -> Result<GraphletBasis> {
        self.isomorphism_check_weights(weights)?;
        if weights.is_empty() {
            // igraph 1.0.0 and 1.0.1 assert (aborting the process) on
            // edgeless graphs.
            return Ok(GraphletBasis {
                cliques: vec![],
                thresholds: vec![],
            });
        }
        let w = Vector::view(weights);
        let mut cliques = VectorIntList::new();
        let mut thresholds = Vector::new();
        igraph_call!(igraph_graphlets_candidate_basis(
            self,
            w.as_ptr(),
            &mut cliques,
            &mut thresholds
        ))?;
        Ok(GraphletBasis {
            cliques: cliques.to_vecs(),
            thresholds: thresholds.into(),
        })
    }

    /// Projects the graph on a graphlet basis (`igraph_graphlets_project`).
    ///
    /// Second step of the graphlet decomposition: fits a weight (`Mu`) for
    /// each clique of `cliques` so that the sum of the cliques, weighted,
    /// explains the edge weights, with `niter` iterations of the
    /// expectation-maximization algorithm. Each clique of size `n` is
    /// normalized by `n (n + 1) / 2`, so that a clique whose edges all have
    /// weight `w` and belong to no other clique converges to
    /// `w (n - 1) / (n + 1)`. `start` optionally provides the
    /// initial weights (one per clique), otherwise all ones are used. The
    /// graph need not be the one used to compute the basis, but must have
    /// matching vertex ids.
    ///
    /// Binds [`igraph_graphlets_project`](https://igraph.org/c/html/latest/igraph-Graphlets.html#igraph_graphlets_project).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `weights` or `start` have the wrong
    /// length, a clique lists a vertex twice, or the graph is not simple;
    /// [`ErrorKind::InvalidVertexId`] if a clique contains an invalid vertex.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false).unwrap();
    /// let w = [5.0, 5.0, 5.0, 1.0];
    /// let mu = g.graphlets_project(&w, &[vec![0, 1, 2], vec![2, 3]], None, 100).unwrap();
    /// // Fixed points w (n - 1) / (n + 1): 5 * 2/4 for the triangle, 1 * 1/3 for the edge.
    /// assert!((mu[0] - 2.5).abs() < 1e-3 && (mu[1] - 1.0 / 3.0).abs() < 1e-3);
    /// ```
    pub fn graphlets_project(
        &self,
        weights: &[f64],
        cliques: &[Vec<VertexId>],
        start: Option<&[f64]>,
        niter: usize,
    ) -> Result<Vec<f64>> {
        self.isomorphism_check_weights(weights)?;
        if let Some(s) = start
            && s.len() != cliques.len()
        {
            return Err(Error::invalid("start must have one weight per clique"));
        }
        // igraph 1.0.0 and 1.0.1 index internal arrays with the clique
        // members unchecked.
        let n = self.vcount() as i64;
        for (i, c) in cliques.iter().enumerate() {
            if let Some(&bad) = c.iter().find(|&&v| !(0..n).contains(&v)) {
                return Err(Error::new(
                    ErrorKind::InvalidVertexId,
                    format!("clique {i} contains the invalid vertex {bad}"),
                ));
            }
            let mut sorted = c.clone();
            sorted.sort_unstable();
            if sorted.windows(2).any(|w| w[0] == w[1]) {
                return Err(Error::invalid(format!(
                    "clique {i} contains a vertex twice"
                )));
            }
        }
        let w = Vector::view(weights);
        let cl = VectorIntList::from(cliques);
        let mut mu = start.map_or_else(Vector::new, Vector::from_slice);
        igraph_call!(igraph_graphlets_project(
            self,
            w.as_ptr(),
            &cl,
            &mut mu,
            start.is_some(),
            niter as igraph_int_t
        ))?;
        Ok(mu.into())
    }

    /// The graphlet decomposition of a weighted graph (`igraph_graphlets`).
    ///
    /// Models the graph as a union of potentially overlapping dense groups
    /// (cliques), each with a weight: runs
    /// [`graphlets_candidate_basis`](Self::graphlets_candidate_basis), then
    /// [`graphlets_project`](Self::graphlets_project) with `niter`
    /// iterations, and sorts the graphlets by decreasing weight. The graph
    /// must be simple; edge directions are ignored. An edgeless graph has no
    /// graphlets. See also the [`cliques`](crate::cliques) module, e.g.
    /// [`Graph::maximal_cliques`], for unweighted dense groups.
    ///
    /// Binds [`igraph_graphlets`](https://igraph.org/c/html/latest/igraph-Graphlets.html#igraph_graphlets).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false).unwrap();
    /// let gl = g.graphlets(&[5.0, 5.0, 5.0, 1.0], 1000).unwrap();
    /// // The heaviest graphlet is the triangle.
    /// let mut top = gl.cliques[0].clone();
    /// top.sort();
    /// assert_eq!(top, vec![0, 1, 2]);
    /// assert!(gl.weights.windows(2).all(|w| w[0] >= w[1]));
    /// ```
    pub fn graphlets(&self, weights: &[f64], niter: usize) -> Result<Graphlets> {
        self.isomorphism_check_weights(weights)?;
        if weights.is_empty() {
            // igraph 1.0.0 and 1.0.1 assert (aborting the process) on
            // edgeless graphs.
            return Ok(Graphlets {
                cliques: vec![],
                weights: vec![],
            });
        }
        let w = Vector::view(weights);
        let mut cliques = VectorIntList::new();
        let mut mu = Vector::new();
        igraph_call!(igraph_graphlets(
            self,
            w.as_ptr(),
            &mut cliques,
            &mut mu,
            niter as igraph_int_t
        ))?;
        Ok(Graphlets {
            cliques: cliques.to_vecs(),
            weights: mu.into(),
        })
    }
}

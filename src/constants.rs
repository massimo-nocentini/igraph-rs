//! Rust enumerations for igraph's C constants (`igraph_constants.h`).
//!
//! Each enum converts into the raw C value with [`From`] (e.g.
//! `igraph_neimode_t::from(NeighborMode::Out)`), and back with [`TryFrom`]
//! (an [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue)
//! error for values that are not part of the enumeration). All of them are
//! re-exported by the [prelude](crate::prelude). Module-specific enums (e.g.
//! layout or community options) live next to the functions that use them.
//!
//! ```
//! use igraph::{ffi, prelude::*};
//!
//! // Out-neighbors follow the edge direction, in-neighbors go against it.
//! let g = Graph::from_edges(&[(0, 1), (2, 1), (1, 1)], 3, true).unwrap();
//! assert_eq!(g.neighbors(1, NeighborMode::Out).unwrap(), vec![1]);
//! assert_eq!(g.neighbors(1, NeighborMode::In).unwrap(), vec![0, 1, 2]);
//! assert_eq!(g.degree(1, NeighborMode::All, Loops::Twice).unwrap(), vec![4]);
//! assert_eq!(g.degree(1, NeighborMode::All, Loops::None).unwrap(), vec![2]);
//!
//! // Conversions to and from the raw C constants.
//! assert_eq!(ffi::igraph_neimode_t::from(NeighborMode::In), ffi::igraph_neimode_t_IGRAPH_IN);
//! assert_eq!(NeighborMode::try_from(ffi::igraph_neimode_t_IGRAPH_ALL).unwrap(), NeighborMode::All);
//! assert!(Order::try_from(42 as ffi::igraph_order_t).is_err());
//! ```
//!
//! | Enum | C type | Used by (for example) |
//! |------|--------|-----------------------|
//! | [`NeighborMode`] | `igraph_neimode_t` | [`Graph::neighbors`](crate::Graph::neighbors), [`Graph::degree`](crate::Graph::degree), [`Graph::bfs`](crate::Graph::bfs), [`Graph::distances`](crate::Graph::distances) |
//! | [`Loops`] | `igraph_loops_t` | [`Graph::degree`](crate::Graph::degree), [`Graph::neighbors_with`](crate::Graph::neighbors_with), [`Graph::get_adjacency`](crate::Graph::get_adjacency) |
//! | [`EdgeTypeSw`], [`AllowedEdgeTypes`] | `igraph_edge_type_sw_t` | [`Graph::rewire`](crate::Graph::rewire), [`Graph::erdos_renyi_game_gnm`](crate::Graph::erdos_renyi_game_gnm) |
//! | [`Order`] | `igraph_order_t` | [`VectorInt::sort_ind`](crate::vector::VectorInt::sort_ind), [`Graph::sort_vertex_ids_by_degree`](crate::Graph::sort_vertex_ids_by_degree) |
//! | [`Connectedness`] | `igraph_connectedness_t` | [`Graph::connected_components`](crate::Graph::connected_components), [`Graph::is_connected`](crate::Graph::is_connected) |
//! | [`Reciprocity`] | `igraph_reciprocity_t` | [`Graph::reciprocity`](crate::Graph::reciprocity) |
//! | [`Adjacency`], [`GetAdjacency`] | `igraph_adjacency_t`, `igraph_get_adjacency_t` | [`Graph::adjacency`](crate::Graph::adjacency), [`Graph::get_adjacency`](crate::Graph::get_adjacency) |
//! | [`StarMode`], [`WheelMode`], [`TreeMode`] | `igraph_star_mode_t`, `igraph_wheel_mode_t`, `igraph_tree_mode_t` | [`Graph::star`](crate::Graph::star), [`Graph::wheel`](crate::Graph::wheel), [`Graph::kary_tree`](crate::Graph::kary_tree) |
//! | [`DegreeSequenceMethod`], [`RealizeDegseq`] | `igraph_degseq_t`, `igraph_realize_degseq_t` | [`Graph::degree_sequence_game`](crate::Graph::degree_sequence_game), [`Graph::realize_degree_sequence`](crate::Graph::realize_degree_sequence) |
//! | [`RandomTreeMethod`], [`BarabasiAlgorithm`], [`ChungLuVariant`] | `igraph_random_tree_t`, `igraph_barabasi_algorithm_t`, `igraph_chung_lu_t` | [`Graph::tree_game`](crate::Graph::tree_game), [`Graph::barabasi_game`](crate::Graph::barabasi_game), [`Graph::chung_lu_game`](crate::Graph::chung_lu_game) |
//! | [`EdgeOrder`] | `igraph_edgeorder_type_t` | [`EdgeSelector::AllOrdered`](crate::selector::EdgeSelector::AllOrdered) |
//! | [`ToDirected`], [`ToUndirected`] | `igraph_to_directed_t`, `igraph_to_undirected_t` | [`Graph::to_directed`](crate::Graph::to_directed), [`Graph::to_undirected`](crate::Graph::to_undirected) |
//! | [`VconnNei`] | `igraph_vconn_nei_t` | [`Graph::st_vertex_connectivity`](crate::Graph::st_vertex_connectivity) |
//! | [`SpincommUpdate`], [`SpinglassImplementation`], [`LpaVariant`], [`CommunityComparison`] | `igraph_spincomm_update_t`, ... | [`Graph::community_spinglass`](crate::Graph::community_spinglass), [`Graph::community_label_propagation`](crate::Graph::community_label_propagation), [`compare_communities`](crate::community::compare_communities) |
//! | [`TransitivityMode`] | `igraph_transitivity_mode_t` | [`Graph::transitivity_local_undirected`](crate::Graph::transitivity_local_undirected) |
//! | [`AddWeights`] | `igraph_add_weights_t` | [`NcolLglOptions`](crate::foreign::NcolLglOptions) |
//! | [`FasAlgorithm`], [`FvsAlgorithm`] | `igraph_fas_algorithm_t`, `igraph_fvs_algorithm_t` | [`Graph::feedback_arc_set`](crate::Graph::feedback_arc_set), [`Graph::feedback_vertex_set`](crate::Graph::feedback_vertex_set) |
//! | [`SubgraphImplementation`] | `igraph_subgraph_implementation_t` | [`Graph::induced_subgraph`](crate::Graph::induced_subgraph) |
//! | [`LayoutGrid`] | `igraph_layout_grid_t` | [`FruchtermanReingoldOptions`](crate::layout::FruchtermanReingoldOptions) |
//! | [`RandomWalkStuck`], [`VoronoiTiebreaker`] | `igraph_random_walk_stuck_t`, `igraph_voronoi_tiebreaker_t` | [`Graph::random_walk`](crate::Graph::random_walk), [`Graph::voronoi`](crate::Graph::voronoi) |
//! | [`MstAlgorithm`], [`Product`] | `igraph_mst_algorithm_t`, `igraph_product_t` | [`Graph::minimum_spanning_tree`](crate::Graph::minimum_spanning_tree), [`Graph::product`](crate::Graph::product) |
//! | [`MatrixStorage`] | `igraph_matrix_storage_t` | raw FFI functions taking flat matrices |

use crate::ffi::*;

/// Declares a Rust enum mirroring a C enum, together with `From`/`TryFrom`
/// conversions to and from the raw type.
///
/// ```ignore
/// ffi_enum! {
///     /// Docs.
///     pub enum NeighborMode: igraph_neimode_t {
///         /// Out-neighbors.
///         Out = igraph_neimode_t_IGRAPH_OUT,
///     }
/// }
/// ```
#[macro_export]
macro_rules! ffi_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident : $raw:ty {
            $( $(#[$vmeta:meta])* $variant:ident = $value:path ),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        $vis enum $name {
            $( $(#[$vmeta])* $variant ),+
        }

        impl From<$name> for $raw {
            fn from(value: $name) -> $raw {
                match value {
                    $( $name::$variant => $value as $raw ),+
                }
            }
        }

        impl TryFrom<$raw> for $name {
            type Error = $crate::error::Error;
            #[allow(non_upper_case_globals, unreachable_patterns)]
            fn try_from(value: $raw) -> $crate::error::Result<Self> {
                $( if value == $value as $raw { return Ok($name::$variant); } )+
                Err($crate::error::Error::invalid(format!(
                    "invalid value {} for {}", value, stringify!($name)
                )))
            }
        }
    };
}

ffi_enum! {
    /// Which neighbors (or incident edges) to consider in directed graphs
    /// (`igraph_neimode_t`). Ignored for undirected graphs.
    pub enum NeighborMode: igraph_neimode_t {
        /// Follow edges along their direction (successors).
        Out = igraph_neimode_t_IGRAPH_OUT,
        /// Follow edges against their direction (predecessors).
        In = igraph_neimode_t_IGRAPH_IN,
        /// Ignore edge directions.
        All = igraph_neimode_t_IGRAPH_ALL,
    }
}

impl NeighborMode {
    /// The mode with the direction reversed (`All` stays `All`).
    pub fn reverse(self) -> Self {
        match self {
            Self::Out => Self::In,
            Self::In => Self::Out,
            Self::All => Self::All,
        }
    }
}

ffi_enum! {
    /// How self-loops are counted (`igraph_loops_t`).
    pub enum Loops: igraph_loops_t {
        /// Ignore loop edges.
        None = igraph_loops_t_IGRAPH_NO_LOOPS,
        /// Count each loop twice, as it has two endpoints at the same vertex
        /// (the graph theoretical convention, which keeps the handshake
        /// lemma true).
        Twice = igraph_loops_t_IGRAPH_LOOPS_TWICE,
        /// Count each loop once. Only meaningful when edge directions are
        /// ignored: for out- or in-degrees of directed graphs each loop
        /// counts once anyway.
        Once = igraph_loops_t_IGRAPH_LOOPS_ONCE,
    }
}

impl From<bool> for Loops {
    /// `true` maps to [`Loops::Twice`] (igraph's `IGRAPH_LOOPS`), `false` to [`Loops::None`].
    fn from(loops: bool) -> Self {
        if loops { Loops::Twice } else { Loops::None }
    }
}

ffi_enum! {
    /// Which kind of edges a random generator may produce (`igraph_edge_type_sw_t`).
    pub enum EdgeTypeSw: igraph_edge_type_sw_t {
        /// Only simple graphs: no loops, no multi-edges.
        Simple = IGRAPH_SIMPLE_SW,
        /// Self-loops allowed.
        Loops = IGRAPH_LOOPS_SW,
        /// Multi-edges allowed.
        Multi = IGRAPH_MULTI_SW,
    }
}

/// Which kinds of edges a graph may contain, as a set of flags
/// (`igraph_edge_type_sw_t`, including the combination
/// `IGRAPH_LOOPS_SW | IGRAPH_MULTI_SW`).
///
/// This is the single crate-wide type for igraph's edge-type switches: it is
/// taken by the random generators of [`games`](crate::games) (e.g.
/// [`Graph::erdos_renyi_game_gnm`](crate::Graph::erdos_renyi_game_gnm)), by the
/// graphicality tests of [`mixing`](crate::mixing) (e.g.
/// [`is_graphical`](crate::mixing::is_graphical)) and by the deterministic
/// realizations of [`constructors`](crate::constructors) (e.g.
/// [`Graph::realize_degree_sequence`](crate::Graph::realize_degree_sequence)).
/// It is re-exported as `games::AllowedEdgeTypes`,
/// `mixing::AllowedEdgeTypes` and `constructors::AllowedEdgeTypes`, and all
/// these paths name the very same type.
///
/// Unlike [`EdgeTypeSw`], which lists the three basic flags, this type can
/// express their combination: build it with the associated constants
/// ([`SIMPLE`](Self::SIMPLE), [`LOOPS`](Self::LOOPS), [`MULTI`](Self::MULTI),
/// [`ALL`](Self::ALL)), with a struct literal, or by or-ing flags together.
/// Every function taking it accepts `impl Into<AllowedEdgeTypes>`, so an
/// [`EdgeTypeSw`] can be passed directly.
///
/// ```
/// use igraph::{ffi, prelude::*};
///
/// let both = AllowedEdgeTypes::LOOPS | AllowedEdgeTypes::MULTI;
/// assert_eq!(both, AllowedEdgeTypes::ALL);
/// assert_eq!(AllowedEdgeTypes::from(EdgeTypeSw::Multi), AllowedEdgeTypes::MULTI);
/// assert_eq!(EdgeTypeSw::Loops | EdgeTypeSw::Multi, AllowedEdgeTypes::ALL);
///
/// // Raw C flags, both ways.
/// let raw = ffi::igraph_edge_type_sw_t::from(AllowedEdgeTypes::ALL);
/// assert_eq!(raw, ffi::IGRAPH_LOOPS_SW | ffi::IGRAPH_MULTI_SW);
/// assert_eq!(AllowedEdgeTypes::try_from(raw).unwrap(), AllowedEdgeTypes::ALL);
///
/// // One name, one type, whatever the module path.
/// let t: igraph::games::AllowedEdgeTypes = igraph::constructors::AllowedEdgeTypes::MULTI;
/// assert_eq!(t, igraph::mixing::AllowedEdgeTypes::MULTI);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AllowedEdgeTypes {
    /// Whether self-loops are allowed.
    pub loops: bool,
    /// Whether multi-edges (parallel edges) are allowed.
    pub multi: bool,
}

impl AllowedEdgeTypes {
    /// Simple graphs only: neither self-loops nor multi-edges (`IGRAPH_SIMPLE_SW`).
    pub const SIMPLE: Self = Self {
        loops: false,
        multi: false,
    };
    /// Self-loops allowed, but no multi-edges (`IGRAPH_LOOPS_SW`); at most
    /// one self-loop per vertex.
    pub const LOOPS: Self = Self {
        loops: true,
        multi: false,
    };
    /// Multi-edges allowed, but no self-loops (`IGRAPH_MULTI_SW`).
    pub const MULTI: Self = Self {
        loops: false,
        multi: true,
    };
    /// Both self-loops and multi-edges allowed (`IGRAPH_LOOPS_SW | IGRAPH_MULTI_SW`).
    pub const ALL: Self = Self {
        loops: true,
        multi: true,
    };

    /// Former enum variant name of [`SIMPLE`](Self::SIMPLE), kept so that
    /// code written for the old `constructors::AllowedEdgeTypes` enum compiles.
    #[doc(hidden)]
    #[allow(non_upper_case_globals)]
    pub const Simple: Self = Self::SIMPLE;
    /// Former enum variant name of [`LOOPS`](Self::LOOPS).
    #[doc(hidden)]
    #[allow(non_upper_case_globals)]
    pub const Loops: Self = Self::LOOPS;
    /// Former enum variant name of [`MULTI`](Self::MULTI).
    #[doc(hidden)]
    #[allow(non_upper_case_globals)]
    pub const Multi: Self = Self::MULTI;
    /// Former enum variant name of [`ALL`](Self::ALL).
    #[doc(hidden)]
    #[allow(non_upper_case_globals)]
    pub const LoopsAndMulti: Self = Self::ALL;

    /// The raw igraph bit flags.
    pub const fn to_raw(self) -> igraph_edge_type_sw_t {
        let mut raw = IGRAPH_SIMPLE_SW as igraph_edge_type_sw_t;
        if self.loops {
            raw |= IGRAPH_LOOPS_SW as igraph_edge_type_sw_t;
        }
        if self.multi {
            raw |= IGRAPH_MULTI_SW as igraph_edge_type_sw_t;
        }
        raw
    }
}

impl From<EdgeTypeSw> for AllowedEdgeTypes {
    fn from(sw: EdgeTypeSw) -> Self {
        match sw {
            EdgeTypeSw::Simple => Self::SIMPLE,
            EdgeTypeSw::Loops => Self::LOOPS,
            EdgeTypeSw::Multi => Self::MULTI,
        }
    }
}

impl From<AllowedEdgeTypes> for igraph_edge_type_sw_t {
    fn from(types: AllowedEdgeTypes) -> Self {
        types.to_raw()
    }
}

impl TryFrom<igraph_edge_type_sw_t> for AllowedEdgeTypes {
    type Error = crate::error::Error;
    /// Accepts exactly the four valid flag combinations; any other bit
    /// pattern is an [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue).
    fn try_from(raw: igraph_edge_type_sw_t) -> crate::error::Result<Self> {
        [Self::SIMPLE, Self::LOOPS, Self::MULTI, Self::ALL]
            .into_iter()
            .find(|t| t.to_raw() == raw)
            .ok_or_else(|| {
                crate::error::Error::invalid(format!("invalid value {raw} for AllowedEdgeTypes"))
            })
    }
}

impl std::ops::BitOr for AllowedEdgeTypes {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self {
            loops: self.loops || rhs.loops,
            multi: self.multi || rhs.multi,
        }
    }
}

impl std::ops::BitOr<EdgeTypeSw> for AllowedEdgeTypes {
    type Output = Self;
    fn bitor(self, rhs: EdgeTypeSw) -> Self {
        self | Self::from(rhs)
    }
}

impl std::ops::BitOr for EdgeTypeSw {
    type Output = AllowedEdgeTypes;
    fn bitor(self, rhs: Self) -> AllowedEdgeTypes {
        AllowedEdgeTypes::from(self) | AllowedEdgeTypes::from(rhs)
    }
}

/// Former name of [`EdgeTypeSw`], kept for backwards compatibility.
#[deprecated(note = "use `EdgeTypeSw`")]
#[allow(non_camel_case_types)]
pub type edge_type_sw_t = EdgeTypeSw;

ffi_enum! {
    /// Sorting order (`igraph_order_t`).
    pub enum Order: igraph_order_t {
        /// Ascending.
        Ascending = igraph_order_t_IGRAPH_ASCENDING,
        /// Descending.
        Descending = igraph_order_t_IGRAPH_DESCENDING,
    }
}

ffi_enum! {
    /// Weak or strong connectedness (`igraph_connectedness_t`).
    pub enum Connectedness: igraph_connectedness_t {
        /// Ignore edge directions.
        Weak = igraph_connectedness_t_IGRAPH_WEAK,
        /// Require directed paths in both directions.
        Strong = igraph_connectedness_t_IGRAPH_STRONG,
    }
}

ffi_enum! {
    /// Reciprocity definition (`igraph_reciprocity_t`).
    pub enum Reciprocity: igraph_reciprocity_t {
        /// Ratio of reciprocated edges to all edges.
        Default = igraph_reciprocity_t_IGRAPH_RECIPROCITY_DEFAULT,
        /// Ratio of mutual vertex pairs to connected vertex pairs.
        Ratio = igraph_reciprocity_t_IGRAPH_RECIPROCITY_RATIO,
    }
}

ffi_enum! {
    /// How to interpret an adjacency matrix (`igraph_adjacency_t`).
    pub enum Adjacency: igraph_adjacency_t {
        /// Directed graph, `A[i][j]` edges from `i` to `j`.
        Directed = igraph_adjacency_t_IGRAPH_ADJ_DIRECTED,
        /// Undirected graph, the matrix must be symmetric.
        Undirected = igraph_adjacency_t_IGRAPH_ADJ_UNDIRECTED,
        /// Undirected graph from the upper triangle.
        Upper = igraph_adjacency_t_IGRAPH_ADJ_UPPER,
        /// Undirected graph from the lower triangle.
        Lower = igraph_adjacency_t_IGRAPH_ADJ_LOWER,
        /// Undirected graph, `min(A[i][j], A[j][i])` edges.
        Min = igraph_adjacency_t_IGRAPH_ADJ_MIN,
        /// Undirected graph, `A[i][j] + A[j][i]` edges.
        Plus = igraph_adjacency_t_IGRAPH_ADJ_PLUS,
        /// Undirected graph, `max(A[i][j], A[j][i])` edges.
        Max = igraph_adjacency_t_IGRAPH_ADJ_MAX,
    }
}

ffi_enum! {
    /// Orientation of a star graph (`igraph_star_mode_t`).
    pub enum StarMode: igraph_star_mode_t {
        /// Edges point from the center outwards.
        Out = igraph_star_mode_t_IGRAPH_STAR_OUT,
        /// Edges point to the center.
        In = igraph_star_mode_t_IGRAPH_STAR_IN,
        /// Undirected star.
        Undirected = igraph_star_mode_t_IGRAPH_STAR_UNDIRECTED,
        /// Mutual directed edges.
        Mutual = igraph_star_mode_t_IGRAPH_STAR_MUTUAL,
    }
}

ffi_enum! {
    /// Orientation of a wheel graph (`igraph_wheel_mode_t`).
    pub enum WheelMode: igraph_wheel_mode_t {
        /// Spokes point from the center outwards.
        Out = igraph_wheel_mode_t_IGRAPH_WHEEL_OUT,
        /// Spokes point to the center.
        In = igraph_wheel_mode_t_IGRAPH_WHEEL_IN,
        /// Undirected wheel.
        Undirected = igraph_wheel_mode_t_IGRAPH_WHEEL_UNDIRECTED,
        /// Mutual directed edges.
        Mutual = igraph_wheel_mode_t_IGRAPH_WHEEL_MUTUAL,
    }
}

ffi_enum! {
    /// Orientation of a tree (`igraph_tree_mode_t`).
    pub enum TreeMode: igraph_tree_mode_t {
        /// Edges point away from the root.
        Out = igraph_tree_mode_t_IGRAPH_TREE_OUT,
        /// Edges point towards the root.
        In = igraph_tree_mode_t_IGRAPH_TREE_IN,
        /// Undirected tree.
        Undirected = igraph_tree_mode_t_IGRAPH_TREE_UNDIRECTED,
    }
}

ffi_enum! {
    /// Which part of the adjacency matrix to produce for undirected graphs
    /// (`igraph_get_adjacency_t`).
    pub enum GetAdjacency: igraph_get_adjacency_t {
        /// Upper triangle only.
        Upper = igraph_get_adjacency_t_IGRAPH_GET_ADJACENCY_UPPER,
        /// Lower triangle only.
        Lower = igraph_get_adjacency_t_IGRAPH_GET_ADJACENCY_LOWER,
        /// Full symmetric matrix.
        Both = igraph_get_adjacency_t_IGRAPH_GET_ADJACENCY_BOTH,
    }
}

ffi_enum! {
    /// Method for sampling graphs with a given degree sequence (`igraph_degseq_t`).
    pub enum DegreeSequenceMethod: igraph_degseq_t {
        /// Configuration model, may produce loops and multi-edges.
        Configuration = igraph_degseq_t_IGRAPH_DEGSEQ_CONFIGURATION,
        /// Viger–Latapy: undirected simple *connected* graphs, sampled
        /// approximately uniformly (a Monte Carlo method based on
        /// degree-preserving edge switches).
        VigerLatapy = igraph_degseq_t_IGRAPH_DEGSEQ_VL,
        /// Fast heuristic generating simple graphs: like the configuration
        /// model, but avoiding loops and multi-edges and restarting when stuck
        /// (not uniform).
        FastHeurSimple = igraph_degseq_t_IGRAPH_DEGSEQ_FAST_HEUR_SIMPLE,
        /// Configuration model with rejection of non-simple graphs: uniform
        /// simple graphs (possibly slow).
        ConfigurationSimple = igraph_degseq_t_IGRAPH_DEGSEQ_CONFIGURATION_SIMPLE,
        /// Edge-switching Markov chain started from a realization of the
        /// sequence: simple (directed or undirected) graphs.
        EdgeSwitchingSimple = igraph_degseq_t_IGRAPH_DEGSEQ_EDGE_SWITCHING_SIMPLE,
    }
}

ffi_enum! {
    /// Vertex choice when realizing a degree sequence (`igraph_realize_degseq_t`).
    pub enum RealizeDegseq: igraph_realize_degseq_t {
        /// Pick the vertex with the smallest remaining degree.
        Smallest = igraph_realize_degseq_t_IGRAPH_REALIZE_DEGSEQ_SMALLEST,
        /// Pick the vertex with the largest remaining degree.
        Largest = igraph_realize_degseq_t_IGRAPH_REALIZE_DEGSEQ_LARGEST,
        /// Pick vertices in index order.
        Index = igraph_realize_degseq_t_IGRAPH_REALIZE_DEGSEQ_INDEX,
    }
}

ffi_enum! {
    /// Random tree sampling algorithm (`igraph_random_tree_t`).
    pub enum RandomTreeMethod: igraph_random_tree_t {
        /// Random Prüfer sequence.
        Prufer = igraph_random_tree_t_IGRAPH_RANDOM_TREE_PRUFER,
        /// Loop-erased random walk.
        Lerw = igraph_random_tree_t_IGRAPH_RANDOM_TREE_LERW,
    }
}

ffi_enum! {
    /// Edge ordering for edge selectors (`igraph_edgeorder_type_t`).
    pub enum EdgeOrder: igraph_edgeorder_type_t {
        /// By edge id.
        Id = igraph_edgeorder_type_t_IGRAPH_EDGEORDER_ID,
        /// By source vertex.
        From = igraph_edgeorder_type_t_IGRAPH_EDGEORDER_FROM,
        /// By target vertex.
        To = igraph_edgeorder_type_t_IGRAPH_EDGEORDER_TO,
    }
}

ffi_enum! {
    /// How to convert undirected edges to directed ones (`igraph_to_directed_t`).
    pub enum ToDirected: igraph_to_directed_t {
        /// One directed edge with arbitrary direction.
        Arbitrary = igraph_to_directed_t_IGRAPH_TO_DIRECTED_ARBITRARY,
        /// Two mutual directed edges.
        Mutual = igraph_to_directed_t_IGRAPH_TO_DIRECTED_MUTUAL,
        /// One directed edge with random direction.
        Random = igraph_to_directed_t_IGRAPH_TO_DIRECTED_RANDOM,
        /// From lower to higher vertex id, giving an acyclic graph.
        Acyclic = igraph_to_directed_t_IGRAPH_TO_DIRECTED_ACYCLIC,
    }
}

ffi_enum! {
    /// How to convert directed edges to undirected ones (`igraph_to_undirected_t`).
    pub enum ToUndirected: igraph_to_undirected_t {
        /// Keep every edge.
        Each = igraph_to_undirected_t_IGRAPH_TO_UNDIRECTED_EACH,
        /// One undirected edge per connected vertex pair.
        Collapse = igraph_to_undirected_t_IGRAPH_TO_UNDIRECTED_COLLAPSE,
        /// One undirected edge per mutual pair.
        Mutual = igraph_to_undirected_t_IGRAPH_TO_UNDIRECTED_MUTUAL,
    }
}

ffi_enum! {
    /// What to do when computing the vertex connectivity of two *adjacent*
    /// vertices, which no vertex removal can disconnect
    /// (`igraph_vconn_nei_t`).
    pub enum VconnNei: igraph_vconn_nei_t {
        /// Report an error.
        Error = igraph_vconn_nei_t_IGRAPH_VCONN_NEI_ERROR,
        /// Return the number of vertices.
        NumberOfNodes = igraph_vconn_nei_t_IGRAPH_VCONN_NEI_NUMBER_OF_NODES,
        /// Ignore the edges between them: count the vertices needed to cut
        /// all the other paths.
        Ignore = igraph_vconn_nei_t_IGRAPH_VCONN_NEI_IGNORE,
        /// Return -1.
        Negative = igraph_vconn_nei_t_IGRAPH_VCONN_NEI_NEGATIVE,
    }
}

ffi_enum! {
    /// Update rule of the spinglass community detection (`igraph_spincomm_update_t`).
    pub enum SpincommUpdate: igraph_spincomm_update_t {
        /// Simple null model.
        Simple = igraph_spincomm_update_t_IGRAPH_SPINCOMM_UPDATE_SIMPLE,
        /// Configuration null model.
        Config = igraph_spincomm_update_t_IGRAPH_SPINCOMM_UPDATE_CONFIG,
    }
}

ffi_enum! {
    /// Value of local transitivity for vertices of degree < 2 (`igraph_transitivity_mode_t`).
    pub enum TransitivityMode: igraph_transitivity_mode_t {
        /// NaN.
        Nan = igraph_transitivity_mode_t_IGRAPH_TRANSITIVITY_NAN,
        /// Zero.
        Zero = igraph_transitivity_mode_t_IGRAPH_TRANSITIVITY_ZERO,
    }
}

ffi_enum! {
    /// Spinglass implementation (`igraph_spinglass_implementation_t`).
    pub enum SpinglassImplementation: igraph_spinglass_implementation_t {
        /// Original implementation (positive weights only).
        Orig = igraph_spinglass_implementation_t_IGRAPH_SPINCOMM_IMP_ORIG,
        /// Variant supporting negative weights.
        Neg = igraph_spinglass_implementation_t_IGRAPH_SPINCOMM_IMP_NEG,
    }
}

ffi_enum! {
    /// Measure used to compare two community structures (`igraph_community_comparison_t`).
    pub enum CommunityComparison: igraph_community_comparison_t {
        /// Variation of information.
        Vi = igraph_community_comparison_t_IGRAPH_COMMCMP_VI,
        /// Normalized mutual information.
        Nmi = igraph_community_comparison_t_IGRAPH_COMMCMP_NMI,
        /// Split-join distance.
        SplitJoin = igraph_community_comparison_t_IGRAPH_COMMCMP_SPLIT_JOIN,
        /// Rand index.
        Rand = igraph_community_comparison_t_IGRAPH_COMMCMP_RAND,
        /// Adjusted Rand index.
        AdjustedRand = igraph_community_comparison_t_IGRAPH_COMMCMP_ADJUSTED_RAND,
    }
}

ffi_enum! {
    /// Whether to add weights when converting from matrices (`igraph_add_weights_t`).
    pub enum AddWeights: igraph_add_weights_t {
        /// No weights.
        No = igraph_add_weights_t_IGRAPH_ADD_WEIGHTS_NO,
        /// Add weights.
        Yes = igraph_add_weights_t_IGRAPH_ADD_WEIGHTS_YES,
        /// Add weights if present.
        IfPresent = igraph_add_weights_t_IGRAPH_ADD_WEIGHTS_IF_PRESENT,
    }
}

ffi_enum! {
    /// Algorithm of the Barabási–Albert generator (`igraph_barabasi_algorithm_t`).
    pub enum BarabasiAlgorithm: igraph_barabasi_algorithm_t {
        /// Bag algorithm (may produce multi-edges).
        Bag = igraph_barabasi_algorithm_t_IGRAPH_BARABASI_BAG,
        /// Partial prefix-sum tree, simple graphs.
        Psumtree = igraph_barabasi_algorithm_t_IGRAPH_BARABASI_PSUMTREE,
        /// Partial prefix-sum tree, multi-edges allowed.
        PsumtreeMultiple = igraph_barabasi_algorithm_t_IGRAPH_BARABASI_PSUMTREE_MULTIPLE,
    }
}

ffi_enum! {
    /// Feedback arc set algorithm (`igraph_fas_algorithm_t`).
    pub enum FasAlgorithm: igraph_fas_algorithm_t {
        /// Minimum feedback arc set by integer programming, letting igraph
        /// pick the best such method (currently [`FasAlgorithm::ExactIpCg`]).
        ExactIp = igraph_fas_algorithm_t_IGRAPH_FAS_EXACT_IP,
        /// Eades–Lin–Smyth heuristic: linear time, at most `|E|/2 - |V|/6`
        /// edges, not necessarily minimum.
        ApproxEades = igraph_fas_algorithm_t_IGRAPH_FAS_APPROX_EADES,
        /// Exact integer programming on a set cover formulation, with
        /// incremental generation of cycle constraints.
        ExactIpCg = igraph_fas_algorithm_t_IGRAPH_FAS_EXACT_IP_CG,
        /// Exact integer programming on a topological order formulation with
        /// triangle inequalities (typically much slower than
        /// [`FasAlgorithm::ExactIpCg`]).
        ExactIpTi = igraph_fas_algorithm_t_IGRAPH_FAS_EXACT_IP_TI,
    }
}

ffi_enum! {
    /// Feedback vertex set algorithm (`igraph_fvs_algorithm_t`).
    pub enum FvsAlgorithm: igraph_fvs_algorithm_t {
        /// Exact integer programming.
        ExactIp = igraph_fvs_algorithm_t_IGRAPH_FVS_EXACT_IP,
    }
}

ffi_enum! {
    /// Implementation strategy for induced subgraphs (`igraph_subgraph_implementation_t`).
    pub enum SubgraphImplementation: igraph_subgraph_implementation_t {
        /// Let igraph decide.
        Auto = igraph_subgraph_implementation_t_IGRAPH_SUBGRAPH_AUTO,
        /// Copy the graph and delete vertices.
        CopyAndDelete = igraph_subgraph_implementation_t_IGRAPH_SUBGRAPH_COPY_AND_DELETE,
        /// Build the subgraph from scratch.
        CreateFromScratch = igraph_subgraph_implementation_t_IGRAPH_SUBGRAPH_CREATE_FROM_SCRATCH,
    }
}

ffi_enum! {
    /// Whether force-directed layouts use a grid (`igraph_layout_grid_t`).
    pub enum LayoutGrid: igraph_layout_grid_t {
        /// Use a grid.
        Grid = igraph_layout_grid_t_IGRAPH_LAYOUT_GRID,
        /// Do not use a grid.
        NoGrid = igraph_layout_grid_t_IGRAPH_LAYOUT_NOGRID,
        /// Use a grid for large graphs only.
        AutoGrid = igraph_layout_grid_t_IGRAPH_LAYOUT_AUTOGRID,
    }
}

ffi_enum! {
    /// What a random walk does when stuck (`igraph_random_walk_stuck_t`).
    pub enum RandomWalkStuck: igraph_random_walk_stuck_t {
        /// Report an error.
        Error = igraph_random_walk_stuck_t_IGRAPH_RANDOM_WALK_STUCK_ERROR,
        /// Return the walk so far.
        Return = igraph_random_walk_stuck_t_IGRAPH_RANDOM_WALK_STUCK_RETURN,
    }
}

ffi_enum! {
    /// Tie breaking in Voronoi partitioning (`igraph_voronoi_tiebreaker_t`).
    pub enum VoronoiTiebreaker: igraph_voronoi_tiebreaker_t {
        /// First generator.
        First = igraph_voronoi_tiebreaker_t_IGRAPH_VORONOI_FIRST,
        /// Last generator.
        Last = igraph_voronoi_tiebreaker_t_IGRAPH_VORONOI_LAST,
        /// Random generator.
        Random = igraph_voronoi_tiebreaker_t_IGRAPH_VORONOI_RANDOM,
    }
}

ffi_enum! {
    /// Variant of the Chung–Lu model (`igraph_chung_lu_t`).
    pub enum ChungLuVariant: igraph_chung_lu_t {
        /// Original model.
        Original = igraph_chung_lu_t_IGRAPH_CHUNG_LU_ORIGINAL,
        /// Maximum entropy variant.
        MaxEnt = igraph_chung_lu_t_IGRAPH_CHUNG_LU_MAXENT,
        /// Norros–Reittu variant.
        Nr = igraph_chung_lu_t_IGRAPH_CHUNG_LU_NR,
    }
}

ffi_enum! {
    /// Storage order of matrices given as flat arrays (`igraph_matrix_storage_t`).
    pub enum MatrixStorage: igraph_matrix_storage_t {
        /// Row-major.
        RowMajor = igraph_matrix_storage_t_IGRAPH_ROW_MAJOR,
        /// Column-major.
        ColumnMajor = igraph_matrix_storage_t_IGRAPH_COLUMN_MAJOR,
    }
}

ffi_enum! {
    /// Minimum spanning tree algorithm (`igraph_mst_algorithm_t`).
    pub enum MstAlgorithm: igraph_mst_algorithm_t {
        /// Let igraph choose.
        Automatic = igraph_mst_algorithm_t_IGRAPH_MST_AUTOMATIC,
        /// Unweighted (BFS based).
        Unweighted = igraph_mst_algorithm_t_IGRAPH_MST_UNWEIGHTED,
        /// Prim's algorithm.
        Prim = igraph_mst_algorithm_t_IGRAPH_MST_PRIM,
        /// Kruskal's algorithm.
        Kruskal = igraph_mst_algorithm_t_IGRAPH_MST_KRUSKAL,
    }
}

ffi_enum! {
    /// Kind of graph product (`igraph_product_t`).
    pub enum Product: igraph_product_t {
        /// Cartesian product.
        Cartesian = igraph_product_t_IGRAPH_PRODUCT_CARTESIAN,
        /// Lexicographic product.
        Lexicographic = igraph_product_t_IGRAPH_PRODUCT_LEXICOGRAPHIC,
        /// Strong product.
        Strong = igraph_product_t_IGRAPH_PRODUCT_STRONG,
        /// Tensor (categorical) product.
        Tensor = igraph_product_t_IGRAPH_PRODUCT_TENSOR,
        /// Modular product.
        Modular = igraph_product_t_IGRAPH_PRODUCT_MODULAR,
    }
}

ffi_enum! {
    /// Label propagation variant (`igraph_lpa_variant_t`).
    pub enum LpaVariant: igraph_lpa_variant_t {
        /// Sample from the dominant labels, and check the dominance of all
        /// vertices after each iteration.
        Dominance = igraph_lpa_variant_t_IGRAPH_LPA_DOMINANCE,
        /// Keep the current label if it is among the dominant ones; only
        /// check the vertices whose labels changed.
        Retention = igraph_lpa_variant_t_IGRAPH_LPA_RETENTION,
        /// Sample from the dominant labels, only checking neighbors (fast).
        Fast = igraph_lpa_variant_t_IGRAPH_LPA_FAST,
    }
}

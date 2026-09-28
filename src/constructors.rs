//! Deterministic graph generators (`igraph_constructors.h`).
//!
//! This module turns igraph's *deterministic constructors* into associated
//! functions of [`Graph`] returning [`Result`]`<Graph>`: given the same
//! arguments they always build the very same graph, with the same vertex and
//! edge ids. (Random generators live in [`crate::games`].)
//!
//! ```
//! use igraph::prelude::*;
//!
//! // The Petersen graph, three ways: by name, as G(5, 2), and from its LCF code.
//! let by_name = Graph::famous("Petersen").unwrap();
//! let gp = Graph::generalized_petersen(5, 2).unwrap();
//! assert_eq!((by_name.vcount(), by_name.ecount()), (10, 15));
//! assert_eq!((gp.vcount(), gp.ecount()), (10, 15));
//! // Both are 3-regular.
//! for g in [&by_name, &gp] {
//!     let degrees = g.degree(VertexSelector::All, NeighborMode::All, Loops::Twice).unwrap();
//!     assert!(degrees.iter().all(|&d| d == 3));
//! }
//!
//! // A 3x4 grid, a 6-cycle and a star with 5 leaves.
//! let grid = Graph::square_lattice(&[3, 4], 1, false, false, None).unwrap();
//! assert_eq!((grid.vcount(), grid.ecount()), (12, 17));
//! let c6 = Graph::cycle_graph(6, false, false).unwrap();
//! assert_eq!(c6.ecount(), 6);
//! let star = Graph::star(6, StarMode::Undirected, 0).unwrap();
//! assert_eq!(star.degree_of(0, NeighborMode::All, Loops::Twice).unwrap(), 5);
//! ```
//!
//! # Provided functionality
//!
//! | Family | Functions |
//! |---|---|
//! | From matrices | [`Graph::adjacency`], [`Graph::weighted_adjacency`], [`Graph::sparse_adjacency`], [`Graph::sparse_weighted_adjacency`] |
//! | From edge lists | [`Graph::small`] (see also [`Graph::from_edges`]) |
//! | Paths, cycles, stars | [`Graph::ring`], [`Graph::path_graph`], [`Graph::cycle_graph`], [`Graph::star`], [`Graph::wheel`] |
//! | Complete graphs | [`Graph::full`], [`Graph::full_citation`], [`Graph::full_multipartite`], [`Graph::turan`] |
//! | Lattices | [`Graph::square_lattice`], [`Graph::triangular_lattice`], [`Graph::hexagonal_lattice`], [`Graph::hypercube`] |
//! | Trees | [`Graph::kary_tree`], [`Graph::symmetric_tree`], [`Graph::regular_tree`], [`Graph::tree_from_parent_vector`], [`Graph::from_prufer`] |
//! | Circulant-like | [`Graph::circulant`], [`Graph::generalized_petersen`], [`Graph::lcf`], [`Graph::extended_chordal_ring`] |
//! | Word graphs | [`Graph::de_bruijn`], [`Graph::kautz`] |
//! | Named graphs | [`Graph::famous`] (with [`FamousGraph`]), [`Graph::atlas`], [`Graph::mycielski_graph`] |
//! | Degree sequences | [`Graph::realize_degree_sequence`], [`Graph::realize_bipartite_degree_sequence`] |
//! | Derived graphs | [`Graph::linegraph`] |
//!
//! The corresponding chapter of the C documentation is
//! [Deterministic graph generators](https://igraph.org/c/html/latest/igraph-Generators.html).
//!
//! # Conventions
//!
//! - Vertex counts and sizes are `usize`; values that may legitimately be
//!   negative (shifts, parent ids, degrees coming from
//!   [`Graph::degree`]) are `i64`.
//! - Every function returns an [`Error`] rather than panicking on invalid
//!   input; most errors are [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue).
//!
//! # See also
//!
//! | Need | Where |
//! |---|---|
//! | Random graphs (Erdős–Rényi, random regular, random trees, degree sequences, ...) | [`crate::games`], e.g. [`Graph::erdos_renyi_game_gnm`], [`Graph::k_regular_game`], [`Graph::tree_game`], [`Graph::degree_sequence_game`] |
//! | The inverse conversions (graph → matrix, tree → Prüfer code) | [`Graph::get_adjacency`], [`Graph::get_adjacency_sparse`], [`Graph::to_prufer`] in [`crate::conversion`] |
//! | Bipartite constructors | [`Graph::full_bipartite`], [`BipartiteGraph`](crate::bipartite::BipartiteGraph) in [`crate::bipartite`] |
//! | Graphs derived from other graphs | [`Graph::complementer`], [`Graph::mycielskian`], [`Graph::disjoint_union`] in [`crate::operators`] |
//! | Is a degree sequence realizable at all? | [`is_graphical`](crate::mixing::is_graphical), [`is_bigraphical`](crate::mixing::is_bigraphical) |
//! | Comparing generated graphs up to relabelling | [`Graph::isomorphic`], [`Graph::count_automorphisms`] in [`crate::isomorphism`] |
//! | Drawing them | [`Graph::layout_circle`] (rings), [`Graph::layout_star`], [`Graph::layout_grid`] (lattices), [`Graph::layout_reingold_tilford`] (trees) in [`crate::layout`] |
//! | Reading graphs from files | [`crate::foreign`] |
//!
//! ```
//! use igraph::prelude::*;
//!
//! // LCF notation and the named graph agree up to relabelling: the Heawood
//! // graph two ways.
//! let heawood = Graph::famous("Heawood")?;
//! assert!(Graph::lcf(14, &[5, -5], 7)?.isomorphic(&heawood)?);
//! // Its automorphism group PGL(2, 7) has order 336, and its girth is 6.
//! assert_eq!(heawood.count_automorphisms(None)?, 336.0);
//! assert_eq!(heawood.girth()?, Some(6));
//!
//! // Round trip through the adjacency matrix of the conversion module.
//! let a = heawood.get_adjacency(GetAdjacency::Both, None, Loops::Twice)?;
//! assert_eq!(Graph::adjacency(&a, Adjacency::Undirected, Loops::Twice)?, heawood);
//! # Ok::<(), igraph::Error>(())
//! ```
//!
//! # Differences from the C library
//!
//! A few defects of igraph 1.0.0 and 1.0.1 (the generator sources are
//! unchanged in 1.0.1) are worked around on the Rust side, so the wrappers
//! behave as documented:
//!
//! - [`Graph::sparse_adjacency`] / [`Graph::sparse_weighted_adjacency`]
//!   agree with their dense counterparts even when an entry below the
//!   diagonal has no mirror entry (igraph drops it in the unweighted `Max`
//!   mode and in the weighted `Max`, `Min` and `Plus` modes), and when
//!   explicit zeros are given in the `Undirected` mode (igraph's structural
//!   symmetry test would reject them).
//! - [`Graph::adjacency`] and [`Graph::sparse_adjacency`] reject NaN,
//!   infinite, negative and fractional edge counts (igraph casts them to
//!   integers unchecked).
//! - [`Graph::star`] and [`Graph::wheel`] with `n = 1` give the singleton
//!   graph (igraph returns the null graph).
//! - [`Graph::lcf`] with `n = 0` gives the null graph (igraph divides by
//!   zero), and LCF shifts and chordal ring offsets are reduced modulo the
//!   number of vertices (igraph adds them without overflow checks).
//! - [`Graph::extended_chordal_ring`] accepts a chord matrix without columns
//!   (igraph divides by the number of columns).

use crate::{
    constants::{Adjacency, Loops, NeighborMode, RealizeDegseq, StarMode, TreeMode, WheelMode},
    error::{Error, Result},
    ffi::*,
    graph::{Graph, VertexId},
    linalg::SparseMat,
    matrix::{Matrix, MatrixInt},
    vector::{Vector, VectorBool, VectorInt},
};
use std::{collections::BTreeMap, ffi::CString};

/// Converts a count to `igraph_int_t`, failing on overflow.
fn int(n: usize, what: &str) -> Result<igraph_int_t> {
    igraph_int_t::try_from(n).map_err(|_| Error::invalid(format!("{what} is too large: {n}")))
}

/// `base^exp` in exact integer arithmetic, `None` on overflow.
///
/// igraph's de Bruijn and Kautz generators compute these powers as doubles
/// and convert them to `igraph_int_t` *before* checking the range, which is
/// undefined behaviour for results of 2^63 and more: callers check here first.
fn checked_ipow(base: igraph_int_t, exp: igraph_int_t) -> Option<igraph_int_t> {
    match base {
        0 => Some(if exp == 0 { 1 } else { 0 }),
        1 => Some(1),
        _ => base.checked_pow(u32::try_from(exp).ok()?),
    }
}

/// For hexagon-shaped lattices (`dims.len() == 3`) igraph computes row counts,
/// lengths and starts from sums of the sizes without overflow checks. Rejects
/// sizes for which `2 * (d0 + d1 + d2) + 4`, a bound on all of them, overflows.
fn check_hex_shape(dims: &[igraph_int_t], what: &str) -> Result<()> {
    if dims.len() == 3 {
        dims.iter()
            .try_fold(0 as igraph_int_t, |acc, &d| acc.checked_add(d))
            .and_then(|sum| sum.checked_mul(2))
            .and_then(|b| b.checked_add(4))
            .ok_or_else(|| {
                Error::invalid(format!("Lattice dimensions {dims:?} too large for {what}."))
            })?;
    }
    Ok(())
}

/// Converts a slice of counts into an owned igraph integer vector.
fn int_vector(values: &[usize], what: &str) -> Result<VectorInt> {
    values
        .iter()
        .map(|&v| int(v, what))
        .collect::<Result<Vec<_>>>()
        .map(VectorInt::from)
}

/// In igraph 1.0.0 and 1.0.1, `igraph_star` (and thus `igraph_wheel`) builds the graph
/// from its edge list with no explicit vertex count, so `n = 1` yields the
/// null graph. Adds the missing vertices, if any.
fn fix_single_vertex(mut g: Graph, n: igraph_int_t) -> Result<Graph> {
    let missing = usize::try_from(n).unwrap_or(0).saturating_sub(g.vcount());
    if missing > 0 {
        g.add_vertices(missing)?;
    }
    Ok(g)
}

/// Which kinds of edges a degree sequence realization may use
/// (`igraph_edge_type_sw_t` flags, including their combination), taken by
/// [`Graph::realize_degree_sequence`] and
/// [`Graph::realize_bipartite_degree_sequence`].
///
/// Re-exported from [`constants`](crate::constants::AllowedEdgeTypes); it is
/// the same type as the one taken by
/// [`is_graphical`](crate::mixing::is_graphical), so the same value can first
/// test and then realize a degree sequence:
///
/// ```
/// use igraph::{constructors::AllowedEdgeTypes, mixing::is_graphical, prelude::*};
/// let degrees = [4, 2];
/// for allowed in [AllowedEdgeTypes::MULTI, AllowedEdgeTypes::ALL] {
///     let graphical = is_graphical(&degrees, None, allowed).unwrap();
///     let realized =
///         Graph::realize_degree_sequence(&degrees, None, allowed, RealizeDegseq::Smallest);
///     assert_eq!(graphical, realized.is_ok());
/// }
/// // Without self-loops, vertex 0 cannot place 4 stubs on its only neighbour's 2.
/// assert!(!is_graphical(&degrees, None, AllowedEdgeTypes::MULTI).unwrap());
/// ```
pub use crate::constants::AllowedEdgeTypes;

macro_rules! famous_graphs {
    ($( $(#[$meta:meta])* $variant:ident = $name:literal, $n:literal, $m:literal; )+) => {
        /// The named graphs known to [`Graph::famous`].
        ///
        /// Each variant documents its size; [`FamousGraph::name`] gives the
        /// name understood by igraph, and [`FamousGraph::ALL`] lists them all.
        /// `FamousGraph` implements `AsRef<str>`, so it can be passed to
        /// [`Graph::famous`] directly:
        ///
        /// ```
        /// use igraph::{constructors::FamousGraph, prelude::*};
        /// let g = Graph::famous(FamousGraph::Heawood).unwrap();
        /// assert_eq!((g.vcount(), g.ecount()), FamousGraph::Heawood.size());
        /// ```
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum FamousGraph {
            $( $(#[$meta])* $variant ),+
        }

        impl FamousGraph {
            /// All the famous graphs, in the order of the igraph documentation
            /// (alphabetical, except that `Noperfectmatching` precedes `Nonline`).
            pub const ALL: &'static [FamousGraph] = &[$(FamousGraph::$variant),+];

            /// The name igraph uses for this graph (matching is case insensitive).
            pub fn name(self) -> &'static str {
                match self { $(FamousGraph::$variant => $name),+ }
            }

            /// The `(vertex count, edge count)` of this graph.
            pub fn size(self) -> (usize, usize) {
                match self { $(FamousGraph::$variant => ($n, $m)),+ }
            }
        }
    };
}

famous_graphs! {
    /// The bull graph: a triangle with two pendant "horns" (5 vertices, 5 edges).
    Bull = "Bull", 5, 5;
    /// The Chvátal graph: the smallest triangle-free, 4-chromatic, 4-regular graph (12, 24).
    Chvatal = "Chvatal", 12, 24;
    /// The Coxeter graph: a non-Hamiltonian cubic symmetric graph (28, 42).
    Coxeter = "Coxeter", 28, 42;
    /// The skeleton of the cube (8, 12).
    Cubical = "Cubical", 8, 12;
    /// The diamond: two triangles sharing an edge (4, 5).
    Diamond = "Diamond", 4, 5;
    /// The skeleton of the dodecahedron (20, 30).
    Dodecahedron = "Dodecahedron", 20, 30;
    /// The Folkman graph: the smallest semisymmetric graph (20, 40).
    Folkman = "Folkman", 20, 40;
    /// The Franklin graph, related to colorings of the Klein bottle (12, 18).
    Franklin = "Franklin", 12, 18;
    /// The Frucht graph: the smallest cubic graph with no non-trivial automorphism (12, 18).
    Frucht = "Frucht", 12, 18;
    /// The Grötzsch graph: triangle-free with chromatic number 4 (11, 20).
    Grotzsch = "Grotzsch", 11, 20;
    /// The Heawood graph: the 6-cage, the smallest cubic graph of girth 6 (14, 21).
    Heawood = "Heawood", 14, 21;
    /// The Herschel graph: the smallest non-Hamiltonian polyhedral graph (11, 18).
    Herschel = "Herschel", 11, 18;
    /// The house graph: a triangle on top of a square (5, 6).
    House = "House", 5, 6;
    /// The house graph with an X in the square (5, 8).
    HouseX = "HouseX", 5, 8;
    /// The skeleton of the icosahedron (12, 30).
    Icosahedron = "Icosahedron", 12, 30;
    /// Krackhardt's kite social network (10, 18).
    KrackhardtKite = "Krackhardt_Kite", 10, 18;
    /// The Levi graph: a 4-arc transitive cubic graph (30, 45).
    Levi = "Levi", 30, 45;
    /// The McGee graph: the unique 3-regular 7-cage (24, 36).
    McGee = "McGee", 24, 36;
    /// The Meredith graph: 4-regular, 4-connected and non-Hamiltonian (70, 140).
    Meredith = "Meredith", 70, 140;
    /// A connected graph without a perfect matching (16, 27).
    NoPerfectMatching = "Noperfectmatching", 16, 27;
    /// The disjoint union of the 9 forbidden subgraphs of line graphs (50, 72).
    Nonline = "Nonline", 50, 72;
    /// The skeleton of the octahedron (6, 12).
    Octahedron = "Octahedron", 6, 12;
    /// The Petersen graph: 3-regular, the smallest hypohamiltonian graph (10, 15).
    Petersen = "Petersen", 10, 15;
    /// The Robertson graph: the unique (4,5)-cage (19, 38).
    Robertson = "Robertson", 19, 38;
    /// A smallest non-trivial graph whose automorphism group is cyclic (9, 15).
    SmallestCyclicGroup = "Smallestcyclicgroup", 9, 15;
    /// The skeleton of the tetrahedron, i.e. `K_4` (4, 6).
    Tetrahedron = "Tetrahedron", 4, 6;
    /// The Thomassen graph: the smallest hypotraceable graph (34, 52).
    Thomassen = "Thomassen", 34, 52;
    /// The Tutte graph: a counterexample to Tait's Hamiltonian conjecture (46, 69).
    Tutte = "Tutte", 46, 69;
    /// A triangle-free, uniquely 3-colorable graph (12, 22).
    Uniquely3Colorable = "Uniquely3colorable", 12, 22;
    /// The Walther graph: an identity graph (25, 31).
    Walther = "Walther", 25, 31;
    /// Zachary's karate club social network (34, 78).
    Zachary = "Zachary", 34, 78;
}

impl AsRef<str> for FamousGraph {
    fn as_ref(&self) -> &str {
        self.name()
    }
}

impl std::fmt::Display for FamousGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Checks that `x` is a valid *edge count* for the unweighted adjacency
/// constructors: finite, non-negative and integral.
///
/// igraph converts matrix entries to integers with a plain C cast, which is
/// undefined behaviour for NaN, infinities and out-of-range values, and
/// silently truncates fractions; all of these are rejected here.
fn check_edge_count(x: f64, row: usize, col: usize) -> Result<()> {
    // 2^63: the first double that does not fit into an igraph_int_t.
    const LIMIT: f64 = 9_223_372_036_854_775_808.0;
    if x.is_finite() && (0.0..LIMIT).contains(&x) && x.fract() == 0.0 {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "adjacency matrix entry ({row}, {col}) = {x} is not a non-negative integer edge count"
        )))
    }
}

/// Validates every entry of a dense matrix with [`check_edge_count`].
fn check_edge_counts(matrix: &Matrix) -> Result<()> {
    let nrow = matrix.nrow();
    // Column-major storage: element k is at (k % nrow, k / nrow).
    for (k, &x) in matrix.as_slice().iter().enumerate() {
        check_edge_count(x, k % nrow, k / nrow)?;
    }
    Ok(())
}

/// Whether the sparse routines of `mode` may combine `A[i][j]` with `A[j][i]`
/// while only visiting the stored entries of the upper triangle.
///
/// In igraph 1.0.0 and 1.0.1 these routines (`igraph_sparse_adjacency` in
/// the `MAX` mode, `igraph_sparse_weighted_adjacency` in the `MAX`, `MIN`
/// and `PLUS` modes) skip every stored entry below the diagonal, so an entry
/// `A[i][j]` (`i > j`) whose mirror `A[j][i]` is not stored is silently lost
/// (the dense routines do not have this problem). Storing an explicit zero
/// at the mirror position restores the documented semantics. (The unweighted
/// `MIN` mode loses nothing, since `min(x, 0) = 0` there, and the unweighted
/// `PLUS` mode visits every entry; padding is harmless for both.)
fn needs_mirror_padding(mode: Adjacency) -> bool {
    matches!(mode, Adjacency::Max | Adjacency::Min | Adjacency::Plus)
}

/// Builds the `n x n` column-compressed adjacency matrix for `mode` from
/// triplets, storing exactly the nonzero entries of the matrix they
/// describe: duplicated positions are summed on the Rust side, and positions
/// whose sum is zero are left out. The latter matters for
/// [`Adjacency::Undirected`], where igraph's symmetry test is *structural*
/// (an explicitly stored zero without a stored mirror makes the matrix
/// "non-symmetric"), while the dense routines only compare values.
///
/// When [`needs_mirror_padding`] holds, an explicit zero is then stored at
/// `(j, i)` for every stored `(i, j)` whose mirror position is empty. With
/// `counts`, every given value and every sum must be an edge count (see
/// [`check_edge_count`]).
///
/// No position is stored twice, so the number of edges igraph's weighted
/// routines emit never exceeds the number of nonzero entries they allocate
/// room for.
fn compressed_adjacency(
    n: usize,
    entries: &[(VertexId, VertexId, f64)],
    mode: Adjacency,
    counts: bool,
) -> Result<SparseMat> {
    let n_int = int(n, "the number of vertices")?;
    for &(i, j, _) in entries {
        if !(0..n_int).contains(&i) || !(0..n_int).contains(&j) {
            return Err(Error::new(
                crate::error::ErrorKind::InvalidVertexId,
                format!("matrix entry ({i}, {j}) out of bounds for {n} vertices"),
            ));
        }
    }
    // In bounds (checked above), so all the `as usize` casts below are lossless.
    if counts {
        for &(i, j, x) in entries {
            check_edge_count(x, i as usize, j as usize)?;
        }
    }
    let mut summed: BTreeMap<(VertexId, VertexId), f64> = BTreeMap::new();
    for &(i, j, x) in entries {
        *summed.entry((i, j)).or_insert(0.0) += x;
    }
    // `0.0 == -0.0`; NaN sums are kept, as a dense matrix would keep them.
    summed.retain(|_, x| *x != 0.0);
    if counts {
        for (&(i, j), &x) in &summed {
            check_edge_count(x, i as usize, j as usize)?;
        }
    }
    let padding: Vec<(VertexId, VertexId)> = if needs_mirror_padding(mode) {
        summed
            .keys()
            .filter(|&&(i, j)| i != j && !summed.contains_key(&(j, i)))
            .map(|&(i, j)| (j, i))
            .collect()
    } else {
        Vec::new()
    };
    let stored = summed.len() + padding.len();
    let mut triplet = SparseMat::with_capacity(n, n, stored.max(1))?;
    for (&(i, j), &x) in &summed {
        triplet.entry(i as usize, j as usize, x)?;
    }
    for &(i, j) in &padding {
        triplet.entry(i as usize, j as usize, 0.0)?;
    }
    triplet.compress()
}

/// Deterministic generators, see the [module docs](self).
impl igraph_t {
    /// Creates a graph from an adjacency matrix.
    ///
    /// Row/column `i` of the square matrix becomes vertex `i`; entries are
    /// *edge counts* (non-negative integers), interpreted according to
    /// `mode` (`A[i][j]` is the element in row `i`, column `j`):
    ///
    /// - [`Adjacency::Directed`]: directed graph with `A[i][j]` edges `i -> j`;
    /// - [`Adjacency::Undirected`]: undirected graph, the matrix must be symmetric;
    /// - [`Adjacency::Max`] / [`Adjacency::Min`] / [`Adjacency::Plus`]:
    ///   `max`, `min` or sum of `A[i][j]` and `A[j][i]` undirected edges;
    /// - [`Adjacency::Upper`] / [`Adjacency::Lower`]: only the upper / lower
    ///   triangle (diagonal included) is used.
    ///
    /// `loops` says how the diagonal is read: [`Loops::None`] ignores it,
    /// [`Loops::Once`] reads it as the number of self-loops, [`Loops::Twice`]
    /// as *twice* that number (the usual degree convention for undirected
    /// graphs; odd values are an error). In the [`Adjacency::Directed`],
    /// [`Adjacency::Upper`] and [`Adjacency::Lower`] modes `Twice` is treated
    /// as `Once`: a directed loop adds one to both the in- and the
    /// out-degree, and a triangle only holds the diagonal once. Edge
    /// ordering is not specified.
    ///
    /// Binds [`igraph_adjacency`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_adjacency).
    /// Time complexity: O(|V|²).
    ///
    /// See also [`Graph::get_adjacency`], the inverse conversion, and
    /// [`sparse_adjacency`](Self::sparse_adjacency) for large sparse graphs.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for a
    /// non-square matrix, entries that are not non-negative integers
    /// (negative, fractional, NaN or infinite; checked on the Rust side, since
    /// igraph would truncate them with an unchecked cast), a non-symmetric
    /// matrix with [`Adjacency::Undirected`], or an odd diagonal with
    /// [`Loops::Twice`] in the modes that honour it.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let a = Matrix::from_rows(&[[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 2.0]]).unwrap();
    /// let g = Graph::adjacency(&a, Adjacency::Undirected, Loops::Twice).unwrap();
    /// // Two edges plus one self-loop on vertex 2 (its diagonal entry counts it twice).
    /// assert_eq!(g.ecount(), 3);
    /// assert_eq!(g.degree_of(2, NeighborMode::All, Loops::Twice).unwrap(), 3);
    /// // `get_adjacency` gives the matrix back (with the same loop convention).
    /// assert_eq!(g.get_adjacency(GetAdjacency::Both, None, Loops::Twice).unwrap(), a);
    /// ```
    pub fn adjacency(matrix: &Matrix, mode: Adjacency, loops: Loops) -> Result<Graph> {
        check_edge_counts(matrix)?;
        Graph::init_with(|g| unsafe { igraph_adjacency(g, matrix, mode.into(), loops.into()) })
    }

    /// Creates a weighted graph from a weighted adjacency matrix, returning
    /// the graph and its edge weights (indexed by edge id).
    ///
    /// Zero entries mean "no edge" (negative weights are allowed). The modes
    /// are as in [`adjacency`](Self::adjacency), except that at most one edge
    /// is created per vertex pair: [`Adjacency::Undirected`] requires a
    /// symmetric matrix, and `Max`/`Min`/`Plus` combine the two weights
    /// `A[i][j]` and `A[j][i]` into the weight of a single edge. For the
    /// diagonal, [`Loops::None`] ignores it, [`Loops::Once`] takes the entry
    /// as the loop weight and [`Loops::Twice`] as twice the loop weight
    /// (halving it; `Twice` is treated as `Once` in the
    /// [`Adjacency::Directed`], [`Adjacency::Upper`] and [`Adjacency::Lower`]
    /// modes).
    ///
    /// Binds [`igraph_weighted_adjacency`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_weighted_adjacency).
    /// Time complexity: O(|V|²).
    ///
    /// See also [`Graph::get_adjacency`] with `weights`, the inverse conversion.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for a
    /// non-square matrix, or a non-symmetric one with [`Adjacency::Undirected`].
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let a = Matrix::from_rows(&[[0.0, 2.5, 0.0], [0.0, 0.0, -1.0], [4.0, 0.0, 0.0]]).unwrap();
    /// let (g, w) = Graph::weighted_adjacency(&a, Adjacency::Directed, Loops::None).unwrap();
    /// let mut edges: Vec<_> = g.edge_list().into_iter().zip(w).collect();
    /// edges.sort_by_key(|e| e.0);
    /// assert_eq!(edges, vec![((0, 1), 2.5), ((1, 2), -1.0), ((2, 0), 4.0)]);
    /// ```
    pub fn weighted_adjacency(
        matrix: &Matrix,
        mode: Adjacency,
        loops: Loops,
    ) -> Result<(Graph, Vec<f64>)> {
        let mut weights = Vector::new();
        let g = Graph::init_with(|g| unsafe {
            igraph_weighted_adjacency(g, matrix, mode.into(), &mut weights, loops.into())
        })?;
        Ok((g, weights.into()))
    }

    /// Creates a graph on `n` vertices from a *sparse* adjacency matrix given
    /// as `(row, column, value)` triplets.
    ///
    /// This is the sparse counterpart of [`adjacency`](Self::adjacency), with
    /// the same `mode` and `loops` semantics and the same result; entries not
    /// listed are zero, and repeated `(row, column)` pairs are summed. The
    /// triplets are summed up and compressed into igraph's column-compressed
    /// sparse matrix before the call.
    ///
    /// igraph 1.0.0 and 1.0.1 lose the entries below the diagonal whose
    /// mirror entry is absent in the [`Adjacency::Max`] mode (and in the
    /// `Max`/`Min`/`Plus` modes of the weighted variant), and their
    /// [`Adjacency::Undirected`] mode rejects an explicitly stored zero
    /// without a stored mirror as non-symmetric. This wrapper stores only the
    /// nonzero sums, plus explicit zeros at the missing mirror positions where
    /// needed, so that the result always agrees with the dense
    /// [`adjacency`](Self::adjacency).
    ///
    /// Binds [`igraph_sparse_adjacency`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_sparse_adjacency).
    /// Time complexity: O(|E|), plus O(t log t) for summing up the `t`
    /// triplets.
    ///
    /// See also [`Graph::get_adjacency_sparse`], whose
    /// [`entries`](crate::conversion::CooMatrix::entries) are exactly the
    /// triplets this function takes.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) if a
    /// triplet lies outside the `n × n` matrix;
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if a value
    /// (or the sum of the values given for one position) is not a
    /// non-negative integer; otherwise as
    /// [`adjacency`](Self::adjacency).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A directed 1000-cycle, without ever allocating a dense 1000 x 1000 matrix.
    /// let n = 1000;
    /// let entries: Vec<_> = (0..n as i64).map(|i| (i, (i + 1) % n as i64, 1.0)).collect();
    /// let g = Graph::sparse_adjacency(n, &entries, Adjacency::Directed, Loops::None).unwrap();
    /// assert_eq!((g.vcount(), g.ecount()), (1000, 1000));
    /// assert!(g.is_directed());
    ///
    /// // Round trip through the sparse adjacency matrix of the conversion module.
    /// let karate = Graph::famous("Zachary").unwrap();
    /// let coo = karate.get_adjacency_sparse(GetAdjacency::Upper, None, Loops::Twice).unwrap();
    /// let back = Graph::sparse_adjacency(34, &coo.entries, Adjacency::Upper, Loops::Twice).unwrap();
    /// assert!(back.isomorphic(&karate).unwrap());
    /// assert_eq!(back.ecount(), 78);
    /// ```
    pub fn sparse_adjacency(
        n: usize,
        entries: &[(VertexId, VertexId, f64)],
        mode: Adjacency,
        loops: Loops,
    ) -> Result<Graph> {
        let mut sparse = compressed_adjacency(n, entries, mode, true)?;
        Graph::init_with(|g| unsafe {
            igraph_sparse_adjacency(g, &mut sparse, mode.into(), loops.into())
        })
    }

    /// Creates a weighted graph on `n` vertices from a sparse weighted
    /// adjacency matrix given as `(row, column, weight)` triplets, returning
    /// the graph and its edge weights.
    ///
    /// The sparse counterpart of [`weighted_adjacency`](Self::weighted_adjacency);
    /// repeated `(row, column)` pairs are summed, explicit zeros mean no edge,
    /// and the result agrees with the dense version (see the note about
    /// mirror entries in [`sparse_adjacency`](Self::sparse_adjacency)).
    ///
    /// Binds [`igraph_sparse_weighted_adjacency`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_sparse_weighted_adjacency).
    /// Time complexity: O(|E|), plus O(t log t) for summing up the `t`
    /// triplets.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) if a
    /// triplet lies outside the `n × n` matrix; otherwise as
    /// [`weighted_adjacency`](Self::weighted_adjacency) (any real weight is
    /// accepted).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let entries = [(0, 1, 0.5), (1, 0, 0.5), (1, 2, 3.0), (2, 1, 3.0)];
    /// let (g, w) =
    ///     Graph::sparse_weighted_adjacency(3, &entries, Adjacency::Undirected, Loops::None)
    ///         .unwrap();
    /// assert_eq!(g.ecount(), 2);
    /// assert_eq!(w.iter().sum::<f64>(), 3.5);
    /// ```
    pub fn sparse_weighted_adjacency(
        n: usize,
        entries: &[(VertexId, VertexId, f64)],
        mode: Adjacency,
        loops: Loops,
    ) -> Result<(Graph, Vec<f64>)> {
        let mut sparse = compressed_adjacency(n, entries, mode, false)?;
        let mut weights = Vector::new();
        let g = Graph::init_with(|g| unsafe {
            igraph_sparse_weighted_adjacency(
                g,
                &mut sparse,
                mode.into(),
                &mut weights,
                loops.into(),
            )
        })?;
        Ok((g, weights.into()))
    }

    /// Shorthand to create a small graph from a flat edge list
    /// `[from0, to0, from1, to1, ...]`, in the spirit of `igraph_small`.
    ///
    /// The C function `igraph_small` is variadic and terminated by `-1`; in
    /// Rust a slice literal does the same job safely. The graph has
    /// `max(n, largest id + 1)` vertices. This is the same as
    /// [`Graph::from_flat_edges`], provided for familiarity with the C API;
    /// [`Graph::from_edges`] takes `(from, to)` pairs instead.
    ///
    /// Binds [`igraph_create`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_create)
    /// (the non-variadic equivalent of
    /// [`igraph_small`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_small)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for an odd
    /// number of ids, [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId)
    /// for negative ids.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let bowtie = Graph::small(5, false, &[0, 1, 1, 2, 2, 0, 2, 3, 3, 4, 4, 2]).unwrap();
    /// assert_eq!(bowtie.degree_of(2, NeighborMode::All, Loops::Twice).unwrap(), 4);
    /// ```
    pub fn small(n: usize, directed: bool, edges: &[VertexId]) -> Result<Graph> {
        Graph::from_flat_edges(edges, n, directed)
    }

    /// Creates a star graph: vertex `center` connected to all the other
    /// `n - 1` vertices.
    ///
    /// `mode` chooses between an undirected star ([`StarMode::Undirected`]),
    /// edges pointing out of ([`StarMode::Out`]) or into ([`StarMode::In`])
    /// the center, or mutual directed edges ([`StarMode::Mutual`]).
    ///
    /// Binds [`igraph_star`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_star).
    /// Time complexity: O(|V|).
    ///
    /// See also [`wheel`](Self::wheel), and [`Graph::layout_star`] to draw it.
    ///
    /// The star on a single vertex is that vertex alone (igraph 1.0.0 and 1.0.1
    /// return the null graph there; this wrapper adds the missing vertex).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `center`
    /// is not a vertex of the graph, in particular always for `n = 0`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let s = Graph::star(5, StarMode::In, 2).unwrap();
    /// assert_eq!(s.degree_of(2, NeighborMode::In, Loops::Twice).unwrap(), 4);
    /// assert_eq!(s.degree_of(2, NeighborMode::Out, Loops::Twice).unwrap(), 0);
    /// ```
    pub fn star(n: usize, mode: StarMode, center: VertexId) -> Result<Graph> {
        let n = int(n, "the number of vertices")?;
        let g = Graph::init_with(|g| unsafe { igraph_star(g, n, mode.into(), center) })?;
        fix_single_vertex(g, n)
    }

    /// Creates a wheel graph: a star (the spokes) plus a cycle through the
    /// `n - 1` non-center vertices (the rim).
    ///
    /// `mode` orients the spokes like in [`star`](Self::star); in the directed
    /// modes the rim is a directed cycle (mutual for [`WheelMode::Mutual`]).
    /// Note that the wheels on 2 and 3 vertices are not simple (they contain a
    /// self-loop and a multi-edge, respectively).
    ///
    /// Binds [`igraph_wheel`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_wheel).
    /// Time complexity: O(|V|).
    ///
    /// The wheel on one vertex is the singleton graph.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `center`
    /// is not a vertex of the graph (always for `n = 0`).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let w = Graph::wheel(6, WheelMode::Undirected, 0).unwrap();
    /// assert_eq!(w.ecount(), 10); // 5 spokes + 5 rim edges
    /// ```
    pub fn wheel(n: usize, mode: WheelMode, center: VertexId) -> Result<Graph> {
        let n = int(n, "the number of vertices")?;
        let g = Graph::init_with(|g| unsafe { igraph_wheel(g, n, mode.into(), center) })?;
        fix_single_vertex(g, n)
    }

    /// The `dim`-dimensional hypercube graph `Q_dim`.
    ///
    /// It has `2^dim` vertices and `dim * 2^(dim-1)` edges; two vertices are
    /// adjacent when the binary representations of their ids differ in
    /// exactly one bit. Directed edges point from lower to higher ids.
    ///
    /// Binds [`igraph_hypercube`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_hypercube).
    /// Time complexity: O(2^dim).
    ///
    /// See also [`square_lattice`](Self::square_lattice): `Q_dim` is the
    /// `2 x 2 x ... x 2` lattice.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `dim > 57`, so that the edge count would not fit into half the range
    /// of `igraph_int_t` (smaller but still huge dimensions fail with an
    /// out-of-memory error).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let q3 = Graph::hypercube(3, false).unwrap();
    /// assert_eq!((q3.vcount(), q3.ecount()), (8, 12));
    /// assert_eq!(q3.neighbors(0, NeighborMode::All).unwrap(), vec![1, 2, 4]);
    /// assert!(q3.isomorphic(&Graph::famous("Cubical").unwrap()).unwrap());
    /// ```
    pub fn hypercube(dim: usize, directed: bool) -> Result<Graph> {
        let dim = int(dim, "the dimension")?;
        Graph::init_with(|g| unsafe { igraph_hypercube(g, dim, directed) })
    }

    /// Creates an arbitrary-dimensional square lattice (grid).
    ///
    /// `dims` gives the size along each dimension (an empty slice gives the
    /// singleton graph). The vertex at position `(i_1, i_2, ..., i_d)` gets id
    /// `i_1 + n_1 * i_2 + n_1 * n_2 * i_3 + ...`. Vertices within `nei` steps
    /// of each other are connected (`nei = 1` is the usual grid).
    /// `periodic`, when given, must have one flag per dimension, making the
    /// lattice wrap around (a torus) along the flagged dimensions.
    /// When `directed`, edges point from lower to higher ids unless `mutual`
    /// (or periodicity) is set.
    ///
    /// Binds [`igraph_square_lattice`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_square_lattice).
    /// Time complexity: O(|V| + |E|) for `nei < 2`.
    ///
    /// See also [`Graph::layout_grid`], which places a 2D lattice on its grid
    /// (pass `Some(dims[0])` as the width), and
    /// [`triangular_lattice`](Self::triangular_lattice) /
    /// [`hexagonal_lattice`](Self::hexagonal_lattice).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `periodic` and `dims` have different lengths.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A 4x4 torus is 4-regular.
    /// let torus = Graph::square_lattice(&[4, 4], 1, false, false, Some(&[true, true])).unwrap();
    /// assert_eq!(torus.ecount(), 32);
    /// let deg = torus.degree(VertexSelector::All, NeighborMode::All, Loops::Twice).unwrap();
    /// assert!(deg.iter().all(|&d| d == 4));
    /// // Vertex (x, y) of a 5 x 3 grid has id x + 5 y, which `layout_grid` puts at (x, y).
    /// let grid = Graph::square_lattice(&[5, 3], 1, false, false, None).unwrap();
    /// let layout = grid.layout_grid(Some(5)).unwrap();
    /// assert_eq!((layout[(7, 0)], layout[(7, 1)]), (2.0, 1.0));
    /// ```
    pub fn square_lattice(
        dims: &[usize],
        nei: usize,
        directed: bool,
        mutual: bool,
        periodic: Option<&[bool]>,
    ) -> Result<Graph> {
        if let Some(p) = periodic
            && p.len() != dims.len()
        {
            return Err(Error::invalid(format!(
                "periodic has {} flags but the lattice has {} dimensions",
                p.len(),
                dims.len()
            )));
        }
        let dims = int_vector(dims, "a lattice dimension")?;
        let nei = int(nei, "the neighborhood order")?;
        let periodic = periodic.map(VectorBool::view);
        let periodic_ptr = periodic.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        Graph::init_with(|g| unsafe {
            igraph_square_lattice(g, &dims, nei, directed, mutual, periodic_ptr)
        })
    }

    /// Creates a cycle graph `C_n` (`circular = true`) or a path graph `P_n`
    /// (`circular = false`).
    ///
    /// In directed graphs all edges follow the same orientation along the
    /// ring, or are mutual when `mutual` is set (ignored when undirected).
    /// For `n` = 1 or 2 the cycle is not simple (a self-loop, or two parallel
    /// edges).
    ///
    /// Binds [`igraph_ring`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_ring).
    /// Time complexity: O(|V|).
    ///
    /// See also [`circulant`](Self::circulant) for rings with chords, and
    /// [`Graph::layout_circle`] to draw it.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let ring = Graph::ring(5, true, false, true).unwrap();
    /// assert_eq!(ring.edge_list(), vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]);
    /// ```
    pub fn ring(n: usize, directed: bool, mutual: bool, circular: bool) -> Result<Graph> {
        let n = int(n, "the number of vertices")?;
        Graph::init_with(|g| unsafe { igraph_ring(g, n, directed, mutual, circular) })
    }

    /// The path graph `P_n` on `n` vertices: `0 - 1 - ... - (n-1)`.
    ///
    /// A convenience form of [`ring`](Self::ring) with `circular = false`;
    /// `mutual` adds both directions in directed graphs.
    ///
    /// Binds [`igraph_path_graph`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_path_graph).
    /// Time complexity: O(|V|).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let p = Graph::path_graph(4, true, true).unwrap();
    /// assert_eq!(p.ecount(), 6); // 3 links, both directions
    /// ```
    pub fn path_graph(n: usize, directed: bool, mutual: bool) -> Result<Graph> {
        let n = int(n, "the number of vertices")?;
        Graph::init_with(|g| unsafe { igraph_path_graph(g, n, directed, mutual) })
    }

    /// The cycle graph `C_n` on `n` vertices.
    ///
    /// A convenience form of [`ring`](Self::ring) with `circular = true`. For
    /// `n` = 1 or 2 the result has a self-loop or parallel edges.
    ///
    /// Binds [`igraph_cycle_graph`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_cycle_graph).
    /// Time complexity: O(|V|).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let c = Graph::cycle_graph(7, false, false).unwrap();
    /// assert_eq!((c.vcount(), c.ecount()), (7, 7));
    /// ```
    pub fn cycle_graph(n: usize, directed: bool, mutual: bool) -> Result<Graph> {
        let n = int(n, "the number of vertices")?;
        Graph::init_with(|g| unsafe { igraph_cycle_graph(g, n, directed, mutual) })
    }

    /// Creates a `children`-ary tree on `n` vertices, filled level by level
    /// in breadth-first order (vertex `i`'s children are
    /// `children*i + 1 ..= children*i + children`).
    ///
    /// For a complete tree with `l` levels below the root use
    /// `n = (children^(l+1) - 1) / (children - 1)`. `mode` gives the edge
    /// orientation (parent → child for [`TreeMode::Out`]). `n = 0` gives the
    /// null graph.
    ///
    /// Binds [`igraph_kary_tree`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_kary_tree).
    /// Time complexity: O(|V| + |E|).
    ///
    /// See also [`Graph::tree_game`] for uniformly random trees and
    /// [`Graph::layout_reingold_tilford`] to draw trees.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `children` is zero.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let t = Graph::kary_tree(7, 2, TreeMode::Out).unwrap(); // a complete binary tree
    /// assert_eq!(t.neighbors(1, NeighborMode::Out).unwrap(), vec![3, 4]);
    /// assert!(t.is_tree(NeighborMode::Out).unwrap());
    /// ```
    pub fn kary_tree(n: usize, children: usize, mode: TreeMode) -> Result<Graph> {
        let n = int(n, "the number of vertices")?;
        let children = int(children, "the number of children")?;
        Graph::init_with(|g| unsafe { igraph_kary_tree(g, n, children, mode.into()) })
    }

    /// Creates a symmetric tree where every vertex at distance `d` from the
    /// root has `branches[d]` children.
    ///
    /// The tree has `1 + b_0 + b_0 b_1 + ...` vertices, numbered in
    /// breadth-first order from the root `0`.
    ///
    /// Binds [`igraph_symmetric_tree`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_symmetric_tree).
    /// Time complexity: O(|V| + |E|).
    ///
    /// An empty `branches` gives the singleton graph.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if a
    /// branch count is zero.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let t = Graph::symmetric_tree(&[3, 2], TreeMode::Undirected).unwrap();
    /// assert_eq!(t.vcount(), 1 + 3 + 3 * 2);
    /// ```
    pub fn symmetric_tree(branches: &[usize], mode: TreeMode) -> Result<Graph> {
        let branches = int_vector(branches, "a branching count")?;
        Graph::init_with(|g| unsafe { igraph_symmetric_tree(g, &branches, mode.into()) })
    }

    /// Creates a regular tree (Bethe lattice) of height `h` in which every
    /// non-leaf vertex has total degree `k`.
    ///
    /// Unlike a [`kary_tree`](Self::kary_tree), the root has `k` children and
    /// the other internal vertices `k - 1`, so that all internal degrees are
    /// equal. `h` is the distance between the root and the leaves.
    ///
    /// Binds [`igraph_regular_tree`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_regular_tree).
    /// Time complexity: O(|V| + |E|).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) unless
    /// `h >= 1` and `k >= 2`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let t = Graph::regular_tree(2, 3, TreeMode::Undirected).unwrap();
    /// assert_eq!(t.vcount(), 1 + 3 + 3 * 2);
    /// ```
    pub fn regular_tree(h: usize, k: usize, mode: TreeMode) -> Result<Graph> {
        let h = int(h, "the height")?;
        let k = int(k, "the degree")?;
        Graph::init_with(|g| unsafe { igraph_regular_tree(g, h, k, mode.into()) })
    }

    /// Builds a tree or forest from a parent vector: `parents[v]` is the
    /// parent of vertex `v`, or a negative value if `v` is a root.
    ///
    /// Such vectors are produced by BFS/DFS traversals, shortest path trees,
    /// dominator trees, etc. The graph has `parents.len()` vertices; with
    /// [`TreeMode::Out`] edges point from parents to children, with
    /// [`TreeMode::In`] from children to parents.
    ///
    /// Binds [`igraph_tree_from_parent_vector`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_tree_from_parent_vector).
    /// Time complexity: O(n).
    ///
    /// See also [`Graph::bfs_simple`] and the other traversals of
    /// [`crate::visitor`], whose `parents` give such vectors (map `None` to `-1`).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the
    /// vector encodes a cycle or a self-loop,
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) for
    /// out-of-range parents.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // Two trees: 0 <- {1, 2}, 2 <- 3, and the lone root 4.
    /// let f = Graph::tree_from_parent_vector(&[-1, 0, 0, 2, -1], TreeMode::Out).unwrap();
    /// assert_eq!(f.ecount(), 3);
    /// assert_eq!(f.neighbors(0, NeighborMode::Out).unwrap(), vec![1, 2]);
    ///
    /// // The BFS tree of the Petersen graph: 1 root, 3 children, 6 grandchildren.
    /// let petersen = Graph::famous("Petersen").unwrap();
    /// let bfs = petersen.bfs_simple(0, NeighborMode::All).unwrap();
    /// let parents: Vec<i64> = bfs.parents.iter().map(|p| p.unwrap_or(-1)).collect();
    /// let tree = Graph::tree_from_parent_vector(&parents, TreeMode::Out).unwrap();
    /// assert!(tree.is_tree(NeighborMode::Out).unwrap());
    /// assert_eq!(tree.ecount(), 9);
    /// ```
    pub fn tree_from_parent_vector(parents: &[VertexId], mode: TreeMode) -> Result<Graph> {
        let parents = VectorInt::view(parents);
        Graph::init_with(|g| unsafe {
            igraph_tree_from_parent_vector(g, parents.as_ptr(), mode.into())
        })
    }

    /// Builds the labelled tree encoded by a Prüfer sequence.
    ///
    /// A sequence of length `n - 2` with entries in `0..n` encodes a unique
    /// tree on `n` vertices (Cayley's formula counts `n^(n-2)` of them); a
    /// vertex appears in the sequence exactly `degree - 1` times.
    ///
    /// Binds [`igraph_from_prufer`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_from_prufer).
    /// Time complexity: O(|V|).
    ///
    /// See also [`Graph::to_prufer`], the inverse conversion.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for an
    /// invalid sequence (entries out of range).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let t = Graph::from_prufer(&[3, 3, 3]).unwrap(); // the star K_{1,4} centered at 3
    /// assert_eq!(t.degree_of(3, NeighborMode::All, Loops::Twice).unwrap(), 4);
    /// assert_eq!(t.to_prufer().unwrap(), vec![3, 3, 3]);
    /// ```
    pub fn from_prufer(prufer: &[VertexId]) -> Result<Graph> {
        let prufer = VectorInt::view(prufer);
        Graph::init_with(|g| unsafe { igraph_from_prufer(g, prufer.as_ptr()) })
    }

    /// Creates the complete graph on `n` vertices.
    ///
    /// Directed complete graphs have both `i -> j` and `j -> i`; with `loops`
    /// every vertex also gets a single self-loop.
    ///
    /// Binds [`igraph_full`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_full).
    /// Time complexity: O(|V|²).
    ///
    /// See also [`Graph::is_complete`], [`Graph::complementer`] (the
    /// complement of `K_n` is the empty graph) and [`Graph::full_bipartite`].
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// assert_eq!(Graph::full(5, false, false).unwrap().ecount(), 10);
    /// assert_eq!(Graph::full(5, true, false).unwrap().ecount(), 20);
    /// assert_eq!(Graph::full(5, false, true).unwrap().ecount(), 15);
    /// ```
    pub fn full(n: usize, directed: bool, loops: bool) -> Result<Graph> {
        let n = int(n, "the number of vertices")?;
        Graph::init_with(|g| unsafe { igraph_full(g, n, directed, loops) })
    }

    /// Creates a complete multipartite graph with partitions of the given
    /// `sizes`, returning the graph and the partition index of each vertex.
    ///
    /// Vertices are numbered partition by partition; every pair of vertices
    /// in different partitions is connected. In directed graphs, `mode`
    /// [`NeighborMode::Out`] points edges from lower to higher partitions,
    /// [`NeighborMode::In`] the opposite, and [`NeighborMode::All`] creates
    /// mutual edges.
    ///
    /// Binds [`igraph_full_multipartite`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_full_multipartite).
    /// Time complexity: O(|V| + |E|).
    ///
    /// See also [`Graph::full_bipartite`] for two partitions with boolean
    /// vertex types, and [`turan`](Self::turan).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let (k233, types) =
    ///     Graph::full_multipartite(&[2, 3, 3], false, NeighborMode::All).unwrap();
    /// assert_eq!(k233.ecount(), 2 * 3 + 2 * 3 + 3 * 3);
    /// assert_eq!(types, vec![0, 0, 1, 1, 1, 2, 2, 2]);
    /// ```
    pub fn full_multipartite(
        sizes: &[usize],
        directed: bool,
        mode: NeighborMode,
    ) -> Result<(Graph, Vec<i64>)> {
        let sizes = int_vector(sizes, "a partition size")?;
        let mut types = VectorInt::new();
        let g = Graph::init_with(|g| unsafe {
            igraph_full_multipartite(g, &mut types, &sizes, directed, mode.into())
        })?;
        Ok((g, types.into()))
    }

    /// Creates the Turán graph `T(n, r)`, returning the graph and the
    /// partition index of each vertex.
    ///
    /// It is the complete `r`-partite graph on `n` vertices with partition
    /// sizes as equal as possible: by Turán's theorem, the densest graph on
    /// `n` vertices without a clique of size `r + 1`. It is undirected; `n = 0`
    /// gives the null graph, and `r > n` gives the complete graph.
    ///
    /// Binds [`igraph_turan`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_turan).
    /// Time complexity: O(|V| + |E|).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `r` is zero.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let (t, types) = Graph::turan(6, 3).unwrap(); // the octahedron K_{2,2,2}
    /// assert_eq!(t.ecount(), 12);
    /// assert_eq!(types, vec![0, 0, 1, 1, 2, 2]);
    /// // Turán's theorem: no clique on r + 1 = 4 vertices.
    /// assert_eq!(t.clique_number().unwrap(), 3);
    /// ```
    pub fn turan(n: usize, r: usize) -> Result<(Graph, Vec<i64>)> {
        let n = int(n, "the number of vertices")?;
        let r = int(r, "the number of partitions")?;
        let mut types = VectorInt::new();
        let g = Graph::init_with(|g| unsafe { igraph_turan(g, &mut types, n, r) })?;
        Ok((g, types.into()))
    }

    /// Creates a full citation graph: the complete directed acyclic graph in
    /// which `i -> j` is an edge exactly when `j < i`.
    ///
    /// With `directed = false` it is just the complete graph.
    ///
    /// Binds [`igraph_full_citation`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_full_citation).
    /// Time complexity: O(|V|²).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::full_citation(4, true).unwrap();
    /// assert_eq!(g.neighbors(3, NeighborMode::Out).unwrap(), vec![0, 1, 2]);
    /// assert!(g.neighbors(0, NeighborMode::Out).unwrap().is_empty());
    /// ```
    pub fn full_citation(n: usize, directed: bool) -> Result<Graph> {
        let n = int(n, "the number of vertices")?;
        Graph::init_with(|g| unsafe { igraph_full_citation(g, n, directed) })
    }

    /// Creates graph number `number` of *An Atlas of Graphs* (Read and
    /// Wilson, 1998).
    ///
    /// The atlas holds all 1253 simple undirected unlabelled graphs on 0 to 7
    /// vertices, ordered by number of vertices, then number of edges, then
    /// degree sequence (lexicographically, e.g. 111223 < 112222), then
    /// increasing number of automorphisms. Graphs on 0, 1, ..., 7 vertices
    /// start at numbers 0, 1, 2, 4, 8, 19, 53 and 209.
    ///
    /// Binds [`igraph_atlas`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_atlas).
    /// Time complexity: O(|V| + |E|).
    ///
    /// See also [`Graph::isomorphic`] and [`Graph::canonical_permutation`] to
    /// locate a given small graph in the atlas.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `number > 1252`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let k7 = Graph::atlas(1252).unwrap(); // the last one is K_7
    /// assert_eq!((k7.vcount(), k7.ecount()), (7, 21));
    /// ```
    pub fn atlas(number: usize) -> Result<Graph> {
        let number = int(number, "the atlas number")?;
        Graph::init_with(|g| unsafe { igraph_atlas(g, number) })
    }

    /// Creates an extended chordal ring: a cycle on `nodes` vertices plus
    /// chords described by the rows of `w`.
    ///
    /// For each row `L` of length `p` (all rows have the same length, which
    /// must divide `nodes`), vertex `i` is connected to vertex
    /// `(i + L[i mod p]) mod nodes`. Entries may be negative. The result is
    /// not simplified: duplicate chords (and chords along the cycle) produce
    /// multi-edges, and offsets that are multiples of `nodes` self-loops.
    /// Note that igraph's definition differs from the one in Kotsis (1993).
    ///
    /// Binds [`igraph_extended_chordal_ring`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_extended_chordal_ring).
    /// Time complexity: O(|V| + |E|).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `nodes < 3`, rows have different lengths, or the row length does not
    /// divide `nodes`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // One row [3]: each vertex also links 3 steps ahead; on 6 vertices the
    /// // three "diameters" appear twice, giving the utility graph K_{3,3} plus duplicates.
    /// let g = Graph::extended_chordal_ring(6, &[[3]], false).unwrap();
    /// assert_eq!(g.ecount(), 6 + 6);
    /// ```
    pub fn extended_chordal_ring<R: AsRef<[i64]>>(
        nodes: usize,
        w: &[R],
        directed: bool,
    ) -> Result<Graph> {
        let nodes = int(nodes, "the number of vertices")?;
        if nodes < 3 {
            return Err(Error::invalid(format!(
                "an extended chordal ring has at least 3 vertices, got {nodes}"
            )));
        }
        // igraph divides by the number of columns: an empty chord matrix
        // (no rows, or only empty rows) is passed as a 0 x 1 matrix, meaning
        // "no chords". Offsets are reduced modulo `nodes`, which keeps their
        // meaning and avoids the unchecked `i + offset` overflow in C.
        let w = if w.iter().all(|row| row.as_ref().is_empty()) {
            MatrixInt::zeros(0, 1)
        } else {
            let rows: Vec<Vec<i64>> = w
                .iter()
                .map(|row| row.as_ref().iter().map(|&x| x.rem_euclid(nodes)).collect())
                .collect();
            MatrixInt::from_rows(&rows)?
        };
        Graph::init_with(|g| unsafe { igraph_extended_chordal_ring(g, nodes, &w, directed) })
    }

    /// The line graph `L(G)` of this graph: one vertex per edge (edge `i`
    /// becomes vertex `i`).
    ///
    /// For undirected graphs, two vertices of `L(G)` are adjacent when the
    /// corresponding edges share an endpoint (twice, if they share both
    /// endpoints in a multigraph; the single vertex of a self-loop counts as
    /// two endpoints, so a self-loop and an edge incident to it are joined
    /// twice). For directed graphs, `e -> f` is an edge when the target of
    /// `e` is the source of `f`. Self-loops are self-adjacent and get a single
    /// self-loop in `L(G)`, in both cases.
    ///
    /// Binds [`igraph_linegraph`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_linegraph).
    /// Time complexity: O(|V| + |E|).
    ///
    /// See also the other graph transformations of [`crate::operators`].
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // The line graph of the star K_{1,4} is K_4.
    /// let l = Graph::star(5, StarMode::Undirected, 0).unwrap().linegraph().unwrap();
    /// assert_eq!((l.vcount(), l.ecount()), (4, 6));
    /// ```
    pub fn linegraph(&self) -> Result<Graph> {
        Graph::init_with(|l| unsafe { igraph_linegraph(self, l) })
    }

    /// The de Bruijn graph `B(m, n)`: vertices are the `m^n` strings of
    /// length `n` over an alphabet of `m` letters, with an edge `v -> w` when
    /// `w` is obtained by dropping the first letter of `v` and appending one.
    ///
    /// Vertex ids are the strings read as base-`m` numbers. Every vertex has
    /// in- and out-degree `m` (loops included), so the graph has `m^(n+1)`
    /// edges and is Eulerian; its Eulerian circuits spell de Bruijn sequences.
    ///
    /// `B(m, 0)` is the singleton graph and `B(0, n)` for `n > 0` the null
    /// graph.
    ///
    /// Binds [`igraph_de_bruijn`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_de_bruijn).
    /// Time complexity: O(|V| + |E|).
    ///
    /// See also [`Graph::eulerian_cycle`] to spell a de Bruijn sequence (as
    /// in the example below) and [`kautz`](Self::kautz).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the
    /// number of vertices `m^n` does not fit in an `i64`, and
    /// [`ErrorKind::Overflow`](crate::ErrorKind::Overflow) if the `m^(n+1)`
    /// edges do not fit in igraph's edge vector (both checked before calling
    /// igraph).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let b = Graph::de_bruijn(2, 3).unwrap();
    /// assert_eq!((b.vcount(), b.ecount()), (8, 16));
    /// // "011" -> "110" and "111"
    /// assert_eq!(b.neighbors(0b011, NeighborMode::Out).unwrap(), vec![0b110, 0b111]);
    ///
    /// // An Eulerian circuit of B(2, 3) spells a de Bruijn sequence of order 4:
    /// // each vertex appends its last letter, and all 16 4-bit words occur
    /// // exactly once as cyclic substrings.
    /// let walk = b.eulerian_cycle().unwrap();
    /// let seq: Vec<i64> = walk.vertices[1..].iter().map(|v| v % 2).collect();
    /// assert_eq!(seq.len(), 16);
    /// let words: std::collections::BTreeSet<i64> = (0..16)
    ///     .map(|i| (0..4).fold(0, |acc, j| 2 * acc + seq[(i + j) % 16]))
    ///     .collect();
    /// assert_eq!(words.len(), 16);
    /// ```
    pub fn de_bruijn(m: usize, n: usize) -> Result<Graph> {
        let m = int(m, "the alphabet size")?;
        let n = int(n, "the string length")?;
        if n > 0 && m > 0 {
            let nodes = checked_ipow(m, n).ok_or_else(|| {
                Error::invalid(format!(
                    "Parameters ({m}, {n}) too large for De Bruijn graph."
                ))
            })?;
            // igraph then needs `2 * m * m^n` edge endpoints (it reports
            // `IGRAPH_EOVERFLOW` otherwise). Checking it here also keeps
            // `m^n` far below 2^63: for `m` close to `i64::MAX` and `n = 1`,
            // `pow(m, 1)` rounds up to 2^63 as a double, and igraph's
            // conversion back to an integer would be undefined behaviour.
            nodes
                .checked_mul(m)
                .and_then(|e| e.checked_mul(2))
                .ok_or_else(|| {
                    Error::new(
                        crate::error::ErrorKind::Overflow,
                        format!("Parameters ({m}, {n}) too large for De Bruijn graph."),
                    )
                })?;
        }
        Graph::init_with(|g| unsafe { igraph_de_bruijn(g, m, n) })
    }

    /// The Kautz graph `K(m, n)`: vertices are the strings of length `n + 1`
    /// over an alphabet of `m + 1` letters with no two equal consecutive
    /// letters; `v -> w` when `w` is `v` shifted by one letter.
    ///
    /// It has `(m+1) m^n` vertices, each with in- and out-degree `m`, and no
    /// self-loops. Degenerate cases: `K(m, 0)` is the complete directed graph
    /// on `m + 1` vertices, and `K(0, n)` for `n > 0` is the null graph.
    ///
    /// Binds [`igraph_kautz`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_kautz).
    /// Time complexity: roughly O(|V| + |E|).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the
    /// graph would be too large (`m^n` or `(m+1)^(n+1)` does not fit in an
    /// `i64`; checked before calling igraph).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let k = Graph::kautz(2, 1).unwrap();
    /// assert_eq!((k.vcount(), k.ecount()), (6, 12));
    /// ```
    pub fn kautz(m: usize, n: usize) -> Result<Graph> {
        let m = int(m, "m")?;
        let n = int(n, "n")?;
        // `K(m, 0)` is the complete digraph on `m + 1` vertices; otherwise igraph
        // needs `m^n` and `(m + 1)^(n + 1)`.
        let fits = match (m, n) {
            (_, 0) => m.checked_add(1).is_some(),
            (0, _) => true,
            _ => {
                checked_ipow(m, n).is_some()
                    && m.checked_add(1)
                        .zip(n.checked_add(1))
                        .and_then(|(b, e)| checked_ipow(b, e))
                        .is_some()
            }
        };
        if !fits {
            return Err(Error::invalid(format!(
                "Parameters ({m}, {n}) too large for Kautz graph."
            )));
        }
        Graph::init_with(|g| unsafe { igraph_kautz(g, m, n) })
    }

    /// The circulant graph `C_n(shifts)`: vertex `j` is connected to
    /// `(j + s) mod n` for every shift `s`.
    ///
    /// Shifts may be negative; shifts that are multiples of `n` are ignored
    /// and no multi-edges or self-loops are created.
    ///
    /// Binds [`igraph_circulant`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_circulant).
    /// Time complexity: O(|V| |shifts|).
    ///
    /// See also [`extended_chordal_ring`](Self::extended_chordal_ring), which
    /// allows position-dependent chords and multi-edges, and
    /// [`Graph::k_regular_game`] for *random* regular graphs.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // C_5(1, 2) is K_5.
    /// assert_eq!(Graph::circulant(5, &[1, 2], false).unwrap().ecount(), 10);
    /// ```
    pub fn circulant(n: usize, shifts: &[i64], directed: bool) -> Result<Graph> {
        let n = int(n, "the number of vertices")?;
        let shifts = VectorInt::view(shifts);
        Graph::init_with(|g| unsafe { igraph_circulant(g, n, shifts.as_ptr(), directed) })
    }

    /// The generalized Petersen graph `G(n, k)`: an outer `n`-cycle
    /// `v_0 .. v_(n-1)` (ids `0..n`), an inner circulant `u_i ~ u_(i+k mod n)`
    /// (ids `n..2n`) and the spokes `v_i ~ u_i`.
    ///
    /// It has `2n` vertices and `3n` edges and is cubic. `G(5, 2)` is the
    /// Petersen graph, `G(4, 1)` the cube, `G(10, 3)` the Desargues graph.
    ///
    /// Binds [`igraph_generalized_petersen`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_generalized_petersen).
    /// Time complexity: O(|V|).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) unless
    /// `n >= 3` and `0 < k < n / 2`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let desargues = Graph::generalized_petersen(10, 3).unwrap();
    /// assert_eq!((desargues.vcount(), desargues.ecount()), (20, 30));
    /// let petersen = Graph::generalized_petersen(5, 2).unwrap();
    /// assert!(petersen.isomorphic(&Graph::famous("Petersen").unwrap()).unwrap());
    /// ```
    pub fn generalized_petersen(n: usize, k: usize) -> Result<Graph> {
        let n = int(n, "the number of vertices")?;
        let k = int(k, "the shift")?;
        Graph::init_with(|g| unsafe { igraph_generalized_petersen(g, n, k) })
    }

    /// Creates a named graph, such as `"Petersen"` or `"Zachary"`.
    ///
    /// The name is case insensitive; the supported graphs (and their sizes)
    /// are listed by [`FamousGraph`], which can be passed directly. Some names
    /// have aliases in igraph: `Dodecahedral`, `Icosahedral`, `Octahedral`,
    /// `Tetrahedral` and `Groetzsch`.
    ///
    /// Binds [`igraph_famous`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_famous).
    /// Time complexity: O(|V| + |E|).
    ///
    /// See also [`atlas`](Self::atlas) for all small graphs, and
    /// [`crate::foreign`] to read graphs from files.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for an
    /// unknown name (or one containing a NUL byte).
    ///
    /// # Examples
    /// ```
    /// use igraph::{constructors::FamousGraph, prelude::*};
    /// let karate = Graph::famous("zachary").unwrap();
    /// assert_eq!((karate.vcount(), karate.ecount()), (34, 78));
    /// let kite = Graph::famous(FamousGraph::KrackhardtKite).unwrap();
    /// assert_eq!(kite.vcount(), 10);
    /// assert_eq!(Graph::famous("Unicorn").unwrap_err().kind(), ErrorKind::InvalidValue);
    /// // The Frucht graph is cubic but has no symmetry at all.
    /// let frucht = Graph::famous(FamousGraph::Frucht).unwrap();
    /// assert_eq!(frucht.count_automorphisms(None).unwrap(), 1.0);
    /// ```
    pub fn famous(name: impl AsRef<str>) -> Result<Graph> {
        let name = CString::new(name.as_ref())
            .map_err(|_| Error::invalid("graph names cannot contain NUL bytes"))?;
        Graph::init_with(|g| unsafe { igraph_famous(g, name.as_ptr()) })
    }

    /// Creates a graph from LCF (Lederberg–Coxeter–Frucht) notation
    /// `[shifts]^repeats` on `n` vertices.
    ///
    /// The graph is the cycle `0 - 1 - ... - (n-1) - 0` plus, going around
    /// the cycle, a chord from vertex `i` to `i + s` for the shifts `s`
    /// repeated `repeats` times. Normally `n = shifts.len() * repeats`, and the
    /// result is a cubic Hamiltonian graph. The result is always simple:
    /// duplicate chords are merged and loops (shifts that are multiples of
    /// `n`) dropped. Shifts are taken modulo `n`, so any `i64` is accepted;
    /// `n = 0` gives the null graph.
    ///
    /// Binds [`igraph_lcf`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_lcf)
    /// (and covers the variadic `igraph_lcf_small`).
    /// Time complexity: O(|V| + |E|).
    ///
    /// See also [`generalized_petersen`](Self::generalized_petersen) and
    /// [`famous`](Self::famous) for other ways to build well-known cubic graphs.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // The Heawood graph is [5, -5]^7.
    /// let h = Graph::lcf(14, &[5, -5], 7).unwrap();
    /// assert_eq!((h.vcount(), h.ecount()), (14, 21));
    /// assert!(h.isomorphic(&Graph::famous("Heawood").unwrap()).unwrap());
    /// ```
    pub fn lcf(n: usize, shifts: &[i64], repeats: usize) -> Result<Graph> {
        let n = int(n, "the number of vertices")?;
        let repeats = int(repeats, "the number of repeats")?;
        if n == 0 {
            // igraph 1.0.0 and 1.0.1 compute `i % n` for every chord: with n = 0 and a
            // non-empty chord list it dies with a division by zero (SIGFPE).
            return Graph::empty(0, false);
        }
        // igraph computes `n + i + shift` without overflow checks (UB in C
        // for huge shifts), and rejects shifts below `-(n + i)`; reducing
        // them modulo `n` keeps the meaning and avoids both problems.
        let shifts: VectorInt = shifts.iter().map(|&s| s.rem_euclid(n)).collect();
        Graph::init_with(|g| unsafe { igraph_lcf(g, n, &shifts, repeats) })
    }

    /// Builds a graph realizing the given degree sequence, deterministically.
    ///
    /// With `in_degrees = None` an undirected graph with degrees
    /// `out_degrees` is created, otherwise a directed graph with the given
    /// out- and in-degrees. Simple graphs are built with the Havel–Hakimi
    /// (undirected) or Kleitman–Wang (directed) algorithm: repeatedly pick a
    /// vertex and connect all its stubs to the vertices with the largest
    /// remaining degrees. Multigraphs use an analogous one-edge-at-a-time
    /// procedure; with self-loops allowed, leftover stubs become loops on a
    /// single vertex.
    ///
    /// `allowed` selects the kind of graph (directed graphs support only
    /// [`AllowedEdgeTypes::SIMPLE`]; [`AllowedEdgeTypes::LOOPS`] alone is not
    /// implemented). `method` selects the vertex order:
    /// [`RealizeDegseq::Smallest`] (smallest remaining degree first; in the
    /// undirected case it yields a *connected* graph whenever one exists, so
    /// it builds a tree from tree degrees), [`RealizeDegseq::Largest`]
    /// (strongly assortative, often disconnected) or
    /// [`RealizeDegseq::Index`] (in vertex order).
    ///
    /// Binds [`igraph_realize_degree_sequence`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_realize_degree_sequence).
    /// Time complexity: O(V + α(V) E) for simple undirected graphs.
    ///
    /// See also [`is_graphical`](crate::mixing::is_graphical), which only
    /// decides whether a realization exists (it takes the same
    /// [`AllowedEdgeTypes`] flags), and
    /// [`Graph::degree_sequence_game`] for *random* realizations.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the
    /// sequence is not graphical for the requested kind of graph, or lengths
    /// or sums of the directed sequences differ;
    /// [`ErrorKind::Unimplemented`](crate::ErrorKind::Unimplemented) for
    /// unsupported combinations.
    ///
    /// # Examples
    /// ```
    /// use igraph::{constructors::AllowedEdgeTypes, prelude::*};
    /// let degrees = [3, 3, 2, 2, 2, 1, 1];
    /// let g = Graph::realize_degree_sequence(
    ///     &degrees, None, AllowedEdgeTypes::SIMPLE, RealizeDegseq::Smallest,
    /// ).unwrap();
    /// assert_eq!(g.degree(VertexSelector::All, NeighborMode::All, Loops::Twice).unwrap(), degrees);
    /// // [3, 3] cannot be realized as a simple graph...
    /// assert!(Graph::realize_degree_sequence(
    ///     &[3, 3], None, AllowedEdgeTypes::SIMPLE, RealizeDegseq::Smallest).is_err());
    /// // ... but it can as a multigraph: three parallel edges.
    /// let m = Graph::realize_degree_sequence(
    ///     &[3, 3], None, AllowedEdgeTypes::MULTI, RealizeDegseq::Smallest).unwrap();
    /// assert_eq!(m.ecount(), 3);
    /// ```
    pub fn realize_degree_sequence(
        out_degrees: &[i64],
        in_degrees: Option<&[i64]>,
        allowed: impl Into<AllowedEdgeTypes>,
        method: RealizeDegseq,
    ) -> Result<Graph> {
        let out = VectorInt::view(out_degrees);
        let ind = in_degrees.map(VectorInt::view);
        let ind_ptr = ind.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let allowed = igraph_edge_type_sw_t::from(allowed.into());
        Graph::init_with(|g| unsafe {
            igraph_realize_degree_sequence(g, out.as_ptr(), ind_ptr, allowed, method.into())
        })
    }

    /// Builds a bipartite graph realizing the bidegree sequence
    /// `(degrees1, degrees2)`, deterministically.
    ///
    /// Vertices `0..degrees1.len()` form the first partition, followed by
    /// the second one. A Havel–Hakimi-like algorithm is used; `allowed` is
    /// [`AllowedEdgeTypes::SIMPLE`] or [`AllowedEdgeTypes::MULTI`] (a
    /// bipartite graph has no self-loops, so igraph ignores the loops flag:
    /// `LOOPS` acts as `SIMPLE`, `ALL` as `MULTI`), and
    /// `method` has the same meaning as in
    /// [`realize_degree_sequence`](Self::realize_degree_sequence) (with
    /// [`RealizeDegseq::Smallest`] the result is connected whenever the
    /// sequence is potentially connected).
    ///
    /// Binds [`igraph_realize_bipartite_degree_sequence`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_realize_bipartite_degree_sequence).
    ///
    /// See also [`is_bigraphical`](crate::mixing::is_bigraphical), which only
    /// decides whether a realization exists.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the
    /// bidegree sequence cannot be realized.
    ///
    /// # Examples
    /// ```
    /// use igraph::{constructors::AllowedEdgeTypes, prelude::*};
    /// // Three students, two projects: who works on what.
    /// let g = Graph::realize_bipartite_degree_sequence(
    ///     &[1, 2, 1], &[2, 2], AllowedEdgeTypes::SIMPLE, RealizeDegseq::Smallest,
    /// ).unwrap();
    /// assert_eq!((g.vcount(), g.ecount()), (5, 4));
    /// assert!(g.is_bipartite().unwrap());
    /// ```
    pub fn realize_bipartite_degree_sequence(
        degrees1: &[i64],
        degrees2: &[i64],
        allowed: impl Into<AllowedEdgeTypes>,
        method: RealizeDegseq,
    ) -> Result<Graph> {
        let d1 = VectorInt::view(degrees1);
        let d2 = VectorInt::view(degrees2);
        let allowed = igraph_edge_type_sw_t::from(allowed.into());
        Graph::init_with(|g| unsafe {
            igraph_realize_bipartite_degree_sequence(
                g,
                d1.as_ptr(),
                d2.as_ptr(),
                allowed,
                method.into(),
            )
        })
    }

    /// Creates a triangular lattice of the given shape.
    ///
    /// Vertices are points `(i, j)` connected to `(i+1, j)`, `(i, j+1)` and
    /// `(i-1, j+1)` when present, so degrees are at most 6. `dims` of length
    /// 1 gives a triangle with `dims[0]` vertices per side, length 2 a
    /// "quasi-rectangle" with sides of `dims[0]` and `dims[1]` vertices,
    /// length 3 a hexagon with the given side lengths. Vertices are ordered
    /// row by row. This is the planar dual of
    /// [`hexagonal_lattice`](Self::hexagonal_lattice) with the same `dims`.
    ///
    /// Binds [`igraph_triangular_lattice`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_triangular_lattice).
    /// Time complexity: O(|V|).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) unless
    /// `dims` has length 1, 2 or 3, or if a hexagon shape is so large that
    /// its row sizes overflow an `i64` (checked before calling igraph).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let t = Graph::triangular_lattice(&[5], false, false).unwrap();
    /// assert_eq!((t.vcount(), t.ecount()), (15, 30));
    /// ```
    pub fn triangular_lattice(dims: &[usize], directed: bool, mutual: bool) -> Result<Graph> {
        let dims = int_vector(dims, "a lattice dimension")?;
        check_hex_shape(&dims, "a triangular lattice")?;
        Graph::init_with(|g| unsafe { igraph_triangular_lattice(g, &dims, directed, mutual) })
    }

    /// Creates a hexagonal (honeycomb) lattice of the given shape.
    ///
    /// `dims` is interpreted as in
    /// [`triangular_lattice`](Self::triangular_lattice), but counts
    /// *hexagons*: the 6-cycles of the result correspond one-to-one to the
    /// vertices of the triangular lattice with the same `dims`. Degrees are
    /// at most 3.
    ///
    /// Binds [`igraph_hexagonal_lattice`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_hexagonal_lattice).
    /// Time complexity: O(|V|).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) unless
    /// `dims` has length 1, 2 or 3, or if a hexagon shape is so large that
    /// its row sizes overflow an `i64` (checked before calling igraph).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let benzene = Graph::hexagonal_lattice(&[1], false, false).unwrap();
    /// assert_eq!((benzene.vcount(), benzene.ecount()), (6, 6));
    /// ```
    pub fn hexagonal_lattice(dims: &[usize], directed: bool, mutual: bool) -> Result<Graph> {
        let dims = int_vector(dims, "a lattice dimension")?;
        check_hex_shape(&dims, "a hexagonal lattice")?;
        Graph::init_with(|g| unsafe { igraph_hexagonal_lattice(g, &dims, directed, mutual) })
    }

    /// The Mycielski graph `M_k`: triangle-free with chromatic number `k`.
    ///
    /// Obtained by iterating the Mycielski construction: `M_0` is the null
    /// graph, `M_1` a single vertex, `M_2` an edge, `M_3` the 5-cycle, `M_4`
    /// the Grötzsch graph. For `k > 1`, `M_k` has `3 * 2^(k-2) - 1` vertices
    /// and `(7 * 3^(k-2) + 1) / 2 - 3 * 2^(k-2)` edges.
    ///
    /// This function is marked *experimental* in igraph 1.0.x.
    ///
    /// Binds [`igraph_mycielski_graph`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_mycielski_graph).
    /// Time complexity: O(3^k).
    ///
    /// See also [`Graph::mycielskian`], which applies the construction to an
    /// arbitrary graph.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let m4 = Graph::mycielski_graph(4).unwrap();
    /// assert_eq!((m4.vcount(), m4.ecount()), (11, 20));
    /// assert!(m4.isomorphic(&Graph::famous("Grotzsch").unwrap()).unwrap());
    /// assert_eq!(m4.count_triangles().unwrap(), 0.0);
    /// ```
    pub fn mycielski_graph(k: usize) -> Result<Graph> {
        let k = int(k, "the order")?;
        Graph::init_with(|g| unsafe { igraph_mycielski_graph(g, k) })
    }
}

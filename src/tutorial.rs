//! Translations of the [igraph C tutorial](https://igraph.org/c/html/latest/igraph-Tutorial.html)
//! into Rust.
//!
//! The igraph C reference manual opens with a tutorial made of three short
//! lessons, whose programs live in `examples/tutorial/tutorial{1,2,3}.c` of
//! the igraph sources. This module translates each of them, one public
//! function per lesson, keeping the *same steps in the same order* (and hence
//! the same pseudo-random draws), so that the Rust versions compute exactly
//! the numbers printed by the C programs with igraph 1.0.1:
//!
//! | Lesson | Rust | C program prints |
//! |--------|------|------------------|
//! | [1. Compiling programs using igraph](https://igraph.org/c/html/latest/igraph-Tutorial.html#tut-lesson-1) | [`example_1`] → [`RandomGraphStats`] | `Diameter of a random graph with average degree 2: 23` |
//! | [2. Creating your first graphs](https://igraph.org/c/html/latest/igraph-Tutorial.html#tut-lesson-2) | [`example_2`] → [`LatticePathLengths`] | `Average path length (lattice): 15.0167`, then `11.8142` once randomized |
//! | [3. Calculating various properties of graphs](https://igraph.org/c/html/latest/igraph-Tutorial.html#tut-lesson-3) | [`example_3`] → [`KarateCentralities`] | maximum degree 17, closeness 0.0172414, betweenness 231.071 |
//!
//! Every result struct implements [`Display`](std::fmt::Display) reproducing,
//! character by character, the lines printed by the corresponding C program
//! (including C's `%g` number formatting), so the translation can be checked
//! against the original at a glance:
//!
//! ```
//! use igraph::tutorial;
//!
//! println!("{}", tutorial::example_1()?);
//! println!("{}", tutorial::example_2()?);
//! println!("{}", tutorial::example_3()?);
//! assert_eq!(
//!     tutorial::example_1()?.to_string(),
//!     "Diameter of a random graph with average degree 2: 23"
//! );
//! # Ok::<(), igraph::Error>(())
//! ```
//!
//! # Lesson 1: compiling programs using igraph
//!
//! The first program generates a random graph and prints its diameter and
//! mean degree. The C tutorial uses it to illustrate a few points, and each
//! of them has a Rust counterpart:
//!
//! - C programs include `igraph.h`; in Rust, `use igraph::prelude::*;` brings
//!   the graph type, the containers and the enums into scope.
//! - C programs must call `igraph_setup()` before anything else. The Rust
//!   bindings do it for you: every call into igraph first makes sure the
//!   library (and the calling thread's error handler and random number
//!   generator) is initialized.
//! - igraph uses `igraph_int_t` for integers and `igraph_real_t` for reals:
//!   these are `i64` and `f64` in Rust. Vertex and edge ids are
//!   [`VertexId`] and [`EdgeId`](crate::EdgeId) (both `i64`), counts are `usize`.
//! - Graphs are `igraph_t` objects, which is exactly what [`Graph`] is (a type
//!   alias enriched with methods). Generators such as
//!   [`Graph::erdos_renyi_game_gnm`] create them; where C calls
//!   `igraph_destroy()`, Rust frees the graph automatically when it goes out
//!   of scope.
//! - C functions return an error code; the Rust methods return a
//!   [`Result`], propagated with `?`.
//!
//! The igraph tutorial then explains how to compile the program with CMake or
//! `pkg-config`; here `cargo build` takes care of it (the build script finds
//! the installed igraph library and generates the raw bindings).
//!
//! # Lesson 2: creating your first graphs
//!
//! Functions creating graphs are called *generators*; randomized ones are
//! called *games*. Deterministic regular structures include stars
//! ([`Graph::star`]), cycles ([`Graph::cycle_graph`]), lattices
//! ([`Graph::square_lattice`]) and trees ([`Graph::kary_tree`]). Most
//! generators, and most other functions, handle both directed and undirected
//! graphs.
//!
//! The second program builds a 30 × 30 periodic square lattice (a torus),
//! computes the average shortest path length, adds ten random edges and
//! computes it again: a handful of random "shortcuts" shrinks the average
//! distance a lot (from about 15 to about 11.8), the essence of the
//! *small-world* effect.
//!
//! In C, igraph uses its own vector types (`igraph_vector_t`,
//! `igraph_vector_int_t`, `igraph_vector_bool_t`, ...) instead of plain
//! arrays, initialized with `igraph_vector_init()` and destroyed with
//! `igraph_vector_destroy()`. The Rust bindings accept slices as inputs and
//! return `Vec`s as outputs (the owned igraph vectors, such as
//! [`VectorInt`](crate::vector::VectorInt), exist too and free themselves on
//! drop). Vertices are identified by ids `0..n`, where `n` is
//! [`Graph::vcount`]. [`Graph::add_edges`] takes `(from, to)` pairs, whereas
//! the C `igraph_add_edges()` takes a flat vector of endpoints
//! ([`Graph::add_edges_from_vector`] is its literal counterpart).
//!
//! As the tutorial warns, drawing random endpoints may create *loop edges*
//! (from a vertex to itself) and *multi-edges* (several edges between the same
//! pair of vertices). igraph graphs can represent them, but some functions
//! expect simple graphs: [`Graph::simplify`] removes them. (With seed 42 the
//! ten edges drawn by lesson 2, see [`LatticePathLengths::random_edges`],
//! happen to keep the lattice simple, although vertex 885 is drawn twice in a
//! row, as the endpoint of two different edges.)
//!
//! # Lesson 3: calculating various properties of graphs
//!
//! The third program computes three *centrality* measures on the friendship
//! network of Zachary's karate club: how central the position of every member
//! is. It builds the graph from a plain array of endpoints
//! ([`ZACHARY_KARATE_EDGES`], with [`Graph::from_flat_edges`]; C creates a
//! non-owning *view* of the array with `igraph_vector_int_view()`, which the
//! Rust bindings do internally), then computes
//!
//! - the degree of every vertex ([`Graph::degree`]),
//! - the closeness centrality ([`Graph::closeness`]),
//! - the betweenness centrality ([`Graph::betweenness`]),
//!
//! and prints the largest value of each, with the vertex attaining it. The
//! instructor (vertex 0) and the administrator (vertex 33) stand out.
//!
//! In C, the argument `igraph_vss_all()` is a *vertex selector* asking for
//! the property of every vertex; in Rust it is [`VertexSelector::All`], or
//! simply the full range `..` (see [`crate::selector`] for the other
//! selectors).

use std::fmt;

use crate::constants::{EdgeTypeSw, Loops, NeighborMode};
use crate::error::{Error, ErrorKind, Result};
use crate::graph::{Graph, VertexId};
use crate::rng;
use crate::selector::VertexSelector;

/// The numbers computed by lesson 1 (see [`example_1`]).
///
/// Its [`Display`](fmt::Display) implementation prints the same line as the
/// C program, e.g. `Diameter of a random graph with average degree 2: 23`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RandomGraphStats {
    /// Length of the longest geodesic, considering every connected component
    /// (the C program passes `unconn = true`).
    pub diameter: f64,
    /// Average degree of the vertices, `2m / n` (self-loops counted, as with
    /// `IGRAPH_LOOPS` in C).
    pub mean_degree: f64,
}

impl fmt::Display for RandomGraphStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Diameter of a random graph with average degree {}: {}",
            format_g(self.mean_degree),
            format_g(self.diameter)
        )
    }
}

/// The numbers computed by lesson 2 (see [`example_2`]).
///
/// Its [`Display`](fmt::Display) implementation prints the same two lines as
/// the C program.
#[derive(Debug, Clone, PartialEq)]
pub struct LatticePathLengths {
    /// Average shortest path length of the 30 × 30 periodic lattice.
    pub lattice: f64,
    /// Average shortest path length after adding [`random_edges`](Self::random_edges).
    pub randomized: f64,
    /// The ten random edges added to the lattice, in the order they were
    /// drawn. Such random draws may in general produce loops and multi-edges;
    /// the ten edges drawn after seeding with 42 happen to contain neither.
    pub random_edges: Vec<(VertexId, VertexId)>,
}

impl fmt::Display for LatticePathLengths {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Average path length (lattice):            {}",
            format_g(self.lattice)
        )?;
        write!(
            f,
            "Average path length (randomized lattice): {}",
            format_g(self.randomized)
        )
    }
}

/// The maximum of a per-vertex measure and the vertex attaining it, the
/// Rust counterpart of the C pair `igraph_vector_max()` /
/// `igraph_vector_which_max()`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Maximum<T> {
    /// The largest value.
    pub value: T,
    /// The (first) vertex attaining it.
    pub vertex: VertexId,
}

impl<T: PartialOrd + Copy> Maximum<T> {
    /// The largest element of `values` (indexed by vertex id) and its index,
    /// or `None` for an empty slice.
    ///
    /// It behaves exactly like igraph's `igraph_vector_max()` and
    /// `igraph_vector_which_max()`: the *first* maximal element wins ties,
    /// and if `values` contains a `NaN` (for floating-point types: any
    /// element not comparable with itself), the first `NaN` and its index are
    /// returned.
    ///
    /// # Examples
    /// ```
    /// use igraph::tutorial::Maximum;
    /// assert_eq!(Maximum::of(&[3, 7, 1, 7]), Some(Maximum { value: 7, vertex: 1 }));
    /// assert_eq!(Maximum::<f64>::of(&[]), None);
    /// // As in igraph, a NaN is the "maximum" of any vector containing one.
    /// let m = Maximum::of(&[1.0, f64::NAN, 2.0]).unwrap();
    /// assert!(m.value.is_nan() && m.vertex == 1);
    /// ```
    pub fn of(values: &[T]) -> Option<Self> {
        let mut best: Option<Self> = None;
        for (i, &value) in values.iter().enumerate() {
            let candidate = Maximum {
                value,
                vertex: i as VertexId,
            };
            // `NaN` is the only value not comparable with itself.
            if value.partial_cmp(&value).is_none() {
                return Some(candidate);
            }
            if best.is_none_or(|b| value > b.value) {
                best = Some(candidate);
            }
        }
        best
    }
}

/// The centrality maxima computed by lesson 3 (see [`example_3`]).
///
/// Its [`Display`](fmt::Display) implementation prints the same three lines
/// as the C program.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KarateCentralities {
    /// Maximum degree.
    pub degree: Maximum<i64>,
    /// Maximum (non-normalized) closeness centrality.
    pub closeness: Maximum<f64>,
    /// Maximum (non-normalized) betweenness centrality.
    pub betweenness: Maximum<f64>,
}

impl fmt::Display for KarateCentralities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Maximum degree is      {:>10}, vertex {:>2}.",
            self.degree.value, self.degree.vertex
        )?;
        writeln!(
            f,
            "Maximum closeness is   {:>10}, vertex {:>2}.",
            format_g(self.closeness.value),
            self.closeness.vertex
        )?;
        write!(
            f,
            "Maximum betweenness is {:>10}, vertex {:>2}.",
            format_g(self.betweenness.value),
            self.betweenness.vertex
        )
    }
}

/// Lesson 1: the diameter and mean degree of a random graph.
///
/// Seeds the calling thread's default random number generator with 42 (as
/// the C program does, "to ensure identical results across runs"), generates
/// a simple undirected Erdős–Rényi `G(n, m)` graph with `n = 1000` vertices
/// and `m = 1000` edges, and computes its diameter (over all components, the
/// graph being disconnected) and its mean degree `2m / n = 2`.
///
/// It translates `examples/tutorial/tutorial1.c` of the
/// [first lesson](https://igraph.org/c/html/latest/igraph-Tutorial.html#tut-lesson-1),
/// and computes exactly the values printed by the C program with igraph
/// 1.0.1: `Diameter of a random graph with average degree 2: 23`.
///
/// # C code
///
/// ```c
/// #include <igraph.h>
///
/// int main(void) {
///     igraph_int_t num_vertices = 1000;
///     igraph_int_t num_edges = 1000;
///     igraph_real_t diameter, mean_degree;
///     igraph_t graph;
///
///     /* Initialize the library. */
///     igraph_setup();
///
///     /* Ensure identical results across runs. */
///     igraph_rng_seed(igraph_rng_default(), 42);
///
///     igraph_erdos_renyi_game_gnm(
///             &graph, num_vertices, num_edges,
///             IGRAPH_UNDIRECTED, IGRAPH_SIMPLE_SW, IGRAPH_EDGE_UNLABELED);
///
///     igraph_diameter(
///         &graph, /* weights = */ NULL,
///         &diameter,
///         /* from = */ NULL, /* to = */ NULL,
///         /* vertex_path = */ NULL, /* edge_path = */ NULL,
///         IGRAPH_UNDIRECTED, /* unconn= */ true);
///
///     igraph_mean_degree(&graph, &mean_degree, IGRAPH_LOOPS);
///     printf("Diameter of a random graph with average degree %g: %g\n",
///            mean_degree, diameter);
///
///     igraph_destroy(&graph);
///
///     return 0;
/// }
/// ```
///
/// # Rust translation
///
/// The body of this function, step by step (no `igraph_setup()` and no
/// `igraph_destroy()` are needed):
///
/// ```
/// use igraph::prelude::*;
///
/// let (num_vertices, num_edges) = (1000, 1000);
///
/// // Ensure identical results across runs.
/// rng::seed(42)?;
///
/// let graph = Graph::erdos_renyi_game_gnm(
///     num_vertices,
///     num_edges,
///     false, // undirected
///     EdgeTypeSw::Simple,
///     false, // unlabeled edges
/// )?;
///
/// let diameter = graph.diameter()?; // unweighted, over all components
/// let mean_degree = graph.mean_degree(true)?; // loops counted
/// println!("Diameter of a random graph with average degree {mean_degree}: {diameter}");
///
/// assert_eq!(diameter, 23.0);
/// assert_eq!(mean_degree, 2.0);
///
/// // ... which is what `example_1` returns.
/// let stats = igraph::tutorial::example_1()?;
/// assert_eq!((stats.diameter, stats.mean_degree), (diameter, mean_degree));
/// assert_eq!(
///     stats.to_string(),
///     "Diameter of a random graph with average degree 2: 23"
/// );
/// # Ok::<(), igraph::Error>(())
/// ```
///
/// # Errors
/// Propagates the errors of the igraph calls (none are expected).
pub fn example_1() -> Result<RandomGraphStats> {
    let num_vertices = 1000;
    let num_edges = 1000;

    // Ensure identical results across runs.
    rng::seed(42)?;

    let graph =
        Graph::erdos_renyi_game_gnm(num_vertices, num_edges, false, EdgeTypeSw::Simple, false)?;

    let diameter = graph.diameter()?;
    let mean_degree = graph.mean_degree(true)?;

    Ok(RandomGraphStats {
        diameter,
        mean_degree,
    })
}

/// Lesson 2: average path length of a lattice, before and after adding a
/// few random edges.
///
/// Builds the undirected 30 × 30 square lattice with periodic boundaries in
/// both dimensions (a torus: every vertex has degree 4) and computes its
/// average shortest path length, `15 · 900 / 899 ≈ 15.0167`. Then it seeds
/// the thread's default random number generator with 42, draws 20 uniform
/// vertex ids in `0..900` (pairing them up as ten edges; in general such
/// draws may include loops and multi-edges, although with seed 42 they do
/// not), adds them to the lattice and computes the average
/// path length again, `≈ 11.8142`: ten random shortcuts reduce distances by
/// more than 20%.
///
/// It translates `examples/tutorial/tutorial2.c` of the
/// [second lesson](https://igraph.org/c/html/latest/igraph-Tutorial.html#tut-lesson-2),
/// drawing the random numbers in the same order, so that the results are
/// exactly those printed by the C program with igraph 1.0.1.
///
/// # C code
///
/// ```c
/// #include <igraph.h>
///
/// int main(void) {
///     igraph_t graph;
///     igraph_vector_int_t dimvector;
///     igraph_vector_int_t edges;
///     igraph_vector_bool_t periodic;
///     igraph_real_t avg_path_len;
///
///     /* Initialize the library. */
///     igraph_setup();
///
///     igraph_vector_int_init(&dimvector, 2);
///     VECTOR(dimvector)[0] = 30;
///     VECTOR(dimvector)[1] = 30;
///
///     igraph_vector_bool_init(&periodic, 2);
///     igraph_vector_bool_fill(&periodic, true);
///     igraph_square_lattice(&graph, &dimvector, 0, IGRAPH_UNDIRECTED,
///                           /* mutual= */ false, &periodic);
///
///     igraph_average_path_length(&graph, NULL, &avg_path_len, NULL,
///                                IGRAPH_UNDIRECTED, /* unconn= */ true);
///     printf("Average path length (lattice):            %g\n", (double) avg_path_len);
///
///     /* Seed the RNG to ensure identical results across runs. */
///     igraph_rng_seed(igraph_rng_default(), 42);
///
///     igraph_vector_int_init(&edges, 20);
///     for (igraph_int_t i = 0; i < igraph_vector_int_size(&edges); i++) {
///         VECTOR(edges)[i] = RNG_INTEGER(0, igraph_vcount(&graph) - 1);
///     }
///
///     igraph_add_edges(&graph, &edges, NULL);
///     igraph_average_path_length(&graph, NULL, &avg_path_len, NULL,
///                                IGRAPH_UNDIRECTED, /* unconn= */ true);
///     printf("Average path length (randomized lattice): %g\n", (double) avg_path_len);
///
///     igraph_vector_bool_destroy(&periodic);
///     igraph_vector_int_destroy(&dimvector);
///     igraph_vector_int_destroy(&edges);
///     igraph_destroy(&graph);
///
///     return 0;
/// }
/// ```
///
/// # Rust translation
///
/// The body of this function, step by step: plain arrays replace the
/// `igraph_vector_*_t` objects, and nothing needs to be destroyed.
///
/// ```
/// use igraph::prelude::*;
///
/// let mut graph = Graph::square_lattice(
///     &[30, 30],          // dimensions
///     0,                  // nei: only direct neighbors
///     false,              // undirected
///     false,              // mutual (directed graphs only)
///     Some(&[true, true]), // periodic in both dimensions
/// )?;
///
/// let lattice = graph.average_path_length(None, false, true)?;
/// println!("Average path length (lattice):            {lattice}");
/// assert!((lattice - 15.0 * 900.0 / 899.0).abs() < 1e-12);
///
/// // Seed the RNG to ensure identical results across runs.
/// rng::seed(42)?;
///
/// let n = graph.vcount() as i64;
/// let edges: Vec<i64> = (0..20).map(|_| rng::integer(0, n - 1)).collect();
///
/// graph.add_edges_from_vector(&edges)?;
/// let randomized = graph.average_path_length(None, false, true)?;
/// println!("Average path length (randomized lattice): {randomized}");
/// assert!((randomized - 11.814163885799037).abs() < 1e-12);
///
/// // ... which is what `example_2` returns.
/// let res = igraph::tutorial::example_2()?;
/// assert_eq!((res.lattice, res.randomized), (lattice, randomized));
/// assert_eq!(res.to_string(), "\
/// Average path length (lattice):            15.0167
/// Average path length (randomized lattice): 11.8142");
/// # Ok::<(), igraph::Error>(())
/// ```
///
/// # Errors
/// Propagates the errors of the igraph calls (none are expected).
pub fn example_2() -> Result<LatticePathLengths> {
    let mut graph = Graph::square_lattice(&[30, 30], 0, false, false, Some(&[true, true]))?;

    let lattice = graph.average_path_length(None, false, true)?;

    // Seed the RNG to ensure identical results across runs.
    rng::seed(42)?;

    // Twenty endpoints drawn one after the other, as in C (tuple fields are
    // evaluated left to right), paired up as ten edges.
    let n = graph.vcount() as VertexId;
    let random_edges: Vec<(VertexId, VertexId)> = (0..10)
        .map(|_| (rng::integer(0, n - 1), rng::integer(0, n - 1)))
        .collect();

    graph.add_edges(&random_edges)?;
    let randomized = graph.average_path_length(None, false, true)?;

    Ok(LatticePathLengths {
        lattice,
        randomized,
        random_edges,
    })
}

/// The friendship network of
/// [Zachary's karate club](https://en.wikipedia.org/wiki/Zachary%27s_karate_club)
/// as the flat endpoint array of the C tutorial (lesson 3): 78 undirected
/// edges among 34 members, `[from0, to0, from1, to1, ...]`.
///
/// Vertex 0 is the instructor ("Mr. Hi") and vertex 33 the administrator
/// ("John A."), whose conflict split the club in two.
///
/// # Examples
/// ```
/// use igraph::prelude::*;
/// use igraph::tutorial::ZACHARY_KARATE_EDGES;
///
/// let karate = Graph::from_flat_edges(&ZACHARY_KARATE_EDGES, 0, false)?;
/// assert_eq!((karate.vcount(), karate.ecount()), (34, 78));
/// # Ok::<(), igraph::Error>(())
/// ```
#[rustfmt::skip]
pub const ZACHARY_KARATE_EDGES: [VertexId; 156] = [
    0,1, 0,2, 0,3, 0,4, 0,5, 0,6, 0,7, 0,8,
    0,10, 0,11, 0,12, 0,13, 0,17, 0,19, 0,21, 0,31,
    1, 2, 1, 3, 1, 7, 1,13, 1,17, 1,19, 1,21, 1,30,
    2, 3, 2, 7, 2,27, 2,28, 2,32, 2, 9, 2, 8, 2,13,
    3, 7, 3,12, 3,13, 4, 6, 4,10, 5, 6, 5,10, 5,16,
    6,16, 8,30, 8,32, 8,33, 9,33, 13,33, 14,32, 14,33,
    15,32, 15,33, 18,32, 18,33, 19,33, 20,32, 20,33,
    22,32, 22,33, 23,25, 23,27, 23,32, 23,33, 23,29,
    24,25, 24,27, 24,31, 25,31, 26,29, 26,33, 27,33,
    28,31, 28,33, 29,32, 29,33, 30,32, 30,33, 31,32,
    31,33, 32,33,
];

/// Lesson 3: degree, closeness and betweenness centrality in Zachary's
/// karate club.
///
/// Creates the undirected friendship graph from [`ZACHARY_KARATE_EDGES`] and
/// returns, for each of the three centrality measures, its maximum and the
/// first vertex attaining it:
///
/// - degree (all neighbors, loops counted): 17, the administrator (vertex 33);
/// - closeness (non-normalized, `1 / Σ distances`): `1/58 ≈ 0.0172414`, the
///   instructor (vertex 0);
/// - betweenness (non-normalized): `≈ 231.071`, the instructor again.
///
/// It translates `examples/tutorial/tutorial3.c` of the
/// [third lesson](https://igraph.org/c/html/latest/igraph-Tutorial.html#tut-lesson-3).
///
/// # C code
///
/// ```c
/// #include <igraph.h>
///
/// int main(void) {
///     igraph_t graph;
///     igraph_vector_int_t result;
///     igraph_vector_t result_real;
///     igraph_int_t edges_array[] = {
///         0,1, 0,2, 0,3, 0,4, 0,5, 0,6, 0,7, 0,8,
///         0,10, 0,11, 0,12, 0,13, 0,17, 0,19, 0,21, 0,31,
///         1, 2, 1, 3, 1, 7, 1,13, 1,17, 1,19, 1,21, 1,30,
///         2, 3, 2, 7, 2,27, 2,28, 2,32, 2, 9, 2, 8, 2,13,
///         3, 7, 3,12, 3,13, 4, 6, 4,10, 5, 6, 5,10, 5,16,
///         6,16, 8,30, 8,32, 8,33, 9,33, 13,33, 14,32, 14,33,
///         15,32, 15,33, 18,32, 18,33, 19,33, 20,32, 20,33,
///         22,32, 22,33, 23,25, 23,27, 23,32, 23,33, 23,29,
///         24,25, 24,27, 24,31, 25,31, 26,29, 26,33, 27,33,
///         28,31, 28,33, 29,32, 29,33, 30,32, 30,33, 31,32,
///         31,33, 32,33
///     };
///     igraph_vector_int_t edges =
///         igraph_vector_int_view(edges_array, sizeof(edges_array) / sizeof(edges_array[0]));
///
///     /* Initialize the library. */
///     igraph_setup();
///
///     igraph_create(&graph, &edges, 0, IGRAPH_UNDIRECTED);
///
///     igraph_vector_int_init(&result, 0);
///     igraph_vector_init(&result_real, 0);
///
///     igraph_degree(&graph, &result, igraph_vss_all(), IGRAPH_ALL, IGRAPH_LOOPS);
///     printf("Maximum degree is      %10" IGRAPH_PRId ", vertex %2" IGRAPH_PRId ".\n",
///            igraph_vector_int_max(&result),
///            igraph_vector_int_which_max(&result));
///
///     igraph_closeness(&graph, &result_real, NULL, NULL, igraph_vss_all(),
///                      IGRAPH_ALL, /* weights= */ NULL, /* normalized= */ false);
///     printf("Maximum closeness is   %10g, vertex %2" IGRAPH_PRId ".\n",
///            (double) igraph_vector_max(&result_real),
///            igraph_vector_which_max(&result_real));
///
///     igraph_betweenness(&graph, /* weights= */ NULL, &result_real, igraph_vss_all(),
///                        IGRAPH_UNDIRECTED, /* normalized= */ false);
///     printf("Maximum betweenness is %10g, vertex %2" IGRAPH_PRId ".\n",
///            (double) igraph_vector_max(&result_real),
///            igraph_vector_which_max(&result_real));
///
///     igraph_vector_int_destroy(&result);
///     igraph_vector_destroy(&result_real);
///     igraph_destroy(&graph);
///
///     return 0;
/// }
/// ```
///
/// # Rust translation
///
/// The body of this function, step by step: the result vectors are plain
/// `Vec`s returned by the methods, and [`Maximum::of`] plays the role of
/// `igraph_vector_max()` plus `igraph_vector_which_max()`.
///
/// ```
/// use igraph::prelude::*;
/// use igraph::tutorial::{Maximum, ZACHARY_KARATE_EDGES};
///
/// // `0` vertices: igraph infers the vertex count from the largest id.
/// let graph = Graph::from_flat_edges(&ZACHARY_KARATE_EDGES, 0, false)?;
///
/// let degree = graph.degree(VertexSelector::All, NeighborMode::All, Loops::Twice)?;
/// let max_degree = Maximum::of(&degree).unwrap();
/// assert_eq!(max_degree, Maximum { value: 17, vertex: 33 });
///
/// let closeness = graph.closeness(VertexSelector::All, NeighborMode::All, None, false)?;
/// let max_closeness = Maximum::of(&closeness).unwrap();
/// assert_eq!(max_closeness.vertex, 0);
/// assert!((max_closeness.value - 1.0 / 58.0).abs() < 1e-15);
///
/// let betweenness = graph.betweenness(None, VertexSelector::All, false, false)?;
/// let max_betweenness = Maximum::of(&betweenness).unwrap();
/// assert_eq!(max_betweenness.vertex, 0);
/// assert!((max_betweenness.value - 231.0714285714286).abs() < 1e-9);
///
/// // ... which is what `example_3` returns.
/// let res = igraph::tutorial::example_3()?;
/// assert_eq!(res.degree, max_degree);
/// assert_eq!(res.to_string(), "\
/// Maximum degree is              17, vertex 33.
/// Maximum closeness is    0.0172414, vertex  0.
/// Maximum betweenness is    231.071, vertex  0.");
/// # Ok::<(), igraph::Error>(())
/// ```
///
/// # Errors
/// Propagates the errors of the igraph calls (none are expected).
pub fn example_3() -> Result<KarateCentralities> {
    let graph = Graph::from_flat_edges(&ZACHARY_KARATE_EDGES, 0, false)?;

    let degree = graph.degree(VertexSelector::All, NeighborMode::All, Loops::Twice)?;
    let closeness = graph.closeness(VertexSelector::All, NeighborMode::All, None, false)?;
    let betweenness = graph.betweenness(None, VertexSelector::All, false, false)?;

    let max = |what: &str| {
        Error::new(
            ErrorKind::Internal,
            format!("no {what} maximum in an empty graph"),
        )
    };
    Ok(KarateCentralities {
        degree: Maximum::of(&degree).ok_or_else(|| max("degree"))?,
        closeness: Maximum::of(&closeness).ok_or_else(|| max("closeness"))?,
        betweenness: Maximum::of(&betweenness).ok_or_else(|| max("betweenness"))?,
    })
}

/// Formats a real number like C's `printf("%g", x)`: six significant digits,
/// trailing zeros removed, scientific notation for exponents below -4 or
/// above 5.
fn format_g(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.into();
    }
    const PRECISION: i32 = 6;
    // Rounding to the significant digits first decides the exponent (e.g.
    // 999999.5 becomes 1e+06).
    let sci = format!("{:.*e}", (PRECISION - 1) as usize, x);
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    if (-4..PRECISION).contains(&exp) {
        let fixed = format!("{:.*}", (PRECISION - 1 - exp) as usize, x);
        strip_zeros(&fixed).to_string()
    } else {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", strip_zeros(mantissa), exp.abs())
    }
}

/// Removes the trailing zeros of the fractional part (and a dangling `.`).
fn strip_zeros(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::format_g;

    #[test]
    fn format_g_matches_printf() {
        let cases = [
            (2.0, "2"),
            (23.0, "23"),
            (15.016685205784205, "15.0167"),
            (11.814163885799037, "11.8142"),
            (0.017241379310344827, "0.0172414"),
            (231.0714285714286, "231.071"),
            (0.0001, "0.0001"),
            (0.00001234, "1.234e-05"),
            (123456.0, "123456"),
            (1234567.0, "1.23457e+06"),
            (999999.5, "1e+06"),
            (-0.5, "-0.5"),
            (1e100, "1e+100"),
            (0.0, "0"),
            (f64::NAN, "nan"),
            (f64::NEG_INFINITY, "-inf"),
        ];
        for (x, expected) in cases {
            assert_eq!(format_g(x), expected, "formatting {x}");
        }
    }
}

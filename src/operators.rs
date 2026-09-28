//! Graph operators: unions, intersections, complements, subgraphs,
//! simplification, rewiring and graph products (`igraph_operators.h`).
//!
//! This module turns the operators of igraph's
//! [Graph Operators](https://igraph.org/c/html/latest/igraph-Operators.html)
//! chapter into methods of [`Graph`]. Operators that *build a new graph*
//! borrow their operands (`&self`) and return a fresh [`Graph`]; operators
//! that igraph performs *in place* take `&mut self`
//! ([`simplify`](Graph::simplify), [`contract_vertices`](Graph::contract_vertices),
//! [`connect_neighborhood`](Graph::connect_neighborhood),
//! [`reverse_edges`](Graph::reverse_edges), [`rewire`](Graph::rewire)). Clone the
//! graph first if you need to keep the original.
//!
//! Operators working on *many* graphs at once (e.g.
//! [`Graph::union_many`]) are associated functions accepting any iterator of
//! graph references, such as `[&g1, &g2, &g3]` or `graphs.iter()`.
//!
//! As in igraph, vertex ids are never "matched by name": two graphs are
//! combined by *identifying vertices with the same id*. Unless stated
//! otherwise, the operands must have the same directedness, and graph,
//! vertex and edge attributes are not handled by these wrappers.
//!
//! # Example
//!
//! ```
//! use igraph::prelude::*;
//!
//! // A 4-cycle and a "diagonal" graph on the same vertex set.
//! let square = Graph::ring(4, false, false, true).unwrap();
//! let diagonals = Graph::from_edges(&[(0, 2), (1, 3)], 4, false).unwrap();
//!
//! // Their union is K4, the complement of the square is the diagonals.
//! let k4 = square.union(&diagonals).unwrap();
//! assert_eq!(k4, Graph::full(4, false, false).unwrap());
//! assert_eq!(square.complementer(false).unwrap(), diagonals);
//! assert_eq!(k4.difference(&square).unwrap(), diagonals);
//!
//! // Two disjoint copies of the square, then the subgraph induced by one of them.
//! let two = square.disjoint_union(&square).unwrap();
//! assert_eq!((two.vcount(), two.ecount()), (8, 8));
//! let back = two.induced_subgraph(4..8, SubgraphImplementation::Auto).unwrap();
//! assert_eq!(back, square);
//!
//! // The Cartesian product of two edges is a square (up to relabeling).
//! let edge = Graph::full(2, false, false).unwrap();
//! let prod = edge.product(&edge, Product::Cartesian).unwrap();
//! assert!(prod.isomorphic(&square).unwrap());
//! ```
//!
//! # Provided functionality
//!
//! | Rust | C function | What it does |
//! |------|------------|--------------|
//! | [`Graph::disjoint_union`] | `igraph_disjoint_union` | side-by-side copy of two graphs |
//! | [`Graph::disjoint_union_many`] | `igraph_disjoint_union_many` | side-by-side copy of many graphs |
//! | [`Graph::union`], [`Graph::union_map`] | `igraph_union` | edges in either graph (+ edge maps) |
//! | [`Graph::union_many`], [`Graph::union_many_map`] | `igraph_union_many` | edges in any graph (+ edge maps) |
//! | [`Graph::intersection`], [`Graph::intersection_map`] | `igraph_intersection` | edges in both graphs (+ edge maps) |
//! | [`Graph::intersection_many`], [`Graph::intersection_many_map`] | `igraph_intersection_many` | edges in all graphs (+ edge maps) |
//! | [`Graph::difference`] | `igraph_difference` | edges of the first graph missing from the second |
//! | [`Graph::join`] | `igraph_join` | disjoint union plus all edges between the two parts |
//! | [`Graph::complementer`] | `igraph_complementer` | complement graph |
//! | [`Graph::compose`], [`Graph::compose_map`] | `igraph_compose` | composition of relations |
//! | [`Graph::contract_vertices`] | `igraph_contract_vertices` | merge groups of vertices (in place) |
//! | [`Graph::permute_vertices`] | `igraph_permute_vertices` | relabel vertices |
//! | [`Graph::connect_neighborhood`] | `igraph_connect_neighborhood` | connect vertices within `k` steps (in place) |
//! | [`Graph::graph_power`] | `igraph_graph_power` | the `k`-th power of a graph |
//! | [`Graph::rewire`] | `igraph_rewire` | degree-preserving random edge switches (in place) |
//! | [`Graph::simplify`] | `igraph_simplify` | remove multi-edges and/or loops (in place) |
//! | [`Graph::induced_subgraph`] | `igraph_induced_subgraph` | subgraph induced by a vertex set |
//! | [`Graph::induced_subgraph_map`] | `igraph_induced_subgraph_map` | same, with the vertex id maps |
//! | [`Graph::induced_subgraph_edges`] | `igraph_induced_subgraph_edges` | ids of the edges inside a vertex set |
//! | [`Graph::subgraph_from_edges`] | `igraph_subgraph_from_edges` | subgraph spanned by an edge set |
//! | [`Graph::reverse_edges`] | `igraph_reverse_edges` | flip the direction of some edges (in place) |
//! | [`Graph::product`] | `igraph_product` | Cartesian, lexicographic, strong, tensor, modular products |
//! | [`Graph::rooted_product`] | `igraph_rooted_product` | rooted product |
//! | [`Graph::mycielskian`] | `igraph_mycielskian` | iterated Mycielski construction |
//!
//! `igraph_add_edge`, also declared in this header, is wrapped by the core
//! method [`Graph::add_edge`].
//!
//! # See also
//!
//! - Ready-made graphs to feed these operators: [`Graph::famous`],
//!   [`Graph::full`], [`Graph::ring`], [`Graph::square_lattice`],
//!   [`Graph::hypercube`], [`Graph::wheel`], [`Graph::mycielski_graph`]
//!   (`constructors`) and [`Graph::full_bipartite`] (`bipartite`).
//! - Comparing results: `==` compares *labelled* graphs
//!   ([`Graph::is_same_graph`]); [`Graph::isomorphic`] and
//!   [`Graph::canonical_permutation`] (`isomorphism`) compare structure up
//!   to relabeling.
//! - Other ways to cut a graph apart: [`Graph::delete_vertices`] (core),
//!   [`Graph::decompose`] and [`Graph::neighborhood_graphs`] (`components`).
//! - Other randomizations: [`Graph::rewire_edges`] and
//!   [`Graph::degree_sequence_game`] (`games`).
//! - Changing directedness rather than edges: [`Graph::to_directed`] and
//!   [`Graph::to_undirected`] (`conversion`).

use crate::{
    constants::*,
    error::{Error, Result},
    ffi::*,
    graph::{EdgeId, Graph, VertexId},
    igraph_call,
    list::VectorIntList,
    selector::{EdgeSelector, VertexSelector},
    vector::VectorInt,
};
use std::{ffi::c_void, mem::MaybeUninit, ptr};

/// A graph produced by a binary operator, together with the edge maps that
/// relate its edges to the edges of the two operands.
///
/// The meaning (and the length) of the maps depends on the operator: see
/// [`Graph::union_map`], [`Graph::intersection_map`] and [`Graph::compose_map`].
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeMapped {
    /// The resulting graph.
    pub graph: Graph,
    /// Edge map relating the result to the first operand (`self`).
    pub edge_map1: Vec<EdgeId>,
    /// Edge map relating the result to the second operand.
    pub edge_map2: Vec<EdgeId>,
}

/// A graph produced by an operator on many graphs, together with one edge
/// map per operand (see [`Graph::union_many_map`] and
/// [`Graph::intersection_many_map`]).
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeMappedMany {
    /// The resulting graph.
    pub graph: Graph,
    /// `edge_maps[i][e]` is the id, in [`graph`](Self::graph), of edge `e`
    /// of the `i`-th operand, or `None` if that edge has no image.
    pub edge_maps: Vec<Vec<Option<EdgeId>>>,
}

/// An induced subgraph together with the correspondence between its
/// vertices and the vertices of the original graph (see
/// [`Graph::induced_subgraph_map`]).
#[derive(Debug, Clone, PartialEq)]
pub struct InducedSubgraph {
    /// The induced subgraph.
    pub graph: Graph,
    /// `map[v]` is the id in [`graph`](Self::graph) of the original vertex
    /// `v`, or `None` if `v` was not selected. Its length is the vertex
    /// count of the original graph.
    pub map: Vec<Option<VertexId>>,
    /// `invmap[w]` is the original id of vertex `w` of the subgraph. Its
    /// length is the vertex count of the subgraph.
    pub invmap: Vec<VertexId>,
}

/// Statistics of a [`Graph::rewire`] run (`igraph_rewiring_stats_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RewiringStats {
    /// Number of rewiring trials that actually switched a pair of edges.
    pub successful_swaps: usize,
}

/// An `igraph_vector_ptr_t` of *borrowed* graph pointers, destroyed (but
/// without touching the graphs) on drop. The borrow keeps the graphs alive.
struct GraphPtrs<'a> {
    raw: igraph_vector_ptr_t,
    _graphs: Vec<&'a Graph>,
}

impl<'a> GraphPtrs<'a> {
    fn new(graphs: impl IntoIterator<Item = &'a Graph>) -> Result<Self> {
        let graphs: Vec<&'a Graph> = graphs.into_iter().collect();
        let mut raw = MaybeUninit::<igraph_vector_ptr_t>::uninit();
        igraph_call!(igraph_vector_ptr_init(
            raw.as_mut_ptr(),
            graphs.len() as igraph_int_t
        ))?;
        let mut raw = unsafe { raw.assume_init() };
        for (i, g) in graphs.iter().enumerate() {
            // igraph only reads the graphs: the const-to-mut cast is harmless.
            let p = *g as *const Graph as *mut c_void;
            unsafe { igraph_vector_ptr_set(&mut raw, i as igraph_int_t, p) };
        }
        Ok(Self {
            raw,
            _graphs: graphs,
        })
    }

    fn as_ptr(&self) -> *const igraph_vector_ptr_t {
        &self.raw
    }
}

impl Drop for GraphPtrs<'_> {
    fn drop(&mut self) {
        // No item destructor is set: only the pointer array is freed.
        unsafe { igraph_vector_ptr_destroy(&mut self.raw) };
    }
}

/// Runs `f` with the raw `igraph_vs_t` of `sel`.
///
/// Vertex lists are viewed in place (the view stays at a fixed address for
/// the whole call, as `igraph_vss_vector` stores a pointer to it); other
/// selectors go through [`VertexSelector::to_raw`].
fn with_vs<R>(sel: VertexSelector<'_>, f: impl FnOnce(igraph_vs_t) -> Result<R>) -> Result<R> {
    match &sel {
        VertexSelector::List(list) => {
            crate::error::ensure_init();
            let view = VectorInt::view(list);
            let vs = unsafe { igraph_vss_vector(view.as_ptr()) };
            f(vs)
        }
        _ => {
            let raw = sel.to_raw()?;
            f(raw.get())
        }
    }
}

/// Runs `f` with the raw `igraph_es_t` of `sel` (see [`with_vs`]).
fn with_es<R>(sel: EdgeSelector<'_>, f: impl FnOnce(igraph_es_t) -> Result<R>) -> Result<R> {
    match &sel {
        EdgeSelector::List(list) => {
            crate::error::ensure_init();
            let view = VectorInt::view(list);
            let es = unsafe { igraph_ess_vector(view.as_ptr()) };
            f(es)
        }
        _ => {
            let raw = sel.to_raw()?;
            f(raw.get())
        }
    }
}

/// Converts a Rust count into an `igraph_int_t`, rejecting values that do
/// not fit (a plain `as` cast would turn them into negative numbers).
fn to_int(n: usize, what: &str) -> Result<igraph_int_t> {
    igraph_int_t::try_from(n).map_err(|_| Error::invalid(format!("{what} is too large: {n}")))
}

/// Converts igraph's `-1`-means-missing id vectors into `Option`s.
fn optional_ids(ids: &[igraph_int_t]) -> Vec<Option<igraph_int_t>> {
    ids.iter().map(|&i| (i >= 0).then_some(i)).collect()
}

impl igraph_t {
    /// The disjoint union of two graphs.
    ///
    /// The vertices of `other` are relabeled so that the two vertex sets are
    /// disjoint: vertex and edge ids of `self` are unchanged, while those of
    /// `other` are shifted by the vertex and edge counts of `self`. The
    /// result has `|V1|+|V2|` vertices and `|E1|+|E2|` edges.
    ///
    /// See also [`decompose`](Self::decompose), which splits a graph back
    /// into its connected components.
    ///
    /// Binds [`igraph_disjoint_union`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_disjoint_union).
    /// Time complexity: O(|V1|+|V2|+|E1|+|E2|).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the two
    /// graphs do not have the same directedness.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let a = Graph::from_edges(&[(0, 1)], 2, false).unwrap();
    /// let b = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// let u = a.disjoint_union(&b).unwrap();
    /// assert_eq!(u.edge_list(), vec![(0, 1), (2, 3), (3, 4)]);
    /// ```
    pub fn disjoint_union(&self, other: &Graph) -> Result<Graph> {
        Graph::init_with(|res| unsafe { igraph_disjoint_union(res, self, other) })
    }

    /// The disjoint union of many graphs, laid side by side in the given
    /// order (see [`disjoint_union`](Self::disjoint_union)).
    ///
    /// Vertex and edge ids of each operand are shifted by the total vertex
    /// and edge counts of the operands before it. With no operand at all the
    /// result is a *directed* graph with no vertices, as in igraph.
    ///
    /// Binds [`igraph_disjoint_union_many`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_disjoint_union_many).
    /// Time complexity: O(|V|+|E|) of the result.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the
    /// graphs have mixed directedness.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let triangle = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, false).unwrap();
    /// let three = Graph::disjoint_union_many([&triangle, &triangle, &triangle]).unwrap();
    /// assert_eq!((three.vcount(), three.ecount()), (9, 9));
    /// assert_eq!(three.edge(8).unwrap(), (6, 8));
    /// ```
    pub fn disjoint_union_many<'a>(graphs: impl IntoIterator<Item = &'a Graph>) -> Result<Graph> {
        let ptrs = GraphPtrs::new(graphs)?;
        Graph::init_with(|res| unsafe { igraph_disjoint_union_many(res, ptrs.as_ptr()) })
    }

    /// The union of two graphs on the same vertex ids: an edge is in the
    /// result if it is in at least one of the operands.
    ///
    /// The result has as many vertices as the larger operand. Multi-edges are
    /// handled by multiplicity: if `self` has `N` edges between `u` and `v`
    /// and `other` has `M`, the union has `max(N, M)`. Use
    /// [`union_map`](Self::union_map) to also learn where each edge went.
    ///
    /// Binds [`igraph_union`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_union).
    /// Time complexity: O(|V|+|E|) of the result.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the two
    /// graphs do not have the same directedness.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let a = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// let b = Graph::from_edges(&[(2, 1), (2, 3)], 4, false).unwrap();
    /// let u = a.union(&b).unwrap();
    /// assert_eq!(u.vcount(), 4);
    /// assert_eq!(u.edge_list(), vec![(0, 1), (1, 2), (2, 3)]);
    /// ```
    pub fn union(&self, other: &Graph) -> Result<Graph> {
        Graph::init_with(|res| unsafe {
            igraph_union(res, self, other, ptr::null_mut(), ptr::null_mut())
        })
    }

    /// Like [`union`](Self::union), also returning the edge maps.
    ///
    /// In the returned [`EdgeMapped`], `edge_map1[e]` is the id in the union
    /// of edge `e` of `self` (one entry per edge of `self`), and `edge_map2`
    /// is the same for `other`.
    ///
    /// Binds [`igraph_union`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_union).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let a = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    /// let b = Graph::from_edges(&[(1, 2), (2, 0)], 3, true).unwrap();
    /// let u = a.union_map(&b).unwrap();
    /// assert_eq!(u.graph.edge_list(), vec![(0, 1), (1, 2), (2, 0)]);
    /// assert_eq!(u.edge_map1, vec![0, 1]);
    /// assert_eq!(u.edge_map2, vec![1, 2]); // the shared edge 1->2 is edge 1
    /// ```
    pub fn union_map(&self, other: &Graph) -> Result<EdgeMapped> {
        let mut m1 = VectorInt::new();
        let mut m2 = VectorInt::new();
        let graph =
            Graph::init_with(|res| unsafe { igraph_union(res, self, other, &mut m1, &mut m2) })?;
        Ok(EdgeMapped {
            graph,
            edge_map1: m1.into(),
            edge_map2: m2.into(),
        })
    }

    /// The union of many graphs: an edge is in the result if it is in at
    /// least one operand, with the *maximum* of the multiplicities.
    ///
    /// The result has as many vertices as the largest operand. With no
    /// operand at all the result is a *directed* graph with no vertices.
    ///
    /// Binds [`igraph_union_many`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_union_many).
    /// Time complexity: O(|V|+|E|), |V| the vertex count of the largest
    /// graph and |E| the edge count of the result.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the
    /// graphs have mixed directedness.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // The three "spokes" of a star, as three single-edge graphs.
    /// let spokes: Vec<Graph> =
    ///     (1..4).map(|i| Graph::from_edges(&[(0, i)], 0, false).unwrap()).collect();
    /// let star = Graph::union_many(&spokes).unwrap();
    /// let mut edges = star.edge_list();
    /// edges.sort();
    /// assert_eq!(edges, vec![(0, 1), (0, 2), (0, 3)]);
    /// ```
    pub fn union_many<'a>(graphs: impl IntoIterator<Item = &'a Graph>) -> Result<Graph> {
        let ptrs = GraphPtrs::new(graphs)?;
        Graph::init_with(|res| unsafe { igraph_union_many(res, ptrs.as_ptr(), ptr::null_mut()) })
    }

    /// Like [`union_many`](Self::union_many), also returning, for each
    /// operand, the id in the union of each of its edges.
    ///
    /// Binds [`igraph_union_many`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_union_many).
    pub fn union_many_map<'a>(
        graphs: impl IntoIterator<Item = &'a Graph>,
    ) -> Result<EdgeMappedMany> {
        let ptrs = GraphPtrs::new(graphs)?;
        let mut maps = VectorIntList::new();
        let graph =
            Graph::init_with(|res| unsafe { igraph_union_many(res, ptrs.as_ptr(), &mut maps) })?;
        let edge_maps = maps.iter().map(|m| optional_ids(m)).collect();
        Ok(EdgeMappedMany { graph, edge_maps })
    }

    /// The intersection of two graphs on the same vertex ids: an edge is in
    /// the result if it is in both operands.
    ///
    /// The result has as many vertices as the larger operand. Multi-edges are
    /// handled by multiplicity: if `self` has `N` edges between `u` and `v`
    /// and `other` has `M`, the intersection has `min(N, M)`.
    ///
    /// Binds [`igraph_intersection`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_intersection).
    /// Time complexity: O(|V|+|E|), |E| the edge count of the smaller graph.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the two
    /// graphs do not have the same directedness.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let a = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let b = Graph::from_edges(&[(2, 1), (3, 0), (3, 2)], 4, false).unwrap();
    /// assert_eq!(a.intersection(&b).unwrap().edge_list(), vec![(1, 2), (2, 3)]);
    /// ```
    pub fn intersection(&self, other: &Graph) -> Result<Graph> {
        Graph::init_with(|res| unsafe {
            igraph_intersection(res, self, other, ptr::null_mut(), ptr::null_mut())
        })
    }

    /// Like [`intersection`](Self::intersection), also returning the edge maps.
    ///
    /// Note the direction of the maps, which differs from
    /// [`union_map`](Self::union_map): they are indexed by the edges of the
    /// *result*, i.e. `edge_map1[e]` is the id in `self` of edge `e` of the
    /// intersection, and `edge_map2[e]` its id in `other`. Both have as many
    /// entries as the intersection has edges.
    ///
    /// Binds [`igraph_intersection`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_intersection).
    ///
    /// # Examples
    /// Taken from igraph's `igraph_intersection.c` example:
    /// ```
    /// use igraph::prelude::*;
    /// let left = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 0, true).unwrap();
    /// let right = Graph::from_edges(&[(1, 0), (5, 4), (1, 2), (3, 2)], 0, true).unwrap();
    /// let isec = left.intersection_map(&right).unwrap();
    /// assert_eq!(isec.graph.edge_list(), vec![(1, 2)]);
    /// assert_eq!((isec.edge_map1, isec.edge_map2), (vec![1], vec![2]));
    /// ```
    pub fn intersection_map(&self, other: &Graph) -> Result<EdgeMapped> {
        let mut m1 = VectorInt::new();
        let mut m2 = VectorInt::new();
        let graph = Graph::init_with(|res| unsafe {
            igraph_intersection(res, self, other, &mut m1, &mut m2)
        })?;
        Ok(EdgeMapped {
            graph,
            edge_map1: m1.into(),
            edge_map2: m2.into(),
        })
    }

    /// The intersection of many graphs: an edge is in the result if it is in
    /// every operand, with the *minimum* of the multiplicities.
    ///
    /// The result has as many vertices as the largest operand. With no
    /// operand at all the result is a *directed* graph with no vertices.
    ///
    /// Binds [`igraph_intersection_many`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_intersection_many).
    /// Time complexity: O(|V|+|E|), |E| the edge count of the smallest graph.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the
    /// graphs have mixed directedness.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let a = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let b = Graph::from_edges(&[(0, 1), (1, 2)], 4, false).unwrap();
    /// let c = Graph::from_edges(&[(1, 2), (0, 3)], 4, false).unwrap();
    /// assert_eq!(Graph::intersection_many([&a, &b, &c]).unwrap().edge_list(), vec![(1, 2)]);
    /// ```
    pub fn intersection_many<'a>(graphs: impl IntoIterator<Item = &'a Graph>) -> Result<Graph> {
        let ptrs = GraphPtrs::new(graphs)?;
        Graph::init_with(|res| unsafe {
            igraph_intersection_many(res, ptrs.as_ptr(), ptr::null_mut())
        })
    }

    /// Like [`intersection_many`](Self::intersection_many), also returning,
    /// for each operand, the id in the result of each of its edges (`None`
    /// for edges that are not in the intersection).
    ///
    /// Unlike [`intersection_map`](Self::intersection_map), these maps are
    /// indexed by the edges of the *operands*.
    ///
    /// Binds [`igraph_intersection_many`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_intersection_many).
    pub fn intersection_many_map<'a>(
        graphs: impl IntoIterator<Item = &'a Graph>,
    ) -> Result<EdgeMappedMany> {
        let ptrs = GraphPtrs::new(graphs)?;
        let mut maps = VectorIntList::new();
        let graph = Graph::init_with(|res| unsafe {
            igraph_intersection_many(res, ptrs.as_ptr(), &mut maps)
        })?;
        let edge_maps = maps.iter().map(|m| optional_ids(m)).collect();
        Ok(EdgeMappedMany { graph, edge_maps })
    }

    /// The difference of two graphs: the edges of `self` that are not in
    /// `sub`, on the vertex set of `self`.
    ///
    /// Multi-edges are subtracted by multiplicity. The result always has the
    /// same number of vertices as `self`.
    ///
    /// Binds [`igraph_difference`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_difference).
    /// Time complexity: O(|V|+|E|), |V| the vertex count of the smaller graph
    /// and |E| the edge count of the result.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the two
    /// graphs do not have the same directedness.
    ///
    /// # Examples
    /// From igraph's `igraph_difference.c` example:
    /// ```
    /// use igraph::prelude::*;
    /// let orig = Graph::from_edges(&[(0, 1), (1, 2), (2, 1), (4, 5), (8, 9)], 0, true).unwrap();
    /// let sub = Graph::from_edges(&[(0, 1), (5, 4), (2, 1), (6, 7)], 0, true).unwrap();
    /// let diff = orig.difference(&sub).unwrap();
    /// assert_eq!(diff.vcount(), 10);
    /// assert_eq!(diff.edge_list(), vec![(1, 2), (4, 5), (8, 9)]);
    /// ```
    pub fn difference(&self, sub: &Graph) -> Result<Graph> {
        Graph::init_with(|res| unsafe { igraph_difference(res, self, sub) })
    }

    /// The join of two graphs: their [disjoint union](Self::disjoint_union)
    /// plus an edge between every vertex of `self` and every vertex of
    /// `other`.
    ///
    /// The result has `|V1|+|V2|` vertices and `|E1|+|E2|+|V1||V2|` edges;
    /// for directed graphs both `(v, u)` and `(u, v)` are added, i.e.
    /// `|E1|+|E2|+2|V1||V2|` edges. Vertex ids of `other` are shifted by
    /// `|V1|`. For example, the join of an empty graph on `m` vertices and
    /// one on `n` vertices is the complete bipartite graph `K(m,n)`.
    ///
    /// See also [`Graph::full_bipartite`] and [`Graph::wheel`], which build
    /// the classic joins (`K(m,n)`, and a single vertex joined to a cycle)
    /// directly.
    ///
    /// Binds [`igraph_join`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_join).
    /// Time complexity: O(|V1||V2|+|E1|+|E2|).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the two
    /// graphs do not have the same directedness.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // K(2,3) as the join of two edgeless graphs.
    /// let k23 = Graph::new(2, false).join(&Graph::new(3, false)).unwrap();
    /// assert_eq!(k23.ecount(), 6);
    /// assert_eq!(k23.neighbors(0, NeighborMode::All).unwrap(), vec![2, 3, 4]);
    /// assert_eq!(k23, Graph::full_bipartite(2, 3, false, NeighborMode::All).unwrap().graph);
    /// ```
    pub fn join(&self, other: &Graph) -> Result<Graph> {
        Graph::init_with(|res| unsafe { igraph_join(res, self, other) })
    }

    /// The complement of the graph: all the edges that are *not* in `self`.
    ///
    /// With `loops = true` self-loops are added to every vertex that has
    /// none. For directed graphs, edge directions are taken into account.
    /// Multi-edges in the input are treated like single edges.
    ///
    /// See also [`Graph::full`]: the complement of the edgeless graph on `n`
    /// vertices is the complete graph `K_n`.
    ///
    /// Binds [`igraph_complementer`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_complementer).
    /// Time complexity: O(|V|+|E1|+|E2|), |E1| and |E2| the edge counts of
    /// the graph and of its complement.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // The path 0-1-2-3 is self-complementary: its complement is 2-0-3-1.
    /// let p4 = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let mut comp = p4.complementer(false).unwrap().edge_list();
    /// comp.sort();
    /// assert_eq!(comp, vec![(0, 2), (0, 3), (1, 3)]);
    /// ```
    pub fn complementer(&self, loops: bool) -> Result<Graph> {
        Graph::init_with(|res| unsafe { igraph_complementer(res, self, loops) })
    }

    /// The composition `self ∘ other` of two graphs, seen as binary relations.
    ///
    /// The result contains an edge `(i, j)` for every vertex `k` such that
    /// `self` has an edge `(i, k)` and `other` has an edge `(k, j)`: it may
    /// thus contain multi-edges (one per such `k`) and self-loops (e.g.
    /// `(i, i)` from `i -> k` in `self` and `k -> i` in `other`, which for
    /// undirected graphs happens for every edge the two operands share at
    /// `k`). The result has as many vertices as the larger operand.
    ///
    /// Binds [`igraph_compose`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_compose).
    /// Time complexity: O(|V| d1 d2), d1 and d2 the average degrees.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the two
    /// graphs do not have the same directedness.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // "parent of" composed with itself is "grandparent of".
    /// let parent = Graph::from_edges(&[(0, 1), (1, 2), (1, 3), (3, 4)], 5, true).unwrap();
    /// let mut grandparent = parent.compose(&parent).unwrap().edge_list();
    /// grandparent.sort();
    /// assert_eq!(grandparent, vec![(0, 2), (0, 3), (1, 4)]);
    /// ```
    pub fn compose(&self, other: &Graph) -> Result<Graph> {
        Graph::init_with(|res| unsafe {
            igraph_compose(res, self, other, ptr::null_mut(), ptr::null_mut())
        })
    }

    /// Like [`compose`](Self::compose), also returning the edge maps:
    /// `edge_map1[e]` is the edge `(i, k)` of `self`, and `edge_map2[e]` the
    /// edge `(k, j)` of `other`, that generated edge `e` of the result.
    ///
    /// Binds [`igraph_compose`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_compose).
    pub fn compose_map(&self, other: &Graph) -> Result<EdgeMapped> {
        let mut m1 = VectorInt::new();
        let mut m2 = VectorInt::new();
        let graph =
            Graph::init_with(|res| unsafe { igraph_compose(res, self, other, &mut m1, &mut m2) })?;
        Ok(EdgeMapped {
            graph,
            edge_map1: m1.into(),
            edge_map2: m2.into(),
        })
    }

    /// Merges groups of vertices into single vertices, in place.
    ///
    /// `mapping[v]` is the id, in the contracted graph, of the original
    /// vertex `v` (so `mapping` must have one entry per vertex). To avoid
    /// isolated "orphan" vertices, the new ids should be the consecutive
    /// integers `0..k`; the contracted graph has `max(mapping) + 1`
    /// vertices. No edge is removed: edges inside a group become self-loops
    /// and parallel connections between groups become multi-edges; call
    /// [`simplify`](Self::simplify) afterwards to clean them up. Vertex
    /// attributes are discarded (edge and graph attributes are kept): use
    /// [`contract_vertices_with_attributes`](Self::contract_vertices_with_attributes)
    /// to combine the vertex attributes of each group instead.
    ///
    /// A typical `mapping` is a community membership vector, e.g. from
    /// [`community_multilevel`](Self::community_multilevel), which contracts
    /// every community to a single vertex. See also
    /// [`induced_subgraph`](Self::induced_subgraph) to zoom *into* a group
    /// instead.
    ///
    /// Binds [`igraph_contract_vertices`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_contract_vertices).
    /// Time complexity: O(|V|+|E|).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `mapping` does not have one entry per vertex or contains negative ids
    /// or [`VertexId::MAX`] (the vertex count `max(mapping) + 1` would
    /// overflow).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // Two triangles joined by an edge, contracted to two "super-vertices".
    /// let mut g = Graph::from_edges(
    ///     &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)], 6, false,
    /// ).unwrap();
    /// g.contract_vertices(&[0, 0, 0, 1, 1, 1]).unwrap();
    /// assert_eq!((g.vcount(), g.ecount()), (2, 7));
    /// g.simplify(true, true).unwrap();
    /// assert_eq!(g.edge_list(), vec![(0, 1)]);
    /// ```
    pub fn contract_vertices(&mut self, mapping: &[VertexId]) -> Result<()> {
        if mapping.len() != self.vcount() {
            return Err(Error::invalid(format!(
                "the mapping has {} entries but the graph has {} vertices",
                mapping.len(),
                self.vcount()
            )));
        }
        // Negative ids index out of bounds in C, and igraph computes the new
        // vertex count as `max + 1`, which overflows for `VertexId::MAX`.
        if let Some(&bad) = mapping.iter().find(|&&m| !(0..VertexId::MAX).contains(&m)) {
            return Err(Error::invalid(format!(
                "the mapping contains the invalid vertex id {bad}"
            )));
        }
        let m = VectorInt::view(mapping);
        igraph_call!(igraph_contract_vertices(self, m.as_ptr(), ptr::null()))
    }

    /// Relabels the vertices according to a permutation, returning a new
    /// graph.
    ///
    /// `permutation[i]` is the id, *in the original graph*, of the vertex
    /// that becomes vertex `i` of the result. Edge ids are unchanged. Use it
    /// e.g. with a canonical permutation to obtain the canonical form of a
    /// graph.
    ///
    /// See also [`canonical_permutation`](Self::canonical_permutation) and
    /// [`canonical_form`](Self::canonical_form) (`isomorphism`), which
    /// compute such a permutation and apply it.
    ///
    /// Binds [`igraph_permute_vertices`](https://igraph.org/c/html/latest/igraph-Isomorphism.html#igraph_permute_vertices).
    /// Time complexity: O(|V|+|E|).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `permutation` is not a permutation of `0..vcount`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    /// // New vertex 0 is old vertex 2, new 1 is old 0, new 2 is old 1.
    /// let h = g.permute_vertices(&[2, 0, 1]).unwrap();
    /// assert_eq!(h.edge_list(), vec![(1, 2), (2, 0)]);
    ///
    /// // Relabeled copies share the same canonical form.
    /// let canon = |g: &Graph| g.permute_vertices(&g.canonical_permutation(None).unwrap()).unwrap();
    /// assert_eq!(canon(&g), canon(&h));
    /// ```
    pub fn permute_vertices(&self, permutation: &[VertexId]) -> Result<Graph> {
        let p = VectorInt::view(permutation);
        Graph::init_with(|res| unsafe { igraph_permute_vertices(self, res, p.as_ptr()) })
    }

    /// Connects every vertex to all the vertices reachable from it in at
    /// most `order` steps, in place.
    ///
    /// Existing connections are not duplicated, and for undirected graphs a
    /// single edge is added per pair. For directed graphs `mode` tells how to
    /// search: with [`NeighborMode::Out`] each vertex `u` gets an edge
    /// `u -> v` to every `v` it reaches along directed paths; with
    /// [`NeighborMode::In`] every `v` reaching `u` gets an edge `v -> u` (so
    /// the new edges still follow the paths: the same set of edges is added
    /// as with `Out`, possibly in another order); [`NeighborMode::All`]
    /// ignores directions and adds a single edge `u -> v` with `u < v` per
    /// newly connected pair. Orders below 2 leave the graph unchanged. See [`graph_power`](Self::graph_power) for a
    /// non-mutating variant that also simplifies.
    ///
    /// See also [`neighborhood`](Self::neighborhood) (`components`), which
    /// lists the vertices within `order` steps without changing the graph.
    ///
    /// Binds [`igraph_connect_neighborhood`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_connect_neighborhood).
    /// Time complexity: O(|V| d^k), d the average degree and k the order.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `order` does not fit in an `igraph_int_t`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A ring of 6 vertices where everybody also knows the neighbors' neighbors.
    /// let edges: Vec<(i64, i64)> = (0..6).map(|i| (i, (i + 1) % 6)).collect();
    /// let mut g = Graph::from_edges(&edges, 6, false).unwrap();
    /// g.connect_neighborhood(2, NeighborMode::All).unwrap();
    /// assert_eq!(g.ecount(), 12);
    /// assert_eq!(g.degree(.., NeighborMode::All, Loops::Twice).unwrap(), vec![4; 6]);
    /// ```
    pub fn connect_neighborhood(&mut self, order: usize, mode: NeighborMode) -> Result<()> {
        let order = to_int(order, "the order")?;
        // igraph builds an IGRAPH_NO_MULTIPLE adjacency list in `mode`, which
        // trusts the cached "has multi-edges" flag: for a directed graph in
        // `All` mode that flag must not be stale (see
        // `Graph::with_fresh_multi_cache`). The graph is borrowed mutably, so
        // the cache is simply dropped before and after the call.
        self.invalidate_cache();
        let res = igraph_call!(igraph_connect_neighborhood(self, order, mode.into()));
        self.invalidate_cache();
        res
    }

    /// The `order`-th power of the graph.
    ///
    /// It is a simple graph on the same vertices where `u` is connected to
    /// `v` if `v` is reachable from `u` in at most `order` steps. The zeroth
    /// power has no edges; the first power is the graph with multi-edges and
    /// loops removed. With `directed = false` edge directions are ignored and
    /// the result is undirected. Graph and vertex attributes are kept, edge
    /// attributes are discarded.
    ///
    /// For directed inputs this wrapper clears igraph's cached graph
    /// properties around the call, working around an igraph 1.0.1 bug that
    /// could otherwise return a double edge for a mutual pair `u -> v`,
    /// `v -> u` when directions are ignored, or abort the process on a later
    /// call (see the source for details). The result is always simple.
    ///
    /// Binds [`igraph_graph_power`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_graph_power).
    /// Time complexity: O(|V| d^k), d the average degree and k the order.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `order` does not fit in an `igraph_int_t`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // The square of the path 0-1-2-3.
    /// let p4 = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let sq = p4.graph_power(2, false).unwrap();
    /// assert_eq!(sq.ecount(), 5); // everything but (0, 3)
    /// assert_eq!(sq.get_eid(0, 3, false).unwrap(), None);
    /// ```
    pub fn graph_power(&self, order: usize, directed: bool) -> Result<Graph> {
        let order = to_int(order, "the order")?;
        // igraph bug (1.0.0 and 1.0.1): `igraph_graph_power` builds an
        // adjacency list with `igraph_adjlist_init(.., IGRAPH_NO_MULTIPLE)`,
        // which trusts and updates the graph's cached "has multi-edges" flag.
        // With directions ignored (mode ALL), a mutual pair `u -> v`,
        // `v -> u` of a *directed* graph looks like a multi-edge, so:
        // - if "no multi-edges" was already cached (e.g. by `is_simple`),
        //   the deduplication is skipped and the result gets a double edge;
        // - otherwise "has multi-edges" is cached, which is wrong for the
        //   directed graph, and a later check that finds none (e.g. a
        //   directed `graph_power`) fails an igraph assertion and aborts
        //   the process.
        // Dropping the cache of a directed input before the call (whoever
        // wrote it) and after an undirected-mode call (which may have
        // written a wrong value) avoids both; it only costs recomputation.
        let guard = self.is_directed();
        if guard {
            self.invalidate_cache();
        }
        let res = Graph::init_with(|res| unsafe { igraph_graph_power(self, res, order, directed) });
        if guard && !directed {
            self.invalidate_cache();
        }
        res
    }

    /// Randomly rewires the graph in place, preserving its degree sequence.
    ///
    /// Performs `trials` degree-preserving *edge switches*: two edges
    /// `(a, b)` and `(c, d)` are picked uniformly at random and replaced by
    /// `(a, d)` and `(c, b)`, provided the result respects `allowed`:
    /// [`EdgeTypeSw::Simple`] forbids loops and multi-edges,
    /// [`EdgeTypeSw::Loops`] allows (single) self-loops. Multigraphs
    /// ([`EdgeTypeSw::Multi`]) are not supported yet by igraph. For directed
    /// graphs both in- and out-degrees are preserved. All attributes are lost.
    ///
    /// Draws from the calling thread's default random number generator
    /// (each thread has its own): seed it with
    /// [`rng::seed`](crate::rng::seed), or install a private generator with
    /// [`Rng::scoped`](crate::rng::Rng::scoped), for reproducible results.
    /// Returns the number of switches actually performed.
    ///
    /// See also [`rewire_edges`](Self::rewire_edges), which rewires edge
    /// endpoints with a given probability (not preserving degrees), and
    /// [`degree_sequence_game`](Self::degree_sequence_game), which samples
    /// a new graph with a prescribed degree sequence (`games`).
    ///
    /// Binds [`igraph_rewire`](https://igraph.org/c/html/latest/igraph-Games.html#igraph_rewire).
    ///
    /// Graphs with fewer than two edges cannot be rewired: they are left
    /// unchanged and zero swaps are reported.
    ///
    /// # Errors
    /// [`ErrorKind::Unimplemented`](crate::ErrorKind::Unimplemented) for
    /// [`EdgeTypeSw::Multi`], which igraph does not support yet;
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `trials` does not fit in an `igraph_int_t`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// rng::seed(42).unwrap();
    /// let edges: Vec<(i64, i64)> = (0..10).map(|i| (i, (i + 1) % 10)).collect();
    /// let mut g = Graph::from_edges(&edges, 10, false).unwrap();
    /// let stats = g.rewire(100, EdgeTypeSw::Simple).unwrap();
    /// assert!(stats.successful_swaps > 0);
    /// // Still 2-regular.
    /// assert_eq!(g.degree(.., NeighborMode::All, Loops::Twice).unwrap(), vec![2; 10]);
    /// ```
    pub fn rewire(&mut self, trials: usize, allowed: EdgeTypeSw) -> Result<RewiringStats> {
        let mut stats = igraph_rewiring_stats_t {
            successful_swaps: 0,
            unused1_: 0,
            unused2_: 0,
            unused3_: 0,
        };
        let trials = to_int(trials, "the number of trials")?;
        // igraph leaves `stats` untouched when the graph has fewer than two
        // edges: the zero initialization above is then the right answer.
        igraph_call!(igraph_rewire(self, trials, allowed.into(), &mut stats))?;
        Ok(RewiringStats {
            successful_swaps: stats.successful_swaps.max(0) as usize,
        })
    }

    /// Removes multi-edges and/or self-loops, in place.
    ///
    /// With `remove_multiple`, parallel edges are merged into one; with
    /// `remove_loops`, self-loops are deleted. The edge order may change,
    /// even if the graph was already simple.
    ///
    /// With the [attribute handler](crate::attributes::enable) on, graph and
    /// vertex attributes are always kept. Edge attributes are discarded
    /// whenever igraph rebuilds the edge set to merge multi-edges, i.e. when
    /// `remove_multiple` is `true` and igraph does not already know (from
    /// its property cache) that the graph has no multi-edges, even if no
    /// edge actually gets merged. When only loops are removed (or there is
    /// nothing to do) the remaining edges keep their attributes. Use
    /// [`simplify_with_attributes`](Self::simplify_with_attributes) to
    /// combine the attributes of the merged edges instead (e.g. summing
    /// their weights), and
    /// [`contract_vertices_with_attributes`](Self::contract_vertices_with_attributes)
    /// for the analogous vertex operation.
    ///
    /// See also [`is_simple`](Self::is_simple),
    /// [`has_multiple`](Self::has_multiple) and
    /// [`count_multiple`](Self::count_multiple) (`structural`) to inspect
    /// a graph before simplifying it.
    ///
    /// Binds [`igraph_simplify`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_simplify).
    /// Time complexity: O(|V|+|E|).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let mut g = Graph::from_edges(&[(0, 1), (1, 0), (1, 1), (1, 2), (1, 2)], 3, false).unwrap();
    /// let mut keep_loops = g.clone();
    /// keep_loops.simplify(true, false).unwrap();
    /// assert_eq!(keep_loops.edge_list(), vec![(0, 1), (1, 1), (1, 2)]);
    /// g.simplify(true, true).unwrap();
    /// assert_eq!(g.edge_list(), vec![(0, 1), (1, 2)]);
    /// ```
    pub fn simplify(&mut self, remove_multiple: bool, remove_loops: bool) -> Result<()> {
        igraph_call!(igraph_simplify(
            self,
            remove_multiple,
            remove_loops,
            ptr::null()
        ))
    }

    /// The subgraph induced by the selected vertices: those vertices and all
    /// the edges among them.
    ///
    /// Duplicate vertices in the selector are considered once and the
    /// selection order is ignored: the subgraph keeps the vertices in
    /// increasing order of their original ids (so vertex `i` of the subgraph
    /// is the `i`-th smallest selected id). `implementation` picks the
    /// strategy: [`SubgraphImplementation::CopyAndDelete`] is best when
    /// keeping most of the graph, [`SubgraphImplementation::CreateFromScratch`]
    /// when extracting a small part; [`SubgraphImplementation::Auto`] chooses
    /// based on the ratio of the sizes. Use
    /// [`induced_subgraph_map`](Self::induced_subgraph_map) to get the id
    /// correspondence.
    ///
    /// See also [`delete_vertices`](Self::delete_vertices) (the in-place
    /// complement of this operation),
    /// [`neighborhood_graphs`](Self::neighborhood_graphs) and
    /// [`decompose`](Self::decompose) (`components`).
    ///
    /// Binds [`igraph_induced_subgraph`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_induced_subgraph).
    /// Time complexity: O(|V|+|E|) of the original graph.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) for
    /// invalid vertex ids.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], 4, false).unwrap();
    /// let tri = g.induced_subgraph(&[2, 0, 1], SubgraphImplementation::Auto).unwrap();
    /// assert_eq!(tri.edge_list(), vec![(0, 1), (1, 2), (0, 2)]);
    /// ```
    pub fn induced_subgraph<'a>(
        &self,
        vids: impl Into<VertexSelector<'a>>,
        implementation: SubgraphImplementation,
    ) -> Result<Graph> {
        with_vs(vids.into(), |vs| {
            Graph::init_with(|res| unsafe {
                igraph_induced_subgraph(self, res, vs, implementation.into())
            })
        })
    }

    /// Like [`induced_subgraph`](Self::induced_subgraph), also returning the
    /// maps between the original vertex ids and the subgraph's.
    ///
    /// Binds [`igraph_induced_subgraph_map`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_induced_subgraph_map).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4)], 5, false).unwrap();
    /// let sub = g.induced_subgraph_map(&[3, 1, 2], SubgraphImplementation::CreateFromScratch).unwrap();
    /// assert_eq!(sub.invmap, vec![1, 2, 3]);
    /// assert_eq!(sub.map, vec![None, Some(0), Some(1), Some(2), None]);
    /// assert_eq!(sub.graph.edge_list(), vec![(0, 1), (1, 2)]);
    /// ```
    pub fn induced_subgraph_map<'a>(
        &self,
        vids: impl Into<VertexSelector<'a>>,
        implementation: SubgraphImplementation,
    ) -> Result<InducedSubgraph> {
        let mut map = VectorInt::new();
        let mut invmap = VectorInt::new();
        let graph = with_vs(vids.into(), |vs| {
            Graph::init_with(|res| unsafe {
                igraph_induced_subgraph_map(
                    self,
                    res,
                    vs,
                    implementation.into(),
                    &mut map,
                    &mut invmap,
                )
            })
        })?;
        Ok(InducedSubgraph {
            graph,
            map: optional_ids(&map),
            invmap: invmap.into(),
        })
    }

    /// Ids of the edges of the subgraph induced by the selected vertices,
    /// i.e. of the edges having both endpoints in the selection.
    ///
    /// Binds [`igraph_induced_subgraph_edges`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_induced_subgraph_edges).
    /// Time complexity: O(mv log(nv)), nv the number of selected vertices and
    /// mv the sum of their degrees.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) for
    /// invalid vertex ids.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], 4, false).unwrap();
    /// let mut inside = g.induced_subgraph_edges(&[0, 1, 2]).unwrap();
    /// inside.sort();
    /// assert_eq!(inside, vec![0, 1, 4]);
    /// ```
    pub fn induced_subgraph_edges<'a>(
        &self,
        vids: impl Into<VertexSelector<'a>>,
    ) -> Result<Vec<EdgeId>> {
        let mut res = VectorInt::new();
        with_vs(vids.into(), |vs| {
            igraph_call!(igraph_induced_subgraph_edges(self, vs, &mut res))
        })?;
        Ok(res.into())
    }

    /// The subgraph made of the selected edges (and their endpoints).
    ///
    /// Edge ids are reassigned consecutively, keeping the original order.
    /// With `delete_vertices = true`, vertices not incident to any selected
    /// edge are removed as well (and vertex ids reassigned in increasing
    /// order); otherwise the subgraph keeps all the vertices of `self`.
    /// Attributes are preserved.
    ///
    /// Binds [`igraph_subgraph_from_edges`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_subgraph_from_edges).
    /// Time complexity: O(|V|+|E|) of the original graph.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidEdgeId`](crate::ErrorKind::InvalidEdgeId) for
    /// invalid edge ids.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4)], 5, false).unwrap();
    /// let kept = g.subgraph_from_edges(&[1, 2], true).unwrap();
    /// assert_eq!((kept.vcount(), kept.edge_list()), (3, vec![(0, 1), (1, 2)]));
    /// let spanning = g.subgraph_from_edges(&[1, 2], false).unwrap();
    /// assert_eq!((spanning.vcount(), spanning.edge_list()), (5, vec![(1, 2), (2, 3)]));
    /// ```
    pub fn subgraph_from_edges<'a>(
        &self,
        eids: impl Into<EdgeSelector<'a>>,
        delete_vertices: bool,
    ) -> Result<Graph> {
        with_es(eids.into(), |es| {
            Graph::init_with(|res| unsafe {
                igraph_subgraph_from_edges(self, res, es, delete_vertices)
            })
        })
    }

    /// Reverses the direction of the selected edges, in place.
    ///
    /// Attributes and the order of vertices and edges are preserved. Pass
    /// `..` to reverse all edges (this is O(1)); it is rarely needed, since
    /// most functions accept [`NeighborMode::In`] to walk edges backwards.
    /// Undirected graphs are left unchanged (and `eids` is then not even
    /// validated). An edge listed twice is reversed twice, i.e. it ends up
    /// with its original direction.
    ///
    /// Binds [`igraph_reverse_edges`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_reverse_edges).
    /// Time complexity: O(1) for all edges, O(|E|) otherwise.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidEdgeId`](crate::ErrorKind::InvalidEdgeId) for
    /// invalid edge ids.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let mut g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, true).unwrap();
    /// g.reverse_edges(1).unwrap();
    /// assert_eq!(g.edge_list(), vec![(0, 1), (2, 1), (2, 3)]);
    /// g.reverse_edges(..).unwrap();
    /// assert_eq!(g.edge_list(), vec![(1, 0), (1, 2), (3, 2)]);
    /// ```
    pub fn reverse_edges<'a>(&mut self, eids: impl Into<EdgeSelector<'a>>) -> Result<()> {
        with_es(eids.into(), |es| {
            igraph_call!(igraph_reverse_edges(self, es))
        })
    }

    /// A graph product of `self` and `other` (*experimental* in igraph).
    ///
    /// The vertices of the product are the pairs `(u, v)` with `u` in `self`
    /// and `v` in `other`, numbered `u * |V2| + v`. Writing `u ~ u'` for
    /// adjacency, `(u, v)` is connected to `(u', v')` when:
    ///
    /// | [`Product`] | condition | edges (undirected) |
    /// |---|---|---|
    /// | [`Cartesian`](Product::Cartesian) | `u = u'` and `v ~ v'`, or `u ~ u'` and `v = v'` | `\|V1\|\|E2\| + \|V2\|\|E1\|` |
    /// | [`Lexicographic`](Product::Lexicographic) | `u = u'` and `v ~ v'`, or `u ~ u'` | `\|V1\|\|E2\| + \|V2\|²\|E1\|` |
    /// | [`Strong`](Product::Strong) | Cartesian or tensor condition | `\|V1\|\|E2\| + \|V2\|\|E1\| + 2\|E1\|\|E2\|` |
    /// | [`Tensor`](Product::Tensor) | `u ~ u'` and `v ~ v'` | `2\|E1\|\|E2\|` |
    /// | [`Modular`](Product::Modular) | both adjacent or both non-adjacent (simple graphs only) | `2\|E1\|\|E2\| + 2\|E1'\|\|E2'\|` |
    ///
    /// In the directed case the factor 2 disappears. All these products are
    /// associative; the lexicographic one is not commutative.
    ///
    /// See also [`Graph::square_lattice`] and [`Graph::hypercube`]: grids
    /// and hypercubes are iterated Cartesian products of paths, cycles and
    /// `K2`, built directly.
    ///
    /// Binds [`igraph_product`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_product).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the two
    /// graphs have different directedness, or are not simple for the modular
    /// product.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // The 3x4 grid is the Cartesian product of two paths.
    /// let p3 = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// let p4 = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let grid = p3.product(&p4, Product::Cartesian).unwrap();
    /// assert_eq!((grid.vcount(), grid.ecount()), (12, 3 * 3 + 4 * 2));
    /// let lattice = Graph::square_lattice(&[4, 3], 1, false, false, None).unwrap();
    /// assert!(grid.isomorphic(&lattice).unwrap());
    /// ```
    pub fn product(&self, other: &Graph, kind: Product) -> Result<Graph> {
        Graph::init_with(|res| unsafe { igraph_product(res, self, other, kind.into()) })
    }

    /// The rooted product of `self` and `other` with root `root` in `other`
    /// (*experimental* in igraph).
    ///
    /// A copy of `other` is attached to every vertex `u` of `self`, glued at
    /// its root: `(u, v)` is connected to `(u', v')` if `u = u'` and
    /// `v ~ v'`, or `u ~ u'` and `v = v' = root`. Vertex ids follow the same
    /// `u * |V2| + v` convention as [`product`](Self::product); the result
    /// has `|V1||E2| + |E1|` edges.
    ///
    /// Binds [`igraph_rooted_product`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_rooted_product).
    /// Time complexity: O(|V1||V2| + |V1||E2| + |E1|).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) if
    /// `root` is not a vertex of `other`;
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for mixed
    /// directedness.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A "comb": a path with a pendant edge hanging from each vertex.
    /// let spine = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// let tooth = Graph::from_edges(&[(0, 1)], 2, false).unwrap();
    /// let comb = spine.rooted_product(&tooth, 0).unwrap();
    /// assert_eq!((comb.vcount(), comb.ecount()), (6, 5));
    /// assert_eq!(comb.degree(.., NeighborMode::All, Loops::Twice).unwrap(), vec![2, 1, 3, 1, 2, 1]);
    /// ```
    pub fn rooted_product(&self, other: &Graph, root: VertexId) -> Result<Graph> {
        Graph::init_with(|res| unsafe { igraph_rooted_product(res, self, other, root) })
    }

    /// The `k`-times iterated Mycielskian of the graph (*experimental* in igraph).
    ///
    /// Mycielski's construction increases the chromatic number by one while
    /// keeping the graph triangle-free. From `G` with vertices `v_1..v_n` it
    /// builds `M(G)`: `G` itself, a copy `u_i` of every `v_i`, and a new
    /// vertex `w`; each `u_i` is joined to `w` and, for every edge
    /// `(v_i, v_j)`, the edges `(u_i, v_j)` and `(v_i, u_j)` are added. So
    /// `M(G)` has `2n + 1` vertices and `3m + n` edges; after `k` iterations
    /// there are `(n + 1) 2^k - 1` vertices. The Mycielskian of the null
    /// graph is the singleton and that of the singleton is the 2-path, so
    /// that iterating from them yields the connected Mycielski graphs
    /// (the Grötzsch graph after 4 steps from the null graph).
    ///
    /// See also [`Graph::mycielski_graph`], which builds the Mycielski graphs
    /// `M_k` directly; `M_k` is the `(k - 2)`-times iterated Mycielskian of
    /// `K2`, up to relabeling.
    ///
    /// Binds [`igraph_mycielskian`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_mycielskian).
    /// Time complexity: O(|V| 2^k + |E| 3^k).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `k`
    /// does not fit in an `igraph_int_t`, and
    /// [`ErrorKind::Overflow`](crate::ErrorKind::Overflow) if the result would
    /// be too large.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // M(K2) is the 5-cycle, M(C5) is the Grötzsch graph.
    /// let k2 = Graph::from_edges(&[(0, 1)], 2, false).unwrap();
    /// let c5 = k2.mycielskian(1).unwrap();
    /// assert_eq!((c5.vcount(), c5.ecount()), (5, 5));
    /// let grotzsch = k2.mycielskian(2).unwrap();
    /// assert_eq!((grotzsch.vcount(), grotzsch.ecount()), (11, 20));
    /// assert!(grotzsch.isomorphic(&Graph::famous("Grotzsch").unwrap()).unwrap());
    /// assert_eq!(grotzsch.girth().unwrap(), Some(4)); // still triangle-free
    /// ```
    pub fn mycielskian(&self, k: usize) -> Result<Graph> {
        let k = to_int(k, "the number of iterations")?;
        Graph::init_with(|res| unsafe { igraph_mycielskian(self, res, k) })
    }
}

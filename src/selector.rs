//! Vertex and edge selectors.
//!
//! Many igraph functions operate on a *set* of vertices or edges, described
//! by an `igraph_vs_t` / `igraph_es_t` selector. In Rust these are the enums
//! [`VertexSelector`] and [`EdgeSelector`], which convert from convenient
//! values thanks to [`From`]:
//!
//! | Rust value              | Selected vertices          | C constructor |
//! |-------------------------|----------------------------|---------------|
//! | `..` or `VertexSelector::All` | all vertices         | `igraph_vs_all` |
//! | `VertexSelector::None`  | no vertex                  | `igraph_vs_none` |
//! | `3` (an `i64`)          | the single vertex 3        | `igraph_vs_1` |
//! | `&[0, 2, 5]`, `vec![..]`, `&VectorInt` | the listed vertices, in order | `igraph_vss_vector` |
//! | `2..6`                  | vertices 2, 3, 4, 5        | `igraph_vs_range` |
//! | `2..=5`                 | vertices 2, 3, 4, 5        | `igraph_vs_range` |
//! | `VertexSelector::adjacent(v, mode)` | the neighbors of `v` | `igraph_vs_adj` |
//! | `VertexSelector::non_adjacent(v, mode)` | the vertices not adjacent to `v` (including `v` itself unless it has a loop) | `igraph_vs_nonadj` |
//!
//! | Rust value              | Selected edges             | C constructor |
//! |-------------------------|----------------------------|---------------|
//! | `..` or `EdgeSelector::All`, `EdgeSelector::AllOrdered(order)` | all edges, by id or by endpoint | `igraph_es_all` |
//! | `EdgeSelector::None`    | no edge                    | `igraph_es_none` |
//! | `3`, `&[0, 2]`, `vec![..]`, `1..4`, `1..=3` | as for vertices | `igraph_es_1`, `igraph_ess_vector`, `igraph_es_range` |
//! | [`EdgeSelector::incident(v, mode)`](EdgeSelector::incident) | the edges incident to `v` | `igraph_es_incident` |
//! | [`EdgeSelector::pairs(&[(a, b), ..], directed)`](EdgeSelector::pairs) | one edge between each pair | `igraph_es_pairs` |
//! | [`EdgeSelector::path(&[a, b, c, ..], directed)`](EdgeSelector::path) | the edges along a path | `igraph_es_path` |
//! | [`EdgeSelector::all_between(a, b, directed)`](EdgeSelector::all_between) | all the (multi-)edges between two vertices | `igraph_es_all_between` |
//!
//! The immediate constructors (`igraph_vss_none`, `igraph_vss_1`,
//! `igraph_vss_range`, `igraph_ess_all`, `igraph_ess_none`, `igraph_ess_1`,
//! `igraph_ess_range`), the copying ones (`igraph_vs_vector_copy`,
//! `igraph_es_vector_copy`, `igraph_vs_copy`, `igraph_es_copy`), the
//! non-immediate vector ones (`igraph_vs_vector`, `igraph_es_vector`), the
//! variadic `*_small` ones (`igraph_vs_vector_small`, `igraph_es_pairs_small`,
//! `igraph_es_path_small`) and the iterators (`igraph_vit_create`,
//! `igraph_vit_destroy`, `igraph_vit_as_vector`, `igraph_eit_create`,
//! `igraph_eit_destroy`, `igraph_eit_as_vector`) are not wrapped: the
//! constructors in the tables above cover the same selections (a Rust
//! selector is rebuilt, not copied, for every call), and
//! [`Graph::select_vertices`](crate::Graph::select_vertices) /
//! [`Graph::select_edges`](crate::Graph::select_edges) resolve a selector
//! to a [`Vec`] where C code would iterate.
//!
//! The C constructors are documented in the
//! [Iterators](https://igraph.org/c/html/latest/igraph-Iterators.html)
//! chapter of the igraph manual. Selectors are resolved against a graph with
//! [`Graph::select_vertices`](crate::Graph::select_vertices) /
//! [`Graph::select_edges`](crate::Graph::select_edges) and counted with
//! [`Graph::vs_size`](crate::Graph::vs_size) /
//! [`Graph::es_size`](crate::Graph::es_size) (`igraph_iterators.h`).
//!
//! Functions accept `impl Into<VertexSelector<'_>>`, e.g.
//! `graph.degree(.., NeighborMode::All, Loops::Twice)` or
//! `graph.degree(&[0, 1][..], ...)`.
//!
//! ```
//! use igraph::prelude::*;
//!
//! let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false).unwrap();
//! assert_eq!(g.degree(.., NeighborMode::All, Loops::Twice).unwrap(), vec![2, 2, 3, 1]);
//! assert_eq!(g.degree(vec![2, 3], NeighborMode::All, Loops::Twice).unwrap(), vec![3, 1]);
//! let adjacent_to_2 = VertexSelector::Adjacent { vertex: 2, mode: NeighborMode::All };
//! assert_eq!(g.select_vertices(adjacent_to_2).unwrap(), vec![0, 1, 3]);
//!
//! // Edge selectors: remove the triangle's edges along the path 0 - 1 - 2.
//! let mut h = g.clone();
//! h.delete_edges(EdgeSelector::path(&[0, 1, 2], false)).unwrap();
//! assert_eq!(h.edge_list(), vec![(0, 2), (2, 3)]);
//! ```
//!
//! Selectors are accepted all over the crate, e.g. by
//! [`Graph::induced_subgraph`](crate::Graph::induced_subgraph),
//! [`Graph::closeness`](crate::Graph::closeness) or
//! [`Graph::subgraph_from_edges`](crate::Graph::subgraph_from_edges).

use crate::{constants::*, error::Result, ffi::*, vector::VectorInt};
use std::{borrow::Cow, mem::MaybeUninit, ops::Range};

/// A set of vertices, see the [module docs](self).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VertexSelector<'a> {
    /// All vertices, in increasing id order.
    All,
    /// No vertex.
    None,
    /// A single vertex.
    Single(igraph_int_t),
    /// The listed vertices, in the given order (duplicates allowed).
    List(Cow<'a, [igraph_int_t]>),
    /// Vertices with id in `start..end`; like a Rust range it is empty
    /// (and selects nothing, whatever the graph) when `start >= end`.
    Range(igraph_int_t, igraph_int_t),
    /// Neighbors of a vertex, each listed once, in increasing order (loops
    /// and multi-edges are ignored; see
    /// [`Graph::adjacent_vertices`](crate::Graph::adjacent_vertices) for
    /// other conventions).
    Adjacent {
        /// The center vertex.
        vertex: igraph_int_t,
        /// Which neighbors, for directed graphs.
        mode: NeighborMode,
    },
    /// Vertices *not* adjacent to a vertex.
    NonAdjacent {
        /// The center vertex.
        vertex: igraph_int_t,
        /// Which neighbors, for directed graphs.
        mode: NeighborMode,
    },
}

impl From<std::ops::RangeFull> for VertexSelector<'_> {
    fn from(_: std::ops::RangeFull) -> Self {
        Self::All
    }
}

impl From<igraph_int_t> for VertexSelector<'_> {
    fn from(v: igraph_int_t) -> Self {
        Self::Single(v)
    }
}

impl<'a> From<&'a [igraph_int_t]> for VertexSelector<'a> {
    fn from(v: &'a [igraph_int_t]) -> Self {
        Self::List(Cow::Borrowed(v))
    }
}

impl<'a, const N: usize> From<&'a [igraph_int_t; N]> for VertexSelector<'a> {
    fn from(v: &'a [igraph_int_t; N]) -> Self {
        Self::List(Cow::Borrowed(v))
    }
}

impl<'a> From<&'a Vec<igraph_int_t>> for VertexSelector<'a> {
    fn from(v: &'a Vec<igraph_int_t>) -> Self {
        Self::List(Cow::Borrowed(v))
    }
}

impl<'a> From<&'a VectorInt> for VertexSelector<'a> {
    fn from(v: &'a VectorInt) -> Self {
        Self::List(Cow::Borrowed(v.as_slice()))
    }
}

impl From<Vec<igraph_int_t>> for VertexSelector<'_> {
    fn from(v: Vec<igraph_int_t>) -> Self {
        Self::List(Cow::Owned(v))
    }
}

impl From<Range<igraph_int_t>> for VertexSelector<'_> {
    fn from(r: Range<igraph_int_t>) -> Self {
        Self::Range(r.start, r.end)
    }
}

impl From<std::ops::RangeInclusive<igraph_int_t>> for VertexSelector<'_> {
    /// `a..=b` selects the vertices `a, a + 1, ..., b` (none if `a > b`).
    fn from(r: std::ops::RangeInclusive<igraph_int_t>) -> Self {
        if r.is_empty() {
            Self::None
        } else {
            // `b + 1` cannot overflow for a valid id; `i64::MAX` is invalid
            // anyway and still makes igraph report an error.
            Self::Range(*r.start(), r.end().saturating_add(1))
        }
    }
}

impl<'a> VertexSelector<'a> {
    /// The neighbors of `vertex` (shorthand for [`VertexSelector::Adjacent`]).
    pub fn adjacent(vertex: igraph_int_t, mode: NeighborMode) -> Self {
        Self::Adjacent { vertex, mode }
    }

    /// The vertices not adjacent to `vertex` (shorthand for
    /// [`VertexSelector::NonAdjacent`]).
    pub fn non_adjacent(vertex: igraph_int_t, mode: NeighborMode) -> Self {
        Self::NonAdjacent { vertex, mode }
    }

    /// An owned copy of the selector, not borrowing anything.
    pub fn into_owned(self) -> VertexSelector<'static> {
        match self {
            Self::All => VertexSelector::All,
            Self::None => VertexSelector::None,
            Self::Single(v) => VertexSelector::Single(v),
            Self::List(l) => VertexSelector::List(Cow::Owned(l.into_owned())),
            Self::Range(a, b) => VertexSelector::Range(a, b),
            Self::Adjacent { vertex, mode } => VertexSelector::Adjacent { vertex, mode },
            Self::NonAdjacent { vertex, mode } => VertexSelector::NonAdjacent { vertex, mode },
        }
    }
}

/// A raw `igraph_vs_t` together with the storage it points to.
///
/// Created by [`VertexSelector::to_raw`]; call [`RawVs::get`] to obtain the
/// `igraph_vs_t` to pass *by value* to igraph functions. The storage stays
/// alive as long as this value does.
pub struct RawVs {
    vs: igraph_vs_t,
    // Boxed: `vs` may point to this vector, whose address must therefore not
    // change when `RawVs` is moved.
    _storage: Option<Box<VectorInt>>,
}

impl RawVs {
    /// A bitwise copy of the selector, to be passed by value to igraph.
    ///
    /// Selectors built here never own heap memory, so copies are harmless.
    pub fn get(&self) -> igraph_vs_t {
        unsafe { std::ptr::read(&self.vs) }
    }

    /// Pointer to the selector, for functions taking `const igraph_vs_t *`.
    pub fn as_ptr(&self) -> *const igraph_vs_t {
        &self.vs
    }

    /// Whether the selector denotes all vertices
    /// ([`igraph_vs_is_all`](https://igraph.org/c/html/latest/igraph-Iterators.html#igraph_vs_is_all)).
    pub fn is_all(&self) -> bool {
        unsafe { igraph_vs_is_all(&self.vs) }
    }

    /// The raw selector type (`igraph_vs_type`), e.g. `IGRAPH_VS_ALL`.
    pub fn raw_type(&self) -> igraph_vs_type_t {
        unsafe { igraph_vs_type(&self.vs) }
    }
}

impl VertexSelector<'_> {
    /// Builds the raw igraph selector (`igraph_vs_t`), for calling raw FFI
    /// functions; the safe wrappers of the crate do it for you.
    ///
    /// Empty ranges (`start >= end`) become `igraph_vs_none`: igraph 1.0.0
    /// and 1.0.1 would reject an empty range starting at `vcount` and accept
    /// a decreasing one with a negative size.
    ///
    /// # Errors
    /// Only if igraph fails to build the selector (e.g. out of memory);
    /// vertex ids are validated when the selector is used.
    pub fn to_raw(&self) -> Result<RawVs> {
        crate::error::ensure_init();
        let mut vs = MaybeUninit::<igraph_vs_t>::uninit();
        let mut storage = None;
        unsafe {
            match self {
                Self::All => crate::igraph_call!(igraph_vs_all(vs.as_mut_ptr()))?,
                Self::None => crate::igraph_call!(igraph_vs_none(vs.as_mut_ptr()))?,
                Self::Single(v) => crate::igraph_call!(igraph_vs_1(vs.as_mut_ptr(), *v))?,
                Self::List(list) => {
                    // The selector stores a *pointer* to the vector: box it so
                    // that it never moves (a moved-from stack slot would dangle).
                    let v = Box::new(VectorInt::from_slice(list));
                    vs.write(igraph_vss_vector(&*v));
                    storage = Some(v);
                }
                // igraph 1.0.0 and 1.0.1 reject an empty range starting at
                // `vcount` and accept a *decreasing* one, whose size
                // (`end - start`) is then negative: map every empty range to
                // "no vertex".
                Self::Range(start, end) if start >= end => {
                    crate::igraph_call!(igraph_vs_none(vs.as_mut_ptr()))?
                }
                Self::Range(start, end) => {
                    crate::igraph_call!(igraph_vs_range(vs.as_mut_ptr(), *start, *end))?
                }
                Self::Adjacent { vertex, mode } => crate::igraph_call!(igraph_vs_adj(
                    vs.as_mut_ptr(),
                    *vertex,
                    (*mode).into(),
                    Loops::None.into(),
                    false
                ))?,
                Self::NonAdjacent { vertex, mode } => {
                    crate::igraph_call!(igraph_vs_nonadj(vs.as_mut_ptr(), *vertex, (*mode).into()))?
                }
            }
            Ok(RawVs {
                vs: vs.assume_init(),
                _storage: storage,
            })
        }
    }
}

/// A set of edges, see the [module docs](self).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeSelector<'a> {
    /// All edges, ordered by id.
    All,
    /// All edges, in the given order.
    AllOrdered(EdgeOrder),
    /// No edge.
    None,
    /// A single edge.
    Single(igraph_int_t),
    /// The listed edges, in the given order.
    List(Cow<'a, [igraph_int_t]>),
    /// Edges with id in `start..end`; like a Rust range it is empty
    /// (and selects nothing, whatever the graph) when `start >= end`.
    Range(igraph_int_t, igraph_int_t),
    /// Edges incident to a vertex, each loop edge listed once (use
    /// [`Graph::incident_edges`](crate::Graph::incident_edges) to choose how
    /// loops are counted).
    Incident {
        /// The vertex.
        vertex: igraph_int_t,
        /// Which incident edges, for directed graphs.
        mode: NeighborMode,
    },
    /// The edges connecting the given `(from, to)` vertex pairs (one per pair).
    Pairs {
        /// Endpoint pairs.
        pairs: Cow<'a, [(igraph_int_t, igraph_int_t)]>,
        /// Whether to respect edge directions.
        directed: bool,
    },
    /// The edges along a path given as a vertex sequence.
    Path {
        /// The vertices of the path.
        vertices: Cow<'a, [igraph_int_t]>,
        /// Whether to respect edge directions.
        directed: bool,
    },
    /// All the (possibly multiple) edges between two vertices.
    AllBetween {
        /// Source vertex.
        from: igraph_int_t,
        /// Target vertex.
        to: igraph_int_t,
        /// Whether to respect edge directions.
        directed: bool,
    },
}

impl From<std::ops::RangeFull> for EdgeSelector<'_> {
    fn from(_: std::ops::RangeFull) -> Self {
        Self::All
    }
}

impl From<igraph_int_t> for EdgeSelector<'_> {
    fn from(e: igraph_int_t) -> Self {
        Self::Single(e)
    }
}

impl<'a> From<&'a [igraph_int_t]> for EdgeSelector<'a> {
    fn from(e: &'a [igraph_int_t]) -> Self {
        Self::List(Cow::Borrowed(e))
    }
}

impl<'a, const N: usize> From<&'a [igraph_int_t; N]> for EdgeSelector<'a> {
    fn from(e: &'a [igraph_int_t; N]) -> Self {
        Self::List(Cow::Borrowed(e))
    }
}

impl<'a> From<&'a Vec<igraph_int_t>> for EdgeSelector<'a> {
    fn from(e: &'a Vec<igraph_int_t>) -> Self {
        Self::List(Cow::Borrowed(e))
    }
}

impl<'a> From<&'a VectorInt> for EdgeSelector<'a> {
    fn from(e: &'a VectorInt) -> Self {
        Self::List(Cow::Borrowed(e.as_slice()))
    }
}

impl From<Vec<igraph_int_t>> for EdgeSelector<'_> {
    fn from(e: Vec<igraph_int_t>) -> Self {
        Self::List(Cow::Owned(e))
    }
}

impl From<Range<igraph_int_t>> for EdgeSelector<'_> {
    fn from(r: Range<igraph_int_t>) -> Self {
        Self::Range(r.start, r.end)
    }
}

impl From<std::ops::RangeInclusive<igraph_int_t>> for EdgeSelector<'_> {
    /// `a..=b` selects the edges `a, a + 1, ..., b` (none if `a > b`).
    fn from(r: std::ops::RangeInclusive<igraph_int_t>) -> Self {
        if r.is_empty() {
            Self::None
        } else {
            Self::Range(*r.start(), r.end().saturating_add(1))
        }
    }
}

impl<'a> EdgeSelector<'a> {
    /// The edges incident to `vertex` (shorthand for [`EdgeSelector::Incident`]).
    pub fn incident(vertex: igraph_int_t, mode: NeighborMode) -> Self {
        Self::Incident { vertex, mode }
    }

    /// One edge between each `(from, to)` pair (shorthand for
    /// [`EdgeSelector::Pairs`]).
    pub fn pairs(pairs: &'a [(igraph_int_t, igraph_int_t)], directed: bool) -> Self {
        Self::Pairs {
            pairs: Cow::Borrowed(pairs),
            directed,
        }
    }

    /// The edges along a vertex path (shorthand for [`EdgeSelector::Path`]).
    pub fn path(vertices: &'a [igraph_int_t], directed: bool) -> Self {
        Self::Path {
            vertices: Cow::Borrowed(vertices),
            directed,
        }
    }

    /// All edges between two vertices (shorthand for [`EdgeSelector::AllBetween`]).
    pub fn all_between(from: igraph_int_t, to: igraph_int_t, directed: bool) -> Self {
        Self::AllBetween { from, to, directed }
    }
}

/// A raw `igraph_es_t` together with the storage it points to; it is
/// destroyed (with `igraph_es_destroy`) on drop.
pub struct RawEs {
    es: igraph_es_t,
    // Boxed: `es` may point to this vector, whose address must therefore not
    // change when `RawEs` is moved.
    _storage: Option<Box<VectorInt>>,
}

impl RawEs {
    /// A bitwise copy of the selector, to be passed by value to igraph.
    /// The copy must not outlive `self` nor be destroyed.
    pub fn get(&self) -> igraph_es_t {
        unsafe { std::ptr::read(&self.es) }
    }

    /// Pointer to the selector, for functions taking `const igraph_es_t *`.
    pub fn as_ptr(&self) -> *const igraph_es_t {
        &self.es
    }

    /// Whether the selector denotes all edges
    /// ([`igraph_es_is_all`](https://igraph.org/c/html/latest/igraph-Iterators.html#igraph_es_is_all)).
    pub fn is_all(&self) -> bool {
        unsafe { igraph_es_is_all(&self.es) }
    }

    /// The raw selector type (`igraph_es_type`), e.g. `IGRAPH_ES_PAIRS`.
    pub fn raw_type(&self) -> igraph_es_type_t {
        unsafe { igraph_es_type(&self.es) }
    }
}

impl Drop for RawEs {
    fn drop(&mut self) {
        unsafe { igraph_es_destroy(&mut self.es) };
    }
}

impl EdgeSelector<'_> {
    /// Builds the raw igraph selector (`igraph_es_t`), for calling raw FFI
    /// functions; the safe wrappers of the crate do it for you. Empty ranges
    /// select nothing, as for [`VertexSelector::to_raw`].
    ///
    /// # Errors
    /// Only if igraph fails to build the selector (e.g. out of memory); ids
    /// are validated when the selector is used.
    pub fn to_raw(&self) -> Result<RawEs> {
        crate::error::ensure_init();
        let mut es = MaybeUninit::<igraph_es_t>::uninit();
        let mut storage = None;
        unsafe {
            match self {
                Self::All => {
                    crate::igraph_call!(igraph_es_all(es.as_mut_ptr(), EdgeOrder::Id.into()))?
                }
                Self::AllOrdered(order) => {
                    crate::igraph_call!(igraph_es_all(es.as_mut_ptr(), (*order).into()))?
                }
                Self::None => crate::igraph_call!(igraph_es_none(es.as_mut_ptr()))?,
                Self::Single(e) => crate::igraph_call!(igraph_es_1(es.as_mut_ptr(), *e))?,
                Self::List(list) => {
                    // The selector stores a *pointer* to the vector: box it so
                    // that it never moves (a moved-from stack slot would dangle).
                    let v = Box::new(VectorInt::from_slice(list));
                    es.write(igraph_ess_vector(&*v));
                    storage = Some(v);
                }
                // See `VertexSelector::to_raw`: empty ranges select nothing.
                Self::Range(start, end) if start >= end => {
                    crate::igraph_call!(igraph_es_none(es.as_mut_ptr()))?
                }
                Self::Range(start, end) => {
                    crate::igraph_call!(igraph_es_range(es.as_mut_ptr(), *start, *end))?
                }
                Self::Incident { vertex, mode } => crate::igraph_call!(igraph_es_incident(
                    es.as_mut_ptr(),
                    *vertex,
                    (*mode).into(),
                    // A selector denotes a *set* of edges: list loops once
                    // (with `Twice`, deleting them would name them twice).
                    Loops::Once.into()
                ))?,
                Self::Pairs { pairs, directed } => {
                    let flat: VectorInt = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
                    crate::igraph_call!(igraph_es_pairs(es.as_mut_ptr(), &flat, *directed))?
                }
                Self::Path { vertices, directed } => {
                    let v = VectorInt::from_slice(vertices);
                    crate::igraph_call!(igraph_es_path(es.as_mut_ptr(), &v, *directed))?
                }
                Self::AllBetween { from, to, directed } => crate::igraph_call!(
                    igraph_es_all_between(es.as_mut_ptr(), *from, *to, *directed)
                )?,
            }
            Ok(RawEs {
                es: es.assume_init(),
                _storage: storage,
            })
        }
    }
}

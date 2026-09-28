//! The [`Graph`] type and igraph's basic interface (`igraph_interface.h`).
//!
//! [`Graph`] is the C struct `igraph_t` itself, made Rusty: it owns its data
//! (destroyed on [`Drop`]), it is [`Clone`] (via `igraph_copy`) and [`Send`],
//! and its methods return [`Result`]s instead of error codes. Vertices and
//! edges are identified by consecutive integer ids starting from zero, as in
//! igraph; ids are `i64` (`igraph_int_t`) while counts are `usize`.
//!
//! This module contains the fundamental operations: construction, adding
//! and removing vertices and edges, and queries about the structure.
//! Algorithms live in the other modules of the crate, most of them as further
//! methods of [`Graph`].
//!
//! ```
//! use igraph::prelude::*;
//!
//! // A directed triangle plus a pendant vertex.
//! let mut g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 4, true).unwrap();
//! g.add_edge(2, 3).unwrap();
//! assert_eq!((g.vcount(), g.ecount()), (4, 4));
//! assert_eq!(g.neighbors(2, NeighborMode::Out).unwrap(), vec![0, 3]);
//! assert_eq!(g.edge(3).unwrap(), (2, 3));
//! assert_eq!(g.get_eid(1, 2, true).unwrap(), Some(1));
//! assert_eq!(g.get_eid(2, 1, true).unwrap(), None);
//!
//! let h = g.clone();
//! g.delete_vertices(3).unwrap();
//! assert_eq!((g.vcount(), h.vcount()), (3, 4));
//! ```
//!
//! The functions of `igraph_interface.h` map as follows:
//!
//! | C function | Rust |
//! |------------|------|
//! | `igraph_empty`, `igraph_create` | [`Graph::new`], [`Graph::empty`], [`Graph::from_edges`], [`Graph::from_flat_edges`] |
//! | `igraph_empty_attrs` | [`Graph::empty`] followed by [`set_graph_attr`](Graph::set_graph_attr) (see [`attributes`](crate::attributes)) |
//! | `igraph_copy`, `igraph_destroy` | [`Clone`], [`Graph::try_clone`], [`Drop`] |
//! | `igraph_add_vertices`, `igraph_add_edge(s)` | [`add_vertices`](Graph::add_vertices), [`add_vertex`](Graph::add_vertex), [`add_edge`](Graph::add_edge), [`add_edge_id`](Graph::add_edge_id), [`add_edges`](Graph::add_edges), [`add_edges_from_slice`](Graph::add_edges_from_slice), [`add_edges_from_vector`](Graph::add_edges_from_vector) |
//! | `igraph_delete_vertices(_map)`, `igraph_delete_edges` | [`delete_vertices`](Graph::delete_vertices), [`delete_vertices_map`](Graph::delete_vertices_map), [`delete_edges`](Graph::delete_edges) |
//! | `igraph_vcount`, `igraph_ecount`, `igraph_is_directed` | [`vcount`](Graph::vcount), [`ecount`](Graph::ecount), [`num_vertices`](Graph::num_vertices), [`num_edges`](Graph::num_edges), [`vertices`](Graph::vertices), [`edge_ids`](Graph::edge_ids), [`is_directed`](Graph::is_directed) |
//! | `igraph_neighbors`, `igraph_incident` | [`neighbors`](Graph::neighbors), [`neighbors_with`](Graph::neighbors_with), [`incident`](Graph::incident) |
//! | `igraph_degree(_1)` | [`degree`](Graph::degree), [`degree_of`](Graph::degree_of) |
//! | `igraph_edge(s)` | [`edge`](Graph::edge), [`edges`](Graph::edges), [`edges_flat`](Graph::edges_flat), [`edge_list`](Graph::edge_list), [`edges_iter`](Graph::edges_iter) |
//! | `IGRAPH_FROM`, `IGRAPH_TO`, `IGRAPH_OTHER` | [`edge_from`](Graph::edge_from), [`edge_to`](Graph::edge_to), [`other_endpoint`](Graph::other_endpoint) |
//! | `igraph_get_eid(s)`, `igraph_get_all_eids_between` | [`get_eid`](Graph::get_eid), [`has_edge`](Graph::has_edge), [`get_eids`](Graph::get_eids), [`get_eids_opt`](Graph::get_eids_opt), [`get_all_eids_between`](Graph::get_all_eids_between) |
//! | `igraph_is_same_graph` | [`is_same_graph`](Graph::is_same_graph), [`PartialEq`] |
//! | `igraph_invalidate_cache` | [`invalidate_cache`](Graph::invalidate_cache) |
//! | `igraph_vs_*`, `igraph_es_*` (`igraph_iterators.h`) | [`select_vertices`](Graph::select_vertices), [`select_edges`](Graph::select_edges), [`vs_size`](Graph::vs_size), [`es_size`](Graph::es_size), [`adjacent_vertices`](Graph::adjacent_vertices), [`incident_edges`](Graph::incident_edges) |
//!
//! For raw FFI users, [`Graph::init_with`] turns any C function that
//! initializes an `igraph_t` into a `Result<Graph>`, and [`Graph::setup`]
//! initializes igraph for the calling thread (every safe wrapper does it
//! automatically, see [`crate::error::ensure_init`]).
//!
//! `Graph` is [`Send`] (it can be moved to another thread) but not [`Sync`]:
//! igraph lazily caches properties inside the graph even through `const`
//! pointers.
//!
//! # Where to go next
//!
//! Most graphs are not built edge by edge:
//!
//! - deterministic graphs (rings, lattices, trees, the named graphs of
//!   [`Graph::famous`], ...) are in [`constructors`](crate::constructors),
//!   random graph models in [`games`](crate::games), and file readers and
//!   writers in [`foreign`](crate::foreign);
//! - conversions to and from matrices and edge lists are in
//!   [`conversion`](crate::conversion) (e.g. [`Graph::get_adjacency`]) and
//!   [`constructors`](crate::constructors) (e.g. [`Graph::adjacency`]);
//! - structural queries beyond degrees (strength, simplicity, multi-edges,
//!   density, ...) are in [`structural`](crate::structural), subgraphs and
//!   simplification in [`operators`](crate::operators), and vertex, edge and
//!   graph attributes in [`attributes`](crate::attributes);
//! - for repeated neighborhood queries, [`adjlist`](crate::adjlist) builds
//!   adjacency and incidence lists once.
//!
//! ```
//! use igraph::prelude::*;
//!
//! // Zachary's karate club, one of igraph's named graphs.
//! let club = Graph::famous("Zachary").unwrap();
//! assert_eq!((club.vcount(), club.ecount()), (34, 78));
//! let degree = club.degree(.., NeighborMode::All, Loops::Twice).unwrap();
//! // The handshake lemma: every edge has two endpoints.
//! assert_eq!(degree.iter().sum::<i64>(), 2 * 78);
//! // The instructor (0) and the administrator (33) are the hubs.
//! assert_eq!((degree[0], degree[33]), (16, 17));
//! assert_eq!(club.maxdegree(.., NeighborMode::All, Loops::Twice).unwrap(), 17);
//! ```

use crate::{
    constants::*,
    error::{Error, Result, ensure_init},
    ffi::*,
    igraph_call,
    selector::{EdgeSelector, VertexSelector},
    vector::VectorInt,
};
use std::{fmt, mem::MaybeUninit, ops::Range};

/// An igraph graph (`igraph_t`), see the [module docs](self).
pub type Graph = igraph_t;

/// Vertex identifier (`igraph_int_t`).
pub type VertexId = igraph_int_t;
/// Edge identifier (`igraph_int_t`).
pub type EdgeId = igraph_int_t;

impl Drop for igraph_t {
    /// Destroys the graph with `igraph_destroy`.
    fn drop(&mut self) {
        // A zeroed (never initialized) graph has null storage: nothing to free.
        if !self.from.stor_begin.is_null() {
            unsafe { igraph_destroy(self) };
        }
    }
}

impl Clone for igraph_t {
    /// Deep copy with `igraph_copy`.
    fn clone(&self) -> Self {
        Self::init_with(|g| unsafe { igraph_copy(g, self) }).expect("igraph failed to copy a graph")
    }
}

// A graph owns its data; the (lazily filled) property cache is mutated even
// through `&Graph`, hence `Send` but not `Sync`.
unsafe impl Send for igraph_t {}

impl fmt::Display for igraph_t {
    /// A one-line summary, e.g. `Undirected graph with 3 vertices and 2 edges`.
    ///
    /// The alternate form (`{:#}`) appends the edge list, one edge per line,
    /// as `from -> to` (directed) or `from -- to` (undirected):
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// assert_eq!(g.to_string(), "Undirected graph with 3 vertices and 2 edges");
    /// assert_eq!(format!("{g:#}"), "Undirected graph with 3 vertices and 2 edges\n0 -- 1\n1 -- 2");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} graph with {} vertices and {} edges",
            if self.is_directed() {
                "Directed"
            } else {
                "Undirected"
            },
            self.vcount(),
            self.ecount()
        )?;
        if f.alternate() {
            let arrow = if self.is_directed() { "->" } else { "--" };
            for (from, to) in self.edge_list() {
                write!(f, "\n{from} {arrow} {to}")?;
            }
        }
        Ok(())
    }
}

impl PartialEq for igraph_t {
    /// Two graphs are equal when they are the same *labelled* graph: same
    /// directedness, vertex count and multiset of edges written in terms of
    /// vertex ids (the order of the edges, and of the endpoints of undirected
    /// edges, does not matter), see [`is_same_graph`](Graph::is_same_graph).
    /// Use the isomorphism functions to compare structure up to relabeling.
    fn eq(&self, other: &Self) -> bool {
        self.is_same_graph(other).unwrap_or(false)
    }
}

impl igraph_t {
    /// Initializes igraph for the calling thread (see [`crate::error::ensure_init`]).
    ///
    /// Calling it is optional: every safe wrapper does it automatically.
    pub fn setup() {
        ensure_init();
    }

    /// Creates a graph by running an igraph *constructor*, i.e. a C function
    /// that initializes the `igraph_t` pointed to by its argument.
    ///
    /// This is the building block of all the wrappers returning new graphs,
    /// and it is useful to call raw constructors not wrapped by this crate.
    /// `f` must either fail or fully initialize the graph; if it reports
    /// success without initializing it, [`ErrorKind::Internal`](crate::error::ErrorKind::Internal)
    /// is returned:
    ///
    /// ```
    /// use igraph::{ffi, prelude::*};
    /// let ring = Graph::init_with(|g| unsafe { ffi::igraph_ring(g, 5, false, false, true) }).unwrap();
    /// assert_eq!(ring.ecount(), 5);
    /// ```
    pub fn init_with(f: impl FnOnce(*mut igraph_t) -> igraph_error_t) -> Result<Self> {
        // Like `igraph_call!`: initializes the thread, forgets any stale
        // error record, and refuses to nest too deeply inside callbacks.
        crate::error::prepare_call()?;
        // Zeroed storage is safe to drop (see `Drop`), should `f` fail early.
        let mut graph = MaybeUninit::<igraph_t>::zeroed();
        crate::error::check(f(graph.as_mut_ptr()))?;
        // On failure igraph has already released whatever it allocated. On
        // success, make sure the closure really initialized the graph: a
        // (safe) closure returning `IGRAPH_SUCCESS` without calling a
        // constructor would otherwise hand out a graph with null storage.
        let graph = unsafe { graph.assume_init() };
        if graph.from.stor_begin.is_null() {
            return Err(Error::new(
                crate::error::ErrorKind::Internal,
                "the constructor passed to Graph::init_with did not initialize the graph",
            ));
        }
        Ok(graph)
    }

    /// Creates a graph with `num_vertices` isolated vertices ([`igraph_empty`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_empty)).
    ///
    /// # Panics
    /// If igraph fails to allocate the graph, like [`Vec`] does, or if
    /// `num_vertices` exceeds igraph's maximum vertex count (see
    /// [`empty`](Self::empty) for the fallible version).
    pub fn new(num_vertices: usize, directed: bool) -> Self {
        Self::empty(num_vertices, directed).expect("igraph failed to create a graph")
    }

    /// Creates a graph with `num_vertices` isolated vertices ([`igraph_empty`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_empty)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if
    /// `num_vertices` does not fit in an `i64`;
    /// [`ErrorKind::Range`](crate::error::ErrorKind::Range) if it exceeds
    /// igraph's maximum vertex count, or
    /// [`ErrorKind::OutOfMemory`](crate::error::ErrorKind::OutOfMemory).
    pub fn empty(num_vertices: usize, directed: bool) -> Result<Self> {
        let n = count_arg(num_vertices, "number of vertices")?;
        Self::init_with(|g| unsafe { igraph_empty(g, n, directed) })
    }

    /// Creates a graph from a list of `(from, to)` edges
    /// ([`igraph_create`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_create)).
    ///
    /// The graph has `max(num_vertices, largest id + 1)` vertices; in an
    /// undirected graph `(a, b)` and `(b, a)` denote the same edge. Multi-edges
    /// and loops are kept; edge `i` is the `i`-th pair. See
    /// [`constructors`](crate::constructors) for ready-made graphs, and
    /// [`Graph::read_graph_edgelist_from_str`] to parse an edge list.
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 5, true).unwrap();
    /// assert_eq!((g.vcount(), g.ecount()), (5, 2));
    /// // The vertex count grows to fit the largest id.
    /// let h = Graph::from_edges(&[(0, 7)], 0, false).unwrap();
    /// assert_eq!(h.vcount(), 8);
    /// ```
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId) for negative ids.
    pub fn from_edges(
        edges: &[(VertexId, VertexId)],
        num_vertices: usize,
        directed: bool,
    ) -> Result<Self> {
        let flat: VectorInt = edges.iter().flat_map(|&(a, b)| [a, b]).collect();
        Self::from_flat_edges(&flat, num_vertices, directed)
    }

    /// Creates a graph from a flat edge list `[from0, to0, from1, to1, ...]`
    /// ([`igraph_create`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_create)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) for an odd length
    /// or a `num_vertices` that does not fit in an `i64`,
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId) for negative ids.
    pub fn from_flat_edges(
        edges: &[VertexId],
        num_vertices: usize,
        directed: bool,
    ) -> Result<Self> {
        if !edges.len().is_multiple_of(2) {
            return Err(Error::invalid(
                "the flat edge list must have an even length",
            ));
        }
        let n = count_arg(num_vertices, "number of vertices")?;
        let view = VectorInt::view(edges);
        Self::init_with(|g| unsafe { igraph_create(g, view.as_ptr(), n, directed) })
    }

    /// Number of vertices ([`igraph_vcount`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_vcount)), O(1).
    pub fn vcount(&self) -> usize {
        unsafe { igraph_vcount(self) as usize }
    }

    /// Number of edges ([`igraph_ecount`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_ecount)), O(1).
    pub fn ecount(&self) -> usize {
        unsafe { igraph_ecount(self) as usize }
    }

    /// Number of vertices; alias of [`vcount`](Self::vcount).
    pub fn num_vertices(&self) -> usize {
        self.vcount()
    }

    /// Number of edges; alias of [`ecount`](Self::ecount).
    pub fn num_edges(&self) -> usize {
        self.ecount()
    }

    /// Whether the graph is directed ([`igraph_is_directed`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_is_directed)).
    pub fn is_directed(&self) -> bool {
        unsafe { igraph_is_directed(self) }
    }

    /// The range of all vertex ids, `0..vcount`.
    pub fn vertices(&self) -> Range<VertexId> {
        0..self.vcount() as VertexId
    }

    /// The range of all edge ids, `0..ecount`.
    pub fn edge_ids(&self) -> Range<EdgeId> {
        0..self.ecount() as EdgeId
    }

    /// Adds `n` isolated vertices, with ids `vcount..vcount + n`
    /// ([`igraph_add_vertices`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_add_vertices)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if
    /// `n` does not fit in an `i64`;
    /// [`ErrorKind::Overflow`](crate::error::ErrorKind::Overflow) or
    /// [`ErrorKind::Range`](crate::error::ErrorKind::Range) if the new vertex
    /// count would overflow or exceed igraph's maximum vertex count.
    pub fn add_vertices(&mut self, n: usize) -> Result<()> {
        let n = count_arg(n, "number of vertices to add")?;
        igraph_call!(igraph_add_vertices(self, n, std::ptr::null()))
    }

    /// Adds a single edge ([`igraph_add_edge`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_add_edge)); for many edges
    /// [`add_edges`](Self::add_edges) is much faster.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId) if an endpoint does not exist.
    pub fn add_edge(&mut self, from: VertexId, to: VertexId) -> Result<()> {
        igraph_call!(igraph_add_edge(self, from, to))
    }

    /// Adds the given `(from, to)` edges, with ids `ecount..` in order
    /// ([`igraph_add_edges`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_add_edges)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId) if an endpoint does not
    /// exist (then no edge is added).
    pub fn add_edges(&mut self, edges: &[(VertexId, VertexId)]) -> Result<()> {
        let flat: VectorInt = edges.iter().flat_map(|&(a, b)| [a, b]).collect();
        self.add_edges_from_vector(&flat)
    }

    /// Adds the given `(from, to)` edges; alias of [`add_edges`](Self::add_edges).
    pub fn add_edges_from_slice(&mut self, edges: &[(VertexId, VertexId)]) -> Result<()> {
        self.add_edges(edges)
    }

    /// Adds the edges of a flat list `[from0, to0, from1, to1, ...]` ([`igraph_add_edges`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_add_edges)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) for an odd length,
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId) for missing vertices.
    pub fn add_edges_from_vector(&mut self, edges: &[VertexId]) -> Result<()> {
        if !edges.len().is_multiple_of(2) {
            return Err(Error::invalid(
                "the flat edge list must have an even length",
            ));
        }
        let view = VectorInt::view(edges);
        igraph_call!(igraph_add_edges(self, view.as_ptr(), std::ptr::null()))
    }

    /// Removes the selected edges ([`igraph_delete_edges`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_delete_edges)); the
    /// remaining edges keep their relative order and are renumbered to stay
    /// consecutive.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidEdgeId`](crate::error::ErrorKind::InvalidEdgeId) for invalid edges.
    pub fn delete_edges<'a>(&mut self, edges: impl Into<EdgeSelector<'a>>) -> Result<()> {
        let es = edges.into().to_raw()?;
        igraph_call!(igraph_delete_edges(self, es.get()))
    }

    /// Removes the selected vertices and their incident edges
    /// ([`igraph_delete_vertices`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_delete_vertices)); the remaining vertices and edges are
    /// renumbered to stay consecutive (see
    /// [`delete_vertices_map`](Self::delete_vertices_map) for the mapping).
    /// To keep the original graph, build the complementary
    /// [`Graph::induced_subgraph`] instead.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId) for invalid vertices.
    pub fn delete_vertices<'a>(&mut self, vertices: impl Into<VertexSelector<'a>>) -> Result<()> {
        let vs = vertices.into().to_raw()?;
        igraph_call!(igraph_delete_vertices(self, vs.get()))
    }

    /// Removes the selected vertices and their incident edges, returning
    /// `(map, invmap)`
    /// ([`igraph_delete_vertices_map`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_delete_vertices_map)):
    /// `map[old]` is the new id of vertex `old`, or `-1` if it was deleted,
    /// and `invmap[new]` is the old id of vertex `new`.
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let mut g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let (map, invmap) = g.delete_vertices_map(&[1]).unwrap();
    /// assert_eq!(map, vec![0, -1, 1, 2]);
    /// assert_eq!(invmap, vec![0, 2, 3]);
    /// assert_eq!(g.edge_list(), vec![(1, 2)]);
    /// ```
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId) for invalid vertices.
    pub fn delete_vertices_map<'a>(
        &mut self,
        vertices: impl Into<VertexSelector<'a>>,
    ) -> Result<(Vec<VertexId>, Vec<VertexId>)> {
        let vs = vertices.into().to_raw()?;
        let mut map = VectorInt::new();
        let mut invmap = VectorInt::new();
        igraph_call!(igraph_delete_vertices_map(
            self,
            vs.get(),
            &mut map,
            &mut invmap
        ))?;
        Ok((map.into(), invmap.into()))
    }

    /// Neighbors of a vertex, sorted, counting loop edges twice and keeping
    /// multi-edges, so that the result has as many entries as the degree
    /// ([`igraph_neighbors`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_neighbors) with `IGRAPH_LOOPS_TWICE`
    /// and `IGRAPH_MULTIPLE`); see [`neighbors_with`](Self::neighbors_with)
    /// for other conventions.
    ///
    /// For many queries on the same graph, an
    /// [`AdjList`](crate::adjlist::AdjList) is faster; for the vertices
    /// within a given distance, see [`Graph::neighborhood`].
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let star = Graph::star(4, StarMode::Undirected, 0).unwrap();
    /// assert_eq!(star.neighbors(0, NeighborMode::All).unwrap(), vec![1, 2, 3]);
    /// assert_eq!(star.neighbors(2, NeighborMode::All).unwrap(), vec![0]);
    /// ```
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId) for an invalid vertex.
    pub fn neighbors(&self, vertex: VertexId, mode: NeighborMode) -> Result<Vec<VertexId>> {
        self.neighbors_with(vertex, mode, Loops::Twice, true)
    }

    /// Neighbors of a vertex with full control over loops and multi-edges
    /// ([`igraph_neighbors`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_neighbors)). The result is sorted.
    ///
    /// `mode` selects out-, in- or all neighbors in directed graphs (it is
    /// ignored for undirected ones); `loops` says whether a loop edge makes
    /// the vertex its own neighbor zero, one or two times; with
    /// `multiple = false` each neighbor appears once however many parallel
    /// edges lead to it.
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 0), (0, 1), (0, 1)], 2, false).unwrap();
    /// assert_eq!(g.neighbors_with(0, NeighborMode::All, Loops::Twice, true).unwrap(), vec![0, 0, 1, 1]);
    /// assert_eq!(g.neighbors_with(0, NeighborMode::All, Loops::None, false).unwrap(), vec![1]);
    /// ```
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId) for an invalid vertex.
    pub fn neighbors_with(
        &self,
        vertex: VertexId,
        mode: NeighborMode,
        loops: Loops,
        multiple: bool,
    ) -> Result<Vec<VertexId>> {
        let mut res = VectorInt::new();
        igraph_call!(igraph_neighbors(
            self,
            &mut res,
            vertex,
            mode.into(),
            loops.into(),
            multiple
        ))?;
        Ok(res.into())
    }

    /// Ids of the edges incident to a vertex, ordered like the neighbors
    /// of [`neighbors_with`](Self::neighbors_with) ([`igraph_incident`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_incident)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId) for an invalid vertex.
    pub fn incident(
        &self,
        vertex: VertexId,
        mode: NeighborMode,
        loops: Loops,
    ) -> Result<Vec<EdgeId>> {
        let mut res = VectorInt::new();
        igraph_call!(igraph_incident(
            self,
            &mut res,
            vertex,
            mode.into(),
            loops.into()
        ))?;
        Ok(res.into())
    }

    /// Degrees of the selected vertices, in selector order
    /// ([`igraph_degree`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_degree)); `loops` says whether a loop edge counts zero,
    /// one or two times (two is the convention of the handshake lemma).
    /// `mode` selects out-, in- or total degrees in directed graphs.
    ///
    /// See also [`Graph::strength`] (weighted degrees), [`Graph::maxdegree`]
    /// and [`Graph::mean_degree`].
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // A loop at 0 and a double edge 0 - 1.
    /// let g = Graph::from_edges(&[(0, 0), (0, 1), (0, 1)], 2, false).unwrap();
    /// assert_eq!(g.degree(.., NeighborMode::All, Loops::Twice).unwrap(), vec![4, 2]);
    /// assert_eq!(g.degree(.., NeighborMode::All, Loops::Once).unwrap(), vec![3, 2]);
    /// assert_eq!(g.degree(&[0], NeighborMode::All, Loops::None).unwrap(), vec![2]);
    /// ```
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId) for invalid vertices.
    pub fn degree<'a>(
        &self,
        vertices: impl Into<VertexSelector<'a>>,
        mode: NeighborMode,
        loops: Loops,
    ) -> Result<Vec<igraph_int_t>> {
        let vs = vertices.into().to_raw()?;
        let mut res = VectorInt::new();
        igraph_call!(igraph_degree(
            self,
            &mut res,
            vs.get(),
            mode.into(),
            loops.into()
        ))?;
        Ok(res.into())
    }

    /// Degree of a single vertex, with the conventions of
    /// [`degree`](Self::degree) but faster (O(1) when loops count twice,
    /// O(degree) otherwise)
    /// ([`igraph_degree_1`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_degree_1)).
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let wheel = Graph::wheel(6, WheelMode::Undirected, 0).unwrap();
    /// assert_eq!(wheel.degree_of(0, NeighborMode::All, Loops::Twice).unwrap(), 5);
    /// assert_eq!(wheel.degree_of(3, NeighborMode::All, Loops::Twice).unwrap(), 3);
    /// assert!(wheel.degree_of(6, NeighborMode::All, Loops::Twice).is_err());
    /// ```
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId) for an invalid vertex.
    pub fn degree_of(
        &self,
        vertex: VertexId,
        mode: NeighborMode,
        loops: Loops,
    ) -> Result<igraph_int_t> {
        // igraph 1.0.0 and 1.0.1 do not validate `vid` in `igraph_degree_1`
        // (it indexes the graph's index vectors directly): check it here.
        if !self.vertices().contains(&vertex) {
            return Err(Error::new(
                crate::error::ErrorKind::InvalidVertexId,
                format!(
                    "vertex {vertex} is not in the graph (it has {} vertices)",
                    self.vcount()
                ),
            ));
        }
        let mut deg = 0;
        igraph_call!(igraph_degree_1(
            self,
            &mut deg,
            vertex,
            mode.into(),
            loops.into()
        ))?;
        Ok(deg)
    }

    /// Endpoints `(from, to)` of an edge ([`igraph_edge`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_edge)); for
    /// undirected graphs `from <= to`.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidEdgeId`](crate::error::ErrorKind::InvalidEdgeId) for an invalid edge.
    pub fn edge(&self, edge: EdgeId) -> Result<(VertexId, VertexId)> {
        let (mut from, mut to) = (0, 0);
        igraph_call!(igraph_edge(self, edge, &mut from, &mut to))?;
        Ok((from, to))
    }

    /// Endpoints of the selected edges, in selector order ([`igraph_edges`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_edges)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidEdgeId`](crate::error::ErrorKind::InvalidEdgeId) for invalid edges.
    pub fn edges<'a>(
        &self,
        edges: impl Into<EdgeSelector<'a>>,
    ) -> Result<Vec<(VertexId, VertexId)>> {
        let es = edges.into().to_raw()?;
        let mut res = VectorInt::new();
        igraph_call!(igraph_edges(self, es.get(), &mut res, false))?;
        Ok(res
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[a, b]| (a, b))
            .collect())
    }

    /// All the edges as `(from, to)` pairs, in edge id order (for
    /// undirected graphs `from <= to`).
    ///
    /// See [`edges_flat`](Self::edges_flat) and [`Graph::get_edgelist`] for
    /// a flat list, and [`Graph::write_graph_edgelist_to_string`] for the
    /// text format.
    pub fn edge_list(&self) -> Vec<(VertexId, VertexId)> {
        self.edges(EdgeSelector::All)
            .expect("listing all edges cannot fail")
    }

    /// Id of an edge between two vertices, or `None` if they are not
    /// connected ([`igraph_get_eid`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_get_eid)). With `directed = false`, edge
    /// directions are ignored in directed graphs. With multi-edges, any one
    /// of them may be returned.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId) for invalid vertices.
    pub fn get_eid(&self, from: VertexId, to: VertexId, directed: bool) -> Result<Option<EdgeId>> {
        let mut eid = -1;
        igraph_call!(igraph_get_eid(self, &mut eid, from, to, directed, false))?;
        Ok((eid >= 0).then_some(eid))
    }

    /// Ids of the edges connecting the given vertex pairs ([`igraph_get_eids`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_get_eids));
    /// see [`get_eids_opt`](Self::get_eids_opt) to tolerate missing edges.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if some pair is not
    /// connected, [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId) for invalid vertices.
    pub fn get_eids(&self, pairs: &[(VertexId, VertexId)], directed: bool) -> Result<Vec<EdgeId>> {
        let flat: VectorInt = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
        let mut res = VectorInt::new();
        igraph_call!(igraph_get_eids(self, &mut res, &flat, directed, true))?;
        Ok(res.into())
    }

    /// Ids of all the (multi-)edges between two vertices
    /// ([`igraph_get_all_eids_between`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_get_all_eids_between)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId) for invalid vertices.
    pub fn get_all_eids_between(
        &self,
        from: VertexId,
        to: VertexId,
        directed: bool,
    ) -> Result<Vec<EdgeId>> {
        let mut res = VectorInt::new();
        igraph_call!(igraph_get_all_eids_between(
            self, &mut res, from, to, directed
        ))?;
        Ok(res.into())
    }

    /// Whether two graphs are the same labelled graph: same directedness,
    /// same number of vertices and precisely the same edges, regardless of
    /// their order ([`igraph_is_same_graph`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_is_same_graph)).
    /// It is what `==` on graphs computes. To compare graphs up to a
    /// relabeling of their vertices, use [`Graph::isomorphic`].
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let a = Graph::from_edges(&[(0, 1), (2, 1)], 3, false).unwrap();
    /// let b = Graph::from_edges(&[(1, 2), (0, 1)], 3, false).unwrap();
    /// let c = Graph::from_edges(&[(0, 2), (1, 2)], 3, false).unwrap(); // isomorphic only
    /// assert!(a.is_same_graph(&b).unwrap());
    /// assert!(!a.is_same_graph(&c).unwrap());
    /// ```
    pub fn is_same_graph(&self, other: &Self) -> Result<bool> {
        let mut res = false;
        igraph_call!(igraph_is_same_graph(self, other, &mut res))?;
        Ok(res)
    }

    /// Resolves a vertex selector into the list of vertex ids it denotes
    /// (`igraph_vs_as_vector`, undocumented in `igraph_iterators.h`).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId) (or
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) for ranges) if the selector names
    /// missing vertices.
    pub fn select_vertices<'a>(
        &self,
        vertices: impl Into<VertexSelector<'a>>,
    ) -> Result<Vec<VertexId>> {
        let vs = vertices.into().to_raw()?;
        let mut res = VectorInt::new();
        igraph_call!(igraph_vs_as_vector(self, vs.get(), &mut res))?;
        Ok(res.into())
    }

    /// Resolves an edge selector into the list of edge ids it denotes
    /// ([`igraph_es_as_vector`](https://igraph.org/c/html/latest/igraph-Iterators.html#igraph_es_as_vector)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidEdgeId`](crate::error::ErrorKind::InvalidEdgeId) for missing edges, and
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) for vertex pairs that are not connected.
    pub fn select_edges<'a>(&self, edges: impl Into<EdgeSelector<'a>>) -> Result<Vec<EdgeId>> {
        let es = edges.into().to_raw()?;
        let mut res = VectorInt::new();
        igraph_call!(igraph_es_as_vector(self, es.get(), &mut res))?;
        Ok(res.into())
    }

    /// Invalidates igraph's internal cache of graph properties
    /// ([`igraph_invalidate_cache`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_invalidate_cache)); only needed after unsafe raw mutation.
    pub fn invalidate_cache(&self) {
        unsafe { igraph_invalidate_cache(self) }
    }

    /// Runs `f` (typically one igraph call) so that igraph's property cache
    /// cannot corrupt an adjacency list built in `IGRAPH_ALL` mode with
    /// `IGRAPH_NO_MULTIPLE`.
    ///
    /// In igraph 1.0.0 and 1.0.1, `igraph_adjlist_init` and
    /// `igraph_lazy_adjlist_init` skip the removal of duplicate neighbors when
    /// the cache says the graph has no multi-edges. For a *directed* graph
    /// with a mutual pair `u -> v`, `v -> u` that is wrong in `IGRAPH_ALL`
    /// mode (each endpoint lists the other twice), and algorithms that rely
    /// on distinct neighbors then fail assertions or corrupt memory;
    /// `igraph_adjlist_init` may even cache a wrong `HAS_MULTI = true`.
    /// Affected C functions include cliques, independent sets, coloring,
    /// girth, chordality, triangles and transitivity, `igraph_ecc`,
    /// eccentricity/radius/center/pseudo-diameter, simple paths, local scan,
    /// Jaccard/Dice similarity, graph power and Reingold–Tilford.
    ///
    /// For directed graphs the cache is dropped before `f` and again after
    /// it, even if `f` panics; undirected graphs are unaffected and `f` runs
    /// directly. It only costs recomputing cached properties. `Graph` is
    /// `Send` but not `Sync`, so no other thread observes the cache meanwhile.
    pub(crate) fn with_fresh_multi_cache<T>(&self, f: impl FnOnce() -> T) -> T {
        struct Invalidate<'a>(&'a igraph_t);
        impl Drop for Invalidate<'_> {
            fn drop(&mut self) {
                self.0.invalidate_cache();
            }
        }
        if !self.is_directed() {
            return f();
        }
        self.invalidate_cache();
        let _after = Invalidate(self);
        f()
    }

    /// A deep copy of the graph, reporting allocation failures as an
    /// error instead of panicking like [`Clone::clone`]
    /// ([`igraph_copy`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_copy)).
    pub fn try_clone(&self) -> Result<Self> {
        Self::init_with(|g| unsafe { igraph_copy(g, self) })
    }

    /// Adds one isolated vertex and returns its id
    /// ([`igraph_add_vertices`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_add_vertices)).
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let mut g = Graph::new(2, false);
    /// let v = g.add_vertex().unwrap();
    /// g.add_edge(0, v).unwrap();
    /// assert_eq!((v, g.vcount()), (2, 3));
    /// ```
    pub fn add_vertex(&mut self) -> Result<VertexId> {
        let id = self.vcount() as VertexId;
        self.add_vertices(1)?;
        Ok(id)
    }

    /// Adds an edge and returns its id (edges are appended, so the new id
    /// is the previous edge count)
    /// ([`igraph_add_edge`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_add_edge)).
    pub fn add_edge_id(&mut self, from: VertexId, to: VertexId) -> Result<EdgeId> {
        let id = self.ecount() as EdgeId;
        self.add_edge(from, to)?;
        Ok(id)
    }

    /// Whether some edge connects `from` and `to` (in this direction if
    /// `directed` is true and the graph is directed)
    /// ([`igraph_get_eid`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_get_eid)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId) for invalid vertices.
    pub fn has_edge(&self, from: VertexId, to: VertexId, directed: bool) -> Result<bool> {
        Ok(self.get_eid(from, to, directed)?.is_some())
    }

    /// Ids of the edges connecting the given vertex pairs, with `None` for
    /// the pairs that are not connected
    /// ([`igraph_get_eids`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_get_eids)
    /// with `error = false`).
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// assert_eq!(g.get_eids_opt(&[(2, 1), (0, 2)], true).unwrap(), vec![Some(1), None]);
    /// ```
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId) for invalid vertices.
    pub fn get_eids_opt(
        &self,
        pairs: &[(VertexId, VertexId)],
        directed: bool,
    ) -> Result<Vec<Option<EdgeId>>> {
        let flat: VectorInt = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
        let mut res = VectorInt::new();
        igraph_call!(igraph_get_eids(self, &mut res, &flat, directed, false))?;
        Ok(res.iter().map(|&e| (e >= 0).then_some(e)).collect())
    }

    /// Endpoints of the selected edges as a flat list; with `bycol = false`
    /// it is `[from0, to0, from1, to1, ...]`, with `bycol = true` it is
    /// `[from0, from1, ..., to0, to1, ...]`
    /// ([`igraph_edges`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_edges)).
    pub fn edges_flat<'a>(
        &self,
        edges: impl Into<EdgeSelector<'a>>,
        bycol: bool,
    ) -> Result<Vec<VertexId>> {
        let es = edges.into().to_raw()?;
        let mut res = VectorInt::new();
        igraph_call!(igraph_edges(self, es.get(), &mut res, bycol))?;
        Ok(res.into())
    }

    /// The source vertex of an edge (`IGRAPH_FROM`).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidEdgeId`](crate::error::ErrorKind::InvalidEdgeId) for an invalid edge.
    pub fn edge_from(&self, edge: EdgeId) -> Result<VertexId> {
        Ok(self.edge(edge)?.0)
    }

    /// The target vertex of an edge (`IGRAPH_TO`).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidEdgeId`](crate::error::ErrorKind::InvalidEdgeId) for an invalid edge.
    pub fn edge_to(&self, edge: EdgeId) -> Result<VertexId> {
        Ok(self.edge(edge)?.1)
    }

    /// The endpoint of `edge` opposite to `vertex` (`IGRAPH_OTHER`); for a
    /// loop edge it is `vertex` itself.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidEdgeId`](crate::error::ErrorKind::InvalidEdgeId) for an invalid edge, and
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if `vertex` is not an
    /// endpoint of `edge`.
    pub fn other_endpoint(&self, edge: EdgeId, vertex: VertexId) -> Result<VertexId> {
        let (from, to) = self.edge(edge)?;
        if to == vertex {
            Ok(from)
        } else if from == vertex {
            Ok(to)
        } else {
            Err(Error::invalid(format!(
                "vertex {vertex} is not an endpoint of edge {edge}"
            )))
        }
    }

    /// Number of vertices in a vertex selector for this graph
    /// ([`igraph_vs_size`](https://igraph.org/c/html/latest/igraph-Iterators.html#igraph_vs_size)).
    ///
    /// This only *counts*, it does not validate: lists and ranges are
    /// counted by their length (duplicates included) even if they name
    /// missing vertices, and a single missing vertex counts as `0`. Only
    /// the adjacent and non-adjacent selectors query the graph, and fail for
    /// a missing center.
    /// Use [`select_vertices`](Self::select_vertices) to resolve and
    /// validate a selector.
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::ring(5, false, false, true).unwrap();
    /// assert_eq!(g.vs_size(..).unwrap(), 5);
    /// assert_eq!(g.vs_size(&[1, 1, 4]).unwrap(), 3);
    /// assert_eq!(g.vs_size(VertexSelector::non_adjacent(0, NeighborMode::All)).unwrap(), 3);
    /// assert_eq!(g.vs_size(9).unwrap(), 0); // not validated
    /// assert!(g.select_vertices(9).is_err());
    /// ```
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId)
    /// for an adjacent or non-adjacent selector around a missing vertex.
    pub fn vs_size<'a>(&self, vertices: impl Into<VertexSelector<'a>>) -> Result<usize> {
        let vs = vertices.into().to_raw()?;
        let mut n = 0;
        igraph_call!(igraph_vs_size(self, vs.as_ptr(), &mut n))?;
        Ok(n as usize)
    }

    /// Number of edges in an edge selector for this graph
    /// ([`igraph_es_size`](https://igraph.org/c/html/latest/igraph-Iterators.html#igraph_es_size)).
    ///
    /// Like [`vs_size`](Self::vs_size), lists and ranges are counted by
    /// their length without validation (and a single missing edge counts as
    /// `0`); incident, pair, path and all-between selectors query the graph.
    /// Use [`select_edges`](Self::select_edges) to resolve and validate a
    /// selector.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId)
    /// for selectors around missing vertices,
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) for
    /// pairs that are not connected.
    pub fn es_size<'a>(&self, edges: impl Into<EdgeSelector<'a>>) -> Result<usize> {
        let es = edges.into().to_raw()?;
        let mut n = 0;
        igraph_call!(igraph_es_size(self, es.as_ptr(), &mut n))?;
        Ok(n as usize)
    }

    /// Neighbors of the vertex through an adjacency *selector*
    /// ([`igraph_vs_adj`](https://igraph.org/c/html/latest/igraph-Iterators.html#igraph_vs_adj) resolved with
    /// `igraph_vs_as_vector`, which is undocumented in `igraph_iterators.h`), with full control of
    /// loops and multi-edges; the result equals
    /// [`neighbors_with`](Self::neighbors_with) with the same arguments.
    /// Unlike [`VertexSelector::Adjacent`], which always ignores loops and
    /// multi-edges, the conventions are configurable.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId) for an invalid vertex.
    pub fn adjacent_vertices(
        &self,
        vertex: VertexId,
        mode: NeighborMode,
        loops: Loops,
        multiple: bool,
    ) -> Result<Vec<VertexId>> {
        ensure_init();
        let mut vs = MaybeUninit::<igraph_vs_t>::uninit();
        igraph_call!(igraph_vs_adj(
            vs.as_mut_ptr(),
            vertex,
            mode.into(),
            loops.into(),
            multiple
        ))?;
        // Adjacency selectors own no memory: no `igraph_vs_destroy` needed.
        let vs = unsafe { vs.assume_init() };
        let mut res = VectorInt::new();
        igraph_call!(igraph_vs_as_vector(self, vs, &mut res))?;
        Ok(res.into())
    }

    /// Incident edges of the vertex with the given loop handling, in
    /// selector order ([`igraph_es_incident`](https://igraph.org/c/html/latest/igraph-Iterators.html#igraph_es_incident) resolved
    /// with [`igraph_es_as_vector`](https://igraph.org/c/html/latest/igraph-Iterators.html#igraph_es_as_vector)); unlike
    /// [`EdgeSelector::Incident`], which lists every loop once, the loop
    /// counting mode is configurable. The result equals
    /// [`incident`](Self::incident) with the same arguments.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::error::ErrorKind::InvalidVertexId) for an invalid vertex.
    pub fn incident_edges(
        &self,
        vertex: VertexId,
        mode: NeighborMode,
        loops: Loops,
    ) -> Result<Vec<EdgeId>> {
        ensure_init();
        let mut es = MaybeUninit::<igraph_es_t>::uninit();
        igraph_call!(igraph_es_incident(
            es.as_mut_ptr(),
            vertex,
            mode.into(),
            loops.into()
        ))?;
        let mut es = unsafe { es.assume_init() };
        let mut res = VectorInt::new();
        let r = igraph_call!(igraph_es_as_vector(self, std::ptr::read(&es), &mut res));
        unsafe { igraph_es_destroy(&mut es) };
        r?;
        Ok(res.into())
    }

    /// Iterates over the edges as `(id, from, to)` triples, in id order.
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    /// let e: Vec<_> = g.edges_iter().collect();
    /// assert_eq!(e, vec![(0, 0, 1), (1, 1, 2)]);
    /// ```
    pub fn edges_iter(&self) -> impl Iterator<Item = (EdgeId, VertexId, VertexId)> + '_ {
        // `from`/`to` are igraph's edge arrays (see `IGRAPH_FROM`/`IGRAPH_TO`);
        // like `igraph_edge`, report undirected edges as `(smaller, larger)`,
        // igraph stores them the other way around.
        let directed = self.is_directed();
        self.from
            .iter()
            .zip(self.to.iter())
            .enumerate()
            .map(move |(e, (&a, &b))| {
                if directed {
                    (e as EdgeId, a, b)
                } else {
                    (e as EdgeId, b, a)
                }
            })
    }
}

/// Converts a count given as `usize` into an `igraph_int_t`, reporting an
/// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) error
/// (instead of wrapping around to a negative number) when it does not fit.
fn count_arg(n: usize, what: &str) -> Result<igraph_int_t> {
    igraph_int_t::try_from(n)
        .map_err(|_| Error::invalid(format!("{what} ({n}) does not fit in an igraph_int_t")))
}

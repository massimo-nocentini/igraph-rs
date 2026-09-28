//! Connectivity: connected components, cut vertices and bridges, vertex
//! separators, cohesive blocks, reachability and neighborhoods.
//!
//! This module binds five igraph headers:
//!
//! - `igraph_components.h`: (weakly or strongly) connected components,
//!   decomposition into component graphs, articulation points, bridges,
//!   biconnected components and percolation curves;
//! - `igraph_separators.h`: vertex separators (sets of vertices whose removal
//!   disconnects the graph);
//! - `igraph_cohesive_blocks.h`: the Moody–White hierarchy of cohesive blocks;
//! - `igraph_reachability.h`: which vertices can reach which others, and the
//!   transitive closure;
//! - `igraph_neighborhood.h`: the vertices within a given distance of some
//!   vertices, as sets, counts or induced subgraphs.
//!
//! All the graph algorithms are methods of [`Graph`]; the only free function
//! is [`edgelist_percolation`]. The randomized percolation curves draw from
//! the calling thread's default random number generator, so they are
//! reproducible after [`rng::seed`](crate::rng::seed).
//!
//! # Example
//!
//! ```
//! use igraph::prelude::*;
//!
//! // Two triangles joined by the edge 2-3, plus an isolated vertex 6.
//! let g = Graph::from_edges(
//!     &[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)],
//!     7,
//!     false,
//! )
//! .unwrap();
//!
//! let cc = g.connected_components(Connectedness::Weak).unwrap();
//! assert_eq!(cc.count, 2);
//! assert_eq!(cc.sizes, vec![6, 1]);
//! assert!(!g.is_connected(Connectedness::Weak).unwrap());
//!
//! // The edge 2-3 (id 3) is the only bridge; 2 and 3 are the cut vertices.
//! assert_eq!(g.bridges().unwrap(), vec![3]);
//! let mut cut = g.articulation_points().unwrap();
//! cut.sort();
//! assert_eq!(cut, vec![2, 3]);
//!
//! // Removing vertex 2 separates vertex 0 from vertex 5.
//! assert!(g.is_separator(2).unwrap());
//!
//! // Vertices at distance at most 1 from vertex 3.
//! assert_eq!(g.neighborhood(3, Some(1), NeighborMode::All, 0).unwrap(), vec![vec![3, 2, 4, 5]]);
//!
//! // Zachary's karate club: one component, a single cut vertex (the
//! // instructor, 0) and a nested hierarchy of cohesive blocks.
//! let karate = Graph::famous("Zachary").unwrap();
//! assert!(karate.is_connected(Connectedness::Weak).unwrap());
//! assert_eq!(karate.articulation_points().unwrap(), vec![0]);
//! let blocks = karate.cohesive_blocks().unwrap();
//! assert_eq!(blocks.cohesion, vec![1, 2, 2, 4, 3, 3, 4, 3]);
//! ```
//!
//! # Provided functionality
//!
//! | Rust | C function | Purpose |
//! |------|------------|---------|
//! | [`Graph::connected_components`] | `igraph_connected_components` | membership, sizes and number of components |
//! | [`Graph::is_connected`] | `igraph_is_connected` | weak / strong connectedness test |
//! | [`Graph::decompose`] | `igraph_decompose` | one [`Graph`] per component |
//! | [`Graph::articulation_points`] | `igraph_articulation_points` | cut vertices |
//! | [`Graph::bridges`] | `igraph_bridges` | cut edges |
//! | [`Graph::biconnected_components`] | `igraph_biconnected_components` | [`BiconnectedComponents`] |
//! | [`Graph::is_biconnected`] | `igraph_is_biconnected` | 2-vertex-connectedness test |
//! | [`Graph::bond_percolation`] | `igraph_bond_percolation` | giant component while adding edges |
//! | [`Graph::site_percolation`] | `igraph_site_percolation` | giant component while adding vertices |
//! | [`edgelist_percolation`] | `igraph_edgelist_percolation` | bond percolation of a bare edge list |
//! | [`Graph::is_separator`] | `igraph_is_separator` | does removing a vertex set disconnect the graph? |
//! | [`Graph::is_minimal_separator`] | `igraph_is_minimal_separator` | ... and no proper subset does? |
//! | [`Graph::all_minimal_st_separators`] | `igraph_all_minimal_st_separators` | all minimal (s,t) separators |
//! | [`Graph::minimum_size_separators`] | `igraph_minimum_size_separators` | all minimum-size vertex separators |
//! | [`Graph::cohesive_blocks`] | `igraph_cohesive_blocks` | [`CohesiveBlocks`] hierarchy |
//! | [`Graph::reachability`] | `igraph_reachability` | [`Reachability`] bitsets per strong component |
//! | [`Graph::count_reachable`] | `igraph_count_reachable` | number of reachable vertices |
//! | [`Graph::transitive_closure`] | `igraph_transitive_closure` | the transitive closure graph |
//! | [`Graph::neighborhood_size`] | `igraph_neighborhood_size` | sizes of the k-neighborhoods |
//! | [`Graph::neighborhood`] | `igraph_neighborhood` | vertices of the k-neighborhoods |
//! | [`Graph::neighborhood_graphs`] | `igraph_neighborhood_graphs` | induced subgraphs of the k-neighborhoods |
//!
//! # See also
//!
//! - Single-source reachability: [`Graph::subcomponent`]; traversals:
//!   [`Graph::bfs`], [`Graph::dfs`]; distances:
//!   [`Graph::distances`].
//! - Quantitative connectivity (minimum cuts, Menger paths):
//!   [`Graph::vertex_connectivity`], [`Graph::edge_connectivity`],
//!   [`Graph::st_vertex_connectivity`], [`Graph::all_st_mincuts`] and
//!   [`Graph::dominator_tree`] in [`flow`](crate::flow).
//! - Forests, DAGs and orderings: [`Graph::is_tree`], [`Graph::is_forest`],
//!   [`Graph::is_dag`], [`Graph::topological_sorting`].
//! - Subgraphs of components or neighborhoods: [`Graph::induced_subgraph`];
//!   graphs whose edges are k-neighborhoods: [`Graph::connect_neighborhood`],
//!   [`Graph::graph_power`].
//! - Another nested "cohesion" decomposition: [`Graph::coreness`]
//!   (k-cores).

use crate::{
    bitset::BitsetList,
    constants::{Connectedness, NeighborMode},
    error::{Error, ErrorKind, Result},
    ffi::*,
    graph::{EdgeId, Graph, VertexId},
    igraph_call,
    list::{GraphList, VectorIntList},
    selector::VertexSelector,
    vector::VectorInt,
};

fn to_usizes(v: VectorInt) -> Vec<usize> {
    v.iter().map(|&x| x as usize).collect()
}

/// Converts a count to `igraph_int_t`, saturating instead of wrapping to a
/// negative value (which igraph would interpret differently).
fn int_sat(x: usize) -> igraph_int_t {
    igraph_int_t::try_from(x).unwrap_or(igraph_int_t::MAX)
}

/// Converts an optional non-negative order / limit into igraph's convention
/// where a negative value means "unlimited".
fn order_raw(order: Option<usize>) -> igraph_int_t {
    order.map_or(-1, int_sat)
}

// ---------------------------------------------------------------------------
// Result structs
// ---------------------------------------------------------------------------

/// The connected components of a graph, see [`Graph::connected_components`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectedComponents {
    /// `membership[v]` is the id (in `0..count`) of the component of vertex `v`.
    pub membership: Vec<VertexId>,
    /// `sizes[c]` is the number of vertices in component `c`.
    pub sizes: Vec<usize>,
    /// The number of components.
    pub count: usize,
}

impl ConnectedComponents {
    /// The vertices of component `c`, in increasing id order (empty if `c`
    /// is out of range).
    pub fn members(&self, c: usize) -> Vec<VertexId> {
        self.membership
            .iter()
            .enumerate()
            .filter(|&(_, &m)| m as usize == c)
            .map(|(v, _)| v as VertexId)
            .collect()
    }

    /// All the components as vertex lists, indexed by component id.
    pub fn groups(&self) -> Vec<Vec<VertexId>> {
        let mut groups: Vec<Vec<VertexId>> =
            self.sizes.iter().map(|&s| Vec::with_capacity(s)).collect();
        for (v, &m) in self.membership.iter().enumerate() {
            groups[m as usize].push(v as VertexId);
        }
        groups
    }

    /// Id of a largest ("giant") component, the first one on ties; `None`
    /// for the null graph.
    pub fn largest(&self) -> Option<usize> {
        let max = *self.sizes.iter().max()?;
        self.sizes.iter().position(|&s| s == max)
    }

    /// Whether vertices `u` and `v` lie in the same component (`false` for
    /// out-of-range ids).
    pub fn same_component(&self, u: VertexId, v: VertexId) -> bool {
        let get = |x: VertexId| usize::try_from(x).ok().and_then(|i| self.membership.get(i));
        matches!((get(u), get(v)), (Some(a), Some(b)) if a == b)
    }
}

/// The biconnected components of a graph, see
/// [`Graph::biconnected_components`].
///
/// All the lists are indexed by component id, in `0..count`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BiconnectedComponents {
    /// The number of biconnected components.
    pub count: usize,
    /// For each component, the edges of one of its spanning trees.
    pub tree_edges: Vec<Vec<EdgeId>>,
    /// For each component, all of its edges. Every non-loop edge of the graph
    /// belongs to exactly one component.
    pub component_edges: Vec<Vec<EdgeId>>,
    /// For each component, its vertices. A vertex may belong to several
    /// components (exactly when it is an articulation point) and isolated
    /// vertices belong to none.
    pub components: Vec<Vec<VertexId>>,
    /// The articulation points (cut vertices) of the graph.
    pub articulation_points: Vec<VertexId>,
}

/// A bond (edge) percolation curve, see [`Graph::bond_percolation`] and
/// [`edgelist_percolation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BondPercolation {
    /// `giant_size[i]` is the size of the largest component after the first
    /// `i + 1` edges have been added.
    pub giant_size: Vec<usize>,
    /// `vertex_count[i]` is the number of vertices having at least one
    /// incident edge after the first `i + 1` edges have been added.
    pub vertex_count: Vec<usize>,
}

/// A site (vertex) percolation curve, see [`Graph::site_percolation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SitePercolation {
    /// `giant_size[i]` is the size of the largest component after the first
    /// `i + 1` vertices have been added.
    pub giant_size: Vec<usize>,
    /// `edge_count[i]` is the number of edges among the first `i + 1` added
    /// vertices.
    pub edge_count: Vec<usize>,
}

/// The cohesive block hierarchy of a graph, see [`Graph::cohesive_blocks`].
///
/// Block `0` is always the whole graph; the lists are indexed by block id.
#[derive(Debug, Clone, PartialEq)]
pub struct CohesiveBlocks {
    /// The vertex ids of each block.
    pub blocks: Vec<Vec<VertexId>>,
    /// The cohesion (vertex connectivity) of each block.
    pub cohesion: Vec<usize>,
    /// The parent block of each block in the hierarchy (`None` for the root).
    pub parent: Vec<Option<usize>>,
    /// The hierarchy as a (directed) tree graph: vertex `i` is block `i`, and
    /// there is an edge from each block to each of its children.
    pub block_tree: Graph,
}

impl CohesiveBlocks {
    /// Number of blocks.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Whether there are no blocks. Results of [`Graph::cohesive_blocks`]
    /// always contain at least the root block (even for the null graph,
    /// whose root block is empty), so this is `false` for them.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Ids of the child blocks of block `b`.
    pub fn children(&self, b: usize) -> Vec<usize> {
        (0..self.parent.len())
            .filter(|&i| self.parent[i] == Some(b))
            .collect()
    }

    /// The largest cohesion of a block containing vertex `v` (its
    /// "embeddedness"), or `None` if no block contains it.
    pub fn max_cohesion_of(&self, v: VertexId) -> Option<usize> {
        self.blocks
            .iter()
            .zip(&self.cohesion)
            .filter(|(b, _)| b.contains(&v))
            .map(|(_, &c)| c)
            .max()
    }
}

/// Reachability information, see [`Graph::reachability`].
///
/// Vertices in the same strongly connected component reach exactly the same
/// vertices, so the reachable sets are stored once per component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reachability {
    /// `membership[v]` is the id of the (strongly, for directed graphs with
    /// `Out`/`In` modes) connected component of `v`.
    pub membership: Vec<VertexId>,
    /// `sizes[c]` is the number of vertices of component `c`.
    pub sizes: Vec<usize>,
    /// The number of components.
    pub count: usize,
    /// `reach[c][v]` is `true` when vertex `v` is reachable from the vertices
    /// of component `c` (every vertex reaches itself).
    pub reach: Vec<Vec<bool>>,
}

impl Reachability {
    /// Whether `to` is reachable from `from` (`false` for out-of-range ids).
    pub fn is_reachable(&self, from: VertexId, to: VertexId) -> bool {
        usize::try_from(from)
            .ok()
            .and_then(|f| self.membership.get(f))
            .and_then(|&c| self.reach.get(c as usize))
            .and_then(|row| usize::try_from(to).ok().and_then(|t| row.get(t).copied()))
            .unwrap_or(false)
    }

    /// The vertices reachable from `from`, in increasing id order (empty for
    /// an out-of-range id).
    pub fn reachable_from(&self, from: VertexId) -> Vec<VertexId> {
        let Some(&c) = usize::try_from(from)
            .ok()
            .and_then(|f| self.membership.get(f))
        else {
            return Vec::new();
        };
        self.reach[c as usize]
            .iter()
            .enumerate()
            .filter(|&(_, &r)| r)
            .map(|(v, _)| v as VertexId)
            .collect()
    }
}

/// Runs `f` with a raw `igraph_vs_t` built from `sel`, keeping any backing
/// storage alive (and at a fixed address) for the whole call.
///
/// Vertex lists are passed as a zero-copy view of the Rust slice, whose
/// `igraph_vector_int_t` header lives in this stack frame and never moves
/// while `f` runs (`igraph_vss_vector` stores a pointer to that header).
/// The other selectors go through [`VertexSelector::to_raw`].
fn with_vs(sel: VertexSelector<'_>, f: impl FnOnce(igraph_vs_t) -> Result<()>) -> Result<()> {
    if let VertexSelector::List(list) = &sel {
        let view = VectorInt::view(list);
        // SAFETY: `view` outlives the call of `f` and is not moved meanwhile.
        let vs = unsafe { igraph_vss_vector(view.as_ptr()) };
        return f(vs);
    }
    let raw = sel.to_raw()?;
    f(raw.get())
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Bond percolation curve of a bare list of vertex pairs.
///
/// Edges `(u, v)` are added one after the other, and after each addition the
/// size of the largest connected component and the number of non-isolated
/// vertices are recorded. It differs from [`Graph::bond_percolation`] in that
/// no graph is needed: vertices are identified by the ids appearing in
/// `edges`, which must be non-negative (the largest id determines how much
/// memory is allocated). Self-loops are allowed.
///
/// This function is marked *experimental* in igraph.
///
/// Time complexity: O(|E| α(|E|)), α being the inverse Ackermann function.
///
/// Binds [`igraph_edgelist_percolation`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_edgelist_percolation).
///
/// # Errors
/// [`ErrorKind::InvalidVertexId`] for negative vertex ids and for the id
/// `i64::MAX` (rejected on the Rust side: igraph 1.0.1 computes the vertex
/// count as the largest id plus one, which would overflow and abort the
/// process). [`ErrorKind::OutOfMemory`] when the largest id is too large
/// to allocate per-vertex storage for.
///
/// # Examples
/// ```
/// use igraph::components::edgelist_percolation;
///
/// // A triangle with an extra self-loop on vertex 0.
/// let p = edgelist_percolation(&[(0, 0), (0, 1), (1, 2), (2, 0)]).unwrap();
/// assert_eq!(p.giant_size, vec![1, 2, 3, 3]);
/// assert_eq!(p.vertex_count, vec![1, 2, 3, 3]);
/// ```
pub fn edgelist_percolation(edges: &[(VertexId, VertexId)]) -> Result<BondPercolation> {
    if edges
        .iter()
        .any(|&(a, b)| a == igraph_int_t::MAX || b == igraph_int_t::MAX)
    {
        return Err(Error::new(
            ErrorKind::InvalidVertexId,
            format!("vertex id {} is too large", igraph_int_t::MAX),
        ));
    }
    let flat: VectorInt = edges.iter().flat_map(|&(a, b)| [a, b]).collect();
    let mut giant = VectorInt::new();
    let mut count = VectorInt::new();
    igraph_call!(igraph_edgelist_percolation(&flat, &mut giant, &mut count))?;
    Ok(BondPercolation {
        giant_size: to_usizes(giant),
        vertex_count: to_usizes(count),
    })
}

// ---------------------------------------------------------------------------
// Graph methods
// ---------------------------------------------------------------------------

impl igraph_t {
    // ----- components -------------------------------------------------------

    /// The (weakly or strongly) connected components of the graph.
    ///
    /// With [`Connectedness::Weak`] edge directions are ignored; with
    /// [`Connectedness::Strong`] two vertices are in the same component when
    /// each is reachable from the other by a directed path. The mode is
    /// ignored for undirected graphs.
    ///
    /// Strongly connected components are numbered in *topological order*:
    /// vertex `v` is reachable from `u` only if
    /// `membership[u] <= membership[v]`. Weak components are numbered in
    /// order of their smallest vertex id.
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// See also [`is_connected`](Self::is_connected) for a (cached) yes/no
    /// answer, [`decompose`](Self::decompose) to get the components as
    /// graphs, and [`Graph::subcomponent`] for the component of a single
    /// vertex.
    ///
    /// Binds [`igraph_connected_components`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_connected_components).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    ///
    /// // 0 -> 1 -> 2 -> 0 is a directed cycle, 3 only has an in-edge.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, true).unwrap();
    /// let weak = g.connected_components(Connectedness::Weak).unwrap();
    /// assert_eq!(weak.count, 1);
    /// let strong = g.connected_components(Connectedness::Strong).unwrap();
    /// assert_eq!(strong.count, 2);
    /// assert!(strong.same_component(0, 2));
    /// assert!(!strong.same_component(0, 3));
    /// ```
    pub fn connected_components(&self, mode: Connectedness) -> Result<ConnectedComponents> {
        let mut membership = VectorInt::new();
        let mut csize = VectorInt::new();
        let mut no: igraph_int_t = 0;
        igraph_call!(igraph_connected_components(
            self,
            &mut membership,
            &mut csize,
            &mut no,
            mode.into()
        ))?;
        Ok(ConnectedComponents {
            membership: membership.into(),
            sizes: to_usizes(csize),
            count: no as usize,
        })
    }

    /// Whether the graph is (weakly or strongly) connected.
    ///
    /// A graph is connected when every vertex is reachable from every other;
    /// for directed graphs, [`Connectedness::Strong`] follows edge directions
    /// while [`Connectedness::Weak`] ignores them. The mode is ignored for
    /// undirected graphs. By definition the null graph (no vertices) is
    /// **not** connected, while the singleton graph is.
    ///
    /// The result is cached inside the graph, so repeated calls without
    /// modifications are O(1); otherwise the time complexity is O(|V| + |E|).
    ///
    /// See also [`is_biconnected`](Self::is_biconnected) for
    /// 2-vertex-connectedness, and [`Graph::vertex_connectivity`] /
    /// [`Graph::edge_connectivity`] for *how* connected a graph is.
    ///
    /// Binds [`igraph_is_connected`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_is_connected).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    /// assert!(g.is_connected(Connectedness::Weak).unwrap());
    /// assert!(!g.is_connected(Connectedness::Strong).unwrap());
    /// assert!(!Graph::new(0, false).is_connected(Connectedness::Weak).unwrap());
    /// ```
    pub fn is_connected(&self, mode: Connectedness) -> Result<bool> {
        let mut res = false;
        igraph_call!(igraph_is_connected(self, &mut res, mode.into()))?;
        Ok(res)
    }

    /// Splits the graph into one separate [`Graph`] per connected component.
    ///
    /// `max_components` limits the number of returned graphs (`None` for no
    /// limit): the first ones found are kept (for weak components, in order
    /// of their smallest vertex id). Components
    /// with fewer than `min_vertices` vertices are skipped (e.g. `2` drops the
    /// isolated vertices). Vertex ids are renumbered in each component graph,
    /// preserving their relative order; directedness is preserved.
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// Each graph is the subgraph induced by one component: use
    /// [`connected_components`](Self::connected_components) with
    /// [`Graph::induced_subgraph`] to extract only some of them, or to keep
    /// the mapping to the original vertex ids.
    ///
    /// Binds [`igraph_decompose`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_decompose).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A triangle, a single edge and an isolated vertex.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4)], 6, false).unwrap();
    /// let parts = g.decompose(Connectedness::Weak, None, 2).unwrap();
    /// let sizes: Vec<_> = parts.iter().map(|p| (p.vcount(), p.ecount())).collect();
    /// assert_eq!(sizes, vec![(3, 3), (2, 1)]);
    /// ```
    pub fn decompose(
        &self,
        mode: Connectedness,
        max_components: Option<usize>,
        min_vertices: usize,
    ) -> Result<Vec<Graph>> {
        let mut list = GraphList::new();
        igraph_call!(igraph_decompose(
            self,
            &mut list,
            mode.into(),
            order_raw(max_components),
            int_sat(min_vertices)
        ))?;
        Ok(list.into_vec())
    }

    /// The articulation points (cut vertices) of the graph.
    ///
    /// A vertex is an articulation point if its removal increases the number
    /// of (weakly) connected components. Edge directions are ignored. The
    /// vertices are returned in no particular order.
    ///
    /// A graph without articulation points is not necessarily biconnected:
    /// the null graph, the singleton graph and edgeless graphs have none
    /// either; use [`is_biconnected`](Self::is_biconnected) for that.
    /// Conversely `K2`, which igraph considers biconnected, has no
    /// articulation points. (The C documentation of 1.0.0 and 1.0.1 lists
    /// `K2` among the counterexamples, but [`is_biconnected`](Self::is_biconnected)
    /// returns `true` for it.)
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// See also [`biconnected_components`](Self::biconnected_components),
    /// which also returns the articulation points, and
    /// [`bridges`](Self::bridges) for the edge analogue.
    ///
    /// Binds [`igraph_articulation_points`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_articulation_points).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A path 0-1-2-3: the inner vertices are cut vertices.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let mut ap = g.articulation_points().unwrap();
    /// ap.sort();
    /// assert_eq!(ap, vec![1, 2]);
    /// ```
    pub fn articulation_points(&self) -> Result<Vec<VertexId>> {
        let mut res = VectorInt::new();
        igraph_call!(igraph_articulation_points(self, &mut res))?;
        Ok(res.into())
    }

    /// The bridges (cut edges) of the graph, as edge ids.
    ///
    /// An edge is a bridge if its removal increases the number of (weakly)
    /// connected components. Edge directions are ignored. The edges are
    /// returned in no particular order. A multi-edge is never a bridge (its
    /// parallel copies keep the endpoints connected), and neither is a
    /// self-loop.
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// A connected graph has a bridge exactly when its
    /// [`edge_connectivity`](Graph::edge_connectivity) is 1 (and it has more
    /// than one vertex).
    ///
    /// Binds [`igraph_bridges`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_bridges).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A square 0-1-2-3 with a pendant edge 3-4 (edge id 4).
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0), (3, 4)], 5, false).unwrap();
    /// assert_eq!(g.bridges().unwrap(), vec![4]);
    /// ```
    pub fn bridges(&self) -> Result<Vec<EdgeId>> {
        let mut res = VectorInt::new();
        igraph_call!(igraph_bridges(self, &mut res))?;
        Ok(res.into())
    }

    /// The biconnected components of the graph (edge directions are ignored).
    ///
    /// A graph is biconnected if removing any single vertex leaves it
    /// connected; a biconnected component is a maximal biconnected subgraph.
    /// The components partition the (non-loop) *edges* of the graph, while a
    /// vertex can belong to several of them (the articulation points) or to
    /// none (isolated vertices). igraph considers the two-vertex complete
    /// graph `K2` biconnected, but not single vertices, so a single
    /// biconnected component does not imply that the graph is biconnected
    /// (there may be isolated vertices): use
    /// [`is_biconnected`](Self::is_biconnected) for that. Self-loops belong to
    /// no component.
    ///
    /// All outputs are computed: the number of components, a spanning tree of
    /// each, their edges and vertices, and the articulation points. Time
    /// complexity: O(|V| + |E|) for the trees alone, but igraph documents
    /// computing the vertex sets as quadratic and the edge sets as cubic in
    /// |V| in the worst case.
    ///
    /// Binds [`igraph_biconnected_components`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_biconnected_components).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // Two triangles sharing vertex 2 ("bowtie").
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)], 5, false)
    ///     .unwrap();
    /// let bc = g.biconnected_components().unwrap();
    /// assert_eq!(bc.count, 2);
    /// assert_eq!(bc.articulation_points, vec![2]);
    /// assert!(bc.components.iter().all(|c| c.len() == 3 && c.contains(&2)));
    /// ```
    pub fn biconnected_components(&self) -> Result<BiconnectedComponents> {
        let mut no: igraph_int_t = 0;
        let mut tree = VectorIntList::new();
        let mut edges = VectorIntList::new();
        let mut comps = VectorIntList::new();
        let mut ap = VectorInt::new();
        igraph_call!(igraph_biconnected_components(
            self, &mut no, &mut tree, &mut edges, &mut comps, &mut ap
        ))?;
        Ok(BiconnectedComponents {
            count: no as usize,
            tree_edges: tree.to_vecs(),
            component_edges: edges.to_vecs(),
            components: comps.to_vecs(),
            articulation_points: ap.into(),
        })
    }

    /// Whether the graph is biconnected (edge directions are ignored).
    ///
    /// A graph is biconnected if removing any single vertex (and its
    /// incident edges) does not disconnect it. igraph does not consider the
    /// null and singleton graphs biconnected, but does consider `K2`
    /// biconnected. For graphs with at least three vertices this is the same
    /// as having [`vertex_connectivity`](Graph::vertex_connectivity) at
    /// least 2, but much cheaper to compute.
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// Binds [`igraph_is_biconnected`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_is_biconnected).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let square = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], 4, false).unwrap();
    /// assert!(square.is_biconnected().unwrap());
    /// let path = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// assert!(!path.is_biconnected().unwrap());
    /// ```
    pub fn is_biconnected(&self) -> Result<bool> {
        let mut res = false;
        igraph_call!(igraph_is_biconnected(self, &mut res))?;
        Ok(res)
    }

    /// Bond (edge) percolation curve: the size of the giant component as the
    /// edges of the graph are added one by one to its vertices.
    ///
    /// `edge_order` gives the order in which the edge ids are added and must
    /// not contain duplicates; it may list only a subset of the edges. With
    /// `None` a uniformly random order is used, drawn from the calling
    /// thread's default random number generator (reproducible after
    /// [`rng::seed`](crate::rng::seed)). Reversing both the order and the
    /// resulting `giant_size` gives the curve for edge *removal*. Edge
    /// directions are ignored; the vertices are only counted once they get
    /// an incident edge, so isolated vertices never contribute.
    ///
    /// This function is marked *experimental* in igraph.
    ///
    /// Time complexity: O(|V| + |E| α(|E|)), α being the inverse Ackermann
    /// function.
    ///
    /// See also [`site_percolation`](Self::site_percolation) for the
    /// vertex version, [`edgelist_percolation`] to percolate arbitrary vertex
    /// pairs, and [`connected_components`](Self::connected_components) for
    /// the component sizes of a fixed graph.
    ///
    /// Binds [`igraph_bond_percolation`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_bond_percolation).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidEdgeId`] for an edge id out of range and
    /// [`ErrorKind::InvalidValue`] for duplicates in `edge_order`. Both are
    /// checked on the Rust side: igraph 1.0.0 and 1.0.1 size their
    /// duplicate-detection bitset by the *length* of `edge_order` rather than
    /// by the number of edges, so an unchecked partial order or out-of-range
    /// id would make them access memory out of bounds.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // The 4-cycle 0-1-2-3-0, adding edges 0-1, 2-3, 1-2, 3-0.
    /// let c4 = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], 4, false).unwrap();
    /// let p = c4.bond_percolation(Some(&[0, 2, 1, 3])).unwrap();
    /// assert_eq!(p.giant_size, vec![2, 2, 4, 4]);
    /// assert_eq!(p.vertex_count, vec![2, 4, 4, 4]);
    ///
    /// // A random order is reproducible after seeding; the final point is
    /// // always the whole (connected) graph.
    /// let grid = Graph::square_lattice(&[5, 5], 1, false, false, None).unwrap();
    /// rng::seed(7).unwrap();
    /// let a = grid.bond_percolation(None).unwrap();
    /// rng::seed(7).unwrap();
    /// let b = grid.bond_percolation(None).unwrap();
    /// assert_eq!(a, b);
    /// assert_eq!(a.giant_size.last(), Some(&25));
    /// ```
    pub fn bond_percolation(&self, edge_order: Option<&[EdgeId]>) -> Result<BondPercolation> {
        if let Some(order) = edge_order {
            let m = self.ecount();
            let mut seen = vec![false; m];
            for &e in order {
                let i = usize::try_from(e).ok().filter(|&i| i < m).ok_or_else(|| {
                    Error::new(
                        ErrorKind::InvalidEdgeId,
                        format!("invalid edge id {e} in edge order (graph has {m} edges)"),
                    )
                })?;
                if std::mem::replace(&mut seen[i], true) {
                    return Err(Error::invalid(format!(
                        "duplicate edge {e} in edge order vector"
                    )));
                }
            }
        }
        let order = edge_order.map(VectorInt::view);
        let op = order.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let mut giant = VectorInt::new();
        let mut count = VectorInt::new();
        igraph_call!(igraph_bond_percolation(self, &mut giant, &mut count, op))?;
        Ok(BondPercolation {
            giant_size: to_usizes(giant),
            vertex_count: to_usizes(count),
        })
    }

    /// Site (vertex) percolation curve: the size of the giant component as
    /// the vertices of the graph are added one by one, together with the
    /// edges among the vertices added so far.
    ///
    /// `vertex_order` gives the order in which vertices are added and must
    /// not contain duplicates; it may list only a subset of the vertices.
    /// With `None` a uniformly random order is used, drawn from the calling
    /// thread's default random number generator exactly as igraph would
    /// (reproducible after [`rng::seed`](crate::rng::seed)). Reversing both
    /// the order and the resulting `giant_size` gives the curve for vertex
    /// *removal* (an attack/failure scenario). Edge directions are ignored;
    /// multi-edges count with their multiplicity in `edge_count`.
    ///
    /// This function is marked *experimental* in igraph.
    ///
    /// Time complexity: O(|V| + |E| α(|E|)).
    ///
    /// See also [`bond_percolation`](Self::bond_percolation) for the edge
    /// version and [`Graph::coreness`] or
    /// [`Graph::vertex_connectivity`] for other robustness measures.
    ///
    /// Binds [`igraph_site_percolation`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_site_percolation).
    ///
    /// # Self-loops
    /// igraph 1.0.0 and 1.0.1 count every self-loop *twice* in `edge_count`
    /// (it lists loops twice among a vertex's neighbors). This wrapper
    /// corrects the count, so a self-loop counts as one edge. To be able to
    /// do so with `vertex_order = None`, the random order is generated on
    /// the Rust side with the same random draws igraph makes internally.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`] for invalid vertex ids and
    /// [`ErrorKind::InvalidValue`] for duplicates in `vertex_order`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let c4 = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], 4, false).unwrap();
    /// let p = c4.site_percolation(Some(&[0, 2, 1, 3])).unwrap();
    /// assert_eq!(p.giant_size, vec![1, 1, 3, 4]);
    /// assert_eq!(p.edge_count, vec![0, 0, 2, 4]);
    ///
    /// // Removing the hub of a star shatters it at once.
    /// let star = Graph::star(6, StarMode::Undirected, 0).unwrap();
    /// let attack = [0, 1, 2, 3, 4, 5]; // hub first
    /// let build: Vec<i64> = attack.iter().rev().copied().collect();
    /// let mut after = star.site_percolation(Some(&build)).unwrap().giant_size;
    /// after.reverse(); // after[k]: largest component with k vertices removed
    /// assert_eq!(after, vec![6, 1, 1, 1, 1, 1]);
    /// ```
    pub fn site_percolation(&self, vertex_order: Option<&[VertexId]>) -> Result<SitePercolation> {
        let shuffled: VectorInt;
        let order: &[VertexId] = match vertex_order {
            Some(order) => order,
            None => {
                // Same as igraph: `igraph_vector_int_init_range` + `_shuffle`.
                let mut all: VectorInt = (0..self.vcount() as VertexId).collect();
                all.shuffle();
                shuffled = all;
                &shuffled
            }
        };
        let view = VectorInt::view(order);
        let mut giant = VectorInt::new();
        let mut count = VectorInt::new();
        igraph_call!(igraph_site_percolation(
            self,
            &mut giant,
            &mut count,
            view.as_ptr()
        ))?;
        let mut edge_count = to_usizes(count);
        if self.has_loop()? {
            // igraph succeeded, so every id in `order` is valid.
            let mut loops = vec![0usize; self.vcount()];
            for (from, to) in self.edge_list() {
                if from == to {
                    loops[from as usize] += 1;
                }
            }
            let mut seen = 0;
            for (count, &v) in edge_count.iter_mut().zip(order) {
                seen += loops[v as usize];
                *count -= seen;
            }
        }
        Ok(SitePercolation {
            giant_size: to_usizes(giant),
            edge_count,
        })
    }

    // ----- separators -------------------------------------------------------

    /// Whether removing the `candidate` vertices disconnects the graph.
    ///
    /// A vertex set `S` is a separator if there are vertices `u` and `v`
    /// outside `S`, connected in the graph, such that every path between
    /// them passes through `S`.
    /// Edge directions are ignored and duplicate ids in `candidate` are
    /// ignored. Removing *all* vertices, or all but one, is never a
    /// separation, and the empty set is never a separator either, not even
    /// in a disconnected graph: what matters is whether *removing* the set
    /// separates some vertices that were connected before.
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// See also [`minimum_size_separators`](Self::minimum_size_separators)
    /// to *find* the smallest separators, and
    /// [`Graph::st_vertex_connectivity`] for the size of the smallest set
    /// separating two given vertices.
    ///
    /// Binds [`igraph_is_separator`](https://igraph.org/c/html/latest/igraph-Separators.html#igraph_is_separator).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A star with center 0.
    /// let star = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], 4, false).unwrap();
    /// assert!(star.is_separator(0).unwrap());
    /// assert!(!star.is_separator(1).unwrap());
    /// assert!(!star.is_separator(1..4).unwrap());
    /// ```
    pub fn is_separator<'a>(&self, candidate: impl Into<VertexSelector<'a>>) -> Result<bool> {
        let mut res = false;
        with_vs(candidate.into(), |vs| {
            igraph_call!(igraph_is_separator(self, vs, &mut res))
        })?;
        Ok(res)
    }

    /// Whether `candidate` is a *minimal* separator: a separator (see
    /// [`is_separator`](Self::is_separator)) none of whose proper subsets is
    /// a separator. Edge directions are ignored.
    ///
    /// Time complexity: O(|V| + |E|).
    ///
    /// Binds [`igraph_is_minimal_separator`](https://igraph.org/c/html/latest/igraph-Separators.html#igraph_is_minimal_separator).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // The path 0-1-2-3.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// assert!(g.is_minimal_separator(1).unwrap());
    /// assert!(g.is_separator(&[1, 2]).unwrap());
    /// assert!(!g.is_minimal_separator(&[1, 2]).unwrap());
    /// ```
    pub fn is_minimal_separator<'a>(
        &self,
        candidate: impl Into<VertexSelector<'a>>,
    ) -> Result<bool> {
        let mut res = false;
        with_vs(candidate.into(), |vs| {
            igraph_call!(igraph_is_minimal_separator(self, vs, &mut res))
        })?;
        Ok(res)
    }

    /// All the vertex sets that are minimal (s, t) separators for some pair
    /// of vertices s, t.
    ///
    /// Some returned sets are not minimal with respect to disconnecting the
    /// graph: in the graph `0-1-2-3-4-1` the sets `{1}`, `{2, 4}` and
    /// `{1, 3}` are returned, and `{1, 3}` is minimal for separating 2 from 4
    /// although `{1}` alone disconnects the graph. Edge directions are
    /// ignored. Unlike [`minimum_size_separators`](Self::minimum_size_separators),
    /// a disconnected graph is handled component by component (its
    /// separators are those of its components). Uses the algorithm of Berry,
    /// Bordat and Cogis (1999).
    ///
    /// Time complexity: O(n |V|³), n being the number of separators.
    ///
    /// Binds [`igraph_all_minimal_st_separators`](https://igraph.org/c/html/latest/igraph-Separators.html#igraph_all_minimal_st_separators).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 1)], 5, false).unwrap();
    /// let mut seps: Vec<Vec<i64>> = g
    ///     .all_minimal_st_separators()
    ///     .unwrap()
    ///     .into_iter()
    ///     .map(|mut s| { s.sort(); s })
    ///     .collect();
    /// seps.sort();
    /// assert_eq!(seps, vec![vec![1], vec![1, 3], vec![2, 4]]);
    /// ```
    pub fn all_minimal_st_separators(&self) -> Result<Vec<Vec<VertexId>>> {
        let mut res = VectorIntList::new();
        igraph_call!(igraph_all_minimal_st_separators(self, &mut res))?;
        Ok(res.to_vecs())
    }

    /// All the vertex separators of minimum size.
    ///
    /// A vertex set is a separator if its removal disconnects the graph. The
    /// graph must be undirected. A graph that is already disconnected has no
    /// separators (an empty list is returned), and neither do complete
    /// graphs. Each separator has exactly
    /// [`vertex_connectivity`](Graph::vertex_connectivity) vertices. The
    /// separators are returned in arbitrary order. Uses the algorithm of
    /// Kanevsky (1993).
    ///
    /// See also [`Graph::all_st_mincuts`] for the edge analogue between two
    /// given vertices.
    ///
    /// Binds [`igraph_minimum_size_separators`](https://igraph.org/c/html/latest/igraph-Separators.html#igraph_minimum_size_separators).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] for
    /// directed graphs.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // The 5-cycle: removing any two non-adjacent vertices disconnects it.
    /// let c5 = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 5, false).unwrap();
    /// let seps = c5.minimum_size_separators().unwrap();
    /// assert_eq!(seps.len(), 5);
    /// assert!(seps.iter().all(|s| s.len() == 2));
    /// ```
    pub fn minimum_size_separators(&self) -> Result<Vec<Vec<VertexId>>> {
        let mut res = VectorIntList::new();
        igraph_call!(igraph_minimum_size_separators(self, &mut res))?;
        Ok(res.to_vecs())
    }

    // ----- cohesive blocks --------------------------------------------------

    /// The hierarchical cohesive block structure of the graph (Moody and
    /// White, 2003).
    ///
    /// A vertex set is *k-cohesive* when the subgraph it induces has vertex
    /// connectivity at least k; it is *maximally* k-cohesive when no superset
    /// is. Cohesive blocking starts from the whole graph and recursively
    /// identifies the maximally l-cohesive subsets, l > k, of each k-cohesive
    /// block, yielding a tree of nested blocks rooted at the whole graph
    /// (block `0`, with `parent` `None`).
    ///
    /// The cohesion of each block is the
    /// [`vertex_connectivity`](Graph::vertex_connectivity) of the subgraph it
    /// induces. The root block has cohesion 0 when the graph is
    /// disconnected; the null graph yields a single, empty root block.
    ///
    /// The graph must be undirected and simple (see [`Graph::simplify`]).
    ///
    /// See also [`Graph::coreness`] for the (cheaper, degree-based) k-core
    /// hierarchy, and [`Graph::cohesion`] for the cohesion of the whole
    /// graph.
    ///
    /// Binds [`igraph_cohesive_blocks`](https://igraph.org/c/html/latest/igraph-Flows.html#igraph_cohesive_blocks).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] for
    /// directed or non-simple graphs.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // Two 4-cliques (0..4 and 3..7) sharing vertex 3.
    /// let mut edges = vec![];
    /// for block in [[0, 1, 2, 3], [3, 4, 5, 6]] {
    ///     for i in 0..4 {
    ///         for j in i + 1..4 {
    ///             edges.push((block[i], block[j]));
    ///         }
    ///     }
    /// }
    /// let g = Graph::from_edges(&edges, 7, false).unwrap();
    /// let cb = g.cohesive_blocks().unwrap();
    /// assert_eq!(cb.len(), 3);
    /// assert_eq!(cb.cohesion, vec![1, 3, 3]);
    /// assert_eq!(cb.parent, vec![None, Some(0), Some(0)]);
    /// assert_eq!(cb.children(0), vec![1, 2]);
    /// ```
    pub fn cohesive_blocks(&self) -> Result<CohesiveBlocks> {
        let mut blocks = VectorIntList::new();
        let mut cohesion = VectorInt::new();
        let mut parent = VectorInt::new();
        let block_tree = Graph::init_with(|tree| unsafe {
            igraph_cohesive_blocks(self, &mut blocks, &mut cohesion, &mut parent, tree)
        })?;
        Ok(CohesiveBlocks {
            blocks: blocks.to_vecs(),
            cohesion: to_usizes(cohesion),
            parent: parent.iter().map(|&p| usize::try_from(p).ok()).collect(),
            block_tree,
        })
    }

    // ----- reachability -----------------------------------------------------

    /// Which vertices are reachable from each vertex.
    ///
    /// The result groups vertices by the component they belong to (strongly
    /// connected components for directed graphs with
    /// [`NeighborMode::Out`]/[`NeighborMode::In`], plain components
    /// otherwise), since vertices of one component reach the same set, and
    /// stores for each component the set of reachable vertices as booleans.
    /// With `Out` edges are followed along their direction, with `In`
    /// against it, with `All` directions are ignored; `mode` is ignored for
    /// undirected graphs. A vertex always reaches itself.
    ///
    /// Time complexity: O(|C||V|/w + |V| + |E|), |C| being the number of
    /// components and w the machine word size.
    ///
    /// See also [`count_reachable`](Self::count_reachable) for the sizes
    /// only, [`transitive_closure`](Self::transitive_closure) for the same
    /// information as a graph, and [`Graph::subcomponent`] for the vertices
    /// reachable from a single vertex.
    ///
    /// Binds [`igraph_reachability`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_reachability).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (3, 2)], 4, true).unwrap();
    /// let r = g.reachability(NeighborMode::Out).unwrap();
    /// assert!(r.is_reachable(0, 2));
    /// assert!(!r.is_reachable(2, 0));
    /// assert_eq!(r.reachable_from(3), vec![2, 3]);
    /// ```
    pub fn reachability(&self, mode: NeighborMode) -> Result<Reachability> {
        let mut membership = VectorInt::new();
        let mut csize = VectorInt::new();
        let mut no: igraph_int_t = 0;
        let mut reach = BitsetList::new();
        igraph_call!(igraph_reachability(
            self,
            &mut membership,
            &mut csize,
            &mut no,
            &mut reach,
            mode.into()
        ))?;
        Ok(Reachability {
            membership: membership.into(),
            sizes: to_usizes(csize),
            count: no as usize,
            reach: reach.iter().map(|bits| bits.to_vec()).collect(),
        })
    }

    /// The number of vertices reachable from each vertex, itself included.
    ///
    /// `mode` has the same meaning as in [`reachability`](Self::reachability):
    /// with [`NeighborMode::In`] it counts the vertices that can reach each
    /// vertex.
    ///
    /// Time complexity: O(|C||V|/w + |V| + |E|).
    ///
    /// Equivalently, it is [`neighborhood_size`](Self::neighborhood_size)
    /// with unlimited order and `mindist = 0`, but faster: the work is shared
    /// by all the vertices of a strongly connected component instead of
    /// running one breadth-first search per vertex.
    ///
    /// Binds [`igraph_count_reachable`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_count_reachable).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // The directed path 0 -> 1 -> 2.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    /// assert_eq!(g.count_reachable(NeighborMode::Out).unwrap(), vec![3, 2, 1]);
    /// assert_eq!(g.count_reachable(NeighborMode::In).unwrap(), vec![1, 2, 3]);
    /// ```
    pub fn count_reachable(&self, mode: NeighborMode) -> Result<Vec<usize>> {
        let mut res = VectorInt::new();
        igraph_call!(igraph_count_reachable(self, &mut res, mode.into()))?;
        Ok(to_usizes(res))
    }

    /// The transitive closure of the graph.
    ///
    /// The result has the same vertices and directedness, and an edge
    /// `i -> j` (`i != j`) exactly when `j` is reachable from `i` in the
    /// original graph; it is simple (no loops nor multi-edges). For undirected
    /// graphs every component becomes a clique.
    ///
    /// Time complexity: O(|V|² + |E|).
    ///
    /// For a bounded number of steps use [`Graph::graph_power`] (a new graph)
    /// or [`Graph::connect_neighborhood`] (in place).
    ///
    /// Binds [`igraph_transitive_closure`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_transitive_closure).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    /// let tc = g.transitive_closure().unwrap();
    /// assert_eq!(tc.ecount(), 3);
    /// assert!(tc.get_eid(0, 2, true).unwrap().is_some());
    /// ```
    pub fn transitive_closure(&self) -> Result<Graph> {
        Graph::init_with(|closure| unsafe { igraph_transitive_closure(self, closure) })
    }

    // ----- neighborhoods ----------------------------------------------------

    /// The sizes of the neighborhoods of the selected vertices.
    ///
    /// The neighborhood of order `k` of a vertex contains the vertices at
    /// distance at most `k` from it: order 0 is the vertex itself, order 1
    /// adds its neighbors, and so on; `order = None` means no limit (the
    /// whole reachable set). Vertices closer than `mindist` are not counted:
    /// `mindist = 1` excludes the vertex itself, `2` its neighbors too, etc.
    /// With [`NeighborMode::Out`] paths follow edge directions, with
    /// [`NeighborMode::In`] they go against them, [`NeighborMode::All`]
    /// ignores them.
    ///
    /// Time complexity: O(n d o), n being the number of selected vertices,
    /// d the average degree and o the order.
    ///
    /// See also [`Graph::distances`] for the distances themselves and
    /// [`Graph::bfs`] for a full breadth-first traversal.
    ///
    /// Binds [`igraph_neighborhood_size`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_neighborhood_size).
    ///
    /// # Errors
    /// Invalid vertex ids, or `mindist > order`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // The path 0-1-2-3-4.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4)], 5, false).unwrap();
    /// assert_eq!(g.neighborhood_size(.., Some(1), NeighborMode::All, 0).unwrap(), vec![2, 3, 3, 3, 2]);
    /// // Vertices at distance exactly 2.
    /// assert_eq!(g.neighborhood_size(.., Some(2), NeighborMode::All, 2).unwrap(), vec![1, 1, 2, 1, 1]);
    /// ```
    pub fn neighborhood_size<'a>(
        &self,
        vids: impl Into<VertexSelector<'a>>,
        order: Option<usize>,
        mode: NeighborMode,
        mindist: usize,
    ) -> Result<Vec<usize>> {
        let mut res = VectorInt::new();
        with_vs(vids.into(), |vs| {
            igraph_call!(igraph_neighborhood_size(
                self,
                &mut res,
                vs,
                order_raw(order),
                mode.into(),
                int_sat(mindist)
            ))
        })?;
        Ok(to_usizes(res))
    }

    /// The neighborhoods of the selected vertices, as vertex lists.
    ///
    /// See [`neighborhood_size`](Self::neighborhood_size) for the meaning of
    /// `order` (`None` = unlimited), `mode` and `mindist`. Each list is in
    /// breadth-first order: the vertex itself first (unless excluded by
    /// `mindist`), then vertices at distance 1, 2, ...
    ///
    /// Time complexity: O(n d o).
    ///
    /// See also [`Graph::connect_neighborhood`], which adds an edge from each
    /// vertex to every member of its neighborhood.
    ///
    /// Binds [`igraph_neighborhood`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_neighborhood).
    ///
    /// # Errors
    /// Invalid vertex ids, or `mindist > order`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // The directed path 0 -> 1 -> 2 -> 3.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, true).unwrap();
    /// assert_eq!(g.neighborhood(1, None, NeighborMode::Out, 0).unwrap(), vec![vec![1, 2, 3]]);
    /// assert_eq!(g.neighborhood(&[1, 3], Some(1), NeighborMode::In, 1).unwrap(), vec![vec![0], vec![2]]);
    /// ```
    pub fn neighborhood<'a>(
        &self,
        vids: impl Into<VertexSelector<'a>>,
        order: Option<usize>,
        mode: NeighborMode,
        mindist: usize,
    ) -> Result<Vec<Vec<VertexId>>> {
        let mut res = VectorIntList::new();
        with_vs(vids.into(), |vs| {
            igraph_call!(igraph_neighborhood(
                self,
                &mut res,
                vs,
                order_raw(order),
                mode.into(),
                int_sat(mindist)
            ))
        })?;
        Ok(res.to_vecs())
    }

    /// The subgraphs induced by the neighborhoods of the selected vertices.
    ///
    /// See [`neighborhood_size`](Self::neighborhood_size) for the meaning of
    /// `order` (`None` = unlimited), `mode` and `mindist`. In each graph the
    /// vertices are renumbered consecutively *preserving their relative
    /// order* in the original graph (as in an induced subgraph), so the
    /// new id of an original vertex `v` is the number of neighborhood
    /// members with a smaller id than `v`. Each graph is thus the same as
    /// [`Graph::induced_subgraph`] of the corresponding
    /// [`neighborhood`](Self::neighborhood) (an "ego network").
    ///
    /// Time complexity: O(n (|V| + |E|)).
    ///
    /// Binds [`igraph_neighborhood_graphs`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_neighborhood_graphs).
    ///
    /// # Errors
    /// Invalid vertex ids, or `mindist > order`.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A star with center 0 and leaves 1..=4: its 1-neighborhoods.
    /// let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (0, 4)], 5, false).unwrap();
    /// let egos = g.neighborhood_graphs(&[0, 1], Some(1), NeighborMode::All, 0).unwrap();
    /// assert_eq!((egos[0].vcount(), egos[0].ecount()), (5, 4));
    /// assert_eq!((egos[1].vcount(), egos[1].ecount()), (2, 1));
    /// ```
    pub fn neighborhood_graphs<'a>(
        &self,
        vids: impl Into<VertexSelector<'a>>,
        order: Option<usize>,
        mode: NeighborMode,
        mindist: usize,
    ) -> Result<Vec<Graph>> {
        let mut res = GraphList::new();
        with_vs(vids.into(), |vs| {
            igraph_call!(igraph_neighborhood_graphs(
                self,
                &mut res,
                vs,
                order_raw(order),
                mode.into(),
                int_sat(mindist)
            ))
        })?;
        Ok(res.into_vec())
    }
}

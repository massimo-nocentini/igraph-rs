//! Cocitation, bibliographic coupling and vertex similarity
//! (`igraph_cocitation.h`).
//!
//! This module binds the functions of igraph's `igraph_cocitation.h` header
//! (documented in the
//! [Similarity measures](https://igraph.org/c/html/latest/igraph-Structural.html#similarity-measures)
//! section of the Structural chapter of the C manual) as methods of [`Graph`]. They all measure how
//! *similar* two vertices are by looking at the neighbors they share:
//!
//! - **cocitation**: how many vertices cite (point to) *both* of them;
//! - **bibliographic coupling**: how many vertices *both* of them cite;
//! - **Jaccard** similarity: `|N(u) ∩ N(v)| / |N(u) ∪ N(v)|`;
//! - **Dice** similarity: `2 |N(u) ∩ N(v)| / (|N(u)| + |N(v)|)`, a monotone
//!   function of the Jaccard one (`D = 2J / (1 + J)`);
//! - **inverse log-weighted** similarity (Adamic–Adar): common neighbors,
//!   each weighted by `1 / ln(degree)`, so that sharing a rare neighbor
//!   counts more than sharing a hub.
//!
//! Such scores are the classic baseline for *link prediction*: two
//! non-adjacent vertices with many common neighbors are likely to become
//! connected.
//!
//! # Example
//!
//! ```
//! use igraph::prelude::*;
//!
//! // The citation network of igraph's own example: 0 -> 1, 2 -> 1, 2 -> 0, 3 -> 0.
//! let g = Graph::from_edges(&[(0, 1), (2, 1), (2, 0), (3, 0)], 4, true)?;
//!
//! // 0 and 1 are both cited by 2.
//! let cocit = g.cocitation(..)?;
//! assert_eq!(cocit[(0, 1)], 1.0);
//! // 2 and 3 both cite 0; 0 and 2 both cite 1.
//! let bib = g.bibcoupling(..)?;
//! assert_eq!((bib[(2, 3)], bib[(0, 2)]), (1.0, 1.0));
//!
//! // Ignoring directions, 1 and 2 share the neighbor 0 out of {0, 1, 2}.
//! let jac = g.similarity_jaccard(1..3, 1..3, NeighborMode::All, false)?;
//! assert!((jac[(0, 1)] - 1.0 / 3.0).abs() < 1e-12);
//! let dice = g.similarity_dice_pairs(&[(1, 2)], NeighborMode::All, false)?;
//! assert!((dice[0] - 0.5).abs() < 1e-12);
//! # Ok::<(), igraph::Error>(())
//! ```
//!
//! # Provided functionality
//!
//! | C function | Method of [`Graph`] | Result |
//! |------------|---------------------|--------|
//! | `igraph_cocitation` | [`cocitation`](Graph::cocitation) | [`Matrix`], one row per selected vertex |
//! | `igraph_bibcoupling` | [`bibcoupling`](Graph::bibcoupling) | [`Matrix`], one row per selected vertex |
//! | `igraph_similarity_inverse_log_weighted` | [`similarity_inverse_log_weighted`](Graph::similarity_inverse_log_weighted) | [`Matrix`], one row per selected vertex |
//! | `igraph_similarity_jaccard` | [`similarity_jaccard`](Graph::similarity_jaccard) | [`Matrix`] `from × to` |
//! | `igraph_similarity_jaccard_pairs` | [`similarity_jaccard_pairs`](Graph::similarity_jaccard_pairs) | `Vec<f64>`, one value per pair |
//! | `igraph_similarity_jaccard_es` | [`similarity_jaccard_es`](Graph::similarity_jaccard_es) | `Vec<f64>`, one value per edge |
//! | `igraph_similarity_dice` | [`similarity_dice`](Graph::similarity_dice) | [`Matrix`] `from × to` |
//! | `igraph_similarity_dice_pairs` | [`similarity_dice_pairs`](Graph::similarity_dice_pairs) | `Vec<f64>`, one value per pair |
//! | `igraph_similarity_dice_es` | [`similarity_dice_es`](Graph::similarity_dice_es) | `Vec<f64>`, one value per edge |
//!
//! All the functions of the header are covered.
//!
//! # Differences from the C functions
//!
//! These wrappers are safer than the C functions they bind, and give the
//! results the C documentation promises in cases where igraph 1.0.1 does not:
//!
//! - `igraph_similarity_jaccard` and `igraph_similarity_dice` fill the
//!   matrix as if `from` and `to` were the same vertex list. If they differ,
//!   the results are wrong, and when `from` is longer than `to` the C code
//!   writes past the end of the matrix. [`similarity_jaccard`](Graph::similarity_jaccard)
//!   and [`similarity_dice`](Graph::similarity_dice) call them only when the
//!   two lists are identical and have no repeated vertex. Otherwise they
//!   compute every `from × to` pair with the `*_pairs` function.
//! - When a vertex appears several times in `vids`, `igraph_cocitation`,
//!   `igraph_bibcoupling` and `igraph_similarity_inverse_log_weighted` fill
//!   only its last row and leave the other rows at zero. The wrappers fill
//!   every row.
//! - On a directed graph, the Jaccard and Dice functions clear igraph's
//!   property cache around the C call. Otherwise a cached "no multi-edges"
//!   flag would count each mutual pair `u -> v`, `v -> u` twice with
//!   [`NeighborMode::All`] (see [`crate::structural`]).
//!
//! # See also
//!
//! - Common neighbors of two vertices: [`Graph::neighbors`] (core).
//! - Other similarity-like structures: [`Graph::get_adjacency`]
//!   ([`conversion`](crate::conversion)) and the transitivity functions in
//!   [`centrality`](crate::centrality).

use crate::{
    constants::NeighborMode,
    error::Result,
    ffi::*,
    graph::{Graph, VertexId},
    igraph_call,
    matrix::Matrix,
    selector::{EdgeSelector, VertexSelector},
    structural::with_multi_cache_guard,
    vector::{Vector, VectorInt},
};
use std::{borrow::Cow, collections::HashMap};

type RowFn =
    unsafe extern "C" fn(*const igraph_t, *mut igraph_matrix_t, igraph_vs_t) -> igraph_error_t;

impl igraph_t {
    /// Runs one of the "one row per vertex of `vids`, one column per vertex
    /// of the graph" functions. If `vids` repeats a vertex, the function runs
    /// on the distinct vertices and the rows are then copied into place,
    /// because the C code fills only the last row of a repeated vertex.
    fn per_vertex_rows<'a>(
        &self,
        vids: impl Into<VertexSelector<'a>>,
        f: impl Fn(&Graph, &mut Matrix, igraph_vs_t) -> Result<()>,
    ) -> Result<Matrix> {
        let list = self.select_vertices(vids)?;
        let mut distinct: Vec<VertexId> = Vec::with_capacity(list.len());
        let mut row_of: HashMap<VertexId, i64> = HashMap::with_capacity(list.len());
        for &v in &list {
            row_of.entry(v).or_insert_with(|| {
                distinct.push(v);
                distinct.len() as i64 - 1
            });
        }
        let vs = VertexSelector::List(Cow::Borrowed(&distinct)).to_raw()?;
        let mut res = Matrix::new();
        f(self, &mut res, vs.get())?;
        if distinct.len() == list.len() {
            return Ok(res);
        }
        let rows: Vec<i64> = list.iter().map(|v| row_of[v]).collect();
        res.select_rows(&rows)
    }

    fn per_vertex_rows_c<'a>(
        &self,
        vids: impl Into<VertexSelector<'a>>,
        cfn: RowFn,
    ) -> Result<Matrix> {
        self.per_vertex_rows(vids, |g, res, vs| igraph_call!(cfn(g, res, vs)))
    }

    /// Cocitation counts: `res[(i, j)]` is the number of vertices that cite
    /// (have an edge pointing to) both the `i`-th vertex of `vids` and vertex
    /// `j`.
    ///
    /// The result has one row per vertex of `vids`, in the order given, and
    /// one column per vertex of the graph. In a simple graph the diagonal is
    /// zero: a vertex is not cocited with itself. Multi-edges count with
    /// multiplicity. For example, `k -> a` twice and `k -> b` once add 2 to
    /// `(a, b)`. They also add 2 to the diagonal entry `(a, a)`, because the
    /// two parallel edges form a pair. A self-loop of an undirected graph
    /// lists its vertex twice among its own neighbors, so it has the same
    /// effect as a double edge (a loop of a directed graph counts once).
    /// In an undirected graph every edge
    /// counts as a citation both ways, so the score is the number of common
    /// neighbors. Cocitation is symmetric, and it equals bibliographic
    /// coupling on the transposed graph. Off the diagonal, it is `AᵀA`, with
    /// `A` the adjacency matrix.
    ///
    /// Binds [`igraph_cocitation`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_cocitation).
    /// Time complexity: `O(|V| d²)`, with `d` the maximum degree.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) (or
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for a range)
    /// if `vids` names a missing vertex.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // Papers 3 and 4 both cite papers 0 and 1; paper 4 also cites 2.
    /// let g = Graph::from_edges(&[(3, 0), (3, 1), (4, 0), (4, 1), (4, 2)], 5, true)?;
    /// let m = g.cocitation(&[0, 2])?;
    /// assert_eq!(m.row(0), vec![0.0, 2.0, 1.0, 0.0, 0.0]);
    /// assert_eq!(m.row(1), vec![1.0, 1.0, 0.0, 0.0, 0.0]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn cocitation<'a>(&self, vids: impl Into<VertexSelector<'a>>) -> Result<Matrix> {
        self.per_vertex_rows_c(vids, igraph_cocitation)
    }

    /// Bibliographic coupling: `res[(i, j)]` is the number of vertices cited
    /// by both the `i`-th vertex of `vids` and vertex `j`.
    ///
    /// The result has one row per vertex of `vids`, in the order given, and
    /// one column per vertex of the graph. The diagonal is zero in a simple
    /// graph; multi-edges (and undirected self-loops) count with multiplicity
    /// as in [`cocitation`](Self::cocitation). In an undirected graph this is the
    /// same as the cocitation (the number of common neighbors). Off the
    /// diagonal, the matrix is `AAᵀ`.
    ///
    /// Binds [`igraph_bibcoupling`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_bibcoupling).
    /// Time complexity: `O(|V| d²)`, with `d` the maximum degree.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) (or
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for a range)
    /// if `vids` names a missing vertex.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // Papers 3 and 4 share two references (0 and 1).
    /// let g = Graph::from_edges(&[(3, 0), (3, 1), (4, 0), (4, 1), (4, 2)], 5, true)?;
    /// let m = g.bibcoupling(4)?;
    /// assert_eq!(m.row(0), vec![0.0, 0.0, 0.0, 2.0, 0.0]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn bibcoupling<'a>(&self, vids: impl Into<VertexSelector<'a>>) -> Result<Matrix> {
        self.per_vertex_rows_c(vids, igraph_bibcoupling)
    }

    /// Inverse log-weighted similarity (Adamic and Adar, 2003): the common
    /// neighbors of two vertices, each weighted by `1 / ln(degree)`.
    ///
    /// The idea is that sharing a low-degree neighbor says more about two
    /// vertices than sharing a hub, which many vertices share by chance.
    /// `mode` chooses which neighbors are compared in a directed graph:
    ///
    /// - [`NeighborMode::Out`]: out-neighbors (the vertices both cite); each
    ///   common neighbor is weighted by its **in**-degree;
    /// - [`NeighborMode::In`]: in-neighbors; weighted by the **out**-degree;
    /// - [`NeighborMode::All`]: the graph is treated as undirected.
    ///
    /// The result has one row per vertex of `vids`, in the order given, and
    /// one column per vertex of the graph. In a simple graph the
    /// self-similarities (the diagonal) are zero, and isolated vertices have
    /// zero similarity to all others. Multi-edges count with multiplicity,
    /// and an undirected loop makes a vertex its own neighbor twice (which
    /// also makes the diagonal nonzero). This can raise *or* lower a
    /// similarity, since it also raises the degrees used as weights, so consider
    /// [`simplify`](Graph::simplify)ing the graph first.
    ///
    /// Binds [`igraph_similarity_inverse_log_weighted`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_similarity_inverse_log_weighted).
    /// Time complexity: `O(|V| d²)`, with `d` the maximum degree.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) (or
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for a range)
    /// if `vids` names a missing vertex.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // 0 and 1 share the neighbor 2, which has degree 3.
    /// let g = Graph::from_edges(&[(0, 2), (1, 2), (2, 3)], 4, false)?;
    /// let m = g.similarity_inverse_log_weighted(0, NeighborMode::All)?;
    /// assert!((m[(0, 1)] - 1.0 / 3f64.ln()).abs() < 1e-12);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn similarity_inverse_log_weighted<'a>(
        &self,
        vids: impl Into<VertexSelector<'a>>,
        mode: NeighborMode,
    ) -> Result<Matrix> {
        self.per_vertex_rows(vids, |g, res, vs| {
            igraph_call!(igraph_similarity_inverse_log_weighted(
                g,
                res,
                vs,
                mode.into()
            ))
        })
    }

    /// Jaccard similarity of every pair in `from × to`: the number of common
    /// neighbors divided by the number of vertices adjacent to at least one of
    /// the two.
    ///
    /// `res[(i, j)]` is the similarity of the `i`-th vertex of `from` and the
    /// `j`-th vertex of `to`. `mode` selects out-, in- or all neighbors in
    /// directed graphs (ignored for undirected ones). Neighbor *sets* are
    /// compared, so loops and multi-edges are ignored. With `loops = true`,
    /// each vertex is added to its own neighbor set, which is the usual choice
    /// for link prediction ("closed neighborhoods"). A vertex has similarity
    /// 1 with itself. Two vertices with no neighbors at all have similarity 0.
    ///
    /// Binds [`igraph_similarity_jaccard`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_similarity_jaccard).
    /// When `from` and `to` are the same list with no repeated vertex, this
    /// calls that function. Otherwise it computes the pairs with
    /// `igraph_similarity_jaccard_pairs`, because the C function is wrong
    /// (and can write out of bounds) in that case (see the
    /// [module docs](self#differences-from-the-c-functions)).
    /// Time complexity: `O(|from| |to| d)`, with `d` the maximum degree.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) (or
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for a range)
    /// if a selector names a missing vertex.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A 4-cycle 0-1-2-3: opposite vertices have the same neighbors.
    /// let g = Graph::ring(4, false, false, true)?;
    /// let m = g.similarity_jaccard(.., .., NeighborMode::All, false)?;
    /// assert_eq!(m[(0, 2)], 1.0);
    /// assert_eq!(m[(0, 1)], 0.0);
    /// // Rectangular: the rows are {0}, the columns {1, 2}.
    /// let r = g.similarity_jaccard(0, &[1, 2], NeighborMode::All, true)?;
    /// assert_eq!(r.shape(), (1, 2));
    /// assert_eq!(r[(0, 1)], 0.5); // {0,1,3} vs {1,2,3}
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn similarity_jaccard<'a, 'b>(
        &self,
        from: impl Into<VertexSelector<'a>>,
        to: impl Into<VertexSelector<'b>>,
        mode: NeighborMode,
        loops: bool,
    ) -> Result<Matrix> {
        self.similarity_matrix(from.into(), to.into(), mode, loops, false)
    }

    /// Jaccard similarity of each vertex pair in `pairs`, in the same order.
    ///
    /// See [`similarity_jaccard`](Self::similarity_jaccard) for the meaning
    /// of `mode` and `loops`. A pair `(v, v)` has similarity 1.
    ///
    /// Binds [`igraph_similarity_jaccard_pairs`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_similarity_jaccard_pairs).
    /// Time complexity: `O(n d)` for `n` pairs, with `d` the maximum degree.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) if a
    /// pair names a missing vertex.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A star with center 0: all leaves have the same neighbor set {0}.
    /// let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], 4, false)?;
    /// let s = g.similarity_jaccard_pairs(&[(1, 2), (0, 1), (3, 3)], NeighborMode::All, false)?;
    /// assert_eq!(s, vec![1.0, 0.0, 1.0]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn similarity_jaccard_pairs(
        &self,
        pairs: &[(VertexId, VertexId)],
        mode: NeighborMode,
        loops: bool,
    ) -> Result<Vec<f64>> {
        let flat: Vec<i64> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
        self.jaccard_pairs_flat(&flat, mode, loops, false)
    }

    /// Jaccard similarity of the two endpoints of each edge of `es`, in the
    /// order of the selector.
    ///
    /// See [`similarity_jaccard`](Self::similarity_jaccard) for the meaning
    /// of `mode` and `loops`. A low value marks an edge between vertices with
    /// different neighborhoods, such as a "bridge" between communities.
    ///
    /// Binds [`igraph_similarity_jaccard_es`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_similarity_jaccard_es).
    /// Time complexity: `O(n d)` for `n` edges, with `d` the maximum degree.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidEdgeId`](crate::ErrorKind::InvalidEdgeId) (or another
    /// kind, depending on the selector) if `es` names missing edges.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // Two triangles joined by the edge 2-3.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)], 6, false)?;
    /// let s = g.similarity_jaccard_es(.., NeighborMode::All, true)?;
    /// // Inside a triangle: {0,1,2} vs {0,1,2,3} = 3/4; across: {0,1,2,3} vs {2,3,4,5} = 2/6.
    /// assert_eq!(s[1], 0.75);
    /// assert!((s[3] - 1.0 / 3.0).abs() < 1e-12);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn similarity_jaccard_es<'a>(
        &self,
        es: impl Into<EdgeSelector<'a>>,
        mode: NeighborMode,
        loops: bool,
    ) -> Result<Vec<f64>> {
        self.similarity_es(es.into(), mode, loops, false)
    }

    /// Dice similarity of every pair in `from × to`: twice the number of
    /// common neighbors divided by the sum of the two neighbor-set sizes.
    ///
    /// Dice and Jaccard similarities are related by `D = 2J / (1 + J)`, so
    /// they rank pairs the same way. Everything else (shape, `mode`, `loops`,
    /// self-similarity 1, and the fallback to the pairs function when `from`
    /// and `to` differ) is as for [`similarity_jaccard`](Self::similarity_jaccard).
    ///
    /// Binds [`igraph_similarity_dice`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_similarity_dice).
    /// Time complexity: `O(|from| |to| d)`, with `d` the maximum degree.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) (or
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for a range)
    /// if a selector names a missing vertex.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::ring(4, false, false, true)?;
    /// let m = g.similarity_dice(.., .., NeighborMode::All, true)?;
    /// // {0,1,3} vs {0,1,2}: 2 * 2 / (3 + 3).
    /// assert!((m[(0, 1)] - 2.0 / 3.0).abs() < 1e-12);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn similarity_dice<'a, 'b>(
        &self,
        from: impl Into<VertexSelector<'a>>,
        to: impl Into<VertexSelector<'b>>,
        mode: NeighborMode,
        loops: bool,
    ) -> Result<Matrix> {
        self.similarity_matrix(from.into(), to.into(), mode, loops, true)
    }

    /// Dice similarity of each vertex pair in `pairs`, in the same order.
    ///
    /// See [`similarity_dice`](Self::similarity_dice).
    ///
    /// Binds [`igraph_similarity_dice_pairs`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_similarity_dice_pairs).
    /// Time complexity: `O(n d)` for `n` pairs, with `d` the maximum degree.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`](crate::ErrorKind::InvalidVertexId) if a
    /// pair names a missing vertex.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 3)], 4, false)?;
    /// // N(0) = {1, 2}, N(3) = {1}: 2 * 1 / (2 + 1).
    /// let s = g.similarity_dice_pairs(&[(0, 3)], NeighborMode::All, false)?;
    /// assert!((s[0] - 2.0 / 3.0).abs() < 1e-12);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn similarity_dice_pairs(
        &self,
        pairs: &[(VertexId, VertexId)],
        mode: NeighborMode,
        loops: bool,
    ) -> Result<Vec<f64>> {
        let flat: Vec<i64> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
        self.jaccard_pairs_flat(&flat, mode, loops, true)
    }

    /// Dice similarity of the two endpoints of each edge of `es`, in the
    /// order of the selector.
    ///
    /// See [`similarity_dice`](Self::similarity_dice) and
    /// [`similarity_jaccard_es`](Self::similarity_jaccard_es).
    ///
    /// Binds [`igraph_similarity_dice_es`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_similarity_dice_es).
    /// Time complexity: `O(n d)` for `n` edges, with `d` the maximum degree.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidEdgeId`](crate::ErrorKind::InvalidEdgeId) (or another
    /// kind, depending on the selector) if `es` names missing edges.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, false)?;
    /// // In a triangle, closed neighborhoods are all {0, 1, 2}.
    /// assert_eq!(g.similarity_dice_es(.., NeighborMode::All, true)?, vec![1.0; 3]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn similarity_dice_es<'a>(
        &self,
        es: impl Into<EdgeSelector<'a>>,
        mode: NeighborMode,
        loops: bool,
    ) -> Result<Vec<f64>> {
        self.similarity_es(es.into(), mode, loops, true)
    }

    fn similarity_matrix(
        &self,
        from: VertexSelector<'_>,
        to: VertexSelector<'_>,
        mode: NeighborMode,
        loops: bool,
        dice: bool,
    ) -> Result<Matrix> {
        let from = self.select_vertices(from)?;
        let to = self.select_vertices(to)?;
        let mut sorted = from.clone();
        sorted.sort_unstable();
        let distinct = sorted.windows(2).all(|w| w[0] != w[1]);
        if from == to && distinct {
            // The only case the C matrix functions handle correctly.
            let vs = VertexSelector::List(Cow::Borrowed(&from)).to_raw()?;
            let cfn = if dice {
                igraph_similarity_dice
            } else {
                igraph_similarity_jaccard
            };
            let mut res = Matrix::new();
            with_multi_cache_guard(self, || {
                igraph_call!(cfn(self, &mut res, vs.get(), vs.get(), mode.into(), loops))
            })?;
            return Ok(res);
        }
        let mut flat = Vec::with_capacity(2 * from.len() * to.len());
        for &u in &from {
            for &v in &to {
                flat.extend([u, v]);
            }
        }
        let values = self.jaccard_pairs_flat(&flat, mode, loops, dice)?;
        if values.is_empty() {
            let mut m = Matrix::new();
            m.resize(from.len(), to.len());
            return Ok(m);
        }
        Matrix::from_row_major(from.len(), to.len(), &values)
    }

    fn jaccard_pairs_flat(
        &self,
        flat: &[i64],
        mode: NeighborMode,
        loops: bool,
        dice: bool,
    ) -> Result<Vec<f64>> {
        debug_assert!(flat.len().is_multiple_of(2));
        let pairs = VectorInt::view(flat);
        let mut res = Vector::new();
        let cfn = if dice {
            igraph_similarity_dice_pairs
        } else {
            igraph_similarity_jaccard_pairs
        };
        with_multi_cache_guard(self, || {
            igraph_call!(cfn(self, &mut res, pairs.as_ptr(), mode.into(), loops))
        })?;
        Ok(res.into())
    }

    fn similarity_es(
        &self,
        es: EdgeSelector<'_>,
        mode: NeighborMode,
        loops: bool,
        dice: bool,
    ) -> Result<Vec<f64>> {
        let es = es.to_raw()?;
        let mut res = Vector::new();
        let cfn = if dice {
            igraph_similarity_dice_es
        } else {
            igraph_similarity_jaccard_es
        };
        with_multi_cache_guard(self, || {
            igraph_call!(cfn(self, &mut res, es.get(), mode.into(), loops))
        })?;
        Ok(res.into())
    }
}

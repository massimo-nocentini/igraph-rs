//! Conversion of graphs to matrices, edge lists, Prüfer sequences, and
//! between directed and undirected graphs (`igraph_conversion.h`).
//!
//! This module binds the whole of igraph's *conversion* chapter. Every
//! function is a method of [`Graph`]:
//!
//! | Method | C function | Result |
//! |---|---|---|
//! | [`Graph::get_adjacency`] | [`igraph_get_adjacency`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_adjacency) | dense adjacency [`Matrix`] (optionally weighted) |
//! | [`Graph::get_adjacency_sparse`] | [`igraph_get_adjacency_sparse`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_adjacency_sparse) | sparse adjacency matrix as a [`CooMatrix`] |
//! | [`Graph::get_stochastic`] | [`igraph_get_stochastic`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_stochastic) | row- or column-stochastic transition [`Matrix`] |
//! | [`Graph::get_stochastic_sparse`] | [`igraph_get_stochastic_sparse`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_stochastic_sparse) | the same, as a [`CooMatrix`] |
//! | [`Graph::get_edgelist`] | [`igraph_get_edgelist`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_edgelist) | flat edge list, row- or column-wise |
//! | [`Graph::to_directed`] / [`Graph::into_directed`] | [`igraph_to_directed`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_to_directed) | undirected → directed, in place / by value |
//! | [`Graph::to_undirected`] / [`Graph::into_undirected`] | [`igraph_to_undirected`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_to_undirected) | directed → undirected, in place / by value |
//! | [`Graph::to_undirected_with_comb`] | [`igraph_to_undirected`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_to_undirected) | the same, combining the edge attributes of merged edges |
//! | [`Graph::to_prufer`] | [`igraph_to_prufer`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_to_prufer) | Prüfer sequence of a labelled tree |
//!
//! # Sparse results
//!
//! igraph returns sparse matrices as `igraph_sparsemat_t` (a CXSparse
//! matrix, exposed by this crate as [`SparseMat`]). The sparse wrappers of
//! this module read that matrix out into a plain-Rust [`CooMatrix`]
//! (coordinate / triplet format), with duplicate entries summed up and the
//! entries sorted in row-major order. It is easy to inspect, compare and
//! iterate over; convert it back with [`CooMatrix::to_sparsemat`] when you
//! need the sparse linear algebra of [`crate::linalg`] (solvers,
//! factorizations, ARPACK eigensolvers).
//!
//! # See also
//!
//! - The inverse conversions are constructors in [`crate::constructors`]:
//!   [`Graph::adjacency`], [`Graph::weighted_adjacency`],
//!   [`Graph::sparse_adjacency`] and [`Graph::sparse_weighted_adjacency`]
//!   build a graph from an adjacency matrix, [`Graph::from_flat_edges`]
//!   from the output of [`Graph::get_edgelist`], and [`Graph::from_prufer`]
//!   from a Prüfer sequence.
//! - Other matrices of a graph: the Laplacian ([`Graph::get_laplacian`],
//!   [`Graph::get_laplacian_sparse`]) in [`crate::structural`], and the
//!   spectra of adjacency matrices ([`Graph::eigen_adjacency`]) in
//!   [`crate::linalg`].
//! - Random walks: [`Graph::random_walk`] simulates the walk whose
//!   transition matrix is [`Graph::get_stochastic`], and
//!   [`Graph::pagerank`] computes its (damped) stationary distribution.
//! - Directedness: [`Graph::is_mutual`], [`Graph::has_mutual`] and
//!   [`Graph::reciprocity`] tell how much [`ToUndirected::Mutual`] and
//!   [`ToUndirected::Collapse`] will differ; [`Graph::is_dag`] checks the
//!   result of [`ToDirected::Acyclic`].
//!
//! # Example
//!
//! ```
//! use igraph::prelude::*;
//!
//! // A directed 3-cycle 0 → 1 → 2 → 0.
//! let mut g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, true).unwrap();
//! let a = g.get_adjacency(GetAdjacency::Both, None, Loops::Twice).unwrap();
//! assert_eq!(a.to_rows(), vec![vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0], vec![1.0, 0.0, 0.0]]);
//!
//! // Forget the directions: the adjacency matrix becomes symmetric.
//! g.to_undirected(ToUndirected::Collapse).unwrap();
//! let a = g.get_adjacency(GetAdjacency::Both, None, Loops::Twice).unwrap();
//! assert_eq!(a, a.transposed());
//! // ... and it is enough to rebuild the graph.
//! let h = Graph::adjacency(&a, Adjacency::Undirected, Loops::Twice).unwrap();
//! assert_eq!(h.ecount(), 3);
//!
//! // Random walk transition probabilities: each row sums to one.
//! let p = g.get_stochastic(false, None).unwrap();
//! assert!(p.rows().all(|r| (r.iter().sum::<f64>() - 1.0).abs() < 1e-12));
//!
//! // A path is a tree: its Prüfer sequence lists the inner vertices.
//! let path = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
//! assert_eq!(path.to_prufer().unwrap(), vec![1, 2]);
//! assert!(Graph::from_prufer(&[1, 2]).unwrap().is_same_graph(&path).unwrap());
//! ```
//!
//! On a larger scale, with Zachary's karate club network: the sparse
//! adjacency matrix stores `2|E|` entries, and its row sums are the degrees.
//!
//! ```
//! use igraph::prelude::*;
//!
//! let karate = Graph::famous("Zachary").unwrap();
//! let a = karate.get_adjacency_sparse(GetAdjacency::Both, None, Loops::Twice).unwrap();
//! assert_eq!(a.nnz(), 2 * karate.ecount());
//! let degrees = karate.degree(.., NeighborMode::All, Loops::Twice).unwrap();
//! let row_sums: Vec<i64> = a.row_sums().iter().map(|&s| s as i64).collect();
//! assert_eq!(row_sums, degrees);
//! ```

use crate::{
    attributes::AttributeCombination,
    constants::{GetAdjacency, Loops, NeighborMode, ToDirected, ToUndirected},
    error::{Error, Result},
    ffi::*,
    graph::{Graph, VertexId},
    igraph_call,
    linalg::SparseMat,
    matrix::Matrix,
    vector::{Vector, VectorInt},
};
use std::collections::BTreeMap;

/// A sparse real matrix in coordinate (triplet) format.
///
/// Returned by the sparse conversion functions of this module
/// ([`Graph::get_adjacency_sparse`], [`Graph::get_stochastic_sparse`]).
/// Each stored element appears exactly once in [`entries`](Self::entries)
/// as a `(row, column, value)` triple; entries are sorted by row, then by
/// column. Elements that are not listed are zero.
///
/// It is a plain-Rust snapshot of an igraph [`SparseMat`], with a few
/// helpers (products with vectors, row and column sums, transposition);
/// [`to_sparsemat`](Self::to_sparsemat) and [`to_dense`](Self::to_dense)
/// convert it to igraph's sparse and dense matrices.
///
/// ```
/// use igraph::prelude::*;
/// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
/// let a = g.get_adjacency_sparse(GetAdjacency::Upper, None, Loops::Twice).unwrap();
/// assert_eq!((a.nrow, a.ncol), (3, 3));
/// assert_eq!(a.entries, vec![(0, 1, 1.0), (1, 2, 1.0)]);
/// assert_eq!(a.get(1, 2), 1.0);
/// assert_eq!(a.get(2, 1), 0.0);
/// assert_eq!(a.to_dense(), g.get_adjacency(GetAdjacency::Upper, None, Loops::Twice).unwrap());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CooMatrix {
    /// Number of rows.
    pub nrow: usize,
    /// Number of columns.
    pub ncol: usize,
    /// The stored `(row, column, value)` triples, sorted by `(row, column)`,
    /// without duplicate positions.
    pub entries: Vec<(i64, i64, f64)>,
}

impl CooMatrix {
    /// Number of stored entries.
    ///
    /// Every structurally non-zero position is stored once. A stored value
    /// may still be `0.0`, e.g. for an edge of weight zero, or in the rows
    /// of zero-strength vertices of a stochastic matrix.
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// The element at `(row, col)`, zero if it is not stored.
    ///
    /// Runs in `O(log nnz)` time with a binary search, so it relies on the
    /// entries being sorted and unique, as they are in every matrix returned
    /// by this module.
    pub fn get(&self, row: i64, col: i64) -> f64 {
        self.entries
            .binary_search_by(|&(i, j, _)| (i, j).cmp(&(row, col)))
            .map_or(0.0, |k| self.entries[k].2)
    }

    /// Iterates over the stored `(row, column, value)` triples, in
    /// row-major order.
    pub fn iter(&self) -> impl Iterator<Item = (i64, i64, f64)> + '_ {
        self.entries.iter().copied()
    }

    /// The transposed matrix (`ncol × nrow`), again sorted in row-major
    /// order.
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (0, 2)], 3, true).unwrap();
    /// let a = g.get_adjacency_sparse(GetAdjacency::Both, None, Loops::Once).unwrap();
    /// assert_eq!(a.transpose().entries, vec![(1, 0, 1.0), (2, 0, 1.0)]);
    /// ```
    pub fn transpose(&self) -> CooMatrix {
        let mut entries: Vec<_> = self.entries.iter().map(|&(i, j, x)| (j, i, x)).collect();
        entries.sort_by_key(|&(i, j, _)| (i, j));
        CooMatrix {
            nrow: self.ncol,
            ncol: self.nrow,
            entries,
        }
    }

    /// The matrix-vector product `A · x`, a vector of length
    /// [`nrow`](Self::nrow). Runs in `O(nrow + nnz)` time.
    ///
    /// # Panics
    /// If `x.len() != ncol`, or if an entry lies outside the matrix.
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // Out-degrees of a directed graph: A · 1.
    /// let g = Graph::from_edges(&[(0, 1), (0, 2), (2, 1)], 3, true).unwrap();
    /// let a = g.get_adjacency_sparse(GetAdjacency::Both, None, Loops::Once).unwrap();
    /// assert_eq!(a.mul_vec(&[1.0; 3]), vec![2.0, 0.0, 1.0]);
    /// ```
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.ncol, "vector length must equal ncol");
        let mut y = vec![0.0; self.nrow];
        for &(i, j, v) in &self.entries {
            y[i as usize] += v * x[j as usize];
        }
        y
    }

    /// The vector-matrix product `xᵀ · A`, a vector of length
    /// [`ncol`](Self::ncol). Runs in `O(ncol + nnz)` time.
    ///
    /// With a row-stochastic matrix `P` (see
    /// [`Graph::get_stochastic_sparse`]) this is one step of a random walk:
    /// if `x` is the distribution of the walker, `xᵀ · P` is its distribution
    /// after one more step.
    ///
    /// # Panics
    /// If `x.len() != nrow`, or if an entry lies outside the matrix.
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // A walker on the directed cycle 0 → 1 → 2 → 0 moves one step ahead.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, true).unwrap();
    /// let p = g.get_stochastic_sparse(false, None).unwrap();
    /// assert_eq!(p.vec_mul(&[1.0, 0.0, 0.0]), vec![0.0, 1.0, 0.0]);
    /// ```
    pub fn vec_mul(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.nrow, "vector length must equal nrow");
        let mut y = vec![0.0; self.ncol];
        for &(i, j, v) in &self.entries {
            y[j as usize] += x[i as usize] * v;
        }
        y
    }

    /// Converts to a dense [`Matrix`] of size `nrow × ncol`.
    ///
    /// # Panics
    /// If an entry lies outside the matrix (only possible for a
    /// hand-built `CooMatrix`).
    pub fn to_dense(&self) -> Matrix {
        let mut m = Matrix::zeros(self.nrow, self.ncol);
        for &(i, j, x) in &self.entries {
            m[(i as usize, j as usize)] += x;
        }
        m
    }

    /// Row sums, a vector of length [`nrow`](Self::nrow).
    ///
    /// # Panics
    /// If an entry lies outside the matrix.
    pub fn row_sums(&self) -> Vec<f64> {
        let mut s = vec![0.0; self.nrow];
        for &(i, _, x) in &self.entries {
            s[i as usize] += x;
        }
        s
    }

    /// Column sums, a vector of length [`ncol`](Self::ncol).
    ///
    /// # Panics
    /// If an entry lies outside the matrix.
    pub fn col_sums(&self) -> Vec<f64> {
        let mut s = vec![0.0; self.ncol];
        for &(_, j, x) in &self.entries {
            s[j as usize] += x;
        }
        s
    }

    /// Converts to an igraph sparse matrix ([`SparseMat`], in triplet
    /// format), for the sparse linear algebra of [`crate::linalg`]. Duplicate
    /// positions (possible only in a hand-built `CooMatrix`) are summed.
    /// Most [`SparseMat`] operations accept the triplet format and compress a
    /// temporary copy when they need to; call [`SparseMat::compress`] once
    /// up front when the matrix is used repeatedly.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if an
    /// entry has a negative index or lies outside the matrix (only possible
    /// for a hand-built `CooMatrix`).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // The squared adjacency matrix of a path counts the walks of length 2.
    /// let path = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let a = path
    ///     .get_adjacency_sparse(GetAdjacency::Both, None, Loops::Twice)
    ///     .unwrap()
    ///     .to_sparsemat()
    ///     .unwrap()
    ///     .compress()
    ///     .unwrap();
    /// let a2 = a.multiply(&a).unwrap();
    /// assert_eq!(a2.get(0, 2), 1.0); // 0 - 1 - 2
    /// assert_eq!(a2.get(1, 1), 2.0); // 1 - 0 - 1 and 1 - 2 - 1
    /// assert_eq!(a2.get(0, 3), 0.0);
    /// ```
    pub fn to_sparsemat(&self) -> Result<SparseMat> {
        let triplets = self
            .entries
            .iter()
            .map(
                |&(i, j, x)| match (usize::try_from(i), usize::try_from(j)) {
                    (Ok(i), Ok(j)) => Ok((i, j, x)),
                    _ => Err(Error::invalid(format!(
                        "negative index ({i}, {j}) in a sparse matrix"
                    ))),
                },
            )
            .collect::<Result<Vec<_>>>()?;
        SparseMat::from_triplets(self.nrow, self.ncol, &triplets)
    }
}

impl From<&CooMatrix> for Matrix {
    fn from(m: &CooMatrix) -> Self {
        m.to_dense()
    }
}

/// Reads a sparse matrix returned by igraph (triplet or column-compressed)
/// into coordinate format, summing duplicate positions and sorting the
/// entries in row-major order.
fn sparsemat_to_coo(a: &SparseMat) -> CooMatrix {
    let mut acc: BTreeMap<(i64, i64), f64> = BTreeMap::new();
    for (i, j, x) in a.iter() {
        *acc.entry((i as i64, j as i64)).or_insert(0.0) += x;
    }
    CooMatrix {
        nrow: a.nrow(),
        ncol: a.ncol(),
        entries: acc.into_iter().map(|((i, j), x)| (i, j, x)).collect(),
    }
}

/// Clears the rows (or columns, if `column_wise`) of the dense stochastic
/// matrix `res` that belong to vertices of zero strength.
///
/// igraph's dense `igraph_get_stochastic` divides by the strength without
/// checking it (still so in igraph 1.0.0 and 1.0.1), so such rows are
/// `0 / 0 = NaN` whenever a vertex has edges but they all have weight zero.
/// The sparse version (normalizing with `allow_zeros = true`) and the C docs
/// give zeros instead.
///
/// The strengths come from the very call the C function makes
/// (`igraph_strength` with `IGRAPH_LOOPS`, in the same mode), so the zero
/// test sees bit-for-bit the same sums as the division did, even when
/// mixed-sign weights make a sum depend on the order of the additions.
fn zero_out_null_strength(
    graph: &Graph,
    res: &mut Matrix,
    column_wise: bool,
    weights: &[f64],
) -> Result<()> {
    let mode = match (graph.is_directed(), column_wise) {
        (false, _) => NeighborMode::All,
        (true, false) => NeighborMode::Out,
        (true, true) => NeighborMode::In,
    };
    let strength = graph.strength(.., mode, Loops::Twice, Some(weights))?;
    let n = strength.len();
    for (v, _) in strength.iter().enumerate().filter(|&(_, &s)| s == 0.0) {
        for u in 0..n {
            if column_wise {
                res[(u, v)] = 0.0;
            } else {
                res[(v, u)] = 0.0;
            }
        }
    }
    Ok(())
}

/// Checks that an optional weight vector has one entry per edge.
///
/// igraph's conversion functions index the weight vector by edge id without
/// checking its length, so this check is required for memory safety.
fn check_weights(graph: &Graph, weights: Option<&[f64]>) -> Result<()> {
    match weights {
        Some(w) if w.len() != graph.ecount() => Err(Error::invalid(format!(
            "weight vector length ({}) must match the number of edges ({})",
            w.len(),
            graph.ecount()
        ))),
        _ => Ok(()),
    }
}

impl igraph_t {
    /// The (dense) adjacency matrix of the graph.
    ///
    /// Entry `(i, j)` of the returned `n × n` [`Matrix`] is the number of
    /// edges from vertex `i` to vertex `j` or, when `weights` are given, the
    /// total weight of those edges (so multi-edges add up).
    ///
    /// - `kind` selects which part of the matrix is filled for *undirected*
    ///   graphs: [`GetAdjacency::Upper`] (upper-right triangle only, each edge
    ///   stored once), [`GetAdjacency::Lower`] (lower-left triangle) or
    ///   [`GetAdjacency::Both`] (the full symmetric matrix). It is ignored
    ///   for directed graphs.
    /// - `weights`: optional edge weights, one per edge; `None` means every
    ///   edge has weight 1.
    /// - `loops` controls the diagonal: [`Loops::None`] ignores self-loops
    ///   (zero diagonal), [`Loops::Once`] counts each loop once and
    ///   [`Loops::Twice`] counts loops twice in *undirected* graphs (it
    ///   counts edge *stems*, the convention that makes row sums equal to
    ///   degrees). In directed graphs `Twice` behaves like `Once`.
    ///
    /// Time complexity: `O(|V|²)`.
    ///
    /// Binds [`igraph_get_adjacency`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_adjacency).
    /// See [`get_adjacency_sparse`](Self::get_adjacency_sparse) for large,
    /// sparse graphs.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `weights` has the
    /// wrong length.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A triangle with a double edge 0-1 and a loop on vertex 2.
    /// let g = Graph::from_edges(&[(0, 1), (0, 1), (1, 2), (2, 0), (2, 2)], 3, false).unwrap();
    /// let a = g.get_adjacency(GetAdjacency::Both, None, Loops::Twice).unwrap();
    /// assert_eq!(a.to_rows(), vec![
    ///     vec![0.0, 2.0, 1.0],
    ///     vec![2.0, 0.0, 1.0],
    ///     vec![1.0, 1.0, 2.0],
    /// ]);
    /// // Row sums are the degrees.
    /// let degrees: Vec<f64> = a.rows().map(|r| r.iter().sum()).collect();
    /// assert_eq!(degrees, vec![3.0, 3.0, 4.0]);
    ///
    /// // Weighted, upper triangle, loops ignored.
    /// let w = [0.5, 1.5, 2.0, 3.0, 9.0];
    /// let a = g.get_adjacency(GetAdjacency::Upper, Some(&w), Loops::None).unwrap();
    /// assert_eq!(a.to_rows(), vec![
    ///     vec![0.0, 2.0, 3.0],
    ///     vec![0.0, 0.0, 2.0],
    ///     vec![0.0, 0.0, 0.0],
    /// ]);
    /// ```
    pub fn get_adjacency(
        &self,
        kind: GetAdjacency,
        weights: Option<&[f64]>,
        loops: Loops,
    ) -> Result<Matrix> {
        check_weights(self, weights)?;
        let w = weights.map(Vector::view);
        let wp = w.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let mut res = Matrix::new();
        igraph_call!(igraph_get_adjacency(
            self,
            &mut res,
            kind.into(),
            wp,
            loops.into()
        ))?;
        Ok(res)
    }

    /// The adjacency matrix of the graph in sparse (coordinate) format.
    ///
    /// Same semantics as [`get_adjacency`](Self::get_adjacency) (`kind`,
    /// `weights` and `loops` mean exactly the same), but only the non-zero
    /// entries are stored, so the memory use is `O(|V| + |E|)` instead of
    /// `O(|V|²)`. The C result (an `igraph_sparsemat_t`) is read into a
    /// [`CooMatrix`] whose parallel entries (from multi-edges) are summed.
    ///
    /// Binds [`igraph_get_adjacency_sparse`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_adjacency_sparse).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `weights` has the
    /// wrong length.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A star with 1000 leaves: 1001² ≈ 10⁶ dense entries, only 2000 stored.
    /// let edges: Vec<(i64, i64)> = (1..=1000).map(|i| (0, i)).collect();
    /// let star = Graph::from_edges(&edges, 1001, false).unwrap();
    /// let a = star.get_adjacency_sparse(GetAdjacency::Both, None, Loops::Twice).unwrap();
    /// assert_eq!(a.nnz(), 2000);
    /// assert_eq!(a.row_sums()[0], 1000.0);
    /// assert_eq!(a.get(7, 0), 1.0);
    /// ```
    pub fn get_adjacency_sparse(
        &self,
        kind: GetAdjacency,
        weights: Option<&[f64]>,
        loops: Loops,
    ) -> Result<CooMatrix> {
        check_weights(self, weights)?;
        let w = weights.map(Vector::view);
        let wp = w.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let mut res = SparseMat::new(0, 0)?;
        igraph_call!(igraph_get_adjacency_sparse(
            self,
            &mut res,
            kind.into(),
            wp,
            loops.into()
        ))?;
        Ok(sparsemat_to_coo(&res))
    }

    /// The stochastic (random walk transition) matrix of the graph.
    ///
    /// This is the adjacency matrix normalized so that each row sums to one
    /// (`column_wise = false`, a *right-stochastic* matrix) or each column
    /// sums to one (`column_wise = true`, *left-stochastic*). Row-wise,
    /// entry `(i, j)` is the probability that a random walker at `i` steps
    /// to `j` following the edge directions; the column-wise matrix relates
    /// to walks moving against the edge directions. Rows (columns) of
    /// vertices with zero out-strength (in-strength) are all zeros.
    ///
    /// Undirected self-loops are counted twice, consistently with degrees.
    /// `weights` are optional edge weights (`None`: all 1), which make the
    /// transition probabilities proportional to the weights. They should be
    /// non-negative, otherwise the result is not a probability matrix.
    ///
    /// A vertex whose incident edges all have weight zero has zero strength:
    /// its row (column) is all zeros, like for a vertex without edges. (The
    /// C function of igraph 1.0.0 and 1.0.1 would fill it with
    /// `0 / 0 = NaN`; this wrapper clears those rows so that the dense and
    /// sparse versions agree, as the C documentation promises.)
    ///
    /// With negative weights a vertex can also have zero strength because
    /// its weights cancel out; its row (column) is cleared as well, like in
    /// [`get_stochastic_sparse`](Self::get_stochastic_sparse).
    ///
    /// The strengths used for the normalization are those of
    /// [`Graph::strength`] with [`Loops::Twice`]
    /// (out-strengths row-wise, in-strengths column-wise). Time complexity:
    /// `O(|V|²)`.
    ///
    /// Binds [`igraph_get_stochastic`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_stochastic).
    /// See also [`Graph::random_walk`], which simulates this walk, and
    /// [`Graph::pagerank`], its damped stationary distribution.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `weights` has the
    /// wrong length.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // Star 0-1, 0-2, 0-3: from the center, each leaf with probability 1/3.
    /// let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], 4, false).unwrap();
    /// let p = g.get_stochastic(false, None).unwrap();
    /// assert_eq!(p.row(0), vec![0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    /// assert_eq!(p.row(2), vec![1.0, 0.0, 0.0, 0.0]);
    /// // Column-wise it is the transpose, for an undirected graph.
    /// assert_eq!(g.get_stochastic(true, None).unwrap(), p.transposed());
    /// ```
    pub fn get_stochastic(&self, column_wise: bool, weights: Option<&[f64]>) -> Result<Matrix> {
        check_weights(self, weights)?;
        let w = weights.map(Vector::view);
        let wp = w.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let mut res = Matrix::new();
        igraph_call!(igraph_get_stochastic(self, &mut res, column_wise, wp))?;
        if let Some(w) = weights {
            zero_out_null_strength(self, &mut res, column_wise, w)?;
        }
        Ok(res)
    }

    /// The stochastic matrix of the graph in sparse (coordinate) format.
    ///
    /// Same as [`get_stochastic`](Self::get_stochastic), but computed in
    /// `O(|V| + |E|)` time and returned as a [`CooMatrix`]. Entries of
    /// zero-strength vertices (all incident weights zero) are stored as
    /// explicit `0.0` values; so are those of a vertex whose (mixed-sign)
    /// weights sum to zero, as igraph scales such a row (column) by zero.
    ///
    /// Binds [`igraph_get_stochastic_sparse`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_stochastic_sparse).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `weights` has the
    /// wrong length.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // Directed: 0 → 1 (weight 3), 0 → 2 (weight 1); vertices 1, 2 are sinks.
    /// let g = Graph::from_edges(&[(0, 1), (0, 2)], 3, true).unwrap();
    /// let p = g.get_stochastic_sparse(false, Some(&[3.0, 1.0])).unwrap();
    /// assert_eq!(p.entries, vec![(0, 1, 0.75), (0, 2, 0.25)]);
    /// assert_eq!(p.row_sums(), vec![1.0, 0.0, 0.0]);
    /// ```
    pub fn get_stochastic_sparse(
        &self,
        column_wise: bool,
        weights: Option<&[f64]>,
    ) -> Result<CooMatrix> {
        check_weights(self, weights)?;
        let w = weights.map(Vector::view);
        let wp = w.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let mut res = SparseMat::new(0, 0)?;
        igraph_call!(igraph_get_stochastic_sparse(
            self,
            &mut res,
            column_wise,
            wp
        ))?;
        Ok(sparsemat_to_coo(&res))
    }

    /// The list of all edges as a flat vector, in edge id order.
    ///
    /// With `bycol = false` the result is `[from0, to0, from1, to1, …]`, the
    /// format accepted by [`Graph::from_flat_edges`] and
    /// [`Graph::add_edges_from_vector`]. With `bycol = true` it is
    /// "column-wise": first all the sources, then all the targets,
    /// `[from0, from1, …, to0, to1, …]`, i.e. edge `e` is
    /// `res[e] → res[|E| + e]`. For `(from, to)` pairs see also
    /// [`Graph::edge_list`].
    ///
    /// Time complexity: `O(|E|)`.
    ///
    /// Binds [`igraph_get_edgelist`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_get_edgelist).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (3, 2)], 4, true).unwrap();
    /// assert_eq!(g.get_edgelist(false).unwrap(), vec![0, 1, 1, 2, 3, 2]);
    /// assert_eq!(g.get_edgelist(true).unwrap(), vec![0, 1, 3, 1, 2, 2]);
    /// // Round trip.
    /// let h = Graph::from_flat_edges(&g.get_edgelist(false).unwrap(), 4, true).unwrap();
    /// assert!(g.is_same_graph(&h).unwrap());
    /// ```
    pub fn get_edgelist(&self, bycol: bool) -> Result<Vec<VertexId>> {
        let mut res = VectorInt::new();
        igraph_call!(igraph_get_edgelist(self, &mut res, bycol))?;
        Ok(res.into())
    }

    /// Converts an undirected graph into a directed one, in place.
    ///
    /// Does nothing if the graph is already directed. The vertex ids are
    /// kept; how edges are directed depends on `mode`:
    ///
    /// - [`ToDirected::Arbitrary`]: each edge becomes one directed edge with
    ///   an arbitrary (implementation-defined) direction; edge ids are kept.
    /// - [`ToDirected::Mutual`]: each edge `e` becomes two directed edges,
    ///   one per direction; the first `|E|` edges keep the ids and endpoint
    ///   order of the original ones, edge `|E| + e` is the reverse of `e`.
    /// - [`ToDirected::Random`]: each edge gets a uniformly random direction
    ///   (drawn from the calling thread's default RNG, so reproducible
    ///   after [`rng::seed`](crate::rng::seed)).
    /// - [`ToDirected::Acyclic`]: each edge is directed from the smaller to
    ///   the larger vertex id; without self-loops the result is a DAG (see
    ///   [`Graph::is_dag`]), and the identity is a topological order.
    ///
    /// `Random` and `Acyclic` keep the edge ids too.
    ///
    /// Graph, vertex and edge attributes (if an attribute handler is
    /// installed) are kept; with `Mutual` both copies of an edge get its
    /// attributes. Time complexity: `O(|V| + |E|)`.
    ///
    /// Binds [`igraph_to_directed`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_to_directed).
    /// See [`into_directed`](Self::into_directed) for a by-value variant.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let mut g = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// g.to_directed(ToDirected::Mutual).unwrap();
    /// assert!(g.is_directed());
    /// assert_eq!(g.edge_list(), vec![(0, 1), (1, 2), (1, 0), (2, 1)]);
    /// ```
    pub fn to_directed(&mut self, mode: ToDirected) -> Result<()> {
        igraph_call!(igraph_to_directed(self, mode.into()))
    }

    /// Consumes the graph and returns its directed version, see
    /// [`to_directed`](Self::to_directed).
    ///
    /// Handy in builder chains:
    /// ```
    /// use igraph::prelude::*;
    /// let dag = Graph::from_edges(&[(2, 0), (1, 2), (0, 1)], 3, false)
    ///     .and_then(|g| g.into_directed(ToDirected::Acyclic))
    ///     .unwrap();
    /// assert!(dag.edge_list().iter().all(|&(a, b)| a < b));
    /// ```
    pub fn into_directed(mut self, mode: ToDirected) -> Result<Graph> {
        self.to_directed(mode)?;
        Ok(self)
    }

    /// Converts a directed graph into an undirected one, in place.
    ///
    /// Does nothing if the graph is already undirected. `mode` decides
    /// which undirected edges are created:
    ///
    /// - [`ToUndirected::Each`]: one undirected edge per directed edge; the
    ///   number of edges (and their ids) is unchanged, so multi-edges may
    ///   appear (e.g. from a mutual pair `a → b`, `b → a`).
    /// - [`ToUndirected::Collapse`]: one undirected edge per pair of vertices
    ///   connected by at least one directed edge in either direction; no
    ///   multi-edges are created.
    /// - [`ToUndirected::Mutual`]: one undirected edge per *mutual pair* of
    ///   directed edges (`a → b` matched with `b → a`); unreciprocated edges
    ///   are dropped, while self-loops are kept unconditionally. May create
    ///   multi-edges when several mutual pairs join the same vertices.
    ///
    /// Edge attributes (with the
    /// [attribute handler](crate::attributes::enable) on) are kept with
    /// `Each` and dropped with `Collapse` and `Mutual`, because the C
    /// `edge_comb` attribute combination argument is passed as `NULL`; use
    /// [`to_undirected_with_comb`](Self::to_undirected_with_comb) or
    /// [`to_undirected_with_attributes`](Self::to_undirected_with_attributes)
    /// to combine the attributes of the merged edges instead. Graph and
    /// vertex attributes are always kept.
    ///
    /// [`Graph::is_mutual`] and [`Graph::reciprocity`] tell which edges
    /// `Mutual` keeps. Time complexity: `O(|V| + |E|)`.
    ///
    /// Binds [`igraph_to_undirected`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_to_undirected).
    /// See [`into_undirected`](Self::into_undirected) for a by-value variant.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // 0 ⇄ 1 is mutual, 1 → 2 is not.
    /// let edges = [(0, 1), (1, 0), (1, 2)];
    /// let mut each = Graph::from_edges(&edges, 3, true).unwrap();
    /// each.to_undirected(ToUndirected::Each).unwrap();
    /// assert_eq!(each.ecount(), 3);
    ///
    /// let mut collapse = Graph::from_edges(&edges, 3, true).unwrap();
    /// collapse.to_undirected(ToUndirected::Collapse).unwrap();
    /// assert_eq!(collapse.edge_list(), vec![(0, 1), (1, 2)]);
    ///
    /// let mut mutual = Graph::from_edges(&edges, 3, true).unwrap();
    /// mutual.to_undirected(ToUndirected::Mutual).unwrap();
    /// assert_eq!(mutual.edge_list(), vec![(0, 1)]);
    /// ```
    pub fn to_undirected(&mut self, mode: ToUndirected) -> Result<()> {
        igraph_call!(igraph_to_undirected(self, mode.into(), std::ptr::null()))
    }

    /// Converts a directed graph into an undirected one, in place, combining
    /// the edge attributes of the directed edges merged into each undirected
    /// edge according to `edge_comb`.
    ///
    /// The structure of the result is the same as with
    /// [`to_undirected`](Self::to_undirected). With
    /// [`ToUndirected::Collapse`] and [`ToUndirected::Mutual`], each new
    /// edge gets the attribute values combined over the directed edges it
    /// replaces (e.g. the sum of their weights, see
    /// [`AttributeCombination`]); with [`ToUndirected::Each`] edges are not
    /// merged and keep their own values. Without an attribute handler (see
    /// [`attributes::enable`](crate::attributes::enable)) this is the same
    /// as [`to_undirected`](Self::to_undirected).
    ///
    /// Binds [`igraph_to_undirected`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_to_undirected)
    /// with a non-null `edge_comb`.
    ///
    /// # Errors
    /// [`ErrorKind::AttributeCombination`](crate::ErrorKind::AttributeCombination)
    /// (or [`ErrorKind::Unimplemented`](crate::ErrorKind::Unimplemented)) if
    /// the combination is not supported for the type of an attribute.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::attributes::{AttributeCombination, AttributeCombinationType as Comb};
    ///
    /// // Traffic between two cities, in each direction.
    /// let mut g = Graph::from_edges(&[(0, 1), (1, 0), (1, 2)], 3, true).unwrap();
    /// g.set_edge_attr_numeric_values("traffic", &[10.0, 7.0, 3.0]).unwrap();
    ///
    /// let comb = AttributeCombination::from_pairs(&[(Some("traffic"), Comb::Sum)]).unwrap();
    /// g.to_undirected_with_comb(ToUndirected::Collapse, &comb).unwrap();
    /// assert_eq!(g.edge_list(), vec![(0, 1), (1, 2)]);
    /// assert_eq!(g.edge_attr_numeric_values("traffic", ..).unwrap(), vec![17.0, 3.0]);
    /// ```
    pub fn to_undirected_with_comb(
        &mut self,
        mode: ToUndirected,
        edge_comb: &AttributeCombination,
    ) -> Result<()> {
        igraph_call!(igraph_to_undirected(self, mode.into(), edge_comb.as_ptr()))
    }

    /// Consumes the graph and returns its undirected version, see
    /// [`to_undirected`](Self::to_undirected).
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 0)], 2, true).unwrap();
    /// let u = g.into_undirected(ToUndirected::Collapse).unwrap();
    /// assert_eq!((u.is_directed(), u.ecount()), (false, 1));
    /// ```
    pub fn into_undirected(mut self, mode: ToUndirected) -> Result<Graph> {
        self.to_undirected(mode)?;
        Ok(self)
    }

    /// The Prüfer sequence of a tree.
    ///
    /// A labelled tree on `n ≥ 2` vertices corresponds one-to-one to a
    /// sequence of `n - 2` vertex ids in `0..n` (Cayley's formula `nⁿ⁻²`
    /// follows). The sequence is obtained by repeatedly removing the leaf with
    /// the smallest id and recording its neighbor. Each vertex appears
    /// exactly `degree - 1` times, so leaves never appear.
    ///
    /// The inverse operation is the constructor [`Graph::from_prufer`];
    /// [`Graph::is_tree`] checks the precondition.
    ///
    /// Binds [`igraph_to_prufer`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_to_prufer).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the graph is not a
    /// tree (edge directions are ignored; the null graph is not a tree) or has
    /// fewer than two vertices.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // A star centered at 3: the center appears n - 2 times.
    /// let star = Graph::from_edges(&[(3, 0), (3, 1), (3, 2), (3, 4)], 5, false).unwrap();
    /// assert_eq!(star.to_prufer().unwrap(), vec![3, 3, 3]);
    ///
    /// // Round trip through the constructor.
    /// let back = Graph::from_prufer(&[3, 3, 3]).unwrap();
    /// assert!(back.is_same_graph(&star).unwrap());
    ///
    /// // A cycle is not a tree.
    /// let tri = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, false).unwrap();
    /// assert_eq!(tri.to_prufer().unwrap_err().kind(), ErrorKind::InvalidValue);
    /// ```
    pub fn to_prufer(&self) -> Result<Vec<VertexId>> {
        let mut res = VectorInt::new();
        igraph_call!(igraph_to_prufer(self, &mut res))?;
        Ok(res.into())
    }
}

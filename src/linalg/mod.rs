//! Linear algebra: sparse matrices, eigensolvers (ARPACK, LAPACK), BLAS
//! helpers and spectral embeddings of graphs.
//!
//! This module covers six igraph C headers:
//!
//! | C header              | Rust                                   | What it offers |
//! |-----------------------|----------------------------------------|----------------|
//! | `igraph_sparsemat.h`  | [`SparseMat`], [`SparseLu`], [`SparseQr`], [`SparseMatIter`] | Owned sparse matrices (CXSparse), triplet and column-compressed formats, arithmetic, triangular/LU/QR/Cholesky solvers, ARPACK on sparse matrices |
//! | `igraph_arpack.h`     | [`arpack_rssolve`], [`arpack_rnsolve`], [`ArpackOptions`], [`ArpackStorage`] | Matrix-free eigensolvers driven by a Rust closure computing `y = A x` |
//! | `igraph_eigen.h`      | [`eigen_matrix_symmetric`], [`eigen_matrix`], [`Graph::eigen_adjacency`](crate::Graph::eigen_adjacency) | A unified front-end choosing between LAPACK and ARPACK |
//! | `igraph_lapack.h`     | [`lapack_dgesv`], [`lapack_dgetrf`], [`lapack_dgetrs`], [`lapack_dsyevr`], [`lapack_dgeev`], [`lapack_dgeevx`], [`lapack_dgehrd`] | Dense linear systems and eigenproblems |
//! | `igraph_blas.h`       | [`blas_dgemv`], [`blas_dgemv_array`], [`blas_dgemm`], [`blas_ddot`], [`blas_dnrm2`] | Dense matrix products, dot products and norms |
//! | `igraph_embedding.h`  | [`Graph::adjacency_spectral_embedding`](crate::Graph::adjacency_spectral_embedding), [`Graph::laplacian_spectral_embedding`](crate::Graph::laplacian_spectral_embedding), [`dim_select`] | Spectral embeddings of graphs and dimensionality selection |
//!
//! Dense matrices are the crate's [`Matrix`](crate::matrix::Matrix) (column-major, like
//! igraph and LAPACK); vectors are plain Rust slices and [`Vec`]s. Complex numbers
//! are [`Complex`] (igraph's `igraph_complex_t`, with [`re`](Complex::re) and
//! [`im`](Complex::im) accessors).
//!
//! # Safety net around the C library
//!
//! The C routines wrapped here trust their callers a lot: a wrong vector length
//! reads out of bounds, and an invalid argument handed to the bundled LAPACK
//! or BLAS makes it *terminate the process*. The wrappers therefore validate
//! every dimension, index and range on the Rust side and report problems as
//! [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) errors; a few
//! igraph quirks (present in both igraph 1.0.0 and 1.0.1: the linear algebra
//! sources did not change in 1.0.1) are worked around and documented on the
//! affected functions (e.g. [`SparseMat::transpose`] of non-square triplet matrices,
//! [`SparseMat::max`], [`blas_dgemm`] with `beta != 0`, 2 × 2 problems in
//! [`arpack_rssolve`]). One cannot be worked around: the last ARPACK error
//! is a process-wide C variable, so [`arpack_last_error`] is `unsafe`; use
//! the thread-safe [`ArpackError::from_error`] instead.
//!
//! igraph's ARPACK is not re-entrant (its iteration state lives in
//! thread-local statics): every ARPACK-based function of the crate (those of
//! this module, and eigenvector centrality, hub and authority scores,
//! eigenvector centralization, PageRank with the ARPACK algorithm and
//! leading eigenvector communities) refuses to run nested inside another
//! ARPACK run on the same thread (from its matrix-vector closure, or from a
//! progress or interruption handler called by it), with an
//! [`ErrorKind::Failure`](crate::ErrorKind::Failure) error (see
//! [`arpack_rssolve`]). Also note that ARPACK starts from `A v0`, not from
//! the start vector `v0`: null vectors of singular operators can be missed,
//! shift the operator instead (see [`arpack_rssolve`]).
//!
//! `igraph_arpack_options_get_default` is not wrapped: it hands out a mutable
//! pointer to igraph's shared default options; use
//! [`ArpackOptions::default()`](ArpackOptions) (`igraph_arpack_options_init`)
//! instead.
//!
//! # Related functionality in other modules
//!
//! | Need | Where |
//! |------|-------|
//! | Adjacency matrix of a graph, dense or sparse | [`Graph::get_adjacency`](crate::Graph::get_adjacency), [`Graph::get_adjacency_sparse`](crate::Graph::get_adjacency_sparse) (conversion) |
//! | Laplacian matrix of a graph, dense or as triplets | [`Graph::get_laplacian`](crate::Graph::get_laplacian), [`Graph::get_laplacian_sparse`](crate::Graph::get_laplacian_sparse) (structural) |
//! | Stochastic (random-walk) matrix | [`Graph::get_stochastic`](crate::Graph::get_stochastic), [`Graph::get_stochastic_sparse`](crate::Graph::get_stochastic_sparse) (conversion) |
//! | A graph from a (sparse) adjacency matrix | [`Graph::adjacency`](crate::Graph::adjacency), [`Graph::sparse_adjacency`](crate::Graph::sparse_adjacency), [`Graph::sparse_weighted_adjacency`](crate::Graph::sparse_weighted_adjacency) (constructors) |
//! | Spectral centralities (ARPACK/PRPACK inside) | [`Graph::eigenvector_centrality`](crate::Graph::eigenvector_centrality), [`Graph::hub_and_authority_scores`](crate::Graph::hub_and_authority_scores), [`Graph::pagerank`](crate::Graph::pagerank) (centrality) |
//! | Spectral community detection | [`Graph::community_leading_eigenvector`](crate::Graph::community_leading_eigenvector) (community) |
//! | Spectral layouts | [`Graph::layout_mds`](crate::Graph::layout_mds) (layout) |
//! | Seeding the random start vectors of ARPACK | [`rng::seed`](crate::rng::seed) (per thread) |
//!
//! # Example: the spectrum of a path graph
//!
//! The Laplacian of the path on `n` vertices has eigenvalues
//! `2 - 2 cos(pi k / n)`, `k = 0, ..., n - 1`. Here we take it from the
//! structural module as sparse triplets, load it into a [`SparseMat`],
//! compute its two largest eigenvalues with ARPACK and check the smallest
//! one with a dense LAPACK solver.
//!
//! ```
//! use igraph::linalg::*;
//! use igraph::prelude::*;
//! use igraph::structural::LaplacianNormalization;
//! use std::f64::consts::PI;
//!
//! let n = 10;
//! let path = Graph::ring(n, false, false, false).unwrap();
//! let triplets: Vec<(usize, usize, f64)> = path
//!     .get_laplacian_sparse(NeighborMode::All, LaplacianNormalization::Unnormalized, None)
//!     .unwrap()
//!     .into_iter()
//!     .map(|(i, j, x)| (i as usize, j as usize, x))
//!     .collect();
//! let lap = SparseMat::from_triplets(n, n, &triplets).unwrap().compress().unwrap();
//! assert!(lap.is_symmetric().unwrap());
//!
//! // ARPACK, largest algebraic eigenvalues (the random start vector comes
//! // from this thread's igraph RNG: seed it for reproducible iterations).
//! rng::seed(42).unwrap();
//! let opts = ArpackOptions::default().with_nev(2).with_which(ArpackWhich::LargestAlgebraic);
//! let top = lap.arpack_rssolve(&opts, SparseSolveMethod::Lu).unwrap();
//! for (k, value) in top.values.iter().enumerate() {
//!     let expected = 2.0 - 2.0 * (PI * (n - 1 - k) as f64 / n as f64).cos();
//!     assert!((value - expected).abs() < 1e-8);
//! }
//!
//! // LAPACK on the dense copy: the smallest eigenvalue is 0.
//! let dense = lap.to_dense().unwrap();
//! let low = lapack_dsyevr(&dense, &SymmetricRange::Select(0..1), 1e-12).unwrap();
//! assert!(low.values[0].abs() < 1e-10);
//! ```

mod arpack;
mod blas;
mod eigen;
mod embedding;
mod lapack;
mod sparsemat;

pub use arpack::*;
pub use blas::*;
pub use eigen::*;
pub use embedding::*;
pub use lapack::*;
pub use sparsemat::*;

use crate::{
    error::{Error, Result},
    ffi::*,
};
use std::ffi::c_int;

/// A complex number (`igraph_complex_t`), with [`re`](Complex::re) and
/// [`im`](Complex::im) accessors and [`Complex::new`].
pub type Complex = igraph_complex_t;

/// Converts a dimension to a C `int` (as used by BLAS, LAPACK and ARPACK),
/// failing with an [`ErrorKind::Overflow`](crate::ErrorKind::Overflow)-like
/// invalid value error when it does not fit.
pub(crate) fn to_c_int(value: usize, what: &str) -> Result<c_int> {
    c_int::try_from(value).map_err(|_| Error::invalid(format!("{what} ({value}) is too large")))
}

/// Fails unless every value is finite: NaN and infinite values make the
/// bundled LAPACK (also used inside ARPACK) terminate the process.
pub(crate) fn check_finite(values: &[f64], what: &str) -> Result<()> {
    if values.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "{what} contains NaN or infinite values"
        )))
    }
}

/// Unpacks LAPACK's compressed real representation of complex eigenvectors:
/// a real eigenvalue owns one column, a complex conjugate pair `(j, j+1)`
/// shares columns `j` (real part) and `j+1` (imaginary part).
pub(crate) fn unpack_lapack_vectors(imag: &[f64], m: &crate::matrix::Matrix) -> Vec<Vec<Complex>> {
    let n = m.nrow();
    let mut out = Vec::with_capacity(imag.len());
    let mut j = 0;
    while j < imag.len() {
        if imag[j] == 0.0 || j + 1 >= m.ncol() {
            out.push(m.column(j).iter().map(|&x| Complex::new(x, 0.0)).collect());
            j += 1;
        } else {
            let (re, im) = (m.column(j), m.column(j + 1));
            out.push((0..n).map(|r| Complex::new(re[r], im[r])).collect());
            out.push((0..n).map(|r| Complex::new(re[r], -im[r])).collect());
            j += 2;
        }
    }
    out
}

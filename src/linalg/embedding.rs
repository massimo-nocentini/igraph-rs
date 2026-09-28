//! Spectral embeddings (`igraph_embedding.h`).

use super::{ArpackOptions, check_finite};
use crate::{
    error::{Error, Result},
    ffi::*,
    igraph_call,
    matrix::Matrix,
    vector::Vector,
};
use std::ptr;

crate::ffi_enum! {
    /// Which eigenvalues (singular values for directed graphs) the spectral
    /// embeddings use (the admissible subset of `igraph_eigen_which_position_t`).
    pub enum EmbeddingWhich: igraph_eigen_which_position_t {
        /// Largest magnitude (`IGRAPH_EIGEN_LM`).
        LargestMagnitude = igraph_eigen_which_position_t_IGRAPH_EIGEN_LM,
        /// Largest algebraic value (`IGRAPH_EIGEN_LA`); the same as
        /// `LargestMagnitude` for directed graphs, whose singular values are
        /// non-negative.
        LargestAlgebraic = igraph_eigen_which_position_t_IGRAPH_EIGEN_LA,
        /// Smallest algebraic value (`IGRAPH_EIGEN_SA`).
        SmallestAlgebraic = igraph_eigen_which_position_t_IGRAPH_EIGEN_SA,
    }
}

crate::ffi_enum! {
    /// The Laplacian used by [`Graph::laplacian_spectral_embedding`](crate::Graph::laplacian_spectral_embedding)
    /// (`igraph_laplacian_spectral_embedding_type_t`). `D` is the degree
    /// (strength) matrix, `A` the adjacency matrix.
    pub enum LaplacianEmbeddingType: igraph_laplacian_spectral_embedding_type_t {
        /// `D - A`, the combinatorial Laplacian; undirected graphs only.
        DA = igraph_laplacian_spectral_embedding_type_t_IGRAPH_EMBEDDING_D_A,
        /// `I - D^-1/2 A D^-1/2`, the symmetric normalized Laplacian;
        /// undirected graphs only.
        IDAD = igraph_laplacian_spectral_embedding_type_t_IGRAPH_EMBEDDING_I_DAD,
        /// `D^-1/2 A D^-1/2`; undirected graphs only.
        DAD = igraph_laplacian_spectral_embedding_type_t_IGRAPH_EMBEDDING_DAD,
        /// `O^-1/2 A P^-1/2` with the out- and in-degree matrices `O` and
        /// `P`; the only choice for directed graphs.
        OAP = igraph_laplacian_spectral_embedding_type_t_IGRAPH_EMBEDDING_OAP,
    }
}

/// A spectral embedding of the vertices of a graph.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralEmbedding {
    /// The latent positions `X` (or the singular vectors `U` when not
    /// scaled): one row per vertex, one column per dimension.
    pub x: Matrix,
    /// The second half of the latent positions `Y` (or `V`), for directed
    /// graphs; equal to [`x`](Self::x) for undirected graphs.
    pub y: Matrix,
    /// The eigenvalues (undirected) or singular values (directed) used; empty
    /// for graphs without edges.
    pub d: Vec<f64>,
}

fn check_common(graph: &igraph_t, no: usize, weights: Option<&[f64]>) -> Result<()> {
    let n = graph.vcount();
    if no == 0 || no > n {
        return Err(Error::invalid(format!(
            "cannot compute a {no}-dimensional embedding of {n} vertices"
        )));
    }
    if let Some(w) = weights {
        if w.len() != graph.ecount() {
            return Err(Error::invalid(
                "the weight vector length must match the number of edges",
            ));
        }
        check_finite(w, "the weight vector")?;
    }
    Ok(())
}

/// The normalized Laplacians divide by square roots of the vertex
/// strengths: a zero (or negative) strength would feed NaN or infinite
/// values to ARPACK, which then aborts the process.
fn check_strengths(
    graph: &igraph_t,
    weights: Option<&[f64]>,
    kind: LaplacianEmbeddingType,
) -> Result<()> {
    if kind == LaplacianEmbeddingType::DA {
        return Ok(());
    }
    let n = graph.vcount();
    let (mut out, mut inn) = (vec![0.0; n], vec![0.0; n]);
    for (e, (a, b)) in graph.edge_list().into_iter().enumerate() {
        let w = weights.map_or(1.0, |w| w[e]);
        out[a as usize] += w;
        inn[b as usize] += w;
    }
    let ok = if graph.is_directed() {
        out.iter().chain(&inn).all(|&s| s > 0.0)
    } else {
        out.iter().zip(&inn).all(|(o, i)| o + i > 0.0)
    };
    if ok {
        Ok(())
    } else {
        Err(Error::invalid(
            "normalized Laplacian embeddings need positive vertex strengths (no isolated vertices; for directed graphs, positive in- and out-strengths)",
        ))
    }
}

impl igraph_t {
    /// Adjacency spectral embedding (`igraph_adjacency_spectral_embedding`).
    ///
    /// Computes a `no`-dimensional Euclidean representation of the graph from
    /// the singular value decomposition of its adjacency matrix,
    /// `A = U D V'`. For undirected graphs `X = U_no D^(1/2)` (and `Y = X`);
    /// for directed graphs `X = U_no D^(1/2)` and `Y = V_no D^(1/2)`. If the
    /// graph is a random dot product graph generated from latent position
    /// vectors in `R^no`, the embedding estimates these latent positions.
    ///
    /// - `weights`: optional edge weights.
    /// - `which`: which eigenvalues (singular values) to use.
    /// - `scaled`: return `X`, `Y` if true, `U`, `V` otherwise.
    /// - `cvec`: added to the diagonal of the adjacency matrix before the
    ///   decomposition; either one value per vertex, a single value for all,
    ///   or `None` for zero. A common choice is half the degrees.
    /// - `options`: ARPACK options; only `tol` and `mxiter` are used (igraph
    ///   sets `which`, `nev` and `ncv`, and starts from a random vector of the
    ///   calling thread's RNG).
    ///
    /// Binds [`igraph_adjacency_spectral_embedding`](https://igraph.org/c/html/latest/igraph-Embedding.html#igraph_adjacency_spectral_embedding).
    ///
    /// See also [`dim_select`] to choose `no` from the
    /// singular values, [`Graph::get_adjacency`](crate::Graph::get_adjacency)
    /// for the matrix being decomposed, and
    /// [`Graph::layout_mds`](crate::Graph::layout_mds) for a distance-based
    /// spectral layout.
    ///
    /// # Errors
    /// If `no` is zero or larger than the number of vertices, or the weight
    /// or `cvec` lengths are wrong.
    ///
    /// # Examples
    ///
    /// Two disjoint 4-cliques: the 2-dimensional embedding places the two
    /// groups on orthogonal axes.
    ///
    /// ```
    /// use igraph::{linalg::*, prelude::*};
    /// let k4 = Graph::full(4, false, false).unwrap();
    /// let g = k4.disjoint_union(&k4).unwrap(); // vertices 0..4 and 4..8
    /// let e = g
    ///     .adjacency_spectral_embedding(2, None, EmbeddingWhich::LargestAlgebraic, true, None, &ArpackOptions::default())
    ///     .unwrap();
    /// assert!((e.d[0] - 3.0).abs() < 1e-8 && (e.d[1] - 3.0).abs() < 1e-8);
    /// let dot = |a: usize, b: usize| e.x.row(a).iter().zip(e.x.row(b)).map(|(p, q)| p * q).sum::<f64>();
    /// assert!(dot(0, 1) > 0.5);      // same clique: similar positions
    /// assert!(dot(0, 5).abs() < 1e-8); // different cliques: orthogonal
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn adjacency_spectral_embedding(
        &self,
        no: usize,
        weights: Option<&[f64]>,
        which: EmbeddingWhich,
        scaled: bool,
        cvec: Option<&[f64]>,
        options: &ArpackOptions,
    ) -> Result<SpectralEmbedding> {
        check_common(self, no, weights)?;
        let n = self.vcount();
        // igraph accepts a single-element `cvec` but then reads it as if it
        // had one element per vertex: always expand it.
        let cvec: Vec<f64> = match cvec {
            None => vec![0.0; n],
            Some([c]) => vec![*c; n],
            Some(c) if c.len() == n => c.to_vec(),
            Some(_) => {
                return Err(Error::invalid(
                    "cvec must have one element per vertex, or a single element",
                ));
            }
        };
        check_finite(&cvec, "cvec")?;
        let w = weights.map(Vector::view);
        let wp = w.as_ref().map_or(ptr::null(), |v| v.as_ptr());
        let cv = Vector::view(&cvec);
        let mut raw_opts = options.to_raw_unchecked_which(n)?;
        let (mut x, mut y, mut d) = (Matrix::new(), Matrix::new(), Vector::new());
        let _arpack = super::arpack::ArpackGuard::enter()?;
        igraph_call!(igraph_adjacency_spectral_embedding(
            self,
            no as igraph_int_t,
            wp,
            which.into(),
            scaled,
            &mut x,
            &mut y,
            &mut d,
            cv.as_ptr(),
            &mut raw_opts
        ))?;
        Ok(SpectralEmbedding { x, y, d: d.into() })
    }

    /// Laplacian spectral embedding (`igraph_laplacian_spectral_embedding`).
    ///
    /// Like [`adjacency_spectral_embedding`](Self::adjacency_spectral_embedding),
    /// but decomposes a Laplacian of the graph, chosen with `kind` (see
    /// [`LaplacianEmbeddingType`]; directed graphs need
    /// [`LaplacianEmbeddingType::OAP`]). Use
    /// [`EmbeddingWhich::SmallestAlgebraic`] with `D - A` for the classic
    /// spectral clustering / Fiedler vector embedding.
    ///
    /// Binds [`igraph_laplacian_spectral_embedding`](https://igraph.org/c/html/latest/igraph-Embedding.html#igraph_laplacian_spectral_embedding).
    ///
    /// The eigenvectors come from ARPACK, started from a random vector of
    /// the calling thread's igraph RNG (seed it with
    /// [`rng::seed`](crate::rng::seed) for reproducible runs). On tiny graphs
    /// with repeated eigenvalues ARPACK may miss an eigenvalue or fail to
    /// converge for some start vectors; there, diagonalize the dense
    /// Laplacian with [`lapack_dsyevr`](super::lapack_dsyevr) instead.
    ///
    /// See also [`Graph::get_laplacian`](crate::Graph::get_laplacian) and
    /// [`Graph::get_laplacian_sparse`](crate::Graph::get_laplacian_sparse)
    /// (structural module) for the Laplacian matrices themselves; `D - A` is
    /// [`LaplacianNormalization::Unnormalized`](crate::structural::LaplacianNormalization::Unnormalized)
    /// and `I - D^-1/2 A D^-1/2` is
    /// [`LaplacianNormalization::Symmetric`](crate::structural::LaplacianNormalization::Symmetric).
    ///
    /// # Errors
    /// As for the adjacency embedding, plus an invalid `kind` for the
    /// directedness of the graph, and — for the normalized Laplacians — a
    /// vertex with zero strength (isolated vertex; for directed graphs, zero
    /// in- or out-strength).
    ///
    /// # Examples
    ///
    /// Spectral bisection of Zachary's karate club: the Fiedler vector
    /// (eigenvalue `~0.4685` of `D - A`, the algebraic connectivity) puts the
    /// instructor (vertex 0) and the administrator (vertex 33) on opposite
    /// sides.
    ///
    /// ```
    /// use igraph::{linalg::*, prelude::*};
    /// let g = Graph::famous("Zachary").unwrap();
    /// let e = g
    ///     .laplacian_spectral_embedding(
    ///         2,
    ///         None,
    ///         EmbeddingWhich::SmallestAlgebraic,
    ///         LaplacianEmbeddingType::DA,
    ///         false,
    ///         &ArpackOptions::default(),
    ///     )
    ///     .unwrap();
    /// // One eigenvalue is 0 (connected graph), the other is the algebraic connectivity.
    /// let fiedler = if e.d[0] > e.d[1] { 0 } else { 1 };
    /// assert!(e.d[1 - fiedler].abs() < 1e-8);
    /// assert!((e.d[fiedler] - 0.4685).abs() < 1e-4);
    /// let f = e.x.column(fiedler);
    /// assert!(f[0] * f[33] < 0.0);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn laplacian_spectral_embedding(
        &self,
        no: usize,
        weights: Option<&[f64]>,
        which: EmbeddingWhich,
        kind: LaplacianEmbeddingType,
        scaled: bool,
        options: &ArpackOptions,
    ) -> Result<SpectralEmbedding> {
        check_common(self, no, weights)?;
        check_strengths(self, weights, kind)?;
        let w = weights.map(Vector::view);
        let wp = w.as_ref().map_or(ptr::null(), |v| v.as_ptr());
        let mut raw_opts = options.to_raw_unchecked_which(self.vcount())?;
        let (mut x, mut y, mut d) = (Matrix::new(), Matrix::new(), Vector::new());
        let _arpack = super::arpack::ArpackGuard::enter()?;
        igraph_call!(igraph_laplacian_spectral_embedding(
            self,
            no as igraph_int_t,
            wp,
            which.into(),
            kind.into(),
            scaled,
            &mut x,
            &mut y,
            &mut d,
            &mut raw_opts
        ))?;
        Ok(SpectralEmbedding { x, y, d: d.into() })
    }
}

/// Dimensionality selection by profile likelihood (`igraph_dim_select`).
///
/// Given the (decreasingly) ordered "importance" values of the dimensions —
/// typically the singular values of an adjacency spectral embedding — it
/// models them as a mixture of two Gaussians with different means and equal
/// variance, and returns the number `d` of leading values that maximizes the
/// likelihood when the first `d` values are assigned to one component and
/// the rest to the other. Also usable for any "where is the gap" problem.
/// Time complexity: O(n).
///
/// Binds [`igraph_dim_select`](https://igraph.org/c/html/latest/igraph-Embedding.html#igraph_dim_select).
///
/// When no profile likelihood is a finite number — e.g. when all the values
/// are equal, so that every one of them is NaN — igraph 1.0.0 and 1.0.1
/// leave the result unset; this wrapper then returns `sv.len()` (all values
/// in one group).
///
/// # Errors
/// If `sv` is empty or contains NaN or infinite values.
///
/// # Examples
///
/// ```
/// use igraph::linalg::dim_select;
/// assert_eq!(dim_select(&[10.0, 9.8, 9.9, 1.0, 1.1, 0.9, 1.0]).unwrap(), 3);
/// assert_eq!(dim_select(&[5.0]).unwrap(), 1);
/// // No gap at all: everything is one group.
/// assert_eq!(dim_select(&[2.0, 2.0, 2.0]).unwrap(), 3);
/// ```
pub fn dim_select(sv: &[f64]) -> Result<usize> {
    if sv.is_empty() {
        return Err(Error::invalid(
            "dimensionality selection needs at least one value",
        ));
    }
    check_finite(sv, "the values")?;
    let v = Vector::view(sv);
    let mut dim: igraph_int_t = 0;
    igraph_call!(igraph_dim_select(v.as_ptr(), &mut dim))?;
    Ok(if dim >= 1 { dim as usize } else { sv.len() })
}

//! The LAPACK interface (`igraph_lapack.h`).
//!
//! Invalid arguments make the LAPACK bundled with igraph terminate the
//! process, so all the dimensions and ranges are validated here first.

use super::{Complex, check_finite, to_c_int, unpack_lapack_vectors};
use crate::{
    error::{Error, ErrorKind, Result},
    ffi::*,
    igraph_call,
    matrix::Matrix,
    vector::{Vector, VectorInt},
};
use std::{ffi::c_int, ops::Range, ptr};

crate::ffi_enum! {
    /// Balancing performed by [`lapack_dgeevx`] (`igraph_lapack_dgeevx_balance_t`).
    pub enum DgeevxBalance: igraph_lapack_dgeevx_balance_t {
        /// Neither permute nor scale.
        None = igraph_lapack_dgeevx_balance_t_IGRAPH_LAPACK_DGEEVX_BALANCE_NONE,
        /// Permute rows and columns to make the matrix more nearly upper
        /// triangular; do not scale.
        Perm = igraph_lapack_dgeevx_balance_t_IGRAPH_LAPACK_DGEEVX_BALANCE_PERM,
        /// Diagonally scale (`D A D^-1`) to make rows and columns closer in
        /// norm; do not permute.
        Scale = igraph_lapack_dgeevx_balance_t_IGRAPH_LAPACK_DGEEVX_BALANCE_SCALE,
        /// Both permute and scale.
        Both = igraph_lapack_dgeevx_balance_t_IGRAPH_LAPACK_DGEEVX_BALANCE_BOTH,
    }
}

fn square(a: &Matrix, what: &str) -> Result<c_int> {
    if a.nrow() != a.ncol() {
        return Err(Error::invalid(format!(
            "{what} needs a square matrix, got {:?}",
            a.shape()
        )));
    }
    if a.nrow() == 0 {
        return Err(Error::invalid(format!("{what} needs a non-empty matrix")));
    }
    check_finite(a.as_slice(), "the matrix")?;
    to_c_int(a.nrow(), "matrix order")
}

/// An LU factorization `A = P L U` computed by [`lapack_dgetrf`].
#[derive(Debug, Clone, PartialEq)]
pub struct LuFactors {
    /// `L` (below the diagonal, unit diagonal not stored) and `U` (on and
    /// above the diagonal), packed in one matrix of the shape of `A`.
    pub lu: Matrix,
    /// Pivot indices, **one-based** as in LAPACK: row `i` was interchanged
    /// with row `ipiv[i] - 1`.
    pub ipiv: Vec<i64>,
    /// LAPACK's `info`: 0 on success, `i > 0` if `U(i-1, i-1)` is exactly
    /// zero (the factorization is complete, but `U` is singular).
    pub info: i32,
}

impl LuFactors {
    /// Whether `U` is exactly singular.
    pub fn is_singular(&self) -> bool {
        self.info > 0
    }

    /// Solves `A X = B` (or `A' X = B` if `transpose`) with these factors,
    /// see [`lapack_dgetrs`].
    pub fn solve(&self, transpose: bool, b: &Matrix) -> Result<Matrix> {
        lapack_dgetrs(transpose, &self.lu, &self.ipiv, b)
    }
}

/// LU factorization with partial pivoting of a general `m` × `n` matrix,
/// `A = P L U` (`igraph_lapack_dgetrf`), where `L` is lower triangular
/// (trapezoidal if `m > n`) with unit diagonal and `U` is upper triangular
/// (trapezoidal if `m < n`).
///
/// A singular matrix is not an error: check [`LuFactors::info`].
///
/// Binds [`igraph_lapack_dgetrf`](https://igraph.org/c/html/latest/igraph-Linalg.html#igraph_lapack_dgetrf).
pub fn lapack_dgetrf(a: &Matrix) -> Result<LuFactors> {
    to_c_int(a.nrow(), "number of rows")?;
    to_c_int(a.ncol(), "number of columns")?;
    let mut lu = a.clone();
    let mut ipiv = VectorInt::new();
    let mut info: c_int = 0;
    igraph_call!(igraph_lapack_dgetrf(&mut lu, &mut ipiv, &mut info))?;
    Ok(LuFactors {
        lu,
        ipiv: ipiv.into(),
        info,
    })
}

/// Solves `A X = B` or `A' X = B` (`transpose`) using the LU factors of a
/// square `A` computed by [`lapack_dgetrf`] (`igraph_lapack_dgetrs`). No
/// check is made for singularity.
///
/// Binds [`igraph_lapack_dgetrs`](https://igraph.org/c/html/latest/igraph-Linalg.html#igraph_lapack_dgetrs).
///
/// # Errors
/// If `lu` is not square, `b` has the wrong number of rows, or the pivots
/// are out of range.
pub fn lapack_dgetrs(transpose: bool, lu: &Matrix, ipiv: &[i64], b: &Matrix) -> Result<Matrix> {
    if lu.nrow() != lu.ncol() {
        return Err(Error::invalid("dgetrs needs a square LU matrix"));
    }
    let n = lu.nrow();
    to_c_int(n, "matrix order")?;
    to_c_int(b.ncol(), "number of right hand sides")?;
    if b.nrow() != n {
        return Err(Error::invalid(
            "the right hand side has the wrong number of rows",
        ));
    }
    if ipiv.len() != n || ipiv.iter().any(|&p| p < 1 || p as usize > n) {
        return Err(Error::invalid("invalid pivot vector"));
    }
    let piv = VectorInt::view(ipiv);
    let mut x = b.clone();
    igraph_call!(igraph_lapack_dgetrs(transpose, lu, piv.as_ptr(), &mut x))?;
    Ok(x)
}

/// The solution of a linear system computed by [`lapack_dgesv`].
#[derive(Debug, Clone, PartialEq)]
pub struct DgesvResult {
    /// The solution `X` of `A X = B` (same shape as `B`).
    pub solution: Matrix,
    /// The LU factors of `A`, see [`LuFactors`].
    pub factors: LuFactors,
}

/// Solves the linear system `A X = B` for a square `A` and one or more
/// right hand sides (the columns of `B`), by LU decomposition with partial
/// pivoting (`igraph_lapack_dgesv`).
///
/// Binds [`igraph_lapack_dgesv`](https://igraph.org/c/html/latest/igraph-Linalg.html#igraph_lapack_dgesv).
///
/// # Errors
/// If the dimensions do not match, or `A` is exactly singular
/// ([`ErrorKind::Failure`](crate::ErrorKind::Failure)).
///
/// # Examples
///
/// ```
/// use igraph::{linalg::lapack_dgesv, prelude::*};
/// // 2x + y = 5, x + 3y = 10  =>  x = 1, y = 3
/// let a = Matrix::from_rows(&[[2.0, 1.0], [1.0, 3.0]]).unwrap();
/// let b = Matrix::from_rows(&[[5.0], [10.0]]).unwrap();
/// let x = lapack_dgesv(&a, &b).unwrap().solution;
/// assert!((x[(0, 0)] - 1.0).abs() < 1e-12 && (x[(1, 0)] - 3.0).abs() < 1e-12);
/// ```
pub fn lapack_dgesv(a: &Matrix, b: &Matrix) -> Result<DgesvResult> {
    if a.nrow() != a.ncol() {
        return Err(Error::invalid("dgesv needs a square coefficient matrix"));
    }
    to_c_int(a.nrow(), "matrix order")?;
    to_c_int(b.ncol(), "number of right hand sides")?;
    if b.nrow() != a.nrow() {
        return Err(Error::invalid(
            "the right hand side has the wrong number of rows",
        ));
    }
    let mut lu = a.clone();
    let mut x = b.clone();
    let mut ipiv = VectorInt::new();
    let mut info: c_int = 0;
    igraph_call!(igraph_lapack_dgesv(&mut lu, &mut ipiv, &mut x, &mut info))?;
    if info > 0 {
        return Err(Error::new(
            ErrorKind::Failure,
            format!("the matrix is exactly singular (U({0},{0}) = 0)", info - 1),
        ));
    }
    Ok(DgesvResult {
        solution: x,
        factors: LuFactors {
            lu,
            ipiv: ipiv.into(),
            info,
        },
    })
}

/// Solves `A x = b` for a single right hand side (via [`lapack_dgesv`]).
///
/// ```
/// use igraph::{linalg::solve, prelude::*};
/// let a = Matrix::from_rows(&[[4.0, -2.0], [1.0, 1.0]]).unwrap();
/// let x = solve(&a, &[2.0, 3.0]).unwrap();
/// assert!((x[0] - 4.0 / 3.0).abs() < 1e-12 && (x[1] - 5.0 / 3.0).abs() < 1e-12);
/// ```
pub fn solve(a: &Matrix, b: &[f64]) -> Result<Vec<f64>> {
    let bm = Matrix::from_column_major(b.len(), 1, b)?;
    Ok(lapack_dgesv(a, &bm)?.solution.as_slice().to_vec())
}

/// Which eigenvalues [`lapack_dsyevr`] computes (`igraph_lapack_dsyev_which_t`
/// plus its parameters).
#[derive(Debug, Clone, PartialEq)]
pub enum SymmetricRange {
    /// All the eigenvalues (`IGRAPH_LAPACK_DSYEV_ALL`).
    All,
    /// The eigenvalues in the half-open interval `(low, high]`
    /// (`IGRAPH_LAPACK_DSYEV_INTERVAL`).
    Interval {
        /// Exclusive lower bound.
        low: f64,
        /// Inclusive upper bound.
        high: f64,
    },
    /// The eigenvalues with the given zero-based positions in increasing
    /// order (`IGRAPH_LAPACK_DSYEV_SELECT` with one-based `il = start + 1`,
    /// `iu = end`).
    Select(Range<usize>),
}

/// Eigenvalues and eigenvectors computed by [`lapack_dsyevr`].
#[derive(Debug, Clone, PartialEq)]
pub struct DsyevrResult {
    /// The selected eigenvalues, in increasing order.
    pub values: Vec<f64>,
    /// The orthonormal eigenvectors, in the columns.
    pub vectors: Matrix,
}

/// Selected eigenvalues and eigenvectors of a real symmetric matrix, with
/// LAPACK's relatively robust representations algorithm
/// (`igraph_lapack_dsyevr`). Only the upper triangle of `a` is used.
///
/// `abstol` is the absolute error tolerance for the eigenvalues: an
/// approximate eigenvalue is accepted when it lies in an interval `[a, b]`
/// of width at most `abstol + eps * max(|a|, |b|)`.
///
/// Binds [`igraph_lapack_dsyevr`](https://igraph.org/c/html/latest/igraph-Linalg.html#igraph_lapack_dsyevr).
/// The eigenvector support output of the C function is not exposed.
///
/// # Examples
///
/// This is igraph's `igraph_lapack_dsyevr.c` example: the matrix
/// `[[2, -1], [-1, 3]]` has eigenvalues `(5 ± √5) / 2`.
///
/// ```
/// use igraph::{linalg::*, prelude::*};
/// let a = Matrix::from_rows(&[[2.0, -1.0], [-1.0, 3.0]]).unwrap();
/// let low = lapack_dsyevr(&a, &SymmetricRange::Select(0..1), 1e-10).unwrap();
/// assert!((low.values[0] - 1.381966).abs() < 1e-6);
/// let high = lapack_dsyevr(&a, &SymmetricRange::Interval { low: 3.0, high: 4.0 }, 1e-10).unwrap();
/// assert!((high.values[0] - 3.618034).abs() < 1e-6);
/// assert_eq!(high.vectors.shape(), (2, 1));
/// ```
pub fn lapack_dsyevr(a: &Matrix, range: &SymmetricRange, abstol: f64) -> Result<DsyevrResult> {
    let n = square(a, "dsyevr")?;
    let (which, vl, vu, vestimate, il, iu) = match range {
        SymmetricRange::All => (
            igraph_lapack_dsyev_which_t_IGRAPH_LAPACK_DSYEV_ALL,
            0.0,
            0.0,
            0,
            0,
            0,
        ),
        SymmetricRange::Interval { low, high } => {
            if low.is_nan() || high.is_nan() || low >= high {
                return Err(Error::invalid(
                    "the eigenvalue interval must satisfy low < high",
                ));
            }
            // `n` is always a correct upper bound on the number of eigenvalues.
            (
                igraph_lapack_dsyev_which_t_IGRAPH_LAPACK_DSYEV_INTERVAL,
                *low,
                *high,
                n,
                0,
                0,
            )
        }
        SymmetricRange::Select(r) => {
            if r.start >= r.end || r.end > n as usize {
                return Err(Error::invalid(format!(
                    "invalid eigenvalue range {r:?} for order {n}"
                )));
            }
            (
                igraph_lapack_dsyev_which_t_IGRAPH_LAPACK_DSYEV_SELECT,
                0.0,
                0.0,
                0,
                r.start as c_int + 1,
                r.end as c_int,
            )
        }
    };
    let mut values = Vector::new();
    let mut vectors = Matrix::new();
    igraph_call!(igraph_lapack_dsyevr(
        a,
        which,
        vl,
        vu,
        vestimate,
        il,
        iu,
        abstol,
        &mut values,
        &mut vectors,
        ptr::null_mut()
    ))?;
    Ok(DsyevrResult {
        values: values.into(),
        vectors,
    })
}

/// Eigenvalues and eigenvectors of a general real matrix computed by
/// [`lapack_dgeev`], in LAPACK's packed real format.
#[derive(Debug, Clone, PartialEq)]
pub struct DgeevResult {
    /// Real parts of the eigenvalues.
    pub values_real: Vec<f64>,
    /// Imaginary parts of the eigenvalues; complex conjugate pairs appear
    /// consecutively, the one with positive imaginary part first.
    pub values_imag: Vec<f64>,
    /// Left eigenvectors `u` (`u^H A = lambda u^H`), if requested, packed:
    /// a real eigenvalue `j` owns column `j`; a complex pair `(j, j+1)` has
    /// its eigenvectors `u_j = col_j + i col_{j+1}` and `u_{j+1} = conj(u_j)`.
    pub vectors_left: Option<Matrix>,
    /// Right eigenvectors `v` (`A v = lambda v`), if requested, packed like
    /// [`vectors_left`](Self::vectors_left).
    pub vectors_right: Option<Matrix>,
}

impl DgeevResult {
    /// The eigenvalues as complex numbers.
    pub fn values(&self) -> Vec<Complex> {
        self.values_real
            .iter()
            .zip(&self.values_imag)
            .map(|(&re, &im)| Complex::new(re, im))
            .collect()
    }

    /// The unpacked right eigenvectors, `result[k]` belonging to eigenvalue `k`.
    pub fn right_eigenvectors(&self) -> Option<Vec<Vec<Complex>>> {
        self.vectors_right
            .as_ref()
            .map(|m| unpack_lapack_vectors(&self.values_imag, m))
    }

    /// The unpacked left eigenvectors, `result[k]` belonging to eigenvalue `k`.
    pub fn left_eigenvectors(&self) -> Option<Vec<Vec<Complex>>> {
        self.vectors_left
            .as_ref()
            .map(|m| unpack_lapack_vectors(&self.values_imag, m))
    }
}

/// Eigenvalues and, optionally, left and/or right eigenvectors of a general
/// real square matrix (`igraph_lapack_dgeev`). The eigenvectors are
/// normalized to unit Euclidean norm with largest component real.
///
/// Binds [`igraph_lapack_dgeev`](https://igraph.org/c/html/latest/igraph-Linalg.html#igraph_lapack_dgeev).
///
/// # Errors
/// If the matrix is not square or empty, or the QR algorithm fails.
///
/// # Examples
///
/// igraph's `igraph_lapack_dgeev.c` example: `[[1, 1], [-1, 1]]` has
/// eigenvalues `1 ± i`.
///
/// ```
/// use igraph::{linalg::lapack_dgeev, prelude::*};
/// let a = Matrix::from_rows(&[[1.0, 1.0], [-1.0, 1.0]]).unwrap();
/// let e = lapack_dgeev(&a, true, true).unwrap();
/// let values = e.values();
/// assert!((values[0].re() - 1.0).abs() < 1e-12 && (values[0].im() - 1.0).abs() < 1e-12);
/// assert!((values[1].re() - 1.0).abs() < 1e-12 && (values[1].im() + 1.0).abs() < 1e-12);
/// // Check A v = (1 + i) v on the first component: (A v)_0 = v_0 + v_1.
/// let v = &e.right_eigenvectors().unwrap()[0];
/// let (lhs_re, lhs_im) = (v[0].re() + v[1].re(), v[0].im() + v[1].im());
/// let (rhs_re, rhs_im) = (v[0].re() - v[0].im(), v[0].re() + v[0].im());
/// assert!((lhs_re - rhs_re).abs() < 1e-12 && (lhs_im - rhs_im).abs() < 1e-12);
/// ```
pub fn lapack_dgeev(a: &Matrix, left: bool, right: bool) -> Result<DgeevResult> {
    square(a, "dgeev")?;
    let (mut re, mut im) = (Vector::new(), Vector::new());
    let mut vl = Matrix::new();
    let mut vr = Matrix::new();
    let vlp = if left {
        &mut vl as *mut Matrix
    } else {
        ptr::null_mut()
    };
    let vrp = if right {
        &mut vr as *mut Matrix
    } else {
        ptr::null_mut()
    };
    // Non-zero on entry: report failures of the QR algorithm as errors.
    let mut info: c_int = 1;
    igraph_call!(igraph_lapack_dgeev(
        a, &mut re, &mut im, vlp, vrp, &mut info
    ))?;
    Ok(DgeevResult {
        values_real: re.into(),
        values_imag: im.into(),
        vectors_left: left.then_some(vl),
        vectors_right: right.then_some(vr),
    })
}

/// Output of [`lapack_dgeevx`].
#[derive(Debug, Clone, PartialEq)]
pub struct DgeevxResult {
    /// Eigenvalues and (packed) left and right eigenvectors, see [`DgeevResult`].
    pub eigen: DgeevResult,
    /// `ilo` of the balancing (one-based): the balanced `A(i, j) = 0` if
    /// `i > j` and `j < ilo` or `i > ihi`.
    pub ilo: i32,
    /// `ihi` of the balancing (one-based).
    pub ihi: i32,
    /// Details of the permutations and scaling factors of the balancing.
    pub scale: Vec<f64>,
    /// The one-norm of the balanced matrix (maximum absolute column sum).
    pub abnrm: f64,
    /// Reciprocal condition numbers of the eigenvalues.
    pub rconde: Vec<f64>,
}

/// Eigenvalues and left and right eigenvectors of a general real matrix,
/// "expert" version (`igraph_lapack_dgeevx`): it also balances the matrix
/// (see [`DgeevxBalance`]) and computes reciprocal condition numbers of the
/// eigenvalues.
///
/// Binds [`igraph_lapack_dgeevx`](https://igraph.org/c/html/latest/igraph-Linalg.html#igraph_lapack_dgeevx).
/// The reciprocal condition numbers of the eigenvectors (`rcondv`) are not
/// computed: igraph 1.0.0 and 1.0.1 allocate too small an integer workspace for them.
///
/// ```
/// use igraph::{linalg::*, prelude::*};
/// let a = Matrix::from_rows(&[[1.0, 1e4], [1e-4, 1.0]]).unwrap();
/// let r = lapack_dgeevx(DgeevxBalance::Both, &a).unwrap();
/// let mut re = r.eigen.values_real.clone();
/// re.sort_by(f64::total_cmp);
/// assert!((re[0] - 0.0).abs() < 1e-9 && (re[1] - 2.0).abs() < 1e-9);
/// assert!(r.rconde.iter().all(|&c| c > 0.0 && c <= 1.0 + 1e-12));
/// ```
pub fn lapack_dgeevx(balance: DgeevxBalance, a: &Matrix) -> Result<DgeevxResult> {
    let n = square(a, "dgeevx")? as usize;
    let (mut re, mut im) = (Vector::new(), Vector::new());
    let (mut vl, mut vr) = (Matrix::new(), Matrix::new());
    let (mut ilo, mut ihi): (c_int, c_int) = (0, 0);
    let mut scale = Vector::new();
    let mut abnrm = 0.0;
    // igraph does not resize `rconde` before LAPACK writes `n` values into it.
    let mut rconde = Vector::zeros(n);
    let mut info: c_int = 1;
    igraph_call!(igraph_lapack_dgeevx(
        balance.into(),
        a,
        &mut re,
        &mut im,
        &mut vl,
        &mut vr,
        &mut ilo,
        &mut ihi,
        &mut scale,
        &mut abnrm,
        &mut rconde,
        ptr::null_mut(),
        &mut info
    ))?;
    Ok(DgeevxResult {
        eigen: DgeevResult {
            values_real: re.into(),
            values_imag: im.into(),
            vectors_left: Some(vl),
            vectors_right: Some(vr),
        },
        ilo,
        ihi,
        scale: scale.into(),
        abnrm,
        rconde: rconde.into(),
    })
}

/// Reduces a general square matrix to upper Hessenberg form by an
/// orthogonal similarity transformation (`igraph_lapack_dgehrd`): the
/// result `H = Q' A Q` has zeros below the first subdiagonal and the same
/// eigenvalues as `A`.
///
/// `ilo` and `ihi` are one-based, `1 <= ilo <= ihi <= n`; use `1` and `n`
/// unless `A` is already upper triangular in rows and columns outside
/// `ilo..=ihi` (e.g. after balancing with [`lapack_dgeevx`]).
///
/// Binds [`igraph_lapack_dgehrd`](https://igraph.org/c/html/latest/igraph-Linalg.html#igraph_lapack_dgehrd).
///
/// ```
/// use igraph::{linalg::lapack_dgehrd, prelude::*};
/// let a = Matrix::from_rows(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]).unwrap();
/// let h = lapack_dgehrd(&a, 1, 3).unwrap();
/// assert_eq!(h[(2, 0)], 0.0);
/// // The trace is preserved by similarity transformations.
/// assert!((h[(0, 0)] + h[(1, 1)] + h[(2, 2)] - 16.0).abs() < 1e-10);
/// ```
pub fn lapack_dgehrd(a: &Matrix, ilo: usize, ihi: usize) -> Result<Matrix> {
    let n = square(a, "dgehrd")? as usize;
    if ilo < 1 || ihi > n || ilo > ihi {
        return Err(Error::invalid(format!(
            "invalid ilo = {ilo} and ihi = {ihi} for order {n}"
        )));
    }
    let mut res = Matrix::new();
    igraph_call!(igraph_lapack_dgehrd(
        a,
        ilo as c_int,
        ihi as c_int,
        &mut res
    ))?;
    Ok(res)
}

//! Generic eigensolvers (`igraph_eigen.h`).

use super::{
    ArpackOptions, Complex, SparseMat,
    arpack::{matvec_trampoline, with_finite_guard},
    check_finite,
    lapack::DgeevxBalance,
    to_c_int,
};
use crate::{
    error::{Error, Result},
    ffi::*,
    igraph_call,
    matrix::{Matrix, MatrixComplex},
    vector::{Vector, VectorComplex},
};
use std::{ffi::c_void, ops::Range, ptr};

crate::ffi_enum! {
    /// Which numerical library [`eigen_matrix_symmetric`], [`eigen_matrix`]
    /// and [`Graph::eigen_adjacency`](crate::Graph::eigen_adjacency) use (`igraph_eigen_algorithm_t`).
    pub enum EigenAlgorithm: igraph_eigen_algorithm_t {
        /// Choose automatically. For symmetric matrices: LAPACK when the matrix
        /// is small (`n < 100`), when as many eigenvalues as the order are
        /// requested, and for [`EigenWhich::All`], [`EigenWhich::Interval`]
        /// and [`EigenWhich::Select`] (which only LAPACK supports); ARPACK
        /// otherwise. ARPACK for [`Graph::eigen_adjacency`](crate::Graph::eigen_adjacency).
        /// Not implemented by igraph (1.0.0 and 1.0.1) for [`eigen_matrix`].
        Auto = igraph_eigen_algorithm_t_IGRAPH_EIGEN_AUTO,
        /// Dense LAPACK routines (the matrix is formed explicitly).
        Lapack = igraph_eigen_algorithm_t_IGRAPH_EIGEN_LAPACK,
        /// ARPACK (matrix-free, only matrix-vector products).
        Arpack = igraph_eigen_algorithm_t_IGRAPH_EIGEN_ARPACK,
        /// Complex variant of `Auto` (not implemented by igraph 1.0.0 and 1.0.1).
        CompAuto = igraph_eigen_algorithm_t_IGRAPH_EIGEN_COMP_AUTO,
        /// Complex variant of `Lapack` (not implemented by igraph 1.0.0 and 1.0.1).
        CompLapack = igraph_eigen_algorithm_t_IGRAPH_EIGEN_COMP_LAPACK,
        /// Complex variant of `Arpack` (not implemented by igraph 1.0.0 and 1.0.1).
        CompArpack = igraph_eigen_algorithm_t_IGRAPH_EIGEN_COMP_ARPACK,
    }
}

/// Which eigenvalues to compute (`igraph_eigen_which_t`).
///
/// Counts must be between 1 and the order of the matrix. Not every choice is
/// supported by every solver: the symmetric solvers accept the magnitude,
/// algebraic, [`BothEnds`](Self::BothEnds), [`All`](Self::All),
/// [`Interval`](Self::Interval) and [`Select`](Self::Select) choices (the
/// last three only with LAPACK); the non-symmetric solver accepts the
/// magnitude, real part, imaginary part, [`All`](Self::All) and
/// [`Select`](Self::Select) choices.
#[derive(Debug, Clone, PartialEq)]
pub enum EigenWhich {
    /// The given number of eigenvalues of largest magnitude.
    LargestMagnitude(usize),
    /// The given number of eigenvalues of smallest magnitude.
    SmallestMagnitude(usize),
    /// The given number of largest (algebraic) eigenvalues; symmetric only.
    LargestAlgebraic(usize),
    /// The given number of smallest (algebraic) eigenvalues; symmetric only.
    SmallestAlgebraic(usize),
    /// The given number of eigenvalues, alternating from the two ends of the
    /// spectrum (largest first); symmetric only. LAPACK needs at least 2.
    BothEnds(usize),
    /// The given number of eigenvalues with largest real part; non-symmetric only.
    LargestReal(usize),
    /// The given number of eigenvalues with smallest real part; non-symmetric only.
    SmallestReal(usize),
    /// The given number of eigenvalues with largest imaginary part; non-symmetric only.
    LargestImaginary(usize),
    /// The given number of eigenvalues with smallest imaginary part; non-symmetric only.
    SmallestImaginary(usize),
    /// All the eigenvalues.
    All,
    /// All the eigenvalues in the half-open interval `(low, high]`;
    /// symmetric LAPACK only.
    Interval {
        /// Exclusive lower bound.
        low: f64,
        /// Inclusive upper bound.
        high: f64,
    },
    /// Eigenvalues by (zero-based) position: for symmetric matrices in
    /// increasing algebraic order, for non-symmetric ones in increasing
    /// magnitude. LAPACK only.
    Select(Range<usize>),
}

impl EigenWhich {
    pub(crate) fn to_raw(&self, n: usize) -> Result<igraph_eigen_which_t> {
        use EigenWhich::*;
        let mut raw = igraph_eigen_which_t {
            pos: igraph_eigen_which_position_t_IGRAPH_EIGEN_ALL,
            howmany: 0,
            il: 0,
            iu: 0,
            vl: 0.0,
            vu: 0.0,
            vestimate: 0,
            balance: DgeevxBalance::None.into(),
        };
        let count = |k: usize| -> Result<std::ffi::c_int> {
            if k == 0 || k > n {
                return Err(Error::invalid(format!(
                    "cannot compute {k} eigenvalues of a matrix of order {n}"
                )));
            }
            to_c_int(k, "number of eigenvalues")
        };
        let (pos, howmany) = match self {
            LargestMagnitude(k) => (igraph_eigen_which_position_t_IGRAPH_EIGEN_LM, count(*k)?),
            SmallestMagnitude(k) => (igraph_eigen_which_position_t_IGRAPH_EIGEN_SM, count(*k)?),
            LargestAlgebraic(k) => (igraph_eigen_which_position_t_IGRAPH_EIGEN_LA, count(*k)?),
            SmallestAlgebraic(k) => (igraph_eigen_which_position_t_IGRAPH_EIGEN_SA, count(*k)?),
            BothEnds(k) => (igraph_eigen_which_position_t_IGRAPH_EIGEN_BE, count(*k)?),
            LargestReal(k) => (igraph_eigen_which_position_t_IGRAPH_EIGEN_LR, count(*k)?),
            SmallestReal(k) => (igraph_eigen_which_position_t_IGRAPH_EIGEN_SR, count(*k)?),
            LargestImaginary(k) => (igraph_eigen_which_position_t_IGRAPH_EIGEN_LI, count(*k)?),
            SmallestImaginary(k) => (igraph_eigen_which_position_t_IGRAPH_EIGEN_SI, count(*k)?),
            All => (igraph_eigen_which_position_t_IGRAPH_EIGEN_ALL, 0),
            Interval { low, high } => {
                if low.is_nan() || high.is_nan() || low >= high {
                    return Err(Error::invalid(
                        "the eigenvalue interval must satisfy low < high",
                    ));
                }
                raw.vl = *low;
                raw.vu = *high;
                // An upper bound on the number of eigenvalues found: always correct.
                raw.vestimate = to_c_int(n, "matrix order")?;
                (igraph_eigen_which_position_t_IGRAPH_EIGEN_INTERVAL, 0)
            }
            Select(r) => {
                if r.start >= r.end || r.end > n {
                    return Err(Error::invalid(format!(
                        "invalid eigenvalue range {r:?} for order {n}"
                    )));
                }
                raw.il = to_c_int(r.start + 1, "range start")?;
                raw.iu = to_c_int(r.end, "range end")?;
                (igraph_eigen_which_position_t_IGRAPH_EIGEN_SELECT, 0)
            }
        };
        raw.pos = pos;
        raw.howmany = howmany;
        Ok(raw)
    }
}

/// Eigenvalues and eigenvectors of a real symmetric matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct SymmetricEigen {
    /// The eigenvalues, in the order implied by the [`EigenWhich`] choice:
    /// decreasing magnitude for `LargestMagnitude`, increasing magnitude for
    /// `SmallestMagnitude`, decreasing for `LargestAlgebraic`, increasing for
    /// `SmallestAlgebraic`, `All`, `Interval` and `Select`, and alternating
    /// largest / smallest for `BothEnds` — whichever algorithm is used.
    pub values: Vec<f64>,
    /// The unit eigenvectors, in the columns (`n` × number of eigenvalues).
    pub vectors: Matrix,
}

/// Eigenvalues and eigenvectors of a general real matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexEigen {
    /// The (possibly complex) eigenvalues.
    pub values: Vec<Complex>,
    /// The eigenvectors: `vectors[k]` belongs to `values[k]`.
    pub vectors: Vec<Vec<Complex>>,
}

enum Operator<'a> {
    Dense(&'a Matrix),
    Sparse(&'a SparseMat),
    Fn(igraph_arpack_function_t, *mut c_void),
}

fn symmetric_common(
    op: Operator<'_>,
    n: usize,
    which: &EigenWhich,
    algorithm: EigenAlgorithm,
    options: &ArpackOptions,
) -> Result<SymmetricEigen> {
    if n == 0 {
        return Err(Error::invalid(
            "cannot compute the eigenvalues of an empty matrix",
        ));
    }
    let cn = to_c_int(n, "matrix order")?;
    // Resolve `Auto` here: igraph (1.0.0 and 1.0.1) compares the (unused,
    // zero) `howmany` of `All`, `Interval` and `Select` with `n` and so picks
    // ARPACK for them when `n >= 100`, where they fail. And for `n <= 2`
    // igraph's "ARPACK" code path is a closed-form shortcut that confuses
    // magnitude with algebraic order (see `uses_2x2_shortcut`): LAPACK is
    // exact there.
    let algorithm = match (algorithm, which) {
        (
            EigenAlgorithm::Auto,
            EigenWhich::All | EigenWhich::Interval { .. } | EigenWhich::Select(_),
        ) => EigenAlgorithm::Lapack,
        (
            EigenAlgorithm::Auto,
            EigenWhich::LargestMagnitude(k)
            | EigenWhich::SmallestMagnitude(k)
            | EigenWhich::LargestAlgebraic(k)
            | EigenWhich::SmallestAlgebraic(k)
            | EigenWhich::BothEnds(k),
        ) if *k == n || n < 100 => EigenAlgorithm::Lapack,
        (EigenAlgorithm::Auto, _) => EigenAlgorithm::Arpack,
        (EigenAlgorithm::Arpack, _) if n <= 2 => EigenAlgorithm::Lapack,
        (other, _) => other,
    };
    // igraph's (1.0.0 and 1.0.1) LAPACK code path for "smallest magnitude" walks outside the
    // eigenvalue array (and the eigenvector matrix) whenever the eigenvalue
    // of smallest magnitude is at either end of the spectrum, e.g. for every
    // positive definite matrix. Compute the whole spectrum with LAPACK
    // instead and pick the eigenpairs here.
    let sm_with_lapack = match which {
        EigenWhich::SmallestMagnitude(_) => {
            which.to_raw(n)?;
            algorithm == EigenAlgorithm::Lapack
        }
        _ => false,
    };
    let raw_which = if sm_with_lapack {
        EigenWhich::All.to_raw(n)?
    } else {
        which.to_raw(n)?
    };
    let mut raw_opts = options.to_raw_unchecked_which(n)?;
    let (a, sa, fun, extra) = match op {
        Operator::Dense(m) => (m as *const Matrix, ptr::null(), None, ptr::null_mut()),
        Operator::Sparse(s) => (ptr::null(), s as *const SparseMat, None, ptr::null_mut()),
        Operator::Fn(f, e) => (ptr::null(), ptr::null(), f, e),
    };
    let mut values = Vector::new();
    let mut vectors = Matrix::new();
    // Only the ARPACK code path uses ARPACK's non-re-entrant state.
    let _arpack = if algorithm == EigenAlgorithm::Lapack {
        None
    } else {
        Some(super::arpack::ArpackGuard::enter()?)
    };
    igraph_call!(igraph_eigen_matrix_symmetric(
        a,
        sa,
        fun,
        cn,
        extra,
        algorithm.into(),
        &raw_which,
        &mut raw_opts,
        ptr::null_mut(),
        &mut values,
        &mut vectors
    ))?;
    SymmetricEigen {
        values: values.into(),
        vectors,
    }
    .ordered(which)
}

impl SymmetricEigen {
    /// Puts the eigenpairs in the documented order for `which` (igraph's
    /// LAPACK and ARPACK code paths do not agree on it), keeping only the
    /// requested number of them.
    fn ordered(self, which: &EigenWhich) -> Result<Self> {
        let key: fn(f64, f64) -> std::cmp::Ordering = match which {
            EigenWhich::LargestMagnitude(_) => |a, b| b.abs().total_cmp(&a.abs()),
            EigenWhich::SmallestMagnitude(_) => |a, b| a.abs().total_cmp(&b.abs()),
            EigenWhich::LargestAlgebraic(_) => |a, b| b.total_cmp(&a),
            EigenWhich::SmallestAlgebraic(_) => |a, b| a.total_cmp(&b),
            _ => return Ok(self),
        };
        let k = match which {
            EigenWhich::LargestMagnitude(k)
            | EigenWhich::SmallestMagnitude(k)
            | EigenWhich::LargestAlgebraic(k)
            | EigenWhich::SmallestAlgebraic(k) => (*k).min(self.values.len()),
            _ => self.values.len(),
        };
        let mut order: Vec<usize> = (0..self.values.len()).collect();
        // Stable: ties keep igraph's order.
        order.sort_by(|&i, &j| key(self.values[i], self.values[j]));
        order.truncate(k);
        let n = self.vectors.nrow();
        let has_vectors = self.vectors.ncol() == self.values.len();
        let mut data = Vec::with_capacity(n * k);
        let mut values = Vec::with_capacity(k);
        for &i in &order {
            values.push(self.values[i]);
            if has_vectors {
                let v = self.vectors.column(i);
                let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if n <= 2 && norm > 0.0 {
                    // igraph's closed-form 2 x 2 "ARPACK" shortcut does not
                    // normalize the eigenvectors.
                    data.extend(v.iter().map(|x| x / norm));
                } else {
                    data.extend_from_slice(v);
                }
            }
        }
        let vectors = if has_vectors {
            Matrix::from_column_major(n, k, &data)?
        } else {
            self.vectors
        };
        Ok(Self { values, vectors })
    }
}

/// Selected eigenvalues and eigenvectors of a real **symmetric** dense
/// matrix (`igraph_eigen_matrix_symmetric`), with LAPACK or ARPACK.
///
/// Only the upper triangle is used by LAPACK. `options` tunes ARPACK: only
/// its `tol` and `mxiter` are used (`which`, `nev` and `ncv` are derived
/// from `which`, and ARPACK starts from a random vector of the calling
/// thread's RNG). Matrices of order at most 2 always use LAPACK.
///
/// Binds `igraph_eigen_matrix_symmetric` (see the
/// [linear algebra chapter](https://igraph.org/c/html/latest/igraph-Linalg.html)).
///
/// # Errors
/// If the matrix is not square or empty, the choice is invalid for the
/// matrix order, or not supported by the algorithm (e.g. `Select` with
/// ARPACK gives [`ErrorKind::Unimplemented`](crate::ErrorKind::Unimplemented)).
///
/// # Examples
///
/// ```
/// use igraph::{linalg::*, prelude::*};
/// // Eigenvalues of [[2, 1], [1, 2]] are 1 and 3.
/// let a = Matrix::from_rows(&[[2.0, 1.0], [1.0, 2.0]]).unwrap();
/// let e = eigen_matrix_symmetric(&a, &EigenWhich::All, EigenAlgorithm::Lapack, &ArpackOptions::default()).unwrap();
/// let mut values = e.values.clone();
/// values.sort_by(f64::total_cmp);
/// assert!((values[0] - 1.0).abs() < 1e-12 && (values[1] - 3.0).abs() < 1e-12);
/// ```
pub fn eigen_matrix_symmetric(
    a: &Matrix,
    which: &EigenWhich,
    algorithm: EigenAlgorithm,
    options: &ArpackOptions,
) -> Result<SymmetricEigen> {
    if a.nrow() != a.ncol() {
        return Err(Error::invalid("eigenvalues need a square matrix"));
    }
    check_finite(a.as_slice(), "the matrix")?;
    symmetric_common(Operator::Dense(a), a.nrow(), which, algorithm, options)
}

/// Selected eigenvalues and eigenvectors of a real **symmetric** operator
/// given by a matrix-vector product closure (`igraph_eigen_matrix_symmetric`
/// with a callback), see [`arpack_rssolve`](super::arpack_rssolve) for the
/// closure contract. With LAPACK the matrix is first formed by applying the
/// closure to the unit vectors.
///
/// Binds `igraph_eigen_matrix_symmetric` (see the
/// [linear algebra chapter](https://igraph.org/c/html/latest/igraph-Linalg.html)).
///
/// # Errors
/// As [`eigen_matrix_symmetric`], plus non-finite values produced by the
/// closure.
///
/// # Examples
///
/// The operator `x -> (sum x) 1` (the all-ones matrix `J_4`) has eigenvalue
/// `4` once and `0` three times.
///
/// ```
/// use igraph::linalg::*;
/// let ones = |x: &[f64], y: &mut [f64]| {
///     let s: f64 = x.iter().sum();
///     y.iter_mut().for_each(|yi| *yi = s);
/// };
/// let e = eigen_symmetric_fn(4, ones, &EigenWhich::All, EigenAlgorithm::Lapack, &ArpackOptions::default())
///     .unwrap();
/// let mut values = e.values.clone();
/// values.sort_by(f64::total_cmp);
/// assert!(values[..3].iter().all(|x| x.abs() < 1e-12) && (values[3] - 4.0).abs() < 1e-12);
/// ```
pub fn eigen_symmetric_fn<F: FnMut(&[f64], &mut [f64])>(
    n: usize,
    matvec: F,
    which: &EigenWhich,
    algorithm: EigenAlgorithm,
    options: &ArpackOptions,
) -> Result<SymmetricEigen> {
    with_finite_guard(matvec, |mut f| {
        let op = Operator::Fn(
            Some(matvec_trampoline::<&mut dyn FnMut(&[f64], &mut [f64])>),
            &mut f as *mut &mut dyn FnMut(&[f64], &mut [f64]) as *mut c_void,
        );
        symmetric_common(op, n, which, algorithm, options)
    })
}

fn complex_common(
    op: Operator<'_>,
    n: usize,
    which: &EigenWhich,
    algorithm: EigenAlgorithm,
) -> Result<ComplexEigen> {
    if n == 0 {
        return Err(Error::invalid(
            "cannot compute the eigenvalues of an empty matrix",
        ));
    }
    let cn = to_c_int(n, "matrix order")?;
    let raw_which = which.to_raw(n)?;
    let (a, sa) = match op {
        Operator::Dense(m) => (m as *const Matrix, ptr::null()),
        Operator::Sparse(s) => (ptr::null(), s as *const SparseMat),
        Operator::Fn(..) => unreachable!("closures are not supported by igraph_eigen_matrix"),
    };
    let mut raw_opts = ArpackOptions::default().to_raw_unchecked_which(n)?;
    let mut values = VectorComplex::new();
    let mut vectors = MatrixComplex::new();
    igraph_call!(igraph_eigen_matrix(
        a,
        sa,
        None,
        cn,
        ptr::null_mut(),
        algorithm.into(),
        &raw_which,
        &mut raw_opts,
        ptr::null_mut(),
        &mut values,
        &mut vectors
    ))?;
    Ok(ComplexEigen {
        values: values.to_vec(),
        vectors: vectors.columns().map(<[Complex]>::to_vec).collect(),
    })
}

/// Selected eigenvalues and eigenvectors of a general real dense matrix
/// (`igraph_eigen_matrix`); only [`EigenAlgorithm::Lapack`] is implemented
/// by igraph 1.0.0 and 1.0.1.
///
/// The eigenvalues are ordered according to `which`; [`EigenWhich::All`]
/// and [`EigenWhich::Select`] use increasing magnitude. Among eigenvalues of
/// equal magnitude (resp. real or imaginary part), real ones come first for
/// the "largest" choices and complex ones come first for the "smallest"
/// choices, `All` and `Select`, so that each "smallest" order is the exact
/// reverse of the corresponding "largest" one.
///
/// Binds `igraph_eigen_matrix` (see the
/// [linear algebra chapter](https://igraph.org/c/html/latest/igraph-Linalg.html)).
/// The callback form of the C function is not exposed: igraph 1.0.0 and 1.0.1
/// dereferences a null matrix in that code path.
///
/// # Examples
///
/// A rotation by 90 degrees has eigenvalues `±i`.
///
/// ```
/// use igraph::{linalg::*, prelude::*};
/// let r = Matrix::from_rows(&[[0.0, -1.0], [1.0, 0.0]]).unwrap();
/// let e = eigen_matrix(&r, &EigenWhich::All, EigenAlgorithm::Lapack).unwrap();
/// assert_eq!(e.values.len(), 2);
/// for v in &e.values {
///     assert!(v.re().abs() < 1e-12 && (v.im().abs() - 1.0).abs() < 1e-12);
/// }
/// ```
pub fn eigen_matrix(
    a: &Matrix,
    which: &EigenWhich,
    algorithm: EigenAlgorithm,
) -> Result<ComplexEigen> {
    if a.nrow() != a.ncol() {
        return Err(Error::invalid("eigenvalues need a square matrix"));
    }
    check_finite(a.as_slice(), "the matrix")?;
    complex_common(Operator::Dense(a), a.nrow(), which, algorithm)
}

impl igraph_sparsemat_t {
    /// Selected eigenvalues and eigenvectors of a symmetric sparse matrix
    /// (`igraph_eigen_matrix_symmetric` with a sparse input); see
    /// [`eigen_matrix_symmetric`].
    pub fn eigen_symmetric(
        &self,
        which: &EigenWhich,
        algorithm: EigenAlgorithm,
        options: &ArpackOptions,
    ) -> Result<SymmetricEigen> {
        if self.nrow() != self.ncol() {
            return Err(Error::invalid("eigenvalues need a square matrix"));
        }
        let cc = self.compress()?;
        check_finite(&cc.getelements()?.x, "the matrix")?;
        symmetric_common(
            Operator::Sparse(&cc),
            self.nrow(),
            which,
            algorithm,
            options,
        )
    }

    /// Selected eigenvalues and eigenvectors of a general sparse matrix
    /// (`igraph_eigen_matrix` with a sparse input); see [`eigen_matrix`].
    pub fn eigen(&self, which: &EigenWhich, algorithm: EigenAlgorithm) -> Result<ComplexEigen> {
        if self.nrow() != self.ncol() {
            return Err(Error::invalid("eigenvalues need a square matrix"));
        }
        let cc = self.compress()?;
        check_finite(&cc.getelements()?.x, "the matrix")?;
        complex_common(Operator::Sparse(&cc), self.nrow(), which, algorithm)
    }
}

impl igraph_t {
    /// Eigenvalues and eigenvectors of the adjacency matrix of an undirected
    /// graph (`igraph_eigen_adjacency`), computed with ARPACK (the only
    /// algorithm implemented by igraph 1.0.0 and 1.0.1, also chosen by
    /// [`EigenAlgorithm::Auto`]). Self-loops count once on the diagonal,
    /// multi-edges add up.
    ///
    /// Supported choices: largest/smallest magnitude, largest/smallest
    /// algebraic.
    ///
    /// Binds `igraph_eigen_adjacency` (see the
    /// [linear algebra chapter](https://igraph.org/c/html/latest/igraph-Linalg.html)).
    ///
    /// ARPACK starts from a random vector drawn from the calling thread's
    /// igraph RNG ([`rng::seed`](crate::rng::seed) makes runs reproducible).
    /// On small graphs whose wanted eigenvalue is highly repeated (e.g. the
    /// eigenvalue `-1` of a complete graph) it can stop with "maximum number
    /// of iterations reached" for some start vectors; LAPACK on the dense
    /// matrix is the robust choice there.
    ///
    /// See also [`Graph::eigenvector_centrality`](crate::Graph::eigenvector_centrality)
    /// (the scaled leading eigenvector, also for directed and weighted
    /// graphs) and [`Graph::get_adjacency`](crate::Graph::get_adjacency) to
    /// form the dense matrix for [`eigen_matrix_symmetric`] or
    /// [`lapack_dsyevr`](super::lapack_dsyevr).
    ///
    /// # Errors
    /// For directed graphs and unsupported choices
    /// ([`ErrorKind::Unimplemented`](crate::ErrorKind::Unimplemented)).
    ///
    /// # Examples
    ///
    /// The complete graph `K_n` has adjacency eigenvalues `n - 1` (once) and
    /// `-1`.
    ///
    /// ```
    /// use igraph::{linalg::*, prelude::*};
    /// let k6 = Graph::full(6, false, false).unwrap();
    /// let opts = ArpackOptions::default();
    /// let e = k6.eigen_adjacency(&EigenWhich::LargestAlgebraic(1), EigenAlgorithm::Auto, &opts).unwrap();
    /// assert!((e.values[0] - 5.0).abs() < 1e-10);
    /// ```
    pub fn eigen_adjacency(
        &self,
        which: &EigenWhich,
        algorithm: EigenAlgorithm,
        options: &ArpackOptions,
    ) -> Result<SymmetricEigen> {
        let n = self.vcount();
        if n == 0 {
            return Err(Error::invalid("the graph has no vertices"));
        }
        let raw_which = which.to_raw(n)?;
        let mut raw_opts = options.to_raw_unchecked_which(n)?;
        let mut values = Vector::new();
        let mut vectors = Matrix::new();
        let mut cvalues = VectorComplex::new();
        let mut cvectors = MatrixComplex::new();
        let _arpack = super::arpack::ArpackGuard::enter()?;
        igraph_call!(igraph_eigen_adjacency(
            self,
            algorithm.into(),
            &raw_which,
            &mut raw_opts,
            ptr::null_mut(),
            &mut values,
            &mut vectors,
            &mut cvalues,
            &mut cvectors
        ))?;
        SymmetricEigen {
            values: values.into(),
            vectors,
        }
        .ordered(which)
    }
}

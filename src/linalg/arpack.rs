//! The ARPACK interface (`igraph_arpack.h`).
//!
//! `igraph_arpack_options_get_default` is not wrapped: it hands out a mutable
//! pointer to igraph's shared default options; use
//! [`ArpackOptions::default()`](ArpackOptions) (`igraph_arpack_options_init`)
//! instead.

use super::{Complex, to_c_int};
use crate::{
    error::{Error, ErrorKind, Result, catch_panic, check, ensure_init},
    ffi::*,
    igraph_call,
    matrix::Matrix,
    vector::Vector,
};
use std::{
    cell::Cell,
    ffi::{CStr, c_char, c_int, c_void},
    fmt,
    mem::MaybeUninit,
    ptr,
};

/// Which eigenvalues ARPACK should compute (the `which` field of
/// `igraph_arpack_options_t`).
///
/// The first five are valid for symmetric problems ([`arpack_rssolve`]),
/// the magnitude-based and the real/imaginary-part-based ones for
/// non-symmetric problems ([`arpack_rnsolve`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ArpackWhich {
    /// Largest magnitude (`"LM"`), the default; symmetric and non-symmetric.
    #[default]
    LargestMagnitude,
    /// Smallest magnitude (`"SM"`); symmetric and non-symmetric.
    SmallestMagnitude,
    /// Largest algebraic value (`"LA"`); symmetric only.
    LargestAlgebraic,
    /// Smallest algebraic value (`"SA"`); symmetric only.
    SmallestAlgebraic,
    /// Half from each end of the spectrum, one more from the high end when
    /// `nev` is odd (`"BE"`); symmetric only.
    BothEnds,
    /// Largest real part (`"LR"`); non-symmetric only.
    LargestReal,
    /// Smallest real part (`"SR"`); non-symmetric only.
    SmallestReal,
    /// Largest imaginary part (`"LI"`); non-symmetric only.
    LargestImaginary,
    /// Smallest imaginary part (`"SI"`); non-symmetric only.
    SmallestImaginary,
}

impl ArpackWhich {
    /// The two-letter ARPACK code, e.g. `"LM"`.
    pub fn code(self) -> &'static str {
        match self {
            Self::LargestMagnitude => "LM",
            Self::SmallestMagnitude => "SM",
            Self::LargestAlgebraic => "LA",
            Self::SmallestAlgebraic => "SA",
            Self::BothEnds => "BE",
            Self::LargestReal => "LR",
            Self::SmallestReal => "SR",
            Self::LargestImaginary => "LI",
            Self::SmallestImaginary => "SI",
        }
    }

    /// Whether this choice is valid for symmetric problems.
    pub fn is_symmetric(self) -> bool {
        matches!(
            self,
            Self::LargestMagnitude
                | Self::SmallestMagnitude
                | Self::LargestAlgebraic
                | Self::SmallestAlgebraic
                | Self::BothEnds
        )
    }

    /// Whether this choice is valid for non-symmetric problems.
    pub fn is_nonsymmetric(self) -> bool {
        !matches!(
            self,
            Self::LargestAlgebraic | Self::SmallestAlgebraic | Self::BothEnds
        )
    }
}

/// The kind of eigenproblem ARPACK solves (the `mode` field of
/// `igraph_arpack_options_t`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ArpackMode {
    /// Standard problem `A x = lambda x`, driven by products `y = A x`
    /// (`mode = 1`). The default.
    #[default]
    Regular,
    /// Shift-and-invert mode (`mode = 3`): the operator is
    /// `(A - sigma I)^-1`, so that the eigenvalues of `A` closest to `sigma`
    /// become the largest in magnitude. With the closure-based solvers the
    /// closure must compute `y = (A - sigma I)^-1 x`; the returned
    /// eigenvalues are those of `A`. [`SparseMat::arpack_rssolve`](super::SparseMat::arpack_rssolve)
    /// factorizes `A - sigma I` itself.
    ShiftInvert {
        /// The shift.
        sigma: f64,
    },
}

/// Options of the ARPACK eigensolvers (a safe subset of
/// `igraph_arpack_options_t`).
///
/// [`Default`] gives igraph's defaults (`igraph_arpack_options_init`): one
/// eigenvalue of largest magnitude, machine precision tolerance, automatic
/// number of Lanczos/Arnoldi vectors, at most 3000 iterations, regular mode
/// and a random start vector (drawn from igraph's RNG, so results are
/// reproducible after [`rng::seed`](crate::rng::seed)).
///
/// The functions that pick `which` and `nev` themselves (e.g.
/// [`eigen_matrix_symmetric`](super::eigen_matrix_symmetric),
/// [`Graph::eigen_adjacency`](crate::Graph::eigen_adjacency) or the spectral
/// embeddings) ignore those fields as well as `ncv`, `mode` and the start
/// vector: they only use `tol` and `mxiter`.
///
/// ```
/// use igraph::linalg::{ArpackOptions, ArpackWhich};
/// let opts = ArpackOptions::default().with_nev(3).with_which(ArpackWhich::SmallestAlgebraic).with_tol(1e-10);
/// assert_eq!(opts.nev, 3);
/// assert_eq!(opts.mxiter, 3000);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ArpackOptions {
    /// Number of eigenvalues to compute (`nev`), at least 1 and smaller than
    /// the order of the matrix (at most `n - 2` for non-symmetric problems).
    pub nev: usize,
    /// Which eigenvalues to compute.
    pub which: ArpackWhich,
    /// Relative accuracy of the Ritz values; `0` means machine precision.
    pub tol: f64,
    /// Number of Lanczos (Arnoldi) vectors; `0` lets igraph choose. Otherwise
    /// it must satisfy `nev < ncv <= n`.
    pub ncv: usize,
    /// Maximum number of Arnoldi update iterations.
    pub mxiter: usize,
    /// Regular or shift-and-invert mode.
    pub mode: ArpackMode,
    /// Starting vector (length `n`); `None` draws a random one.
    pub start: Option<Vec<f64>>,
}

impl Default for ArpackOptions {
    fn default() -> Self {
        Self {
            nev: 1,
            which: ArpackWhich::LargestMagnitude,
            tol: 0.0,
            ncv: 0,
            mxiter: 3000,
            mode: ArpackMode::Regular,
            start: None,
        }
    }
}

impl ArpackOptions {
    /// Sets the number of eigenvalues to compute.
    pub fn with_nev(mut self, nev: usize) -> Self {
        self.nev = nev;
        self
    }

    /// Sets which eigenvalues to compute.
    pub fn with_which(mut self, which: ArpackWhich) -> Self {
        self.which = which;
        self
    }

    /// Sets the tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the number of Lanczos/Arnoldi vectors (`0` = automatic).
    pub fn with_ncv(mut self, ncv: usize) -> Self {
        self.ncv = ncv;
        self
    }

    /// Sets the maximum number of iterations.
    pub fn with_mxiter(mut self, mxiter: usize) -> Self {
        self.mxiter = mxiter;
        self
    }

    /// Selects the shift-and-invert mode with the given shift.
    pub fn with_shift_invert(mut self, sigma: f64) -> Self {
        self.mode = ArpackMode::ShiftInvert { sigma };
        self
    }

    /// Sets the starting vector.
    pub fn with_start(mut self, start: Vec<f64>) -> Self {
        self.start = Some(start);
        self
    }

    /// Validates the options for a problem of order `n` and converts them to
    /// the raw C struct.
    pub(crate) fn to_raw(&self, n: usize, symmetric: bool) -> Result<igraph_arpack_options_t> {
        let cn = to_c_int(n, "matrix order")?;
        if n == 0 {
            return Err(Error::invalid("ARPACK needs a matrix of order at least 1"));
        }
        if self.nev == 0 {
            return Err(Error::invalid("ARPACK: nev must be at least 1"));
        }
        if self.ncv != 0 && (self.ncv <= self.nev || self.ncv > n) {
            return Err(Error::invalid(format!(
                "ARPACK: ncv ({}) must satisfy nev < ncv <= n",
                self.ncv
            )));
        }
        if self.mxiter == 0 {
            return Err(Error::invalid("ARPACK: mxiter must be positive"));
        }
        if symmetric && !self.which.is_symmetric() {
            return Err(Error::invalid(format!(
                "ARPACK: '{}' is not valid for symmetric problems",
                self.which.code()
            )));
        }
        if !symmetric && !self.which.is_nonsymmetric() {
            return Err(Error::invalid(format!(
                "ARPACK: '{}' is not valid for non-symmetric problems",
                self.which.code()
            )));
        }
        if !(self.tol.is_finite() && self.tol >= 0.0) {
            return Err(Error::invalid(
                "ARPACK: tol must be finite and non-negative",
            ));
        }
        if let ArpackMode::ShiftInvert { sigma } = self.mode
            && !sigma.is_finite()
        {
            return Err(Error::invalid("ARPACK: the shift must be finite"));
        }
        if let Some(start) = &self.start
            && start.len() != n
        {
            return Err(Error::invalid(format!(
                "ARPACK: start vector has length {}, expected {n}",
                start.len()
            )));
        }
        let mut raw = default_raw_options();
        raw.n = cn;
        raw.nev = to_c_int(self.nev, "nev")?;
        let code = self.which.code().as_bytes();
        raw.which = [code[0] as c_char, code[1] as c_char];
        raw.tol = self.tol;
        raw.ncv = to_c_int(self.ncv, "ncv")?;
        raw.mxiter = to_c_int(self.mxiter, "mxiter")?;
        match self.mode {
            ArpackMode::Regular => raw.mode = 1,
            ArpackMode::ShiftInvert { sigma } => {
                raw.mode = 3;
                raw.sigma = sigma;
            }
        }
        raw.start = self.start.is_some() as c_int;
        Ok(raw)
    }

    /// Raw options for the functions that set `which`/`nev` themselves: the
    /// start vector is dropped.
    pub(crate) fn to_raw_unchecked_which(&self, n: usize) -> Result<igraph_arpack_options_t> {
        let mut o = self.clone();
        o.start = None;
        o.which = ArpackWhich::LargestMagnitude;
        o.nev = 1;
        o.ncv = 0;
        o.mode = ArpackMode::Regular;
        o.to_raw(n.max(1), true)
    }

    /// The `vectors` matrix to hand to the solver: `n` × 1 holding the start
    /// vector when there is one (igraph reads it from the first column).
    pub(crate) fn start_matrix(&self, n: usize) -> Result<Matrix> {
        match &self.start {
            Some(s) => Matrix::from_column_major(n, 1, s),
            None => Ok(Matrix::new()),
        }
    }
}

fn default_raw_options() -> igraph_arpack_options_t {
    let mut raw = MaybeUninit::<igraph_arpack_options_t>::uninit();
    unsafe {
        igraph_arpack_options_init(raw.as_mut_ptr());
        raw.assume_init()
    }
}

/// Statistics reported by ARPACK after a successful run (the output fields
/// of `igraph_arpack_options_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ArpackInfo {
    /// Number of Arnoldi iterations taken (`noiter`).
    pub iterations: usize,
    /// Number of converged Ritz values (`nconv`).
    pub nconv: usize,
    /// Total number of operator applications, i.e. calls of the matrix-vector
    /// product (`numop`), as reported by ARPACK (the ARPACK bundled with
    /// igraph 1.0.0 and 1.0.1 leave it at 0).
    pub numop: usize,
    /// Total number of re-orthogonalization steps (`numreo`).
    pub numreo: usize,
}

impl ArpackInfo {
    pub(crate) fn from_raw(raw: &igraph_arpack_options_t) -> Self {
        let u = |x: c_int| x.max(0) as usize;
        Self {
            iterations: u(raw.noiter),
            nconv: u(raw.nconv),
            numop: u(raw.numop),
            numreo: u(raw.numreo),
        }
    }
}

/// Result of the symmetric ARPACK solvers.
#[derive(Debug, Clone, PartialEq)]
pub struct ArpackSymmetricResult {
    /// The eigenvalues, ordered according to [`ArpackOptions::which`]:
    /// decreasing for `LargestAlgebraic`, increasing for `SmallestAlgebraic`,
    /// by decreasing (increasing) magnitude for `LargestMagnitude`
    /// (`SmallestMagnitude`), and alternating smallest / largest (starting
    /// from the smallest) for `BothEnds`.
    pub values: Vec<f64>,
    /// The corresponding unit eigenvectors, in the columns (`n` × `nev`).
    pub vectors: Matrix,
    /// Convergence statistics.
    pub info: ArpackInfo,
}

impl ArpackSymmetricResult {
    pub(crate) fn from_raw(values: Vector, vectors: Matrix, raw: &igraph_arpack_options_t) -> Self {
        Self {
            values: values.into(),
            vectors,
            info: ArpackInfo::from_raw(raw),
        }
    }

    /// Picks the requested eigenpairs out of both eigenpairs of a 2 × 2
    /// problem solved by igraph's closed-form shortcut (see
    /// [`uses_2x2_shortcut`]), normalizing the eigenvectors and ordering them
    /// as ARPACK would.
    pub(crate) fn select_2x2(self, which: ArpackWhich, nev: usize) -> Result<Self> {
        let n = self.vectors.nrow();
        let mut pairs: Vec<(f64, Vec<f64>)> = self
            .values
            .iter()
            .enumerate()
            .map(|(j, &value)| {
                let v = self.vectors.column(j);
                let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                let v = if norm > 0.0 {
                    v.iter().map(|x| x / norm).collect()
                } else {
                    v.to_vec()
                };
                (value, v)
            })
            .collect();
        let key: fn(&f64, &f64) -> std::cmp::Ordering = match which {
            ArpackWhich::LargestAlgebraic => |a, b| b.total_cmp(a),
            ArpackWhich::SmallestAlgebraic | ArpackWhich::BothEnds => |a, b| a.total_cmp(b),
            ArpackWhich::SmallestMagnitude => |a, b| a.abs().total_cmp(&b.abs()),
            _ => |a, b| b.abs().total_cmp(&a.abs()),
        };
        pairs.sort_by(|a, b| key(&a.0, &b.0));
        let keep = nev.min(pairs.len());
        if which == ArpackWhich::BothEnds {
            // Increasing order; with a single eigenvalue it is the largest.
            pairs.drain(..pairs.len() - keep);
        } else {
            pairs.truncate(keep);
        }
        let data: Vec<f64> = pairs.iter().flat_map(|(_, v)| v.iter().copied()).collect();
        let mut info = self.info;
        info.nconv = info.nconv.min(keep);
        Ok(Self {
            values: pairs.iter().map(|&(x, _)| x).collect(),
            vectors: Matrix::from_column_major(n, keep, &data)?,
            info,
        })
    }
}

/// Whether igraph solves this symmetric problem with its closed-form 2 × 2
/// shortcut (`igraph_i_arpack_rssolve_2x2`, regular mode only) instead of
/// ARPACK. In igraph 1.0.0 and 1.0.1 that shortcut treats "largest
/// (smallest) magnitude" as "largest (smallest) algebraic" and does not
/// normalize the eigenvectors: the callers ask it for both eigenpairs and
/// finish with [`ArpackSymmetricResult::select_2x2`].
pub(crate) fn uses_2x2_shortcut(n: usize, options: &ArpackOptions) -> bool {
    n == 2 && options.mode == ArpackMode::Regular
}

/// Result of the non-symmetric ARPACK solvers.
#[derive(Debug, Clone, PartialEq)]
pub struct ArpackNonSymmetricResult {
    /// The (possibly complex) eigenvalues; complex ones come in conjugate pairs.
    pub values: Vec<Complex>,
    /// The corresponding eigenvectors, `vectors[k]` belonging to `values[k]`,
    /// with unit Euclidean norm.
    pub vectors: Vec<Vec<Complex>>,
    /// Convergence statistics.
    pub info: ArpackInfo,
}

impl ArpackNonSymmetricResult {
    /// Decodes the packed output of `igraph_arpack_rnsolve` (values: `k` × 2
    /// matrix of real/imaginary parts; vectors: one column per real
    /// eigenvalue, two columns per complex conjugate pair).
    pub(crate) fn from_raw(
        values: Matrix,
        vectors: Matrix,
        raw: &igraph_arpack_options_t,
        nev: usize,
    ) -> Result<Self> {
        let k = nev.min(values.nrow());
        let (re, im): (Vec<f64>, Vec<f64>) =
            (0..k).map(|i| (values[(i, 0)], values[(i, 1)])).unzip();
        let n = vectors.nrow();
        let mut out_values = Vec::with_capacity(k);
        let mut out_vectors = Vec::with_capacity(k);
        let (mut i, mut col) = (0, 0);
        while i < k && col < vectors.ncol() {
            if im[i] == 0.0 {
                out_values.push(Complex::new(re[i], 0.0));
                out_vectors.push(
                    vectors
                        .column(col)
                        .iter()
                        .map(|&x| Complex::new(x, 0.0))
                        .collect(),
                );
                col += 1;
                i += 1;
                continue;
            }
            if col + 1 >= vectors.ncol() {
                break;
            }
            // The stored vector belongs to the eigenvalue with positive imaginary part.
            let (vr, vi) = (vectors.column(col), vectors.column(col + 1));
            let pos: Vec<Complex> = (0..n).map(|r| Complex::new(vr[r], vi[r])).collect();
            let conj: Vec<Complex> = pos.iter().map(|c| Complex::new(c.re(), -c.im())).collect();
            let (first, second) = if im[i] > 0.0 {
                (pos, conj)
            } else {
                (conj, pos)
            };
            out_values.push(Complex::new(re[i], im[i]));
            out_vectors.push(first);
            if i + 1 < k && im[i + 1] == -im[i] && re[i + 1] == re[i] {
                out_values.push(Complex::new(re[i + 1], im[i + 1]));
                out_vectors.push(second);
                i += 2;
            } else {
                i += 1;
            }
            col += 2;
        }
        if n <= 2 {
            // igraph's closed-form 1 × 1 and 2 × 2 shortcuts do not normalize
            // the eigenvectors, unlike ARPACK.
            for v in &mut out_vectors {
                let norm = v
                    .iter()
                    .map(|c| c.re() * c.re() + c.im() * c.im())
                    .sum::<f64>()
                    .sqrt();
                if norm > 0.0 {
                    for c in v.iter_mut() {
                        *c = Complex::new(c.re() / norm, c.im() / norm);
                    }
                }
            }
        }
        Ok(Self {
            values: out_values,
            vectors: out_vectors,
            info: ArpackInfo::from_raw(raw),
        })
    }
}

/// Preallocated working memory for repeated ARPACK runs
/// (`igraph_arpack_storage_t`), freed on drop (`igraph_arpack_storage_destroy`).
///
/// Pass it to [`arpack_rssolve`]/[`arpack_rnsolve`] to avoid reallocating
/// the workspace for every eigenproblem of order at most `maxn` using at most
/// `maxncv` Lanczos/Arnoldi vectors (remember that `ncv = 0` lets igraph pick
/// a value up to `n`).
///
/// Binds [`igraph_arpack_storage_init`](https://igraph.org/c/html/latest/igraph-Linalg.html#igraph_arpack_storage_init).
pub struct ArpackStorage {
    raw: igraph_arpack_storage_t,
    symmetric: bool,
}

// The storage is plain heap memory owned exclusively by this value.
unsafe impl Send for ArpackStorage {}

impl ArpackStorage {
    /// Allocates storage for problems of order at most `maxn`, with at most
    /// `maxncv` Lanczos/Arnoldi vectors; `symmetric` selects the layout for
    /// [`arpack_rssolve`] (`true`) or [`arpack_rnsolve`] (`false`; such a
    /// storage also works for symmetric problems).
    pub fn new(maxn: usize, maxncv: usize, symmetric: bool) -> Result<Self> {
        if maxn == 0 || maxncv == 0 {
            return Err(Error::invalid("ARPACK storage dimensions must be positive"));
        }
        to_c_int(maxn, "maxn")?;
        to_c_int(maxncv, "maxncv")?;
        ensure_init();
        let mut raw = MaybeUninit::<igraph_arpack_storage_t>::zeroed();
        let (n, ncv) = (maxn as igraph_int_t, maxncv as igraph_int_t);
        check(unsafe { igraph_arpack_storage_init(raw.as_mut_ptr(), n, ncv, n, symmetric) })?;
        Ok(Self {
            raw: unsafe { raw.assume_init() },
            symmetric,
        })
    }

    /// Maximum order of the problems.
    pub fn maxn(&self) -> usize {
        self.raw.maxn as usize
    }

    /// Maximum number of Lanczos/Arnoldi vectors.
    pub fn maxncv(&self) -> usize {
        self.raw.maxncv as usize
    }

    /// Whether the storage was allocated for symmetric problems only.
    pub fn is_symmetric(&self) -> bool {
        self.symmetric
    }
}

impl Drop for ArpackStorage {
    fn drop(&mut self) {
        unsafe { igraph_arpack_storage_destroy(&mut self.raw) };
    }
}

impl fmt::Debug for ArpackStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArpackStorage")
            .field("maxn", &self.maxn())
            .field("maxncv", &self.maxncv())
            .field("symmetric", &self.symmetric)
            .finish()
    }
}

thread_local! {
    static ARPACK_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

/// Marks an ARPACK run on the calling thread, to refuse nested runs.
///
/// igraph's vendored ARPACK is not re-entrant: its Fortran-translated
/// routines (`dsaupd`, `dsaup2`, `dnaupd`, `dnaup2`, `dgetv0`, ...) keep the
/// state of the running iteration in `IGRAPH_F77_SAVE` variables, i.e.
/// thread-local statics shared by every ARPACK run on the thread. A second run
/// started from inside a running one (from the matrix-vector closure of
/// [`arpack_rssolve`], or from an interruption or progress handler, which
/// igraph calls from inside the ARPACK loop) overwrites that state, and the
/// outer run then returns wrong results, fails or hangs; as the state includes
/// indices into the work arrays, it could even access memory out of bounds.
///
/// A wrapper whose C function may run ARPACK holds a guard for the whole C
/// call (`let _arpack = crate::linalg::ArpackGuard::enter()?;`); a nested
/// entry returns an [`ErrorKind::Failure`] error instead. Every wrapper whose
/// C function may run ARPACK takes it: those of `crate::linalg`, the spectral
/// centralities and ARPACK PageRank of `centrality.rs`, and the leading
/// eigenvector community detection of `community.rs`.
pub(crate) struct ArpackGuard {
    _not_send: std::marker::PhantomData<*const ()>,
}

impl ArpackGuard {
    /// Enters an ARPACK run, or fails if one is already running on this thread.
    pub(crate) fn enter() -> Result<Self> {
        if ARPACK_ACTIVE.with(|a| a.replace(true)) {
            return Err(Error::new(
                ErrorKind::Failure,
                "ARPACK-based functions cannot be nested inside ARPACK callbacks or handlers",
            ));
        }
        Ok(Self {
            _not_send: std::marker::PhantomData,
        })
    }
}

impl Drop for ArpackGuard {
    fn drop(&mut self) {
        ARPACK_ACTIVE.with(|a| a.set(false));
    }
}

/// The C callback handed to ARPACK: calls the Rust closure `F` with
/// `x = from` and `y = to` (zeroed first), catching panics.
///
/// The closure runs in a fresh level of igraph's "finally" stack
/// (`IGRAPH_FINALLY_ENTER` / `IGRAPH_FINALLY_EXIT`): the running solver keeps
/// its work vectors on that stack, and igraph's error handler frees the
/// current level when a call fails. Without the new level, a failing igraph
/// call made by the closure (whose `Err` the closure may well ignore) would
/// free the workspace ARPACK is still writing into.
pub(crate) unsafe extern "C" fn matvec_trampoline<F: FnMut(&[f64], &mut [f64])>(
    to: *mut igraph_real_t,
    from: *const igraph_real_t,
    n: c_int,
    extra: *mut c_void,
) -> igraph_error_t {
    // SAFETY: plain bookkeeping on igraph's thread-local finally stack; the
    // matching EXIT runs below, as `catch_panic` never unwinds.
    unsafe { IGRAPH_FINALLY_ENTER() };
    let code = catch_panic(|| {
        let f = unsafe { &mut *(extra as *mut F) };
        let n = n.max(0) as usize;
        let (x, y) = unsafe {
            (
                std::slice::from_raw_parts(from, n),
                std::slice::from_raw_parts_mut(to, n),
            )
        };
        y.fill(0.0);
        f(x, y);
        // Non-finite values would make ARPACK's LAPACK routines abort the
        // process: stop with an error instead.
        if y.iter().all(|v| v.is_finite()) {
            igraph_error_type_t_IGRAPH_SUCCESS
        } else {
            igraph_error_type_t_IGRAPH_EINVAL
        }
    });
    unsafe { IGRAPH_FINALLY_EXIT() };
    code
}

/// Runs `call` with a wrapped closure that remembers whether `matvec`
/// produced non-finite values, to report that as a proper error.
pub(crate) fn with_finite_guard<F: FnMut(&[f64], &mut [f64]), R>(
    mut matvec: F,
    call: impl FnOnce(&mut dyn FnMut(&[f64], &mut [f64])) -> Result<R>,
) -> Result<R> {
    let mut nonfinite = false;
    let res = call(&mut |x: &[f64], y: &mut [f64]| {
        matvec(x, y);
        if !y.iter().all(|v| v.is_finite()) {
            nonfinite = true;
        }
    });
    if nonfinite {
        return Err(Error::invalid(
            "the matrix-vector product returned NaN or infinite values",
        ));
    }
    res
}

/// Eigenvalues and eigenvectors of a **symmetric** linear operator given as
/// a Rust closure, with ARPACK's implicitly restarted Lanczos method
/// (`igraph_arpack_rssolve`).
///
/// `matvec(x, y)` must store the product `A x` into `y` (which is zeroed
/// before each call, so accumulating into it is fine); both slices have
/// length `n`. The matrix is never formed: this is the method of choice for
/// large sparse or structured matrices. In
/// [`ArpackMode::ShiftInvert`] the closure must compute `(A - sigma I)^-1 x`
/// instead. A panic in the closure stops ARPACK and is resumed by this
/// function; a product containing NaN or infinite values stops it with an
/// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) error.
///
/// Time complexity: depends on the matrix-vector product; usually a few
/// iterations suffice, so with an O(n) product the eigenvalues are found in
/// about O(n) time.
///
/// In regular mode, 1 × 1 and 2 × 2 problems are solved in closed form by
/// igraph without running ARPACK. For 2 × 2 problems igraph 1.0.0 and 1.0.1
/// confuse the magnitude-based choices with the algebraic ones and return
/// unnormalized eigenvectors; this wrapper computes both eigenpairs and
/// selects and normalizes the requested ones itself.
///
/// # Singular operators: the start vector is multiplied first
///
/// ARPACK does not use the start vector `v0` itself as the first Krylov
/// vector: it first applies the operator and starts from `A v0` (to force the
/// start vector into the range of the operator, see `dgetv0`). For a
/// **singular** symmetric operator the range is orthogonal to the null space,
/// so the component of `v0` along the null vectors is wiped out: when `A v0`
/// has no null component *exactly* (e.g. integer matrices and start vectors,
/// as for graph Laplacians), the eigenvalue `0` can be invisible to the
/// iteration, and ARPACK then silently returns other eigenvalues instead. A start vector
/// lying in the null space fails outright with
/// [`ArpackError::ZeroStart`]. The random default start vector usually
/// recovers the null space through rounding errors, but there is no
/// guarantee.
///
/// The robust fix is a shift: solve for `A + s I` with some `s > 0` that makes
/// the operator non-singular (for a positive semidefinite `A` such as a
/// Laplacian any `s > 0` works, and "smallest magnitude" becomes "smallest
/// algebraic"), then subtract `s` from the eigenvalues; the eigenvectors are
/// unchanged. See the second example below.
///
/// # Nesting
///
/// igraph's ARPACK keeps the state of the running iteration in thread-local
/// statics, so it is not re-entrant: `matvec` (and any interruption or
/// progress handler, which igraph calls from inside the ARPACK loop) must not
/// start another ARPACK computation on the same thread. The ARPACK-based
/// functions of this module (this function, [`arpack_rnsolve`], the
/// [`SparseMat`](super::SparseMat) ARPACK solvers, the ARPACK paths of
/// [`eigen_matrix_symmetric`](super::eigen_matrix_symmetric) and
/// [`eigen_symmetric_fn`](super::eigen_symmetric_fn),
/// [`Graph::eigen_adjacency`](crate::Graph::eigen_adjacency) and the spectral
/// embeddings) check this: started while another one of them runs on the
/// thread, they fail with an
/// [`ErrorKind::Failure`](crate::ErrorKind::Failure) error instead of
/// corrupting the running solver. So do the ARPACK-based functions of other
/// modules: [`Graph::eigenvector_centrality`](crate::Graph::eigenvector_centrality),
/// [`Graph::hub_and_authority_scores`](crate::Graph::hub_and_authority_scores),
/// [`Graph::centralization_eigenvector_centrality`](crate::Graph::centralization_eigenvector_centrality),
/// PageRank with [`PageRankAlgo::Arpack`](crate::centrality::PageRankAlgo::Arpack)
/// and [`Graph::community_leading_eigenvector`](crate::Graph::community_leading_eigenvector).
/// Calling other igraph functions from `matvec` is fine, even
/// failing ones: the closure runs in its own level of igraph's cleanup stack.
///
/// Binds [`igraph_arpack_rssolve`](https://igraph.org/c/html/latest/igraph-Linalg.html#igraph_arpack_rssolve).
///
/// See also [`SparseMat::arpack_rssolve`](super::SparseMat::arpack_rssolve)
/// for a stored sparse matrix, [`eigen_symmetric_fn`](super::eigen_symmetric_fn)
/// for a front-end that can also use LAPACK, and the graph-level spectral
/// routines built on ARPACK, e.g.
/// [`Graph::eigenvector_centrality`](crate::Graph::eigenvector_centrality)
/// and [`Graph::eigen_adjacency`](crate::Graph::eigen_adjacency).
///
/// # Errors
/// Invalid options (see [`ArpackOptions`]) give
/// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue); ARPACK
/// failures give [`ErrorKind::Arpack`](crate::ErrorKind::Arpack), see
/// [`ArpackError::from_error`] for the ARPACK error condition.
///
/// # Examples
///
/// The cycle graph `C_n` has adjacency eigenvalues `2 cos(2 pi k / n)`: the
/// largest is 2, with the constant eigenvector.
///
/// ```
/// use igraph::linalg::{arpack_rssolve, ArpackOptions, ArpackWhich};
/// let n = 12;
/// let cycle = |x: &[f64], y: &mut [f64]| {
///     for i in 0..n {
///         y[i] = x[(i + 1) % n] + x[(i + n - 1) % n];
///     }
/// };
/// let opts = ArpackOptions::default().with_nev(1).with_which(ArpackWhich::LargestAlgebraic);
/// let res = arpack_rssolve(n, cycle, &opts, None).unwrap();
/// assert!((res.values[0] - 2.0).abs() < 1e-10);
/// let v = res.vectors.column(0);
/// assert!(v.iter().all(|x| (x.abs() - 1.0 / (n as f64).sqrt()).abs() < 1e-8));
/// ```
///
/// The Laplacian of the path on 10 vertices is singular (eigenvalue 0, with
/// the constant eigenvector). The ramp `v0 = (0, 1, ..., 9)` has a constant
/// component, but ARPACK iterates from `L v0 = e_9 - e_0`, which has exactly
/// none (and, by the mirror symmetry of the path, neither have the following
/// Krylov vectors): the eigenvalue 0 is missed altogether. Shifting to
/// `L + I` keeps the constant component of `v0` and finds it:
///
/// ```
/// use igraph::linalg::{arpack_rssolve, ArpackOptions, ArpackWhich};
/// use std::f64::consts::PI;
/// let n = 10;
/// // y = (L + shift I) x for the path Laplacian L.
/// let laplacian = move |shift: f64| {
///     move |x: &[f64], y: &mut [f64]| {
///         for i in 0..n {
///             let degree = if i == 0 || i == n - 1 { 1.0 } else { 2.0 };
///             y[i] = (degree + shift) * x[i];
///             if i > 0 {
///                 y[i] -= x[i - 1];
///             }
///             if i + 1 < n {
///                 y[i] -= x[i + 1];
///             }
///         }
///     }
/// };
/// let ramp: Vec<f64> = (0..n).map(|i| i as f64).collect();
/// let lambda1 = 2.0 - 2.0 * (PI / n as f64).cos(); // the smallest non-zero eigenvalue
///
/// let opts = ArpackOptions::default()
///     .with_nev(2)
///     .with_which(ArpackWhich::SmallestMagnitude)
///     .with_start(ramp);
/// let plain = arpack_rssolve(n, laplacian(0.0), &opts, None).unwrap();
/// assert!(plain.values.iter().all(|v| v.abs() > 0.09)); // 0 is missed
/// assert!(plain.values.iter().any(|v| (v - lambda1).abs() < 1e-8));
///
/// let shifted_opts = opts.with_which(ArpackWhich::SmallestAlgebraic);
/// let shifted = arpack_rssolve(n, laplacian(1.0), &shifted_opts, None).unwrap();
/// let values: Vec<f64> = shifted.values.iter().map(|v| v - 1.0).collect();
/// assert!(values[0].abs() < 1e-10);
/// assert!((values[1] - lambda1).abs() < 1e-8);
/// ```
pub fn arpack_rssolve<F: FnMut(&[f64], &mut [f64])>(
    n: usize,
    matvec: F,
    options: &ArpackOptions,
    storage: Option<&mut ArpackStorage>,
) -> Result<ArpackSymmetricResult> {
    let mut raw = options.to_raw(n, true)?;
    let shortcut = uses_2x2_shortcut(n, options);
    if shortcut {
        raw.nev = 2;
    }
    let storage_ptr = storage.map_or(ptr::null_mut(), |s| {
        &mut s.raw as *mut igraph_arpack_storage_t
    });
    let mut values = Vector::new();
    let mut vectors = options.start_matrix(n)?;
    let _arpack = ArpackGuard::enter()?;
    with_finite_guard(matvec, |mut f| {
        igraph_call!(igraph_arpack_rssolve(
            Some(matvec_trampoline::<&mut dyn FnMut(&[f64], &mut [f64])>),
            &mut f as *mut &mut dyn FnMut(&[f64], &mut [f64]) as *mut c_void,
            &mut raw,
            storage_ptr,
            &mut values,
            &mut vectors
        ))
    })?;
    let res = ArpackSymmetricResult::from_raw(values, vectors, &raw);
    if shortcut {
        res.select_2x2(options.which, options.nev)
    } else {
        Ok(res)
    }
}

/// Eigenvalues and eigenvectors of a general (**non-symmetric**) linear
/// operator given as a Rust closure, with ARPACK's implicitly restarted
/// Arnoldi method (`igraph_arpack_rnsolve`).
///
/// `matvec(x, y)` stores `A x` into `y`, as in [`arpack_rssolve`]. The
/// eigenvalues may be complex: they are returned as [`Complex`] numbers,
/// with complex conjugate pairs next to each other, and the eigenvectors are
/// unpacked into complex vectors. Note that ARPACK requires `nev <= n - 2`
/// here; in regular mode 1 × 1 and 2 × 2 problems are solved exactly by
/// igraph (in closed form, without ARPACK).
///
/// As explained for [`arpack_rssolve`], ARPACK starts from `A v0` rather
/// than from the start vector `v0`, so eigenvectors of a singular operator
/// with eigenvalue 0 may be missed (shift the operator to `A + s I` and
/// subtract `s` afterwards), and ARPACK computations cannot be nested inside
/// `matvec` or inside handlers (a nested call fails with
/// [`ErrorKind::Failure`](crate::ErrorKind::Failure)).
///
/// Binds [`igraph_arpack_rnsolve`](https://igraph.org/c/html/latest/igraph-Linalg.html#igraph_arpack_rnsolve).
///
/// See also [`SparseMat::arpack_rnsolve`](super::SparseMat::arpack_rnsolve),
/// [`eigen_matrix`](super::eigen_matrix) for dense matrices, and
/// [`Graph::pagerank`](crate::Graph::pagerank) /
/// [`Graph::hub_and_authority_scores`](crate::Graph::hub_and_authority_scores),
/// the classic non-symmetric eigenproblems on graphs.
///
/// # Errors
/// As [`arpack_rssolve`]: invalid options (see [`ArpackOptions`]; ARPACK
/// rejects `nev > n - 2` for `n > 2`), a non-finite product, or an ARPACK
/// failure ([`ErrorKind::Arpack`](crate::ErrorKind::Arpack), see
/// [`ArpackError::from_error`]).
///
/// # Examples
///
/// A directed cycle is a permutation matrix: its eigenvalues are the `n`-th
/// roots of unity, all of magnitude 1.
///
/// ```
/// use igraph::linalg::{arpack_rnsolve, ArpackOptions, ArpackWhich};
/// let n = 8;
/// let shift = |x: &[f64], y: &mut [f64]| {
///     for i in 0..n {
///         y[i] = x[(i + 1) % n];
///     }
/// };
/// let opts = ArpackOptions::default().with_nev(3).with_which(ArpackWhich::LargestReal);
/// let res = arpack_rnsolve(n, shift, &opts, None).unwrap();
/// assert!((res.values[0].re() - 1.0).abs() < 1e-8 && res.values[0].im().abs() < 1e-8);
/// for v in &res.values {
///     assert!(((v.re() * v.re() + v.im() * v.im()).sqrt() - 1.0).abs() < 1e-8);
/// }
/// ```
pub fn arpack_rnsolve<F: FnMut(&[f64], &mut [f64])>(
    n: usize,
    matvec: F,
    options: &ArpackOptions,
    storage: Option<&mut ArpackStorage>,
) -> Result<ArpackNonSymmetricResult> {
    let mut raw = options.to_raw(n, false)?;
    if let Some(s) = &storage
        && s.symmetric
    {
        return Err(Error::invalid(
            "the non-symmetric ARPACK solver needs a non-symmetric ArpackStorage",
        ));
    }
    let storage_ptr = storage.map_or(ptr::null_mut(), |s| {
        &mut s.raw as *mut igraph_arpack_storage_t
    });
    let mut values = Matrix::new();
    let mut vectors = options.start_matrix(n)?;
    let _arpack = ArpackGuard::enter()?;
    with_finite_guard(matvec, |mut f| {
        igraph_call!(igraph_arpack_rnsolve(
            Some(matvec_trampoline::<&mut dyn FnMut(&[f64], &mut [f64])>),
            &mut f as *mut &mut dyn FnMut(&[f64], &mut [f64]) as *mut c_void,
            &mut raw,
            storage_ptr,
            &mut values,
            &mut vectors
        ))
    })?;
    ArpackNonSymmetricResult::from_raw(values, vectors, &raw, options.nev)
}

/// Rewrites the packed output of the non-symmetric ARPACK solver into a
/// regular form (`igraph_arpack_unpack_complex`).
///
/// `values` is a `k` × 2 matrix of real and imaginary parts, `vectors` the
/// packed eigenvector matrix (one column per real eigenvalue, two columns —
/// real and imaginary part — per complex conjugate pair). The result keeps
/// the first `nev` eigenvalues as an `nev` × 2 matrix, and returns an
/// `n` × `2 nev` eigenvector matrix where eigenvector `k` occupies columns
/// `2k` (real part) and `2k + 1` (imaginary part).
///
/// [`arpack_rnsolve`] already returns unpacked [`Complex`] vectors; this is
/// useful to post-process raw data.
///
/// Binds [`igraph_arpack_unpack_complex`](https://igraph.org/c/html/latest/igraph-Linalg.html#igraph_arpack_unpack_complex).
///
/// # Errors
/// If `values` does not have two columns, `nev` exceeds its rows, the packed
/// vectors are too few, or a complex eigenvalue is not followed by its
/// conjugate.
pub fn arpack_unpack_complex(
    vectors: &Matrix,
    values: &Matrix,
    nev: usize,
) -> Result<(Matrix, Matrix)> {
    if values.ncol() != 2 {
        return Err(Error::invalid(
            "the eigenvalue matrix must have two columns",
        ));
    }
    if nev > values.nrow() {
        return Err(Error::invalid(
            "nev is larger than the number of eigenvalues",
        ));
    }
    // Mirror igraph's walk to make sure it stays within the vector matrix.
    let (mut i, mut col) = (0, 0);
    while i < nev && col < vectors.ncol() {
        if values[(i, 1)] == 0.0 {
            col += 1;
        } else {
            if col + 1 >= vectors.ncol() {
                return Err(Error::invalid(
                    "too few columns in the packed eigenvector matrix",
                ));
            }
            i += 1;
            col += 2;
        }
        i += 1;
    }
    let mut vec_out = vectors.clone();
    let mut val_out = values.clone();
    igraph_call!(igraph_arpack_unpack_complex(
        &mut vec_out,
        &mut val_out,
        nev as igraph_int_t
    ))?;
    Ok((vec_out, val_out))
}

crate::ffi_enum! {
    /// Error conditions reported by ARPACK (`igraph_arpack_error_t`), see
    /// [`ArpackError::from_error`] and [`arpack_error_to_string`].
    pub enum ArpackError: igraph_arpack_error_t {
        /// No error.
        NoError = igraph_arpack_error_t_IGRAPH_ARPACK_NO_ERROR,
        /// Matrix-vector product failed (not used any more).
        Prod = igraph_arpack_error_t_IGRAPH_ARPACK_PROD,
        /// N must be positive.
        NPos = igraph_arpack_error_t_IGRAPH_ARPACK_NPOS,
        /// NEV must be positive.
        NevNPos = igraph_arpack_error_t_IGRAPH_ARPACK_NEVNPOS,
        /// NCV must be bigger.
        NcvSmall = igraph_arpack_error_t_IGRAPH_ARPACK_NCVSMALL,
        /// Maximum number of iterations should be positive.
        NonPosI = igraph_arpack_error_t_IGRAPH_ARPACK_NONPOSI,
        /// Invalid WHICH parameter.
        WhichInv = igraph_arpack_error_t_IGRAPH_ARPACK_WHICHINV,
        /// Invalid BMAT parameter.
        BmatInv = igraph_arpack_error_t_IGRAPH_ARPACK_BMATINV,
        /// WORKL is too small.
        WorklSmall = igraph_arpack_error_t_IGRAPH_ARPACK_WORKLSMALL,
        /// LAPACK error in tridiagonal eigenvalue calculation.
        TriDErr = igraph_arpack_error_t_IGRAPH_ARPACK_TRIDERR,
        /// Starting vector is zero.
        ZeroStart = igraph_arpack_error_t_IGRAPH_ARPACK_ZEROSTART,
        /// MODE is invalid.
        ModeInv = igraph_arpack_error_t_IGRAPH_ARPACK_MODEINV,
        /// MODE and BMAT are not compatible.
        ModeBmat = igraph_arpack_error_t_IGRAPH_ARPACK_MODEBMAT,
        /// ISHIFT must be 0 or 1.
        IShift = igraph_arpack_error_t_IGRAPH_ARPACK_ISHIFT,
        /// NEV and WHICH='BE' are incompatible.
        NevBe = igraph_arpack_error_t_IGRAPH_ARPACK_NEVBE,
        /// Could not build an Arnoldi factorization.
        NoFact = igraph_arpack_error_t_IGRAPH_ARPACK_NOFACT,
        /// No eigenvalues to sufficient accuracy.
        Failed = igraph_arpack_error_t_IGRAPH_ARPACK_FAILED,
        /// HOWMNY is invalid.
        Howmny = igraph_arpack_error_t_IGRAPH_ARPACK_HOWMNY,
        /// HOWMNY='S' is not implemented.
        HowmnyS = igraph_arpack_error_t_IGRAPH_ARPACK_HOWMNYS,
        /// Different number of converged Ritz values.
        EvDiff = igraph_arpack_error_t_IGRAPH_ARPACK_EVDIFF,
        /// Error from calculation of a real Schur form.
        Shur = igraph_arpack_error_t_IGRAPH_ARPACK_SHUR,
        /// LAPACK (dtrevc) error for calculating eigenvectors.
        Lapack = igraph_arpack_error_t_IGRAPH_ARPACK_LAPACK,
        /// Unknown ARPACK error.
        Unknown = igraph_arpack_error_t_IGRAPH_ARPACK_UNKNOWN,
        /// Maximum number of iterations reached.
        MaxIt = igraph_arpack_error_t_IGRAPH_ARPACK_MAXIT,
        /// No shifts could be applied during a cycle of the implicitly
        /// restarted Arnoldi iteration.
        NoShift = igraph_arpack_error_t_IGRAPH_ARPACK_NOSHIFT,
        /// The Schur form could not be reordered.
        Reorder = igraph_arpack_error_t_IGRAPH_ARPACK_REORDER,
    }
}

impl ArpackError {
    const ALL: [Self; 26] = [
        Self::NoError,
        Self::Prod,
        Self::NPos,
        Self::NevNPos,
        Self::NcvSmall,
        Self::NonPosI,
        Self::WhichInv,
        Self::BmatInv,
        Self::WorklSmall,
        Self::TriDErr,
        Self::ZeroStart,
        Self::ModeInv,
        Self::ModeBmat,
        Self::IShift,
        Self::NevBe,
        Self::NoFact,
        Self::Failed,
        Self::Howmny,
        Self::HowmnyS,
        Self::EvDiff,
        Self::Shur,
        Self::Lapack,
        Self::Unknown,
        Self::MaxIt,
        Self::NoShift,
        Self::Reorder,
    ];

    /// The ARPACK error condition behind an
    /// [`ErrorKind::Arpack`](crate::ErrorKind::Arpack) error, recognized from
    /// its message (igraph reports ARPACK failures with the text of
    /// [`arpack_error_to_string`]); `None` for other errors.
    ///
    /// Unlike [`arpack_last_error`] this is thread-safe: it only looks at the
    /// error value returned by the failed call.
    ///
    /// ```
    /// use igraph::{linalg::*, ErrorKind};
    /// // The Laplacian of a path, started from its eigenvector (1, ..., 1) of
    /// // eigenvalue 0: ARPACK's first Krylov vector is zero.
    /// let n = 6;
    /// let path = move |x: &[f64], y: &mut [f64]| {
    ///     for i in 0..n {
    ///         let deg = if i == 0 || i == n - 1 { 1.0 } else { 2.0 };
    ///         y[i] = deg * x[i];
    ///         if i > 0 {
    ///             y[i] -= x[i - 1];
    ///         }
    ///         if i + 1 < n {
    ///             y[i] -= x[i + 1];
    ///         }
    ///     }
    /// };
    /// let opts = ArpackOptions::default()
    ///     .with_start(vec![1.0; n])
    ///     .with_which(ArpackWhich::SmallestMagnitude);
    /// let err = arpack_rssolve(n, path, &opts, None).unwrap_err();
    /// assert_eq!(err.kind(), ErrorKind::Arpack);
    /// assert_eq!(ArpackError::from_error(&err), Some(ArpackError::ZeroStart));
    /// ```
    pub fn from_error(error: &Error) -> Option<Self> {
        if error.kind() != crate::ErrorKind::Arpack {
            return None;
        }
        Self::ALL
            .into_iter()
            .find(|&e| arpack_error_to_string(e) == error.message())
    }
}

impl fmt::Display for ArpackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&arpack_error_to_string(*self))
    }
}

/// Human readable description of an ARPACK error code
/// (`igraph_arpack_error_to_string`).
///
/// Binds [`igraph_arpack_error_to_string`](https://igraph.org/c/html/latest/igraph-Linalg.html#igraph_arpack_error_to_string).
///
/// ```
/// use igraph::linalg::{arpack_error_to_string, ArpackError};
/// assert_eq!(arpack_error_to_string(ArpackError::NoError), "No error");
/// assert!(!arpack_error_to_string(ArpackError::NcvSmall).is_empty());
/// ```
pub fn arpack_error_to_string(error: ArpackError) -> String {
    let s = unsafe { igraph_arpack_error_to_string(error.into()) };
    if s.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
    }
}

/// The error code of the last ARPACK failure in the process
/// (`igraph_arpack_get_last_error`).
///
/// Prefer [`ArpackError::from_error`], which reads the same information
/// from the returned error in a thread-safe way.
///
/// Binds [`igraph_arpack_get_last_error`](https://igraph.org/c/html/latest/igraph-Linalg.html#igraph_arpack_get_last_error).
///
/// # Safety
///
/// In igraph 1.0.0 and 1.0.1 the last ARPACK error is kept in a plain,
/// process-wide C variable (not thread-local, unlike the rest of igraph's
/// state), written by every failing ARPACK run. The caller must make sure
/// that no other thread runs an ARPACK-based igraph computation
/// concurrently (this includes the eigensolvers and spectral embeddings of
/// this module and ARPACK-based functions of other modules, such as
/// eigenvector centrality); otherwise the read is a data race.
pub unsafe fn arpack_last_error() -> ArpackError {
    let raw = unsafe { igraph_arpack_get_last_error() };
    ArpackError::try_from(raw).unwrap_or(ArpackError::Unknown)
}

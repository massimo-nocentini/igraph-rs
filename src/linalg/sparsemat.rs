//! Sparse matrices (`igraph_sparsemat.h`).

use super::{
    arpack::{ArpackNonSymmetricResult, ArpackOptions, ArpackSymmetricResult},
    check_finite,
};
use crate::{
    error::{Error, Result, catch_panic_or, check, ensure_init},
    ffi::*,
    igraph_call,
    matrix::Matrix,
    vector::{Vector, VectorInt},
};
use std::{
    borrow::Cow,
    ffi::{CStr, c_char, c_void},
    fmt,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Add, Mul, Sub},
    ptr,
};

/// An owned sparse matrix of reals (`igraph_sparsemat_t`), backed by the
/// CXSparse library bundled with igraph.
///
/// A sparse matrix is stored in one of two formats (see [`SparseMatType`]):
///
/// - **triplet** (a.k.a. coordinate) format: a list of `(row, col, value)`
///   entries. It is the format in which matrices are *built*: new entries are
///   appended with [`entry`](Self::entry), and entries at the same position are
///   summed. Constructors such as [`new`](Self::new),
///   [`from_triplets`](Self::from_triplets) and [`from_dense`](Self::from_dense)
///   create triplet matrices.
/// - **column-compressed** (CSC) format: the format in which most computations
///   happen. Convert with [`compress`](Self::compress).
///
/// Most read-only operations accept both formats: when the C function
/// requires a column-compressed matrix, the wrapper compresses a temporary
/// copy. Mutating operations that need the compressed format (such as
/// [`dupl`](Self::dupl) or [`fkeep`](Self::fkeep)) convert `self` in place.
///
/// Row and column indices are zero-based. The matrix owns its storage and frees
/// it on [`Drop`]; it is [`Clone`] (`igraph_sparsemat_init_copy`) and
/// [`Send`]/[`Sync`].
///
/// See the [igraph documentation on sparse
/// matrices](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_init).
///
/// See also the graph matrices of other modules, which convert naturally
/// with [`SparseMat::from_triplets`] (casting the indices between
/// [`VertexId`](crate::VertexId) and `usize`):
/// [`Graph::get_adjacency_sparse`](crate::Graph::get_adjacency_sparse),
/// [`Graph::get_stochastic_sparse`](crate::Graph::get_stochastic_sparse)
/// (conversion) and
/// [`Graph::get_laplacian_sparse`](crate::Graph::get_laplacian_sparse)
/// (structural); in the other direction
/// [`Graph::sparse_adjacency`](crate::Graph::sparse_adjacency) and
/// [`Graph::sparse_weighted_adjacency`](crate::Graph::sparse_weighted_adjacency)
/// (constructors) build a graph from [`triplets`](SparseMat::triplets).
///
/// # Examples
///
/// ```
/// use igraph::linalg::SparseMat;
///
/// // [ 4 1 0 ]
/// // [ 1 3 0 ]
/// // [ 0 0 2 ]
/// let a = SparseMat::from_triplets(3, 3, &[(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0), (2, 2, 2.0)])
///     .unwrap()
///     .compress()
///     .unwrap();
/// assert_eq!(a.shape(), (3, 3));
/// assert_eq!(a.get(0, 1), 1.0);
/// assert_eq!(a.mul_vec(&[1.0, 1.0, 1.0]).unwrap(), vec![5.0, 4.0, 2.0]);
///
/// // Solve A x = b by Cholesky factorization (A is symmetric positive definite).
/// let x = a.cholsol(&[5.0, 4.0, 2.0], igraph::linalg::SparseOrdering::Natural).unwrap();
/// for xi in x {
///     assert!((xi - 1.0).abs() < 1e-12);
/// }
///
/// // Arithmetic with operators returns `Result`s.
/// let twice = (&a + &a).unwrap();
/// assert_eq!(twice.get(0, 0), 8.0);
/// ```
pub type SparseMat = igraph_sparsemat_t;

crate::ffi_enum! {
    /// Storage format of a [`SparseMat`] (`igraph_sparsemat_type_t`).
    pub enum SparseMatType: igraph_sparsemat_type_t {
        /// Triplet (coordinate) format: easy to build, see [`SparseMat::entry`].
        Triplet = igraph_sparsemat_type_t_IGRAPH_SPARSEMAT_TRIPLET,
        /// Column-compressed format: the one used for computations.
        ColumnCompressed = igraph_sparsemat_type_t_IGRAPH_SPARSEMAT_CC,
    }
}

crate::ffi_enum! {
    /// How the linear systems of the shift-and-invert mode of
    /// [`SparseMat::arpack_rssolve`] are solved (`igraph_sparsemat_solve_t`).
    pub enum SparseSolveMethod: igraph_sparsemat_solve_t {
        /// LU decomposition.
        Lu = igraph_sparsemat_solve_t_IGRAPH_SPARSEMAT_SOLVE_LU,
        /// QR decomposition.
        Qr = igraph_sparsemat_solve_t_IGRAPH_SPARSEMAT_SOLVE_QR,
    }
}

/// Fill-reducing ordering used by the sparse factorizations
/// ([`SparseMat::cholsol`], [`SparseMat::lusol`], [`SparseMat::lu`],
/// [`SparseMat::qr`]); the integer `order` argument of the C functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SparseOrdering {
    /// Natural ordering (no permutation), `order = 0`.
    #[default]
    Natural,
    /// Approximate minimum degree ordering of `A + A'`, `order = 1`; the
    /// usual choice for Cholesky and LU of (nearly) symmetric matrices.
    MinDegreeSymmetric,
    /// Minimum degree ordering of `A' A` after removing the dense rows of `A`,
    /// `order = 2`; good for LU of unsymmetric matrices.
    MinDegreeNoDenseRows,
    /// Minimum degree ordering of `A' A`, `order = 3`; the usual choice for QR.
    MinDegreeAtA,
}

impl SparseOrdering {
    fn raw(self) -> igraph_int_t {
        match self {
            Self::Natural => 0,
            Self::MinDegreeSymmetric => 1,
            Self::MinDegreeNoDenseRows => 2,
            Self::MinDegreeAtA => 3,
        }
    }
}

/// The elements of a sparse matrix as returned by
/// [`SparseMat::getelements`] (the raw CXSparse arrays).
#[derive(Debug, Clone, PartialEq)]
pub struct SparseElements {
    /// Row index of each stored element.
    pub i: Vec<i64>,
    /// For a triplet matrix: the column index of each stored element. For a
    /// column-compressed matrix: the column pointers, of length `ncol + 1`;
    /// the elements of column `k` are at positions `j[k]..j[k + 1]` of
    /// [`i`](Self::i) and [`x`](Self::x).
    pub j: Vec<i64>,
    /// The value of each stored element.
    pub x: Vec<f64>,
}

// A sparse matrix uniquely owns its CXSparse storage.
unsafe impl Send for igraph_sparsemat_t {}
unsafe impl Sync for igraph_sparsemat_t {}

impl Drop for igraph_sparsemat_t {
    /// Frees the matrix with `igraph_sparsemat_destroy`.
    fn drop(&mut self) {
        if !self.cs.is_null() {
            unsafe { igraph_sparsemat_destroy(self) };
            self.cs = ptr::null_mut();
        }
    }
}

impl Clone for igraph_sparsemat_t {
    /// Deep copy with `igraph_sparsemat_init_copy`, keeping the storage format.
    fn clone(&self) -> Self {
        Self::init_with(|m| unsafe { igraph_sparsemat_init_copy(m, self) })
            .expect("igraph failed to copy a sparse matrix")
    }
}

impl fmt::Display for igraph_sparsemat_t {
    /// Prints the stored entries in igraph's format, see [`SparseMat::print_to_string`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.print_to_string() {
            Ok(s) => f.write_str(&s),
            Err(_) => Err(fmt::Error),
        }
    }
}

fn check_index(value: usize, bound: usize, what: &str) -> Result<()> {
    if value >= bound {
        Err(Error::invalid(format!(
            "{what} {value} out of bounds (size {bound})"
        )))
    } else {
        Ok(())
    }
}

fn check_len(len: usize, expected: usize, what: &str) -> Result<()> {
    if len != expected {
        Err(Error::invalid(format!(
            "{what} has length {len}, expected {expected}"
        )))
    } else {
        Ok(())
    }
}

fn to_int(value: usize, what: &str) -> Result<igraph_int_t> {
    igraph_int_t::try_from(value)
        .map_err(|_| Error::invalid(format!("{what} ({value}) is too large")))
}

fn check_permutation(p: &[usize], n: usize, what: &str) -> Result<VectorInt> {
    check_len(p.len(), n, what)?;
    let mut seen = vec![false; n];
    for &x in p {
        if x >= n || seen[x] {
            return Err(Error::invalid(format!(
                "{what} is not a permutation of 0..{n}"
            )));
        }
        seen[x] = true;
    }
    Ok(p.iter().map(|&x| x as igraph_int_t).collect())
}

/// A factorized linear solver `b -> A^-1 b`.
type Solver = Box<dyn Fn(&[f64]) -> Result<Vec<f64>>>;

struct FkeepData<F> {
    f: F,
    panicked: bool,
}

unsafe extern "C" fn fkeep_trampoline<F: FnMut(usize, usize, f64) -> bool>(
    row: igraph_int_t,
    col: igraph_int_t,
    value: igraph_real_t,
    extra: *mut c_void,
) -> igraph_int_t {
    let data = unsafe { &mut *(extra as *mut FkeepData<F>) };
    if data.panicked {
        return 1;
    }
    let mut failed = true;
    // The closure runs in its own level of igraph's "finally" stack, so that
    // a failing igraph call made by it cannot free the temporaries of the
    // running `igraph_sparsemat_fkeep` (see `arpack::matvec_trampoline`).
    // SAFETY: the matching EXIT runs below, as `catch_panic_or` never unwinds.
    unsafe { IGRAPH_FINALLY_ENTER() };
    let keep = catch_panic_or(1, || {
        let keep = (data.f)(row as usize, col as usize, value) as igraph_int_t;
        failed = false;
        keep
    });
    unsafe { IGRAPH_FINALLY_EXIT() };
    if failed {
        data.panicked = true;
    }
    keep
}

impl igraph_sparsemat_t {
    /// Runs an igraph function initializing a sparse matrix.
    ///
    /// On failure the partially built matrix is leaked rather than risking a
    /// double free (igraph may already have released it).
    fn init_with(f: impl FnOnce(*mut igraph_sparsemat_t) -> igraph_error_t) -> Result<Self> {
        ensure_init();
        let mut raw = MaybeUninit::<igraph_sparsemat_t>::zeroed();
        check(f(raw.as_mut_ptr()))?;
        let m = unsafe { raw.assume_init() };
        if m.cs.is_null() {
            std::mem::forget(m);
            return Err(Error::new(
                crate::ErrorKind::Failure,
                "igraph returned an empty sparse matrix",
            ));
        }
        Ok(m)
    }

    /// A column-compressed view of `self`: borrowed if already compressed,
    /// otherwise a compressed copy.
    fn cc(&self) -> Result<Cow<'_, SparseMat>> {
        if self.is_cc() {
            Ok(Cow::Borrowed(self))
        } else {
            self.compress().map(Cow::Owned)
        }
    }

    /// Converts `self` to column-compressed format in place, if needed.
    fn make_cc(&mut self) -> Result<()> {
        if self.is_triplet() {
            *self = self.compress()?;
        }
        Ok(())
    }

    /// Creates an empty `nrow` × `ncol` sparse matrix in triplet format
    /// (`igraph_sparsemat_init`), ready to receive entries with
    /// [`entry`](Self::entry).
    ///
    /// Binds [`igraph_sparsemat_init`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_init).
    pub fn new(nrow: usize, ncol: usize) -> Result<Self> {
        Self::with_capacity(nrow, ncol, 0)
    }

    /// Creates an empty `nrow` × `ncol` triplet matrix with room for `nzmax`
    /// entries (`igraph_sparsemat_init`). The capacity is only a hint: the
    /// matrix grows as needed.
    ///
    /// Binds [`igraph_sparsemat_init`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_init).
    pub fn with_capacity(nrow: usize, ncol: usize, nzmax: usize) -> Result<Self> {
        let (r, c, z) = (
            to_int(nrow, "nrow")?,
            to_int(ncol, "ncol")?,
            to_int(nzmax, "nzmax")?,
        );
        Self::init_with(|m| unsafe { igraph_sparsemat_init(m, r, c, z) })
    }

    /// Builds a triplet matrix from `(row, col, value)` entries; entries at the
    /// same position are summed.
    ///
    /// # Errors
    /// If an index is out of bounds.
    pub fn from_triplets(
        nrow: usize,
        ncol: usize,
        triplets: &[(usize, usize, f64)],
    ) -> Result<Self> {
        let mut m = Self::with_capacity(nrow, ncol, triplets.len())?;
        for &(i, j, x) in triplets {
            check_index(i, nrow, "row")?;
            check_index(j, ncol, "column")?;
            m.entry(i, j, x)?;
        }
        Ok(m)
    }

    /// Creates the `n` × `n` diagonal matrix with `value` on the diagonal
    /// (`igraph_sparsemat_init_eye`), in column-compressed format if
    /// `compress` is true, in triplet format otherwise. Time complexity: O(n).
    ///
    /// Binds [`igraph_sparsemat_init_eye`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_init_eye).
    ///
    /// ```
    /// use igraph::linalg::SparseMat;
    /// let i3 = SparseMat::eye(3, 1.0, true).unwrap();
    /// assert!(i3.is_cc());
    /// assert_eq!(i3.to_dense().unwrap().to_rows(), vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]]);
    /// ```
    pub fn eye(n: usize, value: f64, compress: bool) -> Result<Self> {
        let n = to_int(n, "n")?;
        Self::init_with(|m| unsafe { igraph_sparsemat_init_eye(m, n, n, value, compress) })
    }

    /// The `n` × `n` identity matrix, in column-compressed format.
    pub fn identity(n: usize) -> Result<Self> {
        Self::eye(n, 1.0, true)
    }

    /// Creates a diagonal matrix with the given diagonal
    /// (`igraph_sparsemat_init_diag`), column-compressed if `compress` is
    /// true. Time complexity: O(n).
    ///
    /// Binds [`igraph_sparsemat_init_diag`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_init_diag).
    pub fn diag(values: &[f64], compress: bool) -> Result<Self> {
        let v = Vector::view(values);
        let nzmax = to_int(values.len(), "diagonal length")?;
        Self::init_with(|m| unsafe { igraph_sparsemat_init_diag(m, nzmax, v.as_ptr(), compress) })
    }

    /// Converts a dense matrix to a triplet sparse matrix, keeping only the
    /// elements whose absolute value is larger than `tol`
    /// (`igraph_matrix_as_sparsemat`). Time complexity: O(mn).
    ///
    /// Binds [`igraph_matrix_as_sparsemat`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_as_sparsemat).
    ///
    /// ```
    /// use igraph::{linalg::SparseMat, prelude::*};
    /// let m = Matrix::from_rows(&[[1.0, 1e-12], [0.0, 2.0]]).unwrap();
    /// let s = SparseMat::from_dense(&m, 1e-9).unwrap();
    /// assert_eq!(s.nonzero_storage(), 2);
    /// ```
    pub fn from_dense(m: &Matrix, tol: f64) -> Result<Self> {
        Self::init_with(|s| unsafe { igraph_matrix_as_sparsemat(s, m, tol) })
    }

    /// Converts to a dense [`Matrix`] (`igraph_sparsemat_as_matrix`); works
    /// with both formats, duplicate entries are summed. Time complexity: O(mn).
    ///
    /// Binds [`igraph_sparsemat_as_matrix`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_as_matrix).
    pub fn to_dense(&self) -> Result<Matrix> {
        let mut res = Matrix::new();
        igraph_call!(igraph_sparsemat_as_matrix(&mut res, self))?;
        Ok(res)
    }

    /// Changes the capacity (maximum number of stored entries) of the matrix
    /// (`igraph_sparsemat_realloc`). Rarely needed: matrices grow automatically.
    ///
    /// Binds [`igraph_sparsemat_realloc`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_realloc).
    pub fn realloc(&mut self, nzmax: usize) -> Result<()> {
        let z = to_int(nzmax.max(self.nonzero_storage()), "nzmax")?;
        igraph_call!(igraph_sparsemat_realloc(self, z))
    }

    /// Number of rows (`igraph_sparsemat_nrow`).
    pub fn nrow(&self) -> usize {
        unsafe { igraph_sparsemat_nrow(self) as usize }
    }

    /// Number of columns (`igraph_sparsemat_ncol`).
    pub fn ncol(&self) -> usize {
        unsafe { igraph_sparsemat_ncol(self) as usize }
    }

    /// `(nrow, ncol)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.nrow(), self.ncol())
    }

    /// The storage format (`igraph_sparsemat_type`).
    pub fn sparse_type(&self) -> SparseMatType {
        match unsafe { igraph_sparsemat_type(self) } {
            igraph_sparsemat_type_t_IGRAPH_SPARSEMAT_CC => SparseMatType::ColumnCompressed,
            _ => SparseMatType::Triplet,
        }
    }

    /// Whether the matrix is in triplet format (`igraph_sparsemat_is_triplet`).
    pub fn is_triplet(&self) -> bool {
        unsafe { igraph_sparsemat_is_triplet(self) }
    }

    /// Whether the matrix is column-compressed (`igraph_sparsemat_is_cc`).
    pub fn is_cc(&self) -> bool {
        unsafe { igraph_sparsemat_is_cc(self) }
    }

    /// Number of stored entries (`igraph_sparsemat_nonzero_storage`); they may
    /// include zeros and duplicates, see [`dupl`](Self::dupl) and
    /// [`dropzeros`](Self::dropzeros).
    pub fn nonzero_storage(&self) -> usize {
        unsafe { igraph_sparsemat_nonzero_storage(self) as usize }
    }

    /// The allocated capacity for entries (`igraph_sparsemat_nzmax`).
    pub fn nzmax(&self) -> usize {
        unsafe { igraph_sparsemat_nzmax(self) as usize }
    }

    /// Appends an entry to a triplet matrix (`igraph_sparsemat_entry`).
    /// Entries at the same position are summed. Time complexity: O(1)
    /// amortized.
    ///
    /// Binds [`igraph_sparsemat_entry`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_entry).
    ///
    /// # Errors
    /// If the matrix is column-compressed, or the position is out of bounds
    /// (use [`add_rows`](Self::add_rows)/[`add_cols`](Self::add_cols) to grow it).
    pub fn entry(&mut self, row: usize, col: usize, value: f64) -> Result<()> {
        check_index(row, self.nrow(), "row")?;
        check_index(col, self.ncol(), "column")?;
        igraph_call!(igraph_sparsemat_entry(
            self,
            row as igraph_int_t,
            col as igraph_int_t,
            value
        ))
    }

    /// The value at `(row, col)` (`igraph_sparsemat_get`), summing duplicate
    /// entries; zero if nothing is stored there or the position is out of
    /// bounds. Time complexity: O(entries in the column) for a
    /// column-compressed matrix, O(nz) for a triplet matrix.
    ///
    /// For triplet matrices the lookup is done on the Rust side: igraph 1.0.0 and 1.0.1
    /// scans them with its sparse matrix iterator, which reads past the end
    /// of the column index array (see [`iter`](Self::iter)).
    ///
    /// Binds [`igraph_sparsemat_get`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_get).
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row >= self.nrow() || col >= self.ncol() {
            return 0.0;
        }
        if self.is_triplet() {
            return self
                .iter()
                .filter(|&(i, j, _)| i == row && j == col)
                .map(|(_, _, x)| x)
                .sum();
        }
        unsafe { igraph_sparsemat_get(self, row as igraph_int_t, col as igraph_int_t) }
    }

    /// Returns a column-compressed copy of the matrix
    /// (`igraph_sparsemat_compress`); if the matrix is already compressed, a
    /// plain copy. Time complexity: O(nz).
    ///
    /// Binds [`igraph_sparsemat_compress`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_compress).
    pub fn compress(&self) -> Result<SparseMat> {
        if self.is_cc() {
            return Ok(self.clone());
        }
        Self::init_with(|res| unsafe { igraph_sparsemat_compress(self, res) })
    }

    /// The transposed matrix (`igraph_sparsemat_transpose`), in the same
    /// format as `self`.
    ///
    /// In igraph 1.0.0 and 1.0.1 the transpose of a *non-square triplet*
    /// matrix swaps the indices but forgets to swap the dimensions; this
    /// wrapper builds that case entry by entry instead.
    ///
    /// Binds [`igraph_sparsemat_transpose`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_transpose).
    pub fn transpose(&self) -> Result<SparseMat> {
        if self.is_triplet() && self.nrow() != self.ncol() {
            let mut t = Self::with_capacity(self.ncol(), self.nrow(), self.nonzero_storage())?;
            for (i, j, x) in self.iter() {
                t.entry(j, i, x)?;
            }
            return Ok(t);
        }
        Self::init_with(|res| unsafe { igraph_sparsemat_transpose(self, res) })
    }

    /// Whether the matrix is symmetric (`igraph_sparsemat_is_symmetric`);
    /// non-square matrices are not.
    ///
    /// Duplicates are summed first, but the comparison is otherwise
    /// structural and exact: an explicitly stored zero at `(i, j)` without a
    /// stored counterpart at `(j, i)` makes the matrix non-symmetric (use
    /// [`dropzeros`](Self::dropzeros) first), and so do values differing by
    /// rounding errors.
    ///
    /// Binds [`igraph_sparsemat_is_symmetric`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_is_symmetric).
    pub fn is_symmetric(&self) -> Result<bool> {
        let mut res = false;
        igraph_call!(igraph_sparsemat_is_symmetric(self, &mut res))?;
        Ok(res)
    }

    /// Sums duplicate entries of the same position into one
    /// (`igraph_sparsemat_dupl`). A triplet matrix is first converted to
    /// column-compressed format in place.
    ///
    /// Binds [`igraph_sparsemat_dupl`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_dupl).
    pub fn dupl(&mut self) -> Result<()> {
        self.make_cc()?;
        igraph_call!(igraph_sparsemat_dupl(self))
    }

    /// Keeps only the stored entries for which `keep(row, col, value)` returns
    /// true (`igraph_sparsemat_fkeep`). A triplet matrix is first converted
    /// to column-compressed format in place.
    ///
    /// Binds [`igraph_sparsemat_fkeep`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_fkeep).
    ///
    /// ```
    /// use igraph::linalg::SparseMat;
    /// // Keep the upper triangle of a 3x3 matrix of ones.
    /// let mut m = SparseMat::from_dense(&igraph::prelude::Matrix::from_rows(&[[1.0; 3]; 3]).unwrap(), 0.0).unwrap();
    /// m.fkeep(|i, j, _| i <= j).unwrap();
    /// assert_eq!(m.nonzero_storage(), 6);
    /// assert_eq!(m.get(2, 0), 0.0);
    /// ```
    pub fn fkeep<F: FnMut(usize, usize, f64) -> bool>(&mut self, keep: F) -> Result<()> {
        self.make_cc()?;
        let mut data = FkeepData {
            f: keep,
            panicked: false,
        };
        igraph_call!(igraph_sparsemat_fkeep(
            self,
            Some(fkeep_trampoline::<F>),
            &mut data as *mut FkeepData<F> as *mut c_void
        ))
    }

    /// Removes the stored entries that are exactly zero
    /// (`igraph_sparsemat_dropzeros`); a triplet matrix is first compressed
    /// in place.
    ///
    /// Binds [`igraph_sparsemat_dropzeros`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_dropzeros).
    pub fn dropzeros(&mut self) -> Result<()> {
        self.make_cc()?;
        igraph_call!(igraph_sparsemat_dropzeros(self))
    }

    /// Removes the stored entries whose absolute value is at most `tol`
    /// (`igraph_sparsemat_droptol`); a triplet matrix is first compressed in
    /// place.
    ///
    /// Binds [`igraph_sparsemat_droptol`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_droptol).
    pub fn droptol(&mut self, tol: f64) -> Result<()> {
        self.make_cc()?;
        igraph_call!(igraph_sparsemat_droptol(self, tol))
    }

    /// Matrix product `self * other` (`igraph_sparsemat_multiply`), a
    /// column-compressed matrix. Also available as `&a * &b`.
    ///
    /// Binds [`igraph_sparsemat_multiply`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_multiply).
    ///
    /// # Errors
    /// If the inner dimensions do not match.
    pub fn multiply(&self, other: &SparseMat) -> Result<SparseMat> {
        if self.ncol() != other.nrow() {
            return Err(Error::invalid(format!(
                "cannot multiply a {:?} matrix by a {:?} matrix",
                self.shape(),
                other.shape()
            )));
        }
        let (a, b) = (self.cc()?, other.cc()?);
        Self::init_with(|res| unsafe { igraph_sparsemat_multiply(&*a, &*b, res) })
    }

    /// Linear combination `alpha * self + beta * other`
    /// (`igraph_sparsemat_add`), a column-compressed matrix. `&a + &b` and
    /// `&a - &b` are shorthands.
    ///
    /// Binds [`igraph_sparsemat_add`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_add).
    ///
    /// # Errors
    /// If the shapes differ.
    pub fn add(&self, other: &SparseMat, alpha: f64, beta: f64) -> Result<SparseMat> {
        if self.shape() != other.shape() {
            return Err(Error::invalid(format!(
                "cannot add a {:?} matrix and a {:?} matrix",
                self.shape(),
                other.shape()
            )));
        }
        let (a, b) = (self.cc()?, other.cc()?);
        Self::init_with(|res| unsafe { igraph_sparsemat_add(&*a, &*b, alpha, beta, res) })
    }

    /// Computes `y + A x` (`igraph_sparsemat_gaxpy`, "generalized A x plus y").
    ///
    /// Binds [`igraph_sparsemat_gaxpy`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_gaxpy).
    ///
    /// # Errors
    /// If `x.len() != ncol` or `y.len() != nrow`.
    pub fn gaxpy(&self, x: &[f64], y: &[f64]) -> Result<Vec<f64>> {
        check_len(x.len(), self.ncol(), "x")?;
        check_len(y.len(), self.nrow(), "y")?;
        let a = self.cc()?;
        let xv = Vector::view(x);
        let mut res = Vector::from_slice(y);
        igraph_call!(igraph_sparsemat_gaxpy(&*a, xv.as_ptr(), &mut res))?;
        Ok(res.into())
    }

    /// The matrix-vector product `A x` (via [`gaxpy`](Self::gaxpy)).
    pub fn mul_vec(&self, x: &[f64]) -> Result<Vec<f64>> {
        self.gaxpy(x, &vec![0.0; self.nrow()])
    }

    /// Makes a column-compressed, duplicate-free, row-sorted copy suitable
    /// for the triangular solvers, and checks that it is triangular with a
    /// stored diagonal.
    fn triangular(&self, lower: bool, what: &str) -> Result<SparseMat> {
        let (n, m) = self.shape();
        if n != m {
            return Err(Error::invalid(format!(
                "{what} needs a square matrix, got {:?}",
                self.shape()
            )));
        }
        let mut t = self.sort()?;
        t.dupl()?;
        let t = t.sort()?;
        let e = t.getelements()?;
        for col in 0..n {
            let (start, end) = (e.j[col] as usize, e.j[col + 1] as usize);
            let diag = if lower { start } else { end.wrapping_sub(1) };
            if start == end || e.i[diag] as usize != col {
                return Err(Error::invalid(format!(
                    "{what}: diagonal element {col} is not stored (the matrix must be triangular with a stored diagonal)"
                )));
            }
            let ok = e.i[start..end].iter().all(|&r| {
                if lower {
                    r as usize >= col
                } else {
                    r as usize <= col
                }
            });
            if !ok {
                return Err(Error::invalid(format!(
                    "{what}: the matrix is not {} triangular",
                    if lower { "lower" } else { "upper" }
                )));
            }
        }
        Ok(t)
    }

    fn solve_with(
        &self,
        b: &[f64],
        lower: bool,
        what: &str,
        f: unsafe extern "C" fn(
            *const igraph_sparsemat_t,
            *const igraph_vector_t,
            *mut igraph_vector_t,
        ) -> igraph_error_t,
    ) -> Result<Vec<f64>> {
        check_len(b.len(), self.nrow(), "b")?;
        let t = self.triangular(lower, what)?;
        let bv = Vector::view(b);
        let mut res = Vector::new();
        igraph_call!(f(&t, bv.as_ptr(), &mut res))?;
        Ok(res.into())
    }

    /// Solves the lower triangular system `L x = b` (`igraph_sparsemat_lsolve`).
    ///
    /// The matrix must be square, lower triangular, and have all its diagonal
    /// elements stored (a zero on the diagonal gives infinite or NaN values).
    ///
    /// Binds [`igraph_sparsemat_lsolve`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_lsolve).
    pub fn lsolve(&self, b: &[f64]) -> Result<Vec<f64>> {
        self.solve_with(b, true, "lsolve", igraph_sparsemat_lsolve)
    }

    /// Solves `L' x = b` where `L` (this matrix) is lower triangular
    /// (`igraph_sparsemat_ltsolve`); requirements as in [`lsolve`](Self::lsolve).
    ///
    /// Binds [`igraph_sparsemat_ltsolve`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_ltsolve).
    pub fn ltsolve(&self, b: &[f64]) -> Result<Vec<f64>> {
        self.solve_with(b, true, "ltsolve", igraph_sparsemat_ltsolve)
    }

    /// Solves the upper triangular system `U x = b` (`igraph_sparsemat_usolve`).
    ///
    /// The matrix must be square, upper triangular, with a stored diagonal.
    ///
    /// Binds [`igraph_sparsemat_usolve`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_usolve).
    pub fn usolve(&self, b: &[f64]) -> Result<Vec<f64>> {
        self.solve_with(b, false, "usolve", igraph_sparsemat_usolve)
    }

    /// Solves `U' x = b` where `U` (this matrix) is upper triangular
    /// (`igraph_sparsemat_utsolve`); requirements as in [`usolve`](Self::usolve).
    ///
    /// Binds [`igraph_sparsemat_utsolve`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_utsolve).
    pub fn utsolve(&self, b: &[f64]) -> Result<Vec<f64>> {
        self.solve_with(b, false, "utsolve", igraph_sparsemat_utsolve)
    }

    /// Solves `A x = b` for a symmetric positive definite `A` via a sparse
    /// Cholesky factorization (`igraph_sparsemat_cholsol`). Only the upper
    /// triangular part of `A` is used.
    ///
    /// Binds [`igraph_sparsemat_cholsol`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_cholsol).
    ///
    /// # Errors
    /// If `A` is not square, `b` has the wrong length, or `A` is not positive
    /// definite.
    pub fn cholsol(&self, b: &[f64], order: SparseOrdering) -> Result<Vec<f64>> {
        self.check_square_rhs(b, "cholsol")?;
        let a = self.cc()?;
        let bv = Vector::view(b);
        let mut res = Vector::new();
        igraph_call!(igraph_sparsemat_cholsol(
            &*a,
            bv.as_ptr(),
            &mut res,
            order.raw()
        ))?;
        Ok(res.into())
    }

    /// Solves `A x = b` via a sparse LU factorization
    /// (`igraph_sparsemat_lusol`). `tol` is the partial pivoting threshold:
    /// `1.0` means classic partial pivoting, smaller values (e.g. `0.001`)
    /// favor sparsity, together with a fill-reducing `order`.
    ///
    /// Binds [`igraph_sparsemat_lusol`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_lusol).
    ///
    /// # Errors
    /// If `A` is not square, `b` has the wrong length or `A` is singular.
    pub fn lusol(&self, b: &[f64], order: SparseOrdering, tol: f64) -> Result<Vec<f64>> {
        self.check_square_rhs(b, "lusol")?;
        let a = self.cc()?;
        let bv = Vector::view(b);
        let mut res = Vector::new();
        igraph_call!(igraph_sparsemat_lusol(
            &*a,
            bv.as_ptr(),
            &mut res,
            order.raw(),
            tol
        ))?;
        Ok(res.into())
    }

    fn check_square_rhs(&self, b: &[f64], what: &str) -> Result<()> {
        if self.nrow() != self.ncol() {
            return Err(Error::invalid(format!(
                "{what} needs a square matrix, got {:?}",
                self.shape()
            )));
        }
        check_len(b.len(), self.nrow(), "b")
    }

    /// Prints the stored entries into a string (`igraph_sparsemat_print`).
    ///
    /// Triplet matrices print one `row col : value` line per entry;
    /// column-compressed ones print, for each column, a `col j: locations a
    /// to b` header followed by `row : value` lines. Also used by the
    /// [`Display`](fmt::Display) implementation.
    ///
    /// Binds [`igraph_sparsemat_print`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_print).
    pub fn print_to_string(&self) -> Result<String> {
        ensure_init();
        let mut buf: *mut c_char = ptr::null_mut();
        let mut size: usize = 0;
        let stream = unsafe { open_memstream(&mut buf, &mut size) };
        if stream.is_null() {
            return Err(Error::new(
                crate::ErrorKind::File,
                "cannot open a memory stream",
            ));
        }
        let res = check(unsafe { igraph_sparsemat_print(self, stream) });
        unsafe { fclose(stream) };
        let out = if buf.is_null() {
            String::new()
        } else {
            let s = unsafe { CStr::from_ptr(buf) }
                .to_string_lossy()
                .into_owned();
            unsafe { free(buf as *mut c_void) };
            s
        };
        res.map(|()| out)
    }

    /// Permutes rows and columns (`igraph_sparsemat_permute`): row `i` of the
    /// result is row `p[i]` of `self`, and column `j` of the result is column
    /// `q[j]` of `self`. The result is column-compressed. Time complexity:
    /// O(m + n + nz).
    ///
    /// Binds [`igraph_sparsemat_permute`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_permute).
    ///
    /// # Errors
    /// If `p` (resp. `q`) is not a permutation of `0..nrow` (resp. `0..ncol`).
    pub fn permute(&self, p: &[usize], q: &[usize]) -> Result<SparseMat> {
        let pv = check_permutation(p, self.nrow(), "row permutation")?;
        let qv = check_permutation(q, self.ncol(), "column permutation")?;
        let a = self.cc()?;
        Self::init_with(|res| unsafe { igraph_sparsemat_permute(&*a, &pv, &qv, res) })
    }

    /// Extracts the submatrix made of the given rows and columns
    /// (`igraph_sparsemat_index`); `None` selects all rows (resp. columns).
    /// Indices may repeat. The result is column-compressed.
    ///
    /// Binds [`igraph_sparsemat_index`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_index).
    ///
    /// # Errors
    /// If an index is out of bounds.
    pub fn index(&self, rows: Option<&[usize]>, cols: Option<&[usize]>) -> Result<SparseMat> {
        let conv = |idx: &[usize], bound: usize, what: &str| -> Result<VectorInt> {
            for &x in idx {
                check_index(x, bound, what)?;
            }
            Ok(idx.iter().map(|&x| x as igraph_int_t).collect())
        };
        let a = self.cc()?;
        let all_rows: Vec<usize>;
        let rows = match (rows, cols) {
            // igraph needs at least one index vector.
            (None, None) => {
                all_rows = (0..self.nrow()).collect();
                Some(&all_rows[..])
            }
            (r, _) => r,
        };
        let p = rows.map(|r| conv(r, self.nrow(), "row")).transpose()?;
        let q = cols.map(|c| conv(c, self.ncol(), "column")).transpose()?;
        let pp = p.as_ref().map_or(ptr::null(), |v| v as *const VectorInt);
        let qp = q.as_ref().map_or(ptr::null(), |v| v as *const VectorInt);
        Self::init_with(|res| unsafe { igraph_sparsemat_index(&*a, pp, qp, res, ptr::null_mut()) })
    }

    /// Computes the symbolic analysis and numeric LU factorization of a
    /// square matrix (`igraph_sparsemat_symblu` + `igraph_sparsemat_lu`), to
    /// solve many systems with the same coefficient matrix via
    /// [`SparseLu::solve`]. `tol` is the partial pivoting threshold, as in
    /// [`lusol`](Self::lusol).
    ///
    /// Binds [`igraph_sparsemat_symblu`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_symblu)
    /// and [`igraph_sparsemat_lu`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_lu).
    ///
    /// ```
    /// use igraph::linalg::{SparseMat, SparseOrdering};
    /// let a = SparseMat::from_triplets(2, 2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)]).unwrap();
    /// let lu = a.lu(SparseOrdering::Natural, 1.0).unwrap();
    /// let x = lu.solve(&[3.0, 4.0]).unwrap();
    /// assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12);
    /// ```
    ///
    /// # Errors
    /// If the matrix is not square or is singular.
    pub fn lu(&self, order: SparseOrdering, tol: f64) -> Result<SparseLu> {
        let n = self.square_dim("LU decomposition")?;
        let a = self.cc()?;
        let symbolic =
            SymbolicGuard::new(|s| unsafe { igraph_sparsemat_symblu(order.raw(), &*a, s) })?;
        let numeric =
            NumericGuard::new(|d| unsafe { igraph_sparsemat_lu(&*a, &symbolic.0, d, tol) })?;
        Ok(SparseLu {
            symbolic,
            numeric,
            n,
        })
    }

    /// Computes the symbolic analysis and numeric QR factorization of a
    /// square matrix (`igraph_sparsemat_symbqr` + `igraph_sparsemat_qr`), to
    /// solve many systems with the same coefficient matrix via
    /// [`SparseQr::solve`].
    ///
    /// Binds [`igraph_sparsemat_symbqr`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_symbqr)
    /// and [`igraph_sparsemat_qr`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_qr).
    ///
    /// # Errors
    /// If the matrix is not square or the factorization fails.
    pub fn qr(&self, order: SparseOrdering) -> Result<SparseQr> {
        let n = self.square_dim("QR decomposition")?;
        let a = self.cc()?;
        let symbolic =
            SymbolicGuard::new(|s| unsafe { igraph_sparsemat_symbqr(order.raw(), &*a, s) })?;
        let numeric = NumericGuard::new(|d| unsafe { igraph_sparsemat_qr(&*a, &symbolic.0, d) })?;
        Ok(SparseQr {
            symbolic,
            numeric,
            n,
        })
    }

    fn square_dim(&self, what: &str) -> Result<usize> {
        let (n, m) = self.shape();
        if n != m || n == 0 {
            return Err(Error::invalid(format!(
                "{what} needs a non-empty square matrix, got {:?}",
                self.shape()
            )));
        }
        Ok(n)
    }

    /// Eigenvalues and eigenvectors of a *symmetric* sparse matrix with
    /// ARPACK (`igraph_sparsemat_arpack_rssolve`).
    ///
    /// With [`ArpackMode::Regular`](super::ArpackMode::Regular) ARPACK works
    /// with products `A x` (this is igraph's driver). With
    /// [`ArpackMode::ShiftInvert`](super::ArpackMode::ShiftInvert)`{ sigma }`
    /// it works with `(A - sigma I)^-1 x`, computed by factorizing `A - sigma
    /// I` once with the given `method` (an LU decomposition with partial
    /// pivoting, or a QR decomposition); combined with
    /// [`ArpackWhich::LargestMagnitude`](super::ArpackWhich::LargestMagnitude)
    /// this finds the eigenvalues closest to `sigma`. `method` is ignored in
    /// regular mode.
    ///
    /// As with [`arpack_rssolve`](super::arpack_rssolve), 2 × 2 matrices in
    /// regular mode are solved in closed form, with the eigenpairs selected
    /// and normalized on the Rust side.
    ///
    /// The shift-and-invert mode is driven from Rust (with
    /// [`lu`](Self::lu)/[`qr`](Self::qr) and [`arpack_rssolve`](super::arpack_rssolve)):
    /// igraph's own driver (1.0.0 and 1.0.1) factorizes without pivoting, which breaks down —
    /// and makes ARPACK abort the process — whenever `A - sigma I` has a zero
    /// on the diagonal.
    ///
    /// Binds [`igraph_sparsemat_arpack_rssolve`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_arpack_rssolve).
    ///
    /// # Errors
    /// If the matrix is not square, contains non-finite values, the options
    /// are invalid, `A - sigma I` is singular, or ARPACK fails.
    pub fn arpack_rssolve(
        &self,
        options: &ArpackOptions,
        method: SparseSolveMethod,
    ) -> Result<ArpackSymmetricResult> {
        let n = self.square_dim("ARPACK")?;
        let a = self.cc()?;
        check_finite(&a.getelements()?.x, "the matrix")?;
        match options.mode {
            super::ArpackMode::Regular => {
                let mut raw = options.to_raw(n, true)?;
                let shortcut = super::arpack::uses_2x2_shortcut(n, options);
                if shortcut {
                    raw.nev = 2;
                }
                let mut values = Vector::new();
                let mut vectors = options.start_matrix(n)?;
                let _arpack = super::arpack::ArpackGuard::enter()?;
                igraph_call!(igraph_sparsemat_arpack_rssolve(
                    &*a,
                    &mut raw,
                    ptr::null_mut(),
                    &mut values,
                    &mut vectors,
                    method.into()
                ))?;
                let res = ArpackSymmetricResult::from_raw(values, vectors, &raw);
                if shortcut {
                    res.select_2x2(options.which, options.nev)
                } else {
                    Ok(res)
                }
            }
            super::ArpackMode::ShiftInvert { sigma } => {
                if !sigma.is_finite() {
                    return Err(Error::invalid("the shift must be finite"));
                }
                // Validate the options before factorizing.
                options.to_raw(n, true)?;
                let shifted = a.add(&SparseMat::identity(n)?, 1.0, -sigma)?;
                let solver: Solver = match method {
                    SparseSolveMethod::Lu => {
                        let lu = shifted.lu(SparseOrdering::MinDegreeSymmetric, 1.0)?;
                        Box::new(move |b| lu.solve(b))
                    }
                    SparseSolveMethod::Qr => {
                        let qr = shifted.qr(SparseOrdering::MinDegreeAtA)?;
                        Box::new(move |b| qr.solve(b))
                    }
                };
                let mut failure = None;
                let res = super::arpack_rssolve(
                    n,
                    |x: &[f64], y: &mut [f64]| match solver(x) {
                        Ok(v) => y.copy_from_slice(&v),
                        Err(e) => {
                            failure = Some(e);
                            // Stops ARPACK (see `arpack_rssolve`).
                            y.fill(f64::NAN);
                        }
                    },
                    options,
                    None,
                );
                match failure {
                    Some(e) => Err(e),
                    None => res,
                }
            }
        }
    }

    /// Eigenvalues and eigenvectors of a general (non-symmetric) sparse
    /// matrix with ARPACK (`igraph_sparsemat_arpack_rnsolve`). Only
    /// [`ArpackMode::Regular`](super::ArpackMode::Regular) is supported.
    ///
    /// Binds [`igraph_sparsemat_arpack_rnsolve`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_arpack_rnsolve).
    pub fn arpack_rnsolve(&self, options: &ArpackOptions) -> Result<ArpackNonSymmetricResult> {
        let n = self.square_dim("ARPACK")?;
        if options.mode != super::ArpackMode::Regular {
            return Err(Error::invalid(
                "the sparse non-symmetric ARPACK solver supports only the regular mode",
            ));
        }
        let a = self.cc()?;
        check_finite(&a.getelements()?.x, "the matrix")?;
        let mut raw = options.to_raw(n, false)?;
        let mut values = Matrix::new();
        let mut vectors = options.start_matrix(n)?;
        let _arpack = super::arpack::ArpackGuard::enter()?;
        igraph_call!(igraph_sparsemat_arpack_rnsolve(
            &*a,
            &mut raw,
            ptr::null_mut(),
            &mut values,
            &mut vectors
        ))?;
        ArpackNonSymmetricResult::from_raw(values, vectors, &raw, options.nev)
    }

    fn stored_values(&mut self) -> Result<Vec<f64>> {
        self.dupl()?;
        Ok(self.getelements()?.x)
    }

    /// The largest *stored* value, after summing duplicates
    /// (`igraph_sparsemat_max`); `-inf` if nothing is stored. Implicit zeros
    /// are not considered. A triplet matrix is compressed in place.
    ///
    /// In igraph 1.0.0 and 1.0.1, `igraph_sparsemat_max` skips the last stored element, so
    /// this wrapper computes the maximum over the entries it reads with
    /// [`getelements`](Self::getelements) after `igraph_sparsemat_dupl`.
    pub fn max(&mut self) -> Result<f64> {
        Ok(self
            .stored_values()?
            .into_iter()
            .fold(f64::NEG_INFINITY, f64::max))
    }

    /// The smallest *stored* value, after summing duplicates
    /// (`igraph_sparsemat_min`); `+inf` if nothing is stored. See
    /// [`max`](Self::max) for the caveats.
    pub fn min(&mut self) -> Result<f64> {
        Ok(self
            .stored_values()?
            .into_iter()
            .fold(f64::INFINITY, f64::min))
    }

    /// `(min, max)` of the stored values (`igraph_sparsemat_minmax`), see
    /// [`max`](Self::max).
    pub fn minmax(&mut self) -> Result<(f64, f64)> {
        let v = self.stored_values()?;
        Ok((
            v.iter().copied().fold(f64::INFINITY, f64::min),
            v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ))
    }

    /// Number of stored entries that are not zero, after summing duplicates
    /// (`igraph_sparsemat_count_nonzero`). A triplet matrix is compressed in
    /// place.
    ///
    /// Binds [`igraph_sparsemat_count_nonzero`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_count_nonzero).
    pub fn count_nonzero(&mut self) -> Result<usize> {
        self.dupl()?;
        Ok(unsafe { igraph_sparsemat_count_nonzero(self) } as usize)
    }

    /// Number of stored entries whose absolute value exceeds `tol`, after
    /// summing duplicates (`igraph_sparsemat_count_nonzerotol`).
    ///
    /// Binds [`igraph_sparsemat_count_nonzerotol`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_count_nonzerotol).
    pub fn count_nonzerotol(&mut self, tol: f64) -> Result<usize> {
        self.dupl()?;
        Ok(unsafe { igraph_sparsemat_count_nonzerotol(self, tol) } as usize)
    }

    /// Row sums (`igraph_sparsemat_rowsums`). Time complexity: O(nz).
    ///
    /// Binds [`igraph_sparsemat_rowsums`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_rowsums).
    pub fn rowsums(&self) -> Result<Vec<f64>> {
        let mut res = Vector::new();
        igraph_call!(igraph_sparsemat_rowsums(self, &mut res))?;
        Ok(res.into())
    }

    /// Column sums (`igraph_sparsemat_colsums`).
    ///
    /// Binds [`igraph_sparsemat_colsums`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_colsums).
    pub fn colsums(&self) -> Result<Vec<f64>> {
        let mut res = Vector::new();
        igraph_call!(igraph_sparsemat_colsums(self, &mut res))?;
        Ok(res.into())
    }

    fn reduce(
        &mut self,
        f: unsafe extern "C" fn(*mut igraph_sparsemat_t, *mut igraph_vector_t) -> igraph_error_t,
    ) -> Result<Vec<f64>> {
        self.make_cc()?;
        let mut res = Vector::new();
        igraph_call!(f(self, &mut res))?;
        Ok(res.into())
    }

    /// Minimum of the *stored* values of each row (`igraph_sparsemat_rowmins`);
    /// `+inf` for rows without stored values. Implicit zeros are not
    /// considered. A triplet matrix is compressed in place first, and
    /// duplicate entries are summed.
    pub fn rowmins(&mut self) -> Result<Vec<f64>> {
        self.reduce(igraph_sparsemat_rowmins)
    }

    /// Minimum of the stored values of each column (`igraph_sparsemat_colmins`),
    /// see [`rowmins`](Self::rowmins).
    pub fn colmins(&mut self) -> Result<Vec<f64>> {
        self.reduce(igraph_sparsemat_colmins)
    }

    /// Maximum of the stored values of each row (`igraph_sparsemat_rowmaxs`);
    /// `-inf` for rows without stored values.
    pub fn rowmaxs(&mut self) -> Result<Vec<f64>> {
        self.reduce(igraph_sparsemat_rowmaxs)
    }

    /// Maximum of the stored values of each column (`igraph_sparsemat_colmaxs`),
    /// see [`rowmaxs`](Self::rowmaxs).
    pub fn colmaxs(&mut self) -> Result<Vec<f64>> {
        self.reduce(igraph_sparsemat_colmaxs)
    }

    fn which_min(
        &mut self,
        f: unsafe extern "C" fn(
            *mut igraph_sparsemat_t,
            *mut igraph_vector_t,
            *mut igraph_vector_int_t,
        ) -> igraph_error_t,
    ) -> Result<(Vec<f64>, Vec<usize>)> {
        self.make_cc()?;
        let mut res = Vector::new();
        let mut pos = VectorInt::new();
        igraph_call!(f(self, &mut res, &mut pos))?;
        Ok((res.into(), pos.iter().map(|&p| p as usize).collect()))
    }

    /// For each row, the minimum stored value and the column where it is
    /// found (`igraph_sparsemat_which_min_rows`); rows without stored values
    /// give `(+inf, 0)`. A triplet matrix is compressed in place first.
    pub fn which_min_rows(&mut self) -> Result<(Vec<f64>, Vec<usize>)> {
        self.which_min(igraph_sparsemat_which_min_rows)
    }

    /// For each column, the minimum stored value and the row where it is
    /// found (`igraph_sparsemat_which_min_cols`); columns without stored
    /// values give `(+inf, 0)`. A triplet matrix is compressed in place first.
    pub fn which_min_cols(&mut self) -> Result<(Vec<f64>, Vec<usize>)> {
        self.which_min(igraph_sparsemat_which_min_cols)
    }

    /// Multiplies every element by `by` (`igraph_sparsemat_scale`). Time
    /// complexity: O(nz).
    ///
    /// Binds [`igraph_sparsemat_scale`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_scale).
    pub fn scale(&mut self, by: f64) -> Result<()> {
        igraph_call!(igraph_sparsemat_scale(self, by))
    }

    /// Multiplies row `i` by `factors[i]` (`igraph_sparsemat_scale_rows`),
    /// i.e. computes `diag(factors) * A`.
    ///
    /// # Errors
    /// If `factors.len() != nrow`.
    pub fn scale_rows(&mut self, factors: &[f64]) -> Result<()> {
        check_len(factors.len(), self.nrow(), "row factors")?;
        let f = Vector::view(factors);
        igraph_call!(igraph_sparsemat_scale_rows(self, f.as_ptr()))
    }

    /// Multiplies column `j` by `factors[j]` (`igraph_sparsemat_scale_cols`),
    /// i.e. computes `A * diag(factors)`.
    ///
    /// # Errors
    /// If `factors.len() != ncol`.
    pub fn scale_cols(&mut self, factors: &[f64]) -> Result<()> {
        check_len(factors.len(), self.ncol(), "column factors")?;
        let f = Vector::view(factors);
        igraph_call!(igraph_sparsemat_scale_cols(self, f.as_ptr()))
    }

    /// Negates every element in place (`igraph_sparsemat_neg`).
    pub fn neg(&mut self) -> Result<()> {
        igraph_call!(igraph_sparsemat_neg(self))
    }

    /// Appends `n` zero rows (`igraph_sparsemat_add_rows`). Time complexity: O(1).
    ///
    /// Binds [`igraph_sparsemat_add_rows`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_add_rows).
    pub fn add_rows(&mut self, n: usize) -> Result<()> {
        let total = self
            .nrow()
            .checked_add(n)
            .ok_or_else(|| Error::invalid("number of rows overflows"))?;
        to_int(total, "number of rows")?;
        igraph_call!(igraph_sparsemat_add_rows(self, n as igraph_int_t))
    }

    /// Appends `n` zero columns (`igraph_sparsemat_add_cols`).
    ///
    /// Binds [`igraph_sparsemat_add_cols`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_add_cols).
    pub fn add_cols(&mut self, n: usize) -> Result<()> {
        let total = self
            .ncol()
            .checked_add(n)
            .ok_or_else(|| Error::invalid("number of columns overflows"))?;
        to_int(total, "number of columns")?;
        igraph_call!(igraph_sparsemat_add_cols(self, n as igraph_int_t))
    }

    /// Resizes to `nrow` × `ncol` and **removes all the entries**
    /// (`igraph_sparsemat_resize`); the result is an empty triplet matrix with
    /// room for `nzmax` entries.
    ///
    /// Binds [`igraph_sparsemat_resize`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_resize).
    pub fn resize(&mut self, nrow: usize, ncol: usize, nzmax: usize) -> Result<()> {
        let (r, c, z) = (
            to_int(nrow, "nrow")?,
            to_int(ncol, "ncol")?,
            to_int(nzmax.max(1), "nzmax")?,
        );
        igraph_call!(igraph_sparsemat_resize(self, r, c, z))
    }

    /// The raw stored elements (`igraph_sparsemat_getelements`), see
    /// [`SparseElements`] for the layout, which depends on the format.
    ///
    /// Binds [`igraph_sparsemat_getelements`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_getelements).
    pub fn getelements(&self) -> Result<SparseElements> {
        let (mut i, mut j, mut x) = (VectorInt::new(), VectorInt::new(), Vector::new());
        igraph_call!(igraph_sparsemat_getelements(self, &mut i, &mut j, &mut x))?;
        Ok(SparseElements {
            i: i.into(),
            j: j.into(),
            x: x.into(),
        })
    }

    /// Like [`getelements`](Self::getelements), with the elements sorted by
    /// column, then by row (`igraph_sparsemat_getelements_sorted`).
    ///
    /// Binds [`igraph_sparsemat_getelements_sorted`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_getelements_sorted).
    pub fn getelements_sorted(&self) -> Result<SparseElements> {
        let (mut i, mut j, mut x) = (VectorInt::new(), VectorInt::new(), Vector::new());
        igraph_call!(igraph_sparsemat_getelements_sorted(
            self, &mut i, &mut j, &mut x
        ))?;
        Ok(SparseElements {
            i: i.into(),
            j: j.into(),
            x: x.into(),
        })
    }

    /// A copy whose entries are sorted by column, then by row
    /// (`igraph_sparsemat_sort`), in the same format as `self`.
    ///
    /// Binds [`igraph_sparsemat_sort`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_sort).
    pub fn sort(&self) -> Result<SparseMat> {
        Self::init_with(|res| unsafe { igraph_sparsemat_sort(self, res) })
    }

    /// The dense product `self * b` of this sparse matrix with a dense matrix
    /// (`igraph_sparsemat_multiply_by_dense`).
    ///
    /// # Errors
    /// If `b.nrow() != self.ncol()`.
    pub fn multiply_by_dense(&self, b: &Matrix) -> Result<Matrix> {
        if b.nrow() != self.ncol() {
            return Err(Error::invalid(
                "invalid dimensions in sparse-dense matrix product",
            ));
        }
        let a = self.cc()?;
        let mut res = Matrix::new();
        igraph_call!(igraph_sparsemat_multiply_by_dense(&*a, b, &mut res))?;
        Ok(res)
    }

    /// Divides each column by its sum (`igraph_sparsemat_normalize_cols`),
    /// making the matrix column-stochastic. The matrix must be square
    /// (igraph sizes the sums by the number of rows).
    ///
    /// # Errors
    /// If a column sums to zero and `allow_zeros` is false, or the matrix is
    /// not square.
    pub fn normalize_cols(&mut self, allow_zeros: bool) -> Result<()> {
        if self.nrow() != self.ncol() {
            return Err(Error::invalid("normalize_cols needs a square matrix"));
        }
        igraph_call!(igraph_sparsemat_normalize_cols(self, allow_zeros))
    }

    /// Divides each row by its sum (`igraph_sparsemat_normalize_rows`),
    /// making the matrix row-stochastic.
    ///
    /// # Errors
    /// If a row sums to zero and `allow_zeros` is false.
    pub fn normalize_rows(&mut self, allow_zeros: bool) -> Result<()> {
        igraph_call!(igraph_sparsemat_normalize_rows(self, allow_zeros))
    }

    /// Iterates over the stored entries as `(row, col, value)`, in storage
    /// order (`igraph_sparsemat_iterator_*`); duplicates are not merged.
    ///
    /// Column-compressed matrices are walked with igraph's iterator. For
    /// triplet matrices, `igraph_sparsemat_iterator_next` of igraph 1.0.0 and 1.0.1 reads
    /// the column array as if it had `ncol + 1` entries (it has `nzmax`), an
    /// out-of-bounds read as soon as `nzmax <= ncol`; the iterator then walks
    /// a copy of the entries made with
    /// [`getelements`](Self::getelements) instead.
    ///
    /// # Panics
    /// If copying the entries of a triplet matrix runs out of memory.
    ///
    /// ```
    /// use igraph::linalg::SparseMat;
    /// let m = SparseMat::diag(&[1.0, 2.0, 3.0], true).unwrap();
    /// let entries: Vec<_> = m.iter().collect();
    /// assert_eq!(entries, vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
    /// ```
    pub fn iter(&self) -> SparseMatIter<'_> {
        SparseMatIter::new(self)
    }

    /// The stored entries as `(row, col, value)` triplets, in storage order.
    pub fn triplets(&self) -> Vec<(usize, usize, f64)> {
        self.iter().collect()
    }
}

/// The dense product `a * b` of a dense matrix with a sparse one
/// (`igraph_sparsemat_dense_multiply`).
///
/// # Errors
/// If `a.ncol() != b.nrow()`.
///
/// ```
/// use igraph::{linalg::{SparseMat, dense_multiply}, prelude::*};
/// let a = Matrix::from_rows(&[[1.0, 2.0]]).unwrap();
/// let b = SparseMat::eye(2, 3.0, true).unwrap();
/// assert_eq!(dense_multiply(&a, &b).unwrap().to_rows(), vec![vec![3.0, 6.0]]);
/// ```
pub fn dense_multiply(a: &Matrix, b: &SparseMat) -> Result<Matrix> {
    if a.ncol() != b.nrow() {
        return Err(Error::invalid(
            "invalid dimensions in dense-sparse matrix product",
        ));
    }
    let bc = b.cc()?;
    let mut res = Matrix::new();
    igraph_call!(igraph_sparsemat_dense_multiply(a, &*bc, &mut res))?;
    Ok(res)
}

/// Iterator over the stored entries of a [`SparseMat`], see [`SparseMat::iter`].
///
/// Wraps `igraph_sparsemat_iterator_t` for column-compressed matrices; it
/// yields `(row, col, value)`.
pub struct SparseMatIter<'a> {
    inner: IterInner,
    _borrow: PhantomData<&'a SparseMat>,
}

enum IterInner {
    /// igraph's iterator over a column-compressed matrix.
    Compressed(igraph_sparsemat_iterator_t),
    /// A copy of the entries of a triplet matrix, and the next position.
    Triplet(SparseElements, usize),
}

impl<'a> SparseMatIter<'a> {
    fn new(m: &'a SparseMat) -> Self {
        let inner = if m.is_cc() {
            let mut raw = MaybeUninit::<igraph_sparsemat_iterator_t>::uninit();
            // Always succeeds.
            unsafe { igraph_sparsemat_iterator_init(raw.as_mut_ptr(), m) };
            IterInner::Compressed(unsafe { raw.assume_init() })
        } else {
            let elements = m
                .getelements()
                .expect("out of memory while copying the entries of a sparse matrix");
            IterInner::Triplet(elements, 0)
        };
        Self {
            inner,
            _borrow: PhantomData,
        }
    }

    /// Restarts the iteration from the first entry (`igraph_sparsemat_iterator_reset`).
    pub fn reset(&mut self) {
        match &mut self.inner {
            IterInner::Compressed(raw) => unsafe {
                igraph_sparsemat_iterator_reset(raw);
            },
            IterInner::Triplet(_, pos) => *pos = 0,
        }
    }

    /// Position of the next entry in the element arrays
    /// (`igraph_sparsemat_iterator_idx`).
    pub fn index(&self) -> usize {
        match &self.inner {
            IterInner::Compressed(raw) => unsafe { igraph_sparsemat_iterator_idx(raw) as usize },
            IterInner::Triplet(_, pos) => *pos,
        }
    }
}

impl Iterator for SparseMatIter<'_> {
    type Item = (usize, usize, f64);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            IterInner::Compressed(raw) => unsafe {
                if igraph_sparsemat_iterator_end(raw) {
                    return None;
                }
                let item = (
                    igraph_sparsemat_iterator_row(raw) as usize,
                    igraph_sparsemat_iterator_col(raw) as usize,
                    igraph_sparsemat_iterator_get(raw),
                );
                igraph_sparsemat_iterator_next(raw);
                Some(item)
            },
            IterInner::Triplet(e, pos) => {
                let k = *pos;
                if k >= e.x.len() {
                    return None;
                }
                *pos += 1;
                Some((e.i[k] as usize, e.j[k] as usize, e.x[k]))
            }
        }
    }
}

impl fmt::Debug for SparseMatIter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseMatIter")
            .field("index", &self.index())
            .finish()
    }
}

/// Owned result of a symbolic analysis (`igraph_sparsemat_symbolic_t`).
struct SymbolicGuard(igraph_sparsemat_symbolic_t);

impl SymbolicGuard {
    fn new(f: impl FnOnce(*mut igraph_sparsemat_symbolic_t) -> igraph_error_t) -> Result<Self> {
        ensure_init();
        let mut raw = igraph_sparsemat_symbolic_t {
            symbolic: ptr::null_mut(),
        };
        check(f(&mut raw))?;
        Ok(Self(raw))
    }
}

impl Drop for SymbolicGuard {
    fn drop(&mut self) {
        if !self.0.symbolic.is_null() {
            unsafe { igraph_sparsemat_symbolic_destroy(&mut self.0) };
        }
    }
}

/// Owned result of a numeric factorization (`igraph_sparsemat_numeric_t`).
struct NumericGuard(igraph_sparsemat_numeric_t);

impl NumericGuard {
    fn new(f: impl FnOnce(*mut igraph_sparsemat_numeric_t) -> igraph_error_t) -> Result<Self> {
        ensure_init();
        let mut raw = igraph_sparsemat_numeric_t {
            numeric: ptr::null_mut(),
        };
        check(f(&mut raw))?;
        Ok(Self(raw))
    }
}

impl Drop for NumericGuard {
    fn drop(&mut self) {
        if !self.0.numeric.is_null() {
            unsafe { igraph_sparsemat_numeric_destroy(&mut self.0) };
        }
    }
}

/// A sparse LU factorization, see [`SparseMat::lu`]. It owns igraph's
/// symbolic and numeric decompositions and frees them on drop
/// (`igraph_sparsemat_symbolic_destroy`, `igraph_sparsemat_numeric_destroy`).
pub struct SparseLu {
    symbolic: SymbolicGuard,
    numeric: NumericGuard,
    n: usize,
}

/// A sparse QR factorization of a square matrix, see [`SparseMat::qr`].
pub struct SparseQr {
    symbolic: SymbolicGuard,
    numeric: NumericGuard,
    n: usize,
}

// Both own their CXSparse structures exclusively and only read them in `solve`.
unsafe impl Send for SparseLu {}
unsafe impl Sync for SparseLu {}
unsafe impl Send for SparseQr {}
unsafe impl Sync for SparseQr {}

impl SparseLu {
    /// Order of the factorized matrix.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Solves `A x = b` with the precomputed factorization
    /// (`igraph_sparsemat_luresol`).
    ///
    /// Binds [`igraph_sparsemat_luresol`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_luresol).
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>> {
        check_len(b.len(), self.n, "b")?;
        let bv = Vector::view(b);
        let mut res = Vector::new();
        igraph_call!(igraph_sparsemat_luresol(
            &self.symbolic.0,
            &self.numeric.0,
            bv.as_ptr(),
            &mut res
        ))?;
        Ok(res.into())
    }
}

impl SparseQr {
    /// Order of the factorized matrix.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Solves `A x = b` with the precomputed factorization
    /// (`igraph_sparsemat_qrresol`).
    ///
    /// Binds [`igraph_sparsemat_qrresol`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_sparsemat_qrresol).
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>> {
        check_len(b.len(), self.n, "b")?;
        let bv = Vector::view(b);
        let mut res = Vector::new();
        igraph_call!(igraph_sparsemat_qrresol(
            &self.symbolic.0,
            &self.numeric.0,
            bv.as_ptr(),
            &mut res
        ))?;
        Ok(res.into())
    }
}

impl fmt::Debug for SparseLu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseLu")
            .field("n", &self.n)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for SparseQr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseQr")
            .field("n", &self.n)
            .finish_non_exhaustive()
    }
}

impl Add for &SparseMat {
    type Output = Result<SparseMat>;
    /// `a + b`, see [`SparseMat::add`].
    fn add(self, rhs: &SparseMat) -> Result<SparseMat> {
        SparseMat::add(self, rhs, 1.0, 1.0)
    }
}

impl Sub for &SparseMat {
    type Output = Result<SparseMat>;
    /// `a - b`, see [`SparseMat::add`].
    fn sub(self, rhs: &SparseMat) -> Result<SparseMat> {
        SparseMat::add(self, rhs, 1.0, -1.0)
    }
}

impl Mul for &SparseMat {
    type Output = Result<SparseMat>;
    /// `a * b`, see [`SparseMat::multiply`].
    fn mul(self, rhs: &SparseMat) -> Result<SparseMat> {
        self.multiply(rhs)
    }
}

impl TryFrom<&Matrix> for SparseMat {
    type Error = Error;
    /// Keeps the non-zero elements, see [`SparseMat::from_dense`].
    fn try_from(m: &Matrix) -> Result<Self> {
        SparseMat::from_dense(m, 0.0)
    }
}

impl TryFrom<&SparseMat> for Matrix {
    type Error = Error;
    /// See [`SparseMat::to_dense`].
    fn try_from(m: &SparseMat) -> Result<Self> {
        m.to_dense()
    }
}

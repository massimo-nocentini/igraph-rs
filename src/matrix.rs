//! Owned, column-major igraph matrices.
//!
//! [`Matrix`] (`igraph_matrix_t`, reals), [`MatrixInt`] (`igraph_matrix_int_t`),
//! [`MatrixBool`] (`igraph_matrix_bool_t`), [`MatrixChar`]
//! (`igraph_matrix_char_t`) and [`MatrixComplex`] (`igraph_matrix_complex_t`)
//! are the C structs themselves,
//! enriched with Rusty behaviour: they own their storage (freed on [`Drop`]),
//! are indexed with `m[(row, col)]`, and convert from and to row-major
//! `Vec<Vec<T>>`. The elements are stored in **column-major** order, as in
//! igraph, and [`as_slice`](Matrix::as_slice) exposes them in that order.
//!
//! ```
//! use igraph::prelude::*;
//!
//! let mut m = Matrix::from_rows(&[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]).unwrap();
//! assert_eq!((m.nrow(), m.ncol()), (3, 2));
//! assert_eq!(m[(2, 1)], 6.0);
//! m[(0, 0)] = 10.0;
//! assert_eq!(m.as_slice(), &[10.0, 3.0, 5.0, 2.0, 4.0, 6.0]); // column-major
//! assert_eq!(m.row(1), vec![3.0, 4.0]);
//! assert_eq!(m.column(1), &[2.0, 4.0, 6.0]);
//! ```
//!
//! The module covers `igraph_matrix_pmt.h`:
//!
//! | Group | Methods | Types |
//! |-------|---------|-------|
//! | construction | `zeros`, `new`, `from_rows`, `from_row_major`, `from_column_major`, `view` (zero-copy input), `identity` (real) | all |
//! | access | `[(i, j)]`, `get`, `row`, `column`, `rows`, `columns`, `diagonal`, `indexed_iter`, `to_rows`, `as_slice` | all |
//! | structure | `set_row`, `set_col`, `select_rows`, `select_cols`, `select_rows_cols`, `swap_rows`, `swap_cols`, `transpose`, `transposed`, `rbind`, `cbind`, `add_rows`, `add_cols`, `remove_row`, `remove_col`, `resize` | all |
//! | queries | `is_symmetric`, `contains`, `search`, `fill`, `capacity`, `shrink_to_fit` | all |
//! | order | `min`, `max`, `which_min`, `which_max`, `minmax`, `maxdifference`, `all_l`, `all_g`, `all_le`, `all_ge` | real, int, char |
//! | arithmetic | `add`, `sub`, `mul_elements`, `div_elements`, `scale`, `add_constant`, `sum`, `prod`, `rowsums`, `colsums` | real, int, complex |
//! | real only | `identity`, `matmul`, `mul_vec`, `all_almost_e`, `zapsmall` | real |
//! | complex only | `from_parts`, `from_polar`, `real`, `imag`, `realimag`, `all_almost_e`, `zapsmall` | complex |
//! | bool only | `count_true` | bool |
//!
//! Integer arithmetic is done in Rust with wrapping semantics (signed
//! overflow and division by zero are undefined behaviour in the C
//! implementation); char matrices have no arithmetic. Every index is checked
//! before reaching igraph, which does not check them.
//!
//! ```
//! use igraph::prelude::*;
//!
//! // The transition matrix of a random walk on the path 0 - 1 - 2.
//! let path = Graph::ring(3, false, false, false).unwrap();
//! let adj = path.get_adjacency(GetAdjacency::Both, None, Loops::Twice).unwrap();
//! assert_eq!(adj, Matrix::from_rows(&[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).unwrap());
//! assert!(adj.is_symmetric());
//! // Row sums of the adjacency matrix are the degrees...
//! let degrees = adj.rowsums();
//! assert_eq!(degrees, vec![1.0, 2.0, 1.0]);
//! // ...and dividing each row by them gives the row-stochastic matrix.
//! let mut p = adj.clone();
//! for (i, d) in degrees.iter().enumerate() {
//!     let row: Vec<f64> = p.row(i).iter().map(|x| x / d).collect();
//!     p.set_row(i, &row).unwrap();
//! }
//! assert_eq!(p, path.get_stochastic(false, None).unwrap());
//! assert_eq!(p.rowsums(), vec![1.0, 1.0, 1.0]);
//! // Two steps from the middle vertex bring the walker back with probability 1.
//! assert_eq!(p.matmul(&p).unwrap()[(1, 1)], 1.0);
//! ```
//!
//! See also [`Graph::get_adjacency`](crate::Graph::get_adjacency) and
//! [`Graph::adjacency`](crate::Graph::adjacency) to convert between graphs
//! and matrices, [`MatrixList`](crate::list::MatrixList) for lists of
//! matrices, [`linalg`](crate::linalg) for eigenvalue problems, BLAS/LAPACK
//! routines and sparse matrices, and [`layout`](crate::layout), whose
//! layouts are `n × 2` (or `n × 3`) coordinate matrices.

use crate::{
    error::{Error, Result},
    ffi::*,
    vector::{Vector, VectorBool, VectorChar, VectorComplex, VectorInt},
};
use std::{
    fmt,
    mem::MaybeUninit,
    ops::{Index, IndexMut},
};

/// Owned column-major matrix of reals (`igraph_matrix_t`), see the [module docs](self).
pub type Matrix = igraph_matrix_t;
/// Owned column-major matrix of integers (`igraph_matrix_int_t`), see the [module docs](self).
pub type MatrixInt = igraph_matrix_int_t;
/// Owned column-major matrix of booleans (`igraph_matrix_bool_t`), see the [module docs](self).
pub type MatrixBool = igraph_matrix_bool_t;
/// Owned column-major matrix of chars (`igraph_matrix_char_t`), see the [module docs](self).
pub type MatrixChar = igraph_matrix_char_t;
/// Owned column-major matrix of complex numbers (`igraph_matrix_complex_t`),
/// see the [module docs](self); igraph returns such matrices e.g. as the
/// eigenvectors of non-symmetric matrices.
pub type MatrixComplex = igraph_matrix_complex_t;

/// How a matrix element is written by the matrices' `Display`.
trait Cell {
    fn cell(&self) -> String;
}

macro_rules! debug_cell {
    ($($t:ty),*) => {$(
        impl Cell for $t {
            fn cell(&self) -> String {
                format!("{self:?}")
            }
        }
    )*};
}
debug_cell!(igraph_real_t, igraph_int_t, igraph_bool_t, std::ffi::c_char);

impl Cell for igraph_complex_t {
    fn cell(&self) -> String {
        self.to_string()
    }
}

macro_rules! impl_matrix {
    ($ty:ident, $elem:ty, init = $init:ident, init_copy = $init_copy:ident,
     destroy = $destroy:ident, resize = $resize:ident) => {
        impl $ty {
            /// Creates a `nrow` × `ncol` matrix filled with zeros
            /// ([`igraph_matrix_init`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_init)).
            ///
            /// # Panics
            /// If igraph cannot allocate the matrix, or the size overflows,
            /// like [`Vec::with_capacity`].
            pub fn zeros(nrow: usize, ncol: usize) -> Self {
                crate::error::ensure_init();
                let mut raw = MaybeUninit::<Self>::uninit();
                crate::error::check(unsafe {
                    $init(
                        raw.as_mut_ptr(),
                        crate::error::int_size(nrow),
                        crate::error::int_size(ncol),
                    )
                })
                .expect("igraph failed to allocate a matrix");
                unsafe { raw.assume_init() }
            }

            /// Creates an empty 0 × 0 matrix.
            pub fn new() -> Self {
                Self::zeros(0, 0)
            }

            /// Builds a matrix from its elements given in column-major order
            /// (igraph's storage order, see
            /// [`igraph_matrix_init_array`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_init_array)).
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue)
            /// if `data.len() != nrow * ncol`.
            pub fn from_column_major(nrow: usize, ncol: usize, data: &[$elem]) -> Result<Self> {
                if nrow.checked_mul(ncol) != Some(data.len()) {
                    return Err(Error::invalid(format!(
                        "{} elements given for a {nrow}x{ncol} matrix",
                        data.len()
                    )));
                }
                let mut m = Self::zeros(nrow, ncol);
                m.as_mut_slice().copy_from_slice(data);
                Ok(m)
            }

            /// Builds a matrix from its elements given in row-major order.
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue)
            /// if `data.len() != nrow * ncol`.
            pub fn from_row_major(nrow: usize, ncol: usize, data: &[$elem]) -> Result<Self> {
                if nrow.checked_mul(ncol) != Some(data.len()) {
                    return Err(Error::invalid(format!(
                        "{} elements given for a {nrow}x{ncol} matrix",
                        data.len()
                    )));
                }
                let mut m = Self::zeros(nrow, ncol);
                for i in 0..nrow {
                    for j in 0..ncol {
                        m[(i, j)] = data[i * ncol + j];
                    }
                }
                Ok(m)
            }

            /// Builds a matrix from a slice of rows (a `0 × 0` matrix for no
            /// rows).
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue)
            /// if the rows do not all have the same length.
            pub fn from_rows<R: AsRef<[$elem]>>(rows: &[R]) -> Result<Self> {
                let nrow = rows.len();
                let ncol = rows.first().map_or(0, |r| r.as_ref().len());
                let mut m = Self::zeros(nrow, ncol);
                for (i, row) in rows.iter().enumerate() {
                    let row = row.as_ref();
                    if row.len() != ncol {
                        return Err(Error::invalid(format!(
                            "row {i} has {} elements, expected {ncol}",
                            row.len()
                        )));
                    }
                    for (j, &x) in row.iter().enumerate() {
                        m[(i, j)] = x;
                    }
                }
                Ok(m)
            }

            /// Number of rows ([`igraph_matrix_nrow`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_nrow)).
            pub fn nrow(&self) -> usize {
                self.nrow as usize
            }

            /// Number of columns ([`igraph_matrix_ncol`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_ncol)).
            pub fn ncol(&self) -> usize {
                self.ncol as usize
            }

            /// `(nrow, ncol)`.
            pub fn shape(&self) -> (usize, usize) {
                (self.nrow(), self.ncol())
            }

            /// Whether the matrix has no elements.
            pub fn is_empty(&self) -> bool {
                self.nrow() * self.ncol() == 0
            }

            /// All the elements, in column-major order.
            pub fn as_slice(&self) -> &[$elem] {
                self.data.as_slice()
            }

            /// All the elements, in column-major order, mutably.
            pub fn as_mut_slice(&mut self) -> &mut [$elem] {
                self.data.as_mut_slice()
            }

            /// The element at `(row, col)`, if in bounds.
            pub fn get(&self, row: usize, col: usize) -> Option<&$elem> {
                if row < self.nrow() && col < self.ncol() {
                    Some(&self.as_slice()[col * self.nrow() + row])
                } else {
                    None
                }
            }

            /// The `j`-th column, as a slice (columns are contiguous in
            /// column-major storage).
            ///
            /// # Panics
            /// If `j >= ncol`.
            pub fn column(&self, j: usize) -> &[$elem] {
                let n = self.nrow();
                &self.as_slice()[j * n..(j + 1) * n]
            }

            /// A copy of the `i`-th row.
            ///
            /// # Panics
            /// If `i >= nrow`.
            pub fn row(&self, i: usize) -> Vec<$elem> {
                assert!(i < self.nrow(), "row index {i} out of bounds");
                (0..self.ncol()).map(|j| self[(i, j)]).collect()
            }

            /// Iterates over copies of the rows.
            pub fn rows(&self) -> impl Iterator<Item = Vec<$elem>> + '_ {
                (0..self.nrow()).map(move |i| self.row(i))
            }

            /// Iterates over the columns, as slices.
            pub fn columns(&self) -> impl Iterator<Item = &[$elem]> + '_ {
                (0..self.ncol()).map(move |j| self.column(j))
            }

            /// Converts into a row-major `Vec` of rows.
            pub fn to_rows(&self) -> Vec<Vec<$elem>> {
                self.rows().collect()
            }

            /// Returns the transposed matrix.
            pub fn transposed(&self) -> Self {
                let mut t = Self::zeros(self.ncol(), self.nrow());
                for i in 0..self.nrow() {
                    for j in 0..self.ncol() {
                        t[(j, i)] = self[(i, j)];
                    }
                }
                t
            }

            /// Resizes to `nrow` × `ncol`
            /// ([`igraph_matrix_resize`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_resize)); igraph
            /// does not preserve the content meaningfully, so this wrapper
            /// resets every element to zero. Use
            /// [`add_rows`](Self::add_rows), [`add_cols`](Self::add_cols),
            /// [`remove_row`](Self::remove_row) and
            /// [`remove_col`](Self::remove_col) to keep the content.
            pub fn resize(&mut self, nrow: usize, ncol: usize) {
                crate::error::check(unsafe {
                    $resize(
                        self,
                        crate::error::int_size(nrow),
                        crate::error::int_size(ncol),
                    )
                })
                .expect("igraph failed to resize a matrix");
                for x in self.as_mut_slice() {
                    *x = Default::default();
                }
            }
        }

        impl Drop for $ty {
            fn drop(&mut self) {
                if !self.data.stor_begin.is_null() {
                    unsafe { $destroy(self) };
                    // `data` has been destroyed by igraph: forget it.
                    self.data.stor_begin = std::ptr::null_mut();
                }
            }
        }

        impl Default for $ty {
            fn default() -> Self {
                Self::new()
            }
        }

        impl Clone for $ty {
            fn clone(&self) -> Self {
                crate::error::ensure_init();
                let mut raw = MaybeUninit::<Self>::uninit();
                crate::error::check(unsafe { $init_copy(raw.as_mut_ptr(), self) })
                    .expect("igraph failed to copy a matrix");
                unsafe { raw.assume_init() }
            }
        }

        impl PartialEq for $ty {
            fn eq(&self, other: &Self) -> bool {
                self.shape() == other.shape() && self.as_slice() == other.as_slice()
            }
        }

        impl Index<(usize, usize)> for $ty {
            type Output = $elem;
            fn index(&self, (i, j): (usize, usize)) -> &$elem {
                assert!(
                    i < self.nrow() && j < self.ncol(),
                    "matrix index ({i}, {j}) out of bounds"
                );
                let n = self.nrow();
                &self.as_slice()[j * n + i]
            }
        }

        impl IndexMut<(usize, usize)> for $ty {
            fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut $elem {
                assert!(
                    i < self.nrow() && j < self.ncol(),
                    "matrix index ({i}, {j}) out of bounds"
                );
                let n = self.nrow();
                &mut self.as_mut_slice()[j * n + i]
            }
        }

        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                for i in 0..self.nrow() {
                    let row = self.row(i);
                    let cells: Vec<String> = row.iter().map(Cell::cell).collect();
                    writeln!(f, "[{}]", cells.join(", "))?;
                }
                Ok(())
            }
        }

        impl From<$ty> for Vec<Vec<$elem>> {
            fn from(m: $ty) -> Self {
                m.to_rows()
            }
        }

        unsafe impl Send for $ty {}
        unsafe impl Sync for $ty {}
    };
}

impl_matrix!(
    igraph_matrix_t,
    igraph_real_t,
    init = igraph_matrix_init,
    init_copy = igraph_matrix_init_copy,
    destroy = igraph_matrix_destroy,
    resize = igraph_matrix_resize
);
impl_matrix!(
    igraph_matrix_int_t,
    igraph_int_t,
    init = igraph_matrix_int_init,
    init_copy = igraph_matrix_int_init_copy,
    destroy = igraph_matrix_int_destroy,
    resize = igraph_matrix_int_resize
);
impl_matrix!(
    igraph_matrix_bool_t,
    igraph_bool_t,
    init = igraph_matrix_bool_init,
    init_copy = igraph_matrix_bool_init_copy,
    destroy = igraph_matrix_bool_destroy,
    resize = igraph_matrix_bool_resize
);
impl_matrix!(
    igraph_matrix_char_t,
    std::ffi::c_char,
    init = igraph_matrix_char_init,
    init_copy = igraph_matrix_char_init_copy,
    destroy = igraph_matrix_char_destroy,
    resize = igraph_matrix_char_resize
);
impl_matrix!(
    igraph_matrix_complex_t,
    igraph_complex_t,
    init = igraph_matrix_complex_init,
    init_copy = igraph_matrix_complex_init_copy,
    destroy = igraph_matrix_complex_destroy,
    resize = igraph_matrix_complex_resize
);

// ---------------------------------------------------------------------------
// Structural operations, for every element type (`igraph_matrix_pmt.h`).
// ---------------------------------------------------------------------------

fn check_positions(index: &[igraph_int_t], len: usize, what: &str) -> Result<()> {
    match index.iter().find(|&&i| i < 0 || i as usize >= len) {
        Some(i) => Err(Error::invalid(format!(
            "{what} index {i} out of bounds (there are {len})"
        ))),
        None => Ok(()),
    }
}

macro_rules! impl_matrix_common {
    (
        $ty:ident, $elem:ty, $vec:ident,
        view = $view:ident, fill = $fill:ident, set_row = $set_row:ident, set_col = $set_col:ident,
        select_rows = $select_rows:ident, select_cols = $select_cols:ident,
        select_rows_cols = $select_rows_cols:ident, swap_rows = $swap_rows:ident,
        swap_cols = $swap_cols:ident, transpose = $transpose:ident,
        rbind = $rbind:ident, cbind = $cbind:ident, add_rows = $add_rows:ident,
        add_cols = $add_cols:ident, remove_row = $remove_row:ident, remove_col = $remove_col:ident,
        is_symmetric = $is_symmetric:ident, contains = $contains:ident, search = $search:ident,
        capacity = $capacity:ident, resize_min = $resize_min:ident
    ) => {
        impl $ty {
            /// A read-only matrix *view* of column-major `data`, without
            /// copying it; the view dereferences to the matrix type, so it
            /// can be handed to igraph as a `const` matrix input
            /// ([`igraph_matrix_view`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_view)).
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue)
            /// if `data.len() != nrow * ncol`.
            pub fn view(
                data: &[$elem],
                nrow: usize,
                ncol: usize,
            ) -> Result<crate::vector::View<'_, Self>> {
                if nrow.checked_mul(ncol) != Some(data.len()) {
                    return Err(Error::invalid(format!(
                        "{} elements given for a {nrow}x{ncol} matrix",
                        data.len()
                    )));
                }
                let raw = unsafe {
                    $view(
                        data.as_ptr(),
                        crate::error::int_size(nrow),
                        crate::error::int_size(ncol),
                    )
                };
                // SAFETY: the view borrows `data` for its whole lifetime.
                Ok(unsafe { crate::vector::View::from_raw(raw) })
            }

            /// Sets every element to `value`
            /// ([`igraph_matrix_fill`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_fill)).
            pub fn fill(&mut self, value: $elem) {
                if !self.is_empty() {
                    unsafe { $fill(self, value) };
                }
            }

            /// Replaces row `i` with `values`
            /// ([`igraph_matrix_set_row`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_set_row)).
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue)
            /// if `i` is out of bounds or `values.len() != ncol`.
            pub fn set_row(&mut self, i: usize, values: &[$elem]) -> Result<()> {
                if i >= self.nrow() || values.len() != self.ncol() {
                    return Err(Error::invalid(format!(
                        "cannot set row {i} of a {}x{} matrix with {} values",
                        self.nrow(),
                        self.ncol(),
                        values.len()
                    )));
                }
                let v = $vec::view(values);
                crate::igraph_call!($set_row(self, v.as_ptr(), i as igraph_int_t))
            }

            /// Replaces column `j` with `values`
            /// ([`igraph_matrix_set_col`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_set_col)).
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue)
            /// if `j` is out of bounds or `values.len() != nrow`.
            pub fn set_col(&mut self, j: usize, values: &[$elem]) -> Result<()> {
                if j >= self.ncol() || values.len() != self.nrow() {
                    return Err(Error::invalid(format!(
                        "cannot set column {j} of a {}x{} matrix with {} values",
                        self.nrow(),
                        self.ncol(),
                        values.len()
                    )));
                }
                let v = $vec::view(values);
                crate::igraph_call!($set_col(self, v.as_ptr(), j as igraph_int_t))
            }

            /// A new matrix made of the given rows, in the given order
            /// (repetitions allowed)
            /// ([`igraph_matrix_select_rows`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_select_rows)).
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) for out of bounds rows.
            pub fn select_rows(&self, rows: &[igraph_int_t]) -> Result<Self> {
                check_positions(rows, self.nrow(), "row")?;
                let r = VectorInt::view(rows);
                let mut res = Self::zeros(rows.len(), self.ncol());
                crate::igraph_call!($select_rows(self, &mut res, r.as_ptr()))?;
                Ok(res)
            }

            /// A new matrix made of the given columns, in the given order
            /// ([`igraph_matrix_select_cols`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_select_cols)).
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) for out of bounds columns.
            pub fn select_cols(&self, cols: &[igraph_int_t]) -> Result<Self> {
                check_positions(cols, self.ncol(), "column")?;
                let c = VectorInt::view(cols);
                let mut res = Self::zeros(self.nrow(), cols.len());
                crate::igraph_call!($select_cols(self, &mut res, c.as_ptr()))?;
                Ok(res)
            }

            /// The submatrix at the intersection of the given rows and columns
            /// ([`igraph_matrix_select_rows_cols`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_select_rows_cols)).
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) for out of bounds indices.
            pub fn select_rows_cols(
                &self,
                rows: &[igraph_int_t],
                cols: &[igraph_int_t],
            ) -> Result<Self> {
                check_positions(rows, self.nrow(), "row")?;
                check_positions(cols, self.ncol(), "column")?;
                let r = VectorInt::view(rows);
                let c = VectorInt::view(cols);
                let mut res = Self::zeros(rows.len(), cols.len());
                crate::igraph_call!($select_rows_cols(self, &mut res, r.as_ptr(), c.as_ptr()))?;
                Ok(res)
            }

            /// Swaps rows `i` and `j`
            /// ([`igraph_matrix_swap_rows`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_swap_rows)).
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) for out of bounds rows.
            pub fn swap_rows(&mut self, i: usize, j: usize) -> Result<()> {
                check_positions(&[i as igraph_int_t, j as igraph_int_t], self.nrow(), "row")?;
                crate::igraph_call!($swap_rows(self, i as igraph_int_t, j as igraph_int_t))
            }

            /// Swaps columns `i` and `j`
            /// ([`igraph_matrix_swap_cols`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_swap_cols)).
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) for out of bounds columns.
            pub fn swap_cols(&mut self, i: usize, j: usize) -> Result<()> {
                check_positions(
                    &[i as igraph_int_t, j as igraph_int_t],
                    self.ncol(),
                    "column",
                )?;
                crate::igraph_call!($swap_cols(self, i as igraph_int_t, j as igraph_int_t))
            }

            /// Transposes the matrix in place
            /// ([`igraph_matrix_transpose`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_transpose));
            /// see also [`transposed`](Self::transposed).
            pub fn transpose(&mut self) {
                crate::error::check(unsafe { $transpose(self) })
                    .expect("igraph failed to transpose a matrix");
            }

            /// Appends the rows of `other` below this matrix
            /// ([`igraph_matrix_rbind`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_rbind)).
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the column counts differ.
            pub fn rbind(&mut self, other: &Self) -> Result<()> {
                if self.ncol() != other.ncol() {
                    return Err(Error::invalid("rbind: the number of columns differ"));
                }
                crate::igraph_call!($rbind(self, other))
            }

            /// Appends the columns of `other` to the right of this matrix
            /// ([`igraph_matrix_cbind`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_cbind)).
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the row counts differ.
            pub fn cbind(&mut self, other: &Self) -> Result<()> {
                if self.nrow() != other.nrow() {
                    return Err(Error::invalid("cbind: the number of rows differ"));
                }
                crate::igraph_call!($cbind(self, other))
            }

            /// Appends `n` rows of zeros
            /// ([`igraph_matrix_add_rows`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_add_rows)).
            pub fn add_rows(&mut self, n: usize) {
                let old = self.nrow();
                crate::error::check(unsafe { $add_rows(self, crate::error::int_size(n)) })
                    .expect("igraph failed to grow a matrix");
                for j in 0..self.ncol() {
                    for i in old..self.nrow() {
                        self[(i, j)] = Default::default();
                    }
                }
            }

            /// Appends `n` columns of zeros
            /// ([`igraph_matrix_add_cols`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_add_cols)).
            pub fn add_cols(&mut self, n: usize) {
                let old = self.ncol();
                crate::error::check(unsafe { $add_cols(self, crate::error::int_size(n)) })
                    .expect("igraph failed to grow a matrix");
                let nrow = self.nrow();
                for x in &mut self.as_mut_slice()[old * nrow..] {
                    *x = Default::default();
                }
            }

            /// Removes row `i`
            /// ([`igraph_matrix_remove_row`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_remove_row)).
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if out of bounds.
            pub fn remove_row(&mut self, i: usize) -> Result<()> {
                check_positions(&[i as igraph_int_t], self.nrow(), "row")?;
                crate::igraph_call!($remove_row(self, i as igraph_int_t))
            }

            /// Removes column `j`
            /// ([`igraph_matrix_remove_col`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_remove_col)).
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if out of bounds.
            pub fn remove_col(&mut self, j: usize) -> Result<()> {
                check_positions(&[j as igraph_int_t], self.ncol(), "column")?;
                crate::igraph_call!($remove_col(self, j as igraph_int_t))
            }

            /// Whether the matrix is square and equal to its transpose
            /// ([`igraph_matrix_is_symmetric`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_is_symmetric)).
            pub fn is_symmetric(&self) -> bool {
                unsafe { $is_symmetric(self) }
            }

            /// Whether some element equals `value`
            /// ([`igraph_matrix_contains`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_contains)).
            pub fn contains(&self, value: $elem) -> bool {
                !self.is_empty() && unsafe { $contains(self, value) }
            }

            /// The `(row, col)` of the first occurrence of `value` at or after
            /// the column-major position `from`, if any
            /// ([`igraph_matrix_search`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_search)).
            pub fn search(&self, from: usize, value: $elem) -> Option<(usize, usize)> {
                if from >= self.as_slice().len() {
                    return None;
                }
                let (mut pos, mut row, mut col) = (0, 0, 0);
                unsafe {
                    $search(
                        self,
                        from as igraph_int_t,
                        value,
                        &mut pos,
                        &mut row,
                        &mut col,
                    )
                }
                .then_some((row as usize, col as usize))
            }

            /// Number of elements the matrix can hold without reallocating
            /// ([`igraph_matrix_capacity`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_capacity)).
            pub fn capacity(&self) -> usize {
                unsafe { $capacity(self) as usize }
            }

            /// Frees the unused storage
            /// ([`igraph_matrix_resize_min`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_resize_min)).
            pub fn shrink_to_fit(&mut self) {
                unsafe { $resize_min(self) };
            }

            /// The main diagonal, `m[(i, i)]` for `i < min(nrow, ncol)`.
            pub fn diagonal(&self) -> Vec<$elem> {
                (0..self.nrow().min(self.ncol()))
                    .map(|i| self[(i, i)])
                    .collect()
            }

            /// Iterates over `((row, col), &value)` in column-major order.
            pub fn indexed_iter(&self) -> impl Iterator<Item = ((usize, usize), &$elem)> + '_ {
                let n = self.nrow().max(1);
                self.as_slice()
                    .iter()
                    .enumerate()
                    .map(move |(k, x)| ((k % n, k / n), x))
            }
        }
    };
}

impl_matrix_common!(
    igraph_matrix_t,
    igraph_real_t,
    Vector,
    view = igraph_matrix_view,
    fill = igraph_matrix_fill,
    set_row = igraph_matrix_set_row,
    set_col = igraph_matrix_set_col,
    select_rows = igraph_matrix_select_rows,
    select_cols = igraph_matrix_select_cols,
    select_rows_cols = igraph_matrix_select_rows_cols,
    swap_rows = igraph_matrix_swap_rows,
    swap_cols = igraph_matrix_swap_cols,
    transpose = igraph_matrix_transpose,
    rbind = igraph_matrix_rbind,
    cbind = igraph_matrix_cbind,
    add_rows = igraph_matrix_add_rows,
    add_cols = igraph_matrix_add_cols,
    remove_row = igraph_matrix_remove_row,
    remove_col = igraph_matrix_remove_col,
    is_symmetric = igraph_matrix_is_symmetric,
    contains = igraph_matrix_contains,
    search = igraph_matrix_search,
    capacity = igraph_matrix_capacity,
    resize_min = igraph_matrix_resize_min
);
impl_matrix_common!(
    igraph_matrix_int_t,
    igraph_int_t,
    VectorInt,
    view = igraph_matrix_int_view,
    fill = igraph_matrix_int_fill,
    set_row = igraph_matrix_int_set_row,
    set_col = igraph_matrix_int_set_col,
    select_rows = igraph_matrix_int_select_rows,
    select_cols = igraph_matrix_int_select_cols,
    select_rows_cols = igraph_matrix_int_select_rows_cols,
    swap_rows = igraph_matrix_int_swap_rows,
    swap_cols = igraph_matrix_int_swap_cols,
    transpose = igraph_matrix_int_transpose,
    rbind = igraph_matrix_int_rbind,
    cbind = igraph_matrix_int_cbind,
    add_rows = igraph_matrix_int_add_rows,
    add_cols = igraph_matrix_int_add_cols,
    remove_row = igraph_matrix_int_remove_row,
    remove_col = igraph_matrix_int_remove_col,
    is_symmetric = igraph_matrix_int_is_symmetric,
    contains = igraph_matrix_int_contains,
    search = igraph_matrix_int_search,
    capacity = igraph_matrix_int_capacity,
    resize_min = igraph_matrix_int_resize_min
);
impl_matrix_common!(
    igraph_matrix_bool_t,
    igraph_bool_t,
    VectorBool,
    view = igraph_matrix_bool_view,
    fill = igraph_matrix_bool_fill,
    set_row = igraph_matrix_bool_set_row,
    set_col = igraph_matrix_bool_set_col,
    select_rows = igraph_matrix_bool_select_rows,
    select_cols = igraph_matrix_bool_select_cols,
    select_rows_cols = igraph_matrix_bool_select_rows_cols,
    swap_rows = igraph_matrix_bool_swap_rows,
    swap_cols = igraph_matrix_bool_swap_cols,
    transpose = igraph_matrix_bool_transpose,
    rbind = igraph_matrix_bool_rbind,
    cbind = igraph_matrix_bool_cbind,
    add_rows = igraph_matrix_bool_add_rows,
    add_cols = igraph_matrix_bool_add_cols,
    remove_row = igraph_matrix_bool_remove_row,
    remove_col = igraph_matrix_bool_remove_col,
    is_symmetric = igraph_matrix_bool_is_symmetric,
    contains = igraph_matrix_bool_contains,
    search = igraph_matrix_bool_search,
    capacity = igraph_matrix_bool_capacity,
    resize_min = igraph_matrix_bool_resize_min
);
impl_matrix_common!(
    igraph_matrix_char_t,
    std::ffi::c_char,
    VectorChar,
    view = igraph_matrix_char_view,
    fill = igraph_matrix_char_fill,
    set_row = igraph_matrix_char_set_row,
    set_col = igraph_matrix_char_set_col,
    select_rows = igraph_matrix_char_select_rows,
    select_cols = igraph_matrix_char_select_cols,
    select_rows_cols = igraph_matrix_char_select_rows_cols,
    swap_rows = igraph_matrix_char_swap_rows,
    swap_cols = igraph_matrix_char_swap_cols,
    transpose = igraph_matrix_char_transpose,
    rbind = igraph_matrix_char_rbind,
    cbind = igraph_matrix_char_cbind,
    add_rows = igraph_matrix_char_add_rows,
    add_cols = igraph_matrix_char_add_cols,
    remove_row = igraph_matrix_char_remove_row,
    remove_col = igraph_matrix_char_remove_col,
    is_symmetric = igraph_matrix_char_is_symmetric,
    contains = igraph_matrix_char_contains,
    search = igraph_matrix_char_search,
    capacity = igraph_matrix_char_capacity,
    resize_min = igraph_matrix_char_resize_min
);
impl_matrix_common!(
    igraph_matrix_complex_t,
    igraph_complex_t,
    VectorComplex,
    view = igraph_matrix_complex_view,
    fill = igraph_matrix_complex_fill,
    set_row = igraph_matrix_complex_set_row,
    set_col = igraph_matrix_complex_set_col,
    select_rows = igraph_matrix_complex_select_rows,
    select_cols = igraph_matrix_complex_select_cols,
    select_rows_cols = igraph_matrix_complex_select_rows_cols,
    swap_rows = igraph_matrix_complex_swap_rows,
    swap_cols = igraph_matrix_complex_swap_cols,
    transpose = igraph_matrix_complex_transpose,
    rbind = igraph_matrix_complex_rbind,
    cbind = igraph_matrix_complex_cbind,
    add_rows = igraph_matrix_complex_add_rows,
    add_cols = igraph_matrix_complex_add_cols,
    remove_row = igraph_matrix_complex_remove_row,
    remove_col = igraph_matrix_complex_remove_col,
    is_symmetric = igraph_matrix_complex_is_symmetric,
    contains = igraph_matrix_complex_contains,
    search = igraph_matrix_complex_search,
    capacity = igraph_matrix_complex_capacity,
    resize_min = igraph_matrix_complex_resize_min
);

// ---------------------------------------------------------------------------
// Ordered element types: reals and integers.
// ---------------------------------------------------------------------------

macro_rules! impl_matrix_ordered {
    (
        $ty:ident, $elem:ty,
        which_min = $which_min:ident,
        which_max = $which_max:ident, maxdifference = $maxdifference:ident,
        all_l = $all_l:ident, all_g = $all_g:ident, all_le = $all_le:ident, all_ge = $all_ge:ident
    ) => {
        impl $ty {
            /// The smallest element, or `None` if the matrix is empty
            /// ([`igraph_matrix_min`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_min)).
            pub fn min(&self) -> Option<$elem> {
                self.which_min().map(|ij| self[ij])
            }

            /// The largest element, or `None` if the matrix is empty
            /// ([`igraph_matrix_max`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_max)).
            pub fn max(&self) -> Option<$elem> {
                self.which_max().map(|ij| self[ij])
            }

            /// `(row, col)` of the smallest element, or `None` if empty
            /// ([`igraph_matrix_which_min`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_which_min)).
            pub fn which_min(&self) -> Option<(usize, usize)> {
                if self.is_empty() {
                    return None;
                }
                let (mut i, mut j) = (0, 0);
                unsafe { $which_min(self, &mut i, &mut j) };
                Some((i as usize, j as usize))
            }

            /// `(row, col)` of the largest element, or `None` if empty
            /// ([`igraph_matrix_which_max`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_which_max)).
            pub fn which_max(&self) -> Option<(usize, usize)> {
                if self.is_empty() {
                    return None;
                }
                let (mut i, mut j) = (0, 0);
                unsafe { $which_max(self, &mut i, &mut j) };
                Some((i as usize, j as usize))
            }

            /// `(min, max)`, or `None` if empty (as `igraph_matrix_minmax`).
            pub fn minmax(&self) -> Option<($elem, $elem)> {
                Some((self.min()?, self.max()?))
            }

            /// The largest absolute element-wise difference, comparing the
            /// elements in column-major order
            /// ([`igraph_matrix_maxdifference`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_maxdifference)).
            pub fn maxdifference(&self, other: &Self) -> f64 {
                if self.is_empty() || other.is_empty() {
                    return 0.0;
                }
                unsafe { $maxdifference(self, other) }
            }

            /// Whether the shapes agree and every element is `<` the other's
            /// ([`igraph_matrix_all_l`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_all_l)).
            pub fn all_l(&self, other: &Self) -> bool {
                self.shape() == other.shape() && (self.is_empty() || unsafe { $all_l(self, other) })
            }

            /// Whether the shapes agree and every element is `>` the other's
            /// ([`igraph_matrix_all_g`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_all_g)).
            pub fn all_g(&self, other: &Self) -> bool {
                self.shape() == other.shape() && (self.is_empty() || unsafe { $all_g(self, other) })
            }

            /// Whether the shapes agree and every element is `<=` the other's
            /// ([`igraph_matrix_all_le`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_all_le)).
            pub fn all_le(&self, other: &Self) -> bool {
                self.shape() == other.shape()
                    && (self.is_empty() || unsafe { $all_le(self, other) })
            }

            /// Whether the shapes agree and every element is `>=` the other's
            /// ([`igraph_matrix_all_ge`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_all_ge)).
            pub fn all_ge(&self, other: &Self) -> bool {
                self.shape() == other.shape()
                    && (self.is_empty() || unsafe { $all_ge(self, other) })
            }
        }
    };
}

impl_matrix_ordered!(
    igraph_matrix_t,
    igraph_real_t,
    which_min = igraph_matrix_which_min,
    which_max = igraph_matrix_which_max,
    maxdifference = igraph_matrix_maxdifference,
    all_l = igraph_matrix_all_l,
    all_g = igraph_matrix_all_g,
    all_le = igraph_matrix_all_le,
    all_ge = igraph_matrix_all_ge
);
impl_matrix_ordered!(
    igraph_matrix_int_t,
    igraph_int_t,
    which_min = igraph_matrix_int_which_min,
    which_max = igraph_matrix_int_which_max,
    maxdifference = igraph_matrix_int_maxdifference,
    all_l = igraph_matrix_int_all_l,
    all_g = igraph_matrix_int_all_g,
    all_le = igraph_matrix_int_all_le,
    all_ge = igraph_matrix_int_all_ge
);
impl_matrix_ordered!(
    igraph_matrix_char_t,
    std::ffi::c_char,
    which_min = igraph_matrix_char_which_min,
    which_max = igraph_matrix_char_which_max,
    maxdifference = igraph_matrix_char_maxdifference,
    all_l = igraph_matrix_char_all_l,
    all_g = igraph_matrix_char_all_g,
    all_le = igraph_matrix_char_all_le,
    all_ge = igraph_matrix_char_all_ge
);

// ---------------------------------------------------------------------------
// Arithmetic on real matrices (igraph) and integer matrices (Rust, wrapping).
// ---------------------------------------------------------------------------

impl igraph_matrix_t {
    /// The `n` × `n` identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }

    fn check_same_shape(&self, other: &Self) -> Result<()> {
        if self.shape() != other.shape() {
            return Err(Error::invalid(format!(
                "matrices of different shapes ({}x{} and {}x{})",
                self.nrow(),
                self.ncol(),
                other.nrow(),
                other.ncol()
            )));
        }
        Ok(())
    }

    /// Element-wise `self += other`
    /// ([`igraph_matrix_add`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_add)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the shapes differ.
    pub fn add(&mut self, other: &Self) -> Result<()> {
        self.check_same_shape(other)?;
        crate::igraph_call!(igraph_matrix_add(self, other))
    }

    /// Element-wise `self -= other`
    /// ([`igraph_matrix_sub`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_sub)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the shapes differ.
    pub fn sub(&mut self, other: &Self) -> Result<()> {
        self.check_same_shape(other)?;
        crate::igraph_call!(igraph_matrix_sub(self, other))
    }

    /// Element-wise (Hadamard) product `self[(i,j)] *= other[(i,j)]`
    /// ([`igraph_matrix_mul_elements`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_mul_elements)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the shapes differ.
    pub fn mul_elements(&mut self, other: &Self) -> Result<()> {
        self.check_same_shape(other)?;
        crate::igraph_call!(igraph_matrix_mul_elements(self, other))
    }

    /// Element-wise division `self[(i,j)] /= other[(i,j)]`
    /// ([`igraph_matrix_div_elements`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_div_elements)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the shapes differ.
    pub fn div_elements(&mut self, other: &Self) -> Result<()> {
        self.check_same_shape(other)?;
        crate::igraph_call!(igraph_matrix_div_elements(self, other))
    }

    /// Multiplies every element by `factor`
    /// ([`igraph_matrix_scale`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_scale)).
    pub fn scale(&mut self, factor: f64) {
        unsafe { igraph_matrix_scale(self, factor) }
    }

    /// Adds `value` to every element
    /// ([`igraph_matrix_add_constant`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_add_constant)).
    pub fn add_constant(&mut self, value: f64) {
        unsafe { igraph_matrix_add_constant(self, value) }
    }

    /// Sum of all elements
    /// ([`igraph_matrix_sum`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_sum)).
    pub fn sum(&self) -> f64 {
        unsafe { igraph_matrix_sum(self) }
    }

    /// Product of all elements (one for an empty matrix)
    /// ([`igraph_matrix_prod`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_prod)).
    pub fn prod(&self) -> f64 {
        unsafe { igraph_matrix_prod(self) }
    }

    /// The sum of each row
    /// ([`igraph_matrix_rowsum`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_rowsum)).
    pub fn rowsums(&self) -> Vec<f64> {
        let mut res = Vector::new();
        crate::error::check(unsafe { igraph_matrix_rowsum(self, &mut res) })
            .expect("igraph failed to allocate a vector");
        res.into()
    }

    /// The sum of each column
    /// ([`igraph_matrix_colsum`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_colsum)).
    pub fn colsums(&self) -> Vec<f64> {
        let mut res = Vector::new();
        crate::error::check(unsafe { igraph_matrix_colsum(self, &mut res) })
            .expect("igraph failed to allocate a vector");
        res.into()
    }

    /// Whether the shapes agree and all elements are equal up to the
    /// relative tolerance `eps`
    /// ([`igraph_matrix_all_almost_e`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_all_almost_e)).
    pub fn all_almost_e(&self, other: &Self, eps: f64) -> bool {
        self.shape() == other.shape() && unsafe { igraph_matrix_all_almost_e(self, other, eps) }
    }

    /// Replaces the elements smaller in magnitude than the *absolute*
    /// tolerance `tol` by exact zeros; `tol = 0` uses igraph's default,
    /// `f64::EPSILON^(2/3)` (about `1e-10`)
    /// ([`igraph_matrix_zapsmall`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_zapsmall)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if `tol < 0`.
    pub fn zapsmall(&mut self, tol: f64) -> Result<()> {
        crate::igraph_call!(igraph_matrix_zapsmall(self, tol))
    }

    /// The matrix product `self · other` (a plain Rust triple loop; for
    /// BLAS-backed products see the linear algebra module).
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let a = Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]]).unwrap();
    /// let b = a.matmul(&Matrix::identity(2)).unwrap();
    /// assert_eq!(a, b);
    /// assert_eq!(a.matmul(&a).unwrap().to_rows(), vec![vec![7.0, 10.0], vec![15.0, 22.0]]);
    /// ```
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if
    /// `self.ncol() != other.nrow()`.
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        if self.ncol() != other.nrow() {
            return Err(Error::invalid(format!(
                "cannot multiply a {}x{} matrix by a {}x{} one",
                self.nrow(),
                self.ncol(),
                other.nrow(),
                other.ncol()
            )));
        }
        let mut res = Self::zeros(self.nrow(), other.ncol());
        for j in 0..other.ncol() {
            for k in 0..self.ncol() {
                let b = other[(k, j)];
                if b == 0.0 {
                    continue;
                }
                for i in 0..self.nrow() {
                    res[(i, j)] += self[(i, k)] * b;
                }
            }
        }
        Ok(res)
    }

    /// The matrix-vector product `self · x`.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if `x.len() != ncol`.
    pub fn mul_vec(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.ncol() {
            return Err(Error::invalid(
                "vector length does not match the number of columns",
            ));
        }
        let mut res = vec![0.0; self.nrow()];
        for (j, col) in self.columns().enumerate() {
            for (r, a) in res.iter_mut().zip(col) {
                *r += a * x[j];
            }
        }
        Ok(res)
    }
}

impl igraph_matrix_int_t {
    fn zip_op(
        &mut self,
        other: &Self,
        f: impl Fn(igraph_int_t, igraph_int_t) -> igraph_int_t,
    ) -> Result<()> {
        if self.shape() != other.shape() {
            return Err(Error::invalid("matrices of different shapes"));
        }
        self.as_mut_slice()
            .iter_mut()
            .zip(other.as_slice())
            .for_each(|(a, &b)| *a = f(*a, b));
        Ok(())
    }

    /// Element-wise `self += other`, wrapping on overflow (Rust counterpart
    /// of `igraph_matrix_int_add`).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the shapes differ.
    pub fn add(&mut self, other: &Self) -> Result<()> {
        self.zip_op(other, igraph_int_t::wrapping_add)
    }

    /// Element-wise `self -= other`, wrapping on overflow (Rust counterpart
    /// of `igraph_matrix_int_sub`).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the shapes differ.
    pub fn sub(&mut self, other: &Self) -> Result<()> {
        self.zip_op(other, igraph_int_t::wrapping_sub)
    }

    /// Element-wise product, wrapping on overflow (Rust counterpart of
    /// `igraph_matrix_int_mul_elements`).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the shapes differ.
    pub fn mul_elements(&mut self, other: &Self) -> Result<()> {
        self.zip_op(other, igraph_int_t::wrapping_mul)
    }

    /// Multiplies every element by `factor`, wrapping on overflow (Rust
    /// counterpart of `igraph_matrix_int_scale`).
    pub fn scale(&mut self, factor: igraph_int_t) {
        self.as_mut_slice()
            .iter_mut()
            .for_each(|x| *x = x.wrapping_mul(factor));
    }

    /// Adds `value` to every element, wrapping on overflow (Rust
    /// counterpart of `igraph_matrix_int_add_constant`).
    pub fn add_constant(&mut self, value: igraph_int_t) {
        self.as_mut_slice()
            .iter_mut()
            .for_each(|x| *x = x.wrapping_add(value));
    }

    /// Element-wise integer division `self[(i,j)] /= other[(i,j)]`, rounding
    /// towards zero (Rust counterpart of `igraph_matrix_int_div_elements`,
    /// whose C version traps on a zero divisor).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if
    /// the shapes differ or some divisor is zero (nothing is modified then).
    pub fn div_elements(&mut self, other: &Self) -> Result<()> {
        if other.as_slice().contains(&0) {
            return Err(Error::invalid("integer division by zero"));
        }
        self.zip_op(other, igraph_int_t::wrapping_div)
    }

    /// Product of all elements (one for an empty matrix), wrapping on
    /// overflow (Rust counterpart of `igraph_matrix_int_prod`).
    pub fn prod(&self) -> igraph_int_t {
        self.as_slice().iter().fold(1, |a, &b| a.wrapping_mul(b))
    }

    /// Sum of all elements, wrapping on overflow (Rust counterpart of
    /// `igraph_matrix_int_sum`).
    pub fn sum(&self) -> igraph_int_t {
        self.as_slice().iter().fold(0, |a, &b| a.wrapping_add(b))
    }

    /// The sum of each row (Rust counterpart of `igraph_matrix_int_rowsum`).
    pub fn rowsums(&self) -> Vec<igraph_int_t> {
        self.rows()
            .map(|r| r.iter().fold(0, |a: igraph_int_t, &b| a.wrapping_add(b)))
            .collect()
    }

    /// The sum of each column (Rust counterpart of `igraph_matrix_int_colsum`).
    pub fn colsums(&self) -> Vec<igraph_int_t> {
        self.columns()
            .map(|c| c.iter().fold(0, |a: igraph_int_t, &b| a.wrapping_add(b)))
            .collect()
    }
}

impl igraph_matrix_complex_t {
    fn check_same_shape(&self, other: &Self) -> Result<()> {
        if self.shape() != other.shape() {
            return Err(Error::invalid(format!(
                "matrices of different shapes ({}x{} and {}x{})",
                self.nrow(),
                self.ncol(),
                other.nrow(),
                other.ncol()
            )));
        }
        Ok(())
    }

    /// Builds a complex matrix from the matrices of its real and imaginary
    /// parts ([`igraph_matrix_complex_create`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_complex_create)).
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let re = Matrix::from_rows(&[[1.0, 0.0], [0.0, 1.0]]).unwrap();
    /// let im = Matrix::from_rows(&[[0.0, -1.0], [1.0, 0.0]]).unwrap();
    /// let z = MatrixComplex::from_parts(&re, &im).unwrap();
    /// assert_eq!(z[(1, 0)].im(), 1.0);
    /// let (r, i) = z.realimag();
    /// assert_eq!((r, i), (re, im));
    /// ```
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the shapes differ.
    pub fn from_parts(re: &Matrix, im: &Matrix) -> Result<Self> {
        re.check_same_shape(im)?;
        // `igraph_matrix_complex_create` *initializes* its output.
        let mut res = MaybeUninit::<Self>::uninit();
        crate::igraph_call!(igraph_matrix_complex_create(res.as_mut_ptr(), re, im))?;
        Ok(unsafe { res.assume_init() })
    }

    /// Builds a complex matrix from the moduli `r` and arguments `theta`
    /// (radians) of its elements
    /// ([`igraph_matrix_complex_create_polar`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_complex_create_polar)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the shapes differ.
    pub fn from_polar(r: &Matrix, theta: &Matrix) -> Result<Self> {
        r.check_same_shape(theta)?;
        let mut res = MaybeUninit::<Self>::uninit();
        crate::igraph_call!(igraph_matrix_complex_create_polar(
            res.as_mut_ptr(),
            r,
            theta
        ))?;
        Ok(unsafe { res.assume_init() })
    }

    /// The real parts
    /// ([`igraph_matrix_complex_real`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_complex_real)).
    pub fn real(&self) -> Matrix {
        let mut res = Matrix::new();
        crate::error::check(unsafe { igraph_matrix_complex_real(self, &mut res) })
            .expect("igraph failed to allocate a matrix");
        res
    }

    /// The imaginary parts
    /// ([`igraph_matrix_complex_imag`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_complex_imag)).
    pub fn imag(&self) -> Matrix {
        let mut res = Matrix::new();
        crate::error::check(unsafe { igraph_matrix_complex_imag(self, &mut res) })
            .expect("igraph failed to allocate a matrix");
        res
    }

    /// `(real parts, imaginary parts)` in one call
    /// ([`igraph_matrix_complex_realimag`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_complex_realimag)).
    pub fn realimag(&self) -> (Matrix, Matrix) {
        let (mut re, mut im) = (Matrix::new(), Matrix::new());
        crate::error::check(unsafe { igraph_matrix_complex_realimag(self, &mut re, &mut im) })
            .expect("igraph failed to allocate a matrix");
        (re, im)
    }

    /// Element-wise `self += other`
    /// ([`igraph_matrix_complex_add`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_add)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the shapes differ.
    pub fn add(&mut self, other: &Self) -> Result<()> {
        self.check_same_shape(other)?;
        crate::igraph_call!(igraph_matrix_complex_add(self, other))
    }

    /// Element-wise `self -= other`
    /// ([`igraph_matrix_complex_sub`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_sub)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the shapes differ.
    pub fn sub(&mut self, other: &Self) -> Result<()> {
        self.check_same_shape(other)?;
        crate::igraph_call!(igraph_matrix_complex_sub(self, other))
    }

    /// Element-wise product
    /// ([`igraph_matrix_complex_mul_elements`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_mul_elements)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the shapes differ.
    pub fn mul_elements(&mut self, other: &Self) -> Result<()> {
        self.check_same_shape(other)?;
        crate::igraph_call!(igraph_matrix_complex_mul_elements(self, other))
    }

    /// Element-wise division
    /// ([`igraph_matrix_complex_div_elements`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_div_elements)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the shapes differ.
    pub fn div_elements(&mut self, other: &Self) -> Result<()> {
        self.check_same_shape(other)?;
        crate::igraph_call!(igraph_matrix_complex_div_elements(self, other))
    }

    /// Multiplies every element by `factor`
    /// ([`igraph_matrix_complex_scale`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_scale)).
    pub fn scale(&mut self, factor: igraph_complex_t) {
        unsafe { igraph_matrix_complex_scale(self, factor) }
    }

    /// Adds `value` to every element
    /// ([`igraph_matrix_complex_add_constant`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_add_constant)).
    pub fn add_constant(&mut self, value: igraph_complex_t) {
        unsafe { igraph_matrix_complex_add_constant(self, value) }
    }

    /// Sum of all elements
    /// ([`igraph_matrix_complex_sum`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_sum)).
    pub fn sum(&self) -> igraph_complex_t {
        unsafe { igraph_matrix_complex_sum(self) }
    }

    /// Product of all elements (one for an empty matrix)
    /// ([`igraph_matrix_complex_prod`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_prod)).
    pub fn prod(&self) -> igraph_complex_t {
        unsafe { igraph_matrix_complex_prod(self) }
    }

    /// The sum of each row
    /// ([`igraph_matrix_complex_rowsum`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_rowsum)).
    pub fn rowsums(&self) -> Vec<igraph_complex_t> {
        let mut res = VectorComplex::new();
        crate::error::check(unsafe { igraph_matrix_complex_rowsum(self, &mut res) })
            .expect("igraph failed to allocate a vector");
        res.into()
    }

    /// The sum of each column
    /// ([`igraph_matrix_complex_colsum`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_colsum)).
    pub fn colsums(&self) -> Vec<igraph_complex_t> {
        let mut res = VectorComplex::new();
        crate::error::check(unsafe { igraph_matrix_complex_colsum(self, &mut res) })
            .expect("igraph failed to allocate a vector");
        res.into()
    }

    /// Whether the shapes agree and all elements are equal up to the
    /// relative tolerance `eps`
    /// ([`igraph_matrix_complex_all_almost_e`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_complex_all_almost_e)).
    pub fn all_almost_e(&self, other: &Self, eps: f64) -> bool {
        // igraph takes non-const pointers but only reads the matrices.
        self.shape() == other.shape()
            && unsafe {
                igraph_matrix_complex_all_almost_e(
                    (self as *const Self).cast_mut(),
                    (other as *const Self).cast_mut(),
                    eps,
                )
            }
    }

    /// Replaces the real and imaginary parts smaller in magnitude than the
    /// absolute tolerance `tol` by exact zeros (`tol = 0`: igraph's default,
    /// about `1e-10`)
    /// ([`igraph_matrix_complex_zapsmall`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_matrix_complex_zapsmall)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if `tol < 0`.
    pub fn zapsmall(&mut self, tol: f64) -> Result<()> {
        crate::igraph_call!(igraph_matrix_complex_zapsmall(self, tol))
    }
}

impl igraph_matrix_bool_t {
    /// Number of `true` elements.
    pub fn count_true(&self) -> usize {
        self.as_slice().iter().filter(|&&b| b).count()
    }
}

impl From<&igraph_matrix_int_t> for igraph_matrix_t {
    /// Converts an integer matrix to a real one.
    fn from(m: &igraph_matrix_int_t) -> Self {
        let mut res = Self::zeros(m.nrow(), m.ncol());
        for (r, &x) in res.as_mut_slice().iter_mut().zip(m.as_slice()) {
            *r = x as f64;
        }
        res
    }
}

//! Owned igraph vectors that behave like Rust slices.
//!
//! igraph stores sequences in its own vector types (`igraph_vector_t` for
//! reals, `igraph_vector_int_t` for integers, ...). The types below are
//! *those very C structs*, enriched with Rusty behaviour:
//!
//! - they own their buffer and destroy it on [`Drop`];
//! - they [`Deref`] to a Rust slice, so every slice method (`len`, `iter`,
//!   `sort`, indexing, `contains`, ...) works with zero copies;
//! - they convert from and to [`Vec`], slices and iterators;
//! - they implement [`Clone`], [`Debug`], [`PartialEq`] and [`Default`].
//!
//! | Rust alias      | C type                    | element                |
//! |-----------------|---------------------------|------------------------|
//! | [`Vector`]      | `igraph_vector_t`         | [`f64`]                |
//! | [`VectorInt`]   | `igraph_vector_int_t`     | [`i64`]                |
//! | [`VectorBool`]  | `igraph_vector_bool_t`    | [`bool`]               |
//! | [`VectorChar`]  | `igraph_vector_char_t`    | [`c_char`](std::ffi::c_char) |
//! | [`VectorComplex`] | `igraph_vector_complex_t` | [`igraph_complex_t`] |
//!
//! Borrowed, read-only *views* over Rust slices (no copy at all) are created
//! with `view`, e.g. [`Vector::view`]; they are what the safe wrappers use to
//! hand weights and other inputs to igraph.
//!
//! Most wrappers in this crate return plain `Vec`s, so these types matter
//! mostly when calling raw FFI functions, or to use igraph's vector
//! algorithms (below) on any slice. Lists of vectors are in
//! [`list`](crate::list), matrices in [`matrix`](crate::matrix), sparse
//! matrices in [`linalg`](crate::linalg) ([`SparseMat`](crate::linalg::SparseMat)),
//! and [`misc`](crate::misc) has more numeric helpers, e.g.
//! [`misc::running_mean`](crate::misc::running_mean) and
//! [`misc::power_law_fit`](crate::misc::power_law_fit).
//!
//! This module covers `igraph_vector_pmt.h` (and the non-templated parts of
//! `igraph_vector.h` and `igraph_complex.h`). Besides everything slices
//! offer, the vectors provide igraph's own algorithms:
//!
//! | Group | Methods | Types |
//! |-------|---------|-------|
//! | editing | `insert`, `remove`, `swap_remove`, `remove_section`, `extend_from_slice`, `push`, `pop`, `truncate`, `resize`, `clear` | all |
//! | reordering | `shuffle` (igraph RNG), `permute`, `select`, `move_interval`, `get_interval`, `search` | all |
//! | memory | `reserve`, `capacity`, `shrink_to_fit` | all |
//! | order | `sort`, `reverse_sort`, `sort_ind`, `min`, `max`, `which_min`, `which_max`, `minmax`, `which_minmax`, `binsearch`, `contains_sorted` | real, int, char |
//! | comparison | `lex_cmp`, `colex_cmp`, `all_l`, `all_g`, `all_le`, `all_ge`, `maxdifference`, `is_in_interval`, `any_smaller` | real, int, char |
//! | sorted sets | `intersect_sorted`, `difference_sorted`, `intersection_size_sorted`, `difference_and_intersection_sorted`, `filter_smaller` | real, int, char |
//! | arithmetic | `sum`, `prod`, `cumsum`, `add_constant`, `scale`, `add`, `sub`, `mul`, `div`, `abs` | real, int, complex |
//! | floating point | `all_almost_e`, `zapsmall`, `floor`, `round`, `is_nan`, `is_any_nan`, `is_all_finite` | real (complex) |
//! | complex | `from_parts`, `from_polar`, `real`, `imag`, `realimag`; [`igraph_complex_t`] has `+ - * /`, `abs`, `arg`, `exp`, `ln`, `sqrt`, `pow`, trigonometry, ... | complex |
//!
//! Integer arithmetic is done in Rust with *wrapping* semantics (signed
//! overflow in the C implementation would be undefined behaviour); real and
//! complex arithmetic uses igraph's functions. All index arguments are
//! validated before reaching igraph, which does not check them.
//!
//! ```
//! use igraph::prelude::*;
//!
//! let scores = Vector::from([0.3, 0.9, 0.1, 0.5]);
//! let ranking = scores.sort_ind(Order::Descending);
//! assert_eq!(ranking, vec![1, 3, 0, 2]);
//! assert_eq!(scores.which_max(), Some(1));
//! assert_eq!(scores.select(&[1, 3]).unwrap(), vec![0.9, 0.5]);
//!
//! let a = VectorInt::from([1, 3, 5, 7]);
//! assert_eq!(a.intersect_sorted(&[3, 4, 5]), vec![3, 5]);
//! assert_eq!(a.binsearch(4), Err(2));
//! ```
//!
//! ```
//! use igraph::prelude::*;
//!
//! let mut v: VectorInt = (0..5).collect();
//! v.push(10);
//! v[0] = -1;
//! assert_eq!(v.len(), 6);
//! assert_eq!(v.iter().sum::<i64>(), 19);
//! assert_eq!(Vec::from(v), vec![-1, 1, 2, 3, 4, 10]);
//!
//! let weights = [0.5, 1.5];
//! let view = Vector::view(&weights);
//! assert_eq!(&view[..], &weights[..]);
//! ```

use crate::ffi::*;
use std::{
    fmt,
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
    ops::{Deref, DerefMut},
};

/// Owned vector of reals (`igraph_vector_t`), see the [module docs](self).
pub type Vector = igraph_vector_t;
/// Owned vector of integers (`igraph_vector_int_t`), see the [module docs](self).
pub type VectorInt = igraph_vector_int_t;
/// Owned vector of booleans (`igraph_vector_bool_t`), see the [module docs](self).
pub type VectorBool = igraph_vector_bool_t;
/// Owned vector of chars (`igraph_vector_char_t`), see the [module docs](self).
pub type VectorChar = igraph_vector_char_t;
/// Owned vector of complex numbers (`igraph_vector_complex_t`), see the [module docs](self).
pub type VectorComplex = igraph_vector_complex_t;

/// A read-only view of a Rust slice as an igraph vector, without copying.
///
/// It dereferences to the underlying C vector type (e.g. [`Vector`]), so
/// `&*view` can be passed where igraph expects a `const igraph_vector_t *`.
/// The view never frees the borrowed memory.
pub struct View<'a, V> {
    raw: ManuallyDrop<V>,
    _borrow: PhantomData<&'a ()>,
}

impl<V> Deref for View<'_, V> {
    type Target = V;
    fn deref(&self) -> &V {
        &self.raw
    }
}

impl<V: fmt::Debug> fmt::Debug for View<'_, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (*self.raw).fmt(f)
    }
}

impl<V> View<'_, V> {
    /// Raw pointer to the viewed C vector, for FFI calls.
    pub fn as_ptr(&self) -> *const V {
        &*self.raw
    }

    /// Wraps a raw igraph *view* struct (one that does not own its buffer,
    /// e.g. returned by `igraph_matrix_view`), so that it is never destroyed.
    ///
    /// # Safety
    /// `raw` must point into memory that stays valid and unmodified for the
    /// lifetime `'a` of the returned view.
    pub(crate) unsafe fn from_raw<'a>(raw: V) -> View<'a, V> {
        View {
            raw: ManuallyDrop::new(raw),
            _borrow: PhantomData,
        }
    }
}

macro_rules! impl_vector {
    (
        $ty:ident, $elem:ty,
        init = $init:ident, init_array = $init_array:ident, init_copy = $init_copy:ident,
        destroy = $destroy:ident, push_back = $push:ident, resize = $resize:ident,
        reserve = $reserve:ident, view = $view:ident
    ) => {
        impl $ty {
            /// Creates an empty vector.
            pub fn new() -> Self {
                Self::zeros(0)
            }

            /// Creates a vector of `len` zero (default) elements
            /// ([`igraph_vector_init`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_init)).
            ///
            /// # Panics
            /// If igraph cannot allocate the vector, or `len` does not fit in
            /// an `i64`, like [`Vec::with_capacity`].
            pub fn zeros(len: usize) -> Self {
                crate::error::ensure_init();
                let mut raw = MaybeUninit::<Self>::uninit();
                crate::error::check(unsafe {
                    $init(raw.as_mut_ptr(), crate::error::int_size(len))
                })
                .expect("igraph failed to allocate a vector");
                unsafe { raw.assume_init() }
            }

            /// Creates a vector of `len` zero (default) elements.
            #[deprecated(note = "it creates a zero-filled vector of length `size`: use `zeros`")]
            pub fn with_capacity(size: usize) -> Self {
                Self::zeros(size)
            }

            /// Creates a vector by copying the elements of a slice
            /// ([`igraph_vector_init_array`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_init_array)).
            pub fn from_slice(data: &[$elem]) -> Self {
                crate::error::ensure_init();
                let mut raw = MaybeUninit::<Self>::uninit();
                crate::error::check(unsafe {
                    $init_array(raw.as_mut_ptr(), data.as_ptr(), data.len() as igraph_int_t)
                })
                .expect("igraph failed to allocate a vector");
                unsafe { raw.assume_init() }
            }

            /// Creates a read-only view of `data`, without copying it
            /// ([`igraph_vector_view`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_view)); the view
            /// dereferences to this vector type and never frees `data`.
            pub fn view(data: &[$elem]) -> View<'_, Self> {
                let raw = unsafe { $view(data.as_ptr(), data.len() as igraph_int_t) };
                View {
                    raw: ManuallyDrop::new(raw),
                    _borrow: PhantomData,
                }
            }

            /// Number of elements (also available through [`Deref`] as `len`).
            pub fn size(&self) -> usize {
                self.as_slice().len()
            }

            /// The elements as a slice.
            pub fn as_slice(&self) -> &[$elem] {
                if self.stor_begin.is_null() {
                    return &[];
                }
                unsafe {
                    let len = self.end.offset_from(self.stor_begin) as usize;
                    std::slice::from_raw_parts(self.stor_begin, len)
                }
            }

            /// The elements as a mutable slice.
            pub fn as_mut_slice(&mut self) -> &mut [$elem] {
                if self.stor_begin.is_null() {
                    return &mut [];
                }
                unsafe {
                    let len = self.end.offset_from(self.stor_begin) as usize;
                    std::slice::from_raw_parts_mut(self.stor_begin, len)
                }
            }

            /// Appends an element at the end, in amortized O(1)
            /// ([`igraph_vector_push_back`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_push_back)).
            pub fn push(&mut self, value: $elem) {
                crate::error::check(unsafe { $push(self, value) })
                    .expect("igraph failed to grow a vector");
            }

            /// Removes and returns the last element, if any.
            pub fn pop(&mut self) -> Option<$elem> {
                let len = self.size();
                if len == 0 {
                    return None;
                }
                let last = unsafe { std::ptr::read(self.stor_begin.add(len - 1)) };
                self.truncate(len - 1);
                Some(last)
            }

            /// Shortens the vector to `len` elements (no-op if already shorter).
            pub fn truncate(&mut self, len: usize) {
                if len < self.size() {
                    self.end = unsafe { self.stor_begin.add(len) };
                }
            }

            /// Removes all the elements, keeping the allocated storage.
            pub fn clear(&mut self) {
                self.truncate(0);
            }

            /// Resizes the vector to `len` elements
            /// ([`igraph_vector_resize`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_resize)); new
            /// elements are unspecified by igraph, so this wrapper zeroes them.
            pub fn resize(&mut self, len: usize) {
                let old = self.size();
                crate::error::check(unsafe { $resize(self, crate::error::int_size(len)) })
                    .expect("igraph failed to resize a vector");
                if len > old {
                    unsafe { std::ptr::write_bytes(self.stor_begin.add(old), 0, len - old) };
                }
            }

            /// Reserves storage for at least `capacity` elements in total
            /// ([`igraph_vector_reserve`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_reserve)).
            pub fn reserve(&mut self, capacity: usize) {
                crate::error::check(unsafe { $reserve(self, crate::error::int_size(capacity)) })
                    .expect("igraph failed to reserve vector storage");
            }

            /// Copies the elements into a [`Vec`].
            pub fn to_vec(&self) -> Vec<$elem> {
                self.as_slice().to_vec()
            }
        }

        impl Drop for $ty {
            /// Frees the storage with the corresponding `igraph_*_destroy`.
            ///
            /// Never let a raw *view* created with the unsafe `igraph_*_view`
            /// functions be dropped: use the safe `view` constructor instead.
            fn drop(&mut self) {
                if !self.stor_begin.is_null() {
                    unsafe { $destroy(self) };
                    self.stor_begin = std::ptr::null_mut();
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
                // A vector may have been moved to a thread that never called
                // into igraph: install the error handler before allocating.
                crate::error::ensure_init();
                let mut raw = MaybeUninit::<Self>::uninit();
                crate::error::check(unsafe { $init_copy(raw.as_mut_ptr(), self) })
                    .expect("igraph failed to copy a vector");
                unsafe { raw.assume_init() }
            }
        }

        impl Deref for $ty {
            type Target = [$elem];
            fn deref(&self) -> &[$elem] {
                self.as_slice()
            }
        }

        impl DerefMut for $ty {
            fn deref_mut(&mut self) -> &mut [$elem] {
                self.as_mut_slice()
            }
        }

        impl AsRef<[$elem]> for $ty {
            fn as_ref(&self) -> &[$elem] {
                self.as_slice()
            }
        }

        impl PartialEq for $ty {
            fn eq(&self, other: &Self) -> bool {
                self.as_slice() == other.as_slice()
            }
        }

        impl PartialEq<[$elem]> for $ty {
            fn eq(&self, other: &[$elem]) -> bool {
                self.as_slice() == other
            }
        }

        impl PartialEq<Vec<$elem>> for $ty {
            fn eq(&self, other: &Vec<$elem>) -> bool {
                self.as_slice() == other.as_slice()
            }
        }

        impl From<&[$elem]> for $ty {
            fn from(data: &[$elem]) -> Self {
                Self::from_slice(data)
            }
        }

        impl<const N: usize> From<[$elem; N]> for $ty {
            fn from(data: [$elem; N]) -> Self {
                Self::from_slice(&data)
            }
        }

        impl From<Vec<$elem>> for $ty {
            fn from(data: Vec<$elem>) -> Self {
                Self::from_slice(&data)
            }
        }

        impl From<&Vec<$elem>> for $ty {
            fn from(data: &Vec<$elem>) -> Self {
                Self::from_slice(data)
            }
        }

        impl From<$ty> for Vec<$elem> {
            fn from(v: $ty) -> Self {
                v.to_vec()
            }
        }

        impl From<&$ty> for Vec<$elem> {
            fn from(v: &$ty) -> Self {
                v.to_vec()
            }
        }

        impl FromIterator<$elem> for $ty {
            fn from_iter<I: IntoIterator<Item = $elem>>(iter: I) -> Self {
                let mut v = Self::new();
                v.extend(iter);
                v
            }
        }

        impl Extend<$elem> for $ty {
            fn extend<I: IntoIterator<Item = $elem>>(&mut self, iter: I) {
                let iter = iter.into_iter();
                let (lower, _) = iter.size_hint();
                self.reserve(self.size() + lower);
                for x in iter {
                    self.push(x);
                }
            }
        }

        impl<'a> IntoIterator for &'a $ty {
            type Item = &'a $elem;
            type IntoIter = std::slice::Iter<'a, $elem>;
            fn into_iter(self) -> Self::IntoIter {
                self.as_slice().iter()
            }
        }

        impl IntoIterator for $ty {
            type Item = $elem;
            type IntoIter = std::vec::IntoIter<$elem>;
            fn into_iter(self) -> Self::IntoIter {
                self.to_vec().into_iter()
            }
        }

        // The buffer is uniquely owned, so moving it across threads is fine.
        unsafe impl Send for $ty {}
        unsafe impl Sync for $ty {}
    };
}

impl_vector!(
    igraph_vector_t,
    igraph_real_t,
    init = igraph_vector_init,
    init_array = igraph_vector_init_array,
    init_copy = igraph_vector_init_copy,
    destroy = igraph_vector_destroy,
    push_back = igraph_vector_push_back,
    resize = igraph_vector_resize,
    reserve = igraph_vector_reserve,
    view = igraph_vector_view
);
impl_vector!(
    igraph_vector_int_t,
    igraph_int_t,
    init = igraph_vector_int_init,
    init_array = igraph_vector_int_init_array,
    init_copy = igraph_vector_int_init_copy,
    destroy = igraph_vector_int_destroy,
    push_back = igraph_vector_int_push_back,
    resize = igraph_vector_int_resize,
    reserve = igraph_vector_int_reserve,
    view = igraph_vector_int_view
);
impl_vector!(
    igraph_vector_bool_t,
    igraph_bool_t,
    init = igraph_vector_bool_init,
    init_array = igraph_vector_bool_init_array,
    init_copy = igraph_vector_bool_init_copy,
    destroy = igraph_vector_bool_destroy,
    push_back = igraph_vector_bool_push_back,
    resize = igraph_vector_bool_resize,
    reserve = igraph_vector_bool_reserve,
    view = igraph_vector_bool_view
);
impl_vector!(
    igraph_vector_char_t,
    std::ffi::c_char,
    init = igraph_vector_char_init,
    init_array = igraph_vector_char_init_array,
    init_copy = igraph_vector_char_init_copy,
    destroy = igraph_vector_char_destroy,
    push_back = igraph_vector_char_push_back,
    resize = igraph_vector_char_resize,
    reserve = igraph_vector_char_reserve,
    view = igraph_vector_char_view
);
impl_vector!(
    igraph_vector_complex_t,
    igraph_complex_t,
    init = igraph_vector_complex_init,
    init_array = igraph_vector_complex_init_array,
    init_copy = igraph_vector_complex_init_copy,
    destroy = igraph_vector_complex_destroy,
    push_back = igraph_vector_complex_push_back,
    resize = igraph_vector_complex_resize,
    reserve = igraph_vector_complex_reserve,
    view = igraph_vector_complex_view
);

// `Debug` for the vectors is derived by bindgen on the raw struct (showing
// pointers); `Display` shows the elements instead.
macro_rules! impl_display {
    ($($ty:ident),*) => {$(
        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_list().entries(self.as_slice().iter()).finish()
            }
        }
    )*};
}
impl_display!(
    igraph_vector_t,
    igraph_vector_int_t,
    igraph_vector_bool_t,
    igraph_vector_char_t
);

impl Clone for igraph_complex_t {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for igraph_complex_t {}
impl PartialEq for igraph_complex_t {
    fn eq(&self, other: &Self) -> bool {
        self.dat == other.dat
    }
}

impl igraph_complex_t {
    /// Builds a complex number from its real and imaginary parts.
    pub const fn new(re: f64, im: f64) -> Self {
        Self { dat: [re, im] }
    }
    /// The real part.
    pub const fn re(&self) -> f64 {
        self.dat[0]
    }
    /// The imaginary part.
    pub const fn im(&self) -> f64 {
        self.dat[1]
    }
}

impl From<(f64, f64)> for igraph_complex_t {
    fn from((re, im): (f64, f64)) -> Self {
        Self::new(re, im)
    }
}

// ---------------------------------------------------------------------------
// Operations available for every element type (`igraph_vector_pmt.h`).
// ---------------------------------------------------------------------------

/// Checks that `index` only contains valid, distinct positions of a vector of
/// length `len` (what `igraph_vector_*_permute` silently assumes).
fn check_permutation(index: &[igraph_int_t], len: usize) -> crate::error::Result<()> {
    if index.len() > len {
        return Err(crate::error::Error::invalid(format!(
            "permutation index has {} entries but the vector only {len}",
            index.len()
        )));
    }
    let mut seen = vec![false; len];
    for &i in index {
        if i < 0 || i as usize >= len || std::mem::replace(&mut seen[i as usize], true) {
            return Err(crate::error::Error::invalid(format!(
                "invalid or repeated position {i} in a permutation of length {len}"
            )));
        }
    }
    Ok(())
}

/// Checks that every entry of `index` is a valid position for length `len`.
fn check_indices(index: &[igraph_int_t], len: usize) -> crate::error::Result<()> {
    match index.iter().find(|&&i| i < 0 || i as usize >= len) {
        Some(i) => Err(crate::error::Error::invalid(format!(
            "index {i} out of bounds for a vector of length {len}"
        ))),
        None => Ok(()),
    }
}

macro_rules! impl_vector_common {
    (
        $ty:ident, $elem:ty,
        insert = $insert:ident, remove = $remove:ident, remove_fast = $remove_fast:ident,
        remove_section = $remove_section:ident, append = $append:ident,
        shuffle = $shuffle:ident, index = $index:ident,
        search = $search:ident, capacity = $capacity:ident, resize_min = $resize_min:ident,
        move_interval = $move_interval:ident, get_interval = $get_interval:ident
    ) => {
        impl $ty {
            /// Inserts `value` at position `pos`, shifting the following
            /// elements to the right, like [`Vec::insert`]
            /// ([`igraph_vector_insert`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_insert)).
            ///
            /// # Panics
            /// If `pos > len`.
            pub fn insert(&mut self, pos: usize, value: $elem) {
                let len = self.size();
                assert!(
                    pos <= len,
                    "insertion index {pos} out of bounds (len {len})"
                );
                crate::error::check(unsafe { $insert(self, pos as igraph_int_t, value) })
                    .expect("igraph failed to grow a vector");
            }

            /// Removes and returns the element at `pos`, shifting the
            /// following ones to the left, like [`Vec::remove`]
            /// ([`igraph_vector_remove`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_remove)).
            ///
            /// # Panics
            /// If `pos >= len`.
            pub fn remove(&mut self, pos: usize) -> $elem {
                let len = self.size();
                assert!(pos < len, "removal index {pos} out of bounds (len {len})");
                let value = self.as_slice()[pos];
                unsafe { $remove(self, pos as igraph_int_t) };
                value
            }

            /// Removes the element at `pos` in O(1) by moving the last
            /// element into its place, like [`Vec::swap_remove`]
            /// (`igraph_vector_remove_fast`, undocumented in `igraph_vector.h`).
            ///
            /// # Panics
            /// If `pos >= len`.
            pub fn swap_remove(&mut self, pos: usize) -> $elem {
                let len = self.size();
                assert!(pos < len, "removal index {pos} out of bounds (len {len})");
                let value = self.as_slice()[pos];
                unsafe { $remove_fast(self, pos as igraph_int_t) };
                value
            }

            /// Removes the elements in `range`
            /// ([`igraph_vector_remove_section`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_remove_section)).
            ///
            /// # Panics
            /// If the range is decreasing or goes past the end.
            pub fn remove_section(&mut self, range: std::ops::Range<usize>) {
                let len = self.size();
                assert!(
                    range.start <= range.end && range.end <= len,
                    "section {range:?} out of bounds (len {len})"
                );
                unsafe {
                    $remove_section(self, range.start as igraph_int_t, range.end as igraph_int_t)
                };
            }

            /// Appends a copy of all the elements of `other`
            /// ([`igraph_vector_append`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_append)).
            pub fn extend_from_slice(&mut self, other: &[$elem]) {
                let view = Self::view(other);
                crate::error::check(unsafe { $append(self, view.as_ptr()) })
                    .expect("igraph failed to grow a vector");
            }

            /// Shuffles the elements in place with the Fisher-Yates algorithm,
            /// drawing from igraph's default random number generator, so the
            /// outcome is reproducible with [`rng::seed`](crate::rng::seed)
            /// ([`igraph_vector_shuffle`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_shuffle)).
            ///
            /// ```
            /// use igraph::prelude::*;
            /// rng::seed(42).unwrap();
            /// let mut deck: VectorInt = (0..52).collect();
            /// deck.shuffle();
            /// let mut again: VectorInt = (0..52).collect();
            /// rng::seed(42).unwrap();
            /// again.shuffle();
            /// assert_eq!(deck, again);
            /// deck.sort();
            /// assert_eq!(deck, (0..52).collect::<Vec<i64>>());
            /// ```
            pub fn shuffle(&mut self) {
                crate::error::ensure_init();
                if !self.stor_begin.is_null() {
                    unsafe { $shuffle(self) };
                }
            }

            /// Permutes the elements in place so that the element at position
            /// `index[i]` moves to position `i`, with the semantics of
            /// [`igraph_vector_permute`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_permute).
            ///
            /// The index is compatible with the one returned by `sort_ind`:
            /// permuting by it sorts the vector. If `index` is *shorter* than
            /// the vector, the elements not mentioned are dropped.
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue)
            /// if `index` has out-of-range or repeated positions.
            pub fn permute(&mut self, index: &[igraph_int_t]) -> crate::error::Result<()> {
                check_permutation(index, self.size())?;
                // Done in Rust for every type: igraph 1.0.0 and 1.0.1 declare,
                // but do not export, `igraph_vector_{bool,complex}_permute`.
                let permuted: Vec<$elem> =
                    index.iter().map(|&i| self.as_slice()[i as usize]).collect();
                self.truncate(permuted.len());
                self.as_mut_slice().copy_from_slice(&permuted);
                Ok(())
            }

            /// A new vector with the elements at the given positions, i.e.
            /// `result[i] = self[index[i]]`; positions may repeat
            /// ([`igraph_vector_index`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_index)).
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue)
            /// if some position is out of bounds.
            pub fn select(&self, index: &[igraph_int_t]) -> crate::error::Result<Self> {
                check_indices(index, self.size())?;
                let idx = VectorInt::view(index);
                let mut res = Self::new();
                crate::igraph_call!($index(self, &mut res, idx.as_ptr()))?;
                Ok(res)
            }

            /// Position of the first occurrence of `value` at or after `from`,
            /// if any ([`igraph_vector_search`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_search)).
            pub fn search(&self, from: usize, value: $elem) -> Option<usize> {
                if from >= self.size() {
                    return None;
                }
                let mut pos = 0;
                unsafe { $search(self, from as igraph_int_t, value, &mut pos) }
                    .then_some(pos as usize)
            }

            /// Number of elements the vector can hold without reallocating
            /// ([`igraph_vector_capacity`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_capacity)).
            pub fn capacity(&self) -> usize {
                if self.stor_begin.is_null() {
                    0
                } else {
                    unsafe { $capacity(self) as usize }
                }
            }

            /// Frees the unused storage, so that the capacity equals the
            /// length ([`igraph_vector_resize_min`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_resize_min)).
            pub fn shrink_to_fit(&mut self) {
                if !self.stor_begin.is_null() {
                    unsafe { $resize_min(self) };
                }
            }

            /// Copies the elements in `src` to the position starting at
            /// `dest`, like [`slice::copy_within`]; the regions may overlap
            /// (`igraph_vector_move_interval`, undocumented in `igraph_vector.h`).
            ///
            /// # Panics
            /// If either region goes past the end of the vector.
            pub fn move_interval(&mut self, src: std::ops::Range<usize>, dest: usize) {
                let len = self.size();
                assert!(
                    src.start <= src.end && src.end <= len && dest + (src.end - src.start) <= len,
                    "interval {src:?} -> {dest} out of bounds (len {len})"
                );
                crate::error::check(unsafe {
                    $move_interval(
                        self,
                        src.start as igraph_int_t,
                        src.end as igraph_int_t,
                        dest as igraph_int_t,
                    )
                })
                .expect("igraph_vector_move_interval cannot fail");
            }

            /// A new vector with a copy of the elements in `range`
            /// (`igraph_vector_get_interval`, undocumented in `igraph_vector.h`).
            ///
            /// # Panics
            /// If the range goes past the end.
            pub fn get_interval(&self, range: std::ops::Range<usize>) -> Self {
                let len = self.size();
                assert!(
                    range.start <= range.end && range.end <= len,
                    "interval {range:?} out of bounds (len {len})"
                );
                let mut res = Self::new();
                crate::error::check(unsafe {
                    $get_interval(
                        self,
                        &mut res,
                        range.start as igraph_int_t,
                        range.end as igraph_int_t,
                    )
                })
                .expect("igraph failed to allocate a vector");
                res
            }
        }
    };
}

impl_vector_common!(
    igraph_vector_t,
    igraph_real_t,
    insert = igraph_vector_insert,
    remove = igraph_vector_remove,
    remove_fast = igraph_vector_remove_fast,
    remove_section = igraph_vector_remove_section,
    append = igraph_vector_append,
    shuffle = igraph_vector_shuffle,
    index = igraph_vector_index,
    search = igraph_vector_search,
    capacity = igraph_vector_capacity,
    resize_min = igraph_vector_resize_min,
    move_interval = igraph_vector_move_interval,
    get_interval = igraph_vector_get_interval
);
impl_vector_common!(
    igraph_vector_int_t,
    igraph_int_t,
    insert = igraph_vector_int_insert,
    remove = igraph_vector_int_remove,
    remove_fast = igraph_vector_int_remove_fast,
    remove_section = igraph_vector_int_remove_section,
    append = igraph_vector_int_append,
    shuffle = igraph_vector_int_shuffle,
    index = igraph_vector_int_index,
    search = igraph_vector_int_search,
    capacity = igraph_vector_int_capacity,
    resize_min = igraph_vector_int_resize_min,
    move_interval = igraph_vector_int_move_interval,
    get_interval = igraph_vector_int_get_interval
);
impl_vector_common!(
    igraph_vector_bool_t,
    igraph_bool_t,
    insert = igraph_vector_bool_insert,
    remove = igraph_vector_bool_remove,
    remove_fast = igraph_vector_bool_remove_fast,
    remove_section = igraph_vector_bool_remove_section,
    append = igraph_vector_bool_append,
    shuffle = igraph_vector_bool_shuffle,
    index = igraph_vector_bool_index,
    search = igraph_vector_bool_search,
    capacity = igraph_vector_bool_capacity,
    resize_min = igraph_vector_bool_resize_min,
    move_interval = igraph_vector_bool_move_interval,
    get_interval = igraph_vector_bool_get_interval
);
impl_vector_common!(
    igraph_vector_char_t,
    std::ffi::c_char,
    insert = igraph_vector_char_insert,
    remove = igraph_vector_char_remove,
    remove_fast = igraph_vector_char_remove_fast,
    remove_section = igraph_vector_char_remove_section,
    append = igraph_vector_char_append,
    shuffle = igraph_vector_char_shuffle,
    index = igraph_vector_char_index,
    search = igraph_vector_char_search,
    capacity = igraph_vector_char_capacity,
    resize_min = igraph_vector_char_resize_min,
    move_interval = igraph_vector_char_move_interval,
    get_interval = igraph_vector_char_get_interval
);
impl_vector_common!(
    igraph_vector_complex_t,
    igraph_complex_t,
    insert = igraph_vector_complex_insert,
    remove = igraph_vector_complex_remove,
    remove_fast = igraph_vector_complex_remove_fast,
    remove_section = igraph_vector_complex_remove_section,
    append = igraph_vector_complex_append,
    shuffle = igraph_vector_complex_shuffle,
    index = igraph_vector_complex_index,
    search = igraph_vector_complex_search,
    capacity = igraph_vector_complex_capacity,
    resize_min = igraph_vector_complex_resize_min,
    move_interval = igraph_vector_complex_move_interval,
    get_interval = igraph_vector_complex_get_interval
);

// ---------------------------------------------------------------------------
// Ordered element types: reals, integers, chars.
// ---------------------------------------------------------------------------

/// Whether `igraph_vector_init_range(start, end)` creates a non-empty real
/// vector; panics if its length `trunc(end - start)` does not fit an `i64`.
fn real_range_nonempty(start: f64, end: f64) -> bool {
    let len = end - start;
    if len.is_nan() || len < 1.0 {
        return false;
    }
    assert!(
        len < 9.223_372_036_854_775e18,
        "capacity overflow: a real range of length {len}"
    );
    true
}

/// Whether `igraph_vector_int_init_range(start, end)` creates a non-empty
/// vector; panics if `end - start` overflows.
fn int_range_nonempty(start: igraph_int_t, end: igraph_int_t) -> bool {
    if end <= start {
        return false;
    }
    assert!(
        end.checked_sub(start).is_some(),
        "capacity overflow: the range {start}..{end} is too long"
    );
    true
}

/// Whether `igraph_vector_char_init_range(start, end)` creates a non-empty
/// vector (C promotes chars to `int`, so there is no overflow).
fn char_range_nonempty(start: std::ffi::c_char, end: std::ffi::c_char) -> bool {
    end > start
}

macro_rules! impl_vector_ordered {
    (
        $ty:ident, $elem:ty,
        init_range = $init_range:ident, range_nonempty = $range_nonempty:ident, sort = $sort:ident, reverse_sort = $reverse_sort:ident,
        sort_ind = $sort_ind:ident, min = $min:ident, max = $max:ident,
        which_min = $which_min:ident, which_max = $which_max:ident,
        binsearch = $binsearch:ident, contains_sorted = $contains_sorted:ident,
        isininterval = $isininterval:ident, any_smaller = $any_smaller:ident,
        maxdifference = $maxdifference:ident, lex_cmp = $lex_cmp:ident, colex_cmp = $colex_cmp:ident,
        all_l = $all_l:ident, all_g = $all_g:ident, all_le = $all_le:ident, all_ge = $all_ge:ident,
        difference_sorted = $difference_sorted:ident, intersect_sorted = $intersect_sorted:ident,
        intersection_size_sorted = $intersection_size_sorted:ident,
        difference_and_intersection_sorted = $dais:ident, filter_smaller = $filter_smaller:ident
    ) => {
        impl $ty {
            /// A vector with the values `start, start + 1, ...`, of length
            /// `end - start` (rounded towards zero for reals), i.e. the
            /// values smaller than `end` for integers; empty if `end <= start`
            /// ([`igraph_vector_init_range`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_init_range)).
            ///
            /// # Panics
            /// If the length does not fit in an `i64` (e.g. an infinite real
            /// range), like [`Vec::with_capacity`]. A NaN bound gives an
            /// empty vector.
            pub fn range(start: $elem, end: $elem) -> Self {
                // igraph computes the length as `end - start` with a plain C
                // conversion: overflow (or a NaN / infinite real) would be
                // undefined behaviour, so the length is validated here.
                if !$range_nonempty(start, end) {
                    return Self::new();
                }
                crate::error::ensure_init();
                let mut raw = MaybeUninit::<Self>::uninit();
                crate::error::check(unsafe { $init_range(raw.as_mut_ptr(), start, end) })
                    .expect("igraph failed to allocate a vector");
                unsafe { raw.assume_init() }
            }

            /// Sorts the elements in ascending order
            /// ([`igraph_vector_sort`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_sort)).
            ///
            /// For real vectors, the position of NaN values after sorting is
            /// unspecified.
            pub fn sort(&mut self) {
                if !self.is_empty() {
                    unsafe { $sort(self) };
                }
            }

            /// Sorts the elements in descending order
            /// ([`igraph_vector_reverse_sort`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_reverse_sort)).
            pub fn reverse_sort(&mut self) {
                if !self.is_empty() {
                    unsafe { $reverse_sort(self) };
                }
            }

            /// The permutation that sorts the vector (an "argsort"): the
            /// position of the smallest (or largest, with
            /// [`Order::Descending`](crate::constants::Order::Descending))
            /// element first, and so on. The sort is stable
            /// ([`igraph_vector_sort_ind`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_sort_ind)).
            ///
            /// Passing the result to `permute` sorts the vector.
            pub fn sort_ind(&self, order: crate::constants::Order) -> VectorInt {
                let mut res = VectorInt::new();
                crate::error::check(unsafe { $sort_ind(self, &mut res, order.into()) })
                    .expect("igraph failed to allocate a vector");
                res
            }

            /// The smallest element, or `None` if the vector is empty
            /// ([`igraph_vector_min`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_min)).
            /// For real vectors containing NaN, the result is NaN.
            pub fn min(&self) -> Option<$elem> {
                (!self.is_empty()).then(|| unsafe { $min(self) })
            }

            /// The largest element, or `None` if the vector is empty
            /// ([`igraph_vector_max`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_max)).
            /// For real vectors containing NaN, the result is NaN.
            pub fn max(&self) -> Option<$elem> {
                (!self.is_empty()).then(|| unsafe { $max(self) })
            }

            /// Position of the (first) smallest element, or `None` if empty
            /// ([`igraph_vector_which_min`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_which_min)).
            pub fn which_min(&self) -> Option<usize> {
                (!self.is_empty()).then(|| unsafe { $which_min(self) } as usize)
            }

            /// Position of the (first) largest element, or `None` if empty
            /// ([`igraph_vector_which_max`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_which_max)).
            pub fn which_max(&self) -> Option<usize> {
                (!self.is_empty()).then(|| unsafe { $which_max(self) } as usize)
            }

            /// `(min, max)` in one pass, or `None` if empty
            /// (same as `igraph_vector_minmax`).
            pub fn minmax(&self) -> Option<($elem, $elem)> {
                Some((self.min()?, self.max()?))
            }

            /// `(which_min, which_max)`, or `None` if empty
            /// (same as `igraph_vector_which_minmax`).
            pub fn which_minmax(&self) -> Option<(usize, usize)> {
                Some((self.which_min()?, self.which_max()?))
            }

            /// Binary search in a **sorted** vector: `Ok(pos)` with the
            /// position of an element equal to `value`, or `Err(pos)` with the
            /// position where it could be inserted keeping the order, like
            /// [`slice::binary_search`]
            /// ([`igraph_vector_binsearch`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_binsearch)).
            pub fn binsearch(&self, value: $elem) -> std::result::Result<usize, usize> {
                if self.is_empty() {
                    return Err(0);
                }
                let mut pos = 0;
                if unsafe { $binsearch(self, value, &mut pos) } {
                    Ok(pos as usize)
                } else {
                    Err(pos as usize)
                }
            }

            /// Whether a **sorted** vector contains `value`, in O(log n)
            /// ([`igraph_vector_contains_sorted`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_contains_sorted)).
            pub fn contains_sorted(&self, value: $elem) -> bool {
                !self.is_empty() && unsafe { $contains_sorted(self, value) }
            }

            /// Whether all elements lie in the closed interval `[low, high]`
            /// (true for an empty vector, false if any element is NaN)
            /// ([`igraph_vector_isininterval`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_isininterval)).
            pub fn is_in_interval(&self, low: $elem, high: $elem) -> bool {
                self.is_empty() || unsafe { $isininterval(self, low, high) }
            }

            /// Whether some element is strictly smaller than `limit`
            /// (`igraph_vector_any_smaller`, undocumented in `igraph_vector.h`).
            pub fn any_smaller(&self, limit: $elem) -> bool {
                !self.is_empty() && unsafe { $any_smaller(self, limit) }
            }

            /// The largest absolute difference between corresponding
            /// elements; the extra elements of the longer vector are ignored
            /// ([`igraph_vector_maxdifference`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_maxdifference)).
            pub fn maxdifference(&self, other: &[$elem]) -> f64 {
                if self.is_empty() || other.is_empty() {
                    return 0.0;
                }
                let o = Self::view(other);
                unsafe { $maxdifference(self, o.as_ptr()) }
            }

            /// Lexicographic comparison: the first differing element decides,
            /// and a proper prefix comes first
            /// ([`igraph_vector_lex_cmp`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_lex_cmp)).
            pub fn lex_cmp(&self, other: &[$elem]) -> std::cmp::Ordering {
                let o = Self::view(other);
                unsafe { $lex_cmp(self, o.as_ptr()) }.cmp(&0)
            }

            /// Colexicographic comparison: like [`lex_cmp`](Self::lex_cmp)
            /// but starting from the *last* elements
            /// ([`igraph_vector_colex_cmp`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_colex_cmp)).
            pub fn colex_cmp(&self, other: &[$elem]) -> std::cmp::Ordering {
                let o = Self::view(other);
                unsafe { $colex_cmp(self, o.as_ptr()) }.cmp(&0)
            }

            /// Whether each element is `<` the corresponding one in `other`
            /// (false if the lengths differ or NaN is involved)
            /// ([`igraph_vector_all_l`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_all_l)).
            pub fn all_l(&self, other: &[$elem]) -> bool {
                self.elementwise(other, |a, b| unsafe { $all_l(a, b) })
            }

            /// Whether each element is `>` the corresponding one in `other`
            /// ([`igraph_vector_all_g`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_all_g)).
            pub fn all_g(&self, other: &[$elem]) -> bool {
                self.elementwise(other, |a, b| unsafe { $all_g(a, b) })
            }

            /// Whether each element is `<=` the corresponding one in `other`
            /// ([`igraph_vector_all_le`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_all_le)).
            pub fn all_le(&self, other: &[$elem]) -> bool {
                self.elementwise(other, |a, b| unsafe { $all_le(a, b) })
            }

            /// Whether each element is `>=` the corresponding one in `other`
            /// ([`igraph_vector_all_ge`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_all_ge)).
            pub fn all_ge(&self, other: &[$elem]) -> bool {
                self.elementwise(other, |a, b| unsafe { $all_ge(a, b) })
            }

            fn elementwise(
                &self,
                other: &[$elem],
                f: impl FnOnce(*const Self, *const Self) -> bool,
            ) -> bool {
                if self.size() != other.len() {
                    return false;
                }
                if other.is_empty() {
                    return true;
                }
                let o = Self::view(other);
                f(self, o.as_ptr())
            }

            /// The elements of this **sorted** vector that are not in the
            /// **sorted** `other` (a multiset difference)
            /// ([`igraph_vector_difference_sorted`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_difference_sorted)).
            pub fn difference_sorted(&self, other: &[$elem]) -> Self {
                let o = Self::view(other);
                let mut res = Self::new();
                crate::error::check(unsafe { $difference_sorted(self, o.as_ptr(), &mut res) })
                    .expect("igraph failed to allocate a vector");
                res
            }

            /// The common elements of this **sorted** vector and the
            /// **sorted** `other` (a multiset intersection)
            /// ([`igraph_vector_intersect_sorted`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_intersect_sorted)).
            pub fn intersect_sorted(&self, other: &[$elem]) -> Self {
                let o = Self::view(other);
                let mut res = Self::new();
                crate::error::check(unsafe { $intersect_sorted(self, o.as_ptr(), &mut res) })
                    .expect("igraph failed to allocate a vector");
                res
            }

            /// Size of the intersection of two **sorted** vectors, without
            /// building it
            /// ([`igraph_vector_intersection_size_sorted`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_intersection_size_sorted)).
            pub fn intersection_size_sorted(&self, other: &[$elem]) -> usize {
                let o = Self::view(other);
                unsafe { $intersection_size_sorted(self, o.as_ptr()) as usize }
            }

            /// `(self \ other, other \ self, self ∩ other)` for two **sorted**
            /// vectors, in a single pass
            /// ([`igraph_vector_difference_and_intersection_sorted`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_difference_and_intersection_sorted)).
            pub fn difference_and_intersection_sorted(
                &self,
                other: &[$elem],
            ) -> (Self, Self, Self) {
                let o = Self::view(other);
                let (mut d12, mut d21, mut inter) = (Self::new(), Self::new(), Self::new());
                crate::error::check(unsafe {
                    $dais(self, o.as_ptr(), &mut d12, &mut d21, &mut inter)
                })
                .expect("igraph failed to allocate a vector");
                (d12, d21, inter)
            }

            /// On a **sorted** vector: removes the elements smaller than
            /// `elem`, and the first half of the elements equal to it
            /// (`igraph_vector_filter_smaller`, used by igraph to compute
            /// medians and similar statistics).
            pub fn filter_smaller(&mut self, elem: $elem) {
                if !self.is_empty() {
                    crate::error::check(unsafe { $filter_smaller(self, elem) })
                        .expect("igraph_vector_filter_smaller cannot fail");
                }
            }
        }
    };
}

impl_vector_ordered!(
    igraph_vector_t,
    igraph_real_t,
    init_range = igraph_vector_init_range,
    range_nonempty = real_range_nonempty,
    sort = igraph_vector_sort,
    reverse_sort = igraph_vector_reverse_sort,
    sort_ind = igraph_vector_sort_ind,
    min = igraph_vector_min,
    max = igraph_vector_max,
    which_min = igraph_vector_which_min,
    which_max = igraph_vector_which_max,
    binsearch = igraph_vector_binsearch,
    contains_sorted = igraph_vector_contains_sorted,
    isininterval = igraph_vector_isininterval,
    any_smaller = igraph_vector_any_smaller,
    maxdifference = igraph_vector_maxdifference,
    lex_cmp = igraph_vector_lex_cmp,
    colex_cmp = igraph_vector_colex_cmp,
    all_l = igraph_vector_all_l,
    all_g = igraph_vector_all_g,
    all_le = igraph_vector_all_le,
    all_ge = igraph_vector_all_ge,
    difference_sorted = igraph_vector_difference_sorted,
    intersect_sorted = igraph_vector_intersect_sorted,
    intersection_size_sorted = igraph_vector_intersection_size_sorted,
    difference_and_intersection_sorted = igraph_vector_difference_and_intersection_sorted,
    filter_smaller = igraph_vector_filter_smaller
);
impl_vector_ordered!(
    igraph_vector_int_t,
    igraph_int_t,
    init_range = igraph_vector_int_init_range,
    range_nonempty = int_range_nonempty,
    sort = igraph_vector_int_sort,
    reverse_sort = igraph_vector_int_reverse_sort,
    sort_ind = igraph_vector_int_sort_ind,
    min = igraph_vector_int_min,
    max = igraph_vector_int_max,
    which_min = igraph_vector_int_which_min,
    which_max = igraph_vector_int_which_max,
    binsearch = igraph_vector_int_binsearch,
    contains_sorted = igraph_vector_int_contains_sorted,
    isininterval = igraph_vector_int_isininterval,
    any_smaller = igraph_vector_int_any_smaller,
    maxdifference = igraph_vector_int_maxdifference,
    lex_cmp = igraph_vector_int_lex_cmp,
    colex_cmp = igraph_vector_int_colex_cmp,
    all_l = igraph_vector_int_all_l,
    all_g = igraph_vector_int_all_g,
    all_le = igraph_vector_int_all_le,
    all_ge = igraph_vector_int_all_ge,
    difference_sorted = igraph_vector_int_difference_sorted,
    intersect_sorted = igraph_vector_int_intersect_sorted,
    intersection_size_sorted = igraph_vector_int_intersection_size_sorted,
    difference_and_intersection_sorted = igraph_vector_int_difference_and_intersection_sorted,
    filter_smaller = igraph_vector_int_filter_smaller
);
impl_vector_ordered!(
    igraph_vector_char_t,
    std::ffi::c_char,
    init_range = igraph_vector_char_init_range,
    range_nonempty = char_range_nonempty,
    sort = igraph_vector_char_sort,
    reverse_sort = igraph_vector_char_reverse_sort,
    sort_ind = igraph_vector_char_sort_ind,
    min = igraph_vector_char_min,
    max = igraph_vector_char_max,
    which_min = igraph_vector_char_which_min,
    which_max = igraph_vector_char_which_max,
    binsearch = igraph_vector_char_binsearch,
    contains_sorted = igraph_vector_char_contains_sorted,
    isininterval = igraph_vector_char_isininterval,
    any_smaller = igraph_vector_char_any_smaller,
    maxdifference = igraph_vector_char_maxdifference,
    lex_cmp = igraph_vector_char_lex_cmp,
    colex_cmp = igraph_vector_char_colex_cmp,
    all_l = igraph_vector_char_all_l,
    all_g = igraph_vector_char_all_g,
    all_le = igraph_vector_char_all_le,
    all_ge = igraph_vector_char_all_ge,
    difference_sorted = igraph_vector_char_difference_sorted,
    intersect_sorted = igraph_vector_char_intersect_sorted,
    intersection_size_sorted = igraph_vector_char_intersection_size_sorted,
    difference_and_intersection_sorted = igraph_vector_char_difference_and_intersection_sorted,
    filter_smaller = igraph_vector_char_filter_smaller
);

// ---------------------------------------------------------------------------
// Arithmetic. Floating point (real and complex) vectors use igraph's own
// functions; integer vectors are handled in Rust with *wrapping* arithmetic,
// because signed overflow in the C implementation would be undefined
// behaviour.
// ---------------------------------------------------------------------------

macro_rules! impl_vector_float_arith {
    (
        $ty:ident, $elem:ty, $name:literal,
        sum = $sum:ident, prod = $prod:ident, cumsum = $cumsum:ident,
        add_constant = $add_constant:ident, scale = $scale:ident,
        add = $add:ident, sub = $sub:ident, mul = $mul:ident, div = $div:ident,
        all_almost_e = $all_almost_e:ident, zapsmall = $zapsmall:ident
    ) => {
        impl $ty {
            #[doc = concat!("Sum of the elements (", $name, "; zero for an empty vector), see ")]
            /// [`igraph_vector_sum`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_sum).
            pub fn sum(&self) -> $elem {
                if self.is_empty() {
                    return Default::default();
                }
                unsafe { $sum(self) }
            }

            #[doc = concat!("Product of the elements (", $name, "; one for an empty vector), see ")]
            /// [`igraph_vector_prod`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_prod).
            pub fn prod(&self) -> $elem {
                if self.is_empty() {
                    return <$elem>::from(1.0);
                }
                unsafe { $prod(self) }
            }

            /// The cumulative sums `[x0, x0 + x1, x0 + x1 + x2, ...]`
            /// (`igraph_vector_cumsum`, undocumented in `igraph_vector.h`).
            pub fn cumsum(&self) -> Self {
                if self.is_empty() {
                    return Self::new();
                }
                let mut res = Self::new();
                crate::error::check(unsafe { $cumsum(&mut res, self) })
                    .expect("igraph failed to allocate a vector");
                res
            }

            /// Adds `value` to every element
            /// ([`igraph_vector_add_constant`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_add_constant)).
            pub fn add_constant(&mut self, value: $elem) {
                if !self.is_empty() {
                    unsafe { $add_constant(self, value) };
                }
            }

            /// Multiplies every element by `factor`
            /// ([`igraph_vector_scale`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_scale)).
            pub fn scale(&mut self, factor: $elem) {
                if !self.is_empty() {
                    unsafe { $scale(self, factor) };
                }
            }

            /// Element-wise `self[i] += other[i]`
            /// ([`igraph_vector_add`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_add)).
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the lengths differ.
            pub fn add(&mut self, other: &[$elem]) -> crate::error::Result<()> {
                self.binary_op(other, |a, b| unsafe { $add(a, b) })
            }

            /// Element-wise `self[i] -= other[i]`
            /// ([`igraph_vector_sub`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_sub)).
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the lengths differ.
            pub fn sub(&mut self, other: &[$elem]) -> crate::error::Result<()> {
                self.binary_op(other, |a, b| unsafe { $sub(a, b) })
            }

            /// Element-wise `self[i] *= other[i]`
            /// ([`igraph_vector_mul`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_mul)).
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the lengths differ.
            pub fn mul(&mut self, other: &[$elem]) -> crate::error::Result<()> {
                self.binary_op(other, |a, b| unsafe { $mul(a, b) })
            }

            /// Element-wise `self[i] /= other[i]` (IEEE semantics for
            /// division by zero)
            /// ([`igraph_vector_div`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_div)).
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the lengths differ.
            pub fn div(&mut self, other: &[$elem]) -> crate::error::Result<()> {
                self.binary_op(other, |a, b| unsafe { $div(a, b) })
            }

            fn binary_op(
                &mut self,
                other: &[$elem],
                f: impl FnOnce(*mut Self, *const Self) -> igraph_error_t,
            ) -> crate::error::Result<()> {
                if self.size() != other.len() {
                    return Err(crate::error::Error::invalid(format!(
                        "vectors of different lengths ({} and {})",
                        self.size(),
                        other.len()
                    )));
                }
                if other.is_empty() {
                    return Ok(());
                }
                let o = Self::view(other);
                crate::igraph_call!(f(self, o.as_ptr()))
            }

            /// Whether both vectors have the same length and all their
            /// elements are equal up to a relative tolerance `eps`
            /// ([`igraph_vector_all_almost_e`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_all_almost_e)).
            pub fn all_almost_e(&self, other: &[$elem], eps: f64) -> bool {
                if self.size() != other.len() {
                    return false;
                }
                if other.is_empty() {
                    return true;
                }
                let o = Self::view(other);
                unsafe { $all_almost_e(self, o.as_ptr(), eps) }
            }

            /// Replaces the elements smaller in magnitude than the *absolute*
            /// tolerance `tol` by exact zeros; `tol = 0` picks igraph's default,
            /// `f64::EPSILON^(2/3)` (about `1e-10`). For complex vectors the
            /// real and imaginary parts are processed separately
            /// ([`igraph_vector_zapsmall`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_zapsmall)).
            ///
            /// ```
            /// use igraph::prelude::*;
            /// let mut v = Vector::from([1.0, 1e-12, -3e-11, 0.5]);
            /// v.zapsmall(0.0).unwrap();
            /// assert_eq!(v, vec![1.0, 0.0, 0.0, 0.5]);
            /// ```
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if `tol < 0`.
            pub fn zapsmall(&mut self, tol: f64) -> crate::error::Result<()> {
                if self.is_empty() {
                    return if tol < 0.0 {
                        Err(crate::error::Error::invalid("negative tolerance"))
                    } else {
                        Ok(())
                    };
                }
                crate::igraph_call!($zapsmall(self, tol))
            }
        }
    };
}

impl From<f64> for igraph_complex_t {
    /// A complex number with zero imaginary part.
    fn from(re: f64) -> Self {
        Self::new(re, 0.0)
    }
}

impl Default for igraph_complex_t {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl_vector_float_arith!(
    igraph_vector_t,
    igraph_real_t,
    "reals",
    sum = igraph_vector_sum,
    prod = igraph_vector_prod,
    cumsum = igraph_vector_cumsum,
    add_constant = igraph_vector_add_constant,
    scale = igraph_vector_scale,
    add = igraph_vector_add,
    sub = igraph_vector_sub,
    mul = igraph_vector_mul,
    div = igraph_vector_div,
    all_almost_e = igraph_vector_all_almost_e,
    zapsmall = igraph_vector_zapsmall
);
impl_vector_float_arith!(
    igraph_vector_complex_t,
    igraph_complex_t,
    "complex numbers",
    sum = igraph_vector_complex_sum,
    prod = igraph_vector_complex_prod,
    cumsum = igraph_vector_complex_cumsum,
    add_constant = igraph_vector_complex_add_constant,
    scale = igraph_vector_complex_scale,
    add = igraph_vector_complex_add,
    sub = igraph_vector_complex_sub,
    mul = igraph_vector_complex_mul,
    div = igraph_vector_complex_div,
    all_almost_e = igraph_vector_complex_all_almost_e,
    zapsmall = igraph_vector_complex_zapsmall
);

impl igraph_vector_int_t {
    /// Sum of the elements, wrapping around on overflow (the Rust
    /// counterpart of `igraph_vector_int_sum`).
    pub fn sum(&self) -> igraph_int_t {
        self.iter().fold(0, |a, &b| a.wrapping_add(b))
    }

    /// Product of the elements (one for an empty vector), wrapping around
    /// on overflow (the Rust counterpart of `igraph_vector_int_prod`).
    pub fn prod(&self) -> igraph_int_t {
        self.iter().fold(1, |a, &b| a.wrapping_mul(b))
    }

    /// The cumulative sums, wrapping around on overflow (the Rust
    /// counterpart of `igraph_vector_int_cumsum`).
    pub fn cumsum(&self) -> Self {
        let mut acc: igraph_int_t = 0;
        self.iter()
            .map(|&x| {
                acc = acc.wrapping_add(x);
                acc
            })
            .collect()
    }

    /// Adds `value` to every element, wrapping around on overflow (the Rust
    /// counterpart of `igraph_vector_int_add_constant`).
    pub fn add_constant(&mut self, value: igraph_int_t) {
        self.iter_mut().for_each(|x| *x = x.wrapping_add(value));
    }

    /// Multiplies every element by `factor`, wrapping around on overflow
    /// (the Rust counterpart of `igraph_vector_int_scale`).
    pub fn scale(&mut self, factor: igraph_int_t) {
        self.iter_mut().for_each(|x| *x = x.wrapping_mul(factor));
    }

    fn zip_op(
        &mut self,
        other: &[igraph_int_t],
        f: impl Fn(igraph_int_t, igraph_int_t) -> igraph_int_t,
    ) -> crate::error::Result<()> {
        if self.size() != other.len() {
            return Err(crate::error::Error::invalid(format!(
                "vectors of different lengths ({} and {})",
                self.size(),
                other.len()
            )));
        }
        self.iter_mut().zip(other).for_each(|(a, &b)| *a = f(*a, b));
        Ok(())
    }

    /// Element-wise `self[i] += other[i]`, wrapping around on overflow (the
    /// Rust counterpart of `igraph_vector_int_add`).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the lengths differ.
    pub fn add(&mut self, other: &[igraph_int_t]) -> crate::error::Result<()> {
        self.zip_op(other, igraph_int_t::wrapping_add)
    }

    /// Element-wise `self[i] -= other[i]`, wrapping around on overflow (the
    /// Rust counterpart of `igraph_vector_int_sub`).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the lengths differ.
    pub fn sub(&mut self, other: &[igraph_int_t]) -> crate::error::Result<()> {
        self.zip_op(other, igraph_int_t::wrapping_sub)
    }

    /// Element-wise `self[i] *= other[i]`, wrapping around on overflow (the
    /// Rust counterpart of `igraph_vector_int_mul`).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the lengths differ.
    pub fn mul(&mut self, other: &[igraph_int_t]) -> crate::error::Result<()> {
        self.zip_op(other, igraph_int_t::wrapping_mul)
    }

    /// Element-wise integer division `self[i] /= other[i]`, rounding
    /// towards zero (the Rust counterpart of `igraph_vector_int_div`).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the lengths differ
    /// or some divisor is zero (nothing is modified then).
    pub fn div(&mut self, other: &[igraph_int_t]) -> crate::error::Result<()> {
        if other.contains(&0) {
            return Err(crate::error::Error::invalid("integer division by zero"));
        }
        self.zip_op(other, igraph_int_t::wrapping_div)
    }

    /// Replaces every element by its absolute value (wrapping for
    /// `i64::MIN`; the Rust counterpart of `igraph_vector_int_abs`).
    pub fn abs(&mut self) {
        self.iter_mut().for_each(|x| *x = x.wrapping_abs());
    }

    /// The order of the pairs `(self[i], second[i])`, sorted
    /// lexicographically with a two-pass radix sort in O(n + maxval); all
    /// values must be in `0..=maxval`, i.e. `maxval` is (an upper bound of)
    /// the largest value, and it also sets the size of the radix buckets
    /// (`igraph_vector_int_pair_order`, undocumented in `igraph_vector.h`).
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let first = VectorInt::from([1, 0, 1, 0]);
    /// let order = first.pair_order(&[1, 1, 0, 0], 2).unwrap();
    /// // (0,0) at 3, (0,1) at 1, (1,0) at 2, (1,1) at 0
    /// assert_eq!(order, vec![3, 1, 2, 0]);
    /// ```
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the lengths differ,
    /// `maxval` is negative or `i64::MAX`, or some value is outside `0..=maxval`.
    pub fn pair_order(
        &self,
        second: &[igraph_int_t],
        maxval: igraph_int_t,
    ) -> crate::error::Result<Self> {
        if self.size() != second.len() {
            return Err(crate::error::Error::invalid("vectors of different lengths"));
        }
        // igraph allocates `maxval + 1` buckets and indexes them without checks.
        if !(0..igraph_int_t::MAX).contains(&maxval) {
            return Err(crate::error::Error::invalid(format!(
                "invalid maximum value {maxval}"
            )));
        }
        if self.iter().chain(second).any(|&x| x < 0 || x > maxval) {
            return Err(crate::error::Error::invalid(format!(
                "values must be in 0..={maxval}"
            )));
        }
        if self.is_empty() {
            return Ok(Self::new());
        }
        let s = Self::view(second);
        let mut res = Self::new();
        crate::igraph_call!(igraph_vector_int_pair_order(
            self,
            s.as_ptr(),
            &mut res,
            maxval
        ))?;
        Ok(res)
    }
}

impl igraph_vector_t {
    /// Replaces every element by its absolute value
    /// (`igraph_vector_abs`, undocumented in `igraph_vector.h`).
    pub fn abs(&mut self) {
        if !self.is_empty() {
            crate::error::check(unsafe { igraph_vector_abs(self) })
                .expect("igraph_vector_abs cannot fail");
        }
    }

    // igraph converts with a plain C cast, which is undefined behaviour for
    // NaN, infinities and out-of-range values: reject them up front.
    fn check_integral_range(&self) -> crate::error::Result<()> {
        const LIMIT: f64 = 9.223_372_036_854_775e18; // 2^63
        match self.iter().find(|x| x.is_nan() || x.abs() >= LIMIT) {
            Some(x) => Err(crate::error::Error::new(
                crate::error::ErrorKind::Overflow,
                format!("{x} cannot be converted to an integer"),
            )),
            None => Ok(()),
        }
    }

    /// The elements rounded down to integers
    /// ([`igraph_vector_floor`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_floor)).
    ///
    /// # Errors
    /// [`ErrorKind::Overflow`](crate::error::ErrorKind::Overflow) if some
    /// element is NaN, infinite or does not fit in an `i64`.
    pub fn floor(&self) -> crate::error::Result<VectorInt> {
        self.check_integral_range()?;
        let mut res = VectorInt::new();
        crate::igraph_call!(igraph_vector_floor(self, &mut res))?;
        Ok(res)
    }

    /// The elements rounded to the nearest integers (halves away from zero)
    /// (`igraph_vector_round`, undocumented in `igraph_vector.h`).
    ///
    /// # Errors
    /// [`ErrorKind::Overflow`](crate::error::ErrorKind::Overflow) if some
    /// element is NaN, infinite or does not fit in an `i64`.
    pub fn round(&self) -> crate::error::Result<VectorInt> {
        self.check_integral_range()?;
        let mut res = VectorInt::new();
        crate::igraph_call!(igraph_vector_round(self, &mut res))?;
        Ok(res)
    }

    /// Which elements are NaN
    /// ([`igraph_vector_is_nan`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_is_nan)).
    pub fn is_nan(&self) -> VectorBool {
        let mut res = VectorBool::new();
        crate::error::check(unsafe { igraph_vector_is_nan(self, &mut res) })
            .expect("igraph failed to allocate a vector");
        res
    }

    /// Whether some element is NaN
    /// ([`igraph_vector_is_any_nan`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_is_any_nan)).
    pub fn is_any_nan(&self) -> bool {
        !self.is_empty() && unsafe { igraph_vector_is_any_nan(self) }
    }

    /// Whether all elements are finite, i.e. neither infinite nor NaN
    /// ([`igraph_vector_is_all_finite`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_is_all_finite)).
    pub fn is_all_finite(&self) -> bool {
        self.is_empty() || unsafe { igraph_vector_is_all_finite(self) }
    }
}

impl igraph_vector_char_t {
    /// Replaces every element by its absolute value (wrapping for the
    /// minimum value; the Rust counterpart of `igraph_vector_char_abs`).
    pub fn abs(&mut self) {
        self.iter_mut().for_each(|x| *x = x.wrapping_abs());
    }
}

impl igraph_vector_complex_t {
    /// Builds a complex vector from the real and imaginary parts
    /// ([`igraph_vector_complex_create`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_complex_create)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the lengths differ.
    pub fn from_parts(re: &[f64], im: &[f64]) -> crate::error::Result<Self> {
        if re.len() != im.len() {
            return Err(crate::error::Error::invalid(
                "real and imaginary parts of different lengths",
            ));
        }
        let (r, i) = (Vector::view(re), Vector::view(im));
        // `igraph_vector_complex_create` *initializes* its output: passing an
        // already initialized vector would leak its buffer.
        let mut res = MaybeUninit::<Self>::uninit();
        crate::igraph_call!(igraph_vector_complex_create(
            res.as_mut_ptr(),
            r.as_ptr(),
            i.as_ptr()
        ))?;
        Ok(unsafe { res.assume_init() })
    }

    /// Builds a complex vector from moduli `r` and arguments `theta`
    /// (in radians) ([`igraph_vector_complex_create_polar`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_complex_create_polar)).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if the lengths differ.
    pub fn from_polar(r: &[f64], theta: &[f64]) -> crate::error::Result<Self> {
        if r.len() != theta.len() {
            return Err(crate::error::Error::invalid(
                "moduli and arguments of different lengths",
            ));
        }
        let (a, b) = (Vector::view(r), Vector::view(theta));
        // The output is initialized by igraph (see `from_parts`).
        let mut res = MaybeUninit::<Self>::uninit();
        crate::igraph_call!(igraph_vector_complex_create_polar(
            res.as_mut_ptr(),
            a.as_ptr(),
            b.as_ptr()
        ))?;
        Ok(unsafe { res.assume_init() })
    }

    /// The real parts ([`igraph_vector_complex_real`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_complex_real)).
    pub fn real(&self) -> Vector {
        let mut res = Vector::new();
        crate::error::check(unsafe { igraph_vector_complex_real(self, &mut res) })
            .expect("igraph failed to allocate a vector");
        res
    }

    /// The imaginary parts ([`igraph_vector_complex_imag`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_complex_imag)).
    pub fn imag(&self) -> Vector {
        let mut res = Vector::new();
        crate::error::check(unsafe { igraph_vector_complex_imag(self, &mut res) })
            .expect("igraph failed to allocate a vector");
        res
    }

    /// `(real parts, imaginary parts)` in one call
    /// ([`igraph_vector_complex_realimag`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_complex_realimag)).
    pub fn realimag(&self) -> (Vector, Vector) {
        let (mut re, mut im) = (Vector::new(), Vector::new());
        crate::error::check(unsafe { igraph_vector_complex_realimag(self, &mut re, &mut im) })
            .expect("igraph failed to allocate a vector");
        (re, im)
    }
}

impl fmt::Display for igraph_vector_complex_t {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, z) in self.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{z}")?;
        }
        write!(f, "]")
    }
}

// ---------------------------------------------------------------------------
// Complex numbers (`igraph_complex.h`).
// ---------------------------------------------------------------------------

macro_rules! complex_unary {
    ($($(#[$meta:meta])* $name:ident => $c:ident),* $(,)?) => {
        impl igraph_complex_t {
            $(
                $(#[$meta])*
                pub fn $name(self) -> Self {
                    unsafe { $c(self) }
                }
            )*
        }
    };
}

complex_unary! {
    /// The complex conjugate (`igraph_complex_conj`, undocumented in `igraph_complex.h`).
    conj => igraph_complex_conj,
    /// The multiplicative inverse `1 / z` (`igraph_complex_inv`, undocumented in `igraph_complex.h`).
    inv => igraph_complex_inv,
    /// The principal square root (`igraph_complex_sqrt`, undocumented in `igraph_complex.h`).
    sqrt => igraph_complex_sqrt,
    /// The exponential `e^z` (`igraph_complex_exp`, undocumented in `igraph_complex.h`).
    exp => igraph_complex_exp,
    /// The principal natural logarithm (`igraph_complex_log`, undocumented in `igraph_complex.h`).
    ln => igraph_complex_log,
    /// The principal base-10 logarithm (`igraph_complex_log10`, undocumented in `igraph_complex.h`).
    log10 => igraph_complex_log10,
    /// The sine (`igraph_complex_sin`, undocumented in `igraph_complex.h`).
    sin => igraph_complex_sin,
    /// The cosine (`igraph_complex_cos`, undocumented in `igraph_complex.h`).
    cos => igraph_complex_cos,
    /// The tangent (`igraph_complex_tan`, undocumented in `igraph_complex.h`).
    tan => igraph_complex_tan,
    /// The secant `1 / cos z` (`igraph_complex_sec`, undocumented in `igraph_complex.h`).
    sec => igraph_complex_sec,
    /// The cosecant `1 / sin z` (`igraph_complex_csc`, undocumented in `igraph_complex.h`).
    csc => igraph_complex_csc,
    /// The cotangent `1 / tan z` (`igraph_complex_cot`, undocumented in `igraph_complex.h`).
    cot => igraph_complex_cot,
}

impl igraph_complex_t {
    /// The complex number `i`.
    pub const I: Self = Self::new(0.0, 1.0);

    /// Builds `r·e^{iθ}` from polar coordinates
    /// (`igraph_complex_polar`, undocumented in `igraph_complex.h`).
    ///
    /// ```
    /// use igraph::igraph_complex_t as Complex;
    /// let z = Complex::from_polar(2.0, std::f64::consts::FRAC_PI_2);
    /// assert!(z.almost_equals(Complex::new(0.0, 2.0), 1e-12));
    /// assert_eq!((z * z).re(), -4.0);
    /// ```
    pub fn from_polar(r: f64, theta: f64) -> Self {
        unsafe { igraph_complex_polar(r, theta) }
    }

    /// The (principal) square root of a real number, which may be imaginary
    /// (`igraph_complex_sqrt_real`, undocumented in `igraph_complex.h`).
    pub fn sqrt_real(x: f64) -> Self {
        unsafe { igraph_complex_sqrt_real(x) }
    }

    /// The modulus `|z|` (`igraph_complex_abs`, undocumented in `igraph_complex.h`).
    pub fn abs(self) -> f64 {
        unsafe { igraph_complex_abs(self) }
    }

    /// The natural logarithm of the modulus, `ln |z|`, computed accurately
    /// (`igraph_complex_logabs`, undocumented in `igraph_complex.h`).
    pub fn logabs(self) -> f64 {
        unsafe { igraph_complex_logabs(self) }
    }

    /// The argument (phase angle) in `(-π, π]`
    /// (`igraph_complex_arg`, undocumented in `igraph_complex.h`).
    pub fn arg(self) -> f64 {
        unsafe { igraph_complex_arg(self) }
    }

    /// Whether `self` and `other` are equal up to the relative tolerance `eps`
    /// ([`igraph_complex_almost_equals`](https://igraph.org/c/html/latest/igraph-Nongraph.html#igraph_complex_almost_equals)).
    pub fn almost_equals(self, other: Self, eps: f64) -> bool {
        unsafe { igraph_complex_almost_equals(self, other, eps) }
    }

    /// The complex power `self^exponent`
    /// (`igraph_complex_pow`, undocumented in `igraph_complex.h`).
    pub fn pow(self, exponent: Self) -> Self {
        unsafe { igraph_complex_pow(self, exponent) }
    }

    /// The real power `self^exponent`
    /// (`igraph_complex_pow_real`, undocumented in `igraph_complex.h`).
    pub fn powf(self, exponent: f64) -> Self {
        unsafe { igraph_complex_pow_real(self, exponent) }
    }

    /// The logarithm in base `base`
    /// (`igraph_complex_log_b`, undocumented in `igraph_complex.h`).
    pub fn log(self, base: Self) -> Self {
        unsafe { igraph_complex_log_b(self, base) }
    }

    /// Adds a real number (`igraph_complex_add_real`, undocumented in `igraph_complex.h`).
    pub fn add_real(self, x: f64) -> Self {
        unsafe { igraph_complex_add_real(self, x) }
    }

    /// Adds an imaginary number `i·y` (`igraph_complex_add_imag`, undocumented in `igraph_complex.h`).
    pub fn add_imag(self, y: f64) -> Self {
        unsafe { igraph_complex_add_imag(self, y) }
    }

    /// Subtracts a real number (`igraph_complex_sub_real`, undocumented in `igraph_complex.h`).
    pub fn sub_real(self, x: f64) -> Self {
        unsafe { igraph_complex_sub_real(self, x) }
    }

    /// Subtracts an imaginary number `i·y` (`igraph_complex_sub_imag`, undocumented in `igraph_complex.h`).
    pub fn sub_imag(self, y: f64) -> Self {
        unsafe { igraph_complex_sub_imag(self, y) }
    }

    /// Multiplies by a real number (`igraph_complex_mul_real`, undocumented in `igraph_complex.h`).
    pub fn mul_real(self, x: f64) -> Self {
        unsafe { igraph_complex_mul_real(self, x) }
    }

    /// Multiplies by an imaginary number `i·y` (`igraph_complex_mul_imag`, undocumented in `igraph_complex.h`).
    pub fn mul_imag(self, y: f64) -> Self {
        unsafe { igraph_complex_mul_imag(self, y) }
    }

    /// Divides by a real number (`igraph_complex_div_real`, undocumented in `igraph_complex.h`).
    pub fn div_real(self, x: f64) -> Self {
        unsafe { igraph_complex_div_real(self, x) }
    }

    /// Divides by an imaginary number `i·y` (`igraph_complex_div_imag`, undocumented in `igraph_complex.h`).
    pub fn div_imag(self, y: f64) -> Self {
        unsafe { igraph_complex_div_imag(self, y) }
    }
}

macro_rules! complex_binop {
    ($($trait:ident, $method:ident, $assign_trait:ident, $assign:ident => $c:ident;)*) => {$(
        impl std::ops::$trait for igraph_complex_t {
            type Output = Self;
            #[doc = concat!("Complex arithmetic with `", stringify!($c), "`.")]
            fn $method(self, rhs: Self) -> Self {
                unsafe { $c(self, rhs) }
            }
        }
        impl std::ops::$assign_trait for igraph_complex_t {
            fn $assign(&mut self, rhs: Self) {
                *self = unsafe { $c(*self, rhs) };
            }
        }
    )*};
}

complex_binop! {
    Add, add, AddAssign, add_assign => igraph_complex_add;
    Sub, sub, SubAssign, sub_assign => igraph_complex_sub;
    Mul, mul, MulAssign, mul_assign => igraph_complex_mul;
    Div, div, DivAssign, div_assign => igraph_complex_div;
}

impl std::ops::Neg for igraph_complex_t {
    type Output = Self;
    /// Negation (`igraph_complex_neg`, undocumented in `igraph_complex.h`).
    fn neg(self) -> Self {
        unsafe { igraph_complex_neg(self) }
    }
}

impl fmt::Display for igraph_complex_t {
    /// Formats like igraph does (`igraph_complex_snprintf`), e.g. `1+2i`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = [0 as std::ffi::c_char; 128];
        let n = unsafe { igraph_complex_snprintf(buf.as_mut_ptr(), buf.len(), *self) };
        if n < 0 {
            return write!(f, "{}{:+}i", self.re(), self.im());
        }
        let s = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) };
        f.write_str(&s.to_string_lossy())
    }
}

/// Formats a real number the way igraph does in its text output
/// (`igraph_real_snprintf`): `%g`-like, with `NaN`, `Inf` and `-Inf` for the
/// special values.
///
/// ```
/// use igraph::vector::format_real;
/// assert_eq!(format_real(0.5), "0.5");
/// assert_eq!(format_real(f64::INFINITY), "Inf");
/// assert_eq!(format_real(f64::NAN), "NaN");
/// ```
pub fn format_real(value: f64) -> String {
    format_real_with(value, false)
}

/// Formats a real number with 15 significant digits
/// (`igraph_real_snprintf_precise`), as used by igraph's writers.
///
/// ```
/// use igraph::vector::format_real_precise;
/// assert_eq!(format_real_precise(0.1), "0.1");
/// assert_eq!(format_real_precise(1.0 / 3.0), "0.333333333333333");
/// ```
pub fn format_real_precise(value: f64) -> String {
    format_real_with(value, true)
}

fn format_real_with(value: f64, precise: bool) -> String {
    let mut buf = [0 as std::ffi::c_char; 64];
    let n = unsafe {
        if precise {
            igraph_real_snprintf_precise(buf.as_mut_ptr(), buf.len(), value)
        } else {
            igraph_real_snprintf(buf.as_mut_ptr(), buf.len(), value)
        }
    };
    if n < 0 {
        return value.to_string();
    }
    unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

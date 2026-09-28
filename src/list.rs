//! Owned typed lists (`igraph_typed_list_pmt.h`): [`VectorIntList`],
//! [`VectorList`], [`MatrixList`], [`GraphList`] and [`BitsetList`].
//!
//! igraph returns collections of vectors (paths, cliques, components, ...)
//! and of graphs (decompositions, ...) through these list types. Each list
//! owns its items: it frees them on [`Drop`], can be indexed (`list[i]`),
//! iterated by reference, and converted into standard Rust collections.
//!
//! The safe wrappers of the crate usually convert them for you, e.g.
//! [`Graph::maximal_cliques`](crate::Graph::maximal_cliques) returns a
//! `Vec<Vec<i64>>` and [`Graph::decompose`](crate::Graph::decompose) a
//! `Vec<Graph>`; the list types matter when calling raw FFI functions or
//! when a wrapper takes a list as input.
//!
//! ```
//! use igraph::prelude::*;
//!
//! let list = VectorIntList::from_iter([vec![0, 1], vec![2, 3, 4]]);
//! assert_eq!(list.len(), 2);
//! assert_eq!(list[1].as_slice(), &[2, 3, 4]);
//! let nested: Vec<Vec<i64>> = list.to_vecs();
//! assert_eq!(nested, vec![vec![0, 1], vec![2, 3, 4]]);
//! ```
//!
//! Every list supports `push`, `pop`, `insert`, `remove`, `swap_remove`,
//! `replace`, `swap`, `reverse`, `permute`, `truncate`, `clear`,
//! `sort_by`/`sort_by_key`, `push_copy`, `reserve`/`capacity`, mutable
//! indexing and iteration. Lists of vectors can also be sorted and
//! deduplicated with igraph's own comparators:
//!
//! ```
//! use igraph::prelude::*;
//!
//! let mut cliques = VectorIntList::from_iter([vec![2, 3], vec![0, 1, 2], vec![2, 3], vec![0, 1]]);
//! cliques.sort(); // lexicographic, a prefix first
//! cliques.dedup();
//! assert_eq!(cliques.to_vecs(), vec![vec![0, 1], vec![0, 1, 2], vec![2, 3]]);
//! cliques.sort_by_key(|c| std::cmp::Reverse(c.len()));
//! assert_eq!(cliques[0], vec![0, 1, 2]);
//! ```

use crate::ffi::*;
use std::{fmt, mem::MaybeUninit, ops::Index};

/// Owned list of integer vectors (`igraph_vector_int_list_t`).
pub type VectorIntList = igraph_vector_int_list_t;
/// Owned list of real vectors (`igraph_vector_list_t`).
pub type VectorList = igraph_vector_list_t;
/// Owned list of real matrices (`igraph_matrix_list_t`).
pub type MatrixList = igraph_matrix_list_t;
/// Owned list of graphs (`igraph_graph_list_t`).
pub type GraphList = igraph_graph_list_t;
/// Owned list of bitsets (`igraph_bitset_list_t`), see [`crate::bitset`].
pub type BitsetList = igraph_bitset_list_t;

macro_rules! impl_list {
    ($ty:ident, $item:ident, init = $init:ident, init_copy = $init_copy:ident,
     destroy = $destroy:ident, push_back = $push:ident, pop_back = $pop:ident,
     remove = $remove:ident) => {
        impl $ty {
            /// Creates an empty list
            /// ([`igraph_*_list_init`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_list_init)).
            pub fn new() -> Self {
                crate::error::ensure_init();
                let mut raw = MaybeUninit::<Self>::uninit();
                crate::error::check(unsafe { $init(raw.as_mut_ptr(), 0) })
                    .expect("igraph failed to allocate a list");
                unsafe { raw.assume_init() }
            }

            /// Number of items.
            pub fn len(&self) -> usize {
                if self.stor_begin.is_null() {
                    0
                } else {
                    unsafe { self.end.offset_from(self.stor_begin) as usize }
                }
            }

            /// Whether the list is empty.
            pub fn is_empty(&self) -> bool {
                self.len() == 0
            }

            /// The items as a slice of the C item type.
            pub fn as_slice(&self) -> &[$item] {
                if self.stor_begin.is_null() {
                    &[]
                } else {
                    unsafe { std::slice::from_raw_parts(self.stor_begin, self.len()) }
                }
            }

            /// The items as a mutable slice of the C item type.
            pub fn as_mut_slice(&mut self) -> &mut [$item] {
                if self.stor_begin.is_null() {
                    &mut []
                } else {
                    let len = self.len();
                    unsafe { std::slice::from_raw_parts_mut(self.stor_begin, len) }
                }
            }

            /// The item at `index`, if any.
            pub fn get(&self, index: usize) -> Option<&$item> {
                self.as_slice().get(index)
            }

            /// Iterates over the items by reference.
            pub fn iter(&self) -> std::slice::Iter<'_, $item> {
                self.as_slice().iter()
            }

            /// Appends an item, transferring its ownership to the list
            /// ([`igraph_*_list_push_back`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_list_push_back)).
            pub fn push(&mut self, mut item: $item) {
                crate::error::check(unsafe { $push(self, &mut item) })
                    .expect("igraph failed to grow a list");
                // The list now owns the item's storage.
                std::mem::forget(item);
            }

            /// Removes and returns the last item, if any; the caller owns it
            /// ([`igraph_*_list_pop_back`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_list_pop_back)).
            pub fn pop(&mut self) -> Option<$item> {
                if self.is_empty() {
                    None
                } else {
                    Some(unsafe { $pop(self) })
                }
            }

            /// Removes the item at `index`, shifting the following ones, and
            /// returns it (ownership is transferred back to the caller)
            /// ([`igraph_*_list_remove`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_list_remove)).
            ///
            /// # Panics
            /// If `index >= len`.
            pub fn remove(&mut self, index: usize) -> $item {
                assert!(index < self.len(), "list index {index} out of bounds");
                let mut out = MaybeUninit::<$item>::uninit();
                crate::error::check(unsafe {
                    $remove(self, index as igraph_int_t, out.as_mut_ptr())
                })
                .expect("igraph failed to remove a list item");
                unsafe { out.assume_init() }
            }

            /// Converts into a [`Vec`] of owned items, without copying them.
            pub fn into_vec(mut self) -> Vec<$item> {
                let mut items = Vec::with_capacity(self.len());
                while let Some(item) = self.pop() {
                    items.push(item);
                }
                items.reverse();
                items
            }
        }

        impl Drop for $ty {
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
                crate::error::ensure_init();
                let mut raw = MaybeUninit::<Self>::uninit();
                crate::error::check(unsafe { $init_copy(raw.as_mut_ptr(), self) })
                    .expect("igraph failed to copy a list");
                unsafe { raw.assume_init() }
            }
        }

        impl Index<usize> for $ty {
            type Output = $item;
            fn index(&self, index: usize) -> &$item {
                &self.as_slice()[index]
            }
        }

        impl<'a> IntoIterator for &'a $ty {
            type Item = &'a $item;
            type IntoIter = std::slice::Iter<'a, $item>;
            fn into_iter(self) -> Self::IntoIter {
                self.iter()
            }
        }

        impl IntoIterator for $ty {
            type Item = $item;
            type IntoIter = std::vec::IntoIter<$item>;
            fn into_iter(self) -> Self::IntoIter {
                self.into_vec().into_iter()
            }
        }

        impl FromIterator<$item> for $ty {
            fn from_iter<I: IntoIterator<Item = $item>>(iter: I) -> Self {
                let mut list = Self::new();
                for item in iter {
                    list.push(item);
                }
                list
            }
        }

        impl From<Vec<$item>> for $ty {
            fn from(items: Vec<$item>) -> Self {
                items.into_iter().collect()
            }
        }

        unsafe impl Send for $ty {}
    };
}

impl_list!(
    igraph_vector_int_list_t,
    igraph_vector_int_t,
    init = igraph_vector_int_list_init,
    init_copy = igraph_vector_int_list_init_copy,
    destroy = igraph_vector_int_list_destroy,
    push_back = igraph_vector_int_list_push_back,
    pop_back = igraph_vector_int_list_pop_back,
    remove = igraph_vector_int_list_remove
);
impl_list!(
    igraph_vector_list_t,
    igraph_vector_t,
    init = igraph_vector_list_init,
    init_copy = igraph_vector_list_init_copy,
    destroy = igraph_vector_list_destroy,
    push_back = igraph_vector_list_push_back,
    pop_back = igraph_vector_list_pop_back,
    remove = igraph_vector_list_remove
);
impl_list!(
    igraph_matrix_list_t,
    igraph_matrix_t,
    init = igraph_matrix_list_init,
    init_copy = igraph_matrix_list_init_copy,
    destroy = igraph_matrix_list_destroy,
    push_back = igraph_matrix_list_push_back,
    pop_back = igraph_matrix_list_pop_back,
    remove = igraph_matrix_list_remove
);
impl_list!(
    igraph_graph_list_t,
    igraph_t,
    init = igraph_graph_list_init,
    init_copy = igraph_graph_list_init_copy,
    destroy = igraph_graph_list_destroy,
    push_back = igraph_graph_list_push_back,
    pop_back = igraph_graph_list_pop_back,
    remove = igraph_graph_list_remove
);

impl_list!(
    igraph_bitset_list_t,
    igraph_bitset_t,
    init = igraph_bitset_list_init,
    init_copy = igraph_bitset_list_init_copy,
    destroy = igraph_bitset_list_destroy,
    push_back = igraph_bitset_list_push_back,
    pop_back = igraph_bitset_list_pop_back,
    remove = igraph_bitset_list_remove
);

unsafe impl Sync for igraph_bitset_list_t {}
unsafe impl Sync for igraph_vector_int_list_t {}
unsafe impl Sync for igraph_vector_list_t {}
unsafe impl Sync for igraph_matrix_list_t {}

macro_rules! impl_vec_conversions {
    ($ty:ident, $elem:ty, $vec:ident) => {
        impl $ty {
            /// Copies the items into nested [`Vec`]s.
            pub fn to_vecs(&self) -> Vec<Vec<$elem>> {
                self.iter().map(|v| v.to_vec()).collect()
            }
        }

        impl FromIterator<Vec<$elem>> for $ty {
            fn from_iter<I: IntoIterator<Item = Vec<$elem>>>(iter: I) -> Self {
                iter.into_iter().map($vec::from).collect()
            }
        }

        impl<'a> FromIterator<&'a [$elem]> for $ty {
            fn from_iter<I: IntoIterator<Item = &'a [$elem]>>(iter: I) -> Self {
                iter.into_iter().map($vec::from_slice).collect()
            }
        }

        impl From<&[Vec<$elem>]> for $ty {
            fn from(items: &[Vec<$elem>]) -> Self {
                items.iter().map(|v| $vec::from_slice(v)).collect()
            }
        }

        impl From<$ty> for Vec<Vec<$elem>> {
            fn from(list: $ty) -> Self {
                list.to_vecs()
            }
        }

        impl PartialEq for $ty {
            fn eq(&self, other: &Self) -> bool {
                self.as_slice() == other.as_slice()
            }
        }
    };
}

impl_vec_conversions!(igraph_vector_int_list_t, igraph_int_t, igraph_vector_int_t);
impl_vec_conversions!(igraph_vector_list_t, igraph_real_t, igraph_vector_t);

// ---------------------------------------------------------------------------
// More list operations (`igraph_typed_list_pmt.h`), for every list type.
// ---------------------------------------------------------------------------

macro_rules! impl_list_extra {
    (
        $ty:ident, $item:ident,
        insert = $insert:ident, replace = $replace:ident, remove_fast = $remove_fast:ident,
        clear = $clear:ident, reverse = $reverse:ident, permute = $permute:ident,
        capacity = $capacity:ident, reserve = $reserve:ident, swap_elements = $swap_elements:ident,
        push_back_copy = $push_back_copy:ident
    ) => {
        impl $ty {
            /// Inserts `item` at position `pos`, shifting the following items;
            /// the list takes ownership of it
            /// ([`igraph_*_list_insert`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_list_insert)).
            ///
            /// # Panics
            /// If `pos > len`.
            pub fn insert(&mut self, pos: usize, mut item: $item) {
                let len = self.len();
                assert!(
                    pos <= len,
                    "list insertion index {pos} out of bounds (len {len})"
                );
                crate::error::check(unsafe { $insert(self, pos as igraph_int_t, &mut item) })
                    .expect("igraph failed to grow a list");
                // The list now owns the item's storage.
                std::mem::forget(item);
            }

            /// Replaces the item at `pos` with `item`, returning the old one
            /// ([`igraph_*_list_replace`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_list_replace)).
            ///
            /// # Panics
            /// If `pos >= len`.
            pub fn replace(&mut self, pos: usize, mut item: $item) -> $item {
                let len = self.len();
                assert!(pos < len, "list index {pos} out of bounds (len {len})");
                // igraph swaps the two structs: `item` now holds the old one.
                unsafe { $replace(self, pos as igraph_int_t, &mut item) };
                item
            }

            /// Removes the item at `pos` in O(1), moving the last item into
            /// its place, and returns it
            /// ([`igraph_*_list_remove_fast`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_list_remove_fast)).
            ///
            /// # Panics
            /// If `pos >= len`.
            pub fn swap_remove(&mut self, pos: usize) -> $item {
                let len = self.len();
                assert!(pos < len, "list index {pos} out of bounds (len {len})");
                let mut out = MaybeUninit::<$item>::uninit();
                crate::error::check(unsafe {
                    $remove_fast(self, pos as igraph_int_t, out.as_mut_ptr())
                })
                .expect("igraph failed to remove a list item");
                unsafe { out.assume_init() }
            }

            /// Destroys all the items, keeping the storage
            /// ([`igraph_*_list_clear`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_list_clear)).
            pub fn clear(&mut self) {
                if !self.stor_begin.is_null() {
                    unsafe { $clear(self) };
                }
            }

            /// Keeps the first `len` items, destroying the others.
            pub fn truncate(&mut self, len: usize) {
                while self.len() > len {
                    drop(self.pop());
                }
            }

            /// Reverses the order of the items
            /// (`igraph_*_list_reverse`, undocumented in `igraph_vector_list.h`).
            pub fn reverse(&mut self) {
                if !self.is_empty() {
                    crate::error::check(unsafe { $reverse(self) })
                        .expect("igraph_*_list_reverse cannot fail");
                }
            }

            /// Swaps the items at positions `i` and `j`
            /// ([`igraph_*_list_swap_elements`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_list_swap_elements)).
            ///
            /// # Panics
            /// If an index is out of bounds.
            pub fn swap(&mut self, i: usize, j: usize) {
                let len = self.len();
                assert!(
                    i < len && j < len,
                    "list indices ({i}, {j}) out of bounds (len {len})"
                );
                unsafe { $swap_elements(self, i as igraph_int_t, j as igraph_int_t) };
            }

            /// Reorders the items so that the item at `index[i]` moves to
            /// position `i`
            /// ([`igraph_*_list_permute`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_list_permute)).
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue)
            /// unless `index` is a permutation of `0..len` (igraph 1.0.0 and
            /// 1.0.1 do not check it, and would leak or double free items with
            /// an invalid one).
            pub fn permute(&mut self, index: &[igraph_int_t]) -> crate::error::Result<()> {
                let len = self.len();
                if index.len() != len {
                    return Err(crate::error::Error::invalid(format!(
                        "permutation of length {} for a list of length {len}",
                        index.len()
                    )));
                }
                let mut seen = vec![false; len];
                for &i in index {
                    if i < 0 || i as usize >= len || std::mem::replace(&mut seen[i as usize], true)
                    {
                        return Err(crate::error::Error::invalid(format!(
                            "invalid or repeated position {i} in a permutation of length {len}"
                        )));
                    }
                }
                if len == 0 {
                    return Ok(());
                }
                let idx = crate::vector::VectorInt::view(index);
                crate::igraph_call!($permute(self, idx.as_ptr()))
            }

            /// Sorts the items with a comparator, like [`slice::sort_by`]
            /// (stable; the items are moved, never copied).
            pub fn sort_by(&mut self, compare: impl FnMut(&$item, &$item) -> std::cmp::Ordering) {
                self.as_mut_slice().sort_by(compare);
            }

            /// Sorts the items by a key, like [`slice::sort_by_key`].
            pub fn sort_by_key<K: Ord>(&mut self, key: impl FnMut(&$item) -> K) {
                self.as_mut_slice().sort_by_key(key);
            }

            /// Number of items the list can hold without reallocating
            /// ([`igraph_*_list_capacity`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_list_capacity)).
            pub fn capacity(&self) -> usize {
                if self.stor_begin.is_null() {
                    0
                } else {
                    unsafe { $capacity(self) as usize }
                }
            }

            /// Reserves storage for at least `capacity` items in total
            /// ([`igraph_*_list_reserve`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_list_reserve)).
            pub fn reserve(&mut self, capacity: usize) {
                crate::error::check(unsafe { $reserve(self, crate::error::int_size(capacity)) })
                    .expect("igraph failed to reserve list storage");
            }

            /// Appends a deep copy of `item`
            /// ([`igraph_*_list_push_back_copy`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_list_push_back_copy)).
            pub fn push_copy(&mut self, item: &$item) {
                crate::error::check(unsafe { $push_back_copy(self, item) })
                    .expect("igraph failed to grow a list");
            }

            /// The item at `index`, mutably, if any.
            pub fn get_mut(&mut self, index: usize) -> Option<&mut $item> {
                self.as_mut_slice().get_mut(index)
            }

            /// The first item, if any.
            pub fn first(&self) -> Option<&$item> {
                self.as_slice().first()
            }

            /// The last item, if any.
            pub fn last(&self) -> Option<&$item> {
                self.as_slice().last()
            }

            /// Iterates over the items mutably.
            pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, $item> {
                self.as_mut_slice().iter_mut()
            }
        }

        impl std::ops::IndexMut<usize> for $ty {
            fn index_mut(&mut self, index: usize) -> &mut $item {
                &mut self.as_mut_slice()[index]
            }
        }

        impl<'a> IntoIterator for &'a mut $ty {
            type Item = &'a mut $item;
            type IntoIter = std::slice::IterMut<'a, $item>;
            fn into_iter(self) -> Self::IntoIter {
                self.iter_mut()
            }
        }

        impl Extend<$item> for $ty {
            fn extend<I: IntoIterator<Item = $item>>(&mut self, iter: I) {
                for item in iter {
                    self.push(item);
                }
            }
        }
    };
}

impl_list_extra!(
    igraph_vector_int_list_t,
    igraph_vector_int_t,
    insert = igraph_vector_int_list_insert,
    replace = igraph_vector_int_list_replace,
    remove_fast = igraph_vector_int_list_remove_fast,
    clear = igraph_vector_int_list_clear,
    reverse = igraph_vector_int_list_reverse,
    permute = igraph_vector_int_list_permute,
    capacity = igraph_vector_int_list_capacity,
    reserve = igraph_vector_int_list_reserve,
    swap_elements = igraph_vector_int_list_swap_elements,
    push_back_copy = igraph_vector_int_list_push_back_copy
);
impl_list_extra!(
    igraph_vector_list_t,
    igraph_vector_t,
    insert = igraph_vector_list_insert,
    replace = igraph_vector_list_replace,
    remove_fast = igraph_vector_list_remove_fast,
    clear = igraph_vector_list_clear,
    reverse = igraph_vector_list_reverse,
    permute = igraph_vector_list_permute,
    capacity = igraph_vector_list_capacity,
    reserve = igraph_vector_list_reserve,
    swap_elements = igraph_vector_list_swap_elements,
    push_back_copy = igraph_vector_list_push_back_copy
);
impl_list_extra!(
    igraph_matrix_list_t,
    igraph_matrix_t,
    insert = igraph_matrix_list_insert,
    replace = igraph_matrix_list_replace,
    remove_fast = igraph_matrix_list_remove_fast,
    clear = igraph_matrix_list_clear,
    reverse = igraph_matrix_list_reverse,
    permute = igraph_matrix_list_permute,
    capacity = igraph_matrix_list_capacity,
    reserve = igraph_matrix_list_reserve,
    swap_elements = igraph_matrix_list_swap_elements,
    push_back_copy = igraph_matrix_list_push_back_copy
);
impl_list_extra!(
    igraph_bitset_list_t,
    igraph_bitset_t,
    insert = igraph_bitset_list_insert,
    replace = igraph_bitset_list_replace,
    remove_fast = igraph_bitset_list_remove_fast,
    clear = igraph_bitset_list_clear,
    reverse = igraph_bitset_list_reverse,
    permute = igraph_bitset_list_permute,
    capacity = igraph_bitset_list_capacity,
    reserve = igraph_bitset_list_reserve,
    swap_elements = igraph_bitset_list_swap_elements,
    push_back_copy = igraph_bitset_list_push_back_copy
);
impl_list_extra!(
    igraph_graph_list_t,
    igraph_t,
    insert = igraph_graph_list_insert,
    replace = igraph_graph_list_replace,
    remove_fast = igraph_graph_list_remove_fast,
    clear = igraph_graph_list_clear,
    reverse = igraph_graph_list_reverse,
    permute = igraph_graph_list_permute,
    capacity = igraph_graph_list_capacity,
    reserve = igraph_graph_list_reserve,
    swap_elements = igraph_graph_list_swap_elements,
    push_back_copy = igraph_graph_list_push_back_copy
);

// ---------------------------------------------------------------------------
// Ordering of vector lists, with igraph's own comparators.
// ---------------------------------------------------------------------------

macro_rules! impl_vec_list_order {
    (
        $ty:ident, $item:ident,
        sort = $sort:ident, sort_ind = $sort_ind:ident, dedup = $dedup:ident,
        lex_cmp = $lex_cmp:ident, colex_cmp = $colex_cmp:ident, all_e = $all_e:ident
    ) => {
        impl $ty {
            /// Sorts the vectors lexicographically (a proper prefix comes
            /// first)
            /// ([`igraph_*_list_sort`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_list_sort)
            /// with [`igraph_vector_lex_cmp`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_lex_cmp)).
            pub fn sort(&mut self) {
                if !self.is_empty() {
                    unsafe { $sort(self, Some($lex_cmp)) };
                }
            }

            /// Sorts the vectors colexicographically, i.e. comparing them
            /// from their last elements
            /// (`igraph_*_list_sort` with `igraph_vector_colex_cmp`).
            pub fn sort_colex(&mut self) {
                if !self.is_empty() {
                    unsafe { $sort(self, Some($colex_cmp)) };
                }
            }

            /// The permutation that sorts the list lexicographically, without
            /// modifying it; pass it to `permute` to sort
            /// ([`igraph_*_list_sort_ind`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_vector_list_sort_ind)).
            pub fn sort_ind(&self) -> Vec<igraph_int_t> {
                if self.is_empty() {
                    return Vec::new();
                }
                let mut ind = crate::vector::VectorInt::new();
                // `sort_ind` takes a mutable pointer but does not modify the
                // list; a copy keeps `&self` honest at the price of O(n) copies.
                let mut copy = self.clone();
                crate::error::check(unsafe { $sort_ind(&mut copy, &mut ind, Some($lex_cmp)) })
                    .expect("igraph failed to sort a list");
                ind.into()
            }

            /// Removes consecutive equal vectors, keeping the first of each
            /// run; on a sorted list it removes all duplicates, like
            /// [`Vec::dedup`]
            /// (`igraph_*_list_remove_consecutive_duplicates`, undocumented in `igraph_vector_list.h`).
            pub fn dedup(&mut self) {
                if !self.is_empty() {
                    unsafe { $dedup(self, Some($all_e)) };
                }
            }
        }
    };
}

impl_vec_list_order!(
    igraph_vector_int_list_t,
    igraph_vector_int_t,
    sort = igraph_vector_int_list_sort,
    sort_ind = igraph_vector_int_list_sort_ind,
    dedup = igraph_vector_int_list_remove_consecutive_duplicates,
    lex_cmp = igraph_vector_int_lex_cmp,
    colex_cmp = igraph_vector_int_colex_cmp,
    all_e = igraph_vector_int_all_e
);
impl_vec_list_order!(
    igraph_vector_list_t,
    igraph_vector_t,
    sort = igraph_vector_list_sort,
    sort_ind = igraph_vector_list_sort_ind,
    dedup = igraph_vector_list_remove_consecutive_duplicates,
    lex_cmp = igraph_vector_lex_cmp,
    colex_cmp = igraph_vector_colex_cmp,
    all_e = igraph_vector_all_e
);

impl igraph_graph_list_t {
    /// Sets whether the empty graphs created by igraph when *growing* this
    /// list (e.g. by `igraph_graph_list_resize`) are directed
    /// (`igraph_graph_list_set_directed`, undocumented in `igraph_graph_list.h`).
    pub fn set_directed(&mut self, directed: bool) {
        // igraph 1.0.0 and 1.0.1 declare, but do not export,
        // `igraph_graph_list_set_directed`, which only sets this field.
        self.directed = directed;
    }
}

// Graphs are not `Sync`, so neither is a list of graphs; the other lists are.
impl fmt::Display for igraph_vector_int_list_t {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.iter().map(|v| v.as_slice()))
            .finish()
    }
}

impl fmt::Display for igraph_vector_list_t {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.iter().map(|v| v.as_slice()))
            .finish()
    }
}

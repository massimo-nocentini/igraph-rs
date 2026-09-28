//! Fixed-size sets of bits (`igraph_bitset.h`, `igraph_bitset_list.h`).
//!
//! [`Bitset`] is igraph's `igraph_bitset_t` made Rusty: a compact, owned
//! sequence of bits (one machine word stores 64 of them) that frees its
//! storage on [`Drop`], can be cloned, compared, iterated and combined with
//! the usual bit operators `&`, `|`, `^` and `!`. igraph uses bitsets e.g. to
//! mark visited vertices; [`BitsetList`] is the owned list of bitsets
//! (`igraph_bitset_list_t`) returned by some algorithms, such as
//! `igraph_reachability` (wrapped by
//! [`Graph::reachability`](crate::Graph::reachability), which unpacks the
//! bitsets into booleans).
//!
//! Bits are indexed from `0` (the *least* significant bit); [`Display`](std::fmt::Display)
//! prints them like igraph does, most significant first, as a binary number.
//!
//! ```
//! use igraph::bitset::Bitset;
//!
//! let mut visited = Bitset::new(10);
//! visited.set(2, true);
//! visited.set(7, true);
//! assert_eq!(visited.count_ones(), 2);
//! assert_eq!(visited.iter_ones().collect::<Vec<_>>(), vec![2, 7]);
//! assert_eq!(visited.to_string(), "0010000100");
//!
//! let evens: Bitset = (0..10).map(|i| i % 2 == 0).collect();
//! assert_eq!((&visited & &evens).iter_ones().collect::<Vec<_>>(), vec![2]);
//! assert_eq!((!&evens).count_ones(), 5);
//! ```
//!
//! | Rust                                    | C                                       |
//! |-----------------------------------------|-----------------------------------------|
//! | [`Bitset::new`], [`Bitset::clone`]      | `igraph_bitset_init`, `igraph_bitset_init_copy` |
//! | [`len`](Bitset::len), [`capacity`](Bitset::capacity) | `igraph_bitset_size`, `igraph_bitset_capacity` |
//! | [`resize`](Bitset::resize), [`reserve`](Bitset::reserve) | `igraph_bitset_resize`, `igraph_bitset_reserve` |
//! | [`get`](Bitset::get), [`set`](Bitset::set), [`toggle`](Bitset::toggle) | `IGRAPH_BIT_TEST`, `IGRAPH_BIT_SET`, `IGRAPH_BIT_CLEAR` |
//! | [`count_ones`](Bitset::count_ones)      | `igraph_bitset_popcount`                |
//! | [`leading_zeros`](Bitset::leading_zeros), [`leading_ones`](Bitset::leading_ones) | `igraph_bitset_countl_zero`, `igraph_bitset_countl_one` |
//! | [`trailing_zeros`](Bitset::trailing_zeros), [`trailing_ones`](Bitset::trailing_ones) | `igraph_bitset_countr_zero`, `igraph_bitset_countr_one` |
//! | [`all`](Bitset::all), [`any`](Bitset::any), [`none`](Bitset::none), [`not_all`](Bitset::not_all) | `igraph_bitset_is_all_one`, `igraph_bitset_is_any_one`, `igraph_bitset_is_all_zero`, `igraph_bitset_is_any_zero` |
//! | `&`, `\|`, `^`, `!` (and `&=`, ...)       | `igraph_bitset_and`, `igraph_bitset_or`, `igraph_bitset_xor`, `igraph_bitset_not` |
//! | [`fill`](Bitset::fill), [`clear`](Bitset::clear) | `igraph_bitset_fill`, `igraph_bitset_null` |
//! | [`Display`](std::fmt::Display)                             | `igraph_bitset_print`                   |

use crate::ffi::*;
use std::{fmt, mem::MaybeUninit, ops};

pub use crate::list::BitsetList;

/// An owned, fixed-size set of bits (`igraph_bitset_t`), see the [module docs](self).
pub type Bitset = igraph_bitset_t;

const WORD_BITS: usize = igraph_uint_t::BITS as usize;

impl igraph_bitset_t {
    /// Creates a bitset of `len` bits, all zero
    /// ([`igraph_bitset_init`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_init)).
    pub fn new(len: usize) -> Self {
        crate::error::ensure_init();
        let mut raw = MaybeUninit::<Self>::uninit();
        crate::error::check(unsafe {
            igraph_bitset_init(raw.as_mut_ptr(), crate::error::int_size(len))
        })
        .expect("igraph failed to allocate a bitset");
        unsafe { raw.assume_init() }
    }

    /// Creates a bitset from a slice of booleans (`bits[i]` is bit `i`).
    pub fn from_bools(bits: &[bool]) -> Self {
        let mut b = Self::new(bits.len());
        for (i, &x) in bits.iter().enumerate() {
            if x {
                b.set(i, true);
            }
        }
        b
    }

    /// Creates a bitset of `len` bits where exactly the listed positions are
    /// set.
    ///
    /// # Panics
    /// If a position is `>= len`.
    pub fn from_ones(len: usize, ones: impl IntoIterator<Item = usize>) -> Self {
        let mut b = Self::new(len);
        for i in ones {
            b.set(i, true);
        }
        b
    }

    /// Number of bits
    /// ([`igraph_bitset_size`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_size)).
    pub fn len(&self) -> usize {
        self.size as usize
    }

    /// Whether the bitset has no bits at all (not whether all bits are zero:
    /// see [`none`](Self::none)).
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Number of bits that fit in the allocated storage
    /// ([`igraph_bitset_capacity`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_capacity)).
    pub fn capacity(&self) -> usize {
        unsafe { igraph_bitset_capacity(self) as usize }
    }

    /// Reserves storage for at least `capacity` bits in total
    /// ([`igraph_bitset_reserve`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_reserve)).
    pub fn reserve(&mut self, capacity: usize) {
        crate::error::check(unsafe {
            igraph_bitset_reserve(self, crate::error::int_size(capacity))
        })
        .expect("igraph failed to reserve bitset storage");
    }

    /// Changes the number of bits; new bits are zero
    /// ([`igraph_bitset_resize`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_resize)).
    pub fn resize(&mut self, len: usize) {
        crate::error::check(unsafe { igraph_bitset_resize(self, crate::error::int_size(len)) })
            .expect("igraph failed to resize a bitset");
    }

    fn words(&self) -> &[igraph_uint_t] {
        let n = self.len().div_ceil(WORD_BITS);
        if n == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.stor_begin, n) }
    }

    fn words_mut(&mut self) -> &mut [igraph_uint_t] {
        let n = self.len().div_ceil(WORD_BITS);
        if n == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.stor_begin, n) }
    }

    fn check_index(&self, i: usize) {
        assert!(
            i < self.len(),
            "bit index {i} out of bounds (len {})",
            self.len()
        );
    }

    /// The value of bit `i` (`IGRAPH_BIT_TEST`).
    ///
    /// # Panics
    /// If `i >= len`.
    pub fn get(&self, i: usize) -> bool {
        self.check_index(i);
        self.words()[i / WORD_BITS] & (1 << (i % WORD_BITS)) != 0
    }

    /// Sets bit `i` to `value` (`IGRAPH_BIT_SET` / `IGRAPH_BIT_CLEAR`).
    ///
    /// # Panics
    /// If `i >= len`.
    pub fn set(&mut self, i: usize, value: bool) {
        self.check_index(i);
        let w = &mut self.words_mut()[i / WORD_BITS];
        if value {
            *w |= 1 << (i % WORD_BITS);
        } else {
            *w &= !(1 << (i % WORD_BITS));
        }
    }

    /// Flips bit `i` and returns its new value.
    ///
    /// # Panics
    /// If `i >= len`.
    pub fn toggle(&mut self, i: usize) -> bool {
        let v = !self.get(i);
        self.set(i, v);
        v
    }

    /// Sets bit `i` and returns whether it was *not* set before (like
    /// [`HashSet::insert`](std::collections::HashSet::insert)).
    ///
    /// # Panics
    /// If `i >= len`.
    pub fn insert(&mut self, i: usize) -> bool {
        let was = self.get(i);
        self.set(i, true);
        !was
    }

    /// Appends a bit at the end, growing the bitset by one.
    pub fn push(&mut self, value: bool) {
        let n = self.len();
        self.resize(n + 1);
        self.set(n, value);
    }

    /// Number of set bits (the population count)
    /// ([`igraph_bitset_popcount`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_popcount)).
    pub fn count_ones(&self) -> usize {
        unsafe { igraph_bitset_popcount(self) as usize }
    }

    /// Number of zero bits.
    pub fn count_zeros(&self) -> usize {
        self.len() - self.count_ones()
    }

    /// Number of zeros before the first one, starting from the *most*
    /// significant bit (`len` if all zero)
    /// ([`igraph_bitset_countl_zero`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_countl_zero)).
    pub fn leading_zeros(&self) -> usize {
        unsafe { igraph_bitset_countl_zero(self) as usize }
    }

    /// Number of ones before the first zero, starting from the *most*
    /// significant bit
    /// ([`igraph_bitset_countl_one`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_countl_one)).
    pub fn leading_ones(&self) -> usize {
        unsafe { igraph_bitset_countl_one(self) as usize }
    }

    /// Number of zeros before the first one, starting from bit 0 (`len` if
    /// all zero); i.e. the index of the first set bit
    /// ([`igraph_bitset_countr_zero`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_countr_zero)).
    pub fn trailing_zeros(&self) -> usize {
        unsafe { igraph_bitset_countr_zero(self) as usize }
    }

    /// Number of ones before the first zero, starting from bit 0
    /// ([`igraph_bitset_countr_one`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_countr_one)).
    pub fn trailing_ones(&self) -> usize {
        unsafe { igraph_bitset_countr_one(self) as usize }
    }

    /// Whether all bits are one (true for an empty bitset)
    /// ([`igraph_bitset_is_all_one`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_is_all_one)).
    pub fn all(&self) -> bool {
        unsafe { igraph_bitset_is_all_one(self) }
    }

    /// Whether some bit is one
    /// ([`igraph_bitset_is_any_one`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_is_any_one)).
    pub fn any(&self) -> bool {
        unsafe { igraph_bitset_is_any_one(self) }
    }

    /// Whether all bits are zero (true for an empty bitset)
    /// ([`igraph_bitset_is_all_zero`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_is_all_zero)).
    pub fn none(&self) -> bool {
        unsafe { igraph_bitset_is_all_zero(self) }
    }

    /// Whether some bit is zero
    /// ([`igraph_bitset_is_any_zero`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_is_any_zero)).
    pub fn not_all(&self) -> bool {
        unsafe { igraph_bitset_is_any_zero(self) }
    }

    /// Sets every bit to `value`
    /// ([`igraph_bitset_fill`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_fill)).
    pub fn fill(&mut self, value: bool) {
        unsafe { igraph_bitset_fill(self, value) }
    }

    /// Sets every bit to zero, keeping the length
    /// ([`igraph_bitset_null`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_null)).
    pub fn clear(&mut self) {
        unsafe { igraph_bitset_null(self) }
    }

    /// Iterates over all the bits, from bit 0.
    pub fn iter(&self) -> impl Iterator<Item = bool> + '_ {
        (0..self.len()).map(move |i| self.get(i))
    }

    /// Iterates over the positions of the set bits, in increasing order,
    /// skipping whole zero words.
    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        let len = self.len();
        self.words().iter().enumerate().flat_map(move |(k, &w)| {
            let mut w = w;
            std::iter::from_fn(move || {
                if w == 0 {
                    return None;
                }
                let b = w.trailing_zeros() as usize;
                w &= w - 1;
                Some(k * WORD_BITS + b)
            })
            .take_while(move |&i| i < len)
        })
    }

    /// The bits as a `Vec<bool>`.
    pub fn to_vec(&self) -> Vec<bool> {
        self.iter().collect()
    }

    fn binary(
        &self,
        other: &Self,
        op: unsafe extern "C" fn(*mut Self, *const Self, *const Self),
    ) -> Self {
        assert_eq!(self.len(), other.len(), "bitsets of different lengths");
        let mut res = Self::new(self.len());
        unsafe { op(&mut res, self, other) };
        res
    }

    /// Bitwise AND ([`igraph_bitset_and`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_and)).
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn and(&self, other: &Self) -> Self {
        self.binary(other, igraph_bitset_and)
    }

    /// Bitwise OR ([`igraph_bitset_or`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_or)).
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn or(&self, other: &Self) -> Self {
        self.binary(other, igraph_bitset_or)
    }

    /// Bitwise XOR ([`igraph_bitset_xor`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_xor)).
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn xor(&self, other: &Self) -> Self {
        self.binary(other, igraph_bitset_xor)
    }

    /// Bitwise complement ([`igraph_bitset_not`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_not)).
    pub fn complement(&self) -> Self {
        let mut res = Self::new(self.len());
        unsafe { igraph_bitset_not(&mut res, self) };
        res
    }

    /// Replaces the content with a copy of `other`, reusing the storage
    /// when possible ([`igraph_bitset_update`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_bitset_update)).
    pub fn update(&mut self, other: &Self) {
        crate::error::check(unsafe { igraph_bitset_update(self, other) })
            .expect("igraph failed to copy a bitset");
    }
}

impl Drop for igraph_bitset_t {
    /// Frees the storage with `igraph_bitset_destroy`.
    fn drop(&mut self) {
        if !self.stor_begin.is_null() {
            unsafe { igraph_bitset_destroy(self) };
            self.stor_begin = std::ptr::null_mut();
        }
    }
}

impl Clone for igraph_bitset_t {
    /// Deep copy with `igraph_bitset_init_copy`.
    fn clone(&self) -> Self {
        crate::error::ensure_init();
        let mut raw = MaybeUninit::<Self>::uninit();
        crate::error::check(unsafe { igraph_bitset_init_copy(raw.as_mut_ptr(), self) })
            .expect("igraph failed to copy a bitset");
        unsafe { raw.assume_init() }
    }
}

impl Default for igraph_bitset_t {
    /// An empty bitset.
    fn default() -> Self {
        Self::new(0)
    }
}

impl PartialEq for igraph_bitset_t {
    /// Two bitsets are equal when they have the same length and bits (the
    /// unused padding bits of the last word are ignored).
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl Eq for igraph_bitset_t {}

impl fmt::Display for igraph_bitset_t {
    /// Writes the bits as `0`/`1` characters, most significant (highest
    /// index) first, like `igraph_bitset_print`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s: String = (0..self.len())
            .rev()
            .map(|i| if self.get(i) { '1' } else { '0' })
            .collect();
        f.pad(&s)
    }
}

impl FromIterator<bool> for igraph_bitset_t {
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        let bits: Vec<bool> = iter.into_iter().collect();
        Self::from_bools(&bits)
    }
}

impl From<&[bool]> for igraph_bitset_t {
    fn from(bits: &[bool]) -> Self {
        Self::from_bools(bits)
    }
}

impl From<&igraph_bitset_t> for Vec<bool> {
    fn from(b: &igraph_bitset_t) -> Self {
        b.to_vec()
    }
}

macro_rules! bitset_op {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign:ident, $c:ident) => {
        impl ops::$trait for &igraph_bitset_t {
            type Output = igraph_bitset_t;
            /// # Panics
            /// If the lengths differ.
            fn $method(self, rhs: Self) -> igraph_bitset_t {
                self.binary(rhs, $c)
            }
        }
        impl ops::$assign_trait<&igraph_bitset_t> for igraph_bitset_t {
            /// # Panics
            /// If the lengths differ.
            fn $assign(&mut self, rhs: &igraph_bitset_t) {
                assert_eq!(self.len(), rhs.len(), "bitsets of different lengths");
                let me: *mut igraph_bitset_t = self;
                // igraph computes word by word, so aliasing dest and src is fine.
                unsafe { $c(me, me, rhs) };
            }
        }
    };
}

bitset_op!(
    BitAnd,
    bitand,
    BitAndAssign,
    bitand_assign,
    igraph_bitset_and
);
bitset_op!(BitOr, bitor, BitOrAssign, bitor_assign, igraph_bitset_or);
bitset_op!(
    BitXor,
    bitxor,
    BitXorAssign,
    bitxor_assign,
    igraph_bitset_xor
);

impl ops::Not for &igraph_bitset_t {
    type Output = igraph_bitset_t;
    fn not(self) -> igraph_bitset_t {
        self.complement()
    }
}

// The storage is uniquely owned and only mutated through `&mut`.
unsafe impl Send for igraph_bitset_t {}
unsafe impl Sync for igraph_bitset_t {}

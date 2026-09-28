//! Owned vectors of strings (`igraph_strvector_t`, `igraph_strvector.h`).
//!
//! igraph uses string vectors for vertex names, file formats and string
//! attributes. [`StrVector`] owns a copy of every string, is indexed with
//! `sv[i]` (a `&str`), and offers the usual editing operations (`push`,
//! `set`, `remove`, `pop`, `swap`, `select`, `append`, `extend_from`,
//! `resize`, `truncate`, `clear`, ...). Strings are stored as C strings: an
//! interior NUL byte truncates a string when it is read back.
//!
//! ```
//! use igraph::prelude::*;
//!
//! let mut names: StrVector = ["alice", "bob"].into_iter().collect();
//! names.push("carol");
//! assert_eq!(names.len(), 3);
//! assert_eq!(names.get(1), Some("bob"));
//! assert_eq!(names.to_vec(), vec!["alice", "bob", "carol"]);
//! ```
//!
//! # Not wrapped
//!
//! A few functions of `igraph_strvector.h` have no direct counterpart:
//! `igraph_strvector_set` and `igraph_strvector_push_back` take
//! NUL-terminated C strings, while a Rust `&str` has no terminator, so
//! [`set`](StrVector::set) and [`push`](StrVector::push) call the
//! length-taking `igraph_strvector_set_len` and
//! `igraph_strvector_push_back_len` instead; `igraph_strvector_swap`
//! (exchanging two whole vectors) is [`std::mem::swap`]; and
//! `igraph_strvector_print` / `igraph_strvector_fprint` are covered by the
//! [`Display`](std::fmt::Display) implementation, which prints the list as
//! `["alice", "bob"]` (igraph's one-string-per-line format is
//! `sv.iter().collect::<Vec<_>>().join("\n")`).

use crate::ffi::*;
use std::{ffi::CStr, fmt, mem::MaybeUninit};

/// Owned vector of strings (`igraph_strvector_t`).
pub type StrVector = igraph_strvector_t;

impl igraph_strvector_t {
    /// Creates an empty string vector
    /// ([`igraph_strvector_init`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_strvector_init)).
    pub fn new() -> Self {
        crate::error::ensure_init();
        let mut raw = MaybeUninit::<Self>::uninit();
        crate::error::check(unsafe { igraph_strvector_init(raw.as_mut_ptr(), 0) })
            .expect("igraph failed to allocate a string vector");
        unsafe { raw.assume_init() }
    }

    /// Number of strings
    /// ([`igraph_strvector_size`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_strvector_size)).
    pub fn len(&self) -> usize {
        unsafe { igraph_strvector_size(self) as usize }
    }

    /// Whether there are no strings.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The string at `index`, or `None` if `index` is out of bounds or the
    /// string is not valid UTF-8 (use [`get_cstr`](Self::get_cstr), or the
    /// lossy [`iter`](Self::iter) and [`to_vec`](Self::to_vec), for such
    /// strings) ([`igraph_strvector_get`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_strvector_get)).
    pub fn get(&self, index: usize) -> Option<&str> {
        self.get_cstr(index).and_then(|s| s.to_str().ok())
    }

    /// The raw C string at `index`, if any, borrowed from the vector
    /// ([`igraph_strvector_get`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_strvector_get)).
    pub fn get_cstr(&self, index: usize) -> Option<&CStr> {
        if index >= self.len() {
            return None;
        }
        let ptr = unsafe { igraph_strvector_get(self, index as igraph_int_t) };
        (!ptr.is_null()).then(|| unsafe { CStr::from_ptr(ptr) })
    }

    /// Appends a copy of `value` (interior NUL bytes truncate it when read
    /// back) ([`igraph_strvector_push_back_len`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_strvector_push_back_len)).
    pub fn push(&mut self, value: &str) {
        crate::error::check(unsafe {
            igraph_strvector_push_back_len(self, value.as_ptr().cast(), value.len())
        })
        .expect("igraph failed to grow a string vector");
    }

    /// Iterates over the strings (lossily converted to UTF-8).
    pub fn iter(&self) -> impl Iterator<Item = String> + '_ {
        (0..self.len()).map(move |i| {
            self.get_cstr(i)
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        })
    }

    /// Copies the strings into a `Vec<String>`.
    pub fn to_vec(&self) -> Vec<String> {
        self.iter().collect()
    }
}

impl Drop for igraph_strvector_t {
    fn drop(&mut self) {
        if !self.stor_begin.is_null() {
            unsafe { igraph_strvector_destroy(self) };
            self.stor_begin = std::ptr::null_mut();
        }
    }
}

impl Default for igraph_strvector_t {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for igraph_strvector_t {
    fn clone(&self) -> Self {
        crate::error::ensure_init();
        let mut raw = MaybeUninit::<Self>::uninit();
        crate::error::check(unsafe { igraph_strvector_init_copy(raw.as_mut_ptr(), self) })
            .expect("igraph failed to copy a string vector");
        unsafe { raw.assume_init() }
    }
}

impl PartialEq for igraph_strvector_t {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && (0..self.len()).all(|i| self.get_cstr(i) == other.get_cstr(i))
    }
}

impl fmt::Display for igraph_strvector_t {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<S: AsRef<str>> FromIterator<S> for igraph_strvector_t {
    fn from_iter<I: IntoIterator<Item = S>>(iter: I) -> Self {
        let mut sv = Self::new();
        for s in iter {
            sv.push(s.as_ref());
        }
        sv
    }
}

impl<S: AsRef<str>> From<&[S]> for igraph_strvector_t {
    fn from(items: &[S]) -> Self {
        items.iter().collect()
    }
}

impl From<&igraph_strvector_t> for Vec<String> {
    fn from(sv: &igraph_strvector_t) -> Self {
        sv.to_vec()
    }
}

unsafe impl Send for igraph_strvector_t {}
unsafe impl Sync for igraph_strvector_t {}

impl igraph_strvector_t {
    /// Creates a vector of `len` empty strings
    /// ([`igraph_strvector_init`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_strvector_init)).
    pub fn with_len(len: usize) -> Self {
        crate::error::ensure_init();
        let mut raw = MaybeUninit::<Self>::uninit();
        crate::error::check(unsafe {
            igraph_strvector_init(raw.as_mut_ptr(), crate::error::int_size(len))
        })
        .expect("igraph failed to allocate a string vector");
        unsafe { raw.assume_init() }
    }

    /// Replaces the string at `index` with a copy of `value`
    /// ([`igraph_strvector_set_len`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_strvector_set_len)).
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let mut names = StrVector::with_len(3);
    /// names.set(1, "hub").unwrap();
    /// assert_eq!(names.to_vec(), vec!["", "hub", ""]);
    /// assert_eq!(&names[1], "hub");
    /// assert!(names.set(3, "nope").is_err());
    /// ```
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if
    /// `index` is out of bounds.
    pub fn set(&mut self, index: usize, value: &str) -> crate::error::Result<()> {
        if index >= self.len() {
            return Err(crate::error::Error::invalid(format!(
                "string index {index} out of bounds (len {})",
                self.len()
            )));
        }
        crate::igraph_call!(igraph_strvector_set_len(
            self,
            index as igraph_int_t,
            value.as_ptr().cast(),
            value.len()
        ))
    }

    /// Removes the string at `index` and returns it (lossily converted to
    /// UTF-8) ([`igraph_strvector_remove`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_strvector_remove)).
    ///
    /// # Panics
    /// If `index` is out of bounds.
    pub fn remove(&mut self, index: usize) -> String {
        let len = self.len();
        assert!(
            index < len,
            "string index {index} out of bounds (len {len})"
        );
        let value = self
            .get_cstr(index)
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        unsafe { igraph_strvector_remove(self, index as igraph_int_t) };
        value
    }

    /// Removes and returns the last string, if any.
    pub fn pop(&mut self) -> Option<String> {
        let len = self.len();
        (len > 0).then(|| self.remove(len - 1))
    }

    /// Removes the strings in `range`
    /// ([`igraph_strvector_remove_section`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_strvector_remove_section)).
    ///
    /// # Panics
    /// If the range goes past the end.
    pub fn remove_section(&mut self, range: std::ops::Range<usize>) {
        let len = self.len();
        assert!(
            range.start <= range.end && range.end <= len,
            "section {range:?} out of bounds (len {len})"
        );
        unsafe {
            igraph_strvector_remove_section(
                self,
                range.start as igraph_int_t,
                range.end as igraph_int_t,
            )
        };
    }

    /// Appends copies of all the strings of `other`
    /// ([`igraph_strvector_append`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_strvector_append)).
    pub fn extend_from(&mut self, other: &Self) {
        crate::error::check(unsafe { igraph_strvector_append(self, other) })
            .expect("igraph failed to grow a string vector");
    }

    /// Moves all the strings of `other` to the end of `self`, leaving
    /// `other` empty, without copying them, like [`Vec::append`]
    /// ([`igraph_strvector_merge`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_strvector_merge)).
    pub fn append(&mut self, other: &mut Self) {
        crate::error::check(unsafe { igraph_strvector_merge(self, other) })
            .expect("igraph failed to grow a string vector");
    }

    /// Resizes to `len` strings: extra strings are dropped, new ones are
    /// empty ([`igraph_strvector_resize`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_strvector_resize)).
    pub fn resize(&mut self, len: usize) {
        crate::error::check(unsafe { igraph_strvector_resize(self, crate::error::int_size(len)) })
            .expect("igraph failed to resize a string vector");
    }

    /// Keeps only the first `len` strings (no-op if already shorter).
    pub fn truncate(&mut self, len: usize) {
        if len < self.len() {
            self.resize(len);
        }
    }

    /// Removes all strings
    /// ([`igraph_strvector_clear`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_strvector_clear)).
    pub fn clear(&mut self) {
        unsafe { igraph_strvector_clear(self) }
    }

    /// A new vector with the strings at the given positions (repetitions
    /// allowed) (`igraph_strvector_index`, undocumented in `igraph_strvector.h`).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::error::ErrorKind::InvalidValue) if
    /// some position is out of bounds.
    pub fn select(&self, index: &[igraph_int_t]) -> crate::error::Result<Self> {
        let len = self.len();
        if let Some(i) = index.iter().find(|&&i| i < 0 || i as usize >= len) {
            return Err(crate::error::Error::invalid(format!(
                "string index {i} out of bounds (len {len})"
            )));
        }
        let idx = crate::vector::VectorInt::view(index);
        let mut res = Self::new();
        crate::igraph_call!(igraph_strvector_index(self, &mut res, idx.as_ptr()))?;
        Ok(res)
    }

    /// Swaps the strings at `i` and `j`
    /// ([`igraph_strvector_swap_elements`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_strvector_swap_elements)).
    ///
    /// # Panics
    /// If an index is out of bounds.
    pub fn swap(&mut self, i: usize, j: usize) {
        let len = self.len();
        assert!(
            i < len && j < len,
            "string indices ({i}, {j}) out of bounds (len {len})"
        );
        unsafe { igraph_strvector_swap_elements(self, i as igraph_int_t, j as igraph_int_t) }
    }

    /// Number of strings that fit without reallocating
    /// ([`igraph_strvector_capacity`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_strvector_capacity)).
    pub fn capacity(&self) -> usize {
        unsafe { igraph_strvector_capacity(self) as usize }
    }

    /// Reserves room for at least `capacity` strings in total
    /// ([`igraph_strvector_reserve`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_strvector_reserve)).
    pub fn reserve(&mut self, capacity: usize) {
        crate::error::check(unsafe {
            igraph_strvector_reserve(self, crate::error::int_size(capacity))
        })
        .expect("igraph failed to reserve string vector storage");
    }

    /// Frees the unused storage
    /// ([`igraph_strvector_resize_min`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_strvector_resize_min)).
    pub fn shrink_to_fit(&mut self) {
        unsafe { igraph_strvector_resize_min(self) }
    }

    /// Whether some string equals `value`.
    pub fn contains(&self, value: &str) -> bool {
        self.position(value).is_some()
    }

    /// Position of the first string equal to `value`, if any.
    pub fn position(&self, value: &str) -> Option<usize> {
        (0..self.len()).find(|&i| {
            self.get_cstr(i)
                .is_some_and(|s| s.to_bytes() == value.as_bytes())
        })
    }
}

impl std::ops::Index<usize> for igraph_strvector_t {
    type Output = str;
    /// The string at `index`.
    ///
    /// # Panics
    /// If `index` is out of bounds or the string is not valid UTF-8.
    fn index(&self, index: usize) -> &str {
        let len = self.len();
        assert!(
            index < len,
            "string index {index} out of bounds (len {len})"
        );
        self.get(index).expect("the string is not valid UTF-8")
    }
}

impl<S: AsRef<str>> Extend<S> for igraph_strvector_t {
    fn extend<I: IntoIterator<Item = S>>(&mut self, iter: I) {
        for s in iter {
            self.push(s.as_ref());
        }
    }
}

impl From<Vec<String>> for igraph_strvector_t {
    fn from(items: Vec<String>) -> Self {
        items.iter().collect()
    }
}

impl From<igraph_strvector_t> for Vec<String> {
    fn from(sv: igraph_strvector_t) -> Self {
        sv.to_vec()
    }
}

impl<'a> IntoIterator for &'a igraph_strvector_t {
    type Item = String;
    type IntoIter = Box<dyn Iterator<Item = String> + 'a>;
    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.iter())
    }
}

impl PartialEq<[&str]> for igraph_strvector_t {
    fn eq(&self, other: &[&str]) -> bool {
        self.len() == other.len()
            && other.iter().enumerate().all(|(i, s)| {
                self.get_cstr(i)
                    .is_some_and(|c| c.to_bytes() == s.as_bytes())
            })
    }
}

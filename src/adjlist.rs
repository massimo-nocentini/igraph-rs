//! Adjacency and incidence lists (`igraph_adjlist.h`).
//!
//! Many algorithms visit the neighbors of every vertex over and over again
//! (shortest paths from all sources, closeness, triangle counting, rewiring,
//! ...). For them it pays off to extract the graph *once* into a list of
//! vectors, one per vertex, and work on that. igraph offers four such
//! structures, all wrapped here as owned Rust types that free their memory on
//! [`Drop`]:
//!
//! | Rust type          | C type                  | contents of list `v`            | built                  |
//! |--------------------|-------------------------|---------------------------------|------------------------|
//! | [`AdjList`]        | `igraph_adjlist_t`      | neighbor vertex ids of `v`      | eagerly, all vertices  |
//! | [`IncList`]        | `igraph_inclist_t`      | incident edge ids of `v`        | eagerly, all vertices  |
//! | [`LazyAdjList`]    | `igraph_lazy_adjlist_t` | neighbor vertex ids of `v`      | on first access of `v` |
//! | [`LazyIncList`]    | `igraph_lazy_inclist_t` | incident edge ids of `v`        | on first access of `v` |
//!
//! [`AdjList`] and [`IncList`] are *independent* of the graph after creation:
//! the graph may be modified or dropped, and the lists may be freely edited
//! (each entry is a [`VectorInt`], so it can grow and shrink) without
//! affecting the graph. This makes them the ideal scratch representation for
//! heavy structural edits: extract with [`Graph::adjlist_init`], edit the
//! lists in O(1) or O(d) per operation, then rebuild a graph with
//! [`AdjList::to_graph`] (`igraph_adjlist`), paying O(|V|+|E|) only once.
//!
//! The *lazy* variants instead borrow the graph (the borrow checker enforces
//! that the graph outlives them and is not mutated meanwhile) and query the
//! neighbors of a vertex only the first time it is asked for, caching the
//! result. They are handy for algorithms that may visit only a small part of
//! a large graph.
//!
//! # Rusty access
//!
//! - `al[v]` gives the neighbors of vertex `v` as a `&[i64]` slice
//!   ([`Index`]), `al[v][i] = x` edits in place ([`IndexMut`]);
//! - [`AdjList::get_mut`] gives the underlying [`VectorInt`] to push, pop,
//!   sort or resize a list;
//! - [`AdjList::iter`] / `for list in &al` iterate over the lists;
//! - [`AdjList::to_vecs`] and `Vec::<Vec<i64>>::from(al)` convert to nested
//!   vectors, and `AdjList::from(vec![...])` / `.collect()` build one from
//!   Rust data;
//! - [`Display`](std::fmt::Display) prints one line per vertex, exactly like
//!   igraph's `igraph_adjlist_print`.
//!
//! # Collapsing multi-edges
//!
//! With `multiple = false`, [`Graph::adjlist_init`] and
//! [`Graph::lazy_adjlist_init`] list each neighbor once. In igraph 1.0.0 and
//! 1.0.1 this is buggy when neighbors are gathered in both directions (mode
//! `All`, or undirected graphs): mutual pairs `u -> w`, `w -> u` and single
//! self-loops are mistaken for multi-edges, which corrupts the graph's cached
//! [`Graph::has_multiple`] answer, and the lists themselves depend on that
//! cache. These wrappers collapse such lists on the Rust side instead, so the
//! result always matches [`Graph::neighbors_with`] and the cache stays
//! correct.
//!
//! # Example
//!
//! ```
//! use igraph::prelude::*;
//! use igraph::adjlist::AdjList;
//!
//! // A directed triangle 0 -> 1 -> 2 -> 0 with an extra edge 0 -> 2.
//! let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (0, 2)], 3, true).unwrap();
//!
//! let out = g.adjlist_init(NeighborMode::Out, Loops::Twice, true).unwrap();
//! assert_eq!(out.len(), 3);
//! assert_eq!(&out[0], &[1, 2]);
//! assert_eq!(out.to_vecs(), vec![vec![1, 2], vec![2], vec![0]]);
//!
//! // Incoming neighbors, then the edge ids incident to each vertex.
//! let inc = g.adjlist_init(NeighborMode::In, Loops::Twice, true).unwrap();
//! assert_eq!(&inc[2], &[0, 1]);
//! let il = g.inclist_init(NeighborMode::Out, Loops::Twice).unwrap();
//! assert_eq!(&il[0], &[0, 3]);
//!
//! // Edit the adjacency list and turn it back into a graph: drop 2 -> 0.
//! let mut al: AdjList = out.clone();
//! al.get_mut(2).unwrap().clear();
//! al.get_mut(0).unwrap().push(0); // and add a self-loop on 0
//! let h = al.to_graph(NeighborMode::Out, false).unwrap();
//! assert_eq!(h.ecount(), 4);
//! assert_eq!(h.neighbors(0, NeighborMode::Out).unwrap(), vec![0, 1, 2]);
//!
//! // Lazy lists query the graph only when asked to.
//! let mut lazy = g.lazy_adjlist_init(NeighborMode::All, Loops::Twice, false).unwrap();
//! assert!(!lazy.has(0));
//! assert_eq!(lazy.get(0).unwrap(), &[1, 2]);
//! assert!(lazy.has(0));
//!
//! // Zachary's karate club: the handshake lemma, and the two hubs.
//! let karate = Graph::famous("Zachary").unwrap();
//! let al = karate.adjlist_init(NeighborMode::All, Loops::Twice, true).unwrap();
//! assert_eq!(al.total_len(), 2 * karate.ecount());
//! assert_eq!((al[0].len(), al[33].len()), (16, 17));
//! ```
//!
//! # Functions of `igraph_adjlist.h`
//!
//! | C function                           | Rust                                           |
//! |--------------------------------------|------------------------------------------------|
//! | `igraph_adjlist_init`                | [`Graph::adjlist_init`], [`AdjList::new`]      |
//! | `igraph_adjlist_init_empty`          | [`AdjList::init_empty`]                        |
//! | `igraph_adjlist_init_complementer`   | [`Graph::adjlist_init_complementer`], [`AdjList::complementer`] |
//! | `igraph_adjlist_init_from_inclist`   | [`Graph::adjlist_init_from_inclist`], [`AdjList::from_inclist`] |
//! | `igraph_adjlist_destroy`             | [`Drop`]                                       |
//! | `igraph_adjlist_size`                | [`AdjList::len`]                               |
//! | `igraph_adjlist_clear`               | [`AdjList::clear`]                             |
//! | `igraph_adjlist_sort`                | [`AdjList::sort`]                              |
//! | `igraph_adjlist_simplify`            | [`AdjList::simplify`]                          |
//! | `igraph_adjlist_print` / `_fprint`   | [`AdjList::print`], [`AdjList::fprint`], [`Display`](std::fmt::Display) |
//! | `igraph_adjlist_has_edge`            | [`AdjList::has_edge`]                          |
//! | `igraph_adjlist_replace_edge`        | [`AdjList::replace_edge`]                      |
//! | `igraph_adjlist_get` (macro)         | [`Index`], [`AdjList::get`], [`AdjList::get_mut`] |
//! | `igraph_adjlist`                     | [`Graph::adjlist`], [`AdjList::to_graph`]      |
//! | `igraph_inclist_init`                | [`Graph::inclist_init`], [`IncList::new`]      |
//! | `igraph_inclist_init_empty`          | [`IncList::init_empty`]                        |
//! | `igraph_inclist_destroy`             | [`Drop`]                                       |
//! | `igraph_inclist_size`                | [`IncList::len`]                               |
//! | `igraph_inclist_clear`               | [`IncList::clear`]                             |
//! | `igraph_inclist_print` / `_fprint`   | [`IncList::print`], [`IncList::fprint`], [`Display`](std::fmt::Display) |
//! | `igraph_inclist_get` (macro)         | [`Index`], [`IncList::get`], [`IncList::get_mut`] |
//! | `igraph_lazy_adjlist_init`           | [`Graph::lazy_adjlist_init`], [`LazyAdjList::new`] |
//! | `igraph_lazy_adjlist_destroy`        | [`Drop`]                                       |
//! | `igraph_lazy_adjlist_clear`          | [`LazyAdjList::clear`]                         |
//! | `igraph_lazy_adjlist_size`           | [`LazyAdjList::len`]                           |
//! | `igraph_lazy_adjlist_has` (macro)    | [`LazyAdjList::has`]                           |
//! | `igraph_lazy_adjlist_get` (macro)    | [`LazyAdjList::get`], [`LazyAdjList::get_mut`] |
//! | `igraph_lazy_inclist_init`           | [`Graph::lazy_inclist_init`], [`LazyIncList::new`] |
//! | `igraph_lazy_inclist_destroy`        | [`Drop`]                                       |
//! | `igraph_lazy_inclist_clear`          | [`LazyIncList::clear`]                         |
//! | `igraph_lazy_inclist_size`           | [`LazyIncList::len`]                           |
//! | `igraph_lazy_inclist_has` (macro)    | [`LazyIncList::has`]                           |
//! | `igraph_lazy_inclist_get` (macro)    | [`LazyIncList::get`], [`LazyIncList::get_mut`] |
//!
//! # See also
//!
//! - [`Graph::neighbors_with`] and [`Graph::incident`] query a single vertex
//!   directly on the graph, without building a whole list;
//! - [`Graph::get_adjacency`] gives the dense adjacency
//!   matrix and [`Graph::adjacency`] builds a graph from
//!   one, the matrix counterparts of [`Graph::adjlist_init`] and
//!   [`Graph::adjlist`];
//! - [`Graph::complementer`] and [`Graph::simplify`]
//!   ([`operators`](crate::operators)) are the whole-graph counterparts of
//!   [`Graph::adjlist_init_complementer`] and [`AdjList::simplify`];
//! - [`Graph::rewire`] implements degree-preserving rewiring on top of
//!   [`AdjList::has_edge`] / [`AdjList::replace_edge`];
//! - [`Graph::bfs_simple`] and the other traversals of
//!   [`visitor`](crate::visitor) walk the graph for you.
//!
//! The igraph C documentation of
//! [adjacency lists](https://igraph.org/c/html/latest/igraph-Data-structures.html)
//! describes the underlying structures.

use crate::{
    constants::{Loops, NeighborMode},
    error::{Error, ErrorKind, Result, check, ensure_init},
    ffi::*,
    graph::{Graph, VertexId},
    igraph_call,
    vector::VectorInt,
};
use std::{
    ffi::c_char,
    fmt,
    iter::FusedIterator,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Index, IndexMut},
};

/// Owned adjacency list (`igraph_adjlist_t`): for each vertex, a vector of
/// its neighbor vertex ids. See the [module docs](self).
pub type AdjList = igraph_adjlist_t;

/// Owned incidence list (`igraph_inclist_t`): for each vertex, a vector of
/// the ids of its incident edges. See the [module docs](self).
pub type IncList = igraph_inclist_t;

/// Iterator over the per-vertex lists of an [`AdjList`] or [`IncList`],
/// yielding each list as a slice (returned by [`AdjList::iter`] and
/// [`IncList::iter`]).
#[derive(Debug, Clone)]
pub struct Iter<'a> {
    inner: std::slice::Iter<'a, VectorInt>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a [igraph_int_t];
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(VectorInt::as_slice)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.inner.nth(n).map(VectorInt::as_slice)
    }
}

impl DoubleEndedIterator for Iter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(VectorInt::as_slice)
    }
}

impl ExactSizeIterator for Iter<'_> {}
impl FusedIterator for Iter<'_> {}

/// Runs `f` on a C `FILE*` backed by memory and returns what was written.
///
/// Used to expose igraph's `*_fprint` functions through [`std::io::Write`].
fn capture_c_output(f: impl FnOnce(*mut FILE) -> igraph_error_t) -> Result<Vec<u8>> {
    ensure_init();
    let mut buf: *mut c_char = std::ptr::null_mut();
    let mut size: usize = 0;
    let file = unsafe { open_memstream(&mut buf, &mut size) };
    if file.is_null() {
        return Err(Error::new(ErrorKind::File, "cannot open a memory stream"));
    }
    let code = f(file);
    // `fclose` finalizes `buf` and `size`.
    let closed = unsafe { fclose(file) };
    let bytes = if buf.is_null() {
        Vec::new()
    } else {
        let bytes = unsafe { std::slice::from_raw_parts(buf as *const u8, size) }.to_vec();
        unsafe { free(buf.cast()) };
        bytes
    };
    check(code)?;
    if closed != 0 {
        return Err(Error::new(
            ErrorKind::File,
            "cannot close the memory stream",
        ));
    }
    Ok(bytes)
}

/// Writes `bytes` to `out`, mapping I/O errors to [`ErrorKind::File`].
fn write_all(out: &mut impl std::io::Write, bytes: &[u8]) -> Result<()> {
    out.write_all(bytes)
        .map_err(|e| Error::new(ErrorKind::File, format!("write failed: {e}")))
}

/// The error returned by the lazy lists when igraph could not compute the
/// requested list (the only possible cause, for a valid vertex, is lack of
/// memory).
fn lazy_failure() -> Error {
    match check(igraph_error_type_t_IGRAPH_ENOMEM) {
        Err(e) => e,
        Ok(()) => Error::new(ErrorKind::OutOfMemory, "cannot query the lazy list"),
    }
}

/// How many copies of a self-loop survive in the list of its vertex when
/// multi-edges are collapsed, or `None` when igraph's own collapsing can be
/// used.
///
/// Workaround for a bug of igraph 1.0.0 and 1.0.1 (`src/graph/adjlist.c`,
/// unchanged in 1.0.1): when `multiple = false` and the neighbors are
/// gathered in both directions (mode `All`, or any undirected graph), igraph
/// mistakes the two entries of a mutual pair `u -> w`, `w -> u` or of a
/// single self-loop for a multi-edge. `igraph_adjlist_init` then caches
/// "this graph has multi-edges" on the graph, so that later
/// [`Graph::has_multiple`] / [`Graph::is_simple`] calls give wrong answers;
/// and once the cache says "no multi-edges" (e.g. after [`Graph::simplify`]),
/// both `igraph_adjlist_init` and `igraph_lazy_adjlist_init` skip collapsing,
/// so mutual pairs of a directed graph are listed twice or once depending on
/// the cache state. In these cases the lists are therefore requested *with*
/// multi-edges and collapsed on the Rust side, with exactly igraph's
/// semantics: each neighbor is listed once, and a self-loop (if kept) once,
/// or twice with [`Loops::Twice`].
fn collapse_rule(graph: &Graph, mode: NeighborMode, loops: Loops, multiple: bool) -> Option<usize> {
    let both_ways = mode == NeighborMode::All || !graph.is_directed();
    if multiple || !both_ways {
        return None;
    }
    Some(if loops == Loops::Twice { 2 } else { 1 })
}

/// Collapses repeated entries of the *sorted* neighbor list of vertex `v`:
/// every neighbor is kept once, `v` itself at most `max_loops` times.
fn collapse_sorted(list: &mut VectorInt, v: VertexId, max_loops: usize) {
    let slice = list.as_mut_slice();
    let (mut read, mut write) = (0, 0);
    while read < slice.len() {
        let x = slice[read];
        let run = slice[read..].iter().take_while(|&&y| y == x).count();
        let keep = if x == v { run.min(max_loops) } else { 1 };
        for _ in 0..keep {
            slice[write] = x;
            write += 1;
        }
        read += run;
    }
    list.truncate(write);
}

fn check_vertex(v: VertexId, n: usize) -> Result<usize> {
    if v < 0 || v as u64 >= n as u64 {
        Err(Error::new(
            ErrorKind::InvalidVertexId,
            format!("vertex id {v} out of range for a list of {n} vertices"),
        ))
    } else {
        Ok(v as usize)
    }
}

// Shared behaviour of the two eager lists, whose C layouts are identical
// (a length and a C array of `igraph_vector_int_t`).
macro_rules! impl_eager_list {
    (
        $ty:ident, $field:ident, $what:literal, $item:literal, c = $c:literal,
        init_empty = $init_empty:ident, destroy = $destroy:ident,
        clear = $clear:ident, print = $print:ident, fprint = $fprint:ident
    ) => {
        impl $ty {
            #[doc = concat!("Creates ", $what, " for `n` vertices, all with empty lists (`", stringify!($init_empty), "`).")]
            ///
            /// Useful to *build* a structure vertex by vertex, e.g. before
            /// turning it into a graph. Time complexity: O(n).
            ///
            #[doc = concat!("Binds [`", stringify!($init_empty), "`](", "https://igraph.org/c/html/latest/igraph-Data-structures.html#", stringify!($init_empty), ").")]
            ///
            /// # Errors
            /// [`ErrorKind::InvalidValue`] if `n` does not fit in an
            /// `igraph_int_t`, or an out-of-memory error if it is too large
            /// to be allocated.
            pub fn init_empty(n: usize) -> Result<Self> {
                // A negative length would make igraph allocate a single slot
                // and record a negative size.
                let n = igraph_int_t::try_from(n)
                    .map_err(|_| Error::invalid(format!("{n} vertices is too many")))?;
                let mut raw = MaybeUninit::<Self>::zeroed();
                igraph_call!($init_empty(raw.as_mut_ptr(), n))?;
                Ok(unsafe { raw.assume_init() })
            }

            #[doc = concat!("Number of vertices, i.e. of lists, in the ", $what, " (`", $c, "_size`).")]
            ///
            /// Time complexity: O(1).
            ///
            #[doc = concat!("Binds [`", $c, "_size`](", "https://igraph.org/c/html/latest/igraph-Data-structures.html#", $c, "_size).")]
            pub fn len(&self) -> usize {
                if self.$field.is_null() { 0 } else { self.length.max(0) as usize }
            }

            /// Whether there are no vertices at all.
            pub fn is_empty(&self) -> bool {
                self.len() == 0
            }

            #[doc = concat!("Total number of entries (", $item, ") stored in all the lists.")]
            pub fn total_len(&self) -> usize {
                self.as_raw_slice().iter().map(|v| v.len()).sum()
            }

            #[doc = concat!("Removes all ", $item, " from every list, keeping the number of vertices (`", stringify!($clear), "`).")]
            ///
            /// Time complexity: O(n), n being the number of vertices.
            ///
            #[doc = concat!("Binds [`", stringify!($clear), "`](", "https://igraph.org/c/html/latest/igraph-Data-structures.html#", stringify!($clear), ").")]
            pub fn clear(&mut self) {
                if !self.$field.is_null() {
                    unsafe { $clear(self) };
                }
            }

            /// The per-vertex lists as a slice of owned igraph vectors.
            pub fn as_raw_slice(&self) -> &[VectorInt] {
                if self.$field.is_null() || self.length <= 0 {
                    &[]
                } else {
                    unsafe { std::slice::from_raw_parts(self.$field, self.length as usize) }
                }
            }

            /// The per-vertex lists as a mutable slice of owned igraph
            /// vectors: every list can be resized, pushed to, sorted, or even
            /// replaced by another [`VectorInt`].
            pub fn as_raw_mut_slice(&mut self) -> &mut [VectorInt] {
                if self.$field.is_null() || self.length <= 0 {
                    &mut []
                } else {
                    unsafe { std::slice::from_raw_parts_mut(self.$field, self.length as usize) }
                }
            }

            #[doc = concat!("The list of vertex `v`, or `None` if `v` is out of range (the [`", $c, "_get`](", "https://igraph.org/c/html/latest/igraph-Data-structures.html#", $c, "_get) macro).")]
            ///
            /// The panicking counterpart is indexing: `list[v]`.
            pub fn get(&self, v: VertexId) -> Option<&[igraph_int_t]> {
                usize::try_from(v).ok().and_then(|i| self.as_raw_slice().get(i)).map(|x| x.as_slice())
            }

            #[doc = concat!("The list of vertex `v` as a mutable [`VectorInt`], or `None` if `v` is out of range.")]
            ///
            /// Unlike `IndexMut`, this allows changing the length of the list
            /// (`push`, `pop`, `resize`, `clear`, ...).
            pub fn get_mut(&mut self, v: VertexId) -> Option<&mut VectorInt> {
                usize::try_from(v).ok().and_then(move |i| self.as_raw_mut_slice().get_mut(i))
            }

            /// Iterates over the lists of the vertices `0, 1, ...`, as slices.
            pub fn iter(&self) -> Iter<'_> {
                Iter { inner: self.as_raw_slice().iter() }
            }

            /// Iterates mutably over the lists of the vertices `0, 1, ...`.
            pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, VectorInt> {
                self.as_raw_mut_slice().iter_mut()
            }

            /// Copies the lists into nested Rust vectors.
            pub fn to_vecs(&self) -> Vec<Vec<igraph_int_t>> {
                self.iter().map(<[igraph_int_t]>::to_vec).collect()
            }

            #[doc = concat!("Prints the lists to the standard output, one vertex per line, entries separated by spaces (`", stringify!($print), "`).")]
            ///
            /// The same text is produced by the `Display` implementation and
            #[doc = concat!("by [`", stringify!($ty), "::fprint`].")]
            ///
            /// Note that the text goes to the C `stdout` stream, bypassing
            /// Rust's output capturing (e.g. in `cargo test`): prefer
            /// `println!("{list}")` in Rust code.
            pub fn print(&self) -> Result<()> {
                igraph_call!($print(self))
            }

            #[doc = concat!("Writes the lists to `out` in igraph's textual format, one vertex per line (`", stringify!($fprint), "`).")]
            ///
            /// The output is produced by the C library itself (through a
            /// memory stream), and it matches the `Display` implementation.
            ///
            /// # Errors
            /// [`ErrorKind::File`] if writing fails.
            pub fn fprint(&self, out: &mut impl std::io::Write) -> Result<()> {
                let bytes = capture_c_output(|f| unsafe { $fprint(self, f) })?;
                write_all(out, &bytes)
            }
        }

        impl Drop for $ty {
            #[doc = concat!("Frees the lists with `", stringify!($destroy), "`.")]
            fn drop(&mut self) {
                if !self.$field.is_null() {
                    unsafe { $destroy(self) };
                    self.$field = std::ptr::null_mut();
                    self.length = 0;
                }
            }
        }

        impl Clone for $ty {
            fn clone(&self) -> Self {
                let mut copy = Self::init_empty(self.len()).expect("igraph failed to allocate a list");
                for (dst, src) in copy.as_raw_mut_slice().iter_mut().zip(self.as_raw_slice()) {
                    *dst = src.clone();
                }
                copy
            }
        }

        impl Default for $ty {
            /// An empty structure with no vertices.
            fn default() -> Self {
                Self::init_empty(0).expect("igraph failed to allocate a list")
            }
        }

        impl PartialEq for $ty {
            /// Equal when both have the same number of vertices and the same
            /// lists, in the same order.
            fn eq(&self, other: &Self) -> bool {
                self.as_raw_slice() == other.as_raw_slice()
            }
        }

        impl Index<usize> for $ty {
            type Output = [igraph_int_t];
            /// The list of the given vertex as a slice.
            ///
            /// # Panics
            /// If the vertex is out of range.
            fn index(&self, v: usize) -> &[igraph_int_t] {
                self.as_raw_slice()[v].as_slice()
            }
        }

        impl IndexMut<usize> for $ty {
            /// The list of the given vertex as a mutable slice (its length
            #[doc = concat!("cannot change: use [`", stringify!($ty), "::get_mut`] for that).")]
            ///
            /// # Panics
            /// If the vertex is out of range.
            fn index_mut(&mut self, v: usize) -> &mut [igraph_int_t] {
                self.as_raw_mut_slice()[v].as_mut_slice()
            }
        }

        impl<'a> IntoIterator for &'a $ty {
            type Item = &'a [igraph_int_t];
            type IntoIter = Iter<'a>;
            fn into_iter(self) -> Iter<'a> {
                self.iter()
            }
        }

        impl fmt::Display for $ty {
            /// One line per vertex with the entries separated by spaces,
            #[doc = concat!("exactly like `", stringify!($print), "`.")]
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                for list in self {
                    let mut first = true;
                    for x in list {
                        if !first {
                            f.write_str(" ")?;
                        }
                        first = false;
                        write!(f, "{x}")?;
                    }
                    writeln!(f)?;
                }
                Ok(())
            }
        }

        impl From<&[Vec<igraph_int_t>]> for $ty {
            /// Builds the structure from nested vectors, one per vertex.
            fn from(lists: &[Vec<igraph_int_t>]) -> Self {
                let mut res = Self::init_empty(lists.len()).expect("igraph failed to allocate a list");
                for (dst, src) in res.as_raw_mut_slice().iter_mut().zip(lists) {
                    *dst = VectorInt::from_slice(src);
                }
                res
            }
        }

        impl From<Vec<Vec<igraph_int_t>>> for $ty {
            /// Builds the structure from nested vectors, one per vertex.
            fn from(lists: Vec<Vec<igraph_int_t>>) -> Self {
                Self::from(lists.as_slice())
            }
        }

        impl FromIterator<Vec<igraph_int_t>> for $ty {
            fn from_iter<I: IntoIterator<Item = Vec<igraph_int_t>>>(iter: I) -> Self {
                let lists: Vec<_> = iter.into_iter().collect();
                Self::from(lists.as_slice())
            }
        }

        impl From<$ty> for Vec<Vec<igraph_int_t>> {
            fn from(list: $ty) -> Self {
                list.to_vecs()
            }
        }

        impl From<&$ty> for Vec<Vec<igraph_int_t>> {
            fn from(list: &$ty) -> Self {
                list.to_vecs()
            }
        }

        // The structure owns plain memory, with no interior mutability and no
        // pointer to a graph.
        unsafe impl Send for $ty {}
        unsafe impl Sync for $ty {}
    };
}

impl_eager_list!(
    igraph_adjlist_t,
    adjs,
    "an adjacency list",
    "neighbor ids",
    c = "igraph_adjlist",
    init_empty = igraph_adjlist_init_empty,
    destroy = igraph_adjlist_destroy,
    clear = igraph_adjlist_clear,
    print = igraph_adjlist_print,
    fprint = igraph_adjlist_fprint
);
impl_eager_list!(
    igraph_inclist_t,
    incs,
    "an incidence list",
    "edge ids",
    c = "igraph_inclist",
    init_empty = igraph_inclist_init_empty,
    destroy = igraph_inclist_destroy,
    clear = igraph_inclist_clear,
    print = igraph_inclist_print,
    fprint = igraph_inclist_fprint
);

impl igraph_adjlist_t {
    /// Adjacency list of `graph`: same as [`Graph::adjlist_init`].
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::adjlist::AdjList;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// let al = AdjList::new(&g, NeighborMode::All, Loops::Twice, true).unwrap();
    /// assert_eq!(&al[1], &[0, 2]);
    /// ```
    pub fn new(graph: &Graph, mode: NeighborMode, loops: Loops, multiple: bool) -> Result<Self> {
        graph.adjlist_init(mode, loops, multiple)
    }

    /// Adjacency list of the complementer of `graph`: same as
    /// [`Graph::adjlist_init_complementer`].
    pub fn complementer(graph: &Graph, mode: NeighborMode, loops: Loops) -> Result<Self> {
        graph.adjlist_init_complementer(mode, loops)
    }

    /// Adjacency list consistent with an incidence list of `graph`: same as
    /// [`Graph::adjlist_init_from_inclist`].
    pub fn from_inclist(graph: &Graph, inclist: &IncList) -> Result<Self> {
        graph.adjlist_init_from_inclist(inclist)
    }

    /// Checks that every stored neighbor id is a valid index of the list.
    fn check_ids(&self) -> Result<()> {
        let n = self.len();
        for (v, list) in self.iter().enumerate() {
            if let Some(&bad) = list.iter().find(|&&u| u < 0 || u as u64 >= n as u64) {
                return Err(Error::new(
                    ErrorKind::InvalidVertexId,
                    format!("the list of vertex {v} contains the invalid vertex id {bad}"),
                ));
            }
        }
        Ok(())
    }

    /// Checks that every edge between distinct vertices is listed from both
    /// endpoints the same number of times, and that every self-loop is
    /// listed an even number of times (the `duplicate = true` format).
    fn check_duplicated(&self) -> Result<()> {
        let mut forward = Vec::with_capacity(self.total_len());
        let mut backward = Vec::with_capacity(self.total_len());
        for (u, list) in self.iter().enumerate() {
            let u = u as igraph_int_t;
            let loops = list.iter().filter(|&&w| w == u).count();
            if loops % 2 != 0 {
                return Err(Error::invalid(format!(
                    "vertex {u} lists itself {loops} times: self-loops must be listed twice \
                     when duplicate = true"
                )));
            }
            for &w in list.iter().filter(|&&w| w != u) {
                forward.push((u, w));
                backward.push((w, u));
            }
        }
        forward.sort_unstable();
        backward.sort_unstable();
        if forward != backward {
            let bad = forward
                .iter()
                .find(|e| backward.binary_search(e).is_err())
                .or_else(|| backward.iter().find(|e| forward.binary_search(e).is_err()))
                .copied()
                .unwrap_or((0, 0));
            return Err(Error::invalid(format!(
                "the undirected edge {{{}, {}}} is not listed equally often by both of its \
                 endpoints, as required when duplicate = true",
                bad.0.min(bad.1),
                bad.0.max(bad.1)
            )));
        }
        Ok(())
    }

    /// Sorts every neighbor list in increasing order (`igraph_adjlist_sort`).
    ///
    /// Lists created by [`Graph::adjlist_init`] are already sorted; this is
    /// useful after edits, or after [`simplify`](Self::simplify), which does
    /// not preserve the order. Sorted lists are required by
    /// [`has_edge`](Self::has_edge) and [`replace_edge`](Self::replace_edge).
    ///
    /// Time complexity: O(m log m), m being the total number of entries.
    ///
    /// Binds [`igraph_adjlist_sort`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_adjlist_sort).
    pub fn sort(&mut self) {
        if !self.adjs.is_null() {
            unsafe { igraph_adjlist_sort(self) };
        }
    }

    /// Removes self-loops and repeated neighbors from every list
    /// (`igraph_adjlist_simplify`).
    ///
    /// After the call, vertex `v` appears in no list of its own and each
    /// neighbor appears at most once per list. The order of the remaining
    /// entries is **not** preserved (removed entries are replaced with the
    /// last one): call [`sort`](Self::sort) afterwards if needed. When the
    /// list comes from a graph, prefer passing `Loops::None` and
    /// `multiple = false` to [`Graph::adjlist_init`] instead; to simplify
    /// the graph itself use [`Graph::simplify`].
    ///
    /// Time complexity: O(|V|+|E|).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`] if some list contains an id that is not
    /// a vertex of the list (checked on the Rust side, as the C code would
    /// read out of bounds).
    ///
    /// # Examples
    /// ```
    /// use igraph::adjlist::AdjList;
    /// let mut al = AdjList::from(vec![vec![0, 1, 1, 2], vec![0, 0], vec![0, 2]]);
    /// al.simplify().unwrap();
    /// al.sort();
    /// assert_eq!(al.to_vecs(), vec![vec![1, 2], vec![0], vec![0]]);
    /// ```
    ///
    /// Binds [`igraph_adjlist_simplify`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_adjlist_simplify).
    pub fn simplify(&mut self) -> Result<()> {
        self.check_ids()?;
        if self.adjs.is_null() {
            return Ok(());
        }
        igraph_call!(igraph_adjlist_simplify(self))
    }

    /// Whether the adjacency list contains the edge `from -> to`
    /// (`igraph_adjlist_has_edge`), by binary search.
    ///
    /// The lists **must be sorted** (see [`sort`](Self::sort)), otherwise the
    /// answer is unspecified. When `directed` is `true`, `to` is searched in
    /// the list of `from`. When it is `false`, the edge is looked up in the
    /// list of the *larger* endpoint only, i.e. `min(from, to)` is searched in
    /// the list of `max(from, to)`: this matches both a full undirected
    /// adjacency list and a "half" one where each vertex only stores its
    /// neighbors with smaller or equal ids (the representation that
    /// [`replace_edge`](Self::replace_edge) keeps consistent).
    ///
    /// Time complexity: O(log d), d being the length of the searched list.
    ///
    /// See also [`Graph::get_eid`], which answers the same question on the
    /// graph itself.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`] if `from` or `to` is out of range.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    /// let al = g.adjlist_init(NeighborMode::Out, Loops::Once, true).unwrap();
    /// assert!(al.has_edge(0, 1, true).unwrap());
    /// assert!(!al.has_edge(1, 0, true).unwrap());
    /// assert_eq!(al.has_edge(0, 9, true).unwrap_err().kind(), ErrorKind::InvalidVertexId);
    /// ```
    ///
    /// Binds `igraph_adjlist_has_edge` (declared in `igraph_adjlist.h` but not part of the
    /// C reference manual).
    pub fn has_edge(&self, from: VertexId, to: VertexId, directed: bool) -> Result<bool> {
        let n = self.len();
        check_vertex(from, n)?;
        check_vertex(to, n)?;
        // The C function takes a mutable pointer but does not modify the list.
        let ptr = self as *const Self as *mut Self;
        ensure_init();
        Ok(unsafe { igraph_adjlist_has_edge(ptr, from, to, directed) })
    }

    /// Replaces the edge `from -> oldto` with `from -> newto`, keeping the
    /// lists sorted (`igraph_adjlist_replace_edge`).
    ///
    /// The lists **must be sorted**. With `directed = true` the change is
    /// made in the list of `from`. With `directed = false` the edges are
    /// canonicalized as in [`has_edge`](Self::has_edge): `{from, oldto}` is
    /// removed from the list of `max(from, oldto)` and `{from, newto}` is
    /// inserted in the list of `max(from, newto)`; the lists of the smaller
    /// endpoints are *not* touched, so this is meant for "half" undirected
    /// adjacency lists where each vertex only stores its neighbors with
    /// smaller or equal ids. This is the primitive of igraph's fast
    /// degree-preserving rewiring, available ready-made as
    /// [`Graph::rewire`].
    ///
    /// Time complexity: O(d), d being the length of the lists involved.
    ///
    /// # Errors
    /// - [`ErrorKind::InvalidVertexId`] if a vertex is out of range;
    /// - [`ErrorKind::InvalidValue`] if the edge to replace does not exist or
    ///   the new edge already exists.
    ///
    /// # Examples
    /// ```
    /// use igraph::adjlist::AdjList;
    /// // Directed: 0 -> 1, 0 -> 2.
    /// let mut al = AdjList::from(vec![vec![1, 2], vec![], vec![], vec![]]);
    /// al.replace_edge(0, 1, 3, true).unwrap();
    /// assert_eq!(&al[0], &[2, 3]);
    /// assert!(al.replace_edge(0, 1, 3, true).is_err()); // 0 -> 1 is gone
    /// ```
    ///
    /// Binds `igraph_adjlist_replace_edge` (declared in `igraph_adjlist.h` but not part of the
    /// C reference manual).
    pub fn replace_edge(
        &mut self,
        from: VertexId,
        oldto: VertexId,
        newto: VertexId,
        directed: bool,
    ) -> Result<()> {
        let n = self.len();
        check_vertex(from, n)?;
        check_vertex(oldto, n)?;
        check_vertex(newto, n)?;
        igraph_call!(igraph_adjlist_replace_edge(
            self, from, oldto, newto, directed
        ))
    }

    /// Builds a graph from this adjacency list: same as [`Graph::adjlist`].
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::adjlist::AdjList;
    /// let al = AdjList::from(vec![vec![1, 2], vec![2], vec![]]);
    /// let g = al.to_graph(NeighborMode::Out, false).unwrap();
    /// assert!(g.is_directed());
    /// assert_eq!(g.edge_list(), vec![(0, 1), (0, 2), (1, 2)]);
    /// ```
    pub fn to_graph(&self, mode: NeighborMode, duplicate: bool) -> Result<Graph> {
        Graph::adjlist(self, mode, duplicate)
    }
}

impl igraph_inclist_t {
    /// Incidence list of `graph`: same as [`Graph::inclist_init`].
    pub fn new(graph: &Graph, mode: NeighborMode, loops: Loops) -> Result<Self> {
        graph.inclist_init(mode, loops)
    }
}

impl igraph_t {
    /// Adjacency list of the graph: for each vertex, the sorted ids of its
    /// neighbors (`igraph_adjlist_init`).
    ///
    /// The list is independent of the graph after creation: it reflects the
    /// graph at the time of the call and can be edited freely.
    ///
    /// - `mode`: in directed graphs, whether to list successors
    ///   ([`NeighborMode::Out`]), predecessors ([`NeighborMode::In`]) or both
    ///   ([`NeighborMode::All`]); ignored for undirected graphs.
    /// - `loops`: [`Loops::None`] drops self-loops; [`Loops::Once`] lists each
    ///   loop edge once in the list of its vertex; [`Loops::Twice`] lists it
    ///   twice, but only if the graph is undirected or `mode` is
    ///   [`NeighborMode::All`] (otherwise it behaves as `Once`).
    /// - `multiple`: `true` keeps parallel edges, so a neighbor appears as many
    ///   times as there are edges to it; `false` lists each neighbor once
    ///   (with mode `All` this also merges a mutual pair `u -> w`, `w -> u` of
    ///   a directed graph) and each kept self-loop once, or twice with
    ///   `Loops::Twice` in an undirected graph or with mode `All`.
    ///
    /// igraph 1.0.0 and 1.0.1 get `multiple = false` wrong when neighbors are
    /// collected in both directions (mode `All`, or an undirected graph):
    /// they mistake mutual pairs and single self-loops for multi-edges and
    /// record that in the graph's property cache (so that
    /// [`Graph::has_multiple`] later answers `true` for a graph without
    /// multi-edges), and, if the cache already says "no multi-edges", they
    /// list mutual pairs twice. This wrapper avoids both problems by
    /// collapsing such lists itself, with the semantics described above.
    ///
    /// `Loops::Twice` with `multiple = true` is the fastest combination.
    /// The lists are currently sorted, but igraph does not guarantee this for
    /// the future: call [`AdjList::sort`] if you rely on it. The list of
    /// vertex `v` equals [`Graph::neighbors_with`]`(v, mode, loops, multiple)`.
    ///
    /// Time complexity: O(|V|+|E|).
    ///
    /// See also [`Graph::lazy_adjlist_init`] (on-demand variant),
    /// [`Graph::inclist_init`] (edge ids instead of vertex ids),
    /// [`Graph::neighbors_with`] (one vertex) and
    /// [`Graph::get_adjacency`] (dense matrix).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // An undirected graph with a double edge 0-1 and a loop on 2.
    /// let g = Graph::from_edges(&[(0, 1), (0, 1), (1, 2), (2, 2)], 3, false).unwrap();
    /// let full = g.adjlist_init(NeighborMode::All, Loops::Twice, true).unwrap();
    /// assert_eq!(full.to_vecs(), vec![vec![1, 1], vec![0, 0, 2], vec![1, 2, 2]]);
    /// let simple = g.adjlist_init(NeighborMode::All, Loops::None, false).unwrap();
    /// assert_eq!(simple.to_vecs(), vec![vec![1], vec![0, 2], vec![1]]);
    /// ```
    ///
    /// Binds [`igraph_adjlist_init`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_adjlist_init).
    pub fn adjlist_init(
        &self,
        mode: NeighborMode,
        loops: Loops,
        multiple: bool,
    ) -> Result<AdjList> {
        let collapse = collapse_rule(self, mode, loops, multiple);
        let mut raw = MaybeUninit::<AdjList>::zeroed();
        igraph_call!(igraph_adjlist_init(
            self,
            raw.as_mut_ptr(),
            mode.into(),
            loops.into(),
            multiple || collapse.is_some()
        ))?;
        let mut al = unsafe { raw.assume_init() };
        if let Some(max_loops) = collapse {
            for (v, list) in al.iter_mut().enumerate() {
                collapse_sorted(list, v as VertexId, max_loops);
            }
        }
        Ok(al)
    }

    /// Adjacency list of the *complementer* graph, i.e. of the graph having
    /// exactly the edges missing from this one
    /// (`igraph_adjlist_init_complementer`).
    ///
    /// Multi-edges of the input are ignored and the lists are sorted.
    ///
    /// - `mode`: which neighbors *in the complementer* to list for directed
    ///   graphs (ignored for undirected ones);
    /// - `loops`: [`Loops::None`] never lists `v` among its own neighbors;
    ///   [`Loops::Once`] lists it once if the graph has no loop on `v`;
    ///   [`Loops::Twice`] lists it twice in that case when `mode` is
    ///   [`NeighborMode::All`], and behaves as `Once` otherwise.
    ///
    /// Time complexity: O(|V|²+|E|).
    ///
    /// See also [`Graph::complementer`], which builds the complementer as a
    /// graph.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // The complement of the path 0 - 1 - 2 - 3 is the path 1 - 3 - 0 - 2.
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// let c = g.adjlist_init_complementer(NeighborMode::All, Loops::None).unwrap();
    /// assert_eq!(c.to_vecs(), vec![vec![2, 3], vec![3], vec![0], vec![0, 1]]);
    /// ```
    ///
    /// Binds [`igraph_adjlist_init_complementer`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_adjlist_init_complementer).
    pub fn adjlist_init_complementer(&self, mode: NeighborMode, loops: Loops) -> Result<AdjList> {
        let mut raw = MaybeUninit::<AdjList>::zeroed();
        igraph_call!(igraph_adjlist_init_complementer(
            self,
            raw.as_mut_ptr(),
            mode.into(),
            loops.into()
        ))?;
        Ok(unsafe { raw.assume_init() })
    }

    /// Adjacency list *consistent* with an incidence list of this graph
    /// (`igraph_adjlist_init_from_inclist`).
    ///
    /// Entry `i` of the list of vertex `v` is the other endpoint of the edge
    /// at entry `i` of `inclist[v]`, so the two structures can be walked in
    /// lockstep (neighbor and connecting edge together). The result is
    /// independent of both the graph and the incidence list.
    ///
    /// Time complexity: O(|V|+|E|).
    ///
    /// # Errors
    /// - [`ErrorKind::InvalidValue`] if the incidence list does not have one
    ///   entry per vertex of the graph;
    /// - [`ErrorKind::InvalidEdgeId`] if it contains an id that is not an edge
    ///   of the graph (checked on the Rust side).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (2, 0)], 3, true).unwrap();
    /// let il = g.inclist_init(NeighborMode::All, Loops::Twice).unwrap();
    /// let al = g.adjlist_init_from_inclist(&il).unwrap();
    /// for v in 0..3 {
    ///     for (&e, &u) in il[v].iter().zip(&al[v]) {
    ///         let (a, b) = g.edge(e).unwrap();
    ///         assert!((a, b) == (v as i64, u) || (a, b) == (u, v as i64));
    ///     }
    /// }
    /// ```
    ///
    /// Binds [`igraph_adjlist_init_from_inclist`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_adjlist_init_from_inclist).
    pub fn adjlist_init_from_inclist(&self, inclist: &IncList) -> Result<AdjList> {
        if inclist.len() != self.vcount() {
            return Err(Error::invalid(format!(
                "incidence list has {} entries but the graph has {} vertices",
                inclist.len(),
                self.vcount()
            )));
        }
        let m = self.ecount() as u64;
        for (v, list) in inclist.iter().enumerate() {
            if let Some(&bad) = list.iter().find(|&&e| e < 0 || e as u64 >= m) {
                return Err(Error::new(
                    ErrorKind::InvalidEdgeId,
                    format!("the incidence list of vertex {v} contains the invalid edge id {bad}"),
                ));
            }
        }
        let mut raw = MaybeUninit::<AdjList>::zeroed();
        igraph_call!(igraph_adjlist_init_from_inclist(
            self,
            raw.as_mut_ptr(),
            inclist
        ))?;
        Ok(unsafe { raw.assume_init() })
    }

    /// Creates a graph from an adjacency list (`igraph_adjlist`), the inverse
    /// of [`Graph::adjlist_init`].
    ///
    /// The graph has one vertex per list.
    ///
    /// - `mode`: [`NeighborMode::All`] creates an *undirected* graph;
    ///   [`NeighborMode::Out`] a directed graph where each list holds the
    ///   successors of its vertex; [`NeighborMode::In`] a directed graph where
    ///   each list holds the predecessors.
    /// - `duplicate`: for undirected graphs only, whether each edge is listed
    ///   twice (in the lists of both endpoints, as produced by
    ///   [`Graph::adjlist_init`]; a self-loop then appears twice in its list)
    ///   or just once.
    ///
    /// Converting a graph into an adjacency list, doing many structural
    /// edits on the lists and converting back costs O(|V|+|E|) overall,
    /// much less than repeated edge deletions and insertions on the graph.
    ///
    /// Time complexity: O(|V|+|E|).
    ///
    /// See also [`Graph::from_edges`] (from an edge list) and
    /// [`Graph::adjacency`] (from an adjacency matrix).
    ///
    /// # Errors
    /// - [`ErrorKind::InvalidValue`] if `duplicate` is set (and `mode` is
    ///   [`NeighborMode::All`]) but the edges are not correctly listed twice:
    ///   `u` must appear in the list of `w` as many times as `w` appears in
    ///   the list of `u`, and loops must be listed an even number of times
    ///   (checked on the Rust side: igraph itself only detects some of these
    ///   cases and may otherwise silently invent or drop edges);
    /// - [`ErrorKind::InvalidVertexId`] if a list contains an id that is not
    ///   one of the list's vertices (checked on the Rust side: igraph itself
    ///   would silently add the missing vertices).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::adjlist::AdjList;
    /// // The undirected triangle, each edge listed from both of its ends.
    /// let al = AdjList::from(vec![vec![1, 2], vec![0, 2], vec![0, 1]]);
    /// let g = Graph::adjlist(&al, NeighborMode::All, true).unwrap();
    /// assert!(!g.is_directed());
    /// assert_eq!(g.ecount(), 3);
    /// // Round trip.
    /// assert_eq!(g.adjlist_init(NeighborMode::All, Loops::Twice, true).unwrap(), al);
    /// ```
    ///
    /// Binds [`igraph_adjlist`](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_adjlist).
    pub fn adjlist(adjlist: &AdjList, mode: NeighborMode, duplicate: bool) -> Result<Graph> {
        // igraph would silently add vertices for too large ids: reject them.
        adjlist.check_ids()?;
        if duplicate && mode == NeighborMode::All {
            adjlist.check_duplicated()?;
        }
        Graph::init_with(|g| unsafe { igraph_adjlist(g, adjlist, mode.into(), duplicate) })
    }

    /// Incidence list of the graph: for each vertex, the ids of its incident
    /// edges (`igraph_inclist_init`).
    ///
    /// The list is independent of the graph after creation.
    ///
    /// - `mode`: in directed graphs, out-edges ([`NeighborMode::Out`]),
    ///   in-edges ([`NeighborMode::In`]) or both ([`NeighborMode::All`]);
    ///   ignored for undirected graphs. With `Out` or `In` each edge id
    ///   appears once in the whole structure, with `All` twice (once per
    ///   endpoint).
    /// - `loops`: [`Loops::None`] drops loop edges, [`Loops::Once`] lists each
    ///   loop edge once for its vertex, [`Loops::Twice`] twice (only for
    ///   undirected graphs or `mode = All`).
    ///
    /// The list of vertex `v` holds the same ids as
    /// [`Graph::incident`]`(v, mode, loops)`. Pair it with
    /// [`Graph::adjlist_init_from_inclist`] to get the matching neighbors.
    ///
    /// Time complexity: O(|V|+|E|).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 2)], 3, false).unwrap();
    /// let il = g.inclist_init(NeighborMode::All, Loops::Once).unwrap();
    /// assert_eq!(il.to_vecs(), vec![vec![0], vec![0, 1], vec![1, 2]]);
    /// ```
    ///
    /// Binds [`igraph_inclist_init`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_inclist_init).
    pub fn inclist_init(&self, mode: NeighborMode, loops: Loops) -> Result<IncList> {
        let mut raw = MaybeUninit::<IncList>::zeroed();
        igraph_call!(igraph_inclist_init(
            self,
            raw.as_mut_ptr(),
            mode.into(),
            loops.into()
        ))?;
        Ok(unsafe { raw.assume_init() })
    }

    /// Lazy adjacency list of the graph (`igraph_lazy_adjlist_init`): the
    /// neighbors of a vertex are computed on its first
    /// [`get`](LazyAdjList::get) and then cached.
    ///
    /// The arguments have the same meaning as in [`Graph::adjlist_init`]; the
    /// lists are sorted. The lazy list borrows the graph, which therefore
    /// cannot be modified while the list is alive. When igraph already knows
    /// (from its property cache, e.g. after [`Graph::has_loop`] or
    /// [`Graph::has_multiple`]) that the graph has no self-loops or no
    /// multi-edges, it skips the corresponding filtering: see
    /// [`LazyAdjList::loops`] and [`LazyAdjList::multiple`]. The lists are
    /// always the same as those of [`Graph::adjlist_init`], whatever the
    /// state of the cache (this wrapper works around an igraph 1.0.0 and
    /// 1.0.1 bug that listed mutual pairs of a directed graph twice in mode
    /// `All` with `multiple = false` once the cache said "no multi-edges").
    ///
    /// Time complexity: O(|V|) for the initialization, O(d) for the first
    /// query of a vertex of degree d, O(1) afterwards.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2)], 3, true).unwrap();
    /// let mut lazy = g.lazy_adjlist_init(NeighborMode::In, Loops::Once, true).unwrap();
    /// assert_eq!(lazy.len(), 3);
    /// assert_eq!(lazy.get(2).unwrap(), &[0, 1]);
    /// assert_eq!(lazy.get(0).unwrap(), &[] as &[i64]);
    /// ```
    ///
    /// Binds [`igraph_lazy_adjlist_init`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_lazy_adjlist_init).
    pub fn lazy_adjlist_init(
        &self,
        mode: NeighborMode,
        loops: Loops,
        multiple: bool,
    ) -> Result<LazyAdjList<'_>> {
        let collapse = collapse_rule(self, mode, loops, multiple);
        let mut raw = MaybeUninit::<igraph_lazy_adjlist_t>::zeroed();
        igraph_call!(igraph_lazy_adjlist_init(
            self,
            raw.as_mut_ptr(),
            mode.into(),
            loops.into(),
            multiple || collapse.is_some()
        ))?;
        Ok(LazyAdjList {
            raw: unsafe { raw.assume_init() },
            collapse,
            _graph: PhantomData,
        })
    }

    /// Lazy incidence list of the graph (`igraph_lazy_inclist_init`): the
    /// incident edges of a vertex are computed on its first
    /// [`get`](LazyIncList::get) and then cached.
    ///
    /// The arguments have the same meaning as in [`Graph::inclist_init`]. The
    /// lazy list borrows the graph.
    ///
    /// Time complexity: O(|V|) for the initialization, O(d) for the first
    /// query of a vertex of degree d, O(1) afterwards.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2)], 3, true).unwrap();
    /// let mut lazy = g.lazy_inclist_init(NeighborMode::Out, Loops::Once).unwrap();
    /// assert_eq!(lazy.get(0).unwrap(), &[0, 1]);
    /// ```
    ///
    /// Binds [`igraph_lazy_inclist_init`](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_lazy_inclist_init).
    pub fn lazy_inclist_init(&self, mode: NeighborMode, loops: Loops) -> Result<LazyIncList<'_>> {
        let mut raw = MaybeUninit::<igraph_lazy_inclist_t>::zeroed();
        igraph_call!(igraph_lazy_inclist_init(
            self,
            raw.as_mut_ptr(),
            mode.into(),
            loops.into()
        ))?;
        Ok(LazyIncList {
            raw: unsafe { raw.assume_init() },
            _graph: PhantomData,
        })
    }
}

/// Lazy adjacency list (`igraph_lazy_adjlist_t`) borrowing a graph: the
/// neighbors of each vertex are queried on first access and cached.
///
/// Create it with [`Graph::lazy_adjlist_init`] or [`LazyAdjList::new`]. Since
/// queries fill the cache, access needs `&mut self`.
pub struct LazyAdjList<'g> {
    raw: igraph_lazy_adjlist_t,
    /// Multi-edge collapsing done on the Rust side (see `collapse_rule`).
    collapse: Option<usize>,
    _graph: PhantomData<&'g Graph>,
}

/// Lazy incidence list (`igraph_lazy_inclist_t`) borrowing a graph: the
/// incident edges of each vertex are queried on first access and cached.
///
/// Create it with [`Graph::lazy_inclist_init`] or [`LazyIncList::new`].
pub struct LazyIncList<'g> {
    raw: igraph_lazy_inclist_t,
    _graph: PhantomData<&'g Graph>,
}

macro_rules! impl_lazy_list {
    (
        $ty:ident, $field:ident, $what:literal, $item:literal, c = $c:literal,
        get_real = $get_real:ident, destroy = $destroy:ident, clear = $clear:ident
    ) => {
        impl<'g> $ty<'g> {
            #[doc = concat!("Number of vertices of the ", $what, ", i.e. of the graph (`", $c, "_size`).")]
            ///
            #[doc = concat!("Binds [`", $c, "_size`](", "https://igraph.org/c/html/latest/igraph-Data-structures.html#", $c, "_size).")]
            pub fn len(&self) -> usize {
                self.raw.length.max(0) as usize
            }

            /// Whether the underlying graph has no vertices.
            pub fn is_empty(&self) -> bool {
                self.len() == 0
            }

            /// The graph this lazy list reads from.
            pub fn graph(&self) -> &'g Graph {
                // The pointer was set from a `&'g Graph` at construction.
                unsafe { &*self.raw.graph }
            }

            /// The effective neighbor mode (always [`NeighborMode::All`] for
            /// undirected graphs).
            pub fn mode(&self) -> NeighborMode {
                NeighborMode::try_from(self.raw.mode).unwrap_or(NeighborMode::All)
            }

            #[doc = concat!("Whether the ", $item, " of vertex `v` were already computed and cached.")]
            ///
            /// Returns `false` for out-of-range vertices. Time complexity: O(1).
            ///
            #[doc = concat!("Binds the [`", $c, "_has`](", "https://igraph.org/c/html/latest/igraph-Data-structures.html#", $c, "_has) macro.")]
            pub fn has(&self, v: VertexId) -> bool {
                match check_vertex(v, self.len()) {
                    Ok(i) => unsafe { !(*self.raw.$field.add(i)).is_null() },
                    Err(_) => false,
                }
            }

            #[doc = concat!("The ", $item, " of vertex `v`, computed on the first call and cached afterwards.")]
            ///
            /// The returned slice borrows the lazy list mutably, so copy it
            /// (`.to_vec()`) if you need to query other vertices while using
            /// it. Time complexity: O(d) on the first call for a vertex of
            /// degree d, O(1) afterwards.
            ///
            #[doc = concat!("Binds the [`", $c, "_get`](", "https://igraph.org/c/html/latest/igraph-Data-structures.html#", $c, "_get) macro.")]
            ///
            /// # Errors
            /// [`ErrorKind::InvalidVertexId`] if `v` is out of range, or an
            /// igraph error if the list cannot be computed (out of memory).
            pub fn get(&mut self, v: VertexId) -> Result<&[igraph_int_t]> {
                Ok(self.get_mut(v)?.as_slice())
            }

            #[doc = concat!("The ", $item, " of vertex `v` as a mutable [`VectorInt`], computed on first access.")]
            ///
            /// Modifying it changes only the cached copy, never the graph.
            ///
            /// # Errors
            /// As [`get`](Self::get).
            pub fn get_mut(&mut self, v: VertexId) -> Result<&mut VectorInt> {
                let i = check_vertex(v, self.len())?;
                ensure_init();
                let cached = unsafe { *self.raw.$field.add(i) };
                if !cached.is_null() {
                    return Ok(unsafe { &mut *cached });
                }
                let ptr = unsafe { $get_real(&mut self.raw, v) };
                if ptr.is_null() {
                    return Err(lazy_failure());
                }
                // The vector is owned by the lazy list, which we borrow mutably.
                let list = unsafe { &mut *ptr };
                self.after_compute(v, list);
                Ok(list)
            }

            /// Computes (if needed) all the lists and copies them into nested
            /// Rust vectors.
            ///
            /// # Errors
            /// As [`get`](Self::get).
            pub fn to_vecs(&mut self) -> Result<Vec<Vec<igraph_int_t>>> {
                (0..self.len() as VertexId).map(|v| self.get(v).map(<[igraph_int_t]>::to_vec)).collect()
            }

            #[doc = concat!("Forgets all the cached lists (`", stringify!($clear), "`); they will be recomputed on demand.")]
            ///
            /// Any edits made through [`get_mut`](Self::get_mut) are lost.
            ///
            #[doc = concat!("Binds [`", stringify!($clear), "`](", "https://igraph.org/c/html/latest/igraph-Data-structures.html#", stringify!($clear), ").")]
            pub fn clear(&mut self) {
                if !self.raw.$field.is_null() {
                    unsafe { $clear(&mut self.raw) };
                }
            }
        }

        impl Drop for $ty<'_> {
            #[doc = concat!("Frees the cached lists with `", stringify!($destroy), "`.")]
            fn drop(&mut self) {
                if !self.raw.$field.is_null() {
                    unsafe { $destroy(&mut self.raw) };
                }
            }
        }

        impl fmt::Debug for $ty<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let cached = (0..self.len() as VertexId).filter(|&v| self.has(v)).count();
                f.debug_struct(stringify!($ty))
                    .field("len", &self.len())
                    .field("mode", &self.mode())
                    .field("cached", &cached)
                    .finish()
            }
        }
    };
}

impl_lazy_list!(
    LazyAdjList,
    adjs,
    "lazy adjacency list",
    "neighbors",
    c = "igraph_lazy_adjlist",
    get_real = igraph_i_lazy_adjlist_get_real,
    destroy = igraph_lazy_adjlist_destroy,
    clear = igraph_lazy_adjlist_clear
);
impl_lazy_list!(
    LazyIncList,
    incs,
    "lazy incidence list",
    "incident edges",
    c = "igraph_lazy_inclist",
    get_real = igraph_i_lazy_inclist_get_real,
    destroy = igraph_lazy_inclist_destroy,
    clear = igraph_lazy_inclist_clear
);

impl<'g> LazyAdjList<'g> {
    /// Post-processes a freshly computed list (see `collapse_rule`).
    fn after_compute(&self, v: VertexId, list: &mut VectorInt) {
        if let Some(max_loops) = self.collapse {
            collapse_sorted(list, v, max_loops);
        }
    }

    /// Lazy adjacency list of `graph`: same as [`Graph::lazy_adjlist_init`].
    pub fn new(graph: &'g Graph, mode: NeighborMode, loops: Loops, multiple: bool) -> Result<Self> {
        graph.lazy_adjlist_init(mode, loops, multiple)
    }

    /// The effective loop handling.
    ///
    /// This is the `loops` argument given at creation, unless igraph already
    /// knew that the graph has no self-loops: then it reports
    /// [`Loops::Twice`] (mode `All`) or [`Loops::Once`] (mode `In`/`Out`),
    /// which avoids a useless filtering pass and yields the same lists.
    pub fn loops(&self) -> Loops {
        Loops::try_from(self.raw.loops).unwrap_or(Loops::Twice)
    }

    /// Whether multi-edges are kept.
    ///
    /// This is the `multiple` argument given at creation, except for an
    /// `Out` or `In` list of a directed graph that igraph already knew to
    /// have no multi-edges: then it is `true`, since there is nothing to
    /// collapse. (Lists gathering neighbors in both directions are always
    /// collapsed as requested; see [`Graph::adjlist_init`] for the igraph
    /// 1.0.0 and 1.0.1 bug this works around.)
    pub fn multiple(&self) -> bool {
        self.collapse.is_none() && self.raw.multiple
    }
}

impl<'g> LazyIncList<'g> {
    /// Incidence lists need no post-processing.
    fn after_compute(&self, _v: VertexId, _list: &mut VectorInt) {}

    /// Lazy incidence list of `graph`: same as [`Graph::lazy_inclist_init`].
    pub fn new(graph: &'g Graph, mode: NeighborMode, loops: Loops) -> Result<Self> {
        graph.lazy_inclist_init(mode, loops)
    }

    /// The loop handling of this list, as given at creation.
    pub fn loops(&self) -> Loops {
        Loops::try_from(self.raw.loops).unwrap_or(Loops::Twice)
    }
}

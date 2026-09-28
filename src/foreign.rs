//! Reading and writing graphs in foreign file formats (`igraph_foreign.h`).
//!
//! igraph can exchange graphs with other software through a number of
//! textual (and one binary) file formats. This module wraps all of them, in
//! two flavours:
//!
//! - **file based**: `Graph::read_graph_*(path, ...)` and
//!   `graph.write_graph_*(path, ...)` take anything implementing
//!   [`AsRef<Path>`](std::path::Path) and open/close the file themselves
//!   (the file handle is always closed, also on errors, and write errors
//!   detected when flushing are reported);
//! - **in memory**: `Graph::read_graph_*_from_str(&str, ...)` parses a string
//!   and `graph.write_graph_*_to_string(...)` returns the serialization as a
//!   [`String`], without touching the file system (they use the POSIX
//!   `fmemopen`/`open_memstream` streams under the hood). igraph copies
//!   string attributes byte by byte, so a `_to_string` writer replaces
//!   invalid UTF-8 (only possible in attributes read from files in another
//!   encoding, e.g. Latin-1 Pajek labels) with U+FFFD: use the file based
//!   writer to keep the exact bytes.
//!
//! On top of that, [`GraphFormat`] together with [`Graph::read_graph`] and
//! [`Graph::write_graph`] offers a format-agnostic entry point, able to guess
//! the format from the file extension ([`GraphFormat::from_path`]).
//!
//! # Example
//!
//! ```
//! use igraph::{foreign::GmlWriteOptions, prelude::*};
//!
//! // A directed triangle with a pendant vertex, serialized as an edge list...
//! let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, true).unwrap();
//! let text = g.write_graph_edgelist_to_string().unwrap();
//! assert_eq!(text, "0 1\n1 2\n2 0\n2 3\n");
//!
//! // ...and parsed back: exactly the same graph.
//! let h = Graph::read_graph_edgelist_from_str(&text, 0, true).unwrap();
//! assert!(g.is_same_graph(&h).unwrap());
//!
//! // GML round trip of Zachary's karate club, through a temporary file.
//! let karate = Graph::famous("Zachary").unwrap();
//! let path = std::env::temp_dir().join(format!("igraph-doc-foreign-{}.gml", std::process::id()));
//! karate.write_graph_gml(&path, &GmlWriteOptions::default()).unwrap();
//! let k = Graph::read_graph_gml(&path).unwrap();
//! std::fs::remove_file(&path).unwrap();
//! assert!(!k.is_directed());
//! assert_eq!(k, karate); // GML keeps vertex ids and edges
//! ```
//!
//! # Supported formats
//!
//! | Format | Read | Write | Attributes (after [`attributes::enable`](crate::attributes::enable)) | Notes |
//! |---|---|---|---|---|
//! | Edge list | [`read_graph_edgelist`](Graph::read_graph_edgelist) | [`write_graph_edgelist`](Graph::write_graph_edgelist) | none | whitespace separated 0-based vertex ids |
//! | NCOL | [`read_graph_ncol`](Graph::read_graph_ncol) | [`write_graph_ncol`](Graph::write_graph_ncol) | vertex `name`, edge `weight` | symbolic (named) weighted edge list of the LGL software |
//! | LGL | [`read_graph_lgl`](Graph::read_graph_lgl) | [`write_graph_lgl`](Graph::write_graph_lgl) | vertex `name`, edge `weight` | adjacency-list-like, `# vertex` headers |
//! | Pajek | [`read_graph_pajek`](Graph::read_graph_pajek) | [`write_graph_pajek`](Graph::write_graph_pajek) | `name`, `x`/`y`/`z`, `color`, bipartite `type`, edge `weight`, ... | `.net` files, 1-based ids |
//! | GraphML | [`read_graph_graphml`](Graph::read_graph_graphml) | [`write_graph_graphml`](Graph::write_graph_graphml), [`write_graph_graphml_with`](Graph::write_graph_graphml_with) | all (typed, with defaults), node ids as vertex `id` | XML; reading requires igraph built with libxml2; the in-memory writer [`write_graph_graphml_to_string`](Graph::write_graph_graphml_to_string) takes the `prefixattr` flag of `write_graph_graphml_with` |
//! | GML | [`read_graph_gml`](Graph::read_graph_gml) | [`write_graph_gml`](Graph::write_graph_gml) | numeric and string ones, numeric vertex `id` | see [`GmlWriteOptions`] |
//! | DIMACS flow | [`read_graph_dimacs_flow`](Graph::read_graph_dimacs_flow) | [`write_graph_dimacs_flow`](Graph::write_graph_dimacs_flow) | none (capacities in [`DimacsFlow`]) | max-flow / edge problems |
//! | graph database | [`read_graph_graphdb`](Graph::read_graph_graphdb) | — | none | binary, see [`read_graph_graphdb_from_bytes`](Graph::read_graph_graphdb_from_bytes) |
//! | UCINET DL | [`read_graph_dl`](Graph::read_graph_dl) | — | vertex `name`, edge `weight` | full matrix, edge list and node list forms |
//! | Graphviz DOT | — | [`write_graph_dot`](Graph::write_graph_dot) | all | output only |
//! | LEDA | — | [`write_graph_leda`](Graph::write_graph_leda) | one vertex and one edge attribute | output only |
//!
//! Every file based function has an in-memory twin with the `_from_str` /
//! `_to_string` suffix (`_from_bytes` for the binary graph database format),
//! taking the same arguments but the path. The one exception is GraphML:
//! [`write_graph_graphml_to_string(prefixattr)`](Graph::write_graph_graphml_to_string)
//! is the twin of
//! [`write_graph_graphml_with(path, prefixattr)`](Graph::write_graph_graphml_with)
//! (pass `false` for the behaviour of
//! [`write_graph_graphml(path)`](Graph::write_graph_graphml)).
//! [`SafeLocale`] and [`with_safe_locale`] bind igraph's locale helpers.
//!
//! # Attributes
//!
//! Many formats carry vertex names, edge weights and other attributes. igraph
//! stores them through a pluggable, **process-wide** *attribute handler*,
//! which is off by default and is turned on by
//! [`attributes::enable`](crate::attributes::enable) (see the
//! [`attributes`](crate::attributes) module for the typed accessors):
//!
//! - **without** the handler, readers parse attributes (validating their
//!   syntax) but discard them: only the structure of the graph is returned.
//!   The options asking to store names or weights ([`NcolLglOptions`]) are
//!   accepted and harmless. Writers see no attribute: those asked to export a
//!   named one (e.g. the `names`/`weights` arguments of
//!   [`write_graph_ncol`](Graph::write_graph_ncol)) emit an igraph *warning*
//!   (see [`take_warnings`](crate::error::take_warnings)) and fall back to
//!   plain vertex ids, and GraphML/GML/DOT files are written without
//!   attributes;
//! - **with** the handler (call [`enable`](crate::attributes::enable)
//!   *before* reading), readers store what they find in the file as graph,
//!   vertex and edge attributes (the "Attributes" column above), and the
//!   writers export the attributes of the graph. Values missing for some
//!   elements are NaN for numeric attributes, `""` for strings and `false`
//!   for booleans, unless the format has its own defaults (a missing
//!   NCOL/LGL weight is 1, a GraphML `<key>` may declare a `<default>`,
//!   Pajek parameters take Pajek's defaults, see the readers); GraphML and
//!   GML omit NaN values when writing.
//!
//! What each format keeps, with the handler on (the individual readers and
//! writers have the details):
//!
//! | Format | Read into attributes | Written from attributes |
//! |---|---|---|
//! | Edge list, DIMACS, graph database | nothing | nothing |
//! | NCOL, LGL | vertex `name` (string), edge `weight` (numeric), as selected by [`NcolLglOptions`] | the vertex / edge attributes named by the `names` / `weights` arguments |
//! | UCINET DL | vertex `name` (labels), edge `weight` (values) | — |
//! | Pajek | vertex `name`, `x`/`y`/`z`, `color`, shapes and the other Pajek parameters, bipartite `type` (boolean), edge `weight` and edge parameters | the attributes named after Pajek parameters; other attributes are ignored |
//! | GraphML | every `<key>`, typed (boolean, numeric, string), for graph, vertices and edges; the node `id`s as the string vertex attribute `id` | every graph, vertex and edge attribute (typed keys), optionally prefixed (`g_`/`v_`/`e_`, [`write_graph_graphml_with`](Graph::write_graph_graphml_with)) |
//! | GML | every numeric or string field of the graph, node and edge records, including the numeric node `id` | numeric and string attributes (booleans as 0/1); a numeric vertex `id` supplies the node ids |
//! | DOT | — | every graph, vertex and edge attribute |
//! | LEDA | — | one vertex and one edge attribute, named in the call |
//!
//! Two GraphML caveats: the node ids land in a string `id` vertex
//! attribute (unless a vertex key already defines an attribute named `id`), which the
//! GraphML writer then exports as an ordinary `<key>`, so a second round
//! trip carries it as data; and when the `<edge>` elements have `id`s, igraph
//! 1.0.0 and 1.0.1 also create a string `id` *edge* attribute but, because
//! of a bug in `src/io/graphml.c`, fill it with the *node* ids (in node
//! order, padded with `""` if there are more edges than nodes): don't rely on it
//! (delete it with
//! [`remove_edge_attr`](Graph::remove_edge_attr) if it gets in the way). See
//! [`read_graph_graphml`](Graph::read_graph_graphml).
//!
//! ```
//! use igraph::{attributes, foreign::NcolLglOptions, prelude::*};
//!
//! attributes::enable().unwrap(); // before reading!
//! let text = "rome paris 1420\nparis london 460\nlondon rome\n";
//! let g = Graph::read_graph_ncol_from_str(text, &[], &NcolLglOptions::default()).unwrap();
//! assert_eq!(g.vertex_attr_str_values("name", ..).unwrap(), ["rome", "paris", "london"]);
//! // A missing weight defaults to 1 when at least one edge has an explicit weight.
//! assert_eq!(g.edge_attr_numeric_values("weight", ..).unwrap(), [1420.0, 460.0, 1.0]);
//! // Writers use the attributes they are asked for.
//! let out = g.write_graph_ncol_to_string(Some("name"), Some("weight")).unwrap();
//! assert_eq!(out, "rome paris 1420\nparis london 460\nrome london 1\n");
//! ```
//!
//! Data that igraph returns through explicit output arguments does not need
//! the handler, e.g. the capacities and source/target vertices of a DIMACS
//! file, returned in [`DimacsFlow`]. When only the vertex naming of an NCOL
//! file matters, the `predefnames` argument of
//! [`read_graph_ncol`](Graph::read_graph_ncol) fixes it without attributes:
//! vertex `i` then is the `i`-th predefined name.
//!
//! # Locale and threads
//!
//! The parsers and writers assume that the C locale uses a decimal *point*.
//! Rust programs start in the `"C"` locale, so nothing needs to be done unless
//! some library called `setlocale`; in that case wrap the I/O with
//! [`SafeLocale`] or [`with_safe_locale`].
//!
//! All the functions can be called from several threads at once (but see
//! [`SafeLocale`] for platforms without per-thread locales). The GML
//! reader of igraph 1.0.0 and 1.0.1 is not reentrant (it uses static
//! buffers), and so is the default GML `Creator` line (it uses `ctime`): these
//! wrappers serialize them internally with a process-wide lock.
//!
//! # See also
//!
//! - [`Graph::from_edges`] and [`Graph::famous`] (in
//!   [`constructors`](crate::constructors)) to build graphs in code;
//! - [`Graph::get_adjacency`] and the rest of the
//!   [`conversion`](crate::conversion) module to export graphs as matrices;
//! - [`Graph::is_same_graph`] to check that a round trip kept the vertex ids,
//!   [`Graph::isomorphic`] when a format (NCOL, LGL, bipartite Pajek)
//!   relabels the vertices;
//! - [`Graph::maxflow`] and [`Graph::st_mincut`] to solve the problems read
//!   from DIMACS files.
//!
//! The C documentation of the whole chapter is at
//! <https://igraph.org/c/html/latest/igraph-Foreign.html>.

use crate::{
    constants::AddWeights,
    error::{Error, ErrorKind, Result},
    ffi::*,
    graph::{Graph, VertexId},
    igraph_call,
    strvector::StrVector,
    vector::{Vector, VectorInt},
};
use std::{
    ffi::{CStr, CString, c_char, c_void},
    marker::PhantomData,
    path::Path,
    sync::Mutex,
};

/// Serializes the non-reentrant parts of igraph's GML code (still present in
/// igraph 1.0.0 and 1.0.1):
///
/// - `igraph_read_graph_gml` keeps intermediate strings in function-local
///   `static` buffers (`strid` and `igraph_i_gml_tostring` in
///   `src/io/gml.c`), so concurrent calls from different threads corrupt each
///   other (and may crash);
/// - `igraph_write_graph_gml` with a `NULL` creator formats the current time
///   with `ctime`, which returns (and igraph then modifies in place) a
///   process-wide static buffer.
///
/// All the other readers and writers are reentrant.
static GML_LOCK: Mutex<()> = Mutex::new(());

/// Acquires [`GML_LOCK`].
fn gml_lock() -> std::sync::MutexGuard<'static, ()> {
    // A poisoned lock only means another thread panicked: the C state is fine.
    GML_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Runs the GML reader while holding [`GML_LOCK`].
fn read_gml(source: Source<'_>) -> Result<Graph> {
    let _guard = gml_lock();
    read_with(source, |g, f| unsafe { igraph_read_graph_gml(g, f) })
}

/// Converts a Rust count/index into an `igraph_int_t`, rejecting values that
/// would wrap around to negative numbers.
fn to_int(value: usize, what: &str) -> Result<igraph_int_t> {
    igraph_int_t::try_from(value)
        .map_err(|_| Error::invalid(format!("{what} {value} is too large")))
}

/// Encodes `&` (unless `only_quot`) and `"` as XML entities, as igraph does
/// for GML string values.
fn gml_entity_encode(s: &str, only_quot: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' if !only_quot => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            c => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// C stream plumbing
// ---------------------------------------------------------------------------

/// Converts a path into a NUL terminated C string.
fn path_to_cstring(path: &Path) -> Result<CString> {
    #[cfg(unix)]
    let bytes = {
        use std::os::unix::ffi::OsStrExt;
        path.as_os_str().as_bytes().to_vec()
    };
    #[cfg(not(unix))]
    let bytes = path
        .to_str()
        .ok_or_else(|| Error::invalid(format!("path {} is not valid UTF-8", path.display())))?
        .as_bytes()
        .to_vec();
    CString::new(bytes).map_err(|_| {
        Error::invalid(format!(
            "path {} contains an interior NUL byte",
            path.display()
        ))
    })
}

/// Converts an optional attribute name into an optional C string.
fn opt_cstring(name: Option<&str>, what: &str) -> Result<Option<CString>> {
    name.map(|s| {
        CString::new(s).map_err(|_| Error::invalid(format!("{what} contains an interior NUL byte")))
    })
    .transpose()
}

/// Pointer of an optional C string (`NULL` for `None`).
fn opt_ptr(s: &Option<CString>) -> *const c_char {
    s.as_ref().map_or(std::ptr::null(), |c| c.as_ptr())
}

/// An owned C `FILE *`, closed on drop. The lifetime ties it to the memory
/// buffer it may read from (for `fmemopen` streams).
struct CFile<'a> {
    ptr: *mut FILE,
    _buffer: PhantomData<&'a [u8]>,
}

impl<'a> CFile<'a> {
    /// Opens a file with `fopen`.
    fn open(path: &Path, mode: &CStr) -> Result<CFile<'static>> {
        let c_path = path_to_cstring(path)?;
        let ptr = unsafe { fopen(c_path.as_ptr(), mode.as_ptr()) };
        if ptr.is_null() {
            let os = std::io::Error::last_os_error();
            return Err(Error::new(
                ErrorKind::File,
                format!("cannot open {}: {os}", path.display()),
            ));
        }
        Ok(CFile {
            ptr,
            _buffer: PhantomData,
        })
    }

    /// Opens a read-only stream over an in-memory buffer (`fmemopen`).
    fn from_bytes(data: &'a [u8]) -> Result<CFile<'a>> {
        let ptr = if data.is_empty() {
            // `fmemopen` may reject zero-sized buffers: an empty temporary
            // file is an equivalent empty stream.
            unsafe { tmpfile() }
        } else {
            // In read mode the buffer is never written to.
            unsafe { fmemopen(data.as_ptr() as *mut c_void, data.len(), c"rb".as_ptr()) }
        };
        if ptr.is_null() {
            let os = std::io::Error::last_os_error();
            return Err(Error::new(
                ErrorKind::File,
                format!("cannot open a memory stream: {os}"),
            ));
        }
        Ok(CFile {
            ptr,
            _buffer: PhantomData,
        })
    }

    /// Closes the stream with `fclose`, reporting failures: earlier write
    /// errors that igraph did not check (its error indicator is set) and
    /// buffered writes that could not be completed.
    fn close(mut self) -> Result<()> {
        let ptr = std::mem::replace(&mut self.ptr, std::ptr::null_mut());
        let had_error = unsafe { ferror(ptr) } != 0;
        if unsafe { fclose(ptr) } != 0 || had_error {
            let os = std::io::Error::last_os_error();
            return Err(Error::new(
                ErrorKind::File,
                format!("cannot close the stream: {os}"),
            ));
        }
        Ok(())
    }
}

impl Drop for CFile<'_> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { fclose(self.ptr) };
        }
    }
}

/// A growable in-memory output stream (`open_memstream`).
struct MemSink {
    file: *mut FILE,
    /// Heap cell (from `Box::into_raw`) whose two fields `open_memstream`
    /// updates with the buffer address and size. It is only accessed through
    /// this raw pointer, so the pointers held by the C library stay valid
    /// until `Drop` reclaims it.
    loc: *mut (*mut c_char, usize),
}

impl MemSink {
    fn new() -> Result<Self> {
        let loc = Box::into_raw(Box::new((std::ptr::null_mut::<c_char>(), 0usize)));
        // SAFETY: `loc` is a valid, uniquely owned allocation.
        let file = unsafe { open_memstream(&raw mut (*loc).0, &raw mut (*loc).1) };
        if file.is_null() {
            let os = std::io::Error::last_os_error();
            // SAFETY: the stream was not created, nobody else points to `loc`.
            drop(unsafe { Box::from_raw(loc) });
            return Err(Error::new(
                ErrorKind::File,
                format!("cannot open a memory stream: {os}"),
            ));
        }
        Ok(Self { file, loc })
    }

    /// Closes the stream and returns what was written.
    ///
    /// Bytes that are not valid UTF-8 (possible only in string attributes
    /// that igraph read from non-UTF-8 files) are replaced by U+FFFD.
    fn finish(mut self) -> Result<String> {
        let file = std::mem::replace(&mut self.file, std::ptr::null_mut());
        let had_error = unsafe { ferror(file) } != 0;
        if unsafe { fclose(file) } != 0 || had_error {
            return Err(Error::new(
                ErrorKind::File,
                "cannot finalize the memory stream",
            ));
        }
        // SAFETY: `fclose` stored the final buffer address and size.
        let (buf, size) = unsafe { *self.loc };
        if buf.is_null() {
            return Ok(String::new());
        }
        let bytes = unsafe { std::slice::from_raw_parts(buf as *const u8, size) };
        Ok(String::from_utf8_lossy(bytes).into_owned())
        // `self` is dropped here, freeing the buffer.
    }
}

impl Drop for MemSink {
    fn drop(&mut self) {
        if !self.file.is_null() {
            unsafe { fclose(self.file) };
        }
        // SAFETY: the stream is closed, so the C library no longer uses
        // `loc`; the buffer it holds was allocated by the C library and must
        // be freed with `free` (`free(NULL)` is a no-op).
        let loc = unsafe { Box::from_raw(self.loc) };
        unsafe { free(loc.0 as *mut c_void) };
    }
}

unsafe extern "C" {
    /// `ferror` from `<stdio.h>` (not in the generated bindings): whether
    /// the error indicator of the stream is set.
    fn ferror(stream: *mut FILE) -> std::ffi::c_int;
}

/// Where a reader takes its input from.
enum Source<'a> {
    Path(&'a Path),
    Bytes(&'a [u8]),
}

impl<'a> Source<'a> {
    fn open(&self) -> Result<CFile<'a>> {
        match *self {
            Source::Path(p) => CFile::open(p, c"rb"),
            Source::Bytes(b) => CFile::from_bytes(b),
        }
    }
}

/// Runs a reader (a C function initializing a graph from a stream).
fn read_with(
    source: Source<'_>,
    f: impl FnOnce(*mut igraph_t, *mut FILE) -> igraph_error_t,
) -> Result<Graph> {
    let file = source.open()?;
    let graph = Graph::init_with(|g| f(g, file.ptr))?;
    // Closing a read stream can't lose data: ignore its outcome.
    let _ = file.close();
    Ok(graph)
}

/// Runs a writer on a newly created (truncated) file.
fn write_to_path(path: &Path, f: impl FnOnce(*mut FILE) -> igraph_error_t) -> Result<()> {
    let file = CFile::open(path, c"wb")?;
    igraph_call!(f(file.ptr))?;
    file.close()
}

/// Runs a writer on an in-memory stream and returns the output.
fn write_to_string(f: impl FnOnce(*mut FILE) -> igraph_error_t) -> Result<String> {
    let sink = MemSink::new()?;
    igraph_call!(f(sink.file))?;
    sink.finish()
}

// ---------------------------------------------------------------------------
// Options and results
// ---------------------------------------------------------------------------

/// Options of the NCOL and LGL readers ([`Graph::read_graph_ncol`],
/// [`Graph::read_graph_lgl`]).
///
/// The defaults are: store names, add weights only if present in the file,
/// undirected graph (the LGL software only handles undirected graphs).
///
/// `names` and `weights` only matter once the attribute handler is enabled
/// with [`attributes::enable`](crate::attributes::enable) (see the
/// [module docs](self#attributes)); without it they are harmless.
///
/// # Examples
/// ```
/// use igraph::{foreign::NcolLglOptions, prelude::*};
/// let opts = NcolLglOptions::default().with_directed(true).with_weights(AddWeights::No);
/// assert!(opts.names && opts.directed);
/// let g = Graph::read_graph_ncol_from_str("a b 2\nb c 3\n", &[], &opts).unwrap();
/// assert!(g.is_directed());
/// assert_eq!(g.edge_list(), vec![(0, 1), (1, 2)]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NcolLglOptions {
    /// Whether to store the symbolic vertex names as the `name` string vertex
    /// attribute.
    pub names: bool,
    /// Whether to store the edge weights as the `weight` numeric edge
    /// attribute: [`AddWeights::Yes`] always (edges without a weight get 1),
    /// [`AddWeights::IfPresent`] only if at least one weight is given in the
    /// file (again with 1 for the others), [`AddWeights::No`] never.
    pub weights: AddWeights,
    /// Whether to create a directed graph: the formats carry no information
    /// about directedness.
    pub directed: bool,
}

impl Default for NcolLglOptions {
    fn default() -> Self {
        Self {
            names: true,
            weights: AddWeights::IfPresent,
            directed: false,
        }
    }
}

impl NcolLglOptions {
    /// Sets [`directed`](Self::directed).
    pub fn with_directed(mut self, directed: bool) -> Self {
        self.directed = directed;
        self
    }

    /// Sets [`names`](Self::names).
    pub fn with_names(mut self, names: bool) -> Self {
        self.names = names;
        self
    }

    /// Sets [`weights`](Self::weights).
    pub fn with_weights(mut self, weights: AddWeights) -> Self {
        self.weights = weights;
        self
    }
}

/// Options of the GML writer ([`Graph::write_graph_gml`]).
///
/// The default writes igraph's own vertex ids (or the numeric `id` vertex
/// attribute, if present), encodes all special characters as entities, and
/// writes a `Creator` line mentioning the igraph version and the current date
/// and time (use [`with_creator`](Self::with_creator)`("")` for reproducible
/// output).
///
/// # Examples
/// ```
/// use igraph::{foreign::GmlWriteOptions, prelude::*};
/// let g = Graph::from_edges(&[(0, 1)], 2, true).unwrap();
/// let opts = GmlWriteOptions::default().with_creator("").with_ids(&[7.0, 9.0]);
/// let gml = g.write_graph_gml_to_string(&opts).unwrap();
/// let lines: Vec<&str> = gml.lines().map(str::trim).collect();
/// assert_eq!(
///     lines,
///     [
///         "Version 1", "graph", "[", "directed 1",
///         "node", "[", "id 7", "]",
///         "node", "[", "id 9", "]",
///         "edge", "[", "source 7", "target 9", "]",
///         "]",
///     ]
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GmlWriteOptions<'a> {
    /// Encode only `"` characters as entities, nothing else
    /// (`IGRAPH_WRITE_GML_ENCODE_ONLY_QUOT_SW`); useful to re-export files
    /// whose entities igraph passed through undecoded.
    pub encode_only_quot: bool,
    /// Numeric vertex ids to write in the `id` fields instead of igraph's
    /// vertex ids (one per vertex). With `None`, a numeric `id` vertex
    /// attribute is used if there is one (the GML reader creates it when the
    /// [attribute handler](crate::attributes::enable) is on, so a GML round
    /// trip keeps the original ids), otherwise igraph's vertex ids. If some
    /// value is not an integer (or is NaN/infinite), or some value is
    /// repeated, igraph emits a warning (see
    /// [`take_warnings`](crate::error::take_warnings)) and ignores all of
    /// them, writing its own vertex ids instead.
    pub ids: Option<&'a [f64]>,
    /// Text of the `Creator` line: `None` writes the igraph version plus the
    /// current date and time, `Some("")` omits the line, anything else is
    /// written as a GML string, with `"` (and `&`, unless
    /// [`encode_only_quot`](Self::encode_only_quot)) encoded as XML entities.
    pub creator: Option<&'a str>,
}

impl<'a> GmlWriteOptions<'a> {
    /// Sets [`encode_only_quot`](Self::encode_only_quot).
    pub fn with_encode_only_quot(mut self, only_quot: bool) -> Self {
        self.encode_only_quot = only_quot;
        self
    }

    /// Sets [`ids`](Self::ids).
    pub fn with_ids(mut self, ids: &'a [f64]) -> Self {
        self.ids = Some(ids);
        self
    }

    /// Sets [`creator`](Self::creator).
    pub fn with_creator(mut self, creator: &'a str) -> Self {
        self.creator = Some(creator);
        self
    }
}

/// The problem described by a DIMACS file, see [`DimacsFlow`].
#[derive(Debug, Clone, PartialEq)]
pub enum DimacsProblem {
    /// A maximum flow problem (`p max`).
    Max {
        /// The source vertex (0-based igraph id; DIMACS ids start from 1),
        /// `None` if the file has no `n <id> s` line.
        source: Option<VertexId>,
        /// The target vertex (0-based igraph id), `None` if the file has no
        /// `n <id> t` line.
        target: Option<VertexId>,
        /// The capacity of each edge, in edge id order.
        capacity: Vec<f64>,
    },
    /// An "edge" problem (`p edge`), e.g. a graph coloring instance (see
    /// [`Graph::vertex_coloring_greedy`] to solve one).
    Edge {
        /// The integer label of each vertex: its 1-based DIMACS index (igraph
        /// 1.0.0 and 1.0.1 cannot parse the `n` lines that would change it).
        labels: Vec<i64>,
    },
}

/// The content of a DIMACS flow file, as returned by
/// [`Graph::read_graph_dimacs_flow`].
///
/// For a [`DimacsProblem::Max`] instance, pass the capacities to
/// [`Graph::maxflow`], [`Graph::maxflow_value`] or [`Graph::st_mincut`] to
/// solve it.
#[derive(Debug, Clone, PartialEq)]
pub struct DimacsFlow {
    /// The graph.
    pub graph: Graph,
    /// The problem type string of the `p` line (`"max"` or `"edge"`).
    pub problem_name: String,
    /// The problem data.
    pub problem: DimacsProblem,
}

/// Graph file formats known to igraph, for the format-agnostic
/// [`Graph::read_graph`] and [`Graph::write_graph`].
///
/// The variants list the file extensions recognized by
/// [`from_path`](Self::from_path); [`can_read`](Self::can_read) and
/// [`can_write`](Self::can_write) tell which direction the generic functions
/// support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphFormat {
    /// Plain edge list of 0-based vertex ids (`.txt`, `.edgelist`, `.edges`, `.el`).
    Edgelist,
    /// NCOL symbolic edge list (`.ncol`).
    Ncol,
    /// LGL format (`.lgl`).
    Lgl,
    /// Pajek (`.net`, `.pajek`).
    Pajek,
    /// GraphML (`.graphml`, `.xml`).
    GraphMl,
    /// GML (`.gml`).
    Gml,
    /// DIMACS flow format (`.dimacs`, `.max`); only [`Graph::read_graph`]
    /// supports it generically (it discards the problem data).
    DimacsFlow,
    /// Binary ARG graph database format (`.graphdb`, and the `.A00`...`.A99`,
    /// `.B00`...`.B99` suffixes of the database's graph pairs); read only.
    GraphDb,
    /// UCINET DL (`.dl`); read only.
    Dl,
    /// Graphviz DOT (`.dot`, `.gv`); write only.
    Dot,
    /// LEDA native format (`.gw`, `.leda`); write only.
    Leda,
}

impl GraphFormat {
    /// Guesses the format from the extension of `path` (case insensitive),
    /// or `None` if the extension is unknown.
    ///
    /// ```
    /// use igraph::foreign::GraphFormat;
    /// assert_eq!(GraphFormat::from_path("karate.GML"), Some(GraphFormat::Gml));
    /// assert_eq!(GraphFormat::from_path("/tmp/g.net"), Some(GraphFormat::Pajek));
    /// assert_eq!(GraphFormat::from_path("noext"), None);
    /// ```
    pub fn from_path(path: impl AsRef<Path>) -> Option<Self> {
        let ext = path.as_ref().extension()?.to_str()?.to_ascii_lowercase();
        Some(match ext.as_str() {
            "txt" | "edgelist" | "edges" | "el" => Self::Edgelist,
            "ncol" => Self::Ncol,
            "lgl" => Self::Lgl,
            "net" | "pajek" => Self::Pajek,
            "graphml" | "xml" => Self::GraphMl,
            "gml" => Self::Gml,
            "dimacs" | "max" => Self::DimacsFlow,
            "graphdb" => Self::GraphDb,
            e if e.len() == 3
                && (e.starts_with('a') || e.starts_with('b'))
                && e[1..].chars().all(|c| c.is_ascii_digit()) =>
            {
                Self::GraphDb
            }
            "dl" => Self::Dl,
            "dot" | "gv" => Self::Dot,
            "gw" | "leda" => Self::Leda,
            _ => return None,
        })
    }

    /// Whether igraph can read this format.
    pub fn can_read(self) -> bool {
        !matches!(self, Self::Dot | Self::Leda)
    }

    /// Whether igraph can write this format (DIMACS needs extra data: use
    /// [`Graph::write_graph_dimacs_flow`] directly).
    pub fn can_write(self) -> bool {
        !matches!(self, Self::GraphDb | Self::Dl | Self::DimacsFlow)
    }
}

// ---------------------------------------------------------------------------
// Readers
// ---------------------------------------------------------------------------

impl igraph_t {
    /// Reads a graph from a file in the given `format`, with default options.
    ///
    /// `directed` is used by the formats that don't encode directedness
    /// (edge list, NCOL, LGL, DIMACS, graph database, DL); Pajek, GraphML and
    /// GML files specify it themselves. Use [`GraphFormat::from_path`] to
    /// guess the format from the extension. The defaults are: no extra
    /// isolated vertices for edge lists, [`NcolLglOptions::default`] (plus
    /// `directed`) for NCOL/LGL, the first graph of a GraphML document; the
    /// problem data of a DIMACS file is discarded (use
    /// [`read_graph_dimacs_flow`](Self::read_graph_dimacs_flow) to get it).
    ///
    /// # Errors
    /// [`ErrorKind::Unimplemented`] if the format can't be read (DOT, LEDA),
    /// plus the errors of the specific reader.
    ///
    /// # Examples
    /// ```
    /// use igraph::{foreign::GraphFormat, prelude::*};
    /// let path = std::env::temp_dir().join(format!("igraph-doc-read-{}.net", std::process::id()));
    /// std::fs::write(&path, "*Vertices 3\n*Edges\n1 2\n2 3\n").unwrap();
    /// let format = GraphFormat::from_path(&path).unwrap();
    /// let g = Graph::read_graph(&path, format, false).unwrap();
    /// std::fs::remove_file(&path).unwrap();
    /// assert_eq!(g.edge_list(), vec![(0, 1), (1, 2)]);
    /// ```
    pub fn read_graph(
        path: impl AsRef<Path>,
        format: GraphFormat,
        directed: bool,
    ) -> Result<Graph> {
        let path = path.as_ref();
        match format {
            GraphFormat::Edgelist => Self::read_graph_edgelist(path, 0, directed),
            GraphFormat::Ncol => Self::read_graph_ncol(
                path,
                &[],
                &NcolLglOptions::default().with_directed(directed),
            ),
            GraphFormat::Lgl => {
                Self::read_graph_lgl(path, &NcolLglOptions::default().with_directed(directed))
            }
            GraphFormat::Pajek => Self::read_graph_pajek(path),
            GraphFormat::GraphMl => Self::read_graph_graphml(path, 0),
            GraphFormat::Gml => Self::read_graph_gml(path),
            GraphFormat::DimacsFlow => Ok(Self::read_graph_dimacs_flow(path, directed)?.graph),
            GraphFormat::GraphDb => Self::read_graph_graphdb(path, directed),
            GraphFormat::Dl => Self::read_graph_dl(path, directed),
            GraphFormat::Dot | GraphFormat::Leda => Err(Error::new(
                ErrorKind::Unimplemented,
                format!("igraph cannot read the {format:?} format"),
            )),
        }
    }

    /// Reads an edge list file: an even number of non-negative integers
    /// (0-based vertex ids) separated by whitespace, conventionally one
    /// `from to` pair per line.
    ///
    /// The graph has `max(n, largest id + 1)` vertices, so `n = 0` is always
    /// safe; a larger `n` adds isolated vertices. See
    /// [`read_graph_ncol`](Self::read_graph_ncol) for files with symbolic
    /// vertex names. Time complexity: O(|V| + |E|).
    ///
    /// Binds [`igraph_read_graph_edgelist`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_read_graph_edgelist).
    ///
    /// # Errors
    /// [`ErrorKind::File`] if the file can't be opened, [`ErrorKind::Parse`]
    /// for syntax errors (non-integers, odd number of ids...).
    pub fn read_graph_edgelist(path: impl AsRef<Path>, n: usize, directed: bool) -> Result<Graph> {
        let n = to_int(n, "vertex count")?;
        read_with(Source::Path(path.as_ref()), |g, f| unsafe {
            igraph_read_graph_edgelist(g, f, n, directed)
        })
    }

    /// Parses an edge list from a string, see
    /// [`read_graph_edgelist`](Self::read_graph_edgelist).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::read_graph_edgelist_from_str("0 1\n1 2\n2 0\n", 5, false).unwrap();
    /// assert_eq!((g.vcount(), g.ecount()), (5, 3)); // two isolated vertices from `n`
    /// let bad = Graph::read_graph_edgelist_from_str("0 1 2", 0, false).unwrap_err();
    /// assert_eq!(bad.kind(), ErrorKind::Parse);
    /// ```
    pub fn read_graph_edgelist_from_str(text: &str, n: usize, directed: bool) -> Result<Graph> {
        let n = to_int(n, "vertex count")?;
        read_with(Source::Bytes(text.as_bytes()), |g, f| unsafe {
            igraph_read_graph_edgelist(g, f, n, directed)
        })
    }

    fn read_ncol_impl(
        source: Source<'_>,
        predefnames: &[&str],
        options: &NcolLglOptions,
    ) -> Result<Graph> {
        if predefnames.iter().any(|s| s.contains('\0')) {
            return Err(Error::invalid(
                "predefined vertex names must not contain NUL bytes",
            ));
        }
        let names: Option<StrVector> =
            (!predefnames.is_empty()).then(|| predefnames.iter().collect());
        let names_ptr = names
            .as_ref()
            .map_or(std::ptr::null(), |n| n as *const StrVector);
        read_with(source, |g, f| unsafe {
            igraph_read_graph_ncol(
                g,
                f,
                names_ptr,
                options.names,
                options.weights.into(),
                options.directed,
            )
        })
    }

    /// Reads an NCOL file, the symbolic weighted edge list format of the
    /// Large Graph Layout software.
    ///
    /// Each line is `name1 name2 [weight]`: two vertex names without
    /// whitespace, optionally followed by a (possibly negative, possibly
    /// scientific notation) weight. Vertex ids are assigned in the order the
    /// names first appear, after the names listed in `predefnames` (which get
    /// ids `0..predefnames.len()`; unknown names found in the file extend
    /// them, and duplicate predefined names are accepted, each with an igraph
    /// warning). An empty `predefnames` means none.
    /// Multi-edges and loops are accepted. Time complexity:
    /// O(|V| + |E| log |V|) ignoring parsing.
    ///
    /// With the [attribute handler](crate::attributes::enable) on, the names
    /// are stored in the `name` vertex attribute and the weights in the
    /// `weight` edge attribute, as requested by `options`; otherwise they are
    /// discarded ([module docs](self#attributes)).
    ///
    /// Binds [`igraph_read_graph_ncol`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_read_graph_ncol).
    ///
    /// # Errors
    /// [`ErrorKind::File`] if the file can't be opened, [`ErrorKind::Parse`]
    /// for syntax errors.
    pub fn read_graph_ncol(
        path: impl AsRef<Path>,
        predefnames: &[&str],
        options: &NcolLglOptions,
    ) -> Result<Graph> {
        Self::read_ncol_impl(Source::Path(path.as_ref()), predefnames, options)
    }

    /// Parses an NCOL graph from a string, see
    /// [`read_graph_ncol`](Self::read_graph_ncol).
    ///
    /// # Examples
    /// ```
    /// use igraph::{foreign::NcolLglOptions, prelude::*};
    /// let text = "alice bob 2.5\nbob carol\ncarol alice -1e2\n";
    /// // Fix the vertex ids through the predefined names: carol=0, bob=1, alice=2.
    /// let g = Graph::read_graph_ncol_from_str(text, &["carol", "bob", "alice"], &NcolLglOptions::default())
    ///     .unwrap();
    /// assert_eq!(g.edge_list(), vec![(1, 2), (0, 1), (0, 2)]);
    /// ```
    pub fn read_graph_ncol_from_str(
        text: &str,
        predefnames: &[&str],
        options: &NcolLglOptions,
    ) -> Result<Graph> {
        Self::read_ncol_impl(Source::Bytes(text.as_bytes()), predefnames, options)
    }

    fn read_lgl_impl(source: Source<'_>, options: &NcolLglOptions) -> Result<Graph> {
        read_with(source, |g, f| unsafe {
            igraph_read_graph_lgl(
                g,
                f,
                options.names,
                options.weights.into(),
                options.directed,
            )
        })
    }

    /// Reads an LGL file (Large Graph Layout format).
    ///
    /// The file is a sequence of blocks: a line `# name` introduces a vertex,
    /// and each following line `other [weight]` adds an edge from it to
    /// `other`, until the next `#` line. A `#` line with no following lines
    /// defines an isolated vertex. Vertex ids are assigned in the order the
    /// names first appear. Time complexity: O(|V| + |E| log |V|) ignoring
    /// parsing.
    ///
    /// With the [attribute handler](crate::attributes::enable) on, the names
    /// are stored in the `name` vertex attribute and the weights in the
    /// `weight` edge attribute, as requested by `options`; otherwise they are
    /// discarded ([module docs](self#attributes)).
    ///
    /// Binds [`igraph_read_graph_lgl`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_read_graph_lgl).
    ///
    /// # Errors
    /// [`ErrorKind::File`] if the file can't be opened, [`ErrorKind::Parse`]
    /// for syntax errors.
    pub fn read_graph_lgl(path: impl AsRef<Path>, options: &NcolLglOptions) -> Result<Graph> {
        Self::read_lgl_impl(Source::Path(path.as_ref()), options)
    }

    /// Parses an LGL graph from a string, see [`read_graph_lgl`](Self::read_graph_lgl).
    ///
    /// # Examples
    /// ```
    /// use igraph::{foreign::NcolLglOptions, prelude::*};
    /// let text = "# a\nb\nc 2.0\n# b\nc\n# lonely\n";
    /// let g = Graph::read_graph_lgl_from_str(text, &NcolLglOptions::default()).unwrap();
    /// assert_eq!(g.vcount(), 4);
    /// assert_eq!(g.edge_list(), vec![(0, 1), (0, 2), (1, 2)]);
    /// ```
    pub fn read_graph_lgl_from_str(text: &str, options: &NcolLglOptions) -> Result<Graph> {
        Self::read_lgl_impl(Source::Bytes(text.as_bytes()), options)
    }

    /// Reads a Pajek `.net` file.
    ///
    /// Only a subset of the format is supported: `.paj` project files,
    /// temporal networks, graphs mixing directed and undirected edges,
    /// permutations/hierarchies/clusters/vectors and multi-relational networks
    /// are not. `*Arcs` sections create a directed graph, `*Edges` an
    /// undirected one; `*Arcslist`/`*Edgeslist` and matrix sections are
    /// understood, as well as bipartite (two-mode) networks. Vertex ids in the
    /// file are 1-based. Time complexity: O(|V| + |E|).
    ///
    /// With the [attribute handler](crate::attributes::enable) on, vertex
    /// labels become the `name` vertex attribute, coordinates `x`, `y` (and
    /// `z`), edge weights (including the entries of `*Matrix` sections) the
    /// `weight` edge attribute, and the other Pajek parameters get
    /// descriptive names (`ic`/`c` → `color`, `bc` → `framecolor`,
    /// `x_fact` → `xfact`, `l` → `label`, `w` → `edgewidth`, ...; unknown ones
    /// are kept as string attributes). A parameter given for some elements
    /// only takes Pajek's default elsewhere (e.g. `color` is `LightOrange`
    /// for vertices and `MidnightBlue` for edges; plain numbers such as the
    /// coordinates and weights are NaN). Two-mode networks get the boolean
    /// `type` vertex attribute (`false` for the first mode), as expected by
    /// the [`bipartite`](crate::bipartite) functions. Without the handler all
    /// of this is discarded ([module docs](self#attributes)).
    ///
    /// Binds [`igraph_read_graph_pajek`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_read_graph_pajek).
    ///
    /// # Errors
    /// [`ErrorKind::File`] if the file can't be opened, [`ErrorKind::Parse`]
    /// for syntax errors.
    pub fn read_graph_pajek(path: impl AsRef<Path>) -> Result<Graph> {
        read_with(Source::Path(path.as_ref()), |g, f| unsafe {
            igraph_read_graph_pajek(g, f)
        })
    }

    /// Parses a Pajek graph from a string, see [`read_graph_pajek`](Self::read_graph_pajek).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let text = "*Vertices 4\n1 \"A\"\n2 \"B\"\n3 \"C\"\n4 \"D\"\n*Arcs\n1 2 0.5\n2 3\n3 1\n";
    /// let g = Graph::read_graph_pajek_from_str(text).unwrap();
    /// assert!(g.is_directed());
    /// assert_eq!((g.vcount(), g.ecount()), (4, 3));
    /// assert_eq!(g.edge(0).unwrap(), (0, 1));
    /// ```
    pub fn read_graph_pajek_from_str(text: &str) -> Result<Graph> {
        read_with(Source::Bytes(text.as_bytes()), |g, f| unsafe {
            igraph_read_graph_pajek(g, f)
        })
    }

    /// Reads a GraphML file.
    ///
    /// Only basic GraphML is supported: no nested graphs, no hyperedges.
    /// Directedness comes from the `edgedefault` attribute of the `graph`
    /// element. If the file contains several graphs, `index` selects which
    /// one to load (0 for the first); note that igraph 1.0.0 and 1.0.1 fail
    /// with "Graph index was too large" ([`ErrorKind::InvalidValue`]) for any
    /// index but 0, even when the document does contain more graphs.
    /// Vertices get ids in order of appearance, edges keep the document
    /// order.
    ///
    /// With the [attribute handler](crate::attributes::enable) on, every
    /// `<key>` becomes a graph, vertex or edge attribute named after its
    /// `attr.name` (or its `id` if `attr.name` is missing): `boolean` keys
    /// give boolean attributes, `int`/`long`/`float`/`double` numeric ones,
    /// `string` string ones (UTF-8). Missing values take the key's
    /// `<default>`, or NaN/`""`/`false`. The GraphML `id`s of the nodes are
    /// kept in the string `id` vertex attribute (unless a vertex key already
    /// defines an attribute named `id`, in which case igraph only warns).
    /// When the edges have ids, igraph 1.0.0 and 1.0.1 also create a string
    /// `id` *edge* attribute (unless an edge key defines one) but, because of
    /// a bug in `src/io/graphml.c`, fill it with the *node* ids, in order,
    /// instead of the edge ids (padded with `""` when there are more edges
    /// than nodes): don't rely on it. Without the handler all of this is discarded
    /// ([module docs](self#attributes)).
    ///
    /// Binds [`igraph_read_graph_graphml`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_read_graph_graphml).
    ///
    /// # Errors
    /// [`ErrorKind::File`] if the file can't be opened, [`ErrorKind::Parse`]
    /// for malformed files, [`ErrorKind::InvalidValue`] if `index` is too
    /// large, [`ErrorKind::Unimplemented`] if the igraph library was compiled
    /// without GraphML (libxml2) support.
    pub fn read_graph_graphml(path: impl AsRef<Path>, index: usize) -> Result<Graph> {
        let index = to_int(index, "graph index")?;
        read_with(Source::Path(path.as_ref()), |g, f| unsafe {
            igraph_read_graph_graphml(g, f, index)
        })
    }

    /// Parses a GraphML document from a string, see
    /// [`read_graph_graphml`](Self::read_graph_graphml).
    ///
    /// # Examples
    /// ```
    /// use igraph::{attributes, prelude::*};
    /// attributes::enable().unwrap();
    /// let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
    /// <graphml xmlns="http://graphml.graphdrawing.org/xmlns">
    ///   <key id="d0" for="node" attr.name="age" attr.type="int"><default>20</default></key>
    ///   <key id="d1" for="edge" attr.name="since" attr.type="double"/>
    ///   <graph edgedefault="directed">
    ///     <node id="ann"><data key="d0">30</data></node>
    ///     <node id="bob"/>
    ///     <edge source="bob" target="ann"><data key="d1">2019</data></edge>
    ///   </graph>
    /// </graphml>"#;
    /// let g = Graph::read_graph_graphml_from_str(xml, 0).unwrap();
    /// assert!(g.is_directed());
    /// assert_eq!(g.edge_list(), vec![(1, 0)]);
    /// assert_eq!(g.vertex_attr_str_values("id", ..).unwrap(), ["ann", "bob"]);
    /// assert_eq!(g.vertex_attr_numeric_values("age", ..).unwrap(), [30.0, 20.0]); // bob: default
    /// assert_eq!(g.edge_attr_numeric("since", 0).unwrap(), 2019.0);
    /// ```
    pub fn read_graph_graphml_from_str(text: &str, index: usize) -> Result<Graph> {
        let index = to_int(index, "graph index")?;
        read_with(Source::Bytes(text.as_bytes()), |g, f| unsafe {
            igraph_read_graph_graphml(g, f, index)
        })
    }

    fn read_dimacs_impl(source: Source<'_>, directed: bool) -> Result<DimacsFlow> {
        let mut problem = StrVector::new();
        let mut labels = VectorInt::new();
        let mut capacity = Vector::new();
        let (mut s, mut t) = (-1, -1);
        let graph = read_with(source, |g, f| unsafe {
            igraph_read_graph_dimacs_flow(
                g,
                f,
                &mut problem,
                &mut labels,
                &mut s,
                &mut t,
                &mut capacity,
                directed,
            )
        })?;
        let problem_name = problem.get(0).unwrap_or_default().to_owned();
        let problem = if problem_name == "edge" {
            DimacsProblem::Edge {
                labels: labels.into(),
            }
        } else {
            // igraph reports a missing `n` line as id -2 (0 - 1 - 1) and
            // doesn't check the ids it read against the vertex count.
            let n = graph.vcount() as VertexId;
            let check = |id: igraph_int_t, what: &str| -> Result<Option<VertexId>> {
                match id {
                    -2 => Ok(None),
                    id if (0..n).contains(&id) => Ok(Some(id)),
                    id => Err(Error::new(
                        ErrorKind::Parse,
                        format!(
                            "DIMACS {what} vertex {} out of range (the graph has {n} vertices)",
                            id + 1
                        ),
                    )),
                }
            };
            DimacsProblem::Max {
                source: check(s, "source")?,
                target: check(t, "target")?,
                capacity: capacity.into(),
            }
        };
        Ok(DimacsFlow {
            graph,
            problem_name,
            problem,
        })
    }

    /// Reads a DIMACS network flow file.
    ///
    /// DIMACS is a line oriented format; the first character of each line
    /// gives its type: `c` comment, `p` the problem line (`p max|edge
    /// <vertices> <edges>`, before any node or arc line), `n` node lines and
    /// `a` arc lines (`a from to capacity`, max-flow problems) or `e` edge
    /// lines (`e from to`, edge problems). In max-flow problems exactly two
    /// node lines `n <id> s|t` mark the source and the target. In edge
    /// problems the labels are the 1-based vertex indices: `n <id> <label>`
    /// lines are meant to change them, but igraph 1.0.0 and 1.0.1 reject them
    /// with a parse error. Vertex ids in the file start from 1, the returned ids
    /// from 0. A max-flow file without the source (or target) `n` line is
    /// accepted, with `None` in [`DimacsProblem::Max`]. Time complexity:
    /// O(|V| + |E| + c), c being the file size. The capacities are returned
    /// in [`DimacsProblem::Max`], never as attributes.
    ///
    /// See [`Graph::maxflow`] and [`Graph::st_mincut`] to solve the problem.
    ///
    /// Binds [`igraph_read_graph_dimacs_flow`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_read_graph_dimacs_flow).
    ///
    /// # Errors
    /// [`ErrorKind::File`] if the file can't be opened, [`ErrorKind::Parse`]
    /// or [`ErrorKind::InvalidValue`] for malformed files, including a source
    /// or target `n` line naming a vertex that doesn't exist.
    pub fn read_graph_dimacs_flow(path: impl AsRef<Path>, directed: bool) -> Result<DimacsFlow> {
        Self::read_dimacs_impl(Source::Path(path.as_ref()), directed)
    }

    /// Parses a DIMACS flow problem from a string, see
    /// [`read_graph_dimacs_flow`](Self::read_graph_dimacs_flow).
    ///
    /// # Examples
    /// ```
    /// use igraph::{foreign::DimacsProblem, prelude::*};
    /// let text = "c a tiny network\np max 3 2\nn 1 s\nn 3 t\na 1 2 4\na 2 3 7\n";
    /// let flow = Graph::read_graph_dimacs_flow_from_str(text, true).unwrap();
    /// assert_eq!(flow.problem_name, "max");
    /// assert_eq!(flow.graph.edge_list(), vec![(0, 1), (1, 2)]);
    /// let DimacsProblem::Max { source: Some(s), target: Some(t), capacity } = flow.problem else {
    ///     panic!("a max-flow problem with both terminals was expected");
    /// };
    /// assert_eq!((s, t, capacity.as_slice()), (0, 2, &[4.0, 7.0][..]));
    /// // Solve it: the bottleneck is the first arc.
    /// assert_eq!(flow.graph.maxflow_value(s, t, Some(&capacity)).unwrap(), 4.0);
    /// ```
    pub fn read_graph_dimacs_flow_from_str(text: &str, directed: bool) -> Result<DimacsFlow> {
        Self::read_dimacs_impl(Source::Bytes(text.as_bytes()), directed)
    }

    /// Reads a graph in the binary format of the ARG graph database (used
    /// to benchmark isomorphism algorithms).
    ///
    /// The file is a sequence of 16-bit little-endian words: the number of
    /// vertices, then for each vertex the number of its out-edges followed by
    /// their (0-based) targets. Only unlabelled graphs are supported. Time
    /// complexity: O(|V| + |E|).
    ///
    /// The database is a benchmark for isomorphism algorithms: see
    /// [`Graph::isomorphic`] and the rest of the
    /// [`isomorphism`](crate::isomorphism) module.
    ///
    /// Binds [`igraph_read_graph_graphdb`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_read_graph_graphdb).
    ///
    /// # Errors
    /// [`ErrorKind::File`] if the file can't be opened, [`ErrorKind::Parse`]
    /// for truncated files or trailing bytes.
    pub fn read_graph_graphdb(path: impl AsRef<Path>, directed: bool) -> Result<Graph> {
        read_with(Source::Path(path.as_ref()), |g, f| unsafe {
            igraph_read_graph_graphdb(g, f, directed)
        })
    }

    /// Parses a graph in the binary graph database format from bytes, see
    /// [`read_graph_graphdb`](Self::read_graph_graphdb).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // 4 vertices; 0 -> [2]; 1 -> [0]; 2 -> []; 3 -> [0, 2] (igraph's unit test file).
    /// let words: [u16; 9] = [4, 1, 2, 1, 0, 0, 2, 0, 2];
    /// let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    /// let g = Graph::read_graph_graphdb_from_bytes(&bytes, true).unwrap();
    /// assert_eq!(g.edge_list(), vec![(0, 2), (1, 0), (3, 0), (3, 2)]);
    /// ```
    pub fn read_graph_graphdb_from_bytes(data: &[u8], directed: bool) -> Result<Graph> {
        read_with(Source::Bytes(data), |g, f| unsafe {
            igraph_read_graph_graphdb(g, f, directed)
        })
    }

    /// Reads a GML file.
    ///
    /// Any syntactically correct GML is parsed, but only a subset is used:
    /// the first `graph` record, its `directed` flag, `node` records (with
    /// their `id`) and `edge` records (with `source` and `target`); other top
    /// level records are ignored. `inf`, `-inf` and `nan` are accepted as
    /// reals (case insensitively). Time complexity: proportional to the file
    /// length.
    ///
    /// With the [attribute handler](crate::attributes::enable) on, every
    /// field of simple type (integer, real, string) of the graph, node and
    /// edge records becomes a numeric or string attribute (including the
    /// numeric `id` of the nodes, and `comment` fields); composite fields
    /// (records) are ignored with a warning, and missing values are NaN or
    /// `""`. Only the `quot`, `amp`, `apos`, `lt` and `gt` entities are
    /// decoded. Without the handler all of this is discarded
    /// ([module docs](self#attributes)).
    ///
    /// The C parser of igraph 1.0.0 and 1.0.1 is not reentrant (it uses
    /// static buffers), so this wrapper serializes GML reads across threads
    /// with a process-wide lock; all the other readers run fully in parallel.
    ///
    /// Binds [`igraph_read_graph_gml`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_read_graph_gml).
    ///
    /// # Errors
    /// [`ErrorKind::File`] if the file can't be opened, [`ErrorKind::Parse`]
    /// for syntax errors and for structural problems: no `graph` record,
    /// duplicate or non-integer node ids, edges without `source`/`target` or
    /// referring to unknown node ids.
    pub fn read_graph_gml(path: impl AsRef<Path>) -> Result<Graph> {
        read_gml(Source::Path(path.as_ref()))
    }

    /// Parses a GML graph from a string, see [`read_graph_gml`](Self::read_graph_gml).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let text = r#"graph [
    ///   directed 0
    ///   node [ id 10 label "x" ]
    ///   node [ id 20 ]
    ///   node [ id 30 ]
    ///   edge [ source 10 target 20 ]
    ///   edge [ source 30 target 10 weight 2.5 ]
    /// ]"#;
    /// let g = Graph::read_graph_gml_from_str(text).unwrap();
    /// assert!(!g.is_directed());
    /// // GML ids are mapped to consecutive vertex ids in order of appearance.
    /// assert_eq!(g.edge_list(), vec![(0, 1), (0, 2)]);
    /// ```
    pub fn read_graph_gml_from_str(text: &str) -> Result<Graph> {
        read_gml(Source::Bytes(text.as_bytes()))
    }

    /// Reads a file in the DL format of UCINET.
    ///
    /// All the forms of the format are supported: full matrix, edge list
    /// (`format = edgelist1`) and node list (`format = nodelist1`), with or
    /// without labels. Labels are case sensitive. With the
    /// [attribute handler](crate::attributes::enable) on, labels are stored
    /// in the `name` vertex attribute and edge values in the `weight` edge
    /// attribute; otherwise they are discarded
    /// ([module docs](self#attributes)). Time complexity: linear in the
    /// number of vertices and edges, quadratic in the number of vertices for
    /// the full matrix form.
    ///
    /// Binds [`igraph_read_graph_dl`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_read_graph_dl).
    ///
    /// # Errors
    /// [`ErrorKind::File`] if the file can't be opened, [`ErrorKind::Parse`]
    /// for syntax errors.
    pub fn read_graph_dl(path: impl AsRef<Path>, directed: bool) -> Result<Graph> {
        read_with(Source::Path(path.as_ref()), |g, f| unsafe {
            igraph_read_graph_dl(g, f, directed)
        })
    }

    /// Parses a UCINET DL graph from a string, see [`read_graph_dl`](Self::read_graph_dl).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // `fullmatrix1.dl` from igraph's examples.
    /// let text = "DL N = 5\nData:\n0 1 1 1 1\n1 0 1 0 0\n1 1 0 0 1\n1 0 0 0 0\n1 0 1 0 0\n";
    /// let g = Graph::read_graph_dl_from_str(text, true).unwrap();
    /// assert_eq!((g.vcount(), g.ecount()), (5, 12));
    /// assert_eq!(g.edge(0).unwrap(), (0, 1));
    /// ```
    pub fn read_graph_dl_from_str(text: &str, directed: bool) -> Result<Graph> {
        read_with(Source::Bytes(text.as_bytes()), |g, f| unsafe {
            igraph_read_graph_dl(g, f, directed)
        })
    }
}

// ---------------------------------------------------------------------------
// Writers
// ---------------------------------------------------------------------------

impl igraph_t {
    /// Writes the graph to a file in the given `format`, with default options
    /// (NCOL/LGL/LEDA without the optional named attributes, GraphML without
    /// prefixes, GML with the default [`GmlWriteOptions`], LGL without
    /// isolated vertices). GraphML, GML and DOT export all the attributes, and
    /// Pajek those naming Pajek parameters, when the
    /// [attribute handler](crate::attributes::enable) is on.
    ///
    /// # Errors
    /// [`ErrorKind::Unimplemented`] if the format can't be written generically
    /// (graph database, DL, DIMACS), plus the errors of the specific writer.
    ///
    /// # Examples
    /// ```
    /// use igraph::{foreign::GraphFormat, prelude::*};
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// let path = std::env::temp_dir().join(format!("igraph-doc-write-{}.dot", std::process::id()));
    /// g.write_graph(&path, GraphFormat::from_path(&path).unwrap()).unwrap();
    /// let dot = std::fs::read_to_string(&path).unwrap();
    /// std::fs::remove_file(&path).unwrap();
    /// assert!(dot.contains("graph {") && dot.contains("1 -- 0;") && dot.contains("2 -- 1;"));
    /// ```
    pub fn write_graph(&self, path: impl AsRef<Path>, format: GraphFormat) -> Result<()> {
        let path = path.as_ref();
        match format {
            GraphFormat::Edgelist => self.write_graph_edgelist(path),
            GraphFormat::Ncol => self.write_graph_ncol(path, None, None),
            GraphFormat::Lgl => self.write_graph_lgl(path, None, None, false),
            GraphFormat::Pajek => self.write_graph_pajek(path),
            GraphFormat::GraphMl => self.write_graph_graphml(path),
            GraphFormat::Gml => self.write_graph_gml(path, &GmlWriteOptions::default()),
            GraphFormat::Dot => self.write_graph_dot(path),
            GraphFormat::Leda => self.write_graph_leda(path, None, None),
            GraphFormat::DimacsFlow | GraphFormat::GraphDb | GraphFormat::Dl => Err(Error::new(
                ErrorKind::Unimplemented,
                format!("the {format:?} format can't be written generically"),
            )),
        }
    }

    /// Writes the edge list of the graph to a file: one `from to` line per
    /// edge (0-based ids, a single space as separator). Lines are sorted by
    /// the first endpoint, so edge ids are preserved by a round trip only if
    /// the edges were already sorted that way; isolated vertices (with
    /// ids above the largest endpoint) are lost unless the vertex count is
    /// passed back to [`read_graph_edgelist`](Self::read_graph_edgelist).
    /// Time complexity: O(|E|).
    ///
    /// Binds [`igraph_write_graph_edgelist`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_write_graph_edgelist).
    ///
    /// # Errors
    /// [`ErrorKind::File`] if the file can't be created or written.
    pub fn write_graph_edgelist(&self, path: impl AsRef<Path>) -> Result<()> {
        write_to_path(path.as_ref(), |f| unsafe {
            igraph_write_graph_edgelist(self, f)
        })
    }

    /// The edge list serialization as a string, see
    /// [`write_graph_edgelist`](Self::write_graph_edgelist).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(2, 1), (0, 1)], 3, true).unwrap();
    /// // Sorted by the source vertex, not by edge id.
    /// assert_eq!(g.write_graph_edgelist_to_string().unwrap(), "0 1\n2 1\n");
    /// ```
    pub fn write_graph_edgelist_to_string(&self) -> Result<String> {
        write_to_string(|f| unsafe { igraph_write_graph_edgelist(self, f) })
    }

    /// Writes the graph to an NCOL file (see
    /// [`read_graph_ncol`](Self::read_graph_ncol)): one `from to [weight]`
    /// line per edge.
    ///
    /// `names` is the name of a string vertex attribute to write instead of
    /// the vertex ids, `weights` the name of a numeric edge attribute to
    /// write as weights; `None` skips them. A requested attribute that does
    /// not exist (always the case without the
    /// [attribute handler](crate::attributes::enable)) is skipped with an
    /// igraph warning ([module docs](self#attributes)). NaN and infinite
    /// weights are written as `NaN`, `Inf` and `-Inf`, and read back as such.
    /// Names must be non-empty and contain no spaces or non-printable
    /// characters. The format can't represent
    /// isolated vertices; multi-edges and loops are written (though they
    /// break the LGL software). Time complexity: O(|E|).
    ///
    /// Binds [`igraph_write_graph_ncol`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_write_graph_ncol).
    ///
    /// # Errors
    /// [`ErrorKind::File`] on I/O errors, [`ErrorKind::InvalidValue`] for
    /// attribute names with NUL bytes or invalid vertex names.
    pub fn write_graph_ncol(
        &self,
        path: impl AsRef<Path>,
        names: Option<&str>,
        weights: Option<&str>,
    ) -> Result<()> {
        let (n, w) = (
            opt_cstring(names, "names")?,
            opt_cstring(weights, "weights")?,
        );
        write_to_path(path.as_ref(), |f| unsafe {
            igraph_write_graph_ncol(self, f, opt_ptr(&n), opt_ptr(&w))
        })
    }

    /// The NCOL serialization as a string, see
    /// [`write_graph_ncol`](Self::write_graph_ncol).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 4, false).unwrap();
    /// assert_eq!(g.write_graph_ncol_to_string(None, None).unwrap(), "0 1\n1 2\n");
    /// ```
    pub fn write_graph_ncol_to_string(
        &self,
        names: Option<&str>,
        weights: Option<&str>,
    ) -> Result<String> {
        let (n, w) = (
            opt_cstring(names, "names")?,
            opt_cstring(weights, "weights")?,
        );
        write_to_string(|f| unsafe { igraph_write_graph_ncol(self, f, opt_ptr(&n), opt_ptr(&w)) })
    }

    /// Writes the graph to an LGL file (see [`read_graph_lgl`](Self::read_graph_lgl)).
    ///
    /// Edges are grouped by their first endpoint: `# from` followed by one
    /// `to [weight]` line per edge. `names`/`weights` name a string vertex
    /// attribute and a numeric edge attribute to write (skipped with a
    /// warning when missing, as always without the
    /// [attribute handler](crate::attributes::enable), see the
    /// [module docs](self#attributes)); names must be non-empty and contain
    /// no spaces, `#` or non-printable characters. With `isolates = true` isolated
    /// vertices are written as lone `# v` lines. Time complexity: O(|E|), or
    /// O(|V| + |E|) with isolates.
    ///
    /// Binds [`igraph_write_graph_lgl`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_write_graph_lgl).
    ///
    /// # Errors
    /// [`ErrorKind::File`] on I/O errors, [`ErrorKind::InvalidValue`] for
    /// attribute names with NUL bytes or invalid vertex names.
    pub fn write_graph_lgl(
        &self,
        path: impl AsRef<Path>,
        names: Option<&str>,
        weights: Option<&str>,
        isolates: bool,
    ) -> Result<()> {
        let (n, w) = (
            opt_cstring(names, "names")?,
            opt_cstring(weights, "weights")?,
        );
        write_to_path(path.as_ref(), |f| unsafe {
            igraph_write_graph_lgl(self, f, opt_ptr(&n), opt_ptr(&w), isolates)
        })
    }

    /// The LGL serialization as a string, see [`write_graph_lgl`](Self::write_graph_lgl).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// // The graph of igraph's `igraph_write_graph_lgl.c` example.
    /// let g = Graph::from_edges(&[(0, 1), (1, 3), (1, 2), (2, 0), (4, 2), (3, 4)], 7, false).unwrap();
    /// let lgl = g.write_graph_lgl_to_string(None, None, true).unwrap();
    /// assert_eq!(lgl, "# 0\n1\n2\n# 1\n2\n3\n# 2\n4\n# 3\n4\n# 5\n# 6\n");
    /// ```
    pub fn write_graph_lgl_to_string(
        &self,
        names: Option<&str>,
        weights: Option<&str>,
        isolates: bool,
    ) -> Result<String> {
        let (n, w) = (
            opt_cstring(names, "names")?,
            opt_cstring(weights, "weights")?,
        );
        write_to_string(|f| unsafe {
            igraph_write_graph_lgl(self, f, opt_ptr(&n), opt_ptr(&w), isolates)
        })
    }

    /// Writes the graph to a GraphML file (without attribute prefixes).
    ///
    /// GraphML is an XML format, see the
    /// [GraphML primer](http://graphml.graphdrawing.org/primer/graphml-primer.html).
    /// Vertices are written as `n0, n1, ...`, edges in id order, and the
    /// `edgedefault` reflects the directedness. With the
    /// [attribute handler](crate::attributes::enable) on, all graph, vertex
    /// and edge attributes are written as typed `<key>`s (`boolean`,
    /// `double` or `string`; NaN values are omitted), so a GraphML round trip
    /// preserves them ([module docs](self#attributes)). Attribute names and
    /// string values should be UTF-8 (igraph copies the bytes as they are)
    /// and must not contain control characters other than tab, CR and LF. Use
    /// [`write_graph_graphml_with`](Self::write_graph_graphml_with) to
    /// control the attribute name prefixes. Time complexity: O(|V| + |E|).
    ///
    /// This replaces the former, infallible `write_graph_graphml(&self, &str)`
    /// that leaked its file handle: the file is now always closed and errors
    /// are reported.
    ///
    /// Binds [`igraph_write_graph_graphml`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_write_graph_graphml).
    ///
    /// # Errors
    /// [`ErrorKind::File`] if the file can't be created or written,
    /// [`ErrorKind::InvalidValue`] if an attribute name or string value
    /// contains a control character other than tab, CR and LF.
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, true).unwrap();
    /// let path = std::env::temp_dir().join(format!("igraph-doc-{}.graphml", std::process::id()));
    /// g.write_graph_graphml(&path).unwrap();
    /// let xml = std::fs::read_to_string(&path).unwrap();
    /// std::fs::remove_file(&path).unwrap();
    /// assert!(xml.contains(r#"edgedefault="directed""#));
    /// assert!(xml.contains(r#"<edge source="n2" target="n0">"#));
    /// ```
    pub fn write_graph_graphml(&self, path: impl AsRef<Path>) -> Result<()> {
        self.write_graph_graphml_with(path, false)
    }

    /// Writes the graph to a GraphML file; with `prefixattr = true`
    /// attribute names get a `g_`, `v_` or `e_` prefix to keep graph, vertex
    /// and edge attributes with the same name distinct. See
    /// [`write_graph_graphml`](Self::write_graph_graphml).
    ///
    /// Binds [`igraph_write_graph_graphml`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_write_graph_graphml).
    ///
    /// # Errors
    /// As [`write_graph_graphml`](Self::write_graph_graphml).
    pub fn write_graph_graphml_with(&self, path: impl AsRef<Path>, prefixattr: bool) -> Result<()> {
        write_to_path(path.as_ref(), |f| unsafe {
            igraph_write_graph_graphml(self, f, prefixattr)
        })
    }

    /// The GraphML serialization as a string, see
    /// [`write_graph_graphml_with`](Self::write_graph_graphml_with).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let mut g = Graph::from_edges(&[(0, 1)], 2, false).unwrap();
    /// // Setting an attribute turns on the attribute handler.
    /// g.set_vertex_attr_str_values("name", &["ann", "bob"]).unwrap();
    /// g.set_edge_attr_numeric("weight", 0, 2.5).unwrap();
    /// let xml = g.write_graph_graphml_to_string(true).unwrap();
    /// assert!(xml.contains(r#"<key id="v_name" for="node" attr.name="name" attr.type="string"/>"#));
    /// assert!(xml.contains(r#"<data key="v_name">bob</data>"#));
    /// assert!(xml.contains(r#"<data key="e_weight">2.5</data>"#));
    /// // Reading it back restores the attributes (under their original names).
    /// let h = Graph::read_graph_graphml_from_str(&xml, 0).unwrap();
    /// assert_eq!(h.vertex_attr_str_values("name", ..).unwrap(), ["ann", "bob"]);
    /// assert_eq!(h.edge_attr_numeric("weight", 0).unwrap(), 2.5);
    /// ```
    pub fn write_graph_graphml_to_string(&self, prefixattr: bool) -> Result<String> {
        write_to_string(|f| unsafe { igraph_write_graph_graphml(self, f, prefixattr) })
    }

    /// Writes the graph to a Pajek `.net` file.
    ///
    /// The format is meant for interoperability with the Pajek software, not
    /// for data exchange. Vertex ids are written 1-based; directed graphs use
    /// an `*Arcs` section, undirected ones `*Edges`. Vertex and edge
    /// parameters come from the attributes named as in
    /// [`read_graph_pajek`](Self::read_graph_pajek) (`name` as the label,
    /// `x`/`y`/`z`, `color`, edge `weight`, ...; other attributes are
    /// discarded), hence are only written with the
    /// [attribute handler](crate::attributes::enable) on
    /// ([module docs](self#attributes)). A boolean `type` vertex attribute
    /// makes a two-mode (bipartite) file: since Pajek needs the vertices of
    /// the first mode first, the vertices are then reordered, and their ids
    /// change on a round trip (without a `name` attribute the labels are then
    /// the original 1-based ids). igraph never writes a UTF-8 byte-order mark.
    /// Time complexity: O(|V| + |E|).
    ///
    /// Binds [`igraph_write_graph_pajek`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_write_graph_pajek).
    ///
    /// # Errors
    /// [`ErrorKind::File`] if the file can't be created or written.
    pub fn write_graph_pajek(&self, path: impl AsRef<Path>) -> Result<()> {
        write_to_path(path.as_ref(), |f| unsafe {
            igraph_write_graph_pajek(self, f)
        })
    }

    /// The Pajek serialization as a string, see
    /// [`write_graph_pajek`](Self::write_graph_pajek).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, true).unwrap();
    /// assert_eq!(g.write_graph_pajek_to_string().unwrap(), "*Vertices 3\n*Arcs\n1 2\n2 3\n3 1\n");
    /// ```
    pub fn write_graph_pajek_to_string(&self) -> Result<String> {
        write_to_string(|f| unsafe { igraph_write_graph_pajek(self, f) })
    }

    /// Writes a maximum flow problem in DIMACS format (see
    /// [`read_graph_dimacs_flow`](Self::read_graph_dimacs_flow)): a comment,
    /// the `p max` line, the source and target `n` lines and one
    /// `a from to capacity` line per edge (1-based ids). `capacity` must have
    /// one entry per edge. Time complexity: O(|E|).
    ///
    /// See [`Graph::maxflow`] to solve the instance directly.
    ///
    /// Binds [`igraph_write_graph_dimacs_flow`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_write_graph_dimacs_flow).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidVertexId`] if `source` or `target` is not a vertex
    /// of the graph (igraph itself would write them unchecked),
    /// [`ErrorKind::InvalidValue`] if `capacity.len() != ecount`,
    /// [`ErrorKind::File`] on I/O errors.
    pub fn write_graph_dimacs_flow(
        &self,
        path: impl AsRef<Path>,
        source: VertexId,
        target: VertexId,
        capacity: &[f64],
    ) -> Result<()> {
        self.check_dimacs_args(source, target, capacity)?;
        let cap = Vector::view(capacity);
        write_to_path(path.as_ref(), |f| unsafe {
            igraph_write_graph_dimacs_flow(self, f, source, target, cap.as_ptr())
        })
    }

    /// The DIMACS serialization as a string, see
    /// [`write_graph_dimacs_flow`](Self::write_graph_dimacs_flow).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    /// let text = g.write_graph_dimacs_flow_to_string(0, 2, &[4.0, 7.5]).unwrap();
    /// assert_eq!(text, "c created by igraph\np max 3 2\nn 1 s\nn 3 t\na 1 2 4\na 2 3 7.5\n");
    /// ```
    pub fn write_graph_dimacs_flow_to_string(
        &self,
        source: VertexId,
        target: VertexId,
        capacity: &[f64],
    ) -> Result<String> {
        self.check_dimacs_args(source, target, capacity)?;
        let cap = Vector::view(capacity);
        write_to_string(|f| unsafe {
            igraph_write_graph_dimacs_flow(self, f, source, target, cap.as_ptr())
        })
    }

    fn check_dimacs_args(
        &self,
        source: VertexId,
        target: VertexId,
        capacity: &[f64],
    ) -> Result<()> {
        // igraph writes any id verbatim, producing files that can't be read back.
        for (what, v) in [("source", source), ("target", target)] {
            if !(0..self.vcount() as VertexId).contains(&v) {
                return Err(Error::new(
                    ErrorKind::InvalidVertexId,
                    format!(
                        "invalid {what} vertex {v} for a graph with {} vertices",
                        self.vcount()
                    ),
                ));
            }
        }
        if capacity.len() != self.ecount() {
            return Err(Error::invalid(format!(
                "the capacity vector has length {}, but the graph has {} edges",
                capacity.len(),
                self.ecount()
            )));
        }
        Ok(())
    }

    /// Writes the graph to a GML file.
    ///
    /// The output lists the `directed` flag, one `node [ id ... ]` record per
    /// vertex and one `edge [ source ... target ... ]` record per edge. See
    /// [`GmlWriteOptions`] for the ids, the creator line and the entity
    /// encoding. Time complexity: proportional to the output size.
    ///
    /// With the [attribute handler](crate::attributes::enable) on, numeric
    /// and string attributes are written too (booleans as 0/1, NaN values
    /// skipped, infinite values kept with a warning since they are not
    /// standard GML). Attribute names are reduced to their alphanumeric
    /// characters (prefixed with `igraph` if they don't start with a
    /// letter); attributes whose name would clash with the GML structure
    /// (`source`/`target` on edges, `directed`, `node` and `edge` on the
    /// graph, `id` on vertices) are skipped with a warning. The `id` vertex
    /// attribute itself is never written as an attribute: a numeric one
    /// supplies the node ids (see [`GmlWriteOptions::ids`]), any other is
    /// dropped ([module docs](self#attributes)).
    ///
    /// Binds [`igraph_write_graph_gml`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_write_graph_gml).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`] if `options.ids` doesn't have one entry per
    /// vertex, or the creator contains a NUL byte; [`ErrorKind::File`] on I/O
    /// errors.
    pub fn write_graph_gml(
        &self,
        path: impl AsRef<Path>,
        options: &GmlWriteOptions<'_>,
    ) -> Result<()> {
        let (opts, ids, creator) = self.gml_args(options)?;
        let ids_ptr = ids.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        // `ctime` (used for the default creator line) is not reentrant.
        let _guard = options.creator.is_none().then(gml_lock);
        write_to_path(path.as_ref(), |f| unsafe {
            igraph_write_graph_gml(self, f, opts, ids_ptr, opt_ptr(&creator))
        })
    }

    /// The GML serialization as a string, see
    /// [`write_graph_gml`](Self::write_graph_gml).
    ///
    /// # Examples
    /// ```
    /// use igraph::{foreign::GmlWriteOptions, prelude::*};
    /// let g = Graph::from_edges(&[(0, 1)], 2, false).unwrap();
    /// let opts = GmlWriteOptions::default().with_creator("").with_ids(&[10.0, 20.0]);
    /// let gml = g.write_graph_gml_to_string(&opts).unwrap();
    /// // igraph stores undirected edges with the larger endpoint first.
    /// assert!(gml.contains("id 10") && gml.contains("source 20") && gml.contains("target 10"));
    /// assert!(!gml.contains("Creator"));
    /// ```
    pub fn write_graph_gml_to_string(&self, options: &GmlWriteOptions<'_>) -> Result<String> {
        let (opts, ids, creator) = self.gml_args(options)?;
        let ids_ptr = ids.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        // `ctime` (used for the default creator line) is not reentrant.
        let _guard = options.creator.is_none().then(gml_lock);
        write_to_string(|f| unsafe {
            igraph_write_graph_gml(self, f, opts, ids_ptr, opt_ptr(&creator))
        })
    }

    /// Validates and converts the GML writer options.
    #[allow(clippy::type_complexity)]
    fn gml_args<'a>(
        &self,
        options: &GmlWriteOptions<'a>,
    ) -> Result<(
        igraph_write_gml_sw_t,
        Option<crate::vector::View<'a, Vector>>,
        Option<CString>,
    )> {
        if let Some(ids) = options.ids
            && ids.len() != self.vcount()
        {
            return Err(Error::invalid(format!(
                "the id vector has length {}, but the graph has {} vertices",
                ids.len(),
                self.vcount()
            )));
        }
        let flags = if options.encode_only_quot {
            IGRAPH_WRITE_GML_ENCODE_ONLY_QUOT_SW
        } else {
            IGRAPH_WRITE_GML_DEFAULT_SW
        } as igraph_write_gml_sw_t;
        // igraph 1.0.0 and 1.0.1 compute the entity-encoded creator but then
        // print the raw string, so a `"` would end the GML string early:
        // encode it here.
        let creator = options
            .creator
            .map(|c| gml_entity_encode(c, options.encode_only_quot));
        let creator = opt_cstring(creator.as_deref(), "the GML creator")?;
        Ok((flags, options.ids.map(Vector::view), creator))
    }

    /// Writes the graph to a Graphviz DOT file.
    ///
    /// The structure is written as `a -- b;` (`a -> b;` for directed
    /// graphs) lines after the list of vertices; igraph adds no layout or
    /// visualization information of its own. With the
    /// [attribute handler](crate::attributes::enable) on, the graph
    /// attributes are written in a `graph [ ... ]` block and every vertex and
    /// edge attribute in the `[ ... ]` block of its element (numbers as
    /// written by igraph, `NaN` included, booleans as 0/1 with a warning,
    /// strings and names quoted and escaped when needed). The format is meant
    /// for interoperability with Graphviz, not for data exchange. Time
    /// complexity: proportional to the output size.
    ///
    /// See [`Graph::layout_circle`] and the [`layout`](crate::layout) module
    /// to compute coordinates to add as attributes (e.g. `pos`).
    ///
    /// Binds [`igraph_write_graph_dot`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_write_graph_dot).
    ///
    /// # Errors
    /// [`ErrorKind::File`] if the file can't be created or written.
    pub fn write_graph_dot(&self, path: impl AsRef<Path>) -> Result<()> {
        write_to_path(path.as_ref(), |f| unsafe {
            igraph_write_graph_dot(self, f)
        })
    }

    /// The DOT serialization as a string, see [`write_graph_dot`](Self::write_graph_dot).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    /// let dot = g.write_graph_dot_to_string().unwrap();
    /// assert!(dot.starts_with("/* Created by igraph"));
    /// assert!(dot.contains("digraph {") && dot.contains("  0 -> 1;\n") && dot.contains("  1 -> 2;\n"));
    /// ```
    pub fn write_graph_dot_to_string(&self) -> Result<String> {
        write_to_string(|f| unsafe { igraph_write_graph_dot(self, f) })
    }

    /// Writes the graph in the LEDA native graph format.
    ///
    /// Only the graph section is written: the vertices, then the edges with
    /// 1-based endpoints (and, for undirected graphs, a `-2` direction flag).
    /// `vertex_attr`/`edge_attr` name one vertex and one edge attribute whose
    /// values are stored, with their LEDA type (`double`, `string` or `bool`)
    /// in the header; `void` is written for `None` and for attributes that
    /// don't exist (with a warning; always the case without the
    /// [attribute handler](crate::attributes::enable), see the
    /// [module docs](self#attributes)). Time complexity: O(|V| + |E|).
    ///
    /// Binds [`igraph_write_graph_leda`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_write_graph_leda).
    ///
    /// # Errors
    /// [`ErrorKind::File`] on I/O errors, [`ErrorKind::InvalidValue`] for
    /// attribute names with NUL bytes or string values containing a newline.
    pub fn write_graph_leda(
        &self,
        path: impl AsRef<Path>,
        vertex_attr: Option<&str>,
        edge_attr: Option<&str>,
    ) -> Result<()> {
        let (v, e) = (
            opt_cstring(vertex_attr, "vertex_attr")?,
            opt_cstring(edge_attr, "edge_attr")?,
        );
        write_to_path(path.as_ref(), |f| unsafe {
            igraph_write_graph_leda(self, f, opt_ptr(&v), opt_ptr(&e))
        })
    }

    /// The LEDA serialization as a string, see [`write_graph_leda`](Self::write_graph_leda).
    ///
    /// # Examples
    /// ```
    /// use igraph::prelude::*;
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, true).unwrap();
    /// let leda = g.write_graph_leda_to_string(None, None).unwrap();
    /// assert_eq!(
    ///     leda,
    ///     "LEDA.GRAPH\nvoid\nvoid\n-1\n# Vertices\n3\n|{}|\n|{}|\n|{}|\n# Edges\n3\n\
    ///      1 2 0 |{}|\n2 3 0 |{}|\n3 1 0 |{}|\n"
    /// );
    /// ```
    pub fn write_graph_leda_to_string(
        &self,
        vertex_attr: Option<&str>,
        edge_attr: Option<&str>,
    ) -> Result<String> {
        let (v, e) = (
            opt_cstring(vertex_attr, "vertex_attr")?,
            opt_cstring(edge_attr, "edge_attr")?,
        );
        write_to_string(|f| unsafe { igraph_write_graph_leda(self, f, opt_ptr(&v), opt_ptr(&e)) })
    }
}

// ---------------------------------------------------------------------------
// Locale
// ---------------------------------------------------------------------------

/// RAII guard temporarily switching the numeric locale to `"C"`
/// (`igraph_enter_safelocale` / `igraph_exit_safelocale`).
///
/// igraph's readers and writers need a locale with a decimal *point*. While
/// the guard is alive the `"C"` numeric locale is in effect for the current
/// thread (on platforms without per-thread locales igraph falls back to the
/// process-wide `setlocale`, which is not thread safe); dropping it restores
/// the previous locale. Rust programs start in the `"C"` locale, so this is
/// only needed when some other code changed it.
///
/// The guard is neither [`Send`] nor [`Sync`]: it must be dropped on the
/// thread that created it. Nested guards restore their locales correctly
/// when dropped in reverse order of creation (as scopes do).
///
/// Binds [`igraph_enter_safelocale`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_enter_safelocale)
/// and [`igraph_exit_safelocale`](https://igraph.org/c/html/latest/igraph-Foreign.html#igraph_exit_safelocale).
///
/// ```
/// use igraph::{foreign::SafeLocale, prelude::*};
/// let g = Graph::from_edges(&[(0, 1)], 2, true).unwrap();
/// let text = {
///     let _locale = SafeLocale::enter().unwrap();
///     g.write_graph_dimacs_flow_to_string(0, 1, &[0.5]).unwrap()
/// }; // previous locale restored here
/// assert!(text.ends_with("a 1 2 0.5\n"));
/// ```
#[derive(Debug)]
pub struct SafeLocale {
    loc: igraph_safelocale_t,
}

impl SafeLocale {
    /// Switches the current thread to the `"C"` numeric locale until the
    /// returned guard is dropped.
    ///
    /// # Errors
    /// [`ErrorKind::Failure`] if the `"C"` locale can't be created.
    pub fn enter() -> Result<Self> {
        let mut loc: igraph_safelocale_t = std::ptr::null_mut();
        if let Err(e) = igraph_call!(igraph_enter_safelocale(&mut loc)) {
            // igraph 1.0.0 and 1.0.1 return the error without freeing the
            // state they allocated (and never switched to): release it.
            if !loc.is_null() {
                unsafe { igraph_free(loc.cast()) };
            }
            return Err(e);
        }
        Ok(Self { loc })
    }
}

impl Drop for SafeLocale {
    fn drop(&mut self) {
        if !self.loc.is_null() {
            unsafe { igraph_exit_safelocale(&mut self.loc) };
        }
    }
}

/// Runs `f` with the `"C"` numeric locale in effect, see [`SafeLocale`].
///
/// # Errors
/// The errors of [`SafeLocale::enter`]; `f`'s own result is returned inside
/// the `Ok`.
///
/// ```
/// use igraph::{foreign::with_safe_locale, prelude::*};
/// let g = with_safe_locale(|| Graph::read_graph_edgelist_from_str("0 1", 0, false)).unwrap().unwrap();
/// assert_eq!(g.ecount(), 1);
/// ```
pub fn with_safe_locale<T>(f: impl FnOnce() -> T) -> Result<T> {
    let _guard = SafeLocale::enter()?;
    Ok(f())
}

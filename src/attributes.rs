//! Graph, vertex and edge attributes (`igraph_attributes.h`).
//!
//! igraph can attach *attributes* to a graph as a whole, to its vertices and
//! to its edges: a vertex `name`, an edge `weight`, a graph `title`, ... The
//! C core does not store attributes by itself: it notifies an *attribute
//! handler* (a table of callbacks, `igraph_attribute_table_t`) of every
//! structural change so that the handler can keep its data aligned with the
//! vertex and edge ids. This module plugs igraph's own C attribute handler
//! (`igraph_cattribute_table`, the one behind the `VAN`/`SETVAN`/... macros of
//! the C API) into the crate and exposes it with a typed, Rusty API on
//! [`Graph`].
//!
//! # Enabling attributes
//!
//! The attribute handler is a **process-wide** setting of the C library (it is
//! *not* thread-local, even in a thread-safe igraph build, see
//! `src/graph/attributes.c`). It is turned on with [`enable`]:
//!
//! - it is idempotent and irreversible: once enabled, attributes stay enabled
//!   for the rest of the process (detaching a handler would make igraph leak
//!   the attribute storage and let it get out of sync with the graphs);
//! - it is also called implicitly by every attribute *setter*, so writing an
//!   attribute simply works;
//! - it must be called *before reading graphs from files* if you want the
//!   foreign readers (GraphML, GML, Pajek, NCOL, LGL, ...) to keep vertex
//!   names, edge weights and the other attributes found in the file, and
//!   before *writing* them if you want the writers to emit your attributes;
//! - graphs created before [`enable`] carry no attribute storage: they keep
//!   working (this crate installs a *guarded* version of the C handler that
//!   ignores attribute-less graphs instead of crashing, as the raw C handler
//!   would), and their storage is created lazily by the first setter call;
//! - call it early (e.g. at the start of `main`): the C library stores the
//!   handler in a plain global variable, so enabling it while other threads
//!   are in the middle of igraph calls is best avoided.
//!
//! # Semantics of the C attribute handler
//!
//! - Attribute values are numbers (`f64`), booleans or strings
//!   ([`AttributeType`]); an attribute has a single type for all the
//!   vertices (or edges) of a graph, fixed when it is first created. Writing a
//!   value of another type is an error.
//! - Setting a vertex (edge) attribute on a single vertex (edge) creates it
//!   for all the vertices (edges), with the default value (`NaN`, `false` or
//!   `""`) everywhere else. Vertices and edges added later also get the
//!   default value.
//! - Attributes follow the structure: deleting vertices or edges
//!   ([`Graph::delete_vertices`], [`Graph::delete_edges`]), taking subgraphs
//!   ([`Graph::induced_subgraph`]), permuting vertices
//!   ([`Graph::permute_vertices`]), copying (and [`Clone`]-ing) a graph keep
//!   every value attached to the right vertex or edge.
//! - Operations that merge vertices or edges decide what to do with the
//!   attributes of the merged elements via an [`AttributeCombination`]. The
//!   plain wrappers [`Graph::simplify`], [`Graph::contract_vertices`] and
//!   [`Graph::to_undirected`] pass *no* combination, so they drop the
//!   attributes of the kind they merge: vertex attributes for
//!   `contract_vertices`; edge attributes for `to_undirected` (except in
//!   [`ToUndirected::Each`] mode) and for `simplify` when it actually merges
//!   multi-edges (when it only deletes loops, or igraph already knows there
//!   are no multi-edges, the remaining edges keep their attributes). Use
//!   their attribute-aware twins
//!   [`Graph::simplify_with_attributes`],
//!   [`Graph::contract_vertices_with_attributes`] and
//!   [`Graph::to_undirected_with_attributes`] to keep them.
//! - The [`Random`](AttributeCombinationType::Random) combination and the
//!   tie-breaking of boolean majority votes use the default random number
//!   generator of the calling thread: seed it with [`rng::seed`](crate::rng::seed)
//!   for reproducible results.
//!
//! # Example
//!
//! ```
//! use igraph::prelude::*;
//! use igraph::attributes::{self, AttributeKind, AttributeValue};
//!
//! attributes::enable().unwrap();
//! let mut g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, false).unwrap();
//!
//! g.set_graph_attr_str("title", "triangle").unwrap();
//! g.set_vertex_attr_str_values("name", &["alice", "bob", "carol"]).unwrap();
//! g.set_edge_attr_numeric_values("weight", &[1.0, 2.5, 4.0]).unwrap();
//! g.set_vertex_attr_bool("admin", 0, true).unwrap();
//!
//! assert_eq!(g.graph_attr_str("title").unwrap(), "triangle");
//! assert_eq!(g.vertex_attr_str("name", 1).unwrap(), "bob");
//! assert_eq!(g.vertex_attr("admin", 2).unwrap(), AttributeValue::Boolean(false));
//! assert!(g.has_attribute(AttributeKind::Vertex, "name"));
//!
//! // Numeric edge attributes are the weight vectors of weighted algorithms.
//! let w = g.edge_attr_numeric_values("weight", ..).unwrap();
//! assert_eq!(w, vec![1.0, 2.5, 4.0]);
//! let strength = g.strength(.., NeighborMode::All, Loops::Twice, Some(&w)).unwrap();
//! assert_eq!(strength, vec![5.0, 3.5, 6.5]);
//!
//! // Attributes follow the structure of the graph.
//! g.delete_vertices(0).unwrap();
//! assert_eq!(g.vertex_attr_str_values("name", ..).unwrap(), vec!["bob", "carol"]);
//! assert_eq!(g.edge_attr_numeric_values("weight", ..).unwrap(), vec![2.5]);
//! ```
//!
//! # Provided functionality
//!
//! | Rust | C |
//! |------|---|
//! | [`enable`], [`is_enabled`], [`has_attribute_table`] | `igraph_set_attribute_table(&igraph_cattribute_table)`, `igraph_has_attribute_table` |
//! | [`Graph::graph_attr_numeric`], [`Graph::graph_attr_bool`], [`Graph::graph_attr_str`], [`Graph::graph_attr`] | `igraph_cattribute_GAN`, `GAB`, `GAS` |
//! | [`Graph::vertex_attr_numeric`], [`Graph::vertex_attr_bool`], [`Graph::vertex_attr_str`], [`Graph::vertex_attr`] | `igraph_cattribute_VAN`, `VAB`, `VAS` |
//! | [`Graph::edge_attr_numeric`], [`Graph::edge_attr_bool`], [`Graph::edge_attr_str`], [`Graph::edge_attr`] | `igraph_cattribute_EAN`, `EAB`, `EAS` |
//! | [`Graph::vertex_attr_numeric_values`], [`Graph::vertex_attr_bool_values`], [`Graph::vertex_attr_str_values`], [`Graph::vertex_attr_values`] | `igraph_cattribute_VANV`, `VABV`, `VASV` |
//! | [`Graph::edge_attr_numeric_values`], [`Graph::edge_attr_bool_values`], [`Graph::edge_attr_str_values`], [`Graph::edge_attr_values`] | `igraph_cattribute_EANV`, `EABV`, `EASV` |
//! | [`Graph::set_graph_attr_numeric`], [`Graph::set_graph_attr_bool`], [`Graph::set_graph_attr_str`], [`Graph::set_graph_attr`] | `igraph_cattribute_GAN_set`, `GAB_set`, `GAS_set` |
//! | [`Graph::set_vertex_attr_numeric`], [`Graph::set_vertex_attr_bool`], [`Graph::set_vertex_attr_str`], [`Graph::set_vertex_attr`] | `igraph_cattribute_VAN_set`, `VAB_set`, `VAS_set` |
//! | [`Graph::set_edge_attr_numeric`], [`Graph::set_edge_attr_bool`], [`Graph::set_edge_attr_str`], [`Graph::set_edge_attr`] | `igraph_cattribute_EAN_set`, `EAB_set`, `EAS_set` |
//! | [`Graph::set_vertex_attr_numeric_values`], [`Graph::set_vertex_attr_bool_values`], [`Graph::set_vertex_attr_str_values`], [`Graph::set_vertex_attr_values`] | `igraph_cattribute_VAN_setv`, `VAB_setv`, `VAS_setv` |
//! | [`Graph::set_edge_attr_numeric_values`], [`Graph::set_edge_attr_bool_values`], [`Graph::set_edge_attr_str_values`], [`Graph::set_edge_attr_values`] | `igraph_cattribute_EAN_setv`, `EAB_setv`, `EAS_setv` |
//! | [`Graph::attribute_list`], [`Graph::attribute_names`], [`Graph::attribute_type`] | `igraph_cattribute_list` |
//! | [`Graph::has_attribute`] | `igraph_cattribute_has_attr` |
//! | [`Graph::remove_graph_attr`], [`Graph::remove_vertex_attr`], [`Graph::remove_edge_attr`], [`Graph::remove_all_attributes`] | `igraph_cattribute_remove_g`, `remove_v`, `remove_e`, `remove_all` |
//! | [`Graph::add_vertices_with_attributes`], [`Graph::add_edges_with_attributes`] | `igraph_add_vertices`, `igraph_add_edges` with an attribute record list |
//! | [`Graph::simplify_with_attributes`], [`Graph::contract_vertices_with_attributes`], [`Graph::to_undirected_with_attributes`] | `igraph_simplify`, `igraph_contract_vertices`, `igraph_to_undirected` with an attribute combination |
//! | [`AttributeRecord`] | `igraph_attribute_record_*` |
//! | [`AttributeCombination`] | `igraph_attribute_combination_*` |
//!
//! # See also
//!
//! - [`crate::foreign`]: the GraphML, GML, Pajek, NCOL, LGL, ... readers and
//!   writers load and save attributes once [`enable`] has been called, e.g.
//!   [`Graph::read_graph_graphml_from_str`] and
//!   [`Graph::write_graph_graphml_to_string`].
//! - [`crate::operators`] and [`crate::conversion`]: the structural
//!   operations whose attribute handling is described above.
//! - Weighted algorithms take the weights as a slice: feed them
//!   `g.edge_attr_numeric_values("weight", ..)`, e.g. [`Graph::strength`],
//!   [`Graph::distances_dijkstra`] or [`Graph::community_multilevel`].

#[cfg(doc)]
use crate::graph::Graph;
use crate::{
    constants::ToUndirected,
    error::{Error, ErrorKind, Result, ensure_init},
    ffi::*,
    graph::{EdgeId, VertexId},
    igraph_call,
    selector::{EdgeSelector, VertexSelector},
    strvector::StrVector,
    vector::{Vector, VectorBool, VectorInt},
};
use std::{
    ffi::{CStr, CString, c_char},
    fmt,
    mem::MaybeUninit,
    ptr,
    sync::{
        Once,
        atomic::{AtomicBool, Ordering},
    },
};

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

crate::ffi_enum! {
    /// The type of an attribute (`igraph_attribute_type_t`).
    ///
    /// The C attribute handler supports [`Numeric`](Self::Numeric),
    /// [`Boolean`](Self::Boolean) and [`String`](Self::String) attributes;
    /// the other two variants exist for completeness (they are used by the
    /// attribute handlers of the high-level igraph interfaces).
    pub enum AttributeType: igraph_attribute_type_t {
        /// No type yet (a freshly initialized, empty [`AttributeRecord`]).
        Unspecified = igraph_attribute_type_t_IGRAPH_ATTRIBUTE_UNSPECIFIED,
        /// Real numbers (`f64`).
        Numeric = igraph_attribute_type_t_IGRAPH_ATTRIBUTE_NUMERIC,
        /// Booleans.
        Boolean = igraph_attribute_type_t_IGRAPH_ATTRIBUTE_BOOLEAN,
        /// Strings.
        String = igraph_attribute_type_t_IGRAPH_ATTRIBUTE_STRING,
        /// Opaque objects of a high-level language (unsupported by the C handler).
        Object = igraph_attribute_type_t_IGRAPH_ATTRIBUTE_OBJECT,
    }
}

impl fmt::Display for AttributeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Unspecified => "unspecified",
            Self::Numeric => "numeric",
            Self::Boolean => "boolean",
            Self::String => "string",
            Self::Object => "object",
        })
    }
}

crate::ffi_enum! {
    /// What an attribute is attached to (`igraph_attribute_elemtype_t`).
    pub enum AttributeKind: igraph_attribute_elemtype_t {
        /// The graph as a whole.
        Graph = igraph_attribute_elemtype_t_IGRAPH_ATTRIBUTE_GRAPH,
        /// Each vertex.
        Vertex = igraph_attribute_elemtype_t_IGRAPH_ATTRIBUTE_VERTEX,
        /// Each edge.
        Edge = igraph_attribute_elemtype_t_IGRAPH_ATTRIBUTE_EDGE,
    }
}

impl fmt::Display for AttributeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Graph => "graph",
            Self::Vertex => "vertex",
            Self::Edge => "edge",
        })
    }
}

crate::ffi_enum! {
    /// How to combine the attribute values of vertices or edges that are
    /// merged into one (`igraph_attribute_combination_type_t`).
    ///
    /// The C attribute handler supports the following combinations, per
    /// attribute type (anything else fails with
    /// [`ErrorKind::AttributeCombination`] or [`ErrorKind::Unimplemented`]):
    ///
    /// | combination | numeric | boolean | string |
    /// |-------------|---------|---------|--------|
    /// | `Sum`       | sum     | any is true | – |
    /// | `Prod`      | product | all are true | – |
    /// | `Min`/`Max` | min/max | all/any are true | – |
    /// | `Mean`      | mean    | majority (ties broken at random) | – |
    /// | `Median`    | – (unimplemented) | majority (ties broken at random) | – |
    /// | `Random`, `First`, `Last` | yes | yes | yes |
    /// | `Concat`    | – | – | concatenation |
    /// | `Function`  | yes | yes | – (not supported by this crate) |
    ///
    /// `Ignore` and `Default` drop the attribute for every type.
    pub enum AttributeCombinationType: igraph_attribute_combination_type_t {
        /// Drop the attribute from the result.
        Ignore = igraph_attribute_combination_type_t_IGRAPH_ATTRIBUTE_COMBINE_IGNORE,
        /// The default behaviour of the handler: for the C handler, the
        /// attribute is dropped, like [`Ignore`](Self::Ignore).
        Default = igraph_attribute_combination_type_t_IGRAPH_ATTRIBUTE_COMBINE_DEFAULT,
        /// A user supplied C function, see [`AttributeCombination::add_function`].
        Function = igraph_attribute_combination_type_t_IGRAPH_ATTRIBUTE_COMBINE_FUNCTION,
        /// Sum of the values (logical *or* for booleans).
        Sum = igraph_attribute_combination_type_t_IGRAPH_ATTRIBUTE_COMBINE_SUM,
        /// Product of the values (logical *and* for booleans).
        Prod = igraph_attribute_combination_type_t_IGRAPH_ATTRIBUTE_COMBINE_PROD,
        /// Minimum of the values (logical *and* for booleans).
        Min = igraph_attribute_combination_type_t_IGRAPH_ATTRIBUTE_COMBINE_MIN,
        /// Maximum of the values (logical *or* for booleans).
        Max = igraph_attribute_combination_type_t_IGRAPH_ATTRIBUTE_COMBINE_MAX,
        /// A value chosen uniformly at random (uses igraph's RNG).
        Random = igraph_attribute_combination_type_t_IGRAPH_ATTRIBUTE_COMBINE_RANDOM,
        /// The value of the first element of the merged group (in the order
        /// the merging function lists them, usually increasing id).
        First = igraph_attribute_combination_type_t_IGRAPH_ATTRIBUTE_COMBINE_FIRST,
        /// The value of the last element of the merged group (usually the
        /// highest id).
        Last = igraph_attribute_combination_type_t_IGRAPH_ATTRIBUTE_COMBINE_LAST,
        /// Arithmetic mean (majority vote for booleans, with ties broken at
        /// random using igraph's RNG).
        Mean = igraph_attribute_combination_type_t_IGRAPH_ATTRIBUTE_COMBINE_MEAN,
        /// Median (majority vote for booleans; not implemented for numbers).
        Median = igraph_attribute_combination_type_t_IGRAPH_ATTRIBUTE_COMBINE_MEDIAN,
        /// Concatenation of the strings.
        ///
        /// **Note:** in igraph 1.0.0 and 1.0.1 the C handler
        /// (`igraph_i_cattributes_cs_concat`) concatenates the values of the
        /// *first* `k` elements of the graph (where `k` is the size of the
        /// merged group) instead of the values of the merged elements.
        Concat = igraph_attribute_combination_type_t_IGRAPH_ATTRIBUTE_COMBINE_CONCAT,
    }
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

/// A single attribute value, as stored by the C attribute handler.
///
/// It converts from `f64`, `i32`, `i64`, `bool`, `&str` and `String`, so the
/// generic setters such as [`Graph::set_vertex_attr`] accept those directly.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    /// A numeric value.
    Numeric(f64),
    /// A boolean value.
    Boolean(bool),
    /// A string value.
    String(String),
}

impl AttributeValue {
    /// The [`AttributeType`] of this value.
    pub fn attribute_type(&self) -> AttributeType {
        match self {
            Self::Numeric(_) => AttributeType::Numeric,
            Self::Boolean(_) => AttributeType::Boolean,
            Self::String(_) => AttributeType::String,
        }
    }

    /// The number, if this is a numeric value.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Numeric(x) => Some(*x),
            _ => None,
        }
    }

    /// The boolean, if this is a boolean value.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// The string, if this is a string value.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }
}

impl fmt::Display for AttributeValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Numeric(x) => write!(f, "{x}"),
            Self::Boolean(b) => write!(f, "{b}"),
            Self::String(s) => write!(f, "{s:?}"),
        }
    }
}

impl From<f64> for AttributeValue {
    fn from(x: f64) -> Self {
        Self::Numeric(x)
    }
}

impl From<i32> for AttributeValue {
    fn from(x: i32) -> Self {
        Self::Numeric(x.into())
    }
}

impl From<i64> for AttributeValue {
    /// Converts to a numeric value (precision is lost beyond 2^53).
    fn from(x: i64) -> Self {
        Self::Numeric(x as f64)
    }
}

impl From<bool> for AttributeValue {
    fn from(b: bool) -> Self {
        Self::Boolean(b)
    }
}

impl From<&str> for AttributeValue {
    fn from(s: &str) -> Self {
        Self::String(s.to_owned())
    }
}

impl From<String> for AttributeValue {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

/// The values of an attribute for several vertices or edges.
///
/// Converts from `Vec<f64>`, `Vec<bool>`, `Vec<String>`, `Vec<&str>` and the
/// corresponding slices, so that [`Graph::set_vertex_attr_values`] accepts
/// them directly.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValues {
    /// Numeric values.
    Numeric(Vec<f64>),
    /// Boolean values.
    Boolean(Vec<bool>),
    /// String values.
    String(Vec<String>),
}

impl AttributeValues {
    /// The [`AttributeType`] of the values.
    pub fn attribute_type(&self) -> AttributeType {
        match self {
            Self::Numeric(_) => AttributeType::Numeric,
            Self::Boolean(_) => AttributeType::Boolean,
            Self::String(_) => AttributeType::String,
        }
    }

    /// Number of values.
    pub fn len(&self) -> usize {
        match self {
            Self::Numeric(v) => v.len(),
            Self::Boolean(v) => v.len(),
            Self::String(v) => v.len(),
        }
    }

    /// Whether there are no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The value at `index`, if any.
    pub fn get(&self, index: usize) -> Option<AttributeValue> {
        match self {
            Self::Numeric(v) => v.get(index).copied().map(AttributeValue::Numeric),
            Self::Boolean(v) => v.get(index).copied().map(AttributeValue::Boolean),
            Self::String(v) => v.get(index).cloned().map(AttributeValue::String),
        }
    }

    /// The numbers, if these are numeric values.
    pub fn as_numeric(&self) -> Option<&[f64]> {
        match self {
            Self::Numeric(v) => Some(v),
            _ => None,
        }
    }

    /// The booleans, if these are boolean values.
    pub fn as_bool(&self) -> Option<&[bool]> {
        match self {
            Self::Boolean(v) => Some(v),
            _ => None,
        }
    }

    /// The strings, if these are string values.
    pub fn as_strings(&self) -> Option<&[String]> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }
}

impl From<Vec<f64>> for AttributeValues {
    fn from(v: Vec<f64>) -> Self {
        Self::Numeric(v)
    }
}

impl From<&[f64]> for AttributeValues {
    fn from(v: &[f64]) -> Self {
        Self::Numeric(v.to_vec())
    }
}

impl From<Vec<bool>> for AttributeValues {
    fn from(v: Vec<bool>) -> Self {
        Self::Boolean(v)
    }
}

impl From<&[bool]> for AttributeValues {
    fn from(v: &[bool]) -> Self {
        Self::Boolean(v.to_vec())
    }
}

impl From<Vec<String>> for AttributeValues {
    fn from(v: Vec<String>) -> Self {
        Self::String(v)
    }
}

impl From<Vec<&str>> for AttributeValues {
    fn from(v: Vec<&str>) -> Self {
        Self::String(v.into_iter().map(str::to_owned).collect())
    }
}

impl From<&[&str]> for AttributeValues {
    fn from(v: &[&str]) -> Self {
        Self::String(v.iter().map(|s| (*s).to_owned()).collect())
    }
}

/// Names and types of all the attributes of a graph, see [`Graph::attribute_list`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AttributeList {
    /// Graph attributes, in creation order.
    pub graph: Vec<(String, AttributeType)>,
    /// Vertex attributes, in creation order.
    pub vertex: Vec<(String, AttributeType)>,
    /// Edge attributes, in creation order.
    pub edge: Vec<(String, AttributeType)>,
}

impl AttributeList {
    /// The attributes of the given kind.
    pub fn of(&self, kind: AttributeKind) -> &[(String, AttributeType)] {
        match kind {
            AttributeKind::Graph => &self.graph,
            AttributeKind::Vertex => &self.vertex,
            AttributeKind::Edge => &self.edge,
        }
    }

    /// Whether the graph has no attributes at all.
    pub fn is_empty(&self) -> bool {
        self.graph.is_empty() && self.vertex.is_empty() && self.edge.is_empty()
    }
}

// ---------------------------------------------------------------------------
// The attribute handler
// ---------------------------------------------------------------------------

/// Fetches a callback of the C attribute handler `igraph_cattribute_table`.
macro_rules! c_handler {
    ($field:ident) => {
        // SAFETY: `igraph_cattribute_table` is an immutable, statically
        // initialized C constant.
        unsafe { igraph_cattribute_table.$field }
    };
}

/// Reports an error from inside a callback, like `IGRAPH_ERROR` in C.
fn raise(reason: &'static CStr, code: igraph_error_t) -> igraph_error_t {
    unsafe {
        igraph_error(
            reason.as_ptr(),
            c"attributes.rs".as_ptr(),
            line!() as i32,
            code,
        )
    }
}

fn missing_callback() -> igraph_error_t {
    raise(
        c"The C attribute handler lacks a callback.",
        igraph_error_type_t_IGRAPH_FAILURE,
    )
}

fn no_such_attribute() -> igraph_error_t {
    raise(
        c"Attribute does not exist (the graph carries no attributes).",
        igraph_error_type_t_IGRAPH_EINVAL,
    )
}

// The guarded handler: igraph's C attribute handler dereferences `graph->attr`
// unconditionally in most callbacks, which would crash on graphs created
// before the handler was attached (their `attr` is NULL). The callbacks below
// delegate to the C handler for graphs that carry attribute storage and treat
// attribute-less graphs as graphs without attributes.

unsafe extern "C" fn g_init(
    graph: *mut igraph_t,
    attr: *const igraph_attribute_record_list_t,
) -> igraph_error_t {
    match c_handler!(init) {
        Some(f) => unsafe { f(graph, attr) },
        None => missing_callback(),
    }
}

unsafe extern "C" fn g_destroy(graph: *mut igraph_t) {
    if unsafe { !(*graph).attr.is_null() }
        && let Some(f) = c_handler!(destroy)
    {
        unsafe { f(graph) }
    }
}

unsafe extern "C" fn g_copy(
    to: *mut igraph_t,
    from: *const igraph_t,
    ga: igraph_bool_t,
    va: igraph_bool_t,
    ea: igraph_bool_t,
) -> igraph_error_t {
    if unsafe { (*from).attr.is_null() } {
        return igraph_error_type_t_IGRAPH_SUCCESS;
    }
    match c_handler!(copy) {
        Some(f) => unsafe { f(to, from, ga, va, ea) },
        None => missing_callback(),
    }
}

unsafe extern "C" fn g_add_vertices(
    graph: *mut igraph_t,
    nv: igraph_int_t,
    attr: *const igraph_attribute_record_list_t,
) -> igraph_error_t {
    if unsafe { (*graph).attr.is_null() } {
        return igraph_error_type_t_IGRAPH_SUCCESS;
    }
    match c_handler!(add_vertices) {
        Some(f) => unsafe { f(graph, nv, attr) },
        None => missing_callback(),
    }
}

unsafe extern "C" fn g_add_edges(
    graph: *mut igraph_t,
    edges: *const igraph_vector_int_t,
    attr: *const igraph_attribute_record_list_t,
) -> igraph_error_t {
    if unsafe { (*graph).attr.is_null() } {
        return igraph_error_type_t_IGRAPH_SUCCESS;
    }
    match c_handler!(add_edges) {
        Some(f) => unsafe { f(graph, edges, attr) },
        None => missing_callback(),
    }
}

unsafe fn both_have_storage(graph: *const igraph_t, newgraph: *const igraph_t) -> bool {
    unsafe { !(*graph).attr.is_null() && !(*newgraph).attr.is_null() }
}

unsafe extern "C" fn g_permute_vertices(
    graph: *const igraph_t,
    newgraph: *mut igraph_t,
    idx: *const igraph_vector_int_t,
) -> igraph_error_t {
    if unsafe { !both_have_storage(graph, newgraph) } {
        return igraph_error_type_t_IGRAPH_SUCCESS;
    }
    match c_handler!(permute_vertices) {
        Some(f) => unsafe { f(graph, newgraph, idx) },
        None => missing_callback(),
    }
}

unsafe extern "C" fn g_permute_edges(
    graph: *const igraph_t,
    newgraph: *mut igraph_t,
    idx: *const igraph_vector_int_t,
) -> igraph_error_t {
    if unsafe { !both_have_storage(graph, newgraph) } {
        return igraph_error_type_t_IGRAPH_SUCCESS;
    }
    match c_handler!(permute_edges) {
        Some(f) => unsafe { f(graph, newgraph, idx) },
        None => missing_callback(),
    }
}

unsafe extern "C" fn g_combine_vertices(
    graph: *const igraph_t,
    newgraph: *mut igraph_t,
    merges: *const igraph_vector_int_list_t,
    comb: *const igraph_attribute_combination_t,
) -> igraph_error_t {
    if unsafe { !both_have_storage(graph, newgraph) } {
        return igraph_error_type_t_IGRAPH_SUCCESS;
    }
    match c_handler!(combine_vertices) {
        Some(f) => unsafe { f(graph, newgraph, merges, comb) },
        None => missing_callback(),
    }
}

unsafe extern "C" fn g_combine_edges(
    graph: *const igraph_t,
    newgraph: *mut igraph_t,
    merges: *const igraph_vector_int_list_t,
    comb: *const igraph_attribute_combination_t,
) -> igraph_error_t {
    if unsafe { !both_have_storage(graph, newgraph) } {
        return igraph_error_type_t_IGRAPH_SUCCESS;
    }
    match c_handler!(combine_edges) {
        Some(f) => unsafe { f(graph, newgraph, merges, comb) },
        None => missing_callback(),
    }
}

unsafe extern "C" fn g_get_info(
    graph: *const igraph_t,
    gnames: *mut igraph_strvector_t,
    gtypes: *mut igraph_vector_int_t,
    vnames: *mut igraph_strvector_t,
    vtypes: *mut igraph_vector_int_t,
    enames: *mut igraph_strvector_t,
    etypes: *mut igraph_vector_int_t,
) -> igraph_error_t {
    if unsafe { (*graph).attr.is_null() } {
        for names in [gnames, vnames, enames] {
            if !names.is_null() {
                unsafe { igraph_strvector_clear(names) };
            }
        }
        for types in [gtypes, vtypes, etypes] {
            if !types.is_null() {
                unsafe { igraph_vector_int_clear(types) };
            }
        }
        return igraph_error_type_t_IGRAPH_SUCCESS;
    }
    match c_handler!(get_info) {
        Some(f) => unsafe { f(graph, gnames, gtypes, vnames, vtypes, enames, etypes) },
        None => missing_callback(),
    }
}

unsafe extern "C" fn g_has_attr(
    graph: *const igraph_t,
    kind: igraph_attribute_elemtype_t,
    name: *const c_char,
) -> igraph_bool_t {
    if unsafe { (*graph).attr.is_null() } {
        return false;
    }
    match c_handler!(has_attr) {
        Some(f) => unsafe { f(graph, kind, name) },
        None => false,
    }
}

unsafe extern "C" fn g_get_type(
    graph: *const igraph_t,
    type_: *mut igraph_attribute_type_t,
    elemtype: igraph_attribute_elemtype_t,
    name: *const c_char,
) -> igraph_error_t {
    if unsafe { (*graph).attr.is_null() } {
        return no_such_attribute();
    }
    match c_handler!(get_type) {
        Some(f) => unsafe { f(graph, type_, elemtype, name) },
        None => missing_callback(),
    }
}

macro_rules! guarded_graph_getter {
    ($name:ident, $field:ident, $out:ty) => {
        unsafe extern "C" fn $name(
            graph: *const igraph_t,
            name: *const c_char,
            value: *mut $out,
        ) -> igraph_error_t {
            if unsafe { (*graph).attr.is_null() } {
                return no_such_attribute();
            }
            match c_handler!($field) {
                Some(f) => unsafe { f(graph, name, value) },
                None => missing_callback(),
            }
        }
    };
}

macro_rules! guarded_elem_getter {
    ($name:ident, $field:ident, $sel:ty, $out:ty) => {
        unsafe extern "C" fn $name(
            graph: *const igraph_t,
            name: *const c_char,
            sel: $sel,
            value: *mut $out,
        ) -> igraph_error_t {
            if unsafe { (*graph).attr.is_null() } {
                return no_such_attribute();
            }
            match c_handler!($field) {
                Some(f) => unsafe { f(graph, name, sel, value) },
                None => missing_callback(),
            }
        }
    };
}

guarded_graph_getter!(g_get_num_g, get_numeric_graph_attr, igraph_vector_t);
guarded_graph_getter!(g_get_str_g, get_string_graph_attr, igraph_strvector_t);
guarded_graph_getter!(g_get_bool_g, get_bool_graph_attr, igraph_vector_bool_t);
guarded_elem_getter!(
    g_get_num_v,
    get_numeric_vertex_attr,
    igraph_vs_t,
    igraph_vector_t
);
guarded_elem_getter!(
    g_get_str_v,
    get_string_vertex_attr,
    igraph_vs_t,
    igraph_strvector_t
);
guarded_elem_getter!(
    g_get_bool_v,
    get_bool_vertex_attr,
    igraph_vs_t,
    igraph_vector_bool_t
);
guarded_elem_getter!(
    g_get_num_e,
    get_numeric_edge_attr,
    igraph_es_t,
    igraph_vector_t
);
guarded_elem_getter!(
    g_get_str_e,
    get_string_edge_attr,
    igraph_es_t,
    igraph_strvector_t
);
guarded_elem_getter!(
    g_get_bool_e,
    get_bool_edge_attr,
    igraph_es_t,
    igraph_vector_bool_t
);

/// igraph's C attribute handler, guarded against attribute-less graphs.
static GUARDED_TABLE: igraph_attribute_table_t = igraph_attribute_table_t {
    init: Some(g_init),
    destroy: Some(g_destroy),
    copy: Some(g_copy),
    add_vertices: Some(g_add_vertices),
    permute_vertices: Some(g_permute_vertices),
    combine_vertices: Some(g_combine_vertices),
    add_edges: Some(g_add_edges),
    permute_edges: Some(g_permute_edges),
    combine_edges: Some(g_combine_edges),
    get_info: Some(g_get_info),
    has_attr: Some(g_has_attr),
    get_type: Some(g_get_type),
    get_numeric_graph_attr: Some(g_get_num_g),
    get_string_graph_attr: Some(g_get_str_g),
    get_bool_graph_attr: Some(g_get_bool_g),
    get_numeric_vertex_attr: Some(g_get_num_v),
    get_string_vertex_attr: Some(g_get_str_v),
    get_bool_vertex_attr: Some(g_get_bool_v),
    get_numeric_edge_attr: Some(g_get_num_e),
    get_string_edge_attr: Some(g_get_str_e),
    get_bool_edge_attr: Some(g_get_bool_e),
};

static ENABLE: Once = Once::new();
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Turns on igraph's C attribute handler for the whole process.
///
/// This attaches igraph's C attribute handler (`igraph_cattribute_table`,
/// in a variant guarded against graphs without attribute storage) with
/// [`igraph_set_attribute_table`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_set_attribute_table).
/// From then on every newly created graph carries attribute storage, and
/// igraph keeps graph, vertex and edge attributes in sync with every
/// structural change.
///
/// - It is idempotent (cheap after the first call) and **irreversible**.
/// - The setting is process-wide, not per thread: call it early, ideally at
///   the start of `main`, before other threads use igraph.
/// - Attribute setters call it implicitly; call it explicitly before reading
///   graphs from files whose attributes (vertex names, edge weights, ...) you
///   want to keep.
/// - Graphs created before enabling keep working; they simply have no
///   attributes until the first setter creates their storage.
///
/// # Errors
///
/// [`ErrorKind::Exists`] if a different attribute handler was attached
/// beforehand through the raw FFI; that handler is left in place.
///
/// # Examples
///
/// ```
/// use igraph::attributes;
/// attributes::enable().unwrap();
/// attributes::enable().unwrap(); // no-op
/// assert!(attributes::is_enabled());
/// assert!(attributes::has_attribute_table());
/// ```
pub fn enable() -> Result<()> {
    ensure_init();
    ENABLE.call_once(|| {
        let ours: *const igraph_attribute_table_t = &GUARDED_TABLE;
        let old = unsafe { igraph_set_attribute_table(ours) };
        if old.is_null() || ptr::eq(old, ours) {
            ENABLED.store(true, Ordering::Release);
        } else {
            // Someone attached another handler: put it back.
            unsafe { igraph_set_attribute_table(old) };
        }
    });
    if is_enabled() {
        Ok(())
    } else {
        Err(Error::new(
            ErrorKind::Exists,
            "another attribute handler was attached through the raw FFI",
        ))
    }
}

/// Whether this crate's attribute handler is attached (see [`enable`]).
pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Acquire)
}

/// Whether *some* attribute handler is attached to igraph
/// ([`igraph_has_attribute_table`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_has_attribute_table)).
///
/// This is `true` after [`enable`], but also when a handler was attached
/// through the raw FFI.
pub fn has_attribute_table() -> bool {
    unsafe { igraph_has_attribute_table() }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn c_string(what: &str, s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error::invalid(format!("the {what} {s:?} contains a NUL byte")))
}

fn missing(kind: AttributeKind, name: &str) -> Error {
    Error::invalid(format!("{kind} attribute '{name}' does not exist"))
}

fn lossy(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

fn to_strvector<S: AsRef<str>>(values: &[S]) -> Result<StrVector> {
    let mut sv = StrVector::new();
    for s in values {
        let s = s.as_ref();
        if s.contains('\0') {
            return Err(Error::invalid(format!(
                "the string value {s:?} contains a NUL byte"
            )));
        }
        sv.push(s);
    }
    Ok(sv)
}

fn raw_type(raw: igraph_int_t) -> AttributeType {
    AttributeType::try_from(raw as igraph_attribute_type_t).unwrap_or(AttributeType::Object)
}

/// An owned `igraph_attribute_record_list_t`, used to hand records to igraph.
struct RecordList(igraph_attribute_record_list_t);

impl RecordList {
    fn from_records(records: &[AttributeRecord]) -> Result<Self> {
        let mut raw = MaybeUninit::<igraph_attribute_record_list_t>::uninit();
        igraph_call!(igraph_attribute_record_list_init(raw.as_mut_ptr(), 0))?;
        let mut list = RecordList(unsafe { raw.assume_init() });
        for rec in records {
            if rec.name().is_none() {
                return Err(Error::invalid("attribute records must have a name"));
            }
            // igraph's C handler aborts (fatal error) on untyped records.
            if !matches!(
                rec.attribute_type(),
                AttributeType::Numeric | AttributeType::Boolean | AttributeType::String
            ) {
                return Err(Error::invalid(format!(
                    "attribute record '{}' has no supported type",
                    rec.name().unwrap_or_default()
                )));
            }
            igraph_call!(igraph_attribute_record_list_push_back_copy(
                &mut list.0,
                rec
            ))?;
        }
        Ok(list)
    }
}

impl Drop for RecordList {
    fn drop(&mut self) {
        unsafe { igraph_attribute_record_list_destroy(&mut self.0) };
    }
}

// ---------------------------------------------------------------------------
// Graph methods
// ---------------------------------------------------------------------------

impl igraph_t {
    /// Whether this graph can be read through the C attribute handler.
    fn has_attr_storage(&self) -> bool {
        is_enabled() && !self.attr.is_null()
    }

    /// Enables attributes and makes sure this graph has attribute storage.
    fn ensure_attr_storage(&mut self) -> Result<()> {
        enable()?;
        if self.attr.is_null() {
            igraph_call!(g_init(self, ptr::null()))?;
        }
        Ok(())
    }

    fn check_vid(&self, vid: VertexId) -> Result<()> {
        if vid < 0 || vid as usize >= self.vcount() {
            return Err(Error::new(
                ErrorKind::InvalidVertexId,
                format!("vertex id {vid} is out of range 0..{}", self.vcount()),
            ));
        }
        Ok(())
    }

    fn check_eid(&self, eid: EdgeId) -> Result<()> {
        if eid < 0 || eid as usize >= self.ecount() {
            return Err(Error::new(
                ErrorKind::InvalidEdgeId,
                format!("edge id {eid} is out of range 0..{}", self.ecount()),
            ));
        }
        Ok(())
    }

    /// Checks that attribute `name` of the given kind exists with type `expected`.
    fn expect_type(&self, kind: AttributeKind, name: &str, expected: AttributeType) -> Result<()> {
        match self.attribute_type(kind, name)? {
            None => Err(missing(kind, name)),
            Some(t) if t == expected => Ok(()),
            Some(t) => Err(Error::invalid(format!(
                "{kind} attribute '{name}' is {t}, not {expected}"
            ))),
        }
    }

    // --- listing --------------------------------------------------------------

    /// Names and types of all the graph, vertex and edge attributes
    /// ([`igraph_cattribute_list`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_list)).
    ///
    /// Attributes are listed in creation order. A graph without attribute
    /// storage (created before [`enable`]) has an empty list.
    ///
    /// Time complexity: O(Ag+Av+Ae), the total number of attributes.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::attributes::AttributeType;
    ///
    /// let mut g = Graph::new(2, false);
    /// g.set_graph_attr_numeric("year", 2024.0).unwrap();
    /// g.set_vertex_attr_str("label", 0, "a").unwrap();
    /// let list = g.attribute_list().unwrap();
    /// assert_eq!(list.graph, vec![("year".to_string(), AttributeType::Numeric)]);
    /// assert_eq!(list.vertex, vec![("label".to_string(), AttributeType::String)]);
    /// assert!(list.edge.is_empty());
    /// ```
    pub fn attribute_list(&self) -> Result<AttributeList> {
        if !self.has_attr_storage() {
            return Ok(AttributeList::default());
        }
        let (mut gn, mut vn, mut en) = (StrVector::new(), StrVector::new(), StrVector::new());
        let (mut gt, mut vt, mut et) = (VectorInt::new(), VectorInt::new(), VectorInt::new());
        igraph_call!(igraph_cattribute_list(
            self, &mut gn, &mut gt, &mut vn, &mut vt, &mut en, &mut et
        ))?;
        let zip = |n: &StrVector, t: &VectorInt| -> Vec<(String, AttributeType)> {
            n.iter()
                .zip(t.iter())
                .map(|(n, &t)| (n, raw_type(t)))
                .collect()
        };
        Ok(AttributeList {
            graph: zip(&gn, &gt),
            vertex: zip(&vn, &vt),
            edge: zip(&en, &et),
        })
    }

    /// Names of the attributes of one kind, in creation order (from
    /// [`igraph_cattribute_list`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_list)).
    pub fn attribute_names(&self, kind: AttributeKind) -> Result<Vec<String>> {
        Ok(self
            .attribute_list()?
            .of(kind)
            .iter()
            .map(|(n, _)| n.clone())
            .collect())
    }

    /// The type of an attribute, or `None` if there is no such attribute
    /// (from [`igraph_cattribute_list`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_list)).
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::attributes::{AttributeKind, AttributeType};
    ///
    /// let mut g = Graph::from_edges(&[(0, 1)], 2, false).unwrap();
    /// g.set_edge_attr_bool("bridge", 0, true).unwrap();
    /// assert_eq!(g.attribute_type(AttributeKind::Edge, "bridge").unwrap(), Some(AttributeType::Boolean));
    /// assert_eq!(g.attribute_type(AttributeKind::Vertex, "bridge").unwrap(), None);
    /// ```
    pub fn attribute_type(&self, kind: AttributeKind, name: &str) -> Result<Option<AttributeType>> {
        if !self.has_attr_storage() {
            return Ok(None);
        }
        let mut names = StrVector::new();
        let mut types = VectorInt::new();
        let (n, t) = (&mut names as *mut StrVector, &mut types as *mut VectorInt);
        let null_s = ptr::null_mut::<igraph_strvector_t>();
        let null_t = ptr::null_mut::<igraph_vector_int_t>();
        let args = match kind {
            AttributeKind::Graph => (n, t, null_s, null_t, null_s, null_t),
            AttributeKind::Vertex => (null_s, null_t, n, t, null_s, null_t),
            AttributeKind::Edge => (null_s, null_t, null_s, null_t, n, t),
        };
        igraph_call!(igraph_cattribute_list(
            self, args.0, args.1, args.2, args.3, args.4, args.5
        ))?;
        Ok((0..names.len())
            .find(|&i| {
                names
                    .get_cstr(i)
                    .is_some_and(|c| c.to_bytes() == name.as_bytes())
            })
            .map(|i| raw_type(types[i])))
    }

    /// Whether the graph has a graph, vertex or edge attribute called `name`
    /// ([`igraph_cattribute_has_attr`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_has_attr)).
    ///
    /// Always `false` for graphs without attribute storage (and for names
    /// containing a NUL byte).
    ///
    /// Time complexity: O(A), the number of attributes of that kind.
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::attributes::AttributeKind;
    /// let mut g = Graph::new(1, false);
    /// g.set_vertex_attr_str("name", 0, "solo").unwrap();
    /// assert!(g.has_attribute(AttributeKind::Vertex, "name"));
    /// assert!(!g.has_attribute(AttributeKind::Graph, "name"));
    /// ```
    pub fn has_attribute(&self, kind: AttributeKind, name: &str) -> bool {
        if !self.has_attr_storage() {
            return false;
        }
        match CString::new(name) {
            Ok(c) => unsafe { igraph_cattribute_has_attr(self, kind.into(), c.as_ptr()) },
            Err(_) => false,
        }
    }

    // --- graph attributes -----------------------------------------------------

    /// The value of a numeric graph attribute
    /// ([`igraph_cattribute_GAN`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_GAN)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if the attribute does not exist or is not
    /// numeric (the C function would only emit a warning and return `NaN`).
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let mut g = Graph::new(0, false);
    /// g.set_graph_attr_numeric("density", 0.25).unwrap();
    /// assert_eq!(g.graph_attr_numeric("density").unwrap(), 0.25);
    /// assert_eq!(g.graph_attr_numeric("nope").unwrap_err().kind(), ErrorKind::InvalidValue);
    /// ```
    pub fn graph_attr_numeric(&self, name: &str) -> Result<f64> {
        let c = c_string("attribute name", name)?;
        self.expect_type(AttributeKind::Graph, name, AttributeType::Numeric)?;
        Ok(unsafe { igraph_cattribute_GAN(self, c.as_ptr()) })
    }

    /// The value of a boolean graph attribute
    /// ([`igraph_cattribute_GAB`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_GAB)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if the attribute does not exist or is not boolean.
    pub fn graph_attr_bool(&self, name: &str) -> Result<bool> {
        let c = c_string("attribute name", name)?;
        self.expect_type(AttributeKind::Graph, name, AttributeType::Boolean)?;
        Ok(unsafe { igraph_cattribute_GAB(self, c.as_ptr()) })
    }

    /// The value of a string graph attribute, copied into a [`String`]
    /// ([`igraph_cattribute_GAS`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_GAS)).
    ///
    /// Invalid UTF-8 is replaced lossily.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if the attribute does not exist or is not a string.
    pub fn graph_attr_str(&self, name: &str) -> Result<String> {
        let c = c_string("attribute name", name)?;
        self.expect_type(AttributeKind::Graph, name, AttributeType::String)?;
        Ok(lossy(unsafe { igraph_cattribute_GAS(self, c.as_ptr()) }))
    }

    /// The value of a graph attribute, whatever its type (`GAN`/`GAB`/`GAS`).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if the attribute does not exist.
    pub fn graph_attr(&self, name: &str) -> Result<AttributeValue> {
        match self.attribute_type(AttributeKind::Graph, name)? {
            Some(AttributeType::Numeric) => {
                self.graph_attr_numeric(name).map(AttributeValue::Numeric)
            }
            Some(AttributeType::Boolean) => self.graph_attr_bool(name).map(AttributeValue::Boolean),
            Some(AttributeType::String) => self.graph_attr_str(name).map(AttributeValue::String),
            _ => Err(missing(AttributeKind::Graph, name)),
        }
    }

    /// Sets a numeric graph attribute, creating it if needed
    /// ([`igraph_cattribute_GAN_set`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_GAN_set)).
    ///
    /// Enables attributes (see [`enable`]) if they are not yet.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if the attribute exists with another type.
    pub fn set_graph_attr_numeric(&mut self, name: &str, value: f64) -> Result<()> {
        let c = c_string("attribute name", name)?;
        self.ensure_attr_storage()?;
        igraph_call!(igraph_cattribute_GAN_set(self, c.as_ptr(), value))
    }

    /// Sets a boolean graph attribute, creating it if needed
    /// ([`igraph_cattribute_GAB_set`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_GAB_set)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if the attribute exists with another type.
    pub fn set_graph_attr_bool(&mut self, name: &str, value: bool) -> Result<()> {
        let c = c_string("attribute name", name)?;
        self.ensure_attr_storage()?;
        igraph_call!(igraph_cattribute_GAB_set(self, c.as_ptr(), value))
    }

    /// Sets a string graph attribute (the value is copied), creating it if
    /// needed ([`igraph_cattribute_GAS_set`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_GAS_set)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if the attribute exists with another type,
    /// or if `name` or `value` contain a NUL byte.
    pub fn set_graph_attr_str(&mut self, name: &str, value: &str) -> Result<()> {
        let c = c_string("attribute name", name)?;
        let v = c_string("string value", value)?;
        self.ensure_attr_storage()?;
        igraph_call!(igraph_cattribute_GAS_set(self, c.as_ptr(), v.as_ptr()))
    }

    /// Sets a graph attribute of any type (`GAN_set`/`GAB_set`/`GAS_set`).
    ///
    /// # Errors
    ///
    /// As the typed setters, e.g. [`set_graph_attr_numeric`](Self::set_graph_attr_numeric).
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::attributes::AttributeValue;
    /// let mut g = Graph::new(0, true);
    /// g.set_graph_attr("name", "empty").unwrap();
    /// g.set_graph_attr("order", 0).unwrap();
    /// assert_eq!(g.graph_attr("name").unwrap(), AttributeValue::String("empty".into()));
    /// assert_eq!(g.graph_attr("order").unwrap().as_f64(), Some(0.0));
    /// ```
    pub fn set_graph_attr(&mut self, name: &str, value: impl Into<AttributeValue>) -> Result<()> {
        match value.into() {
            AttributeValue::Numeric(x) => self.set_graph_attr_numeric(name, x),
            AttributeValue::Boolean(b) => self.set_graph_attr_bool(name, b),
            AttributeValue::String(s) => self.set_graph_attr_str(name, &s),
        }
    }

    /// Removes a graph attribute, returning whether it existed
    /// ([`igraph_cattribute_remove_g`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_remove_g)).
    pub fn remove_graph_attr(&mut self, name: &str) -> bool {
        self.remove_attr(AttributeKind::Graph, name)
    }

    fn remove_attr(&mut self, kind: AttributeKind, name: &str) -> bool {
        if !self.has_attribute(kind, name) {
            return false;
        }
        let Ok(c) = CString::new(name) else {
            return false;
        };
        unsafe {
            match kind {
                AttributeKind::Graph => igraph_cattribute_remove_g(self, c.as_ptr()),
                AttributeKind::Vertex => igraph_cattribute_remove_v(self, c.as_ptr()),
                AttributeKind::Edge => igraph_cattribute_remove_e(self, c.as_ptr()),
            }
        }
        true
    }

    /// Removes all the graph, vertex and/or edge attributes
    /// ([`igraph_cattribute_remove_all`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_remove_all)).
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let mut g = Graph::new(3, false);
    /// g.set_graph_attr_str("title", "t").unwrap();
    /// g.set_vertex_attr_numeric("x", 0, 1.0).unwrap();
    /// g.remove_all_attributes(false, true, true);
    /// let list = g.attribute_list().unwrap();
    /// assert_eq!(list.graph.len(), 1);
    /// assert!(list.vertex.is_empty());
    /// ```
    pub fn remove_all_attributes(&mut self, graph: bool, vertex: bool, edge: bool) {
        if self.has_attr_storage() {
            unsafe { igraph_cattribute_remove_all(self, graph, vertex, edge) }
        }
    }

    // --- vertex attributes ----------------------------------------------------

    /// The value of a numeric vertex attribute for one vertex
    /// ([`igraph_cattribute_VAN`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_VAN)).
    ///
    /// Vertices that never had the attribute set hold `NaN`.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] for an invalid vertex,
    /// [`ErrorKind::InvalidValue`] if the attribute does not exist or is not numeric.
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let mut g = Graph::new(3, false);
    /// g.set_vertex_attr_numeric("age", 1, 42.0).unwrap();
    /// assert_eq!(g.vertex_attr_numeric("age", 1).unwrap(), 42.0);
    /// assert!(g.vertex_attr_numeric("age", 0).unwrap().is_nan());
    /// assert_eq!(g.vertex_attr_numeric("age", 3).unwrap_err().kind(), ErrorKind::InvalidVertexId);
    /// ```
    pub fn vertex_attr_numeric(&self, name: &str, vid: VertexId) -> Result<f64> {
        let c = c_string("attribute name", name)?;
        self.check_vid(vid)?;
        self.expect_type(AttributeKind::Vertex, name, AttributeType::Numeric)?;
        Ok(unsafe { igraph_cattribute_VAN(self, c.as_ptr(), vid) })
    }

    /// The value of a boolean vertex attribute for one vertex
    /// ([`igraph_cattribute_VAB`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_VAB)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] for an invalid vertex,
    /// [`ErrorKind::InvalidValue`] if the attribute does not exist or is not boolean.
    pub fn vertex_attr_bool(&self, name: &str, vid: VertexId) -> Result<bool> {
        let c = c_string("attribute name", name)?;
        self.check_vid(vid)?;
        self.expect_type(AttributeKind::Vertex, name, AttributeType::Boolean)?;
        Ok(unsafe { igraph_cattribute_VAB(self, c.as_ptr(), vid) })
    }

    /// The value of a string vertex attribute for one vertex, copied into a
    /// [`String`] ([`igraph_cattribute_VAS`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_VAS)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] for an invalid vertex,
    /// [`ErrorKind::InvalidValue`] if the attribute does not exist or is not a string.
    pub fn vertex_attr_str(&self, name: &str, vid: VertexId) -> Result<String> {
        let c = c_string("attribute name", name)?;
        self.check_vid(vid)?;
        self.expect_type(AttributeKind::Vertex, name, AttributeType::String)?;
        Ok(lossy(unsafe {
            igraph_cattribute_VAS(self, c.as_ptr(), vid)
        }))
    }

    /// The value of a vertex attribute of any type for one vertex (`VAN`/`VAB`/`VAS`).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] for an invalid vertex,
    /// [`ErrorKind::InvalidValue`] if the attribute does not exist.
    pub fn vertex_attr(&self, name: &str, vid: VertexId) -> Result<AttributeValue> {
        match self.attribute_type(AttributeKind::Vertex, name)? {
            Some(AttributeType::Numeric) => self
                .vertex_attr_numeric(name, vid)
                .map(AttributeValue::Numeric),
            Some(AttributeType::Boolean) => self
                .vertex_attr_bool(name, vid)
                .map(AttributeValue::Boolean),
            Some(AttributeType::String) => {
                self.vertex_attr_str(name, vid).map(AttributeValue::String)
            }
            _ => Err(missing(AttributeKind::Vertex, name)),
        }
    }

    /// The values of a numeric vertex attribute for the selected vertices, in
    /// selector order ([`igraph_cattribute_VANV`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_VANV)).
    ///
    /// Use `..` to select all the vertices.
    ///
    /// Time complexity: O(v), the number of selected vertices.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] for an invalid vertex,
    /// [`ErrorKind::InvalidValue`] if the attribute does not exist or is not numeric.
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let mut g = Graph::new(4, false);
    /// g.set_vertex_attr_numeric_values("x", &[0.0, 0.5, 1.0, 1.5]).unwrap();
    /// assert_eq!(g.vertex_attr_numeric_values("x", vec![3, 0]).unwrap(), vec![1.5, 0.0]);
    /// assert_eq!(g.vertex_attr_numeric_values("x", 1..3).unwrap(), vec![0.5, 1.0]);
    /// ```
    pub fn vertex_attr_numeric_values<'a>(
        &self,
        name: &str,
        vids: impl Into<VertexSelector<'a>>,
    ) -> Result<Vec<f64>> {
        let c = c_string("attribute name", name)?;
        let vs = vids.into().to_raw()?;
        if !self.has_attr_storage() {
            return Err(missing(AttributeKind::Vertex, name));
        }
        let mut res = Vector::new();
        igraph_call!(igraph_cattribute_VANV(self, c.as_ptr(), vs.get(), &mut res))?;
        Ok(res.into())
    }

    /// The values of a boolean vertex attribute for the selected vertices
    /// ([`igraph_cattribute_VABV`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_VABV)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] for an invalid vertex,
    /// [`ErrorKind::InvalidValue`] if the attribute does not exist or is not boolean.
    pub fn vertex_attr_bool_values<'a>(
        &self,
        name: &str,
        vids: impl Into<VertexSelector<'a>>,
    ) -> Result<Vec<bool>> {
        let c = c_string("attribute name", name)?;
        let vs = vids.into().to_raw()?;
        if !self.has_attr_storage() {
            return Err(missing(AttributeKind::Vertex, name));
        }
        let mut res = VectorBool::new();
        igraph_call!(igraph_cattribute_VABV(self, c.as_ptr(), vs.get(), &mut res))?;
        Ok(res.into())
    }

    /// The values of a string vertex attribute for the selected vertices
    /// ([`igraph_cattribute_VASV`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_VASV)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] for an invalid vertex,
    /// [`ErrorKind::InvalidValue`] if the attribute does not exist or is not a string.
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let mut g = Graph::famous("Krackhardt_Kite").unwrap();
    /// let names = ["Andre", "Beverly", "Carol", "Diane", "Ed", "Fernando", "Garth", "Heather", "Ike", "Jane"];
    /// g.set_vertex_attr_str_values("name", &names).unwrap();
    /// // The names of Jane's neighbors (Jane is the tail of the kite).
    /// let jane = VertexSelector::Adjacent { vertex: 9, mode: NeighborMode::All };
    /// assert_eq!(g.vertex_attr_str_values("name", jane).unwrap(), vec!["Ike"]);
    /// ```
    pub fn vertex_attr_str_values<'a>(
        &self,
        name: &str,
        vids: impl Into<VertexSelector<'a>>,
    ) -> Result<Vec<String>> {
        let c = c_string("attribute name", name)?;
        let vs = vids.into().to_raw()?;
        if !self.has_attr_storage() {
            return Err(missing(AttributeKind::Vertex, name));
        }
        let mut res = StrVector::new();
        igraph_call!(igraph_cattribute_VASV(self, c.as_ptr(), vs.get(), &mut res))?;
        Ok(res.to_vec())
    }

    /// The values of a vertex attribute of any type for the selected vertices
    /// (`VANV`/`VABV`/`VASV`).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] for an invalid vertex,
    /// [`ErrorKind::InvalidValue`] if the attribute does not exist.
    pub fn vertex_attr_values<'a>(
        &self,
        name: &str,
        vids: impl Into<VertexSelector<'a>>,
    ) -> Result<AttributeValues> {
        match self.attribute_type(AttributeKind::Vertex, name)? {
            Some(AttributeType::Numeric) => self
                .vertex_attr_numeric_values(name, vids)
                .map(AttributeValues::Numeric),
            Some(AttributeType::Boolean) => self
                .vertex_attr_bool_values(name, vids)
                .map(AttributeValues::Boolean),
            Some(AttributeType::String) => self
                .vertex_attr_str_values(name, vids)
                .map(AttributeValues::String),
            _ => Err(missing(AttributeKind::Vertex, name)),
        }
    }

    /// Sets a numeric vertex attribute for one vertex
    /// ([`igraph_cattribute_VAN_set`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_VAN_set)).
    ///
    /// A new attribute is created for all the vertices, with `NaN` for the
    /// others. Enables attributes (see [`enable`]) if they are not yet.
    ///
    /// Time complexity: O(n) if the attribute is new, O(1) otherwise.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] for an invalid vertex,
    /// [`ErrorKind::InvalidValue`] if the attribute exists with another type.
    pub fn set_vertex_attr_numeric(&mut self, name: &str, vid: VertexId, value: f64) -> Result<()> {
        let c = c_string("attribute name", name)?;
        self.check_vid(vid)?;
        self.ensure_attr_storage()?;
        igraph_call!(igraph_cattribute_VAN_set(self, c.as_ptr(), vid, value))
    }

    /// Sets a boolean vertex attribute for one vertex (`false` for the others
    /// if new) ([`igraph_cattribute_VAB_set`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_VAB_set)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] for an invalid vertex,
    /// [`ErrorKind::InvalidValue`] if the attribute exists with another type.
    pub fn set_vertex_attr_bool(&mut self, name: &str, vid: VertexId, value: bool) -> Result<()> {
        let c = c_string("attribute name", name)?;
        self.check_vid(vid)?;
        self.ensure_attr_storage()?;
        igraph_call!(igraph_cattribute_VAB_set(self, c.as_ptr(), vid, value))
    }

    /// Sets a string vertex attribute for one vertex (`""` for the others if
    /// new) ([`igraph_cattribute_VAS_set`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_VAS_set)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] for an invalid vertex,
    /// [`ErrorKind::InvalidValue`] if the attribute exists with another type
    /// or a string contains a NUL byte.
    pub fn set_vertex_attr_str(&mut self, name: &str, vid: VertexId, value: &str) -> Result<()> {
        let c = c_string("attribute name", name)?;
        let v = c_string("string value", value)?;
        self.check_vid(vid)?;
        self.ensure_attr_storage()?;
        igraph_call!(igraph_cattribute_VAS_set(self, c.as_ptr(), vid, v.as_ptr()))
    }

    /// Sets a vertex attribute of any type for one vertex (`VAN_set`/`VAB_set`/`VAS_set`).
    ///
    /// # Errors
    ///
    /// As the typed setters, e.g. [`set_vertex_attr_numeric`](Self::set_vertex_attr_numeric).
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let mut g = Graph::new(2, false);
    /// g.set_vertex_attr("color", 0, "red").unwrap();
    /// g.set_vertex_attr("size", 1, 3.5).unwrap();
    /// assert_eq!(g.vertex_attr_str_values("color", ..).unwrap(), vec!["red", ""]);
    /// ```
    pub fn set_vertex_attr(
        &mut self,
        name: &str,
        vid: VertexId,
        value: impl Into<AttributeValue>,
    ) -> Result<()> {
        match value.into() {
            AttributeValue::Numeric(x) => self.set_vertex_attr_numeric(name, vid, x),
            AttributeValue::Boolean(b) => self.set_vertex_attr_bool(name, vid, b),
            AttributeValue::String(s) => self.set_vertex_attr_str(name, vid, &s),
        }
    }

    /// Sets a numeric vertex attribute for all the vertices at once
    /// ([`igraph_cattribute_VAN_setv`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_VAN_setv)).
    ///
    /// `values[i]` is the value of vertex `i`.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if `values.len()` is not the number of
    /// vertices or the attribute exists with another type.
    pub fn set_vertex_attr_numeric_values(&mut self, name: &str, values: &[f64]) -> Result<()> {
        let c = c_string("attribute name", name)?;
        self.ensure_attr_storage()?;
        let v = Vector::view(values);
        igraph_call!(igraph_cattribute_VAN_setv(self, c.as_ptr(), v.as_ptr()))
    }

    /// Sets a boolean vertex attribute for all the vertices at once
    /// ([`igraph_cattribute_VAB_setv`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_VAB_setv)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if `values.len()` is not the number of
    /// vertices or the attribute exists with another type.
    pub fn set_vertex_attr_bool_values(&mut self, name: &str, values: &[bool]) -> Result<()> {
        let c = c_string("attribute name", name)?;
        self.ensure_attr_storage()?;
        let v = VectorBool::view(values);
        igraph_call!(igraph_cattribute_VAB_setv(self, c.as_ptr(), v.as_ptr()))
    }

    /// Sets a string vertex attribute for all the vertices at once
    /// ([`igraph_cattribute_VAS_setv`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_VAS_setv)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if `values.len()` is not the number of
    /// vertices, the attribute exists with another type or a string contains
    /// a NUL byte.
    pub fn set_vertex_attr_str_values<S: AsRef<str>>(
        &mut self,
        name: &str,
        values: &[S],
    ) -> Result<()> {
        let c = c_string("attribute name", name)?;
        let sv = to_strvector(values)?;
        self.ensure_attr_storage()?;
        igraph_call!(igraph_cattribute_VAS_setv(self, c.as_ptr(), &sv))
    }

    /// Sets a vertex attribute of any type for all the vertices at once
    /// (`VAN_setv`/`VAB_setv`/`VAS_setv`).
    ///
    /// # Errors
    ///
    /// As the typed setters, e.g. [`set_vertex_attr_numeric_values`](Self::set_vertex_attr_numeric_values).
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let mut g = Graph::new(3, false);
    /// g.set_vertex_attr_values("seen", vec![true, false, true]).unwrap();
    /// assert_eq!(g.vertex_attr_bool_values("seen", ..).unwrap(), vec![true, false, true]);
    /// assert!(g.set_vertex_attr_values("seen", vec![true]).is_err()); // wrong length
    /// ```
    pub fn set_vertex_attr_values(
        &mut self,
        name: &str,
        values: impl Into<AttributeValues>,
    ) -> Result<()> {
        match values.into() {
            AttributeValues::Numeric(v) => self.set_vertex_attr_numeric_values(name, &v),
            AttributeValues::Boolean(v) => self.set_vertex_attr_bool_values(name, &v),
            AttributeValues::String(v) => self.set_vertex_attr_str_values(name, &v),
        }
    }

    /// Removes a vertex attribute, returning whether it existed
    /// ([`igraph_cattribute_remove_v`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_remove_v)).
    ///
    /// Once removed, the name can be reused for an attribute of another type.
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let mut g = Graph::new(2, false);
    /// g.set_vertex_attr_numeric("tmp", 0, 1.0).unwrap();
    /// assert!(g.remove_vertex_attr("tmp"));
    /// assert!(!g.remove_vertex_attr("tmp"));
    /// g.set_vertex_attr_str("tmp", 0, "now a string").unwrap();
    /// ```
    pub fn remove_vertex_attr(&mut self, name: &str) -> bool {
        self.remove_attr(AttributeKind::Vertex, name)
    }

    // --- edge attributes ------------------------------------------------------

    /// The value of a numeric edge attribute for one edge
    /// ([`igraph_cattribute_EAN`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_EAN)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidEdgeId`] for an invalid edge,
    /// [`ErrorKind::InvalidValue`] if the attribute does not exist or is not numeric.
    pub fn edge_attr_numeric(&self, name: &str, eid: EdgeId) -> Result<f64> {
        let c = c_string("attribute name", name)?;
        self.check_eid(eid)?;
        self.expect_type(AttributeKind::Edge, name, AttributeType::Numeric)?;
        Ok(unsafe { igraph_cattribute_EAN(self, c.as_ptr(), eid) })
    }

    /// The value of a boolean edge attribute for one edge
    /// ([`igraph_cattribute_EAB`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_EAB)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidEdgeId`] for an invalid edge,
    /// [`ErrorKind::InvalidValue`] if the attribute does not exist or is not boolean.
    pub fn edge_attr_bool(&self, name: &str, eid: EdgeId) -> Result<bool> {
        let c = c_string("attribute name", name)?;
        self.check_eid(eid)?;
        self.expect_type(AttributeKind::Edge, name, AttributeType::Boolean)?;
        Ok(unsafe { igraph_cattribute_EAB(self, c.as_ptr(), eid) })
    }

    /// The value of a string edge attribute for one edge
    /// ([`igraph_cattribute_EAS`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_EAS)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidEdgeId`] for an invalid edge,
    /// [`ErrorKind::InvalidValue`] if the attribute does not exist or is not a string.
    pub fn edge_attr_str(&self, name: &str, eid: EdgeId) -> Result<String> {
        let c = c_string("attribute name", name)?;
        self.check_eid(eid)?;
        self.expect_type(AttributeKind::Edge, name, AttributeType::String)?;
        Ok(lossy(unsafe {
            igraph_cattribute_EAS(self, c.as_ptr(), eid)
        }))
    }

    /// The value of an edge attribute of any type for one edge (`EAN`/`EAB`/`EAS`).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidEdgeId`] for an invalid edge,
    /// [`ErrorKind::InvalidValue`] if the attribute does not exist.
    pub fn edge_attr(&self, name: &str, eid: EdgeId) -> Result<AttributeValue> {
        match self.attribute_type(AttributeKind::Edge, name)? {
            Some(AttributeType::Numeric) => self
                .edge_attr_numeric(name, eid)
                .map(AttributeValue::Numeric),
            Some(AttributeType::Boolean) => {
                self.edge_attr_bool(name, eid).map(AttributeValue::Boolean)
            }
            Some(AttributeType::String) => {
                self.edge_attr_str(name, eid).map(AttributeValue::String)
            }
            _ => Err(missing(AttributeKind::Edge, name)),
        }
    }

    /// The values of a numeric edge attribute for the selected edges, in
    /// selector order ([`igraph_cattribute_EANV`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_EANV)).
    ///
    /// This is the natural way to obtain a weight vector for the weighted
    /// algorithms of the crate: `g.edge_attr_numeric_values("weight", ..)`
    /// (see e.g. [`Graph::distances_dijkstra`] and [`Graph::strength`]).
    ///
    /// Time complexity: O(e), the number of selected edges.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidEdgeId`] for an invalid edge,
    /// [`ErrorKind::InvalidValue`] if the attribute does not exist or is not numeric.
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let mut g = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// g.set_edge_attr_numeric_values("weight", &[0.5, 2.0]).unwrap();
    /// let weights = g.edge_attr_numeric_values("weight", ..).unwrap();
    /// assert_eq!(weights.iter().sum::<f64>(), 2.5);
    /// let d = g.distances_dijkstra(0, 2, Some(&weights), NeighborMode::All).unwrap();
    /// assert_eq!(d[(0, 0)], 2.5);
    /// ```
    pub fn edge_attr_numeric_values<'a>(
        &self,
        name: &str,
        eids: impl Into<EdgeSelector<'a>>,
    ) -> Result<Vec<f64>> {
        let c = c_string("attribute name", name)?;
        let es = eids.into().to_raw()?;
        if !self.has_attr_storage() {
            return Err(missing(AttributeKind::Edge, name));
        }
        let mut res = Vector::new();
        igraph_call!(igraph_cattribute_EANV(self, c.as_ptr(), es.get(), &mut res))?;
        Ok(res.into())
    }

    /// The values of a boolean edge attribute for the selected edges
    /// ([`igraph_cattribute_EABV`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_EABV)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidEdgeId`] for an invalid edge,
    /// [`ErrorKind::InvalidValue`] if the attribute does not exist or is not boolean.
    pub fn edge_attr_bool_values<'a>(
        &self,
        name: &str,
        eids: impl Into<EdgeSelector<'a>>,
    ) -> Result<Vec<bool>> {
        let c = c_string("attribute name", name)?;
        let es = eids.into().to_raw()?;
        if !self.has_attr_storage() {
            return Err(missing(AttributeKind::Edge, name));
        }
        let mut res = VectorBool::new();
        igraph_call!(igraph_cattribute_EABV(self, c.as_ptr(), es.get(), &mut res))?;
        Ok(res.into())
    }

    /// The values of a string edge attribute for the selected edges
    /// ([`igraph_cattribute_EASV`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_EASV)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidEdgeId`] for an invalid edge,
    /// [`ErrorKind::InvalidValue`] if the attribute does not exist or is not a string.
    pub fn edge_attr_str_values<'a>(
        &self,
        name: &str,
        eids: impl Into<EdgeSelector<'a>>,
    ) -> Result<Vec<String>> {
        let c = c_string("attribute name", name)?;
        let es = eids.into().to_raw()?;
        if !self.has_attr_storage() {
            return Err(missing(AttributeKind::Edge, name));
        }
        let mut res = StrVector::new();
        igraph_call!(igraph_cattribute_EASV(self, c.as_ptr(), es.get(), &mut res))?;
        Ok(res.to_vec())
    }

    /// The values of an edge attribute of any type for the selected edges
    /// (`EANV`/`EABV`/`EASV`).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidEdgeId`] for an invalid edge,
    /// [`ErrorKind::InvalidValue`] if the attribute does not exist.
    pub fn edge_attr_values<'a>(
        &self,
        name: &str,
        eids: impl Into<EdgeSelector<'a>>,
    ) -> Result<AttributeValues> {
        match self.attribute_type(AttributeKind::Edge, name)? {
            Some(AttributeType::Numeric) => self
                .edge_attr_numeric_values(name, eids)
                .map(AttributeValues::Numeric),
            Some(AttributeType::Boolean) => self
                .edge_attr_bool_values(name, eids)
                .map(AttributeValues::Boolean),
            Some(AttributeType::String) => self
                .edge_attr_str_values(name, eids)
                .map(AttributeValues::String),
            _ => Err(missing(AttributeKind::Edge, name)),
        }
    }

    /// Sets a numeric edge attribute for one edge (`NaN` for the others if
    /// new) ([`igraph_cattribute_EAN_set`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_EAN_set)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidEdgeId`] for an invalid edge,
    /// [`ErrorKind::InvalidValue`] if the attribute exists with another type.
    pub fn set_edge_attr_numeric(&mut self, name: &str, eid: EdgeId, value: f64) -> Result<()> {
        let c = c_string("attribute name", name)?;
        self.check_eid(eid)?;
        self.ensure_attr_storage()?;
        igraph_call!(igraph_cattribute_EAN_set(self, c.as_ptr(), eid, value))
    }

    /// Sets a boolean edge attribute for one edge (`false` for the others if
    /// new) ([`igraph_cattribute_EAB_set`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_EAB_set)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidEdgeId`] for an invalid edge,
    /// [`ErrorKind::InvalidValue`] if the attribute exists with another type.
    pub fn set_edge_attr_bool(&mut self, name: &str, eid: EdgeId, value: bool) -> Result<()> {
        let c = c_string("attribute name", name)?;
        self.check_eid(eid)?;
        self.ensure_attr_storage()?;
        igraph_call!(igraph_cattribute_EAB_set(self, c.as_ptr(), eid, value))
    }

    /// Sets a string edge attribute for one edge (`""` for the others if new)
    /// ([`igraph_cattribute_EAS_set`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_EAS_set)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidEdgeId`] for an invalid edge,
    /// [`ErrorKind::InvalidValue`] if the attribute exists with another type
    /// or a string contains a NUL byte.
    pub fn set_edge_attr_str(&mut self, name: &str, eid: EdgeId, value: &str) -> Result<()> {
        let c = c_string("attribute name", name)?;
        let v = c_string("string value", value)?;
        self.check_eid(eid)?;
        self.ensure_attr_storage()?;
        igraph_call!(igraph_cattribute_EAS_set(self, c.as_ptr(), eid, v.as_ptr()))
    }

    /// Sets an edge attribute of any type for one edge (`EAN_set`/`EAB_set`/`EAS_set`).
    ///
    /// A new attribute gets the default value (`NaN`, `false` or `""`) on
    /// the other edges.
    ///
    /// # Errors
    ///
    /// As the typed setters, e.g. [`set_edge_attr_numeric`](Self::set_edge_attr_numeric).
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::attributes::AttributeValue;
    /// let mut g = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    /// g.set_edge_attr("capacity", 1, 10).unwrap();
    /// g.set_edge_attr("road", 0, true).unwrap();
    /// assert_eq!(g.edge_attr("capacity", 1).unwrap(), AttributeValue::Numeric(10.0));
    /// assert!(g.edge_attr_numeric("capacity", 0).unwrap().is_nan());
    /// assert_eq!(g.edge_attr_bool_values("road", ..).unwrap(), vec![true, false]);
    /// ```
    pub fn set_edge_attr(
        &mut self,
        name: &str,
        eid: EdgeId,
        value: impl Into<AttributeValue>,
    ) -> Result<()> {
        match value.into() {
            AttributeValue::Numeric(x) => self.set_edge_attr_numeric(name, eid, x),
            AttributeValue::Boolean(b) => self.set_edge_attr_bool(name, eid, b),
            AttributeValue::String(s) => self.set_edge_attr_str(name, eid, &s),
        }
    }

    /// Sets a numeric edge attribute for all the edges at once
    /// ([`igraph_cattribute_EAN_setv`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_EAN_setv)).
    ///
    /// `values[i]` is the value of edge `i`.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if `values.len()` is not the number of
    /// edges or the attribute exists with another type.
    pub fn set_edge_attr_numeric_values(&mut self, name: &str, values: &[f64]) -> Result<()> {
        let c = c_string("attribute name", name)?;
        self.ensure_attr_storage()?;
        let v = Vector::view(values);
        igraph_call!(igraph_cattribute_EAN_setv(self, c.as_ptr(), v.as_ptr()))
    }

    /// Sets a boolean edge attribute for all the edges at once
    /// ([`igraph_cattribute_EAB_setv`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_EAB_setv)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if `values.len()` is not the number of
    /// edges or the attribute exists with another type.
    pub fn set_edge_attr_bool_values(&mut self, name: &str, values: &[bool]) -> Result<()> {
        let c = c_string("attribute name", name)?;
        self.ensure_attr_storage()?;
        let v = VectorBool::view(values);
        igraph_call!(igraph_cattribute_EAB_setv(self, c.as_ptr(), v.as_ptr()))
    }

    /// Sets a string edge attribute for all the edges at once
    /// ([`igraph_cattribute_EAS_setv`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_EAS_setv)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if `values.len()` is not the number of
    /// edges, the attribute exists with another type or a string contains a
    /// NUL byte.
    pub fn set_edge_attr_str_values<S: AsRef<str>>(
        &mut self,
        name: &str,
        values: &[S],
    ) -> Result<()> {
        let c = c_string("attribute name", name)?;
        let sv = to_strvector(values)?;
        self.ensure_attr_storage()?;
        igraph_call!(igraph_cattribute_EAS_setv(self, c.as_ptr(), &sv))
    }

    /// Sets an edge attribute of any type for all the edges at once
    /// (`EAN_setv`/`EAB_setv`/`EAS_setv`).
    ///
    /// # Errors
    ///
    /// As the typed setters, e.g. [`set_edge_attr_numeric_values`](Self::set_edge_attr_numeric_values).
    pub fn set_edge_attr_values(
        &mut self,
        name: &str,
        values: impl Into<AttributeValues>,
    ) -> Result<()> {
        match values.into() {
            AttributeValues::Numeric(v) => self.set_edge_attr_numeric_values(name, &v),
            AttributeValues::Boolean(v) => self.set_edge_attr_bool_values(name, &v),
            AttributeValues::String(v) => self.set_edge_attr_str_values(name, &v),
        }
    }

    /// Removes an edge attribute, returning whether it existed
    /// ([`igraph_cattribute_remove_e`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_cattribute_remove_e)).
    pub fn remove_edge_attr(&mut self, name: &str) -> bool {
        self.remove_attr(AttributeKind::Edge, name)
    }

    // --- structure + attributes -----------------------------------------------

    /// Adds `n` vertices together with their attribute values
    /// ([`igraph_add_vertices`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_add_vertices)
    /// with an attribute record list).
    ///
    /// Each record must be named and hold exactly `n` values. New attributes
    /// are created (with default values for the existing vertices);
    /// existing attributes not mentioned get their default value for the new
    /// vertices. Enables attributes if they are not yet.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if a record has the wrong length, no name,
    /// or a type different from the existing attribute with the same name.
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::attributes::AttributeRecord;
    ///
    /// let mut g = Graph::new(1, false);
    /// g.set_vertex_attr_str("name", 0, "root").unwrap();
    /// let names = AttributeRecord::string("name", &["left", "right"]).unwrap();
    /// g.add_vertices_with_attributes(2, &[names]).unwrap();
    /// assert_eq!(g.vertex_attr_str_values("name", ..).unwrap(), vec!["root", "left", "right"]);
    /// ```
    pub fn add_vertices_with_attributes(
        &mut self,
        n: usize,
        records: &[AttributeRecord],
    ) -> Result<()> {
        let n = igraph_int_t::try_from(n)
            .map_err(|_| Error::invalid(format!("cannot add {n} vertices")))?;
        let list = RecordList::from_records(records)?;
        self.ensure_attr_storage()?;
        igraph_call!(igraph_add_vertices(self, n, &list.0))
    }

    /// Adds edges together with their attribute values
    /// ([`igraph_add_edges`](https://igraph.org/c/html/latest/igraph-Basic.html#igraph_add_edges)
    /// with an attribute record list).
    ///
    /// Each record must be named and hold exactly `edges.len()` values.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidVertexId`] for invalid endpoints,
    /// [`ErrorKind::InvalidValue`] for invalid records (see
    /// [`add_vertices_with_attributes`](Self::add_vertices_with_attributes)).
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::attributes::AttributeRecord;
    ///
    /// let mut g = Graph::new(3, true);
    /// let w = AttributeRecord::numeric("weight", &[1.5, 2.5]).unwrap();
    /// g.add_edges_with_attributes(&[(0, 1), (1, 2)], &[w]).unwrap();
    /// assert_eq!(g.edge_attr_numeric("weight", 1).unwrap(), 2.5);
    /// ```
    pub fn add_edges_with_attributes(
        &mut self,
        edges: &[(VertexId, VertexId)],
        records: &[AttributeRecord],
    ) -> Result<()> {
        let flat: VectorInt = edges.iter().flat_map(|&(a, b)| [a, b]).collect();
        let list = RecordList::from_records(records)?;
        self.ensure_attr_storage()?;
        igraph_call!(igraph_add_edges(self, &flat, &list.0))
    }

    // --- merging with attribute combinations ------------------------------------

    /// Removes multi-edges and/or self-loops like [`Graph::simplify`],
    /// combining the attributes of the merged edges according to `edge_comb`
    /// ([`igraph_simplify`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_simplify)
    /// with an edge attribute combination).
    ///
    /// When multi-edges are removed, igraph rebuilds the graph: each group of
    /// parallel edges becomes one edge whose attributes are computed from the
    /// values of the group (a group of a single edge is combined too, e.g.
    /// [`Sum`](AttributeCombinationType::Sum) keeps its value). Attributes
    /// the combination maps to [`Ignore`](AttributeCombinationType::Ignore)
    /// or [`Default`](AttributeCombinationType::Default), and all of them for
    /// an empty combination, are dropped. The edge order may change.
    ///
    /// `edge_comb` is **not** used, and the remaining edges keep all their
    /// attributes, when nothing has to be merged: if `remove_multiple` is
    /// `false` (loops are then deleted in place), or if igraph already knows
    /// that the graph has no multi-edges (e.g. it was simplified before), or
    /// nothing at all is to be removed. Graph and vertex attributes are
    /// always kept. Without attributes enabled this is exactly
    /// [`Graph::simplify`].
    ///
    /// Time complexity: O(|V|+|E|), plus the cost of the combinations.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::AttributeCombination`] if a combination does not apply to
    /// the type of an attribute (e.g. summing strings),
    /// [`ErrorKind::Unimplemented`] for the numeric median.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::attributes::{AttributeCombination, AttributeCombinationType as Comb};
    ///
    /// // Three calls between 0 and 1, one between 1 and 2.
    /// let mut calls = Graph::from_edges(&[(0, 1), (1, 2), (0, 1), (0, 1)], 3, false).unwrap();
    /// calls.set_edge_attr_numeric_values("minutes", &[5.0, 2.0, 1.0, 4.0]).unwrap();
    /// let comb = AttributeCombination::all(Comb::Sum).unwrap();
    /// calls.simplify_with_attributes(true, true, &comb).unwrap();
    /// assert_eq!(calls.edge_list(), vec![(0, 1), (1, 2)]);
    /// assert_eq!(calls.edge_attr_numeric_values("minutes", ..).unwrap(), vec![10.0, 2.0]);
    /// ```
    pub fn simplify_with_attributes(
        &mut self,
        remove_multiple: bool,
        remove_loops: bool,
        edge_comb: &AttributeCombination,
    ) -> Result<()> {
        igraph_call!(igraph_simplify(
            self,
            remove_multiple,
            remove_loops,
            edge_comb.as_ptr()
        ))
    }

    /// Merges groups of vertices into single vertices like
    /// [`Graph::contract_vertices`], combining their attributes according to
    /// `vertex_comb` ([`igraph_contract_vertices`](https://igraph.org/c/html/latest/igraph-Operators.html#igraph_contract_vertices)
    /// with a vertex attribute combination).
    ///
    /// `mapping[v]` is the id of the vertex that `v` becomes; use consecutive
    /// ids starting at 0 to avoid creating isolated "orphan" vertices. No
    /// edge is removed (merged groups get self-loops and multi-edges: use
    /// [`simplify_with_attributes`](Self::simplify_with_attributes)
    /// afterwards), and edge and graph attributes are kept unchanged.
    ///
    /// Time complexity: O(|V|+|E|), plus the cost of the combinations.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if `mapping` does not have one entry per
    /// vertex or contains negative ids (or `i64::MAX`); the
    /// combination errors of
    /// [`simplify_with_attributes`](Self::simplify_with_attributes).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::attributes::{AttributeCombination, AttributeCombinationType as Comb};
    ///
    /// // Four towns in two provinces.
    /// let mut g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    /// g.set_vertex_attr_numeric_values("population", &[10.0, 5.0, 7.0, 3.0]).unwrap();
    /// g.set_vertex_attr_str_values("name", &["Firenze", "Prato", "Pisa", "Lucca"]).unwrap();
    /// let comb = AttributeCombination::from_pairs(&[
    ///     (Some("population"), Comb::Sum),
    ///     (Some("name"), Comb::First),
    /// ]).unwrap();
    /// g.contract_vertices_with_attributes(&[0, 0, 1, 1], &comb).unwrap();
    /// assert_eq!(g.vertex_attr_numeric_values("population", ..).unwrap(), vec![15.0, 10.0]);
    /// assert_eq!(g.vertex_attr_str_values("name", ..).unwrap(), vec!["Firenze", "Pisa"]);
    /// assert_eq!(g.ecount(), 3); // two self-loops and the edge between the provinces
    /// ```
    pub fn contract_vertices_with_attributes(
        &mut self,
        mapping: &[VertexId],
        vertex_comb: &AttributeCombination,
    ) -> Result<()> {
        if mapping.len() != self.vcount() {
            return Err(Error::invalid(format!(
                "the mapping has {} entries but the graph has {} vertices",
                mapping.len(),
                self.vcount()
            )));
        }
        // Negative ids index out of bounds in C, and igraph computes the new
        // vertex count as `max + 1`, which overflows for `VertexId::MAX`.
        if let Some(&bad) = mapping.iter().find(|&&m| !(0..VertexId::MAX).contains(&m)) {
            return Err(Error::invalid(format!(
                "the mapping contains the invalid vertex id {bad}"
            )));
        }
        let m = VectorInt::view(mapping);
        igraph_call!(igraph_contract_vertices(
            self,
            m.as_ptr(),
            vertex_comb.as_ptr()
        ))
    }

    /// Converts a directed graph to an undirected one like
    /// [`Graph::to_undirected`], combining the attributes of the directed
    /// edges that become a single undirected edge according to `edge_comb`
    /// ([`igraph_to_undirected`](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_to_undirected)
    /// with an edge attribute combination).
    ///
    /// With [`ToUndirected::Collapse`] all the edges between a pair of
    /// vertices are merged; with [`ToUndirected::Mutual`] each mutual pair
    /// `u -> v`, `v -> u` is merged into one edge (non-mutual edges are lost,
    /// loops are kept); with [`ToUndirected::Each`] every edge is kept together
    /// with its attributes and `edge_comb` is not used. Graph and vertex
    /// attributes are kept. Undirected graphs are left unchanged.
    ///
    /// Time complexity: O(|V|+|E|), plus the cost of the combinations.
    ///
    /// # Errors
    ///
    /// The combination errors of
    /// [`simplify_with_attributes`](Self::simplify_with_attributes).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// use igraph::attributes::{AttributeCombination, AttributeCombinationType as Comb};
    ///
    /// // Messages sent in both directions between 0 and 1, and from 1 to 2.
    /// let mut g = Graph::from_edges(&[(0, 1), (1, 0), (1, 2)], 3, true).unwrap();
    /// g.set_edge_attr_numeric_values("messages", &[3.0, 4.0, 1.0]).unwrap();
    /// let comb = AttributeCombination::all(Comb::Sum).unwrap();
    /// g.to_undirected_with_attributes(ToUndirected::Collapse, &comb).unwrap();
    /// assert_eq!(g.edge_list(), vec![(0, 1), (1, 2)]);
    /// assert_eq!(g.edge_attr_numeric_values("messages", ..).unwrap(), vec![7.0, 1.0]);
    /// ```
    pub fn to_undirected_with_attributes(
        &mut self,
        mode: ToUndirected,
        edge_comb: &AttributeCombination,
    ) -> Result<()> {
        igraph_call!(igraph_to_undirected(self, mode.into(), edge_comb.as_ptr()))
    }
}

// ---------------------------------------------------------------------------
// Attribute records
// ---------------------------------------------------------------------------

/// A named, typed vector of attribute values (`igraph_attribute_record_t`).
///
/// Attribute records are the unit of data exchanged between igraph and an
/// attribute handler; in Rust they are mostly useful to add vertices or edges
/// together with their attributes, see [`Graph::add_vertices_with_attributes`].
/// A record owns its values and frees them on [`Drop`]; [`Clone`] makes a
/// deep copy, default value included.
///
/// A record also has a *default value*, used to fill new slots when it is
/// [resized](Self::resize): `NaN`, `false` or `""` unless changed with
/// [`set_default`](Self::set_default).
///
/// ```
/// use igraph::attributes::{AttributeRecord, AttributeType, AttributeValue, AttributeValues};
///
/// let mut rec = AttributeRecord::numeric("score", &[1.0, 2.0]).unwrap();
/// rec.set_default(AttributeValue::Numeric(-1.0)).unwrap();
/// rec.resize(4).unwrap();
/// assert_eq!(rec.name(), Some("score"));
/// assert_eq!(rec.attribute_type(), AttributeType::Numeric);
/// assert_eq!(rec.values(), AttributeValues::Numeric(vec![1.0, 2.0, -1.0, -1.0]));
/// ```
pub type AttributeRecord = igraph_attribute_record_t;

impl igraph_attribute_record_t {
    /// Creates an empty record with the given name and type
    /// ([`igraph_attribute_record_init`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_attribute_record_init)).
    ///
    /// An [`AttributeType::Unspecified`] record is untyped: it holds no
    /// values until [`set_type`](Self::set_type) gives it a type.
    ///
    /// Time complexity: O(1).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if `name` contains a NUL byte or the type
    /// is [`AttributeType::Object`] (checked in Rust: since igraph 1.0.1 the
    /// C library treats an unsupported type as a *fatal* error and aborts the
    /// process, while 1.0.0 returned an error).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::attributes::{AttributeRecord, AttributeType};
    /// let mut rec = AttributeRecord::new("color", AttributeType::String).unwrap();
    /// assert!(rec.is_empty());
    /// rec.resize(2).unwrap();
    /// assert_eq!(rec.values().as_strings().unwrap(), ["", ""]);
    /// assert!(AttributeRecord::new("o", AttributeType::Object).is_err());
    /// ```
    pub fn new(name: &str, attribute_type: AttributeType) -> Result<Self> {
        let c = c_string("attribute name", name)?;
        if attribute_type == AttributeType::Object {
            return Err(Error::invalid("object attributes are not supported"));
        }
        // SAFETY: `c` is a valid C string and the type is supported.
        unsafe { Self::init_raw(c.as_ptr(), attribute_type.into()) }
    }

    /// `igraph_attribute_record_init` into a new record.
    ///
    /// # Safety
    ///
    /// `name` must be NULL or a valid C string, and `type_` must be
    /// `UNSPECIFIED`, `NUMERIC`, `BOOLEAN` or `STRING` (igraph 1.0.1 aborts
    /// the process on any other type).
    unsafe fn init_raw(name: *const c_char, type_: igraph_attribute_type_t) -> Result<Self> {
        let mut raw = MaybeUninit::<Self>::uninit();
        // On failure the record must *not* be destroyed: igraph 1.0.1 has
        // already released everything it allocated (the name through the
        // "finally" stack, which leaves `name` dangling), so destroying it
        // would free the name twice.
        igraph_call!(igraph_attribute_record_init(raw.as_mut_ptr(), name, type_))?;
        // SAFETY: `init` succeeded, so every field is initialized.
        Ok(unsafe { raw.assume_init() })
    }

    /// Creates a numeric record holding `values`.
    pub fn numeric(name: &str, values: &[f64]) -> Result<Self> {
        let mut rec = Self::new(name, AttributeType::Numeric)?;
        let v = Vector::view(values);
        igraph_call!(igraph_vector_update(
            *unsafe { rec.value.as_vector.as_mut() },
            v.as_ptr()
        ))?;
        Ok(rec)
    }

    /// Creates a boolean record holding `values`.
    pub fn boolean(name: &str, values: &[bool]) -> Result<Self> {
        let mut rec = Self::new(name, AttributeType::Boolean)?;
        let v = VectorBool::view(values);
        igraph_call!(igraph_vector_bool_update(
            *unsafe { rec.value.as_vector_bool.as_mut() },
            v.as_ptr()
        ))?;
        Ok(rec)
    }

    /// Creates a string record holding `values`.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if a string contains a NUL byte.
    pub fn string<S: AsRef<str>>(name: &str, values: &[S]) -> Result<Self> {
        let sv = to_strvector(values)?;
        let mut rec = Self::new(name, AttributeType::String)?;
        igraph_call!(igraph_strvector_update(
            *unsafe { rec.value.as_strvector.as_mut() },
            &sv
        ))?;
        Ok(rec)
    }

    /// Creates a record of the type of `values`.
    pub fn from_values(name: &str, values: impl Into<AttributeValues>) -> Result<Self> {
        match values.into() {
            AttributeValues::Numeric(v) => Self::numeric(name, &v),
            AttributeValues::Boolean(v) => Self::boolean(name, &v),
            AttributeValues::String(v) => Self::string(name, &v),
        }
    }

    /// The name of the attribute (`None` if unset or not UTF-8).
    pub fn name(&self) -> Option<&str> {
        if self.name.is_null() {
            None
        } else {
            unsafe { CStr::from_ptr(self.name) }.to_str().ok()
        }
    }

    /// Renames the record
    /// ([`igraph_attribute_record_set_name`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_attribute_record_set_name)).
    pub fn set_name(&mut self, name: &str) -> Result<()> {
        let c = c_string("attribute name", name)?;
        igraph_call!(igraph_attribute_record_set_name(self, c.as_ptr()))
    }

    /// The type of the values.
    pub fn attribute_type(&self) -> AttributeType {
        AttributeType::try_from(self.type_).unwrap_or(AttributeType::Object)
    }

    /// Changes the type of the record
    /// ([`igraph_attribute_record_set_type`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_attribute_record_set_type)).
    ///
    /// When the type changes, the values are discarded and the record becomes
    /// empty; setting the same type is a no-op.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] for [`AttributeType::Unspecified`] and
    /// [`AttributeType::Object`], checked in Rust: igraph 1.0.1 aborts the
    /// process on these (a fatal error in `igraph_attribute_record_set_type`;
    /// 1.0.0 returned `IGRAPH_EINVAL`).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::attributes::{AttributeRecord, AttributeType};
    /// let mut rec = AttributeRecord::numeric("x", &[1.0, 2.0]).unwrap();
    /// rec.set_type(AttributeType::Numeric).unwrap(); // same type: no-op
    /// assert_eq!(rec.len(), 2);
    /// rec.set_type(AttributeType::Boolean).unwrap(); // new type: values dropped
    /// assert!(rec.is_empty());
    /// assert!(rec.set_type(AttributeType::Unspecified).is_err());
    /// ```
    pub fn set_type(&mut self, attribute_type: AttributeType) -> Result<()> {
        if matches!(
            attribute_type,
            AttributeType::Unspecified | AttributeType::Object
        ) {
            return Err(Error::invalid(format!(
                "cannot set the record type to {attribute_type}"
            )));
        }
        igraph_call!(igraph_attribute_record_set_type(
            self,
            attribute_type.into()
        ))
    }

    /// Checks that the record has the expected type
    /// ([`igraph_attribute_record_check_type`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_attribute_record_check_type)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if the types differ.
    pub fn check_type(&self, expected: AttributeType) -> Result<()> {
        igraph_call!(igraph_attribute_record_check_type(self, expected.into()))
    }

    /// Number of values
    /// ([`igraph_attribute_record_size`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_attribute_record_size)).
    pub fn len(&self) -> usize {
        match self.attribute_type() {
            AttributeType::Numeric | AttributeType::Boolean | AttributeType::String => unsafe {
                igraph_attribute_record_size(self) as usize
            },
            _ => 0,
        }
    }

    /// Whether the record holds no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resizes the value vector, filling new slots with the default value
    /// ([`igraph_attribute_record_resize`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_attribute_record_resize)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if the record has no type yet.
    pub fn resize(&mut self, len: usize) -> Result<()> {
        if !matches!(
            self.attribute_type(),
            AttributeType::Numeric | AttributeType::Boolean | AttributeType::String
        ) {
            return Err(Error::invalid("the attribute record has no type yet"));
        }
        let len = igraph_int_t::try_from(len)
            .map_err(|_| Error::invalid(format!("record length {len} is too large")))?;
        igraph_call!(igraph_attribute_record_resize(self, len))
    }

    /// Sets the value used to fill new slots on [`resize`](Self::resize)
    /// ([`igraph_attribute_record_set_default_numeric`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_attribute_record_set_default_numeric),
    /// [`_boolean`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_attribute_record_set_default_boolean),
    /// [`_string`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_attribute_record_set_default_string)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] if the value's type is not the record's type.
    pub fn set_default(&mut self, value: impl Into<AttributeValue>) -> Result<()> {
        let value = value.into();
        if value.attribute_type() != self.attribute_type() {
            return Err(Error::invalid(format!(
                "a {} default value does not fit a {} attribute record",
                value.attribute_type(),
                self.attribute_type()
            )));
        }
        match value {
            AttributeValue::Numeric(x) => {
                igraph_call!(igraph_attribute_record_set_default_numeric(self, x))
            }
            AttributeValue::Boolean(b) => {
                igraph_call!(igraph_attribute_record_set_default_boolean(self, b))
            }
            AttributeValue::String(s) => {
                let c = c_string("string value", &s)?;
                igraph_call!(igraph_attribute_record_set_default_string(self, c.as_ptr()))
            }
        }
    }

    /// The value used to fill new slots on [`resize`](Self::resize), see
    /// [`set_default`](Self::set_default); `None` for an untyped record.
    ///
    /// Unless changed, it is `NaN`, `false` or `""` depending on the type.
    ///
    /// ```
    /// use igraph::attributes::{AttributeRecord, AttributeValue};
    /// let mut rec = AttributeRecord::string("city", &["Firenze"]).unwrap();
    /// assert_eq!(rec.default_value(), Some(AttributeValue::String(String::new())));
    /// rec.set_default("unknown").unwrap();
    /// assert_eq!(rec.default_value(), Some(AttributeValue::String("unknown".into())));
    /// ```
    pub fn default_value(&self) -> Option<AttributeValue> {
        // SAFETY: the active union member is the one matching `type_`.
        unsafe {
            match self.attribute_type() {
                AttributeType::Numeric => Some(AttributeValue::Numeric(
                    *self.default_value.numeric.as_ref(),
                )),
                AttributeType::Boolean => Some(AttributeValue::Boolean(
                    *self.default_value.boolean.as_ref(),
                )),
                AttributeType::String => Some(AttributeValue::String(lossy(
                    *self.default_value.string.as_ref(),
                ))),
                _ => None,
            }
        }
    }

    /// A copy of the values (an empty [`AttributeValues::Numeric`] for an
    /// untyped record).
    pub fn values(&self) -> AttributeValues {
        unsafe {
            match self.attribute_type() {
                AttributeType::Numeric => {
                    AttributeValues::Numeric((**self.value.as_vector.as_ref()).to_vec())
                }
                AttributeType::Boolean => {
                    AttributeValues::Boolean((**self.value.as_vector_bool.as_ref()).to_vec())
                }
                AttributeType::String => {
                    AttributeValues::String((**self.value.as_strvector.as_ref()).to_vec())
                }
                _ => AttributeValues::Numeric(Vec::new()),
            }
        }
    }
}

impl Drop for igraph_attribute_record_t {
    /// Frees the name and the values (`igraph_attribute_record_destroy`).
    fn drop(&mut self) {
        unsafe { igraph_attribute_record_destroy(self) }
    }
}

impl Clone for igraph_attribute_record_t {
    /// Deep copy of the name, type, values and [default
    /// value](Self::default_value).
    ///
    /// It does not use `igraph_attribute_record_init_copy`: that function
    /// does not copy the default value, and when it fails its partially
    /// initialized result can be neither destroyed nor leaked safely.
    ///
    /// # Panics
    ///
    /// If igraph fails to allocate the copy.
    fn clone(&self) -> Self {
        let ty = self.attribute_type();
        assert!(
            ty != AttributeType::Object,
            "attribute records of unsupported types cannot be cloned"
        );
        // SAFETY: `self.name` is NULL or a valid C string owned by `self`,
        // and the type is supported (checked above).
        let mut copy = unsafe { Self::init_raw(self.name, ty.into()) }
            .expect("igraph failed to copy an attribute record");
        // From here on `copy` is a valid record, freed by `Drop` on panic.
        // SAFETY: both records have type `ty`, so the matching union
        // members are the active ones and point to initialized vectors.
        igraph_call!(match ty {
            AttributeType::Numeric => igraph_vector_update(
                *copy.value.as_vector.as_mut(),
                *self.value.as_vector.as_ref(),
            ),
            AttributeType::Boolean => igraph_vector_bool_update(
                *copy.value.as_vector_bool.as_mut(),
                *self.value.as_vector_bool.as_ref(),
            ),
            AttributeType::String => igraph_strvector_update(
                *copy.value.as_strvector.as_mut(),
                *self.value.as_strvector.as_ref(),
            ),
            _ => igraph_error_type_t_IGRAPH_SUCCESS,
        })
        .expect("igraph failed to copy an attribute record");
        if let Some(default) = self.default_value() {
            copy.set_default(default)
                .expect("igraph failed to copy the default value of an attribute record");
        }
        copy
    }
}

impl PartialEq for igraph_attribute_record_t {
    /// Records are equal when their names, types, values and default values
    /// are. Records are compared as data: a `NaN` number equals another
    /// `NaN` (the default value of a numeric record is `NaN` unless changed,
    /// so with `f64` semantics a numeric record would never equal its clone).
    fn eq(&self, other: &Self) -> bool {
        fn same(a: f64, b: f64) -> bool {
            a == b || (a.is_nan() && b.is_nan())
        }
        let same_values = match (self.values(), other.values()) {
            (AttributeValues::Numeric(a), AttributeValues::Numeric(b)) => {
                a.len() == b.len() && a.iter().zip(&b).all(|(&x, &y)| same(x, y))
            }
            (a, b) => a == b,
        };
        let same_default = match (self.default_value(), other.default_value()) {
            (Some(AttributeValue::Numeric(a)), Some(AttributeValue::Numeric(b))) => same(a, b),
            (a, b) => a == b,
        };
        self.name() == other.name()
            && self.attribute_type() == other.attribute_type()
            && same_values
            && same_default
    }
}

impl fmt::Debug for igraph_attribute_record_t {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AttributeRecord")
            .field("name", &self.name())
            .field("type", &self.attribute_type())
            .field("values", &self.values())
            .field("default", &self.default_value())
            .finish()
    }
}

// A record exclusively owns its heap data.
unsafe impl Send for igraph_attribute_record_t {}
unsafe impl Sync for igraph_attribute_record_t {}

// ---------------------------------------------------------------------------
// Attribute combinations
// ---------------------------------------------------------------------------

/// Signature of a numeric combination function of the C attribute handler:
/// `input` holds the values of the merged elements, the result goes to `output`.
pub type NumericCombineFn = unsafe extern "C" fn(
    input: *const igraph_vector_t,
    output: *mut igraph_real_t,
) -> igraph_error_t;

/// Signature of a boolean combination function of the C attribute handler.
pub type BooleanCombineFn = unsafe extern "C" fn(
    input: *const igraph_vector_bool_t,
    output: *mut igraph_bool_t,
) -> igraph_error_t;

/// A user supplied C function combining attribute values, see
/// [`AttributeCombination::add_function`].
#[derive(Debug, Clone, Copy)]
pub enum CombineFunction {
    /// For numeric attributes.
    Numeric(NumericCombineFn),
    /// For boolean attributes.
    Boolean(BooleanCombineFn),
}

/// How to combine attributes when vertices or edges are merged
/// (`igraph_attribute_combination_t`).
///
/// It maps attribute names to an [`AttributeCombinationType`]; the entry
/// with no name (`None`) is the default for every attribute not listed. An
/// empty combination drops all the attributes of merged elements.
///
/// Pass it to [`Graph::simplify_with_attributes`],
/// [`Graph::contract_vertices_with_attributes`] or
/// [`Graph::to_undirected_with_attributes`], or to other igraph functions
/// merging elements through the raw FFI with [`as_ptr`](Self::as_ptr).
///
/// ```
/// use igraph::attributes::{AttributeCombination, AttributeCombinationType as Comb};
///
/// let comb = AttributeCombination::from_pairs(&[
///     (Some("weight"), Comb::Sum),
///     (Some("name"), Comb::First),
///     (None, Comb::Ignore),
/// ]).unwrap();
/// assert_eq!(comb.query(Some("weight")).unwrap(), Comb::Sum);
/// assert_eq!(comb.query(Some("color")).unwrap(), Comb::Ignore); // falls back to the default
///
/// // Merge the two parallel edges: weights are summed, other attributes dropped.
/// use igraph::prelude::*;
/// let mut g = Graph::from_edges(&[(0, 1), (0, 1)], 2, false).unwrap();
/// g.set_edge_attr_numeric_values("weight", &[1.5, 2.0]).unwrap();
/// g.set_edge_attr_str_values("color", &["red", "blue"]).unwrap();
/// g.simplify_with_attributes(true, true, &comb).unwrap();
/// assert_eq!(g.edge_attr_numeric("weight", 0).unwrap(), 3.5);
/// assert!(!g.has_attribute(igraph::attributes::AttributeKind::Edge, "color"));
/// ```
pub type AttributeCombination = igraph_attribute_combination_t;

impl igraph_attribute_combination_t {
    /// Creates an empty combination list
    /// ([`igraph_attribute_combination_init`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_attribute_combination_init)).
    ///
    /// # Panics
    ///
    /// If igraph fails to allocate it.
    pub fn new() -> Self {
        ensure_init();
        let mut raw = MaybeUninit::<Self>::uninit();
        crate::error::check(unsafe { igraph_attribute_combination_init(raw.as_mut_ptr()) })
            .expect("igraph failed to allocate an attribute combination");
        unsafe { raw.assume_init() }
    }

    /// Creates a combination list from `(name, type)` pairs, `None` naming the
    /// default entry (the Rust counterpart of the variadic
    /// [`igraph_attribute_combination`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_attribute_combination)).
    ///
    /// # Errors
    ///
    /// As [`add`](Self::add).
    pub fn from_pairs(pairs: &[(Option<&str>, AttributeCombinationType)]) -> Result<Self> {
        let mut comb = Self::new();
        for &(name, kind) in pairs {
            comb.add(name, kind)?;
        }
        Ok(comb)
    }

    /// A combination applying `kind` to every attribute (a single default
    /// entry).
    ///
    /// # Errors
    ///
    /// As [`add`](Self::add).
    ///
    /// ```
    /// use igraph::attributes::{AttributeCombination, AttributeCombinationType as Comb};
    /// let comb = AttributeCombination::all(Comb::Max).unwrap();
    /// assert_eq!(comb.query(None).unwrap(), Comb::Max);
    /// assert_eq!(comb.query(Some("anything")).unwrap(), Comb::Max);
    /// ```
    pub fn all(kind: AttributeCombinationType) -> Result<Self> {
        Self::from_pairs(&[(None, kind)])
    }

    /// Adds or replaces the entry of an attribute (`None`: the default entry)
    /// ([`igraph_attribute_combination_add`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_attribute_combination_add)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidValue`] for [`AttributeCombinationType::Function`]
    /// (use [`add_function`](Self::add_function)) or a name containing a NUL byte.
    pub fn add(&mut self, name: Option<&str>, kind: AttributeCombinationType) -> Result<()> {
        if kind == AttributeCombinationType::Function {
            return Err(Error::invalid(
                "use `add_function` to combine attributes with a function",
            ));
        }
        let c = name.map(|n| c_string("attribute name", n)).transpose()?;
        let p = c.as_ref().map_or(ptr::null(), |c| c.as_ptr());
        igraph_call!(igraph_attribute_combination_add(self, p, kind.into(), None))
    }

    /// Adds or replaces the entry of an attribute with a user supplied C
    /// function ([`igraph_attribute_combination_add`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_attribute_combination_add)
    /// with `IGRAPH_ATTRIBUTE_COMBINE_FUNCTION`).
    ///
    /// For each merged vertex or edge, the function receives the values of the
    /// merged elements and writes the combined value.
    ///
    /// # Safety
    ///
    /// The C attribute handler chooses the calling convention from the *type
    /// of the attribute* at combination time: the function must match it, i.e.
    /// a [`CombineFunction::Numeric`] may only be registered for numeric
    /// attributes and a [`CombineFunction::Boolean`] for boolean ones
    /// (registering a default entry with `name = None` requires *every*
    /// combined attribute to have that type). The function must not unwind.
    /// String functions are not supported by this wrapper (igraph would need to
    /// free a string allocated by Rust).
    ///
    /// ```
    /// use igraph::{ffi, attributes::{AttributeCombination, CombineFunction}};
    ///
    /// unsafe extern "C" fn count(
    ///     input: *const ffi::igraph_vector_t,
    ///     output: *mut f64,
    /// ) -> ffi::igraph_error_t {
    ///     unsafe { *output = (&*input).len() as f64 };
    ///     ffi::igraph_error_type_t_IGRAPH_SUCCESS
    /// }
    ///
    /// let mut comb = AttributeCombination::new();
    /// // SAFETY: "multiplicity" will be a numeric attribute.
    /// unsafe { comb.add_function(Some("multiplicity"), CombineFunction::Numeric(count)) }.unwrap();
    /// ```
    pub unsafe fn add_function(&mut self, name: Option<&str>, func: CombineFunction) -> Result<()> {
        let c = name.map(|n| c_string("attribute name", n)).transpose()?;
        let p = c.as_ref().map_or(ptr::null(), |c| c.as_ptr());
        // SAFETY: igraph stores the pointer type-erased and casts it back to the
        // signature matching the attribute type (the caller's responsibility).
        let erased: unsafe extern "C" fn() = unsafe {
            match func {
                CombineFunction::Numeric(f) => {
                    std::mem::transmute::<NumericCombineFn, unsafe extern "C" fn()>(f)
                }
                CombineFunction::Boolean(f) => {
                    std::mem::transmute::<BooleanCombineFn, unsafe extern "C" fn()>(f)
                }
            }
        };
        igraph_call!(igraph_attribute_combination_add(
            self,
            p,
            AttributeCombinationType::Function.into(),
            Some(erased)
        ))
    }

    /// Removes the entry of an attribute (`None`: the default entry); missing
    /// entries are ignored
    /// ([`igraph_attribute_combination_remove`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_attribute_combination_remove)).
    pub fn remove(&mut self, name: Option<&str>) -> Result<()> {
        let c = name.map(|n| c_string("attribute name", n)).transpose()?;
        let p = c.as_ref().map_or(ptr::null(), |c| c.as_ptr());
        igraph_call!(igraph_attribute_combination_remove(self, p))
    }

    /// How the attribute `name` will be combined
    /// ([`igraph_attribute_combination_query`](https://igraph.org/c/html/latest/igraph-Attributes.html#igraph_attribute_combination_query)).
    ///
    /// Falls back to the default entry, and to
    /// [`AttributeCombinationType::Default`] when there is none.
    pub fn query(&self, name: Option<&str>) -> Result<AttributeCombinationType> {
        let c = name.map(|n| c_string("attribute name", n)).transpose()?;
        let p = c.as_ref().map_or(ptr::null(), |c| c.as_ptr());
        let mut kind = igraph_attribute_combination_type_t_IGRAPH_ATTRIBUTE_COMBINE_DEFAULT;
        let mut func: igraph_function_pointer_t = None;
        igraph_call!(igraph_attribute_combination_query(
            self, p, &mut kind, &mut func
        ))?;
        AttributeCombinationType::try_from(kind)
    }

    /// Raw pointer for the igraph functions taking a
    /// `const igraph_attribute_combination_t *`.
    pub fn as_ptr(&self) -> *const igraph_attribute_combination_t {
        self
    }
}

impl Default for igraph_attribute_combination_t {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for igraph_attribute_combination_t {
    /// Frees the list (`igraph_attribute_combination_destroy`).
    fn drop(&mut self) {
        if !self.list.stor_begin.is_null() {
            unsafe { igraph_attribute_combination_destroy(self) };
            self.list.stor_begin = ptr::null_mut();
        }
    }
}

// The list exclusively owns its records.
unsafe impl Send for igraph_attribute_combination_t {}

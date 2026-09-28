//! Integration tests of the foreign formats module (`igraph_foreign.h`).
//!
//! The attribute handler is process-wide: the tests about attributes enable
//! it (see `attrs()`), the others hold whether it is enabled or not, since
//! the tests of this binary run in parallel in an unspecified order.

mod common;

use common::*;
use igraph::{
    attributes::{self, AttributeKind, AttributeType},
    error::take_warnings,
    foreign::*,
    prelude::*,
};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

/// A temporary file, unique per process and per call, removed on drop.
struct TempFile(PathBuf);

impl TempFile {
    fn new(tag: &str, ext: &str) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let name = format!("igraph-rs-foreign-{}-{tag}-{n}.{ext}", std::process::id());
        Self(std::env::temp_dir().join(name))
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Sorted degree sequence, an isomorphism invariant.
fn degree_sequence(g: &Graph) -> Vec<i64> {
    let mut d = g
        .degree(VertexSelector::All, NeighborMode::All, Loops::Twice)
        .unwrap();
    d.sort_unstable();
    d
}

/// Edges as unordered, sorted pairs (to compare undirected graphs whose
/// endpoints may be stored in either order).
fn undirected_edges(g: &Graph) -> Vec<(i64, i64)> {
    let mut e: Vec<_> = g
        .edge_list()
        .into_iter()
        .map(|(a, b)| (a.min(b), a.max(b)))
        .collect();
    e.sort_unstable();
    e
}

fn directed_karate() -> Graph {
    Graph::from_edges(&KARATE_EDGES, 34, true).unwrap()
}

/// Turns on the (process-wide) attribute handler, before reading files whose
/// attributes a test inspects.
fn attrs() {
    attributes::enable().unwrap();
}

/// Compares float slices treating NaN as equal to NaN.
fn assert_same_floats(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a.is_nan() && e.is_nan()) || a == e,
            "{actual:?} vs {expected:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Edge list
// ---------------------------------------------------------------------------

#[test]
fn edgelist_output_is_sorted_by_first_endpoint() {
    // Edge ids are 0: 2->0, 1: 0->1, 2: 1->2, but lines follow the source vertex.
    let g = Graph::from_edges(&[(2, 0), (0, 1), (1, 2)], 3, true).unwrap();
    assert_eq!(
        g.write_graph_edgelist_to_string().unwrap(),
        "0 1\n1 2\n2 0\n"
    );
}

#[test]
fn edgelist_string_round_trip_is_identical() {
    for g in [karate(), directed_karate(), cycle(7)] {
        let text = g.write_graph_edgelist_to_string().unwrap();
        assert_eq!(text.lines().count(), g.ecount());
        let h = Graph::read_graph_edgelist_from_str(&text, g.vcount(), g.is_directed()).unwrap();
        assert!(g.is_same_graph(&h).unwrap());
        assert_eq!(g, h);
    }
}

#[test]
fn edgelist_file_round_trip_keeps_isolated_vertices_only_with_n() {
    let mut g = path(4);
    g.add_vertices(3).unwrap(); // vertices 4, 5, 6 are isolated
    let file = TempFile::new("edgelist", "txt");
    g.write_graph_edgelist(&file.0).unwrap();
    assert_eq!(std::fs::read_to_string(&file.0).unwrap(), "0 1\n1 2\n2 3\n");

    let lossy = Graph::read_graph_edgelist(&file.0, 0, false).unwrap();
    assert_eq!(lossy.vcount(), 4);
    let exact = Graph::read_graph_edgelist(&file.0, 7, false).unwrap();
    assert_eq!(exact, g);
}

#[test]
fn edgelist_parsing_details() {
    // Any whitespace separates ids; the vertex count is max(n, max id + 1).
    let g = Graph::read_graph_edgelist_from_str("  0 3\t3 3\n\n\n 1\n2 ", 2, true).unwrap();
    assert_eq!(g.vcount(), 4);
    assert_eq!(g.edge_list(), vec![(0, 3), (3, 3), (1, 2)]);
    // The empty input is a graph with `n` isolated vertices.
    let empty = Graph::read_graph_edgelist_from_str("", 5, false).unwrap();
    assert_eq!((empty.vcount(), empty.ecount()), (5, 0));
}

#[test]
fn edgelist_syntax_errors() {
    for bad in ["0 1 2", "0 x", "0 -1", "1.5 2"] {
        let err = Graph::read_graph_edgelist_from_str(bad, 0, false).unwrap_err();
        assert!(
            matches!(
                err.kind(),
                ErrorKind::Parse | ErrorKind::InvalidValue | ErrorKind::InvalidVertexId
            ),
            "{bad:?}: {err}"
        );
    }
}

// ---------------------------------------------------------------------------
// File handling errors
// ---------------------------------------------------------------------------

#[test]
fn missing_file_is_a_file_error() {
    let missing = std::env::temp_dir().join("igraph-rs-definitely/missing/file.txt");
    let err = Graph::read_graph_edgelist(&missing, 0, false).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::File);
    assert!(err.message().contains("file.txt"), "{}", err.message());
    assert_eq!(
        Graph::read_graph_gml(&missing).unwrap_err().kind(),
        ErrorKind::File
    );
    assert_eq!(
        Graph::read_graph_graphdb(&missing, true)
            .unwrap_err()
            .kind(),
        ErrorKind::File
    );
}

#[test]
fn writing_into_a_missing_directory_fails_cleanly() {
    let g = cycle(3);
    let target = std::env::temp_dir().join("igraph-rs-no-such-dir/sub/out.graphml");
    // The old binding ignored this failure (and leaked a FILE*); now it's reported.
    let err = g.write_graph_graphml(&target).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::File);
    assert_eq!(
        g.write_graph_dot(&target).unwrap_err().kind(),
        ErrorKind::File
    );
}

#[test]
fn interior_nul_in_path_is_rejected() {
    let err = Graph::read_graph_pajek("bad\0name.net").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

#[test]
fn non_ascii_paths_work() {
    let file = TempFile::new("grafo-ñandú-図", "gml");
    let g = karate();
    g.write_graph_gml(&file.0, &GmlWriteOptions::default())
        .unwrap();
    let h = Graph::read_graph_gml(&file.0).unwrap();
    assert_eq!(undirected_edges(&g), undirected_edges(&h));
}

// ---------------------------------------------------------------------------
// NCOL and LGL
// ---------------------------------------------------------------------------

#[test]
fn ncol_names_get_ids_by_first_appearance() {
    let text = "paris london 3.5\nlondon berlin\nberlin paris 1e1\nrome paris\n";
    let g = Graph::read_graph_ncol_from_str(text, &[], &NcolLglOptions::default()).unwrap();
    assert!(!g.is_directed());
    // paris=0, london=1, berlin=2, rome=3.
    assert_eq!(g.vcount(), 4);
    assert_eq!(undirected_edges(&g), vec![(0, 1), (0, 2), (0, 3), (1, 2)]);

    let directed = Graph::read_graph_ncol_from_str(
        text,
        &[],
        &NcolLglOptions::default()
            .with_directed(true)
            .with_weights(AddWeights::Yes),
    )
    .unwrap();
    assert!(directed.is_directed());
    assert_eq!(directed.edge_list(), vec![(0, 1), (1, 2), (2, 0), (3, 0)]);
}

#[test]
fn ncol_predefined_names_fix_the_ids_and_extend_with_a_warning() {
    take_warnings();
    let text = "a b\nb c\nz a\n";
    let g = Graph::read_graph_ncol_from_str(text, &["c", "b", "a"], &NcolLglOptions::default())
        .unwrap();
    // c=0, b=1, a=2 and the unknown z gets the next id, 3.
    assert_eq!(g.edge_list().len(), 3);
    assert_eq!(undirected_edges(&g), vec![(0, 1), (1, 2), (2, 3)]);
    let warnings = take_warnings();
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("predefined names extended")),
        "{warnings:?}"
    );
}

#[test]
fn ncol_write_with_missing_attributes_falls_back_to_ids() {
    // Missing attributes (always missing without the attribute handler) are
    // skipped with a warning.
    take_warnings();
    let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    let text = g
        .write_graph_ncol_to_string(Some("name"), Some("weight"))
        .unwrap();
    assert_eq!(text, "0 1\n1 2\n");
    let warnings = take_warnings();
    assert!(
        warnings.iter().any(|w| w.contains("'name' does not exist")),
        "{warnings:?}"
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("'weight' does not exist")),
        "{warnings:?}"
    );
    assert_eq!(
        g.write_graph_ncol_to_string(Some("a\0b"), None)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn ncol_file_round_trip() {
    let g = karate();
    let file = TempFile::new("karate", "ncol");
    g.write_graph_ncol(&file.0, None, None).unwrap();
    // Numeric ids are just names: their first appearance order is 0, 1, 2, ...
    // only because the karate edge list is sorted by the first endpoint.
    let h = Graph::read_graph_ncol(&file.0, &[], &NcolLglOptions::default()).unwrap();
    assert_eq!(degree_sequence(&g), degree_sequence(&h));
    assert_eq!(h.ecount(), 78);
    // With the ids as predefined names the graph is reproduced exactly.
    let names: Vec<String> = (0..34).map(|i| i.to_string()).collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let exact = Graph::read_graph_ncol(&file.0, &names, &NcolLglOptions::default()).unwrap();
    assert_eq!(undirected_edges(&exact), undirected_edges(&g));
}

#[test]
fn lgl_matches_igraph_example_output() {
    // examples/simple/igraph_write_graph_lgl.c and its .out file.
    let g = Graph::from_edges(&[(0, 1), (1, 3), (1, 2), (2, 0), (4, 2), (3, 4)], 7, false).unwrap();
    assert_eq!(
        g.write_graph_lgl_to_string(None, None, false).unwrap(),
        "# 0\n1\n2\n# 1\n2\n3\n# 2\n4\n# 3\n4\n"
    );
    assert_eq!(
        g.write_graph_lgl_to_string(None, None, true).unwrap(),
        "# 0\n1\n2\n# 1\n2\n3\n# 2\n4\n# 3\n4\n# 5\n# 6\n"
    );
}

#[test]
fn lgl_round_trip_with_isolates() {
    let mut g = karate();
    g.add_vertices(2).unwrap();
    let file = TempFile::new("karate", "lgl");
    g.write_graph_lgl(&file.0, None, None, true).unwrap();
    let h = Graph::read_graph_lgl(&file.0, &NcolLglOptions::default()).unwrap();
    assert_eq!((h.vcount(), h.ecount()), (36, 78));
    assert_eq!(degree_sequence(&g), degree_sequence(&h));
    // Without isolates the two extra vertices are lost.
    g.write_graph_lgl(&file.0, None, None, false).unwrap();
    assert_eq!(
        Graph::read_graph_lgl(&file.0, &NcolLglOptions::default())
            .unwrap()
            .vcount(),
        34
    );
}

#[test]
fn lgl_weights_and_errors() {
    let g = Graph::read_graph_lgl_from_str(
        "# hub\nx 1.5\ny -2\nz\n",
        &NcolLglOptions::default()
            .with_names(false)
            .with_weights(AddWeights::Yes),
    )
    .unwrap();
    assert_eq!(g.edge_list(), vec![(0, 1), (0, 2), (0, 3)]);
    // An edge line before any "# vertex" header is a syntax error.
    let err = Graph::read_graph_lgl_from_str("x\n# a\n", &NcolLglOptions::default()).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Parse);
}

// ---------------------------------------------------------------------------
// Pajek
// ---------------------------------------------------------------------------

#[test]
fn pajek_output_matches_igraph_example() {
    // examples/simple/igraph_write_graph_pajek.c: a directed ring on 10 vertices.
    let edges: Vec<(i64, i64)> = (0..10).map(|i| (i, (i + 1) % 10)).collect();
    let ring = Graph::from_edges(&edges, 10, true).unwrap();
    let expected = "*Vertices 10\n*Arcs\n1 2\n2 3\n3 4\n4 5\n5 6\n6 7\n7 8\n8 9\n9 10\n10 1\n";
    assert_eq!(ring.write_graph_pajek_to_string().unwrap(), expected);
    assert_eq!(Graph::read_graph_pajek_from_str(expected).unwrap(), ring);
}

#[test]
fn pajek_round_trip_undirected_file() {
    let g = karate();
    let file = TempFile::new("karate", "net");
    g.write_graph_pajek(&file.0).unwrap();
    let text = std::fs::read_to_string(&file.0).unwrap();
    assert!(text.starts_with("*Vertices 34\n*Edges\n"));
    let h = Graph::read_graph_pajek(&file.0).unwrap();
    assert!(!h.is_directed());
    assert_eq!(undirected_edges(&g), undirected_edges(&h));
}

#[test]
fn pajek_reads_attributes_lists_and_bipartite_matrices() {
    // tests/unit/pajek1.net (with visual attributes, discarded here).
    let mut text = String::from("*Vertices 10\n");
    for i in 1..=10 {
        text.push_str(&format!(
            "{i} \"Vert {i}\" 0 0 box x_fact 1 y_fact 1 ic Green\n"
        ));
    }
    text.push_str("*Edges\n");
    for i in 1..10 {
        text.push_str(&format!("{i} {}\n", i + 1));
    }
    let g = Graph::read_graph_pajek_from_str(&text).unwrap();
    assert_eq!((g.vcount(), g.ecount(), g.is_directed()), (10, 9, false));

    // *Arcslist: each line lists a vertex and its out-neighbors.
    let g = Graph::read_graph_pajek_from_str("*Vertices 4\n*Arcslist\n1 2 3 4\n3 1\n").unwrap();
    assert!(g.is_directed());
    assert_eq!(g.edge_list(), vec![(0, 1), (0, 2), (0, 3), (2, 0)]);

    // tests/unit/pajek_bip.net: a two-mode network, 10 + 5 vertices.
    let bip = "*vertices 15 10\n\
               1 \"A\"\n2 \"B\"\n3 \"C\"\n4 \"D\"\n5 \"E\"\n6 \"F\"\n7 \"G\"\n8 \"H\"\n9 \"I\"\n10 \"J\"\n\
               11 \"1\"\n12 \"2\"\n13 \"3\"\n14 \"4\"\n15 \"5\"\n\
               *matrix\n1 0 0 0 0\n1 1 0 0 0\n1 1 1 0 0\n1 1 1 1 0\n1 1 1 1 1\n\
               0 0 0 0 1\n1 0 0 0 1\n1 1 0 1 1\n0 0 0 0 0\n1 0 1 0 1\n";
    let g = Graph::read_graph_pajek_from_str(bip).unwrap();
    assert_eq!((g.vcount(), g.ecount()), (15, 25));
    // Every edge joins a "letter" vertex (0..10) to a "number" vertex (10..15).
    for (a, b) in g.edge_list() {
        assert!((a < 10) != (b < 10), "edge {a}-{b} inside one mode");
    }
    assert_eq!(g.degree_of(8, NeighborMode::All, Loops::Twice).unwrap(), 0); // row "I" is empty
}

#[test]
fn pajek_rejects_out_of_range_vertices() {
    let err = Graph::read_graph_pajek_from_str("*Vertices 2\n*Edges\n1 5\n").unwrap_err();
    assert!(
        matches!(err.kind(), ErrorKind::Parse | ErrorKind::InvalidValue),
        "{err}"
    );
}

// ---------------------------------------------------------------------------
// GraphML
// ---------------------------------------------------------------------------

/// examples/simple/test.graphml (abridged comments), written by the JAVA GraphML Library.
const TEST_GRAPHML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <key id="d0" for="node" attr.name="color" attr.type="string"><default>yellow</default></key>
  <key id="d1" for="edge" attr.name="weight" attr.type="double"/>
  <graph id="G" edgedefault="undirected">
    <node id="n0"><data key="d0">green</data></node>
    <node id="n1"/>
    <node id="n2"><data key="d0">blue</data></node>
    <node id="n3"><data key="d0">red &quot;with entities&quot;</data></node>
    <node id="n4"/>
    <node id="n5"><data key="d0">turquoise</data></node>
    <edge id="e0" source="n0" target="n2"><data key="d1">1.0</data></edge>
    <edge id="e1" source="n0" target="n1"><data key="d1">1.0</data></edge>
    <edge id="e2" source="n1" target="n3"><data key="d1">2.0</data></edge>
    <edge id="e3" source="n3" target="n2"/>
    <edge id="e4" source="n2" target="n4"/>
    <edge id="e5" source="n3" target="n5"/>
    <edge id="e6" source="n5" target="n4"><data key="d1">1.1</data></edge>
  </graph>
  <graph id="H" edgedefault="directed">
    <node id="a"/><node id="b"/>
    <edge source="b" target="a"/>
  </graph>
</graphml>
"#;

#[test]
fn graphml_parses_known_document_and_selects_graphs_by_index() {
    let g = Graph::read_graph_graphml_from_str(TEST_GRAPHML, 0).unwrap();
    assert!(!g.is_directed());
    assert_eq!((g.vcount(), g.ecount()), (6, 7));
    assert_eq!(
        undirected_edges(&g),
        vec![(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]
    );
    // igraph 1.0.0 and 1.0.1 only reach the first graph of a multi-graph
    // document: any larger index reports "Graph index was too large".
    let err = Graph::read_graph_graphml_from_str(TEST_GRAPHML, 1).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    assert!(err.message().contains("index"), "{err}");
}

#[test]
fn graphml_legacy_method_name_round_trip() {
    for g in [karate(), directed_karate()] {
        let file = TempFile::new("karate", "graphml");
        g.write_graph_graphml(&file.0).unwrap();
        let h = Graph::read_graph_graphml(&file.0, 0).unwrap();
        assert_eq!(h, g, "GraphML keeps vertex ids and edge order");
        // The prefixed variant and the in-memory one produce the same document
        // when there are no attributes.
        g.write_graph_graphml_with(&file.0, true).unwrap();
        assert_eq!(
            std::fs::read_to_string(&file.0).unwrap(),
            g.write_graph_graphml_to_string(false).unwrap()
        );
    }
}

#[test]
fn graphml_malformed_input_is_a_parse_error() {
    let err = Graph::read_graph_graphml_from_str("<graphml><graph>", 0).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Parse, "{err}");
    let err = Graph::read_graph_graphml_from_str("", 0).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Parse, "{err}");
}

// ---------------------------------------------------------------------------
// GML
// ---------------------------------------------------------------------------

/// tests/unit/graph1.gml.
const GRAPH1_GML: &str = r#"# comment lines are ignored
Version 1
graph [
    directed 1
    node [
        id 1
        a [ b 1 ]
        b +2
        c 5.6
        graphics [ x 0 y 0 ]
    ]
    node [
        a 1
        id 2
        a 2
        b "asd"
        d -Inf
        nan "foo"
    ]
    edge [
        source 5 target 1
        weight -0.45
        label -0.45
        num1 NAN
    ]
    node [
        id 5
        c [ str "bar" ]
    ]
    edge [
        source 2 target 5
        label "Tom &amp; Jerry&apos;s &quot;friendship&quot;"
        sou_rce 1.0
        num1 +inF
    ]
    AttOne 1
    AttOne "x"
    AttTwo 3.14
]

# second graph is ignored
graph [ node [ ] ]
"#;

#[test]
fn gml_parses_igraph_unit_test_file() {
    let g = Graph::read_graph_gml_from_str(GRAPH1_GML).unwrap();
    assert!(g.is_directed());
    // GML ids 1, 2, 5 become vertices 0, 1, 2.
    assert_eq!(g.vcount(), 3);
    assert_eq!(g.edge_list(), vec![(2, 0), (1, 2)]);
}

#[test]
fn gml_write_options() {
    let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    let default = g
        .write_graph_gml_to_string(&GmlWriteOptions::default())
        .unwrap();
    assert!(default.starts_with("Creator \"igraph version"), "{default}");

    let quiet = g
        .write_graph_gml_to_string(&GmlWriteOptions::default().with_creator(""))
        .unwrap();
    assert!(
        quiet.starts_with("Version 1\ngraph\n[\n  directed 1\n"),
        "{quiet}"
    );

    let custom = g
        .write_graph_gml_to_string(
            &GmlWriteOptions::default()
                .with_creator("my tool")
                .with_ids(&[100.0, 200.0, 300.0]),
        )
        .unwrap();
    assert!(custom.starts_with("Creator \"my tool\"\n"));
    assert!(custom.contains("source 100\n    target 200"));
    // Reading back maps the custom ids to consecutive vertices again.
    assert_eq!(Graph::read_graph_gml_from_str(&custom).unwrap(), g);

    // Non-integer ids are ignored altogether by igraph (with a warning).
    let fallback = g
        .write_graph_gml_to_string(
            &GmlWriteOptions::default()
                .with_creator("")
                .with_ids(&[0.5, 1.0, 2.0]),
        )
        .unwrap();
    assert_eq!(fallback, quiet);

    // The id vector must have one entry per vertex.
    let err = g
        .write_graph_gml_to_string(&GmlWriteOptions::default().with_ids(&[1.0]))
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    // Duplicate ids are ignored altogether too, with a warning.
    take_warnings();
    let dup = g
        .write_graph_gml_to_string(
            &GmlWriteOptions::default()
                .with_creator("")
                .with_ids(&[7.0, 7.0, 8.0]),
        )
        .unwrap();
    assert_eq!(dup, quiet);
    assert!(!take_warnings().is_empty());
    let only_quot = GmlWriteOptions::default()
        .with_encode_only_quot(true)
        .with_creator("");
    assert_eq!(g.write_graph_gml_to_string(&only_quot).unwrap(), quiet);
}

#[test]
fn gml_creator_with_special_characters_stays_valid_gml() {
    // igraph 1.0.0 and 1.0.1 would print the raw creator, and a `"` would
    // terminate the GML string early; the wrapper encodes it as an entity.
    let g = Graph::from_edges(&[(0, 1)], 2, false).unwrap();
    let gml = g
        .write_graph_gml_to_string(&GmlWriteOptions::default().with_creator(r#"Tom & "Jerry""#))
        .unwrap();
    assert!(
        gml.starts_with("Creator \"Tom &amp; &quot;Jerry&quot;\"\n"),
        "{gml}"
    );
    assert_eq!(Graph::read_graph_gml_from_str(&gml).unwrap(), g);
    let gml = g
        .write_graph_gml_to_string(
            &GmlWriteOptions::default()
                .with_creator(r#"Tom & "Jerry""#)
                .with_encode_only_quot(true),
        )
        .unwrap();
    assert!(
        gml.starts_with("Creator \"Tom & &quot;Jerry&quot;\"\n"),
        "{gml}"
    );
    assert_eq!(Graph::read_graph_gml_from_str(&gml).unwrap(), g);
    // NUL bytes can't be passed to C.
    let err = g
        .write_graph_gml_to_string(&GmlWriteOptions::default().with_creator("a\0b"))
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

#[test]
fn oversized_counts_and_nul_names_are_rejected() {
    let err = Graph::read_graph_edgelist_from_str("0 1", usize::MAX, false).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let err = Graph::read_graph_ncol_from_str("a b\n", &["a\0x"], &NcolLglOptions::default())
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let err = cycle(3)
        .write_graph_ncol_to_string(Some("na\0me"), None)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

#[test]
fn gml_errors() {
    // Edge pointing to an unknown node id.
    let err = Graph::read_graph_gml_from_str("graph [ node [ id 1 ] edge [ source 1 target 2 ] ]")
        .unwrap_err();
    assert!(
        matches!(err.kind(), ErrorKind::Parse | ErrorKind::InvalidValue),
        "{err}"
    );
    // Unbalanced brackets.
    let err = Graph::read_graph_gml_from_str("graph [ node [ id 1 ]").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Parse, "{err}");
    // Structural problems are parse errors too (src/io/gml.c).
    for bad in [
        "graph [ node [ id 1 ] node [ id 1 ] ]",
        "graph [ node [ id 1.5 ] ]",
        "graph [ node [ id 1 ] edge [ target 1 ] ]",
        "Version 1",
    ] {
        let err = Graph::read_graph_gml_from_str(bad).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Parse, "{bad:?}: {err}");
    }
    // Nodes without an id are allowed when no edge refers to them.
    let g = Graph::read_graph_gml_from_str("graph [ node [ ] node [ id 3 ] ]").unwrap();
    assert_eq!((g.vcount(), g.ecount()), (2, 0));
}

// ---------------------------------------------------------------------------
// DIMACS
// ---------------------------------------------------------------------------

#[test]
fn dimacs_output_matches_igraph_unit_test() {
    // tests/unit/igraph_write_graph_dimacs_flow.c and its .out file.
    let g = Graph::from_edges(
        &[
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 4),
            (3, 5),
            (4, 5),
        ],
        6,
        true,
    )
    .unwrap();
    let capacity = [5.0, 2.0, 2.0, 3.0, 4.0, 1.0, 2.0, 5.0];
    let text = g
        .write_graph_dimacs_flow_to_string(0, 5, &capacity)
        .unwrap();
    assert_eq!(
        text,
        "c created by igraph\np max 6 8\nn 1 s\nn 6 t\n\
         a 1 2 5\na 1 3 2\na 2 3 2\na 2 4 3\na 3 5 4\na 4 5 1\na 4 6 2\na 5 6 5\n"
    );
    // igraph's unit test also writes the null graph with source 0 and target
    // 5, producing "n 1 s\nn 6 t" lines for vertices that don't exist: the
    // wrapper rejects such ids instead of writing an unreadable instance.
    let err = Graph::new(0, true)
        .write_graph_dimacs_flow_to_string(0, 5, &[])
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    let err = g
        .write_graph_dimacs_flow_to_string(-1, 5, &capacity)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    let err = g
        .write_graph_dimacs_flow_to_string(0, 6, &capacity)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    // ...and the reader rejects them too.
    let err = Graph::read_graph_dimacs_flow_from_str(
        "c created by igraph\np max 0 0\nn 1 s\nn 6 t\n",
        true,
    )
    .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Parse, "{err}");

    let flow = Graph::read_graph_dimacs_flow_from_str(&text, true).unwrap();
    assert_eq!(flow.graph, g);
    assert_eq!(
        flow.problem,
        DimacsProblem::Max {
            source: Some(0),
            target: Some(5),
            capacity: capacity.to_vec()
        }
    );
}

#[test]
fn dimacs_missing_terminals_are_none() {
    // No "n" lines at all: igraph accepts the file, the terminals are unknown.
    let flow =
        Graph::read_graph_dimacs_flow_from_str("p max 3 2\na 1 2 1.5\na 2 3 2\n", true).unwrap();
    assert_eq!(
        flow.problem,
        DimacsProblem::Max {
            source: None,
            target: None,
            capacity: vec![1.5, 2.0]
        }
    );
    // Only the target given.
    let flow = Graph::read_graph_dimacs_flow_from_str("p max 3 1\nn 3 t\na 1 2 1\n", true).unwrap();
    let DimacsProblem::Max { source, target, .. } = flow.problem else {
        panic!("max problem expected");
    };
    assert_eq!((source, target), (None, Some(2)));
    // Vertex 0 does not exist in DIMACS (ids start from 1).
    let err = Graph::read_graph_dimacs_flow_from_str("p max 2 1\nn 0 s\nn 2 t\na 1 2 1\n", true)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Parse);
    // Duplicate source lines are rejected by igraph.
    let err = Graph::read_graph_dimacs_flow_from_str("p max 2 1\nn 1 s\nn 2 s\na 1 2 1\n", true)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Parse);
}

#[test]
fn dimacs_edge_problem_labels() {
    let text = "c a coloring instance\np edge 4 3\ne 1 2\ne 2 3\ne 3 4\n";
    let flow = Graph::read_graph_dimacs_flow_from_str(text, false).unwrap();
    assert_eq!(flow.problem_name, "edge");
    assert!(!flow.graph.is_directed());
    assert_eq!(undirected_edges(&flow.graph), vec![(0, 1), (1, 2), (2, 3)]);
    // Labels default to the 1-based index.
    assert_eq!(
        flow.problem,
        DimacsProblem::Edge {
            labels: vec![1, 2, 3, 4]
        }
    );
    // igraph 1.0.0 and 1.0.1 reject "n <index> <label>" lines in edge
    // problems (`src/io/dimacs.c` reads two fields but expects one).
    let with_label = "p edge 2 1\nn 1 7\ne 1 2\n";
    let err = Graph::read_graph_dimacs_flow_from_str(with_label, false).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Parse);
}

#[test]
fn dimacs_errors() {
    let g = cycle(3);
    let err = g
        .write_graph_dimacs_flow_to_string(0, 1, &[1.0])
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let file = TempFile::new("bad", "dimacs");
    assert_eq!(
        g.write_graph_dimacs_flow(&file.0, 0, 1, &[])
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    // Arc lines are not allowed in edge problems, unknown problem types are rejected.
    assert!(Graph::read_graph_dimacs_flow_from_str("p edge 2 1\na 1 2 3\n", true).is_err());
    assert!(Graph::read_graph_dimacs_flow_from_str("p flow 2 1\n", true).is_err());
}

// ---------------------------------------------------------------------------
// Graph database (binary)
// ---------------------------------------------------------------------------

fn graphdb_bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

#[test]
fn graphdb_reads_igraph_unit_test_data_and_rejects_bad_files() {
    // tests/unit/si2_b06m_s20.A98.
    let good = graphdb_bytes(&[4, 1, 2, 1, 0, 0, 2, 0, 2]);
    let g = Graph::read_graph_graphdb_from_bytes(&good, true).unwrap();
    assert!(g.is_directed());
    assert_eq!(g.vcount(), 4);
    assert_eq!(g.edge_list(), vec![(0, 2), (1, 0), (3, 0), (3, 2)]);

    // Trailing bytes and truncated files are parse errors.
    let mut extra = good.clone();
    extra.extend([0, 0]);
    let err = Graph::read_graph_graphdb_from_bytes(&extra, true).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Parse);
    let err = Graph::read_graph_graphdb_from_bytes(&good[..good.len() - 3], true).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Parse);
    // An empty input has not even the vertex count.
    assert!(Graph::read_graph_graphdb_from_bytes(&[], true).is_err());
}

#[test]
fn graphdb_from_file() {
    // Encode the directed karate club in the graph database format by hand.
    let g = directed_karate();
    let mut words = vec![g.vcount() as u16];
    for v in g.vertices() {
        let out = g.neighbors(v, NeighborMode::Out).unwrap();
        words.push(out.len() as u16);
        words.extend(out.iter().map(|&u| u as u16));
    }
    let file = TempFile::new("karate", "A00");
    std::fs::write(&file.0, graphdb_bytes(&words)).unwrap();
    assert_eq!(GraphFormat::from_path(&file.0), Some(GraphFormat::GraphDb));
    let h = Graph::read_graph_graphdb(&file.0, true).unwrap();
    assert_eq!(h, g);
    let u = Graph::read_graph(&file.0, GraphFormat::GraphDb, false).unwrap();
    assert!(!u.is_directed());
    assert_eq!(degree_sequence(&u), degree_sequence(&karate()));
    assert!(u.isomorphic(&Graph::famous("Zachary").unwrap()).unwrap());
}

// ---------------------------------------------------------------------------
// UCINET DL
// ---------------------------------------------------------------------------

#[test]
fn dl_all_forms_match_igraph_example_output() {
    // examples/simple/*.dl with the expected edge lists of igraph_read_graph_dl.out.
    let fullmatrix1 = "DL N = 5\nData:\n0 1 1 1 1\n1 0 1 0 0\n1 1 0 0 1\n1 0 0 0 0\n1 0 1 0 0\n";
    let g = Graph::read_graph_dl_from_str(fullmatrix1, true).unwrap();
    assert_eq!(
        g.edge_list(),
        vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 4),
            (3, 0),
            (4, 0),
            (4, 2)
        ]
    );

    let edgelist1 = "DL n=5\nformat = edgelist1\nlabels:\ngeorge, sally, jim, billy, jane\n\
                     data:\n1 2\n1 3\n2 3\n3 1\n4 3\n";
    let expected = vec![(0, 1), (0, 2), (1, 2), (2, 0), (3, 2)];
    assert_eq!(
        Graph::read_graph_dl_from_str(edgelist1, true)
            .unwrap()
            .edge_list(),
        expected
    );

    let nodelist1 = "DL n=5\nformat = nodelist1\nlabels:\ngeorge, sally, jim, billy, jane\n\
                     data:\n1 2 3\n2 3\n3 1\n4 3\n";
    let g = Graph::read_graph_dl_from_str(nodelist1, true).unwrap();
    assert_eq!(g.edge_list(), expected);
    assert_eq!(g.vcount(), 5); // jane is isolated

    let nodelist2 = "DL n=5\nformat = nodelist1\nlabels embedded:\ndata:\n\
                     george sally jim\nsally jim\nbilly george\njane jim\n";
    assert_eq!(
        Graph::read_graph_dl_from_str(nodelist2, true)
            .unwrap()
            .edge_list(),
        vec![(0, 1), (0, 2), (1, 2), (3, 0), (4, 2)]
    );

    let edgelist7 = "DL n=4\nformat = edgelist1\ndata:\n1 2\n2 3\n2 4\n";
    let g = Graph::read_graph_dl_from_str(edgelist7, false).unwrap();
    assert!(!g.is_directed());
    assert_eq!(g.ecount(), 3);
}

#[test]
fn dl_errors() {
    assert_eq!(
        Graph::read_graph_dl_from_str("this is not DL", true)
            .unwrap_err()
            .kind(),
        ErrorKind::Parse
    );
}

// ---------------------------------------------------------------------------
// DOT and LEDA (write only)
// ---------------------------------------------------------------------------

#[test]
fn dot_output_structure() {
    let g = Graph::from_edges(&[(0, 1), (1, 1), (2, 0)], 4, false).unwrap();
    let dot = g.write_graph_dot_to_string().unwrap();
    let body: Vec<&str> = dot.lines().skip(1).collect(); // skip the version comment
    assert_eq!(
        body,
        vec![
            "graph {",
            "  0;",
            "  1;",
            "  2;",
            "  3;",
            "",
            "  1 -- 0;",
            "  1 -- 1;",
            "  2 -- 0;",
            "}"
        ]
    );

    // The first line names the igraph version in use.
    let version = igraph::misc::version();
    assert_eq!(
        dot.lines().next().unwrap(),
        format!("/* Created by igraph {version} */")
    );

    let d = directed_karate();
    let dot = d.write_graph_dot_to_string().unwrap();
    assert!(dot.contains("digraph {"));
    assert_eq!(dot.matches(" -> ").count(), 78);
}

#[test]
fn leda_matches_igraph_unit_test() {
    // tests/unit/igraph_write_graph_leda.out: undirected ring of 5, no attributes.
    let ring = cycle(5);
    let expected = "LEDA.GRAPH\nvoid\nvoid\n-2\n# Vertices\n5\n|{}|\n|{}|\n|{}|\n|{}|\n|{}|\n# Edges\n5\n\
                    1 2 0 |{}|\n1 5 0 |{}|\n2 3 0 |{}|\n3 4 0 |{}|\n4 5 0 |{}|\n";
    assert_eq!(
        ring.write_graph_leda_to_string(None, None).unwrap(),
        expected
    );

    // Missing attributes are warned about and written as void.
    take_warnings();
    assert_eq!(
        ring.write_graph_leda_to_string(Some("name"), Some("weight"))
            .unwrap(),
        expected
    );
    assert!(!take_warnings().is_empty());

    let file = TempFile::new("ring", "gw");
    ring.write_graph_leda(&file.0, None, None).unwrap();
    assert_eq!(std::fs::read_to_string(&file.0).unwrap(), expected);
}

// ---------------------------------------------------------------------------
// Format-agnostic API
// ---------------------------------------------------------------------------

#[test]
fn format_detection_and_capabilities() {
    let cases = [
        ("a.txt", GraphFormat::Edgelist),
        ("a.EDGES", GraphFormat::Edgelist),
        ("a.ncol", GraphFormat::Ncol),
        ("a.lgl", GraphFormat::Lgl),
        ("a.net", GraphFormat::Pajek),
        ("a.graphml", GraphFormat::GraphMl),
        ("a.gml", GraphFormat::Gml),
        ("a.max", GraphFormat::DimacsFlow),
        ("iso_b03_m1000.A00", GraphFormat::GraphDb),
        ("iso_b03_m1000.B00", GraphFormat::GraphDb),
        ("a.dl", GraphFormat::Dl),
        ("a.gv", GraphFormat::Dot),
        ("a.gw", GraphFormat::Leda),
    ];
    for (name, format) in cases {
        assert_eq!(GraphFormat::from_path(name), Some(format), "{name}");
    }
    assert_eq!(GraphFormat::from_path("a.docx"), None);
    assert_eq!(GraphFormat::from_path("a.a0x"), None);
    assert!(GraphFormat::Dot.can_write() && !GraphFormat::Dot.can_read());
    assert!(GraphFormat::Dl.can_read() && !GraphFormat::Dl.can_write());

    let g = cycle(3);
    let file = TempFile::new("x", "dl");
    assert_eq!(
        g.write_graph(&file.0, GraphFormat::Dl).unwrap_err().kind(),
        ErrorKind::Unimplemented
    );
    assert_eq!(
        Graph::read_graph(&file.0, GraphFormat::Leda, false)
            .unwrap_err()
            .kind(),
        ErrorKind::Unimplemented
    );
}

#[test]
fn every_readable_and_writable_format_round_trips_karate() {
    let g = Graph::famous("Zachary").unwrap();
    assert_eq!(g, karate());
    for format in [
        GraphFormat::Edgelist,
        GraphFormat::Ncol,
        GraphFormat::Lgl,
        GraphFormat::Pajek,
        GraphFormat::GraphMl,
        GraphFormat::Gml,
    ] {
        assert!(format.can_read() && format.can_write());
        let file = TempFile::new("roundtrip", &format!("{format:?}").to_lowercase());
        g.write_graph(&file.0, format).unwrap();
        let h = Graph::read_graph(&file.0, format, false).unwrap();
        assert_eq!((h.vcount(), h.ecount()), (34, 78), "{format:?}");
        assert_eq!(degree_sequence(&h), degree_sequence(&g), "{format:?}");
        if format != GraphFormat::Ncol && format != GraphFormat::Lgl {
            // These formats keep the vertex ids: the very same edge set.
            assert_eq!(h, g, "{format:?}");
        } else {
            // NCOL and LGL relabel the vertices by first appearance.
            assert!(h.isomorphic(&g).unwrap(), "{format:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// Streams, locale and threads
// ---------------------------------------------------------------------------

#[test]
fn large_in_memory_outputs_grow_correctly() {
    // 100 000 edges: several reallocations of the open_memstream buffer.
    let n = 100_000;
    let g = cycle(n);
    let text = g.write_graph_edgelist_to_string().unwrap();
    assert_eq!(text.lines().count(), n as usize);
    // Lines are sorted by the first endpoint: the closing edge comes second.
    assert!(text.starts_with("0 1\n0 99999\n1 2\n"));
    let h = Graph::read_graph_edgelist_from_str(&text, 0, false).unwrap();
    assert_eq!(h, g);
}

#[test]
fn safe_locale_guard_and_helper() {
    let g = Graph::from_edges(&[(0, 1)], 2, true).unwrap();
    {
        let _guard = SafeLocale::enter().unwrap();
        let nested = SafeLocale::enter().unwrap(); // nesting is fine
        drop(nested);
        assert!(
            g.write_graph_dimacs_flow_to_string(0, 1, &[2.25])
                .unwrap()
                .ends_with("a 1 2 2.25\n")
        );
    }
    let text = with_safe_locale(|| g.write_graph_pajek_to_string())
        .unwrap()
        .unwrap();
    assert_eq!(text, "*Vertices 2\n*Arcs\n1 2\n");
}

#[test]
fn concurrent_io_on_many_threads() {
    let handles: Vec<_> = (3..11)
        .map(|n| {
            std::thread::spawn(move || {
                let g = cycle(n);
                let file = TempFile::new(&format!("thread{n}"), "gml");
                g.write_graph_gml(&file.0, &GmlWriteOptions::default())
                    .unwrap();
                let from_file = Graph::read_graph_gml(&file.0).unwrap();
                let from_mem =
                    Graph::read_graph_pajek_from_str(&g.write_graph_pajek_to_string().unwrap())
                        .unwrap();
                assert_eq!(undirected_edges(&from_file), undirected_edges(&g));
                assert_eq!(undirected_edges(&from_mem), undirected_edges(&g));
                // igraph's GML parser is not reentrant: the wrapper serializes
                // it, so hammering it from all threads must stay correct.
                let gml = std::fs::read_to_string(&file.0).unwrap();
                for _ in 0..100 {
                    let h = Graph::read_graph_gml_from_str(&gml).unwrap();
                    assert_eq!(h.ecount(), n as usize);
                }
                from_file.ecount()
            })
        })
        .collect();
    let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
    assert_eq!(total, (3..11).sum::<i64>() as usize);
}

// ---------------------------------------------------------------------------
// Use case stories
// ---------------------------------------------------------------------------

/// Story: a partner ships a road network as a DIMACS max-flow instance. We
/// load it, check the instance, keep the capacities aside while converting
/// the topology to GraphML for a visualization tool, and when the network
/// comes back (with vertex ids preserved by GraphML) we re-export exactly the
/// same flow problem, verifying the max-flow = min-cut bound by brute force
/// on this small instance.
#[test]
fn use_case_dimacs_to_graphml_and_back() {
    let instance = "c depot (1) to harbour (6)\n\
                    p max 6 9\n\
                    n 1 s\nn 6 t\n\
                    a 1 2 16\na 1 3 13\na 2 3 10\na 3 2 4\na 2 4 12\n\
                    a 3 5 14\na 4 3 9\na 4 6 20\na 5 4 7\n";
    // (One edge short of the famous CLRS network: the 5 -> 6 road is closed.)
    let flow = Graph::read_graph_dimacs_flow_from_str(instance, true).unwrap();
    let DimacsProblem::Max {
        source: Some(source),
        target: Some(target),
        capacity,
    } = flow.problem.clone()
    else {
        panic!("expected a max-flow problem with both terminals");
    };
    assert_eq!((source, target), (0, 5));
    assert_eq!(flow.graph.ecount(), capacity.len());

    // Export the topology for the visualization team...
    let exported = TempFile::new("roads", "graphml");
    flow.graph
        .write_graph(&exported.0, GraphFormat::from_path(&exported.0).unwrap())
        .unwrap();
    // ...and read it back after their round trip.
    let back = Graph::read_graph(&exported.0, GraphFormat::GraphMl, true).unwrap();
    assert_eq!(back, flow.graph);

    // Minimum s-t cut by brute force over the 2^4 vertex subsets containing
    // the source and not the target; with the 5 -> 6 road closed the only way
    // into the harbour is 4 -> 6 (capacity 20), but less than that reaches 4.
    let edges = back.edge_list();
    let mut min_cut = f64::INFINITY;
    for mask in 0u32..16 {
        let side = |v: i64| v == source || (v != target && mask & (1 << (v - 1)) != 0);
        let cut: f64 = edges
            .iter()
            .zip(&capacity)
            .filter(|&(&(a, b), _)| side(a) && !side(b))
            .map(|(_, &c)| c)
            .sum();
        min_cut = min_cut.min(cut);
    }
    // Vertex 4 (id 3) receives at most 12 (from 2) + 7 (from 5) = 19 < 20.
    assert_eq!(min_cut, 19.0);
    // The flow module agrees: max-flow = min-cut = 19, and the cut found is
    // made of the two roads into vertex 4 (edges 2 -> 4 and 5 -> 4).
    assert_eq!(
        back.maxflow_value(source, target, Some(&capacity)).unwrap(),
        19.0
    );
    let cut = back.st_mincut(source, target, Some(&capacity)).unwrap();
    assert_eq!(cut.value, 19.0);
    let mut cut_edges: Vec<_> = cut.cut.iter().map(|&e| back.edge(e).unwrap()).collect();
    cut_edges.sort_unstable();
    assert_eq!(cut_edges, vec![(1, 3), (4, 3)]);

    // Re-export the very same instance.
    let out = TempFile::new("roads", "max");
    back.write_graph_dimacs_flow(&out.0, source, target, &capacity)
        .unwrap();
    let again = Graph::read_graph_dimacs_flow(&out.0, true).unwrap();
    assert_eq!(again.graph, flow.graph);
    assert_eq!(again.problem, flow.problem);
}

/// Story: Zachary's karate club travels through every tool of a lab: the
/// raw edge list from the field notes, NCOL for LGL, Pajek for a colleague,
/// GML for a paper figure. Each conversion must preserve the network: same
/// size, same degree sequence (an isomorphism invariant), same club leaders.
#[test]
fn use_case_karate_club_through_every_tool() {
    let field_notes: String = KARATE_EDGES
        .iter()
        .map(|(a, b)| format!("{a} {b}\n"))
        .collect();
    let g = Graph::read_graph_edgelist_from_str(&field_notes, 34, false).unwrap();
    assert_eq!(g, karate());

    let via_ncol = Graph::read_graph_ncol_from_str(
        &g.write_graph_ncol_to_string(None, None).unwrap(),
        &[],
        &NcolLglOptions::default(),
    )
    .unwrap();
    let via_pajek =
        Graph::read_graph_pajek_from_str(&via_ncol.write_graph_pajek_to_string().unwrap()).unwrap();
    let via_gml = Graph::read_graph_gml_from_str(
        &via_pajek
            .write_graph_gml_to_string(&GmlWriteOptions::default().with_creator("lab"))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(degree_sequence(&via_gml), degree_sequence(&g));

    // The two leaders (instructor and administrator) have the two largest
    // degrees, 16 and 17, whatever ids they got along the way.
    let degrees = via_gml
        .degree(VertexSelector::All, NeighborMode::All, Loops::Twice)
        .unwrap();
    let mut top: Vec<i64> = degrees.clone();
    top.sort_unstable_by(|a, b| b.cmp(a));
    assert_eq!(&top[..2], &[17, 16]);
    // Pajek and GML preserve ids, so the NCOL relabeling is the only change:
    // the leaders are the NCOL ids of original vertices 33 and 0.
    let leader = degrees.iter().position(|&d| d == 17).unwrap();
    assert_eq!(
        via_ncol
            .degree_of(leader as i64, NeighborMode::All, Loops::Twice)
            .unwrap(),
        17
    );
}

#[cfg(target_os = "linux")]
#[test]
fn write_errors_on_a_full_device_are_reported() {
    // /dev/full accepts the open but fails every write with ENOSPC: whether
    // igraph notices it or the final flush in `fclose` does, the caller
    // must get a File error rather than a silently truncated file.
    let full = std::path::Path::new("/dev/full");
    if !full.exists() {
        return;
    }
    let g = karate();
    for result in [
        g.write_graph_edgelist(full),
        g.write_graph_pajek(full),
        g.write_graph_graphml(full),
        g.write_graph_gml(full, &GmlWriteOptions::default().with_creator("")),
    ] {
        assert_eq!(result.unwrap_err().kind(), ErrorKind::File);
    }
}

// ---------------------------------------------------------------------------
// Attributes (with the attribute handler enabled)
// ---------------------------------------------------------------------------

#[test]
fn ncol_attributes_round_trip_like_igraph_unit_test() {
    // tests/unit/ncol.c and ncol.out: names and weights (some NaN, one -Inf)
    // survive a write/read cycle of a directed multigraph.
    attrs();
    let mut g =
        Graph::from_edges(&[(0, 1), (0, 1), (0, 1), (1, 2), (2, 3), (3, 0)], 4, true).unwrap();
    g.set_edge_attr_numeric("edge_attr", 0, 12.3).unwrap();
    g.set_edge_attr_numeric("edge_attr", 4, f64::NEG_INFINITY)
        .unwrap();
    for v in 0..4 {
        g.set_vertex_attr_str("vertex_attr", v, &format!("vertex_name{v}"))
            .unwrap();
    }
    let file = TempFile::new("attrs", "ncol");
    g.write_graph_ncol(&file.0, Some("vertex_attr"), Some("edge_attr"))
        .unwrap();
    let text = std::fs::read_to_string(&file.0).unwrap();
    assert!(
        text.starts_with("vertex_name0 vertex_name1 12.3\nvertex_name0 vertex_name1 NaN\n"),
        "{text}"
    );
    assert!(text.contains(" -Inf\n"), "{text}");

    let h = Graph::read_graph_ncol(
        &file.0,
        &[],
        &NcolLglOptions::default()
            .with_directed(true)
            .with_weights(AddWeights::Yes),
    )
    .unwrap();
    assert_eq!(h, g);
    assert_eq!(
        h.vertex_attr_str_values("name", ..).unwrap(),
        [
            "vertex_name0",
            "vertex_name1",
            "vertex_name2",
            "vertex_name3"
        ]
    );
    let nan = f64::NAN;
    assert_same_floats(
        &h.edge_attr_numeric_values("weight", ..).unwrap(),
        &[12.3, nan, nan, nan, f64::NEG_INFINITY, nan],
    );
}

#[test]
fn ncol_and_lgl_weight_options() {
    attrs();
    let text = "a b\nb c 3\n";
    let opts = NcolLglOptions::default();
    // IfPresent: stored because one weight is explicit, 1 for the others.
    let g = Graph::read_graph_ncol_from_str(text, &[], &opts).unwrap();
    assert_eq!(
        g.edge_attr_numeric_values("weight", ..).unwrap(),
        [1.0, 3.0]
    );
    // ...and not stored at all when no weight is given.
    let g = Graph::read_graph_ncol_from_str("a b\nb c\n", &[], &opts).unwrap();
    assert!(!g.has_attribute(AttributeKind::Edge, "weight"));
    assert!(g.has_attribute(AttributeKind::Vertex, "name"));
    // Yes: always stored; No: never; names = false: no `name`.
    let g = Graph::read_graph_ncol_from_str(
        "a b\nb c\n",
        &[],
        &opts.with_weights(AddWeights::Yes).with_names(false),
    )
    .unwrap();
    assert_eq!(
        g.edge_attr_numeric_values("weight", ..).unwrap(),
        [1.0, 1.0]
    );
    assert!(!g.has_attribute(AttributeKind::Vertex, "name"));
    let g = Graph::read_graph_ncol_from_str(text, &[], &opts.with_weights(AddWeights::No)).unwrap();
    assert!(!g.has_attribute(AttributeKind::Edge, "weight"));

    // Predefined names become the `name` attribute of the first vertices.
    let g = Graph::read_graph_ncol_from_str(text, &["c", "b", "a"], &opts).unwrap();
    assert_eq!(
        g.vertex_attr_str_values("name", ..).unwrap(),
        ["c", "b", "a"]
    );

    // LGL: the same attributes, with names written back by the writer.
    let lgl = "# hub\nx 1.5\ny -2\nz\n";
    let g = Graph::read_graph_lgl_from_str(lgl, &opts).unwrap();
    assert_eq!(
        g.vertex_attr_str_values("name", ..).unwrap(),
        ["hub", "x", "y", "z"]
    );
    assert_eq!(
        g.edge_attr_numeric_values("weight", ..).unwrap(),
        [1.5, -2.0, 1.0]
    );
    assert_eq!(
        g.write_graph_lgl_to_string(Some("name"), Some("weight"), false)
            .unwrap(),
        "# hub\nx 1.5\ny -2\nz 1\n"
    );
}

/// tests/unit/pajek_signed.net: a signed directed network in matrix form.
const PAJEK_SIGNED: &str = "*NETWORK First.net; 14.04.2009 / 09:46:56\n\
*Vertices 10\n\
1 \"S65\"\n2 \"S29\"\n3 \"S04\"\n4 \"S75\"\n5 \"S24\"\n\
6 \"S81\"\n7 \"S51\"\n8 \"S78\"\n9 \"S86\"\n10 \"S39\"\n\
*Matrix\n\
 0 0 0 0 0 1 0 0 0 -1\n\
 0 0 1 1 0 1 1 0 1 0\n\
 -1 0 0 1 0 0 1 0 1 0\n\
 -1 1 0 0 1 1 1 0 1 0\n\
 0 1 0 1 0 0 0 0 -1 -1\n\
 1 -1 0 0 0 0 0 1 0 0\n\
 0 -1 1 1 0 -1 0 0 1 0\n\
 0 0 1 1 0 1 -1 0 1 1\n\
 0 0 0 0 0 0 0 0 0 -1\n\
 1 1 1 1 1 1 1 1 1 0\n";

#[test]
fn pajek_signed_matrix_like_igraph_unit_test() {
    // pajek_signed.out: one edge per non-zero entry, in row-major order,
    // weighted by the entry; labels become names.
    attrs();
    let g = Graph::read_graph_pajek_from_str(PAJEK_SIGNED).unwrap();
    assert!(g.is_directed());
    assert_eq!(
        g.vertex_attr_str_values("name", ..).unwrap(),
        [
            "S65", "S29", "S04", "S75", "S24", "S81", "S51", "S78", "S86", "S39"
        ]
    );
    let matrix: Vec<Vec<f64>> = PAJEK_SIGNED
        .lines()
        .skip_while(|l| *l != "*Matrix")
        .skip(1)
        .map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect())
        .collect();
    let (mut edges, mut weights) = (vec![], vec![]);
    for (i, row) in matrix.iter().enumerate() {
        for (j, &w) in row.iter().enumerate() {
            if w != 0.0 {
                edges.push((i as i64, j as i64));
                weights.push(w);
            }
        }
    }
    assert_eq!(edges.len(), 45);
    assert_eq!(g.edge_list(), edges);
    assert_eq!(g.edge_attr_numeric_values("weight", ..).unwrap(), weights);
    // 10 negative ties (as in the .out file).
    assert_eq!(weights.iter().filter(|&&w| w < 0.0).count(), 10);
}

#[test]
fn pajek_bipartite_writer_like_igraph_unit_test() {
    // tests/unit/pajek_bipartite.c and .out: a 10-ring with alternating
    // types; Pajek wants the first mode first, so the vertices are reordered.
    attrs();
    let mut ring = Graph::ring(10, false, false, true).unwrap();
    let types: Vec<bool> = (0..10).map(|i| i % 2 == 1).collect();
    ring.set_vertex_attr_bool_values("type", &types).unwrap();
    let text = ring.write_graph_pajek_to_string().unwrap();
    assert_eq!(
        text,
        "*Vertices 10 5\n1 \"1\"\n2 \"3\"\n3 \"5\"\n4 \"7\"\n5 \"9\"\n\
         6 \"2\"\n7 \"4\"\n8 \"6\"\n9 \"8\"\n10 \"10\"\n\
         *Edges\n1 6\n6 2\n2 7\n7 3\n3 8\n8 4\n4 9\n9 5\n5 10\n1 10\n"
    );

    // Reading it back gives the `type` attribute of the bipartite module,
    // and the original ids as names: an isomorphic, relabeled ring.
    let back = Graph::read_graph_pajek_from_str(&text).unwrap();
    let back_types = back.vertex_attr_bool_values("type", ..).unwrap();
    assert_eq!(back_types, [[false; 5], [true; 5]].concat());
    assert!(back.is_bipartite().unwrap());
    for (a, b) in back.edge_list() {
        assert_ne!(back_types[a as usize], back_types[b as usize]);
    }
    assert!(back.isomorphic(&ring).unwrap());
    assert_eq!(
        back.vertex_attr_str_values("name", ..).unwrap(),
        ["1", "3", "5", "7", "9", "2", "4", "6", "8", "10"]
    );
}

#[test]
fn pajek_vertex_parameters_become_named_attributes() {
    attrs();
    let text = "*Vertices 3\n1 \"A\" 0.1 0.2 0.5 ic Red bc Blue\n2 \"B\" 0.3 0.4 0.0\n3 \"C\"\n\
                *Arcs\n1 2 0.5 c Green\n2 3\n";
    let g = Graph::read_graph_pajek_from_str(text).unwrap();
    assert_eq!(
        g.vertex_attr_str_values("name", ..).unwrap(),
        ["A", "B", "C"]
    );
    assert_same_floats(
        &g.vertex_attr_numeric_values("x", ..).unwrap(),
        &[0.1, 0.3, f64::NAN],
    );
    assert_same_floats(
        &g.vertex_attr_numeric_values("z", ..).unwrap(),
        &[0.5, 0.0, f64::NAN],
    );
    // Parameters missing on some vertices take Pajek's defaults.
    assert_eq!(
        g.vertex_attr_str_values("color", ..).unwrap(),
        ["Red", "LightOrange", "LightOrange"]
    );
    assert_eq!(
        g.vertex_attr_str_values("framecolor", ..).unwrap(),
        ["Blue", "Brown", "Brown"]
    );
    assert_same_floats(
        &g.edge_attr_numeric_values("weight", ..).unwrap(),
        &[0.5, f64::NAN],
    );
    assert_eq!(
        g.edge_attr_str_values("color", ..).unwrap(),
        ["Green", "MidnightBlue"]
    );
}

/// tests/unit/graphml-default-attrs.xml.
const GRAPHML_DEFAULTS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <key id="type" for="node" attr.name="type" attr.type="boolean">
    <default>TRUE</default>
  </key>
  <key id="gender" for="node" attr.name="gender" attr.type="string">
    <default>male</default>
  </key>
  <key id="age" for="node" attr.name="age" attr.type="int">
    <default>20</default>
  </key>
  <key id="retired" for="node" attr.name="retired" attr.type="boolean">
    <default>FALSE</default>
  </key>
  <graph id="G" edgedefault="directed">
    <node id='p1'><data key='type'>FALSE</data><data key='age'>30</data></node>
    <node id='o1'><data key='gender'>female</data></node>
    <node id='o2'></node>
    <edge id='e1' source='p1' target='o1'></edge>
    <edge id='e2' source='p1' target='o2'></edge>
  </graph>
</graphml>
"#;

#[test]
fn graphml_default_attributes_like_igraph_unit_test() {
    // igraph_read_graph_graphml.out, "Graph with default attributes".
    attrs();
    let g = Graph::read_graph_graphml_from_str(GRAPHML_DEFAULTS, 0).unwrap();
    assert!(g.is_directed());
    assert_eq!(g.edge_list(), vec![(0, 1), (0, 2)]);
    assert_eq!(
        g.vertex_attr_bool_values("type", ..).unwrap(),
        [false, true, true]
    );
    assert_eq!(
        g.vertex_attr_str_values("gender", ..).unwrap(),
        ["male", "female", "male"]
    );
    assert_eq!(
        g.vertex_attr_numeric_values("age", ..).unwrap(),
        [30.0, 20.0, 20.0]
    );
    assert_eq!(
        g.vertex_attr_bool_values("retired", ..).unwrap(),
        [false; 3]
    );
    assert_eq!(
        g.vertex_attr_str_values("id", ..).unwrap(),
        ["p1", "o1", "o2"]
    );
    // igraph 1.0.0 and 1.0.1 bug (src/io/graphml.c): the edge `id`
    // attribute is filled with the *node* ids, not with "e1", "e2".
    assert_eq!(g.edge_attr_str_values("id", ..).unwrap(), ["p1", "o1"]);
    assert_eq!(
        g.attribute_type(AttributeKind::Vertex, "age").unwrap(),
        Some(AttributeType::Numeric)
    );

    // The writer exports every attribute with its type; reading the
    // document back restores all of them.
    let xml = g.write_graph_graphml_to_string(false).unwrap();
    assert!(
        xml.contains(r#"attr.name="type" attr.type="boolean""#),
        "{xml}"
    );
    assert!(
        xml.contains(r#"attr.name="age" attr.type="double""#),
        "{xml}"
    );
    let h = Graph::read_graph_graphml_from_str(&xml, 0).unwrap();
    assert_eq!(h, g);
    for name in ["type", "gender", "age", "retired"] {
        assert_eq!(
            h.vertex_attr_values(name, ..).unwrap(),
            g.vertex_attr_values(name, ..).unwrap(),
            "{name}"
        );
    }
}

#[test]
fn graphml_missing_values_use_key_defaults_or_nan() {
    attrs();
    let g = Graph::read_graph_graphml_from_str(TEST_GRAPHML, 0).unwrap();
    // `color` has a default ("yellow"), `weight` has none (NaN).
    assert_eq!(
        g.vertex_attr_str_values("color", ..).unwrap(),
        [
            "green",
            "yellow",
            "blue",
            "red \"with entities\"",
            "yellow",
            "turquoise"
        ]
    );
    let nan = f64::NAN;
    assert_same_floats(
        &g.edge_attr_numeric_values("weight", ..).unwrap(),
        &[1.0, 1.0, 2.0, nan, nan, nan, 1.1],
    );
}

#[test]
fn gml_attributes_match_igraph_unit_test_output() {
    // tests/unit/gml.c writes graph1.gml back with creator "igraph":
    // gml.out lists the expected document.
    attrs();
    take_warnings();
    let g = Graph::read_graph_gml_from_str(GRAPH1_GML).unwrap();
    assert!(
        take_warnings()
            .iter()
            .any(|w| w.contains("Composite vertex attribute 'graphics' ignored"))
    );
    // The GML node ids are kept as a numeric attribute...
    assert_eq!(
        g.vertex_attr_numeric_values("id", ..).unwrap(),
        [1.0, 2.0, 5.0]
    );
    // ...entities are decoded, the last of repeated keys wins.
    assert_eq!(
        g.edge_attr_str_values("label", ..).unwrap(),
        ["-0.45", "Tom & Jerry's \"friendship\""]
    );
    assert_eq!(g.graph_attr_str("AttOne").unwrap(), "x");
    // `AttTwo 3.14` in the file (a literal value, not an approximation of pi).
    assert_eq!(
        g.graph_attr_numeric("AttTwo").unwrap(),
        "3.14".parse::<f64>().unwrap()
    );

    let expected = r#"Creator "igraph"
Version 1
graph
[
  directed 1
  AttOne "x"
  AttTwo 3.14
  node
  [
    id 1
    b "2"
    c 5.6
    nan ""
  ]
  node
  [
    id 2
    a 2
    b "asd"
    d -Inf
    nan "foo"
  ]
  node
  [
    id 5
    b ""
    nan ""
  ]
  edge
  [
    source 5
    target 1
    weight -0.45
    label "-0.45"
  ]
  edge
  [
    source 2
    target 5
    label "Tom &amp; Jerry's &quot;friendship&quot;"
    num1 Inf
  ]
]
"#;
    let out = g
        .write_graph_gml_to_string(&GmlWriteOptions::default().with_creator("igraph"))
        .unwrap();
    assert_eq!(out, expected);
    // `sou_rce` would become `source`: skipped with a warning.
    assert!(
        take_warnings()
            .iter()
            .any(|w| w.contains("'sou_rce' was ignored"))
    );

    // Explicit ids take precedence over the `id` attribute.
    let out = g
        .write_graph_gml_to_string(
            &GmlWriteOptions::default()
                .with_creator("")
                .with_ids(&[10.0, 20.0, 50.0]),
        )
        .unwrap();
    assert!(out.contains("source 50\n    target 10\n"), "{out}");
}

#[test]
fn gml_round_trip_keeps_the_original_ids_through_the_id_attribute() {
    attrs();
    let text = "graph [ directed 0 node [ id 100 ] node [ id 7 ] edge [ source 7 target 100 ] ]";
    let g = Graph::read_graph_gml_from_str(text).unwrap();
    let out = g
        .write_graph_gml_to_string(&GmlWriteOptions::default().with_creator(""))
        .unwrap();
    assert!(out.contains("id 100\n") && out.contains("id 7\n"), "{out}");
    let h = Graph::read_graph_gml_from_str(&out).unwrap();
    assert_eq!(h, g);
    assert_eq!(
        h.vertex_attr_numeric_values("id", ..).unwrap(),
        [100.0, 7.0]
    );
}

#[test]
fn dot_with_attributes_matches_igraph_unit_test() {
    // tests/unit/igraph_write_graph_dot.c and .out.
    attrs();
    let mut g = Graph::from_edges(
        &[
            (0, 1),
            (0, 2),
            (1, 1),
            (1, 3),
            (2, 0),
            (2, 3),
            (4, 3),
            (4, 3),
        ],
        6,
        false,
    )
    .unwrap();
    for (v, x) in [(0, -1.0), (1, 2.1), (2, 1.23e-6), (3, 1e7)] {
        g.set_vertex_attr_numeric("VAN", v, x).unwrap();
    }
    g.set_vertex_attr_str("VAS", 0, "foo").unwrap();
    g.set_vertex_attr_str("VAS", 1, "bar").unwrap();
    g.set_vertex_attr_bool("VAB", 0, true).unwrap();
    g.set_vertex_attr_bool("VAB", 1, false).unwrap();
    g.set_edge_attr_numeric("EAN", 0, -100.1).unwrap();
    g.set_edge_attr_numeric("EAN", 2, 100.0).unwrap();
    g.set_edge_attr_str("EAS", 0, "Blue").unwrap();
    g.set_edge_attr_str("EAS", 2, "RED").unwrap();
    g.set_edge_attr_bool("EAB", 0, true).unwrap();
    g.set_edge_attr_bool("EAB", 2, false).unwrap();

    let vertex = |van: &str, vas: &str, vab: u8| {
        format!(" [\n    VAN={van}\n    VAS={vas}\n    VAB={vab}\n  ];\n")
    };
    let edge = |ean: &str, eas: &str, eab: u8| {
        format!(" [\n    EAN={ean}\n    EAS={eas}\n    EAB={eab}\n  ];\n")
    };
    let mut expected = format!(
        "/* Created by igraph {} */\ngraph {{\n",
        igraph::misc::version()
    );
    for (v, (van, vas, vab)) in [
        ("-1", "foo", 1),
        ("2.1", "bar", 0),
        ("\"1.23e-06\"", "\"\"", 0),
        ("10000000", "\"\"", 0),
        ("NaN", "\"\"", 0),
        ("NaN", "\"\"", 0),
    ]
    .into_iter()
    .enumerate()
    {
        expected += &format!("  {v}{}", vertex(van, vas, vab));
    }
    expected += "\n";
    for (e, (ean, eas, eab)) in [
        ("1 -- 0", ("-100.1", "Blue", 1)),
        ("2 -- 0", ("NaN", "\"\"", 0)),
        ("1 -- 1", ("100", "RED", 0)),
        ("3 -- 1", ("NaN", "\"\"", 0)),
        ("2 -- 0", ("NaN", "\"\"", 0)),
        ("3 -- 2", ("NaN", "\"\"", 0)),
        ("4 -- 3", ("NaN", "\"\"", 0)),
        ("4 -- 3", ("NaN", "\"\"", 0)),
    ] {
        expected += &format!("  {e}{}", edge(ean, eas, eab));
    }
    expected += "}\n";
    assert_eq!(g.write_graph_dot_to_string().unwrap(), expected);
}

#[test]
fn leda_with_attributes_matches_igraph_unit_test() {
    // tests/unit/igraph_write_graph_leda.out, third to sixth blocks.
    attrs();
    let mut directed = Graph::ring(5, true, false, true).unwrap();
    directed
        .set_vertex_attr_numeric_values("name", &[5.0, 6.0, 7.0, 8.0, 9.0])
        .unwrap();
    let edges_void = "# Edges\n5\n1 2 0 |{}|\n2 3 0 |{}|\n3 4 0 |{}|\n4 5 0 |{}|\n5 1 0 |{}|\n";
    assert_eq!(
        directed
            .write_graph_leda_to_string(Some("name"), None)
            .unwrap(),
        format!(
            "LEDA.GRAPH\ndouble\nvoid\n-1\n# Vertices\n5\n|{{5}}|\n|{{6}}|\n|{{7}}|\n|{{8}}|\n|{{9}}|\n{edges_void}"
        )
    );
    directed.remove_vertex_attr("name");
    directed
        .set_vertex_attr_str_values("name", &["foo", "bar", "baz", "spam", "eggs"])
        .unwrap();
    assert!(
        directed
            .write_graph_leda_to_string(Some("name"), None)
            .unwrap()
            .starts_with("LEDA.GRAPH\nstring\nvoid\n-1\n# Vertices\n5\n|{foo}|\n|{bar}|\n")
    );

    let mut undirected = Graph::ring(5, false, false, true).unwrap();
    undirected
        .set_edge_attr_numeric_values("weight", &[5.0, 6.0, 7.0, 8.0, 9.0])
        .unwrap();
    assert_eq!(
        undirected
            .write_graph_leda_to_string(None, Some("weight"))
            .unwrap(),
        "LEDA.GRAPH\nvoid\ndouble\n-2\n# Vertices\n5\n|{}|\n|{}|\n|{}|\n|{}|\n|{}|\n\
         # Edges\n5\n1 2 0 |{5}|\n1 5 0 |{9}|\n2 3 0 |{6}|\n3 4 0 |{7}|\n4 5 0 |{8}|\n"
    );
}

#[test]
fn dl_labels_and_values_become_attributes() {
    attrs();
    let text = "DL n=5\nformat = edgelist1\nlabels:\ngeorge, sally, jim, billy, jane\n\
                data:\n1 2 0.5\n1 3\n2 3\n3 1\n4 3\n";
    let file = TempFile::new("attrs", "dl");
    std::fs::write(&file.0, text).unwrap();
    let g = Graph::read_graph_dl(&file.0, true).unwrap();
    assert_eq!(
        Graph::read_graph(&file.0, GraphFormat::Dl, true).unwrap(),
        g
    );
    assert_eq!(
        g.vertex_attr_str_values("name", ..).unwrap(),
        ["george", "sally", "jim", "billy", "jane"]
    );
    let nan = f64::NAN;
    assert_same_floats(
        &g.edge_attr_numeric_values("weight", ..).unwrap(),
        &[0.5, nan, nan, nan, nan],
    );
}

/// Story: a delivery company keeps its road map as a weighted NCOL file with
/// city names. We load it with its attributes, compute the shortest delivery
/// distances with the weights, and hand the map to the visualization team as
/// GraphML: names and distances must survive, and the GraphML copy must give
/// the very same distances.
#[test]
fn use_case_named_weighted_road_map() {
    attrs();
    let roads = "depot a 4\ndepot b 1\nb a 2\na c 1\nb c 5\nc harbour 3\n";
    let g = Graph::read_graph_ncol_from_str(roads, &[], &NcolLglOptions::default()).unwrap();
    let names = g.vertex_attr_str_values("name", ..).unwrap();
    assert_eq!(names, ["depot", "a", "b", "c", "harbour"]);
    let weights = g.edge_attr_numeric_values("weight", ..).unwrap();
    assert_eq!(weights, [4.0, 1.0, 2.0, 1.0, 5.0, 3.0]);

    let id = |name: &str| names.iter().position(|n| n == name).unwrap() as i64;
    let d = g
        .distances_dijkstra(id("depot"), .., Some(&weights), NeighborMode::All)
        .unwrap();
    // depot -> b -> a -> c -> harbour = 1 + 2 + 1 + 3.
    assert_eq!(d.row(0), vec![0.0, 3.0, 1.0, 4.0, 7.0]);

    let file = TempFile::new("roads", "graphml");
    g.write_graph(&file.0, GraphFormat::from_path(&file.0).unwrap())
        .unwrap();
    let copy = Graph::read_graph_graphml(&file.0, 0).unwrap();
    assert_eq!(copy, g);
    assert_eq!(copy.vertex_attr_str_values("name", ..).unwrap(), names);
    let copy_weights = copy.edge_attr_numeric_values("weight", ..).unwrap();
    assert_eq!(copy_weights, weights);
    let d2 = copy
        .distances_dijkstra(id("depot"), .., Some(&copy_weights), NeighborMode::All)
        .unwrap();
    assert_eq!(d2.row(0), d.row(0));
}

#[test]
fn non_utf8_labels_are_kept_by_files_and_replaced_in_strings() {
    // A Latin-1 Pajek file ("Jos\xe9"): igraph keeps the raw bytes.
    attrs();
    let input = TempFile::new("latin1", "net");
    std::fs::write(
        &input.0,
        b"*Vertices 2\n1 \"Jos\xe9\"\n2 \"Ana\"\n*Edges\n1 2\n",
    )
    .unwrap();
    let g = Graph::read_graph_pajek(&input.0).unwrap();
    // The in-memory writer must return a String: the invalid byte becomes U+FFFD.
    let text = g.write_graph_pajek_to_string().unwrap();
    assert!(text.contains("\"Jos\u{FFFD}\""), "{text}");
    // The file based writer reproduces the original bytes exactly.
    let output = TempFile::new("latin1-out", "net");
    g.write_graph_pajek(&output.0).unwrap();
    let bytes = std::fs::read(&output.0).unwrap();
    assert!(bytes.windows(5).any(|w| w == b"Jos\xe9\""), "{bytes:?}");
}

/// With the attribute handler on, the GraphML reader keeps the node ids in a
/// string `id` vertex attribute; edge ids also create an `id` edge attribute,
/// which igraph 1.0.x fills with node ids (bug in `src/io/graphml.c`, see
/// the module docs), so it must not match the edge ids of the document.
#[test]
fn graphml_node_ids_and_buggy_edge_ids() {
    attrs();
    let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <graph edgedefault="directed">
    <node id="a"/><node id="b"/><node id="c"/>
    <edge id="e1" source="a" target="b"/>
    <edge id="e2" source="b" target="c"/>
  </graph>
</graphml>"#;
    let mut g = Graph::read_graph_graphml_from_str(xml, 0).unwrap();
    assert_eq!(g.edge_list(), vec![(0, 1), (1, 2)]);
    assert_eq!(g.vertex_attr_str_values("id", ..).unwrap(), ["a", "b", "c"]);
    let edge_ids = g.edge_attr_str_values("id", ..).unwrap();
    assert_eq!(edge_ids.len(), 2);
    // The first node ids instead of "e1", "e2" (igraph 1.0.0 and 1.0.1).
    assert_eq!(edge_ids, ["a", "b"]);
    // The workaround suggested by the docs.
    assert!(g.remove_edge_attr("id"));
    // The node ids are written back as an ordinary vertex key.
    let out = g.write_graph_graphml_to_string(false).unwrap();
    assert!(out.contains(r#"<key id="id" for="node" attr.name="id" attr.type="string"/>"#));
    assert!(!out.contains(r#"for="edge" attr.name="id""#));

    // More edges than nodes: the buggy edge `id`s are padded with "".
    let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <graph edgedefault="undirected">
    <node id="x"/><node id="y"/>
    <edge id="e1" source="x" target="y"/>
    <edge id="e2" source="y" target="x"/>
    <edge id="e3" source="x" target="x"/>
  </graph>
</graphml>"#;
    let g = Graph::read_graph_graphml_from_str(xml, 0).unwrap();
    assert_eq!(g.edge_attr_str_values("id", ..).unwrap(), ["x", "y", ""]);
}

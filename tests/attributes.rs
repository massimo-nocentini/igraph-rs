//! Integration tests for the `attributes` module (igraph's C attribute handler).

mod common;

use common::*;
use igraph::attributes::{
    self, AttributeCombination, AttributeCombinationType as Comb, AttributeKind, AttributeRecord,
    AttributeType, AttributeValue, AttributeValues, CombineFunction,
};
use igraph::{ffi, prelude::*};

fn setup() {
    attributes::enable().unwrap();
}

/// Simplifies `g` in place (multi-edges and loops), combining edge attributes.
fn simplify(g: &mut Graph, comb: &AttributeCombination) {
    g.simplify_with_attributes(true, true, comb).unwrap();
}

// ---------------------------------------------------------------------------
// Enabling
// ---------------------------------------------------------------------------

#[test]
fn enable_is_idempotent() {
    setup();
    setup();
    assert!(attributes::is_enabled());
    assert!(attributes::has_attribute_table());
}

#[test]
fn setters_enable_attributes_implicitly() {
    let mut g = Graph::new(2, false);
    g.set_vertex_attr_numeric("x", 0, 1.0).unwrap();
    assert!(attributes::is_enabled());
    assert_eq!(g.vertex_attr_numeric("x", 0).unwrap(), 1.0);
}

/// Runs in a child process (see `graphs_created_before_enable_keep_working`):
/// the attribute handler is global, so a pristine process is needed to
/// create graphs *before* enabling it.
#[test]
fn child_pre_enable_graph() {
    if std::env::var_os("IGRAPH_RS_ATTR_CHILD").is_none() {
        return;
    }
    assert!(!attributes::is_enabled());
    let mut old = Graph::famous("Zachary").unwrap();
    let reader_view = old.clone();
    assert!(!old.has_attribute(AttributeKind::Vertex, "name"));
    assert!(old.attribute_list().unwrap().is_empty());
    assert_eq!(
        old.vertex_attr_numeric_values("x", ..).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );

    attributes::enable().unwrap();

    // Structural operations on the attribute-less graph must not crash.
    old.delete_vertices(vec![0, 33]).unwrap();
    old.add_vertices(2).unwrap();
    old.add_edge(0, 32).unwrap();
    old.delete_edges(0).unwrap();
    let copy = old.clone();
    assert!(copy.attribute_list().unwrap().is_empty());
    let mut sub = old
        .induced_subgraph(0..10, SubgraphImplementation::Auto)
        .unwrap();
    // The new subgraph was created after enabling: it has (empty) storage.
    sub.set_vertex_attr_numeric("x", 3, 1.0).unwrap();
    simplify(&mut old, &AttributeCombination::all(Comb::Sum).unwrap());
    old.contract_vertices_with_attributes(
        &(0..old.vcount() as i64).map(|v| v / 2).collect::<Vec<_>>(),
        &AttributeCombination::all(Comb::First).unwrap(),
    )
    .unwrap();
    old.to_undirected_with_attributes(ToUndirected::Collapse, &AttributeCombination::new())
        .unwrap();
    assert!(
        old.write_graph_graphml_to_string(false)
            .unwrap()
            .contains("<graphml")
    );
    assert!(old.attribute_list().unwrap().is_empty());

    // The first setter lazily creates storage for the old graph.
    old.set_vertex_attr_numeric("x", 5, 5.0).unwrap();
    assert_eq!(old.vertex_attr_numeric("x", 5).unwrap(), 5.0);
    assert!(old.vertex_attr_numeric("x", 4).unwrap().is_nan());
    old.delete_vertices(4).unwrap();
    assert_eq!(old.vertex_attr_numeric("x", 4).unwrap(), 5.0);
    drop(reader_view);
    println!("CHILD-OK");
}

#[test]
fn graphs_created_before_enable_keep_working() {
    let exe = std::env::current_exe().unwrap();
    let out = std::process::Command::new(exe)
        .args([
            "--exact",
            "child_pre_enable_graph",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("IGRAPH_RS_ATTR_CHILD", "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "child failed: {stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("CHILD-OK"), "{stdout}");
}

// ---------------------------------------------------------------------------
// Round trips
// ---------------------------------------------------------------------------

#[test]
fn graph_attributes_round_trip() {
    setup();
    let mut g = cycle(5);
    g.set_graph_attr_numeric("girth", 5.0).unwrap();
    g.set_graph_attr_bool("bipartite", false).unwrap();
    g.set_graph_attr_str("name", "C5").unwrap();
    assert_eq!(g.graph_attr_numeric("girth").unwrap(), 5.0);
    assert!(!g.graph_attr_bool("bipartite").unwrap());
    assert_eq!(g.graph_attr_str("name").unwrap(), "C5");

    // Overwrite.
    g.set_graph_attr_str("name", "pentagon").unwrap();
    g.set_graph_attr("girth", 6).unwrap();
    assert_eq!(
        g.graph_attr("name").unwrap(),
        AttributeValue::String("pentagon".into())
    );
    assert_eq!(g.graph_attr("girth").unwrap(), AttributeValue::Numeric(6.0));
    assert_eq!(
        g.graph_attr("bipartite").unwrap(),
        AttributeValue::Boolean(false)
    );

    // Unicode survives the trip through C strings.
    g.set_graph_attr_str("motto", "ciclo · κύκλος · 環")
        .unwrap();
    assert_eq!(g.graph_attr_str("motto").unwrap(), "ciclo · κύκλος · 環");
}

#[test]
fn vertex_attributes_round_trip() {
    setup();
    let mut g = path(4);
    g.set_vertex_attr_numeric_values("x", &[0.0, 1.5, -2.0, 1e300])
        .unwrap();
    g.set_vertex_attr_bool_values("leaf", &[true, false, false, true])
        .unwrap();
    g.set_vertex_attr_str_values("name", &["a", "b", "c", "d"])
        .unwrap();

    assert_eq!(
        g.vertex_attr_numeric_values("x", ..).unwrap(),
        vec![0.0, 1.5, -2.0, 1e300]
    );
    assert_eq!(
        g.vertex_attr_bool_values("leaf", ..).unwrap(),
        vec![true, false, false, true]
    );
    assert_eq!(
        g.vertex_attr_str_values("name", ..).unwrap(),
        vec!["a", "b", "c", "d"]
    );
    for v in g.vertices() {
        let deg = g.degree_of(v, NeighborMode::All, Loops::Twice).unwrap();
        assert_eq!(g.vertex_attr_bool("leaf", v).unwrap(), deg == 1);
    }
    assert_eq!(g.vertex_attr_numeric("x", 2).unwrap(), -2.0);
    assert_eq!(g.vertex_attr_str("name", 3).unwrap(), "d");

    // Selectors: order and duplicates are respected.
    assert_eq!(
        g.vertex_attr_str_values("name", vec![3, 0, 3]).unwrap(),
        vec!["d", "a", "d"]
    );
    assert_eq!(
        g.vertex_attr_numeric_values("x", 1..3).unwrap(),
        vec![1.5, -2.0]
    );
    let neighbors_of_1 = VertexSelector::Adjacent {
        vertex: 1,
        mode: NeighborMode::All,
    };
    assert_eq!(
        g.vertex_attr_str_values("name", neighbors_of_1).unwrap(),
        vec!["a", "c"]
    );
    assert!(
        g.vertex_attr_bool_values("leaf", VertexSelector::None)
            .unwrap()
            .is_empty()
    );

    // Single-value setters.
    g.set_vertex_attr_str("name", 1, "B").unwrap();
    g.set_vertex_attr_bool("leaf", 1, true).unwrap();
    g.set_vertex_attr("x", 1, 42).unwrap();
    assert_eq!(
        g.vertex_attr_values("name", ..).unwrap(),
        AttributeValues::String(vec!["a".into(), "B".into(), "c".into(), "d".into()])
    );
    assert_eq!(
        g.vertex_attr("leaf", 1).unwrap(),
        AttributeValue::Boolean(true)
    );
    assert_eq!(g.vertex_attr("x", 1).unwrap().as_f64(), Some(42.0));
}

#[test]
fn edge_attributes_round_trip() {
    setup();
    let mut g = complete(4);
    let m = g.ecount();
    let weights: Vec<f64> = (0..m).map(|i| i as f64 * 0.5).collect();
    let heavy: Vec<bool> = weights.iter().map(|&w| w >= 1.0).collect();
    let labels: Vec<String> = g
        .edge_list()
        .iter()
        .map(|(a, b)| format!("{a}-{b}"))
        .collect();
    g.set_edge_attr_numeric_values("weight", &weights).unwrap();
    g.set_edge_attr_bool_values("heavy", &heavy).unwrap();
    g.set_edge_attr_str_values("label", &labels).unwrap();

    assert_eq!(g.edge_attr_numeric_values("weight", ..).unwrap(), weights);
    assert_eq!(g.edge_attr_bool_values("heavy", ..).unwrap(), heavy);
    assert_eq!(g.edge_attr_str_values("label", ..).unwrap(), labels);
    for e in g.edge_ids() {
        let (a, b) = g.edge(e).unwrap();
        assert_eq!(g.edge_attr_str("label", e).unwrap(), format!("{a}-{b}"));
        assert_eq!(
            g.edge_attr_numeric("weight", e).unwrap(),
            weights[e as usize]
        );
        assert_eq!(g.edge_attr_bool("heavy", e).unwrap(), heavy[e as usize]);
    }
    assert_eq!(
        g.edge_attr_str_values("label", vec![5, 0]).unwrap(),
        vec!["2-3", "0-1"]
    );

    g.set_edge_attr("label", 0, "first").unwrap();
    g.set_edge_attr("heavy", 0, true).unwrap();
    g.set_edge_attr("weight", 0, -1.0).unwrap();
    assert_eq!(g.edge_attr("label", 0).unwrap().as_str(), Some("first"));
    assert_eq!(g.edge_attr("heavy", 0).unwrap().as_bool(), Some(true));
    assert_eq!(
        g.edge_attr_values("weight", 0..2).unwrap(),
        AttributeValues::Numeric(vec![-1.0, 0.5])
    );
    g.set_edge_attr_values("weight", vec![1.0; m]).unwrap();
    assert_eq!(
        g.edge_attr_numeric_values("weight", ..)
            .unwrap()
            .iter()
            .sum::<f64>(),
        m as f64
    );
}

#[test]
fn defaults_for_new_elements() {
    setup();
    let mut g = Graph::new(2, false);
    g.set_vertex_attr_numeric("x", 0, 1.0).unwrap();
    g.set_vertex_attr_bool("b", 0, true).unwrap();
    g.set_vertex_attr_str("s", 0, "zero").unwrap();
    g.add_vertices(2).unwrap();
    let xs = g.vertex_attr_numeric_values("x", ..).unwrap();
    assert_eq!(xs[0], 1.0);
    assert!(xs[1..].iter().all(|x| x.is_nan()));
    assert_eq!(
        g.vertex_attr_bool_values("b", ..).unwrap(),
        vec![true, false, false, false]
    );
    assert_eq!(
        g.vertex_attr_str_values("s", ..).unwrap(),
        vec!["zero", "", "", ""]
    );

    g.add_edge(0, 1).unwrap();
    g.set_edge_attr_str("kind", 0, "road").unwrap();
    g.add_edges(&[(1, 2), (2, 3)]).unwrap();
    assert_eq!(
        g.edge_attr_str_values("kind", ..).unwrap(),
        vec!["road", "", ""]
    );
}

// ---------------------------------------------------------------------------
// Attributes follow the structure
// ---------------------------------------------------------------------------

#[test]
fn attributes_survive_clone_and_are_independent() {
    setup();
    let mut g = Graph::famous("Zachary").unwrap();
    let names: Vec<String> = (0..34).map(|i| format!("member{i}")).collect();
    g.set_vertex_attr_str_values("name", &names).unwrap();
    g.set_graph_attr_str("title", "Zachary's karate club")
        .unwrap();
    g.set_edge_attr_numeric_values("weight", &vec![1.0; 78])
        .unwrap();

    let mut h = g.clone();
    assert_eq!(h.vertex_attr_str_values("name", ..).unwrap(), names);
    assert_eq!(h.graph_attr_str("title").unwrap(), "Zachary's karate club");
    assert_eq!(h.attribute_list().unwrap(), g.attribute_list().unwrap());

    // Deep copy: changes to one side are invisible to the other.
    h.set_vertex_attr_str("name", 0, "Mr. Hi").unwrap();
    h.remove_edge_attr("weight");
    assert_eq!(g.vertex_attr_str("name", 0).unwrap(), "member0");
    assert!(g.has_attribute(AttributeKind::Edge, "weight"));
    drop(g);
    assert_eq!(h.vertex_attr_str("name", 0).unwrap(), "Mr. Hi");
}

#[test]
fn vertex_deletion_keeps_attributes_aligned() {
    setup();
    let mut g = Graph::famous("Zachary").unwrap();
    let ids: Vec<f64> = (0..34).map(f64::from).collect();
    g.set_vertex_attr_numeric_values("orig", &ids).unwrap();
    let labels: Vec<String> = g
        .edge_list()
        .iter()
        .map(|(a, b)| format!("{a}-{b}"))
        .collect();
    g.set_edge_attr_str_values("ends", &labels).unwrap();

    // Delete every third vertex.
    let doomed: Vec<i64> = (0..34).filter(|v| v % 3 == 0).collect();
    g.delete_vertices(&doomed).unwrap();
    let orig = g.vertex_attr_numeric_values("orig", ..).unwrap();
    let expected: Vec<f64> = (0..34).filter(|v| v % 3 != 0).map(|v| v as f64).collect();
    assert_eq!(orig, expected);

    // Every surviving edge still carries the label of its original endpoints.
    for e in g.edge_ids() {
        let (a, b) = g.edge(e).unwrap();
        let (oa, ob) = (orig[a as usize] as i64, orig[b as usize] as i64);
        assert_eq!(g.edge_attr_str("ends", e).unwrap(), format!("{oa}-{ob}"));
    }
    assert_eq!(
        g.edge_attr_str_values("ends", ..).unwrap().len(),
        g.ecount()
    );
}

#[test]
fn edge_deletion_keeps_attributes_aligned() {
    setup();
    let mut g = cycle(6);
    g.set_edge_attr_numeric_values("id", &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        .unwrap();
    g.delete_edges(&[1, 4]).unwrap();
    assert_eq!(
        g.edge_attr_numeric_values("id", ..).unwrap(),
        vec![0.0, 2.0, 3.0, 5.0]
    );
}

#[test]
fn subgraph_and_permutation_carry_attributes() {
    setup();
    let mut g = path(5);
    g.set_vertex_attr_str_values("name", &["a", "b", "c", "d", "e"])
        .unwrap();
    g.set_edge_attr_numeric_values("w", &[10.0, 20.0, 30.0, 40.0])
        .unwrap();

    let sub = g
        .induced_subgraph(2..5, SubgraphImplementation::CreateFromScratch)
        .unwrap();
    assert_eq!(
        sub.vertex_attr_str_values("name", ..).unwrap(),
        vec!["c", "d", "e"]
    );
    assert_eq!(
        sub.edge_attr_numeric_values("w", ..).unwrap(),
        vec![30.0, 40.0]
    );

    // Reverse the vertex order.
    let rev = g.permute_vertices(&[4, 3, 2, 1, 0]).unwrap();
    assert_eq!(
        rev.vertex_attr_str_values("name", ..).unwrap(),
        vec!["e", "d", "c", "b", "a"]
    );
    // A non-involutive permutation: every edge keeps its weight and every
    // vertex its name, whatever the new ids are.
    let h = g.permute_vertices(&[2, 0, 3, 4, 1]).unwrap();
    let names = h.vertex_attr_str_values("name", ..).unwrap();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(sorted, vec!["a", "b", "c", "d", "e"]);
    assert_eq!(
        h.edge_attr_numeric_values("w", ..).unwrap(),
        vec![10.0, 20.0, 30.0, 40.0]
    );
    for e in h.edge_ids() {
        let (a, b) = h.edge(e).unwrap();
        let mut ends = [names[a as usize].clone(), names[b as usize].clone()];
        ends.sort();
        let expected = ["ab", "bc", "cd", "de"][e as usize];
        assert_eq!(ends.concat(), expected);
    }
}

// ---------------------------------------------------------------------------
// Listing, querying, removing
// ---------------------------------------------------------------------------

#[test]
fn list_has_type_and_remove() {
    setup();
    let mut g = Graph::from_edges(&[(0, 1)], 2, true).unwrap();
    assert!(g.attribute_list().unwrap().is_empty());
    g.set_graph_attr_bool("directed", true).unwrap();
    g.set_vertex_attr_numeric("x", 0, 0.0).unwrap();
    g.set_vertex_attr_str("name", 0, "u").unwrap();
    g.set_edge_attr_bool("tree", 0, true).unwrap();

    let list = g.attribute_list().unwrap();
    assert_eq!(
        list.graph,
        vec![("directed".to_string(), AttributeType::Boolean)]
    );
    assert_eq!(
        list.vertex,
        vec![
            ("x".to_string(), AttributeType::Numeric),
            ("name".to_string(), AttributeType::String)
        ]
    );
    assert_eq!(
        list.of(AttributeKind::Edge),
        &[("tree".to_string(), AttributeType::Boolean)]
    );
    assert_eq!(
        g.attribute_names(AttributeKind::Vertex).unwrap(),
        vec!["x", "name"]
    );
    assert_eq!(
        g.attribute_type(AttributeKind::Vertex, "name").unwrap(),
        Some(AttributeType::String)
    );
    assert_eq!(
        g.attribute_type(AttributeKind::Graph, "name").unwrap(),
        None
    );

    assert!(g.has_attribute(AttributeKind::Vertex, "x"));
    assert!(!g.has_attribute(AttributeKind::Edge, "x"));
    assert!(!g.has_attribute(AttributeKind::Graph, "no\0such"));

    assert!(g.remove_vertex_attr("x"));
    assert!(!g.remove_vertex_attr("x"));
    assert!(!g.has_attribute(AttributeKind::Vertex, "x"));
    assert!(g.remove_graph_attr("directed"));
    assert!(!g.remove_edge_attr("nope"));

    g.remove_all_attributes(true, true, true);
    assert!(g.attribute_list().unwrap().is_empty());
    // A removed attribute can be recreated with another type.
    g.set_vertex_attr_bool("name", 1, true).unwrap();
    assert_eq!(
        g.attribute_type(AttributeKind::Vertex, "name").unwrap(),
        Some(AttributeType::Boolean)
    );
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[test]
fn errors_on_missing_attributes() {
    setup();
    let mut g = cycle(3);
    g.set_vertex_attr_numeric("x", 0, 1.0).unwrap();
    for err in [
        g.graph_attr_numeric("missing").unwrap_err(),
        g.graph_attr_bool("missing").unwrap_err(),
        g.graph_attr_str("missing").unwrap_err(),
        g.graph_attr("missing").unwrap_err(),
        g.vertex_attr_numeric("missing", 0).unwrap_err(),
        g.vertex_attr_str("missing", 0).unwrap_err(),
        g.vertex_attr("missing", 0).unwrap_err(),
        g.vertex_attr_numeric_values("missing", ..).unwrap_err(),
        g.vertex_attr_bool_values("missing", ..).unwrap_err(),
        g.vertex_attr_values("missing", ..).unwrap_err(),
        g.edge_attr_numeric("missing", 0).unwrap_err(),
        g.edge_attr_bool("missing", 0).unwrap_err(),
        g.edge_attr("missing", 0).unwrap_err(),
        g.edge_attr_str_values("missing", ..).unwrap_err(),
        g.edge_attr_values("missing", ..).unwrap_err(),
    ] {
        assert_eq!(err.kind(), ErrorKind::InvalidValue, "{err}");
        assert!(err.message().contains("missing"), "{err}");
    }
    // The vectorized getters report igraph's own message.
    let err = g.edge_attr_numeric_values("weight", ..).unwrap_err();
    assert!(err.message().contains("does not exist"), "{err}");
}

#[test]
fn errors_on_type_mismatch() {
    setup();
    let mut g = cycle(3);
    g.set_vertex_attr_numeric("x", 0, 1.0).unwrap();
    g.set_graph_attr_str("s", "text").unwrap();
    assert!(
        g.vertex_attr_str("x", 0)
            .unwrap_err()
            .message()
            .contains("numeric")
    );
    assert!(g.vertex_attr_bool("x", 0).is_err());
    assert!(g.vertex_attr_bool_values("x", ..).is_err());
    assert!(g.graph_attr_numeric("s").is_err());
    // Writing a value of another type does not change the attribute's type.
    assert_eq!(
        g.set_vertex_attr_str("x", 1, "one").unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert!(g.set_vertex_attr_bool_values("x", &[true; 3]).is_err());
    assert!(g.set_graph_attr_numeric("s", 1.0).is_err());
    assert_eq!(
        g.attribute_type(AttributeKind::Vertex, "x").unwrap(),
        Some(AttributeType::Numeric)
    );
    assert_eq!(g.vertex_attr_numeric("x", 0).unwrap(), 1.0);
}

#[test]
fn errors_on_invalid_ids_lengths_and_nul_bytes() {
    setup();
    let mut g = cycle(3);
    g.set_vertex_attr_numeric("x", 0, 1.0).unwrap();
    g.set_edge_attr_numeric("w", 0, 1.0).unwrap();

    assert_eq!(
        g.vertex_attr_numeric("x", 3).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        g.vertex_attr_numeric("x", -1).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        g.set_vertex_attr_numeric("x", 7, 0.0).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        g.vertex_attr_numeric_values("x", vec![0, 9])
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        g.edge_attr_numeric("w", 3).unwrap_err().kind(),
        ErrorKind::InvalidEdgeId
    );
    assert_eq!(
        g.set_edge_attr_bool("b", 3, true).unwrap_err().kind(),
        ErrorKind::InvalidEdgeId
    );
    assert_eq!(
        g.edge_attr_numeric_values("w", vec![5]).unwrap_err().kind(),
        ErrorKind::InvalidEdgeId
    );

    assert_eq!(
        g.set_vertex_attr_numeric_values("y", &[1.0])
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert!(g.set_edge_attr_str_values("s", &["a", "b"]).is_err());
    assert!(g.set_vertex_attr_bool_values("b", &[]).is_err());
    // A failed setv does not leave a half-created attribute behind.
    assert!(!g.has_attribute(AttributeKind::Vertex, "y"));

    assert_eq!(
        g.set_graph_attr_str("bad\0name", "v").unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert!(g.set_graph_attr_str("name", "bad\0value").is_err());
    assert!(
        g.set_vertex_attr_str_values("s", &["a", "b\0", "c"])
            .is_err()
    );
    assert!(g.vertex_attr_numeric("x\0", 0).is_err());
}

// ---------------------------------------------------------------------------
// Attribute records
// ---------------------------------------------------------------------------

#[test]
fn attribute_records() {
    setup();
    let mut rec = AttributeRecord::new("flag", AttributeType::Boolean).unwrap();
    assert!(rec.is_empty());
    assert_eq!(rec.name(), Some("flag"));
    rec.check_type(AttributeType::Boolean).unwrap();
    assert_eq!(
        rec.check_type(AttributeType::String).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );

    rec.set_default(true).unwrap();
    assert!(rec.set_default(1.0).is_err());
    rec.resize(3).unwrap();
    assert_eq!(rec.values(), AttributeValues::Boolean(vec![true; 3]));
    rec.set_name("visited").unwrap();
    assert_eq!(rec.name(), Some("visited"));

    // Changing the type drops the values; numeric defaults to NaN.
    rec.set_type(AttributeType::Numeric).unwrap();
    assert!(rec.is_empty());
    rec.resize(2).unwrap();
    assert!(
        rec.values()
            .as_numeric()
            .unwrap()
            .iter()
            .all(|x| x.is_nan())
    );
    // Unsupported types are rejected in Rust: igraph 1.0.1 would abort.
    assert_eq!(
        rec.set_type(AttributeType::Object).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert!(rec.set_type(AttributeType::Unspecified).is_err());
    assert_eq!(rec.attribute_type(), AttributeType::Numeric);
    assert_eq!(rec.len(), 2);
    assert!(AttributeRecord::new("o", AttributeType::Object).is_err());
    assert!(
        AttributeRecord::new("u", AttributeType::Unspecified)
            .unwrap()
            .resize(2)
            .is_err()
    );

    let mut s = AttributeRecord::string("city", &["Firenze", "Siena"]).unwrap();
    s.set_default("unknown").unwrap();
    s.resize(3).unwrap();
    assert_eq!(s.len(), 3);
    assert_eq!(
        s.values().get(2),
        Some(AttributeValue::String("unknown".into()))
    );

    let copy = s.clone();
    assert_eq!(copy, s);
    s.set_name("town").unwrap();
    assert_ne!(copy, s);
    assert_eq!(copy.name(), Some("city"));

    let from = AttributeRecord::from_values("w", vec![1.0, 2.0]).unwrap();
    assert_eq!(from.attribute_type(), AttributeType::Numeric);
    assert_eq!(from.len(), 2);
}

#[test]
fn add_vertices_and_edges_with_attributes() {
    setup();
    let mut g = Graph::new(2, false);
    g.set_vertex_attr_str_values("name", &["a", "b"]).unwrap();
    g.set_vertex_attr_numeric_values("x", &[1.0, 2.0]).unwrap();

    let names = AttributeRecord::string("name", &["c", "d", "e"]).unwrap();
    let color = AttributeRecord::string("color", &["red", "green", "blue"]).unwrap();
    g.add_vertices_with_attributes(3, &[names, color]).unwrap();
    assert_eq!(g.vcount(), 5);
    assert_eq!(
        g.vertex_attr_str_values("name", ..).unwrap(),
        vec!["a", "b", "c", "d", "e"]
    );
    assert_eq!(
        g.vertex_attr_str_values("color", ..).unwrap(),
        vec!["", "", "red", "green", "blue"]
    );
    let x = g.vertex_attr_numeric_values("x", ..).unwrap();
    assert_eq!(&x[..2], &[1.0, 2.0]);
    assert!(x[2..].iter().all(|v| v.is_nan()));

    // Wrong length and wrong type are rejected and leave the graph untouched.
    let short = AttributeRecord::string("name", &["f"]).unwrap();
    assert_eq!(
        g.add_vertices_with_attributes(2, &[short])
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    let wrong = AttributeRecord::numeric("name", &[1.0]).unwrap();
    assert!(g.add_vertices_with_attributes(1, &[wrong]).is_err());
    let untyped = AttributeRecord::new("u", AttributeType::Unspecified).unwrap();
    assert!(g.add_vertices_with_attributes(0, &[untyped]).is_err());
    assert_eq!(g.vcount(), 5);
    assert_eq!(g.vertex_attr_str_values("name", ..).unwrap().len(), 5);

    let w = AttributeRecord::numeric("weight", &[0.5, 1.5]).unwrap();
    let t = AttributeRecord::boolean("tree", &[true, false]).unwrap();
    g.add_edges_with_attributes(&[(0, 2), (3, 4)], &[w, t])
        .unwrap();
    assert_eq!(
        g.edge_attr_numeric_values("weight", ..).unwrap(),
        vec![0.5, 1.5]
    );
    assert_eq!(
        g.edge_attr_bool_values("tree", ..).unwrap(),
        vec![true, false]
    );
    let w = AttributeRecord::numeric("weight", &[9.0]).unwrap();
    assert_eq!(
        g.add_edges_with_attributes(&[(0, 99)], &[w])
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidVertexId
    );
    g.add_edges_with_attributes(&[(1, 2)], &[]).unwrap();
    assert!(g.edge_attr_numeric("weight", 2).unwrap().is_nan());
}

// ---------------------------------------------------------------------------
// Attribute combinations
// ---------------------------------------------------------------------------

/// Mirrors igraph's `examples/simple/cattributes3.c`: edges 0->1 (three
/// times), 1->2, 2->3 with weights 1..=5, simplified with various combinations.
#[test]
fn simplify_with_combinations_matches_igraph_example() {
    setup();
    let mut g = Graph::from_edges(&[(0, 1), (0, 1), (0, 1), (1, 2), (2, 3)], 4, true).unwrap();
    g.set_edge_attr_numeric_values("weight", &[1.0, 2.0, 3.0, 4.0, 5.0])
        .unwrap();
    let cases = [
        (Comb::Sum, 6.0),
        (Comb::Prod, 6.0),
        (Comb::Min, 1.0),
        (Comb::Max, 3.0),
        (Comb::First, 1.0),
        (Comb::Last, 3.0),
        (Comb::Mean, 2.0),
    ];
    for (kind, expected) in cases {
        let mut h = g.clone();
        let comb =
            AttributeCombination::from_pairs(&[(Some("weight"), kind), (None, Comb::Ignore)])
                .unwrap();
        simplify(&mut h, &comb);
        assert_eq!(h.ecount(), 3);
        assert_eq!(
            h.edge_attr_numeric_values("weight", ..).unwrap(),
            vec![expected, 4.0, 5.0],
            "{kind:?}"
        );
    }

    // "" -> MEAN for every attribute (the last case of the C example).
    let mut h = g.clone();
    simplify(&mut h, &AttributeCombination::all(Comb::Mean).unwrap());
    assert_eq!(
        h.edge_attr_numeric_values("weight", ..).unwrap(),
        vec![2.0, 4.0, 5.0]
    );

    // Ignore drops the attribute; an empty combination too.
    let mut h = g.clone();
    simplify(&mut h, &AttributeCombination::new());
    assert!(!h.has_attribute(AttributeKind::Edge, "weight"));

    // Concatenating numbers is an attribute combination error.
    let mut h = g.clone();
    let comb = AttributeCombination::all(Comb::Concat).unwrap();
    let err = h.simplify_with_attributes(true, true, &comb).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::AttributeCombination);
}

unsafe extern "C" fn zero(
    _input: *const ffi::igraph_vector_t,
    output: *mut f64,
) -> ffi::igraph_error_t {
    unsafe { *output = 0.0 };
    ffi::igraph_error_type_t_IGRAPH_SUCCESS
}

unsafe extern "C" fn spread(
    input: *const ffi::igraph_vector_t,
    output: *mut f64,
) -> ffi::igraph_error_t {
    let values: &Vector = unsafe { &*input };
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    unsafe { *output = max - min };
    ffi::igraph_error_type_t_IGRAPH_SUCCESS
}

unsafe extern "C" fn xor(
    input: *const ffi::igraph_vector_bool_t,
    output: *mut bool,
) -> ffi::igraph_error_t {
    let values: &VectorBool = unsafe { &*input };
    unsafe { *output = values.iter().fold(false, |acc, &b| acc ^ b) };
    ffi::igraph_error_type_t_IGRAPH_SUCCESS
}

#[test]
fn simplify_with_combination_functions() {
    setup();
    let mut g = Graph::from_edges(&[(0, 1), (0, 1), (0, 1), (1, 2), (2, 3)], 4, true).unwrap();
    g.set_edge_attr_numeric_values("weight", &[1.0, 2.0, 3.0, 4.0, 5.0])
        .unwrap();
    g.set_edge_attr_bool_values("odd", &[true, false, true, false, true])
        .unwrap();

    let mut comb = AttributeCombination::new();
    unsafe { comb.add_function(Some("weight"), CombineFunction::Numeric(zero)) }.unwrap();
    comb.add(None, Comb::Ignore).unwrap();
    assert_eq!(comb.query(Some("weight")).unwrap(), Comb::Function);
    let mut h = g.clone();
    simplify(&mut h, &comb);
    assert_eq!(
        h.edge_attr_numeric_values("weight", ..).unwrap(),
        vec![0.0; 3]
    );

    let mut comb = AttributeCombination::new();
    unsafe {
        comb.add_function(Some("weight"), CombineFunction::Numeric(spread))
            .unwrap();
        comb.add_function(Some("odd"), CombineFunction::Boolean(xor))
            .unwrap();
    }
    let mut h = g.clone();
    simplify(&mut h, &comb);
    assert_eq!(
        h.edge_attr_numeric_values("weight", ..).unwrap(),
        vec![2.0, 0.0, 0.0]
    );
    // true ^ false ^ true = false for the merged edge.
    assert_eq!(
        h.edge_attr_bool_values("odd", ..).unwrap(),
        vec![false, false, true]
    );
}

#[test]
fn combination_list_management() {
    setup();
    let mut comb = AttributeCombination::default();
    assert_eq!(comb.query(Some("anything")).unwrap(), Comb::Default);
    comb.add(Some("weight"), Comb::Sum).unwrap();
    comb.add(Some("weight"), Comb::Max).unwrap(); // replaces
    comb.add(None, Comb::First).unwrap();
    assert_eq!(comb.query(Some("weight")).unwrap(), Comb::Max);
    assert_eq!(comb.query(Some("name")).unwrap(), Comb::First);
    assert_eq!(comb.query(None).unwrap(), Comb::First);
    comb.remove(None).unwrap();
    comb.remove(Some("not there")).unwrap();
    assert_eq!(comb.query(Some("name")).unwrap(), Comb::Default);
    comb.remove(Some("weight")).unwrap();
    assert_eq!(comb.query(Some("weight")).unwrap(), Comb::Default);
    assert_eq!(
        comb.add(Some("x"), Comb::Function).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert!(comb.add(Some("x\0y"), Comb::Sum).is_err());
}

#[test]
fn contract_vertices_combines_vertex_attributes() {
    setup();
    let mut g = path(4);
    g.set_vertex_attr_str_values("name", &["a", "b", "c", "d"])
        .unwrap();
    g.set_vertex_attr_numeric_values("pop", &[1.0, 2.0, 3.0, 4.0])
        .unwrap();
    g.set_vertex_attr_bool_values("capital", &[false, true, false, false])
        .unwrap();
    let comb = AttributeCombination::from_pairs(&[
        (Some("name"), Comb::First),
        (Some("pop"), Comb::Sum),
        (Some("capital"), Comb::Max),
    ])
    .unwrap();
    g.set_edge_attr_str_values("road", &["ab", "bc", "cd"])
        .unwrap();
    g.set_graph_attr_str("region", "Toscana").unwrap();

    // The plain operator drops every vertex attribute.
    let mut plain = g.clone();
    plain.contract_vertices(&[0, 0, 1, 1]).unwrap();
    assert!(
        plain
            .attribute_names(AttributeKind::Vertex)
            .unwrap()
            .is_empty()
    );
    assert_eq!(plain.edge_attr_str_values("road", ..).unwrap().len(), 3);

    // Merge {a, b} and {c, d}.
    g.contract_vertices_with_attributes(&[0, 0, 1, 1], &comb)
        .unwrap();
    assert_eq!(g.vcount(), 2);
    // Edges (and their attributes) and graph attributes are untouched.
    assert_eq!(g.edge_list(), vec![(0, 0), (0, 1), (1, 1)]);
    assert_eq!(
        g.edge_attr_str_values("road", ..).unwrap(),
        vec!["ab", "bc", "cd"]
    );
    assert_eq!(g.graph_attr_str("region").unwrap(), "Toscana");
    assert_eq!(
        g.vertex_attr_str_values("name", ..).unwrap(),
        vec!["a", "c"]
    );
    assert_eq!(
        g.vertex_attr_numeric_values("pop", ..).unwrap(),
        vec![3.0, 7.0]
    );
    assert_eq!(
        g.vertex_attr_bool_values("capital", ..).unwrap(),
        vec![true, false]
    );

    // Invalid mappings are rejected before reaching igraph (`i64::MAX`
    // would overflow igraph's `max + 1` vertex count).
    for bad in [&[0, 0, 1][..], &[0, -1, 1, 1][..], &[0, 0, 1, i64::MAX][..]] {
        assert_eq!(
            g.clone()
                .contract_vertices_with_attributes(bad, &comb)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidValue
        );
    }
}

/// Bug of igraph 1.0.0 and 1.0.1 (`igraph_i_cattributes_cs_concat` indexes
/// the old values with the position inside the group instead of the merged
/// element id): the `Concat` combination concatenates the strings of the
/// *first* elements of the graph. This test pins the current behaviour;
/// update it once fixed upstream.
#[test]
fn concat_combination_upstream_behaviour() {
    setup();
    let mut g = path(4);
    g.set_vertex_attr_str_values("name", &["a", "b", "c", "d"])
        .unwrap();
    let comb = AttributeCombination::all(Comb::Concat).unwrap();
    g.contract_vertices_with_attributes(&[0, 0, 1, 1], &comb)
        .unwrap();
    // The correct result would be ["ab", "cd"].
    assert_eq!(
        g.vertex_attr_str_values("name", ..).unwrap(),
        vec!["ab", "ab"]
    );
}

// ---------------------------------------------------------------------------
// Foreign formats
// ---------------------------------------------------------------------------

#[test]
fn gml_reader_fills_attributes() {
    setup();
    let g = Graph::read_graph_gml_from_str(
        r#"graph [
             directed 0
             name "trio"
             node [ id 0 label "alpha" ]
             node [ id 1 label "beta" ]
             node [ id 2 label "gamma" ]
             edge [ source 0 target 1 weight 2.5 ]
             edge [ source 1 target 2 weight 0.5 ]
           ]"#,
    )
    .unwrap();
    assert_eq!(g.graph_attr_str("name").unwrap(), "trio");
    assert_eq!(
        g.vertex_attr_str_values("label", ..).unwrap(),
        vec!["alpha", "beta", "gamma"]
    );
    assert_eq!(
        g.edge_attr_numeric_values("weight", ..).unwrap(),
        vec![2.5, 0.5]
    );
    assert_eq!(
        g.attribute_type(AttributeKind::Vertex, "id").unwrap(),
        Some(AttributeType::Numeric)
    );
}

#[test]
fn graphml_writer_emits_attributes() {
    setup();
    let mut g = Graph::from_edges(&[(0, 1)], 2, false).unwrap();
    g.set_graph_attr_str("title", "tiny").unwrap();
    g.set_vertex_attr_str_values("name", &["x", "y"]).unwrap();
    g.set_edge_attr_numeric("weight", 0, 3.0).unwrap();
    g.set_edge_attr_bool("ok", 0, true).unwrap();
    let xml = g.write_graph_graphml_to_string(false).unwrap();
    assert!(
        xml.contains(r#"attr.name="title" attr.type="string""#),
        "{xml}"
    );
    assert!(
        xml.contains(r#"attr.name="weight" attr.type="double""#),
        "{xml}"
    );
    assert!(
        xml.contains(r#"attr.name="ok" attr.type="boolean""#),
        "{xml}"
    );
    assert!(xml.contains(">y</data>"), "{xml}");

    // Round trip through the GraphML reader: structure and attributes survive.
    let h = Graph::read_graph_graphml_from_str(&xml, 0).unwrap();
    assert_eq!(h.edge_list(), vec![(0, 1)]);
    assert_eq!(h.graph_attr_str("title").unwrap(), "tiny");
    assert_eq!(
        h.vertex_attr_str_values("name", ..).unwrap(),
        vec!["x", "y"]
    );
    assert_eq!(h.edge_attr_numeric("weight", 0).unwrap(), 3.0);
    assert!(h.edge_attr_bool("ok", 0).unwrap());
    // The reader also stores the GraphML node ids as a vertex attribute.
    assert_eq!(
        h.vertex_attr_str_values("id", ..).unwrap(),
        vec!["n0", "n1"]
    );
    let list = h.attribute_list().unwrap();
    assert_eq!(list.graph, g.attribute_list().unwrap().graph);
    assert_eq!(list.edge, g.attribute_list().unwrap().edge);
}

// ---------------------------------------------------------------------------
// Threads
// ---------------------------------------------------------------------------

#[test]
fn attributes_work_across_threads() {
    setup();
    let handles: Vec<_> = (0..8)
        .map(|t| {
            std::thread::spawn(move || {
                let mut g = cycle(10 + t);
                let xs: Vec<f64> = (0..10 + t).map(|i| (i * t) as f64).collect();
                g.set_vertex_attr_numeric_values("x", &xs).unwrap();
                g.delete_vertices(0).unwrap();
                (g.vertex_attr_numeric_values("x", ..).unwrap(), xs)
            })
        })
        .collect();
    for h in handles {
        let (got, xs) = h.join().unwrap();
        assert_eq!(got, xs[1..]);
    }
    // A graph with attributes can be moved to another thread.
    let mut g = path(3);
    g.set_vertex_attr_str_values("name", &["p", "q", "r"])
        .unwrap();
    let names = std::thread::spawn(move || g.vertex_attr_str_values("name", ..).unwrap())
        .join()
        .unwrap();
    assert_eq!(names, vec!["p", "q", "r"]);
}

// ---------------------------------------------------------------------------
// Use case
// ---------------------------------------------------------------------------

/// A small story: the karate club members get names and a faction label; the
/// friendships get a weight. The club splits: the members loyal to the
/// instructor (Mr. Hi, vertex 0) leave with him. We keep working with the
/// administrator's faction and check that every attribute still describes the
/// right member and friendship, then summarize the repeated interactions of a
/// logbook with an attribute combination.
#[test]
fn use_case_karate_club_split() {
    setup();
    // Faction of each member after the split (Zachary, 1977).
    const MR_HI: [i64; 17] = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21, 8];
    let mut club = Graph::famous("Zachary").unwrap();
    club.set_graph_attr_str("name", "Zachary's karate club")
        .unwrap();
    club.set_graph_attr_numeric("year", 1977.0).unwrap();
    let names: Vec<String> = (0..34)
        .map(|v| match v {
            0 => "Mr. Hi".to_string(),
            33 => "John A.".to_string(),
            v => format!("member {v}"),
        })
        .collect();
    club.set_vertex_attr_str_values("name", &names).unwrap();
    let faction: Vec<bool> = (0..34).map(|v| MR_HI.contains(&v)).collect();
    club.set_vertex_attr_bool_values("mr_hi", &faction).unwrap();
    // Friendship strength: number of common friends + 1.
    let strength: Vec<f64> = club
        .edge_list()
        .iter()
        .map(|&(a, b)| {
            let na = club.neighbors(a, NeighborMode::All).unwrap();
            let nb = club.neighbors(b, NeighborMode::All).unwrap();
            1.0 + na.iter().filter(|x| nb.contains(x)).count() as f64
        })
        .collect();
    club.set_edge_attr_numeric_values("strength", &strength)
        .unwrap();

    // The split: Mr. Hi's faction leaves.
    let mut admin = club.clone();
    let loyal: Vec<i64> = club
        .vertices()
        .filter(|&v| club.vertex_attr_bool("mr_hi", v).unwrap())
        .collect();
    admin.delete_vertices(&loyal).unwrap();
    assert_eq!(admin.vcount(), 34 - MR_HI.len());
    assert!(
        admin
            .vertex_attr_bool_values("mr_hi", ..)
            .unwrap()
            .iter()
            .all(|&b| !b)
    );
    assert!(
        admin
            .vertex_attr_str_values("name", ..)
            .unwrap()
            .contains(&"John A.".to_string())
    );
    assert_eq!(
        admin.graph_attr_str("name").unwrap(),
        "Zachary's karate club"
    );

    // Every remaining friendship keeps its strength: look the endpoints up by name.
    let original_id = |name: &str| names.iter().position(|n| n == name).unwrap() as i64;
    for e in admin.edge_ids() {
        let (a, b) = admin.edge(e).unwrap();
        let (na, nb) = (
            admin.vertex_attr_str("name", a).unwrap(),
            admin.vertex_attr_str("name", b).unwrap(),
        );
        let orig = club
            .get_eid(original_id(&na), original_id(&nb), false)
            .unwrap()
            .unwrap();
        assert_eq!(
            admin.edge_attr_numeric("strength", e).unwrap(),
            strength[orig as usize]
        );
    }
    // The strongest tie of the administrator's faction.
    let s = admin.edge_attr_numeric_values("strength", ..).unwrap();
    let best = (0..s.len()).max_by(|&i, &j| s[i].total_cmp(&s[j])).unwrap();
    let (a, b) = admin.edge(best as i64).unwrap();
    let pair = [
        admin.vertex_attr_str("name", a).unwrap(),
        admin.vertex_attr_str("name", b).unwrap(),
    ];
    assert!(pair.contains(&"John A.".to_string()), "{pair:?}");

    // A logbook of training sessions between the two leaders' groups, one
    // edge per session, summarized into one weighted edge per pair.
    let mut log = Graph::new(3, false);
    let hours = AttributeRecord::numeric("hours", &[1.0, 2.0, 1.5, 3.0]).unwrap();
    let topic = AttributeRecord::string("topic", &["kata", "sparring", "kata", "rules"]).unwrap();
    log.add_edges_with_attributes(&[(0, 1), (0, 1), (1, 2), (0, 1)], &[hours, topic])
        .unwrap();
    let comb = AttributeCombination::from_pairs(&[
        (Some("hours"), Comb::Sum),
        (Some("topic"), Comb::Last),
    ])
    .unwrap();
    simplify(&mut log, &comb);
    assert_eq!(log.ecount(), 2);
    assert_eq!(
        log.edge_attr_numeric_values("hours", ..).unwrap(),
        vec![6.0, 1.5]
    );
    assert_eq!(
        log.edge_attr_str_values("topic", ..).unwrap(),
        vec!["rules", "kata"]
    );
}

// ---------------------------------------------------------------------------
// Review additions
// ---------------------------------------------------------------------------

#[test]
fn record_clone_keeps_default_and_equality_is_complete() {
    setup();
    let mut s = AttributeRecord::string("city", &["Firenze"]).unwrap();
    s.set_default("unknown").unwrap();
    // igraph_attribute_record_init_copy does not copy the default: Clone must.
    let mut copy = s.clone();
    assert_eq!(copy.default_value(), s.default_value());
    copy.resize(2).unwrap();
    assert_eq!(
        copy.values(),
        AttributeValues::String(vec!["Firenze".into(), "unknown".into()])
    );
    // The original is untouched.
    assert_eq!(s.len(), 1);

    let mut n = AttributeRecord::numeric("x", &[1.0]).unwrap();
    assert!(n.default_value().unwrap().as_f64().unwrap().is_nan());
    n.set_default(0.0).unwrap();
    let mut other = n.clone();
    assert_eq!(n, other);
    other.set_default(1.0).unwrap();
    assert_ne!(n, other, "different defaults");

    // An untyped record differs from an empty numeric one with the same name.
    let untyped = AttributeRecord::new("e", AttributeType::Unspecified).unwrap();
    let empty = AttributeRecord::numeric("e", &[]).unwrap();
    assert_eq!(untyped.values(), empty.values());
    assert_ne!(untyped, empty);
    assert_eq!(untyped.default_value(), None);
    assert_eq!(untyped.clone(), untyped);

    // Absurd lengths are rejected instead of reaching igraph's assertions.
    assert_eq!(
        n.resize(usize::MAX).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(n.len(), 1);
    assert!(format!("{n:?}").contains("default"));
}

#[test]
fn boolean_majority_and_unimplemented_median() {
    setup();
    // Three parallel edges 0->1, and a single 1->2.
    let mut g = Graph::from_edges(&[(0, 1), (0, 1), (0, 1), (1, 2)], 3, true).unwrap();
    g.set_edge_attr_bool_values("up", &[true, false, true, false])
        .unwrap();
    g.set_edge_attr_numeric_values("w", &[3.0, 1.0, 2.0, 7.0])
        .unwrap();

    let mut h = g.clone();
    let comb = AttributeCombination::from_pairs(&[(Some("up"), Comb::Mean), (None, Comb::Ignore)])
        .unwrap();
    simplify(&mut h, &comb);
    // 2 of 3 are true: the majority is true.
    assert_eq!(
        h.edge_attr_bool_values("up", ..).unwrap(),
        vec![true, false]
    );
    assert!(!h.has_attribute(AttributeKind::Edge, "w"));

    // Booleans: Min = all, Max = any.
    for (kind, expected) in [(Comb::Min, false), (Comb::Max, true), (Comb::Prod, false)] {
        let mut h = g.clone();
        simplify(
            &mut h,
            &AttributeCombination::from_pairs(&[(Some("up"), kind)]).unwrap(),
        );
        assert_eq!(h.edge_attr_bool("up", 0).unwrap(), expected, "{kind:?}");
    }

    // The numeric median is not implemented by the C handler.
    let mut h = g.clone();
    let comb = AttributeCombination::from_pairs(&[(Some("w"), Comb::Median)]).unwrap();
    let err = h.simplify_with_attributes(true, true, &comb).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Unimplemented);
    // Strings cannot be summed.
    let mut h = g.clone();
    h.set_edge_attr_str_values("s", &["a", "b", "c", "d"])
        .unwrap();
    let comb = AttributeCombination::from_pairs(&[(Some("s"), Comb::Sum)]).unwrap();
    let err = h.simplify_with_attributes(true, true, &comb).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::AttributeCombination);
}

#[test]
fn value_conversions() {
    assert_eq!(AttributeValue::from(3), AttributeValue::Numeric(3.0));
    assert_eq!(AttributeValue::from(3_i64).as_f64(), Some(3.0));
    assert_eq!(AttributeValue::from(true).as_bool(), Some(true));
    assert_eq!(AttributeValue::from("s").as_str(), Some("s"));
    assert_eq!(AttributeValue::from("s").as_f64(), None);
    assert_eq!(
        AttributeValue::from(String::from("t")).attribute_type(),
        AttributeType::String
    );
    assert_eq!(AttributeValue::from("q").to_string(), "\"q\"");
    assert_eq!(AttributeValue::from(1.5).to_string(), "1.5");

    let v = AttributeValues::from(vec!["a", "b"]);
    assert_eq!(v.len(), 2);
    assert_eq!(v.get(1), Some(AttributeValue::String("b".into())));
    assert_eq!(v.get(2), None);
    assert_eq!(v.as_strings().map(<[String]>::len), Some(2));
    assert!(AttributeValues::from(&[] as &[f64]).is_empty());
    assert_eq!(
        AttributeValues::from(&[true][..]).as_bool(),
        Some(&[true][..])
    );
    assert_eq!(AttributeType::Boolean.to_string(), "boolean");
    assert_eq!(AttributeKind::Edge.to_string(), "edge");
}

// ---------------------------------------------------------------------------
// Attribute-aware operators
// ---------------------------------------------------------------------------

/// Mirrors igraph's `tests/unit/cattributes5.c`: a boolean edge attribute
/// `type` = [1, 1, 0, 0, 1] on the edges 0->1 (three times), 1->2, 2->3,
/// simplified with every combination; the expected values of the merged edge
/// are those of `cattributes5.out` (the other two edges keep false, true).
#[test]
fn boolean_combinations_match_igraph_unit_test() {
    setup();
    let mut g = Graph::from_edges(&[(0, 1), (0, 1), (0, 1), (1, 2), (2, 3)], 4, true).unwrap();
    g.set_edge_attr_bool_values("type", &[true, true, false, false, true])
        .unwrap();

    // The first block of cattributes5.out, byte for byte.
    let mut h = g.clone();
    simplify(
        &mut h,
        &AttributeCombination::from_pairs(&[
            (Some("weight"), Comb::Sum),
            (Some("type"), Comb::First),
            (None, Comb::Ignore),
        ])
        .unwrap(),
    );
    let expected = r#"<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
<!-- Created by igraph -->
  <key id="e_type" for="edge" attr.name="type" attr.type="boolean"/>
  <graph id="G" edgedefault="directed">
    <node id="n0">
    </node>
    <node id="n1">
    </node>
    <node id="n2">
    </node>
    <node id="n3">
    </node>
    <edge source="n0" target="n1">
      <data key="e_type">true</data>
    </edge>
    <edge source="n1" target="n2">
      <data key="e_type">false</data>
    </edge>
    <edge source="n2" target="n3">
      <data key="e_type">true</data>
    </edge>
  </graph>
</graphml>
"#;
    assert_eq!(h.write_graph_graphml_to_string(true).unwrap(), expected);

    let cases = [
        (Comb::Last, false),
        (Comb::Sum, true),
        (Comb::Prod, false),
        (Comb::Min, false),
        (Comb::Max, true),
        (Comb::Mean, true),
        (Comb::Median, true),
    ];
    for (kind, merged) in cases {
        let mut h = g.clone();
        simplify(
            &mut h,
            &AttributeCombination::from_pairs(&[(Some("type"), kind), (None, Comb::Ignore)])
                .unwrap(),
        );
        assert_eq!(
            h.edge_attr_bool_values("type", ..).unwrap(),
            vec![merged, false, true],
            "{kind:?}"
        );
    }
}

#[test]
fn plain_operators_drop_merged_attributes() {
    setup();
    let mut g = Graph::from_edges(&[(0, 1), (1, 0), (0, 1), (1, 2)], 3, true).unwrap();
    g.set_edge_attr_numeric_values("w", &[1.0, 2.0, 4.0, 8.0])
        .unwrap();
    g.set_vertex_attr_str_values("name", &["a", "b", "c"])
        .unwrap();

    let mut h = g.clone();
    h.simplify(true, true).unwrap();
    assert!(!h.has_attribute(AttributeKind::Edge, "w"));
    assert!(h.has_attribute(AttributeKind::Vertex, "name"));

    let mut h = g.clone();
    h.to_undirected(ToUndirected::Collapse).unwrap();
    assert!(!h.has_attribute(AttributeKind::Edge, "w"));

    // `Each` keeps every edge and its attributes, with or without a combination.
    let mut h = g.clone();
    h.to_undirected(ToUndirected::Each).unwrap();
    assert_eq!(
        h.edge_attr_numeric_values("w", ..).unwrap(),
        vec![1.0, 2.0, 4.0, 8.0]
    );
}

#[test]
fn to_undirected_with_attribute_combinations() {
    setup();
    // 0 <-> 1 is mutual (with a duplicate 0 -> 1), 1 -> 2 is not, 2 -> 2 is a loop.
    let mut g = Graph::from_edges(&[(0, 1), (1, 0), (0, 1), (1, 2), (2, 2)], 3, true).unwrap();
    g.set_edge_attr_numeric_values("w", &[1.0, 2.0, 4.0, 8.0, 16.0])
        .unwrap();
    g.set_vertex_attr_str_values("name", &["a", "b", "c"])
        .unwrap();
    let sum = AttributeCombination::all(Comb::Sum).unwrap();

    let mut h = g.clone();
    h.to_undirected_with_attributes(ToUndirected::Collapse, &sum)
        .unwrap();
    assert!(!h.is_directed());
    let w = h.edge_attr_numeric_values("w", ..).unwrap();
    let total: f64 = w.iter().sum();
    assert_eq!(total, 31.0, "Sum preserves the total weight");
    for e in h.edge_ids() {
        let expected = match h.edge(e).unwrap() {
            (0, 1) => 7.0,
            (1, 2) => 8.0,
            (2, 2) => 16.0,
            other => panic!("unexpected edge {other:?}"),
        };
        assert_eq!(w[e as usize], expected);
    }
    assert_eq!(
        h.vertex_attr_str_values("name", ..).unwrap(),
        vec!["a", "b", "c"]
    );

    // Mutual: one mutual pair (0->1, 1->0) merged, 1->2 lost, the loop kept.
    let mut h = g.clone();
    h.to_undirected_with_attributes(ToUndirected::Mutual, &sum)
        .unwrap();
    let mut pairs: Vec<((i64, i64), f64)> = h
        .edge_list()
        .into_iter()
        .zip(h.edge_attr_numeric_values("w", ..).unwrap())
        .collect();
    pairs.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert!(pairs.iter().all(|&(e, _)| e != (1, 2)));
    assert!(pairs.contains(&((2, 2), 16.0)), "{pairs:?}");

    // Each: `edge_comb` is not used, all edges and values are kept.
    let mut h = g.clone();
    h.to_undirected_with_attributes(ToUndirected::Each, &AttributeCombination::new())
        .unwrap();
    assert_eq!(
        h.edge_attr_numeric_values("w", ..).unwrap(),
        vec![1.0, 2.0, 4.0, 8.0, 16.0]
    );

    // Summing strings is an attribute combination error.
    let mut h = g.clone();
    h.set_edge_attr_str_values("s", &["a", "b", "c", "d", "e"])
        .unwrap();
    assert_eq!(
        h.to_undirected_with_attributes(ToUndirected::Collapse, &sum)
            .unwrap_err()
            .kind(),
        ErrorKind::AttributeCombination
    );
}

/// `Random` uses the default RNG of the calling thread: with the same seed,
/// every thread picks the same values, and the picks come from the group.
#[test]
fn random_combination_is_reproducible_per_thread() {
    setup();
    fn picks(seed: u64) -> Vec<f64> {
        rng::seed(seed).unwrap();
        // 40 parallel edges 0-1 and 40 parallel edges 1-2.
        let edges: Vec<(i64, i64)> = (0..80).map(|i| (i / 40, i / 40 + 1)).collect();
        let mut g = Graph::from_edges(&edges, 3, false).unwrap();
        let w: Vec<f64> = (0..80).map(f64::from).collect();
        g.set_edge_attr_numeric_values("w", &w).unwrap();
        simplify(&mut g, &AttributeCombination::all(Comb::Random).unwrap());
        assert_eq!(g.edge_list(), vec![(0, 1), (1, 2)]);
        g.edge_attr_numeric_values("w", ..).unwrap()
    }
    let reference = picks(42);
    assert!((0.0..40.0).contains(&reference[0]), "{reference:?}");
    assert!((40.0..80.0).contains(&reference[1]), "{reference:?}");
    let handles: Vec<_> = (0..6)
        .map(|_| std::thread::spawn(|| (picks(42), picks(42))))
        .collect();
    for h in handles {
        let (a, b) = h.join().unwrap();
        assert_eq!(a, reference);
        assert_eq!(b, reference);
    }
    // Different seeds eventually pick different values.
    assert!((0..20).any(|s| picks(s) != reference));
}

/// `igraph_simplify` only combines edge attributes when it rebuilds the graph
/// to merge multi-edges: deleting loops alone, or simplifying a graph igraph
/// already knows to be simple, keeps every remaining edge attribute, even
/// with the plain `Graph::simplify` or an empty combination.
#[test]
fn simplify_keeps_attributes_when_nothing_is_merged() {
    setup();
    let mut g = Graph::from_edges(&[(0, 1), (0, 1), (1, 1), (1, 2)], 3, false).unwrap();
    g.set_edge_attr_numeric_values("w", &[1.0, 2.0, 4.0, 8.0])
        .unwrap();

    // Loops only: deleted in place, the combination is not used.
    g.simplify_with_attributes(false, true, &AttributeCombination::new())
        .unwrap();
    assert_eq!(g.edge_list(), vec![(0, 1), (0, 1), (1, 2)]);
    assert_eq!(
        g.edge_attr_numeric_values("w", ..).unwrap(),
        vec![1.0, 2.0, 8.0]
    );

    // Merging multi-edges combines them.
    simplify(&mut g, &AttributeCombination::all(Comb::Sum).unwrap());
    assert_eq!(g.edge_attr_numeric_values("w", ..).unwrap(), vec![3.0, 8.0]);

    // Now igraph knows the graph is simple: nothing is rebuilt, nothing lost.
    g.simplify(true, true).unwrap();
    simplify(&mut g, &AttributeCombination::new());
    assert_eq!(g.edge_list(), vec![(0, 1), (1, 2)]);
    assert_eq!(g.edge_attr_numeric_values("w", ..).unwrap(), vec![3.0, 8.0]);
}

#[test]
fn record_clone_is_deep_for_every_type() {
    setup();
    let records = [
        AttributeRecord::numeric("n", &[1.0, f64::INFINITY, f64::NAN]).unwrap(),
        AttributeRecord::boolean("b", &[true, false, true]).unwrap(),
        AttributeRecord::string("s", &["x", "", "z"]).unwrap(),
        AttributeRecord::new("u", AttributeType::Unspecified).unwrap(),
    ];
    for rec in &records {
        let mut copy = rec.clone();
        // Records compare as data: NaN values and NaN defaults are equal.
        assert_eq!(&copy, rec);
        assert_eq!(copy.attribute_type(), rec.attribute_type());
        if copy.attribute_type() != AttributeType::Unspecified {
            copy.resize(0).unwrap();
            assert!(copy.is_empty());
            assert!(!rec.is_empty(), "the original keeps its values");
            assert_ne!(&copy, rec);
        }
    }
    let mut other = records[0].clone();
    other.set_default(0.0).unwrap();
    assert_ne!(other, records[0], "different defaults");

    // Clones are usable as attribute records of new edges.
    let mut g = Graph::new(2, false);
    let w = records[0].clone();
    g.add_edges_with_attributes(&[(0, 1), (1, 0), (1, 1)], &[w])
        .unwrap();
    let n = g.edge_attr_numeric_values("n", ..).unwrap();
    assert_eq!(n[..2], [1.0, f64::INFINITY]);
    assert!(n[2].is_nan());
}

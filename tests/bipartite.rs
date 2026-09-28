//! Integration tests for the `bipartite` module (bipartite graphs and matchings).

mod common;

use common::{complete, cycle, karate, path};
use igraph::bipartite::{BipartiteGameOptions, BipartiteGraph, UNMATCHED};
use igraph::prelude::*;

/// Seeds this thread's default RNG (each test thread has its own), then runs `f`.
fn seeded<R>(seed: u64, f: impl FnOnce() -> R) -> R {
    rng::seed(seed).unwrap();
    f()
}

/// Size of a maximum bipartite matching computed as a maximum flow: a source
/// feeds every `false` vertex, every `true` vertex drains into a sink, and
/// every edge is oriented from `false` to `true`, all with unit capacity.
fn matching_size_by_maxflow(g: &Graph, types: &[bool]) -> f64 {
    let n = g.vcount() as i64;
    let (source, sink) = (n, n + 1);
    let mut arcs: Vec<(i64, i64)> = g
        .edge_list()
        .into_iter()
        .map(|(u, v)| if types[u as usize] { (v, u) } else { (u, v) })
        .collect();
    for (v, &t) in (0..).zip(types) {
        arcs.push(if t { (v, sink) } else { (source, v) });
    }
    let net = Graph::from_edges(&arcs, n as usize + 2, true).unwrap();
    net.maxflow_value(source, sink, None).unwrap()
}

fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
    v.sort();
    v
}

/// Size of a maximum matching by exhaustive search (small graphs only).
fn brute_force_matching(edges: &[(i64, i64)], weights: &[f64], n: usize) -> (usize, f64) {
    fn go(
        i: usize,
        edges: &[(i64, i64)],
        w: &[f64],
        used: &mut Vec<bool>,
        best: &mut (usize, f64),
        cur: (usize, f64),
    ) {
        if cur.0 > best.0 {
            best.0 = cur.0;
        }
        if cur.1 > best.1 {
            best.1 = cur.1;
        }
        for j in i..edges.len() {
            let (u, v) = (edges[j].0 as usize, edges[j].1 as usize);
            if !used[u] && !used[v] {
                used[u] = true;
                used[v] = true;
                go(j + 1, edges, w, used, best, (cur.0 + 1, cur.1 + w[j]));
                used[u] = false;
                used[v] = false;
            }
        }
    }
    let mut best = (0, 0.0);
    go(0, edges, weights, &mut vec![false; n], &mut best, (0, 0.0));
    best
}

// ---------------------------------------------------------------------------
// Bipartiteness
// ---------------------------------------------------------------------------

#[test]
fn even_cycles_are_bipartite_odd_cycles_are_not() {
    for n in 3..=12 {
        let c = cycle(n);
        assert_eq!(c.is_bipartite().unwrap(), n % 2 == 0, "cycle of length {n}");
        match c.bipartite_types().unwrap() {
            Some(types) => {
                assert_eq!(n % 2, 0);
                // A proper 2-coloring of a cycle alternates.
                for (u, v) in c.edge_list() {
                    assert_ne!(types[u as usize], types[v as usize]);
                }
            }
            None => assert_eq!(n % 2, 1),
        }
    }
}

#[test]
fn trees_and_forests_are_bipartite_cliques_are_not() {
    assert!(path(10).is_bipartite().unwrap());
    assert!(Graph::new(5, false).is_bipartite().unwrap()); // no edges at all
    assert!(Graph::new(0, false).is_bipartite().unwrap());
    assert!(complete(2).is_bipartite().unwrap());
    assert!(!complete(3).is_bipartite().unwrap());
    // The karate club has plenty of triangles.
    assert!(!karate().is_bipartite().unwrap());
    assert!(
        BipartiteGraph::from_graph(Graph::famous("Zachary").unwrap())
            .unwrap()
            .is_none()
    );
}

#[test]
fn famous_graphs_bipartiteness_girth_and_perfect_matchings() {
    // (name, is bipartite, |V|, class sizes when bipartite, maximum matching)
    let cases = [
        ("Cubical", true, 8, (4, 4), 4),
        ("Franklin", true, 12, (6, 6), 6),
        ("Heawood", true, 14, (7, 7), 7),
        ("Herschel", true, 11, (5, 6), 5),
        ("Folkman", true, 20, (10, 10), 10),
        ("Levi", true, 30, (15, 15), 15),
        ("Petersen", false, 10, (0, 0), 0),
        ("Coxeter", false, 28, (0, 0), 0),
        ("Dodecahedron", false, 20, (0, 0), 0),
    ];
    for (name, bip, n, sizes, matching) in cases {
        let g = Graph::famous(name).unwrap();
        assert_eq!(g.vcount(), n, "{name}");
        assert_eq!(g.is_bipartite().unwrap(), bip, "{name}");
        // A graph is bipartite iff it has no odd cycle: bipartite graphs have
        // an even girth (these non-bipartite ones happen to have odd girth).
        let girth = g.girth().unwrap().unwrap();
        assert_eq!(girth.is_multiple_of(2), bip, "{name}: girth {girth}");
        let Some(b) = BipartiteGraph::from_graph(g).unwrap() else {
            continue;
        };
        let (a, c) = b.part_sizes();
        assert_eq!((a.min(c), a.max(c)), sizes, "{name}");
        let m = b.maximum_matching(None).unwrap();
        assert_eq!(m.size, matching, "{name}");
        assert_eq!(
            matching_size_by_maxflow(&b.graph, &b.types),
            matching as f64
        );
    }
}

#[test]
fn heawood_projections_are_the_fano_plane() {
    // The Heawood graph is the incidence graph of the Fano plane: any two
    // points lie on exactly one line and any two lines meet in exactly one
    // point, so both projections are K7 with all multiplicities 1.
    let b = BipartiteGraph::from_graph(Graph::famous("Heawood").unwrap())
        .unwrap()
        .unwrap();
    let p = b.projection().unwrap();
    for (proj, mult) in [(&p.proj1, &p.multiplicity1), (&p.proj2, &p.multiplicity2)] {
        assert_eq!((proj.vcount(), proj.ecount()), (7, 21));
        assert!(proj.is_complete().unwrap());
        assert_eq!(mult, &vec![1; 21]);
    }
    // The biadjacency matrix is the 7x7 incidence matrix: 3 ones per row
    // and per column.
    let inc = b.biadjacency(None).unwrap().matrix.to_rows();
    assert!(inc.iter().all(|row| row.iter().sum::<f64>() == 3.0));
    assert!((0..7).all(|j| inc.iter().map(|row| row[j]).sum::<f64>() == 3.0));
}

#[test]
fn bipartite_types_drive_the_bipartite_layout() {
    let g = Graph::famous("Cubical").unwrap();
    let types = g.bipartite_types().unwrap().unwrap();
    let layout = g.layout_bipartite(&types, 1.0, 2.0, 100).unwrap();
    assert_eq!((layout.nrow(), layout.ncol()), (8, 2));
    for (v, &t) in types.iter().enumerate() {
        // `true` vertices on y = 0, `false` ones on y = vgap.
        assert_eq!(layout[(v, 1)], if t { 0.0 } else { 2.0 }, "vertex {v}");
    }
}

#[test]
fn realized_bidegree_sequences_are_bipartite() {
    let (d1, d2) = ([3, 2, 2, 1], [2, 2, 2, 2]);
    assert!(igraph::mixing::is_bigraphical(&d1, &d2, EdgeTypeSw::Simple).unwrap());
    let g = Graph::realize_bipartite_degree_sequence(
        &d1,
        &d2,
        igraph::constructors::AllowedEdgeTypes::Simple,
        RealizeDegseq::Smallest,
    )
    .unwrap();
    // The first 4 vertices form one class, the last 4 the other.
    let types: Vec<bool> = (0..8).map(|v| v >= 4).collect();
    let b = BipartiteGraph::new(types, &g.edge_list(), false).unwrap();
    let degrees: Vec<i64> = (0..8)
        .map(|v| {
            b.graph
                .degree_of(v, NeighborMode::All, Loops::Twice)
                .unwrap()
        })
        .collect();
    assert_eq!(degrees, [&d1[..], &d2[..]].concat());
    // Every row / column sum of the biadjacency matrix is a degree.
    let rows = b.biadjacency(None).unwrap().matrix.to_rows();
    let row_sums: Vec<f64> = rows.iter().map(|r| r.iter().sum()).collect();
    assert_eq!(row_sums, vec![3.0, 2.0, 2.0, 1.0]);
    // Hall's condition holds (every class-1 set has enough neighbors), and
    // the class of size 4 can be perfectly matched.
    assert_eq!(b.maximum_matching(None).unwrap().size, 4);
}

#[test]
fn self_loop_prevents_bipartiteness() {
    let g = Graph::from_edges(&[(0, 1), (1, 1)], 2, false).unwrap();
    assert!(!g.is_bipartite().unwrap());
}

#[test]
fn bipartite_types_is_a_valid_coloring_for_directed_graphs() {
    // Directions are ignored: an oriented 4-cycle is still bipartite.
    let g = Graph::from_edges(&[(0, 1), (2, 1), (2, 3), (0, 3)], 4, true).unwrap();
    let b = BipartiteGraph::from_graph(g).unwrap().unwrap();
    for (u, v) in b.graph.edge_list() {
        assert_ne!(b.types[u as usize], b.types[v as usize]);
    }
    assert_eq!(b.part_sizes(), (2, 2));
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

#[test]
fn create_bipartite_matches_igraph_example() {
    // examples/simple/igraph_bipartite_create.c
    let edges = [
        (0, 1),
        (1, 2),
        (3, 4),
        (5, 6),
        (6, 5),
        (1, 4),
        (1, 6),
        (0, 3),
    ];
    let types: Vec<bool> = (0..7).map(|i| i % 2 == 1).collect();
    let g = Graph::create_bipartite(&types, &edges, true).unwrap();
    assert!(g.is_directed());
    let expected = vec![
        (0, 1),
        (0, 3),
        (1, 2),
        (1, 4),
        (1, 6),
        (3, 4),
        (5, 6),
        (6, 5),
    ];
    assert_eq!(sorted(g.edge_list()), expected);
}

#[test]
fn create_bipartite_rejects_bad_input() {
    let types = [false, true, false];
    let err = Graph::create_bipartite(&types, &[(0, 2)], false).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let err = Graph::create_bipartite(&types, &[(0, 3)], false).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    let err = Graph::create_bipartite(&types, &[(-1, 1)], false).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    // No edges: just isolated vertices.
    let g = Graph::create_bipartite(&types, &[], false).unwrap();
    assert_eq!((g.vcount(), g.ecount()), (3, 0));
}

#[test]
fn full_bipartite_counts_and_directions() {
    for (n1, n2) in [(0, 0), (0, 3), (3, 0), (1, 1), (3, 4), (5, 2)] {
        let b = Graph::full_bipartite(n1, n2, false, NeighborMode::All).unwrap();
        assert_eq!(b.graph.vcount(), n1 + n2);
        assert_eq!(b.graph.ecount(), n1 * n2);
        assert_eq!(b.part_sizes(), (n1, n2));
        assert!(b.graph.is_bipartite().unwrap());
    }
    let out = Graph::full_bipartite(2, 3, true, NeighborMode::Out).unwrap();
    assert!(out.graph.edge_list().iter().all(|&(u, v)| u < 2 && v >= 2));
    let inn = Graph::full_bipartite(2, 3, true, NeighborMode::In).unwrap();
    assert!(inn.graph.edge_list().iter().all(|&(u, v)| u >= 2 && v < 2));
    let all = Graph::full_bipartite(2, 2, true, NeighborMode::All).unwrap();
    assert_eq!(all.graph.ecount(), 8);
    assert_eq!(all.part(false), vec![0, 1]);
    assert_eq!(all.part(true), vec![2, 3]);
}

#[test]
fn biadjacency_from_igraph_unit_test() {
    // tests/unit/igraph_biadjacency.c, "five vertices" case.
    let m = Matrix::from_rows(&[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).unwrap();
    let all = Graph::biadjacency(&m, true, NeighborMode::All, true).unwrap();
    assert_eq!(all.types, vec![false, false, true, true, true]);
    assert_eq!(all.graph.ecount(), 30); // 2 * (0+1+2+3+4+5)
    let out = Graph::biadjacency(&m, true, NeighborMode::Out, true).unwrap();
    assert_eq!(out.graph.ecount(), 15);
    assert!(out.graph.edge_list().iter().all(|&(u, v)| u < 2 && v >= 2));
    let single = Graph::biadjacency(&m, false, NeighborMode::All, false).unwrap();
    assert_eq!(single.graph.ecount(), 5); // one per non-zero entry

    // Degenerate shapes: no rows gives only "true" vertices.
    let empty = Graph::biadjacency(&Matrix::zeros(0, 5), true, NeighborMode::All, false).unwrap();
    assert_eq!(empty.types, vec![true; 5]);
    assert_eq!(empty.graph.ecount(), 0);

    // Non-integer multiplicities are truncated.
    let frac = Matrix::from_rows(&[[0.0, 1.2, 1.8], [5.0, 0.0, 2.2]]).unwrap();
    let g = Graph::biadjacency(&frac, false, NeighborMode::All, true).unwrap();
    assert_eq!(g.graph.ecount(), 1 + 1 + 5 + 2);
}

#[test]
fn biadjacency_rejects_invalid_multiplicities() {
    for bad in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e300] {
        let m = Matrix::from_rows(&[[1.0, bad]]).unwrap();
        let err = Graph::biadjacency(&m, false, NeighborMode::All, true).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidValue, "entry {bad}");
        // Without `multiple`, any non-zero entry (NaN included) is one edge.
        let g = Graph::biadjacency(&m, false, NeighborMode::All, false).unwrap();
        assert_eq!(g.graph.ecount(), 2, "entry {bad}");
    }
}

#[test]
fn weighted_biadjacency_from_igraph_unit_test() {
    // tests/unit/igraph_weighted_biadjacency.c, "Infinity and NaN" case.
    let m = Matrix::from_rows(&[
        [0.0, -4.5],
        [f64::INFINITY, f64::NEG_INFINITY],
        [f64::NAN, 0.0],
    ])
    .unwrap();
    let w = Graph::weighted_biadjacency(&m, false, NeighborMode::All).unwrap();
    assert_eq!(w.types, vec![false, false, false, true, true]);
    let mut edges: Vec<((i64, i64), f64)> = w
        .graph
        .edge_list()
        .into_iter()
        .zip(w.weights.iter().copied())
        .collect();
    edges.sort_by_key(|e| e.0);
    assert_eq!(
        edges.iter().map(|e| e.0).collect::<Vec<_>>(),
        vec![(0, 4), (1, 3), (1, 4), (2, 3)]
    );
    assert_eq!(edges[0].1, -4.5);
    assert_eq!(edges[1].1, f64::INFINITY);
    assert_eq!(edges[2].1, f64::NEG_INFINITY);
    assert!(edges[3].1.is_nan());

    // Directed with mutual edges: each weight appears twice.
    let m = Matrix::from_rows(&[[1.25]]).unwrap();
    let w = Graph::weighted_biadjacency(&m, true, NeighborMode::All).unwrap();
    assert_eq!(sorted(w.graph.edge_list()), vec![(0, 1), (1, 0)]);
    assert_eq!(w.weights, vec![1.25, 1.25]);
}

// ---------------------------------------------------------------------------
// Biadjacency matrices and projections
// ---------------------------------------------------------------------------

#[test]
fn get_biadjacency_from_igraph_unit_test() {
    // tests/unit/igraph_get_biadjacency.c: disconnected graph with multi-edges.
    let edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 0),
        (2, 0),
        (2, 3),
        (3, 4),
        (3, 4),
    ];
    let g = Graph::from_edges(&edges, 6, false).unwrap();
    let types = [false, true, true, false, true, false];
    let b = g.get_biadjacency(&types, None).unwrap();
    assert_eq!(
        b.matrix.to_rows(),
        vec![
            vec![1.0, 3.0, 0.0],
            vec![1.0, 1.0, 2.0],
            vec![0.0, 0.0, 0.0]
        ]
    );
    assert_eq!(b.row_ids, vec![0, 3, 5]);
    assert_eq!(b.col_ids, vec![1, 2, 4]);

    // A non-bipartite edge (0-3) is ignored with a warning.
    let mut g2 = g.clone();
    g2.add_edge(0, 3).unwrap();
    let _ = igraph::error::take_warnings();
    let b2 = g2.get_biadjacency(&types, None).unwrap();
    assert_eq!(b2.matrix, b.matrix);
    assert!(!igraph::error::take_warnings().is_empty());

    // Wrong type vector length.
    let err = g.get_biadjacency(&[], None).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let err = g.get_biadjacency(&types, Some(&[1.0])).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

#[test]
fn biadjacency_round_trip() {
    let rows = [
        [2.0, 0.0, 1.0, 0.0],
        [0.0, 3.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ];
    let m = Matrix::from_rows(&rows).unwrap();
    let b = Graph::biadjacency(&m, false, NeighborMode::All, true).unwrap();
    let back = b.biadjacency(None).unwrap();
    assert_eq!(back.matrix, m);
    assert_eq!(back.row_ids, vec![0, 1, 2]);
    assert_eq!(back.col_ids, vec![3, 4, 5, 6]);

    // Weighted round trip.
    let wrows = [[0.5, 0.0], [0.0, -2.0], [7.0, 1.5]];
    let wm = Matrix::from_rows(&wrows).unwrap();
    let w = Graph::weighted_biadjacency(&wm, false, NeighborMode::All).unwrap();
    let back = w.graph.get_biadjacency(&w.types, Some(&w.weights)).unwrap();
    assert_eq!(back.matrix, wm);
}

#[test]
fn projection_matches_igraph_example() {
    // examples/simple/igraph_bipartite_projection.c, probe1 = 1.
    let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 3), (3, 2), (2, 4), (4, 5)], 6, false).unwrap();
    let types = [false, true, false, true, true, false];
    let p = g.bipartite_projection(&types, Some(1)).unwrap();
    // proj1 contains vertex 1, i.e. the "true" vertices {1, 3, 4}.
    assert_eq!(p.proj1.vcount(), 3);
    let with_mult =
        |g: &Graph, m: &[i64]| sorted(g.edge_list().into_iter().zip(m.iter().copied()).collect());
    assert_eq!(
        with_mult(&p.proj1, &p.multiplicity1),
        vec![((0, 1), 2), ((0, 2), 1), ((1, 2), 1)]
    );
    // proj2: the "false" vertices {0, 2, 5}.
    assert_eq!(p.proj2.vcount(), 3);
    assert_eq!(
        with_mult(&p.proj2, &p.multiplicity2),
        vec![((0, 1), 2), ((1, 2), 1)]
    );

    // Default order: proj1 is the "false" class.
    let q = g.bipartite_projection(&types, None).unwrap();
    assert_eq!(q.proj1, p.proj2);
    assert_eq!(q.proj2, p.proj1);

    // Single projections agree.
    let (only_true, m_true) = g.bipartite_projection_of(&types, true).unwrap();
    assert_eq!(only_true, q.proj2);
    assert_eq!(m_true, q.multiplicity2);
    let (only_false, m_false) = g.bipartite_projection_of(&types, false).unwrap();
    assert_eq!(only_false, q.proj1);
    assert_eq!(m_false, q.multiplicity1);

    // Sizes without building the graphs.
    let s = g.bipartite_projection_size(&types).unwrap();
    assert_eq!((s.vcount1, s.ecount1), (q.proj1.vcount(), q.proj1.ecount()));
    assert_eq!((s.vcount2, s.ecount2), (q.proj2.vcount(), q.proj2.ecount()));
}

#[test]
fn projection_errors() {
    let g = Graph::from_edges(&[(0, 1), (0, 2)], 3, false).unwrap();
    let types = [false, true, false];
    // Edge 0-2 joins two "false" vertices.
    let err = g.bipartite_projection(&types, None).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    assert!(!err.message().is_empty());
    let err = g.bipartite_projection_size(&types).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let ok = Graph::from_edges(&[(0, 1), (2, 1)], 3, false).unwrap();
    let err = ok.bipartite_projection(&types, Some(3)).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    let err = ok.bipartite_projection(&types, Some(-2)).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    assert!(ok.bipartite_projection(&[false], None).is_err());
}

#[test]
fn projection_multiplicities_count_paths_in_multigraphs() {
    // Vertex 0 reaches 2 through a double edge to 1, vertex 3 through a
    // single edge; the directed orientation is ignored.
    let types = [false, true, false, true];
    let g = Graph::from_edges(&[(0, 1), (1, 0), (2, 1), (2, 3)], 4, true).unwrap();
    let p = g.bipartite_projection(&types, None).unwrap();
    assert!(!p.proj1.is_directed());
    assert_eq!(p.proj1.edge_list(), vec![(0, 1)]);
    assert_eq!(p.multiplicity1, vec![2]);
    assert_eq!(p.proj2.edge_list(), vec![(0, 1)]);
    assert_eq!(p.multiplicity2, vec![1]);
    let s = g.bipartite_projection_size(&types).unwrap();
    assert_eq!((s.vcount1, s.ecount1, s.vcount2, s.ecount2), (2, 1, 2, 1));
}

#[test]
fn projection_of_complete_bipartite_is_complete() {
    let k = Graph::full_bipartite(4, 6, false, NeighborMode::All).unwrap();
    let p = k.projection().unwrap();
    assert_eq!((p.proj1.vcount(), p.proj1.ecount()), (4, 6));
    assert_eq!((p.proj2.vcount(), p.proj2.ecount()), (6, 15));
    // Two bottom vertices share all 6 top vertices, and vice versa.
    assert!(p.multiplicity1.iter().all(|&m| m == 6));
    assert!(p.multiplicity2.iter().all(|&m| m == 4));
}

// ---------------------------------------------------------------------------
// Random bipartite graphs
// ---------------------------------------------------------------------------

#[test]
fn gnp_extremes_and_properties() {
    seeded(42, || {
        let opts = BipartiteGameOptions::default();
        let empty = Graph::bipartite_game_gnp(5, 7, 0.0, &opts).unwrap();
        assert_eq!((empty.graph.vcount(), empty.graph.ecount()), (12, 0));
        let full = Graph::bipartite_game_gnp(5, 7, 1.0, &opts).unwrap();
        assert_eq!(full.graph.ecount(), 35);
        assert_eq!(full.part_sizes(), (5, 7));

        let b = Graph::bipartite_game_gnp(30, 40, 0.2, &opts).unwrap();
        for (u, v) in b.graph.edge_list() {
            assert_ne!(b.types[u as usize], b.types[v as usize]);
        }
        // Expected 240 edges; allow a wide margin.
        assert!(
            (150..=330).contains(&b.graph.ecount()),
            "{}",
            b.graph.ecount()
        );

        let directed =
            Graph::bipartite_game_gnp(4, 4, 1.0, &opts.with_directed(true, NeighborMode::In))
                .unwrap();
        assert!(directed.graph.is_directed());
        assert!(
            directed
                .graph
                .edge_list()
                .iter()
                .all(|&(u, v)| u >= 4 && v < 4)
        );
        let mutual =
            Graph::bipartite_game_gnp(3, 3, 1.0, &opts.with_directed(true, NeighborMode::All))
                .unwrap();
        assert_eq!(mutual.graph.ecount(), 18);

        // Multigraph extension: p is an expected multiplicity and may exceed 1.
        let multi = opts.with_allowed_edge_types(EdgeTypeSw::Multi);
        let m = Graph::bipartite_game_gnp(3, 3, 5.0, &multi).unwrap();
        assert!(m.graph.ecount() > 9);

        assert!(Graph::bipartite_game_gnp(3, 3, 1.5, &opts).is_err());
        assert!(Graph::bipartite_game_gnp(3, 3, -0.5, &opts).is_err());
        for bad in [f64::NAN, f64::INFINITY] {
            for o in [opts, multi, multi.with_edge_labeled(true)] {
                let err = Graph::bipartite_game_gnp(3, 3, bad, &o).unwrap_err();
                assert_eq!(err.kind(), ErrorKind::InvalidValue);
            }
        }
        // Edge-labeled G(n1, n2, p) exists only for multigraphs in igraph 1.0.0 and 1.0.1.
        let err = Graph::bipartite_game_gnp(3, 3, 0.5, &opts.with_edge_labeled(true)).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unimplemented);
        let m = Graph::bipartite_game_gnp(3, 3, 2.0, &multi.with_edge_labeled(true)).unwrap();
        assert!(m.graph.is_bipartite().unwrap());
        // Self-loops cannot occur: `Loops` behaves like `Simple`.
        let loops = opts.with_allowed_edge_types(EdgeTypeSw::Loops);
        let full = Graph::bipartite_game_gnp(3, 3, 1.0, &loops).unwrap();
        assert_eq!(full.graph.ecount(), 9);
    });
}

#[test]
fn gnm_exact_edge_counts() {
    seeded(42, || {
        let opts = BipartiteGameOptions::default();
        for m in [0, 1, 10, 24] {
            let b = Graph::bipartite_game_gnm(4, 6, m, &opts).unwrap();
            assert_eq!(b.graph.ecount(), m);
            assert!(b.graph.is_bipartite().unwrap());
        }
        // All 24 possible edges: this is K(4, 6).
        let k = Graph::bipartite_game_gnm(4, 6, 24, &opts).unwrap();
        assert_eq!(
            k.graph.get_biadjacency(&k.types, None).unwrap().matrix,
            Matrix::from_rows(&[[1.0; 6]; 4]).unwrap()
        );
        let err = Graph::bipartite_game_gnm(4, 6, 25, &opts).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidValue);
        // With multi-edges, any m is fine, also edge-labeled.
        let multi = opts
            .with_allowed_edge_types(EdgeTypeSw::Multi)
            .with_edge_labeled(true);
        let b = Graph::bipartite_game_gnm(2, 2, 50, &multi).unwrap();
        assert_eq!(b.graph.ecount(), 50);
        assert_eq!(
            b.biadjacency(None)
                .unwrap()
                .matrix
                .as_slice()
                .iter()
                .sum::<f64>(),
            50.0
        );
        // Directed, both orientations drawn independently: up to 2*n1*n2 edges.
        let d = Graph::bipartite_game_gnm(2, 3, 12, &opts.with_directed(true, NeighborMode::All))
            .unwrap();
        assert_eq!(d.graph.ecount(), 12);
    });
}

#[test]
fn random_games_are_reproducible() {
    let opts = BipartiteGameOptions::default();
    let draw = |seed| {
        seeded(seed, || {
            let a = Graph::bipartite_game_gnm(10, 10, 30, &opts).unwrap();
            let b = Graph::bipartite_iea_game(10, 10, 30, true, NeighborMode::Out).unwrap();
            let c = Graph::bipartite_game_gnp(10, 10, 0.3, &opts).unwrap();
            (a, b, c)
        })
    };
    assert_eq!(draw(2024), draw(2024));
    assert_ne!(draw(2024).0, draw(2025).0);
}

#[test]
fn seeded_games_are_reproducible_in_parallel_threads() {
    // Every thread has its own default RNG: the same seed gives the same
    // graphs in every thread, whatever the other threads are doing.
    let run = |seed: u64| {
        std::thread::spawn(move || {
            rng::seed(seed).unwrap();
            let opts = BipartiteGameOptions::default();
            (0..20)
                .map(|_| {
                    let g = Graph::bipartite_game_gnp(8, 9, 0.4, &opts).unwrap();
                    g.graph.edge_list()
                })
                .collect::<Vec<_>>()
        })
    };
    let handles: Vec<_> = [5, 5, 6, 5, 6].into_iter().map(run).collect();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(results[0], results[1]);
    assert_eq!(results[0], results[3]);
    assert_eq!(results[2], results[4]);
    assert_ne!(results[0], results[2]);
}

#[test]
fn iea_game_properties() {
    seeded(7, || {
        let b = Graph::bipartite_iea_game(3, 5, 40, false, NeighborMode::All).unwrap();
        assert_eq!((b.graph.vcount(), b.graph.ecount()), (8, 40));
        // The biadjacency matrix counts every edge once.
        let total: f64 = b.biadjacency(None).unwrap().matrix.as_slice().iter().sum();
        assert_eq!(total, 40.0);
        let d = Graph::bipartite_iea_game(3, 5, 20, true, NeighborMode::In).unwrap();
        assert!(d.graph.edge_list().iter().all(|&(u, v)| u >= 3 && v < 3));
        let out = Graph::bipartite_iea_game(3, 5, 20, true, NeighborMode::Out).unwrap();
        assert!(out.graph.edge_list().iter().all(|&(u, v)| u < 3 && v >= 3));
        // No bottom or no top vertices: only m = 0 is possible.
        assert_eq!(
            Graph::bipartite_iea_game(0, 5, 0, false, NeighborMode::All)
                .unwrap()
                .graph
                .vcount(),
            5
        );
        assert!(Graph::bipartite_iea_game(0, 5, 1, false, NeighborMode::All).is_err());
    });
}

#[test]
fn iea_game_is_edge_labeled_gnm() {
    // `bipartite_iea_game` draws exactly the edge-labeled multigraph
    // G(n1, n2, m) model, random number for random number.
    let labeled = BipartiteGameOptions::default()
        .with_allowed_edge_types(EdgeTypeSw::Multi)
        .with_edge_labeled(true);
    for mode in [NeighborMode::Out, NeighborMode::In, NeighborMode::All] {
        let iea = seeded(99, || {
            Graph::bipartite_iea_game(4, 3, 25, true, mode).unwrap()
        });
        let gnm = seeded(99, || {
            Graph::bipartite_game_gnm(4, 3, 25, &labeled.with_directed(true, mode)).unwrap()
        });
        assert_eq!(iea, gnm, "{mode:?}");
    }
}

#[test]
fn iea_game_samples_by_independent_edge_assignment() {
    // One bottom vertex, two top vertices, two undirected edges: the
    // multigraphs are {a, a}, {b, b} and {a, b}. Uniform sampling (the
    // edge-unlabeled G(n1, n2, m) model) gives each probability 1/3, while
    // independent edge assignment gives the simple graph {a, b} probability
    // 1/2. igraph 1.0.0 and 1.0.1's `igraph_bipartite_iea_game` samples
    // uniformly by mistake; the Rust wrapper must not.
    const DRAWS: usize = 6000;
    let simple_fraction = |draw: &dyn Fn() -> Graph| {
        let simple = (0..DRAWS)
            .filter(|_| {
                let e = draw().edge_list();
                e[0] != e[1]
            })
            .count();
        simple as f64 / DRAWS as f64
    };
    seeded(2025, || {
        let iea = simple_fraction(&|| {
            Graph::bipartite_iea_game(1, 2, 2, false, NeighborMode::Out)
                .unwrap()
                .graph
        });
        // Standard deviation: sqrt(1/4 / 6000) < 0.007.
        assert!((iea - 0.5).abs() < 0.035, "IEA: {iea}");
        let multi = BipartiteGameOptions::default().with_allowed_edge_types(EdgeTypeSw::Multi);
        let uniform =
            simple_fraction(&|| Graph::bipartite_game_gnm(1, 2, 2, &multi).unwrap().graph);
        assert!((uniform - 1.0 / 3.0).abs() < 0.035, "uniform: {uniform}");
    });
}

#[test]
fn upstream_bipartite_iea_game_still_samples_uniformly() {
    // Pins the igraph bug worked around by `Graph::bipartite_iea_game`: the
    // C function `igraph_bipartite_iea_game` forwards to the edge-unlabeled
    // multigraph model, so on 1 x 2 vertices with 2 edges it draws the simple
    // graph with probability 1/3 instead of 1/2. If this test starts failing
    // after an igraph upgrade, the bug was fixed upstream and the wrapper can
    // call the C function again.
    const DRAWS: usize = 6000;
    seeded(2025, || {
        let simple = (0..DRAWS)
            .filter(|_| {
                let g = Graph::init_with(|g| unsafe {
                    igraph::ffi::igraph_bipartite_iea_game(
                        g,
                        std::ptr::null_mut(),
                        1,
                        2,
                        2,
                        false,
                        NeighborMode::Out.into(),
                    )
                })
                .unwrap();
                let e = g.edge_list();
                e[0] != e[1]
            })
            .count();
        let fraction = simple as f64 / DRAWS as f64;
        assert!((fraction - 1.0 / 3.0).abs() < 0.035, "C IEA: {fraction}");
    });
}

// ---------------------------------------------------------------------------
// Matchings
// ---------------------------------------------------------------------------

#[test]
fn perfect_matchings_of_even_cycles_and_complete_bipartite_graphs() {
    for n in [2, 4, 6, 10, 20] {
        let c = BipartiteGraph::from_graph(cycle(n)).unwrap().unwrap();
        let m = c.maximum_matching(None).unwrap();
        assert_eq!(m.size, n as usize / 2);
        assert!(
            m.matching.iter().all(|&v| v != UNMATCHED),
            "perfect on C{n}"
        );
        assert!(c.graph.is_matching(Some(&c.types), &m.matching).unwrap());
        assert!(
            c.graph
                .is_maximal_matching(Some(&c.types), &m.matching)
                .unwrap()
        );
    }
    for (n1, n2) in [(1, 1), (3, 3), (2, 5), (6, 4)] {
        let k = Graph::full_bipartite(n1, n2, false, NeighborMode::All).unwrap();
        let m = k.maximum_matching(None).unwrap();
        assert_eq!(m.size, n1.min(n2));
        assert_eq!(m.weight, n1.min(n2) as f64);
    }
}

#[test]
fn paths_have_floor_n_over_2_matchings() {
    for n in 2..12 {
        let b = BipartiteGraph::from_graph(path(n)).unwrap().unwrap();
        let m = b.maximum_matching(None).unwrap();
        assert_eq!(m.size, n as usize / 2);
        assert_eq!(m.pairs().len(), m.size);
    }
}

#[test]
fn leda_tutorial_matching() {
    // examples/simple/igraph_maximum_bipartite_matching.c
    let edges = [
        (0, 8),
        (0, 12),
        (0, 14),
        (1, 9),
        (1, 10),
        (1, 13),
        (2, 8),
        (2, 9),
        (3, 10),
        (3, 11),
        (3, 13),
        (4, 9),
        (4, 14),
        (5, 14),
        (6, 9),
        (6, 14),
        (7, 8),
        (7, 12),
        (7, 14),
    ];
    let types: Vec<bool> = (0..15).map(|i| i >= 8).collect();
    let g = Graph::create_bipartite(&types, &edges, false).unwrap();
    let m = g.maximum_bipartite_matching(&types, None).unwrap();
    assert_eq!((m.size, m.weight), (6, 6.0));
    assert!(g.is_maximal_matching(Some(&types), &m.matching).unwrap());
    // Vertices 4, 5 and 6 only reach {9, 14} and {14}: at least one stays single.
    let single = [4, 5, 6].iter().filter(|&&v| !m.is_matched(v)).count();
    assert!(single >= 1);
}

#[test]
fn weighted_matching_mit_notes() {
    // tests/unit/igraph_maximum_bipartite_matching.c
    let edges = [
        (0, 6),
        (0, 7),
        (0, 8),
        (0, 9),
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 8),
        (1, 9),
        (2, 5),
        (2, 6),
        (2, 7),
        (2, 8),
        (2, 9),
        (3, 5),
        (3, 7),
        (3, 9),
        (4, 7),
    ];
    let weights = [
        2.0, 7.0, 2.0, 3.0, 1.0, 3.0, 9.0, 3.0, 3.0, 1.0, 3.0, 3.0, 1.0, 2.0, 4.0, 1.0, 2.0, 3.0,
    ];
    let types: Vec<bool> = (0..10).map(|i| i >= 5).collect();
    let g = Graph::create_bipartite(&types, &edges, false).unwrap();
    let m = g
        .maximum_bipartite_matching_eps(&types, Some(&weights), 0.0)
        .unwrap();
    assert_eq!(m.size, 4);
    assert_eq!(m.weight, 19.0);
    assert!(g.is_maximal_matching(Some(&types), &m.matching).unwrap());
    // The reported weight is the sum over the matched edges.
    let sum: f64 = m
        .pairs()
        .iter()
        .map(|&(u, v)| weights[g.get_eid(u, v, false).unwrap().unwrap() as usize])
        .sum();
    assert_eq!(sum, 19.0);
    assert_eq!(brute_force_matching(&edges, &weights, 10).1, 19.0);
}

#[test]
fn weighted_matching_generated_case() {
    // Case 2 of the same unit test: expected weight 41.
    let g =
        Graph::from_edges(&[(0, 5), (0, 6), (1, 7), (2, 5), (3, 5), (3, 9)], 10, false).unwrap();
    let types: Vec<bool> = (0..10).map(|i| i >= 5).collect();
    let m = g
        .maximum_bipartite_matching(&types, Some(&[20.0, 4.0, 20.0, 3.0, 13.0, 1.0]))
        .unwrap();
    assert_eq!(m.weight, 41.0);
}

#[test]
fn matching_agrees_with_brute_force_on_random_graphs() {
    seeded(123, || {
        let opts = BipartiteGameOptions::default();
        for round in 0..25 {
            let b = Graph::bipartite_game_gnp(5, 6, 0.35, &opts).unwrap();
            let edges = b.graph.edge_list();
            let weights: Vec<f64> = (0..edges.len())
                .map(|_| rng::integer(1, 9) as f64)
                .collect();
            let (best_size, best_weight) = brute_force_matching(&edges, &weights, 11);

            let m = b.maximum_matching(None).unwrap();
            assert_eq!(m.size, best_size, "round {round}");
            assert_eq!(
                matching_size_by_maxflow(&b.graph, &b.types),
                best_size as f64
            );
            assert!(
                b.graph
                    .is_maximal_matching(Some(&b.types), &m.matching)
                    .unwrap()
            );

            let w = b.maximum_matching(Some(&weights)).unwrap();
            assert_eq!(w.weight, best_weight, "round {round}");
            assert!(b.graph.is_matching(Some(&b.types), &w.matching).unwrap());
        }
    });
}

#[test]
fn is_matching_edge_cases() {
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, true).unwrap();
    // Directions are ignored.
    assert!(g.is_matching(None, &[1, 0, 3, 2]).unwrap());
    assert!(g.is_matching(None, &[UNMATCHED; 4]).unwrap());
    // Wrong length, asymmetric, out of range: not a matching (no error).
    assert!(!g.is_matching(None, &[1, 0]).unwrap());
    assert!(!g.is_matching(None, &[1, 2, 1, -1]).unwrap());
    assert!(!g.is_matching(None, &[4, -1, -1, -1]).unwrap());
    assert!(!g.is_matching(None, &[-2, -1, -1, -1]).unwrap());
    // Same-type vertices cannot be matched when types are given.
    let types = [false, true, true, false];
    assert!(g.is_matching(None, &[-1, 2, 1, -1]).unwrap());
    assert!(!g.is_matching(Some(&types), &[-1, 2, 1, -1]).unwrap());
    // The empty matching is valid but not maximal.
    assert!(!g.is_maximal_matching(None, &[UNMATCHED; 4]).unwrap());
    // Wrong types length is a Rust-side error.
    assert_eq!(
        g.is_matching(Some(&[true]), &[-1; 4]).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn maximum_matching_argument_errors() {
    let b = Graph::full_bipartite(2, 2, false, NeighborMode::All).unwrap();
    let err = b
        .graph
        .maximum_bipartite_matching(&[false, true], None)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let err = b.maximum_matching(Some(&[1.0, 2.0])).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

#[test]
fn maximum_matching_rejects_non_bipartite_edges() {
    // igraph's greedy phase never inspects edge 1-3 (both "true") and would
    // return a matching for the wrong graph: the Rust side must catch it.
    let types = [false, true, false, true];
    let g = Graph::from_edges(&[(0, 1), (2, 3), (1, 3)], 4, false).unwrap();
    for w in [None, Some(&[1.0, 1.0, 100.0][..])] {
        let err = g.maximum_bipartite_matching(&types, w).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidValue);
    }
    // A self-loop also joins two vertices of the same type.
    let looped = Graph::from_edges(&[(0, 1), (1, 1)], 2, false).unwrap();
    assert!(
        looped
            .maximum_bipartite_matching(&[false, true], None)
            .is_err()
    );
}

#[test]
fn maximum_matching_rejects_non_finite_numbers() {
    let b = Graph::full_bipartite(1, 1, false, NeighborMode::All).unwrap();
    // Infinite weights and a NaN tolerance make igraph loop forever, NaN
    // weights give a meaningless result.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let err = b.maximum_matching(Some(&[bad])).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidValue);
    }
    let err = b
        .graph
        .maximum_bipartite_matching_eps(&b.types, Some(&[1.0]), f64::NAN)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    // An infinite tolerance is accepted (every slack is "tight").
    let m = b
        .graph
        .maximum_bipartite_matching_eps(&b.types, Some(&[1.0]), f64::INFINITY)
        .unwrap();
    assert_eq!(m.size, 1);
}

#[test]
fn weighted_matching_maximizes_weight_not_cardinality() {
    // Path 0-1-2-3 with types alternating: the perfect matching {0-1, 2-3}
    // weighs 2, the single middle edge 1-2 weighs 10.
    let b = BipartiteGraph::from_graph(path(4)).unwrap().unwrap();
    let w = [1.0, 10.0, 1.0];
    let heavy = b.maximum_matching(Some(&w)).unwrap();
    assert_eq!((heavy.size, heavy.weight), (1, 10.0));
    assert_eq!(heavy.pairs(), vec![(1, 2)]);
    assert!(
        b.graph
            .is_maximal_matching(Some(&b.types), &heavy.matching)
            .unwrap()
    );
    assert_eq!(b.maximum_matching(None).unwrap().size, 2);
    // Negative edges are never worth taking.
    let neg = b.maximum_matching(Some(&[-1.0, -1.0, -1.0])).unwrap();
    assert_eq!((neg.size, neg.weight), (0, 0.0));
    // Fractional weights with an explicit tolerance.
    let frac = b
        .graph
        .maximum_bipartite_matching_eps(&b.types, Some(&[0.5, 0.75, 0.5]), 1e-12)
        .unwrap();
    assert_eq!(frac.weight, 1.0);
}

#[test]
fn matching_helpers() {
    let b = BipartiteGraph::from_graph(path(5)).unwrap().unwrap();
    let m = b.maximum_matching(None).unwrap();
    assert_eq!(m.size, 2);
    let pairs = m.pairs();
    assert_eq!(pairs.len(), 2);
    for &(u, v) in &pairs {
        assert!(u < v);
        assert_eq!(m.mate(u), Some(v));
        assert_eq!(m.mate(v), Some(u));
    }
    assert_eq!(m.mate(-1), None);
    assert_eq!(m.mate(99), None);
    assert_eq!((0..5).filter(|&v| m.is_matched(v)).count(), 4);

    // `into_parts` gives back exactly the fields.
    let (graph, types) = b.clone().into_parts();
    assert_eq!((graph, types), (b.graph, b.types));
}

// ---------------------------------------------------------------------------
// Use cases
// ---------------------------------------------------------------------------

/// A small staffing agency assigns nurses to hospital shifts. Each nurse
/// rates the shifts they are qualified for; the agency wants (1) to cover as
/// many shifts as possible and (2) among the best assignments, the happiest
/// one. Hall's theorem explains why not every shift can be covered.
#[test]
fn use_case_staffing_nurses_to_shifts() {
    let nurses = ["Ada", "Bea", "Cid", "Dan"];
    let shifts = ["Mon-night", "Tue-day", "Wed-day", "Thu-night", "Fri-day"];
    // (nurse, shift, preference 1..10)
    let prefs = [
        (0, 0, 9.0),
        (0, 1, 3.0),
        (1, 1, 8.0),
        (1, 2, 6.0),
        (2, 1, 7.0),
        (2, 2, 7.0),
        (3, 3, 5.0),
        (3, 4, 10.0),
    ];
    let n = nurses.len() as i64;
    let mut types = vec![false; nurses.len()];
    types.extend(vec![true; shifts.len()]);
    let edges: Vec<(i64, i64)> = prefs.iter().map(|&(a, s, _)| (a, n + s)).collect();
    let weights: Vec<f64> = prefs.iter().map(|p| p.2).collect();
    let roster = BipartiteGraph::new(types, &edges, false).unwrap();

    // (1) At most 4 shifts can be covered: there are only 4 nurses.
    let most = roster.maximum_matching(None).unwrap();
    assert_eq!(most.size, 4);

    // (2) The happiest roster.
    let best = roster.maximum_matching(Some(&weights)).unwrap();
    let assignment: Vec<(&str, &str)> = best
        .pairs()
        .iter()
        .map(|&(nurse, shift)| (nurses[nurse as usize], shifts[(shift - n) as usize]))
        .collect();
    assert_eq!(
        assignment,
        vec![
            ("Ada", "Mon-night"),
            ("Bea", "Tue-day"),
            ("Cid", "Wed-day"),
            ("Dan", "Fri-day")
        ]
    );
    assert_eq!(best.weight, 9.0 + 8.0 + 7.0 + 10.0);
    assert!(
        roster
            .graph
            .is_maximal_matching(Some(&roster.types), &best.matching)
            .unwrap()
    );

    // Thursday night stays uncovered: only Dan can do it, and he is happier on Friday.
    assert!(!best.is_matched(n + 3));

    // Which nurses are interchangeable? Project onto nurses: two are linked
    // if they can cover a common shift.
    let (colleagues, overlap) = roster
        .graph
        .bipartite_projection_of(&roster.types, false)
        .unwrap();
    let links: Vec<((i64, i64), i64)> =
        sorted(colleagues.edge_list().into_iter().zip(overlap).collect());
    // Ada-Bea share Tue; Ada-Cid share Tue; Bea-Cid share Tue and Wed. Dan is alone.
    assert_eq!(links, vec![((0, 1), 1), ((0, 2), 1), ((1, 2), 2)]);
    assert_eq!(
        colleagues
            .degree_of(3, NeighborMode::All, Loops::None)
            .unwrap(),
        0
    );
}

/// Southern women (Davis et al.) in miniature: from an event attendance
/// matrix, build the two-mode network, then the "who met whom" projection.
#[test]
fn use_case_attendance_matrix_to_social_network() {
    // Rows: 4 women, columns: 3 events; entries: times attended.
    let attendance = Matrix::from_rows(&[
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    .unwrap();
    let net = Graph::biadjacency(&attendance, false, NeighborMode::All, false).unwrap();
    assert_eq!(net.part_sizes(), (4, 3));
    assert!(net.graph.is_bipartite().unwrap());

    let size = net.graph.bipartite_projection_size(&net.types).unwrap();
    let p = net.projection().unwrap();
    assert_eq!((size.vcount1, size.ecount1), (4, 3));
    // Woman 0 met 1 (event 0) and 2 (event 1); 2 met 3 (event 2).
    assert_eq!(sorted(p.proj1.edge_list()), vec![(0, 1), (0, 2), (2, 3)]);
    // Events 0-1 share woman 0, events 1-2 share woman 2.
    assert_eq!(sorted(p.proj2.edge_list()), vec![(0, 1), (1, 2)]);
    // And the matrix comes back unchanged.
    assert_eq!(net.biadjacency(None).unwrap().matrix, attendance);
}

/// A two-mode Pajek network (igraph's `tests/unit/bipartite.net`): eight
/// people A..H and five clubs x-1..x-5. The reader stores the classes in the
/// boolean `type` vertex attribute, which is all this module needs.
#[test]
fn use_case_two_mode_pajek_network() {
    igraph::attributes::enable().unwrap();
    let mut text = String::from("*Vertices 13 8\n");
    for (i, name) in ["A", "B", "C", "D", "E", "F", "G", "H"].iter().enumerate() {
        text.push_str(&format!("{} \"{name}\"\n", i + 1));
    }
    for j in 1..=5 {
        text.push_str(&format!("{} \"x-{j}\"\n", j + 8));
    }
    text.push_str("*Edges\n1 10\n1 13\n2 12\n3 10\n3 11\n4 11\n5 12\n5 13\n6 12\n8 11\n8 13\n");
    let g = Graph::read_graph_pajek_from_str(&text).unwrap();
    let types = g.vertex_attr_bool_values("type", ..).unwrap();
    assert_eq!(types, [vec![false; 8], vec![true; 5]].concat());
    let net = BipartiteGraph { graph: g, types };
    assert!(net.graph.is_bipartite().unwrap());
    assert_eq!(net.part_sizes(), (8, 5));

    // Club x-1 has no members and G joined nothing: at most 4 people can
    // each chair a different club (x-2..x-5).
    let chairs = net.maximum_matching(None).unwrap();
    assert_eq!(chairs.size, 4);
    assert!(!chairs.is_matched(6) && !chairs.is_matched(8));
    assert_eq!(matching_size_by_maxflow(&net.graph, &net.types), 4.0);

    // People who share a club. x-2 = {A, C}, x-3 = {C, D, H},
    // x-4 = {B, E, F}, x-5 = {A, E, H}.
    let size = net.graph.bipartite_projection_size(&net.types).unwrap();
    let p = net.projection().unwrap();
    assert_eq!((size.vcount1, size.ecount1), (8, p.proj1.ecount()));
    let links: Vec<((i64, i64), i64)> = sorted(
        p.proj1
            .edge_list()
            .into_iter()
            .zip(p.multiplicity1)
            .collect(),
    );
    assert_eq!(
        links,
        vec![
            ((0, 2), 1), // A-C: x-2
            ((0, 4), 1), // A-E: x-5
            ((0, 7), 1), // A-H: x-5
            ((1, 4), 1), // B-E: x-4
            ((1, 5), 1), // B-F: x-4
            ((2, 3), 1), // C-D: x-3
            ((2, 7), 1), // C-H: x-3
            ((3, 7), 1), // D-H: x-3
            ((4, 5), 1), // E-F: x-4
            ((4, 7), 1), // E-H: x-5
        ]
    );
    // Clubs sharing members: x-2/x-3 (C), x-2/x-5 (A), x-3/x-5 (H), x-4/x-5 (E).
    assert_eq!(
        sorted(p.proj2.edge_list()),
        vec![(1, 2), (1, 4), (2, 4), (3, 4)]
    );
}

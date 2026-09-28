//! Integration tests for the deterministic generators (`igraph_constructors.h`).

mod common;

use common::{KARATE_EDGES, karate};
use igraph::{
    constructors::{AllowedEdgeTypes, FamousGraph},
    prelude::*,
};
use std::collections::BTreeSet;

// ---------------------------------------------------------------- helpers ---

fn degrees(g: &Graph) -> Vec<i64> {
    g.degree(VertexSelector::All, NeighborMode::All, Loops::Twice)
        .unwrap()
}

fn out_degrees(g: &Graph) -> Vec<i64> {
    g.degree(VertexSelector::All, NeighborMode::Out, Loops::Twice)
        .unwrap()
}

fn in_degrees(g: &Graph) -> Vec<i64> {
    g.degree(VertexSelector::All, NeighborMode::In, Loops::Twice)
        .unwrap()
}

fn is_regular(g: &Graph, k: i64) -> bool {
    degrees(g).iter().all(|&d| d == k)
}

/// Sorted undirected edge set, `(min, max)` pairs.
fn edge_set(g: &Graph) -> BTreeSet<(i64, i64)> {
    g.edge_list()
        .into_iter()
        .map(|(a, b)| (a.min(b), a.max(b)))
        .collect()
}

fn has_loops_or_multi(g: &Graph) -> bool {
    g.has_loop().unwrap() || g.has_multiple().unwrap()
}

fn is_connected(g: &Graph) -> bool {
    g.is_connected(Connectedness::Weak).unwrap()
}

fn is_tree(g: &Graph) -> bool {
    g.is_tree(NeighborMode::All).unwrap()
}

/// Unweighted diameter of a connected graph.
fn diameter(g: &Graph) -> usize {
    g.diameter().unwrap() as usize
}

/// Length of the shortest cycle (0 if acyclic).
fn girth(g: &Graph) -> usize {
    g.girth().unwrap().unwrap_or(0)
}

fn triangle_count(g: &Graph) -> usize {
    g.count_triangles().unwrap() as usize
}

fn isomorphic(a: &Graph, b: &Graph) -> bool {
    a.isomorphic(b).unwrap()
}

/// Chromatic number by brute-force backtracking (small graphs only).
fn chromatic_number(g: &Graph) -> usize {
    fn colorable(g: &Graph, k: usize, colors: &mut Vec<usize>, v: usize) -> bool {
        if v == g.vcount() {
            return true;
        }
        let nbrs = g.neighbors(v as i64, NeighborMode::All).unwrap();
        for c in 0..k {
            if nbrs
                .iter()
                .all(|&w| (w as usize) >= v || colors[w as usize] != c)
            {
                colors[v] = c;
                if colorable(g, k, colors, v + 1) {
                    return true;
                }
            }
        }
        false
    }
    (0..=g.vcount())
        .find(|&k| colorable(g, k, &mut vec![0; g.vcount()], 0))
        .unwrap()
}

/// Canonical form of a simple graph: its sorted edge set after relabelling
/// by the canonical permutation (two graphs are isomorphic iff their forms
/// are equal).
fn canonical(g: &Graph) -> Vec<(i64, i64)> {
    let perm = g.canonical_permutation(None).unwrap();
    let mut e: Vec<_> = g
        .permute_vertices(&perm)
        .unwrap()
        .edge_list()
        .into_iter()
        .map(|(a, b)| (a.min(b), a.max(b)))
        .collect();
    e.sort();
    e
}

// ------------------------------------------------------- famous & atlas ---

#[test]
fn every_famous_graph_has_its_documented_size() {
    for &f in FamousGraph::ALL {
        let g = Graph::famous(f).unwrap();
        assert_eq!((g.vcount(), g.ecount()), f.size(), "{f}");
        assert!(!g.is_directed(), "{f}");
        assert!(!has_loops_or_multi(&g), "{f} should be simple");
        // Names are case insensitive.
        let lower = Graph::famous(f.name().to_lowercase()).unwrap();
        assert_eq!(lower, g);
    }
    assert_eq!(FamousGraph::ALL.len(), 31);
}

#[test]
fn famous_aliases_and_unknown_names() {
    for (a, b) in [
        ("Dodecahedral", "Dodecahedron"),
        ("Icosahedral", "Icosahedron"),
        ("Octahedral", "Octahedron"),
        ("Tetrahedral", "Tetrahedron"),
        ("Groetzsch", "Grotzsch"),
    ] {
        assert_eq!(Graph::famous(a).unwrap(), Graph::famous(b).unwrap());
    }
    let err = Graph::famous("Unicorn").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    assert!(err.message().contains("Unicorn"));
    assert_eq!(
        Graph::famous("Pet\0ersen").unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn famous_graph_invariants() {
    // Regularity of the regular ones (Frucht and Tutte are regular but asymmetric).
    for (f, k) in [
        (FamousGraph::Petersen, 3),
        (FamousGraph::Heawood, 3),
        (FamousGraph::Coxeter, 3),
        (FamousGraph::McGee, 3),
        (FamousGraph::Levi, 3),
        (FamousGraph::Frucht, 3),
        (FamousGraph::Tutte, 3),
        (FamousGraph::Chvatal, 4),
        (FamousGraph::Robertson, 4),
        (FamousGraph::Meredith, 4),
        (FamousGraph::Folkman, 4),
        (FamousGraph::Icosahedron, 5),
        (FamousGraph::Octahedron, 4),
        (FamousGraph::Cubical, 3),
        (FamousGraph::Dodecahedron, 3),
        (FamousGraph::Tetrahedron, 3),
    ] {
        let g = Graph::famous(f).unwrap();
        assert!(is_regular(&g, k), "{f} should be {k}-regular");
    }
    // Girths: the cages.
    assert_eq!(girth(&Graph::famous("Petersen").unwrap()), 5);
    assert_eq!(girth(&Graph::famous("Heawood").unwrap()), 6);
    assert_eq!(girth(&Graph::famous("McGee").unwrap()), 7);
    assert_eq!(girth(&Graph::famous("Robertson").unwrap()), 5);
    assert_eq!(girth(&Graph::famous("Levi").unwrap()), 8);
    // Petersen graph: diameter 2.
    assert_eq!(diameter(&Graph::famous("Petersen").unwrap()), 2);
    // Chromatic numbers.
    assert_eq!(chromatic_number(&Graph::famous("Grotzsch").unwrap()), 4);
    assert_eq!(chromatic_number(&Graph::famous("Chvatal").unwrap()), 4);
    assert_eq!(
        chromatic_number(&Graph::famous("Uniquely3colorable").unwrap()),
        3
    );
    assert_eq!(triangle_count(&Graph::famous("Grotzsch").unwrap()), 0);
    assert_eq!(triangle_count(&Graph::famous("Tetrahedron").unwrap()), 4);
    assert_eq!(triangle_count(&Graph::famous("Octahedron").unwrap()), 8);
    // Nonline is the union of the 9 forbidden induced subgraphs of line graphs.
    let nonline = Graph::famous("Nonline").unwrap();
    let components = nonline.connected_components(Connectedness::Weak).unwrap();
    assert_eq!(components.count, 9);
    // One of them is the claw K_{1,3}.
    assert!(
        (0..components.count)
            .map(|c| components.sizes[c])
            .any(|size| size == 4)
    );
}

#[test]
fn famous_graph_automorphism_groups_and_diameters() {
    // Orders of the automorphism groups and diameters, as tabulated in the
    // literature (e.g. Petersen: S_5, Heawood and Coxeter: PGL(2, 7) of
    // order 336, Levi (the Tutte-Coxeter graph): Aut(S_6) of order 1440,
    // Frucht and Walther: asymmetric).
    for (f, aut, diam) in [
        (FamousGraph::Petersen, 120.0, 2),
        (FamousGraph::Heawood, 336.0, 3),
        (FamousGraph::Coxeter, 336.0, 4),
        (FamousGraph::McGee, 32.0, 4),
        (FamousGraph::Levi, 1440.0, 4),
        (FamousGraph::Frucht, 1.0, 4),
        (FamousGraph::Tutte, 3.0, 8),
        (FamousGraph::Walther, 1.0, 8),
        (FamousGraph::Folkman, 3840.0, 4),
        (FamousGraph::Robertson, 24.0, 3),
        (FamousGraph::Chvatal, 8.0, 2),
        (FamousGraph::Grotzsch, 10.0, 2),
        (FamousGraph::Herschel, 12.0, 4),
        (FamousGraph::Franklin, 48.0, 3),
        (FamousGraph::Meredith, 38_698_352_640.0, 8),
        (FamousGraph::Tetrahedron, 24.0, 1),
        (FamousGraph::Cubical, 48.0, 3),
        (FamousGraph::Octahedron, 48.0, 2),
        (FamousGraph::Dodecahedron, 120.0, 5),
        (FamousGraph::Icosahedron, 120.0, 3),
        (FamousGraph::SmallestCyclicGroup, 3.0, 3),
        (FamousGraph::Zachary, 480.0, 5),
    ] {
        let g = Graph::famous(f).unwrap();
        assert_eq!(g.count_automorphisms(None).unwrap(), aut, "{f}");
        assert_eq!(diameter(&g), diam, "{f}");
    }
    // Zachary's karate club: 45 triangles and a largest clique of 5 members.
    let z = Graph::famous(FamousGraph::Zachary).unwrap();
    assert_eq!(triangle_count(&z), 45);
    assert_eq!(z.clique_number().unwrap(), 5);
}

#[test]
fn famous_zachary_matches_the_karate_club() {
    let z = Graph::famous(FamousGraph::Zachary).unwrap();
    assert_eq!(edge_set(&z), edge_set(&karate()));
    assert_eq!(degrees(&z), degrees(&karate()));
    assert_eq!(KARATE_EDGES.len(), z.ecount());
}

#[test]
fn atlas_matches_igraph_example_output() {
    // examples/simple/igraph_atlas.out
    let g45 = Graph::atlas(45).unwrap();
    assert_eq!(g45.vcount(), 5);
    assert_eq!(
        edge_set(&g45),
        BTreeSet::from([(0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])
    );
    let g0 = Graph::atlas(0).unwrap();
    assert_eq!((g0.vcount(), g0.ecount()), (0, 0));
    let k7 = Graph::atlas(1252).unwrap();
    assert_eq!(
        edge_set(&k7),
        edge_set(&Graph::full(7, false, false).unwrap())
    );
    assert_eq!(
        Graph::atlas(1253).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn atlas_is_ordered_and_complete_for_small_orders() {
    // Graphs on n vertices start at 0, 1, 2, 4, 8, 19, 53, 209: there are
    // 1, 1, 2, 4, 11, 34, 156, 1044 unlabelled graphs on 0..=7 vertices.
    let starts = [0usize, 1, 2, 4, 8, 19, 53, 209, 1253];
    for n in 0..8 {
        for i in starts[n]..starts[n + 1] {
            let g = Graph::atlas(i).unwrap();
            assert_eq!(g.vcount(), n, "atlas {i}");
        }
    }
    // Within a fixed order, edge counts are non-decreasing and graphs
    // pairwise non-isomorphic (checked exhaustively up to 6 vertices).
    for n in 0..7 {
        let graphs: Vec<Graph> = (starts[n]..starts[n + 1])
            .map(|i| Graph::atlas(i).unwrap())
            .collect();
        assert!(graphs.windows(2).all(|w| w[0].ecount() <= w[1].ecount()));
        let forms: BTreeSet<_> = graphs.iter().map(canonical).collect();
        assert_eq!(forms.len(), graphs.len(), "order {n}");
        // Every simple graph on n vertices appears: 2^(n choose 2) labelled
        // graphs collapse to exactly these canonical forms.
        let pairs: Vec<(i64, i64)> = (0..n as i64)
            .flat_map(|a| (a + 1..n as i64).map(move |b| (a, b)))
            .collect();
        let all: BTreeSet<_> = (0u32..1 << pairs.len())
            .map(|mask| {
                let e: Vec<_> = pairs
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask >> i & 1 == 1)
                    .map(|(_, &p)| p)
                    .collect();
                canonical(&Graph::from_edges(&e, n, false).unwrap())
            })
            .collect();
        assert_eq!(all, forms, "order {n}");
    }
}

// ---------------------------------------------------- rings, stars, full ---

#[test]
fn rings_paths_and_cycles() {
    // examples/simple/igraph_ring.c shapes.
    let c = Graph::ring(10, false, false, true).unwrap();
    assert!(is_regular(&c, 2));
    assert_eq!(c.ecount(), 10);
    let p = Graph::ring(10, false, false, false).unwrap();
    assert_eq!(p.ecount(), 9);
    assert_eq!(diameter(&p), 9);
    let dm = Graph::ring(10, true, true, true).unwrap();
    assert_eq!(dm.ecount(), 20);
    assert!(out_degrees(&dm).iter().all(|&d| d == 2));

    assert_eq!(Graph::path_graph(10, false, false).unwrap(), p);
    assert_eq!(Graph::cycle_graph(10, false, false).unwrap(), c);
    // Degenerate cycles are not simple.
    let c1 = Graph::cycle_graph(1, false, false).unwrap();
    assert_eq!(c1.edge_list(), vec![(0, 0)]);
    let c2 = Graph::cycle_graph(2, false, false).unwrap();
    assert_eq!(c2.edge_list(), vec![(0, 1), (0, 1)]);
    assert_eq!(Graph::path_graph(0, false, false).unwrap().vcount(), 0);
}

#[test]
fn stars_in_all_modes() {
    let out = Graph::star(6, StarMode::Out, 0).unwrap();
    assert!(out.is_directed());
    assert_eq!(
        out.edge_list(),
        vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    );
    let inward = Graph::star(6, StarMode::In, 0).unwrap();
    assert_eq!(in_degrees(&inward)[0], 5);
    let mutual = Graph::star(6, StarMode::Mutual, 3).unwrap();
    assert_eq!(mutual.ecount(), 10);
    assert_eq!(out_degrees(&mutual)[3], 5);
    assert_eq!(in_degrees(&mutual)[3], 5);
    let und = Graph::star(6, StarMode::Undirected, 5).unwrap();
    assert_eq!(degrees(&und), vec![1, 1, 1, 1, 1, 5]);
    assert_eq!(
        Graph::star(6, StarMode::Out, 6).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn wheels() {
    let w = Graph::wheel(7, WheelMode::Undirected, 0).unwrap();
    assert_eq!(w.ecount(), 12);
    let mut deg = degrees(&w);
    assert_eq!(deg.remove(0), 6);
    assert!(deg.iter().all(|&d| d == 3));
    let m = Graph::wheel(7, WheelMode::Mutual, 2).unwrap();
    assert_eq!(m.ecount(), 24);
    let o = Graph::wheel(5, WheelMode::Out, 0).unwrap();
    assert_eq!(in_degrees(&o)[0], 0);
    assert_eq!(out_degrees(&o)[0], 4);
    // tests/unit/igraph_wheel.out (which lists the edges sorted).
    let sorted_edges = |n, mode, center| {
        let mut e = Graph::wheel(n, mode, center).unwrap().edge_list();
        e.sort();
        e
    };
    assert_eq!(
        sorted_edges(2, WheelMode::Undirected, 0),
        vec![(0, 1), (1, 1)]
    );
    assert_eq!(
        sorted_edges(4, WheelMode::Out, 0),
        vec![(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 1)]
    );
    assert_eq!(
        sorted_edges(4, WheelMode::In, 0),
        vec![(1, 0), (1, 2), (2, 0), (2, 3), (3, 0), (3, 1)]
    );
    assert_eq!(
        sorted_edges(4, WheelMode::Out, 2),
        vec![(0, 1), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0)]
    );
    // W_4 (undirected, 4 vertices) is K_4.
    assert!(
        Graph::wheel(4, WheelMode::Undirected, 0)
            .unwrap()
            .is_complete()
            .unwrap()
    );
    // Removing the hub leaves the rim, a cycle.
    let mut rim = Graph::wheel(8, WheelMode::Undirected, 0).unwrap();
    rim.delete_vertices(0).unwrap();
    assert!(is_regular(&rim, 2) && is_connected(&rim));
}

#[test]
fn full_graphs() {
    for n in 0..8usize {
        let k = Graph::full(n, false, false).unwrap();
        assert_eq!(k.ecount(), n * n.saturating_sub(1) / 2);
        assert_eq!(
            Graph::full(n, true, false).unwrap().ecount(),
            n * n.saturating_sub(1)
        );
        assert_eq!(
            Graph::full(n, false, true).unwrap().ecount(),
            n * (n + 1) / 2
        );
        assert_eq!(Graph::full(n, true, true).unwrap().ecount(), n * n);
    }
    for n in 1..8usize {
        let k = Graph::full(n, false, false).unwrap();
        assert!(k.is_complete().unwrap());
        // The complement of K_n has no edges; the complement of the empty
        // graph is K_n.
        assert_eq!(k.complementer(false).unwrap().ecount(), 0);
        assert!(
            Graph::empty(n, false)
                .unwrap()
                .complementer(false)
                .unwrap()
                .is_same_graph(&k)
                .unwrap()
        );
    }
    let fc = Graph::full_citation(6, true).unwrap();
    assert!(fc.is_dag().unwrap());
    assert_eq!(fc.ecount(), 15);
    assert!(fc.edge_list().iter().all(|&(a, b)| b < a));
    assert_eq!(
        edge_set(&Graph::full_citation(6, false).unwrap()),
        edge_set(&Graph::full(6, false, false).unwrap())
    );
}

#[test]
fn full_multipartite_and_turan() {
    // tests/unit/igraph_full_multipartite: directed, 3 partitions [2, 3, 3].
    let (g, types) = Graph::full_multipartite(&[2, 3, 3], true, NeighborMode::All).unwrap();
    assert_eq!(types, vec![0, 0, 1, 1, 1, 2, 2, 2]);
    assert_eq!(g.ecount(), 2 * (2 * 3 + 2 * 3 + 3 * 3));
    assert!(
        g.edge_list()
            .iter()
            .all(|&(a, b)| types[a as usize] != types[b as usize])
    );
    let (out, _) = Graph::full_multipartite(&[2, 3], true, NeighborMode::Out).unwrap();
    assert!(out.edge_list().iter().all(|&(a, b)| a < 2 && b >= 2));
    let (inw, _) = Graph::full_multipartite(&[2, 3], true, NeighborMode::In).unwrap();
    assert!(inw.edge_list().iter().all(|&(a, b)| a >= 2 && b < 2));
    // Two partitions: the complete bipartite graph of the bipartite module.
    let (k34, types) = Graph::full_multipartite(&[3, 4], false, NeighborMode::All).unwrap();
    let bip = Graph::full_bipartite(3, 4, false, NeighborMode::All).unwrap();
    assert_eq!(edge_set(&k34), edge_set(&bip.graph));
    assert_eq!(types.iter().map(|&t| t == 1).collect::<Vec<_>>(), bip.types);

    // tests/unit/igraph_turan.out
    let (t, types) = Graph::turan(10, 1).unwrap();
    assert_eq!((t.vcount(), t.ecount()), (10, 0));
    assert_eq!(types, vec![0; 10]);
    let (t, types) = Graph::turan(4, 6).unwrap();
    assert_eq!(
        edge_set(&t),
        edge_set(&Graph::full(4, false, false).unwrap())
    );
    assert_eq!(types, vec![0, 1, 2, 3]);
    let (t, _) = Graph::turan(0, 3).unwrap();
    assert_eq!(t.vcount(), 0);
    assert_eq!(
        Graph::turan(5, 0).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );

    // Turán's theorem: T(n, r) has floor((1 - 1/r) n^2 / 2) edges and no K_{r+1}.
    for n in 1..12usize {
        for r in 1..=n {
            let (t, types) = Graph::turan(n, r).unwrap();
            let expected = ((r - 1) * n * n) / (2 * r);
            assert_eq!(t.ecount(), expected, "T({n}, {r})");
            // Partition sizes differ by at most one.
            let sizes: Vec<usize> = (0..r as i64)
                .map(|p| types.iter().filter(|&&x| x == p).count())
                .collect();
            assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1);
            // ... and its largest clique has exactly r vertices.
            assert_eq!(t.clique_number().unwrap(), r, "T({n}, {r})");
        }
    }
    // The octahedron is T(6, 3).
    assert_eq!(
        canonical(&Graph::turan(6, 3).unwrap().0),
        canonical(&Graph::famous("Octahedron").unwrap())
    );
}

// ------------------------------------------------------------- lattices ---

#[test]
fn square_lattices_and_hypercubes() {
    let grid = Graph::square_lattice(&[5, 4], 1, false, false, None).unwrap();
    assert_eq!((grid.vcount(), grid.ecount()), (20, 5 * 3 + 4 * 4));
    // Vertex (i, j) has id i + 5 j: (1, 1) = 6 neighbors (0,1),(2,1),(1,0),(1,2).
    assert_eq!(
        grid.neighbors(6, NeighborMode::All).unwrap(),
        vec![1, 5, 7, 11]
    );
    assert_eq!(diameter(&grid), 4 + 3);
    let torus = Graph::square_lattice(&[5, 4], 1, false, false, Some(&[true, true])).unwrap();
    assert!(is_regular(&torus, 4));
    let cyl = Graph::square_lattice(&[5, 4], 1, false, false, Some(&[true, false])).unwrap();
    assert_eq!(cyl.ecount(), 5 * 4 + 5 * 3);
    // nei = 2 adds the vertices at distance 2 along the lattice.
    let p2 = Graph::square_lattice(&[6], 2, false, false, None).unwrap();
    assert_eq!(p2.ecount(), 5 + 4);
    // Directed: from lower to higher ids.
    let d = Graph::square_lattice(&[3, 3], 1, true, false, None).unwrap();
    assert!(d.edge_list().iter().all(|&(a, b)| a < b));
    let dm = Graph::square_lattice(&[3, 3], 1, true, true, None).unwrap();
    assert_eq!(dm.ecount(), 24);
    // The zero-dimensional lattice is the singleton graph.
    assert_eq!(
        Graph::square_lattice(&[], 1, false, false, None)
            .unwrap()
            .vcount(),
        1
    );
    // Mismatched periodicity.
    let err = Graph::square_lattice(&[3, 3], 1, false, false, Some(&[true])).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);

    // Q_n = the 2 x 2 x ... x 2 lattice.
    for n in 0..7usize {
        let q = Graph::hypercube(n, false).unwrap();
        assert_eq!(q.vcount(), 1 << n);
        assert_eq!(q.ecount(), n * (1 << n) / 2);
        assert!(is_regular(&q, n as i64));
        let lattice = Graph::square_lattice(&vec![2; n], 1, false, false, None).unwrap();
        assert_eq!(edge_set(&q), edge_set(&lattice));
        assert!(
            q.edge_list()
                .iter()
                .all(|&(a, b)| (a ^ b).count_ones() == 1)
        );
    }
    assert_eq!(
        canonical(&Graph::hypercube(3, false).unwrap()),
        canonical(&Graph::famous("Cubical").unwrap())
    );
}

#[test]
fn triangular_and_hexagonal_lattices() {
    // Values from tests/unit/igraph_{triangular,hexagonal}_lattice.out
    assert_eq!(
        Graph::triangular_lattice(&[3, 0], true, false)
            .unwrap()
            .vcount(),
        0
    );
    assert_eq!(
        Graph::triangular_lattice(&[1], true, false)
            .unwrap()
            .vcount(),
        1
    );
    let t = Graph::triangular_lattice(&[5], true, false).unwrap();
    assert_eq!((t.vcount(), t.ecount()), (15, 30));
    assert!(degrees(&t).iter().all(|&d| d <= 6));
    // Every edge of a triangular lattice lies on a triangle: 1 + 2 + 3 + 4 up
    // triangles plus 1 + 2 + 3 down triangles.
    let tu = Graph::triangular_lattice(&[5], false, false).unwrap();
    assert_eq!(triangle_count(&tu), 10 + 6);

    let expected = [
        (vec![1usize], 6, 6),
        (vec![5], 46, 60),
        (vec![3, 4, 5], 94, 129),
    ];
    for (dims, n, m) in expected {
        let h = Graph::hexagonal_lattice(&dims, false, false).unwrap();
        assert_eq!((h.vcount(), h.ecount()), (n, m), "{dims:?}");
        assert!(degrees(&h).iter().all(|&d| d <= 3));
        assert_eq!(triangle_count(&h), 0);
        // Duality: the triangular lattice with the same dims has one vertex
        // per hexagon; by Euler's formula, faces = E - V + 2 (incl. outer face).
        let t = Graph::triangular_lattice(&dims, false, false).unwrap();
        assert_eq!(t.vcount(), m - n + 1, "{dims:?}");
    }
    let h = Graph::hexagonal_lattice(&[4, 5], true, true).unwrap();
    assert_eq!((h.vcount(), h.ecount()), (58, 154));
    assert_eq!(
        Graph::hexagonal_lattice(&[3, 4, 5, 5], false, false)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        Graph::triangular_lattice(&[], false, false)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

// ---------------------------------------------------------------- trees ---

#[test]
fn kary_symmetric_and_regular_trees() {
    let t = Graph::kary_tree(15, 2, TreeMode::Out).unwrap();
    assert!(is_tree(&t));
    assert_eq!(in_degrees(&t)[0], 0);
    assert!(in_degrees(&t)[1..].iter().all(|&d| d == 1));
    assert_eq!(out_degrees(&t).iter().filter(|&&d| d == 0).count(), 8); // leaves
    let tin = Graph::kary_tree(15, 2, TreeMode::In).unwrap();
    assert_eq!(out_degrees(&tin)[0], 0);
    assert!(
        !Graph::kary_tree(15, 2, TreeMode::Undirected)
            .unwrap()
            .is_directed()
    );
    assert_eq!(Graph::kary_tree(0, 3, TreeMode::Out).unwrap().vcount(), 0);
    // Parent of i is (i - 1) / k.
    let t3 = Graph::kary_tree(40, 3, TreeMode::Out).unwrap();
    assert!(t3.edge_list().iter().all(|&(p, c)| p == (c - 1) / 3));

    // tests/unit/symmetric_tree: sizes are products of branching counts.
    let s = Graph::symmetric_tree(&[3, 4, 2], TreeMode::Out).unwrap();
    assert_eq!(s.vcount(), 1 + 3 + 12 + 24);
    assert!(is_tree(&s));
    let out = out_degrees(&s);
    assert_eq!(out[0], 3);
    assert!(out[1..4].iter().all(|&d| d == 4));
    assert!(out[4..16].iter().all(|&d| d == 2));
    assert!(out[16..].iter().all(|&d| d == 0));
    assert_eq!(
        Graph::symmetric_tree(&[], TreeMode::Undirected)
            .unwrap()
            .vcount(),
        1
    );
    // A symmetric tree with constant branching is a complete k-ary tree.
    assert_eq!(
        Graph::symmetric_tree(&[2, 2, 2], TreeMode::Out).unwrap(),
        Graph::kary_tree(15, 2, TreeMode::Out).unwrap()
    );

    // Regular tree (Bethe lattice): all internal vertices have degree k.
    let r = Graph::regular_tree(3, 4, TreeMode::Undirected).unwrap();
    assert!(is_tree(&r));
    assert_eq!(r.vcount(), 1 + 4 + 12 + 36);
    assert!(degrees(&r).iter().all(|&d| d == 4 || d == 1));
    // The leaves are at distance h = 3 from the root.
    assert_eq!(r.bfs_simple(0, NeighborMode::All).unwrap().num_layers(), 4);
    assert_eq!(diameter(&r), 6);
}

#[test]
fn trees_from_parent_vectors() {
    // A forest: two trees and an isolated root.
    let parents = [-1, 0, 0, 1, 1, -1, 5, -1];
    let f = Graph::tree_from_parent_vector(&parents, TreeMode::Out).unwrap();
    assert_eq!((f.vcount(), f.ecount()), (8, 5));
    let mut e = f.edge_list();
    e.sort();
    assert_eq!(e, vec![(0, 1), (0, 2), (1, 3), (1, 4), (5, 6)]);
    let fin = Graph::tree_from_parent_vector(&parents, TreeMode::In).unwrap();
    assert!(
        fin.edge_list()
            .iter()
            .all(|&(c, p)| parents[c as usize] == p)
    );
    // Round trip with the k-ary tree parent formula.
    let parents: Vec<i64> = (0..31)
        .map(|i| if i == 0 { -1 } else { (i - 1) / 2 })
        .collect();
    let t = Graph::tree_from_parent_vector(&parents, TreeMode::Out).unwrap();
    assert_eq!(
        edge_set(&t),
        edge_set(&Graph::kary_tree(31, 2, TreeMode::Out).unwrap())
    );
    // Cycles and self-loops are rejected.
    assert!(Graph::tree_from_parent_vector(&[1, 2, 0], TreeMode::Out).is_err());
    assert!(Graph::tree_from_parent_vector(&[0], TreeMode::Out).is_err());
    assert!(Graph::tree_from_parent_vector(&[-1, 7], TreeMode::Out).is_err());
}

#[test]
fn prufer_sequences() {
    // tests/unit/igraph_from_prufer.out
    let t = Graph::from_prufer(&[2, 3, 2, 3]).unwrap();
    assert_eq!(t.vcount(), 6);
    assert_eq!(
        edge_set(&t),
        BTreeSet::from([(0, 2), (1, 3), (2, 4), (2, 3), (3, 5)])
    );
    let t = Graph::from_prufer(&[0, 2, 4, 1, 1, 0]).unwrap();
    assert_eq!(
        edge_set(&t),
        BTreeSet::from([(0, 3), (2, 5), (2, 4), (1, 4), (1, 6), (0, 1), (0, 7)])
    );
    let t = Graph::from_prufer(&[]).unwrap();
    assert_eq!(t.edge_list(), vec![(0, 1)]);
    // A star's code is its center repeated.
    let star = Graph::star(7, StarMode::Undirected, 4).unwrap();
    assert_eq!(star.to_prufer().unwrap(), vec![4; 5]);
    assert_eq!(
        Graph::from_prufer(&[5, 0]).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );

    // Cayley's formula: the 5^3 = 125 sequences of length 3 give 125
    // distinct labelled trees on 5 vertices, where degree = occurrences + 1.
    let mut trees = BTreeSet::new();
    for a in 0..5 {
        for b in 0..5 {
            for c in 0..5 {
                let seq = [a, b, c];
                let t = Graph::from_prufer(&seq).unwrap();
                assert!(is_tree(&t));
                for v in 0..5i64 {
                    let occ = seq.iter().filter(|&&x| x == v).count() as i64;
                    assert_eq!(degrees(&t)[v as usize], occ + 1);
                }
                // to_prufer is the inverse bijection.
                assert_eq!(t.to_prufer().unwrap(), seq);
                trees.insert(edge_set(&t));
            }
        }
    }
    assert_eq!(trees.len(), 125);
}

// ---------------------------------------------------- circulant & friends ---

#[test]
fn circulants_petersen_and_lcf() {
    let c = Graph::circulant(10, &[1, 3], false).unwrap();
    assert!(is_regular(&c, 4));
    assert_eq!(c.ecount(), 20);
    // Shift n/2 gives a perfect matching (no duplicates), shift 0 nothing.
    let m = Graph::circulant(8, &[4, 0, 8], false).unwrap();
    assert_eq!(m.ecount(), 4);
    let d = Graph::circulant(7, &[1, -1], true).unwrap();
    assert_eq!(d.ecount(), 14);
    assert!(out_degrees(&d).iter().all(|&x| x == 2));
    assert_eq!(
        edge_set(&Graph::circulant(9, &[1], false).unwrap()),
        edge_set(&Graph::cycle_graph(9, false, false).unwrap())
    );

    // Generalized Petersen graphs are cubic with 2n vertices and 3n edges.
    for n in 3..12usize {
        for k in 1..n.div_ceil(2) {
            let g = Graph::generalized_petersen(n, k).unwrap();
            assert_eq!((g.vcount(), g.ecount()), (2 * n, 3 * n), "G({n}, {k})");
            assert!(is_regular(&g, 3));
        }
    }
    let p = Graph::generalized_petersen(5, 2).unwrap();
    assert_eq!(girth(&p), 5);
    assert_eq!(diameter(&p), 2);
    assert!(isomorphic(&p, &Graph::famous("Petersen").unwrap()));
    assert!(isomorphic(
        &Graph::generalized_petersen(4, 1).unwrap(),
        &Graph::famous("Cubical").unwrap()
    ));
    // G(10, 3) is the Desargues graph: bipartite, girth 6, 240 automorphisms.
    let desargues = Graph::generalized_petersen(10, 3).unwrap();
    assert!(desargues.is_bipartite().unwrap());
    assert_eq!(girth(&desargues), 6);
    assert_eq!(desargues.count_automorphisms(None).unwrap(), 240.0);
    assert!(Graph::generalized_petersen(2, 1).is_err());
    assert!(Graph::generalized_petersen(6, 3).is_err());
    assert!(Graph::generalized_petersen(6, 0).is_err());

    // examples/simple/igraph_lcf.out: the Heawood graph [5, -5]^7.
    let h = Graph::lcf(14, &[5, -5], 7).unwrap();
    let expected: Vec<i64> = vec![
        0, 1, 0, 5, 0, 13, 1, 2, 1, 10, 2, 3, 2, 7, 3, 4, 3, 12, 4, 5, 4, 9, 5, 6, 6, 7, 6, 11, 7,
        8, 8, 9, 8, 13, 9, 10, 10, 11, 11, 12, 12, 13,
    ];
    let pairs: BTreeSet<_> = expected.chunks(2).map(|p| (p[0], p[1])).collect();
    assert_eq!(edge_set(&h), pairs);
    assert_eq!(girth(&h), 6);
    // Other LCF codes: the cube [3, -3]^4, the Möbius–Kantor graph [5, -5]^8.
    assert!(isomorphic(
        &Graph::lcf(8, &[3, -3], 4).unwrap(),
        &Graph::famous("Cubical").unwrap()
    ));
    assert!(isomorphic(&h, &Graph::famous("Heawood").unwrap()));
    let mk = Graph::lcf(16, &[5, -5], 8).unwrap();
    assert!(is_regular(&mk, 3) && girth(&mk) == 6);
    // The Möbius–Kantor graph is the generalized Petersen graph G(8, 3).
    assert!(isomorphic(&mk, &Graph::generalized_petersen(8, 3).unwrap()));
    assert_eq!(mk.count_automorphisms(None).unwrap(), 96.0);
    // Frucht graph: [-5, -2, -4, 2, 5, -2, 2, 5, -2, -5, 4, 2].
    let frucht = Graph::lcf(12, &[-5, -2, -4, 2, 5, -2, 2, 5, -2, -5, 4, 2], 1).unwrap();
    assert!(is_regular(&frucht, 3));
    assert!(isomorphic(&frucht, &Graph::famous("Frucht").unwrap()));

    // Extended chordal ring: [[3]] on 6 vertices is C6 plus doubled diameters.
    let r = Graph::extended_chordal_ring(6, &[[3]], false).unwrap();
    assert_eq!(r.ecount(), 12);
    let mut simple = edge_set(&r);
    assert_eq!(simple.len(), 9);
    simple.retain(|&(a, b)| (b - a) % 3 == 0);
    assert_eq!(simple.len(), 3);
    // Rows of length 2: alternating chords.
    let r = Graph::extended_chordal_ring(8, &[[2, -2]], false).unwrap();
    assert_eq!(r.ecount(), 16);
    assert!(Graph::extended_chordal_ring(7, &[[2, 3]], false).is_err());
    assert_eq!(
        Graph::extended_chordal_ring(5, &[] as &[[i64; 1]], false)
            .unwrap()
            .ecount(),
        5
    );
}

// ------------------------------------------------------- word graphs ---

#[test]
fn de_bruijn_and_kautz() {
    for m in 1..4usize {
        for n in 1..4u32 {
            let b = Graph::de_bruijn(m, n as usize).unwrap();
            assert_eq!(b.vcount(), m.pow(n));
            assert_eq!(b.ecount(), m.pow(n + 1));
            // Each vertex has in- and out-degree m.
            assert!(out_degrees(&b).iter().all(|&d| d == m as i64));
            assert!(in_degrees(&b).iter().all(|&d| d == m as i64));
            // v -> (v * m + a) mod m^n.
            let size = m.pow(n) as i64;
            for (v, w) in b.edge_list() {
                assert!((0..m as i64).any(|a| (v * m as i64 + a) % size == w));
            }
        }
    }
    // Kautz graph K(m, n): (m+1) m^n vertices, regular of degree m, no loops.
    for m in 1..4usize {
        for n in 0..4u32 {
            let k = Graph::kautz(m, n as usize).unwrap();
            assert_eq!(k.vcount(), (m + 1) * m.pow(n));
            assert_eq!(k.ecount(), (m + 1) * m.pow(n + 1));
            assert!(out_degrees(&k).iter().all(|&d| d == m as i64));
            assert!(k.edge_list().iter().all(|&(a, b)| a != b));
        }
    }
    // Degenerate cases: strings of length 0 give a single vertex without edges.
    let b = Graph::de_bruijn(3, 0).unwrap();
    assert_eq!((b.vcount(), b.ecount()), (1, 0));
    // K(m, 0) is the complete directed graph on m + 1 vertices.
    assert_eq!(
        Graph::kautz(3, 0).unwrap().ecount(),
        Graph::full(4, true, false).unwrap().ecount()
    );
}

#[test]
fn de_bruijn_sequence_from_an_eulerian_circuit() {
    // Use case: build a de Bruijn sequence B(2, 4) (every 4-bit word appears
    // exactly once as a cyclic substring) by walking an Eulerian circuit of
    // the de Bruijn graph on 3-bit words.
    let (k, n) = (2usize, 3usize);
    let g = Graph::de_bruijn(k, n).unwrap();
    let status = g.is_eulerian().unwrap();
    assert!(status.has_cycle && status.has_path);
    let circuit = g.eulerian_cycle().unwrap();
    assert_eq!(circuit.edges.len(), g.ecount());
    assert_eq!(circuit.vertices.len(), g.ecount() + 1);
    assert_eq!(circuit.vertices.first(), circuit.vertices.last());
    // Consecutive vertices of the walk are joined by the walked edges.
    for (i, &e) in circuit.edges.iter().enumerate() {
        assert_eq!(
            g.edge(e).unwrap(),
            (circuit.vertices[i], circuit.vertices[i + 1])
        );
    }
    // Each step appends the last letter of the next word.
    let seq: Vec<i64> = circuit.vertices[1..].iter().map(|w| w % k as i64).collect();
    assert_eq!(seq.len(), 16);
    let words: BTreeSet<i64> = (0..seq.len())
        .map(|i| (0..=n).fold(0, |acc, j| acc * 2 + seq[(i + j) % seq.len()]))
        .collect();
    assert_eq!(words.len(), 16, "every 4-bit word appears once");
}

// ------------------------------------------------------- degree sequences ---

#[test]
fn realize_degree_sequence_methods() {
    let karate = karate();
    let degs = degrees(&karate);
    for method in [
        RealizeDegseq::Smallest,
        RealizeDegseq::Largest,
        RealizeDegseq::Index,
    ] {
        let g =
            Graph::realize_degree_sequence(&degs, None, AllowedEdgeTypes::SIMPLE, method).unwrap();
        assert_eq!(degrees(&g), degs, "{method:?}");
        assert!(!has_loops_or_multi(&g));
        assert!(!g.is_directed());
    }
    // Smallest-first yields a connected graph whenever possible, so a tree's
    // degree sequence gives back a tree.
    let tree_degs = [1, 1, 1, 1, 2, 2, 3, 3];
    let t = Graph::realize_degree_sequence(
        &tree_degs,
        None,
        AllowedEdgeTypes::SIMPLE,
        RealizeDegseq::Smallest,
    )
    .unwrap();
    assert!(is_tree(&t));
    // Largest-first on the same sequence, on the contrary, tends to split it:
    // the two degree-3 vertices get linked to each other first.
    let l = Graph::realize_degree_sequence(
        &tree_degs,
        None,
        AllowedEdgeTypes::SIMPLE,
        RealizeDegseq::Largest,
    )
    .unwrap();
    assert_eq!(degrees(&l), tree_degs);

    // tests/unit/igraph_realize_degree_sequence.out, sequence (1 2 2 3).
    let seq = [1, 2, 2, 3];
    let realize = |allowed: AllowedEdgeTypes, method| {
        Graph::realize_degree_sequence(&seq, None, allowed, method)
            .unwrap()
            .edge_list()
    };
    let (sl, ss, si) = (
        vec![(3, 0), (3, 2), (3, 1), (2, 1)],
        vec![(3, 0), (3, 2), (3, 1), (2, 1)],
        vec![(3, 0), (3, 1), (2, 1), (3, 2)],
    );
    let normalize = |e: Vec<(i64, i64)>| -> Vec<(i64, i64)> {
        e.into_iter().map(|(a, b)| (a.max(b), a.min(b))).collect()
    };
    use AllowedEdgeTypes as A;
    assert_eq!(normalize(realize(A::SIMPLE, RealizeDegseq::Largest)), sl);
    assert_eq!(normalize(realize(A::SIMPLE, RealizeDegseq::Smallest)), ss);
    assert_eq!(normalize(realize(A::SIMPLE, RealizeDegseq::Index)), si);
    let ml = vec![(3, 1), (3, 2), (1, 0), (3, 2)];
    for allowed in [A::MULTI, A::ALL] {
        assert_eq!(normalize(realize(allowed, RealizeDegseq::Largest)), ml);
        assert_eq!(normalize(realize(allowed, RealizeDegseq::Smallest)), si);
        assert_eq!(normalize(realize(allowed, RealizeDegseq::Index)), si);
    }

    // Directed: out- and in-degrees are both matched.
    let out = [2, 1, 1, 0];
    let ind = [0, 1, 1, 2];
    let d = Graph::realize_degree_sequence(
        &out,
        Some(&ind),
        EdgeTypeSw::Simple,
        RealizeDegseq::Smallest,
    )
    .unwrap();
    assert!(d.is_directed());
    assert_eq!(out_degrees(&d), out);
    assert_eq!(in_degrees(&d), ind);

    // Multigraphs, with and without loops.
    let m = Graph::realize_degree_sequence(
        &[4, 2],
        None,
        AllowedEdgeTypes::MULTI,
        RealizeDegseq::Smallest,
    );
    assert!(m.is_err(), "no loopless multigraph has degrees [4, 2]");
    let ml = Graph::realize_degree_sequence(
        &[4, 2],
        None,
        AllowedEdgeTypes::ALL,
        RealizeDegseq::Smallest,
    )
    .unwrap();
    assert_eq!(degrees(&ml), vec![4, 2]);
    assert!(has_loops_or_multi(&ml));

    // Non-graphical input and mismatched directed sequences.
    let err = Graph::realize_degree_sequence(
        &[3, 3, 1, 1],
        None,
        AllowedEdgeTypes::SIMPLE,
        RealizeDegseq::Smallest,
    )
    .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    assert!(
        Graph::realize_degree_sequence(
            &[1, 2],
            None,
            AllowedEdgeTypes::SIMPLE,
            RealizeDegseq::Index
        )
        .is_err()
    );
    assert!(
        Graph::realize_degree_sequence(
            &[1, 0],
            Some(&[0, 0]),
            AllowedEdgeTypes::SIMPLE,
            RealizeDegseq::Index
        )
        .is_err()
    );
    assert!(
        Graph::realize_degree_sequence(
            &[1, 0],
            Some(&[1]),
            AllowedEdgeTypes::SIMPLE,
            RealizeDegseq::Index
        )
        .is_err()
    );
}

#[test]
fn realizability_agrees_with_the_graphicality_tests() {
    // Every undirected sequence of length <= 4 with entries <= 4: a
    // realization exists exactly when mixing::is_graphical says so, and then
    // it has the requested degrees and kind of edges.
    use igraph::mixing::{is_bigraphical, is_graphical};
    let mut sequences = vec![vec![]];
    for _ in 0..4 {
        let longer: Vec<Vec<i64>> = sequences
            .iter()
            .filter(|s: &&Vec<i64>| s.len() == sequences.last().unwrap().len())
            .flat_map(|s| {
                (0..=4).map(move |d| {
                    let mut t = s.clone();
                    t.push(d);
                    t
                })
            })
            .collect();
        sequences.extend(longer);
    }
    let mut realized = 0;
    for seq in &sequences {
        for allowed in [
            AllowedEdgeTypes::SIMPLE,
            AllowedEdgeTypes::MULTI,
            AllowedEdgeTypes::ALL,
        ] {
            let graphical = is_graphical(seq, None, allowed).unwrap();
            let g = Graph::realize_degree_sequence(seq, None, allowed, RealizeDegseq::Smallest);
            assert_eq!(g.is_ok(), graphical, "{seq:?} {allowed:?}");
            if let Ok(g) = g {
                realized += 1;
                assert_eq!(degrees(&g), *seq, "{seq:?} {allowed:?}");
                if allowed == AllowedEdgeTypes::SIMPLE {
                    assert!(g.is_simple(true).unwrap());
                }
                if allowed == AllowedEdgeTypes::MULTI {
                    assert!(!g.has_loop().unwrap());
                }
            }
        }
    }
    assert!(realized > 100);
    // Bipartite sequences against is_bigraphical.
    for d1 in [
        vec![2, 2],
        vec![3, 1],
        vec![1, 1, 2],
        vec![4],
        vec![2, 2, 2],
    ] {
        for d2 in [
            vec![2, 2],
            vec![1, 3],
            vec![1, 1, 1, 1],
            vec![4],
            vec![3, 3],
        ] {
            for allowed in [AllowedEdgeTypes::SIMPLE, AllowedEdgeTypes::MULTI] {
                let ok = is_bigraphical(&d1, &d2, allowed).unwrap();
                let g = Graph::realize_bipartite_degree_sequence(
                    &d1,
                    &d2,
                    allowed,
                    RealizeDegseq::Smallest,
                );
                assert_eq!(g.is_ok(), ok, "{d1:?} {d2:?} {allowed:?}");
            }
        }
    }
}

#[test]
fn realize_bipartite_degree_sequences() {
    // tests/unit/igraph_realize_bipartite_degree_sequence.out
    let sorted = |g: &Graph| -> Vec<(i64, i64)> {
        let mut e: Vec<_> = g
            .edge_list()
            .into_iter()
            .map(|(a, b)| (a.min(b), a.max(b)))
            .collect();
        e.sort();
        e
    };
    let g = Graph::realize_bipartite_degree_sequence(
        &[2, 3, 2, 1],
        &[3, 1, 2, 1, 1],
        AllowedEdgeTypes::SIMPLE,
        RealizeDegseq::Smallest,
    )
    .unwrap();
    assert_eq!(
        sorted(&g),
        vec![
            (0, 5),
            (0, 6),
            (1, 4),
            (1, 7),
            (1, 8),
            (2, 4),
            (2, 6),
            (3, 4)
        ]
    );
    assert!(g.is_simple(true).unwrap() && is_connected(&g) && g.is_bipartite().unwrap());
    let m = Graph::realize_bipartite_degree_sequence(
        &[2, 3, 1],
        &[4, 2],
        AllowedEdgeTypes::MULTI,
        RealizeDegseq::Smallest,
    )
    .unwrap();
    assert_eq!(
        sorted(&m),
        vec![(0, 3), (0, 3), (1, 3), (1, 4), (1, 4), (2, 3)]
    );
    assert!(m.has_multiple().unwrap() && is_connected(&m));

    let d1 = [3, 2, 2, 1];
    let d2 = [2, 2, 2, 1, 1];
    for method in [
        RealizeDegseq::Smallest,
        RealizeDegseq::Largest,
        RealizeDegseq::Index,
    ] {
        let g =
            Graph::realize_bipartite_degree_sequence(&d1, &d2, AllowedEdgeTypes::SIMPLE, method)
                .unwrap();
        assert_eq!(degrees(&g), [&d1[..], &d2[..]].concat());
        // Every edge crosses the bipartition.
        assert!(g.edge_list().iter().all(|&(a, b)| (a < 4) != (b < 4)));
        assert!(!has_loops_or_multi(&g));
    }
    // Smallest-first gives a connected bipartite graph (8 edges, 9 vertices: a tree).
    let g = Graph::realize_bipartite_degree_sequence(
        &d1,
        &d2,
        AllowedEdgeTypes::SIMPLE,
        RealizeDegseq::Smallest,
    )
    .unwrap();
    assert!(is_tree(&g));
    // [3] vs [1, 1] is only realizable with multi-edges... or not at all (sums differ).
    assert!(
        Graph::realize_bipartite_degree_sequence(
            &[3],
            &[1, 1],
            AllowedEdgeTypes::MULTI,
            RealizeDegseq::Smallest
        )
        .is_err()
    );
    assert!(
        Graph::realize_bipartite_degree_sequence(
            &[2],
            &[1],
            AllowedEdgeTypes::SIMPLE,
            RealizeDegseq::Smallest
        )
        .is_err()
    );
    let m = Graph::realize_bipartite_degree_sequence(
        &[2],
        &[2],
        AllowedEdgeTypes::MULTI,
        RealizeDegseq::Smallest,
    )
    .unwrap();
    assert_eq!(m.edge_list(), vec![(0, 1), (0, 1)]);
    // Bipartite graphs have no self-loops: igraph ignores the loops flag.
    for (allowed, same_as) in [
        (AllowedEdgeTypes::LOOPS, AllowedEdgeTypes::SIMPLE),
        (AllowedEdgeTypes::ALL, AllowedEdgeTypes::MULTI),
    ] {
        let realize = |allowed| {
            Graph::realize_bipartite_degree_sequence(&[2], &[2], allowed, RealizeDegseq::Smallest)
                .map(|g| g.edge_list())
                .map_err(|e| e.kind())
        };
        assert_eq!(realize(allowed), realize(same_as), "{allowed:?}");
    }
    assert!(
        Graph::realize_bipartite_degree_sequence(
            &[2],
            &[2],
            AllowedEdgeTypes::SIMPLE,
            RealizeDegseq::Smallest
        )
        .is_err()
    );
}

// ------------------------------------------------------------- matrices ---

#[test]
fn adjacency_modes() {
    let a = Matrix::from_rows(&[[0.0, 2.0, 1.0], [0.0, 0.0, 1.0], [3.0, 1.0, 2.0]]).unwrap();
    let dir = Graph::adjacency(&a, Adjacency::Directed, Loops::Once).unwrap();
    assert!(dir.is_directed());
    assert_eq!(dir.ecount(), 2 + 1 + 1 + 3 + 1 + 2);
    assert_eq!(dir.get_all_eids_between(0, 1, true).unwrap().len(), 2);
    let count = |g: &Graph, a, b| g.get_all_eids_between(a, b, false).unwrap().len();
    let max = Graph::adjacency(&a, Adjacency::Max, Loops::Once).unwrap();
    assert_eq!(
        (
            count(&max, 0, 1),
            count(&max, 0, 2),
            count(&max, 1, 2),
            count(&max, 2, 2)
        ),
        (2, 3, 1, 2)
    );
    let min = Graph::adjacency(&a, Adjacency::Min, Loops::None).unwrap();
    assert_eq!(
        (count(&min, 0, 1), count(&min, 0, 2), count(&min, 1, 2)),
        (0, 1, 1)
    );
    assert_eq!(min.ecount(), 2);
    let plus = Graph::adjacency(&a, Adjacency::Plus, Loops::None).unwrap();
    assert_eq!(
        (count(&plus, 0, 1), count(&plus, 0, 2), count(&plus, 1, 2)),
        (2, 4, 2)
    );
    let upper = Graph::adjacency(&a, Adjacency::Upper, Loops::None).unwrap();
    assert_eq!(upper.ecount(), 2 + 1 + 1);
    let upper = Graph::adjacency(&a, Adjacency::Upper, Loops::Once).unwrap();
    assert_eq!(upper.ecount(), 2 + 1 + 1 + 2);
    let lower = Graph::adjacency(&a, Adjacency::Lower, Loops::Once).unwrap();
    assert_eq!(lower.ecount(), 3 + 1 + 2);
    // igraph 1.0.x honours `loops` in the directed mode too.
    let dir = Graph::adjacency(&a, Adjacency::Directed, Loops::None).unwrap();
    assert_eq!(dir.ecount(), 2 + 1 + 1 + 3 + 1);
    // ... where `Twice` is treated as `Once` (a directed loop adds 1 to each degree).
    let dir = Graph::adjacency(&a, Adjacency::Directed, Loops::Twice).unwrap();
    assert_eq!(dir.ecount(), 2 + 1 + 1 + 3 + 1 + 2);
    // Undirected requires symmetry.
    assert_eq!(
        Graph::adjacency(&a, Adjacency::Undirected, Loops::None)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    // Loops::Twice rejects an odd diagonal, and halves an even one.
    let s = Matrix::from_rows(&[[1.0, 1.0], [1.0, 0.0]]).unwrap();
    assert!(Graph::adjacency(&s, Adjacency::Undirected, Loops::Twice).is_err());
    let s = Matrix::from_rows(&[[4.0, 1.0], [1.0, 0.0]]).unwrap();
    assert_eq!(
        Graph::adjacency(&s, Adjacency::Undirected, Loops::Twice)
            .unwrap()
            .ecount(),
        3
    );
    assert_eq!(
        Graph::adjacency(&s, Adjacency::Undirected, Loops::Once)
            .unwrap()
            .ecount(),
        5
    );
    // Non-square and negative matrices.
    let r = Matrix::from_rows(&[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]).unwrap();
    assert!(Graph::adjacency(&r, Adjacency::Directed, Loops::None).is_err());
    let neg = Matrix::from_rows(&[[0.0, -1.0], [0.0, 0.0]]).unwrap();
    assert!(Graph::adjacency(&neg, Adjacency::Directed, Loops::None).is_err());
}

#[test]
fn adjacency_round_trip_on_karate() {
    let k = Graph::famous(FamousGraph::Zachary).unwrap();
    let n = k.vcount();
    // Dense: get_adjacency and adjacency are inverse to each other.
    let a = k
        .get_adjacency(GetAdjacency::Both, None, Loops::Twice)
        .unwrap();
    assert_eq!((a.nrow(), a.ncol()), (n, n));
    let g = Graph::adjacency(&a, Adjacency::Undirected, Loops::Twice).unwrap();
    assert_eq!(edge_set(&g), edge_set(&k));
    for mode in [Adjacency::Max, Adjacency::Min, Adjacency::Plus] {
        let g = Graph::adjacency(&a, mode, Loops::Twice).unwrap();
        let expected = if mode == Adjacency::Plus { 2 * 78 } else { 78 };
        assert_eq!(g.ecount(), expected, "{mode:?}");
    }
    // Sparse: the coordinate entries of get_adjacency_sparse are our triplets.
    for (kind, mode) in [
        (GetAdjacency::Both, Adjacency::Undirected),
        (GetAdjacency::Upper, Adjacency::Upper),
        (GetAdjacency::Lower, Adjacency::Lower),
        (GetAdjacency::Upper, Adjacency::Max),
        (GetAdjacency::Lower, Adjacency::Max),
    ] {
        let coo = k.get_adjacency_sparse(kind, None, Loops::Twice).unwrap();
        let s = Graph::sparse_adjacency(n, &coo.entries, mode, Loops::Twice).unwrap();
        assert_eq!(edge_set(&s), edge_set(&k), "{kind:?} {mode:?}");
        assert_eq!(s.ecount(), 78);
    }
    // Weighted round trip: weights survive, keyed by endpoints.
    let weights: Vec<f64> = (0..k.ecount()).map(|e| 1.0 + e as f64).collect();
    let wa = k
        .get_adjacency(GetAdjacency::Both, Some(&weights), Loops::Twice)
        .unwrap();
    let (wg, ww) = Graph::weighted_adjacency(&wa, Adjacency::Undirected, Loops::Twice).unwrap();
    assert_eq!(
        weighted_edges(&wg, Some(&ww)),
        weighted_edges(&k, Some(&weights))
    );
    let coo = k
        .get_adjacency_sparse(GetAdjacency::Upper, Some(&weights), Loops::Twice)
        .unwrap();
    let (sg, sw) =
        Graph::sparse_weighted_adjacency(n, &coo.entries, Adjacency::Upper, Loops::Twice).unwrap();
    assert_eq!(
        weighted_edges(&sg, Some(&sw)),
        weighted_edges(&k, Some(&weights))
    );
}

#[test]
fn weighted_adjacency_modes() {
    // Pattern of examples/simple/igraph_weighted_adjacency.c
    let a = Matrix::from_rows(&[
        [0.0, 2.0, 0.0, 1.0],
        [2.0, 0.0, 0.5, 0.0],
        [0.0, 3.0, 4.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    .unwrap();
    let weights_of = |g: &Graph, w: &[f64]| -> Vec<((i64, i64), f64)> {
        let mut v: Vec<_> = g
            .edge_list()
            .into_iter()
            .map(|(a, b)| (a.min(b), a.max(b)))
            .zip(w.iter().copied())
            .collect();
        v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        v
    };
    let (d, w) = Graph::weighted_adjacency(&a, Adjacency::Directed, Loops::Once).unwrap();
    assert_eq!(d.ecount(), 7);
    assert_eq!(w.iter().sum::<f64>(), 13.5);
    let (mx, w) = Graph::weighted_adjacency(&a, Adjacency::Max, Loops::Once).unwrap();
    assert_eq!(
        weights_of(&mx, &w),
        vec![((0, 1), 2.0), ((0, 3), 1.0), ((1, 2), 3.0), ((2, 2), 4.0)]
    );
    // Undirected mode insists on symmetry; on a symmetric matrix it agrees with Max.
    let err = Graph::weighted_adjacency(&a, Adjacency::Undirected, Loops::Once).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let mut sym = a.clone();
    sym[(1, 2)] = 3.0;
    let (un, w2) = Graph::weighted_adjacency(&sym, Adjacency::Undirected, Loops::Once).unwrap();
    assert_eq!(weights_of(&un, &w2), weights_of(&mx, &w));
    let (mn, w) = Graph::weighted_adjacency(&a, Adjacency::Min, Loops::None).unwrap();
    assert_eq!(
        weights_of(&mn, &w),
        vec![((0, 1), 2.0), ((0, 3), 1.0), ((1, 2), 0.5)]
    );
    let (pl, w) = Graph::weighted_adjacency(&a, Adjacency::Plus, Loops::Twice).unwrap();
    assert_eq!(
        weights_of(&pl, &w),
        vec![((0, 1), 4.0), ((0, 3), 2.0), ((1, 2), 3.5), ((2, 2), 2.0)]
    );
    let (up, w) = Graph::weighted_adjacency(&a, Adjacency::Upper, Loops::Once).unwrap();
    assert_eq!(
        weights_of(&up, &w),
        vec![((0, 1), 2.0), ((0, 3), 1.0), ((1, 2), 0.5), ((2, 2), 4.0)]
    );
    let (lo, w) = Graph::weighted_adjacency(&a, Adjacency::Lower, Loops::Once).unwrap();
    assert_eq!(
        weights_of(&lo, &w),
        vec![((0, 1), 2.0), ((0, 3), 1.0), ((1, 2), 3.0), ((2, 2), 4.0)]
    );
    let r = Matrix::zeros(2, 3);
    assert!(Graph::weighted_adjacency(&r, Adjacency::Directed, Loops::None).is_err());
}

#[test]
fn sparse_adjacency_matches_dense() {
    let rows = [
        [0.0, 2.0, 0.0, 1.0],
        [2.0, 0.0, 0.5, 0.0],
        [0.0, 3.0, 4.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ];
    let dense = Matrix::from_rows(&rows).unwrap();
    let triplets: Vec<(i64, i64, f64)> = rows
        .iter()
        .enumerate()
        .flat_map(|(i, r)| {
            r.iter()
                .enumerate()
                .filter(|(_, x)| **x != 0.0)
                .map(move |(j, &x)| (i as i64, j as i64, x))
        })
        .collect();
    for mode in [
        Adjacency::Directed,
        Adjacency::Max,
        Adjacency::Min,
        Adjacency::Plus,
        Adjacency::Upper,
        Adjacency::Lower,
    ] {
        for loops in [Loops::None, Loops::Once, Loops::Twice] {
            let (g1, w1) = Graph::weighted_adjacency(&dense, mode, loops).unwrap();
            let (g2, w2) = Graph::sparse_weighted_adjacency(4, &triplets, mode, loops).unwrap();
            let mut e1: Vec<_> = g1
                .edge_list()
                .into_iter()
                .zip(w1)
                .map(|((a, b), w)| (a, b, (w * 4.0) as i64))
                .collect();
            let mut e2: Vec<_> = g2
                .edge_list()
                .into_iter()
                .zip(w2)
                .map(|((a, b), w)| (a, b, (w * 4.0) as i64))
                .collect();
            e1.sort();
            e2.sort();
            assert_eq!(e1, e2, "{mode:?} {loops:?}");
        }
    }
    // Duplicate triplets are summed, like entries of a dense matrix.
    let g = Graph::sparse_adjacency(
        2,
        &[(0, 1, 1.0), (0, 1, 2.0)],
        Adjacency::Directed,
        Loops::None,
    )
    .unwrap();
    assert_eq!(g.ecount(), 3);
    // Out of range entries are caught on the Rust side.
    let err =
        Graph::sparse_adjacency(2, &[(0, 2, 1.0)], Adjacency::Directed, Loops::None).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    assert!(Graph::sparse_adjacency(2, &[(-1, 0, 1.0)], Adjacency::Directed, Loops::None).is_err());
    // The empty matrix gives the null graph; a matrix without entries, isolated vertices.
    assert_eq!(
        Graph::sparse_adjacency(0, &[], Adjacency::Directed, Loops::None)
            .unwrap()
            .vcount(),
        0
    );
    let iso = Graph::sparse_adjacency(5, &[], Adjacency::Undirected, Loops::None).unwrap();
    assert_eq!((iso.vcount(), iso.ecount()), (5, 0));
    assert!(Graph::sparse_adjacency(2, &[(0, 1, -1.0)], Adjacency::Directed, Loops::None).is_err());
}

/// A tiny deterministic pseudo-random generator (64-bit LCG) for test data.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, bound: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) % bound
    }
}

/// Canonical multiset of `(from, to, weight)` of an undirected or directed
/// graph (undirected endpoints sorted).
fn weighted_edges(g: &Graph, w: Option<&[f64]>) -> Vec<(i64, i64, i64)> {
    let mut e: Vec<_> = g
        .edge_list()
        .into_iter()
        .enumerate()
        .map(|(k, (a, b))| {
            let (a, b) = if g.is_directed() {
                (a, b)
            } else {
                (a.min(b), a.max(b))
            };
            (a, b, w.map_or(0, |w| (w[k] * 4.0) as i64))
        })
        .collect();
    e.sort();
    e
}

#[test]
fn sparse_adjacency_agrees_with_dense_on_asymmetric_matrices() {
    // Regression test for an igraph 1.0.0 and 1.0.1 bug worked around by the
    // wrapper (see `upstream_defects_still_present_in_igraph_1_0_1`):
    // the sparse Max (and weighted Min/Plus) routines dropped the entries
    // below the diagonal whose mirror entry was absent.
    let lower_only = [(1, 0, 2.0)];
    let d = Matrix::from_rows(&[[0.0, 0.0], [2.0, 0.0]]).unwrap();
    let (s, _) =
        Graph::sparse_weighted_adjacency(2, &lower_only, Adjacency::Plus, Loops::None).unwrap();
    assert_eq!(s.edge_list(), vec![(0, 1)]);
    let s = Graph::sparse_adjacency(2, &lower_only, Adjacency::Max, Loops::None).unwrap();
    assert_eq!(
        s,
        Graph::adjacency(&d, Adjacency::Max, Loops::None).unwrap()
    );
    assert_eq!(s.ecount(), 2);

    let modes = [
        Adjacency::Directed,
        Adjacency::Undirected,
        Adjacency::Max,
        Adjacency::Min,
        Adjacency::Plus,
        Adjacency::Upper,
        Adjacency::Lower,
    ];
    let weights = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, -1.5, 2.5];
    let mut rng = Lcg(2024);
    for trial in 0..60 {
        let n = 1 + rng.next(6) as usize;
        let symmetric = trial % 3 == 0;
        let mut counts = Matrix::zeros(n, n);
        let mut weighted = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if symmetric && j < i {
                    counts[(i, j)] = counts[(j, i)];
                    weighted[(i, j)] = weighted[(j, i)];
                } else {
                    // Even diagonal counts, so that Loops::Twice is valid.
                    let c = rng.next(3) as f64;
                    counts[(i, j)] = if i == j { 2.0 * c } else { c };
                    weighted[(i, j)] = weights[rng.next(weights.len() as u64) as usize];
                }
            }
        }
        // Some nonzero entries are split into two triplets (integers into two
        // integers, the other weights into exact halves), and some zero
        // entries are given explicitly: neither may change the result.
        let triplets = |m: &Matrix| -> Vec<(i64, i64, f64)> {
            let mut t = vec![];
            for i in 0..n {
                for j in 0..n {
                    let (a, b, x) = (i as i64, j as i64, m[(i, j)]);
                    if x == 0.0 {
                        if (i * 7 + j + trial) % 3 == 0 {
                            t.push((a, b, 0.0));
                        }
                    } else if (i + j + trial) % 2 == 0 && x.fract() == 0.0 && x.abs() >= 2.0 {
                        let half = (x / 2.0).floor();
                        t.push((a, b, half));
                        t.push((a, b, x - half));
                    } else if (i + j + trial) % 2 == 0 && x.fract() != 0.0 {
                        t.push((a, b, x / 2.0));
                        t.push((a, b, x / 2.0));
                    } else {
                        t.push((a, b, x));
                    }
                }
            }
            t
        };
        for mode in modes {
            if mode == Adjacency::Undirected && !symmetric {
                continue;
            }
            for loops in [Loops::None, Loops::Once, Loops::Twice] {
                let dense = Graph::adjacency(&counts, mode, loops).unwrap();
                let sparse = Graph::sparse_adjacency(n, &triplets(&counts), mode, loops).unwrap();
                assert_eq!(
                    weighted_edges(&dense, None),
                    weighted_edges(&sparse, None),
                    "trial {trial}, {mode:?}, {loops:?}"
                );
                let (dg, dw) = Graph::weighted_adjacency(&weighted, mode, loops).unwrap();
                let (sg, sw) =
                    Graph::sparse_weighted_adjacency(n, &triplets(&weighted), mode, loops).unwrap();
                assert_eq!(
                    weighted_edges(&dg, Some(&dw)),
                    weighted_edges(&sg, Some(&sw)),
                    "weighted, trial {trial}, {mode:?}, {loops:?}"
                );
            }
        }
    }
}

#[test]
fn sparse_undirected_accepts_explicit_zeros_and_cancelling_duplicates() {
    // A symmetric matrix stays symmetric however its zeros are spelled: an
    // explicit zero on one side only, a negative zero, or weights that cancel.
    let g = Graph::sparse_adjacency(
        3,
        &[(0, 1, 1.0), (1, 0, 1.0), (1, 2, 0.0)],
        Adjacency::Undirected,
        Loops::None,
    )
    .unwrap();
    assert_eq!(g.edge_list(), vec![(0, 1)]);
    let (g, w) = Graph::sparse_weighted_adjacency(
        3,
        &[
            (0, 1, 2.0),
            (1, 0, 2.0),
            (1, 2, 1.0),
            (1, 2, -1.0),
            (2, 0, -0.0),
            (0, 2, 0.0),
        ],
        Adjacency::Undirected,
        Loops::None,
    )
    .unwrap();
    assert_eq!((g.edge_list(), w), (vec![(0, 1)], vec![2.0]));
    // Duplicates are summed before the symmetry test: 1 + 2 on one side, 3 on the other.
    let (g, w) = Graph::sparse_weighted_adjacency(
        2,
        &[(0, 1, 1.0), (0, 1, 2.0), (1, 0, 3.0)],
        Adjacency::Undirected,
        Loops::None,
    )
    .unwrap();
    assert_eq!((g.ecount(), w), (1, vec![3.0]));
    // Sums of valid counts must still be valid counts.
    let err = Graph::sparse_adjacency(
        2,
        &[(0, 1, 9.0e18), (0, 1, 9.0e18)],
        Adjacency::Directed,
        Loops::None,
    )
    .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    // A genuinely asymmetric matrix is still rejected.
    assert_eq!(
        Graph::sparse_adjacency(2, &[(0, 1, 1.0)], Adjacency::Undirected, Loops::None)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn adjacency_rejects_values_that_are_not_edge_counts() {
    for bad in [f64::NAN, f64::INFINITY, -1.0, 1.5, 1e300] {
        let m = Matrix::from_rows(&[[0.0, bad], [0.0, 0.0]]).unwrap();
        let err = Graph::adjacency(&m, Adjacency::Directed, Loops::None).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidValue, "{bad}");
        let err = Graph::sparse_adjacency(2, &[(0, 1, bad)], Adjacency::Directed, Loops::None)
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidValue, "{bad}");
    }
    // Weighted variants accept any real weight.
    let m = Matrix::from_rows(&[[0.0, 1.5], [-2.0, 0.0]]).unwrap();
    let (_, w) = Graph::weighted_adjacency(&m, Adjacency::Directed, Loops::None).unwrap();
    assert_eq!(w.len(), 2);
    // Out-of-range ids are reported as such even when the value is bad too.
    let err = Graph::sparse_adjacency(2, &[(5, 0, f64::NAN)], Adjacency::Directed, Loops::None)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
}

/// Proves, by calling the raw C functions, that the upstream defects the
/// wrappers work around are still there in the igraph release in use (1.0.0
/// and 1.0.1 both have them; igraph's own `tests/unit/igraph_wheel.out` still
/// pins the null graph for the one-vertex wheel). When a future igraph fixes
/// them this test says so, and the workarounds can go.
#[test]
fn upstream_defects_still_present_in_igraph_1_0_1() {
    use igraph::{ffi, linalg::SparseMat};
    if igraph::misc::version().triple() > (1, 0, 1) {
        return;
    }
    // igraph_star / igraph_wheel with n = 1 give the null graph.
    let raw_star =
        Graph::init_with(|g| unsafe { ffi::igraph_star(g, 1, StarMode::Out.into(), 0) }).unwrap();
    assert_eq!(
        raw_star.vcount(),
        0,
        "igraph_star(n = 1) was fixed upstream"
    );
    let raw_wheel =
        Graph::init_with(|g| unsafe { ffi::igraph_wheel(g, 1, WheelMode::Undirected.into(), 0) })
            .unwrap();
    assert_eq!(
        raw_wheel.vcount(),
        0,
        "igraph_wheel(n = 1) was fixed upstream"
    );
    assert_eq!(Graph::star(1, StarMode::Out, 0).unwrap().vcount(), 1);

    // igraph_sparse_adjacency in the MAX mode drops A[1][0] when A[0][1] is
    // not stored, while the dense igraph_adjacency keeps it.
    let mut sparse = SparseMat::from_triplets(2, 2, &[(1, 0, 2.0)])
        .unwrap()
        .compress()
        .unwrap();
    let raw_sparse_max = Graph::init_with(|g| unsafe {
        ffi::igraph_sparse_adjacency(g, &mut sparse, Adjacency::Max.into(), Loops::None.into())
    })
    .unwrap();
    assert_eq!(
        raw_sparse_max.ecount(),
        0,
        "igraph_sparse_adjacency (MAX) was fixed upstream"
    );
    let dense = Matrix::from_rows(&[[0.0, 0.0], [2.0, 0.0]]).unwrap();
    assert_eq!(
        Graph::adjacency(&dense, Adjacency::Max, Loops::None)
            .unwrap()
            .ecount(),
        2
    );
    assert_eq!(
        Graph::sparse_adjacency(2, &[(1, 0, 2.0)], Adjacency::Max, Loops::None)
            .unwrap()
            .ecount(),
        2
    );

    // igraph_sparse_adjacency in the UNDIRECTED mode tests symmetry
    // structurally, so one explicitly stored zero makes it fail.
    let mut sparse = SparseMat::from_triplets(2, 2, &[(0, 1, 0.0)])
        .unwrap()
        .compress()
        .unwrap();
    let raw_undirected = Graph::init_with(|g| unsafe {
        ffi::igraph_sparse_adjacency(
            g,
            &mut sparse,
            Adjacency::Undirected.into(),
            Loops::None.into(),
        )
    });
    assert!(
        raw_undirected.is_err(),
        "igraph_sparse_adjacency (UNDIRECTED, explicit zero) was fixed upstream"
    );
    let g = Graph::sparse_adjacency(2, &[(0, 1, 0.0)], Adjacency::Undirected, Loops::None).unwrap();
    assert_eq!((g.vcount(), g.ecount()), (2, 0));
}

#[test]
fn degenerate_arguments_are_handled_safely() {
    // igraph 1.0.0 and 1.0.1 return the null graph for the one-vertex star;
    // the wrapper restores the vertex.
    for mode in [
        StarMode::Out,
        StarMode::In,
        StarMode::Mutual,
        StarMode::Undirected,
    ] {
        let s = Graph::star(1, mode, 0).unwrap();
        assert_eq!((s.vcount(), s.ecount()), (1, 0), "{mode:?}");
        assert_eq!(s.is_directed(), mode != StarMode::Undirected);
        assert_eq!(
            Graph::star(0, mode, 0).unwrap_err().kind(),
            ErrorKind::InvalidValue
        );
    }
    let w = Graph::wheel(1, WheelMode::Undirected, 0).unwrap();
    assert_eq!((w.vcount(), w.ecount()), (1, 0));
    assert!(Graph::star(3, StarMode::Out, -1).is_err());

    // LCF on 0 vertices crashes igraph 1.0.0 and 1.0.1 with a division by
    // zero; the wrapper returns the null graph instead.
    let g = Graph::lcf(0, &[1, 2], 3).unwrap();
    assert_eq!((g.vcount(), g.ecount()), (0, 0));
    // Shifts are reduced modulo n: huge or very negative shifts are fine.
    let reference = Graph::lcf(8, &[3, -3], 4).unwrap();
    for shifts in [[3 + 8 * 1000, -3 - 8 * 1000], [i64::MAX - 4, i64::MIN + 5]] {
        // i64::MAX - 4 = 3 (mod 8), i64::MIN + 5 = 5 = -3 (mod 8).
        let g = Graph::lcf(8, &shifts, 4).unwrap();
        assert_eq!(edge_set(&g), edge_set(&reference), "{shifts:?}");
    }
    // Chords that are multiples of n are dropped: the plain cycle remains.
    let c = Graph::lcf(5, &[0, 5, -10], 2).unwrap();
    assert_eq!(
        edge_set(&c),
        edge_set(&Graph::cycle_graph(5, false, false).unwrap())
    );

    // Extended chordal rings: offsets are taken modulo the ring size.
    let a = Graph::extended_chordal_ring(8, &[[2, -2]], false).unwrap();
    let b = Graph::extended_chordal_ring(8, &[[2 + 8 * (1 << 40), i64::MIN + 6]], false).unwrap();
    assert_eq!(a, b);
    assert_eq!(
        Graph::extended_chordal_ring(2, &[[1]], false)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    // Ragged rows are an error; only-empty rows mean "no chords".
    let ragged: [&[i64]; 2] = [&[1, 2], &[1]];
    assert!(Graph::extended_chordal_ring(6, &ragged, false).is_err());
    let empty: [&[i64]; 2] = [&[], &[]];
    assert_eq!(
        Graph::extended_chordal_ring(6, &empty, false)
            .unwrap()
            .ecount(),
        6
    );

    // Tree generators reject empty branchings.
    assert!(Graph::kary_tree(5, 0, TreeMode::Out).is_err());
    assert!(Graph::symmetric_tree(&[2, 0], TreeMode::Out).is_err());
    assert!(Graph::regular_tree(0, 3, TreeMode::Undirected).is_err());
    assert!(Graph::regular_tree(2, 1, TreeMode::Undirected).is_err());

    // Word graphs over an empty alphabet, and with empty words.
    assert_eq!(Graph::de_bruijn(0, 3).unwrap().vcount(), 0);
    assert_eq!(Graph::de_bruijn(0, 0).unwrap().vcount(), 1);
    assert_eq!(Graph::kautz(0, 3).unwrap().vcount(), 0);
    assert_eq!(Graph::kautz(0, 0).unwrap().vcount(), 1);

    // Too large requests fail cleanly instead of overflowing.
    for dim in [58, 70] {
        assert_eq!(
            Graph::hypercube(dim, false).unwrap_err().kind(),
            ErrorKind::InvalidValue
        );
    }
    assert!(Graph::de_bruijn(1 << 20, 1 << 20).is_err());
    assert!(Graph::kautz(1 << 20, 1 << 20).is_err());
    assert!(Graph::full(usize::MAX, false, false).is_err());
}

// ----------------------------------------------------- derived graphs ---

#[test]
fn line_graphs() {
    // L(K_{1,n}) = K_n, L(C_n) = C_n, L(P_n) = P_{n-1}.
    let l = Graph::star(6, StarMode::Undirected, 0)
        .unwrap()
        .linegraph()
        .unwrap();
    assert_eq!(
        edge_set(&l),
        edge_set(&Graph::full(5, false, false).unwrap())
    );
    let l = Graph::cycle_graph(7, false, false)
        .unwrap()
        .linegraph()
        .unwrap();
    assert!(is_regular(&l, 2) && is_connected(&l) && l.vcount() == 7);
    let l = Graph::path_graph(6, false, false)
        .unwrap()
        .linegraph()
        .unwrap();
    assert_eq!((l.vcount(), l.ecount()), (5, 4));
    // L(K_4) is the octahedron, L(Petersen) is 4-regular with 15 vertices.
    let l = Graph::full(4, false, false).unwrap().linegraph().unwrap();
    assert_eq!(
        canonical(&l),
        canonical(&Graph::famous("Octahedron").unwrap())
    );
    let lp = Graph::famous("Petersen").unwrap().linegraph().unwrap();
    assert_eq!((lp.vcount(), lp.ecount()), (15, 30));
    assert!(is_regular(&lp, 4));
    // In general |E(L(G))| = sum_v C(d_v, 2).
    let k = karate();
    let expected: i64 = degrees(&k).iter().map(|d| d * (d - 1) / 2).sum();
    assert_eq!(k.linegraph().unwrap().ecount() as i64, expected);
    // Directed: e -> f iff target(e) == source(f). L(directed C_n) = directed C_n.
    let dc = Graph::cycle_graph(5, true, false).unwrap();
    let l = dc.linegraph().unwrap();
    assert!(l.is_directed());
    assert_eq!(l.ecount(), 5);
    for (e, f) in l.edge_list() {
        assert_eq!(dc.edge(e).unwrap().1, dc.edge(f).unwrap().0);
    }
    // A self-loop gets a single self-loop in the line graph, directed or not.
    for directed in [false, true] {
        let g = Graph::from_edges(&[(0, 0)], 1, directed).unwrap();
        assert_eq!(g.linegraph().unwrap().edge_list(), vec![(0, 0)]);
    }
    // The vertex of an undirected self-loop counts as two endpoints: the loop
    // and an edge at that vertex are joined twice; parallel edges, too.
    let g = Graph::from_edges(&[(0, 0), (0, 1)], 2, false).unwrap();
    assert_eq!(
        g.linegraph().unwrap().edge_list(),
        vec![(0, 0), (0, 1), (0, 1)]
    );
    let g = Graph::from_edges(&[(0, 1), (0, 1)], 2, false).unwrap();
    assert_eq!(g.linegraph().unwrap().edge_list(), vec![(0, 1), (0, 1)]);
}

#[test]
fn mycielski_graphs() {
    // M_0..M_4: null, K_1, K_2, C_5, Grötzsch.
    assert_eq!(Graph::mycielski_graph(0).unwrap().vcount(), 0);
    assert_eq!(Graph::mycielski_graph(1).unwrap().vcount(), 1);
    assert_eq!(Graph::mycielski_graph(2).unwrap().edge_list().len(), 1);
    let m3 = Graph::mycielski_graph(3).unwrap();
    assert!(is_regular(&m3, 2) && is_connected(&m3) && m3.vcount() == 5);
    let m4 = Graph::mycielski_graph(4).unwrap();
    assert!(isomorphic(&m4, &Graph::famous("Grotzsch").unwrap()));
    // M_{k+1} is the Mycielskian of M_k.
    for k in 2..6 {
        let next = Graph::mycielski_graph(k).unwrap().mycielskian(1).unwrap();
        assert!(
            isomorphic(&next, &Graph::mycielski_graph(k + 1).unwrap()),
            "M_{k}"
        );
    }
    for k in 2..8u32 {
        let m = Graph::mycielski_graph(k as usize).unwrap();
        let n = 3 * 2usize.pow(k - 2) - 1;
        let e = (7 * 3usize.pow(k - 2)).div_ceil(2) - 3 * 2usize.pow(k - 2);
        assert_eq!((m.vcount(), m.ecount()), (n, e), "M_{k}");
        assert_eq!(triangle_count(&m), 0, "M_{k} is triangle-free");
    }
    // Chromatic number grows by one at each step.
    for k in 1..5 {
        assert_eq!(chromatic_number(&Graph::mycielski_graph(k).unwrap()), k);
    }
}

#[test]
fn small_graphs_from_flat_lists() {
    let g = Graph::small(4, true, &[0, 1, 1, 2, 2, 3, 3, 0]).unwrap();
    assert_eq!(g, Graph::cycle_graph(4, true, false).unwrap());
    // The vertex count grows to fit the ids.
    assert_eq!(Graph::small(0, false, &[0, 5]).unwrap().vcount(), 6);
    assert_eq!(
        Graph::small(3, false, &[0, 1, 2]).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert!(Graph::small(3, false, &[0, -1]).is_err());
}

#[test]
fn allowed_edge_types_conversions() {
    assert_eq!(
        AllowedEdgeTypes::from(EdgeTypeSw::Simple),
        AllowedEdgeTypes::SIMPLE
    );
    assert_eq!(
        AllowedEdgeTypes::from(EdgeTypeSw::Loops),
        AllowedEdgeTypes::LOOPS
    );
    assert_eq!(
        AllowedEdgeTypes::from(EdgeTypeSw::Multi),
        AllowedEdgeTypes::MULTI
    );
    let raw = igraph::ffi::igraph_edge_type_sw_t::from(AllowedEdgeTypes::ALL);
    assert_eq!(
        raw,
        igraph::ffi::IGRAPH_LOOPS_SW | igraph::ffi::IGRAPH_MULTI_SW
    );
    assert_eq!(
        AllowedEdgeTypes::try_from(raw).unwrap(),
        AllowedEdgeTypes::ALL
    );
    // Only the four valid flag combinations convert back.
    assert_eq!(
        AllowedEdgeTypes::try_from(2 as igraph::ffi::igraph_edge_type_sw_t)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    // One type crate-wide: the constructors, games, mixing and constants
    // paths all name it, and the old enum-style names are aliases.
    let from_games: igraph::games::AllowedEdgeTypes = AllowedEdgeTypes::MULTI;
    let from_mixing: igraph::mixing::AllowedEdgeTypes = from_games;
    let from_constants: igraph::constants::AllowedEdgeTypes = from_mixing;
    assert_eq!(from_constants, AllowedEdgeTypes::MULTI);
    assert_eq!(AllowedEdgeTypes::Simple, AllowedEdgeTypes::SIMPLE);
    assert_eq!(AllowedEdgeTypes::Loops, AllowedEdgeTypes::LOOPS);
    assert_eq!(AllowedEdgeTypes::Multi, AllowedEdgeTypes::MULTI);
    assert_eq!(AllowedEdgeTypes::LoopsAndMulti, AllowedEdgeTypes::ALL);
    for flags in [
        AllowedEdgeTypes::SIMPLE,
        AllowedEdgeTypes::LOOPS,
        AllowedEdgeTypes::MULTI,
        AllowedEdgeTypes::ALL,
    ] {
        let raw = igraph::ffi::igraph_edge_type_sw_t::from(flags);
        assert_eq!(raw, flags.to_raw());
        assert_eq!(AllowedEdgeTypes::try_from(raw).unwrap(), flags);
    }
}

// --------------------------------------------------------- use case ---

/// Use case: designing an interconnection network for a small parallel
/// machine with 16 processors. We compare candidate topologies built with the
/// deterministic generators by the metrics an architect cares about: wiring
/// cost (edges), port count (max degree), latency (diameter) and fault
/// tolerance (does the network stay connected after any single link fails?).
#[test]
fn use_case_choosing_an_interconnection_network() {
    struct Candidate {
        name: &'static str,
        graph: Graph,
    }
    let candidates = vec![
        Candidate {
            name: "ring",
            graph: Graph::ring(16, false, false, true).unwrap(),
        },
        Candidate {
            name: "star",
            graph: Graph::star(16, StarMode::Undirected, 0).unwrap(),
        },
        Candidate {
            name: "binary tree",
            graph: Graph::kary_tree(16, 2, TreeMode::Undirected).unwrap(),
        },
        Candidate {
            name: "4x4 mesh",
            graph: Graph::square_lattice(&[4, 4], 1, false, false, None).unwrap(),
        },
        Candidate {
            name: "4x4 torus",
            graph: Graph::square_lattice(&[4, 4], 1, false, false, Some(&[true, true])).unwrap(),
        },
        Candidate {
            name: "hypercube Q4",
            graph: Graph::hypercube(4, false).unwrap(),
        },
        Candidate {
            name: "chordal ring C16(1,4)",
            graph: Graph::circulant(16, &[1, 4], false).unwrap(),
        },
        Candidate {
            name: "Moebius-Kantor [5,-5]^8",
            graph: Graph::lcf(16, &[5, -5], 8).unwrap(),
        },
        Candidate {
            name: "complete",
            graph: Graph::full(16, false, false).unwrap(),
        },
    ];

    let survives_any_link_failure = |g: &Graph| {
        g.edge_ids().all(|e| {
            let mut h = g.clone();
            h.delete_edges(e).unwrap();
            is_connected(&h)
        })
    };

    let mut report = vec![];
    for c in &candidates {
        let g = &c.graph;
        assert_eq!(g.vcount(), 16, "{}", c.name);
        let max_deg = *degrees(g).iter().max().unwrap();
        report.push((
            c.name,
            g.ecount(),
            max_deg,
            diameter(g),
            survives_any_link_failure(g),
        ));
    }
    let find = |name: &str| *report.iter().find(|r| r.0 == name).unwrap();

    // Textbook facts about these topologies.
    assert_eq!(find("ring"), ("ring", 16, 2, 8, true));
    assert_eq!(find("star"), ("star", 15, 15, 2, false));
    assert!(!find("binary tree").4);
    assert_eq!(find("4x4 mesh"), ("4x4 mesh", 24, 4, 6, true));
    assert_eq!(find("4x4 torus"), ("4x4 torus", 32, 4, 4, true));
    assert_eq!(find("hypercube Q4"), ("hypercube Q4", 32, 4, 4, true));
    assert_eq!(find("complete"), ("complete", 120, 15, 1, true));
    // The cubic Möbius–Kantor graph reaches diameter 4 with only 3 ports.
    assert_eq!(
        find("Moebius-Kantor [5,-5]^8"),
        ("Moebius-Kantor [5,-5]^8", 24, 3, 4, true)
    );

    // The architect's rule: at most 4 ports, fault tolerant, then minimize
    // latency and then wiring cost.
    let best = report
        .iter()
        .filter(|r| r.2 <= 4 && r.4)
        .min_by_key(|r| (r.3, r.1))
        .unwrap();
    assert_eq!(
        best.3, 3,
        "some 4-port topology reaches diameter 3: {best:?}"
    );
    assert_eq!(best.0, "chordal ring C16(1,4)");
}

#[test]
fn huge_de_bruijn_and_kautz_are_rejected_before_igraph() {
    // igraph converts pow(m, n) (a double) to an integer before checking its
    // range: the wrappers must reject these parameters themselves.
    let big = i64::MAX as usize;
    for (m, n) in [(3, 40), (2, 63), (1 << 20, 1 << 20), (big, 2)] {
        assert_eq!(
            Graph::de_bruijn(m, n).unwrap_err().kind(),
            ErrorKind::InvalidValue,
            "de_bruijn({m}, {n})"
        );
    }
    for (m, n) in [(3, 40), (2, 40), (big, 1), (big, 0), (1 << 20, 1 << 20)] {
        assert_eq!(
            Graph::kautz(m, n).unwrap_err().kind(),
            ErrorKind::InvalidValue,
            "kautz({m}, {n})"
        );
    }
    // m^n fits but the m^(n+1) edges do not: igraph's EOVERFLOW. For
    // m = i64::MAX and n = 1, pow(m, 1) is 2^63 as a double, so this must
    // also be caught before igraph converts it back to an integer.
    for (m, n) in [(big, 1), (big - 100, 1), (2, 62), (3, 39)] {
        assert_eq!(
            Graph::de_bruijn(m, n).unwrap_err().kind(),
            ErrorKind::Overflow,
            "de_bruijn({m}, {n})"
        );
    }
    // Small parameters work, and m = 1 has a single vertex.
    let b = Graph::de_bruijn(1, 1 << 40).unwrap();
    assert_eq!((b.vcount(), b.ecount()), (1, 1));
    let b = Graph::de_bruijn(2, 10).unwrap();
    assert_eq!((b.vcount(), b.ecount()), (1024, 2048));
    let k = Graph::kautz(1, 5).unwrap();
    assert_eq!((k.vcount(), k.ecount()), (2, 2));
    let k = Graph::kautz(2, 3).unwrap();
    assert_eq!((k.vcount(), k.ecount()), (3 * 8, 2 * 3 * 8));
}

#[test]
fn huge_hexagon_shaped_lattices_are_rejected_before_igraph() {
    let m = i64::MAX as usize;
    for dims in [[3, 1 << 62, 1 << 62], [m, m, m], [1, 1, m / 2]] {
        assert_eq!(
            Graph::triangular_lattice(&dims, false, false)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidValue,
            "{dims:?}"
        );
        assert_eq!(
            Graph::hexagonal_lattice(&dims, false, false)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidValue,
            "{dims:?}"
        );
    }
    // Ordinary hexagons are unaffected: the triangular hexagon of side 2 has
    // 7 vertices (a centre and its 6 neighbours) and 12 edges.
    let t = Graph::triangular_lattice(&[2, 2, 2], false, false).unwrap();
    assert_eq!((t.vcount(), t.ecount()), (7, 12));
    // Its dual is coronene: 7 hexagons, 24 vertices, 30 edges.
    let h = Graph::hexagonal_lattice(&[2, 2, 2], false, false).unwrap();
    assert_eq!((h.vcount(), h.ecount()), (24, 30));
}

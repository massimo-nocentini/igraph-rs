//! Integration tests for the `structural` module (`igraph_structural.h`).

mod common;

use common::*;
use igraph::games::{AllowedEdgeTypes, BarabasiOptions};
use igraph::linalg::{SymmetricRange, lapack_dsyevr};
use igraph::prelude::*;
use igraph::structural::LaplacianNormalization;

/// Graph made of the given edge ids of `g` (same vertex set).
fn edge_subgraph(g: &Graph, eids: &[i64]) -> Graph {
    g.subgraph_from_edges(eids, false).unwrap()
}

fn petersen() -> Graph {
    Graph::famous("Petersen").unwrap()
}

/// Eigenvalues (increasing) of the unnormalized Laplacian of an undirected graph.
fn laplacian_spectrum(g: &Graph) -> Vec<f64> {
    let l = g
        .get_laplacian(
            NeighborMode::All,
            LaplacianNormalization::Unnormalized,
            None,
        )
        .unwrap();
    lapack_dsyevr(&l, &SymmetricRange::All, 1e-12)
        .unwrap()
        .values
}

fn assert_vec_close(a: &[f64], b: &[f64], eps: f64) {
    assert_eq!(a.len(), b.len(), "{a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        if y.is_nan() {
            assert!(x.is_nan(), "{a:?} vs {b:?}");
        } else {
            assert_close(*x, *y, eps);
        }
    }
}

// ---------------------------------------------------------------------------
// Density and degrees
// ---------------------------------------------------------------------------

#[test]
fn mean_degree_and_density_of_karate() {
    let g = karate();
    assert_close(g.mean_degree(true).unwrap(), 2.0 * 78.0 / 34.0, 1e-12);
    assert_close(g.mean_degree(false).unwrap(), 2.0 * 78.0 / 34.0, 1e-12);
    assert_close(
        g.density(None, false).unwrap(),
        78.0 / (34.0 * 33.0 / 2.0),
        1e-12,
    );
    assert_close(
        g.density(None, true).unwrap(),
        78.0 / (34.0 * 35.0 / 2.0),
        1e-12,
    );
    // Uniform weights of 2 double the density.
    let w = vec![2.0; 78];
    assert_close(
        g.density(Some(&w), false).unwrap(),
        2.0 * 78.0 / 561.0,
        1e-12,
    );
}

#[test]
fn mean_degree_directed_and_null() {
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (0, 0)], 3, true).unwrap();
    assert_close(g.mean_degree(true).unwrap(), 4.0 / 3.0, 1e-12);
    assert_close(g.mean_degree(false).unwrap(), 1.0, 1e-12);
    assert!(Graph::new(0, true).mean_degree(true).unwrap().is_nan());
}

#[test]
fn complete_graph_density_is_one() {
    for n in 2..8 {
        let g = complete(n);
        assert_eq!(g.density(None, false).unwrap(), 1.0);
        assert!(g.is_complete().unwrap());
    }
    assert!(Graph::new(0, false).is_complete().unwrap());
    assert!(Graph::new(1, false).is_complete().unwrap());
    assert!(!path(3).is_complete().unwrap());
}

#[test]
fn handshake_lemma_via_strength() {
    let g = karate();
    let s = g
        .strength(.., NeighborMode::All, Loops::Twice, None)
        .unwrap();
    assert_eq!(s.iter().sum::<f64>(), 2.0 * 78.0);
    // Weighted: every edge contributes its weight twice.
    let w: Vec<f64> = (0..78).map(|i| 0.5 + i as f64).collect();
    let s = g
        .strength(.., NeighborMode::All, Loops::Twice, Some(&w))
        .unwrap();
    assert_close(s.iter().sum::<f64>(), 2.0 * w.iter().sum::<f64>(), 1e-9);
    // Strength without weights equals the degree.
    let deg = g.degree(.., NeighborMode::All, Loops::Twice).unwrap();
    let s = g
        .strength(.., NeighborMode::All, Loops::Twice, None)
        .unwrap();
    assert!(deg.iter().zip(&s).all(|(&d, &x)| d as f64 == x));
}

#[test]
fn strength_loop_handling() {
    let g = Graph::from_edges(&[(0, 0), (0, 1)], 2, false).unwrap();
    let w = [3.0, 1.0];
    let s = |l| g.strength(0, NeighborMode::All, l, Some(&w)).unwrap()[0];
    assert_eq!(s(Loops::Twice), 7.0);
    assert_eq!(s(Loops::Once), 4.0);
    assert_eq!(s(Loops::None), 1.0);
}

#[test]
fn maxdegree_and_sorting_on_karate() {
    let g = karate();
    assert_eq!(
        g.maxdegree(.., NeighborMode::All, Loops::Twice).unwrap(),
        17
    );
    assert_eq!(
        g.maxdegree(&[1, 2, 3], NeighborMode::All, Loops::Twice)
            .unwrap(),
        10
    );
    assert_eq!(
        g.maxdegree(VertexSelector::None, NeighborMode::All, Loops::Twice)
            .unwrap(),
        0
    );

    let hubs = g
        .sort_vertex_ids_by_degree(
            ..,
            NeighborMode::All,
            Loops::Twice,
            Order::Descending,
            false,
        )
        .unwrap();
    assert_eq!(&hubs[..3], &[33, 0, 32]);
    let asc = g
        .sort_vertex_ids_by_degree(.., NeighborMode::All, Loops::Twice, Order::Ascending, false)
        .unwrap();
    let deg = g.degree(.., NeighborMode::All, Loops::Twice).unwrap();
    assert!(
        asc.windows(2)
            .all(|p| deg[p[0] as usize] <= deg[p[1] as usize])
    );

    // only_indices refers to positions within the selection.
    let sel = [5, 33, 0];
    let idx = g
        .sort_vertex_ids_by_degree(
            &sel,
            NeighborMode::All,
            Loops::Twice,
            Order::Descending,
            true,
        )
        .unwrap();
    assert_eq!(idx, vec![1, 2, 0]);
    let ids = g
        .sort_vertex_ids_by_degree(
            &sel,
            NeighborMode::All,
            Loops::Twice,
            Order::Descending,
            false,
        )
        .unwrap();
    assert_eq!(ids, vec![33, 0, 5]);
}

#[test]
fn diversity_matches_igraph_unit_test() {
    // From tests/unit/igraph_diversity.c
    let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)], 4, false).unwrap();
    let d = g.diversity(&[3.0, 2.0, 8.0, 1.0, 1.0], ..).unwrap();
    assert_vec_close(&d, &[0.970951, 0.75, 0.69137, 1.0], 1e-5);

    let empty = Graph::new(5, false);
    assert!(empty.diversity(&[], ..).unwrap().iter().all(|x| x.is_nan()));

    let directed = Graph::from_edges(&[(0, 1)], 2, true).unwrap();
    assert_eq!(
        directed.diversity(&[1.0], ..).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.diversity(&[1.0], ..).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

// ---------------------------------------------------------------------------
// Loops, multi-edges, mutuality
// ---------------------------------------------------------------------------

#[test]
fn multiple_edges_like_igraph_example() {
    // examples/simple/igraph_is_multiple.c
    let g = Graph::from_flat_edges(&[0, 1, 1, 2, 2, 1, 0, 1, 1, 0, 3, 4, 11, 10], 0, true).unwrap();
    assert_eq!(
        g.is_multiple(..).unwrap(),
        vec![false, false, false, true, false, false, false]
    );
    assert!(g.has_multiple().unwrap());
    assert_eq!(g.count_multiple(..).unwrap(), vec![2, 1, 1, 2, 1, 1, 1]);
    assert_eq!(g.count_multiple_1(3).unwrap(), 2);
    assert_eq!(
        g.count_multiple_1(99).unwrap_err().kind(),
        ErrorKind::InvalidEdgeId
    );

    let u = Graph::from_flat_edges(
        &[
            0, 0, 1, 2, 1, 1, 2, 2, 2, 1, 2, 3, 2, 4, 2, 5, 2, 6, 2, 2, 3, 2, 0, 0, 6, 2, 2, 2, 0,
            0,
        ],
        0,
        false,
    )
    .unwrap();
    let expected = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
    assert_eq!(
        u.is_multiple(..).unwrap(),
        expected.iter().map(|&x| x == 1).collect::<Vec<_>>()
    );
    // Deleting the flagged edges leaves no multi-edges.
    let flagged: Vec<i64> = u
        .is_multiple(..)
        .unwrap()
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| m.then_some(i as i64))
        .collect();
    let mut simple = u.clone();
    simple.delete_edges(flagged).unwrap();
    assert!(!simple.has_multiple().unwrap());
    assert!(simple.count_multiple(..).unwrap().iter().all(|&m| m == 1));
}

#[test]
fn loops_like_igraph_example() {
    // examples/simple/igraph_is_loop.c
    let g = Graph::from_flat_edges(&[0, 1, 1, 2, 2, 1, 0, 1, 1, 0, 3, 4, 11, 10], 0, true).unwrap();
    assert!(!g.has_loop().unwrap());
    assert_eq!(g.count_loops().unwrap(), 0);
    assert!(g.is_loop(..).unwrap().iter().all(|&l| !l));

    let u = Graph::from_flat_edges(
        &[0, 0, 1, 1, 2, 2, 2, 3, 2, 4, 2, 5, 2, 6, 2, 2, 0, 0],
        0,
        false,
    )
    .unwrap();
    assert!(u.has_loop().unwrap());
    assert_eq!(u.count_loops().unwrap(), 5);
    assert_eq!(
        u.is_loop(..).unwrap(),
        vec![true, true, true, false, false, false, false, true, true]
    );
    assert_eq!(u.is_loop(&[3, 7]).unwrap(), vec![false, true]);
    assert!(!u.is_simple(true).unwrap());
}

#[test]
fn mutual_edges_and_simplicity() {
    let g = Graph::from_edges(&[(0, 1), (1, 0), (1, 2), (2, 3), (3, 3)], 4, true).unwrap();
    assert_eq!(
        g.is_mutual(.., false).unwrap(),
        vec![true, true, false, false, false]
    );
    assert_eq!(
        g.is_mutual(.., true).unwrap(),
        vec![true, true, false, false, true]
    );
    assert!(g.has_mutual(false).unwrap());

    let oriented = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, true).unwrap();
    assert!(!oriented.has_mutual(true).unwrap());
    assert!(oriented.is_simple(false).unwrap());

    let two_way = Graph::from_edges(&[(0, 1), (1, 0)], 2, true).unwrap();
    assert!(two_way.is_simple(true).unwrap());
    assert!(!two_way.is_simple(false).unwrap());

    // Undirected: mutual iff there are edges.
    assert!(path(3).has_mutual(false).unwrap());
    assert!(!Graph::new(3, false).has_mutual(false).unwrap());
}

#[test]
fn reciprocity_like_igraph_example() {
    // examples/simple/igraph_reciprocity.c
    assert_eq!(
        cycle(100).reciprocity(false, Reciprocity::Default).unwrap(),
        1.0
    );
    let g = Graph::from_flat_edges(&[0, 1, 0, 2, 0, 3, 1, 0, 2, 3, 3, 2], 0, true).unwrap();
    assert_eq!(g.reciprocity(false, Reciprocity::Ratio).unwrap(), 0.5);
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 1)], 0, true).unwrap();
    assert_close(
        g.reciprocity(false, Reciprocity::Default).unwrap(),
        2.0 / 3.0,
        1e-15,
    );
    assert!(
        Graph::new(3, true)
            .reciprocity(false, Reciprocity::Default)
            .unwrap()
            .is_nan()
    );
}

#[test]
fn are_adjacent_follows_directions() {
    let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    assert!(g.are_adjacent(0, 1).unwrap());
    assert!(!g.are_adjacent(1, 0).unwrap());
    assert!(!g.are_adjacent(0, 2).unwrap());
    assert_eq!(
        g.are_adjacent(-1, 0).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
    let k = karate();
    for &(a, b) in KARATE_EDGES.iter() {
        assert!(k.are_adjacent(a, b).unwrap() && k.are_adjacent(b, a).unwrap());
    }
}

// ---------------------------------------------------------------------------
// Graph classes
// ---------------------------------------------------------------------------

#[test]
fn trees_forests_and_acyclicity() {
    let p = path(5);
    assert!(p.is_tree(NeighborMode::All).unwrap());
    assert_eq!(p.tree_root(NeighborMode::All).unwrap(), Some(0));
    assert!(!cycle(5).is_tree(NeighborMode::All).unwrap());
    assert!(!Graph::new(0, false).is_tree(NeighborMode::All).unwrap());
    assert!(Graph::new(0, false).is_forest(NeighborMode::All).unwrap());

    // In-tree towards vertex 3.
    let intree = Graph::from_edges(&[(0, 3), (1, 3), (2, 1)], 4, true).unwrap();
    assert!(!intree.is_tree(NeighborMode::Out).unwrap());
    assert_eq!(intree.tree_root(NeighborMode::In).unwrap(), Some(3));

    let forest = Graph::from_edges(&[(0, 1), (1, 2), (3, 4)], 6, false).unwrap();
    assert!(!forest.is_tree(NeighborMode::All).unwrap());
    assert!(forest.is_forest(NeighborMode::All).unwrap());
    assert_eq!(
        forest.forest_roots(NeighborMode::All).unwrap(),
        Some(vec![0, 3, 5])
    );
    assert_eq!(cycle(4).forest_roots(NeighborMode::All).unwrap(), None);

    // Directed: acyclic but with an undirected cycle.
    let dag = Graph::from_edges(&[(0, 1), (0, 2), (1, 3), (2, 3)], 4, true).unwrap();
    assert!(dag.is_acyclic().unwrap());
    assert!(!dag.is_forest(NeighborMode::All).unwrap());
    let cyc = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, true).unwrap();
    assert!(!cyc.is_acyclic().unwrap());
    // Self-loops are cycles.
    let lp = Graph::from_edges(&[(0, 0)], 1, false).unwrap();
    assert!(!lp.is_acyclic().unwrap());
}

#[test]
fn girth_values() {
    // examples/simple/igraph_girth.c: ring of 100 plus the chord 0-50.
    let mut ring = cycle(100);
    ring.add_edge(0, 50).unwrap();
    assert_eq!(ring.girth().unwrap(), Some(51));
    let (len, circle) = ring.girth_with_cycle().unwrap().unwrap();
    assert_eq!(len, 51);
    assert_eq!(circle.len(), 51);
    // Consecutive cycle vertices are adjacent.
    for i in 0..circle.len() {
        let (a, b) = (circle[i], circle[(i + 1) % circle.len()]);
        assert!(ring.are_adjacent(a, b).unwrap());
    }
    assert_eq!(Graph::new(0, false).girth().unwrap(), None);
    assert_eq!(Graph::new(0, false).girth_with_cycle().unwrap(), None);
    assert_eq!(petersen().girth().unwrap(), Some(5));
    assert_eq!(karate().girth().unwrap(), Some(3));
    // Loops and multi-edges are ignored.
    let g = Graph::from_edges(&[(0, 0), (0, 1), (0, 1), (1, 2)], 3, false).unwrap();
    assert_eq!(g.girth().unwrap(), None);
}

#[test]
fn cliques_and_independent_sets() {
    let g = karate();
    assert!(g.is_clique(&[0, 1, 2, 3, 7], false).unwrap());
    assert!(g.is_clique(&[0, 1, 2, 3, 13], false).unwrap());
    assert!(!g.is_clique(&[0, 1, 2, 3, 7, 13], false).unwrap());
    assert!(g.is_clique(VertexSelector::None, false).unwrap());
    assert!(g.is_independent_vertex_set(&[4, 9, 14]).unwrap());
    assert!(!g.is_independent_vertex_set(&[0, 1]).unwrap());

    let d = Graph::from_edges(&[(0, 1), (1, 0), (1, 2)], 3, true).unwrap();
    assert!(d.is_clique(&[0, 1], true).unwrap());
    assert!(!d.is_clique(&[1, 2], true).unwrap());
    assert!(d.is_clique(&[1, 2], false).unwrap());
}

#[test]
fn perfect_graphs() {
    assert!(!cycle(5).is_perfect().unwrap());
    assert!(!cycle(7).is_perfect().unwrap());
    assert!(cycle(6).is_perfect().unwrap());
    assert!(complete(6).is_perfect().unwrap());
    assert!(!petersen().is_perfect().unwrap()); // contains induced 5-cycles
    let directed = Graph::from_edges(&[(0, 1)], 2, true).unwrap();
    assert_eq!(
        directed.is_perfect().unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    let multi = Graph::from_edges(&[(0, 1), (0, 1)], 2, false).unwrap();
    assert_eq!(
        multi.is_perfect().unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn chordality_like_igraph_unit_test() {
    // tests/unit/igraph_is_chordal.c
    assert!(Graph::new(0, false).is_chordal().unwrap());
    assert!(Graph::new(1, false).is_chordal().unwrap());
    let g = Graph::from_flat_edges(
        &[0, 1, 0, 2, 1, 1, 1, 3, 2, 0, 2, 0, 2, 3, 3, 4, 3, 4],
        6,
        false,
    )
    .unwrap();
    assert!(!g.is_chordal().unwrap());
    let c = g.is_chordal_with(None, None).unwrap();
    assert!(!c.is_chordal);
    assert_eq!(c.fill_in, vec![(3, 0)]);
    assert_eq!(c.triangulated.vcount(), 6);
    assert_eq!(c.triangulated.ecount(), 10);
    assert!(c.triangulated.is_chordal().unwrap());

    let mcs = g.maximum_cardinality_search().unwrap();
    let mut sorted = mcs.alpha.clone();
    sorted.sort();
    assert_eq!(sorted, (0..6).collect::<Vec<_>>());
    for (v, &r) in mcs.alpha.iter().enumerate() {
        assert_eq!(mcs.alpham1[r as usize], v as i64);
    }
    let with_both = g
        .is_chordal_with(Some(&mcs.alpha), Some(&mcs.alpham1))
        .unwrap();
    assert_eq!(with_both, c);
    let with_alpha = g.is_chordal_with(Some(&mcs.alpha), None).unwrap();
    assert_eq!(with_alpha.fill_in, c.fill_in);
    let with_inv = g.is_chordal_with(None, Some(&mcs.alpham1)).unwrap();
    assert_eq!(with_inv.fill_in, c.fill_in);

    let err = g.is_chordal_with(Some(&[]), None).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    // Out-of-range or repeated ranks are rejected before reaching igraph.
    for bad in [[0, 1, 2, 3, 4, 99], [0, 0, 1, 2, 3, 4], [-1, 0, 1, 2, 3, 4]] {
        let err = g.is_chordal_with(Some(&bad), None).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidValue);
        let err = g.is_chordal_with(None, Some(&bad)).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidValue);
    }
    // Two valid but mutually inconsistent permutations.
    let id: Vec<i64> = (0..6).collect();
    let rot: Vec<i64> = (0..6).map(|i| (i + 1) % 6).collect();
    assert!(g.is_chordal_with(Some(&id), Some(&rot)).is_err());

    // Chordal graphs are perfect.
    let fan = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)], 4, false).unwrap();
    assert!(fan.is_chordal().unwrap() && fan.is_perfect().unwrap());
}

#[test]
fn subcomponent_reachability() {
    let g = Graph::from_edges(&[(0, 1), (1, 2), (3, 1), (4, 5)], 6, true).unwrap();
    let sorted = |mut v: Vec<i64>| {
        v.sort();
        v
    };
    assert_eq!(
        sorted(g.subcomponent(0, NeighborMode::Out).unwrap()),
        vec![0, 1, 2]
    );
    assert_eq!(
        sorted(g.subcomponent(1, NeighborMode::In).unwrap()),
        vec![0, 1, 3]
    );
    assert_eq!(
        sorted(g.subcomponent(2, NeighborMode::All).unwrap()),
        vec![0, 1, 2, 3]
    );
    assert_eq!(g.subcomponent(0, NeighborMode::Out).unwrap()[0], 0);
    assert_eq!(
        g.subcomponent(6, NeighborMode::Out).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        karate().subcomponent(5, NeighborMode::All).unwrap().len(),
        34
    );
}

// ---------------------------------------------------------------------------
// Spanning trees
// ---------------------------------------------------------------------------

#[test]
fn minimum_spanning_tree_algorithms_agree() {
    let g = karate();
    let w: Vec<f64> = (0..78)
        .map(|i| ((i * 37) % 11) as f64 + 0.25 * (i % 3) as f64)
        .collect();
    let total = |t: &[i64]| t.iter().map(|&e| w[e as usize]).sum::<f64>();
    let prim = g
        .minimum_spanning_tree(Some(&w), MstAlgorithm::Prim)
        .unwrap();
    let kruskal = g
        .minimum_spanning_tree(Some(&w), MstAlgorithm::Kruskal)
        .unwrap();
    let auto = g
        .minimum_spanning_tree(Some(&w), MstAlgorithm::Automatic)
        .unwrap();
    assert_eq!(prim.len(), 33);
    assert_close(total(&prim), total(&kruskal), 1e-9);
    assert_close(total(&prim), total(&auto), 1e-9);
    assert!(edge_subgraph(&g, &prim).is_tree(NeighborMode::All).unwrap());

    // A maximum spanning tree is at least as heavy as any spanning tree.
    let neg: Vec<f64> = w.iter().map(|x| -x).collect();
    let max = g
        .minimum_spanning_tree(Some(&neg), MstAlgorithm::Automatic)
        .unwrap();
    let any = g
        .minimum_spanning_tree(None, MstAlgorithm::Unweighted)
        .unwrap();
    assert!(total(&max) >= total(&any) && total(&any) >= total(&prim));

    // Disconnected graph: a spanning forest with n - c edges.
    let f = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4)], 6, false).unwrap();
    assert_eq!(
        f.minimum_spanning_tree(None, MstAlgorithm::Automatic)
            .unwrap()
            .len(),
        3
    );

    let err = g
        .minimum_spanning_tree(Some(&[1.0]), MstAlgorithm::Prim)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

#[test]
fn random_spanning_trees_are_uniform_on_a_square() {
    // A 4-cycle has exactly 4 spanning trees: each omits one edge.
    rng::seed(42).unwrap();
    let g = cycle(4);
    let mut counts = [0usize; 4];
    let samples = 4000;
    for _ in 0..samples {
        let tree = g.random_spanning_tree(None).unwrap();
        assert_eq!(tree.len(), 3);
        assert!(edge_subgraph(&g, &tree).is_tree(NeighborMode::All).unwrap());
        let missing = (0..4).find(|e| !tree.contains(e)).unwrap();
        counts[missing as usize] += 1;
    }
    for c in counts {
        assert!((850..1150).contains(&c), "{counts:?}");
    }
}

#[test]
fn random_spanning_tree_of_one_component() {
    rng::seed(1).unwrap();
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], 6, false).unwrap();
    let t = g.random_spanning_tree(Some(4)).unwrap();
    assert_eq!(t.len(), 2);
    assert!(t.iter().all(|&e| e >= 3));
    assert_eq!(g.random_spanning_tree(None).unwrap().len(), 4);
    for bad in [-3, 6] {
        let err = g.random_spanning_tree(Some(bad)).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    }
}

#[test]
fn unfold_tree_like_igraph_unit_test() {
    // tests/unit/igraph_unfold_tree.c
    let flat = [0, 1, 0, 2, 1, 1, 1, 3, 2, 0, 2, 3, 3, 4, 3, 4];
    let g = Graph::from_flat_edges(&flat, 6, true).unwrap();
    let u = g.unfold_tree(NeighborMode::Out, &[0]).unwrap();
    assert!(u.tree.is_directed());
    assert_eq!(u.tree.vcount(), 10);
    assert_eq!(u.tree.ecount(), 8);
    assert_eq!(u.vertex_index, vec![0, 1, 2, 3, 4, 5, 1, 0, 3, 4]);

    let ug = Graph::from_flat_edges(&flat, 6, false).unwrap();
    let u = ug.unfold_tree(NeighborMode::All, &[0]).unwrap();
    assert_eq!(u.vertex_index, vec![0, 1, 2, 3, 4, 5, 2, 1, 3, 4]);
    // The unfolded graph has no cycles: a forest (isolated vertex 5 included).
    assert!(u.tree.is_forest(NeighborMode::All).unwrap());

    let empty = Graph::new(0, false)
        .unfold_tree(NeighborMode::All, &[])
        .unwrap();
    assert_eq!(empty.tree.vcount(), 0);

    let err = g.unfold_tree(NeighborMode::Out, &[-1, 0]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
}

// ---------------------------------------------------------------------------
// Degree correlations
// ---------------------------------------------------------------------------

#[test]
fn knn_of_karate_matches_igraph_example() {
    // examples/simple/igraph_avg_nearest_neighbor_degree.out
    let g = karate();
    let nd = g
        .avg_nearest_neighbor_degree(.., NeighborMode::All, NeighborMode::All, None)
        .unwrap();
    let knn = [
        4.3125, 5.77778, 6.6, 7.66667, 7.66667, 6.25, 6.25, 10.25, 11.8, 13.5, 7.66667, 16.0, 11.0,
        11.6, 14.5, 14.5, 4.0, 12.5, 14.5, 14.0, 14.5, 12.5, 14.5, 8.0, 4.33333, 4.66667, 10.5,
        8.75, 11.0, 9.0, 10.75, 9.0, 5.08333, 3.82353,
    ];
    let nan = f64::NAN;
    let knnk = [
        16.0, 12.4091, 8.22222, 8.54167, 10.4667, 8.33333, nan, nan, 5.77778, 6.6, nan, 5.08333,
        nan, nan, nan, 4.3125, 3.82353,
    ];
    assert_vec_close(&nd.knn, &knn, 1e-4);
    assert_vec_close(&nd.knnk, &knnk, 1e-4);

    // Subset of vertices.
    let sub = g
        .avg_nearest_neighbor_degree(&[0, 33], NeighborMode::All, NeighborMode::All, None)
        .unwrap();
    assert_vec_close(&sub.knn, &[4.3125, 3.82353], 1e-4);
}

#[test]
fn degree_correlation_vector_agrees_with_knnk() {
    // For undirected graphs, k_nn(k) from both functions coincides (with an
    // index shift: avg_nearest_neighbor_degree starts at degree 1).
    let g = karate();
    let nd = g
        .avg_nearest_neighbor_degree(.., NeighborMode::All, NeighborMode::All, None)
        .unwrap();
    let dcv = g
        .degree_correlation_vector(None, NeighborMode::All, NeighborMode::All, true)
        .unwrap();
    assert_eq!(dcv.len(), nd.knnk.len() + 1);
    assert_vec_close(&dcv[1..], &nd.knnk, 1e-9);

    // Weighted variant runs and has the right shape.
    let w = vec![1.0; 78];
    let dcvw = g
        .degree_correlation_vector(Some(&w), NeighborMode::All, NeighborMode::All, true)
        .unwrap();
    assert_vec_close(&dcvw, &dcv, 1e-12);
}

#[test]
fn rich_club_matches_igraph_unit_test() {
    // tests/unit/rich_club.c, test 3
    let g = Graph::from_edges(
        &[
            (0, 3),
            (1, 3),
            (2, 3),
            (4, 3),
            (5, 3),
            (5, 6),
            (1, 2),
            (2, 5),
        ],
        7,
        false,
    )
    .unwrap();
    let order: Vec<i64> = (0..7).collect();
    let seq = g
        .rich_club_sequence(None, &order, true, false, false)
        .unwrap();
    let nan = f64::NAN;
    assert_vec_close(
        &seq,
        &[0.380952, 0.466667, 0.5, 0.5, 0.333333, 1.0, nan],
        1e-6,
    );
    let rev: Vec<i64> = order.iter().rev().copied().collect();
    let seq = g
        .rich_club_sequence(None, &rev, true, false, false)
        .unwrap();
    assert_vec_close(
        &seq,
        &[0.380952, 0.466667, 0.5, 0.666667, 0.333333, 0.0, nan],
        1e-6,
    );
    // Unnormalized: remaining edge counts.
    let counts = g
        .rich_club_sequence(None, &order, false, false, false)
        .unwrap();
    assert_eq!(counts[0], 8.0);
    assert_eq!(*counts.last().unwrap(), 0.0);
    assert!(counts.windows(2).all(|p| p[0] >= p[1]));

    let err = g
        .rich_club_sequence(None, &[0, 1], true, false, false)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

// ---------------------------------------------------------------------------
// Laplacian
// ---------------------------------------------------------------------------

#[test]
fn laplacian_like_igraph_example() {
    // examples/simple/igraph_get_laplacian.c: directed weighted 5-ring.
    let ring = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 5, true).unwrap();
    let w = [1.0, 2.0, 3.0, 4.0, 5.0];
    let l = ring
        .get_laplacian(
            NeighborMode::Out,
            LaplacianNormalization::Symmetric,
            Some(&w),
        )
        .unwrap();
    let expected = [
        [1.0, -std::f64::consts::FRAC_1_SQRT_2, 0.0, 0.0, 0.0],
        [0.0, 1.0, -0.816497, 0.0, 0.0],
        [0.0, 0.0, 1.0, -0.866025, 0.0],
        [0.0, 0.0, 0.0, 1.0, -0.894427],
        [-2.23607, 0.0, 0.0, 0.0, 1.0],
    ];
    for (i, row) in expected.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            assert_close(l[(i, j)], x, 1e-5);
        }
    }
}

#[test]
fn laplacian_rows_sum_to_zero_and_spectrum_trace() {
    let g = karate();
    let l = g
        .get_laplacian(
            NeighborMode::All,
            LaplacianNormalization::Unnormalized,
            None,
        )
        .unwrap();
    assert_eq!(l.shape(), (34, 34));
    for row in l.rows() {
        assert_close(row.iter().sum::<f64>(), 0.0, 1e-12);
    }
    // Trace = sum of degrees = 2|E|.
    let trace: f64 = (0..34).map(|i| l[(i, i)]).sum();
    assert_eq!(trace, 156.0);
    // Symmetric normalization has unit diagonal (no isolated vertices).
    let ls = g
        .get_laplacian(NeighborMode::All, LaplacianNormalization::Symmetric, None)
        .unwrap();
    assert!((0..34).all(|i| (ls[(i, i)] - 1.0).abs() < 1e-12));
    // Left normalization: rows sum to zero.
    let ll = g
        .get_laplacian(NeighborMode::All, LaplacianNormalization::Left, None)
        .unwrap();
    for row in ll.rows() {
        assert_close(row.iter().sum::<f64>(), 0.0, 1e-12);
    }
    // Right normalization: columns sum to zero.
    let lr = g
        .get_laplacian(NeighborMode::All, LaplacianNormalization::Right, None)
        .unwrap();
    for col in lr.columns() {
        assert_close(col.iter().sum::<f64>(), 0.0, 1e-12);
    }
}

#[test]
fn sparse_laplacian_equals_dense() {
    let mut multi = karate();
    multi.add_edges(&[(0, 1), (5, 5), (2, 2)]).unwrap();
    let directed =
        Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3), (3, 3), (0, 1)], 4, true).unwrap();
    for g in [karate(), multi, directed] {
        let m = g.ecount();
        let w: Vec<f64> = (0..m).map(|i| 1.0 + (i % 4) as f64).collect();
        for mode in [NeighborMode::Out, NeighborMode::In, NeighborMode::All] {
            for norm in [
                LaplacianNormalization::Unnormalized,
                LaplacianNormalization::Symmetric,
                LaplacianNormalization::Left,
                LaplacianNormalization::Right,
            ] {
                for weights in [None, Some(&w[..])] {
                    let dense = g.get_laplacian(mode, norm, weights).unwrap();
                    let sparse = g.get_laplacian_sparse(mode, norm, weights).unwrap();
                    let mut rebuilt = Matrix::zeros(g.vcount(), g.vcount());
                    for &(i, j, x) in &sparse {
                        assert_ne!(x, 0.0);
                        rebuilt[(i as usize, j as usize)] = x;
                    }
                    for i in 0..g.vcount() {
                        for j in 0..g.vcount() {
                            assert_close(rebuilt[(i, j)], dense[(i, j)], 1e-12);
                        }
                    }
                    // Sorted and without duplicates.
                    assert!(
                        sparse
                            .windows(2)
                            .all(|p| (p[0].0, p[0].1) < (p[1].0, p[1].1))
                    );
                }
            }
        }
    }
}

#[test]
fn weight_length_is_validated() {
    let g = path(4);
    let bad = [1.0, 2.0];
    let kinds = [
        g.density(Some(&bad), false).unwrap_err().kind(),
        g.strength(.., NeighborMode::All, Loops::Twice, Some(&bad))
            .unwrap_err()
            .kind(),
        g.get_laplacian(
            NeighborMode::All,
            LaplacianNormalization::Unnormalized,
            Some(&bad),
        )
        .unwrap_err()
        .kind(),
        g.get_laplacian_sparse(
            NeighborMode::All,
            LaplacianNormalization::Unnormalized,
            Some(&bad),
        )
        .unwrap_err()
        .kind(),
        g.degree_correlation_vector(Some(&bad), NeighborMode::All, NeighborMode::All, true)
            .unwrap_err()
            .kind(),
        g.avg_nearest_neighbor_degree(.., NeighborMode::All, NeighborMode::All, Some(&bad))
            .unwrap_err()
            .kind(),
    ];
    assert!(kinds.iter().all(|&k| k == ErrorKind::InvalidValue));
}

#[test]
fn laplacian_normalization_roundtrip() {
    use igraph::ffi::igraph_laplacian_normalization_t;
    for n in [
        LaplacianNormalization::Unnormalized,
        LaplacianNormalization::Symmetric,
        LaplacianNormalization::Left,
        LaplacianNormalization::Right,
    ] {
        let raw: igraph_laplacian_normalization_t = n.into();
        assert_eq!(LaplacianNormalization::try_from(raw).unwrap(), n);
    }
    assert_eq!(
        LaplacianNormalization::default(),
        LaplacianNormalization::Unnormalized
    );
    assert!(LaplacianNormalization::try_from(42).is_err());
}

// ---------------------------------------------------------------------------
// Use-case stories
// ---------------------------------------------------------------------------

/// Story: a regional utility must connect six towns with a fibre network.
/// Laying cable costs proportionally to the distance; the cheapest network
/// that connects everybody is a minimum spanning tree. We check it against
/// a hand computation and then ask how *robust* it is: a tree has no cycles
/// (girth = None) and every cable is a single point of failure, while the
/// full road graph is 2-connected enough to have a short girth.
#[test]
fn use_case_cheapest_fibre_network() {
    let towns = ["Arezzo", "Siena", "Firenze", "Pisa", "Lucca", "Livorno"];
    let roads: [(i64, i64, f64); 10] = [
        (0, 1, 60.0), // Arezzo - Siena
        (0, 2, 80.0), // Arezzo - Firenze
        (1, 2, 70.0), // Siena - Firenze
        (1, 3, 110.0),
        (2, 3, 85.0), // Firenze - Pisa
        (2, 4, 75.0), // Firenze - Lucca
        (3, 4, 20.0), // Pisa - Lucca
        (3, 5, 25.0), // Pisa - Livorno
        (4, 5, 45.0),
        (1, 5, 120.0),
    ];
    let edges: Vec<(i64, i64)> = roads.iter().map(|&(a, b, _)| (a, b)).collect();
    let cost: Vec<f64> = roads.iter().map(|&(_, _, c)| c).collect();
    let g = Graph::from_edges(&edges, towns.len(), false).unwrap();

    assert!(g.is_simple(true).unwrap());
    assert_eq!(g.girth().unwrap(), Some(3));

    let mst = g
        .minimum_spanning_tree(Some(&cost), MstAlgorithm::Automatic)
        .unwrap();
    let total: f64 = mst.iter().map(|&e| cost[e as usize]).sum();
    // Pisa-Lucca 20, Pisa-Livorno 25, Arezzo-Siena 60, Siena-Firenze 70, Firenze-Lucca 75.
    assert_eq!(total, 250.0);
    let network = edge_subgraph(&g, &mst);
    assert!(network.is_tree(NeighborMode::All).unwrap());
    assert_eq!(network.girth().unwrap(), None);

    // The busiest hub of the fibre network.
    let hubs = network
        .sort_vertex_ids_by_degree(
            ..,
            NeighborMode::All,
            Loops::Twice,
            Order::Descending,
            false,
        )
        .unwrap();
    let hub_degree = network
        .maxdegree(.., NeighborMode::All, Loops::Twice)
        .unwrap();
    assert_eq!(hub_degree, 2);
    assert!(["Siena", "Firenze", "Pisa", "Lucca"].contains(&towns[hubs[0] as usize]));

    // Monthly traffic (strength) through each town if every cable carries its cost in load.
    let tree_cost: Vec<f64> = mst.iter().map(|&e| cost[e as usize]).collect();
    let load = network
        .strength(.., NeighborMode::All, Loops::Twice, Some(&tree_cost))
        .unwrap();
    assert_eq!(load.iter().sum::<f64>(), 2.0 * total);
}

/// Story: in Zachary's karate club, are the leaders (the instructor, vertex
/// 0, and the administrator, vertex 33) talking mostly to low-degree members?
/// A decreasing `k_nn(k)` reveals a *disassortative* network, and the
/// rich-club sequence shows how the core densifies as peripheral members are
/// peeled away.
#[test]
fn use_case_karate_club_hubs() {
    let g = karate();
    let deg = g.degree(.., NeighborMode::All, Loops::Twice).unwrap();
    let nd = g
        .avg_nearest_neighbor_degree(.., NeighborMode::All, NeighborMode::All, None)
        .unwrap();
    // Leaders have neighbors of low average degree...
    assert!(nd.knn[0] < g.mean_degree(true).unwrap());
    assert!(nd.knn[33] < g.mean_degree(true).unwrap());
    // ...while the lone degree-1 member (vertex 11) hangs on the instructor.
    assert_eq!(deg[11], 1);
    assert_eq!(nd.knn[11], 16.0);
    // k_nn(1) > k_nn(17): disassortative mixing.
    assert!(nd.knnk[0] > *nd.knnk.last().unwrap());

    // Peel vertices by increasing degree: the remaining core becomes denser.
    let order = g
        .sort_vertex_ids_by_degree(.., NeighborMode::All, Loops::Twice, Order::Ascending, false)
        .unwrap();
    let seq = g
        .rich_club_sequence(None, &order, true, false, false)
        .unwrap();
    assert_close(seq[0], g.density(None, false).unwrap(), 1e-12);
    // The 5 highest-degree members (0, 1, 2, 32, 33) ...
    let core_density = seq[34 - 5];
    assert!(core_density > 3.0 * seq[0]);
    // ...and the two leaders are not directly connected.
    assert!(!g.are_adjacent(0, 33).unwrap());
    assert!(g.is_independent_vertex_set(&[0, 33]).unwrap());
}

// ---------------------------------------------------------------------------
// Review additions: edge cases and error paths
// ---------------------------------------------------------------------------

#[test]
fn count_multiple_1_agrees_with_count_multiple_on_loops() {
    // The count_multiple_1 of igraph 1.0.0 and 1.0.1 counts an undirected
    // self-loop twice (see `upstream_quirks_are_still_present_in_igraph_1_0_1`);
    // the wrapper corrects it so that both functions always agree.
    for directed in [false, true] {
        let g = Graph::from_edges(
            &[(0, 0), (0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (2, 2)],
            3,
            directed,
        )
        .unwrap();
        let all = g.count_multiple(..).unwrap();
        for e in 0..g.ecount() as i64 {
            assert_eq!(g.count_multiple_1(e).unwrap(), all[e as usize], "{e}");
        }
        assert_eq!(all[0], 2);
    }
    assert_eq!(
        path(3).count_multiple_1(-1).unwrap_err().kind(),
        ErrorKind::InvalidEdgeId
    );
}

#[test]
fn weighted_knn_follows_barrat() {
    // Path 1 - 0 - 2 with weights 1 and 3: k_nn,w(0) = (1*1 + 3*1) / 4 = 1,
    // the leaves see the center of degree 2.
    let g = Graph::from_edges(&[(0, 1), (0, 2)], 3, false).unwrap();
    let nd = g
        .avg_nearest_neighbor_degree(.., NeighborMode::All, NeighborMode::All, Some(&[1.0, 3.0]))
        .unwrap();
    assert_eq!(nd.knn, vec![1.0, 2.0, 2.0]);
    // Directed: out-neighbors' in-degrees.
    let d = Graph::from_edges(&[(0, 1), (0, 2), (1, 2)], 3, true).unwrap();
    let nd = d
        .avg_nearest_neighbor_degree(.., NeighborMode::Out, NeighborMode::In, None)
        .unwrap();
    assert_eq!(nd.knn[0], 1.5); // in-degrees of 1 and 2 are 1 and 2
    assert_eq!(nd.knn[1], 2.0);
    assert!(nd.knn[2].is_nan()); // no out-neighbors
}

#[test]
fn laplacian_error_paths() {
    let g = Graph::from_edges(&[(0, 1)], 3, true).unwrap();
    // Vertex 1 has an in-edge but zero out-degree.
    for f in [
        |g: &Graph| {
            g.get_laplacian(NeighborMode::Out, LaplacianNormalization::Symmetric, None)
                .map(|_| ())
        },
        |g: &Graph| {
            g.get_laplacian_sparse(NeighborMode::Out, LaplacianNormalization::Symmetric, None)
                .map(|_| ())
        },
    ] {
        assert_eq!(f(&g).unwrap_err().kind(), ErrorKind::InvalidValue);
    }
    let neg = [-1.0];
    assert_eq!(
        g.get_laplacian_sparse(
            NeighborMode::All,
            LaplacianNormalization::Unnormalized,
            Some(&neg)
        )
        .unwrap_err()
        .kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.get_laplacian(
            NeighborMode::All,
            LaplacianNormalization::Unnormalized,
            Some(&[f64::NAN])
        )
        .unwrap_err()
        .kind(),
        ErrorKind::InvalidValue
    );
    // The null graph has an empty Laplacian.
    let e = Graph::new(0, false);
    assert!(
        e.get_laplacian_sparse(NeighborMode::All, LaplacianNormalization::Symmetric, None)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn rich_club_rejects_non_permutations_and_diversity_negative_weights() {
    let g = cycle(4);
    for bad in [[0, 0, 1, 2], [0, 1, 2, 4], [-1, 0, 1, 2]] {
        let err = g
            .rich_club_sequence(None, &bad, true, false, false)
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidValue);
    }
    let err = g.diversity(&[1.0, -1.0, 1.0, 1.0], ..).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    // Weighted rich club without normalization: remaining total weight.
    let w = [1.0, 2.0, 3.0, 4.0];
    let seq = g
        .rich_club_sequence(Some(&w), &[0, 1, 2, 3], false, false, false)
        .unwrap();
    // Edges 0-1 (1), 1-2 (2), 2-3 (3), 3-0 (4).
    assert_eq!(seq, vec![10.0, 5.0, 3.0, 0.0]);
}

#[test]
fn chordality_of_directed_input_ignores_directions() {
    let square = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], 4, true).unwrap();
    let c = square.is_chordal_with(None, None).unwrap();
    assert!(!c.is_chordal);
    assert_eq!(c.fill_in.len(), 1);
    assert!(c.triangulated.is_directed());
    assert_eq!(c.triangulated.ecount(), 5);
    // The first edges of the triangulated graph are the original ones.
    for e in 0..4 {
        assert_eq!(c.triangulated.edge(e).unwrap(), square.edge(e).unwrap());
    }
    assert!(c.triangulated.is_chordal().unwrap());
    // Any tree is chordal and needs no fill-in.
    let t = path(6).is_chordal_with(None, None).unwrap();
    assert!(t.is_chordal && t.fill_in.is_empty());
}

// ---------------------------------------------------------------------------
// igraph 1.0.1 unit tests, cross-module identities and threads
// ---------------------------------------------------------------------------

#[test]
fn karate_helper_is_zacharys_graph() {
    // The shared `karate()` helper and the famous graph agree edge by edge.
    let famous = Graph::famous("Zachary").unwrap();
    assert_eq!(famous.edge_list(), karate().edge_list());
    assert_eq!(famous.count_multiple(..).unwrap(), vec![1; 78]);
}

#[test]
fn minimum_spanning_tree_matches_igraph_unit_test() {
    // tests/unit/minimum_spanning_tree.c (+ .out): a seeded G(50, 100)
    // multigraph with self-loops and uniform random weights.
    rng::seed(77685).unwrap();
    let g = Graph::erdos_renyi_game_gnm(50, 100, false, AllowedEdgeTypes::ALL, false).unwrap();
    let w: Vec<f64> = (0..g.ecount()).map(|_| rng::uniform01()).collect();
    let comps = g.connected_components(Connectedness::Weak).unwrap().count;
    let weighted = [
        0, 1, 5, 6, 7, 8, 13, 16, 17, 23, 25, 26, 28, 31, 32, 34, 36, 38, 39, 42, 44, 46, 50, 52,
        58, 59, 60, 62, 63, 64, 67, 69, 70, 71, 73, 75, 78, 83, 84, 85, 87, 91, 92, 93, 94, 97, 98,
    ];
    for method in [
        MstAlgorithm::Automatic,
        MstAlgorithm::Prim,
        MstAlgorithm::Kruskal,
    ] {
        let mut t = g.minimum_spanning_tree(Some(&w), method).unwrap();
        t.sort();
        assert_eq!(t, weighted, "{method:?}");
        let total: f64 = t.iter().map(|&e| w[e as usize]).sum();
        assert_close(total, 14.6873, 1e-4);
        assert!(edge_subgraph(&g, &t).is_forest(NeighborMode::All).unwrap());
        assert_eq!(t.len(), 50 - comps);
    }
    let mut t = g
        .minimum_spanning_tree(Some(&w), MstAlgorithm::Unweighted)
        .unwrap();
    t.sort();
    assert_eq!(
        t,
        [
            0, 1, 2, 6, 8, 12, 13, 14, 19, 25, 28, 29, 30, 31, 33, 36, 39, 40, 41, 42, 43, 46, 47,
            51, 54, 56, 59, 60, 61, 62, 63, 65, 66, 70, 71, 72, 75, 77, 83, 84, 90, 91, 93, 94, 97,
            98, 99,
        ]
    );
    assert!(edge_subgraph(&g, &t).is_forest(NeighborMode::All).unwrap());
}

#[test]
fn random_spanning_tree_like_igraph_unit_test() {
    // tests/unit/random_spanning_tree.c
    rng::seed(987).unwrap();
    // Guaranteed to be connected: every new vertex cites 2 older ones.
    let opts = BarabasiOptions::default().with_m(2).with_power(2.0);
    let g = Graph::barabasi_game(100, &opts).unwrap();
    assert!(g.is_connected(Connectedness::Weak).unwrap());
    let t = g.random_spanning_tree(Some(0)).unwrap();
    assert_eq!(t.len(), g.vcount() - 1);
    assert!(edge_subgraph(&g, &t).is_tree(NeighborMode::All).unwrap());

    // A forest has a single spanning forest: itself.
    let f = Graph::from_edges(&[(0, 1), (2, 3)], 4, false).unwrap();
    assert_eq!(f.random_spanning_tree(Some(0)).unwrap(), vec![0]);
    assert_eq!(f.random_spanning_tree(None).unwrap(), vec![0, 1]);
}

#[test]
fn seeded_random_spanning_trees_are_reproducible_in_parallel_threads() {
    // Every thread has its own default generator: seeding one thread does
    // not disturb the others, so each thread sees the same sequence.
    let sample = |seed: u64| {
        rng::seed(seed).unwrap();
        let g = Graph::famous("Zachary").unwrap();
        (0..20)
            .map(|_| g.random_spanning_tree(None).unwrap())
            .collect::<Vec<_>>()
    };
    let reference = sample(2024);
    let handles: Vec<_> = (0..8)
        .map(|i| std::thread::spawn(move || sample(if i % 2 == 0 { 2024 } else { 7 })))
        .collect();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    for (i, r) in results.iter().enumerate() {
        if i % 2 == 0 {
            assert_eq!(r, &reference);
        } else {
            assert_eq!(r, &results[1]);
        }
    }
    assert_ne!(results[1], reference);
}

#[test]
fn laplacian_spectrum_counts_components_and_spanning_trees() {
    // Petersen graph: spectrum {0, 2^5, 5^4}, hence 2^5 * 5^4 / 10 = 2000
    // spanning trees (Kirchhoff's matrix-tree theorem).
    let spec = laplacian_spectrum(&petersen());
    let expected = [0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0];
    assert_vec_close(&spec, &expected, 1e-9);
    assert_close(spec[1..].iter().product::<f64>() / 10.0, 2000.0, 1e-6);

    // Cayley's formula: K_n has n^(n-2) spanning trees.
    for n in 3..8usize {
        let spec = laplacian_spectrum(&Graph::full(n, false, false).unwrap());
        let trees = spec[1..].iter().product::<f64>() / n as f64;
        assert_close(trees, (n as f64).powi(n as i32 - 2), 1e-6 * trees);
    }

    // The multiplicity of the zero eigenvalue is the number of components.
    let g = petersen()
        .disjoint_union(&cycle(4))
        .unwrap()
        .disjoint_union(&Graph::new(2, false))
        .unwrap();
    let zeros = laplacian_spectrum(&g)
        .iter()
        .filter(|x| x.abs() < 1e-9)
        .count();
    let comps = g.connected_components(Connectedness::Weak).unwrap().count;
    assert_eq!((zeros, comps), (4, 4));
}

#[test]
fn sparsemat_laplacian_equals_dense() {
    let directed =
        Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3), (3, 3), (0, 1)], 4, true).unwrap();
    for g in [karate(), directed] {
        for mode in [NeighborMode::Out, NeighborMode::In, NeighborMode::All] {
            let dense = g
                .get_laplacian(mode, LaplacianNormalization::Left, None)
                .unwrap();
            let sparse = g
                .get_laplacian_sparsemat(mode, LaplacianNormalization::Left, None)
                .unwrap()
                .to_dense()
                .unwrap();
            assert_eq!(sparse.shape(), dense.shape());
            for i in 0..g.vcount() {
                for j in 0..g.vcount() {
                    assert_close(sparse[(i, j)], dense[(i, j)], 1e-12);
                }
            }
        }
    }
    assert_eq!(
        path(3)
            .get_laplacian_sparsemat(
                NeighborMode::All,
                LaplacianNormalization::Unnormalized,
                Some(&[1.0])
            )
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn girth_agrees_with_cycle_bases_and_generalized_petersen_graphs() {
    use igraph::cycles::MinimumCycleBasisOptions;
    // Known girths: Petersen GP(5,2) = 5, dodecahedron GP(10,2) = 5,
    // Moebius-Kantor GP(8,3) = 6, Desargues GP(10,3) = 6; prism GP(n,1) = 4.
    for (n, k, girth) in [(5, 2, 5), (10, 2, 5), (8, 3, 6), (10, 3, 6), (6, 1, 4)] {
        let g = Graph::generalized_petersen(n, k).unwrap();
        assert_eq!(g.girth().unwrap(), Some(girth), "GP({n},{k})");
        let (len, cyc) = g.girth_with_cycle().unwrap().unwrap();
        assert_eq!(len, cyc.len());
        let basis = g
            .minimum_cycle_basis(&MinimumCycleBasisOptions::default())
            .unwrap();
        assert_eq!(basis.iter().map(Vec::len).min(), Some(girth));
    }
    for name in ["Zachary", "Heawood", "Coxeter", "Tutte"] {
        let g = Graph::famous(name).unwrap();
        let basis = g
            .minimum_cycle_basis(&MinimumCycleBasisOptions::default())
            .unwrap();
        assert_eq!(
            g.girth().unwrap(),
            basis.iter().map(Vec::len).min(),
            "{name}"
        );
    }
    // Heawood graph: the (3,6)-cage; Coxeter graph: girth 7.
    assert_eq!(Graph::famous("Heawood").unwrap().girth().unwrap(), Some(6));
    assert_eq!(Graph::famous("Coxeter").unwrap().girth().unwrap(), Some(7));
}

#[test]
fn graph_classes_agree_with_other_modules() {
    rng::seed(3).unwrap();
    for _ in 0..20 {
        let g = Graph::erdos_renyi_game_gnp(30, 0.06, false, EdgeTypeSw::Simple, false).unwrap();
        let cc = g.connected_components(Connectedness::Weak).unwrap();
        // A simple graph is a forest iff |E| = |V| - #components.
        assert_eq!(
            g.is_forest(NeighborMode::All).unwrap(),
            g.ecount() == g.vcount() - cc.count
        );
        assert_eq!(
            g.is_tree(NeighborMode::All).unwrap(),
            cc.count == 1 && g.ecount() == g.vcount() - 1
        );
        if let Some(roots) = g.forest_roots(NeighborMode::All).unwrap() {
            assert_eq!(roots.len(), cc.count);
        }
        // subcomponent is the connected component of the vertex.
        for v in [0, 7, 29] {
            let mut sub = g.subcomponent(v, NeighborMode::All).unwrap();
            sub.sort();
            let c = cc.membership[v as usize];
            let members: Vec<i64> = (0..30)
                .filter(|&u| cc.membership[u as usize] == c)
                .collect();
            assert_eq!(sub, members);
        }
        // Undirected acyclicity is being a forest (the graph is simple).
        assert_eq!(
            g.is_acyclic().unwrap(),
            g.is_forest(NeighborMode::All).unwrap()
        );
    }
    // Directed acyclicity: a DAG has a topological order.
    let dag = Graph::from_edges(&[(0, 1), (0, 2), (1, 3), (2, 3)], 4, true).unwrap();
    assert!(dag.is_acyclic().unwrap() && dag.is_dag().unwrap());
    assert_eq!(dag.topological_sorting(NeighborMode::Out).unwrap().len(), 4);
    let cyc = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, true).unwrap();
    assert!(!cyc.is_acyclic().unwrap() && !cyc.is_dag().unwrap());
    assert!(cyc.find_cycle(NeighborMode::Out).unwrap().is_some());
    // A k-ary tree is an out-tree rooted at 0.
    let t = Graph::kary_tree(40, 3, TreeMode::Out).unwrap();
    assert_eq!(t.tree_root(NeighborMode::Out).unwrap(), Some(0));
    assert_eq!(t.tree_root(NeighborMode::In).unwrap(), None);
}

#[test]
fn vertex_sets_agree_with_cliques_module() {
    let g = karate();
    let largest = g.largest_cliques().unwrap();
    assert_eq!(largest[0].len(), g.clique_number().unwrap());
    for c in &largest {
        assert!(g.is_clique(c, false).unwrap());
        // Adding any other vertex breaks a largest clique.
        for v in (0..34).filter(|v| !c.contains(v)) {
            let mut bigger = c.clone();
            bigger.push(v);
            assert!(!g.is_clique(&bigger, false).unwrap());
        }
    }
    let petersen = petersen();
    let indep = petersen.largest_independent_vertex_sets().unwrap();
    assert_eq!(indep[0].len(), petersen.independence_number().unwrap());
    assert_eq!(indep[0].len(), 4);
    for s in &indep {
        assert!(petersen.is_independent_vertex_set(s).unwrap());
        // An independent set of G is a clique of the complement.
        let comp = petersen.complementer(false).unwrap();
        assert!(comp.is_clique(s, false).unwrap());
    }
}

#[test]
fn perfect_graphs_like_igraph_unit_test() {
    // tests/unit/igraph_perfect.c: Chvatal is not perfect, House is.
    assert!(!Graph::famous("Chvatal").unwrap().is_perfect().unwrap());
    assert!(Graph::famous("House").unwrap().is_perfect().unwrap());
    // Weak perfect graph theorem: complements of perfect graphs are perfect.
    for g in [cycle(6), karate().complementer(false).unwrap(), complete(5)] {
        let perfect = g.is_perfect().unwrap();
        let comp = g.complementer(false).unwrap();
        assert_eq!(comp.is_perfect().unwrap(), perfect);
    }
    // C7 and its complement (odd hole / antihole) are not perfect.
    assert!(!cycle(7).is_perfect().unwrap());
    assert!(!cycle(7).complementer(false).unwrap().is_perfect().unwrap());
    // Bipartite graphs are perfect.
    let grid = Graph::square_lattice(&[4, 5], 1, false, false, None).unwrap();
    assert!(grid.is_bipartite().unwrap() && grid.is_perfect().unwrap());
}

#[test]
fn loops_and_multi_edges_in_complete_and_independent_checks() {
    let mut k3 = complete(3);
    k3.add_edges(&[(0, 0), (0, 1)]).unwrap();
    assert!(k3.is_complete().unwrap());
    let mut g = Graph::from_edges(&[(0, 1)], 3, false).unwrap();
    g.add_edge(2, 2).unwrap();
    assert!(g.is_independent_vertex_set(&[0, 2]).unwrap());
    assert!(g.is_independent_vertex_set(&[2]).unwrap());
    assert!(!g.is_independent_vertex_set(&[0, 1]).unwrap());
}

#[test]
fn disassortative_knn_goes_with_negative_assortativity() {
    // Zachary's karate club is disassortative: r = -0.47561 (Newman).
    let g = Graph::famous("Zachary").unwrap();
    let r = g.assortativity_degree(false).unwrap();
    assert_close(r, -0.475613, 1e-5);
    let dcv = g
        .degree_correlation_vector(None, NeighborMode::All, NeighborMode::All, true)
        .unwrap();
    // k_nn(k) of the lowest degrees exceeds that of the hubs.
    assert!(dcv[1] > dcv[16] && dcv[2] > dcv[17]);
    // A star is maximally disassortative (r = -1).
    let star = Graph::star(6, StarMode::Undirected, 0).unwrap();
    assert_close(star.assortativity_degree(false).unwrap(), -1.0, 1e-12);
    let nd = star
        .avg_nearest_neighbor_degree(.., NeighborMode::All, NeighborMode::All, None)
        .unwrap();
    assert_eq!(nd.knn, vec![1.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
}

#[test]
fn unfolding_a_connected_graph_gives_a_tree_with_one_edge_per_edge() {
    // Unfolding from one root: every edge becomes a tree edge, so the tree
    // has |E| + 1 vertices, and the copies map back onto real vertices.
    for g in [karate(), petersen(), complete(6)] {
        let u = g.unfold_tree(NeighborMode::All, &[0]).unwrap();
        assert_eq!(u.tree.ecount(), g.ecount());
        assert_eq!(u.tree.vcount(), g.ecount() + 1);
        assert!(u.tree.is_tree(NeighborMode::All).unwrap());
        assert!(u.tree.edge_list().iter().all(|&(a, b)| {
            g.are_adjacent(u.vertex_index[a as usize], u.vertex_index[b as usize])
                .unwrap()
        }));
    }
}

/// The workarounds of the wrappers are still needed with igraph 1.0.0 and
/// 1.0.1 (the C sources involved did not change in 1.0.1): call the raw C
/// functions and observe the quirks. If a future igraph release fixes one of
/// them, this test fails and the corresponding workaround can be removed.
#[test]
fn upstream_quirks_are_still_present_in_igraph_1_0_1() {
    use igraph::ffi;
    // Not pinned to 1.0.1: the run-time library may be 1.0.0 when an
    // `LD_LIBRARY_PATH` overrides the RUNPATH set by build.rs.
    assert!(igraph::misc::version().triple() >= (1, 0, 0));

    // 1. igraph_count_multiple_1 counts an undirected self-loop twice.
    let g = Graph::from_edges(&[(0, 0), (0, 1)], 2, false).unwrap();
    let mut raw = 0;
    let code = unsafe { ffi::igraph_count_multiple_1(&g, &mut raw, 0) };
    assert_eq!(code, ffi::igraph_error_type_t_IGRAPH_SUCCESS);
    assert_eq!(raw, 2);
    assert_eq!(g.count_multiple_1(0).unwrap(), 1);
    assert_eq!(g.count_multiple(0).unwrap(), vec![1]);

    // 2. igraph_get_laplacian_sparse does not symmetrize directed graphs
    //    with IGRAPH_ALL, while igraph_get_laplacian does.
    let d = Graph::from_edges(&[(0, 1)], 2, true).unwrap();
    let mut sm = igraph::linalg::SparseMat::new(2, 2).unwrap();
    let code = unsafe {
        ffi::igraph_get_laplacian_sparse(
            &d,
            &mut sm,
            ffi::igraph_neimode_t_IGRAPH_ALL,
            LaplacianNormalization::Unnormalized.into(),
            std::ptr::null(),
        )
    };
    assert_eq!(code, ffi::igraph_error_type_t_IGRAPH_SUCCESS);
    let raw = sm.to_dense().unwrap();
    assert_eq!((raw[(0, 1)], raw[(1, 0)]), (-1.0, 0.0));
    let dense = d
        .get_laplacian(
            NeighborMode::All,
            LaplacianNormalization::Unnormalized,
            None,
        )
        .unwrap();
    assert_eq!((dense[(0, 1)], dense[(1, 0)]), (-1.0, -1.0));
    let fixed = d
        .get_laplacian_sparse(
            NeighborMode::All,
            LaplacianNormalization::Unnormalized,
            None,
        )
        .unwrap();
    assert_eq!(
        fixed,
        vec![(0, 0, 1.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 1.0)]
    );

    // 3. igraph_diversity decides degree-one vertices by the weight of edge
    //    0: with weights [0, 1, 1] on a 3-star, leaves 2 and 3 (own weight 1)
    //    get NaN instead of 0.
    let star = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], 4, false).unwrap();
    let w = Vector::from_slice(&[0.0, 1.0, 1.0]);
    let mut raw = Vector::new();
    let code = unsafe { ffi::igraph_diversity(&star, &w, &mut raw, ffi::igraph_vss_all()) };
    assert_eq!(code, ffi::igraph_error_type_t_IGRAPH_SUCCESS);
    assert!(raw[2].is_nan() && raw[3].is_nan());
    let fixed = star.diversity(&[0.0, 1.0, 1.0], ..).unwrap();
    assert!(fixed[1].is_nan());
    assert_eq!(&fixed[2..], &[0.0, 0.0]);

    // Not a quirk but a documented C convention the wrapper replaces:
    // igraph_random_spanning_tree reads any negative vertex id as "span all
    // components"; the wrapper spells that `None` and rejects negative ids.
    let f = Graph::from_edges(&[(0, 1), (2, 3)], 4, false).unwrap();
    let mut res = VectorInt::new();
    let code = unsafe { ffi::igraph_random_spanning_tree(&f, &mut res, -5) };
    assert_eq!(code, ffi::igraph_error_type_t_IGRAPH_SUCCESS);
    assert_eq!(res.len(), 2);
    assert_eq!(
        f.random_spanning_tree(Some(-5)).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
}

#[test]
fn diversity_of_degree_one_vertices_uses_their_own_edge() {
    // Leaves of a weighted star: 0 when their edge has positive weight, NaN
    // when it has zero weight, whatever the weight of edge 0.
    let star = Graph::star(5, StarMode::Undirected, 0).unwrap();
    for w in [
        [0.0, 2.0, 0.0, 3.0],
        [1.0, 0.0, 5.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ] {
        let d = star.diversity(&w, ..).unwrap();
        for leaf in 1..5 {
            if w[leaf - 1] > 0.0 {
                assert_eq!(d[leaf], 0.0, "{w:?}");
            } else {
                assert!(d[leaf].is_nan(), "{w:?}");
            }
        }
        // Selecting a subset, in any order, gives the same values.
        let sub = star.diversity(&w, &[4, 1, 0]).unwrap();
        assert_vec_close(&sub, &[d[4], d[1], d[0]], 1e-15);
        // Center: entropy of the positive weights over log 4.
        let s: f64 = w.iter().sum();
        let h: f64 = -w
            .iter()
            .filter(|&&x| x > 0.0)
            .map(|x| x / s * (x / s).ln())
            .sum::<f64>();
        assert_close(d[0], h / 4f64.ln(), 1e-12);
    }
}

/// Regression: once `has_multiple()` has cached "no multi-edges", igraph 1.0.1
/// stops deduplicating the mutual pairs of a directed graph in its undirected
/// adjacency lists. That made maximum cardinality search and the chordality
/// test write out of bounds (heap corruption) and made the girth 2.
#[test]
fn mutual_pairs_with_a_primed_multi_edge_cache() {
    let edges = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0), (2, 3)];
    let fresh = Graph::from_edges(&edges, 4, true).unwrap();
    let mcs = fresh.maximum_cardinality_search().unwrap();
    let chordal = fresh.is_chordal_with(None, None).unwrap();
    assert!(fresh.is_chordal().unwrap());
    assert!(chordal.is_chordal && chordal.fill_in.is_empty());

    let primed = || {
        let g = Graph::from_edges(&edges, 4, true).unwrap();
        assert!(!g.has_multiple().unwrap());
        g
    };
    let g = primed();
    assert_eq!(g.maximum_cardinality_search().unwrap(), mcs);
    assert!(
        !g.has_multiple().unwrap(),
        "the cache must not say HAS_MULTI"
    );
    assert!(primed().is_chordal().unwrap());
    let c = primed().is_chordal_with(None, None).unwrap();
    assert_eq!(
        (c.is_chordal, c.fill_in),
        (chordal.is_chordal, chordal.fill_in)
    );
    // A square made of mutual pairs is not chordal, primed or not.
    let sq = Graph::from_edges(
        &[
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 0),
            (0, 3),
        ],
        4,
        true,
    )
    .unwrap();
    assert!(sq.is_simple(true).unwrap());
    assert!(!sq.is_chordal().unwrap());
    assert_eq!(sq.is_chordal_with(None, None).unwrap().fill_in.len(), 1);

    // Girth: a path of mutual pairs has no cycle (was Some(2) when primed).
    let path = Graph::from_edges(&[(0, 1), (1, 0), (1, 2), (2, 1)], 3, true).unwrap();
    assert!(!path.has_multiple().unwrap());
    assert_eq!(path.girth().unwrap(), None);
    assert_eq!(path.girth_with_cycle().unwrap(), None);
    let tri = primed();
    assert_eq!(tri.girth().unwrap(), Some(3));
    let (len, cycle) = tri.girth_with_cycle().unwrap().unwrap();
    assert_eq!((len, cycle.len()), (3, 3));
}

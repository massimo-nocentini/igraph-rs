//! Integration tests for cliques, independent sets and colorings.

use igraph::{
    cliques::{ColoringGreedy, WeightedCliqueOptions},
    prelude::*,
};
use std::{collections::BTreeSet, ops::ControlFlow};

// Test graphs come from the `constructors` module.

/// Zachary's karate club.
fn karate() -> Graph {
    Graph::famous("Zachary").unwrap()
}

/// The undirected cycle C_n.
fn cycle(n: usize) -> Graph {
    Graph::cycle_graph(n, false, false).unwrap()
}

/// The undirected path P_n on `n` vertices.
fn path(n: usize) -> Graph {
    Graph::path_graph(n, false, false).unwrap()
}

/// The complete graph K_n.
fn complete(n: usize) -> Graph {
    Graph::full(n, false, false).unwrap()
}

/// The Petersen graph.
fn petersen() -> Graph {
    Graph::famous("Petersen").unwrap()
}

/// A complete binary tree with `n` vertices (vertex i has children 2i+1, 2i+2).
fn binary_tree(n: usize) -> Graph {
    Graph::kary_tree(n, 2, TreeMode::Undirected).unwrap()
}

/// Sorts each set and then the list, for order-independent comparisons.
fn canon(mut sets: Vec<Vec<i64>>) -> Vec<Vec<i64>> {
    sets.iter_mut().for_each(|s| s.sort());
    sets.sort();
    sets
}

/// igraph's canonical order: by size first, then lexicographically.
fn canon_by_size(sets: Vec<Vec<i64>>) -> Vec<Vec<i64>> {
    let mut sets = canon(sets);
    sets.sort_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(b)));
    sets
}

fn adjacent(g: &Graph, u: i64, v: i64) -> bool {
    g.get_eid(u, v, false).unwrap().is_some()
}

/// Checked with `Graph::is_clique` from the `structural` module.
fn is_clique(g: &Graph, set: &[i64]) -> bool {
    g.is_clique(set, false).unwrap()
}

/// Checked with `Graph::is_independent_vertex_set` from `structural`.
fn is_independent(g: &Graph, set: &[i64]) -> bool {
    g.is_independent_vertex_set(set).unwrap()
}

/// The graph of igraph's `examples/simple/igraph_cliques.c`: K6 minus the
/// edges 0-1, 0-2 and 3-5.
fn k6_minus() -> Graph {
    let mut edges = vec![];
    for i in 0..6 {
        for j in i + 1..6 {
            if ![(0, 1), (0, 2), (3, 5)].contains(&(i, j)) {
                edges.push((i, j));
            }
        }
    }
    Graph::from_edges(&edges, 6, false).unwrap()
}

/// Test graph of igraph's `tests/unit/igraph_weighted_cliques.c`.
fn weighted_test_graph() -> (Graph, Vec<f64>) {
    let e = [
        0, 1, 0, 6, 0, 7, 0, 8, 0, 9, 1, 2, 1, 3, 1, 4, 1, 6, 1, 7, 1, 8, 1, 9, 2, 3, 2, 5, 2, 6,
        2, 7, 2, 9, 3, 5, 3, 6, 3, 7, 3, 9, 4, 5, 4, 6, 4, 7, 4, 9, 5, 8, 6, 7, 6, 8, 7, 8, 8, 9,
    ];
    let g = Graph::from_flat_edges(&e, 10, false).unwrap();
    (g, vec![3., 2., 3., 5., 2., 3., 1., 3., 5., 5.])
}

// ----------------------------------------------------------------------
// Cliques
// ----------------------------------------------------------------------

#[test]
fn complete_graph_clique_number_is_n() {
    for n in 1..=8 {
        let k = complete(n);
        assert_eq!(k.clique_number().unwrap(), n);
        assert_eq!(k.largest_cliques().unwrap().len(), 1);
        assert_eq!(k.maximal_cliques_count(..).unwrap(), 1);
        assert_eq!(k.independence_number().unwrap(), 1);
        // Clique size histogram of K_n is the row of binomial coefficients.
        let hist = k.clique_size_hist(..).unwrap();
        let mut binom = vec![];
        let mut c = 1usize;
        for k in 1..=n {
            c = c * (n - k + 1) / k;
            binom.push(c);
        }
        assert_eq!(hist, binom, "K_{n}");
        // Hence 2^n - 1 cliques in total.
        assert_eq!(k.cliques(.., None).unwrap().len(), (1usize << n) - 1);
    }
}

#[test]
fn igraph_cliques_example() {
    // Values from examples/simple/igraph_cliques.out.
    let g = k6_minus();
    assert_eq!(
        canon(g.cliques(4.., None).unwrap()),
        vec![vec![1, 2, 3, 4], vec![1, 2, 4, 5]]
    );
    assert_eq!(canon(g.cliques(2..=2, None).unwrap()).len(), 15 - 3);
    assert_eq!(
        canon(g.largest_cliques().unwrap()),
        vec![vec![1, 2, 3, 4], vec![1, 2, 4, 5]]
    );
    assert_eq!(g.clique_number().unwrap(), 4);
    // A tree has no cliques of size 5.
    assert!(binary_tree(31).cliques(5..=5, None).unwrap().is_empty());
}

#[test]
fn every_reported_clique_is_a_clique() {
    let g = karate();
    let all = g.cliques(.., None).unwrap();
    assert!(all.iter().all(|c| is_clique(&g, c)));
    // Size histogram and full list agree.
    let hist = g.clique_size_hist(..).unwrap();
    assert_eq!(hist.iter().sum::<usize>(), all.len());
    assert_eq!(hist[0], 34);
    assert_eq!(hist[1], 78); // every edge is a 2-clique
    assert_eq!(hist[2], 45); // the karate club has 45 triangles
    assert_eq!(hist.len(), 5); // clique number 5
    assert_eq!(g.clique_number().unwrap(), 5);
}

#[test]
fn max_results_limits_the_output() {
    let g = complete(6);
    assert_eq!(g.cliques(.., Some(10)).unwrap().len(), 10);
    assert_eq!(g.cliques(.., Some(0)).unwrap().len(), 0);
    let g = karate();
    assert_eq!(g.maximal_cliques(.., Some(3)).unwrap().len(), 3);
    assert_eq!(g.independent_vertex_sets(3..=3, Some(7)).unwrap().len(), 7);
}

#[test]
fn clique_size_hist_matches_igraph_unit_test() {
    // tests/unit/igraph_clique_size_hist.out
    assert!(
        Graph::new(0, false)
            .clique_size_hist(..)
            .unwrap()
            .is_empty()
    );
    let g = Graph::from_flat_edges(
        &[0, 1, 0, 2, 1, 1, 1, 2, 1, 3, 2, 0, 2, 3, 3, 4, 3, 4],
        6,
        true,
    )
    .unwrap();
    assert_eq!(g.clique_size_hist(..).unwrap(), vec![6, 6, 2]);
    assert_eq!(g.clique_size_hist(2..).unwrap(), vec![0, 6, 2]);
    assert_eq!(g.clique_size_hist(..=2).unwrap(), vec![6, 6]);
    assert!(g.clique_size_hist(10..=10).unwrap().is_empty());
}

#[test]
fn empty_size_ranges_are_rejected() {
    let g = complete(4);
    #[allow(clippy::reversed_empty_ranges)]
    let err = g.cliques(4..=2, None).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    // `..1` would mean "max size 0", which igraph reads as "unbounded".
    assert_eq!(
        g.maximal_cliques(..1, None).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.clique_size_hist(..=0).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    // Exclusive ranges are converted correctly.
    assert_eq!(g.cliques(2..3, None).unwrap().len(), 6);
    assert!(g.cliques(1..2, None).unwrap().iter().all(|c| c.len() == 1));
}

#[test]
fn cliques_callback_matches_cliques() {
    let g = k6_minus();
    let listed = g.cliques(2..=3, None).unwrap();
    let mut streamed = vec![];
    g.cliques_callback(2..=3, |c| {
        streamed.push(c.to_vec());
        ControlFlow::Continue(())
    })
    .unwrap();
    assert_eq!(listed, streamed);

    // Stopping early is not an error.
    let mut seen = 0;
    g.cliques_callback(.., |_| {
        seen += 1;
        if seen == 5 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .unwrap();
    assert_eq!(seen, 5);
}

#[test]
fn maximal_cliques_callback_can_stop_and_count() {
    let g = karate();
    let total = g.maximal_cliques_count(..).unwrap();
    let mut n = 0;
    g.maximal_cliques_callback(.., |c| {
        assert!(is_clique(&g, c));
        n += 1;
        ControlFlow::Continue(())
    })
    .unwrap();
    assert_eq!(n, total);

    let mut n = 0;
    g.maximal_cliques_callback(3.., |_| {
        n += 1;
        if n == 2 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .unwrap();
    assert_eq!(n, 2);
}

#[test]
fn panics_in_callbacks_propagate() {
    let g = complete(5);
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        g.maximal_cliques_callback(.., |_| panic!("boom")).ok();
    }));
    assert!(caught.is_err());
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        g.cliques_callback(.., |_| panic!("boom")).ok();
    }));
    assert!(caught.is_err());
    // The thread is still usable afterwards.
    assert_eq!(g.clique_number().unwrap(), 5);
}

// ----------------------------------------------------------------------
// Maximal cliques
// ----------------------------------------------------------------------

#[test]
fn maximal_cliques_hist_matches_igraph_unit_test() {
    // tests/unit/maximal_cliques_hist.out: "1 1 2".
    let g = Graph::from_edges(&[(1, 2), (2, 3), (3, 4), (4, 5), (5, 2), (2, 4)], 6, false).unwrap();
    assert_eq!(g.maximal_cliques_hist(..).unwrap(), vec![1, 1, 2]);
    assert_eq!(g.maximal_cliques_count(..).unwrap(), 4);
    assert_eq!(g.maximal_cliques_hist(3..).unwrap(), vec![0, 0, 2]);
}

#[test]
fn maximal_cliques_are_maximal() {
    let g = karate();
    let maximal = g.maximal_cliques(.., None).unwrap();
    for c in &maximal {
        assert!(is_clique(&g, c));
        // No outside vertex is adjacent to all members.
        let members: BTreeSet<_> = c.iter().copied().collect();
        for v in g.vertices().filter(|v| !members.contains(v)) {
            assert!(!c.iter().all(|&u| adjacent(&g, u, v)), "{c:?} + {v}");
        }
    }
    let hist = g.maximal_cliques_hist(..).unwrap();
    assert_eq!(hist.iter().sum::<usize>(), maximal.len());
    assert_eq!(g.maximal_cliques_count(..).unwrap(), maximal.len());
    // Largest cliques are the maximal cliques of maximum size.
    let omega = g.clique_number().unwrap();
    let largest = canon(g.largest_cliques().unwrap());
    let maximal_top = canon(maximal.into_iter().filter(|c| c.len() == omega).collect());
    assert_eq!(largest, maximal_top);
    assert_eq!(largest, vec![vec![0, 1, 2, 3, 7], vec![0, 1, 2, 3, 13]]);
}

#[test]
fn maximal_cliques_subset_partitions_the_work() {
    let g = karate();
    let all = canon(g.maximal_cliques(.., None).unwrap());
    let evens: Vec<i64> = (0..34).filter(|v| v % 2 == 0).collect();
    let odds: Vec<i64> = (0..34).filter(|v| v % 2 == 1).collect();
    let mut union = g.maximal_cliques_subset(&evens, .., None).unwrap();
    union.extend(g.maximal_cliques_subset(&odds, .., None).unwrap());
    assert_eq!(canon(union), all);

    let err = g.maximal_cliques_subset(&[0, 34], .., None).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
}

#[test]
fn write_maximal_cliques_matches_igraph_unit_test() {
    // tests/unit/igraph_maximal_cliques_file.out
    let mut out = Vec::new();
    Graph::new(0, false)
        .write_maximal_cliques(&mut out, .., None)
        .unwrap();
    assert!(out.is_empty());

    let g = Graph::from_flat_edges(
        &[0, 1, 0, 2, 1, 1, 1, 2, 1, 3, 2, 0, 2, 3, 3, 4, 3, 4],
        6,
        true,
    )
    .unwrap();
    let mut out = Vec::new();
    g.write_maximal_cliques(&mut out, .., None).unwrap();
    assert_eq!(String::from_utf8(out).unwrap(), "5\n3 4\n3 1 2\n0 1 2\n");

    let mut out = Vec::new();
    g.write_maximal_cliques(&mut out, 10.., None).unwrap();
    assert!(out.is_empty());

    let mut out = Vec::new();
    g.write_maximal_cliques(&mut out, ..=2, None).unwrap();
    let text = String::from_utf8(out).unwrap();
    assert_eq!(text.lines().collect::<Vec<_>>(), vec!["5", "3 4"]);
}

// ----------------------------------------------------------------------
// Weighted cliques (tests/unit/igraph_weighted_cliques.out)
// ----------------------------------------------------------------------

fn weight(set: &[i64], w: &[f64]) -> f64 {
    set.iter().map(|&v| w[v as usize]).sum()
}

#[test]
fn weighted_cliques_match_igraph_unit_test() {
    let (g, w) = weighted_test_graph();
    let above6 = g
        .weighted_cliques(
            Some(&w),
            &WeightedCliqueOptions::default().with_min_weight(6.0),
        )
        .unwrap();
    assert_eq!(above6.len(), 63);
    assert!(
        above6
            .iter()
            .all(|c| is_clique(&g, c) && weight(c, &w) >= 6.0)
    );

    let between = WeightedCliqueOptions::default()
        .with_min_weight(5.0)
        .with_max_weight(10.0);
    let r = g.weighted_cliques(Some(&w), &between).unwrap();
    assert_eq!(r.len(), 53);
    assert!(r.iter().all(|c| (5.0..=10.0).contains(&weight(c, &w))));

    let max7 = WeightedCliqueOptions::default()
        .with_maximal(true)
        .with_min_weight(7.0);
    assert_eq!(g.weighted_cliques(Some(&w), &max7).unwrap().len(), 8);
    let r = g
        .weighted_cliques(Some(&w), &between.with_maximal(true))
        .unwrap();
    assert_eq!(r.len(), 4);

    assert_eq!(
        canon(g.largest_weighted_cliques(Some(&w)).unwrap()),
        vec![vec![0, 1, 8, 9], vec![1, 2, 3, 9]]
    );
    assert_eq!(g.weighted_clique_number(Some(&w)).unwrap(), 15.0);
}

#[test]
fn weighted_cliques_unweighted_fallback_and_unit_weights() {
    let (g, _) = weighted_test_graph();
    let sizes = WeightedCliqueOptions::default()
        .with_min_weight(4.0)
        .with_max_weight(5.0);
    let none = g.weighted_cliques(None, &sizes).unwrap();
    assert_eq!(none.len(), 15);
    assert_eq!(canon(none.clone()), canon(g.cliques(4..=5, None).unwrap()));
    assert_eq!(
        g.weighted_cliques(None, &sizes.with_maximal(true))
            .unwrap()
            .len(),
        5
    );
    assert_eq!(g.largest_weighted_cliques(None).unwrap().len(), 2);
    assert_eq!(g.weighted_clique_number(None).unwrap(), 5.0);

    let ones = vec![1.0; 10];
    assert_eq!(
        canon(g.weighted_cliques(Some(&ones), &sizes).unwrap()),
        canon(none)
    );
    assert_eq!(
        canon(g.largest_weighted_cliques(Some(&ones)).unwrap()),
        canon(g.largest_cliques().unwrap())
    );
    assert_eq!(g.weighted_clique_number(Some(&ones)).unwrap(), 5.0);
}

#[test]
fn weighted_cliques_errors() {
    let (g, mut w) = weighted_test_graph();
    let opts = WeightedCliqueOptions::default();
    assert_eq!(
        g.weighted_cliques(Some(&w[..3]), &opts).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.weighted_clique_number(Some(&w[..3])).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    w[0] = 0.0;
    assert_eq!(
        g.largest_weighted_cliques(Some(&w)).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    let (g, w) = weighted_test_graph();
    let bad = WeightedCliqueOptions::default()
        .with_min_weight(10.0)
        .with_max_weight(5.0);
    assert_eq!(
        g.weighted_cliques(Some(&w), &bad).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

// ----------------------------------------------------------------------
// Independent vertex sets (examples/simple/igraph_independent_sets.out)
// ----------------------------------------------------------------------

#[test]
fn independent_sets_match_igraph_example() {
    let tree = binary_tree(5);
    assert!(tree.independent_vertex_sets(4.., None).unwrap().is_empty());
    assert_eq!(
        canon(tree.independent_vertex_sets(2..=2, None).unwrap()),
        vec![
            vec![0, 3],
            vec![0, 4],
            vec![1, 2],
            vec![2, 3],
            vec![2, 4],
            vec![3, 4]
        ]
    );
    assert_eq!(
        canon(tree.largest_independent_vertex_sets().unwrap()),
        vec![vec![0, 3, 4], vec![2, 3, 4]]
    );
    assert_eq!(tree.independent_vertex_sets(.., None).unwrap().len(), 13);

    let tree = binary_tree(10);
    let maximal = canon(tree.maximal_independent_vertex_sets(.., None).unwrap());
    assert_eq!(
        maximal,
        vec![
            vec![0, 3, 4, 5, 6],
            vec![0, 3, 5, 6, 9],
            vec![0, 4, 5, 6, 7, 8],
            vec![0, 5, 6, 7, 8, 9],
            vec![1, 2, 7, 8, 9],
            vec![1, 5, 6, 7, 8, 9],
            vec![2, 3, 4],
            vec![2, 3, 9],
            vec![2, 4, 7, 8],
        ]
    );
    assert_eq!(
        canon(tree.maximal_independent_vertex_sets(4..=5, None).unwrap()).len(),
        4
    );
    assert_eq!(tree.independence_number().unwrap(), 6);
}

#[test]
fn petersen_independence_and_cliques() {
    let p = petersen();
    assert_eq!(p.ecount(), 15);
    // The Petersen graph is triangle-free with independence number 4.
    assert_eq!(p.clique_number().unwrap(), 2);
    assert_eq!(p.largest_cliques().unwrap().len(), 15);
    assert_eq!(p.independence_number().unwrap(), 4);
    let largest = p.largest_independent_vertex_sets().unwrap();
    // It has exactly 5 maximum independent sets.
    assert_eq!(largest.len(), 5);
    assert!(
        largest
            .iter()
            .all(|s| s.len() == 4 && is_independent(&p, s))
    );
    // Every maximal independent set is also dominating.
    for s in p.maximal_independent_vertex_sets(.., None).unwrap() {
        assert!(is_independent(&p, &s));
        for v in p.vertices().filter(|v| !s.contains(v)) {
            assert!(s.iter().any(|&u| adjacent(&p, u, v)));
        }
    }
    // Its chromatic number is 3; α(G) * χ(G) >= n holds: 4 * 3 >= 10.
    let colors = p.vertex_coloring_greedy(ColoringGreedy::DSatur).unwrap();
    assert!(p.is_vertex_coloring(&colors).unwrap());
    assert_eq!(*colors.iter().max().unwrap() + 1, 3);
}

#[test]
fn independence_number_of_cycles_and_paths() {
    for n in 3..=12 {
        assert_eq!(cycle(n).independence_number().unwrap(), n / 2, "C_{n}");
        assert_eq!(
            path(n).independence_number().unwrap(),
            n.div_ceil(2),
            "P_{n}"
        );
    }
    // Independent sets are the cliques of the complement.
    let g = cycle(7);
    let comp = g.complementer(false).unwrap();
    assert_eq!(comp.ecount(), 7 * 6 / 2 - 7);
    assert_eq!(
        canon(g.independent_vertex_sets(.., None).unwrap()),
        canon(comp.cliques(.., None).unwrap())
    );
    assert_eq!(
        canon(g.maximal_independent_vertex_sets(.., None).unwrap()),
        canon(comp.maximal_cliques(.., None).unwrap())
    );
    assert_eq!(
        g.independence_number().unwrap(),
        comp.clique_number().unwrap()
    );
}

// ----------------------------------------------------------------------
// Colorings
// ----------------------------------------------------------------------

#[test]
fn greedy_colorings_are_valid() {
    for heuristic in [ColoringGreedy::ColoredNeighbors, ColoringGreedy::DSatur] {
        for g in [
            karate(),
            petersen(),
            complete(7),
            cycle(9),
            path(6),
            Graph::new(0, false),
        ] {
            let colors = g.vertex_coloring_greedy(heuristic).unwrap();
            assert_eq!(colors.len(), g.vcount());
            assert!(g.is_vertex_coloring(&colors).unwrap());
            // At least ω(G) colors are needed.
            let used = colors.iter().max().map_or(0, |&c| c as usize + 1);
            assert!(used >= g.clique_number().unwrap());
        }
    }
    // K_n needs exactly n colors, odd cycles 3, even cycles 2 (DSatur is exact there).
    let colors = complete(7)
        .vertex_coloring_greedy(ColoringGreedy::default())
        .unwrap();
    assert_eq!(colors.iter().collect::<BTreeSet<_>>().len(), 7);
    let colors = cycle(9)
        .vertex_coloring_greedy(ColoringGreedy::DSatur)
        .unwrap();
    assert_eq!(*colors.iter().max().unwrap(), 2);
    let colors = cycle(10)
        .vertex_coloring_greedy(ColoringGreedy::DSatur)
        .unwrap();
    assert_eq!(*colors.iter().max().unwrap(), 1);
}

#[test]
fn is_vertex_coloring_checks() {
    let g = cycle(4);
    assert!(g.is_vertex_coloring(&[0, 1, 0, 1]).unwrap());
    assert!(!g.is_vertex_coloring(&[0, 0, 1, 1]).unwrap());
    // Self-loops are ignored.
    let loops = Graph::from_edges(&[(0, 0), (0, 1)], 2, false).unwrap();
    assert!(loops.is_vertex_coloring(&[0, 1]).unwrap());
    assert_eq!(
        g.is_vertex_coloring(&[0, 1]).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn is_bipartite_coloring_reports_orientation() {
    let types = [false, true, false, true];
    assert_eq!(
        cycle(4).is_bipartite_coloring(&types).unwrap(),
        Some(NeighborMode::All)
    );
    assert_eq!(
        cycle(5)
            .is_bipartite_coloring(&[false, true, false, true, false])
            .unwrap(),
        None
    );

    let out = Graph::from_edges(&[(0, 1), (2, 1), (2, 3)], 4, true).unwrap();
    assert_eq!(
        out.is_bipartite_coloring(&types).unwrap(),
        Some(NeighborMode::Out)
    );
    let inn = Graph::from_edges(&[(1, 0), (1, 2), (3, 2)], 4, true).unwrap();
    assert_eq!(
        inn.is_bipartite_coloring(&types).unwrap(),
        Some(NeighborMode::In)
    );
    let mixed = Graph::from_edges(&[(0, 1), (1, 2)], 4, true).unwrap();
    assert_eq!(
        mixed.is_bipartite_coloring(&types).unwrap(),
        Some(NeighborMode::All)
    );
    assert_eq!(
        out.is_bipartite_coloring(&[true]).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn is_edge_coloring_checks() {
    // A cycle of even length is 2-edge-colorable by alternating colors.
    let c6 = cycle(6);
    assert!(c6.is_edge_coloring(&[0, 1, 0, 1, 0, 1]).unwrap());
    assert!(!c6.is_edge_coloring(&[0, 0, 1, 0, 1, 1]).unwrap());
    // An odd cycle is not: the alternating pattern breaks somewhere.
    assert!(!cycle(5).is_edge_coloring(&[0, 1, 0, 1, 0]).unwrap());
    assert!(cycle(5).is_edge_coloring(&[0, 1, 0, 1, 2]).unwrap());
    // Self-loops do not conflict with themselves.
    let g = Graph::from_edges(&[(0, 0), (0, 1)], 2, false).unwrap();
    assert!(g.is_edge_coloring(&[0, 1]).unwrap());
    assert_eq!(
        c6.is_edge_coloring(&[0]).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

// ----------------------------------------------------------------------
// Use case
// ----------------------------------------------------------------------

/// Story: a conference must schedule talks into time slots. Two talks that
/// share a speaker or an audience conflict and cannot run in the same slot.
/// The conflict graph's cliques give lower bounds on the number of slots
/// (all talks of a clique pairwise conflict), a greedy coloring gives an
/// actual schedule, and the largest independent sets tell how many talks can
/// run in parallel in the busiest slot.
#[test]
fn use_case_conference_scheduling() {
    let talks = [
        "Rust FFI",
        "igraph internals",
        "Cliques",
        "Colorings",
        "Flows",
        "Layouts",
        "Community detection",
        "Random graphs",
    ];
    // Conflicts: talks by the same speaker, or targeting the same audience.
    let conflicts = [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 3), // "graph theory" track, mostly one room
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4), // "applications" track
        (0, 7),
    ];
    let g = Graph::from_edges(&conflicts, talks.len(), false).unwrap();

    // Talks 0, 1, 2 pairwise conflict: at least 3 slots are needed.
    let omega = g.clique_number().unwrap();
    assert_eq!(omega, 3);
    let hardest = canon(g.largest_cliques().unwrap());
    assert_eq!(hardest, vec![vec![0, 1, 2], vec![1, 2, 3]]);

    // Greedy heuristics give valid schedules, but not necessarily optimal
    // ones: try both and keep the best.
    let n_colors = |c: &[i64]| *c.iter().max().unwrap() as usize + 1;
    let greedy = [ColoringGreedy::ColoredNeighbors, ColoringGreedy::DSatur]
        .map(|h| g.vertex_coloring_greedy(h).unwrap());
    for c in &greedy {
        assert!(g.is_vertex_coloring(c).unwrap());
        assert!(n_colors(c) >= omega);
    }
    let best = greedy.into_iter().min_by_key(|c| n_colors(c)).unwrap();
    // Greedy needs at most (max degree + 1) colors; here max degree = 4.
    assert!(n_colors(&best) <= 5);

    // The organizer finds a 3-slot schedule by hand: since ω = 3 it is optimal.
    let slots: Vec<i64> = vec![0, 1, 2, 0, 1, 0, 1, 2];
    assert!(g.is_vertex_coloring(&slots).unwrap());
    let n_slots = n_colors(&slots);
    assert_eq!(n_slots, omega);
    assert!(n_slots <= n_colors(&best));

    // Print the schedule (visible with `--nocapture`).
    for s in 0..n_slots as i64 {
        let in_slot: Vec<_> = (0..talks.len())
            .filter(|&t| slots[t] == s)
            .map(|t| talks[t])
            .collect();
        println!("slot {s}: {}", in_slot.join(", "));
        // Talks in the same slot form an independent set.
        let ids: Vec<i64> = (0..talks.len() as i64)
            .filter(|&t| slots[t as usize] == s)
            .collect();
        assert!(is_independent(&g, &ids));
    }

    // At most α(G) talks can ever run in parallel.
    let alpha = g.independence_number().unwrap();
    assert!(slots.iter().filter(|&&s| s == 0).count() <= alpha);
    assert!(alpha * n_slots >= talks.len());
    let biggest = g.largest_independent_vertex_sets().unwrap();
    assert!(
        biggest
            .iter()
            .all(|s| s.len() == alpha && is_independent(&g, s))
    );
    println!(
        "at most {alpha} talks in parallel, e.g. {:?}",
        canon_by_size(biggest)[0]
    );
}

// ----------------------------------------------------------------------
// Robustness (review fixes)
// ----------------------------------------------------------------------

#[test]
fn nested_cliquer_searches_are_refused_not_corrupted() {
    let g = complete(6);
    let mut outer = 0;
    let mut inner_errors = 0;
    g.cliques_callback(.., |_| {
        outer += 1;
        // Cliquer keeps its search state in per-thread globals: a nested
        // Cliquer search must be refused...
        let nested = [
            g.cliques(2..=2, None).map(|_| ()),
            g.clique_size_hist(..).map(|_| ()),
            g.cliques_callback(.., |_| ControlFlow::Continue(())),
            g.weighted_clique_number(Some(&[1.0; 6])).map(|_| ()),
            g.largest_weighted_cliques(Some(&[1.0; 6])).map(|_| ()),
            g.weighted_cliques(None, &WeightedCliqueOptions::default())
                .map(|_| ()),
        ];
        for r in nested {
            assert_eq!(r.unwrap_err().kind(), ErrorKind::Failure);
            inner_errors += 1;
        }
        // ...while the other algorithms are fine.
        assert_eq!(g.clique_number().unwrap(), 6);
        assert_eq!(g.maximal_cliques(.., None).unwrap().len(), 1);
        assert_eq!(g.weighted_clique_number(None).unwrap(), 6.0);
        ControlFlow::Continue(())
    })
    .unwrap();
    assert_eq!(outer, 63);
    assert_eq!(inner_errors, 6 * 63);
    // The guard is released afterwards, also after a panic.
    assert_eq!(g.cliques(.., None).unwrap().len(), 63);
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        g.cliques_callback(.., |_| panic!("boom")).ok();
    }));
    assert!(caught.is_err());
    assert_eq!(g.clique_size_hist(..).unwrap().len(), 6);

    // Maximal-clique callbacks do not use Cliquer: anything may be nested.
    let mut n = 0;
    g.maximal_cliques_callback(.., |c| {
        assert_eq!(g.cliques(c.len().., None).unwrap().len(), 1);
        n += 1;
        ControlFlow::Continue(())
    })
    .unwrap();
    assert_eq!(n, 1);
}

#[test]
fn oversized_bounds_give_empty_results() {
    let g = karate();
    // Lower bounds beyond the vertex count: nothing, without huge allocations.
    assert!(g.cliques(usize::MAX.., None).unwrap().is_empty());
    assert!(g.clique_size_hist(35..).unwrap().is_empty());
    assert!(g.clique_size_hist(1_000_000_000..).unwrap().is_empty());
    assert_eq!(g.maximal_cliques_count(35..).unwrap(), 0);
    assert!(g.maximal_cliques_hist(100..).unwrap().is_empty());
    assert!(g.independent_vertex_sets(35.., None).unwrap().is_empty());
    assert!(
        g.maximal_independent_vertex_sets(35.., None)
            .unwrap()
            .is_empty()
    );
    assert!(
        g.maximal_cliques_subset(&[0], 35.., None)
            .unwrap()
            .is_empty()
    );
    let mut out = Vec::new();
    g.write_maximal_cliques(&mut out, 35.., None).unwrap();
    assert!(out.is_empty());
    let mut called = false;
    g.cliques_callback(usize::MAX.., |_| {
        called = true;
        ControlFlow::Continue(())
    })
    .unwrap();
    assert!(!called);
    // Upper bounds beyond the vertex count are the same as no bound (and do
    // not make igraph allocate a histogram of that length).
    assert_eq!(
        g.clique_size_hist(..=1_000_000_000).unwrap(),
        g.clique_size_hist(..).unwrap()
    );
    assert_eq!(
        g.maximal_cliques_count(3..=usize::MAX).unwrap(),
        g.maximal_cliques_count(3..).unwrap()
    );
    // Invalid ids in the subset are reported even when the range is empty.
    assert_eq!(
        g.maximal_cliques_subset(&[-1], 35.., None)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidVertexId
    );
}

#[test]
fn zero_max_results_returns_nothing() {
    let g = karate();
    assert!(g.maximal_cliques(.., Some(0)).unwrap().is_empty());
    assert!(g.independent_vertex_sets(.., Some(0)).unwrap().is_empty());
    assert!(
        g.maximal_independent_vertex_sets(.., Some(0))
            .unwrap()
            .is_empty()
    );
    assert!(
        g.maximal_cliques_subset(&[0, 1], .., Some(0))
            .unwrap()
            .is_empty()
    );
    let (g, w) = weighted_test_graph();
    let opts = WeightedCliqueOptions::default().with_max_results(0);
    assert!(g.weighted_cliques(Some(&w), &opts).unwrap().is_empty());
    let opts = WeightedCliqueOptions::default().with_max_results(5);
    assert_eq!(g.weighted_cliques(Some(&w), &opts).unwrap().len(), 5);
}

#[test]
fn weight_validation() {
    let (g, w) = weighted_test_graph();
    let all = WeightedCliqueOptions::default();
    for bad in [f64::NAN, f64::INFINITY, -3.0, 0.5, 1e12] {
        let mut v = w.clone();
        v[4] = bad;
        assert_eq!(
            g.weighted_cliques(Some(&v), &all).unwrap_err().kind(),
            ErrorKind::InvalidValue,
            "weight {bad}"
        );
        assert_eq!(
            g.weighted_clique_number(Some(&v)).unwrap_err().kind(),
            ErrorKind::InvalidValue
        );
    }
    // Total weight must fit in a C int.
    let huge = vec![1e9; 10];
    assert_eq!(
        g.weighted_clique_number(Some(&huge)).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    // Fractional weights are truncated (igraph warns).
    let frac: Vec<f64> = w.iter().map(|x| x + 0.9).collect();
    assert_eq!(g.weighted_clique_number(Some(&frac)).unwrap(), 15.0);

    // Bounds: `Some(0.0)` as maximum is an empty range, not "no bound".
    for opts in [
        WeightedCliqueOptions::default().with_max_weight(0.0),
        WeightedCliqueOptions::default().with_max_weight(0.7),
        WeightedCliqueOptions::default().with_min_weight(f64::NAN),
        WeightedCliqueOptions::default().with_max_weight(f64::INFINITY),
    ] {
        assert_eq!(
            g.weighted_cliques(Some(&w), &opts).unwrap_err().kind(),
            ErrorKind::InvalidValue,
            "{opts:?}"
        );
    }
    // A huge maximum is no bound; a huge minimum yields nothing.
    let n_all = g.weighted_cliques(Some(&w), &all).unwrap().len();
    let big_max = WeightedCliqueOptions::default().with_max_weight(1e15);
    assert_eq!(g.weighted_cliques(Some(&w), &big_max).unwrap().len(), n_all);
    let big_min = WeightedCliqueOptions::default().with_min_weight(1e15);
    assert!(g.weighted_cliques(Some(&w), &big_min).unwrap().is_empty());
    assert!(g.weighted_cliques(None, &big_min).unwrap().is_empty());
    // Negative minimum = no lower bound; fractional bounds are truncated.
    let neg = WeightedCliqueOptions::default().with_min_weight(-5.0);
    assert_eq!(g.weighted_cliques(Some(&w), &neg).unwrap().len(), n_all);
    let frac = WeightedCliqueOptions::default()
        .with_min_weight(5.9)
        .with_max_weight(10.9);
    assert_eq!(g.weighted_cliques(Some(&w), &frac).unwrap().len(), 53);
}

#[test]
fn directed_graphs_are_treated_as_undirected() {
    // A directed triangle with both a 3-cycle and a reciprocal edge.
    let d = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (1, 0), (2, 3)], 4, true).unwrap();
    let u = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false).unwrap();
    assert_eq!(d.clique_number().unwrap(), u.clique_number().unwrap());
    assert_eq!(
        canon(d.maximal_cliques(.., None).unwrap()),
        canon(u.maximal_cliques(.., None).unwrap())
    );
    assert_eq!(
        canon(d.cliques(.., None).unwrap()),
        canon(u.cliques(.., None).unwrap())
    );
    assert_eq!(
        canon(d.largest_independent_vertex_sets().unwrap()),
        canon(u.largest_independent_vertex_sets().unwrap())
    );
    // A directed graph without edges: orientation is reported as `All`.
    let empty = Graph::new(3, true);
    assert_eq!(
        empty.is_bipartite_coloring(&[false, true, false]).unwrap(),
        Some(NeighborMode::All)
    );
    // A directed edge coloring treats parallel/antiparallel edges as adjacent.
    let anti = Graph::from_edges(&[(0, 1), (1, 0)], 2, true).unwrap();
    assert!(!anti.is_edge_coloring(&[0, 0]).unwrap());
    assert!(anti.is_edge_coloring(&[0, 1]).unwrap());
}

// ----------------------------------------------------------------------
// igraph 1.0.1 unit tests reproduced with the seeded default RNG
// ----------------------------------------------------------------------

#[test]
fn maximal_cliques_match_igraph_unit_tests() {
    // tests/unit/igraph_maximal_cliques2.out, first part: C10.
    let ring = Graph::ring(10, false, false, true).unwrap();
    let mut expected: Vec<Vec<i64>> = (0..9).map(|i| vec![i, i + 1]).collect();
    expected.push(vec![0, 9]);
    assert_eq!(
        canon(ring.maximal_cliques(.., None).unwrap()),
        canon(expected)
    );
    assert_eq!(ring.maximal_cliques_count(..).unwrap(), 10);

    // Second part: G(50, 0.5) with seed 42 (igraph's default generator is
    // PCG32, as the per-thread generator installed by this crate).
    rng::seed(42).unwrap();
    let g = Graph::erdos_renyi_game_gnp(50, 0.5, false, EdgeTypeSw::Simple, false).unwrap();
    let big = canon(g.maximal_cliques(8.., None).unwrap());
    assert_eq!(
        big,
        vec![
            vec![6, 15, 22, 23, 27, 33, 40, 44],
            vec![6, 22, 23, 27, 33, 40, 44, 48],
            vec![13, 15, 22, 24, 27, 33, 40, 44],
            vec![13, 15, 22, 24, 33, 36, 38, 42],
            vec![15, 22, 24, 29, 33, 36, 38, 42],
        ]
    );
    assert_eq!(g.maximal_cliques_count(8..).unwrap(), 5);
    assert_eq!(g.clique_number().unwrap(), 8);
    assert_eq!(canon(g.largest_cliques().unwrap()), big);

    // tests/unit/igraph_maximal_cliques.c: a triangle with a self-loop.
    let looped = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (0, 0)], 3, false).unwrap();
    assert_eq!(
        canon(looped.maximal_cliques(3.., None).unwrap()),
        vec![vec![0, 1, 2]]
    );
}

#[test]
fn maximal_cliques_subset_matches_igraph_unit_test() {
    // tests/unit/igraph_maximal_cliques4.out: G(100, 0.5) with seed 42 has
    // 50 maximal cliques of size >= 9; 15 of them are found from the
    // initial vertices 0..13 and 35 from 13..100.
    rng::seed(42).unwrap();
    let g = Graph::erdos_renyi_game_gnp(100, 0.5, false, EdgeTypeSw::Simple, false).unwrap();
    let all = canon(g.maximal_cliques(9.., None).unwrap());
    assert_eq!(all.len(), 50);
    assert_eq!(all[0], vec![1, 22, 27, 29, 51, 57, 58, 59, 95]);
    let head: Vec<i64> = (0..13).collect();
    let tail: Vec<i64> = (13..100).collect();
    let cl1 = canon(g.maximal_cliques_subset(&head, 9.., None).unwrap());
    let cl2 = g.maximal_cliques_subset(&tail, 9.., None).unwrap();
    assert_eq!((cl1.len(), cl2.len()), (15, 35));
    assert_eq!(cl1[0], vec![6, 11, 14, 15, 27, 49, 67, 87, 88]);
    assert_eq!(cl1[14], vec![14, 27, 48, 49, 57, 62, 80, 92, 95]);
    let mut union = cl1;
    union.extend(cl2);
    assert_eq!(canon(union), all);
}

#[test]
fn greedy_coloring_matches_igraph_unit_test() {
    // tests/unit/coloring.c + igraph_coloring.out.
    let small = Graph::from_flat_edges(
        &[
            0, 3, 0, 4, 1, 4, 2, 4, 1, 5, 2, 5, 3, 5, 0, 6, 1, 6, 2, 6, 3, 6, 0, 7, 1, 7, 2, 7, 4,
            7, 5, 7,
        ],
        8,
        false,
    )
    .unwrap();
    assert_eq!(
        small
            .vertex_coloring_greedy(ColoringGreedy::ColoredNeighbors)
            .unwrap(),
        vec![2, 2, 2, 3, 1, 1, 0, 0]
    );
    assert_eq!(
        small
            .vertex_coloring_greedy(ColoringGreedy::DSatur)
            .unwrap(),
        vec![2, 2, 2, 0, 1, 1, 1, 0]
    );
    // The same graph with extra multi-edges and self-loops.
    let mut multi_edges = vec![
        0, 3, 0, 4, 1, 4, 2, 4, 1, 5, 2, 5, 3, 5, 0, 6, 1, 6, 2, 6, 3, 6, 0, 7, 1, 7, 2, 7, 4, 7,
        5, 7,
    ];
    multi_edges.extend([0, 4, 0, 4, 3, 5, 3, 5, 3, 5, 0, 0, 0, 0, 1, 1]);
    let multi = Graph::from_flat_edges(&multi_edges, 8, false).unwrap();
    assert_eq!(
        multi
            .vertex_coloring_greedy(ColoringGreedy::ColoredNeighbors)
            .unwrap(),
        vec![3, 1, 1, 1, 0, 0, 0, 2]
    );
    assert_eq!(
        multi
            .vertex_coloring_greedy(ColoringGreedy::DSatur)
            .unwrap(),
        vec![2, 2, 2, 0, 1, 1, 1, 0]
    );
    // Isolated vertices.
    let isolated = Graph::from_edges(&[(0, 1)], 4, false).unwrap();
    assert_eq!(
        isolated
            .vertex_coloring_greedy(ColoringGreedy::ColoredNeighbors)
            .unwrap(),
        vec![0, 1, 0, 0]
    );
    assert_eq!(
        isolated
            .vertex_coloring_greedy(ColoringGreedy::DSatur)
            .unwrap(),
        vec![1, 0, 0, 0]
    );
    // Wheels: an odd rim (even vertex count) needs 4 colors, an even rim 3.
    for heuristic in [ColoringGreedy::ColoredNeighbors, ColoringGreedy::DSatur] {
        let w11 = Graph::wheel(11, WheelMode::Undirected, 0).unwrap();
        assert_eq!(
            w11.vertex_coloring_greedy(heuristic).unwrap(),
            vec![0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
        );
        let w12 = Graph::wheel(12, WheelMode::Undirected, 0).unwrap();
        assert_eq!(
            w12.vertex_coloring_greedy(heuristic).unwrap(),
            vec![0, 3, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
        );
    }
    // A large random graph, G(1000, 10000) with seed 42: 11 colors with the
    // colored-neighbors heuristic, 8 with DSatur.
    rng::seed(42).unwrap();
    let large = Graph::erdos_renyi_game_gnm(1000, 10000, false, EdgeTypeSw::Simple, false).unwrap();
    let n_colors = |c: &[i64]| c.iter().max().map_or(0, |&m| m + 1);
    let cn = large
        .vertex_coloring_greedy(ColoringGreedy::ColoredNeighbors)
        .unwrap();
    let ds = large
        .vertex_coloring_greedy(ColoringGreedy::DSatur)
        .unwrap();
    assert!(large.is_vertex_coloring(&cn).unwrap());
    assert!(large.is_vertex_coloring(&ds).unwrap());
    assert_eq!((n_colors(&cn), n_colors(&ds)), (11, 8));
}

// ----------------------------------------------------------------------
// Cross-module identities
// ----------------------------------------------------------------------

#[test]
fn triangles_are_the_three_cliques() {
    let g = karate();
    let mut triangles: Vec<Vec<i64>> = g
        .list_triangles()
        .unwrap()
        .into_iter()
        .map(|t| t.to_vec())
        .collect();
    triangles = canon(triangles);
    assert_eq!(triangles.len(), 45);
    assert_eq!(canon(g.cliques(3..=3, None).unwrap()), triangles);
    assert_eq!(
        g.clique_size_hist(..).unwrap()[2] as f64,
        g.count_triangles().unwrap()
    );
    // Global transitivity = 3 * triangles / connected triples.
    let triples: f64 = g
        .degree(.., NeighborMode::All, Loops::None)
        .unwrap()
        .iter()
        .map(|&d| (d * (d - 1) / 2) as f64)
        .sum();
    let t = g.transitivity_undirected(TransitivityMode::Nan).unwrap();
    assert!((t - 3.0 * 45.0 / triples).abs() < 1e-12);
}

#[test]
fn turan_and_moon_moser_graphs() {
    for (n, r) in [(6, 3), (7, 3), (10, 4), (9, 9), (5, 2)] {
        let (t, types) = Graph::turan(n, r).unwrap();
        // K_{r+1}-free with a K_r: ω = r; α = size of the largest part.
        assert_eq!(t.clique_number().unwrap(), r, "T({n}, {r})");
        assert_eq!(t.independence_number().unwrap(), n.div_ceil(r));
        // The parts give an optimal r-coloring.
        assert!(t.is_vertex_coloring(&types).unwrap());
        // Maximal cliques pick one vertex per part: product of part sizes.
        let product: usize = (0..r as i64)
            .map(|p| types.iter().filter(|&&x| x == p).count())
            .product();
        assert_eq!(t.maximal_cliques_count(..).unwrap(), product);
        // The maximal independent sets are exactly the parts.
        assert_eq!(
            t.maximal_independent_vertex_sets(.., None).unwrap().len(),
            r
        );
    }
    // Moon–Moser: K_{3,3,3,3} = T(12, 4) has 3^4 maximal cliques, the most a
    // 12-vertex graph can have; its complement (4 disjoint triangles) has
    // 3^4 maximal independent sets.
    let (mm, _) = Graph::turan(12, 4).unwrap();
    assert_eq!(mm.maximal_cliques_count(..).unwrap(), 81);
    assert_eq!(mm.maximal_cliques_hist(..).unwrap(), vec![0, 0, 0, 81]);
    let triangles = mm.complementer(false).unwrap();
    assert_eq!(
        triangles
            .maximal_independent_vertex_sets(.., None)
            .unwrap()
            .len(),
        81
    );
    assert_eq!(triangles.largest_cliques().unwrap().len(), 4);
}

#[test]
fn mycielski_graphs_are_triangle_free_but_need_many_colors() {
    // Mycielski's construction keeps ω = 2 while raising χ by one.
    let k2 = Graph::from_edges(&[(0, 1)], 2, false).unwrap();
    for k in 0..4 {
        let m = k2.mycielskian(k).unwrap();
        assert_eq!(m.vcount(), 3 * (1 << k) - 1);
        assert_eq!(m.clique_number().unwrap(), 2, "M^{k}(K2)");
        assert_eq!(m.clique_size_hist(3..).unwrap(), Vec::<usize>::new());
        let chi = k + 2;
        for heuristic in [ColoringGreedy::ColoredNeighbors, ColoringGreedy::DSatur] {
            let colors = m.vertex_coloring_greedy(heuristic).unwrap();
            assert!(m.is_vertex_coloring(&colors).unwrap());
            assert!(*colors.iter().max().unwrap() as usize + 1 >= chi);
        }
    }
    // The Grötzsch graph M^2(K2): 11 vertices, ω = 2, α = 5, χ = 4.
    let g = Graph::famous("Grotzsch").unwrap();
    assert_eq!(g.clique_number().unwrap(), 2);
    assert_eq!(g.independence_number().unwrap(), 5);
    assert!(!g.is_perfect().unwrap());
    let colors = g.vertex_coloring_greedy(ColoringGreedy::DSatur).unwrap();
    assert!(*colors.iter().max().unwrap() >= 3);
}

#[test]
fn bipartite_colorings_agree_with_the_bipartite_module() {
    for g in [binary_tree(31), cycle(10), path(7), petersen(), cycle(9)] {
        let types = g.bipartite_types().unwrap();
        assert_eq!(types.is_some(), g.is_bipartite().unwrap());
        let ds = g.vertex_coloring_greedy(ColoringGreedy::DSatur).unwrap();
        let used = *ds.iter().max().unwrap() + 1;
        match types {
            Some(types) => {
                assert_eq!(
                    g.is_bipartite_coloring(&types).unwrap(),
                    Some(NeighborMode::All)
                );
                // DSatur is exact on bipartite graphs, and a 2-coloring is
                // a bipartition.
                assert_eq!(used, 2);
                let as_bool: Vec<bool> = ds.iter().map(|&c| c == 1).collect();
                assert!(g.is_bipartite_coloring(&as_bool).unwrap().is_some());
                assert!(g.clique_number().unwrap() <= 2);
            }
            None => {
                // An odd cycle is present: at least 3 colors.
                assert!(used >= 3);
                let all_false = vec![false; g.vcount()];
                assert_eq!(g.is_bipartite_coloring(&all_false).unwrap(), None);
            }
        }
    }
}

#[test]
fn chordal_graphs_are_perfect_and_dsatur_is_optimal_on_them() {
    // A chordal "fan": vertex 0 joined to the path 1-2-3-4-5, plus a K4.
    let mut edges = vec![(1, 2), (2, 3), (3, 4), (4, 5)];
    edges.extend((1..=5).map(|v| (0, v)));
    edges.extend([(5, 6), (5, 7), (5, 8), (6, 7), (6, 8), (7, 8)]);
    let g = Graph::from_edges(&edges, 9, false).unwrap();
    assert!(g.is_chordal().unwrap());
    assert!(g.is_perfect().unwrap());
    // Perfect: χ = ω = 4 (the K4 {5, 6, 7, 8}), and the largest clique is
    // unique.
    assert_eq!(g.clique_number().unwrap(), 4);
    assert_eq!(canon(g.largest_cliques().unwrap()), vec![vec![5, 6, 7, 8]]);
    let colors = g.vertex_coloring_greedy(ColoringGreedy::DSatur).unwrap();
    assert!(g.is_vertex_coloring(&colors).unwrap());
    assert_eq!(*colors.iter().max().unwrap(), 3);
    // Degeneracy bound: ω ≤ degeneracy + 1.
    let degeneracy = *g.coreness(NeighborMode::All).unwrap().iter().max().unwrap();
    assert!(g.clique_number().unwrap() as i64 <= degeneracy + 1);
}

#[test]
fn random_graphs_satisfy_clique_identities() {
    for seed in 0..6 {
        rng::seed(seed).unwrap();
        let g = Graph::erdos_renyi_game_gnp(22, 0.35, false, EdgeTypeSw::Simple, false).unwrap();
        let comp = g.complementer(false).unwrap();
        let n = g.vcount();

        // Independent sets of G = cliques of its complement.
        assert_eq!(
            canon(g.independent_vertex_sets(.., None).unwrap()),
            canon(comp.cliques(.., None).unwrap())
        );
        assert_eq!(
            canon(g.maximal_independent_vertex_sets(.., None).unwrap()),
            canon(comp.maximal_cliques(.., None).unwrap())
        );
        assert_eq!(
            canon(g.largest_independent_vertex_sets().unwrap()),
            canon(comp.largest_cliques().unwrap())
        );
        let alpha = g.independence_number().unwrap();
        assert_eq!(alpha, comp.clique_number().unwrap());

        // Histograms agree with the enumerations.
        let omega = g.clique_number().unwrap();
        let hist = g.maximal_cliques_hist(..).unwrap();
        assert_eq!(hist.len(), omega);
        assert_eq!(
            hist.iter().sum::<usize>(),
            g.maximal_cliques_count(..).unwrap()
        );
        let all_hist = g.clique_size_hist(..).unwrap();
        assert_eq!(all_hist.len(), omega);
        assert_eq!((all_hist[0], all_hist[1]), (n, g.ecount()));
        assert_eq!(all_hist[2] as f64, g.count_triangles().unwrap());

        // Coloring bounds: ω ≤ χ ≤ greedy, and α χ ≥ n.
        for heuristic in [ColoringGreedy::ColoredNeighbors, ColoringGreedy::DSatur] {
            let colors = g.vertex_coloring_greedy(heuristic).unwrap();
            assert!(g.is_vertex_coloring(&colors).unwrap());
            let used = *colors.iter().max().unwrap() as usize + 1;
            assert!(used >= omega && alpha * used >= n, "seed {seed}");
            let max_degree = g.maxdegree(.., NeighborMode::All, Loops::None).unwrap() as usize;
            assert!(used <= max_degree + 1);
        }

        // Reseeding reproduces the graph, hence all results.
        rng::seed(seed).unwrap();
        let again =
            Graph::erdos_renyi_game_gnp(22, 0.35, false, EdgeTypeSw::Simple, false).unwrap();
        assert_eq!(
            canon(again.maximal_cliques(.., None).unwrap()),
            canon(g.maximal_cliques(.., None).unwrap())
        );
    }
}

#[test]
fn cliquer_searches_on_other_threads_are_independent() {
    // The nesting restriction of Cliquer-based searches is per thread: while
    // a `cliques_callback` runs here, other threads can use Cliquer freely,
    // and concurrent searches do not disturb each other.
    let g = complete(5);
    let mut outer = 0;
    g.cliques_callback(3..=3, |_| {
        outer += 1;
        std::thread::scope(|s| {
            let handles: Vec<_> = (4..8)
                .map(|n| {
                    s.spawn(move || {
                        let k = complete(n);
                        let hist = k.clique_size_hist(..).unwrap();
                        let mut streamed = 0;
                        k.cliques_callback(.., |_| {
                            streamed += 1;
                            ControlFlow::Continue(())
                        })
                        .unwrap();
                        (hist.iter().sum::<usize>(), streamed)
                    })
                })
                .collect();
            for (n, h) in (4..8).zip(handles) {
                let total = (1usize << n) - 1;
                assert_eq!(h.join().unwrap(), (total, total));
            }
        });
        ControlFlow::Continue(())
    })
    .unwrap();
    assert_eq!(outer, 10);
}

#[test]
fn failing_igraph_calls_inside_callbacks_do_not_break_the_search() {
    // igraph's error handler frees the temporary objects of the current
    // level of its "finally" stack: a nested call that fails must not free
    // those of the clique search that is still running.
    let g = karate();
    let total = g.maximal_cliques_count(..).unwrap();
    let mut n = 0;
    g.maximal_cliques_callback(.., |c| {
        // Wrong lengths: igraph itself reports these errors.
        assert_eq!(
            g.is_vertex_coloring(&[0]).unwrap_err().kind(),
            ErrorKind::InvalidValue
        );
        assert!(g.is_edge_coloring(&[0]).is_err());
        // A nested search that works still gives the right answer.
        assert!(g.maximal_cliques_count(c.len()..=c.len()).unwrap() > 0);
        n += 1;
        ControlFlow::Continue(())
    })
    .unwrap();
    assert_eq!(n, total);

    let mut triangles = 0;
    g.cliques_callback(3..=3, |_| {
        assert!(g.is_bipartite_coloring(&[true]).is_err());
        assert_eq!(g.maximal_cliques(5.., None).unwrap().len(), 2);
        triangles += 1;
        ControlFlow::Continue(())
    })
    .unwrap();
    assert_eq!(triangles, 45);
    // igraph's cleanup stack is balanced afterwards.
    assert_eq!(g.clique_number().unwrap(), 5);
}

#[test]
fn unweighted_weight_bounds_are_sizes() {
    let g = karate();
    // Without weights the bounds are sizes: a minimum above the vertex count
    // gives nothing (without asking Cliquer for cliques of a billion vertices).
    for maximal in [false, true] {
        let opts = WeightedCliqueOptions::default()
            .with_maximal(maximal)
            .with_min_weight(1e9);
        assert!(g.weighted_cliques(None, &opts).unwrap().is_empty());
        let opts = WeightedCliqueOptions::default()
            .with_maximal(maximal)
            .with_min_weight(5.0)
            .with_max_weight(1e9);
        assert_eq!(g.weighted_cliques(None, &opts).unwrap().len(), 2);
    }
    // With weights the bound is the total weight.
    let w = vec![2.0; 34];
    let opts = WeightedCliqueOptions::default().with_min_weight(69.0);
    assert!(g.weighted_cliques(Some(&w), &opts).unwrap().is_empty());
    let opts = WeightedCliqueOptions::default()
        .with_min_weight(10.0)
        .with_max_weight(68.0);
    assert_eq!(g.weighted_cliques(Some(&w), &opts).unwrap().len(), 2);
    // The null graph.
    let null = Graph::new(0, false);
    assert!(
        null.weighted_cliques(None, &WeightedCliqueOptions::default())
            .unwrap()
            .is_empty()
    );
    assert_eq!(null.independence_number().unwrap(), 0);
    assert_eq!(null.weighted_clique_number(Some(&[])).unwrap(), 0.0);
}

#[test]
fn multi_edges_and_loops_do_not_affect_valid_colorings() {
    // DSatur works on the simplified graph, so parallel edges and self-loops
    // do not change its result. The colored-neighbors heuristic counts
    // parallel edges when ordering the vertices (igraph_coloring.out, see
    // greedy_coloring_matches_igraph_unit_test), but its coloring stays valid.
    let simple = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], 5, false).unwrap();
    let multi = Graph::from_edges(
        &[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 2),
            (1, 2),
            (1, 2),
            (3, 3),
        ],
        5,
        false,
    )
    .unwrap();
    assert_eq!(
        simple
            .vertex_coloring_greedy(ColoringGreedy::DSatur)
            .unwrap(),
        multi
            .vertex_coloring_greedy(ColoringGreedy::DSatur)
            .unwrap()
    );
    for g in [&simple, &multi] {
        for h in [ColoringGreedy::ColoredNeighbors, ColoringGreedy::DSatur] {
            let c = g.vertex_coloring_greedy(h).unwrap();
            assert!(g.is_vertex_coloring(&c).unwrap());
            assert_eq!(*c.iter().max().unwrap(), 2);
        }
    }
}

/// A directed graph with mutual pairs: `0 <-> 1`, `1 <-> 2`, `0 <-> 2`,
/// `2 -> 3`. It has no multi-edges (igraph only calls same-direction
/// parallel edges multi-edges), and ignoring directions it is a triangle
/// with a pendant vertex.
fn mutual_triangle() -> Graph {
    Graph::from_edges(
        &[(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0), (2, 3)],
        4,
        true,
    )
    .unwrap()
}

fn sorted_sets(mut sets: Vec<Vec<i64>>) -> Vec<Vec<i64>> {
    sets.iter_mut().for_each(|s| s.sort());
    sets.sort();
    sets
}

/// Regression test for igraph's property-cache bug: the clique functions
/// build an `IGRAPH_ALL` / `IGRAPH_NO_MULTIPLE` adjacency list, which in
/// igraph 1.0.1 trusts a cached "no multi-edges" flag (so mutual pairs were
/// counted twice) and otherwise caches a wrong "has multi-edges" flag.
#[test]
fn directed_mutual_pairs_do_not_corrupt_the_property_cache() {
    // Every call runs on a fresh graph, either with an empty property cache
    // or with a (correct) "no multi-edges" flag cached by `has_multiple`,
    // and must leave the cache telling the truth, also for clones.
    fn run<T>(primed: bool, f: impl FnOnce(&Graph) -> T) -> T {
        let g = mutual_triangle();
        if primed {
            assert!(!g.has_multiple().unwrap());
        }
        let res = f(&g);
        assert!(!g.clone().has_multiple().unwrap());
        assert!(!g.has_multiple().unwrap());
        assert!(g.is_simple(true).unwrap());
        res
    }
    for primed in [false, true] {
        assert_eq!(run(primed, |g| g.clique_number().unwrap()), 3);
        assert_eq!(
            sorted_sets(run(primed, |g| g.maximal_cliques(.., None).unwrap())),
            vec![vec![0, 1, 2], vec![2, 3]]
        );
        assert_eq!(run(primed, |g| g.maximal_cliques_count(..).unwrap()), 2);
        assert_eq!(
            run(primed, |g| g.maximal_cliques_hist(..).unwrap()),
            vec![0, 1, 1]
        );
        assert_eq!(run(primed, |g| g.largest_cliques().unwrap()).len(), 1);
        assert_eq!(
            sorted_sets(run(primed, |g| g
                .maximal_cliques_subset(&[0, 1, 2, 3], .., None)
                .unwrap())),
            vec![vec![0, 1, 2], vec![2, 3]]
        );
        let n = run(primed, |g| {
            let mut n = 0;
            g.maximal_cliques_callback(.., |_| {
                n += 1;
                ControlFlow::Continue(())
            })
            .unwrap();
            n
        });
        assert_eq!(n, 2);
        let out = run(primed, |g| {
            let mut out = Vec::new();
            g.write_maximal_cliques(&mut out, .., None).unwrap();
            out
        });
        assert_eq!(String::from_utf8(out).unwrap().lines().count(), 2);
        assert_eq!(run(primed, |g| g.independence_number().unwrap()), 2);
        assert_eq!(
            sorted_sets(run(primed, |g| g
                .independent_vertex_sets(2.., None)
                .unwrap())),
            vec![vec![0, 3], vec![1, 3]]
        );
        assert_eq!(
            sorted_sets(run(primed, |g| g
                .largest_independent_vertex_sets()
                .unwrap())),
            vec![vec![0, 3], vec![1, 3]]
        );
        assert_eq!(
            sorted_sets(run(primed, |g| g
                .maximal_independent_vertex_sets(.., None)
                .unwrap())),
            vec![vec![0, 3], vec![1, 3], vec![2]]
        );
        run(primed, |g| {
            let colors = g.vertex_coloring_greedy(ColoringGreedy::DSatur).unwrap();
            assert!(g.is_vertex_coloring(&colors).unwrap());
            assert_eq!(colors.iter().max().copied(), Some(2));
        });
    }
}

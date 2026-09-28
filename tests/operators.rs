//! Integration tests for the graph operators (`igraph_operators.h`).

mod common;

use common::*;
use igraph::operators::{EdgeMapped, RewiringStats};
use igraph::prelude::*;

/// Undirected edge list normalized as sorted `(min, max)` pairs.
fn normalized(g: &Graph) -> Vec<(i64, i64)> {
    let mut e: Vec<(i64, i64)> = g
        .edge_list()
        .into_iter()
        .map(|(a, b)| {
            if g.is_directed() {
                (a, b)
            } else {
                (a.min(b), a.max(b))
            }
        })
        .collect();
    e.sort();
    e
}

fn degrees(g: &Graph) -> Vec<i64> {
    g.degree(.., NeighborMode::All, Loops::Twice).unwrap()
}

/// Number of triangles of a simple undirected graph (`isomorphism` module).
fn triangles(g: &Graph) -> usize {
    g.count_triangles().unwrap() as usize
}

/// Sorted edge list, for comparisons against igraph's canonical printouts.
fn sorted_edges(g: &Graph) -> Vec<(i64, i64)> {
    let mut e = g.edge_list();
    e.sort();
    e
}

// ---------------------------------------------------------------- unions

#[test]
fn disjoint_union_shifts_ids() {
    let a = path(3);
    let b = cycle(3);
    let u = a.disjoint_union(&b).unwrap();
    assert_eq!(u.vcount(), 6);
    assert_eq!(normalized(&u), vec![(0, 1), (1, 2), (3, 4), (3, 5), (4, 5)]);
    // The first operand's edges keep their ids.
    assert_eq!(u.edge(0).unwrap(), a.edge(0).unwrap());
}

#[test]
fn disjoint_union_many_and_empty_list() {
    let parts: Vec<Graph> = (2..5).map(complete).collect();
    let u = Graph::disjoint_union_many(&parts).unwrap();
    assert_eq!(u.vcount(), 2 + 3 + 4);
    assert_eq!(u.ecount(), 1 + 3 + 6);

    let none = Graph::disjoint_union_many(std::iter::empty()).unwrap();
    assert_eq!(none.vcount(), 0);
    assert!(
        none.is_directed(),
        "igraph documents a directed null graph here"
    );
}

#[test]
fn mixed_directedness_is_rejected() {
    let und = path(3);
    let dir = Graph::from_edges(&[(0, 1)], 2, true).unwrap();
    for res in [
        und.disjoint_union(&dir),
        und.union(&dir),
        und.intersection(&dir),
        und.join(&dir),
        und.compose(&dir),
        und.product(&dir, Product::Cartesian),
        Graph::union_many([&und, &dir]),
    ] {
        assert_eq!(res.unwrap_err().kind(), ErrorKind::InvalidValue);
    }
}

#[test]
fn union_matches_igraph_example() {
    // examples/simple/igraph_union.c
    let left =
        Graph::from_edges(&[(0, 1), (1, 2), (2, 2), (2, 3), (2, 3), (3, 2)], 4, true).unwrap();
    let right = Graph::from_edges(&[(0, 1), (1, 2), (2, 2), (2, 3), (5, 2)], 6, true).unwrap();
    let EdgeMapped {
        graph,
        edge_map1,
        edge_map2,
    } = left.union_map(&right).unwrap();
    assert_eq!(
        graph.edge_list(),
        vec![(0, 1), (1, 2), (2, 2), (2, 3), (2, 3), (3, 2), (5, 2)]
    );
    assert_eq!(edge_map1, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(edge_map2, vec![0, 1, 2, 3, 6]);
    assert_eq!(left.union(&right).unwrap(), graph);
}

#[test]
fn union_many_maps_every_edge() {
    let a = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    let b = Graph::from_edges(&[(1, 2), (2, 3)], 4, false).unwrap();
    let c = Graph::from_edges(&[(3, 0)], 4, false).unwrap();
    let res = Graph::union_many_map([&a, &b, &c]).unwrap();
    assert_eq!(normalized(&res.graph), vec![(0, 1), (0, 3), (1, 2), (2, 3)]);
    assert_eq!(res.edge_maps.len(), 3);
    for (g, map) in [&a, &b, &c].iter().zip(&res.edge_maps) {
        assert_eq!(map.len(), g.ecount());
        for (e, image) in map.iter().enumerate() {
            let (x, y) = g.edge(e as i64).unwrap();
            let image = image.expect("every edge is in the union");
            let (p, q) = res.graph.edge(image).unwrap();
            assert_eq!((x.min(y), x.max(y)), (p.min(q), p.max(q)));
        }
    }
    // The shared edge (1, 2) has the same image from a and b.
    assert_eq!(res.edge_maps[0][1], res.edge_maps[1][0]);
}

#[test]
fn union_takes_max_multiplicity_intersection_min() {
    let a = Graph::from_edges(&[(0, 1), (0, 1), (0, 1)], 2, false).unwrap();
    let b = Graph::from_edges(&[(0, 1)], 2, false).unwrap();
    assert_eq!(a.union(&b).unwrap().ecount(), 3);
    assert_eq!(a.intersection(&b).unwrap().ecount(), 1);
    assert_eq!(a.difference(&b).unwrap().ecount(), 2);
}

// ---------------------------------------------------------- intersections

#[test]
fn intersection_matches_igraph_example() {
    // examples/simple/igraph_intersection.c
    let left = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 0, true).unwrap();
    let right = Graph::from_edges(&[(1, 0), (5, 4), (1, 2), (3, 2)], 0, true).unwrap();
    let isec = left.intersection_map(&right).unwrap();
    assert_eq!(isec.graph.vcount(), 6);
    assert_eq!(isec.graph.edge_list(), vec![(1, 2)]);
    assert_eq!(isec.edge_map1, vec![1]);
    assert_eq!(isec.edge_map2, vec![2]);
}

#[test]
fn intersection_many_edge_cases() {
    // No operand: directed null graph.
    let none = Graph::intersection_many(std::iter::empty()).unwrap();
    assert_eq!(none.vcount(), 0);
    assert!(none.is_directed());

    // An empty operand kills every edge but not the vertices.
    let g1 = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 0, true).unwrap();
    let g2 = g1.clone();
    let g3 = Graph::new(10, true);
    let isec = Graph::intersection_many([&g1, &g2, &g3]).unwrap();
    assert_eq!((isec.vcount(), isec.ecount()), (10, 0));
}

#[test]
fn intersection_many_maps_are_indexed_by_operand_edges() {
    let a = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
    let b = Graph::from_edges(&[(3, 2), (0, 1)], 4, false).unwrap();
    let res = Graph::intersection_many_map([&a, &b]).unwrap();
    assert_eq!(normalized(&res.graph), vec![(0, 1), (2, 3)]);
    assert_eq!(res.edge_maps[0].len(), 3);
    assert_eq!(
        res.edge_maps[0][1], None,
        "edge (1, 2) of `a` is not shared"
    );
    assert_eq!(res.edge_maps[0][0], res.edge_maps[1][1]);
    assert_eq!(res.edge_maps[0][2], res.edge_maps[1][0]);
}

// ------------------------------------------------------------ difference

#[test]
fn difference_matches_igraph_examples() {
    // Subtracting a graph from itself leaves the vertices only.
    let orig = Graph::from_edges(&[(0, 1), (1, 2), (2, 1), (4, 5)], 0, true).unwrap();
    let d = orig.difference(&orig).unwrap();
    assert_eq!((d.vcount(), d.ecount()), (6, 0));

    // Subtracting the empty graph changes nothing.
    let d = orig.difference(&Graph::new(3, true)).unwrap();
    assert_eq!(d.edge_list(), orig.edge_list());

    // "real example, undirected"
    let orig = Graph::from_edges(
        &[
            (0, 1),
            (1, 2),
            (2, 1),
            (4, 5),
            (8, 9),
            (8, 10),
            (8, 13),
            (8, 11),
            (8, 12),
        ],
        0,
        false,
    )
    .unwrap();
    let sub = Graph::from_edges(
        &[(0, 1), (5, 4), (2, 1), (6, 7), (8, 10), (8, 13)],
        0,
        false,
    )
    .unwrap();
    assert_eq!(
        normalized(&orig.difference(&sub).unwrap()),
        vec![(1, 2), (8, 9), (8, 11), (8, 12)]
    );

    // GitHub issue #597: undirected with a self-loop.
    let ring10: Vec<(i64, i64)> = (0..10).map(|i| (i, (i + 1) % 10)).collect();
    let mut orig = Graph::from_edges(&[(0, 0)], 10, false).unwrap();
    orig.add_edges(&ring10).unwrap();
    let sub = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 0, false).unwrap();
    assert_eq!(
        normalized(&orig.difference(&sub).unwrap()),
        vec![(0, 0), (0, 9), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
    );
}

// -------------------------------------------------------- join/complement

#[test]
fn join_builds_complete_bipartite_and_cones() {
    let k34 = Graph::new(3, false).join(&Graph::new(4, false)).unwrap();
    assert_eq!(k34.ecount(), 12);
    assert_eq!(degrees(&k34), vec![4, 4, 4, 3, 3, 3, 3]);

    // Joining a single vertex to a cycle gives a wheel.
    let wheel = Graph::new(1, false).join(&cycle(5)).unwrap();
    assert_eq!(wheel.ecount(), 10);
    assert_eq!(degrees(&wheel), vec![5, 3, 3, 3, 3, 3]);

    // Directed joins add both directions.
    let a = Graph::from_edges(&[(0, 1)], 2, true).unwrap();
    let b = Graph::new(3, true);
    assert_eq!(a.join(&b).unwrap().ecount(), 1 + 2 * 2 * 3);
}

#[test]
fn complementer_matches_igraph_example() {
    let empty = Graph::new(5, true);
    assert_eq!(empty.complementer(true).unwrap().ecount(), 25);
    assert_eq!(empty.complementer(false).unwrap().ecount(), 20);

    // Complement of a loopless complete digraph with loops allowed: just loops.
    let full: Vec<(i64, i64)> = (0..5)
        .flat_map(|i| (0..5).filter(move |&j| j != i).map(move |j| (i, j)))
        .collect();
    let full = Graph::from_edges(&full, 5, true).unwrap();
    assert_eq!(
        full.complementer(true).unwrap().edge_list(),
        vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    );

    // Undirected: |E(G)| + |E(complement)| = n(n-1)/2.
    let g = karate();
    let c = g.complementer(false).unwrap();
    assert_eq!(g.ecount() + c.ecount(), 34 * 33 / 2);
    assert_eq!(g.intersection(&c).unwrap().ecount(), 0);
    assert_eq!(normalized(&c.complementer(false).unwrap()), normalized(&g));
}

// ---------------------------------------------------------------- compose

#[test]
fn compose_matches_igraph_example() {
    // Composition with an empty graph is empty, both ways.
    let e = Graph::new(5, true);
    let full: Vec<(i64, i64)> = (0..5)
        .flat_map(|i| (0..5).filter(move |&j| j != i).map(move |j| (i, j)))
        .collect();
    let full = Graph::from_edges(&full, 5, true).unwrap();
    let r = e.compose_map(&full).unwrap();
    assert_eq!(r.graph.ecount(), 0);
    assert!(r.edge_map1.is_empty() && r.edge_map2.is_empty());
    assert_eq!(full.compose(&e).unwrap().ecount(), 0);

    // Proper directed graphs.
    let g1 = Graph::from_edges(&[(0, 1), (1, 2), (5, 6)], 0, true).unwrap();
    let g2 = Graph::from_edges(&[(0, 1), (2, 4), (5, 6)], 0, true).unwrap();
    let r = g1.compose_map(&g2).unwrap();
    assert_eq!(r.graph.edge_list(), vec![(1, 4)]);
    assert_eq!((r.edge_map1, r.edge_map2), (vec![1], vec![1]));

    // Undirected: symmetric relations, so loops appear.
    let g1 = Graph::from_edges(&[(0, 1), (1, 2), (5, 6)], 0, false).unwrap();
    let g2 = Graph::from_edges(&[(0, 1), (0, 4), (5, 6)], 0, false).unwrap();
    let r = g1.compose_map(&g2).unwrap();
    assert_eq!(
        normalized(&r.graph),
        vec![(0, 0), (0, 2), (1, 1), (1, 4), (5, 5), (6, 6)]
    );
    assert_eq!(r.edge_map1, vec![0, 0, 0, 1, 2, 2]);
    assert_eq!(r.edge_map2, vec![0, 1, 0, 0, 2, 2]);
}

// ------------------------------------------------- contraction & permutation

#[test]
fn contract_vertices_keeps_all_edges() {
    let mut g = karate();
    // Map every vertex to its id modulo 3.
    let mapping: Vec<i64> = (0..34).map(|v| v % 3).collect();
    g.contract_vertices(&mapping).unwrap();
    assert_eq!((g.vcount(), g.ecount()), (3, 78));
    g.simplify(true, false).unwrap();
    assert_eq!(g.ecount(), 6, "3 loops + 3 connections between the classes");
}

#[test]
fn contract_vertices_validates_mapping() {
    let mut g = path(4);
    assert_eq!(
        g.contract_vertices(&[0, 1]).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.contract_vertices(&[0, -1, 0, 1]).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    // `i64::MAX` would make igraph compute `max + 1` with a signed overflow.
    assert_eq!(
        g.contract_vertices(&[0, i64::MAX, 0, 1])
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(g.vcount(), 4, "the graph is untouched after an error");
}

#[test]
fn permute_vertices_roundtrip() {
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 1)], 4, true).unwrap();
    let perm = [3, 0, 2, 1];
    let h = g.permute_vertices(&perm).unwrap();
    assert_eq!(h.ecount(), g.ecount());
    // Edge (a, b) of `g` becomes (inv[a], inv[b]) in `h`.
    let mut inv = [0i64; 4];
    for (new, &old) in perm.iter().enumerate() {
        inv[old as usize] = new as i64;
    }
    for e in g.edge_ids() {
        let (a, b) = g.edge(e).unwrap();
        assert_eq!(h.edge(e).unwrap(), (inv[a as usize], inv[b as usize]));
    }
    // Applying the inverse permutation gives the original graph back.
    assert_eq!(h.permute_vertices(&inv).unwrap(), g);

    let err = g.permute_vertices(&[0, 0, 1, 2]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    assert!(g.permute_vertices(&[0, 1]).is_err());
}

// --------------------------------------------------- neighborhood & power

#[test]
fn graph_power_of_cycles_and_paths() {
    let c = cycle(10);
    for k in 0..6 {
        let p = c.graph_power(k, false).unwrap();
        let expected_degree = (2 * k as i64).min(9);
        assert_eq!(degrees(&p), vec![expected_degree; 10], "power {k}");
    }
    // The 9th power of a 10-path is the complete graph.
    let p = path(10).graph_power(9, false).unwrap();
    assert_eq!(p.ecount(), 45);

    // The first power removes loops and multi-edges.
    let multi = Graph::from_edges(&[(0, 1), (0, 1), (1, 1)], 2, false).unwrap();
    assert_eq!(multi.graph_power(1, false).unwrap().edge_list().len(), 1);

    // Directed: the square of a directed path only goes forward.
    let dp = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, true).unwrap();
    let sq = dp.graph_power(2, true).unwrap();
    assert!(sq.is_directed());
    assert_eq!(
        normalized(&sq),
        vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    );
    assert!(!dp.graph_power(2, false).unwrap().is_directed());
}

#[test]
fn connect_neighborhood_modes() {
    let dp = || Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, true).unwrap();

    let mut out = dp();
    out.connect_neighborhood(3, NeighborMode::Out).unwrap();
    assert_eq!(out.ecount(), 6);
    assert_eq!(out.neighbors(0, NeighborMode::Out).unwrap(), vec![1, 2, 3]);

    // `In` follows the same paths backwards: the same edges are added.
    let mut inn = dp();
    inn.connect_neighborhood(3, NeighborMode::In).unwrap();
    let mut a = out.edge_list();
    let mut b = inn.edge_list();
    a.sort();
    b.sort();
    assert_eq!(a, b);

    // Two arrows into the same vertex: only `All` relates their tails.
    let vee = || Graph::from_edges(&[(0, 1), (2, 1)], 3, true).unwrap();
    let mut g = vee();
    g.connect_neighborhood(2, NeighborMode::Out).unwrap();
    assert_eq!(g.ecount(), 2);
    let mut g = vee();
    g.connect_neighborhood(2, NeighborMode::All).unwrap();
    assert_eq!(g.edge_list(), vec![(0, 1), (2, 1), (0, 2)]);

    // Same result as graph_power on a simple undirected graph.
    let mut g = cycle(8);
    g.connect_neighborhood(3, NeighborMode::All).unwrap();
    assert_eq!(
        normalized(&g),
        normalized(&cycle(8).graph_power(3, false).unwrap())
    );
}

// --------------------------------------------------------------- rewiring

#[test]
fn rewire_preserves_degrees() {
    rng::seed(42).unwrap();
    let mut g = karate();
    let before = degrees(&g);
    let RewiringStats { successful_swaps } = g.rewire(1000, EdgeTypeSw::Simple).unwrap();
    assert!(successful_swaps > 0 && successful_swaps <= 1000);
    assert_eq!(degrees(&g), before);
    assert_eq!(g.ecount(), 78);
    // Still simple: simplification removes nothing.
    let mut s = g.clone();
    s.simplify(true, true).unwrap();
    assert_eq!(s.ecount(), 78);
    // And (very likely) it is no longer the karate club.
    assert_ne!(normalized(&g), normalized(&karate()));
}

#[test]
fn rewire_directed_preserves_in_and_out_degrees() {
    rng::seed(7).unwrap();
    let edges: Vec<(i64, i64)> = (0..20)
        .flat_map(|i| [(i, (i + 1) % 20), (i, (i + 5) % 20)])
        .collect();
    let mut g = Graph::from_edges(&edges, 20, true).unwrap();
    g.rewire(500, EdgeTypeSw::Loops).unwrap();
    assert_eq!(
        g.degree(.., NeighborMode::Out, Loops::Twice).unwrap(),
        vec![2; 20]
    );
    assert_eq!(
        g.degree(.., NeighborMode::In, Loops::Twice).unwrap(),
        vec![2; 20]
    );
}

#[test]
fn rewire_is_reproducible_across_parallel_threads() {
    // Every thread owns its default generator: seeding it in one thread
    // does not disturb the others, so seeded runs agree even in parallel.
    let run = |seed: u64| {
        rng::seed(seed).unwrap();
        let mut g = Graph::famous("Zachary").unwrap();
        let stats = g.rewire(200, EdgeTypeSw::Simple).unwrap();
        (stats, sorted_edges(&g))
    };
    let reference = run(2024);
    let handles: Vec<_> = (0..4)
        .map(|i| std::thread::spawn(move || run(if i % 2 == 0 { 2024 } else { 7 })))
        .collect();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(results[0], reference);
    assert_eq!(results[2], reference);
    assert_eq!(results[1], results[3]);
    assert_ne!(
        results[1].1, reference.1,
        "different seeds, different graphs"
    );
}

#[test]
fn rewire_is_reproducible_with_a_private_rng() {
    // A private generator installed for the duration of the call gives the
    // same result as seeding the thread's default one with the same seed
    // (both are PCG32).
    let scoped = {
        let mut rng = Rng::new(RngType::Pcg32, 2024).unwrap();
        let mut g = karate();
        rng.scoped(|| g.rewire(200, EdgeTypeSw::Simple)).unwrap();
        g
    };
    rng::seed(2024).unwrap();
    let mut seeded = karate();
    seeded.rewire(200, EdgeTypeSw::Simple).unwrap();
    assert_eq!(scoped, seeded);
}

#[test]
fn rewire_rejects_multigraph_mode() {
    let mut g = karate();
    assert_eq!(
        g.rewire(10, EdgeTypeSw::Multi).unwrap_err().kind(),
        ErrorKind::Unimplemented
    );
    assert_eq!(normalized(&g), normalized(&karate()), "left untouched");
}

#[test]
fn rewire_tiny_graphs_reports_zero_swaps() {
    // igraph returns early without filling the statistics here.
    let mut single = path(2);
    let stats = single.rewire(100, EdgeTypeSw::Simple).unwrap();
    assert_eq!(stats, RewiringStats::default());
    assert_eq!(single.edge_list(), vec![(0, 1)]);
    // Any two edges of a triangle share a vertex, so every switch would
    // create a self-loop or a multi-edge: nothing can change.
    let mut tri = cycle(3);
    let stats = tri.rewire(100, EdgeTypeSw::Simple).unwrap();
    assert_eq!(stats.successful_swaps, 0);
    assert_eq!(normalized(&tri), normalized(&cycle(3)));
}

#[test]
fn oversized_counts_are_rejected_not_wrapped() {
    // `usize::MAX as i64` would be -1: the wrappers must refuse it instead.
    let mut g = cycle(5);
    assert_eq!(
        g.rewire(usize::MAX, EdgeTypeSw::Simple).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.connect_neighborhood(usize::MAX, NeighborMode::All)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.graph_power(usize::MAX, false).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.mycielskian(usize::MAX).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    // A representable but absurd iteration count overflows inside igraph.
    assert_eq!(g.mycielskian(200).unwrap_err().kind(), ErrorKind::Overflow);
    assert_eq!(normalized(&g), normalized(&cycle(5)));
}

// ------------------------------------------------------------- simplify

#[test]
fn simplify_variants() {
    let make = || {
        Graph::from_edges(
            &[(0, 0), (0, 1), (1, 0), (0, 1), (1, 2), (2, 2), (2, 2)],
            3,
            false,
        )
        .unwrap()
    };
    let mut g = make();
    g.simplify(false, false).unwrap();
    assert_eq!(g.ecount(), 7);

    let mut g = make();
    g.simplify(true, false).unwrap();
    assert_eq!(normalized(&g), vec![(0, 0), (0, 1), (1, 2), (2, 2)]);

    let mut g = make();
    g.simplify(false, true).unwrap();
    assert_eq!(normalized(&g), vec![(0, 1), (0, 1), (0, 1), (1, 2)]);

    let mut g = make();
    g.simplify(true, true).unwrap();
    assert_eq!(normalized(&g), vec![(0, 1), (1, 2)]);

    // Directed: (0, 1) and (1, 0) are different edges.
    let mut d = Graph::from_edges(&[(0, 1), (1, 0), (0, 1)], 2, true).unwrap();
    d.simplify(true, true).unwrap();
    assert_eq!(d.edge_list(), vec![(0, 1), (1, 0)]);
}

// ------------------------------------------------------------ subgraphs

#[test]
fn induced_subgraph_implementations_agree() {
    let g = karate();
    let vids: Vec<i64> = vec![33, 0, 1, 2, 3, 7, 13, 32, 8, 30];
    let a = g
        .induced_subgraph(&vids, SubgraphImplementation::CopyAndDelete)
        .unwrap();
    let b = g
        .induced_subgraph(&vids, SubgraphImplementation::CreateFromScratch)
        .unwrap();
    let c = g
        .induced_subgraph(&vids, SubgraphImplementation::Auto)
        .unwrap();
    assert_eq!(a.vcount(), 10);
    assert_eq!(normalized(&a), normalized(&b));
    assert_eq!(normalized(&a), normalized(&c));
    // The number of edges equals the count from induced_subgraph_edges.
    assert_eq!(a.ecount(), g.induced_subgraph_edges(&vids).unwrap().len());
}

#[test]
fn induced_subgraph_map_is_consistent() {
    let g = karate();
    let sub = g
        .induced_subgraph_map(
            vec![5, 4, 6, 10, 16, 5],
            SubgraphImplementation::CreateFromScratch,
        )
        .unwrap();
    assert_eq!(sub.invmap, vec![4, 5, 6, 10, 16]);
    assert_eq!(sub.map.len(), 34);
    for (new, &old) in sub.invmap.iter().enumerate() {
        assert_eq!(sub.map[old as usize], Some(new as i64));
    }
    assert_eq!(sub.map.iter().filter(|m| m.is_some()).count(), 5);
    // Every edge of the subgraph is an edge of the original graph.
    for (a, b) in sub.graph.edge_list() {
        let (oa, ob) = (sub.invmap[a as usize], sub.invmap[b as usize]);
        assert!(g.get_eid(oa, ob, false).unwrap().is_some());
    }
    assert_eq!(sub.graph.ecount(), 6);
}

#[test]
fn induced_subgraph_with_ranges_and_errors() {
    let g = cycle(6);
    let half = g
        .induced_subgraph(0..3, SubgraphImplementation::Auto)
        .unwrap();
    assert_eq!(normalized(&half), vec![(0, 1), (1, 2)]);
    let all = g
        .induced_subgraph(.., SubgraphImplementation::Auto)
        .unwrap();
    assert_eq!(normalized(&all), normalized(&g));

    let err = g
        .induced_subgraph(&[0, 17], SubgraphImplementation::Auto)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    let err = g.induced_subgraph_edges(&[0, 17]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    assert!(
        g.induced_subgraph_map(99, SubgraphImplementation::Auto)
            .is_err()
    );
}

#[test]
fn subgraph_from_edges_variants() {
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 5, true).unwrap();
    let s = g.subgraph_from_edges(&[3, 0], true).unwrap();
    // Edges keep their relative order; vertices 0, 1, 3, 4 remain.
    assert_eq!(s.vcount(), 4);
    assert_eq!(s.edge_list(), vec![(0, 1), (2, 3)]);
    let s = g.subgraph_from_edges(&[3, 0], false).unwrap();
    assert_eq!(s.vcount(), 5);
    assert_eq!(s.edge_list(), vec![(0, 1), (3, 4)]);
    let none = g.subgraph_from_edges(EdgeSelector::None, true).unwrap();
    assert_eq!((none.vcount(), none.ecount()), (0, 0));
    let incident = g.subgraph_from_edges(
        EdgeSelector::Incident {
            vertex: 0,
            mode: NeighborMode::All,
        },
        false,
    );
    assert_eq!(incident.unwrap().ecount(), 2);

    let err = g.subgraph_from_edges(&[0, 42], true).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidEdgeId);
}

#[test]
fn reverse_edges_selected_and_all() {
    let mut g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], 4, true).unwrap();
    g.reverse_edges(&[0, 2]).unwrap();
    assert_eq!(g.edge_list(), vec![(1, 0), (1, 2), (3, 2), (3, 0)]);
    g.reverse_edges(..).unwrap();
    assert_eq!(g.edge_list(), vec![(0, 1), (2, 1), (2, 3), (0, 3)]);
    assert_eq!(
        g.reverse_edges(&[7]).unwrap_err().kind(),
        ErrorKind::InvalidEdgeId
    );

    let mut u = path(3);
    u.reverse_edges(..).unwrap();
    assert_eq!(normalized(&u), normalized(&path(3)));
    // Undirected graphs are a no-op: ids are not even checked.
    assert!(u.reverse_edges(&[42]).is_ok());

    // An id listed twice is flipped twice.
    let mut d = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    d.reverse_edges(&[0, 0, 1]).unwrap();
    assert_eq!(d.edge_list(), vec![(0, 1), (2, 1)]);
}

// -------------------------------------------------------------- products

#[test]
fn product_edge_counts_follow_the_formulas() {
    let g1 = path(3); // |V1| = 3, |E1| = 2
    let g2 = cycle(4); // |V2| = 4, |E2| = 4
    let (v1, e1, v2, e2) = (3, 2, 4, 4);
    let count = |kind| g1.product(&g2, kind).unwrap().ecount();
    assert_eq!(count(Product::Cartesian), v1 * e2 + v2 * e1);
    assert_eq!(count(Product::Lexicographic), v1 * e2 + v2 * v2 * e1);
    assert_eq!(count(Product::Strong), v1 * e2 + v2 * e1 + 2 * e1 * e2);
    assert_eq!(count(Product::Tensor), 2 * e1 * e2);
    // Complements: P3 has 1 non-edge, C4 has 2.
    assert_eq!(count(Product::Modular), 2 * e1 * e2 + 2 * 2);
    for kind in [Product::Cartesian, Product::Tensor, Product::Modular] {
        assert_eq!(g1.product(&g2, kind).unwrap().vcount(), v1 * v2);
    }
}

#[test]
fn product_vertex_numbering_and_hypercube() {
    // (u, v) is vertex u * |V2| + v: in K2 x K2, (0,0)=0 is adjacent to (0,1)=1 and (1,0)=2.
    let k2 = complete(2);
    let sq = k2.product(&k2, Product::Cartesian).unwrap();
    assert_eq!(sq.neighbors(0, NeighborMode::All).unwrap(), vec![1, 2]);
    // K2^3 is the 3-cube: 8 vertices, 3-regular, 12 edges.
    let cube = sq.product(&k2, Product::Cartesian).unwrap();
    assert_eq!((cube.vcount(), cube.ecount()), (8, 12));
    assert_eq!(degrees(&cube), vec![3; 8]);
    // The tensor product of K2 with itself is a perfect matching.
    let t = k2.product(&k2, Product::Tensor).unwrap();
    assert_eq!(normalized(&t), vec![(0, 3), (1, 2)]);
}

#[test]
fn product_directed_and_non_commutative() {
    let a = Graph::from_edges(&[(0, 1)], 2, true).unwrap();
    let b = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    assert_eq!(a.product(&b, Product::Tensor).unwrap().ecount(), 2);
    assert_eq!(
        a.product(&b, Product::Strong).unwrap().ecount(),
        2 * 2 + 3 + 2
    );
    // Lexicographic: |V1||E2| + |V2|^2 |E1| differs when swapping operands.
    let ab = a.product(&b, Product::Lexicographic).unwrap().ecount();
    let ba = b.product(&a, Product::Lexicographic).unwrap().ecount();
    assert_eq!((ab, ba), (2 * 2 + 9, 3 + 4 * 2));
}

#[test]
fn modular_product_requires_simple_graphs() {
    let multi = Graph::from_edges(&[(0, 1), (0, 1)], 2, false).unwrap();
    let err = multi.product(&path(3), Product::Modular).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

#[test]
fn rooted_product_attaches_copies() {
    // Rooted product of a triangle with a star rooted at its center:
    // every triangle vertex gets 3 pendant leaves.
    let star = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], 4, false).unwrap();
    let tri = cycle(3);
    let r = tri.rooted_product(&star, 0).unwrap();
    assert_eq!((r.vcount(), r.ecount()), (12, 3 * 3 + 3));
    let d = degrees(&r);
    for u in 0..3 {
        assert_eq!(
            d[u * 4],
            5,
            "root of copy {u}: 3 leaves + 2 triangle neighbours"
        );
    }
    // Rooting at a leaf gives a different (non-isomorphic) degree profile.
    let r2 = tri.rooted_product(&star, 1).unwrap();
    assert_eq!(r2.ecount(), 12);
    assert_eq!(degrees(&r2)[1], 3);

    assert_eq!(
        tri.rooted_product(&star, 9).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
}

// ----------------------------------------------------------- Mycielskian

#[test]
fn mycielskian_matches_igraph_unit_test() {
    // tests/unit/mycielskian.c: k = 0 is the identity.
    let g = Graph::from_edges(&[(0, 1), (0, 3), (1, 2), (2, 4), (2, 3), (3, 4)], 5, false).unwrap();
    assert_eq!(normalized(&g.mycielskian(0).unwrap()), normalized(&g));

    // Null graph: k = 1 is the singleton, k = 3 is the 5-cycle.
    let null = Graph::new(0, false);
    let m1 = null.mycielskian(1).unwrap();
    assert_eq!((m1.vcount(), m1.ecount()), (1, 0));
    let m3 = null.mycielskian(3).unwrap();
    assert_eq!(m3.vcount(), 5);
    assert_eq!(
        normalized(&m3),
        vec![(0, 1), (0, 3), (1, 2), (2, 4), (3, 4)]
    );

    // Singleton, k = 3, and the 2-path, k = 2, both give the Grötzsch graph.
    let expected = vec![
        (0, 1),
        (0, 3),
        (0, 6),
        (0, 8),
        (1, 2),
        (1, 5),
        (1, 7),
        (2, 4),
        (2, 6),
        (2, 9),
        (3, 4),
        (3, 5),
        (3, 9),
        (4, 7),
        (4, 8),
        (5, 10),
        (6, 10),
        (7, 10),
        (8, 10),
        (9, 10),
    ];
    assert_eq!(
        normalized(&Graph::new(1, false).mycielskian(3).unwrap()),
        expected
    );
    assert_eq!(normalized(&path(2).mycielskian(2).unwrap()), expected);
}

#[test]
fn mycielskian_is_triangle_free_and_grows_as_documented() {
    // n_k = (n + 1) 2^k - 1, m_k = ((2m + 2n + 1) 3^k - n_{k+1}) / 2.
    let g = cycle(5);
    let (n, m) = (5i64, 5i64);
    for k in 0..3u32 {
        let mk = g.mycielskian(k as usize).unwrap();
        let nk = (n + 1) * 2i64.pow(k) - 1;
        let nk1 = (n + 1) * 2i64.pow(k + 1) - 1;
        assert_eq!(mk.vcount() as i64, nk);
        assert_eq!(
            mk.ecount() as i64,
            ((2 * m + 2 * n + 1) * 3i64.pow(k) - nk1) / 2
        );
        assert_eq!(
            triangles(&mk),
            0,
            "Mycielski preserves triangle-freeness (k = {k})"
        );
    }
    // Directed graphs are supported too.
    let d = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    let md = d.mycielskian(1).unwrap();
    assert!(md.is_directed());
    assert_eq!((md.vcount(), md.ecount()), (7, 3 * 2 + 3));
}

// ------------------------------------------- igraph unit-test transcripts

/// The directed multigraph used by igraph's `igraph_connect_neighborhood.c`
/// and `igraph_contract_vertices.c` unit tests.
fn unit_test_multigraph(directed: bool) -> Graph {
    Graph::from_edges(
        &[(0, 1), (0, 2), (1, 1), (1, 3), (2, 3), (3, 4), (3, 4)],
        6,
        directed,
    )
    .unwrap()
}

#[test]
fn connect_neighborhood_matches_igraph_unit_test() {
    // tests/unit/igraph_connect_neighborhood.out (edges printed sorted).
    let run = |order: usize, mode: NeighborMode, directed: bool| {
        let mut g = unit_test_multigraph(directed);
        g.connect_neighborhood(order, mode).unwrap();
        assert_eq!(g.is_directed(), directed);
        assert_eq!(g.vcount(), 6);
        sorted_edges(&g)
    };
    let original = vec![(0, 1), (0, 2), (1, 1), (1, 3), (2, 3), (3, 4), (3, 4)];
    assert_eq!(run(0, NeighborMode::Out, true), original);
    assert_eq!(run(1, NeighborMode::Out, true), original);
    let order2 = vec![
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 1),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
        (3, 4),
    ];
    assert_eq!(run(2, NeighborMode::Out, true), order2);
    assert_eq!(run(2, NeighborMode::In, true), order2);
    let all2 = vec![
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
        (3, 4),
    ];
    assert_eq!(run(2, NeighborMode::All, true), all2);
    let order3 = vec![
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 1),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
        (3, 4),
    ];
    assert_eq!(run(3, NeighborMode::Out, true), order3);
    assert_eq!(run(12, NeighborMode::In, true), order3);
    // Undirected: `mode` is ignored, same edges as `All` above.
    assert_eq!(
        normalized(&{
            let mut g = unit_test_multigraph(false);
            g.connect_neighborhood(2, NeighborMode::Out).unwrap();
            g
        }),
        all2
    );
    // The null graph is fine.
    let mut null = Graph::new(0, false);
    null.connect_neighborhood(10, NeighborMode::All).unwrap();
    assert_eq!((null.vcount(), null.ecount()), (0, 0));
}

#[test]
fn graph_power_matches_igraph_unit_test() {
    // tests/unit/igraph_graph_power.out
    let g = |edges: &[(i64, i64)]| Graph::from_edges(edges, 3, true).unwrap();
    let triangle = vec![(0, 1), (0, 2), (1, 2)];
    // A->B->C
    let abc = g(&[(0, 1), (1, 2)]);
    assert_eq!(sorted_edges(&abc.graph_power(2, true).unwrap()), triangle);
    assert_eq!(normalized(&abc.graph_power(2, false).unwrap()), triangle);
    // A->B<-C: nothing new when following directions.
    let vee = g(&[(0, 1), (2, 1)]);
    assert_eq!(
        sorted_edges(&vee.graph_power(2, true).unwrap()),
        vec![(0, 1), (2, 1)]
    );
    assert_eq!(normalized(&vee.graph_power(2, false).unwrap()), triangle);
    // A<-B<-C
    let cba = g(&[(1, 0), (2, 1)]);
    assert_eq!(
        sorted_edges(&cba.graph_power(2, true).unwrap()),
        vec![(1, 0), (2, 0), (2, 1)]
    );
    // A->B->C->A: the square of a directed 3-cycle is the complete digraph.
    let c3 = g(&[(0, 1), (1, 2), (2, 0)]);
    assert_eq!(
        c3.graph_power(2, true).unwrap(),
        Graph::full(3, true, false).unwrap()
    );
    // A->B<->C->A. igraph's own transcript prints the undirected square
    // with a double 1-2 edge: the directed call before it cached "no
    // multi-edges" on the input, and igraph then skips the deduplication of
    // the mutual pair. Called the other way round, raw igraph caches "has
    // multi-edges" and the directed call then aborts on an internal
    // assertion. The wrapper avoids both, in any call order.
    let mutual = g(&[(0, 1), (1, 2), (2, 1), (2, 0)]);
    assert!(mutual.is_simple(true).unwrap());
    assert_eq!(normalized(&mutual.graph_power(2, false).unwrap()), triangle);
    assert_eq!(normalized(&mutual.graph_power(1, false).unwrap()), triangle);
    assert_eq!(
        mutual.graph_power(2, true).unwrap(),
        Graph::full(3, true, false).unwrap()
    );
    assert_eq!(normalized(&mutual.graph_power(2, false).unwrap()), triangle);
    assert!(mutual.is_simple(true).unwrap());
    for order in 1..4 {
        let p = mutual.graph_power(order, false).unwrap();
        assert!(p.is_simple(true).unwrap(), "order {order}");
    }
    // Order 0 keeps the vertices only; directed null graph stays directed.
    let z = unit_test_multigraph(true).graph_power(0, true).unwrap();
    assert_eq!((z.vcount(), z.ecount(), z.is_directed()), (6, 0, true));
    let null = Graph::new(0, true).graph_power(10, false).unwrap();
    assert_eq!((null.vcount(), null.is_directed()), (0, false));
}

#[test]
fn graph_power_leaves_no_stale_cache_behind() {
    // Raw igraph 1.0.1 would cache "has multi-edges" on this simple digraph
    // during the undirected power, then abort the process in the directed
    // one (or report a multi-edge that does not exist).
    let g = Graph::from_edges(&[(0, 1), (1, 0), (1, 2)], 3, true).unwrap();
    let und = g.graph_power(2, false).unwrap();
    assert_eq!(normalized(&und), vec![(0, 1), (0, 2), (1, 2)]);
    assert!(!g.has_multiple().unwrap());
    let dir = g.graph_power(2, true).unwrap();
    assert_eq!(sorted_edges(&dir), vec![(0, 1), (0, 2), (1, 0), (1, 2)]);
    assert!(g.is_simple(true).unwrap());
}

#[test]
fn permute_vertices_matches_igraph_unit_test() {
    // tests/unit/igraph_permute_vertices.out
    let g = Graph::from_edges(&[(0, 1), (1, 0), (2, 1), (2, 2), (5, 4), (5, 4)], 6, true).unwrap();
    let mut perm = vec![0, 1, 5, 3, 4, 2];
    let h = g.permute_vertices(&perm).unwrap();
    assert_eq!(
        h.edge_list(),
        vec![(0, 1), (1, 0), (5, 1), (5, 5), (2, 4), (2, 4)]
    );
    assert!(h.isomorphic(&g).unwrap());
    // Out-of-range value, duplicate value, wrong length: all EINVAL.
    perm[0] = 7;
    assert_eq!(
        g.permute_vertices(&perm).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    perm[0] = 1;
    assert_eq!(
        g.permute_vertices(&perm).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.permute_vertices(&[0, 1, 2, 3, 4]).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn contract_vertices_matches_igraph_unit_test() {
    // tests/unit/igraph_contract_vertices.out: the successive contractions
    // 5 -> 4, 4 -> 3, 3 -> 2 and 2 -> 0 of a directed multigraph with loops.
    let mut g = Graph::from_edges(
        &[
            (0, 1),
            (0, 2),
            (1, 1),
            (1, 3),
            (2, 0),
            (2, 3),
            (3, 4),
            (3, 4),
        ],
        6,
        true,
    )
    .unwrap();
    let before = g.edge_list();
    g.contract_vertices(&[0, 1, 2, 3, 4, 5]).unwrap();
    assert_eq!(
        g.edge_list(),
        before,
        "the identity mapping changes nothing"
    );
    g.contract_vertices(&[0, 1, 2, 3, 4, 4]).unwrap();
    assert_eq!(g.vcount(), 5);
    assert_eq!(g.edge_list(), before, "vertex 5 was isolated");
    g.contract_vertices(&[0, 1, 2, 3, 3]).unwrap();
    assert_eq!(
        g.edge_list(),
        vec![
            (0, 1),
            (0, 2),
            (1, 1),
            (1, 3),
            (2, 0),
            (2, 3),
            (3, 3),
            (3, 3)
        ]
    );
    g.contract_vertices(&[0, 1, 2, 2]).unwrap();
    assert_eq!(
        g.edge_list(),
        vec![
            (0, 1),
            (0, 2),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 2),
            (2, 2),
            (2, 2)
        ]
    );
    g.contract_vertices(&[0, 1, 0]).unwrap();
    assert_eq!(g.vcount(), 2);
    assert_eq!(g.ecount(), 8, "contraction never removes edges");
    // The null graph with an empty mapping.
    let mut null = Graph::new(0, true);
    null.contract_vertices(&[]).unwrap();
    assert_eq!(null.vcount(), 0);
}

// ------------------------------------------- identities with other modules

#[test]
fn operators_rebuild_named_graphs() {
    // Join: K(m,n) and wheels.
    let k34 = Graph::new(3, false).join(&Graph::new(4, false)).unwrap();
    assert_eq!(
        k34,
        Graph::full_bipartite(3, 4, false, NeighborMode::All)
            .unwrap()
            .graph
    );
    let wheel = Graph::new(1, false)
        .join(&Graph::ring(5, false, false, true).unwrap())
        .unwrap();
    assert!(
        wheel
            .isomorphic(&Graph::wheel(6, WheelMode::Undirected, 0).unwrap())
            .unwrap()
    );

    // Complement: of the edgeless graph is K_n; of L(K5) is the Petersen graph.
    assert_eq!(
        Graph::new(6, false).complementer(false).unwrap(),
        Graph::full(6, false, false).unwrap()
    );
    let petersen = Graph::full(5, false, false)
        .unwrap()
        .linegraph()
        .unwrap()
        .complementer(false)
        .unwrap();
    assert!(
        petersen
            .isomorphic(&Graph::famous("Petersen").unwrap())
            .unwrap()
    );

    // Products: K2^d is the d-cube, P_m x P_n the m x n grid, C_m x C_n the torus.
    let k2 = Graph::full(2, false, false).unwrap();
    let mut cube = k2.clone();
    for d in 2..=4 {
        cube = cube.product(&k2, Product::Cartesian).unwrap();
        assert!(
            cube.isomorphic(&Graph::hypercube(d, false).unwrap())
                .unwrap()
        );
    }
    let torus = cycle(4).product(&cycle(5), Product::Cartesian).unwrap();
    let lattice = Graph::square_lattice(&[5, 4], 1, false, false, Some(&[true, true])).unwrap();
    assert!(torus.isomorphic(&lattice).unwrap());

    // Mycielskian: iterating from K2 gives the Mycielski graphs M_k.
    for k in 2..=5 {
        let m = k2.mycielskian(k - 2).unwrap();
        assert!(
            m.isomorphic(&Graph::mycielski_graph(k).unwrap()).unwrap(),
            "M_{k}"
        );
        assert_eq!(triangles(&m), 0);
    }
}

#[test]
fn operators_preserve_invariants() {
    let g = Graph::famous("Zachary").unwrap();
    // Relabeling keeps the structure: same canonical form, same triangles.
    let mut perm: Vec<i64> = (0..34).rev().collect();
    perm.swap(0, 17);
    let h = g.permute_vertices(&perm).unwrap();
    assert!(h.isomorphic(&g).unwrap());
    assert_eq!(
        h.canonical_form(None).unwrap(),
        g.canonical_form(None).unwrap()
    );
    assert_eq!(triangles(&h), 45, "Zachary's karate club has 45 triangles");

    // Disjoint union and decomposition are inverse to each other.
    let parts = [cycle(3), path(4), complete(5)];
    let u = Graph::disjoint_union_many(&parts).unwrap();
    let back = u.decompose(Connectedness::Weak, None, 1).unwrap();
    assert_eq!(back.len(), 3);
    for (a, b) in parts.iter().zip(&back) {
        assert!(a.isomorphic(b).unwrap());
    }

    // Induced subgraphs agree with deleting the other vertices.
    let keep: Vec<i64> = (0..34).filter(|v| v % 3 != 0).collect();
    let drop: Vec<i64> = (0..34).filter(|v| v % 3 == 0).collect();
    let mut deleted = g.clone();
    deleted.delete_vertices(&drop).unwrap();
    assert_eq!(
        g.induced_subgraph(&keep, SubgraphImplementation::Auto)
            .unwrap(),
        deleted
    );

    // Simplification makes any graph simple.
    let mut multi = unit_test_multigraph(false);
    assert!(!multi.is_simple(true).unwrap());
    multi.simplify(true, true).unwrap();
    assert!(multi.is_simple(true).unwrap());
    assert_eq!(
        normalized(&multi),
        vec![(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
    );
}

// ------------------------------------------------------------ use cases

/// A story: two snapshots of a friendship network, a year apart.
///
/// Set operators answer the natural questions — which friendships lasted,
/// which are new, which were lost — and structural operators summarize the
/// network at the level of groups.
#[test]
fn use_case_friendship_snapshots() {
    // 2023: the karate club as observed by Zachary.
    let before = Graph::famous("Zachary").unwrap();
    assert_eq!(before, karate(), "same labelled graph as KARATE_EDGES");
    // 2024: the club split. Friendships crossing the two factions are gone,
    // and a few new friendships appeared inside the factions.
    let mr_hi: Vec<i64> = vec![0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21];
    let faction = |v: i64| mr_hi.contains(&v);
    let kept: Vec<(i64, i64)> = KARATE_EDGES
        .iter()
        .copied()
        .filter(|&(a, b)| faction(a) == faction(b))
        .collect();
    let mut after = Graph::from_edges(&kept, 34, false).unwrap();
    let new_friendships = [(4, 5), (16, 10), (24, 26)];
    after.add_edges(&new_friendships).unwrap();

    // Lasting friendships: the intersection.
    let lasting = before.intersection(&after).unwrap();
    assert_eq!(lasting.ecount(), kept.len());
    // New friendships: what is in `after` but not in `before`.
    let new = after.difference(&before).unwrap();
    assert_eq!(normalized(&new), vec![(4, 5), (10, 16), (24, 26)]);
    // Lost friendships: exactly the ones crossing the factions.
    let lost = before.difference(&after).unwrap();
    assert_eq!(lost.ecount(), 78 - kept.len());
    assert!(
        lost.edge_list()
            .iter()
            .all(|&(a, b)| faction(a) != faction(b))
    );
    // Everything that ever existed: the union, with |A ∪ B| = |A| + |B| - |A ∩ B|.
    let ever = before.union(&after).unwrap();
    assert_eq!(
        ever.ecount(),
        before.ecount() + after.ecount() - lasting.ecount()
    );

    // Zoom out: contract each faction to a single vertex in the old network.
    let mut factions = before.clone();
    let mapping: Vec<i64> = (0..34).map(|v| if faction(v) { 0 } else { 1 }).collect();
    factions.contract_vertices(&mapping).unwrap();
    let crossing = factions.get_all_eids_between(0, 1, false).unwrap().len();
    assert_eq!(
        crossing,
        lost.ecount(),
        "the split cut exactly the crossing ties"
    );

    // Zoom in: Mr. Hi's faction as its own graph, before and after.
    let hi_before = before
        .induced_subgraph(&mr_hi, SubgraphImplementation::Auto)
        .unwrap();
    let hi_after = after
        .induced_subgraph(&mr_hi, SubgraphImplementation::Auto)
        .unwrap();
    assert_eq!(
        hi_after.ecount(),
        hi_before.ecount() + 2,
        "(4, 5) and (10, 16) are new"
    );
}

/// A second story: building a small "data center" topology from pieces.
///
/// Racks are complete graphs, racks are connected through a ring of
/// switches via the rooted product, and a backup network is the square of
/// the ring, so that any single switch failure keeps the ring connected.
#[test]
fn use_case_network_topology() {
    // A ring of 4 switches.
    let ring = cycle(4);
    // Each switch serves a rack: a star with the switch as its root (vertex 0).
    let rack = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], 4, false).unwrap();
    let dc = ring.rooted_product(&rack, 0).unwrap();
    assert_eq!(dc.vcount(), 16);
    // Switches are vertices 0, 4, 8, 12; each has 3 servers + 2 ring neighbours.
    for s in [0, 4, 8, 12] {
        assert_eq!(dc.degree_of(s, NeighborMode::All, Loops::Twice).unwrap(), 5);
    }
    // Servers only talk to their switch.
    assert_eq!(dc.neighbors(5, NeighborMode::All).unwrap(), vec![4]);

    // The backbone is the subgraph spanned by the ring edges between switches.
    let switches = [0i64, 4, 8, 12];
    let backbone_edges = dc.induced_subgraph_edges(&switches).unwrap();
    assert_eq!(backbone_edges.len(), 4);
    let backbone = dc.subgraph_from_edges(&backbone_edges, true).unwrap();
    assert_eq!(degrees(&backbone), vec![2; 4]);

    // Add "express links": the square of the backbone is K4.
    let express = backbone.graph_power(2, false).unwrap();
    assert_eq!(express.ecount(), 6);
    // Removing any switch from the express network keeps the others fully connected.
    for s in 0..4 {
        let remaining: Vec<i64> = (0..4).filter(|&v| v != s).collect();
        let sub = express
            .induced_subgraph(&remaining, SubgraphImplementation::Auto)
            .unwrap();
        assert_eq!(sub.ecount(), 3);
    }
    // Two data centers side by side, then fully cross-connected backbones via a join.
    let two = backbone.join(&backbone).unwrap();
    assert_eq!(two.ecount(), 4 + 4 + 16);
}

/// The plain `simplify` drops edge attributes whenever it rebuilds the edge
/// set to merge multi-edges, but not when it only deletes loops.
#[test]
fn simplify_edge_attributes_follow_the_docs() {
    igraph::attributes::enable().unwrap();
    let mut loops_only = Graph::from_edges(&[(0, 1), (1, 1), (1, 2)], 3, false).unwrap();
    loops_only
        .set_edge_attr_numeric_values("w", &[1.0, 2.0, 3.0])
        .unwrap();
    loops_only.simplify(false, true).unwrap();
    assert_eq!(
        loops_only.edge_attr_numeric_values("w", ..).unwrap(),
        vec![1.0, 3.0]
    );

    let mut multi = Graph::from_edges(&[(0, 1), (0, 1), (1, 2)], 3, false).unwrap();
    multi
        .set_edge_attr_numeric_values("w", &[1.0, 2.0, 3.0])
        .unwrap();
    multi.simplify(true, true).unwrap();
    assert_eq!(multi.edge_list(), vec![(0, 1), (1, 2)]);
    assert!(!multi.has_attribute(igraph::attributes::AttributeKind::Edge, "w"));

    // Merging is attempted (and edge attributes dropped) even when the graph
    // turns out to have no multi-edges, while graph and vertex attributes
    // survive.
    let mut simple = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    simple
        .set_edge_attr_numeric_values("w", &[1.0, 2.0])
        .unwrap();
    simple
        .set_vertex_attr_str_values("name", &["a", "b", "c"])
        .unwrap();
    simple.set_graph_attr_numeric("year", 2026.0).unwrap();
    simple.simplify(true, false).unwrap();
    assert_eq!(simple.edge_list(), vec![(0, 1), (1, 2)]);
    assert!(!simple.has_attribute(igraph::attributes::AttributeKind::Edge, "w"));
    assert_eq!(
        simple.vertex_attr_str_values("name", ..).unwrap(),
        ["a", "b", "c"]
    );
    assert_eq!(simple.graph_attr_numeric("year").unwrap(), 2026.0);
}

//! Integration tests for the random graph generators (`igraph_games.h`).

mod common;

use common::assert_close;
use igraph::games::{
    AllowedEdgeTypes, BarabasiAgingOptions, BarabasiOptions, RecentDegreeAgingOptions,
    RecentDegreeOptions,
};
use igraph::misc::Metric;
use igraph::prelude::*;
use std::collections::{BTreeSet, HashSet};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Edges in canonical form (sorted endpoints for undirected graphs), sorted:
/// igraph's `print_graph_canon`, used by its unit tests' expected outputs.
fn canon(g: &Graph) -> Vec<(i64, i64)> {
    let mut e: Vec<_> = g
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

fn is_simple(g: &Graph) -> bool {
    g.is_simple(true).unwrap()
}

fn has_loops(g: &Graph) -> bool {
    g.has_loop().unwrap()
}

fn has_multi(g: &Graph) -> bool {
    g.has_multiple().unwrap()
}

fn is_connected(g: &Graph) -> bool {
    g.is_connected(Connectedness::Weak).unwrap()
}

fn degrees(g: &Graph, mode: NeighborMode) -> Vec<i64> {
    g.degree(.., mode, Loops::Twice).unwrap()
}

fn maxdeg(g: &Graph) -> i64 {
    g.maxdegree(.., NeighborMode::All, Loops::Twice).unwrap()
}

// ---------------------------------------------------------------------------
// Erdős–Rényi and friends
// ---------------------------------------------------------------------------

/// igraph tutorial, lesson 1: a G(n, m) graph with 1000 vertices and 1000
/// edges has mean degree exactly 2.
#[test]
fn tutorial_lesson_1_mean_degree() {
    rng::seed(42).unwrap();
    let g = Graph::erdos_renyi_game_gnm(1000, 1000, false, EdgeTypeSw::Simple, false).unwrap();
    assert_eq!((g.vcount(), g.ecount()), (1000, 1000));
    assert!(!g.is_directed());
    assert!(is_simple(&g));
    let deg = degrees(&g, NeighborMode::All);
    let mean = deg.iter().sum::<i64>() as f64 / deg.len() as f64;
    assert_eq!(mean, 2.0);
}

#[test]
fn gnm_is_deterministic_under_seed() {
    rng::seed(42).unwrap();
    let a = Graph::erdos_renyi_game_gnm(100, 300, true, EdgeTypeSw::Simple, false).unwrap();
    rng::seed(42).unwrap();
    let b = Graph::erdos_renyi_game_gnm(100, 300, true, EdgeTypeSw::Simple, false).unwrap();
    assert_eq!(a.edge_list(), b.edge_list());
    rng::seed(43).unwrap();
    let c = Graph::erdos_renyi_game_gnm(100, 300, true, EdgeTypeSw::Simple, false).unwrap();
    assert_ne!(a.edge_list(), c.edge_list());
}

#[test]
fn gnm_edge_types() {
    rng::seed(42).unwrap();
    // Complete directed graph: every ordered pair.
    let g = Graph::erdos_renyi_game_gnm(6, 30, true, EdgeTypeSw::Simple, false).unwrap();
    assert_eq!(canon(&g).len(), 30);
    assert!(is_simple(&g));
    // With loops, 36 ordered pairs are available.
    let g = Graph::erdos_renyi_game_gnm(6, 36, true, EdgeTypeSw::Loops, false).unwrap();
    assert_eq!(g.edge_list().iter().filter(|(a, b)| a == b).count(), 6);
    assert!(!has_multi(&g));
    // Too many edges.
    let err = Graph::erdos_renyi_game_gnm(6, 31, true, EdgeTypeSw::Simple, false).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    // Multigraphs can hold any number of edges; no loops unless requested.
    let g = Graph::erdos_renyi_game_gnm(4, 100, false, EdgeTypeSw::Multi, false).unwrap();
    assert_eq!(g.ecount(), 100);
    assert!(!has_loops(&g));
    let g = Graph::erdos_renyi_game_gnm(4, 100, false, AllowedEdgeTypes::ALL, true).unwrap();
    assert_eq!(g.ecount(), 100);
}

#[test]
fn allowed_edge_types_flags() {
    assert_eq!(AllowedEdgeTypes::SIMPLE.to_raw(), 0);
    assert_eq!(
        AllowedEdgeTypes::from(EdgeTypeSw::Loops).to_raw(),
        AllowedEdgeTypes::LOOPS.to_raw()
    );
    assert_eq!(
        igraph::igraph_edge_type_sw_t::from(AllowedEdgeTypes::MULTI),
        AllowedEdgeTypes::MULTI.to_raw()
    );
    assert_eq!(
        AllowedEdgeTypes::LOOPS | EdgeTypeSw::Multi,
        AllowedEdgeTypes::ALL
    );
    assert_eq!(AllowedEdgeTypes::default(), AllowedEdgeTypes::SIMPLE);
    // One single type across the crate: the flags accepted by the games are
    // those of the graphicality tests, and combining two `EdgeTypeSw` values
    // yields a value both accept.
    let both: igraph::mixing::AllowedEdgeTypes = EdgeTypeSw::Loops | EdgeTypeSw::Multi;
    assert!(igraph::mixing::is_graphical(&[1, 2, 5], None, both).unwrap());
    rng::seed(1).unwrap();
    let g = Graph::erdos_renyi_game_gnm(2, 10, false, both, false).unwrap();
    assert_eq!(g.ecount(), 10); // 3 vertex pairs (with loops) only
    assert!(has_multi(&g));
}

#[test]
fn gnp_extremes_and_density() {
    let g = Graph::erdos_renyi_game_gnp(7, 1.0, true, EdgeTypeSw::Simple, false).unwrap();
    assert_eq!(g.ecount(), 42);
    let g = Graph::erdos_renyi_game_gnp(7, 1.0, true, EdgeTypeSw::Loops, false).unwrap();
    assert_eq!(g.ecount(), 49);
    let g = Graph::erdos_renyi_game_gnp(7, 1.0, false, EdgeTypeSw::Loops, false).unwrap();
    assert_eq!(g.ecount(), 28);

    rng::seed(42).unwrap();
    let n = 2000;
    let g =
        Graph::erdos_renyi_game_gnp(n, 5.0 / n as f64, false, EdgeTypeSw::Simple, false).unwrap();
    let mean = 2.0 * g.ecount() as f64 / n as f64;
    assert_close(mean, 5.0 * (n - 1) as f64 / n as f64, 0.3);
    assert!(is_simple(&g));

    let err = Graph::erdos_renyi_game_gnp(10, 1.5, false, EdgeTypeSw::Simple, false).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

#[test]
fn iea_game_counts() {
    rng::seed(42).unwrap();
    let g = Graph::iea_game(10, 200, true, false).unwrap();
    assert_eq!((g.vcount(), g.ecount()), (10, 200));
    assert!(!has_loops(&g));
    assert!(has_multi(&g)); // pigeonhole: only 90 ordered pairs
    let g = Graph::iea_game(10, 200, false, true).unwrap();
    assert_eq!(g.ecount(), 200);
}

// ---------------------------------------------------------------------------
// Growing models
// ---------------------------------------------------------------------------

#[test]
fn barabasi_variants() {
    rng::seed(42).unwrap();
    // Directed Price model: every vertex but the first cites m others.
    let opts = BarabasiOptions::default().with_m(2).with_directed(true);
    let g = Graph::barabasi_game(200, &opts).unwrap();
    assert!(g.is_directed());
    let out = degrees(&g, NeighborMode::Out);
    assert_eq!(out[0], 0);
    assert_eq!(out[1], 1);
    assert!(out[2..].iter().all(|&d| d == 2));
    // Citations point to older vertices.
    assert!(g.edge_list().iter().all(|&(from, to)| from > to));
    assert!(is_simple(&g));

    // Explicit out-degree sequence.
    let outseq = [0, 1, 1, 2, 3, 1];
    let g = Graph::barabasi_game(6, &BarabasiOptions::default().with_outseq(&outseq)).unwrap();
    assert_eq!(g.ecount(), 1 + 1 + 2 + 3 + 1);

    // The bag algorithm allows multi-edges.
    let opts = BarabasiOptions::default()
        .with_m(5)
        .with_algorithm(BarabasiAlgorithm::Bag);
    let g = Graph::barabasi_game(30, &opts).unwrap();
    assert_eq!(g.vcount(), 30);

    // Growing from a seed graph: a triangle.
    let start = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, false).unwrap();
    let opts = BarabasiOptions::default().with_m(2).with_start_from(&start);
    let g = Graph::barabasi_game(10, &opts).unwrap();
    assert_eq!(g.vcount(), 10);
    assert_eq!(g.ecount(), 3 + 7 * 2);
    assert_eq!(&g.edge_list()[..3], &start.edge_list()[..]);

    // Non-linear attachment with power 0.5 works with psumtree.
    let opts = BarabasiOptions::default()
        .with_power(0.5)
        .with_a(0.1)
        .with_outpref(true);
    assert_eq!(Graph::barabasi_game(50, &opts).unwrap().ecount(), 49);

    // The bag algorithm only supports power = 1.
    let opts = BarabasiOptions::default()
        .with_power(2.0)
        .with_algorithm(BarabasiAlgorithm::Bag);
    assert_eq!(
        Graph::barabasi_game(10, &opts).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn barabasi_aging_and_recent_degree() {
    rng::seed(42).unwrap();
    let opts = BarabasiAgingOptions {
        m: 3,
        aging_exp: -2.0,
        aging_bins: 20,
        ..Default::default()
    };
    let g = Graph::barabasi_aging_game(100, &opts).unwrap();
    assert!(g.is_directed());
    assert_eq!(g.ecount(), 99 * 3);

    // igraph's unit test: a strong recent-degree preference makes a star of
    // double edges (all pointing to vertex 0).
    let opts = RecentDegreeOptions {
        power: 30.0,
        window: 100,
        m: 2,
        zero_appeal: 0.001,
        directed: true,
        ..Default::default()
    };
    let g = Graph::recent_degree_game(10, &opts).unwrap();
    let expected: Vec<(i64, i64)> = (1..10).flat_map(|i| [(i, 0), (i, 0)]).collect();
    assert_eq!(canon(&g), expected);

    let err = Graph::recent_degree_game(
        3,
        &RecentDegreeOptions {
            zero_appeal: -1.0,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);

    let outseq = [0, 1, 2, 2, 1];
    let opts = RecentDegreeAgingOptions {
        outseq: Some(&outseq),
        aging_exp: -1.0,
        ..Default::default()
    };
    let g = Graph::recent_degree_aging_game(5, &opts).unwrap();
    assert_eq!(g.ecount(), 6);
}

#[test]
fn growing_random() {
    rng::seed(42).unwrap();
    let g = Graph::growing_random_game(50, 3, false, false).unwrap();
    assert_eq!((g.vcount(), g.ecount()), (50, 49 * 3));
    let g = Graph::growing_random_game(50, 1, true, true).unwrap();
    // One citation per new vertex to an older one: a random recursive tree.
    assert!(is_connected(&g));
    assert!(degrees(&g, NeighborMode::Out)[1..].iter().all(|&d| d == 1));
}

// ---------------------------------------------------------------------------
// Degree-constrained models
// ---------------------------------------------------------------------------

#[test]
fn degree_sequence_realized_exactly_by_every_method() {
    rng::seed(42).unwrap();
    let deg = [4, 3, 3, 2, 2, 2, 1, 1];
    for method in [
        DegreeSequenceMethod::Configuration,
        DegreeSequenceMethod::ConfigurationSimple,
        DegreeSequenceMethod::FastHeurSimple,
        DegreeSequenceMethod::EdgeSwitchingSimple,
        DegreeSequenceMethod::VigerLatapy,
    ] {
        let g = Graph::degree_sequence_game(&deg, None, method).unwrap();
        assert_eq!(degrees(&g, NeighborMode::All), deg, "{method:?}");
        if method != DegreeSequenceMethod::Configuration {
            assert!(is_simple(&g), "{method:?}");
        }
        if method == DegreeSequenceMethod::VigerLatapy {
            assert!(is_connected(&g));
        }
    }
}

#[test]
fn directed_degree_sequence() {
    rng::seed(42).unwrap();
    let out = [2, 1, 1, 0, 1];
    let inn = [0, 1, 2, 2, 0];
    for method in [
        DegreeSequenceMethod::Configuration,
        DegreeSequenceMethod::FastHeurSimple,
    ] {
        let g = Graph::degree_sequence_game(&out, Some(&inn), method).unwrap();
        assert!(g.is_directed());
        assert_eq!(degrees(&g, NeighborMode::Out), out);
        assert_eq!(degrees(&g, NeighborMode::In), inn);
    }
    // Odd degree sum and mismatched sums are rejected.
    let err = Graph::degree_sequence_game(&[1, 1, 1], None, DegreeSequenceMethod::Configuration)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let err =
        Graph::degree_sequence_game(&[1, 1], Some(&[1, 0]), DegreeSequenceMethod::Configuration)
            .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

#[test]
fn k_regular() {
    rng::seed(42).unwrap();
    let g = Graph::k_regular_game(10, 4, false, false).unwrap();
    assert_eq!(degrees(&g, NeighborMode::All), vec![4; 10]);
    assert!(is_simple(&g));
    let g = Graph::k_regular_game(12, 3, true, false).unwrap();
    assert_eq!(degrees(&g, NeighborMode::Out), vec![3; 12]);
    assert_eq!(degrees(&g, NeighborMode::In), vec![3; 12]);
    let g = Graph::k_regular_game(10, 3, false, true).unwrap();
    assert_eq!(degrees(&g, NeighborMode::All), vec![3; 10]);
    assert!(Graph::k_regular_game(9, 3, false, false).is_err());
}

#[test]
fn static_fitness_and_power_law() {
    rng::seed(42).unwrap();
    let fit_out = [1.0, 1.0, 1.0, 0.0, 1.0];
    let fit_in = [0.0, 1.0, 1.0, 1.0, 1.0];
    let g = Graph::static_fitness_game(8, &fit_out, Some(&fit_in), EdgeTypeSw::Simple).unwrap();
    assert!(g.is_directed());
    assert_eq!(g.ecount(), 8);
    assert_eq!(degrees(&g, NeighborMode::Out)[3], 0);
    assert_eq!(degrees(&g, NeighborMode::In)[0], 0);
    assert!(is_simple(&g));

    // Negative fitness is rejected.
    assert!(Graph::static_fitness_game(1, &[1.0, -1.0], None, EdgeTypeSw::Simple).is_err());

    // Power law: hubs appear; directed variant too.
    let g = Graph::static_power_law_game(2000, 4000, 2.2, None, EdgeTypeSw::Simple, false).unwrap();
    assert_eq!(g.ecount(), 4000);
    let max = maxdeg(&g);
    assert!(max > 40, "max degree {max}");
    let g =
        Graph::static_power_law_game(500, 1000, 2.5, Some(3.0), EdgeTypeSw::Simple, true).unwrap();
    assert!(g.is_directed());
    assert_eq!(g.ecount(), 1000);
    // igraph requires exponents of at least 2.
    assert!(
        Graph::static_power_law_game(10, 5, 3.0, Some(1.5), EdgeTypeSw::Simple, false).is_err()
    );
    assert!(
        Graph::static_power_law_game(10, 5, 3.0, Some(-2.0), EdgeTypeSw::Simple, false).is_err()
    );
}

#[test]
fn chung_lu_expected_degrees() {
    rng::seed(42).unwrap();
    // Directed with loops: the expected out/in degrees are exactly the weights.
    let n = 400;
    let w_out: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 3.0 }).collect();
    let w_in: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 3.0 } else { 1.0 }).collect();
    let trials = 20;
    let mut out_even = 0.0;
    for _ in 0..trials {
        let g = Graph::chung_lu_game(&w_out, Some(&w_in), true, ChungLuVariant::Original).unwrap();
        let out = degrees(&g, NeighborMode::Out);
        out_even += out.iter().step_by(2).sum::<i64>() as f64 / (n / 2) as f64;
    }
    assert_close(out_even / trials as f64, 1.0, 0.1);

    for v in [ChungLuVariant::MaxEnt, ChungLuVariant::Nr] {
        let g = Graph::chung_lu_game(&w_out, None, false, v).unwrap();
        assert!(is_simple(&g));
    }
    // In/out weights must have the same sum.
    let err = Graph::chung_lu_game(
        &[1.0, 1.0],
        Some(&[1.0, 2.0]),
        true,
        ChungLuVariant::Original,
    )
    .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

// ---------------------------------------------------------------------------
// Small worlds and rewiring
// ---------------------------------------------------------------------------

#[test]
fn watts_strogatz_lattice_and_rewiring() {
    rng::seed(42).unwrap();
    let lattice = Graph::watts_strogatz_game(1, 20, 3, 0.0, EdgeTypeSw::Simple).unwrap();
    assert_eq!(degrees(&lattice, NeighborMode::All), vec![6; 20]);
    let torus = Graph::watts_strogatz_game(2, 5, 1, 0.0, EdgeTypeSw::Simple).unwrap();
    assert_eq!(torus.vcount(), 25);
    assert_eq!(degrees(&torus, NeighborMode::All), vec![4; 25]);
    let random = Graph::watts_strogatz_game(1, 50, 2, 1.0, EdgeTypeSw::Simple).unwrap();
    assert_eq!(random.ecount(), 100);
    assert!(is_simple(&random));
}

#[test]
fn rewiring_preserves_what_it_should() {
    rng::seed(42).unwrap();
    let mut g = common::cycle(30);
    g.rewire_edges(0.5, EdgeTypeSw::Simple).unwrap();
    assert_eq!((g.vcount(), g.ecount()), (30, 30));
    assert!(is_simple(&g));

    let mut d = Graph::erdos_renyi_game_gnm(40, 120, true, EdgeTypeSw::Simple, false).unwrap();
    let indeg = degrees(&d, NeighborMode::In);
    d.rewire_directed_edges(1.0, false, NeighborMode::In)
        .unwrap();
    assert_eq!(degrees(&d, NeighborMode::In), indeg);
    assert!(!has_loops(&d));
    let outdeg = degrees(&d, NeighborMode::Out);
    d.rewire_directed_edges(1.0, true, NeighborMode::Out)
        .unwrap();
    assert_eq!(degrees(&d, NeighborMode::Out), outdeg);

    let err = common::cycle(5)
        .rewire_edges(2.0, EdgeTypeSw::Simple)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

/// Use case: the small-world phenomenon (Watts & Strogatz, 1998). Rewiring a
/// few percent of the edges of a ring lattice collapses the typical distance
/// between vertices while the local clustering barely changes.
#[test]
fn story_small_world_emerges() {
    rng::seed(42).unwrap();
    let ring = Graph::watts_strogatz_game(1, 500, 5, 0.0, EdgeTypeSw::Simple).unwrap();
    let small_world = Graph::watts_strogatz_game(1, 500, 5, 0.05, EdgeTypeSw::Simple).unwrap();
    let random = Graph::watts_strogatz_game(1, 500, 5, 1.0, EdgeTypeSw::Simple).unwrap();

    // Mean distance over connected pairs, and average local clustering.
    let l = |g: &Graph| g.average_path_length(None, false, true).unwrap();
    let c = |g: &Graph| {
        g.transitivity_avglocal_undirected(TransitivityMode::Zero)
            .unwrap()
    };
    let (l0, c0) = (l(&ring), c(&ring));
    let (l1, c1) = (l(&small_world), c(&small_world));
    let (l2, c2) = (l(&random), c(&random));

    // Ring lattice with k = 10: C = 3(k-2)/(4(k-1)) = 2/3. Distances: the
    // 499 other vertices are at distance ceil(d / 5) for d = 1..=250 on each
    // side, which averages (2 * 5 * (1 + ... + 50) - 50) / 499 = 12700 / 499.
    assert_close(c0, 2.0 / 3.0, 1e-12);
    assert_close(l0, 12700.0 / 499.0, 1e-12);
    // Small world: path length close to the random graph, clustering close to the lattice.
    assert!(l1 < l0 / 4.0, "L = {l1}");
    assert!(l1 < 2.0 * l2);
    // Rewiring both endpoints keeps an edge with probability (1-p)^2, so
    // C ≈ (2/3) (1-p)^6 ≈ 0.49.
    assert!(c1 > 0.4, "C = {c1}");
    assert!(c1 > 5.0 * c2);
    assert!(c2 < 0.1, "C = {c2}");
}

// ---------------------------------------------------------------------------
// Block models and type-based games
// ---------------------------------------------------------------------------

#[test]
fn sbm_matches_igraph_unit_test() {
    // From igraph's tests/unit/igraph_sbm_game.c: probabilities 0/1 make the
    // result deterministic.
    let mut pref = Matrix::zeros(3, 3);
    pref[(0, 1)] = 1.0;
    pref[(2, 2)] = 1.0;
    let g = Graph::sbm_game(&pref, &[2, 2, 2], true, EdgeTypeSw::Loops).unwrap();
    assert_eq!(
        canon(&g),
        vec![
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (4, 4),
            (4, 5),
            (5, 4),
            (5, 5)
        ]
    );
    let g = Graph::sbm_game(&pref, &[2, 2, 2], true, EdgeTypeSw::Simple).unwrap();
    assert_eq!(
        canon(&g),
        vec![(0, 2), (0, 3), (1, 2), (1, 3), (4, 5), (5, 4)]
    );
    // Undirected graphs need a symmetric matrix.
    let err = Graph::sbm_game(&pref, &[2, 2, 2], false, EdgeTypeSw::Simple).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    // Block sizes must match the matrix.
    assert!(Graph::sbm_game(&pref, &[2], true, EdgeTypeSw::Simple).is_err());
    // Multigraph smoke test: expected multiplicity 100 between two vertices.
    rng::seed(42).unwrap();
    let mut pref = Matrix::zeros(2, 2);
    for i in 0..2 {
        for j in 0..2 {
            pref[(i, j)] = 100.0;
        }
    }
    let mean = (0..100)
        .map(|_| {
            Graph::sbm_game(&pref, &[1, 1], false, EdgeTypeSw::Multi)
                .unwrap()
                .ecount()
        })
        .sum::<usize>() as f64
        / 100.0;
    assert!(80.0 < mean && mean < 120.0);
}

#[test]
fn hsbm_matches_igraph_unit_test() {
    let c = Matrix::from_rows(&[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).unwrap();
    let g = Graph::hsbm_game(10, 10, &[0.6, 0.4, 0.0], &c, 0.0).unwrap();
    let expected: Vec<(i64, i64)> = (0..6).flat_map(|a| (6..10).map(move |b| (a, b))).collect();
    assert_eq!(canon(&g), expected);

    // Two blocks of 5 (clusters 3 + 2) with p = 1 between blocks: every
    // vertex is linked to all others except its own cluster mates.
    let g = Graph::hsbm_game(10, 5, &[0.6, 0.4, 0.0], &c, 1.0).unwrap();
    assert_eq!(g.ecount(), 45 - 2 * (3 + 1));

    // The list version with blocks of different shapes.
    let one = Matrix::from_rows(&[[1.0]]).unwrap();
    let g = Graph::hsbm_list_game(
        7,
        &[3, 4],
        &[vec![1.0], vec![0.5, 0.5]],
        &[one, Matrix::from_rows(&[[0.0, 1.0], [1.0, 0.0]]).unwrap()],
        0.0,
    )
    .unwrap();
    // A triangle plus K(2, 2).
    assert_eq!(
        canon(&g),
        vec![(0, 1), (0, 2), (1, 2), (3, 5), (3, 6), (4, 5), (4, 6)]
    );
    assert!(Graph::hsbm_list_game(7, &[3, 3], &[vec![1.0]], &[], 0.0).is_err());
}

#[test]
fn preference_games_respect_types() {
    rng::seed(42).unwrap();
    // Assortative: only same-type edges.
    let pref = Matrix::from_rows(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).unwrap();
    let r = Graph::preference_game(30, None, true, &pref, false, false).unwrap();
    assert_eq!(r.types.len(), 30);
    // Equal groups of 10: three disjoint K10.
    for t in 0..3 {
        assert_eq!(r.types.iter().filter(|&&x| x == t).count(), 10);
    }
    assert_eq!(r.graph.ecount(), 3 * 45);
    for (a, b) in r.graph.edge_list() {
        assert_eq!(r.types[a as usize], r.types[b as usize]);
    }

    // Random type distribution.
    let r = Graph::preference_game(50, Some(&[1.0, 3.0, 0.0]), false, &pref, true, true).unwrap();
    assert!(r.types.iter().all(|&t| t == 0 || t == 1));

    // Asymmetric: out-type 0 only links to in-type 1.
    let pref = Matrix::from_rows(&[[0.0, 1.0], [0.0, 0.0]]).unwrap();
    let r = Graph::asymmetric_preference_game(40, None, &pref, false).unwrap();
    assert!(r.graph.is_directed());
    assert_eq!((r.out_types.len(), r.in_types.len()), (40, 40));
    for (a, b) in r.graph.edge_list() {
        assert_eq!((r.out_types[a as usize], r.in_types[b as usize]), (0, 1));
    }
    let expected = (0..40)
        .flat_map(|a| (0..40).map(move |b| (a, b)))
        .filter(|&(a, b)| a != b && r.out_types[a] == 0 && r.in_types[b] == 1)
        .count();
    assert_eq!(r.graph.ecount(), expected);
    // Joint type distribution: only (out 1, in 0) possible.
    let joint = Matrix::from_rows(&[[0.0, 0.0], [1.0, 0.0]]).unwrap();
    let r = Graph::asymmetric_preference_game(10, Some(&joint), &pref, false).unwrap();
    assert!(r.out_types.iter().all(|&t| t == 1));
    assert!(r.in_types.iter().all(|&t| t == 0));
    assert_eq!(r.graph.ecount(), 0);
}

#[test]
fn callaway_and_establishment() {
    rng::seed(42).unwrap();
    let pref = Matrix::from_rows(&[[1.0, 0.0], [0.0, 1.0]]).unwrap();
    let r = Graph::callaway_traits_game(100, 2, Some(&[0.5, 0.5]), &pref, false).unwrap();
    assert_eq!(r.types.len(), 100);
    assert!(r.graph.ecount() > 0);
    for (a, b) in r.graph.edge_list() {
        assert_eq!(r.types[a as usize], r.types[b as usize]);
    }

    let r = Graph::establishment_game(100, 3, None, &pref, true).unwrap();
    assert!(r.graph.is_directed());
    for (a, b) in r.graph.edge_list() {
        assert_eq!(r.types[a as usize], r.types[b as usize]);
        assert!(a > b, "new vertices connect to older ones");
    }
}

// ---------------------------------------------------------------------------
// Geometric, citation and miscellaneous games
// ---------------------------------------------------------------------------

/// Reproduces igraph's `examples/simple/igraph_grg_game.c`: with seed 42 the
/// average Euclidean-weighted shortest path length of a 200-vertex geometric
/// graph of radius 0.1 is 0.645025.
#[test]
fn grg_example_average_distance() {
    rng::seed(42).unwrap();
    let r = Graph::grg_game(200, 0.1, false).unwrap();
    assert!(r.x.windows(2).all(|w| w[0] <= w[1]), "vertices sorted by x");
    assert!(r.x.iter().chain(&r.y).all(|&c| (0.0..1.0).contains(&c)));
    let n = r.graph.vcount();
    let points = Matrix::from_column_major(n, 2, &[r.x.clone(), r.y.clone()].concat()).unwrap();
    let lengths = r
        .graph
        .spatial_edge_lengths(&points, Metric::Euclidean)
        .unwrap();
    assert!(lengths.iter().all(|&w| w < 0.1));
    let avg = r
        .graph
        .average_path_length(Some(&lengths), false, true)
        .unwrap();
    assert_eq!(format!("{avg:.6}"), "0.645025");

    // The same points, connected by the spatial module with a cutoff
    // distance, give exactly the same graph.
    let nn =
        Graph::nearest_neighbor_graph(&points, Metric::Euclidean, None, Some(0.1), false).unwrap();
    assert!(nn.is_same_graph(&r.graph).unwrap());

    // On a torus, wrap-around distances count: a radius of 1/2 * sqrt(2)
    // covers the whole torus (maximum torus distance), so the graph is complete.
    let r = Graph::grg_game(50, 0.75, true).unwrap();
    assert_eq!(r.graph.ecount(), 50 * 49 / 2);
    // Without the torus, a radius above sqrt(2) is also complete.
    let r = Graph::grg_game(30, 1.5, false).unwrap();
    assert_eq!(r.graph.ecount(), 30 * 29 / 2);
    // Radius zero: no pair is *strictly* closer than 0.
    assert_eq!(Graph::grg_game(30, 0.0, false).unwrap().graph.ecount(), 0);
}

#[test]
fn citation_games_match_igraph_unit_tests() {
    rng::seed(42).unwrap();
    // lastcit: hugely prefer the most recently cited vertex: a star.
    let g = Graph::lastcit_game(9, 1, 1, &[1e30, 1e-30], true).unwrap();
    assert_eq!(canon(&g), (1..9).map(|i| (i, 0)).collect::<Vec<_>>());
    assert!(Graph::lastcit_game(9, 1, 1, &[1.0, 0.0], false).is_err()); // never-cited must be > 0
    assert!(Graph::lastcit_game(9, 1, 1, &[1.0], false).is_err()); // wrong length

    // cited_type: exponentially growing preferences make a line.
    let pref = [1e-30, 1e-20, 1e-10, 1.0, 1e10, 1e20, 0.0];
    let g = Graph::cited_type_game(&[0, 1, 2, 3, 4, 5, 6], &pref, 1, true).unwrap();
    assert_eq!(
        canon(&g),
        vec![(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5)]
    );
    assert!(Graph::cited_type_game(&[0, -5], &[1.0, 1.0], 5, false).is_err());
    assert!(Graph::cited_type_game(&[0, 1], &[1.0, -1.0], 5, false).is_err());

    // citing_cited_type: bipartite preferences give a bipartite multigraph.
    let types = [0, 1, 1, 0, 1, 1, 1, 0, 0, 0];
    let pref = Matrix::from_rows(&[[0.0, 1.0], [1.0, 0.0]]).unwrap();
    let g = Graph::citing_cited_type_game(&types, &pref, 5, false).unwrap();
    assert_eq!(g.ecount(), 45);
    for (a, b) in g.edge_list() {
        assert_ne!(types[a as usize], types[b as usize]);
    }
    let line = Matrix::from_rows(&[
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    ])
    .unwrap();
    let g = Graph::citing_cited_type_game(&[0, 1, 2, 3, 4], &line, 1, true).unwrap();
    assert_eq!(canon(&g), vec![(1, 0), (2, 1), (3, 2), (4, 3)]);
    assert!(Graph::citing_cited_type_game(&[0, 1, 2, 3, 4], &Matrix::new(), 1, true).is_err());
}

#[test]
fn forest_fire_matches_igraph_unit_test() {
    rng::seed(42).unwrap();
    assert_eq!(
        Graph::forest_fire_game(0, 0.0, 0.0, 1, false)
            .unwrap()
            .vcount(),
        0
    );
    assert_eq!(
        Graph::forest_fire_game(10, 0.0, 0.0, 0, false)
            .unwrap()
            .ecount(),
        0
    );
    let g = Graph::forest_fire_game(5, 0.0, 0.0, 100, true).unwrap();
    let expected: Vec<(i64, i64)> = (1..5).flat_map(|a| (0..a).map(move |b| (a, b))).collect();
    assert_eq!(canon(&g), expected);
    let g = Graph::forest_fire_game(200, 0.37, 0.32 / 0.37, 1, true).unwrap();
    assert!(is_connected(&g));
    assert!(is_simple(&g));
    assert!(Graph::forest_fire_game(5, -0.5, 0.0, 1, true).is_err());
}

#[test]
fn interconnected_islands() {
    rng::seed(42).unwrap();
    let (k, size, n_inter) = (4, 10, 3);
    let g = Graph::simple_interconnected_islands_game(k, size, 0.5, n_inter).unwrap();
    assert_eq!(g.vcount(), k * size);
    assert!(is_simple(&g));
    let island = |v: i64| v as usize / size;
    let mut between = std::collections::BTreeMap::new();
    for (a, b) in g.edge_list() {
        let (ia, ib) = (island(a).min(island(b)), island(a).max(island(b)));
        if ia != ib {
            *between.entry((ia, ib)).or_insert(0) += 1;
        }
    }
    assert_eq!(between.len(), k * (k - 1) / 2);
    assert!(between.values().all(|&c| c == n_inter));
}

#[test]
fn correlated_games() {
    rng::seed(42).unwrap();
    let g = Graph::erdos_renyi_game_gnp(40, 0.2, false, EdgeTypeSw::Simple, false).unwrap();
    // Perfect correlation + a permutation: exactly the relabelled copy that
    // `permute_vertices` (same convention) produces.
    let perm: Vec<i64> = (0..40).map(|i| (i * 7 + 3) % 40).collect();
    let h = g.correlated_game(1.0, 0.2, Some(&perm)).unwrap();
    assert!(
        h.is_same_graph(&g.permute_vertices(&perm).unwrap())
            .unwrap()
    );
    assert!(h.isomorphic(&g).unwrap());

    // Zero correlation: essentially independent graphs of similar density.
    let (a, b) = Graph::correlated_pair_game(100, 0.0, 0.1, false, None).unwrap();
    assert!(is_simple(&a) && is_simple(&b));
    let ea: HashSet<_> = canon(&a).into_iter().collect();
    let common = canon(&b).into_iter().filter(|e| ea.contains(e)).count();
    // Expected overlap ≈ p^2 * pairs = 0.01 * 4950 ≈ 50 vs ≈ 495 edges.
    assert!(common < 150, "overlap {common}");
    // High correlation: large overlap.
    let (a, b) = Graph::correlated_pair_game(100, 0.9, 0.1, true, None).unwrap();
    assert!(a.is_directed() && b.is_directed());
    let ea: HashSet<_> = canon(&a).into_iter().collect();
    let common = canon(&b).into_iter().filter(|e| ea.contains(e)).count();
    assert!(common as f64 > 0.8 * a.ecount() as f64);
    // Invalid correlation.
    assert!(g.correlated_game(1.5, 0.2, None).is_err());
}

#[test]
fn random_trees() {
    rng::seed(42).unwrap();
    for method in [RandomTreeMethod::Prufer, RandomTreeMethod::Lerw] {
        let t = Graph::tree_game(100, false, method).unwrap();
        assert_eq!(t.ecount(), 99);
        assert!(is_connected(&t));
    }
    let t = Graph::tree_game(30, true, RandomTreeMethod::Lerw).unwrap();
    assert!(t.is_directed());
    // Oriented away from the root: every vertex but one has in-degree 1.
    let indeg = degrees(&t, NeighborMode::In);
    assert_eq!(indeg.iter().filter(|&&d| d == 0).count(), 1);
    assert!(Graph::tree_game(10, true, RandomTreeMethod::Prufer).is_err());
    assert_eq!(
        Graph::tree_game(0, false, RandomTreeMethod::Lerw)
            .unwrap()
            .vcount(),
        0
    );

    // Uniformity sanity check: all 3 labelled trees on 3 vertices appear.
    let mut seen = BTreeSet::new();
    for _ in 0..50 {
        seen.insert(canon(
            &Graph::tree_game(3, false, RandomTreeMethod::Prufer).unwrap(),
        ));
    }
    assert_eq!(seen.len(), 3);
}

#[test]
fn dot_product_matches_igraph_unit_test() {
    rng::seed(42).unwrap();
    let vecs = Matrix::from_rows(&[
        [1.0, 0.0, 0.0, -10.0, 100.0],
        [0.0, 0.01, 0.0, -100.0, 100.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0],
    ])
    .unwrap();
    let g = Graph::dot_product_game(&vecs, true).unwrap();
    assert_eq!(
        canon(&g),
        vec![(0, 1), (0, 4), (1, 0), (1, 4), (4, 0), (4, 1)]
    );
    let empty = Graph::dot_product_game(&Matrix::new(), false).unwrap();
    assert_eq!(empty.vcount(), 0);
}

/// Use case: "the rich get richer". In a network grown by preferential
/// attachment the largest hub is far bigger than in a uniformly random graph
/// with exactly the same number of vertices and edges.
#[test]
fn story_preferential_attachment_creates_hubs() {
    rng::seed(42).unwrap();
    let n = 3000;
    let ba = Graph::barabasi_game(n, &BarabasiOptions::default().with_m(2)).unwrap();
    let er = Graph::erdos_renyi_game_gnm(n, ba.ecount(), false, EdgeTypeSw::Simple, false).unwrap();
    let (max_ba, max_er) = (maxdeg(&ba), maxdeg(&er));
    assert!(max_ba > 5 * max_er, "BA hub {max_ba} vs ER hub {max_er}");
    // Both have the same mean degree, of course.
    assert_eq!(ba.ecount(), er.ecount());
    // And the BA graph (m = 2, psumtree) is connected and simple.
    assert!(is_connected(&ba) && is_simple(&ba));
}

/// Parameter validation performed by igraph (and by the Rust layer) for the
/// block and type based models, the power-law game and the correlated game.
#[test]
fn invalid_parameters_are_rejected() {
    rng::seed(42).unwrap();
    let kind = |r: igraph::Result<Graph>| r.unwrap_err().kind();

    // Power law: `Some(_)` always means directed, so a negative or NaN
    // in-exponent is an error instead of silently producing an undirected graph.
    for bad in [-1.0, f64::NAN, 1.9] {
        let r = Graph::static_power_law_game(10, 5, 3.0, Some(bad), EdgeTypeSw::Simple, false);
        assert_eq!(kind(r), ErrorKind::InvalidValue, "exponent_in = {bad}");
    }
    let r = Graph::static_power_law_game(10, 5, f64::NAN, None, EdgeTypeSw::Simple, false);
    assert_eq!(kind(r), ErrorKind::InvalidValue);
    // Infinite exponents are allowed and give Erdős–Rényi-like graphs.
    let g = Graph::static_power_law_game(
        50,
        60,
        f64::INFINITY,
        Some(f64::INFINITY),
        EdgeTypeSw::Simple,
        false,
    )
    .unwrap();
    assert!(g.is_directed());
    assert_eq!(g.ecount(), 60);

    // HSBM: block size must divide n, rho must sum to one, C symmetric.
    let c = Matrix::from_rows(&[[1.0, 0.0], [0.0, 1.0]]).unwrap();
    assert_eq!(
        kind(Graph::hsbm_game(10, 3, &[0.5, 0.5], &c, 0.0)),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        kind(Graph::hsbm_game(10, 10, &[0.5, 0.4], &c, 0.0)),
        ErrorKind::InvalidValue
    );
    let asym = Matrix::from_rows(&[[1.0, 0.5], [0.0, 1.0]]).unwrap();
    assert_eq!(
        kind(Graph::hsbm_game(10, 10, &[0.5, 0.5], &asym, 0.0)),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        kind(Graph::hsbm_game(10, 10, &[0.5, 0.5], &c, 1.5)),
        ErrorKind::InvalidValue
    );

    // Preference game: symmetric matrix of probabilities for undirected graphs.
    let r = Graph::preference_game(10, None, false, &asym, false, false);
    assert_eq!(r.unwrap_err().kind(), ErrorKind::InvalidValue);
    let big = Matrix::from_rows(&[[2.0]]).unwrap();
    let r = Graph::preference_game(10, None, false, &big, true, false);
    assert_eq!(r.unwrap_err().kind(), ErrorKind::InvalidValue);
    // Fixed group sizes must add up to the number of vertices.
    let r = Graph::preference_game(10, Some(&[3.0, 3.0]), true, &c, false, false);
    assert_eq!(r.unwrap_err().kind(), ErrorKind::InvalidValue);
    // Directed graphs may use an asymmetric matrix.
    let r = Graph::preference_game(10, Some(&[5.0, 5.0]), true, &asym, true, false).unwrap();
    assert!(r.graph.is_directed());

    // Correlated game: p must be in the open interval (0, 1), and the
    // permutation must cover every vertex.
    let g = common::cycle(6);
    assert_eq!(
        kind(g.correlated_game(0.5, 0.0, None)),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        kind(g.correlated_game(0.5, 1.0, None)),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        kind(g.correlated_game(0.5, 0.3, Some(&[0, 1, 2]))),
        ErrorKind::InvalidValue
    );
    let r = Graph::correlated_pair_game(10, 0.5, 0.0, false, None);
    assert_eq!(r.unwrap_err().kind(), ErrorKind::InvalidValue);
    // The input of the correlated game must be simple.
    let multi = Graph::from_edges(&[(0, 1), (0, 1), (1, 2)], 3, false).unwrap();
    assert_eq!(
        kind(multi.correlated_game(0.5, 0.3, None)),
        ErrorKind::InvalidValue
    );
}

/// The type-based growing models: in the establishment game the first `k`
/// vertices cannot connect (they have fewer than `k` older vertices), and
/// every edge goes from a newer vertex to an older one.
#[test]
fn establishment_growth_structure() {
    rng::seed(42).unwrap();
    let all = Matrix::from_rows(&[[1.0]]).unwrap();
    let k = 3;
    let r = Graph::establishment_game(20, k, None, &all, true).unwrap();
    assert!(r.types.iter().all(|&t| t == 0));
    // With probability one, vertices k..n each cite exactly k older ones.
    let out = degrees(&r.graph, NeighborMode::Out);
    assert!(out[..k].iter().all(|&d| d == 0));
    assert!(out[k..].iter().all(|&d| d == k as i64));
    assert!(is_simple(&r.graph));

    // Callaway: with a connection probability of one, exactly
    // `edges_per_step` edges are added in every step after the first vertex
    // (the endpoints are drawn with replacement, so loops may appear).
    let r = Graph::callaway_traits_game(30, 2, None, &all, true).unwrap();
    assert!(r.graph.is_directed());
    assert_eq!(r.graph.ecount(), 29 * 2);
}

// ---------------------------------------------------------------------------
// Reproducibility across threads
// ---------------------------------------------------------------------------

/// Every thread has its own default generator: the same seed gives the same
/// graphs in concurrently running threads, whatever the other threads do.
#[test]
fn seeded_games_are_reproducible_in_parallel_threads() {
    let sample = |seed: u64| {
        rng::seed(seed).unwrap();
        let a = Graph::erdos_renyi_game_gnm(200, 600, false, EdgeTypeSw::Simple, false).unwrap();
        let b = Graph::barabasi_game(200, &BarabasiOptions::default().with_m(2)).unwrap();
        let t = Graph::tree_game(100, false, RandomTreeMethod::Prufer).unwrap();
        (a.edge_list(), b.edge_list(), t.edge_list())
    };
    let reference = sample(42);
    let handles: Vec<_> = (0..8)
        .map(|i| {
            std::thread::spawn(move || {
                // Odd threads first consume numbers from a different seed.
                if i % 2 == 1 {
                    let _ = sample(1000 + i);
                }
                sample(42)
            })
        })
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), reference);
    }
    assert_ne!(sample(43), reference);
}

// ---------------------------------------------------------------------------
// Deterministic outputs of igraph's own unit tests (tests/unit/*.out)
// ---------------------------------------------------------------------------

/// `tests/unit/igraph_barabasi_aging_game.c`: extreme parameters make the
/// outcome deterministic.
#[test]
fn barabasi_aging_matches_igraph_unit_test() {
    rng::seed(42).unwrap();
    let base = BarabasiAgingOptions {
        aging_exp: 1.0,
        aging_bins: 1,
        zero_deg_appeal: 0.0,
        directed: false,
        ..Default::default()
    };
    let g = Graph::barabasi_aging_game(0, &base).unwrap();
    assert_eq!((g.vcount(), g.is_directed()), (0, false));
    let g = Graph::barabasi_aging_game(
        5,
        &BarabasiAgingOptions {
            m: 0,
            ..base.clone()
        },
    )
    .unwrap();
    assert_eq!((g.vcount(), g.ecount()), (5, 0));

    // One edge per step makes an in-tree.
    let opts = BarabasiAgingOptions {
        m: 1,
        pa_exp: 0.5,
        aging_exp: -0.5,
        aging_bins: 2,
        zero_deg_appeal: 0.1,
        deg_coef: 0.1,
        age_coef: 0.1,
        directed: true,
        ..Default::default()
    };
    let g = Graph::barabasi_aging_game(10, &opts).unwrap();
    assert_eq!(g.ecount(), 9);
    assert!(g.is_tree(NeighborMode::In).unwrap());

    // Prefer old vertices: a star of triple edges.
    let opts = BarabasiAgingOptions {
        m: 3,
        outpref: true,
        pa_exp: 0.0,
        aging_exp: 10.0,
        aging_bins: 6,
        zero_deg_appeal: 1.0,
        deg_coef: 0.0,
        ..Default::default()
    };
    let g = Graph::barabasi_aging_game(5, &opts).unwrap();
    let star: Vec<_> = (1..5).flat_map(|i| [(i, 0); 3]).collect();
    assert_eq!(canon(&g), star);

    // Prefer new vertices: a line of double edges.
    let young = BarabasiAgingOptions {
        m: 2,
        pa_exp: 0.0,
        aging_exp: -10.0,
        aging_bins: 6,
        zero_deg_appeal: 0.1,
        deg_coef: 0.1,
        ..Default::default()
    };
    let g = Graph::barabasi_aging_game(5, &young).unwrap();
    let line: Vec<_> = (1..5).flat_map(|i| [(i, i - 1); 2]).collect();
    assert_eq!(canon(&g), line);

    // Increasing thickness of the line using outseq (its first entry is
    // ignored: vertex 0 has nobody to cite).
    let outseq = [1, 2, 3, 4, 5];
    let opts = BarabasiAgingOptions {
        outseq: Some(&outseq),
        pa_exp: 0.1,
        age_coef: 10.0,
        ..young
    };
    let g = Graph::barabasi_aging_game(5, &opts).unwrap();
    let thick: Vec<_> = (1..5i64)
        .flat_map(|i| std::iter::repeat_n((i, i - 1), i as usize + 1))
        .collect();
    assert_eq!(canon(&g), thick);

    // Error paths.
    let bad = |o: BarabasiAgingOptions| Graph::barabasi_aging_game(5, &o).unwrap_err().kind();
    let d = BarabasiAgingOptions::default();
    assert_eq!(
        bad(BarabasiAgingOptions {
            aging_bins: 0,
            ..d.clone()
        }),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        bad(BarabasiAgingOptions {
            deg_coef: -1.0,
            ..d.clone()
        }),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        bad(BarabasiAgingOptions {
            zero_age_appeal: -1.0,
            ..d.clone()
        }),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        bad(BarabasiAgingOptions {
            outseq: Some(&[1, 1]),
            ..d
        }),
        ErrorKind::InvalidValue
    );
}

/// `tests/unit/igraph_recent_degree_aging_game.c`.
#[test]
fn recent_degree_aging_matches_igraph_unit_test() {
    rng::seed(42).unwrap();
    let g = Graph::recent_degree_aging_game(
        5,
        &RecentDegreeAgingOptions {
            m: 0,
            aging_exp: 1.0,
            aging_bins: 6,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!((g.vcount(), g.ecount()), (5, 0));

    // Prefer more edges: a star of double edges.
    let opts = RecentDegreeAgingOptions {
        m: 2,
        pa_exp: 20.0,
        aging_exp: 0.0,
        aging_bins: 1,
        window: 100,
        zero_appeal: 0.001,
        ..Default::default()
    };
    let g = Graph::recent_degree_aging_game(5, &opts).unwrap();
    let star: Vec<_> = (1..5).flat_map(|i| [(0, i); 2]).collect();
    assert_eq!(canon(&g), star);

    // Prefer older vertices: a star, with the multiplicities of outseq.
    let outseq = [1, 2, 1, 2, 1, 2, 1];
    let opts = RecentDegreeAgingOptions {
        outseq: Some(&outseq),
        m: 0,
        pa_exp: 0.0,
        aging_exp: 20.0,
        aging_bins: 8,
        window: 100,
        ..Default::default()
    };
    let g = Graph::recent_degree_aging_game(7, &opts).unwrap();
    assert_eq!(
        canon(&g),
        vec![
            (0, 1),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 5),
            (0, 6)
        ]
    );

    // One edge per step makes a tree.
    let opts = RecentDegreeAgingOptions {
        outpref: true,
        pa_exp: 2.0,
        aging_exp: 2.0,
        aging_bins: 5,
        window: 4,
        ..Default::default()
    };
    let g = Graph::recent_degree_aging_game(10, &opts).unwrap();
    assert!(g.is_tree(NeighborMode::All).unwrap());

    // Error paths.
    for o in [
        RecentDegreeAgingOptions {
            aging_bins: 0,
            ..Default::default()
        },
        RecentDegreeAgingOptions {
            zero_appeal: -1.0,
            ..Default::default()
        },
    ] {
        let err = Graph::recent_degree_aging_game(20, &o).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidValue);
    }
}

/// `tests/unit/igraph_simple_interconnected_islands_game.c`: the whole call
/// sequence is replayed after seeding with 42, so even the random last graph
/// matches the expected output exactly.
#[test]
fn islands_match_igraph_unit_test() {
    rng::seed(42).unwrap();
    let g = Graph::simple_interconnected_islands_game(0, 1, 1.0, 1).unwrap();
    assert_eq!(g.vcount(), 0);
    let g = Graph::simple_interconnected_islands_game(1, 4, 0.0, 2).unwrap();
    assert_eq!((g.vcount(), g.ecount()), (4, 0));
    let g = Graph::simple_interconnected_islands_game(1, 4, 1.0, 2).unwrap();
    assert!(
        g.is_same_graph(&Graph::full(4, false, false).unwrap())
            .unwrap()
    );
    let g = Graph::simple_interconnected_islands_game(2, 4, 1.0, 0).unwrap();
    assert_eq!(
        g.connected_components(Connectedness::Weak).unwrap().sizes,
        [4, 4]
    );
    assert_eq!(g.ecount(), 12);
    // 16 = 4 * 4 connections: every pair of islands is completely joined.
    let g = Graph::simple_interconnected_islands_game(3, 4, 1.0, 16).unwrap();
    assert_eq!(g.ecount(), 18 + 48);
    assert!(
        g.is_same_graph(&Graph::full(12, false, false).unwrap())
            .unwrap()
    );
    let g = Graph::simple_interconnected_islands_game(3, 4, 0.5, 3).unwrap();
    assert_eq!(
        canon(&g),
        vec![
            (0, 4),
            (0, 11),
            (1, 10),
            (2, 4),
            (3, 4),
            (3, 10),
            (4, 5),
            (4, 6),
            (5, 7),
            (5, 9),
            (6, 8),
            (7, 11),
            (9, 10),
            (10, 11)
        ]
    );
    // Errors: probability out of range, more links than vertex pairs.
    for r in [
        Graph::simple_interconnected_islands_game(2, 4, 2.0, 0),
        Graph::simple_interconnected_islands_game(3, 4, 1.0, 20),
    ] {
        assert_eq!(r.unwrap_err().kind(), ErrorKind::InvalidValue);
    }
}

/// `tests/unit/igraph_rewire_directed_edges.c`, replayed call by call after
/// seeding with 42 (the last, random rewiring matches the expected output).
#[test]
fn rewire_directed_edges_matches_igraph_unit_test() {
    rng::seed(42).unwrap();
    let mut g = Graph::new(5, true);
    g.rewire_directed_edges(0.1, false, NeighborMode::All)
        .unwrap();
    assert_eq!((g.vcount(), g.ecount()), (5, 0));

    let edges = [
        (0, 1),
        (0, 3),
        (5, 4),
        (4, 8),
        (9, 2),
        (9, 3),
        (9, 7),
        (7, 7),
        (7, 8),
    ];
    let mut g = Graph::from_edges(&edges, 10, true).unwrap();
    let copy = g.clone();
    g.rewire_directed_edges(0.0, false, NeighborMode::All)
        .unwrap();
    assert!(g.is_same_graph(&copy).unwrap());
    g.rewire_directed_edges(0.5, true, NeighborMode::All)
        .unwrap();
    assert!(!g.is_same_graph(&copy).unwrap()); // guaranteed for this seed
    assert_eq!((g.vcount(), g.ecount()), (10, 9));

    // An out-star stays an out-star when the targets are moved.
    let star: Vec<_> = (1..10).map(|i| (0, i)).collect();
    let mut g = Graph::from_edges(&star, 10, true).unwrap();
    g.rewire_directed_edges(1.0, false, NeighborMode::Out)
        .unwrap();
    assert_eq!(g.degree_of(0, NeighborMode::All, Loops::None).unwrap(), 9);

    // With mode = All both endpoints move, multi-edges may appear.
    let dag: Vec<_> = (0..5)
        .flat_map(|a| (a + 1..5).map(move |b| (a, b)))
        .collect();
    let mut g = Graph::from_edges(&dag, 5, true).unwrap();
    g.rewire_directed_edges(1.0, false, NeighborMode::All)
        .unwrap();
    assert_eq!(
        canon(&g),
        vec![
            (0, 4),
            (0, 4),
            (1, 0),
            (1, 2),
            (1, 2),
            (1, 2),
            (2, 0),
            (2, 1),
            (3, 1),
            (3, 1)
        ]
    );

    for p in [-0.1, 1.1] {
        let err = g
            .rewire_directed_edges(p, false, NeighborMode::All)
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidValue);
    }
}

/// `tests/unit/igraph_correlated_pair_game.c` and friends.
#[test]
fn correlated_pair_matches_igraph_unit_test() {
    rng::seed(42).unwrap();
    let (a, b) = Graph::correlated_pair_game(0, 1.0, 0.99, true, None).unwrap();
    assert_eq!((a.vcount(), b.vcount(), a.is_directed()), (0, 0, true));
    let (a, b) = Graph::correlated_pair_game(10, 1.0, 0.5, true, None).unwrap();
    assert!(a.is_same_graph(&b).unwrap());
}

// ---------------------------------------------------------------------------
// More properties and error paths
// ---------------------------------------------------------------------------

#[test]
fn barabasi_start_graph_rules() {
    rng::seed(42).unwrap();
    // A directed start graph: its edges come first, then m citations each.
    let start = Graph::from_edges(&[(1, 0), (2, 0)], 3, true).unwrap();
    let opts = BarabasiOptions::default()
        .with_m(2)
        .with_directed(true)
        .with_start_from(&start);
    let g = Graph::barabasi_game(8, &opts).unwrap();
    assert_eq!(&g.edge_list()[..2], &[(1, 0), (2, 0)]);
    assert_eq!(degrees(&g, NeighborMode::Out)[3..], [2; 5]);
    // Growing from an undirected start graph into a directed one requires
    // counting the total degree (outpref).
    let und = Graph::ring(3, false, false, true).unwrap();
    let opts = BarabasiOptions::default()
        .with_directed(true)
        .with_start_from(&und);
    let err = Graph::barabasi_game(8, &opts).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    assert!(Graph::barabasi_game(8, &opts.clone().with_outpref(true)).is_ok());
    // Start graphs must be non-empty and not larger than the result.
    let empty = Graph::new(0, false);
    let opts = BarabasiOptions::default().with_start_from(&empty);
    assert!(Graph::barabasi_game(5, &opts).is_err());
    let opts = BarabasiOptions::default().with_start_from(&und);
    assert!(Graph::barabasi_game(2, &opts).is_err());
    // A must be positive without outpref (directed graphs).
    let opts = BarabasiOptions::default().with_directed(true).with_a(0.0);
    assert!(Graph::barabasi_game(5, &opts).is_err());
    // The default undirected tree (m = 1) is a tree.
    let t = Graph::barabasi_game(100, &BarabasiOptions::default()).unwrap();
    assert!(t.is_tree(NeighborMode::All).unwrap());
}

#[test]
fn forest_fire_parameter_ranges() {
    rng::seed(42).unwrap();
    // fw_prob must be < 1, and so must be bw_factor * fw_prob.
    assert!(Graph::forest_fire_game(5, 1.0, 0.0, 1, true).is_err());
    assert!(Graph::forest_fire_game(5, 0.5, 2.0, 1, true).is_err());
    assert!(Graph::forest_fire_game(5, 0.5, 1.9, 1, true).is_ok());
    // With a single ambassador and no burning, every vertex cites one older
    // vertex: a random recursive tree (a DAG with out-degree one).
    let g = Graph::forest_fire_game(50, 0.0, 0.0, 1, true).unwrap();
    assert!(g.is_tree(NeighborMode::In).unwrap());
    assert!(g.is_dag().unwrap());
}

/// Use case: detect a planted partition. A stochastic block model with dense
/// blocks and sparse links between them hides four communities; the Louvain
/// method recovers them exactly.
#[test]
fn story_planted_partition_is_recovered() {
    rng::seed(42).unwrap();
    let k = 4;
    let size = 25;
    let mut pref = Matrix::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            pref[(i, j)] = if i == j { 0.5 } else { 0.01 };
        }
    }
    let g = Graph::sbm_game(&pref, &[size as i64; 4], false, EdgeTypeSw::Simple).unwrap();
    let planted: Vec<i64> = (0..k * size).map(|v| (v / size) as i64).collect();
    let found = g.community_multilevel(None, 1.0).unwrap().membership;
    let nmi =
        igraph::community::compare_communities(&planted, &found, CommunityComparison::Nmi).unwrap();
    assert_close(nmi, 1.0, 1e-12);
    // The planted partition has a high modularity.
    let q = g.modularity(&planted, None, 1.0, false).unwrap();
    assert!(q > 0.6, "Q = {q}");
}

/// Use case: is the local clustering of Zachary's karate club explained by
/// its degrees alone? Random graphs with exactly the same degrees (a
/// configuration null model) are far less clustered locally.
#[test]
fn story_karate_clustering_beats_degree_preserving_null_model() {
    rng::seed(42).unwrap();
    let karate = Graph::famous("Zachary").unwrap();
    let deg = degrees(&karate, NeighborMode::All);
    let clustering = |g: &Graph| {
        g.transitivity_avglocal_undirected(TransitivityMode::Zero)
            .unwrap()
    };
    let observed = clustering(&karate);
    assert_close(observed, 0.5706385, 1e-6);
    let samples = 50;
    let mut mean = 0.0;
    for _ in 0..samples {
        let g = Graph::degree_sequence_game(&deg, None, DegreeSequenceMethod::VigerLatapy).unwrap();
        assert_eq!(degrees(&g, NeighborMode::All), deg);
        assert!(is_simple(&g) && is_connected(&g));
        mean += clustering(&g) / samples as f64;
    }
    assert!(mean < 0.75 * observed, "null model C = {mean}");
    // The same null model via degree-preserving rewiring of the original.
    let mut rewired = karate.clone();
    rewired.rewire(10_000, EdgeTypeSw::Simple).unwrap();
    assert_eq!(degrees(&rewired, NeighborMode::All), deg);
    assert!(clustering(&rewired) < observed);
}

#[test]
fn k_regular_and_tree_invariants() {
    rng::seed(42).unwrap();
    // igraph's unit test: all the combinations of parity and multiplicity.
    for (n, k, directed, multiple) in [
        (10, 4, false, false),
        (10, 3, false, false),
        (10, 4, false, true),
        (10, 3, false, true),
        (10, 4, true, false),
        (10, 3, true, false),
        (9, 3, true, false),
        (9, 3, true, true),
    ] {
        let g = Graph::k_regular_game(n, k, directed, multiple).unwrap();
        let mode = if directed {
            NeighborMode::Out
        } else {
            NeighborMode::All
        };
        assert_eq!(degrees(&g, mode), vec![k as i64; n]);
        if directed {
            assert_eq!(degrees(&g, NeighborMode::In), vec![k as i64; n]);
        }
        if !multiple {
            assert!(is_simple(&g));
        }
    }
    assert!(Graph::k_regular_game(9, 3, false, true).is_err());

    for method in [RandomTreeMethod::Prufer, RandomTreeMethod::Lerw] {
        for n in [1, 2, 3, 30] {
            let t = Graph::tree_game(n, false, method).unwrap();
            assert!(t.is_tree(NeighborMode::All).unwrap(), "{method:?} n = {n}");
        }
    }
}

/// Inputs that igraph 1.0.1 does not validate and that would crash the
/// process (division by zero, out-of-bounds reads, assertion failures), hang
/// it forever, or silently produce garbage: the Rust layer rejects them with
/// an ordinary error.
#[test]
fn pathological_inputs_are_rejected_instead_of_crashing() {
    rng::seed(42).unwrap();
    let invalid = |r: igraph::Result<()>| {
        assert_eq!(r.unwrap_err().kind(), ErrorKind::InvalidValue);
    };
    let nan = f64::NAN;
    let inf = f64::INFINITY;

    // Rewiring a single-vertex graph without self-loops: SIGFPE in C.
    let lonely = || Graph::from_edges(&[(0, 0)], 1, true).unwrap();
    for sw in [EdgeTypeSw::Simple, EdgeTypeSw::Multi] {
        invalid(lonely().rewire_edges(1.0, sw));
    }
    for mode in [NeighborMode::Out, NeighborMode::In, NeighborMode::All] {
        invalid(lonely().rewire_directed_edges(0.5, false, mode));
    }
    // ... but it is fine when loops are allowed, or nothing is rewired.
    let mut g = lonely();
    g.rewire_edges(1.0, EdgeTypeSw::Loops).unwrap();
    g.rewire_directed_edges(1.0, true, NeighborMode::Out)
        .unwrap();
    g.rewire_edges(0.0, EdgeTypeSw::Simple).unwrap();
    assert_eq!(g.edge_list(), [(0, 0)]);
    // NaN probabilities: infinite loops in C.
    invalid(common::cycle(5).rewire_edges(nan, EdgeTypeSw::Simple));
    invalid(common::cycle(5).rewire_directed_edges(nan, true, NeighborMode::Out));
    invalid(Graph::watts_strogatz_game(1, 10, 1, nan, EdgeTypeSw::Simple).map(drop));

    // Negative vertex types: out-of-bounds reads in C.
    let one = Matrix::from_rows(&[[1.0]]).unwrap();
    invalid(Graph::citing_cited_type_game(&[0, -1, -5, 0], &one, 1, true).map(drop));
    // Infinite preferences: infinite loop.
    invalid(Graph::cited_type_game(&[0, 0, 1], &[inf, 1.0], 1, false).map(drop));

    // Static fitness: negative directed fitnesses used to create edges to
    // non-existent vertices; non-finite ones hang.
    invalid(
        Graph::static_fitness_game(
            3,
            &[1.0, 1.0, 1.0],
            Some(&[1.0, -5.0, 1.0]),
            EdgeTypeSw::Multi,
        )
        .map(drop),
    );
    invalid(
        Graph::static_fitness_game(
            3,
            &[-1.0, 1.0, 1.0],
            Some(&[1.0, 1.0, 1.0]),
            EdgeTypeSw::Multi,
        )
        .map(drop),
    );
    for bad in [nan, inf] {
        invalid(
            Graph::static_fitness_game(2, &[bad, 1.0, 1.0], None, EdgeTypeSw::Simple).map(drop),
        );
    }

    // Type distributions: all zero (abort), infinite (hang), fractional
    // fixed sizes (uninitialized vertex types).
    let half = Matrix::from_rows(&[[0.5]]).unwrap();
    let halves = Matrix::from_rows(&[[0.5, 0.5], [0.5, 0.5]]).unwrap();
    invalid(Graph::preference_game(10, Some(&[0.0]), false, &half, false, false).map(drop));
    invalid(Graph::preference_game(10, Some(&[inf, 1.0]), false, &halves, false, false).map(drop));
    invalid(Graph::preference_game(10, Some(&[2.5, 7.5]), true, &halves, false, false).map(drop));
    let zero = Matrix::from_rows(&[[0.0]]).unwrap();
    invalid(Graph::asymmetric_preference_game(10, Some(&zero), &half, false).map(drop));
    let inf_dist = Matrix::from_rows(&[[inf, 1.0], [1.0, 1.0]]).unwrap();
    invalid(Graph::asymmetric_preference_game(10, Some(&inf_dist), &halves, false).map(drop));
    invalid(Graph::callaway_traits_game(10, 1, Some(&[inf, 1.0]), &halves, false).map(drop));
    invalid(Graph::establishment_game(10, 1, Some(&[inf, 1.0]), &halves, false).map(drop));
    // Zero weights for some types are fine.
    let r = Graph::preference_game(10, Some(&[0.0, 1.0]), false, &halves, false, false).unwrap();
    assert_eq!(r.types, vec![1; 10]);

    // NaN probabilities that igraph accepts (crashing, or silently).
    invalid(Graph::simple_interconnected_islands_game(2, 3, nan, 1).map(drop));
    invalid(Graph::forest_fire_game(10, nan, 0.0, 1, true).map(drop));
    invalid(Graph::forest_fire_game(10, 0.0, inf, 1, true).map(drop));
    invalid(Graph::hsbm_game(4, 2, &[1.0], &one, nan).map(drop));
    let nan_matrix = Matrix::from_rows(&[[nan]]).unwrap();
    invalid(Graph::hsbm_game(4, 2, &[1.0], &nan_matrix, 0.5).map(drop));
    invalid(
        Graph::hsbm_list_game(
            4,
            &[2, 2],
            &[vec![1.0], vec![1.0]],
            &[one.clone(), one.clone()],
            nan,
        )
        .map(drop),
    );
    invalid(Graph::sbm_game(&nan_matrix, &[5], false, EdgeTypeSw::Simple).map(drop));
    invalid(Graph::erdos_renyi_game_gnp(0, nan, false, EdgeTypeSw::Simple, false).map(drop));
    let ring = common::cycle(6);
    invalid(ring.correlated_game(nan, 0.5, None).map(drop));
    invalid(ring.correlated_game(0.5, nan, None).map(drop));
}

/// Documented corner cases of the C implementation.
#[test]
fn documented_corner_cases() {
    rng::seed(42).unwrap();
    // The edge-labeled G(n, p) model needs multi-edges.
    let err = Graph::erdos_renyi_game_gnp(10, 0.5, false, EdgeTypeSw::Simple, true).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Unimplemented);
    let g = Graph::erdos_renyi_game_gnp(10, 0.5, false, EdgeTypeSw::Multi, true).unwrap();
    assert!(!has_loops(&g));

    // Directed Prüfer trees are only an error from two vertices on.
    assert_eq!(
        Graph::tree_game(1, true, RandomTreeMethod::Prufer)
            .unwrap()
            .vcount(),
        1
    );
    assert!(Graph::tree_game(2, true, RandomTreeMethod::Prufer).is_err());

    // The correlated game ignores the permutation for corr = 0 (only its
    // length is checked), but validates it otherwise.
    let g = common::cycle(4);
    assert!(g.correlated_game(0.0, 0.5, Some(&[0, 0, 0, 0])).is_ok());
    assert!(g.correlated_game(0.5, 0.5, Some(&[0, 0, 0, 0])).is_err());
    assert!(g.correlated_game(0.5, 0.5, Some(&[0, 0, 0])).is_err());

    // Negative radii behave like zero; points are dropped in [0, 1)².
    assert_eq!(Graph::grg_game(20, -1.0, true).unwrap().graph.ecount(), 0);

    // The fixed group sizes of the preference game are assigned in order,
    // equal groups (the first ones one larger) by default.
    let halves = Matrix::from_rows(&[[0.5, 0.5], [0.5, 0.5]]).unwrap();
    let r = Graph::preference_game(5, None, true, &halves, false, false).unwrap();
    assert_eq!(r.types, [0, 0, 0, 1, 1]);
    let r = Graph::preference_game(5, Some(&[1.0, 4.0]), true, &halves, false, false).unwrap();
    assert_eq!(r.types, [0, 1, 1, 1, 1]);

    // Static power law: the last vertices get the largest fitness (without
    // the finite size correction), so they are the hubs on average.
    let mut late = 0;
    let mut early = 0;
    for _ in 0..20 {
        let g =
            Graph::static_power_law_game(100, 200, 2.1, None, EdgeTypeSw::Simple, false).unwrap();
        let d = degrees(&g, NeighborMode::All);
        early += d[..10].iter().sum::<i64>();
        late += d[90..].iter().sum::<i64>();
    }
    assert!(late > 5 * early, "late {late} vs early {early}");
}

// ---------------------------------------------------------------------------
// Inputs that igraph 1.0.1 does not guard against (overflow, NaN, endless
// loops): the wrappers must reject them before calling C.
// ---------------------------------------------------------------------------

const BIG: usize = i64::MAX as usize;

fn err_kind(r: Result<Graph>) -> ErrorKind {
    r.map(|g| g.vcount()).unwrap_err().kind()
}

#[test]
fn chung_lu_rejects_weights_whose_sum_or_products_overflow() {
    for variant in [
        ChungLuVariant::Original,
        ChungLuVariant::MaxEnt,
        ChungLuVariant::Nr,
    ] {
        // Each weight is finite, but w_i * w_j (resp. the sum) is not.
        assert_eq!(
            err_kind(Graph::chung_lu_game(
                &[1e200, 1e200, 1.0],
                None,
                true,
                variant
            )),
            ErrorKind::InvalidValue
        );
        assert_eq!(
            err_kind(Graph::chung_lu_game(&[1e308; 3], None, true, variant)),
            ErrorKind::InvalidValue
        );
        assert_eq!(
            err_kind(Graph::chung_lu_game(
                &[1.0, 1e300],
                Some(&[1e300, 1.0]),
                false,
                variant
            )),
            ErrorKind::InvalidValue
        );
    }
    // Large but harmless weights still work: two vertices of weight 1e100
    // are connected with probability min(1e100 * 1e100 / 2e100, 1) = 1.
    rng::seed(3).unwrap();
    let g = Graph::chung_lu_game(&[1e100, 1e100], None, false, ChungLuVariant::Original).unwrap();
    assert_eq!(g.edge_list(), [(0, 1)]);
}

#[test]
fn static_fitness_rejects_huge_edge_counts_and_infinite_sums() {
    for m in [BIG, (1 << 62) | (1 << 61), (1 << 62) + 1] {
        assert_eq!(
            err_kind(Graph::static_fitness_game(
                m,
                &[1.0; 3],
                None,
                EdgeTypeSw::Multi
            )),
            ErrorKind::Overflow
        );
        assert_eq!(
            err_kind(Graph::static_power_law_game(
                10,
                m,
                2.5,
                None,
                EdgeTypeSw::Multi,
                true
            )),
            ErrorKind::Overflow
        );
    }
    // Finite fitnesses with an infinite sum made igraph loop forever.
    assert_eq!(
        err_kind(Graph::static_fitness_game(
            3,
            &[1e308; 3],
            None,
            EdgeTypeSw::Simple
        )),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        err_kind(Graph::static_fitness_game(
            3,
            &[1.0; 3],
            Some(&[1e308; 3]),
            EdgeTypeSw::Simple
        )),
        ErrorKind::InvalidValue
    );
    // Scale does not matter: huge (finite-sum) fitnesses still work.
    rng::seed(1).unwrap();
    let g = Graph::static_fitness_game(3, &[1e300; 3], None, EdgeTypeSw::Simple).unwrap();
    assert_eq!(g.ecount(), 3);
}

#[test]
fn gnp_and_lastcit_reject_i64_max_vertices() {
    assert_eq!(
        err_kind(Graph::erdos_renyi_game_gnp(
            BIG,
            1e-300,
            false,
            EdgeTypeSw::Loops,
            false
        )),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        err_kind(Graph::lastcit_game(BIG, 1, 1, &[1.0, 1.0], true)),
        ErrorKind::InvalidValue
    );
}

#[test]
fn recent_degree_games_clamp_the_window() {
    // A window longer than the process is the same model as window = n:
    // with the same seed the graphs coincide, and a huge window no longer
    // overflows igraph's history capacity.
    let n = 40;
    let sample = |window| {
        rng::seed(11).unwrap();
        let opts = RecentDegreeOptions {
            window,
            m: 2,
            power: 1.5,
            ..Default::default()
        };
        Graph::recent_degree_game(n, &opts).unwrap().edge_list()
    };
    assert_eq!(sample(BIG), sample(n));
    assert_eq!(sample(usize::MAX), sample(n));
    let sample_aging = |window| {
        rng::seed(12).unwrap();
        let opts = RecentDegreeAgingOptions {
            window,
            m: 2,
            aging_bins: 4,
            ..Default::default()
        };
        Graph::recent_degree_aging_game(n, &opts)
            .unwrap()
            .edge_list()
    };
    assert_eq!(sample_aging(BIG), sample_aging(n));
    assert_eq!(
        Graph::recent_degree_game(
            10,
            &RecentDegreeOptions {
                window: BIG,
                ..Default::default()
            }
        )
        .unwrap()
        .vcount(),
        10
    );
}

#[test]
fn cited_type_games_reject_overflowing_sizes_and_types() {
    // n * edges_per_step overflows.
    assert_eq!(
        err_kind(Graph::cited_type_game(&[0; 4], &[1.0], 1 << 62, true)),
        ErrorKind::Overflow
    );
    assert_eq!(
        err_kind(Graph::cited_type_game(&[0; 3], &[1.0], BIG, true)),
        ErrorKind::Overflow
    );
    // A type not covered by `pref` (igraph's error path computes max + 1).
    for t in [1, i64::MAX] {
        assert_eq!(
            err_kind(Graph::cited_type_game(&[0, 0, t], &[1.0], 1, true)),
            ErrorKind::InvalidValue
        );
    }
    let pref = Matrix::from_rows(&[[1.0]]).unwrap();
    for t in [1, i64::MAX] {
        assert_eq!(
            err_kind(Graph::citing_cited_type_game(&[0, 0, t], &pref, 1, true)),
            ErrorKind::InvalidValue
        );
    }
    // Valid input still works.
    let g = Graph::citing_cited_type_game(&[0, 0, 0], &pref, 1, true).unwrap();
    assert_eq!(g.ecount(), 2);
}

#[test]
fn block_models_reject_overflowing_sizes() {
    let zeros = Matrix::from_rows(&[[0.0, 0.0], [0.0, 0.0]]).unwrap();
    let half = i64::MAX / 2 + 1;
    assert_eq!(
        err_kind(Graph::sbm_game(
            &zeros,
            &[half, half],
            false,
            EdgeTypeSw::Simple
        )),
        ErrorKind::Overflow
    );
    assert_eq!(
        err_kind(Graph::sbm_game(&zeros, &[-1, 3], false, EdgeTypeSw::Simple)),
        ErrorKind::InvalidValue
    );
    let tiny = Matrix::from_rows(&[[1e-300]]).unwrap();
    assert_eq!(
        err_kind(Graph::hsbm_game(BIG, BIG, &[1.0], &tiny, 0.0)),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        err_kind(Graph::hsbm_game(1 << 53, 1 << 53, &[1.0], &tiny, 0.0)),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        err_kind(Graph::hsbm_list_game(
            BIG,
            &[i64::MAX],
            &[vec![1.0]],
            std::slice::from_ref(&tiny),
            0.0
        )),
        ErrorKind::InvalidValue
    );
    let many = vec![(1i64 << 53) - 1; 2048];
    let rhos = vec![vec![1.0]; many.len()];
    let cs = vec![tiny.clone(); many.len()];
    assert_eq!(
        err_kind(Graph::hsbm_list_game(10, &many, &rhos, &cs, 0.0)),
        ErrorKind::Overflow
    );
}

#[test]
fn islands_game_rejects_overflowing_products() {
    for (n, size, inter) in [
        (2, 3_100_000_000, 1),
        (BIG, 1, 1),
        (1 << 32, 1 << 32, 0),
        (1 << 20, 1, 1 << 30),
    ] {
        assert_eq!(
            err_kind(Graph::simple_interconnected_islands_game(
                n, size, 0.0, inter
            )),
            ErrorKind::Overflow,
            "({n}, {size}, {inter})"
        );
    }
    // A single island never has inter-island edges.
    let g = Graph::simple_interconnected_islands_game(1, 5, 1.0, 0).unwrap();
    assert_eq!(g.ecount(), 10);
}

#[test]
fn multigraph_games_reject_infinite_or_degenerate_multiplicities() {
    // An infinite expected multiplicity used to give the empty graph
    // (edge-labeled: after converting NaN to an integer).
    for labeled in [false, true] {
        assert_eq!(
            err_kind(Graph::erdos_renyi_game_gnp(
                3,
                f64::INFINITY,
                false,
                EdgeTypeSw::Multi,
                labeled
            )),
            ErrorKind::InvalidValue
        );
    }
    // Edge-labeled: the sampled edge count would be doubled unchecked.
    assert_eq!(
        err_kind(Graph::erdos_renyi_game_gnp(
            3,
            1e19,
            false,
            EdgeTypeSw::Multi,
            true
        )),
        ErrorKind::Overflow
    );
    // Moderate multiplicities work: 3 vertex pairs, 4 expected edges each.
    rng::seed(8).unwrap();
    let g = Graph::erdos_renyi_game_gnp(3, 4.0, false, EdgeTypeSw::Multi, true).unwrap();
    assert!(!g.has_loop().unwrap());

    // SBM: x / (1 + x) == 1 made igraph loop forever; inf gave no edges.
    for x in [1e20, f64::INFINITY] {
        let pref = Matrix::from_rows(&[[x]]).unwrap();
        assert_eq!(
            err_kind(Graph::sbm_game(&pref, &[3], false, EdgeTypeSw::Multi)),
            ErrorKind::InvalidValue
        );
    }
    // Without multi-edges such entries are igraph's own range error.
    let pref = Matrix::from_rows(&[[1e20]]).unwrap();
    assert_eq!(
        err_kind(Graph::sbm_game(&pref, &[3], false, EdgeTypeSw::Simple)),
        ErrorKind::InvalidValue
    );
    let pref = Matrix::from_rows(&[[2.0]]).unwrap();
    rng::seed(8).unwrap();
    let g = Graph::sbm_game(&pref, &[3], false, EdgeTypeSw::Multi).unwrap();
    assert_eq!(g.vcount(), 3);
}

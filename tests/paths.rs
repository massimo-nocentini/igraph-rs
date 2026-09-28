//! Integration tests for the `paths` module (igraph_paths.h).

mod common;

use common::*;
use igraph::{
    paths::{FloydWarshallAlgorithm, SimplePathsOptions, VoronoiPartition, expand_path_to_pairs},
    prelude::*,
};

const INF: f64 = f64::INFINITY;

/// The 10-vertex directed graph used by igraph's `distances.c`,
/// `bellman_ford.c` and `igraph_get_shortest_paths_dijkstra.c` examples.
fn example_graph() -> (Graph, Vec<f64>) {
    let edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 4),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 2),
        (3, 6),
        (4, 5),
        (4, 7),
        (5, 6),
        (5, 8),
        (5, 9),
        (7, 5),
        (7, 8),
        (8, 9),
        (5, 2),
        (2, 1),
    ];
    let weights = vec![
        0., 2., 1., 0., 5., 2., 1., 1., 0., 2., 2., 8., 1., 1., 3., 1., 1., 4., 2., 1.,
    ];
    (Graph::from_edges(&edges, 10, true).unwrap(), weights)
}

/// A `side x side` square lattice (a torus when `periodic`), with vertex
/// `(x, y)` numbered `x + side * y`.
fn lattice(side: usize, periodic: bool) -> Graph {
    Graph::square_lattice(&[side, side], 1, false, false, Some(&[periodic; 2])).unwrap()
}

fn assert_matrix(m: &Matrix, expected: &[&[f64]]) {
    let rows = m.to_rows();
    assert_eq!(rows.len(), expected.len());
    for (r, e) in rows.iter().zip(expected) {
        assert_eq!(&r[..], *e);
    }
}

// ----------------------------------------------------------------------------
// Diameter and friends.

#[test]
fn diameter_backwards_compatible() {
    assert_eq!(path(10).diameter().unwrap(), 9.0);
    assert_eq!(cycle(10).diameter().unwrap(), 5.0);
    assert_eq!(complete(6).diameter().unwrap(), 1.0);
    assert!(Graph::new(0, false).diameter().unwrap().is_nan());
    assert_eq!(Graph::new(1, false).diameter().unwrap(), 0.0);
    // Disconnected: longest geodesic within a component (unconn = true).
    let g = Graph::from_edges(&[(0, 1), (1, 2), (3, 4)], 5, false).unwrap();
    assert_eq!(g.diameter().unwrap(), 2.0);
    // Directed graphs follow edge directions.
    let directed = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, true).unwrap();
    assert_eq!(directed.diameter().unwrap(), 2.0);
    assert_eq!(karate().diameter().unwrap(), 5.0);
}

#[test]
fn diameter_with_path_like_igraph_example() {
    // examples/simple/igraph_diameter.c: directed ring of 10 vertices (not circular).
    let edges: Vec<_> = (0..9).map(|i| (i, i + 1)).collect();
    let ring = Graph::from_edges(&edges, 10, true).unwrap();
    let d = ring.diameter_with_path(None, true, true).unwrap();
    assert_eq!(d.length, 9.0);
    assert_eq!((d.from, d.to), (Some(0), Some(9)));
    assert_eq!(d.path.vertices, (0..10).collect::<Vec<_>>());
    assert_eq!(d.path.edges, (0..9).collect::<Vec<_>>());

    // Not connected and unconn = false: infinite.
    let g = Graph::from_edges(&[(0, 1), (2, 3)], 4, false).unwrap();
    assert_eq!(
        g.diameter_with_path(None, false, false).unwrap().length,
        INF
    );
    // Null graph: NaN and no endpoints.
    let d = Graph::new(0, false)
        .diameter_with_path(None, false, true)
        .unwrap();
    assert!(d.length.is_nan());
    assert_eq!((d.from, d.to), (None, None));
    assert!(!d.path.exists());
}

#[test]
fn weighted_diameter_path_is_consistent() {
    let g = karate();
    let w: Vec<f64> = (0..g.ecount()).map(|e| 1.0 + (e % 5) as f64).collect();
    let d = g.diameter_with_path(Some(&w), false, true).unwrap();
    assert_close(d.path.weight(&w), d.length, 1e-9);
    let (from, to) = (d.from.unwrap(), d.to.unwrap());
    assert_eq!(d.path.vertices.first(), Some(&from));
    assert_eq!(d.path.vertices.last(), Some(&to));
    // It is the maximum entry of the distance matrix.
    let dist = g
        .distances_dijkstra(.., .., Some(&w), NeighborMode::All)
        .unwrap();
    let max = dist.as_slice().iter().cloned().fold(0.0, f64::max);
    assert_close(max, d.length, 1e-9);
    assert_close(dist[(from as usize, to as usize)], d.length, 1e-9);
}

#[test]
fn eccentricity_radius_center() {
    // examples/simple/igraph_eccentricity.c
    let star_edges: Vec<_> = (1..10).map(|i| (0, i)).collect();
    let star = Graph::from_edges(&star_edges, 10, false).unwrap();
    let mut expected = vec![2.0; 10];
    expected[0] = 1.0;
    assert_eq!(
        star.eccentricity(.., None, NeighborMode::Out).unwrap(),
        expected
    );
    let out_star = Graph::from_edges(&star_edges, 10, true).unwrap();
    assert_eq!(
        out_star.eccentricity(.., None, NeighborMode::All).unwrap(),
        expected
    );
    let mut expected_out = vec![0.0; 10];
    expected_out[0] = 1.0;
    assert_eq!(
        out_star.eccentricity(.., None, NeighborMode::Out).unwrap(),
        expected_out
    );
    assert_eq!(
        star.eccentricity(vec![3, 0], None, NeighborMode::All)
            .unwrap(),
        vec![2.0, 1.0]
    );

    // Karate club: radius 3, diameter 5; the center realizes the radius.
    let k = karate();
    let ecc = k.eccentricity(.., None, NeighborMode::All).unwrap();
    let radius = k.radius(None, NeighborMode::All).unwrap();
    assert_eq!(radius, 3.0);
    assert_eq!(
        ecc.iter().cloned().fold(0.0, f64::max),
        k.diameter().unwrap()
    );
    let center = k.graph_center(None, NeighborMode::All).unwrap();
    assert!(!center.is_empty());
    for (v, e) in ecc.iter().enumerate() {
        assert_eq!(center.contains(&(v as i64)), *e == radius, "vertex {v}");
    }
    // Isolated vertices have eccentricity zero.
    assert_eq!(
        Graph::new(3, false)
            .eccentricity(.., None, NeighborMode::All)
            .unwrap(),
        vec![0.0; 3]
    );
}

#[test]
fn pseudo_diameter_is_a_lower_bound() {
    let k = karate();
    let diameter = k.diameter().unwrap();
    for start in k.vertices() {
        let pd = k.pseudo_diameter(None, Some(start), false, true).unwrap();
        assert!(pd.length <= diameter);
        assert!(pd.length >= k.radius(None, NeighborMode::All).unwrap());
        let (from, to) = (pd.from.unwrap(), pd.to.unwrap());
        let d = k.distances(from, to, None, NeighborMode::All).unwrap();
        assert_eq!(d[(0, 0)], pd.length);
    }
    // Exact on paths, whatever the (random) start.
    for seed in 0..5 {
        rng::seed(seed).unwrap();
        let pd = path(12).pseudo_diameter(None, None, false, true).unwrap();
        assert_eq!(pd.length, 11.0);
        let mut ends = [pd.from.unwrap(), pd.to.unwrap()];
        ends.sort();
        assert_eq!(ends, [0, 11]);
    }
    assert!(
        Graph::new(0, false)
            .pseudo_diameter(None, None, false, true)
            .unwrap()
            .length
            .is_nan()
    );
}

// ----------------------------------------------------------------------------
// Average path length, histogram, efficiency.

#[test]
fn tutorial_lesson_2_periodic_lattice_average_path_length() {
    // The igraph tutorial (examples/tutorial/tutorial2.c, which also uses
    // igraph_square_lattice): a 30 x 30 periodic lattice has average path
    // length ~15 (it prints "15.0167").
    let mut g = lattice(30, true);
    assert_eq!((g.vcount(), g.ecount()), (900, 1800));
    let apl = g.average_path_length(None, false, true).unwrap();
    // Each coordinate contributes 7.5 on average over all 900 targets
    // (self included), hence 15 * 900 / 899 over distinct pairs.
    assert_close(apl, 15.0 * 900.0 / 899.0, 1e-9);
    assert_eq!(format!("{apl:.4}"), "15.0167");
    assert_eq!(g.diameter().unwrap(), 30.0);

    // Second part of the lesson: 10 random shortcuts (20 random endpoints)
    // already shrink the distances noticeably: a small world emerges.
    rng::seed(42).unwrap();
    let shortcuts: Vec<(i64, i64)> = (0..10)
        .map(|_| (rng::integer(0, 899), rng::integer(0, 899)))
        .collect();
    g.add_edges(&shortcuts).unwrap();
    let randomized = g.average_path_length(None, false, true).unwrap();
    assert!(randomized < apl, "{randomized} vs {apl}");
    println!("average path length: lattice {apl:.4}, randomized {randomized:.4}");
}

#[test]
fn average_path_length_agrees_with_distances_and_histogram() {
    let k = karate();
    let apl = k.average_path_length(None, false, true).unwrap();
    assert_close(apl, 2.408199643, 1e-9);

    let n = k.vcount() as f64;
    let d = k.distances(.., .., None, NeighborMode::All).unwrap();
    let total: f64 = d.as_slice().iter().sum();
    assert_close(total / (n * (n - 1.0)), apl, 1e-12);

    let h = k.path_length_hist(false).unwrap();
    assert_eq!(h.unconnected, 0.0);
    assert_eq!(h.counts.iter().sum::<f64>(), n * (n - 1.0) / 2.0);
    assert_eq!(h.counts.len(), 5); // up to the diameter
    assert_eq!(h.counts[0], k.ecount() as f64);
    let weighted_sum: f64 = h
        .counts
        .iter()
        .enumerate()
        .map(|(i, c)| (i + 1) as f64 * c)
        .sum();
    assert_close(weighted_sum / h.counts.iter().sum::<f64>(), apl, 1e-12);
}

#[test]
fn path_length_hist_like_igraph_unit_test() {
    // tests/unit/igraph_path_length_hist.c
    let g = Graph::from_edges(
        &[
            (0, 1),
            (0, 2),
            (1, 1),
            (1, 2),
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
    let h = g.path_length_hist(false).unwrap();
    assert_eq!((h.counts, h.unconnected), (vec![6.0, 3.0, 1.0], 5.0));
    let h = g.path_length_hist(true).unwrap();
    assert_eq!((h.counts, h.unconnected), (vec![7.0, 5.0, 1.0], 17.0));
    let h = Graph::new(2, false).path_length_hist(true).unwrap();
    assert_eq!((h.counts, h.unconnected), (vec![], 1.0));
}

#[test]
fn average_path_length_disconnected() {
    let g = Graph::from_edges(&[(0, 1), (1, 2), (3, 4)], 5, false).unwrap();
    let apl = g.average_path_length_details(None, false, true).unwrap();
    // Connected ordered pairs: 6 in the path (1,1,2 twice) + 2 in the edge.
    assert_close(apl.average, (2.0 * (1.0 + 1.0 + 2.0) + 2.0) / 8.0, 1e-12);
    assert_eq!(apl.unconnected_pairs, 12.0);
    assert_eq!(g.average_path_length(None, false, false).unwrap(), INF);
    assert!(
        Graph::new(1, false)
            .average_path_length(None, false, true)
            .unwrap()
            .is_nan()
    );
}

#[test]
fn efficiency_like_igraph_unit_test() {
    // tests/unit/efficiency.c: directed 4-ring.
    let ring = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], 4, true).unwrap();
    for w in [None, Some(&[1.0; 4][..])] {
        assert_close(ring.global_efficiency(w, false).unwrap(), 0.833333, 1e-6);
        assert_close(ring.global_efficiency(w, true).unwrap(), 0.611111, 1e-6);
        assert_close(
            ring.average_local_efficiency(w, false, NeighborMode::All)
                .unwrap(),
            0.5,
            1e-12,
        );
        assert_close(
            ring.average_local_efficiency(w, true, NeighborMode::All)
                .unwrap(),
            0.25,
            1e-12,
        );
        assert_eq!(
            ring.average_local_efficiency(w, true, NeighborMode::Out)
                .unwrap(),
            0.0
        );
        assert_eq!(
            ring.local_efficiency(.., w, false, NeighborMode::All)
                .unwrap(),
            vec![0.5; 4]
        );
        assert_eq!(
            ring.local_efficiency(.., w, true, NeighborMode::All)
                .unwrap(),
            vec![0.25; 4]
        );
        assert_eq!(
            ring.local_efficiency(.., w, true, NeighborMode::In)
                .unwrap(),
            vec![0.0; 4]
        );
    }
    // Global efficiency is the mean of the inverse distances.
    let k = karate();
    let d = k.distances(.., .., None, NeighborMode::All).unwrap();
    let n = k.vcount() as f64;
    let inv: f64 = d
        .as_slice()
        .iter()
        .filter(|&&x| x > 0.0)
        .map(|x| 1.0 / x)
        .sum();
    assert_close(
        k.global_efficiency(None, false).unwrap(),
        inv / (n * (n - 1.0)),
        1e-12,
    );
    // In a complete graph, removing a vertex leaves its neighbors fully connected.
    assert_eq!(
        complete(5)
            .local_efficiency(0, None, false, NeighborMode::All)
            .unwrap(),
        vec![1.0]
    );
}

// ----------------------------------------------------------------------------
// Distance matrices.

#[test]
fn distances_like_igraph_example() {
    // examples/simple/distances.c and its .out file.
    let (g, w) = example_graph();
    let unweighted: [&[f64]; 10] = [
        &[0., 1., 1., 1., 2., 2., 2., 3., 3., 3.],
        &[INF, 0., 1., 2., 1., 1., 2., 2., 2., 2.],
        &[INF, 1., 0., 1., 2., 2., 1., 3., 3., 3.],
        &[INF, 2., 1., 0., 3., 3., 1., 4., 4., 4.],
        &[INF, 3., 2., 3., 0., 1., 2., 1., 2., 2.],
        &[INF, 2., 1., 2., 3., 0., 1., 4., 1., 1.],
        &[INF, INF, INF, INF, INF, INF, 0., INF, INF, INF],
        &[INF, 3., 2., 3., 4., 1., 2., 0., 1., 2.],
        &[INF, INF, INF, INF, INF, INF, INF, INF, 0., 1.],
        &[INF, INF, INF, INF, INF, INF, INF, INF, INF, 0.],
    ];
    assert_matrix(
        &g.distances(.., .., None, NeighborMode::Out).unwrap(),
        &unweighted,
    );

    let cut = g
        .distances_cutoff(.., .., None, NeighborMode::Out, Some(3.0))
        .unwrap();
    assert_eq!(cut.row(3), vec![INF, 2., 1., 0., 3., 3., 1., INF, INF, INF]);
    assert_eq!(cut.row(7), vec![INF, 3., 2., 3., INF, 1., 2., 0., 1., 2.]);
    // No cutoff is the same as plain distances.
    assert_matrix(
        &g.distances_cutoff(.., .., None, NeighborMode::Out, None)
            .unwrap(),
        &unweighted,
    );

    let weighted: [&[f64]; 10] = [
        &[0., 0., 0., 1., 5., 2., 1., 13., 3., 5.],
        &[INF, 0., 0., 1., 5., 2., 1., 13., 3., 5.],
        &[INF, 1., 0., 1., 6., 3., 1., 14., 4., 6.],
        &[INF, 1., 0., 0., 6., 3., 1., 14., 4., 6.],
        &[INF, 5., 4., 5., 0., 2., 3., 8., 3., 5.],
        &[INF, 3., 2., 3., 8., 0., 1., 16., 1., 3.],
        &[INF, INF, INF, INF, INF, INF, 0., INF, INF, INF],
        &[INF, 4., 3., 4., 9., 1., 2., 0., 1., 4.],
        &[INF, INF, INF, INF, INF, INF, INF, INF, 0., 4.],
        &[INF, INF, INF, INF, INF, INF, INF, INF, INF, 0.],
    ];
    let wd = Some(&w[..]);
    assert_matrix(
        &g.distances(.., .., wd, NeighborMode::Out).unwrap(),
        &weighted,
    );
    assert_matrix(
        &g.distances_dijkstra(.., .., wd, NeighborMode::Out).unwrap(),
        &weighted,
    );
    assert_matrix(
        &g.distances_bellman_ford(.., .., wd, NeighborMode::Out)
            .unwrap(),
        &weighted,
    );
    assert_matrix(
        &g.distances_johnson(.., .., wd, NeighborMode::Out).unwrap(),
        &weighted,
    );
    for method in [
        FloydWarshallAlgorithm::Automatic,
        FloydWarshallAlgorithm::Original,
        FloydWarshallAlgorithm::Tree,
    ] {
        assert_matrix(
            &g.distances_floyd_warshall(.., .., wd, NeighborMode::Out, method)
                .unwrap(),
            &weighted,
        );
    }

    let cut = g
        .distances_dijkstra_cutoff(.., .., wd, NeighborMode::Out, Some(8.0))
        .unwrap();
    assert_eq!(cut.row(0), vec![0., 0., 0., 1., 5., 2., 1., INF, 3., 5.]);
    assert_eq!(cut.row(4), vec![INF, 5., 4., 5., 0., 2., 3., 8., 3., 5.]);
    let cut = g
        .distances_cutoff(.., .., wd, NeighborMode::Out, Some(8.0))
        .unwrap();
    assert_eq!(cut.row(7), vec![INF, 4., 3., 4., INF, 1., 2., 0., 1., 4.]);

    // Mode In gives the transposed matrix; subsets select rows and columns.
    let d_in = g.distances(.., .., None, NeighborMode::In).unwrap();
    assert_eq!(
        d_in.transposed(),
        g.distances(.., .., None, NeighborMode::Out).unwrap()
    );
    let sub = g
        .distances(vec![4, 0], 5..8, None, NeighborMode::Out)
        .unwrap();
    assert_eq!(sub.to_rows(), vec![vec![1., 2., 1.], vec![2., 2., 3.]]);
}

#[test]
fn negative_weights_bellman_ford_johnson_floyd_warshall() {
    // examples/simple/bellman_ford.c, second graph.
    let g = Graph::from_edges(
        &[
            (0, 1),
            (0, 3),
            (1, 3),
            (1, 4),
            (2, 1),
            (3, 2),
            (3, 4),
            (4, 0),
            (4, 2),
        ],
        5,
        true,
    )
    .unwrap();
    let w = [6., 7., 8., -4., -2., -3., 9., 2., 7.];
    let expected: [&[f64]; 5] = [
        &[0., 2., 4., 7., -2.],
        &[-2., 0., 2., 5., -4.],
        &[-4., -2., 0., 3., -6.],
        &[-7., -5., -3., 0., -9.],
        &[2., 4., 6., 9., 0.],
    ];
    assert_matrix(
        &g.distances_bellman_ford(.., .., Some(&w), NeighborMode::Out)
            .unwrap(),
        &expected,
    );
    assert_matrix(
        &g.distances_johnson(.., .., Some(&w), NeighborMode::Out)
            .unwrap(),
        &expected,
    );
    assert_matrix(
        &g.distances(.., .., Some(&w), NeighborMode::Out).unwrap(),
        &expected,
    );
    assert_matrix(
        &g.distances_floyd_warshall(
            ..,
            ..,
            Some(&w),
            NeighborMode::Out,
            FloydWarshallAlgorithm::Tree,
        )
        .unwrap(),
        &expected,
    );
    // Dijkstra refuses negative weights.
    let err = g
        .distances_dijkstra(.., .., Some(&w), NeighborMode::Out)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);

    // The shortest path from 0 to 4 uses the negative edges: 0 → 1 → 4 (6 - 4 = 2)?
    // No: 0 → 3 → 2 → 1 → 4 costs 7 - 3 - 2 - 4 = -2.
    let p = g
        .get_shortest_path_bellman_ford(0, 4, Some(&w), NeighborMode::Out)
        .unwrap();
    assert_eq!(p.vertices, vec![0, 3, 2, 1, 4]);
    assert_eq!(p.weight(&w), -2.0);
    let sp = g
        .get_shortest_paths_bellman_ford(0, .., Some(&w), NeighborMode::Out)
        .unwrap();
    for (target, path) in sp.vertices.iter().enumerate() {
        let edges = &sp.edges[target];
        let len: f64 = edges.iter().map(|&e| w[e as usize]).sum();
        assert_eq!(len, expected[0][target], "path {path:?}");
    }

    // A negative cycle is detected.
    let w_cycle = [6., 7., 2., -4., -2., -3., 9., 2., 7.];
    for err in [
        g.distances_bellman_ford(.., .., Some(&w_cycle), NeighborMode::Out)
            .unwrap_err(),
        g.distances_johnson(.., .., Some(&w_cycle), NeighborMode::Out)
            .unwrap_err(),
        g.get_shortest_path_bellman_ford(0, 4, Some(&w_cycle), NeighborMode::Out)
            .unwrap_err(),
    ] {
        assert_eq!(err.kind(), ErrorKind::NegativeCycle);
    }
    // Johnson rejects undirected graphs with negative weights, reporting a
    // negative cycle (an undirected negative edge is one), even from a
    // source that cannot reach the negative edge.
    let u = Graph::from_edges(&[(0, 1), (2, 3)], 4, false).unwrap();
    assert_eq!(
        u.distances_johnson(0, .., Some(&[1.0, -1.0]), NeighborMode::All)
            .unwrap_err()
            .kind(),
        ErrorKind::NegativeCycle
    );
    // Bellman-Ford only complains when the negative edge is reachable.
    assert_eq!(
        u.distances_bellman_ford(0, .., Some(&[1.0, -1.0]), NeighborMode::All)
            .unwrap()
            .row(0),
        vec![0.0, 1.0, INF, INF]
    );
}

#[test]
fn distances_on_grid_are_manhattan() {
    let g = lattice(6, false);
    let d = g.distances(.., .., None, NeighborMode::All).unwrap();
    for a in 0..36usize {
        for b in 0..36usize {
            let manhattan = (a % 6).abs_diff(b % 6) + (a / 6).abs_diff(b / 6);
            assert_eq!(d[(a, b)], manhattan as f64);
        }
    }
}

// ----------------------------------------------------------------------------
// Shortest paths.

#[test]
fn get_shortest_paths_dijkstra_like_igraph_example() {
    // examples/simple/igraph_get_shortest_paths_dijkstra.c
    let (g, w) = example_graph();
    let sp = g
        .get_shortest_paths_dijkstra(0, vec![0, 1, 3, 5, 2, 1], Some(&w), NeighborMode::Out)
        .unwrap();
    assert_eq!(
        sp.vertices,
        vec![
            vec![0],
            vec![0, 1],
            vec![0, 3],
            vec![0, 1, 5],
            vec![0, 1, 2],
            vec![0, 1]
        ]
    );
    assert_eq!(
        sp.edges,
        vec![vec![], vec![0], vec![2], vec![0, 5], vec![0, 3], vec![0]]
    );
    assert_eq!(sp.parents, vec![-1, 0, 1, 0, 1, 1, 2, -2, 5, 5]);
    assert_eq!(sp.inbound_edges, vec![-1, 0, 3, 2, 4, 5, 7, -1, 13, 14]);
    assert_eq!(sp.path(3).unwrap().vertices, vec![0, 1, 5]);
    assert_eq!(sp.path(6), None);

    // The generic interface with weights gives the same paths.
    let generic = g
        .get_shortest_paths(0, vec![0, 1, 3, 5, 2, 1], Some(&w), NeighborMode::Out)
        .unwrap();
    assert_eq!(generic.vertices, sp.vertices);
}

#[test]
fn single_shortest_path_variants_agree() {
    let g = karate();
    let w: Vec<f64> = (0..g.ecount()).map(|e| ((e * 7) % 11 + 1) as f64).collect();
    let dist = g
        .distances_dijkstra(.., .., Some(&w), NeighborMode::All)
        .unwrap();
    for (from, to) in [(0, 33), (16, 25), (4, 29), (11, 9)] {
        let d = dist[(from as usize, to as usize)];
        let dj = g
            .get_shortest_path_dijkstra(from, to, Some(&w), NeighborMode::All)
            .unwrap();
        let bf = g
            .get_shortest_path_bellman_ford(from, to, Some(&w), NeighborMode::All)
            .unwrap();
        let generic = g
            .get_shortest_path(from, to, Some(&w), NeighborMode::All)
            .unwrap();
        let astar = g
            .get_shortest_path_astar(from, to, Some(&w), NeighborMode::All, |_, _| 0.0)
            .unwrap();
        for p in [&dj, &bf, &generic, &astar] {
            assert_eq!(p.weight(&w), d);
            assert_eq!(p.vertices[0], from);
            assert_eq!(*p.vertices.last().unwrap(), to);
            // The vertex path can be recovered from the edge path.
            assert_eq!(
                g.vertex_path_from_edge_path(Some(from), &p.edges, NeighborMode::All)
                    .unwrap(),
                p.vertices
            );
            // ... and the edge path from the vertex path.
            let pairs = expand_path_to_pairs(&p.vertices).unwrap();
            let eids = g.get_eids(&pairs, false).unwrap();
            let len: f64 = eids.iter().map(|&e| w[e as usize]).sum();
            assert_eq!(len, d);
        }
    }
    // Unreachable targets give an empty path.
    let g = Graph::from_edges(&[(0, 1)], 3, false).unwrap();
    let p = g.get_shortest_path(0, 2, None, NeighborMode::All).unwrap();
    assert!(!p.exists() && p.is_empty());
}

#[test]
fn astar_on_grid_with_manhattan_heuristic() {
    let side = 12;
    let g = lattice(side as usize, false);
    let manhattan =
        |a: i64, b: i64| ((a % side - b % side).abs() + (a / side - b / side).abs()) as f64;
    let mut calls = 0usize;
    let target = side * side - 1;
    let p = g
        .get_shortest_path_astar(0, target, None, NeighborMode::All, |v, to| {
            assert_eq!(to, target);
            calls += 1;
            manhattan(v, to)
        })
        .unwrap();
    assert_eq!(p.len() as i64, 2 * (side - 1));
    assert!(calls > 0);
    // Consistency: each step decreases the Manhattan distance by one.
    for pair in p.vertices.windows(2) {
        assert_eq!(manhattan(pair[0], target) - 1.0, manhattan(pair[1], target));
    }
    // Trivial path.
    assert_eq!(
        g.get_shortest_path_astar(5, 5, None, NeighborMode::All, |_, _| 0.0)
            .unwrap()
            .vertices,
        vec![5]
    );
}

#[test]
fn astar_heuristic_panic_is_propagated() {
    let g = path(5);
    let result = std::panic::catch_unwind(|| {
        let _ = g.get_shortest_path_astar(0, 4, None, NeighborMode::All, |_, _| -> f64 {
            panic!("boom")
        });
    });
    let payload = result.unwrap_err();
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"boom"));
    // The library is still usable afterwards.
    assert_eq!(
        g.get_shortest_path(0, 4, None, NeighborMode::All)
            .unwrap()
            .len(),
        4
    );
}

#[test]
fn all_shortest_paths_count_lattice_geodesics() {
    // In an n x n grid, there are C(2(n-1), n-1) geodesics between opposite corners.
    let g = lattice(4, false);
    let all = g
        .get_all_shortest_paths(0, 15, None, NeighborMode::All)
        .unwrap();
    assert_eq!(all.vertices.len(), 20);
    assert_eq!(all.edges.len(), 20);
    assert_eq!(all.nrgeo[15], 20);
    let mut unique = all.vertices.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), 20);
    for (vp, ep) in all.vertices.iter().zip(&all.edges) {
        assert_eq!(vp.len(), 7);
        assert_eq!(
            g.vertex_path_from_edge_path(Some(0), ep, NeighborMode::All)
                .unwrap(),
            *vp
        );
    }
    // To all vertices: nrgeo(x, y) = C(x + y, x).
    let all = g
        .get_all_shortest_paths(0, .., None, NeighborMode::All)
        .unwrap();
    let binom = |n: i64, k: i64| (0..k).fold(1, |acc, i| acc * (n - i) / (i + 1));
    for v in 0..16 {
        assert_eq!(all.nrgeo[v as usize], binom(v % 4 + v / 4, v % 4));
    }
    assert_eq!(all.vertices.len() as i64, all.nrgeo.iter().sum::<i64>());

    // Weighted: making one edge expensive removes the geodesics through it.
    let c4 = cycle(4);
    let w = [1.0, 1.0, 1.0, 5.0]; // edge (3, 0) is expensive
    let all = c4
        .get_all_shortest_paths_dijkstra(0, 2, Some(&w), NeighborMode::All)
        .unwrap();
    assert_eq!(all.vertices, vec![vec![0, 1, 2]]);
    let all = c4
        .get_all_shortest_paths(0, 2, Some(&w), NeighborMode::All)
        .unwrap();
    assert_eq!(all.vertices, vec![vec![0, 1, 2]]);
    let all = c4
        .get_all_shortest_paths_dijkstra(0, 2, Some(&[1.0; 4]), NeighborMode::All)
        .unwrap();
    assert_eq!(all.nrgeo[2], 2);
}

#[test]
fn k_shortest_paths_are_sorted_and_distinct() {
    let g = karate();
    let w: Vec<f64> = (0..g.ecount()).map(|e| ((e * 3) % 7 + 1) as f64).collect();
    let paths = g
        .get_k_shortest_paths(16, 25, 8, Some(&w), NeighborMode::All)
        .unwrap();
    assert_eq!(paths.len(), 8);
    let lengths: Vec<f64> = paths.iter().map(|p| p.weight(&w)).collect();
    assert!(lengths.windows(2).all(|l| l[0] <= l[1]));
    let shortest = g
        .distances_dijkstra(16, 25, Some(&w), NeighborMode::All)
        .unwrap()[(0, 0)];
    assert_eq!(lengths[0], shortest);
    for (i, p) in paths.iter().enumerate() {
        assert_eq!((p.vertices[0], *p.vertices.last().unwrap()), (16, 25));
        let mut seen = p.vertices.clone();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), p.vertices.len(), "path {i} is not simple");
        for q in &paths[..i] {
            assert_ne!(q.edges, p.edges);
        }
    }
    // k = 0 gives nothing; a path graph has a single path.
    assert!(
        g.get_k_shortest_paths(0, 1, 0, None, NeighborMode::All)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        path(5)
            .get_k_shortest_paths(0, 4, 3, None, NeighborMode::All)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn all_simple_paths_in_complete_graph() {
    // In K5, the simple paths from 0 to 1 are: 0-1, and 0-S-1 for any
    // ordered selection S of the 3 other vertices: 1 + 3 + 6 + 6 = 16.
    let k5 = complete(5);
    let all = k5
        .get_all_simple_paths(0, 1, NeighborMode::All, SimplePathsOptions::default())
        .unwrap();
    assert_eq!(all.len(), 16);
    for p in &all {
        assert_eq!((p[0], *p.last().unwrap()), (0, 1));
    }
    let opts = SimplePathsOptions::default()
        .with_min_len(2)
        .with_max_len(3);
    assert_eq!(
        k5.get_all_simple_paths(0, 1, NeighborMode::All, opts)
            .unwrap()
            .len(),
        9
    );
    let opts = SimplePathsOptions::default().with_max_results(4);
    assert_eq!(
        k5.get_all_simple_paths(0, 1, NeighborMode::All, opts)
            .unwrap()
            .len(),
        4
    );
    // To every other vertex.
    let all = k5
        .get_all_simple_paths(0, .., NeighborMode::All, SimplePathsOptions::default())
        .unwrap();
    assert_eq!(all.len(), 4 * 16);
    // Directed: follow the directions only.
    let dag = Graph::from_edges(&[(0, 1), (1, 2), (0, 2)], 3, true).unwrap();
    assert_eq!(
        dag.get_all_simple_paths(0, 2, NeighborMode::Out, SimplePathsOptions::default())
            .unwrap()
            .len(),
        2
    );
    assert!(
        dag.get_all_simple_paths(2, 0, NeighborMode::Out, SimplePathsOptions::default())
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        dag.get_all_simple_paths(2, 0, NeighborMode::In, SimplePathsOptions::default())
            .unwrap()
            .len(),
        2
    );
}

// ----------------------------------------------------------------------------
// Widest paths.

#[test]
fn widest_paths_maximize_the_bottleneck() {
    let g = karate();
    let w: Vec<f64> = (0..g.ecount()).map(|e| ((e * 13) % 17) as f64).collect();
    let fw = g
        .widest_path_widths_floyd_warshall(.., .., &w, NeighborMode::All)
        .unwrap();
    let dj = g
        .widest_path_widths_dijkstra(.., .., &w, NeighborMode::All)
        .unwrap();
    assert_eq!(fw, dj);
    let sp = g.get_widest_paths(0, .., &w, NeighborMode::All).unwrap();
    for target in 1..34 {
        let edges = &sp.edges[target];
        let bottleneck = edges
            .iter()
            .map(|&e| w[e as usize])
            .fold(f64::INFINITY, f64::min);
        assert_eq!(bottleneck, dj[(0, target)]);
        let single = g
            .get_widest_path(0, target as i64, &w, NeighborMode::All)
            .unwrap();
        let b2 = single
            .edges
            .iter()
            .map(|&e| w[e as usize])
            .fold(f64::INFINITY, f64::min);
        assert_eq!(b2, bottleneck);
    }
    assert_eq!(dj[(5, 5)], f64::INFINITY);
    // Unreachable vertices have width -inf.
    let g = Graph::from_edges(&[(0, 1)], 3, false).unwrap();
    let w = g
        .widest_path_widths_dijkstra(0, .., &[4.0], NeighborMode::All)
        .unwrap();
    assert_eq!(w.row(0), vec![INF, 4.0, -INF]);
    // Weights are mandatory and must match the edge count.
    assert_eq!(
        g.get_widest_path(0, 1, &[], NeighborMode::All)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

// ----------------------------------------------------------------------------
// Random walks, spanners, Voronoi.

#[test]
fn random_walk_behaviour() {
    let k = karate();
    let walk_from_0 = |seed: u64| {
        rng::seed(seed).unwrap();
        k.random_walk(0, 1000, None, NeighborMode::All, RandomWalkStuck::Error)
            .unwrap()
    };
    let walk = walk_from_0(42);
    assert_eq!((walk.vertices.len(), walk.edges.len()), (1001, 1000));
    assert_eq!(
        k.vertex_path_from_edge_path(Some(0), &walk.edges, NeighborMode::All)
            .unwrap(),
        walk.vertices
    );

    // Same seed, same walk.
    assert_eq!(walk_from_0(42), walk);
    assert_ne!(walk_from_0(43), walk);
    // A private generator installed with `Rng::scoped` gives its own,
    // equally reproducible, stream.
    let scoped = |seed: u64| {
        Rng::new(RngType::Pcg64, seed).unwrap().scoped(|| {
            k.random_walk(0, 1000, None, NeighborMode::All, RandomWalkStuck::Error)
                .unwrap()
        })
    };
    assert_eq!(scoped(42), scoped(42));

    // The stationary distribution of a random walk is proportional to the
    // degree: in a long walk the hub 0 (degree 16) is visited far more often
    // than the leaf 11 (degree 1).
    let long = walk_from_0(5);
    let visits_0 = long.vertices.iter().filter(|&&v| v == 0).count();
    let visits_11 = long.vertices.iter().filter(|&&v| v == 11).count();
    assert!(visits_0 > 4 * visits_11, "{visits_0} vs {visits_11}");

    // Zero-weight edges are never taken.
    let mut w = vec![1.0; k.ecount()];
    w[0] = 0.0; // edge (0, 1)
    rng::seed(1).unwrap();
    let walk = k
        .random_walk(0, 2000, Some(&w), NeighborMode::All, RandomWalkStuck::Error)
        .unwrap();
    assert!(!walk.edges.contains(&0));

    // A directed path gets stuck at its end.
    let dp = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    let short = dp
        .random_walk(0, 10, None, NeighborMode::Out, RandomWalkStuck::Return)
        .unwrap();
    assert_eq!(short.vertices, vec![0, 1, 2]);
    let err = dp
        .random_walk(0, 10, None, NeighborMode::Out, RandomWalkStuck::Error)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::RandomWalkStuck);
}

#[test]
fn spanner_respects_the_stretch() {
    rng::seed(7).unwrap();
    let g = complete(20);
    let w: Vec<f64> = (0..g.ecount())
        .map(|e| 1.0 + ((e * 37) % 10) as f64)
        .collect();
    for stretch in [1.0, 3.0, 5.0] {
        let kept = g.spanner(stretch, Some(&w)).unwrap();
        let all_edges = g.edge_list();
        let edges: Vec<_> = kept.iter().map(|&e| all_edges[e as usize]).collect();
        let sw: Vec<f64> = kept.iter().map(|&e| w[e as usize]).collect();
        let h = Graph::from_edges(&edges, g.vcount(), false).unwrap();
        let dg = g.distances(.., .., Some(&w), NeighborMode::All).unwrap();
        let dh = h.distances(.., .., Some(&sw), NeighborMode::All).unwrap();
        for (a, b) in dg.as_slice().iter().zip(dh.as_slice()) {
            assert!(
                *b <= stretch * a + 1e-9,
                "stretch {stretch} violated: {b} > {stretch} * {a}"
            );
        }
        if stretch >= 3.0 {
            assert!(kept.len() < g.ecount());
        }
    }
}

#[test]
fn voronoi_cells() {
    let g = lattice(5, false);
    // Generators at two opposite corners; the anti-diagonal is at equal distance.
    let first = g
        .voronoi(&[0, 24], None, NeighborMode::All, VoronoiTiebreaker::First)
        .unwrap();
    let last = g
        .voronoi(&[0, 24], None, NeighborMode::All, VoronoiTiebreaker::Last)
        .unwrap();
    for v in 0..25usize {
        let (x, y) = (v % 5, v / 5);
        let d0 = (x + y) as f64;
        let d1 = (8 - x - y) as f64;
        assert_eq!(first.distances[v], d0.min(d1));
        match (x + y).cmp(&4) {
            std::cmp::Ordering::Less => {
                assert_eq!((first.membership[v], last.membership[v]), (0, 0))
            }
            std::cmp::Ordering::Greater => {
                assert_eq!((first.membership[v], last.membership[v]), (1, 1))
            }
            std::cmp::Ordering::Equal => {
                assert_eq!((first.membership[v], last.membership[v]), (0, 1))
            }
        }
    }
    // Unreachable vertices.
    let g = Graph::from_edges(&[(0, 1)], 3, true).unwrap();
    let v = g
        .voronoi(&[0], None, NeighborMode::Out, VoronoiTiebreaker::Random)
        .unwrap();
    assert_eq!(v.membership, vec![0, 0, -1]);
    assert_eq!(v.distances, vec![0.0, 1.0, INF]);
    // Weighted: a heavy edge moves the border.
    let p = path(4);
    let v = p
        .voronoi(
            &[0, 3],
            Some(&[1.0, 10.0, 1.0]),
            NeighborMode::All,
            VoronoiTiebreaker::First,
        )
        .unwrap();
    assert_eq!(v.membership, vec![0, 0, 1, 1]);
}

#[test]
fn path_conversions() {
    assert_eq!(expand_path_to_pairs(&[]).unwrap(), vec![]);
    assert_eq!(
        expand_path_to_pairs(&[1, 2, 3, 4]).unwrap(),
        vec![(1, 2), (2, 3), (3, 4)]
    );
    let g = cycle(5);
    // Walk around the cycle twice.
    let edges: Vec<i64> = (0..10).map(|i| i % 5).collect();
    let walk = g
        .vertex_path_from_edge_path(Some(0), &edges, NeighborMode::All)
        .unwrap();
    assert_eq!(walk, vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0]);
    // The start vertex can be inferred.
    assert_eq!(
        g.vertex_path_from_edge_path(None, &[0, 1], NeighborMode::All)
            .unwrap(),
        vec![0, 1, 2]
    );
    assert_eq!(
        g.vertex_path_from_edge_path(Some(3), &[], NeighborMode::All)
            .unwrap(),
        vec![3]
    );
    // Discontinuous walks are rejected.
    let err = g
        .vertex_path_from_edge_path(Some(0), &[0, 2], NeighborMode::All)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

// ----------------------------------------------------------------------------
// Error paths.

#[test]
fn errors_are_reported() {
    let g = path(4);
    let kind = |r: Result<Matrix>| r.unwrap_err().kind();
    assert_eq!(
        kind(g.distances(10, .., None, NeighborMode::All)),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        kind(g.distances(.., vec![0, 99], None, NeighborMode::All)),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        kind(g.distances(.., .., Some(&[1.0]), NeighborMode::All)),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        kind(g.distances_dijkstra(.., .., Some(&[1.0, f64::NAN, 1.0]), NeighborMode::All)),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.get_shortest_path(0, 9, None, NeighborMode::All)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        g.get_shortest_paths(-1, .., None, NeighborMode::All)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        g.eccentricity(.., Some(&[1.0, -1.0, 1.0]), NeighborMode::All)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.random_walk(7, 3, None, NeighborMode::All, RandomWalkStuck::Return)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.voronoi(&[17], None, NeighborMode::All, VoronoiTiebreaker::First)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidVertexId
    );
    let err = g
        .diameter_with_path(Some(&[1.0; 5]), false, true)
        .unwrap_err();
    assert!(err.message().contains("weight"));
}

#[test]
fn invalid_ids_and_arguments_are_rejected() {
    let g = path(4);
    // Edge ids out of range are caught on the Rust side (igraph itself would
    // read out of bounds here).
    for bad in [&[0, 3][..], &[-1][..], &[0, 1, 99][..]] {
        assert_eq!(
            g.vertex_path_from_edge_path(Some(0), bad, NeighborMode::All)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidEdgeId
        );
    }
    // An explicit negative start vertex is an error, not "infer it".
    assert_eq!(
        g.vertex_path_from_edge_path(Some(-1), &[0], NeighborMode::All)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        g.vertex_path_from_edge_path(Some(4), &[], NeighborMode::All)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidVertexId
    );
    // The start cannot be inferred from an empty walk.
    assert_eq!(
        g.vertex_path_from_edge_path(None, &[], NeighborMode::All)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    for start in [-3, 4] {
        assert_eq!(
            g.pseudo_diameter(None, Some(start), false, true)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidVertexId
        );
    }
    assert_eq!(
        g.get_shortest_path_astar(0, 9, None, NeighborMode::All, |_, _| 0.0)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        g.get_k_shortest_paths(0, 9, 2, None, NeighborMode::All)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        g.get_all_simple_paths(9, .., NeighborMode::All, SimplePathsOptions::default())
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidVertexId
    );
    // Duplicate targets are not allowed in distance matrices.
    assert_eq!(
        g.distances(0, vec![1, 1], None, NeighborMode::All)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    // A NaN cutoff is meaningless.
    assert_eq!(
        g.distances_cutoff(0, .., None, NeighborMode::All, Some(f64::NAN))
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    // Stretch factors below one, NaN or infinite are meaningless (igraph
    // would loop forever on an infinite one).
    for stretch in [0.5, f64::NAN, f64::INFINITY] {
        assert_eq!(
            g.spanner(stretch, None).unwrap_err().kind(),
            ErrorKind::InvalidValue
        );
    }
    // Absurd step counts are rejected instead of overflowing in C.
    assert_eq!(
        g.random_walk(
            0,
            usize::MAX,
            None,
            NeighborMode::All,
            RandomWalkStuck::Return
        )
        .unwrap_err()
        .kind(),
        ErrorKind::InvalidValue
    );
    // Huge limits saturate instead of wrapping to "unlimited".
    let none = SimplePathsOptions::default().with_min_len(usize::MAX);
    assert!(
        g.get_all_simple_paths(0, .., NeighborMode::All, none)
            .unwrap()
            .is_empty()
    );
    let all = SimplePathsOptions::default().with_max_results(usize::MAX);
    assert_eq!(
        g.get_all_simple_paths(0, .., NeighborMode::All, all)
            .unwrap()
            .len(),
        3
    );
    // k larger than the number of paths just returns all of them.
    assert_eq!(
        cycle(5)
            .get_k_shortest_paths(0, 2, usize::MAX, None, NeighborMode::All)
            .unwrap()
            .len(),
        2
    );
}

// ----------------------------------------------------------------------------
// A use case story.

/// A small city metro network: stations are vertices, travel times in minutes
/// are the weights and track capacities (trains per hour) are the widths.
#[test]
fn use_case_metro_network() {
    const STATIONS: [&str; 8] = [
        "Harbor",
        "Market",
        "Museum",
        "Central",
        "Park",
        "Stadium",
        "University",
        "Airport",
    ];
    let tracks = [
        (0, 1, 4.0, 30.0), // Harbor - Market
        (1, 2, 3.0, 30.0), // Market - Museum
        (2, 3, 2.0, 40.0), // Museum - Central
        (1, 3, 6.0, 12.0), // Market - Central (old express line)
        (3, 4, 3.0, 40.0), // Central - Park
        (4, 5, 5.0, 20.0), // Park - Stadium
        (3, 6, 7.0, 25.0), // Central - University
        (6, 7, 9.0, 15.0), // University - Airport
        (5, 7, 6.0, 10.0), // Stadium - Airport
    ];
    let edges: Vec<_> = tracks.iter().map(|&(a, b, _, _)| (a, b)).collect();
    let minutes: Vec<f64> = tracks.iter().map(|t| t.2).collect();
    let capacity: Vec<f64> = tracks.iter().map(|t| t.3).collect();
    let metro = Graph::from_edges(&edges, STATIONS.len(), false).unwrap();
    let name = |v: &i64| STATIONS[*v as usize];

    // 1. The fastest trip from the Harbor to the Airport.
    let trip = metro
        .get_shortest_path_dijkstra(0, 7, Some(&minutes), NeighborMode::All)
        .unwrap();
    let route: Vec<_> = trip.vertices.iter().map(name).collect();
    assert_eq!(
        route,
        [
            "Harbor", "Market", "Museum", "Central", "Park", "Stadium", "Airport"
        ]
    );
    assert_eq!(trip.weight(&minutes), 23.0);

    // 2. Alternatives, should a line be closed: the three fastest itineraries.
    let options = metro
        .get_k_shortest_paths(0, 7, 3, Some(&minutes), NeighborMode::All)
        .unwrap();
    let times: Vec<f64> = options.iter().map(|p| p.weight(&minutes)).collect();
    assert_eq!(times, vec![23.0, 24.0, 25.0]);
    // The runner-up takes the old express line from the Market straight to Central.
    let second: Vec<_> = options[1].vertices.iter().map(name).collect();
    assert_eq!(
        second,
        ["Harbor", "Market", "Central", "Park", "Stadium", "Airport"]
    );

    // 3. The route carrying most trains per hour (maximum bottleneck).
    let fat = metro
        .get_widest_path(0, 7, &capacity, NeighborMode::All)
        .unwrap();
    let bottleneck = fat
        .edges
        .iter()
        .map(|&e| capacity[e as usize])
        .fold(f64::INFINITY, f64::min);
    assert_eq!(bottleneck, 15.0);
    assert!(fat.vertices.iter().map(name).any(|s| s == "University"));

    // 4. Where to put the control room: the station minimizing the worst-case travel time.
    let center = metro
        .graph_center(Some(&minutes), NeighborMode::All)
        .unwrap();
    assert_eq!(center.iter().map(name).collect::<Vec<_>>(), ["Park"]);
    let radius = metro.radius(Some(&minutes), NeighborMode::All).unwrap();
    assert_eq!(radius, 12.0);
    let ecc = metro
        .eccentricity(4, Some(&minutes), NeighborMode::All)
        .unwrap();
    assert_eq!(ecc, vec![radius]);

    // 5. Two depots (Harbor and Airport): which trains go where?
    let depots = metro
        .voronoi(
            &[0, 7],
            Some(&minutes),
            NeighborMode::All,
            VoronoiTiebreaker::First,
        )
        .unwrap();
    let served_by_harbor: Vec<_> = metro
        .vertices()
        .filter(|&v| depots.membership[v as usize] == 0)
        .map(|v| STATIONS[v as usize])
        .collect();
    assert_eq!(served_by_harbor, ["Harbor", "Market", "Museum", "Central"]);

    // 6. Network-wide figures.
    let longest = metro
        .diameter_with_path(Some(&minutes), false, true)
        .unwrap();
    assert_eq!(longest.length, 23.0);
    assert!(metro.global_efficiency(Some(&minutes), false).unwrap() > 0.0);
    assert_eq!(metro.diameter().unwrap(), 4.0); // in number of hops
}

// ----------------------------------------------------------------------------
// More values from igraph's own unit tests.

#[test]
fn distances_johnson_like_igraph_unit_test() {
    // tests/unit/igraph_distances_johnson.c: a directed graph with a loop and
    // multi-edges, and some negative weights.
    let g = Graph::from_edges(
        &[
            (0, 1),
            (0, 2),
            (1, 1),
            (1, 2),
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
    let w = [-1., 0., 1., -2., 2., 3., 4., 5., 6.];
    let expected: [&[f64]; 6] = [
        &[0., -1., -3., 1., 6., INF],
        &[1., 0., -2., 2., 7., INF],
        &[3., 2., 0., 4., 9., INF],
        &[INF, INF, INF, 0., 5., INF],
        &[INF, INF, INF, INF, 0., INF],
        &[INF, INF, INF, INF, INF, 0.],
    ];
    let d = g
        .distances_johnson(.., .., Some(&w), NeighborMode::Out)
        .unwrap();
    assert_matrix(&d, &expected);
    // Subsets of rows and columns.
    let sub = g
        .distances_johnson(1..3, 1..3, Some(&w), NeighborMode::Out)
        .unwrap();
    assert_eq!(sub.to_rows(), vec![vec![0., -2.], vec![2., 0.]]);
    assert_eq!(
        g.distances_johnson(0, 2, Some(&w), NeighborMode::Out)
            .unwrap()
            .to_rows(),
        vec![vec![-3.]]
    );
    // IGRAPH_IN gives the transposed matrix, as Bellman–Ford does.
    let d_in = g
        .distances_johnson(.., .., Some(&w), NeighborMode::In)
        .unwrap();
    assert_eq!(d_in, d.transposed());
    assert_eq!(
        d_in,
        g.distances_bellman_ford(.., .., Some(&w), NeighborMode::In)
            .unwrap()
    );
    // A negative loop is a negative cycle; so is any negative edge when
    // directions are ignored.
    let neg_loop = [-4., -3., -2., -1., 0., 1., 2., 3., 4.];
    for (weights, mode) in [(&neg_loop, NeighborMode::Out), (&w, NeighborMode::All)] {
        assert_eq!(
            g.distances_johnson(.., .., Some(weights), mode)
                .unwrap_err()
                .kind(),
            ErrorKind::NegativeCycle
        );
    }
    // Graphs without edges.
    let empty = Graph::new(3, false)
        .distances_johnson(.., .., Some(&[]), NeighborMode::Out)
        .unwrap();
    assert_matrix(&empty, &[&[0., INF, INF], &[INF, 0., INF], &[INF, INF, 0.]]);
    let null = Graph::new(0, true)
        .distances_johnson(.., .., Some(&[]), NeighborMode::Out)
        .unwrap();
    assert_eq!((null.nrow(), null.ncol()), (0, 0));
}

#[test]
fn widest_paths_like_igraph_unit_test() {
    // tests/unit/widest_paths.c, cases 6 to 8.
    // 6. Unreachable vertices (vertex 3 only has an edge *into* 2).
    let g = Graph::from_edges(&[(0, 2), (0, 1), (1, 2), (3, 2)], 4, true).unwrap();
    let w = [1.0, 2.0, 3.0, 5.0];
    let widths = g
        .widest_path_widths_dijkstra(0, .., &w, NeighborMode::Out)
        .unwrap();
    assert_eq!(widths.row(0), vec![INF, 2.0, 2.0, -INF]);
    assert_eq!(
        g.widest_path_widths_floyd_warshall(0, .., &w, NeighborMode::Out)
            .unwrap(),
        widths
    );
    let wp = g.get_widest_paths(0, .., &w, NeighborMode::Out).unwrap();
    assert_eq!(
        wp.vertices,
        vec![vec![0], vec![0, 1], vec![0, 1, 2], vec![]]
    );
    assert_eq!(wp.edges, vec![vec![], vec![1], vec![1, 2], vec![]]);
    assert_eq!(wp.parents, vec![-1, 0, 1, -2]);
    assert_eq!(wp.inbound_edges, vec![-1, 1, 2, -1]);
    assert!(
        !g.get_widest_path(0, 3, &w, NeighborMode::Out)
            .unwrap()
            .exists()
    );

    // 7. Self-loops never widen a path.
    let g = Graph::from_edges(&[(0, 2), (0, 1), (1, 2), (0, 0), (1, 1), (2, 2)], 3, false).unwrap();
    let w = [1.0, 2.0, 3.0, 5.0, 5.0, 1.0];
    let all = g
        .widest_path_widths_dijkstra(.., .., &w, NeighborMode::Out)
        .unwrap();
    assert_matrix(
        &all,
        &[&[INF, 2.0, 2.0], &[2.0, INF, 3.0], &[2.0, 3.0, INF]],
    );
    assert_eq!(
        g.widest_path_widths_floyd_warshall(.., .., &w, NeighborMode::Out)
            .unwrap(),
        all
    );
    let wp = g.get_widest_paths(0, .., &w, NeighborMode::Out).unwrap();
    assert_eq!(wp.parents, vec![-1, 0, 1]);
    assert_eq!(wp.inbound_edges, vec![-1, 1, 2]);
    let p = g.get_widest_path(0, 2, &w, NeighborMode::Out).unwrap();
    assert_eq!((p.vertices, p.edges), (vec![0, 1, 2], vec![1, 2]));

    // 8. Among parallel edges, the widest one is used.
    let g = Graph::from_edges(&[(0, 1); 8], 2, false).unwrap();
    let w = [2.0, 2.0, 2.0, 10.0, 2.0, 2.0, 2.0, 2.0];
    let p = g.get_widest_path(0, 1, &w, NeighborMode::Out).unwrap();
    assert_eq!((p.vertices, p.edges), (vec![0, 1], vec![3]));
    assert_eq!(
        g.widest_path_widths_dijkstra(0, .., &w, NeighborMode::Out)
            .unwrap()
            .row(0),
        vec![INF, 10.0]
    );
}

#[test]
fn graph_center_like_igraph_unit_test() {
    // tests/unit/igraph_graph_center.c (unweighted, then with weights 2, 3, ...).
    let weights: Vec<f64> = (2..13).map(f64::from).collect();
    let center = |g: &Graph, weighted: bool, mode: NeighborMode| {
        let w = weighted.then(|| &weights[..g.ecount()]);
        (g.graph_center(w, mode).unwrap(), g.radius(w, mode).unwrap())
    };
    let null = Graph::new(0, false);
    let (c, r) = center(&null, false, NeighborMode::Out);
    assert!(c.is_empty() && r.is_nan());
    let four = Graph::new(4, false);
    assert_eq!(
        center(&four, false, NeighborMode::Out),
        (vec![0, 1, 2, 3], 0.0)
    );

    let isolated = Graph::from_edges(&[(0, 2)], 3, false).unwrap();
    assert_eq!(center(&isolated, false, NeighborMode::Out), (vec![1], 0.0));
    assert_eq!(center(&isolated, true, NeighborMode::Out), (vec![1], 0.0));
    let p5 = Graph::path_graph(5, false, false).unwrap();
    assert_eq!(center(&p5, false, NeighborMode::Out), (vec![2], 2.0));
    assert_eq!(center(&p5, true, NeighborMode::Out), (vec![2, 3], 9.0));
    let small =
        Graph::from_edges(&[(0, 1), (1, 2), (0, 2), (3, 0), (4, 1), (2, 5)], 6, false).unwrap();
    assert_eq!(
        center(&small, false, NeighborMode::Out),
        (vec![0, 1, 2], 2.0)
    );
    assert_eq!(center(&small, true, NeighborMode::Out), (vec![2], 9.0));
    let dp5 = Graph::path_graph(5, true, false).unwrap();
    assert_eq!(center(&dp5, false, NeighborMode::Out), (vec![4], 0.0));
    assert_eq!(center(&dp5, true, NeighborMode::Out), (vec![4], 0.0));

    let star = Graph::star(10, StarMode::Undirected, 0).unwrap();
    assert_eq!(center(&star, false, NeighborMode::Out), (vec![0], 1.0));
    assert_eq!(center(&star, true, NeighborMode::Out), (vec![0], 10.0));
    let out_star = Graph::star(10, StarMode::Out, 0).unwrap();
    let leaves: Vec<i64> = (1..10).collect();
    assert_eq!(
        center(&out_star, false, NeighborMode::Out),
        (leaves.clone(), 0.0)
    );
    assert_eq!(center(&out_star, true, NeighborMode::Out), (leaves, 0.0));
    assert_eq!(center(&out_star, false, NeighborMode::All), (vec![0], 1.0));
    assert_eq!(center(&out_star, true, NeighborMode::All), (vec![0], 10.0));
    let in_star = Graph::star(10, StarMode::In, 0).unwrap();
    assert_eq!(center(&in_star, false, NeighborMode::Out), (vec![0], 0.0));

    let dc5 = Graph::cycle_graph(5, true, false).unwrap();
    assert_eq!(center(&dc5, true, NeighborMode::Out), (vec![0], 14.0));
    assert_eq!(center(&dc5, true, NeighborMode::In), (vec![4], 14.0));
    // The center vertices realize the radius.
    for (g, mode) in [(&p5, NeighborMode::All), (&dc5, NeighborMode::Out)] {
        let (c, r) = center(g, true, mode);
        let w = &weights[..g.ecount()];
        assert!(
            g.eccentricity(c, Some(w), mode)
                .unwrap()
                .iter()
                .all(|&e| e == r)
        );
    }
}

// ----------------------------------------------------------------------------
// Cross-checks with other modules.

#[test]
fn famous_zachary_is_the_karate_club() {
    // `Graph::famous` (constructors) ships the same network as the test helper.
    let zachary = Graph::famous("Zachary").unwrap();
    let k = karate();
    assert_eq!((zachary.vcount(), zachary.ecount()), (34, 78));
    let dz = zachary.distances(.., .., None, NeighborMode::All).unwrap();
    assert_eq!(dz, k.distances(.., .., None, NeighborMode::All).unwrap());
    assert_eq!(zachary.diameter().unwrap(), 5.0);
    assert_eq!(zachary.path_length_hist(false).unwrap().counts, {
        k.path_length_hist(false).unwrap().counts
    });
}

#[test]
fn distance_based_centralities_agree_with_distances() {
    let g = Graph::famous("Zachary").unwrap();
    let n = g.vcount();
    let d = g.distances(.., .., None, NeighborMode::All).unwrap();

    // Closeness: (n - 1) / (sum of distances), when normalized.
    let closeness = g.closeness(.., NeighborMode::All, None, true).unwrap();
    for (v, c) in closeness.iter().enumerate() {
        let total: f64 = d.row(v).iter().sum();
        assert_close(*c, (n - 1) as f64 / total, 1e-12);
    }

    // Global efficiency is the mean normalized harmonic centrality.
    let harmonic = g
        .harmonic_centrality(.., NeighborMode::All, None, true)
        .unwrap();
    assert_close(
        g.global_efficiency(None, false).unwrap(),
        harmonic.iter().sum::<f64>() / n as f64,
        1e-12,
    );

    // Betweenness counts the fraction of geodesics passing through a vertex:
    // rebuild it from the enumeration of all shortest paths.
    let mut betweenness = vec![0.0; n];
    for s in g.vertices() {
        let all = g
            .get_all_shortest_paths(s, .., None, NeighborMode::All)
            .unwrap();
        for p in &all.vertices {
            let t = *p.last().unwrap() as usize;
            // The trivial path [s] has no interior vertex.
            for &v in p.iter().skip(1).take(p.len().saturating_sub(2)) {
                betweenness[v as usize] += 1.0 / all.nrgeo[t] as f64;
            }
        }
    }
    let expected = g.betweenness(None, .., false, false).unwrap();
    for (ours, theirs) in betweenness.iter().zip(&expected) {
        // Each unordered pair was counted from both ends.
        assert_close(ours / 2.0, *theirs, 1e-9);
    }
}

#[test]
fn seeded_random_walks_are_reproducible_in_parallel_threads() {
    // Every thread has its own default generator: seeding it in one thread
    // neither affects nor is affected by the other threads.
    let walk = || {
        let g = Graph::famous("Zachary").unwrap();
        rng::seed(2024).unwrap();
        let mut walks = vec![];
        for _ in 0..3 {
            walks.push(
                g.random_walk(0, 200, None, NeighborMode::All, RandomWalkStuck::Error)
                    .unwrap(),
            );
            // Interleave some unrelated random draws.
            let _ = rng::integer(0, 1000);
        }
        walks
    };
    let reference = walk();
    let handles: Vec<_> = (0..8).map(|_| std::thread::spawn(walk)).collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), reference);
    }
}

#[test]
fn spanner_as_a_subgraph() {
    // With `subgraph_from_edges` (operators) the spanner is a graph again.
    rng::seed(11).unwrap();
    let g = Graph::famous("Zachary").unwrap();
    let kept = g.spanner(3.0, None).unwrap();
    let h = g.subgraph_from_edges(kept.clone(), false).unwrap();
    assert_eq!((h.vcount(), h.ecount()), (g.vcount(), kept.len()));
    let dg = g.distances(.., .., None, NeighborMode::All).unwrap();
    let dh = h.distances(.., .., None, NeighborMode::All).unwrap();
    for (a, b) in dg.as_slice().iter().zip(dh.as_slice()) {
        assert!(*b >= *a && *b <= 3.0 * a);
    }
    // A 1-spanner must keep every shortest path, i.e. every edge of a simple graph.
    assert_eq!(g.spanner(1.0, None).unwrap().len(), g.ecount());
}

/// Regression test: an igraph call that fails inside the A* heuristic must
/// not free the temporaries of the running search.
#[test]
fn astar_heuristic_survives_failing_nested_calls() {
    let g = lattice(5, false);
    let before = unsafe { igraph::IGRAPH_FINALLY_STACK_SIZE() };
    let mut calls = 0;
    let p = g
        .get_shortest_path_astar(0, 24, None, NeighborMode::All, |_, _| {
            let err = g
                .distances(.., vec![0, 99], None, NeighborMode::All)
                .unwrap_err();
            assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
            calls += 1;
            0.0
        })
        .unwrap();
    assert!(calls > 0);
    assert_eq!(p.edges.len(), 8);
    assert_eq!(unsafe { igraph::IGRAPH_FINALLY_STACK_SIZE() }, before);
}

/// Regression test: infinite weights, or finite ones whose sum overflows,
/// made igraph 1.0.1 loop forever while sampling the next step.
#[test]
fn random_walk_rejects_infinite_weight_totals() {
    let tri = cycle(3);
    let stuck = RandomWalkStuck::Error;
    for w in [
        [f64::INFINITY, 1.0, 1.0],
        [f64::NEG_INFINITY, 1.0, 1.0],
        [f64::MAX, f64::MAX, 1.0],
        [f64::MAX, 1.0, 1.0], // conservative: above f64::MAX / 2 at vertices 0 and 1
    ] {
        for start in 0..3 {
            let err = tri
                .random_walk(start, 3, Some(&w), NeighborMode::All, stuck)
                .unwrap_err();
            assert_eq!(err.kind(), ErrorKind::InvalidValue, "{w:?}");
        }
    }
    // Large but safe weights still work.
    rng::seed(3).unwrap();
    let walk = tri
        .random_walk(0, 10, Some(&[1e300, 1e300, 1.0]), NeighborMode::All, stuck)
        .unwrap();
    assert_eq!(walk.edges.len(), 10);
    // Negative and NaN weights are still reported (by igraph or by Rust).
    for w in [[-1.0, 1.0, 1.0], [f64::NAN, 1.0, 1.0]] {
        let err = tri
            .random_walk(0, 3, Some(&w), NeighborMode::All, stuck)
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidValue);
    }
}

#[test]
fn random_walk_weight_check_is_per_vertex() {
    let stuck = RandomWalkStuck::Error;
    // The grand total overflows (1000 * 1e306 = inf), yet every vertex of the
    // ring only sees 2e306: the walk is fine.
    let ring = cycle(1000);
    let w = vec![1e306; 1000];
    assert!(!w.iter().sum::<f64>().is_finite());
    rng::seed(5).unwrap();
    let walk = ring
        .random_walk(0, 50, Some(&w), NeighborMode::All, stuck)
        .unwrap();
    assert_eq!(walk.edges.len(), 50);
    // A loop counts twice at its vertex in All mode (igraph lists it twice):
    // a 0.3 * MAX loop weighs 0.6 * MAX, over the MAX / 2 limit, while a
    // 0.2 * MAX loop is fine. Vertex 0 carries the loop; the walk starts at 1.
    let big = 0.3 * f64::MAX;
    let looped = Graph::from_edges(&[(0, 0), (0, 1)], 2, false).unwrap();
    let err = looped
        .random_walk(1, 3, Some(&[big, 1.0]), NeighborMode::All, stuck)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    rng::seed(5).unwrap();
    let ok = looped
        .random_walk(1, 3, Some(&[0.2 * f64::MAX, 1.0]), NeighborMode::All, stuck)
        .unwrap();
    assert_eq!(ok.edges.len(), 3);
    // Directed Out mode only sums the out-edges: 0 -> 1 and 2 -> 1, each
    // 0.4 * MAX, give vertex 1 an in-total of 0.8 * MAX, which Out ignores.
    let star = Graph::from_edges(&[(0, 1), (2, 1), (1, 0), (1, 2)], 3, true).unwrap();
    let w = [0.4 * f64::MAX, 0.4 * f64::MAX, 1.0, 1.0];
    rng::seed(5).unwrap();
    let ok = star
        .random_walk(0, 6, Some(&w), NeighborMode::Out, stuck)
        .unwrap();
    assert_eq!(ok.edges.len(), 6);
    let err = star
        .random_walk(0, 6, Some(&w), NeighborMode::In, stuck)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

#[test]
fn voronoi_partition_type_name() {
    let p = path(6);
    let v: VoronoiPartition = p
        .voronoi(&[0, 5], None, NeighborMode::All, VoronoiTiebreaker::First)
        .unwrap();
    assert_eq!(v.membership, vec![0, 0, 0, 1, 1, 1]);
}

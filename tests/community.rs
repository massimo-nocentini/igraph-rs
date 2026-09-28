//! Integration tests for the `community` module (igraph_community.h, igraph_hrg.h).
//!
//! Every thread has its own igraph random number generator, so seeded tests
//! can run in parallel without interfering with each other.

mod common;

use common::assert_close;
use igraph::community::*;
use igraph::prelude::*;
use std::ops::ControlFlow;

/// Zachary's karate club network (34 vertices, 78 edges).
fn karate() -> Graph {
    Graph::famous("Zachary").unwrap()
}

/// The complete graph on `n` vertices.
fn complete(n: usize) -> Graph {
    Graph::full(n, false, false).unwrap()
}

/// The undirected cycle on `n` vertices.
fn cycle(n: usize) -> Graph {
    Graph::ring(n, false, false, true).unwrap()
}

/// The undirected path on `n` vertices.
fn path(n: usize) -> Graph {
    Graph::ring(n, false, false, false).unwrap()
}

/// The two factions of Zachary's karate club after the split (0-based ids):
/// Mr. Hi's club (0) and the officer's club (1).
fn karate_factions() -> Vec<i64> {
    const OFFICER: [i64; 18] = [
        8, 9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    ];
    (0..34).map(|v| i64::from(OFFICER.contains(&v))).collect()
}

/// Two triangles {0,1,2} and {3,4,5} joined by the edge 2-3.
fn two_triangles() -> Graph {
    Graph::from_edges(
        &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)],
        6,
        false,
    )
    .unwrap()
}

/// `k` cliques of size `s`, the i-th joined to the next one by a single edge.
fn ring_of_cliques(k: i64, s: i64) -> Graph {
    let mut edges = vec![];
    for c in 0..k {
        for i in 0..s {
            for j in i + 1..s {
                edges.push((c * s + i, c * s + j));
            }
        }
        edges.push((c * s, ((c + 1) % k) * s + 1));
    }
    Graph::from_edges(&edges, (k * s) as usize, false).unwrap()
}

fn same_partition(a: &[i64], b: &[i64]) -> bool {
    split_join_distance(a, b).unwrap() == (0, 0)
}

// ---------------------------------------------------------------------------
// Use case: the karate club split
// ---------------------------------------------------------------------------

/// Zachary observed the karate club splitting in two after a conflict
/// between the instructor (vertex 0) and the administrator (vertex 33).
/// Can community detection, looking only at who interacts with whom,
/// predict who went with whom?
#[test]
fn story_karate_club_factions() {
    let g = karate();
    let factions = karate_factions();

    // The real split is itself a good partition of the network...
    let q_split = g.modularity(&factions, None, 1.0, false).unwrap();
    assert_close(q_split, 0.3715, 1e-4);

    // ... but modularity optimizers find even better ones, with 4 groups.
    rng::seed(42).unwrap();
    let leiden = g
        .community_leiden_simple(
            None,
            LeidenObjective::Modularity,
            &LeidenOptions::default().with_iterations(None),
        )
        .unwrap();
    assert!(leiden.quality > 0.4, "Leiden modularity {}", leiden.quality);
    assert_close(leiden.quality, 0.4198, 1e-4); // the known maximum
    assert_eq!(leiden.nb_clusters, 4);
    assert_close(
        g.modularity(&leiden.membership, None, 1.0, false).unwrap(),
        leiden.quality,
        1e-12,
    );

    // Louvain is randomized: the best of a few runs exceeds 0.4 as well.
    let best = (0..10)
        .map(|seed| {
            rng::seed(seed).unwrap();
            g.community_multilevel(None, 1.0).unwrap()
        })
        .max_by(|a, b| a.modularity().total_cmp(&b.modularity()))
        .unwrap();
    assert!(best.modularity() > 0.41);

    // Each of Leiden's four groups lies entirely within one faction: the
    // detected partition is a refinement of the real split.
    let (d12, d21) = split_join_distance(&leiden.membership, &factions).unwrap();
    assert_eq!(d12, 0);
    assert!(d21 > 0);
    let nmi = compare_communities(&leiden.membership, &factions, CommunityComparison::Nmi).unwrap();
    assert!(nmi > 0.6, "NMI {nmi}");

    // Newman's leading eigenvector method, stopped after its first split,
    // predicts the two factions *exactly*.
    let first_split = g
        .community_leading_eigenvector(None, Some(1), None)
        .unwrap();
    assert_eq!(first_split.num_communities(), 2);
    let vi =
        compare_communities(&first_split.membership, &factions, CommunityComparison::Vi).unwrap();
    assert_eq!(vi, 0.0);
    assert_eq!(
        compare_communities(
            &first_split.membership,
            &factions,
            CommunityComparison::Rand
        )
        .unwrap(),
        1.0
    );
    // The instructor and the administrator end up on different sides.
    assert_ne!(first_split.membership[0], first_split.membership[33]);

    // Exact optimization (when igraph has GLPK) confirms Leiden hit the optimum.
    match g.community_optimal_modularity(None, 1.0) {
        Ok(opt) => {
            assert_close(opt.modularity, leiden.quality, 1e-9);
            assert_eq!(opt.num_communities(), 4);
        }
        Err(e) => assert_eq!(e.kind(), ErrorKind::Unimplemented),
    }
}

/// Plant a partition, then check that every method recovers it.
#[test]
fn story_planted_partition_is_recovered_by_all_methods() {
    let g = ring_of_cliques(6, 5);
    let planted: Vec<i64> = (0..30).map(|v| v / 5).collect();
    // Every randomized method is seeded, so that the story is reproducible.
    let seeded = |f: &dyn Fn() -> Vec<i64>| {
        rng::seed(2024).unwrap();
        f()
    };
    let results: Vec<(&str, Vec<i64>)> = vec![
        (
            "multilevel",
            seeded(&|| g.community_multilevel(None, 1.0).unwrap().membership),
        ),
        (
            "leiden",
            seeded(&|| {
                g.community_leiden_simple(
                    None,
                    LeidenObjective::Modularity,
                    &LeidenOptions::default(),
                )
                .unwrap()
                .membership
            }),
        ),
        (
            "fastgreedy",
            g.community_fastgreedy(None).unwrap().membership,
        ),
        (
            "walktrap",
            g.community_walktrap(None, 4).unwrap().membership,
        ),
        (
            "edge betweenness",
            g.community_edge_betweenness(None, None, false)
                .unwrap()
                .membership,
        ),
        (
            "spinglass",
            seeded(&|| {
                g.community_spinglass(None, &SpinglassOptions::default())
                    .unwrap()
                    .membership
            }),
        ),
        (
            "infomap",
            seeded(&|| {
                g.community_infomap(None, None, &InfomapOptions::default())
                    .unwrap()
                    .membership
            }),
        ),
    ];
    for (name, membership) in results {
        let nmi = compare_communities(&membership, &planted, CommunityComparison::Nmi).unwrap();
        assert!(nmi > 0.99, "{name} found {membership:?} (NMI {nmi})");
    }
    // Fluid communities, told that there are 6 groups, finds a good partition
    // (though not always exactly the planted one).
    let fluid = seeded(&|| g.community_fluid_communities(6).unwrap());
    assert!(g.modularity(&fluid, None, 1.0, false).unwrap() > 0.5);
    // The spectral method is less accurate on this graph (it splits some
    // cliques), yet it stays close to the planted partition.
    let le = g.community_leading_eigenvector(None, None, None).unwrap();
    let nmi = compare_communities(&le.membership, &planted, CommunityComparison::Nmi).unwrap();
    assert!(nmi > 0.85, "leading eigenvector NMI {nmi}");
    assert!(le.modularity > 0.6);
}

// ---------------------------------------------------------------------------
// Modularity
// ---------------------------------------------------------------------------

#[test]
fn modularity_of_two_triangles() {
    let g = two_triangles();
    let m = [0, 0, 0, 1, 1, 1];
    // Q = 2 * (3/7 - (7/14)^2) = 5/14
    assert_close(
        g.modularity(&m, None, 1.0, true).unwrap(),
        5.0 / 14.0,
        1e-12,
    );
    // Everything in one community gives Q = 1 - γ.
    assert_close(g.modularity(&[0; 6], None, 1.0, true).unwrap(), 0.0, 1e-12);
    assert_close(g.modularity(&[0; 6], None, 0.5, true).unwrap(), 0.5, 1e-12);
    // Labels need not be contiguous nor small.
    assert_close(
        g.modularity(&[7, 7, 7, 42, 42, 42], None, 1.0, true)
            .unwrap(),
        5.0 / 14.0,
        1e-12,
    );
    // Heavier bridge => lower modularity.
    let w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0];
    assert!(g.modularity(&m, Some(&w), 1.0, true).unwrap() < 5.0 / 14.0);
}

#[test]
fn modularity_edge_cases_and_errors() {
    // Graphs without edges have undefined (NaN) modularity.
    let empty = Graph::new(3, false);
    assert!(
        empty
            .modularity(&[0, 1, 2], None, 1.0, false)
            .unwrap()
            .is_nan()
    );
    let g = two_triangles();
    assert_eq!(
        g.modularity(&[0, 1], None, 1.0, false).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.modularity(&[0; 6], Some(&[1.0]), 1.0, false)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.modularity(&[0; 6], None, -1.0, false).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn modularity_matrix_matches_igraph_unit_test() {
    // Triangle and a point with a self-loop.
    let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2), (3, 3)], 4, false).unwrap();
    let b = g.modularity_matrix(None, 1.0, false).unwrap();
    let expected = [
        [-0.5, 0.5, 0.5, -0.5],
        [0.5, -0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5, -0.5],
        [-0.5, -0.5, -0.5, 1.5],
    ];
    for (i, row) in expected.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            assert_close(b[(i, j)], x, 1e-12);
        }
    }
    // Directed version.
    let d = Graph::from_edges(&[(0, 1), (0, 2), (1, 2), (3, 3)], 4, true).unwrap();
    let b = d.modularity_matrix(None, 1.0, true).unwrap();
    let expected = [
        [0.0, 0.5, 0.0, -0.5],
        [0.0, -0.25, 0.5, -0.25],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, -0.25, -0.5, 0.75],
    ];
    for (i, row) in expected.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            assert_close(b[(i, j)], x, 1e-12);
        }
    }
    // Weighted triangle.
    let t = Graph::from_edges(&[(0, 1), (0, 2), (1, 2)], 3, false).unwrap();
    let b = t
        .modularity_matrix(Some(&[0.0, 1.0, 2.0]), 1.0, false)
        .unwrap();
    assert_close(b[(0, 0)], -1.0 / 6.0, 1e-12);
    assert_close(b[(2, 2)], -1.5, 1e-12);
}

#[test]
fn modularity_equals_normalized_sum_of_modularity_matrix() {
    let g = karate();
    let factions = karate_factions();
    let b = g.modularity_matrix(None, 1.0, false).unwrap();
    let m2 = 2.0 * g.ecount() as f64;
    let mut sum = 0.0;
    for i in 0..34 {
        for j in 0..34 {
            if factions[i] == factions[j] {
                sum += b[(i, j)];
            }
        }
    }
    assert_close(
        sum / m2,
        g.modularity(&factions, None, 1.0, false).unwrap(),
        1e-12,
    );
    // Rows of B sum to zero (at resolution 1).
    for i in 0..34 {
        assert_close((0..34).map(|j| b[(i, j)]).sum::<f64>(), 0.0, 1e-9);
    }
}

// ---------------------------------------------------------------------------
// Louvain and Leiden
// ---------------------------------------------------------------------------

#[test]
fn multilevel_on_blondel_example() {
    // The unweighted test graph from the paper of Blondel et al.
    let edges = [
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 2),
        (1, 4),
        (1, 7),
        (2, 4),
        (2, 5),
        (2, 6),
        (3, 7),
        (4, 10),
        (5, 7),
        (5, 11),
        (6, 7),
        (6, 11),
        (8, 9),
        (8, 10),
        (8, 11),
        (8, 14),
        (8, 15),
        (9, 12),
        (9, 14),
        (10, 11),
        (10, 12),
        (10, 13),
        (10, 14),
        (11, 13),
    ];
    let g = Graph::from_edges(&edges, 16, false).unwrap();
    rng::seed(42).unwrap();
    let res = g.community_multilevel(None, 1.0).unwrap();
    assert_eq!(res.levels.len(), res.modularities.len());
    assert!(!res.levels.is_empty());
    assert_eq!(res.levels.last().unwrap(), &res.membership);
    // Modularity increases level after level, and matches `modularity()`.
    assert!(res.modularities.windows(2).all(|w| w[0] <= w[1]));
    for (level, &q) in res.levels.iter().zip(&res.modularities) {
        assert_close(g.modularity(level, None, 1.0, false).unwrap(), q, 1e-12);
    }
    // igraph's example output: two halves 0..8 and 8..16, Q = 0.392219.
    assert_close(res.modularity(), 0.392219, 1e-6);
    assert_eq!(
        res.membership,
        vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    );
    // Higher resolution gives more, smaller communities.
    let fine = g.community_multilevel(None, 1.5).unwrap();
    assert!(fine.num_communities() > res.num_communities());
    assert_eq!(fine.sizes().iter().sum::<usize>(), 16);
}

#[test]
fn multilevel_ring_of_cliques_and_errors() {
    let g = ring_of_cliques(30, 5);
    rng::seed(42).unwrap();
    let res = g.community_multilevel(None, 1.0).unwrap();
    assert!(res.modularity() > 0.87);
    for c in res.communities() {
        // Every community is a union of whole cliques.
        assert_eq!(c.len() % 5, 0);
    }
    let directed = Graph::from_edges(&[(0, 1)], 2, true).unwrap();
    assert!(directed.community_multilevel(None, 1.0).is_err());
    assert_eq!(
        g.community_multilevel(Some(&[1.0]), 1.0)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn leiden_reproduces_igraph_example() {
    let mut edges = vec![];
    for base in [0, 5] {
        for i in 0..5 {
            for j in i + 1..5 {
                edges.push((base + i, base + j));
            }
        }
    }
    edges.push((0, 5));
    let g = Graph::from_edges(&edges, 10, false).unwrap();
    rng::seed(0).unwrap();
    let cpm = LeidenOptions::default()
        .with_resolution(0.05)
        .with_iterations(Some(1));
    let res = g.community_leiden(None, None, None, &cpm).unwrap();
    assert_eq!(res.nb_clusters, 2);
    assert_close(res.quality, 0.8929, 1e-4);
    assert_eq!(res.membership, vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

    // Continue from the previous partition for 10 more iterations.
    let more = LeidenOptions::default()
        .with_resolution(0.05)
        .with_iterations(Some(10))
        .with_initial(&res.membership);
    let res2 = g.community_leiden(None, None, None, &more).unwrap();
    assert_close(res2.quality, res.quality, 1e-12);

    // Degrees as vertex weights and γ = 1/(2m): modularity.
    let degrees: Vec<f64> = g
        .degree(.., NeighborMode::All, Loops::Twice)
        .unwrap()
        .into_iter()
        .map(|d| d as f64)
        .collect();
    let opts = LeidenOptions::default()
        .with_resolution(1.0 / (2.0 * g.ecount() as f64))
        .with_iterations(None);
    let res = g
        .community_leiden(None, Some(&degrees), None, &opts)
        .unwrap();
    assert_close(res.quality, 0.4524, 1e-4);
    // The simple interface computes the same.
    let simple = g
        .community_leiden_simple(None, LeidenObjective::Modularity, &LeidenOptions::default())
        .unwrap();
    assert_close(simple.quality, res.quality, 1e-12);
}

#[test]
fn leiden_directed_example_and_objectives() {
    let g = Graph::from_edges(
        &[
            (0, 1),
            (0, 3),
            (0, 5),
            (1, 3),
            (1, 4),
            (2, 3),
            (4, 1),
            (5, 2),
            (5, 4),
        ],
        6,
        true,
    )
    .unwrap();
    let outdeg: Vec<f64> = g
        .degree(.., NeighborMode::Out, Loops::Twice)
        .unwrap()
        .iter()
        .map(|&d| d as f64)
        .collect();
    let indeg: Vec<f64> = g
        .degree(.., NeighborMode::In, Loops::Twice)
        .unwrap()
        .iter()
        .map(|&d| d as f64)
        .collect();
    rng::seed(0).unwrap();
    let opts = LeidenOptions::default().with_resolution(1.0 / g.ecount() as f64);
    let res = g
        .community_leiden(None, Some(&outdeg), Some(&indeg), &opts)
        .unwrap();
    assert_eq!(res.nb_clusters, 3);
    assert_close(res.quality, 0.1852, 1e-4);
    assert_close(
        g.modularity(&res.membership, None, 1.0, true).unwrap(),
        res.quality,
        1e-9,
    );

    // CPM with a huge resolution splits everything into singletons.
    let k = karate();
    let res = k
        .community_leiden_simple(
            None,
            LeidenObjective::Cpm,
            &LeidenOptions::default().with_resolution(10.0),
        )
        .unwrap();
    assert_eq!(res.nb_clusters, 34);
    // ER objective with zero resolution merges each component into one cluster.
    let res = k
        .community_leiden_simple(
            None,
            LeidenObjective::ErdosRenyi,
            &LeidenOptions::default().with_resolution(0.0),
        )
        .unwrap();
    assert_eq!(res.nb_clusters, 1);
    // Wrong initial membership length.
    let bad = LeidenOptions::default().with_initial(&[0, 1]);
    assert_eq!(
        k.community_leiden_simple(None, LeidenObjective::Cpm, &bad)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

// ---------------------------------------------------------------------------
// Hierarchical methods and dendrograms
// ---------------------------------------------------------------------------

#[test]
fn fastgreedy_example_and_dendrogram_cuts() {
    let g = Graph::from_edges(
        &[
            (0, 1),
            (1, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 4),
            (3, 5),
            (4, 5),
        ],
        6,
        false,
    )
    .unwrap();
    let w = [10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let d = g.community_fastgreedy(Some(&w)).unwrap();
    assert_eq!(d.merges, vec![(1, 0), (2, 6), (3, 4), (8, 5), (9, 7)]);
    assert_eq!(d.modularity.len(), d.merges.len() + 1);
    assert_eq!(d.num_vertices, 6);

    // The best membership is the cut at the modularity maximum.
    let best_steps = d
        .modularity
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap();
    let cut = d.cut(6 - best_steps).unwrap();
    assert!(same_partition(&cut, &d.membership));
    assert_close(
        g.modularity(&cut, Some(&w), 1.0, false).unwrap(),
        d.max_modularity(),
        1e-12,
    );
    // Extreme cuts.
    assert_eq!(d.cut(6).unwrap(), vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(d.cut(1).unwrap(), vec![0; 6]);
    assert!(d.cut(7).is_err());

    // Multigraphs are refused.
    let multi = Graph::from_edges(&[(0, 1), (0, 1)], 2, false).unwrap();
    assert!(multi.community_fastgreedy(None).is_err());
}

#[test]
fn walktrap_matches_igraph_unit_test() {
    let triangle = cycle(3);
    let d = triangle.community_walktrap(None, 4).unwrap();
    assert_eq!(d.merges, vec![(1, 2), (0, 3)]);
    assert_eq!(d.modularity.len(), 3);
    assert_close(d.modularity[0], -1.0 / 3.0, 1e-6);
    assert_close(d.modularity[1], -2.0 / 9.0, 1e-6);
    assert_close(d.modularity[2], 0.0, 1e-12);
    assert_eq!(d.membership, vec![0, 0, 0]);

    // Isolated vertices are allowed.
    let mut g = two_triangles();
    g.add_vertices(2).unwrap();
    let d = g.community_walktrap(None, 4).unwrap();
    assert_eq!(d.membership.len(), 8);
    assert!(same_partition(&d.membership[..6], &[0, 0, 0, 1, 1, 1]));
}

#[test]
fn edge_betweenness_and_eb_get_merges_agree() {
    let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (1, 4)], 5, false).unwrap();
    let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let res = g.community_edge_betweenness(Some(&w), None, false).unwrap();
    // Values from igraph's example.
    assert_eq!(res.removed_edges, vec![0, 1, 3, 4, 2, 5]);
    assert_eq!(res.edge_betweenness, vec![2.0, 3.5, 6.0, 2.0, 1.0, 1.0]);
    assert_eq!(res.bridges, vec![5, 4, 3, 2]);
    assert_eq!(res.merges.len(), 4);

    // Replaying the same removal order gives the same dendrogram.
    let replay = g
        .community_eb_get_merges(false, &res.removed_edges, Some(&w))
        .unwrap();
    assert_eq!(replay.dendrogram.merges, res.merges);
    assert_eq!(replay.bridges, res.bridges);
    assert_eq!(replay.dendrogram.modularity, res.modularity);
    assert_eq!(replay.dendrogram.membership, res.membership);

    // Invalid edge ids are reported.
    assert_eq!(
        g.community_eb_get_merges(false, &[0, 1, 2, 3, 4, 99], None)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidEdgeId
    );
}

#[test]
fn edge_betweenness_cuts_bridge_first() {
    let g = two_triangles();
    let res = g.community_edge_betweenness(None, None, false).unwrap();
    // The bridge 2-3 (edge 6) carries all 9 inter-triangle shortest paths.
    assert_eq!(res.removed_edges[0], 6);
    assert_eq!(res.edge_betweenness[0], 9.0);
    assert!(same_partition(&res.membership, &[0, 0, 0, 1, 1, 1]));
    assert_eq!(res.num_communities(), 2);
    // With lengths making the bridge very long, nothing changes (it is still
    // the only connection) but other betweenness values are computed with lengths.
    let lengths = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0];
    let res2 = g
        .community_edge_betweenness(None, Some(&lengths), false)
        .unwrap();
    assert_eq!(res2.removed_edges[0], 6);
}

#[test]
fn community_to_membership_and_errors() {
    let merges = [(0, 1), (2, 3), (4, 5)];
    let (m, sizes) = community_to_membership(&merges, 4, 0).unwrap();
    assert_eq!(m, vec![0, 1, 2, 3]);
    assert_eq!(sizes, vec![1, 1, 1, 1]);
    let (m, sizes) = community_to_membership(&merges, 4, 3).unwrap();
    assert_eq!(m, vec![0; 4]);
    assert_eq!(sizes, vec![4]);
    let (m, _) = community_to_membership(&merges, 4, 1).unwrap();
    assert!(same_partition(&m, &[0, 0, 1, 2]));
    // More steps than merges.
    assert_eq!(
        community_to_membership(&merges, 4, 4).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn le_community_to_membership_matches_igraph_unit_test() {
    let (m, s) = le_community_to_membership(&[], 0, &[0]).unwrap();
    assert_eq!((m, s), (vec![0], vec![1]));
    let (m, s) = le_community_to_membership(&[(1, 3)], 1, &[0, 1, 2, 3, 4]).unwrap();
    assert_eq!((m, s), (vec![1, 0, 2, 0, 3], vec![2, 1, 1, 1]));
    let (m, s) =
        le_community_to_membership(&[(0, 3), (2, 5)], 2, &[0, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5])
            .unwrap();
    assert_eq!(m, vec![1, 1, 2, 0, 0, 1, 1, 3, 3, 0, 0, 0]);
    assert_eq!(s, vec![5, 4, 1, 2]);

    let err = |merges: &[(i64, i64)], steps, memb: &[i64]| {
        le_community_to_membership(merges, steps, memb)
            .unwrap_err()
            .kind()
    };
    assert_eq!(
        err(&[(1, 3), (1, 4)], 2, &[0, 1, 2, 3, 4]),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        err(&[(1, 2), (3, 4)], 2, &[-1, 0, 1, 2, 3]),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        err(&[(1, 2), (3, 4)], 2, &[0, 0, 2, 3, 4]),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        err(&[(1, 2), (3, 4)], 20, &[0, 1, 2, 3, 4]),
        ErrorKind::InvalidValue
    );
}

// ---------------------------------------------------------------------------
// Leading eigenvector
// ---------------------------------------------------------------------------

#[test]
fn leading_eigenvector_matches_igraph_example() {
    let g = karate();
    let one = g
        .community_leading_eigenvector(None, Some(1), None)
        .unwrap();
    assert_eq!(one.merges, vec![(0, 1)]);
    assert_eq!(
        one.membership,
        vec![
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1
        ]
    );

    let all = g.community_leading_eigenvector(None, None, None).unwrap();
    assert_eq!(all.merges, vec![(1, 3), (0, 2), (5, 4)]);
    assert_eq!(
        all.membership,
        vec![
            0, 2, 2, 2, 0, 0, 0, 2, 1, 1, 0, 0, 2, 2, 1, 1, 0, 2, 1, 2, 1, 2, 1, 3, 3, 3, 1, 3, 3,
            1, 1, 3, 1, 1
        ]
    );
    assert_close(all.modularity, 0.3934, 1e-4);
    assert_close(
        g.modularity(&all.membership, None, 1.0, false).unwrap(),
        all.modularity,
        1e-12,
    );
    assert_eq!(all.history[0], LeadingEigenvectorEvent::StartFull);
    let splits = all
        .history
        .iter()
        .filter(|e| matches!(e, LeadingEigenvectorEvent::Split { .. }))
        .count();
    assert_eq!(splits, all.num_communities() - 1);
    assert_eq!(all.eigenvalues.len(), all.eigenvectors.len());

    // Undoing all the splits with le_community_to_membership.
    let (merged, sizes) = le_community_to_membership(&all.merges, 3, &all.membership).unwrap();
    assert_eq!(merged, vec![0; 34]);
    assert_eq!(sizes, vec![34]);
    // Undoing only the last two splits gives back the first split.
    let (two, _) = le_community_to_membership(&all.merges, 2, &all.membership).unwrap();
    assert!(same_partition(&two, &one.membership));
}

#[test]
fn leading_eigenvector_from_given_start_and_components() {
    // Two disjoint triangles: the start is the two components.
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], 6, false).unwrap();
    let res = g.community_leading_eigenvector(None, None, None).unwrap();
    assert_eq!(res.num_communities(), 2);
    assert!(res.eigenvalues[0].is_nan()); // the split given by the components
    assert!(res.eigenvectors[0].is_empty());

    let start = [0, 0, 0, 0, 0, 0];
    let res = g
        .community_leading_eigenvector(None, None, Some(&start))
        .unwrap();
    assert_eq!(
        res.history[0],
        LeadingEigenvectorEvent::StartGiven { communities: 1 }
    );
    assert!(same_partition(&res.membership, &[0, 0, 0, 1, 1, 1]));
    assert_eq!(
        g.community_leading_eigenvector(None, None, Some(&[0, 0]))
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn leading_eigenvector_callback_can_stop_and_panics_propagate() {
    let g = karate();
    let mut calls = 0;
    let res = g
        .community_leading_eigenvector_with(None, None, None, |step| {
            calls += 1;
            assert_eq!(step.membership().len(), 34);
            assert_eq!(step.community(), 0);
            assert_eq!(step.eigenvector().len(), 34);
            // The eigenvector really is one: B v = λ v.
            let bv = step.multiply(step.eigenvector()).unwrap();
            for (x, v) in bv.iter().zip(step.eigenvector()) {
                assert!((x - step.eigenvalue() * v).abs() < 1e-6);
            }
            assert!(step.multiply(&[1.0]).is_err());
            ControlFlow::Break(())
        })
        .unwrap();
    assert_eq!(calls, 1);
    // Stopped before the first split.
    assert_eq!(res.num_communities(), 1);

    let caught = std::panic::catch_unwind(|| {
        let _ = karate().community_leading_eigenvector_with(None, None, None, |_| panic!("boom"));
    });
    assert!(caught.is_err());
    // igraph is still usable afterwards.
    assert_eq!(
        karate()
            .community_leading_eigenvector(None, Some(1), None)
            .unwrap()
            .num_communities(),
        2
    );
}

// ---------------------------------------------------------------------------
// Other methods
// ---------------------------------------------------------------------------

#[test]
fn spinglass_on_karate() {
    let g = karate();
    rng::seed(1).unwrap();
    let res = g
        .community_spinglass(None, &SpinglassOptions::default())
        .unwrap();
    assert!(res.modularity > 0.4);
    assert_eq!(res.csize.iter().sum::<i64>(), 34);
    assert_eq!(res.csize.len(), res.num_communities());
    assert_close(
        g.modularity(&res.membership, None, 1.0, false).unwrap(),
        res.modularity,
        1e-9,
    );
    assert!(res.temperature > 0.0 && res.temperature < 1.0);

    let single = g
        .community_spinglass_single(None, 0, &SpinglassOptions::default())
        .unwrap();
    assert!(single.community.contains(&0));
    assert!(single.inner_links > single.outer_links);

    // Disconnected graphs are refused.
    let disc = Graph::from_edges(&[(0, 1), (2, 3)], 4, false).unwrap();
    assert!(
        disc.community_spinglass(None, &SpinglassOptions::default())
            .is_err()
    );
    // Invalid vertices are rejected, including `vcount` itself, which
    // igraph (1.0.0 and 1.0.1) accepts and answers with an empty community.
    for v in [34, 99, -1] {
        assert_eq!(
            g.community_spinglass_single(None, v, &SpinglassOptions::default())
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidVertexId,
            "vertex {v}"
        );
    }
}

#[test]
fn spinglass_matches_igraph_unit_test() {
    // Two 5-cliques connected by a single edge; igraph finds modularity
    // 0.452381 with every implementation and update scheme.
    let mut edges = vec![];
    for base in [0, 5] {
        for i in 0..5 {
            for j in i + 1..5 {
                edges.push((base + i, base + j));
            }
        }
    }
    edges.push((0, 5));
    let g = Graph::from_edges(&edges, 10, false).unwrap();
    let two_cliques = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let base = SpinglassOptions {
        spins: 10,
        ..Default::default()
    };
    let variants = [
        base.clone(),
        SpinglassOptions {
            parallel_update: true,
            ..base.clone()
        },
        SpinglassOptions {
            implementation: SpinglassImplementation::Neg,
            gamma_minus: 0.0,
            ..base.clone()
        },
    ];
    for (i, opts) in variants.iter().enumerate() {
        rng::seed(137).unwrap();
        let res = g.community_spinglass(None, opts).unwrap();
        assert_close(res.modularity, 0.452381, 1e-6);
        assert!(same_partition(&res.membership, &two_cliques), "variant {i}");
        assert_eq!(res.csize, vec![5, 5]);
    }
    // The single-vertex version finds the clique of vertex 0, whose edges
    // are counted exactly: 10 inside, 1 (the bridge) leaving.
    rng::seed(137).unwrap();
    let single = g
        .community_spinglass_single(
            None,
            0,
            &SpinglassOptions {
                spins: 2,
                ..Default::default()
            },
        )
        .unwrap();
    let mut community = single.community.clone();
    community.sort_unstable();
    assert_eq!(community, vec![0, 1, 2, 3, 4]);
    assert_eq!((single.inner_links, single.outer_links), (10.0, 1.0));
    // Weighted links are summed.
    let mut w = vec![2.0; g.ecount()];
    w[g.ecount() - 1] = 0.5;
    rng::seed(137).unwrap();
    let weighted = g
        .community_spinglass_single(Some(&w), 7, &SpinglassOptions::default())
        .unwrap();
    let mut community = weighted.community.clone();
    community.sort_unstable();
    assert_eq!(community, vec![5, 6, 7, 8, 9]);
    assert_eq!((weighted.inner_links, weighted.outer_links), (20.0, 0.5));
    // Fewer than two spins are rejected.
    assert_eq!(
        g.community_spinglass_single(
            None,
            0,
            &SpinglassOptions {
                spins: 1,
                ..Default::default()
            }
        )
        .unwrap_err()
        .kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn spinglass_negative_weights() {
    // Positive edges inside the triangles, a negative bridge.
    let g = two_triangles();
    let w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0];
    rng::seed(5).unwrap();
    let opts = SpinglassOptions {
        implementation: SpinglassImplementation::Neg,
        ..Default::default()
    };
    let res = g.community_spinglass(Some(&w), &opts).unwrap();
    assert!(same_partition(&res.membership, &[0, 0, 0, 1, 1, 1]));
    // The original implementation refuses negative weights.
    assert!(
        g.community_spinglass(Some(&w), &SpinglassOptions::default())
            .is_err()
    );
}

#[test]
fn label_propagation_variants_respect_fixed_labels() {
    let g = karate();
    let mut initial = vec![-1; 34];
    initial[0] = 0;
    initial[33] = 1;
    let mut fixed = vec![false; 34];
    fixed[0] = true;
    fixed[33] = true;
    for variant in [
        LpaVariant::Dominance,
        LpaVariant::Retention,
        LpaVariant::Fast,
    ] {
        rng::seed(11).unwrap();
        let opts = LabelPropagationOptions {
            initial: Some(&initial),
            fixed: Some(&fixed),
            variant,
            ..Default::default()
        };
        let m = g.community_label_propagation(None, &opts).unwrap();
        assert_ne!(m[0], m[33], "{variant:?}");
        assert!(m.iter().all(|&l| l == m[0] || l == m[33]));
        // Most of the club follows the real split.
        let factions = karate_factions();
        let agree = (0..34)
            .filter(|&v| (m[v] == m[33]) == (factions[v] == 1))
            .count();
        assert!(agree >= 24, "{variant:?}: {m:?}");
    }
    // Without constraints, every vertex gets some label.
    let free = g
        .community_label_propagation(None, &LabelPropagationOptions::default())
        .unwrap();
    assert!(free.iter().all(|&l| l >= 0));
    assert!(
        g.community_label_propagation(
            None,
            &LabelPropagationOptions {
                initial: Some(&[0, 1]),
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn label_propagation_directed_modes() {
    // Two directed 3-cycles (strongly connected components) and a one-way
    // edge 2 -> 3 between them.
    let g = Graph::from_edges(
        &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)],
        6,
        true,
    )
    .unwrap();
    let scc = g.connected_components(Connectedness::Strong).unwrap();
    assert_eq!(scc.count, 2);
    for mode in [NeighborMode::Out, NeighborMode::In, NeighborMode::All] {
        for seed in 0..10 {
            rng::seed(seed).unwrap();
            let opts = LabelPropagationOptions {
                mode,
                ..Default::default()
            };
            let m = g.community_label_propagation(None, &opts).unwrap();
            assert_eq!(m.len(), 6);
            // Along a directed cycle every vertex must end up with the label of
            // its only predecessor (Out) or successor (In): each strongly
            // connected component is monochromatic.
            for c in 0..2 {
                let labels: Vec<i64> = (0..6)
                    .filter(|&v| scc.membership[v] == c)
                    .map(|v| m[v])
                    .collect();
                assert!(
                    labels.iter().all(|&l| l == labels[0]),
                    "{mode:?}, seed {seed}: {m:?}"
                );
            }
        }
    }
    // Labels can only flow along the edge 2 -> 3 in Out mode: fixing a
    // label on the downstream cycle cannot reach the upstream one.
    let initial = [-1, -1, -1, 5, -1, -1];
    let fixed = [false, false, false, true, false, false];
    let opts = LabelPropagationOptions {
        mode: NeighborMode::Out,
        initial: Some(&initial),
        fixed: Some(&fixed),
        ..Default::default()
    };
    rng::seed(1).unwrap();
    let m = g.community_label_propagation(None, &opts).unwrap();
    assert_eq!(m[3], m[4]);
    assert_eq!(m[4], m[5]);
    // In Out mode, vertices 0..3 receive no label from 3, 4 or 5.
    assert!((0..3).all(|v| m[v] != m[3]), "{m:?}");
    // Labels must be vertex-range ids. igraph itself (1.0.0 and 1.0.1) only
    // rejects labels above |V| and would index out of bounds with |V|.
    for label in [6, 7, i64::MAX] {
        let bad = [-1, -1, -1, label, -1, -1];
        assert_eq!(
            g.community_label_propagation(
                None,
                &LabelPropagationOptions {
                    initial: Some(&bad),
                    ..Default::default()
                }
            )
            .unwrap_err()
            .kind(),
            ErrorKind::InvalidValue,
            "label {label}"
        );
    }
    // The null graph with an (empty) initial labeling: igraph would take the
    // maximum of an empty vector and abort the process.
    let null = Graph::new(0, true);
    let opts = LabelPropagationOptions {
        initial: Some(&[]),
        fixed: Some(&[]),
        ..Default::default()
    };
    assert!(
        null.community_label_propagation(None, &opts)
            .unwrap()
            .is_empty()
    );
    // Zero weights are allowed, negative ones are not.
    let p = path(3);
    assert_eq!(
        p.community_label_propagation(Some(&[0.0, 1.0]), &Default::default())
            .unwrap()
            .len(),
        3
    );
    assert_eq!(
        p.community_label_propagation(Some(&[-1.0, 1.0]), &Default::default())
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn infomap_matches_igraph_unit_test() {
    let opts = InfomapOptions {
        trials: 5,
        ..Default::default()
    };
    // Two triangles connected by the edge 0-5.
    let g = Graph::from_edges(
        &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 5)],
        6,
        false,
    )
    .unwrap();
    rng::seed(42).unwrap();
    let res = g.community_infomap(None, None, &opts).unwrap();
    assert_close(res.codelength, 2.32073, 1e-5);
    assert!(same_partition(&res.membership, &[0, 0, 0, 1, 1, 1]));
    // Two 4-cliques joined by two edges.
    let g = Graph::from_edges(
        &[
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (7, 4),
            (7, 5),
            (7, 6),
            (4, 5),
            (4, 6),
            (5, 6),
            (0, 4),
            (1, 5),
        ],
        8,
        false,
    )
    .unwrap();
    let res = g.community_infomap(None, None, &opts).unwrap();
    assert_close(res.codelength, 2.74930, 1e-5);
    assert!(same_partition(&res.membership, &[0, 0, 0, 0, 1, 1, 1, 1]));
    // Zachary's karate club: three modules; regularized: a single one.
    let k = karate();
    let res = k.community_infomap(None, None, &opts).unwrap();
    assert_close(res.codelength, 4.31179, 1e-5);
    assert!(same_partition(
        &res.membership,
        &[
            0, 0, 0, 0, 1, 1, 1, 0, 2, 0, 1, 0, 0, 0, 2, 2, 1, 0, 2, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2
        ]
    ));
    let reg = InfomapOptions {
        regularized: true,
        ..opts
    };
    let res = k.community_infomap(None, None, &reg).unwrap();
    assert_close(res.codelength, 4.95662, 1e-5);
    assert_eq!(res.membership, vec![0; 34]);
    // Trivial graphs have zero code length.
    let res = Graph::new(2, false)
        .community_infomap(None, None, &opts)
        .unwrap();
    assert_eq!(res.codelength, 0.0);
    assert_eq!(res.membership, vec![0, 1]);
}

#[test]
fn infomap_two_cliques_and_weights() {
    let g = ring_of_cliques(2, 6);
    rng::seed(3).unwrap();
    let res = g
        .community_infomap(None, None, &InfomapOptions::default())
        .unwrap();
    assert_eq!(res.num_communities(), 2);
    assert!(res.codelength > 0.0);
    // A single community for a complete graph, with codelength log2(n).
    let k = complete(8);
    let one = k
        .community_infomap(
            None,
            None,
            &InfomapOptions {
                trials: 2,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(one.num_communities(), 1);
    assert_close(one.codelength, 3.0, 1e-9);
    let reg = InfomapOptions {
        regularized: true,
        regularization_strength: 0.5,
        ..Default::default()
    };
    assert!(g.community_infomap(None, Some(&[1.0; 12]), &reg).is_ok());
    assert_eq!(
        g.community_infomap(None, Some(&[1.0]), &InfomapOptions::default())
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn fluid_communities_and_errors() {
    let g = ring_of_cliques(4, 5);
    rng::seed(9).unwrap();
    let m = g.community_fluid_communities(4).unwrap();
    assert_eq!(m.len(), 20);
    let mut labels = m.clone();
    labels.sort_unstable();
    labels.dedup();
    assert_eq!(labels, vec![0, 1, 2, 3]);
    assert!(g.modularity(&m, None, 1.0, false).unwrap() > 0.3);
    assert!(g.community_fluid_communities(0).is_err());
    assert!(g.community_fluid_communities(21).is_err());
    let disc = Graph::from_edges(&[(0, 1), (2, 3)], 4, false).unwrap();
    assert!(disc.community_fluid_communities(2).is_err());
}

#[test]
fn voronoi_matches_igraph_unit_test() {
    let null = Graph::new(0, false);
    let res = null
        .community_voronoi(None, None, NeighborMode::All, None)
        .unwrap();
    assert!(res.membership.is_empty() && res.generators.is_empty());
    let two = Graph::new(2, false);
    let res = two
        .community_voronoi(None, None, NeighborMode::All, None)
        .unwrap();
    assert_eq!(res.membership, vec![0, 1]);
    assert_eq!(res.generators, vec![0, 1]);

    let g = karate();
    rng::seed(42).unwrap();
    let res = g
        .community_voronoi(None, None, NeighborMode::All, None)
        .unwrap();
    assert_eq!(
        res.membership,
        vec![
            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 2, 2, 0, 0, 0,
            0, 0, 2, 0, 0
        ]
    );
    assert_eq!(res.generators, vec![33, 0, 24]);
    assert_close(
        g.modularity(&res.membership, None, 1.0, false).unwrap(),
        res.modularity,
        1e-12,
    );
    // Each generator belongs to its own community.
    for (c, &generator) in res.generators.iter().enumerate() {
        assert_eq!(res.membership[generator as usize], c as i64);
    }
    // A huge radius gives a single community.
    let one = g
        .community_voronoi(None, None, NeighborMode::All, Some(100.0))
        .unwrap();
    assert_eq!(one.num_communities(), 1);
}

#[test]
fn optimal_modularity_example() {
    let g = Graph::from_edges(
        &[
            (0, 3),
            (0, 4),
            (0, 1),
            (0, 2),
            (0, 5),
            (0, 6),
            (1, 3),
            (3, 5),
            (4, 7),
            (7, 8),
            (2, 4),
            (1, 2),
            (6, 8),
        ],
        9,
        false,
    )
    .unwrap();
    match g.community_optimal_modularity(None, 1.0) {
        Ok(res) => {
            assert_close(res.modularity, 0.221893, 1e-6);
            assert_eq!(res.membership, vec![0, 0, 0, 0, 1, 0, 1, 1, 1]);
            // No heuristic can beat it.
            rng::seed(1).unwrap();
            let ml = g.community_multilevel(None, 1.0).unwrap();
            assert!(ml.modularity() <= res.modularity + 1e-12);
        }
        Err(e) => assert_eq!(e.kind(), ErrorKind::Unimplemented),
    }
}

// ---------------------------------------------------------------------------
// Comparing partitions
// ---------------------------------------------------------------------------

#[test]
fn compare_communities_matches_igraph_unit_test() {
    let a = [2, 0, 2, 1, 1, 0, 2, 2, 1, 2];
    let b = [1, 1, 2, 1, 1, 0, 2, 2, 0, 2];
    let cmp = |m| compare_communities(&a, &b, m).unwrap();
    assert_close(cmp(CommunityComparison::Vi), 1.1343, 1e-4);
    assert_close(cmp(CommunityComparison::Rand), 0.711111, 1e-6);
    assert_close(cmp(CommunityComparison::AdjustedRand), 0.312573, 1e-6);
    assert_close(cmp(CommunityComparison::Nmi), 0.455859, 1e-6);
    assert_eq!(cmp(CommunityComparison::SplitJoin), 6.0);
    // Symmetry.
    for m in [
        CommunityComparison::Vi,
        CommunityComparison::Nmi,
        CommunityComparison::Rand,
    ] {
        assert_close(compare_communities(&b, &a, m).unwrap(), cmp(m), 1e-12);
    }

    // Equal but differently labeled partitions.
    assert_eq!(
        compare_communities(&[0, 1], &[1, 0], CommunityComparison::Vi).unwrap(),
        0.0
    );
    assert_eq!(
        compare_communities(&[0, 1], &[1, 0], CommunityComparison::Rand).unwrap(),
        1.0
    );
    assert!(
        compare_communities(&[0, 1], &[1, 0], CommunityComparison::AdjustedRand)
            .unwrap()
            .is_nan()
    );
    // Rand indices need at least two elements.
    assert_eq!(
        compare_communities(&[0], &[0], CommunityComparison::Rand)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        compare_communities(&[], &[], CommunityComparison::Nmi).unwrap(),
        1.0
    );
    assert!(compare_communities(&[0, 1], &[0], CommunityComparison::Vi).is_err());
}

#[test]
fn split_join_distance_matches_igraph_unit_test() {
    assert_eq!(split_join_distance(&[], &[]).unwrap(), (0, 0));
    assert_eq!(
        split_join_distance(&[0, 1, 2, 3, 4], &[0; 5]).unwrap(),
        (0, 4)
    );
    assert_eq!(
        split_join_distance(&[2, 0, 1, 2, 1, 2, 0], &[1, 3, 0, 1, 2, 2, 1]).unwrap(),
        (3, 2)
    );
    assert_eq!(split_join_distance(&[0, 0, 2], &[0, 2, 2]).unwrap(), (1, 1));
    assert_eq!(
        split_join_distance(&[0, 1, 2], &[0; 5]).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    // The sum is the split-join distance of compare_communities.
    let (d12, d21) = split_join_distance(&[2, 0, 1, 2, 1, 2, 0], &[1, 3, 0, 1, 2, 2, 1]).unwrap();
    let sj = compare_communities(
        &[2, 0, 1, 2, 1, 2, 0],
        &[1, 3, 0, 1, 2, 2, 1],
        CommunityComparison::SplitJoin,
    )
    .unwrap();
    assert_eq!((d12 + d21) as f64, sj);
}

#[test]
fn reindex_membership_makes_ids_contiguous() {
    // Ids in 0..n are numbered by first appearance...
    let mut m = vec![3, 3, 1, 4, 1];
    let new_to_old = reindex_membership(&mut m).unwrap();
    assert_eq!(new_to_old, vec![3, 1, 4]);
    assert_eq!(m, vec![0, 0, 1, 2, 1]);
    // ... otherwise (5 >= n) in increasing order of the old ids.
    let mut m = vec![3, 3, 1, 5, 1];
    let new_to_old = reindex_membership(&mut m).unwrap();
    assert_eq!(new_to_old, vec![1, 3, 5]);
    assert_eq!(m, vec![1, 1, 0, 2, 0]);
    // Large and negative ids use the slower path.
    let mut m = vec![-10, 1_000_000, -10];
    let new_to_old = reindex_membership(&mut m).unwrap();
    assert_eq!(new_to_old.len(), 2);
    assert!(same_partition(&m, &[0, 1, 0]));
    assert!(m.iter().all(|&c| (0..2).contains(&c)));
    let mut empty: Vec<i64> = vec![];
    assert!(reindex_membership(&mut empty).unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// Cores and trusses
// ---------------------------------------------------------------------------

#[test]
fn coreness_matches_igraph_unit_test() {
    let g = karate();
    assert_eq!(
        g.coreness(NeighborMode::All).unwrap(),
        vec![
            4, 4, 4, 4, 3, 3, 3, 4, 4, 2, 3, 1, 2, 4, 2, 2, 2, 2, 2, 3, 2, 2, 2, 3, 3, 3, 2, 3, 3,
            3, 4, 3, 4, 4
        ]
    );
    assert_eq!(complete(5).coreness(NeighborMode::All).unwrap(), vec![4; 5]);
    // Full directed graph on 5 vertices.
    let mut edges = vec![];
    for i in 0..5 {
        for j in 0..5 {
            if i != j {
                edges.push((i, j));
            }
        }
    }
    let d = Graph::from_edges(&edges, 5, true).unwrap();
    assert_eq!(d.coreness(NeighborMode::All).unwrap(), vec![8; 5]);
    assert_eq!(d.coreness(NeighborMode::Out).unwrap(), vec![4; 5]);
    assert_eq!(d.coreness(NeighborMode::In).unwrap(), vec![4; 5]);
    assert!(
        Graph::new(0, false)
            .coreness(NeighborMode::All)
            .unwrap()
            .is_empty()
    );
    assert_eq!(path(3).coreness(NeighborMode::All).unwrap(), vec![1, 1, 1]);
}

#[test]
fn trussness_matches_igraph_unit_test() {
    let edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
        (3, 6),
        (3, 11),
        (4, 5),
        (4, 6),
        (5, 6),
        (5, 7),
        (5, 8),
        (5, 9),
        (6, 7),
        (6, 10),
        (6, 11),
        (7, 8),
        (7, 9),
        (8, 9),
        (8, 10),
    ];
    let g = Graph::from_edges(&edges, 0, false).unwrap();
    let expected = [
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 4, 4, 4, 3, 2, 3, 4, 4, 4, 2,
    ];
    assert_eq!(g.trussness().unwrap(), expected);

    // Loops are allowed and get trussness 2.
    let mut with_loops = g.clone();
    with_loops.add_edges(&[(0, 0), (7, 7), (5, 5)]).unwrap();
    let t = with_loops.trussness().unwrap();
    assert_eq!(&t[..25], &expected[..]);
    assert_eq!(&t[25..], &[2, 2, 2]);

    assert!(Graph::new(10, false).trussness().unwrap().is_empty());
    // Multigraphs are not supported.
    let multi = Graph::from_edges(&[(0, 1), (0, 1), (1, 2), (2, 0)], 3, false).unwrap();
    assert_eq!(
        multi.trussness().unwrap_err().kind(),
        ErrorKind::Unimplemented
    );
}

// ---------------------------------------------------------------------------
// Hierarchical random graphs
// ---------------------------------------------------------------------------

#[test]
fn hrg_create_and_sample_match_igraph_unit_test() {
    // Two leaves always connected.
    let tree = Graph::from_edges(&[(0, 1), (0, 2)], 3, true).unwrap();
    let hrg = Hrg::create(&tree, &[1.0]).unwrap();
    assert_eq!(hrg.size(), 2);
    let s = Graph::hrg_game(&hrg).unwrap();
    assert!(!s.is_directed());
    assert_eq!(s.edge_list(), vec![(0, 1)]);

    // Four leaves, one connected to all others.
    let tree =
        Graph::from_edges(&[(0, 3), (0, 1), (1, 4), (1, 2), (2, 5), (2, 6)], 7, true).unwrap();
    let hrg = Hrg::create(&tree, &[1.0, 0.0, 0.0]).unwrap();
    assert_eq!(hrg.size(), 4);
    assert_eq!(hrg.left(), &[-2, -3, 2]);
    assert_eq!(hrg.right(), &[0, 1, 3]);
    assert_eq!(hrg.prob(), &[1.0, 0.0, 0.0]);
    for s in hrg.sample_many(5).unwrap() {
        assert_eq!(s.edge_list(), vec![(0, 1), (0, 2), (0, 3)]);
    }
    assert!(hrg.sample_many(0).unwrap().is_empty());

    // The dendrogram round-trips.
    let (dendro, prob) = hrg.dendrogram().unwrap();
    assert_eq!(dendro.vcount(), 7);
    assert_eq!(dendro.ecount(), 6);
    assert!(dendro.is_directed());
    assert!(prob[..4].iter().all(|p| p.is_nan()));
    assert_eq!(&prob[4..], hrg.prob());
    // Its internal vertices come last, so relabel them first for `create`.
    let n = 4;
    let relabel = |v: i64| if v >= n { v - n } else { v + n - 1 };
    let edges: Vec<(i64, i64)> = dendro
        .edge_list()
        .into_iter()
        .map(|(a, b)| (relabel(a), relabel(b)))
        .collect();
    let tree2 = Graph::from_edges(&edges, 7, true).unwrap();
    let hrg2 = Hrg::create(&tree2, &prob[4..]).unwrap();
    assert_eq!(
        hrg2.sample().unwrap().edge_list(),
        vec![(0, 1), (0, 2), (0, 3)]
    );
    let (dendro2, _) = Graph::from_hrg_dendrogram(&hrg2).unwrap();
    assert_eq!(dendro2.ecount(), 6);
}

#[test]
fn hrg_create_errors() {
    let err = |tree: &Graph, prob: &[f64]| Hrg::create(tree, prob).unwrap_err().kind();
    let tree = Graph::from_edges(&[(0, 1), (0, 2)], 3, true).unwrap();
    assert_eq!(err(&tree, &[1.0, 0.0, 0.0]), ErrorKind::InvalidValue);
    let with_loop = Graph::from_edges(&[(0, 0), (0, 2)], 3, true).unwrap();
    assert_eq!(err(&with_loop, &[1.0]), ErrorKind::InvalidValue);
    assert_eq!(err(&Graph::new(3, true), &[1.0]), ErrorKind::InvalidValue);
    assert_eq!(
        err(
            &Graph::from_edges(&[(0, 1), (0, 2)], 3, false).unwrap(),
            &[1.0]
        ),
        ErrorKind::InvalidValue
    );
    // Internal vertex with a large id: rejected on the Rust side.
    let late_root = Graph::from_edges(&[(2, 0), (2, 1)], 3, true).unwrap();
    assert_eq!(err(&late_root, &[1.0]), ErrorKind::InvalidValue);
    // Undirected trees are reported as such by igraph.
    let undirected = Graph::from_edges(&[(0, 1), (0, 2)], 3, false).unwrap();
    let e = Hrg::create(&undirected, &[1.0]).unwrap_err();
    assert!(e.to_string().contains("directed"), "{e}");
}

#[test]
fn hrg_fit_consensus_predict() {
    let g = ring_of_cliques(3, 4);
    rng::seed(42).unwrap();
    let hrg = g.hrg_fit(0).unwrap();
    assert_eq!(hrg.size(), 12);
    assert_eq!(hrg.left().len(), 11);
    assert!(hrg.prob().iter().all(|p| (0.0..=1.0).contains(p)));
    // The root subtree contains all the leaves and edges.
    assert_eq!(hrg.vertices().iter().max(), Some(&12));
    let copy = hrg.clone();
    assert_eq!(copy, hrg);
    assert!(format!("{hrg:?}").contains("size: 12"));

    let mut refined = hrg.clone();
    g.hrg_refit(&mut refined, 100).unwrap();
    assert_eq!(refined.size(), 12);

    let samples = hrg.sample_many(3).unwrap();
    assert_eq!(samples.len(), 3);
    assert!(samples.iter().all(|s| s.vcount() == 12));

    let cons = g.hrg_consensus(Some(&hrg), 100).unwrap();
    assert!(cons.parents.len() >= 12);
    assert_eq!(cons.parents.len(), 12 + cons.weights.len());
    assert!(cons.parents.contains(&-1));
    let cons2 = g.hrg_consensus(None, 50).unwrap();
    assert_eq!(cons2.hrg.size(), 12);

    let pred = g.hrg_predict(Some(&hrg), 100, 25).unwrap();
    assert_eq!(pred.edges.len(), pred.prob.len());
    // Candidates are exactly the non-adjacent pairs, sorted by probability.
    assert_eq!(pred.edges.len(), 12 * 11 / 2 - g.ecount());
    for &(a, b) in &pred.edges {
        assert_eq!(g.get_eid(a, b, false).unwrap(), None);
    }
    assert!(pred.prob.windows(2).all(|w| w[0] >= w[1]));
    assert!(pred.prob.iter().all(|p| (0.0..=1.0).contains(p)));

    // Mismatched start models and too small graphs.
    let small = cycle(5);
    assert_eq!(
        small.hrg_consensus(Some(&hrg), 10).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        small.hrg_predict(Some(&hrg), 10, 10).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    let mut h = hrg.clone();
    assert_eq!(
        small.hrg_refit(&mut h, 10).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        path(2).hrg_fit(10).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    for n in 0..3 {
        let tiny = path(n);
        assert_eq!(tiny.hrg_fit(0).unwrap_err().kind(), ErrorKind::InvalidValue);
        assert_eq!(
            tiny.hrg_consensus(None, 10).unwrap_err().kind(),
            ErrorKind::InvalidValue
        );
        assert_eq!(
            tiny.hrg_predict(None, 10, 10).unwrap_err().kind(),
            ErrorKind::InvalidValue
        );
    }
    // A failed refit leaves the model untouched.
    let before = h.clone();
    assert!(small.hrg_refit(&mut h, 10).is_err());
    assert_eq!(h, before);
}

/// Link prediction story: hide an edge inside a dense group and ask the HRG
/// which missing links are most plausible.
#[test]
fn story_hrg_predicts_hidden_link_inside_a_clique() {
    let mut g = ring_of_cliques(2, 5);
    // Hide the edge 1-2 inside the first clique.
    let e = g.get_eid(1, 2, false).unwrap().unwrap();
    g.delete_edges(e).unwrap();
    rng::seed(7).unwrap();
    let pred = g.hrg_predict(None, 500, 25).unwrap();
    // The hidden intra-clique link is among the three most likely candidates.
    let rank = pred
        .edges
        .iter()
        .position(|&(a, b)| (a, b) == (1, 2) || (a, b) == (2, 1))
        .unwrap();
    assert!(
        rank < 3,
        "hidden edge ranked {rank}: {:?}",
        &pred.edges[..5]
    );
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

#[test]
fn wrong_weight_lengths_are_rejected_on_the_rust_side() {
    let g = two_triangles();
    let w = [1.0; 3];
    let kinds = [
        g.community_fastgreedy(Some(&w)).unwrap_err().kind(),
        g.community_walktrap(Some(&w), 4).unwrap_err().kind(),
        g.community_edge_betweenness(Some(&w), None, false)
            .unwrap_err()
            .kind(),
        g.community_edge_betweenness(None, Some(&w), false)
            .unwrap_err()
            .kind(),
        g.community_leading_eigenvector(Some(&w), None, None)
            .unwrap_err()
            .kind(),
        g.community_spinglass(Some(&w), &SpinglassOptions::default())
            .unwrap_err()
            .kind(),
        g.community_label_propagation(Some(&w), &LabelPropagationOptions::default())
            .unwrap_err()
            .kind(),
        g.community_voronoi(Some(&w), None, NeighborMode::All, None)
            .unwrap_err()
            .kind(),
        g.community_optimal_modularity(Some(&w), 1.0)
            .unwrap_err()
            .kind(),
        g.modularity_matrix(Some(&w), 1.0, false)
            .unwrap_err()
            .kind(),
        g.community_leiden(Some(&w), None, None, &LeidenOptions::default())
            .unwrap_err()
            .kind(),
    ];
    assert!(kinds.iter().all(|&k| k == ErrorKind::InvalidValue));
}

#[test]
fn result_helpers() {
    let c = Clustering {
        membership: vec![1, 0, 1, 2],
        modularity: 0.0,
    };
    assert_eq!(c.num_communities(), 3);
    assert_eq!(c.sizes(), vec![1, 2, 1]);
    assert_eq!(c.communities(), vec![vec![1], vec![0, 2], vec![3]]);
    let ml = Multilevel {
        membership: vec![],
        levels: vec![],
        modularities: vec![],
    };
    assert!(ml.modularity().is_nan());
    assert_eq!(
        LeidenObjective::try_from(
            igraph::ffi::igraph_leiden_objective_t_IGRAPH_LEIDEN_OBJECTIVE_CPM
        )
        .unwrap(),
        LeidenObjective::Cpm
    );
}

// ---------------------------------------------------------------------------
// Inputs igraph does not validate itself (would be out-of-bounds accesses)
// ---------------------------------------------------------------------------

#[test]
fn malformed_merges_are_rejected_before_reaching_igraph() {
    let kind = |merges: &[(i64, i64)], nodes, steps| {
        community_to_membership(merges, nodes, steps)
            .unwrap_err()
            .kind()
    };
    // Ids out of range: negative, beyond the leaves, or a cluster that does
    // not exist yet (merge 0 may only use leaves).
    assert_eq!(kind(&[(-1, 0)], 3, 1), ErrorKind::InvalidValue);
    assert_eq!(kind(&[(0, 99)], 3, 1), ErrorKind::InvalidValue);
    assert_eq!(kind(&[(0, 3)], 3, 1), ErrorKind::InvalidValue);
    assert_eq!(kind(&[(0, 1), (2, 5)], 3, 2), ErrorKind::InvalidValue);
    assert_eq!(
        kind(&[(0, 1), (i64::MIN, 2)], 3, 2),
        ErrorKind::InvalidValue
    );
    // A cluster merged twice.
    assert_eq!(kind(&[(0, 1), (1, 2)], 3, 2), ErrorKind::InvalidValue);
    assert_eq!(kind(&[(0, 0)], 3, 1), ErrorKind::InvalidValue);
    // Rows beyond `steps` are not used, hence not checked.
    let (m, _) = community_to_membership(&[(0, 1), (7, 7)], 3, 1).unwrap();
    assert!(same_partition(&m, &[0, 0, 1]));
    // Cutting a dendrogram whose merges were tampered with.
    let d = Dendrogram {
        num_vertices: 3,
        merges: vec![(0, 1), (3, 42)],
        modularity: vec![],
        membership: vec![0, 0, 0],
    };
    assert!(d.cut(2).is_ok());
    assert_eq!(d.cut(1).unwrap_err().kind(), ErrorKind::InvalidValue);
    // The same checks apply to leading eigenvector dendrograms.
    assert_eq!(
        le_community_to_membership(&[(0, 9)], 1, &[0, 1, 2])
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        le_community_to_membership(&[(0, 1)], 1, &[0, 7, 2])
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn eb_get_merges_requires_a_permutation_of_the_edges() {
    let g = two_triangles();
    // A repeated edge (so a missing one) would leave igraph's outputs
    // partially uninitialized.
    assert_eq!(
        g.community_eb_get_merges(false, &[6, 6, 0, 1, 2, 3, 4], None)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.community_eb_get_merges(false, &[6, 0, 1], None)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.community_eb_get_merges(false, &[6, 0, 1, 2, 3, 4, -1], None)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidEdgeId
    );
    // Removing the bridge first: the best division is the two triangles,
    // each built by two merges, and the bridge is the last merge.
    let res = g
        .community_eb_get_merges(false, &[6, 0, 1, 2, 3, 4, 5], None)
        .unwrap();
    assert!(same_partition(
        &res.dendrogram.membership,
        &[0, 0, 0, 1, 1, 1]
    ));
    assert_eq!(res.dendrogram.merges.len(), 5);
    assert_eq!(res.bridges[4], 0);
    assert_close(res.dendrogram.max_modularity(), 5.0 / 14.0, 1e-12);
    assert_eq!(res.dendrogram.modularity.len(), 6);
    // Null graph.
    let null = Graph::new(0, false);
    let res = null.community_eb_get_merges(false, &[], None).unwrap();
    assert!(res.dendrogram.merges.is_empty() && res.dendrogram.membership.is_empty());
}

#[test]
fn leading_eigenvector_start_ids_must_be_vertex_range() {
    let g = two_triangles();
    // igraph would only warn, then write out of bounds.
    assert_eq!(
        g.community_leading_eigenvector(None, None, Some(&[0, 0, 0, 1, 1, 99]))
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.community_leading_eigenvector(None, None, Some(&[0, 0, 0, 1, 1, -1]))
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    // Starting from the right answer: nothing more to split.
    let res = g
        .community_leading_eigenvector(None, None, Some(&[0, 0, 0, 1, 1, 1]))
        .unwrap();
    assert_eq!(res.membership, vec![0, 0, 0, 1, 1, 1]);
    assert_eq!(res.merges, vec![(0, 1)]);
    assert_close(res.modularity, 5.0 / 14.0, 1e-12);
    // The initial division counts towards `steps`.
    let cc =
        Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], 6, false).unwrap();
    assert_eq!(
        cc.community_leading_eigenvector(None, Some(0), None)
            .unwrap()
            .num_communities(),
        2
    );
    // The null graph, with or without a (necessarily empty) start vector.
    let null = Graph::new(0, false);
    assert!(
        null.community_leading_eigenvector(None, None, None)
            .unwrap()
            .membership
            .is_empty()
    );
    assert!(
        null.community_leading_eigenvector(None, None, Some(&[]))
            .unwrap()
            .membership
            .is_empty()
    );
}

/// On a 3-vertex path the random initial dendrogram is often already the
/// most likely one: igraph then records nothing, and the binding must still
/// return a real tree (it used to return the zero-filled placeholder).
#[test]
fn hrg_fit_with_few_steps_always_returns_a_valid_dendrogram() {
    let g = path(3);
    for seed in 0..20 {
        rng::seed(seed).unwrap();
        let hrg = g.hrg_fit(1).unwrap();
        assert_eq!(hrg.size(), 3);
        // Two internal nodes: every leaf appears once, internal node 1 once.
        let mut children: Vec<i64> = hrg.left().iter().chain(hrg.right()).copied().collect();
        children.sort_unstable();
        assert_eq!(children, vec![-2, 0, 1, 2], "seed {seed}: {hrg:?}");
        assert_eq!(hrg.vertices()[0], 3);
        // Each edge is counted at the lowest common ancestor of its endpoints.
        assert_eq!(hrg.edges().iter().sum::<i64>(), 2);
        // The dendrogram is a tree with 5 vertices and 4 edges.
        let (tree, _) = hrg.dendrogram().unwrap();
        assert_eq!((tree.vcount(), tree.ecount()), (5, 4));
        // Samples have the right size.
        assert_eq!(hrg.sample().unwrap().vcount(), 3);
    }
}

#[test]
fn hrg_parameter_validation() {
    let g = ring_of_cliques(2, 4);
    assert_eq!(
        g.hrg_predict(None, 10, 0).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.hrg_consensus(None, usize::MAX).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.hrg_fit(usize::MAX).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    // Consensus/prediction started from a model return that model unchanged.
    rng::seed(8).unwrap();
    let hrg = g.hrg_fit(0).unwrap();
    let cons = g.hrg_consensus(Some(&hrg), 20).unwrap();
    assert_eq!(cons.hrg, hrg);
    let pred = g.hrg_predict(Some(&hrg), 20, 10).unwrap();
    assert_eq!(pred.hrg, hrg);
}

#[test]
fn voronoi_radius_must_be_non_negative() {
    let g = karate();
    for r in [-1.0, f64::NAN] {
        assert_eq!(
            g.community_voronoi(None, None, NeighborMode::All, Some(r))
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidValue
        );
    }
}

#[test]
fn multilevel_without_any_merge() {
    // No edges: nothing to merge, a single modularity value (NaN) and no level.
    let g = Graph::new(4, false);
    let res = g.community_multilevel(None, 1.0).unwrap();
    assert!(res.levels.is_empty());
    assert_eq!(res.membership, vec![0, 1, 2, 3]);
    assert_eq!(res.modularities.len(), 1);
    assert_eq!(res.num_communities(), 4);
}

// ---------------------------------------------------------------------------
// Randomness: per-thread generators
// ---------------------------------------------------------------------------

/// Runs every randomized method once, after seeding the thread's generator.
fn randomized_fingerprint(g: &Graph, seed: u64) -> Vec<Vec<i64>> {
    rng::seed(seed).unwrap();
    vec![
        g.community_multilevel(None, 1.0).unwrap().membership,
        g.community_leiden_simple(None, LeidenObjective::Modularity, &LeidenOptions::default())
            .unwrap()
            .membership,
        g.community_label_propagation(None, &LabelPropagationOptions::default())
            .unwrap(),
        g.community_fluid_communities(2).unwrap(),
        g.community_infomap(None, None, &InfomapOptions::default())
            .unwrap()
            .membership,
        g.community_spinglass(None, &SpinglassOptions::default())
            .unwrap()
            .membership,
        g.hrg_fit(500).unwrap().left().to_vec(),
    ]
}

/// Seeded runs give the same results in concurrently running threads, since
/// each thread owns its generator (no locking is needed).
#[test]
fn seeded_results_are_reproducible_in_parallel_threads() {
    let expected = randomized_fingerprint(&karate(), 2024);
    let handles: Vec<_> = (0..4)
        .map(|_| std::thread::spawn(|| randomized_fingerprint(&karate(), 2024)))
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), expected);
    }
    // Reseeding reproduces the run in the same thread, too.
    assert_eq!(randomized_fingerprint(&karate(), 2024), expected);
}

/// A dedicated generator installed with `Rng::scoped` gives the same result
/// as seeding the default one, and leaves the default generator untouched.
#[test]
fn scoped_generator_drives_randomized_methods() {
    let g = karate();
    let mut a = Rng::new(RngType::Pcg32, 5).unwrap();
    let mut b = Rng::new(RngType::Pcg32, 5).unwrap();
    rng::seed(99).unwrap();
    let first = a.scoped(|| g.community_multilevel(None, 1.0).unwrap());
    let default_after = rng::integer(0, i64::MAX / 2);
    let second = b.scoped(|| g.community_multilevel(None, 1.0).unwrap());
    assert_eq!(first, second);
    rng::seed(99).unwrap();
    assert_eq!(rng::integer(0, i64::MAX / 2), default_after);
}

// ---------------------------------------------------------------------------
// Consistency with other modules
// ---------------------------------------------------------------------------

/// Story: sample graphs with a planted block structure from a stochastic
/// block model and check that the methods recover the blocks.
#[test]
fn story_sbm_planted_partition_is_recovered() {
    rng::seed(7).unwrap();
    let pref = Matrix::from_rows(&[
        [0.5, 0.02, 0.02, 0.02],
        [0.02, 0.5, 0.02, 0.02],
        [0.02, 0.02, 0.5, 0.02],
        [0.02, 0.02, 0.02, 0.5],
    ])
    .unwrap();
    let g = Graph::sbm_game(&pref, &[25; 4], false, EdgeTypeSw::Simple).unwrap();
    assert!(g.is_connected(Connectedness::Weak).unwrap());
    let planted: Vec<i64> = (0..100).map(|v| v / 25).collect();
    let q_planted = g.modularity(&planted, None, 1.0, false).unwrap();
    assert!(q_planted > 0.5, "planted modularity {q_planted}");

    let louvain = g.community_multilevel(None, 1.0).unwrap();
    let leiden = g
        .community_leiden_simple(
            None,
            LeidenObjective::Modularity,
            &LeidenOptions::default().with_iterations(None),
        )
        .unwrap();
    let walktrap = g.community_walktrap(None, 4).unwrap();
    let infomap = g
        .community_infomap(None, None, &InfomapOptions::default())
        .unwrap();
    for (name, m) in [
        ("louvain", &louvain.membership),
        ("leiden", &leiden.membership),
        ("walktrap", &walktrap.membership),
        ("infomap", &infomap.membership),
    ] {
        let nmi = compare_communities(m, &planted, CommunityComparison::Nmi).unwrap();
        assert!(nmi > 0.95, "{name}: NMI {nmi}");
        // Nothing much better than the planted partition exists.
        let q = g.modularity(m, None, 1.0, false).unwrap();
        assert!(q > q_planted - 0.02, "{name}: {q} vs {q_planted}");
    }
    // A cut of the walktrap dendrogram into 4 clusters is the planted one
    // or very close to it.
    let four = walktrap.cut(4).unwrap();
    let ari = compare_communities(&four, &planted, CommunityComparison::AdjustedRand).unwrap();
    assert!(ari > 0.9, "ARI {ari}");
}

/// The unnormalized nominal assortativity of a partition is its modularity.
#[test]
fn modularity_equals_unnormalized_nominal_assortativity() {
    let g = karate();
    let factions = karate_factions();
    let q = g.modularity(&factions, None, 1.0, false).unwrap();
    let r_raw = g.assortativity_nominal(&factions, false, false).unwrap();
    assert_close(q, r_raw, 1e-12);
    // The normalized coefficient is Q / (1 - Σ_c a_c²), a_c being the share
    // of edge endpoints in community c.
    let degrees = g.degree(.., NeighborMode::All, Loops::Twice).unwrap();
    let two_m = (2 * g.ecount()) as f64;
    let mut a = [0.0; 2];
    for (v, &d) in degrees.iter().enumerate() {
        a[factions[v] as usize] += d as f64 / two_m;
    }
    let r = g.assortativity_nominal(&factions, false, true).unwrap();
    assert_close(r, q / (1.0 - a.iter().map(|x| x * x).sum::<f64>()), 1e-12);
    // Same for a directed graph with directed modularity.
    let d = Graph::from_edges(
        &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)],
        6,
        true,
    )
    .unwrap();
    let m = [0, 0, 0, 1, 1, 1];
    assert_close(
        d.modularity(&m, None, 1.0, true).unwrap(),
        d.assortativity_nominal(&m, true, false).unwrap(),
        1e-12,
    );
}

/// Contracting each community into a single vertex (keeping loops and
/// multi-edges) preserves modularity, which gives a "community graph".
#[test]
fn contracted_community_graph_preserves_modularity() {
    let g = karate();
    rng::seed(3).unwrap();
    let res = g.community_multilevel(None, 1.0).unwrap();
    let k = res.num_communities();
    let mut quotient = g.clone();
    quotient.contract_vertices(&res.membership).unwrap();
    assert_eq!(quotient.vcount(), k);
    assert_eq!(quotient.ecount(), g.ecount());
    let singletons: Vec<i64> = (0..k as i64).collect();
    assert_close(
        quotient.modularity(&singletons, None, 1.0, false).unwrap(),
        res.modularity(),
        1e-12,
    );
    // Louvain on the community graph cannot improve modularity further.
    let again = quotient.community_multilevel(None, 1.0).unwrap();
    assert!(again.modularity() <= res.modularity() + 1e-12);
    // Each community, extracted as an induced subgraph, is connected.
    for members in res.communities() {
        let sub = g
            .induced_subgraph(members.as_slice(), SubgraphImplementation::Auto)
            .unwrap();
        assert!(sub.is_connected(Connectedness::Weak).unwrap());
    }
}

/// Girvan–Newman starts by removing the edge of highest edge betweenness.
#[test]
fn girvan_newman_follows_edge_betweenness() {
    let g = karate();
    let eb = g.edge_betweenness(None, .., false, false).unwrap();
    let res = g.community_edge_betweenness(None, None, false).unwrap();
    let first = res.removed_edges[0] as usize;
    let max = eb.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(eb[first], max);
    assert_close(res.edge_betweenness[0], max, 1e-9);
    // Recomputing on the graph without the first edge gives the second value.
    let mut rest = g.clone();
    rest.delete_edges(first as i64).unwrap();
    let eb2 = rest.edge_betweenness(None, .., false, false).unwrap();
    let max2 = eb2.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert_close(res.edge_betweenness[1], max2, 1e-9);
    // Every removed edge is removed exactly once.
    let mut removed = res.removed_edges.clone();
    removed.sort_unstable();
    assert_eq!(removed, (0..78).collect::<Vec<i64>>());
}

/// Divisive and agglomerative methods never merge different components, and
/// the leading eigenvector method starts from them.
#[test]
fn communities_never_span_components() {
    // A triangle, a 4-clique, and an isolated vertex.
    let g = Graph::from_edges(
        &[
            (0, 1),
            (1, 2),
            (2, 0),
            (3, 4),
            (3, 5),
            (3, 6),
            (4, 5),
            (4, 6),
            (5, 6),
        ],
        8,
        false,
    )
    .unwrap();
    let cc = g.connected_components(Connectedness::Weak).unwrap();
    assert_eq!(cc.count, 3);
    rng::seed(4).unwrap();
    let memberships = [
        g.community_multilevel(None, 1.0).unwrap().membership,
        g.community_fastgreedy(None).unwrap().membership,
        g.community_walktrap(None, 4).unwrap().membership,
        g.community_edge_betweenness(None, None, false)
            .unwrap()
            .membership,
        g.community_leading_eigenvector(None, None, None)
            .unwrap()
            .membership,
    ];
    for m in &memberships {
        // Here the best partition is exactly the components.
        assert!(same_partition(m, &cc.membership), "{m:?}");
    }
    let le = g.community_leading_eigenvector(None, None, None).unwrap();
    assert_eq!(le.history[0], LeadingEigenvectorEvent::StartFull);
    // Spinglass and fluid communities refuse disconnected graphs.
    assert!(
        g.community_spinglass(None, &SpinglassOptions::default())
            .is_err()
    );
    assert!(g.community_fluid_communities(3).is_err());
}

/// Coreness is bounded by the degree, and the k-core has minimum degree k.
#[test]
fn coreness_and_k_cores() {
    let g = karate();
    let core = g.coreness(NeighborMode::All).unwrap();
    let deg = g.degree(.., NeighborMode::All, Loops::Twice).unwrap();
    assert!(core.iter().zip(&deg).all(|(c, d)| c <= d));
    let max = *core.iter().max().unwrap();
    for k in 1..=max {
        let members: Vec<i64> = (0..34).filter(|&v| core[v as usize] >= k).collect();
        let kcore = g
            .induced_subgraph(members.as_slice(), SubgraphImplementation::Auto)
            .unwrap();
        let d = kcore.degree(.., NeighborMode::All, Loops::Twice).unwrap();
        assert!(d.iter().all(|&x| x >= k), "k = {k}");
    }
    // The innermost core of the karate club is the 4-core, with 10 members.
    assert_eq!(max, 4);
    assert_eq!(core.iter().filter(|&&c| c == 4).count(), 10);
}

/// Trussness is at least 3 exactly on the edges lying in a triangle, and at
/// least k on the edges of a k-clique.
#[test]
fn trussness_agrees_with_triangles_and_cliques() {
    let g = karate();
    let truss = g.trussness().unwrap();
    let mut in_triangle = vec![false; g.ecount()];
    for [a, b, c] in g.list_triangles().unwrap() {
        for (x, y) in [(a, b), (b, c), (a, c)] {
            let e = g.get_eid(x, y, false).unwrap().unwrap();
            in_triangle[e as usize] = true;
        }
    }
    for (e, &t) in truss.iter().enumerate() {
        assert_eq!(t >= 3, in_triangle[e], "edge {e}");
    }
    for clique in g.maximal_cliques(3.., None).unwrap() {
        let k = clique.len() as i64;
        for (i, &a) in clique.iter().enumerate() {
            for &b in &clique[i + 1..] {
                let e = g.get_eid(a, b, false).unwrap().unwrap();
                assert!(truss[e as usize] >= k);
            }
        }
    }
    // The karate club's largest cliques have 5 vertices: its 5-truss exists.
    assert_eq!(truss.iter().max(), Some(&5));
}

/// A fitted HRG accounts for every edge, and its consensus tree is a forest.
#[test]
fn hrg_models_are_consistent_with_the_graph() {
    let g = karate();
    rng::seed(12).unwrap();
    let hrg = g.hrg_fit(5000).unwrap();
    assert_eq!(hrg.size(), 34);
    assert_eq!(hrg.edges().iter().sum::<i64>(), 78);
    assert_eq!(hrg.vertices()[0], 34);
    let mut refit = hrg.clone();
    g.hrg_refit(&mut refit, 1000).unwrap();
    assert_eq!(refit.edges().iter().sum::<i64>(), 78);
    // The dendrogram is an out-tree whose internal vertices have 2 children.
    let (tree, prob) = hrg.dendrogram().unwrap();
    assert!(tree.is_tree(NeighborMode::Out).unwrap());
    let out = tree.degree(.., NeighborMode::Out, Loops::Twice).unwrap();
    assert!(out[..34].iter().all(|&d| d == 0));
    assert!(out[34..].iter().all(|&d| d == 2));
    assert!(prob[34..].iter().all(|p| (0.0..=1.0).contains(p)));

    let cons = g.hrg_consensus(Some(&hrg), 50).unwrap();
    let n = cons.parents.len();
    for start in 0..n {
        // Following parents always reaches a root, without cycles, and the
        // vertices of the graph are leaves (nobody's parent).
        let (mut v, mut hops) = (start as i64, 0);
        while v != -1 {
            v = cons.parents[v as usize];
            assert!(v == -1 || v >= 34);
            hops += 1;
            assert!(hops <= n, "cycle through {start}");
        }
    }
}

// ---------------------------------------------------------------------------
// Audit regressions: inputs igraph 1.0.1 does not validate
// ---------------------------------------------------------------------------

fn tri() -> Graph {
    Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, false).unwrap()
}

fn invalid<T: std::fmt::Debug>(r: Result<T>) {
    assert_eq!(r.unwrap_err().kind(), ErrorKind::InvalidValue);
}

#[test]
fn voronoi_rejects_non_finite_or_overflowing_lengths() {
    // Used to abort on `IGRAPH_ASSERT(isfinite(lo))` in the radius search.
    let inf = f64::INFINITY;
    invalid(tri().community_voronoi(Some(&[inf; 3]), None, NeighborMode::All, None));
    let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 2), (2, 3)], 4, false).unwrap();
    invalid(g.community_voronoi(Some(&[1.0, 1.0, 1.0, inf]), None, NeighborMode::All, None));
    invalid(g.community_voronoi(Some(&[f64::MAX; 4]), None, NeighborMode::All, None));
    invalid(g.community_voronoi(None, Some(&[1.0, inf, 1.0, 1.0]), NeighborMode::All, None));
    invalid(g.community_voronoi(
        None,
        Some(&[1.0, f64::NAN, 1.0, 1.0]),
        NeighborMode::All,
        None,
    ));
    // Large but safe lengths still work, with the same partition as unit ones.
    rng::seed(1).unwrap();
    let unit = g
        .community_voronoi(None, None, NeighborMode::All, None)
        .unwrap();
    rng::seed(1).unwrap();
    let big = g
        .community_voronoi(Some(&[1e100; 4]), None, NeighborMode::All, None)
        .unwrap();
    assert_eq!(unit.membership, big.membership);
}

#[test]
fn leading_eigenvector_rejects_non_finite_or_huge_weights() {
    // Each of these used to make ARPACK abort the process (f2c STOP).
    let g = karate();
    let m = g.ecount();
    for bad in [f64::NAN, f64::INFINITY, -1e308] {
        let mut w = vec![1.0; m];
        w[0] = bad;
        invalid(g.community_leading_eigenvector(Some(&w), None, None));
    }
    invalid(g.community_leading_eigenvector(Some(&vec![f64::MAX; m]), None, None));
    // Modularity is scale invariant: large (but safe) uniform weights give
    // the unweighted result.
    let plain = g.community_leading_eigenvector(None, None, None).unwrap();
    let big = g
        .community_leading_eigenvector(Some(&vec![1e100; m]), None, None)
        .unwrap();
    assert_eq!(plain.membership, big.membership);
    assert_close(plain.modularity, big.modularity, 1e-9);
}

#[test]
fn leading_eigenvector_callback_survives_failing_igraph_calls() {
    // A failing igraph call inside the callback used to free the temporaries
    // of the running computation (use after free, SIGSEGV).
    let disconnected = Graph::from_edges(&[(0, 1), (2, 3)], 4, false).unwrap();
    let mut calls = 0;
    let res = karate()
        .community_leading_eigenvector_with(None, None, None, |step| {
            calls += 1;
            assert!(disconnected.community_fluid_communities(2).is_err());
            assert!(step.multiply(step.eigenvector()).is_ok());
            ControlFlow::Continue(())
        })
        .unwrap();
    assert!(calls > 1);
    let plain = karate()
        .community_leading_eigenvector(None, None, None)
        .unwrap();
    assert_eq!(res.membership, plain.membership);
}

#[test]
fn spinglass_rejects_parameters_that_never_terminate() {
    // Each of these used to loop forever (or overflow `spins + 1`).
    let g = Graph::from_edges(
        &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)],
        6,
        false,
    )
    .unwrap();
    let m = g.ecount();
    let d = SpinglassOptions::default;
    let neg = || SpinglassOptions {
        implementation: SpinglassImplementation::Neg,
        ..d()
    };
    let cases = [
        SpinglassOptions {
            start_temperature: f64::INFINITY,
            ..d()
        },
        SpinglassOptions {
            start_temperature: f64::MAX,
            stop_temperature: 1.0,
            ..d()
        },
        SpinglassOptions {
            gamma: f64::NAN,
            ..d()
        },
        SpinglassOptions {
            gamma: f64::INFINITY,
            ..d()
        },
        SpinglassOptions {
            cooling_factor: f64::NAN,
            ..d()
        },
        SpinglassOptions {
            spins: i64::MAX as usize,
            ..d()
        },
        SpinglassOptions { spins: 1, ..d() },
        SpinglassOptions {
            gamma_minus: f64::NAN,
            ..neg()
        },
        SpinglassOptions {
            start_temperature: f64::INFINITY,
            ..neg()
        },
        // Finite but overflowing resolutions: `Neg` never terminated.
        SpinglassOptions {
            gamma: f64::MAX,
            gamma_minus: f64::MAX,
            ..neg()
        },
        SpinglassOptions {
            gamma_minus: -f64::MAX,
            ..neg()
        },
        SpinglassOptions {
            gamma: 1e151,
            ..d()
        },
    ];
    for o in &cases {
        invalid(g.community_spinglass(None, o));
    }
    for bad in [f64::NAN, f64::INFINITY] {
        let mut w = vec![1.0; m];
        w[0] = bad;
        invalid(g.community_spinglass(Some(&w), &d()));
        invalid(g.community_spinglass(Some(&w), &neg()));
        invalid(g.community_spinglass_single(Some(&w), 0, &d()));
    }
    invalid(g.community_spinglass_single(
        None,
        0,
        &SpinglassOptions {
            gamma: f64::NAN,
            ..d()
        },
    ));
    // Overflowing weight sums (NaN energies) used to hang as well.
    invalid(g.community_spinglass(Some(&vec![f64::MAX; m]), &d()));
    // `gamma_minus` is ignored by the `Orig` implementation, so it is not
    // validated there; the bounds themselves are accepted.
    let orig_nan_minus = SpinglassOptions {
        gamma_minus: f64::NAN,
        ..d()
    };
    assert!(g.community_spinglass(None, &orig_nan_minus).is_ok());
    let neg_bounds = SpinglassOptions {
        gamma_minus: -1e150,
        ..neg()
    };
    assert!(g.community_spinglass(None, &neg_bounds).is_ok());
    // The defaults still work.
    rng::seed(3).unwrap();
    assert_eq!(
        g.community_spinglass(None, &d()).unwrap().num_communities(),
        2
    );
}

#[test]
fn infomap_rejects_truncated_trials() {
    // 2^32 trials used to be truncated to 0 by igraph's `unsigned int`.
    let g = karate();
    for trials in [0, 1 << 32, usize::MAX] {
        let o = InfomapOptions {
            trials,
            ..InfomapOptions::default()
        };
        invalid(g.community_infomap(None, None, &o));
    }
    let o = InfomapOptions {
        trials: 1,
        ..InfomapOptions::default()
    };
    rng::seed(5).unwrap();
    assert!(
        g.community_infomap(None, None, &o)
            .unwrap()
            .num_communities()
            > 1
    );
}

#[test]
fn community_to_membership_huge_node_counts() {
    // Used to abort (allocation failure) or panic (capacity overflow) in the
    // Rust-side validation itself.
    invalid(community_to_membership(&[], usize::MAX, 0));
    let err = community_to_membership(&[], 1 << 60, 0).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::OutOfMemory);
    // Merges referring to clusters beyond the available ones are rejected.
    invalid(community_to_membership(&[(0, 1 << 61)], 1 << 60, 1));
    invalid(community_to_membership(&[(0, 1), (1, 2)], 3, 2));
    assert_eq!(
        community_to_membership(&[(0, 1), (2, 3)], 3, 2).unwrap(),
        (vec![0, 0, 0], vec![3])
    );
}

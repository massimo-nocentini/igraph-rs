//! Integration tests for the `centrality` module (igraph_centrality.h, igraph_scan.h).

mod common;

use common::{assert_close, complete, cycle, karate, path};
use igraph::centrality::{
    PageRankAlgo, PageRankOptions, centralization, centralization_betweenness_tmax,
    centralization_closeness_tmax, centralization_degree_tmax,
    centralization_eigenvector_centrality_tmax,
};
use igraph::prelude::*;

const EPS: f64 = 1e-6;

fn assert_all_close(a: &[f64], b: &[f64], eps: f64) {
    assert_eq!(a.len(), b.len(), "{a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        if x.is_nan() && y.is_nan() {
            continue;
        }
        assert_close(*x, *y, eps);
    }
}

fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// Undirected star with center 0 and `n - 1` leaves.
fn star(n: usize) -> Graph {
    Graph::star(n, StarMode::Undirected, 0).unwrap()
}

// ---------------------------------------------------------------------------
// Tutorial lesson 3 on the karate club
// ---------------------------------------------------------------------------

#[test]
fn karate_tutorial_lesson_3() {
    let g = Graph::famous("Zachary").unwrap();

    let degree = g.degree(.., NeighborMode::All, Loops::Twice).unwrap();
    let (vmax, &dmax) = degree.iter().enumerate().max_by_key(|&(_, d)| *d).unwrap();
    assert_eq!((vmax, dmax), (33, 17));

    let clo = g.closeness(.., NeighborMode::All, None, false).unwrap();
    let c0 = argmax(&clo);
    assert_eq!(c0, 0);
    assert_close(clo[0], 1.0 / 58.0, 1e-7);
    assert_close(clo[0], 0.0172414, 1e-7);

    let btw = g.betweenness(None, .., false, false).unwrap();
    assert_eq!(argmax(&btw), 0);
    assert_close(btw[0], 231.0714, 1e-4);
    assert_close(btw[33], 160.5516, 1e-4);
}

// ---------------------------------------------------------------------------
// Closeness and harmonic centrality
// ---------------------------------------------------------------------------

#[test]
fn closeness_normalization_and_reachability() {
    let g = karate();
    let raw = g.closeness(.., NeighborMode::All, None, false).unwrap();
    let norm = g.closeness(.., NeighborMode::All, None, true).unwrap();
    for (r, n) in raw.iter().zip(&norm) {
        assert_close(*n, r * 33.0, 1e-12);
    }
    let det = g
        .closeness_reachability(.., NeighborMode::All, None, true, None)
        .unwrap();
    assert_eq!(det.scores, norm);
    assert!(det.reachable_count.iter().all(|&c| c == 33));
    assert!(det.all_reachable);

    // An isolated vertex gets NaN, and the graph is detected as disconnected.
    let g = Graph::from_edges(&[(0, 1)], 3, false).unwrap();
    let det = g
        .closeness_reachability(.., NeighborMode::All, None, true, None)
        .unwrap();
    assert_eq!(&det.scores[..2], &[1.0, 1.0]);
    assert!(det.scores[2].is_nan());
    assert_eq!(det.reachable_count, vec![1, 1, 0]);
    assert!(!det.all_reachable);
}

#[test]
fn closeness_directed_modes_and_weights() {
    // Directed path 0 -> 1 -> 2.
    let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    let out = g.closeness(.., NeighborMode::Out, None, false).unwrap();
    assert_close(out[0], 1.0 / 3.0, 1e-12);
    assert_close(out[1], 1.0, 1e-12);
    assert!(out[2].is_nan());
    let inn = g.closeness(.., NeighborMode::In, None, false).unwrap();
    assert!(inn[0].is_nan());
    assert_close(inn[2], 1.0 / 3.0, 1e-12);

    // Weights are path lengths.
    let u = path(3);
    let w = [2.0, 3.0];
    let c = u.closeness(.., NeighborMode::All, Some(&w), false).unwrap();
    assert_all_close(&c, &[1.0 / 7.0, 1.0 / 5.0, 1.0 / 8.0], 1e-12);
}

#[test]
fn closeness_cutoff_matches_exact_when_unlimited() {
    let g = karate();
    let exact = g.closeness(.., NeighborMode::All, None, true).unwrap();
    assert_eq!(
        g.closeness_cutoff(.., NeighborMode::All, None, true, None)
            .unwrap(),
        exact
    );
    assert_eq!(
        g.closeness_cutoff(.., NeighborMode::All, None, true, Some(100.0))
            .unwrap(),
        exact
    );
    // Within distance 1, normalized closeness is 1 for everybody with a neighbor.
    let one = g
        .closeness_cutoff(.., NeighborMode::All, None, true, Some(1.0))
        .unwrap();
    assert!(one.iter().all(|&c| (c - 1.0).abs() < 1e-12));
    let det = g
        .closeness_reachability(.., NeighborMode::All, None, true, Some(1.0))
        .unwrap();
    let deg = g.degree(.., NeighborMode::All, Loops::None).unwrap();
    assert_eq!(det.reachable_count, deg);
    assert!(!det.all_reachable);
}

#[test]
fn harmonic_centrality_igraph_unit_test_values() {
    // Directed path on 7 vertices, as in igraph's tests/unit/harmonic_centrality.c.
    let edges: Vec<(i64, i64)> = (0..6).map(|i| (i, i + 1)).collect();
    let g = Graph::from_edges(&edges, 7, true).unwrap();

    let und = g
        .harmonic_centrality(.., NeighborMode::All, None, true)
        .unwrap();
    assert_all_close(
        &und,
        &[
            0.408333, 0.547222, 0.597222, 0.611111, 0.597222, 0.547222, 0.408333,
        ],
        EPS,
    );
    let dir = g
        .harmonic_centrality(.., NeighborMode::Out, None, true)
        .unwrap();
    assert_all_close(
        &dir,
        &[0.408333, 0.380556, 0.347222, 0.305556, 0.25, 0.166667, 0.0],
        EPS,
    );

    let zero = g
        .harmonic_centrality_cutoff(.., NeighborMode::All, None, true, Some(0.0))
        .unwrap();
    assert_eq!(zero, vec![0.0; 7]);
    let one = g
        .harmonic_centrality_cutoff(.., NeighborMode::All, None, true, Some(1.0))
        .unwrap();
    assert_all_close(
        &one,
        &[
            0.166667, 0.333333, 0.333333, 0.333333, 0.333333, 0.333333, 0.166667,
        ],
        EPS,
    );

    let w: Vec<f64> = (1..=6).map(f64::from).collect();
    let wu = g
        .harmonic_centrality(.., NeighborMode::All, Some(&w), true)
        .unwrap();
    assert_all_close(
        &wu,
        &[
            0.285714, 0.32209, 0.241402, 0.187963, 0.149146, 0.116534, 0.0795695,
        ],
        EPS,
    );
    let wd = g
        .harmonic_centrality(.., NeighborMode::Out, Some(&w), true)
        .unwrap();
    assert_all_close(
        &wd,
        &[
            0.285714, 0.155423, 0.102513, 0.0712963, 0.0484848, 0.0277778, 0.0,
        ],
        EPS,
    );
}

#[test]
fn harmonic_centrality_on_disconnected_graphs_is_finite() {
    let g = Graph::from_edges(&[(0, 1), (2, 3), (3, 4)], 6, false).unwrap();
    let h = g
        .harmonic_centrality(.., NeighborMode::All, None, false)
        .unwrap();
    assert_eq!(h, vec![1.0, 1.0, 1.5, 2.0, 1.5, 0.0]);
    // Selecting vertices keeps the selector's order.
    assert_eq!(
        g.harmonic_centrality(&[3, 0], NeighborMode::All, None, false)
            .unwrap(),
        vec![2.0, 1.0]
    );
}

#[test]
fn harmonic_centrality_averages_to_global_efficiency() {
    // Global efficiency is the mean inverse distance over ordered pairs, i.e.
    // the mean of the normalized harmonic centralities.
    for g in [
        karate(),
        Graph::from_edges(&[(0, 1), (2, 3), (3, 4)], 6, false).unwrap(),
    ] {
        let h = g
            .harmonic_centrality(.., NeighborMode::All, None, true)
            .unwrap();
        let mean = h.iter().sum::<f64>() / h.len() as f64;
        assert_close(mean, g.global_efficiency(None, false).unwrap(), 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Betweenness
// ---------------------------------------------------------------------------

#[test]
fn betweenness_sum_identities_on_karate() {
    // In a connected undirected graph, Σ_v b(v) = Σ_{s<t} (d(s,t) - 1) and
    // Σ_e b(e) = Σ_{s<t} d(s,t): every shortest path of length d has d - 1
    // inner vertices and d edges.
    let g = karate();
    let dist = g
        .distances(.., .., None, NeighborMode::All)
        .unwrap()
        .to_rows();
    let n = g.vcount();
    let mut sum_d = 0.0;
    let mut pairs = 0.0;
    for (s, row) in dist.iter().enumerate() {
        for &d in row.iter().skip(s + 1) {
            assert!(d.is_finite());
            sum_d += d;
            pairs += 1.0;
        }
    }
    let vb: f64 = g.betweenness(None, .., false, false).unwrap().iter().sum();
    let eb: f64 = g
        .edge_betweenness(None, .., false, false)
        .unwrap()
        .iter()
        .sum();
    assert_close(vb, sum_d - pairs, 1e-8);
    assert_close(eb, sum_d, 1e-8);
    assert_eq!(pairs as usize, n * (n - 1) / 2);
}

#[test]
fn betweenness_normalization() {
    let g = karate();
    let n = g.vcount() as f64;
    let raw = g.betweenness(None, .., false, false).unwrap();
    let norm = g.betweenness(None, .., false, true).unwrap();
    // Normalized by the number of unordered vertex pairs.
    let pairs = n * (n - 1.0) / 2.0;
    for (r, x) in raw.iter().zip(&norm) {
        assert_close(*x, r / pairs, 1e-12);
    }
}

#[test]
fn betweenness_directed_and_weighted() {
    // Directed cycle 0 -> 1 -> 2 -> 3 -> 0: every vertex is inner to
    // 1 + 2 = 3 paths (lengths 2 and 3 through it).
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], 4, true).unwrap();
    assert_eq!(g.betweenness(None, .., true, false).unwrap(), vec![3.0; 4]);
    // Ignoring directions, each opposite pair has two shortest paths, so the
    // two vertices in between get 1/2 each.
    assert_eq!(g.betweenness(None, .., false, false).unwrap(), vec![0.5; 4]);

    // Square 0-1-2-3-0 where the route through 1 is expensive: 0 -> 2 goes via 3.
    let sq = cycle(4);
    let w = [5.0, 5.0, 1.0, 1.0]; // edges (0,1) (1,2) (2,3) (3,0)
    let b = sq.betweenness(Some(&w), .., false, false).unwrap();
    // 1 is at distance 5 from 0 and from 2; 0-2 goes 0-3-2, 1-3 has two paths.
    assert_all_close(&b, &[0.5, 0.0, 0.5, 1.0], 1e-12);
}

#[test]
fn betweenness_cutoff_behaviour() {
    let g = karate();
    let exact = g.betweenness(None, .., false, false).unwrap();
    assert_eq!(
        g.betweenness_cutoff(None, .., false, false, None).unwrap(),
        exact
    );
    assert_eq!(
        g.betweenness_cutoff(None, .., false, false, Some(-1.0))
            .unwrap(),
        exact
    );
    // With cutoff 1 no path has an inner vertex.
    assert!(
        g.betweenness_cutoff(None, .., false, false, Some(1.0))
            .unwrap()
            .iter()
            .all(|&b| b == 0.0)
    );
    // With cutoff 2 the betweenness of v counts, for each pair of non-adjacent
    // neighbors, the fraction of their common neighbors that v is.
    let c2 = g
        .betweenness_cutoff(None, .., false, false, Some(2.0))
        .unwrap();
    assert!(c2.iter().zip(&exact).all(|(a, b)| a <= &(b + 1e-9)));

    let eb1 = g
        .edge_betweenness_cutoff(None, .., false, false, Some(1.0))
        .unwrap();
    assert_eq!(eb1, vec![1.0; g.ecount()]);
    let eb = g.edge_betweenness(None, .., false, false).unwrap();
    assert_eq!(
        g.edge_betweenness_cutoff(None, .., false, false, None)
            .unwrap(),
        eb
    );
}

#[test]
fn betweenness_subset_unit_test_path() {
    // Path 0 - 1 - 2 - 3 - 4 with sources {1,2,3,4} and all targets (igraph's
    // unit test "subset without 0"): scores of vertices 1..=4 are 1.5, 3, 2.5, 0.
    let g = path(5);
    let subset = [1, 2, 3, 4];
    let b = g
        .betweenness_subset(None, &subset, .., &subset, false)
        .unwrap();
    assert_eq!(b, vec![1.5, 3.0, 2.5, 0.0]);
    let w = [1.0; 4];
    let bw = g
        .betweenness_subset(Some(&w), &subset, .., &subset, false)
        .unwrap();
    assert_eq!(bw, b);
    // "subset without 2": sources {0,1,3,4}, scores of 0, 1, 3, 4.
    let subset = [0, 1, 3, 4];
    assert_eq!(
        g.betweenness_subset(None, &subset, .., &subset, false)
            .unwrap(),
        vec![0.0, 2.5, 2.5, 0.0]
    );
}

#[test]
fn subset_betweenness_with_everything_is_full_betweenness() {
    let g = karate();
    let full = g.betweenness(None, .., false, false).unwrap();
    let sub = g.betweenness_subset(None, .., .., .., false).unwrap();
    assert_all_close(&sub, &full, 1e-8);
    let efull = g.edge_betweenness(None, .., false, false).unwrap();
    let esub = g.edge_betweenness_subset(None, .., .., .., false).unwrap();
    assert_all_close(&esub, &efull, 1e-8);
    // Paths from one side only: sources {0}, targets {33}.
    let one = g.betweenness_subset(None, 0, 33, .., false).unwrap();
    // d(0, 33) = 2 with four shortest paths, through 8, 13, 19 and 31. In
    // undirected graphs igraph counts each source-target pair with weight 1/2
    // (the pair is seen from both ends when sources == targets).
    assert_close(one.iter().sum::<f64>(), 0.5, 1e-12);
    for v in [8, 13, 19, 31] {
        assert_close(one[v], 0.125, 1e-12);
    }
}

#[test]
fn edge_betweenness_finds_the_bridge() {
    // Two 4-cliques joined by a single bridge edge.
    let mut edges = vec![];
    for base in [0, 4] {
        for i in 0..4 {
            for j in i + 1..4 {
                edges.push((base + i, base + j));
            }
        }
    }
    edges.push((3, 4));
    let g = Graph::from_edges(&edges, 8, false).unwrap();
    let eb = g.edge_betweenness(None, .., false, false).unwrap();
    let bridge = argmax(&eb);
    assert_eq!(g.edge(bridge as i64).unwrap(), (3, 4));
    assert_eq!(eb[bridge], 16.0); // 4 x 4 pairs cross it
    // Selecting a single edge.
    assert_eq!(
        g.edge_betweenness(None, bridge as i64, false, false)
            .unwrap(),
        vec![16.0]
    );
    // Directed version of an undirected graph is the same.
    assert_eq!(g.edge_betweenness(None, .., true, false).unwrap(), eb);
}

// ---------------------------------------------------------------------------
// PageRank
// ---------------------------------------------------------------------------

#[test]
fn pagerank_basic_properties() {
    let g = karate();
    let opts = PageRankOptions::default();
    let pr = g.pagerank(None, .., &opts).unwrap();
    assert_eq!(pr.value, 1.0);
    assert_close(pr.scores.iter().sum(), 1.0, 1e-10);
    // The two leaders are the top-ranked members.
    let mut order: Vec<usize> = (0..34).collect();
    order.sort_by(|&a, &b| pr.scores[b].partial_cmp(&pr.scores[a]).unwrap());
    assert_eq!(&order[..2], &[33, 0]);

    // ARPACK and PRPACK agree.
    let arpack = g
        .pagerank(None, .., &opts.with_algo(PageRankAlgo::Arpack))
        .unwrap();
    assert_close(arpack.value, 1.0, 1e-8);
    assert_all_close(&arpack.scores, &pr.scores, 1e-8);

    // Damping 0: a uniform random restart every step.
    let uni = g.pagerank(None, .., &opts.with_damping(0.0)).unwrap();
    assert!(uni.scores.iter().all(|&p| (p - 1.0 / 34.0).abs() < 1e-12));

    // Selecting vertices keeps order.
    let sel = g.pagerank(None, &[33, 0], &opts).unwrap();
    assert_eq!(sel.scores, vec![pr.scores[33], pr.scores[0]]);
}

#[test]
fn pagerank_directed_and_weighted() {
    // 1 and 2 both point to 0, 0 points to 1.
    let g = Graph::from_edges(&[(1, 0), (2, 0), (0, 1)], 3, true).unwrap();
    let pr = g.pagerank(None, .., &PageRankOptions::default()).unwrap();
    assert!(pr.scores[0] > pr.scores[1] && pr.scores[1] > pr.scores[2]);
    // Vertex 2 has no in-links: it only gets the teleportation mass (1 - d) / n.
    assert_close(pr.scores[2], 0.15 / 3.0, 1e-10);
    // Ignoring directions changes the ranking of 1 and 2.
    let und = g
        .pagerank(None, .., &PageRankOptions::default().with_directed(false))
        .unwrap();
    assert!(und.scores[1] > und.scores[2]);

    // Weights: 0 -> {1, 2} with weight 3 : 1 sends three times more to 1.
    let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 0), (2, 0)], 3, true).unwrap();
    let pr = g
        .pagerank(Some(&[3.0, 1.0, 1.0, 1.0]), .., &PageRankOptions::default())
        .unwrap();
    assert!(pr.scores[1] > pr.scores[2]);
}

#[test]
fn personalized_pagerank_variants_agree() {
    let g = karate();
    let opts = PageRankOptions::default();
    let mut reset = vec![0.0; 34];
    reset[5] = 1.0;
    let a = g
        .personalized_pagerank(None, Some(&reset), .., &opts)
        .unwrap();
    let b = g.personalized_pagerank_vs(None, 5, .., &opts).unwrap();
    assert_all_close(&a.scores, &b.scores, 1e-10);
    assert_eq!(argmax(&a.scores), 5);

    // Reset over two vertices with equal mass == reset_vids [2, 7].
    let mut reset2 = vec![0.0; 34];
    reset2[2] = 3.0;
    reset2[7] = 3.0; // need not be normalized
    let c = g
        .personalized_pagerank(None, Some(&reset2), .., &opts)
        .unwrap();
    let d = g
        .personalized_pagerank_vs(None, &[2, 7], .., &opts)
        .unwrap();
    assert_all_close(&c.scores, &d.scores, 1e-10);

    // No reset vector == plain PageRank.
    let e = g.personalized_pagerank(None, None, .., &opts).unwrap();
    let f = g.pagerank(None, .., &opts).unwrap();
    assert_all_close(&e.scores, &f.scores, 1e-12);
}

#[test]
fn pagerank_errors() {
    let g = karate();
    let opts = PageRankOptions::default();
    let err = g.pagerank(None, 100, &opts).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    let err = g.pagerank(None, .., &opts.with_damping(1.5)).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let err = g
        .personalized_pagerank(None, Some(&[1.0; 3]), .., &opts)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let err = g.pagerank(Some(&[1.0; 3]), .., &opts).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

#[test]
fn pagerank_igraph_unit_test_values() {
    // igraph's tests/unit/igraph_pagerank.{c,out}; the first two graphs are
    // directed but the computation ignores directions.
    let undirected = PageRankOptions::default().with_directed(false);
    for algo in [PageRankAlgo::Prpack, PageRankAlgo::Arpack] {
        let opts = undirected.with_algo(algo);

        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 2), (0, 2)], 4, true).unwrap();
        let pr = g.pagerank(None, .., &opts).unwrap();
        assert_all_close(&pr.scores, &[0.288717, 0.201989, 0.389109, 0.120186], EPS);
        assert_close(pr.value, 1.0, 1e-12);

        let g = Graph::from_edges(
            &[
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (2, 0),
                (3, 0),
                (3, 4),
                (3, 5),
                (3, 6),
                (3, 7),
                (4, 0),
                (5, 0),
                (6, 0),
                (7, 0),
            ],
            8,
            true,
        )
        .unwrap();
        let pr = g.pagerank(None, .., &opts).unwrap();
        assert_all_close(
            &pr.scores,
            &[
                0.336204, 0.0759047, 0.0759047, 0.205964, 0.0765056, 0.0765056, 0.0765056,
                0.0765056,
            ],
            EPS,
        );

        let s = star(11);
        let pr = s.pagerank(None, .., &opts).unwrap();
        assert_close(pr.scores[0], 0.46683, EPS);
        assert!(pr.scores[1..].iter().all(|&p| (p - 0.053317).abs() < EPS));

        // Personalized on vertex 1 with damping 0.5.
        let pr = s
            .personalized_pagerank_vs(None, 1, .., &opts.with_damping(0.5))
            .unwrap();
        assert_close(pr.scores[0], 1.0 / 3.0, EPS);
        assert_close(pr.scores[1], 0.516667, EPS);
        assert!(pr.scores[2..].iter().all(|&p| (p - 0.0166667).abs() < EPS));

        // Edgeless graphs: the stationary distribution is the reset one.
        let e = Graph::new(10, false);
        let pr = e.pagerank(None, .., &opts).unwrap();
        assert_all_close(&pr.scores, &[0.1; 10], 1e-12);
        let e = Graph::new(4, false);
        let pr = e
            .personalized_pagerank(None, Some(&[1.0, 2.0, 3.0, 4.0]), .., &opts)
            .unwrap();
        assert_all_close(&pr.scores, &[0.1, 0.2, 0.3, 0.4], 1e-12);

        // Zero weights everywhere behave like no edges at all.
        let k = complete(10);
        let pr = k.pagerank(Some(&[0.0; 45]), .., &opts).unwrap();
        assert_all_close(&pr.scores, &[0.1; 10], 1e-12);
    }
}

/// The ARPACK PageRank perturbs its starting vector with random numbers. The
/// default RNG is per thread, so seeding makes the result reproducible even
/// when the computations run concurrently.
#[test]
fn seeded_arpack_pagerank_is_reproducible_across_threads() {
    let run = || {
        igraph::rng::seed(137).unwrap();
        let g = Graph::erdos_renyi_game_gnm(60, 240, true, EdgeTypeSw::Simple, false).unwrap();
        let opts = PageRankOptions::default().with_algo(PageRankAlgo::Arpack);
        (g.edge_list(), g.pagerank(None, .., &opts).unwrap().scores)
    };
    let handles: Vec<_> = (0..4).map(|_| std::thread::spawn(run)).collect();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    for r in &results[1..] {
        assert_eq!(r, &results[0]);
    }
    let prpack = Graph::from_edges(&results[0].0, 60, true)
        .unwrap()
        .pagerank(None, .., &PageRankOptions::default())
        .unwrap();
    assert_all_close(&results[0].1, &prpack.scores, 1e-8);
}

// ---------------------------------------------------------------------------
// Eigenvector centrality and HITS
// ---------------------------------------------------------------------------

#[test]
fn eigenvector_centrality_complete_graph_and_star() {
    let k5 = complete(5);
    let ec = k5.eigenvector_centrality(NeighborMode::All, None).unwrap();
    assert_close(ec.value, 4.0, 1e-9);
    assert_all_close(&ec.scores, &[1.0; 5], 1e-9);

    // Unweighted star with k leaves: eigenvalue sqrt(k), leaves 1/sqrt(k).
    let s = star(10);
    let ec = s.eigenvector_centrality(NeighborMode::All, None).unwrap();
    assert_close(ec.value, 3.0, 1e-9);
    assert_close(ec.scores[0], 1.0, 1e-12);
    for &x in &ec.scores[1..] {
        assert_close(x, 1.0 / 3.0, 1e-9);
    }
}

#[test]
fn eigenvector_centrality_weighted_star_example() {
    // igraph's examples/simple/eigenvector_centrality.c
    let s = star(10);
    let w: Vec<f64> = (1..10).map(f64::from).collect();
    let ec = s
        .eigenvector_centrality(NeighborMode::Out, Some(&w))
        .unwrap();
    assert_close(ec.value, 16.8819, 1e-4);
    let expected = [
        1.0, 0.0592349, 0.11847, 0.177705, 0.23694, 0.296174, 0.355409, 0.414644, 0.473879,
        0.533114,
    ];
    assert_all_close(&ec.scores, &expected, 1e-6);
}

#[test]
fn eigenvector_centrality_of_a_dag_has_zero_eigenvalue() {
    let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    igraph::error::take_warnings();
    let ec = g.eigenvector_centrality(NeighborMode::Out, None).unwrap();
    assert_close(ec.value, 0.0, 1e-12);
    // With `Out` (left eigenvector) the sinks share the score, the others are 0.
    assert_eq!(ec.scores, vec![0.0, 0.0, 1.0]);
    assert!(
        igraph::error::take_warnings()
            .iter()
            .any(|w| w.contains("acyclic"))
    );
}

#[test]
fn eigenvector_centrality_of_a_disconnected_graph_warns() {
    // A triangle and a disjoint edge: the component with the larger
    // eigenvalue (2 vs 1) takes all the mass.
    let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 2), (3, 4)], 5, false).unwrap();
    igraph::error::take_warnings();
    let ec = g.eigenvector_centrality(NeighborMode::All, None).unwrap();
    assert_close(ec.value, 2.0, 1e-9);
    assert_all_close(&chop_zeros(&ec.scores), &[1.0, 1.0, 1.0, 0.0, 0.0], 1e-9);
    assert!(!igraph::error::take_warnings().is_empty());
    // Per-component computation gives the meaningful answer: each component
    // is vertex-transitive, so all its vertices get score 1.
    let comps = g.connected_components(Connectedness::Weak).unwrap();
    assert_eq!(comps.count, 2);
    for c in 0..comps.count {
        let sub = g
            .induced_subgraph(comps.members(c), SubgraphImplementation::Auto)
            .unwrap();
        let ec = sub.eigenvector_centrality(NeighborMode::All, None).unwrap();
        assert_all_close(&ec.scores, &vec![1.0; comps.sizes[c]], 1e-9);
    }
    assert!(igraph::error::take_warnings().is_empty());
}

#[test]
fn hubs_and_authorities() {
    // On undirected graphs HITS reduces to eigenvector centrality.
    let g = karate();
    let hits = g.hub_and_authority_scores(None).unwrap();
    let ec = g.eigenvector_centrality(NeighborMode::All, None).unwrap();
    assert_all_close(&hits.hubs, &ec.scores, 1e-6);
    assert_all_close(&hits.authorities, &ec.scores, 1e-6);
    assert_close(hits.value, ec.value * ec.value, 1e-6);

    // Singleton with three loops (igraph unit test): value 9.
    let g = Graph::from_edges(&[(0, 0), (0, 0), (0, 0)], 1, true).unwrap();
    let hits = g.hub_and_authority_scores(None).unwrap();
    assert_eq!((hits.hubs, hits.authorities), (vec![1.0], vec![1.0]));
    assert_close(hits.value, 9.0, 1e-9);

    // Null graph.
    let hits = Graph::new(0, true).hub_and_authority_scores(None).unwrap();
    assert!(hits.hubs.is_empty() && hits.authorities.is_empty());
}

#[test]
fn hits_consistency_relations() {
    // h ∝ A a and a ∝ Aᵀ h on a small directed "web".
    let edges = [
        (0, 3),
        (0, 4),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (3, 4),
        (5, 3),
    ];
    let g = Graph::from_edges(&edges, 6, true).unwrap();
    let hits = g.hub_and_authority_scores(None).unwrap();
    let mut ah = [0.0; 6];
    let mut ata = [0.0; 6];
    for &(u, v) in &edges {
        ah[u as usize] += hits.authorities[v as usize];
        ata[v as usize] += hits.hubs[u as usize];
    }
    let mh = ah.iter().cloned().fold(0.0, f64::max);
    let ma = ata.iter().cloned().fold(0.0, f64::max);
    for i in 0..6 {
        assert_close(ah[i] / mh, hits.hubs[i], 1e-6);
        assert_close(ata[i] / ma, hits.authorities[i], 1e-6);
    }
    assert_eq!(argmax(&hits.authorities), 4);
    assert_eq!(argmax(&hits.hubs), 1);
}

#[test]
fn hits_igraph_unit_test_values() {
    // igraph's tests/unit/hub_and_authority.{c,out}.
    let chop = |v: &[f64]| -> Vec<f64> {
        v.iter()
            .map(|&x| if x.abs() < 1e-9 { 0.0 } else { x })
            .collect()
    };

    // Singleton with one loop: value 1.
    let g = Graph::from_edges(&[(0, 0)], 1, true).unwrap();
    let hits = g.hub_and_authority_scores(None).unwrap();
    assert_eq!((hits.hubs, hits.authorities), (vec![1.0], vec![1.0]));
    assert_close(hits.value, 1.0, 1e-12);

    // No edges: all-ones scores and a zero eigenvalue (with a warning).
    igraph::error::take_warnings();
    let hits = Graph::new(3, true).hub_and_authority_scores(None).unwrap();
    assert_eq!(hits.hubs, vec![1.0; 3]);
    assert_eq!(hits.authorities, vec![1.0; 3]);
    assert_eq!(hits.value, 0.0);
    assert!(!igraph::error::take_warnings().is_empty());

    // Weighted example from the Stanford IR book (with max-scaling).
    let edges = [
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 2),
        (2, 3),
        (3, 3),
        (3, 4),
        (4, 6),
        (5, 5),
        (5, 6),
        (6, 3),
        (6, 4),
        (6, 6),
    ];
    let w = [
        1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0,
    ];
    let g = Graph::from_edges(&edges, 7, true).unwrap();
    let hits = g.hub_and_authority_scores(Some(&w)).unwrap();
    assert_all_close(
        &hits.hubs,
        &[0.100055, 0.109548, 0.944987, 0.5126, 0.10588, 0.115926, 1.0],
        EPS,
    );
    assert_all_close(
        &hits.authorities,
        &[
            0.214644, 0.0248828, 0.262253, 1.0, 0.343572, 0.0263314, 0.277521,
        ],
        EPS,
    );
    assert_close(hits.value, 11.5396, 1e-4);

    // Undirected graph with a self-loop and a multi-edge: self-loops count
    // twice, and the result is the eigenvector centrality.
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 2), (2, 3), (2, 3)], 4, false).unwrap();
    igraph::error::take_warnings();
    let hits = g.hub_and_authority_scores(None).unwrap();
    let expected = [0.0906902, 0.314507, 1.0, 0.576713];
    assert_all_close(&chop(&hits.hubs), &expected, EPS);
    assert_all_close(&chop(&hits.authorities), &expected, EPS);
    assert_close(hits.value, 12.0266, 1e-4);
    // igraph warns that hub/authority scores of undirected graphs are
    // eigenvector centralities.
    assert!(
        igraph::error::take_warnings()
            .iter()
            .any(|w| w.contains("undirected"))
    );
    let ec = g.eigenvector_centrality(NeighborMode::All, None).unwrap();
    assert_all_close(&hits.hubs, &ec.scores, 1e-9);
    assert_close(hits.value, ec.value * ec.value, 1e-9);

    // Degenerate example: the leading eigenvalue is the golden ratio squared.
    let g = Graph::from_edges(&[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 0)], 4, true).unwrap();
    let hits = g.hub_and_authority_scores(None).unwrap();
    let phi = (1.0 + 5f64.sqrt()) / 2.0;
    assert_close(hits.value, phi * phi, 1e-9);

    // Invalid weight vector.
    let g = Graph::from_edges(&[(0, 2), (1, 2)], 3, true).unwrap();
    let err = g.hub_and_authority_scores(Some(&[1.0; 3])).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

/// igraph 1.0.1 fixed the threshold of the "too many zero scores" HITS
/// warning: igraph 1.0.0 warned as soon as a single hub score was zero (on
/// graphs with at least 10 vertices), 1.0.1 only when more than 30% are.
#[test]
fn hits_zero_score_warning_threshold_igraph_1_0_1() {
    let version = igraph::misc::version();
    assert!(
        version.triple() >= (1, 0, 1),
        "igraph {version} was loaded at run time, but the crate targets 1.0.1: \
         check that LD_LIBRARY_PATH does not point to an older build"
    );
    // igraph treats |x| < 10 ε as zero.
    let zeros = |v: &[f64]| v.iter().filter(|x| x.abs() < 10.0 * f64::EPSILON).count();
    // A complete directed graph on `10 - sinks` vertices plus `sinks` sinks
    // reached from vertex 0: exactly the sinks have zero hub scores (every
    // vertex has a positive authority score). igraph warns when *more than*
    // ⌊0.3 · 10⌋ = 3 hub scores are zero.
    for (sinks, warns) in [(1, false), (3, false), (4, true)] {
        let k = 10 - sinks;
        let mut g = Graph::full(k, true, false).unwrap();
        g.add_vertices(sinks).unwrap();
        let edges: Vec<(i64, i64)> = (k..10).map(|s| (0, s as i64)).collect();
        g.add_edges(&edges).unwrap();
        igraph::error::take_warnings();
        let hits = g.hub_and_authority_scores(None).unwrap();
        assert_eq!(zeros(&hits.hubs), sinks);
        assert!(hits.hubs[k..].iter().all(|h| h.abs() < 10.0 * f64::EPSILON));
        assert!(hits.authorities.iter().all(|&a| a > 1e-6));
        let warnings = igraph::error::take_warnings();
        assert_eq!(
            warnings.iter().any(|w| w.contains("More than 30%")),
            warns,
            "{sinks} sinks: {warnings:?}"
        );
    }

    // An out-star on 11 vertices: 10 of 11 hub scores are zero, which is
    // well above 30%, so igraph warns that the result may not be meaningful.
    let g = Graph::star(11, StarMode::Out, 0).unwrap();
    let hits = g.hub_and_authority_scores(None).unwrap();
    let mut expected = vec![0.0; 11];
    expected[0] = 1.0;
    assert_eq!(chop_zeros(&hits.hubs), expected);
    assert_eq!(hits.authorities[0], 0.0);
    assert!(
        hits.authorities[1..]
            .iter()
            .all(|&a| (a - 1.0).abs() < 1e-12)
    );
    let warnings = igraph::error::take_warnings();
    assert!(
        warnings.iter().any(|w| w.contains("More than 30%")),
        "{warnings:?}"
    );
}

fn chop_zeros(v: &[f64]) -> Vec<f64> {
    v.iter()
        .map(|&x| if x.abs() < 1e-12 { 0.0 } else { x })
        .collect()
}

// ---------------------------------------------------------------------------
// Constraint and convergence degree
// ---------------------------------------------------------------------------

#[test]
fn constraint_igraph_unit_test_values() {
    assert!(
        Graph::new(0, false)
            .constraint(.., None)
            .unwrap()
            .is_empty()
    );
    assert!(Graph::new(1, false).constraint(.., None).unwrap()[0].is_nan());

    let line = path(4);
    assert_eq!(line.constraint(.., None).unwrap(), vec![1.0, 0.5, 0.5, 1.0]);

    let k4 = complete(4); // edges (0,1)(0,2)(0,3)(1,2)(1,3)(2,3)
    let line_w = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    assert_all_close(
        &k4.constraint(.., Some(&line_w)).unwrap(),
        &[1.25, 0.5625, 0.5625, 1.25],
        1e-12,
    );
    let c = k4.constraint(.., None).unwrap();
    assert_all_close(&c, &[0.925926; 4], EPS);
    assert_all_close(&k4.constraint(.., Some(&[1.0; 6])).unwrap(), &c, 1e-12);

    // Two 4-cliques bridged through vertex 4: the broker has the lowest constraint.
    let hole = Graph::from_edges(
        &[
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (5, 7),
            (5, 8),
            (6, 7),
            (6, 8),
            (7, 8),
        ],
        9,
        false,
    )
    .unwrap();
    let c = hole.constraint(.., None).unwrap();
    assert_all_close(
        &c,
        &[
            0.865741, 0.865741, 0.865741, 0.583333, 0.5, 0.583333, 0.865741, 0.865741, 0.865741,
        ],
        EPS,
    );
    assert_eq!(hole.constraint(4, None).unwrap(), vec![0.5]);
}

#[test]
fn convergence_degree_igraph_unit_test_values() {
    let g = Graph::from_edges(
        &[
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (3, 4),
            (4, 5),
            (4, 6),
            (5, 6),
        ],
        7,
        false,
    )
    .unwrap();
    let cd = g.convergence_degree().unwrap();
    assert_all_close(
        &cd.result,
        &[0.0, 0.0, 0.6, 0.0, 0.6, 0.6, 0.1429, 0.6667, 0.6667, 0.0],
        1e-4,
    );
    assert_eq!(cd.ins.len(), g.ecount());
    assert_eq!(cd.outs.len(), g.ecount());

    let g = Graph::from_edges(&[(1, 0), (2, 0), (3, 0), (4, 0), (0, 5)], 6, true).unwrap();
    let cd = g.convergence_degree().unwrap();
    assert_all_close(
        &cd.result,
        &[-1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0],
        1e-4,
    );
    // Directed case: result = (ins - outs) / (ins + outs).
    for i in 0..g.ecount() {
        let (a, b) = (cd.ins[i], cd.outs[i]);
        assert_close(cd.result[i], (a - b) / (a + b), 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Centralization
// ---------------------------------------------------------------------------

#[test]
fn star_is_maximally_centralized() {
    // igraph's examples/simple/centralization.c
    let s = star(10);
    let deg = s
        .centralization_degree(NeighborMode::All, Loops::None, true)
        .unwrap();
    assert_close(deg.centralization, 1.0, 1e-12);
    let btw = s.centralization_betweenness(false, true).unwrap();
    assert_close(btw.centralization, 1.0, 1e-12);
    let clo = s.centralization_closeness(NeighborMode::All, true).unwrap();
    assert_close(clo.centralization, 1.0, 1e-12);

    let single = Graph::from_edges(&[(0, 1)], 10, false).unwrap();
    let eig = single
        .centralization_eigenvector_centrality(NeighborMode::All, true)
        .unwrap();
    assert_close(eig.centralization, 1.0, 1e-9);
    assert_close(eig.value, 1.0, 1e-9);
    assert_eq!(eig.theoretical_max, 8.0);
}

#[test]
fn centralization_is_consistent_with_its_parts() {
    let g = karate();
    let deg = g
        .centralization_degree(NeighborMode::All, Loops::None, false)
        .unwrap();
    let d: Vec<f64> = g
        .degree(.., NeighborMode::All, Loops::None)
        .unwrap()
        .into_iter()
        .map(|x| x as f64)
        .collect();
    assert_eq!(deg.scores, d);
    assert_close(deg.centralization, centralization(&d, 0.0, false), 1e-12);
    let normalized = g
        .centralization_degree(NeighborMode::All, Loops::None, true)
        .unwrap();
    assert_close(
        normalized.centralization,
        deg.centralization / deg.theoretical_max,
        1e-12,
    );
    assert_close(normalized.centralization, 0.4000, 1e-3);

    let btw = g.centralization_betweenness(false, false).unwrap();
    assert_eq!(btw.scores, g.betweenness(None, .., false, false).unwrap());
    let clo = g
        .centralization_closeness(NeighborMode::All, false)
        .unwrap();
    assert_eq!(
        clo.scores,
        g.closeness(.., NeighborMode::All, None, true).unwrap()
    );
    let eig = g
        .centralization_eigenvector_centrality(NeighborMode::All, true)
        .unwrap();
    assert_close(
        eig.centralization,
        centralization(&eig.scores, eig.theoretical_max, true),
        1e-12,
    );

    // Normalized centralizations lie in [0, 1].
    for c in [
        g.centralization_degree(NeighborMode::All, Loops::None, true)
            .unwrap()
            .centralization,
        g.centralization_betweenness(false, true)
            .unwrap()
            .centralization,
        g.centralization_closeness(NeighborMode::All, true)
            .unwrap()
            .centralization,
        eig.centralization,
    ] {
        assert!((0.0..=1.0).contains(&c), "{c}");
    }
}

#[test]
fn theoretical_maxima_graph_and_free_functions_agree() {
    let g = karate();
    let n = g.vcount();
    assert_eq!(
        g.centralization_degree_tmax(NeighborMode::All, Loops::None)
            .unwrap(),
        centralization_degree_tmax(n, NeighborMode::All, Loops::None).unwrap()
    );
    assert_eq!(
        g.centralization_betweenness_tmax(false).unwrap(),
        centralization_betweenness_tmax(n, false).unwrap()
    );
    assert_eq!(
        g.centralization_closeness_tmax(NeighborMode::All).unwrap(),
        centralization_closeness_tmax(n, NeighborMode::All).unwrap()
    );
    assert_eq!(
        g.centralization_eigenvector_centrality_tmax(NeighborMode::All)
            .unwrap(),
        centralization_eigenvector_centrality_tmax(n, NeighborMode::All).unwrap()
    );

    // Closed forms.
    let nf = n as f64;
    assert_eq!(
        centralization_degree_tmax(n, NeighborMode::All, Loops::None).unwrap(),
        (nf - 1.0) * (nf - 2.0)
    );
    assert_eq!(
        centralization_degree_tmax(n, NeighborMode::Out, Loops::None).unwrap(),
        (nf - 1.0) * (nf - 1.0)
    );
    assert_eq!(
        centralization_betweenness_tmax(n, false).unwrap(),
        (nf - 1.0) * (nf - 1.0) * (nf - 2.0) / 2.0
    );
    assert_eq!(
        centralization_betweenness_tmax(n, true).unwrap(),
        (nf - 1.0) * (nf - 1.0) * (nf - 2.0)
    );
    assert!(
        centralization_degree_tmax(0, NeighborMode::All, Loops::None)
            .unwrap()
            .is_nan()
    );

    // The tmax is precisely the centralization of the star.
    let s = star(12);
    let deg = s
        .centralization_degree(NeighborMode::All, Loops::None, false)
        .unwrap();
    assert_eq!(deg.centralization, deg.theoretical_max);
    let btw = s.centralization_betweenness(false, false).unwrap();
    assert_close(btw.centralization, btw.theoretical_max, 1e-9);
}

#[test]
fn centralization_free_function() {
    assert_eq!(centralization(&[5.0, 5.0, 5.0], 10.0, true), 0.0);
    assert_eq!(centralization(&[4.0, 0.0, 1.0], 0.0, false), 7.0);
    assert_eq!(centralization(&[4.0, 0.0, 1.0], 14.0, true), 0.5);
    // No scores at all: undefined.
    assert!(centralization(&[], 1.0, true).is_nan());
}

#[test]
fn edge_betweenness_subset_with_list_selectors() {
    // Several list selectors built in a row (sources, targets and edges) must
    // all stay valid during the C call.
    // Directed path 0 -> 1 -> 2 -> 3 -> 4.
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4)], 5, true).unwrap();
    // Paths 0->3, 0->4, 1->3, 1->4: edge 2->3 is used by all four,
    // edge 0->1 by the two starting at 0, edge 3->4 by the two ending at 4.
    let eb = g
        .edge_betweenness_subset(None, &[0, 1], &[3, 4], &[0, 2, 3], true)
        .unwrap();
    assert_eq!(eb, vec![2.0, 4.0, 2.0]);
    // Weighted with unit weights: identical.
    let w = [1.0; 4];
    assert_eq!(
        g.edge_betweenness_subset(Some(&w), &[0, 1], &[3, 4], &[0, 2, 3], true)
            .unwrap(),
        eb
    );
    // Vertex version: 1 is inner on the paths from 0, 3 on those to 4.
    let vb = g
        .betweenness_subset(None, &[0, 1], &[3, 4], &[1, 2, 3], true)
        .unwrap();
    assert_eq!(vb, vec![2.0, 4.0, 2.0]);
    // Invalid edge id in the list.
    let err = g
        .edge_betweenness_subset(None, &[0], &[4], &[0, 9], true)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidEdgeId);
}

// ---------------------------------------------------------------------------
// Local scan statistics
// ---------------------------------------------------------------------------

/// igraph's `g_lm`: directed, with a loop and a multi-edge.
fn g_lm(directed: bool) -> Graph {
    Graph::from_edges(
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
        directed,
    )
    .unwrap()
}

#[test]
fn local_scan_igraph_unit_test_values() {
    let g = g_lm(true);
    assert_eq!(
        g.local_scan_k_ecount(0, None, NeighborMode::In).unwrap(),
        vec![1.0, 2.0, 1.0, 2.0, 2.0, 0.0]
    );
    assert_eq!(
        g.local_scan_0(None, NeighborMode::In).unwrap(),
        vec![1.0, 2.0, 1.0, 2.0, 2.0, 0.0]
    );
    let k1 = vec![2.0, 2.0, 2.0, 3.0, 2.0, 0.0];
    assert_eq!(
        g.local_scan_k_ecount(1, None, NeighborMode::In).unwrap(),
        k1
    );
    assert_eq!(g.local_scan_1_ecount(None, NeighborMode::In).unwrap(), k1);
    let k1_all = vec![4.0, 3.0, 3.0, 5.0, 2.0, 0.0];
    assert_eq!(
        g.local_scan_1_ecount(None, NeighborMode::All).unwrap(),
        k1_all
    );

    let u = g_lm(false);
    assert_eq!(
        u.local_scan_1_ecount(None, NeighborMode::In).unwrap(),
        k1_all
    );
    let w = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let ws = u.local_scan_1_ecount(Some(&w), NeighborMode::In).unwrap();
    assert_all_close(&ws, &[0.3, 0.2, 0.7, 1.8, 1.1, 0.0], 1e-12);
    assert_all_close(
        &u.local_scan_k_ecount(1, Some(&w), NeighborMode::All)
            .unwrap(),
        &ws,
        1e-12,
    );

    assert!(
        Graph::new(0, false)
            .local_scan_k_ecount(2, None, NeighborMode::All)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        Graph::new(1, false)
            .local_scan_k_ecount(2, None, NeighborMode::All)
            .unwrap(),
        vec![0.0]
    );
}

#[test]
fn scan_1_is_degree_plus_triangles_on_simple_graphs() {
    let g = karate();
    let deg = g.local_scan_0(None, NeighborMode::All).unwrap();
    let tri = g.count_adjacent_triangles(..).unwrap();
    let s1 = g.local_scan_1_ecount(None, NeighborMode::All).unwrap();
    for v in 0..34 {
        assert_eq!(s1[v], deg[v] + tri[v]);
    }
    // A large k covers the whole (connected) graph.
    let all = g.local_scan_k_ecount(10, None, NeighborMode::All).unwrap();
    assert!(all.iter().all(|&x| x == 78.0));
}

#[test]
#[allow(deprecated)] // local_scan_neighborhood_ecount is deprecated in igraph
fn local_scan_subset_and_neighborhood() {
    let g = g_lm(true);
    let empty: Vec<Vec<i64>> = vec![vec![]; 6];
    assert_eq!(
        g.local_scan_subset_ecount(None, &empty).unwrap(),
        vec![0.0; 6]
    );

    // Each vertex's closed in-neighborhood reproduces the k = 1 statistic.
    let hoods: Vec<Vec<i64>> = (0..6)
        .map(|v| {
            let mut h = g.neighbors(v, NeighborMode::In).unwrap();
            h.push(v);
            h.sort();
            h.dedup();
            h
        })
        .collect();
    let expected = g.local_scan_1_ecount(None, NeighborMode::In).unwrap();
    assert_eq!(g.local_scan_subset_ecount(None, &hoods).unwrap(), expected);
    assert_eq!(
        g.local_scan_neighborhood_ecount(None, &hoods).unwrap(),
        expected
    );

    // Subsets can be arbitrary and of any number.
    let k = karate();
    let everything: Vec<i64> = (0..34).collect();
    assert_eq!(
        k.local_scan_subset_ecount(None, &[everything.as_slice()])
            .unwrap(),
        vec![78.0]
    );

    // neighborhood_ecount wants one set per vertex.
    let err = g
        .local_scan_neighborhood_ecount(None, &hoods[..2])
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let err = g
        .local_scan_subset_ecount(None, &[vec![0, 99]])
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

#[test]
fn them_statistics_with_identical_graphs_equal_us_statistics() {
    let g = karate();
    let h = g.clone();
    assert_eq!(
        g.local_scan_0_them(&h, None, NeighborMode::All).unwrap(),
        g.local_scan_0(None, NeighborMode::All).unwrap()
    );
    assert_eq!(
        g.local_scan_1_ecount_them(&h, None, NeighborMode::All)
            .unwrap(),
        g.local_scan_1_ecount(None, NeighborMode::All).unwrap()
    );
    for k in 0..4 {
        assert_eq!(
            g.local_scan_k_ecount_them(&h, k, None, NeighborMode::All)
                .unwrap(),
            g.local_scan_k_ecount(k, None, NeighborMode::All).unwrap()
        );
    }
}

#[test]
fn them_statistics_errors() {
    let g = path(4);
    let err = g
        .local_scan_0_them(&path(5), None, NeighborMode::All)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let directed = Graph::from_edges(&[(0, 1)], 4, true).unwrap();
    let err = g
        .local_scan_1_ecount_them(&directed, None, NeighborMode::All)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let err = g
        .local_scan_k_ecount_them(&cycle(4), 2, Some(&[1.0; 3]), NeighborMode::All)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

// ---------------------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------------------

#[test]
fn invalid_arguments_are_reported() {
    let g = karate();
    let err = g.closeness(34, NeighborMode::All, None, false).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    let err = g
        .betweenness(None, [0, 50].as_slice(), false, false)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    let err = g.edge_betweenness(None, 500, false, false).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidEdgeId);
    let err = g
        .harmonic_centrality(.., NeighborMode::All, Some(&[1.0]), false)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let err = g.constraint(-1, None).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    let err = g.betweenness_subset(None, 99, .., .., false).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    let nan = vec![f64::NAN; g.ecount()];
    let err = g.betweenness(Some(&nan), .., false, false).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

// ---------------------------------------------------------------------------
// Use cases
// ---------------------------------------------------------------------------

/// Story: in Zachary's karate club, the instructor (vertex 0) and the
/// administrator (vertex 33) lead the two factions that split the club.
/// Different centralities agree that they are the key players, and Burt's
/// constraint shows that they are the least constrained brokers, while
/// peripheral members embedded in a single faction are highly constrained.
#[test]
fn use_case_karate_club_leaders_and_brokers() {
    let g = karate();
    let top2 = |v: &[f64]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
        let mut t = vec![idx[0], idx[1]];
        t.sort();
        t
    };
    let leaders = vec![0, 33];
    assert_eq!(
        top2(&g.betweenness(None, .., false, true).unwrap()),
        leaders
    );
    assert_eq!(
        top2(
            &g.pagerank(None, .., &PageRankOptions::default())
                .unwrap()
                .scores
        ),
        leaders
    );
    assert_eq!(
        top2(
            &g.harmonic_centrality(.., NeighborMode::All, None, true)
                .unwrap()
        ),
        leaders
    );
    let ec = g.eigenvector_centrality(NeighborMode::All, None).unwrap();
    assert_eq!(argmax(&ec.scores), 33);

    // Least constrained = best positioned brokers.
    let c = g.constraint(.., None).unwrap();
    let neg: Vec<f64> = c.iter().map(|x| -x).collect();
    assert_eq!(top2(&neg), leaders);

    // "Who is closest to the instructor?" Personalized PageRank restarting at
    // vertex 0 shifts the probability mass towards the instructor's circle
    // and away from the administrator.
    let pr = g
        .pagerank(None, .., &PageRankOptions::default())
        .unwrap()
        .scores;
    let ppr = g
        .personalized_pagerank_vs(None, 0, .., &PageRankOptions::default())
        .unwrap()
        .scores;
    let circle = g.neighbors(0, NeighborMode::All).unwrap();
    let mass = |v: &[f64]| circle.iter().map(|&u| v[u as usize]).sum::<f64>();
    assert!(mass(&ppr) > mass(&pr));
    assert!(ppr[33] < pr[33]);
    let mut idx: Vec<usize> = (1..34).collect();
    idx.sort_by(|&a, &b| ppr[b].partial_cmp(&ppr[a]).unwrap());
    assert!(circle.contains(&(idx[0] as i64)));
}

/// Story: two snapshots of a communication network. In the second one a
/// group of five peripheral members, who never talked to each other, suddenly
/// forms a clique. The scan statistic (edges inside each vertex's closed
/// neighborhood) jumps for exactly those members, flagging the anomaly. The
/// "them" statistic, which counts the new edges inside the *old*
/// neighborhoods, tells that the jump is not a mere intensification of old
/// relations: it comes from brand new contacts.
#[test]
fn use_case_scan_statistics_detect_an_emerging_clique() {
    let before = karate();
    let group = [11, 12, 14, 15, 16];
    let mut edges = common::KARATE_EDGES.to_vec();
    for (i, &a) in group.iter().enumerate() {
        for &b in &group[i + 1..] {
            assert!(!edges.contains(&(a, b)) && !edges.contains(&(b, a)));
            edges.push((a, b));
        }
    }
    let after = Graph::from_edges(&edges, 34, false).unwrap();

    let old = before.local_scan_1_ecount(None, NeighborMode::All).unwrap();
    let new = after.local_scan_1_ecount(None, NeighborMode::All).unwrap();
    let growth: Vec<f64> = new.iter().zip(&old).map(|(n, o)| n - o).collect();
    assert!(growth.iter().all(|&x| x >= 0.0));
    let mut idx: Vec<usize> = (0..34).collect();
    idx.sort_by(|&a, &b| growth[b].partial_cmp(&growth[a]).unwrap());
    let mut flagged: Vec<i64> = idx[..5].iter().map(|&v| v as i64).collect();
    flagged.sort();
    assert_eq!(flagged, group);

    // New edges inside old neighborhoods only: for the group this is unchanged.
    let them = before
        .local_scan_1_ecount_them(&after, None, NeighborMode::All)
        .unwrap();
    for &v in &group {
        assert_eq!(them[v as usize], old[v as usize]);
    }
    // And in general "them" in old neighborhoods never exceeds "us" in new ones.
    assert!(them.iter().zip(&new).all(|(t, n)| t <= n));
}

// ---------------------------------------------------------------------------
// Audit regressions: inputs igraph 1.0.1 does not validate
// ---------------------------------------------------------------------------

fn assert_invalid<T: std::fmt::Debug>(r: igraph::Result<T>) {
    assert_eq!(r.unwrap_err().kind(), ErrorKind::InvalidValue);
}

/// A directed 10-vertex graph: a cycle with a few chords.
fn directed10() -> Graph {
    let mut edges: Vec<(i64, i64)> = (0..10).map(|i| (i, (i + 1) % 10)).collect();
    edges.extend([(0, 5), (3, 7), (8, 2)]);
    Graph::from_edges(&edges, 10, true).unwrap()
}

#[test]
fn spectral_centralities_reject_non_finite_weights() {
    // Each of these used to make ARPACK abort the process (f2c STOP).
    let g = directed10();
    let m = g.ecount();
    for bad in [f64::NAN, f64::INFINITY] {
        let mut w = vec![1.0; m];
        w[0] = bad;
        assert_invalid(g.eigenvector_centrality(NeighborMode::All, Some(&w)));
        assert_invalid(g.eigenvector_centrality(NeighborMode::Out, Some(&w)));
        assert_invalid(g.hub_and_authority_scores(Some(&w)));
    }
}

#[test]
fn spectral_centralities_rescale_huge_weights() {
    // All weights f64::MAX used to overflow ARPACK's iterations (abort);
    // they are now rescaled, giving the unweighted scores and a scaled
    // eigenvalue.
    let g = directed10();
    let m = g.ecount();
    let huge = vec![f64::MAX; m];
    let plain = g.eigenvector_centrality(NeighborMode::All, None).unwrap();
    let big = g
        .eigenvector_centrality(NeighborMode::All, Some(&huge))
        .unwrap();
    assert_all_close(&plain.scores, &big.scores, EPS);
    assert!(big.value.is_infinite());

    let scale = 1e200;
    let big = g
        .eigenvector_centrality(NeighborMode::All, Some(&vec![scale; m]))
        .unwrap();
    assert_all_close(&plain.scores, &big.scores, EPS);
    assert_close(big.value / scale, plain.value, EPS);

    // HITS on the karate club with both directions of every edge: a unique
    // (non-degenerate) principal eigenvector.
    let edges: Vec<(i64, i64)> = common::KARATE_EDGES
        .into_iter()
        .flat_map(|(a, b)| [(a, b), (b, a)])
        .collect();
    let g = Graph::from_edges(&edges, 34, true).unwrap();
    let m = g.ecount();
    let plain = g.hub_and_authority_scores(Some(&vec![1.0; m])).unwrap();
    for scale in [1e100, f64::MAX] {
        let big = g.hub_and_authority_scores(Some(&vec![scale; m])).unwrap();
        assert_all_close(&plain.hubs, &big.hubs, EPS);
        assert_all_close(&plain.authorities, &big.authorities, EPS);
        if scale == 1e100 {
            assert_close(big.value / 1e200, plain.value, EPS);
        }
    }
}

#[test]
fn pagerank_rejects_nan_damping_and_non_finite_weights() {
    // With ARPACK these used to abort the process, with PRPACK they
    // silently returned NaN scores.
    let g = directed10();
    let m = g.ecount();
    for algo in [PageRankAlgo::Arpack, PageRankAlgo::Prpack] {
        let nan_damping = PageRankOptions {
            damping: f64::NAN,
            algo,
            ..PageRankOptions::default()
        };
        assert_invalid(g.pagerank(None, .., &nan_damping));
        assert_invalid(g.personalized_pagerank(None, None, .., &nan_damping));
        assert_invalid(g.personalized_pagerank_vs(None, 0, .., &nan_damping));
        let opts = PageRankOptions {
            algo,
            ..PageRankOptions::default()
        };
        for bad in [f64::NAN, f64::INFINITY] {
            let mut w = vec![1.0; m];
            w[0] = bad;
            assert_invalid(g.pagerank(Some(&w), .., &opts));
            assert_invalid(g.personalized_pagerank(Some(&w), None, .., &opts));
            assert_invalid(g.personalized_pagerank_vs(Some(&w), 0, .., &opts));
        }
        // Huge weights: PageRank does not depend on their scale.
        let plain = g.pagerank(None, .., &opts).unwrap();
        let big = g.pagerank(Some(&vec![f64::MAX; m]), .., &opts).unwrap();
        assert_all_close(&plain.scores, &big.scores, EPS);
    }
}

#[test]
fn rescaling_huge_weights_keeps_tiny_ones() {
    // Rescaling must not flush the small weights to zero: vertex 0's only
    // out-edge has weight 1e-100. Dividing all weights by the largest one
    // (1e300) would make it 1e-400, i.e. 0, turning vertex 0 into a sink
    // (and a subnormal instead would give infinite, then NaN, transition
    // probabilities). Every vertex but 2 has a single out-edge and vertex 2
    // has two of equal weight, so the weighted scores equal the unweighted
    // ones.
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3), (3, 0)], 4, true).unwrap();
    let huge = [1e-100, 1e300, 1e300, 1e300, 1e300];
    // (PRPACK only: igraph's ARPACK PageRank loses such a 1e-400 weight
    // ratio to overflow on its own, at any scale.)
    let opts = PageRankOptions {
        algo: PageRankAlgo::Prpack,
        ..PageRankOptions::default()
    };
    let big = g.pagerank(Some(&huge), .., &opts).unwrap();
    let unweighted = g.pagerank(None, .., &opts).unwrap();
    assert_all_close(&big.scores, &unweighted.scores, EPS);
}

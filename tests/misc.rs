//! Integration tests for `igraph::misc`: epidemics, spatial graphs,
//! non-graph utilities, LSAP, sampling, prefix-sum trees and handlers.

mod common;

use common::*;
use igraph::ffi;
use igraph::games::BarabasiOptions;
use igraph::misc::{self, Metric, PsumTree};
use igraph::prelude::*;
use std::{
    cell::{Cell, RefCell},
    cmp::Ordering,
    collections::BTreeSet,
    ops::ControlFlow,
    rc::Rc,
};

/// Zachary's karate club network, from igraph's collection of famous graphs.
fn zachary() -> Graph {
    Graph::famous("Zachary").unwrap()
}

fn sorted_edges(g: &Graph) -> Vec<(i64, i64)> {
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

fn edge_set(g: &Graph) -> BTreeSet<(i64, i64)> {
    sorted_edges(g).into_iter().collect()
}

/// `n` random points in the unit square, from the default RNG.
fn random_points(n: usize) -> Matrix {
    let rows: Vec<[f64; 2]> = (0..n)
        .map(|_| [rng::uniform01(), rng::uniform01()])
        .collect();
    Matrix::from_rows(&rows).unwrap()
}

// ---------------------------------------------------------------------------
// Linear sum assignment
// ---------------------------------------------------------------------------

#[test]
fn lsap_igraph_test_case() {
    // tests/unit/igraph_solve_lsap.c
    let cost = Matrix::from_rows(&[
        [9.0, 2.0, 7.0, 8.0],
        [6.0, 4.0, 3.0, 7.0],
        [5.0, 8.0, 1.0, 8.0],
        [7.0, 6.0, 9.0, 4.0],
    ])
    .unwrap();
    assert_eq!(misc::solve_lsap(&cost).unwrap(), vec![1, 0, 2, 3]);
    assert!(misc::solve_lsap(&Matrix::new()).unwrap().is_empty());
}

#[test]
fn lsap_rejects_non_square() {
    let m34 = Matrix::from_rows(&[
        [3.0, 3.0, 2.0, 3.0],
        [2.0, 3.0, 3.0, 3.0],
        [3.0, 2.0, 3.0, 3.0],
    ])
    .unwrap();
    assert_eq!(
        misc::solve_lsap(&m34).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        misc::solve_lsap(&m34.transposed()).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    let mut out = vec![];
    for p in permutations(n - 1) {
        for pos in 0..=p.len() {
            let mut q = p.clone();
            q.insert(pos, n - 1);
            out.push(q);
        }
    }
    out
}

#[test]
fn lsap_matches_brute_force_and_maximization_by_negation() {
    rng::seed(2024).unwrap();
    let n = 6;
    for _ in 0..5 {
        let rows: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..n).map(|_| rng::integer(0, 50) as f64).collect())
            .collect();
        let cost = Matrix::from_rows(&rows).unwrap();
        let total = |a: &[usize]| a.iter().enumerate().map(|(i, &j)| rows[i][j]).sum::<f64>();
        let perms = permutations(n);
        let best = perms.iter().map(|p| total(p)).fold(f64::INFINITY, f64::min);
        let worst = perms
            .iter()
            .map(|p| total(p))
            .fold(f64::NEG_INFINITY, f64::max);

        let sol: Vec<usize> = misc::solve_lsap(&cost)
            .unwrap()
            .iter()
            .map(|&j| j as usize)
            .collect();
        let mut seen = sol.clone();
        seen.sort();
        assert_eq!(
            seen,
            (0..n).collect::<Vec<_>>(),
            "the result is a permutation"
        );
        assert_eq!(total(&sol), best);

        let neg_rows: Vec<Vec<f64>> = rows
            .iter()
            .map(|r| r.iter().map(|x| -x).collect())
            .collect();
        let neg = Matrix::from_rows(&neg_rows).unwrap();
        let sol: Vec<usize> = misc::solve_lsap(&neg)
            .unwrap()
            .iter()
            .map(|&j| j as usize)
            .collect();
        assert_eq!(total(&sol), worst);
    }
}

// ---------------------------------------------------------------------------
// Non-graph utilities
// ---------------------------------------------------------------------------

#[test]
fn power_law_fit_recovers_the_exponent_of_synthetic_data() {
    rng::seed(42).unwrap();
    // Continuous power law with alpha = 2.5 and xmin = 1, by inverse transform.
    let alpha = 2.5;
    let sample: Vec<f64> = (0..20_000)
        .map(|_| (1.0 - rng::uniform01()).powf(-1.0 / (alpha - 1.0)))
        .collect();

    let fixed = misc::power_law_fit(&sample, Some(1.0), false).unwrap();
    assert!(fixed.continuous);
    assert_eq!(fixed.xmin, 1.0);
    assert_close(fixed.alpha, alpha, 0.05);
    assert!(fixed.log_likelihood < 0.0);
    assert!(fixed.ks_statistic >= 0.0 && fixed.ks_statistic < 0.05);
    assert_eq!(fixed.data.len(), sample.len());

    // With an estimated xmin the exponent is still close.
    let estimated = misc::power_law_fit(&sample, None, false).unwrap();
    assert!(estimated.xmin >= 1.0);
    assert_close(estimated.alpha, alpha, 0.2);

    // The p-value of a genuine power law is not small.
    let small: Vec<f64> = sample[..500].to_vec();
    let fit = misc::power_law_fit(&small, Some(1.0), false).unwrap();
    let p = fit.p_value(0.1).unwrap();
    assert!((0.0..=1.0).contains(&p));
    assert!(p > 0.05, "p = {p}");
    assert_eq!(
        fit.p_value(0.0).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn power_law_fit_is_discrete_for_integers_unless_forced() {
    let data = [
        1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 1.0, 2.0, 6.0, 1.0, 1.0, 10.0,
    ];
    assert!(
        !misc::power_law_fit(&data, Some(1.0), false)
            .unwrap()
            .continuous
    );
    assert!(
        misc::power_law_fit(&data, Some(1.0), true)
            .unwrap()
            .continuous
    );
    assert_eq!(
        misc::power_law_fit(&data, Some(-2.0), false)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn power_law_fit_of_barabasi_degrees_matches_igraph_example() {
    // examples/simple/igraph_power_law_fit.c and its .out file.
    rng::seed(42).unwrap();
    let g = Graph::barabasi_game(
        10_000,
        &BarabasiOptions {
            m: 2,
            algorithm: BarabasiAlgorithm::Bag,
            ..Default::default()
        },
    )
    .unwrap();
    let degrees: Vec<f64> = g
        .degree(.., NeighborMode::All, Loops::None)
        .unwrap()
        .into_iter()
        .map(|d| d as f64)
        .collect();
    let fit = misc::power_law_fit(&degrees, None, false).unwrap();
    assert!(!fit.continuous);
    assert_close(fit.alpha, 3.04393, 1e-5);
    assert_close(fit.xmin, 7.0, 1e-12);
    assert_close(fit.log_likelihood, -3103.87560, 1e-5);
}

#[test]
fn running_mean_igraph_cases() {
    // tests/unit/igraph_running_mean.c
    assert_eq!(
        misc::running_mean(&[], 0).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        misc::running_mean(&[], 1).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(misc::running_mean(&[1.0], 1).unwrap(), vec![1.0]);
    let data = [1.0, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(misc::running_mean(&data, 1).unwrap(), data.to_vec());
    assert_eq!(
        misc::running_mean(&data, 2).unwrap(),
        vec![1.5, 2.5, 3.5, 4.5]
    );
    assert_eq!(misc::running_mean(&data, 5).unwrap(), vec![3.0]);
    assert_eq!(
        misc::running_mean(&data, 6).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn random_sample_properties() {
    rng::seed(11).unwrap();
    let s = misc::random_sample(-5, 5, 11).unwrap();
    assert_eq!(
        s,
        (-5..=5).collect::<Vec<_>>(),
        "the whole interval, in order"
    );
    assert_eq!(misc::random_sample(7, 7, 1).unwrap(), vec![7]);
    assert!(misc::random_sample(0, 10, 0).unwrap().is_empty());

    let s = misc::random_sample(100, 1_000_000, 1000).unwrap();
    assert_eq!(s.len(), 1000);
    assert!(s.windows(2).all(|w| w[0] < w[1]));
    assert!(s.iter().all(|&x| (100..=1_000_000).contains(&x)));

    assert_eq!(
        misc::random_sample(5, 4, 1).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        misc::random_sample(0, 9, 11).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn tolerant_comparisons() {
    assert!(misc::almost_equals(1.0, 1.0 + 1e-10, 1e-8));
    assert!(!misc::almost_equals(1.0, 1.0 + 1e-6, 1e-8));
    assert!(misc::almost_equals(0.0, 0.0, 1e-8));
    assert!(!misc::almost_equals(f64::NAN, f64::NAN, 1.0));

    assert_eq!(misc::cmp_epsilon(1.0, 1.0 + 1e-10, 1e-8), Ordering::Equal);
    assert_eq!(misc::cmp_epsilon(1.0, 1.0 + 1e-6, 1e-8), Ordering::Less);
    assert_eq!(misc::cmp_epsilon(2.0, 1.0, 1e-8), Ordering::Greater);
    assert_eq!(
        misc::cmp_epsilon(1.0, 1.0 + 1e-15, 0.0),
        Ordering::Less,
        "eps = 0 is exact"
    );
    assert_eq!(
        misc::cmp_epsilon(f64::INFINITY, f64::INFINITY, 1e-8),
        Ordering::Equal
    );
    assert_eq!(
        misc::cmp_epsilon(f64::NEG_INFINITY, -1e308, 0.9),
        Ordering::Less
    );
    assert_ne!(misc::cmp_epsilon(f64::NAN, f64::NAN, 1.0), Ordering::Equal);
    assert_ne!(misc::cmp_epsilon(f64::NAN, 1.0, 1.0), Ordering::Equal);
}

#[test]
fn version_is_consistent() {
    let v = misc::version();
    // The crate targets igraph 1.0.1 (the soname of 1.0.x is libigraph.so.4).
    assert_eq!(v.triple(), (1, 0, 1));
    assert!(v.string.starts_with("1.0.1"), "{v}");
    assert_eq!(v.to_string(), v.string);
}

// ---------------------------------------------------------------------------
// SIR epidemics
// ---------------------------------------------------------------------------

fn check_sir_invariants(run: &misc::SirRun, n: i64) {
    assert!(!run.is_empty());
    assert_eq!(run.times.len(), run.susceptible.len());
    assert_eq!(run.times.len(), run.infected.len());
    assert_eq!(run.times.len(), run.recovered.len());
    assert_eq!(run.times[0], 0.0);
    assert_eq!(
        (run.susceptible[0], run.infected[0], run.recovered[0]),
        (n - 1, 1, 0)
    );
    assert!(
        run.times.windows(2).all(|w| w[0] <= w[1]),
        "time flows forward"
    );
    for (_, s, i, r) in run.states() {
        assert_eq!(s + i + r, n, "S + I + R is constant");
    }
    // S never increases, R never decreases, and each event changes one count by one.
    assert!(run.susceptible.windows(2).all(|w| w[1] <= w[0]));
    assert!(run.recovered.windows(2).all(|w| w[1] >= w[0]));
    for k in 1..run.len() {
        let infection = run.susceptible[k - 1] - run.susceptible[k] == 1
            && run.recovered[k] == run.recovered[k - 1];
        let recovery = run.recovered[k] - run.recovered[k - 1] == 1
            && run.susceptible[k] == run.susceptible[k - 1];
        assert!(
            infection ^ recovery,
            "event {k} is neither an infection nor a recovery"
        );
    }
    assert_eq!(*run.infected.last().unwrap(), 0, "the epidemic dies out");
    assert_eq!(run.final_size(), n - run.susceptible.last().unwrap());
    assert!(run.peak_infected() >= 1);
    assert_eq!(run.duration(), *run.times.last().unwrap());
    // Number of events = 2 * final_size - 1 (the patient zero was never "infected" by an event).
    assert_eq!(run.len() as i64, 2 * run.final_size());
}

#[test]
fn sir_counts_are_consistent_on_karate() {
    rng::seed(43).unwrap();
    let g = zachary();
    let runs = g.sir(0.5, 1.0, 50).unwrap();
    assert_eq!(runs.len(), 50);
    for run in &runs {
        check_sir_invariants(run, 34);
    }
}

#[test]
fn sir_small_cases_from_igraph_tests() {
    // tests/unit/igraph_sir.c
    rng::seed(43).unwrap();
    let single = Graph::new(1, false);
    for run in single.sir(0.1, 0.0001, 2).unwrap() {
        assert_eq!(run.susceptible, vec![0, 0]);
        assert_eq!(run.infected, vec![1, 0]);
        assert_eq!(run.recovered, vec![0, 1]);
    }
    let two = Graph::new(2, false);
    for run in two.sir(1.0, 1.0, 2).unwrap() {
        assert_eq!(run.susceptible, vec![1, 1]);
        assert_eq!(run.final_size(), 1, "nobody else can be infected");
    }
    // On the complete graph the epidemic is larger than on a path, on average.
    let mean_size = |g: &Graph| {
        let runs = g.sir(1.0, 1.0, 400).unwrap();
        runs.iter().map(|r| r.final_size() as f64).sum::<f64>() / runs.len() as f64
    };
    assert!(mean_size(&complete(5)) > mean_size(&path(5)));
}

#[test]
fn sir_errors() {
    let g = zachary();
    let multi = Graph::from_edges(&[(0, 1), (0, 1), (1, 2)], 3, false).unwrap();
    let with_loop = Graph::from_edges(&[(0, 0), (0, 1)], 2, false).unwrap();
    assert_eq!(
        multi.sir(1.0, 1.0, 1).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        with_loop.sir(1.0, 1.0, 1).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        Graph::new(0, false).sir(1.0, 1.0, 1).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.sir(-1.0, 1.0, 1).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.sir(1.0, 0.0, 1).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.sir(1.0, 1.0, 0).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn sir_on_directed_graph_warns_and_ignores_directions() {
    rng::seed(5).unwrap();
    igraph::error::take_warnings();
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, true).unwrap();
    let runs = g.sir(1.0, 1.0, 3).unwrap();
    for run in &runs {
        check_sir_invariants(run, 4);
    }
    assert!(!igraph::error::take_warnings().is_empty());
}

#[test]
fn sir_is_reproducible_with_a_seed() {
    let g = zachary();
    rng::seed(99).unwrap();
    let a = g.sir(0.7, 1.0, 5).unwrap();
    rng::seed(99).unwrap();
    let b = g.sir(0.7, 1.0, 5).unwrap();
    assert_eq!(a, b);
}

// ---------------------------------------------------------------------------
// Spatial graphs
// ---------------------------------------------------------------------------

#[test]
fn delaunay_lattice_matches_igraph_output() {
    // tests/unit/igraph_delaunay_graph.c, "3x3 square lattice".
    let pts: Vec<[f64; 2]> = (0..9).map(|i| [(i / 3) as f64, (i % 3) as f64]).collect();
    let g = Graph::delaunay_graph(&Matrix::from_rows(&pts).unwrap()).unwrap();
    let expected = vec![
        (0, 1),
        (0, 3),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 5),
        (3, 4),
        (3, 6),
        (3, 7),
        (4, 5),
        (4, 7),
        (5, 7),
        (5, 8),
        (6, 7),
        (7, 8),
    ];
    assert!(!g.is_directed());
    assert_eq!(sorted_edges(&g), expected);
}

#[test]
fn delaunay_degenerate_and_invalid_inputs() {
    // Triangle with one subdivided edge.
    let pts = Matrix::from_rows(&[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [1.0, 0.0]]).unwrap();
    assert_eq!(
        sorted_edges(&Graph::delaunay_graph(&pts).unwrap()),
        vec![(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    );
    // Collinear points and duplicates are rejected.
    let line = Matrix::from_rows(&[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]).unwrap();
    assert_eq!(
        Graph::delaunay_graph(&line).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    let dup = Matrix::from_rows(&[[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).unwrap();
    assert_eq!(
        Graph::delaunay_graph(&dup).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    let nan = Matrix::from_rows(&[[0.0, 0.0], [1.0, f64::NAN], [0.0, 1.0]]).unwrap();
    assert_eq!(
        Graph::delaunay_graph(&nan).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    // Trivial cases.
    let one = Matrix::from_rows(&[[0.5, 0.5]]).unwrap();
    let g = Graph::delaunay_graph(&one).unwrap();
    assert_eq!((g.vcount(), g.ecount()), (1, 0));
    let g = Graph::delaunay_graph(&Matrix::zeros(0, 2)).unwrap();
    assert_eq!(g.vcount(), 0);
}

#[test]
fn proximity_graph_hierarchy_on_random_points() {
    // Euclidean MST ⊆ RNG ⊆ Gabriel ⊆ Delaunay: check the inclusions we can build.
    rng::seed(123).unwrap();
    let pts = random_points(80);
    let delaunay = edge_set(&Graph::delaunay_graph(&pts).unwrap());
    let gabriel_graph = Graph::gabriel_graph(&pts).unwrap();
    let gabriel = edge_set(&gabriel_graph);
    let rng_graph = Graph::relative_neighborhood_graph(&pts).unwrap();
    let rn = edge_set(&rng_graph);
    assert!(rn.is_subset(&gabriel));
    assert!(gabriel.is_subset(&delaunay));
    assert!(rn.len() < gabriel.len() && gabriel.len() < delaunay.len());
    // A planar triangulation has at most 3n - 6 edges; the RNG is connected, so >= n - 1.
    assert!(delaunay.len() <= 3 * 80 - 6);
    assert!(rn.len() >= 79);

    // The Gabriel graph is the beta-skeleton with beta = 1 (both flavors).
    assert_eq!(
        edge_set(&Graph::lune_beta_skeleton(&pts, 1.0).unwrap()),
        gabriel
    );
    assert_eq!(
        edge_set(&Graph::circle_beta_skeleton(&pts, 1.0).unwrap()),
        gabriel
    );
    // Larger beta, sparser skeleton.
    let lune2 = edge_set(&Graph::lune_beta_skeleton(&pts, 2.0).unwrap());
    let lune15 = edge_set(&Graph::lune_beta_skeleton(&pts, 1.5).unwrap());
    assert!(lune2.is_subset(&lune15) && lune15.is_subset(&gabriel));
    // In general position, the beta = 2 skeleton is the RNG.
    assert_eq!(lune2, rn);
    // beta < 1 is denser than the Gabriel graph (2D only).
    let lune05 = edge_set(&Graph::lune_beta_skeleton(&pts, 0.5).unwrap());
    assert!(gabriel.is_subset(&lune05));

    // Every edge of the Gabriel graph has its diametral disk empty.
    let rows = pts.to_rows();
    for &(a, b) in &gabriel {
        let (pa, pb) = (&rows[a as usize], &rows[b as usize]);
        let c = [(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0];
        let r2 = ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2)) / 4.0;
        for (k, p) in rows.iter().enumerate() {
            if k as i64 != a && k as i64 != b {
                assert!((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) > r2);
            }
        }
    }
}

#[test]
fn beta_weighted_gabriel_graph_summarizes_skeletons() {
    rng::seed(321).unwrap();
    let pts = random_points(60);
    let (g, weights) = Graph::beta_weighted_gabriel_graph(&pts, f64::INFINITY).unwrap();
    assert_eq!(weights.len(), g.ecount());
    assert_eq!(edge_set(&g), edge_set(&Graph::gabriel_graph(&pts).unwrap()));
    assert!(weights.iter().all(|&w| w >= 1.0));

    for beta in [1.2, 1.7, 2.5] {
        let skeleton = edge_set(&Graph::lune_beta_skeleton(&pts, beta).unwrap());
        for (e, (a, b)) in g.edge_list().into_iter().enumerate() {
            let key = (a.min(b), a.max(b));
            if weights[e] > beta + 1e-6 {
                assert!(
                    skeleton.contains(&key),
                    "edge {key:?} (w = {}) at beta {beta}",
                    weights[e]
                );
            } else if weights[e] < beta - 1e-6 {
                assert!(
                    !skeleton.contains(&key),
                    "edge {key:?} (w = {}) at beta {beta}",
                    weights[e]
                );
            }
        }
    }
    // With a cutoff, persistent edges are reported as infinite.
    let (_, capped) = Graph::beta_weighted_gabriel_graph(&pts, 1.5).unwrap();
    for (w, c) in weights.iter().zip(&capped) {
        if *w < 1.5 - 1e-6 {
            assert_close(*w, *c, 1e-6);
        } else if *w > 1.5 + 1e-6 {
            assert!(c.is_infinite() || *c > 1.5 - 1e-6);
        }
    }
}

#[test]
fn circle_skeleton_requires_2d() {
    // A regular tetrahedron.
    let pts3 = Matrix::from_rows(&[
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ])
    .unwrap();
    assert_eq!(
        Graph::circle_beta_skeleton(&pts3, 1.0).unwrap_err().kind(),
        ErrorKind::Unimplemented
    );
    assert_eq!(
        Graph::lune_beta_skeleton(&pts3, 0.5).unwrap_err().kind(),
        ErrorKind::Unimplemented
    );
    // Gabriel graphs work in any dimension: a tetrahedron's corners are all adjacent.
    assert_eq!(Graph::gabriel_graph(&pts3).unwrap().ecount(), 6);
    // A corner of the unit cube: the origin is on the boundary of the
    // diametral balls of the other pairs, so only the 3 spokes remain.
    let corner = Matrix::from_rows(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    .unwrap();
    assert_eq!(
        sorted_edges(&Graph::gabriel_graph(&corner).unwrap()),
        vec![(0, 1), (0, 2), (0, 3)]
    );
    assert_eq!(
        Graph::lune_beta_skeleton(&corner, 0.0).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        Graph::lune_beta_skeleton(&corner, f64::INFINITY)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        Graph::circle_beta_skeleton(&corner, f64::NAN)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

/// Brute-force β-skeletons in 2D, straight from the definitions (closed regions).
fn brute_skeleton(rows: &[Vec<f64>], beta: f64, circle: bool) -> BTreeSet<(i64, i64)> {
    let d2 = |p: &[f64], q: &[f64]| (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2);
    let n = rows.len();
    let mut edges = BTreeSet::new();
    for a in 0..n {
        for b in a + 1..n {
            let (pa, pb) = (&rows[a], &rows[b]);
            let ab2 = d2(pa, pb);
            let mid = [(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0];
            let blocked = (0..n).filter(|&k| k != a && k != b).any(|k| {
                let p = &rows[k];
                if beta < 1.0 || circle {
                    // Disks through A and B, centered on the bisector of AB.
                    let r = if beta < 1.0 { 0.5 / beta } else { 0.5 * beta };
                    let f = (r * r - 0.25).sqrt();
                    let perp = [-(pa[1] - pb[1]) * f, (pa[0] - pb[0]) * f];
                    let c1 = [mid[0] + perp[0], mid[1] + perp[1]];
                    let c2 = [mid[0] - perp[0], mid[1] - perp[1]];
                    let (in1, in2) = (d2(p, &c1) <= r * r * ab2, d2(p, &c2) <= r * r * ab2);
                    if beta < 1.0 { in1 && in2 } else { in1 || in2 }
                } else {
                    // Balls centered on the line AB.
                    let r = 0.5 * beta;
                    let c1 = [
                        pa[0] + (r - 1.0) * (pa[0] - pb[0]),
                        pa[1] + (r - 1.0) * (pa[1] - pb[1]),
                    ];
                    let c2 = [
                        pb[0] + (r - 1.0) * (pb[0] - pa[0]),
                        pb[1] + (r - 1.0) * (pb[1] - pa[1]),
                    ];
                    d2(p, &c1) <= r * r * ab2 && d2(p, &c2) <= r * r * ab2
                }
            });
            if !blocked {
                edges.insert((a as i64, b as i64));
            }
        }
    }
    edges
}

#[test]
fn beta_skeletons_match_their_definition_for_all_betas() {
    // igraph 1.0.0 gets beta > 2 (lune) and beta < 0.5 wrong; the wrappers do not.
    rng::seed(321).unwrap();
    let pts = random_points(40);
    let rows = pts.to_rows();
    for beta in [0.1, 0.3, 0.49, 0.5, 0.7, 1.0, 1.3, 2.0, 2.2, 3.0, 6.0] {
        let lune = edge_set(&Graph::lune_beta_skeleton(&pts, beta).unwrap());
        assert_eq!(
            lune,
            brute_skeleton(&rows, beta, false),
            "lune, beta = {beta}"
        );
        let circle = edge_set(&Graph::circle_beta_skeleton(&pts, beta).unwrap());
        assert_eq!(
            circle,
            brute_skeleton(&rows, beta, true),
            "circle, beta = {beta}"
        );
    }
}

#[test]
fn equilateral_triangle_rng_vs_beta2() {
    let h = 3f64.sqrt() / 2.0;
    let pts = Matrix::from_rows(&[[0.0, 0.0], [1.0, 0.0], [0.5, h]]).unwrap();
    // Documented difference: the RNG connects the corners, the beta = 2 skeleton does not.
    assert_eq!(
        Graph::relative_neighborhood_graph(&pts).unwrap().ecount(),
        3
    );
    assert_eq!(Graph::lune_beta_skeleton(&pts, 2.0).unwrap().ecount(), 0);
}

#[test]
fn nearest_neighbor_graph_igraph_1d_cases() {
    // tests/unit/igraph_nearest_neighbor_graph.c with its .out file.
    let pts = Matrix::from_rows(&[[12.0], [8.0], [5.0], [10.0], [12.0]]).unwrap();
    let g = Graph::nearest_neighbor_graph(&pts, Metric::L2, Some(2), Some(3.0), true).unwrap();
    assert!(g.is_directed());
    assert_eq!(
        sorted_edges(&g),
        vec![(0, 3), (0, 4), (1, 3), (3, 0), (3, 1), (4, 0), (4, 3)]
    );
    let g = Graph::nearest_neighbor_graph(&pts, Metric::L2, Some(1), None, true).unwrap();
    assert_eq!(
        sorted_edges(&g),
        vec![(0, 4), (1, 3), (2, 1), (3, 0), (4, 0)]
    );
    let g = Graph::nearest_neighbor_graph(&pts, Metric::L2, None, None, true).unwrap();
    assert_eq!(g.ecount(), 20, "complete directed graph");
}

#[test]
fn nearest_neighbor_graph_properties() {
    rng::seed(8).unwrap();
    let pts = random_points(100);
    let rows = pts.to_rows();
    for metric in [Metric::Euclidean, Metric::Manhattan] {
        let dist = |a: &[f64], b: &[f64]| match metric {
            Metric::Euclidean => (a[0] - b[0]).hypot(a[1] - b[1]),
            Metric::Manhattan => (a[0] - b[0]).abs() + (a[1] - b[1]).abs(),
        };
        let g = Graph::nearest_neighbor_graph(&pts, metric, Some(3), None, true).unwrap();
        let out = g.degree(.., NeighborMode::Out, Loops::Twice).unwrap();
        assert!(out.iter().all(|&d| d == 3));
        // Each chosen neighbor is at least as close as any non-chosen point.
        for v in 0..100i64 {
            let nbrs = g.neighbors(v, NeighborMode::Out).unwrap();
            let far = nbrs
                .iter()
                .map(|&u| dist(&rows[v as usize], &rows[u as usize]))
                .fold(0.0, f64::max);
            for u in 0..100i64 {
                if u != v && !nbrs.contains(&u) {
                    assert!(dist(&rows[v as usize], &rows[u as usize]) >= far - 1e-12);
                }
            }
        }
        // Edge lengths respect the cutoff.
        let cut = Graph::nearest_neighbor_graph(&pts, metric, None, Some(0.15), false).unwrap();
        assert!(!cut.is_directed());
        let lengths = cut.spatial_edge_lengths(&pts, metric).unwrap();
        assert!(lengths.iter().all(|&l| l <= 0.15));
        // ... and all close pairs are there.
        let expected = (0..100)
            .flat_map(|a| (a + 1..100).map(move |b| (a, b)))
            .filter(|&(a, b)| dist(&rows[a], &rows[b]) < 0.15 - 1e-12)
            .count();
        assert_eq!(cut.ecount(), expected);
    }
}

#[test]
fn spatial_edge_lengths_square() {
    // tests/unit/igraph_spatial_edge_lengths.c: K4 on the corners of a unit square.
    let k4 = complete(4);
    let pts = Matrix::from_rows(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).unwrap();
    let l2 = k4.spatial_edge_lengths(&pts, Metric::L2).unwrap();
    let s = 2f64.sqrt();
    for (a, b) in l2.iter().zip([1.0, 1.0, s, s, 1.0, 1.0]) {
        assert_close(*a, b, 1e-12);
    }
    assert_eq!(
        k4.spatial_edge_lengths(&pts, Metric::L1).unwrap(),
        vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0]
    );
    let three = Matrix::from_rows(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]).unwrap();
    assert_eq!(
        k4.spatial_edge_lengths(&three, Metric::L2)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert!(
        Graph::new(0, false)
            .spatial_edge_lengths(&Matrix::new(), Metric::L2)
            .unwrap()
            .is_empty()
    );
    assert_eq!(Metric::try_from(1u32).unwrap(), Metric::Manhattan);
    assert_eq!(Metric::L2, Metric::Euclidean);
}

#[test]
fn convex_hull_matches_igraph_output() {
    // tests/unit/igraph_convex_hull_2d.c, test_simple.
    let pts = Matrix::from_rows(&[
        [3.0, 2.0],
        [5.0, 1.0],
        [4.0, 4.0],
        [6.0, 4.0],
        [4.0, 3.0],
        [2.0, 5.0],
        [1.0, 3.0],
        [2.0, 4.0],
        [6.0, 3.0],
        [9.0, 2.0],
    ])
    .unwrap();
    let hull = misc::convex_hull_2d(&pts).unwrap();
    assert_eq!(hull.vertices, vec![1, 6, 5, 3, 9]);
    assert_eq!(
        hull.points,
        vec![[5.0, 1.0], [1.0, 3.0], [2.0, 5.0], [6.0, 4.0], [9.0, 2.0]]
    );
    // Collinear points: only the extremes.
    let line =
        Matrix::from_rows(&[[3.0, 2.0], [5.0, 1.0], [7.0, 0.0], [11.0, -2.0], [1.0, 3.0]]).unwrap();
    let mut v = misc::convex_hull_2d(&line).unwrap().vertices;
    v.sort();
    assert_eq!(v, vec![3, 4]);
    // Degenerate sizes.
    assert_eq!(
        misc::convex_hull_2d(&Matrix::from_rows(&[[3.0, 2.0]]).unwrap())
            .unwrap()
            .vertices,
        vec![0]
    );
    let empty = misc::convex_hull_2d(&Matrix::zeros(0, 2)).unwrap();
    assert!(empty.vertices.is_empty() && empty.points.is_empty());
    assert_eq!(empty.area(), 0.0);
    assert_eq!(
        misc::convex_hull_2d(&Matrix::zeros(3, 3))
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn convex_hull_of_random_points_contains_them_all() {
    rng::seed(77).unwrap();
    let pts = random_points(300);
    let hull = misc::convex_hull_2d(&pts).unwrap();
    let n = hull.points.len();
    assert!(n >= 3);
    // The hull is convex and all points lie on the same side of every side.
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let orientation = cross(hull.points[0], hull.points[1], hull.points[2]).signum();
    for i in 0..n {
        let (a, b) = (hull.points[i], hull.points[(i + 1) % n]);
        for p in pts.rows() {
            assert!(cross(a, b, [p[0], p[1]]) * orientation >= -1e-12);
        }
    }
    assert!(hull.area() > 0.8 && hull.area() <= 1.0);
    assert!(hull.perimeter() > 3.5 && hull.perimeter() <= 4.0);
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

#[test]
fn sphere_samplers() {
    rng::seed(42).unwrap();
    let surface = misc::sample_sphere_surface(5, 200, 3.0, false).unwrap();
    assert_eq!(surface.len(), 200);
    for p in &surface {
        assert_eq!(p.len(), 5);
        assert_close(p.iter().map(|x| x * x).sum::<f64>().sqrt(), 3.0, 1e-9);
    }
    // Symmetric: the mean is near the origin; some coordinates are negative.
    let mean0 = surface.iter().map(|p| p[0]).sum::<f64>() / 200.0;
    assert!(mean0.abs() < 0.5);
    assert!(surface.iter().any(|p| p[0] < 0.0));

    let volume = misc::sample_sphere_volume(2, 2000, 1.0, true).unwrap();
    for p in &volume {
        assert!(p.iter().all(|&x| x >= 0.0));
        assert!(p[0].hypot(p[1]) <= 1.0 + 1e-12);
    }
    // Uniform in the quarter disk: P(r < 1/2) = 1/4.
    let inner = volume.iter().filter(|p| p[0].hypot(p[1]) < 0.5).count() as f64 / 2000.0;
    assert_close(inner, 0.25, 0.04);

    assert!(
        misc::sample_sphere_surface(3, 0, 1.0, false)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        misc::sample_sphere_surface(1, 3, 1.0, false)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        misc::sample_sphere_volume(3, 3, 0.0, false)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn dirichlet_sampler() {
    rng::seed(42).unwrap();
    let alpha = [1.0, 2.0, 3.0, 4.0, 5.0];
    let samples = misc::sample_dirichlet(3000, &alpha).unwrap();
    assert_eq!(samples.len(), 3000);
    for p in &samples {
        assert_close(p.iter().sum(), 1.0, 1e-9);
        assert!(p.iter().all(|&x| x >= 0.0));
    }
    // The mean is alpha / sum(alpha).
    for (k, a) in alpha.iter().enumerate() {
        let mean = samples.iter().map(|p| p[k]).sum::<f64>() / 3000.0;
        assert_close(mean, a / 15.0, 0.01);
    }
    // Huge concentrations: the distribution is localized (igraph test).
    for p in misc::sample_dirichlet(2, &[1e30, 1e30]).unwrap() {
        assert_close(p[0], 0.5, 1e-9);
        assert_close(p[1], 0.5, 1e-9);
    }
    assert!(misc::sample_dirichlet(0, &[1.0, 1.0]).unwrap().is_empty());
    assert_eq!(
        misc::sample_dirichlet(0, &[]).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        misc::sample_dirichlet(1, &[-1.0, 1.0]).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn explicit_generators_are_reproducible_and_independent() {
    let mut a = Rng::new(RngType::Mt19937, 17).unwrap();
    let mut b = Rng::new(RngType::Mt19937, 17).unwrap();
    rng::seed(1).unwrap();
    let x = a.sample_sphere_volume(3, 4, 2.0, false).unwrap();
    rng::seed(2).unwrap(); // the default generator does not matter
    let y = b.sample_sphere_volume(3, 4, 2.0, false).unwrap();
    assert_eq!(x, y);
    assert_eq!(
        a.sample_dirichlet(3, &[1.0, 1.0]).unwrap(),
        b.sample_dirichlet(3, &[1.0, 1.0]).unwrap()
    );
    assert_eq!(
        a.sample_sphere_surface(2, 5, 1.0, true).unwrap(),
        b.sample_sphere_surface(2, 5, 1.0, true).unwrap()
    );
}

// ---------------------------------------------------------------------------
// Prefix-sum trees
// ---------------------------------------------------------------------------

#[test]
fn psumtree_basics() {
    let mut t = PsumTree::new(5).unwrap();
    assert_eq!(t.len(), 5);
    assert!(!t.is_empty());
    assert_eq!(t.sum(), 0.0);
    assert_eq!(t.sample(), None);
    assert_eq!(t.search(0.0).unwrap_err().kind(), ErrorKind::InvalidValue);
    for (i, w) in [2.0, 0.0, 1.0, 0.5, 4.5].into_iter().enumerate() {
        t.update(i, w).unwrap();
    }
    assert_eq!(t.sum(), 8.0);
    assert_eq!(t.weights(), vec![2.0, 0.0, 1.0, 0.5, 4.5]);
    assert_eq!(t.get(4), Some(4.5));
    assert_eq!(t.get(5), None);
    // Cumulative intervals: [0,2) -> 0, [2,3) -> 2, [3,3.5) -> 3, [3.5,8) -> 4.
    let cases = [
        (0.0, 0),
        (1.99, 0),
        (2.0, 2),
        (2.5, 2),
        (3.0, 3),
        (3.49, 3),
        (3.5, 4),
        (7.99, 4),
    ];
    for (x, i) in cases {
        assert_eq!(t.search(x).unwrap(), i, "search({x})");
    }
    assert_eq!(t.search(8.0).unwrap_err().kind(), ErrorKind::InvalidValue);
    assert_eq!(t.search(-0.1).unwrap_err().kind(), ErrorKind::InvalidValue);
    assert_eq!(
        t.update(5, 1.0).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        t.update(0, -1.0).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        t.update(0, f64::NAN).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );

    let copy = t.clone();
    t.reset();
    assert_eq!(t.sum(), 0.0);
    assert_eq!(copy.sum(), 8.0);
    assert_eq!(copy.search(3.2).unwrap(), 3);

    assert_eq!(
        PsumTree::new(0).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        PsumTree::from_weights(&[]).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    let single = PsumTree::from_weights(&[0.25]).unwrap();
    assert_eq!(single.search(0.1).unwrap(), 0);
}

#[test]
fn psumtree_sampling_follows_the_weights() {
    rng::seed(4242).unwrap();
    let weights = [1.0, 0.0, 2.0, 3.0, 4.0, 0.0, 10.0];
    let tree = PsumTree::from_weights(&weights).unwrap();
    let draws = 40_000;
    let mut counts = [0usize; 7];
    for _ in 0..draws {
        counts[tree.sample().unwrap()] += 1;
    }
    let total: f64 = weights.iter().sum();
    for (c, w) in counts.iter().zip(weights) {
        assert_close(*c as f64 / draws as f64, w / total, 0.01);
    }
    assert_eq!(
        counts[1] + counts[5],
        0,
        "zero weight items are never drawn"
    );
}

// ---------------------------------------------------------------------------
// Progress, status and interruption handlers
// ---------------------------------------------------------------------------

/// Betweenness centrality: an igraph computation that reports its progress.
fn betweenness(g: &Graph) -> igraph::Result<Vec<f64>> {
    g.betweenness(None, .., false, false)
}

#[test]
fn progress_handler_sees_betweenness_progress() {
    let g = zachary();
    let reports = Rc::new(RefCell::new(Vec::<(String, f64)>::new()));
    let sink = Rc::clone(&reports);
    let bw = misc::with_progress_handler(
        move |msg, pct| {
            sink.borrow_mut().push((msg.to_owned(), pct));
            ControlFlow::Continue(())
        },
        || betweenness(&g),
    )
    .unwrap();
    assert_close(bw[0], 231.0714, 1e-3);
    {
        let reports = reports.borrow();
        assert!(reports.len() >= 2);
        assert!(reports.iter().all(|(m, _)| m.contains("Betweenness")));
        assert_eq!(reports.first().unwrap().1, 0.0);
        assert_eq!(reports.last().unwrap().1, 100.0);
        assert!(
            reports.windows(2).all(|w| w[0].1 <= w[1].1),
            "percentages increase"
        );
    }

    // Once the scope is over, the handler (and its captured state) is gone.
    assert_eq!(Rc::strong_count(&reports), 1);
    let before = reports.borrow().len();
    betweenness(&zachary()).unwrap();
    misc::progress("nobody listens", 1.0).unwrap();
    assert_eq!(reports.borrow().len(), before);
}

#[test]
fn progress_handler_can_stop_a_computation() {
    let calls = Rc::new(Cell::new(0));
    let counter = Rc::clone(&calls);
    let _guard = misc::set_progress_handler(move |_, pct| {
        counter.set(counter.get() + 1);
        if pct >= 20.0 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    let err = betweenness(&zachary()).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Interrupted);
    assert!(calls.get() >= 2);
}

#[test]
fn progress_handlers_nest_and_restore() {
    let log = Rc::new(RefCell::new(Vec::<&'static str>::new()));
    let (l1, l2) = (Rc::clone(&log), Rc::clone(&log));
    let outer = misc::set_progress_handler(move |_, _| {
        l1.borrow_mut().push("outer");
        ControlFlow::Continue(())
    });
    misc::progress("a", 0.0).unwrap();
    {
        let _inner = misc::set_progress_handler(move |_, _| {
            l2.borrow_mut().push("inner");
            ControlFlow::Continue(())
        });
        misc::progress("b", 50.0).unwrap();
        {
            // The C stderr handler can be stacked too (it prints to stderr).
            let _stderr = misc::set_progress_handler_stderr();
            misc::progress("printed to stderr by igraph", 75.0).unwrap();
        }
        misc::progress("c", 90.0).unwrap();
    }
    misc::progress("d", 100.0).unwrap();
    drop(outer);
    misc::progress("e", 100.0).unwrap();
    assert_eq!(*log.borrow(), vec!["outer", "inner", "inner", "outer"]);
    assert_eq!(
        misc::progress("nul\0byte", 1.0).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn progress_handler_panics_are_resumed_in_rust() {
    let result = std::panic::catch_unwind(|| {
        misc::with_progress_handler(|_, _| panic!("boom in handler"), || betweenness(&zachary()))
    });
    let payload = result.unwrap_err();
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"boom in handler"));
    // The thread is still usable afterwards, with no handler installed.
    assert_eq!(betweenness(&zachary()).unwrap().len(), 34);
}

#[test]
fn progress_handlers_are_per_thread() {
    let hits = Rc::new(Cell::new(0));
    let h = Rc::clone(&hits);
    let _guard = misc::set_progress_handler(move |_, _| {
        h.set(h.get() + 1);
        ControlFlow::Continue(())
    });
    std::thread::spawn(|| {
        // No handler in this thread.
        misc::progress("other thread", 50.0).unwrap();
        betweenness(&zachary()).unwrap();
    })
    .join()
    .unwrap();
    assert_eq!(hits.get(), 0);
    misc::progress("this thread", 50.0).unwrap();
    assert_eq!(hits.get(), 1);
}

#[test]
fn status_handler_receives_messages() {
    let log = Rc::new(RefCell::new(Vec::<String>::new()));
    let sink = Rc::clone(&log);
    misc::with_status_handler(
        move |msg| {
            sink.borrow_mut().push(msg.to_owned());
            if msg == "stop" {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        },
        || {
            misc::status("phase 1").unwrap();
            misc::status("phase 2").unwrap();
            assert_eq!(
                misc::status("stop").unwrap_err().kind(),
                ErrorKind::Interrupted
            );
        },
    );
    misc::status("unheard").unwrap();
    {
        let _g = misc::set_status_handler_stderr();
        misc::status("igraph prints this on stderr\n").unwrap();
    }
    assert_eq!(*log.borrow(), vec!["phase 1", "phase 2", "stop"]);
}

#[test]
fn interruption_handler_stops_sir() {
    rng::seed(1).unwrap();
    let g = zachary();
    assert!(!misc::allow_interruption(), "no handler: never interrupt");
    let polls = Rc::new(Cell::new(0));
    let p = Rc::clone(&polls);
    let err = misc::with_interruption_handler(
        move || {
            p.set(p.get() + 1);
            p.get() > 3
        },
        || g.sir(2.0, 1.0, 100),
    )
    .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Interrupted);
    assert_eq!(polls.get(), 4);
    // Handler removed: the same computation completes.
    assert_eq!(g.sir(2.0, 1.0, 100).unwrap().len(), 100);

    let _guard = misc::set_interruption_handler(|| true);
    assert!(misc::allow_interruption());
}

#[test]
fn interruption_handler_panic_is_resumed() {
    let g = zachary();
    let result = std::panic::catch_unwind(|| {
        misc::with_interruption_handler(|| panic!("cancelled loudly"), || g.sir(1.0, 1.0, 3))
    });
    assert_eq!(
        result.unwrap_err().downcast_ref::<&str>(),
        Some(&"cancelled loudly")
    );
    let result = std::panic::catch_unwind(|| {
        let _g = misc::set_interruption_handler(|| panic!("direct"));
        misc::allow_interruption()
    });
    assert!(result.is_err());
    assert!(!misc::allow_interruption());
}

// ---------------------------------------------------------------------------
// A use case: an outbreak in a small town
// ---------------------------------------------------------------------------

/// 150 households are scattered over a square town. People mostly meet their
/// spatial neighbors, so the contact network is the Gabriel graph of the
/// household positions. We (1) simulate an outbreak, (2) vaccinate the most
/// connected households and check that the outbreaks shrink, and (3) send
/// three mobile clinics, parked at the corners of the town's convex hull, to
/// the three busiest households, minimizing the total driving distance.
#[test]
fn use_case_outbreak_in_a_small_town() {
    rng::seed(2025).unwrap();
    let n = 150;
    let homes = random_points(n);
    let contacts = Graph::gabriel_graph(&homes).unwrap();
    assert_eq!(contacts.vcount(), n);

    // (1) The outbreak: many runs with a progress-like interruption budget.
    let polls = Rc::new(Cell::new(0usize));
    let p = Rc::clone(&polls);
    let runs = misc::with_interruption_handler(
        move || {
            p.set(p.get() + 1);
            false // never cancel, just count the checkpoints
        },
        || contacts.sir(1.5, 1.0, 200),
    )
    .unwrap();
    assert!(polls.get() > 0, "igraph polled the interruption handler");
    for run in &runs {
        check_sir_invariants(run, n as i64);
    }
    let mean_final = |runs: &[misc::SirRun]| {
        runs.iter().map(|r| r.final_size() as f64).sum::<f64>() / runs.len() as f64
    };
    let baseline = mean_final(&runs);

    // (2) Vaccinate the 30 most connected households: remove them from the network.
    let degrees = contacts
        .degree(.., NeighborMode::All, Loops::Twice)
        .unwrap();
    let mut by_degree: Vec<i64> = (0..n as i64).collect();
    by_degree.sort_by_key(|&v| std::cmp::Reverse(degrees[v as usize]));
    let mut vaccinated = contacts.clone();
    vaccinated
        .delete_vertices(by_degree[..30].to_vec())
        .unwrap();
    let after = mean_final(&vaccinated.sir(1.5, 1.0, 200).unwrap());
    assert!(after < baseline, "vaccination helps: {after} < {baseline}");

    // (3) Dispatch clinics from three hull corners to the three busiest homes.
    let hull = misc::convex_hull_2d(&homes).unwrap();
    assert!(hull.vertices.len() >= 3);
    let depots: Vec<[f64; 2]> = hull.points[..3].to_vec();
    let rows = homes.to_rows();
    let targets: Vec<&Vec<f64>> = by_degree[..3].iter().map(|&v| &rows[v as usize]).collect();
    let mut cost = Matrix::zeros(3, 3);
    for (i, d) in depots.iter().enumerate() {
        for (j, t) in targets.iter().enumerate() {
            cost[(i, j)] = (d[0] - t[0]).hypot(d[1] - t[1]);
        }
    }
    let plan = misc::solve_lsap(&cost).unwrap();
    let plan_cost: f64 = plan
        .iter()
        .enumerate()
        .map(|(i, &j)| cost[(i, j as usize)])
        .sum();
    let best = permutations(3)
        .iter()
        .map(|p| {
            p.iter()
                .enumerate()
                .map(|(i, &j)| cost[(i, j)])
                .sum::<f64>()
        })
        .fold(f64::INFINITY, f64::min);
    assert_close(plan_cost, best, 1e-12);

    // The street lengths of the contact network, e.g. to weigh travel times.
    let lengths = contacts
        .spatial_edge_lengths(&homes, Metric::Euclidean)
        .unwrap();
    assert_eq!(lengths.len(), contacts.ecount());
    assert!(lengths.iter().all(|&l| l > 0.0 && l < 2f64.sqrt()));
}

// ---------------------------------------------------------------------------
// Edge cases found in review: inputs igraph 1.0.0 mishandles (infinite loops,
// aborts, garbage results) are rejected or handled on the Rust side.
// ---------------------------------------------------------------------------

#[test]
fn lsap_rejects_non_finite_costs() {
    // igraph's Hungarian method would loop forever on these.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let cost = Matrix::from_rows(&[[bad, 1.0], [1.0, 2.0]]).unwrap();
        assert_eq!(
            misc::solve_lsap(&cost).unwrap_err().kind(),
            ErrorKind::InvalidValue
        );
    }
    // A large finite cost forbids an assignment instead.
    let cost = Matrix::from_rows(&[[1e9, 1.0], [1.0, 1e9]]).unwrap();
    assert_eq!(misc::solve_lsap(&cost).unwrap(), vec![1, 0]);
}

#[test]
fn random_sample_of_size_zero_from_a_single_value_is_empty() {
    // igraph 1.0.0 and 1.0.1 alone would return [7] here.
    assert!(misc::random_sample(7, 7, 0).unwrap().is_empty());
    assert_eq!(
        misc::random_sample(8, 7, 0).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        misc::random_sample(0, 3, usize::MAX).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn power_law_fit_edge_cases() {
    assert_eq!(
        misc::power_law_fit(&[], None, false).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        misc::power_law_fit(&[1.0, f64::NAN, 3.0], None, false)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        misc::power_law_fit(&[1.0, 2.0], Some(f64::NAN), false)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    // No sample reaches xmin: a degenerate (but documented) model.
    let fit = misc::power_law_fit(&[1.0, 2.0, 3.0], Some(10.0), false).unwrap();
    assert!(!fit.alpha.is_finite());
}

#[test]
fn samplers_reject_invalid_parameters() {
    rng::seed(3).unwrap();
    for radius in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            misc::sample_sphere_surface(3, 2, radius, false)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidValue
        );
        assert_eq!(
            misc::sample_sphere_volume(3, 2, radius, false)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidValue
        );
    }
    assert_eq!(
        misc::sample_sphere_surface(1, 2, 1.0, false)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    for alpha in [
        vec![1.0],
        vec![1.0, 0.0],
        vec![f64::NAN, 1.0],
        vec![f64::INFINITY, 1.0],
    ] {
        assert_eq!(
            misc::sample_dirichlet(2, &alpha).unwrap_err().kind(),
            ErrorKind::InvalidValue
        );
    }
    assert!(misc::sample_dirichlet(0, &[1.0, 1.0]).unwrap().is_empty());
}

#[test]
fn spatial_functions_reject_non_finite_coordinates() {
    let pts = Matrix::from_rows(&[[0.0, 0.0], [f64::NAN, 0.0], [1.0, 0.0]]).unwrap();
    for beta in [0.3, 0.7, 1.5, 3.0] {
        assert_eq!(
            Graph::lune_beta_skeleton(&pts, beta).unwrap_err().kind(),
            ErrorKind::InvalidValue,
            "lune, beta = {beta}"
        );
        assert_eq!(
            Graph::circle_beta_skeleton(&pts, beta).unwrap_err().kind(),
            ErrorKind::InvalidValue,
            "circle, beta = {beta}"
        );
    }
    assert_eq!(
        misc::convex_hull_2d(&pts).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        Graph::gabriel_graph(&pts).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    let ok = Matrix::from_rows(&[[0.0, 0.0], [1.0, 0.0]]).unwrap();
    assert_eq!(
        Graph::beta_weighted_gabriel_graph(&ok, f64::NAN)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    for cutoff in [-1.0, f64::NAN] {
        assert_eq!(
            Graph::nearest_neighbor_graph(&ok, Metric::L2, None, Some(cutoff), false)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidValue
        );
    }
    // A huge k simply means "no limit".
    let g = Graph::nearest_neighbor_graph(&ok, Metric::L1, Some(usize::MAX), None, false).unwrap();
    assert_eq!(g.ecount(), 1);
}

#[test]
fn skeletons_of_tiny_point_sets() {
    // No points, or no more points than dimensions: igraph cannot triangulate
    // them, but every skeleton is still defined (the lune of beta > 2 used to
    // go through the Delaunay-based weighted Gabriel graph).
    let empty = Matrix::zeros(0, 2);
    for beta in [0.3, 1.0, 3.0] {
        let g = Graph::lune_beta_skeleton(&empty, beta).unwrap();
        assert_eq!((g.vcount(), g.ecount()), (0, 0));
        let g = Graph::circle_beta_skeleton(&empty, beta).unwrap();
        assert_eq!((g.vcount(), g.ecount()), (0, 0));
    }
    let two = Matrix::from_rows(&[[0.0, 0.0], [3.0, 1.0]]).unwrap();
    for beta in [0.2, 0.7, 1.0, 2.0, 2.5, 10.0] {
        assert_eq!(Graph::lune_beta_skeleton(&two, beta).unwrap().ecount(), 1);
        assert_eq!(Graph::circle_beta_skeleton(&two, beta).unwrap().ecount(), 1);
    }
    // Three points in 3D (n <= dim, handled pair by pair): compare with the
    // definition of the lune.
    let bent = Matrix::from_rows(&[[0.0, 0.0, 0.0], [1.0, 0.3, 0.0], [2.0, 0.0, 0.0]]).unwrap();
    let rows = bent.to_rows();
    for beta in [1.0, 1.5, 2.0, 3.0, 8.0] {
        let got = edge_set(&Graph::lune_beta_skeleton(&bent, beta).unwrap());
        // Brute force in 3D: the lune centers are on the line AB.
        let r = beta / 2.0;
        let d2 =
            |p: &[f64], q: &[f64]| -> f64 { p.iter().zip(q).map(|(x, y)| (x - y).powi(2)).sum() };
        let mut expected = BTreeSet::new();
        for a in 0..3usize {
            for b in a + 1..3 {
                let (pa, pb) = (&rows[a], &rows[b]);
                let c1: Vec<f64> = pa
                    .iter()
                    .zip(pb)
                    .map(|(x, y)| x + (r - 1.0) * (x - y))
                    .collect();
                let c2: Vec<f64> = pa
                    .iter()
                    .zip(pb)
                    .map(|(x, y)| y + (r - 1.0) * (y - x))
                    .collect();
                let rad2 = r * r * d2(pa, pb);
                let k = 3 - a - b;
                if !(d2(&rows[k], &c1) <= rad2 && d2(&rows[k], &c2) <= rad2) {
                    expected.insert((a as i64, b as i64));
                }
            }
        }
        assert_eq!(got, expected, "beta = {beta}");
    }
    // The same answers explicitly: the long side 0-2 never survives (the
    // middle point is in its lune), the short sides always do (the far
    // endpoint projects beyond them, outside even the widest lune).
    let expected: [(f64, &[(i64, i64)]); 3] = [
        (1.0, &[(0, 1), (1, 2)]),
        (2.0, &[(0, 1), (1, 2)]),
        (8.0, &[(0, 1), (1, 2)]),
    ];
    for (beta, edges) in expected {
        let got: Vec<(i64, i64)> = edge_set(&Graph::lune_beta_skeleton(&bent, beta).unwrap())
            .into_iter()
            .collect();
        assert_eq!(got, edges, "beta = {beta}");
    }
}

// ---------------------------------------------------------------------------
// igraph 1.0.1 still has the bugs worked around above: call the raw C
// functions to prove that the workarounds are still needed.
// ---------------------------------------------------------------------------

#[test]
fn raw_random_sample_bug_is_still_present_in_1_0_1() {
    // igraph 1.0.0 and 1.0.1 return [low] for an empty sample of [low, low].
    let mut res = VectorInt::new();
    igraph::igraph_call!(ffi::igraph_random_sample(&mut res, 7, 7, 0)).unwrap();
    assert_eq!(res.to_vec(), vec![7]);
    // The wrapper returns the empty sample.
    assert!(misc::random_sample(7, 7, 0).unwrap().is_empty());
}

#[test]
fn raw_lune_skeleton_bugs_are_still_present_in_1_0_1() {
    rng::seed(321).unwrap();
    let pts = random_points(40);
    let rows = pts.to_rows();
    let raw = |beta: f64| {
        edge_set(
            &Graph::init_with(|g| unsafe { ffi::igraph_lune_beta_skeleton(g, &pts, beta) })
                .unwrap(),
        )
    };
    for beta in [0.3, 3.0, 6.0] {
        let truth = brute_skeleton(&rows, beta, false);
        let c = raw(beta);
        // igraph 1.0.0 and 1.0.1: spurious edges, never missing ones.
        assert!(truth.is_subset(&c), "beta = {beta}");
        assert!(c.len() > truth.len(), "beta = {beta}: the C bug is fixed?");
        assert_eq!(
            edge_set(&Graph::lune_beta_skeleton(&pts, beta).unwrap()),
            truth
        );
    }
    let circle = edge_set(
        &Graph::init_with(|g| unsafe { ffi::igraph_circle_beta_skeleton(g, &pts, 0.3) }).unwrap(),
    );
    assert!(circle.len() > brute_skeleton(&rows, 0.3, true).len());
}

// ---------------------------------------------------------------------------
// Per-thread random number generators
// ---------------------------------------------------------------------------

#[test]
fn seeded_results_are_reproducible_in_parallel_threads() {
    // Every thread has its own default generator: seeding one thread does
    // not disturb the others, so concurrent seeded runs agree exactly.
    let run = || {
        rng::seed(2718).unwrap();
        let g = Graph::famous("Zachary").unwrap();
        let sir = g.sir(0.8, 1.0, 10).unwrap();
        let sample = misc::random_sample(0, 1_000_000, 20).unwrap();
        let sphere = misc::sample_sphere_surface(3, 5, 1.0, false).unwrap();
        (sir, sample, sphere)
    };
    let expected = run();
    let handles: Vec<_> = (0..8).map(|_| std::thread::spawn(run)).collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), expected);
    }
}

// ---------------------------------------------------------------------------
// Interplay with other modules
// ---------------------------------------------------------------------------

#[test]
fn euclidean_mst_is_inside_the_relative_neighborhood_graph() {
    // EMST ⊆ RNG ⊆ Gabriel ⊆ Delaunay: the EMST of the points is the MST of
    // their complete graph weighted by length, and it has the same total
    // length as the MST of the Delaunay graph.
    rng::seed(99).unwrap();
    let n = 60;
    let pts = random_points(n);
    let full = Graph::full(n, false, false).unwrap();
    let len = full.spatial_edge_lengths(&pts, Metric::Euclidean).unwrap();
    let mst = full
        .minimum_spanning_tree(Some(&len), MstAlgorithm::Prim)
        .unwrap();
    assert_eq!(mst.len(), n - 1);
    let edges = full.edge_list();
    let emst: BTreeSet<(i64, i64)> = mst
        .iter()
        .map(|&e| {
            let (a, b) = edges[e as usize];
            (a.min(b), a.max(b))
        })
        .collect();
    let rn = Graph::relative_neighborhood_graph(&pts).unwrap();
    assert!(emst.is_subset(&edge_set(&rn)));
    assert!(rn.is_connected(Connectedness::Weak).unwrap());

    let total = |g: &Graph, tree: &[i64]| {
        let l = g.spatial_edge_lengths(&pts, Metric::Euclidean).unwrap();
        tree.iter().map(|&e| l[e as usize]).sum::<f64>()
    };
    let delaunay = Graph::delaunay_graph(&pts).unwrap();
    let dl = delaunay
        .spatial_edge_lengths(&pts, Metric::Euclidean)
        .unwrap();
    let dmst = delaunay
        .minimum_spanning_tree(Some(&dl), MstAlgorithm::Kruskal)
        .unwrap();
    assert_close(total(&delaunay, &dmst), total(&full, &mst), 1e-12);
}

#[test]
fn nearest_neighbor_graph_with_a_cutoff_is_a_geometric_random_graph() {
    // A geometric random graph connects the points closer than its radius:
    // exactly the unlimited nearest neighbor graph with that cutoff.
    rng::seed(5).unwrap();
    let (grg, pts) = geometric_random_graph(200, 0.1);
    let nn =
        Graph::nearest_neighbor_graph(&pts, Metric::Euclidean, None, Some(0.1), false).unwrap();
    assert_eq!(edge_set(&nn), edge_set(&grg));
    assert!(grg.ecount() > 0);
}

/// A geometric random graph and the positions of its vertices.
fn geometric_random_graph(n: usize, radius: f64) -> (Graph, Matrix) {
    let geo = Graph::grg_game(n, radius, false).unwrap();
    let rows: Vec<[f64; 2]> = geo.x.iter().zip(&geo.y).map(|(&x, &y)| [x, y]).collect();
    (geo.graph, Matrix::from_rows(&rows).unwrap())
}

#[test]
fn sphere_samples_as_dot_product_graph_positions() {
    // Latent positions on the positive unit sphere: all dot products lie in
    // [0, 1], so they are valid connection probabilities.
    rng::seed(12).unwrap();
    let n = 300;
    let vecs = misc::sample_sphere_surface(3, n, 1.0, true).unwrap();
    let mut m = Matrix::zeros(3, n);
    for (j, v) in vecs.iter().enumerate() {
        for (i, &x) in v.iter().enumerate() {
            m[(i, j)] = x;
        }
    }
    igraph::error::take_warnings();
    let g = Graph::dot_product_game(&m, false).unwrap();
    assert!(
        igraph::error::take_warnings().is_empty(),
        "no probability exceeds 1"
    );
    // The expected number of edges is the sum of the pairwise dot products.
    let expected: f64 = (0..n)
        .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
        .map(|(a, b)| (0..3).map(|k| vecs[a][k] * vecs[b][k]).sum::<f64>())
        .sum();
    let sd = expected.sqrt();
    assert!(
        (g.ecount() as f64 - expected).abs() < 5.0 * sd,
        "{} vs {expected}",
        g.ecount()
    );
}

/// The continuous sample of igraph's `tests/unit/igraph_power_law_fit.c`.
const PLFIT_CONTINUOUS: [f64; 1000] = [
    1.52219974,
    6.80675663,
    1.02798042,
    1.31180733,
    3.97473174,
    1.17209342,
    1.64889191,
    2.47764721,
    1.32939375,
    3.03762554,
    1.62638327,
    6.08405495,
    1.70890382,
    1.05294973,
    1.17408407,
    4.48945532,
    1.16777371,
    2.52502391,
    1.09755984,
    1.63838051,
    1.03811206,
    1.47224168,
    1.57161431,
    1.60163451,
    2.08280263,
    1.04678340,
    1.33317526,
    1.58588741,
    1.26484666,
    1.02367503,
    1.57045702,
    3.42374138,
    1.23190611,
    1.09378228,
    1.04959505,
    1.05818408,
    1.43879491,
    2.22750459,
    1.41027204,
    1.81964745,
    2.80239939,
    1.25399323,
    1.07479219,
    3.94616077,
    1.26367914,
    1.87367507,
    1.35741026,
    1.14867526,
    7.33024762,
    1.87957274,
    2.79258534,
    1.21682159,
    1.61194300,
    2.81885973,
    1.21514746,
    1.12850917,
    51.85245035,
    1.21883209,
    1.04861029,
    1.69215609,
    2.18429429,
    1.59752172,
    1.41909984,
    3.14393355,
    1.18298455,
    1.67063821,
    1.88568524,
    1.07445906,
    1.45007973,
    1.12568920,
    1.56806310,
    1.36996101,
    1.19440982,
    6.57296980,
    1.35860725,
    1.06552137,
    1.16950701,
    1.34750790,
    1.66977492,
    1.22658722,
    1.62247444,
    1.23458784,
    8.55843760,
    1.70020162,
    4.76368831,
    1.04846170,
    1.13689661,
    1.94449567,
    1.10584812,
    1.32525767,
    1.26640912,
    1.91372972,
    1.56185373,
    2.37829675,
    1.04616674,
    2.43549177,
    1.14961092,
    1.82106455,
    1.25818298,
    1.64763037,
    1.43019402,
    1.50439978,
    1.90281251,
    1.34827040,
    1.57935671,
    1.77260751,
    1.06976614,
    1.12236012,
    2.19770254,
    1.51825533,
    1.19027804,
    1.08307524,
    1.57912902,
    3.33313888,
    2.14005088,
    1.38341873,
    1.20088138,
    1.25870539,
    1.03811620,
    1.86622820,
    2.99310953,
    1.55615055,
    2.12364873,
    4.49081000,
    1.01274439,
    1.22373389,
    3.79059729,
    3.10099275,
    2.70218546,
    1.03609624,
    2.20776919,
    1.00651347,
    1.87344592,
    1.04903307,
    1.24899747,
    1.20377911,
    1.12706494,
    1.01706713,
    7.01069306,
    1.05363146,
    2.50105512,
    1.11168552,
    1.71133998,
    1.17714528,
    1.37986755,
    2.20981534,
    1.18179277,
    2.07982010,
    4.04967099,
    1.00680257,
    1.62850069,
    2.58816230,
    1.35079027,
    1.03382890,
    4.54326500,
    1.62489905,
    1.36102570,
    1.52349738,
    1.06606346,
    7.80558026,
    1.02602538,
    1.43330925,
    1.36040920,
    9.29692547,
    15.27015690,
    1.75966437,
    1.02635409,
    1.40421505,
    2.87296958,
    1.46232202,
    1.87065204,
    3.37278803,
    1.82589564,
    1.06488044,
    1.72568108,
    1.21062115,
    4.39311214,
    1.12636227,
    2.20820528,
    1.09826903,
    2.58989998,
    1.34944949,
    1.08654244,
    2.38021951,
    3.96308780,
    1.37494639,
    1.18245279,
    3.72506217,
    3.79775023,
    1.19018356,
    2.86924476,
    3.40015888,
    1.92317855,
    1.55203754,
    1.34985008,
    1.31480190,
    1.65899877,
    4.77446435,
    1.41073246,
    1.35555456,
    2.40543613,
    2.72162935,
    1.34475982,
    1.41342115,
    5.15278473,
    1.69654436,
    3.21081899,
    1.18822397,
    1.40394863,
    1.06793574,
    1.67085563,
    1.08125975,
    1.11765459,
    1.17245045,
    1.15711479,
    1.18656910,
    1.61296203,
    1.71427634,
    1.24017302,
    2.05291524,
    2.52658791,
    2.04645295,
    34.07541626,
    1.32670899,
    1.03893757,
    1.08957199,
    5.55332328,
    1.17276097,
    1.60389480,
    2.02098430,
    2.92934928,
    1.00558653,
    1.05830070,
    1.81440889,
    3.85044779,
    1.12317456,
    1.39547640,
    2.93105179,
    1.95048788,
    1.05602445,
    1.96855429,
    1.60432293,
    3.28820202,
    1.50117325,
    1.19775674,
    1.28280841,
    1.08318646,
    1.02098264,
    1.24861938,
    1.06511473,
    1.07549717,
    3.57739126,
    1.07265409,
    1.06312441,
    1.16296512,
    3.83654484,
    2.02366951,
    1.73168875,
    1.60443228,
    2.30779766,
    1.50531775,
    1.31925607,
    1.87926179,
    1.86249354,
    2.14768716,
    2.31583955,
    2.15651148,
    1.29677318,
    1.10110071,
    1.03383916,
    1.50665009,
    1.16502917,
    1.40055008,
    2.80847193,
    1.29824634,
    2.76239920,
    1.73123621,
    1.15286577,
    1.89493526,
    1.63112634,
    1.17828846,
    1.01293513,
    1.84834048,
    4.19026736,
    1.82684815,
    3.51812301,
    1.33499862,
    2.03087497,
    1.32419883,
    1.34126954,
    1.98250684,
    1.00025697,
    1.59416883,
    6.38249787,
    2.79055559,
    1.57750678,
    1.36953983,
    1.37513919,
    3.63573178,
    1.15637432,
    9.28386344,
    1.16947695,
    1.54995742,
    1.44018755,
    1.29332881,
    1.81274872,
    1.14900153,
    1.07117403,
    1.17035915,
    1.39229249,
    1.96645872,
    1.09147706,
    1.25211993,
    1.07092474,
    1.85394206,
    1.29807741,
    3.41499510,
    1.22444449,
    1.00913782,
    3.87431854,
    1.01072376,
    1.01186727,
    3.00175639,
    2.52183377,
    1.23992099,
    1.69819010,
    1.36850400,
    1.14577814,
    1.06035078,
    1.08414298,
    1.55920217,
    5.07059630,
    1.15434572,
    1.41873305,
    1.24712256,
    1.10478618,
    1.30707247,
    1.85719110,
    1.89873207,
    1.72629431,
    1.65171651,
    7.10864875,
    2.31945709,
    1.06722361,
    1.26696259,
    2.23845503,
    1.38674196,
    1.91015397,
    1.29590323,
    1.10448028,
    4.52757499,
    2.00258408,
    1.38299092,
    1.01431427,
    1.54039270,
    1.34880396,
    1.08784083,
    1.35553378,
    1.37307373,
    1.32320467,
    1.50261683,
    6.91050685,
    1.06083157,
    1.20841351,
    2.92719840,
    2.82178183,
    2.05765813,
    1.84621661,
    1.04677388,
    2.13801850,
    1.39654855,
    1.13037727,
    1.37887598,
    1.03221650,
    1.15981176,
    1.09896163,
    1.88624084,
    1.43459062,
    1.54587662,
    1.48604380,
    2.06197392,
    1.97079675,
    4.31388672,
    2.94376994,
    3.48708489,
    1.09674551,
    2.46926816,
    1.23705940,
    1.57512843,
    1.15595205,
    1.18432818,
    1.54298936,
    1.60600489,
    1.07361787,
    1.38666771,
    1.45533003,
    1.78940830,
    1.33799752,
    1.12955889,
    4.59400278,
    1.15170228,
    1.39346636,
    1.61408789,
    2.21293753,
    5.33166143,
    1.18147947,
    1.54426891,
    1.32496426,
    1.25037632,
    3.31244261,
    1.36211171,
    1.82239599,
    1.75235087,
    1.67044831,
    1.24802350,
    1.34776327,
    1.34740665,
    1.30664120,
    1.06852680,
    1.22513631,
    1.25310923,
    1.36394926,
    1.07796356,
    3.10823551,
    1.46770227,
    1.40264883,
    1.08787681,
    1.26460358,
    1.10348946,
    2.03168839,
    1.09435135,
    1.66991715,
    1.19738540,
    1.28922229,
    2.85704149,
    1.33952521,
    1.73497688,
    2.90052876,
    5.34596348,
    1.36399078,
    3.38399264,
    1.06089658,
    1.09370142,
    1.37523679,
    3.01964907,
    1.40684792,
    1.11312672,
    2.44666372,
    1.73953904,
    1.65569280,
    1.05813000,
    2.02893022,
    1.72877601,
    1.55758690,
    1.83904301,
    1.14316984,
    1.17792251,
    1.44106281,
    9.67126482,
    1.93207441,
    1.08242887,
    2.87271135,
    2.19095115,
    2.13195479,
    1.02355472,
    1.18218470,
    1.30907724,
    1.13291587,
    2.85659336,
    12.62726889,
    1.18818589,
    1.02852443,
    1.12838670,
    1.36349361,
    1.34817100,
    1.30535737,
    3.22225028,
    1.28680350,
    1.83979657,
    1.11088952,
    1.43866586,
    8.52587567,
    3.73988696,
    2.65816056,
    1.17373111,
    2.61567111,
    3.24024082,
    2.96798864,
    1.05335616,
    1.31159271,
    1.36485918,
    1.24988767,
    7.80609746,
    1.54892174,
    1.10682809,
    1.21728827,
    1.20429971,
    1.72719055,
    1.78534831,
    1.04414979,
    1.25646988,
    1.19788383,
    1.08854812,
    1.04859628,
    1.04676064,
    5.07295341,
    3.83595341,
    1.61079632,
    1.10528426,
    1.15050241,
    2.78129736,
    1.25494119,
    1.28692155,
    1.06812292,
    3.29393761,
    1.37542463,
    1.67241953,
    1.21698665,
    10.57727604,
    8.63598976,
    1.18886984,
    1.30609583,
    9.47777457,
    1.69612900,
    2.23002585,
    1.58461615,
    1.04110023,
    3.08140806,
    1.39599251,
    1.06575789,
    1.29741002,
    1.75253864,
    1.82594258,
    1.15111702,
    1.17370053,
    1.15254396,
    1.94401179,
    5.36344596,
    4.66322185,
    1.15073993,
    3.21478159,
    1.39843306,
    1.03961906,
    5.72845289,
    1.72454161,
    1.04610704,
    1.38975310,
    1.77732797,
    1.10139931,
    2.23656355,
    1.89952669,
    1.72136921,
    1.15798212,
    1.59545971,
    1.08789161,
    1.93272206,
    2.57480708,
    1.04977784,
    2.00874078,
    3.40065861,
    1.00978603,
    3.97804652,
    1.54762586,
    1.01015493,
    1.15148220,
    1.15246483,
    19.67426012,
    1.33290993,
    2.33137522,
    1.12841749,
    1.73407057,
    2.00469493,
    1.27418995,
    1.49814918,
    1.10398785,
    1.20063760,
    1.05536150,
    1.87616599,
    1.49305736,
    1.60241346,
    1.16666060,
    1.05013736,
    1.77929210,
    1.00206028,
    3.41096863,
    1.47499925,
    1.14071240,
    1.65361002,
    1.76466424,
    8.49298111,
    1.41069285,
    2.11681605,
    4.90260842,
    1.13029658,
    1.20802818,
    1.42525579,
    1.00310774,
    1.08082363,
    9.95194247,
    2.82773946,
    2.77420002,
    1.82543685,
    1.28557906,
    1.97711769,
    1.19001264,
    1.95712650,
    1.54230291,
    1.31625757,
    2.36364128,
    1.11523099,
    1.00343756,
    1.71299382,
    1.44667100,
    2.38154868,
    1.41174217,
    1.80660493,
    1.51020853,
    1.16761479,
    1.25898190,
    1.18150781,
    1.58465451,
    2.03560597,
    3.48531184,
    1.21187672,
    1.35111036,
    1.02954922,
    1.90892663,
    3.99078548,
    5.67385199,
    4.38055264,
    1.17446048,
    13.41617858,
    1.60241740,
    1.14811206,
    4.68120263,
    3.83763710,
    2.66095263,
    1.83338503,
    4.75973082,
    1.08982301,
    4.04104276,
    1.34220189,
    1.06135891,
    2.71185882,
    1.46085873,
    1.09915614,
    10.35178646,
    2.54402271,
    2.65696704,
    1.31388649,
    1.02942408,
    1.57780748,
    1.01552697,
    2.24860361,
    2.22011778,
    1.13595134,
    1.11492512,
    2.11966788,
    1.20420149,
    1.11112428,
    3.09324603,
    2.87240762,
    1.50486558,
    1.92227231,
    4.12480449,
    1.58244751,
    1.69922308,
    6.28134904,
    2.91944178,
    1.85386792,
    1.41799519,
    1.64636127,
    2.05837832,
    1.07153521,
    2.05376943,
    2.60053549,
    1.09773382,
    1.54671309,
    1.68007415,
    3.43941489,
    1.41601033,
    2.00237256,
    1.20830978,
    1.25582363,
    1.10830461,
    1.24850906,
    1.88035202,
    1.70557719,
    1.04191110,
    1.33501003,
    1.33554804,
    1.36935735,
    4.79153510,
    1.06566392,
    1.14495966,
    1.90020028,
    1.08266994,
    1.20588153,
    1.40730214,
    4.34320304,
    1.71762330,
    1.06620797,
    1.39695239,
    1.03024563,
    3.94971225,
    5.02945862,
    1.06145571,
    1.42511911,
    2.13889169,
    1.04986044,
    1.91400616,
    5.50708156,
    1.52870464,
    1.11303137,
    1.05282759,
    1.83793940,
    3.05244089,
    2.64499634,
    1.51688076,
    2.63350152,
    1.31014486,
    1.69462474,
    1.67792130,
    1.34236945,
    1.02358460,
    1.04593509,
    1.04007620,
    1.87990081,
    1.28585413,
    1.01636283,
    3.55338495,
    1.19542700,
    1.23630628,
    1.32321942,
    4.03762786,
    1.25379147,
    1.12330233,
    1.24966418,
    1.26323243,
    1.14779989,
    1.20378343,
    1.01531796,
    1.44500318,
    1.72723672,
    15.68799957,
    1.37641063,
    7.00788166,
    3.89674130,
    1.68303382,
    1.10089816,
    1.72831362,
    2.70479861,
    1.75821836,
    2.32404215,
    2.64165162,
    1.42441301,
    1.83256456,
    1.12548819,
    4.81273800,
    2.52840227,
    2.68430190,
    1.00928919,
    1.02438446,
    1.33909276,
    2.32261242,
    1.01299124,
    1.07614975,
    1.66823898,
    1.97172786,
    1.01707292,
    1.68325092,
    1.76834032,
    1.08952069,
    1.02265517,
    1.96843176,
    1.83351706,
    1.92704772,
    18.44811035,
    1.00178046,
    2.70555953,
    1.35839004,
    1.04834633,
    1.26649072,
    2.87152600,
    4.12536409,
    1.25200853,
    1.71199647,
    1.61175739,
    1.26313274,
    1.75224120,
    2.70412800,
    1.33998630,
    1.61271556,
    2.65784769,
    10.38771107,
    1.33121364,
    1.01207979,
    2.00238212,
    2.50195600,
    1.96917548,
    1.71618169,
    1.37050585,
    10.11861690,
    1.18339112,
    1.80083386,
    2.88582103,
    1.21935761,
    2.37900131,
    1.49449487,
    4.75106319,
    2.33977804,
    2.87963540,
    1.01807103,
    3.74847411,
    1.71981276,
    1.50726964,
    1.20723219,
    1.37904840,
    1.04565533,
    1.59877004,
    1.11481349,
    2.17320556,
    2.07108468,
    1.23274077,
    1.75180110,
    1.27558910,
    1.63240839,
    1.58760550,
    1.01266256,
    1.30395323,
    1.14618521,
    1.02385023,
    2.24198100,
    1.26765471,
    1.15855534,
    1.83936251,
    1.32970987,
    1.25844192,
    1.31133485,
    4.74300303,
    6.19325623,
    1.31832913,
    3.97645560,
    1.00545340,
    1.24431862,
    1.25855820,
    1.15514241,
    1.35986865,
    1.72446070,
    1.13069572,
    2.45890932,
    1.00394684,
    1.03533631,
    1.87698184,
    2.34576160,
    1.03997887,
    1.02694456,
    2.52227100,
    2.66278467,
    1.17002905,
    3.42239624,
    2.46753038,
    1.17103623,
    1.07832850,
    1.42782632,
    1.29110546,
    1.03435772,
    1.33512109,
    1.14337058,
    1.34103634,
    1.15155161,
    2.59805360,
    2.09650343,
    1.53399143,
    1.02319185,
    1.32210667,
    1.05720671,
    1.20882651,
    2.34881662,
    1.05163662,
    3.26219380,
    10.58124156,
    1.07283644,
    1.02105339,
    1.23268679,
    1.81469813,
    1.49393533,
    1.29760853,
    5.37676625,
    1.02529938,
    1.86815537,
    1.57961476,
    3.77408176,
    2.79405589,
    3.25246617,
    1.63913824,
    3.12133428,
    1.03787574,
    4.17232960,
    1.33406468,
    1.57119541,
    1.13675102,
    3.42874720,
    1.13066210,
    1.33896458,
    1.23883935,
    1.35272696,
    1.15172654,
    2.18633755,
    1.23251881,
    1.59742606,
    1.08718410,
    1.06168544,
    1.19926517,
    1.00214807,
    1.29121086,
    3.44575916,
    1.26524744,
    1.16718301,
    4.11789988,
    1.25375574,
    1.35753968,
    1.69247751,
    1.28473150,
    2.20669768,
    1.53213883,
    2.30598771,
    1.68420243,
    1.37320685,
    2.08619411,
    1.26990265,
    1.82215898,
    1.10656122,
    1.40229835,
    1.11896817,
    1.00127366,
    2.88218857,
    2.79105702,
    1.28699225,
    1.15929737,
    1.07928363,
    10.54130128,
    8.79261793,
    1.15699405,
    1.69050500,
    2.76586152,
    1.22802809,
    1.38014655,
    2.19208585,
    1.64409370,
    1.46918371,
    2.99582898,
    1.37759923,
    1.29776632,
    1.82884215,
    2.67317357,
    1.37063041,
    1.26884340,
    1.07874723,
    1.48172681,
    1.01771849,
    2.40642202,
    1.37115433,
    1.05954574,
    2.12998246,
    2.34178079,
    1.54515623,
    1.00179963,
    2.12228030,
    1.46007334,
    1.20664530,
    1.31417158,
    1.03322353,
    1.95420119,
    1.30541569,
    1.15016102,
    2.17036908,
    2.81707947,
    1.16173181,
    2.01742565,
    1.02478594,
    1.57428560,
    1.21209176,
    2.20735202,
    1.12935761,
    2.08850147,
    1.05353378,
    1.02324910,
    1.49636415,
    1.48061026,
    2.25651770,
    3.04296168,
    1.24380806,
    1.07707360,
    2.00284318,
    10.02810932,
    3.38695326,
    6.82841534,
    2.13556915,
    1.19152238,
];

#[test]
fn power_law_fit_matches_igraph_unit_test_output() {
    // tests/unit/igraph_power_law_fit.out, first two results.
    let fit = misc::power_law_fit(&PLFIT_CONTINUOUS, None, false).unwrap();
    assert!(fit.continuous);
    assert_close(fit.alpha, 2.81976, 1e-5);
    assert_close(fit.xmin, 1.00979, 1e-5);
    assert_close(fit.log_likelihood, -946.14703, 1e-5);
    assert_close(fit.ks_statistic, 0.01454, 1e-5);

    let fit = misc::power_law_fit(&PLFIT_CONTINUOUS, Some(2.0), false).unwrap();
    assert!(fit.continuous);
    assert_close(fit.alpha, 2.81157, 1e-5);
    assert_eq!(fit.xmin, 2.0);
    assert_close(fit.log_likelihood, -463.92064, 1e-5);
    assert_close(fit.ks_statistic, 0.05091, 1e-5);

    // The continuous maximum likelihood estimator has a closed form:
    // alpha = 1 + n / sum(ln(x_i / xmin)) over the n samples >= xmin.
    let tail: Vec<f64> = PLFIT_CONTINUOUS
        .iter()
        .copied()
        .filter(|&x| x >= 2.0)
        .collect();
    let mle = 1.0 + tail.len() as f64 / tail.iter().map(|x| (x / 2.0).ln()).sum::<f64>();
    assert_close(fit.alpha, mle, 1e-4);
}

#[test]
fn sir_matches_igraph_unit_test_output() {
    // tests/unit/igraph_sir.c and .out: the same sequence of calls with the
    // same seed gives exactly the same epidemics.
    rng::seed(43).unwrap();
    let states = |g: &Graph, beta: f64, gamma: f64| -> Vec<(Vec<i64>, Vec<i64>, Vec<i64>)> {
        g.sir(beta, gamma, 2)
            .unwrap()
            .into_iter()
            .map(|r| (r.susceptible, r.infected, r.recovered))
            .collect()
    };
    let single = Graph::new(1, false);
    let two = Graph::new(2, false);
    let line = path(5);
    let full = Graph::full(5, false, false).unwrap();
    let one = (vec![0, 0], vec![1, 0], vec![0, 1]);
    assert_eq!(states(&single, 0.1, 0.0001), vec![one.clone(), one]);
    let two_runs = (vec![1, 1], vec![1, 0], vec![0, 1]);
    assert_eq!(states(&two, 1.0, 1.0), vec![two_runs.clone(), two_runs]);
    assert_eq!(
        states(&line, 1.0, 1.0),
        vec![
            (vec![4, 3, 3, 3], vec![1, 2, 1, 0], vec![0, 0, 1, 2]),
            (
                vec![4, 3, 2, 2, 2, 2],
                vec![1, 2, 3, 2, 1, 0],
                vec![0, 0, 0, 1, 2, 3]
            ),
        ]
    );
    let lone = (vec![4, 4], vec![1, 0], vec![0, 1]);
    assert_eq!(states(&line, 0.0001, 1.0), vec![lone.clone(), lone]);
    assert_eq!(
        states(&full, 1.0, 1.0),
        vec![
            (
                vec![4, 3, 2, 1, 0, 0, 0, 0, 0, 0],
                vec![1, 2, 3, 4, 5, 4, 3, 2, 1, 0],
                vec![0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
            ),
            (
                vec![4, 3, 3, 2, 1, 0, 0, 0, 0, 0],
                vec![1, 2, 1, 2, 3, 4, 3, 2, 1, 0],
                vec![0, 0, 1, 1, 1, 1, 2, 3, 4, 5]
            ),
        ]
    );
}

#[test]
fn random_sample_handles_extreme_intervals() {
    // igraph negates `low`, which is undefined behavior in C for i64::MIN:
    // the wrapper samples from a shifted interval instead.
    assert_eq!(
        misc::random_sample(i64::MIN, i64::MIN + 3, 4).unwrap(),
        vec![i64::MIN, i64::MIN + 1, i64::MIN + 2, i64::MIN + 3]
    );
    let s = misc::random_sample(i64::MIN, -2, 5).unwrap();
    assert_eq!(s.len(), 5);
    assert!(s.windows(2).all(|w| w[0] < w[1]));
    assert!(s.iter().all(|&x| x <= -2));
    assert_eq!(
        misc::random_sample(i64::MAX - 2, i64::MAX, 3).unwrap(),
        vec![i64::MAX - 2, i64::MAX - 1, i64::MAX]
    );
    // Intervals with more than i64::MAX elements.
    for (low, high) in [(i64::MIN, i64::MAX), (i64::MIN, 0), (0, i64::MAX)] {
        assert_eq!(
            misc::random_sample(low, high, 1).unwrap_err().kind(),
            ErrorKind::Overflow
        );
        // An empty sample is always fine.
        assert!(misc::random_sample(low, high, 0).unwrap().is_empty());
    }
    // Shifting the interval shifts the sample, with the same generator state.
    rng::seed(8).unwrap();
    let a = misc::random_sample(1000, 2_000_999, 30).unwrap();
    rng::seed(8).unwrap();
    let b = misc::random_sample(0, 1_999_999, 30).unwrap();
    assert_eq!(a, b.iter().map(|x| x + 1000).collect::<Vec<_>>());
}

#[test]
fn power_law_fit_thresholds_and_degenerate_samples() {
    let data = [1.0, 1.0, 2.0, 3.0, 1.0, 5.0, 2.0, 8.0];
    // xmin = 0 is rejected: at least 1 (discrete), positive (continuous).
    for force_continuous in [false, true] {
        assert_eq!(
            misc::power_law_fit(&data, Some(0.0), force_continuous)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidValue
        );
    }
    let all = misc::power_law_fit(&data, Some(1.0), false).unwrap();
    assert!(!all.continuous && all.alpha.is_finite());
    // No sample reaches xmin: degenerate discrete fit, failed continuous fit.
    let fit = misc::power_law_fit(&data, Some(100.0), false).unwrap();
    assert_eq!(fit.alpha, f64::INFINITY);
    assert!(fit.log_likelihood.is_nan());
    assert_eq!(
        misc::power_law_fit(&data, Some(100.0), true)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    // A single sample cannot be fitted.
    for x in [2.0, 2.5] {
        assert!(misc::power_law_fit(&[x], None, false).is_err());
    }
}

#[test]
fn p_value_rejects_invalid_models_instead_of_crashing() {
    // PowerLawFit has public fields: igraph would crash (empty sample) or
    // loop forever (alpha <= 1 or NaN for a discrete law) on such models.
    let data = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 1.0, 1.0, 2.0];
    let fit = misc::power_law_fit(&data, None, false).unwrap();
    assert!(!fit.continuous);
    let invalid = |f: &misc::PowerLawFit<'_>| f.p_value(0.2).unwrap_err().kind();
    let mut f = fit.clone();
    f.data = &[];
    assert_eq!(invalid(&f), ErrorKind::InvalidValue);
    f = fit.clone();
    f.data = &[1.0, f64::NAN];
    assert_eq!(invalid(&f), ErrorKind::InvalidValue);
    for alpha in [0.5, 1.0, f64::NAN, f64::INFINITY] {
        f = fit.clone();
        f.alpha = alpha;
        assert_eq!(invalid(&f), ErrorKind::InvalidValue, "alpha = {alpha}");
    }
    for xmin in [f64::NAN, -3.0, 0.0, 0.5, 1e300] {
        f = fit.clone();
        f.xmin = xmin;
        assert_eq!(invalid(&f), ErrorKind::InvalidValue, "xmin = {xmin}");
    }
    f = fit.clone();
    f.continuous = true;
    f.xmin = 0.0;
    assert_eq!(invalid(&f), ErrorKind::InvalidValue);
    // A degenerate fit returned by igraph itself.
    let degenerate = misc::power_law_fit(&data, Some(100.0), false).unwrap();
    assert_eq!(invalid(&degenerate), ErrorKind::InvalidValue);
    // Precisions giving no rounds, or more rounds than a C long can count.
    for precision in [0.6, 1e-10, f64::MIN_POSITIVE, f64::NAN, -0.1] {
        assert_eq!(
            fit.p_value(precision).unwrap_err().kind(),
            ErrorKind::InvalidValue,
            "precision = {precision}"
        );
    }
    // The valid model still works.
    rng::seed(4).unwrap();
    let p = fit.p_value(0.2).unwrap();
    assert!((0.0..=1.0).contains(&p));
}

// ---------------------------------------------------------------------------
// Audit regressions: handlers calling failing igraph functions, guard order,
// non-finite power-law samples.
// ---------------------------------------------------------------------------

/// Makes a failing igraph call (an edge to a missing vertex): igraph's error
/// handler then frees the current level of its "finally" stack.
fn failing_igraph_call() {
    let mut g = Graph::empty(2, false).unwrap();
    assert_eq!(
        g.add_edge(0, 99).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
}

fn finally_stack_size() -> i32 {
    unsafe { ffi::IGRAPH_FINALLY_STACK_SIZE() }
}

#[test]
fn failing_igraph_call_in_progress_handler_leaves_the_computation_intact() {
    let g = cycle(30);
    let expected = betweenness(&g).unwrap();
    let calls = Rc::new(Cell::new(0));
    let counter = Rc::clone(&calls);
    let bw = misc::with_progress_handler(
        move |_, _| {
            counter.set(counter.get() + 1);
            failing_igraph_call();
            ControlFlow::Continue(())
        },
        || betweenness(&g),
    )
    .unwrap();
    assert!(calls.get() > 0);
    assert_eq!(bw, expected);
    assert_eq!(finally_stack_size(), 0);
}

#[test]
fn failing_igraph_call_in_interruption_handler_leaves_the_computation_intact() {
    let g = cycle(30);
    let expected = betweenness(&g).unwrap();
    let calls = Rc::new(Cell::new(0));
    let counter = Rc::clone(&calls);
    let bw = misc::with_interruption_handler(
        move || {
            counter.set(counter.get() + 1);
            failing_igraph_call();
            false
        },
        || betweenness(&g),
    )
    .unwrap();
    assert!(
        calls.get() > 0,
        "betweenness polls the interruption handler"
    );
    assert_eq!(bw, expected);
    assert_eq!(finally_stack_size(), 0);
}

#[test]
fn failing_igraph_call_in_status_handler_is_harmless() {
    let calls = Rc::new(Cell::new(0));
    let counter = Rc::clone(&calls);
    misc::with_status_handler(
        move |_| {
            counter.set(counter.get() + 1);
            failing_igraph_call();
            ControlFlow::Continue(())
        },
        || {
            misc::status("one").unwrap();
            misc::status("two").unwrap();
        },
    );
    assert_eq!(calls.get(), 2);
    assert_eq!(finally_stack_size(), 0);
}

/// A progress handler counting its calls in `hits`.
fn counting_progress(hits: &Rc<Cell<u32>>) -> misc::ProgressHandlerGuard {
    let hits = Rc::clone(hits);
    misc::set_progress_handler(move |_, _| {
        hits.set(hits.get() + 1);
        ControlFlow::Continue(())
    })
}

#[test]
fn progress_guards_dropped_out_of_order_uninstall_their_own_handler() {
    let (a_hits, b_hits) = (Rc::new(Cell::new(0)), Rc::new(Cell::new(0)));
    let a = counting_progress(&a_hits);
    let b = counting_progress(&b_hits);
    misc::progress("x", 1.0).unwrap();
    assert_eq!((a_hits.get(), b_hits.get()), (0, 1), "b is active");
    // Dropping the older guard first leaves b active...
    drop(a);
    misc::progress("x", 2.0).unwrap();
    assert_eq!((a_hits.get(), b_hits.get()), (0, 2));
    // ... and dropping b afterwards does not bring a back.
    drop(b);
    misc::progress("x", 3.0).unwrap();
    assert_eq!((a_hits.get(), b_hits.get()), (0, 2), "no handler any more");
}

#[test]
fn progress_guards_in_a_vec_can_be_dropped_in_any_order() {
    let hits: Vec<Rc<Cell<u32>>> = (0..4).map(|_| Rc::new(Cell::new(0))).collect();
    let mut guards: Vec<Option<misc::ProgressHandlerGuard>> =
        hits.iter().map(|h| Some(counting_progress(h))).collect();
    let report = || misc::progress("x", 0.0).unwrap();
    let counts = || hits.iter().map(|h| h.get()).collect::<Vec<_>>();
    report();
    assert_eq!(counts(), [0, 0, 0, 1]);
    guards[1] = None; // not the active one: nothing changes
    report();
    assert_eq!(counts(), [0, 0, 0, 2]);
    guards[3] = None; // the active one: the most recent survivor (2) takes over
    report();
    assert_eq!(counts(), [0, 0, 1, 2]);
    // A C-only handler on top, then removed again.
    {
        let _stderr = misc::set_progress_handler_stderr();
        guards[2] = None; // below the stderr handler
    }
    report();
    assert_eq!(counts(), [1, 0, 1, 2], "0 is the only one left");
    guards.clear();
    report();
    assert_eq!(counts(), [1, 0, 1, 2]);
}

#[test]
fn status_and_interruption_guards_dropped_out_of_order() {
    let log = Rc::new(RefCell::new(Vec::<&str>::new()));
    let (l1, l2) = (Rc::clone(&log), Rc::clone(&log));
    let a = misc::set_status_handler(move |_| {
        l1.borrow_mut().push("a");
        ControlFlow::Continue(())
    });
    let b = misc::set_status_handler(move |_| {
        l2.borrow_mut().push("b");
        ControlFlow::Continue(())
    });
    drop(a);
    misc::status("s").unwrap();
    drop(b);
    misc::status("s").unwrap();
    assert_eq!(*log.borrow(), ["b"]);

    let ia = misc::set_interruption_handler(|| true);
    let ib = misc::set_interruption_handler(|| false);
    assert!(!misc::allow_interruption());
    drop(ia);
    assert!(!misc::allow_interruption(), "ib is still active");
    drop(ib);
    assert!(
        !misc::allow_interruption(),
        "no handler: ia is not reinstalled"
    );
}

#[test]
fn guard_dropped_from_inside_its_own_handler_call() {
    // The handler owns another guard and drops it while running; then the
    // outer guard is dropped from within the handler too.
    let inner_hits = Rc::new(Cell::new(0));
    let slot: Rc<RefCell<Option<misc::ProgressHandlerGuard>>> = Rc::new(RefCell::new(None));
    let outer_hits = Rc::new(Cell::new(0));
    let (s2, h2) = (Rc::clone(&slot), Rc::clone(&outer_hits));
    let outer = misc::set_progress_handler(move |_, _| {
        h2.set(h2.get() + 1);
        // Drops the guard stored by the test (it uninstalls this very handler).
        drop(s2.borrow_mut().take());
        ControlFlow::Continue(())
    });
    *slot.borrow_mut() = Some(outer);
    let below = counting_progress(&inner_hits);
    // `below` is on top: the outer handler is parked.
    misc::progress("x", 0.0).unwrap();
    assert_eq!((outer_hits.get(), inner_hits.get()), (0, 1));
    drop(below);
    // Now the outer handler runs and drops its own guard.
    misc::progress("x", 0.0).unwrap();
    assert_eq!(outer_hits.get(), 1);
    misc::progress("x", 0.0).unwrap();
    assert_eq!(outer_hits.get(), 1, "uninstalled by its own call");
    assert!(slot.borrow().is_none());
}

#[test]
fn power_law_fit_rejects_infinite_samples() {
    for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        for force_continuous in [false, true] {
            let data = [bad, 1.0, 2.0, 3.0];
            assert_eq!(
                misc::power_law_fit(&data, None, force_continuous)
                    .unwrap_err()
                    .kind(),
                ErrorKind::InvalidValue,
                "sample {bad}, continuous = {force_continuous}"
            );
        }
    }
    // Huge discrete samples make plfit's L-BFGS fit fail (and igraph then
    // reads a dangling error message): they are rejected up front...
    let huge = [7.75e202, 2.17e161];
    let err = misc::power_law_fit(&huge, None, false).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    assert!(err.message().contains("2^62"), "{err}");
    for data in [&[1e300, 1.0, 2.0, 3.0][..], &[1.0, 2.0, 4.7e18]] {
        assert_eq!(
            misc::power_law_fit(data, None, false).unwrap_err().kind(),
            ErrorKind::InvalidValue
        );
    }
    assert_eq!(
        misc::power_law_fit(&[1.0, 2.0], Some(1e19), false)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    // ... but fitted as continuous samples, and non-integer ones are
    // continuous anyway.
    for data in [&huge[..], &[1e300, 1.0, 2.0, 3.0], &[1e300, 1.5, 2.0, 3.0]] {
        let fit = misc::power_law_fit(data, None, true).unwrap();
        assert!(fit.continuous && fit.alpha > 1.0);
    }
    let fit = misc::power_law_fit(&[1e300, 1.5, 2.0, 3.0], None, false).unwrap();
    assert!(fit.continuous);
    // Discrete samples just below the bound are fitted.
    let fit = misc::power_law_fit(&[1.0, 2.0, 3.0, 5.0, 60.0, 4e18], None, false).unwrap();
    assert!(!fit.continuous && fit.alpha > 1.0);
}

#[test]
fn p_value_rejects_non_finite_and_huge_discrete_samples() {
    let data = [1.0, 2.0, 3.0, 5.0, 60.0, 7e18];
    // power_law_fit refuses such a discrete sample; build the model by hand.
    let mut fit = misc::power_law_fit(&data[..5], None, false).unwrap();
    assert!(!fit.continuous);
    fit.data = &data;
    // Samples of 2^62 and more lead plfit's sampler to overflowing draws.
    assert_eq!(
        fit.p_value(0.1).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    let mut f = fit.clone();
    f.data = &[1.0, 2.0, f64::INFINITY];
    assert_eq!(f.p_value(0.1).unwrap_err().kind(), ErrorKind::InvalidValue);
    // The same huge sample is fine for a continuous model.
    let cont = misc::power_law_fit(&data, None, true).unwrap();
    assert!(cont.continuous);
    rng::seed(3).unwrap();
    let p = cont.p_value(0.2).unwrap();
    assert!((0.0..=1.0).contains(&p));
    // Moderate samples are accepted.
    let small = [1.0, 2.0, 3.0, 5.0, 60.0, 1e6];
    let fit = misc::power_law_fit(&small, None, false).unwrap();
    rng::seed(3).unwrap();
    assert!((0.0..=1.0).contains(&fit.p_value(0.2).unwrap()));
}

//! Regression tests for igraph's property-cache bug (1.0.0 and 1.0.1).
//!
//! Many igraph functions that ignore edge directions build an adjacency list
//! with `IGRAPH_ALL` and `IGRAPH_NO_MULTIPLE`, which skips the removal of
//! duplicate neighbors when the cache says "no multi-edges". On a directed
//! graph with mutual pairs `u -> v`, `v -> u` that is wrong: each endpoint
//! then lists the other twice. The wrappers drop the cache around those
//! calls (`Graph::with_fresh_multi_cache`).
//!
//! Every check below runs twice, on a fresh graph and on a graph whose cache
//! was primed with `has_multiple()` (which correctly caches "no
//! multi-edges"), and compares the result with the same computation on the
//! equivalent undirected simple graph. Afterwards the cache must still say
//! the directed graph is simple.

use igraph::{paths::SimplePathsOptions, prelude::*};

/// Directed graph where every undirected edge of `undirected()` appears as a
/// mutual pair, plus one single-direction edge.
fn directed() -> Graph {
    Graph::from_edges(
        &[
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 0),
            (0, 2),
            (0, 3),
            (3, 0),
            (3, 2),
            (2, 4),
        ],
        5,
        true,
    )
    .unwrap()
}

/// The undirected simple graph with the same adjacency as `directed()`.
fn undirected() -> Graph {
    Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (0, 3), (3, 2), (2, 4)], 5, false).unwrap()
}

/// Runs `f` on a fresh and on a cache-primed directed graph, checks that it
/// matches `expected` and that the cache still reports a simple graph.
fn check<T: PartialEq + std::fmt::Debug>(name: &str, expected: T, f: impl Fn(&Graph) -> T) {
    for primed in [false, true] {
        let g = directed();
        if primed {
            assert!(!g.has_multiple().unwrap());
        }
        let got = f(&g);
        assert_eq!(got, expected, "{name} (primed cache: {primed})");
        assert!(
            !g.has_multiple().unwrap(),
            "{name} corrupted the cache (primed: {primed})"
        );
        assert!(
            g.is_simple(true).unwrap(),
            "{name} corrupted the cache (primed: {primed})"
        );
    }
}

#[test]
fn transitivity_ignores_mutual_pairs_whatever_the_cache() {
    let u = undirected();
    let nan = TransitivityMode::Zero;
    check(
        "transitivity_undirected",
        u.transitivity_undirected(nan).unwrap(),
        |g| g.transitivity_undirected(nan).unwrap(),
    );
    check(
        "transitivity_local_undirected",
        u.transitivity_local_undirected(.., nan).unwrap(),
        |g| g.transitivity_local_undirected(.., nan).unwrap(),
    );
    check(
        "transitivity_avglocal_undirected",
        u.transitivity_avglocal_undirected(nan).unwrap(),
        |g| g.transitivity_avglocal_undirected(nan).unwrap(),
    );
}

#[test]
fn barrat_and_edge_clustering_run_safely() {
    // Barrat and `ecc` interpret the (multi-)edges themselves, so their values
    // on the directed graph legitimately differ from the undirected ones: only
    // determinism across cache states and cache integrity are checked.
    let barrat = directed()
        .transitivity_barrat(.., None, TransitivityMode::Zero)
        .unwrap();
    check("transitivity_barrat", barrat, |g| {
        g.transitivity_barrat(.., None, TransitivityMode::Zero)
            .unwrap()
    });
    let ecc = directed().ecc(.., 3, true, true).unwrap();
    check("ecc", ecc, |g| g.ecc(.., 3, true, true).unwrap());
}

#[test]
fn eccentricity_radius_center_and_pseudo_diameter_ignore_mutual_pairs() {
    let u = undirected();
    let all = NeighborMode::All;
    check(
        "eccentricity",
        u.eccentricity(.., None, all).unwrap(),
        |g| g.eccentricity(.., None, all).unwrap(),
    );
    check("radius", u.radius(None, all).unwrap(), |g| {
        g.radius(None, all).unwrap()
    });
    check("graph_center", u.graph_center(None, all).unwrap(), |g| {
        g.graph_center(None, all).unwrap()
    });
    check(
        "pseudo_diameter",
        u.pseudo_diameter(None, Some(4), false, true)
            .unwrap()
            .length,
        |g| {
            g.pseudo_diameter(None, Some(4), false, true)
                .unwrap()
                .length
        },
    );
}

#[test]
fn simple_paths_are_not_duplicated_by_mutual_pairs() {
    let opts = SimplePathsOptions::default();
    let mut expected = undirected()
        .get_all_simple_paths(4, .., NeighborMode::All, opts)
        .unwrap();
    expected.sort();
    check("get_all_simple_paths", expected, |g| {
        let mut paths = g
            .get_all_simple_paths(4, .., NeighborMode::All, opts)
            .unwrap();
        paths.sort();
        paths
    });
}

/// `directed()` with each mutual pair replaced by a single arc: the same
/// neighborhoods in `All` mode, but without mutual pairs.
fn oriented() -> Graph {
    Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (0, 3), (3, 2), (2, 4)], 5, true).unwrap()
}

#[test]
fn local_scan_them_counts_each_neighbor_once() {
    // `us` and `them` must have the same directedness. `oriented()` has the
    // same `All`-mode neighborhoods as `directed()` without mutual pairs, so
    // it gives the expected counts.
    let them = directed();
    let reference = oriented();
    check(
        "local_scan_1_ecount_them",
        reference
            .local_scan_1_ecount_them(&them, None, NeighborMode::All)
            .unwrap(),
        |g| {
            g.local_scan_1_ecount_them(&them, None, NeighborMode::All)
                .unwrap()
        },
    );
    check(
        "local_scan_k_ecount_them",
        reference
            .local_scan_k_ecount_them(&them, 2, None, NeighborMode::All)
            .unwrap(),
        |g| {
            g.local_scan_k_ecount_them(&them, 2, None, NeighborMode::All)
                .unwrap()
        },
    );
}

#[test]
fn voronoi_rejects_mutual_pairs_and_keeps_the_cache() {
    // igraph requires a graph without mutual pairs here, so the adjacency
    // list bug cannot be reached; the call must fail cleanly either way.
    check("community_voronoi", ErrorKind::InvalidValue, |g| {
        g.community_voronoi(None, None, NeighborMode::All, None)
            .unwrap_err()
            .kind()
    });
    let ok = oriented()
        .community_voronoi(None, None, NeighborMode::All, None)
        .unwrap();
    assert_eq!(ok.membership.len(), 5);
}

#[test]
fn connect_neighborhood_adds_no_duplicate_edges() {
    for primed in [false, true] {
        let mut g = directed();
        if primed {
            assert!(!g.has_multiple().unwrap());
        }
        g.connect_neighborhood(2, NeighborMode::All).unwrap();
        // Directed output: every new edge goes from a vertex to a vertex at
        // undirected distance 2; none may be duplicated.
        let mut edges = g.edge_list();
        let n = edges.len();
        edges.sort();
        edges.dedup();
        assert_eq!(edges.len(), n, "duplicate edges (primed cache: {primed})");
        assert!(!g.has_multiple().unwrap(), "primed cache: {primed}");
    }
}

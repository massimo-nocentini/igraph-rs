//! ARPACK keeps thread-local state, so an ARPACK computation must never be
//! started while another one is running on the same thread (e.g. from a
//! matrix-vector callback or an interruption handler). Every wrapper whose C
//! function may run ARPACK refuses such nesting with `ErrorKind::Failure`.

use igraph::{
    centrality::{PageRankAlgo, PageRankOptions},
    linalg::{ArpackOptions, ArpackWhich, arpack_rssolve},
    prelude::*,
};

/// Adjacency matrix of the `n`-cycle as a matrix-vector product.
fn cycle_matvec(n: usize) -> impl Fn(&[f64], &mut [f64]) {
    move |x: &[f64], y: &mut [f64]| {
        for i in 0..n {
            y[i] = x[(i + 1) % n] + x[(i + n - 1) % n];
        }
    }
}

#[test]
fn centrality_and_community_arpack_wrappers_refuse_nesting() {
    let n = 12;
    let karate = Graph::famous("Zachary").unwrap();
    let options = ArpackOptions::default()
        .with_nev(1)
        .with_which(ArpackWhich::LargestAlgebraic);
    let mut outcomes: Vec<(&str, std::result::Result<(), ErrorKind>)> = Vec::new();
    let mut checked = false;
    let matvec = cycle_matvec(n);
    rng::seed(3).unwrap();
    let res = arpack_rssolve(
        n,
        |x: &[f64], y: &mut [f64]| {
            if !checked {
                checked = true;
                let kind = |r: Result<()>| r.map_err(|e| e.kind());
                let arpack = PageRankOptions {
                    algo: PageRankAlgo::Arpack,
                    ..Default::default()
                };
                let prpack = PageRankOptions {
                    algo: PageRankAlgo::Prpack,
                    ..Default::default()
                };
                outcomes.push((
                    "eigenvector_centrality",
                    kind(
                        karate
                            .eigenvector_centrality(NeighborMode::All, None)
                            .map(drop),
                    ),
                ));
                outcomes.push((
                    "hub_and_authority_scores",
                    kind(karate.hub_and_authority_scores(None).map(drop)),
                ));
                outcomes.push((
                    "centralization_eigenvector_centrality",
                    kind(
                        karate
                            .centralization_eigenvector_centrality(NeighborMode::All, true)
                            .map(drop),
                    ),
                ));
                outcomes.push((
                    "pagerank (ARPACK)",
                    kind(karate.pagerank(None, .., &arpack).map(drop)),
                ));
                outcomes.push((
                    "community_leading_eigenvector",
                    kind(
                        karate
                            .community_leading_eigenvector(None, None, None)
                            .map(drop),
                    ),
                ));
                // PRPACK does not use ARPACK: allowed.
                outcomes.push((
                    "pagerank (PRPACK)",
                    kind(karate.pagerank(None, .., &prpack).map(drop)),
                ));
            }
            matvec(x, y)
        },
        &options,
        None,
    );
    // The outer computation is unaffected by the refused nested calls.
    let res = res.unwrap();
    assert!((res.values[0] - 2.0).abs() < 1e-8, "{:?}", res.values);
    assert!(checked);
    for (name, outcome) in &outcomes {
        if name.contains("PRPACK") {
            assert_eq!(*outcome, Ok(()), "{name}");
        } else {
            assert_eq!(*outcome, Err(ErrorKind::Failure), "{name} was not refused");
        }
    }
}

#[test]
fn arpack_wrappers_work_again_after_the_outer_run() {
    let karate = Graph::famous("Zachary").unwrap();
    let options = ArpackOptions::default()
        .with_nev(1)
        .with_which(ArpackWhich::LargestAlgebraic);
    arpack_rssolve(8, cycle_matvec(8), &options, None).unwrap();
    // The guard is released: sequential ARPACK runs are fine.
    let scores = karate
        .eigenvector_centrality(NeighborMode::All, None)
        .unwrap();
    assert_eq!(scores.scores.len(), 34);
    let lev = karate
        .community_leading_eigenvector(None, None, None)
        .unwrap();
    assert_eq!(lev.membership.len(), 34);
}

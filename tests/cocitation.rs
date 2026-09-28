//! Integration tests for the `cocitation` module (`igraph_cocitation.h`).

mod common;

use common::*;
use igraph::prelude::*;

/// The directed graph of igraph's `igraph_cocitation.c` / `igraph_similarity.c`
/// examples and unit test: 0 -> 1, 2 -> 1, 2 -> 0, 3 -> 0.
fn citations() -> Graph {
    Graph::from_edges(&[(0, 1), (2, 1), (2, 0), (3, 0)], 4, true).unwrap()
}

fn assert_matrix(m: &Matrix, expected: &[&[f64]], eps: f64) {
    assert_eq!(
        m.shape(),
        (expected.len(), expected.first().map_or(0, |r| r.len()))
    );
    for (i, row) in expected.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            assert!(
                (m[(i, j)] - x).abs() <= eps,
                "entry ({i}, {j}): got {}, expected {x}",
                m[(i, j)]
            );
        }
    }
}

fn assert_vec(v: &[f64], expected: &[f64], eps: f64) {
    assert_eq!(v.len(), expected.len());
    for (&a, &b) in v.iter().zip(expected) {
        assert_close(a, b, eps);
    }
}

/// All ordered pairs (i, j) with i ascending and j descending, as in the C tests.
fn c_test_pairs(n: i64) -> Vec<(i64, i64)> {
    (0..n)
        .flat_map(|i| (0..n).rev().map(move |j| (i, j)))
        .collect()
}

// ---------------------------------------------------------------------------
// Cocitation and bibliographic coupling
// ---------------------------------------------------------------------------

#[test]
fn cocitation_and_bibcoupling_match_the_c_example() {
    // examples/simple/igraph_cocitation.out
    let g = citations();
    let bib = g.bibcoupling(..).unwrap();
    assert_matrix(
        &bib,
        &[
            &[0.0, 0.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 1.0],
            &[0.0, 0.0, 1.0, 0.0],
        ],
        0.0,
    );
    let cocit = g.cocitation(..).unwrap();
    assert_matrix(
        &cocit,
        &[
            &[0.0, 1.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0],
        ],
        0.0,
    );
}

#[test]
fn cocitation_rows_follow_the_selector_order() {
    let g = citations();
    let all = g.cocitation(..).unwrap();
    let some = g.cocitation(&[3, 1, 0]).unwrap();
    assert_eq!(some.shape(), (3, 4));
    assert_eq!(some.row(0), all.row(3));
    assert_eq!(some.row(1), all.row(1));
    assert_eq!(some.row(2), all.row(0));
    // Single vertex and empty selections.
    assert_eq!(g.bibcoupling(2).unwrap().row(0), vec![1.0, 0.0, 0.0, 1.0]);
    assert_eq!(g.cocitation(VertexSelector::None).unwrap().shape(), (0, 4));
}

#[test]
fn repeated_vertices_get_every_row_filled() {
    // Regression: igraph fills only the last row of a repeated vertex.
    let g = citations();
    let all = g.bibcoupling(..).unwrap();
    let m = g.bibcoupling(vec![2, 0, 2, 2]).unwrap();
    assert_eq!(m.shape(), (4, 4));
    for (r, v) in [2usize, 0, 2, 2].into_iter().enumerate() {
        assert_eq!(m.row(r), all.row(v), "row {r}");
    }
    let il = g
        .similarity_inverse_log_weighted(vec![1, 1], NeighborMode::All)
        .unwrap();
    assert_eq!(il.row(0), il.row(1));
    assert!(il.row(0).iter().any(|&x| x > 0.0));
    let c = g.cocitation(vec![0, 0]).unwrap();
    assert_eq!(c.row(0), vec![0.0, 1.0, 0.0, 0.0]);
    assert_eq!(c.row(1), vec![0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn cocitation_is_bibcoupling_of_the_reversed_graph() {
    let edges = [
        (0, 3),
        (1, 3),
        (2, 3),
        (0, 4),
        (1, 4),
        (4, 5),
        (3, 5),
        (2, 5),
        (5, 0),
    ];
    let g = Graph::from_edges(&edges, 6, true).unwrap();
    let reversed: Vec<(i64, i64)> = edges.iter().map(|&(a, b)| (b, a)).collect();
    let r = Graph::from_edges(&reversed, 6, true).unwrap();
    assert_eq!(g.cocitation(..).unwrap(), r.bibcoupling(..).unwrap());
    assert_eq!(g.bibcoupling(..).unwrap(), r.cocitation(..).unwrap());
    // Both are symmetric.
    assert!(g.cocitation(..).unwrap().is_symmetric());
    assert!(g.bibcoupling(..).unwrap().is_symmetric());
}

#[test]
fn undirected_cocitation_counts_common_neighbors() {
    // In a simple undirected graph, cocitation = bibcoupling = A², off the diagonal.
    let g = karate();
    let c = g.cocitation(..).unwrap();
    assert_eq!(c, g.bibcoupling(..).unwrap());
    let n = g.vcount() as i64;
    for u in 0..n {
        let nu = g.neighbors(u, NeighborMode::All).unwrap();
        for v in 0..n {
            let expected = if u == v {
                0
            } else {
                let nv = g.neighbors(v, NeighborMode::All).unwrap();
                nu.iter().filter(|x| nv.contains(x)).count()
            };
            assert_eq!(c[(u as usize, v as usize)], expected as f64, "({u}, {v})");
        }
    }
    // The two leaders share 4 friends (8, 13, 19, 31).
    assert_eq!(c[(0, 33)], 4.0);
}

#[test]
fn multi_edges_count_with_multiplicity() {
    // 2 cites 0 twice and 1 once.
    let g = Graph::from_edges(&[(2, 0), (2, 0), (2, 1)], 3, true).unwrap();
    let c = g.cocitation(..).unwrap();
    assert_eq!(c[(0, 1)], 2.0);
    assert_eq!(c[(1, 0)], 2.0);
    assert_eq!(c[(0, 0)], 2.0); // the parallel edges form a pair
    assert_eq!(c[(1, 1)], 0.0);
}

#[test]
fn undirected_loops_act_like_double_edges() {
    // An undirected loop lists 0 twice among its own neighbors: 0 "cites"
    // itself twice, 1 and 2 once. A directed loop counts once.
    let g = Graph::from_edges(&[(0, 0), (0, 1), (0, 2)], 3, false).unwrap();
    let c = g.cocitation(..).unwrap();
    assert_eq!(
        c.to_rows(),
        vec![
            vec![2.0, 2.0, 2.0],
            vec![2.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0]
        ]
    );
    // Vertex 0 has degree 4 (the loop counts twice): weight 1 / ln 4.
    let il = g
        .similarity_inverse_log_weighted(.., NeighborMode::All)
        .unwrap();
    assert_close(il[(1, 2)], 1.0 / 4f64.ln(), 1e-12);
    assert_close(il[(0, 0)], 2.0 / 4f64.ln(), 1e-12);
    let d = Graph::from_edges(&[(0, 0), (0, 1), (2, 1)], 3, true).unwrap();
    let dc = d.cocitation(..).unwrap();
    assert_eq!(dc.diagonal(), vec![0.0; 3]);
    assert_eq!(dc[(0, 1)], 1.0);
}

// ---------------------------------------------------------------------------
// Jaccard and Dice (tests/unit/igraph_similarity.out, examples/simple/igraph_similarity.out)
// ---------------------------------------------------------------------------

const JACCARD_ALL_LOOPS: [&[f64]; 4] = [
    &[1.0, 0.75, 0.75, 0.5],
    &[0.75, 1.0, 1.0, 0.25],
    &[0.75, 1.0, 1.0, 0.25],
    &[0.5, 0.25, 0.25, 1.0],
];
const JACCARD_OUT_LOOPS: [&[f64]; 4] = [
    &[1.0, 0.5, 2.0 / 3.0, 1.0 / 3.0],
    &[0.5, 1.0, 1.0 / 3.0, 0.0],
    &[2.0 / 3.0, 1.0 / 3.0, 1.0, 0.25],
    &[1.0 / 3.0, 0.0, 0.25, 1.0],
];
const JACCARD_IN_NOLOOPS: [&[f64]; 4] = [
    &[1.0, 1.0 / 3.0, 0.0, 0.0],
    &[1.0 / 3.0, 1.0, 0.0, 0.0],
    &[0.0, 0.0, 1.0, 0.0],
    &[0.0, 0.0, 0.0, 1.0],
];

fn dice_of(rows: &[&[f64]]) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|r| r.iter().map(|&j| 2.0 * j / (1.0 + j)).collect())
        .collect()
}

#[test]
fn jaccard_matrices_match_the_c_unit_test() {
    let g = citations();
    let eps = 1e-6;
    let cases: [(NeighborMode, bool, &[&[f64]]); 3] = [
        (NeighborMode::All, true, &JACCARD_ALL_LOOPS),
        (NeighborMode::Out, true, &JACCARD_OUT_LOOPS),
        (NeighborMode::In, false, &JACCARD_IN_NOLOOPS),
    ];
    for (mode, loops, expected) in cases {
        let m = g.similarity_jaccard(.., .., mode, loops).unwrap();
        assert_matrix(&m, expected, eps);
        // Pairs and edges agree with the matrix (the C test's consistency checks).
        let pairs = c_test_pairs(4);
        let p = g.similarity_jaccard_pairs(&pairs, mode, loops).unwrap();
        for (k, &(i, j)) in pairs.iter().enumerate() {
            assert_close(p[k], m[(i as usize, j as usize)], eps);
        }
        let order = EdgeSelector::AllOrdered(EdgeOrder::From);
        let eids = g.select_edges(order.clone()).unwrap();
        let es = g.similarity_jaccard_es(order, mode, loops).unwrap();
        for (k, &e) in eids.iter().enumerate() {
            let (a, b) = g.edge(e).unwrap();
            assert_close(es[k], m[(a as usize, b as usize)], eps);
        }
    }
    // Sub-matrix without loops: vertices 1 and 2.
    let sub = g
        .similarity_jaccard(1..3, 1..3, NeighborMode::All, false)
        .unwrap();
    assert_matrix(&sub, &[&[1.0, 1.0 / 3.0], &[1.0 / 3.0, 1.0]], 1e-12);
}

#[test]
fn dice_matrices_match_the_c_unit_test() {
    let g = citations();
    let cases: [(NeighborMode, bool, &[&[f64]]); 3] = [
        (NeighborMode::All, true, &JACCARD_ALL_LOOPS),
        (NeighborMode::Out, true, &JACCARD_OUT_LOOPS),
        (NeighborMode::In, false, &JACCARD_IN_NOLOOPS),
    ];
    for (mode, loops, jac) in cases {
        let expected = dice_of(jac);
        let rows: Vec<&[f64]> = expected.iter().map(|r| r.as_slice()).collect();
        let m = g.similarity_dice(.., .., mode, loops).unwrap();
        assert_matrix(&m, &rows, 1e-6);
        let pairs = c_test_pairs(4);
        let p = g.similarity_dice_pairs(&pairs, mode, loops).unwrap();
        for (k, &(i, j)) in pairs.iter().enumerate() {
            assert_close(p[k], m[(i as usize, j as usize)], 1e-12);
        }
    }
    // Spot values printed by the unit test.
    let m = g.similarity_dice(.., .., NeighborMode::All, true).unwrap();
    assert_close(m[(0, 1)], 0.857143, 1e-6);
    assert_close(m[(1, 3)], 0.4, 1e-12);
}

#[test]
fn pairs_and_edges_match_the_c_example() {
    // examples/simple/igraph_similarity.out
    let g = citations();
    let pairs = c_test_pairs(4);
    let jac = g
        .similarity_jaccard_pairs(&pairs, NeighborMode::All, false)
        .unwrap();
    let third = 1.0 / 3.0;
    assert_vec(
        &jac,
        &[
            0.0, 0.25, 0.25, 1.0, 0.5, third, 1.0, 0.25, 0.5, 1.0, third, 0.25, 1.0, 0.5, 0.5, 0.0,
        ],
        1e-12,
    );
    let dice = g
        .similarity_dice_pairs(&pairs, NeighborMode::All, false)
        .unwrap();
    assert_vec(
        &dice,
        &[
            0.0,
            0.4,
            0.4,
            1.0,
            2.0 / 3.0,
            0.5,
            1.0,
            0.4,
            2.0 / 3.0,
            1.0,
            0.5,
            0.4,
            1.0,
            2.0 / 3.0,
            2.0 / 3.0,
            0.0,
        ],
        1e-12,
    );
    let order = EdgeSelector::AllOrdered(EdgeOrder::From);
    let jes = g
        .similarity_jaccard_es(order.clone(), NeighborMode::In, false)
        .unwrap();
    assert_vec(&jes, &[third, 0.0, 0.0, 0.0], 1e-12);
    let des = g
        .similarity_dice_es(order, NeighborMode::In, false)
        .unwrap();
    assert_vec(&des, &[0.5, 0.0, 0.0, 0.0], 1e-12);
    let dsub = g
        .similarity_dice(1..3, 1..3, NeighborMode::All, false)
        .unwrap();
    assert_matrix(&dsub, &[&[1.0, 0.5], &[0.5, 1.0]], 1e-12);
}

#[test]
fn rectangular_similarity_matrices_are_correct() {
    // Regression: the C matrix functions assume from == to; with more rows
    // than columns they write past the end of the matrix.
    let g = citations();
    let col = g
        .similarity_jaccard(.., 1, NeighborMode::All, false)
        .unwrap();
    assert_matrix(&col, &[&[0.25], &[1.0], &[1.0 / 3.0], &[0.5]], 1e-12);
    // igraph's own unit test calls from = {1, 2}, to = {1}.
    let small = g
        .similarity_jaccard(1..3, 1, NeighborMode::All, false)
        .unwrap();
    assert_matrix(&small, &[&[1.0], &[1.0 / 3.0]], 1e-12);

    let k = karate();
    let full = k
        .similarity_jaccard(.., .., NeighborMode::All, true)
        .unwrap();
    let from = [33, 5, 0, 16, 2];
    let to = [0, 1, 2];
    for (m, dice) in [
        (
            k.similarity_jaccard(&from, &to, NeighborMode::All, true)
                .unwrap(),
            false,
        ),
        (
            k.similarity_dice(&from, &to, NeighborMode::All, true)
                .unwrap(),
            true,
        ),
    ] {
        assert_eq!(m.shape(), (5, 3));
        for (i, &u) in from.iter().enumerate() {
            for (j, &v) in to.iter().enumerate() {
                let x = full[(u as usize, v as usize)];
                let expected = if dice { 2.0 * x / (1.0 + x) } else { x };
                assert_close(m[(i, j)], expected, 1e-12);
            }
        }
    }
    // Wide matrix, repeated vertices, and loops = true for vertices only in `to`.
    let wide = k
        .similarity_jaccard(&[1, 1], .., NeighborMode::All, true)
        .unwrap();
    assert_eq!(wide.shape(), (2, 34));
    assert_eq!(wide.row(0), full.row(1));
    assert_eq!(wide.row(1), full.row(1));
    // Empty selections.
    let e = k
        .similarity_dice(VertexSelector::None, .., NeighborMode::All, true)
        .unwrap();
    assert_eq!(e.shape(), (0, 34));
}

#[test]
fn dice_is_a_monotone_function_of_jaccard() {
    let g = karate();
    for loops in [false, true] {
        let j = g
            .similarity_jaccard(.., .., NeighborMode::All, loops)
            .unwrap();
        let d = g.similarity_dice(.., .., NeighborMode::All, loops).unwrap();
        assert!(j.is_symmetric() && d.is_symmetric());
        for ((idx, &x), &y) in j.indexed_iter().zip(d.as_slice()) {
            assert_close(y, 2.0 * x / (1.0 + x), 1e-12);
            assert!((0.0..=1.0).contains(&x), "{idx:?}");
            assert!(y >= x);
        }
        assert_eq!(j.diagonal(), vec![1.0; 34]);
    }
}

#[test]
fn isolated_vertices_and_self_pairs() {
    let g = Graph::from_edges(&[(0, 1)], 3, false).unwrap();
    let m = g
        .similarity_jaccard(.., .., NeighborMode::All, false)
        .unwrap();
    // Vertex 2 is isolated: 0 to everyone else, 1 to itself.
    assert_eq!(m.row(2), vec![0.0, 0.0, 1.0]);
    // With closed neighborhoods, 0 and 1 are identical.
    let s = g
        .similarity_jaccard_pairs(&[(0, 1), (2, 2), (0, 2)], NeighborMode::All, true)
        .unwrap();
    assert_eq!(s, vec![1.0, 1.0, 0.0]);
    assert!(
        g.similarity_jaccard_pairs(&[], NeighborMode::All, true)
            .unwrap()
            .is_empty()
    );
    assert!(
        g.similarity_dice_es(EdgeSelector::None, NeighborMode::All, true)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn mutual_edges_count_once_even_with_a_primed_cache() {
    // Regression: after has_multiple() caches "no multi-edges", igraph 1.0.1's
    // adjacency lists stop deduplicating mutual edges in NeighborMode::All.
    let edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 0)];
    let fresh = Graph::from_edges(&edges, 4, true).unwrap();
    let expected_m = fresh
        .similarity_jaccard(.., .., NeighborMode::All, false)
        .unwrap();
    let expected_es = fresh
        .similarity_dice_es(.., NeighborMode::All, true)
        .unwrap();
    let pairs = c_test_pairs(4);
    let expected_p = fresh
        .similarity_jaccard_pairs(&pairs, NeighborMode::All, false)
        .unwrap();
    // N(1) = {0, 2}, N(3) = {0, 2} ignoring directions.
    assert_eq!(expected_m[(1, 3)], 1.0);

    let primed = Graph::from_edges(&edges, 4, true).unwrap();
    assert!(!primed.has_multiple().unwrap());
    assert_eq!(
        primed
            .similarity_jaccard(.., .., NeighborMode::All, false)
            .unwrap(),
        expected_m
    );
    assert!(!primed.has_multiple().unwrap());
    assert_eq!(
        primed
            .similarity_dice_es(.., NeighborMode::All, true)
            .unwrap(),
        expected_es
    );
    assert!(!primed.has_multiple().unwrap());
    assert_eq!(
        primed
            .similarity_jaccard_pairs(&pairs, NeighborMode::All, false)
            .unwrap(),
        expected_p
    );
    // The cache is still right afterwards.
    assert!(!primed.has_multiple().unwrap());
}

// ---------------------------------------------------------------------------
// Inverse log-weighted similarity
// ---------------------------------------------------------------------------

#[test]
fn inverse_log_weighted_matches_the_c_unit_test() {
    let g = citations();
    let a = 1.0 / 2f64.ln(); // 1.4427
    let b = 1.0 / 3f64.ln(); // 0.910239
    let all = g
        .similarity_inverse_log_weighted(.., NeighborMode::All)
        .unwrap();
    assert_matrix(
        &all,
        &[
            &[0.0, a, a, 0.0],
            &[a, 0.0, b, b],
            &[a, b, 0.0, b],
            &[0.0, b, b, 0.0],
        ],
        1e-12,
    );
    let out = g
        .similarity_inverse_log_weighted(.., NeighborMode::Out)
        .unwrap();
    assert_matrix(
        &out,
        &[
            &[0.0, 0.0, a, 0.0],
            &[0.0, 0.0, 0.0, 0.0],
            &[a, 0.0, 0.0, a],
            &[0.0, 0.0, a, 0.0],
        ],
        1e-12,
    );
    let inn = g
        .similarity_inverse_log_weighted(.., NeighborMode::In)
        .unwrap();
    assert_matrix(
        &inn,
        &[
            &[0.0, a, 0.0, 0.0],
            &[a, 0.0, 0.0, 0.0],
            &[0.0; 4],
            &[0.0; 4],
        ],
        1e-12,
    );
}

#[test]
fn inverse_log_weighted_is_a_weighted_common_neighbor_count() {
    // On a simple undirected graph: sum over common neighbors w of 1 / ln(deg w).
    let g = karate();
    let m = g
        .similarity_inverse_log_weighted(.., NeighborMode::All)
        .unwrap();
    let deg = g.degree(.., NeighborMode::All, Loops::Twice).unwrap();
    for (u, v) in [(0, 33), (2, 33), (4, 5), (16, 0), (8, 30)] {
        let nu = g.neighbors(u, NeighborMode::All).unwrap();
        let nv = g.neighbors(v, NeighborMode::All).unwrap();
        let expected: f64 = nu
            .iter()
            .filter(|x| nv.contains(x))
            .map(|&w| 1.0 / (deg[w as usize] as f64).ln())
            .sum();
        assert_close(m[(u as usize, v as usize)], expected, 1e-12);
    }
    assert_eq!(m.diagonal(), vec![0.0; 34]);
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[test]
fn invalid_ids_are_errors() {
    let g = citations();
    let kind = |r: Result<Matrix>| r.unwrap_err().kind();
    assert_eq!(kind(g.cocitation(&[0, 4])), ErrorKind::InvalidVertexId);
    assert_eq!(kind(g.bibcoupling(-1)), ErrorKind::InvalidVertexId);
    assert_eq!(
        kind(g.similarity_inverse_log_weighted(&[7], NeighborMode::All)),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        kind(g.similarity_jaccard(&[0, 9], .., NeighborMode::All, true)),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        kind(g.similarity_dice(.., &[4], NeighborMode::All, true)),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        g.similarity_jaccard_pairs(&[(0, 1), (2, 4)], NeighborMode::All, false)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        g.similarity_dice_pairs(&[(-1, 0)], NeighborMode::All, false)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidVertexId
    );
    assert!(
        g.similarity_jaccard_es(&[4], NeighborMode::All, false)
            .is_err()
    );
    assert!(
        g.similarity_dice_es(&[0, 10], NeighborMode::All, false)
            .is_err()
    );
}

// ---------------------------------------------------------------------------
// Use case: link prediction in Zachary's karate club
// ---------------------------------------------------------------------------

#[test]
fn use_case_link_prediction_in_the_karate_club() {
    // Rank the non-adjacent pairs of members by Jaccard similarity: the best
    // candidates for a future friendship.
    let g = karate();
    let n = g.vcount() as i64;
    let jac = g
        .similarity_jaccard(.., .., NeighborMode::All, false)
        .unwrap();
    let mut candidates = Vec::new();
    for u in 0..n {
        for v in u + 1..n {
            if !g.are_adjacent(u, v).unwrap() {
                candidates.push((jac[(u as usize, v as usize)], u, v));
            }
        }
    }
    // 34 * 33 / 2 - 78 non-edges.
    assert_eq!(candidates.len(), 561 - 78);
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2))));

    // The perfect scores are the structurally equivalent members: 14, 15, 18,
    // 20 and 22 are friends of exactly {32, 33}; 17 and 21 of exactly {0, 1}.
    let perfect: Vec<(i64, i64)> = candidates
        .iter()
        .take_while(|c| c.0 == 1.0)
        .map(|c| (c.1, c.2))
        .collect();
    assert_eq!(perfect.len(), 10 + 1);
    assert!(perfect.contains(&(17, 21)));
    for &(u, v) in &perfect {
        let mut nu = g.neighbors(u, NeighborMode::All).unwrap();
        let mut nv = g.neighbors(v, NeighborMode::All).unwrap();
        nu.sort_unstable();
        nv.sort_unstable();
        assert_eq!(nu, nv);
    }
    // The pairs agree with the pair and edge variants.
    let top: Vec<(i64, i64)> = candidates[..20].iter().map(|c| (c.1, c.2)).collect();
    let again = g
        .similarity_jaccard_pairs(&top, NeighborMode::All, false)
        .unwrap();
    assert_eq!(
        again,
        candidates[..20].iter().map(|c| c.0).collect::<Vec<_>>()
    );
    // Existing friendships never score 1 (the endpoints are not in their own sets).
    let on_edges = g
        .similarity_jaccard_es(.., NeighborMode::All, false)
        .unwrap();
    assert_eq!(on_edges.len(), 78);
    assert!(on_edges.iter().all(|&x| x < 1.0));

    // Adamic–Adar weighs rare common friends more: its top prediction is
    // member 2 joining the instructor's (33) side, with whom 2 shares six friends.
    let aa = g
        .similarity_inverse_log_weighted(.., NeighborMode::All)
        .unwrap();
    let best = candidates
        .iter()
        .map(|&(_, u, v)| (aa[(u as usize, v as usize)], u, v))
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap();
    assert_eq!((best.1, best.2), (2, 33));
    assert_close(best.0, 4.719381261461351, 1e-9);
    assert_eq!(g.cocitation(2).unwrap()[(0, 33)], 6.0);
}

//! Integration tests for `igraph::conversion` (igraph_conversion.h).

mod common;

use common::{assert_close, complete, cycle, karate, path};
use igraph::centrality::PageRankOptions;
use igraph::conversion::CooMatrix;
use igraph::linalg::SparseMat;
use igraph::prelude::*;
use igraph::structural::LaplacianNormalization;

/// The graph of igraph's `tests/unit/igraph_get_adjacency.c`:
/// 0-1, 1-2, 2-3, 3-4, 4-0, 0-3, 2-2 (loop), 0-1 (multi-edge).
fn unit_test_graph(n: usize, directed: bool) -> Graph {
    Graph::from_edges(
        &[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 0),
            (0, 3),
            (2, 2),
            (0, 1),
        ],
        n,
        directed,
    )
    .unwrap()
}

const UNIT_WEIGHTS: [f64; 8] = [5.0, 4.0, 3.0, 2.0, 1.0, 6.0, 3.0, 2.0];

fn rows(m: &Matrix) -> Vec<Vec<f64>> {
    m.to_rows()
}

fn m(r: &[&[f64]]) -> Vec<Vec<f64>> {
    r.iter().map(|x| x.to_vec()).collect()
}

// ---------------------------------------------------------------------------
// get_adjacency: values pinned by tests/unit/igraph_get_adjacency.out
// ---------------------------------------------------------------------------

#[test]
fn adjacency_undirected_unweighted_matches_igraph_unit_test() {
    let g = unit_test_graph(5, false);
    let upper_no = g
        .get_adjacency(GetAdjacency::Upper, None, Loops::None)
        .unwrap();
    assert_eq!(
        rows(&upper_no),
        m(&[
            &[0., 2., 0., 1., 1.],
            &[0., 0., 1., 0., 0.],
            &[0., 0., 0., 1., 0.],
            &[0., 0., 0., 0., 1.],
            &[0., 0., 0., 0., 0.],
        ])
    );
    let lower_twice = g
        .get_adjacency(GetAdjacency::Lower, None, Loops::Twice)
        .unwrap();
    assert_eq!(
        rows(&lower_twice),
        m(&[
            &[0., 0., 0., 0., 0.],
            &[2., 0., 0., 0., 0.],
            &[0., 1., 2., 0., 0.],
            &[1., 0., 1., 0., 0.],
            &[1., 0., 0., 1., 0.],
        ])
    );
    let both_once = g
        .get_adjacency(GetAdjacency::Both, None, Loops::Once)
        .unwrap();
    assert_eq!(
        rows(&both_once),
        m(&[
            &[0., 2., 0., 1., 1.],
            &[2., 0., 1., 0., 0.],
            &[0., 1., 1., 1., 0.],
            &[1., 0., 1., 0., 1.],
            &[1., 0., 0., 1., 0.],
        ])
    );
    // Upper + Lower - diagonal == Both.
    let upper = g
        .get_adjacency(GetAdjacency::Upper, None, Loops::Twice)
        .unwrap();
    let lower = g
        .get_adjacency(GetAdjacency::Lower, None, Loops::Twice)
        .unwrap();
    let both = g
        .get_adjacency(GetAdjacency::Both, None, Loops::Twice)
        .unwrap();
    for i in 0..5 {
        for j in 0..5 {
            let diag = if i == j { upper[(i, i)] } else { 0.0 };
            assert_eq!(upper[(i, j)] + lower[(i, j)] - diag, both[(i, j)]);
        }
    }
}

#[test]
fn adjacency_weighted_matches_igraph_unit_test() {
    let g = unit_test_graph(5, false);
    let a = g
        .get_adjacency(GetAdjacency::Both, Some(&UNIT_WEIGHTS), Loops::Twice)
        .unwrap();
    assert_eq!(
        rows(&a),
        m(&[
            &[0., 7., 0., 6., 1.],
            &[7., 0., 4., 0., 0.],
            &[0., 4., 6., 3., 0.],
            &[6., 0., 3., 0., 2.],
            &[1., 0., 0., 2., 0.],
        ])
    );
    let d = unit_test_graph(5, true);
    let a = d
        .get_adjacency(GetAdjacency::Both, Some(&UNIT_WEIGHTS), Loops::Twice)
        .unwrap();
    // Directed: "loops twice" is the same as once.
    assert_eq!(
        rows(&a),
        m(&[
            &[0., 7., 0., 6., 0.],
            &[0., 0., 4., 0., 0.],
            &[0., 0., 3., 3., 0.],
            &[0., 0., 0., 0., 2.],
            &[1., 0., 0., 0., 0.],
        ])
    );
}

#[test]
fn adjacency_kind_is_ignored_for_directed_graphs() {
    let d = unit_test_graph(5, true);
    let reference = d
        .get_adjacency(GetAdjacency::Both, None, Loops::Once)
        .unwrap();
    for kind in [GetAdjacency::Upper, GetAdjacency::Lower] {
        assert_eq!(d.get_adjacency(kind, None, Loops::Once).unwrap(), reference);
    }
}

#[test]
fn handshake_lemma_from_adjacency() {
    // Row sums of the "loops twice" adjacency matrix are the degrees and add
    // up to 2|E|.
    let g = Graph::famous("Zachary").unwrap();
    assert!(g.is_same_graph(&karate()).unwrap());
    let a = g
        .get_adjacency(GetAdjacency::Both, None, Loops::Twice)
        .unwrap();
    let total: f64 = a.as_slice().iter().sum();
    assert_eq!(total, 2.0 * g.ecount() as f64);
    let deg = g
        .degree(VertexSelector::All, NeighborMode::All, Loops::Twice)
        .unwrap();
    for (i, r) in a.rows().enumerate() {
        assert_eq!(r.iter().sum::<f64>(), deg[i] as f64);
    }
    // Vertex 33 (the administrator) has degree 17, vertex 0 (the instructor) 16.
    assert_eq!(deg[33], 17);
    assert_eq!(deg[0], 16);
}

#[test]
fn triangles_from_trace_of_cube() {
    // trace(A³) = 6 · #triangles; K5 has C(5,3) = 10 triangles.
    let g = complete(5);
    let a = g
        .get_adjacency(GetAdjacency::Both, None, Loops::None)
        .unwrap();
    let n = a.nrow();
    let mul = |x: &Matrix, y: &Matrix| {
        let mut r = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                r[(i, j)] = (0..n).map(|k| x[(i, k)] * y[(k, j)]).sum();
            }
        }
        r
    };
    let a3 = mul(&mul(&a, &a), &a);
    let trace: f64 = (0..n).map(|i| a3[(i, i)]).sum();
    assert_eq!(trace / 6.0, 10.0);
}

#[test]
fn adjacency_of_empty_and_null_graphs() {
    let null = Graph::empty(0, false).unwrap();
    let a = null
        .get_adjacency(GetAdjacency::Both, None, Loops::Twice)
        .unwrap();
    assert_eq!(a.shape(), (0, 0));
    let empty = Graph::empty(3, true).unwrap();
    let a = empty
        .get_adjacency(GetAdjacency::Both, None, Loops::Twice)
        .unwrap();
    assert_eq!(a, Matrix::zeros(3, 3));
    let s = empty
        .get_adjacency_sparse(GetAdjacency::Both, None, Loops::Twice)
        .unwrap();
    assert_eq!((s.nrow, s.ncol, s.nnz()), (3, 3, 0));
}

#[test]
fn wrong_weight_length_is_rejected() {
    let g = cycle(4);
    let short = [1.0, 2.0];
    let err = g
        .get_adjacency(GetAdjacency::Both, Some(&short), Loops::None)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    assert!(
        g.get_adjacency_sparse(GetAdjacency::Both, Some(&short), Loops::None)
            .is_err()
    );
    assert!(g.get_stochastic(false, Some(&short)).is_err());
    assert!(g.get_stochastic_sparse(true, Some(&short)).is_err());
}

// ---------------------------------------------------------------------------
// get_adjacency_sparse: consistent with the dense version
// ---------------------------------------------------------------------------

#[test]
fn sparse_adjacency_agrees_with_dense_for_every_combination() {
    for directed in [false, true] {
        let g = unit_test_graph(6, directed);
        for kind in [GetAdjacency::Upper, GetAdjacency::Lower, GetAdjacency::Both] {
            for loops in [Loops::None, Loops::Once, Loops::Twice] {
                for w in [None, Some(&UNIT_WEIGHTS[..])] {
                    let dense = g.get_adjacency(kind, w, loops).unwrap();
                    let sparse = g.get_adjacency_sparse(kind, w, loops).unwrap();
                    assert_eq!(
                        sparse.to_dense(),
                        dense,
                        "{directed} {kind:?} {loops:?} {w:?}"
                    );
                    assert_eq!((sparse.nrow, sparse.ncol), (6, 6));
                    // Entries are sorted, unique, and in range.
                    assert!(
                        sparse
                            .entries
                            .windows(2)
                            .all(|p| (p[0].0, p[0].1) < (p[1].0, p[1].1))
                    );
                    for &(i, j, x) in &sparse.entries {
                        assert_eq!(sparse.get(i, j), x);
                    }
                }
            }
        }
    }
}

#[test]
fn sparse_adjacency_sums_multi_edges() {
    // Three parallel edges 0-1 collapse into one stored entry of value 3.
    let g = Graph::from_edges(&[(0, 1), (1, 0), (0, 1)], 2, false).unwrap();
    let s = g
        .get_adjacency_sparse(GetAdjacency::Both, None, Loops::Twice)
        .unwrap();
    assert_eq!(s.entries, vec![(0, 1, 3.0), (1, 0, 3.0)]);
    assert_eq!(Matrix::from(&s), s.to_dense());
}

// ---------------------------------------------------------------------------
// get_stochastic: values pinned by tests/unit/igraph_get_stochastic.out
// ---------------------------------------------------------------------------

fn assert_matrix_close(a: &Matrix, expected: &[&[f64]], eps: f64) {
    assert_eq!(a.nrow(), expected.len());
    for (i, row) in expected.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            assert_close(a[(i, j)], x, eps);
        }
    }
}

#[test]
fn stochastic_matches_igraph_unit_test() {
    let g = unit_test_graph(6, false);
    let p = g.get_stochastic(false, None).unwrap();
    assert_matrix_close(
        &p,
        &[
            &[0., 0.5, 0., 0.25, 0.25, 0.],
            &[2. / 3., 0., 1. / 3., 0., 0., 0.],
            &[0., 0.25, 0.5, 0.25, 0., 0.],
            &[1. / 3., 0., 1. / 3., 0., 1. / 3., 0.],
            &[0.5, 0., 0., 0.5, 0., 0.],
            &[0., 0., 0., 0., 0., 0.],
        ],
        1e-12,
    );
    let pw = g.get_stochastic(false, Some(&UNIT_WEIGHTS)).unwrap();
    assert_close(pw[(0, 1)], 0.5, 1e-12);
    assert_close(pw[(0, 3)], 0.428571, 1e-6);
    assert_close(pw[(2, 2)], 0.461538, 1e-6);

    let d = unit_test_graph(6, true);
    let pd = d.get_stochastic(true, Some(&UNIT_WEIGHTS)).unwrap();
    assert_close(pd[(0, 3)], 0.666667, 1e-6);
    assert_close(pd[(1, 2)], 0.571429, 1e-6);
    assert_close(pd[(2, 2)], 0.428571, 1e-6);
}

#[test]
fn stochastic_is_adjacency_divided_by_strength() {
    // P = D⁻¹ A row-wise (out-strengths), A D⁻¹ column-wise (in-strengths).
    for directed in [false, true] {
        let g = unit_test_graph(6, directed);
        for w in [None, Some(&UNIT_WEIGHTS[..])] {
            let a = g
                .get_adjacency(GetAdjacency::Both, w, Loops::Twice)
                .unwrap();
            let (out_mode, in_mode) = if directed {
                (NeighborMode::Out, NeighborMode::In)
            } else {
                (NeighborMode::All, NeighborMode::All)
            };
            let out_s = g.strength(.., out_mode, Loops::Twice, w).unwrap();
            let in_s = g.strength(.., in_mode, Loops::Twice, w).unwrap();
            let rw = g.get_stochastic(false, w).unwrap();
            let cw = g.get_stochastic(true, w).unwrap();
            for i in 0..6 {
                for j in 0..6 {
                    let r = if out_s[i] == 0.0 {
                        0.0
                    } else {
                        a[(i, j)] / out_s[i]
                    };
                    let c = if in_s[j] == 0.0 {
                        0.0
                    } else {
                        a[(i, j)] / in_s[j]
                    };
                    assert_close(rw[(i, j)], r, 1e-12);
                    assert_close(cw[(i, j)], c, 1e-12);
                }
            }
        }
    }
}

#[test]
fn laplacian_is_degree_minus_adjacency() {
    // L = D - A for the unnormalized Laplacian of a loopless graph.
    let g = karate();
    let a = g
        .get_adjacency(GetAdjacency::Both, None, Loops::Twice)
        .unwrap();
    let l = g
        .get_laplacian(
            NeighborMode::All,
            LaplacianNormalization::Unnormalized,
            None,
        )
        .unwrap();
    let deg = g.degree(.., NeighborMode::All, Loops::Twice).unwrap();
    for i in 0..34 {
        for j in 0..34 {
            let d = if i == j { deg[i] as f64 } else { 0.0 };
            assert_eq!(l[(i, j)], d - a[(i, j)]);
        }
    }
}

#[test]
fn adjacency_round_trips_through_constructors() {
    // Unweighted, with multi-edges and a loop.
    for directed in [false, true] {
        let g = unit_test_graph(6, directed);
        let a = g
            .get_adjacency(GetAdjacency::Both, None, Loops::Twice)
            .unwrap();
        let mode = if directed {
            Adjacency::Directed
        } else {
            Adjacency::Undirected
        };
        let h = Graph::adjacency(&a, mode, Loops::Twice).unwrap();
        assert_eq!(h.is_directed(), directed);
        assert_eq!(
            h.get_adjacency(GetAdjacency::Both, None, Loops::Twice)
                .unwrap(),
            a
        );
        // The sparse constructor accepts the sparse output.
        let s = g
            .get_adjacency_sparse(GetAdjacency::Both, None, Loops::Twice)
            .unwrap();
        let h = Graph::sparse_adjacency(6, &s.entries, mode, Loops::Twice).unwrap();
        assert_eq!(
            h.get_adjacency(GetAdjacency::Both, None, Loops::Twice)
                .unwrap(),
            a
        );
    }
    // Weighted, simple graph: weights survive the round trip. The upper
    // triangle is enough to rebuild an undirected graph.
    let g = karate();
    let w: Vec<f64> = (0..g.ecount()).map(|e| 1.0 + e as f64).collect();
    let upper = g
        .get_adjacency(GetAdjacency::Upper, Some(&w), Loops::None)
        .unwrap();
    let (h, hw) = Graph::weighted_adjacency(&upper, Adjacency::Upper, Loops::None).unwrap();
    assert_eq!(h.ecount(), g.ecount());
    assert_eq!(
        h.get_adjacency(GetAdjacency::Both, Some(&hw), Loops::None)
            .unwrap(),
        g.get_adjacency(GetAdjacency::Both, Some(&w), Loops::None)
            .unwrap()
    );
}

#[test]
fn stochastic_rows_or_columns_sum_to_one_except_isolated() {
    for directed in [false, true] {
        let g = unit_test_graph(6, directed);
        for w in [None, Some(&UNIT_WEIGHTS[..])] {
            let rw = g.get_stochastic(false, w).unwrap();
            for (i, r) in rw.rows().enumerate() {
                let expected = if i == 5 { 0.0 } else { 1.0 };
                assert_close(r.iter().sum(), expected, 1e-12);
            }
            let cw = g.get_stochastic(true, w).unwrap();
            for (j, c) in cw.columns().enumerate() {
                let expected = if j == 5 { 0.0 } else { 1.0 };
                assert_close(c.iter().sum(), expected, 1e-12);
            }
            // The sparse version agrees.
            let srw = g.get_stochastic_sparse(false, w).unwrap().to_dense();
            let scw = g.get_stochastic_sparse(true, w).unwrap().to_dense();
            for (x, y) in srw.as_slice().iter().zip(rw.as_slice()) {
                assert_close(*x, *y, 1e-12);
            }
            for (x, y) in scw.as_slice().iter().zip(cw.as_slice()) {
                assert_close(*x, *y, 1e-12);
            }
        }
    }
}

#[test]
fn stationary_distribution_of_random_walk_is_proportional_to_degree() {
    // For a connected undirected graph, π_i = d_i / 2|E| satisfies πP = π.
    let g = karate();
    let p = g.get_stochastic(false, None).unwrap();
    let deg = g
        .degree(VertexSelector::All, NeighborMode::All, Loops::Twice)
        .unwrap();
    let two_m = 2.0 * g.ecount() as f64;
    let pi: Vec<f64> = deg.iter().map(|&d| d as f64 / two_m).collect();
    let n = pi.len();
    for j in 0..n {
        let pj: f64 = (0..n).map(|i| pi[i] * p[(i, j)]).sum();
        assert_close(pj, pi[j], 1e-12);
    }
}

#[test]
fn stochastic_zero_strength_rows_are_zero_not_nan() {
    // 0 → 1 has weight 0: vertex 0 has an out-edge but zero out-strength.
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 2)], 3, true).unwrap();
    let w = [0.0, 2.0, 1.0];
    let p = g.get_stochastic(false, Some(&w)).unwrap();
    assert!(p.as_slice().iter().all(|x| x.is_finite()));
    assert_eq!(rows(&p), m(&[&[0., 0., 0.], &[0., 0., 1.], &[0., 0., 1.]]));
    let sp = g.get_stochastic_sparse(false, Some(&w)).unwrap();
    assert_eq!(sp.to_dense(), p);
    // Column-wise: vertex 1 has zero in-strength.
    let c = g.get_stochastic(true, Some(&w)).unwrap();
    assert_eq!(
        rows(&c),
        m(&[&[0., 0., 0.], &[0., 0., 2. / 3.], &[0., 0., 1. / 3.]])
    );
    assert_eq!(
        g.get_stochastic_sparse(true, Some(&w)).unwrap().to_dense(),
        c
    );
    // Undirected: vertex 0 only touches a zero-weight edge.
    let u = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    for column_wise in [false, true] {
        let p = u.get_stochastic(column_wise, Some(&[0.0, 1.0])).unwrap();
        assert!(p.as_slice().iter().all(|x| x.is_finite()));
        let sp = u
            .get_stochastic_sparse(column_wise, Some(&[0.0, 1.0]))
            .unwrap();
        assert_eq!(sp.to_dense(), p);
    }
}

#[test]
fn stochastic_zero_test_uses_the_same_strengths_as_igraph() {
    // Mixed-sign weights whose sum depends on the order of the additions.
    // Undirected edges are stored with FROM >= TO, so vertex 1 is the FROM
    // end of edges 0 and 2 (both 0-1) and the TO end of edge 1 (1-2).
    // Summing per edge in id order gives 1e16 + 1 - 1e16 = 0, but
    // igraph_strength (the call that igraph_get_stochastic makes) first adds
    // all FROM ends, then all TO ends: 1e16 - 1e16 + 1 = 1. So vertex 1 has
    // a finite row that must not be cleared, while vertex 0 (1e16 - 1e16 = 0
    // in any order) gets a NaN row from C that must be cleared.
    let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 1)], 3, false).unwrap();
    let w = [1e16, 1.0, -1e16];
    assert_eq!(w[0] + w[1] + w[2], 0.0);
    let s = g
        .strength(.., NeighborMode::All, Loops::Twice, Some(&w))
        .unwrap();
    assert_eq!(s, vec![0.0, 1.0, 1.0]);
    let p = g.get_stochastic(false, Some(&w)).unwrap();
    assert_eq!(p.row(0), vec![0.0, 0.0, 0.0]);
    assert_eq!(p.row(1), vec![0.0, 0.0, 1.0]);
    assert_eq!(p.row(2), vec![0.0, 1.0, 0.0]);
}

#[test]
fn stochastic_with_cancelling_weights_is_zero_in_both_versions() {
    // Vertex 0 has weights +1 and -1: zero strength although its edges do
    // not all have weight zero. Dense (C: ±1/0 = ±inf) is cleared by the
    // wrapper; igraph's sparse normalization scales the row by zero.
    let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2)], 3, false).unwrap();
    let w = [1.0, -1.0, 2.0];
    let p = g.get_stochastic(false, Some(&w)).unwrap();
    assert_eq!(p.row(0), vec![0.0, 0.0, 0.0]);
    assert_eq!(p.row(1), vec![1.0 / 3.0, 0.0, 2.0 / 3.0]);
    let sp = g.get_stochastic_sparse(false, Some(&w)).unwrap();
    assert_eq!(sp.to_dense(), p);
    for column_wise in [false, true] {
        let d = g.get_stochastic(column_wise, Some(&w)).unwrap();
        let s = g.get_stochastic_sparse(column_wise, Some(&w)).unwrap();
        assert!(d.as_slice().iter().all(|x| x.is_finite()));
        assert_eq!(s.to_dense(), d);
    }
}

#[test]
fn stochastic_with_undirected_self_loop_counts_it_twice() {
    // Vertex 0 has a loop (2 stems) and an edge to 1: degree 3.
    let g = Graph::from_edges(&[(0, 0), (0, 1)], 2, false).unwrap();
    let p = g.get_stochastic(false, None).unwrap();
    assert_matrix_close(&p, &[&[2. / 3., 1. / 3.], &[1., 0.]], 1e-15);
    let sp = g.get_stochastic_sparse(false, None).unwrap();
    assert_eq!(sp.nnz(), 3);
    assert_close(sp.get(0, 0), 2. / 3., 1e-15);
}

// ---------------------------------------------------------------------------
// CooMatrix helpers
// ---------------------------------------------------------------------------

#[test]
fn coo_matrix_algebra_matches_dense() {
    let g = unit_test_graph(6, true);
    let a = g
        .get_adjacency_sparse(GetAdjacency::Both, Some(&UNIT_WEIGHTS), Loops::Once)
        .unwrap();
    let d = a.to_dense();
    let x = [1.0, -2.0, 0.5, 3.0, 0.0, 7.0];
    let ax: Vec<f64> = (0..6)
        .map(|i| (0..6).map(|j| d[(i, j)] * x[j]).sum())
        .collect();
    let xa: Vec<f64> = (0..6)
        .map(|j| (0..6).map(|i| x[i] * d[(i, j)]).sum())
        .collect();
    assert_eq!(a.mul_vec(&x), ax);
    assert_eq!(a.vec_mul(&x), xa);
    // Transposition swaps the two products and the row/column sums.
    let t = a.transpose();
    assert_eq!(t.to_dense(), d.transposed());
    assert_eq!(t.mul_vec(&x), xa);
    assert_eq!(t.row_sums(), a.col_sums());
    assert_eq!(t.transpose(), a);
    assert_eq!(a.iter().count(), a.nnz());
    // A · 1 gives the weighted out-strengths, 1ᵀ · A the in-strengths.
    assert_eq!(a.mul_vec(&[1.0; 6]), a.row_sums());
    assert_eq!(a.vec_mul(&[1.0; 6]), a.col_sums());
}

#[test]
#[should_panic(expected = "vector length")]
fn coo_mul_vec_rejects_wrong_length() {
    let a = cycle(4)
        .get_adjacency_sparse(GetAdjacency::Both, None, Loops::Twice)
        .unwrap();
    a.mul_vec(&[1.0, 2.0]);
}

#[test]
fn coo_matrix_to_sparsemat_round_trip() {
    let g = unit_test_graph(6, true);
    let a = g
        .get_adjacency_sparse(GetAdjacency::Both, Some(&UNIT_WEIGHTS), Loops::Once)
        .unwrap();
    let s: SparseMat = a.to_sparsemat().unwrap();
    assert_eq!(s.shape(), (6, 6));
    assert_eq!(s.to_dense().unwrap(), a.to_dense());
    let c = s.compress().unwrap();
    for &(i, j, x) in &a.entries {
        assert_eq!(c.get(i as usize, j as usize), x);
    }
    assert_eq!(c.rowsums().unwrap(), a.row_sums());
    assert_eq!(c.colsums().unwrap(), a.col_sums());
    let x = [1.0, -2.0, 0.5, 3.0, 0.0, 7.0];
    assert_eq!(c.mul_vec(&x).unwrap(), a.mul_vec(&x));

    // An empty matrix converts too; invalid hand-built entries are rejected.
    let empty = CooMatrix {
        nrow: 2,
        ncol: 3,
        entries: vec![],
    };
    assert_eq!(empty.to_sparsemat().unwrap().shape(), (2, 3));
    for bad in [(-1, 0, 1.0), (0, 3, 1.0), (2, 0, 1.0)] {
        let m = CooMatrix {
            nrow: 2,
            ncol: 3,
            entries: vec![bad],
        };
        assert_eq!(
            m.to_sparsemat().unwrap_err().kind(),
            ErrorKind::InvalidValue,
            "{bad:?}"
        );
    }
}

#[test]
fn coo_matrix_accessors() {
    let g = Graph::from_edges(&[(0, 1), (0, 2), (2, 2)], 3, true).unwrap();
    let a = g
        .get_adjacency_sparse(GetAdjacency::Both, Some(&[2.0, 0.0, 5.0]), Loops::Once)
        .unwrap();
    // The zero-weight edge 0 → 2 is structurally present.
    assert_eq!(a.nnz(), 3);
    assert_eq!(a.get(0, 2), 0.0);
    assert_eq!(a.get(2, 2), 5.0);
    assert_eq!(a.get(1, 0), 0.0);
    assert_eq!(
        a.iter().collect::<Vec<_>>(),
        vec![(0, 1, 2.0), (0, 2, 0.0), (2, 2, 5.0)]
    );
    assert_eq!(a.row_sums(), vec![2.0, 0.0, 5.0]);
    assert_eq!(a.col_sums(), vec![0.0, 2.0, 5.0]);
    assert_eq!(Matrix::from(&a), a.to_dense());
}

// ---------------------------------------------------------------------------
// get_edgelist
// ---------------------------------------------------------------------------

#[test]
fn edgelist_row_and_column_wise() {
    let g = karate();
    let flat = g.get_edgelist(false).unwrap();
    let bycol = g.get_edgelist(true).unwrap();
    let m = g.ecount();
    assert_eq!(flat.len(), 2 * m);
    for (e, &(a, b)) in g.edge_list().iter().enumerate() {
        assert_eq!((flat[2 * e], flat[2 * e + 1]), (a, b));
        assert_eq!((bycol[e], bycol[m + e]), (a, b));
    }
    let rebuilt = Graph::from_flat_edges(&flat, 34, false).unwrap();
    assert!(rebuilt.is_same_graph(&g).unwrap());
    assert!(
        Graph::empty(3, false)
            .unwrap()
            .get_edgelist(true)
            .unwrap()
            .is_empty()
    );
}

// ---------------------------------------------------------------------------
// to_directed: values pinned by tests/unit/igraph_to_directed.out
// ---------------------------------------------------------------------------

fn to_directed_unit_graph() -> Graph {
    Graph::from_edges(
        &[
            (0, 2),
            (1, 4),
            (2, 3),
            (3, 2),
            (0, 4),
            (0, 5),
            (2, 3),
            (3, 5),
            (2, 5),
            (0, 1),
        ],
        6,
        false,
    )
    .unwrap()
}

#[test]
fn to_directed_modes_match_igraph_unit_test() {
    let expected: Vec<(i64, i64)> = vec![
        (0, 2),
        (1, 4),
        (2, 3),
        (2, 3),
        (0, 4),
        (0, 5),
        (2, 3),
        (3, 5),
        (2, 5),
        (0, 1),
    ];

    let mut g = to_directed_unit_graph();
    g.to_directed(ToDirected::Arbitrary).unwrap();
    assert!(g.is_directed());
    assert_eq!(g.vcount(), 6);
    assert_eq!(g.edge_list(), expected);

    let g = to_directed_unit_graph()
        .into_directed(ToDirected::Acyclic)
        .unwrap();
    assert_eq!(g.edge_list(), expected);

    let g = to_directed_unit_graph()
        .into_directed(ToDirected::Mutual)
        .unwrap();
    let mut mutual = expected.clone();
    mutual.extend(expected.iter().map(|&(a, b)| (b, a)));
    assert_eq!(g.edge_list(), mutual);

    rng::seed(42).unwrap();
    let g = to_directed_unit_graph()
        .into_directed(ToDirected::Random)
        .unwrap();
    assert_eq!(g.ecount(), 10);
    // Each random edge joins the same vertex pair as the original.
    let orig = to_directed_unit_graph().edge_list();
    for ((a, b), (c, d)) in g.edge_list().into_iter().zip(orig) {
        assert_eq!((a.min(b), a.max(b)), (c.min(d), c.max(d)));
    }
}

#[test]
fn to_directed_random_is_reproducible_per_thread() {
    // Each thread has its own default RNG: the same seed gives the same
    // orientation in every thread, whatever the other threads do.
    let orient = |seed: u64| {
        rng::seed(seed).unwrap();
        karate()
            .into_directed(ToDirected::Random)
            .unwrap()
            .edge_list()
    };
    let reference = orient(7);
    let handles: Vec<_> = (0..4)
        .map(|t| {
            std::thread::spawn(move || {
                // Interleave another seed to stress independence.
                let other = orient(1000 + t);
                (orient(7), other)
            })
        })
        .collect();
    for h in handles {
        let (same, _) = h.join().unwrap();
        assert_eq!(same, reference);
    }
    // Random orientations keep the endpoints, and are not all "acyclic".
    let acyclic = karate()
        .into_directed(ToDirected::Acyclic)
        .unwrap()
        .edge_list();
    assert_ne!(reference, acyclic);
    for (&(a, b), &(c, d)) in reference.iter().zip(&acyclic) {
        assert_eq!((a.min(b), a.max(b)), (c, d));
    }
}

#[test]
fn acyclic_orientation_is_a_dag_in_topological_order() {
    let d = karate().into_directed(ToDirected::Acyclic).unwrap();
    assert!(d.is_dag().unwrap());
    let order = d.topological_sorting(NeighborMode::Out).unwrap();
    let mut pos = vec![0; 34];
    for (k, &v) in order.iter().enumerate() {
        pos[v as usize] = k;
    }
    for (a, b) in d.edge_list() {
        assert!(pos[a as usize] < pos[b as usize]);
    }
    // With a self-loop the result is not a DAG any more.
    let looped = Graph::from_edges(&[(0, 1), (1, 1)], 2, false)
        .unwrap()
        .into_directed(ToDirected::Acyclic)
        .unwrap();
    assert!(!looped.is_dag().unwrap());
}

#[test]
fn to_directed_is_a_noop_on_directed_graphs() {
    let mut g = Graph::from_edges(&[(1, 0), (2, 1)], 3, true).unwrap();
    g.to_directed(ToDirected::Mutual).unwrap();
    assert_eq!(g.edge_list(), vec![(1, 0), (2, 1)]);
}

#[test]
fn mutual_directed_adjacency_equals_undirected_adjacency() {
    let g = karate();
    let undirected = g
        .get_adjacency(GetAdjacency::Both, None, Loops::Once)
        .unwrap();
    let d = g.clone().into_directed(ToDirected::Mutual).unwrap();
    assert_eq!(d.ecount(), 2 * g.ecount());
    let directed = d
        .get_adjacency(GetAdjacency::Both, None, Loops::Once)
        .unwrap();
    assert_eq!(directed, undirected);
}

// ---------------------------------------------------------------------------
// to_undirected: values pinned by examples/simple/igraph_to_undirected.out
// ---------------------------------------------------------------------------

#[test]
fn to_undirected_mutual_matches_igraph_example() {
    let flat = [
        0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 3, 5, 6, 6, 5, 6, 7, 6, 7, 7, 6, 7, 8, 7, 8, 8, 7, 8, 7,
        8, 8, 9, 9, 9, 9,
    ];
    let mut g = Graph::from_flat_edges(&flat, 10, true).unwrap();
    g.to_undirected(ToUndirected::Mutual).unwrap();
    assert!(!g.is_directed());
    let mut edges = g.edge_list();
    edges.sort();
    assert_eq!(
        edges,
        vec![(5, 6), (6, 7), (7, 8), (7, 8), (8, 8), (9, 9), (9, 9)]
    );
}

#[test]
fn to_undirected_collapse_of_mutual_lattice_matches_igraph_example() {
    // A 5×5 directed lattice with mutual edges collapses to the 40 edges of
    // the undirected lattice.
    let mut edges = vec![];
    for r in 0..5i64 {
        for c in 0..5i64 {
            let v = 5 * r + c;
            if c < 4 {
                edges.push((v, v + 1));
                edges.push((v + 1, v));
            }
            if r < 4 {
                edges.push((v, v + 5));
                edges.push((v + 5, v));
            }
        }
    }
    let g = Graph::from_edges(&edges, 25, true).unwrap();
    assert_eq!(g.ecount(), 80);
    let u = g.into_undirected(ToUndirected::Collapse).unwrap();
    let mut got = u.edge_list();
    got.sort();
    assert_eq!(got.len(), 40);
    assert_eq!(&got[..4], &[(0, 1), (0, 5), (1, 2), (1, 6)]);
    assert_eq!(got.last(), Some(&(23, 24)));
}

#[test]
fn to_undirected_mutual_keeps_one_edge_per_mutual_pair() {
    // A random directed graph: Mutual keeps (number of mutual non-loop
    // edges) / 2 edges plus the loops, which is what is_mutual counts.
    rng::seed(3).unwrap();
    let g = Graph::erdos_renyi_game_gnp(40, 0.2, true, EdgeTypeSw::Simple, false).unwrap();
    let mutual = g.is_mutual(.., false).unwrap();
    let n_mutual = mutual.iter().filter(|&&m| m).count();
    assert!(n_mutual > 0 && n_mutual < g.ecount());
    let u = g.clone().into_undirected(ToUndirected::Mutual).unwrap();
    assert_eq!(u.ecount(), n_mutual / 2);
    // Reciprocity (default mode) is the fraction of mutual edges.
    assert_close(
        g.reciprocity(true, Reciprocity::Default).unwrap(),
        n_mutual as f64 / g.ecount() as f64,
        1e-12,
    );
    // Collapse keeps one edge per connected pair.
    let c = g.clone().into_undirected(ToUndirected::Collapse).unwrap();
    assert_eq!(c.ecount(), g.ecount() - n_mutual / 2);
    // Mutual is a subgraph of Collapse, which is simple.
    assert!(c.is_simple(false).unwrap());
    let ca = c
        .get_adjacency(GetAdjacency::Both, None, Loops::Twice)
        .unwrap();
    let ua = u
        .get_adjacency(GetAdjacency::Both, None, Loops::Twice)
        .unwrap();
    for (x, y) in ua.as_slice().iter().zip(ca.as_slice()) {
        assert!(x <= y);
    }
}

#[test]
fn to_undirected_with_comb_without_attributes_matches_to_undirected() {
    use igraph::attributes::{AttributeCombination, AttributeCombinationType as Comb};
    let comb = AttributeCombination::all(Comb::Sum).unwrap();
    let flat = [0, 1, 1, 0, 1, 2, 2, 2, 3, 2, 2, 3];
    for mode in [
        ToUndirected::Each,
        ToUndirected::Collapse,
        ToUndirected::Mutual,
    ] {
        let mut a = Graph::from_flat_edges(&flat, 4, true).unwrap();
        let mut b = a.clone();
        a.to_undirected(mode).unwrap();
        b.to_undirected_with_comb(mode, &comb).unwrap();
        assert!(!b.is_directed());
        assert!(a.is_same_graph(&b).unwrap(), "{mode:?}");
    }
}

#[test]
fn to_undirected_each_keeps_edge_count() {
    let g = Graph::from_edges(&[(0, 1), (1, 0), (1, 1), (2, 1)], 3, true).unwrap();
    let u = g.into_undirected(ToUndirected::Each).unwrap();
    assert_eq!(u.ecount(), 4);
    let a = u
        .get_adjacency(GetAdjacency::Both, None, Loops::Once)
        .unwrap();
    assert_eq!(a[(0, 1)], 2.0);
    assert_eq!(a[(1, 1)], 1.0);
    assert_eq!(a[(1, 2)], 1.0);
}

#[test]
fn directed_round_trip_is_identity_on_simple_graphs() {
    for g in [karate(), cycle(7), path(5), complete(6)] {
        for mode in [
            ToDirected::Arbitrary,
            ToDirected::Mutual,
            ToDirected::Acyclic,
        ] {
            let back = g
                .clone()
                .into_directed(mode)
                .and_then(|d| d.into_undirected(ToUndirected::Collapse))
                .unwrap();
            let a = g
                .get_adjacency(GetAdjacency::Both, None, Loops::Twice)
                .unwrap();
            let b = back
                .get_adjacency(GetAdjacency::Both, None, Loops::Twice)
                .unwrap();
            assert_eq!(a, b, "{mode:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// to_prufer: values from tests/unit/igraph_to_prufer.c
// ---------------------------------------------------------------------------

fn from_prufer(seq: &[i64]) -> Graph {
    Graph::from_prufer(seq).unwrap()
}

#[test]
fn prufer_round_trips_match_igraph_unit_test() {
    for seq in [&[2, 3, 2, 3][..], &[0, 2, 4, 1, 1, 0], &[2, 4, 5, 1, 3]] {
        let tree = from_prufer(seq);
        assert_eq!(tree.vcount(), seq.len() + 2);
        assert_eq!(tree.to_prufer().unwrap(), seq);
    }
}

#[test]
fn prufer_of_small_trees() {
    // Two vertices: the empty sequence.
    let edge = Graph::from_edges(&[(0, 1)], 2, false).unwrap();
    assert!(edge.to_prufer().unwrap().is_empty());
    // Direction is ignored.
    let dpath = Graph::from_edges(&[(1, 0), (1, 2), (3, 2)], 4, true).unwrap();
    assert_eq!(dpath.to_prufer().unwrap(), vec![1, 2]);
    // Each vertex appears degree - 1 times.
    let tree =
        Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (3, 4), (3, 5), (5, 6)], 7, false).unwrap();
    let seq = tree.to_prufer().unwrap();
    let deg = tree
        .degree(VertexSelector::All, NeighborMode::All, Loops::Twice)
        .unwrap();
    for v in 0..7 {
        let count = seq.iter().filter(|&&x| x == v).count() as i64;
        assert_eq!(count, deg[v as usize] - 1);
    }
}

#[test]
fn prufer_rejects_non_trees() {
    let null = Graph::empty(0, false).unwrap();
    assert_eq!(
        null.to_prufer().unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    let looped = Graph::from_edges(&[(0, 1), (1, 1)], 2, false).unwrap();
    assert!(looped.to_prufer().is_err());
    let single = Graph::empty(1, false).unwrap();
    assert_eq!(
        single.to_prufer().unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    let forest = Graph::from_edges(&[(0, 1), (2, 3)], 4, false).unwrap();
    assert_eq!(
        forest.to_prufer().unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert!(cycle(5).to_prufer().is_err());
    // to_prufer succeeds exactly on the (at least 2-vertex) trees.
    for g in [path(6), cycle(6), forest, looped, single] {
        assert_eq!(
            g.to_prufer().is_ok(),
            g.vcount() >= 2 && g.is_tree(NeighborMode::All).unwrap()
        );
    }
}

#[test]
fn prufer_of_kary_tree() {
    // The complete binary tree on 7 vertices: 0 has children 1, 2; 1 has
    // 3, 4; 2 has 5, 6. Removing leaves 3, 4 (→ 1, 1), then 1 (→ 0), then
    // 5, 6 (→ 2, 2) leaves the edge 0-2.
    let t = Graph::kary_tree(7, 2, TreeMode::Undirected).unwrap();
    assert_eq!(t.to_prufer().unwrap(), vec![1, 1, 0, 2, 2]);
    assert!(from_prufer(&[1, 1, 0, 2, 2]).is_same_graph(&t).unwrap());
}

#[test]
fn cayley_formula_via_prufer_bijection() {
    // All 4^2 = 16 sequences of length 2 over 0..4 give 16 distinct labelled
    // trees on 4 vertices, and to_prufer inverts from_prufer on each.
    let mut seen = std::collections::HashSet::new();
    for a in 0..4 {
        for b in 0..4 {
            let tree = from_prufer(&[a, b]);
            assert_eq!(tree.to_prufer().unwrap(), vec![a, b]);
            let mut e: Vec<(i64, i64)> = tree
                .edge_list()
                .into_iter()
                .map(|(x, y)| (x.min(y), x.max(y)))
                .collect();
            e.sort();
            seen.insert(e);
        }
    }
    assert_eq!(seen.len(), 16);
}

// ---------------------------------------------------------------------------
// Use case
// ---------------------------------------------------------------------------

/// Story: a small web of 5 pages links to each other. We compute PageRank
/// by power iteration on the (sparse) row-stochastic transition matrix P,
/// iterating
/// r ← (1-d)/n + d · rᵀP (dangling pages redistribute uniformly). Then we
/// throw away link directions and check that on the undirected version the
/// ranks become proportional to the degrees (d = 1), a classic identity.
#[test]
fn use_case_pagerank_by_power_iteration() {
    // Page 4 is a dangling page (no out-links).
    let links = [(0, 1), (0, 2), (1, 2), (2, 0), (3, 2), (3, 4), (1, 4)];
    let web = Graph::from_edges(&links, 5, true).unwrap();
    let p = web.get_stochastic_sparse(false, None).unwrap();
    let n = web.vcount();
    let dangling: Vec<bool> = p.row_sums().iter().map(|&s| s == 0.0).collect();
    assert_eq!(dangling, vec![false, false, false, false, true]);

    let damping = 0.85;
    let mut r = vec![1.0 / n as f64; n];
    for _ in 0..200 {
        let lost: f64 = (0..n).filter(|&i| dangling[i]).map(|i| r[i]).sum();
        let teleport = (1.0 - damping) / n as f64 + damping * lost / n as f64;
        r = p
            .vec_mul(&r)
            .iter()
            .map(|x| teleport + damping * x)
            .collect();
    }
    assert_close(r.iter().sum(), 1.0, 1e-12);
    // Page 2 collects links from 0, 1 and 3, and passes all its rank on to
    // page 0: these two are the most important pages. Page 3 has no
    // in-links, so it only gets the teleportation and dangling mass.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| r[b].total_cmp(&r[a]));
    let mut top2 = [order[0], order[1]];
    top2.sort();
    assert_eq!(top2, [0, 2]);
    assert_eq!(order[n - 1], 3);
    assert_close(
        r[3],
        (1.0 - damping) / n as f64 + damping * r[4] / n as f64,
        1e-12,
    );
    // igraph's PageRank (same damping, same uniform handling of dangling
    // pages) agrees with our power iteration.
    let pr = web
        .pagerank(None, .., &PageRankOptions::default())
        .unwrap()
        .scores;
    for v in 0..n {
        assert_close(pr[v], r[v], 1e-9);
    }

    // Undirected, no damping: the stationary distribution is degree / 2|E|.
    let social = web.into_undirected(ToUndirected::Collapse).unwrap();
    let p = social.get_stochastic_sparse(false, None).unwrap();
    let mut r = vec![1.0 / n as f64; n];
    for _ in 0..2000 {
        // Lazy walk (½ I + ½ P) to avoid periodicity.
        let step = p.vec_mul(&r);
        r = r.iter().zip(&step).map(|(x, y)| 0.5 * (x + y)).collect();
    }
    let deg = social
        .degree(VertexSelector::All, NeighborMode::All, Loops::Twice)
        .unwrap();
    let two_m = 2.0 * social.ecount() as f64;
    for v in 0..n {
        assert_close(r[v], deg[v] as f64 / two_m, 1e-9);
    }
}

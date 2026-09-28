//! Tests of transitivity, assortativity, mixing matrices and graphicality.

mod common;

use common::{assert_close, complete, cycle, karate, path};
use igraph::mixing::{
    AllowedEdgeTypes, JointDegreeDistributionOptions, is_bigraphical, is_graphical,
};
use igraph::prelude::*;

const NAN: TransitivityMode = TransitivityMode::Nan;
const ZERO: TransitivityMode = TransitivityMode::Zero;

fn star(leaves: usize) -> Graph {
    Graph::star(leaves + 1, StarMode::Undirected, 0).unwrap()
}

fn zachary() -> Graph {
    Graph::famous("Zachary").unwrap()
}

fn degrees(g: &Graph) -> Vec<i64> {
    g.degree(.., NeighborMode::All, Loops::Twice).unwrap()
}

fn assert_vec_close(a: &[f64], b: &[f64], eps: f64) {
    assert_eq!(a.len(), b.len(), "{a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        if y.is_nan() {
            assert!(x.is_nan(), "{a:?} vs {b:?}");
        } else {
            assert_close(*x, *y, eps);
        }
    }
}

// ---------------------------------------------------------------------------
// Transitivity
// ---------------------------------------------------------------------------

#[test]
fn karate_global_transitivity() {
    let g = karate();
    assert_close(
        g.transitivity_undirected(NAN).unwrap(),
        0.2556818181818182,
        1e-9,
    );
    // 45 triangles in the karate club: 3 * 45 / triples = 0.2556818 => 528 triples.
    let triples: f64 = degrees(&g).iter().map(|&d| (d * (d - 1) / 2) as f64).sum();
    assert_eq!(triples, 528.0);
    assert_close(
        g.transitivity_undirected(ZERO).unwrap() * triples / 3.0,
        45.0,
        1e-9,
    );
}

#[test]
fn karate_average_local_transitivity() {
    let g = karate();
    // Well-known value (e.g. networkx.average_clustering): 0.5706384782076823.
    let avg = g.transitivity_avglocal_undirected(ZERO).unwrap();
    assert_close(avg, 0.5706384782076823, 1e-12);
    // Every karate member has degree >= 1; only vertex 11 has degree 1.
    let local = g.transitivity_local_undirected(.., NAN).unwrap();
    assert!(local[11].is_nan());
    let valid: Vec<f64> = local.iter().copied().filter(|x| !x.is_nan()).collect();
    assert_eq!(valid.len(), 33);
    let mean = valid.iter().sum::<f64>() / 33.0;
    assert_close(
        g.transitivity_avglocal_undirected(NAN).unwrap(),
        mean,
        1e-12,
    );
    // Mr. Hi (0) and the administrator (33): low clustering, being hubs.
    assert_close(local[0], 0.15, 1e-12);
    assert_close(local[33], 0.11029411764705882, 1e-12);
    // The whole vector, from tests/unit/igraph_local_transitivity.out.
    let expected = [
        0.15,
        0.333333,
        0.244444,
        0.666667,
        0.666667,
        0.5,
        0.5,
        1.0,
        0.5,
        0.0,
        0.666667,
        f64::NAN,
        1.0,
        0.6,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.333333,
        1.0,
        1.0,
        1.0,
        0.4,
        0.333333,
        0.333333,
        1.0,
        0.166667,
        0.333333,
        0.666667,
        0.5,
        0.2,
        0.19697,
        0.110294,
    ];
    assert_vec_close(&local, &expected, 1e-6);
}

#[test]
fn complete_graph_is_fully_clustered() {
    for n in 3..8 {
        let g = complete(n);
        assert_eq!(g.transitivity_undirected(NAN).unwrap(), 1.0);
        assert_eq!(g.transitivity_avglocal_undirected(NAN).unwrap(), 1.0);
        assert!(
            g.transitivity_local_undirected(.., NAN)
                .unwrap()
                .iter()
                .all(|&t| t == 1.0)
        );
        // Barrat's transitivity with arbitrary positive weights is still 1 in K_n.
        let w: Vec<f64> = (0..g.ecount()).map(|i| 1.0 + i as f64).collect();
        let b = g.transitivity_barrat(.., Some(&w), NAN).unwrap();
        assert_vec_close(&b, &vec![1.0; n as usize], 1e-12);
    }
}

#[test]
fn triangle_free_graphs_have_zero_transitivity() {
    for g in [cycle(6), path(5), star(5)] {
        assert_eq!(g.transitivity_undirected(NAN).unwrap(), 0.0);
        assert_eq!(g.transitivity_avglocal_undirected(ZERO).unwrap(), 0.0);
    }
    // Graphs without connected triples: NaN vs zero.
    let g = Graph::from_edges(&[(0, 1), (2, 3)], 5, false).unwrap();
    assert!(g.transitivity_undirected(NAN).unwrap().is_nan());
    assert_eq!(g.transitivity_undirected(ZERO).unwrap(), 0.0);
    assert!(g.transitivity_avglocal_undirected(NAN).unwrap().is_nan());
    assert_eq!(g.transitivity_avglocal_undirected(ZERO).unwrap(), 0.0);
}

#[test]
fn transitivity_ignores_directions_and_multiplicities() {
    let und = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4, false).unwrap();
    let dir = Graph::from_edges(&[(0, 1), (2, 1), (2, 0), (3, 2), (1, 0)], 4, true).unwrap();
    let multi =
        Graph::from_edges(&[(0, 1), (0, 1), (1, 2), (2, 0), (2, 3), (2, 3)], 4, false).unwrap();
    let expected = und.transitivity_local_undirected(.., ZERO).unwrap();
    for g in [&dir, &multi] {
        assert_eq!(
            g.transitivity_undirected(NAN).unwrap(),
            und.transitivity_undirected(NAN).unwrap()
        );
        assert_vec_close(
            &g.transitivity_local_undirected(.., ZERO).unwrap(),
            &expected,
            1e-12,
        );
    }
}

#[test]
fn local_transitivity_follows_selector_order() {
    let g = karate();
    let all = g.transitivity_local_undirected(.., ZERO).unwrap();
    let some = g
        .transitivity_local_undirected(vec![33, 0, 5], ZERO)
        .unwrap();
    assert_eq!(some, vec![all[33], all[0], all[5]]);
    let one = g.transitivity_local_undirected(5, ZERO).unwrap();
    assert_eq!(one, vec![all[5]]);
    let err = g.transitivity_local_undirected(34, ZERO).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
}

#[test]
fn barrat_matches_igraph_unit_test() {
    // tests/unit/igraph_transitivity_barrat.c
    let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)], 6, false).unwrap();
    let w = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    let nan = g.transitivity_barrat(.., Some(&w), NAN).unwrap();
    assert_vec_close(
        &nan,
        &[1.0, 0.75, 0.625, 0.2777777777777778, f64::NAN, f64::NAN],
        1e-9,
    );
    let zero = g.transitivity_barrat(.., Some(&w), ZERO).unwrap();
    assert_vec_close(
        &zero,
        &[1.0, 0.75, 0.625, 0.2777777777777778, 0.0, 0.0],
        1e-9,
    );
    assert!(
        g.transitivity_barrat(VertexSelector::None, Some(&w), ZERO)
            .unwrap()
            .is_empty()
    );

    // Without weights, igraph warns and computes the unweighted version.
    let _ = igraph::error::take_warnings();
    let unweighted = g.transitivity_barrat(.., None, NAN).unwrap();
    assert_vec_close(
        &unweighted,
        &[1.0, 2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, f64::NAN, f64::NAN],
        1e-9,
    );
    assert!(!igraph::error::take_warnings().is_empty());

    // Trivial graphs.
    let null = Graph::new(0, false);
    assert!(null.transitivity_barrat(.., None, ZERO).unwrap().is_empty());
    let single = Graph::new(1, false);
    assert_eq!(
        single.transitivity_barrat(.., Some(&[]), ZERO).unwrap(),
        vec![0.0]
    );
    assert!(single.transitivity_barrat(.., Some(&[]), NAN).unwrap()[0].is_nan());
}

#[test]
fn barrat_with_uniform_weights_is_unweighted() {
    let g = karate();
    let w = vec![2.5; g.ecount()];
    let b = g.transitivity_barrat(.., Some(&w), ZERO).unwrap();
    let l = g.transitivity_local_undirected(.., ZERO).unwrap();
    assert_vec_close(&b, &l, 1e-12);
}

#[test]
fn barrat_errors() {
    let g = complete(4);
    let err = g
        .transitivity_barrat(.., Some(&[1.0, 2.0]), ZERO)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let multi = Graph::from_edges(&[(0, 1), (0, 1), (1, 2), (2, 0)], 3, false).unwrap();
    let err = multi
        .transitivity_barrat(.., Some(&[1.0; 4]), ZERO)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

// ---------------------------------------------------------------------------
// Edge clustering coefficient
// ---------------------------------------------------------------------------

#[test]
fn ecc_matches_igraph_unit_test() {
    // tests/unit/igraph_ecc.out
    let k5 = complete(5);
    assert_vec_close(&k5.ecc(.., 3, false, true).unwrap(), &[1.0; 10], 1e-12);
    assert_vec_close(
        &k5.ecc(.., 4, false, true).unwrap(),
        &[2.0 / 3.0; 10],
        1e-12,
    );
    // Selecting a range of all edges gives the same answer as "all".
    assert_eq!(
        k5.ecc(0..10, 4, false, true).unwrap(),
        k5.ecc(.., 4, false, true).unwrap()
    );

    let p2 = path(2);
    assert!(p2.ecc(.., 3, false, true).unwrap()[0].is_nan());
    // With the offset, the normalization divides 1 by s = 0.
    assert_eq!(p2.ecc(.., 3, true, true).unwrap(), vec![f64::INFINITY]);
    assert_eq!(p2.ecc(.., 4, true, true).unwrap(), vec![f64::INFINITY]);
    // igraph gives self-loops z = s = 0.
    let lp = Graph::from_edges(&[(0, 0), (0, 1), (1, 2), (2, 0)], 3, false).unwrap();
    assert!(lp.ecc(0, 3, false, true).unwrap()[0].is_nan());
    assert_eq!(lp.ecc(0, 3, true, false).unwrap(), vec![1.0]);
    assert!(
        Graph::new(1, false)
            .ecc(.., 3, false, true)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn ecc_offset_and_normalization() {
    let g = karate();
    let raw = g.ecc(.., 3, false, false).unwrap();
    let off = g.ecc(.., 3, true, false).unwrap();
    for (r, o) in raw.iter().zip(&off) {
        assert_eq!(r + 1.0, *o);
    }
    // Unnormalized, offset-free 3-ECC = number of triangles per edge; each triangle has 3 edges.
    assert_eq!(raw.iter().sum::<f64>(), 3.0 * 45.0);
    // Radicchi's definition on a specific edge: 0-1 (degrees 16 and 9) is in 7 triangles.
    let e = g.get_eid(0, 1, false).unwrap().unwrap();
    assert_vec_close(&g.ecc(e, 3, true, true).unwrap(), &[8.0 / 8.0], 1e-12);
}

#[test]
fn ecc_invalid_k() {
    let g = complete(4);
    assert_eq!(
        g.ecc(.., 2, false, true).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.ecc(.., 5, false, true).unwrap_err().kind(),
        ErrorKind::Unimplemented
    );
    assert_eq!(
        g.ecc(17, 3, false, true).unwrap_err().kind(),
        ErrorKind::InvalidEdgeId
    );
}

// ---------------------------------------------------------------------------
// Assortativity
// ---------------------------------------------------------------------------

#[test]
fn karate_degree_assortativity() {
    let g = karate();
    assert_close(
        g.assortativity_degree(false).unwrap(),
        -0.47561309768461413,
        1e-12,
    );
    // `directed` is ignored for undirected graphs.
    assert_close(
        g.assortativity_degree(true).unwrap(),
        -0.47561309768461413,
        1e-12,
    );
    // Same thing via the generic function with degrees as values.
    let d: Vec<f64> = degrees(&g).into_iter().map(|d| d as f64).collect();
    assert_close(
        g.assortativity(None, &d, None, false, true).unwrap(),
        -0.47561309768461413,
        1e-12,
    );
}

#[test]
fn assortativity_is_invariant_under_affine_maps() {
    let g = karate();
    let x: Vec<f64> = (0..34).map(|i| ((i * 7) % 11) as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| 3.0 * v - 5.0).collect();
    let rx = g.assortativity(None, &x, None, false, true).unwrap();
    let ry = g.assortativity(None, &y, None, false, true).unwrap();
    assert_close(rx, ry, 1e-12);
    assert!((-1.0..=1.0).contains(&rx));
    // The unnormalized covariance scales with the square of the factor.
    let cx = g.assortativity(None, &x, None, false, false).unwrap();
    let cy = g.assortativity(None, &y, None, false, false).unwrap();
    assert_close(cy, 9.0 * cx, 1e-9);
}

#[test]
fn integer_weights_act_as_multiplicities() {
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], 4, false).unwrap();
    let w = [1.0, 2.0, 3.0, 1.0, 2.0];
    let multi = Graph::from_edges(
        &[
            (0, 1),
            (1, 2),
            (1, 2),
            (2, 3),
            (2, 3),
            (2, 3),
            (3, 0),
            (0, 2),
            (0, 2),
        ],
        4,
        false,
    )
    .unwrap();
    let values = [0.5, 1.0, -2.0, 4.0];
    let a = g
        .assortativity(Some(&w), &values, None, false, true)
        .unwrap();
    let b = multi
        .assortativity(None, &values, None, false, true)
        .unwrap();
    assert_close(a, b, 1e-12);
}

#[test]
fn directed_assortativity_with_in_values() {
    // Directed star: all edges from the hub (out-degree 3) to leaves (in-degree 1).
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2)], 4, true).unwrap();
    let out: Vec<f64> = g
        .degree(.., NeighborMode::Out, Loops::Twice)
        .unwrap()
        .iter()
        .map(|&d| d as f64)
        .collect();
    let inn: Vec<f64> = g
        .degree(.., NeighborMode::In, Loops::Twice)
        .unwrap()
        .iter()
        .map(|&d| d as f64)
        .collect();
    let r = g.assortativity(None, &out, Some(&inn), true, true).unwrap();
    assert_close(r, g.assortativity_degree(true).unwrap(), 1e-12);
}

#[test]
fn assortativity_errors() {
    let g = path(4);
    let err = g
        .assortativity(None, &[1.0, 2.0], None, false, true)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let err = g
        .assortativity(Some(&[1.0]), &[1.0; 4], None, false, true)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let err = g.assortativity_nominal(&[0, 1], false, true).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let err = g
        .assortativity_nominal(&[0, -1, 0, 1], false, true)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

#[test]
fn nominal_assortativity_extremes() {
    // Two disjoint triangles, labeled by triangle: perfectly assortative.
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], 6, false).unwrap();
    assert_close(
        g.assortativity_nominal(&[0, 0, 0, 1, 1, 1], false, true)
            .unwrap(),
        1.0,
        1e-12,
    );
    // Unnormalized = modularity of two equal disjoint communities = 1/2.
    assert_close(
        g.assortativity_nominal(&[0, 0, 0, 1, 1, 1], false, false)
            .unwrap(),
        0.5,
        1e-12,
    );
    // Null graph: NaN.
    assert!(
        Graph::new(0, false)
            .assortativity_nominal(&[], false, true)
            .unwrap()
            .is_nan()
    );
}

// ---------------------------------------------------------------------------
// Mixing matrices
// ---------------------------------------------------------------------------

#[test]
fn joint_type_distribution_matches_igraph_unit_test() {
    // tests/unit/igraph_joint_type_distribution.c
    let null = Graph::new(0, false);
    assert_eq!(
        null.joint_type_distribution(None, &[], None, false, false)
            .unwrap()
            .shape(),
        (0, 0)
    );
    let loop1 = Graph::from_edges(&[(0, 0)], 1, false).unwrap();
    assert_eq!(
        loop1
            .joint_type_distribution(None, &[0], None, false, false)
            .unwrap()
            .to_rows(),
        vec![vec![2.0]]
    );

    let single = Graph::new(1, false);
    assert_eq!(
        single
            .joint_type_distribution(None, &[0], None, false, false)
            .unwrap()
            .to_rows(),
        vec![vec![0.0]]
    );
    // A directed loop read without directions is also counted twice.
    let dloop = Graph::from_edges(&[(0, 0)], 1, true).unwrap();
    assert_eq!(
        dloop
            .joint_type_distribution(None, &[0], None, false, false)
            .unwrap()
            .to_rows(),
        vec![vec![2.0]]
    );

    let g = Graph::from_flat_edges(
        &[1, 1, 2, 4, 0, 2, 2, 2, 3, 2, 3, 0, 2, 1, 2, 3, 4, 1, 2, 4],
        6,
        true,
    )
    .unwrap();
    let t = [0, 0, 1, 1, 2, 2];
    let same = g
        .joint_type_distribution(None, &t, None, true, false)
        .unwrap();
    assert_eq!(
        same.to_rows(),
        vec![
            vec![1.0, 1.0, 0.0],
            vec![2.0, 3.0, 2.0],
            vec![1.0, 0.0, 0.0]
        ]
    );
    assert_eq!(same.as_slice().iter().sum::<f64>(), g.ecount() as f64);
    // "From and to types are different".
    let t2 = [0, 1, 1, 1, 0, 1];
    let diff = g
        .joint_type_distribution(None, &t, Some(&t2), true, false)
        .unwrap();
    assert_eq!(
        diff.to_rows(),
        vec![vec![0.0, 2.0], vec![3.0, 4.0], vec![0.0, 1.0]]
    );
}

#[test]
fn mixing_matrix_gives_modularity_and_nominal_assortativity() {
    let g = karate();
    // The historical split of the club (Zachary 1977).
    let officer = [
        9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    ];
    let types: Vec<i64> = (0..34).map(|v| officer.contains(&v) as i64).collect();
    let m = g
        .joint_type_distribution(None, &types, None, false, true)
        .unwrap();
    assert_close(m.as_slice().iter().sum::<f64>(), 1.0, 1e-12);
    let a: Vec<f64> = (0..2).map(|i| m.row(i).iter().sum()).collect();
    let b: Vec<f64> = (0..2).map(|j| m.column(j).iter().sum()).collect();
    let sum_ab: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let q = m[(0, 0)] + m[(1, 1)] - sum_ab;
    assert_close(
        q,
        g.assortativity_nominal(&types, false, false).unwrap(),
        1e-12,
    );
    assert_close(
        q / (1.0 - sum_ab),
        g.assortativity_nominal(&types, false, true).unwrap(),
        1e-12,
    );
    // Known modularity of the factional split (Zachary's "club" attribute).
    assert_close(q, 0.3582347140039448, 1e-12);
}

#[test]
fn joint_type_distribution_with_distinct_types() {
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, true).unwrap();
    let from = [0, 0, 1, 1];
    let to = [0, 1, 2, 3];
    let m = g
        .joint_type_distribution(None, &from, Some(&to), true, false)
        .unwrap();
    assert_eq!(m.shape(), (2, 4));
    assert_eq!(
        m.to_rows(),
        vec![vec![0.0, 1.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]]
    );
    let w = [10.0, 20.0, 30.0];
    let mw = g
        .joint_type_distribution(Some(&w), &from, Some(&to), true, false)
        .unwrap();
    assert_eq!(
        mw.to_rows(),
        vec![vec![0.0, 10.0, 20.0, 0.0], vec![0.0, 0.0, 0.0, 30.0]]
    );
    assert_eq!(
        g.joint_type_distribution(None, &from, Some(&to[..2]), true, false)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn joint_degree_matrix_vs_distribution() {
    let g = karate();
    let jdm = g.joint_degree_matrix(None, None, None).unwrap();
    let maxd = *degrees(&g).iter().max().unwrap() as usize;
    assert_eq!(jdm.shape(), (maxd, maxd));
    // Undirected: symmetric, but each edge counted once overall.
    assert_eq!(jdm.transposed(), jdm);
    let mut total = 0.0;
    for (i, row) in jdm.to_rows().iter().enumerate() {
        total += row[i..].iter().sum::<f64>();
    }
    assert_eq!(total, 78.0);

    let opts = JointDegreeDistributionOptions::default().with_normalized(false);
    let p = g.joint_degree_distribution(None, &opts).unwrap();
    assert_eq!(p.shape(), (maxd + 1, maxd + 1));
    assert_eq!(p.as_slice().iter().sum::<f64>(), 2.0 * 78.0);
    for i in 1..=maxd {
        for j in 1..=maxd {
            let factor = if i == j { 2.0 } else { 1.0 };
            assert_eq!(p[(i, j)], factor * jdm[(i - 1, j - 1)]);
        }
    }
    // Normalized version sums to one.
    let pn = g
        .joint_degree_distribution(None, &JointDegreeDistributionOptions::default())
        .unwrap();
    assert_close(pn.as_slice().iter().sum::<f64>(), 1.0, 1e-12);
}

#[test]
fn joint_degree_distribution_limits_and_knn() {
    let s = star(4);
    let opts = JointDegreeDistributionOptions::default()
        .with_max_degrees(Some(2), Some(6))
        .with_normalized(false);
    let p = s.joint_degree_distribution(None, &opts).unwrap();
    assert_eq!(p.shape(), (3, 7));
    // Only leaf -> hub pairs fit (source degree 1, target degree 4).
    assert_eq!(p[(1, 4)], 4.0);
    assert_eq!(p.as_slice().iter().sum::<f64>(), 4.0);

    // Degree correlation function k_nn(k) of the karate club from P_ij:
    // average degree of the neighbors of degree-k vertices.
    let g = karate();
    let pn = g
        .joint_degree_distribution(None, &JointDegreeDistributionOptions::default())
        .unwrap();
    let d = degrees(&g);
    let knn = |k: usize| {
        let row = pn.row(k);
        let num: f64 = row.iter().enumerate().map(|(j, p)| j as f64 * p).sum();
        num / row.iter().sum::<f64>()
    };
    // Direct computation for k = 16 (Mr. Hi alone).
    let direct: f64 = g
        .neighbors(0, NeighborMode::All)
        .unwrap()
        .iter()
        .map(|&u| d[u as usize] as f64)
        .sum::<f64>()
        / 16.0;
    assert_close(knn(16), direct, 1e-12);

    let jdm = s.joint_degree_matrix(None, Some(2), Some(6)).unwrap();
    assert_eq!(jdm.shape(), (2, 6));
    // Leaf (degree 1) - hub (degree 4) edges fit as (1, 4) but not as (4, 1).
    assert_eq!(jdm[(0, 3)], 4.0);
    assert_eq!(jdm.as_slice().iter().sum::<f64>(), 4.0);
}

#[test]
fn directed_joint_degree_distribution_modes() {
    // tests/unit: an "all/all" undirected-neighbors distribution counts each edge twice.
    let g = Graph::from_edges(&[(0, 0), (0, 1), (0, 1), (1, 0)], 2, true).unwrap();
    let dflt = JointDegreeDistributionOptions::default().with_normalized(false);
    let p = g.joint_degree_distribution(None, &dflt).unwrap();
    assert_eq!(p.as_slice().iter().sum::<f64>(), 4.0);
    let both = dflt
        .with_modes(NeighborMode::All, NeighborMode::All)
        .with_directed_neighbors(false);
    let p = g.joint_degree_distribution(None, &both).unwrap();
    assert_eq!(p.as_slice().iter().sum::<f64>(), 8.0);
    let w = [1.0, 2.0, 3.0, 4.0];
    let pw = g.joint_degree_distribution(Some(&w), &dflt).unwrap();
    assert_eq!(pw.as_slice().iter().sum::<f64>(), 10.0);
    assert_eq!(
        g.joint_degree_distribution(Some(&w[..1]), &dflt)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

// ---------------------------------------------------------------------------
// Graphicality
// ---------------------------------------------------------------------------

#[test]
fn graphicality_matches_igraph_unit_test() {
    // (sequence, simple, loops, multi, multi+loops) from tests/unit/igraph_is_graphical.out
    let cases: &[(&[i64], [bool; 4])] = &[
        (&[], [true, true, true, true]),
        (&[0, 0], [true, true, true, true]),
        (&[3, 3, 3, 3, 3, 3, 3, 3], [true, true, true, true]),
        (&[3, -2, 3, 3, 3, 3, 3, 3], [false, false, false, false]),
        (&[3, 3, 3, 3, 3, 3, 3], [false, false, false, false]),
        (
            &[4, 4, 5, 3, 6, 2, 2, 8, 1, 1, 10],
            [true, true, true, true],
        ),
        (&[3, 3], [false, true, true, true]),
        (&[4, 4, 4], [false, true, true, true]),
        (&[1, 2, 3], [false, true, true, true]),
        (&[1, 2, 5], [false, false, false, true]),
        (&[1, 1, 4], [false, true, false, true]),
        (&[7, 7, 3, 2, 2, 1], [false, false, true, true]),
        (&[7, 7, 3, 3, 2, 2], [false, true, true, true]),
        (&[6, 6, 6, 4, 2, 0], [false, false, true, true]),
    ];
    let kinds = [
        AllowedEdgeTypes::SIMPLE,
        AllowedEdgeTypes::LOOPS,
        AllowedEdgeTypes::MULTI,
        AllowedEdgeTypes::ALL,
    ];
    for (seq, expected) in cases {
        for (kind, exp) in kinds.iter().zip(expected) {
            assert_eq!(
                is_graphical(seq, None, *kind).unwrap(),
                *exp,
                "{seq:?} with {kind:?}"
            );
        }
    }
}

#[test]
fn degree_sequences_of_real_graphs_are_graphical() {
    for g in [karate(), complete(7), cycle(9), star(6)] {
        assert!(is_graphical(&degrees(&g), None, EdgeTypeSw::Simple).unwrap());
    }
    // Directed: any directed simple graph's (out, in) sequence.
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (0, 2), (3, 0)], 4, true).unwrap();
    let out = g.degree(.., NeighborMode::Out, Loops::Twice).unwrap();
    let inn = g.degree(.., NeighborMode::In, Loops::Twice).unwrap();
    assert!(is_graphical(&out, Some(&inn), EdgeTypeSw::Simple).unwrap());
    // Mismatched sums: never graphical.
    assert!(!is_graphical(&[1, 1], Some(&[1, 0]), AllowedEdgeTypes::ALL).unwrap());
    // A single vertex with out/in degree 1 needs a self-loop.
    assert!(!is_graphical(&[1], Some(&[1]), EdgeTypeSw::Simple).unwrap());
    assert!(is_graphical(&[1], Some(&[1]), EdgeTypeSw::Loops).unwrap());
    // Different lengths: error.
    assert_eq!(
        is_graphical(&[1, 1], Some(&[2]), EdgeTypeSw::Simple)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn bigraphical_sequences() {
    // K_{3,3} and K_{2,4}.
    assert!(is_bigraphical(&[3, 3, 3], &[3, 3, 3], EdgeTypeSw::Simple).unwrap());
    assert!(is_bigraphical(&[4, 4], &[2, 2, 2, 2], EdgeTypeSw::Simple).unwrap());
    // Unequal sums.
    assert!(!is_bigraphical(&[2, 2], &[1, 1, 1], AllowedEdgeTypes::MULTI).unwrap());
    // Gale-Ryser violation that multi-edges fix.
    assert!(!is_bigraphical(&[3, 1], &[2, 2], EdgeTypeSw::Simple).unwrap());
    assert!(is_bigraphical(&[3, 1], &[2, 2], EdgeTypeSw::Multi).unwrap());
    // Loops are irrelevant for bipartite graphs.
    assert!(!is_bigraphical(&[3, 1], &[2, 2], EdgeTypeSw::Loops).unwrap());
    assert!(is_bigraphical(&[], &[], EdgeTypeSw::Simple).unwrap());
    assert!(!is_bigraphical(&[-1, 1], &[0, 0], AllowedEdgeTypes::ALL).unwrap());
}

#[test]
fn allowed_edge_types_flags() {
    assert_eq!(AllowedEdgeTypes::default(), AllowedEdgeTypes::SIMPLE);
    assert_eq!(AllowedEdgeTypes::SIMPLE.to_raw(), 0);
    assert_eq!(AllowedEdgeTypes::LOOPS.to_raw(), 1);
    assert_eq!(AllowedEdgeTypes::MULTI.to_raw(), 6);
    assert_eq!(AllowedEdgeTypes::ALL.to_raw(), 7);
    assert_eq!(
        AllowedEdgeTypes::SIMPLE | EdgeTypeSw::Loops,
        AllowedEdgeTypes::LOOPS
    );
    assert_eq!(EdgeTypeSw::Multi | EdgeTypeSw::Loops, AllowedEdgeTypes::ALL);
}

/// igraph 1.0.0 and 1.0.1 sum degrees in `igraph_int_t` without overflow
/// checks in `src/misc/graphicality.c` (undefined behavior in C): huge
/// sequences are answered or rejected on the Rust side.
#[test]
fn huge_degree_sequences_never_reach_igraph() {
    let big = i64::MAX;
    for allowed in [
        AllowedEdgeTypes::SIMPLE,
        AllowedEdgeTypes::LOOPS,
        AllowedEdgeTypes::MULTI,
        AllowedEdgeTypes::ALL,
    ] {
        // Would overflow `dsum`, `2 * dmax` or the parity update.
        for seq in [&[big, big][..], &[1, big], &[big / 2 + 1, big / 2 + 1]] {
            let err = is_graphical(seq, None, allowed).unwrap_err();
            assert_eq!(err.kind(), ErrorKind::InvalidValue, "{seq:?}");
            let err = is_graphical(seq, Some(seq), allowed).unwrap_err();
            assert_eq!(err.kind(), ErrorKind::InvalidValue, "{seq:?}");
            let err = is_bigraphical(seq, seq, allowed).unwrap_err();
            assert_eq!(err.kind(), ErrorKind::InvalidValue, "{seq:?}");
        }
        // A negative entry makes any sequence non-graphical, even after huge ones.
        assert!(!is_graphical(&[big, big, -1], None, allowed).unwrap());
        assert!(!is_graphical(&[big, 0], Some(&[0, -1]), allowed).unwrap());
        assert!(!is_bigraphical(&[big, big], &[-1], allowed).unwrap());
        // Different lengths are still reported as such.
        assert_eq!(
            is_graphical(&[big, big], Some(&[big]), allowed)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidValue
        );
    }
    // Large but safe sums still go to igraph and get the exact answer.
    let h = 1_i64 << 60;
    assert!(is_graphical(&[h, h], None, AllowedEdgeTypes::MULTI).unwrap());
    assert!(!is_graphical(&[h, h + 2], None, AllowedEdgeTypes::MULTI).unwrap());
    assert!(is_graphical(&[h, h + 2], None, AllowedEdgeTypes::ALL).unwrap());
    assert!(!is_graphical(&[h, h], None, AllowedEdgeTypes::SIMPLE).unwrap());
    assert!(is_graphical(&[h, 0], Some(&[0, h]), AllowedEdgeTypes::MULTI).unwrap());
    assert!(is_bigraphical(&[h], &[h], EdgeTypeSw::Multi).unwrap());
    assert!(!is_bigraphical(&[h], &[h], EdgeTypeSw::Simple).unwrap());
}

// ---------------------------------------------------------------------------
// Use case
// ---------------------------------------------------------------------------

/// A story: is the karate club a "small world" with community structure?
///
/// We compare the club with a degree-preserving randomization (seeded
/// rewiring of the same graph, whose degree sequence we first check to be
/// graphical) and show that the real club is far more clustered, that its two
/// factions mix assortatively, and that its hubs avoid each other.
#[test]
fn use_case_karate_club_mixing_story() {
    let g = Graph::famous("Zachary").unwrap();
    let deg = degrees(&g);

    // 1. The degree sequence is realizable as a simple graph (obviously), and
    //    even as a bipartite graph between the two factions' degree lists?
    assert!(is_graphical(&deg, None, EdgeTypeSw::Simple).unwrap());
    let officer = [
        9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    ];
    let faction: Vec<i64> = (0..34).map(|v| officer.contains(&v) as i64).collect();
    let (d0, d1): (Vec<i64>, Vec<i64>) = {
        let mut a = vec![];
        let mut b = vec![];
        for v in 0..34 {
            if faction[v] == 0 {
                a.push(deg[v])
            } else {
                b.push(deg[v])
            }
        }
        (a, b)
    };
    // Mr. Hi has degree 16 but the other side has only 17 members: fine for sums?
    let sums_match = d0.iter().sum::<i64>() == d1.iter().sum::<i64>();
    assert_eq!(
        is_bigraphical(&d0, &d1, EdgeTypeSw::Multi).unwrap(),
        sums_match
    );

    // 2. Clustering: the club (C ~ 0.26, <C_i> ~ 0.57) vs the expectation of a
    //    random graph with the same density, p = 2m / (n (n - 1)) ~ 0.139.
    let c = g.transitivity_undirected(NAN).unwrap();
    let c_avg = g.transitivity_avglocal_undirected(ZERO).unwrap();
    let density = g.density(None, false).unwrap();
    assert_close(density, 2.0 * 78.0 / (34.0 * 33.0), 1e-12);
    assert!(c > 1.5 * density);
    assert!(c_avg > 4.0 * density);
    //    The degree-preserving null model (seeded rewiring) has as many
    //    triangles as the club, but they sit on the hubs: the average local
    //    clustering, dominated by the many low-degree members, collapses, and
    //    the hubs no longer avoid each other.
    rng::seed(1977).unwrap();
    let mut shuffled = g.clone();
    let stats = shuffled.rewire(1000, EdgeTypeSw::Simple).unwrap();
    assert!(stats.successful_swaps > 100);
    assert_eq!(degrees(&shuffled), deg);
    assert!(is_graphical(&degrees(&shuffled), None, EdgeTypeSw::Simple).unwrap());
    let c_avg_shuffled = shuffled.transitivity_avglocal_undirected(ZERO).unwrap();
    assert!(c_avg_shuffled < 0.75 * c_avg, "{c_avg_shuffled} vs {c_avg}");
    assert!(shuffled.assortativity_degree(false).unwrap() > g.assortativity_degree(false).unwrap());

    // 3. Factions mix assortatively (strongly positive nominal assortativity)...
    let r_faction = g.assortativity_nominal(&faction, false, true).unwrap();
    assert!(r_faction > 0.7, "r = {r_faction}");
    // ...while degrees are disassortative: the two leaders are hubs that
    // mostly talk to low-degree members.
    let r_deg = g.assortativity_degree(false).unwrap();
    assert!(r_deg < -0.4);

    // 4. The joint degree matrix confirms it: the two hubs (16 and 17) are
    //    joined by no edge at all, but have many edges to degree-2..5 members.
    let jdm = g.joint_degree_matrix(None, None, None).unwrap();
    assert_eq!(jdm[(15, 16)], 0.0);
    let hub_to_small: f64 = (1..5).map(|k| jdm[(15, k)] + jdm[(16, k)]).sum();
    assert!(hub_to_small >= 15.0, "{hub_to_small}");

    // 5. Edge clustering: bridges between factions have low ECC (Radicchi's
    //    community-detection criterion) compared with intra-faction edges.
    let ecc = g.ecc(.., 3, true, true).unwrap();
    let (mut inter, mut intra) = (vec![], vec![]);
    for (e, &(u, v)) in g.edge_list().iter().enumerate() {
        if ecc[e].is_nan() {
            continue;
        }
        if faction[u as usize] == faction[v as usize] {
            intra.push(ecc[e])
        } else {
            inter.push(ecc[e])
        }
    }
    let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
    assert!(
        mean(&inter) < mean(&intra),
        "{} vs {}",
        mean(&inter),
        mean(&intra)
    );
}

/// The reciprocal-counting cases computed on the Rust side (where igraph
/// 1.0.0 and 1.0.1 would index out of bounds, `mixing_matrix()` in
/// `src/misc/mixing.c` being unchanged in 1.0.1) agree with the C code run on the equivalent
/// graph in which every connection is made explicitly bidirectional.
#[test]
fn reciprocal_mixing_matches_symmetric_digraph() {
    let und = karate();
    let mut both: Vec<(i64, i64)> = vec![];
    for (u, v) in und.edge_list() {
        both.push((u, v));
        both.push((v, u));
    }
    let sym = Graph::from_edges(&both, 34, true).unwrap();
    let w: Vec<f64> = (0..78).map(|i| 1.0 + (i % 5) as f64).collect();
    let w2: Vec<f64> = w.iter().flat_map(|&x| [x, x]).collect();

    // Distinct source and target categories.
    let from: Vec<i64> = (0..34).map(|v| v % 3).collect();
    let to: Vec<i64> = (0..34).map(|v| v % 5).collect();
    for normalized in [false, true] {
        let a = und
            .joint_type_distribution(Some(&w), &from, Some(&to), false, normalized)
            .unwrap();
        let b = sym
            .joint_type_distribution(Some(&w2), &from, Some(&to), true, normalized)
            .unwrap();
        assert_eq!(a.shape(), (3, 5));
        assert_vec_close(a.as_slice(), b.as_slice(), 1e-12);
    }

    // Non-square degree limits in an undirected graph.
    let opts = JointDegreeDistributionOptions::default().with_max_degrees(Some(4), Some(12));
    let a = und.joint_degree_distribution(None, &opts).unwrap();
    // In the symmetric digraph out-degree = in-degree = undirected degree.
    let b = sym.joint_degree_distribution(None, &opts).unwrap();
    assert_eq!(a.shape(), (5, 13));
    assert_vec_close(a.as_slice(), b.as_slice(), 1e-12);
    assert_close(a.as_slice().iter().sum::<f64>(), 1.0, 1e-12);
}

#[test]
fn directed_distribution_without_directed_neighbors() {
    // A directed graph read as undirected connections, with source out-degree
    // and target in-degree: v -> u contributes (out(v), in(u)).
    let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2), (3, 2)], 4, true).unwrap();
    let opts = JointDegreeDistributionOptions::default()
        .with_normalized(false)
        .with_directed_neighbors(false);
    let p = g.joint_degree_distribution(None, &opts).unwrap();
    let out = g.degree(.., NeighborMode::Out, Loops::Twice).unwrap();
    let inn = g.degree(.., NeighborMode::In, Loops::Twice).unwrap();
    assert_eq!(p.shape(), (3, 4)); // max out-degree 2, max in-degree 3
    let mut expected = Matrix::zeros(3, 4);
    for (u, v) in g.edge_list() {
        expected[(out[u as usize] as usize, inn[v as usize] as usize)] += 1.0;
        expected[(out[v as usize] as usize, inn[u as usize] as usize)] += 1.0;
    }
    assert_eq!(p, expected);
    assert_eq!(p.as_slice().iter().sum::<f64>(), 8.0);
}

// ---------------------------------------------------------------------------
// Soundness guards
// ---------------------------------------------------------------------------

/// igraph 1.0.0 and 1.0.1 validate `from_types` twice and never `to_types`
/// (`mixing_matrix()` in `src/misc/mixing.c`, unchanged in 1.0.1): a negative
/// target type on a directed graph would make the C code write before the
/// start of the matrix. The wrapper must reject it up front.
#[test]
fn negative_or_huge_types_are_rejected_before_calling_igraph() {
    let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    for directed in [true, false] {
        let err = g
            .joint_type_distribution(None, &[0, 0, 1], Some(&[0, -1, 0]), directed, false)
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidValue);
        let err = g
            .joint_type_distribution(None, &[0, -3, 1], None, directed, false)
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidValue);
    }
    // `type + 1` would overflow in C.
    let err = g
        .joint_type_distribution(None, &[0, i64::MAX, 0], None, true, false)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let err = g
        .assortativity_nominal(&[0, i64::MAX, 0], true, true)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

/// Limits that do not fit an `i64` used to wrap to -1 ("unlimited"); huge but
/// representable ones must give an igraph error, not an abort.
#[test]
fn degree_limits_are_validated() {
    let g = star(3);
    let err = g
        .joint_degree_matrix(None, Some(usize::MAX), None)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let opts = JointDegreeDistributionOptions::default().with_max_degrees(None, Some(usize::MAX));
    let err = g.joint_degree_distribution(None, &opts).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let opts =
        JointDegreeDistributionOptions::default().with_max_degrees(Some(i64::MAX as usize), None);
    assert!(g.joint_degree_distribution(None, &opts).is_err());
    // Non-square (Rust path) and square (C path) matrices too big to allocate.
    let opts =
        JointDegreeDistributionOptions::default().with_max_degrees(Some(1 << 40), Some(1 << 41));
    assert!(g.joint_degree_distribution(None, &opts).is_err());
    let opts =
        JointDegreeDistributionOptions::default().with_max_degrees(Some(1 << 40), Some(1 << 40));
    assert!(g.joint_degree_distribution(None, &opts).is_err());
}

#[test]
fn weighted_joint_degree_matrix() {
    // Directed path 0 -> 1 -> 2 -> 3 plus 0 -> 2: out-degrees (2,1,1,0), in-degrees (0,1,2,1).
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (0, 2)], 4, true).unwrap();
    let w = [1.0, 10.0, 100.0, 1000.0];
    let j = g.joint_degree_matrix(Some(&w), None, None).unwrap();
    assert_eq!(j.shape(), (2, 2));
    // (out 2 -> in 1): edge 0->1; (out 1 -> in 2): edge 1->2;
    // (out 1 -> in 1): edge 2->3; (out 2 -> in 2): edge 0->2.
    assert_eq!(j.to_rows(), vec![vec![100.0, 10.0], vec![1.0, 1000.0]]);
    assert_eq!(j.as_slice().iter().sum::<f64>(), w.iter().sum::<f64>());
    // A zero limit gives an empty dimension.
    assert_eq!(
        g.joint_degree_matrix(None, Some(0), None).unwrap().shape(),
        (0, 2)
    );
}

#[test]
fn directed_nominal_assortativity() {
    // All edges go from type 0 to type 1. As a directed network, sources are
    // always of type 0 and targets of type 1, so even a random directed
    // network with these endpoint types would never mix within a type: the
    // expected and observed within-type fractions are both 0, and so is the
    // coefficient. Ignoring directions, it is perfectly disassortative.
    let g = Graph::from_edges(&[(0, 2), (1, 3), (0, 3)], 4, true).unwrap();
    let types = [0, 0, 1, 1];
    assert_eq!(g.assortativity_nominal(&types, true, true).unwrap(), 0.0);
    assert_close(
        g.assortativity_nominal(&types, false, true).unwrap(),
        -1.0,
        1e-12,
    );
    // Agreement with the mixing matrix formula for a directed graph.
    let k = karate();
    let und: Vec<(i64, i64)> = k.edge_list();
    let dir = Graph::from_edges(&und, 34, true).unwrap();
    let t: Vec<i64> = (0..34).map(|v| v % 3).collect();
    let m = dir
        .joint_type_distribution(None, &t, None, true, true)
        .unwrap();
    let a: Vec<f64> = (0..3).map(|i| m.row(i).iter().sum()).collect();
    let b: Vec<f64> = (0..3).map(|j| m.column(j).iter().sum()).collect();
    let sab: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let q: f64 = (0..3).map(|i| m[(i, i)]).sum::<f64>() - sab;
    assert_close(
        dir.assortativity_nominal(&t, true, false).unwrap(),
        q,
        1e-12,
    );
    assert_close(
        dir.assortativity_nominal(&t, true, true).unwrap(),
        q / (1.0 - sab),
        1e-12,
    );
}

// ---------------------------------------------------------------------------
// igraph 1.0.1 unit test values (tests/unit/*.out)
// ---------------------------------------------------------------------------

/// The directed multigraph with loops of `global_transitivity.c` and
/// `igraph_local_transitivity.c` (20 vertices, 106 edges).
fn unit_test_multigraph() -> Graph {
    #[rustfmt::skip]
    const EDGES: [i64; 212] = [
        15, 12, 12, 10, 15, 0, 11, 10, 2, 8, 8, 6, 13, 17, 10, 10, 17, 2, 14,
        0, 16, 13, 14, 14, 0, 5, 6, 4, 0, 9, 0, 6, 10, 9, 16, 4, 14, 5, 17,
        15, 14, 9, 17, 17, 1, 4, 10, 16, 7, 0, 11, 12, 6, 13, 2, 17, 4, 0, 0,
        14, 4, 0, 6, 16, 16, 14, 13, 13, 12, 11, 3, 11, 11, 3, 6, 7, 4, 14,
        10, 8, 13, 7, 14, 2, 5, 2, 0, 14, 3, 15, 5, 5, 7, 2, 14, 15, 5, 10,
        10, 16, 7, 9, 14, 0, 15, 7, 13, 1, 15, 1, 4, 5, 4, 6, 16, 13, 6, 17,
        8, 6, 9, 3, 8, 6, 6, 14, 11, 14, 6, 10, 10, 5, 1, 0, 16, 17, 9, 1, 5,
        0, 5, 15, 8, 0, 0, 8, 5, 3, 9, 4, 13, 12, 11, 0, 11, 0, 10, 6, 4, 13,
        8, 9, 11, 11, 3, 16, 1, 2, 16, 0, 9, 8, 3, 8, 8, 7, 12, 10, 9, 3, 13,
        5, 3, 9, 6, 2, 11, 10, 1, 16, 0, 2, 10, 17, 16, 8, 11, 5, 13, 0, 19, 19,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ];
    Graph::from_flat_edges(&EDGES, 20, true).unwrap()
}

#[test]
fn transitivity_of_unit_test_multigraph() {
    // tests/unit/global_transitivity.out and igraph_local_transitivity.out:
    // directions, multi-edges and loops are all ignored.
    let expected_local = [
        0.474359,
        0.47619,
        0.428571,
        0.266667,
        0.642857,
        0.388889,
        0.533333,
        0.52381,
        0.535714,
        0.357143,
        0.285714,
        0.4,
        0.166667,
        0.416667,
        0.472222,
        0.214286,
        0.444444,
        0.4,
        f64::NAN,
        f64::NAN,
    ];
    let mut g = unit_test_multigraph();
    for step in 0..3 {
        assert_close(g.transitivity_undirected(NAN).unwrap(), 0.4357541899, 1e-10);
        let local = g.transitivity_local_undirected(.., NAN).unwrap();
        assert_vec_close(&local, &expected_local, 1e-6);
        let range = g.transitivity_local_undirected(0..20, NAN).unwrap();
        assert_vec_close(&range, &local, 0.0);
        assert_close(
            g.transitivity_avglocal_undirected(NAN).unwrap(),
            0.4126407543,
            1e-10,
        );
        match step {
            0 => g.to_undirected(ToUndirected::Collapse).unwrap(),
            1 => g.simplify(true, true).unwrap(),
            _ => {}
        }
    }
}

#[test]
fn transitivity_of_trivial_graphs() {
    // tests/unit/global_transitivity.out, igraph_local_transitivity.out and
    // igraph_transitivity_avglocal_undirected.out.
    let null = Graph::new(0, false);
    let single = Graph::new(1, false);
    let edge = path(2);
    let triangle = Graph::full(3, false, false).unwrap();
    let two_star = Graph::from_edges(&[(0, 2), (0, 1)], 3, false).unwrap();
    assert!(null.transitivity_undirected(NAN).unwrap().is_nan());
    assert!(single.transitivity_undirected(NAN).unwrap().is_nan());
    assert!(edge.transitivity_undirected(NAN).unwrap().is_nan());
    assert_eq!(triangle.transitivity_undirected(NAN).unwrap(), 1.0);
    assert_eq!(two_star.transitivity_undirected(NAN).unwrap(), 0.0);

    assert!(
        null.transitivity_local_undirected(.., NAN)
            .unwrap()
            .is_empty()
    );
    assert_vec_close(
        &single.transitivity_local_undirected(.., NAN).unwrap(),
        &[f64::NAN],
        0.0,
    );
    assert_vec_close(
        &edge.transitivity_local_undirected(.., NAN).unwrap(),
        &[f64::NAN, f64::NAN],
        0.0,
    );
    assert_eq!(
        triangle.transitivity_local_undirected(.., NAN).unwrap(),
        vec![1.0; 3]
    );
    assert_vec_close(
        &two_star.transitivity_local_undirected(.., NAN).unwrap(),
        &[0.0, f64::NAN, f64::NAN],
        0.0,
    );

    for g in [&null, &single] {
        assert!(g.transitivity_avglocal_undirected(NAN).unwrap().is_nan());
        assert_eq!(g.transitivity_avglocal_undirected(ZERO).unwrap(), 0.0);
    }
    // A directed simple graph and an undirected multigraph with a loop.
    let simple =
        Graph::from_edges(&[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)], 6, true).unwrap();
    let multi = Graph::from_edges(
        &[
            (0, 1),
            (0, 2),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 1),
            (2, 3),
            (3, 4),
            (3, 4),
        ],
        6,
        false,
    )
    .unwrap();
    assert_close(
        simple.transitivity_avglocal_undirected(ZERO).unwrap(),
        4.0 / 9.0,
        1e-12,
    );
    assert_close(
        multi.transitivity_avglocal_undirected(ZERO).unwrap(),
        4.0 / 9.0,
        1e-12,
    );
    assert_close(
        multi.transitivity_avglocal_undirected(NAN).unwrap(),
        2.0 / 3.0,
        1e-12,
    );
}

#[test]
fn assortativity_matches_igraph_unit_test() {
    // tests/unit/assortativity.out
    let null = Graph::new(0, false);
    assert!(
        null.assortativity_nominal(&[], false, true)
            .unwrap()
            .is_nan()
    );
    assert!(
        null.assortativity(None, &[], None, false, true)
            .unwrap()
            .is_nan()
    );
    let mut single = Graph::new(1, false);
    for normalized in [true, false] {
        assert!(
            single
                .assortativity_nominal(&[0], false, normalized)
                .unwrap()
                .is_nan()
        );
        assert!(
            single
                .assortativity(None, &[0.0], None, false, normalized)
                .unwrap()
                .is_nan()
        );
    }
    single.add_edges(&[(0, 0)]).unwrap();
    assert!(
        single
            .assortativity_nominal(&[0], false, true)
            .unwrap()
            .is_nan()
    );
    assert_eq!(
        single.assortativity_nominal(&[0], false, false).unwrap(),
        0.0
    );
    assert!(
        single
            .assortativity(None, &[0.0], None, false, true)
            .unwrap()
            .is_nan()
    );
    assert_eq!(
        single
            .assortativity(None, &[0.0], None, false, false)
            .unwrap(),
        0.0
    );

    // Karate club with degrees as categories: the unnormalized nominal
    // assortativity is the modularity of the partition by degree.
    let g = zachary();
    let deg = degrees(&g);
    let r = g.assortativity_nominal(&deg, false, true).unwrap();
    let q = g.assortativity_nominal(&deg, false, false).unwrap();
    assert_close(r, -0.077745, 1e-6);
    assert_close(q, -0.0693623, 1e-7);
    assert_close(q, g.modularity(&deg, None, 1.0, false).unwrap(), 1e-15);

    // Vertex ids as values; invariant under a shift of the values.
    let ids: Vec<f64> = (0..34).map(|i| i as f64).collect();
    let shifted: Vec<f64> = ids.iter().map(|x| x - 5.0).collect();
    for (normalized, expected, eps) in [(true, 0.448049, 1e-6), (false, 69.3478, 1e-4)] {
        let a = g
            .assortativity(None, &ids, None, false, normalized)
            .unwrap();
        let b = g
            .assortativity(None, &shifted, None, false, normalized)
            .unwrap();
        assert_close(a, expected, eps);
        assert_close(a, b, 1e-12);
    }
}

#[test]
fn degree_assortativity_matches_igraph_unit_test() {
    // tests/unit/assortativity.out: (graph, normalized, unnormalized).
    let small = |edges: &[(i64, i64)], directed| Graph::from_edges(edges, 4, directed).unwrap();
    let cases = [
        (zachary(), -0.475613, -13.6943),
        (
            small(&[(0, 1), (1, 2), (2, 0), (0, 3), (3, 2)], true),
            -0.666667,
            -0.16,
        ),
        // Undirected / directed with a self-loop.
        (
            small(&[(0, 1), (1, 2), (2, 0), (0, 3), (3, 3)], false),
            0.166667,
            0.04,
        ),
        (
            small(&[(0, 1), (1, 2), (2, 0), (0, 3), (3, 2), (2, 2)], true),
            -std::f64::consts::FRAC_1_SQRT_2,
            -0.333333,
        ),
        // Undirected / directed with multi-edges.
        (
            small(&[(0, 1), (1, 2), (2, 0), (0, 3), (1, 2)], false),
            -0.111111,
            -0.04,
        ),
        (
            small(&[(0, 1), (1, 2), (2, 0), (0, 3), (3, 2), (0, 2)], true),
            -0.333333,
            -0.333333,
        ),
    ];
    for (g, normalized, unnormalized) in cases {
        let directed = g.is_directed();
        assert_close(g.assortativity_degree(true).unwrap(), normalized, 1e-6);
        // The same values from `assortativity` with (out-/in-)strengths.
        let (out_mode, in_mode) = if directed {
            (NeighborMode::Out, NeighborMode::In)
        } else {
            (NeighborMode::All, NeighborMode::All)
        };
        let out = g.strength(.., out_mode, Loops::Twice, None).unwrap();
        let inn = g.strength(.., in_mode, Loops::Twice, None).unwrap();
        let values_in = directed.then_some(inn.as_slice());
        let cov = g.assortativity(None, &out, values_in, true, false).unwrap();
        assert_close(cov, unnormalized, 1e-4);
        let r = g.assortativity(None, &out, values_in, true, true).unwrap();
        assert_close(r, normalized, 1e-6);
    }
}

#[test]
fn bigraphicality_matches_igraph_unit_test() {
    // tests/unit/igraph_is_bigraphical.out: (degrees1, degrees2, simple, multi).
    let cases: [(&[i64], &[i64], bool, bool); 8] = [
        (&[], &[], true, true),
        (&[3, 3], &[1, 2, 3], false, true),
        (&[3, 2, 1], &[1, 2, 3], true, true),
        (&[1, 1, 1, 1], &[2, 3], false, false),
        (&[1, 1, 1, 1], &[2, 2], true, true),
        (&[1, 2, 0, 3, 0], &[2, 3, 1], true, true),
        (&[5, 2], &[1, 2, 2, 2], false, true),
        (&[-2, 2, 6], &[0, 2, 2, 2], false, false),
    ];
    for (d1, d2, simple, multi) in cases {
        assert_eq!(
            is_bigraphical(d1, d2, EdgeTypeSw::Simple).unwrap(),
            simple,
            "{d1:?} {d2:?}"
        );
        assert_eq!(
            is_bigraphical(d1, d2, EdgeTypeSw::Multi).unwrap(),
            multi,
            "{d1:?} {d2:?}"
        );
        // The test is symmetric in the two partitions.
        assert_eq!(is_bigraphical(d2, d1, EdgeTypeSw::Simple).unwrap(), simple);
        // A realization exists exactly when the test says so.
        for (allowed, ok) in [(EdgeTypeSw::Simple, simple), (EdgeTypeSw::Multi, multi)] {
            match Graph::realize_bipartite_degree_sequence(d1, d2, allowed, RealizeDegseq::Smallest)
            {
                Ok(g) => {
                    assert!(ok, "{d1:?} {d2:?} realized with {allowed:?}");
                    let deg = degrees(&g);
                    assert_eq!(&deg[..d1.len()], d1);
                    assert_eq!(&deg[d1.len()..], d2);
                    assert!(g.is_bipartite().unwrap());
                }
                Err(_) => assert!(!ok, "{d1:?} {d2:?} not realized with {allowed:?}"),
            }
        }
    }
}

/// `check_jdm`, `check_assort` and `check_knnk` of
/// tests/unit/igraph_joint_degree_distribution.c on seeded random graphs
/// with self-loops, with and without weights.
#[test]
fn joint_degree_distribution_identities_on_random_graphs() {
    rng::seed(137).unwrap();
    let dir_modes = [
        (NeighborMode::Out, NeighborMode::In),
        (NeighborMode::In, NeighborMode::Out),
        (NeighborMode::Out, NeighborMode::Out),
        (NeighborMode::In, NeighborMode::In),
        (NeighborMode::All, NeighborMode::In),
        (NeighborMode::All, NeighborMode::Out),
        (NeighborMode::Out, NeighborMode::All),
        (NeighborMode::In, NeighborMode::All),
    ];
    let und_modes = [(NeighborMode::All, NeighborMode::All)];
    for directed in [true, false] {
        let g =
            Graph::erdos_renyi_game_gnm(10, 30, directed, AllowedEdgeTypes::LOOPS, false).unwrap();
        let range: Vec<f64> = (0..30).map(|i| i as f64).collect();
        for weights in [None, Some(range.as_slice())] {
            // check_jdm: the JDM is P without its zero row and column, with
            // the undirected diagonal halved.
            let jdm = g.joint_degree_matrix(weights, None, None).unwrap();
            let opts = JointDegreeDistributionOptions::default().with_normalized(false);
            let p = g.joint_degree_distribution(weights, &opts).unwrap();
            assert_eq!((p.nrow() - 1, p.ncol() - 1), jdm.shape());
            assert!(p.row(0).iter().all(|&x| x == 0.0));
            assert!(p.column(0).iter().all(|&x| x == 0.0));
            let total = weights.map_or(30.0, |w| w.iter().sum()) * if directed { 1.0 } else { 2.0 };
            assert_eq!(p.as_slice().iter().sum::<f64>(), total);
            for i in 0..jdm.nrow() {
                for j in 0..jdm.ncol() {
                    let f = if !directed && i == j { 2.0 } else { 1.0 };
                    assert_eq!(p[(i + 1, j + 1)], f * jdm[(i, j)]);
                }
            }

            let modes: &[_] = if directed { &dir_modes } else { &und_modes };
            for &(from, to) in modes {
                let opts = JointDegreeDistributionOptions::default().with_modes(from, to);
                let p = g.joint_degree_distribution(weights, &opts).unwrap();
                let (nrow, ncol) = p.shape();
                let a: Vec<f64> = (0..nrow).map(|i| p.row(i).iter().sum()).collect();
                let b: Vec<f64> = (0..ncol).map(|j| p.column(j).iter().sum()).collect();

                // check_assort: unnormalized assortativity with (unweighted)
                // degree values = Σ_ij i j (P_ij - a_i b_j).
                let dfrom = g.strength(.., from, Loops::Twice, None).unwrap();
                let dto = g.strength(.., to, Loops::Twice, None).unwrap();
                let r1 = g
                    .assortativity(weights, &dfrom, directed.then_some(&dto), true, false)
                    .unwrap();
                let mut r2 = 0.0;
                for i in 0..nrow {
                    for j in 0..ncol {
                        r2 += (p[(i, j)] - a[i] * b[j]) * (i * j) as f64;
                    }
                }
                assert_close(r1, r2, 1e-12);

                // check_knnk: k_nn(k) = Σ_j j P_kj / Σ_j P_kj.
                let knnk = g
                    .degree_correlation_vector(weights, from, to, true)
                    .unwrap();
                assert_eq!(knnk.len(), nrow);
                for k in 0..nrow {
                    let num: f64 = (0..ncol).map(|j| j as f64 * p[(k, j)]).sum();
                    let expected = num / a[k];
                    if expected.is_nan() {
                        assert!(knnk[k].is_nan());
                    } else {
                        assert_close(knnk[k], expected, 1e-12);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Consistency with other modules
// ---------------------------------------------------------------------------

#[test]
fn famous_zachary_is_the_common_karate_graph() {
    let (a, b) = (zachary(), karate());
    assert_eq!(a.edge_list(), b.edge_list());
    assert_eq!(a.transitivity_local_undirected(.., NAN).unwrap().len(), 34);
}

#[test]
fn transitivity_agrees_with_triangle_counts() {
    for g in [zachary(), unit_test_multigraph(), complete(6), cycle(5)] {
        let mut simple = g.clone();
        simple.to_undirected(ToUndirected::Collapse).unwrap();
        simple.simplify(true, true).unwrap();
        let triangles = simple.count_triangles().unwrap();
        let triples: f64 = degrees(&simple)
            .iter()
            .map(|&d| (d * (d - 1) / 2) as f64)
            .sum();
        assert_close(
            g.transitivity_undirected(NAN).unwrap(),
            3.0 * triangles / triples,
            1e-12,
        );
        // Local: t_v / (d_v choose 2).
        let adjacent = simple.count_adjacent_triangles(..).unwrap();
        let local = g.transitivity_local_undirected(.., ZERO).unwrap();
        for (v, &d) in degrees(&simple).iter().enumerate() {
            let pairs = (d * (d - 1) / 2) as f64;
            let expected = if d < 2 { 0.0 } else { adjacent[v] / pairs };
            assert_close(local[v], expected, 1e-12);
        }
    }
    assert_eq!(zachary().count_triangles().unwrap(), 45.0);
}

#[test]
fn ecc_counts_listed_triangles() {
    let g = zachary();
    let raw = g.ecc(.., 3, false, false).unwrap();
    let mut per_edge = vec![0.0; g.ecount()];
    for [a, b, c] in g.list_triangles().unwrap() {
        for (u, v) in [(a, b), (b, c), (a, c)] {
            per_edge[g.get_eid(u, v, false).unwrap().unwrap() as usize] += 1.0;
        }
    }
    assert_eq!(raw, per_edge);
}

#[test]
fn nominal_assortativity_of_detected_communities_is_their_modularity() {
    let g = zachary();
    let res = g.community_multilevel(None, 1.0).unwrap();
    let membership = res.membership.clone();
    let q = *res.modularities.last().unwrap();
    assert_close(
        g.assortativity_nominal(&membership, false, false).unwrap(),
        q,
        1e-12,
    );
    assert_close(
        g.modularity(&membership, None, 1.0, false).unwrap(),
        q,
        1e-12,
    );
    // Detected communities are strongly assortative, more than the factions.
    assert!(g.assortativity_nominal(&membership, false, true).unwrap() > 0.5);
    let m = g
        .joint_type_distribution(None, &membership, None, false, true)
        .unwrap();
    let k = m.nrow();
    let trace: f64 = (0..k).map(|i| m[(i, i)]).sum();
    let ab: f64 = (0..k)
        .map(|i| m.row(i).iter().sum::<f64>() * m.column(i).iter().sum::<f64>())
        .sum();
    assert_close(trace - ab, q, 1e-12);
}

#[test]
fn graphicality_agrees_with_realization() {
    let cases: &[&[i64]] = &[
        &[],
        &[0, 0],
        &[3, 3, 3, 3, 3, 3, 3, 3],
        &[3, 3, 3, 3, 3, 3, 3],
        &[4, 4, 5, 3, 6, 2, 2, 8, 1, 1, 10],
        &[3, 3],
        &[4, 4, 4],
        &[1, 2, 3],
        &[1, 2, 5],
        &[7, 7, 3, 3, 2, 2],
        &[6, 6, 6, 4, 2, 0],
    ];
    // igraph does not realize "loops but no multi-edges" sequences.
    for allowed in [
        AllowedEdgeTypes::SIMPLE,
        AllowedEdgeTypes::MULTI,
        AllowedEdgeTypes::ALL,
    ] {
        for &seq in cases {
            let graphical = is_graphical(seq, None, allowed).unwrap();
            let realized =
                Graph::realize_degree_sequence(seq, None, allowed, RealizeDegseq::Smallest);
            assert_eq!(realized.is_ok(), graphical, "{seq:?} with {allowed:?}");
            if let Ok(g) = realized {
                assert_eq!(degrees(&g), seq);
                if allowed == AllowedEdgeTypes::SIMPLE {
                    assert!(g.is_simple(false).unwrap());
                }
            }
        }
    }
    // Directed: igraph realizes simple digraphs (Kleitman-Wang) exactly when
    // the Fulkerson-Chen-Anstee test succeeds.
    let directed: &[(&[i64], &[i64])] = &[
        (&[1, 1, 1], &[1, 1, 1]),
        (&[2, 0], &[0, 2]),
        (&[1], &[1]),
        (&[2, 2, 2], &[2, 2, 2]),
        (&[3, 3, 3], &[3, 3, 3]),
        (&[3, 1, 1, 1], &[1, 1, 1, 3]),
        (&[3, 0, 0, 0], &[0, 1, 1, 1]),
        (&[2, 2, 0], &[1, 1, 2]),
    ];
    for &(out, inn) in directed {
        let graphical = is_graphical(out, Some(inn), AllowedEdgeTypes::SIMPLE).unwrap();
        let realized = Graph::realize_degree_sequence(
            out,
            Some(inn),
            AllowedEdgeTypes::SIMPLE,
            RealizeDegseq::Smallest,
        );
        assert_eq!(realized.is_ok(), graphical, "{out:?} {inn:?}");
        if let Ok(g) = realized {
            assert_eq!(g.degree(.., NeighborMode::Out, Loops::Twice).unwrap(), out);
            assert_eq!(g.degree(.., NeighborMode::In, Loops::Twice).unwrap(), inn);
            assert!(g.is_simple(true).unwrap());
        }
    }
    // One flag type shared by the three modules: no conversion needed.
    let realize_flags: igraph::constructors::AllowedEdgeTypes = AllowedEdgeTypes::ALL;
    let game_flags: igraph::games::AllowedEdgeTypes = realize_flags;
    assert_eq!(game_flags, AllowedEdgeTypes::ALL);
}

#[test]
fn sampled_degree_sequences_preserve_graphicality() {
    rng::seed(7).unwrap();
    let g = zachary();
    let deg = degrees(&g);
    let sample =
        Graph::degree_sequence_game(&deg, None, DegreeSequenceMethod::VigerLatapy).unwrap();
    assert_eq!(degrees(&sample), deg);
    assert!(sample.is_simple(false).unwrap());
    // Same degrees (hence the same JDM dimensions), but random wiring puts
    // the triangles on the hubs: the average local clustering drops.
    let c_real = g.transitivity_avglocal_undirected(ZERO).unwrap();
    let c_random = sample.transitivity_avglocal_undirected(ZERO).unwrap();
    assert!(c_random < 0.75 * c_real, "{c_random} vs {c_real}");
    let jdm = sample.joint_degree_matrix(None, None, None).unwrap();
    assert_eq!(jdm.shape(), (17, 17));
}

#[test]
fn watts_strogatz_lattice_clustering() {
    // A ring lattice where each vertex is joined to its `k` nearest neighbors
    // on each side has C = 3 (k - 1) / (2 (2k - 1)) (global and local alike).
    for k in 2..5 {
        let g = Graph::watts_strogatz_game(1, 40, k, 0.0, AllowedEdgeTypes::SIMPLE).unwrap();
        let expected = 3.0 * (k as f64 - 1.0) / (2.0 * (2.0 * k as f64 - 1.0));
        assert_close(g.transitivity_undirected(NAN).unwrap(), expected, 1e-12);
        assert_close(
            g.transitivity_avglocal_undirected(NAN).unwrap(),
            expected,
            1e-12,
        );
        // Regular: degree assortativity undefined.
        assert!(g.assortativity_degree(false).unwrap().is_nan());
    }
    // Erdős–Rényi graphs have C ≈ p, far below a slightly rewired lattice.
    rng::seed(3).unwrap();
    let er =
        Graph::erdos_renyi_game_gnm(1000, 3000, false, AllowedEdgeTypes::SIMPLE, false).unwrap();
    let p = 3000.0 / (1000.0 * 999.0 / 2.0);
    let c = er.transitivity_undirected(NAN).unwrap();
    assert!(c < 5.0 * p, "{c} vs {p}");
    let ws = Graph::watts_strogatz_game(1, 1000, 3, 0.05, AllowedEdgeTypes::SIMPLE).unwrap();
    assert!(ws.transitivity_avglocal_undirected(ZERO).unwrap() > 0.4);
}

/// Each thread has its own default RNG: seeded computations are
/// reproducible when run concurrently.
#[test]
fn seeded_results_are_reproducible_in_parallel_threads() {
    let run = || {
        rng::seed(2024).unwrap();
        let g =
            Graph::erdos_renyi_game_gnm(60, 200, false, AllowedEdgeTypes::SIMPLE, false).unwrap();
        (
            g.transitivity_undirected(NAN).unwrap(),
            g.assortativity_degree(false).unwrap(),
            g.joint_degree_matrix(None, None, None).unwrap(),
        )
    };
    let reference = run();
    let handles: Vec<_> = (0..4).map(|_| std::thread::spawn(run)).collect();
    for h in handles {
        let (t, r, jdm) = h.join().unwrap();
        assert_eq!(t.to_bits(), reference.0.to_bits());
        assert_eq!(r.to_bits(), reference.1.to_bits());
        assert_eq!(jdm, reference.2);
    }
}

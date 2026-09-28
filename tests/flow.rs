//! Integration tests for maximum flows, minimum cuts and connectivity.

mod common;

use common::{assert_close, complete, cycle, karate, path};
use igraph::prelude::*;

/// Net flow out of each vertex (outflow - inflow) for a directed graph.
fn net_outflow(g: &Graph, flow: &[f64]) -> Vec<f64> {
    let mut net = vec![0.0; g.vcount()];
    for (e, (from, to)) in g.edge_list().into_iter().enumerate() {
        net[from as usize] += flow[e];
        net[to as usize] -= flow[e];
    }
    net
}

/// The 3-dimensional hypercube graph.
fn cube() -> Graph {
    Graph::hypercube(3, false).unwrap()
}

// ---------------------------------------------------------------------------
// Maximum flow
// ---------------------------------------------------------------------------

#[test]
fn maxflow_matches_igraph_flow2_example() {
    // examples/simple/flow2.c and its .out file.
    let g = Graph::from_edges(
        &[(0, 1), (1, 2), (2, 3), (0, 5), (5, 4), (4, 3), (3, 0)],
        6,
        true,
    )
    .unwrap();
    let capacity = [3.0, 1.0, 2.0, 10.0, 1.0, 3.0, 2.0];
    let mf = g.maxflow(0, 2, Some(&capacity)).unwrap();
    assert_eq!(mf.value, 1.0);
    assert_eq!(mf.flow, vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(mf.partition, vec![0, 1, 3, 4, 5]);
    assert_eq!(mf.partition2, vec![2]);
    assert_eq!(g.edges(&mf.cut).unwrap(), vec![(1, 2)]);
}

#[test]
fn maxflow_undirected_matches_igraph_unit_test() {
    // tests/unit/igraph_maxflow.c, the small undirected case.
    let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)], 4, false).unwrap();
    let capacity = [4.0, 2.0, 10.0, 2.0, 2.0];
    let mf = g.maxflow(0, 3, Some(&capacity)).unwrap();
    assert_eq!(mf.value, 4.0);
    assert_eq!(mf.flow, vec![4.0, 0.0, 2.0, 2.0, 2.0]);
    assert_eq!(mf.partition, vec![0, 1, 2]);
    assert_eq!(mf.partition2, vec![3]);
    assert_eq!(g.edges(&mf.cut).unwrap(), vec![(1, 3), (2, 3)]);
}

#[test]
fn undirected_flow_sign_encodes_direction() {
    // Flow from 2 to 0 along the path 0 - 1 - 2 goes "downwards" in id order.
    let g = path(3);
    let mf = g.maxflow(2, 0, None).unwrap();
    assert_eq!(mf.value, 1.0);
    assert_eq!(mf.flow, vec![-1.0, -1.0]);
}

#[test]
fn flow_is_feasible_and_conserved_on_a_dense_network() {
    // A directed network built from the karate club, orienting every edge
    // from the smaller to the larger id, with varied integer capacities.
    let edges: Vec<(i64, i64)> = common::KARATE_EDGES
        .iter()
        .map(|&(a, b)| (a.min(b), a.max(b)))
        .collect();
    let g = Graph::from_edges(&edges, 34, true).unwrap();
    let capacity: Vec<f64> = (0..g.ecount()).map(|e| (e % 7 + 1) as f64).collect();
    let (s, t) = (0, 33);
    let mf = g.maxflow(s, t, Some(&capacity)).unwrap();
    assert!(mf.value > 0.0);

    // Capacity constraints.
    for (f, c) in mf.flow.iter().zip(&capacity) {
        assert!(*f >= 0.0 && *f <= *c + 1e-9);
    }
    // Conservation everywhere except at the terminals.
    let net = net_outflow(&g, &mf.flow);
    for v in g.vertices() {
        match v {
            _ if v == s => assert_close(net[v as usize], mf.value, 1e-9),
            _ if v == t => assert_close(net[v as usize], -mf.value, 1e-9),
            _ => assert_close(net[v as usize], 0.0, 1e-9),
        }
    }
    // Max-flow = min-cut, three different ways.
    let cut_cap: f64 = mf.cut.iter().map(|&e| capacity[e as usize]).sum();
    assert_close(cut_cap, mf.value, 1e-9);
    assert_close(
        g.st_mincut_value(s, t, Some(&capacity)).unwrap(),
        mf.value,
        1e-9,
    );
    assert_close(
        g.maxflow_value(s, t, Some(&capacity)).unwrap(),
        mf.value,
        1e-9,
    );

    // The partition is a proper split of the vertex set, and every cut edge
    // goes from the source side to the target side.
    let mut all: Vec<i64> = mf.partition.iter().chain(&mf.partition2).copied().collect();
    all.sort();
    assert_eq!(all, (0..34).collect::<Vec<_>>());
    assert!(mf.partition.contains(&s) && mf.partition2.contains(&t));
    for (a, b) in g.edges(&mf.cut).unwrap() {
        assert!(mf.partition.contains(&a) && mf.partition2.contains(&b));
    }
    assert!(mf.stats.bfs_runs >= 1);
}

#[test]
fn maxflow_value_with_stats_reports_work() {
    let g = karate();
    let (value, stats) = g.maxflow_value_with_stats(0, 33, None).unwrap();
    // Vertices 0 and 33 are the two leaders; unit-capacity flow = edge connectivity.
    assert_eq!(value, g.st_edge_connectivity(0, 33).unwrap() as f64);
    assert!(stats.bfs_runs >= 1);
    assert!(stats.pushes > 0);
}

#[test]
fn maxflow_between_disconnected_vertices_is_zero() {
    let g = Graph::from_edges(&[(0, 1), (2, 3)], 4, true).unwrap();
    let mf = g.maxflow(0, 3, None).unwrap();
    assert_eq!(mf.value, 0.0);
    assert!(mf.cut.is_empty());
    assert_eq!(g.maxflow_value(3, 0, None).unwrap(), 0.0);
}

// ---------------------------------------------------------------------------
// Minimum cuts
// ---------------------------------------------------------------------------

#[test]
fn mincut_matches_igraph_example() {
    // examples/simple/igraph_mincut.c, directed cases.
    let g = Graph::from_edges(
        &[(0, 1), (1, 2), (2, 3), (0, 5), (5, 4), (4, 3), (3, 0)],
        6,
        true,
    )
    .unwrap();
    let w = [3.0, 1.0, 2.0, 10.0, 1.0, 3.0, 2.0];
    let cut = g.mincut(Some(&w)).unwrap();
    assert_eq!(cut.value, 1.0);
    assert_eq!(cut.partition, vec![1]);
    assert_eq!(cut.partition2, vec![0, 2, 3, 4, 5]);
    assert_eq!(g.edges(&cut.cut).unwrap(), vec![(1, 2)]);
    assert_eq!(g.mincut_value(Some(&w)).unwrap(), 1.0);

    // A directed path is not strongly connected: nothing needs to be cut.
    let p = Graph::from_edges(&[(4, 3), (3, 2), (2, 1), (1, 0)], 5, true).unwrap();
    let cut = p.mincut(Some(&[1.0; 4])).unwrap();
    assert_eq!(cut.value, 0.0);
    assert_eq!(cut.partition, vec![0]);
    assert_eq!(cut.partition2, vec![1, 2, 3, 4]);
    assert!(cut.cut.is_empty());
}

#[test]
fn global_mincut_is_minimum_over_st_cuts() {
    let g = karate();
    let capacity: Vec<f64> = (0..g.ecount()).map(|e| ((e * 13) % 5 + 1) as f64).collect();
    let global = g.mincut_value(Some(&capacity)).unwrap();
    let mut best = f64::INFINITY;
    for t in 1..34 {
        best = best.min(g.st_mincut_value(0, t, Some(&capacity)).unwrap());
    }
    assert_close(global, best, 1e-9);

    let cut = g.mincut(Some(&capacity)).unwrap();
    assert_close(cut.value, global, 1e-9);
    let cut_cap: f64 = cut.cut.iter().map(|&e| capacity[e as usize]).sum();
    assert_close(cut_cap, global, 1e-9);
    assert_eq!(cut.partition.len() + cut.partition2.len(), 34);
}

#[test]
fn st_mincut_separates_barbell() {
    // Two K4s joined by a single bridge between 3 and 4.
    let mut edges = vec![];
    for i in 0..4 {
        for j in i + 1..4 {
            edges.push((i, j));
            edges.push((i + 4, j + 4));
        }
    }
    edges.push((3, 4));
    let g = Graph::from_edges(&edges, 8, false).unwrap();
    let cut = g.st_mincut(0, 7, None).unwrap();
    assert_eq!(cut.value, 1.0);
    assert_eq!(g.edges(&cut.cut).unwrap(), vec![(3, 4)]);
    assert_eq!(cut.partition, vec![0, 1, 2, 3]);
    assert_eq!(cut.partition2, vec![4, 5, 6, 7]);
    assert_eq!(g.mincut_value(None).unwrap(), 1.0);
}

// ---------------------------------------------------------------------------
// Connectivity
// ---------------------------------------------------------------------------

#[test]
fn st_vertex_connectivity_matches_igraph_unit_test() {
    // tests/unit/igraph_st_vertex_connectivity.{c,out}
    let g = Graph::new(2, false);
    assert_eq!(g.st_vertex_connectivity(0, 1, VconnNei::Error).unwrap(), 0);

    let g = Graph::from_edges(&[(0, 1)], 2, false).unwrap();
    assert_eq!(
        g.st_vertex_connectivity(0, 1, VconnNei::Negative).unwrap(),
        -1
    );
    assert_eq!(
        g.st_vertex_connectivity(0, 1, VconnNei::NumberOfNodes)
            .unwrap(),
        2
    );
    let err = g.st_vertex_connectivity(0, 1, VconnNei::Error).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);

    let g = Graph::from_edges(&[(0, 1), (0, 1), (0, 1)], 2, false).unwrap();
    assert_eq!(g.st_vertex_connectivity(0, 1, VconnNei::Ignore).unwrap(), 0);

    assert_eq!(
        path(6)
            .st_vertex_connectivity(0, 5, VconnNei::Error)
            .unwrap(),
        1
    );
    assert_eq!(
        complete(6)
            .st_vertex_connectivity(0, 1, VconnNei::Ignore)
            .unwrap(),
        4
    );

    let g = Graph::from_edges(&[(0, 1), (0, 1), (1, 2), (1, 2), (1, 2), (1, 2)], 3, false).unwrap();
    assert_eq!(g.st_vertex_connectivity(0, 2, VconnNei::Error).unwrap(), 1);

    let err = g.st_vertex_connectivity(0, 0, VconnNei::Error).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

#[test]
fn connectivity_of_classic_graphs() {
    for n in 3..8 {
        assert_eq!(cycle(n).vertex_connectivity(true).unwrap(), 2);
        assert_eq!(cycle(n).edge_connectivity(false).unwrap(), 2);
        assert_eq!(
            complete(n).vertex_connectivity(false).unwrap(),
            (n - 1) as usize
        );
        assert_eq!(
            complete(n).edge_connectivity(true).unwrap(),
            (n - 1) as usize
        );
        assert_eq!(path(n).edge_connectivity(true).unwrap(), 1);
    }
    let q3 = cube();
    assert_eq!(q3.vertex_connectivity(false).unwrap(), 3);
    assert_eq!(q3.edge_connectivity(false).unwrap(), 3);
    assert_eq!(q3.cohesion(true).unwrap(), 3);
    assert_eq!(q3.adhesion(true).unwrap(), 3);

    // Trivial and disconnected graphs.
    assert_eq!(Graph::new(1, false).edge_connectivity(true).unwrap(), 0);
    assert_eq!(Graph::new(0, false).edge_connectivity(false).unwrap(), 0);
    assert_eq!(Graph::new(4, false).vertex_connectivity(true).unwrap(), 0);
}

#[test]
fn checks_do_not_change_the_result() {
    let g = karate();
    for checks in [false, true] {
        // Vertex 11 has a single neighbour, so both connectivities are 1.
        assert_eq!(g.vertex_connectivity(checks).unwrap(), 1);
        assert_eq!(g.edge_connectivity(checks).unwrap(), 1);
        assert_eq!(g.cohesion(checks).unwrap(), 1);
        assert_eq!(g.adhesion(checks).unwrap(), 1);
    }
    let q3 = cube();
    assert_eq!(
        q3.vertex_connectivity(true).unwrap(),
        q3.vertex_connectivity(false).unwrap()
    );
    assert_eq!(
        q3.edge_connectivity(true).unwrap(),
        q3.edge_connectivity(false).unwrap()
    );
}

#[test]
fn whitney_inequality_and_menger() {
    // vertex connectivity <= edge connectivity <= minimum degree.
    let g = karate();
    let min_deg = *g
        .degree(.., NeighborMode::All, Loops::Twice)
        .unwrap()
        .iter()
        .min()
        .unwrap();
    assert!(g.vertex_connectivity(true).unwrap() <= g.edge_connectivity(true).unwrap());
    assert!(g.edge_connectivity(true).unwrap() as i64 <= min_deg);

    // Menger: edge-disjoint paths = local edge connectivity = unit maxflow.
    for (s, t) in [(0, 33), (5, 16), (2, 9), (23, 26)] {
        let k = g.st_edge_connectivity(s, t).unwrap();
        assert_eq!(g.edge_disjoint_paths(s, t).unwrap(), k);
        assert_eq!(g.maxflow_value(s, t, None).unwrap(), k as f64);
    }
    // Menger for vertices, non-adjacent pairs: disjoint paths = connectivity.
    for (s, t) in [(0, 33), (4, 9), (11, 25)] {
        assert_eq!(g.get_eid(s, t, false).unwrap(), None);
        let k = g.st_vertex_connectivity(s, t, VconnNei::Error).unwrap();
        assert_eq!(g.vertex_disjoint_paths(s, t).unwrap() as i64, k);
    }
    // Adjacent pair: the direct edge adds one extra path.
    let k = g.st_vertex_connectivity(0, 1, VconnNei::Ignore).unwrap();
    assert_eq!(g.vertex_disjoint_paths(0, 1).unwrap() as i64, k + 1);
}

#[test]
fn directed_connectivity_respects_orientation() {
    // A directed cycle: one path each way around.
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], 4, true).unwrap();
    assert_eq!(g.st_edge_connectivity(0, 2).unwrap(), 1);
    assert_eq!(g.edge_disjoint_paths(2, 0).unwrap(), 1);
    assert_eq!(g.edge_connectivity(true).unwrap(), 1);
    assert_eq!(g.vertex_connectivity(false).unwrap(), 1);
    // A one-way path is not strongly connected.
    let p = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    assert_eq!(p.adhesion(true).unwrap(), 0);
    assert_eq!(p.cohesion(false).unwrap(), 0);
    assert_eq!(p.st_edge_connectivity(2, 0).unwrap(), 0);
}

// ---------------------------------------------------------------------------
// Reductions and residual graphs
// ---------------------------------------------------------------------------

#[test]
fn even_tarjan_reduction_computes_vertex_connectivity() {
    // examples/simple/even_tarjan.c: the vertex connectivity equals the
    // minimum over non-adjacent pairs (i, j) of the flow from i'' to j' in
    // the reduced graph (with unit capacities).
    for g in [cube(), karate(), cycle(7)] {
        let n = g.vcount() as i64;
        let et = g.even_tarjan_reduction().unwrap();
        assert!(et.graph.is_directed());
        assert_eq!(et.graph.vcount(), 2 * g.vcount());
        assert_eq!(et.graph.ecount(), g.vcount() + 2 * g.ecount());
        assert_eq!(et.capacity.len(), et.graph.ecount());
        assert!(et.capacity[..g.vcount()].iter().all(|&c| c == 1.0));
        assert!(et.capacity[g.vcount()..].iter().all(|&c| c == n as f64));

        let mut k2 = f64::INFINITY;
        for i in 0..n {
            for j in i + 1..n {
                if g.get_eid(i, j, false).unwrap().is_none() {
                    k2 = k2.min(et.graph.maxflow_value(i + n, j, None).unwrap());
                }
            }
        }
        assert_eq!(g.vertex_connectivity(false).unwrap() as f64, k2);
    }
}

#[test]
fn residual_graph_matches_igraph_unit_test() {
    // tests/unit/igraph_residual_graph.c
    let g = Graph::from_edges(
        &[
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 4),
            (3, 5),
            (4, 5),
        ],
        6,
        true,
    )
    .unwrap();
    let capacity = [4.0, 2.0, 2.0, 3.0, 4.0, 1.0, 2.0, 5.0];
    let flow = [3.0, 2.0, 1.0, 2.0, 3.0, 1.0, 1.0, 4.0];
    let res = g.residual_graph(&capacity, &flow).unwrap();
    assert_eq!(res.graph.vcount(), 6);
    assert_eq!(
        res.graph.edge_list(),
        vec![(0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 5)]
    );
    assert_eq!(res.capacity, vec![1.0; 6]);

    // Length mismatches are caught.
    let err = g.residual_graph(&capacity[..3], &flow).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let err = g.residual_graph(&capacity, &flow[..7]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

#[test]
fn residual_graph_of_a_maximum_flow_has_no_augmenting_path() {
    let g = Graph::from_edges(
        &[
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 4),
            (3, 5),
            (4, 5),
        ],
        6,
        true,
    )
    .unwrap();
    let capacity = [4.0, 2.0, 2.0, 3.0, 4.0, 1.0, 2.0, 5.0];
    let mf = g.maxflow(0, 5, Some(&capacity)).unwrap();
    // The residual graph keeps only the unsaturated edges; since every
    // minimum-cut edge is saturated, the target is no longer reachable.
    let res = g.residual_graph(&capacity, &mf.flow).unwrap();
    assert_eq!(res.graph.maxflow_value(0, 5, None).unwrap(), 0.0);
    for &e in &mf.cut {
        assert_eq!(mf.flow[e as usize], capacity[e as usize]);
    }

    let rr = g.reverse_residual_graph(Some(&capacity), &mf.flow).unwrap();
    assert!(rr.is_directed());
    let positive = mf.flow.iter().filter(|&&f| f > 0.0).count();
    let unsaturated = mf.flow.iter().zip(&capacity).filter(|(f, c)| f < c).count();
    assert_eq!(rr.ecount(), positive + unsaturated);
}

#[test]
fn reverse_residual_graph_with_unit_capacities() {
    let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    // Edge 0 saturated, edge 1 empty.
    let rr = g.reverse_residual_graph(None, &[1.0, 0.0]).unwrap();
    assert_eq!(rr.edge_list(), vec![(0, 1), (2, 1)]);
    let err = g.reverse_residual_graph(None, &[1.0]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

// ---------------------------------------------------------------------------
// Dominator trees
// ---------------------------------------------------------------------------

#[test]
fn dominator_tree_matches_igraph_example() {
    // examples/simple/dominator_tree.{c,out}
    let g = Graph::from_edges(
        &[
            (0, 9),
            (1, 0),
            (1, 2),
            (2, 3),
            (2, 7),
            (3, 1),
            (4, 1),
            (4, 3),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 8),
            (6, 5),
            (6, 9),
            (8, 7),
        ],
        10,
        true,
    )
    .unwrap();
    let dt = g.dominator_tree(9, NeighborMode::In).unwrap();
    let raw: Vec<i64> = [9, 0, 3, 1, 1, 1, 9, -2, -2, -1].to_vec();
    let expected: Vec<Option<i64>> = raw.iter().map(|&d| (d >= 0).then_some(d)).collect();
    assert_eq!(dt.dom, expected);
    assert_eq!(dt.leftout, vec![7, 8]);
    assert_eq!(dt.tree.vcount(), 10);
    assert_eq!(
        dt.tree.edge_list(),
        vec![(0, 9), (1, 0), (2, 3), (3, 1), (4, 1), (5, 1), (6, 9)]
    );
    assert!(dt.dominators(9).is_empty());
    assert!(dt.dominators(7).is_empty());
    assert_eq!(dt.dominators(5), vec![1, 0, 9]);
}

#[test]
fn dominator_tree_of_a_control_flow_graph() {
    // entry(0) -> if(1) -> {then(2), else(3)} -> join(4) -> exit(5), plus a
    // loop back edge 4 -> 1.
    let cfg = Graph::from_edges(
        &[(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (4, 1)],
        6,
        true,
    )
    .unwrap();
    let dt = cfg.dominator_tree(0, NeighborMode::Out).unwrap();
    assert_eq!(
        dt.dom,
        vec![None, Some(0), Some(1), Some(1), Some(1), Some(4)]
    );
    assert!(dt.leftout.is_empty());
    // Out mode: the tree edges go from the dominator to the dominated vertex.
    let mut tree_edges = dt.tree.edge_list();
    tree_edges.sort();
    assert_eq!(tree_edges, vec![(0, 1), (1, 2), (1, 3), (1, 4), (4, 5)]);
    // Neither branch dominates the join point.
    assert!(!dt.dominators(4).contains(&2));
    assert!(!dt.dominators(4).contains(&3));

    // Post-dominators: with In mode rooted at the exit.
    let pdt = cfg.dominator_tree(5, NeighborMode::In).unwrap();
    assert_eq!(pdt.dom[1], Some(4));
    assert_eq!(pdt.dom[0], Some(1));
}

#[test]
fn dominator_tree_errors() {
    let g = karate();
    let err = g.dominator_tree(0, NeighborMode::Out).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue); // undirected
    let d = Graph::from_edges(&[(0, 1)], 2, true).unwrap();
    let err = d.dominator_tree(0, NeighborMode::All).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let err = d.dominator_tree(5, NeighborMode::Out).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
}

// ---------------------------------------------------------------------------
// Listing cuts
// ---------------------------------------------------------------------------

#[test]
fn all_st_cuts_matches_igraph_unit_test() {
    // tests/unit/igraph_all_st_cuts.{c,out}: 9 cuts.
    let g = Graph::from_edges(
        &[(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (1, 5), (5, 4)],
        6,
        true,
    )
    .unwrap();
    let all = g.all_st_cuts(0, 4).unwrap();
    assert_eq!(
        all.partition1s,
        vec![
            vec![0],
            vec![0, 1],
            vec![0, 1, 5],
            vec![0, 1, 3],
            vec![0, 1, 3, 5],
            vec![0, 1, 2],
            vec![0, 1, 2, 5],
            vec![0, 1, 2, 3],
            vec![0, 1, 2, 3, 5],
        ]
    );
    assert_eq!(
        all.cuts,
        vec![
            vec![0],
            vec![1, 2, 5],
            vec![1, 2, 6],
            vec![1, 4, 5],
            vec![1, 4, 6],
            vec![2, 3, 5],
            vec![2, 3, 6],
            vec![3, 4, 5],
            vec![3, 4, 6],
        ]
    );
    // Each cut is exactly the set of edges leaving its partition.
    let edges = g.edge_list();
    for (cut, part) in all.cuts.iter().zip(&all.partition1s) {
        let leaving: Vec<i64> = (0..edges.len() as i64)
            .filter(|&e| {
                let (a, b) = edges[e as usize];
                part.contains(&a) && !part.contains(&b)
            })
            .collect();
        assert_eq!(&leaving, cut);
    }
}

#[test]
fn all_st_cuts_requires_directed_graph() {
    let err = path(3).all_st_cuts(0, 2).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Unimplemented);
}

#[test]
fn all_st_mincuts_matches_igraph_unit_test() {
    // tests/unit/igraph_all_st_mincuts.{c,out}, graphs 1 and 4.
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4)], 5, true).unwrap();
    let m = g.all_st_mincuts(0, 4, None).unwrap();
    assert_eq!(m.value, 1.0);
    assert_eq!(m.cuts, vec![vec![0], vec![1], vec![2], vec![3]]);
    assert_eq!(
        m.partition1s,
        vec![vec![0], vec![0, 1], vec![0, 1, 2], vec![0, 1, 2, 3]]
    );

    let g = Graph::from_edges(
        &[
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (1, 4),
            (4, 2),
            (1, 5),
            (5, 2),
            (1, 6),
            (6, 2),
            (1, 7),
            (7, 2),
            (1, 8),
            (8, 2),
        ],
        9,
        true,
    )
    .unwrap();
    let m = g.all_st_mincuts(0, 3, None).unwrap();
    assert_eq!(m.value, 2.0);
    let as_pairs: Vec<Vec<(i64, i64)>> = m.cuts.iter().map(|c| g.edges(c).unwrap()).collect();
    assert_eq!(
        as_pairs,
        vec![
            vec![(0, 1), (0, 2)],
            vec![(0, 1), (2, 3)],
            vec![(1, 3), (2, 3)],
        ]
    );
    assert_eq!(m.partition1s[0], vec![0]);
    assert_eq!(m.partition1s[1], vec![0, 2]);
    assert_eq!(m.partition1s[2], vec![0, 2, 1, 8, 7, 6, 5, 4]);
}

#[test]
fn every_minimum_cut_is_a_cut() {
    // All minimum cuts appear among all cuts, and have the minimum value.
    let g = Graph::from_edges(
        &[(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (1, 5), (5, 4)],
        6,
        true,
    )
    .unwrap();
    let capacity = [5.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0];
    let all = g.all_st_cuts(0, 4).unwrap();
    let min = g.all_st_mincuts(0, 4, Some(&capacity)).unwrap();
    let cost = |c: &Vec<i64>| c.iter().map(|&e| capacity[e as usize]).sum::<f64>();
    let best = all.cuts.iter().map(cost).fold(f64::INFINITY, f64::min);
    assert_eq!(best, min.value);
    assert_eq!(best, g.maxflow_value(0, 4, Some(&capacity)).unwrap());
    let optimal = all.cuts.iter().filter(|c| cost(c) == best).count();
    assert_eq!(min.cuts.len(), optimal);
    for c in &min.cuts {
        let mut sorted = c.clone();
        sorted.sort();
        assert!(all.cuts.contains(&sorted));
        assert_eq!(cost(c), best);
    }
}

#[test]
fn all_st_mincuts_errors() {
    let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    let err = g.all_st_mincuts(0, 2, Some(&[1.0, 0.0])).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue); // non-positive capacity
    let err = g.all_st_mincuts(0, 0, None).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let err = g.all_st_mincuts(0, 9, None).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    let err = path(3).all_st_mincuts(0, 2, None).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Unimplemented);
}

// ---------------------------------------------------------------------------
// Gomory-Hu trees
// ---------------------------------------------------------------------------

#[test]
fn gomory_hu_tree_encodes_all_pairwise_flows() {
    // tests/unit/igraph_gomory_hu_tree.c
    let g = Graph::from_edges(
        &[
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 4),
            (3, 4),
            (3, 5),
            (4, 5),
        ],
        6,
        false,
    )
    .unwrap();
    let capacity = [1.0, 7.0, 1.0, 3.0, 2.0, 4.0, 1.0, 6.0, 2.0];
    let gh = g.gomory_hu_tree(Some(&capacity)).unwrap();
    assert_eq!(gh.tree.vcount(), 6);
    assert_eq!(gh.tree.ecount(), 5);
    assert_eq!(gh.flows.len(), 5);
    assert!(!gh.tree.is_directed());
    for u in 0..6 {
        for v in u + 1..6 {
            let expected = g.maxflow_value(u, v, Some(&capacity)).unwrap();
            assert_eq!(gh.flow_between(u, v), Some(expected), "pair ({u}, {v})");
        }
    }
    assert_eq!(gh.flow_between(2, 2), None);
    assert_eq!(gh.flow_between(0, 17), None);
}

#[test]
fn gomory_hu_tree_of_karate_and_complete_graphs() {
    // Github issue #1810: K4 with unit capacities, every flow is 3.
    let gh = complete(4).gomory_hu_tree(None).unwrap();
    assert_eq!(gh.flows, vec![3.0; 3]);

    let g = karate();
    let gh = g.gomory_hu_tree(None).unwrap();
    // The smallest tree annotation is the global minimum cut.
    let min = gh.flows.iter().copied().fold(f64::INFINITY, f64::min);
    assert_eq!(min, g.mincut_value(None).unwrap());
    for (u, v) in [(0, 33), (0, 11), (5, 6), (32, 33)] {
        assert_eq!(
            gh.flow_between(u, v).unwrap(),
            g.maxflow_value(u, v, None).unwrap()
        );
    }
}

#[test]
fn gomory_hu_tree_edge_cases() {
    let gh = Graph::new(0, false).gomory_hu_tree(None).unwrap();
    assert_eq!(gh.tree.vcount(), 0);
    assert!(gh.flows.is_empty());

    let d = Graph::from_edges(&[(0, 1)], 2, true).unwrap();
    assert_eq!(
        d.gomory_hu_tree(None).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

// ---------------------------------------------------------------------------
// Argument validation
// ---------------------------------------------------------------------------

#[test]
fn invalid_arguments_are_reported() {
    let g = cycle(4);
    let kind = |r: Result<f64>| r.unwrap_err().kind();
    assert_eq!(
        kind(g.maxflow_value(0, 4, None)),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        kind(g.maxflow_value(-1, 2, None)),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(kind(g.maxflow_value(1, 1, None)), ErrorKind::InvalidValue);
    assert_eq!(
        kind(g.maxflow_value(0, 2, Some(&[1.0]))),
        ErrorKind::InvalidValue
    );
    assert_eq!(kind(g.st_mincut_value(0, 0, None)), ErrorKind::InvalidValue);
    assert_eq!(
        kind(g.mincut_value(Some(&[1.0; 5]))),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.maxflow(0, 9, None).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        g.st_mincut(0, 2, Some(&[])).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.mincut(Some(&[2.0])).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.st_edge_connectivity(0, 0).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.st_edge_connectivity(0, 7).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        g.edge_disjoint_paths(1, 1).unwrap_err().kind(),
        ErrorKind::Unimplemented
    );
    assert_eq!(
        g.vertex_disjoint_paths(2, 2).unwrap_err().kind(),
        ErrorKind::Unimplemented
    );
    assert_eq!(
        g.st_vertex_connectivity(0, 8, VconnNei::Ignore)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        g.gomory_hu_tree(Some(&[1.0])).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn invalid_capacity_values_are_rejected() {
    // igraph does not validate capacity signs: without the Rust-side check a
    // negative capacity would silently yield a "minimum cut" of value -1.
    let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    for bad in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let cap = [bad, 2.0];
        let err = g.maxflow(0, 2, Some(&cap)).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidValue);
        assert!(err.message().contains("edge 0"));
        assert_eq!(
            g.maxflow_value(0, 2, Some(&cap)).unwrap_err().kind(),
            ErrorKind::InvalidValue
        );
        assert_eq!(
            g.st_mincut(0, 2, Some(&cap)).unwrap_err().kind(),
            ErrorKind::InvalidValue
        );
        assert_eq!(
            g.mincut(Some(&cap)).unwrap_err().kind(),
            ErrorKind::InvalidValue
        );
        assert_eq!(
            g.mincut_value(Some(&cap)).unwrap_err().kind(),
            ErrorKind::InvalidValue
        );
        assert_eq!(
            g.all_st_mincuts(0, 2, Some(&cap)).unwrap_err().kind(),
            ErrorKind::InvalidValue
        );
        assert_eq!(
            g.residual_graph(&cap, &[0.0, 0.0]).unwrap_err().kind(),
            ErrorKind::InvalidValue
        );
        assert_eq!(
            g.reverse_residual_graph(Some(&cap), &[0.0, 0.0])
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidValue
        );
    }
    // Zero capacities are fine: they simply block the edge.
    assert_eq!(g.maxflow_value(0, 2, Some(&[0.0, 2.0])).unwrap(), 0.0);
    // ... except for all_st_mincuts, which igraph requires to be positive.
    assert_eq!(
        g.all_st_mincuts(0, 2, Some(&[0.0, 2.0]))
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.all_st_mincuts(0, 3, None).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
}

#[test]
fn mincut_of_degenerate_graphs() {
    // Fewer than two vertices: there is nothing to separate, igraph reports
    // an infinite minimum cut.
    for directed in [false, true] {
        let g = Graph::new(1, directed);
        assert_eq!(g.mincut_value(None).unwrap(), f64::INFINITY);
        let cut = g.mincut(None).unwrap();
        assert_eq!(cut.value, f64::INFINITY);
        assert!(cut.cut.is_empty());
    }
    // Two isolated vertices are already disconnected.
    for directed in [false, true] {
        let g = Graph::new(2, directed);
        let cut = g.mincut(None).unwrap();
        assert_eq!(cut.value, 0.0);
        assert!(cut.cut.is_empty());
        assert_eq!(cut.partition, vec![0]);
        assert_eq!(cut.partition2, vec![1]);
        assert_eq!(g.vertex_connectivity(true).unwrap(), 0);
        assert_eq!(g.edge_connectivity(false).unwrap(), 0);
    }
}

#[test]
fn mincut_of_graphs_without_vertices() {
    // Undirected: igraph treats the null graph as disconnected (value 0).
    let g = Graph::new(0, false);
    assert_eq!(g.mincut_value(None).unwrap(), 0.0);
    let cut = g.mincut(None).unwrap();
    assert_eq!(cut.value, 0.0);
    assert!(cut.cut.is_empty() && cut.partition.is_empty() && cut.partition2.is_empty());
    // Directed: no pair of vertices to minimise over, so the value is infinite.
    let g = Graph::new(0, true);
    assert_eq!(g.mincut_value(None).unwrap(), f64::INFINITY);
    let cut = g.mincut(None).unwrap();
    assert_eq!(cut.value, f64::INFINITY);
    assert!(cut.cut.is_empty() && cut.partition.is_empty() && cut.partition2.is_empty());
}

#[test]
fn all_st_cuts_lists_only_minimal_cuts() {
    // On a directed path each single edge is a cut; the non-minimal cut made
    // of both edges is not listed.
    let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    let all = g.all_st_cuts(0, 2).unwrap();
    assert_eq!(all.cuts, vec![vec![0], vec![1]]);
    assert_eq!(all.partition1s, vec![vec![0], vec![0, 1]]);
    // Nothing to cut: the target is unreachable, or equal to the source.
    assert!(g.all_st_cuts(2, 0).unwrap().cuts.is_empty());
    assert!(g.all_st_cuts(1, 1).unwrap().cuts.is_empty());
    let m = g.all_st_mincuts(2, 0, None).unwrap();
    assert_eq!(m.value, 0.0);
    assert!(m.cuts.is_empty() && m.partition1s.is_empty());
}

#[test]
fn vertex_disjoint_paths_count_only_forward_direct_edges() {
    // 0 -> 1 directly, 0 -> 2 -> 1 around, and a reverse edge 1 -> 0.
    let d = Graph::from_edges(&[(0, 1), (1, 0), (0, 2), (2, 1)], 3, true).unwrap();
    assert_eq!(d.vertex_disjoint_paths(0, 1).unwrap(), 2);
    // From 1 to 0 only the direct edge 1 -> 0 exists.
    assert_eq!(d.vertex_disjoint_paths(1, 0).unwrap(), 1);
    // Undirected, both parallel 0 - 1 edges count, plus the detour via 2.
    let u = Graph::from_edges(&[(0, 1), (1, 0), (0, 2), (2, 1)], 3, false).unwrap();
    assert_eq!(u.vertex_disjoint_paths(0, 1).unwrap(), 3);
    assert_eq!(u.vertex_disjoint_paths(1, 0).unwrap(), 3);
}

#[test]
fn st_mincut_matches_igraph_unit_test() {
    // tests/unit/igraph_st_mincut.{c,out}
    let g = Graph::from_edges(&[(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)], 5, true).unwrap();
    let cut = g.st_mincut(0, 4, None).unwrap();
    assert_eq!(cut.value, 1.0);
    assert_eq!(cut.cut, vec![0]);
    assert_eq!(cut.partition, vec![0]);
    let mut p2 = cut.partition2.clone();
    p2.sort();
    assert_eq!(p2, vec![1, 2, 3, 4]);

    let capacity = [8.0, 2.0, 3.0, 3.0, 2.0];
    let cut = g.st_mincut(0, 4, Some(&capacity)).unwrap();
    assert_eq!(cut.value, 4.0);
    let sorted = |v: &[i64]| {
        let mut v = v.to_vec();
        v.sort();
        v
    };
    assert_eq!(sorted(&cut.cut), vec![1, 4]);
    assert_eq!(sorted(&cut.partition), vec![0, 1, 3]);
    assert_eq!(sorted(&cut.partition2), vec![2, 4]);
}

#[test]
fn st_values_match_igraph_unit_tests() {
    // tests/unit/igraph_st_mincut_value.c and igraph_st_edge_connectivity.c
    let g = Graph::from_edges(
        &[
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 4),
            (3, 5),
            (4, 5),
        ],
        6,
        true,
    )
    .unwrap();
    let capacity = [5.0, 2.0, 2.0, 3.0, 4.0, 1.0, 2.0, 5.0];
    assert_eq!(g.st_mincut_value(0, 5, Some(&capacity)).unwrap(), 7.0);
    assert_eq!(g.maxflow_value(0, 5, Some(&capacity)).unwrap(), 7.0);
    assert_eq!(g.st_edge_connectivity(0, 5).unwrap(), 2);
}

#[test]
fn disjoint_paths_match_igraph_unit_tests() {
    // tests/unit/igraph_edge_disjoint_paths.c (the self-loop 3-3 is ignored).
    let mut g = Graph::from_edges(
        &[
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 4),
            (3, 5),
            (4, 5),
            (3, 3),
        ],
        6,
        true,
    )
    .unwrap();
    assert_eq!(g.edge_disjoint_paths(0, 5).unwrap(), 2);
    assert_eq!(g.edge_disjoint_paths(0, 3).unwrap(), 1);
    assert_eq!(g.edge_disjoint_paths(3, 0).unwrap(), 0);
    assert_eq!(g.edge_disjoint_paths(3, 5).unwrap(), 2);
    g.to_undirected(ToUndirected::Each).unwrap();
    assert_eq!(g.edge_disjoint_paths(4, 3).unwrap(), 3);

    // tests/unit/igraph_vertex_disjoint_paths.c: multi-edges, a loop and
    // direct edges between the endpoints, which each count as one path.
    let mut g = Graph::from_edges(
        &[
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 4),
            (3, 5),
            (4, 5),
            (0, 5),
            (3, 3),
            (5, 2),
            (1, 3),
            (3, 1),
        ],
        7,
        true,
    )
    .unwrap();
    assert_eq!(g.vertex_disjoint_paths(0, 5).unwrap(), 3);
    assert_eq!(g.vertex_disjoint_paths(1, 3).unwrap(), 2);
    assert_eq!(g.vertex_disjoint_paths(4, 0).unwrap(), 0);
    g.to_undirected(ToUndirected::Each).unwrap();
    assert_eq!(g.vertex_disjoint_paths(4, 0).unwrap(), 3);
    assert_eq!(g.vertex_disjoint_paths(1, 3).unwrap(), 5);
}

#[test]
fn adhesion_and_cohesion_match_igraph_unit_tests() {
    // tests/unit/igraph_adhesion.c and igraph_cohesion.c
    let edges = [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 4),
        (3, 5),
        (4, 5),
        (1, 6),
        (6, 3),
    ];
    let d = Graph::from_edges(&edges, 7, true).unwrap();
    assert_eq!(d.adhesion(true).unwrap(), 0);
    let mut with_back_edge = edges.to_vec();
    with_back_edge.push((5, 0));
    let d = Graph::from_edges(&with_back_edge, 7, true).unwrap();
    assert_eq!(d.cohesion(true).unwrap(), 1);
    assert_eq!(d.cohesion(false).unwrap(), 1);
    let u = Graph::from_edges(&edges, 7, false).unwrap();
    for checks in [true, false] {
        assert_eq!(u.adhesion(checks).unwrap(), 2);
        assert_eq!(u.cohesion(checks).unwrap(), 2);
    }
}

#[test]
fn famous_graphs_have_their_textbook_connectivity() {
    // Connectivity of well-known graphs: (name, vertex, edge connectivity).
    for (name, kappa, lambda) in [
        ("Petersen", 3, 3),
        ("Dodecahedron", 3, 3),
        ("Icosahedron", 5, 5),
        ("Octahedron", 4, 4),
        ("Heawood", 3, 3),
        ("Tutte", 3, 3),
        ("Zachary", 1, 1),
    ] {
        let g = Graph::famous(name).unwrap();
        assert_eq!(g.vertex_connectivity(false).unwrap(), kappa, "{name}");
        assert_eq!(g.edge_connectivity(false).unwrap(), lambda, "{name}");
        assert_eq!(g.mincut_value(None).unwrap(), lambda as f64, "{name}");
    }
    // Complete bipartite K(m, n): both connectivities are min(m, n).
    for (m, n) in [(2, 5), (3, 3), (4, 6)] {
        let k = Graph::full_bipartite(m, n, false, NeighborMode::All).unwrap();
        assert_eq!(k.graph.vertex_connectivity(false).unwrap(), m.min(n));
        assert_eq!(k.graph.edge_connectivity(false).unwrap(), m.min(n));
    }
}

#[test]
fn connectivity_agrees_with_separators_bridges_and_articulation_points() {
    let zachary = Graph::famous("Zachary").unwrap();
    assert_eq!(zachary.edge_list(), karate().edge_list());
    for g in [
        zachary,
        cube(),
        Graph::famous("Petersen").unwrap(),
        Graph::ring(8, false, false, true).unwrap(),
        Graph::square_lattice(&[3, 4], 1, false, false, None).unwrap(),
    ] {
        // Every minimum-size separator has exactly `vertex_connectivity`
        // vertices, and really separates the graph.
        let kappa = g.vertex_connectivity(false).unwrap();
        let seps = g.minimum_size_separators().unwrap();
        assert!(!seps.is_empty());
        for s in &seps {
            assert_eq!(s.len(), kappa);
            assert!(g.is_separator(s.as_slice()).unwrap());
        }
        // Linear-time shortcuts: articulation points iff kappa == 1, bridges
        // iff lambda == 1 (the graphs are connected, with >= 3 vertices).
        let lambda = g.edge_connectivity(false).unwrap();
        assert_eq!(!g.articulation_points().unwrap().is_empty(), kappa == 1);
        assert_eq!(!g.bridges().unwrap().is_empty(), lambda == 1);
    }
}

#[test]
fn maximum_bipartite_matching_is_a_unit_flow() {
    // Classic reduction: a super-source feeding the left side, a super-sink
    // fed by the right side, unit capacities everywhere.
    let g = Graph::from_edges(
        &[
            (0, 5),
            (0, 6),
            (1, 5),
            (2, 6),
            (2, 7),
            (3, 7),
            (3, 8),
            (4, 8),
        ],
        9,
        false,
    )
    .unwrap();
    let types: Vec<bool> = (0..9).map(|v| v >= 5).collect();
    let matching = g.maximum_bipartite_matching(&types, None).unwrap();

    let (source, sink) = (9, 10);
    let mut edges: Vec<(i64, i64)> = g.edge_list();
    edges.extend((0..5).map(|l| (source, l)));
    edges.extend((5..9).map(|r| (r, sink)));
    let net = Graph::from_edges(&edges, 11, true).unwrap();
    let mf = net.maxflow(source, sink, None).unwrap();
    assert_eq!(mf.value, 4.0);
    assert_eq!(matching.size, 4);
    // The saturated left-to-right edges form a matching of that size.
    let used: Vec<(i64, i64)> = (0..g.ecount())
        .filter(|&e| mf.flow[e] == 1.0)
        .map(|e| edges[e])
        .collect();
    assert_eq!(used.len(), 4);
    let mut endpoints: Vec<i64> = used.iter().flat_map(|&(a, b)| [a, b]).collect();
    endpoints.sort();
    endpoints.dedup();
    assert_eq!(endpoints.len(), 8);
}

#[test]
fn dimacs_instance_round_trip() {
    // A maximum flow instance in DIMACS format (1-based vertex ids).
    let text = "c CLRS network\np max 6 10\nn 1 s\nn 6 t\n\
                a 1 2 16\na 1 3 13\na 2 3 10\na 3 2 4\na 2 4 12\n\
                a 4 3 9\na 3 5 14\na 5 4 7\na 4 6 20\na 5 6 4\n";
    let inst = Graph::read_graph_dimacs_flow_from_str(text, true).unwrap();
    let igraph::foreign::DimacsProblem::Max {
        source: Some(s),
        target: Some(t),
        capacity,
    } = &inst.problem
    else {
        panic!("not a max-flow problem: {:?}", inst.problem);
    };
    assert_eq!((*s, *t), (0, 5));
    assert_eq!(
        inst.graph.maxflow_value(*s, *t, Some(capacity)).unwrap(),
        23.0
    );

    // Writing it back and re-reading gives the same instance.
    let written = inst
        .graph
        .write_graph_dimacs_flow_to_string(*s, *t, capacity)
        .unwrap();
    let again = Graph::read_graph_dimacs_flow_from_str(&written, true).unwrap();
    assert_eq!(again.problem, inst.problem);
    assert_eq!(again.graph.edge_list(), inst.graph.edge_list());
}

#[test]
fn flow_identities_on_random_graphs() {
    rng::seed(20240901).unwrap();
    for round in 0..10 {
        let directed = round % 2 == 1;
        let g = Graph::erdos_renyi_game_gnm(12, 30, directed, EdgeTypeSw::Simple, false).unwrap();
        let capacity: Vec<f64> = (0..g.ecount()).map(|e| (e % 5 + 1) as f64).collect();

        // Max-flow = min-cut for every ordered pair from vertex 0.
        for t in 1..12 {
            let mf = g.maxflow(0, t, Some(&capacity)).unwrap();
            let cut: f64 = mf.cut.iter().map(|&e| capacity[e as usize]).sum();
            assert_close(cut, mf.value, 1e-9);
            assert_close(
                g.st_mincut_value(0, t, Some(&capacity)).unwrap(),
                mf.value,
                1e-9,
            );
            // Menger with unit capacities.
            assert_eq!(
                g.maxflow_value(0, t, None).unwrap(),
                g.edge_disjoint_paths(0, t).unwrap() as f64
            );
        }

        // Whitney: kappa <= lambda <= minimum degree (out/in for digraphs).
        let kappa = g.vertex_connectivity(false).unwrap();
        let lambda = g.edge_connectivity(false).unwrap();
        assert!(kappa <= lambda);
        for mode in [NeighborMode::Out, NeighborMode::In] {
            let deg = g.degree(.., mode, Loops::Twice).unwrap();
            assert!(lambda as i64 <= *deg.iter().min().unwrap());
        }
        assert_eq!(lambda as f64, g.mincut_value(None).unwrap());

        // Gomory-Hu (undirected only) reproduces every pairwise flow.
        if !directed {
            let gh = g.gomory_hu_tree(Some(&capacity)).unwrap();
            for u in 0..12 {
                for v in u + 1..12 {
                    assert_close(
                        gh.flow_between(u, v).unwrap(),
                        g.maxflow_value(u, v, Some(&capacity)).unwrap(),
                        1e-9,
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Use case
// ---------------------------------------------------------------------------

/// A city's water network: a reservoir, pumping stations and a district.
/// How much water per second can reach the district, which pipes are the
/// bottleneck, and what happens if the city upgrades them?
#[test]
fn use_case_water_supply_network() {
    const RESERVOIR: i64 = 0;
    const DISTRICT: i64 = 5;
    // Pipes (from, to) with capacities in litres per second.
    let pipes = [
        ((RESERVOIR, 1), 16.0), // 0: main north
        ((RESERVOIR, 2), 13.0), // 1: main south
        ((1, 2), 10.0),         // 2
        ((2, 1), 4.0),          // 3
        ((1, 3), 12.0),         // 4
        ((3, 2), 9.0),          // 5
        ((2, 4), 14.0),         // 6
        ((4, 3), 7.0),          // 7
        ((3, DISTRICT), 20.0),  // 8
        ((4, DISTRICT), 4.0),   // 9
    ];
    let edges: Vec<(i64, i64)> = pipes.iter().map(|p| p.0).collect();
    let mut capacity: Vec<f64> = pipes.iter().map(|p| p.1).collect();
    let net = Graph::from_edges(&edges, 6, true).unwrap();

    // This is the classic CLRS network: at most 23 l/s reach the district.
    let mf = net.maxflow(RESERVOIR, DISTRICT, Some(&capacity)).unwrap();
    assert_eq!(mf.value, 23.0);

    // The bottleneck pipes: a minimum cut whose capacity equals the flow.
    let bottleneck = net.st_mincut(RESERVOIR, DISTRICT, Some(&capacity)).unwrap();
    assert_eq!(bottleneck.value, 23.0);
    let pipes_in_cut = net.edges(&bottleneck.cut).unwrap();
    assert_eq!(pipes_in_cut, vec![(1, 3), (4, 3), (4, 5)]);
    // Every bottleneck pipe runs at full capacity.
    for &e in &bottleneck.cut {
        assert_eq!(mf.flow[e as usize], capacity[e as usize]);
    }
    // Water is conserved at every pumping station.
    let balance = net_outflow(&net, &mf.flow);
    assert_eq!(balance, vec![23.0, 0.0, 0.0, 0.0, 0.0, -23.0]);

    // Upgrading a pipe that is NOT in the cut does not help at all...
    capacity[0] += 100.0;
    assert_eq!(
        net.maxflow_value(RESERVOIR, DISTRICT, Some(&capacity))
            .unwrap(),
        23.0
    );
    capacity[0] -= 100.0;

    // ...while doubling the pipe 4 -> district (in the cut) does.
    capacity[9] *= 2.0;
    let upgraded = net
        .maxflow_value(RESERVOIR, DISTRICT, Some(&capacity))
        .unwrap();
    assert!(upgraded > 23.0);
    // Not by the full 4 l/s, though: now the cut {1 -> 3, 2 -> 4} of
    // capacity 12 + 14 = 26 becomes the new bottleneck.
    assert_eq!(upgraded, 26.0);
    let new_cut = net.st_mincut(RESERVOIR, DISTRICT, Some(&capacity)).unwrap();
    assert_eq!(net.edges(&new_cut.cut).unwrap(), vec![(1, 3), (2, 4)]);

    // Resilience: how many pipes must burst to cut the district off?
    assert_eq!(net.st_edge_connectivity(RESERVOIR, DISTRICT).unwrap(), 2);
    assert_eq!(net.edge_disjoint_paths(RESERVOIR, DISTRICT).unwrap(), 2);
    // And how many pumping stations must fail?
    assert_eq!(
        net.st_vertex_connectivity(RESERVOIR, DISTRICT, VconnNei::Error)
            .unwrap(),
        2
    );

    // Engineers want the alternative bottlenecks too (unit capacities: which
    // pairs of pipes are critical?).
    let critical = net.all_st_mincuts(RESERVOIR, DISTRICT, None).unwrap();
    assert_eq!(critical.value, 2.0);
    assert!(critical.cuts.contains(&vec![0, 1]));
    assert!(critical.cuts.iter().any(|c| c == &vec![8, 9]));
}

//! Integration tests for the `cycles` module (igraph_cycles.h, igraph_eulerian.h).

mod common;

use common::{complete, cycle, karate, path};
use igraph::{
    cycles::{Cycle, EulerianStatus, EulerianWalk, MinimumCycleBasisOptions, SimpleCyclesOptions},
    prelude::*,
};
use std::{collections::HashSet, ops::ControlFlow};

// ---------------------------------------------------------------- helpers

/// Directed graph from an edge list.
fn digraph(edges: &[(i64, i64)], n: usize) -> Graph {
    Graph::from_edges(edges, n, true).unwrap()
}

/// Undirected graph from an edge list.
fn ugraph(edges: &[(i64, i64)], n: usize) -> Graph {
    Graph::from_edges(edges, n, false).unwrap()
}

/// Checks that `c` is a closed walk of `g` following the `Cycle` layout:
/// `edges[i]` connects `vertices[i]` to `vertices[i + 1]` (cyclically), in the
/// direction dictated by `mode`, and no vertex nor edge is repeated.
fn assert_valid_cycle(g: &Graph, c: &Cycle, mode: NeighborMode) {
    let n = c.vertices.len();
    assert_eq!(n, c.edges.len(), "{c:?}");
    assert!(n > 0);
    for i in 0..n {
        let (from, to) = g.edge(c.edges[i]).unwrap();
        let (a, b) = (c.vertices[i], c.vertices[(i + 1) % n]);
        let ok = match (g.is_directed(), mode) {
            (true, NeighborMode::Out) => (from, to) == (a, b),
            (true, NeighborMode::In) => (from, to) == (b, a),
            _ => (from, to) == (a, b) || (from, to) == (b, a),
        };
        assert!(
            ok,
            "edge {} = {from}->{to} does not join {a} and {b} in {c:?}",
            c.edges[i]
        );
    }
    assert_eq!(
        c.vertices.iter().collect::<HashSet<_>>().len(),
        n,
        "repeated vertex {c:?}"
    );
    assert_eq!(
        c.edges.iter().collect::<HashSet<_>>().len(),
        n,
        "repeated edge {c:?}"
    );
}

/// Checks that `w` traverses every edge of `g` exactly once, consistently.
fn assert_valid_eulerian(g: &Graph, w: &EulerianWalk, closed: bool) {
    assert_eq!(w.edges.len(), g.ecount());
    let mut sorted = w.edges.clone();
    sorted.sort();
    assert_eq!(sorted, (0..g.ecount() as i64).collect::<Vec<_>>());
    if w.edges.is_empty() {
        return;
    }
    assert_eq!(w.vertices.len(), w.edges.len() + 1);
    for (i, &e) in w.edges.iter().enumerate() {
        let (from, to) = g.edge(e).unwrap();
        let (a, b) = (w.vertices[i], w.vertices[i + 1]);
        if g.is_directed() {
            assert_eq!((from, to), (a, b));
        } else {
            assert!((from, to) == (a, b) || (from, to) == (b, a));
        }
    }
    if closed {
        assert_eq!(w.vertices.first(), w.vertices.last());
    }
}

/// An edge set forms a union of cycles iff every vertex has even degree in it
/// (i.e. it's an element of the cycle space over GF(2)).
fn is_cycle_space_element(g: &Graph, edges: &[i64]) -> bool {
    let mut deg = vec![0usize; g.vcount()];
    for &e in edges {
        let (a, b) = g.edge(e).unwrap();
        deg[a as usize] += 1;
        deg[b as usize] += 1;
    }
    deg.iter().all(|d| d % 2 == 0)
}

/// Rank over GF(2) of a set of edge sets: a cycle basis must have full rank.
fn gf2_rank(sets: &[Vec<i64>], m: usize) -> usize {
    let mut rows: Vec<Vec<bool>> = sets
        .iter()
        .map(|s| {
            let mut r = vec![false; m];
            for &e in s {
                r[e as usize] ^= true;
            }
            r
        })
        .collect();
    let mut rank = 0;
    for col in 0..m {
        if let Some(p) = (rank..rows.len()).find(|&r| rows[r][col]) {
            rows.swap(rank, p);
            for r in 0..rows.len() {
                if r != rank && rows[r][col] {
                    let pivot = rows[rank].clone();
                    rows[r].iter_mut().zip(pivot).for_each(|(x, y)| *x ^= y);
                }
            }
            rank += 1;
        }
    }
    rank
}

/// Is the graph acyclic (a forest if undirected, a DAG if directed)? Checked
/// both with `find_cycle` and with `is_acyclic` from the structural module.
fn acyclic(g: &Graph) -> bool {
    let res = g.find_cycle(NeighborMode::Out).unwrap().is_none();
    assert_eq!(res, g.is_acyclic().unwrap());
    res
}

/// The undirected wheel graph with `n` vertices: hub 0 and a rim cycle 1..n-1.
fn wheel(n: usize) -> Graph {
    Graph::wheel(n, WheelMode::Undirected, 0).unwrap()
}

/// The Petersen graph.
fn petersen() -> Graph {
    Graph::famous("Petersen").unwrap()
}

/// Dimension of the cycle space: |E| - |V| + number of connected components.
fn cyclomatic_number(g: &Graph) -> usize {
    let c = g.connected_components(Connectedness::Weak).unwrap().count;
    g.ecount() + c - g.vcount()
}

fn binomial(n: u64, k: u64) -> u64 {
    (0..k).fold(1, |acc, i| acc * (n - i) / (i + 1))
}

fn factorial(n: u64) -> u64 {
    (1..=n).product()
}

// ---------------------------------------------------------------- is_dag / topological sorting

#[test]
fn is_dag_basic_cases() {
    assert!(Graph::new(0, true).is_dag().unwrap());
    assert!(Graph::new(5, true).is_dag().unwrap());
    assert!(digraph(&[(0, 1), (1, 2), (0, 2)], 3).is_dag().unwrap());
    assert!(!digraph(&[(0, 1), (1, 2), (2, 0)], 3).is_dag().unwrap());
    // A self-loop is a cycle.
    assert!(!digraph(&[(0, 1), (1, 1)], 2).is_dag().unwrap());
    // Undirected graphs are never DAGs.
    assert!(!path(4).is_dag().unwrap());
    assert!(!Graph::new(3, false).is_dag().unwrap());
}

#[test]
fn is_dag_cache_is_invalidated_on_modification() {
    let mut g = digraph(&[(0, 1), (1, 2)], 3);
    assert!(g.is_dag().unwrap());
    g.add_edge(2, 0).unwrap();
    assert!(!g.is_dag().unwrap());
    g.delete_edges(2).unwrap();
    assert!(g.is_dag().unwrap());
}

#[test]
fn topological_sorting_igraph_example() {
    // examples/simple/igraph_topological_sorting.c and its .out file.
    let g = digraph(
        &[
            (0, 3),
            (0, 4),
            (1, 3),
            (2, 4),
            (2, 7),
            (3, 5),
            (3, 6),
            (3, 7),
            (4, 6),
        ],
        8,
    );
    assert_eq!(
        g.topological_sorting(NeighborMode::Out).unwrap(),
        vec![0, 1, 2, 3, 4, 5, 7, 6]
    );
    assert_eq!(
        g.topological_sorting(NeighborMode::In).unwrap(),
        vec![5, 6, 7, 4, 3, 2, 0, 1]
    );
}

#[test]
fn topological_sorting_respects_every_edge() {
    // A "full citation" DAG with a pseudo-random subset of edges i -> j, i < j,
    // relabelled by a permutation so that the identity is not a valid order.
    let n = 30i64;
    let perm: Vec<i64> = (0..n).map(|i| (i * 7 + 3) % n).collect();
    let mut edges = vec![];
    for i in 0..n {
        for j in i + 1..n {
            if (i * 31 + j * 17) % 5 < 2 {
                edges.push((perm[i as usize], perm[j as usize]));
            }
        }
    }
    let g = digraph(&edges, n as usize);
    assert!(g.is_dag().unwrap());
    for mode in [NeighborMode::Out, NeighborMode::In] {
        let order = g.topological_sorting(mode).unwrap();
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>(), "must be a permutation");
        let mut pos = vec![0; n as usize];
        for (p, &v) in order.iter().enumerate() {
            pos[v as usize] = p;
        }
        for &(a, b) in &edges {
            match mode {
                NeighborMode::Out => assert!(pos[a as usize] < pos[b as usize]),
                _ => assert!(pos[a as usize] > pos[b as usize]),
            }
        }
    }
}

#[test]
fn topological_sorting_errors_and_self_loops() {
    let cyclic = digraph(&[(0, 1), (1, 2), (2, 0), (2, 3)], 4);
    let err = cyclic.topological_sorting(NeighborMode::Out).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    // Self-loops are ignored by the topological sort (but not by is_dag).
    let looped = digraph(&[(0, 1), (1, 1), (1, 2)], 3);
    assert!(!looped.is_dag().unwrap());
    assert_eq!(
        looped.topological_sorting(NeighborMode::Out).unwrap(),
        vec![0, 1, 2]
    );
    // Undirected graphs and NeighborMode::All are rejected.
    for (g, mode) in [
        (Graph::new(3, false), NeighborMode::Out),
        (digraph(&[(0, 1)], 2), NeighborMode::All),
    ] {
        let err = g.topological_sorting(mode).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidValue);
    }
}

// ---------------------------------------------------------------- find_cycle

#[test]
fn find_cycle_matches_igraph_unit_tests() {
    // tests/unit/igraph_find_cycle.c, "Small directed graph" (vertices rotated
    // to the `Cycle` layout).
    let g = digraph(&[(0, 1), (1, 2), (2, 3), (3, 4), (2, 0)], 5);
    let out = g.find_cycle(NeighborMode::Out).unwrap().unwrap();
    assert_eq!(
        out,
        Cycle {
            vertices: vec![0, 1, 2],
            edges: vec![0, 1, 4]
        }
    );
    for mode in [NeighborMode::Out, NeighborMode::In, NeighborMode::All] {
        let c = g.find_cycle(mode).unwrap().unwrap();
        assert_eq!(c.len(), 3);
        assert_valid_cycle(&g, &c, mode);
    }

    // "Isolated vertices with self-loops": a self-loop is a 1-cycle.
    let loops = ugraph(&[(1, 1), (2, 2)], 3);
    let c = loops.find_cycle(NeighborMode::All).unwrap().unwrap();
    assert_eq!(
        c,
        Cycle {
            vertices: vec![1],
            edges: vec![0]
        }
    );

    // "Small undirected multigraph": parallel edges form a 2-cycle.
    let multi = ugraph(&[(1, 2), (3, 4), (3, 4), (3, 4)], 5);
    let c = multi.find_cycle(NeighborMode::All).unwrap().unwrap();
    assert_eq!(c.len(), 2);
    assert_eq!(c.edges, vec![1, 2]);
    assert_valid_cycle(&multi, &c, NeighborMode::All);
}

#[test]
fn find_cycle_on_acyclic_graphs() {
    assert_eq!(
        Graph::new(0, false).find_cycle(NeighborMode::All).unwrap(),
        None
    );
    assert_eq!(path(10).find_cycle(NeighborMode::All).unwrap(), None);
    // A DAG has no directed cycle, but its underlying graph may have one.
    let dag = digraph(&[(0, 1), (0, 2), (1, 3), (2, 3)], 4);
    assert_eq!(dag.find_cycle(NeighborMode::Out).unwrap(), None);
    assert_eq!(dag.find_cycle(NeighborMode::In).unwrap(), None);
    let c = dag.find_cycle(NeighborMode::All).unwrap().unwrap();
    assert_eq!(c.len(), 4);
    assert_valid_cycle(&dag, &c, NeighborMode::All);
}

#[test]
fn find_cycle_agrees_with_is_dag() {
    for n in 2..8 {
        let ring: Vec<(i64, i64)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
        let g = digraph(&ring, n as usize);
        assert_eq!(
            g.is_dag().unwrap(),
            g.find_cycle(NeighborMode::Out).unwrap().is_none()
        );
        let c = g.find_cycle(NeighborMode::Out).unwrap().unwrap();
        assert_eq!(c.len(), n as usize);
        assert_valid_cycle(&g, &c, NeighborMode::Out);
    }
}

// ---------------------------------------------------------------- simple cycles

fn count_cycles(g: &Graph, opts: &SimpleCyclesOptions) -> usize {
    let cycles = g.simple_cycles(opts).unwrap();
    for c in &cycles {
        assert_valid_cycle(g, c, opts.mode);
        if let Some(min) = opts.min_cycle_length {
            assert!(c.len() >= min);
        }
        if let Some(max) = opts.max_cycle_length {
            assert!(c.len() <= max);
        }
    }
    // Each cycle is reported once: edge sets are pairwise distinct.
    let distinct: HashSet<Vec<i64>> = cycles
        .iter()
        .map(|c| {
            let mut e = c.edges.clone();
            e.sort();
            e
        })
        .collect();
    assert_eq!(distinct.len(), cycles.len());
    cycles.len()
}

#[test]
fn simple_cycles_counts_from_igraph_unit_tests() {
    let all = SimpleCyclesOptions::default();
    // Null and edgeless graphs.
    assert_eq!(count_cycles(&Graph::new(0, true), &all), 0);
    assert_eq!(count_cycles(&Graph::new(5, false), &all), 0);
    // Directed 10-cycle: one cycle.
    let ring: Vec<(i64, i64)> = (0..10).map(|i| (i, (i + 1) % 10)).collect();
    let dring = digraph(&ring, 10);
    let cycles = dring.simple_cycles(&all).unwrap();
    assert_eq!(cycles.len(), 1);
    assert_eq!(cycles[0].vertices, (0..10).collect::<Vec<_>>());
    assert_eq!(cycles[0].edges, (0..10).collect::<Vec<_>>());
    // Undirected ring: still one cycle (both orientations count once).
    assert_eq!(count_cycles(&cycle(10), &all), 1);
    // Complete DAG on 5 vertices: acyclic, but 37 cycles when ignoring directions.
    let mut citation = vec![];
    for i in 0..5 {
        for j in 0..i {
            citation.push((i, j));
        }
    }
    let dag = digraph(&citation, 5);
    assert_eq!(count_cycles(&dag, &all), 0);
    assert_eq!(count_cycles(&dag, &all.with_mode(NeighborMode::All)), 37);
    // Undirected wheel with 10 vertices: 73 cycles, 9 triangles, 18 of length <= 4.
    let w = wheel(10);
    assert_eq!(count_cycles(&w, &all), 73);
    assert_eq!(count_cycles(&w, &all.with_max_length(3)), 9);
    assert_eq!(count_cycles(&w, &all.with_max_length(4)), 18);
    // The "envelope": 4 triangles + 5 squares + 4 pentagons.
    let envelope = ugraph(
        &[
            (0, 1),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 4),
        ],
        5,
    );
    assert_eq!(count_cycles(&envelope, &all), 13);
    assert_eq!(count_cycles(&envelope, &all.with_max_length(3)), 4);
    assert_eq!(count_cycles(&envelope, &all.with_min_length(5)), 4);
    assert_eq!(
        count_cycles(&envelope, &all.with_min_length(4).with_max_length(4)),
        5
    );
    // The "boat" and the "house".
    let boat = ugraph(&[(0, 2), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4)], 5);
    assert_eq!(count_cycles(&boat, &all), 6);
    let house = ugraph(&[(0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3)], 5);
    assert_eq!(count_cycles(&house, &all), 3);
}

#[test]
fn simple_cycles_in_complete_graphs_match_the_formula() {
    // K_n has sum_{k=3}^{n} C(n, k) (k-1)! / 2 simple cycles.
    for n in 3..=6u64 {
        let expected: u64 = (3..=n).map(|k| binomial(n, k) * factorial(k - 1) / 2).sum();
        let g = complete(n as i64);
        assert_eq!(
            count_cycles(&g, &SimpleCyclesOptions::default()) as u64,
            expected,
            "K{n}"
        );
        // And exactly C(n, k)(k-1)!/2 of each length k.
        for k in 3..=n {
            let opts = SimpleCyclesOptions::default()
                .with_min_length(k as usize)
                .with_max_length(k as usize);
            assert_eq!(
                count_cycles(&g, &opts) as u64,
                binomial(n, k) * factorial(k - 1) / 2
            );
        }
    }
}

#[test]
fn simple_cycles_with_loops_and_multi_edges() {
    let all = SimpleCyclesOptions::default();
    assert_eq!(count_cycles(&digraph(&[(0, 0)], 1), &all), 1);
    assert_eq!(count_cycles(&ugraph(&[(0, 1), (0, 1)], 2), &all), 1);
    assert_eq!(count_cycles(&ugraph(&[(0, 1), (0, 1), (0, 1)], 2), &all), 3);
    assert_eq!(count_cycles(&ugraph(&[(0, 1), (0, 1), (0, 0)], 2), &all), 2);
    // Mutual directed edges form a 2-cycle.
    assert_eq!(count_cycles(&digraph(&[(0, 1), (1, 0)], 2), &all), 1);
    // Parallel arcs 0 -> 1 do not, unless directions are ignored.
    let arcs = digraph(&[(0, 1), (0, 1)], 2);
    assert_eq!(count_cycles(&arcs, &all), 0);
    assert_eq!(count_cycles(&arcs, &all.with_mode(NeighborMode::All)), 1);
    assert_eq!(arcs.find_cycle(NeighborMode::Out).unwrap(), None);
    assert_eq!(
        arcs.find_cycle(NeighborMode::All).unwrap().unwrap().len(),
        2
    );
}

#[test]
fn simple_cycles_max_results_and_zero_limits() {
    let g = complete(6);
    let opts = SimpleCyclesOptions::default().with_max_results(5);
    assert_eq!(g.simple_cycles(&opts).unwrap().len(), 5);
    assert!(
        g.simple_cycles(&SimpleCyclesOptions::default().with_max_results(0))
            .unwrap()
            .is_empty()
    );
    assert!(
        g.simple_cycles(&SimpleCyclesOptions::default().with_max_length(0))
            .unwrap()
            .is_empty()
    );
    // No cycle is shorter than 3 in a simple graph.
    assert!(
        g.simple_cycles(&SimpleCyclesOptions::default().with_max_length(2))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn simple_cycles_callback_agrees_with_simple_cycles() {
    let g = karate();
    let opts = SimpleCyclesOptions::default().with_max_length(4);
    let stored = g.simple_cycles(&opts).unwrap();
    let mut streamed = vec![];
    g.simple_cycles_callback(&opts, |vs, es| {
        streamed.push(Cycle {
            vertices: vs.to_vec(),
            edges: es.to_vec(),
        });
        ControlFlow::Continue(())
    })
    .unwrap();
    assert_eq!(stored, streamed);
    assert!(!stored.is_empty());
    // Karate has 45 triangles.
    let triangles = g.simple_cycles(&opts.with_max_length(3)).unwrap();
    assert_eq!(triangles.len(), 45);
}

#[test]
fn simple_cycles_callback_early_stop_and_max_results() {
    let g = complete(7);
    let mut seen = 0;
    g.simple_cycles_callback(&SimpleCyclesOptions::default(), |_, _| {
        seen += 1;
        if seen == 10 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .unwrap();
    assert_eq!(seen, 10);

    let mut seen = 0;
    g.simple_cycles_callback(
        &SimpleCyclesOptions::default().with_max_results(3),
        |_, _| {
            seen += 1;
            ControlFlow::Continue(())
        },
    )
    .unwrap();
    assert_eq!(seen, 3);

    let mut called = false;
    g.simple_cycles_callback(
        &SimpleCyclesOptions::default().with_max_results(0),
        |_, _| {
            called = true;
            ControlFlow::Continue(())
        },
    )
    .unwrap();
    assert!(!called);
}

#[test]
fn simple_cycles_callback_propagates_panics() {
    let g = complete(5);
    let result = std::panic::catch_unwind(|| {
        g.simple_cycles_callback(&SimpleCyclesOptions::default(), |_, _| -> ControlFlow<()> {
            panic!("boom")
        })
    });
    let payload = result.unwrap_err();
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"boom"));
    // igraph is still usable on this thread afterwards.
    assert_eq!(
        g.simple_cycles(&SimpleCyclesOptions::default())
            .unwrap()
            .len(),
        37
    );
}

// ---------------------------------------------------------------- cycle bases

fn disconnected_test_graph() -> Graph {
    // tests/unit/cycle_bases.c, "Disconnected".
    ugraph(
        &[
            (1, 2),
            (2, 3),
            (3, 1),
            (4, 5),
            (5, 4),
            (4, 5),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 6),
            (6, 8),
            (10, 10),
            (10, 11),
            (12, 12),
        ],
        0,
    )
}

#[test]
fn fundamental_cycles_match_igraph_unit_tests() {
    let g = disconnected_test_graph();
    let basis = g.fundamental_cycles(None, None).unwrap();
    assert_eq!(
        basis,
        vec![
            vec![1, 0, 2],
            vec![4, 5],
            vec![3, 5],
            vec![7, 6, 10],
            vec![8, 10, 9],
            vec![11],
            vec![13]
        ]
    );
    // Trivial cases.
    assert!(
        Graph::new(0, false)
            .fundamental_cycles(None, None)
            .unwrap()
            .is_empty()
    );
    assert!(path(5).fundamental_cycles(None, None).unwrap().is_empty());
    assert_eq!(
        ugraph(&[(0, 0)], 1).fundamental_cycles(None, None).unwrap(),
        vec![vec![0]]
    );
    assert_eq!(
        ugraph(&[(0, 1), (0, 1)], 2)
            .fundamental_cycles(None, None)
            .unwrap(),
        vec![vec![0, 1]]
    );
}

#[test]
fn fundamental_cycles_form_a_basis_of_the_cycle_space() {
    for g in [
        karate(),
        petersen(),
        complete(6),
        wheel(8),
        disconnected_test_graph(),
    ] {
        let basis = g.fundamental_cycles(None, None).unwrap();
        let dim = cyclomatic_number(&g);
        assert_eq!(basis.len(), dim);
        assert!(basis.iter().all(|cyc| is_cycle_space_element(&g, cyc)));
        assert_eq!(
            gf2_rank(&basis, g.ecount()),
            dim,
            "basis must be independent"
        );
    }
}

#[test]
fn fundamental_cycles_start_vertex_and_cutoff() {
    let g = disconnected_test_graph();
    // Only the component of vertex 6 (edges 6..=10): 5 edges, 4 vertices -> 2 cycles.
    let comp = g.fundamental_cycles(Some(6), None).unwrap();
    assert_eq!(comp.len(), 2);
    assert!(comp.iter().flatten().all(|&e| (6..=10).contains(&e)));
    // Invalid start vertex.
    let err = g.fundamental_cycles(Some(100), None).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    // With a BFS cutoff of 1 only cycles of length <= 3 are found.
    let c10 = cycle(10);
    assert!(c10.fundamental_cycles(None, Some(1)).unwrap().is_empty());
    assert_eq!(c10.fundamental_cycles(None, Some(5)).unwrap().len(), 1);
    let short = wheel(8).fundamental_cycles(None, Some(1)).unwrap();
    assert!(short.iter().all(|c| c.len() <= 3));
}

#[test]
fn minimum_cycle_basis_matches_igraph_unit_tests() {
    let g = disconnected_test_graph();
    let basis = g
        .minimum_cycle_basis(&MinimumCycleBasisOptions::default())
        .unwrap();
    assert_eq!(
        basis,
        vec![
            vec![11],
            vec![13],
            vec![3, 5],
            vec![4, 5],
            vec![0, 1, 2],
            vec![6, 7, 10],
            vec![8, 9, 10]
        ]
    );
    // Periodic 5x6 grid (a torus): 60 edges, 30 vertices -> 31 cycles, of
    // which 29 are squares, then one 5-cycle and one 6-cycle around the torus.
    let (rows, cols) = (5i64, 6i64);
    let mut edges = vec![];
    for r in 0..cols {
        for c in 0..rows {
            let v = r * rows + c;
            edges.push((v, r * rows + (c + 1) % rows));
            edges.push((v, ((r + 1) % cols) * rows + c));
        }
    }
    let torus = ugraph(&edges, 30);
    let basis = torus
        .minimum_cycle_basis(&MinimumCycleBasisOptions::default())
        .unwrap();
    let lengths: Vec<usize> = basis.iter().map(Vec::len).collect();
    let mut expected = vec![4; 29];
    expected.extend([5, 6]);
    assert_eq!(lengths, expected);
    assert_eq!(gf2_rank(&basis, torus.ecount()), 31);
}

#[test]
fn minimum_cycle_basis_is_no_longer_than_fundamental() {
    for g in [karate(), petersen(), wheel(9), complete(6)] {
        let fundamental = g.fundamental_cycles(None, None).unwrap();
        let minimum = g
            .minimum_cycle_basis(&MinimumCycleBasisOptions::default())
            .unwrap();
        assert_eq!(fundamental.len(), minimum.len());
        let total = |b: &Vec<Vec<i64>>| b.iter().map(Vec::len).sum::<usize>();
        assert!(total(&minimum) <= total(&fundamental));
        // Sorted by length, each a real cycle.
        assert!(minimum.windows(2).all(|w| w[0].len() <= w[1].len()));
        assert!(minimum.iter().all(|c| is_cycle_space_element(&g, c)));
        assert_eq!(gf2_rank(&minimum, g.ecount()), minimum.len());
    }
    // Petersen graph: girth 5, cycle space of dimension 6, all basis cycles are pentagons.
    let p = petersen()
        .minimum_cycle_basis(&MinimumCycleBasisOptions::default().with_cycle_order(false))
        .unwrap();
    assert_eq!(p.iter().map(Vec::len).collect::<Vec<_>>(), vec![5; 6]);
    // The wheel's minimum basis consists of its n-1 triangles.
    let w = wheel(9)
        .minimum_cycle_basis(&MinimumCycleBasisOptions::default())
        .unwrap();
    assert_eq!(w.iter().map(Vec::len).collect::<Vec<_>>(), vec![3; 8]);
}

#[test]
fn minimum_cycle_basis_with_cutoff() {
    // Two disjoint cycles, a triangle and a 9-cycle.
    let mut edges = vec![(0, 1), (1, 2), (2, 0)];
    edges.extend((0..9).map(|i| (3 + i, 3 + (i + 1) % 9)));
    let g = ugraph(&edges, 12);
    // Incomplete: only the short cycles (length <= 2*1 + 1).
    let partial = g
        .minimum_cycle_basis(
            &MinimumCycleBasisOptions::default()
                .with_bfs_cutoff(1)
                .with_complete(false),
        )
        .unwrap();
    assert_eq!(partial, vec![vec![0, 1, 2]]);
    // Complete: the long cycle is added too.
    let complete_basis = g
        .minimum_cycle_basis(&MinimumCycleBasisOptions::default().with_bfs_cutoff(1))
        .unwrap();
    assert_eq!(complete_basis.len(), 2);
    assert_eq!(complete_basis[1].len(), 9);
}

// ---------------------------------------------------------------- feedback sets

/// The graph of examples/simple/igraph_feedback_arc_set*.c.
fn fas_example(with_loops: bool) -> Graph {
    let mut edges = vec![
        (0, 1),
        (1, 2),
        (2, 0),
        (2, 3),
        (2, 4),
        (0, 4),
        (4, 3),
        (5, 0),
        (6, 5),
    ];
    if with_loops {
        edges.extend([(1, 1), (4, 4)]);
    }
    digraph(&edges, 7)
}

#[test]
fn feedback_arc_set_matches_igraph_examples() {
    // Eades heuristic (igraph_feedback_arc_set.out).
    let g = fas_example(false);
    assert_eq!(
        g.feedback_arc_set(None, FasAlgorithm::ApproxEades).unwrap(),
        vec![2]
    );
    let w = [1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    assert_eq!(
        g.feedback_arc_set(Some(&w), FasAlgorithm::ApproxEades)
            .unwrap(),
        vec![1]
    );
    let looped = fas_example(true);
    assert_eq!(
        looped
            .feedback_arc_set(None, FasAlgorithm::ApproxEades)
            .unwrap(),
        vec![2, 9, 10]
    );

    // Exact integer programming (igraph_feedback_arc_set_ip.out).
    assert_eq!(
        g.feedback_arc_set(None, FasAlgorithm::ExactIp).unwrap(),
        vec![0]
    );
    assert_eq!(
        g.feedback_arc_set(Some(&w), FasAlgorithm::ExactIp).unwrap(),
        vec![0]
    );
    assert_eq!(
        looped
            .feedback_arc_set(None, FasAlgorithm::ExactIp)
            .unwrap(),
        vec![0, 9, 10]
    );

    // Null and singleton graphs.
    for n in [0, 1] {
        for algo in [FasAlgorithm::ApproxEades, FasAlgorithm::ExactIp] {
            assert!(
                Graph::new(n, true)
                    .feedback_arc_set(None, algo)
                    .unwrap()
                    .is_empty()
            );
        }
    }
}

#[test]
fn exact_feedback_arc_set_methods_agree() {
    // A tournament-like graph with many cycles: i -> j if (j - i) mod 7 in {1, 2, 4}.
    let mut edges = vec![];
    for i in 0..7i64 {
        for d in [1, 2, 4] {
            edges.push((i, (i + d) % 7));
        }
    }
    let g = digraph(&edges, 7);
    let sizes: Vec<usize> = [
        FasAlgorithm::ExactIp,
        FasAlgorithm::ExactIpCg,
        FasAlgorithm::ExactIpTi,
    ]
    .into_iter()
    .map(|algo| {
        let fas = g.feedback_arc_set(None, algo).unwrap();
        let mut h = g.clone();
        h.delete_edges(&fas).unwrap();
        assert!(h.is_dag().unwrap(), "{algo:?}");
        fas.len()
    })
    .collect();
    assert!(sizes.windows(2).all(|w| w[0] == w[1]), "{sizes:?}");
    // The heuristic is never better than the optimum, and within its bound.
    let approx = g.feedback_arc_set(None, FasAlgorithm::ApproxEades).unwrap();
    assert!(approx.len() >= sizes[0]);
    assert!((approx.len() as f64) < g.ecount() as f64 / 2.0 - g.vcount() as f64 / 6.0);
}

#[test]
fn feedback_arc_set_undirected_is_complement_of_spanning_forest() {
    let g = karate();
    let fas = g.feedback_arc_set(None, FasAlgorithm::ExactIp).unwrap();
    assert_eq!(fas.len(), 78 - 34 + 1);
    let mut h = g.clone();
    h.delete_edges(&fas).unwrap();
    assert!(acyclic(&h));
    // Weighted: in a triangle with one heavy edge, a light edge is removed.
    let t = cycle(3);
    let fas = t
        .feedback_arc_set(Some(&[5.0, 1.0, 5.0]), FasAlgorithm::ApproxEades)
        .unwrap();
    assert_eq!(fas, vec![1]);
}

#[test]
fn feedback_arc_set_rejects_bad_weights() {
    let g = fas_example(false);
    let err = g
        .feedback_arc_set(Some(&[1.0, 2.0]), FasAlgorithm::ExactIp)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

#[test]
fn feedback_vertex_set_examples() {
    let exact = FvsAlgorithm::ExactIp;
    // Bowtie: the shared vertex.
    let bowtie = ugraph(&[(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)], 5);
    assert_eq!(bowtie.feedback_vertex_set(None, exact).unwrap(), vec![0]);
    // Forests and DAGs need nothing.
    assert!(path(6).feedback_vertex_set(None, exact).unwrap().is_empty());
    assert!(
        digraph(&[(0, 1), (1, 2), (0, 2)], 3)
            .feedback_vertex_set(None, exact)
            .unwrap()
            .is_empty()
    );
    // K_n needs n - 2 vertices removed.
    for n in 3..7 {
        assert_eq!(
            complete(n).feedback_vertex_set(None, exact).unwrap().len(),
            (n - 2) as usize
        );
    }
    // The Petersen graph has decycling number 3.
    let p = petersen();
    let fvs = p.feedback_vertex_set(None, exact).unwrap();
    assert_eq!(fvs.len(), 3);
    let mut h = p.clone();
    h.delete_vertices(&fvs).unwrap();
    assert!(acyclic(&h));
    // Self-loops force their vertex in.
    let looped = digraph(&[(0, 0), (1, 2)], 3);
    assert_eq!(looped.feedback_vertex_set(None, exact).unwrap(), vec![0]);
}

#[test]
fn feedback_vertex_set_weighted_and_errors() {
    // Directed 5-cycle: removing any vertex works; the lightest one is chosen.
    let ring: Vec<(i64, i64)> = (0..5).map(|i| (i, (i + 1) % 5)).collect();
    let g = digraph(&ring, 5);
    let w = [3.0, 2.0, 5.0, 0.5, 4.0];
    assert_eq!(
        g.feedback_vertex_set(Some(&w), FvsAlgorithm::ExactIp)
            .unwrap(),
        vec![3]
    );
    let err = g
        .feedback_vertex_set(Some(&[1.0]), FvsAlgorithm::ExactIp)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

// ---------------------------------------------------------------- Eulerian paths and cycles

#[test]
fn eulerian_paths_match_igraph_unit_tests() {
    // tests/unit/igraph_eulerian_path.c: empty, single edge, 2-path, triangle.
    let empty = Graph::new(0, false);
    assert_eq!(empty.eulerian_path().unwrap(), EulerianWalk::default());
    let edge = ugraph(&[(0, 1)], 2);
    assert_eq!(
        edge.eulerian_path().unwrap(),
        EulerianWalk {
            vertices: vec![0, 1],
            edges: vec![0]
        }
    );
    let p = ugraph(&[(0, 1), (1, 2)], 3);
    assert_eq!(
        p.eulerian_path().unwrap(),
        EulerianWalk {
            vertices: vec![0, 1, 2],
            edges: vec![0, 1]
        }
    );
    let t = ugraph(&[(0, 1), (1, 2), (2, 0)], 3);
    assert_eq!(
        t.eulerian_path().unwrap(),
        EulerianWalk {
            vertices: vec![0, 1, 2, 0],
            edges: vec![0, 1, 2]
        }
    );
    // Multigraph 0-1, 1-2, 2-1, 1-0.
    let m = ugraph(&[(0, 1), (1, 2), (2, 1), (1, 0)], 3);
    let w = m.eulerian_path().unwrap();
    assert_valid_eulerian(&m, &w, false);
}

#[test]
fn is_eulerian_classic_cases() {
    let both = EulerianStatus {
        has_path: true,
        has_cycle: true,
    };
    let path_only = EulerianStatus {
        has_path: true,
        has_cycle: false,
    };
    let neither = EulerianStatus {
        has_path: false,
        has_cycle: false,
    };
    assert_eq!(Graph::new(0, false).is_eulerian().unwrap(), both);
    assert_eq!(Graph::new(4, true).is_eulerian().unwrap(), both);
    assert_eq!(cycle(7).is_eulerian().unwrap(), both);
    assert_eq!(path(7).is_eulerian().unwrap(), path_only);
    // Star with three leaves: four odd vertices.
    assert_eq!(
        ugraph(&[(0, 1), (0, 2), (0, 3)], 4).is_eulerian().unwrap(),
        neither
    );
    // Two disjoint triangles: all degrees even but disconnected.
    let two = ugraph(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], 6);
    assert_eq!(two.is_eulerian().unwrap(), neither);
    // Isolated vertices don't matter.
    let with_isolated = ugraph(&[(0, 1), (1, 2), (2, 0)], 10);
    assert_eq!(with_isolated.is_eulerian().unwrap(), both);
    // K_n: cycle iff n is odd; K4 has four odd vertices.
    assert_eq!(complete(5).is_eulerian().unwrap(), both);
    assert_eq!(complete(4).is_eulerian().unwrap(), neither);
    // Directed: a directed path has a path but no cycle; balanced graphs have both.
    assert_eq!(
        digraph(&[(0, 1), (1, 2)], 3).is_eulerian().unwrap(),
        path_only
    );
    assert_eq!(
        digraph(&[(0, 1), (1, 0), (1, 2), (2, 1)], 3)
            .is_eulerian()
            .unwrap(),
        both
    );
    // Two sources: no path.
    assert_eq!(
        digraph(&[(0, 2), (1, 2)], 3).is_eulerian().unwrap(),
        neither
    );
}

#[test]
fn eulerian_walks_are_valid() {
    for (g, closed) in [
        (complete(5), true),
        (complete(7), true),
        (path(6), false),
        (cycle(5), true),
    ] {
        let status = g.is_eulerian().unwrap();
        assert!(status.has_path);
        let w = g.eulerian_path().unwrap();
        assert_valid_eulerian(&g, &w, false);
        // With all degrees even, any Eulerian path is closed.
        assert_eq!(w.vertices.first() == w.vertices.last(), closed);
        if closed {
            assert!(status.has_cycle);
            let c = g.eulerian_cycle().unwrap();
            assert_valid_eulerian(&g, &c, true);
        }
    }
    // Directed path must start at the vertex with out-degree excess.
    let g = digraph(&[(1, 2), (2, 0), (0, 1), (1, 3)], 4);
    let w = g.eulerian_path().unwrap();
    assert_valid_eulerian(&g, &w, false);
    assert_eq!(w.vertices.first(), Some(&1));
    assert_eq!(w.vertices.last(), Some(&3));
}

#[test]
fn eulerian_errors() {
    let star = ugraph(&[(0, 1), (0, 2), (0, 3)], 4);
    assert_eq!(
        star.eulerian_path().unwrap_err().kind(),
        ErrorKind::NoSolution
    );
    assert_eq!(
        star.eulerian_cycle().unwrap_err().kind(),
        ErrorKind::NoSolution
    );
    assert_eq!(
        path(3).eulerian_cycle().unwrap_err().kind(),
        ErrorKind::NoSolution
    );
    // Königsberg.
    let k = ugraph(&[(0, 1), (0, 1), (0, 2), (0, 2), (0, 3), (1, 3), (2, 3)], 4);
    assert_eq!(k.eulerian_path().unwrap_err().kind(), ErrorKind::NoSolution);
}

// ---------------------------------------------------------------- use cases

/// Kautz graphs (and de Bruijn graphs) are balanced and strongly connected,
/// hence Eulerian.
#[test]
fn kautz_graphs_are_eulerian() {
    for (m, n) in [(1usize, 1usize), (2, 1), (2, 3), (3, 2)] {
        let g = Graph::kautz(m, n).unwrap();
        assert!(g.is_eulerian().unwrap().has_cycle, "K({m}, {n})");
        let tour = g.eulerian_cycle().unwrap();
        assert_valid_eulerian(&g, &tour, true);
        // No self-loops, but plenty of cycles: never a DAG.
        assert!(!g.is_dag().unwrap());
    }
}

/// Use case: build a de Bruijn sequence of order `k` over an alphabet of `m`
/// symbols — a cyclic string of length m^k containing every k-letter word
/// exactly once — as an Eulerian cycle of the de Bruijn graph `B(m, k - 1)`
/// (vertices: (k-1)-letter words; edges: k-letter words, from their prefix to
/// their suffix). Such sequences are used to crack keypads that accept a code
/// as soon as its digits appear.
#[test]
fn use_case_de_bruijn_sequence() {
    for (m, k) in [(2i64, 2u32), (2, 5), (2, 8), (3, 3), (4, 4), (10, 3)] {
        let g = Graph::de_bruijn(m as usize, (k - 1) as usize).unwrap();
        assert_eq!(g.ecount(), m.pow(k) as usize);
        assert_eq!(
            g.is_eulerian().unwrap(),
            EulerianStatus {
                has_path: true,
                has_cycle: true
            }
        );
        let tour = g.eulerian_cycle().unwrap();
        assert_valid_eulerian(&g, &tour, true);
        // The sequence is the last letter of each visited vertex (in base m).
        let letters: Vec<i64> = tour.vertices[1..].iter().map(|v| v % m).collect();
        let n = letters.len();
        assert_eq!(n, m.pow(k) as usize);
        // Every k-letter word appears exactly once, cyclically.
        let words: HashSet<i64> = (0..n)
            .map(|start| (0..k as usize).fold(0, |acc, j| acc * m + letters[(start + j) % n]))
            .collect();
        assert_eq!(words.len(), n, "B({m}, {k})");
    }
}

/// Use case: a university curriculum. Courses with prerequisites must be
/// taken in a topological order; a data-entry mistake introduces circular
/// prerequisites, which we detect, report and repair with the smallest
/// number of dropped prerequisite links.
#[test]
fn use_case_curriculum_planning() {
    let courses = [
        "Calculus I",        // 0
        "Calculus II",       // 1
        "Linear Algebra",    // 2
        "Programming I",     // 3
        "Data Structures",   // 4
        "Algorithms",        // 5
        "Probability",       // 6
        "Machine Learning",  // 7
        "Numerical Methods", // 8
    ];
    let mut prereqs = vec![
        (0, 1),
        (0, 2),
        (3, 4),
        (4, 5),
        (1, 6),
        (2, 7),
        (6, 7),
        (5, 7),
        (1, 8),
        (2, 8),
        (3, 8),
    ];
    let g = digraph(&prereqs, courses.len());
    assert!(g.is_dag().unwrap());
    let plan = g.topological_sorting(NeighborMode::Out).unwrap();
    let pos = |c: i64| plan.iter().position(|&x| x == c).unwrap();
    for &(a, b) in &prereqs {
        assert!(
            pos(a) < pos(b),
            "{} must come before {}",
            courses[a as usize],
            courses[b as usize]
        );
    }
    // The capstone course comes last when reading prerequisites backwards.
    let backwards = g.topological_sorting(NeighborMode::In).unwrap();
    assert!(backwards[..2].contains(&7) || backwards[..2].contains(&8));

    // Oops: someone declared that "Calculus I" requires "Machine Learning",
    // and that "Programming I" requires "Algorithms".
    prereqs.extend([(7, 0), (5, 3)]);
    let mut g = digraph(&prereqs, courses.len());
    assert!(!g.is_dag().unwrap());
    assert_eq!(
        g.topological_sorting(NeighborMode::Out).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    let loop_found = g.find_cycle(NeighborMode::Out).unwrap().unwrap();
    assert_valid_cycle(&g, &loop_found, NeighborMode::Out);
    // All circular chains of requirements:
    let loops = g.simple_cycles(&SimpleCyclesOptions::default()).unwrap();
    assert_eq!(loops.len(), 3); // 0-1-6-7-0, 0-2-7-0 and 3-4-5-3
    // Minimum repair: drop two prerequisite links (the two bogus ones break
    // all loops, but any optimal solution works).
    let fas = g.feedback_arc_set(None, FasAlgorithm::ExactIp).unwrap();
    assert_eq!(fas.len(), 2);
    g.delete_edges(&fas).unwrap();
    assert!(g.is_dag().unwrap());
    assert_eq!(
        g.topological_sorting(NeighborMode::Out).unwrap().len(),
        courses.len()
    );
}

/// Use case: a mail carrier (or a snow plow) wants to traverse every street
/// exactly once. On a 3x3 grid of blocks this is impossible, but after
/// doubling the streets between the odd corners (a "Chinese postman" fix) an
/// Eulerian tour exists.
#[test]
fn use_case_street_sweeping() {
    // 3x3 grid of intersections, ids r*3+c.
    let mut streets = vec![];
    for r in 0..3i64 {
        for c in 0..3i64 {
            let v = r * 3 + c;
            if c < 2 {
                streets.push((v, v + 1));
            }
            if r < 2 {
                streets.push((v, v + 3));
            }
        }
    }
    let city = ugraph(&streets, 9);
    let status = city.is_eulerian().unwrap();
    assert!(!status.has_path, "the four side midpoints have odd degree");
    // Odd-degree intersections: 1, 3, 5, 7. Duplicate streets 1-4, 4-7 ... cheaper:
    // pair them up via the centre-free paths 1-0-3 and 5-8-7.
    let mut doubled = streets.clone();
    doubled.extend([(0, 1), (0, 3), (5, 8), (7, 8)]);
    let city = ugraph(&doubled, 9);
    assert!(city.is_eulerian().unwrap().has_cycle);
    let tour = city.eulerian_cycle().unwrap();
    assert_valid_eulerian(&city, &tour, true);
    assert_eq!(tour.edges.len(), streets.len() + 4);
}

// ---------------------------------------------------------------- extra edge cases (review)

/// Do the edges, in the given order, form a closed walk (each consecutive pair,
/// cyclically, shares an endpoint that the walk passes through)?
fn is_edge_ordered_cycle(g: &Graph, edges: &[i64]) -> bool {
    let n = edges.len();
    if n <= 1 {
        return true;
    }
    let (a, b) = g.edge(edges[0]).unwrap();
    // Try both orientations of the first edge.
    [(a, b), (b, a)].into_iter().any(|(start, mut cur)| {
        for &e in &edges[1..] {
            let (x, y) = g.edge(e).unwrap();
            cur = if x == cur {
                y
            } else if y == cur {
                x
            } else {
                return false;
            };
        }
        cur == start
    })
}

#[test]
fn cycle_bases_list_edges_in_cycle_order() {
    for g in [karate(), petersen(), wheel(7), disconnected_test_graph()] {
        for c in g.fundamental_cycles(None, None).unwrap() {
            assert!(is_edge_ordered_cycle(&g, &c), "fundamental {c:?}");
        }
        let opts = MinimumCycleBasisOptions::default();
        assert!(opts.use_cycle_order && opts.complete && opts.bfs_cutoff.is_none());
        for c in g.minimum_cycle_basis(&opts).unwrap() {
            assert!(is_edge_ordered_cycle(&g, &c), "minimum {c:?}");
        }
    }
}

#[test]
fn fundamental_cycles_rejects_negative_start() {
    let err = cycle(4).fundamental_cycles(Some(-1), None).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
}

#[test]
fn simple_cycles_in_mode_follows_edges_backwards() {
    // Directed triangle plus a 2-cycle: listing against the directions.
    let g = digraph(&[(0, 1), (1, 2), (2, 0), (2, 3), (3, 2)], 4);
    let opts = SimpleCyclesOptions::default().with_mode(NeighborMode::In);
    assert_eq!(count_cycles(&g, &opts), 2);
    // Ignoring directions, the edges 2->3 and 3->2 still form a 2-cycle.
    assert_eq!(count_cycles(&g, &opts.with_mode(NeighborMode::All)), 2);
}

#[test]
fn simple_cycles_huge_limits_mean_unlimited() {
    let g = complete(5);
    let opts = SimpleCyclesOptions::default()
        .with_min_length(0)
        .with_max_length(usize::MAX)
        .with_max_results(usize::MAX);
    assert_eq!(g.simple_cycles(&opts).unwrap().len(), 37);
    let mut n = 0;
    g.simple_cycles_callback(&opts, |_, _| {
        n += 1;
        ControlFlow::Continue(())
    })
    .unwrap();
    assert_eq!(n, 37);
    // The Cycle helpers.
    let c = &g.simple_cycles(&opts.with_max_results(1)).unwrap()[0];
    assert_eq!(c.len(), c.vertices.len());
    assert!(!c.is_empty());
    assert!(Cycle::default().is_empty());
}

#[test]
fn feedback_sets_reject_non_finite_weights() {
    let g = fas_example(false);
    let mut w = vec![1.0; g.ecount()];
    w[3] = f64::NAN;
    let err = g
        .feedback_arc_set(Some(&w), FasAlgorithm::ApproxEades)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let mut vw = vec![1.0; g.vcount()];
    vw[0] = f64::INFINITY;
    let err = g
        .feedback_vertex_set(Some(&vw), FvsAlgorithm::ExactIp)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

#[test]
fn eulerian_walks_with_self_loops() {
    // A single self-loop is an Eulerian cycle of length 1.
    for directed in [false, true] {
        let g = Graph::from_edges(&[(0, 0)], 1, directed).unwrap();
        assert!(g.is_eulerian().unwrap().has_cycle);
        let c = g.eulerian_cycle().unwrap();
        assert_eq!(
            c,
            EulerianWalk {
                vertices: vec![0, 0],
                edges: vec![0]
            }
        );
    }
    // A "lollipop": path 0-1 with a loop at 1. Degrees: 0 -> 1, 1 -> 3.
    let g = ugraph(&[(0, 1), (1, 1)], 2);
    let status = g.is_eulerian().unwrap();
    assert!(status.has_path && !status.has_cycle);
    let w = g.eulerian_path().unwrap();
    assert_valid_eulerian(&g, &w, false);
    assert_eq!(
        g.eulerian_cycle().unwrap_err().kind(),
        ErrorKind::NoSolution
    );
    // Two parallel arcs 0 -> 1: vertex 0 has out-degree excess 2, so even
    // though the underlying graph is connected there is no directed path.
    let g = digraph(&[(0, 1), (0, 1)], 2);
    assert_eq!(g.eulerian_path().unwrap_err().kind(), ErrorKind::NoSolution);
}

// ---------------------------------------------------------------- cross-module consistency

#[test]
fn is_dag_agrees_with_is_acyclic_on_random_digraphs() {
    rng::seed(2024).unwrap();
    let mut dags = 0;
    for m in [5, 8, 10, 12, 15, 20] {
        for _ in 0..10 {
            let g = Graph::erdos_renyi_game_gnm(10, m, true, EdgeTypeSw::Simple, false).unwrap();
            let is_dag = g.is_dag().unwrap();
            assert_eq!(is_dag, g.is_acyclic().unwrap());
            assert_eq!(is_dag, g.find_cycle(NeighborMode::Out).unwrap().is_none());
            assert_eq!(is_dag, g.topological_sorting(NeighborMode::Out).is_ok());
            // A DAG has no simple cycle, a non-DAG has at least one.
            let one = SimpleCyclesOptions::default().with_max_results(1);
            assert_eq!(is_dag, g.simple_cycles(&one).unwrap().is_empty());
            // Removing a minimum feedback arc set always yields a DAG.
            let fas = g.feedback_arc_set(None, FasAlgorithm::ExactIp).unwrap();
            assert_eq!(fas.is_empty(), is_dag);
            let mut h = g.clone();
            h.delete_edges(&fas).unwrap();
            assert!(h.is_dag().unwrap());
            dags += usize::from(is_dag);
        }
    }
    // Sparse graphs are mostly DAGs, dense ones mostly not: both cases occur.
    assert!(dags > 0 && dags < 60, "{dags}");
}

#[test]
fn cycles_agree_with_girth_and_triangles() {
    let karate = Graph::famous("Zachary").unwrap();
    assert!(karate.is_same_graph(&common::karate()).unwrap());
    // The triangles found by Johnson's algorithm are those of the isomorphism module.
    let triangles = karate
        .simple_cycles(&SimpleCyclesOptions::default().with_max_length(3))
        .unwrap();
    assert_eq!(triangles.len() as f64, karate.count_triangles().unwrap());
    let mut from_cycles: Vec<Vec<i64>> = triangles
        .iter()
        .map(|c| {
            let mut v = c.vertices.clone();
            v.sort();
            v
        })
        .collect();
    from_cycles.sort();
    let mut listed: Vec<Vec<i64>> = karate
        .list_triangles()
        .unwrap()
        .into_iter()
        .map(|t| {
            let mut v = t.to_vec();
            v.sort();
            v
        })
        .collect();
    listed.sort();
    assert_eq!(from_cycles, listed);

    // The shortest cycle of a simple graph: girth = first minimum-basis
    // length = shortest simple cycle, and find_cycle is never shorter.
    for (name, girth) in [
        ("Petersen", 5),
        ("Heawood", 6),
        ("Coxeter", 7),
        ("Zachary", 3),
        ("Dodecahedron", 5),
    ] {
        let g = Graph::famous(name).unwrap();
        assert_eq!(g.girth().unwrap(), Some(girth), "{name}");
        let basis = g
            .minimum_cycle_basis(&MinimumCycleBasisOptions::default())
            .unwrap();
        assert_eq!(basis[0].len(), girth, "{name}");
        assert_eq!(basis.len(), cyclomatic_number(&g), "{name}");
        let shortest = g
            .simple_cycles(&SimpleCyclesOptions::default().with_max_length(girth))
            .unwrap();
        assert!(!shortest.is_empty());
        assert!(shortest.iter().all(|c| c.len() == girth));
        assert!(
            g.simple_cycles(&SimpleCyclesOptions::default().with_max_length(girth - 1))
                .unwrap()
                .is_empty()
        );
        let c = g.find_cycle(NeighborMode::All).unwrap().unwrap();
        assert!(c.len() >= girth);
        assert_valid_cycle(&g, &c, NeighborMode::All);
    }
}

#[test]
fn undirected_feedback_arc_set_is_complement_of_maximum_spanning_forest() {
    rng::seed(7).unwrap();
    let g = Graph::erdos_renyi_game_gnm(20, 45, false, EdgeTypeSw::Simple, false).unwrap();
    let w: Vec<f64> = (0..g.ecount())
        .map(|_| rng::integer(1, 100) as f64)
        .collect();
    let fas = g.feedback_arc_set(Some(&w), FasAlgorithm::ExactIp).unwrap();
    assert_eq!(fas.len(), cyclomatic_number(&g));
    // A maximum weight spanning forest is a minimum one for 101 - w.
    let flipped: Vec<f64> = w.iter().map(|x| 101.0 - x).collect();
    let forest = g
        .minimum_spanning_tree(Some(&flipped), MstAlgorithm::Prim)
        .unwrap();
    let total = |edges: &[i64]| edges.iter().map(|&e| w[e as usize]).sum::<f64>();
    let all: f64 = w.iter().sum();
    assert_eq!(total(&fas), all - total(&forest));
    let mut h = g.clone();
    h.delete_edges(&fas).unwrap();
    assert!(h.is_forest(NeighborMode::All).unwrap());
    assert_eq!(
        h.connected_components(Connectedness::Weak).unwrap().count,
        g.connected_components(Connectedness::Weak).unwrap().count
    );
}

#[test]
fn is_eulerian_agrees_with_eulers_theorem_on_random_graphs() {
    rng::seed(99).unwrap();
    let (mut with_cycle, mut with_path) = (0, 0);
    for _ in 0..200 {
        let g = Graph::erdos_renyi_game_gnm(7, 9, false, EdgeTypeSw::Multi, false).unwrap();
        // Connectivity of the non-isolated part, via the components module.
        let comps = g.connected_components(Connectedness::Weak).unwrap();
        let deg = g.degree(.., NeighborMode::All, Loops::Twice).unwrap();
        let used: HashSet<i64> = (0..g.vcount())
            .filter(|&v| deg[v] > 0)
            .map(|v| comps.membership[v])
            .collect();
        let connected = used.len() <= 1;
        let odd = deg.iter().filter(|&&d| d % 2 == 1).count();
        let status = g.is_eulerian().unwrap();
        assert_eq!(
            status.has_cycle,
            connected && odd == 0,
            "{:?}",
            g.edge_list()
        );
        assert_eq!(
            status.has_path,
            connected && odd <= 2,
            "{:?}",
            g.edge_list()
        );
        if status.has_path {
            with_path += 1;
            let walk = g.eulerian_path().unwrap();
            assert_valid_eulerian(&g, &walk, false);
            if odd == 2 {
                // It must start and end at the two odd-degree vertices.
                let ends = [walk.vertices[0], *walk.vertices.last().unwrap()];
                assert!(ends.iter().all(|&v| deg[v as usize] % 2 == 1));
            }
        } else {
            assert_eq!(g.eulerian_path().unwrap_err().kind(), ErrorKind::NoSolution);
        }
        if status.has_cycle {
            with_cycle += 1;
            assert_valid_eulerian(&g, &g.eulerian_cycle().unwrap(), true);
        } else {
            assert_eq!(
                g.eulerian_cycle().unwrap_err().kind(),
                ErrorKind::NoSolution
            );
        }
    }
    assert!(with_path > 0 && with_cycle > 0 && with_path < 200);
}

#[test]
fn seeded_random_cycle_queries_are_reproducible_across_threads() {
    // Each thread has its own RNG: the same seed gives the same graph, hence
    // the same cycle structure, whatever runs in parallel.
    let run = || {
        rng::seed(123).unwrap();
        let g = Graph::erdos_renyi_game_gnm(12, 18, true, EdgeTypeSw::Simple, false).unwrap();
        let cycles = g.simple_cycles(&SimpleCyclesOptions::default()).unwrap();
        let fas = g.feedback_arc_set(None, FasAlgorithm::ExactIp).unwrap();
        (g.edge_list(), cycles, fas)
    };
    let results: Vec<_> = (0..4)
        .map(|_| std::thread::spawn(run))
        .collect::<Vec<_>>()
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    assert!(results.windows(2).all(|w| w[0] == w[1]));
    assert_eq!(results[0], run());
}

/// Regression test: an igraph call that fails inside the closure must not
/// free the temporaries of the running cycle search (igraph's error handler
/// unwinds the current level of the "finally" stack).
#[test]
fn simple_cycles_callback_survives_failing_nested_calls() {
    let g = complete(5);
    let before = unsafe { igraph::IGRAPH_FINALLY_STACK_SIZE() };
    let mut failures = 0;
    let mut cycles = 0;
    g.simple_cycles_callback(&SimpleCyclesOptions::default(), |vs, _| {
        let err = g
            .distances(.., vec![0, 99], None, NeighborMode::All)
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
        failures += 1;
        cycles += 1;
        assert!(vs.len() >= 3);
        ControlFlow::Continue(())
    })
    .unwrap();
    // K5 has 10 + 15 + 12 = 37 simple cycles (lengths 3, 4 and 5).
    assert_eq!(cycles, 37);
    assert_eq!(failures, 37);
    assert_eq!(unsafe { igraph::IGRAPH_FINALLY_STACK_SIZE() }, before);
    // The search still works afterwards.
    assert_eq!(
        g.simple_cycles(&SimpleCyclesOptions::default())
            .unwrap()
            .len(),
        37
    );
}

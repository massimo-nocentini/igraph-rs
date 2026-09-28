//! Tests for the BFS/DFS traversals of `igraph::visitor`.

use igraph::prelude::*;
use igraph::visitor::{BfsOptions, BfsVisit, DfsEvent, DfsOptions};
use std::ops::ControlFlow;

/// Zachary's karate club (34 vertices, 78 edges).
fn karate() -> Graph {
    Graph::famous("Zachary").unwrap()
}

/// The path `0 - 1 - ... - (n-1)`.
fn path(n: usize) -> Graph {
    Graph::ring(n, false, false, false).unwrap()
}

/// The undirected cycle on `n` vertices.
fn cycle(n: usize) -> Graph {
    Graph::ring(n, false, false, true).unwrap()
}

/// Converts igraph's sentinel-based vectors (as printed in its `.out` files)
/// into the `Option`s used by the bindings.
fn opt(v: &[i64]) -> Vec<Option<i64>> {
    v.iter().map(|&x| (x >= 0).then_some(x)).collect()
}

fn optu(v: &[i64]) -> Vec<Option<usize>> {
    v.iter().map(|&x| usize::try_from(x).ok()).collect()
}

/// Two disjoint undirected 10-rings, built exactly as in igraph's
/// `tests/unit/bfs.c` (`igraph_ring` + `igraph_disjoint_union`).
fn two_rings() -> Graph {
    let ring = cycle(10);
    ring.disjoint_union(&ring).unwrap()
}

/// Binary tree on `n` vertices with edges `i -> 2i+1, 2i+2`.
fn binary_tree(n: usize, directed: bool) -> Graph {
    let mode = if directed {
        TreeMode::Out
    } else {
        TreeMode::Undirected
    };
    Graph::kary_tree(n, 2, mode).unwrap()
}

fn visit_order(g: &Graph, roots: &[i64], opts: &BfsOptions<'_>) -> Vec<i64> {
    let mut seen = vec![];
    g.bfs_with(roots, opts, |v| {
        seen.push(v.vid);
        ControlFlow::Continue(())
    })
    .unwrap();
    seen
}

// ---------------------------------------------------------------------------
// BFS: igraph's reference outputs
// ---------------------------------------------------------------------------

#[test]
fn bfs_matches_igraph_reference_output() {
    // tests/unit/bfs.out (first six vectors)
    let g = two_rings();
    let r = g
        .bfs(&[0], &BfsOptions::default().with_unreachable(true))
        .unwrap();
    assert_eq!(
        r.order,
        [
            0, 1, 9, 2, 8, 3, 7, 4, 6, 5, 10, 11, 19, 12, 18, 13, 17, 14, 16, 15
        ]
    );
    assert_eq!(
        r.rank,
        optu(&[
            0, 1, 3, 5, 7, 9, 8, 6, 4, 2, 10, 11, 13, 15, 17, 19, 18, 16, 14, 12
        ])
    );
    assert_eq!(
        r.parents,
        opt(&[
            -1, 0, 1, 2, 3, 4, 7, 8, 9, 0, -1, 10, 11, 12, 13, 14, 17, 18, 19, 10
        ])
    );
    assert_eq!(
        r.pred,
        opt(&[
            -1, 0, 9, 8, 7, 6, 4, 3, 2, 1, -1, 10, 19, 18, 17, 16, 14, 13, 12, 11
        ])
    );
    assert_eq!(
        r.succ,
        opt(&[
            1, 9, 8, 7, 6, -1, 5, 4, 3, 2, 11, 19, 18, 17, 16, -1, 15, 14, 13, 12
        ])
    );
    assert_eq!(
        r.dist,
        optu(&[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1])
    );
    assert!(!r.stopped);
    assert_eq!(r.roots(), [0, 10]);
}

#[test]
fn bfs_callback_orders_match_igraph_unit_test() {
    // tests/unit/bfs.c / bfs.out (callback orders), and
    // examples/simple/igraph_bfs_callback.out for the first one.
    let g = two_rings();
    let all = BfsOptions::default().with_unreachable(true);
    assert_eq!(
        visit_order(&g, &[0], &all),
        [
            0, 1, 9, 2, 8, 3, 7, 4, 6, 5, 10, 11, 19, 12, 18, 13, 17, 14, 16, 15
        ]
    );
    assert_eq!(
        visit_order(&g, &[2], &all),
        [
            2, 1, 3, 0, 4, 9, 5, 8, 6, 7, 10, 11, 19, 12, 18, 13, 17, 14, 16, 15
        ]
    );

    let restricted: Vec<i64> = (5..20).collect();
    let tail = [5, 6, 7, 8, 9, 10, 11, 19, 12, 18, 13, 17, 14, 16, 15];
    let restr_all = all.with_restricted(&restricted);
    assert_eq!(visit_order(&g, &[5], &restr_all), tail);
    // A root outside the restricted set is skipped.
    assert_eq!(visit_order(&g, &[4], &restr_all), tail);
    let restr_only = BfsOptions::default().with_restricted(&restricted);
    assert!(visit_order(&g, &[3], &restr_only).is_empty());
    // Multiple roots: 3 and 4 are not allowed, 6 is.
    assert_eq!(visit_order(&g, &[3, 4, 6], &restr_only), [6, 5, 7, 8, 9]);
    // No roots at all.
    assert!(visit_order(&g, &[], &restr_only).is_empty());
}

#[test]
fn bfs_multiple_roots_skip_already_visited_ones() {
    let g = two_rings();
    // Root 5 is reached from 0 and is skipped; then 12 starts a new tree.
    let r = g.bfs(&[0, 5, 12], &BfsOptions::default()).unwrap();
    assert_eq!(r.roots(), [0, 12]);
    assert_eq!(r.order.len(), 20);
    assert_eq!(r.dist[12], Some(0));
    assert_eq!(r.dist[17], Some(5));
}

#[test]
fn bfs_visit_records_are_consistent_with_result() {
    let g = karate();
    let mut visits: Vec<BfsVisit> = vec![];
    let r = g
        .bfs_with(&[0], &BfsOptions::default(), |v| {
            visits.push(v);
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(visits.len(), 34);
    for (i, v) in visits.iter().enumerate() {
        assert_eq!(v.rank, i);
        assert_eq!(r.order[i], v.vid);
        assert_eq!(r.rank[v.vid as usize], Some(i));
        assert_eq!(r.dist[v.vid as usize], Some(v.dist));
        assert_eq!(r.pred[v.vid as usize], v.pred);
        assert_eq!(r.succ[v.vid as usize], v.succ);
        // pred/succ link consecutive visits
        assert_eq!(v.pred, i.checked_sub(1).map(|j| visits[j].vid));
        assert_eq!(v.succ, visits.get(i + 1).map(|w| w.vid));
    }
    // Distances never decrease along a BFS order.
    assert!(visits.windows(2).all(|w| w[0].dist <= w[1].dist));
}

#[test]
fn bfs_tree_invariants_on_karate() {
    let g = karate();
    let r = g.bfs(&[33], &BfsOptions::default()).unwrap();
    for v in g.vertices() {
        let d = r.dist[v as usize].unwrap();
        match r.parents[v as usize] {
            None => assert_eq!(v, 33),
            Some(p) => {
                assert_eq!(r.dist[p as usize], Some(d - 1));
                assert!(
                    g.get_eid(p, v, false).unwrap().is_some(),
                    "{p}-{v} is not an edge"
                );
            }
        }
        let path = r.path_to(v).unwrap();
        assert_eq!(path.len(), d + 1);
        assert_eq!(path[0], 33);
        assert_eq!(*path.last().unwrap(), v);
    }
    // The number of BFS layers is the eccentricity of the root plus one.
    let s = g.bfs_simple(33, NeighborMode::All).unwrap();
    let max = r.dist.iter().flatten().max().copied().unwrap();
    assert_eq!(s.num_layers(), max + 1);
}

#[test]
fn bfs_directed_modes() {
    // 0 -> 1 -> 2, 3 -> 1
    let g = Graph::from_edges(&[(0, 1), (1, 2), (3, 1)], 4, true).unwrap();
    let out = g.bfs(&[1], &BfsOptions::default()).unwrap();
    assert_eq!(out.order, [1, 2]);
    let inn = g
        .bfs(&[1], &BfsOptions::default().with_mode(NeighborMode::In))
        .unwrap();
    assert_eq!(inn.order, [1, 0, 3]);
    let all = g
        .bfs(&[1], &BfsOptions::default().with_mode(NeighborMode::All))
        .unwrap();
    assert_eq!(all.order, [1, 0, 2, 3]);
    assert!(!out.is_visited(0));
    assert_eq!(out.path_to(0), None);
    assert_eq!(out.parents[0], None);
    assert_eq!(out.dist[3], None);
}

// ---------------------------------------------------------------------------
// BFS: early stop, panics, errors
// ---------------------------------------------------------------------------

#[test]
fn bfs_early_stop_leaves_partial_results() {
    let g = two_rings();
    let r = g
        .bfs_with(&[0], &BfsOptions::default().with_unreachable(true), |v| {
            if v.vid == 3 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .unwrap();
    assert!(r.stopped);
    assert_eq!(r.order, [0, 1, 9, 2, 8, 3]);
    assert_eq!(r.rank[3], Some(5));
    // The vertex we stopped at has no successor recorded.
    assert_eq!(r.succ[3], None);
    assert_eq!(r.succ[8], Some(3));
    // Vertices of the second ring were never reached.
    assert!((10..20).all(|v| !r.is_visited(v)));
    // Vertex 7 was already enqueued (igraph assigned it a parent) but never
    // visited: the bindings report it as unvisited everywhere.
    assert!(!r.is_visited(7));
    assert_eq!(r.parents[7], None);
    assert_eq!(r.dist[7], None);
    assert_eq!(r.path_to(7), None);
    // Every vertex with a parent was visited, and so was its parent.
    for (v, p) in r.parents.iter().enumerate() {
        if let Some(p) = p {
            assert!(r.is_visited(v as i64) && r.is_visited(*p));
        }
    }
}

#[test]
fn bfs_stop_on_first_vertex() {
    let g = path(5);
    let r = g
        .bfs_with(&[2], &BfsOptions::default(), |_| ControlFlow::Break(()))
        .unwrap();
    assert!(r.stopped);
    assert_eq!(r.order, [2]);
}

#[test]
fn bfs_visitor_panic_propagates_and_library_recovers() {
    let g = karate();
    let caught = std::panic::catch_unwind(|| {
        let _ = g.bfs_with(&[0], &BfsOptions::default(), |v| {
            if v.rank == 7 {
                panic!("boom at {}", v.vid);
            }
            ControlFlow::Continue(())
        });
    });
    let payload = caught.unwrap_err();
    let msg = payload.downcast_ref::<String>().unwrap();
    assert!(msg.starts_with("boom at"));
    // The thread's igraph state is still usable afterwards.
    let r = g.bfs(&[0], &BfsOptions::default()).unwrap();
    assert_eq!(r.order.len(), 34);
}

#[test]
fn bfs_invalid_arguments() {
    let g = path(4);
    let err = g.bfs(&[4], &BfsOptions::default()).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    let err = g.bfs(&[-1], &BfsOptions::default()).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    let bad = [0, 9];
    let err = g
        .bfs(&[0], &BfsOptions::default().with_restricted(&bad))
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    // The visitor is never called when the arguments are invalid.
    let mut called = false;
    let res = g.bfs_with(&[10], &BfsOptions::default(), |_| {
        called = true;
        ControlFlow::Continue(())
    });
    assert!(res.is_err());
    assert!(!called);
}

#[test]
fn bfs_on_empty_graph() {
    let g = Graph::new(0, false);
    let r = g
        .bfs(&[], &BfsOptions::default().with_unreachable(true))
        .unwrap();
    assert!(r.order.is_empty() && r.rank.is_empty());
    let err = g.bfs(&[0], &BfsOptions::default()).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
}

// ---------------------------------------------------------------------------
// bfs_simple
// ---------------------------------------------------------------------------

#[test]
fn bfs_simple_matches_igraph_reference_output() {
    // tests/unit/bfs_simple.out (the path case is also
    // examples/simple/igraph_bfs_simple.out)
    let ring = path(10);
    let r = ring.bfs_simple(0, NeighborMode::All).unwrap();
    assert_eq!(r.order, (0..10).collect::<Vec<_>>());
    assert_eq!(r.layers, (0..=10).collect::<Vec<usize>>());
    assert_eq!(r.parents, opt(&[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]));

    let tree = binary_tree(20, false);
    let r = tree.bfs_simple(0, NeighborMode::All).unwrap();
    assert_eq!(r.order, (0..20).collect::<Vec<_>>());
    assert_eq!(r.layers, [0, 1, 3, 7, 15, 20]);
    assert_eq!(
        r.parents,
        opt(&[-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9])
    );
    let widths: Vec<usize> = r.iter_layers().map(<[i64]>::len).collect();
    assert_eq!(widths, [1, 2, 4, 8, 5]);
    assert_eq!(r.layer(4), [15, 16, 17, 18, 19]);
    assert!(r.layer(5).is_empty());

    let out_tree = binary_tree(20, true);
    let r = out_tree.bfs_simple(7, NeighborMode::Out).unwrap();
    assert_eq!(r.layers, [0, 1, 3]);
    assert_eq!(r.order, [7, 15, 16]);
    // Even though most vertices are unreachable, the eccentricity (computed
    // with the same mode) is still the index of the last layer.
    assert_eq!(
        out_tree.eccentricity(7, None, NeighborMode::Out).unwrap(),
        [(r.num_layers() - 1) as f64]
    );
    assert_eq!(
        r.parents,
        opt(&[
            -2, -2, -2, -2, -2, -2, -2, -1, -2, -2, -2, -2, -2, -2, -2, 7, 7, -2, -2, -2
        ])
    );
}

#[test]
fn bfs_simple_agrees_with_full_bfs() {
    let g = karate();
    for root in [0, 16, 25, 33] {
        let s = g.bfs_simple(root, NeighborMode::All).unwrap();
        let f = g.bfs(&[root], &BfsOptions::default()).unwrap();
        assert_eq!(s.order, f.order);
        assert_eq!(s.parents, f.parents);
        for (d, layer) in s.iter_layers().enumerate() {
            assert!(layer.iter().all(|&v| f.dist[v as usize] == Some(d)));
        }
    }
}

#[test]
fn bfs_simple_invalid_root() {
    let g = path(3);
    assert_eq!(
        g.bfs_simple(3, NeighborMode::All).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        g.bfs_simple(-1, NeighborMode::All).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
    let empty = Graph::new(0, true);
    assert!(empty.bfs_simple(0, NeighborMode::Out).is_err());
}

// ---------------------------------------------------------------------------
// DFS
// ---------------------------------------------------------------------------

#[test]
fn dfs_pre_and_post_order_of_binary_tree() {
    let t = binary_tree(7, false);
    let r = t.dfs(0, &DfsOptions::default()).unwrap();
    assert_eq!(r.order, [0, 1, 3, 4, 2, 5, 6]);
    assert_eq!(r.order_out, [3, 4, 1, 5, 6, 2, 0]);
    assert_eq!(r.parents, opt(&[-1, 0, 0, 1, 1, 2, 2]));
    assert_eq!(r.dist, optu(&[0, 1, 1, 2, 2, 2, 2]));
    assert_eq!(r.path_to(6), Some(vec![0, 2, 6]));
    assert!(!r.stopped);
}

#[test]
fn dfs_on_cycle_goes_all_the_way_round() {
    let g = cycle(6);
    let r = g.dfs(0, &DfsOptions::default()).unwrap();
    assert_eq!(r.order, [0, 1, 2, 3, 4, 5]);
    assert_eq!(r.order_out, [5, 4, 3, 2, 1, 0]);
    assert_eq!(r.dist[5], Some(5));
}

#[test]
fn dfs_unreachable_restarts_with_correct_depths() {
    // Two paths 0-1-2 and 3-4-5: igraph's raw `dist` drifts for the second
    // tree (igraph 1.0.0 and 1.0.1, see
    // `raw_igraph_dfs_quirks_are_still_present`); the bindings report the
    // true depths.
    let g = Graph::from_edges(&[(0, 1), (1, 2), (3, 4), (4, 5)], 6, false).unwrap();
    let only = g.dfs(0, &DfsOptions::default()).unwrap();
    assert_eq!(only.order, [0, 1, 2]);
    assert!(!only.is_visited(3));
    assert_eq!(only.dist[4], None);

    let mut events = vec![];
    let all = g
        .dfs_with(0, &DfsOptions::default().with_unreachable(true), |e| {
            events.push(e);
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(all.order, [0, 1, 2, 3, 4, 5]);
    assert_eq!(all.dist, optu(&[0, 1, 2, 0, 1, 2]));
    assert_eq!(all.parents, opt(&[-1, 0, 1, -1, 3, 4]));
    use DfsEvent::*;
    assert_eq!(
        events,
        [
            Discover { vid: 0, dist: 0 },
            Discover { vid: 1, dist: 1 },
            Discover { vid: 2, dist: 2 },
            Finish { vid: 2, dist: 2 },
            Finish { vid: 1, dist: 1 },
            Finish { vid: 0, dist: 0 },
            Discover { vid: 3, dist: 0 },
            Discover { vid: 4, dist: 1 },
            Discover { vid: 5, dist: 2 },
            Finish { vid: 5, dist: 2 },
            Finish { vid: 4, dist: 1 },
            Finish { vid: 3, dist: 0 },
        ]
    );
}

#[test]
fn dfs_events_are_well_parenthesised() {
    let g = karate();
    let mut stack: Vec<i64> = vec![];
    let mut discovered = 0;
    let r = g
        .dfs_with(5, &DfsOptions::default(), |e| {
            match e {
                DfsEvent::Discover { vid, dist } => {
                    assert_eq!(dist, stack.len());
                    stack.push(vid);
                    discovered += 1;
                }
                DfsEvent::Finish { vid, dist } => {
                    assert_eq!(stack.pop(), Some(vid));
                    assert_eq!(dist, stack.len());
                }
            }
            ControlFlow::Continue(())
        })
        .unwrap();
    assert!(stack.is_empty());
    assert_eq!(discovered, 34);
    assert_eq!(r.order[0], 5);
    assert_eq!(*r.order_out.last().unwrap(), 5);
    // Every DFS-tree edge is a graph edge and depths grow by one.
    for v in g.vertices() {
        if let Some(p) = r.parents[v as usize] {
            assert!(g.get_eid(p, v, false).unwrap().is_some());
            assert_eq!(r.dist[v as usize], r.dist[p as usize].map(|d| d + 1));
        }
    }
}

#[test]
fn dfs_directed_in_mode() {
    // 0 -> 2, 1 -> 2, 2 -> 3: walking backwards from 3 finds everyone.
    let g = Graph::from_edges(&[(0, 2), (1, 2), (2, 3)], 4, true).unwrap();
    let fwd = g.dfs(3, &DfsOptions::default()).unwrap();
    assert_eq!(fwd.order, [3]);
    let back = g
        .dfs(3, &DfsOptions::default().with_mode(NeighborMode::In))
        .unwrap();
    assert_eq!(back.order, [3, 2, 0, 1]);
    assert_eq!(back.order_out, [0, 1, 2, 3]);
}

#[test]
fn dfs_early_stop_on_discover_and_on_finish() {
    let t = binary_tree(7, false);
    // Stop when discovering vertex 4.
    let r = t
        .dfs_with(0, &DfsOptions::default(), |e| match e {
            DfsEvent::Discover { vid: 4, .. } => ControlFlow::Break(()),
            _ => ControlFlow::Continue(()),
        })
        .unwrap();
    assert!(r.stopped);
    assert_eq!(r.order, [0, 1, 3, 4]);
    assert_eq!(r.order_out, [3]);
    assert!(!r.is_visited(2));

    // Stop when the subtree of 1 is complete.
    let r = t
        .dfs_with(0, &DfsOptions::default(), |e| match e {
            DfsEvent::Finish { vid: 1, .. } => ControlFlow::Break(()),
            _ => ControlFlow::Continue(()),
        })
        .unwrap();
    assert!(r.stopped);
    assert_eq!(r.order, [0, 1, 3, 4]);
    assert_eq!(r.order_out, [3, 4, 1]);

    // Stop right at the root.
    let r = t
        .dfs_with(0, &DfsOptions::default(), |_| ControlFlow::Break(()))
        .unwrap();
    assert_eq!(r.order, [0]);
    assert!(r.order_out.is_empty());
}

#[test]
fn dfs_visitor_panic_propagates() {
    let g = karate();
    let caught = std::panic::catch_unwind(|| {
        g.dfs_with(0, &DfsOptions::default(), |e| {
            if let DfsEvent::Finish { .. } = e {
                panic!("finished!");
            }
            ControlFlow::Continue(())
        })
    });
    assert_eq!(
        *caught.unwrap_err().downcast_ref::<&str>().unwrap(),
        "finished!"
    );
    assert_eq!(g.dfs(0, &DfsOptions::default()).unwrap().order.len(), 34);
}

#[test]
fn dfs_invalid_root() {
    let g = path(3);
    assert_eq!(
        g.dfs(3, &DfsOptions::default()).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
    let empty = Graph::new(0, false);
    assert!(empty.dfs(0, &DfsOptions::default()).is_err());
}

#[test]
fn traversals_run_in_parallel_threads() {
    let handles: Vec<_> = (0..8)
        .map(|i| {
            std::thread::spawn(move || {
                let g = karate();
                let b = g.bfs(&[i], &BfsOptions::default()).unwrap();
                let d = g.dfs(i, &DfsOptions::default()).unwrap();
                (b.order.len(), d.order.len())
            })
        })
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), (34, 34));
    }
}

#[test]
fn loops_multi_edges_and_singletons() {
    // Self-loops and parallel edges never make a vertex visited twice.
    let g = Graph::from_edges(&[(0, 0), (0, 1), (0, 1), (1, 2), (2, 2)], 3, false).unwrap();
    let b = g.bfs(&[0], &BfsOptions::default()).unwrap();
    assert_eq!(b.order, [0, 1, 2]);
    assert_eq!(b.dist, [Some(0), Some(1), Some(2)]);
    let d = g.dfs(0, &DfsOptions::default()).unwrap();
    assert_eq!(d.order, [0, 1, 2]);
    assert_eq!(d.order_out, [2, 1, 0]);

    // A single isolated vertex: one layer made of the root alone.
    let one = Graph::new(1, true);
    let s = one.bfs_simple(0, NeighborMode::Out).unwrap();
    assert_eq!(s.order, [0]);
    assert_eq!(s.layers, [0, 1]);
    assert_eq!(s.num_layers(), 1);
    assert_eq!(s.parents, [None]);
    let d = one.dfs(0, &DfsOptions::default()).unwrap();
    assert_eq!(
        (d.order, d.order_out, d.dist),
        (vec![0], vec![0], vec![Some(0)])
    );
}

#[test]
fn bfs_empty_restricted_set_visits_nothing() {
    let g = path(4);
    let none: [i64; 0] = [];
    let opts = BfsOptions::default()
        .with_unreachable(true)
        .with_restricted(&none);
    let r = g.bfs(&[0, 1], &opts).unwrap();
    assert!(r.order.is_empty());
    assert!(r.rank.iter().all(Option::is_none));
    assert!(r.parents.iter().all(Option::is_none));
    assert!(r.roots().is_empty());
}

#[test]
fn bfs_restricted_distances_follow_allowed_vertices_only() {
    // 10-ring restricted to 0..=5: to reach 5 from 0 the search must go the
    // long way round (0-1-2-3-4-5), since 6..=9 are forbidden.
    let g = cycle(10);
    let allowed: Vec<i64> = (0..=5).collect();
    let r = g
        .bfs(&[0], &BfsOptions::default().with_restricted(&allowed))
        .unwrap();
    assert_eq!(r.order, [0, 1, 2, 3, 4, 5]);
    assert_eq!(r.dist[5], Some(5));
    assert_eq!(r.path_to(5), Some(vec![0, 1, 2, 3, 4, 5]));
    assert!((6..10).all(|v| !r.is_visited(v)));
    // Without the restriction, 5 is at distance 5 either way but 9 is at 1.
    let free = g.bfs(&[0], &BfsOptions::default()).unwrap();
    assert_eq!(free.dist[9], Some(1));
}

#[test]
fn dfs_directed_unreachable_in_mode() {
    // 0 -> 1 -> 2 and 3 -> 2: walking against the edges from 2 reaches
    // everyone; from 0 nothing else (then the search restarts at 1 and 2).
    let g = Graph::from_edges(&[(0, 1), (1, 2), (3, 2)], 4, true).unwrap();
    let opts = DfsOptions::default()
        .with_mode(NeighborMode::In)
        .with_unreachable(true);
    let r = g.dfs(0, &opts).unwrap();
    assert_eq!(r.order, [0, 1, 2, 3]);
    // Restarts at 1 and 2; 3 is then reached from 2 through the edge 3 -> 2
    // (igraph 1.0.0 and 1.0.1's raw `dist` gives 3 the depth -1 here, see
    // `raw_igraph_dfs_quirks_are_still_present`; the bindings give 1).
    assert_eq!(r.parents, [None, None, None, Some(2)]);
    assert_eq!(r.dist, [Some(0), Some(0), Some(0), Some(1)]);
    let r = g.dfs(2, &opts).unwrap();
    assert_eq!(r.order, [2, 1, 0, 3]);
    assert_eq!(r.parents, [Some(1), Some(2), None, Some(2)]);
    assert_eq!(r.dist, [Some(2), Some(1), Some(0), Some(1)]);
}

#[test]
fn dfs_event_accessors_and_result_helpers() {
    let d = DfsEvent::Discover { vid: 4, dist: 2 };
    let f = DfsEvent::Finish { vid: 7, dist: 0 };
    assert_eq!((d.vid(), d.dist()), (4, 2));
    assert_eq!((f.vid(), f.dist()), (7, 0));

    // Out-of-range or negative ids are simply "not visited".
    let g = path(3);
    let b = g.bfs(&[0], &BfsOptions::default()).unwrap();
    assert!(!b.is_visited(-1) && !b.is_visited(3));
    assert_eq!(b.path_to(-1), None);
    assert_eq!(b.path_to(99), None);
    let r = g.dfs(0, &DfsOptions::default()).unwrap();
    assert!(!r.is_visited(-1) && !r.is_visited(3));
    assert_eq!(r.path_to(3), None);
    assert_eq!(r.path_to(2), Some(vec![0, 1, 2]));

    let opts = DfsOptions::default()
        .with_mode(NeighborMode::In)
        .with_unreachable(true);
    assert_eq!(opts.mode, NeighborMode::In);
    assert!(opts.unreachable);
    assert_eq!(DfsOptions::default().mode, NeighborMode::Out);
    assert!(!DfsOptions::default().unreachable);
}

#[test]
fn finish_events_report_the_discovery_depth() {
    // The raw C out-callback receives one less than the discovery depth;
    // the bindings report the same depth for both events of a vertex.
    let t = binary_tree(15, false);
    let mut depth_at_discovery = vec![None; 15];
    t.dfs_with(0, &DfsOptions::default(), |e| {
        match e {
            DfsEvent::Discover { vid, dist } => depth_at_discovery[vid as usize] = Some(dist),
            DfsEvent::Finish { vid, dist } => {
                assert_eq!(depth_at_discovery[vid as usize], Some(dist))
            }
        }
        ControlFlow::Continue(())
    })
    .unwrap();
    // Complete binary tree on 15 vertices: depths 0, 1 (x2), 2 (x4), 3 (x8).
    let mut hist = [0; 4];
    for d in depth_at_discovery.iter().flatten() {
        hist[*d] += 1;
    }
    assert_eq!(hist, [1, 2, 4, 8]);
}

/// Proves that the raw `igraph_dfs` quirks worked around by the bindings (depth
/// drift after a restart, off-by-one depth in the out-callback, `EINVAL` for an
/// invalid root) are
/// still present in igraph 1.0.1 (if this test ever fails after an igraph
/// upgrade, the corresponding workarounds in `src/visitor.rs` can go).
#[test]
fn raw_igraph_dfs_quirks_are_still_present() {
    use igraph::{ffi, igraph_call};
    use std::ptr::null_mut;

    // Two paths 0-1-2 and 3-4-5, searched with `unreachable = true`.
    let g = Graph::from_edges(&[(0, 1), (1, 2), (3, 4), (4, 5)], 6, false).unwrap();
    let mut dist = VectorInt::new();
    igraph_call!(ffi::igraph_dfs(
        &g,
        0,
        NeighborMode::All.into(),
        true,
        null_mut(),
        null_mut(),
        null_mut(),
        &mut dist,
        None,
        None,
        null_mut(),
    ))
    .unwrap();
    // The depth counter is not reset for the second tree: 4 and 5 get
    // depths 0 and 1 instead of 1 and 2.
    assert_eq!(dist.to_vec(), [0, 1, 2, 0, 0, 1]);
    let fixed = g
        .dfs(0, &DfsOptions::default().with_unreachable(true))
        .unwrap();
    assert_eq!(fixed.dist, optu(&[0, 1, 2, 0, 1, 2]));

    // The drift accumulates over the restarts and can even produce the
    // "not visited" sentinel -1: 0 -> 1 -> 2, 3 -> 2 walked against the
    // edges from 0 restarts at 1 and then at 2, which reaches 3.
    let d = Graph::from_edges(&[(0, 1), (1, 2), (3, 2)], 4, true).unwrap();
    let mut dist = VectorInt::new();
    igraph_call!(ffi::igraph_dfs(
        &d,
        0,
        NeighborMode::In.into(),
        true,
        null_mut(),
        null_mut(),
        null_mut(),
        &mut dist,
        None,
        None,
        null_mut(),
    ))
    .unwrap();
    assert_eq!(dist.to_vec(), [0, 0, 0, -1]);

    // The out-callback receives one less than the discovery depth (here on
    // a single tree, so without any drift): path 0-1-2 from 0.
    unsafe extern "C" fn record(
        _g: *const ffi::igraph_t,
        vid: ffi::igraph_int_t,
        dist: ffi::igraph_int_t,
        extra: *mut std::ffi::c_void,
    ) -> ffi::igraph_error_t {
        // SAFETY: `extra` is the `Vec` below, alive during the call.
        let log = unsafe { &mut *(extra as *mut Vec<(i64, i64)>) };
        log.push((vid, dist));
        ffi::igraph_error_type_t_IGRAPH_SUCCESS
    }
    let p = path(3);
    let mut discovered: Vec<(i64, i64)> = vec![];
    let mut finished: Vec<(i64, i64)> = vec![];
    igraph_call!(ffi::igraph_dfs(
        &p,
        0,
        NeighborMode::All.into(),
        false,
        null_mut(),
        null_mut(),
        null_mut(),
        null_mut(),
        Some(record),
        None,
        &mut discovered as *mut _ as *mut std::ffi::c_void,
    ))
    .unwrap();
    igraph_call!(ffi::igraph_dfs(
        &p,
        0,
        NeighborMode::All.into(),
        false,
        null_mut(),
        null_mut(),
        null_mut(),
        null_mut(),
        None,
        Some(record),
        &mut finished as *mut _ as *mut std::ffi::c_void,
    ))
    .unwrap();
    assert_eq!(discovered, [(0, 0), (1, 1), (2, 2)]);
    assert_eq!(finished, [(2, 1), (1, 0), (0, -1)]);
    let mut events = vec![];
    p.dfs_with(0, &DfsOptions::default(), |e| {
        if let DfsEvent::Finish { vid, dist } = e {
            events.push((vid, dist));
        }
        ControlFlow::Continue(())
    })
    .unwrap();
    assert_eq!(events, [(2, 2), (1, 1), (0, 0)]);

    // An invalid root is reported as `IGRAPH_EINVAL`, not `IGRAPH_EINVVID`.
    let err = igraph_call!(ffi::igraph_dfs(
        &g,
        6,
        NeighborMode::All.into(),
        false,
        null_mut(),
        null_mut(),
        null_mut(),
        null_mut(),
        None,
        None,
        null_mut(),
    ))
    .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    assert_eq!(
        g.dfs(6, &DfsOptions::default()).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
}

// ---------------------------------------------------------------------------
// Agreement with other modules
// ---------------------------------------------------------------------------

#[test]
fn bfs_distances_and_paths_agree_with_shortest_paths() {
    let g = karate();
    let all = g
        .distances(.., .., None, NeighborMode::All)
        .unwrap()
        .to_rows();
    for root in g.vertices() {
        let r = g.bfs(&[root], &BfsOptions::default()).unwrap();
        let bfs_dist: Vec<f64> = r.dist.iter().map(|d| d.unwrap() as f64).collect();
        assert_eq!(bfs_dist, all[root as usize]);
        // BFS-tree paths are shortest paths.
        for target in [0, 16, 26, 33] {
            let sp = g
                .get_shortest_path(root, target, None, NeighborMode::All)
                .unwrap();
            assert_eq!(r.path_to(target).unwrap().len(), sp.vertices.len());
        }
        // The deepest BFS layer is the eccentricity of the root.
        let s = g.bfs_simple(root, NeighborMode::All).unwrap();
        let ecc = g.eccentricity(root, None, NeighborMode::All).unwrap();
        assert_eq!((s.num_layers() - 1) as f64, ecc[0]);
    }
    // Largest eccentricity = diameter (5 for the karate club).
    let diam = g
        .vertices()
        .map(|v| g.bfs_simple(v, NeighborMode::All).unwrap().num_layers() - 1)
        .max()
        .unwrap();
    assert_eq!(diam, 5);
    assert_eq!(g.diameter().unwrap(), 5.0);
}

#[test]
fn directed_reachability_agrees_with_subcomponent() {
    // Out-tree: from vertex 1 the out-BFS reaches exactly its subtree, the
    // in-BFS exactly its ancestors.
    let t = binary_tree(15, true);
    for (mode, v) in [(NeighborMode::Out, 1), (NeighborMode::In, 11)] {
        let mut bfs = g_order(&t, v, mode);
        let mut dfs = t
            .dfs(v, &DfsOptions::default().with_mode(mode))
            .unwrap()
            .order;
        let mut sub = t.subcomponent(v, mode).unwrap();
        bfs.sort();
        dfs.sort();
        sub.sort();
        assert_eq!(bfs, sub);
        assert_eq!(dfs, sub);
    }
    assert_eq!(g_order(&t, 11, NeighborMode::In), [11, 5, 2, 0]);

    fn g_order(g: &Graph, v: i64, mode: NeighborMode) -> Vec<i64> {
        g.bfs(&[v], &BfsOptions::default().with_mode(mode))
            .unwrap()
            .order
    }
}

#[test]
fn bfs_forest_matches_connected_components_of_random_graphs() {
    rng::seed(2024).unwrap();
    for n in [1, 10, 50, 200] {
        for p in [0.0, 0.01, 0.05, 0.2] {
            let g = Graph::erdos_renyi_game_gnp(n, p, false, EdgeTypeSw::Simple, false).unwrap();
            let cc = g.connected_components(Connectedness::Weak).unwrap();
            let r = g
                .bfs(&[0], &BfsOptions::default().with_unreachable(true))
                .unwrap();
            let roots = r.roots();
            assert_eq!(roots.len(), cc.count, "n={n} p={p}");
            // Each search tree is exactly one component.
            for v in g.vertices() {
                let root = r.path_to(v).unwrap()[0];
                assert!(cc.same_component(v, root));
            }
            // DFS finds the same forest structure.
            let d = g
                .dfs(0, &DfsOptions::default().with_unreachable(true))
                .unwrap();
            assert_eq!(d.parents.iter().filter(|p| p.is_none()).count(), cc.count);
            // A spanning forest has |V| - #components edges.
            let tree_edges = r.parents.iter().flatten().count();
            assert_eq!(tree_edges, n - cc.count);
        }
    }
}

#[test]
fn dfs_topological_order_agrees_with_cycles_module() {
    rng::seed(7).unwrap();
    // Random DAG: orient every edge of a random graph from lower to higher id.
    let r = Graph::erdos_renyi_game_gnm(30, 80, false, EdgeTypeSw::Simple, false).unwrap();
    let edges: Vec<(i64, i64)> = r
        .edge_list()
        .into_iter()
        .map(|(a, b)| (a.min(b), a.max(b)))
        .collect();
    let dag = Graph::from_edges(&edges, 30, true).unwrap();
    assert!(dag.is_dag().unwrap());
    let d = dag
        .dfs(0, &DfsOptions::default().with_unreachable(true))
        .unwrap();
    let topo: Vec<i64> = d.order_out.iter().rev().copied().collect();
    let mut pos = vec![0; 30];
    for (i, &v) in topo.iter().enumerate() {
        pos[v as usize] = i;
    }
    assert!(
        edges
            .iter()
            .all(|&(a, b)| pos[a as usize] < pos[b as usize])
    );
    // `topological_sorting` returns another valid order of the same vertices.
    let mut other = dag.topological_sorting(NeighborMode::Out).unwrap();
    other.sort();
    assert_eq!(other, (0..30).collect::<Vec<_>>());
    assert_eq!(dag.find_cycle(NeighborMode::Out).unwrap(), None);
}

#[test]
fn unfold_tree_size_matches_bfs_non_tree_edges() {
    // Unfolding a connected graph from one root copies one vertex per
    // non-tree edge: the tree has |E| + 1 vertices, the BFS tree |V| - 1
    // edges.
    let g = karate();
    let r = g.bfs(&[0], &BfsOptions::default()).unwrap();
    let tree_edges = r.parents.iter().flatten().count();
    assert_eq!(tree_edges, 33);
    let unfolded = g.unfold_tree(NeighborMode::All, &[0]).unwrap();
    assert_eq!(
        unfolded.tree.vcount(),
        g.vcount() + (g.ecount() - tree_edges)
    );
    assert_eq!(unfolded.tree.vcount(), 79);
}

#[test]
fn seeded_random_traversals_are_reproducible_in_parallel() {
    // Each thread has its own default RNG: seeding it gives the same random
    // graph, hence the same traversal, whatever the other threads do.
    let run = |seed: u64| {
        std::thread::spawn(move || {
            rng::seed(seed).unwrap();
            let g = Graph::erdos_renyi_game_gnm(60, 90, false, EdgeTypeSw::Simple, false).unwrap();
            let b = g
                .bfs(&[0], &BfsOptions::default().with_unreachable(true))
                .unwrap();
            let d = g
                .dfs(0, &DfsOptions::default().with_unreachable(true))
                .unwrap();
            (b, d)
        })
    };
    let handles: Vec<_> = (0..8)
        .map(|i| run(if i % 2 == 0 { 11 } else { 12 }))
        .collect();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    for pair in results.chunks(2).collect::<Vec<_>>().windows(2) {
        assert_eq!(pair[0][0], pair[1][0]);
        assert_eq!(pair[0][1], pair[1][1]);
    }
    assert_ne!(results[0], results[1]);
}

// ---------------------------------------------------------------------------
// Use cases
// ---------------------------------------------------------------------------

/// Story: the karate club's instructor (vertex 0) wants to pass a message to
/// the administrator (vertex 33) through as few members as possible, and to
/// know who is "two handshakes" away.
#[test]
fn use_case_message_passing_in_the_karate_club() {
    let club = karate();
    let bfs = club.bfs(&[0], &BfsOptions::default()).unwrap();
    // 0 and 33 are not friends but share friends (e.g. 8): distance 2.
    assert_eq!(bfs.dist[33], Some(2));
    let chain = bfs.path_to(33).unwrap();
    assert_eq!(chain.len(), 3);
    assert!(club.get_eid(chain[0], chain[1], false).unwrap().is_some());
    assert!(club.get_eid(chain[1], chain[2], false).unwrap().is_some());

    // Layers: 16 friends at distance 1 (the instructor's degree).
    let s = club.bfs_simple(0, NeighborMode::All).unwrap();
    assert_eq!(s.layer(1).len(), 16);
    assert_eq!(
        s.layer(0).len() + s.layer(1).len() + s.layer(2).len() + s.layer(3).len(),
        34
    );

    // Broadcast with a budget: stop as soon as 10 members got the message.
    let mut informed = vec![];
    let r = club
        .bfs_with(&[0], &BfsOptions::default(), |v| {
            informed.push(v.vid);
            if informed.len() == 10 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .unwrap();
    assert!(r.stopped);
    assert_eq!(r.order, informed);
    assert!(
        informed
            .iter()
            .skip(1)
            .all(|&v| r.dist[v as usize] == Some(1))
    );
}

/// Story: a build system must compile tasks after their dependencies, and
/// refuse cyclic dependency graphs. DFS gives both a topological order
/// (reverse post-order) and cycle detection (a back edge to a vertex still
/// on the DFS stack).
#[test]
fn use_case_build_order_and_cycle_detection() {
    // Edge a -> b means "a must be built before b".
    const TASKS: [&str; 6] = ["fetch", "configure", "compile", "test", "docs", "package"];
    let deps = [(0, 1), (1, 2), (2, 3), (1, 4), (3, 5), (4, 5)];

    fn plan(deps: &[(i64, i64)], n: usize) -> Option<Vec<i64>> {
        let g = Graph::from_edges(deps, n, true).unwrap();
        let mut on_stack = vec![false; n];
        let mut finished = vec![];
        // One DFS over the whole graph (restarting from unreachable
        // vertices); a cycle shows up as an edge from the vertex just
        // discovered back to a vertex still on the DFS stack.
        let r = g
            .dfs_with(0, &DfsOptions::default().with_unreachable(true), |e| {
                match e {
                    DfsEvent::Discover { vid, .. } => {
                        on_stack[vid as usize] = true;
                        let succs = g.neighbors(vid, NeighborMode::Out).unwrap();
                        if succs.iter().any(|&w| on_stack[w as usize]) {
                            return ControlFlow::Break(());
                        }
                    }
                    DfsEvent::Finish { vid, .. } => {
                        on_stack[vid as usize] = false;
                        finished.push(vid);
                    }
                }
                ControlFlow::Continue(())
            })
            .unwrap();
        // The finishing order is exactly the post-order reported by igraph.
        if !r.stopped {
            assert_eq!(finished, r.order_out);
        }
        (!r.stopped).then(|| finished.into_iter().rev().collect())
    }

    let order = plan(&deps, TASKS.len()).expect("the dependencies are acyclic");
    let pos = |t: i64| order.iter().position(|&x| x == t).unwrap();
    for &(a, b) in &deps {
        assert!(
            pos(a) < pos(b),
            "{} must come before {}",
            TASKS[a as usize],
            TASKS[b as usize]
        );
    }
    assert_eq!(order[0], 0);
    assert_eq!(*order.last().unwrap(), 5);

    // Adding "package -> configure" creates a cycle.
    let mut cyclic = deps.to_vec();
    cyclic.push((5, 1));
    assert_eq!(plan(&cyclic, TASKS.len()), None);
}

/// Story: count the islands of an archipelago (connected components) with
/// a single BFS that restarts from unreachable vertices.
#[test]
fn use_case_counting_islands() {
    // Three islands: a triangle, an edge and an isolated rock.
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4)], 6, false).unwrap();
    let r = g
        .bfs(&[0], &BfsOptions::default().with_unreachable(true))
        .unwrap();
    let roots = r.roots();
    assert_eq!(roots, [0, 3, 5]);
    let mut island = vec![0; 6];
    for v in g.vertices() {
        let root = r.path_to(v).unwrap()[0];
        island[v as usize] = roots.iter().position(|&x| x == root).unwrap();
    }
    assert_eq!(island, [0, 0, 0, 1, 1, 2]);
}

/// A failing igraph call made (and ignored) inside a visitor must not free the
/// temporaries of the running search: the callback runs in its own level of
/// igraph's finally stack. Before the fix this aborted the process
/// (`dqueue.c: Assertion failed: q->stor_begin != NULL`).
#[test]
fn failing_igraph_call_inside_bfs_visitor_is_harmless() {
    let g = karate();
    let expected = g.bfs(&[0], &BfsOptions::default()).unwrap();
    let mut order = vec![];
    let r = g
        .bfs_with(&[0], &BfsOptions::default(), |v| {
            let mut other = Graph::new(3, false);
            let err = other.add_edge(0, 7).unwrap_err();
            assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
            order.push(v.vid);
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(r, expected);
    assert_eq!(order, expected.order);
}

/// Same as above for the depth-first search (before the fix:
/// `stack.c: Assertion failed: s->stor_begin != NULL`).
#[test]
fn failing_igraph_call_inside_dfs_visitor_is_harmless() {
    let g = karate();
    let expected = g.dfs(0, &DfsOptions::default()).unwrap();
    let mut discovered = vec![];
    let r = g
        .dfs_with(0, &DfsOptions::default(), |e| {
            let _ = Graph::new(3, false).add_edge(0, 7);
            if let DfsEvent::Discover { vid, .. } = e {
                discovered.push(vid);
            }
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(r, expected);
    assert_eq!(discovered, expected.order);
}

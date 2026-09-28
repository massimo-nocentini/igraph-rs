//! Integration tests for adjacency and incidence lists (`igraph_adjlist.h`).
//!
//! Expected values come from igraph's own unit tests
//! (`tests/unit/adjlist.out`, `inclist.out`, `igraph_adjlist_simplify.out`,
//! `igraph_adjlist_init_complementer.out`), from Zachary's karate club
//! (`Graph::famous("Zachary")`), from simple identities, and from the
//! whole-graph functions of other modules (`neighbors_with`, `incident`,
//! `complementer`, `simplify`, `get_adjacency`, `bfs_simple`,
//! `count_triangles`, ...) that must agree with the lists.

mod common;

use common::*;
use igraph::adjlist::{AdjList, IncList, LazyAdjList, LazyIncList};
use igraph::prelude::*;
use std::collections::VecDeque;

fn v(lists: &[&[i64]]) -> Vec<Vec<i64>> {
    lists.iter().map(|l| l.to_vec()).collect()
}

/// The graph of `test_loop_elimination_*` in igraph's `tests/unit/adjlist.c`.
fn loop_graph(directed: bool) -> Graph {
    let edges = [
        (0, 1),
        (0, 3),
        (1, 2),
        (2, 2),
        (2, 3),
        (3, 0),
        (3, 4),
        (4, 4),
        (4, 4),
    ];
    Graph::from_edges(&edges, 5, directed).unwrap()
}

/// The graph of `test_multiedge_elimination_*` in `tests/unit/adjlist.c`.
fn multi_graph(directed: bool) -> Graph {
    let edges = [
        (0, 1),
        (0, 3),
        (0, 8),
        (1, 2),
        (2, 2),
        (2, 3),
        (3, 0),
        (3, 4),
        (4, 4),
        (4, 4),
        (4, 5),
        (4, 5),
        (4, 5),
        (5, 6),
        (6, 7),
        (6, 8),
        (8, 0),
    ];
    Graph::from_edges(&edges, 9, directed).unwrap()
}

/// The graph of `tests/unit/inclist.c` (13 edges on 7 vertices).
fn inc_graph(directed: bool) -> Graph {
    let edges = [
        (0, 1),
        (0, 3),
        (1, 2),
        (2, 2),
        (2, 3),
        (3, 0),
        (3, 4),
        (4, 0),
        (4, 4),
        (4, 5),
        (4, 6),
        (4, 4),
        (6, 5),
    ];
    Graph::from_edges(&edges, 7, directed).unwrap()
}

/// The graph of `igraph_adjlist_simplify.c` and `igraph_adjlist_init_complementer.c`.
fn small_directed() -> Graph {
    let edges = [
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 3),
        (2, 0),
        (2, 3),
        (3, 4),
        (3, 4),
    ];
    Graph::from_edges(&edges, 6, true).unwrap()
}

/// Zachary's karate club, from the `constructors` module.
fn zachary() -> Graph {
    Graph::famous("Zachary").unwrap()
}

fn sorted_edges(g: &Graph) -> Vec<(i64, i64)> {
    let mut e: Vec<_> = g
        .edge_list()
        .into_iter()
        .map(|(a, b)| {
            if g.is_directed() || a <= b {
                (a, b)
            } else {
                (b, a)
            }
        })
        .collect();
    e.sort();
    e
}

// ---------------------------------------------------------------------------
// igraph_adjlist_init: loops and multi-edges (values from adjlist.out)
// ---------------------------------------------------------------------------

#[test]
fn undirected_loop_elimination_matches_igraph() {
    let g = loop_graph(false);
    // For undirected graphs the mode makes no difference.
    let none = v(&[&[1, 3, 3], &[0, 2], &[1, 3], &[0, 0, 2, 4], &[3]]);
    let once = v(&[&[1, 3, 3], &[0, 2], &[1, 2, 3], &[0, 0, 2, 4], &[3, 4, 4]]);
    let twice = v(&[
        &[1, 3, 3],
        &[0, 2],
        &[1, 2, 2, 3],
        &[0, 0, 2, 4],
        &[3, 4, 4, 4, 4],
    ]);
    for mode in [NeighborMode::All, NeighborMode::In, NeighborMode::Out] {
        assert_eq!(
            g.adjlist_init(mode, Loops::None, true).unwrap().to_vecs(),
            none
        );
        assert_eq!(
            g.adjlist_init(mode, Loops::Once, true).unwrap().to_vecs(),
            once
        );
        assert_eq!(
            g.adjlist_init(mode, Loops::Twice, true).unwrap().to_vecs(),
            twice
        );
        // The lazy list agrees.
        let mut lazy = g.lazy_adjlist_init(mode, Loops::Once, true).unwrap();
        assert_eq!(lazy.to_vecs().unwrap(), once);
        assert_eq!(lazy.mode(), NeighborMode::All);
    }
}

#[test]
fn directed_loop_elimination_matches_igraph() {
    let g = loop_graph(true);
    use NeighborMode::*;
    let cases: Vec<(NeighborMode, Loops, Vec<Vec<i64>>)> = vec![
        (In, Loops::None, v(&[&[3], &[0], &[1], &[0, 2], &[3]])),
        (
            In,
            Loops::Once,
            v(&[&[3], &[0], &[1, 2], &[0, 2], &[3, 4, 4]]),
        ),
        (
            In,
            Loops::Twice,
            v(&[&[3], &[0], &[1, 2], &[0, 2], &[3, 4, 4]]),
        ),
        (Out, Loops::None, v(&[&[1, 3], &[2], &[3], &[0, 4], &[]])),
        (
            Out,
            Loops::Once,
            v(&[&[1, 3], &[2], &[2, 3], &[0, 4], &[4, 4]]),
        ),
        (
            Out,
            Loops::Twice,
            v(&[&[1, 3], &[2], &[2, 3], &[0, 4], &[4, 4]]),
        ),
        (
            All,
            Loops::None,
            v(&[&[1, 3, 3], &[0, 2], &[1, 3], &[0, 0, 2, 4], &[3]]),
        ),
        (
            All,
            Loops::Once,
            v(&[&[1, 3, 3], &[0, 2], &[1, 2, 3], &[0, 0, 2, 4], &[3, 4, 4]]),
        ),
        (
            All,
            Loops::Twice,
            v(&[
                &[1, 3, 3],
                &[0, 2],
                &[1, 2, 2, 3],
                &[0, 0, 2, 4],
                &[3, 4, 4, 4, 4],
            ]),
        ),
    ];
    for (mode, loops, expected) in cases {
        let al = g.adjlist_init(mode, loops, true).unwrap();
        assert_eq!(al.to_vecs(), expected, "{mode:?} {loops:?}");
        let mut lazy = g.lazy_adjlist_init(mode, loops, true).unwrap();
        assert_eq!(lazy.to_vecs().unwrap(), expected, "lazy {mode:?} {loops:?}");
    }
}

#[test]
fn undirected_multiedge_elimination_matches_igraph() {
    let g = multi_graph(false);
    let none = v(&[
        &[1, 3, 8],
        &[0, 2],
        &[1, 3],
        &[0, 2, 4],
        &[3, 5],
        &[4, 6],
        &[5, 7, 8],
        &[6],
        &[0, 6],
    ]);
    let once = v(&[
        &[1, 3, 8],
        &[0, 2],
        &[1, 2, 3],
        &[0, 2, 4],
        &[3, 4, 5],
        &[4, 6],
        &[5, 7, 8],
        &[6],
        &[0, 6],
    ]);
    let twice = v(&[
        &[1, 3, 8],
        &[0, 2],
        &[1, 2, 2, 3],
        &[0, 2, 4],
        &[3, 4, 4, 5],
        &[4, 6],
        &[5, 7, 8],
        &[6],
        &[0, 6],
    ]);
    assert_eq!(
        g.adjlist_init(NeighborMode::All, Loops::None, false)
            .unwrap()
            .to_vecs(),
        none
    );
    assert_eq!(
        g.adjlist_init(NeighborMode::In, Loops::Once, false)
            .unwrap()
            .to_vecs(),
        once
    );
    assert_eq!(
        g.adjlist_init(NeighborMode::Out, Loops::Twice, false)
            .unwrap()
            .to_vecs(),
        twice
    );
    let mut lazy = LazyAdjList::new(&g, NeighborMode::Out, Loops::Twice, false).unwrap();
    assert_eq!(lazy.to_vecs().unwrap(), twice);
}

#[test]
fn directed_multiedge_elimination_matches_igraph() {
    let g = multi_graph(true);
    let in_none = v(&[
        &[3, 8],
        &[0],
        &[1],
        &[0, 2],
        &[3],
        &[4],
        &[5],
        &[6],
        &[0, 6],
    ]);
    let in_once = v(&[
        &[3, 8],
        &[0],
        &[1, 2],
        &[0, 2],
        &[3, 4],
        &[4],
        &[5],
        &[6],
        &[0, 6],
    ]);
    assert_eq!(
        g.adjlist_init(NeighborMode::In, Loops::None, false)
            .unwrap()
            .to_vecs(),
        in_none
    );
    assert_eq!(
        g.adjlist_init(NeighborMode::In, Loops::Once, false)
            .unwrap()
            .to_vecs(),
        in_once
    );
    assert_eq!(
        g.adjlist_init(NeighborMode::In, Loops::Twice, false)
            .unwrap()
            .to_vecs(),
        in_once
    );

    // Out-edges: loops listed once even if Loops::Twice is given.
    let out_none = v(&[
        &[1, 3, 8],
        &[2],
        &[3],
        &[0, 4],
        &[5],
        &[6],
        &[7, 8],
        &[],
        &[0],
    ]);
    let out_once = v(&[
        &[1, 3, 8],
        &[2],
        &[2, 3],
        &[0, 4],
        &[4, 5],
        &[6],
        &[7, 8],
        &[],
        &[0],
    ]);
    // In- and out-edges: the mutual pairs 0 <-> 3 and 0 <-> 8 are listed
    // once, like true multi-edges (this goes through the Rust-side collapse).
    let all_none = v(&[
        &[1, 3, 8],
        &[0, 2],
        &[1, 3],
        &[0, 2, 4],
        &[3, 5],
        &[4, 6],
        &[5, 7, 8],
        &[6],
        &[0, 6],
    ]);
    let all_once = v(&[
        &[1, 3, 8],
        &[0, 2],
        &[1, 2, 3],
        &[0, 2, 4],
        &[3, 4, 5],
        &[4, 6],
        &[5, 7, 8],
        &[6],
        &[0, 6],
    ]);
    let all_twice = v(&[
        &[1, 3, 8],
        &[0, 2],
        &[1, 2, 2, 3],
        &[0, 2, 4],
        &[3, 4, 4, 5],
        &[4, 6],
        &[5, 7, 8],
        &[6],
        &[0, 6],
    ]);
    use NeighborMode::*;
    let cases = [
        (In, Loops::None, &in_none),
        (In, Loops::Once, &in_once),
        (In, Loops::Twice, &in_once),
        (Out, Loops::None, &out_none),
        (Out, Loops::Once, &out_once),
        (Out, Loops::Twice, &out_once),
        (All, Loops::None, &all_none),
        (All, Loops::Once, &all_once),
        (All, Loops::Twice, &all_twice),
    ];
    for (mode, loops, expected) in cases {
        let al = g.adjlist_init(mode, loops, false).unwrap();
        assert_eq!(&al.to_vecs(), expected, "{mode:?} {loops:?}");
        let mut lazy = g.lazy_adjlist_init(mode, loops, false).unwrap();
        assert_eq!(
            &lazy.to_vecs().unwrap(),
            expected,
            "lazy {mode:?} {loops:?}"
        );
    }
    // The graph really has multi-edges (the double loop and the triple 4 -> 5).
    assert!(g.has_multiple().unwrap());
}

/// Canary for the igraph bug worked around by `Graph::adjlist_init` and
/// `Graph::lazy_adjlist_init` (present in igraph 1.0.0 and 1.0.1, and even
/// recorded in igraph's own `tests/unit/adjlist.out`, `test_caching`: "graph:
/// loop, loop: 2 multi: 0 ... multi cached: 1"). It calls the raw C function,
/// bypassing the wrapper. If this test starts failing after an igraph
/// upgrade, the bug was fixed upstream and the Rust-side collapsing can be
/// reconsidered.
#[test]
fn upstream_igraph_still_miscaches_multi_edges() {
    use igraph::ffi::{igraph_adjlist_init, igraph_error_type_t_IGRAPH_SUCCESS};
    use std::mem::MaybeUninit;

    fn raw_adjlist_init(g: &Graph, mode: NeighborMode, loops: Loops) -> AdjList {
        let mut raw = MaybeUninit::<AdjList>::zeroed();
        // Error handling was set up on this thread when `g` was created.
        let code =
            unsafe { igraph_adjlist_init(g, raw.as_mut_ptr(), mode.into(), loops.into(), false) };
        assert_eq!(code, igraph_error_type_t_IGRAPH_SUCCESS);
        unsafe { raw.assume_init() }
    }

    // A single undirected self-loop, listed twice, is taken for a multi-edge.
    let g = Graph::from_edges(&[(0, 0), (0, 1)], 2, false).unwrap();
    let al = raw_adjlist_init(&g, NeighborMode::All, Loops::Twice);
    assert_eq!(al.to_vecs(), v(&[&[0, 0, 1], &[0]]));
    assert!(
        g.has_multiple().unwrap(),
        "igraph now caches HAS_MULTI correctly"
    );
    // The truth, from a fresh copy of the graph with an empty cache.
    let fresh = Graph::from_edges(&[(0, 0), (0, 1)], 2, false).unwrap();
    assert!(!fresh.has_multiple().unwrap());

    // A directed mutual pair is taken for a multi-edge in mode All.
    let d = mutual();
    let al = raw_adjlist_init(&d, NeighborMode::All, Loops::None);
    assert_eq!(al.to_vecs(), v(&[&[1], &[0, 2], &[1]]));
    assert!(
        d.has_multiple().unwrap(),
        "igraph now caches HAS_MULTI correctly"
    );
    assert!(!mutual().has_multiple().unwrap());

    // Once the cache says "no multi-edges", the pair is listed twice.
    let s = mutual();
    assert!(!s.has_multiple().unwrap());
    let al = raw_adjlist_init(&s, NeighborMode::All, Loops::None);
    assert_eq!(al.to_vecs(), v(&[&[1, 1], &[0, 0, 2], &[1]]));
    // ... while the wrapper does not depend on the cache.
    assert_eq!(
        s.adjlist_init(NeighborMode::All, Loops::None, false)
            .unwrap()
            .to_vecs(),
        v(&[&[1], &[0, 2], &[1]])
    );
}

// ---------------------------------------------------------------------------
// Complementer and simplify (values from the igraph unit tests)
// ---------------------------------------------------------------------------

#[test]
fn complementer_matches_igraph() {
    let g = small_directed();
    use NeighborMode::*;
    let cases: Vec<(NeighborMode, Loops, Vec<Vec<i64>>)> = vec![
        (
            In,
            Loops::None,
            v(&[
                &[1, 3, 4, 5],
                &[2, 3, 4, 5],
                &[1, 3, 4, 5],
                &[0, 4, 5],
                &[0, 1, 2, 5],
                &[0, 1, 2, 3, 4],
            ]),
        ),
        (
            In,
            Loops::Once,
            v(&[
                &[0, 1, 3, 4, 5],
                &[2, 3, 4, 5],
                &[1, 2, 3, 4, 5],
                &[0, 3, 4, 5],
                &[0, 1, 2, 4, 5],
                &[0, 1, 2, 3, 4, 5],
            ]),
        ),
        (
            In,
            Loops::Twice,
            v(&[
                &[0, 1, 3, 4, 5],
                &[2, 3, 4, 5],
                &[1, 2, 3, 4, 5],
                &[0, 3, 4, 5],
                &[0, 1, 2, 4, 5],
                &[0, 1, 2, 3, 4, 5],
            ]),
        ),
        (
            Out,
            Loops::None,
            v(&[
                &[3, 4, 5],
                &[0, 2, 4, 5],
                &[1, 4, 5],
                &[0, 1, 2, 5],
                &[0, 1, 2, 3, 5],
                &[0, 1, 2, 3, 4],
            ]),
        ),
        (
            Out,
            Loops::Once,
            v(&[
                &[0, 3, 4, 5],
                &[0, 2, 4, 5],
                &[1, 2, 4, 5],
                &[0, 1, 2, 3, 5],
                &[0, 1, 2, 3, 4, 5],
                &[0, 1, 2, 3, 4, 5],
            ]),
        ),
        (
            All,
            Loops::None,
            v(&[
                &[3, 4, 5],
                &[2, 4, 5],
                &[1, 4, 5],
                &[0, 5],
                &[0, 1, 2, 5],
                &[0, 1, 2, 3, 4],
            ]),
        ),
        (
            All,
            Loops::Once,
            v(&[
                &[0, 3, 4, 5],
                &[2, 4, 5],
                &[1, 2, 4, 5],
                &[0, 3, 5],
                &[0, 1, 2, 4, 5],
                &[0, 1, 2, 3, 4, 5],
            ]),
        ),
        (
            All,
            Loops::Twice,
            v(&[
                &[0, 0, 3, 4, 5],
                &[2, 4, 5],
                &[1, 2, 2, 4, 5],
                &[0, 3, 3, 5],
                &[0, 1, 2, 4, 4, 5],
                &[0, 1, 2, 3, 4, 5, 5],
            ]),
        ),
    ];
    for (mode, loops, expected) in cases {
        assert_eq!(
            AdjList::complementer(&g, mode, loops).unwrap().to_vecs(),
            expected,
            "{mode:?} {loops:?}"
        );
    }
}

#[test]
fn complementer_of_karate_and_complete_graph() {
    let g = zachary();
    // The famous graph is the same as the hand-written edge list.
    assert!(g.is_same_graph(&karate()).unwrap());
    let al = g
        .adjlist_init(NeighborMode::All, Loops::None, false)
        .unwrap();
    let co = g
        .adjlist_init_complementer(NeighborMode::All, Loops::None)
        .unwrap();
    for u in 0..34 {
        // Neighbors and non-neighbors partition the other 33 vertices.
        assert_eq!(al[u].len() + co[u].len(), 33);
        assert!(al[u].iter().all(|x| !co[u].contains(x)));
    }
    // Edges of the complement: C(34, 2) - 78.
    assert_eq!(co.total_len() / 2, 34 * 33 / 2 - 78);
    // Same lists as the adjacency list of the complementer graph.
    let cg = g.complementer(false).unwrap();
    assert_eq!(cg.ecount(), 34 * 33 / 2 - 78);
    assert_eq!(
        cg.adjlist_init(NeighborMode::All, Loops::None, false)
            .unwrap(),
        co
    );
    // With loops: the complementer graph gets a loop on every vertex, which
    // the adjacency list reports twice (Loops::Twice) as usual.
    let cgl = g.complementer(true).unwrap();
    assert_eq!(
        cgl.adjlist_init(NeighborMode::All, Loops::Twice, true)
            .unwrap(),
        g.adjlist_init_complementer(NeighborMode::All, Loops::Twice)
            .unwrap()
    );
    let empty = complete(6)
        .adjlist_init_complementer(NeighborMode::All, Loops::None)
        .unwrap();
    assert_eq!(empty.total_len(), 0);
    assert_eq!(empty.len(), 6);
}

#[test]
fn simplify_matches_igraph() {
    let g = small_directed();
    let mut al = g
        .adjlist_init(NeighborMode::All, Loops::Twice, true)
        .unwrap();
    assert_eq!(
        al.to_vecs(),
        v(&[
            &[1, 2, 2],
            &[0, 1, 1, 3],
            &[0, 0, 3],
            &[1, 2, 4, 4],
            &[3, 3],
            &[]
        ])
    );
    al.simplify().unwrap();
    assert_eq!(
        al.to_vecs(),
        v(&[&[1, 2], &[0, 3], &[0, 3], &[1, 2, 4], &[3], &[]])
    );

    // No vertices at all.
    let mut empty = Graph::new(0, true)
        .adjlist_init(NeighborMode::All, Loops::Twice, true)
        .unwrap();
    empty.simplify().unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.to_string(), "");
}

#[test]
fn simplify_agrees_with_graph_simplify() {
    for directed in [false, true] {
        let g = multi_graph(directed);
        let mut al = g
            .adjlist_init(NeighborMode::All, Loops::Twice, true)
            .unwrap();
        al.simplify().unwrap();
        al.sort();
        let mut h = g.clone();
        h.simplify(true, true).unwrap();
        assert!(h.is_simple(true).unwrap());
        // Asking adjlist_init for no loops and no multi-edges is the same as
        // simplifying the lists, and as listing the neighbors in the
        // simplified graph (for a directed graph, mode All also merges the
        // mutual pairs 0 <-> 3 and 0 <-> 8), whatever igraph's property cache
        // says: `h` is known to have no multi-edges, `g` is not.
        let simple = g
            .adjlist_init(NeighborMode::All, Loops::None, false)
            .unwrap();
        assert_eq!(&simple[3], &[0, 2, 4]);
        assert_eq!(
            h.adjlist_init(NeighborMode::All, Loops::None, false)
                .unwrap(),
            simple,
            "directed = {directed}"
        );
        assert_eq!(al, simple);
    }
}

#[test]
fn simplify_order_and_sort() {
    let mut al = AdjList::from(vec![vec![3, 1, 0, 1, 2], vec![1, 1], vec![], vec![0, 0, 0]]);
    al.simplify().unwrap();
    al.sort();
    assert_eq!(al.to_vecs(), v(&[&[1, 2, 3], &[], &[], &[0]]));
}

// ---------------------------------------------------------------------------
// Incidence lists (values from inclist.out)
// ---------------------------------------------------------------------------

#[test]
fn undirected_inclist_matches_igraph() {
    let g = inc_graph(false);
    let none = v(&[
        &[0, 5, 1, 7],
        &[0, 2],
        &[2, 4],
        &[5, 1, 4, 6],
        &[7, 6, 9, 10],
        &[9, 12],
        &[10, 12],
    ]);
    let once = v(&[
        &[0, 5, 1, 7],
        &[0, 2],
        &[2, 3, 4],
        &[5, 1, 4, 6],
        &[7, 6, 11, 8, 9, 10],
        &[9, 12],
        &[10, 12],
    ]);
    let twice = v(&[
        &[0, 5, 1, 7],
        &[0, 2],
        &[2, 3, 3, 4],
        &[5, 1, 4, 6],
        &[7, 6, 11, 8, 11, 8, 9, 10],
        &[9, 12],
        &[10, 12],
    ]);
    for mode in [NeighborMode::All, NeighborMode::In, NeighborMode::Out] {
        assert_eq!(g.inclist_init(mode, Loops::None).unwrap().to_vecs(), none);
        assert_eq!(g.inclist_init(mode, Loops::Once).unwrap().to_vecs(), once);
        assert_eq!(
            IncList::new(&g, mode, Loops::Twice).unwrap().to_vecs(),
            twice
        );
        let mut lazy = g.lazy_inclist_init(mode, Loops::Twice).unwrap();
        assert_eq!(lazy.to_vecs().unwrap(), twice);
        assert_eq!(lazy.loops(), Loops::Twice);
    }
}

#[test]
fn directed_inclist_and_consistent_adjlist() {
    let g = inc_graph(true);
    let in_none = g.inclist_init(NeighborMode::In, Loops::None).unwrap();
    assert_eq!(
        in_none.to_vecs(),
        v(&[&[5, 7], &[0], &[2], &[1, 4], &[6], &[9, 12], &[10]])
    );
    let all_once = g.inclist_init(NeighborMode::All, Loops::Once).unwrap();
    assert_eq!(
        all_once.to_vecs(),
        v(&[
            &[0, 1, 5, 7],
            &[0, 2],
            &[2, 3, 4],
            &[5, 1, 4, 6],
            &[7, 6, 11, 8, 9, 10],
            &[9, 12],
            &[10, 12]
        ])
    );

    // test_adjlist_from_inclist: fprint output and the consistent adjlist.
    let mut il = g.inclist_init(NeighborMode::All, Loops::Twice).unwrap();
    let mut out = Vec::new();
    il.fprint(&mut out).unwrap();
    let text = String::from_utf8(out).unwrap();
    assert_eq!(
        text,
        "0 1 5 7\n0 2\n2 3 3 4\n5 1 4 6\n7 6 11 11 8 8 9 10\n9 12\n10 12\n"
    );
    assert_eq!(il.to_string(), text);

    let al = g.adjlist_init_from_inclist(&il).unwrap();
    assert_eq!(
        al.to_vecs(),
        v(&[
            &[1, 3, 3, 4],
            &[0, 2],
            &[1, 2, 2, 3],
            &[0, 0, 2, 4],
            &[0, 3, 4, 4, 4, 4, 5, 6],
            &[4, 6],
            &[4, 5]
        ])
    );
    assert_eq!(AdjList::from_inclist(&g, &il).unwrap(), al);

    // Clearing keeps one (empty) line per vertex.
    il.clear();
    assert_eq!(il.len(), 7);
    assert_eq!(il.total_len(), 0);
    assert_eq!(il.to_string(), "\n".repeat(7));
    // The adjacency list is independent of the incidence list.
    assert_eq!(al.total_len(), 26);
}

#[test]
fn lists_agree_with_per_vertex_queries() {
    let mut simplified = multi_graph(true);
    simplified.simplify(true, true).unwrap();
    for g in [
        multi_graph(true),
        multi_graph(false),
        inc_graph(true),
        mutual(),
        simplified,
    ] {
        for mode in [NeighborMode::Out, NeighborMode::In, NeighborMode::All] {
            for loops in [Loops::None, Loops::Once, Loops::Twice] {
                let il = g.inclist_init(mode, loops).unwrap();
                for multiple in [false, true] {
                    let al = g.adjlist_init(mode, loops, multiple).unwrap();
                    let mut lazy = g.lazy_adjlist_init(mode, loops, multiple).unwrap();
                    assert_eq!(lazy.to_vecs().unwrap(), al.to_vecs());
                    for u in g.vertices() {
                        assert_eq!(
                            &al[u as usize],
                            g.neighbors_with(u, mode, loops, multiple)
                                .unwrap()
                                .as_slice(),
                            "{mode:?} {loops:?} {multiple} vertex {u}"
                        );
                    }
                }
                for u in g.vertices() {
                    let mut ours = il[u as usize].to_vec();
                    let mut theirs = g.incident(u, mode, loops).unwrap();
                    ours.sort();
                    theirs.sort();
                    assert_eq!(ours, theirs, "{mode:?} {loops:?} vertex {u}");
                }
            }
        }
    }
}

#[test]
fn adjlist_matches_the_adjacency_matrix() {
    let g = multi_graph(false);
    let al = g
        .adjlist_init(NeighborMode::All, Loops::Twice, true)
        .unwrap();
    // With loops counted twice, entry (u, w) of the adjacency matrix is the
    // number of times w appears in the list of u.
    let m = g
        .get_adjacency(GetAdjacency::Both, None, Loops::Twice)
        .unwrap();
    for u in 0..al.len() {
        for w in 0..al.len() {
            let count = al[u].iter().filter(|&&x| x == w as i64).count();
            assert_eq!(m[(u, w)], count as f64, "({u}, {w})");
        }
    }
}

#[test]
fn inclist_edge_ids_cover_every_edge() {
    let g = zachary();
    let il = g.inclist_init(NeighborMode::All, Loops::Twice).unwrap();
    let mut count = vec![0; g.ecount()];
    for (u, edges) in il.iter().enumerate() {
        assert_eq!(
            edges.len() as i64,
            g.degree_of(u as i64, NeighborMode::All, Loops::Twice)
                .unwrap()
        );
        for &e in edges {
            let (a, b) = g.edge(e).unwrap();
            assert!(a == u as i64 || b == u as i64);
            count[e as usize] += 1;
        }
    }
    // Each undirected edge appears once per endpoint.
    assert!(count.iter().all(|&c| c == 2));
}

// ---------------------------------------------------------------------------
// Conversion back to graphs (igraph_adjlist)
// ---------------------------------------------------------------------------

#[test]
fn tree_roundtrips_like_igraph_example() {
    // examples/simple/adjlist.c and tests/unit/adjlist.c (test_simple_trees).
    let g = Graph::kary_tree(42, 3, TreeMode::Out).unwrap();

    let out = g
        .adjlist_init(NeighborMode::Out, Loops::Once, true)
        .unwrap();
    let g2 = Graph::adjlist(&out, NeighborMode::Out, false).unwrap();
    assert!(g2.is_directed());
    assert_eq!(sorted_edges(&g2), sorted_edges(&g));

    let inn = g.adjlist_init(NeighborMode::In, Loops::Once, true).unwrap();
    let g3 = inn.to_graph(NeighborMode::In, false).unwrap();
    assert_eq!(sorted_edges(&g3), sorted_edges(&g));

    let u = Graph::kary_tree(42, 3, TreeMode::Undirected).unwrap();
    let all = u
        .adjlist_init(NeighborMode::Out, Loops::Twice, true)
        .unwrap();
    let u2 = all.to_graph(NeighborMode::All, true).unwrap();
    assert!(!u2.is_directed());
    assert_eq!(sorted_edges(&u2), sorted_edges(&u));
    // A tree on 42 vertices has 41 edges and every child's in-list is its
    // parent: (c - 1) / 3.
    assert_eq!(g.ecount(), 41);
    for c in 1..42 {
        assert_eq!(&inn[c], &[(c as i64 - 1) / 3]);
    }
}

#[test]
fn multigraph_with_loops_roundtrips() {
    let g = multi_graph(false);
    let al = g
        .adjlist_init(NeighborMode::All, Loops::Twice, true)
        .unwrap();
    let h = al.to_graph(NeighborMode::All, true).unwrap();
    assert_eq!(h.ecount(), g.ecount());
    assert_eq!(sorted_edges(&h), sorted_edges(&g));
    // Listing each edge once ("half" list: smaller neighbors only, and each
    // loop once instead of twice) also works, with duplicate = false.
    let half: AdjList = al
        .iter()
        .enumerate()
        .map(|(u, list)| {
            let u = u as i64;
            let loops = list.iter().filter(|&&x| x == u).count() / 2;
            list.iter()
                .copied()
                .filter(|&x| x < u)
                .chain(std::iter::repeat_n(u, loops))
                .collect()
        })
        .collect();
    let h2 = half.to_graph(NeighborMode::All, false).unwrap();
    assert_eq!(sorted_edges(&h2), sorted_edges(&g));
}

#[test]
fn karate_roundtrip_and_neighbors() {
    let g = zachary();
    let al = AdjList::new(&g, NeighborMode::All, Loops::Twice, true).unwrap();
    assert_eq!(al.len(), 34);
    // Handshake lemma.
    assert_eq!(al.total_len(), 2 * 78);
    // The instructor (0) and the administrator (33) are the hubs.
    let degrees: Vec<usize> = al.iter().map(<[i64]>::len).collect();
    assert_eq!(degrees[0], 16);
    assert_eq!(degrees[33], 17);
    for u in g.vertices() {
        assert_eq!(
            &al[u as usize],
            g.neighbors(u, NeighborMode::All).unwrap().as_slice()
        );
        assert!(al[u as usize].windows(2).all(|w| w[0] <= w[1]));
    }
    let h = al.to_graph(NeighborMode::All, true).unwrap();
    assert_eq!(sorted_edges(&h), sorted_edges(&g));
    assert_eq!(
        h.adjlist_init(NeighborMode::All, Loops::Twice, true)
            .unwrap(),
        al
    );
}

#[test]
fn to_graph_errors() {
    // Neighbor ids out of range.
    let al = AdjList::from(vec![vec![1], vec![7]]);
    assert_eq!(
        al.to_graph(NeighborMode::Out, false).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
    // Edges not duplicated as promised.
    let al = AdjList::from(vec![vec![1, 2], vec![], vec![]]);
    assert_eq!(
        al.to_graph(NeighborMode::All, true).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    // ... but fine when each edge is listed once.
    assert_eq!(al.to_graph(NeighborMode::All, false).unwrap().ecount(), 2);
}

#[test]
fn to_graph_rejects_badly_duplicated_lists() {
    // Plain igraph would turn this into the single spurious loop (0, 0):
    // the edge {0, 1} is listed twice by 1 but never by 0.
    let al = AdjList::from(vec![vec![], vec![0, 0], vec![]]);
    let err = al.to_graph(NeighborMode::All, true).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    assert!(err.message().contains("{0, 1}"), "{}", err.message());
    // ... and would silently drop a loop listed only once.
    let al = AdjList::from(vec![vec![0]]);
    assert_eq!(
        al.to_graph(NeighborMode::All, true).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    // Multiplicities must agree on both sides.
    let al = AdjList::from(vec![vec![1, 1], vec![0]]);
    assert!(al.to_graph(NeighborMode::All, true).is_err());
    // A correctly duplicated multigraph with a double loop is accepted.
    let al = AdjList::from(vec![vec![1, 1], vec![0, 0, 1, 1, 1, 1]]);
    let g = al.to_graph(NeighborMode::All, true).unwrap();
    assert_eq!(sorted_edges(&g), vec![(0, 1), (0, 1), (1, 1), (1, 1)]);
    // For directed graphs `duplicate` is ignored, so nothing is checked.
    let al = AdjList::from(vec![vec![], vec![0, 0], vec![]]);
    let g = al.to_graph(NeighborMode::In, true).unwrap();
    assert!(g.is_directed());
    assert_eq!(sorted_edges(&g), vec![(0, 1), (0, 1)]);
}

#[test]
fn init_empty_rejects_impossible_sizes() {
    assert_eq!(
        AdjList::init_empty(usize::MAX).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        IncList::init_empty(usize::MAX).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    let il = IncList::init_empty(0).unwrap();
    assert!(il.is_empty());
    assert_eq!(il.to_string(), "");
}

// ---------------------------------------------------------------------------
// has_edge / replace_edge
// ---------------------------------------------------------------------------

#[test]
fn has_edge_directed_and_undirected() {
    let g = zachary();
    let al = g
        .adjlist_init(NeighborMode::All, Loops::Twice, true)
        .unwrap();
    for u in 0..34 {
        for w in 0..34 {
            let adjacent = g.get_eid(u, w, false).unwrap().is_some();
            assert_eq!(al.has_edge(u, w, false).unwrap(), adjacent);
            assert_eq!(al.has_edge(u, w, true).unwrap(), adjacent);
        }
    }
    let d = Graph::from_edges(&[(0, 1), (2, 1)], 3, true).unwrap();
    let out = d
        .adjlist_init(NeighborMode::Out, Loops::Once, true)
        .unwrap();
    assert!(out.has_edge(2, 1, true).unwrap());
    assert!(!out.has_edge(1, 2, true).unwrap());
    // Undirected lookup reads the list of the larger endpoint: 2 -> 1 is there.
    assert!(out.has_edge(1, 2, false).unwrap());
    assert_eq!(
        out.has_edge(-1, 0, true).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        out.has_edge(0, 3, false).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
}

#[test]
fn replace_edge_keeps_lists_sorted() {
    let mut al = AdjList::from(vec![vec![1, 3, 5], vec![], vec![], vec![], vec![], vec![]]);
    al.replace_edge(0, 5, 2, true).unwrap();
    assert_eq!(&al[0], &[1, 2, 3]);
    al.replace_edge(0, 1, 4, true).unwrap();
    assert_eq!(&al[0], &[2, 3, 4]);

    let err = al.replace_edge(0, 1, 5, true).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    assert!(err.message().contains("does not exist"));
    let err = al.replace_edge(0, 2, 3, true).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    assert!(err.message().contains("already exists"));
    assert_eq!(
        al.replace_edge(0, 2, 6, true).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
    // Failed replacements leave the list untouched.
    assert_eq!(&al[0], &[2, 3, 4]);

    // Undirected, "half" representation: {u, w} stored in the list of max(u, w).
    let mut half = AdjList::from(vec![vec![], vec![0], vec![1], vec![]]);
    half.replace_edge(1, 2, 3, false).unwrap(); // {1,2} -> {1,3}
    assert_eq!(half.to_vecs(), v(&[&[], &[0], &[], &[1]]));
    assert!(half.has_edge(1, 3, false).unwrap());
    assert!(half.has_edge(3, 1, false).unwrap());
    assert!(!half.has_edge(1, 2, false).unwrap());
}

// ---------------------------------------------------------------------------
// Lazy lists
// ---------------------------------------------------------------------------

#[test]
fn lazy_lists_cache_on_demand() {
    let g = zachary();
    let mut lazy = g
        .lazy_adjlist_init(NeighborMode::All, Loops::Twice, true)
        .unwrap();
    assert_eq!(lazy.len(), 34);
    assert!(!lazy.is_empty());
    assert!((0..34).all(|u| !lazy.has(u)));
    assert_eq!(lazy.get(33).unwrap().len(), 17);
    assert!(lazy.has(33));
    assert!(!lazy.has(0));
    assert!(!lazy.has(34));
    assert!(format!("{lazy:?}").contains("cached: 1"));

    // Modifying the cached vector does not affect the graph...
    lazy.get_mut(33).unwrap().clear();
    assert_eq!(lazy.get(33).unwrap(), &[] as &[i64]);
    assert_eq!(g.neighbors(33, NeighborMode::All).unwrap().len(), 17);
    // ... and clearing the cache brings the true neighbors back.
    lazy.clear();
    assert!(!lazy.has(33));
    assert_eq!(lazy.get(33).unwrap().len(), 17);
    assert!(std::ptr::eq(lazy.graph(), &g));

    assert_eq!(lazy.get(34).unwrap_err().kind(), ErrorKind::InvalidVertexId);
    assert_eq!(lazy.get(-1).unwrap_err().kind(), ErrorKind::InvalidVertexId);

    // Eager and lazy agree for every mode on a directed graph.
    let d = multi_graph(true);
    for mode in [NeighborMode::Out, NeighborMode::In, NeighborMode::All] {
        for loops in [Loops::None, Loops::Once, Loops::Twice] {
            for multiple in [false, true] {
                let eager = d.adjlist_init(mode, loops, multiple).unwrap().to_vecs();
                let mut lazy = LazyAdjList::new(&d, mode, loops, multiple).unwrap();
                assert_eq!(
                    lazy.to_vecs().unwrap(),
                    eager,
                    "{mode:?} {loops:?} {multiple}"
                );
            }
            let eager = d.inclist_init(mode, loops).unwrap().to_vecs();
            let mut lazy = LazyIncList::new(&d, mode, loops).unwrap();
            assert_eq!(lazy.to_vecs().unwrap(), eager);
        }
    }
}

#[test]
fn lazy_inclist_basic() {
    let g = inc_graph(true);
    let mut lazy = g.lazy_inclist_init(NeighborMode::Out, Loops::None).unwrap();
    assert!(!lazy.has(4));
    assert_eq!(lazy.get(4).unwrap(), &[7, 9, 10]);
    assert!(lazy.has(4));
    assert_eq!(lazy.mode(), NeighborMode::Out);
    lazy.get_mut(4).unwrap().push(99);
    assert_eq!(lazy.get(4).unwrap(), &[7, 9, 10, 99]);
    lazy.clear();
    assert_eq!(lazy.get(4).unwrap(), &[7, 9, 10]);
    assert_eq!(lazy.get(7).unwrap_err().kind(), ErrorKind::InvalidVertexId);

    let empty = Graph::new(0, false);
    let mut lazy = empty
        .lazy_inclist_init(NeighborMode::All, Loops::Twice)
        .unwrap();
    assert!(lazy.is_empty());
    assert!(lazy.to_vecs().unwrap().is_empty());
}

#[test]
fn lazy_adjlist_uses_the_property_cache() {
    // Nothing is known about a fresh graph: the arguments are kept.
    let g = cycle(5);
    let lazy = g
        .lazy_adjlist_init(NeighborMode::All, Loops::None, false)
        .unwrap();
    assert_eq!((lazy.loops(), lazy.multiple()), (Loops::None, false));
    drop(lazy);

    // Once igraph has checked that there are no loops, the lazy list skips
    // the loop filtering, reporting the relaxed setting. (Collapsing of
    // multi-edges is always done when neighbors are gathered both ways.)
    assert!(!g.has_loop().unwrap());
    assert!(!g.has_multiple().unwrap());
    let mut lazy = g
        .lazy_adjlist_init(NeighborMode::All, Loops::None, false)
        .unwrap();
    assert_eq!((lazy.loops(), lazy.multiple()), (Loops::Twice, false));
    // ... with the same lists.
    assert_eq!(
        lazy.to_vecs().unwrap(),
        g.adjlist_init(NeighborMode::All, Loops::None, false)
            .unwrap()
            .to_vecs()
    );

    // Directed graph, mode Out: loops are relaxed to Once and, once igraph
    // knows there are no multi-edges, `multiple` to true.
    let d = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    let lazy = d
        .lazy_adjlist_init(NeighborMode::Out, Loops::None, false)
        .unwrap();
    assert_eq!((lazy.loops(), lazy.multiple()), (Loops::None, false));
    drop(lazy);
    assert!(!d.has_loop().unwrap());
    assert!(!d.has_multiple().unwrap());
    let lazy = d
        .lazy_adjlist_init(NeighborMode::Out, Loops::None, false)
        .unwrap();
    assert_eq!((lazy.loops(), lazy.multiple()), (Loops::Once, true));
    assert_eq!(lazy.mode(), NeighborMode::Out);

    // A graph with loops keeps the requested handling.
    let l = loop_graph(true);
    assert!(l.has_loop().unwrap());
    assert!(l.has_multiple().unwrap());
    let lazy = l
        .lazy_adjlist_init(NeighborMode::In, Loops::None, false)
        .unwrap();
    assert_eq!((lazy.loops(), lazy.multiple()), (Loops::None, false));
    // The lazy incidence list always reports the requested loop handling.
    let lazy = l.lazy_inclist_init(NeighborMode::In, Loops::Once).unwrap();
    assert_eq!(lazy.loops(), Loops::Once);
    assert!(format!("{lazy:?}").contains("LazyIncList"));
}

// ---------------------------------------------------------------------------
// Workaround for the multi-edge collapsing bug of igraph 1.0.0 and 1.0.1
// ---------------------------------------------------------------------------

/// A directed graph with a mutual pair 0 <-> 1 and no multi-edges.
fn mutual() -> Graph {
    Graph::from_edges(&[(0, 1), (1, 0), (1, 2)], 3, true).unwrap()
}

#[test]
fn mutual_pairs_are_collapsed_and_do_not_poison_the_cache() {
    // igraph 1.0.0 and 1.0.1 would record "has multi-edges" here, making
    // has_multiple() return true and is_simple() false afterwards.
    let g = mutual();
    let al = g
        .adjlist_init(NeighborMode::All, Loops::None, false)
        .unwrap();
    assert_eq!(al.to_vecs(), v(&[&[1], &[0, 2], &[1]]));
    assert!(!g.has_multiple().unwrap());
    assert!(g.is_simple(true).unwrap());
    assert_eq!(g.count_multiple(..).unwrap(), vec![1, 1, 1]);

    // With the cache saying "no multi-edges", igraph would list the mutual
    // pair twice; the result must not depend on the cache.
    let mut h = mutual();
    h.simplify(true, true).unwrap();
    for loops in [Loops::None, Loops::Once, Loops::Twice] {
        assert_eq!(
            h.adjlist_init(NeighborMode::All, loops, false)
                .unwrap()
                .to_vecs(),
            v(&[&[1], &[0, 2], &[1]]),
            "{loops:?}"
        );
        let mut lazy = h
            .lazy_adjlist_init(NeighborMode::All, loops, false)
            .unwrap();
        assert!(!lazy.multiple());
        assert_eq!(lazy.to_vecs().unwrap(), v(&[&[1], &[0, 2], &[1]]));
    }
    // Keeping multi-edges still lists the pair twice, as it should.
    assert_eq!(
        h.adjlist_init(NeighborMode::All, Loops::None, true)
            .unwrap()
            .to_vecs(),
        v(&[&[1, 1], &[0, 0, 2], &[1]])
    );
    // Same answer as the per-vertex query.
    assert_eq!(
        h.neighbors_with(1, NeighborMode::All, Loops::None, false)
            .unwrap(),
        vec![0, 2]
    );
}

#[test]
fn single_self_loops_are_not_multi_edges() {
    // An undirected self-loop is listed twice with Loops::Twice; igraph 1.0.0
    // and 1.0.1 took that for a multi-edge and cached it.
    let g = Graph::from_edges(&[(0, 0), (0, 1)], 2, false).unwrap();
    let al = g
        .adjlist_init(NeighborMode::All, Loops::Twice, false)
        .unwrap();
    assert_eq!(al.to_vecs(), v(&[&[0, 0, 1], &[0]]));
    assert!(!g.has_multiple().unwrap());
    assert!(g.has_loop().unwrap());

    // A genuine multigraph: a double loop and a double edge.
    let m = Graph::from_edges(&[(0, 0), (0, 0), (0, 1), (0, 1)], 2, false).unwrap();
    let expected = [
        (Loops::None, v(&[&[1], &[0]])),
        (Loops::Once, v(&[&[0, 1], &[0]])),
        (Loops::Twice, v(&[&[0, 0, 1], &[0]])),
    ];
    for (loops, lists) in expected {
        assert_eq!(
            m.adjlist_init(NeighborMode::All, loops, false)
                .unwrap()
                .to_vecs(),
            lists,
            "{loops:?}"
        );
        let mut lazy = m
            .lazy_adjlist_init(NeighborMode::All, loops, false)
            .unwrap();
        assert_eq!(lazy.to_vecs().unwrap(), lists, "lazy {loops:?}");
        for u in 0..2 {
            assert_eq!(
                &lists[u as usize],
                &m.neighbors_with(u, NeighborMode::All, loops, false)
                    .unwrap()
            );
        }
    }
    assert!(m.has_multiple().unwrap());

    // Directed, mode Out: a double loop is listed once at most.
    let d = Graph::from_edges(&[(0, 0), (0, 0), (0, 1)], 2, true).unwrap();
    for loops in [Loops::Once, Loops::Twice] {
        assert_eq!(
            d.adjlist_init(NeighborMode::Out, loops, false)
                .unwrap()
                .to_vecs(),
            v(&[&[0, 1], &[]])
        );
    }
    // Directed, mode All: twice with Loops::Twice, once with Loops::Once.
    assert_eq!(
        d.adjlist_init(NeighborMode::All, Loops::Twice, false)
            .unwrap()
            .to_vecs(),
        v(&[&[0, 0, 1], &[0]])
    );
    assert_eq!(
        d.adjlist_init(NeighborMode::All, Loops::Once, false)
            .unwrap()
            .to_vecs(),
        v(&[&[0, 1], &[0]])
    );
}

// ---------------------------------------------------------------------------
// Rusty container behaviour
// ---------------------------------------------------------------------------

#[test]
fn container_traits() {
    let nested = vec![vec![1, 2], vec![0], vec![0, 0, 1]];
    let mut al: AdjList = nested.clone().into_iter().collect();
    assert_eq!(al.len(), 3);
    assert_eq!(Vec::<Vec<i64>>::from(&al), nested);
    assert_eq!(al.get(2), Some(&[0, 0, 1][..]));
    assert_eq!(al.get(3), None);
    assert_eq!(al.get(-1), None);
    assert!(al.get_mut(3).is_none());

    let copy = al.clone();
    al[0][1] = 0;
    al.get_mut(1).unwrap().push(2);
    assert_ne!(al, copy);
    assert_eq!(copy.to_vecs(), nested);
    assert_eq!(al.to_vecs(), v(&[&[1, 0], &[0, 2], &[0, 0, 1]]));

    // Iteration by reference, both ways.
    let lens: Vec<usize> = (&al).into_iter().map(|l| l.len()).collect();
    assert_eq!(lens, vec![2, 2, 3]);
    assert_eq!(al.iter().next_back().unwrap(), &[0, 0, 1]);
    assert_eq!(al.iter().len(), 3);
    for list in al.iter_mut() {
        list.sort();
    }
    assert_eq!(al.to_vecs(), v(&[&[0, 1], &[0, 2], &[0, 0, 1]]));

    // Display is igraph's print format, and matches the C fprint output.
    assert_eq!(al.to_string(), "0 1\n0 2\n0 0 1\n");
    let mut buf = Vec::new();
    al.fprint(&mut buf).unwrap();
    assert_eq!(String::from_utf8(buf).unwrap(), al.to_string());
    al.print().unwrap();

    al.clear();
    assert_eq!(al.len(), 3);
    assert_eq!(al.total_len(), 0);

    let empty = AdjList::default();
    assert!(empty.is_empty());
    assert_eq!(empty.iter().count(), 0);
    let e = AdjList::init_empty(4).unwrap();
    assert_eq!(e.to_vecs(), vec![Vec::<i64>::new(); 4]);
    let il = IncList::init_empty(2).unwrap();
    assert_eq!(il, IncList::from(vec![vec![], vec![]]));
    assert_eq!(Vec::<Vec<i64>>::from(il), vec![Vec::<i64>::new(); 2]);
}

#[test]
fn inclist_container_traits() {
    let g = path(4);
    let mut il = g.inclist_init(NeighborMode::All, Loops::Twice).unwrap();
    assert_eq!(il.as_raw_slice().len(), 4);
    assert_eq!(il.get(1), Some(&[0, 1][..]));
    assert_eq!(il.get(4), None);
    // Reverse each list in place; the copy is unaffected.
    let copy = il.clone();
    for list in il.iter_mut() {
        list.reverse();
    }
    assert_eq!(il.to_vecs(), v(&[&[0], &[1, 0], &[2, 1], &[2]]));
    assert_eq!(copy.to_vecs(), v(&[&[0], &[0, 1], &[1, 2], &[2]]));
    il[1][0] = 7;
    il.get_mut(3).unwrap().push(9);
    assert_eq!(il.to_string(), "0\n7 0\n2 1\n2 9\n");
    let mut buf = Vec::new();
    il.fprint(&mut buf).unwrap();
    assert_eq!(String::from_utf8(buf).unwrap(), il.to_string());
    il.print().unwrap();
    let collected: IncList = il.iter().map(<[i64]>::to_vec).collect();
    assert_eq!(collected, il);
    il.clear();
    assert_eq!((il.len(), il.total_len()), (4, 0));
    assert!(IncList::default().is_empty());
}

#[test]
fn lists_are_independent_of_the_graph_and_send() {
    let mut g = cycle(5);
    let al = g
        .adjlist_init(NeighborMode::All, Loops::Twice, true)
        .unwrap();
    let il = g.inclist_init(NeighborMode::All, Loops::Twice).unwrap();
    g.add_edge(0, 2).unwrap();
    drop(g);
    // Move them to another thread and use them there.
    let handle = std::thread::spawn(move || (al.to_vecs(), il.total_len()));
    let (lists, total) = handle.join().unwrap();
    assert_eq!(lists, v(&[&[1, 4], &[0, 2], &[1, 3], &[2, 4], &[0, 3]]));
    assert_eq!(total, 10);
}

#[test]
fn input_validation_errors() {
    let g = path(4);
    let il = IncList::from(vec![vec![0], vec![0, 1]]);
    assert_eq!(
        g.adjlist_init_from_inclist(&il).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    let il = IncList::from(vec![vec![0], vec![0, 1], vec![1, 5], vec![2]]);
    assert_eq!(
        g.adjlist_init_from_inclist(&il).unwrap_err().kind(),
        ErrorKind::InvalidEdgeId
    );

    let mut al = AdjList::from(vec![vec![1], vec![2]]);
    assert_eq!(
        al.simplify().unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
    let mut al = AdjList::from(vec![vec![-1]]);
    assert_eq!(
        al.simplify().unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
}

// ---------------------------------------------------------------------------
// Use cases
// ---------------------------------------------------------------------------

/// Story: the karate club's instructor (vertex 0) wants to know how far the
/// news of the split travels. A breadth-first search over a *lazy* adjacency
/// list touches every vertex's neighbors exactly once; everybody is within
/// three hops of the instructor.
#[test]
fn use_case_bfs_over_lazy_adjlist_in_the_karate_club() {
    let g = zachary();
    let mut lazy = g
        .lazy_adjlist_init(NeighborMode::All, Loops::None, false)
        .unwrap();
    let mut dist = vec![-1i64; lazy.len()];
    let mut queue = VecDeque::from([0i64]);
    dist[0] = 0;
    while let Some(u) = queue.pop_front() {
        let neighbors = lazy.get(u).unwrap().to_vec();
        for w in neighbors {
            if dist[w as usize] < 0 {
                dist[w as usize] = dist[u as usize] + 1;
                queue.push_back(w);
            }
        }
    }
    assert!(dist.iter().all(|&d| d >= 0), "the karate club is connected");
    assert_eq!(
        *dist.iter().max().unwrap(),
        3,
        "eccentricity of the instructor"
    );
    assert_eq!(
        dist.iter().filter(|&&d| d == 1).count(),
        16,
        "direct friends"
    );
    assert_eq!(
        dist[33], 2,
        "instructor and administrator have common friends"
    );
    assert!((0..34).all(|u| lazy.has(u)));

    // The hand-written BFS agrees with the library's traversal and with the
    // eccentricity computed by the `paths` module.
    let bfs = g.bfs_simple(0, NeighborMode::All).unwrap();
    assert_eq!(bfs.layers.len() - 1, 4, "layers at distance 0, 1, 2, 3");
    for (d, w) in bfs.layers.windows(2).enumerate() {
        for &u in &bfs.order[w[0]..w[1]] {
            assert_eq!(dist[u as usize], d as i64);
        }
    }
    assert_eq!(
        g.eccentricity(0, None, NeighborMode::All).unwrap(),
        vec![3.0]
    );
}

/// Story: counting triangles by merging sorted neighbor lists. Zachary's
/// karate club has 45 triangles.
#[test]
fn use_case_triangle_counting_with_sorted_adjlists() {
    let g = zachary();
    let al = g
        .adjlist_init(NeighborMode::All, Loops::None, false)
        .unwrap();
    let mut triangles = 0;
    for u in 0..al.len() {
        for &w in al[u].iter().filter(|&&w| w > u as i64) {
            // Common neighbors x > w, found by merging two sorted lists.
            let (a, b) = (&al[u], &al[w as usize]);
            let (mut i, mut j) = (0, 0);
            while i < a.len() && j < b.len() {
                match a[i].cmp(&b[j]) {
                    std::cmp::Ordering::Less => i += 1,
                    std::cmp::Ordering::Greater => j += 1,
                    std::cmp::Ordering::Equal => {
                        if a[i] > w {
                            triangles += 1;
                        }
                        i += 1;
                        j += 1;
                    }
                }
            }
        }
    }
    assert_eq!(triangles, 45);
    // Same as the library's triangle counting.
    assert_eq!(g.count_triangles().unwrap(), 45.0);
    assert_eq!(g.list_triangles().unwrap().len(), 45);
}

/// Degree-preserving rewiring on a sorted out-adjacency list, the way
/// igraph's `igraph_rewire` does it: random pairs of edges `a -> b`,
/// `c -> d` are swapped into `a -> d`, `c -> b` whenever this creates no loop
/// and no multi-edge (checked with `has_edge`), updating the lists with
/// `replace_edge`. Uses the calling thread's default RNG. Returns the
/// rewired graph and the number of successful swaps.
fn rewire_with_adjlist(g: &Graph, trials: usize) -> (Graph, usize) {
    let mut al = g
        .adjlist_init(NeighborMode::Out, Loops::Once, true)
        .unwrap();
    let mut edge_list = g.edge_list();
    let mut swaps = 0;
    for _ in 0..trials {
        let i = rng::integer(0, edge_list.len() as i64 - 1) as usize;
        let j = rng::integer(0, edge_list.len() as i64 - 1) as usize;
        let ((a, b), (c, d)) = (edge_list[i], edge_list[j]);
        if i == j || a == c || b == d || a == d || c == b {
            continue;
        }
        if al.has_edge(a, d, true).unwrap() || al.has_edge(c, b, true).unwrap() {
            continue;
        }
        al.replace_edge(a, b, d, true).unwrap();
        al.replace_edge(c, d, b, true).unwrap();
        edge_list[i] = (a, d);
        edge_list[j] = (c, b);
        swaps += 1;
    }
    let h = al.to_graph(NeighborMode::Out, false).unwrap();
    let mut rewired = edge_list;
    rewired.sort();
    assert_eq!(sorted_edges(&h), rewired, "lists and edge list agree");
    (h, swaps)
}

/// A directed circulant graph: every vertex points to the next 1, 2 and 7.
fn circulant(n: i64) -> Graph {
    let edges: Vec<(i64, i64)> = (0..n)
        .flat_map(|i| [(i, (i + 1) % n), (i, (i + 2) % n), (i, (i + 7) % n)])
        .collect();
    Graph::from_edges(&edges, n as usize, true).unwrap()
}

/// Story: degree-preserving rewiring by hand. A directed network is loaded
/// into a sorted out-adjacency list and rewired with `has_edge` /
/// `replace_edge`. The rebuilt graph keeps every in- and out-degree and stays
/// simple, yet its edge set has changed -- just like the library's own
/// `Graph::rewire`.
#[test]
fn use_case_degree_preserving_rewiring() {
    rng::seed(42).unwrap();
    let g = circulant(30);
    let (h, swaps) = rewire_with_adjlist(&g, 2000);
    assert!(swaps > 100, "only {swaps} swaps");

    assert_eq!(h.ecount(), g.ecount());
    for mode in [NeighborMode::Out, NeighborMode::In] {
        assert_eq!(
            h.degree(.., mode, Loops::Twice).unwrap(),
            g.degree(.., mode, Loops::Twice).unwrap()
        );
    }
    // Still simple, and actually rewired.
    assert!(h.is_simple(true).unwrap());
    let mut check = h
        .adjlist_init(NeighborMode::Out, Loops::Twice, true)
        .unwrap();
    let before = check.clone();
    check.simplify().unwrap();
    check.sort();
    assert_eq!(check, before);
    assert_ne!(sorted_edges(&h), sorted_edges(&g));

    // The library's rewiring gives a graph with the same guarantees.
    let mut lib = g.clone();
    lib.rewire(2000, EdgeTypeSw::Simple).unwrap();
    assert!(lib.is_simple(true).unwrap());
    for mode in [NeighborMode::Out, NeighborMode::In] {
        assert_eq!(
            lib.degree(.., mode, Loops::Twice).unwrap(),
            h.degree(.., mode, Loops::Twice).unwrap()
        );
    }
}

/// Seeding is per thread: the same seed gives the same rewiring in parallel
/// threads, and each thread's RNG is unaffected by the others.
#[test]
fn seeded_rewiring_is_reproducible_across_threads() {
    let run = |seed: u64| {
        rng::seed(seed).unwrap();
        let (h, swaps) = rewire_with_adjlist(&circulant(30), 500);
        (sorted_edges(&h), swaps)
    };
    let handles: Vec<_> = [7u64, 7, 8, 7]
        .into_iter()
        .map(|seed| std::thread::spawn(move || run(seed)))
        .collect();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(results[0], results[1]);
    assert_eq!(results[0], results[3]);
    assert_ne!(
        results[0].0, results[2].0,
        "different seeds, different graphs"
    );
    // And the main thread reproduces them too.
    assert_eq!(run(7), results[0]);
}

//! Integration tests for the `components` module: connected components,
//! articulation points and bridges, biconnected components, percolation,
//! separators, cohesive blocks, reachability and neighborhoods.

mod common;

use common::*;
use igraph::components::{BondPercolation, SitePercolation, edgelist_percolation};
use igraph::prelude::*;

fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
    v.sort();
    v
}

/// Sorts each inner list, then the outer list, for order-independent checks.
fn canonical(v: Vec<Vec<i64>>) -> Vec<Vec<i64>> {
    sorted(v.into_iter().map(sorted).collect())
}

fn directed(edges: &[(i64, i64)], n: usize) -> Graph {
    Graph::from_edges(edges, n, true).unwrap()
}

fn undirected(edges: &[(i64, i64)], n: usize) -> Graph {
    Graph::from_edges(edges, n, false).unwrap()
}

// ---------------------------------------------------------------------------
// Connected components
// ---------------------------------------------------------------------------

#[test]
fn components_of_disjoint_union() {
    // A triangle, a path of 3 vertices, and 2 isolated vertices.
    let g = undirected(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5)], 8);
    let cc = g.connected_components(Connectedness::Weak).unwrap();
    assert_eq!(cc.count, 4);
    assert_eq!(cc.membership, vec![0, 0, 0, 1, 1, 1, 2, 3]);
    assert_eq!(cc.sizes, vec![3, 3, 1, 1]);
    assert_eq!(cc.sizes.iter().sum::<usize>(), g.vcount());
    assert_eq!(cc.members(1), vec![3, 4, 5]);
    assert_eq!(
        cc.groups(),
        vec![vec![0, 1, 2], vec![3, 4, 5], vec![6], vec![7]]
    );
    assert_eq!(cc.largest(), Some(0));
    assert!(cc.same_component(3, 5));
    assert!(!cc.same_component(2, 3));
    assert!(!cc.same_component(0, 100));
}

#[test]
fn strong_components_are_topologically_ordered() {
    // Two directed 3-cycles linked by 2 -> 3, and a sink 6 reached from 5.
    let g = directed(
        &[
            (0, 1),
            (1, 2),
            (2, 0),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 3),
            (5, 6),
        ],
        7,
    );
    let weak = g.connected_components(Connectedness::Weak).unwrap();
    assert_eq!(weak.count, 1);
    let strong = g.connected_components(Connectedness::Strong).unwrap();
    assert_eq!(strong.count, 3);
    assert_eq!(sorted(strong.sizes.clone()), vec![1, 3, 3]);
    // u reaches v only if membership[u] <= membership[v].
    let m = &strong.membership;
    assert!(m[0] <= m[3] && m[3] <= m[6]);
    assert!(m[0] < m[3]);
    assert_eq!(m[0], m[1]);
    assert_eq!(m[3], m[5]);
}

#[test]
fn connectedness_edge_cases() {
    // The null graph is not connected by definition; the singleton is.
    assert!(
        !Graph::new(0, false)
            .is_connected(Connectedness::Weak)
            .unwrap()
    );
    assert!(
        Graph::new(1, false)
            .is_connected(Connectedness::Weak)
            .unwrap()
    );
    assert_eq!(
        Graph::new(0, true)
            .connected_components(Connectedness::Strong)
            .unwrap()
            .count,
        0
    );
    assert_eq!(
        Graph::new(0, true)
            .connected_components(Connectedness::Weak)
            .unwrap()
            .largest(),
        None
    );

    assert!(karate().is_connected(Connectedness::Weak).unwrap());
    assert!(cycle(10).is_connected(Connectedness::Strong).unwrap());

    // A directed cycle is strongly connected, a directed path only weakly.
    let dcycle = directed(&[(0, 1), (1, 2), (2, 0)], 3);
    assert!(dcycle.is_connected(Connectedness::Strong).unwrap());
    let dpath = directed(&[(0, 1), (1, 2)], 3);
    assert!(dpath.is_connected(Connectedness::Weak).unwrap());
    assert!(!dpath.is_connected(Connectedness::Strong).unwrap());
    // Cached result stays consistent.
    assert!(!dpath.is_connected(Connectedness::Strong).unwrap());
}

#[test]
fn decompose_like_igraph_example() {
    // examples/simple/igraph_decompose.c
    let ring = cycle(10);
    let parts = ring.decompose(Connectedness::Weak, None, 0).unwrap();
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0].vcount(), 10);
    assert_eq!(parts[0].ecount(), 10);

    let g = directed(
        &[
            (0, 1),
            (1, 2),
            (2, 0),
            (3, 4),
            (4, 5),
            (5, 6),
            (8, 9),
            (9, 10),
        ],
        0,
    );
    let parts = g.decompose(Connectedness::Weak, Some(3), 2).unwrap();
    let lists: Vec<_> = parts.iter().map(|p| p.edge_list()).collect();
    assert_eq!(
        lists,
        vec![
            vec![(0, 1), (1, 2), (2, 0)],
            vec![(0, 1), (1, 2), (2, 3)],
            vec![(0, 1), (1, 2)]
        ]
    );
    assert!(parts.iter().all(|p| p.is_directed()));

    // Isolated vertex 7 is its own component: kept with min_vertices = 1.
    assert_eq!(g.decompose(Connectedness::Weak, None, 1).unwrap().len(), 4);
    assert_eq!(
        g.decompose(Connectedness::Weak, Some(1), 0).unwrap().len(),
        1
    );
    // Strong components of the first cycle stay together.
    let strong = g.decompose(Connectedness::Strong, None, 2).unwrap();
    assert_eq!(strong.len(), 1);
    assert_eq!(strong[0].ecount(), 3);
}

#[test]
fn decomposition_preserves_counts() {
    rng::seed(42).unwrap();
    let mut edges = vec![];
    // Three disjoint cliques of sizes 3, 4, 5 in an interleaved order.
    let blocks: [&[i64]; 3] = [&[0, 3, 6], &[1, 4, 7, 9], &[2, 5, 8, 10, 11]];
    for b in blocks {
        for i in 0..b.len() {
            for j in i + 1..b.len() {
                edges.push((b[i], b[j]));
            }
        }
    }
    let g = undirected(&edges, 12);
    let parts = g.decompose(Connectedness::Weak, None, 0).unwrap();
    let mut shapes: Vec<_> = parts.iter().map(|p| (p.vcount(), p.ecount())).collect();
    shapes.sort();
    assert_eq!(shapes, vec![(3, 3), (4, 6), (5, 10)]);
    assert_eq!(parts.iter().map(|p| p.ecount()).sum::<usize>(), g.ecount());
}

// ---------------------------------------------------------------------------
// Articulation points, bridges, biconnectivity
// ---------------------------------------------------------------------------

#[test]
fn karate_cut_vertex_and_bridge() {
    let g = karate();
    // Member 11 only knows the instructor (0): 0 is the only cut vertex and
    // the edge 0-11 the only bridge.
    assert_eq!(g.articulation_points().unwrap(), vec![0]);
    let bridges = g.bridges().unwrap();
    assert_eq!(bridges.len(), 1);
    assert_eq!(g.edge(bridges[0]).unwrap(), (0, 11));
    assert!(!g.is_biconnected().unwrap());
}

#[test]
fn trees_are_all_bridges() {
    // In a tree every edge is a bridge and every internal vertex is a cut vertex.
    // The complete binary tree 0-{1,2}, 1-{3,4}, 2-{5,6}.
    let tree = Graph::kary_tree(7, 2, TreeMode::Undirected).unwrap();
    assert!(tree.is_tree(NeighborMode::All).unwrap());
    assert_eq!(sorted(tree.bridges().unwrap()), (0..6).collect::<Vec<_>>());
    assert_eq!(sorted(tree.articulation_points().unwrap()), vec![0, 1, 2]);
    let bc = tree.biconnected_components().unwrap();
    assert_eq!(bc.count, 6);
    assert!(bc.components.iter().all(|c| c.len() == 2));

    // A cycle has neither.
    let c = Graph::ring(6, false, false, true).unwrap();
    assert!(c.bridges().unwrap().is_empty());
    assert!(c.articulation_points().unwrap().is_empty());
    assert!(c.is_biconnected().unwrap());
}

#[test]
fn biconnected_components_like_igraph_example() {
    // examples/simple/igraph_biconnected_components.c (+ .out)
    let g = undirected(
        &[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (2, 4),
            (4, 5),
            (5, 2),
            (0, 6),
            (0, 7),
        ],
        7,
    );
    let bc = g.biconnected_components().unwrap();
    assert_eq!(bc.count, 4);
    assert_eq!(bc.components.len(), 4);
    assert_eq!(bc.component_edges.len(), 4);
    assert_eq!(bc.tree_edges.len(), 4);
    assert_eq!(
        canonical(bc.components.clone()),
        vec![vec![0, 1, 2, 3], vec![0, 6], vec![0, 7], vec![2, 4, 5]]
    );
    let edge_sets: Vec<Vec<(i64, i64)>> = bc
        .component_edges
        .iter()
        .map(|es| {
            sorted(
                es.iter()
                    .map(|&e| g.edge(e).unwrap())
                    .map(|(a, b)| (a.min(b), a.max(b)))
                    .collect(),
            )
        })
        .collect();
    assert_eq!(
        sorted(edge_sets),
        vec![
            vec![(0, 1), (0, 3), (1, 2), (2, 3)],
            vec![(0, 6)],
            vec![(0, 7)],
            vec![(2, 4), (2, 5), (4, 5)],
        ]
    );
    // The edges of the components partition the edge set.
    let mut all: Vec<i64> = bc.component_edges.iter().flatten().copied().collect();
    all.sort();
    assert_eq!(all, (0..g.ecount() as i64).collect::<Vec<_>>());
    // A spanning tree of a component with k vertices has k - 1 edges.
    for (tree, comp) in bc.tree_edges.iter().zip(&bc.components) {
        assert_eq!(tree.len(), comp.len() - 1);
    }
    assert_eq!(sorted(bc.articulation_points.clone()), vec![0, 2]);
    assert_eq!(sorted(g.articulation_points().unwrap()), vec![0, 2]);
}

#[test]
fn biconnectivity_conventions() {
    // igraph: K2 is biconnected, the singleton and null graphs are not.
    assert!(path(2).is_biconnected().unwrap());
    assert!(!Graph::new(1, false).is_biconnected().unwrap());
    assert!(!Graph::new(0, false).is_biconnected().unwrap());
    // No articulation points does not imply biconnected.
    let two_isolated = Graph::new(2, false);
    assert!(two_isolated.articulation_points().unwrap().is_empty());
    assert!(!two_isolated.is_biconnected().unwrap());
    assert_eq!(two_isolated.biconnected_components().unwrap().count, 0);
    // Complete graphs are biconnected; directions are ignored.
    assert!(complete(5).is_biconnected().unwrap());
    assert!(
        directed(&[(0, 1), (1, 2), (2, 0)], 3)
            .is_biconnected()
            .unwrap()
    );
}

// ---------------------------------------------------------------------------
// Percolation
// ---------------------------------------------------------------------------

#[test]
fn bond_percolation_values_from_igraph_tests() {
    // tests/unit/percolation.out
    let c4 = cycle(4);
    let p = c4.bond_percolation(Some(&[0, 2, 1, 3])).unwrap();
    assert_eq!(p.giant_size, vec![2, 2, 4, 4]);
    assert_eq!(p.vertex_count, vec![2, 4, 4, 4]);

    let order: Vec<i64> = (0..78).collect();
    let p = karate().bond_percolation(Some(&order)).unwrap();
    assert_eq!(p.giant_size.len(), 78);
    assert_eq!(&p.giant_size[..16], &(2..=17).collect::<Vec<usize>>()[..]);
    assert_eq!(*p.giant_size.last().unwrap(), 34);
    assert_eq!(*p.vertex_count.last().unwrap(), 34);

    // Null and singleton graphs give empty curves.
    let empty = Graph::new(1, false).bond_percolation(None).unwrap();
    assert!(empty.giant_size.is_empty() && empty.vertex_count.is_empty());
}

#[test]
fn random_bond_percolation_is_monotone() {
    rng::seed(42).unwrap();
    let g = complete(3);
    let p = g.bond_percolation(None).unwrap();
    // K3: whatever the order, the curve is 2 3 3.
    assert_eq!(p.giant_size, vec![2, 3, 3]);
    assert_eq!(p.vertex_count, vec![2, 3, 3]);

    let k = karate();
    let p = k.bond_percolation(None).unwrap();
    assert_eq!(p.giant_size.len(), k.ecount());
    assert!(p.giant_size.windows(2).all(|w| w[0] <= w[1]));
    assert!(p.vertex_count.windows(2).all(|w| w[0] <= w[1]));
    assert!(
        p.giant_size
            .iter()
            .zip(&p.vertex_count)
            .all(|(g, v)| g <= v)
    );
    assert_eq!(*p.giant_size.last().unwrap(), 34);
}

#[test]
fn site_percolation_values_from_igraph_tests() {
    let k5 = complete(5);
    let order: Vec<i64> = (0..5).collect();
    let p = k5.site_percolation(Some(&order)).unwrap();
    assert_eq!(p.giant_size, vec![1, 2, 3, 4, 5]);
    // Edges among the first i vertices of K5 are i choose 2.
    assert_eq!(p.edge_count, vec![0, 1, 3, 6, 10]);

    let p = cycle(4).site_percolation(Some(&[0, 2, 1, 3])).unwrap();
    assert_eq!(p.giant_size, vec![1, 1, 3, 4]);
    assert_eq!(p.edge_count, vec![0, 0, 2, 4]);

    // A partial order ("missing vertices") is fine.
    let p = k5.site_percolation(Some(&[0, 1, 2])).unwrap();
    assert_eq!(p.giant_size, vec![1, 2, 3]);
    assert_eq!(p.edge_count, vec![0, 1, 3]);

    let order: Vec<i64> = (0..34).collect();
    let p = karate().site_percolation(Some(&order)).unwrap();
    assert_eq!(*p.edge_count.last().unwrap(), 78);
    assert_eq!(&p.giant_size[..13], &(1..=13).collect::<Vec<usize>>()[..]);

    let single = Graph::new(1, false).site_percolation(None).unwrap();
    assert_eq!(single.giant_size, vec![1]);
    assert_eq!(single.edge_count, vec![0]);
}

#[test]
fn percolation_errors() {
    let k5 = complete(5);
    assert!(k5.site_percolation(Some(&[0, 1, 1])).is_err());
    assert!(k5.site_percolation(Some(&[0, 7])).is_err());
    assert_eq!(
        k5.bond_percolation(Some(&[0, 0])).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        k5.bond_percolation(Some(&[100])).unwrap_err().kind(),
        ErrorKind::InvalidEdgeId
    );
    assert_eq!(
        k5.bond_percolation(Some(&[-1])).unwrap_err().kind(),
        ErrorKind::InvalidEdgeId
    );
    assert_eq!(
        k5.bond_percolation(Some(&[10])).unwrap_err().kind(),
        ErrorKind::InvalidEdgeId
    );
    assert_eq!(
        k5.site_percolation(Some(&[-3])).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        edgelist_percolation(&[(-1, 1), (0, 0)]).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
}

#[test]
fn bond_percolation_with_partial_order() {
    // Only some edges are added, with ids larger than the order's length.
    // igraph 1.0.0 and 1.0.1 size their duplicate-detection bitset by the
    // order's length, so without the Rust-side validation these calls would
    // access memory out of bounds inside igraph.
    let c4 = cycle(4); // edges 0-1, 1-2, 2-3, 3-0
    let p = c4.bond_percolation(Some(&[3])).unwrap();
    assert_eq!(p.giant_size, vec![2]);
    assert_eq!(p.vertex_count, vec![2]);
    let p = c4.bond_percolation(Some(&[2, 3])).unwrap();
    assert_eq!(p.giant_size, vec![2, 3]);
    assert_eq!(p.vertex_count, vec![2, 3]);
    let p = c4.bond_percolation(Some(&[])).unwrap();
    assert!(p.giant_size.is_empty());
}

#[test]
fn edgelist_percolation_matches_bond_percolation() {
    // tests/unit/percolation.out: K3 with a loop.
    let p = edgelist_percolation(&[(0, 0), (0, 1), (1, 2), (2, 0)]).unwrap();
    assert_eq!(p.giant_size, vec![1, 2, 3, 3]);
    assert_eq!(p.vertex_count, vec![1, 2, 3, 3]);

    // On a graph's own edge list it agrees with bond percolation in storage order.
    let k = karate();
    let by_list = edgelist_percolation(&KARATE_EDGES).unwrap();
    let order: Vec<i64> = k.edge_ids().collect();
    let by_graph = k.bond_percolation(Some(&order)).unwrap();
    assert_eq!(by_list, by_graph);

    assert!(edgelist_percolation(&[]).unwrap().giant_size.is_empty());
}

// ---------------------------------------------------------------------------
// Separators
// ---------------------------------------------------------------------------

#[test]
fn is_separator_like_igraph_example() {
    // tests/unit/igraph_is_separator.c
    let star = Graph::star(10, StarMode::Undirected, 0).unwrap();
    assert!(star.is_separator(0).unwrap());
    assert!(!star.is_separator(6).unwrap());
    assert!(!star.is_separator(1..10).unwrap());
    assert!(!star.is_separator(0..10).unwrap());
    assert!(!star.is_separator(..).unwrap());

    let k = Graph::famous("Zachary").unwrap();
    assert!(k.is_separator(&[32, 33]).unwrap());
    assert!(!k.is_separator(&[8, 9, 19, 30, 31]).unwrap());
    assert!(k.is_separator(vec![0]).unwrap());

    assert_eq!(
        star.is_separator(42).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
}

#[test]
fn minimal_separators() {
    let g = path(5);
    assert!(g.is_minimal_separator(2).unwrap());
    assert!(!g.is_minimal_separator(&[1, 2]).unwrap());
    assert!(!g.is_minimal_separator(0).unwrap());
    let c = cycle(6);
    assert!(!c.is_minimal_separator(0).unwrap());
    assert!(c.is_minimal_separator(&[0, 3]).unwrap());
    assert!(c.is_minimal_separator(&[0, 2]).unwrap());
    assert!(!c.is_minimal_separator(&[0, 2, 4]).unwrap());
}

#[test]
fn all_minimal_st_separators_documented_example() {
    let g = undirected(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 1)], 5);
    let seps = canonical(g.all_minimal_st_separators().unwrap());
    assert_eq!(seps, vec![vec![1], vec![1, 3], vec![2, 4]]);
}

#[test]
fn karate_minimal_st_separators_are_separators() {
    // examples/simple/igraph_minimal_separators.c
    let k = karate();
    let seps = k.all_minimal_st_separators().unwrap();
    assert!(!seps.is_empty());
    for s in &seps {
        assert!(k.is_separator(s).unwrap(), "{s:?} is not a separator");
    }
    // The minimum size separators are among them, and so is {0}.
    assert!(seps.iter().any(|s| s == &vec![0]));
}

#[test]
fn minimum_size_separators_from_igraph_tests() {
    // tests/unit/igraph_minimum_size_separators.c (+ .out)
    let star = undirected(&(1..7).map(|i| (i, 0)).collect::<Vec<_>>(), 7);
    assert_eq!(star.minimum_size_separators().unwrap(), vec![vec![0]]);

    let k32 = undirected(&[(0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4)], 5);
    assert_eq!(
        canonical(k32.minimum_size_separators().unwrap()),
        vec![vec![3, 4]]
    );

    let g = undirected(&[(2, 0), (3, 0), (4, 0), (2, 1), (3, 1), (4, 1)], 5);
    assert_eq!(
        canonical(g.minimum_size_separators().unwrap()),
        vec![vec![0, 1]]
    );

    let mut edges: Vec<(i64, i64)> = [0, 1, 5, 6, 7, 8, 9]
        .iter()
        .flat_map(|&v| [(v, 2), (v, 3)])
        .collect();
    edges.extend([(2, 4), (4, 3)]);
    let g = undirected(&edges, 10);
    assert_eq!(
        canonical(g.minimum_size_separators().unwrap()),
        vec![vec![2, 3]]
    );

    // Complete graphs have no vertex separators.
    assert!(complete(4).minimum_size_separators().unwrap().is_empty());
    // Disconnected graphs neither (by convention).
    assert!(
        Graph::new(3, false)
            .minimum_size_separators()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn minimum_size_separators_of_cycles_and_errors() {
    // Removing any 2 non-adjacent vertices of C_n disconnects it:
    // n(n-3)/2 separators.
    for n in 4..9_i64 {
        let seps = cycle(n).minimum_size_separators().unwrap();
        assert_eq!(seps.len() as i64, n * (n - 3) / 2, "C_{n}");
        assert!(seps.iter().all(|s| s.len() == 2));
    }
    let d = directed(&[(0, 1), (1, 2)], 3);
    assert_eq!(
        d.minimum_size_separators().unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

// ---------------------------------------------------------------------------
// Cohesive blocks
// ---------------------------------------------------------------------------

#[test]
fn karate_cohesive_blocks_like_igraph_example() {
    // examples/simple/cohesive_blocks.c (+ .out)
    let cb = karate().cohesive_blocks().unwrap();
    let expected: Vec<Vec<i64>> = vec![
        (0..34).collect(),
        vec![
            0, 1, 2, 3, 7, 8, 9, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
            29, 30, 31, 32, 33,
        ],
        vec![0, 4, 5, 6, 10, 16],
        vec![0, 1, 2, 3, 7],
        vec![0, 1, 2, 8, 30, 32, 33],
        vec![0, 4, 5, 6, 10],
        vec![0, 1, 2, 3, 13],
        vec![2, 23, 24, 25, 27, 28, 29, 31, 32, 33],
    ];
    assert_eq!(cb.blocks, expected);
    assert_eq!(cb.cohesion, vec![1, 2, 2, 4, 3, 3, 4, 3]);
    assert_eq!(
        cb.parent,
        vec![
            None,
            Some(0),
            Some(0),
            Some(1),
            Some(1),
            Some(2),
            Some(1),
            Some(1)
        ]
    );
    assert_eq!(cb.len(), 8);
    assert!(!cb.is_empty());
    assert_eq!(
        sorted(cb.block_tree.edge_list()),
        vec![(0, 1), (0, 2), (1, 3), (1, 4), (1, 6), (1, 7), (2, 5)]
    );
    assert_eq!(cb.children(1), vec![3, 4, 6, 7]);
    // Embeddedness: the instructor is in a 4-cohesive block, member 11 only in the root.
    assert_eq!(cb.max_cohesion_of(0), Some(4));
    assert_eq!(cb.max_cohesion_of(11), Some(1));
    assert_eq!(cb.max_cohesion_of(99), None);
}

#[test]
fn cohesive_blocks_invariants_and_errors() {
    let cb = complete(5).cohesive_blocks().unwrap();
    assert_eq!(cb.blocks, vec![vec![0, 1, 2, 3, 4]]);
    assert_eq!(cb.cohesion, vec![4]);
    assert_eq!(cb.parent, vec![None]);

    // Children are more cohesive than their parents, and nested in them.
    let cb = karate().cohesive_blocks().unwrap();
    for (b, p) in cb.parent.iter().enumerate() {
        if let Some(p) = *p {
            assert!(cb.cohesion[b] > cb.cohesion[p]);
            assert!(cb.blocks[b].iter().all(|v| cb.blocks[p].contains(v)));
        }
    }

    let multi = undirected(&[(0, 1), (0, 1), (1, 2)], 3);
    assert!(multi.cohesive_blocks().is_err());
    let d = directed(&[(0, 1), (1, 2)], 3);
    assert!(d.cohesive_blocks().is_err());
}

// ---------------------------------------------------------------------------
// Reachability and transitive closure
// ---------------------------------------------------------------------------

fn small_directed() -> Graph {
    // tests/unit/reachability.c, "Small directed graph".
    directed(
        &[
            (0, 1),
            (1, 2),
            (2, 0),
            (1, 3),
            (3, 4),
            (4, 5),
            (5, 4),
            (7, 4),
            (7, 8),
            (9, 8),
            (10, 9),
            (8, 10),
            (11, 6),
            (12, 6),
        ],
        13,
    )
}

#[test]
fn reachability_from_igraph_tests() {
    let g = small_directed();
    let r = g.reachability(NeighborMode::Out).unwrap();
    assert_eq!(r.membership, vec![5, 5, 5, 6, 7, 7, 4, 2, 3, 3, 3, 1, 0]);
    assert_eq!(r.sizes, vec![1, 1, 1, 3, 1, 3, 1, 2]);
    assert_eq!(r.count, 8);
    assert_eq!(r.reach.len(), 8);
    assert!(r.reach.iter().all(|row| row.len() == 13));
    assert_eq!(r.reachable_from(12), vec![6, 12]);
    assert_eq!(r.reachable_from(0), vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(r.reachable_from(7), vec![4, 5, 7, 8, 9, 10]);
    assert!(r.is_reachable(1, 5));
    assert!(!r.is_reachable(5, 1));
    assert!(!r.is_reachable(-1, 0));
    assert!(r.reachable_from(99).is_empty());

    let counts = g.count_reachable(NeighborMode::Out).unwrap();
    assert_eq!(counts, vec![6, 6, 6, 3, 2, 2, 1, 6, 3, 3, 3, 2, 2]);
    // count_reachable agrees with the reachability bitsets.
    for v in g.vertices() {
        assert_eq!(counts[v as usize], r.reachable_from(v).len());
    }
    assert_eq!(
        g.count_reachable(NeighborMode::In).unwrap(),
        vec![3, 3, 3, 4, 7, 7, 3, 1, 4, 4, 4, 1, 1]
    );
}

#[test]
fn reachability_is_symmetric_under_mode_reversal() {
    let g = small_directed();
    let out = g.reachability(NeighborMode::Out).unwrap();
    let inn = g.reachability(NeighborMode::In).unwrap();
    for u in g.vertices() {
        for v in g.vertices() {
            assert_eq!(out.is_reachable(u, v), inn.is_reachable(v, u));
        }
    }
    // Ignoring directions, reachability is weak connectivity.
    let all = g.reachability(NeighborMode::All).unwrap();
    let weak = g.connected_components(Connectedness::Weak).unwrap();
    assert_eq!(all.count, weak.count);
    for u in g.vertices() {
        for v in g.vertices() {
            assert_eq!(all.is_reachable(u, v), weak.same_component(u, v));
        }
    }
}

#[test]
fn reachability_on_trivial_graphs() {
    let r = Graph::new(0, true).reachability(NeighborMode::Out).unwrap();
    assert_eq!(r.count, 0);
    assert!(r.reach.is_empty());
    let r = Graph::new(1, true).reachability(NeighborMode::Out).unwrap();
    assert_eq!(r.reach, vec![vec![true]]);
    // More than 64 vertices: bits span several words.
    let n = 150;
    let chain: Vec<(i64, i64)> = (0..n - 1).map(|i| (i, i + 1)).collect();
    let g = directed(&chain, n as usize);
    let r = g.reachability(NeighborMode::Out).unwrap();
    assert_eq!(r.reachable_from(70), (70..n).collect::<Vec<_>>());
    assert_eq!(
        g.count_reachable(NeighborMode::Out).unwrap(),
        (1..=n as usize).rev().collect::<Vec<_>>()
    );
}

#[test]
fn transitive_closure_from_igraph_tests() {
    let dag = directed(
        &[
            (8, 7),
            (7, 6),
            (6, 3),
            (6, 0),
            (3, 2),
            (3, 1),
            (5, 0),
            (4, 1),
        ],
        9,
    );
    let tc = dag.transitive_closure().unwrap();
    assert!(tc.is_directed());
    assert_eq!(tc.vcount(), 9);
    assert_eq!(
        sorted(tc.edge_list()),
        vec![
            (3, 1),
            (3, 2),
            (4, 1),
            (5, 0),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (7, 6),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 6),
            (8, 7),
        ]
    );

    let u = undirected(&[(0, 1), (1, 2), (3, 4)], 6);
    let tc = u.transitive_closure().unwrap();
    assert!(!tc.is_directed());
    assert_eq!(sorted(tc.edge_list()), vec![(0, 1), (0, 2), (1, 2), (3, 4)]);
}

#[test]
fn transitive_closure_agrees_with_count_reachable() {
    let g = small_directed();
    let tc = g.transitive_closure().unwrap();
    let counts = g.count_reachable(NeighborMode::Out).unwrap();
    let outdeg = tc.degree(.., NeighborMode::Out, Loops::None).unwrap();
    for v in 0..13 {
        assert_eq!(outdeg[v] as usize + 1, counts[v]);
    }
    // The closure is transitively closed: closing again changes nothing.
    let tc2 = tc.transitive_closure().unwrap();
    assert_eq!(sorted(tc2.edge_list()), sorted(tc.edge_list()));
}

// ---------------------------------------------------------------------------
// Neighborhoods
// ---------------------------------------------------------------------------

fn loops_and_multi() -> Graph {
    // tests/unit/igraph_neighborhood.c
    directed(
        &[
            (0, 1),
            (0, 2),
            (1, 1),
            (1, 3),
            (2, 0),
            (2, 3),
            (3, 4),
            (3, 4),
        ],
        6,
    )
}

#[test]
fn neighborhood_from_igraph_tests() {
    let g = loops_and_multi();
    assert_eq!(
        g.neighborhood(.., Some(0), NeighborMode::All, 0).unwrap(),
        (0..6).map(|v| vec![v]).collect::<Vec<_>>()
    );
    assert_eq!(
        g.neighborhood(.., Some(1), NeighborMode::All, 0).unwrap(),
        vec![
            vec![0, 1, 2],
            vec![1, 0, 3],
            vec![2, 0, 3],
            vec![3, 1, 2, 4],
            vec![4, 3],
            vec![5]
        ]
    );
    assert_eq!(
        g.neighborhood(.., Some(1), NeighborMode::In, 0).unwrap(),
        vec![
            vec![0, 2],
            vec![1, 0],
            vec![2, 0],
            vec![3, 1, 2],
            vec![4, 3],
            vec![5]
        ]
    );
    assert_eq!(
        g.neighborhood(.., Some(10), NeighborMode::All, 0).unwrap(),
        vec![
            vec![0, 1, 2, 3, 4],
            vec![1, 0, 3, 2, 4],
            vec![2, 0, 3, 1, 4],
            vec![3, 1, 2, 4, 0],
            vec![4, 3, 1, 2, 0],
            vec![5]
        ]
    );
    assert_eq!(
        g.neighborhood(.., Some(2), NeighborMode::Out, 2).unwrap(),
        vec![vec![3], vec![4], vec![1, 4], vec![], vec![], vec![]]
    );
    assert!(
        Graph::new(0, false)
            .neighborhood(.., Some(1), NeighborMode::All, 0)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn neighborhood_size_is_consistent() {
    let g = loops_and_multi();
    for order in [Some(0), Some(1), Some(2), Some(3), None] {
        for mode in [NeighborMode::Out, NeighborMode::In, NeighborMode::All] {
            for mindist in 0..=order.unwrap_or(3) {
                let sets = g.neighborhood(.., order, mode, mindist).unwrap();
                let sizes = g.neighborhood_size(.., order, mode, mindist).unwrap();
                let graphs = g.neighborhood_graphs(.., order, mode, mindist).unwrap();
                assert_eq!(sets.iter().map(Vec::len).collect::<Vec<_>>(), sizes);
                assert_eq!(graphs.iter().map(Graph::vcount).collect::<Vec<_>>(), sizes);
            }
        }
    }
    // Unlimited order with Out mode is exactly reachability.
    let sizes = g.neighborhood_size(.., None, NeighborMode::Out, 0).unwrap();
    assert_eq!(sizes, g.count_reachable(NeighborMode::Out).unwrap());
}

#[test]
fn neighborhood_on_karate() {
    let k = karate();
    // 1-neighborhood sizes are degree + 1 on a simple graph.
    let sizes = k
        .neighborhood_size(.., Some(1), NeighborMode::All, 0)
        .unwrap();
    let degrees = k.degree(.., NeighborMode::All, Loops::Twice).unwrap();
    assert!(
        sizes
            .iter()
            .zip(&degrees)
            .all(|(&s, &d)| s == d as usize + 1)
    );
    // The karate club has diameter 5: every 5-neighborhood is everyone.
    assert!(
        k.neighborhood_size(.., Some(5), NeighborMode::All, 0)
            .unwrap()
            .iter()
            .all(|&s| s == 34)
    );
    // Selecting a list of vertices, with duplicates.
    let sel = k
        .neighborhood_size(&[33, 0, 33], Some(1), NeighborMode::All, 1)
        .unwrap();
    assert_eq!(sel, vec![17, 16, 17]);
}

#[test]
fn neighborhood_graphs_are_ego_networks() {
    let k = karate();
    let egos = k
        .neighborhood_graphs(&[0, 33], Some(1), NeighborMode::All, 0)
        .unwrap();
    assert_eq!(egos.len(), 2);
    assert_eq!(egos[0].vcount(), 17);
    assert_eq!(egos[1].vcount(), 18);
    // Ids keep their relative order: the instructor (0) is the first vertex of
    // its ego network and the president (33) the last of his. The center is
    // adjacent to everybody else.
    for (ego, center) in egos.iter().zip([0, 17]) {
        let deg = ego
            .degree_of(center, NeighborMode::All, Loops::Twice)
            .unwrap();
        assert_eq!(deg as usize, ego.vcount() - 1);
    }
    // The ego network of 0 has as many edges as the subgraph its neighborhood induces.
    let hood = k
        .neighborhood(0, Some(1), NeighborMode::All, 0)
        .unwrap()
        .remove(0);
    let inner = k
        .edge_list()
        .iter()
        .filter(|(a, b)| hood.contains(a) && hood.contains(b))
        .count();
    assert_eq!(egos[0].ecount(), inner);
    // Whole-graph neighborhoods are copies of the graph.
    let whole = k
        .neighborhood_graphs(0, None, NeighborMode::All, 0)
        .unwrap();
    assert_eq!((whole[0].vcount(), whole[0].ecount()), (34, 78));
}

#[test]
fn neighborhood_errors() {
    let g = path(4);
    let err = g
        .neighborhood(9, Some(1), NeighborMode::All, 0)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    let err = g
        .neighborhood_size(0, Some(1), NeighborMode::All, 2)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let err = g
        .neighborhood_graphs(&[0, 1], Some(0), NeighborMode::All, 1)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    // Huge Rust-side counts saturate instead of wrapping to negative values.
    let err = g
        .neighborhood_size(0, Some(1), NeighborMode::All, usize::MAX)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    assert_eq!(
        g.neighborhood_size(0, Some(usize::MAX), NeighborMode::All, 0)
            .unwrap(),
        vec![4]
    );
    assert!(
        g.decompose(Connectedness::Weak, Some(usize::MAX), usize::MAX)
            .unwrap()
            .is_empty()
    );
    // mindist beyond the eccentricity is fine with unlimited order.
    assert_eq!(
        g.neighborhood_size(0, None, NeighborMode::All, 10).unwrap(),
        vec![0]
    );
}

// ---------------------------------------------------------------------------
// Use cases
// ---------------------------------------------------------------------------

/// A small power grid: two meshed regional networks joined by a single
/// long-distance line, with a radial feeder hanging off one substation.
/// Where are the single points of failure, and how robust is the grid to
/// the targeted removal of its busiest substations?
#[test]
fn use_case_power_grid_resilience() {
    let north = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]; // meshed, 4 substations
    let south = [(4, 5), (5, 6), (6, 7), (7, 4), (5, 7)]; // meshed, 4 substations
    let tie_line = [(3, 4)];
    let feeder = [(6, 8), (8, 9)]; // radial: 6 - 8 - 9
    let edges: Vec<(i64, i64)> = north
        .iter()
        .chain(&south)
        .chain(&tie_line)
        .chain(&feeder)
        .copied()
        .collect();
    let grid = undirected(&edges, 10);
    assert!(grid.is_connected(Connectedness::Weak).unwrap());

    // Single points of failure: substations and lines whose loss islands part of the grid.
    let critical_substations = sorted(grid.articulation_points().unwrap());
    assert_eq!(critical_substations, vec![3, 4, 6, 8]);
    let critical_lines: Vec<(i64, i64)> = sorted(
        grid.bridges()
            .unwrap()
            .into_iter()
            .map(|e| grid.edge(e).unwrap())
            .collect(),
    );
    assert_eq!(critical_lines, vec![(3, 4), (6, 8), (8, 9)]);

    // The robust cores are the biconnected components with at least 3 substations.
    let bc = grid.biconnected_components().unwrap();
    let cores: Vec<Vec<i64>> =
        canonical(bc.components.into_iter().filter(|c| c.len() >= 3).collect());
    assert_eq!(cores, vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);

    // Targeted attack: knock out substations by decreasing degree and watch
    // the largest island shrink (site percolation, reversed).
    let degrees = grid.degree(.., NeighborMode::All, Loops::Twice).unwrap();
    let mut attack: Vec<i64> = grid.vertices().collect();
    attack.sort_by_key(|&v| std::cmp::Reverse(degrees[v as usize]));
    let build_order: Vec<i64> = attack.iter().rev().copied().collect();
    let p = grid.site_percolation(Some(&build_order)).unwrap();
    let mut after_removals: Vec<usize> = p.giant_size.clone();
    after_removals.reverse(); // after_removals[k] = giant size with k hubs removed
    assert_eq!(after_removals[0], 10);
    assert!(after_removals.windows(2).all(|w| w[0] >= w[1]));
    // The three busiest substations (0, 2, 3) are all in the north: removing
    // them wipes out the northern region, leaving the south + feeder (6) as
    // the largest island, and the grid fragments quickly afterwards.
    assert_eq!(&attack[..3], &[0, 2, 3]);
    assert_eq!(after_removals, vec![10, 9, 7, 6, 5, 4, 2, 2, 2, 1]);

    // After a storm takes out the tie line, the grid splits into two islands.
    let mut storm = grid.clone();
    let tie = grid.get_eid(3, 4, false).unwrap().unwrap();
    storm.delete_edges(tie).unwrap();
    let islands = storm.decompose(Connectedness::Weak, None, 1).unwrap();
    let mut sizes: Vec<usize> = islands.iter().map(Graph::vcount).collect();
    sizes.sort();
    assert_eq!(sizes, vec![4, 6]);
}

/// Information flow in a small organisation: who ultimately receives a memo
/// sent by each person, which groups talk to each other in both directions,
/// and who is within two hops of the CEO?
#[test]
fn use_case_memo_propagation() {
    // 0 = CEO; 1, 2 = directors who report to each other; 3..=6 = staff.
    let reports = [
        (0, 1),
        (1, 2),
        (2, 1),
        (1, 3),
        (2, 4),
        (4, 5),
        (5, 4),
        (3, 6),
        (6, 3),
    ];
    let org = directed(&reports, 7);

    // Mutual-communication groups are the strongly connected components.
    let scc = org.connected_components(Connectedness::Strong).unwrap();
    assert_eq!(scc.count, 4);
    assert!(scc.same_component(1, 2) && scc.same_component(4, 5) && scc.same_component(3, 6));

    // A memo from the CEO reaches everyone; one from staff member 4 stays with 4 and 5.
    let r = org.reachability(NeighborMode::Out).unwrap();
    assert_eq!(r.reachable_from(0), (0..7).collect::<Vec<_>>());
    assert_eq!(r.reachable_from(4), vec![4, 5]);
    // Who can have influenced staff member 6 (reverse reachability)?
    let heard_by = org.reachability(NeighborMode::In).unwrap();
    assert_eq!(heard_by.reachable_from(6), vec![0, 1, 2, 3, 6]);

    // Audience sizes, and a "direct memo" graph via the transitive closure.
    let audience = org.count_reachable(NeighborMode::Out).unwrap();
    assert_eq!(audience, vec![7, 6, 6, 2, 2, 2, 2]);
    let direct = org.transitive_closure().unwrap();
    assert_eq!(
        direct.degree_of(0, NeighborMode::Out, Loops::None).unwrap(),
        6
    );

    // The CEO's two-hop circle, excluding the CEO.
    let circle = org.neighborhood(0, Some(2), NeighborMode::Out, 1).unwrap();
    assert_eq!(sorted(circle[0].clone()), vec![1, 2, 3]);
}

// ---------------------------------------------------------------------------
// Known values from igraph's unit tests (tests/unit/*.c, *.out)
// ---------------------------------------------------------------------------

#[test]
fn common_karate_is_the_famous_zachary_graph() {
    let zachary = Graph::famous("Zachary").unwrap();
    assert!(zachary.is_same_graph(&karate()).unwrap());
}

#[test]
fn bridges_from_igraph_tests() {
    // tests/unit/igraph_bridges.c (+ .out)
    for n in 0..3 {
        assert!(Graph::new(n, false).bridges().unwrap().is_empty());
    }
    let g = undirected(
        &[
            (0, 1),
            (1, 2),
            (0, 2),
            (0, 3),
            (3, 4),
            (4, 5),
            (3, 5),
            (4, 6),
        ],
        7,
    );
    assert_eq!(sorted(g.bridges().unwrap()), vec![3, 7]);
    // A disconnected graph.
    let g = undirected(
        &[
            (0, 1),
            (1, 2),
            (1, 3),
            (4, 5),
            (5, 6),
            (4, 6),
            (4, 7),
            (7, 8),
            (4, 8),
            (9, 10),
            (10, 11),
            (11, 12),
            (9, 12),
            (9, 13),
            (13, 14),
        ],
        16,
    );
    assert_eq!(sorted(g.bridges().unwrap()), vec![0, 1, 2, 13, 14]);
    // Multi-edges and self-loops are never bridges.
    let g = undirected(&[(0, 1), (0, 1), (1, 2), (2, 2)], 3);
    assert_eq!(g.bridges().unwrap(), vec![2]);
}

#[test]
fn is_biconnected_from_igraph_tests() {
    // tests/unit/igraph_is_biconnected.c
    // (edges, number of vertices, biconnected?)
    type Case = (&'static [(i64, i64)], usize, bool);
    let cases: &[Case] = &[
        (&[], 0, false),
        (&[], 1, false),
        (&[], 2, false),
        (&[(0, 1)], 2, true),
        (
            &[(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (4, 5), (5, 2)],
            6,
            false,
        ),
        (&[(0, 1), (1, 2), (2, 0), (1, 3)], 7, false),
        (&[(0, 1), (1, 2), (2, 0), (1, 3), (3, 4), (4, 2)], 5, true),
        (&[(0, 1), (1, 2), (2, 0), (1, 3), (3, 4)], 7, false),
        // Two disjoint cycles; a cycle and an isolated vertex.
        (&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], 6, false),
        (&[(0, 1), (1, 2), (2, 0)], 4, false),
        // The DFS root is an articulation point.
        (&[(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)], 5, false),
    ];
    for &(edges, n, expected) in cases {
        let g = undirected(edges, n);
        assert_eq!(g.is_biconnected().unwrap(), expected, "{edges:?} on {n}");
    }
    assert!(
        Graph::ring(10, false, false, true)
            .unwrap()
            .is_biconnected()
            .unwrap()
    );
    // K2 is biconnected yet acyclic: the caches must not get confused.
    let k2 = Graph::full(2, false, false).unwrap();
    assert!(k2.is_biconnected().unwrap());
    assert!(k2.is_acyclic().unwrap());
    assert!(k2.is_biconnected().unwrap());
    assert!(k2.articulation_points().unwrap().is_empty());
}

/// Checks minimality by definition, like tests/unit/igraph_is_separator.c:
/// a separator none of whose one-smaller subsets is a separator.
fn brute_force_minimal(g: &Graph, s: &[i64]) -> bool {
    let s: Vec<i64> = sorted(s.to_vec()).into_iter().fold(vec![], |mut acc, v| {
        if acc.last() != Some(&v) {
            acc.push(v);
        }
        acc
    });
    g.is_separator(s.as_slice()).unwrap()
        && (0..s.len()).all(|i| {
            let mut smaller = s.clone();
            smaller.remove(i);
            !g.is_separator(smaller.as_slice()).unwrap()
        })
}

#[test]
fn separators_of_disconnected_graphs_from_igraph_tests() {
    // tests/unit/igraph_is_separator.c, "test graph 2": components
    // {0, 1, 2, 3} (a triangle with a pendant 3 and a loop on 1) and {4, 5, 6}.
    let g = undirected(
        &[
            (0, 1),
            (1, 2),
            (2, 0),
            (0, 3),
            (4, 5),
            (5, 6),
            (1, 1),
            (5, 6),
        ],
        7,
    );
    let cases: &[(&[i64], bool, bool)] = &[
        (&[0], true, true),
        (&[5], true, true),
        (&[1, 0], true, false),
        (&[1, 3, 0, 1, 3, 3], false, false),
        (&[4, 1, 0], true, false),
        (&[4, 1, 0, 2], false, false),
        (&[4, 1, 0, 5, 2], false, false),
        (&[1, 0, 5, 2], true, false),
        (&[5, 0], true, false),
        (&[], false, false),
    ];
    for &(s, sep, min) in cases {
        assert_eq!(g.is_separator(s).unwrap(), sep, "is_separator({s:?})");
        assert_eq!(g.is_minimal_separator(s).unwrap(), min, "minimal({s:?})");
        assert_eq!(brute_force_minimal(&g, s), min, "brute force({s:?})");
    }
    // Already-disconnected vertices do not make a set a separator.
    let isolated = Graph::new(3, false);
    assert!(!isolated.is_separator(&[] as &[i64]).unwrap());
    assert!(!isolated.is_separator(0).unwrap());
    // The minimal (s,t) separators of a disconnected graph are those of its
    // components, while it has no minimum-size separators at all.
    let two_paths = path(3).disjoint_union(&path(3)).unwrap();
    assert_eq!(
        canonical(two_paths.all_minimal_st_separators().unwrap()),
        vec![vec![1], vec![4]]
    );
    assert!(two_paths.minimum_size_separators().unwrap().is_empty());
}

#[test]
fn separators_of_two_linked_cliques_from_igraph_tests() {
    // tests/unit/igraph_is_separator.c, "test graph 4": two K4s {0..3} and
    // {4..7} linked by the edges 2-5 and 3-4.
    let k4 = Graph::full(4, false, false).unwrap();
    let mut g = k4.disjoint_union(&k4).unwrap();
    g.add_edges(&[(2, 5), (3, 4)]).unwrap();
    for s in [&[2, 3][..], &[2, 4], &[3, 5]] {
        assert!(g.is_separator(s).unwrap());
        assert!(g.is_minimal_separator(s).unwrap());
    }
    assert!(g.is_separator(&[2, 3, 5]).unwrap());
    assert!(!g.is_minimal_separator(&[2, 3, 5]).unwrap());
    // The minimum-size separators: one endpoint of each link, or both
    // endpoints on one side. Their size is the vertex connectivity.
    assert_eq!(g.vertex_connectivity(false).unwrap(), 2);
    assert_eq!(
        canonical(g.minimum_size_separators().unwrap()),
        vec![vec![2, 3], vec![2, 4], vec![3, 5], vec![4, 5]]
    );
    assert!(!g.is_separator(&[2, 5]).unwrap());
    assert!(!g.is_separator(&[3, 4]).unwrap());
}

#[test]
fn karate_minimal_separator_test_agrees_with_definition() {
    // tests/unit/igraph_is_separator.c, karate club section.
    let k = Graph::famous("Zachary").unwrap();
    for s in k.all_minimal_st_separators().unwrap() {
        assert!(k.is_separator(s.as_slice()).unwrap());
        assert_eq!(
            k.is_minimal_separator(s.as_slice()).unwrap(),
            brute_force_minimal(&k, &s),
            "{s:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Identities with other modules
// ---------------------------------------------------------------------------

#[test]
fn cohesion_of_blocks_is_their_vertex_connectivity() {
    let k = karate();
    let cb = k.cohesive_blocks().unwrap();
    for (block, &cohesion) in cb.blocks.iter().zip(&cb.cohesion) {
        let sub = k
            .induced_subgraph(block.as_slice(), SubgraphImplementation::Auto)
            .unwrap();
        assert_eq!(sub.vertex_connectivity(false).unwrap(), cohesion);
        assert_eq!(sub.cohesion(false).unwrap(), cohesion);
    }
    // A disconnected graph: a 0-cohesive root with its components below.
    let two_triangles = cycle(3).disjoint_union(&cycle(3)).unwrap();
    let cb = two_triangles.cohesive_blocks().unwrap();
    assert_eq!(
        cb.blocks,
        vec![(0..6).collect(), vec![0, 1, 2], vec![3, 4, 5]]
    );
    assert_eq!(cb.cohesion, vec![0, 2, 2]);
    assert_eq!(cb.parent, vec![None, Some(0), Some(0)]);
    // The null graph: a single, empty root block.
    let cb = Graph::new(0, false).cohesive_blocks().unwrap();
    assert_eq!(cb.blocks, vec![Vec::<i64>::new()]);
    assert_eq!(cb.cohesion, vec![0]);
    assert!(!cb.is_empty());
}

#[test]
fn minimum_size_separators_have_vertex_connectivity_size() {
    let graphs = [
        karate(),
        Graph::famous("Petersen").unwrap(),
        Graph::square_lattice(&[3, 4], 1, false, false, None).unwrap(),
        Graph::ring(7, false, false, true).unwrap(),
    ];
    for g in &graphs {
        let k = g.vertex_connectivity(false).unwrap();
        let seps = g.minimum_size_separators().unwrap();
        assert!(!seps.is_empty());
        for s in &seps {
            assert_eq!(s.len(), k);
            assert!(g.is_separator(s.as_slice()).unwrap());
            assert!(g.is_minimal_separator(s.as_slice()).unwrap());
        }
    }
    // The only minimum separator of the karate club is the instructor.
    assert_eq!(karate().minimum_size_separators().unwrap(), vec![vec![0]]);
    // The Petersen graph is 3-connected and its only 3-separators are the
    // neighborhoods of its vertices.
    let petersen = &graphs[1];
    let hoods = canonical(
        petersen
            .neighborhood(.., Some(1), NeighborMode::All, 1)
            .unwrap(),
    );
    assert_eq!(
        canonical(petersen.minimum_size_separators().unwrap()),
        hoods
    );
}

#[test]
fn biconnectivity_agrees_with_connectivity_on_random_graphs() {
    rng::seed(2024).unwrap();
    for _ in 0..40 {
        let g = Graph::erdos_renyi_game_gnp(8, 0.35, false, EdgeTypeSw::Simple, false).unwrap();
        let vc = g.vertex_connectivity(false).unwrap();
        let ec = g.edge_connectivity(false).unwrap();
        let connected = g.is_connected(Connectedness::Weak).unwrap();
        assert_eq!(connected, vc >= 1);
        assert_eq!(g.is_biconnected().unwrap(), vc >= 2);
        let cc = g.connected_components(Connectedness::Weak).unwrap();
        assert_eq!(cc.count == 1, connected);
        if connected {
            assert_eq!(!g.articulation_points().unwrap().is_empty(), vc == 1);
            assert_eq!(!g.bridges().unwrap().is_empty(), ec == 1);
        }
        // Each biconnected component with at least 3 vertices is 2-connected.
        for comp in g.biconnected_components().unwrap().components {
            if comp.len() >= 3 {
                let sub = g
                    .induced_subgraph(comp.as_slice(), SubgraphImplementation::Auto)
                    .unwrap();
                assert!(sub.vertex_connectivity(false).unwrap() >= 2);
            }
        }
    }
}

#[test]
fn decompose_gives_induced_subgraphs_of_components() {
    let g = small_directed();
    for mode in [Connectedness::Weak, Connectedness::Strong] {
        let cc = g.connected_components(mode).unwrap();
        let parts = g.decompose(mode, None, 0).unwrap();
        assert_eq!(parts.len(), cc.count);
        // For weak components the parts come in component id order.
        if mode == Connectedness::Weak {
            for (c, part) in parts.iter().enumerate() {
                let induced = g
                    .induced_subgraph(cc.members(c).as_slice(), SubgraphImplementation::Auto)
                    .unwrap();
                assert!(part.is_same_graph(&induced).unwrap());
            }
        }
        let mut shapes: Vec<usize> = parts.iter().map(Graph::vcount).collect();
        shapes.sort();
        assert_eq!(shapes, sorted(cc.sizes.clone()));
    }
}

#[test]
fn reachability_agrees_with_subcomponent_and_graph_power() {
    let g = small_directed();
    for mode in [NeighborMode::Out, NeighborMode::In, NeighborMode::All] {
        let r = g.reachability(mode).unwrap();
        for v in g.vertices() {
            assert_eq!(
                r.reachable_from(v),
                sorted(g.subcomponent(v, mode).unwrap())
            );
        }
    }
    // The transitive closure is the (n-1)-th power of the graph.
    let tc = g.transitive_closure().unwrap();
    let power = g.graph_power(g.vcount() - 1, true).unwrap();
    assert_eq!(sorted(tc.edge_list()), sorted(power.edge_list()));
    // On a DAG, a topological order never goes against reachability.
    let dag = Graph::kary_tree(15, 2, TreeMode::Out).unwrap();
    let order = dag.topological_sorting(NeighborMode::Out).unwrap();
    let r = dag.reachability(NeighborMode::Out).unwrap();
    for (i, &u) in order.iter().enumerate() {
        for &v in &order[..i] {
            assert!(!r.is_reachable(u, v));
        }
    }
}

#[test]
fn neighborhoods_agree_with_distances_and_induced_subgraphs() {
    let k = karate();
    let d = k.distances(.., .., None, NeighborMode::All).unwrap();
    for order in 0..=3 {
        for mindist in 0..=order {
            let hoods = k
                .neighborhood(.., Some(order), NeighborMode::All, mindist)
                .unwrap();
            for (v, hood) in hoods.iter().enumerate() {
                let expected: Vec<i64> = (0..34)
                    .filter(|&u| {
                        let dist = d[(v, u as usize)];
                        dist >= mindist as f64 && dist <= order as f64
                    })
                    .collect();
                assert_eq!(sorted(hood.clone()), expected);
            }
        }
    }
    // Directed: follow (Out) or oppose (In) edge directions.
    let g = small_directed();
    for mode in [NeighborMode::Out, NeighborMode::In] {
        let d = g.distances(.., .., None, mode).unwrap();
        let hoods = g.neighborhood(.., Some(2), mode, 1).unwrap();
        for (v, hood) in hoods.iter().enumerate() {
            let expected: Vec<i64> = (0..13)
                .filter(|&u| (1.0..=2.0).contains(&d[(v, u as usize)]))
                .collect();
            assert_eq!(sorted(hood.clone()), expected);
        }
    }
    // Ego networks are induced subgraphs of the neighborhoods.
    let hoods = k.neighborhood(.., Some(2), NeighborMode::All, 0).unwrap();
    let egos = k
        .neighborhood_graphs(.., Some(2), NeighborMode::All, 0)
        .unwrap();
    for (hood, ego) in hoods.iter().zip(&egos) {
        let induced = k
            .induced_subgraph(hood.as_slice(), SubgraphImplementation::Auto)
            .unwrap();
        assert!(ego.is_same_graph(&induced).unwrap());
    }
}

// ---------------------------------------------------------------------------
// Percolation: self-loops and randomness
// ---------------------------------------------------------------------------

#[test]
fn site_percolation_counts_self_loops_once() {
    // igraph 1.0.0 and 1.0.1 count self-loops twice; the wrapper corrects it.
    let g = undirected(&[(0, 0), (0, 1), (1, 1), (1, 1), (2, 2)], 3);
    let p = g.site_percolation(Some(&[0, 1, 2])).unwrap();
    assert_eq!(p.edge_count, vec![1, 4, 5]);
    assert_eq!(p.giant_size, vec![1, 2, 2]);
    let p = g.site_percolation(Some(&[2, 1])).unwrap();
    assert_eq!(p.edge_count, vec![1, 3]);
    assert_eq!(p.giant_size, vec![1, 1]);
    let d = directed(&[(0, 0), (0, 1), (1, 0)], 2);
    assert_eq!(
        d.site_percolation(Some(&[1, 0])).unwrap().edge_count,
        vec![0, 3]
    );
    // With a random order, all the edges are there at the end.
    rng::seed(3).unwrap();
    let p = g.site_percolation(None).unwrap();
    assert_eq!(p.edge_count.len(), 3);
    assert_eq!(p.edge_count.last(), Some(&g.ecount()));
    // Bond percolation was never affected: a loop only adds its vertex.
    let p = g.bond_percolation(Some(&[0, 1, 2, 3, 4])).unwrap();
    assert_eq!(p.giant_size, vec![1, 2, 2, 2, 2]);
    assert_eq!(p.vertex_count, vec![1, 2, 2, 2, 3]);
}

#[test]
fn random_site_percolation_uses_the_same_order_as_igraph() {
    // Our wrapper draws the random order itself; it must be the one igraph
    // would draw internally for the same seed.
    let k = karate();
    rng::seed(5).unwrap();
    let ours = k.site_percolation(None).unwrap();
    rng::seed(5).unwrap();
    let mut giant = VectorInt::new();
    let mut count = VectorInt::new();
    let rc =
        unsafe { igraph::igraph_site_percolation(&k, &mut giant, &mut count, std::ptr::null()) };
    assert_eq!(rc, igraph::igraph_error_type_t_IGRAPH_SUCCESS);
    let theirs_giant: Vec<usize> = giant.iter().map(|&x| x as usize).collect();
    let theirs_count: Vec<usize> = count.iter().map(|&x| x as usize).collect();
    assert_eq!(ours.giant_size, theirs_giant);
    assert_eq!(ours.edge_count, theirs_count);
    assert_eq!(ours.edge_count.last(), Some(&78));
}

#[test]
fn seeded_percolation_is_reproducible_in_parallel_threads() {
    // Every thread has its own default RNG: seeding it makes the thread's
    // random curves reproducible regardless of what other threads do.
    fn run() -> (BondPercolation, SitePercolation) {
        rng::seed(99).unwrap();
        let g = Graph::famous("Zachary").unwrap();
        (
            g.bond_percolation(None).unwrap(),
            g.site_percolation(None).unwrap(),
        )
    }
    let expected = run();
    let handles: Vec<_> = (0..4).map(|_| std::thread::spawn(run)).collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), expected);
    }
}

#[test]
fn edgelist_percolation_rejects_ids_whose_successor_overflows() {
    // igraph computes `max id + 1` without an overflow check (and then
    // aborts on the negative size): the wrapper must reject i64::MAX.
    assert_eq!(
        edgelist_percolation(&[(0, i64::MAX)]).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        edgelist_percolation(&[(i64::MAX, 0)]).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
    // Merely huge ids fail cleanly with an allocation error.
    assert_eq!(
        edgelist_percolation(&[(0, i64::MAX - 1)])
            .unwrap_err()
            .kind(),
        ErrorKind::OutOfMemory
    );
}

#[test]
fn self_loops_belong_to_no_biconnected_component() {
    // A triangle 0-1-2 with a pendant 2-4, and loops on 0, 3 and 4.
    let g = undirected(&[(0, 1), (1, 2), (2, 0), (0, 0), (3, 3), (2, 4), (4, 4)], 6);
    let bc = g.biconnected_components().unwrap();
    assert_eq!(bc.count, 2);
    let mut edges: Vec<i64> = bc.component_edges.iter().flatten().copied().collect();
    edges.sort();
    assert_eq!(edges, vec![0, 1, 2, 5]);
    assert_eq!(
        canonical(bc.components.clone()),
        vec![vec![0, 1, 2], vec![2, 4]]
    );
    assert_eq!(bc.articulation_points, vec![2]);
    // Loops and multi-edges do not affect biconnectivity.
    assert!(undirected(&[(0, 1), (0, 0)], 2).is_biconnected().unwrap());
    assert!(undirected(&[(0, 1), (0, 1)], 2).is_biconnected().unwrap());
}

#[test]
fn separators_only_split_connected_vertices() {
    // The path 0-1-2 plus an isolated edge 3-4.
    let g = undirected(&[(0, 1), (1, 2), (3, 4)], 5);
    assert!(g.is_separator(1).unwrap());
    // Removing 3 isolates nothing that was connected to anything else.
    assert!(!g.is_separator(3).unwrap());
    // Removing an end of the path leaves the rest connected.
    assert!(!g.is_separator(0).unwrap());
    assert!(!g.is_separator(&[] as &[i64]).unwrap());
}

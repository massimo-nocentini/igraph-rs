//! Integration tests for the `isomorphism` module (isomorphism, motifs,
//! graphlets).

mod common;

use common::*;
use igraph::isomorphism::{
    BlissSh, MotifSample, TriadCensus, Vf2Options, graph_count, invert_permutation,
};
use igraph::prelude::*;
use std::collections::{BTreeSet, HashSet};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// The Petersen graph, from igraph's catalogue of famous graphs.
fn petersen() -> Graph {
    Graph::famous("Petersen").unwrap()
}

/// Relabels the vertices of `g`: vertex `v` becomes `perm[v]`. Edges are
/// also shuffled, to make sure nothing depends on the edge order.
fn relabel(g: &Graph, perm: &[i64]) -> Graph {
    let mut edges: Vec<(i64, i64)> = g
        .edge_list()
        .into_iter()
        .map(|(a, b)| (perm[a as usize], perm[b as usize]))
        .collect();
    edges.reverse();
    Graph::from_edges(&edges, g.vcount(), g.is_directed()).unwrap()
}

/// A random permutation of `0..n`, from igraph's (seeded) RNG.
fn random_perm(n: usize) -> Vec<i64> {
    let mut p: Vec<i64> = (0..n as i64).collect();
    for i in (1..n).rev() {
        let j = rng::integer(0, i as i64) as usize;
        p.swap(i, j);
    }
    p
}

/// Whether `perm` maps the edge set of `g` onto the edge set of `h`.
fn is_isomorphism(g: &Graph, h: &Graph, perm: &[i64]) -> bool {
    g.edge_list().into_iter().all(|(a, b)| {
        h.get_eid(perm[a as usize], perm[b as usize], true)
            .unwrap()
            .is_some()
    }) && g.ecount() == h.ecount()
}

fn directed_star(n: usize) -> Graph {
    Graph::star(n, StarMode::Out, 0).unwrap()
}

// ---------------------------------------------------------------------------
// Generic isomorphism
// ---------------------------------------------------------------------------

#[test]
fn relabeled_karate_is_isomorphic() {
    rng::seed(42).unwrap();
    let k = karate();
    let perm = random_perm(34);
    let k2 = relabel(&k, &perm);
    assert!(k.isomorphic(&k2).unwrap());
    // Removing one edge breaks it.
    let mut k3 = k2.clone();
    k3.delete_edges(0).unwrap();
    assert!(!k.isomorphic(&k3).unwrap());
}

#[test]
fn isomorphic_rejects_mixed_directedness() {
    let u = path(3);
    let d = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    assert_eq!(
        u.isomorphic(&d).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn isomorphic_handles_multigraphs() {
    // Two triangles with a doubled edge: isomorphic iff the double sits on
    // "the same" position, which for a triangle is always the case...
    let a = Graph::from_edges(&[(0, 1), (0, 1), (1, 2), (2, 0)], 3, false).unwrap();
    let b = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 0)], 3, false).unwrap();
    assert!(a.isomorphic(&b).unwrap());
    // ... but a path with the double edge at the end differs from one with a
    // double edge *and* a loop.
    let p = Graph::from_edges(&[(0, 1), (0, 1), (1, 2), (3, 3)], 4, false).unwrap();
    let q = Graph::from_edges(&[(0, 1), (1, 2), (1, 2), (0, 0)], 4, false).unwrap();
    assert!(!p.isomorphic(&q).unwrap());
    let r = Graph::from_edges(&[(3, 2), (2, 1), (2, 1), (0, 0)], 4, false).unwrap();
    assert!(p.isomorphic(&r).unwrap());
}

#[test]
fn subisomorphic_basic() {
    let k = karate();
    assert!(k.subisomorphic(&complete(5)).unwrap()); // karate has a 5-clique
    assert!(!k.subisomorphic(&complete(6)).unwrap());
}

// ---------------------------------------------------------------------------
// simplify_and_colorize (values from igraph's tests/unit/simplify_and_colorize.out)
// ---------------------------------------------------------------------------

#[test]
fn simplify_and_colorize_matches_igraph() {
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 2), (2, 2), (2, 1)], 3, false).unwrap();
    let c = g.simplify_and_colorize().unwrap();
    assert_eq!(c.graph.edge_list(), vec![(0, 1), (0, 2), (1, 2)]);
    assert_eq!(c.vertex_color, vec![0, 0, 2]);
    assert_eq!(c.edge_color, vec![1, 1, 2]);

    let d = Graph::from_edges(
        &[(0, 1), (0, 1), (1, 0), (1, 0), (1, 0), (1, 1), (1, 1)],
        4,
        true,
    )
    .unwrap();
    let c = d.simplify_and_colorize().unwrap();
    assert!(c.graph.is_directed());
    assert_eq!(c.graph.edge_list(), vec![(0, 1), (1, 0)]);
    assert_eq!(c.vertex_color, vec![0, 2, 0, 0]);
    assert_eq!(c.edge_color, vec![2, 3]);

    let empty = Graph::new(0, false).simplify_and_colorize().unwrap();
    assert_eq!(empty.graph.vcount(), 0);
    assert!(empty.vertex_color.is_empty() && empty.edge_color.is_empty());
}

#[test]
fn colorized_vf2_decides_multigraph_isomorphism() {
    // VF2 needs simple graphs: colorize, then compare colored graphs.
    let a = Graph::from_edges(&[(0, 1), (0, 1), (1, 2), (2, 2)], 3, false).unwrap();
    let b = Graph::from_edges(&[(0, 0), (0, 1), (1, 2), (1, 2)], 3, false).unwrap();
    let c = Graph::from_edges(&[(0, 1), (1, 2), (1, 2), (2, 2)], 3, false).unwrap();
    let ca = a.simplify_and_colorize().unwrap();
    let cb = b.simplify_and_colorize().unwrap();
    let cc = c.simplify_and_colorize().unwrap();
    let mut o = Vf2Options::new()
        .with_vertex_colors(&ca.vertex_color, &cb.vertex_color)
        .with_edge_colors(&ca.edge_color, &cb.edge_color);
    assert!(
        ca.graph
            .isomorphic_vf2(&cb.graph, &mut o)
            .unwrap()
            .is_some()
    );
    let mut o = Vf2Options::new()
        .with_vertex_colors(&ca.vertex_color, &cc.vertex_color)
        .with_edge_colors(&ca.edge_color, &cc.edge_color);
    assert!(
        ca.graph
            .isomorphic_vf2(&cc.graph, &mut o)
            .unwrap()
            .is_none()
    );
    // And the generic function agrees.
    assert!(a.isomorphic(&b).unwrap());
    assert!(!a.isomorphic(&c).unwrap());
}

// ---------------------------------------------------------------------------
// VF2
// ---------------------------------------------------------------------------

#[test]
fn vf2_mapping_is_an_isomorphism() {
    rng::seed(7).unwrap();
    let p = petersen();
    let perm = random_perm(10);
    let q = relabel(&p, &perm);
    let m = p
        .isomorphic_vf2(&q, &mut Vf2Options::new())
        .unwrap()
        .unwrap();
    assert!(is_isomorphism(&p, &q, &m.map12));
    assert!(is_isomorphism(&q, &p, &m.map21));
    assert_eq!(invert_permutation(&m.map12).unwrap(), m.map21);
}

#[test]
fn vf2_counts_petersen_symmetries() {
    let p = petersen();
    assert_eq!(
        p.count_isomorphisms_vf2(&p, &mut Vf2Options::new())
            .unwrap(),
        120
    );
    let all = p.get_isomorphisms_vf2(&p, &mut Vf2Options::new()).unwrap();
    assert_eq!(all.len(), 120);
    let distinct: HashSet<Vec<i64>> = all.iter().cloned().collect();
    assert_eq!(distinct.len(), 120);
    assert!(all.iter().all(|m| is_isomorphism(&p, &p, m)));
}

#[test]
fn vf2_non_isomorphic_gives_none_and_empty() {
    let c6 = cycle(6);
    let two_triangles =
        Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], 6, false).unwrap();
    // Same degree sequence, not isomorphic.
    let mut o = Vf2Options::new();
    assert!(c6.isomorphic_vf2(&two_triangles, &mut o).unwrap().is_none());
    assert_eq!(
        c6.count_isomorphisms_vf2(&two_triangles, &mut o).unwrap(),
        0
    );
    assert!(
        c6.get_isomorphisms_vf2(&two_triangles, &mut o)
            .unwrap()
            .is_empty()
    );
    assert!(!c6.isomorphic(&two_triangles).unwrap());
}

#[test]
fn vf2_vertex_and_edge_colors() {
    let c6 = cycle(6);
    // Color vertices alternately: rotations by an even amount and some
    // reflections survive, 6 of the 12 symmetries.
    let colors = [0, 1, 0, 1, 0, 1];
    let mut o = Vf2Options::new().with_vertex_colors(&colors, &colors);
    assert_eq!(c6.count_isomorphisms_vf2(&c6, &mut o).unwrap(), 6);
    assert_eq!(c6.count_automorphisms(Some(&colors)).unwrap(), 6.0);
    // Color one edge: only the identity and the reflection fixing it.
    let ecol = [1, 0, 0, 0, 0, 0];
    let mut o = Vf2Options::new().with_edge_colors(&ecol, &ecol);
    assert_eq!(c6.count_isomorphisms_vf2(&c6, &mut o).unwrap(), 2);
}

#[test]
fn vf2_color_length_is_validated() {
    let c = cycle(4);
    let bad = [0, 1];
    let mut o = Vf2Options::new().with_vertex_colors(&bad, &bad);
    assert_eq!(
        c.count_isomorphisms_vf2(&c, &mut o).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn vf2_compat_closures() {
    // Match vertices only to vertices with the same parity of id; on C6 this
    // behaves like the alternating coloring.
    let c6 = cycle(6);
    let mut calls = 0;
    let mut o = Vf2Options::new().with_node_compat(|a, b| {
        calls += 1;
        a % 2 == b % 2
    });
    assert_eq!(c6.count_isomorphisms_vf2(&c6, &mut o).unwrap(), 6);
    drop(o);
    assert!(calls > 0);
    // Edge compat: edge 0 must map to itself.
    let mut o = Vf2Options::new().with_edge_compat(|e1, e2| (e1 == 0) == (e2 == 0));
    assert_eq!(c6.count_isomorphisms_vf2(&c6, &mut o).unwrap(), 2);
    // The options (and their closures) can be reused.
    assert_eq!(c6.count_isomorphisms_vf2(&c6, &mut o).unwrap(), 2);
}

#[test]
fn vf2_callback_stops_early() {
    let k5 = complete(5);
    let mut n = 0;
    k5.get_isomorphisms_vf2_callback(&k5, &mut Vf2Options::new(), |m12, m21| {
        assert_eq!(invert_permutation(m12).unwrap(), m21);
        n += 1;
        n < 10
    })
    .unwrap();
    assert_eq!(n, 10);
    // Without stopping, all 120 are seen.
    let mut n = 0;
    k5.get_isomorphisms_vf2_callback(&k5, &mut Vf2Options::new(), |_, _| {
        n += 1;
        true
    })
    .unwrap();
    assert_eq!(n, 120);
}

#[test]
fn vf2_callback_panic_is_propagated() {
    let k4 = complete(4);
    let res = std::panic::catch_unwind(|| {
        k4.get_isomorphisms_vf2_callback(&k4, &mut Vf2Options::new(), |_, _| panic!("boom"))
    });
    let payload = res.unwrap_err();
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"boom"));
    // The thread is still usable afterwards.
    assert_eq!(
        k4.count_isomorphisms_vf2(&k4, &mut Vf2Options::new())
            .unwrap(),
        24
    );
}

#[test]
fn vf2_compat_panic_is_propagated() {
    let k4 = complete(4);
    let res = std::panic::catch_unwind(|| {
        let mut o = Vf2Options::new().with_node_compat(|_, _| panic!("compat"));
        k4.count_isomorphisms_vf2(&k4, &mut o)
    });
    assert!(res.is_err());
    assert_eq!(
        k4.count_isomorphisms_vf2(&k4, &mut Vf2Options::new())
            .unwrap(),
        24
    );
}

#[test]
fn vf2_subisomorphisms() {
    let k = karate();
    let tri = complete(3);
    // Every triangle is found 6 times (its automorphisms); karate has 45.
    assert_eq!(
        k.count_subisomorphisms_vf2(&tri, &mut Vf2Options::new())
            .unwrap(),
        45 * 6
    );
    let maps = k
        .get_subisomorphisms_vf2(&tri, &mut Vf2Options::new())
        .unwrap();
    let sets: BTreeSet<Vec<i64>> = maps
        .into_iter()
        .map(|mut m| {
            m.sort();
            m
        })
        .collect();
    let listed: BTreeSet<Vec<i64>> = k
        .list_triangles()
        .unwrap()
        .into_iter()
        .map(|t| {
            let mut t = t.to_vec();
            t.sort();
            t
        })
        .collect();
    assert_eq!(sets, listed);

    let m = k
        .subisomorphic_vf2(&complete(5), &mut Vf2Options::new())
        .unwrap()
        .unwrap();
    assert_eq!(m.map21.len(), 5);
    assert_eq!(m.map12.iter().filter(|&&x| x >= 0).count(), 5);
    for i in 0..5 {
        for j in i + 1..5 {
            assert!(k.get_eid(m.map21[i], m.map21[j], false).unwrap().is_some());
        }
    }
    assert!(
        k.subisomorphic_vf2(&complete(6), &mut Vf2Options::new())
            .unwrap()
            .is_none()
    );
}

#[test]
fn vf2_subisomorphism_callback_with_colors() {
    // Find edges between "red" (color 1) vertices of a 6-cycle.
    let c6 = cycle(6);
    let colors = [1, 1, 1, 0, 0, 0];
    let pattern = path(2);
    let pcol = [1, 1];
    let mut found = BTreeSet::new();
    let mut o = Vf2Options::new().with_vertex_colors(&colors, &pcol);
    c6.get_subisomorphisms_vf2_callback(&pattern, &mut o, |_, m21| {
        found.insert((m21[0].min(m21[1]), m21[0].max(m21[1])));
        true
    })
    .unwrap();
    assert_eq!(found, BTreeSet::from([(0, 1), (1, 2)]));
}

// ---------------------------------------------------------------------------
// LAD (values from igraph's examples/simple/igraph_subisomorphic_lad.out)
// ---------------------------------------------------------------------------

fn lad_graphs() -> (Graph, Graph) {
    let target = Graph::from_edges(
        &[
            (0, 1),
            (0, 4),
            (0, 6),
            (1, 4),
            (1, 2),
            (2, 3),
            (3, 4),
            (3, 5),
            (3, 7),
            (3, 8),
            (4, 5),
            (4, 6),
            (5, 6),
            (5, 8),
            (7, 8),
        ],
        9,
        false,
    )
    .unwrap();
    let pattern =
        Graph::from_edges(&[(0, 1), (0, 4), (1, 4), (1, 2), (2, 3), (3, 4)], 5, false).unwrap();
    (pattern, target)
}

#[test]
fn lad_matches_igraph_example() {
    let (pattern, target) = lad_graphs();
    let map = pattern
        .subisomorphic_lad(&target, None, false)
        .unwrap()
        .unwrap();
    assert_eq!(map, vec![1, 0, 6, 5, 4]);
    let all = pattern
        .get_subisomorphisms_lad(&target, None, false)
        .unwrap();
    assert_eq!(all.len(), 20);
    assert_eq!(all[0], vec![1, 0, 6, 5, 4]);
    assert_eq!(all[19], vec![7, 8, 5, 4, 3]);

    let induced = pattern
        .get_subisomorphisms_lad(&target, None, true)
        .unwrap();
    assert_eq!(
        induced,
        vec![
            vec![0, 1, 2, 3, 4],
            vec![5, 3, 2, 1, 4],
            vec![5, 4, 1, 2, 3],
            vec![0, 4, 3, 2, 1]
        ]
    );

    let domains = vec![
        vec![0, 2, 8],
        vec![4, 5, 6, 7],
        vec![1, 3, 5, 6, 7, 8],
        vec![0, 2, 8],
        vec![1, 3, 7, 8],
    ];
    let with_dom = pattern
        .get_subisomorphisms_lad(&target, Some(&domains), false)
        .unwrap();
    assert_eq!(with_dom, vec![vec![0, 4, 3, 2, 1]]);
    assert_eq!(
        pattern
            .subisomorphic_lad(&target, Some(&domains), false)
            .unwrap(),
        Some(vec![0, 4, 3, 2, 1])
    );
}

#[test]
fn lad_agrees_with_vf2_and_validates_domains() {
    let (pattern, target) = lad_graphs();
    let vf2 = target
        .count_subisomorphisms_vf2(&pattern, &mut Vf2Options::new())
        .unwrap();
    let lad = pattern
        .get_subisomorphisms_lad(&target, None, false)
        .unwrap()
        .len();
    assert_eq!(vf2, lad);
    // No 4-clique in the target.
    assert_eq!(
        complete(4).subisomorphic_lad(&target, None, false).unwrap(),
        None
    );
    // Wrong number of domains.
    let err = pattern
        .subisomorphic_lad(&target, Some(&[vec![0]]), false)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
}

// ---------------------------------------------------------------------------
// Bliss (values from igraph's tests/unit/bliss_automorphisms.out)
// ---------------------------------------------------------------------------

#[test]
fn bliss_group_sizes() {
    assert_eq!(
        petersen()
            .count_automorphisms_bliss(None, BlissSh::F)
            .unwrap()
            .group_size,
        "120"
    );
    let k23 = complete(23);
    let info = k23.count_automorphisms_bliss(None, BlissSh::F).unwrap();
    assert_eq!(info.group_size, "25852016738884976640000");
    assert_eq!(info.group_size_u128(), Some(25852016738884976640000));
    assert_close(
        k23.count_automorphisms(None).unwrap(),
        2.585201673888498e22,
        1e8,
    );
    let star = directed_star(17);
    assert_eq!(
        star.count_automorphisms_bliss(None, BlissSh::F)
            .unwrap()
            .group_size,
        "20922789888000"
    );
    assert_eq!(
        Graph::new(0, false)
            .count_automorphisms_bliss(None, BlissSh::F)
            .unwrap()
            .group_size,
        "1"
    );
    assert_eq!(
        Graph::new(1, false)
            .count_automorphisms_bliss(None, BlissSh::F)
            .unwrap()
            .group_size,
        "1"
    );
}

#[test]
fn bliss_heuristics_agree() {
    let p = petersen();
    for sh in BlissSh::ALL {
        let info = p.count_automorphisms_bliss(None, sh).unwrap();
        assert_eq!(info.group_size, "120", "{sh:?}");
        assert_eq!(info.group_size_f64(), Some(120.0));
        let (gens, info) = p.automorphism_group_bliss(None, sh).unwrap();
        assert_eq!(gens.len() as u64, info.nof_generators);
        assert!(gens.iter().all(|g| is_isomorphism(&p, &p, g)));
        let (lab, _) = p.canonical_permutation_bliss(None, sh).unwrap();
        assert_eq!(lab.len(), 10);
    }
    assert_eq!(BlissSh::default(), BlissSh::Fl);
    assert_eq!(BlissSh::try_from(5).unwrap(), BlissSh::Fsm);
}

#[test]
fn automorphism_generators_generate_the_group() {
    // Close the generators under composition and check the group order.
    let p = petersen();
    let gens = p.automorphism_group(None).unwrap();
    let id: Vec<i64> = (0..10).collect();
    let mut group: HashSet<Vec<i64>> = HashSet::from([id.clone()]);
    let mut frontier = vec![id];
    while let Some(g) = frontier.pop() {
        for s in &gens {
            let h: Vec<i64> = g.iter().map(|&x| s[x as usize]).collect();
            if group.insert(h.clone()) {
                frontier.push(h);
            }
        }
    }
    assert_eq!(group.len(), 120);
}

#[test]
fn bliss_isomorphism_with_colors() {
    rng::seed(3).unwrap();
    let k = karate();
    let perm = random_perm(34);
    let k2 = relabel(&k, &perm);
    let r = k.isomorphic_bliss(&k2, None, None, BlissSh::Fsm).unwrap();
    let m = r.mapping.as_ref().unwrap();
    assert!(is_isomorphism(&k, &k2, &m.map12));
    assert_eq!(r.info1.group_size, r.info2.group_size);
    assert_eq!(
        r.info1.group_size,
        k.count_automorphisms_bliss(None, BlissSh::Fl)
            .unwrap()
            .group_size
    );

    // Mark the two leaders (0 and 33) and carry the colors over.
    let mut c1 = vec![0; 34];
    c1[0] = 1;
    c1[33] = 2;
    let mut c2 = vec![0; 34];
    c2[perm[0] as usize] = 1;
    c2[perm[33] as usize] = 2;
    let r = k
        .isomorphic_bliss(&k2, Some(&c1), Some(&c2), BlissSh::Fl)
        .unwrap();
    let m = r.mapping.unwrap();
    assert_eq!(m.map12[0], perm[0]);
    assert_eq!(m.map12[33], perm[33]);
    // Swapping the leaders' colors makes them non-isomorphic (as colored
    // graphs): the leaders have different degrees.
    c2.swap(perm[0] as usize, perm[33] as usize);
    let r = k
        .isomorphic_bliss(&k2, Some(&c1), Some(&c2), BlissSh::Fl)
        .unwrap();
    assert!(!r.is_isomorphic());
}

#[test]
fn bliss_different_sizes() {
    let r = cycle(4)
        .isomorphic_bliss(&cycle(5), None, None, BlissSh::Fl)
        .unwrap();
    assert!(r.mapping.is_none());
    assert!(r.info1.group_size.is_empty());
    assert_eq!(r.info1.group_size_f64(), None);
}

#[test]
fn canonical_forms_are_invariants() {
    rng::seed(11).unwrap();
    let k = karate();
    let canon = k.canonical_form(None).unwrap();
    for _ in 0..5 {
        let perm = random_perm(34);
        let k2 = relabel(&k, &perm);
        assert!(k2.canonical_form(None).unwrap() == canon);
    }
    let mut other = k.clone();
    other.delete_edges(5).unwrap();
    assert!(other.canonical_form(None).unwrap() != canon);
    // Canonical permutations are permutations.
    let mut lab = k.canonical_permutation(None).unwrap();
    lab.sort();
    assert_eq!(lab, (0..34).collect::<Vec<_>>());
}

#[test]
fn invert_permutation_roundtrip() {
    rng::seed(5).unwrap();
    let p = random_perm(50);
    let q = invert_permutation(&p).unwrap();
    for i in 0..50 {
        assert_eq!(q[p[i] as usize], i as i64);
    }
    assert_eq!(invert_permutation(&q).unwrap(), p);
    assert!(invert_permutation(&[0, 5]).is_err());
}

// ---------------------------------------------------------------------------
// Isomorphism classes
// ---------------------------------------------------------------------------

#[test]
fn graph_counts_match_oeis() {
    let undirected: Vec<usize> = (0..8).map(|n| graph_count(n, false).unwrap()).collect();
    assert_eq!(undirected, vec![1, 1, 2, 4, 11, 34, 156, 1044]); // A000088
    let directed: Vec<usize> = (0..5).map(|n| graph_count(n, true).unwrap()).collect();
    assert_eq!(directed, vec![1, 1, 3, 16, 218]); // A000273
    assert_eq!(
        graph_count(1000, false).unwrap_err().kind(),
        ErrorKind::Overflow
    );
}

#[test]
fn isoclass_create_roundtrip() {
    for (size, directed) in [
        (3, true),
        (4, true),
        (3, false),
        (4, false),
        (5, false),
        (6, false),
    ] {
        let n = graph_count(size, directed).unwrap();
        let mut prev_edges = 0;
        for class in 0..n {
            let g = Graph::isoclass_create(size, class, directed).unwrap();
            assert_eq!(g.vcount(), size);
            assert_eq!(g.isoclass().unwrap(), class);
            if class == 0 {
                assert_eq!(g.ecount(), 0);
            }
            prev_edges = g.ecount();
        }
        // The last class is the complete graph.
        let full = size * (size - 1) / if directed { 1 } else { 2 };
        assert_eq!(prev_edges, full);
    }
    assert!(Graph::isoclass_create(3, 4, false).is_err());
    assert!(Graph::isoclass_create(7, 0, false).is_err());
    assert!(Graph::new(7, false).isoclass().is_err());
}

#[test]
fn isoclass_subgraph_consistency() {
    let k = karate();
    // The isoclass of an induced subgraph equals the isoclass of the
    // subgraph built by hand.
    for vids in [
        [0i64, 1, 2, 3],
        [0, 4, 5, 6],
        [23, 25, 27, 31],
        [8, 30, 32, 33],
    ] {
        let mut edges = vec![];
        for i in 0..4 {
            for j in i + 1..4 {
                if k.get_eid(vids[i], vids[j], false).unwrap().is_some() {
                    edges.push((i as i64, j as i64));
                }
            }
        }
        let sub = Graph::from_edges(&edges, 4, false).unwrap();
        assert_eq!(k.isoclass_subgraph(&vids).unwrap(), sub.isoclass().unwrap());
    }
    assert_eq!(k.isoclass_subgraph(&[0, 1, 2, 3]).unwrap(), 10); // K4
}

// ---------------------------------------------------------------------------
// Motifs
// ---------------------------------------------------------------------------

#[test]
fn karate_four_motifs() {
    // From igraph's examples/simple/igraph_motifs_randesu.out.
    let hist = karate().motifs_randesu(4, None).unwrap();
    assert_eq!(hist.len(), 11);
    for i in [0, 1, 2, 3, 5] {
        assert!(hist[i].is_nan());
    }
    let sum: f64 = hist.iter().filter(|x| !x.is_nan()).sum();
    let expected = [
        (4, 0.464664),
        (6, 0.288193),
        (7, 0.191282),
        (8, 0.0152349),
        (9, 0.0359712),
        (10, 0.0046551),
    ];
    for (i, frac) in expected {
        assert_close(hist[i] / sum, frac, 1e-6);
    }
    // The total equals the number of connected induced 4-subgraphs.
    assert_eq!(sum, karate().motifs_randesu_no(4, None).unwrap());
}

#[test]
fn motifs_three_vs_triangles() {
    let k = karate();
    let hist = k.motifs_randesu(3, None).unwrap();
    assert_eq!(hist[3], k.count_triangles().unwrap());
    assert_eq!(hist[3], 45.0);
    // Open triads (2-paths) = sum over v of C(deg, 2) - 3 * triangles.
    let deg = k.degree(.., NeighborMode::All, Loops::Twice).unwrap();
    let wedges: i64 = deg.iter().map(|d| d * (d - 1) / 2).sum();
    assert_eq!(hist[2], (wedges - 3 * 45) as f64);
}

#[test]
fn motifs_randesu_no_values() {
    // From igraph's tests/unit/igraph_motifs_randesu_no.out.
    let k50 = complete(50);
    assert_eq!(k50.motifs_randesu_no(3, None).unwrap(), 19600.0);
    assert_eq!(k50.motifs_randesu_no(4, None).unwrap(), 230300.0);
    let tri_plus = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 4, false).unwrap();
    assert_eq!(tri_plus.motifs_randesu_no(4, None).unwrap(), 0.0);
    assert_eq!(
        Graph::new(0, false).motifs_randesu_no(3, None).unwrap(),
        0.0
    );
    let cp = [0.1, 0.1, 0.1];
    let err = tri_plus.motifs_randesu_no(14, Some(&cp)).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    // Sampling can only lose motifs.
    rng::seed(42).unwrap();
    let sampled = k50.motifs_randesu_no(3, Some(&cp)).unwrap();
    assert!(sampled < 19600.0 && sampled > 10000.0);
}

#[test]
fn motifs_randesu_estimate_values() {
    let k50 = complete(50);
    let all: Vec<i64> = (0..50).collect();
    let est = k50
        .motifs_randesu_estimate(3, None, MotifSample::Vertices(&all))
        .unwrap();
    assert_eq!(est, 19600.0);
    // From igraph's tests/unit/igraph_motifs_randesu_estimate.out.
    let first41: Vec<i64> = (0..41).collect(); // igraph_vector_int_init_range(0, 41)
    let est = k50
        .motifs_randesu_estimate(3, None, MotifSample::Vertices(&first41))
        .unwrap();
    assert_eq!(est, 23800.0);
    rng::seed(1).unwrap();
    let est = k50
        .motifs_randesu_estimate(3, None, MotifSample::Random(20))
        .unwrap();
    assert!(est > 0.0);
    let tri_plus = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 4, false).unwrap();
    assert!(
        tri_plus
            .motifs_randesu_estimate(4, None, MotifSample::Random(40))
            .is_err()
    );
    assert_eq!(
        tri_plus
            .motifs_randesu_estimate(4, None, MotifSample::Vertices(&[0, 9]))
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidVertexId
    );
}

#[test]
fn motifs_callback_agrees_and_stops() {
    let k = karate();
    let hist = k.motifs_randesu(3, None).unwrap();
    let mut counts = [0.0; 4];
    k.motifs_randesu_callback(3, None, |vids, class| {
        assert_eq!(vids.len(), 3);
        assert_eq!(k.isoclass_subgraph(vids).unwrap(), class);
        counts[class] += 1.0;
        true
    })
    .unwrap();
    assert_eq!(&counts[2..], &hist[2..]);

    let mut seen = 0;
    k.motifs_randesu_callback(3, None, |_, _| {
        seen += 1;
        seen < 7
    })
    .unwrap();
    assert_eq!(seen, 7);

    let res =
        std::panic::catch_unwind(|| k.motifs_randesu_callback(3, None, |_, _| panic!("motif")));
    assert!(res.is_err());
}

#[test]
fn directed_motifs_feed_forward_loops() {
    // Two feed-forward loops a->b->c, a->c sharing the edge 0->2.
    let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 2), (0, 3), (3, 2)], 4, true).unwrap();
    let ffl = Graph::from_edges(&[(0, 1), (1, 2), (0, 2)], 3, true).unwrap();
    let class = ffl.isoclass().unwrap();
    let hist = g.motifs_randesu(3, None).unwrap();
    assert_eq!(hist.len(), 16);
    assert_eq!(hist[class], 2.0);
    let census = g.triad_census().unwrap();
    assert_eq!(census["030T"], 2.0);
}

// ---------------------------------------------------------------------------
// Dyad and triad census (values from igraph's unit tests)
// ---------------------------------------------------------------------------

#[test]
fn dyad_census_values() {
    let edges = [
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 3),
        (2, 0),
        (2, 0),
        (2, 3),
        (3, 4),
        (3, 4),
    ];
    let d = Graph::from_edges(&edges, 6, true)
        .unwrap()
        .dyad_census()
        .unwrap();
    assert_eq!((d.mutual, d.asymmetric, d.null), (1.0, 4.0, 10.0));
    let u = Graph::from_edges(&edges, 6, false)
        .unwrap()
        .dyad_census()
        .unwrap();
    assert_eq!((u.mutual, u.asymmetric, u.null), (5.0, 0.0, 10.0));
    let two = Graph::from_edges(&[(0, 1)], 2, true)
        .unwrap()
        .dyad_census()
        .unwrap();
    assert_eq!((two.mutual, two.asymmetric, two.null), (0.0, 1.0, 0.0));
}

#[test]
fn triad_census_values() {
    let flat = [
        0, 2, 1, 4, 2, 5, 2, 7, 3, 7, 3, 8, 4, 2, 5, 8, 6, 0, 6, 1, 6, 2, 7, 0, 8, 0, 8, 2, 8, 3,
        8, 5, 9, 2, 9, 3, 9, 4, 9, 5,
    ];
    let g = Graph::from_flat_edges(&flat, 10, true).unwrap();
    let t = g.triad_census().unwrap();
    assert_eq!(
        t.counts,
        [
            25.0, 45.0, 7.0, 7.0, 12.0, 11.0, 2.0, 4.0, 4.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
        ]
    );
    assert_eq!(t.total(), 120.0); // C(10, 3)
    assert_eq!(t.get("012"), Some(45.0));
    assert_eq!(t.get("999"), None);
    assert_eq!(t[1], t["012"]);
    assert_eq!(t.iter().count(), 16);
    assert_eq!(TriadCensus::NAMES[15], "300");

    // Undirected: collapse the edges (dedupe), then census.
    let mut pairs: BTreeSet<(i64, i64)> = BTreeSet::new();
    for e in flat.chunks(2) {
        pairs.insert((e[0].min(e[1]), e[0].max(e[1])));
    }
    let edges: Vec<(i64, i64)> = pairs.into_iter().collect();
    let u = Graph::from_edges(&edges, 10, false).unwrap();
    let t = u.triad_census().unwrap();
    assert_eq!(
        t.counts,
        [
            25.0, 0.0, 52.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 37.0, 0.0, 0.0, 0.0, 0.0, 6.0
        ]
    );
    // The number of complete triads is the number of triangles.
    assert_eq!(t["300"], u.count_triangles().unwrap());
    igraph::error::take_warnings();
}

// ---------------------------------------------------------------------------
// Triangles
// ---------------------------------------------------------------------------

#[test]
fn triangle_identities() {
    let k = karate();
    let total = k.count_triangles().unwrap();
    assert_eq!(total, 45.0);
    let per_vertex = k.count_adjacent_triangles(..).unwrap();
    assert_eq!(per_vertex.iter().sum::<f64>(), 3.0 * total);
    assert_eq!(k.list_triangles().unwrap().len(), 45);
    assert_eq!(
        k.count_adjacent_triangles(&[0, 33][..]).unwrap(),
        vec![per_vertex[0], per_vertex[33]]
    );
    // K5: C(5,3) = 10 triangles, each vertex in C(4,2) = 6.
    let k5 = complete(5);
    assert_eq!(k5.count_triangles().unwrap(), 10.0);
    assert_eq!(k5.count_adjacent_triangles(..).unwrap(), vec![6.0; 5]);
    let tris = k5.list_triangles().unwrap();
    let set: BTreeSet<[i64; 3]> = tris
        .into_iter()
        .map(|mut t| {
            t.sort();
            t
        })
        .collect();
    assert_eq!(set.len(), 10);
    assert!(cycle(5).list_triangles().unwrap().is_empty());
    assert_eq!(
        k.count_adjacent_triangles(99).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
}

// ---------------------------------------------------------------------------
// Graphlets
// ---------------------------------------------------------------------------

#[test]
fn graphlets_recover_planted_groups() {
    // Two overlapping groups: a heavy 4-clique {0,1,2,3} (weight 4) and a
    // light triangle {3,4,5} (weight 1); the edge weights are the sums.
    let mut edges = vec![];
    let mut weights = vec![];
    for i in 0..4 {
        for j in i + 1..4 {
            edges.push((i, j));
            weights.push(4.0);
        }
    }
    for (a, b) in [(3, 4), (3, 5), (4, 5)] {
        edges.push((a, b));
        weights.push(1.0);
    }
    let g = Graph::from_edges(&edges, 6, false).unwrap();

    let basis = g.graphlets_candidate_basis(&weights).unwrap();
    assert_eq!(basis.cliques.len(), basis.thresholds.len());
    let as_sets: Vec<BTreeSet<i64>> = basis
        .cliques
        .iter()
        .map(|c| c.iter().copied().collect())
        .collect();
    assert!(as_sets.contains(&BTreeSet::from([0, 1, 2, 3])));
    assert!(as_sets.contains(&BTreeSet::from([3, 4, 5])));

    let gl = g.graphlets(&weights, 1000).unwrap();
    let top: BTreeSet<i64> = gl.cliques[0].iter().copied().collect();
    assert_eq!(top, BTreeSet::from([0, 1, 2, 3]));
    // igraph's EM normalizes a clique of size n by n (n + 1) / 2, so a
    // group of weight w on its own converges to w (n - 1) / (n + 1).
    let unscale = |mu: f64, n: usize| mu * (n as f64 + 1.0) / (n as f64 - 1.0);
    assert_close(unscale(gl.weights[0], 4), 4.0, 1e-3);
    let tri = gl
        .cliques
        .iter()
        .position(|c| c.len() == 3 && c.contains(&4))
        .unwrap();
    assert_close(unscale(gl.weights[tri], 3), 1.0, 1e-3);

    // Projection onto the true basis, with or without a starting point.
    let cl = vec![vec![0, 1, 2, 3], vec![3, 4, 5]];
    let mu = g.graphlets_project(&weights, &cl, None, 1000).unwrap();
    assert_close(unscale(mu[0], 4), 4.0, 1e-3);
    assert_close(unscale(mu[1], 3), 1.0, 1e-3);
    assert_eq!(mu, gl.weights);
    let mu2 = g
        .graphlets_project(&weights, &cl, Some(&[2.0, 2.0]), 1000)
        .unwrap();
    assert_close(mu2[0], mu[0], 1e-6);

    assert!(g.graphlets(&[1.0], 10).is_err());
    assert!(
        g.graphlets_project(&weights, &cl, Some(&[1.0]), 10)
            .is_err()
    );
}

// ---------------------------------------------------------------------------
// Use-case story
// ---------------------------------------------------------------------------

/// Story: a chemist enumerates every labeled graph on 4 vertices (2^6 = 64
/// of them), and wants the distinct *shapes*. Canonical forms act as hash
/// keys up to isomorphism, so deduplicating yields exactly the 11 unlabeled
/// graphs known to igraph, each matching one isomorphism class — and the
/// number of labelings of each shape is 4! / |Aut|, the orbit-stabilizer
/// theorem.
#[test]
fn story_enumerate_shapes_up_to_isomorphism() {
    let pairs: Vec<(i64, i64)> = (0..4)
        .flat_map(|i| (i + 1..4).map(move |j| (i, j)))
        .collect();
    let mut shapes: Vec<(Graph, usize)> = vec![];
    for mask in 0u32..64 {
        let edges: Vec<(i64, i64)> = pairs
            .iter()
            .enumerate()
            .filter(|(k, _)| mask & (1 << k) != 0)
            .map(|(_, &e)| e)
            .collect();
        let g = Graph::from_edges(&edges, 4, false).unwrap();
        let canon = g.canonical_form(None).unwrap();
        match shapes.iter_mut().find(|(c, _)| *c == canon) {
            Some((_, n)) => *n += 1,
            None => shapes.push((canon, 1)),
        }
    }
    assert_eq!(shapes.len(), graph_count(4, false).unwrap());
    let classes: BTreeSet<usize> = shapes.iter().map(|(g, _)| g.isoclass().unwrap()).collect();
    assert_eq!(classes, (0..11).collect());
    for (g, labelings) in &shapes {
        let aut = g.count_automorphisms(None).unwrap() as usize;
        assert_eq!(
            labelings * aut,
            24,
            "orbit-stabilizer for {:?}",
            g.edge_list()
        );
        // Every shape is isomorphic to the representative of its class.
        let rep = Graph::isoclass_create(4, g.isoclass().unwrap(), false).unwrap();
        assert!(g.isomorphic(&rep).unwrap());
    }
}

/// Story: in the karate club, find where the two leaders (0 and 33) sit in
/// triangles, and locate all "friend of a friend" squares (4-cycles, as
/// induced subgraphs) passing through the instructor.
#[test]
fn story_karate_local_structure() {
    let k = karate();
    let tri = k.count_adjacent_triangles(&[0, 33][..]).unwrap();
    assert_eq!(tri, vec![18.0, 15.0]);

    let c4 = cycle(4);
    let mut squares = BTreeSet::new();
    // Restrict the pattern's vertex 0 to the instructor with a LAD domain.
    let mut domains: Vec<Vec<i64>> = vec![(0..34).collect(); 4];
    domains[0] = vec![0];
    for m in c4
        .get_subisomorphisms_lad(&k, Some(&domains), true)
        .unwrap()
    {
        assert_eq!(m[0], 0);
        let mut s = m.clone();
        s.sort();
        squares.insert(s);
    }
    // Cross-check with VF2 and a node compatibility closure.
    let mut vf2_squares = BTreeSet::new();
    let mut o = Vf2Options::new().with_node_compat(|kv, pv| (pv == 0) == (kv == 0));
    k.get_subisomorphisms_vf2_callback(&c4, &mut o, |_, m21| {
        let mut s = m21.to_vec();
        s.sort();
        // VF2 subgraphs are not induced: keep only chordless squares.
        let chord1 = k.get_eid(m21[0], m21[2], false).unwrap().is_some();
        let chord2 = k.get_eid(m21[1], m21[3], false).unwrap().is_some();
        if !chord1 && !chord2 {
            vf2_squares.insert(s);
        }
        true
    })
    .unwrap();
    assert_eq!(squares, vf2_squares);
    assert!(!squares.is_empty());
}

// ---------------------------------------------------------------------------
// Input validation (inputs igraph itself does not check, and that would
// otherwise corrupt memory or abort the process)
// ---------------------------------------------------------------------------

#[test]
fn lad_rejects_out_of_range_domains() {
    let (pattern, target) = lad_graphs();
    for bad in [9, -1, 1_000_000] {
        let mut domains: Vec<Vec<i64>> = vec![(0..9).collect(); 5];
        domains[2] = vec![0, bad];
        let err = pattern
            .subisomorphic_lad(&target, Some(&domains), false)
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidVertexId, "{bad}");
        assert!(
            pattern
                .get_subisomorphisms_lad(&target, Some(&domains), true)
                .is_err()
        );
    }
    // Empty domains are fine: nothing can match.
    let domains: Vec<Vec<i64>> = vec![vec![]; 5];
    assert_eq!(
        pattern
            .subisomorphic_lad(&target, Some(&domains), false)
            .unwrap(),
        None
    );
}

#[test]
fn lad_edge_cases() {
    let (_, target) = lad_graphs();
    // The null pattern occurs exactly once, with the empty mapping.
    let null = Graph::new(0, false);
    assert_eq!(
        null.subisomorphic_lad(&target, None, false).unwrap(),
        Some(vec![])
    );
    assert_eq!(
        null.get_subisomorphisms_lad(&target, None, false).unwrap(),
        vec![Vec::<i64>::new()]
    );
    // A pattern larger than the target never matches.
    assert!(
        complete(10)
            .get_subisomorphisms_lad(&target, None, false)
            .unwrap()
            .is_empty()
    );
    // Mixed directedness is an error.
    let directed = Graph::from_edges(&[(0, 1)], 2, true).unwrap();
    assert_eq!(
        directed
            .subisomorphic_lad(&target, None, false)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    // Directed LAD respects edge directions: a directed 2-path occurs in a
    // directed 3-cycle once per start vertex, i.e. 3 times.
    let dtri = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, true).unwrap();
    let dpath = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    assert_eq!(
        dpath
            .get_subisomorphisms_lad(&dtri, None, false)
            .unwrap()
            .len(),
        3
    );
    // Induced: the triangle has a chord 2->0 that the path lacks.
    assert!(
        dpath
            .get_subisomorphisms_lad(&dtri, None, true)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn graphlets_on_edgeless_graphs() {
    let g = Graph::new(4, false);
    let basis = g.graphlets_candidate_basis(&[]).unwrap();
    assert!(basis.cliques.is_empty() && basis.thresholds.is_empty());
    let gl = g.graphlets(&[], 10).unwrap();
    assert!(gl.cliques.is_empty() && gl.weights.is_empty());
    assert!(g.graphlets(&[1.0], 10).is_err());
}

#[test]
fn graphlets_project_validates_cliques() {
    let g = complete(4);
    let w = [1.0; 6];
    for bad in [vec![0, 4], vec![-1, 2], vec![0, 1, 1]] {
        let err = g
            .graphlets_project(&w, std::slice::from_ref(&bad), None, 5)
            .unwrap_err();
        let expected = if bad.iter().all(|&v| (0..4).contains(&v)) {
            ErrorKind::InvalidValue
        } else {
            ErrorKind::InvalidVertexId
        };
        assert_eq!(err.kind(), expected, "{bad:?}");
    }
    // Non-simple graphs are rejected by igraph.
    let multi = Graph::from_edges(&[(0, 1), (0, 1)], 2, false).unwrap();
    assert_eq!(
        multi.graphlets(&[1.0, 1.0], 5).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    // A single clique covering all edges of weight 1 of K4 converges to
    // (n - 1) / (n + 1) = 3/5.
    let mu = g
        .graphlets_project(&w, &[vec![0, 1, 2, 3]], None, 200)
        .unwrap();
    assert_close(mu[0], 0.6, 1e-3);
}

#[test]
fn isoclass_subgraph_rejects_repeated_vertices() {
    let k = karate();
    assert_eq!(
        k.isoclass_subgraph(&[0, 1, 1][..]).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        k.isoclass_subgraph(&[0, 1, 99][..]).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        k.isoclass_subgraph(&[0, 1][..]).unwrap_err().kind(),
        ErrorKind::Unimplemented
    );
}

#[test]
fn vf2_rejects_self_loops() {
    let g = Graph::from_edges(&[(0, 1), (1, 1)], 2, false).unwrap();
    assert_eq!(
        g.isomorphic_vf2(&g, &mut Vf2Options::new())
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    // ... while Bliss and the generic test handle them.
    assert!(g.isomorphic(&g).unwrap());
    assert_eq!(g.count_automorphisms(None).unwrap(), 1.0);
}

#[test]
fn canonical_permutation_convention() {
    // igraph 1.0: labeling[i] is the original vertex that becomes vertex i.
    rng::seed(17).unwrap();
    let p = petersen();
    let q = relabel(&p, &random_perm(10));
    for g in [&p, &q] {
        let labeling = g.canonical_permutation(None).unwrap();
        let pos = invert_permutation(&labeling).unwrap();
        let mut edges: Vec<(i64, i64)> = g
            .edge_list()
            .into_iter()
            .map(|(a, b)| {
                let (a, b) = (pos[a as usize], pos[b as usize]);
                (a.min(b), a.max(b))
            })
            .collect();
        edges.sort();
        let by_hand = Graph::from_edges(&edges, 10, false).unwrap();
        assert!(by_hand == g.canonical_form(None).unwrap());
    }
    assert!(p.canonical_form(None).unwrap() == q.canonical_form(None).unwrap());
    // Colors are part of the canonical form: the Petersen graph is
    // vertex-transitive, so marking any single vertex gives the same form.
    let mut c0 = vec![0; 10];
    c0[0] = 1;
    let mut c7 = vec![0; 10];
    c7[7] = 1;
    assert!(p.canonical_form(Some(&c0)).unwrap() == p.canonical_form(Some(&c7)).unwrap());
    assert_eq!(p.count_automorphisms(Some(&c0)).unwrap(), 12.0); // 120 / 10
}

#[test]
fn bliss_rejects_bad_colors() {
    let c = cycle(4);
    assert_eq!(
        c.count_automorphisms(Some(&[0, 1])).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    // Colors are arbitrary labels (negative ones too): pinning vertex 1
    // leaves the identity and the reflection through 1 and 3.
    assert_eq!(c.count_automorphisms(Some(&[0, -1, 0, 0])).unwrap(), 2.0);
    // Bliss stores colors as C ints.
    assert_eq!(
        c.count_automorphisms(Some(&[0, 1 << 40, 0, 0]))
            .unwrap_err()
            .kind(),
        ErrorKind::Overflow
    );
}

// ---------------------------------------------------------------------------
// Cross-module checks (constructors, operators, mixing, structural)
// ---------------------------------------------------------------------------

#[test]
fn famous_graphs_have_known_symmetry_groups() {
    // Orders of the automorphism groups of classic graphs.
    let table = [
        ("Tetrahedral", 24),
        ("Octahedral", 48),
        ("Cubical", 48),
        ("Icosahedral", 120),
        ("Dodecahedral", 120),
        ("Petersen", 120),
        ("Heawood", 336),
        ("Coxeter", 336),
        ("Franklin", 48),
        ("Frucht", 1),
        ("Tutte", 3),
        ("Levi", 1440),
    ];
    for (name, order) in table {
        let g = Graph::famous(name).unwrap();
        assert_eq!(g.count_automorphisms(None).unwrap(), order as f64, "{name}");
        let info = g.count_automorphisms_bliss(None, BlissSh::Fsm).unwrap();
        assert_eq!(info.group_size_u128(), Some(order), "{name}");
        // Frucht's graph is the smallest cubic graph without symmetries.
        let gens = g.automorphism_group(None).unwrap();
        assert_eq!(gens.is_empty(), order == 1, "{name}");
        assert!(gens.iter().all(|p| is_isomorphism(&g, &g, p)), "{name}");
    }
    // K_n and C_n: n! and the dihedral group of order 2n.
    assert_eq!(
        Graph::full(6, false, false)
            .unwrap()
            .count_automorphisms(None)
            .unwrap(),
        720.0
    );
    assert_eq!(
        Graph::ring(9, false, false, true)
            .unwrap()
            .count_automorphisms(None)
            .unwrap(),
        18.0
    );
}

#[test]
fn famous_graphs_agree_with_other_constructions() {
    // The hand-built Petersen graph (outer 5-cycle, inner pentagram, spokes)
    // is the catalogued one, and the generalized Petersen graph GP(5, 2).
    let mut edges = vec![];
    for i in 0..5 {
        edges.push((i, (i + 1) % 5));
        edges.push((5 + i, 5 + (i + 2) % 5));
        edges.push((i, i + 5));
    }
    let by_hand = Graph::from_edges(&edges, 10, false).unwrap();
    let p = petersen();
    assert!(p.isomorphic(&by_hand).unwrap());
    assert!(p.canonical_form(None).unwrap() == by_hand.canonical_form(None).unwrap());
    assert!(
        p.isomorphic(&Graph::generalized_petersen(5, 2).unwrap())
            .unwrap()
    );
    // The Heawood graph has LCF notation [5, -5]^7.
    let heawood = Graph::lcf(14, &[5, -5], 7).unwrap();
    assert!(
        heawood
            .isomorphic(&Graph::famous("Heawood").unwrap())
            .unwrap()
    );
    // The cube is the 3-dimensional hypercube, and GP(4, 1).
    let cube = Graph::famous("Cubical").unwrap();
    assert!(
        cube.isomorphic(&Graph::hypercube(3, false).unwrap())
            .unwrap()
    );
    assert!(
        cube.isomorphic(&Graph::generalized_petersen(4, 1).unwrap())
            .unwrap()
    );
    // Karate from the catalogue is the graph of tests/common.
    assert!(
        Graph::famous("Zachary")
            .unwrap()
            .is_same_graph(&karate())
            .unwrap()
    );
}

#[test]
fn canonical_form_is_the_permuted_graph() {
    rng::seed(23).unwrap();
    let g = relabel(&karate(), &random_perm(34));
    let labeling = g.canonical_permutation(None).unwrap();
    let permuted = g.permute_vertices(&labeling).unwrap();
    let canon = g.canonical_form(None).unwrap();
    // Same labeled edge set, only the edge order may differ.
    let norm = |h: &Graph| {
        let mut e: Vec<(i64, i64)> = h
            .edge_list()
            .into_iter()
            .map(|(a, b)| (a.min(b), a.max(b)))
            .collect();
        e.sort();
        e
    };
    assert_eq!(norm(&permuted), norm(&canon));
    // Relabeled graphs are isomorphic but (almost surely) not the same
    // labeled graph, while their canonical forms are.
    let h = relabel(&g, &random_perm(34));
    assert!(g.isomorphic(&h).unwrap());
    assert!(!g.is_same_graph(&h).unwrap());
    assert!(
        canon
            .is_same_graph(&h.canonical_form(None).unwrap())
            .unwrap()
    );
}

#[test]
fn isoclass_create_matches_constructors() {
    // The last class is always the complete graph, class 0 the empty one.
    for n in 3..=6 {
        let last = graph_count(n, false).unwrap() - 1;
        let full = Graph::full(n, false, false).unwrap();
        assert_eq!(full.isoclass().unwrap(), last);
        assert!(
            Graph::isoclass_create(n, last, false)
                .unwrap()
                .isomorphic(&full)
                .unwrap()
        );
    }
    for n in 3..=4 {
        let last = graph_count(n, true).unwrap() - 1;
        assert_eq!(
            Graph::full(n, true, false).unwrap().isoclass().unwrap(),
            last
        );
    }
    // Stars, cycles and paths land in the classes igraph assigns them.
    let c5 = Graph::ring(5, false, false, true).unwrap();
    let class = c5.isoclass().unwrap();
    assert!(
        Graph::isoclass_create(5, class, false)
            .unwrap()
            .isomorphic(&c5)
            .unwrap()
    );
    let s4 = Graph::star(4, StarMode::Undirected, 0).unwrap();
    let p4 = Graph::ring(4, false, false, false).unwrap();
    assert_ne!(s4.isoclass().unwrap(), p4.isoclass().unwrap());
}

#[test]
fn triangle_counts_match_transitivity() {
    // Global clustering: 3 * triangles / connected triples.
    let k = karate();
    let triangles = k.count_triangles().unwrap();
    assert_eq!(triangles, 45.0);
    let deg = k.degree(.., NeighborMode::All, Loops::Twice).unwrap();
    let triples: f64 = deg.iter().map(|&d| (d * (d - 1) / 2) as f64).sum();
    assert_close(
        k.transitivity_undirected(TransitivityMode::Zero).unwrap(),
        3.0 * triangles / triples,
        1e-12,
    );
    // Local clustering: t(v) / C(d(v), 2).
    let t = k.count_adjacent_triangles(..).unwrap();
    let local = k
        .transitivity_local_undirected(.., TransitivityMode::Zero)
        .unwrap();
    for v in 0..34 {
        let pairs = (deg[v] * (deg[v] - 1) / 2) as f64;
        let expected = if pairs == 0.0 { 0.0 } else { t[v] / pairs };
        assert_close(local[v], expected, 1e-12);
    }
    // Triangles are exactly the 3-cliques.
    let mut listed: Vec<Vec<i64>> = k
        .list_triangles()
        .unwrap()
        .into_iter()
        .map(|t| {
            let mut t = t.to_vec();
            t.sort();
            t
        })
        .collect();
    listed.sort();
    let mut cliques: Vec<Vec<i64>> = k
        .cliques(3..=3, None)
        .unwrap()
        .into_iter()
        .map(|mut c| {
            c.sort();
            c
        })
        .collect();
    cliques.sort();
    assert_eq!(listed, cliques);
}

#[test]
fn dyad_census_matches_reciprocity() {
    rng::seed(99).unwrap();
    let g = Graph::erdos_renyi_game_gnp(40, 0.1, true, EdgeTypeSw::Simple, false).unwrap();
    let d = g.dyad_census().unwrap();
    assert_eq!(d.mutual + d.asymmetric + d.null, 40.0 * 39.0 / 2.0);
    assert_eq!(2.0 * d.mutual + d.asymmetric, g.ecount() as f64);
    assert_close(
        g.reciprocity(false, Reciprocity::Default).unwrap(),
        2.0 * d.mutual / (2.0 * d.mutual + d.asymmetric),
        1e-12,
    );
    assert_close(
        g.reciprocity(false, Reciprocity::Ratio).unwrap(),
        d.mutual / (d.mutual + d.asymmetric),
        1e-12,
    );
}

#[test]
fn motif_functions_validate_arguments() {
    let k = karate();
    for size in [0, 1, 2] {
        assert_eq!(
            k.motifs_randesu_no(size, None).unwrap_err().kind(),
            ErrorKind::InvalidValue
        );
        assert!(
            k.motifs_randesu_estimate(size, None, MotifSample::Random(3))
                .is_err()
        );
    }
    assert!(k.motifs_randesu(7, None).is_err());
    assert!(k.motifs_randesu(3, Some(&[0.0, 0.0])).is_err());
    assert!(k.motifs_randesu_no(3, Some(&[0.0])).is_err());
    assert!(
        k.motifs_randesu_estimate(3, None, MotifSample::Vertices(&[]))
            .is_err()
    );
    assert!(
        k.motifs_randesu_estimate(3, None, MotifSample::Vertices(&[34]))
            .is_err()
    );
    // A random sample cannot be larger than the graph.
    assert!(
        k.motifs_randesu_estimate(3, None, MotifSample::Random(35))
            .is_err()
    );
    // Sampling every vertex, in any order, gives the exact count.
    let all: Vec<i64> = (0..34).rev().collect();
    assert_eq!(
        k.motifs_randesu_estimate(4, None, MotifSample::Vertices(&all))
            .unwrap(),
        k.motifs_randesu_no(4, None).unwrap()
    );
}

#[test]
fn seeded_sampling_is_reproducible_in_parallel_threads() {
    // Every thread has its own default RNG: seeding one thread does not
    // disturb the others, and the same seed gives the same results.
    let run = |seed: u64| {
        rng::seed(seed).unwrap();
        let k = karate();
        let est = k
            .motifs_randesu_estimate(4, None, MotifSample::Random(10))
            .unwrap();
        let cut = k.motifs_randesu(4, Some(&[0.0, 0.2, 0.2, 0.2])).unwrap();
        let perm = random_perm(34);
        (est, cut, perm)
    };
    let expected = run(2024);
    let handles: Vec<_> = (0..8)
        .map(|_| std::thread::spawn(move || run(2024)))
        .collect();
    for h in handles {
        let (est, cut, perm) = h.join().unwrap();
        assert_eq!(est, expected.0);
        assert_eq!(perm, expected.2);
        // NaN != NaN: compare bit patterns.
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&cut), bits(&expected.1));
    }
}

// ---------------------------------------------------------------------------
// Review regressions
// ---------------------------------------------------------------------------

#[test]
fn failing_igraph_calls_inside_callbacks_are_contained() {
    // A failing igraph call inside a callback must only clean up its own
    // temporaries, not those of the igraph function running the callback
    // (this used to abort on an assertion, or worse).
    let k = karate();
    let expected: f64 = k
        .motifs_randesu(4, None)
        .unwrap()
        .iter()
        .filter(|x| !x.is_nan())
        .sum();
    for _ in 0..10 {
        let mut n = 0.0;
        k.motifs_randesu_callback(4, None, |_, _| {
            n += 1.0;
            assert!(k.isoclass_subgraph(&[0, 1][..]).is_err());
            true
        })
        .unwrap();
        assert_eq!(n, expected);
    }
    let k5 = complete(5);
    let mut n = 0;
    k5.get_isomorphisms_vf2_callback(&k5, &mut Vf2Options::new(), |_, _| {
        n += 1;
        assert!(invert_permutation(&[0, 0]).is_err());
        true
    })
    .unwrap();
    assert_eq!(n, 120);
    let mut o = Vf2Options::new().with_node_compat(|_, _| invert_permutation(&[1, 1]).is_err());
    assert_eq!(
        k5.count_subisomorphisms_vf2(&complete(3), &mut o).unwrap(),
        60
    );
    let mut n = 0;
    k5.get_subisomorphisms_vf2_callback(&complete(3), &mut Vf2Options::new(), |_, _| {
        n += 1;
        assert!(graph_count(100, false).is_err());
        true
    })
    .unwrap();
    assert_eq!(n, 60);
}

#[test]
fn oversized_arguments_are_errors() {
    // Sizes that do not fit in an igraph_int_t used to abort the process.
    let k = karate();
    for n in [usize::MAX, 1 << 63] {
        assert_eq!(
            k.motifs_randesu_estimate(3, None, MotifSample::Random(n))
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidValue
        );
        assert_eq!(
            graph_count(n, true).unwrap_err().kind(),
            ErrorKind::Overflow
        );
    }
    // The largest counts that fit in 64 bits (OEIS A000088, A000273).
    assert_eq!(graph_count(14, false).unwrap(), 29054155657235488);
    assert_eq!(graph_count(9, true).unwrap(), 13027956824399552);
    assert_eq!(
        graph_count(15, false).unwrap_err().kind(),
        ErrorKind::Overflow
    );
    assert_eq!(
        graph_count(10, true).unwrap_err().kind(),
        ErrorKind::Overflow
    );
    // The null graph has no motifs, whatever the sample.
    assert_eq!(
        Graph::new(0, false)
            .motifs_randesu_estimate(3, None, MotifSample::Random(0))
            .unwrap(),
        0.0
    );
}

#[test]
fn colored_canonical_forms_need_the_relabeled_colors() {
    // The canonical form does not carry the colors: a 3-path with no marked
    // vertex and one with a marked middle vertex have the same form...
    let p3 = Graph::ring(3, false, false, false).unwrap();
    let (plain, marked) = ([0, 0, 0], [0, 1, 0]);
    assert!(p3.canonical_form(Some(&plain)).unwrap() == p3.canonical_form(Some(&marked)).unwrap());
    assert!(
        !p3.isomorphic_bliss(&p3, Some(&plain), Some(&marked), BlissSh::Fl)
            .unwrap()
            .is_isomorphic()
    );
    // ... but the colors relabeled by the canonical labelings differ.
    let relabeled = |c: &[i64]| -> Vec<i64> {
        p3.canonical_permutation(Some(c))
            .unwrap()
            .iter()
            .map(|&v| c[v as usize])
            .collect()
    };
    assert_ne!(relabeled(&plain), relabeled(&marked));
    // Marking either end gives the same form and the same relabeled colors.
    assert!(
        p3.canonical_form(Some(&[1, 0, 0])).unwrap()
            == p3.canonical_form(Some(&[0, 0, 1])).unwrap()
    );
    assert_eq!(relabeled(&[1, 0, 0]), relabeled(&[0, 0, 1]));
}

/// Regression test for igraph's property-cache bug: the triangle functions
/// build an `IGRAPH_ALL` / `IGRAPH_NO_MULTIPLE` adjacency list, which in
/// igraph 1.0.1 trusts a cached "no multi-edges" flag (so mutual pairs of a
/// directed graph were counted twice) and otherwise caches a wrong "has
/// multi-edges" flag.
#[test]
fn triangles_of_directed_mutual_pairs_keep_the_cache_correct() {
    // 0 <-> 1, 1 <-> 2, 0 <-> 2, 2 -> 3: no multi-edges in the directed sense.
    let edges = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0), (2, 3)];
    // Every call runs on a fresh graph, either with an empty property cache
    // or with a (correct) "no multi-edges" flag cached by `has_multiple`,
    // and must leave the cache telling the truth, also for clones.
    let run = |primed: bool, f: &dyn Fn(&Graph) -> Vec<f64>| {
        let g = Graph::from_edges(&edges, 4, true).unwrap();
        if primed {
            assert!(!g.has_multiple().unwrap());
        }
        let res = f(&g);
        assert!(!g.clone().has_multiple().unwrap());
        assert!(!g.has_multiple().unwrap());
        assert!(g.is_simple(true).unwrap());
        res
    };
    for primed in [false, true] {
        assert_eq!(run(primed, &|g| vec![g.count_triangles().unwrap()]), [1.0]);
        assert_eq!(
            run(primed, &|g| g.count_adjacent_triangles(..).unwrap()),
            [1.0, 1.0, 1.0, 0.0]
        );
        assert_eq!(
            run(primed, &|g| g.count_adjacent_triangles(0).unwrap()),
            [1.0]
        );
        run(primed, &|g| {
            let mut t = g.list_triangles().unwrap();
            assert_eq!(t.len(), 1);
            t[0].sort();
            assert_eq!(t[0], [0, 1, 2]);
            Vec::new()
        });
    }
    // A later directed-mode computation must not trip over a stale cache.
    let g = Graph::from_edges(&edges, 4, true).unwrap();
    g.count_triangles().unwrap();
    let layout = g
        .layout_reingold_tilford(NeighborMode::Out, None, None)
        .unwrap();
    assert_eq!(layout.nrow(), 4);
}

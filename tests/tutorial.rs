//! Tests for the translations of the igraph C tutorial (`igraph::tutorial`).
//!
//! Expected values are those printed by the original C programs
//! `examples/tutorial/tutorial{1,2,3}.c`, compiled against igraph 1.0.1
//! (printed with `%.17g` for the exact comparisons).

mod common;

use common::{assert_close, karate};
use igraph::prelude::*;
use igraph::tutorial::{
    self, KarateCentralities, LatticePathLengths, Maximum, RandomGraphStats, ZACHARY_KARATE_EDGES,
};

// ---------------------------------------------------------------- lesson 1

#[test]
fn lesson_1_matches_the_c_program() {
    let stats = tutorial::example_1().unwrap();
    assert_eq!(
        stats,
        RandomGraphStats {
            diameter: 23.0,
            mean_degree: 2.0
        }
    );
    assert_eq!(
        stats.to_string(),
        "Diameter of a random graph with average degree 2: 23"
    );
}

#[test]
fn lesson_1_is_reproducible_and_thread_independent() {
    let here = tutorial::example_1().unwrap();
    // Reseeding makes a second run identical, even after other draws.
    rng::seed(7).unwrap();
    let _ = rng::integer(0, 1000);
    assert_eq!(tutorial::example_1().unwrap(), here);
    // Each thread has its own generator, seeded by the lesson itself.
    let there = std::thread::spawn(|| tutorial::example_1().unwrap())
        .join()
        .unwrap();
    assert_eq!(there, here);
}

#[test]
fn lesson_1_random_graph_story() {
    // Rebuild the graph of lesson 1 and look at it more closely: with
    // average degree 2 a G(n, m) graph is past the giant component
    // threshold (average degree 1) but far from connected.
    rng::seed(42).unwrap();
    let g = Graph::erdos_renyi_game_gnm(1000, 1000, false, EdgeTypeSw::Simple, false).unwrap();
    assert_eq!((g.vcount(), g.ecount()), (1000, 1000));
    assert!(g.is_simple(true).unwrap());
    assert!(!g.is_directed());

    // Handshake lemma: the mean degree is 2m / n.
    let degrees = g.degree(.., NeighborMode::All, Loops::Twice).unwrap();
    assert_eq!(degrees.iter().sum::<i64>(), 2000);

    let comps = g.connected_components(Connectedness::Weak).unwrap();
    assert!(!g.is_connected(Connectedness::Weak).unwrap());
    assert_eq!(comps.count, 150); // as reported by igraph 1.0.1 in C
    let giant = *comps.sizes.iter().max().unwrap();
    assert!(giant > 500, "giant component has {giant} vertices");

    // The diameter (over all components) and the longest geodesic found by C.
    let d = g.diameter_with_path(None, false, true).unwrap();
    assert_eq!(d.length, 23.0);
    assert_eq!((d.from, d.to), (Some(568), Some(881)));
    assert_eq!(d.path.vertices.len(), 24);
    assert_eq!(d.path.edges.len(), 23);
    assert_eq!(g.diameter().unwrap(), 23.0);
    // With `unconn = false` a disconnected graph has infinite diameter.
    assert!(
        g.diameter_with_path(None, false, false)
            .unwrap()
            .length
            .is_infinite()
    );
}

// ---------------------------------------------------------------- lesson 2

#[test]
fn lesson_2_matches_the_c_program() {
    let res = tutorial::example_2().unwrap();
    // A 30 x 30 torus: every vertex sees distances 0..=15 in each
    // coordinate, averaging 7.5 each, so the mean over the 899 other
    // vertices is 15 * 900 / 899.
    assert_close(res.lattice, 15.0 * 900.0 / 899.0, 1e-12);
    assert_close(res.lattice, 15.016685205784205, 1e-12);
    assert_close(res.randomized, 11.814163885799037, 1e-12);
    assert_eq!(
        res.to_string(),
        "Average path length (lattice):            15.0167\n\
         Average path length (randomized lattice): 11.8142"
    );
}

#[test]
fn lesson_2_draws_the_same_edges_as_c() {
    let res = tutorial::example_2().unwrap();
    // The 20 numbers drawn by `RNG_INTEGER(0, 899)` after seeding with 42.
    let expected = [
        (123, 489),
        (756, 885),
        (885, 665),
        (830, 21),
        (108, 521),
        (806, 851),
        (870, 575),
        (273, 728),
        (131, 562),
        (150, 582),
    ];
    assert_eq!(res.random_edges, expected);
    assert!(
        res.random_edges
            .iter()
            .all(|&(a, b)| (0..900).contains(&a) && (0..900).contains(&b))
    );
}

#[test]
fn lesson_2_step_by_step() {
    let mut g = Graph::square_lattice(&[30, 30], 0, false, false, Some(&[true, true])).unwrap();
    assert_eq!((g.vcount(), g.ecount()), (900, 1800));
    let deg = g.degree(.., NeighborMode::All, Loops::Twice).unwrap();
    assert!(deg.iter().all(|&d| d == 4));
    // Periodic in both dimensions: the diameter is 15 + 15.
    assert_eq!(g.diameter().unwrap(), 30.0);

    let LatticePathLengths {
        lattice,
        randomized,
        random_edges,
    } = tutorial::example_2().unwrap();
    assert_eq!(g.average_path_length(None, false, true).unwrap(), lattice);

    // Adding the lesson's random edges reproduces the randomized value.
    g.add_edges(&random_edges).unwrap();
    assert_eq!(g.ecount(), 1810);
    assert_eq!(
        g.average_path_length(None, false, true).unwrap(),
        randomized
    );
    // These particular edges create neither loops nor multi-edges, so
    // simplifying changes nothing.
    assert!(g.is_simple(false).unwrap());
    g.simplify(true, true).unwrap();
    assert_eq!(g.ecount(), 1810);
    // Shortcuts can only shorten paths, and shrink the diameter too.
    assert!(randomized < lattice);
    assert!(g.diameter().unwrap() < 30.0);
}

#[test]
fn lesson_2_without_periodic_boundaries_paths_are_longer() {
    // A variation on the lesson: on the open n x n grid each coordinate
    // contributes (n^2 - 1) / (3n) on average over all ordered pairs
    // (self-pairs included); excluding the n^2 self-pairs gives exactly
    // 2 (n^2 - 1) / (3n) * n^2 / (n^2 - 1) = 2n / 3 = 20.
    let grid = Graph::square_lattice(&[30, 30], 0, false, false, None).unwrap();
    let apl = grid.average_path_length(None, false, true).unwrap();
    assert_close(apl, 20.0, 1e-12);
    assert!(apl > tutorial::example_2().unwrap().lattice);
}

// ---------------------------------------------------------------- lesson 3

#[test]
fn lesson_3_matches_the_c_program() {
    let res = tutorial::example_3().unwrap();
    assert_eq!(
        res.degree,
        Maximum {
            value: 17,
            vertex: 33
        }
    );
    assert_eq!(res.closeness.vertex, 0);
    assert_close(res.closeness.value, 1.0 / 58.0, 1e-15);
    assert_eq!(res.betweenness.vertex, 0);
    assert_close(res.betweenness.value, 231.0714285714286, 1e-9);
    assert_eq!(
        res.to_string(),
        "Maximum degree is              17, vertex 33.\n\
         Maximum closeness is    0.0172414, vertex  0.\n\
         Maximum betweenness is    231.071, vertex  0."
    );
}

#[test]
fn lesson_3_edge_array_is_the_karate_club() {
    let g = Graph::from_flat_edges(&ZACHARY_KARATE_EDGES, 0, false).unwrap();
    assert_eq!((g.vcount(), g.ecount()), (34, 78));
    assert!(g.is_simple(true).unwrap());
    assert!(g.is_connected(Connectedness::Weak).unwrap());

    // Same edge set as the test fixture, up to the order of the edges.
    let normalize = |mut e: Vec<(i64, i64)>| {
        for p in &mut e {
            if p.0 > p.1 {
                *p = (p.1, p.0);
            }
        }
        e.sort_unstable();
        e
    };
    assert_eq!(
        normalize(g.edge_list()),
        normalize(karate().edge_list()),
        "tutorial array and common::KARATE_EDGES differ"
    );
}

#[test]
fn lesson_3_centralities_satisfy_identities() {
    let g = karate();
    let n = g.vcount() as f64;

    // Degrees: the two leaders have the largest degrees.
    let degree = g.degree(.., NeighborMode::All, Loops::Twice).unwrap();
    assert_eq!(degree.iter().sum::<i64>(), 2 * 78);
    assert_eq!((degree[0], degree[33]), (16, 17));

    // Closeness is the inverse of the sum of distances.
    let dist = g
        .distances(.., .., None, NeighborMode::All)
        .unwrap()
        .to_rows();
    let closeness = g
        .closeness(VertexSelector::All, NeighborMode::All, None, false)
        .unwrap();
    for (v, row) in dist.iter().enumerate() {
        assert_close(closeness[v], 1.0 / row.iter().sum::<f64>(), 1e-15);
    }
    assert_eq!(dist[0].iter().sum::<f64>(), 58.0);

    // Betweenness identity for connected undirected graphs: every unordered
    // pair (u, v) contributes d(u, v) - 1 in total to the inner vertices of
    // its shortest paths, so sum(betweenness) = pairs * (mean distance - 1).
    let betweenness = g
        .betweenness(None, VertexSelector::All, false, false)
        .unwrap();
    let apl = g.average_path_length(None, false, true).unwrap();
    let pairs = n * (n - 1.0) / 2.0;
    assert_close(betweenness.iter().sum::<f64>(), pairs * (apl - 1.0), 1e-9);

    // The maxima are those of lesson 3.
    let KarateCentralities {
        degree: dmax,
        closeness: cmax,
        betweenness: bmax,
    } = tutorial::example_3().unwrap();
    assert_eq!(Maximum::of(&degree), Some(dmax));
    assert_eq!(Maximum::of(&closeness), Some(cmax));
    assert_eq!(Maximum::of(&betweenness), Some(bmax));
}

#[test]
fn lesson_3_story_the_two_leaders() {
    // Rank the members by betweenness: the instructor (0) and the
    // administrator (33) are the two top brokers of the club.
    let g = karate();
    let b = g
        .betweenness(None, VertexSelector::All, false, false)
        .unwrap();
    let mut order: Vec<usize> = (0..b.len()).collect();
    order.sort_by(|&x, &y| b[y].total_cmp(&b[x]));
    assert_eq!(&order[..2], &[0, 33]);
    // igraph 1.0 normalizes by the number of unordered vertex pairs n(n-1)/2.
    let nb = g
        .betweenness(None, VertexSelector::All, false, true)
        .unwrap();
    assert_close(nb[0], b[0] / (34.0 * 33.0 / 2.0), 1e-12);
}

// ------------------------------------------------------------- Maximum::of

#[test]
fn maximum_of_mimics_igraph_which_max() {
    assert_eq!(Maximum::<i64>::of(&[]), None);
    assert_eq!(
        Maximum::of(&[5]),
        Some(Maximum {
            value: 5,
            vertex: 0
        })
    );
    // The first maximum wins ties.
    assert_eq!(
        Maximum::of(&[1, 9, 3, 9]),
        Some(Maximum {
            value: 9,
            vertex: 1
        })
    );
    assert_eq!(Maximum::of(&[-3.0, -1.5, -2.0]).unwrap().vertex, 1);
    // As igraph_vector_max()/which_max(): the first NaN wins, wherever it is.
    let m = Maximum::of(&[1.0, f64::NAN, 2.0]).unwrap();
    assert!(m.value.is_nan());
    assert_eq!(m.vertex, 1);
    let m = Maximum::of(&[1.0, 3.0, 2.0, f64::NAN, f64::NAN]).unwrap();
    assert!(m.value.is_nan());
    assert_eq!(m.vertex, 3);
    let m = Maximum::of(&[f64::NAN, 1.0]).unwrap();
    assert!(m.value.is_nan());
    assert_eq!(m.vertex, 0);
    // Infinities are ordinary values.
    assert_eq!(
        Maximum::of(&[f64::NEG_INFINITY, f64::INFINITY, 0.0]),
        Some(Maximum {
            value: f64::INFINITY,
            vertex: 1
        })
    );
}

#[test]
fn display_formats_like_printf() {
    // `%g` switches to scientific notation and `%10` right-aligns.
    let res = KarateCentralities {
        degree: Maximum {
            value: 1234567,
            vertex: 5,
        },
        closeness: Maximum {
            value: 1.0e-5,
            vertex: 12,
        },
        betweenness: Maximum {
            value: 1234567.0,
            vertex: 0,
        },
    };
    assert_eq!(
        res.to_string(),
        "Maximum degree is         1234567, vertex  5.\n\
         Maximum closeness is        1e-05, vertex 12.\n\
         Maximum betweenness is 1.23457e+06, vertex  0."
    );
    let stats = RandomGraphStats {
        diameter: f64::NAN,
        mean_degree: 2.5,
    };
    assert_eq!(
        stats.to_string(),
        "Diameter of a random graph with average degree 2.5: nan"
    );
}

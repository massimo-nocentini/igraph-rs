//! Integration tests for the `layout` module (`igraph_layout.h`).

mod common;

use common::{assert_close, complete, cycle, path};
use igraph::layout::*;
use igraph::misc::Metric;
use igraph::prelude::*;

const EPS: f64 = 1e-5;

fn dist(l: &Matrix, i: usize, j: usize) -> f64 {
    (0..l.ncol())
        .map(|k| (l[(i, k)] - l[(j, k)]).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn assert_matrix_close(m: &Matrix, expected: &[[f64; 2]], eps: f64) {
    assert_eq!(m.shape(), (expected.len(), 2), "shape mismatch: {m}");
    for (i, row) in expected.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            assert!(
                (m[(i, j)] - x).abs() <= eps,
                "entry ({i}, {j}) = {} != {x}\n{m}",
                m[(i, j)]
            );
        }
    }
}

/// Undirected k-ary tree 0 -> {1, .., k}, 1 -> {k + 1, .., 2k}, ...
fn kary_tree(n: usize, k: usize) -> Graph {
    Graph::kary_tree(n, k, TreeMode::Undirected).unwrap()
}

/// Zachary's karate club, from the constructors module.
fn karate() -> Graph {
    Graph::famous("Zachary").unwrap()
}

/// Seeds the calling thread's default generator, then runs `f`. Each test
/// thread has its own default generator, so this is reproducible even while
/// other tests run in parallel.
fn seeded<R>(seed: u64, f: impl FnOnce() -> R) -> R {
    rng::seed(seed).unwrap();
    f()
}

/// Mean Euclidean edge length of a drawing.
fn mean_edge_length(g: &Graph, l: &Matrix) -> f64 {
    let lengths = g.spatial_edge_lengths(l, Metric::Euclidean).unwrap();
    lengths.iter().sum::<f64>() / lengths.len() as f64
}

fn all_finite(m: &Matrix) -> bool {
    m.as_slice().iter().all(|x| x.is_finite())
}

// ---------------------------------------------------------------------------
// Geometric layouts
// ---------------------------------------------------------------------------

#[test]
fn random_layouts_are_in_the_unit_box_and_reproducible() {
    let g = Graph::new(50, false);
    let a = seeded(42, || g.layout_random().unwrap());
    let b = seeded(42, || g.layout_random().unwrap());
    assert_eq!(a, b);
    assert_eq!(a.shape(), (50, 2));
    assert!(a.as_slice().iter().all(|x| (-1.0..=1.0).contains(x)));

    let c = g.layout_random_3d().unwrap();
    assert_eq!(c.shape(), (50, 3));
    assert!(c.as_slice().iter().all(|x| (-1.0..=1.0).contains(x)));
}

#[test]
fn circle_layout_has_radius_one_and_equal_angles() {
    let g = cycle(12);
    let l = g.layout_circle(..).unwrap();
    assert_eq!(l.shape(), (12, 2));
    for i in 0..12 {
        assert_close(l[(i, 0)].hypot(l[(i, 1)]), 1.0, 1e-12);
        let phi = 2.0 * std::f64::consts::PI * i as f64 / 12.0;
        assert_close(l[(i, 0)], phi.cos(), 1e-12);
        assert_close(l[(i, 1)], phi.sin(), 1e-12);
    }
    // Consecutive vertices of a regular 12-gon are 2 sin(pi/12) apart.
    let side = 2.0 * (std::f64::consts::PI / 12.0).sin();
    for i in 0..12 {
        assert_close(dist(&l, i, (i + 1) % 12), side, 1e-12);
    }
}

#[test]
fn circle_layout_with_partial_order() {
    let g = Graph::new(5, false);
    let l = g.layout_circle(vec![4, 2]).unwrap();
    assert_matrix_close(
        &l,
        &[[0.0, 0.0], [0.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
        1e-12,
    );
    let err = g.layout_circle(vec![0, 9]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
}

#[test]
fn star_layout_matches_igraph_unit_test() {
    // tests/unit/igraph_layout_star.out
    let g = Graph::new(9, false);
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let l = g.layout_star(0, None).unwrap();
    assert_matrix_close(
        &l,
        &[
            [0.0, 0.0],
            [1.0, 0.0],
            [s, s],
            [0.0, 1.0],
            [-s, s],
            [-1.0, 0.0],
            [-s, -s],
            [0.0, -1.0],
            [s, -s],
        ],
        1e-12,
    );
    let rev: Vec<i64> = (0..9).rev().collect();
    let l = g.layout_star(0, Some(&rev)).unwrap();
    assert_matrix_close(
        &l,
        &[
            [0.0, 0.0],
            [s, -s],
            [0.0, -1.0],
            [-s, -s],
            [-1.0, 0.0],
            [-s, s],
            [0.0, 1.0],
            [s, s],
            [1.0, 0.0],
        ],
        1e-12,
    );
    // Null graph: any center is accepted.
    assert_eq!(
        Graph::new(0, false).layout_star(0, None).unwrap().shape(),
        (0, 2)
    );
    // Singleton: at the origin.
    assert_eq!(
        Graph::new(1, false).layout_star(0, None).unwrap().row(0),
        vec![0.0, 0.0]
    );
}

#[test]
fn star_layout_center_is_at_origin_and_others_on_unit_circle() {
    let g = karate();
    let l = g.layout_star(33, None).unwrap();
    assert_eq!(l.row(33), vec![0.0, 0.0]);
    for i in 0..33 {
        assert_close(l[(i, 0)].hypot(l[(i, 1)]), 1.0, 1e-12);
    }
}

#[test]
fn star_layout_errors() {
    let g = Graph::new(9, false);
    assert_eq!(
        g.layout_star(-10, None).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.layout_star(9, None).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    let bad = [-1, -1, -1, 10, 10, 10, 2, 1, 0];
    assert_eq!(
        g.layout_star(0, Some(&bad)).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.layout_star(0, Some(&[0, 1])).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    // A repeated vertex (igraph would leave the missing vertex's row
    // uninitialized) is rejected too.
    let repeated = [0, 1, 1, 3, 4, 5, 6, 7, 8];
    assert_eq!(
        g.layout_star(0, Some(&repeated)).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn grid_layout_coordinates() {
    let g = Graph::new(10, false);
    // Automatic width: ceil(sqrt(10)) = 4.
    let l = g.layout_grid(None).unwrap();
    for i in 0..10 {
        assert_eq!(l.row(i), vec![(i % 4) as f64, (i / 4) as f64]);
    }
    let l = g.layout_grid(Some(3)).unwrap();
    for i in 0..10 {
        assert_eq!(l.row(i), vec![(i % 3) as f64, (i / 3) as f64]);
    }
}

#[test]
fn grid_3d_layout_coordinates() {
    let g = Graph::new(27, false);
    // Automatic: 3 x 3 x 3 cube.
    let l = g.layout_grid_3d(None, None).unwrap();
    assert_eq!(l.shape(), (27, 3));
    for i in 0..27 {
        assert_eq!(
            l.row(i),
            vec![(i % 3) as f64, (i / 3 % 3) as f64, (i / 9) as f64]
        );
    }
    let l = Graph::new(12, false)
        .layout_grid_3d(Some(2), Some(3))
        .unwrap();
    for i in 0..12 {
        assert_eq!(
            l.row(i),
            vec![(i % 2) as f64, (i / 2 % 3) as f64, (i / 6) as f64]
        );
    }
}

#[test]
fn sphere_layout_is_on_the_unit_sphere_and_spreads_out() {
    let g = Graph::new(100, false);
    let l = g.layout_sphere().unwrap();
    assert_eq!(l.shape(), (100, 3));
    for p in l.rows() {
        assert_close((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt(), 1.0, 1e-9);
    }
    // The points span the whole sphere (from pole to pole along z)...
    let z = l.column(2);
    assert_close(z.iter().cloned().fold(f64::INFINITY, f64::min), -1.0, 1e-9);
    assert_close(
        z.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        1.0,
        1e-9,
    );
    // ... and the center of mass is close to the origin.
    for k in 0..3 {
        let mean: f64 = l.column(k).iter().sum::<f64>() / 100.0;
        assert!(mean.abs() < 0.05, "mean of column {k} = {mean}");
    }
}

// ---------------------------------------------------------------------------
// Force-directed layouts
// ---------------------------------------------------------------------------

#[test]
fn fruchterman_reingold_is_reproducible_with_seed() {
    let g = karate();
    let opts = FruchtermanReingoldOptions::default();
    let a = seeded(123, || g.layout_fruchterman_reingold(&opts).unwrap());
    let b = seeded(123, || g.layout_fruchterman_reingold(&opts).unwrap());
    assert_eq!(a, b);
    assert_eq!(a.shape(), (34, 2));
    assert!(all_finite(&a));

    // The grid variant runs as well.
    let grid = FruchtermanReingoldOptions {
        grid: LayoutGrid::Grid,
        ..Default::default()
    };
    let c = g.layout_fruchterman_reingold(&grid).unwrap();
    assert_eq!(c.shape(), (34, 2));
    assert!(all_finite(&c));
}

#[test]
fn fruchterman_reingold_places_neighbors_closer() {
    let g = karate();
    let l = seeded(1, || {
        g.layout_fruchterman_reingold(&FruchtermanReingoldOptions::default())
            .unwrap()
    });
    let mean_edge = mean_edge_length(&g, &l);
    let mut sum = 0.0;
    let mut count = 0;
    for i in 0..34 {
        for j in i + 1..34 {
            sum += dist(&l, i, j);
            count += 1;
        }
    }
    let mean_all = sum / count as f64;
    assert!(
        mean_edge < mean_all,
        "edges {mean_edge} vs all pairs {mean_all}"
    );
}

#[test]
fn fruchterman_reingold_heavy_edges_are_shorter() {
    // A 4-cycle where edge 0-1 is much heavier than the others.
    let g = cycle(4);
    let weights = [10.0, 1.0, 1.0, 1.0];
    let l = seeded(5, || {
        g.layout_fruchterman_reingold(&FruchtermanReingoldOptions::default().with_weights(&weights))
            .unwrap()
    });
    assert!(dist(&l, 0, 1) < dist(&l, 1, 2));
    assert!(dist(&l, 0, 1) < dist(&l, 2, 3));
}

#[test]
fn fruchterman_reingold_respects_bounds_and_seeds() {
    let g = karate();
    let lo = vec![-1.0; 34];
    let hi = vec![1.0; 34];
    let opts = FruchtermanReingoldOptions {
        minx: Some(&lo),
        maxx: Some(&hi),
        miny: Some(&lo),
        maxy: Some(&hi),
        ..Default::default()
    };
    let l = seeded(9, || g.layout_fruchterman_reingold(&opts).unwrap());
    assert!(l.as_slice().iter().all(|x| (-1.0..=1.0).contains(x)));

    // Zero iterations from a given start leave it untouched.
    let start = g.layout_circle(..).unwrap();
    let opts = FruchtermanReingoldOptions::default()
        .with_initial(&start)
        .with_niter(0);
    assert_eq!(g.layout_fruchterman_reingold(&opts).unwrap(), start);

    // 3D, bounded along z.
    let z0 = vec![0.0; 34];
    let opts = FruchtermanReingoldOptions {
        minz: Some(&z0),
        maxz: Some(&z0),
        ..Default::default()
    };
    let l3 = g.layout_fruchterman_reingold_3d(&opts).unwrap();
    assert_eq!(l3.shape(), (34, 3));
    assert!(l3.column(2).iter().all(|&z| z == 0.0));
}

#[test]
fn fruchterman_reingold_errors() {
    let g = cycle(4);
    let bad_weights = [1.0, -1.0, 1.0, 1.0];
    let opts = FruchtermanReingoldOptions::default().with_weights(&bad_weights);
    assert_eq!(
        g.layout_fruchterman_reingold(&opts).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    let short = [1.0];
    let opts = FruchtermanReingoldOptions::default().with_weights(&short);
    assert_eq!(
        g.layout_fruchterman_reingold(&opts).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    let (lo, hi) = ([1.0; 4], [0.0; 4]);
    let opts = FruchtermanReingoldOptions {
        minx: Some(&lo),
        maxx: Some(&hi),
        ..Default::default()
    };
    assert_eq!(
        g.layout_fruchterman_reingold(&opts).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    let wrong = Matrix::zeros(4, 3);
    let opts = FruchtermanReingoldOptions::default().with_initial(&wrong);
    assert_eq!(
        g.layout_fruchterman_reingold(&opts).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn kamada_kawai_is_deterministic_and_draws_a_cycle_as_a_circle() {
    let g = cycle(10);
    let opts = KamadaKawaiOptions::default();
    let a = g.layout_kamada_kawai(&opts).unwrap();
    let b = g.layout_kamada_kawai(&opts).unwrap();
    assert_eq!(a, b, "KK starts from a circle: no randomness involved");
    // All vertices at the same distance from the barycenter.
    let cx = a.column(0).iter().sum::<f64>() / 10.0;
    let cy = a.column(1).iter().sum::<f64>() / 10.0;
    let radii: Vec<f64> = (0..10)
        .map(|i| (a[(i, 0)] - cx).hypot(a[(i, 1)] - cy))
        .collect();
    for r in &radii {
        assert_close(*r, radii[0], 1e-3);
    }
    // Equal side lengths.
    for i in 0..10 {
        assert_close(dist(&a, i, (i + 1) % 10), dist(&a, 0, 1), 1e-3);
    }
}

#[test]
fn kamada_kawai_distances_follow_graph_distances() {
    // A path can reach zero energy: it is drawn straight, every pair at the
    // rest length sqrt(n) * d / diameter of its spring.
    let g = path(6);
    let l = g
        .layout_kamada_kawai(&KamadaKawaiOptions::default())
        .unwrap();
    let unit = 6f64.sqrt() / 5.0;
    for i in 0..6 {
        for j in i + 1..6 {
            assert_close(dist(&l, i, j), unit * (j - i) as f64, 1e-3);
        }
    }
    // Longer edges (weights are lengths in KK).
    let g = cycle(4);
    let w = [3.0, 1.0, 1.0, 1.0];
    let l = g
        .layout_kamada_kawai(&KamadaKawaiOptions::default().with_weights(&w))
        .unwrap();
    assert!(dist(&l, 0, 1) > dist(&l, 2, 3));
}

#[test]
fn kamada_kawai_3d_and_errors() {
    let g = complete(4);
    let l = g
        .layout_kamada_kawai_3d(&KamadaKawaiOptions::default())
        .unwrap();
    assert_eq!(l.shape(), (4, 3));
    // K4 in 3D: a regular tetrahedron, all pairwise distances equal.
    let d01 = dist(&l, 0, 1);
    for i in 0..4 {
        for j in i + 1..4 {
            assert_close(dist(&l, i, j), d01, 1e-3);
        }
    }
    let opts = KamadaKawaiOptions {
        kkconst: Some(0.0),
        ..Default::default()
    };
    assert_eq!(
        g.layout_kamada_kawai(&opts).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    let start = Matrix::zeros(4, 2);
    let opts = KamadaKawaiOptions::default().with_initial(&start);
    assert_eq!(
        g.layout_kamada_kawai_3d(&opts).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn lgl_layout_on_connected_graph() {
    let g = karate();
    let opts = LglOptions {
        root: Some(0),
        ..Default::default()
    };
    let a = seeded(2, || g.layout_lgl(&opts).unwrap());
    let b = seeded(2, || g.layout_lgl(&opts).unwrap());
    assert_eq!(a, b);
    assert_eq!(a.shape(), (34, 2));
    assert!(all_finite(&a));
    // The null graph gives an empty 0 x 2 layout.
    assert_eq!(
        Graph::new(0, false)
            .layout_lgl(&LglOptions::default())
            .unwrap()
            .shape(),
        (0, 2)
    );
    let bad = LglOptions {
        coolexp: 0.0,
        ..Default::default()
    };
    assert_eq!(
        g.layout_lgl(&bad).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    let bad = LglOptions {
        root: Some(34),
        ..Default::default()
    };
    assert_eq!(
        g.layout_lgl(&bad).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
}

#[test]
fn graphopt_gem_davidson_harel_basic_properties() {
    let g = karate();
    let a = seeded(11, || {
        g.layout_graphopt(&GraphoptOptions::default()).unwrap()
    });
    assert_eq!(a.shape(), (34, 2));
    assert!(all_finite(&a));

    // Refining: zero further iterations keep the given layout.
    let opts = GraphoptOptions {
        niter: 0,
        initial: Some(&a),
        ..Default::default()
    };
    assert_eq!(g.layout_graphopt(&opts).unwrap(), a);

    let small = cycle(8);
    let gem = seeded(11, || small.layout_gem(&GemOptions::default()).unwrap());
    assert_eq!(gem.shape(), (8, 2));
    assert!(all_finite(&gem));
    assert_eq!(
        Graph::new(0, false)
            .layout_gem(&GemOptions::default())
            .unwrap()
            .shape(),
        (0, 2)
    );
    let bad = GemOptions {
        temp_min: 100.0,
        ..Default::default()
    };
    assert_eq!(
        small.layout_gem(&bad).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );

    let dh = seeded(11, || {
        small
            .layout_davidson_harel(&DavidsonHarelOptions::default())
            .unwrap()
    });
    assert_eq!(dh.shape(), (8, 2));
    assert!(all_finite(&dh));
    let bad = DavidsonHarelOptions {
        cool_fact: 1.5,
        ..Default::default()
    };
    assert_eq!(
        small.layout_davidson_harel(&bad).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn drl_options_templates() {
    let d = DrlOptions::default();
    assert_eq!(d.edge_cut, 0.8);
    assert_eq!(
        (
            d.liquid_iterations,
            d.expansion_iterations,
            d.cooldown_iterations
        ),
        (200, 200, 200)
    );
    assert_eq!((d.crunch_iterations, d.simmer_iterations), (50, 100));
    let coarsest = DrlOptions::from_template(DrlTemplate::Coarsest);
    assert_eq!(coarsest.crunch_iterations, 200);
    let refine = DrlOptions::from_template(DrlTemplate::Refine);
    assert_eq!(refine.liquid_iterations, 0);
    let fin = DrlOptions::from_template(DrlTemplate::Final);
    assert_eq!(fin.simmer_iterations, 25);
    let copy = fin;
    assert_eq!(copy.simmer_iterations, 25);
    assert_eq!(DrlTemplate::try_from(1).unwrap(), DrlTemplate::Coarsen);
}

#[test]
fn drl_layouts() {
    let g = karate();
    let a = seeded(4, || {
        g.layout_drl(&DrlOptions::default(), None, None).unwrap()
    });
    let b = seeded(4, || {
        g.layout_drl(&DrlOptions::default(), None, None).unwrap()
    });
    assert_eq!(a, b);
    assert_eq!(a.shape(), (34, 2));
    assert!(all_finite(&a));
    // Refine the result.
    let refined = g
        .layout_drl(
            &DrlOptions::from_template(DrlTemplate::Refine),
            None,
            Some(&a),
        )
        .unwrap();
    assert_eq!(refined.shape(), (34, 2));

    let w = vec![1.0; g.ecount()];
    let c = g
        .layout_drl_3d(&DrlOptions::default(), Some(&w), None)
        .unwrap();
    assert_eq!(c.shape(), (34, 3));
    assert!(all_finite(&c));

    let bad = DrlOptions {
        crunch_damping_mult: -1.0,
        ..Default::default()
    };
    assert_eq!(
        g.layout_drl(&bad, None, None).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    let mut zero_w = w.clone();
    zero_w[0] = 0.0;
    assert_eq!(
        g.layout_drl(&DrlOptions::default(), Some(&zero_w), None)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn umap_layouts_and_weights() {
    // Two 5-cliques joined by one edge.
    let mut edges = vec![];
    for base in [0, 5] {
        for i in 0..5 {
            for j in i + 1..5 {
                edges.push((base + i, base + j));
            }
        }
    }
    edges.push((0, 5));
    let g = Graph::from_edges(&edges, 10, false).unwrap();
    let distances: Vec<f64> = edges
        .iter()
        .map(|&(u, v)| if (u < 5) == (v < 5) { 0.1 } else { 5.0 })
        .collect();

    let w = g.layout_umap_compute_weights(Some(&distances)).unwrap();
    assert_eq!(w.len(), g.ecount());
    assert!(w.iter().all(|&x| (0.0..=1.0).contains(&x)));
    // The bridge is the weakest link.
    let bridge = *w.last().unwrap();
    assert!(w[..w.len() - 1].iter().all(|&x| x > bridge));

    let opts = UmapOptions {
        distances: Some(&distances),
        epochs: 100,
        ..Default::default()
    };
    let l = seeded(8, || g.layout_umap(&opts).unwrap());
    assert_eq!(l.shape(), (10, 2));
    assert!(all_finite(&l));
    assert_eq!(seeded(8, || g.layout_umap(&opts).unwrap()), l);

    let opts = UmapOptions {
        distances: Some(&w),
        distances_are_weights: true,
        epochs: 50,
        ..Default::default()
    };
    let l3 = g.layout_umap_3d(&opts).unwrap();
    assert_eq!(l3.shape(), (10, 3));
    assert!(all_finite(&l3));

    let opts = UmapOptions {
        min_dist: -1.0,
        ..Default::default()
    };
    assert_eq!(
        g.layout_umap(&opts).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

// ---------------------------------------------------------------------------
// Trees
// ---------------------------------------------------------------------------

#[test]
fn reingold_tilford_binary_tree() {
    let g = kary_tree(15, 2);
    let l = g
        .layout_reingold_tilford(NeighborMode::All, Some(&[0]), None)
        .unwrap();
    // y is the depth.
    for v in 0..15usize {
        let depth = ((v + 1) as f64).log2().floor();
        assert_eq!(l[(v, 1)], depth);
    }
    // Parents are centered above their children, and children are ordered.
    for p in 0..7usize {
        let (c1, c2) = (2 * p + 1, 2 * p + 2);
        assert_close(l[(p, 0)], (l[(c1, 0)] + l[(c2, 0)]) / 2.0, 1e-12);
        assert!(l[(c1, 0)] < l[(c2, 0)]);
    }
    // Leaves are at distinct positions, at least one unit apart.
    let mut xs: Vec<f64> = (7..15).map(|v| l[(v, 0)]).collect();
    xs.sort_by(f64::total_cmp);
    for w in xs.windows(2) {
        assert!(w[1] - w[0] >= 1.0 - 1e-12);
    }
}

#[test]
fn reingold_tilford_directed_and_automatic_roots() {
    // An in-tree: edges point to the root 0.
    let edges: Vec<(i64, i64)> = (1..7).map(|v| (v, (v - 1) / 2)).collect();
    let g = Graph::from_edges(&edges, 7, true).unwrap();
    let l = g
        .layout_reingold_tilford(NeighborMode::In, None, None)
        .unwrap();
    assert_eq!(l.column(1), &[0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);

    let roots = g
        .roots_for_tree_layout(NeighborMode::In, RootChoice::Degree)
        .unwrap();
    assert_eq!(roots, vec![0]);
    let roots = kary_tree(7, 2)
        .roots_for_tree_layout(NeighborMode::All, RootChoice::Eccentricity)
        .unwrap();
    assert_eq!(roots, vec![0]);
}

#[test]
fn reingold_tilford_forest_with_rootlevel() {
    // Two small trees 0 -> {1, 2} and 3 -> {4, 5}; the second one one level lower.
    let g = Graph::from_edges(&[(0, 1), (0, 2), (3, 4), (3, 5)], 6, true).unwrap();
    let roots = [0, 3];
    let l = g
        .layout_reingold_tilford(NeighborMode::Out, Some(&roots), Some(&[0, 1]))
        .unwrap();
    assert_eq!(l.shape(), (6, 2));
    assert_eq!(l[(3, 1)] - l[(0, 1)], 1.0);
    assert_eq!(l[(4, 1)], l[(3, 1)] + 1.0);
    // The caller's roots are not modified by igraph.
    assert_eq!(roots, [0, 3]);

    assert_eq!(
        g.layout_reingold_tilford(NeighborMode::Out, Some(&[0, 3]), Some(&[1]))
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.layout_reingold_tilford(NeighborMode::Out, Some(&[7]), None)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidVertexId
    );
}

#[test]
fn reingold_tilford_circular_matches_igraph_unit_test() {
    // tests/unit/igraph_layout_reingold_tilford_circular.out
    assert_eq!(
        Graph::new(0, false)
            .layout_reingold_tilford_circular(NeighborMode::All, None, None)
            .unwrap()
            .shape(),
        (0, 2)
    );
    let single = Graph::new(1, false)
        .layout_reingold_tilford_circular(NeighborMode::All, None, None)
        .unwrap();
    assert_eq!(single.row(0), vec![0.0, 0.0]);

    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (0, 4)], 5, true).unwrap();
    let l = g
        .layout_reingold_tilford_circular(NeighborMode::Out, Some(&[1]), None)
        .unwrap();
    assert_matrix_close(
        &l,
        &[
            [1.0, 0.0],
            [0.0, 0.0],
            [-0.104528, 0.994522],
            [-0.978148, -0.207912],
            [0.309017, -0.951057],
        ],
        EPS,
    );

    let g = Graph::from_edges(&[(0, 1), (0, 2), (3, 4), (3, 5)], 6, true).unwrap();
    let l = g
        .layout_reingold_tilford_circular(NeighborMode::Out, Some(&[0, 3]), None)
        .unwrap();
    assert_matrix_close(
        &l,
        &[
            [0.642788, 0.766044],
            [2.0, 0.0],
            [-0.347296, 1.96962],
            [-0.34202, -0.939693],
            [-1.87939, -0.68404],
            [1.0, -1.73205],
        ],
        EPS,
    );
    let l = g
        .layout_reingold_tilford_circular(NeighborMode::Out, Some(&[0, 3]), Some(&[10, 20]))
        .unwrap();
    assert_matrix_close(
        &l,
        &[
            [5.5, 9.52628],
            [12.0, 0.0],
            [-6.0, 10.3923],
            [-10.5, -18.1865],
            [-22.0, 0.0],
            [11.0, -19.0526],
        ],
        1e-4,
    );
}

#[test]
fn reingold_tilford_circular_radius_is_depth() {
    let g = kary_tree(13, 3);
    let l = g
        .layout_reingold_tilford_circular(NeighborMode::All, Some(&[0]), None)
        .unwrap();
    assert_eq!(l.row(0), vec![0.0, 0.0]);
    for v in 1..4 {
        assert_close(l[(v, 0)].hypot(l[(v, 1)]), 1.0, 1e-12);
    }
    for v in 4..13 {
        assert_close(l[(v, 0)].hypot(l[(v, 1)]), 2.0, 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Layered layouts
// ---------------------------------------------------------------------------

fn sugiyama_test_graph() -> Graph {
    let edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (2, 2),
        (1, 4),
        (2, 5),
        (4, 6),
        (5, 7),
        (6, 8),
        (7, 8),
        (3, 8),
        (8, 1),
        (8, 2),
    ];
    Graph::from_edges(&edges, 9, true).unwrap()
}

#[test]
fn sugiyama_matches_igraph_unit_test() {
    // tests/unit/igraph_layout_sugiyama.out
    let g = sugiyama_test_graph();
    let layers = [0, 1, 1, 2, 3, 3, 4, 4, 5];
    let s = g
        .layout_sugiyama(&SugiyamaOptions::default().with_layers(&layers))
        .unwrap();
    assert_matrix_close(
        &s.coords,
        &[
            [2.5, 0.0],
            [0.5, 1.0],
            [2.5, 1.0],
            [4.0, 2.0],
            [0.0, 3.0],
            [2.0, 3.0],
            [0.0, 4.0],
            [2.0, 4.0],
            [2.0, 5.0],
        ],
        1e-12,
    );
    assert_eq!(s.routing.len(), 14);
    let expected_routing: Vec<Vec<Vec<f64>>> = vec![
        vec![],
        vec![],
        vec![vec![4.0, 1.0]],
        vec![],
        vec![],
        vec![vec![0.0, 2.0]],
        vec![vec![2.0, 2.0]],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![vec![4.0, 3.0], vec![4.0, 4.0]],
        vec![vec![1.0, 4.0], vec![1.0, 3.0], vec![1.0, 2.0]],
        vec![vec![3.0, 4.0], vec![3.0, 3.0], vec![3.0, 2.0]],
    ];
    for (e, (m, rows)) in s.routing.iter().zip(&expected_routing).enumerate() {
        assert_eq!(m.ncol(), 2, "edge {e}");
        assert_eq!(&m.to_rows(), rows, "edge {e}");
    }

    // Automatic layering.
    let s = g.layout_sugiyama(&SugiyamaOptions::default()).unwrap();
    assert_matrix_close(
        &s.coords,
        &[
            [2.5, 0.0],
            [1.0, 1.0],
            [2.5, 2.0],
            [4.0, 1.0],
            [0.0, 2.0],
            [2.0, 3.0],
            [0.0, 3.0],
            [2.0, 4.0],
            [2.0, 5.0],
        ],
        1e-12,
    );
    assert_eq!(s.routing[1].to_rows(), vec![vec![2.5, 1.0]]);
}

#[test]
fn sugiyama_gaps_scale_the_layout() {
    let g = sugiyama_test_graph();
    let layers = [0, 1, 1, 2, 3, 3, 4, 4, 5];
    let base = g
        .layout_sugiyama(&SugiyamaOptions::default().with_layers(&layers))
        .unwrap();
    // Empty layers do not change the horizontal placement, but keep their
    // vertical room (igraph's "gaps in layers" unit test).
    let gappy = [0, 2, 2, 4, 6, 6, 12, 12, 15];
    let s = g
        .layout_sugiyama(&SugiyamaOptions::default().with_layers(&gappy))
        .unwrap();
    assert_eq!(s.coords.column(0), base.coords.column(0));
    assert_eq!(s.coords.column(1), gappy.map(|l| l as f64));
    assert_eq!(s.routing[2].to_rows(), vec![vec![4.0, 2.0]]);
    let opts = SugiyamaOptions {
        vgap: 3.0,
        ..SugiyamaOptions::default().with_layers(&layers)
    };
    let s = g.layout_sugiyama(&opts).unwrap();
    for (v, &layer) in layers.iter().enumerate() {
        assert_eq!(s.coords[(v, 1)], 3.0 * layer as f64);
    }
    assert_eq!(
        g.layout_sugiyama(&SugiyamaOptions::default().with_layers(&[0, 1]))
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn bipartite_layout_matches_igraph_unit_test() {
    // tests/unit/igraph_layout_bipartite.out
    let g = Graph::from_edges(
        &[(0, 5), (0, 7), (1, 6), (1, 7), (1, 8), (2, 5), (3, 8)],
        10,
        false,
    )
    .unwrap();
    let types: Vec<bool> = (0..10).map(|i| i >= 5).collect();
    let l = g.layout_bipartite(&types, 1.0, 1.0, 100).unwrap();
    assert_matrix_close(
        &l,
        &[
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 1.0],
            [3.0, 1.0],
            [4.0, 1.0],
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
            [5.0, 0.0],
        ],
        1e-12,
    );
    let g = cycle(4);
    let l = g
        .layout_bipartite(&[false, true, false, true], 1.0, -1.0, 100)
        .unwrap();
    assert_matrix_close(
        &l,
        &[[0.0, -1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 0.0]],
        1e-12,
    );
    assert_eq!(
        g.layout_bipartite(&[false, true, false, true], -1.0, 1.0, 100)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.layout_bipartite(&[false], 1.0, 1.0, 100)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

// ---------------------------------------------------------------------------
// MDS, merging and alignment
// ---------------------------------------------------------------------------

#[test]
fn mds_matches_igraph_unit_test() {
    // tests/unit/igraph_layout_mds.out: undirected binary tree on 10 vertices.
    assert_eq!(
        Graph::new(0, false).layout_mds(None, 2).unwrap().shape(),
        (0, 2)
    );
    let g = kary_tree(10, 2);
    let mut l = g.layout_mds(None, 2).unwrap();
    // Fix the arbitrary signs of the axes as the C test does.
    if l[(0, 0)] > 0.0 {
        for i in 0..10 {
            l[(i, 0)] *= -1.0;
        }
    }
    if l[(0, 1)] < 0.0 {
        for i in 0..10 {
            l[(i, 1)] *= -1.0;
        }
    }
    assert_matrix_close(
        &l,
        &[
            [-0.692039, 0.0247583],
            [0.399957, 0.178289],
            [-1.78403, -0.128772],
            [1.25186, -0.697105],
            [0.891182, 1.32979],
            [-2.6988, -0.242629],
            [-2.6988, -0.242629],
            [1.97413, -1.3515],
            [1.97413, -1.3515],
            [1.38241, 2.4813],
        ],
        EPS,
    );
}

#[test]
fn mds_recovers_euclidean_distances() {
    // 8 points in the plane: their distance matrix is exactly realizable in 2D.
    let pts: [(f64, f64); 8] = [
        (0.0, 0.0),
        (3.0, 1.0),
        (5.0, 7.0),
        (-2.0, 4.0),
        (1.0, -3.0),
        (6.0, 0.0),
        (-4.0, -1.0),
        (2.0, 2.0),
    ];
    let n = pts.len();
    let mut d = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            d[(i, j)] = (pts[i].0 - pts[j].0).hypot(pts[i].1 - pts[j].1);
        }
    }
    let g = complete(n as i64);
    let l = g.layout_mds(Some(&d), 2).unwrap();
    for i in 0..n {
        for j in i + 1..n {
            assert_close(dist(&l, i, j), d[(i, j)], 1e-6);
        }
    }
    // Higher dimensions work on connected graphs.
    assert_eq!(g.layout_mds(Some(&d), 3).unwrap().shape(), (n, 3));
    // ... but not on disconnected ones, and dimensions must be in 2..=n.
    let two_edges = Graph::from_edges(&[(0, 1), (2, 3)], 4, false).unwrap();
    assert_eq!(two_edges.layout_mds(None, 2).unwrap().shape(), (4, 2));
    assert!(two_edges.layout_mds(None, 3).is_err());
    assert_eq!(
        g.layout_mds(None, 1).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.layout_mds(None, n + 1).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.layout_mds(Some(&Matrix::zeros(2, 2)), 2)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn merge_dla_matches_igraph_unit_test() {
    // tests/unit/igraph_layout_merge2.out
    let big = cycle(10);
    let small = cycle(3);
    let lb = big.layout_circle(..).unwrap();
    let ls = small.layout_circle(..).unwrap();
    let merged = Rng::new(RngType::Pcg32, 42)
        .unwrap()
        .scoped(|| layout_merge_dla([&big, &small], &[lb.clone(), ls.clone()]).unwrap());
    assert_eq!(merged.shape(), (13, 2));
    // The largest layout is centered at the origin and scaled uniformly.
    assert_matrix_close(
        &Matrix::from_rows(&merged.to_rows()[..2]).unwrap(),
        &[[4.0748, 0.0], [3.2966, 2.3951]],
        1e-4,
    );
    // Each block is a scaled, translated copy of the input layout.
    let scale_big = dist(&merged, 0, 1) / dist(&lb, 0, 1);
    for i in 0..10 {
        for j in i + 1..10 {
            assert_close(dist(&merged, i, j), scale_big * dist(&lb, i, j), 1e-9);
        }
    }
    let scale_small = dist(&merged, 10, 11) / dist(&ls, 0, 1);
    assert_close(dist(&merged, 11, 12), scale_small * dist(&ls, 1, 2), 1e-9);
    // Components do not overlap: the small one lies outside the big circle.
    for v in 10..13 {
        assert!(merged[(v, 0)].hypot(merged[(v, 1)]) > 4.0748);
    }
}

#[test]
fn merge_dla_errors() {
    let g = cycle(3);
    let l = g.layout_circle(..).unwrap();
    assert!(layout_merge_dla(Vec::<Graph>::new(), &[]).is_err());
    assert!(layout_merge_dla([&g], &[]).is_err());
    assert!(layout_merge_dla([&g], &[Matrix::zeros(3, 3)]).is_err());
    assert!(layout_merge_dla([&g], &[Matrix::zeros(2, 2)]).is_err());
    assert!(layout_merge_dla([&g], &[l]).is_ok());
}

#[test]
fn align_centers_and_rotates() {
    // A path drawn along the diagonal y = x, shifted away from the origin.
    let g = path(5);
    let mut l = Matrix::from_rows(
        &(0..5)
            .map(|i| [10.0 + i as f64, 20.0 + i as f64])
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let before: Vec<f64> = (0..4).map(|i| dist(&l, i, i + 1)).collect();
    g.layout_align(&mut l).unwrap();
    // Centered...
    assert_close(l.column(0).iter().sum::<f64>(), 0.0, 1e-9);
    assert_close(l.column(1).iter().sum::<f64>(), 0.0, 1e-9);
    // ... aligned with the x axis ...
    assert!(l.column(1).iter().all(|y| y.abs() < 1e-9));
    // ... and rigidly moved.
    for (i, d) in before.iter().enumerate() {
        assert_close(dist(&l, i, i + 1), *d, 1e-9);
    }
    // Works in 3D too, and checks the shape.
    let mut l3 = g.layout_random_3d().unwrap();
    g.layout_align(&mut l3).unwrap();
    assert_eq!(l3.shape(), (5, 3));
    let mut wrong = Matrix::zeros(4, 2);
    assert_eq!(
        g.layout_align(&mut wrong).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

// ---------------------------------------------------------------------------
// Use case
// ---------------------------------------------------------------------------

/// Story: drawing the karate club for a poster. We want a pretty,
/// reproducible picture where the two factions (led by the instructor,
/// vertex 0, and the administrator, vertex 33) are well separated, the
/// drawing is axis-aligned, and it fits a normalized `[0, 1]²` canvas.
#[test]
fn use_case_poster_of_the_karate_club() {
    let g = karate();

    // 1. A deterministic initial placement: Kamada-Kawai starts from a circle.
    let kk = g
        .layout_kamada_kawai(&KamadaKawaiOptions::default())
        .unwrap();

    // 2. Polish it with a seeded Fruchterman-Reingold run starting from KK.
    let opts = FruchtermanReingoldOptions::default().with_initial(&kk);
    let mut layout = seeded(2024, || g.layout_fruchterman_reingold(&opts).unwrap());
    assert_eq!(
        seeded(2024, || g.layout_fruchterman_reingold(&opts).unwrap()),
        layout,
        "the poster is reproducible"
    );

    // 3. Align it with the axes and normalize to the unit square.
    g.layout_align(&mut layout).unwrap();
    for k in 0..2 {
        let col = layout.column(k).to_vec();
        let (lo, hi) = col
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
                (a.min(x), b.max(x))
            });
        for (i, x) in col.iter().enumerate() {
            layout[(i, k)] = (x - lo) / (hi - lo);
        }
    }
    assert!(layout.as_slice().iter().all(|x| (0.0..=1.0).contains(x)));

    // 4. The leaders are far apart: farther than the typical edge.
    let mean_edge = mean_edge_length(&g, &layout);
    assert!(dist(&layout, 0, 33) > 1.5 * mean_edge);

    // 5. Each member is (on average) closer to the leader of its own faction.
    //    Known factions after the split (Zachary 1977).
    let instructor_side = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21];
    let mut correct = 0;
    for v in 0..34usize {
        if v == 0 || v == 33 {
            continue;
        }
        let closer_to_instructor = dist(&layout, v, 0) < dist(&layout, v, 33);
        if closer_to_instructor == instructor_side.contains(&v) {
            correct += 1;
        }
    }
    assert!(
        correct >= 26,
        "only {correct}/32 members drawn near their leader"
    );

    // 6. For the legend: a tree view of a BFS-like hierarchy from the
    //    instructor, with depth as the y coordinate.
    let tree = g
        .layout_reingold_tilford(NeighborMode::All, Some(&[0]), None)
        .unwrap();
    assert_eq!(tree[(0, 1)], 0.0);
    assert_eq!(tree[(33, 1)], 2.0); // 0 - 8 - 33
    assert!(tree.column(1).iter().all(|&y| y <= 3.0));
}

// ---------------------------------------------------------------------------
// Review additions: edge cases
// ---------------------------------------------------------------------------

#[test]
fn huge_counts_saturate_instead_of_wrapping() {
    // `usize::MAX` must not wrap to a negative width (which igraph would read
    // as "automatic"): all vertices end up on a single row.
    let g = Graph::new(10, false);
    let l = g.layout_grid(Some(usize::MAX)).unwrap();
    assert!(l.column(1).iter().all(|&y| y == 0.0));
    assert_eq!(l.column(0), (0..10).map(f64::from).collect::<Vec<_>>());
    // Some(0) means automatic, like None.
    assert_eq!(
        g.layout_grid(Some(0)).unwrap(),
        g.layout_grid(None).unwrap()
    );
    let l = g.layout_grid_3d(Some(usize::MAX), None).unwrap();
    assert!(
        l.column(1)
            .iter()
            .chain(l.column(2).iter())
            .all(|&c| c == 0.0)
    );
}

#[test]
fn reingold_tilford_unreachable_vertices_and_several_roots() {
    // Two out-trees 0 -> {1, 2} and 3 -> {4, 5}.
    let g = Graph::from_edges(&[(0, 1), (0, 2), (3, 4), (3, 5)], 6, true).unwrap();
    // Only root 0: the vertices it cannot reach hang directly below it.
    let l = g
        .layout_reingold_tilford(NeighborMode::Out, Some(&[0]), None)
        .unwrap();
    assert_eq!(l.column(1), &[0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    // Several roots sit below a hidden common root: at depth 1.
    let l = g
        .layout_reingold_tilford(NeighborMode::Out, Some(&[0, 3]), None)
        .unwrap();
    assert_eq!(l.column(1), &[1.0, 2.0, 2.0, 1.0, 2.0, 2.0]);
    // Automatic roots on a small graph (under 500 vertices): igraph 1.0.1
    // picks the highest degree vertex, not the lowest eccentricity one.
    let broom =
        Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (0, 6)], 7, false).unwrap();
    assert_eq!(
        broom
            .roots_for_tree_layout(NeighborMode::All, RootChoice::Degree)
            .unwrap(),
        vec![0]
    );
    assert_eq!(
        broom
            .roots_for_tree_layout(NeighborMode::All, RootChoice::Eccentricity)
            .unwrap(),
        vec![2]
    );
    let auto = broom
        .layout_reingold_tilford(NeighborMode::All, None, None)
        .unwrap();
    assert_eq!(auto[(0, 1)], 0.0);
}

#[test]
fn umap_weights_do_not_detect_multi_edges() {
    // igraph rejects loops but not parallel edges: one of the two parallel
    // edges carries the symmetrized weight, the other gets 0.
    let multi = Graph::from_edges(&[(0, 1), (0, 1), (1, 2)], 3, false).unwrap();
    let w = multi
        .layout_umap_compute_weights(Some(&[1.0, 2.0, 3.0]))
        .unwrap();
    assert_eq!(w.len(), 3);
    assert!(w[0] > 0.0);
    assert_eq!(w[1], 0.0);
    let looped = Graph::from_edges(&[(0, 1), (1, 1)], 2, false).unwrap();
    assert_eq!(
        looped.layout_umap_compute_weights(None).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        multi
            .layout_umap_compute_weights(Some(&[1.0, f64::NAN, 3.0]))
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn reingold_tilford_empty_rootlevel_is_ignored() {
    let g = Graph::from_edges(&[(0, 1), (0, 2), (3, 4), (3, 5)], 6, true).unwrap();
    let plain = g
        .layout_reingold_tilford(NeighborMode::Out, Some(&[0, 3]), None)
        .unwrap();
    let empty = g
        .layout_reingold_tilford(NeighborMode::Out, Some(&[0, 3]), Some(&[]))
        .unwrap();
    assert_eq!(plain, empty);
    // Negative levels are rejected.
    assert_eq!(
        g.layout_reingold_tilford(NeighborMode::Out, Some(&[0, 3]), Some(&[0, -1]))
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn roots_for_tree_layout_per_component_and_direction() {
    // Out-tree 0 -> {1, 2}, and a separate out-tree 3 -> 4.
    let g = Graph::from_edges(&[(0, 1), (0, 2), (3, 4)], 5, true).unwrap();
    let mut out = g
        .roots_for_tree_layout(NeighborMode::Out, RootChoice::Degree)
        .unwrap();
    out.sort();
    assert_eq!(out, vec![0, 3]);
    // Following edges backwards, the roots are the sinks 1, 2 and 4 (the
    // strongly connected components without outgoing edges): each one
    // reaches only its own ancestors, so all three are needed.
    let mut into = g
        .roots_for_tree_layout(NeighborMode::In, RootChoice::Degree)
        .unwrap();
    into.sort();
    assert_eq!(into, vec![1, 2, 4]);
    assert!(
        g.roots_for_tree_layout(NeighborMode::All, RootChoice::Eccentricity)
            .unwrap()
            .len()
            == 2
    );
}

#[test]
fn kamada_kawai_respects_bounds() {
    let g = cycle(6);
    let lo = [0.0; 6];
    let hi = [0.5; 6];
    let opts = KamadaKawaiOptions {
        minx: Some(&lo),
        maxx: Some(&hi),
        miny: Some(&lo),
        maxy: Some(&hi),
        ..Default::default()
    };
    let l = seeded(3, || g.layout_kamada_kawai(&opts).unwrap());
    assert!(l.as_slice().iter().all(|x| (0.0..=0.5).contains(x)));
    let opts = KamadaKawaiOptions {
        minz: Some(&lo),
        maxz: Some(&lo),
        ..Default::default()
    };
    let l3 = seeded(3, || g.layout_kamada_kawai_3d(&opts).unwrap());
    assert!(l3.column(2).iter().all(|&z| z == 0.0));
    // Wrong bound length.
    let opts = KamadaKawaiOptions {
        minx: Some(&lo[..2]),
        ..Default::default()
    };
    assert_eq!(
        g.layout_kamada_kawai(&opts).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn drl_options_are_copy() {
    let a = DrlOptions::from_template(DrlTemplate::Coarsen);
    let mut b = a;
    b.edge_cut = 0.0;
    assert_eq!(
        a.edge_cut,
        DrlOptions::from_template(DrlTemplate::Coarsen).edge_cut
    );
    assert_eq!(b.edge_cut, 0.0);
}

// ---------------------------------------------------------------------------
// Values and checks from igraph's own unit tests (tests/unit/*.c, *.out)
// ---------------------------------------------------------------------------

#[test]
fn graphopt_matches_igraph_unit_test() {
    // tests/unit/igraph_layout_graphopt.out: 4 vertices in a line, no
    // repulsion, spring length 1 and a seed.
    let g = Graph::ring(4, false, false, false).unwrap();
    let start =
        Matrix::from_rows(&[[0.15, -0.15], [0.05, -0.05], [-0.05, 0.05], [-0.15, 0.15]]).unwrap();
    let opts = GraphoptOptions {
        node_charge: 0.0,
        spring_length: 1.0,
        spring_constant: 10.0,
        initial: Some(&start),
        ..Default::default()
    };
    assert_matrix_close(
        &g.layout_graphopt(&opts).unwrap(),
        &[
            [1.06066, -1.06066],
            [0.353553, -0.353553],
            [-0.353553, 0.353553],
            [-1.06066, 1.06066],
        ],
        EPS,
    );
    // Consecutive vertices end up at the rest length of the springs.
    let l = g.layout_graphopt(&opts).unwrap();
    for i in 0..3 {
        assert_close(dist(&l, i, i + 1), 1.0, 1e-5);
    }
    // ... and with no allowed movement, nothing moves.
    let frozen = GraphoptOptions {
        max_sa_movement: 0.0,
        ..opts.clone()
    };
    assert_eq!(g.layout_graphopt(&frozen).unwrap(), start);

    // Empty graph, and bounded results on small graphs.
    assert_eq!(
        Graph::new(0, false)
            .layout_graphopt(&GraphoptOptions::default())
            .unwrap()
            .shape(),
        (0, 2)
    );
    let full = Graph::full(4, false, false).unwrap();
    let l = seeded(42, || {
        full.layout_graphopt(&GraphoptOptions::default()).unwrap()
    });
    assert!(l.as_slice().iter().all(|x| x.abs() <= 20.0));
    let no_repulsion = GraphoptOptions {
        node_charge: 0.0,
        ..Default::default()
    };
    let l = seeded(42, || full.layout_graphopt(&no_repulsion).unwrap());
    assert!(l.as_slice().iter().all(|x| x.abs() <= 1.0));
    // Wrongly shaped start.
    let bad = GraphoptOptions {
        initial: Some(&Matrix::zeros(3, 2)),
        ..Default::default()
    };
    assert_eq!(
        g.layout_graphopt(&bad).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn gem_matches_igraph_unit_test() {
    // tests/unit/igraph_layout_gem.out: a seeded singleton with no iterations
    // stays where it is.
    let single = Graph::new(1, false);
    let start = Matrix::from_rows(&[[3.0, 4.0]]).unwrap();
    let opts = GemOptions {
        maxiter: Some(0),
        temp_max: Some(1.0),
        temp_min: 0.1,
        temp_init: Some(1.0),
        initial: Some(&start),
    };
    assert_eq!(single.layout_gem(&opts).unwrap(), start);

    // Two vertices with many iterations stay bounded.
    let g = Graph::from_edges(&[(0, 1)], 2, false).unwrap();
    let start = Matrix::from_rows(&[[3.0, 4.0], [5.0, 6.0]]).unwrap();
    let opts = GemOptions {
        maxiter: Some(100_000),
        temp_max: Some(1.0),
        temp_min: 0.1,
        temp_init: Some(1.0),
        initial: Some(&start),
    };
    let l = seeded(42, || g.layout_gem(&opts).unwrap());
    assert!(l.as_slice().iter().all(|x| x.abs() < 1000.0));

    // Invalid temperatures and seeds.
    let base = GemOptions {
        maxiter: Some(3),
        temp_max: Some(1.0),
        temp_min: 0.1,
        temp_init: Some(1.0),
        initial: None,
    };
    for bad in [
        GemOptions {
            temp_max: Some(-1.0),
            ..base.clone()
        },
        GemOptions {
            temp_min: -1.0,
            ..base.clone()
        },
        GemOptions {
            temp_init: Some(-1.0),
            ..base.clone()
        },
        GemOptions {
            temp_max: Some(1.0),
            temp_min: 2.0,
            temp_init: Some(1.5),
            ..base.clone()
        },
        GemOptions {
            initial: Some(&Matrix::zeros(10, 2)),
            ..base.clone()
        },
    ] {
        assert_eq!(
            g.layout_gem(&bad).unwrap_err().kind(),
            ErrorKind::InvalidValue
        );
    }
}

#[test]
fn davidson_harel_matches_igraph_unit_test() {
    // tests/unit/igraph_layout_davidson_harel.c: the layouts stay within
    // (-20, 20) with these energy weights.
    let opts = DavidsonHarelOptions {
        maxiter: 5,
        fineiter: Some(5),
        cool_fact: 0.75,
        weight_node_dist: 1.0,
        weight_border: 0.1,
        weight_edge_lengths: Some(0.5),
        weight_edge_crossings: Some(0.8),
        weight_node_edge_dist: Some(0.2),
        initial: None,
    };
    let empty = Graph::new(0, true).layout_davidson_harel(&opts).unwrap();
    assert_eq!(empty.nrow(), 0);
    for g in [Graph::new(10, true), Graph::full(10, true, true).unwrap()] {
        let l = seeded(42, || g.layout_davidson_harel(&opts).unwrap());
        assert_eq!(l.shape(), (10, 2));
        assert!(l.as_slice().iter().all(|x| x.abs() < 20.0), "{l}");
    }
    // Zero iterations from a given start keep it.
    let g = cycle(5);
    let start = g.layout_circle(..).unwrap();
    let frozen = DavidsonHarelOptions {
        maxiter: 0,
        fineiter: Some(0),
        initial: Some(&start),
        ..Default::default()
    };
    assert_eq!(g.layout_davidson_harel(&frozen).unwrap(), start);
}

#[test]
fn kamada_kawai_matches_igraph_unit_test() {
    // tests/unit/igraph_layout_kamada_kawai.c
    let empty = Graph::new(0, false);
    assert_eq!(
        empty
            .layout_kamada_kawai(&KamadaKawaiOptions::default())
            .unwrap()
            .shape(),
        (0, 2)
    );
    assert_eq!(
        empty
            .layout_kamada_kawai_3d(&KamadaKawaiOptions::default())
            .unwrap()
            .shape(),
        (0, 3)
    );

    // Two connected vertices settle at the rest length of their spring,
    // sqrt(n) * d / diameter = sqrt(2).
    let g = Graph::from_edges(&[(0, 1)], 2, false).unwrap();
    let opts = KamadaKawaiOptions {
        maxiter: Some(1000),
        kkconst: Some(2.0),
        ..Default::default()
    };
    let l = g.layout_kamada_kawai(&opts).unwrap();
    assert_close(dist(&l, 0, 1), 2f64.sqrt(), 1e-3);
    let l = g.layout_kamada_kawai_3d(&opts).unwrap();
    assert_close(dist(&l, 0, 1), 2f64.sqrt(), 1e-3);

    // "Full graph of 5 vertices, seed and no iterations": the seed is kept.
    let full = Graph::full(5, false, false).unwrap();
    let start =
        Matrix::from_rows(&[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]).unwrap();
    let opts = KamadaKawaiOptions {
        maxiter: Some(0),
        initial: Some(&start),
        ..Default::default()
    };
    assert_eq!(full.layout_kamada_kawai(&opts).unwrap(), start);
}

#[test]
fn lgl_on_trees_matches_igraph_unit_test() {
    // tests/unit/igraph_layout_lgl.c runs LGL repeatedly on a ternary tree
    // (stress-testing the internal grid).
    let g = kary_tree(100, 3);
    let n = g.vcount() as f64;
    let opts = LglOptions {
        maxiter: 150,
        maxdelta: Some(n),
        area: Some(n * n),
        coolexp: 1.5,
        repulserad: Some(n * n * n),
        cellsize: Some(n.sqrt().sqrt()),
        root: Some(0),
    };
    rng::seed(33).unwrap();
    for _ in 0..10 {
        let l = g.layout_lgl(&opts).unwrap();
        assert_eq!(l.shape(), (100, 2));
        assert!(all_finite(&l));
    }
    // Neighbors are drawn closer than arbitrary pairs.
    let l = g.layout_lgl(&opts).unwrap();
    let mean_edge = mean_edge_length(&g, &l);
    let mut sum = 0.0;
    for i in 0..100 {
        for j in i + 1..100 {
            sum += dist(&l, i, j);
        }
    }
    assert!(mean_edge < sum / (100.0 * 99.0 / 2.0));
}

// ---------------------------------------------------------------------------
// Randomness and threads
// ---------------------------------------------------------------------------

#[test]
fn seeded_layouts_are_reproducible_in_parallel_threads() {
    // Every thread has its own default generator: seeding it in each thread
    // gives the same layouts, whatever the other threads do meanwhile.
    let run = || {
        let g = karate();
        rng::seed(7).unwrap();
        let fr = g
            .layout_fruchterman_reingold(&FruchtermanReingoldOptions::default())
            .unwrap();
        let drl = g.layout_drl(&DrlOptions::default(), None, None).unwrap();
        (fr, drl)
    };
    let expected = run();
    let handles: Vec<_> = (0..4).map(|_| std::thread::spawn(run)).collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), expected);
    }
}

#[test]
fn scoped_generator_leaves_the_default_stream_alone() {
    let g = Graph::new(20, false);
    // The scoped generator gives the same result as a default generator of
    // the same type and seed would...
    let scoped = Rng::new(RngType::Pcg32, 5)
        .unwrap()
        .scoped(|| g.layout_random().unwrap());
    let again = Rng::new(RngType::Pcg32, 5)
        .unwrap()
        .scoped(|| g.layout_random().unwrap());
    assert_eq!(scoped, again);
    // ... and does not consume numbers of the thread's default generator.
    rng::seed(1).unwrap();
    let direct = g.layout_random().unwrap();
    rng::seed(1).unwrap();
    let _ = Rng::new(RngType::Pcg64, 9)
        .unwrap()
        .scoped(|| g.layout_random().unwrap());
    assert_eq!(g.layout_random().unwrap(), direct);
}

// ---------------------------------------------------------------------------
// Cross-module stories
// ---------------------------------------------------------------------------

#[test]
fn umap_separates_the_clusters_of_a_nearest_neighbor_graph() {
    // Two well separated blobs of points in the plane; UMAP of their
    // 3-nearest-neighbor graph keeps the blobs apart.
    let mut pts = vec![];
    for (cx, cy) in [(0.0, 0.0), (100.0, 0.0)] {
        for k in 0..8 {
            let a = k as f64 * std::f64::consts::TAU / 8.0;
            pts.push([cx + a.cos(), cy + a.sin()]);
        }
    }
    let points = Matrix::from_rows(&pts).unwrap();
    let g =
        Graph::nearest_neighbor_graph(&points, Metric::Euclidean, Some(3), None, false).unwrap();
    // No edge between the blobs.
    assert!(g.edge_list().iter().all(|&(u, v)| (u < 8) == (v < 8)));
    let distances = g.spatial_edge_lengths(&points, Metric::Euclidean).unwrap();
    let opts = UmapOptions {
        distances: Some(&distances),
        epochs: 200,
        ..Default::default()
    };
    let l = seeded(3, || g.layout_umap(&opts).unwrap());
    assert!(all_finite(&l));
    let mean = |pairs: &mut dyn Iterator<Item = (usize, usize)>| {
        let v: Vec<f64> = pairs.map(|(i, j)| dist(&l, i, j)).collect();
        v.iter().sum::<f64>() / v.len() as f64
    };
    let within = mean(
        &mut (0..16)
            .flat_map(|i| (i + 1..16).map(move |j| (i, j)))
            .filter(|&(i, j)| (i < 8) == (j < 8)),
    );
    let across = mean(&mut (0..8).flat_map(|i| (8..16).map(move |j| (i, j))));
    assert!(within < across, "within {within} vs across {across}");
}

#[test]
fn bipartite_layout_from_computed_types() {
    // K_{3,4}: types from the bipartite module, no crossings to avoid, rows
    // at y = 0 (type true) and y = vgap (type false).
    let k34 = Graph::full_bipartite(3, 4, false, NeighborMode::All).unwrap();
    let g = k34.graph;
    // The 2-coloring of a connected bipartite graph is unique up to a flip.
    let types = g.bipartite_types().unwrap().unwrap();
    let flipped: Vec<bool> = types.iter().map(|t| !t).collect();
    assert!(types == k34.types || flipped == k34.types);
    let l = g.layout_bipartite(&types, 1.0, 2.0, 100).unwrap();
    for (v, &t) in types.iter().enumerate() {
        assert_eq!(l[(v, 1)], if t { 0.0 } else { 2.0 });
    }
    // Vertices of a row are at least hgap apart.
    for t in [false, true] {
        let mut xs: Vec<f64> = (0..7)
            .filter(|&v| types[v] == t)
            .map(|v| l[(v, 0)])
            .collect();
        xs.sort_by(f64::total_cmp);
        assert!(xs.windows(2).all(|w| w[1] - w[0] >= 1.0 - 1e-12));
    }
}

#[test]
fn mds_default_distances_are_shortest_paths() {
    // On a path, the shortest path distances are realizable on a line, so
    // MDS without a matrix reproduces them exactly (like with the explicit
    // matrix computed by the paths module).
    let g = path(6);
    let d = g.distances(.., .., None, NeighborMode::All).unwrap();
    let implicit = g.layout_mds(None, 2).unwrap();
    let explicit = g.layout_mds(Some(&d), 2).unwrap();
    for i in 0..6 {
        for j in 0..6 {
            assert_close(dist(&implicit, i, j), d[(i, j)], 1e-9);
            assert_close(dist(&explicit, i, j), d[(i, j)], 1e-9);
        }
    }
}

#[test]
fn merge_dla_of_decomposed_components() {
    // A graph with three components, laid out separately then merged.
    let g = Graph::from_edges(
        &[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 4),
            (7, 8),
        ],
        9,
        false,
    )
    .unwrap();
    let parts = g.decompose(Connectedness::Weak, None, 1).unwrap();
    assert_eq!(parts.len(), 3);
    let layouts: Vec<Matrix> = parts.iter().map(|p| p.layout_circle(..).unwrap()).collect();
    let refs: Vec<&Graph> = parts.iter().collect();
    let merged = seeded(1, || layout_merge_dla(&refs, &layouts).unwrap());
    assert_eq!(merged.shape(), (9, 2));
    // Rows follow the concatenation of the component layouts, and every
    // block is a scaled copy of its layout (a square, a triangle, an edge).
    let mut offset = 0;
    for (p, l) in parts.iter().zip(&layouts) {
        let n = p.vcount();
        let scale = dist(&merged, offset, offset + 1) / dist(l, 0, 1);
        for i in 0..n {
            for j in i + 1..n {
                assert_close(
                    dist(&merged, offset + i, offset + j),
                    scale * dist(l, i, j),
                    1e-9,
                );
            }
        }
        offset += n;
    }
}

#[test]
fn sugiyama_reverses_a_feedback_arc_set() {
    // A directed 3-cycle: automatic layering must break the cycle, so the
    // three vertices end up on three different layers, and exactly one
    // edge (a feedback arc set of size 1) points upwards.
    let g = Graph::ring(3, true, false, true).unwrap();
    assert_eq!(
        g.feedback_arc_set(None, FasAlgorithm::ExactIp)
            .unwrap()
            .len(),
        1
    );
    let s = g.layout_sugiyama(&SugiyamaOptions::default()).unwrap();
    let mut ys = s.coords.column(1).to_vec();
    ys.sort_by(f64::total_cmp);
    assert_eq!(ys, vec![0.0, 1.0, 2.0]);
    let upwards = g
        .edge_list()
        .iter()
        .filter(|&&(u, v)| s.coords[(u as usize, 1)] > s.coords[(v as usize, 1)])
        .count();
    assert_eq!(upwards, 1);
}

#[test]
fn natural_layouts_of_lattices_stars_and_trees() {
    // The grid layout draws a square lattice at its lattice points: every
    // edge has length 1.
    let lattice = Graph::square_lattice(&[4, 3], 1, false, false, None).unwrap();
    let l = lattice.layout_grid(Some(4)).unwrap();
    let lengths = lattice.spatial_edge_lengths(&l, Metric::Euclidean).unwrap();
    assert!(lengths.iter().all(|&d| d == 1.0));

    // Star layout of a star graph: all spokes have length 1.
    let star = Graph::star(7, StarMode::Undirected, 0).unwrap();
    let l = star.layout_star(0, None).unwrap();
    let lengths = star.spatial_edge_lengths(&l, Metric::Euclidean).unwrap();
    assert!(lengths.iter().all(|&d| (d - 1.0).abs() < 1e-12));

    // A tree drawn by Reingold-Tilford: every edge joins consecutive levels.
    let t = kary_tree(40, 3);
    assert!(t.is_tree(NeighborMode::All).unwrap());
    let l = t
        .layout_reingold_tilford(NeighborMode::All, Some(&[0]), None)
        .unwrap();
    for (u, v) in t.edge_list() {
        assert_eq!((l[(u as usize, 1)] - l[(v as usize, 1)]).abs(), 1.0);
    }
}

#[test]
fn fruchterman_reingold_equilibrium_edge_length() {
    // f_a(d) = -w d² balances f_r(d) = 1/d at d = w^(-1/3) (the C
    // documentation of igraph 1.0.0 and 1.0.1 says `1/w^3`, a typo).
    let g = Graph::from_edges(&[(0, 1)], 2, false).unwrap();
    for w in [1.0, 8.0, 0.125] {
        let weights = [w];
        let opts = FruchtermanReingoldOptions::default()
            .with_weights(&weights)
            .with_niter(2000);
        let l = seeded(1, || g.layout_fruchterman_reingold(&opts).unwrap());
        assert_close(dist(&l, 0, 1), w.powf(-1.0 / 3.0), 1e-3);
    }
}

// ---------------------------------------------------------------------------
// Regression tests for inputs that used to abort the process or reach
// undefined behavior inside igraph.
// ---------------------------------------------------------------------------

fn invalid<T: std::fmt::Debug>(r: Result<T>) {
    assert_eq!(r.unwrap_err().kind(), ErrorKind::InvalidValue);
}

#[test]
fn lgl_rejects_non_finite_and_degenerate_parameters() {
    let g = cycle(5);
    let lgl = |o: LglOptions| g.layout_lgl(&o);
    let d = LglOptions::default;
    // These used to fail IGRAPH_ASSERTs in igraph's grid code (abort).
    invalid(lgl(LglOptions {
        area: Some(f64::INFINITY),
        ..d()
    }));
    invalid(lgl(LglOptions {
        area: Some(f64::NAN),
        ..d()
    }));
    invalid(lgl(LglOptions {
        cellsize: Some(f64::NAN),
        ..d()
    }));
    invalid(lgl(LglOptions {
        maxdelta: Some(f64::INFINITY),
        ..d()
    }));
    invalid(lgl(LglOptions {
        repulserad: Some(f64::NAN),
        ..d()
    }));
    invalid(lgl(LglOptions {
        coolexp: f64::NAN,
        ..d()
    }));
    invalid(lgl(LglOptions {
        coolexp: 0.0,
        ..d()
    }));
    // A grid of absurdly many cells (it used to overflow a C integer).
    invalid(lgl(LglOptions {
        area: Some(1e300),
        cellsize: Some(1e-300),
        ..d()
    }));
    invalid(lgl(LglOptions {
        area: Some(1e30),
        cellsize: Some(1e-10),
        ..d()
    }));
    // An infinite repulsion radius is harmless.
    let l = seeded(1, || {
        lgl(LglOptions {
            repulserad: Some(f64::INFINITY),
            ..d()
        })
        .unwrap()
    });
    assert!(all_finite(&l));
    // Reasonable explicit parameters still work.
    let l = seeded(1, || {
        lgl(LglOptions {
            area: Some(100.0),
            cellsize: Some(0.5),
            root: Some(0),
            ..d()
        })
        .unwrap()
    });
    assert!(all_finite(&l));
    // The null graph skips all parameter checks, like igraph does.
    let empty = Graph::new(0, false);
    assert_eq!(
        empty
            .layout_lgl(&LglOptions {
                area: Some(f64::NAN),
                ..d()
            })
            .unwrap()
            .nrow(),
        0
    );
}

#[test]
fn align_rejects_non_finite_layouts() {
    let g = cycle(3);
    for bad in [f64::INFINITY, f64::NAN] {
        let mut m = Matrix::from_rows(&[[0.0, 0.0], [1.0, 0.0], [bad, 1.0]]).unwrap();
        invalid(g.layout_align(&mut m));
    }
}

#[test]
fn seeded_layouts_reject_non_finite_starts() {
    let g = cycle(3);
    let start = Matrix::from_rows(&[[f64::NAN, 0.0], [1.0, 0.0], [0.0, 1.0]]).unwrap();
    let fr = FruchtermanReingoldOptions {
        niter: 3,
        grid: LayoutGrid::Grid,
        initial: Some(&start),
        ..Default::default()
    };
    invalid(g.layout_fruchterman_reingold(&fr));
    invalid(g.layout_kamada_kawai(&KamadaKawaiOptions::default().with_initial(&start)));
    invalid(g.layout_drl(&DrlOptions::default(), None, Some(&start)));
    invalid(g.layout_graphopt(&GraphoptOptions {
        initial: Some(&start),
        ..Default::default()
    }));
    invalid(g.layout_gem(&GemOptions {
        initial: Some(&start),
        ..Default::default()
    }));
    invalid(g.layout_davidson_harel(&DavidsonHarelOptions {
        initial: Some(&start),
        ..Default::default()
    }));
    let inf = Matrix::from_rows(&[[0.0, 0.0], [f64::INFINITY, 0.0], [0.0, 1.0]]).unwrap();
    invalid(
        g.layout_fruchterman_reingold(&FruchtermanReingoldOptions::default().with_initial(&inf)),
    );
}

#[test]
fn force_directed_bounds_reject_nan_but_accept_infinity() {
    let g = cycle(4);
    let nan = [0.0, f64::NAN, 0.0, 0.0];
    invalid(g.layout_fruchterman_reingold(&FruchtermanReingoldOptions {
        minx: Some(&nan),
        ..Default::default()
    }));
    // A lower bound of +inf or an upper bound of -inf used to make igraph's
    // random start fail with an error the layouts ignore, then corrupt the
    // heap (`malloc(): invalid size`).
    let mut bad_lo = [0.0; 4];
    bad_lo[1] = f64::INFINITY;
    let mut bad_hi = [0.0; 4];
    bad_hi[2] = f64::NEG_INFINITY;
    for grid in [LayoutGrid::NoGrid, LayoutGrid::Grid] {
        invalid(g.layout_fruchterman_reingold(&FruchtermanReingoldOptions {
            minx: Some(&bad_lo),
            grid,
            ..Default::default()
        }));
        invalid(g.layout_fruchterman_reingold(&FruchtermanReingoldOptions {
            maxy: Some(&bad_hi),
            grid,
            ..Default::default()
        }));
    }
    invalid(
        g.layout_fruchterman_reingold_3d(&FruchtermanReingoldOptions {
            minz: Some(&bad_lo),
            ..Default::default()
        }),
    );
    invalid(g.layout_kamada_kawai(&KamadaKawaiOptions {
        maxx: Some(&bad_hi),
        ..Default::default()
    }));
    invalid(g.layout_kamada_kawai_3d(&KamadaKawaiOptions {
        maxz: Some(&bad_hi),
        ..Default::default()
    }));
    // An infinite bound in the right direction means "no bound".
    let (lo, hi) = ([f64::NEG_INFINITY; 4], [f64::INFINITY; 4]);
    let l = seeded(3, || {
        g.layout_fruchterman_reingold(&FruchtermanReingoldOptions {
            minx: Some(&lo),
            maxx: Some(&hi),
            ..Default::default()
        })
        .unwrap()
    });
    assert!(all_finite(&l));
}

#[test]
fn drl_rejects_non_finite_and_huge_weights() {
    let g = cycle(5);
    let opts = DrlOptions::default();
    for w in [f64::INFINITY, f64::NAN, 1e200, 1e308, 0.0, -1.0] {
        let weights = [1.0, 1.0, w, 1.0, 1.0];
        invalid(g.layout_drl(&opts, Some(&weights), None));
        invalid(g.layout_drl_3d(&opts, Some(&weights), None));
    }
    invalid(g.layout_drl(&opts, Some(&[1e200; 5]), None));
    // Large but sane weights (up to 1e20) are fine.
    let l = seeded(2, || g.layout_drl(&opts, Some(&[1e20; 5]), None).unwrap());
    assert!(all_finite(&l));
}

#[test]
fn reingold_tilford_rejects_huge_rootlevels() {
    let g = cycle(5);
    for circular in [false, true] {
        let f = |levels: &[i64]| {
            if circular {
                g.layout_reingold_tilford_circular(NeighborMode::All, Some(&[0, 2]), Some(levels))
            } else {
                g.layout_reingold_tilford(NeighborMode::All, Some(&[0, 2]), Some(levels))
            }
        };
        invalid(f(&[i64::MAX, 1]));
        invalid(f(&[i64::MAX / 2, i64::MAX / 2]));
        invalid(f(&[1 << 41, 0]));
        assert_eq!(f(&[1, 0]).unwrap().nrow(), 5);
    }
}

#[test]
fn davidson_harel_rejects_overflowing_iteration_counts() {
    let g = cycle(5);
    invalid(g.layout_davidson_harel(&DavidsonHarelOptions {
        maxiter: 3,
        fineiter: Some(i64::MAX as usize),
        ..Default::default()
    }));
}

#[test]
fn mds_rejects_non_finite_distances() {
    let g = cycle(3);
    let d =
        Matrix::from_rows(&[[0.0, 1.0, f64::NAN], [1.0, 0.0, 1.0], [f64::NAN, 1.0, 0.0]]).unwrap();
    invalid(g.layout_mds(Some(&d), 2));
    let d =
        Matrix::from_rows(&[[0.0, 1.0, f64::INFINITY], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]).unwrap();
    invalid(g.layout_mds(Some(&d), 2));
}

/// A directed graph with mutual edges: an ALL-mode adjacency list built on it
/// (e.g. by `count_triangles`) used to cache a wrong "has multi-edges" flag,
/// and the OUT-mode Reingold–Tilford layout then aborted the process
/// (`caching.c: Assertion failed`).
#[test]
fn reingold_tilford_after_all_mode_adjlist_on_mutual_edges() {
    let g = Graph::from_edges(&[(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)], 3, true).unwrap();
    for circular in [false, true] {
        let _ = g.count_triangles().unwrap();
        for mode in [NeighborMode::Out, NeighborMode::In, NeighborMode::All] {
            let l = if circular {
                g.layout_reingold_tilford_circular(mode, Some(&[0]), None)
            } else {
                g.layout_reingold_tilford(mode, Some(&[0]), None)
            }
            .unwrap();
            assert_eq!(l.nrow(), 3);
            // The layout does not leave a wrong cached value behind.
            assert!(!g.has_multiple().unwrap());
        }
    }
}

/// `layout_merge_dla` accepts any collection of graphs or graph references.
#[test]
fn merge_dla_accepts_graph_collections() {
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4)], 5, false).unwrap();
    let parts = g.decompose(Connectedness::Weak, None, 1).unwrap();
    let layouts: Vec<Matrix> = parts.iter().map(|p| p.layout_circle(..).unwrap()).collect();
    let a = seeded(4, || layout_merge_dla(&parts, &layouts).unwrap());
    let refs: Vec<&Graph> = parts.iter().collect();
    let b = seeded(4, || layout_merge_dla(&refs, &layouts).unwrap());
    let c = seeded(4, || {
        layout_merge_dla(refs.iter().copied(), &layouts).unwrap()
    });
    let d = seeded(4, || layout_merge_dla(parts.clone(), &layouts).unwrap());
    assert_eq!(a, b);
    assert_eq!(a, c);
    assert_eq!(a, d);
}

//! Cross-module, real-world style use cases of the bindings.
//!
//! Every test tells a small story that goes through several modules of the
//! crate (constructors, games, centrality, community, layout, paths, flow,
//! isomorphism, bipartite, misc, foreign, attributes, linalg, ...) and checks
//! *real* facts along the way: textbook results, mathematical identities
//! and well known properties of the classic models of network science.
//! Random tests are seeded (the RNG is per thread) and assert robust
//! properties rather than exact random outputs.

mod common;

use common::assert_close;
use igraph::attributes;
use igraph::community::{LeidenObjective, LeidenOptions, compare_communities};
use igraph::games::AllowedEdgeTypes;
use igraph::layout::{FruchtermanReingoldOptions, KamadaKawaiOptions};
use igraph::linalg::{
    ArpackOptions, ArpackWhich, EigenAlgorithm, EigenWhich, arpack_rssolve, eigen_matrix_symmetric,
};
use igraph::misc::solve_lsap;
use igraph::prelude::*;
use igraph::structural::LaplacianNormalization;

/// The officer's faction of Zachary's karate club (0-based ids), as assigned
/// by Zachary from the factional alignment of the members before the split;
/// everybody else sided with the instructor, "Mr. Hi" (vertex 0). Member 8
/// (9 in Zachary's 1-based numbering) was a weak supporter of the officers
/// but joined Mr. Hi's new club, so the clubs *after* the split differ from
/// these factions in that single member.
const OFFICER: [i64; 18] = [
    8, 9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
];

/// The two-faction split as a membership vector (1 = officer's faction).
fn karate_factions() -> Vec<i64> {
    (0..34).map(|v| i64::from(OFFICER.contains(&v))).collect()
}

/// The indices of the `k` largest values (ties broken by index).
fn top_k(values: &[f64], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
    idx.truncate(k);
    idx.sort_unstable();
    idx
}

/// Renders a 2D layout as a small standalone SVG picture, one `<circle>` per
/// vertex (colored by community) and one `<line>` per edge.
fn layout_to_svg(g: &Graph, layout: &Matrix, membership: &[i64], size: f64) -> String {
    const PALETTE: [&str; 6] = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    ];
    let margin = 20.0;
    let bounds = |k: usize| {
        layout
            .column(k)
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
                (lo.min(x), hi.max(x))
            })
    };
    let ((x0, x1), (y0, y1)) = (bounds(0), bounds(1));
    let scale = (size - 2.0 * margin) / (x1 - x0).max(y1 - y0).max(f64::EPSILON);
    let pos = |v: i64| {
        let v = v as usize;
        (
            margin + (layout[(v, 0)] - x0) * scale,
            margin + (layout[(v, 1)] - y0) * scale,
        )
    };
    let mut svg =
        format!(r#"<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">"#);
    svg.push('\n');
    for (u, v) in g.edge_list() {
        let ((ax, ay), (bx, by)) = (pos(u), pos(v));
        svg.push_str(&format!(
            "  <line x1=\"{ax:.1}\" y1=\"{ay:.1}\" x2=\"{bx:.1}\" y2=\"{by:.1}\" stroke=\"#999\"/>\n"
        ));
    }
    for v in g.vertices() {
        let (x, y) = pos(v);
        let fill = PALETTE[membership[v as usize] as usize % PALETTE.len()];
        svg.push_str(&format!(
            "  <circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"6\" fill=\"{fill}\"><title>{v}</title></circle>\n"
        ));
    }
    svg.push_str("</svg>\n");
    svg
}

// ---------------------------------------------------------------------------
// (a) Zachary's karate club, end to end.
// ---------------------------------------------------------------------------

/// Famous graph → centralities → communities → comparison with the
/// two factions → layout → SVG.
#[test]
fn karate_club_end_to_end() {
    // 1. The graph ships with igraph: 34 members, 78 friendships, and it is
    //    the very same graph as the hand-written edge list of `common`.
    let g = Graph::famous("Zachary").unwrap();
    assert_eq!((g.vcount(), g.ecount()), (34, 78));
    assert!(g.is_same_graph(&common::karate()).unwrap());

    // 2. Centralities: whatever the notion, the two leaders of the conflict
    //    (the instructor 0 and the president 33) are the two most central
    //    members of the club.
    let degree: Vec<f64> = g
        .degree(VertexSelector::All, NeighborMode::All, Loops::Twice)
        .unwrap()
        .into_iter()
        .map(|d| d as f64)
        .collect();
    assert_eq!((degree[0], degree[33]), (16.0, 17.0));
    let betweenness = g
        .betweenness(None, VertexSelector::All, false, false)
        .unwrap();
    // Known values: 231.0714 for the instructor, 160.5516 for the president.
    assert_close(betweenness[0], 231.071_428_571, 1e-6);
    assert_close(betweenness[33], 160.551_587_302, 1e-6);
    let closeness = g
        .closeness(VertexSelector::All, NeighborMode::All, None, true)
        .unwrap();
    let pagerank = g
        .pagerank(
            None,
            VertexSelector::All,
            &igraph::centrality::PageRankOptions::default(),
        )
        .unwrap();
    assert_close(pagerank.scores.iter().sum::<f64>(), 1.0, 1e-9);
    for scores in [&degree, &betweenness, &pagerank.scores] {
        assert_eq!(top_k(scores, 2), vec![0, 33]);
    }
    // Closeness puts the instructor first (33/58 = 0.5690), then the
    // "broker" 2 (0.5593), then the president (0.55).
    assert_eq!(top_k(&closeness, 3), vec![0, 2, 33]);
    assert_close(closeness[0], 33.0 / 58.0, 1e-12);
    assert_close(closeness[33], 0.55, 1e-12);

    // 3. Communities. The partition of maximum modularity (Q = 0.4198) has
    //    four groups and *refines* the factional split: every community found
    //    lies entirely inside one of the two factions.
    let factions = karate_factions();
    rng::seed(42).unwrap();
    let leiden = g
        .community_leiden_simple(
            None,
            LeidenObjective::Modularity,
            &LeidenOptions::default().with_iterations(None),
        )
        .unwrap();
    assert_eq!(leiden.nb_clusters, 4);
    assert_close(leiden.quality, 0.419_789_612, 1e-6);
    for community in leiden.communities() {
        let side = factions[community[0] as usize];
        assert!(community.iter().all(|&v| factions[v as usize] == side));
    }
    // Louvain is a greedy heuristic: depending on the random vertex order
    // it ends anywhere between Q = 0.395 and the optimum.
    let multilevel = g.community_multilevel(None, 1.0).unwrap();
    assert!(multilevel.modularity() > 0.39 && multilevel.modularity() <= leiden.quality + 1e-12);
    // Modularity reported by the algorithms agrees with a recomputation.
    assert_close(
        g.modularity(&leiden.membership, None, 1.0, false).unwrap(),
        leiden.quality,
        1e-9,
    );
    // Normalized mutual information with the factions: well above chance
    // (a refinement of a 2-split into 4 groups cannot reach 1).
    let nmi_leiden =
        compare_communities(&leiden.membership, &factions, CommunityComparison::Nmi).unwrap();
    let nmi_louvain =
        compare_communities(&multilevel.membership, &factions, CommunityComparison::Nmi).unwrap();
    assert!(nmi_leiden > 0.55 && nmi_leiden < 1.0, "NMI {nmi_leiden}");
    assert!(nmi_louvain > 0.45, "NMI {nmi_louvain}");
    // NMI of a partition with itself is 1, with the trivial partition 0.
    assert_close(
        compare_communities(&factions, &factions, CommunityComparison::Nmi).unwrap(),
        1.0,
        1e-12,
    );
    // The factional split is a good, but not optimal, partition.
    let q_real = g.modularity(&factions, None, 1.0, false).unwrap();
    assert_close(q_real, 0.371_466_140, 1e-6);
    assert!(q_real < leiden.quality);

    // 4. Layout: Kamada-Kawai is deterministic; refine it with a seeded
    //    Fruchterman-Reingold run.
    let kk = g
        .layout_kamada_kawai(&KamadaKawaiOptions::default())
        .unwrap();
    let mut layout = g
        .layout_fruchterman_reingold(&FruchtermanReingoldOptions::default().with_initial(&kk))
        .unwrap();
    g.layout_align(&mut layout).unwrap();
    assert_eq!((layout.nrow(), layout.ncol()), (34, 2));
    assert!(layout.as_slice().iter().all(|x| x.is_finite()));
    // Force-directed layouts pull friends together: adjacent members are on
    // average closer than random pairs of members.
    let dist = |u: usize, v: usize| {
        ((layout[(u, 0)] - layout[(v, 0)]).powi(2) + (layout[(u, 1)] - layout[(v, 1)]).powi(2))
            .sqrt()
    };
    let edge_mean = g
        .edge_list()
        .iter()
        .map(|&(u, v)| dist(u as usize, v as usize))
        .sum::<f64>()
        / 78.0;
    let pair_mean = (0..34)
        .flat_map(|u| (u + 1..34).map(move |v| (u, v)))
        .map(|(u, v)| dist(u, v))
        .sum::<f64>()
        / (34.0 * 33.0 / 2.0);
    assert!(edge_mean < 0.6 * pair_mean);

    // 5. SVG output: one line per friendship, one circle per member, all of
    //    them inside the canvas.
    let svg = layout_to_svg(&g, &layout, &leiden.membership, 400.0);
    assert!(svg.starts_with("<svg") && svg.trim_end().ends_with("</svg>"));
    assert_eq!(svg.matches("<line").count(), 78);
    assert_eq!(svg.matches("<circle").count(), 34);
    assert_eq!(svg.matches("<title>33</title>").count(), 1);
    for coord in svg.split('"').filter_map(|s| s.parse::<f64>().ok()) {
        assert!((0.0..=400.0).contains(&coord));
    }
}

// ---------------------------------------------------------------------------
// (b) A city road network.
// ---------------------------------------------------------------------------

/// A `w × h` grid of crossings: vertex `r * w + c` is the crossing at row `r`,
/// column `c`; the roads are the grid edges (horizontal ones first).
fn grid_city(w: i64, h: i64) -> Graph {
    let mut edges = Vec::new();
    for r in 0..h {
        for c in 0..w - 1 {
            edges.push((r * w + c, r * w + c + 1));
        }
    }
    for r in 0..h - 1 {
        for c in 0..w {
            edges.push((r * w + c, (r + 1) * w + c));
        }
    }
    Graph::from_edges(&edges, (w * h) as usize, false).unwrap()
}

/// Weighted grid with seeded random travel times → Dijkstra routes; the same
/// grid with random road capacities → max flow and min cut between the west
/// and the east districts, with max-flow = min-cut.
#[test]
fn city_road_network_routes_and_capacity() {
    let (w, h) = (8, 6);
    let city = grid_city(w, h);
    assert_eq!(city.ecount() as i64, (w - 1) * h + w * (h - 1));
    rng::seed(2024).unwrap();
    let minutes: Vec<f64> = (0..city.ecount())
        .map(|_| rng::integer(1, 9) as f64)
        .collect();

    // Routes from the north-west corner to every crossing.
    let (home, office) = (0, w * h - 1);
    let dist = city
        .distances_dijkstra(home, VertexSelector::All, Some(&minutes), NeighborMode::All)
        .unwrap();
    assert_eq!((dist.nrow(), dist.ncol()), (1, (w * h) as usize));
    assert_eq!(dist[(0, 0)], 0.0);
    let route = city
        .get_shortest_path_dijkstra(home, office, Some(&minutes), NeighborMode::All)
        .unwrap();
    // The route is a real walk on the grid whose travel time is the distance.
    assert_eq!(route.vertices.first(), Some(&home));
    assert_eq!(route.vertices.last(), Some(&office));
    assert_eq!(route.edges.len() + 1, route.vertices.len());
    let travel: f64 = route.edges.iter().map(|&e| minutes[e as usize]).sum();
    assert_close(travel, dist[(0, office as usize)], 1e-9);
    for (k, &e) in route.edges.iter().enumerate() {
        let (a, b) = city.edge(e).unwrap();
        let (u, v) = (route.vertices[k], route.vertices[k + 1]);
        assert!((a, b) == (u, v) || (a, b) == (v, u));
    }
    // A shortest route needs at least the Manhattan number of blocks, and
    // is never slower than the "go east, then go south" route.
    assert!(route.edges.len() as i64 >= (w - 1) + (h - 1));
    let naive: f64 = (0..w - 1)
        .map(|c| minutes[city.get_eid(c, c + 1, false).unwrap().unwrap() as usize])
        .chain((0..h - 1).map(|r| {
            let (u, v) = (r * w + w - 1, (r + 1) * w + w - 1);
            minutes[city.get_eid(u, v, false).unwrap().unwrap() as usize]
        }))
        .sum();
    assert!(travel <= naive);
    // Bellman's optimality: every road satisfies the triangle inequality.
    for (e, (u, v)) in city.edge_list().into_iter().enumerate() {
        let (du, dv) = (dist[(0, u as usize)], dist[(0, v as usize)]);
        assert!((du - dv).abs() <= minutes[e] + 1e-9);
    }

    // Capacity between districts: a super source feeds the west column, a
    // super sink drains the east column (vertices w*h and w*h+1), through
    // roads of (practically) unlimited capacity.
    let capacity: Vec<f64> = (0..city.ecount())
        .map(|_| rng::integer(1, 5) as f64 * 100.0)
        .collect();
    let (src, dst) = (w * h, w * h + 1);
    let mut net = city.clone();
    net.add_vertices(2).unwrap();
    let mut caps = capacity.clone();
    for r in 0..h {
        net.add_edges(&[(src, r * w), (r * w + w - 1, dst)])
            .unwrap();
        caps.extend([1e6, 1e6]);
    }
    let flow = net.maxflow(src, dst, Some(&caps)).unwrap();
    let cut = net.st_mincut(src, dst, Some(&caps)).unwrap();
    // The max-flow min-cut theorem, checked three ways.
    assert_close(flow.value, cut.value, 1e-9);
    let cut_capacity: f64 = cut.cut.iter().map(|&e| caps[e as usize]).sum();
    assert_close(cut_capacity, cut.value, 1e-9);
    assert_close(
        net.maxflow_value(src, dst, Some(&caps)).unwrap(),
        flow.value,
        1e-9,
    );
    // The cut edges are exactly the edges crossing the two sides.
    let mut side = vec![false; net.vcount()];
    for &v in &cut.partition {
        side[v as usize] = true;
    }
    assert!(side[src as usize] && !side[dst as usize]);
    let mut crossing: Vec<i64> = net
        .edge_list()
        .into_iter()
        .enumerate()
        .filter(|&(_, (u, v))| side[u as usize] != side[v as usize])
        .map(|(e, _)| e as i64)
        .collect();
    let mut cut_edges = cut.cut.clone();
    crossing.sort_unstable();
    cut_edges.sort_unstable();
    assert_eq!(crossing, cut_edges);
    // Any "north-south" line between two adjacent columns is a cut, so it
    // bounds the flow from above.
    let column_cut = |c: i64| -> f64 {
        (0..h)
            .map(|r| {
                capacity[city
                    .get_eid(r * w + c, r * w + c + 1, false)
                    .unwrap()
                    .unwrap() as usize]
            })
            .sum()
    };
    let best_line = (0..w - 1).map(column_cut).fold(f64::INFINITY, f64::min);
    assert!(flow.value <= best_line + 1e-9);
    // The flow respects capacities and is conserved at every crossing
    // (positive flow on an undirected edge goes from the smaller id).
    let mut balance = vec![0.0; net.vcount()];
    for (e, (u, v)) in net.edge_list().into_iter().enumerate() {
        let f = flow.flow[e];
        assert!(f.abs() <= caps[e] + 1e-9);
        let (lo, hi) = (u.min(v) as usize, u.max(v) as usize);
        balance[lo] -= f;
        balance[hi] += f;
    }
    for (v, b) in balance.iter().enumerate() {
        let expected = if v as i64 == src {
            -flow.value
        } else if v as i64 == dst {
            flow.value
        } else {
            0.0
        };
        assert_close(*b, expected, 1e-6);
    }
}

// ---------------------------------------------------------------------------
// (c) The Erdős–Rényi phase transition.
// ---------------------------------------------------------------------------

/// Fraction of the vertices in the largest connected component.
fn giant_fraction(g: &Graph) -> f64 {
    let cc = g.connected_components(Connectedness::Weak).unwrap();
    *cc.sizes.iter().max().unwrap() as f64 / g.vcount() as f64
}

/// Below mean degree 1 all components are tiny (O(log n)); above it a giant
/// component appears, holding the fraction `S` solving `S = 1 - exp(-c S)`.
#[test]
fn erdos_renyi_giant_component_phase_transition() {
    let n = 4000;
    // Theory: fixed point of S = 1 - exp(-c S), found by iteration.
    let giant_theory = |c: f64| {
        let mut s: f64 = 1.0;
        for _ in 0..1000 {
            s = 1.0 - (-c * s).exp();
        }
        s
    };
    assert_close(giant_theory(2.0), 0.796_812_130, 1e-6);
    assert!(giant_theory(0.5) < 1e-12);

    for seed in [1, 2, 3, 4, 5] {
        rng::seed(seed).unwrap();
        for c in [0.5, 2.0, 3.0] {
            let p = c / (n as f64 - 1.0);
            let g = Graph::erdos_renyi_game_gnp(n, p, false, EdgeTypeSw::Simple, false).unwrap();
            let mean_degree = 2.0 * g.ecount() as f64 / n as f64;
            assert!((mean_degree - c).abs() < 0.15, "mean degree {mean_degree}");
            let s = giant_fraction(&g);
            if c < 1.0 {
                // Subcritical: the largest component has O(log n) vertices.
                assert!(s < 0.01, "seed {seed}, c {c}: {s}");
            } else {
                // Compare with the theory at the realized mean degree.
                let expected = giant_theory(mean_degree);
                assert!(
                    (s - expected).abs() < 0.03,
                    "seed {seed}, c {c}: {s} vs {expected}"
                );
            }
        }
    }

    // The same story with G(n, m): the giant grows monotonically with the
    // number of edges, and is unique (the second component is tiny).
    rng::seed(99).unwrap();
    let mut last = 0.0;
    for c in [0.25, 0.5, 1.5, 2.0, 4.0] {
        let m = (c * n as f64 / 2.0) as usize;
        let g = Graph::erdos_renyi_game_gnm(n, m, false, AllowedEdgeTypes::SIMPLE, false).unwrap();
        let cc = g.connected_components(Connectedness::Weak).unwrap();
        let mut sizes = cc.sizes.clone();
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        let s = sizes[0] as f64 / n as f64;
        assert!(s >= last);
        last = s;
        if c > 1.0 {
            assert!(sizes[1] < 60, "second largest component: {}", sizes[1]);
        }
    }
    assert!(last > 0.97);
}

// ---------------------------------------------------------------------------
// (d) Motifs, triads and isomorphism classes.
// ---------------------------------------------------------------------------

/// Triad census and size-3 motif counts of a directed random graph tell the
/// same story, linked through isomorphism classes.
#[test]
fn directed_motifs_and_triad_census() {
    rng::seed(7).unwrap();
    let n = 60usize;
    let g = Graph::erdos_renyi_game_gnp(n, 0.08, true, EdgeTypeSw::Simple, false).unwrap();
    assert!(g.is_directed());

    // The triad census partitions all the C(n, 3) vertex triples.
    let census = g.triad_census().unwrap();
    let triples = (n * (n - 1) * (n - 2) / 6) as f64;
    assert_close(census.counts.iter().sum::<f64>(), triples, 1e-6);

    // Map each of the 16 directed isomorphism classes on three vertices to
    // its MAN label: the census of a 3-vertex graph is a single 1.
    let class_to_triad: Vec<usize> = (0..16)
        .map(|class| {
            let t = Graph::isoclass_create(3, class, true).unwrap();
            // Round trip: the class of the canonical graph is the class.
            assert_eq!(t.isoclass().unwrap(), class);
            let c = t.triad_census().unwrap();
            assert_eq!(c.counts.iter().sum::<f64>(), 1.0);
            c.counts.iter().position(|&x| x == 1.0).unwrap()
        })
        .collect();
    // It is a bijection between classes and labels.
    let mut labels = class_to_triad.clone();
    labels.sort_unstable();
    assert_eq!(labels, (0..16).collect::<Vec<_>>());

    // Motifs count the weakly connected induced subgraphs: they agree with
    // the census on the connected classes and are NaN on the others.
    let motifs = g.motifs_randesu(3, None).unwrap();
    assert_eq!(motifs.len(), 16);
    let disconnected = ["003", "012", "102"];
    let mut connected_total = 0.0;
    for (class, &count) in motifs.iter().enumerate() {
        let label = igraph::isomorphism::TriadCensus::NAMES[class_to_triad[class]];
        if disconnected.contains(&label) {
            assert!(count.is_nan(), "class {class} ({label})");
        } else {
            assert_eq!(count, census.counts[class_to_triad[class]], "{label}");
            connected_total += count;
        }
    }
    assert_eq!(g.motifs_randesu_no(3, None).unwrap(), connected_total);

    // Isomorphism is blind to relabelling: a random permutation of the
    // vertices gives an isomorphic graph with the very same census.
    let mut perm: Vec<i64> = (0..n as i64).collect();
    rng::shuffle(&mut perm);
    let h = g.permute_vertices(&perm).unwrap();
    assert!(g.isomorphic(&h).unwrap());
    assert_eq!(h.triad_census().unwrap(), census);
    // ... while adding a single arc breaks it.
    let mut k = h.clone();
    let (u, v) = (0..n as i64)
        .flat_map(|u| (0..n as i64).map(move |v| (u, v)))
        .find(|&(u, v)| u != v && h.get_eid(u, v, true).unwrap().is_none())
        .unwrap();
    k.add_edges(&[(u, v)]).unwrap();
    assert!(!g.isomorphic(&k).unwrap());

    // A transitive triad (030T) and a directed 3-cycle (030C) have the same
    // number of arcs but are not isomorphic.
    let t = Graph::from_edges(&[(0, 1), (1, 2), (0, 2)], 3, true).unwrap();
    let c = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, true).unwrap();
    assert!(!t.isomorphic(&c).unwrap());
    assert_eq!(t.triad_census().unwrap()["030T"], 1.0);
    assert_eq!(c.triad_census().unwrap()["030C"], 1.0);
}

// ---------------------------------------------------------------------------
// (e) Job assignment: weighted bipartite matching vs the Hungarian method.
// ---------------------------------------------------------------------------

/// Workers `0..n` and jobs `n..2n` with integer profits: the maximum weight
/// bipartite matching and the linear sum assignment on the complementary
/// costs pick equally profitable, complete assignments; brute force agrees.
#[test]
fn job_assignment_matching_vs_hungarian() {
    let n = 6usize;
    for seed in [1, 2, 3] {
        rng::seed(seed).unwrap();
        let profit: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..n).map(|_| rng::integer(1, 30) as f64).collect())
            .collect();

        // The complete bipartite "who can do what" graph.
        let k = Graph::full_bipartite(n, n, false, NeighborMode::All).unwrap();
        let weights: Vec<f64> = k
            .graph
            .edge_list()
            .into_iter()
            .map(|(u, v)| {
                let (worker, job) = (u.min(v) as usize, u.max(v) as usize - n);
                profit[worker][job]
            })
            .collect();
        let matching = k
            .graph
            .maximum_bipartite_matching_eps(&k.types, Some(&weights), 0.0)
            .unwrap();
        assert_eq!(matching.size, n); // positive profits: everybody works
        let pairs = matching.pairs();
        assert_eq!(pairs.len(), n);
        let matched_profit: f64 = pairs
            .iter()
            .map(|&(w, j)| profit[w as usize][j as usize - n])
            .sum();
        assert_eq!(matched_profit, matching.weight);

        // The Hungarian method minimizes a cost: use (max profit - profit).
        let max = profit.iter().flatten().cloned().fold(0.0, f64::max);
        let cost = Matrix::from_rows(
            &profit
                .iter()
                .map(|row| row.iter().map(|p| max - p).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let assignment = solve_lsap(&cost).unwrap();
        let mut jobs = assignment.clone();
        jobs.sort_unstable();
        assert_eq!(jobs, (0..n as i64).collect::<Vec<_>>());
        let hungarian_profit: f64 = assignment
            .iter()
            .enumerate()
            .map(|(w, &j)| profit[w][j as usize])
            .sum();
        assert_eq!(hungarian_profit, matched_profit);

        // Brute force over all 720 permutations.
        let mut perm: Vec<usize> = (0..n).collect();
        let mut best = 0.0f64;
        permutations(&mut perm, 0, &mut |p| {
            best = best.max(p.iter().enumerate().map(|(w, &j)| profit[w][j]).sum());
        });
        assert_eq!(best, matched_profit);
    }
}

/// Visits all the permutations of `a[k..]` (recursive swapping: `a[k]` takes
/// each value in turn, then the rest is permuted), restoring `a` afterwards.
fn permutations(a: &mut [usize], k: usize, f: &mut impl FnMut(&[usize])) {
    if k == a.len() {
        f(a);
        return;
    }
    for i in k..a.len() {
        a.swap(k, i);
        permutations(a, k + 1, f);
        a.swap(k, i);
    }
}

// ---------------------------------------------------------------------------
// (f) GraphML round trip with attributes.
// ---------------------------------------------------------------------------

/// A named, weighted network written to GraphML and read back: names,
/// weights and graph attributes survive, and weighted analyses give the
/// same answers on the rebuilt graph.
#[test]
fn graphml_round_trip_with_attributes() {
    attributes::enable().unwrap();
    let mut g = Graph::famous("Zachary").unwrap();
    let names: Vec<String> = (0..34).map(|v| format!("member-{v:02}")).collect();
    g.set_vertex_attr_str_values("name", &names).unwrap();
    // A friendship is stronger when the two friends share many friends.
    let weights: Vec<f64> = g
        .edge_list()
        .into_iter()
        .map(|(u, v)| {
            let nu = g.neighbors(u, NeighborMode::All).unwrap();
            let nv = g.neighbors(v, NeighborMode::All).unwrap();
            1.0 + nu.iter().filter(|x| nv.contains(x)).count() as f64
        })
        .collect();
    g.set_edge_attr_numeric_values("weight", &weights).unwrap();
    g.set_graph_attr_str("title", "Zachary's karate club")
        .unwrap();
    g.set_vertex_attr_bool("leader", 0, true).unwrap();
    g.set_vertex_attr_bool("leader", 33, true).unwrap();

    let xml = g.write_graph_graphml_to_string(false).unwrap();
    assert!(xml.contains("<graphml"));
    assert!(xml.contains("member-33"));
    assert!(xml.contains(r#"attr.name="weight""#));
    assert_eq!(xml.matches("<node ").count(), 34);
    assert_eq!(xml.matches("<edge ").count(), 78);

    let h = Graph::read_graph_graphml_from_str(&xml, 0).unwrap();
    assert!(!h.is_directed());
    assert_eq!(h.edge_list(), g.edge_list());
    assert_eq!(h.vertex_attr_str_values("name", ..).unwrap(), names);
    assert_eq!(h.edge_attr_numeric_values("weight", ..).unwrap(), weights);
    assert_eq!(h.graph_attr_str("title").unwrap(), "Zachary's karate club");
    let leaders: Vec<i64> = h
        .vertex_attr_bool_values("leader", ..)
        .unwrap()
        .into_iter()
        .enumerate()
        .filter(|&(_, l)| l)
        .map(|(v, _)| v as i64)
        .collect();
    assert_eq!(leaders, vec![0, 33]);

    // Rebuild a plain graph from the file content only, keyed by name, and
    // check that weighted analyses are unchanged.
    let read_names = h.vertex_attr_str_values("name", ..).unwrap();
    let id = |name: &str| read_names.iter().position(|n| n == name).unwrap() as i64;
    let rebuilt_edges: Vec<(i64, i64)> = h
        .edge_list()
        .into_iter()
        .map(|(u, v)| (id(&read_names[u as usize]), id(&read_names[v as usize])))
        .collect();
    let rebuilt = Graph::from_edges(&rebuilt_edges, 34, false).unwrap();
    let w2 = h.edge_attr_numeric_values("weight", ..).unwrap();
    let d1 = g
        .distances_dijkstra(
            VertexSelector::All,
            VertexSelector::All,
            Some(&weights),
            NeighborMode::All,
        )
        .unwrap();
    let d2 = rebuilt
        .distances_dijkstra(
            VertexSelector::All,
            VertexSelector::All,
            Some(&w2),
            NeighborMode::All,
        )
        .unwrap();
    assert_eq!(d1.to_rows(), d2.to_rows());
    // Louvain visits the vertices in random order: with the same seed, the
    // same graph and the same weights it must give the very same answer.
    rng::seed(1).unwrap();
    let q1 = g.community_multilevel(Some(&weights), 1.0).unwrap();
    rng::seed(1).unwrap();
    let q2 = rebuilt.community_multilevel(Some(&w2), 1.0).unwrap();
    assert_eq!(q1.membership, q2.membership);
    assert_close(q1.modularity(), q2.modularity(), 1e-12);
    // And the weighted modularity of a fixed partition is the same too.
    assert_close(
        g.modularity(&q1.membership, Some(&weights), 1.0, false)
            .unwrap(),
        rebuilt
            .modularity(&q1.membership, Some(&w2), 1.0, false)
            .unwrap(),
        1e-12,
    );

    // The reader also keeps the GraphML node ids, as an extra vertex
    // attribute "id"; without it, a second round trip is a fixed point.
    let ids: Vec<String> = (0..34).map(|v| format!("n{v}")).collect();
    assert_eq!(h.vertex_attr_str_values("id", ..).unwrap(), ids);
    let mut h = h;
    h.remove_vertex_attr("id");
    assert_eq!(h.write_graph_graphml_to_string(false).unwrap(), xml);
}

// ---------------------------------------------------------------------------
// (g) Small worlds.
// ---------------------------------------------------------------------------

/// Watts–Strogatz: a little rewiring collapses the path length while the
/// clustering stays high; a lot of rewiring destroys clustering too.
#[test]
fn watts_strogatz_small_world() {
    let (n, nei) = (1000usize, 5usize); // ring of 1000, each linked to 10 neighbours
    let measure = |p: f64| {
        let mut c = 0.0;
        let mut l = 0.0;
        let runs = 3;
        for _ in 0..runs {
            let g = Graph::watts_strogatz_game(1, n, nei, p, EdgeTypeSw::Simple).unwrap();
            assert_eq!(g.ecount(), n * nei);
            c += g.transitivity_undirected(TransitivityMode::Zero).unwrap();
            l += g.average_path_length(None, false, true).unwrap();
        }
        (c / runs as f64, l / runs as f64)
    };
    rng::seed(11).unwrap();
    let (c0, l0) = measure(0.0);
    // Regular ring lattice with k = 2 nei: C = 3(k-2) / (4(k-1)), and a
    // path length of about n / (2k).
    let k = 2.0 * nei as f64;
    assert_close(c0, 3.0 * (k - 2.0) / (4.0 * (k - 1.0)), 1e-12);
    assert!(l0 > 40.0 && l0 < 60.0, "L(0) = {l0}");

    let ps = [0.001, 0.01, 0.1, 1.0];
    let results: Vec<(f64, f64)> = ps.iter().map(|&p| measure(p)).collect();
    // Both decrease as the rewiring probability grows.
    for pair in results.windows(2) {
        assert!(pair[1].0 < pair[0].0 && pair[1].1 < pair[0].1);
    }
    // The small-world regime (p = 0.01): clustering almost intact, paths
    // already short.
    let (c_sw, l_sw) = results[1];
    assert!(c_sw / c0 > 0.9, "C(0.01)/C(0) = {}", c_sw / c0);
    assert!(l_sw / l0 < 0.3, "L(0.01)/L(0) = {}", l_sw / l0);
    // Fully random (p = 1): like a random graph with the same density,
    // C ≈ k / n and L ≈ ln n / ln k.
    let (c_rand, l_rand) = results[3];
    assert!(c_rand < 5.0 * k / n as f64, "C(1) = {c_rand}");
    let l_theory = (n as f64).ln() / k.ln();
    assert!(
        (l_rand - l_theory).abs() < 0.6,
        "L(1) = {l_rand} vs {l_theory}"
    );
}

// ---------------------------------------------------------------------------
// (h) Epidemics on scale-free vs random networks.
// ---------------------------------------------------------------------------

/// SIR epidemics: with the same number of people and contacts, a hub-rich
/// scale-free network (Barabási–Albert) lets a mild disease spread much
/// further than a homogeneous random network (Erdős–Rényi).
#[test]
fn sir_epidemic_scale_free_vs_random() {
    rng::seed(3).unwrap();
    let n = 2000;
    let ba = Graph::barabasi_game(n, &igraph::games::BarabasiOptions::default().with_m(2)).unwrap();
    let er = Graph::erdos_renyi_game_gnm(n, ba.ecount(), false, EdgeTypeSw::Simple, false).unwrap();
    assert_eq!(ba.ecount(), er.ecount());
    // Same mean degree (about 4), very different second moments.
    let second_moment = |g: &Graph| {
        let d = g
            .degree(VertexSelector::All, NeighborMode::All, Loops::Twice)
            .unwrap();
        d.iter().map(|&x| (x * x) as f64).sum::<f64>() / d.len() as f64
    };
    let (k2_ba, k2_er) = (second_moment(&ba), second_moment(&er));
    assert!(k2_ba > 1.5 * k2_er, "<k^2>: BA {k2_ba}, ER {k2_er}");

    // Transmissibility T = beta / (beta + gamma) = 0.2: below the ER
    // threshold <k> / (<k^2> - <k>) = 1/4, above the BA one.
    let (beta, gamma, sims) = (0.25, 1.0, 200);
    let mean_final_size = |g: &Graph| {
        let runs = g.sir(beta, gamma, sims).unwrap();
        assert_eq!(runs.len(), sims);
        let mut total = 0.0;
        for run in &runs {
            // Start: one infected; end: nobody infected; people conserved.
            assert_eq!(run.infected[0], 1);
            assert_eq!(*run.infected.last().unwrap(), 0);
            assert!(run.times.windows(2).all(|t| t[0] <= t[1]));
            for i in 0..run.times.len() {
                assert_eq!(
                    run.susceptible[i] + run.infected[i] + run.recovered[i],
                    n as i64
                );
            }
            // Recovered people never become susceptible again.
            assert!(run.recovered.windows(2).all(|r| r[0] <= r[1]));
            total += *run.recovered.last().unwrap() as f64;
        }
        total / sims as f64 / n as f64
    };
    let (ba_size, er_size) = (mean_final_size(&ba), mean_final_size(&er));
    assert!(ba_size > 3.0 * er_size, "BA {ba_size} vs ER {er_size}");
    assert!(er_size < 0.05, "ER outbreaks stay small: {er_size}");
}

// ---------------------------------------------------------------------------
// (i) Spectral graph theory: counting components with the Laplacian.
// ---------------------------------------------------------------------------

/// The multiplicity of the eigenvalue 0 of the Laplacian `L = D - A` is the
/// number of connected components; computed with dense LAPACK and with
/// matrix-free ARPACK.
#[test]
fn laplacian_spectrum_counts_components() {
    // Karate club + a 10-cycle + a 5-path + a triangle + an isolated vertex:
    // five components, 34 + 10 + 5 + 3 + 1 = 53 vertices.
    let pieces = [
        Graph::famous("Zachary").unwrap(),
        common::cycle(10),
        common::path(5),
        common::complete(3),
        Graph::from_edges(&[], 1, false).unwrap(),
    ];
    let g = Graph::disjoint_union_many(pieces.iter()).unwrap();
    let n = g.vcount();
    assert_eq!(n, 53);
    let components = g.connected_components(Connectedness::Weak).unwrap().count;
    assert_eq!(components, 5);

    let lap = g
        .get_laplacian(
            NeighborMode::All,
            LaplacianNormalization::Unnormalized,
            None,
        )
        .unwrap();
    // Trace of L = sum of degrees = 2|E|.
    let trace: f64 = (0..n).map(|i| lap[(i, i)]).sum();
    assert_eq!(trace, 2.0 * g.ecount() as f64);

    // Dense LAPACK: the whole spectrum.
    let dense = eigen_matrix_symmetric(
        &lap,
        &EigenWhich::All,
        EigenAlgorithm::Lapack,
        &ArpackOptions::default(),
    )
    .unwrap();
    let mut spectrum = dense.values.clone();
    spectrum.sort_by(f64::total_cmp);
    assert_eq!(spectrum.len(), n);
    assert!(spectrum[0] > -1e-9); // L is positive semidefinite
    let zeros = spectrum.iter().filter(|x| x.abs() < 1e-9).count();
    assert_eq!(zeros, components);
    assert_close(spectrum.iter().sum::<f64>(), trace, 1e-8);
    // The first non-zero eigenvalue is the smallest algebraic connectivity
    // among the pieces: the 5-path, 2 - 2 cos(pi / 5).
    assert_close(
        spectrum[components],
        2.0 - 2.0 * (std::f64::consts::PI / 5.0).cos(),
        1e-9,
    );
    // The largest eigenvalue is at most twice the maximum degree.
    assert!(spectrum[n - 1] <= 2.0 * 17.0);

    // Matrix-free ARPACK: the product L x is computed from the adjacency
    // lists, without ever forming L. Lanczos methods are notoriously bad at
    // resolving *repeated* eigenvalues, so the zeros are peeled off one at a
    // time with Hotelling deflation: each eigenvector `u` found is shifted
    // away by adding `sigma u u^T` to the operator, with `sigma` above the
    // largest eigenvalue (at most twice the maximum degree).
    //
    // One more subtlety: ARPACK first multiplies its (random) starting
    // vector by the operator, "forcing it into the range of OP". For a
    // singular operator this wipes out the null space components, and an
    // *exact* null vector such as the indicator of an isolated vertex is
    // then never found (the others only reappear through rounding errors).
    // So we work with the non-singular `L + I` and subtract 1 at the end.
    let adjacency: Vec<Vec<i64>> = g
        .vertices()
        .map(|v| g.neighbors(v, NeighborMode::All).unwrap())
        .collect();
    let sigma = 2.0 * 17.0 + 1.0;
    let mut null_space: Vec<Vec<f64>> = Vec::new();
    rng::seed(5).unwrap();
    let opts = ArpackOptions::default()
        .with_nev(1)
        .with_which(ArpackWhich::SmallestAlgebraic)
        .with_tol(1e-12)
        .with_mxiter(10_000);
    let algebraic_connectivity = loop {
        let deflated = |x: &[f64], y: &mut [f64]| {
            for (v, nei) in adjacency.iter().enumerate() {
                y[v] =
                    (nei.len() + 1) as f64 * x[v] - nei.iter().map(|&u| x[u as usize]).sum::<f64>();
            }
            for u in &null_space {
                let dot: f64 = u.iter().zip(x).map(|(a, b)| a * b).sum();
                for (yi, ui) in y.iter_mut().zip(u) {
                    *yi += sigma * dot * ui;
                }
            }
        };
        let res = arpack_rssolve(n, deflated, &opts, None).unwrap();
        let lambda = res.values[0] - 1.0;
        if lambda.abs() > 1e-8 {
            break lambda;
        }
        null_space.push(res.vectors.column(0).to_vec());
        assert!(null_space.len() <= n);
    };
    assert_eq!(null_space.len(), components);
    // The first non-zero eigenvalue agrees with LAPACK.
    assert_close(algebraic_connectivity, spectrum[components], 1e-8);
    // The null space is spanned by the component indicator vectors: every
    // eigenvector of 0 is constant on each component (L x = 0 means
    // x_u = x_v along every edge), and they are orthonormal.
    for (i, x) in null_space.iter().enumerate() {
        for (u, v) in g.edge_list() {
            assert_close(x[u as usize], x[v as usize], 1e-6);
        }
        for (j, y) in null_space.iter().enumerate() {
            let dot: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();
            assert_close(dot, if i == j { 1.0 } else { 0.0 }, 1e-6);
        }
    }

    // Adding one bridge between two components removes exactly one zero.
    let mut bridged = g.clone();
    bridged.add_edges(&[(0, 34)]).unwrap();
    let lap2 = bridged
        .get_laplacian(
            NeighborMode::All,
            LaplacianNormalization::Unnormalized,
            None,
        )
        .unwrap();
    let e2 = eigen_matrix_symmetric(
        &lap2,
        &EigenWhich::All,
        EigenAlgorithm::Auto,
        &ArpackOptions::default(),
    )
    .unwrap();
    assert_eq!(
        e2.values.iter().filter(|x| x.abs() < 1e-9).count(),
        components - 1
    );
}

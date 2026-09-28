//! Use case: a complete analysis of Zachary's karate club.
//!
//! ```sh
//! cargo run --example uc_karate_club               # report on stdout
//! cargo run --example uc_karate_club -- club.svg   # also write a picture
//! ```
//!
//! The karate club of Zachary (1977) split in two after a conflict between
//! the instructor ("Mr. Hi", vertex 0) and the president (vertex 33). This
//! example goes from the raw graph to a picture:
//!
//! 1. load the famous graph shipped with igraph;
//! 2. rank the members by four centrality measures;
//! 3. detect communities with Leiden and Louvain (multilevel) and compare
//!    them with the two factions (normalized mutual information);
//! 4. compute a force-directed layout and render it as SVG, coloring the
//!    members by community and drawing the faction as the node outline.

use igraph::centrality::PageRankOptions;
use igraph::community::{LeidenObjective, LeidenOptions, compare_communities};
use igraph::layout::{FruchtermanReingoldOptions, KamadaKawaiOptions};
use igraph::prelude::*;

/// The officer's faction (0-based ids), as assigned by Zachary from the
/// alignment of the members before the split. Member 8 was a weak officer
/// supporter who nevertheless joined Mr. Hi's new club.
const OFFICER: [i64; 18] = [
    8, 9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
];

fn main() -> igraph::Result<()> {
    let g = Graph::famous("Zachary")?;
    println!(
        "Zachary's karate club: {} members, {} friendships",
        g.vcount(),
        g.ecount()
    );
    let factions: Vec<i64> = g
        .vertices()
        .map(|v| i64::from(OFFICER.contains(&v)))
        .collect();

    // --- Centralities -----------------------------------------------------
    let degree: Vec<f64> = g
        .degree(VertexSelector::All, NeighborMode::All, Loops::Twice)?
        .into_iter()
        .map(|d| d as f64)
        .collect();
    let betweenness = g.betweenness(None, VertexSelector::All, false, false)?;
    let closeness = g.closeness(VertexSelector::All, NeighborMode::All, None, true)?;
    let pagerank = g
        .pagerank(None, VertexSelector::All, &PageRankOptions::default())?
        .scores;
    println!("\nMost central members:");
    for (name, scores) in [
        ("degree", &degree),
        ("betweenness", &betweenness),
        ("closeness", &closeness),
        ("pagerank", &pagerank),
    ] {
        let mut idx: Vec<usize> = (0..scores.len()).collect();
        idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        let top: Vec<String> = idx[..3]
            .iter()
            .map(|&v| format!("{v:>2} ({:.3})", scores[v]))
            .collect();
        println!("  {name:<12} {}", top.join("  "));
    }

    // --- Communities ------------------------------------------------------
    rng::seed(42)?;
    let leiden = g.community_leiden_simple(
        None,
        LeidenObjective::Modularity,
        &LeidenOptions::default().with_iterations(None),
    )?;
    let louvain = g.community_multilevel(None, 1.0)?;
    let q_real = g.modularity(&factions, None, 1.0, false)?;
    println!("\nCommunities (modularity Q, NMI with the two factions):");
    println!("  factions     2 groups  Q = {q_real:.4}  NMI = 1.000");
    for (name, membership, q) in [
        ("Leiden", &leiden.membership, leiden.quality),
        ("Louvain", &louvain.membership, louvain.modularity()),
    ] {
        let groups = membership.iter().max().map_or(0, |&m| m + 1);
        let nmi = compare_communities(membership, &factions, CommunityComparison::Nmi)?;
        println!("  {name:<12} {groups} groups  Q = {q:.4}  NMI = {nmi:.3}");
    }
    println!("\nLeiden communities, with the faction of their members:");
    for (c, members) in leiden.communities().iter().enumerate() {
        let officers = members
            .iter()
            .filter(|&&v| factions[v as usize] == 1)
            .count();
        let side = match officers {
            0 => "all in the instructor's faction".to_string(),
            k if k == members.len() => "all in the officer's faction".to_string(),
            k => format!("{k} of {} in the officer's faction", members.len()),
        };
        println!("  #{c}: {members:?} -> {side}");
    }

    // --- Layout and SVG ---------------------------------------------------
    let kk = g.layout_kamada_kawai(&KamadaKawaiOptions::default())?;
    let mut layout =
        g.layout_fruchterman_reingold(&FruchtermanReingoldOptions::default().with_initial(&kk))?;
    g.layout_align(&mut layout)?;
    let svg = to_svg(&g, &layout, &leiden.membership, &factions, 500.0);
    println!(
        "\nSVG picture: {} bytes, {} edges, {} vertices",
        svg.len(),
        svg.matches("<line").count(),
        svg.matches("<circle").count()
    );
    if let Some(path) = std::env::args().nth(1) {
        std::fs::write(&path, &svg).map_err(|e| Error::invalid(e.to_string()))?;
        println!("written to {path}");
    }
    Ok(())
}

/// Renders the layout: fill color = community, outline = real faction.
fn to_svg(g: &Graph, layout: &Matrix, membership: &[i64], factions: &[i64], size: f64) -> String {
    const PALETTE: [&str; 6] = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    ];
    let margin = 25.0;
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
        format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{size}\" height=\"{size}\">\n");
    for (u, v) in g.edge_list() {
        let ((ax, ay), (bx, by)) = (pos(u), pos(v));
        svg.push_str(&format!(
            "  <line x1=\"{ax:.1}\" y1=\"{ay:.1}\" x2=\"{bx:.1}\" y2=\"{by:.1}\" stroke=\"#aaa\"/>\n"
        ));
    }
    for v in g.vertices() {
        let (x, y) = pos(v);
        let fill = PALETTE[membership[v as usize] as usize % PALETTE.len()];
        let stroke = if factions[v as usize] == 1 {
            "#000"
        } else {
            "#fff"
        };
        svg.push_str(&format!(
            "  <circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"8\" fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"2\"><title>{v}</title></circle>\n"
        ));
    }
    svg.push_str("</svg>\n");
    svg
}

//! Draws Zachary's karate club as an SVG picture, printed on standard output.
//!
//! ```sh
//! cargo run --example layout_svg > karate.svg
//! ```
//!
//! The graph comes from [`Graph::famous`]. The layout is computed with
//! Kamada-Kawai (deterministic, from a circle), refined with a
//! Fruchterman-Reingold run using a private seeded generator, aligned with the
//! axes and scaled to the canvas. Vertices are colored by the communities
//! found by the multilevel (Louvain) algorithm, and the two faction leaders
//! (vertices 0 and 33) are drawn larger.

use igraph::layout::{FruchtermanReingoldOptions, KamadaKawaiOptions};
use igraph::prelude::*;

const PALETTE: [&str; 6] = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
];

fn main() -> igraph::Result<()> {
    let g = Graph::famous("Zachary")?;

    let kk = g.layout_kamada_kawai(&KamadaKawaiOptions::default())?;
    let opts = FruchtermanReingoldOptions::default().with_initial(&kk);
    let mut layout =
        Rng::new(RngType::Pcg64, 2024)?.scoped(|| g.layout_fruchterman_reingold(&opts))?;
    g.layout_align(&mut layout)?;

    // Communities for the colors (Louvain is randomized too: seed it).
    rng::seed(2024)?;
    let membership = g.community_multilevel(None, 1.0)?.membership;

    // Fit into a 600 x 600 canvas with a 30 px margin.
    let (size, margin) = (600.0, 30.0);
    let bounds = |k: usize| {
        layout
            .column(k)
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
                (lo.min(x), hi.max(x))
            })
    };
    let ((x0, x1), (y0, y1)) = (bounds(0), bounds(1));
    let scale = (size - 2.0 * margin) / (x1 - x0).max(y1 - y0);
    let pos = |v: i64| {
        let v = v as usize;
        (
            margin + (layout[(v, 0)] - x0) * scale,
            margin + (layout[(v, 1)] - y0) * scale,
        )
    };

    println!(r#"<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">"#);
    for (u, v) in g.edge_list() {
        let ((ax, ay), (bx, by)) = (pos(u), pos(v));
        println!(
            r##"  <line x1="{ax:.1}" y1="{ay:.1}" x2="{bx:.1}" y2="{by:.1}" stroke="#999"/>"##
        );
    }
    for v in g.vertices() {
        let (x, y) = pos(v);
        let r = if v == 0 || v == 33 { 10 } else { 6 };
        let fill = PALETTE[membership[v as usize] as usize % PALETTE.len()];
        println!(
            r#"  <circle cx="{x:.1}" cy="{y:.1}" r="{r}" fill="{fill}"><title>{v}</title></circle>"#
        );
    }
    println!("</svg>");
    Ok(())
}

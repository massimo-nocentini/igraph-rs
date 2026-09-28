//! Use case: the small-world phenomenon (Watts and Strogatz, 1998).
//!
//! ```sh
//! cargo run --example uc_small_world
//! ```
//!
//! Start from a ring of 1000 people, each knowing their 10 nearest
//! neighbours, and rewire every friendship at random with probability `p`.
//! The table shows the clustering coefficient `C(p)` and the average path
//! length `L(p)`, both relative to the regular ring (`p = 0`): a tiny amount
//! of rewiring makes paths short while the clustering stays high, the
//! "small world" of the famous Figure 2 of the paper.

use igraph::prelude::*;

fn main() -> igraph::Result<()> {
    let (n, nei, runs) = (1000, 5, 5);
    rng::seed(1998)?;
    let measure = |p: f64| -> igraph::Result<(f64, f64)> {
        let (mut c, mut l) = (0.0, 0.0);
        for _ in 0..runs {
            let g = Graph::watts_strogatz_game(1, n, nei, p, EdgeTypeSw::Simple)?;
            c += g.transitivity_undirected(TransitivityMode::Zero)?;
            l += g.average_path_length(None, false, true)?;
        }
        Ok((c / runs as f64, l / runs as f64))
    };
    let (c0, l0) = measure(0.0)?;
    println!("Ring lattice, n = {n}, k = {}:", 2 * nei);
    println!(
        "  C(0) = {c0:.4} (theory 3(k-2)/(4(k-1)) = {:.4})",
        3.0 * 8.0 / 36.0
    );
    println!("  L(0) = {l0:.2}");
    println!(
        "\n{:>8}  {:>9}  {:>9}  {:>8}  {:>8}",
        "p", "C(p)/C(0)", "L(p)/L(0)", "C(p)", "L(p)"
    );
    for k in 0..=12 {
        let p = 10f64.powf(-4.0 + k as f64 / 3.0).min(1.0);
        let (c, l) = measure(p)?;
        let bar = "#".repeat((40.0 * l / l0).round() as usize);
        println!(
            "{p:>8.4}  {:>9.3}  {:>9.3}  {c:>8.4}  {l:>8.2}  {bar}",
            c / c0,
            l / l0
        );
    }
    println!(
        "\nRandom graph estimates: C ~ k/n = {:.4}, L ~ ln n / ln k = {:.2}",
        10.0 / n as f64,
        (n as f64).ln() / 10f64.ln()
    );
    Ok(())
}

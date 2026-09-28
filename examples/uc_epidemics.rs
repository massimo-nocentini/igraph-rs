//! Use case: how network structure shapes an epidemic.
//!
//! ```sh
//! cargo run --example uc_epidemics
//! ```
//!
//! Two populations of 2000 people with exactly the same number of contacts:
//! a scale-free network (Barabási–Albert, a few hubs with many contacts)
//! and a homogeneous random network (Erdős–Rényi). We run many stochastic
//! SIR epidemics (susceptible → infected → recovered) with increasing
//! transmissibility and report the average fraction of people ever
//! infected. Hubs make the scale-free network vulnerable to much milder
//! diseases: its epidemic threshold `<k> / (<k²> - <k>)` is lower.

use igraph::games::BarabasiOptions;
use igraph::prelude::*;

fn moments(g: &Graph) -> igraph::Result<(f64, f64, i64)> {
    let d = g.degree(VertexSelector::All, NeighborMode::All, Loops::Twice)?;
    let n = d.len() as f64;
    let k1 = d.iter().sum::<i64>() as f64 / n;
    let k2 = d.iter().map(|&x| (x * x) as f64).sum::<f64>() / n;
    Ok((k1, k2, d.iter().copied().max().unwrap_or(0)))
}

fn main() -> igraph::Result<()> {
    let n = 2000;
    rng::seed(3)?;
    let ba = Graph::barabasi_game(n, &BarabasiOptions::default().with_m(2))?;
    let er = Graph::erdos_renyi_game_gnm(n, ba.ecount(), false, EdgeTypeSw::Simple, false)?;
    println!("{n} people, {} contacts in both networks\n", ba.ecount());
    println!(
        "{:<14} {:>6} {:>8} {:>8} {:>10}",
        "network", "<k>", "<k^2>", "max k", "threshold"
    );
    for (name, g) in [("scale-free", &ba), ("random", &er)] {
        let (k1, k2, kmax) = moments(g)?;
        println!(
            "{name:<14} {k1:>6.2} {k2:>8.2} {kmax:>8} {:>10.3}",
            k1 / (k2 - k1)
        );
    }

    let (gamma, sims) = (1.0, 200);
    println!("\nMean final epidemic size ({sims} runs each, recovery rate {gamma}):");
    println!(
        "{:>6} {:>6}  {:>11} {:>8}",
        "beta", "T", "scale-free", "random"
    );
    for beta in [0.1, 0.2, 0.25, 0.3, 0.4, 0.6, 1.0] {
        let t = beta / (beta + gamma);
        let mut sizes = Vec::new();
        for g in [&ba, &er] {
            let runs = g.sir(beta, gamma, sims)?;
            let mean = runs
                .iter()
                .map(|r| *r.recovered.last().unwrap_or(&0) as f64)
                .sum::<f64>()
                / (sims * n) as f64;
            sizes.push(mean);
        }
        println!(
            "{beta:>6.2} {t:>6.3}  {:>10.1}% {:>7.1}%",
            100.0 * sizes[0],
            100.0 * sizes[1]
        );
    }

    // Most outbreaks die out early; follow the largest of 50 of them.
    let run = ba
        .sir(0.4, gamma, 50)?
        .into_iter()
        .max_by_key(|r| *r.recovered.last().unwrap_or(&0))
        .ok_or_else(|| Error::invalid("no simulation"))?;
    let peak = (0..run.times.len())
        .max_by_key(|&i| run.infected[i])
        .unwrap_or(0);
    println!(
        "\nLargest of 50 outbreaks on the scale-free network (beta = 0.4): peak of {} infected at t = {:.2}, \
         {} recovered in the end (t = {:.2})",
        run.infected[peak],
        run.times[peak],
        run.recovered.last().unwrap_or(&0),
        run.times.last().unwrap_or(&0.0)
    );
    Ok(())
}

//! Use case: predicting the split of Zachary's karate club.
//!
//! Zachary observed a university karate club splitting in two after a
//! conflict between the instructor (vertex 0) and the administrator
//! (vertex 33). This example asks whether community detection, looking only
//! at who interacted with whom, predicts the two factions.
//!
//! Run with `cargo run --example community_karate`.

use igraph::community::{LeidenObjective, LeidenOptions, compare_communities, split_join_distance};
use igraph::prelude::*;

fn main() -> igraph::Result<()> {
    let g = Graph::famous("Zachary")?;
    // Who joined the administrator's club (0-based ids) after the split.
    let officer = [
        8, 9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    ];
    let factions: Vec<i64> = (0..34).map(|v| i64::from(officer.contains(&v))).collect();
    println!(
        "modularity of the real split: {:.4}",
        g.modularity(&factions, None, 1.0, false)?
    );

    // Leiden, seeded for a reproducible run, finds the modularity optimum.
    rng::seed(42)?;
    let leiden = g.community_leiden_simple(
        None,
        LeidenObjective::Modularity,
        &LeidenOptions::default().with_iterations(None),
    )?;
    println!(
        "Leiden: {} communities, modularity {:.4}",
        leiden.nb_clusters, leiden.quality
    );
    for (c, members) in leiden.communities().iter().enumerate() {
        println!("  community {c}: {members:?}");
    }
    // A split-join distance of 0 from Leiden to the factions means that each
    // Leiden community lies within a single faction.
    let (d12, d21) = split_join_distance(&leiden.membership, &factions)?;
    println!("split-join distances to the real split: {d12} / {d21}");

    // The community graph: one vertex per community; the edges inside a
    // community become self-loops, the others multi-edges between communities.
    let mut quotient = g.clone();
    quotient.contract_vertices(&leiden.membership)?;
    let mut internal = vec![0; leiden.nb_clusters];
    for (a, b) in quotient.edge_list() {
        if a == b {
            internal[a as usize] += 1;
        }
    }
    println!("edges inside each community: {internal:?}");

    // Newman's spectral method, stopped after its first split, recovers the
    // factions exactly.
    let spectral = g.community_leading_eigenvector(None, Some(1), None)?;
    let nmi = compare_communities(&spectral.membership, &factions, CommunityComparison::Nmi)?;
    println!("leading eigenvector (one split) vs real split: NMI = {nmi:.3}");
    Ok(())
}
